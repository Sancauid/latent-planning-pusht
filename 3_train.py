import torch
import torch.nn as nn
import torch.optim as optim
import zarr
import numpy as np
import time
import math

# --- CONFIG ---
ZARR_PATH = "./data/precomputed_zarr"
BATCH_SIZE = 256
LR = 5e-4
EPOCHS = 100
DEVICE = "cuda"
ROLLOUT_STEPS = 6  

# --- ADALN TRANSFORMER BLOCK ---
def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

class AdaLNBlock(nn.Module):
    """
    A Transformer Block where LayerNorm is 'modulated' by the Action.
    This forces the Action to influence every single layer of the brain.
    """
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden_dim),
            nn.GELU(),
            nn.Linear(mlp_hidden_dim, hidden_size)
        )
        
        # AdaLN: Regress scale/shift from action condition
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        # x: (B, Seq, Dim) - Image/State tokens
        # c: (B, Dim)      - Action embedding
        
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        
        # 1. Attention Block (Modulated)
        x_norm = modulate(self.norm1(x), shift_msa, scale_msa)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + gate_msa.unsqueeze(1) * attn_out
        
        # 2. MLP Block (Modulated)
        x_norm = modulate(self.norm2(x), shift_mlp, scale_mlp)
        mlp_out = self.mlp(x_norm)
        x = x + gate_mlp.unsqueeze(1) * mlp_out
        
        return x

class UltimateJepa(nn.Module):
    def __init__(self, state_dim, embed_dim=384, num_heads=6, depth=6):
        super().__init__()
        
        # Encoders
        self.action_enc = nn.Sequential(nn.Linear(2, embed_dim), nn.GELU(), nn.Linear(embed_dim, embed_dim))
        self.state_enc = nn.Sequential(nn.Linear(state_dim, embed_dim), nn.GELU(), nn.Linear(embed_dim, embed_dim))
        
        # Positional Embed: 256 patches + 1 state = 257 (Action is now injected via AdaLN)
        self.pos_embed = nn.Parameter(torch.randn(1, 257, embed_dim) * 0.02)
        
        # Stack of AdaLN Blocks
        self.blocks = nn.ModuleList([
            AdaLNBlock(embed_dim, num_heads) for _ in range(depth)
        ])
        
        # Final Norm (AdaLN)
        self.final_layer = nn.Sequential(nn.SiLU(), nn.Linear(embed_dim, 2 * embed_dim, bias=True))
        self.norm_final = nn.LayerNorm(embed_dim, elementwise_affine=False, eps=1e-6)
        
        # Heads
        self.img_head = nn.Linear(embed_dim, embed_dim)
        self.state_head = nn.Sequential(nn.Linear(embed_dim, embed_dim), nn.GELU(), nn.Linear(embed_dim, state_dim))

    def forward(self, z, s, a):
        # Embed Inputs
        s_emb = self.state_enc(s).unsqueeze(1) # (B, 1, D)
        a_emb = self.action_enc(a)             # (B, D) -> Used for Conditioning
        
        # Sequence: [State, ImagePatches]
        # Note: Action is NOT in the sequence anymore. It controls the layers.
        seq = torch.cat([s_emb, z], dim=1) + self.pos_embed
        
        # Run AdaLN Blocks
        for block in self.blocks:
            seq = block(seq, a_emb)
            
        # Final Norm
        shift, scale = self.final_layer(a_emb).chunk(2, dim=1)
        seq = modulate(self.norm_final(seq), shift, scale)
        
        # Decode
        # Image is tokens 1 onwards
        z_pred = self.img_head(seq[:, 1:, :])
        # State is token 0
        s_pred = self.state_head(seq[:, 0, :])
        
        return z_pred, s_pred

# --- TRAINING LOOP ---
def load_stats(root):
    return {
        'a_mean': torch.tensor(root.attrs['action_mean'], device=DEVICE),
        'a_std': torch.tensor(root.attrs['action_std'], device=DEVICE),
        's_mean': torch.tensor(root.attrs['state_mean'], device=DEVICE),
        's_std': torch.tensor(root.attrs['state_std'], device=DEVICE),
    }

def train():
    print("⏳ Loading Data...")
    root = zarr.open(ZARR_PATH, mode='r')
    stats = load_stats(root)
    
    # Load all to GPU
    emb = torch.from_numpy(root['embeddings'][:]).to(DEVICE, non_blocking=True)
    state = torch.from_numpy(root['states'][:]).to(DEVICE, non_blocking=True)
    act = torch.from_numpy(root['actions'][:]).to(DEVICE, non_blocking=True)
    ep_idx = torch.from_numpy(root['episode_index'][:]).to(DEVICE, non_blocking=True)
    
    N = emb.shape[0]
    state_dim = state.shape[1]
    
    # Setup Ultimate Model
    model = UltimateJepa(state_dim=state_dim).to(DEVICE)
    model = torch.compile(model)
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    criterion = nn.MSELoss()
    
    print(f"🚀 Training ULTIMATE JEPA (AdaLN + Proprio + Rollout={ROLLOUT_STEPS})...")
    
    for epoch in range(EPOCHS):
        start = time.time()
        indices = torch.randint(0, N - ROLLOUT_STEPS - 1, (N // BATCH_SIZE * BATCH_SIZE,), device=DEVICE)
        
        total_loss = 0
        batches = 0
        
        model.train()
        for i in range(0, len(indices), BATCH_SIZE):
            idx = indices[i:i+BATCH_SIZE]
            
            curr_z = emb[idx]
            curr_s = (state[idx] - stats['s_mean']) / stats['s_std']
            
            batch_loss = 0
            
            with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
                for k in range(ROLLOUT_STEPS):
                    curr_a = (act[idx + k] - stats['a_mean']) / stats['a_std']
                    
                    # AdaLN Forward
                    pred_z, pred_s = model(curr_z, curr_s, curr_a)
                    
                    # Targets
                    target_z = emb[idx + k + 1]
                    target_s = (state[idx + k + 1] - stats['s_mean']) / stats['s_std']
                    
                    l_img = criterion(pred_z, target_z)
                    l_state = criterion(pred_s, target_s)
                    
                    batch_loss += l_img + l_state
                    
                    curr_z = pred_z
                    curr_s = pred_s

            optimizer.zero_grad(set_to_none=True)
            batch_loss.backward()
            optimizer.step()
            
            total_loss += batch_loss.item() / ROLLOUT_STEPS
            batches += 1
            
        print(f"Epoch {epoch+1:03d} | Loss: {total_loss/batches:.4f} | Time: {time.time()-start:.2f}s")
        
        if (epoch+1) % 5 == 0:
            torch.save(model.state_dict(), f"checkpoints/sota_epoch_{epoch+1}.pth")

if __name__ == "__main__":
    train()