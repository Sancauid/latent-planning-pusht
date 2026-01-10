import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from torchvision import transforms
import zarr
import os

# --- CONFIG ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CHECKPOINT_PATH = "checkpoints/sota_epoch_40.pth" 
DATASET_REPO = "lerobot/pusht"
ZARR_PATH = "./data/precomputed_zarr"

# --- LOAD STATS ---
root = zarr.open(ZARR_PATH, mode='r')
STATS = {
    'a_mean': torch.tensor(root.attrs['action_mean'], device=DEVICE),
    'a_std': torch.tensor(root.attrs['action_std'], device=DEVICE),
    's_mean': torch.tensor(root.attrs['state_mean'], device=DEVICE),
    's_std': torch.tensor(root.attrs['state_std'], device=DEVICE),
}

# --- ARCHITECTURE ---
def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

class AdaLNBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(hidden_size, mlp_hidden_dim), nn.GELU(), nn.Linear(mlp_hidden_dim, hidden_size))
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 6 * hidden_size, bias=True))

    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        x_norm = modulate(self.norm1(x), shift_msa, scale_msa)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + gate_msa.unsqueeze(1) * attn_out
        x_norm = modulate(self.norm2(x), shift_mlp, scale_mlp)
        mlp_out = self.mlp(x_norm)
        x = x + gate_mlp.unsqueeze(1) * mlp_out
        return x

class UltimateJepa(nn.Module):
    def __init__(self, state_dim, embed_dim=384, num_heads=6, depth=6):
        super().__init__()
        self.action_enc = nn.Sequential(nn.Linear(2, embed_dim), nn.GELU(), nn.Linear(embed_dim, embed_dim))
        self.state_enc = nn.Sequential(nn.Linear(state_dim, embed_dim), nn.GELU(), nn.Linear(embed_dim, embed_dim))
        self.pos_embed = nn.Parameter(torch.randn(1, 257, embed_dim) * 0.02)
        self.blocks = nn.ModuleList([AdaLNBlock(embed_dim, num_heads) for _ in range(depth)])
        self.final_layer = nn.Sequential(nn.SiLU(), nn.Linear(embed_dim, 2 * embed_dim, bias=True))
        self.norm_final = nn.LayerNorm(embed_dim, elementwise_affine=False, eps=1e-6)
        self.img_head = nn.Linear(embed_dim, embed_dim)
        self.state_head = nn.Sequential(nn.Linear(embed_dim, embed_dim), nn.GELU(), nn.Linear(embed_dim, state_dim))

    def forward(self, z, s, a):
        s_emb = self.state_enc(s).unsqueeze(1)
        a_emb = self.action_enc(a)
        seq = torch.cat([s_emb, z], dim=1) + self.pos_embed
        for block in self.blocks:
            seq = block(seq, a_emb)
        shift, scale = self.final_layer(a_emb).chunk(2, dim=1)
        seq = modulate(self.norm_final(seq), shift, scale)
        return self.img_head(seq[:, 1:, :]), self.state_head(seq[:, 0, :])

# --- HELPERS ---
def load_components(state_dim):
    print("🦕 Loading DINOv2...")
    dino = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14').to(DEVICE)
    dino.eval()
    print("🧠 Loading Ultimate JEPA...")
    model = UltimateJepa(state_dim).to(DEVICE)
    
    if not os.path.exists(CHECKPOINT_PATH):
        alt_path = "checkpoints/sota_epoch_60.pth"
        if os.path.exists(alt_path):
            print(f"⚠️ {CHECKPOINT_PATH} not found, using {alt_path}")
            checkpoint = alt_path
        else:
            print(f"❌ Error: No checkpoint found at {CHECKPOINT_PATH}")
            exit()
    else:
        checkpoint = CHECKPOINT_PATH

    state_dict = torch.load(checkpoint, map_location=DEVICE)
    new_state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(new_state_dict)
    model.eval()
    return dino, model

def get_embedding(dino, image_tensor):
    transform = transforms.Compose([
        transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    img = transform(image_tensor).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        features = dino.forward_features(img)
        patches = features['x_norm_patchtokens']
    return patches

# --- CEM PLANNER ---
def plan_cem(model, z_start, s_start, z_goal):
    HORIZON = 10
    SAMPLES = 512
    ITERATIONS = 5
    ELITES = 50
    
    # Initialize Distribution (Mean=0, Std=1 in Normalized Space)
    opt_mean = torch.zeros(HORIZON, 2, device=DEVICE)
    opt_std = torch.ones(HORIZON, 2, device=DEVICE)
    
    # Normalize Start State
    s_start_norm = (s_start - STATS['s_mean']) / STATS['s_std']
    
    for i in range(ITERATIONS):
        # 1. Sample Actions (Normalized)
        noise = torch.randn(SAMPLES, HORIZON, 2, device=DEVICE)
        actions_norm = opt_mean + (noise * opt_std)
        
        # 2. Rollout
        # In this loop, we predict state t+1 from t. 
        curr_z = z_start.repeat(SAMPLES, 1, 1)
        curr_s = s_start_norm.repeat(SAMPLES, 1)
        
        with torch.no_grad():
            for t in range(HORIZON):
                # We feed NORMALIZED actions to this model (check train script loop)
                act = actions_norm[:, t, :]
                
                # Predict Next Step
                pred_z, pred_s = model(curr_z, curr_s, act)
                
                # Auto-regressive update
                curr_z = pred_z
                curr_s = pred_s
        
        # 3. Cost (Distance to Goal Image)
        # Note: We prioritize Image Goal matching here
        diff = curr_z - z_goal
        costs = diff.pow(2).mean(dim=(1, 2))
        
        # 4. Elites
        top_costs, top_idxs = torch.topk(costs, ELITES, largest=False)
        elites = actions_norm[top_idxs]
        
        # 5. Update
        new_mean = elites.mean(dim=0)
        new_std = elites.std(dim=0) + 0.1
        
        opt_mean = 0.2 * opt_mean + 0.8 * new_mean
        opt_std = 0.2 * opt_std + 0.8 * new_std
        
        print(f"   Refining... Cost: {top_costs[0].item():.5f}")
        
    # Return Real Actions
    return (opt_mean * STATS['a_std']) + STATS['a_mean']

def main():
    ds = LeRobotDataset(DATASET_REPO, root="./data")
    state_dim = ds[0]['observation.state'].shape[0]
    dino, model = load_components(state_dim)
    
    idx = 997
    print(f"🧪 Testing Ultimate JEPA on Frame {idx}...")
    
    img = ds[idx]['observation.image']
    state = ds[idx]['observation.state'].to(DEVICE)
    img_goal = ds[idx + 10]['observation.image']
    
    z_start = get_embedding(dino, img)
    z_goal = get_embedding(dino, img_goal)
    
    best_actions = plan_cem(model, z_start, state, z_goal).cpu().numpy()
    
    gt_actions = []
    for i in range(10): gt_actions.append(ds[idx+i]['action'])
    gt_actions = torch.stack(gt_actions).numpy()
    
    fig, ax = plt.subplots(figsize=(6,6))
    show_img = img.permute(1,2,0).numpy()
    show_img = (show_img - show_img.min()) / (show_img.max() - show_img.min())
    ax.imshow(show_img, extent=[0, 512, 512, 0], alpha=0.8)
    
    ax.plot(gt_actions[:,0], gt_actions[:,1], c='lime', linewidth=4, label='Human')
    ax.plot(best_actions[:,0], best_actions[:,1], c='red', linewidth=3, linestyle='--', label='Ultimate JEPA')
    ax.scatter(best_actions[:,0], best_actions[:,1], c='red', s=50, marker='x')
    
    ax.legend()
    ax.set_title("AdaLN + Proprioception Result")
    ax.set_xlim(0, 512); ax.set_ylim(512, 0)
    
    plt.savefig("ultimate_result.png")
    print("\n🎉 Done! Check 'ultimate_result.png'.")

if __name__ == "__main__":
    main()