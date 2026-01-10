import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from lerobot.datasets.lerobot_dataset import LeRobotDataset
import zarr
import numpy as np
from tqdm import tqdm
import shutil
import os

# --- CONFIG ---
DATASET_REPO = "lerobot/pusht"
BATCH_SIZE = 64
OUTPUT_DIR = "./data/precomputed_zarr"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def main():
    print(f"🚀 Re-computing Dataset with Proprioception on {DEVICE}...")
    
    # 1. Load DINOv2
    dino = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14').to(DEVICE)
    dino.eval()
    
    # 2. Transform
    transform = transforms.Compose([
        transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 3. Load Dataset
    dataset = LeRobotDataset(DATASET_REPO, root="./data")
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, num_workers=0)
    
    # 4. Reset Zarr
    if os.path.exists(OUTPUT_DIR): shutil.rmtree(OUTPUT_DIR)
    store = zarr.DirectoryStore(OUTPUT_DIR)
    root = zarr.group(store=store, overwrite=True)
    
    # Determine State Dimension
    sample_item = dataset[0]
    state_dim = sample_item['observation.state'].shape[0]
    print(f"ℹ️ Found Proprioception State Dim: {state_dim}")

    # Create Arrays
    z_embed = root.zeros('embeddings', shape=(0, 256, 384), chunks=(100, 256, 384), dtype='float32')
    z_state = root.zeros('states', shape=(0, state_dim), chunks=(100, state_dim), dtype='float32')
    z_action = root.zeros('actions', shape=(0, 2), chunks=(100, 2), dtype='float32')
    z_episode = root.zeros('episode_index', shape=(0,), chunks=(100,), dtype='int32')
    
    # Stats
    all_actions = []
    all_states = []

    print("⚙️ Processing...")
    for batch in tqdm(loader):
        imgs = batch['observation.image'].to(DEVICE)
        states = batch['observation.state']
        actions = batch['action']
        ep_idx = batch['episode_index']
        
        with torch.no_grad():
            imgs_resized = transform(imgs)
            features = dino.forward_features(imgs_resized)
            patches = features['x_norm_patchtokens'].cpu().numpy()
            
            z_embed.append(patches)
            z_state.append(states.numpy())
            z_action.append(actions.numpy())
            z_episode.append(ep_idx.numpy())
            
            all_actions.append(actions.numpy())
            all_states.append(states.numpy())

    # Save Statistics (Crucial for normalizing)
    all_actions = np.concatenate(all_actions, axis=0)
    all_states = np.concatenate(all_states, axis=0)
    
    root.attrs['action_mean'] = np.mean(all_actions, axis=0).tolist()
    root.attrs['action_std'] = np.std(all_actions, axis=0).tolist()
    root.attrs['state_mean'] = np.mean(all_states, axis=0).tolist()
    root.attrs['state_std'] = np.std(all_states, axis=0).tolist()
    
    print("✅ Done! Stats saved.")

if __name__ == "__main__":
    main()