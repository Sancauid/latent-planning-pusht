import torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset

def check_data():
    print("⬇️ Downloading/Loading Push-T Dataset...")
    
    # This automatically handles downloading and caching
    dataset = LeRobotDataset("lerobot/pusht", root="./data", video_backend="pyav")
    
    print(f"✅ Loaded {len(dataset)} trajectories.")
    
    # Get one item to check shapes
    item = dataset[0]
    
    # Push-T usually returns:clear
    # 'observation.image': (C, H, W)
    # 'action': (2,) -> (x, y) velocity
    
    print("\n🔍 Data Structure:")
    for k, v in item.items():
        if isinstance(v, torch.Tensor):
            print(f"  - {k}: {v.shape} | range: [{v.min():.2f}, {v.max():.2f}]")
        else:
            print(f"  - {k}: {type(v)}")

    # CRITICAL CHECK: DINOv2 requires 224x224 (usually)
    # If the image is 96x96, we need to know NOW.
    img_shape = item['observation.image'].shape
    print(f"\n🖼️ Image Shape: {img_shape}")
    
    if img_shape[1] != 224:
        print("⚠️ NOTE: We will need a Transform to resize to 224x224 for DINO.")

if __name__ == "__main__":
    check_data()