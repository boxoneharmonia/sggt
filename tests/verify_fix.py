import sys
import os
import torch
import torch.nn as nn

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pose.net import Decoder, PoseLoss
from config import Config

def verify_fix():
    print("Initializing Config with Fix...")
    config = Config()

    # APPLY FIX:
    # Original was 8. Dataset has 8 corners. We need 2 channels per corner.
    # So maps should be 8 corners * 2 = 16.
    config.maps = 16

    config.use_cuda = False
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print(f"Config.maps set to: {config.maps}")

    print("Initializing Models...")
    decoder = Decoder(config).to(device)
    criterion = PoseLoss(config).to(device)

    print("Generating Dummy Data...")
    B = 1
    S = config.seq_len
    H, W = config.original_size
    C = 3

    num_features = len(config.feature_idx)
    cam_token_list = [torch.randn(B, S, config.embed_dim).to(device) for _ in range(num_features)]
    features_list = [torch.randn(B, config.num_patches, config.embed_dim).to(device) for _ in range(num_features)]
    image_size = (H, W)

    # Forward Pass
    print("Running Forward Pass...")
    try:
        pose, mask, pvmap, pmap = decoder(cam_token_list, features_list, image_size)
        print("Forward Pass Successful.")
        print(f"Output Shapes:")
        if pose is not None: print(f"  Pose: {pose.shape}")
        if mask is not None: print(f"  Mask: {mask.shape}")
        if pvmap is not None: print(f"  PVMap: {pvmap.shape}")

    except Exception as e:
        print(f"Forward Pass Failed: {e}")
        return

    print(f"\nPVMap channels: {pvmap.shape[1]}")
    predicted_corners = pvmap.shape[1] // 2
    print(f"Predicted number of corners based on PVMap: {predicted_corners}")

    print("Preparing Ground Truth Data...")
    num_gt_corners = 8
    data_dict = {
        'R_cam': torch.eye(3).unsqueeze(0).repeat(B, 1, 1).to(device),
        't_cam': torch.zeros(B, 3, 1).to(device), # Corrected shape (B, 3, 1)
        'mask': torch.zeros(B, 1, H, W).to(device),
        'pts3d': torch.randn(B, num_gt_corners, 3).to(device), # 8 corners
        'cam_K': torch.eye(3).unsqueeze(0).repeat(B, 1, 1).to(device),
        'pcloud': torch.randn(B, 3, H, W).to(device),
        'pconf': torch.zeros(B, 1, H, W).to(device),
        'images': torch.zeros(B, S, C, H, W).to(device)
    }

    out_dict = {
        'pose': pose,
        'mask': mask,
        'pvmap': pvmap,
        'pmap': pmap
    }

    # Loss Calculation
    print("\nRunning Loss Calculation...")
    try:
        loss, loss_dict = criterion(out_dict, data_dict)
        print("Loss Calculation Successful.")
        print(f"Total Loss: {loss.item()}")
        print("\nSUCCESS: Network policy works with fixed config.")
    except RuntimeError as e:
        print("\n!!! RuntimeError Detected !!!")
        print(f"Error message: {e}")
        print("\nFix failed to resolve the issue.")

if __name__ == "__main__":
    verify_fix()
