import sys
import os
import torch
import torch.nn as nn

# Add the repository root to the python path so we can import modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pose.net import Decoder, PoseLoss
from config import Config

class MockEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed_dim = config.embed_dim
        self.num_patches = config.num_patches

    def forward(self, x):
        # x is (B, seq_len, 3, H, W)
        b, s = x.shape[0], x.shape[1]
        image_size = x.shape[-2:]

        # Simulate cam_token_list: List of tensors (B, s, c)
        # In the original code, it returns a list of tokens from different layers.
        # Let's say we have 4 layers of interest as per feature_idx=[1,3,4,5]
        num_features = len(self.config.feature_idx)
        cam_token_list = [torch.randn(b, s, self.embed_dim) for _ in range(num_features)]

        # Simulate features_list: List of tensors (B, num_patches, c)
        # Note: In Encoder.forward:
        # features = x[:,0,0:n] # (b n c) -> It seems to take only the first frame's features?
        # Let's check the code in pose/net.py
        # cam_token = x[:,:,n] # (b s c)
        # features = x[:,0,0:n] # (b n c)
        # So features_list has shape (B, N, C) corresponding to the first frame (index 0).

        # We need to calculate N (num_patches).
        # num_patches in config is 576.
        n = self.num_patches
        features_list = [torch.randn(b, n, self.embed_dim) for _ in range(num_features)]

        return cam_token_list, features_list, image_size

def test_network_policy():
    print("Initializing Config...")
    config = Config()

    # Force cpu for testing to avoid cuda errors if not available, though environment likely has cuda
    config.use_cuda = False
    # But checking if cuda is available is better
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    print("Initializing Models...")
    # encoder = MockEncoder(config).to(device) # We don't strictly need the encoder instance, just its output
    decoder = Decoder(config).to(device)
    criterion = PoseLoss(config).to(device)

    print("Generating Dummy Data...")
    B = 1
    S = config.seq_len
    H, W = config.original_size
    C = 3

    # Inputs
    # cam_token_list, features_list, image_size
    num_features = len(config.feature_idx) # 4
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
        if pmap is not None: print(f"  PMap: {pmap.shape}")

    except Exception as e:
        print(f"Forward Pass Failed: {e}")
        return

    # Check assumption about map channels
    # config.maps = 8.
    # DPTHead output = 8 + 1 (conf) = 9.
    # mask = maps[:, -1:] -> (B, 1, H, W)
    # pvmap = maps[:, :-1] -> (B, 8, H, W)
    # pvmap corresponds to 2D vectors for each corner.
    # If there are 8 channels, that means 4 corners (4 * 2 = 8).

    print(f"\nPVMap channels: {pvmap.shape[1]}")
    predicted_corners = pvmap.shape[1] // 2
    print(f"Predicted number of corners based on PVMap: {predicted_corners}")

    # Prepare Data Dict for Loss
    # We need to mock the ground truth.
    # Dataset.py says:
    # pts3d_corner = pts_cam_all[:8] -> 8 corners.
    # So pts3d ground truth has shape (B, 8, 3).

    print("Preparing Ground Truth Data...")
    num_gt_corners = 8
    data_dict = {
        'R_cam': torch.eye(3).unsqueeze(0).repeat(B, 1, 1).to(device), # (B, 3, 3)
        't_cam': torch.zeros(B, 3).to(device), # (B, 3)
        'mask': torch.zeros(B, 1, H, W).to(device), # (B, 1, H, W)
        'pts3d': torch.randn(B, num_gt_corners, 3).to(device), # (B, 8, 3)
        'cam_K': torch.eye(3).unsqueeze(0).repeat(B, 1, 1).to(device), # (B, 3, 3)
        'pcloud': torch.randn(B, 3, H, W).to(device),
        'pconf': torch.zeros(B, 1, H, W).to(device),
        'images': torch.zeros(B, S, C, H, W).to(device) # Needed for getting H_orig, W_orig
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
    except RuntimeError as e:
        print("\n!!! RuntimeError Detected !!!")
        print(f"Error message: {e}")
        print("\nAnalysis:")
        if "shape" in str(e) or "broadcast" in str(e) or "match" in str(e):
            print(f"Likely due to mismatch between Predicted Corners ({predicted_corners}) and GT Corners ({num_gt_corners}).")
            print("config.maps is set to 8, which implies 4 corners (2 channels per corner).")
            print("The dataset provides 8 corners.")
            print("The voting_loss function tries to match them directly, causing a failure.")

if __name__ == "__main__":
    test_network_policy()
