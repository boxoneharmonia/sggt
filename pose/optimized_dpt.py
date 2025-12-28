import torch
import torch.nn as nn
from einops import rearrange
from .module import FeatureFusionBlock, create_uv_grid, position_grid_to_embed

class OptimizedDPTHead(nn.Module):
    def __init__(self, inp, oup, hidden_ratio=2.0, features=256, patch_size=16, pos_emb=True, activation:nn.Module=nn.Identity(), use_conf=False):
        super().__init__()
        self.patch_size = patch_size
        self.pose_emb = pos_emb
        self.norm = nn.ModuleList([nn.LayerNorm(inp) for _ in range(4)])
        hidden_dim = int(hidden_ratio*features)
        self.projects = nn.ModuleList(
            [
                nn.Conv2d(in_channels=inp, out_channels=hidden_dim, kernel_size=1, stride=1, padding=0),
                nn.Conv2d(in_channels=inp, out_channels=hidden_dim, kernel_size=1, stride=1, padding=0),
                nn.Conv2d(in_channels=inp, out_channels=features, kernel_size=1, stride=1, padding=0),
                nn.Conv2d(in_channels=inp, out_channels=hidden_dim, kernel_size=1, stride=1, padding=0),
             ]
        )
        self.resize = nn.ModuleList(
            [
                nn.ConvTranspose2d(in_channels=hidden_dim, out_channels=features, kernel_size=4, stride=4, padding=0),
                nn.ConvTranspose2d(in_channels=hidden_dim, out_channels=features, kernel_size=2, stride=2, padding=0),
                nn.Identity(),
                nn.Conv2d(in_channels=hidden_dim, out_channels=features, kernel_size=3, stride=2, padding=1),
            ]
        )
        self.refinement = nn.ModuleList(
            [
                FeatureFusionBlock(features, hidden_ratio, has_residual=False),
                FeatureFusionBlock(features, hidden_ratio, has_residual=True),
                FeatureFusionBlock(features, hidden_ratio, has_residual=True),
                FeatureFusionBlock(features, hidden_ratio, has_residual=True),
            ]
        )
        self.outproj = nn.Sequential(
            nn.Conv2d(in_channels=features, out_channels=oup, kernel_size=1, stride=1, padding=0),
            activation
        )
        self.outconf = nn.Conv2d(in_channels=features, out_channels=1, kernel_size=1, stride=1, padding=0) if use_conf is True else None

        # Optimization: Pre-calculate needed channels for pos embedding
        self.needed_channels = set()
        if self.pose_emb:
            for i in range(len(self.projects)):
                self.needed_channels.add(self.projects[i].out_channels)

        # Cache for pos embeddings: key=(W, H, channels, device), value=tensor
        # We can implement a simple LRU or just a dict that clears if size changes?
        # For now, let's keep it simple: just cache within forward or use a dedicated cache if inputs are stable.
        # But per requirements, we should just optimize the forward loop redundancy.

    def _compute_pos_embed(self, H_feat: int, W_feat: int, channels: int, W: int, H: int, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        """
        Helper to compute positional embedding once.
        """
        pos_embed = create_uv_grid(
            width=W_feat,
            height=H_feat,
            aspect_ratio=W / H,
            dtype=dtype,
            device=device
        )

        pos_embed = position_grid_to_embed(pos_embed, channels)
        pos_embed = pos_embed.permute(2, 0, 1)
        return pos_embed

    def forward(self, token_list, image_size, frames_chunk_size=2):
        B, N, C = token_list[0].shape
        H, W = image_size
        patch_h, patch_w = H // self.patch_size, W // self.patch_size
        outputs = []
        chunk_size = frames_chunk_size

        # Optimize: Precompute positional embeddings if enabled
        # We do this once per forward pass
        pos_embeds = {}
        if self.pose_emb:
            sample_token = token_list[0]
            # needed_channels is small (size 2), iteration is fast
            for ch in self.needed_channels:
                pos_embeds[ch] = self._compute_pos_embed(
                    H_feat=patch_h,
                    W_feat=patch_w,
                    channels=ch,
                    W=W,
                    H=H,
                    dtype=sample_token.dtype,
                    device=sample_token.device
                )

        for start_idx in range(0, B, chunk_size):
            end_idx = min(start_idx + chunk_size, B)
            pyramid_features = []
            for i, token in enumerate(token_list):
                t = token[start_idx:end_idx]
                t = self.norm[i](t)
                t = rearrange(t, 'b (h w) c -> b c h w', h=patch_h, w=patch_w)
                t = self.projects[i](t)

                if self.pose_emb:
                    ch = t.shape[1]
                    t = t + pos_embeds[ch]

                t = self.resize[i](t)
                pyramid_features.append(t)

            out_features = self.refinement[0]([pyramid_features[3]])
            out_features = self.refinement[1]([out_features , pyramid_features[2]])
            out_features = self.refinement[2]([out_features , pyramid_features[1]])
            out_features = self.refinement[3]([out_features, pyramid_features[0]])
            out = self.outproj(out_features)
            if self.outconf is not None:
                conf = self.outconf(out_features)
                out = torch.cat([out, conf], dim=1)
            outputs.append(out)

        return torch.cat(outputs, dim=0)
