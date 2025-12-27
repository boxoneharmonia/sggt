import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from .grid_cache import GridCache
from .module import FeatureFusionBlock

class DPTHeadOptimized(nn.Module):
    def __init__(self, inp, oup, hidden_ratio=2.0, features=256, patch_size=16, pos_emb=True, activation:nn.Module=nn.Identity(), use_conf=False):
        super().__init__()
        self.patch_size = patch_size
        self.pose_emb = pos_emb
        self.norm = nn.ModuleList([nn.LayerNorm(inp) for _ in range(4)])
        hidden_dim = int(hidden_ratio*features)
        self.hidden_dim = hidden_dim
        self.features_dim = features

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

        # Precompute omegas for sinusoidal embedding
        if self.pose_emb:
            self.register_buffer('omega_hidden', self._precompute_omega(hidden_dim))
            self.register_buffer('omega_features', self._precompute_omega(features))

        self.pos_embed_cache = {}

    def _precompute_omega(self, embed_dim, omega_0=100.0):
        # We need omega for make_sincos_pos_embed(embed_dim // 2)
        # That function expects 'dim' = embed_dim // 2
        # And creates omega of size dim // 2 = embed_dim // 4

        dim = embed_dim // 2
        omega = torch.arange(dim // 2, dtype=torch.float32)
        omega /= dim / 2.0
        omega = 1.0 / omega_0**omega
        return omega

    def _make_sincos_pos_embed_cached(self, pos: torch.Tensor, omega: torch.Tensor) -> torch.Tensor:
        # Optimized version using precomputed omega
        # pos: (M,)
        # omega: (D/2,)
        # Returns: (M, D)

        # Outer product: pos is (M), omega is (D/2) -> (M, D/2)
        out = torch.einsum("m,d->md", pos, omega)

        emb_sin = torch.sin(out)
        emb_cos = torch.cos(out)

        emb = torch.cat([emb_sin, emb_cos], dim=1)
        return emb.float() # Ensure float32

    def _position_grid_to_embed_cached(self, pos_grid: torch.Tensor, omega: torch.Tensor) -> torch.Tensor:
        # pos_grid: (H, W, 2)
        H, W, grid_dim = pos_grid.shape
        pos_flat = pos_grid.reshape(-1, grid_dim) # (H*W, 2)

        emb_x = self._make_sincos_pos_embed_cached(pos_flat[:, 0], omega) # (HW, D/2)
        emb_y = self._make_sincos_pos_embed_cached(pos_flat[:, 1], omega) # (HW, D/2)

        emb = torch.cat([emb_x, emb_y], dim=-1) # (HW, D)
        return emb.view(H, W, -1)

    def forward(self, token_list, image_size, frames_chunk_size=2):
        B, N, C = token_list[0].shape
        H, W = image_size
        patch_h, patch_w = H // self.patch_size, W // self.patch_size

        # Precompute positional embeddings if enabled
        pos_embed_hidden = None
        pos_embed_features = None

        if self.pose_emb:
            cache_key = (patch_h, patch_w, token_list[0].device)
            if cache_key in self.pos_embed_cache:
                pos_embed_hidden, pos_embed_features = self.pos_embed_cache[cache_key]
            else:
                # Create grid once
                # We assume dtype/device from first token
                dtype = token_list[0].dtype
                device = token_list[0].device

                uv_grid = GridCache.get_uv_grid(
                    width=patch_w,
                    height=patch_h,
                    aspect_ratio=W / H,
                    device=device,
                    dtype=dtype
                )

                # Compute embeddings
                pos_embed_hidden = self._position_grid_to_embed_cached(uv_grid, self.omega_hidden)
                pos_embed_hidden = pos_embed_hidden.permute(2, 0, 1) # (C, H, W)

                pos_embed_features = self._position_grid_to_embed_cached(uv_grid, self.omega_features)
                pos_embed_features = pos_embed_features.permute(2, 0, 1) # (C, H, W)

                if len(self.pos_embed_cache) > 4: # Simple LRU-ish / limit size
                     self.pos_embed_cache.clear()
                self.pos_embed_cache[cache_key] = (pos_embed_hidden, pos_embed_features)

        outputs = []
        chunk_size = frames_chunk_size
        for start_idx in range(0, B, chunk_size):
            end_idx = min(start_idx + chunk_size, B)
            pyramid_features = []
            for i, token in enumerate(token_list):
                t = token[start_idx:end_idx]
                t = self.norm[i](t)
                t = rearrange(t, 'b (h w) c -> b c h w', h=patch_h, w=patch_w)
                t = self.projects[i](t)

                if self.pose_emb:
                    # Select appropriate embedding
                    if i == 2:
                        t = t + pos_embed_features
                    else:
                        t = t + pos_embed_hidden

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
