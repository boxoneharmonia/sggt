import torch
import torch.nn as nn
from einops import rearrange
from .module import DPTHead, create_uv_grid, position_grid_to_embed

class OptimizedDPTHead(DPTHead):
    """
    Optimized version of DPTHead that pre-computes positional embeddings
    to avoid redundant calculations inside the chunk loop.
    """
    def forward(self, token_list, image_size, frames_chunk_size=2):
        B, N, C = token_list[0].shape
        H, W = image_size
        patch_h, patch_w = H // self.patch_size, W // self.patch_size
        outputs = []
        chunk_size = frames_chunk_size

        # Optimization: Pre-compute positional embeddings
        # We need to compute embeddings for the output channel dimensions of self.projects
        # self.projects[i] outputs shape (B, C_out, patch_h, patch_w)

        cached_pos_embeds = {}

        if self.pose_emb:
            # Identify unique output channels required
            # The channels are at index 1 of the output
            # self.projects is a ModuleList of Conv2d.
            required_channels = set()
            for proj in self.projects:
                 required_channels.add(proj.out_channels)

            # Use properties from the first token for device/dtype
            # Assuming all tokens are on same device/dtype
            dtype = token_list[0].dtype
            device = token_list[0].device

            # create_uv_grid uses caching internally so it's cheap to call
            uv_grid = create_uv_grid(
                width=patch_w,
                height=patch_h,
                aspect_ratio=W / H,
                dtype=dtype,
                device=device
            )

            # Pre-compute embedding for each required channel dim
            for dim in required_channels:
                # position_grid_to_embed computes sin/cos embeddings
                pos_embed = position_grid_to_embed(uv_grid, dim)
                pos_embed = pos_embed.permute(2, 0, 1) # (C, H, W)
                cached_pos_embeds[dim] = pos_embed

        for start_idx in range(0, B, chunk_size):
            end_idx = min(start_idx + chunk_size, B)
            pyramid_features = []
            for i, token in enumerate(token_list):
                t = token[start_idx:end_idx]
                t = self.norm[i](t)
                t = rearrange(t, 'b (h w) c -> b c h w', h=patch_h, w=patch_w)
                t = self.projects[i](t)

                if self.pose_emb:
                    # Use cached embedding based on channel dimension
                    t = t + cached_pos_embeds[t.shape[1]]

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
