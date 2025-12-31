
import torch
import torch.nn as nn
from einops import rearrange
from pose.module import DPTHead, create_uv_grid, position_grid_to_embed

class OptimizedDPTHead(DPTHead):
    """
    Optimized version of DPTHead that precomputes positional embeddings
    to avoid redundant calculations inside the chunk/scale loops.
    """
    def forward(self, token_list, image_size, frames_chunk_size=2):
        B, N, C = token_list[0].shape
        H, W = image_size
        patch_h, patch_w = H // self.patch_size, W // self.patch_size

        # Precompute positional embeddings once per forward pass
        pos_embeds = {}
        if self.pose_emb:
            # Identify required channel dimensions from projection layers
            needed_dims = set()
            for proj in self.projects:
                if isinstance(proj, nn.Conv2d):
                    needed_dims.add(proj.out_channels)

            # Create embeddings for each unique channel dimension
            dtype = token_list[0].dtype
            device = token_list[0].device

            # create_uv_grid is cached internally by GridCache
            uv_grid = create_uv_grid(
                width=patch_w,
                height=patch_h,
                aspect_ratio=W / H,
                dtype=dtype,
                device=device
            )

            for d in needed_dims:
                # position_grid_to_embed is the operation we want to hoist out of loops
                emb = position_grid_to_embed(uv_grid, d)
                # Permute to (C, H, W) for broadcasting with (B, C, H, W)
                emb = emb.permute(2, 0, 1)
                pos_embeds[d] = emb

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
                    # Apply precomputed embedding
                    t = t + pos_embeds[t.shape[1]]

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
