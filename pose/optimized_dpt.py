import torch
from einops import rearrange
from .module import DPTHead, create_uv_grid, position_grid_to_embed

class OptimizedDPTHead(DPTHead):
    """
    Optimized version of DPTHead that precomputes positional embeddings
    outside of the chunk and pyramid loops to reduce redundant computations.
    """
    def forward(self, token_list, image_size, frames_chunk_size=2):
        B, N, C = token_list[0].shape
        H, W = image_size
        patch_h, patch_w = H // self.patch_size, W // self.patch_size
        outputs = []
        chunk_size = frames_chunk_size

        # Precompute embeddings
        precomputed_embeds = []
        if self.pose_emb:
             device = token_list[0].device
             dtype = token_list[0].dtype

             # Create grid once using GridCache (via create_uv_grid)
             grid = create_uv_grid(
                width=patch_w,
                height=patch_h,
                aspect_ratio=W / H,
                dtype=dtype,
                device=device
             )

             # Compute embedding for each level efficiently
             unique_embeds = {}

             for i in range(len(self.projects)):
                 out_channels = self.projects[i].out_channels
                 if out_channels not in unique_embeds:
                     pos_emb = position_grid_to_embed(grid, out_channels)
                     pos_emb = pos_emb.permute(2, 0, 1) # (C, H, W)
                     unique_embeds[out_channels] = pos_emb
                 precomputed_embeds.append(unique_embeds[out_channels])

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
                    t = t + precomputed_embeds[i]

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
