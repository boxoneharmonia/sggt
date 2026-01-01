
import torch
import torch.nn as nn
from einops import rearrange
from .module import DPTHead, create_uv_grid, position_grid_to_embed

class OptimizedDPTHead(DPTHead):
    """
    Optimized version of DPTHead that precomputes positional embeddings
    outside of the chunk loop to avoid redundant calculations.
    """
    def forward(self, token_list, image_size, frames_chunk_size=2):
        B, N, C = token_list[0].shape
        H, W = image_size
        patch_h, patch_w = H // self.patch_size, W // self.patch_size
        outputs = []
        chunk_size = frames_chunk_size

        # Optimization: Precompute positional embedding if needed
        # It depends on patch_h, patch_w, W, H and hidden_dim.
        # Check if pose_emb is enabled
        pos_embed_tensor = None
        if self.pose_emb:
            # We need to know the hidden dimension.
            # In DPTHead.__init__, self.projects is defined.
            # All elements in self.projects map to 'hidden_dim', except the 3rd one which maps to 'features'.
            # Wait, let's check DPTHead.__init__ again.
            # self.projects[0]: out=hidden_dim
            # self.projects[1]: out=hidden_dim
            # self.projects[2]: out=features
            # self.projects[3]: out=hidden_dim

            # The logic in original forward:
            # for i, token in enumerate(token_list):
            #    ... projects[i](t) ...
            #    _apply_pos_embed(t, W, H)

            # So the channel dim passed to _apply_pos_embed matches the output of projects[i].
            # This means the embedding might differ in channel dimension for i=2 vs others.

            # So we should compute one embedding for 'hidden_dim' and one for 'features' if they differ.
            # DPTHead.__init__:
            # hidden_dim = int(hidden_ratio*features)

            # So if hidden_ratio != 1.0, we have two different channel sizes.

            # Let's precompute both if necessary, or compute on demand but cache it for the loop.
            # Since the number of layers is small (4), we can just compute per-layer embeddings once.

            pos_embeds = []
            for i in range(len(token_list)):
                # Determine channels
                # self.projects is a ModuleList corresponding to token_list indices
                out_channels = self.projects[i].out_channels

                # Create embedding
                # We can replicate logic from _apply_pos_embed but return just the embedding
                pos_embed = self._create_pos_embed_tensor(out_channels, patch_h, patch_w, W, H, token_list[0].dtype, token_list[0].device)
                pos_embeds.append(pos_embed)

        for start_idx in range(0, B, chunk_size):
            end_idx = min(start_idx + chunk_size, B)
            pyramid_features = []
            for i, token in enumerate(token_list):
                t = token[start_idx:end_idx]
                t = self.norm[i](t)
                t = rearrange(t, 'b (h w) c -> b c h w', h=patch_h, w=patch_w)
                t = self.projects[i](t)

                if self.pose_emb:
                    # t = self._apply_pos_embed(t, W, H)
                    # Instead of calling _apply_pos_embed which recomputes, we add the precomputed one.
                    # _apply_pos_embed returns x + pos_embed
                    t = t + pos_embeds[i]

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

    def _create_pos_embed_tensor(self, channels, patch_h, patch_w, W, H, dtype, device):
        pos_embed = create_uv_grid(
            width=patch_w,
            height=patch_h,
            aspect_ratio=W / H,
            dtype=dtype,
            device=device
        )

        pos_embed = position_grid_to_embed(pos_embed, channels)
        pos_embed = pos_embed.permute(2, 0, 1)
        return pos_embed
