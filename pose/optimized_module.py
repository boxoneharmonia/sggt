import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
from .module import AltRefAttBlock

class OptimizedAltRefAttBlock(AltRefAttBlock):
    """
    Optimized version of AltRefAttBlock that reduces redundant computations
    in the cross-frame attention mechanism.

    Optimization:
    The original AltRefAttBlock expands the reference frame (frame 0) `s-1` times
    and concatenates it with other frames before feeding it to the Attention module.
    This causes the linear projection (QKV) for the reference frame to be computed `s-1` times.

    This optimized version projects the reference frame once, and the other frames once,
    then expands the projected features. This reduces the FLOPs of the first linear layer
    by a factor of approximately 2*(s-1)/s.
    """
    def forward(self, x):
        b, s, n, c = x.shape
        x_norm = self.norm1(x)
        x_ref = x_norm[:, 0] # (b, n, c)

        # --- Optimized Attention Start ---
        # 1. Project Reference Frame Once
        # qkv_ref: (b, n, out_dim)
        qkv_ref = self.attn1.qkv(x_ref)

        # 2. Project Other Frames
        x_oth = x_norm[:, 1:] # (b, s-1, n, c)
        qkv_oth = self.attn1.qkv(x_oth.flatten(0, 1)) # (b*(s-1), n, out_dim)
        qkv_oth = qkv_oth.view(b, s - 1, n, -1)

        # 3. Construct Pairwise QKV
        # Expand projected reference features instead of input features
        qkv_ref_expanded = qkv_ref.unsqueeze(1).expand(-1, s - 1, -1, -1) # (b, s-1, n, out_dim)
        qkv = torch.cat([qkv_ref_expanded, qkv_oth], dim=2) # (b, s-1, 2n, out_dim)
        qkv = qkv.flatten(0, 1) # (b*(s-1), 2n, out_dim)

        # 4. Perform Attention (Replicating Attention.forward logic)
        # B_pair = b*(s-1), N_pair = 2n
        B_pair, N_pair, _ = qkv.shape

        # q, k, v splitting
        q, k, v = torch.split(qkv, [self.attn1.q_dim, self.attn1.kv_dim, self.attn1.kv_dim], dim=-1)

        q = q.reshape(B_pair, N_pair, self.attn1.num_heads, self.attn1.head_dim).permute(0, 2, 1, 3)
        k = k.reshape(B_pair, N_pair, self.attn1.kv_heads, self.attn1.head_dim).permute(0, 2, 1, 3)
        v = v.reshape(B_pair, N_pair, self.attn1.kv_heads, self.attn1.head_dim).permute(0, 2, 1, 3)

        with sdpa_kernel([SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]):
            x_att = F.scaled_dot_product_attention(
                q, k, v,
                dropout_p=self.attn1.attn_drop,
                scale=self.attn1.scale,
                enable_gqa=True if self.attn1.groups != 1 else False
            )

        x_att = x_att.transpose(1, 2)
        x_att = x_att.flatten(2)
        x_att = self.attn1.proj(x_att)
        x_pair_att = self.attn1.proj_drop(x_att)
        # --- Optimized Attention End ---

        x_pair_att = x_pair_att.view(b, s - 1, 2 * n, c)
        x_ref_updated_copies, x_oth_updated = x_pair_att.split(n, dim=2)

        agg_weights = self.aggregation_proj(x_ref_updated_copies) # (b, s-1, n, 1)
        agg_weights = F.softmax(agg_weights, dim=1)
        x_ref_final = (x_ref_updated_copies * agg_weights).sum(dim=1)

        x_ref_out = x[:, 0] + self.drop_path(self.ls1(x_ref_final))
        x_oth_out = x[:, 1:] + self.drop_path(self.ls1(x_oth_updated))

        x_out = torch.cat([x_ref_out.unsqueeze(1), x_oth_out], dim=1) # (b, s, n, c)

        x_out = x_out.view(b, s*n, c)
        x_out = x_out + self.drop_path(self.ls2(self.mlp2(self.norm2(x_out))))
        x_out = x_out.view(b*s, n, c)
        x_out = x_out + self.drop_path(self.ls3(self.attn3(self.norm3(x_out))))
        x_out = x_out + self.drop_path(self.ls4(self.mlp4(self.norm4(x_out))))
        x_out = x_out.view(b, s, n, c)
        return x_out
