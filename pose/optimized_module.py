import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
from pose.module import AltRefAttBlock, Attention

class OptimizedAltRefAttBlock(AltRefAttBlock):
    def forward(self, x):
        b, s, n, c = x.shape
        x_norm = self.norm1(x)
        x_ref = x_norm[:, 0] # (b, n, c)
        x_oth = x_norm[:, 1:] # (b, s-1, n, c)

        # Optimized attention for the first block (attn1)
        # Instead of constructing x_pair and passing it to attn1, we process ref and oth separately

        # Original logic:
        # x_ref_expanded = x_ref.unsqueeze(1).expand(-1, s - 1, -1, -1) # (b, s-1, n, c)
        # x_pair = torch.cat([x_ref_expanded, x_oth], dim=2)
        # x_pair_flat = x_pair.flatten(0, 1)
        # x_pair_att = self.attn1(x_pair_flat)

        # Optimized logic:
        x_pair_att = self.attn1_optimized(x_ref, x_oth)

        # Reshape as in original
        x_pair_att = x_pair_att.view(b, s - 1, 2 * n, c)
        x_ref_updated_copies, x_oth_updated = x_pair_att.split(n, dim=2)

        agg_weights = self.aggregation_proj(x_ref_updated_copies) # (b, s-1, n, 1)
        agg_weights = F.softmax(agg_weights, dim=1)
        x_ref_final = (x_ref_updated_copies * agg_weights).sum(dim=1)

        x_ref_out = x[:, 0] + self.drop_path(self.ls1(x_ref_final))
        x_oth_out = x[:, 1:] + self.drop_path(self.ls1(x_oth_updated))

        x = torch.cat([x_ref_out.unsqueeze(1), x_oth_out], dim=1) # (b, s, n, c)

        x = x.view(b, s*n, c)
        x = x + self.drop_path(self.ls2(self.mlp2(self.norm2(x))))
        x = x.view(b*s, n, c)
        x = x + self.drop_path(self.ls3(self.attn3(self.norm3(x))))
        x = x + self.drop_path(self.ls4(self.mlp4(self.norm4(x))))
        x = x.view(b, s, n, c)
        return x

    def attn1_optimized(self, x_ref, x_oth):
        """
        Optimized forward pass for attn1 that avoids redundant projection of x_ref.
        """
        attn_module = self.attn1
        B, N, C = x_ref.shape
        _, S_minus_1, _, _ = x_oth.shape

        # 1. Project x_ref ONCE
        qkv_ref = attn_module.qkv(x_ref) # (B, N, 3*head_dim*num_heads)

        # 2. Project x_oth (flatten first)
        x_oth_flat = x_oth.flatten(0, 1) # (B*(S-1), N, C)
        qkv_oth = attn_module.qkv(x_oth_flat) # (B*(S-1), N, 3*...)

        # 3. Process ref QKV
        q_dim = attn_module.q_dim
        kv_dim = attn_module.kv_dim
        q_ref, k_ref, v_ref = torch.split(qkv_ref, [q_dim, kv_dim, kv_dim], dim=-1)

        # Reshape to (B, N, n_heads, head_dim) -> permute to (B, n_heads, N, head_dim)
        q_ref = q_ref.reshape(B, N, attn_module.num_heads, attn_module.head_dim).permute(0, 2, 1, 3)
        k_ref = k_ref.reshape(B, N, attn_module.kv_heads, attn_module.head_dim).permute(0, 2, 1, 3)
        v_ref = v_ref.reshape(B, N, attn_module.kv_heads, attn_module.head_dim).permute(0, 2, 1, 3)

        # 4. Expand ref to match batch size B*(S-1)
        # We need to broadcast across the (S-1) dimension.
        # But q, k, v are 4D tensors (B, H, N, D).
        # We want effectively (B*(S-1), H, N, D).

        # (B, 1, n_heads, N, head_dim) -> expand -> flatten first 2 dims
        q_ref = q_ref.unsqueeze(1).expand(-1, S_minus_1, -1, -1, -1).flatten(0, 1)
        k_ref = k_ref.unsqueeze(1).expand(-1, S_minus_1, -1, -1, -1).flatten(0, 1)
        v_ref = v_ref.unsqueeze(1).expand(-1, S_minus_1, -1, -1, -1).flatten(0, 1)

        # 5. Process oth QKV
        q_oth, k_oth, v_oth = torch.split(qkv_oth, [q_dim, kv_dim, kv_dim], dim=-1)
        q_oth = q_oth.reshape(B*S_minus_1, N, attn_module.num_heads, attn_module.head_dim).permute(0, 2, 1, 3)
        k_oth = k_oth.reshape(B*S_minus_1, N, attn_module.kv_heads, attn_module.head_dim).permute(0, 2, 1, 3)
        v_oth = v_oth.reshape(B*S_minus_1, N, attn_module.kv_heads, attn_module.head_dim).permute(0, 2, 1, 3)

        # 6. Concatenate along the sequence length dimension (dim 2)
        # Sequence length becomes 2*N (N from ref, N from oth)
        q = torch.cat([q_ref, q_oth], dim=2) # (B_total, n_heads, 2N, head_dim)
        k = torch.cat([k_ref, k_oth], dim=2)
        v = torch.cat([v_ref, v_oth], dim=2)

        # 7. SDPA
        with sdpa_kernel([SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]):
            x = F.scaled_dot_product_attention(
                q, k, v,
                dropout_p=attn_module.attn_drop,
                scale=attn_module.scale,
                enable_gqa=True if attn_module.groups != 1 else False
            )

        # 8. Projection
        x = x.transpose(1, 2).flatten(2) # (B_total, 2N, C)
        x = attn_module.proj(x)
        x = attn_module.proj_drop(x)

        return x
