import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch.nn.attention import SDPBackend, sdpa_kernel

def drop_path(x, drop_prob: float = 0., training: bool = False):
    """
    Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output
    
class DropPath(nn.Module):
    """
    Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=0.):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

class LayerScale(nn.Module):
    def __init__(self, dim, init_values=1e-2) -> None:
        super().__init__()
        self.lambda1 = nn.Parameter(init_values * torch.ones(dim), requires_grad=True)

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        return hidden_state * self.lambda1
    
class Attention(nn.Module):
    def __init__(self,
                 dim,   # 输入token的dim
                 num_heads=8,
                 qkv_bias=False,
                 qk_scale=None,
                 attn_drop_ratio=0.,
                 proj_drop_ratio=0.,
                 groups=1):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.kv_heads = num_heads // groups
        self.groups = groups
        self.scale = qk_scale or self.head_dim ** -0.5
        self.q_dim = num_heads * self.head_dim
        self.kv_dim = self.kv_heads * self.head_dim
        self.qkv = nn.Linear(dim, self.q_dim + self.kv_dim * 2, bias=qkv_bias)
        self.attn_drop = attn_drop_ratio
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop_ratio)

    def forward(self, x):
        # x (batch_size, num_patches, embed_dim)
        B, N, C = x.shape
        qkv = self.qkv(x)
        q, k, v = torch.split(qkv, [self.q_dim, self.kv_dim, self.kv_dim], dim=-1)

        q = q.reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = k.reshape(B, N, self.kv_heads, self.head_dim).permute(0, 2, 1, 3)
        v = v.reshape(B, N, self.kv_heads, self.head_dim).permute(0, 2, 1, 3)

        with sdpa_kernel([SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]):
            x = F.scaled_dot_product_attention(
                q, k, v,
                dropout_p=self.attn_drop,
                scale=self.scale,
                enable_gqa=True if self.groups != 1 else False
            )
        x = x.transpose(1, 2)
        x = rearrange(x, 'b n h c -> b n (h c)')
        x = self.proj(x)
        x = self.proj_drop(x)
        return x    

class MLPSwiGLU(nn.Module):
    def __init__(self,inp,oup=None,hidden=None,drop=0.0, norm=nn.LayerNorm):
        super(MLPSwiGLU, self).__init__()
        oup = oup or inp
        hidden = hidden or inp
        hidden = (int(hidden * 2 / 3) + 7) // 8 * 8
        self.fc1 = nn.Sequential(
            nn.Linear(inp,2*hidden),
            norm(2*hidden),
            nn.Dropout(drop) if drop > 0. else nn.Identity()
        )
        self.fc2 = nn.Sequential(
            nn.Linear(hidden,oup),
            norm(oup),
        )
    
    def forward(self,x):
        x = self.fc1(x)
        x1, x2 = x.chunk(2, dim=-1)
        x3 = self.fc2(F.silu(x1) * x2)
        return x3

class MLP(nn.Module):
    def __init__(self,inp,oup=None,hidden=None, norm=nn.RMSNorm):
        super(MLP, self).__init__()
        oup = oup or inp
        hidden = hidden or inp
        self.fc = nn.Sequential(
            nn.Linear(inp,hidden),
            norm(hidden),
            nn.GELU(),
            nn.Linear(hidden,oup),
        )
    
    def forward(self,x):
        return self.fc(x)

class AltAttBlock(nn.Module):
    def __init__(self,
                 dim,
                 num_heads,
                 groups=1,
                 mlp_ratio=4.0,
                 qkv_bias=False,
                 qk_scale=None,
                 drop_ratio=0.,
                 attn_drop_ratio=0.,
                 drop_path_ratio=0.,
                 norm_layer=nn.RMSNorm,
                 ):
        super().__init__()
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.norm1 = norm_layer(dim)
        self.attn1 = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                                attn_drop_ratio=attn_drop_ratio, proj_drop_ratio=drop_ratio, groups=groups)
        self.norm2 = norm_layer(dim)
        self.mlp2 = MLPSwiGLU(dim, hidden=mlp_hidden_dim, drop=drop_ratio, norm=norm_layer)
        self.norm3 = norm_layer(dim)
        self.attn3 = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                                attn_drop_ratio=attn_drop_ratio, proj_drop_ratio=drop_ratio, groups=groups)
        self.norm4 = norm_layer(dim)
        self.mlp4 = MLPSwiGLU(dim, hidden=mlp_hidden_dim, drop=drop_ratio, norm=norm_layer)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path_ratio) if drop_path_ratio > 0. else nn.Identity()
        self.ls1 = LayerScale(dim)
        self.ls2 = LayerScale(dim)
        self.ls3 = LayerScale(dim)
        self.ls4 = LayerScale(dim)

    def forward(self, x):
        s = x.shape[1]
        x = rearrange(x, 'b s n c -> b (s n) c', s=s)
        x = x + self.drop_path(self.ls1(self.attn1(self.norm1(x))))
        x = x + self.drop_path(self.ls2(self.mlp2(self.norm2(x))))
        x = rearrange(x, 'b (s n) c -> (b s) n c', s=s)
        x = x + self.drop_path(self.ls3(self.attn3(self.norm3(x))))
        x = x + self.drop_path(self.ls4(self.mlp4(self.norm4(x))))
        x = rearrange(x, '(b s) n c -> b s n c', s=s)
        return x

class AltRefAttBlock(nn.Module):
    def __init__(self,
                 dim,
                 num_heads,
                 groups=1,
                 mlp_ratio=4.0,
                 qkv_bias=False,
                 qk_scale=None,
                 drop_ratio=0.,
                 attn_drop_ratio=0.,
                 drop_path_ratio=0.,
                 norm_layer=nn.RMSNorm,
                 ):
        super().__init__()
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.norm1 = norm_layer(dim)
        self.attn1 = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                                attn_drop_ratio=attn_drop_ratio, proj_drop_ratio=drop_ratio, groups=groups)
        self.norm2 = norm_layer(dim)
        self.mlp2 = MLPSwiGLU(dim, hidden=mlp_hidden_dim, drop=drop_ratio, norm=norm_layer)
        self.norm3 = norm_layer(dim)
        self.attn3 = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                                attn_drop_ratio=attn_drop_ratio, proj_drop_ratio=drop_ratio, groups=groups)
        self.norm4 = norm_layer(dim)
        self.mlp4 = MLPSwiGLU(dim, hidden=mlp_hidden_dim, drop=drop_ratio, norm=norm_layer)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path_ratio) if drop_path_ratio > 0. else nn.Identity()
        self.ls1 = LayerScale(dim)
        self.ls2 = LayerScale(dim)
        self.ls3 = LayerScale(dim)
        self.ls4 = LayerScale(dim)
        self.aggregation_proj = nn.Linear(dim, 1)

    def forward(self, x):
        s = x.shape[1]
        x_norm = self.norm1(x)
        x_ref = x_norm[:, 0] # (b, n, c)
        x_ref = x_ref.unsqueeze(1).expand(-1, s - 1, -1, -1) # (b, s-1, n, c)
        x_oth = x_norm[:, 1:] # (b, s-1, n, c)
        x_pair = torch.cat([x_ref, x_oth], dim=2)
        x_pair = rearrange(x_pair, 'b s n c -> (b s) n c', s=s-1)
        x_pair_att = self.attn1(x_pair)
        x_pair_att = rearrange(x_pair_att, '(b s) n c -> b s n c', s=s-1)
        x_ref_att, x_oth_att = x_pair_att.chunk(2, dim=2)
        agg_weights = self.aggregation_proj(x_ref_att)
        agg_weights = F.softmax(agg_weights, dim=1)
        x_ref_att = (x_ref_att * agg_weights).sum(dim=1, keepdim=True)
        x_att = torch.cat([x_ref_att, x_oth_att], dim=1)
        x_att = rearrange(x_att, 'b s n c -> b (s n) c', s=s)

        x = rearrange(x, 'b s n c -> b (s n) c', s=s)
        x = x + self.drop_path(self.ls1(x_att))
        x = x + self.drop_path(self.ls2(self.mlp2(self.norm2(x))))
        x = rearrange(x, 'b (s n) c -> (b s) n c', s=s)
        x = x + self.drop_path(self.ls3(self.attn3(self.norm3(x))))
        x = x + self.drop_path(self.ls4(self.mlp4(self.norm4(x))))
        x = rearrange(x, '(b s) n c -> b s n c', s=s)
        return x

class MHSABlock(nn.Module):
    def __init__(self,
                 dim,
                 num_heads,
                 groups=1,
                 mlp_ratio=4.0,
                 qkv_bias=False,
                 qk_scale=None,
                 drop_ratio=0.,
                 attn_drop_ratio=0.,
                 drop_path_ratio=0.,
                 norm_layer=nn.RMSNorm,
                 ):
        super().__init__()
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.norm1 = norm_layer(dim)
        self.attn1 = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                                attn_drop_ratio=attn_drop_ratio, proj_drop_ratio=drop_ratio, groups=groups)
        self.norm2 = norm_layer(dim)
        self.mlp2 = MLPSwiGLU(dim, hidden=mlp_hidden_dim, drop=drop_ratio, norm=norm_layer)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path_ratio) if drop_path_ratio > 0. else nn.Identity()
        self.ls1 = LayerScale(dim)
        self.ls2 = LayerScale(dim)

    def forward(self, x):
        x = x + self.drop_path(self.ls1(self.attn1(self.norm1(x))))
        x = x + self.drop_path(self.ls2(self.mlp2(self.norm2(x))))
        return x

class CrossAttention(nn.Module):
    def __init__(self,
                 dim,
                 num_heads=8,
                 qkv_bias=False,
                 qk_scale=None,
                 attn_drop_ratio=0.,
                 proj_drop_ratio=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)

        self.attn_drop = attn_drop_ratio
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop_ratio)

    def forward(self, query, memory):  # query = q, memory = k/v
        B, N, C = query.shape
        _, M, _ = memory.shape

        q = self.q(query).reshape(B, N, self.num_heads, C // self.num_heads).transpose(1, 2)
        kv = self.kv(memory).reshape(B, M, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        with sdpa_kernel([SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION]):
            x = F.scaled_dot_product_attention(
                q, k, v,
                dropout_p=self.attn_drop,
                scale=self.scale
            )

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class DecoderBlock(nn.Module):
    def __init__(self,
                 dim,
                 num_heads,
                 seq_len,
                 mlp_ratio=4.0,
                 qkv_bias=False,
                 qk_scale=None,
                 drop_ratio=0.0,
                 attn_drop_ratio=0.0,
                 drop_path_ratio=0.0,
                 norm_layer=nn.RMSNorm):
        super().__init__()
        self.sequence_lenth = seq_len
        self.norm1 = norm_layer(dim)
        self.self_attn = Attention(dim, num_heads, qkv_bias, qk_scale, attn_drop_ratio, drop_ratio)
        self.norm2 = norm_layer(dim)
        self.cross_attn = CrossAttention(dim, num_heads, qkv_bias, qk_scale, attn_drop_ratio, drop_ratio)
        self.drop_path = DropPath(drop_path_ratio) if drop_path_ratio > 0. else nn.Identity()
        self.norm3 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLPSwiGLU(dim, hidden=mlp_hidden_dim, drop=drop_ratio, norm=norm_layer)
        self.ls1 = LayerScale(dim)
        self.ls2 = LayerScale(dim)
        self.ls3 = LayerScale(dim)

    def forward(self, query, memory):
        s = self.sequence_lenth
        query = rearrange(query, '(b s) n c -> b (s n) c', s=s)
        query = query + self.drop_path(self.ls1(self.self_attn(self.norm1(query))))              # Self-Attention
        query = rearrange(query, 'b (s n) c -> (b s) n c', s=s)
        query = query + self.drop_path(self.ls2(self.cross_attn(self.norm2(query), memory)))     # Cross-Attention
        query = query + self.drop_path(self.ls3(self.mlp(self.norm3(query))))                    # FFN
        return query


## DPT
class ConvDw(nn.Module):
    # Serapable convolution module consisting of
    # 1. Depthwise convolution (3x3)
    # 2. pointwise convolution (1x1)
    # Reference:
    # Andrew G. Howard, Menglong Zhu, Bo Chen, Dmitry Kalenichenko, Weijun Wang,
    # Tobias Weyand, Marco Andreetto, and Hartwig Adam. MobileNets: Efficient
    # convolutional neural neworks for mobile vision applications. CoRR, abs/1704.04861, 2017.

    def __init__(self, inp, oup, stride):
        super(ConvDw, self).__init__()
        self.conv = nn.Sequential(
            # dw
            nn.Conv2d(inp, inp, 3, stride=stride, padding=1, groups=inp, bias=True),
            # nn.BatchNorm2d(inp),
            nn.ReLU(inplace=True),
            # pw
            nn.Conv2d(inp, oup, 1, stride=1, padding=0, bias=True),
            # nn.BatchNorm2d(oup),
            nn.ReLU(inplace=True),
        )
        # self.depth = oup
    def forward(self, x):
        return self.conv(x)

class InvertedBottleneck(nn.Module):
    def __init__(self, inp, oup, stride, hidden_ratio) -> None:
        super().__init__()
        hidden_dim = int(inp * hidden_ratio)
        self.conv = nn.Sequential(
            nn.Conv2d(inp, hidden_dim, 1, stride=1, padding=0, bias=True),
            # nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            ConvDw(hidden_dim, oup, stride)
        )
        self.residual = False
        if inp == oup and stride == 1:
            self.residual = True
            self.skip_add = torch.ao.nn.quantized.FloatFunctional()

    def forward(self, x):
        if self.residual:
            return self.skip_add.add(x, self.conv(x))
        else:
            return self.conv(x) 

class FeatureFusionBlock(nn.Module):
    """Feature fusion block."""

    def __init__(
        self,
        features,
        hidden_ratio=2.0,
        expand=False,
        align_corners=True,
        has_residual=True,
    ):
        super().__init__()
        self.align_corners = align_corners
        self.expand = expand
        out_features = features if self.expand == False else features // 2
        self.out_conv = nn.Conv2d(features, out_features, kernel_size=1, stride=1, padding=0, bias=True)
        if has_residual:
            self.resConfUnit1 = InvertedBottleneck(features, features, 1, hidden_ratio)
            self.skip_add = torch.ao.nn.quantized.FloatFunctional()

        self.has_residual = has_residual
        self.resConfUnit2 = InvertedBottleneck(features, features, 1, hidden_ratio)
        
    def forward(self, xs):
        """Forward pass.

        Returns:
            tensor: output
        """
        output = xs[0]

        if self.has_residual:
            res = self.resConfUnit1(xs[1])
            output = self.skip_add.add(output, res)

        output = self.resConfUnit2(output)
        output = F.interpolate(output, scale_factor=2.0, mode='bilinear', align_corners=self.align_corners)
        output = self.out_conv(output)
        return output

def position_grid_to_embed(pos_grid: torch.Tensor, embed_dim: int, omega_0: float = 100) -> torch.Tensor:
    """
    Convert 2D position grid (HxWx2) to sinusoidal embeddings (HxWxC)

    Args:
        pos_grid: Tensor of shape (H, W, 2) containing 2D coordinates
        embed_dim: Output channel dimension for embeddings

    Returns:
        Tensor of shape (H, W, embed_dim) with positional embeddings
    """
    H, W, grid_dim = pos_grid.shape
    assert grid_dim == 2
    pos_flat = pos_grid.reshape(-1, grid_dim)  # Flatten to (H*W, 2)

    # Process x and y coordinates separately
    emb_x = make_sincos_pos_embed(embed_dim // 2, pos_flat[:, 0], omega_0=omega_0)  # [1, H*W, D/2]
    emb_y = make_sincos_pos_embed(embed_dim // 2, pos_flat[:, 1], omega_0=omega_0)  # [1, H*W, D/2]

    # Combine and reshape
    emb = torch.cat([emb_x, emb_y], dim=-1)  # [1, H*W, D]

    return emb.view(H, W, embed_dim)  # [H, W, D]


def make_sincos_pos_embed(embed_dim: int, pos: torch.Tensor, omega_0: float = 100) -> torch.Tensor:
    """
    This function generates a 1D positional embedding from a given grid using sine and cosine functions.

    Args:
    - embed_dim: The embedding dimension.
    - pos: The position to generate the embedding from.

    Returns:
    - emb: The generated 1D positional embedding.
    """
    assert embed_dim % 2 == 0
    device = pos.device
    omega = torch.arange(embed_dim // 2, dtype=torch.float32, device=device)
    omega /= embed_dim / 2.0
    omega = 1.0 / omega_0**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = torch.einsum("m,d->md", pos, omega)  # (M, D/2), outer product

    emb_sin = torch.sin(out)  # (M, D/2)
    emb_cos = torch.cos(out)  # (M, D/2)

    emb = torch.cat([emb_sin, emb_cos], dim=1)  # (M, D)
    return emb.float()


# Inspired by https://github.com/microsoft/moge


def create_uv_grid(
    width: int, height: int, aspect_ratio: float, dtype: torch.dtype, device: torch.device
) -> torch.Tensor:
    """
    Create a normalized UV grid of shape (height, width, 2).

    The grid spans horizontally and vertically according to an aspect ratio,
    ensuring the top-left corner is at (-x_span, -y_span) and the bottom-right
    corner is at (x_span, y_span), normalized by the diagonal of the plane.

    Args:
        width (int): Number of points horizontally.
        height (int): Number of points vertically.
        aspect_ratio (float, optional): Width-to-height ratio. Defaults to width/height.
        dtype (torch.dtype, optional): Data type of the resulting tensor.
        device (torch.device, optional): Device on which the tensor is created.

    Returns:
        torch.Tensor: A (height, width 2) tensor of UV coordinates.
    """
    # Derive aspect ratio if not explicitly provided
    if aspect_ratio is None:
        aspect_ratio = float(width) / float(height)

    # Compute normalized spans for X and Y
    diag_factor = (aspect_ratio**2 + 1.0) ** 0.5
    span_x = aspect_ratio / diag_factor
    span_y = 1.0 / diag_factor

    # Establish the linspace boundaries
    left_x = -span_x * (width - 1) / width
    right_x = span_x * (width - 1) / width
    top_y = -span_y * (height - 1) / height
    bottom_y = span_y * (height - 1) / height

    # Generate 1D coordinates
    x_coords = torch.linspace(left_x, right_x, steps=width, dtype=dtype, device=device)
    y_coords = torch.linspace(top_y, bottom_y, steps=height, dtype=dtype, device=device)

    # Create 2D meshgrid (width x height) and stack into UV
    uu, vv = torch.meshgrid(x_coords, y_coords, indexing="xy")
    uv_grid = torch.stack((uu, vv), dim=-1)

    return uv_grid

class DPTHead(nn.Module):
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
    
    def forward(self, token_list, image_size, frames_chunk_size=2):
        B, S, N, C = token_list[0].shape
        H, W = image_size
        patch_h, patch_w = H // self.patch_size, W // self.patch_size
        outputs = []
        chunk_size = frames_chunk_size if frames_chunk_size is not None else S
        for start_idx in range(0, S, chunk_size):
            end_idx = min(start_idx + chunk_size, S)
            current_s = end_idx - start_idx
            pyramid_features = []
            for i, token in enumerate(token_list):
                t = token[:, start_idx:end_idx]
                t = rearrange(t, 'b s n c -> (b s) n c')
                t = self.norm[i](t)
                t = rearrange(t, 'bs (h w) c -> bs c h w', h=patch_h, w=patch_w)
                t = self.projects[i](t)
                
                if self.pose_emb:
                    t = self._apply_pos_embed(t, W, H)

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
            out = out.view(B, current_s, *out.shape[1:])
            outputs.append(out)
            
        return torch.cat(outputs, dim=1)
    
    def _apply_pos_embed(self, x: torch.Tensor, W: int, H: int) -> torch.Tensor:
        """
        Apply positional embedding to tensor x.
        Args:
            x: Feature map (B, C, H_feat, W_feat)
            W: Original image width
            H: Original image height
        """
        patch_h, patch_w = x.shape[-2], x.shape[-1]
        pos_embed = create_uv_grid(
            width=patch_w, 
            height=patch_h, 
            aspect_ratio=W / H, 
            dtype=x.dtype, 
            device=x.device
        )
        
        pos_embed = position_grid_to_embed(pos_embed, x.shape[1])
        pos_embed = pos_embed.permute(2, 0, 1).unsqueeze(0)
        
        return x + pos_embed