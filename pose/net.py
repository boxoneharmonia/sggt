import os
import numpy as np
import random
import torch
import torch.nn as nn
from transformers import AutoModel
from transformers.optimization import get_scheduler
import torch.optim as optim
import bitsandbytes as bnb
from tqdm import tqdm
import pandas as pd
import math
import json

from .module import *

class Encoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embed_dim = config.embed_dim
        self.base = AutoModel.from_pretrained(config.base, use_safetensors=True)
        for param in self.base.parameters():
            param.requires_grad = False
        self.num_registers = 4 # from <Vit need registers>
        self.proj_encoder = nn.Linear(self.base.config.hidden_size, config.embed_dim) if self.base.config.hidden_size != config.embed_dim else nn.Identity()
        encoder_layers = []
        for i in range(config.encoder_depth//2):
            encoder_layers.append(AltRefAttBlock(dim=config.embed_dim, num_heads=config.num_heads, mlp_ratio=config.mlp_ratio, qkv_bias=config.qkv_bias, qk_scale=config.qk_scale,
                                        drop_ratio=config.drop_ratio, attn_drop_ratio=config.attn_drop_ratio, drop_path_ratio=config.drop_path_ratio, groups=config.groups))
        self.encoder = nn.ModuleList(encoder_layers)
        self.cam_token = nn.Parameter(torch.zeros(1, 1, 1, config.embed_dim), requires_grad=True)
        self.register = nn.Parameter(torch.zeros(1, 1, self.num_registers, config.embed_dim), requires_grad=True)
        nn.init.normal_(self.cam_token, std=1e-4)
        nn.init.normal_(self.register, std=1e-4)
        self.spatial_emb = nn.Parameter(torch.zeros(1, 1, config.num_patches + 1 + self.num_registers, config.embed_dim), requires_grad=True)
        self.feature_idx = set(config.feature_idx)

    def forward(self, x:torch.Tensor):
        """ 
        Forward pass of the transformer.
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, 3, H, W).
        """
        b,s = x.shape[0], x.shape[1]
        image_size = x.shape[-2:]
        x = x.flatten(0,1)
        x = self.base(x).last_hidden_state[:,1:,:]
        x = self.proj_encoder(x)  # (bs n c)
        n = x.shape[1]
        x = rearrange(x, '(b s) n c -> b s n c', b=b)

        cam_tokens = self.cam_token.expand(b, s, -1, -1).contiguous()
        registers = self.register.expand(b, s, -1, -1).contiguous()
        x = torch.cat((x, cam_tokens, registers), dim=2)   # (b s n+1+4 c)
        temporal_pe = self.get_sinusoidal_encoding(s, self.embed_dim, x.device, x.dtype)
        x = x + self.spatial_emb + temporal_pe
        cam_token_list = []
        features_list = []
        for i, layer in enumerate(self.encoder):
            x = layer(x)
            if i in self.feature_idx:
                cam_token = x[:,:,n] # (b s c)
                cam_token_list.append(cam_token)
                features = x[:,:,0:n] # (b s n c)
                features_list.append(features)

        return cam_token_list, features_list, image_size

    def get_sinusoidal_encoding(self, seq_len, dim, device, dtype):
        position = torch.arange(seq_len, device=device, dtype=dtype).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2, device=device, dtype=dtype) * (-math.log(10000.0) / dim))
        pe = torch.zeros(seq_len, dim, device=device, dtype=dtype)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.view(1, seq_len, 1, dim)

class Decoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        cam_layers = []
        for i in range(config.decoder_depth):
            cam_layers.append(MHSABlock(dim=config.embed_dim, num_heads=config.num_heads, mlp_ratio=config.mlp_ratio//2, qkv_bias=config.qkv_bias, qk_scale=config.qk_scale,
                                        drop_ratio=config.drop_ratio, attn_drop_ratio=config.attn_drop_ratio, drop_path_ratio=config.drop_path_ratio))
        self.cam_decoder = nn.Sequential(*cam_layers)
        self.R_head = MLP(config.embed_dim, 6) # 6D rotation
        self.t_head = MLP(config.embed_dim, 3)
        self.map_head = DPTHead(inp=config.embed_dim, oup=config.maps, hidden_ratio=config.mlp_ratio//2, features=config.dpt_features, patch_size=config.patch_size, use_conf=True)
        self.pmap_head = DPTHead(inp=config.embed_dim, oup=3, hidden_ratio=config.mlp_ratio//2, features=config.dpt_features, patch_size=config.patch_size, use_conf=True)
        features_len = len(config.feature_idx)
        self.cam_route = nn.Parameter(torch.full([features_len, 1], 1.))
        self.map_route = nn.Parameter(torch.full([features_len, 4], 1.))
        self.pmap_route = nn.Parameter(torch.full([features_len, 4], 1.))

    def forward(self, cam_token_list, features_list_all, image_size):
        cam_token = self._apply_routing(cam_token_list, self.cam_route)[0]
        cam = self.cam_decoder(cam_token)
        pose_R_6d = self.R_head(cam)
        pose_R = rotation_6d_to_matrix(pose_R_6d) # (b s 3 3)
        pose_t = self.t_head(cam) # (b s 3)
        pose = torch.cat([pose_R, pose_t[..., None]], dim=-1)
        map_features  = self._apply_routing(features_list_all, self.map_route)
        pmap_features = self._apply_routing(features_list_all, self.pmap_route)
        maps = self.map_head(map_features, image_size)
        mask = maps[:, :, -1:]
        heatmap = maps[:, :, :-1]
        pmap = self.pmap_head(pmap_features, image_size)
        return pose, mask, heatmap, pmap
    
    def _apply_routing(self, features_list, route_weights):
        """
        Args:
            features_list: List of L tensors, each (B, S, N, C)
            route_weights: Parameter of shape (L, l)
        Returns:
            routed_features: List of l tensors, each (B, S, N, C)
        """
        weights = F.softmax(route_weights, dim=0)
        out_dim = weights.shape[1]
        
        routed_features = []
        for k in range(out_dim): 
            weighted_feat = 0
            for i, feat in enumerate(features_list):
                w = weights[i, k]
                weighted_feat = weighted_feat + feat * w
            routed_features.append(weighted_feat)
            
        return routed_features
    
class MyNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.encoder = Encoder(config)
        self.decoder = Decoder(config)

    def forward(self, x:torch.Tensor):
        cam_token_list, features_list, image_size = self.encoder(x)
        pose, mask, heatmap, pmap = self.decoder(cam_token_list, features_list, image_size)
        out_dict = {
            'poses': pose,
            'masks': mask,
            'heatmaps': heatmap,
            'pmaps': pmap
        }
        return out_dict

# Loss fn
class PoseLoss(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.lossfn_t = nn.MSELoss(reduction='mean')
        self.lossfn_R = nn.SmoothL1Loss(reduction='mean', beta=config.R_beta)
        self.lossfn_bce_mask = nn.BCEWithLogitsLoss(reduction='mean')
        self.lossfn_bce_conf = nn.BCEWithLogitsLoss(reduction='none')
        self.lossfn_pts = nn.MSELoss(reduction='none')
        self.pcloud_alpha = config.pcloud_alpha

        self.t_weight = config.t_weight
        self.R_weight = config.R_weight
        self.geo_weight = config.geo_weight
        self.mask_weight = config.mask_weight
        self.pts_weight = config.pts_weight
        self.var_weight = config.var_weight
        self.pcloud_weight = config.pcloud_weight
        self.pconf_weight = config.pconf_weight

        ptsfile = os.path.join(config.root, config.ptsfile)
        self.pts3d = torch.tensor(json.load(open(ptsfile, 'r'))[0]) / 1000.0 # shape (n 3)

    def forward(self, out_dict, data_dict, eps=1e-6):
        # 1) pose
        R_cams = data_dict['R_cams'].flatten(0, 1)
        t_cams = data_dict['t_cams'].flatten(0, 1)
        SE3_pred = out_dict['poses']
        R_preds, t_preds = compute_abs_pose(SE3_pred)
        R_preds = R_preds.flatten(0, 1)
        t_preds = t_preds.flatten(0, 1)
        loss_t = self.lossfn_t(t_preds, t_cams)
        R_errs  = R_preds.transpose(-1, -2) @ R_cams
        traces = R_errs.diagonal(offset=0, dim1=-2, dim2=-1).sum(-1)
        cos_theta = ((traces - 1) / 2.0).clamp(min=-1+eps, max=1-eps)
        theta = torch.acos(cos_theta)
        loss_R = self.lossfn_R(theta, torch.zeros_like(theta))
        # 2) add
        pts3d = self.pts3d.to(R_cams.device)
        pts3d = pts3d.unsqueeze(0).expand(R_cams.shape[0], -1, -1)
        pts3d_gt = pts3d @ R_cams.transpose(-1, -2) + t_cams.view(-1, 1, 3)
        pts3d_pred = pts3d @ R_preds.transpose(-1, -2) + t_preds.view(-1, 1, 3)
        loss_add = self.lossfn_t(pts3d_pred, pts3d_gt)
        # 3) Wide-Depth-Range 
        cam_Ks = data_dict['cam_Ks'].flatten(0, 1)
        pts2d_gt = data_dict['pts2d'].flatten(0, 1)
        pts2d_homo = torch.cat([pts2d_gt, torch.ones_like(pts2d_gt[..., :1])], dim=-1)
        K_inv = torch.linalg.inv(cam_Ks)
        rays = pts2d_homo @ K_inv.transpose(-1, -2)
        rays_norm = F.normalize(rays, p=2, dim=-1)
        dot_prod = F.relu(pts3d_pred * rays_norm).sum(dim=-1, keepdim=True) 
        proj_on_ray = dot_prod * rays_norm
        dist_vec = pts3d_pred - proj_on_ray
        loss_geo = self.lossfn_t(dist_vec, torch.zeros_like(dist_vec))
        # 4) mask
        masks_pred = out_dict['masks'].flatten(0, 1)
        masks_gt = data_dict['masks'].flatten(0, 1)
        size = masks_pred.shape[-2:]
        masks_gt = F.interpolate(masks_gt, size, mode='nearest-exact')
        loss_mask = self.lossfn_bce_mask(masks_pred, masks_gt)
        # 5) pts2d
        _, _, _, H, W = data_dict['images'].shape
        heatmaps = out_dict['heatmaps'].flatten(0, 1)
        pts_vis = data_dict['pts_vis'].flatten(0, 1)
        scale_pts = torch.tensor([W, H], device=pts2d_gt.device).float()
        pts_gt_norm = pts2d_gt / scale_pts.view(1, 1, 2)
        pts_pred, var = integral_heatmap_layer_with_variance(heatmaps)
        num_vis = pts_vis.sum().clamp(min=1)
        loss_pts = (self.lossfn_pts(pts_pred, pts_gt_norm).sum(dim=-1) * pts_vis).sum() / num_vis
        loss_var = (var * pts_vis).sum() / num_vis
        # 6) pclouds
        pmaps = out_dict['pmaps'].flatten(0, 1)
        pclouds_gt = data_dict['pclouds'].flatten(0, 1)
        pconfs_gt = data_dict['pconfs'].flatten(0, 1)
        pclouds_gt = F.interpolate(pclouds_gt, size, mode='nearest-exact')
        pconfs_gt = F.interpolate(pconfs_gt, size, mode='nearest-exact')
        pclouds, pconfs_logits  = torch.split(pmaps, [3,1], dim=1)
        pdist = torch.norm(pclouds - pclouds_gt, p=1, dim=1, keepdim=True)
        pconfs = torch.sigmoid(pconfs_logits)
        num_pclouds = pconfs_gt.sum() + 1e-6
        loss_valid = (pconfs * pdist) - self.pcloud_alpha * torch.log(pconfs + 1e-6)
        loss_pcloud = (loss_valid * pconfs_gt).sum() / num_pclouds
        background = ((1 - masks_gt) * (1 - pconfs_gt))
        loss_invalid = self.lossfn_bce_conf(pconfs_logits, torch.zeros_like(pconfs_logits))
        loss_pconf = (loss_invalid * background).sum() / background.sum()
        loss_dict = {
            'loss_t': loss_t,
            'loss_R': loss_R,
            'loss_add': loss_add,
            'loss_geo': loss_geo,
            'loss_mask': loss_mask,
            'loss_pts': loss_pts,
            'loss_var': loss_var,
            'loss_pcloud': loss_pcloud,
            'loss_pconf': loss_pconf,
        }
        loss = loss_add + self.t_weight *loss_t + self.R_weight * loss_R + self.geo_weight * loss_geo + self.mask_weight * loss_mask + self.pts_weight * loss_pts \
            + self.var_weight * loss_var + self.pcloud_weight * loss_pcloud + self.pconf_weight * loss_pconf
        return loss, loss_dict

def se3_exp(x, eps=1e-6):
    """
    Batched SE(3) exponential map.
    Args:
        x (torch.Tensor): A tensor of shape (b, 6), v = x[..., :3] (translation), omega = x[..., 3:] (rotation).
    Returns:
        torch.Tensor:   A tensor of shape (b, 3, 3), representing the SO(3) transformation matrices.
                        A tensor of shape (b, 3, 1), representing the R(3) vectors.
    """
    b = x.shape[0]
    v, omega = x.split(3, dim=-1)

    zeros = torch.zeros(b, 1, device=x.device, dtype=x.dtype)
    omega_x, omega_y, omega_z = omega.split(1, dim=-1)
    row1 = torch.cat([zeros, -omega_z, omega_y], dim=-1)
    row2 = torch.cat([omega_z, zeros, -omega_x], dim=-1)
    row3 = torch.cat([-omega_y, omega_x, zeros], dim=-1)
    omega_hat = torch.stack([row1, row2, row3], dim=1) # (b, 3, 3)

    theta = torch.linalg.norm(omega, dim=-1, keepdim=True).unsqueeze(-1) # (b, 1, 1)
    small_angle_mask = theta < eps
    
    # Use Taylor expansion for small angles
    # A = 1 - theta^2/6, B = 1/2 - theta^2/24, C = 1/6 - theta^2/120
    A_small = 1.0 - theta**2 / 6.0
    B_small = 0.5 - theta**2 / 24.0
    C_small = 1.0/6.0 - theta**2 / 120.0
    
    # Standard formula for large angles
    theta_safe = theta.clamp(min=eps)
    A_large = torch.sin(theta) / theta_safe
    B_large = (1 - torch.cos(theta)) / theta_safe**2
    C_large = (1 - A_large) / theta_safe**2

    # Use torch.where to select based on angle magnitude
    A = torch.where(small_angle_mask, A_small, A_large)
    B = torch.where(small_angle_mask, B_small, B_large)
    C = torch.where(small_angle_mask, C_small, C_large)

    I = torch.eye(3, device=x.device, dtype=x.dtype).expand(b, -1, -1)
    R = I + A * omega_hat + B * (omega_hat @ omega_hat) # (b, 3, 3)
    V = I + B * omega_hat + C * (omega_hat @ omega_hat)

    t = (V @ v.unsqueeze(-1)) # (b, 3, 1)
    return R, t

def rotation_6d_to_matrix(d6: torch.Tensor) -> torch.Tensor:
    """
    Converts 6D rotation representation to 3x3 rotation matrix.
    Based on Zhou et al., "On the Continuity of Rotation Representations in Neural Networks", CVPR 2019
    Input:
        d6: (B, ..., 6)
    Output:
        R: (B, ..., 3, 3)
    """
    a1, a2 = d6[..., :3], d6[..., 3:]
    b1 = F.normalize(a1, dim=-1)
    proj = (b1 * a2).sum(dim=-1, keepdim=True) * b1
    b2 = a2 - proj
    b2 = F.normalize(b2, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack((b1, b2, b3), dim=-1)

def compute_abs_pose(pose):
    """
    Computes absolute poses from a sequence of relative poses (Matrix form).
    
    Args:
        pose: (B, S, 4, 4) or (B, S, 3, 4)
              - pose[:, 0] is the reference pose (Anchor).
              - pose[:, 1:] are the relative poses T_{ref->i}.
    
    Returns:
        R: (B, S, 3, 3) Absolute rotation matrices
        t: (B, S, 3, 1) Absolute translation vectors
    """
    B, S, H, W = pose.shape
    device = pose.device
    dtype = pose.dtype

    if H == 3 and W == 4:
        padding = torch.tensor([0, 0, 0, 1], device=device, dtype=dtype).view(1, 1, 1, 4)
        padding = padding.expand(B, S, 1, 4)
        pose = torch.cat([pose, padding], dim=2) # (B, S, 4, 4)

    T0 = pose[:, 0] 
    dT = pose[:, 1:]
    T_abs = torch.zeros(B, S, 4, 4, device=device, dtype=dtype)
    T_abs[:, 0] = T0
    T_abs[:, 1:] = T0.unsqueeze(1) @ dT
    R = T_abs[:, :, :3, :3]   # (B, S, 3, 3)
    t = T_abs[:, :, :3, 3:]   # (B, S, 3, 1)
    return R, t

def integral_heatmap_layer_with_variance(heatmaps):
    """
    Soft Argmax with Variance calculation
    Args:
        heatmaps: (B, K, H, W) logits (before softmax)
    Returns:
        coords: (B, K, 2) normalized coordinates [0, 1]
        variance: (B, K) spatial variance (spread of the heatmap)
    """
    B, K, H, W = heatmaps.shape
    heatmaps = heatmaps.reshape(B, K, -1)
    probs = F.softmax(heatmaps, dim=-1) # (B, K, H*W)
    probs = probs.reshape(B, K, H, W)
    
    device = heatmaps.device
    x_range = torch.linspace(0, 1, W, device=device)
    y_range = torch.linspace(0, 1, H, device=device)
    grid_y, grid_x = torch.meshgrid(y_range, x_range, indexing='ij') # (H, W)
    
    coord_x = (probs * grid_x).sum(dim=[-2, -1]) # (B, K)
    coord_y = (probs * grid_y).sum(dim=[-2, -1]) # (B, K)
    coords = torch.stack([coord_x, coord_y], dim=-1) # (B, K, 2)
    
    # var_x = sum(P * (grid_x - coord_x)^2)
    mu_x = coord_x.view(B, K, 1, 1)
    mu_y = coord_y.view(B, K, 1, 1)
    dist_sq_x = (grid_x.view(1, 1, H, W) - mu_x) ** 2
    dist_sq_y = (grid_y.view(1, 1, H, W) - mu_y) ** 2
    var_x = (probs * dist_sq_x).sum(dim=[-2, -1])
    var_y = (probs * dist_sq_y).sum(dim=[-2, -1])
    variance = var_x + var_y
    return coords, variance

# Training loop
class AverageMeter(object):
    """ Computes and stores the average and current value.
    """
    def __init__(self, unit='-'):
        self.reset()
        self.unit = unit

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val    = val
        self.sum   += val * n
        self.count += n
        self.avg    = self.sum / self.count if self.count != 0 else 0

class DummyBar:
    def __init__(self, *args, **kwargs):
        pass
    def __enter__(self):
        return self
    def __exit__(self, *args, **kwargs):
        pass
    def update(self, *args, **kwargs):
        pass
    def set_postfix(self, *args, **kwargs):
        pass

def train_one_epoch(model:MyNet, dataloader, optimizer, scheduler, criterion:PoseLoss, accelerator, epoch, config):
    if accelerator.is_main_process:
        log_path = os.path.join(config.train_log_dir, 'log.csv')
        loss_t_meter = AverageMeter('-')
        loss_R_meter = AverageMeter('-')
        loss_mask_meter = AverageMeter('-')
        loss_add_meter = AverageMeter('-')
        loss_geo_meter = AverageMeter('-')
        loss_pts_meter = AverageMeter('-')
        loss_var_meter = AverageMeter('-')
        loss_pcloud_meter = AverageMeter('-')
        loss_pconf_meter = AverageMeter('-')
        pbar_context = tqdm(total=len(dataloader), desc=f"Epoch {epoch+1}", mininterval=1)
    else:
        pbar_context = DummyBar()

    os.makedirs(config.train_log_dir, exist_ok=True) 
    with pbar_context as pbar:
        for idx, data_dict in enumerate(dataloader):
            images = data_dict['images']
            with accelerator.accumulate(model):
                pred = model(images)
                loss, loss_dict = criterion(pred, data_dict)
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), config.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            if accelerator.is_main_process:
                loss_t_meter.update(loss_dict['loss_t'].item())
                loss_R_meter.update(loss_dict['loss_R'].item())
                loss_mask_meter.update(loss_dict['loss_mask'].item())
                loss_add_meter.update(loss_dict['loss_add'].item())
                loss_geo_meter.update(loss_dict['loss_geo'].item())
                loss_pts_meter.update(loss_dict['loss_pts'].item())
                loss_var_meter.update(loss_dict['loss_var'].item())
                loss_pcloud_meter.update(loss_dict['loss_pcloud'].item())
                loss_pconf_meter.update(loss_dict['loss_pconf'].item())
                pbar.set_postfix({
                    'L_t': f'{loss_t_meter.avg:.4f}',
                    'L_R': f'{loss_R_meter.avg:.4f}',
                    'L_a': f'{loss_add_meter.avg:.4f}',
                    'L_g': f'{loss_geo_meter.avg:.4f}',
                    'L_m': f'{loss_mask_meter.avg:.4f}',
                    'L_p': f'{loss_pts_meter.avg:.4f}',
                    'L_pc': f'{loss_pcloud_meter.avg:.4f}',
                    'L_cf': f'{loss_pconf_meter.avg:.4f}',
                })
                pbar.update(1)

    if accelerator.is_main_process:
        log_data = {
            'loss_t': loss_t_meter.avg,
            'loss_R': loss_R_meter.avg,
            'loss_add': loss_add_meter.avg,
            'loss_geo': loss_geo_meter.avg,
            'loss_mask': loss_mask_meter.avg,
            'loss_pts': loss_pts_meter.avg,
            'loss_var': loss_var_meter.avg,
            'loss_pcloud': loss_pcloud_meter.avg,
            'loss_pconf': loss_pconf_meter.avg,
        }
        df = pd.DataFrame([log_data])
        accelerator.log(log_data, step=epoch)
        if epoch == 0 and config.use_pretrained is False:
            df.to_csv(log_path, mode='w', header=True, index=False)
        else:
            df.to_csv(log_path, mode='a', header=False, index=False)

    return

def build_optimizer(model:MyNet, config):
    """
    Build an optimizer for the model.
    Args:
        model (torch.nn.Module): The model to optimize.
        config (Config): Configuration object containing optimizer parameters.
    Returns:
        torch.optim.Optimizer: The optimizer instance.
    """
    params = [
        {
            "params": model.parameters(),
            "lr": config.learning_rate
        }
    ]
    if config.optimizer == 'adamw':
        optimizer = optim.AdamW(params, weight_decay=config.weight_decay, betas=(config.adam_beta1, config.adam_beta2))
    elif config.optimizer == 'adamw8bit':
        optimizer = bnb.optim.AdamW8bit(params, weight_decay=config.weight_decay, betas=(config.adam_beta1, config.adam_beta2))
    elif config.optimizer == 'adam':
        optimizer = optim.Adam(params, weight_decay=config.weight_decay, betas=(config.adam_beta1, config.adam_beta2))
    elif config.optimizer == 'adam8bit':
        optimizer = bnb.optim.Adam8bit(params, weight_decay=config.weight_decay, betas=(config.adam_beta1, config.adam_beta2))     
    else:
        raise NotImplementedError

    return optimizer

def build_scheduler(optimizer, config, steps_per_epoch=1):
    """
    Build a learning rate scheduler for the optimizer.
    Args:
        optimizer (torch.optim.Optimizer): The optimizer to schedule.
        config (Config): Configuration object containing scheduler parameters.
        steps_per_epoch (int): Number of steps per epoch for the scheduler from len(dataloader).
    Returns:
        torch.optim.lr_scheduler: The learning rate scheduler instance.
    """
    scheduler_specific_kwargs = {}
    if config.scheduler == 'cosine_with_min_lr':
        scheduler_specific_kwargs = {
            'min_lr': config.min_lr_rate
        }
    elif config.scheduler == 'polynomial':
        scheduler_specific_kwargs = {
            'power': config.power
        }
    all_steps = (config.max_epochs * steps_per_epoch) // config.accumulate + 1
    scheduler = get_scheduler(
        config.scheduler,
        optimizer,
        num_warmup_steps=int(all_steps * config.warmup_proportion),
        num_training_steps=all_steps,
        scheduler_specific_kwargs= scheduler_specific_kwargs
    )
    return scheduler

def set_all_seeds(config):
    # Set seeds
    os.environ['PYTHONHASHSEED'] = str(config.seed)
    np.random.seed(config.seed)
    random.seed(config.seed)
    torch.manual_seed(config.seed)
    if config.use_cuda:
        torch.cuda.manual_seed(config.seed)
        torch.backends.cudnn.benchmark     = True
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.enabled       = True
