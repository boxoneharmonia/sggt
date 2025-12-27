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

from .module import *
from .grid_cache import GridCache

class Encoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embed_dim = config.embed_dim
        self.base = AutoModel.from_pretrained(config.base, use_safetensors=True)
        for param in self.base.parameters():
            param.requires_grad = False
        self.k_embed = CamKEmbed(config.embed_dim, config.patch_size)
        self.proj_encoder = nn.Linear(self.base.config.hidden_size, config.embed_dim) if self.base.config.hidden_size != config.embed_dim else nn.Identity()
        encoder_layers = []
        for i in range(config.encoder_depth//2):
            encoder_layers.append(AltRefAttBlock(dim=config.embed_dim, num_heads=config.num_heads, mlp_ratio=config.mlp_ratio, qkv_bias=config.qkv_bias, qk_scale=config.qk_scale,
                                        drop_ratio=config.drop_ratio, attn_drop_ratio=config.attn_drop_ratio, drop_path_ratio=config.drop_path_ratio, groups=config.groups))
        self.encoder = nn.ModuleList(encoder_layers)
        self.cam_token = nn.Parameter(torch.zeros(1, 1, 1, config.embed_dim), requires_grad=True)
        self.num_registers = 4 # from <Vit need registers>
        self.register = nn.Parameter(torch.zeros(1, 1, self.num_registers, config.embed_dim), requires_grad=True)
        nn.init.normal_(self.cam_token, std=1e-4)
        nn.init.normal_(self.register, std=1e-4)
        self.spatial_emb = nn.Parameter(torch.zeros(1, 1, config.num_patches + 1 + self.num_registers, config.embed_dim), requires_grad=True)
        self.feature_idx = set(config.feature_idx)

    def forward(self, x:torch.Tensor, coord:torch.Tensor):
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
        coord = coord.flatten(0,1)
        k_embed = self.k_embed(coord)
        x = x + k_embed
        n = x.shape[1]
        x = rearrange(x, '(b s) n c -> b s n c', b=b)

        cam_tokens = self.cam_token.expand(b, s, -1, -1).contiguous()
        registers = self.register.expand(b, s, -1, -1).contiguous()
        x = torch.cat((x, cam_tokens, registers), dim=2)   # (b s n+1+4 c)
        temporal_pe = self.get_sinusoidal_encoding(s, self.embed_dim, x.device, x.dtype)
        x = x + self.spatial_emb + temporal_pe
        features_list = []
        for i, layer in enumerate(self.encoder):
            x = layer(x)
            if i in self.feature_idx:
                cam_token = x[:,:,n] # (b s c)
                features = x[:,0,0:n] # (b n c)
                features_list.append(features)

        cam_token = x[:,:,n] # (b s c)
        return cam_token, features_list, image_size

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
        self.heads_config = config.heads
        hidden_ratio = config.mlp_ratio // 2
        features_len = len(config.feature_idx)
        if self.heads_config['pose']:
            R_layers = []
            t_layers = []
            for i in range(config.decoder_depth):
                R_layers.append(MHSABlock(dim=config.embed_dim, num_heads=config.num_heads, mlp_ratio=hidden_ratio, qkv_bias=config.qkv_bias, qk_scale=config.qk_scale,
                                            drop_ratio=config.drop_ratio, attn_drop_ratio=config.attn_drop_ratio, drop_path_ratio=config.drop_path_ratio))
                t_layers.append(MHSABlock(dim=config.embed_dim, num_heads=config.num_heads, mlp_ratio=hidden_ratio, qkv_bias=config.qkv_bias, qk_scale=config.qk_scale,
                                            drop_ratio=config.drop_ratio, attn_drop_ratio=config.attn_drop_ratio, drop_path_ratio=config.drop_path_ratio))
            self.R_decoder = nn.Sequential(*R_layers)
            self.t_decoder = nn.Sequential(*t_layers)
            self.R_head = MLP(config.embed_dim, 6) # 6D rotation
            self.t_head = MLP(config.embed_dim, 3)

        if self.heads_config['corner']:   
            self.map_head = DPTHead(inp=config.embed_dim, oup=config.maps, hidden_ratio=hidden_ratio, features=config.dpt_features, patch_size=config.patch_size, use_conf=True)
        
        if self.heads_config['pmap']:
            self.pmap_head = DPTHead(inp=config.embed_dim, oup=3, hidden_ratio=hidden_ratio, features=config.dpt_features, patch_size=config.patch_size, use_conf=True)

    def forward(self, cam_token, features_list, image_size):

        if self.heads_config['pose']:
            R_cam = self.R_decoder(cam_token)
            t_cam = self.t_decoder(cam_token)
            pose_R_6d = self.R_head(R_cam)
            pose_R = rotation_6d_to_matrix(pose_R_6d) # (b s 3 3)
            pose_t = self.t_head(t_cam) # (b s 3)
            pose = torch.cat([pose_R, pose_t[..., None]], dim=-1)
        else:
            pose = None

        if self.heads_config['corner']:
            maps = self.map_head(features_list, image_size)
            mask = maps[:, -1:]
            pvmap = maps[:, :-1]
        else:
            mask = None
            pvmap = None
            
        if self.heads_config['pmap']:
            pmap = self.pmap_head(features_list, image_size)
        else:
            pmap = None

        return pose, mask, pvmap, pmap
    
class MyNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.encoder = Encoder(config)
        self.decoder = Decoder(config)

    def forward(self, x:torch.Tensor, coord:torch.Tensor):
        cam_token, features_list, image_size = self.encoder(x, coord)
        pose, mask, pvmap, pmap = self.decoder(cam_token, features_list, image_size)
        out_dict = {
            'pose': pose,
            'mask': mask,
            'pvmap': pvmap,
            'pmap': pmap
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
        self.pcloud_alpha = config.pcloud_alpha

        self.t_weight = config.t_weight
        self.R_weight = config.R_weight
        self.geo_weight = config.geo_weight
        self.mask_weight = config.mask_weight
        self.pcloud_weight = config.pcloud_weight
        self.pconf_weight = config.pconf_weight

    def forward(self, out_dict, data_dict, eps=1e-6):
        loss = 0.0
        loss_dict = {}
        # 1) pose
        if out_dict['pose'] is not None:
            R_cams = data_dict['R_cams'].flatten(0, 1)
            t_cams = data_dict['t_cams'].flatten(0, 1)
            SE3_pred = out_dict['pose']
            R_preds, t_preds = compute_abs_pose(SE3_pred)
            R_preds = R_preds.flatten(0, 1)
            t_preds = t_preds.flatten(0, 1)
            loss_t = self.lossfn_t(t_preds, t_cams)
            R_errs  = R_preds.transpose(-1, -2) @ R_cams
            traces = R_errs.diagonal(offset=0, dim1=-2, dim2=-1).sum(-1)
            cos_theta = ((traces - 1) / 2.0).clamp(min=-1+eps, max=1-eps)
            theta = torch.acos(cos_theta)
            loss_R = self.lossfn_R(theta, torch.zeros_like(theta))

            loss += self.t_weight * loss_t + self.R_weight * loss_R
            loss_dict['loss_t'] = loss_t
            loss_dict['loss_R'] = loss_R
        else:
            loss_dict['loss_t'] = torch.tensor(0.0)
            loss_dict['loss_R'] = torch.tensor(0.0)

        # 2) mask
        if out_dict['mask'] is not None:
            mask_pred = out_dict['mask']
            mask_gt = data_dict['mask']
            size = mask_pred.shape[-2:]
            mask_gt = F.interpolate(mask_gt, size, mode='nearest-exact')
            loss_mask = self.lossfn_bce_mask(mask_pred, mask_gt)
            loss += self.mask_weight * loss_mask
            loss_dict['loss_mask'] = loss_mask
        else:
            loss_dict['loss_mask'] = torch.tensor(0.0)
        # 3) geo
        if out_dict['pvmap'] is not None:
            cam_K = data_dict['cam_K']
            pts3d_gt = data_dict['pts3d']
            pvmap = out_dict['pvmap']
            H_orig, W_orig = data_dict['images'].shape[-2:]
            scale_tensor = torch.tensor((W_orig-1., H_orig-1.), device=pvmap.device).view(1, 1, 2)
            mask = torch.sigmoid(mask_pred) if loss_mask.item() < 0.1 else mask_gt
            loss_geo = self.voting_loss(pvmap, mask, pts3d_gt, cam_K, scale_tensor)
            loss += self.geo_weight * loss_geo
            loss_dict['loss_geo'] = loss_geo
        else:
            loss_dict['loss_geo'] = torch.tensor(0.0)
        # 4) pclouds
        if out_dict['pmap'] is not None:
            pmap = out_dict['pmap']
            pcloud_gt = data_dict['pcloud']
            pconf_gt = data_dict['pconf']
            pcloud_gt = F.interpolate(pcloud_gt, size, mode='nearest-exact')
            pconf_gt = F.interpolate(pconf_gt, size, mode='nearest-exact')
            pcloud, pconf_logits  = torch.split(pmap, [3,1], dim=1)
            pdist = torch.norm(pcloud - pcloud_gt, p=1, dim=1, keepdim=True)
            pconf = torch.sigmoid(pconf_logits)
            num_pclouds = pconf_gt.sum() + 1e-6
            loss_valid = (pconf * pdist) - self.pcloud_alpha * torch.log(pconf + 1e-6)
            loss_pcloud = (loss_valid * pconf_gt).sum() / num_pclouds
            background = ((1 - mask_gt) * (1 - pconf_gt)) if mask_gt is not None else (1 - pconf_gt)
            loss_invalid = self.lossfn_bce_conf(pconf_logits, torch.zeros_like(pconf_logits))
            loss_pconf = (loss_invalid * background).sum() / background.sum()
            loss += self.pcloud_weight * loss_pcloud + self.pconf_weight * loss_pconf
            loss_dict['loss_pcloud'] = loss_pcloud
            loss_dict['loss_pconf'] = loss_pconf
        else:
            loss_dict['loss_pcloud'] = torch.tensor(0.0)
            loss_dict['loss_pconf'] = torch.tensor(0.0)

        return loss, loss_dict

    def voting_loss(self, pvmap, mask_gt, pts3d_gt, cam_K, scale_tensor):
        B, C, H_feat, W_feat = pvmap.shape
        num_corners = C // 2
        device = pvmap.device

        grid_norm = GridCache.get_mesh_grid(H_feat, W_feat, device=device) # (H_feat, W_feat, 2)
        grid_norm = grid_norm.permute(2, 0, 1)
        corners_offset = pvmap.view(B, num_corners, 2, H_feat, W_feat)
        corners_2d_norm = grid_norm + corners_offset # (B, N_corners, 2, H_feat, W_feat)

        fx = cam_K[:, 0, 0].view(B, 1, 1)
        fy = cam_K[:, 1, 1].view(B, 1, 1)
        cx = cam_K[:, 0, 2].view(B, 1, 1)
        cy = cam_K[:, 1, 2].view(B, 1, 1)

        voter_mask = (mask_gt > 0.5).reshape(B, 1, H_feat * W_feat)
        num_voters_per_image = voter_mask.sum(dim=-1).clamp(min=1.0) # (B, 1)
        scale_tensor = scale_tensor.view(1, 1, 2, 1, 1)
        corners_2d_scaled = corners_2d_norm * scale_tensor
        corners_2d_scaled = corners_2d_scaled.flatten(3)
        u = corners_2d_scaled[:, :, 0, :] # (B, N, HW)
        v = corners_2d_scaled[:, :, 1, :] # (B, N, HW)
        ray_x = (u - cx) / fx
        ray_y = (v - cy) / fy
        ray_z = torch.ones_like(ray_x)

        rays = torch.stack([ray_x, ray_y, ray_z], dim=-1)
        rays_norm = F.normalize(rays, p=2, dim=-1)

        pts3d_gt_expanded = pts3d_gt.unsqueeze(2)
        dot_prod = (pts3d_gt_expanded * rays_norm).sum(dim=-1, keepdim=True) # (B, N, HW, 1)
        proj_point = dot_prod * rays_norm # (B, N, HW, 3)
        dist_3d = torch.norm(pts3d_gt_expanded - proj_point, p=2, dim=-1) # (B, N, HW)
        loss_per_corner = (dist_3d * voter_mask).sum(dim=-1) # (B, N)
        loss_per_corner = loss_per_corner / num_voters_per_image
        return loss_per_corner.mean()     

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
        loss_geo_meter = AverageMeter('-')
        loss_pcloud_meter = AverageMeter('-')
        loss_pconf_meter = AverageMeter('-')
        pbar_context = tqdm(total=len(dataloader), desc=f"Epoch {epoch+1}", mininterval=1)
    else:
        pbar_context = DummyBar()

    os.makedirs(config.train_log_dir, exist_ok=True) 
    with pbar_context as pbar:
        for idx, data_dict in enumerate(dataloader):
            images = data_dict['images']
            coords = data_dict['coords']
            with accelerator.accumulate(model):
                pred = model(images, coords)
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
                loss_geo_meter.update(loss_dict['loss_geo'].item())
                loss_pcloud_meter.update(loss_dict['loss_pcloud'].item())
                loss_pconf_meter.update(loss_dict['loss_pconf'].item())
                pbar.set_postfix({
                    'L_t': f'{loss_t_meter.avg:.4f}',
                    'L_R': f'{loss_R_meter.avg:.4f}',
                    'L_g': f'{loss_geo_meter.avg:.4f}',
                    'L_m': f'{loss_mask_meter.avg:.4f}',
                    'L_pc': f'{loss_pcloud_meter.avg:.4f}',
                    'L_cf': f'{loss_pconf_meter.avg:.4f}',
                })
                pbar.update(1)

    if accelerator.is_main_process:
        log_data = {
            'loss_t': loss_t_meter.avg,
            'loss_R': loss_R_meter.avg,
            'loss_geo': loss_geo_meter.avg,
            'loss_mask': loss_mask_meter.avg,
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
    r_params = []
    other_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        
        if 'decoder.R_head' in name or 'decoder.R_decoder' in name:
            r_params.append(param)
        else:
            other_params.append(param)
    params = [
        {
            "params": other_params,
            "lr": config.learning_rate
        },
        {
            "params": r_params,
            "lr": config.learning_rate * 1.0,
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
