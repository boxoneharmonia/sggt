import random
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as T
from PIL import Image
import os
import json
import kornia.geometry.liegroup

class MyDataset(Dataset):
    def __init__(self, config, transform, is_train=True):
        super().__init__()
        self.root_dir = config.root
        if is_train:
            jsonfile = os.path.join(self.root_dir, config.training)
        else:
            jsonfile = os.path.join(self.root_dir, config.testing)

        ptsfile = os.path.join(self.root_dir, config.ptsfile)
        self.json = json.load(open(jsonfile, 'r'))
        self.pts3d = json.load(open(ptsfile, 'r'))[0] # shape (n 3)
        self.size = config.original_size
        self.sequence_length = config.seq_len
        self.transform = transform 
        self.is_train = is_train
        self.index_map = []
        for seq_key in self.json.keys():
            n_total_frames = len(self.json[seq_key]['se3'])
            if n_total_frames < self.sequence_length:
                continue
            for i in range(n_total_frames - self.sequence_length + 1):
                start_idx = self.sequence_length + i
                self.index_map.append((seq_key, start_idx))
        self.scale = config.scale
    
    def get_one_img(self, seq_name, frame_idx):
        sequence_data = self.json[seq_name]
        imgname = frame_idx.zfill(6) + '.jpg'
        mskname = frame_idx.zfill(6) + '_000000.png'
        imgpath = os.path.join(self.root_dir, seq_name, 'rgb', imgname)
        mskpath = os.path.join(self.root_dir, seq_name, 'mask_visib', mskname)
        image   = Image.open(imgpath).convert('RGB')
        mask    = Image.open(mskpath)

        se3_m2c = np.array(sequence_data['se3'][frame_idx], dtype=np.float32).reshape(6)
        cam_K   = np.array(sequence_data['cam'][frame_idx]['cam_K'], dtype=np.float32).reshape(3,3)
        
        image = T.to_tensor(image)  
        mask  = T.to_tensor(mask)
        _, orig_h, orig_w = image.shape

        se3_group = kornia.geometry.liegroup.Se3.exp(torch.from_numpy(se3_m2c).float())
        SE3_matrix = se3_group.matrix().detach()
        R_cam = SE3_matrix[:3,:3]
        t_cam = SE3_matrix[:3,3:]

        target_h, target_w = self.size
        scale_x = target_w / orig_w
        scale_y = target_h / orig_h

        image = T.resize(image, self.size, interpolation=T.InterpolationMode.BILINEAR)
        if self.is_train and self.transform is not None:
            image = self.transform(image)
        mask = T.resize(mask, self.size, interpolation=T.InterpolationMode.NEAREST_EXACT)
        mask = (mask > 0).float()
    
        pts3d_orig_np = np.array(self.pts3d)
        pts3d_sampled_np = sample_points_on_box(pts3d_orig_np, total_points=10000)
        pts3d_all_np = np.concatenate([pts3d_orig_np, pts3d_sampled_np], axis=0)
        pts3d_all = torch.from_numpy(pts3d_all_np).float() / 1000.0

        cam_K[0, :] *= scale_x
        cam_K[1, :] *= scale_y
        K_tensor = torch.from_numpy(cam_K).float()
        pts_cam_all = pts3d_all @ R_cam.t() + t_cam.view(1, 3)
        pts_proj_all = pts_cam_all @ K_tensor.t()
        pts2d_orig_all = pts_proj_all[:, :2] / (pts_proj_all[:, 2:3] + 1e-5)
        pts2d_all = torch.zeros_like(pts2d_orig_all)
        pts2d_all[:, 0] = pts2d_orig_all[:, 0]
        pts2d_all[:, 1] = pts2d_orig_all[:, 1]
        pts2d_corner = pts2d_all[:8]

        is_in_w = (pts2d_corner[:, 0] >= 0) & (pts2d_corner[:, 0] < target_w)
        is_in_h = (pts2d_corner[:, 1] >= 0) & (pts2d_corner[:, 1] < target_h)
        pts_vis = (is_in_w & is_in_h).float()

        point_cloud = torch.zeros((3, target_h, target_w), dtype=torch.float32)
        point_conf = torch.zeros((1, target_h, target_w), dtype=torch.float32)
        u_coords = torch.round(pts2d_all[:, 0]).int()
        v_coords = torch.round(pts2d_all[:, 1]).int()
        depths = pts_cam_all[:, 2] # Z-depth

        valid_indices = (u_coords >= 0) & (u_coords < target_w) & \
                        (v_coords >= 0) & (v_coords < target_h) & \
                        (depths > 0)
        if valid_indices.any():
            u_valid = u_coords[valid_indices]
            v_valid = v_coords[valid_indices]
            p_valid = pts_cam_all[valid_indices] # (M, 3) XYZ in Cam Frame
            d_valid = depths[valid_indices]
            sort_idx = torch.argsort(d_valid, descending=True)
            u_sorted = u_valid[sort_idx]
            v_sorted = v_valid[sort_idx]
            p_sorted = p_valid[sort_idx]
            point_hwc = torch.zeros((target_h, target_w, 3), dtype=torch.float32)
            point_hwc[v_sorted, u_sorted] = p_sorted
            point_cloud = point_hwc.permute(2, 0, 1).contiguous()
            point_conf[:, v_sorted, u_sorted] = 1.0

        return image, mask, R_cam, t_cam, pts2d_corner, pts_vis, point_cloud, point_conf, K_tensor
    
    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, idx):
        seq_name, first_idx = self.index_map[idx]
        selected_indices = [first_idx] + [i for i in range(first_idx-self.sequence_length+1, first_idx)][::-1]
        images, masks, R_cams, t_cams, pts2d, pts_vis, pclouds, pconfs, cam_Ks = [], [], [], [], [], [], [], [], []
        for current_idx in selected_indices:
            image, mask, R_cam, t_cam, pts, vis, pcloud, pconf, cam_K = self.get_one_img(seq_name, str(current_idx))
            images.append(image)
            masks.append(mask)
            R_cams.append(R_cam)
            t_cams.append(t_cam)
            pts2d.append(pts)
            pts_vis.append(vis)
            pclouds.append(pcloud)
            pconfs.append(pconf)
            cam_Ks.append(cam_K)

        ret_dict = {
            'images': torch.stack(images, dim=0),
            'masks': torch.stack(masks, dim=0),
            'R_cams': torch.stack(R_cams, dim=0),
            't_cams': torch.stack(t_cams, dim=0),
            'pts2d': torch.stack(pts2d, dim=0),
            'pts_vis': torch.stack(pts_vis, dim=0),
            'pclouds': torch.stack(pclouds, dim=0),
            'pconfs': torch.stack(pconfs, dim=0),
            'cam_Ks': torch.stack(cam_Ks, dim=0),
        }
        return ret_dict

def build_dataloader(config, is_train=True):
    """ Build a DataLoader for the EventSequenceDataset. """
    transform = None if is_train == False else build_transform() 
    dataset = MyDataset(config, transform, is_train)
    if is_train:
        dataloader = DataLoader(
            dataset, batch_size=config.batch_size,
            shuffle=True,
            num_workers=config.num_workers, 
            pin_memory=True,
            drop_last=True)
    else:
        dataloader = DataLoader(
            dataset, batch_size=1,
            shuffle=False,
            num_workers=1, 
            pin_memory=True,
            drop_last=True)
    return dataloader

class Normalize(object):
    def __call__(self, image, *args, **kwargs):
        image = T.normalize(image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        return image
    
class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, *args, **kwargs):
        for t in self.transforms:
            image = t(image, *args, **kwargs)
        return image
    
class RandomApply(object):
    def __init__(self, transform, p=0.5):
        self.transform = transform
        self.p = p

    def __call__(self, x):
        if np.random.rand() < self.p:
            return self.transform(x)
        return x
   
class GaussianNoise(object):
    """ Add random Gaussian white noise

        image: torch.Tensor image (0 ~ 1)
        std:   noise standard deviation (0 ~ 255)
    """
    def __init__(self, std=25):
        self.std = std/255

    def __call__(self, image, *args, **kwargs):
        noise = torch.randn(image.shape, dtype=torch.float32) * self.std
        image = torch.clamp(image + noise, 0, 1)
        return image
    
class BrightnessContrast(object):
    """ Adjust brightness and contrast of the image in a fashion of
        OpenCV's convertScaleAbs, where

        newImage = alpha * image + beta

        image: torch.Tensor image (0 ~ 1)
        alpha: multiplicative factor
        beta:  additive factor (0 ~ 255)
    """
    def __init__(self, alpha=(0.5, 2.0), beta=(-25, 25)):
        self.alpha = torch.tensor(alpha).log()
        self.beta  = torch.tensor(beta)/255

    def __call__(self, image, *args, **kwargs):
        loga = torch.rand(1) * (self.alpha[1] - self.alpha[0]) + self.alpha[0]
        a = loga.exp()
        b = torch.rand(1) * (self.beta[1]  - self.beta[0])  + self.beta[0]
        image = torch.clamp(a*image + b, 0, 1)
        return image

class RandomPixelDropout(object):
    """
    Randomly sets a certain percentage of pixels to 0.
    
    image: torch.Tensor image (C, H, W) with values between 0 ~ 1.
    dropout_ratio: The percentage of pixels to be set to 0 (e.g., 0.1 for 10%).
    """
    def __init__(self, dropout_ratio=0.2):
        assert 0 <= dropout_ratio <= 1, "dropout_ratio must be between 0 and 1."
        self.dropout_ratio = dropout_ratio

    def __call__(self, image, *args, **kwargs):
        if self.dropout_ratio == 0:
            return image
        
        mask = torch.rand(image.shape[1:], device=image.device) < self.dropout_ratio
        image = image.masked_fill(mask.unsqueeze(0), 0)
        return image

class GaussianBlur(object):
    """
    Apply Gaussian blur to an image.
    
    image: torch.Tensor image (C, H, W) with values between 0 ~ 1.
    kernel_size: Size of the Gaussian kernel.
    sigma: Standard deviation of the Gaussian kernel. Can be a range (min, max).
    """
    def __init__(self, kernel_size=5, sigma=(0.1, 2.0)):
        self.kernel_size = [kernel_size, kernel_size]
        self.sigma = sigma

    def __call__(self, image, *args, **kwargs):
        # Sample sigma from the given range
        sigma = [float(np.random.uniform(self.sigma[0], self.sigma[1]))]
        return T.gaussian_blur(image, kernel_size=self.kernel_size, sigma=sigma)

def build_transform():
    transforms_list = []
    transforms_list.append(RandomApply(BrightnessContrast()))
    transforms_list.append(RandomApply(GaussianNoise()))
    transforms_list.append(RandomApply(GaussianBlur()))
    transforms_list.append(RandomApply(RandomPixelDropout()))

    return Compose(transforms_list)

def sample_points_on_box(bbox_corners, total_points=1500):
    """
    bbox_corners: (8, 3) array of box vertices
    total_points: desired number of sampled points (1000~2000 recommended)
    return: (N, 3) array of sampled surface points
    """

    xs = bbox_corners[:, 0]
    ys = bbox_corners[:, 1]
    zs = bbox_corners[:, 2]
    xmin, xmax = xs.min(), xs.max()
    ymin, ymax = ys.min(), ys.max()
    zmin, zmax = zs.min(), zs.max()

    faces = [
        ("xy_low",  xmin, xmax, ymin, ymax, zmin),  # z = zmin
        ("xy_high", xmin, xmax, ymin, ymax, zmax),  # z = zmax
        
        ("xz_low",  xmin, xmax, zmin, zmax, ymin),  # y = ymin
        ("xz_high", xmin, xmax, zmin, zmax, ymax),  # y = ymax
        
        ("yz_low",  ymin, ymax, zmin, zmax, xmin),  # x = xmin
        ("yz_high", ymin, ymax, zmin, zmax, xmax),  # x = xmax
    ]

    areas = np.array([
        (xmax - xmin) * (ymax - ymin),   # xy
        (xmax - xmin) * (ymax - ymin),
        
        (xmax - xmin) * (zmax - zmin),   # xz
        (xmax - xmin) * (zmax - zmin),
        
        (ymax - ymin) * (zmax - zmin),   # yz
        (ymax - ymin) * (zmax - zmin),
    ])
    
    points_per_face = (areas / areas.sum() * total_points).astype(int)
    sampled_points = []
    for (face, a0, a1, b0, b1, const), n_pts in zip(faces, points_per_face):
        if n_pts <= 0:
            continue
        
        us = np.random.uniform(a0, a1, n_pts)
        vs = np.random.uniform(b0, b1, n_pts)

        if face.startswith("xy"):
            zs = np.full(n_pts, const)
            pts = np.stack([us, vs, zs], axis=1)
        elif face.startswith("xz"):
            ys = np.full(n_pts, const)
            pts = np.stack([us, ys, vs], axis=1)
        else:  # yz
            xs = np.full(n_pts, const)
            pts = np.stack([xs, us, vs], axis=1)

        sampled_points.append(pts)

    return np.concatenate(sampled_points, axis=0)