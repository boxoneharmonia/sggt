import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as T
from PIL import Image
import os
import json
import cv2
import random

class MyDataset(Dataset):
    def __init__(self, config, transform, is_train=True):
        super().__init__()
        self.root_dir = config.root
        if is_train:
            jsondir = os.path.join(self.root_dir, config.training)
        else:
            jsondir = os.path.join(self.root_dir, config.testing)

        ptsfile = os.path.join(self.root_dir, config.ptsfile)
        self.pts3d = np.array(json.load(open(ptsfile, 'r'))[0])*0.95  # shape (n 3)
        self.size = config.original_size
        self.sequence_length = config.seq_len
        self.transform = transform 
        self.is_train = is_train
        self.scale_limit = 0.1
        self.rotate_limit = 180
        self.data = {}
        self.index_map = []
        json_files = sorted(os.listdir(jsondir))
        for json_file in json_files:
            jsons = json.load(open(os.path.join(jsondir, json_file), 'r'))
            self.data.update(jsons)
            for seq_key in jsons.keys():
                n_total_frames = len(jsons[seq_key]['SE3'])
                if n_total_frames < self.sequence_length:
                    continue
                for i in range(n_total_frames): # the filename idx starts from 000001
                    self.index_map.append((seq_key, i+1))

    def get_one_img(self, seq_name, frame_idx, M, M_k, M_r, scale, transform_params=None, only_pose=True):
        sequence_data = self.data[seq_name]
        target_h, target_w = self.size

        imgname = frame_idx.zfill(6) + '.jpg'
        imgpath = os.path.join(self.root_dir, seq_name, 'rgb', imgname)
        image = np.array(Image.open(imgpath).convert('RGB'))
        orig_h, orig_w, _ = image.shape
        image = cv2.warpAffine(image, M[:2], (orig_w, orig_h), flags=cv2.INTER_LANCZOS4, borderValue=(128, 128, 128))
        image = T.to_tensor(image)  
        if self.is_train and self.transform is not None:
            image = self.transform(image, transform_params)

        SE3_matrix = torch.tensor(sequence_data['SE3'][frame_idx], dtype=torch.float32).reshape(4,4)
        R_cam = SE3_matrix[:3, :3]
        t_cam = SE3_matrix[:3, 3:]
        R_cam = torch.from_numpy(M_r) @ R_cam
        t_cam = torch.from_numpy(M_r) @ t_cam
        t_cam[-1] = t_cam[-1] / scale

        mskname = frame_idx.zfill(6) + '_000000.png'
        mskpath = os.path.join(self.root_dir, seq_name, 'mask_visib', mskname)
        mask = np.array(Image.open(mskpath))
        mask = cv2.warpAffine(mask, M[:2], (orig_w, orig_h), flags=cv2.INTER_AREA)
        mask  = T.to_tensor(mask)
        
        y_coords, x_coords = torch.where(mask.squeeze(0) > 0)
        ymin = int(y_coords.min() * 0.95)
        ymax = int(y_coords.max() * 1.05)
        xmin = int(x_coords.min() * 0.95)
        xmax = int(x_coords.max() * 1.05)
        width = xmax - xmin
        height = ymax - ymin
        center_x = (xmin + xmax) / 2.0
        center_y = (ymin + ymax) / 2.0
        if self.is_train:
            center_x = center_x + random.uniform(-0.01, 0.01) * width
            center_y = center_y + random.uniform(-0.01, 0.01) * height
            width = random.uniform(0.9, 1.0) * width
            height = random.uniform(0.9, 1.0) * height
        crop_size = int(max(width, height))
        xmin = int(center_x - crop_size / 2.0)
        ymin = int(center_y - crop_size / 2.0)
        xmin = max(0, min(xmin, orig_w - crop_size))
        ymin = max(0, min(ymin, orig_h - crop_size))
        xmax = xmin + crop_size
        ymax = ymin + crop_size
        # bbox = torch.tensor([xmin/orig_w, ymin/orig_h, xmax/orig_w, ymax/orig_h]).float() 
        image = T.resized_crop(image, ymin, xmin, crop_size, crop_size, self.size, interpolation=T.InterpolationMode.BILINEAR)
        mask = T.resized_crop(mask, ymin, xmin, crop_size, crop_size, self.size, interpolation=T.InterpolationMode.BILINEAR)
        mask = (mask > 0.0).float()

        cam_K = torch.tensor(sequence_data['cam'][frame_idx], dtype=torch.float32).reshape(3,3)
        cam_K = torch.from_numpy(M_k).float() @ cam_K
        scale_x = target_w / crop_size
        scale_y = target_h / crop_size
        K_crop = torch.tensor([
            [scale_x, 0,          -xmin * scale_x],
            [0,          scale_y, -ymin * scale_y],
            [0,          0,          1]
        ], dtype=torch.float32)
        cam_K = K_crop @ cam_K

        coord_map = generate_coord_map(target_h, target_w, cam_K)

        if only_pose:
            return (image, coord_map, R_cam, t_cam)
        else:
            pts3d_orig_np = self.pts3d.copy()
            pts3d_sampled_np = sample_points_on_box(pts3d_orig_np, total_points=10000)
            pts3d_all_np = np.concatenate([pts3d_orig_np, pts3d_sampled_np], axis=0)
            pts3d_all = torch.from_numpy(pts3d_all_np).float() / 1000.0

            K_tensor = cam_K
            pts_cam_all = pts3d_all @ R_cam.t() + t_cam.view(1, 3)
            pts3d_corner = pts_cam_all[:8]
            pts_proj_all = pts_cam_all @ K_tensor.t()
            pts2d_orig_all = pts_proj_all[:, :2] / (pts_proj_all[:, 2:3] + 1e-5)
            pts2d_all = torch.zeros_like(pts2d_orig_all)
            pts2d_all[:, 0] = pts2d_orig_all[:, 0]
            pts2d_all[:, 1] = pts2d_orig_all[:, 1]
            pts2d_corner = pts2d_all[:8]

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

            return image, coord_map, mask, R_cam, t_cam, pts2d_corner, pts3d_corner, point_cloud, point_conf, K_tensor
    
    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, idx):
        seq_name, first_idx = self.index_map[idx]
        sample_window = self.sequence_length + 2
        if first_idx < sample_window:
            sampled = [i for i in range(first_idx + 1, first_idx + sample_window)]
            selected_indices = [first_idx] + sorted(np.random.choice(sampled, self.sequence_length - 1, replace=False).tolist())
        else:
            sampled = [i for i in range(first_idx - sample_window + 1, first_idx)]
            selected_indices = [first_idx] + sorted(np.random.choice(sampled, self.sequence_length - 1, replace=False).tolist())[::-1]
        
        tmp_img_path = os.path.join(self.root_dir, seq_name, 'rgb', str(first_idx).zfill(6) + '.jpg')
        with Image.open(tmp_img_path) as tmp_img:
            orig_w, orig_h = tmp_img.size
        target_h, target_w = self.size
        M_resize = get_resize_matrix(orig_h, orig_w, target_h, target_w)
        M_k = M_resize

        if self.is_train:
            M_aug, M_r, scale = get_center_aug_params(
                self.scale_limit, self.rotate_limit, target_w, target_h
            )
            M_total = np.matmul(M_aug, M_resize)
        else:
            M_total = M_resize
            M_r = np.eye(3, dtype=np.float32)
            scale = 1.0

        # Generate consistent augmentation params for the sequence
        transform_params = None
        if self.is_train and self.transform is not None:
            transform_params = self.transform.get_params()

        images, coords, R_cams, t_cams = [], [], [], []
        for current_idx in selected_indices:
            if current_idx == selected_indices[0]:
                image, coord_map, mask, R_cam, t_cam, pts2d_corner, pts3d_corner, pcloud, pconf, cam_K = self.get_one_img(seq_name, str(current_idx), M_total, M_k, M_r, scale, transform_params=transform_params, only_pose=False)
                images.append(image)
                coords.append(coord_map)
                R_cams.append(R_cam)
                t_cams.append(t_cam)
            else:
                image, coord_map, R_cam, t_cam = self.get_one_img(seq_name, str(current_idx), M_total, M_k, M_r, scale, transform_params=transform_params)
                images.append(image)
                coords.append(coord_map)
                R_cams.append(R_cam)
                t_cams.append(t_cam)

        ret_dict = {
            'images': torch.stack(images, dim=0),
            'coords': torch.stack(coords, dim=0),
            'R_cams': torch.stack(R_cams, dim=0),
            't_cams': torch.stack(t_cams, dim=0),
            'mask': mask,
            'pts2d': pts2d_corner,
            'pts3d': pts3d_corner,
            'pcloud': pcloud,
            'pconf': pconf,
            'cam_K': cam_K
        }
        return ret_dict

def get_resize_matrix(h, w, target_h, target_w):
    scale_x = target_w / w
    scale_y = target_h / h
    return np.array([
        [scale_x, 0, 0],
        [0, scale_y, 0],
        [0, 0, 1]
    ], dtype=np.float32)

def get_center_aug_params(scale_limit, rotate_limit, target_w, target_h):
    sfactor = np.random.uniform(1, 1 + scale_limit)
    ang_deg = np.random.uniform(-rotate_limit, rotate_limit)
    ang_rad = np.deg2rad(ang_deg)
    cx = target_w / 2.0
    cy = target_h / 2.0
    M_aug_2x3 = cv2.getRotationMatrix2D((cx, cy), ang_deg, sfactor)
    M_aug_3x3 = np.concatenate((M_aug_2x3, [[0, 0, 1]]), axis=0).astype(np.float32)
    cos_a = np.cos(ang_rad)
    sin_a = np.sin(ang_rad)
    M_rot = np.array([
        [cos_a, sin_a, 0],
        [-sin_a, cos_a, 0],
        [0,     0,     1]
    ], dtype=np.float32)
    return M_aug_3x3, M_rot, sfactor

def generate_coord_map(height, width, K):
    xs, ys = torch.meshgrid(
        torch.arange(width, dtype=torch.float32), 
        torch.arange(height, dtype=torch.float32), 
        indexing='xy'
    )
    fx = K[0, 0]
    fy = K[1, 1]
    cx = K[0, 2]
    cy = K[1, 2]
    coord_x = (xs - cx) / fx
    coord_y = (ys - cy) / fy
    coord_map = torch.stack([coord_x, coord_y], dim=0)
    return coord_map

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
        image = T.normalize(image, mean=[0.16, 0.16, 0.16], std=[0.31, 0.31, 0.31])
        return image
    
class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def get_params(self):
        params = {}
        for t in self.transforms:
            # We use the class name as the key.
            params[type(t).__name__] = t.get_params()
        return params

    def __call__(self, image, params=None):
        for t in self.transforms:
            # If params is provided, look up the params for this transform type.
            # If not provided, pass None.
            t_params = None
            if params is not None:
                t_params = params.get(type(t).__name__)

            image = t(image, t_params)
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
    def __init__(self, std=25, p=0.5):
        self.std = std/255
        self.p = p

    def get_params(self):
        if np.random.rand() > self.p:
            return {'apply': False}
        return {'apply': True}

    def __call__(self, image, params=None):
        # Determine if we should apply
        apply = False
        if params is not None:
            # Use provided params
            apply = params.get('apply', False)
        else:
            # Randomly decide if params not provided
            apply = (np.random.rand() <= self.p)

        if not apply:
            return image

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
    def __init__(self, alpha=(0.5, 2.0), beta=(-10, 10), p=0.5):
        self.alpha_range = torch.tensor(alpha).log()
        self.beta_range  = torch.tensor(beta)/255
        self.p = p

    def get_params(self):
        if np.random.rand() > self.p:
            return {'apply': False}

        loga = torch.rand(1) * (self.alpha_range[1] - self.alpha_range[0]) + self.alpha_range[0]
        a = loga.exp().item()
        b = (torch.rand(1) * (self.beta_range[1]  - self.beta_range[0])  + self.beta_range[0]).item()
        return {'apply': True, 'alpha': a, 'beta': b}

    def __call__(self, image, params=None):
        a = 1.0
        b = 0.0

        if params is not None:
             if not params.get('apply', True):
                 return image
             a = params.get('alpha', 1.0)
             b = params.get('beta', 0.0)
        else:
             # If params is NOT passed, use legacy random behavior
             if np.random.rand() <= self.p:
                loga = torch.rand(1) * (self.alpha_range[1] - self.alpha_range[0]) + self.alpha_range[0]
                a = loga.exp().item()
                b = (torch.rand(1) * (self.beta_range[1]  - self.beta_range[0])  + self.beta_range[0]).item()
             else:
                return image

        image = torch.clamp(a*image + b, 0, 1)
        return image

class RandomPixelDropout(object):
    """
    Randomly sets a certain percentage of pixels to 0.
    
    image: torch.Tensor image (C, H, W) with values between 0 ~ 1.
    dropout_ratio: The percentage of pixels to be set to 0 (e.g., 0.1 for 10%).
    """
    def __init__(self, dropout_ratio=0.25, p=0.5):
        assert 0 <= dropout_ratio <= 1, "dropout_ratio must be between 0 and 1."
        self.dropout_ratio = dropout_ratio
        self.p = p

    def get_params(self):
        if np.random.rand() > self.p:
            return {'apply': False}
        return {'apply': True}

    def __call__(self, image, params=None):
        if self.dropout_ratio == 0:
            return image

        apply = False
        if params is not None:
            apply = params.get('apply', False)
        else:
            apply = (np.random.rand() <= self.p)

        if not apply:
            return image
        
        mask = torch.rand(image.shape[1:], device=image.device) < self.dropout_ratio
        image = image.masked_fill(mask.unsqueeze(0), 0.5)
        return image

class GaussianBlur(object):
    """
    Apply Gaussian blur to an image.
    
    image: torch.Tensor image (C, H, W) with values between 0 ~ 1.
    kernel_size: Size of the Gaussian kernel.
    sigma: Standard deviation of the Gaussian kernel. Can be a range (min, max).
    """
    def __init__(self, kernel_size=5, sigma=(0.1, 2.0), p=0.5):
        self.kernel_size = [kernel_size, kernel_size]
        self.sigma = sigma
        self.p = p

    def get_params(self):
        if np.random.rand() > self.p:
            return {'apply': False}
        sigma = float(np.random.uniform(self.sigma[0], self.sigma[1]))
        return {'apply': True, 'sigma': sigma}

    def __call__(self, image, params=None):
        sigma_val = None

        if params is not None:
            if not params.get('apply', True):
                return image
            sigma_val = params.get('sigma')
        else:
             if np.random.rand() <= self.p:
                 sigma_val = float(np.random.uniform(self.sigma[0], self.sigma[1]))

        if sigma_val is None:
            return image

        # T.gaussian_blur expects sigma as list
        return T.gaussian_blur(image, kernel_size=self.kernel_size, sigma=[sigma_val])

def build_transform():
    transforms_list = []
    
    transforms_list.append(BrightnessContrast(p=0.5))
    # transforms_list.append(GaussianNoise(p=0.5))
    transforms_list.append(GaussianBlur(p=0.5))
    transforms_list.append(RandomPixelDropout(p=0.5))
    # transforms_list.append(Normalize())

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