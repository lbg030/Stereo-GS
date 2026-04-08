#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
from torch import nn
import numpy as np
from utils.graphics_utils import getWorld2View3, getProjectionMatrix, fov2focal
from utils.general_utils import PILtoTorch
import cv2

class Camera(nn.Module):
    def __init__(self, colmap_id, R, T, baseline, FoVx, FoVy, left_image, right_image, stereo_depth, 
                 uid, stereo_disp, tracked_depth, depth_mask = None, l_sky_mask = None, r_sky_mask=None, 
                 W = None, H=None, resolution=None, depth_params = None,
                 trans=np.array([0.0, 0.0, 0.0]), scale=1.0, data_device = "cuda",
                 train_test_exp = False,
                 ):
        super(Camera, self).__init__()

        self.uid = uid
        self.colmap_id = colmap_id
        self.R = R
        self.T = T

        self.baseline = baseline
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.image_name = uid

        self.last_stereo_iter = -1
        self.last_stereo_disp = None

        try:
            self.data_device = torch.device(data_device)
        except Exception as e:
            print(e)
            print(f"[Warning] Custom device {data_device} failed, fallback to default cuda device" )
            self.data_device = torch.device("cuda")

        if left_image is not None:
            self.left_original_image = torch.tensor((left_image / 255.0))
            self.right_original_image = torch.tensor((right_image / 255.0))
            self.image_width = self.left_original_image.shape[2]
            self.image_height = self.left_original_image.shape[1]
        else:
            self.image_width = W
            self.image_height = H

        if stereo_depth is not None:
            self.stereo_depth = torch.tensor(stereo_depth)
            self.stereo_disp = torch.tensor(stereo_disp)
            self.tracked_depth = torch.tensor(tracked_depth)

        if depth_mask is not None:
            self.depth_mask = torch.tensor(depth_mask)
        
        if l_sky_mask is not None and r_sky_mask is not None:
            self.l_sky_mask = torch.tensor(l_sky_mask)
            self.r_sky_mask = torch.tensor(r_sky_mask)

        self.zfar = 100.0
        self.znear = 0.01

        self.trans = trans
        self.scale = scale

        self.world_view_transform = torch.tensor(getWorld2View3(R, T, trans, scale)).transpose(0, 1).cuda()

        self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy).transpose(0,1).cuda()

        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)

        self.camera_center = self.world_view_transform.inverse()[3, :3]
    
    def set_last_stereo_disp(self, disp, iteration):
        self.last_stereo_disp = disp
        self.last_stereo_iter = iteration

    def get_last_stereo_disp(self):
        return self.last_stereo_disp.cuda()
    
    def update_pose(self, new_pose):
        pose = new_pose
        R = pose[:3, :3]
        T = pose[:3, 3]

        self.R = R
        self.T = T

        self.world_view_transform = torch.tensor(getWorld2View3(R, T, self.trans, self.scale)).transpose(0, 1).cuda()

        self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy).transpose(0,1).cuda()

        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)

        self.camera_center = self.world_view_transform.inverse()[3, :3]

        return

    def init_from_gui(uid, T, FoVx, FoVy, fx, fy, cx, cy, H, W):
        R = T[:3, :3]
        t = T[:3, 3]
        return Camera(
            colmap_id=uid,
            R=R,
            T=t,
            baseline = 0.0,
            FoVx=FoVx,
            FoVy=FoVy,
            left_image=None,
            right_image=None,
            stereo_depth=None,
            stereo_disp=None,
            tracked_depth=None,
            uid = uid,
            H=H,
            W=W,
        )
    
    def update_RT(self, R, t):
        self.R = R.to(device=self.data_device)
        self.T = t.to(device=self.data_device)
    
    def update_depth(self, depth):
        self.tracked_depth = torch.tensor(depth).cpu()

    def get_camera_matrix(self):
        focal_x = fov2focal(self.FoVx, self.image_width)  # original focal length
        focal_y = fov2focal(self.FoVy, self.image_height)
        intrinsic_matrix = torch.tensor([[focal_x, 0, self.image_width / 2], [0, focal_y, self.image_height / 2], [0, 0, 1]]).float()
        extrinsic_matrix = self.world_view_transform.transpose(0,1).contiguous() # world2cam
        return intrinsic_matrix.cuda(), extrinsic_matrix.cuda()

    def get_focal(self):
        focal_x = fov2focal(self.FoVx, self.image_width)
        focal_y = fov2focal(self.FoVy, self.image_height)
        return focal_x, focal_y

class MiniCam:
    def __init__(self, width, height, fovy, fovx, znear, zfar, world_view_transform, full_proj_transform):
        self.image_width = width
        self.image_height = height    
        self.FoVy = fovy
        self.FoVx = fovx
        self.znear = znear
        self.zfar = zfar
        self.world_view_transform = world_view_transform
        self.full_proj_transform = full_proj_transform
        view_inv = torch.inverse(self.world_view_transform)
        self.camera_center = view_inv[3][:3]

