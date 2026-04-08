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

from scene.cameras import Camera
import numpy as np
from utils.graphics_utils import fov2focal
from PIL import Image
import cv2

WARNED = False

def loadCam(args, cam_info):

    left_image = cam_info.left_image
    right_image = cam_info.right_image

    stereo_depth = cam_info.stereo_depth
    stereo_disp = cam_info.stereo_disp
    tracked_depth = cam_info.tracked_depth

    scale = 1.0

    orig_w, orig_h = left_image.shape[2], left_image.shape[1]
    resolution = (int(orig_w / scale), int(orig_h / scale))

    return Camera(resolution = resolution, colmap_id=cam_info.uid, R=cam_info.R, T=cam_info.T, baseline=cam_info.baseline,
                  FoVx=cam_info.FovX, FoVy=cam_info.FovY, depth_params=cam_info.depth_params,
                  left_image=left_image, right_image = right_image, 
                  stereo_depth=stereo_depth, stereo_disp=stereo_disp, tracked_depth=tracked_depth, depth_mask=cam_info.depth_mask,
                    l_sky_mask=cam_info.l_sky_mask, r_sky_mask = cam_info.r_sky_mask,
                  uid=cam_info.uid, data_device=args.data_device,
                  train_test_exp=False)

def cameraList_from_camInfos(cam_infos, args):
    camera_list = []

    for id, c in enumerate(cam_infos):
        camera_list.append(loadCam(args, c))

    return camera_list

def camera_to_JSON(id, camera : Camera):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = camera.R.transpose()
    Rt[:3, 3] = camera.T
    Rt[3, 3] = 1.0

    W2C = np.linalg.inv(Rt)
    pos = W2C[:3, 3]
    rot = W2C[:3, :3]
    serializable_array_2d = [x.tolist() for x in rot]
    camera_entry = {
        'id' : id,
        'img_name' : camera.image_name,
        'width' : camera.width,
        'height' : camera.height,
        'position': pos.tolist(),
        'rotation': serializable_array_2d,
        'fy' : fov2focal(camera.FovY, camera.height),
        'fx' : fov2focal(camera.FovX, camera.width)
    }
    return camera_entry