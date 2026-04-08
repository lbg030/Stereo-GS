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

import os
import random
import json

import torch
import numpy as np
from scene.dataset_readers import CameraInfo
from utils.camera_utils import loadCam

from utils.system_utils import searchForMaxIteration
from scene.dataset_readers import sceneLoadTypeCallbacks
from scene.gaussian_model import GaussianModel
from arguments import ModelParams
from utils.camera_utils import cameraList_from_camInfos, camera_to_JSON
from lietorch import SE3
from scene.dataset_readers import getNerfppNorm
from scene.cameras import Camera

class Scene:

    gaussians : GaussianModel

    def __init__(self, args : ModelParams, gaussians : GaussianModel, load_iteration=None, shuffle=True, resolution_scales=[1.0], scene_info=None):
        """b
        :param path: Path to colmap scene main folder.
        """
        # self.model_path = args.model_path
        self.loaded_iter = load_iteration
        self.gaussians = gaussians

        # if load_iteration:
        #     if load_iteration == -1:
        #         self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
        #     else:
        #         self.loaded_iter = load_iteration
            # print("Loading trained model at iteration {}".format(self.loaded_iter))

        self.train_cameras = {}

        camlist = []
        if scene_info.train_cameras:
            camlist.extend(scene_info.train_cameras)

        self.FovX = scene_info.train_cameras[0].FovX
        self.FovY = scene_info.train_cameras[0].FovY
        self.width = scene_info.train_cameras[0].width
        self.height = scene_info.train_cameras[0].height
        if shuffle:
            random.shuffle(scene_info.train_cameras)  # Multi-res consistent random shuffling

        self.cameras_extent = scene_info.nerf_normalization["radius"]

        for resolution_scale in resolution_scales:
            self.train_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.train_cameras, args)

        self.gaussians.create_from_pcd(scene_info.point_cloud, scene_info.train_cameras, self.cameras_extent)

    def save(self, iteration):
        point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))

    def nerf_normalization(self):
        cameras = self.getTrainCameras().copy()
        self.cameras_extent = getNerfppNorm(cameras)['radius']
        
    def getTrainCameras(self, scale=1.0):
        return self.train_cameras[scale]

    def getTestCameras(self, scale=1.0):
        return self.test_cameras[scale]
    
    def scene_update(self, args, video, cam_id, l_mask=None, r_mask=None):
        pose = SE3(video.poses[cam_id]).matrix().detach().cpu().numpy()
        # pose = np.linalg.inv(pose)
        R = pose[:3, :3]
        T = pose[:3, 3]

        # left_image = video.images[cam_id, 0].detach().cpu().numpy()
        # right_image = video.images[cam_id, 1].detach().cpu().numpy()
        img_tstamp = int(video.tstamp[cam_id])
        left_image = video.original_images[img_tstamp, 0].detach().cpu().numpy()
        right_image = video.original_images[img_tstamp, 1].detach().cpu().numpy()
        
        stereo_depth = video.stereo_depths[cam_id].detach().cpu().numpy()
        # stereo_disp = video.stereo_disps[cam_id].detach().cpu().numpy()
        tracked_depth = video.disps_up[cam_id].detach().cpu().numpy()

        # if mask is not None:
        #     depth_mask = mask.squeeze().detach().cpu().numpy()
        # else:
        depth_mask = video.depth_masks[cam_id].squeeze().detach().cpu().numpy()
        if l_mask is not None and r_mask is not None:
            l_sky_mask = l_mask
            r_sky_mask = r_mask
        else:
            l_sky_mask = None
            r_sky_mask = None
        
        cam_info = CameraInfo(uid=cam_id, R=R, T=T, baseline = video.baseline, FovY=self.FovY, FovX=self.FovX, depth_params=None,
                              stereo_depth = stereo_depth, stereo_disp=None, tracked_depth=tracked_depth,depth_mask=depth_mask, l_sky_mask=l_sky_mask, r_sky_mask=r_sky_mask,
                              left_image=left_image, right_image = right_image, width=self.width, height=self.height, is_test=False)
        
        self.train_cameras[1.0].append(loadCam(args, cam_info))
        # self.cameras_extent = getNerfppNorm([cam_info])['radius']
        return

    def update_pose(self, new_pose):
        for cam in self.train_cameras[1.0]:
            cam.update_pose(new_pose[cam.uid])
        return
    
    def update_depths(self, new_depth):
        for cam in self.train_cameras[1.0]:
            cam.update_depth(new_depth[cam.uid])
        return
    
    def getShiftedCamera(self, camera, trans_dist):
        intrinsic, extrinsic = camera.get_camera_matrix()
        point = torch.tensor([trans_dist, 0.0, 0.0, 1.0], device="cuda")
        point_world = torch.inverse(extrinsic) @ point
        point_world = point_world[:3]
        camera_center_trans = (point_world - camera.camera_center).cpu().numpy()
        camera = Camera(
            colmap_id=camera.colmap_id,
            R=camera.R,
            T=camera.T,
            FoVx=camera.FoVx,
            FoVy=camera.FoVy,
            left_image=torch.ones_like(camera.left_original_image).detach().cpu().numpy(),
            right_image=torch.ones_like(camera.right_original_image).detach().cpu().numpy(),
            uid=camera.uid,
            trans=camera_center_trans,
            data_device=camera.data_device,
            baseline=camera.baseline,
            stereo_depth=camera.stereo_depth.detach().cpu().numpy(),
            stereo_disp = camera.stereo_disp.detach().cpu().numpy(),
            tracked_depth=camera.tracked_depth.detach().cpu().numpy(),
        )
        return camera