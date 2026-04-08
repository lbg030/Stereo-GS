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
import sys
from PIL import Image
from typing import NamedTuple
from scene.colmap_loader import read_extrinsics_text, read_intrinsics_text, qvec2rotmat, \
    read_extrinsics_binary, read_intrinsics_binary, read_points3D_binary, read_points3D_text
from utils.graphics_utils import getWorld2View3, focal2fov, fov2focal
from lietorch import SE3

import torch
import numpy as np
import json
from scene import depth_fusion
from pathlib import Path
from plyfile import PlyData, PlyElement
from utils.sh_utils import SH2RGB
from scene.gaussian_model import BasicPointCloud
import torch.nn.functional as F

from visualization_utils import *


class CameraInfo(NamedTuple):
    uid: int
    R: np.array
    T: np.array
    baseline: float
    FovY: np.array
    FovX: np.array
    depth_params: dict
    left_image: np.array
    right_image: np.array
    stereo_depth: np.array
    stereo_disp : np.array
    tracked_depth: np.array
    depth_mask: np.array
    l_sky_mask: np.array
    r_sky_mask: np.array
    width: int
    height: int
    is_test: bool

class SceneInfo(NamedTuple):
    point_cloud: BasicPointCloud
    train_cameras: list
    test_cameras: list
    nerf_normalization: dict
    ply_path: str
    is_nerf_synthetic: bool

def getNerfppNorm(cam_info):
    def get_center_and_diag(cam_centers):
        cam_centers = np.hstack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.max(dist)
        return center.flatten(), diagonal

    cam_centers = []

    for cam in cam_info:
        W2C = getWorld2View3(cam.R, cam.T)
        C2W = np.linalg.inv(W2C)
        cam_centers.append(C2W[:3, 3:4])

    center, diagonal = get_center_and_diag(cam_centers)
    radius = diagonal * 1.1

    translate = -center

    return {"translate": translate, "radius": radius}

def readColmapCameras(cam_extrinsics, cam_intrinsics, depths_params, images_folder, depths_folder, test_cam_names_list):
    cam_infos = []
    for idx, key in enumerate(cam_extrinsics):
        sys.stdout.write('\r')
        # the exact output you're looking for:
        sys.stdout.write("Reading camera {}/{}".format(idx+1, len(cam_extrinsics)))
        sys.stdout.flush()

        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        height = intr.height
        width = intr.width

        uid = intr.id
        R = np.transpose(qvec2rotmat(extr.qvec))
        T = np.array(extr.tvec)

        if intr.model=="SIMPLE_PINHOLE":
            focal_length_x = intr.params[0]
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model=="PINHOLE":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
        else:
            assert False, "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"

        n_remove = len(extr.name.split('.')[-1]) + 1
        depth_params = None
        if depths_params is not None:
            try:
                depth_params = depths_params[extr.name[:-n_remove]]
            except:
                print("\n", key, "not found in depths_params")

        image_path = os.path.join(images_folder, extr.name)
        image_name = extr.name
        depth_path = os.path.join(depths_folder, f"{extr.name[:-n_remove]}.png") if depths_folder != "" else ""

        cam_info = CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, depth_params=depth_params,
                              image_path=image_path, image_name=image_name, depth_path=depth_path,
                              width=width, height=height, is_test=image_name in test_cam_names_list)
        cam_infos.append(cam_info)

    sys.stdout.write('\n')
    return cam_infos

def fetchPly(path):
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
    normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
    return BasicPointCloud(points=positions, colors=colors, normals=normals)

def storePly(path, xyz, rgb):
    # Define the dtype for the structured array
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    
    normals = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)

def readColmapSceneInfo(path, images, depths, eval, train_test_exp, llffhold=8):
    try:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.bin")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.bin")
        cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
    except:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.txt")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.txt")
        cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)

    depth_params_file = os.path.join(path, "sparse/0", "depth_params.json")
    ## if depth_params_file isnt there AND depths file is here -> throw error
    depths_params = None
    if depths != "":
        try:
            with open(depth_params_file, "r") as f:
                depths_params = json.load(f)
            all_scales = np.array([depths_params[key]["scale"] for key in depths_params])
            if (all_scales > 0).sum():
                med_scale = np.median(all_scales[all_scales > 0])
            else:
                med_scale = 0
            for key in depths_params:
                depths_params[key]["med_scale"] = med_scale

        except FileNotFoundError:
            print(f"Error: depth_params.json file not found at path '{depth_params_file}'.")
            sys.exit(1)
        except Exception as e:
            print(f"An unexpected error occurred when trying to open depth_params.json file: {e}")
            sys.exit(1)

    if eval:
        if "360" in path:
            llffhold = 8
        if llffhold:
            print("------------LLFF HOLD-------------")
            cam_names = [cam_extrinsics[cam_id].name for cam_id in cam_extrinsics]
            cam_names = sorted(cam_names)
            test_cam_names_list = [name for idx, name in enumerate(cam_names) if idx % llffhold == 0]
        else:
            with open(os.path.join(path, "sparse/0", "test.txt"), 'r') as file:
                test_cam_names_list = [line.strip() for line in file]
    else:
        test_cam_names_list = []

    reading_dir = "images" if images == None else images
    cam_infos_unsorted = readColmapCameras(
        cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_intrinsics, depths_params=depths_params,
        images_folder=os.path.join(path, reading_dir), 
        depths_folder=os.path.join(path, depths) if depths != "" else "", test_cam_names_list=test_cam_names_list)
    cam_infos = sorted(cam_infos_unsorted.copy(), key = lambda x : x.image_name)

    train_cam_infos = [c for c in cam_infos if train_test_exp or not c.is_test]
    test_cam_infos = [c for c in cam_infos if c.is_test]

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "sparse/0/points3D.ply")
    bin_path = os.path.join(path, "sparse/0/points3D.bin")
    txt_path = os.path.join(path, "sparse/0/points3D.txt")
    if not os.path.exists(ply_path):
        print("Converting point3d.bin to .ply, will happen only the first time you open the scene.")
        try:
            xyz, rgb, _ = read_points3D_binary(bin_path)
        except:
            xyz, rgb, _ = read_points3D_text(txt_path)
        storePly(ply_path, xyz, rgb)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path,
                           is_nerf_synthetic=False)
    return scene_info

def readCamerasFromTransforms(path, transformsfile, depths_folder, white_background, is_test, extension=".png"):
    cam_infos = []

    with open(os.path.join(path, transformsfile)) as json_file:
        contents = json.load(json_file)
        fovx = contents["camera_angle_x"]

        frames = contents["frames"]
        for idx, frame in enumerate(frames):
            cam_name = os.path.join(path, frame["file_path"] + extension)

            # NeRF 'transform_matrix' is a camera-to-world transform
            c2w = np.array(frame["transform_matrix"])
            # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
            c2w[:3, 1:3] *= -1

            # get the world-to-camera transform and set R, T
            w2c = np.linalg.inv(c2w)
            R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
            T = w2c[:3, 3]

            image_path = os.path.join(path, cam_name)
            image_name = Path(cam_name).stem
            image = Image.open(image_path)

            im_data = np.array(image.convert("RGBA"))

            bg = np.array([1,1,1]) if white_background else np.array([0, 0, 0])

            norm_data = im_data / 255.0
            arr = norm_data[:,:,:3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])
            image = Image.fromarray(np.array(arr*255.0, dtype=np.byte), "RGB")

            fovy = focal2fov(fov2focal(fovx, image.size[0]), image.size[1])
            FovY = fovy 
            FovX = fovx

            depth_path = os.path.join(depths_folder, f"{image_name}.png") if depths_folder != "" else ""

            cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX,
                            image_path=image_path, image_name=image_name,
                            width=image.size[0], height=image.size[1], depth_path=depth_path, depth_params=None, is_test=is_test))
            
    return cam_infos

def readNerfSyntheticInfo(path, white_background, depths, eval, extension=".png"):

    depths_folder=os.path.join(path, depths) if depths != "" else ""
    print("Reading Training Transforms")
    train_cam_infos = readCamerasFromTransforms(path, "transforms_train.json", depths_folder, white_background, False, extension)
    print("Reading Test Transforms")
    test_cam_infos = readCamerasFromTransforms(path, "transforms_test.json", depths_folder, white_background, True, extension)
    
    if not eval:
        train_cam_infos.extend(test_cam_infos)
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "points3d.ply")
    if not os.path.exists(ply_path):
        # Since this data set has no colmap data, we start with random points
        num_pts = 100_000
        print(f"Generating random point cloud ({num_pts})...")
        
        # We create random points inside the bounds of the synthetic Blender scenes
        xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
        shs = np.random.random((num_pts, 3)) / 255.0
        pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))

        storePly(ply_path, xyz, SH2RGB(shs) * 255)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path,
                           is_nerf_synthetic=True)
    return scene_info

sceneLoadTypeCallbacks = {
    "Colmap": readColmapSceneInfo,
    "Blender" : readNerfSyntheticInfo
}

def make_scene_info(config, video, cam_id, tmp_pcd, l_mask=None, r_mask=None):
    # height, width = video.ht, video.wd
    # intrinsic = video.intrinsics[cam_id] * 8.0
    # images = video.images[:cam_id + 1].detach().cpu()

    intrinsic = video.original_intrinsic
    height, width = video.original_images.shape[-2], video.original_images.shape[-1]
    fx = intrinsic[0].item()
    fy = intrinsic[1].item()

    FovX = focal2fov(fx, width)
    FovY = focal2fov(fy, height)

    img_tstamp = int(video.tstamp[cam_id])
    images = video.original_images[img_tstamp].detach().cpu()

    cam_infos = []

    left_image = images[0].squeeze().numpy()
    right_image = images[1].squeeze().numpy()
    tracked_depth = video.disps_up[cam_id].squeeze().detach().cpu().numpy()
    stereo_depth = video.stereo_depths[cam_id].squeeze().detach().cpu().numpy()
    # stereo_disp = video.stereo_disps[cam_id].squeeze().detach().cpu().numpy()

    # if mask is not None:
    #     depth_mask = np.logical_and(mask.squeeze().detach().cpu().numpy(), video.depth_masks[cam_id].squeeze().detach().cpu().numpy())
    # else:
    depth_mask = video.depth_masks[cam_id].squeeze().detach().cpu().numpy()
    if l_mask is not None and r_mask is not None:
        l_sky_mask = video.sky_mask[cam_id][0].squeeze().detach().cpu().numpy()
        r_sky_mask = video.sky_mask[cam_id][1].squeeze().detach().cpu().numpy()
    else:
        l_sky_mask = None
        r_sky_mask = None
        
    c2w = SE3(video.poses[cam_id]).matrix().cpu().numpy()
    # c2w = np.linalg.inv(w2c)
    R = c2w[:3,:3]
    T = c2w[:3,3]
    cam_info = CameraInfo(uid=cam_id, R=R, T=T, baseline=video.baseline ,FovY=FovY, FovX=FovX, depth_params=None,  
                        stereo_depth = stereo_depth, stereo_disp = None, tracked_depth=tracked_depth,
                        depth_mask=depth_mask, l_sky_mask=l_sky_mask, r_sky_mask=r_sky_mask, left_image=left_image, right_image = right_image, width=width, height=height, is_test=False)
    cam_infos.append(cam_info)

    pcd = BasicPointCloud(points=np.array(tmp_pcd.points), colors=np.array(tmp_pcd.colors), normals=np.zeros_like(np.array(tmp_pcd.points)))
    nerf_normalization = getNerfppNorm(cam_infos)
    scene_info = SceneInfo(point_cloud=pcd, train_cameras=cam_infos, test_cameras=[], 
                           nerf_normalization=nerf_normalization, ply_path=None, is_nerf_synthetic=False)
    return scene_info

def create_point_cloud(filtering_param, cam_id, video, filter_idx_list, image_mask = None, init=False):

    start_idx, ref_idx = filter_idx_list[0], filter_idx_list[1]
    # idxs = [start_idx, start_idx+1, ref_idx, ref_idx+1, ref_idx+2]
    idxs = [ref_idx-2, ref_idx-1, ref_idx, ref_idx+1, ref_idx+2]
    # idxs = [start_idx+1, ref_idx, ref_idx+1]

    depths = video.stereo_depths[idxs].unsqueeze(0).unsqueeze(2)
    depths_mask = video.depth_masks[ref_idx]

    ref_tstamp = int(video.tstamp[ref_idx])
    ref_image = video.original_images[ref_tstamp, 0].permute(1,2,0).cuda()
    # ref_image = video.images[ref_idx, 0].permute(1,2,0)

    poses = SE3(video.poses[idxs]).matrix()
    # intrinsic = video.intrinsics[cam_id] * 8.0
    intrinsic = video.original_intrinsic
    intr_mat = torch.tensor([[intrinsic[0].item(), 0, intrinsic[2].item(), 0], [0, intrinsic[1].item(), intrinsic[3].item(), 0], [0, 0, 1, 0], [0, 0, 0, 1]]).unsqueeze(0).expand(poses.size(0), -1, -1).to(video.device)

    proj_mat = torch.stack((poses, intr_mat), dim=1).unsqueeze(0)   # 1, N, 2, 4, 4
    reproj_mask = create_reproj_filter(filtering_param, ref_image, depths, depths_mask, image_mask, proj_mat)

    ref_depth = depths[:, 2, ...].squeeze()  # [1, 1, H, W]
    ref_pose = poses[2]

    # for outdoor scene, we use depth quantization mask to remove big depth values
    # depth_threshold = torch.quantile(ref_depth, 0.95)
    sky_mask = video.sky_mask[ref_idx][0]

    valid_mask = depths_mask & sky_mask & reproj_mask
    # valid_mask = depths_mask & reproj_mask

    if image_mask is not None:
        valid_mask = valid_mask & image_mask
        
    pts, clr = generate_pcd(ref_image, ref_depth, ref_pose, intrinsic, valid_mask)
    point_cloud = create_point_actor(pts, clr)

    return point_cloud

def create_reproj_filter(filtering_param, ref_image, depths, depth_mask, image_mask, proj_mat):

    # 참조 깊이 및 소스 깊이 분리
    # ref_depth = depths[:, -1, ...]  # [1, 1, H, W]
    # src_depths = depths[:, :-1, ...]  # [1, N-1, 1, H, W]
    # ref_cam = proj_mat[:, -1, ...]  # [1, 2, 4, 4]
    # src_cams = proj_mat[:, :-1, ...]  # [1, N-1, 2, 4, 4]
    # ref_depth = depths[:, 2, ...]  # [1, 1, H, W]
    # src_depths = torch.cat([depths[:, 0, ...], depths[:, 1, ...], depths[:, 3, ...], depths[:, 4, ...] ], dim=1).unsqueeze(2)
    # ref_cam = proj_mat[:, 2, ...]  # [1, 2, 4, 4]
    # src_cams = torch.cat([proj_mat[:, 0, ...].unsqueeze(1), proj_mat[:, 1, ...].unsqueeze(1), proj_mat[:, 3, ...].unsqueeze(1), proj_mat[:, 4, ...].unsqueeze(1) ], dim=1)

    ref_depth = depths[:, 1, ...]  # [1, 1, H, W]
    src_depths = torch.cat([depths[:, 0, ...], depths[:, 2, ...]], dim=1).unsqueeze(2)
    ref_cam = proj_mat[:, 1, ...]  # [1, 2, 4, 4]
    src_cams = torch.cat([proj_mat[:, 0, ...].unsqueeze(1), proj_mat[:, 2, ...].unsqueeze(1)], dim=1)

    thresh_dips, filter_thresh, thresh_view = filtering_param

    # 재투영 좌표 및 가시성 마스크 계산
    reproj_xyd, in_range = depth_fusion.get_reproj(ref_depth, src_depths, ref_cam, src_cams)
    vis_masks, vis_mask = depth_fusion.vis_filter(
        ref_depth, reproj_xyd, in_range,
        thresh_dips, filter_thresh, thresh_view
    )
    H, W = ref_depth.shape[-2:]
    # 평균 깊이 계산
    # ref_depth_avg = depth_fusion.ave_fusion(ref_depth, reproj_xyd, vis_masks)

    # 픽셀 그리드 생성 및 월드 좌표 변환
    # H, W = ref_depth_avg.shape[-2:]
    # idx_img = depth_fusion.get_pixel_grids(H, W).unsqueeze(0)
    # idx_cam = depth_fusion.idx_img2cam(idx_img, ref_depth_avg, ref_cam)
    # points_world = depth_fusion.idx_cam2world(idx_cam, ref_cam)[..., :3, 0]  # [1, H, W, 3]

    # 텐서 평탄화
    # points_flat = points_world.view(-1, 3)
    # colors_flat = ref_image.view(-1, 3)
    mask_flat = vis_mask.view(-1)

    # ref_image_mask = (ref_image.sum(dim=-1) > 0).view(-1)  # RGB 값이 모두 0이면 False
    # Adding depth value mask ( too big depth values should be ignored )
    if depth_mask is None:
        depth_mask = torch.ones_like(mask_flat)

    if image_mask is not None:
        mask_flat = mask_flat  & depth_mask.view(-1) & image_mask.view(-1)
    else:
        mask_flat = mask_flat  & depth_mask.view(-1)

    # 마스크 적용 및 데이터 CPU로 이동
    # valid_indices = torch.nonzero(mask_flat, as_tuple=False).squeeze()
    # if valid_indices.numel() == 0:
    #     return np.array([]), np.array([])
    # pts = points_flat[valid_indices].detach().cpu().numpy()
    # clr = colors_flat[valid_indices].detach().cpu().numpy() / 255.0
    
    # return pts, clr, mask_flat.reshape(H, W)
    return mask_flat.reshape(H, W)

def generate_pcd(image, depth, W2C, K, depth_mask):
    import torch
    
    H, W = image.shape[:2]

    # 기존 depth 범위 조건에 depth_mask를 AND 연산으로 결합
    # valid_depth_mask = (depth > 0) & (depth < 45.0) & depth_mask
    flat_depth_mask = depth_mask.view(-1)

    fx, fy, cx, cy = K[0].item(), K[1].item(), K[2].item(), K[3].item()

    y_coords, x_coords = torch.meshgrid(
        torch.arange(H, device=depth.device), 
        torch.arange(W, device=depth.device),
        indexing='ij'
    )

    x_coords_flat = x_coords.reshape(-1)[flat_depth_mask]
    y_coords_flat = y_coords.reshape(-1)[flat_depth_mask]
    z_coords_flat = depth.reshape(-1)[flat_depth_mask]

    Xc = (x_coords_flat - cx) * z_coords_flat / fx
    Yc = (y_coords_flat - cy) * z_coords_flat / fy
    Zc = z_coords_flat

    ones = torch.ones_like(Xc)
    camera_points = torch.stack([Xc, Yc, Zc, ones], dim=1)

    C2W = torch.inverse(W2C)
    world_points_homo = (C2W @ camera_points.T).T

    world_points = world_points_homo[:, :3] / world_points_homo[:, 3:].clamp(1e-8)
    color_flat = image.view(-1, 3)[flat_depth_mask] / 255.0
    
    return world_points.detach().cpu().numpy(), color_flat.detach().cpu().numpy()


def compute_psnr(image, gt_image):
    mse = F.mse_loss(image, gt_image, reduction='none').mean(dim=0)  # H x W
    psnr = 10 * torch.log10(1 / mse)
    return psnr
    
def get_image_mask(image, gt_image, threshold):
    with torch.no_grad():
        psnr_map = compute_psnr(image, gt_image)
        mask = psnr_map < threshold
    return mask

def get_alpha_mask(alpha, threshold):
    with torch.no_grad():
        mask = alpha < threshold
    return mask

def create_point_actor(points, colors):
    import open3d as o3d

    # points가 float32였다면 float64로 변환
    points = points.astype(np.float64)  
    colors = colors.astype(np.float64)
    
    if points.size != 0:
        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(points)
        point_cloud.colors = o3d.utility.Vector3dVector(colors)
        return point_cloud
    else:
        return None