import os
import glob
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import argparse

from tqdm import tqdm
# euroc dataset
def euroc_image_stream(datapath, calib, image_size=[320,512], stereo=False, stride=1):
    with open(calib, "r") as f:
        # left_intrinsic
        line = f.readline()
        fx, fy, cx, cy = [float(x) for x in line.split()[1:]]
        Kl = np.array([[fx, 0, cx],
                     [0, fy, cy],
                     [0, 0, 1]])

        # left_ distortion
        line = f.readline()
        k1, k2, p1, p2, k3 = [float(x) for x in line.split()[1:]]
        left_dist_coeffs = np.array([k1, k2, p1, p2, k3])

        # right_intrinsic
        line = f.readline()
        fx, fy, cx, cy = [float(x) for x in line.split()[1:]]
        Kr = np.array([[fx, 0, cx],
                     [0, fy, cy],
                     [0, 0, 1]])

        # right_distortion
        line = f.readline()
        k1, k2, p1, p2, k3 = [float(x) for x in line.split()[1:]]
        right_dist_coeffs = np.array([k1, k2, p1, p2, k3])

        # left camera extrinsic
        line = f.readline()
        left_extrinsic = np.array([float(x) for x in line.split()[1:]]).reshape(4, 4)
        inv_left_extrinsic = np.linalg.inv(left_extrinsic)

        # right camera extrinsic
        line = f.readline()
        right_extrinsic = np.array([float(x) for x in line.split()[1:]]).reshape(4, 4)
        inv_right_extrinsic = np.linalg.inv(right_extrinsic)
    
    R_l = inv_left_extrinsic[:3, :3]
    t_l = inv_left_extrinsic[:3, 3]

    R_r = inv_right_extrinsic[:3, :3]
    t_r = inv_right_extrinsic[:3, 3]

    # Left 카메라 좌표계 기준으로 변환
    R = R_r @ R_l.T              # 두 카메라 간의 상대 회전
    T = t_r - (R @ t_l)          # 두 카메라 간의 상대 이동

    orig_image_size = (752, 480)  

    R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
        Kl, left_dist_coeffs, Kr, right_dist_coeffs, orig_image_size, R, T, alpha=0
    )

    original_intrinsic = torch.tensor([P1[0, 0], P1[1, 1], P1[0, 2], P1[1, 2]], dtype=torch.float32).cuda()

    map_l = cv2.initUndistortRectifyMap(Kl, left_dist_coeffs, R1, P1, orig_image_size, cv2.CV_32F)
    map_r = cv2.initUndistortRectifyMap(Kr, right_dist_coeffs, R2, P2, orig_image_size, cv2.CV_32F)

    images_left = sorted(glob.glob(os.path.join(datapath, 'mav0/cam0/data/*.png')))[::stride]
    images_right = [x.replace('cam0', 'cam1') for x in images_left]

    original_images = []
    resize_images_list = []
    intrinsics_list = []
    tstamps = []
    
    # Stereo matching -> undistoriton and rectification
    for t, (imgL, imgR) in tqdm(enumerate(zip(images_left, images_right))):
        if stereo and not os.path.isfile(imgR):
            continue
        # tstamp = float(imgL.split('/')[-1][:-4])        
        images = [cv2.remap(cv2.imread(imgL), map_l[0], map_l[1], interpolation=cv2.INTER_LINEAR)]
        if stereo:
            images += [cv2.remap(cv2.imread(imgR), map_r[0], map_r[1], interpolation=cv2.INTER_LINEAR)]
        
        images = torch.from_numpy(np.stack(images, 0))
        images = images.permute(0, 3, 1, 2).to(torch.uint8)

        resize_images = F.interpolate(images, tuple(image_size), mode="bilinear", align_corners=False).cuda()
        
        intrinsics = torch.tensor([P1[0,0], P1[1,1], P1[0,2], P1[1,2]], dtype=torch.float32).cuda()
        intrinsics[0] *= image_size[1] / orig_image_size[0]
        intrinsics[1] *= image_size[0] / orig_image_size[1]
        intrinsics[2] *= image_size[1] / orig_image_size[0]
        intrinsics[3] *= image_size[0] / orig_image_size[1]

        original_images.append(images)
        resize_images_list.append(resize_images)
        intrinsics_list.append(intrinsics)
        tstamps.append(t)

        # yield stride*t, images, intrinsics,

    return tstamps, resize_images_list, intrinsics_list, torch.stack(original_images), original_intrinsic

def tartan_image_stream(datapath, calib, image_size=(320, 512), stereo=False, stride=1, device="cuda:0"):
    # 1) calib 파싱: P가 12개면 3x4 projection, 4개면 [fx,fy,cx,cy]로 가정
    with open(calib, 'r') as f:
        line = f.readline().strip()
        vals = [float(x) for x in line.split()]

        fx, fy, cx, cy = vals

    original_intrinsic = torch.tensor([fx, fy, cx, cy], dtype=torch.float32, device=device)

    # 2) 이미지 목록
    images_left = sorted(glob.glob(os.path.join(datapath, "image_left/*.png")))[::stride]
    images_right = [x.replace("left", "right") for x in images_left] if stereo else None
    if len(images_left) == 0:
        raise FileNotFoundError("No images found under image_left/*.png")

    # 3) 원본 해상도는 첫 프레임에서 자동 추출
    img0 = cv2.cvtColor(cv2.imread(images_left[0]), cv2.COLOR_BGR2RGB)
    H0, W0 = img0.shape[:2]
    Hn, Wn = int(image_size[0]), int(image_size[1])   # (H', W') 형태로 받음
    sx, sy = Wn / W0, Hn / H0

    original_images = []
    resized_images_list = []
    intrinsics_list = []
    tstamps = []

    for t, imgL_path in tqdm(enumerate(images_left)):
        frames = [cv2.cvtColor(cv2.imread(imgL_path), cv2.COLOR_BGR2RGB)]
        if stereo:
            imgR_path = images_right[t]
            frames.append(cv2.cvtColor(cv2.imread(imgR_path), cv2.COLOR_BGR2RGB))

        # [V, H, W, 3] -> [V, 3, H, W]
        images = torch.from_numpy(np.stack(frames, 0)).permute(0, 3, 1, 2).to(device=device, dtype=torch.float32)

        # 리사이즈 (H', W')
        resized = F.interpolate(images, size=(Hn, Wn), mode="bilinear",
                                align_corners=False, antialias=True)

        # 4) intrinsics 스케일링 (fx,cx: sx / fy,cy: sy)
        intrinsics = torch.tensor([fx * sx, fy * sy, cx * sx, cy * sy],
                                  dtype=torch.float32, device=device)

        original_images.append(images)          # 메모리 이슈면 필요 시 삭제/CPU로 이동
        resized_images_list.append(resized)
        intrinsics_list.append(intrinsics)
        tstamps.append(t)

    return tstamps, resized_images_list, intrinsics_list, torch.stack(original_images), original_intrinsic
# KITTI dataset
# def image_stream(datapath, calib, image_size=[320, 1056], stereo=False, stride=1):
#     with open(calib, 'r') as f:
#         line = f.readline()
#         left_Projection = np.array([float(x) for x in line.split()[1:]]).reshape(3, 4)

#         line = f.readline()
#         right_Projection = np.array([float(x) for x in line.split()[1:]]).reshape(3, 4)

#     images_left = sorted(glob.glob(os.path.join(datapath, "image_02/data/*.png")))[::stride]
#     images_right = [x.replace("image_02", "image_03") for x in images_left]
#     orig_image_size = [375, 1242]

#     for t, (imgL, imgR) in enumerate(zip(images_left, images_right)):    
#         images = [cv2.cvtColor(cv2.imread(imgL), cv2.COLOR_BGR2RGB)]
#         if stereo:
#             images += [cv2.cvtColor(cv2.imread(imgR), cv2.COLOR_BGR2RGB)]

#         images = torch.from_numpy(np.stack(images, 0))
#         images = images.permute(0, 3, 1, 2).to("cuda:0", dtype=torch.float32)
#         images = F.interpolate(images, image_size, mode="bilinear", align_corners=False)

#         intrinsics = torch.as_tensor([left_Projection[0,0], left_Projection[1,1], left_Projection[0,2], left_Projection[1,2]]).cuda()
#         intrinsics[0] *= image_size[1] / orig_image_size[1]
#         intrinsics[1] *= image_size[0] / orig_image_size[0]
#         intrinsics[2] *= image_size[1] / orig_image_size[1]
#         intrinsics[3] *= image_size[0] / orig_image_size[0]

#         yield stride*t, images, intrinsics