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
from PIL import Image
import numpy as np
import matplotlib.cm as cm
import os

def mse(img1, img2):
    return (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)

def psnr(img1, img2):
    mse = (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)
    return 20 * torch.log10(1.0 / torch.sqrt(mse))

def save_img(rendered_img, gt_image, uid, scene_name, direction="left"):
    import matplotlib
    matplotlib.use('TkAgg')
    
    import matplotlib.pyplot as plt
    import os

    if direction == "left":
        directory = f"res/euroc/{scene_name}/left"
    elif direction == "right":
        directory = f"res/euroc/{scene_name}/right"
    
    os.makedirs(directory, exist_ok=True)

    rendered_img = rendered_img.detach().cpu().numpy().transpose(1, 2, 0)
    gt_image = gt_image.detach().cpu().numpy().transpose(1, 2, 0)

    # plt.imshow(rendered_img)
    # plt.axis("off")
    # plt.savefig(f'{directory}/{uid}_rendered.png', bbox_inches='tight')
    # plt.close()
    
    img = (rendered_img * 255).astype(np.uint8)
    Image.fromarray(img).save(f"{directory}/{uid}_rendered.png")

    img = (gt_image * 255).astype(np.uint8)
    Image.fromarray(img).save(f"{directory}/{uid}_gt.png")
    

    # plt.imshow(gt_image)
    # plt.axis("off")
    # plt.savefig(f'{directory}/{uid}_gt.png', bbox_inches='tight')
    # plt.close()

    return

def save_depth(left_depth, right_depth, uid, scene_name, cmap_name="jet"):
    os.makedirs(f"res/{scene_name}/depth", exist_ok=True)

    def _save_one(depth, path, cmap_name):
        depth = depth.detach().cpu().numpy().squeeze()
        depth = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)

        # 정규화
        d_min, d_max = depth.min(), depth.max()
        if d_max > d_min:
            depth_norm = (depth - d_min) / (d_max - d_min)
        else:
            depth_norm = np.zeros_like(depth)

        # colormap 적용
        cmap = cm.get_cmap(cmap_name)
        depth_color = cmap(depth_norm)[:, :, :3]  # RGBA -> RGB
        depth_color = (depth_color * 255).astype(np.uint8)

        Image.fromarray(depth_color).save(path)

    # 저장 실행
    _save_one(left_depth, f"res/{scene_name}/depth/{uid}_left_rendered.png", cmap_name)
    _save_one(right_depth, f"res/{scene_name}/depth/{uid}_right_rendered.png", cmap_name)

def save_eval_result(left_psnr_scores, left_ssim_scores, left_lpips_scores,
                     right_psnr_scores, right_ssim_scores, right_lpips_scores,
                     scene_name, num_gaussians):
    import os
    import matplotlib.pyplot as plt

    # 결과 저장할 디렉터리 생성
    os.makedirs(f"res/{scene_name}/eval", exist_ok=True)
    
    # 텍스트 파일 경로
    txt_file_path = f"res/{scene_name}/eval/metrics.txt"

    # 텍스트 파일로 저장
    with open(txt_file_path, "a", encoding="utf-8") as f:
        f.write("\n")
        f.write("=== Left View Metrics ===\n")
        f.write("PSNR:\n")
        f.write(f"{left_psnr_scores}\n")
        f.write("\nSSIM:\n")
        f.write(f"{left_ssim_scores}\n")
        f.write("\nLPIPS:\n")
        f.write(f"{left_lpips_scores}\n")

        f.write("\n=== Right View Metrics ===\n")
        f.write("PSNR:\n")
        f.write(f"{right_psnr_scores}\n")
        f.write("\nSSIM:\n")
        f.write(f"{right_ssim_scores}\n")
        f.write("\nLPIPS:\n")
        f.write(f"{right_lpips_scores}\n")
        
        f.write(f"\n# Gaussians: {num_gaussians}\n")
        