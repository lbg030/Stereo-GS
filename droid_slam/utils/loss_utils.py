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
import torch.nn  as nn
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp
try:
    from diff_gaussian_rasterization._C import fusedssim, fusedssim_backward
except:
    pass


C1 = 0.01 ** 2
C2 = 0.03 ** 2

class FusedSSIMMap(torch.autograd.Function):
    @staticmethod
    def forward(ctx, C1, C2, img1, img2):
        ssim_map = fusedssim(C1, C2, img1, img2)
        ctx.save_for_backward(img1.detach(), img2)
        ctx.C1 = C1
        ctx.C2 = C2
        return ssim_map

    @staticmethod
    def backward(ctx, opt_grad):
        img1, img2 = ctx.saved_tensors
        C1, C2 = ctx.C1, ctx.C2
        grad = fusedssim_backward(C1, C2, img1, img2, opt_grad)
        return None, None, grad, None

# def l1_loss(network_output, gt):
#     return torch.abs((network_output - gt)).mean()

def l1_loss(network_output, gt, mask=None):
    if mask is not None:
        return torch.abs((network_output*mask - gt*mask)).mean()
    return torch.abs((network_output - gt)).mean()

def l2_loss(network_output, gt):
    return ((network_output - gt) ** 2).mean()

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def ssim(img1, img2, window_size=11, size_average=True, mask=None):
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average, mask)

def _ssim(img1, img2, window, window_size, channel, size_average=True, mask=None):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        if mask is not None:
            ssim_map = ssim_map * mask
            ssim_map = ssim_map.sum() / mask.sum()
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


def fast_ssim(img1, img2):
    ssim_map = FusedSSIMMap.apply(C1, C2, img1, img2)
    return ssim_map.mean()

class SmoothLoss(nn.Module):
    def __init__(self):
        super(SmoothLoss, self).__init__()
        self.edge_conv_x_3 = torch.nn.Conv2d(3, 1, 3, bias=False).cuda()
        self.edge_conv_y_3 = torch.nn.Conv2d(3, 1, 3, bias=False).cuda()
        self.edge_conv_x_1 = torch.nn.Conv2d(1, 1, 3, bias=False).cuda()
        self.edge_conv_y_1 = torch.nn.Conv2d(1, 1, 3, bias=False).cuda()

        # Set layer weights to be edge filters
        with torch.no_grad():
            for layer in [self.edge_conv_x_3, self.edge_conv_x_1]:
                for ch in range(layer.weight.size(1)):
                    layer.weight[0, ch] = torch.Tensor([[0, 0, 0], [-0.5, 0, 0.5], [0, 0, 0]]).cuda()

            for layer in [self.edge_conv_y_3, self.edge_conv_y_1]:
                for ch in range(layer.weight.size(1)):
                    layer.weight[0, ch] = torch.Tensor([[0, -0.5, 0], [0, 0, 0], [0, 0.5, 0]]).cuda()

    def forward(self, disparity, image):
        edge_x_im = torch.exp((self.edge_conv_x_3(image).abs() * -0.33))
        edge_y_im = torch.exp((self.edge_conv_y_3(image).abs() * -0.33))
        edge_x_d = self.edge_conv_x_1(disparity)
        edge_y_d = self.edge_conv_y_1(disparity)
        return ((edge_x_im * edge_x_d)).abs().mean() + ((edge_y_im * edge_y_d)).abs().mean()

def depth_smoothness_loss(depth_map):
    """
    인접 픽셀 간 depth 값의 차이를 최소화하는 smoothing loss

    Args:
        depth_map (torch.Tensor): (H, W) 형태의 Depth Map Tensor. GPU에 있어야 함.
        
    Returns:
        torch.Tensor: Smoothness loss 값 (scalar)
    """
    # x 방향 차이 계산: 오른쪽 픽셀과의 차이
    depth_dx = torch.abs(depth_map[:, :-1] - depth_map[:, 1:])
    
    # y 방향 차이 계산: 아래쪽 픽셀과의 차이
    depth_dy = torch.abs(depth_map[:-1, :] - depth_map[1:, :])
    
    # 전체 loss는 x, y 방향 차이의 평균
    loss = torch.mean(depth_dx) + torch.mean(depth_dy)
    
    return loss

import torch
import torch.nn.functional as F

def disparity_loss(left_image, right_image, disparity):
    """
    Batch=1로 고정된 상태에서,
    left_image, right_image와 disparity를 이용해 L2 Loss를 계산.
    
    - left_image, right_image  : (C, H, W) 또는 (H, W)
    - disparity                : (H, W), 오른쪽 방향으로의 시차라고 가정
    - 모두 GPU 사용 (자동 미분 가능)

    Returns:
        torch.Tensor: 모든 픽셀에 대한 L2 Loss(mean) 값
    """
    device = torch.device("cuda")

    # 1) GPU로 옮기고, left/right 이미지를 requires_grad=True로 설정
    left_image = left_image.to(device)
    right_image = right_image.to(device)
    disparity = disparity.to(device)

    # 2) 만약 (H, W) 형태라면 (1, 1, H, W)로,
    #    (C, H, W) 형태라면 (1, C, H, W)로 바꿔서 grid_sample에 맞춤
    if left_image.dim() == 2:
        # (H, W) -> (1, 1, H, W)
        left_image = left_image.unsqueeze(0).unsqueeze(0)
    elif left_image.dim() == 3:
        # (C, H, W) -> (1, C, H, W)
        left_image = left_image.unsqueeze(0)

    if right_image.dim() == 2:
        right_image = right_image.unsqueeze(0).unsqueeze(0)
    elif right_image.dim() == 3:
        right_image = right_image.unsqueeze(0)

    # 이제 (B=1, C, H, W) 형태
    B, C, H, W = left_image.shape

    # 3) base 좌표 생성 (0 ~ W-1, 0 ~ H-1)
    x_base = torch.linspace(0, W - 1, W, device=device).repeat(H, 1)  # (H, W)
    y_base = torch.linspace(0, H - 1, H, device=device).repeat(W, 1).T  # (H, W)

    # disparity도 (H, W)로 맞춤
    if disparity.dim() > 2:
        disparity = disparity.squeeze()  # e.g. (1, H, W) -> (H, W)

    # 4) x좌표에서 disparity만큼 뺀 위치 계산
    x_right = x_base - disparity
    x_right = torch.clamp(x_right, 0, W - 1)  # 이미지 범위 밖 방지

    # 5) grid_sample에 넣기 위해 -1~1로 정규화
    x_right_norm = (x_right / (W - 1)) * 2 - 1  # [0, W-1] -> [-1, 1]
    y_base_norm = (y_base / (H - 1)) * 2 - 1    # [0, H-1] -> [-1, 1]

    # (H, W, 2)를 (1, H, W, 2)로 만들어 batch dimension 맞춤
    grid = torch.stack((x_right_norm, y_base_norm), dim=-1).unsqueeze(0)

    # 6) grid_sample로 right_image 픽셀을 bilinear 보간
    sampled_right = F.grid_sample(
        right_image.float(), 
        grid, 
        mode='bicubic', 
        align_corners=True
    )
    
    # 7) L2 Loss 계산
    loss = ((left_image - sampled_right) ** 2).mean()
    return loss

def normalize(input, mean=None, std=None):
    input_mean = torch.mean(input, dim=1, keepdim=True) if mean is None else mean
    input_std = torch.std(input, dim=1, keepdim=True) if std is None else std
    return (input - input_mean) / (input_std + 1e-2*torch.std(input.reshape(-1)))

def patchify(input, patch_size):
    patches = F.unfold(input, kernel_size=patch_size, stride=patch_size).permute(0,2,1).view(-1, 1*patch_size*patch_size)
    return patches

def margin_l2_loss(network_output, gt, margin, return_mask=False, low_conf_mask=None):
    mask = torch.logical_or((network_output - gt).abs() > margin, low_conf_mask)
    if not return_mask:
        return ((network_output - gt)[mask] ** 2).mean()
    else:
        return ((network_output - gt)[mask] ** 2).mean(), mask
    
    
def patch_norm_mse_loss(input, target, margin, mask=None, return_mask=False):
    input_patches = normalize(input)
    target_patches = normalize(target)
    return margin_l2_loss(input_patches, target_patches, margin, return_mask, mask)

def patch_norm_mse_loss_global(input, target, margin, mask=None, return_mask=False):
    input_patches = normalize(input, std = input.std().detach())
    target_patches = normalize(target, std = target.std().detach())
    return margin_l2_loss(input_patches, target_patches, margin, return_mask, mask)
