import numpy as np
import torch
import time
import open3d as o3d
import matplotlib.pyplot as plt
from tqdm import tqdm

from utils.eval import *
from utils.image_utils import save_img, save_depth, save_eval_result

from gaussian_renderer import render
from random import randint
from utils.loss_utils import l1_loss, ssim
# from utils.graphics_utils import inverse_warp_images, depth_fusion
from scene import Scene, GaussianModel
from scene.dataset_readers import make_scene_info, create_point_cloud, get_image_mask, get_alpha_mask
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
from lietorch import SE3 
from stereo_utils.utils import InputPadder, get_occlusion_mask
import random 

import threading
import torch.multiprocessing as mp
from torch.multiprocessing import Process, Queue
from gui import slam_gui, gui_utils
from gui.gui_utils import clone_obj
from custom_utils.gs_plotting import vis_gs

def score_func(view, gaussians, pipeline, background, scores):
    img_scores = torch.zeros_like(scores)
    img_scores.requires_grad = True

    image = render(view, gaussians, pipeline, background,
                   scores=img_scores)['render']

    # Backward computes and stores grad squared values
    # in img_scores's grad
    image.sum().backward()

    scores += img_scores.grad
    
def prune(scene, gaussians, pipe, background, prune_ratio):
    torch.cuda.synchronize()
    with torch.enable_grad():
        pbar = tqdm(
            total=len(scene.getTrainCameras()),
            desc='Computing Pruning Scores')
        scores = torch.zeros_like(gaussians.get_opacity)
        for view in scene.getTrainCameras():
            score_func(view, gaussians, pipe, background, scores)
            pbar.update(1)
        pbar.close()

    gaussians.prune_gaussians(prune_ratio, scores)
    
    return None
    
class GaussianTrain:
    def __init__(self, q_front2gs, q_gs2front, video, args, model):
        self.args = args
        self.model = model

        self.unexplored_pcd_thresh = 40
        # self.unexplored_pcd_thresh = 30.0
        self.thresh_disp = 0.5
        self.filter_thresh = 0.005
        self.thresh_view = 2
        self.filtering_params = (self.thresh_disp, self.filter_thresh, self.thresh_view)

        self.video = video
        self.q_front2gs = q_front2gs
        self.q_gs2front = q_gs2front

        parser = ArgumentParser(description="Training script parameters")
        
        self.dataset = ModelParams(parser)
        self.opt = OptimizationParams(parser)
        self.pipe = PipelineParams(parser)

        # self.gaussians = GaussianModel(self.dataset.sh_degree, self.opt.optimizer_type)
        self.gaussians = GaussianModel(self.dataset.sh_degree)
        self.pcd = o3d.geometry.PointCloud()

        bg_color = [1, 1, 1] if self.dataset._white_background else [0, 0, 0]
        self.background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        if self.args.use_gui:
            self.visualization()

        self.iteration = 0
        self.scene = None
        self.initialized = False
        self.viewpoint_stack = None

        self.before_pose = {}
        # self.iteration_started = False

        self.progress_bar = tqdm(total=self.opt.iterations, desc="Training Progress")

    def visualization(self):
        # Gaussian Splatting Viewer
        # self.params_gui = None
        self.q_main2vis = Queue()
        self.q_vis2main = Queue()
        self.params_gui = gui_utils.ParamsGUI(
            pipe=self.pipe,
            background=self.background,
            gaussians=self.gaussians,
            q_main2vis=self.q_main2vis,
            q_vis2main=self.q_vis2main,
        )
        gui_process = mp.Process(target=slam_gui.run, args=(self.params_gui,))
        gui_process.start()
        time.sleep(5)

    def run(self):
        while True:
            if not self.q_front2gs.empty():
                msg = self.q_front2gs.get()
                message_type = msg[0]

                if message_type == "init":
                    self.process_init_message(msg)
                elif message_type == "update":
                    self.process_update_message(msg)
                elif message_type == "GBA":
                    self.process_GBA_message(msg)
                else:
                    self.process_terminate_message(msg)
                    break
            else:
                time.sleep(0.01)
            
            if self.initialized:
                self.train_gaussians()
    
    def process_init_message(self, msg):
        self.initialized = True
        cam_id = msg[1] # self.t1-3
        for start_idx in range(2, cam_id):
            filter_idx_list = []
            if start_idx > cam_id-2:
                break
            filter_idx_list.append(start_idx)
            filter_idx_list.append(start_idx + 2)
            pcd = create_point_cloud(self.filtering_params, cam_id, self.video, filter_idx_list, init = True)
            if pcd is None:
                continue
            downsampled_pcd = pcd.voxel_down_sample(voxel_size=0.005)
            # l_sky_mask = self.video.sky_mask[start_idx+2][0].squeeze().detach().cpu().numpy()
            # r_sky_mask = self.video.sky_mask[start_idx+2][1].squeeze().detach().cpu().numpy()
            l_sky_mask = None
            r_sky_mask = None
            if start_idx == 2:
                scene_info = make_scene_info(self.dataset, self.video, start_idx+2, downsampled_pcd, l_sky_mask, r_sky_mask)
                self.scene = Scene(self.dataset, self.gaussians, scene_info=scene_info)
                self.gaussians.training_setup(self.opt)
            else:
                self.scene.scene_update(self.dataset, self.video, start_idx+2, l_sky_mask, r_sky_mask)
                self.gaussians.update_pcd(downsampled_pcd, start_idx+2)
            self.before_pose[start_idx+2] = SE3(self.video.poses[start_idx+2]).matrix().detach().cpu().numpy()
            self.pcd += downsampled_pcd
        self.scene.nerf_normalization()
        
        print("Number of points at initialisation : ", self.gaussians.get_xyz.shape[0])
        return

    def process_update_message(self, msg):
        idx_list = msg[1]
        cam_id = idx_list[1]
        filter_idx_list = [idx_list[0], idx_list[1]]

        self.scene.scene_update(self.dataset, self.video, cam_id)

        new_added_cam = self.scene.getTrainCameras()[-1]
        image = render(new_added_cam, self.gaussians, self.pipe, self.background)["render"]
        gt_image = new_added_cam.left_original_image.cuda()
        image_mask = get_image_mask(image, gt_image, self.unexplored_pcd_thresh)

        new_pcd = create_point_cloud(self.filtering_params, cam_id, self.video, filter_idx_list, image_mask=image_mask, init = True)

        # new_added_cam.l_sky_mask = self.video.sky_mask[cam_id][0].squeeze().detach().cpu().numpy()
        # new_added_cam.r_sky_mask = self.video.sky_mask[cam_id][1].squeeze().detach().cpu().numpy()

        new_added_cam.l_sky_mask = None
        new_added_cam.r_sky_mask = None
            
        # self.scene.scene_update(self.dataset, self.video, cam_id)
        # voxel downsample
        if new_pcd is not None:
            new_pcd = new_pcd.voxel_down_sample(voxel_size=0.005)
            self.pcd += new_pcd
            self.gaussians.update_pcd(new_pcd, cam_id)
        # else:
        #     print(f"Point cloud for camera {cam_id} is empty, skipping update.")

        self.scene.nerf_normalization()
        self.before_pose[cam_id] = SE3(self.video.poses[cam_id]).matrix().detach().cpu().numpy()
        
        return
    
    def process_GBA_message(self, msg):
        with torch.no_grad():
            new_pose = {key : SE3(self.video.poses[key]).matrix().detach().cpu().numpy() for key in self.before_pose.keys()}
            self.gaussians.deformation_gaussian(self.before_pose, new_pose)
            self.scene.update_pose(new_pose)
            
            # new initialization for next GBA Process
            self.before_pose = new_pose
        print("Gaussian Deformation Done")
        return

    def process_terminate_message(self, msg):
        final_iteration = msg[1]
        # final refinment
        print("Final refinement Started")
        print("current_iteration: ", self.iteration)

        for _ in range(final_iteration):
            # if self.iteration >= 30000:
            #     print("30k iteration reached...")
            #     break
            self.train_gaussians(final=True)

        print("Final refinement Done")
        print()

        print("evaluation start")
        len_cam = len(self.scene.getTrainCameras())
        left_psnr_scores = 0.0
        left_ssim_scores = 0.0
        left_lpips_scores = 0.0
        
        right_psnr_scores = 0.0
        right_ssim_scores = 0.0
        right_lpips_scores = 0.0
        
        device = "cuda"
        with torch.no_grad():
            for camera in self.scene.getTrainCameras():
                l_sky_mask = None
                r_sky_mask = None
                
                # l_sky_mask = torch.as_tensor(camera.l_sky_mask, dtype=torch.bool, device=device)
                # r_sky_mask = torch.as_tensor(camera.r_sky_mask, dtype=torch.bool, device=device)

                left_render = render(camera, self.gaussians, self.pipe, self.background)
                left_rendered_image = left_render['render'].clamp(0, 1) if l_sky_mask is None else left_render['render'].clamp(0, 1) * l_sky_mask
                # left_rendered_depth = left_render['depth'].squeeze() if l_sky_mask is None else left_render['depth'].squeeze() * l_sky_mask
                left_gt_image = camera.left_original_image.cuda().float() if l_sky_mask is None else camera.left_original_image.cuda().float() * l_sky_mask

                # right의 depth에 대해서는 sky_mask를 적용할 수 없음
                shifted_cam = self.scene.getShiftedCamera(camera, self.args.baseline)
                right_render = render(shifted_cam, self.gaussians, self.pipe, self.background)
                shifted_image = right_render["render"].clamp(0, 1) if r_sky_mask is None else right_render["render"].clamp(0, 1) * r_sky_mask
                # right_rendered_depth = right_render["depth"].squeeze() if r_sky_mask is None else right_render["depth"].squeeze() * r_sky_mask
                right_gt_image = camera.right_original_image.cuda().float() if r_sky_mask is None else camera.right_original_image.cuda().float() * r_sky_mask

                left_psnr_scores += compute_psnr(left_rendered_image, left_gt_image).item()
                left_ssim_scores += ssim(left_rendered_image, left_gt_image).item()
                left_lpips_scores += cal_lpips(left_rendered_image.unsqueeze(0), left_gt_image.unsqueeze(0)).item()

                right_psnr_scores += compute_psnr(shifted_image, right_gt_image).item()
                right_ssim_scores += ssim(shifted_image, right_gt_image).item()
                right_lpips_scores += cal_lpips(shifted_image.unsqueeze(0), right_gt_image.unsqueeze(0)).item()

                if self.args.save_img:
                    save_img(left_rendered_image, left_gt_image, camera.uid, self.args.scene_name, direction="left")
                    save_img(shifted_image, right_gt_image, camera.uid, self.args.scene_name, direction="right")
                    # save_depth(left_rendered_depth, right_rendered_depth, camera.uid, self.args.scene_name)


        print("left score")
        print("mean psnr: ", left_psnr_scores / len_cam)
        print("mean ssim: ", left_ssim_scores / len_cam)
        print("mean lpips: ", left_lpips_scores / len_cam)
        print()

        print("right score")
        print("mean psnr: ", right_psnr_scores / len_cam)
        print("mean ssim: ", right_ssim_scores / len_cam)
        print("mean lpips: ", right_lpips_scores / len_cam)
        print()

        save_eval_result(left_psnr_scores/len_cam, left_ssim_scores/len_cam, left_lpips_scores/len_cam, right_psnr_scores/len_cam, right_ssim_scores/len_cam, right_lpips_scores/len_cam, self.args.scene_name, self.gaussians.get_xyz.shape[0])

        # self.gaussians.save_ply("ply_folder/aa.ply")
        self.q_gs2front.put(["terminate"])
        
    def train_gaussians(self, v_cam=None, final=False):
        # Initialize viewpoint stack if empty
        if not self.viewpoint_stack:
            self.viewpoint_stack = self.scene.getTrainCameras().copy()
        
        # Select random viewpoint camera
        v_cam = self.viewpoint_stack.pop(randint(0, len(self.viewpoint_stack) - 1))
        
        # Update iteration counter and progress
        self.iteration += 1
        self.progress_bar.update(1)
        
        # Update learning rate
        self.gaussians.update_learning_rate(self.iteration)
        
        # Update spherical harmonics degree every 1000 iterations
        if self.iteration % 1000 == 0:
            self.gaussians.oneupSHdegree()
        
        # Render current view
        render_pkg = render(v_cam, self.gaussians, self.pipe, self.background)
        image, viewspace_point_tensor, visibility_filter, radii = (
            render_pkg["render"], 
            render_pkg["viewspace_points"], 
            render_pkg["visibility_filter"], 
            render_pkg["radii"]
        )
        
        # ==================== Photometric Loss Calculation ====================
        # Get ground truth images
        gt_image = v_cam.left_original_image.cuda()
        right_gt_image = v_cam.right_original_image.cuda()
        
        # Render shifted camera view for stereo
        shifted_cam = self.scene.getShiftedCamera(v_cam, self.args.baseline)
        right_image = render(shifted_cam, self.gaussians, self.pipe, self.background)["render"]
        
        # Apply sky masks if available
        # if v_cam.l_sky_mask is not None and v_cam.r_sky_mask is not None:
        #     # l_sky_mask = torch.tensor(v_cam.l_sky_mask).cuda()
        #     # r_sky_mask = torch.tensor(v_cam.r_sky_mask).cuda()
        #     device = torch.device('cuda')
        #     l_sky_mask = torch.as_tensor(v_cam.l_sky_mask, dtype=torch.bool, device=device)
        #     r_sky_mask = torch.as_tensor(v_cam.r_sky_mask, dtype=torch.bool, device=device)

            
        #     image = image * l_sky_mask
        #     gt_image = gt_image * l_sky_mask
        #     right_image = right_image * r_sky_mask
        #     right_gt_image = right_gt_image * r_sky_mask
        
        # Calculate losses
        
        image, gt_image = image.float(), gt_image.float()
        right_image, right_gt_image = right_image.float(), right_gt_image.float()
        
        Ll1 = l1_loss(image, gt_image)
        right_Ll1 = l1_loss(right_image, right_gt_image)
        
        ssim_value = ssim(image, gt_image)
        right_ssim_value = ssim(right_image, right_gt_image)
        
        # Average stereo losses
        Ll1 = 0.5 * (Ll1 + right_Ll1)
        ssim_value = 0.5 * (ssim_value + right_ssim_value)
        
        # Combined photometric loss
        photometric_loss = (
            (1.0 - self.opt.lambda_dssim) * Ll1 + 
            self.opt.lambda_dssim * (1.0 - ssim_value)
        )
        photometric_loss.backward()
        
        # ==================== Gaussian Management (No Grad) ====================
        with torch.no_grad():
            # Densification phase
            if self.iteration < self.opt.densify_until_iter:
                # Track max radii for pruning
                self.gaussians.max_radii2D[visibility_filter] = torch.max(
                    self.gaussians.max_radii2D[visibility_filter], 
                    radii[visibility_filter]
                )
                self.gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)
                
                # Densify and prune
                if (self.iteration > self.opt.densify_from_iter and 
                    self.iteration % self.opt.densification_interval == 0):
                    size_threshold = 20 if self.iteration > self.opt.opacity_reset_interval else None
                    self.gaussians.densify_and_prune(
                        self.opt.densify_grad_threshold, 
                        0.005, 
                        self.scene.cameras_extent, 
                        size_threshold
                    )
                
                # Pruning during densification
                # if (self.iteration >= self.opt.prune_from_iter and
                #     self.iteration < self.opt.prune_until_iter and
                #     self.iteration % self.opt.prune_interval == 0):
                #     prune_pkg = prune(
                #         self.scene, self.gaussians, self.pipe, self.background,
                #         self.opt.densify_prune_ratio
                #     )
                
                # Reset opacity
                if (self.iteration % self.opt.opacity_reset_interval == 0 or 
                    (self.dataset._white_background and self.iteration == self.opt.densify_from_iter)):
                    self.gaussians.reset_opacity()
            
            # Optimizer step
            if self.iteration < self.opt.iterations:
                self.gaussians.optimizer.step()
                self.gaussians.optimizer.zero_grad(set_to_none=True)
            
            # Post-densification pruning
            # if (self.iteration >= self.opt.densify_until_iter and
            #     self.iteration >= self.opt.prune_from_iter and
            #     self.iteration < self.opt.prune_until_iter and
            #     self.iteration % self.opt.prune_interval == 0):
            #     prune_pkg = prune(
            #         self.scene, self.gaussians, self.pipe, self.background,
            #         self.opt.after_densify_prune_ratio
            #     )
        
        # ==================== GUI Updates ====================
        with torch.no_grad():
            if self.args.use_gui and self.iteration % 200 == 0:
                if not final:
                    packet = gui_utils.GaussianPacket(
                        gaussians=clone_obj(self.gaussians),
                        current_frame=self.scene.getTrainCameras()[-1],
                        keyframes=self.scene.getTrainCameras().copy(),
                        gtcolor=self.scene.getTrainCameras()[-1].left_original_image.clone().detach().cpu(),
                        gtdepth=self.scene.getTrainCameras()[-1].stereo_depth.clone().detach().cpu(),
                    )
                else:
                    packet = gui_utils.GaussianPacket(
                        gaussians=clone_obj(self.gaussians)
                    )
                self.q_main2vis.put(packet)
        
        # Clean up GPU memory
        torch.cuda.empty_cache()

def gauss_train_run(q_front2gs, q_gs2front, video, args, model):
    try:
        gs = GaussianTrain(q_front2gs, q_gs2front, video, args, model)
        gs.run()
    finally:
        # CUDA 텐서가 포함된 큐를 닫고 참조를 해제합니다.
        q_front2gs.close()
        q_gs2front.close()
        # gs.q_main2vis.close()
        # gs.q_vis2main.close()
        
        # 모든 CUDA 텐서에 대한 참조를 명시적으로 삭제합니다.
        torch.cuda.empty_cache()
        del gs
