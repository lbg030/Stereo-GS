import torch
import lietorch
import numpy as np

import torch.nn.functional as F
from skimage.exposure import match_histograms
import torch.nn.functional as F

from stereo_utils.utils import InputPadder, get_occlusion_mask
from lietorch import SE3
from factor_graph import FactorGraph

import matplotlib.pyplot as plt
from visualization_utils import show_plot
class DroidFrontend:
    def __init__(self, net, video, args, q_front2gs, q_gs2front, stereo_model):
        self.args = args
        self.video = video
        self.update_op = net.update
        self.graph = FactorGraph(video, net.update, max_factors=48, upsample=args.upsample)

        self.stereo_model = stereo_model

        # local optimization window
        self.t0 = 0
        self.t1 = 0

        # frontent variables
        self.is_initialized = False
        self.count = 0

        self.max_age = 25
        self.iters1 = 4
        self.iters2 = 2

        self.warmup = args.warmup
        self.beta = args.beta
        self.frontend_nms = args.frontend_nms
        self.keyframe_thresh = args.keyframe_thresh
        self.frontend_window = args.frontend_window
        self.frontend_thresh = args.frontend_thresh
        self.frontend_radius = args.frontend_radius

        self.backend_thresh = args.backend_thresh
        self.backend_radius = args.backend_radius
        self.backend_nms = args.backend_nms

        self.mapping_initialized = False
        self.q_front2gs = q_front2gs
        self.q_gs2front = q_gs2front

    def __update(self):
        """ add edges, perform update """
        is_keyframe_remove = False

        self.count += 1
        self.t1 += 1

        if self.graph.corr is not None:
            self.graph.rm_factors(self.graph.age > self.max_age, store=True)

        self.graph.add_proximity_factors(self.t1-5, max(self.t1-self.frontend_window, 0), 
            rad=self.frontend_radius, nms=self.frontend_nms, thresh=self.frontend_thresh, beta=self.beta, remove=True)

        for itr in range(self.iters1):
            self.graph.update(None, None, use_inactive=True)

        # set initial pose for next frame
        poses = SE3(self.video.poses)
        d = self.video.distance([self.t1-3], [self.t1-2], beta=self.beta, bidirectional=True)

        if d.item() < self.keyframe_thresh:
            self.graph.rm_keyframe(self.t1 - 2)
            is_keyframe_remove = True
            
            with self.video.get_lock():
                self.video.counter.value -= 1
                self.t1 -= 1

        else:
            for itr in range(self.iters2):
                self.graph.update(None, None, use_inactive=True)

        # set pose for next itration
        self.video.poses[self.t1] = self.video.poses[self.t1-1]
        self.video.disps[self.t1] = self.video.disps[self.t1-1].mean()

        if not is_keyframe_remove:
            idx = self.t1 - 3
            # update poses and disps
            if not self.mapping_initialized:
                for i in range(2, idx+1):
                    image_tstamp = int(self.video.tstamp[i])
                    # left_image = self.video.images[i, 0].unsqueeze(0).float()
                    # right_image = self.video.images[i, 1].unsqueeze(0).float()
                    left_image = self.video.original_images[image_tstamp, 0].unsqueeze(0).cuda().float()
                    right_image = self.video.original_images[image_tstamp, 1].unsqueeze(0).cuda().float()

                    # Step for assert the image size is divisble by 32
                    padder = InputPadder(left_image.shape, divis_by=32)
                    image1, image2 = padder.pad(left_image, right_image)
                    
                    # with torch.cuda.amp.autocast(True):
                    with torch.amp.autocast('cuda', enabled=True):
                        # Left to Right disparity
                        LR_disp, LR_nconv = self.stereo_model.forward(image1, image2, iters=self.args.valid_iters, test_mode=True)
                        LR_disp = padder.unpad(LR_disp)
                        LR_nconv = padder.unpad(torch.from_numpy(LR_nconv)[None, None]).numpy().squeeze()

                        # raw_stereo = ((self.video.intrinsics[0, 0] * 8.0) * self.video.baseline / abs(LR_disp))
                        raw_stereo = (self.video.original_intrinsic[0] * self.video.baseline / abs(LR_disp))

                        # Right to Left disparity
                        flip_image1 = torch.flip(image2, [3])
                        flip_image2 = torch.flip(image1, [3])
                        RL_disp, nconv = self.stereo_model.forward(flip_image1, flip_image2, iters=self.args.valid_iters, test_mode=True)
                        RL_disp = torch.flip(RL_disp, [3])
                        RL_disp = padder.unpad(RL_disp)

                        occlusion_mask = get_occlusion_mask(LR_disp.squeeze().detach().cpu().numpy(), RL_disp.squeeze().detach().cpu().numpy(), 3.0)
                        unvalid_mask = (LR_disp.squeeze().detach().cpu().numpy() < self.video.baseline * 4)
                        final_mask = torch.tensor(occlusion_mask & ~unvalid_mask & LR_nconv, device=raw_stereo.device)

                    # self.video.sky_mask[i][1] = ((self.video.intrinsics[0, 0] * 8.0) * self.video.baseline / abs(RL_disp)) < 30
                    # self.video.sky_mask[i][0] = ((self.video.intrinsics[0, 0] * 8.0) * self.video.baseline / abs(LR_disp)) < 30
                    self.video.sky_mask[i][1] = (self.video.original_intrinsic[0] * self.video.baseline / abs(RL_disp)) < 30
                    self.video.sky_mask[i][0] = (self.video.original_intrinsic[0] * self.video.baseline / abs(LR_disp)) < 30
                    
                    self.video.stereo_depths[i] = raw_stereo
                    # self.video.stereo_disps[i] = LR_disp
                    self.video.depth_masks[i] = final_mask

                self.q_front2gs.put(["init", idx])
                self.mapping_initialized = True

                print("Mapping initialized")
            else:
                image_tstamp = int(self.video.tstamp[idx])
                
                # left_image = self.video.images[idx, 0].unsqueeze(0).float()
                # right_image = self.video.images[idx, 1].unsqueeze(0).float()

                left_image = self.video.original_images[image_tstamp, 0].unsqueeze(0).cuda().float()
                right_image = self.video.original_images[image_tstamp, 1].unsqueeze(0).cuda().float()
                
                padder = InputPadder(left_image.shape, divis_by=32)
                image1, image2 = padder.pad(left_image, right_image)
                
                # It seems that the unit of the disparity value is not in pixels.
                # with torch.cuda.amp.autocast(True):
                with torch.amp.autocast('cuda', enabled=True):
                    # Left to Right disparity
                    LR_disp, LR_nconv = self.stereo_model.forward(image1, image2, iters=self.args.valid_iters, test_mode=True)
                    LR_disp = padder.unpad(LR_disp)
                    LR_nconv = padder.unpad(torch.from_numpy(LR_nconv)[None, None]).numpy().squeeze()
                    # raw_stereo = ((self.video.intrinsics[0, 0] * 8.0) * self.video.baseline / abs(LR_disp))
                    raw_stereo = (self.video.original_intrinsic[0] * self.video.baseline / abs(LR_disp))

                    # Right to Left disparity
                    flip_image1 = torch.flip(image2, [3])
                    flip_image2 = torch.flip(image1, [3])
                    RL_disp, nconv = self.stereo_model.forward(flip_image1, flip_image2, iters=self.args.valid_iters, test_mode=True)
                    RL_disp = torch.flip(RL_disp, [3])
                    RL_disp = padder.unpad(RL_disp)

                    occlusion_mask = get_occlusion_mask(LR_disp.squeeze().detach().cpu().numpy(), RL_disp.squeeze().detach().cpu().numpy(), 3.0)
                    unvalid_mask = (LR_disp.squeeze().detach().cpu().numpy() < self.video.baseline * 4)
                    final_mask = torch.tensor(occlusion_mask & ~unvalid_mask & LR_nconv, device=raw_stereo.device)
                    # final_mask = torch.tensor(occlusion_mask & LR_nconv, device=raw_stereo.device)
            
                # self.video.sky_mask[idx][1] = ((self.video.intrinsics[0, 0] * 8.0) * self.video.baseline / abs(RL_disp)) < 30
                # self.video.sky_mask[idx][0] = ((self.video.intrinsics[0, 0] * 8.0) * self.video.baseline / abs(LR_disp)) < 30
                self.video.sky_mask[idx][1] = (self.video.original_intrinsic[0] * self.video.baseline / abs(RL_disp)) < 30
                self.video.sky_mask[idx][0] = (self.video.original_intrinsic[0] * self.video.baseline / abs(LR_disp)) < 30
                self.video.stereo_depths[idx] = raw_stereo
                # self.video.stereo_disps[idx] = LR_disp
                self.video.depth_masks[idx] = final_mask
                
                self.q_front2gs.put(["update", [idx-2, idx]])
            
            if (self.t1 % 30 == 0) and self.is_initialized:
                self.GBA()
                self.q_front2gs.put(["GBA"])

    @torch.no_grad()
    def GBA(self, steps=12):
        """ main update """
        torch.cuda.empty_cache()

        t = self.video.counter.value
        print()
        print(f"GBA: {t}")
        
        graph = FactorGraph(self.video, self.update_op, corr_impl="alt", max_factors=16*t)
        graph.add_proximity_factors(rad=self.backend_radius, 
                                    nms=self.backend_nms, 
                                    thresh=self.backend_thresh, 
                                    beta=self.beta)

        graph.update_lowmem(steps=steps)
        graph.clear_edges()
        # print(f"update_lowmem: {steps} done")
        return True
    
    def __initialize(self):
        """ initialize the SLAM system """

        self.t0 = 0
        self.t1 = self.video.counter.value      # 현재 처리하고 있는 frame의 인덱스

        self.graph.add_neighborhood_factors(self.t0, self.t1, r=3)

        for itr in range(8):
            self.graph.update(1, use_inactive=True)

        self.graph.add_proximity_factors(0, 0, rad=2, nms=2, thresh=self.frontend_thresh, remove=False)

        for itr in range(8):
            self.graph.update(1, use_inactive=True)


        # self.video.normalize()
        self.video.poses[self.t1] = self.video.poses[self.t1-1].clone()
        self.video.disps[self.t1] = self.video.disps[self.t1-4:self.t1].mean()

        # initialization complete
        self.is_initialized = True
        self.last_pose = self.video.poses[self.t1-1].clone()
        self.last_disp = self.video.disps[self.t1-1].clone()
        self.last_time = self.video.tstamp[self.t1-1].clone()

        with self.video.get_lock():
            self.video.ready.value = 1
            self.video.dirty[:self.t1] = True

        self.graph.rm_factors(self.graph.ii < self.warmup-4, store=True)

    def __call__(self):
        """ main update """

        # do initialization
        # 충분한 keyframe이 모일때까지 기다린다. ( warmup )
        if not self.is_initialized and self.video.counter.value == self.warmup:
            self.__initialize()
            
        # do update
        elif self.is_initialized and self.t1 < self.video.counter.value:
            self.__update()
