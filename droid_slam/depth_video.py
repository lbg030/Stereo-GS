import numpy as np
import torch
import lietorch
import droid_backends

from torch.multiprocessing import Process, Queue, Lock, Value
from collections import OrderedDict

from droid_net import cvx_upsample, upsample_disp
import geom.projective_ops as pops

from utils.depth_align_utils import align_scale_and_shift
from geom.ba import JDSA

class DepthVideo:
    def __init__(self, image_size=[480, 640], buffer=1024, stereo=False, baseline=None, device="cuda:0", original_shape=[480, 720]):
                
        # current keyframe count
        self.counter = Value('i', 0)
        self.ready = Value('i', 0)
        self.ht = ht = image_size[0]
        self.wd = wd = image_size[1]
        
        self.orig_ht, self.orig_wd = original_shape
        self.device = "cuda"

        self.stereo = stereo
        c = 1 if not self.stereo else 2

        ### state attributes ###
        self.tstamp = torch.zeros(buffer, device="cuda", dtype=torch.float).share_memory_()
        self.images = torch.zeros(buffer, c, 3, ht, wd, device="cuda", dtype=torch.uint8)
        self.dirty = torch.zeros(buffer, device="cuda", dtype=torch.bool).share_memory_()
        self.red = torch.zeros(buffer, device="cuda", dtype=torch.bool).share_memory_()
        self.poses = torch.zeros(buffer, 7, device="cuda", dtype=torch.float).share_memory_()
        self.disps = torch.ones(buffer, ht//8, wd//8, device="cuda", dtype=torch.float).share_memory_()
        self.disps_sens = torch.zeros(buffer, ht//8, wd//8, device="cuda", dtype=torch.float).share_memory_()
        self.disps_up = torch.zeros(buffer, ht, wd, device="cuda", dtype=torch.float).share_memory_()

        ### Stereo attributes ###
        self.baseline = baseline
        self.stereo_depths = torch.zeros(buffer, self.orig_ht, self.orig_wd, device="cuda", dtype=torch.float).share_memory_()
        # self.stereo_depths_down = torch.ones(buffer, ht//8, wd//8, device="cuda", dtype=torch.float).share_memory_()
        # self.stereo_disps = torch.zeros(buffer, self.orig_ht, self.orig_wd, device="cuda", dtype=torch.float).share_memory_()
        self.depth_masks = torch.ones(buffer, self.orig_ht, self.orig_wd, device="cuda", dtype=torch.bool).share_memory_()
        self.intrinsics = torch.zeros(buffer, 4, device="cuda", dtype=torch.float).share_memory_()
        self.sky_mask = torch.zeros(buffer,2, self.orig_ht, self.orig_wd, device="cuda", dtype=torch.bool).share_memory_()

        ### feature attributes ###
        self.fmaps = torch.zeros(buffer, c, 128, ht//8, wd//8, dtype=torch.half, device="cuda").share_memory_()
        self.nets = torch.zeros(buffer, 128, ht//8, wd//8, dtype=torch.half, device="cuda").share_memory_()
        self.inps = torch.zeros(buffer, 128, ht//8, wd//8, dtype=torch.half, device="cuda").share_memory_()

        # initialize poses to identity transformation
        self.poses[:] = torch.as_tensor([0, 0, 0, 0, 0, 0, 1], dtype=torch.float, device="cuda")
        
    def get_lock(self):
        return self.counter.get_lock()

    def __item_setter(self, index, item):
        if isinstance(index, int) and index >= self.counter.value:
            self.counter.value = index + 1
        
        elif isinstance(index, torch.Tensor) and index.max().item() > self.counter.value:
            self.counter.value = index.max().item() + 1

        # self.dirty[index] = True
        self.tstamp[index] = item[0]
        self.images[index] = item[1]

        if item[2] is not None:
            self.poses[index] = item[2]

        if item[3] is not None:
            self.disps[index] = item[3]

        if item[4] is not None:
            depth = item[4][3::8,3::8]
            self.disps_sens[index] = torch.where(depth>0, 1.0/depth, depth)

        if item[5] is not None:
            self.intrinsics[index] = item[5]

        if len(item) > 6:
            self.fmaps[index] = item[6]

        if len(item) > 7:
            self.nets[index] = item[7]

        if len(item) > 8:
            self.inps[index] = item[8]
        
        if len(item) > 9:
            self.baseline = item[9]

    def __setitem__(self, index, item):
        with self.get_lock():
            self.__item_setter(index, item)

    def __getitem__(self, index):
        """ index the depth video """

        with self.get_lock():
            # support negative indexing
            if isinstance(index, int) and index < 0:
                index = self.counter.value + index

            item = (
                self.poses[index],
                self.disps[index],
                self.intrinsics[index],
                self.fmaps[index],
                self.nets[index],
                self.inps[index])

        return item

    def append(self, *item):
        with self.get_lock():
            self.__item_setter(self.counter.value, item)


    ### geometric operations ###

    @staticmethod
    def format_indicies(ii, jj):
        """ to device, long, {-1} """

        if not isinstance(ii, torch.Tensor):
            ii = torch.as_tensor(ii)

        if not isinstance(jj, torch.Tensor):
            jj = torch.as_tensor(jj)

        ii = ii.to(device="cuda", dtype=torch.long).reshape(-1)
        jj = jj.to(device="cuda", dtype=torch.long).reshape(-1)

        return ii, jj

    def upsample(self, ix, mask):
        """ upsample disparity """

        disps_up = cvx_upsample(self.disps[ix].unsqueeze(-1), mask)
        self.disps_up[ix] = disps_up.squeeze()
        # disps_up = upsample_disp(self.disps[ix].unsqueeze(-1), mask)
        # return disps_up.squeeze()

    def normalize(self):
        """ normalize depth and poses """

        with self.get_lock():
            s = self.disps[:self.counter.value].mean()
            self.disps[:self.counter.value] /= s
            self.poses[:self.counter.value,:3] *= s
            self.dirty[:self.counter.value] = True


    def reproject(self, ii, jj):
        """ project points from ii -> jj """
        ii, jj = DepthVideo.format_indicies(ii, jj)
        Gs = lietorch.SE3(self.poses[None])


        # coords : ii에 있는 keyframes들의 depth를 jj에 있는 keyframes의 depth로 projection
        # valid_mask : ii->jj로 projection이 되었을 때, depth가 너무 작지 않은 point들에 대한 mask
        coords, valid_mask = \
            pops.projective_transform(Gs, self.disps[None], self.intrinsics[None], ii, jj)

        return coords, valid_mask

    def distance(self, ii=None, jj=None, beta=0.3, bidirectional=True):
        """ frame distance metric """

        return_matrix = False
        if ii is None:
            return_matrix = True
            N = self.counter.value
            ii, jj = torch.meshgrid(torch.arange(N), torch.arange(N))
        
        ii, jj = DepthVideo.format_indicies(ii, jj)

        if bidirectional:

            poses = self.poses[:self.counter.value].clone()

            d1 = droid_backends.frame_distance(
                poses, self.disps, self.intrinsics[0], ii, jj, beta)

            d2 = droid_backends.frame_distance(
                poses, self.disps, self.intrinsics[0], jj, ii, beta)

            d = .5 * (d1 + d2)

        else:
            d = droid_backends.frame_distance(
                self.poses, self.disps, self.intrinsics[0], ii, jj, beta)

        if return_matrix:
            return d.reshape(N, N)

        return d

    def ba(self, target, weight, eta, ii, jj, t0=1, t1=None, itrs=2, lm=1e-4, ep=0.1, motion_only=False):
        """ dense bundle adjustment (DBA) """

        with self.get_lock():

            # [t0, t1] window of bundle adjustment optimization
            if t1 is None:
                t1 = max(ii.max().item(), jj.max().item()) + 1

            droid_backends.ba(self.poses, self.disps, self.intrinsics[0], self.disps_sens,
                target, weight, eta, ii, jj, t0, t1, itrs, lm, ep, motion_only)
            
            ### Use Stereo prior 
            # poses = lietorch.SE3(self.poses[:t1][None])
            # disps = self.disps[:t1][None]
            # disps, _, _ = JDSA(target, weight, eta, poses, disps, self.intrinsics[None], self.stereo_depths_down, ii, jj, dscales=None, alpha=None)
            # self.disps[:t1] = disps[0]

            self.disps.clamp_(min=0.001)

    def depth_align(self, raw_stereo, target_idx, mask=None):
        # align stereo
        # target_mask = masks[target_idx].cuda()
        droid_depth = 1 / (self.disps_up[target_idx])

        # remove outliers
        lower = torch.quantile(droid_depth, 0.05)
        upper = torch.quantile(droid_depth, 0.95)
        if mask == None:
            outlier_mask = (droid_depth < lower) | (droid_depth > upper)
        else:
            outlier_mask = ((droid_depth < lower) | (droid_depth > upper)) & ~mask
        # droid_depth[outlier_mask] = 0.0

        # align stereo depth to droid depth
        # mask = target_mask & ~outlier_mask
        mask = ~outlier_mask
        with torch.no_grad():
            scale, shift, _ = align_scale_and_shift(raw_stereo.squeeze(0), droid_depth.unsqueeze(0), mask.unsqueeze(0))

        return raw_stereo * scale + shift
    
    def update_stereo_depth(self, idx):
        # Todo: use the uncertainty mask of droid slam depth map
        for i in range(2, idx+1):
            before_stereo = self.stereo_depths[i]
            droid_depth = 1 / (self.disps_up[i])

            # remove outliers
            lower = torch.quantile(droid_depth, 0.05)
            upper = torch.quantile(droid_depth, 0.95)
            outlier_mask = (droid_depth < lower) | (droid_depth > upper)
            # droid_depth[outlier_mask] = 0.0

            mask = ~outlier_mask
            with torch.no_grad():
                scale, shift, _ = align_scale_and_shift(before_stereo.squeeze(0), droid_depth.unsqueeze(0), mask.unsqueeze(0))

            new_stereo = before_stereo * scale + shift

            self.stereo_depths[i] = new_stereo
            # self.stereo_disps[i] = (self.intrinsics[0,0] * 8.0) * self.baseline / new_stereo
        return