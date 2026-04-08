import sys
from pathlib import Path

MODULE_DIR = Path(__file__).resolve().parent
for path in (MODULE_DIR, MODULE_DIR / "core"):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

import torch
import lietorch
import numpy as np
import time

from core.foundation_stereo import *

from droid_net import DroidNet
from depth_video import DepthVideo
from motion_filter import MotionFilter
from droid_frontend import DroidFrontend
from droid_backend import DroidBackend
from trajectory_filler import PoseTrajectoryFiller

from collections import OrderedDict
from torch.multiprocessing import Process, Queue

from mapping import GaussianTrain, gauss_train_run


class Droid:
    def __init__(self, args, cfg, original_images = None, original_intrinsic=None):
        super(Droid, self).__init__()
        self.load_weights(args.weights)
        self.args = args
        self.disable_vis = args.disable_vis

        self.model = FoundationStereo(cfg)
        ckpt = torch.load(args.ckpt_dir, weights_only=False)
        self.model.load_state_dict(ckpt['model'])
        self.model.cuda().eval()
        self.model.requires_grad_(False)
        # self.model.eval()

        # queues for communication between frontend and mapping
        self.q_front2gs = Queue()
        self.q_gs2front = Queue()

        # store images, depth, poses, intrinsics (shared between processes)
        self.video = DepthVideo(args.image_size, args.buffer, stereo=args.stereo, baseline=args.baseline,
                                original_shape=[original_images.shape[-2], original_images.shape[-1]])

        self.video.original_intrinsic = original_intrinsic
        self.video.original_images = original_images
        
        # filter incoming frames so that there is enough motion
        self.filterx = MotionFilter(self.net, self.video, thresh=args.filter_thresh)

        # frontend process
        self.frontend = DroidFrontend(self.net, self.video, self.args, self.q_front2gs, self.q_gs2front, self.model)
        
        # backend process
        # self.backend = DroidBackend(self.net, self.video, self.args)

        # mapping process
        self.mapping = Process(target=gauss_train_run, args=(self.q_front2gs, self.q_gs2front, self.video, self.args, self.model))
        self.mapping.start()

        # visualizer
        if not self.disable_vis:
            from visualization import droid_visualization
            self.visualizer = Process(target=droid_visualization, args=(self.video,))
            self.visualizer.start()

        # post processor - fill in poses for non-keyframes
        # self.traj_filler = PoseTrajectoryFiller(self.net, self.video)

    def load_weights(self, weights):
        """ load trained model weights """

        # print(weights)
        self.net = DroidNet()
        state_dict = OrderedDict([
            (k.replace("module.", ""), v) for (k, v) in torch.load(weights).items()])

        state_dict["update.weight.2.weight"] = state_dict["update.weight.2.weight"][:2]
        state_dict["update.weight.2.bias"] = state_dict["update.weight.2.bias"][:2]
        state_dict["update.delta.2.weight"] = state_dict["update.delta.2.weight"][:2]
        state_dict["update.delta.2.bias"] = state_dict["update.delta.2.bias"][:2]

        self.net.load_state_dict(state_dict)
        self.net.to("cuda:0").eval()
        self.net.requires_grad_(False)

    def track(self, tstamp, image, depth=None, intrinsics=None):
        """ main thread - update map """

        # with torch.no_grad():
        with torch.inference_mode():
            # check there is enough motion
            # keyframe detection & depth video update (L, R)
            self.filterx.track(tstamp, image, depth, intrinsics)

            # local bundle adjustment
            # graph construction and optimization
            self.frontend()

            # global bundle adjustment
            # 실제로는 terminate 이후에 실행된다. ( optimal X )
            # self.backend()

    def terminate(self, final_iter, stream=None):
        """ terminate the visualization process, return poses [t, q] """
        torch.cuda.empty_cache()

        print("########################################################")
        print("Final GBA before color refinement")
        print("########################################################")
        
        self.frontend.GBA(20)
        self.q_front2gs.put(["GBA"])
        
        del self.frontend
        
        self.q_front2gs.put(["terminate", final_iter])

        while True:
            if not self.q_gs2front.empty():
                msg = self.q_gs2front.get()
                if msg[0] == "terminate":
                    break
        
        print("End of Stereo SLAM")

        # camera_trajectory = self.traj_filler(stream)
        # return camera_trajectory.inv().data.cpu().numpy()
