import torch
import numpy as np
import matplotlib
matplotlib.use('TkAgg')

import matplotlib.pyplot as plt

def vis_img(img):
    if isinstance(img, torch.Tensor):
        img = img.detach().cpu().numpy()
    
    img = img.squeeze()
    img = img.transpose(1,2,0).astype(np.int32)

    plt.imshow(img)

def vis_depth(depth):
    if isinstance(depth, torch.Tensor):
        depth = depth.detach().cpu().numpy()
    
    depth = depth.squeeze()

    plt.imshow(depth)