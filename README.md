# Stereo-GS

## Online 3D Gaussian Splatting Mapping Using Stereo Depth Estimation

**Official repository for:**  
*Stereo-GS: Online 3D Gaussian Splatting Mapping Using Stereo Depth Estimation*  
[Paper (Electronics 2025)](https://www.mdpi.com/2079-9292/14/22/4436)

---

## Overview

Stereo-GS is an **online stereo 3D Gaussian Splatting mapping framework** that combines:

- **stereo SLAM** for camera tracking,
- **stereo depth estimation** for metric scene geometry,
- **online Gaussian map optimization** for photorealistic reconstruction.

Unlike offline 3DGS pipelines, Stereo-GS incrementally reconstructs the scene from a **stream of stereo frames**. The current implementation couples a DROID-SLAM-style frontend with a stereo depth network and an online Gaussian mapping module, enabling continuous map updates as new keyframes arrive.

The repository currently focuses on **stereo sequence processing** and includes data loading utilities for **TartanAir** and **EuRoC**.

---

## Pipeline

Stereo-GS consists of the following stages:

1. **Tracking / Pose Estimation**  
   A DROID-SLAM-based frontend tracks incoming frames, selects keyframes, and performs local/global optimization.

2. **Stereo Depth Estimation**  
   A foundation stereo model predicts dense disparity from left-right image pairs, followed by occlusion-aware masking and depth conversion using camera intrinsics and stereo baseline.

3. **Online Gaussian Mapping**  
   Stereo depth is converted into point clouds for Gaussian initialization and incremental map updates. The Gaussian map is continuously refined through differentiable rendering and photometric optimization.

4. **Global Pose Update and Gaussian Deformation**  
   After global bundle adjustment, the Gaussian map is deformed to stay aligned with the refined camera poses.

5. **Evaluation / Rendering**  
   The system renders left/right views and reports **PSNR**, **SSIM**, and **LPIPS**.

---

## Highlights

- **Online stereo-to-Gaussian reconstruction** from sequential stereo image streams
- **Depth-consistent Gaussian initialization** using stereo disparity estimation
- **Joint tracking and mapping** through a multi-process pipeline
- **Global BA-aware Gaussian update** after pose refinement
- **Stereo dataset support** for TartanAir and EuRoC in the current codebase

---

## Installation

### 1) Create environment

```bash
conda create -n stereo-gs python=3.10 -y
conda activate stereo-gs

conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
```

### 2) Clone repository

```bash
git clone --recursive https://github.com/lbg030/Stereo-GS.git
cd Stereo-GS
git submodule update --init --recursive
```

### 3) Install Python dependencies

```bash
pip install open3d opencv-python omegaconf matplotlib imageio joblib timm einops scipy scikit-image plyfile tqdm pillow torchmetrics trimesh pyyaml gdown
```

### 4) Install submodules / CUDA extensions

```bash
pip install -e thirdparty/lietorch
python setup.py install
pip install ./submodules/diff-gaussian-rasterization
pip install ./submodules/simple-knn
```

### 5) Set `PYTHONPATH`

Some modules in the current research code rely on direct path imports.

```bash
export PYTHONPATH=$PWD:$PWD/droid_slam:$PYTHONPATH
```

> **Note**
>
> - The current repository snapshot expects the DROID-SLAM backend sources referenced by `setup.py`.
> - If your checkout is missing required build files such as `src/` CUDA backend sources or `thirdparty/` dependencies, sync/add them before running `python setup.py install`.

---

## Pretrained Weights

Stereo-GS expects the following checkpoints:

- `droid.pth` for the tracking frontend
- `model_best_bp2.pth` for the stereo depth model
- `cfg.yaml` for the stereo model configuration

Checkpoint sources:

- **FoundationStereo** pretrained stereo checkpoint: [NVlabs/FoundationStereo](https://github.com/NVlabs/FoundationStereo)
- **DROID-SLAM** pretrained tracking checkpoint: [princeton-vl/droid-slam](https://github.com/princeton-vl/droid-slam)

Recommended layout:

```bash
weight/pretrained_models/
├── cfg.yaml
├── droid.pth
└── model_best_bp2.pth
```

> `stereo_demo.py` loads `cfg.yaml` from the same directory as `model_best_bp2.pth`.

---

## Dataset Layout

### TartanAir

```bash
/path/to/TartanAir/stereo/SE001/
├── image_left/
│   ├── 000000_left.png
│   └── ...
└── image_right/
    ├── 000000_right.png
    └── ...
```

- calibration: `calib/tartan.txt`
- baseline: `0.25`

### EuRoC

```bash
/path/to/EuRoC/MH_01_easy/
└── mav0/
    ├── cam0/data/
    │   ├── 1403636579763555584.png
    │   └── ...
    └── cam1/data/
        ├── 1403636579763555584.png
        └── ...
```

- calibration: `calib/euroc.txt`
- baseline: `0.1101`

---

## Running Stereo-GS

### Single-sequence CLI

`stereo_demo.py` now runs a **single stereo sequence** through a small CLI.

Basic usage:

```bash
export PYTHONPATH=$PWD:$PWD/droid_slam:$PYTHONPATH
python stereo_demo.py \
  --dataset tartan \
  --datapath /path/to/TartanAir/stereo/SE001
```

### Example: TartanAir

```bash
python stereo_demo.py \
  --dataset tartan \
  --datapath /path/to/TartanAir/stereo/SE001 \
  --weights weight/pretrained_models/droid.pth \
  --ckpt_dir weight/pretrained_models/model_best_bp2.pth
```

### Example: EuRoC

```bash
python stereo_demo.py \
  --dataset euroc \
  --datapath /path/to/EuRoC/MH_01_easy \
  --weights weight/pretrained_models/droid.pth \
  --ckpt_dir weight/pretrained_models/model_best_bp2.pth
```

You can also provide `--data_root` and `--scene` instead of `--datapath`:

```bash
python stereo_demo.py \
  --dataset tartan \
  --data_root /path/to/TartanAir/stereo \
  --scene SE001
```

> **Important**
>
> - The default checkpoint paths assume `weight/pretrained_models/droid.pth` and `weight/pretrained_models/model_best_bp2.pth`.
> - `euroc.sh` and `tartanair.sh` now expect the dataset root as the first argument or from `EUROC_ROOT` / `TARTANAIR_ROOT`.

---

## Outputs

The current implementation saves:

- rendered left-view images
- rendered right-view images
- per-scene evaluation metrics

Typical output locations:

```bash
res/
├── euroc/<scene>/left/
├── euroc/<scene>/right/
└── <scene>/eval/metrics.txt
```

The evaluation stage reports:

- **PSNR**
- **SSIM**
- **LPIPS**
- number of optimized Gaussians

---

## Repository Structure

```bash
Stereo-GS/
├── stereo_demo.py                 # main research demo script
├── image_stream.py                # stereo dataset loaders
├── calib/                         # calibration files
├── droid_slam/
│   ├── droid.py                   # tracking + mapping orchestration
│   ├── droid_frontend.py          # frontend optimization and stereo depth update
│   ├── mapping.py                 # online Gaussian training and evaluation
│   ├── scene/                     # Gaussian scene representation
│   ├── gaussian_renderer/         # differentiable rendering
│   └── core/                      # stereo depth network
├── evaluation_scripts/            # evaluation utilities
├── submodules/
│   ├── diff-gaussian-rasterization
│   └── simple-knn
└── weight/pretrained_models/      # config / checkpoints
```

---

## Acknowledgements

This repository builds upon several excellent projects, including:

- DROID-SLAM
- 3D Gaussian Splatting
- diff-gaussian-rasterization
- simple-knn
- stereo depth estimation components integrated in `droid_slam/core`

---

## Citation

If you find this work useful, please consider citing:

```bibtex
@article{park2025stereo,
  title={Stereo-GS: Online 3D Gaussian Splatting Mapping Using Stereo Depth Estimation},
  author={Park, Junkyu and Lee, Byeonggwon and Lee, Sanggi and Song, Soohwan},
  journal={Electronics},
  volume={14},
  number={22},
  pages={4436},
  year={2025},
  publisher={MDPI}
}
```
