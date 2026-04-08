import argparse
import sys
import time
from pathlib import Path

import torch
from omegaconf import OmegaConf
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parent
DROID_SLAM_DIR = REPO_ROOT / "droid_slam"

for path in (REPO_ROOT, DROID_SLAM_DIR):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from image_stream import euroc_image_stream, tartan_image_stream
from droid_slam.droid import Droid


DATASET_PRESETS = {
    "tartan": {
        "baseline": 0.25,
        "calib": REPO_ROOT / "calib" / "tartan.txt",
        "stream_fn": tartan_image_stream,
    },
    "euroc": {
        "baseline": 0.1101,
        "calib": REPO_ROOT / "calib" / "euroc.txt",
        "stream_fn": euroc_image_stream,
    },
}

DEFAULT_WEIGHTS = REPO_ROOT / "weight" / "pretrained_models" / "droid.pth"
DEFAULT_STEREO_CKPT = REPO_ROOT / "weight" / "pretrained_models" / "model_best_bp2.pth"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run Stereo-GS on a single stereo sequence."
    )
    parser.add_argument(
        "--dataset",
        choices=sorted(DATASET_PRESETS.keys()),
        default="tartan",
        help="Dataset preset that selects the stereo loader, default calibration, and baseline.",
    )
    parser.add_argument(
        "--datapath",
        help="Path to a single stereo sequence. If omitted, --data_root and --scene must be set.",
    )
    parser.add_argument(
        "--data_root",
        help="Root directory that contains dataset sequences. Used together with --scene.",
    )
    parser.add_argument(
        "--scene",
        help="Sequence name under --data_root, for example SE001 or MH_01_easy.",
    )
    parser.add_argument("--calib", help="Path to calibration file.")
    parser.add_argument("--gt", help="Optional ground-truth file.")
    parser.add_argument("--weights", default=str(DEFAULT_WEIGHTS))
    parser.add_argument("--buffer", type=int, default=256)
    parser.add_argument(
        "--image_size",
        nargs=2,
        type=int,
        metavar=("HEIGHT", "WIDTH"),
        default=[256, 320],
    )
    parser.add_argument("--disable_vis", action="store_true", default=True)
    parser.add_argument("--stride", type=int, default=3)
    parser.add_argument(
        "--stereo",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable stereo processing. Stereo-GS expects stereo input.",
    )
    parser.add_argument("--baseline", type=float, help="Stereo baseline in meters.")

    parser.add_argument("--beta", type=float, default=0.3)
    parser.add_argument("--filter_thresh", type=float, default=2.4)
    parser.add_argument("--warmup", type=int, default=8, help="Number of warmup frames.")
    parser.add_argument(
        "--keyframe_thresh",
        type=float,
        default=4.0,
        help="Threshold to create a new keyframe.",
    )
    parser.add_argument(
        "--frontend_thresh",
        type=float,
        default=16.0,
        help="Add edges between frames within this distance.",
    )
    parser.add_argument(
        "--frontend_window", type=int, default=25, help="Frontend optimization window."
    )
    parser.add_argument(
        "--frontend_radius", type=int, default=2, help="Force edges within this radius."
    )
    parser.add_argument(
        "--frontend_nms", type=int, default=1, help="Non-maximal suppression of edges."
    )

    parser.add_argument("--backend_thresh", type=float, default=24.0)
    parser.add_argument("--backend_radius", type=int, default=2)
    parser.add_argument("--backend_nms", type=int, default=2)
    parser.add_argument("--upsample", action="store_true")

    parser.add_argument("--final_refinement_iters", type=int, default=26000)
    parser.add_argument(
        "--save_img",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--use_gui",
        action=argparse.BooleanOptionalAction,
        default=True,
    )

    parser.add_argument(
        "--ckpt_dir",
        default=str(DEFAULT_STEREO_CKPT),
        type=str,
        help="Path to the FoundationStereo checkpoint.",
    )
    parser.add_argument(
        "--scale",
        default=1.0,
        type=float,
        help="Downsize the image by scale, must be <= 1.",
    )
    parser.add_argument(
        "--hiera",
        default=0,
        type=int,
        help="Hierarchical inference for high-resolution images.",
    )
    parser.add_argument(
        "--valid_iters",
        type=int,
        default=32,
        help="Number of flow-field updates during forward pass.",
    )
    return parser.parse_args()


def resolve_sequence_path(args):
    if args.datapath:
        return Path(args.datapath).expanduser().resolve()

    if args.data_root and args.scene:
        return (Path(args.data_root).expanduser() / args.scene).resolve()

    raise ValueError("Provide either --datapath or both --data_root and --scene.")


def build_runtime_args(args):
    preset = DATASET_PRESETS[args.dataset]

    sequence_path = resolve_sequence_path(args)
    calib_path = Path(args.calib).expanduser().resolve() if args.calib else preset["calib"]
    baseline = args.baseline if args.baseline is not None else preset["baseline"]

    ckpt_path = Path(args.ckpt_dir).expanduser().resolve()
    weights_path = Path(args.weights).expanduser().resolve()
    cfg_path = ckpt_path.with_name("cfg.yaml")

    cfg_dict = vars(args).copy()
    cfg_dict.update(
        {
            "datapath": str(sequence_path),
            "calib": str(calib_path),
            "baseline": baseline,
            "weights": str(weights_path),
            "ckpt_dir": str(ckpt_path),
            "scene_name": sequence_path.name,
            "image_size": [int(v) for v in args.image_size],
            "stereo": True,
        }
    )

    missing_paths = [path for path in (calib_path, ckpt_path, cfg_path, weights_path) if not path.exists()]
    if missing_paths:
        missing_str = "\n".join(f"- {path}" for path in missing_paths)
        raise FileNotFoundError(f"Missing required files:\n{missing_str}")

    return OmegaConf.create(cfg_dict), OmegaConf.load(cfg_path), preset["stream_fn"]


def main():
    args = parse_args()
    runtime_args, stereo_cfg, stream_fn = build_runtime_args(args)

    torch.multiprocessing.set_start_method("spawn", force=True)

    (
        tstamps_list,
        resize_images_list,
        intrinsics_list,
        original_images,
        original_intrinsic,
    ) = stream_fn(
        runtime_args.datapath,
        runtime_args.calib,
        image_size=runtime_args.image_size,
        stereo=runtime_args.stereo,
        stride=runtime_args.stride,
    )

    droid = Droid(
        runtime_args,
        stereo_cfg,
        original_images=original_images,
        original_intrinsic=original_intrinsic,
    )
    time.sleep(5)

    print(f"Processing {runtime_args.scene_name}")
    print("Start Stereo GS SLAM")

    iterator = zip(tstamps_list, resize_images_list, intrinsics_list)
    for tstamp, image, intrinsics in tqdm(iterator, total=len(tstamps_list)):
        droid.track(tstamp, image, intrinsics=intrinsics)

    droid.terminate(runtime_args.final_refinement_iters)


if __name__ == "__main__":
    main()
