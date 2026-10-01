#!/usr/bin/env python3
"""MCAP session -> Depth Pro -> WiLoR pipeline.

The Stera app's own on-device depth (ARCore) is low resolution (e.g.
160x90) and noisy. This pipeline keeps the MCAP session's other sensor
streams -- most importantly the ARCore-tracked camera 6-DoF pose -- but
replaces its depth with Apple's Depth Pro (run on the RGB frames, at RGB
resolution, locked to the session's real calibrated focal length), and
feeds that depth into WiLoR for depth-anchored 3D hand pose.

Output episode contains exactly the three things asked for:
  - /cam-pose   : verbatim from the MCAP session (ARCore SLAM/VIO)
  - /depth      : from Depth Pro (replaces the low-res MCAP depth)
  - /hand-pose  : from WiLoR, anchored using the Depth Pro depth

Plus a world-stabilized Rerun (.rrd) view: moving camera frustum + trail
(from the MCAP camera pose) with the WiLoR hand skeleton placed in world
frame via ``optical_to_world``, reusing stera.viz.Visualizer.

Usage::

    python3 pipeline_mcap_depth_wilor.py \\
        --mcap data/session_data.mcap \\
        --wilor-dir WiLoR \\
        --output-dir pipeline_output_mcap
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import replace
from pathlib import Path

import h5py
import numpy as np
import torch
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("pipeline_mcap_depth_wilor")


def get_device(device_str: str | None = None) -> torch.device:
    if device_str:
        return torch.device(device_str)
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    return torch.device("cpu")


class DepthProSession:
    """Thin MCAPReader wrapper that overrides the RGB intrinsics' K with ``rgb_K.npy``.

    Both ``rgb_intrinsics`` and ``depth_intrinsics`` return the RGB camera's
    calibration (width/height/D from the MCAP, K from ``rgb_K.npy``): Depth
    Pro produces depth at RGB resolution (not the native low-res ARCore depth
    sensor's), so every consumer (Visualizer's frustum/pinhole, point-cloud
    back-projection) needs to agree with the RGB camera's calibration.
    """

    def __init__(self, mcap_path: str, rgb_K: np.ndarray | None = None):
        from stera.data import MCAPReader
        self._session = MCAPReader(mcap_path)
        self._rgb_K = rgb_K

    @property
    def rgb_intrinsics(self):
        intr = self._session.rgb_intrinsics
        if intr is None or self._rgb_K is None:
            return intr
        return replace(intr, K=self._rgb_K)

    @property
    def depth_intrinsics(self):
        return self.rgb_intrinsics

    def __getattr__(self, name):
        return getattr(self._session, name)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="MCAP camera pose + Depth Pro depth + WiLoR hand pose.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--mcap", type=str, default="data/session_data.mcap", help="Input MCAP session.")
    parser.add_argument("--wilor-dir", type=str, default="WiLoR", help="Path to the WiLoR repo clone.")
    parser.add_argument("--output-dir", type=str, default="pipeline_output_mcap", help="Output directory.")
    parser.add_argument("--rgb-k", type=str,
                         default=str(Path(__file__).resolve().parent / "ml-depth-pro" / "cam" / "rgb_K.npy"),
                         help="3x3 RGB camera intrinsics (.npy) used to configure Depth Pro (f_px = fx).")
    parser.add_argument("--max-frames", type=int, default=None, help="Limit number of frames processed.")
    parser.add_argument("--yolo-conf", type=float, default=0.4, help="YOLO hand-detector confidence threshold.")
    parser.add_argument("--device", type=str, default=None, help="Torch device (default: auto).")
    parser.add_argument("--no-half", dest="half", action="store_false", help="Disable FP16 for Depth Pro.")
    parser.add_argument("--map-3d", type=str, default="auto",
                         choices=["auto", "mesh", "mesh_cloud", "point_cloud", "both", "none"],
                         help="ARCore scene map to show in the Rerun 3D view (if the session has one).")
    parser.add_argument("--live-pointcloud", action="store_true",
                         help="Also log a live Depth-Pro point cloud per frame (bigger .rrd).")
    parser.set_defaults(half=True)
    args = parser.parse_args()

    mcap_path = Path(args.mcap).resolve()
    wilor_dir = Path(args.wilor_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not mcap_path.exists():
        sys.exit(f"MCAP session not found: {mcap_path}")
    if not wilor_dir.is_dir():
        sys.exit(f"WiLoR directory not found: {wilor_dir}")

    device = get_device(args.device)

    # --- Load session (camera pose + RGB intrinsics come straight from MCAP) ---
    rgb_k_path = Path(args.rgb_k).resolve()
    if not rgb_k_path.exists():
        sys.exit(f"rgb_K.npy not found: {rgb_k_path}")
    K = np.load(rgb_k_path).astype(np.float64)
    if K.shape != (3, 3):
        sys.exit(f"rgb_K.npy must be 3x3, got {K.shape}: {rgb_k_path}")

    session = DepthProSession(str(mcap_path), rgb_K=K)
    rgb_i = session.rgb_intrinsics
    if rgb_i is None:
        sys.exit("MCAP session has no RGB camera intrinsics; cannot proceed.")
    n_frames = session.num_rgb_frames
    if args.max_frames:
        n_frames = min(n_frames, args.max_frames)
    f_px = float(K[0, 0])
    logger.info(
        "Session: %d frames, RGB %dx%d; intrinsics from %s: fx=%.2f fy=%.2f cx=%.2f cy=%.2f "
        "(locking Depth Pro f_px=fx)",
        n_frames, rgb_i.width, rgb_i.height, rgb_k_path, K[0, 0], K[1, 1], K[0, 2], K[1, 2],
    )
    if abs(K[0, 2] * 2 - rgb_i.width) > 0.1 * rgb_i.width or abs(K[1, 2] * 2 - rgb_i.height) > 0.1 * rgb_i.height:
        logger.warning("rgb_K principal point is far from the image center of %dx%d; check K matches the RGB resolution.",
                       rgb_i.width, rgb_i.height)

    # --- Load Depth Pro ---
    from depth_pro import create_model_and_transforms
    from depth_pro.depth_pro import DEFAULT_MONODEPTH_CONFIG_DICT

    checkpoint = Path(__file__).resolve().parent / "ml-depth-pro" / "checkpoints" / "depth_pro.pt"
    dp_config = replace(DEFAULT_MONODEPTH_CONFIG_DICT, checkpoint_uri=str(checkpoint))
    precision = torch.half if args.half and device.type == "cuda" else torch.float32
    logger.info("Loading Depth Pro (device=%s, precision=%s)", device, precision)
    dp_model, dp_transform = create_model_and_transforms(config=dp_config, device=device, precision=precision)
    dp_model.eval()
    f_px_tensor = torch.as_tensor(f_px, device=device, dtype=torch.float32)

    # --- Load WiLoR ---
    from stera.models.wilor import WiLoRHandTracker, WiLoRConfig

    wilor_config = WiLoRConfig(wilor_dir=str(wilor_dir), yolo_conf=args.yolo_conf)
    tracker = WiLoRHandTracker(wilor_config)
    tracker.load()

    # --- Rerun visualizer (reuses the SDK's canonical world-frame logger) ---
    from stera.viz import Visualizer

    viz = Visualizer(
        session, output=str(output_dir / "visualization.rrd"),
        map_3d=args.map_3d, max_viz=args.live_pointcloud,
    )

    # --- annotation.hdf5: /depth streamed frame-by-frame (Depth Pro output) ---
    h5_path = output_dir / "annotation.hdf5"
    h5f = h5py.File(h5_path, "w")
    dh, dw = rgb_i.height, rgb_i.width
    depth_grp = h5f.create_group("depth")
    depth_frames_dset = depth_grp.create_dataset(
        "frames", shape=(n_frames, dh, dw), dtype=np.uint16,
        chunks=(1, dh, dw), compression="gzip", compression_opts=4,
    )
    depth_ts_dset = depth_grp.create_dataset("timestamps", shape=(n_frames,), dtype=np.float64)
    depth_valid_dset = depth_grp.create_dataset("valid", shape=(n_frames,), dtype=bool)
    depth_grp.attrs["units"] = "mm"
    depth_grp.attrs["height"] = dh
    depth_grp.attrs["width"] = dw
    depth_grp.attrs["source"] = "depth_pro (replaces low-res MCAP/ARCore depth)"

    logger.info("=== Depth Pro (metric depth @ RGB res, locked fx) + WiLoR (depth-anchored hands) ===")
    n_with_hands = 0
    frame_ts = np.zeros(n_frames, dtype=np.float64)
    pose_valid = np.zeros(n_frames, dtype=bool)
    pose_t = np.zeros((n_frames, 3), dtype=np.float32)
    pose_R = np.tile(np.eye(3, dtype=np.float32), (n_frames, 1, 1))
    for frame in tqdm(session.frames(), total=n_frames, desc="Frames", unit="fr"):
        if frame.index >= n_frames:
            break

        frame_ts[frame.index] = frame.timestamp
        if frame.camera_pose is not None:
            pose_valid[frame.index] = True
            pose_t[frame.index] = frame.camera_pose.translation
            pose_R[frame.index] = frame.camera_pose.rotation

        tensor_image = dp_transform(frame.rgb)
        with torch.no_grad():
            pred = dp_model.infer(tensor_image, f_px=f_px_tensor)
        depth_m = pred["depth"].detach().cpu().numpy().squeeze()
        depth_mm = np.clip(depth_m * 1000.0, 0, 65535).astype(np.uint16)

        # Override the MCAP frame's (low-res) depth in place so the Visualizer's
        # depth colormap / point cloud also uses Depth Pro's output.
        frame.depth = depth_mm

        depth_frames_dset[frame.index] = depth_mm
        depth_ts_dset[frame.index] = float(frame.timestamp)
        depth_valid_dset[frame.index] = True

        hands = tracker.detect_hands(frame.rgb, depth=depth_mm, intrinsics=K)
        session.add_hand_pose(frame.index, hands)
        if hands:
            n_with_hands += 1

        viz.log_frame(frame, hands=hands)

    viz.export(str(output_dir / "visualization.rrd"))

    # --- /cam-pose: MCAP/ARCore pose synced to each RGB frame (same index as
    # /depth and /hand-pose, not the raw pose-topic stream, which may run at
    # a different rate than RGB in general recordings). ---
    logger.info("Writing /cam-pose (from MCAP, synced per RGB frame)")
    cp = h5f.create_group("cam-pose")
    cp.create_dataset("timestamps", data=frame_ts)
    cp.create_dataset("translations", data=pose_t)
    cp.create_dataset("rotations", data=pose_R)
    cp.create_dataset("valid", data=pose_valid)
    cp.attrs["source"] = "mcap (ARCore SLAM/VIO), nearest-neighbor synced to each RGB frame"

    # --- /hand-pose (from WiLoR, depth-anchored) ---
    logger.info("Writing /hand-pose (frames with hands=%d)", n_with_hands)
    from stera.data.export import _hands_to_arrays

    arrs = _hands_to_arrays(session.hand_poses, n_rgb=n_frames)
    hp = h5f.create_group("hand-pose")
    hp.create_dataset("left_joints", data=arrs["left_joints"])
    hp.create_dataset("right_joints", data=arrs["right_joints"])
    hp.create_dataset("left_valid", data=arrs["left_valid"])
    hp.create_dataset("right_valid", data=arrs["right_valid"])
    hp.create_dataset("left_confidence", data=arrs["left_confidence"])
    hp.create_dataset("right_confidence", data=arrs["right_confidence"])
    hp.attrs["coord_frame"] = "camera_3d" if arrs["has_3d"] else "image_2d"
    hp.attrs["source"] = "wilor (depth-anchored with depth_pro)"
    for key in ("left_kpts_2d_rgb", "right_kpts_2d_rgb"):
        if arrs.get(key) is not None:
            hp.create_dataset(key, data=arrs[key])

    meta = h5f.create_group("metadata")
    meta.attrs["num_rgb_frames"] = int(n_frames)
    meta.attrs["depth_source"] = "depth_pro"
    meta.attrs["hand_pose_source"] = "wilor"
    meta.attrs["cam_pose_source"] = "mcap_arcore"

    h5f.close()

    logger.info(
        "Done. %d/%d frames with >=1 detected hand. annotation.hdf5 (depth+cam-pose+hand-pose) "
        "and visualization.rrd in %s", n_with_hands, n_frames, output_dir,
    )


if __name__ == "__main__":
    main()
