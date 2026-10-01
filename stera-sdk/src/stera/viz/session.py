"""Rerun session visualizer for FPV recordings."""

from __future__ import annotations

import logging
from typing import Optional

import cv2
import numpy as np
import rerun as rr
import rerun.blueprint as rrb

from stera.core.transforms import (
    rot_to_quat, optical_to_world, depth_to_pointcloud,
)
from stera.annotations.hands.hand import HandPose

logger = logging.getLogger(__name__)


from stera.processing.mesh import (
    clean_mesh_by_edge_length as _clean_mesh,
    brighten_colors as _brighten_colors,
    compute_vertex_normals as _compute_vertex_normals,
)


HAND_EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
    (5, 9), (9, 13), (13, 17),
]


def _log_imu_samples(imu_samples) -> None:
    """Stream N IMU samples as 6 scalar time-series.

    Prefers bulk columnar logging via ``rr.send_columns`` when available for
    speed; falls back to per-sample ``rr.log`` on older rerun versions.
    """
    if not imu_samples:
        return
    timestamps = np.array([ts for ts, _ in imu_samples], dtype=np.float64)
    accel = np.stack([d.get("linear_acceleration", np.zeros(3))
                      for _, d in imu_samples], axis=0).astype(np.float32)
    gyro = np.stack([d.get("angular_velocity", np.zeros(3))
                     for _, d in imu_samples], axis=0).astype(np.float32)
    axes = ("x", "y", "z")

    # Columnar bulk API (rerun 0.22+): one ~6-ms round-trip per entity.
    if hasattr(rr, "send_columns") and hasattr(rr, "TimeColumn"):
        try:
            time_col = rr.TimeColumn("time", timestamp=timestamps)
            for ax, i in zip(axes, range(3)):
                rr.send_columns(
                    f"/imu/accel/{ax}",
                    indexes=[time_col],
                    columns=rr.Scalars.columns(scalars=accel[:, i]),
                )
                rr.send_columns(
                    f"/imu/gyro/{ax}",
                    indexes=[time_col],
                    columns=rr.Scalars.columns(scalars=gyro[:, i]),
                )
            return
        except Exception as e:
            print(f"[viz] imu columnar log failed ({e}); falling back per-sample")

    # Slow fallback: one log call per sample. Fine for shorter mcaps.
    for (ts, _), a, g in zip(imu_samples, accel, gyro):
        rr.set_time("time", timestamp=float(ts))
        for ax, v in zip(axes, a):
            rr.log(f"/imu/accel/{ax}", rr.Scalars([float(v)]))
        for ax, v in zip(axes, g):
            rr.log(f"/imu/gyro/{ax}", rr.Scalars([float(v)]))


# Yellow frustum (world/frustum) length = p90 depth * _FRUSTUM_SCALE, clamped to [MIN, MAX] (m)
_FRUSTUM_SCALE, _FRUSTUM_MIN, _FRUSTUM_MAX = 0.1, 0.1, 0.3
# 3D hand skeleton radii (m), used by both world/skeleton and world/hands/*
_BONE_RADIUS, _JOINT_RADIUS = 0.001, 0.003
# Estimated (not measured) body skeleton, average human proportions (m). Visualization only.
_BODY = dict(neck_down=0.12, shoulder_half=0.20, torso=0.50, hip_half=0.10, upper_arm=0.30, forearm=0.27)
# Map points farther than this from the world origin are dropped (m)
_MAP_MAX_RADIUS = 5.0
# Rerun Pinhole frustum (camera/optical/pinhole) image-plane distance (m)
_PINHOLE_DIST = 0.1
_GROUND_ZMIN, _GROUND_BIN, _GROUND_BINS, _GROUND_EXTENT = -5.0, 0.02, 500, 1.0


class Visualizer:
    """Rerun visualizer for an FPV MCAP session with hand tracking.

    Handles all Rerun setup, blueprint, static data, and per-frame logging.

    Usage::

        from stera.data import MCAPReader
        from stera.viz import Visualizer

        session = MCAPReader("recording.mcap")
        viz = Visualizer(session, output="viz.rrd")
        for frame in session.frames():
            viz.log_frame(frame, hands=hand_poses)  # setup runs lazily
    """

    def __init__(
        self,
        session,
        output: Optional[str] = None,
        app_id: str = "stera",
        max_viz: bool = False,
        overlay_size: tuple[int, int] | None = None,
        jpeg_quality: int = 50,
        trail_len: int = 30,
        frustum_depth: float = 0.3,
        adaptive_frustum: bool = True,
        ground_plane: bool = True,
        body_skeleton: bool = True,
        live_pc_max: int = 50_000,
        live_pc_radius: float = 0.005,
        max_map_points: int = 200_000,
        dense_map: bool = False,
        dense_every_n: int = 10,
        dense_cam_exclude: float = 1.0,
        dense_voxel_size: float = 0.02,
        map_3d: str = "auto",
    ):
        """Build the visualizer.

        ``map_3d`` controls which map representation is logged into the 3D
        scene at setup time:

        - ``"auto"`` (default): triangle mesh from ``/map/mesh`` if present,
          otherwise the point cloud (``/map/mesh_cloud`` → ``/map/point_cloud``
          fallback chain).
        - ``"mesh"``: only log the triangle mesh; nothing if the topic
          is missing.
        - ``"mesh_cloud"``: log the accumulated point cloud from
          ``/map/mesh_cloud`` only.
        - ``"point_cloud"``: log the raw point cloud from
          ``/map/point_cloud`` only.
        - ``"both"``: log BOTH the mesh and the point cloud. The point cloud
          is hidden by default in the viewer blueprint (toggle from the
          entity tree to compare against the mesh).
        - ``"none"``: skip the 3D map entirely.

        Whenever a mesh is logged (modes ``"auto"``/``"mesh"``/``"both"``)
        the raw, unrefined mesh from ``/map/mesh`` is also logged to
        ``world/mesh_raw`` and starts hidden in the blueprint — kept around
        as an A/B reference against the cleaned mesh.
        """
        valid_map_3d = {"auto", "mesh", "mesh_cloud", "point_cloud", "both", "none"}
        if map_3d not in valid_map_3d:
            raise ValueError(
                f"map_3d={map_3d!r}: expected one of {sorted(valid_map_3d)}"
            )
        self._session = session
        if output is None:
            import tempfile
            # Dummy backing file; user promotes to the final path via .export().
            fd = tempfile.NamedTemporaryFile(suffix=".rrd", delete=False,
                                             prefix="stera_viz_")
            fd.close()
            output = fd.name
        self._output = output
        self._app_id = app_id
        self._max_viz = max_viz
        self._overlay_size = overlay_size
        self._jpeg_quality = jpeg_quality
        self._trail_len = trail_len
        self._frustum_depth = frustum_depth
        self._adaptive_frustum = adaptive_frustum
        self._ground_plane = ground_plane
        self._body_skeleton = body_skeleton
        self._body_fwd: Optional[np.ndarray] = None  # EMA of horizontal heading
        self._live_pc_max = live_pc_max
        self._live_pc_radius = live_pc_radius
        self._max_map_points = max_map_points
        self._dense_map = dense_map
        self._dense_every_n = dense_every_n
        self._dense_cam_exclude = dense_cam_exclude
        self._dense_voxel_size = dense_voxel_size
        self._map_3d = map_3d

        # State
        self._trail: list[list[float]] = []
        self._fc_link: Optional[np.ndarray] = None
        self._fc_unit_link: Optional[np.ndarray] = None  # frustum corners at depth 1 m
        self._frustum_d: Optional[float] = None  # EMA of scene depth
        self._ground_hist = np.zeros(_GROUND_BINS, dtype=np.int64)
        self._ground_z: Optional[float] = None
        self._ground_drawn_z: Optional[float] = None
        self._frame_i = 0
        self._R_o2l: Optional[np.ndarray] = None
        self._setup_done = False
        self._pointcloud_logged = False
        self._mesh_raw_logged = False

    def export(self, path: str) -> str:
        """Flush pending logs and copy the backing .rrd to ``path``.

        When no ``output`` was passed at construction, the visualizer streams
        into a temp file; this call promotes that temp file to the final
        destination the caller cares about.
        """
        import shutil
        from pathlib import Path as _P
        logger.info("Flushing Rerun stream")
        try:
            rr.flush_blocking()
        except Exception:
            pass
        src = _P(self._output)
        dst = _P(path)
        if src.resolve() == dst.resolve():
            logger.info("Visualization saved to %s", dst)
            return str(dst)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dst)
        logger.info("Visualization saved to %s", dst)
        return str(dst)

    def show(self, block: bool = False) -> None:
        """Open the saved .rrd in the Rerun viewer.

        Call after the frame loop finishes. ``block=True`` waits for the
        viewer to exit; the default returns immediately so the main script
        can keep running.
        """
        import shutil
        import subprocess
        try:
            rr.flush_blocking()
        except Exception:
            pass
        if shutil.which("rerun") is None:
            raise RuntimeError(
                "`rerun` CLI not found on PATH. Install rerun-sdk or run "
                f"`rerun {self._output}` manually."
            )
        runner = subprocess.run if block else subprocess.Popen
        runner(["rerun", self._output])

    def setup(self) -> None:
        """Initialize Rerun, send blueprint, log static data."""
        logger.info("Setting up Rerun visualizer (output=%s)", self._output)
        rr.init(self._app_id)
        rr.save(self._output)

        # LineGrid3D / EyeControls3D only exist on rerun-sdk 0.24+; pinning
        # numpy<2 (needed for smplx/chumpy) forces us onto 0.22.x in some
        # envs, so feature-detect and omit the extras there.
        view3d_kwargs: dict = {
            "name": "3D Scene",
            "origin": "/world",
            "background": rrb.Background(color=[30, 30, 30]),
        }
        # LineGrid3D lives in rrb.archetypes on 0.23.x, top-level on newer SDKs
        _line_grid = getattr(rrb, "LineGrid3D", None) or getattr(
            getattr(rrb, "archetypes", None), "LineGrid3D", None)
        if _line_grid is not None:
            view3d_kwargs["line_grid"] = _line_grid(visible=False)
        if hasattr(rrb, "EyeControls3D") and hasattr(rrb, "components") and hasattr(rrb.components, "Eye3DKind"):
            view3d_kwargs["eye_controls"] = rrb.EyeControls3D(
                kind=rrb.components.Eye3DKind.Orbital,
                tracking_entity="/world/frustum",
                position=[0.0, 1.5, 0.0],
                eye_up=[0.0, 1.0, 0.0],
            )
        # Per-entity visibility overrides via EntityBehavior. Entities stay
        # in the view's contents (so they appear in the blueprint tree with
        # an eye-icon toggle) but are hidden by default.
        #   • world/mesh_raw     → always hidden (A/B reference for refined mesh)
        #   • world/pointcloud   → hidden when mode='both' (mesh wins by default)
        if hasattr(rrb, "EntityBehavior"):
            hide = {}
            if self._map_3d in ("auto", "mesh", "both"):
                hide["/world/mesh_raw"] = rrb.EntityBehavior(visible=False)
            if self._map_3d == "both":
                hide["/world/pointcloud"] = rrb.EntityBehavior(visible=False)
            if hide:
                view3d_kwargs["overrides"] = hide

        try:
            # Left column: RGB, RGB+Hands, Depth.
            left_col_children: list = [
                rrb.Spatial2DView(name="RGB", origin="/camera/rgb"),
                rrb.Spatial2DView(name="RGB + Hands", origin="/camera/rgb_overlay"),
                rrb.Spatial2DView(name="Depth", origin="/camera/depth"),
            ]
            left_col = rrb.Vertical(*left_col_children)

            # Right column: big 3D scene on top, IMU accel + gyro side-by-side beneath.
            scene_view = rrb.Spatial3DView(**view3d_kwargs)
            if hasattr(rrb, "TimeSeriesView"):
                imu_row = rrb.Horizontal(
                    rrb.TimeSeriesView(name="IMU accel (m/s²)", origin="/imu/accel"),
                    rrb.TimeSeriesView(name="IMU gyro (rad/s)", origin="/imu/gyro"),
                )
                right_col = rrb.Vertical(scene_view, imu_row, row_shares=[4, 1])
            else:
                right_col = scene_view

            layout = rrb.Horizontal(
                left_col, right_col, column_shares=[1, 2],
            )

            # Panels default-collapsed so the viewer opens with the image
            # + 3D + IMU panels filling the viewport; user can expand any of
            # them from the toolbar.
            def _try(fn, *args, **kwargs):
                try:
                    return fn(*args, **kwargs)
                except Exception:
                    return None

            panels: list = []
            tp = _try(rrb.TimePanel, timeline="time", playback_speed=1.0,
                      state="collapsed")
            if tp is None:
                tp = _try(rrb.TimePanel, state="collapsed")
            if tp is None:
                tp = _try(rrb.TimePanel, timeline="time", playback_speed=1.0)
            if tp is None:
                tp = _try(rrb.TimePanel)
            if tp is not None:
                panels.append(tp)

            if hasattr(rrb, "BlueprintPanel"):
                bp = _try(rrb.BlueprintPanel, state="collapsed")
                if bp is None:
                    bp = _try(rrb.BlueprintPanel)
                if bp is not None:
                    panels.append(bp)

            if hasattr(rrb, "SelectionPanel"):
                sp = _try(rrb.SelectionPanel, state="collapsed")
                if sp is None:
                    sp = _try(rrb.SelectionPanel)
                if sp is not None:
                    panels.append(sp)

            rr.send_blueprint(rrb.Blueprint(layout, *panels))
        except Exception as e:
            # Blueprint customization is cosmetic; fall back to default layout.
            logger.warning("Rerun blueprint skipped (%s: %s)", type(e).__name__, e)

        # Tell Rerun this is a Y-up coordinate system (ARCore world frame, gravity along Y)
        rr.log("/", rr.ViewCoordinates.RIGHT_HAND_Y_UP, static=True)
        rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Y_UP, static=True)

        # Static: map mesh / point cloud (selectable via map_3d)
        self._log_map_3d()

        # Static: optical frame transform + pinhole (read actual TF from MCAP)
        self._R_o2l = self._session.R_optical_to_link
        R_o2l = self._R_o2l
        R_l2o = R_o2l.T  # link -> optical for Rerun child transform
        qx, qy, qz, qw = rot_to_quat(R_l2o)
        rr.log("world/camera/optical", rr.Transform3D(
            quaternion=[qx, qy, qz, qw],
            axis_length=0.1,  # optical axes: X red=right, Y green=down, Z blue=forward
        ), static=True)

        depth_intr = self._session.depth_intrinsics
        if depth_intr:
            K = depth_intr.K
            rr.log("world/camera/optical/pinhole", rr.Pinhole(
                focal_length=[K[0, 0], K[1, 1]],
                principal_point=[K[0, 2], K[1, 2]],
                resolution=[depth_intr.width, depth_intr.height],
                image_plane_distance=_PINHOLE_DIST,
            ), static=True)

            # Pre-compute frustum corners in link frame
            fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
            d = self._frustum_depth
            fc_opt = np.array([
                [(0 - cx) / fx * d, (0 - cy) / fy * d, d],
                [(depth_intr.width - cx) / fx * d, (0 - cy) / fy * d, d],
                [(depth_intr.width - cx) / fx * d, (depth_intr.height - cy) / fy * d, d],
                [(0 - cx) / fx * d, (depth_intr.height - cy) / fy * d, d],
            ])
            self._fc_link = (R_o2l @ fc_opt.T).T
            self._fc_unit_link = self._fc_link / d

        # IMU: log every sample at full mcap rate as 6 scalar streams under
        # /imu/accel/{x,y,z} and /imu/gyro/{x,y,z}. Done once at setup so the
        # per-frame log_frame loop stays cheap.
        try:
            imu_samples = self._session.all_imu_samples()
        except Exception as e:
            imu_samples = []
            logger.warning("IMU load skipped (%s: %s)", type(e).__name__, e)

        if imu_samples:
            logger.info("Logging %d IMU samples to Rerun", len(imu_samples))
            _log_imu_samples(imu_samples)

        self._setup_done = True
        logger.info("Rerun setup complete")

    def run(
        self,
        hand_tracker=None,
        skeleton_estimator=None,
        enable_pointcloud: Optional[bool] = None,
    ) -> None:
        """Process all frames at once with detection and visualization.

        Parameters
        ----------
        hand_tracker : HandTracker instance (optional).
        skeleton_estimator : UpperBodyEstimator instance (optional).
        enable_pointcloud : Log live RGB-D point cloud per frame. Defaults to max_viz.
        """
        from tqdm import tqdm

        if not self._setup_done:
            self.setup()

        # Ensure skeleton estimator has the correct optical-to-link transform
        if skeleton_estimator is not None and skeleton_estimator._R_o2l is None:
            skeleton_estimator._R_o2l = self._R_o2l

        depth_K = self._session.depth_intrinsics
        depth_K = depth_K.K if depth_K else None

        total = self._session.num_rgb_frames
        for frame in tqdm(self._session.frames(), total=total, desc="Visualizing", unit="frame"):
            hands = None
            if hand_tracker is not None:
                hands = hand_tracker.detect(frame.rgb, depth=frame.depth, intrinsics=depth_K)

            skeleton = None
            if skeleton_estimator is not None:
                skeleton = skeleton_estimator.estimate(frame, hands=hands)

            self.log_frame(frame, hands=hands, skeleton=skeleton,
                           enable_pointcloud=enable_pointcloud)

    def _log_map_3d(self) -> None:
        """Log the chosen 3D map representation to ``world/`` (static)."""
        mode = self._map_3d
        if mode == "none":
            return

        logger.info("Logging 3D map (mode=%s)", mode)

        mesh_logged = False
        if mode in ("auto", "mesh", "both"):
            mesh_data = self._session.mesh()
            if mesh_data is not None:
                # Always also log the raw mesh under world/mesh_raw — hidden
                # by default in the blueprint, available as an A/B reference.
                raw_verts, raw_faces, raw_colors = mesh_data
                raw_kw = dict(
                    vertex_positions=raw_verts,
                    triangle_indices=raw_faces,
                )
                if raw_colors is not None:
                    raw_kw["vertex_colors"] = raw_colors
                rr.log("world/mesh_raw", rr.Mesh3D(**raw_kw), static=True)
                self._mesh_raw_logged = True
                logger.info("Logged raw mesh (hidden): %d verts / %d tris",
                            len(raw_verts), len(raw_faces))

                verts, faces, colors = mesh_data
                faces = _clean_mesh(verts, faces)
                if colors is None:
                    logger.info("Coloring mesh from camera frames")
                    colors = self._session.color_mesh(verts)
                colors = _brighten_colors(colors)
                normals = _compute_vertex_normals(verts, faces)
                logger.info("Logged mesh: %d verts / %d tris", len(verts), len(faces))
                rr.log("world/mesh", rr.Mesh3D(
                    vertex_positions=verts,
                    triangle_indices=faces,
                    vertex_colors=colors,
                    vertex_normals=normals,
                ), static=True)
                mesh_logged = True
                if mode in ("auto", "mesh"):
                    return
            elif mode == "mesh":
                logger.warning("map_3d='mesh' but /map/mesh is empty; skipping 3D map")
                return
            elif mode == "both":
                logger.warning("map_3d='both' but /map/mesh is empty; only point cloud will be logged")

        # Point-cloud branch (auto fallback, mesh_cloud, point_cloud, or 'both').
        if self._dense_map:
            logger.info("Building dense point cloud from depth frames")
            xyz, rgb_colors = self._session.dense_point_cloud(
                every_n=self._dense_every_n,
                cam_exclude_radius=self._dense_cam_exclude,
                voxel_size=self._dense_voxel_size,
            )
            logger.info("Dense cloud: %d points after voxel downsample", len(xyz))
        else:
            # 'both' shares the auto fallback chain (mesh_cloud → point_cloud).
            pc_source = "auto" if mode in ("auto", "both") else mode
            xyz, rgb_colors = self._session.point_cloud(source=pc_source)

        # Drop far-away map points: SLAM outliers (tens of metres away) blow up
        # the viewer's auto-fit bounds and make the actual scene look tiny.
        if len(xyz) > 0:
            near = np.linalg.norm(xyz, axis=1) <= _MAP_MAX_RADIUS
            if not near.all():
                logger.info("Dropped %d/%d map points beyond %.1f m of the origin",
                            int((~near).sum()), len(xyz), _MAP_MAX_RADIUS)
                xyz = xyz[near]
                if rgb_colors is not None:
                    rgb_colors = rgb_colors[near]

        if len(xyz) > self._max_map_points:
            idx = np.random.default_rng(42).choice(
                len(xyz), self._max_map_points, replace=False,
            )
            xyz = xyz[idx]
            if rgb_colors is not None:
                rgb_colors = rgb_colors[idx]

        if len(xyz) > 0:
            logger.info("Logged pointcloud: %d points", len(xyz))
            rr.log("world/pointcloud", rr.Points3D(
                xyz,
                colors=rgb_colors if rgb_colors is not None else [80, 80, 90],
                radii=0.004,
            ), static=True)
            self._pointcloud_logged = True
        elif mode in ("mesh_cloud", "point_cloud"):
            logger.warning("map_3d=%r but topic is empty; skipping 3D map", mode)
        elif mode == "both" and not mesh_logged:
            logger.warning("map_3d='both' but neither mesh nor point cloud available")

    def log_frame(
        self,
        frame,
        hands: Optional[list[HandPose]] = None,
        skeleton=None,
        enable_pointcloud: Optional[bool] = None,
    ) -> None:
        """Log a single synchronized frame to Rerun.

        Parameters
        ----------
        frame : SyncedFrame from MCAPReader.frames().
        hands : Full 21-joint HandPose list (from WiLoR finger tracking).
        skeleton : SkeletonFrame from UpperBodyEstimator (optional).
        enable_pointcloud : Log live RGB-D point cloud. Defaults to max_viz setting.
        """
        if enable_pointcloud is None:
            enable_pointcloud = self._max_viz

        if not self._setup_done:
            self.setup()

        rr.set_time("time", timestamp=frame.timestamp)

        rgb_intr = self._session.rgb_intrinsics
        depth_intr = self._session.depth_intrinsics

        # RGB overlay
        rgb_bgr = cv2.cvtColor(frame.rgb, cv2.COLOR_RGB2BGR)
        # Plain RGB (no hand drawings). Encoded the same way as the overlay
        # view so file size stays comparable.
        _, jpg_plain = cv2.imencode(
            ".jpg", rgb_bgr, [cv2.IMWRITE_JPEG_QUALITY, self._jpeg_quality],
        )
        rr.log(
            "camera/rgb",
            rr.EncodedImage(contents=jpg_plain.tobytes(), media_type="image/jpeg"),
        )

        if self._overlay_size is not None:
            overlay = cv2.resize(rgb_bgr, self._overlay_size)
        else:
            overlay = rgb_bgr.copy()
        oh, ow = overlay.shape[:2]

        # Depth colormap
        if frame.depth is not None:
            d_norm = np.clip(frame.depth.astype(np.float32) / 2000.0, 0, 1)
            d_color = cv2.applyColorMap((d_norm * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
            # JPEG at the same quality knob as RGB; this is a viz colormap.
            _, jpg_d = cv2.imencode(
                ".jpg", d_color, [cv2.IMWRITE_JPEG_QUALITY, self._jpeg_quality],
            )
            rr.log("camera/depth", rr.EncodedImage(contents=jpg_d.tobytes(), media_type="image/jpeg"))

        # Camera pose
        R_map = t_map = None
        if frame.camera_pose is not None:
            pose = frame.camera_pose
            R_map = pose.rotation
            t_map = pose.translation
            qx, qy, qz, qw = rot_to_quat(R_map)

            rr.log("world/camera", rr.Transform3D(
                translation=t_map,
                quaternion=[qx, qy, qz, qw],
            ))

            # Frustum
            if self._fc_link is not None:
                fc_link = self._fc_link
                if self._adaptive_frustum and self._fc_unit_link is not None:
                    d_scene = self._scene_depth(frame.depth)
                    if d_scene is not None:
                        fc_link = self._fc_unit_link * d_scene
                fc_w = (R_map @ fc_link.T).T + t_map
                lines = [[t_map.tolist(), fc_w[j].tolist()] for j in range(4)] + \
                        [[fc_w[j].tolist(), fc_w[(j + 1) % 4].tolist()] for j in range(4)]
                rr.log("world/frustum", rr.LineStrips3D(lines, colors=[[255, 255, 0]], radii=[0.004]))

            # Camera trail
            self._trail.append(t_map.tolist())
            tw = self._trail[-self._trail_len:]
            if len(tw) >= 2:
                rr.log("world/trail", rr.LineStrips3D([tw], colors=[[0, 220, 240]], radii=[0.006]))

        # Hands
        # camera/rgb_overlay is plain RGB; hands are logged as native rerun
        # 2D entities (Points2D + LineStrips2D) under it.
        _, jpg_o = cv2.imencode(
            ".jpg", rgb_bgr, [cv2.IMWRITE_JPEG_QUALITY, self._jpeg_quality],
        )
        rr.log(
            "camera/rgb_overlay",
            rr.EncodedImage(contents=jpg_o.tobytes(), media_type="image/jpeg"),
        )

        hand_wrists: dict[str, np.ndarray] = {}
        visible = set()
        visible_2d = set()  # sides that have 2D overlay logged this frame
        if hands:
            for hp in hands:
                kps = hp.all_keypoints
                joints_3d_world_src = np.array(
                    [[kp.x, kp.y, kp.z] for kp in kps]
                )
                has_3d = kps[0].z != 0.0

                if has_3d and R_map is not None:
                    joints_world = optical_to_world(
                        joints_3d_world_src, R_map, t_map, self._R_o2l,
                    )
                    self._log_hand_3d(hp.hand_side, joints_world, hp.confidence)
                    hand_wrists[hp.hand_side] = joints_world[0]
                    visible.add(hp.hand_side)

                # 2D overlay on /camera/rgb_overlay. Prefer the tracker's
                # pixel-space keypoints (hp._kpts_2d_rgb) which live in the
                # same frame as the stored RGB.
                kp_2d = getattr(hp, "_kpts_2d_rgb", None)
                if kp_2d is None and not has_3d:
                    # Legacy fallback: Keypoint.x/y carries pixel coords
                    # when no 3D was available.
                    kp_2d = joints_3d_world_src[:, :2]
                if kp_2d is not None and len(kp_2d) == 21:
                    kp_2d_draw = kp_2d
                    color = [255, 120, 120] if hp.hand_side == "left" else [120, 255, 120]
                    rr.log(
                        f"camera/rgb_overlay/hand_{hp.hand_side}/joints",
                        rr.Points2D(kp_2d_draw, colors=[color], radii=3.0),
                    )
                    strips = [
                        [kp_2d_draw[a].tolist(), kp_2d_draw[b].tolist()]
                        for a, b in HAND_EDGES
                    ]
                    rr.log(
                        f"camera/rgb_overlay/hand_{hp.hand_side}/bones",
                        rr.LineStrips2D(strips, colors=[color], radii=1.5),
                    )
                    visible_2d.add(hp.hand_side)

        # Clear absent hands (both 3D and 2D entities).
        for ht in ("left", "right"):
            if ht not in visible:
                rr.log(f"world/hands/{ht}/joints", rr.Clear(recursive=False))
                rr.log(f"world/hands/{ht}/bones", rr.Clear(recursive=False))
            if ht not in visible_2d:
                rr.log(f"camera/rgb_overlay/hand_{ht}/joints",
                       rr.Clear(recursive=False))
                rr.log(f"camera/rgb_overlay/hand_{ht}/bones",
                       rr.Clear(recursive=False))

        if self._body_skeleton:
            self._update_body(R_map, t_map, hand_wrists)

        # Skeleton
        if skeleton is not None:
            bone_lines = skeleton.bone_lines()
            if bone_lines:
                rr.log("world/skeleton/bones", rr.LineStrips3D(
                    bone_lines, colors=[[255, 200, 0]], radii=_BONE_RADIUS,
                ))
            vis_joints = skeleton.visible_joints()
            if len(vis_joints) > 0:
                rr.log("world/skeleton/joints", rr.Points3D(
                    vis_joints, colors=[[255, 200, 0]], radii=_JOINT_RADIUS,
                ))
        else:
            rr.log("world/skeleton/bones", rr.Clear(recursive=False))
            rr.log("world/skeleton/joints", rr.Clear(recursive=False))

        if self._ground_plane and R_map is not None:
            self._update_ground(frame, R_map, t_map, depth_intr)

        # Live RGB-D point cloud
        if enable_pointcloud and frame.depth is not None and R_map is not None and depth_intr:
            pts_cam, colors_pc = depth_to_pointcloud(
                frame.depth, frame.rgb, depth_intr.K, max_pts=self._live_pc_max,
            )
            if len(pts_cam) > 0:
                pts_world = optical_to_world(pts_cam, R_map, t_map, self._R_o2l)
                rr.log("world/live_pointcloud", rr.Points3D(
                    pts_world, colors=colors_pc, radii=self._live_pc_radius,
                ))

    # Private helpers

    def _update_body(self, R_map, t_map, wrists: dict) -> None:
        """Draw an *estimated* head/torso/arm skeleton (average proportions).

        Head = camera position; heading from the camera view direction; shoulders
        at fixed offsets; elbows from 2-bone IK toward the WiLoR wrist positions.
        Assumes a Y-up world. Not measured; visualization only.
        """
        if R_map is None:
            rr.log("world/body/bones", rr.Clear(recursive=False))
            rr.log("world/body/joints", rr.Clear(recursive=False))
            return
        up = np.array([0.0, 1.0, 0.0])
        o = optical_to_world(np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]]),
                             R_map, t_map, self._R_o2l)
        head, fwd, img_up = o[0], o[1] - o[0], o[2] - o[0]
        # fwd + image-up stays well-defined whether looking level or straight down
        h = (fwd + img_up) * np.array([1.0, 0.0, 1.0])
        n = np.linalg.norm(h)
        if n < 1e-6:
            return
        h /= n
        if self._body_fwd is not None:
            h = 0.85 * self._body_fwd + 0.15 * h
            h /= max(np.linalg.norm(h), 1e-9)
        self._body_fwd = h
        right = np.cross(h, up)
        B = _BODY
        neck = head - up * B["neck_down"]
        sh = {"left": neck - right * B["shoulder_half"], "right": neck + right * B["shoulder_half"]}
        pelvis = neck - up * B["torso"]
        hip = {"left": pelvis - right * B["hip_half"], "right": pelvis + right * B["hip_half"]}
        joints = {"head": head, "neck": neck, "pelvis": pelvis,
                  "sh_l": sh["left"], "sh_r": sh["right"], "hip_l": hip["left"], "hip_r": hip["right"]}
        edges = [("head", "neck"), ("sh_l", "sh_r"), ("neck", "pelvis"), ("hip_l", "hip_r"),
                 ("sh_l", "hip_l"), ("sh_r", "hip_r")]
        for side, w in wrists.items():
            S = sh[side]
            v = w - S
            dist = float(np.linalg.norm(v))
            if dist < 1e-6:
                continue
            u = v / dist
            L1, L2 = B["upper_arm"], B["forearm"]
            d = float(np.clip(dist, abs(L1 - L2) + 1e-3, L1 + L2 - 1e-3))
            a = (L1 * L1 - L2 * L2 + d * d) / (2 * d)
            hh = np.sqrt(max(L1 * L1 - a * a, 0.0))
            out = -right if side == "left" else right
            pole = -up + 0.5 * out          # elbow points down and slightly outward
            pole -= np.dot(pole, u) * u
            pn = np.linalg.norm(pole)
            pole = pole / pn if pn > 1e-6 else -up
            elbow = S + a * u + hh * pole
            joints[f"el_{side}"], joints[f"wr_{side}"] = elbow, w
            edges += [("sh_l" if side == "left" else "sh_r", f"el_{side}"), (f"el_{side}", f"wr_{side}")]
        rr.log("world/body/bones", rr.LineStrips3D(
            [[joints[a].tolist(), joints[b].tolist()] for a, b in edges],
            colors=[[80, 160, 255]], radii=_BONE_RADIUS * 2))
        rr.log("world/body/joints", rr.Points3D(
            np.array(list(joints.values())), colors=[[80, 160, 255]], radii=_JOINT_RADIUS * 1.5))

    def _scene_depth(self, depth: Optional[np.ndarray]) -> Optional[float]:
        """Smoothed, scaled 90th-percentile depth (m) used as frustum length."""
        if depth is None:
            return self._frustum_d
        z = depth[::8, ::8].astype(np.float32).ravel() / 1000.0
        z = z[(z > 0.3) & (z < 5.0)]
        if len(z) < 50:
            return self._frustum_d
        d = float(np.clip(np.percentile(z, 90) * _FRUSTUM_SCALE, _FRUSTUM_MIN, _FRUSTUM_MAX))
        self._frustum_d = d if self._frustum_d is None else 0.9 * self._frustum_d + 0.1 * d
        return self._frustum_d

    def _update_ground(self, frame, R_map, t_map, depth_intr) -> None:
        """Estimate a horizontal ground plane from Depth Pro points and draw it.

        Assumes the world frame is Y-up (gravity-aligned SLAM). Accumulates a
        histogram of world Y for points below the camera; the ground is the
        lowest well-supported peak (>= 50% of the strongest one).
        """
        self._frame_i += 1
        if frame.depth is None or depth_intr is None or self._frame_i % 5:
            return
        pts_cam, _ = depth_to_pointcloud(frame.depth, frame.rgb, depth_intr.K, max_pts=5000)
        if len(pts_cam) == 0:
            return
        z = optical_to_world(pts_cam, R_map, t_map, self._R_o2l)[:, 1]
        z = z[z < t_map[1] - 0.3]
        idx = np.floor((z - _GROUND_ZMIN) / _GROUND_BIN).astype(int)
        idx = idx[(idx >= 0) & (idx < _GROUND_BINS)]
        if len(idx) == 0:
            return
        self._ground_hist += np.bincount(idx, minlength=_GROUND_BINS)
        h = np.convolve(self._ground_hist, np.ones(3), mode="same")
        peaks = np.nonzero(h >= 0.5 * h.max())[0]
        gz = _GROUND_ZMIN + (peaks[0] + 0.5) * _GROUND_BIN
        self._ground_z = gz if self._ground_z is None else 0.8 * self._ground_z + 0.2 * gz
        if self._ground_drawn_z is not None and abs(self._ground_z - self._ground_drawn_z) < 0.01 \
                and self._frame_i % 30:
            return
        self._ground_drawn_z = self._ground_z
        cx, cy = round(float(t_map[0])), round(float(t_map[2]))
        ticks = np.arange(-_GROUND_EXTENT, _GROUND_EXTENT + 1e-6, 0.25)
        lines = [[[cx + a, self._ground_z, cy - _GROUND_EXTENT], [cx + a, self._ground_z, cy + _GROUND_EXTENT]] for a in ticks]
        lines += [[[cx - _GROUND_EXTENT, self._ground_z, cy + a], [cx + _GROUND_EXTENT, self._ground_z, cy + a]] for a in ticks]
        rr.log("world/ground", rr.LineStrips3D(lines, colors=[[90, 90, 90]], radii=[0.002]))

    def _log_hand_3d(self, hand_side: str, joints_world: np.ndarray, confidence: float) -> None:
        """Log a full 21-joint hand skeleton in world frame."""
        color = [255, 100, 100] if hand_side == "left" else [100, 255, 100]

        rr.log(f"world/hands/{hand_side}/joints", rr.Points3D(
            joints_world, colors=[color], radii=_JOINT_RADIUS,
        ))
        strips = [[joints_world[a].tolist(), joints_world[b].tolist()] for a, b in HAND_EDGES]
        rr.log(f"world/hands/{hand_side}/bones", rr.LineStrips3D(
            strips, colors=[color], radii=_BONE_RADIUS,
        ))

    def _draw_hand_overlay(self, overlay, joints, hand_side, confidence, img_w, img_h, ow, oh):
        """Draw 21-joint hand skeleton on the 2D overlay."""
        col_bgr = (100, 100, 255) if hand_side == "left" else (100, 255, 100)
        pts_2d = np.column_stack([
            joints[:, 0] * ow / img_w,
            joints[:, 1] * oh / img_h,
        ]).astype(int)
        for a, b in HAND_EDGES:
            cv2.line(overlay, tuple(pts_2d[a]), tuple(pts_2d[b]), col_bgr, 2)
        for j in range(21):
            cv2.circle(overlay, tuple(pts_2d[j]), 6, col_bgr, -1)
        cv2.putText(overlay, f"{confidence:.2f}",
                    (pts_2d[0, 0] + 5, pts_2d[0, 1] - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)



# Back-compat alias — older code still imports RerunVisualizer.
RerunVisualizer = Visualizer
