#!/usr/bin/env python3
"""Step 2: 轉換 .mcap -> Depth Pro 深度圖 -> WiLoR 手部姿態 -> wrist Kalman filter。

包裝 stera-sdk/pipeline_mcap_depth_wilor.py。輸出 (annotation.hdf5 含
/depth /cam-pose /hand-pose,以及 visualization.rrd) 到 data/processed/<session>/。

之後對每隻手的 wrist (joint 0) 跑 constant-velocity Kalman filter:
  - 在 world frame 濾波 (用 /cam-pose 扣掉頭部/相機運動),量測雜訊在相機座標
    為各向異性 (depth 方向較大),並依 WiLoR confidence 縮放
  - Mahalanobis gating 剔除短暫的錯誤偵測 (outlier),以預測值取代
  - 短暫漏偵測 (<= --kf-max-gap 秒) 以 CV 模型外插預測補上
  - 預設再做 RTS smoother (離線,前後向),--kf-causal 則只用前向濾波
結果寫入 /hand-pose/{side}_*_kf,原始資料不變。

用法: python3 02_depth_wilor.py [session_dir] [--max-frames N] ...
      python3 02_depth_wilor.py [session_dir] --kf-only   # 只重跑 filter
未指定 session_dir 時使用 data/raw 下最新的 session。
"""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parent
SDK = ROOT / "stera-sdk"
K_PATH = SDK / "ml-depth-pro" / "cam" / "rgb_K.npy"  # RGB 內參 (kpts_2d_rgb 所在的 1920x1080 影像)

# /hand-pose/{side}_kf_status 的值
KF_NONE, KF_MEASURED, KF_PREDICTED, KF_REJECTED = 0, 1, 2, 3
CHI2_3DOF = {0.95: 7.815, 0.99: 11.345, 0.997: 14.156, 0.999: 16.266}


def latest_session(raw_root: Path) -> Path:
    sessions = sorted(p for p in raw_root.glob("session_*") if p.is_dir())
    if not sessions:
        sys.exit(f"{raw_root} 下沒有 session,請先執行 01_pull_latest.py")
    return sessions[-1]


def sdk_env() -> dict:
    # editable 安裝的 depth_pro 可能指向舊路徑,直接使用本地 src;
    # site-packages 內另有舊版 stera 副本,也要讓本地 src/ 優先,否則改動不會生效
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        filter(None, [str(SDK / "src"), str(SDK / "ml-depth-pro" / "src"),
                      env.get("PYTHONPATH")]))
    return env


def optical_to_link(mcap: Path) -> np.ndarray:
    """從 MCAP TF 讀 camera_optical -> camera_link 旋轉 (與 Visualizer 相同來源)。"""
    code = ("import sys,json; from stera.data import MCAPReader; "
            "print(json.dumps(MCAPReader(sys.argv[1]).R_optical_to_link.tolist()))")
    out = subprocess.run([sys.executable, "-c", code, str(mcap)], check=True,
                         cwd=SDK, env=sdk_env(), capture_output=True, text=True)
    return np.array(json.loads(out.stdout.strip().splitlines()[-1]), dtype=np.float64)


# ---------------------------------------------------------------------------
# Wrist Kalman filter
# ---------------------------------------------------------------------------

def cv_model(dt: float, q: float) -> tuple[np.ndarray, np.ndarray]:
    """Constant-velocity 模型 (state = [p, v]),white-noise acceleration 過程雜訊。"""
    I = np.eye(3)
    F = np.block([[I, dt * I], [np.zeros((3, 3)), I]])
    Q = q * np.block([[dt ** 3 / 3 * I, dt ** 2 / 2 * I],
                      [dt ** 2 / 2 * I, dt * I]])
    return F, Q


def rts_smooth(xs, Ps, xps, Pps, Fs):
    """Rauch-Tung-Striebel smoother。xps/Pps/Fs[k] 為由 k-1 預測到 k 的量。"""
    xs, Ps = xs.copy(), Ps.copy()
    for k in range(len(xs) - 2, -1, -1):
        C = Ps[k] @ Fs[k + 1].T @ np.linalg.inv(Pps[k + 1])
        xs[k] = xs[k] + C @ (xs[k + 1] - xps[k + 1])
        Ps[k] = Ps[k] + C @ (Ps[k + 1] - Pps[k + 1]) @ C.T
    return xs, Ps


def filter_wrist(ts, wrist_cam, valid, conf, M, t_cw, pose_ok, args):
    """對單手 wrist 軌跡做 KF (+RTS)。

    ts: (F,) 秒; wrist_cam: (F,3) 相機 optical frame; M: (F,3,3) optical->world
    旋轉 (R_c2w @ R_o2l); t_cw: (F,3) 相機在 world 位置。
    回傳 world frame 的位置/速度、狀態碼、標準差。
    """
    n = len(ts)
    pos = np.full((n, 3), np.nan)
    vel = np.full((n, 3), np.nan)
    std = np.full(n, np.nan)
    status = np.zeros(n, dtype=np.uint8)

    R_cam = np.diag([args.kf_sigma_xy ** 2] * 2 + [args.kf_sigma_z ** 2])
    gate = CHI2_3DOF[args.kf_gate]
    dt_nom = float(np.median(np.diff(ts))) if n > 1 else 1 / 15
    H = np.hstack([np.eye(3), np.zeros((3, 3))])

    def meas(k):
        z = M[k] @ wrist_cam[k] + t_cw[k]
        s = np.clip(args.kf_conf_ref / max(conf[k], 1e-3), 0.5, 3.0)
        return z, (s ** 2) * (M[k] @ R_cam @ M[k].T)

    seg = None  # 目前連續追蹤段的暫存 (供 RTS 使用)
    x = P = None
    last_meas_t = -np.inf
    n_reject_run = 0

    def start(k):
        """以第 k 幀量測開新 track (速度未知,給大不確定度)。"""
        nonlocal seg, x, P, last_meas_t, n_reject_run
        z, Rk = meas(k)
        x = np.concatenate([z, np.zeros(3)])
        P = np.zeros((6, 6))
        P[:3, :3] = Rk
        P[3:, 3:] = np.eye(3) * args.kf_init_vel_std ** 2
        last_meas_t, n_reject_run = ts[k], 0
        status[k] = KF_MEASURED
        seg = {"idx": [k], "x": [x.copy()], "P": [P.copy()],
               "xp": [x.copy()], "Pp": [P.copy()], "F": [np.eye(6)]}

    def close_segment():
        nonlocal seg, x
        x = None
        if seg is None:
            return
        idx = np.array(seg["idx"])
        xs, Ps = np.array(seg["x"]), np.array(seg["P"])
        if not args.kf_causal and len(idx) > 1:
            xs, Ps = rts_smooth(xs, Ps, np.array(seg["xp"]), np.array(seg["Pp"]),
                                np.array(seg["F"]))
        # 段尾若是純預測 (之後 track lost),預設丟掉這些外插,避免遺失前的漂移
        n_keep = len(idx)
        if args.kf_trim_tail:
            n_keep = 1 + max(i for i, k in enumerate(idx) if status[k] == KF_MEASURED)
        status[idx[n_keep:]] = KF_NONE
        idx, xs, Ps = idx[:n_keep], xs[:n_keep], Ps[:n_keep]
        pos[idx], vel[idx] = xs[:, :3], xs[:, 3:]
        std[idx] = np.sqrt(np.trace(Ps[:, :3, :3], axis1=1, axis2=2) / 3)
        seg = None

    for k in range(n):
        has_meas = bool(valid[k]) and bool(pose_ok[k]) and np.all(np.isfinite(wrist_cam[k]))
        if x is None:  # 尚未追蹤: 等第一個量測初始化
            if has_meas:
                start(k)
            continue

        dt = ts[k] - ts[k - 1]
        if not (0 < dt < 1.0):
            dt = dt_nom
        F, Q = cv_model(dt, args.kf_q)
        xp, Pp = F @ x, F @ P @ F.T + Q

        accepted = False
        if has_meas:
            z, Rk = meas(k)
            y = z - H @ xp
            S = H @ Pp @ H.T + Rk
            if float(y @ np.linalg.solve(S, y)) <= gate:
                K = Pp @ H.T @ np.linalg.inv(S)
                IKH = np.eye(6) - K @ H
                x = xp + K @ y
                P = IKH @ Pp @ IKH.T + K @ Rk @ K.T  # Joseph form
                accepted = True
                n_reject_run = 0
                last_meas_t = ts[k]
                status[k] = KF_MEASURED
            else:
                n_reject_run += 1

        if not accepted:
            # 連續多次被拒: 視為真的跳動 (或先前追錯); 漏太久: track lost
            if (has_meas and n_reject_run >= args.kf_reinit_after) or \
                    ts[k] - last_meas_t > args.kf_max_gap:
                close_segment()
                if has_meas:
                    start(k)
                continue
            x, P = xp, Pp
            status[k] = KF_REJECTED if has_meas else KF_PREDICTED

        seg["idx"].append(k)
        seg["x"].append(x.copy())
        seg["P"].append(P.copy())
        seg["xp"].append(xp)
        seg["Pp"].append(Pp)
        seg["F"].append(F)
    close_segment()
    return pos, vel, std, status


def reproject_joints(joints, kpts_2d, K):
    """以 2D 關鍵點反投影修正 3D 關節: 保留各關節深度 z,x/y 改由像素 (u,v) 與內參決定,
    使 3D 骨架投影回影像後與 2D 關鍵點一致 (SDK 原始手指 3D 與 2D 差 ~100 px)。
    2D 缺值/非有限的關節維持原值。"""
    z = joints[..., 2]
    xy = (kpts_2d[..., :2] - [K[0, 2], K[1, 2]]) / [K[0, 0], K[1, 1]] * z[..., None]
    out = np.concatenate([xy, z[..., None]], -1)
    bad = ~np.isfinite(out).all(-1) | ~np.isfinite(kpts_2d[..., :2]).all(-1) | (z <= 0)
    out[bad] = joints[bad]
    return out


def run_wrist_kf(h5_path: Path, R_o2l: np.ndarray, args) -> None:
    with h5py.File(h5_path, "r+") as f:
        hp = f["hand-pose"]
        if hp.attrs.get("coord_frame") != "camera_3d":
            print("[kf] hand-pose 不是 camera_3d,略過 wrist filter", file=sys.stderr)
            return
        ts = f["cam-pose/timestamps"][:].astype(np.float64)
        R_c2w = f["cam-pose/rotations"][:].astype(np.float64)
        t_cw = f["cam-pose/translations"][:].astype(np.float64)
        pose_ok = f["cam-pose/valid"][:]
        M = R_c2w @ R_o2l  # optical -> world
        K = np.load(K_PATH).astype(np.float64)

        for side in ("left", "right"):
            joints = hp[f"{side}_joints"][:].astype(np.float64)
            if args.reproject and f"{side}_kpts_2d_rgb" in hp:
                joints = reproject_joints(joints, hp[f"{side}_kpts_2d_rgb"][:].astype(np.float64), K)
            valid = hp[f"{side}_valid"][:]
            conf = hp[f"{side}_confidence"][:]
            pos_w, vel_w, std, status = filter_wrist(
                ts, joints[:, 0], valid, conf, M, t_cw, pose_ok, args)

            ok = status != KF_NONE
            wrist_cam = np.full_like(pos_w, np.nan)
            wrist_cam[ok] = np.einsum("nji,nj->ni", M[ok], pos_w[ok] - t_cw[ok])

            # 其餘 20 個關節各自跑同樣的 KF (wrist 本身即 j=0, 用 --kf-q)
            joints_kf = np.full_like(joints, np.nan)
            joints_kf[ok, 0] = wrist_cam[ok]
            jargs = argparse.Namespace(**{**vars(args), "kf_q": args.kf_q_joint})
            jstat = np.zeros(joints.shape[:2], dtype=np.uint8)
            jstat[:, 0] = status
            for j in range(1, joints.shape[1]):
                pj, _, _, sj = filter_wrist(
                    ts, joints[:, j], valid, conf, M, t_cw, pose_ok, jargs)
                okj = (sj != KF_NONE) & ok
                joints_kf[okj, j] = np.einsum("nji,nj->ni", M[okj], pj[okj] - t_cw[okj])
                jstat[:, j] = sj
            # 關節缺值 (該關節 track 不到但 wrist 有) 以 wrist 平移的最近手形補上
            shape = None
            for k in range(len(ts)):
                if not ok[k]:
                    continue
                miss = ~np.all(np.isfinite(joints_kf[k]), axis=1)
                if not miss.any():
                    shape = joints_kf[k] - joints_kf[k, 0]
                elif shape is not None:
                    joints_kf[k, miss] = (shape + wrist_cam[k])[miss]

            out = {
                f"{side}_joints_kf": joints_kf.astype(np.float32),
                f"{side}_wrist_kf": wrist_cam.astype(np.float32),
                f"{side}_wrist_kf_world": pos_w.astype(np.float32),
                f"{side}_wrist_vel_kf_world": vel_w.astype(np.float32),
                f"{side}_wrist_kf_std": std.astype(np.float32),
                f"{side}_kf_valid": ok,
                f"{side}_kf_status": status,
                f"{side}_joints_kf_status": jstat,
            }
            for key, val in out.items():
                if key in hp:
                    del hp[key]
                hp.create_dataset(key, data=val)

            n_meas = int(valid.sum())
            print(f"[kf] {side}: raw valid {n_meas}, measured {(status == KF_MEASURED).sum()}, "
                  f"rejected {(status == KF_REJECTED).sum()}, "
                  f"gap-filled {(status == KF_PREDICTED).sum()}, kf valid {ok.sum()}/{len(ts)}",
                  file=sys.stderr)

        hp.attrs["kf_model"] = "constant_velocity, world frame" + (
            "" if args.kf_causal else " + RTS smoother")
        hp.attrs["joints_kf_reprojected"] = bool(args.reproject)
        hp.attrs["kf_status_codes"] = "0=none 1=measured 2=predicted(gap) 3=rejected(outlier->predicted)"
        hp.attrs["kf_params"] = json.dumps({
            k[3:]: v for k, v in vars(args).items() if k.startswith("kf_") and k != "kf_only"})
        hp.attrs["R_optical_to_link"] = R_o2l


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("session_dir", nargs="?", type=Path)
    ap.add_argument("--out-root", type=Path, default=ROOT / "data" / "processed")
    ap.add_argument("--max-frames", type=int)
    ap.add_argument("--yolo-conf", type=float)
    ap.add_argument("--device")
    ap.add_argument("--map-3d", default="auto")
    ap.add_argument("--live-pointcloud", action="store_true")
    kf = ap.add_argument_group("wrist Kalman filter")
    kf.add_argument("--no-kf", action="store_true", help="不跑 wrist filter")
    kf.add_argument("--kf-only", action="store_true", help="跳過 pipeline,只對既有 annotation.hdf5 重跑 filter")
    kf.add_argument("--kf-causal", action="store_true", help="只用前向 KF (不做 RTS smoother)")
    kf.add_argument("--kf-q", type=float, default=0.4, help="加速度過程雜訊強度 [m^2/s^3]")
    kf.add_argument("--no-reproject", dest="reproject", action="store_false",
                    help="不以 2D 關鍵點反投影修正 3D 關節 (用 SDK 原始 joints 進 filter)")
    kf.add_argument("--kf-q-joint", type=float, default=2.0, help="手指關節 (wrist 以外) 的過程雜訊強度")
    kf.add_argument("--kf-sigma-xy", type=float, default=0.008, help="相機 x/y 方向量測雜訊 [m]")
    kf.add_argument("--kf-sigma-z", type=float, default=0.025, help="相機深度方向量測雜訊 [m]")
    kf.add_argument("--kf-conf-ref", type=float, default=0.8, help="雜訊依 (conf_ref/conf)^2 縮放")
    kf.add_argument("--kf-gate", type=float, default=0.997, choices=sorted(CHI2_3DOF),
                    help="Mahalanobis gating 的 chi2 機率")
    kf.add_argument("--kf-reinit-after", type=int, default=3, help="連續被拒 N 次後重新初始化")
    kf.add_argument("--kf-max-gap", type=float, default=0.5, help="最長預測補點時間 [s]")
    kf.add_argument("--kf-init-vel-std", type=float, default=0.5, help="初始速度不確定度 [m/s]")
    kf.add_argument("--kf-no-trim-tail", dest="kf_trim_tail", action="store_false",
                    help="保留 track lost 前的尾端外插預測")
    args = ap.parse_args()

    session = (args.session_dir or latest_session(ROOT / "data" / "raw")).resolve()
    mcaps = sorted(session.glob("*.mcap"))
    if not mcaps:
        sys.exit(f"{session} 內沒有 .mcap 檔")
    out_dir = (args.out_root / session.name).resolve()

    if not args.kf_only:
        cmd = [sys.executable, str(SDK / "pipeline_mcap_depth_wilor.py"),
               "--mcap", str(mcaps[0]), "--wilor-dir", str(SDK / "WiLoR"),
               "--output-dir", str(out_dir), "--map-3d", args.map_3d]
        if args.max_frames:
            cmd += ["--max-frames", str(args.max_frames)]
        if args.yolo_conf is not None:
            cmd += ["--yolo-conf", str(args.yolo_conf)]
        if args.device:
            cmd += ["--device", args.device]
        if args.live_pointcloud:
            cmd.append("--live-pointcloud")
        # cwd=SDK 讓 WiLoR 的相對路徑 (pretrained_models 等) 與 depth_pro 匯入可用
        subprocess.run(cmd, check=True, cwd=SDK, env=sdk_env())

    if not args.no_kf:
        h5_path = out_dir / "annotation.hdf5"
        if not h5_path.exists():
            sys.exit(f"找不到 {h5_path}")
        run_wrist_kf(h5_path, optical_to_link(mcaps[0]), args)
    print(out_dir)


if __name__ == "__main__":
    main()
