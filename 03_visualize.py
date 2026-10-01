#!/usr/bin/env python3
"""Step 3: 以 Rerun 開啟處理結果 (相機軌跡 + Depth Pro 深度 + WiLoR 手部骨架)。

用法: python3 03_visualize.py [processed_session_dir] [--detach]
未指定時使用 data/processed 下最新的 session。

若 annotation.hdf5 含 02 產生的 wrist Kalman filter 結果,會另存
visualization_kf.rrd (與原 rrd 同一 recording,viewer 會合併顯示),
(--no-kf 可略過) 內容:
3D scene: world/hands/{left,right}/{joints,bones} 以 KF 濾波後的 21 關節 (joints_kf) 覆蓋
  原始手部骨架 (同路徑、同時間戳,後載入者勝出)。
3D scene: world/body (估計的頭/軀幹/手臂) 也以 KF wrist 重算,手臂末端才會接在濾波後的手上。
"RGB + Hands" 畫面 (camera/rgb_overlay/kf/{left,right}/) 疊加 2D 軌跡:
  trail      最近 --trail 秒的濾波後軌跡 (實線) 與目前 wrist 點
  pred       以「這一幀」的 KF 狀態 (位置+速度) 等速外插未來 --horizon 秒 (虛線+點);
             每幀重新預測一小段,而不是一次畫出整條軌跡
軌跡存在 world frame,每幀用當下 camera pose 投影,相機移動時仍對得上。
"""
import argparse
import subprocess
import sys
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parent
K_PATH = ROOT / "stera-sdk" / "ml-depth-pro" / "cam" / "rgb_K.npy"
PRED_COLOR = {"left": [255, 140, 0], "right": [255, 60, 200]}
OVERLAY = "camera/rgb_overlay/kf"  # "RGB + Hands" view 的 origin 之下
HAND_COLOR = {"left": [255, 100, 100], "right": [100, 255, 100]}  # 同 SDK Visualizer
HAND_EDGES = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8),
              (0, 9), (9, 10), (10, 11), (11, 12), (0, 13), (13, 14), (14, 15), (15, 16),
              (0, 17), (17, 18), (18, 19), (19, 20), (5, 9), (9, 13), (13, 17)]
BONE_RADIUS, JOINT_RADIUS = 0.001, 0.003
SIDE_COLOR = {"left": [80, 160, 255], "right": [80, 255, 120]}


def project(p_world, R_cw, t_cw, M, K, min_z=0.05):
    """world -> 此幀相機 optical frame -> 像素; 回傳 (N,2) 與「在相機前方」遮罩。"""
    pc = (p_world - t_cw) @ M  # M^T (p - t), M = R_c2w @ R_o2l
    ok = pc[:, 2] > min_z
    z = np.where(ok, pc[:, 2], 1.0)
    uv = np.stack([K[0, 0] * pc[:, 0] / z + K[0, 2], K[1, 1] * pc[:, 1] / z + K[1, 2]], 1)
    return uv, ok


def log_overlay_2d(side, k, ts, st, kf_w, vel_w, R_cw, t_cw, M, K, horizon, trail, step):
    """在第 k 幀的 RGB 畫面疊加濾波軌跡 + 短時間預測。"""
    import rerun as rr

    base = f"{OVERLAY}/{side}"
    if st[k] == 0 or not np.isfinite(kf_w[k]).all():
        rr.log(base, rr.Clear(recursive=True))
        return
    idx = np.arange(np.searchsorted(ts, ts[k] - trail), k + 1)
    idx = idx[st[idx] != 0]
    uv, ok = project(kf_w[idx], R_cw[k], t_cw[k], M[k], K)
    segs = [[uv[i].tolist(), uv[i + 1].tolist()] for i in range(len(uv) - 1)
            if ok[i] and ok[i + 1] and idx[i + 1] - idx[i] <= 3]
    c = SIDE_COLOR[side]
    rr.log(f"{base}/trail", rr.LineStrips2D(segs, colors=[c], radii=2.0) if segs
           else rr.Clear(recursive=False))
    rr.log(f"{base}/wrist", rr.Points2D(uv[-1:], colors=[c], radii=6.0) if ok[-1]
           else rr.Clear(recursive=False))
    taus = np.arange(step, horizon + 1e-9, step)
    uvp, okp = project(np.vstack([kf_w[k], kf_w[k] + taus[:, None] * vel_w[k]]),
                       R_cw[k], t_cw[k], M[k], K)
    dash = [[uvp[i].tolist(), uvp[i + 1].tolist()] for i in range(0, len(uvp) - 1, 2)
            if okp[i] and okp[i + 1]]
    pc = PRED_COLOR[side]
    rr.log(f"{base}/pred", rr.LineStrips2D(dash, colors=[pc], radii=2.0) if dash
           else rr.Clear(recursive=False))
    rr.log(f"{base}/pred_pts", rr.Points2D(uvp[1:][okp[1:]], colors=[pc], radii=4.0)
           if okp[1:].any() else rr.Clear(recursive=False))


def log_hand_3d(side, k, st, joints_w):
    """以 KF 濾波後的關節覆蓋 world/hands/{side} (與 SDK 原始手同路徑)。"""
    import rerun as rr

    base = f"world/hands/{side}"
    if st[k] == 0 or not np.isfinite(joints_w[k]).all():
        rr.log(f"{base}/joints", rr.Clear(recursive=False))
        rr.log(f"{base}/bones", rr.Clear(recursive=False))
        return
    j, c = joints_w[k], HAND_COLOR[side]
    rr.log(f"{base}/joints", rr.Points3D(j, colors=[c], radii=JOINT_RADIUS))
    rr.log(f"{base}/bones", rr.LineStrips3D(
        [[j[a].tolist(), j[b].tolist()] for a, b in HAND_EDGES], colors=[c], radii=BONE_RADIUS))


BODY = dict(neck_down=0.12, shoulder_half=0.20, torso=0.50, hip_half=0.10, upper_arm=0.30, forearm=0.27)
BODY_COLOR = [80, 160, 255]


def log_body(k, M, t_cw, wrists, state):
    """複製 SDK Visualizer._update_body: 以 KF wrist 重算估計的頭/軀幹/手臂骨架 (world/body)。
    state['fwd'] 保存前一幀朝向 (與 SDK 相同的 0.85/0.15 平滑)。"""
    import rerun as rr

    up = np.array([0.0, 1.0, 0.0])  # ARCore world: Y-up
    o = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]]) @ M[k].T + t_cw[k]
    head, fwd, img_up = o[0], o[1] - o[0], o[2] - o[0]
    h = (fwd + img_up) * np.array([1.0, 0.0, 1.0])
    n = np.linalg.norm(h)
    if n < 1e-6:
        return
    h /= n
    if state.get("fwd") is not None:
        h = 0.85 * state["fwd"] + 0.15 * h
        h /= max(np.linalg.norm(h), 1e-9)
    state["fwd"] = h
    right = np.cross(h, up)
    B = BODY
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
        pole = -up + 0.5 * out
        pole -= np.dot(pole, u) * u
        pn = np.linalg.norm(pole)
        pole = pole / pn if pn > 1e-6 else -up
        joints[f"el_{side}"], joints[f"wr_{side}"] = S + a * u + hh * pole, w
        edges += [("sh_l" if side == "left" else "sh_r", f"el_{side}"), (f"el_{side}", f"wr_{side}")]
    rr.log("world/body/bones", rr.LineStrips3D(
        [[joints[a].tolist(), joints[b].tolist()] for a, b in edges],
        colors=[BODY_COLOR], radii=BONE_RADIUS * 2))
    rr.log("world/body/joints", rr.Points3D(
        np.array(list(joints.values())), colors=[BODY_COLOR], radii=JOINT_RADIUS * 1.5))


def build_kf_rrd(d: Path, rrd: Path, horizon: float = 0.2, trail: float = 1.0,
                 step: float = 0.05):
    """由 annotation.hdf5 的 *_kf 欄位產生 visualization_kf.rrd; 無資料回傳 None。"""
    import rerun as rr

    h5 = d / "annotation.hdf5"
    if not h5.exists():
        return None
    with h5py.File(h5, "r") as f:
        hp = f["hand-pose"]
        if "right_wrist_kf_world" not in hp and "left_wrist_kf_world" not in hp:
            return None
        ts = f["cam-pose/timestamps"][:]
        R_cw = f["cam-pose/rotations"][:].astype(np.float64)
        M = R_cw @ np.asarray(hp.attrs["R_optical_to_link"])
        t_cw = f["cam-pose/translations"][:].astype(np.float64)
        data = {}
        for side in ("left", "right"):
            if f"{side}_wrist_kf_world" not in hp:
                continue
            data[side] = (hp[f"{side}_wrist_kf_world"][:].astype(np.float64),
                          hp[f"{side}_kf_status"][:],
                          hp[f"{side}_wrist_vel_kf_world"][:].astype(np.float64),
                          np.einsum("nij,nkj->nki", M, hp[f"{side}_joints_kf"][:].astype(np.float64))
                          + t_cw[:, None] if f"{side}_joints_kf" in hp else None)
    K = np.load(K_PATH).astype(np.float64)

    # 沿用原 rrd 的 application/recording id,viewer 同時開啟兩檔時會合併
    src = rr.dataframe.load_recording(str(rrd))
    rr.init(src.application_id(), recording_id=src.recording_id())
    # 時間戳直接取自原 rrd,確保與原手部骨架完全同一時間點 (否則 latest-at 會讓原始資料勝出)
    times = src.view(index="time", contents="/camera/rgb_overlay").select().read_all() \
        .column("time").to_numpy().astype("datetime64[ns]")
    if len(times) != len(ts) or np.abs((times.astype("int64") * 1e-9 - ts)).max() > 1e-3:
        print("[warn] rrd 時間戳與 annotation 不一致,3D 手部覆蓋可能失效", file=sys.stderr)
    out = d / "visualization_kf.rrd"
    rr.save(str(out))

    body = {}
    for k in range(len(ts)):
        rr.set_time("time", timestamp=times[k] if len(times) == len(ts) else float(ts[k]))
        wrists = {sd: kw[k] for sd, (kw, st, _, _) in data.items()
                  if st[k] != 0 and np.isfinite(kw[k]).all()}
        log_body(k, M, t_cw, wrists, body)
    for side, (kf_w, st, vel_w, joints_w) in data.items():
        for k in range(len(ts)):
            rr.set_time("time", timestamp=times[k] if len(times) == len(ts) else float(ts[k]))
            if joints_w is not None:
                log_hand_3d(side, k, st, joints_w)
            log_overlay_2d(side, k, ts, st, kf_w, vel_w, R_cw, t_cw, M, K, horizon, trail, step)
    rr.disconnect()
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("session_dir", nargs="?", type=Path)
    ap.add_argument("--no-kf", action="store_true", help="不顯示濾波後 wrist 軌跡")
    ap.add_argument("--detach", action="store_true", help="背景開啟 viewer 後立即返回")
    ap.add_argument("--horizon", type=float, default=0.2, help="每幀向前預測的時間長度 [s]")
    ap.add_argument("--trail", type=float, default=1.0, help="顯示過去濾波軌跡的長度 [s]")
    ap.add_argument("--pred-step", type=float, default=0.05, help="預測點間隔 [s]")
    args = ap.parse_args()

    if args.session_dir:
        d = args.session_dir.resolve()
    else:
        cands = sorted(p for p in (ROOT / "data" / "processed").glob("session_*")
                       if (p / "visualization.rrd").exists())
        if not cands:
            sys.exit("data/processed 下沒有 visualization.rrd,請先執行 02_depth_wilor.py")
        d = cands[-1]
    rrd = d / "visualization.rrd"
    if not rrd.exists():
        sys.exit(f"找不到 {rrd}")

    files = [rrd]
    if not args.no_kf:
        kf_rrd = build_kf_rrd(d, rrd, args.horizon, args.trail, args.pred_step)
        if kf_rrd:
            files.append(kf_rrd)
        else:
            print("annotation.hdf5 無 wrist KF 結果,只顯示原始 rrd (先跑 02)", file=sys.stderr)
    cmd = [sys.executable, "-m", "rerun", *map(str, files)]
    print("開啟:", *files, file=sys.stderr)
    if args.detach:
        subprocess.Popen(cmd, start_new_session=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    else:
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
