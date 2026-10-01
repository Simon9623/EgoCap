# Egocentric manipulation pipeline

```
01_pull_latest.py   adb 抓最新 session -> data/raw/<session>
02_depth_wilor.py   mcap -> Depth Pro 深度 -> WiLoR 手部 3D -> wrist Kalman filter
                    -> data/processed/<session>/annotation.hdf5 (+ visualization.rrd)
03_visualize.py     套用 KF 結果,產生 visualization_kf.rrd 並用 Rerun 開啟
run_all.sh          依序執行 01 -> 02 -> 03   (--skip-pull / --force / --max-frames N / --no-viz)
```

```
stera-sdk/
  pipeline_mcap_depth_wilor.py   02 呼叫的底層管線
  src/stera/                     MCAP 讀取、Rerun 視覺化、WiLoR 包裝
  WiLoR/                         WiLoR 程式 + pretrained_models/ + mano_data/
  ml-depth-pro/                  Depth Pro 程式 + checkpoints/depth_pro.pt + cam/rgb_K.npy
```

安裝依賴: `pip install -r requirements.txt`
`02` 會自動把 `stera-sdk/src` 與 `ml-depth-pro/src` 加進 PYTHONPATH,不需 pip install SDK。
