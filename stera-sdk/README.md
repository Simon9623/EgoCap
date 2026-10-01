# stera-sdk (trimmed)

只保留 egocentric 管線需要的部分:

- `src/stera/data` — MCAP 讀取與 episode/hdf5 匯出
- `src/stera/viz` — Rerun 視覺化
- `src/stera/models/wilor` — WiLoR 手部追蹤包裝
- `src/stera/{core,annotations,processing}` — 型別、座標轉換、mesh 小工具
- `pipeline_mcap_depth_wilor.py` — MCAP → Depth Pro → WiLoR → `annotation.hdf5`
- `WiLoR/`、`ml-depth-pro/` — 第三方模型程式與權重 (見專案根目錄 README)

License: Apache-2.0 (see LICENSE).
