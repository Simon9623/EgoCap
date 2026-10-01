#!/usr/bin/env python3
"""Step 1: 從 Android 裝置抓最新一次錄製的 session 到 data/raw/。

用法: python3 01_pull_latest.py [--app-id ID] [--raw-root DIR]
成功時最後一行 stdout 印出本機 session 目錄路徑。
"""
import argparse
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def adb(*args: str) -> str:
    return subprocess.run(["adb", *args], check=True, capture_output=True, text=True).stdout.replace("\r", "")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--app-id", default="open.fpvlabs.stera")
    ap.add_argument("--raw-root", type=Path, default=ROOT / "data" / "raw")
    ap.add_argument("--force", action="store_true", help="本機已存在同名 session 時覆蓋重抓")
    args = ap.parse_args()

    if not shutil.which("adb"):
        sys.exit("找不到 adb,請安裝 Android platform-tools。")
    devices = [l.split()[0] for l in adb("devices").splitlines()[1:] if l.strip().endswith("device")]
    if not devices:
        sys.exit("沒有連接的 Android 裝置,請接上手機並開啟 USB 偵錯。")

    remote_root = f"/storage/emulated/0/Android/data/{args.app_id}/files/ar_sessions"
    names = sorted(n.strip() for n in adb("shell", f"ls {remote_root}").splitlines() if n.strip().startswith("session_"))
    if not names:
        sys.exit(f"裝置上找不到 session: {remote_root}")
    latest = names[-1]  # session_YYYYMMDD_HHMMSS 依字典序即時間序
    print(f"最新 session: {latest}", file=sys.stderr)

    args.raw_root.mkdir(parents=True, exist_ok=True)
    dest = args.raw_root / latest
    if dest.exists() and not args.force:
        print(f"已存在,略過下載 (--force 可重抓): {dest}", file=sys.stderr)
    else:
        if dest.exists():
            shutil.rmtree(dest)
        subprocess.run(["adb", "pull", f"{remote_root}/{latest}", str(args.raw_root)], check=True)
    print(dest)


if __name__ == "__main__":
    main()
