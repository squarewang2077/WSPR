#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Download + prepare Tiny-ImageNet-200 for torchvision ImageFolder.

What it does:
  1) downloads http://cs231n.stanford.edu/tiny-imagenet-200.zip
  2) extracts to <data_root>/tiny-imagenet-200
  3) reorganizes val/ into per-class subfolders using val_annotations.txt

Usage:
  python prepare_tiny_imagenet.py --data_root ./dataset
  # optional flags:
  #   --force     : redo download/extract/reorg even if targets exist
  #   --no_verify : skip basic size check of the downloaded file

After this, you should have:
  <data_root>/tiny-imagenet-200/train/...
  <data_root>/tiny-imagenet-200/val/<wnid>/...
"""

import argparse
import os
import sys
import zipfile
import shutil
import urllib.request
from pathlib import Path

URL = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"
FILENAME = "tiny-imagenet-200.zip"
EXPECTED_MIN_BYTES = 200 * 1024 * 1024  # ~200MB; real file is ~244MB

# -------- pretty printing --------
def human(n):
    for u in ["B","KB","MB","GB","TB"]:
        if n < 1024: return f"{n:.1f}{u}"
        n /= 1024
    return f"{n:.1f}PB"

def log(msg): print(f"[tiny-imagenet] {msg}")

# -------- download with progress --------
def _progress_hook(block_num, block_size, total_size):
    downloaded = block_num * block_size
    percent = 100.0 * downloaded / max(total_size, 1)
    bar_len = 30
    filled = int(bar_len * percent / 100.0)
    bar = "#" * filled + "-" * (bar_len - filled)
    end = "\r" if downloaded < total_size else "\n"
    sys.stdout.write(f"\rDownloading: |{bar}| {percent:6.2f}% "
                     f"({human(downloaded)} / {human(total_size)})")
    sys.stdout.flush()
    if end == "\n":
        sys.stdout.write(end)

def download(url: str, dst_zip: Path, force: bool, verify: bool):
    if dst_zip.exists() and not force:
        log(f"zip already exists: {dst_zip}")
        if verify:
            size = dst_zip.stat().st_size
            if size < EXPECTED_MIN_BYTES:
                raise RuntimeError(f"File too small ({human(size)}). "
                                   f"Delete it or use --force to redownload.")
        return

    dst_zip.parent.mkdir(parents=True, exist_ok=True)
    log(f"downloading to {dst_zip} ...")
    try:
        urllib.request.urlretrieve(url, dst_zip.as_posix(), _progress_hook)
    except Exception as e:
        if dst_zip.exists(): dst_zip.unlink(missing_ok=True)
        raise RuntimeError(f"Failed to download: {e}") from e

    if verify:
        size = dst_zip.stat().st_size
        if size < EXPECTED_MIN_BYTES:
            raise RuntimeError(f"Downloaded file suspiciously small ({human(size)}). "
                               f"Try again with --force.")

def extract(zip_path: Path, out_dir: Path, force: bool):
    target = out_dir / "tiny-imagenet-200"
    if target.exists() and not force:
        log(f"folder already exists: {target}")
        return target

    if force and target.exists():
        log(f"--force: removing existing folder {target}")
        shutil.rmtree(target)

    log(f"extracting {zip_path} -> {out_dir}")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(out_dir)
    return target

def reorganize_val(tiny_root: Path, force: bool):
    val_dir = tiny_root / "val"
    images_dir = val_dir / "images"
    ann_file = val_dir / "val_annotations.txt"

    # If images/ is already empty, we assume it's reorganized.
    if not images_dir.exists():
        log("val/images does not exist (already reorganized?)")
        return

    if not ann_file.exists():
        raise FileNotFoundError(f"Missing {ann_file}")

    log("reorganizing val/ into per-class subfolders ...")
    # read annotations: filename <tab> wnid ...
    with ann_file.open("r") as f:
        lines = [ln.strip().split("\t") for ln in f if ln.strip()]
    mapping = {fname: wnid for (fname, wnid, *_) in lines}

    moved = 0
    for fname, wnid in mapping.items():
        src = images_dir / fname
        if not src.exists():
            # maybe already moved
            continue
        wnid_dir = val_dir / wnid
        wnid_dir.mkdir(exist_ok=True)
        dst = wnid_dir / fname
        shutil.move(src.as_posix(), dst.as_posix())
        moved += 1

    # remove empty images/ folder if everything moved
    try:
        if not any(images_dir.iterdir()):
            images_dir.rmdir()
    except FileNotFoundError:
        pass

    log(f"moved {moved} images; val/ now ready for ImageFolder")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, default="./dataset",
                    help="where to put tiny-imagenet-200")
    ap.add_argument("--force", action="store_true",
                    help="redownload/reextract/reorganize even if present")
    ap.add_argument("--no_verify", action="store_true",
                    help="skip basic size check after download")
    args = ap.parse_args()

    root = Path(args.data_root).resolve()
    zip_path = root / FILENAME

    root.mkdir(parents=True, exist_ok=True)

    # 1) download
    download(URL, zip_path, force=args.force, verify=not args.no_verify)

    # 2) extract
    tiny_root = extract(zip_path, root, force=args.force)

    # 3) reorganize val/
    reorganize_val(tiny_root, force=args.force)

    log("All done ✅")
    log(f"Train folder: {tiny_root/'train'}")
    log(f"Val   folder: {tiny_root/'val'}")

if __name__ == "__main__":
    main()
