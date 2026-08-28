#!/usr/bin/env python3
"""
Redraw the preview jpgs for a completed batch run.

Reads each phenotype's exported parquet and its original seed csv, and
regenerates the jpg using the current preview_fig in batch_run.py. Nothing
else is touched -- no training, no fitting, no re-export.

Usage:
    python redraw_previews.py --root Phenotypes --out ./batch_out
"""

import argparse
import glob
import os
import shutil

import pandas as pd

from batch_run import CONFIG, find_seed, preview_fig, stage_seed
from diffusion_103 import load_pages


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="directory of phenotype folders")
    ap.add_argument("--out", default="./batch_out")
    ap.add_argument("--pattern", default="raw", help="substring marking the seed file")
    args = ap.parse_args()

    parquets = sorted(glob.glob(os.path.join(args.out, "*_DDPM.parquet")))
    if not parquets:
        raise SystemExit(f"No *_DDPM.parquet files in {args.out}")

    for pq_path in parquets:
        tag = os.path.basename(pq_path)[: -len("_DDPM.parquet")]
        folder = os.path.join(args.root, tag)
        stage = None
        try:
            if not os.path.isdir(folder):
                raise FileNotFoundError(f"no seed folder {folder}")
            stage, _ = stage_seed(find_seed(folder, args.pattern), tag)
            pages = load_pages(stage, verbose=False)
            gen_df = pd.read_parquet(pq_path)
            jpg = pq_path.replace(".parquet", ".jpg")
            preview_fig(pages[0], gen_df, CONFIG["preview_len"], jpg, tag)
            print(f"[{tag}] real Popen {pages[0][:, 0].mean():.3f}, "
                  f"generated {gen_df['Channels'].to_numpy().mean():.3f} -> {jpg}")
        except Exception as e:
            print(f"[{tag}] FAILED: {type(e).__name__}: {e}")
        finally:
            if stage:
                shutil.rmtree(stage, ignore_errors=True)


if __name__ == "__main__":
    main()
