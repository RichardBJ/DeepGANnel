#!/usr/bin/env python3
"""
Pull parameter counts and the eps-MSE loss curve out of the archived
Phenotype_*_DDPM.pt checkpoints.

Prints a summary table and writes loss_history.csv for Figure 2A.

Only the fine stage's loss curve is stored by the current diffusion_app.py,
so the coarse stage does not appear here.

Usage:
    python read_checkpoints.py
    python read_checkpoints.py --out ./batch_out
"""

import argparse
import csv
import glob
import os

import torch


ap = argparse.ArgumentParser()
ap.add_argument("--out", default="./batch_out")
args = ap.parse_args()

paths = sorted(glob.glob(os.path.join(args.out, "*_DDPM.pt")))
if not paths:
    raise SystemExit(f"No *_DDPM.pt files in {args.out}")

rows = []
print(f"{'phenotype':<14}{'params fine':>13}{'params coarse':>15}"
      f"{'loss start':>12}{'loss end':>11}{'points':>8}")

for path in paths:
    tag = os.path.basename(path).replace("_DDPM.pt", "")

    # weights_only defaults to True from torch 2.6 and rejects a checkpoint
    # holding plain Python objects, which these do.
    ck = torch.load(path, map_location="cpu", weights_only=False)

    def n_params(key):
        sd = (ck.get(key) or {}).get("model_state_dict")
        return sum(v.numel() for v in sd.values()) if sd else 0

    fine, coarse = n_params("fine"), n_params("coarse")
    hist = list((ck.get("resume") or {}).get("history", []))

    if hist:
        start, end = hist[0][1], hist[-1][1]
        tail = [v for _, v in hist[-20:]]
        print(f"{tag:<14}{fine:>13,}{coarse:>15,}{start:>12.4g}{end:>11.4g}{len(hist):>8}")
        print(f"{'':<14}mean of last 20 logged points: {sum(tail) / len(tail):.4g}")
    else:
        print(f"{tag:<14}{fine:>13,}{coarse:>15,}{'-':>12}{'-':>11}{0:>8}")

    for step, val in hist:
        rows.append({"phenotype": tag, "step": step, "eps_mse": val})

out_csv = os.path.join(args.out, "loss_history.csv")
with open(out_csv, "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=["phenotype", "step", "eps_mse"])
    w.writeheader()
    w.writerows(rows)

print(f"\n{len(rows)} loss points -> {out_csv}")
