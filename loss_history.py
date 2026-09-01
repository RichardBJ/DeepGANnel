#!/usr/bin/env python3
"""
Pull training loss histories out of the .pt files written by batch_run.py.

Each checkpoint carries ck["resume"]["history"], a list of
(step, mean eps-MSE over the last viz_every steps) recorded during the fine
detail stage. This walks a folder of *_DDPM.pt files, writes one tidy csv of
every curve, and draws them on a single log-scale axis.

Caveat: only the fine stage is in there. The stage 1 coarse envelope losses
were never saved to the checkpoint, so those curves are gone unless you kept
the terminal log.

Usage:
    python loss_history.py --dir ./batch_out
    python loss_history.py --dir ./batch_out --out ./loss_curves
"""

import argparse
import glob
import os

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_ckpt(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:                      # torch < 2.0 has no weights_only
        return torch.load(path, map_location="cpu")


def history_of(ck, epochs_fallback):
    """Return (steps, losses, steps_per_epoch) or None if nothing was stored.

    steps_per_epoch comes from resume["step"] / resume["epoch"], i.e. what the
    run itself recorded. If either is missing we fall back to the last logged
    step divided by epochs_fallback.
    """
    res = ck.get("resume") or {}
    hist = res.get("history") or []
    if not hist:
        return None
    steps = np.array([h[0] for h in hist], dtype=float)
    losses = np.array([h[1] for h in hist], dtype=float)
    last = float(res.get("step") or steps[-1])
    done = float(res.get("epoch") or 0)
    spe = last / done if done > 0 else last / float(epochs_fallback)
    return steps, losses, spe


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True, help="folder of *_DDPM.pt checkpoints")
    ap.add_argument("--out", default=None, help="output folder (default: --dir)")
    ap.add_argument("--glob", default="*.pt")
    ap.add_argument("--dpi", type=int, default=600)
    ap.add_argument("--x", choices=["epoch", "step"], default="epoch",
                    help="x axis: epoch puts every run on the same scale")
    ap.add_argument("--epochs", type=float, default=20,
                    help="fallback total epochs if the checkpoint lacks it")
    ap.add_argument("--xscale", choices=["linear", "log"], default="linear",
                    help="log spreads out the early rapid drop")
    ap.add_argument("--yscale", choices=["linear", "log"], default="log")
    args = ap.parse_args()

    out_dir = args.out or args.dir
    os.makedirs(out_dir, exist_ok=True)

    files = sorted(glob.glob(os.path.join(args.dir, args.glob)))
    if not files:
        raise SystemExit(f"no files matching {args.glob} in {args.dir}")

    rows, curves = [], []
    for f in files:
        name = os.path.splitext(os.path.basename(f))[0]
        try:
            ck = load_ckpt(f)
        except Exception as e:
            print(f"[{name}] unreadable: {type(e).__name__}: {e}")
            continue

        h = history_of(ck, args.epochs)
        if h is None:
            print(f"[{name}] no history in checkpoint")
            continue
        steps, losses, spe = h
        tag = ck.get("phenotype") or name
        eps = steps / spe

        for s, e, l in zip(steps, eps, losses):
            rows.append({"phenotype": tag,
                         "file": os.path.basename(f),
                         "step": int(s),
                         "epoch": e,
                         "steps_per_epoch": spe,
                         "eps_mse": l})
        curves.append((tag, steps, eps, losses))
        print(f"[{tag}] {len(steps)} points, {spe:.0f} steps/epoch, "
              f"{int(steps[-1])} steps = {eps[-1]:.1f} epochs, "
              f"loss {losses[0]:.4g} -> {losses[-1]:.4g}, "
              f"best {losses.min():.4g} at epoch {eps[losses.argmin()]:.1f}")

    if not rows:
        raise SystemExit("nothing to plot")

    csv_path = os.path.join(out_dir, "loss_history.csv")
    pd.DataFrame(rows).to_csv(csv_path, index=False)

    fig, ax = plt.subplots(figsize=(8, 5))
    for tag, steps, eps, losses in curves:
        ax.plot(eps if args.x == "epoch" else steps, losses, lw=1.2, label=tag)
    ax.set_yscale(args.yscale)
    ax.set_xscale(args.xscale)
    ax.set_xlabel("epoch" if args.x == "epoch" else "training step")
    ax.set_ylabel("eps-MSE (mean over logging window)")
    ax.set_title("Fine stage training loss")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8)
    fig.tight_layout()
    tif_path = os.path.join(out_dir, "loss_history.tif")
    fig.savefig(tif_path, dpi=args.dpi, format="tiff",
                pil_kwargs={"compression": "tiff_lzw"})
    plt.close(fig)

    print(f"\n{len(curves)} curve(s) -> {csv_path}\n                 -> {tif_path} "
          f"({args.dpi} dpi, LZW, {os.path.getsize(tif_path) / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
