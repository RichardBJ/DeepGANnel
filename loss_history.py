#!/usr/bin/env python3
"""
Pull training loss histories out of the .pt files written by batch_run.py.

Two stages are read, and each gets its own figure:

  fine    ck["resume"]["history"] -- (step, mean eps-MSE over the last
          viz_every steps). Present in every checkpoint.
  coarse  ck["coarse"]["history"] -- (step, instantaneous eps-MSE) for the
          stage 1 envelope model. Only present if diffusion_app.py was
          patched to keep chist in coarse_blob, so older checkpoints will
          report no coarse history and the coarse figure is skipped.

Steps are not comparable across phenotypes, because a shorter seed record
gives fewer windows per epoch. Both figures default to an epoch x axis,
derived from what each run recorded, so the curves line up.

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


# stage -> (checkpoint key, y axis label, figure title, filename suffix)
STAGES = {
    "fine": ("resume", "eps-MSE (mean over logging window)",
             "Stage 2 (fine detail) training loss", "_fine"),
    "coarse": ("coarse", "eps-MSE (instantaneous)",
               "Stage 1 (coarse envelope) training loss", "_coarse"),
}


def load_ckpt(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:                      # torch < 2.0 has no weights_only
        return torch.load(path, map_location="cpu")


def history_of(ck, stage, epochs_fallback):
    """Return (steps, losses, steps_per_epoch) or None if nothing was stored.

    steps_per_epoch comes from the run's own step and epoch counters. If
    either is missing we fall back to the last logged step divided by
    epochs_fallback.
    """
    blob = ck.get(STAGES[stage][0]) or {}
    hist = blob.get("history") or []
    if not hist:
        return None
    steps = np.array([h[0] for h in hist], dtype=float)
    losses = np.array([h[1] for h in hist], dtype=float)
    last = float(blob.get("step") or blob.get("steps") or steps[-1])
    done = float(blob.get("epoch") or blob.get("epochs") or 0)
    spe = last / done if done > 0 else last / float(epochs_fallback)
    return steps, losses, spe


def draw(curves, stage, args, out_dir):
    label, title, suffix = STAGES[stage][1:]
    fig, ax = plt.subplots(figsize=(8, 5))
    for tag, steps, eps, losses in curves:
        ax.plot(eps if args.x == "epoch" else steps, losses, lw=1.2, label=tag)
    ax.set_yscale(args.yscale)
    ax.set_xscale(args.xscale)
    ax.set_xlabel("epoch" if args.x == "epoch" else "training step")
    ax.set_ylabel(label)
    ax.set_title(title)
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = os.path.join(out_dir, f"loss_history{suffix}.tif")
    fig.savefig(path, dpi=args.dpi, format="tiff",
                pil_kwargs={"compression": "tiff_lzw"})
    plt.close(fig)
    print(f"{len(curves):2d} {stage:6s} curve(s) -> {path} "
          f"({args.dpi} dpi, LZW, {os.path.getsize(path) / 1e6:.1f} MB)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True, help="folder of *_DDPM.pt checkpoints")
    ap.add_argument("--out", default=None, help="output folder (default: --dir)")
    ap.add_argument("--glob", default="*.pt")
    ap.add_argument("--dpi", type=int, default=600)
    ap.add_argument("--x", choices=["epoch", "step"], default="epoch",
                    help="x axis: epoch puts every run on the same scale")
    ap.add_argument("--epochs", type=float, default=20,
                    help="fallback fine epochs if the checkpoint lacks it")
    ap.add_argument("--coarse-epochs", type=float, default=80,
                    help="fallback coarse epochs (epochs * slow_mult)")
    #For me log log looks best so I've set these to defaultans
    ap.add_argument("--xscale", choices=["linear", "log"], default="log",
                    help="log spreads out the early rapid drop")
    ap.add_argument("--yscale", choices=["linear", "log"], default="log")
    args = ap.parse_args()

    out_dir = args.out or args.dir
    os.makedirs(out_dir, exist_ok=True)

    files = sorted(glob.glob(os.path.join(args.dir, args.glob)))
    if not files:
        raise SystemExit(f"no files matching {args.glob} in {args.dir}")

    fallback = {"fine": args.epochs, "coarse": args.coarse_epochs}
    rows, curves = [], {s: [] for s in STAGES}

    for f in files:
        name = os.path.splitext(os.path.basename(f))[0]
        try:
            ck = load_ckpt(f)
        except Exception as e:
            print(f"[{name}] unreadable: {type(e).__name__}: {e}")
            continue
        tag = ck.get("phenotype") or name

        for stage in STAGES:
            h = history_of(ck, stage, fallback[stage])
            if h is None:
                print(f"[{tag}] no {stage} history in checkpoint")
                continue
            steps, losses, spe = h
            eps = steps / spe

            for s, e, l in zip(steps, eps, losses):
                rows.append({"phenotype": tag,
                             "stage": stage,
                             "file": os.path.basename(f),
                             "step": int(s),
                             "epoch": e,
                             "steps_per_epoch": spe,
                             "eps_mse": l})
            curves[stage].append((tag, steps, eps, losses))
            print(f"[{tag}] {stage:6s} {len(steps):4d} points, {spe:.0f} steps/epoch, "
                  f"{int(steps[-1])} steps = {eps[-1]:.1f} epochs, "
                  f"loss {losses[0]:.4g} -> {losses[-1]:.4g}, "
                  f"best {losses.min():.4g} at epoch {eps[losses.argmin()]:.1f}")

    if not rows:
        raise SystemExit("nothing to plot")

    csv_path = os.path.join(out_dir, "loss_history.csv")
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    print(f"\n{len(rows)} row(s) -> {csv_path}")

    for stage in STAGES:
        if curves[stage]:
            draw(curves[stage], stage, args, out_dir)


if __name__ == "__main__":
    main()
