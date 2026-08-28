#!/usr/bin/env python3
"""
Batch benchmark driver.

Walks a root directory of phenotype folders, and for each one:
  1. finds the RAW seed file (filename contains 'raw'),
  2. stages it as a clean 2-column csv (Channels, Noisy Current),
  3. trains the diffusion model on it,
  4. fits the Markov transition matrix to its idealisation,
  5. simulates and exports a synthetic record,
  6. writes model, data, preview and provenance into the output folder.

Everything runs single-channel. The topology is taken from DEFAULT_P in the
app: zeros stay zero, non-zero entries are fitted. Settings live in CONFIG
below and mirror the app's defaults -- edit there, not in the app.

A phenotype that fails is logged to its own .txt and the run moves on.

Usage:
    python batch_run.py --root /path/to/phenotypes --out ./batch_out
    python batch_run.py --root /path/to/phenotypes --out ./batch_out --only Phenotype_E
"""

import argparse
import os
import glob
import shutil
import tempfile
import time
import traceback
from datetime import datetime

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from diffusion_app import (
    CKPT_PATH, DEFAULT_P, SCRATCH_LABEL, SAMPLE_DT,
    run_training, fit_matrix_to_record, generate_and_export,
    write_run_metadata,
)
from diffusion_103 import load_pages


CONFIG = {
    # training
    "seq_len": 128,
    "epochs": 20,
    "batch_size": 32,
    "lr": 1e-4,
    "timesteps": 200,
    "base_channels": 64,
    "stride": 16,
    "viz_every": 25,
    "n_samples": 3,
    "use_slow": True,
    "slow_factor": 64,
    "slow_mult": 4,
    "slow_phases": 1,
    # markov fit
    "open_states": "3,4",
    "fit_samples": 50_000,
    "restarts": 3,
    "prior": 0.5,
    # generation
    "n_steps": 100_000,
    "decimate": 1,
    "seed": 0,
    # fixed for this experiment
    "n_channels": 1,
    "preview_len": 5000,
}

# A channels column may hold no more than this many discrete levels.
MAX_LEVELS = 10

_noop = lambda *a, **k: None


def find_seed(folder, pattern):
    hits = sorted(f for f in glob.glob(os.path.join(folder, "*.csv"))
                  if pattern.lower() in os.path.basename(f).lower())
    if not hits:
        raise FileNotFoundError(f"no file matching '{pattern}' in {folder}")
    if len(hits) > 1:
        raise ValueError(f"{len(hits)} files match '{pattern}' in {folder}: "
                         f"{[os.path.basename(h) for h in hits]}")
    return hits[0]


def stage_seed(seed_file, tag):
    """
    Write a clean 2-column csv (Channels, Noisy Current) to a temp folder, so
    load_pages matches on headers and never has to infer anything.

    Time is the constant-step increasing column. Channels is whichever of the
    rest has fewest distinct values after rounding to 10% of its range; those
    levels are rank-mapped onto 0, 1, 2... Current is what's left.
    """
    df = pd.read_csv(seed_file, header=None)
    df = df.apply(pd.to_numeric, errors="coerce").dropna().reset_index(drop=True)
    if df.shape[1] < 2:
        raise ValueError(f"{os.path.basename(seed_file)}: only {df.shape[1]} numeric column(s)")

    def quantise(s):
        span = float(s.max() - s.min())
        return np.round(s / (0.1 * span)) if span > 0 else np.zeros(len(s))

    time_col, levels = None, {}
    for c in df.columns:
        s = df[c].to_numpy()
        d = np.diff(s)
        if time_col is None and np.all(d > 0) and np.allclose(d, d[0], rtol=1e-3, atol=1e-9):
            time_col = c
            continue
        levels[c] = np.unique(quantise(s))

    chan_col = min(levels, key=lambda c: len(levels[c]))
    if len(levels[chan_col]) > MAX_LEVELS:
        raise ValueError(f"no column with <= {MAX_LEVELS} levels; "
                         f"counts {[len(v) for v in levels.values()]}")
    cur_col = max((c for c in levels if c != chan_col), key=lambda c: len(levels[c]))
    chan = np.searchsorted(levels[chan_col], quantise(df[chan_col].to_numpy()))

    raw = np.unique(df[chan_col].to_numpy())
    stage = tempfile.mkdtemp(prefix=f"seed_{tag}_")
    pd.DataFrame({"Channels": chan.astype(int),
                  "Noisy Current": df[cur_col].to_numpy()}).to_csv(
        os.path.join(stage, os.path.basename(seed_file)), index=False)
    return stage, {
        "time column": time_col,
        "channels column": chan_col,
        "current column": cur_col,
        "channel levels": len(levels[chan_col]),
        "raw channel values": np.round(raw, 4).tolist() if len(raw) <= MAX_LEVELS else "many",
    }


def _best_window(chan, n, lo=0.4, hi=0.8):
    """Start index of a window whose Popen sits in [lo, hi], or the nearest to it."""
    if len(chan) <= n:
        return 0
    c = np.concatenate([[0.0], np.cumsum(chan, dtype=float)])
    k = (c[n:] - c[:-n]) / n
    inside = np.flatnonzero((k >= lo) & (k <= hi))
    if inside.size:
        return int(inside[len(inside) // 2])
    return int(np.argmin(np.abs(k - 0.5 * (lo + hi))))


def preview_fig(real, gen_df, n, path, tag):
    cur = gen_df["Noisy Current"].to_numpy()
    chan = gen_df["Channels"].to_numpy()
    ri = _best_window(real[:, 0], n)
    gi = _best_window(chan, n)
    rp = real[ri:ri + n, 0].mean()
    gp = chan[gi:gi + n].mean()
    fig, ax = plt.subplots(4, 1, figsize=(7, 8))
    ax[0].plot(real[ri:ri + n, 1], color="tab:green", lw=0.5)
    ax[0].set_ylabel("current")
    ax[0].set_title(f"{tag}: real seed record (from {ri}, Popen {rp:.2f})")
    ax[1].plot(cur[gi:gi + n], color="tab:orange", lw=0.5)
    ax[1].set_ylabel("current")
    ax[1].set_title(f"generated (from {gi}, Popen {gp:.2f})")
    ax[2].plot(chan[gi:gi + n], color="tab:gray", drawstyle="steps-post")
    ax[2].set_ylabel("open channels")
    ax[2].set_title("simulated idealisation")
    ax[2].set_xlabel("sample")
    ax[3].hist(real[:, 1], bins=80, density=True, alpha=0.5,
               label=f"real (Popen {real[:, 0].mean():.3f})", color="tab:green")
    ax[3].hist(cur, bins=80, density=True, alpha=0.5,
               label=f"generated (Popen {chan.mean():.3f})", color="tab:orange")
    ax[3].set_xlabel("current")
    ax[3].set_ylabel("density")
    ax[3].set_title("all-points amplitude histogram (whole record)")
    ax[3].legend()
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def stamp_parquet(src, dest, seed_file, tag):
    tbl = pq.read_table(src)
    meta = dict(tbl.schema.metadata or {})
    meta[b"seed_file"] = seed_file.encode()
    meta[b"phenotype"] = tag.encode()
    pq.write_table(tbl.replace_schema_metadata(meta), dest)
    os.remove(src)


def run_one(folder, out_dir, pattern):
    tag = os.path.basename(os.path.normpath(folder)).replace(" ", "_")
    stem = os.path.join(out_dir, f"{tag}_DDPM")
    t0 = time.time()
    stage = None

    try:
        seed_file = find_seed(folder, pattern)
        seed_name = os.path.basename(seed_file)

        # Stage the seed alone, so load_pages cannot pick up the GAN files.
        stage, colmap = stage_seed(seed_file, tag)

        pages = load_pages(stage, verbose=False)
        if not pages:
            raise ValueError(f"{seed_name} produced no readable rows")
        levels = np.unique(np.rint(pages[0][:, 0]).astype(int))
        if levels.max() > CONFIG["n_channels"]:
            raise ValueError(f"{seed_name} shows levels {levels.tolist()}, "
                             f"batch mode is single-channel only")

        print(f"\n[{tag}] training on {seed_name} ({len(pages[0])} rows, "
              f"Popen {pages[0][:, 0].mean():.3f})")
        t_train = time.time()
        for _ in run_training(
            stage, False, CONFIG["seq_len"], CONFIG["epochs"], CONFIG["batch_size"],
            CONFIG["lr"], CONFIG["timesteps"], CONFIG["base_channels"],
            CONFIG["stride"], CONFIG["viz_every"], CONFIG["n_samples"],
            CONFIG["use_slow"], CONFIG["slow_factor"], CONFIG["slow_mult"],
            CONFIG["slow_phases"], SCRATCH_LABEL,
        ):
            pass
        t_train = time.time() - t_train

        # Archive the checkpoint with the seed filename baked in.
        ck = torch.load(CKPT_PATH, map_location="cpu")
        ck["seed_file"] = seed_name
        ck["phenotype"] = tag
        torch.save(ck, stem + ".pt")

        print(f"[{tag}] fitting transition matrix")
        t_fit = time.time()
        P, fit_status = fit_matrix_to_record(
            DEFAULT_P, CONFIG["open_states"], stage, 0, CONFIG["n_channels"],
            CONFIG["fit_samples"], CONFIG["restarts"], CONFIG["prior"],
            progress=_noop,
        )
        t_fit = time.time() - t_fit

        print(f"[{tag}] generating {CONFIG['n_steps']} rows")
        t_gen = time.time()
        _, tmp_parquet, gen_status = generate_and_export(
            P, CONFIG["open_states"], CONFIG["n_channels"], CONFIG["n_steps"],
            CONFIG["decimate"], CONFIG["seed"], stem + ".pt", progress=_noop,
        )
        t_gen = time.time() - t_gen
        stamp_parquet(tmp_parquet, stem + ".parquet", seed_name, tag)

        gen_df = pd.read_parquet(stem + ".parquet")
        preview_fig(pages[0], gen_df, CONFIG["preview_len"], stem + ".jpg", tag)

        write_run_metadata(stem + ".txt", f"Batch run: {tag}", {
            "Phenotype": tag,
            "Seed file": seed_name,
            "Seed folder": os.path.abspath(folder),
            "Seed rows": len(pages[0]),
            "Seed levels": levels.tolist(),
            "Seed Popen": f"{pages[0][:, 0].mean():.4f}",
            "Generated Popen": f"{gen_df['Channels'].to_numpy().mean():.4f}",
            **colmap,
            "Run started": datetime.fromtimestamp(t0).isoformat(timespec="seconds"),
            "Train seconds": f"{t_train:.1f}",
            "Fit seconds": f"{t_fit:.1f}",
            "Generate seconds": f"{t_gen:.1f}",
            "Total seconds": f"{time.time() - t0:.1f}",
            "Device": "cuda" if torch.cuda.is_available() else "cpu",
            "Torch": torch.__version__,
            "Training seed": 42,
            "Generation seed": CONFIG["seed"],
            "Sample dt": SAMPLE_DT,
            "Topology (zeros held)": np.asarray(DEFAULT_P).tolist(),
            "Fitted P": np.round(np.asarray(P, float), 6).tolist(),
            "Fit status": fit_status,
            "Export status": gen_status,
            **{f"cfg.{k}": v for k, v in CONFIG.items()},
        })
        print(f"[{tag}] done in {time.time() - t0:.0f}s -> {stem}.*")
        return True

    except Exception as e:
        write_run_metadata(stem + ".txt", f"Batch run FAILED: {tag}", {
            "Phenotype": tag,
            "Seed folder": os.path.abspath(folder),
            "Failed after seconds": f"{time.time() - t0:.1f}",
            "Error": f"{type(e).__name__}: {e}",
            "Traceback": traceback.format_exc().replace("\n", " | "),
        })
        print(f"[{tag}] FAILED: {type(e).__name__}: {e}")
        return False

    finally:
        if stage:
            shutil.rmtree(stage, ignore_errors=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="directory of phenotype folders")
    ap.add_argument("--out", default="./batch_out")
    ap.add_argument("--pattern", default="raw", help="substring marking the seed file")
    ap.add_argument("--only", default=None, help="run just this phenotype folder")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    folders = sorted(d.path for d in os.scandir(args.root) if d.is_dir())
    if args.only:
        folders = [f for f in folders if os.path.basename(f) == args.only]
        if not folders:
            raise SystemExit(f"No folder named '{args.only}' in {args.root}")
    if not folders:
        raise SystemExit(f"No phenotype folders in {args.root}")

    print(f"{len(folders)} phenotype(s): {[os.path.basename(f) for f in folders]}")
    t0 = time.time()
    ok = sum(run_one(f, args.out, args.pattern) for f in folders)
    print(f"\n{ok}/{len(folders)} succeeded in {time.time() - t0:.0f}s -> {args.out}")


if __name__ == "__main__":
    main()
