#!/usr/bin/env python3
"""
Text-mode replacement for diffusion_app.py's Gradio UI, for driving the
conditional diffusion generator from a machine with no browser access.

Does NOT reimplement any training/fitting/generation logic - it imports and
calls the exact same functions the Gradio app itself calls (run_training,
fit_matrix_to_record, generate_and_export), so behaviour matches the
interactive app exactly. Progress is printed to stdout instead of streamed
to a browser.

Usage:
    python3 diffusion_headless.py config.json

Config is one JSON object with an "action" key ("train", "fit_matrix", or
"generate") and a same-named sub-object of parameters. Any parameter not
given falls back to the same default the Gradio widget uses. See
example_*.json in this file's directory for one example per action.
"""
import glob as globmod
import json
import os
import shutil
import sys
import tempfile
from datetime import datetime
from pathlib import Path

import diffusion_app
from diffusion_app import (
    run_training,
    fit_matrix_to_record,
    generate_and_export,
    SCRATCH_LABEL,
    RESUME_LABEL,
    DEFAULT_P,
    CKPT_PATH,
)

TRAIN_DEFAULTS = {
    "folder": "./pages",
    "use_synthetic": False,
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
    "resume": False,
    # Headless-only additions (no equivalent Gradio widget):
    # - base_checkpoint: what to resume FROM when resume=true. Never written to -
    #   each file's output goes to its own seedname_<timestamp>_DDPM.pt instead of
    #   overwriting the shared checkpoint (unlike the Gradio app, which always
    #   overwrites CKPT_PATH in place).
    # - out_dir: where per-file outputs land.
    "base_checkpoint": CKPT_PATH,
    "out_dir": ".",
}

FIT_MATRIX_DEFAULTS = {
    "matrix": DEFAULT_P,
    "open_states": "3,4",
    "folder": "./pages",
    "page_idx": 0,
    "fit_channels": 0,
    "fit_samples": 50_000,
    "restarts": 3,
    "prior": 0.5,
}

GENERATE_DEFAULTS = {
    "matrix": DEFAULT_P,
    "open_states": "3,4",
    "n_channels": 1,
    "n_steps": 100_000,
    "decimate": 1,
    "seed": 0,
    "ckpt_path": CKPT_PATH,
    "out_path": None,  # None = leave at generate_and_export's own "generated_dataset.parquet"
}


def _existing_dated_archives():
    """Snapshot of every *_model_DDPM.pt/.txt run_training could create, so we
    can tell afterwards which ones are new (its internal timestamp is computed
    at its own completion time, which we can't predict from out here)."""
    return set(globmod.glob("*_model_DDPM.pt")) | set(globmod.glob("*_model_DDPM.txt"))


def _train_one(p, folder, resume_label, ckpt_path):
    """Run run_training() against `folder`, with diffusion_app.CKPT_PATH
    monkeypatched to ckpt_path for the duration of the call. run_training reads
    CKPT_PATH as a module global at call time, so this redirects every internal
    read/write (resume-from check, checkpoint save, archive copy) to ckpt_path
    without touching diffusion_app.py itself."""
    before = _existing_dated_archives()
    original = diffusion_app.CKPT_PATH
    diffusion_app.CKPT_PATH = ckpt_path
    try:
        gen = run_training(
            folder, p["use_synthetic"], p["seq_len"], p["epochs"], p["batch_size"],
            p["lr"], p["timesteps"], p["base_channels"], p["stride"], p["viz_every"],
            p["n_samples"], p["use_slow"], p["slow_factor"], p["slow_mult"],
            p["slow_phases"], resume_label,
        )
        last_status = None
        for payload in gen:
            status = payload[0]
            if status != last_status:
                print(status, flush=True)
                last_status = status
    finally:
        diffusion_app.CKPT_PATH = original
    after = _existing_dated_archives()
    return sorted(after - before)  # the archive .pt/.txt this run just created


def run_train(cfg):
    p = {**TRAIN_DEFAULTS, **cfg}
    resume_label = RESUME_LABEL if p["resume"] else SCRATCH_LABEL
    out_dir = Path(p["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    if p["use_synthetic"]:
        # No real files to iterate per-file over - one synthetic run, still
        # named/isolated the same way as a real seed would be.
        files = [None]
    else:
        folder = Path(p["folder"])
        files = sorted(folder.glob("*.csv")) + sorted(folder.glob("*.parquet"))
        if not files:
            raise SystemExit(f"No CSV/Parquet files found in {p['folder']}")

    print(f"[train] {len(files)} file(s), resume={p['resume']} "
          f"base_checkpoint={p['base_checkpoint']} out_dir={out_dir}", flush=True)

    results = []
    for f in files:
        seedname = f.stem if f is not None else "synthetic"
        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        ckpt_path = str(out_dir / f"{seedname}_{ts}_DDPM.pt")

        if p["resume"]:
            if not os.path.exists(p["base_checkpoint"]):
                raise SystemExit(
                    f"resume=true but base_checkpoint not found: {p['base_checkpoint']}")
            # Stage a copy under this seed's own name so run_training's own
            # `os.path.exists(CKPT_PATH)` resume-check finds a starting point
            # WITHOUT ever writing to base_checkpoint itself.
            shutil.copy(p["base_checkpoint"], ckpt_path)

        print(f"[train] === {seedname} -> {ckpt_path} ===", flush=True)

        if f is None:
            new_archives = _train_one(p, p["folder"], resume_label, ckpt_path)
        else:
            with tempfile.TemporaryDirectory() as stage_dir:
                os.symlink(f.resolve(), os.path.join(stage_dir, f.name))
                new_archives = _train_one(p, stage_dir, resume_label, ckpt_path)

        # Rename run_training's own auto-archive (bare "<timestamp>_model_DDPM.*",
        # no seed identity) to carry the seedname too, so nothing generically-named
        # is left lying around to collide with another file's run.
        for old_name in new_archives:
            ext = ".pt" if old_name.endswith(".pt") else ".txt"
            new_name = out_dir / f"{seedname}_{Path(old_name).stem}{ext}"
            shutil.move(old_name, new_name)
            print(f"[train] archived -> {new_name}", flush=True)

        print(f"[train] {seedname} done -> {ckpt_path}", flush=True)
        results.append(ckpt_path)

    print(f"[train] all done, {len(results)} model(s):", flush=True)
    for r in results:
        print(f"  {r}", flush=True)
    return results


def run_fit_matrix(cfg):
    p = {**FIT_MATRIX_DEFAULTS, **cfg}
    print(f"[fit_matrix] folder={p['folder']} page_idx={p['page_idx']} "
          f"open_states={p['open_states']}", flush=True)

    P, status = fit_matrix_to_record(
        p["matrix"], p["open_states"], p["folder"], p["page_idx"],
        p["fit_channels"], p["fit_samples"], p["restarts"], p["prior"],
    )
    print(status, flush=True)
    print("[fit_matrix] fitted matrix:", flush=True)
    print(json.dumps(P, indent=2), flush=True)
    return P


def run_generate(cfg):
    p = {**GENERATE_DEFAULTS, **cfg}
    print(f"[generate] ckpt={p['ckpt_path']} n_steps={p['n_steps']} "
          f"n_channels={p['n_channels']} seed={p['seed']}", flush=True)

    P, out_path, status = generate_and_export(
        p["matrix"], p["open_states"], p["n_channels"], p["n_steps"],
        p["decimate"], p["seed"], p["ckpt_path"],
    )
    print(status, flush=True)

    if p["out_path"] and p["out_path"] != out_path:
        shutil.move(out_path, p["out_path"])
        out_path = p["out_path"]
        print(f"[generate] moved output to {out_path}", flush=True)
    print(f"[generate] done -> {out_path}", flush=True)


PLOT_QC_DEFAULTS = {
    "files": None,      # explicit list of file paths
    "folder": None,     # or every .csv/.parquet in a folder
    "out_dir": "./qc_plots",
    "n_points": 1500,
    "start": 0,         # slice offset into the file
}


def _read_trace(path):
    import pandas as pd
    if str(path).endswith(".parquet"):
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path)
    cols = {c.lower(): c for c in df.columns}
    current = df[cols["noisy current"]].to_numpy()
    channels = df[cols["channels"]].to_numpy() if "channels" in cols else None
    return current, channels


def plot_trace_png(path, out_png, n_points=1500, start=0):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    current, channels = _read_trace(path)
    end = min(start + n_points, len(current))
    cur_slice = current[start:end]
    x = range(start, end)

    if channels is not None:
        fig, axes = plt.subplots(2, 1, figsize=(8, 4), sharex=True,
                                  gridspec_kw={"height_ratios": [3, 1]})
        axes[0].plot(x, cur_slice, color="tab:blue", linewidth=0.7)
        axes[0].set_ylabel("Noisy current")
        axes[0].set_title(f"{Path(path).name}  [{start}:{end}]")
        chan_slice = channels[start:end]
        axes[1].step(x, chan_slice, color="tab:red", where="post")
        axes[1].set_ylabel("Channel state")
        axes[1].set_xlabel("sample")
    else:
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.plot(x, cur_slice, color="tab:blue", linewidth=0.7)
        ax.set_ylabel("Noisy current")
        ax.set_xlabel("sample")
        ax.set_title(f"{Path(path).name}  [{start}:{end}]")

    fig.tight_layout()
    fig.savefig(out_png, dpi=120)
    plt.close(fig)


def run_plot_qc(cfg):
    p = {**PLOT_QC_DEFAULTS, **cfg}
    out_dir = Path(p["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    if p["files"]:
        files = [Path(f) for f in p["files"]]
    elif p["folder"]:
        folder = Path(p["folder"])
        files = sorted(folder.glob("*.csv")) + sorted(folder.glob("*.parquet"))
    else:
        raise SystemExit("plot_qc needs either 'files' or 'folder' in config")

    if not files:
        raise SystemExit(f"No CSV/Parquet files found for plot_qc")

    print(f"[plot_qc] {len(files)} file(s) -> {out_dir}", flush=True)
    made = []
    for f in files:
        out_png = out_dir / f"{f.stem}.png"
        try:
            plot_trace_png(f, out_png, n_points=p["n_points"], start=p["start"])
            print(f"[plot_qc] {f.name} -> {out_png}", flush=True)
            made.append(str(out_png))
        except Exception as e:
            print(f"[plot_qc] FAILED {f.name}: {e}", flush=True)

    print(f"[plot_qc] done, {len(made)}/{len(files)} PNGs written", flush=True)
    return made


ACTIONS = {
    "train": run_train,
    "fit_matrix": run_fit_matrix,
    "generate": run_generate,
    "plot_qc": run_plot_qc,
}


def main():
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} config.json", file=sys.stderr)
        raise SystemExit(1)

    with open(sys.argv[1]) as f:
        config = json.load(f)

    action = config.get("action")
    if action not in ACTIONS:
        print(f"config 'action' must be one of {list(ACTIONS)}, got {action!r}",
              file=sys.stderr)
        raise SystemExit(1)

    ACTIONS[action](config.get(action, {}))


if __name__ == "__main__":
    main()
