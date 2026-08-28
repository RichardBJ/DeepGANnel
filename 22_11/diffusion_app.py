#!/usr/bin/env python3
"""
Live training dashboard for the conditional diffusion generator.

Press Start and watch, every few optimiser steps:
  - the loss curve so far,
  - the latest generated raw trace beside a real one (the "cartoon"),
  - an all-points amplitude histogram of real vs generated current.

It streams updates by yielding from the training loop, so the browser animates
as training progresses -- the Gradio equivalent of popping up the latest
simulation every few seconds.

Run:
    python gradio_app.py
"""

import os
import time
from collections import deque

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import glob
import shutil
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import gradio as gr

from diffusion_103 import (
    ConditionalDiffusion,
    load_pages,
    build_dataset,
    build_fine_dataset,
    decimate_pages,
    add_clock_columns,
    build_coarse_dataset,
    dwell_clock,
    burst_clock,
    block_mean,
    generate_synthetic_page,
    simulate_markov_idealisation,
    generate_long_current,
    mmd_rbf,
    set_seed,
)

CKPT_PATH = "ion_channel_diffusion_model.pt"
# Shut runs shorter than this many samples count as flickers, not burst ends.
BURST_TC = 1000

# Acquisition interval in seconds, used only to label the exported Time column
# in real units (the model itself is time-agnostic). 1e-4 == 10 kHz sampling.
SAMPLE_DT = 1e-4


def _loss_fig(history):
    fig, ax = plt.subplots(figsize=(5, 2.6))
    if history:
        steps, vals = zip(*history)
        ax.plot(steps, vals, color="tab:blue")
        ax.set_yscale("log")
    ax.set_xlabel("optimiser step")
    ax.set_ylabel("eps-MSE")
    ax.set_title("Training loss")
    fig.tight_layout()
    return fig


def _trace_fig(ideal, real, gen, ylim=None, ideal_ylim=None):
    fig, axes = plt.subplots(3, 1, figsize=(6, 4), sharex=True)
    axes[0].plot(ideal, color="tab:gray")
    axes[0].set_ylabel("idealisation")
    if ideal_ylim is not None:
        axes[0].set_ylim(ideal_ylim)
    axes[1].plot(real, color="tab:green")
    axes[1].set_ylabel("real")
    axes[2].plot(gen, color="tab:orange")
    axes[2].set_ylabel("generated")
    axes[2].set_xlabel("sample")
    if ylim is not None:
        axes[1].set_ylim(ylim)
        axes[2].set_ylim(ylim)
    axes[0].set_title("Latest simulation vs real (same idealisation)")
    fig.tight_layout()
    return fig


def _hist_fig(real_all, gen_all):
    fig, ax = plt.subplots(figsize=(5, 2.6))
    ax.hist(real_all, bins=60, density=True, alpha=0.5, label="real", color="tab:green")
    ax.hist(gen_all, bins=60, density=True, alpha=0.5, label="generated", color="tab:orange")
    ax.set_xlabel("current")
    ax.set_ylabel("density")
    ax.set_title("All-points amplitude histogram")
    ax.legend()
    fig.tight_layout()
    return fig


_ENV_CACHE = {}  # holds full real/gen envelope arrays for on-the-fly redraw


# Latest training frame, published by run_training and pulled by a gr.Timer, so the
# live view survives the training generator's SSE stream dropping on long runs.
# "seq" lets the timer skip repainting when nothing has changed -- otherwise it would
# re-render (and re-serialize) the growing figures every tick and starve training.
LATEST = {"v": (gr.update(), gr.update(), gr.update(), gr.update(), gr.update()), "seq": 0}
_LAST_SENT = {"seq": -1}


def _publish(payload):
    LATEST["v"] = payload
    LATEST["seq"] += 1
    return payload


def _pull_live():
    # Only hand Gradio new figures when there is genuinely a new frame. The envelope
    # plot is deliberately excluded so the env_show slider keeps ownership of it.
    if LATEST["seq"] == _LAST_SENT["seq"]:
        return gr.update(), gr.update(), gr.update(), gr.update()
    _LAST_SENT["seq"] = LATEST["seq"]
    s, lf, tf, hf, _ = LATEST["v"]
    return s, lf, tf, hf


def _envelope_fig(real_env, gen_env):
    fig, ax = plt.subplots(figsize=(6, 2.6))
    ax.plot(real_env, color="tab:green", label="real envelope")
    ax.plot(gen_env, color="tab:orange", alpha=0.85, label="coarse model")
    ax.set_xlabel("coarse step")
    ax.set_ylabel("level")
    ax.set_title("Slow envelope: real vs coarse model (is stage 1 the limiter?)")
    ax.legend()
    fig.tight_layout()
    return fig


def write_run_metadata(txt_path, title, info):
    """Write a flat 'key = value' provenance file next to an output artefact."""
    lines = [title, "=" * len(title), ""]
    lines += [f"{k} = {v}" for k, v in info.items()]
    with open(txt_path, "w") as fh:
        fh.write("\n".join(lines) + "\n")

SCRATCH_LABEL = "Start fresh (ignore any saved checkpoint)"
RESUME_LABEL = "Resume from last checkpoint"


def run_training(
    folder, use_synthetic, seq_len, epochs, batch_size, lr,
    timesteps, base_channels, stride, viz_every, n_samples,
    use_slow, slow_factor, slow_mult, slow_phases,
    resume,
):
    plt.close("all")
    # clear any leftover frame from a previous run before the timer can repaint it
    LATEST["v"] = ("Starting...", _loss_fig([]), None, None, gr.update())
    LATEST["seq"] += 1
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    seq_len = max(32, (int(seq_len) // 4) * 4)  # U-Net needs multiples of 4
    n_samples = int(n_samples)
    use_slow = bool(use_slow)
    resume = (resume == RESUME_LABEL)
    factor = max(2, int(slow_factor))
    coarse_epochs = max(1, int(round(int(epochs) * float(slow_mult))))

    if use_synthetic:
        pages = [generate_synthetic_page(n=5000)]
        data_source = "SYNTHETIC toy data (forced by checkbox)"
        train_files = ["synthetic toy data"]
    else:
        pages = load_pages(folder, verbose=False)
        train_files = sorted(
            os.path.basename(f) for f in
            glob.glob(os.path.join(folder, "*.csv")) + glob.glob(os.path.join(folder, "*.parquet"))
        )
        if pages:
            nrows = sum(len(p) for p in pages)
            data_source = f"REAL data: {len(pages)} file(s) from {folder} ({nrows} rows)"
        else:
            pages = [generate_synthetic_page(n=5000)]
            data_source = f"SYNTHETIC FALLBACK -- no CSVs found in {folder}"
            train_files = ["synthetic toy data (fallback)"]

    env_fig = None          # envelope diagnostic, filled in once stage 1 trains
    coarse_blob = None       # coarse checkpoint payload, only if use_slow

    # ============================================================
    # Stage 1 (optional): coarse envelope model -- the slow phase.
    # Learns long-range amplitude structure from a decimated view.
    # ============================================================
    if use_slow:
        cpages_by_id = [(i, cp)
                        for i, p in enumerate(add_clock_columns(pages, burst_tc=BURST_TC))
                        for cp in decimate_pages([p], factor, phases=int(slow_phases))]
        max_clen = max((len(cp) for _, cp in cpages_by_id), default=0)
        coarse_seq_len = max(32, (min(int(seq_len), max_clen) // 4) * 4)
        if max_clen < coarse_seq_len:
            raise gr.Error(
                f"Data too short for 'how slow' = {factor}: longest decimated page "
                f"is {max_clen} pts (need >= {coarse_seq_len}). Lower the slider, "
                f"or hit 'Scan input' to set its range."
            )

        Xc, Yc, Pc = build_coarse_dataset(cpages_by_id, seq_len=coarse_seq_len, stride=1)
        ccond_mean = Xc.mean(axis=(0, 2), keepdims=True)          # per channel
        ccond_std = Xc.std(axis=(0, 2), keepdims=True) + 1e-8
        cenv_mean, cenv_std = float(Yc.mean()), float(Yc.std()) + 1e-8
        cloader = DataLoader(
            TensorDataset(
                torch.tensor((Xc - ccond_mean) / ccond_std, dtype=torch.float32),
                torch.tensor((Yc - cenv_mean) / cenv_std, dtype=torch.float32),
                torch.tensor(Pc, dtype=torch.long),
            ),
            batch_size=int(batch_size), shuffle=True,
        )
        n_pages, emb_dim, emb_drop = len(pages), 4, 0.1
        page_emb = nn.Embedding(n_pages, emb_dim).to(device)
        nn.init.normal_(page_emb.weight, std=0.1)
        coarse = ConditionalDiffusion(
            base_channels=int(base_channels), time_dim=128,
            timesteps=int(timesteps), cond_channels=4 + emb_dim,
        ).to(device)
        copt = optim.AdamW(list(coarse.parameters()) + list(page_emb.parameters()),
                           lr=float(lr), weight_decay=1e-4)

        def _with_page(cond, pid, train=True):
            z = page_emb(pid)
            if train and emb_drop > 0:
                keep = (torch.rand(z.size(0), 1, device=z.device) >= emb_drop).float()
                z = z * keep
            return torch.cat(
                [cond, z[:, :, None].expand(-1, -1, cond.size(-1))], dim=1)

        chist, cstep = [], 0
        ctotal = coarse_epochs * len(cloader)
        yield _publish((f"DATA: {data_source}\nStage 1/2 (coarse envelope) starting "
               f"-- {ctotal} steps planned...",
               _loss_fig([]), None, None, None))
        for _ in range(coarse_epochs):
            for cond, x0, pid in cloader:
                cond, x0, pid = cond.to(device), x0.to(device), pid.to(device)
                copt.zero_grad(set_to_none=True)
                loss = coarse.loss(_with_page(cond, pid), x0)
                loss.backward()
                copt.step()
                cstep += 1
                if cstep % int(viz_every) == 0:
                    chist.append((cstep, loss.item()))
                    plt.close("all")
                    yield _publish((
                        f"DATA: {data_source}\nStage 1/2 (coarse envelope) "
                        f"step {cstep}/{ctotal} ({100 * cstep / ctotal:.0f}%) "
                        f"| eps-MSE {loss.item():.4g}",
                        _loss_fig(chist), None, None, None,
                    ))
        coarse.eval()

        # Envelope diagnostic: real slow envelope vs the coarse model's, on page 0.
        with torch.no_grad():
            cur0 = pages[0][:, 1].astype(np.float32)
            chan0 = pages[0][:, 0].astype(np.float32)
            real_env = block_mean(cur0, factor)
            since0, prev0 = dwell_clock(chan0)
            bsince0 = burst_clock(chan0, tc=BURST_TC)
            cide0 = np.stack([block_mean(chan0, factor),
                              block_mean(since0, factor),
                              block_mean(prev0, factor),
                              block_mean(bsince0, factor)], axis=0)
            cide0 = (cide0 - ccond_mean[0]) / ccond_std[0]
            z0 = page_emb.weight[0].detach().cpu().numpy()[:, None]
            cide0 = np.concatenate(
                [cide0, np.repeat(z0, cide0.shape[1], axis=1)], axis=0)
            gen_env = generate_long_current(
                coarse, cide0.astype(np.float32), device) * cenv_std + cenv_mean
        L = min(len(real_env), len(gen_env))
        _ENV_CACHE["real"], _ENV_CACHE["gen"] = real_env[:L], gen_env[:L]
        env_fig = _envelope_fig(real_env[:L], gen_env[:L])

        coarse_blob = {
            "model_state_dict": coarse.state_dict(),
            "config": {"base_channels": int(base_channels), "time_dim": 128,
                       "timesteps": int(timesteps), "cond_channels": 4 + emb_dim},
            "norm": {"cond_mean": ccond_mean.ravel().tolist(),
                     "cond_std": ccond_std.ravel().tolist(),
                     "cur_mean": cenv_mean, "cur_std": cenv_std,
                     "page_vecs": page_emb.weight.detach().cpu().numpy().tolist()},
        }

    # ============================================================
    # Stage 2 (always): fine detail model. Conditioned on the
    # idealisation alone, or on [idealisation, envelope] if slow.
    # ============================================================
    if use_slow:
        Xcond, Y = build_fine_dataset(pages, factor, seq_len=seq_len, stride=int(stride))
        ide, env = Xcond[:, 0], Xcond[:, 1]
        cond_mean, cond_std = float(ide.mean()), float(ide.std()) + 1e-8
        cur_mean, cur_std = float(Y.mean()), float(Y.std()) + 1e-8
        Xs = np.empty_like(Xcond)
        Xs[:, 0] = (ide - cond_mean) / cond_std
        Xs[:, 1] = (env - cur_mean) / cur_std      # envelope shares current units
        cond_channels = 2
    else:
        X, Y = build_dataset(pages, seq_len=seq_len, stride=int(stride))
        ide = X
        cond_mean, cond_std = float(X.mean()), float(X.std()) + 1e-8
        cur_mean, cur_std = float(Y.mean()), float(Y.std()) + 1e-8
        Xs = (X - cond_mean) / cond_std
        cond_channels = 1
    Ys = (Y - cur_mean) / cur_std

    loader = DataLoader(
        TensorDataset(
            torch.tensor(Xs, dtype=torch.float32),
            torch.tensor(Ys, dtype=torch.float32),
        ),
        batch_size=int(batch_size), shuffle=True,
    )
    n_samples = min(n_samples, len(Ys))

    # resume: adopt the architecture-defining settings (width, diffusion steps) from
    # the checkpoint *before* building the model. lr / windows / epochs stay yours.
    # cond_channels is data-driven, so a mismatch there means we can't resume.
    eff_base_channels = int(base_channels)
    eff_timesteps = int(timesteps)
    resume_ck = None
    resume_note = ""
    if resume and os.path.exists(CKPT_PATH):
        try:
            ck = torch.load(CKPT_PATH, map_location=device)
            saved = ck["fine"]["config"]
            if saved.get("cond_channels") == cond_channels and "resume" in ck:
                eff_base_channels = int(saved["base_channels"])
                eff_timesteps = int(saved["timesteps"])
                resume_ck = ck
            else:
                resume_note = " | conditioning differs -- starting fresh"
        except Exception as e:
            resume_note = f" | resume failed ({e}) -- starting fresh"

    model = ConditionalDiffusion(
        base_channels=eff_base_channels,
        time_dim=128,
        timesteps=eff_timesteps,
        cond_channels=cond_channels,
    ).to(device)
    opt = optim.AdamW(model.parameters(), lr=float(lr), weight_decay=1e-4)

    history = []
    step = 0
    start_epoch = 0
    if resume_ck is not None:
        try:
            model.load_state_dict(resume_ck["fine"]["model_state_dict"])
            opt.load_state_dict(resume_ck["resume"]["optimizer_state_dict"])
            step = int(resume_ck["resume"]["step"])
            start_epoch = int(resume_ck["resume"]["epoch"])
            history = list(resume_ck["resume"].get("history", []))
            adopted = ""
            if (eff_base_channels, eff_timesteps) != (int(base_channels), int(timesteps)):
                adopted = f" (adopted width={eff_base_channels}, steps={eff_timesteps})"
            resume_note = f" | RESUMED at epoch {start_epoch}, step {step}{adopted}"
        except Exception as e:  # corrupt/old checkpoint -> don't die, just start over
            resume_note = f" | resume failed ({e}) -- starting fresh"
    running, rcount = 0.0, 0
    recent_extremes = deque(maxlen=10)
    steps_per_epoch = max(1, len(loader))
    ispan = float(ide.max() - ide.min())
    ipad = 0.05 * ispan if ispan > 0 else 0.5
    ideal_ylim = (float(ide.min()) - ipad, float(ide.max()) + ipad)
    stage_label = "Stage 2/2 (fine detail)" if use_slow else "Single stage (slow phase off)"
    t0 = time.time()

    def snapshot():
        plt.close("all")
        idx = np.random.randint(0, len(Ys), size=n_samples)
        cond = torch.tensor(Xs[idx], dtype=torch.float32, device=device)
        real = Ys[idx]
        model.eval()
        with torch.no_grad():
            gen = model.sample(cond).cpu().numpy()
        model.train()
        gen_units = gen * cur_std + cur_mean
        real_units = real * cur_std + cur_mean
        ic = cond[0, 0] if cond.dim() == 3 else cond[0]
        ideal_units = ic.cpu().numpy() * cond_std + cond_mean
        mmd = mmd_rbf(real, gen)

        # freeze-proof terminal readout, independent of the browser stream;
        # \r keeps it on one refreshing line (trailing spaces clear the old line)
        print(
            f"\r[{time.time() - t0:7.0f}s] epoch {step / steps_per_epoch:6.2f}/{int(epochs)} "
            f"step {step:>7} | loss {history[-1][1]:.4g} | MMD {mmd:.4g}    ",
            end="", flush=True,
        )

        lo = float(min(real_units[0].min(), gen_units[0].min()))
        hi = float(max(real_units[0].max(), gen_units[0].max()))
        recent_extremes.append((lo, hi))
        ylim = (min(l for l, _ in recent_extremes),
                max(h for _, h in recent_extremes))

        status = (
            f"DATA: {data_source}\n"
            f"{stage_label} | epoch {step / steps_per_epoch:.2f}/{int(epochs)} "
            f"({steps_per_epoch} steps/epoch)\n"
            f"step {step} | eps-MSE {history[-1][1]:.4g} | "
            f"MMD {mmd:.4g} | {time.time() - t0:.0f}s elapsed"
        )
        return (
            status,
            _loss_fig(history),
            _trace_fig(ideal_units, real_units[0], gen_units[0], ylim=ylim,
                       ideal_ylim=ideal_ylim),
            _hist_fig(real_units.ravel(), gen_units.ravel()),
            gr.update(),  # leave the envelope plot alone so the width slider sticks
        )

    def save_ckpt():
        ck = {
            "fine": {
                "model_state_dict": model.state_dict(),
                "config": {"base_channels": eff_base_channels, "time_dim": 128,
                           "timesteps": eff_timesteps, "cond_channels": cond_channels},
                "norm": {"cond_mean": cond_mean, "cond_std": cond_std,
                         "cur_mean": cur_mean, "cur_std": cur_std},
            },
            "two_stage": use_slow,
            "factor": int(factor),
            "resume": {"optimizer_state_dict": opt.state_dict(),
                       "step": step, "epoch": epoch_idx + 1, "history": history},
        }
        if use_slow:
            ck["coarse"] = coarse_blob
        torch.save(ck, CKPT_PATH + ".tmp")      # write then atomic-rename
        os.replace(CKPT_PATH + ".tmp", CKPT_PATH)  # so a crash mid-save can't corrupt it

    yield _publish((f"DATA: {data_source}\n{stage_label} starting...{resume_note}",
           _loss_fig([]), None, None, env_fig))

    epoch_idx = start_epoch - 1  # bound up-front so the final save survives an empty loop
    for epoch_idx in range(start_epoch, int(epochs)):
        for cond, x0 in loader:
            cond, x0 = cond.to(device), x0.to(device)
            opt.zero_grad(set_to_none=True)
            loss = model.loss(cond, x0)
            loss.backward()
            opt.step()

            step += 1
            running += loss.item()
            rcount += 1

            if step % int(viz_every) == 0:
                history.append((step, running / rcount))
                running, rcount = 0.0, 0
                yield _publish(snapshot())
        save_ckpt()  # end-of-epoch checkpoint: a crash now costs at most one epoch

    if rcount:
        history.append((step, running / rcount))

    save_ckpt()

    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    dated_pt = f"{ts}_model_DDPM.pt"
    shutil.copy(CKPT_PATH, dated_pt)
    write_run_metadata(
        dated_pt.replace(".pt", ".txt"),
        "Conditional diffusion training run",
        {
            "Saved model": dated_pt,
            "Resume checkpoint": CKPT_PATH,
            "Data source": data_source,
            "Training files": ", ".join(train_files),
            "Sequence length": seq_len,
            "Window stride": int(stride),
            "Batch size": int(batch_size),
            "Epochs requested": int(epochs),
            "Learning rate": float(lr),
            "Diffusion steps": eff_timesteps,
            "U-Net channels": eff_base_channels,
            "Conditioning channels": cond_channels,
            "Two-stage (slow phase)": use_slow,
            "Slow factor": int(factor) if use_slow else "n/a",
            "Coarse epochs": coarse_epochs if use_slow else "n/a",
            "Decimation phases": int(slow_phases) if use_slow else "n/a",
            "Seed": 42,
            "Fine norm": f"cond_mean={cond_mean:.6g}, cond_std={cond_std:.6g}, "
                         f"cur_mean={cur_mean:.6g}, cur_std={cur_std:.6g}",
            "Coarse norm": (
                "cond_mean={}, cond_std={}, cur_mean={:.6g}, cur_std={:.6g}".format(
                    np.round(coarse_blob["norm"]["cond_mean"], 4).tolist(),
                    np.round(coarse_blob["norm"]["cond_std"], 4).tolist(),
                    coarse_blob["norm"]["cur_mean"], coarse_blob["norm"]["cur_std"])
                if use_slow else "n/a"
            ),
        },
    )

    s, lf, tf, hf, ef = snapshot()
    yield _publish((s + f"\nmodel saved -> {CKPT_PATH}  (archived -> {dated_pt})", lf, tf, hf, ef))


def _load_model(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device)

    def build(sub):
        cfg = sub["config"]
        m = ConditionalDiffusion(
            base_channels=cfg["base_channels"],
            time_dim=cfg["time_dim"],
            timesteps=cfg["timesteps"],
            cond_channels=cfg.get("cond_channels", 1),
        ).to(device)
        m.load_state_dict(sub["model_state_dict"])
        return m

    fine = build(ckpt["fine"])
    two_stage = bool(ckpt.get("two_stage", "coarse" in ckpt))
    factor = int(ckpt.get("factor", 1))
    if two_stage:
        coarse = build(ckpt["coarse"])
        return coarse, ckpt["coarse"]["norm"], fine, ckpt["fine"]["norm"], factor, True
    return None, None, fine, ckpt["fine"]["norm"], factor, False


# Each coarse training window needs this many decimated points to be healthy;
# it sets the largest 'how slow' the data can support.
COARSE_MIN_POINTS = 256


def scan_input(folder, use_synthetic):
    """Set the 'how slow' slider's range from the longest available input page."""
    if use_synthetic:
        longest, src = 5000, "synthetic toy data (5000 rows)"
    else:
        pages = load_pages(folder, verbose=False)
        if not pages:
            longest, src = 5000, f"no files in {folder}; assuming synthetic (5000 rows)"
        else:
            longest = max(len(p) for p in pages)
            src = f"{len(pages)} file(s), longest {longest} rows"

    fmax = max(8, (longest // COARSE_MIN_POINTS // 4) * 4)
    fmax = min(fmax, 2048)
    default = min(64, fmax)
    return (
        gr.update(minimum=8, maximum=fmax, value=default),
        f"Scanned: {src} -> 'how slow' can run up to {fmax} (each step averages that many samples).",
    )


PREVIEW_STEPS = 20000


def _normalise_rows(matrix):
    """Coax each row to sum to 1; leave all-zero rows untouched rather than NaN."""
    P = np.asarray(getattr(matrix, "values", matrix), dtype=float)
    sums = P.sum(axis=1, keepdims=True)
    sums[sums == 0] = 1.0
    return P / sums


def _parse_open_states(open_states):
    open_set = {int(s) for s in str(open_states).replace(" ", "").split(",") if s != ""}
    if not open_set:
        raise gr.Error("Specify at least one open-state index, e.g. '2,3'.")
    return open_set


def _ideal_fig(ideal, open_set, n_channels, real=None, real_label=None, gen=None, gen_label=None):
    rows = 1 + (real is not None) + (gen is not None)
    fig, axes = plt.subplots(rows, 1, figsize=(6, 2.4 * rows))
    axes = [axes] if rows == 1 else list(axes)
    axes[0].plot(ideal, color="tab:gray", drawstyle="steps-post")
    axes[0].set_yticks(range(int(n_channels) + 1))
    axes[0].set_ylabel("open channels")
    axes[0].set_title(
        f"Simulated idealisation (open states {sorted(open_set)}, "
        f"{int(n_channels)} channel(s), {len(ideal)} pts)"
    )
    i = 1
    if real is not None:
        axes[i].plot(real, color="tab:green")
        axes[i].set_ylabel("current")
        axes[i].set_title(f"Real example: {real_label}")
        i += 1
    if gen is not None:
        axes[i].plot(gen, color="tab:orange")
        axes[i].set_ylabel("current")
        axes[i].set_title(f"Generated (noised): {gen_label}")
    axes[-1].set_xlabel("sample")
    fig.tight_layout()
    return fig

def redraw_envelope(n_show):
    real, gen = _ENV_CACHE.get("real"), _ENV_CACHE.get("gen")
    if real is None:
        return None
    L = min(len(real), len(gen), int(n_show))
    return _envelope_fig(real[:L], gen[:L])

def _coarse_cond(cond_int, cnorm, factor, page=0):
    """Standardised (4+emb, N/factor) conditioning for the coarse stage."""
    since, prev = dwell_clock(cond_int.astype(np.float32))
    bsince = burst_clock(cond_int.astype(np.float32), tc=BURST_TC)
    c = np.stack([block_mean(cond_int, factor),
                  block_mean(since, factor),
                  block_mean(prev, factor),
                  block_mean(bsince, factor)], axis=0)
    c = (c - np.asarray(cnorm["cond_mean"], np.float32)[:, None]) \
        / np.asarray(cnorm["cond_std"], np.float32)[:, None]
    vecs = np.asarray(cnorm.get("page_vecs", []), np.float32)
    if vecs.size:
        z = vecs[int(page) % len(vecs)][:, None]
        c = np.concatenate([c, np.repeat(z, c.shape[1], axis=1)], axis=0)
    return c.astype(np.float32)


def _preview_synthesize(cond_int, ckpt_path, device):
    coarse, cnorm, fine, fnorm, factor, two_stage = _load_model(ckpt_path, device)
    if two_stage:
        cide_std = _coarse_cond(cond_int, cnorm, factor)
        env_std = generate_long_current(coarse, cide_std, device)
        env_coarse = env_std * cnorm["cur_std"] + cnorm["cur_mean"]
        env_full = np.repeat(env_coarse, factor)[: len(cond_int)]
        if len(env_full) < len(cond_int):
            env_full = np.concatenate(
                [env_full, np.full(len(cond_int) - len(env_full), env_coarse[-1], np.float32)]
            )
        ide_std = (cond_int.astype(np.float32) - fnorm["cond_mean"]) / fnorm["cond_std"]
        env_fine = (env_full.astype(np.float32) - fnorm["cur_mean"]) / fnorm["cur_std"]
        fine_cond = np.stack([ide_std, env_fine], axis=0).astype(np.float32)
        cur_std = generate_long_current(fine, fine_cond, device)
    else:
        ide_std = (cond_int.astype(np.float32) - fnorm["cond_mean"]) / fnorm["cond_std"]
        cur_std = generate_long_current(fine, ide_std, device)
    return cur_std * fnorm["cur_std"] + fnorm["cur_mean"]

def preview_ideal(matrix, open_states, n_channels, preview_len, folder, ckpt_path):
    plt.close("all")
    P = _normalise_rows(matrix)
    open_set = _parse_open_states(open_states)
    ideal = simulate_markov_idealisation(
        P, open_set, n_channels=int(n_channels), n_steps=int(preview_len), seed=None
    )

    pages = load_pages(folder, verbose=False)
    real, real_label = None, None
    if pages:
        files = sorted(
            os.path.basename(f) for f in
            glob.glob(os.path.join(folder, "*.csv")) + glob.glob(os.path.join(folder, "*.parquet"))
        )
        idx = np.random.randint(len(pages))
        real = pages[idx][:int(preview_len), 1]
        real_label = files[idx] if idx < len(files) else f"page {idx}"

    gen, gen_label = None, None
    if os.path.exists(ckpt_path):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        try:
            gen = _preview_synthesize(ideal, ckpt_path, device)
            gen_label = os.path.basename(ckpt_path)
        except Exception:
            pass  # no model yet, or shape mismatch -- just skip this panel

    return (np.round(P, 4).tolist(),
            _ideal_fig(ideal, open_set, int(n_channels), real=real, real_label=real_label,
                       gen=gen, gen_label=gen_label))

def generate_and_export(matrix, open_states, n_channels, n_steps, decimate, seed, ckpt_path, progress=gr.Progress()):
    if not os.path.exists(ckpt_path):
        raise gr.Error(f"No trained model at '{ckpt_path}'. Train first (it saves on finish).")

    P = _normalise_rows(matrix)
    open_set = _parse_open_states(open_states)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    coarse, cnorm, fine, fnorm, factor, two_stage = _load_model(ckpt_path, device)

    cond_int = simulate_markov_idealisation(
        P, open_set, n_channels=int(n_channels), n_steps=int(n_steps), seed=int(seed),
        progress=lambda f: progress(0.3 * f, desc="Simulating idealisation"),
    )

    if two_stage:
        # Stage 1: slow envelope from the decimated idealisation.
        cide_std = _coarse_cond(cond_int, cnorm, factor)
        env_std = generate_long_current(
            coarse, cide_std, device,
            progress=lambda f: progress(0.3 + 0.28 * f, desc="Sampling envelope"),
        )
        env_coarse = env_std * cnorm["cur_std"] + cnorm["cur_mean"]
        env_full = np.repeat(env_coarse, factor)[: len(cond_int)]
        if len(env_full) < len(cond_int):
            env_full = np.concatenate(
                [env_full, np.full(len(cond_int) - len(env_full), env_coarse[-1], np.float32)]
            )

        # Stage 2: fine detail conditioned on [idealisation, envelope].
        ide_std = (cond_int.astype(np.float32) - fnorm["cond_mean"]) / fnorm["cond_std"]
        env_fine = (env_full.astype(np.float32) - fnorm["cur_mean"]) / fnorm["cur_std"]
        fine_cond = np.stack([ide_std, env_fine], axis=0).astype(np.float32)
        cur_std = generate_long_current(
            fine, fine_cond, device,
            progress=lambda f: progress(0.58 + 0.42 * f, desc="Sampling detail"),
        )
    else:
        # Single stage: condition on the idealisation alone.
        ide_std = (cond_int.astype(np.float32) - fnorm["cond_mean"]) / fnorm["cond_std"]
        cur_std = generate_long_current(
            fine, ide_std, device,
            progress=lambda f: progress(0.3 + 0.7 * f, desc="Sampling"),
        )

    current = cur_std * fnorm["cur_std"] + fnorm["cur_mean"]

    d = max(1, int(decimate))
    chan = cond_int[::d].astype(int)
    cur = current[::d]
    df = pd.DataFrame({
        "Time": np.arange(len(chan)) * d * SAMPLE_DT,
        "Channels": chan,
        "Noisy Current": cur,
    })

    out_path = "generated_dataset.parquet"
    df.to_parquet(out_path)
    return np.round(P, 4).tolist(), out_path, f"Generated {len(df)} rows (levels {sorted(set(chan.tolist()))}) -> {out_path}"


DEFAULT_P = [
    [0.97, 0.03, 0.00, 0.00],
    [0.02, 0.90, 0.08, 0.00],
    [0.00, 0.10, 0.85, 0.05],
    [0.00, 0.00, 0.20, 0.80],
]


with gr.Blocks(title="Ion diffusion trainer") as demo:
    gr.Markdown("## Ion conditional diffusion: live training &nbsp;&nbsp;[Help](/gradio_api/file=help.html)")

    with gr.Row():
        with gr.Column(scale=1):
            folder = gr.Textbox("./pages", label="CSV folder")
            use_synthetic = gr.Checkbox(False, label="Force synthetic toy data (ignore CSV folder)")
            seq_len = gr.Slider(32, 512, value=128, step=4, label="Sequence length")
            epochs = gr.Slider(1, 200, value=20, step=1, label="Epochs")
            batch_size = gr.Slider(4, 128, value=32, step=4, label="Batch size")
            lr = gr.Number(1e-4, label="Learning rate")
            timesteps = gr.Slider(20, 1000, value=200, step=10, label="Diffusion steps")
            base_channels = gr.Slider(16, 128, value=64, step=16, label="Model width (U-Net channels)")
            stride = gr.Slider(1, 128, value=16, step=1, label="Window stride")
            viz_every = gr.Slider(5, 200, value=25, step=5, label="Update every N steps")
            n_samples = gr.Slider(1, 8, value=3, step=1, label="Sample traces")
            use_slow = gr.Checkbox(True, label="Model a slow phase (envelope cascade)")
            slow_factor = gr.Slider(8, 512, value=64, step=4,
                                    label="How slow (samples averaged per envelope step)")
            slow_mult = gr.Slider(1, 10, value=4, step=1,
                                  label="Slow-phase training x (coarse epochs vs fine)")
            slow_phases = gr.Slider(1, 8, value=1, step=1,
                                    label="Decimation phases (coarse augmentation; 1 = off)")
            resume = gr.Radio([SCRATCH_LABEL, RESUME_LABEL], value=SCRATCH_LABEL,
                              label="When I press Start")
            scan_btn = gr.Button("Scan input (set 'how slow' range)")
            start = gr.Button("Start training", variant="primary")

        with gr.Column(scale=2):
            status = gr.Textbox(label="Progress")
            loss_plot = gr.Plot(label="Loss")
            trace_plot = gr.Plot(label="Latest simulation")
            hist_plot = gr.Plot(label="Amplitude histogram")
            envelope_plot = gr.Plot(label="Slow envelope (real vs coarse model)")
            env_show = gr.Slider(50, 5000, value=500, step=50,
                                 label="Envelope samples to show")

    scan_btn.click(scan_input, inputs=[folder, use_synthetic], outputs=[slow_factor, status])

    start.click(
        run_training,
        inputs=[folder, use_synthetic, seq_len, epochs, batch_size, lr,
                timesteps, base_channels, stride, viz_every, n_samples,
                use_slow, slow_factor, slow_mult, slow_phases, resume],
        outputs=[status, loss_plot, trace_plot, hist_plot, envelope_plot],
    )

    env_show.release(redraw_envelope, inputs=env_show, outputs=envelope_plot)

    # Poll the latest training frame on a short tick. Because each tick is its own
    # tiny request, the live view survives the training generator's stream dropping
    # on multi-hour runs -- a dead tick is simply followed by a live one. _pull_live
    # repaints only on a new frame and leaves the envelope plot to the slider.
    live = gr.Timer(2.0)
    live.tick(_pull_live, None,
              outputs=[status, loss_plot, trace_plot, hist_plot],
              queue=False)

    gr.Markdown("## Generate dataset (Markov idealisation + learned noise)")
    gr.Markdown(
        "Simulate an unlimited idealisation from a transition matrix, then dress "
        "it with the trained model's noise. Exports Time / Channels / Noisy Current."
    )
    with gr.Row():
        with gr.Column(scale=1):
            matrix = gr.Dataframe(
                value=DEFAULT_P, datatype="number", row_count=(4, "dynamic"),
                col_count=(4, "dynamic"),
                label="Transition probability matrix (each row sums to 1)",
            )
            open_states = gr.Textbox("2,3", label="Open-state indices (comma-separated)")
            n_channels_g = gr.Slider(1, 10, value=1, step=1, label="Number of channels")
            preview_len = gr.Slider(1000, 100_000, value=PREVIEW_STEPS, step=1000,
                                    label="Idealisation preview length (samples)")
            n_steps_g = gr.Slider(1000, 50_000_000, value=100_000, step=1000, label="Rows to generate")
            decimate = gr.Slider(1, 50, value=1, step=1, label="Decimation factor")
            seed_g = gr.Number(0, label="Seed")
            ckpt_g = gr.Textbox(CKPT_PATH, label="Trained model checkpoint")
            preview_btn = gr.Button("Preview idealisation")
            gen_btn = gr.Button("Generate & download Parquet", variant="primary")

        with gr.Column(scale=2):
            ideal_preview = gr.Plot(label="Idealisation preview")
            gen_status = gr.Textbox(label="Export status")
            gen_file = gr.File(label="Download Parquet")

    gen_btn.click(
        generate_and_export,
        inputs=[matrix, open_states, n_channels_g, n_steps_g, decimate, seed_g, ckpt_g],
        outputs=[matrix, gen_file, gen_status],
    )

    preview_btn.click(
        preview_ideal,
        inputs=[matrix, open_states, n_channels_g, preview_len, folder, ckpt_g],
        outputs=[matrix, ideal_preview],
    )

if __name__ == "__main__":
    demo.launch(allowed_paths=["help.html"])