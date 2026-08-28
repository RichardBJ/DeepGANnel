#!/usr/bin/env python3
"""
Annotation review tool (staged pipeline).

Three stages, each runnable on its own so you don't redo expensive work:
  1. Train CNN        (slow; skip if you already have a checkpoint)
  2. Compute emissions(runs the CNN over the file; cached)
  3. Decode & flag    (fast; HMM + flagging -- re-run freely to tune)

The flag score is the model's posterior probability that the HUMAN's label is
WRONG at that point -- not the model's confidence in its own pick. Nothing is
auto-corrected: tick 'accept' on events you agree with and export a copy with 
a new'Channels' column.

Run:
    python review_app.py
"""

import os
import argparse
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import gradio as gr

from diffusion_103 import _read_table, normalise_column_name, infer_structure
from inverse_model import flag_events, load_inverse_model, predict_logprobs, train_model
from hmm_idealise import idealise_channels

CKPT_PATH = "ion_channel_inverse_model.pt"


def _read_labelled_csv(path):
    df = _read_table(path)
    norm = {normalise_column_name(c): c for c in df.columns}
    chan_col = norm.get("channels")
    cur_col = norm.get("noisy current") or norm.get("current")
    time_col = norm.get("time")
    if chan_col is None or cur_col is None:
        m = infer_structure(df, verbose=False)
        chan_col, cur_col, time_col = m["channels"], m["current"], m["time"]
    chan = np.rint(pd.to_numeric(df[chan_col], errors="coerce")).to_numpy()
    cur = pd.to_numeric(df[cur_col], errors="coerce").to_numpy(dtype=np.float32)
    time = pd.to_numeric(df[time_col], errors="coerce").to_numpy() if time_col else np.arange(len(df))
    return time, chan, cur, df, (time_col, chan_col, cur_col)


# ---------------- Stage 1: train CNN ----------------

def train_cnn(folder, epochs, ckpt_out):
    if not os.path.isdir(folder):
        raise gr.Error(f"No data folder '{folder}'.")
    args = argparse.Namespace(
        folder=folder, seq_len=256, stride=8, batch_size=32, epochs=int(epochs),
        lr=1e-3, weight_decay=1e-4, val_fraction=0.1, base_channels=64,
        class_weight="sqrt", seed=42, cpu=False, save_model=ckpt_out)
    train_model(args)   # progress prints to the terminal
    return f"Training done. Saved to '{ckpt_out}'. Now compute emissions."


# ---------------- Stage 2: compute emissions (cached) ----------------

def compute_emissions(csv_path, ckpt_path):
    if not os.path.exists(csv_path):
        raise gr.Error(f"No file at '{csv_path}'.")
    if not os.path.exists(ckpt_path):
        raise gr.Error(f"No CNN checkpoint at '{ckpt_path}'. Train it in stage 1.")

    time, chan, cur, df, cols = _read_labelled_csv(csv_path)

    # Emissions are per-file and expensive, so cache them on disk next to the CSV.
    # Re-used automatically unless the CSV or the checkpoint has changed since.
    cache = csv_path + ".emis.npz"
    sig = f"{os.path.getmtime(ckpt_path):.0f}|{os.path.getmtime(csv_path):.0f}|{len(cur)}"
    logprob = level_values = None
    note = "computed"
    if os.path.exists(cache):
        try:
            d = np.load(cache, allow_pickle=True)
            if str(d["sig"]) == sig:
                logprob, level_values = d["logprob"], d["level_values"]
                note = "loaded from cache"
        except Exception:
            logprob = None

    if logprob is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model, norm, seq_len = load_inverse_model(ckpt_path, device)
        logprob, level_values = predict_logprobs(model, cur, norm, device,
                                                 seq_len=seq_len, stride=seq_len // 2)
        try:
            np.savez_compressed(cache, logprob=logprob.astype(np.float32),
                                level_values=np.asarray(level_values), sig=sig)
        except Exception:
            pass

    level_values = np.asarray(level_values)
    if not np.array_equal(level_values, np.arange(level_values.max() + 1)):
        raise gr.Error(f"Channel model needs contiguous levels 0..N; CNN has {list(level_values)}.")

    cur_lo, cur_hi = float(np.nanmin(cur)), float(np.nanmax(cur))
    cpad = 0.05 * (cur_hi - cur_lo + 1e-9)
    state = {"time": time, "chan": chan, "cur": cur, "df": df, "cols": cols,
             "logprob": logprob, "level_values": level_values,
             "y_lo": cur_lo - cpad, "y_hi": cur_hi + cpad,
             "lev_lo": int(np.min(np.rint(chan))), "lev_hi": int(level_values.max()),
             "events": pd.DataFrame()}
    status = (f"Emissions {note}: {len(cur)} samples, levels {list(level_values)}. "
              f"Now press Decode & flag (cheap -- re-run to tune).")
    return state, status


# ---------------- Stage 3: decode & flag (fast, re-runnable) ----------------

def decode(state, temperature, min_len, anomaly_thresh):
    if not state or "logprob" not in state:
        raise gr.Error("Compute emissions first (stage 2).")
    logprob = np.asarray(state["logprob"])
    chan = state["chan"]
    level_values = np.asarray(state["level_values"])
    K = logprob.shape[0]

    # Temperature-soften the CNN probabilities, then decode.
    T = max(float(temperature), 1e-3)
    s = logprob / T
    s = s - (s.max(0, keepdims=True) + np.log(np.exp(s - s.max(0, keepdims=True)).sum(0, keepdims=True)))
    path, _, (a_fit, b_fit), gamma = idealise_channels(s, log_prior=None,
                                                       verbose=True, show_matrix=True)
    pred = level_values[path]

    # Anomaly = posterior probability the HUMAN's label is wrong at each sample.
    human_idx = np.clip(np.rint(chan).astype(int), 0, K - 1)
    human_post = gamma[human_idx, np.arange(len(chan))]
    anomaly = (1.0 - human_post).astype(np.float32)

    events = flag_events(chan, pred, anomaly, min_len=int(min_len),
                         conf_thresh=float(anomaly_thresh))
    flagged = int(events["length"].sum()) if len(events) else 0
    status = (f"{len(events)} suspect event(s) covering {flagged}/{len(chan)} samples "
              f"({100 * flagged / max(1, len(chan)):.2f}%). "
              f"Agreement: {100 * np.mean(pred == np.rint(chan)):.2f}%. "
              f"Fitted a={a_fit:.2e}, b={b_fit:.2e}.")

    table = events.copy()
    table["accept"] = 0
    state = {**state, "pred": pred, "conf": anomaly, "events": events}
    sel_max = max(1, len(events) - 1)
    first_plot = view_event(state, 0)
    return state, table, status, gr.update(maximum=sel_max, value=0), first_plot


# ---------------- review plotting ----------------

def view_event(state, idx, fit=False, xpad=None, ymin=None, ymax=None):
    if not state or "events" not in state or len(state["events"]) == 0:
        return None
    events = state["events"]
    idx = int(np.clip(idx, 0, len(events) - 1))
    row = events.iloc[idx]
    s, e = int(row["start"]), int(row["end"])
    pad = int(xpad) if xpad else max(50, 3 * (e - s + 1))
    a, b = max(0, s - pad), min(len(state["cur"]), e + pad + 1)

    cur = state["cur"][a:b]
    chan = state["chan"][a:b]
    pred = state["pred"][a:b]
    xs = np.arange(a, b)

    plt.close("all")
    fig, axes = plt.subplots(2, 1, figsize=(7, 4), sharex=True)
    axes[0].plot(xs, cur, color="tab:blue", lw=0.8)
    axes[0].axvspan(s, e, color="tab:red", alpha=0.15)
    axes[0].set_ylabel("noisy current")
    if ymin is not None and ymax is not None and ymax > ymin:
        axes[0].set_ylim(float(ymin), float(ymax))     # manual override wins
    elif fit:
        lo, hi = float(np.min(cur)), float(np.max(cur))
        p = 0.05 * (hi - lo + 1e-9)
        axes[0].set_ylim(lo - p, hi + p)
    else:
        axes[0].set_ylim(state["y_lo"], state["y_hi"])
    axes[0].set_title(f"Event {idx}: samples {s}-{e}  |  human={int(row['human'])} "
                      f"model={int(row['model'])}  P(human wrong)={row['mean_conf']}"
                      f"{'  [zoomed]' if fit else ''}")
    axes[1].plot(xs, chan, color="tab:gray", drawstyle="steps-post", label="human")
    axes[1].plot(xs, pred, color="tab:orange", drawstyle="steps-post", alpha=0.8, label="model")
    axes[1].axvspan(s, e, color="tab:red", alpha=0.15)
    axes[1].set_ylabel("open channels")
    axes[1].set_ylim(state["lev_lo"] - 0.5, state["lev_hi"] + 0.5)
    axes[1].set_xlabel("sample")
    axes[1].legend(loc="upper right")
    fig.tight_layout()
    return fig

def view_window(state, frac, half):
    if not state or "pred" not in state:
        return None
    n = len(state["cur"])
    c = int(np.clip(float(frac), 0.0, 1.0) * (n - 1))
    h = int(half)
    a, b = max(0, c - h), min(n, c + h + 1)
    cur, chan, pred = state["cur"][a:b], state["chan"][a:b], state["pred"][a:b]
    xs = np.arange(a, b)
    plt.close("all")
    fig, axes = plt.subplots(2, 1, figsize=(7, 4), sharex=True)
    axes[0].plot(xs, cur, color="tab:blue", lw=0.8)
    axes[0].set_ylabel("noisy current")
    axes[0].set_ylim(state["y_lo"], state["y_hi"])
    axes[0].set_title(f"Browse: samples {a}-{b - 1}")
    axes[1].plot(xs, chan, color="tab:gray", drawstyle="steps-post", label="human")
    axes[1].plot(xs, pred, color="tab:orange", drawstyle="steps-post", alpha=0.8, label="model")
    axes[1].set_ylabel("open channels")
    axes[1].set_ylim(state["lev_lo"] - 0.5, state["lev_hi"] + 0.5)
    axes[1].set_xlabel("sample")
    axes[1].legend(loc="upper right")
    fig.tight_layout()
    return fig


def fit_event(state, idx, xpad=None, ymin=None, ymax=None):
    return view_event(state, idx, fit=True, xpad=xpad, ymin=ymin, ymax=ymax)


def view_full(state, idx, xpad, ymin=None, ymax=None):
    return view_event(state, idx, fit=False, xpad=xpad, ymin=ymin, ymax=ymax)


def export(state, edited, out_path):
    if not state or "events" not in state or len(state["events"]) == 0:
        raise gr.Error("Nothing to export -- decode first.")
    events = state["events"]
    tbl = edited.values if hasattr(edited, "values") else np.asarray(edited)
    accept_col = tbl[:, -1]
    suggested = np.rint(state["chan"]).astype(int).copy()
    n_accepted = 0
    for i, acc in enumerate(accept_col):
        if float(acc) >= 0.5 and i < len(events):
            r = events.iloc[i]
            suggested[int(r["start"]): int(r["end"]) + 1] = int(r["model"])
            n_accepted += 1
    out = state["df"].copy()
    out["Channels"] = suggested[: len(out)]
    out.to_csv(out_path, index=False)
    return out_path, f"Wrote {out_path}. Applied {n_accepted} accepted event(s)."


def run_all(retrain, train_folder, epochs, ckpt_out, csv_path, ckpt,
            temperature, min_len, anomaly_thresh):
    tstat = "Training skipped (reusing existing checkpoint)."
    use_ckpt = ckpt
    if retrain:
        tstat = train_cnn(train_folder, epochs, ckpt_out)
        use_ckpt = ckpt_out
    state, _ = compute_emissions(csv_path, use_ckpt)
    state, table, status, idx_update, plot = decode(state, temperature, min_len, anomaly_thresh)
    return tstat, state, table, status, idx_update, plot


EVENT_COLS = ["start", "end", "length", "human", "model", "mean_conf", "direction", "accept"]


def _coerce(table):
    arr = table.values if isinstance(table, pd.DataFrame) else np.array(table, dtype=object)
    return np.array(arr, dtype=object)


def accept_all(table):
    arr = _coerce(table)
    if arr.size == 0:
        return table
    arr[:, -1] = 1
    return pd.DataFrame(arr, columns=EVENT_COLS)


def clear_all(table):
    arr = _coerce(table)
    if arr.size == 0:
        return table
    arr[:, -1] = 0
    return pd.DataFrame(arr, columns=EVENT_COLS)


def accept_above(table, thresh):
    arr = _coerce(table)
    if arr.size == 0:
        return table
    for i in range(len(arr)):
        try:
            sc = float(arr[i][5])          # mean_conf = P(human wrong)
        except (ValueError, TypeError):
            sc = 0.0
        arr[i][-1] = 1 if sc >= float(thresh) else 0
    return pd.DataFrame(arr, columns=EVENT_COLS)


with gr.Blocks(title="Annotation review") as demo:
    gr.Markdown("## Idealisation review: find likely-wrong events, you decide")
    st = gr.State()

    with gr.Row():
        with gr.Column(scale=1):
            with gr.Group():
                gr.Markdown("**Run everything in one go**")
                retrain = gr.Checkbox(value=True,
                                      label="Retrain CNN first (uncheck to reuse the checkpoint)")
                run_all_btn = gr.Button("Run all: train -> emissions -> decode", variant="primary")

            with gr.Group():
                gr.Markdown("**Stage 1 - Train CNN** (Can skip if you've a checkpoint)")
                train_folder = gr.Textbox("./pages", label="Training data folder")
                epochs = gr.Slider(1, 100, value=20, step=1, label="Epochs")
                ckpt_out = gr.Textbox(CKPT_PATH, label="Save checkpoint to")
                train_btn = gr.Button("Train CNN (slow; watch terminal)")
                train_status = gr.Textbox(label="Training status")

            with gr.Group():
                gr.Markdown("**Stage 2 - Compute emissions**")
                csv_path = gr.Textbox("./pages/your_file.csv", label="Labelled CSV to review")
                ckpt = gr.Textbox(CKPT_PATH, label="CNN checkpoint")
                emis_btn = gr.Button("Compute emissions", variant="primary")

            with gr.Group():
                gr.Markdown("**Stage 3 - Decode & flag** (cheap; re-run to tune)")
                temperature = gr.Slider(1.0, 50.0, value=5.0, step=0.5,
                                        label="Emission temperature: higher = more smoothing")
                min_len = gr.Slider(1, 200, value=10, step=1, label="Min event length (samples)")
                anomaly_thresh = gr.Slider(0.5, 0.9999, value=0.9, step=0.005,
                                           label="Min P(human label is wrong) to flag")
                decode_btn = gr.Button("Decode & flag", variant="primary")

            with gr.Group():
                gr.Markdown("**Export**")
                out_path = gr.Textbox("reviewed_dataset.csv", label="Export filename")
                export_btn = gr.Button("Export accepted corrections")

        with gr.Column(scale=2):
            status = gr.Textbox(label="Summary")
            event_plot = gr.Plot(label="Selected event (human vs model)")
            # Event chooser + view controls sit directly under the figure.
            event_idx = gr.Slider(0, 1, value=0, step=1, label="View event #")
            browse = gr.Slider(0, 1, value=0, step=0.001,
                               label="Browse anywhere (fraction of whole file)")
            with gr.Row():
                xwin = gr.Slider(50, 20000, value=300, step=50,
                                 label="X window: samples each side of the event")
                zoom_btn = gr.Button("Fit Y to this event (resets on step)")
            with gr.Row():
                ymin = gr.Number(value=None, label="Y min (blank = auto)")
                ymax = gr.Number(value=None, label="Y max (blank = auto)")
            events_table = gr.Dataframe(
                headers=EVENT_COLS,
                datatype=["number", "number", "number", "number", "number", "number", "str", "number"],
                label="Suspect events (mean_conf = P(human wrong); accept=1 applies on export)",
                interactive=True,
            )
            with gr.Row():
                accept_all_btn = gr.Button("Accept all")
                clear_all_btn = gr.Button("Clear all")
                accept_thresh = gr.Number(value=0.99, label="score >=")
                accept_above_btn = gr.Button("Accept those >= score")
            export_status = gr.Textbox(label="Export status")
            export_file = gr.File(label="Download reviewed CSV")

    run_all_btn.click(run_all,
                      inputs=[retrain, train_folder, epochs, ckpt_out, csv_path, ckpt,
                              temperature, min_len, anomaly_thresh],
                      outputs=[train_status, st, events_table, status, event_idx, event_plot])
    train_btn.click(train_cnn, inputs=[train_folder, epochs, ckpt_out], outputs=[train_status])
    emis_btn.click(compute_emissions, inputs=[csv_path, ckpt], outputs=[st, status])
    decode_btn.click(decode, inputs=[st, temperature, min_len, anomaly_thresh],
                     outputs=[st, events_table, status, event_idx, event_plot])
    event_idx.change(view_full, inputs=[st, event_idx, xwin, ymin, ymax], outputs=[event_plot])
    browse.change(view_window, inputs=[st, browse, xwin], outputs=[event_plot])
    xwin.change(view_full, inputs=[st, event_idx, xwin, ymin, ymax], outputs=[event_plot])
    ymin.change(view_full, inputs=[st, event_idx, xwin, ymin, ymax], outputs=[event_plot])
    ymax.change(view_full, inputs=[st, event_idx, xwin, ymin, ymax], outputs=[event_plot])
    zoom_btn.click(fit_event, inputs=[st, event_idx, xwin, ymin, ymax], outputs=[event_plot])
    export_btn.click(export, inputs=[st, events_table, out_path],
                     outputs=[export_file, export_status])
    accept_all_btn.click(accept_all, inputs=[events_table], outputs=[events_table])
    clear_all_btn.click(clear_all, inputs=[events_table], outputs=[events_table])
    accept_above_btn.click(accept_above, inputs=[events_table, accept_thresh], outputs=[events_table])


if __name__ == "__main__":
    demo.launch()
