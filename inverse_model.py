#!/usr/bin/env python3
"""
Inverse model: raw Noisy Current  ->  per-sample Channels level.

Purpose
-------
The conditional diffusion generator learns  idealisation -> current.
This is the other direction: a 1D segmenter that predicts the open-channel
count at every sample straight from the noisy current.

It is NOT an oracle. It is trained on the same human idealisation, so it learns
the human's habits too. What it gives you is *internal consistency*: where it
confidently predicts a different level from the human label, the local current
pattern disagrees with how similar current was labelled elsewhere. Those are the
spots worth a human eye -- not automatic corrections.

    model gives, per sample: predicted level + a confidence (max softmax prob).

A confident disagreement that sits BELOW the human label hints at a missed
closing (label too high); one that sits ABOVE hints at a noise dip read as a
transition (label too low).

Contract:
    train_model(args)                  -> trained ChannelSegmenter (+ checkpoint)
    predict_trace(model, current, ...)  -> (pred_level[N], confidence[N])
    flag_events(human, pred, conf, ...) -> DataFrame of suspect runs
"""

import os
import argparse

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, random_split

from diffusion_103 import load_pages, set_seed


# ------------------------------------------------------------
# 1. Backbone: dilated 1D residual CNN (TCN-style segmenter)
# ------------------------------------------------------------

class ResBlock(nn.Module):
    def __init__(self, ch, dilation, groups=8):
        super().__init__()
        pad = dilation
        self.norm1 = nn.GroupNorm(groups, ch)
        self.conv1 = nn.Conv1d(ch, ch, 3, padding=pad, dilation=dilation)
        self.norm2 = nn.GroupNorm(groups, ch)
        self.conv2 = nn.Conv1d(ch, ch, 3, padding=pad, dilation=dilation)

    def forward(self, x):
        h = self.conv1(F.silu(self.norm1(x)))
        h = self.conv2(F.silu(self.norm2(h)))
        return x + h


class ChannelSegmenter(nn.Module):
    """
    Input  : (B, 1, L) standardised current.
    Output : (B, n_levels, L) per-sample class logits.
    No downsampling, so any sequence length is fine -- handy at inference.
    """

    def __init__(self, n_levels, base=64, dilations=(1, 2, 4, 8, 16, 32)):
        super().__init__()
        self.in_conv = nn.Conv1d(1, base, 3, padding=1)
        self.blocks = nn.ModuleList([ResBlock(base, d) for d in dilations])
        self.out_norm = nn.GroupNorm(8, base)
        self.out_conv = nn.Conv1d(base, n_levels, 1)

    def forward(self, x):
        h = self.in_conv(x)
        for blk in self.blocks:
            h = blk(h)
        return self.out_conv(F.silu(self.out_norm(h)))


# ------------------------------------------------------------
# 2. Data: current windows -> integer level windows
# ------------------------------------------------------------

def build_inverse_dataset(pages, level_values, seq_len=256, stride=1):
    """X: (n, seq_len) current ; Y: (n, seq_len) class index in [0, n_levels)."""
    val_to_idx = {int(v): i for i, v in enumerate(level_values)}
    Xc, Yc = [], []
    for page in pages:
        chan = np.rint(page[:, 0]).astype(np.int64)
        cur = page[:, 1].astype(np.float32)
        if len(page) < seq_len:
            continue
        for i in range(0, len(page) - seq_len + 1, stride):
            seg_c = chan[i:i + seq_len]
            # Skip windows holding a level the model has no class for.
            if not np.all(np.isin(seg_c, level_values)):
                continue
            Xc.append(cur[i:i + seq_len])
            Yc.append(np.array([val_to_idx[int(v)] for v in seg_c], dtype=np.int64))
    if not Xc:
        raise ValueError("No usable windows; check seq_len against page lengths.")
    return np.asarray(Xc, np.float32), np.asarray(Yc, np.int64)


def class_weights(Y, n_levels, mode="inv"):
    """Per-class loss weights to stop the common levels drowning the rare ones."""
    counts = np.bincount(Y.reshape(-1), minlength=n_levels).astype(np.float64)
    counts = np.clip(counts, 1.0, None)
    if mode == "none":
        return None
    inv = counts.sum() / (n_levels * counts)     # mean ~ 1
    if mode == "sqrt":
        inv = np.sqrt(inv)
    return torch.tensor(inv, dtype=torch.float32)


# ------------------------------------------------------------
# 3. Inference + flagging
# ------------------------------------------------------------

@torch.no_grad()
def predict_trace(model, current, norm, device, seq_len=256, stride=128):
    """
    Run the segmenter over a whole trace with overlapping windows, averaging
    logits where they overlap. Returns (pred_level[N], confidence[N]).
    """
    model.eval()
    cur = (np.asarray(current, np.float32) - norm["cur_mean"]) / norm["cur_std"]
    N = len(cur)
    n_levels = model.out_conv.out_channels
    logit_sum = np.zeros((n_levels, N), np.float32)
    count = np.zeros(N, np.float32)

    starts = list(range(0, max(1, N - seq_len + 1), stride))
    if not starts or starts[-1] != N - seq_len:
        starts.append(max(0, N - seq_len))
    for s in starts:
        seg = cur[s:s + seq_len]
        x = torch.tensor(seg[None, None], dtype=torch.float32, device=device)
        lg = model(x)[0].cpu().numpy()           # (n_levels, len(seg))
        logit_sum[:, s:s + len(seg)] += lg
        count[s:s + len(seg)] += 1.0

    count[count == 0] = 1.0
    logits = logit_sum / count
    probs = np.exp(logits - logits.max(0, keepdims=True))
    probs /= probs.sum(0, keepdims=True)
    pred_idx = probs.argmax(0)
    conf = probs.max(0)
    level_values = np.asarray(norm["level_values"])
    return level_values[pred_idx], conf.astype(np.float32)


@torch.no_grad()
def predict_logprobs(model, current, norm, device, seq_len=256, stride=128):
    """
    Like predict_trace but returns the full per-sample log-posteriors (K, N)
    plus the level values -- these feed the HMM as learned emissions.
    """
    model.eval()
    cur = (np.asarray(current, np.float32) - norm["cur_mean"]) / norm["cur_std"]
    N = len(cur)
    n_levels = model.out_conv.out_channels
    logit_sum = np.zeros((n_levels, N), np.float32)
    count = np.zeros(N, np.float32)

    starts = list(range(0, max(1, N - seq_len + 1), stride))
    if not starts or starts[-1] != N - seq_len:
        starts.append(max(0, N - seq_len))
    for s in starts:
        seg = cur[s:s + seq_len]
        x = torch.tensor(seg[None, None], dtype=torch.float32, device=device)
        lg = model(x)[0].cpu().numpy()
        logit_sum[:, s:s + len(seg)] += lg
        count[s:s + len(seg)] += 1.0

    count[count == 0] = 1.0
    logits = logit_sum / count
    logZ = logits.max(0, keepdims=True) + np.log(np.exp(logits - logits.max(0, keepdims=True)).sum(0, keepdims=True))
    logprob = (logits - logZ).astype(np.float64)         # (K, N) log P(level | x)
    return logprob, np.asarray(norm["level_values"])


def flag_events(human, pred, conf, min_len=5, conf_thresh=0.8):
    """
    Group contiguous samples where the model confidently disagrees with the
    human label into events. Returns a DataFrame, one row per suspect run.
    """
    human = np.rint(np.asarray(human)).astype(int)
    pred = np.rint(np.asarray(pred)).astype(int)
    conf = np.asarray(conf)

    flag = (pred != human) & (conf >= conf_thresh)
    rows = []
    i, N = 0, len(flag)
    while i < N:
        if not flag[i]:
            i += 1
            continue
        j = i
        while j < N and flag[j]:
            j += 1
        seg = slice(i, j)
        if (j - i) >= min_len:
            h_mode = int(np.bincount(human[seg]).argmax())
            p_mode = int(np.bincount(pred[seg]).argmax())
            direction = ("model lower (possible missed closing / label too high)"
                         if p_mode < h_mode else
                         "model higher (possible noise dip read as event / label too low)")
            rows.append({
                "start": int(i),
                "end": int(j - 1),
                "length": int(j - i),
                "human": h_mode,
                "model": p_mode,
                "mean_conf": round(float(conf[seg].mean()), 3),
                "direction": direction,
            })
        i = j
    return pd.DataFrame(rows, columns=[
        "start", "end", "length", "human", "model", "mean_conf", "direction"
    ])


# ------------------------------------------------------------
# 4. Train / evaluate
# ------------------------------------------------------------

def _epoch(model, loader, device, weight, optimizer=None):
    train = optimizer is not None
    model.train(train)
    tot_loss, tot_n, correct, seen = 0.0, 0, 0, 0
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x.unsqueeze(1))
            loss = F.cross_entropy(logits, y, weight=weight)
            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            tot_loss += loss.item() * x.size(0)
            tot_n += x.size(0)
            correct += (logits.argmax(1) == y).sum().item()
            seen += y.numel()
    return tot_loss / tot_n, correct / seen


def train_model(args):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"Using device: {device}")

    pages = load_pages(args.folder, verbose=True)
    if not pages:
        raise SystemExit(f"No data found in {args.folder} -- the inverse model needs your real labelled CSVs.")

    level_values = sorted({int(v) for p in pages for v in np.rint(p[:, 0]).astype(int)})
    n_levels = len(level_values)
    print(f"Channel levels found: {level_values}")

    X, Y = build_inverse_dataset(pages, level_values, seq_len=args.seq_len, stride=args.stride)
    cur_mean, cur_std = float(X.mean()), float(X.std()) + 1e-8
    Xn = (X - cur_mean) / cur_std
    print(f"Dataset: current={Xn.shape}, labels={Y.shape}")

    weight = class_weights(Y, n_levels, mode=args.class_weight)
    if weight is not None:
        weight = weight.to(device)
        print("Class weights:", {level_values[i]: round(float(w), 2) for i, w in enumerate(weight.cpu())})

    ds = TensorDataset(torch.tensor(Xn), torch.tensor(Y))
    val_n = max(1, int(len(ds) * args.val_fraction))
    train_ds, val_ds = random_split(
        ds, [len(ds) - val_n, val_n],
        generator=torch.Generator().manual_seed(args.seed),
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    model = ChannelSegmenter(n_levels, base=args.base_channels).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    print("Trainable parameters:", sum(p.numel() for p in model.parameters()))

    norm = {"cur_mean": cur_mean, "cur_std": cur_std, "level_values": level_values}
    best_val = float("inf")
    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc = _epoch(model, train_loader, device, weight, optimizer)
        va_loss, va_acc = _epoch(model, val_loader, device, weight)
        print(f"Epoch {epoch:03d}/{args.epochs} | "
              f"train CE={tr_loss:.4g} acc={tr_acc:.3f} | val CE={va_loss:.4g} acc={va_acc:.3f}")
        if va_loss < best_val:
            best_val = va_loss
            if args.save_model:
                torch.save({
                    "model_state_dict": model.state_dict(),
                    "n_levels": n_levels,
                    "base_channels": args.base_channels,
                    "seq_len": args.seq_len,
                    "norm": norm,
                }, args.save_model)
    if args.save_model:
        print(f"Best model saved to: {args.save_model}")
    return model


def load_inverse_model(ckpt_path, device):
    blob = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = ChannelSegmenter(blob["n_levels"], base=blob["base_channels"]).to(device)
    model.load_state_dict(blob["model_state_dict"])
    model.eval()
    return model, blob["norm"], blob["seq_len"]


def parse_args():
    p = argparse.ArgumentParser(description="Inverse segmenter: current -> channel level.")
    p.add_argument("--folder", type=str, default="./pages")
    p.add_argument("--seq-len", type=int, default=256)
    p.add_argument("--stride", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--val-fraction", type=float, default=0.1)
    p.add_argument("--base-channels", type=int, default=64)
    p.add_argument("--class-weight", choices=["none", "inv", "sqrt"], default="sqrt",
                   help="rare-level reweighting for the loss (ties into the imbalance fix).")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--save-model", type=str, default="ion_channel_inverse_model.pt")
    args = p.parse_args()
    if args.save_model == "":
        args.save_model = None
    return args


if __name__ == "__main__":
    train_model(parse_args())
