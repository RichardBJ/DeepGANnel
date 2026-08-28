#!/usr/bin/env python3
"""
Conditional diffusion generator.

Goal
----
Synthesise realistic data, conditioned on


Expected CSV files in ./pages/ with columns:

    Time, Channels, Noisy Current

The model learns:

    idealised channel state  ->  realistic raw current (with noise/artefacts)

Generator contract (so a different backbone, e.g. a LoRA-fine-tuned Chronos,
can be dropped in for a head-to-head later):

    model.loss(cond, x0)   -> scalar training loss
    model.sample(cond)     -> generated raw trace, same shape as cond
"""

import os
import glob
import math
import random
import argparse

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, random_split

try:
    from numba import njit
    _HAVE_NUMBA = True
except ImportError:
    _HAVE_NUMBA = False


# ------------------------------------------------------------
# 0. Reproducibility
# ------------------------------------------------------------

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ------------------------------------------------------------
# 1. Denoiser backbone: small 1D U-Net
# ------------------------------------------------------------

def timestep_embedding(t, dim):
    """Standard sinusoidal embedding of diffusion timestep."""
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000.0) * torch.arange(half, device=t.device) / max(half - 1, 1)
    )
    args = t.float()[:, None] * freqs[None]
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    if dim % 2:
        emb = F.pad(emb, (0, 1))
    return emb


class ResBlock1D(nn.Module):
    def __init__(self, in_ch, out_ch, time_dim, groups=8):
        super().__init__()
        self.norm1 = nn.GroupNorm(groups, in_ch)
        self.conv1 = nn.Conv1d(in_ch, out_ch, 3, padding=1)
        self.time = nn.Linear(time_dim, out_ch)
        self.norm2 = nn.GroupNorm(groups, out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, 3, padding=1)
        self.skip = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, temb):
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.time(F.silu(temb))[:, :, None]
        h = self.conv2(F.silu(self.norm2(h)))
        return h + self.skip(x)


class UNet1D(nn.Module):
    """
    Depth-2 U-Net. Input is 2 channels (noisy current + conditioning
    idealisation); output is 1 channel (predicted noise). Sequence length must
    be divisible by 4.
    """

    def __init__(self, base=64, time_dim=128, in_channels=2):
        super().__init__()
        c1, c2 = base, base * 2

        self.time_dim = time_dim
        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )

        self.in_conv = nn.Conv1d(in_channels, c1, 3, padding=1)

        self.rb_d1 = ResBlock1D(c1, c1, time_dim)
        self.down1 = nn.Conv1d(c1, c1, 4, stride=2, padding=1)

        self.rb_d2 = ResBlock1D(c1, c2, time_dim)
        self.down2 = nn.Conv1d(c2, c2, 4, stride=2, padding=1)

        self.rb_mid = ResBlock1D(c2, c2, time_dim)

        self.up2 = nn.ConvTranspose1d(c2, c2, 4, stride=2, padding=1)
        self.rb_u2 = ResBlock1D(c2 + c2, c1, time_dim)

        self.up1 = nn.ConvTranspose1d(c1, c1, 4, stride=2, padding=1)
        self.rb_u1 = ResBlock1D(c1 + c1, c1, time_dim)

        self.out_norm = nn.GroupNorm(8, c1)
        self.out_conv = nn.Conv1d(c1, 1, 3, padding=1)

    def forward(self, inp, t):
        temb = self.time_mlp(timestep_embedding(t, self.time_dim))

        h = self.in_conv(inp)

        h1 = self.rb_d1(h, temb)
        h = self.down1(h1)

        h2 = self.rb_d2(h, temb)
        h = self.down2(h2)

        h = self.rb_mid(h, temb)

        h = self.up2(h)
        h = self.rb_u2(torch.cat([h, h2], dim=1), temb)

        h = self.up1(h)
        h = self.rb_u1(torch.cat([h, h1], dim=1), temb)

        return self.out_conv(F.silu(self.out_norm(h)))


# ------------------------------------------------------------
# 2. Conditional diffusion wrapper
# ------------------------------------------------------------

class ConditionalDiffusion(nn.Module):
    """DDPM over the raw current, conditioned on the idealisation."""

    def __init__(self, base_channels=64, time_dim=128, timesteps=1000, cond_channels=1):
        super().__init__()
        self.timesteps = timesteps
        self.net = UNet1D(base=base_channels, time_dim=time_dim,
                          in_channels=1 + cond_channels)

        betas = torch.linspace(1e-4, 0.02, timesteps)
        alphas = 1.0 - betas
        acp = torch.cumprod(alphas, dim=0)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("acp", acp)
        self.register_buffer("sqrt_acp", torch.sqrt(acp))
        self.register_buffer("sqrt_one_minus_acp", torch.sqrt(1.0 - acp))

    def _net(self, x_t, t, cond):
        if cond.dim() == 2:
            cond = cond.unsqueeze(1)          # (B, L) -> (B, 1, L)
        inp = torch.cat([x_t.unsqueeze(1), cond], dim=1)
        return self.net(inp, t).squeeze(1)

    def loss(self, cond, x0):
        b = x0.size(0)
        t = torch.randint(0, self.timesteps, (b,), device=x0.device)
        noise = torch.randn_like(x0)
        x_t = (
            self.sqrt_acp[t][:, None] * x0
            + self.sqrt_one_minus_acp[t][:, None] * noise
        )
        eps_pred = self._net(x_t, t, cond)
        return F.mse_loss(eps_pred, noise)

    @torch.no_grad()
    def sample(self, cond):
        """Generate a raw trace for each conditioning sequence in `cond`."""
        x = torch.randn(cond.size(0), cond.size(-1), device=cond.device)
        for i in reversed(range(self.timesteps)):
            t = torch.full((cond.size(0),), i, device=cond.device, dtype=torch.long)
            eps = self._net(x, t, cond)
            alpha = self.alphas[i]
            acp = self.acp[i]
            coef = (1.0 - alpha) / torch.sqrt(1.0 - acp)
            mean = (x - coef * eps) / torch.sqrt(alpha)
            if i > 0:
                x = mean + torch.sqrt(self.betas[i]) * torch.randn_like(x)
            else:
                x = mean
        return x

    @torch.no_grad()
    def sample_inpaint(self, cond, known, known_len):
        """
        Like sample(), but the first `known_len` samples are clamped to `known`
        at every denoising step, so the block continues seamlessly from the
        previously generated tail (RePaint-style context carry).
        """
        x = torch.randn(cond.size(0), cond.size(-1), device=cond.device)
        for i in reversed(range(self.timesteps)):
            if known_len > 0:
                if i > 0:
                    noised = (
                        self.sqrt_acp[i] * known
                        + self.sqrt_one_minus_acp[i] * torch.randn_like(known)
                    )
                else:
                    noised = known
                x[:, :known_len] = noised
            t = torch.full((cond.size(0),), i, device=cond.device, dtype=torch.long)
            eps = self._net(x, t, cond)
            alpha = self.alphas[i]
            acp = self.acp[i]
            coef = (1.0 - alpha) / torch.sqrt(1.0 - acp)
            mean = (x - coef * eps) / torch.sqrt(alpha)
            if i > 0:
                x = mean + torch.sqrt(self.betas[i]) * torch.randn_like(x)
            else:
                x = mean
        if known_len > 0:
            x[:, :known_len] = known
        return x


@torch.no_grad()
def generate_long_current(model, cond, device, block=8192*4, context=512, progress=None):
    """
    Generate a current trace as long as `cond` (standardised conditioning) with
    no window seams: one convolutional pass when short enough, otherwise
    overlapping blocks bridged by context-carry inpainting.

    `cond` may be 1D (N,) for a single conditioning channel, or 2D (C, N) when
    several conditioning channels are stacked (e.g. idealisation + envelope).
    """
    model.eval()
    block = max(4, (block // 4) * 4)
    context = max(0, (context // 4) * 4)

    cond = np.atleast_2d(cond)                  # (C, N)
    N = cond.shape[-1]
    pad = (-N) % 4
    cpad = (np.concatenate([cond, np.repeat(cond[:, -1:], pad, axis=1)], axis=1)
            if pad else cond.copy())
    M = cpad.shape[-1]
    out = np.zeros(M, dtype=np.float32)

    if M <= block:
        c = torch.tensor(cpad[None], dtype=torch.float32, device=device)
        out[:] = model.sample(c).cpu().numpy()[0]
        return out[:N]

    step = max(4, block - context)
    n_blocks = (M - block) // step + 2          # safe upper bound
    start, first, done = 0, True, 0
    while True:
        if start + block > M:
            start = M - block
        c = torch.tensor(cpad[:, start:start + block][None], dtype=torch.float32, device=device)
        if first:
            gen = model.sample(c).cpu().numpy()[0]
            first = False
        else:
            known = torch.tensor(out[start:start + context][None], dtype=torch.float32, device=device)
            gen = model.sample_inpaint(c, known, context).cpu().numpy()[0]
        out[start:start + block] = gen
        done += 1
        if progress is not None:
            progress(min(done / n_blocks, 1.0))
        if start == M - block:
            break
        start += step

    return out[:N]


# ------------------------------------------------------------
# 3. Robust data loading
# ------------------------------------------------------------

def normalise_column_name(name: str) -> str:
    return (
        str(name)
        .strip()
        .lower()
        .replace("_", " ")
        .replace("-", " ")
    )


def _read_table(path):
    if path.lower().endswith(".parquet"):
        return pd.read_parquet(path)
    return pd.read_csv(path)


def infer_structure(df, verbose=True):
    """
    Work out which columns are Time / Channels / Noisy Current when the headers
    are not the expected names.

      Time     : monotonic, increasing by a constant step (may be absent).
      Channels : integer-valued (even if stored as float), few discrete levels,
                 not monotonic.
      Noisy Current : continuous, many distinct values (the noisy one).

    Returns dict(time=name|None, channels=name, current=name).
    """
    feats = {}
    for c in df.columns:
        s = pd.to_numeric(df[c], errors="coerce").dropna().to_numpy()
        if s.size < 3:
            continue
        d = np.diff(s)
        time_like = bool(np.all(d > 0) and np.allclose(d, d[0], rtol=1e-3, atol=1e-9))
        feats[c] = {
            "is_int": bool(np.allclose(s, np.round(s))),
            "n_unique": int(np.unique(s).size),
            "frac_unique": float(np.unique(s).size / s.size),
            "time_like": time_like,
        }

    if not feats:
        raise ValueError(f"Could not read any numeric columns from: {list(df.columns)}")

    # Time: a constant-step increasing column (optional).
    time_col = next((c for c, f in feats.items() if f["time_like"]), None)

    rest = [c for c in feats if c != time_col]

    # Channels: integer-valued with the fewest distinct levels.
    chan_cands = sorted(
        (c for c in rest if feats[c]["is_int"]),
        key=lambda c: feats[c]["n_unique"],
    )
    channels_col = chan_cands[0] if chan_cands else None

    # Noisy current: most continuous remaining column.
    cur_cands = sorted(
        (c for c in rest if c != channels_col),
        key=lambda c: feats[c]["frac_unique"],
        reverse=True,
    )
    current_col = cur_cands[0] if cur_cands else None

    if channels_col is None or current_col is None:
        raise ValueError(
            "Could not infer Channels and/or Noisy Current columns.\n"
            f"Column features: {feats}"
        )

    if verbose:
        print(
            f"  inferred structure -> time={time_col!r}, "
            f"channels={channels_col!r}, current={current_col!r}"
        )

    return {"time": time_col, "channels": channels_col, "current": current_col}


def load_pages(folder="./pages", verbose=True):
    """Load all CSV/Parquet files. Returns arrays with columns: channels, current."""
    files = sorted(
        glob.glob(os.path.join(folder, "*.csv"))
        + glob.glob(os.path.join(folder, "*.parquet"))
    )

    if not files:
        if verbose:
            print(f"No CSV or Parquet files found in {folder}")
        return []

    pages = []

    for f in files:
        df = _read_table(f)

        normalised = {normalise_column_name(col): col for col in df.columns}

        channel_col = normalised.get("channels")
        current_col = normalised.get("noisy current")
        if current_col is None:
            current_col = normalised.get("current")

        # Headers don't match the expected names -> work it out.
        if channel_col is None or current_col is None:
            if verbose:
                print(f"{os.path.basename(f)}: unrecognised headers {list(df.columns)}")
            mapping = infer_structure(df, verbose=verbose)
            channel_col = mapping["channels"]
            current_col = mapping["current"]

        page_df = pd.DataFrame({
            "channels": pd.to_numeric(df[channel_col], errors="coerce"),
            "current": pd.to_numeric(df[current_col], errors="coerce"),
        }).dropna()

        if len(page_df) == 0:
            raise ValueError(f"File {f} contains no valid numeric rows.")

        if verbose:
            print(
                f"{os.path.basename(f)}: {len(page_df)} rows, "
                f"channels={page_df['channels'].min():.3g}..{page_df['channels'].max():.3g}, "
                f"current={page_df['current'].min():.4g}..{page_df['current'].max():.4g}"
            )

        pages.append(page_df[["channels", "current"]].to_numpy())

    return pages


def generate_synthetic_page(n=5000, max_channels=3, dwell=40, unit=5.0, seed=0):
    """
    Fallback toy data: a random-telegraph idealisation low-pass filtered into a
    realistic-ish noisy current. Temporal structure makes the diffusion task
    non-trivial.
    """
    rng = np.random.default_rng(seed)

    channels = np.zeros(n, dtype=float)
    state = 0
    i = 0
    while i < n:
        length = max(1, int(rng.exponential(dwell)))
        channels[i:i + length] = state
        state = rng.integers(0, max_channels + 1)
        i += length

    clean = channels * unit
    # crude low-pass (filter artefact) + coloured-ish noise
    kernel = np.ones(8) / 8.0
    filtered = np.convolve(clean, kernel, mode="same")
    noise = rng.normal(0, 0.8, size=n)
    current = filtered + noise

    return np.stack([channels, current], axis=1)


def _step_chain_py(cdf, r, s):
    """Plain-Python fallback: advance one Markov chain through r.shape[0] steps."""
    n = r.shape[0]
    states = np.empty(n, dtype=np.int64)
    for t in range(n):
        states[t] = s
        s = int(np.searchsorted(cdf[s], r[t]))
    return states, s


if _HAVE_NUMBA:
    @njit(cache=True)
    def _step_chain_numba(cdf, r, s):
        n = r.shape[0]
        states = np.empty(n, dtype=np.int64)
        for t in range(n):
            states[t] = s
            s = np.searchsorted(cdf[s], r[t])
        return states, s

    _step_chain = _step_chain_numba
else:
    _step_chain = _step_chain_py


def simulate_markov_idealisation(P, open_states, n_channels=1, n_steps=100000, seed=None, progress=None):
    """
    Generate an idealised Channels sequence from a discrete-time Markov chain,
    mirroring the Large-Scale Markov Chain Simulator.

    P            : (S, S) transition probability matrix (each row sums to 1).
    open_states  : iterable of state indices counted as OPEN (e.g. {2, 3}).
    n_channels   : number of independent channels summed together.
    n_steps      : length of the sequence.

    Returns an integer array (0 .. n_channels) of open-channel counts.
    """
    P = np.asarray(P, dtype=float)
    S = P.shape[0]
    if P.shape != (S, S):
        raise ValueError(f"Transition matrix must be square, got {P.shape}.")
    if not np.allclose(P.sum(axis=1), 1.0, atol=1e-6):
        raise ValueError("Each row of the transition matrix must sum to 1.")

    rng = np.random.default_rng(seed)
    cdf = np.cumsum(P, axis=1)
    open_mask = np.zeros(S, dtype=bool)
    open_mask[list(open_states)] = True

    channels = np.zeros(n_steps, dtype=np.int64)
    total_steps = int(n_channels) * n_steps
    chunk = min(1_000_000, n_steps)
    for c in range(int(n_channels)):
        r = rng.random(n_steps)
        s = int(rng.integers(0, S))
        states = np.empty(n_steps, dtype=np.int64)
        done = 0
        while done < n_steps:
            end = min(done + chunk, n_steps)
            states[done:end], s = _step_chain(cdf, r[done:end], s)
            done = end
            if progress is not None:
                progress((c * n_steps + done) / total_steps)
        channels += open_mask[states]

    return channels


def make_windows(page, seq_len=256, stride=1):
    X, Y = [], []
    if len(page) < seq_len:
        return np.empty((0, seq_len)), np.empty((0, seq_len))

    for i in range(0, len(page) - seq_len + 1, stride):
        window = page[i:i + seq_len]
        X.append(window[:, 0].astype(np.float32))   # idealisation (condition)
        Y.append(window[:, 1].astype(np.float32))   # raw current  (target)

    return np.asarray(X), np.asarray(Y)


def build_dataset(pages, seq_len=256, stride=1):
    X_all, Y_all = [], []
    for page in pages:
        X, Y = make_windows(page, seq_len=seq_len, stride=stride)
        if len(X) > 0:
            X_all.append(X)
            Y_all.append(Y)

    if not X_all:
        raise ValueError(
            "No usable windows were created. "
            "Check that your files contain more rows than seq_len."
        )

    return np.concatenate(X_all, 0), np.concatenate(Y_all, 0)


# ------------------------------------------------------------
# 3b. Coarse/fine cascade helpers (the "bigger picture")
# ------------------------------------------------------------

def block_mean(x, factor):
    """Average a 1D signal in non-overlapping blocks of `factor` -> short array."""
    x = np.asarray(x, dtype=np.float32)
    factor = max(1, int(factor))
    nb = len(x) // factor
    if nb == 0:
        return np.array([x.mean()], dtype=np.float32)
    return x[:nb * factor].reshape(nb, factor).mean(axis=1).astype(np.float32)


def current_envelope(current, factor):
    """
    Slow-varying envelope of a 1D current: block-mean over `factor` samples then
    held (repeated) back to the original length. This is the long-range structure
    the fine model cannot see through its short window on its own.
    """
    current = np.asarray(current, dtype=np.float32)
    n = len(current)
    head = block_mean(current, factor)
    env = np.repeat(head, max(1, int(factor)))
    if len(env) < n:
        env = np.concatenate([env, np.full(n - len(env), head[-1], dtype=np.float32)])
    return env[:n].astype(np.float32)


def dwell_clock(chan):
    """
    Two relaxation coordinates derived from an idealisation:
      since : log(1 + samples elapsed in the current state)
      prev  : log(1 + length of the preceding dwell)

    Neither imposes a relaxation shape. They are axes along which the model may
    learn one, or learn that there isn't one.
    """
    chan = np.asarray(chan, dtype=np.float32)
    n = len(chan)
    edges = np.flatnonzero(np.diff(chan) != 0) + 1
    starts = np.concatenate([[0], edges]).astype(np.int64)
    ends = np.concatenate([edges, [n]]).astype(np.int64)
    since = np.zeros(n, dtype=np.float32)
    prev = np.zeros(n, dtype=np.float32)
    for k in range(len(starts)):
        s, e = starts[k], ends[k]
        since[s:e] = np.arange(e - s, dtype=np.float32)
        prev[s:e] = float(ends[k - 1] - starts[k - 1]) if k else 0.0
    return np.log1p(since), np.log1p(prev)


def burst_clock(chan, tc=1000):
    """
    Log time since the start of the current burst, where shut runs shorter than
    `tc` samples are treated as flickers and do not end the burst. This is the
    coordinate a slow relaxation across a long, flicker-riddled activation lives
    on; `dwell_clock` resets at every flicker and cannot see it.
    """
    chan = (np.asarray(chan, dtype=np.float32) > 0.5).astype(np.float32)
    d = np.flatnonzero(np.diff(chan))
    b = np.concatenate([[0], d + 1, [len(chan)]]).astype(np.int64)
    merged = chan.copy()
    for i in range(len(b) - 1):
        if chan[b[i]] < 0.5 and (b[i + 1] - b[i]) < tc and b[i] > 0:
            merged[b[i]:b[i + 1]] = 1.0
    since, _ = dwell_clock(merged)
    return since


def add_clock_columns(pages, burst_tc=1000):
    """[chan, current] pages -> [chan, current, since, prev, burst_since]."""
    out = []
    for p in pages:
        since, prev = dwell_clock(p[:, 0])
        bsince = burst_clock(p[:, 0], tc=burst_tc)
        out.append(np.concatenate(
            [p.astype(np.float32), since[:, None], prev[:, None],
             bsince[:, None]], axis=1))
    return out


def build_coarse_dataset(pages_by_id, seq_len=256, stride=1):
    """
    pages_by_id : list of (page_id, decimated page
                  [chan, cur, since, prev, burst_since])
    Returns Xc (n, 4, L) conditioning, Yc (n, L) target, pid (n,) page index.
    """
    Xs, Ys, Ps = [], [], []
    for pid, page in pages_by_id:
        if len(page) < seq_len:
            continue
        cond = page[:, [0, 2, 3, 4]].T.astype(np.float32)
        cur = page[:, 1].astype(np.float32)
        for i in range(0, len(page) - seq_len + 1, stride):
            Xs.append(cond[:, i:i + seq_len])
            Ys.append(cur[i:i + seq_len])
            Ps.append(pid)
    if not Xs:
        raise ValueError("No usable coarse windows; lower 'how slow' or seq_len.")
    return (np.asarray(Xs, np.float32), np.asarray(Ys, np.float32),
            np.asarray(Ps, np.int64))


def decimate_pages(pages, factor, phases=1):
    """
    Block-mean each page down by `factor` -> coarse [channels, current] pages.

    `phases` > 1 also emits offset decimation grids (started part-way into a
    block) as extra, correlated views of the same slow signal -- cheap
    augmentation for the data-starved coarse stage. Channels and current use the
    same offset so each (condition, target) pair stays aligned. phases=1 keeps
    the original single grid.
    """
    factor = max(1, int(factor))
    phases = max(1, int(phases))
    offsets = sorted({round(k * factor / phases) for k in range(phases)})
    out = []
    for page in pages:
        for off in offsets:
            sub = page[off:]
            cols = [block_mean(sub[:, j], factor) for j in range(sub.shape[1])]
            if len(cols[0]) >= 1:
                out.append(np.stack(cols, axis=1).astype(np.float32))
    return out


def make_fine_windows(page, factor, seq_len=256, stride=1):
    """
    Windows for the fine model.
      Xcond : (n, 2, seq_len) -> [idealisation, held envelope]
      Y     : (n, seq_len)    -> raw current
    """
    if len(page) < seq_len:
        return (np.empty((0, 2, seq_len), np.float32),
                np.empty((0, seq_len), np.float32))
    chan = page[:, 0].astype(np.float32)
    cur = page[:, 1].astype(np.float32)
    env = current_envelope(cur, factor)
    Xc, Y = [], []
    for i in range(0, len(page) - seq_len + 1, stride):
        Xc.append(np.stack([chan[i:i + seq_len], env[i:i + seq_len]], axis=0))
        Y.append(cur[i:i + seq_len])
    return np.asarray(Xc, np.float32), np.asarray(Y, np.float32)


def build_fine_dataset(pages, factor, seq_len=256, stride=1):
    Xc_all, Y_all = [], []
    for page in pages:
        Xc, Y = make_fine_windows(page, factor, seq_len=seq_len, stride=stride)
        if len(Xc) > 0:
            Xc_all.append(Xc)
            Y_all.append(Y)
    if not Xc_all:
        raise ValueError("No usable fine windows; check seq_len vs page lengths.")
    return np.concatenate(Xc_all, 0), np.concatenate(Y_all, 0)


# ------------------------------------------------------------
# 4. Training / evaluation
# ------------------------------------------------------------

def train_one_epoch(model, loader, optimizer, device):
    model.train()
    total_loss, total_n = 0.0, 0
    for cond, x0 in loader:
        cond, x0 = cond.to(device), x0.to(device)
        optimizer.zero_grad(set_to_none=True)
        loss = model.loss(cond, x0)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * cond.size(0)
        total_n += cond.size(0)
    return total_loss / total_n


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total_loss, total_n = 0.0, 0
    for cond, x0 in loader:
        cond, x0 = cond.to(device), x0.to(device)
        total_loss += model.loss(cond, x0).item() * cond.size(0)
        total_n += cond.size(0)
    return total_loss / total_n


def mmd_rbf(X, Y, gamma=None):
    """Gaussian-kernel maximum mean discrepancy between two sets of windows."""
    X = torch.as_tensor(X, dtype=torch.float32)
    Y = torch.as_tensor(Y, dtype=torch.float32)
    if gamma is None:
        gamma = 1.0 / X.shape[1]

    def k(A, B):
        d = torch.cdist(A, B) ** 2
        return torch.exp(-gamma * d)

    return (k(X, X).mean() + k(Y, Y).mean() - 2 * k(X, Y).mean()).item()


# ------------------------------------------------------------
# 5. Main routine
# ------------------------------------------------------------

def train_model(args):
    set_seed(args.seed)

    device = torch.device(
        "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    )
    print(f"Using device: {device}")

    pages = load_pages(args.folder, verbose=True)
    if not pages:
        print("=" * 64)
        print(f"WARNING: no CSV files found in {args.folder}")
        print("Falling back to SYNTHETIC toy data -- this is NOT your real data.")
        print("=" * 64)
        pages = [generate_synthetic_page(n=5000)]
    else:
        print(f"DATA SOURCE: real CSVs from {args.folder}")

    X, Y = build_dataset(pages, seq_len=args.seq_len, stride=args.stride)
    print(f"Created dataset: cond={X.shape}, current={Y.shape}")

    # Standardise both channels; keep stats to invert at sampling time.
    cond_mean, cond_std = X.mean(), X.std() + 1e-8
    cur_mean, cur_std = Y.mean(), Y.std() + 1e-8
    X = (X - cond_mean) / cond_std
    Y = (Y - cur_mean) / cur_std

    dataset = TensorDataset(
        torch.tensor(X, dtype=torch.float32),
        torch.tensor(Y, dtype=torch.float32),
    )

    val_size = max(1, int(len(dataset) * args.val_fraction))
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(args.seed),
    )

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    model = ConditionalDiffusion(
        base_channels=args.base_channels,
        time_dim=args.time_dim,
        timesteps=args.timesteps,
    ).to(device)

    optimizer = optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    print("Trainable parameters:", sum(p.numel() for p in model.parameters()))

    best_val = float("inf")
    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        val_loss = evaluate(model, val_loader, device)
        print(
            f"Epoch {epoch:03d}/{args.epochs} | "
            f"train eps-MSE={train_loss:.6g} | val eps-MSE={val_loss:.6g}"
        )

        if val_loss < best_val:
            best_val = val_loss
            if args.save_model:
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "seq_len": args.seq_len,
                        "norm": (cond_mean, cond_std, cur_mean, cur_std),
                        "args": vars(args),
                    },
                    args.save_model,
                )

    # Quick distribution sanity check: real vs generated windows.
    cond_b, real_b = next(iter(val_loader))
    cond_b = cond_b[:64].to(device)
    real_b = real_b[:64]
    gen_b = model.sample(cond_b).cpu()
    print(f"MMD(real, generated) on val sample: {mmd_rbf(real_b, gen_b):.6g}")

    if args.save_model:
        print(f"Best model saved to: {args.save_model}")

    return model


def parse_args():
    p = argparse.ArgumentParser(
        description="Conditional diffusion generator for ion-channel raw current."
    )
    p.add_argument("--folder", type=str, default="./pages")
    p.add_argument("--seq-len", type=int, default=256)
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--val-fraction", type=float, default=0.1)
    p.add_argument("--base-channels", type=int, default=64)
    p.add_argument("--time-dim", type=int, default=128)
    p.add_argument("--timesteps", type=int, default=1000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--save-model", type=str, default="ion_channel_diffusion_model.pt")

    args = p.parse_args()
    if args.seq_len % 4 != 0:
        raise ValueError("seq-len must be divisible by 4 for the U-Net.")
    if args.save_model == "":
        args.save_model = None
    return args


if __name__ == "__main__":
    train_model(parse_args())
