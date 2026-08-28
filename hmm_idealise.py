#!/usr/bin/env python3
"""
SKM-style HMM idealiser  (the QuB idea, stripped to its core).

Hidden states = open-channel levels. Emissions = Gaussian on the current
amplitude per level. Transitions = a fitted matrix whose diagonal carries the
dwell-time prior -- the kinetic knowledge the CNN never had.

Fitting is segmental k-means (hard EM, like QuB's SKM): Viterbi-decode, then
re-estimate means / variances / transitions from the hard path, repeat. A final
log-space forward-backward pass gives a real posterior, used as the confidence.

The point of the dwell prior: a one-channel noise dip and a one-channel closing
look identical in amplitude. Only the cost of switching state tells them apart.
Crank `self_bias` up and the idealiser refuses to switch on brief wobbles.

Public contract (matches inverse_model.predict_trace, so it drops into the app):
    idealise_trace(current, level_values=None, ...) -> (levels[N], confidence[N])
"""

import argparse

import numpy as np
import pandas as pd
from math import comb

LOG2PI = np.log(2.0 * np.pi)


def _logsumexp(a, axis=None):
    m = np.max(a, axis=axis, keepdims=True)
    out = m + np.log(np.sum(np.exp(a - m), axis=axis, keepdims=True))
    return np.squeeze(out, axis=axis) if axis is not None else out.reshape(())


def _emission_loglik(x, mu, sigma):
    """(K, N) log Gaussian likelihood of each sample under each state."""
    z = (x[None, :] - mu[:, None]) / sigma[:, None]
    return -0.5 * z * z - np.log(sigma[:, None]) - 0.5 * LOG2PI


def _viterbi(logB, log_pi, log_A):
    K, N = logB.shape
    delta = np.full((K, N), -np.inf)
    psi = np.zeros((K, N), dtype=np.int64)
    delta[:, 0] = log_pi + logB[:, 0]
    for t in range(1, N):
        scores = delta[:, t - 1][:, None] + log_A          # (K_from, K_to)
        psi[:, t] = np.argmax(scores, axis=0)
        delta[:, t] = scores[psi[:, t], np.arange(K)] + logB[:, t]
    path = np.zeros(N, dtype=np.int64)
    path[-1] = int(np.argmax(delta[:, -1]))
    for t in range(N - 2, -1, -1):
        path[t] = psi[path[t + 1], t + 1]
    return path


def _forward_backward(logB, log_pi, log_A):
    K, N = logB.shape
    log_alpha = np.full((K, N), -np.inf)
    log_alpha[:, 0] = log_pi + logB[:, 0]
    for t in range(1, N):
        log_alpha[:, t] = logB[:, t] + _logsumexp(log_alpha[:, t - 1][:, None] + log_A, axis=0)
    log_beta = np.zeros((K, N))
    for t in range(N - 2, -1, -1):
        log_beta[:, t] = _logsumexp(log_A + (logB[:, t + 1] + log_beta[:, t + 1])[None, :], axis=1)
    log_gamma = log_alpha + log_beta
    log_gamma -= _logsumexp(log_gamma, axis=0)[None, :]
    return np.exp(log_gamma)


def _kmeans_init(x, k, iters=10):
    centres = np.quantile(x, np.linspace(0.05, 0.95, k))
    for _ in range(iters):
        assign = np.argmin(np.abs(x[:, None] - centres[None, :]), axis=1)
        for j in range(k):
            if np.any(assign == j):
                centres[j] = x[assign == j].mean()
    return np.sort(centres)


def _masked_log_A(prior, level_values, jump_penalty):
    """
    Normalised log transition matrix for INDEPENDENT channels.

    A jump of d levels needs d separate channels to switch within one sample, so
    its probability is scaled by jump_penalty**d (the product of d single-channel
    switch probabilities). d=0 (stay) and d=1 (one channel) are favoured; bigger
    jumps are allowed but vanishingly unlikely -- not forbidden.
    """
    vals = np.asarray(level_values, dtype=float)
    dist = np.abs(np.subtract.outer(vals, vals))
    p = prior * (jump_penalty ** dist)
    rowsum = p.sum(axis=1, keepdims=True)
    A = np.divide(p, rowsum, out=np.zeros_like(p), where=rowsum > 0)
    with np.errstate(divide="ignore"):
        log_A = np.log(A)
    return log_A


def _fmt_transition(log_A, level_values):
    """Pretty-print the transition matrix (rows = from level, cols = to level)."""
    A = np.exp(log_A)

    def cell(v):
        return f"{v:9.5f}" if v >= 5e-4 else f"{v:9.1e}"

    head = "    from\\to " + " ".join(f"{int(v):>9d}" for v in level_values)
    rows = [head]
    for i, v in enumerate(level_values):
        rows.append(f"    {int(v):>6d}   " + " ".join(cell(A[i, j]) for j in range(len(level_values))))
    return "\n".join(rows)


def idealise_trace(current, level_values=None, n_levels=None, human=None,
                   self_bias=50.0, max_iter=40, sigma_floor=1e-3, rel_tol=1e-4,
                   jump_penalty=0.01, verbose=False):
    """
    Returns
    -------
    levels : int array, the idealised open-channel count per sample
    conf   : float array, posterior probability of the chosen state per sample
    """
    x = np.asarray(current, dtype=np.float64)
    N = len(x)

    # Decide the state set and initialise emission means.
    if human is not None:
        level_values = sorted({int(v) for v in np.rint(human)})
        mu = np.array([x[np.rint(human).astype(int) == v].mean() if np.any(np.rint(human).astype(int) == v)
                       else x.mean() for v in level_values], dtype=np.float64)
    elif level_values is not None:
        level_values = sorted(int(v) for v in level_values)
        mu = _kmeans_init(x, len(level_values))
    else:
        if n_levels is None:
            raise ValueError("Give one of: human labels, level_values, or n_levels.")
        level_values = list(range(n_levels))
        mu = _kmeans_init(x, n_levels)

    K = len(level_values)
    sigma = np.full(K, max(x.std(), sigma_floor))
    log_pi = np.full(K, -np.log(K))

    # Transition prior: sticky diagonal (the dwell-time / kinetic prior).
    prior = np.ones((K, K)) + self_bias * np.eye(K)

    if verbose:
        import time
        t0 = time.time()
        print(f"Fitting HMM: {N} samples, {K} states. Each pass is one sweep over the trace...")

    prev_path = None
    for it in range(max_iter):
        logB = _emission_loglik(x, mu, sigma)
        log_A = _masked_log_A(prior, level_values, jump_penalty)
        path = _viterbi(logB, log_pi, log_A)

        # Re-estimate from the hard path (the "k-means" half of SKM).
        for k in range(K):
            sel = path == k
            if np.any(sel):
                mu[k] = x[sel].mean()
                sigma[k] = max(x[sel].std(), sigma_floor)
        counts = np.full((K, K), 0.0) + self_bias * np.eye(K)
        idx = np.stack([path[:-1], path[1:]])
        np.add.at(counts, (idx[0], idx[1]), 1.0)
        prior = counts + 1e-3

        changed = N if prev_path is None else int(np.sum(path != prev_path))
        if verbose:
            tag = "(first pass, nothing to compare yet)" if prev_path is None \
                  else f"changed={changed:>8d} ({100 * changed / N:5.2f}%)"
            print(f"  iter {it + 1:02d}/{max_iter}  {tag}  elapsed={time.time() - t0:5.1f}s")
        if prev_path is not None and changed <= rel_tol * N:
            break
        prev_path = path

    # Final soft posterior for an honest confidence.
    logB = _emission_loglik(x, mu, sigma)
    log_A = _masked_log_A(prior, level_values, jump_penalty)
    path = _viterbi(logB, log_pi, log_A)
    gamma = _forward_backward(logB, log_pi, log_A)
    conf = gamma[path, np.arange(N)].astype(np.float32)

    levels = np.asarray(level_values, dtype=np.int64)[path]
    return levels, conf


def idealise_with_emissions(emission_logprob, level_values, self_bias=50.0,
                            log_prior=None, max_iter=40, rel_tol=1e-4,
                            jump_penalty=0.01, verbose=False, show_matrix=False):
    """
    Hybrid decode: emissions come from the CNN (fixed), we only fit the dwell-
    prior transition matrix and decode. `emission_logprob` is (K, N) log P(level|x).

    Subtracting `log_prior` turns posteriors into scaled likelihoods, which also
    de-biases the common levels (helps the rare-level imbalance). `jump_penalty`
    scales multi-channel jumps for independent channels: a d-level jump costs
    jump_penalty**d -- rare, but never forbidden.
    """
    logB = np.asarray(emission_logprob, dtype=np.float64)
    if log_prior is not None:
        logB = logB - np.asarray(log_prior, dtype=np.float64)[:, None]
    K, N = logB.shape
    log_pi = np.full(K, -np.log(K))
    prior = np.ones((K, K)) + self_bias * np.eye(K)

    if verbose:
        import time
        t0 = time.time()
        print(f"Hybrid decode: {N} samples, {K} states (CNN emissions, HMM transitions)...")

    prev_path = None
    for it in range(max_iter):
        log_A = _masked_log_A(prior, level_values, jump_penalty)
        path = _viterbi(logB, log_pi, log_A)
        counts = np.full((K, K), 0.0) + self_bias * np.eye(K)
        idx = np.stack([path[:-1], path[1:]])
        np.add.at(counts, (idx[0], idx[1]), 1.0)
        prior = counts + 1e-3
        changed = N if prev_path is None else int(np.sum(path != prev_path))
        if verbose:
            tag = "(first pass, nothing to compare yet)" if prev_path is None \
                  else f"changed={changed:>8d} ({100 * changed / N:5.2f}%)"
            print(f"  iter {it + 1:02d}/{max_iter}  {tag}  elapsed={time.time() - t0:5.1f}s")
        if show_matrix:
            print(_fmt_transition(log_A, level_values))
        if prev_path is not None and changed <= rel_tol * N:
            break
        prev_path = path

    log_A = _masked_log_A(prior, level_values, jump_penalty)
    if verbose or show_matrix:
        print("Final fitted transition matrix:")
        print(_fmt_transition(log_A, level_values))
    path = _viterbi(logB, log_pi, log_A)
    gamma = _forward_backward(logB, log_pi, log_A)
    conf = gamma[path, np.arange(N)].astype(np.float32)
    levels = np.asarray(level_values, dtype=np.int64)[path]
    return levels, conf


# ------------------------------------------------------------
# Independent-channels model: N two-state (C<->O) channels.
# The aggregate transition matrix is DERIVED from per-sample open/close
# probabilities (a, b), so multi-channel jumps are ~a^d automatically -- no
# jump penalty, no self-bias. The observable level = number of open channels.
# ------------------------------------------------------------

from math import comb


def _binom_pmf(n, p):
    p = min(max(float(p), 1e-12), 1.0 - 1e-12)
    k = np.arange(n + 1)
    return np.array([comb(n, int(j)) * p ** int(j) * (1 - p) ** (n - int(j)) for j in k])


def aggregate_A(n_channels, a, b):
    """
    Transition matrix over open-counts 0..n_channels for N independent
    CLOSED<->OPEN channels. From i open: the i open each stay open w.p. (1-b),
    the (N-i) closed each open w.p. a; next count is the sum (a convolution).
    A[0, N] = a**N falls straight out.
    """
    K = n_channels + 1
    A = np.zeros((K, K))
    for i in range(K):
        stay = _binom_pmf(i, 1.0 - b)          # open channels that remain open
        opn = _binom_pmf(n_channels - i, a)    # closed channels that open
        dist = np.convolve(stay, opn)          # length K
        A[i, :len(dist)] = dist
    return A


def idealise_channels(emission_logprob, n_channels=None, a=0.01, b=0.01, fit=True,
                      log_prior=None, max_iter=40, rel_tol=1e-4, verbose=False,
                      show_matrix=False):
    """
    Decode using the independent-channels transition model. Emission rows must be
    the open-counts 0..n_channels (row k = level k). Returns (levels, conf, (a, b)).
    """
    logB = np.asarray(emission_logprob, dtype=np.float64)
    if log_prior is not None:
        logB = logB - np.asarray(log_prior, dtype=np.float64)[:, None]
    K, N = logB.shape
    if n_channels is None:
        n_channels = K - 1
    if n_channels + 1 != K:
        raise ValueError(f"emissions have {K} levels but n_channels+1={n_channels + 1}; "
                         "the channel model needs contiguous counts 0..N.")
    levels_idx = np.arange(K)
    log_pi = np.full(K, -np.log(K))

    if verbose:
        import time
        t0 = time.time()
        print(f"Channel decode: {N} samples, {n_channels} independent channels...")

    prev_path = None
    for it in range(max_iter):
        A = aggregate_A(n_channels, a, b)
        log_A = np.log(A + 1e-300)
        path = _viterbi(logB, log_pi, log_A)
        if fit:
            d = np.diff(path)
            openings = float(d[d > 0].sum())
            closings = float(-d[d < 0].sum())
            open_opp = float((n_channels - path[:-1]).sum())
            close_opp = float(path[:-1].sum())
            a = min(max(openings / max(open_opp, 1.0), 1e-6), 0.5)
            b = min(max(closings / max(close_opp, 1.0), 1e-6), 0.5)
        changed = N if prev_path is None else int(np.sum(path != prev_path))
        if verbose:
            tag = "(first pass)" if prev_path is None else f"changed={changed} ({100*changed/N:.2f}%)"
            print(f"  iter {it+1:02d}/{max_iter}  a={a:.2e} b={b:.2e}  {tag}  "
                  f"elapsed={time.time()-t0:5.1f}s")
        if prev_path is not None and changed <= rel_tol * N:
            break
        prev_path = path

    A = aggregate_A(n_channels, a, b)
    log_A = np.log(A + 1e-300)
    if verbose or show_matrix:
        print(f"Fitted per-sample probs: open a={a:.3e}, close b={b:.3e}")
        print(_fmt_transition(log_A, levels_idx))
    path = _viterbi(logB, log_pi, log_A)
    gamma = _forward_backward(logB, log_pi, log_A)
    conf = gamma[path, np.arange(N)].astype(np.float32)
    return path.astype(np.int64), conf, (a, b), gamma


def _read_csv(path):
    df = pd.read_csv(path)
    norm = {str(c).strip().lower(): c for c in df.columns}
    chan = norm.get("channels")
    cur = norm.get("noisy current") or norm.get("current")
    if chan is None or cur is None:        # positional fallback: ..., channels, current
        chan, cur = df.columns[-2], df.columns[-1]
    h = np.rint(pd.to_numeric(df[chan], errors="coerce")).to_numpy()
    c = pd.to_numeric(df[cur], errors="coerce").to_numpy(dtype=np.float64)
    return h, c


def main():
    p = argparse.ArgumentParser(description="SKM-style HMM idealiser.")
    p.add_argument("csv", help="labelled CSV with Channels + Noisy Current columns")
    p.add_argument("--self-bias", type=float, default=50.0,
                   help="dwell-prior strength; higher = fewer, stickier transitions")
    p.add_argument("--out", type=str, default=None, help="optional CSV to write idealisation")
    args = p.parse_args()

    human, current = _read_csv(args.csv)
    levels, conf = idealise_trace(current, human=human, self_bias=args.self_bias, verbose=True)
    agree = 100.0 * np.mean(levels == np.rint(human))
    print(f"Samples: {len(current)} | states: {sorted(set(levels.tolist()))} "
          f"| agreement with human: {agree:.2f}% | mean confidence: {conf.mean():.3f}")
    if args.out:
        pd.DataFrame({"Channels_hmm": levels, "confidence": conf}).to_csv(args.out, index=False)
        print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
