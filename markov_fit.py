"""
Fit a discrete-time Markov model to an observed idealisation (open-channel
count, not state). Several states can map to the same observed count, so
this is an aggregated HMM with a fixed, deterministic emission matrix,
fitted by Baum-Welch (forward-backward + row-normalised transition counts).

Likelihood is over the full sequence, not dwell-time histograms, since
histograms discard dwell ordering.

For n_channels > 1 the composite chain is the tensor product of n_channels
copies of P; per-channel counts are recovered by marginalising the composite
expected counts, with parameters tied across channels.
"""

import numpy as np

try:
    from numba import njit
    _HAVE_NUMBA = True
except Exception:
    _HAVE_NUMBA = False


MAX_COMPOSITE = 4096


# ------------------------------------------------------------
# Forward-backward (scaled)
# ------------------------------------------------------------

def _forward_backward_py(Pc, pi, obs_rows):
    """
    Pc       : (A, A) composite transition matrix
    pi       : (A,)   composite initial distribution
    obs_rows : (T, A) 0/1 emission compatibility, obs_rows[t, a] = 1 if
               composite state a produces the observed count at time t

    Returns alpha (T, A), beta (T, A), scale (T,), loglik.
    """
    T, A = obs_rows.shape
    alpha = np.zeros((T, A))
    beta = np.zeros((T, A))
    scale = np.zeros(T)

    a = pi * obs_rows[0]
    s = a.sum()
    if s <= 0.0:
        return alpha, beta, scale, -np.inf
    alpha[0] = a / s
    scale[0] = s
    for t in range(1, T):
        a = (alpha[t - 1] @ Pc) * obs_rows[t]
        s = a.sum()
        if s <= 0.0:
            return alpha, beta, scale, -np.inf
        alpha[t] = a / s
        scale[t] = s

    beta[T - 1] = 1.0
    for t in range(T - 2, -1, -1):
        beta[t] = (Pc @ (obs_rows[t + 1] * beta[t + 1])) / scale[t + 1]

    return alpha, beta, scale, float(np.log(scale).sum())


if _HAVE_NUMBA:
    @njit(cache=True)
    def _forward_backward_nb(Pc, pi, obs_rows):
        T, A = obs_rows.shape
        alpha = np.zeros((T, A))
        beta = np.zeros((T, A))
        scale = np.zeros(T)

        s = 0.0
        for i in range(A):
            alpha[0, i] = pi[i] * obs_rows[0, i]
            s += alpha[0, i]
        if s <= 0.0:
            return alpha, beta, scale, -np.inf
        for i in range(A):
            alpha[0, i] /= s
        scale[0] = s

        for t in range(1, T):
            s = 0.0
            for j in range(A):
                acc = 0.0
                for i in range(A):
                    acc += alpha[t - 1, i] * Pc[i, j]
                alpha[t, j] = acc * obs_rows[t, j]
                s += alpha[t, j]
            if s <= 0.0:
                return alpha, beta, scale, -np.inf
            for j in range(A):
                alpha[t, j] /= s
            scale[t] = s

        for i in range(A):
            beta[T - 1, i] = 1.0
        for t in range(T - 2, -1, -1):
            inv = 1.0 / scale[t + 1]
            for i in range(A):
                acc = 0.0
                for j in range(A):
                    acc += Pc[i, j] * obs_rows[t + 1, j] * beta[t + 1, j]
                beta[t, i] = acc * inv

        ll = 0.0
        for t in range(T):
            ll += np.log(scale[t])
        return alpha, beta, scale, ll

    _forward_backward = _forward_backward_nb
else:
    _forward_backward = _forward_backward_py


# ------------------------------------------------------------
# Composite state space
# ------------------------------------------------------------

def _composite_space(n_states, n_channels, open_mask):
    """
    Enumerate the n_states ** n_channels composite states as base-S digits.

    Returns digits (A, n_channels) and the open-channel count of each.
    """
    A = int(n_states) ** int(n_channels)
    if A > MAX_COMPOSITE:
        raise ValueError(
            f"{n_states} states x {n_channels} channels = {A} composite states, "
            f"above the {MAX_COMPOSITE} limit. Reduce states or channels."
        )
    idx = np.arange(A)
    digits = np.empty((A, int(n_channels)), dtype=np.int64)
    for c in range(int(n_channels)):
        digits[:, c] = (idx // (int(n_states) ** c)) % int(n_states)
    counts = open_mask[digits].sum(axis=1).astype(np.int64)
    return digits, counts


def _kron_pow(P, k):
    Pc = np.array([[1.0]])
    for _ in range(k):
        Pc = np.kron(Pc, P)
    return Pc


def _stationary(P):
    """Stationary distribution by eigen-decomposition, with a uniform fallback."""
    S = P.shape[0]
    try:
        w, v = np.linalg.eig(P.T)
        k = int(np.argmin(np.abs(w - 1.0)))
        pi = np.real(v[:, k])
        pi = np.clip(pi, 0.0, None)
        if pi.sum() > 0:
            return pi / pi.sum()
    except np.linalg.LinAlgError:
        pass
    return np.full(S, 1.0 / S)


def _random_P(rng, mask):
    S = mask.shape[0]
    P = rng.random((S, S)) * mask
    P += np.eye(S) * (rng.random(S) * 4.0 + 4.0) * np.diag(mask)
    rows = P.sum(axis=1, keepdims=True)
    rows[rows == 0] = 1.0
    return P / rows


# ------------------------------------------------------------
# Main fitter
# ------------------------------------------------------------

def fit_markov_to_idealisation(
    chan,
    n_states,
    open_states,
    n_channels=None,
    allowed=None,
    n_restarts=4,
    max_iter=150,
    tol=1e-5,
    prior=0.5,
    max_samples=200_000,
    seed=None,
    P_init=None,
    progress=None,
):
    """
    chan         : 1D array of observed open-channel counts (the idealisation).
    n_states     : S, the number of states in the model.
    open_states  : iterable of 0-based state indices counted as OPEN.
    n_channels   : channels in the record. None infers chan.max(), which is a
                   LOWER BOUND -- if every channel is never open at once you
                   will underestimate it.
    allowed      : (S, S) 0/1 mask of permitted transitions. Diagonal is always
                   permitted. None means a full dense matrix.
    prior        : Dirichlet pseudo-count added to every permitted transition.
                   0 gives the maximum-likelihood fit; >0 gives the MAP fit and
                   stops rarely-visited transitions collapsing to zero.
    max_samples  : cap on trace length used for fitting (runtime is linear in it).
    P_init       : optional (S, S) starting point, used as the first restart.

    Returns a dict with P, loglik, n_channels, n_iter, converged, restart_logliks.
    """
    chan = np.asarray(chan).astype(np.int64).ravel()
    if chan.size < 2:
        raise ValueError("Need at least two samples to fit.")
    if len(chan) > max_samples:
        chan = chan[:max_samples]

    S = int(n_states)
    open_mask = np.zeros(S, dtype=bool)
    open_mask[list(open_states)] = True
    if not open_mask.any() or open_mask.all():
        raise ValueError("Need at least one open state and one closed state.")

    N = int(chan.max()) if n_channels is None else int(n_channels)
    N = max(1, N)
    if chan.max() > N:
        raise ValueError(f"Trace shows {chan.max()} open channels but n_channels={N}.")

    if allowed is None:
        mask = np.ones((S, S), dtype=float)
    else:
        mask = np.asarray(allowed, dtype=float)
        if mask.shape != (S, S):
            raise ValueError(f"allowed must be ({S}, {S}), got {mask.shape}.")
        mask = (mask != 0).astype(float)
    np.fill_diagonal(mask, 1.0)
    if (mask.sum(axis=1) == 0).any():
        raise ValueError("Every state needs at least one permitted transition.")

    digits, counts = _composite_space(S, N, open_mask)
    A = digits.shape[0]

    # Deterministic emissions: obs_rows[t, a] = 1 iff composite state a shows
    # the observed number of open channels at time t.
    lookup = np.zeros((N + 1, A))
    for k in range(N + 1):
        lookup[k] = (counts == k)
    obs_rows = np.ascontiguousarray(lookup[chan])

    # One-hot maps used to marginalise composite counts back to per-channel ones.
    E = np.zeros((N, A, S))
    for c in range(N):
        E[c, np.arange(A), digits[:, c]] = 1.0

    rng = np.random.default_rng(seed)
    best = {"loglik": -np.inf, "P": None, "n_iter": 0, "converged": False}
    restart_lls = []

    for r in range(int(n_restarts)):
        P = np.asarray(P_init, dtype=float).copy() if (r == 0 and P_init is not None) \
            else _random_P(rng, mask)
        P = P * mask
        rows = P.sum(axis=1, keepdims=True)
        rows[rows == 0] = 1.0
        P /= rows

        prev_ll, converged, it = -np.inf, False, 0
        for it in range(1, int(max_iter) + 1):
            Pc = _kron_pow(P, N)
            pi = _kron_pow(_stationary(P).reshape(1, -1), N).ravel()

            alpha, beta, scale, ll = _forward_backward(
                np.ascontiguousarray(Pc), np.ascontiguousarray(pi), obs_rows
            )
            if not np.isfinite(ll):
                break

            # Xi[a, b] = expected number of composite a -> b transitions.
            M = obs_rows[1:] * beta[1:] / scale[1:, None]
            Xi = Pc * (alpha[:-1].T @ M)

            # Tie across channels: sum the marginalised counts.
            C = np.zeros((S, S))
            for c in range(N):
                C += E[c].T @ Xi @ E[c]

            C = (C + prior) * mask
            rows = C.sum(axis=1, keepdims=True)
            rows[rows == 0] = 1.0
            P = C / rows

            if progress is not None:
                progress((r + it / max_iter) / max(1, int(n_restarts)))
            if ll - prev_ll < tol * max(1.0, abs(prev_ll)):
                prev_ll, converged = ll, True
                break
            prev_ll = ll

        restart_lls.append(prev_ll)
        if prev_ll > best["loglik"]:
            best = {"loglik": prev_ll, "P": P.copy(), "n_iter": it, "converged": converged}

    if best["P"] is None:
        raise RuntimeError("All restarts failed. Check the state count and open states.")

    n_free = int(mask.sum() - S)
    best["n_channels"] = N
    best["n_states"] = S
    best["restart_logliks"] = restart_lls
    best["n_samples"] = int(len(chan))
    best["bic"] = -2.0 * best["loglik"] + n_free * np.log(len(chan))
    return best


def dwell_summary(chan):
    """Mean dwell per observed level -- a quick check that a fit looks sane."""
    chan = np.asarray(chan).ravel()
    edges = np.flatnonzero(np.diff(chan) != 0) + 1
    starts = np.concatenate([[0], edges])
    ends = np.concatenate([edges, [len(chan)]])
    lens = ends - starts
    levels = chan[starts]
    return {int(k): float(lens[levels == k].mean()) for k in np.unique(levels)}
