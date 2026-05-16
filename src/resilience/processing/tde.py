"""
tde.py — Time-Delay Embedding for state-space reconstruction.

Takens' theorem (1981): reconstruct the dynamics of a system from a single
observed variable.

    1. AMI (Fraser & Swinney 1986)      → first local minimum = optimal τ
    2. FNN (Kennel et al. 1992)         → 1% threshold      = optimal m
    3. Phase space reconstruction       → matrix [X(t), X(t+τ), ..., X(t+(m-1)τ)]
"""

from __future__ import annotations
from typing import Optional, Tuple

import numpy as np
from scipy.signal import find_peaks
from sklearn.neighbors import NearestNeighbors

from resilience import config


# ════════════════════════════════════════════════════════════════════════════
# AMI
# ════════════════════════════════════════════════════════════════════════════

def compute_ami(signal_1d: np.ndarray,
                  max_lag: Optional[int] = None,
                  num_bins: int = 20) -> Tuple[int, np.ndarray]:
    """AMI for lags 1..max_lag; return τ at the first local minimum."""
    if max_lag is None:
        max_lag = config.TDE_MAX_LAG

    signal_1d = np.asarray(signal_1d, dtype=float)
    ami_values = np.zeros(max_lag)
    bins = np.linspace(signal_1d.min(), signal_1d.max(), num_bins + 1)

    for lag in range(1, max_lag + 1):
        x = signal_1d[:-lag]
        y = signal_1d[lag:]
        joint_hist, _, _ = np.histogram2d(x, y, bins=[bins, bins])
        joint_prob = joint_hist / joint_hist.sum()
        px = joint_prob.sum(axis=1)
        py = joint_prob.sum(axis=0)

        mi = 0.0
        for i in range(num_bins):
            for j in range(num_bins):
                if joint_prob[i, j] > 0:
                    mi += joint_prob[i, j] * np.log(
                        joint_prob[i, j] / (px[i] * py[j] + 1e-10)
                    )
        ami_values[lag - 1] = mi

    peaks, _ = find_peaks(-ami_values, prominence=0.01)
    if len(peaks) > 0:
        tau_optimal = int(peaks[0]) + 1
    else:
        threshold = ami_values[0] / np.e
        below = np.where(ami_values < threshold)[0]
        tau_optimal = int(below[0]) + 1 if len(below) > 0 else 10

    return tau_optimal, ami_values


# ════════════════════════════════════════════════════════════════════════════
# FNN
# ════════════════════════════════════════════════════════════════════════════

def _compute_fnn_for_dimension(signal_1d: np.ndarray, dim: int, tau: int,
                                 r_threshold: float = 15.0) -> int:
    n = len(signal_1d)
    max_idx = n - (dim - 1) * tau
    if max_idx <= 1:
        return 0

    embedded = np.zeros((max_idx, dim))
    for i in range(dim):
        embedded[:, i] = signal_1d[i * tau : i * tau + max_idx]

    valid_rows = ~np.isnan(embedded).any(axis=1)
    embedded_clean = embedded[valid_rows]
    base_indices = np.where(valid_rows)[0]
    if len(embedded_clean) < 2:
        return 0

    nbrs = NearestNeighbors(n_neighbors=2, algorithm="kd_tree").fit(embedded_clean)
    distances, indices = nbrs.kneighbors(embedded_clean)

    fnn_count = 0
    for i in range(len(embedded_clean)):
        nearest_dist = distances[i, 1]
        if nearest_dist < 1e-10:
            continue
        neighbor_idx = indices[i, 1]
        i_sig = base_indices[i] + dim * tau
        j_sig = base_indices[neighbor_idx] + dim * tau
        if i_sig >= n or j_sig >= n:
            continue
        val_i = signal_1d[i_sig]
        val_j = signal_1d[j_sig]
        if np.isnan(val_i) or np.isnan(val_j):
            continue
        if abs(val_i - val_j) / nearest_dist > r_threshold:
            fnn_count += 1

    return fnn_count


def compute_fnn(signal_1d: np.ndarray,
                  max_dim: Optional[int] = None,
                  tau: int = 10,
                  threshold: Optional[float] = None,
                  r_threshold: Optional[float] = None) -> Tuple[int, np.ndarray]:
    """% of false nearest neighbours for dim 1..max_dim; return optimal m."""
    if max_dim is None:     max_dim     = config.TDE_MAX_DIM
    if threshold is None:   threshold   = config.TDE_FNN_THRESHOLD
    if r_threshold is None: r_threshold = config.TDE_FNN_R

    signal_1d = np.asarray(signal_1d, dtype=float)
    fnn_rates = np.zeros(max_dim)

    for dim in range(1, max_dim + 1):
        fnn_count = _compute_fnn_for_dimension(signal_1d, dim, tau, r_threshold)
        n_points = max(len(signal_1d) - dim * tau, 1)
        fnn_rates[dim - 1] = 100.0 * fnn_count / n_points

    below = np.where(fnn_rates < threshold)[0]
    if len(below) > 0:
        dim_optimal = int(below[0]) + 1
    else:
        dim_optimal = int(np.argmin(fnn_rates)) + 1
        print(f"   ⚠️  No dim below {threshold}% — fallback argmin → m={dim_optimal}")

    return dim_optimal, fnn_rates


# ════════════════════════════════════════════════════════════════════════════
# PHASE SPACE
# ════════════════════════════════════════════════════════════════════════════

def phase_space_reconstruction(signal_1d: np.ndarray, tau: int, dim: int) -> np.ndarray:
    """Takens embedding matrix."""
    signal_1d = np.asarray(signal_1d, dtype=float)
    n_points = len(signal_1d) - (dim - 1) * tau
    if n_points <= 0:
        raise ValueError(f"Signal too short for dim={dim}, tau={tau}")

    phase_space = np.zeros((n_points, dim))
    for d in range(dim):
        phase_space[:, d] = signal_1d[d * tau : d * tau + n_points]
    return phase_space


# ════════════════════════════════════════════════════════════════════════════
# ORCHESTRATOR
# ════════════════════════════════════════════════════════════════════════════

def auto_parameters(signal_1d: np.ndarray,
                      max_lag: Optional[int] = None,
                      max_dim: Optional[int] = None,
                      threshold: Optional[float] = None,
                      r_threshold: Optional[float] = None,
                      verbose: bool = True) -> dict:
    """AMI → τ, then FNN(τ) → m, then 3D phase space for visualisation."""
    if verbose: print("   → compute_ami() …")
    tau, ami = compute_ami(signal_1d, max_lag=max_lag)
    if verbose: print(f"     optimal τ = {tau}")

    if verbose: print("   → compute_fnn() …")
    m, fnn = compute_fnn(signal_1d, max_dim=max_dim, tau=tau,
                          threshold=threshold, r_threshold=r_threshold)
    if verbose: print(f"     optimal m = {m}")

    ps_3d = phase_space_reconstruction(signal_1d, tau=tau, dim=config.TDE_VIZ_DIM)

    return {"tau": tau, "m": m, "ami": ami, "fnn": fnn,
            "signal": signal_1d, "phase_space_3d": ps_3d}
