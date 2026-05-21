"""
resilience_dynamics.py — Locomotor resilience dynamics (Antoine Dufourneau method, adapted).

Ported from `resilience-analysis-matlab/real_main.m`.

Method
------
1. Pre-perturbation reference: phase-space trajectory (Tau, m from notebook 03)
   smoothed by Fourier expansion (order 8) on 360° phase angles.
2. Distance signal `XR_distances`: Euclidean distance between actual phase-space
   trajectory and the reference at the matching phase angle.
3. Stability thresholds T1/T2/T3 from local ellipsoid standard deviations,
   scaled by the 97.5% quantile of the pre-perturbation distance distribution.
4. Resilience metrics per trial:
       lag_time      : pert_entry → first sample in T3/>T3 with at least 3 of
                       the next N_CONSECUTIVE samples also in T3/>T3 (compromise
                       between strict 10-consecutive criterion and single-sample
                       false positives)
       peak_time     : lag → peak deviation (XR_distances local max), searched
                       on [lag, pert_exit + POST_PEAK_BUFFER_S]; peak must be ≥ lag
       recovery_time : pert_exit → stable return (post-perturbation),
                       constrained to recovery_idx > peak_idx AND > pert_exit_idx
                       (10 consecutive heel-strikes in T1/T2, tolerance 5 NaN)
       persistent    : mean post-recovery distance vs pre-perturbation
                       (4 windows: 15, 30, 45, 60 s)

Adaptations from Antoine's original
-----------------------------------
- 1 perturbation per trial (vs 2 in Antoine's protocol)
- Perturbation has duration (~4s sand crossing), not a single instant
  → peak searched over [lag, pert_exit + buffer]
  → lag measured from pert_entry
  → recovery measured from pert_exit (post-perturbation, conventional)
  → strict chronology enforced: pert_entry ≤ lag ≤ peak ≤ recovery
- No resynchronisation analysis (no analog beep signal)
"""

from __future__ import annotations
from typing import Optional

import numpy as np
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d

from resilience import config


# ════════════════════════════════════════════════════════════════════════════
# PHASE-SPACE EMBEDDING
# ════════════════════════════════════════════════════════════════════════════

def phase_space_reconstruction(data: np.ndarray, tau: int, dim: int) -> np.ndarray:
    """Embed a 1-D time series into a (N-(dim-1)*tau) x dim matrix."""
    data = np.asarray(data, dtype=float).ravel()
    max_idx = len(data) - (dim - 1) * tau
    return np.column_stack([data[i * tau: i * tau + max_idx] for i in range(dim)])


# ════════════════════════════════════════════════════════════════════════════
# REFERENCE TRAJECTORY (Fourier smoothing on phase angles)
# ════════════════════════════════════════════════════════════════════════════

def _fit_plane(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
    """Fit z = p00 + p10*x + p01*y. Returns coeffs [p00, p10, p01]."""
    A = np.column_stack([np.ones_like(x), x, y])
    coeffs, *_ = np.linalg.lstsq(A, z, rcond=None)
    return coeffs


def _vrrotvec2mat(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Rotation matrix that rotates unit vector a onto unit vector b."""
    a = a / (np.linalg.norm(a) + 1e-12)
    b = b / (np.linalg.norm(b) + 1e-12)
    c = np.cross(a, b)
    s = np.linalg.norm(c)
    if s < 1e-12:
        return np.eye(3)
    angle = np.arccos(np.clip(np.dot(a, b), -1, 1))
    axis = c / s
    K = np.array([[   0,       -axis[2],  axis[1]],
                  [ axis[2],    0,       -axis[0]],
                  [-axis[1],  axis[0],    0      ]])
    return np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)


def _fourier_design_matrix(phases_deg: np.ndarray, order: int) -> np.ndarray:
    """Build [N x (1 + 2*order)] Fourier design matrix."""
    phi = np.deg2rad(phases_deg).reshape(-1, 1)
    k   = np.arange(1, order + 1).reshape(1, -1)
    return np.hstack([np.ones((len(phases_deg), 1)),
                      np.sin(phi * k), np.cos(phi * k)])


def compute_reference_trajectory(xr_pre: np.ndarray,
                                   fourier_order: int = None,
                                   resolution: int = None) -> dict:
    """Fourier-smoothed reference trajectory from pre-perturbation phase-space."""
    if fourier_order is None:
        fourier_order = config.RESILIENCE_FOURIER_ORDER
    if resolution is None:
        resolution = config.RESILIENCE_RESOLUTION_PHASE

    pre_x = xr_pre[:, 0]; pre_y = xr_pre[:, 1]; pre_z = xr_pre[:, 2]
    Xo = float(np.nanmean(pre_x))
    Yo = float(np.nanmean(pre_y))
    Zo = float(np.nanmean(pre_z))

    coeffs = _fit_plane(pre_x, pre_y, pre_z)
    normal_vec = np.array([-coeffs[1], -coeffs[2], 1.0])
    R = _vrrotvec2mat(np.array([0., 0., 1.]), normal_vec)

    proj_pre = np.column_stack([pre_x - Xo, pre_y - Yo, pre_z - Zo]) @ R
    phase_pre = (np.degrees(np.arctan2(proj_pre[:, 1], proj_pre[:, 0])) + 180)

    idx_sorted = np.argsort(np.round(phase_pre))
    phase_sorted = np.round(phase_pre)[idx_sorted].astype(int)
    pre_xs = pre_x[idx_sorted]
    pre_ys = pre_y[idx_sorted]
    pre_zs = pre_z[idx_sorted]

    phase_fit = np.concatenate([phase_sorted, phase_sorted + 360])
    x_fit = np.concatenate([pre_xs, pre_xs])
    y_fit = np.concatenate([pre_ys, pre_ys])
    z_fit = np.concatenate([pre_zs, pre_zs])
    valid = np.isfinite(phase_fit) & np.isfinite(x_fit) & np.isfinite(y_fit) & np.isfinite(z_fit)
    phase_fit = phase_fit[valid]
    x_fit = x_fit[valid]; y_fit = y_fit[valid]; z_fit = z_fit[valid]

    phase_ref = np.arange(resolution).astype(float)
    A_fit = _fourier_design_matrix(phase_fit, fourier_order)
    A_ref = _fourier_design_matrix(phase_ref, fourier_order)

    bx, *_ = np.linalg.lstsq(A_fit, x_fit, rcond=None)
    by, *_ = np.linalg.lstsq(A_fit, y_fit, rcond=None)
    bz, *_ = np.linalg.lstsq(A_fit, z_fit, rcond=None)

    ref_traj = np.column_stack([A_ref @ bx, A_ref @ by, A_ref @ bz])

    return {'ref_traj':         ref_traj,
            'center':            np.array([Xo, Yo, Zo]),
            'R':                 R,
            'phase_pre':         phase_sorted,
            'xr_pre_sorted':     np.column_stack([pre_xs, pre_ys, pre_zs])}


# ════════════════════════════════════════════════════════════════════════════
# LOCAL STD ON REFERENCE TRAJECTORY
# ════════════════════════════════════════════════════════════════════════════

def compute_local_std(xr_pre_sorted: np.ndarray,
                        phase_sorted: np.ndarray,
                        ref_traj: np.ndarray,
                        resolution: int = None) -> tuple:
    """Per-phase-angle standard deviation of pre-perturbation samples."""
    if resolution is None:
        resolution = config.RESILIENCE_RESOLUTION_PHASE

    std_x = np.full(resolution, np.nan)
    std_y = np.full(resolution, np.nan)
    std_z = np.full(resolution, np.nan)

    for phi in range(resolution):
        mask = (phase_sorted == phi)
        if mask.sum() < 2:
            continue
        ref_pt = ref_traj[phi]
        diffs = xr_pre_sorted[mask] - ref_pt
        std_x[phi] = float(np.sqrt(np.sum(diffs[:, 0]**2) / (mask.sum() - 1)))
        std_y[phi] = float(np.sqrt(np.sum(diffs[:, 1]**2) / (mask.sum() - 1)))
        std_z[phi] = float(np.sqrt(np.sum(diffs[:, 2]**2) / (mask.sum() - 1)))

    for arr in (std_x, std_y, std_z):
        nans = np.isnan(arr)
        if nans.any() and (~nans).any():
            idx = np.arange(len(arr))
            arr[nans] = np.interp(idx[nans], idx[~nans], arr[~nans])

    sigma = config.RESILIENCE_GAUSS_SMOOTH_WIN / 6.0
    return (gaussian_filter1d(std_x, sigma),
            gaussian_filter1d(std_y, sigma),
            gaussian_filter1d(std_z, sigma))


# ════════════════════════════════════════════════════════════════════════════
# XR_DISTANCES (signal of resilience)
# ════════════════════════════════════════════════════════════════════════════

def compute_xr_distances(xr: np.ndarray, ref: dict) -> tuple:
    """Euclidean distance between phase-space trajectory and reference."""
    Xo, Yo, Zo = ref['center']
    R = ref['R']
    ref_traj = ref['ref_traj']

    proj = np.column_stack([xr[:, 0] - Xo,
                              xr[:, 1] - Yo,
                              xr[:, 2] - Zo]) @ R
    phase = (np.degrees(np.arctan2(proj[:, 1], proj[:, 0])) + 180)
    phase = np.round(phase).astype(int) % 360

    distances = np.full(len(xr), np.nan)
    for i, phi in enumerate(phase):
        distances[i] = float(np.linalg.norm(xr[i, :3] - ref_traj[phi]))

    return distances, phase


# ════════════════════════════════════════════════════════════════════════════
# STABILITY ZONES (T1, T2, T3 thresholds)
# ════════════════════════════════════════════════════════════════════════════

def compute_stability_zones(xr_distances: np.ndarray,
                              xr_phase: np.ndarray,
                              std_x: np.ndarray, std_y: np.ndarray, std_z: np.ndarray,
                              pert_entry_idx: int,
                              fs: int = None) -> dict:
    """Build T1/T2/T3 stability thresholds and segment xr_distances."""
    if fs is None:
        fs = config.FS_CLEAN

    stds = np.stack([std_x, std_y, std_z], axis=1)
    stds_sorted = np.sort(stds, axis=1)[:, ::-1]

    T1 = stds_sorted[xr_phase, 0]
    T2 = stds_sorted[xr_phase, 0] + stds_sorted[xr_phase, 1]
    T3 = T2 + stds_sorted[xr_phase, 2]

    ratios = xr_distances / (T3 + 1e-12)
    ratios_base = ratios[:pert_entry_idx]
    ratios_base = ratios_base[np.isfinite(ratios_base)]
    if len(ratios_base) == 0:
        resolution4 = 1.0
    else:
        q = config.RESILIENCE_QUANTILE_BASELINE
        resolution4 = max(1.0, float(np.quantile(ratios_base, q)))

    T1 *= resolution4
    T2 *= resolution4
    T3 *= resolution4

    zone = np.full(len(xr_distances), -1, dtype=int)
    stability_01    = np.full_like(xr_distances, np.nan)
    stability_01_02 = np.full_like(xr_distances, np.nan)
    instability_02_03 = np.full_like(xr_distances, np.nan)
    instability_03  = np.full_like(xr_distances, np.nan)

    for i, d in enumerate(xr_distances):
        if not np.isfinite(d):
            continue
        if   d <= T1[i]: zone[i] = 0; stability_01[i]    = d
        elif d <= T2[i]: zone[i] = 1; stability_01_02[i] = d
        elif d <= T3[i]: zone[i] = 2; instability_02_03[i] = d
        else:            zone[i] = 3; instability_03[i]  = d

    pre = zone[:pert_entry_idx]
    n_pre = (pre >= 0).sum() if (pre >= 0).any() else 1
    pct_T1 = 100 * (pre == 0).sum() / n_pre
    pct_T2 = 100 * (pre == 1).sum() / n_pre
    pct_T3 = 100 * (pre == 2).sum() / n_pre
    pct_out = 100 * (pre == 3).sum() / n_pre

    return {'T1': T1, 'T2': T2, 'T3': T3,
            'zone': zone,
            'stability_01':      stability_01,
            'stability_01_02':   stability_01_02,
            'instability_02_03': instability_02_03,
            'instability_03':    instability_03,
            'resolution4': resolution4,
            'pct_T1': pct_T1, 'pct_T2': pct_T2,
            'pct_T3': pct_T3, 'pct_out': pct_out}


# ════════════════════════════════════════════════════════════════════════════
# RESILIENCE METRICS (lag, peak, recovery) — v3 with strict chronology
# ════════════════════════════════════════════════════════════════════════════

def detect_heel_strikes(signal_z: np.ndarray) -> np.ndarray:
    """Heel-strike proxy: negative peaks of vertical sacrum signal."""
    peaks, _ = find_peaks(-signal_z)
    return peaks


def compute_resilience_metrics(xr_distances: np.ndarray,
                                 stab: dict,
                                 pert_entry_idx: int,
                                 pert_exit_idx: int,
                                 heel_strikes: np.ndarray,
                                 fs: int = None) -> dict:
    """
    Compute lag, peak, recovery for ONE perturbation with strict chronology.

    Chronology enforced
    -------------------
    pert_entry ≤ lag ≤ peak ≤ pert_exit ≤ recovery
    (lag and recovery measured from pert_entry and pert_exit respectively)

    Criteria
    --------
    Lag      : first sample in T3 or >T3 at or after pert_entry, confirmed
               by ≥ 3 of the next N_CONSECUTIVE samples also in T3/>T3.
               If no such sample → lag = NaN.
    Peak     : local max of xr_distances on [lag, pert_exit + POST_BUFFER_S].
               If no local max → fallback to global max in the same range.
               Constraint: peak_idx ≥ lag_idx.
    Recovery : from max(peak_idx, pert_exit_idx) + 1, first sample where the
               next N_STEPS_WINDOW heel-strikes are mostly in T1/T2
               (≤ TOLERANCE_NAN unstable in between).
               Constraint: recovery_idx > peak_idx AND > pert_exit_idx.
    """
    if fs is None:
        fs = config.FS_CLEAN

    n_consec    = config.RESILIENCE_N_CONSECUTIVE
    n_steps_win = config.RESILIENCE_N_STEPS_WINDOW
    tol_nan     = config.RESILIENCE_TOLERANCE_NAN
    post_buffer = int(config.RESILIENCE_POST_PEAK_BUFFER_S * fs)
    post_valid  = int(config.RESILIENCE_POST_PEAK_VALID_S * fs)

    instability_02_03 = stab['instability_02_03']
    instability_03    = stab['instability_03']
    stability_01      = stab['stability_01']
    stability_01_02   = stab['stability_01_02']

    out = {'lag_idx':          np.nan,
           'peak_idx':         np.nan,
           'recovery_idx':     np.nan,
           'lag_time_s':       np.nan,
           'peak_time_s':      np.nan,
           'recovery_time_s':  np.nan,
           'n_steps_recovery': np.nan,
           'peak_distance_mm': np.nan}

    N = len(xr_distances)

    # ────────────────────────────────────────────────────────────────────
    # LAG : first T3/>T3 sample confirmed by 3+ of next n_consec also unstable
    # ────────────────────────────────────────────────────────────────────
    lag_idx = np.nan
    min_confirmations = 3
    for k in range(pert_entry_idx, N):
        # Is sample k itself in T3 or >T3?
        if np.isnan(instability_02_03[k]) and np.isnan(instability_03[k]):
            continue
        # Count unstable samples in next n_consec
        win_end = min(N, k + n_consec)
        w1 = instability_02_03[k: win_end]
        w2 = instability_03[k: win_end]
        n_unstable = int(np.sum(~np.isnan(w1) | ~np.isnan(w2)))
        if n_unstable >= min_confirmations:
            lag_idx = k
            break

    # ────────────────────────────────────────────────────────────────────
    # PEAK : local max on [lag, pert_exit + post_buffer]
    # ────────────────────────────────────────────────────────────────────
    peak_idx = np.nan
    if not np.isnan(lag_idx):
        search_start = int(lag_idx)
        search_end = min(N, pert_exit_idx + post_buffer)
        if search_end > search_start:
            search_range = np.arange(search_start, search_end)
            sig = xr_distances[search_range]
            # Try local maxima first (with post-validation that it's not just
            # the start of an upslope)
            all_peaks_rel, _ = find_peaks(sig)
            for pk_rel in all_peaks_rel:
                candidate = int(search_range[0]) + pk_rel
                win_end = min(N, candidate + post_valid)
                post_win = xr_distances[candidate: win_end]
                if len(post_win) <= 1 or post_win[0] >= np.max(post_win[1:]):
                    peak_idx = candidate
                    break
            # Fallback: global max
            if np.isnan(peak_idx):
                peak_idx = int(search_range[0]) + int(np.argmax(sig))

    # ────────────────────────────────────────────────────────────────────
    # RECOVERY : strict chronology recovery > max(peak, pert_exit)
    # ────────────────────────────────────────────────────────────────────
    recovery_idx = np.nan
    if not np.isnan(peak_idx):
        recovery_start = max(int(peak_idx), pert_exit_idx) + 1
        for k in range(recovery_start, N):
            # Sample k must be in T1 or T2 (stable)
            if np.isnan(stability_01_02[k]) and np.isnan(stability_01[k]):
                continue
            # Look ahead to next n_steps_win heel-strikes
            next_steps = heel_strikes[heel_strikes > k]
            if len(next_steps) < n_steps_win:
                break
            step_n = int(next_steps[n_steps_win - 1])
            rng = np.arange(k, step_n + 1)
            unstable_count = np.sum(
                np.isnan(stability_01[rng]) & np.isnan(stability_01_02[rng])
            )
            if unstable_count <= tol_nan:
                recovery_idx = k
                break

    # ────────────────────────────────────────────────────────────────────
    # Output filling with strict time references
    # ────────────────────────────────────────────────────────────────────
    if not np.isnan(lag_idx):
        out['lag_idx'] = int(lag_idx)
        out['lag_time_s'] = (lag_idx - pert_entry_idx) / fs

    if not np.isnan(peak_idx):
        out['peak_idx'] = int(peak_idx)
        out['peak_distance_mm'] = float(xr_distances[int(peak_idx)])
        if not np.isnan(lag_idx) and int(peak_idx) >= int(lag_idx):
            out['peak_time_s'] = (peak_idx - lag_idx) / fs
        # else: peak before lag = inconsistent → leave peak_time_s as NaN

    if not np.isnan(recovery_idx):
        out['recovery_idx'] = int(recovery_idx)
        out['recovery_time_s'] = (recovery_idx - pert_exit_idx) / fs
        if not np.isnan(peak_idx):
            steps_in = heel_strikes[(heel_strikes >= int(peak_idx)) &
                                      (heel_strikes <= int(recovery_idx))]
            out['n_steps_recovery'] = len(steps_in)

    return out


# ════════════════════════════════════════════════════════════════════════════
# PERSISTENT DIFFERENCE
# ════════════════════════════════════════════════════════════════════════════

def compute_persistent_difference(xr_distances: np.ndarray,
                                     pert_entry_idx: int,
                                     recovery_idx: Optional[int],
                                     fs: int = None,
                                     windows_s: list = None) -> dict:
    """
    Persistent difference: mean post-recovery distance vs pre-perturbation
    baseline (last ~5 s before perturbation entry).
    """
    if fs is None:
        fs = config.FS_CLEAN
    if windows_s is None:
        windows_s = config.RESILIENCE_PERSISTENT_WINDOWS_S

    baseline_duration_s = 5.0
    bd_samples = int(baseline_duration_s * fs)
    b_s = max(0, pert_entry_idx - bd_samples)
    b_e = max(b_s + 1, pert_entry_idx)
    baseline_mean = float(np.nanmean(xr_distances[b_s: b_e]))

    out = {'baseline_mean':       baseline_mean,
           'post_means':          {},
           'persistent_diff':     {},
           'persistent_diff_pct': {}}

    has_recovery = (recovery_idx is not None and
                    not (isinstance(recovery_idx, float) and np.isnan(recovery_idx)))
    start_post = int(recovery_idx) if has_recovery else b_e

    for w in windows_s:
        n = int(w * fs)
        end_post = min(len(xr_distances), start_post + n)
        if end_post <= start_post:
            out['post_means'][w] = np.nan
            out['persistent_diff'][w] = np.nan
            out['persistent_diff_pct'][w] = np.nan
            continue
        post_mean = float(np.nanmean(xr_distances[start_post: end_post]))
        out['post_means'][w] = post_mean
        out['persistent_diff'][w] = post_mean - baseline_mean
        out['persistent_diff_pct'][w] = 100 * (post_mean - baseline_mean) / (baseline_mean + 1e-12)

    return out


# ════════════════════════════════════════════════════════════════════════════
# DRIFT DETRENDING (linear)
# ════════════════════════════════════════════════════════════════════════════

def detrend_drift(signal: np.ndarray,
                    threshold_mm: float = None) -> tuple:
    """Linear detrend if drift amplitude exceeds threshold."""
    if threshold_mm is None:
        threshold_mm = config.RESILIENCE_THRESHOLD_DETREND_MM

    t = np.arange(len(signal), dtype=float)
    p_coef = np.polyfit(t, signal, 1)
    fit = np.polyval(p_coef, t)
    drift_amp = float(fit.max() - fit.min())

    if drift_amp > threshold_mm:
        return signal - fit, drift_amp, True
    return signal.copy(), drift_amp, False
