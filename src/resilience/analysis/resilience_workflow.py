"""
resilience_workflow.py — Locomotor resilience pipeline (BBOX-only).

Method
------
- Perturbation : longest segment in the sand-bed bbox
                 (config.BAC_SAND_CORNERS_MM, 4.4m × 1m)
- Baseline    : all passages in the lateral baseline bbox
                 (config.BAC_BASELINE_LEFT_CORNERS_MM, 4.4m × 1m,
                  adjacent to the bed on the walker's left side)

Both zones are detected by the same point-in-rectangle test (BBOX), ensuring
strict geometric comparability between perturbation and baseline.

Time reference
--------------
NOTE: in the .npz produced by the preprocess, the raw_Dos arrays are already
in POST-CROP reference (the first crop_start_s seconds have been removed
upstream), even though their name suggests otherwise. As a consequence,
BBOX-derived timestamps (i / FS_RAW) are directly in post-crop reference
and match time_final without any conversion needed.

Metrics per trial
-----------------
- z_var, z_radius     : (pert - median_baseline) / std_baseline
- ratio_var, ratio_radius : pert / baseline (unsigned)
- delta_cadence_hz    : pert_cadence - baseline_cadence

Exploitability criterion: at least 3 valid baseline passages, else NaN.
"""

from __future__ import annotations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import welch

from resilience import config, participants, paths
from resilience.io import writer
from resilience.processing import detection, tde


# ════════════════════════════════════════════════════════════════════════════
# CONFIG
# ════════════════════════════════════════════════════════════════════════════

MIN_PASSAGE_S            = 0.5
MIN_BASELINE_PASSAGES    = 3
MIN_SAMPLES_PER_PASSAGE  = 5
WELCH_WIN_S              = 1.0


# ════════════════════════════════════════════════════════════════════════════
# 1. TDE PARAMS
# ════════════════════════════════════════════════════════════════════════════

def load_tde_params(csv_path: Path | None = None) -> dict:
    """Load per-participant TDE parameters from the export of notebook 03."""
    if csv_path is None:
        csv_path = Path(paths.OUTPUTS_DIR) / 'tables' / '03_tde_per_participant.csv'
    df = pd.read_csv(csv_path)
    return {row['participant']: (int(row['tau_p']), int(row['m_p']))
            for _, row in df.iterrows()}


# ════════════════════════════════════════════════════════════════════════════
# 2. RESILIENCE SIGNALS
# ════════════════════════════════════════════════════════════════════════════

def compute_resilience_signals(signal_z: np.ndarray, tau: int, m: int,
                                 fs: int | None = None,
                                 win_s: float = WELCH_WIN_S) -> dict:
    """
    Compute the 3 resilience signals from the vertical sacrum trace.

    Returns
    -------
    dict with keys:
        sig_var      : rolling variance (centred, window win_s)
        sig_radius   : phase-space radius (distance to TDE centroid)
        sig_cadence  : local dominant frequency (Hz, non-overlapping Welch)
        time         : post-crop time vector (s)
    """
    if fs is None:
        fs = config.FS_CLEAN

    n   = len(signal_z)
    win = int(win_s * fs)

    sig_var = np.zeros(n)
    half = win // 2
    for i in range(n):
        i0, i1 = max(0, i - half), min(n, i + half)
        sig_var[i] = np.var(signal_z[i0:i1])

    ps       = tde.phase_space_reconstruction(signal_z, tau=tau, dim=m)
    centroid = ps.mean(axis=0)
    radius   = np.linalg.norm(ps - centroid, axis=1)
    pad      = (m - 1) * tau
    sig_radius = np.concatenate([np.full(pad, np.nan), radius])

    sig_cadence = np.full(n, np.nan)
    step = win
    for i0 in range(0, n - win, step):
        seg = signal_z[i0:i0 + win]
        f, pxx = welch(seg, fs=fs, nperseg=min(len(seg), 512))
        mask = (f >= 0.5) & (f <= 3.0)
        if mask.any():
            sig_cadence[i0:i0 + step] = f[mask][np.argmax(pxx[mask])]

    return {'sig_var':     sig_var,
            'sig_radius':  sig_radius,
            'sig_cadence': sig_cadence,
            'time':        np.arange(n) / fs}


# ════════════════════════════════════════════════════════════════════════════
# 3. SACRUM XY HELPER
# ════════════════════════════════════════════════════════════════════════════

def _build_sacrum_xy(npz: dict) -> np.ndarray:
    """Mean of Dos01-04 XY (nan-safe) with linear NaN interpolation."""
    dos_xy = np.stack([npz[f'raw_Dos{i:02d}'][:, :2] for i in range(1, 5)], axis=0)
    sacrum_xy = np.nanmean(dos_xy, axis=0)
    for ax in range(2):
        col = sacrum_xy[:, ax]
        if np.isnan(col).any():
            idx = np.arange(len(col))
            valid = ~np.isnan(col)
            if valid.sum() >= 2:
                col[~valid] = np.interp(idx[~valid], idx[valid], col[valid])
            sacrum_xy[:, ax] = col
    return sacrum_xy


# ════════════════════════════════════════════════════════════════════════════
# 4. TRIAL LOADING (perturbation + baseline via BBOX)
# ════════════════════════════════════════════════════════════════════════════

def load_trial_with_passages(code: str, cond: str) -> dict | None:
    """
    Load a trial and detect:
        - perturbation: longest segment in sand bbox
        - baseline passages: all segments in lateral baseline bbox

    All timestamps are in POST-CROP reference (matching time_final).
    Note: raw_Dos arrays in the .npz are already post-crop despite their name,
    so BBOX timestamps (i / FS_RAW) need no further conversion.

    Returns
    -------
    dict or None
        signal_z, time, fs,
        pert_entry, pert_exit  (post-crop, s),
        baseline_passages: list of (entry, exit) tuples (post-crop, s),
        n_pert_segments_raw, n_baseline_segments_raw
    """
    d = writer.load_processed(code, cond)
    signal_z = d['signal_final'].astype(float)
    time     = d['time_final'].astype(float)
    fs       = int(d['fs_final'])
    sacrum_xy = _build_sacrum_xy(d)

    # ---- Perturbation: longest segment in sand bbox
    det_sand = detection.detect_sand_by_bbox(
        sacrum_xy,
        corners=config.BAC_SAND_CORNERS_MM,
        margin_mm=config.BAC_SAND_MARGIN_MM,
        min_duration_s=MIN_PASSAGE_S,
    )
    if det_sand['n_segments'] == 0 or np.isnan(det_sand['entry_t']):
        return None
    pert_entry = det_sand['entry_t']
    pert_exit  = det_sand['exit_t']

    # ---- Baseline: all segments in lateral baseline bbox
    det_baseline = detection.detect_sand_by_bbox(
        sacrum_xy,
        corners=config.BAC_BASELINE_LEFT_CORNERS_MM,
        margin_mm=config.BAC_BASELINE_LEFT_MARGIN_MM,
        min_duration_s=MIN_PASSAGE_S,
    )
    baseline_passages = [
        (i0 / config.FS_RAW, i1 / config.FS_RAW)
        for (i0, i1) in det_baseline['segments']
    ]

    return {'signal_z':                 signal_z,
            'time':                     time,
            'fs':                       fs,
            'pert_entry':               pert_entry,
            'pert_exit':                pert_exit,
            'baseline_passages':        baseline_passages,
            'n_pert_segments_raw':      det_sand['n_segments'],
            'n_baseline_segments_raw':  det_baseline['n_segments']}


def load_all_trials(verbose: bool = True) -> dict:
    """Load every non-excluded trial with perturbation + baseline passages."""
    trials = {}
    for code, P in participants.PARTICIPANTS.items():
        for cond, trial in P.trials.items():
            if trial.excluded:
                continue
            try:
                data = load_trial_with_passages(code, cond)
            except FileNotFoundError:
                if verbose:
                    print(f"  ⚠️  No .npz for {code}/{cond}")
                continue
            if data is None:
                if verbose:
                    print(f"  ⚠️  {code}/{cond}: no sand bbox segment detected")
                continue
            trials[(code, cond)] = data
    if verbose:
        print(f"  ✅ {len(trials)} trials loaded")
    return trials


# ════════════════════════════════════════════════════════════════════════════
# 5. METRICS
# ════════════════════════════════════════════════════════════════════════════

def _nan_row(code: str, cond: str, n_baseline: int, pert_duration: float) -> dict:
    return {'participant':         code,
            'condition':           cond,
            'n_baseline_passages': n_baseline,
            'pert_duration_s':     round(pert_duration, 2),
            **{k: np.nan for k in (
                'pert_var', 'pert_radius', 'pert_cadence',
                'baseline_var', 'baseline_radius', 'baseline_cadence',
                'z_var', 'z_radius', 'ratio_var', 'ratio_radius',
                'delta_cadence_hz')}}


def compute_metrics(trials: dict, tde_params: dict,
                      verbose: bool = True) -> tuple:
    """
    Compute resilience metrics for every trial.

    Returns
    -------
    df : pd.DataFrame
    signals_cache : dict[(code, cond)] = signals dict
    """
    rows = []
    signals_cache = {}

    for (code, cond), data in trials.items():
        if code not in tde_params:
            if verbose:
                print(f"  ⚠️  {code}/{cond}: no TDE params, skip")
            continue
        tau, m = tde_params[code]
        signals = compute_resilience_signals(data['signal_z'], tau=tau, m=m,
                                                fs=data['fs'])
        signals_cache[(code, cond)] = signals
        t = signals['time']
        pert_duration = data['pert_exit'] - data['pert_entry']

        # Perturbation window
        pert_mask = (t >= data['pert_entry']) & (t <= data['pert_exit'])
        if pert_mask.sum() < MIN_SAMPLES_PER_PASSAGE:
            if verbose:
                print(f"  ⚠️  {code}/{cond}: perturbation too short ({pert_mask.sum()} samples)")
            continue

        pert_var     = float(np.nanmean(signals['sig_var'][pert_mask]))
        pert_radius  = float(np.nanmean(signals['sig_radius'][pert_mask]))
        pert_cadence = float(np.nanmean(signals['sig_cadence'][pert_mask]))

        # Baseline windows
        n_bl_raw = len(data['baseline_passages'])
        if n_bl_raw < MIN_BASELINE_PASSAGES:
            if verbose:
                print(f"  ⚠️  {code}/{cond}: only {n_bl_raw} baseline passages (<{MIN_BASELINE_PASSAGES}) → NaN")
            rows.append(_nan_row(code, cond, n_bl_raw, pert_duration))
            continue

        bvars, bradii, bcads = [], [], []
        for (e, x) in data['baseline_passages']:
            msk = (t >= e) & (t <= x)
            if msk.sum() < MIN_SAMPLES_PER_PASSAGE:
                continue
            bvars.append(float(np.nanmean(signals['sig_var'][msk])))
            bradii.append(float(np.nanmean(signals['sig_radius'][msk])))
            bcads.append(float(np.nanmean(signals['sig_cadence'][msk])))

        if len(bvars) < MIN_BASELINE_PASSAGES:
            if verbose:
                print(f"  ⚠️  {code}/{cond}: only {len(bvars)} usable baseline passages → NaN")
            rows.append(_nan_row(code, cond, len(bvars), pert_duration))
            continue

        baseline_var     = float(np.median(bvars))
        baseline_radius  = float(np.median(bradii))
        baseline_cadence = float(np.median(bcads))
        std_var          = float(np.std(bvars))
        std_radius       = float(np.std(bradii))

        rows.append({
            'participant':         code,
            'condition':           cond,
            'n_baseline_passages': len(bvars),
            'pert_duration_s':     round(pert_duration, 2),
            'pert_var':            round(pert_var, 2),
            'pert_radius':         round(pert_radius, 2),
            'pert_cadence':        round(pert_cadence, 3),
            'baseline_var':        round(baseline_var, 2),
            'baseline_radius':     round(baseline_radius, 2),
            'baseline_cadence':    round(baseline_cadence, 3),
            'z_var':               round((pert_var - baseline_var)       / (std_var + 1e-6), 2),
            'z_radius':            round((pert_radius - baseline_radius) / (std_radius + 1e-6), 2),
            'ratio_var':           round(pert_var    / (baseline_var + 1e-6), 2),
            'ratio_radius':        round(pert_radius / (baseline_radius + 1e-6), 2),
            'delta_cadence_hz':    round(pert_cadence - baseline_cadence, 3),
        })

    return pd.DataFrame(rows), signals_cache


# ════════════════════════════════════════════════════════════════════════════
# 6. EXPORT
# ════════════════════════════════════════════════════════════════════════════

def export(df: pd.DataFrame, out_dir: Path | None = None) -> Path:
    """Write resilience metrics to outputs/tables/."""
    if out_dir is None:
        out_dir = Path(paths.OUTPUTS_DIR) / 'tables'
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / '04_resilience_metrics.csv'
    df.to_csv(path, index=False)
    return path
