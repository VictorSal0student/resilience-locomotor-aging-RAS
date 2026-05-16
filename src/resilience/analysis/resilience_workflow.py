"""
resilience_workflow.py — Locomotor resilience pipeline (hybrid bbox + PCA).

Method
------
- Perturbation : longest bbox segment (sand bed, precise)
- Baseline    : other corridor passages detected via PCA-calibrated corridor
                (excluding any passage that overlaps the perturbation ± 2 s)

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
PERT_BASELINE_MARGIN_S   = 2.0   # passages within ±2 s of pert entry are excluded
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
    dict
        sig_var      : rolling variance (centred, window win_s)
        sig_radius   : phase-space radius (distance to TDE centroid)
        sig_cadence  : local dominant frequency (Hz, non-overlapping Welch)
        time         : post-crop time vector (s)
    """
    if fs is None:
        fs = config.FS_CLEAN

    n   = len(signal_z)
    win = int(win_s * fs)

    # Rolling variance
    sig_var = np.zeros(n)
    half = win // 2
    for i in range(n):
        i0, i1 = max(0, i - half), min(n, i + half)
        sig_var[i] = np.var(signal_z[i0:i1])

    # Phase-space radius
    ps       = tde.phase_space_reconstruction(signal_z, tau=tau, dim=m)
    centroid = ps.mean(axis=0)
    radius   = np.linalg.norm(ps - centroid, axis=1)
    pad      = (m - 1) * tau
    sig_radius = np.concatenate([np.full(pad, np.nan), radius])

    # Local cadence (Welch)
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
# 3. SACRUM XY HELPER (shared with TDE workflow)
# ════════════════════════════════════════════════════════════════════════════

def _build_sacrum_xy(npz: dict) -> np.ndarray:
    """Mean of Dos01-04 XY with linear NaN interpolation."""
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
# 4. PCA CALIBRATION + TRIAL LOADING
# ════════════════════════════════════════════════════════════════════════════

def calibrate_all_participants(verbose: bool = True) -> dict:
    """Per-participant PCA corridor calibration using CRF annotations."""
    calibs = {}
    for code, P in participants.PARTICIPANTS.items():
        sacrum_dict = {}
        for cond, trial in P.trials.items():
            if trial.excluded:
                continue
            try:
                d = writer.load_processed(code, cond)
            except FileNotFoundError:
                continue
            sacrum_dict[cond] = _build_sacrum_xy(d)
        if not sacrum_dict:
            continue
        try:
            calibs[code] = detection.calibrate_for_participant(sacrum_dict, code)
        except ValueError as e:
            if verbose:
                print(f"  ⚠️  {code}: {e}")
    if verbose:
        print(f"  ✅ {len(calibs)} participants PCA-calibrated")
    return calibs


def load_trial_with_passages(code: str, cond: str, pca_calib: dict) -> dict | None:
    """
    Load a trial and detect perturbation (bbox) + baseline passages (PCA).

    Returns
    -------
    dict or None
        signal_z, time, pert_entry, pert_exit (post-crop, s),
        baseline_passages: list of (entry, exit) tuples,
        n_pca_passages: total passages detected by PCA before filtering
    """
    d = writer.load_processed(code, cond)
    signal_z = d['signal_final'].astype(float)
    time     = d['time_final'].astype(float)
    sacrum_xy = _build_sacrum_xy(d)

    # Perturbation via bbox
    det_bbox = detection.detect_sand_by_bbox(sacrum_xy, min_duration_s=MIN_PASSAGE_S)
    if det_bbox['n_segments'] == 0 or np.isnan(det_bbox['entry_t']):
        return None
    pert_entry = det_bbox['entry_t']
    pert_exit  = det_bbox['exit_t']

    # Baseline via PCA corridor
    det_pca = detection.detect_sand_transition(
        sacrum_xy,
        center=pca_calib['center'],
        axis_main=pca_calib['axis_main'],
        axis_perp=pca_calib['axis_perp'],
        s_min=pca_calib['s_min'],
        s_max=pca_calib['s_max'],
        d_max=pca_calib['d_max'],
        min_duration_s=MIN_PASSAGE_S,
    )
    all_pca_passages = [(i0 / config.FS_RAW, i1 / config.FS_RAW)
                         for (i0, i1) in det_pca['segments']]

    # Exclude passages overlapping the perturbation (±margin)
    baseline_passages = [
        (e, x) for (e, x) in all_pca_passages
        if not (abs(e - pert_entry) < PERT_BASELINE_MARGIN_S
                 or (e <= pert_exit and x >= pert_entry))
    ]

    return {'signal_z':          signal_z,
            'time':              time,
            'pert_entry':        pert_entry,
            'pert_exit':         pert_exit,
            'baseline_passages': baseline_passages,
            'n_pca_passages':    len(all_pca_passages)}


def load_all_trials(pca_calibs: dict, verbose: bool = True) -> dict:
    """Load every non-excluded trial with its passage information."""
    trials = {}
    for code, P in participants.PARTICIPANTS.items():
        if code not in pca_calibs:
            continue
        for cond, trial in P.trials.items():
            if trial.excluded:
                continue
            try:
                data = load_trial_with_passages(code, cond, pca_calibs[code])
            except FileNotFoundError:
                if verbose:
                    print(f"  ⚠️  No .npz for {code}/{cond}")
                continue
            if data is None:
                if verbose:
                    print(f"  ⚠️  {code}/{cond}: no bbox segment detected")
                continue
            trials[(code, cond)] = data
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
        tau, m = tde_params[code]
        signals = compute_resilience_signals(data['signal_z'], tau=tau, m=m)
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

        # Baseline
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
