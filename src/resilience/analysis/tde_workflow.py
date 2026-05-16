"""
tde_workflow.py — TDE pipeline orchestration: load → clean → compute → aggregate.
"""

from __future__ import annotations
from pathlib import Path

import numpy as np
import pandas as pd

from resilience import config, participants, paths
from resilience.io import writer
from resilience.processing import detection, tde


# ════════════════════════════════════════════════════════════════════════════
# CONFIG
# ════════════════════════════════════════════════════════════════════════════

MARGIN_PRE_S  = 5.0
MARGIN_POST_S = 30.0

MAX_LAG       = 100
MAX_DIM       = 10
FNN_THRESHOLD = 10.0   # %   (Mehdizadeh 2020)
FNN_R         = 15.0   #     (Kennel 1992)

MEAN_CADENCE_HZ      = 1.75
MIN_CYCLES_FOR_FNN   = 200


# ════════════════════════════════════════════════════════════════════════════
# 1. LOAD
# ════════════════════════════════════════════════════════════════════════════

def load_trials_with_detection() -> dict:
    """
    Load every non-excluded trial along with its bbox-detected perturbation
    entry/exit timestamps (in the raw reference, i.e. including crop_start_s).

    Returns
    -------
    dict
        {(participant, condition): {
            'signal_final':  np.ndarray,
            'time_final':    np.ndarray (post-crop),
            'crop_start_s':  float,
            'entry_brut':    float (raw ref) or NaN,
            'exit_brut':     float (raw ref) or NaN,
        }}
    """
    trials = {}
    for code, P in participants.PARTICIPANTS.items():
        for cond, trial in P.trials.items():
            if trial.excluded:
                continue
            try:
                d = writer.load_processed(code, cond)
            except FileNotFoundError:
                print(f"  ⚠️  No .npz for {code}/{cond}")
                continue

            sacrum_xy = _build_sacrum_xy(d)
            det = detection.detect_sand_by_bbox(sacrum_xy)
            offset = trial.crop_start_s if trial.crop_start_s is not None else 0.0

            entry_brut = det['entry_t'] + offset if not np.isnan(det['entry_t']) else np.nan
            exit_brut  = det['exit_t']  + offset if not np.isnan(det['exit_t'])  else np.nan

            trials[(code, cond)] = {
                'signal_final': d['signal_final'].astype(float),
                'time_final':   d['time_final'].astype(float),
                'crop_start_s': offset,
                'entry_brut':   entry_brut,
                'exit_brut':    exit_brut,
            }
    return trials


def _build_sacrum_xy(npz: dict) -> np.ndarray:
    """Mean of Dos01-04 XY trajectories with linear NaN interpolation."""
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
# 2. CLEAN SIGNAL (PERTURBATION EXCLUDED)
# ════════════════════════════════════════════════════════════════════════════

def build_clean_signal(trial_data: dict,
                         margin_pre_s: float = MARGIN_PRE_S,
                         margin_post_s: float = MARGIN_POST_S) -> tuple:
    """
    Concatenate pre-perturbation and post-perturbation segments for stationary TDE.

    Returns
    -------
    sig_clean : np.ndarray
    n_kept : int
    n_excluded : int
    """
    sig  = trial_data['signal_final']
    t    = trial_data['time_final']        # post-crop, starts at 0
    offs = trial_data['crop_start_s']

    if np.isnan(trial_data['entry_brut']):
        return sig.copy(), len(sig), 0

    entry_local = trial_data['entry_brut'] - offs
    exit_local  = trial_data['exit_brut']  - offs

    mask_pre  = t < (entry_local - margin_pre_s)
    mask_post = t > (exit_local  + margin_post_s)

    sig_clean  = np.concatenate([sig[mask_pre], sig[mask_post]])
    n_excluded = int(((~mask_pre) & (~mask_post)).sum())
    return sig_clean, len(sig_clean), n_excluded


# ════════════════════════════════════════════════════════════════════════════
# 3. COMPUTE TDE
# ════════════════════════════════════════════════════════════════════════════

def compute_tde_per_trial(trials: dict, verbose: bool = True) -> tuple:
    """
    Compute AMI/FNN and pick (τ, m) for every trial.

    Returns
    -------
    df_tde : pd.DataFrame
    curves : dict[(participant, condition)] = {'ami', 'fnn', 'tau', 'm'}
    """
    results = []
    curves  = {}
    min_samples = MIN_CYCLES_FOR_FNN * config.FS_CLEAN / MEAN_CADENCE_HZ

    for (code, cond), data in trials.items():
        sig_clean, n_kept, _ = build_clean_signal(data)
        duration_s     = n_kept / config.FS_CLEAN
        n_cycles_estim = int(duration_s * MEAN_CADENCE_HZ)

        if verbose and n_kept < min_samples:
            print(f"  ⚠️  {code}/{cond}: only {n_cycles_estim} cycles (<{MIN_CYCLES_FOR_FNN})")

        tau, ami_values = tde.compute_ami(sig_clean, max_lag=MAX_LAG)
        m, fnn_rates    = tde.compute_fnn(sig_clean, max_dim=MAX_DIM, tau=tau,
                                            threshold=FNN_THRESHOLD, r_threshold=FNN_R)

        results.append({
            'participant':   code,
            'condition':     cond,
            'n_samples':     n_kept,
            'duration_s':    round(duration_s, 1),
            'n_cycles_est':  n_cycles_estim,
            'tau':           tau,
            'm':             m,
        })
        curves[(code, cond)] = {'ami': ami_values, 'fnn': fnn_rates, 'tau': tau, 'm': m}

        if verbose:
            print(f"  {code}/{cond:22s}: τ={tau:3d}, m={m}, n_cycles≈{n_cycles_estim}")

    return pd.DataFrame(results), curves


# ════════════════════════════════════════════════════════════════════════════
# 4. AGGREGATE PER PARTICIPANT
# ════════════════════════════════════════════════════════════════════════════

def aggregate_per_participant(df_tde: pd.DataFrame) -> pd.DataFrame:
    """Median τ / m across the 3 conditions for each participant."""
    df_p = df_tde.groupby('participant').agg(
        n_trials = ('tau', 'count'),
        tau_min  = ('tau', 'min'),
        tau_med  = ('tau', 'median'),
        tau_max  = ('tau', 'max'),
        m_min    = ('m',   'min'),
        m_med    = ('m',   'median'),
        m_max    = ('m',   'max'),
    ).reset_index()

    df_p['tau_p']     = df_p['tau_med'].round().astype(int)
    df_p['m_p']       = df_p['m_med'].round().astype(int)
    df_p['tau_range'] = df_p['tau_max'] - df_p['tau_min']
    df_p['m_range']   = df_p['m_max']   - df_p['m_min']
    return df_p


# ════════════════════════════════════════════════════════════════════════════
# 5. EXPORT
# ════════════════════════════════════════════════════════════════════════════

def export(df_tde: pd.DataFrame, df_p: pd.DataFrame,
            out_dir: Path | None = None) -> tuple:
    """Save per-trial and per-participant CSVs. Returns the two output paths."""
    if out_dir is None:
        out_dir = Path(paths.OUTPUTS_DIR) / 'tables'
    out_dir.mkdir(parents=True, exist_ok=True)

    path_trial = out_dir / '03_tde_per_trial.csv'
    path_part  = out_dir / '03_tde_per_participant.csv'

    df_tde.to_csv(path_trial, index=False)
    df_p[['participant', 'tau_p', 'm_p', 'tau_range', 'm_range']].to_csv(path_part, index=False)
    return path_trial, path_part
