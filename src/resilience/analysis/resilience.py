"""
resilience.py — Métriques de résilience locomotrice (Option C).

Approche : comparaison des métriques DANS le bac sable vs autres passages
couloir (baseline naturelle, marche en boucle).
"""

from __future__ import annotations
from typing import Optional

import numpy as np
import pandas as pd
from scipy.signal import welch

from resilience import config
from resilience.processing import detection
from resilience.processing.tde import phase_space_reconstruction


# ════════════════════════════════════════════════════════════════════════════
# SIGNAUX
# ════════════════════════════════════════════════════════════════════════════

def compute_resilience_signals(signal_z: np.ndarray, tau: int, m: int,
                                 fs: int = None, win_s: float = 1.0) -> dict:
    """Variance glissante + radius phase space + cadence locale."""
    if fs is None:
        fs = config.FS_CLEAN

    n = len(signal_z)
    win = int(win_s * fs)
    half = win // 2

    sig_var = np.zeros(n)
    for i in range(n):
        i0, i1 = max(0, i - half), min(n, i + half)
        sig_var[i] = np.var(signal_z[i0:i1])

    ps = phase_space_reconstruction(signal_z, tau=tau, dim=m)
    centroid = ps.mean(axis=0)
    radius = np.linalg.norm(ps - centroid, axis=1)
    pad = (m - 1) * tau
    sig_radius = np.concatenate([np.full(pad, np.nan), radius])

    sig_cadence = np.full(n, np.nan)
    step = win
    for i0 in range(0, n - win, step):
        seg = signal_z[i0:i0 + win]
        f, pxx = welch(seg, fs=fs, nperseg=min(len(seg), 512))
        mask = (f >= 0.5) & (f <= 3.0)
        if mask.any():
            sig_cadence[i0:i0 + step] = f[mask][np.argmax(pxx[mask])]

    return {"sig_var": sig_var, "sig_radius": sig_radius,
            "sig_cadence": sig_cadence, "time": np.arange(n) / fs}


# ════════════════════════════════════════════════════════════════════════════
# DÉTECTION DE TOUS LES PASSAGES
# ════════════════════════════════════════════════════════════════════════════

def detect_corridor_passages(s: np.ndarray, d: np.ndarray, fs: int,
                                s_min: float, s_max: float, d_max: float,
                                min_duration_s: float = 0.5) -> list:
    """Détecte tous les passages dans le couloir."""
    in_corridor = (s >= s_min) & (s <= s_max) & (np.abs(d) <= d_max)
    n = len(s)
    min_samples = int(min_duration_s * fs)
    passages = []
    i = 0
    while i < n:
        if in_corridor[i]:
            j = i
            while j < n and in_corridor[j]:
                j += 1
            if (j - i) >= min_samples:
                passages.append((i / fs, (j - 1) / fs))
            i = j
        else:
            i += 1
    return passages


def recalibrate_d_max(sacrum_xy_concat: np.ndarray,
                        center: np.ndarray, axis_main: np.ndarray, axis_perp: np.ndarray,
                        s_min: float, s_max: float,
                        percentile: float = 95, margin_mm: float = 200) -> Optional[float]:
    """Recalibre d_max sur toute la trajectoire, plus robuste pour marcheurs linéaires."""
    s_full, d_full = detection.project_to_axes(
        sacrum_xy_concat, center, axis_main, axis_perp
    )
    mask = (s_full >= s_min) & (s_full <= s_max)
    if mask.sum() < 100:
        return None
    return float(np.percentile(np.abs(d_full[mask]), percentile)) + margin_mm


# ════════════════════════════════════════════════════════════════════════════
# MÉTRIQUES PAR ESSAI
# ════════════════════════════════════════════════════════════════════════════

def compute_resilience_metrics(signals: dict, detection_out: dict, calib: dict,
                                 fs_raw: int = None, min_passages: int = 3,
                                 min_duration_passage_s: float = 0.5,
                                 entry_tolerance_s: float = 2.0) -> Optional[dict]:
    """Métriques résilience pour un essai."""
    if fs_raw is None:
        fs_raw = config.FS_RAW

    entry_t = detection_out["entry_t"]
    exit_t  = detection_out["exit_t"]
    if np.isnan(entry_t):
        return None

    t = signals["time"]
    mask_pert = (t >= entry_t) & (t <= exit_t)
    if mask_pert.sum() < 5:
        return None

    pert_var      = float(np.nanmean(signals["sig_var"][mask_pert]))
    pert_radius   = float(np.nanmean(signals["sig_radius"][mask_pert]))
    pert_cadence  = float(np.nanmean(signals["sig_cadence"][mask_pert]))

    passages = detect_corridor_passages(
        detection_out["s"], detection_out["d"], fs_raw,
        calib["s_min"], calib["s_max"], calib["d_max"],
        min_duration_s=min_duration_passage_s,
    )
    clean_passages = [(e, x) for (e, x) in passages
                      if not (abs(e - entry_t) < entry_tolerance_s)]

    if len(clean_passages) < min_passages:
        return None

    clean_vars, clean_radii, clean_cadences = [], [], []
    for (e, x) in clean_passages:
        mask = (t >= e) & (t <= x)
        if mask.sum() >= 5:
            clean_vars.append(np.nanmean(signals["sig_var"][mask]))
            clean_radii.append(np.nanmean(signals["sig_radius"][mask]))
            clean_cadences.append(np.nanmean(signals["sig_cadence"][mask]))

    baseline_var      = float(np.median(clean_vars))
    baseline_radius   = float(np.median(clean_radii))
    baseline_cadence  = float(np.median(clean_cadences))
    std_var           = float(np.std(clean_vars))
    std_radius        = float(np.std(clean_radii))

    return {
        "pert_duration_s":   round(exit_t - entry_t, 2),
        "n_clean_passages":  len(clean_passages),
        "pert_var":          round(pert_var, 2),
        "pert_radius":       round(pert_radius, 2),
        "pert_cadence":      round(pert_cadence, 3),
        "baseline_var":      round(baseline_var, 2),
        "baseline_radius":   round(baseline_radius, 2),
        "baseline_cadence":  round(baseline_cadence, 3),
        "z_var":             round((pert_var - baseline_var) / (std_var + 1e-6), 2),
        "z_radius":          round((pert_radius - baseline_radius) / (std_radius + 1e-6), 2),
        "ratio_var":         round(pert_var / (baseline_var + 1e-6), 2),
        "ratio_radius":      round(pert_radius / (baseline_radius + 1e-6), 2),
        "delta_cadence_hz":  round(pert_cadence - baseline_cadence, 3),
    }


# ════════════════════════════════════════════════════════════════════════════
# ORCHESTRATEUR
# ════════════════════════════════════════════════════════════════════════════

def run(datasets: dict, detections: dict, calibrations: dict,
        tde_results: dict, recalibrate: bool = True) -> tuple:
    """
    Métriques de résilience pour tous les essais.

    Parameters
    ----------
    datasets     : { participant : { condition : data_npz } }
    detections   : { participant : { condition : detection_out } }
    calibrations : { participant : calib } (modifié en place si recalibrate=True)
    tde_results  : { participant : { condition : tde_out } }

    Returns
    -------
    df : pd.DataFrame
    resilience_signals_dict : { participant : { condition : signals } }
    """
    if recalibrate:
        for p in datasets:
            all_sacrum = np.vstack([ds["sacrum_xyz_raw"][:, :2]
                                      for ds in datasets[p].values()])
            calib = calibrations[p]
            new_d_max = recalibrate_d_max(
                all_sacrum,
                calib["center"], calib["axis_main"], calib["axis_perp"],
                calib["s_min"], calib["s_max"],
            )
            if new_d_max is not None:
                calibrations[p]["d_max"] = new_d_max

    resilience_signals_dict = {}
    for p in datasets:
        resilience_signals_dict[p] = {}
        for cond, ds in datasets[p].items():
            tde_out = tde_results[p][cond]
            sig = compute_resilience_signals(
                ds["signal_final"], tau=tde_out["tau"], m=tde_out["m"],
            )
            resilience_signals_dict[p][cond] = sig

    rows = []
    for p in datasets:
        for cond, ds in datasets[p].items():
            metrics = compute_resilience_metrics(
                resilience_signals_dict[p][cond],
                detections[p][cond], calibrations[p],
            )
            if metrics is not None:
                rows.append({"participant": p, "condition": cond, **metrics})

    return pd.DataFrame(rows), resilience_signals_dict
