"""
preprocess.py — Full sacrum-signal preprocessing pipeline.

Stages (in `run`):
    1. zero_to_nan()                            → Codamotion 0-coordinate → NaN
    2. cluster_filling.run_cluster_fill()       → rigid-cluster gap filling (Dos01-04)
    3. makima_fill()                            → cubic Hermite fill of residual NaN
    4. despike_mad()                            → robust outlier removal (per marker)
    5. compute_barycenter()                     → Dos01-04 mean → 3D barycentre
    6. apply_emd_decomposition()                → walking-band IMF selection
    7. butterworth_filter()                     → zero-phase low-pass
    8. downsample_signal()                      → anti-aliased decimation

Stages 1-3 come from the MOCAP_Preprocess pipeline (Damm, Anaqi). Stage 4 is
the MAD-based outlier filter. Stages 5-8 are unchanged from the previous
version.

Adaptive IMF selection:
    - AUTO (default): IMFs whose dominant frequency falls in [0.5–2.5 Hz].
    - Heuristic fallback: config.EMD_IMF_INDICES if auto returns 0 IMF.
    - Explicit mode: imf_indices=(...) (legacy compatibility).
"""

from __future__ import annotations
from typing import Optional, Tuple

import numpy as np
from scipy.interpolate import Akima1DInterpolator
from scipy.signal import butter, filtfilt, decimate, welch

from resilience import config
from resilience.processing import cluster_filling


# ────────────────────────────────────────────────────────────────────────────
# EMD library detection
# ────────────────────────────────────────────────────────────────────────────

EMD_AVAILABLE: Optional[str] = None
try:
    from PyEMD import EMD as _PyEMD
    EMD_AVAILABLE = "PyEMD"
except ImportError:
    try:
        import emd as _emd
        EMD_AVAILABLE = "emd"
    except ImportError:
        EMD_AVAILABLE = None


# ════════════════════════════════════════════════════════════════════════════
# 1. ZERO → NAN (port of NaNMOCAP)
# ════════════════════════════════════════════════════════════════════════════

def zero_to_nan(trajectories: dict) -> dict:
    """
    Codamotion uses (0, 0, 0) as the placeholder for occluded samples.
    Convert those to NaN so the rest of the pipeline can treat them
    uniformly.
    """
    out = {}
    for m, xyz in trajectories.items():
        x = xyz.copy()
        zero_mask = (x == 0.0).any(axis=1)
        x[zero_mask] = np.nan
        out[m] = x
    return out


# ════════════════════════════════════════════════════════════════════════════
# 2. INTERPOLATION HELPERS (makima + linear NaN fill)
# ════════════════════════════════════════════════════════════════════════════

def makima_fill_1d(signal_1d: np.ndarray) -> np.ndarray:
    """
    Fill NaN samples in a 1D signal using modified Akima (makima) cubic
    Hermite interpolation. Falls back to linear interpolation at the edges
    if Akima cannot extrapolate.
    """
    s = np.asarray(signal_1d, dtype=float).copy()
    if not np.any(np.isnan(s)):
        return s
    idx = np.arange(len(s))
    valid = ~np.isnan(s)
    if valid.sum() < 2:
        return s
    interp = Akima1DInterpolator(idx[valid], s[valid], method="makima")
    s[~valid] = interp(idx[~valid])
    if np.any(np.isnan(s)):
        s[np.isnan(s)] = np.interp(idx[np.isnan(s)], idx[valid], s[valid])
    return s


def interpolate_nans(signal_1d: np.ndarray) -> np.ndarray:
    """Linearly interpolate NaN values in a 1D signal (legacy helper)."""
    s = np.array(signal_1d, dtype=float).copy()
    if not np.any(np.isnan(s)):
        return s
    idx = np.arange(len(s))
    valid = ~np.isnan(s)
    if valid.sum() < 2:
        return s
    s[~valid] = np.interp(idx[~valid], idx[valid], s[valid])
    return s


DOS_MARKERS = ("Dos01", "Dos02", "Dos03", "Dos04")


def makima_fill_trajectories(trajectories: dict) -> tuple:
    """
    Apply makima fill to every marker × axis.

    Returns
    -------
    out : dict
        Filled trajectories (same structure as input).
    stats : dict
        n_filled_dos    : samples interpolated across Dos01-04 × 3 axes
                          (residual NaN after cluster fill, the meaningful count)
        n_filled_other  : samples interpolated on the other markers
                          (feet — not used by the downstream pipeline)
        n_filled_total  : sum of the two
        per_marker      : { marker : n_samples_filled } across all 3 axes
    """
    out = {}
    per_marker = {}
    n_filled_dos   = 0
    n_filled_other = 0
    for m, xyz in trajectories.items():
        filled = xyz.copy()
        n_marker = 0
        for ax in range(3):
            n_nan_before = int(np.isnan(filled[:, ax]).sum())
            filled[:, ax] = makima_fill_1d(filled[:, ax])
            n_marker += n_nan_before
        per_marker[m] = n_marker
        if m in DOS_MARKERS:
            n_filled_dos += n_marker
        else:
            n_filled_other += n_marker
        out[m] = filled

    stats = {
        "n_filled_dos":   n_filled_dos,
        "n_filled_other": n_filled_other,
        "n_filled_total": n_filled_dos + n_filled_other,
        "per_marker":     per_marker,
    }
    return out, stats


# ════════════════════════════════════════════════════════════════════════════
# 3. BARYCENTRE
# ════════════════════════════════════════════════════════════════════════════

def compute_barycenter(trajectories: dict,
                        marker_names: tuple = ("Dos01", "Dos02", "Dos03", "Dos04"),
                        center_on_mean: bool = True) -> np.ndarray:
    """3D barycentre of the back markers. Assumes NaN have been filled."""
    markers_xyz = []
    for m in marker_names:
        if m not in trajectories:
            raise KeyError(f"Marker {m} missing from trajectories")
        xyz = trajectories[m].copy()
        for ax in range(3):
            xyz[:, ax] = interpolate_nans(xyz[:, ax])
        markers_xyz.append(xyz)

    all_markers = np.stack(markers_xyz, axis=0)
    barycenter  = np.mean(all_markers, axis=0)

    if center_on_mean:
        barycenter = barycenter - np.mean(barycenter, axis=0)

    return barycenter


# ════════════════════════════════════════════════════════════════════════════
# 4. DESPIKE (MAD)
# ════════════════════════════════════════════════════════════════════════════

def despike_mad(signal_1d: np.ndarray,
                k: float = 8.0,
                verbose: bool = True) -> tuple[np.ndarray, dict]:
    """
    Remove sporadic spikes via a MAD threshold + linear interpolation.

    Method (robust to outliers, unlike a ±n·sigma threshold):
        1. Compute median(signal) and MAD = median(|signal - median|)
        2. Spike indices: |signal - median| > k · MAD
        3. Mark as NaN, then linear interpolation
    """
    s = np.asarray(signal_1d, dtype=float).copy()
    med = np.nanmedian(s)
    mad = np.nanmedian(np.abs(s - med))

    if mad < 1e-9:
        return s, {"n_spikes": 0, "threshold": np.nan, "indices_spike": np.array([])}

    threshold = k * mad
    spike_mask = np.abs(s - med) > threshold
    n_spikes = int(spike_mask.sum())

    if n_spikes > 0:
        s[spike_mask] = np.nan
        s = interpolate_nans(s)

    if verbose:
        pct = 100.0 * n_spikes / len(signal_1d)
        print(f"   → despike_mad(k={k}): {n_spikes} spikes interpolated "
              f"({pct:.2f}% of samples, threshold=±{threshold:.1f} mm)")

    return s, {
        "n_spikes": n_spikes,
        "threshold": threshold,
        "indices_spike": np.where(spike_mask)[0],
    }


def despike_trajectories(trajectories: dict, k: float = 8.0,
                           verbose: bool = False) -> tuple:
    """Apply despike MAD to every marker × axis."""
    out = {}
    total_spikes = 0
    for m, xyz in trajectories.items():
        cleaned = xyz.copy()
        for ax in range(3):
            cleaned[:, ax], stats = despike_mad(cleaned[:, ax], k=k, verbose=False)
            total_spikes += stats["n_spikes"]
        out[m] = cleaned
    if verbose:
        print(f"   → despike on markers: {total_spikes} spikes total interpolated")
    return out, total_spikes


# ════════════════════════════════════════════════════════════════════════════
# 5. ADAPTIVE IMF SELECTION
# ════════════════════════════════════════════════════════════════════════════

def imf_dominant_frequency(imf: np.ndarray, fs: int) -> float:
    """Dominant frequency of an IMF (PSD peak via Welch)."""
    nperseg = min(len(imf), 4 * fs)
    f, pxx = welch(imf, fs=fs, nperseg=nperseg)
    if len(f) == 0 or np.all(pxx == 0):
        return 0.0
    return float(f[np.argmax(pxx)])


def select_walking_imfs(IMFs: np.ndarray, fs: int,
                          band_hz: Optional[Tuple[float, float]] = None,
                          verbose: bool = True) -> list[int]:
    """Select IMF indices whose dominant frequency falls within `band_hz`."""
    if band_hz is None:
        band_hz = config.WALKING_BAND_HZ
    f_low, f_high = band_hz

    selected = []
    freqs = []
    for i, imf in enumerate(IMFs):
        f_dom = imf_dominant_frequency(imf, fs)
        freqs.append(f_dom)
        if f_low <= f_dom <= f_high:
            selected.append(i)

    if verbose:
        freq_str = ", ".join(f"IMF{i}={f:.2f}Hz" for i, f in enumerate(freqs))
        print(f"   IMF frequencies: {freq_str}")
        print(f"   Band [{f_low}–{f_high} Hz] → selected IMFs: {selected}")

    return selected


# ════════════════════════════════════════════════════════════════════════════
# 6. EMD
# ════════════════════════════════════════════════════════════════════════════

def apply_emd_decomposition(signal_1d: np.ndarray,
                              fs: Optional[int] = None,
                              imf_indices: Optional[tuple] = None,
                              band_hz: Optional[Tuple[float, float]] = None,
                              verbose: bool = True) -> np.ndarray:
    """Decompose into IMFs and reconstruct from the chosen indices."""
    if fs is None:
        fs = config.FS_RAW

    if EMD_AVAILABLE is None:
        if verbose:
            print("   ⚠️  EMD skipped (library missing — install: pip install EMD-signal)")
        return signal_1d.copy()

    if EMD_AVAILABLE == "PyEMD":
        emd_obj = _PyEMD()
        IMFs = emd_obj(signal_1d)
    else:
        IMFs = _emd.sift.sift(signal_1d).T

    n_imfs = IMFs.shape[0]

    if imf_indices is not None:
        valid_indices = [i for i in imf_indices if i < n_imfs]
        mode = "explicit"
    else:
        valid_indices = select_walking_imfs(IMFs, fs=fs, band_hz=band_hz, verbose=verbose)
        mode = "auto"

        if not valid_indices:
            valid_indices = [i for i in config.EMD_IMF_INDICES if i < n_imfs]
            mode = "fallback heuristic"
            if verbose:
                print(f"   ⚠️  Auto selection empty → fallback {config.EMD_IMF_INDICES}")

    if not valid_indices:
        if verbose:
            print(f"   ⚠️  No valid IMF (n_imfs={n_imfs}) → signal unchanged")
        return signal_1d.copy()

    reconstructed = np.sum(IMFs[valid_indices, :], axis=0)
    if verbose:
        print(f"   EMD: {n_imfs} IMFs extracted, mode='{mode}', kept={valid_indices}")
    return reconstructed


# ════════════════════════════════════════════════════════════════════════════
# 7. BUTTERWORTH
# ════════════════════════════════════════════════════════════════════════════

def butterworth_filter(signal_1d: np.ndarray,
                        cutoff: Optional[float] = None,
                        fs: Optional[int] = None,
                        order: Optional[int] = None) -> np.ndarray:
    """Zero-phase low-pass Butterworth filter."""
    if cutoff is None: cutoff = config.BUTTER_CUTOFF_HZ
    if fs is None:     fs     = config.FS_RAW
    if order is None:  order  = config.BUTTER_ORDER

    nyquist = 0.5 * fs
    b, a = butter(order, cutoff / nyquist, btype="low")
    s = np.asarray(signal_1d, dtype=float).copy()
    s[~np.isfinite(s)] = 0.0
    return filtfilt(b, a, s)


# ════════════════════════════════════════════════════════════════════════════
# 8. DECIMATION
# ════════════════════════════════════════════════════════════════════════════

def downsample_signal(signal_1d: np.ndarray, factor: Optional[int] = None) -> np.ndarray:
    """Decimation with automatic anti-aliasing."""
    if factor is None:
        factor = config.DECIMATE_FACTOR
    return decimate(signal_1d, factor, zero_phase=True)


# ════════════════════════════════════════════════════════════════════════════
# ORCHESTRATOR
# ════════════════════════════════════════════════════════════════════════════

def run(trajectories: dict,
        axis: Optional[str] = None,
        fs_in: Optional[int] = None,
        fs_out: Optional[int] = None,
        cutoff: Optional[float] = None,
        order: Optional[int] = None,
        factor: Optional[int] = None,
        apply_cluster_fill: bool = True,
        apply_makima: bool = True,
        apply_despike: bool = True,
        despike_k: float = 8.0,
        apply_emd: bool = True,
        imf_indices: Optional[tuple] = None,
        band_hz: Optional[Tuple[float, float]] = None,
        verbose: bool = True) -> dict:
    """
    Full preprocessing pipeline.

    Parameters
    ----------
    trajectories : dict
        Output of loader.extract_marker_trajectories (Dos01-04 required).
    axis : 'X', 'Y' or 'Z' — axis used for the final 1D signal.
    fs_in, fs_out : input/output sampling rates.
    cutoff, order : Butterworth low-pass parameters.
    factor : decimation factor.
    apply_cluster_fill : run rigid-cluster gap filling on Dos01-04.
    apply_makima : makima-interpolate residual NaN before despike.
    apply_despike : MAD despike on every marker (k = despike_k).
    apply_emd : run EMD walking-band reconstruction.
    imf_indices : if provided, explicit EMD selection.
    band_hz : frequency band for auto IMF selection.

    Returns
    -------
    dict with keys:
        sacrum_xyz_raw, sacrum_axis_raw, sacrum_despiked, sacrum_emd,
        sacrum_filt, signal_final, time_final, fs_final, axis,
        emd_applied, despike_applied, despike_stats,
        cluster_fill_applied, cluster_fill_stats,
        makima_applied, n_makima_filled
    """
    if axis   is None: axis   = config.DEFAULT_AXIS
    if fs_in  is None: fs_in  = config.FS_RAW
    if fs_out is None: fs_out = config.FS_CLEAN
    if cutoff is None: cutoff = config.BUTTER_CUTOFF_HZ
    if order  is None: order  = config.BUTTER_ORDER
    if factor is None: factor = config.DECIMATE_FACTOR

    axis_idx = {"X": 0, "Y": 1, "Z": 2}[axis]

    # ─── 1. Zero → NaN ──────────────────────────────────────────────────
    if verbose: print("   → zero_to_nan()")
    trajectories = zero_to_nan(trajectories)

    # ─── 2. Cluster fill (Dos01-04) ─────────────────────────────────────
    if apply_cluster_fill:
        if verbose: print("   → cluster_filling.run_cluster_fill()")
        try:
            filled_dos, cluster_stats = cluster_filling.run_cluster_fill(
                trajectories,
                marker_names=("Dos01", "Dos02", "Dos03", "Dos04"),
                verbose=verbose,
            )
            for m, arr in filled_dos.items():
                trajectories[m] = arr
            cluster_fill_applied = True
        except ValueError as e:
            if verbose:
                print(f"   ⚠️  cluster fill skipped: {e}")
            cluster_stats = {"error": str(e)}
            cluster_fill_applied = False
    else:
        if verbose: print("   → cluster fill skipped")
        cluster_stats = {}
        cluster_fill_applied = False

    # ─── 3. Makima fill on residual NaN ─────────────────────────────────
    if apply_makima:
        if verbose: print("   → makima_fill_trajectories()")
        trajectories, makima_stats = makima_fill_trajectories(trajectories)
        makima_applied = True
        if verbose:
            print(f"     → Dos01-04 : {makima_stats['n_filled_dos']} samples filled "
                  f"(residual after cluster fill)")
            print(f"     → Feet     : {makima_stats['n_filled_other']} samples filled")
    else:
        if verbose: print("   → makima skipped")
        makima_stats = {"n_filled_dos": 0, "n_filled_other": 0,
                          "n_filled_total": 0, "per_marker": {}}
        makima_applied = False

    # ─── 4. Despike MAD per marker ──────────────────────────────────────
    if apply_despike:
        if verbose: print(f"   → despike_trajectories(k={despike_k})")
        trajectories, n_marker_spikes = despike_trajectories(
            trajectories, k=despike_k, verbose=verbose)
    else:
        if verbose: print("   → despike skipped")
        n_marker_spikes = 0

    # ─── 5. Barycentre ──────────────────────────────────────────────────
    if verbose: print("   → compute_barycenter()")
    barycenter      = compute_barycenter(trajectories)
    sacrum_axis_raw = barycenter[:, axis_idx]

    if apply_despike:
        sacrum_despiked, despike_stats = despike_mad(sacrum_axis_raw,
                                                      k=despike_k,
                                                      verbose=verbose)
    else:
        sacrum_despiked = sacrum_axis_raw.copy()
        despike_stats = {"n_spikes": 0, "threshold": np.nan,
                          "indices_spike": np.array([])}

    # ─── 6. EMD ─────────────────────────────────────────────────────────
    if verbose: print("   → apply_emd_decomposition()" if apply_emd else "   → EMD skipped")
    sacrum_emd  = apply_emd_decomposition(sacrum_despiked, fs=fs_in,
                                            imf_indices=imf_indices,
                                            band_hz=band_hz,
                                            verbose=verbose) \
                  if apply_emd else sacrum_despiked.copy()
    emd_applied = apply_emd and (EMD_AVAILABLE is not None)

    # ─── 7. Butterworth ─────────────────────────────────────────────────
    if verbose: print(f"   → butterworth_filter(cutoff={cutoff} Hz, order={order})")
    sacrum_filt = butterworth_filter(sacrum_emd, cutoff=cutoff, fs=fs_in, order=order)

    # ─── 8. Decimation ──────────────────────────────────────────────────
    if verbose: print(f"   → downsample_signal(factor={factor})  |  {fs_in} → {fs_out} Hz")
    signal_final = downsample_signal(sacrum_filt, factor=factor)
    time_final   = np.arange(len(signal_final)) / fs_out

    return {
        "sacrum_xyz_raw":         barycenter,
        "sacrum_axis_raw":        sacrum_axis_raw,
        "sacrum_despiked":        sacrum_despiked,
        "sacrum_emd":             sacrum_emd,
        "sacrum_filt":            sacrum_filt,
        "signal_final":           signal_final,
        "time_final":             time_final,
        "fs_final":               fs_out,
        "axis":                   axis,
        "emd_applied":            emd_applied,
        "despike_applied":        apply_despike,
        "despike_stats":          despike_stats,
        "cluster_fill_applied":   cluster_fill_applied,
        "cluster_fill_stats":     cluster_stats,
        "makima_applied":         makima_applied,
        "makima_stats":           makima_stats,
    }
