"""
preprocess.py — Full sacrum-signal preprocessing pipeline.

Stages (in `run`):
    1. compute_barycenter(Dos01-04)            → 3D barycentre
    2. despike_mad()                            → outlier removal
    3. apply_emd_decomposition()                → walking-band IMF selection
    4. butterworth_filter()                     → zero-phase low-pass
    5. downsample_signal()                      → anti-aliased decimation

Adaptive IMF selection:
    - AUTO (default): IMFs whose dominant frequency falls in [0.5–2.5 Hz]
                      (walking band for older adults).
    - Heuristic fallback: config.EMD_IMF_INDICES if auto returns 0 IMF.
    - Explicit mode: imf_indices=(...) (legacy compatibility).

Despike (stage 2):
    - Detection: |x - median| > k · MAD (k=8 by default)
    - Action: linear interpolation of flagged samples
    - Rationale: a few trials show sporadic spikes from Codamotion
      reconstruction artefacts that would otherwise contaminate deep IMFs.
"""

from __future__ import annotations
from typing import Optional, Tuple

import numpy as np
from scipy.signal import butter, filtfilt, decimate, welch

from resilience import config


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
# INTERPOLATION
# ════════════════════════════════════════════════════════════════════════════

def interpolate_nans(signal_1d: np.ndarray) -> np.ndarray:
    """Linearly interpolate NaN values in a 1D signal."""
    s = np.array(signal_1d, dtype=float).copy()
    if not np.any(np.isnan(s)):
        return s
    idx = np.arange(len(s))
    valid = ~np.isnan(s)
    if valid.sum() < 2:
        return s
    s[~valid] = np.interp(idx[~valid], idx[valid], s[valid])
    return s


# ════════════════════════════════════════════════════════════════════════════
# BARYCENTRE
# ════════════════════════════════════════════════════════════════════════════

def compute_barycenter(trajectories: dict,
                        marker_names: tuple = ("Dos01", "Dos02", "Dos03", "Dos04"),
                        center_on_mean: bool = True) -> np.ndarray:
    """3D barycentre of the back markers (NaN-interpolated first)."""
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
# DESPIKE
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

    Parameters
    ----------
    signal_1d : np.array (N,)
        Input signal.
    k : float
        MAD multiplier (default 8, generous so that the walking signal —
        peak-to-peak ~30–60 mm, MAD ~10 mm — is not touched).
    verbose : bool

    Returns
    -------
    cleaned : np.array (N,)
        Signal with interpolated spikes.
    stats : dict
        n_spikes, threshold, indices_spike (useful for diagnostics).
    """
    s = np.asarray(signal_1d, dtype=float).copy()
    med = np.nanmedian(s)
    mad = np.nanmedian(np.abs(s - med))

    if mad < 1e-9:
        # Near-constant signal, nothing to do
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


# ════════════════════════════════════════════════════════════════════════════
# ADAPTIVE IMF SELECTION
# ════════════════════════════════════════════════════════════════════════════

def imf_dominant_frequency(imf: np.ndarray, fs: int) -> float:
    """Dominant frequency of an IMF (PSD peak via Welch)."""
    nperseg = min(len(imf), 4 * fs)
    f, pxx = welch(imf, fs=fs, nperseg=nperseg)
    if len(f) == 0 or np.all(pxx == 0):
        return 0.0
    return float(f[np.argmax(pxx)])


def select_walking_imfs(IMFs: np.ndarray,
                          fs: int,
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
# EMD
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
# BUTTERWORTH
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
# DECIMATION
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
    apply_despike : if True (default), interpolate spikes |x - med| > k·MAD
                    before EMD.
    despike_k : MAD multiplier for despike (default 8).
    apply_emd : if False, skip EMD.
    imf_indices : if provided, explicit EMD selection (legacy mode).
    band_hz : frequency band for auto IMF selection.

    Returns
    -------
    dict with keys:
        sacrum_xyz_raw, sacrum_axis_raw, sacrum_despiked, sacrum_emd,
        sacrum_filt, signal_final, time_final, fs_final, emd_applied,
        despike_applied, despike_stats, axis
    """
    if axis   is None: axis   = config.DEFAULT_AXIS
    if fs_in  is None: fs_in  = config.FS_RAW
    if fs_out is None: fs_out = config.FS_CLEAN
    if cutoff is None: cutoff = config.BUTTER_CUTOFF_HZ
    if order  is None: order  = config.BUTTER_ORDER
    if factor is None: factor = config.DECIMATE_FACTOR

    axis_idx = {"X": 0, "Y": 1, "Z": 2}[axis]

    if verbose: print("   → compute_barycenter()")
    barycenter      = compute_barycenter(trajectories)
    sacrum_axis_raw = barycenter[:, axis_idx]

    # ─── Despike (before EMD) ─────────────────────────────────────────────
    if apply_despike:
        sacrum_despiked, despike_stats = despike_mad(sacrum_axis_raw,
                                                       k=despike_k,
                                                       verbose=verbose)
        despike_applied = True
    else:
        if verbose: print("   → despike skipped")
        sacrum_despiked = sacrum_axis_raw.copy()
        despike_stats = {"n_spikes": 0, "threshold": np.nan,
                          "indices_spike": np.array([])}
        despike_applied = False

    # ─── EMD ──────────────────────────────────────────────────────────────
    if verbose: print("   → apply_emd_decomposition()" if apply_emd else "   → EMD skipped")
    sacrum_emd  = apply_emd_decomposition(sacrum_despiked, fs=fs_in,
                                            imf_indices=imf_indices,
                                            band_hz=band_hz,
                                            verbose=verbose) \
                  if apply_emd else sacrum_despiked.copy()
    emd_applied = apply_emd and (EMD_AVAILABLE is not None)

    if verbose: print(f"   → butterworth_filter(cutoff={cutoff} Hz, order={order})")
    sacrum_filt = butterworth_filter(sacrum_emd, cutoff=cutoff, fs=fs_in, order=order)

    if verbose: print(f"   → downsample_signal(factor={factor})  |  {fs_in} → {fs_out} Hz")
    signal_final = downsample_signal(sacrum_filt, factor=factor)
    time_final   = np.arange(len(signal_final)) / fs_out

    return {
        "sacrum_xyz_raw":     barycenter,
        "sacrum_axis_raw":    sacrum_axis_raw,
        "sacrum_despiked":    sacrum_despiked,
        "sacrum_emd":         sacrum_emd,
        "sacrum_filt":        sacrum_filt,
        "signal_final":       signal_final,
        "time_final":         time_final,
        "fs_final":           fs_out,
        "emd_applied":        emd_applied,
        "despike_applied":    despike_applied,
        "despike_stats":      despike_stats,
        "axis":               axis,
    }
