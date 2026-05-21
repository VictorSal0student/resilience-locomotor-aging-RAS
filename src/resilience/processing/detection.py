"""
detection.py — Automatic detection of the hard-floor ↔ sand transition.

Method: absolute BBOX — point-in-rectangle test using the sand bed corner
coordinates measured at CODAMOTION (config.BAC_SAND_CORNERS_MM).
Independent of CRF annotations, hence usable even when CRFs are missing.
"""

from __future__ import annotations
from typing import Optional

import numpy as np

from resilience import config
from resilience.processing.preprocess import interpolate_nans


# ════════════════════════════════════════════════════════════════════════════
# SACRUM XY BARYCENTRE
# ════════════════════════════════════════════════════════════════════════════

def compute_sacrum_xy(trajectories: dict,
                        marker_names: tuple = ("Dos01", "Dos02", "Dos03", "Dos04")
                        ) -> np.ndarray:
    """Dos01-04 XY barycentre (NOT centred, absolute position in room frame)."""
    stack = []
    for m in marker_names:
        if m not in trajectories:
            continue
        xy = trajectories[m][:, :2].copy()
        xy[:, 0] = interpolate_nans(xy[:, 0])
        xy[:, 1] = interpolate_nans(xy[:, 1])
        stack.append(xy)

    if not stack:
        raise KeyError(f"None of the markers {marker_names} were found")

    return np.mean(np.stack(stack, axis=0), axis=0)


# ════════════════════════════════════════════════════════════════════════════
# POINT-IN-RECTANGLE TEST
# ════════════════════════════════════════════════════════════════════════════

def point_in_rectangle(xy: np.ndarray, corners: dict,
                         margin_mm: float = 0.0) -> np.ndarray:
    """
    Point-in-rectangle test for an arbitrary rectangle (not necessarily
    axis-aligned).

    Method: for each point, compute the 2 dot products on the local axes of
    the rectangle (AB and AD). A point is inside iff both projections fall
    in [0, |AB|] and [0, |AD|] respectively.

    Parameters
    ----------
    xy : np.array (N, 2)
        Points to test (mm).
    corners : dict {'A', 'B', 'C', 'D'}
        Rectangle corners in order (front-left, front-right, rear-right,
        rear-left). Coordinates in mm.
    margin_mm : float
        Margin added around the rectangle (extension in every direction).

    Returns
    -------
    np.array (N,) bool: True if the point lies inside (± margin).
    """
    A = np.array(corners['A'], dtype=float)
    B = np.array(corners['B'], dtype=float)
    D = np.array(corners['D'], dtype=float)

    AB = B - A
    AD = D - A
    len_AB = np.linalg.norm(AB)
    len_AD = np.linalg.norm(AD)
    u_AB = AB / len_AB
    u_AD = AD / len_AD

    AP = xy - A
    proj_AB = AP @ u_AB
    proj_AD = AP @ u_AD

    return ((proj_AB >= -margin_mm) & (proj_AB <= len_AB + margin_mm) &
            (proj_AD >= -margin_mm) & (proj_AD <= len_AD + margin_mm))


# ════════════════════════════════════════════════════════════════════════════
# BBOX DETECTOR
# ════════════════════════════════════════════════════════════════════════════

def detect_sand_by_bbox(sacrum_xy: np.ndarray,
                          corners: Optional[dict] = None,
                          margin_mm: Optional[float] = None,
                          fs: Optional[int] = None,
                          min_duration_s: Optional[float] = None,
                          trim_end_s: Optional[float] = None) -> dict:
    """
    Sand-crossing detection via point-in-rectangle test.

    Method independent of CRF annotations, based on the physical sand bed
    corner coordinates measured at CODAMOTION (config.BAC_SAND_CORNERS_MM).

    Parameters
    ----------
    sacrum_xy : np.array (N, 2)
        Dos01-04 XY barycentre (absolute room-frame position, mm).
    corners : dict, optional
        Bed corners. Default: config.BAC_SAND_CORNERS_MM.
    margin_mm : float, optional
        ± margin around the bed. Default: config.BAC_SAND_MARGIN_MM.
    fs : int, optional
        Sampling frequency. Default: config.FS_RAW.
    min_duration_s : float, optional
        Minimum segment duration to be retained.
    trim_end_s : float, optional
        Ignore any segment after this timestamp (end-of-trial artefacts).

    Returns
    -------
    dict with keys:
        entry_t, exit_t : timestamps of the LONGEST in-bbox segment (s)
        in_bbox : np.array bool (N,) — "inside the bed" mask
        segments : list of (i_start, i_end) — all detected segments
        n_segments : int
        method : 'bbox'
    """
    if corners is None:        corners        = config.BAC_SAND_CORNERS_MM
    if margin_mm is None:      margin_mm      = config.BAC_SAND_MARGIN_MM
    if fs is None:             fs             = config.FS_RAW
    if min_duration_s is None: min_duration_s = config.DETECT_MIN_DURATION_S

    n = len(sacrum_xy)
    t = np.arange(n) / fs

    in_bbox = point_in_rectangle(sacrum_xy, corners, margin_mm=margin_mm)

    if trim_end_s is not None:
        in_bbox[t >= trim_end_s] = False

    # Extract continuous segments
    min_samples = int(min_duration_s * fs)
    segments = []
    i = 0
    while i < n:
        if in_bbox[i]:
            j = i
            while j < n and in_bbox[j]:
                j += 1
            if (j - i) >= min_samples:
                segments.append((i, j - 1))
            i = j
        else:
            i += 1

    if not segments:
        return {"entry_t": np.nan, "exit_t": np.nan,
                "in_bbox": in_bbox,
                "segments": [], "n_segments": 0,
                "method": "bbox"}

    # Longest segment = perturbation
    longest = max(segments, key=lambda x: x[1] - x[0])
    return {"entry_t":    longest[0] / fs,
            "exit_t":     longest[1] / fs,
            "in_bbox":    in_bbox,
            "segments":   segments,
            "n_segments": len(segments),
            "method":     "bbox"}


# ════════════════════════════════════════════════════════════════════════════
# VALIDATION
# ════════════════════════════════════════════════════════════════════════════

def validate_vs_crf(detection: dict, crf_entry: float, crf_exit: float) -> dict:
    """Compare an automatic detection against the CRF timestamps."""
    err_in   = (detection["entry_t"] - crf_entry) if not np.isnan(detection["entry_t"]) else np.nan
    err_out  = (detection["exit_t"]  - crf_exit)  if not np.isnan(detection["exit_t"])  else np.nan
    dur_auto = detection["exit_t"] - detection["entry_t"] if not np.isnan(detection["entry_t"]) else np.nan
    dur_crf  = crf_exit - crf_entry
    return {"err_entry_s": err_in, "err_exit_s": err_out,
            "duration_auto": dur_auto, "duration_crf": dur_crf,
            "duration_diff": dur_auto - dur_crf if not np.isnan(dur_auto) else np.nan}
