"""
loader.py — Load and inspect CODAMOTION .mat files.

Supports:
    - .mat v5/v6 (MATLAB < 7.3) → scipy.io.loadmat
    - .mat v7.3 (HDF5)          → h5py

Expected Odin/CODAMOTION structure:
    data['Marker'][marker_name].value    → np.array (N, 3) XYZ in mm
    data['Marker'][marker_name].occluded → np.array (N,)   1 if hidden
    data['Marker'][marker_name].Rate     → 400.0 Hz
"""

from __future__ import annotations
from pathlib import Path
from typing import Optional

import numpy as np
import scipy.io
import h5py

from resilience import config


# ════════════════════════════════════════════════════════════════════════════
# FORMAT DETECTION
# ════════════════════════════════════════════════════════════════════════════

def detect_mat_format(filepath: str | Path) -> str:
    """Return 'hdf5' for .mat v7.3 files, 'scipy' for v5/v6."""
    with open(filepath, "rb") as f:
        header = f.read(8)
    return "hdf5" if header[:4] == b"\x89HDF" else "scipy"


# ════════════════════════════════════════════════════════════════════════════
# GENERIC LOADER
# ════════════════════════════════════════════════════════════════════════════

def load_mat(filepath: str | Path) -> tuple:
    """
    Load a .mat file regardless of its version.

    Returns
    -------
    data : dict-like
    fmt  : 'hdf5' or 'scipy'
    """
    filepath = str(filepath)
    fmt = detect_mat_format(filepath)

    if fmt == "scipy":
        data = scipy.io.loadmat(filepath, squeeze_me=True, struct_as_record=False)
        data = {k: v for k, v in data.items() if not k.startswith("__")}
    else:
        data = h5py.File(filepath, "r")

    return data, fmt


# ════════════════════════════════════════════════════════════════════════════
# TRAJECTORY EXTRACTION
# ════════════════════════════════════════════════════════════════════════════

def extract_marker_trajectories(data, fmt: str,
                                 marker_names: Optional[list] = None,
                                 source: Optional[str] = None) -> Optional[dict]:
    """
    Extract XYZ trajectories from the Odin CODAMOTION structure.

    Returns
    -------
    dict { marker_name : np.array (N, 3) }  with NaN on occluded frames.
    """
    if marker_names is None:
        marker_names = config.ALL_MARKERS
    if source is None:
        source = config.DEFAULT_SOURCE

    if fmt != "scipy":
        print("⚠️  HDF5 format not currently supported.")
        return None

    if source not in data:
        print(f"⚠️  Key '{source}' missing — available keys: {list(data.keys())}")
        return None

    src = data[source]
    trajectories = {}

    for m in marker_names:
        if not hasattr(src, m):
            continue
        marker_struct = getattr(src, m)
        xyz = np.array(marker_struct.value, dtype=float)
        occ = np.array(marker_struct.occluded, dtype=bool)
        xyz[occ] = np.nan
        trajectories[m] = xyz

    return trajectories if trajectories else None


# ════════════════════════════════════════════════════════════════════════════
# VISIBILITY REPORT
# ════════════════════════════════════════════════════════════════════════════

def summary_visibility(trajectories: dict) -> dict:
    """% of non-NaN frames per marker, and group medians."""
    if not trajectories:
        return {"per_marker": {}, "medians": {}, "n_frames": 0}

    n = next(iter(trajectories.values())).shape[0]
    per_marker = {}
    for m, xyz in trajectories.items():
        valid = np.sum(~np.isnan(xyz[:, 0])) / n * 100
        per_marker[m] = round(valid, 1)

    medians = {}
    for group_name, markers in config.MARKERS.items():
        vals = [per_marker[m] for m in markers if m in per_marker]
        medians[group_name] = round(float(np.median(vals)), 1) if vals else None

    return {"per_marker": per_marker, "medians": medians, "n_frames": n}


def print_visibility_report(filepath: str | Path, trajectories: dict,
                              fs: Optional[int] = None,
                              trim_start_s: Optional[int] = None) -> None:
    """Print a formatted diagnostic report."""
    if fs is None:           fs           = config.FS_RAW
    if trim_start_s is None: trim_start_s = config.TRIM_START_S

    print("\n" + "═" * 60)
    print("  DIAGNOSTIC REPORT")
    print(f"  File: {Path(filepath).name}")
    print("═" * 60)

    if not trajectories:
        print("  ❌ No trajectory extracted — structure not recognised.")
        print("═" * 60 + "\n")
        return

    summary = summary_visibility(trajectories)
    n_frames = summary["n_frames"]
    duration_s = n_frames / fs

    print(f"  Total duration : {duration_s:.1f} s  ({n_frames} frames @ {fs} Hz)")
    print(f"  After trim     : {duration_s - trim_start_s:.1f} s")
    print(f"  Markers found  : {len(summary['per_marker'])}/{len(config.ALL_MARKERS)}")

    print("\n  Visibility per marker (% non-NaN frames):")
    for group_name, markers in config.MARKERS.items():
        median = summary["medians"].get(group_name)
        print(f"    [{group_name}]  median = {median}%")
        for m in markers:
            v = summary["per_marker"].get(m)
            if v is None:
                print(f"      ❌ {m:12s} → not found")
            else:
                flag = "⚠️ " if v < 70 else "   "
                print(f"      {flag}{m:12s} → {v}%")

    print("═" * 60 + "\n")
