"""
writer.py — Save preprocessed results to .npz files.
"""

from __future__ import annotations
from pathlib import Path
import numpy as np

from resilience.paths import processed_file, processed_file_for_writing, ensure_dir


def save_processed(participant: str, condition: str, preprocess_out: dict,
                     trajectories_raw: dict | None = None) -> Path:
    """
    Save the output of preprocess.run() to a compressed .npz file.

    Convention: PROCESSED_DIR/{participant}/{participant}_{NN}_{condition}.npz
    The NN index is recovered from the matching raw .mat filename
    (raw <-> processed traceability).

    Parameters
    ----------
    participant : str
    condition : str
    preprocess_out : dict
        Output of processing.preprocess.run().
    trajectories_raw : dict, optional
        If provided, raw Dos01-04 XYZ trajectories are stored alongside
        the processed signal (used downstream for sand detection).

    Returns
    -------
    Path
        Absolute path of the written .npz file.
    """
    out_path = processed_file_for_writing(participant, condition)
    ensure_dir(out_path.parent)

    payload = {
        # Core preprocessed signals
        "sacrum_xyz_raw":        preprocess_out["sacrum_xyz_raw"],
        "sacrum_axis_raw":       preprocess_out["sacrum_axis_raw"],
        "sacrum_despiked":       preprocess_out["sacrum_despiked"],
        "sacrum_emd":            preprocess_out["sacrum_emd"],
        "sacrum_filt":           preprocess_out["sacrum_filt"],
        "signal_final":          preprocess_out["signal_final"],
        "time_final":            preprocess_out["time_final"],
        # Metadata
        "fs_final":              preprocess_out["fs_final"],
        "axis":                  preprocess_out["axis"],
        "emd_applied":           preprocess_out["emd_applied"],
        "despike_applied":       preprocess_out["despike_applied"],
        "cluster_fill_applied":  preprocess_out["cluster_fill_applied"],
        "makima_applied":        preprocess_out["makima_applied"],
        # Diagnostics (dicts wrapped as object arrays, unpack via .item())
        "despike_stats":         np.array(preprocess_out["despike_stats"],     dtype=object),
        "cluster_fill_stats":    np.array(preprocess_out["cluster_fill_stats"], dtype=object),
        "makima_stats":          np.array(preprocess_out["makima_stats"],       dtype=object),
        # Trial identifiers
        "participant":           participant,
        "condition":             condition,
    }

    # Optional raw back markers (Dos01-04) for downstream sand detection
    if trajectories_raw is not None:
        for m in ("Dos01", "Dos02", "Dos03", "Dos04"):
            if m in trajectories_raw:
                payload[f"raw_{m}"] = trajectories_raw[m]

    np.savez_compressed(out_path, **payload)
    return out_path


def load_processed(participant: str, condition: str) -> dict:
    """
    Load a preprocessed .npz file and return a dict of arrays.

    Raises
    ------
    FileNotFoundError
        If no .npz exists for this participant/condition pair.
    """
    path = processed_file(participant, condition)
    if not path.exists():
        raise FileNotFoundError(f"No .npz for {participant}/{condition}: {path}")
    with np.load(path, allow_pickle=True) as data:
        return {k: data[k] for k in data.files}
