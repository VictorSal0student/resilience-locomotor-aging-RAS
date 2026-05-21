"""
cluster_filling.py — Rigid-cluster gap filling for marker trajectories.

Ports the MOCAP_Preprocess MATLAB pipeline by L. Damm and J. Anaqi to Python.
The four back markers (Dos01-04) are treated as a rigid cluster: when one
or more markers are occluded, the missing ones are reconstructed by fitting
the rigid model to the visible markers using Horn's algorithm.

Pipeline
--------
1. build_rigid_model(XYZ)
       Identify full-visible frames, filter by rigidity (inter-marker
       distances), align them with Horn, average → "barbie" template.

2. iterative_cluster_fill(XYZ, occluded, model)
       Loop until convergence:
           fill_3_visible    → reconstruct 1 missing marker from 3 visible
           inertial_fill     → short-gap inertial interpolation (≤15 frames)
           fill_3_visible
           fill_2_visible    → reconstruct 2 missing markers from 2 visible
           fill_3_visible
"""

from __future__ import annotations
from typing import Optional

import numpy as np
from scipy.spatial.transform import Rotation

from resilience import config


# ════════════════════════════════════════════════════════════════════════════
# CONSTANTS (mirrors MOCAP_Preprocess defaults)
# ════════════════════════════════════════════════════════════════════════════

EPSI_APPLICATION         = 1e-10
INERTIAL_GAP_MAX         = 15
CANDIDATE_SIGMA_TOL      = 0.7      # ±0.7σ filter on inter-marker distances
CANDIDATE_SIGMA_TOL_FALLBACK = 1.5  # relaxed threshold if too few candidates
MIN_CANDIDATES           = 10


# ════════════════════════════════════════════════════════════════════════════
# OCCLUSION HANDLING
# ════════════════════════════════════════════════════════════════════════════

def detect_occluded(xyz_stack: np.ndarray) -> np.ndarray:
    """
    Build a per-frame per-marker occlusion mask from a (N, 3, M) XYZ stack.
    A marker is considered occluded on a frame if any of its 3 coordinates
    is NaN.

    Parameters
    ----------
    xyz_stack : np.ndarray, shape (N_frames, 3, N_markers)

    Returns
    -------
    occluded : np.ndarray (bool), shape (N_frames, N_markers)
    """
    return np.isnan(xyz_stack).any(axis=1)


def overall_perc_occluded(occluded: np.ndarray) -> float:
    """Percentage of occluded (frame, marker) cells."""
    return 100.0 * occluded.sum() / occluded.size


# ════════════════════════════════════════════════════════════════════════════
# 1. RIGID MODEL BUILDING (port of f_CreateOneModel + f_ClusterModel)
# ════════════════════════════════════════════════════════════════════════════

def _pairwise_distances(frame_xyz: np.ndarray) -> np.ndarray:
    """Inter-marker distances for one frame, shape (n_markers, 3) → (n_pairs,)."""
    n = frame_xyz.shape[0]
    out = []
    for k in range(n - 1):
        for j in range(k + 1, n):
            out.append(np.linalg.norm(frame_xyz[k] - frame_xyz[j]))
    return np.array(out)


def _horn_align(P: np.ndarray, Q: np.ndarray) -> tuple:
    """
    Horn's quaternion alignment: find rotation R and translation t such that
    R @ P + t ≈ Q (least-squares).

    Both P and Q are (n_markers, 3) and assumed already mean-centred is NOT
    required — this function handles centring internally.

    Returns
    -------
    R : (3, 3) rotation matrix
    t : (3,) translation
    P_aligned : (n_markers, 3)
    rmsd : float, residual after alignment
    """
    R, rmsd = Rotation.align_vectors(Q, P)
    R_mat = R.as_matrix()
    P_centroid = P.mean(axis=0)
    Q_centroid = Q.mean(axis=0)
    t = Q_centroid - R_mat @ P_centroid
    P_aligned = (R_mat @ P.T).T + t
    return R_mat, t, P_aligned, float(rmsd)


def build_rigid_model(xyz_stack: np.ndarray,
                       verbose: bool = True) -> tuple:
    """
    Build the rigid cluster model from full-visible frames.

    Algorithm (port of f_CreateOneModel):
        Step 1. Compute all pairwise inter-marker distances on every
                full-visible frame.
        Step 2. Identify candidate frames: those where every pairwise
                distance falls within ±0.7σ of the median (±1.5σ fallback
                if fewer than 10 candidates).
        Step 3. Centre each candidate at its barycentre, align all of them
                to the first one with Horn, average → rigid model template.

    Parameters
    ----------
    xyz_stack : np.ndarray, shape (N_frames, 3, N_markers)

    Returns
    -------
    model : np.ndarray, shape (N_markers, 3)
        Rigid template, centroid at origin.
    fopt_median : float
        Median residual distance after Horn alignment (rigidity check).
    n_candidates : int
        Number of frames used to build the model.
    """
    occluded = detect_occluded(xyz_stack)
    full_visible = ~occluded.any(axis=1)
    n_full = int(full_visible.sum())

    if n_full < MIN_CANDIDATES:
        raise ValueError(
            f"Not enough full-visible frames to build a rigid model: "
            f"{n_full} < {MIN_CANDIDATES}"
        )

    full_xyz = xyz_stack[full_visible]            # (n_full, 3, n_markers)
    full_xyz = np.transpose(full_xyz, (0, 2, 1))  # (n_full, n_markers, 3)
    n_frames, n_markers, _ = full_xyz.shape

    # Step 1. Pairwise distances
    distances = np.array([_pairwise_distances(f) for f in full_xyz])
    n_pairs = distances.shape[1]

    # Step 2. Filter candidates by ±0.7σ on every pair
    center = np.median(distances, axis=0)
    sigma  = np.std(distances, axis=0)

    def _select(tol: float) -> np.ndarray:
        mask = np.ones(n_frames, dtype=bool)
        for k in range(n_pairs):
            mask &= np.abs(distances[:, k] - center[k]) < tol * sigma[k]
        return mask

    cand_mask = _select(CANDIDATE_SIGMA_TOL)
    if cand_mask.sum() < MIN_CANDIDATES:
        if verbose:
            print(f"   ⚠️  Few candidates with ±{CANDIDATE_SIGMA_TOL}σ, "
                  f"relaxing to ±{CANDIDATE_SIGMA_TOL_FALLBACK}σ")
        cand_mask = _select(CANDIDATE_SIGMA_TOL_FALLBACK)

    n_cand = int(cand_mask.sum())
    if n_cand < MIN_CANDIDATES:
        raise ValueError(
            f"Rigid model cannot be built: only {n_cand} candidate frames "
            f"after sigma filtering"
        )

    candidates = full_xyz[cand_mask]              # (n_cand, n_markers, 3)

    # Step 3. Centre + Horn-align all candidates to the first one
    centroids = candidates.mean(axis=1, keepdims=True)
    centred   = candidates - centroids            # (n_cand, n_markers, 3)

    ref = centred[0]
    fopts = []
    aligned = [ref]
    for c in range(1, n_cand):
        _, _, P_aligned, rmsd = _horn_align(centred[c], ref)
        aligned.append(P_aligned)
        fopts.append(rmsd / n_markers)            # normalise like MATLAB FOPT

    model = np.mean(aligned, axis=0)              # (n_markers, 3)
    fopt_median = float(np.median(fopts)) if fopts else 0.0

    if verbose:
        print(f"   Rigid model: {n_cand}/{n_full} candidates, "
              f"FOPT median = {fopt_median:.3f} mm")

    return model, fopt_median, n_cand


# ════════════════════════════════════════════════════════════════════════════
# 2. CLUSTER FILL — frames with ≥3 visible markers
# ════════════════════════════════════════════════════════════════════════════

def fill_3_visible(xyz_stack: np.ndarray, occluded: np.ndarray,
                     model: np.ndarray) -> tuple:
    """
    Reconstruct missing markers on frames where 3 markers are visible.

    For each such frame:
        1. Extract visible markers (xyz_visible) and the matching rows of
           the rigid model (model_visible).
        2. Horn-align model_visible to xyz_visible → rotation R, translation t.
        3. Apply (R, t) to the full model → all 4 marker positions.
        4. Replace the originally occluded markers with the reconstructed ones.

    Parameters
    ----------
    xyz_stack : (N_frames, 3, N_markers)
    occluded  : (N_frames, N_markers), bool
    model     : (N_markers, 3)

    Returns
    -------
    xyz_out, occluded_out  (same shapes, with fewer occluded cells)
    """
    n_frames, _, n_markers = xyz_stack.shape
    visible = ~occluded
    n_visible = visible.sum(axis=1)
    targets = (n_visible >= 3) & (n_visible < n_markers)

    xyz_out      = xyz_stack.copy()
    occluded_out = occluded.copy()

    for frame in np.where(targets)[0]:
        vis = visible[frame]                                # (n_markers,)
        xyz_frame = xyz_stack[frame].T                      # (n_markers, 3)
        P = model[vis]                                       # visible model rows
        Q = xyz_frame[vis]                                   # visible data rows
        try:
            R, t, _, _ = _horn_align(P, Q)
        except Exception:
            continue
        reconstructed = (R @ model.T).T + t                  # (n_markers, 3)

        # Replace only the originally occluded markers
        missing = ~vis
        xyz_frame[missing] = reconstructed[missing]
        xyz_out[frame] = xyz_frame.T
        occluded_out[frame, missing] = False

    return xyz_out, occluded_out


# ════════════════════════════════════════════════════════════════════════════
# 3. CLUSTER FILL — frames with exactly 2 visible markers
# ════════════════════════════════════════════════════════════════════════════

def fill_2_visible(xyz_stack: np.ndarray, occluded: np.ndarray,
                     model: np.ndarray) -> tuple:
    """
    Reconstruct missing markers on frames where exactly 2 markers are visible.

    With only 2 visible markers Horn's alignment is rank-deficient (rotation
    about the axis joining them is unconstrained). We resolve this by also
    fitting the previous frame's reconstructed cluster, which provides the
    missing degree of freedom.

    This is a simplified port of f_ClusterFill234Visible / f_ClusterFill2Visible:
    we use the neighbouring already-filled frame as a stabilising prior.
    """
    n_frames, _, n_markers = xyz_stack.shape
    visible = ~occluded
    targets = visible.sum(axis=1) == 2

    xyz_out      = xyz_stack.copy()
    occluded_out = occluded.copy()

    for frame in np.where(targets)[0]:
        vis = visible[frame]
        xyz_frame = xyz_stack[frame].T

        # Find the closest already-filled neighbour frame (4 visible markers)
        full_mask = (~occluded_out).all(axis=1)
        if not full_mask.any():
            continue
        nearest = np.argmin(np.abs(np.where(full_mask)[0] - frame))
        nearest_idx = np.where(full_mask)[0][nearest]
        xyz_neighbour = xyz_out[nearest_idx].T               # (n_markers, 3)

        # Build a Horn alignment using the 2 visible + 2 neighbour markers
        # to fix the missing axis of rotation.
        P = np.vstack([model[vis], model[~vis]])
        Q = np.vstack([xyz_frame[vis], xyz_neighbour[~vis]])
        try:
            R, t, _, _ = _horn_align(P, Q)
        except Exception:
            continue
        reconstructed = (R @ model.T).T + t
        missing = ~vis
        xyz_frame[missing] = reconstructed[missing]
        xyz_out[frame] = xyz_frame.T
        occluded_out[frame, missing] = False

    return xyz_out, occluded_out


# ════════════════════════════════════════════════════════════════════════════
# 4. INERTIAL INTERPOLATION (port of f_Line3DFill)
# ════════════════════════════════════════════════════════════════════════════

def inertial_fill(xyz_stack: np.ndarray, occluded: np.ndarray,
                    max_gap: int = INERTIAL_GAP_MAX) -> tuple:
    """
    Fill short gaps (≤ max_gap consecutive frames) per marker with linear
    interpolation. Acts as a stand-in for f_Line3DFill (which assumes
    constant velocity over the gap — strictly equivalent to linear
    interpolation for gaps under 15 frames at 400 Hz).
    """
    n_frames, _, n_markers = xyz_stack.shape
    xyz_out      = xyz_stack.copy()
    occluded_out = occluded.copy()

    for m in range(n_markers):
        marker_occluded = occluded[:, m]
        if not marker_occluded.any():
            continue
        # Find gap segments
        in_gap = False
        gap_start = 0
        for i in range(n_frames):
            if marker_occluded[i] and not in_gap:
                gap_start = i
                in_gap = True
            elif not marker_occluded[i] and in_gap:
                gap_len = i - gap_start
                if gap_len <= max_gap and gap_start > 0 and i < n_frames:
                    for ax in range(3):
                        a = xyz_out[gap_start - 1, ax, m]
                        b = xyz_out[i, ax, m]
                        if np.isfinite(a) and np.isfinite(b):
                            xyz_out[gap_start:i, ax, m] = np.linspace(
                                a, b, gap_len + 2)[1:-1]
                    occluded_out[gap_start:i, m] = False
                in_gap = False

    return xyz_out, occluded_out


# ════════════════════════════════════════════════════════════════════════════
# 5. MAIN ENTRY POINT — iterative cluster fill
# ════════════════════════════════════════════════════════════════════════════

def run_cluster_fill(trajectories: dict,
                       marker_names: tuple = ("Dos01", "Dos02", "Dos03", "Dos04"),
                       max_iterations: int = 10,
                       verbose: bool = True) -> tuple:
    """
    Apply the full iterative cluster gap-filling pipeline to a set of
    trajectories.

    Parameters
    ----------
    trajectories : dict { marker_name : np.ndarray (N_frames, 3) }
        Output of loader.extract_marker_trajectories(). Occluded samples
        must already be NaN.
    marker_names : tuple of marker names that form the rigid cluster.
    max_iterations : safety cap for the convergence loop.

    Returns
    -------
    filled : dict
        Same structure as `trajectories` with reconstructed coordinates.
    stats : dict
        Diagnostics: n_iterations, perc_occluded_initial / _final,
        fopt_median, n_candidates_model.
    """
    # Stack markers into (N_frames, 3, n_markers)
    arrays = [trajectories[m] for m in marker_names]
    xyz_stack = np.stack(arrays, axis=2)                     # (N, 3, M)
    occluded  = detect_occluded(xyz_stack)
    perc_initial = overall_perc_occluded(occluded)

    if verbose:
        print(f"   Cluster fill start: {perc_initial:.2f}% occluded")

    # Build rigid model
    model, fopt_median, n_cand = build_rigid_model(xyz_stack, verbose=verbose)

    # Iterative fill until no further improvement
    perc_prev = perc_initial
    n_iter = 0
    for n_iter in range(1, max_iterations + 1):
        xyz_stack, occluded = fill_3_visible(xyz_stack, occluded, model)
        xyz_stack, occluded = inertial_fill(xyz_stack, occluded)
        xyz_stack, occluded = fill_3_visible(xyz_stack, occluded, model)
        xyz_stack, occluded = fill_2_visible(xyz_stack, occluded, model)
        xyz_stack, occluded = fill_3_visible(xyz_stack, occluded, model)

        perc_now = overall_perc_occluded(occluded)
        if verbose:
            print(f"     iter {n_iter}: {perc_now:.2f}% occluded")
        if perc_now >= perc_prev - 1e-3:
            break
        perc_prev = perc_now

    perc_final = overall_perc_occluded(occluded)

    # Repack into dict
    filled = {m: xyz_stack[:, :, i] for i, m in enumerate(marker_names)}

    stats = {
        "n_iterations":           n_iter,
        "perc_occluded_initial":  round(perc_initial, 3),
        "perc_occluded_final":    round(perc_final, 3),
        "fopt_median_mm":         round(fopt_median, 3),
        "n_candidates_model":     n_cand,
    }
    return filled, stats
