"""
figures_memoire.py — Publication-ready figures for the memoire.

Each `make_figN()` function produces one figure of the manuscript.
All figures are saved to outputs/figures/ at 150 DPI.

Example participant: 012WaCh (3 full 420s trials, clean signal, 0 spikes).
"""

from __future__ import annotations
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from scipy.signal import welch

from resilience import config, participants, paths
from resilience.io import loader, writer
from resilience.processing import preprocess, detection, tde


# ════════════════════════════════════════════════════════════════════════════
# CONFIG
# ════════════════════════════════════════════════════════════════════════════

EXAMPLE_P    = "012WaCh"
EXAMPLE_COND = "silence"

COND_ORDER  = ["silence", "tempo_random", "beatmove_adaptatif"]
COND_LABELS = {"silence": "Silence",
               "tempo_random": "Tempo random",
               "beatmove_adaptatif": "BeatMove"}
COND_COLORS = {"silence": "#4C72B0",
               "tempo_random": "#DD8452",
               "beatmove_adaptatif": "#55A868"}

FIG_DIR = Path(paths.OUTPUTS_DIR) / "figures"


def setup_style() -> None:
    """Apply publication-ready matplotlib style."""
    plt.rcParams.update({
        "font.size":       10,
        "axes.titlesize":  11,
        "axes.labelsize":  10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 8,
        "figure.dpi":      100,
        "savefig.dpi":     150,
        "savefig.bbox":    "tight",
    })
    FIG_DIR.mkdir(parents=True, exist_ok=True)


# ════════════════════════════════════════════════════════════════════════════
# HELPERS (shared by several figures)
# ════════════════════════════════════════════════════════════════════════════

def _build_sacrum_xy(npz: dict) -> np.ndarray:
    """Mean of Dos01-04 XY with linear NaN interpolation."""
    dos_xy = np.stack([npz[f"raw_Dos{i:02d}"][:, :2] for i in range(1, 5)], axis=0)
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


def _load_tde_params_one(code: str) -> tuple[int, int]:
    """Read τ_p and m_p for one participant from the notebook 03 export."""
    df = pd.read_csv(Path(paths.OUTPUTS_DIR) / "tables" / "03_tde_per_participant.csv")
    row = df[df["participant"] == code].iloc[0]
    return int(row["tau_p"]), int(row["m_p"])


# ════════════════════════════════════════════════════════════════════════════
# FIG 1 — Preprocessing pipeline (5 panels)
# ════════════════════════════════════════════════════════════════════════════

def make_fig1(code: str = EXAMPLE_P, cond: str = EXAMPLE_COND,
                zoom: tuple = (60.0, 90.0)) -> plt.Figure:
    """5-panel preprocessing pipeline: raw → despike → EMD → Butterworth → final."""
    raw_path = paths.raw_file(code, cond)
    data, fmt = loader.load_mat(raw_path)
    trajectories = loader.extract_marker_trajectories(data, fmt)

    trial = participants.get_trial(code, cond)
    n_before = next(iter(trajectories.values())).shape[0]
    i_start = int(trial.crop_start_s * config.FS_RAW) if trial.crop_start_s else 0
    i_end   = int(trial.crop_end_s   * config.FS_RAW) if trial.crop_end_s   else n_before
    trajectories = {m: xyz[i_start:i_end] for m, xyz in trajectories.items()}

    out = preprocess.run(trajectories, axis="Z", apply_despike=True,
                          despike_k=8.0, verbose=False)

    t_raw   = np.arange(len(out["sacrum_axis_raw"])) / config.FS_RAW
    t_clean = out["time_final"]
    z0, z1  = zoom
    mask_raw   = (t_raw   >= z0) & (t_raw   <= z1)
    mask_clean = (t_clean >= z0) & (t_clean <= z1)

    fig, axes = plt.subplots(5, 1, figsize=(10, 11), sharex=True)
    n_spikes = out["despike_stats"]["n_spikes"]

    panels = [
        (out["sacrum_axis_raw"], t_raw,   mask_raw,   "#999", 0.5,
            "(a) Raw sacrum Z — Dos01-04 barycentre, centred"),
        (out["sacrum_despiked"], t_raw,   mask_raw,   "#777", 0.5,
            f"(b) After MAD despike (k=8, {n_spikes} spikes interpolated)"),
        (out["sacrum_emd"],      t_raw,   mask_raw,   "#555", 0.7,
            "(c) After adaptive EMD reconstruction — IMFs selected in [0.5–2.5 Hz]"),
        (out["sacrum_filt"],     t_raw,   mask_raw,   "#333", 0.7,
            "(d) After Butterworth low-pass 5 Hz, order 4, zero-phase"),
        (out["signal_final"],    t_clean, mask_clean, "#000", 0.7,
            "(e) Final signal after decimation 400 → 100 Hz"),
    ]
    for ax, (sig, t, msk, color, lw, title) in zip(axes, panels):
        ax.plot(t[msk], sig[msk], color=color, linewidth=lw)
        ax.set_title(title, loc="left")
        ax.set_ylabel("Z (mm)")
        ax.grid(alpha=0.3, linewidth=0.5)
        ax.set_xlim(z0, z1)
    axes[-1].set_xlabel("Time (s)")

    plt.suptitle(f"Preprocessing pipeline — {code}/{cond}, zoom {z0:.0f}–{z1:.0f}s",
                 fontweight="bold")
    plt.tight_layout()
    fig.savefig(FIG_DIR / "fig1_pipeline_preprocessing.png")
    return fig


# ════════════════════════════════════════════════════════════════════════════
# FIG 2 — Sand detection (XY, 3 conditions)
# ════════════════════════════════════════════════════════════════════════════

def make_fig2(code: str = EXAMPLE_P) -> plt.Figure:
    """Sacrum XY trajectory + sand bed + in-bbox highlight for the 3 conditions."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    C = config.BAC_SAND_CORNERS_MM

    for ax, cond in zip(axes, COND_ORDER):
        d = writer.load_processed(code, cond)
        xy = _build_sacrum_xy(d)
        det = detection.detect_sand_by_bbox(xy)
        in_bbox = det["in_bbox"]

        ax.plot(xy[:, 0], xy[:, 1], color="lightgray", linewidth=0.3,
                alpha=0.7, label="Sacrum trajectory")
        if in_bbox.any():
            ax.scatter(xy[in_bbox, 0], xy[in_bbox, 1], s=3,
                         color=COND_COLORS[cond], alpha=0.6, label="In-bbox samples")

        rect = Polygon([C["A"], C["B"], C["C"], C["D"]], closed=True, fill=False,
                         edgecolor="red", linewidth=2, label="Sand bed (4.4 × 1 m)")
        ax.add_patch(rect)

        ax.set_aspect("equal")
        ax.set_title(COND_LABELS[cond], fontweight="bold")
        ax.set_xlabel("X (mm)")
        ax.set_ylabel("Y (mm)")
        ax.legend(loc="upper right", fontsize=7)
        ax.grid(alpha=0.3)

    plt.suptitle(f"Sand bed detection (bbox) — {code}", fontweight="bold")
    plt.tight_layout()
    fig.savefig(FIG_DIR / "fig2_sand_detection.png")
    return fig


# ════════════════════════════════════════════════════════════════════════════
# FIG 3 — TDE parameters (AMI + FNN)
# ════════════════════════════════════════════════════════════════════════════

def make_fig3(code: str = EXAMPLE_P, cond: str = EXAMPLE_COND) -> plt.Figure:
    """AMI and FNN curves for one trial, showing how τ and m are picked."""
    d = writer.load_processed(code, cond)
    sig = d["signal_final"].astype(float)

    tau, ami_values = tde.compute_ami(sig, max_lag=100)
    m, fnn_rates    = tde.compute_fnn(sig, max_dim=10, tau=tau,
                                        threshold=10.0, r_threshold=15.0)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    axes[0].plot(range(1, len(ami_values) + 1), ami_values,
                 color="#4C72B0", linewidth=1.5)
    axes[0].axvline(tau, color="red", linestyle="--", linewidth=1,
                    label=f"τ = {tau} samples ({tau/100*1000:.0f} ms)")
    axes[0].set_xlabel("Lag (samples)")
    axes[0].set_ylabel("Average Mutual Information")
    axes[0].set_title("(a) τ selection — first local minimum of AMI", loc="left")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].semilogy(range(1, len(fnn_rates) + 1), fnn_rates, "-o",
                       color="#DD8452", linewidth=1.5, markersize=6)
    axes[1].axhline(10, color="red", linestyle="--", linewidth=1, alpha=0.5,
                     label="10% threshold")
    axes[1].axvline(m, color="red", linestyle="--", linewidth=1, label=f"m = {m}")
    axes[1].set_xlabel("Embedding dimension")
    axes[1].set_ylabel("FNN rate (%)")
    axes[1].set_title("(b) m selection — first dim below 10% FNN", loc="left")
    axes[1].legend()
    axes[1].grid(alpha=0.3, which="both")

    plt.suptitle(f"TDE parameters — {code}/{cond}", fontweight="bold")
    plt.tight_layout()
    fig.savefig(FIG_DIR / "fig3_tde_parameters.png")
    return fig


# ════════════════════════════════════════════════════════════════════════════
# FIG 4 — Phase-space torus (3 conditions)
# ════════════════════════════════════════════════════════════════════════════

def _ellipsoid_2sigma(points: np.ndarray, n_pts: int = 30) -> tuple:
    """2σ ellipsoid via covariance eigen-decomposition."""
    centroid = points.mean(axis=0)
    cov      = np.cov(points.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    radii    = 2 * np.sqrt(eigvals)

    u = np.linspace(0, 2 * np.pi, n_pts)
    v = np.linspace(0, np.pi, n_pts)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones_like(u), np.cos(v))

    pts_unit    = np.stack([x.ravel(), y.ravel(), z.ravel()], axis=1)
    pts_scaled  = pts_unit * radii
    pts_rotated = pts_scaled @ eigvecs.T
    pts_final   = pts_rotated + centroid
    return (pts_final[:, 0].reshape(n_pts, n_pts),
            pts_final[:, 1].reshape(n_pts, n_pts),
            pts_final[:, 2].reshape(n_pts, n_pts),
            centroid)


def make_fig4(code: str = EXAMPLE_P) -> plt.Figure:
    """3D phase-space trajectory (baseline + perturbation + 2σ ellipsoid) per condition."""
    tau_p, m_p = _load_tde_params_one(code)
    fig = plt.figure(figsize=(18, 6))

    for i, cond in enumerate(COND_ORDER):
        d   = writer.load_processed(code, cond)
        sig = d["signal_final"].astype(float)
        t   = d["time_final"].astype(float)

        sac_xy = _build_sacrum_xy(d)
        det = detection.detect_sand_by_bbox(sac_xy)
        pert_entry, pert_exit = det["entry_t"], det["exit_t"]

        XR = tde.phase_space_reconstruction(sig, tau=tau_p, dim=m_p)
        pad = (m_p - 1) * tau_p
        t_xr = t[pad:pad + XR.shape[0]]
        pert_mask = (t_xr >= pert_entry) & (t_xr <= pert_exit)

        baseline_3d = XR[~pert_mask, :3]
        ex, ey, ez, centroid = _ellipsoid_2sigma(baseline_3d)

        ax = fig.add_subplot(1, 3, i + 1, projection="3d")
        ax.plot(XR[~pert_mask, 0], XR[~pert_mask, 1], XR[~pert_mask, 2],
                color="gray", linewidth=0.2, alpha=0.4, label="Baseline")
        ax.plot(XR[pert_mask, 0], XR[pert_mask, 1], XR[pert_mask, 2],
                color="red", linewidth=0.8, alpha=0.9, label="Perturbation")
        ax.scatter(*centroid, color="black", s=40, zorder=5)
        ax.plot_surface(ex, ey, ez, color="orange", alpha=0.15, edgecolor="none")

        ax.set_title(COND_LABELS[cond], fontweight="bold")
        ax.set_xlabel("X(t)")
        ax.set_ylabel("X(t+τ)")
        ax.set_zlabel("X(t+2τ)")
        ax.legend(fontsize=8, loc="upper right")

    plt.suptitle(f"Phase-space reconstruction (torus) — {code}, τ={tau_p}, m={m_p}",
                 fontweight="bold")
    plt.tight_layout()
    fig.savefig(FIG_DIR / "fig4_torus_3d_simple.png")
    return fig


# ════════════════════════════════════════════════════════════════════════════
# FIG 5 — Resilience signals (sig_var + sig_radius)
# ════════════════════════════════════════════════════════════════════════════

def _compute_resilience_signals_minimal(signal_z: np.ndarray, tau: int, m: int,
                                           fs: int | None = None,
                                           win_s: float = 1.0) -> tuple:
    """Lightweight inline version (only var + radius, no cadence)."""
    if fs is None:
        fs = config.FS_CLEAN
    n   = len(signal_z)
    win = int(win_s * fs)
    half = win // 2

    sig_var = np.zeros(n)
    for i in range(n):
        i0, i1 = max(0, i - half), min(n, i + half)
        sig_var[i] = np.var(signal_z[i0:i1])

    ps = tde.phase_space_reconstruction(signal_z, tau=tau, dim=m)
    centroid = ps.mean(axis=0)
    radius = np.linalg.norm(ps - centroid, axis=1)
    pad = (m - 1) * tau
    sig_radius = np.concatenate([np.full(pad, np.nan), radius])
    return sig_var, sig_radius


def make_fig5(code: str = EXAMPLE_P) -> plt.Figure:
    """Time-series of sig_var and sig_radius around the perturbation, 3 conditions."""
    tau_p, m_p = _load_tde_params_one(code)
    fig, axes = plt.subplots(3, 2, figsize=(14, 8), sharex="col")

    for row, cond in enumerate(COND_ORDER):
        d   = writer.load_processed(code, cond)
        sig = d["signal_final"].astype(float)
        t   = d["time_final"].astype(float)

        sac_xy = _build_sacrum_xy(d)
        det = detection.detect_sand_by_bbox(sac_xy)
        pert_entry, pert_exit = det["entry_t"], det["exit_t"]

        sig_var, sig_radius = _compute_resilience_signals_minimal(sig, tau=tau_p, m=m_p)

        z0  = max(0, pert_entry - 30)
        z1  = min(t[-1], pert_exit + 60)
        msk = (t >= z0) & (t <= z1)

        axes[row, 0].plot(t[msk], sig_var[msk], color=COND_COLORS[cond], linewidth=0.8)
        axes[row, 0].axvspan(pert_entry, pert_exit, color="red", alpha=0.15,
                              label="Perturbation")
        axes[row, 0].set_ylabel(f"{COND_LABELS[cond]}\nZ variance (mm²)")
        axes[row, 0].grid(alpha=0.3)
        if row == 0:
            axes[row, 0].legend(loc="upper right")
            axes[row, 0].set_title("Rolling variance (1 s window)", loc="left")

        axes[row, 1].plot(t[msk], sig_radius[msk], color=COND_COLORS[cond], linewidth=0.8)
        axes[row, 1].axvspan(pert_entry, pert_exit, color="red", alpha=0.15)
        axes[row, 1].set_ylabel("Phase-space radius (mm)")
        axes[row, 1].grid(alpha=0.3)
        if row == 0:
            axes[row, 1].set_title("Distance to TDE centroid", loc="left")

    axes[2, 0].set_xlabel("Time (s)")
    axes[2, 1].set_xlabel("Time (s)")

    plt.suptitle(f"Resilience signals around the perturbation — {code}",
                 fontweight="bold")
    plt.tight_layout()
    fig.savefig(FIG_DIR / "fig5_resilience_signals.png")
    return fig


# ════════════════════════════════════════════════════════════════════════════
# FIG 6 — Stats (ratio_var + ratio_radius)
# ════════════════════════════════════════════════════════════════════════════

def make_fig6() -> plt.Figure:
    """Clean version of the stats figure focusing on the two significant metrics."""
    tables_dir = Path(paths.OUTPUTS_DIR) / "tables"
    df_res   = pd.read_csv(tables_dir / "04_resilience_metrics.csv")
    df_fried = pd.read_csv(tables_dir / "05_friedman.csv")
    df_wilc  = pd.read_csv(tables_dir / "05_wilcoxon_posthoc.csv")

    metrics = ["ratio_var", "ratio_radius"]
    labels  = {"ratio_var":    "Ratio variance perturbation / baseline",
               "ratio_radius": "Ratio phase-space radius perturbation / baseline"}

    fig, axes = plt.subplots(1, 2, figsize=(11, 6))

    for ax, metric in zip(axes, metrics):
        pv = (df_res
              .pivot(index="participant", columns="condition", values=metric)
              .reindex(columns=COND_ORDER)
              .dropna())
        data_by_cond = [pv[c].values for c in COND_ORDER]

        bp = ax.boxplot(data_by_cond, labels=[COND_LABELS[c] for c in COND_ORDER],
                          patch_artist=True, widths=0.55,
                          medianprops={"color": "black", "linewidth": 1.5})
        for patch, c in zip(bp["boxes"], COND_ORDER):
            patch.set_facecolor(COND_COLORS[c])
            patch.set_alpha(0.5)

        for p_code in pv.index:
            vals = pv.loc[p_code].values
            ax.plot([1, 2, 3], vals, "-", color="gray", alpha=0.35, linewidth=0.7)
            ax.scatter([1, 2, 3], vals, s=25, alpha=0.8,
                         c=[COND_COLORS[c] for c in COND_ORDER],
                         edgecolors="black", linewidths=0.4, zorder=5)

        ax.axhline(1, color="black", linestyle="--", linewidth=0.7, alpha=0.5)

        fried_row = df_fried[df_fried["metric"] == metric].iloc[0]
        title = (f"{labels[metric]}\n"
                 f"(N={int(fried_row['N'])}, Friedman p={fried_row['p']:.3f}, "
                 f"W={fried_row['kendall_W']:.2f})")
        ax.set_title(title, fontweight="bold", fontsize=10)
        ax.set_ylabel(labels[metric])
        ax.grid(alpha=0.3, axis="y")

        # Wilcoxon annotations
        sub = df_wilc[df_wilc["metric"] == metric]
        pair_x = {"sil vs tem": (1, 2),
                  "sil vs bea": (1, 3),
                  "tem vs bea": (2, 3)}
        y_top    = max(np.concatenate(data_by_cond))
        y_range  = y_top - min(np.concatenate(data_by_cond))
        y_offset = y_range * 0.08

        j_sig = 0
        for _, row in sub.iterrows():
            if row["sig_bonf"] != "ns":
                x1, x2 = pair_x[row["pair"]]
                y = y_top + y_offset * (1 + j_sig)
                ax.plot([x1, x2], [y, y], "k-", linewidth=1.2)
                ax.text((x1 + x2) / 2, y + y_offset * 0.15,
                          f"{row['sig_bonf']} (p_a={row['p_bonf']:.3f}, r={row['r']:.2f})",
                          ha="center", fontsize=9, fontweight="bold")
                j_sig += 1

    plt.suptitle("Resilience metrics by experimental condition",
                 fontweight="bold", fontsize=12)
    plt.tight_layout()
    fig.savefig(FIG_DIR / "fig6_stats_clean.png")
    return fig


# ════════════════════════════════════════════════════════════════════════════
# FIG 7 — Heatmap participant × condition
# ════════════════════════════════════════════════════════════════════════════

def make_fig7() -> plt.Figure:
    """Individual ratio_var per participant × condition (heatmap)."""
    df_res = pd.read_csv(Path(paths.OUTPUTS_DIR) / "tables" / "04_resilience_metrics.csv")
    pv = df_res.pivot(index="participant", columns="condition",
                       values="ratio_var").reindex(columns=COND_ORDER)

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(pv.values, cmap="RdYlGn_r", aspect="auto", vmin=1.0, vmax=2.7)

    for i in range(pv.shape[0]):
        for j in range(pv.shape[1]):
            v = pv.values[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                          color="black" if v < 1.9 else "white",
                          fontsize=9, fontweight="bold")
            else:
                ax.text(j, i, "NA", ha="center", va="center",
                          color="gray", fontsize=8)

    ax.set_xticks(range(len(COND_ORDER)))
    ax.set_xticklabels([COND_LABELS[c] for c in COND_ORDER])
    ax.set_yticks(range(len(pv.index)))
    ax.set_yticklabels(pv.index)

    cbar = plt.colorbar(im, ax=ax, fraction=0.04)
    cbar.set_label("Ratio variance pert / baseline", fontsize=10)

    ax.set_title("Variance ratio per participant and condition\n"
                 "(green = resilient, red = high cost)",
                 fontweight="bold")
    plt.tight_layout()
    fig.savefig(FIG_DIR / "fig7_heatmap_ratiovar.png")
    return fig
