"""
plots.py — Standard plot helpers for the pipeline.
"""

from __future__ import annotations
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from resilience import config


# ════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ════════════════════════════════════════════════════════════════════════════

COND_COLORS = {
    "silence":            "steelblue",
    "tempo_random":       "darkorange",
    "beatmove_adaptatif": "forestgreen",
}

COND_ORDER = ["silence", "tempo_random", "beatmove_adaptatif"]

METRICS_INFO = [
    ("z_var",            "Z-score variance",                0),
    ("z_radius",         "Z-score radius",                  0),
    ("ratio_var",        "Ratio var (pert / baseline)",     1),
    ("ratio_radius",     "Ratio radius (pert / baseline)",  1),
    ("delta_cadence_hz", "Delta cadence (Hz)",              0),
]


# ════════════════════════════════════════════════════════════════════════════
# UTILS
# ════════════════════════════════════════════════════════════════════════════

def save_fig(fig: plt.Figure, output_path: str | Path, dpi: int = 150) -> None:
    """Save figure to disk, creating parent directories as needed."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    print(f"   ✅ Figure saved: {output_path}")


def _group_curves_by_participant(curves: dict) -> dict:
    by_p = {}
    for (code, cond), c in curves.items():
        by_p.setdefault(code, {})[cond] = c
    return by_p


# ════════════════════════════════════════════════════════════════════════════
# MARKER VISIBILITY
# ════════════════════════════════════════════════════════════════════════════

def plot_marker_visibility(visibility_summary: dict, title: str = "",
                             figsize: tuple = (10, 5)) -> plt.Figure:
    """Horizontal bar chart of per-marker visibility (% non-NaN frames)."""
    fig, ax = plt.subplots(figsize=figsize)
    per_marker = visibility_summary["per_marker"]
    colors_group = {"dos": "#2196F3", "pied_G": "#FF9800", "pied_D": "#E91E63"}

    bar_colors, names, values = [], [], []
    for group, markers in config.MARKERS.items():
        for m in markers:
            if m in per_marker:
                names.append(m)
                values.append(per_marker[m])
                bar_colors.append(colors_group[group])

    ax.barh(names, values, color=bar_colors, edgecolor="black", linewidth=0.5)
    ax.axvline(70, color="gray", linestyle="--", linewidth=1, alpha=0.6, label="70% threshold")
    ax.set_xlim(0, 100)
    ax.set_xlabel("Visibility (%)")
    ax.set_title(title or "Marker visibility", fontweight="bold")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3, axis="x")
    plt.tight_layout()
    return fig


# ════════════════════════════════════════════════════════════════════════════
# DOS Z TRAJECTORIES
# ════════════════════════════════════════════════════════════════════════════

def plot_dos_trajectories(trajectories: dict, participant_id: str, condition: str,
                            fs: int | None = None,
                            perturbation_t: float = 240,
                            trim_t: float | None = None) -> plt.Figure | None:
    """Raw Z trajectories of the 4 back markers, with trim and perturbation markers."""
    if fs is None:     fs     = config.FS_RAW
    if trim_t is None: trim_t = config.TRIM_START_S

    dos_markers = [m for m in config.MARKERS["dos"] if m in trajectories]
    if not dos_markers:
        print("⚠️  No back marker found.")
        return None

    n_frames = trajectories[dos_markers[0]].shape[0]
    t = np.arange(n_frames) / fs

    fig, axes = plt.subplots(len(dos_markers), 1,
                              figsize=(14, 2.5 * len(dos_markers)), sharex=True)
    if len(dos_markers) == 1:
        axes = [axes]

    palette = ["#2196F3", "#4CAF50", "#FF9800", "#E91E63"]
    for ax, marker, color in zip(axes, dos_markers, palette):
        z = trajectories[marker][:, 2]
        ax.plot(t, z, color=color, linewidth=0.5, label=marker)
        ax.axvline(trim_t, color="gray", linestyle="--", linewidth=1,
                   label=f"Trim {trim_t}s")
        ax.axvline(perturbation_t, color="red", linestyle="--", linewidth=1,
                   label=f"Pert. ~{perturbation_t}s")
        ax.set_ylabel("Z (mm)", fontsize=9)
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Time (s)", fontsize=10)
    fig.suptitle(f"{participant_id} — {condition} | Back markers (Z axis)",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    return fig


# ════════════════════════════════════════════════════════════════════════════
# TDE — AMI / FNN / DISTRIBUTIONS
# ════════════════════════════════════════════════════════════════════════════

def plot_ami_curves(curves: dict,
                      save_to: str | Path | None = None) -> plt.Figure:
    """AMI vs lag per participant (3 conditions overlaid)."""
    by_p   = _group_curves_by_participant(curves)
    n_p    = len(by_p)
    n_cols = 3
    n_rows = (n_p + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 3 * n_rows))
    for i, (code, c_by_cond) in enumerate(sorted(by_p.items())):
        ax = axes.flat[i]
        for cond, c in c_by_cond.items():
            ax.plot(range(1, len(c["ami"]) + 1), c["ami"],
                    color=COND_COLORS[cond], linewidth=1, alpha=0.8,
                    label=f"{cond[:3]} τ={c['tau']}")
            ax.axvline(c["tau"], color=COND_COLORS[cond], linestyle=":",
                       linewidth=0.8, alpha=0.5)
        ax.set_title(code, fontsize=9)
        ax.set_xlabel("Lag", fontsize=8)
        ax.set_ylabel("AMI", fontsize=8)
        ax.legend(fontsize=6, loc="upper right")
        ax.grid(alpha=0.3)
        ax.tick_params(labelsize=7)
    for j in range(i + 1, n_rows * n_cols):
        axes.flat[j].axis("off")

    plt.suptitle("AMI vs lag (3 conditions per participant) — τ = first local minimum",
                 fontweight="bold")
    plt.tight_layout()
    if save_to is not None:
        save_fig(fig, save_to)
    return fig


def plot_fnn_curves(curves: dict, fnn_threshold: float = 10.0,
                      save_to: str | Path | None = None) -> plt.Figure:
    """FNN rate vs embedding dim per participant (3 conditions overlaid, log y)."""
    by_p   = _group_curves_by_participant(curves)
    n_p    = len(by_p)
    n_cols = 3
    n_rows = (n_p + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 3 * n_rows))
    for i, (code, c_by_cond) in enumerate(sorted(by_p.items())):
        ax = axes.flat[i]
        for cond, c in c_by_cond.items():
            ax.plot(range(1, len(c["fnn"]) + 1), c["fnn"],
                    color=COND_COLORS[cond], linewidth=1, alpha=0.8,
                    marker="o", markersize=3,
                    label=f"{cond[:3]} m={c['m']}")
            ax.axvline(c["m"], color=COND_COLORS[cond], linestyle=":",
                       linewidth=0.8, alpha=0.5)
        ax.axhline(fnn_threshold, color="red", linestyle="--",
                   linewidth=0.6, alpha=0.4, label=f"{fnn_threshold}%")
        ax.set_title(code, fontsize=9)
        ax.set_xlabel("Embedding dim", fontsize=8)
        ax.set_ylabel("FNN (%)", fontsize=8)
        ax.set_yscale("log")
        ax.legend(fontsize=6, loc="upper right")
        ax.grid(alpha=0.3, which="both")
        ax.tick_params(labelsize=7)
    for j in range(i + 1, n_rows * n_cols):
        axes.flat[j].axis("off")

    plt.suptitle(f"FNN rate vs dim (3 conditions per participant) — m = first dim below {fnn_threshold}%",
                 fontweight="bold")
    plt.tight_layout()
    if save_to is not None:
        save_fig(fig, save_to)
    return fig


def plot_tde_distributions(df_tde: pd.DataFrame,
                              save_to: str | Path | None = None) -> plt.Figure:
    """Global histograms of τ and m across all trials."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].hist(df_tde["tau"],
                 bins=range(df_tde["tau"].min(), df_tde["tau"].max() + 2),
                 color="steelblue", edgecolor="black", alpha=0.7)
    axes[0].axvline(df_tde["tau"].median(), color="red", linestyle="--",
                    label=f"median = {df_tde['tau'].median():.0f}")
    axes[0].set_xlabel("τ (samples)")
    axes[0].set_ylabel("Trial count")
    axes[0].set_title(f"τ distribution across {len(df_tde)} trials")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].hist(df_tde["m"],
                 bins=range(df_tde["m"].min(), df_tde["m"].max() + 2),
                 color="darkorange", edgecolor="black", alpha=0.7)
    axes[1].axvline(df_tde["m"].median(), color="red", linestyle="--",
                    label=f"median = {df_tde['m'].median():.0f}")
    axes[1].set_xlabel("m (embedding dim)")
    axes[1].set_ylabel("Trial count")
    axes[1].set_title(f"m distribution across {len(df_tde)} trials")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    if save_to is not None:
        save_fig(fig, save_to)
    return fig


# ════════════════════════════════════════════════════════════════════════════
# RESILIENCE
# ════════════════════════════════════════════════════════════════════════════

def plot_resilience_boxplots(df_res: pd.DataFrame,
                               save_to: str | Path | None = None) -> plt.Figure:
    """Boxplots per condition for the 5 resilience metrics + individual scatter."""
    fig, axes = plt.subplots(1, 5, figsize=(20, 5))

    for ax, (metric, label, ref_line) in zip(axes, METRICS_INFO):
        data_by_cond = [df_res[df_res["condition"] == c][metric].dropna().values
                          for c in COND_ORDER]
        bp = ax.boxplot(data_by_cond, labels=[c[:3] for c in COND_ORDER],
                          patch_artist=True, widths=0.5)
        for patch, c in zip(bp["boxes"], COND_ORDER):
            patch.set_facecolor(COND_COLORS[c])
            patch.set_alpha(0.4)

        for i, c in enumerate(COND_ORDER):
            sub = df_res[df_res["condition"] == c]
            x = np.random.normal(i + 1, 0.05, len(sub))
            ax.scatter(x, sub[metric], s=30, alpha=0.7,
                         color=COND_COLORS[c], edgecolors="black", linewidths=0.5)

        ax.axhline(ref_line, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
        ax.set_title(label, fontsize=11, fontweight="bold")
        ax.set_ylabel(label, fontsize=10)
        ax.grid(alpha=0.3)

    n_p = df_res["participant"].nunique()
    plt.suptitle(f"Resilience metrics by condition (N={n_p} participants, {len(df_res)} trials)",
                 fontweight="bold", fontsize=12, y=1.02)
    plt.tight_layout()
    if save_to is not None:
        save_fig(fig, save_to)
    return fig


def plot_resilience_individual_trajectories(df_res: pd.DataFrame,
                                                save_to: str | Path | None = None) -> plt.Figure:
    """One line per participant across the 3 conditions, with the median overlaid."""
    fig, axes = plt.subplots(1, 5, figsize=(20, 5))

    for ax, (metric, label, ref_line) in zip(axes, METRICS_INFO):
        pivot = df_res.pivot(index="participant", columns="condition", values=metric)
        pivot = pivot[COND_ORDER]

        for code in pivot.index:
            vals = pivot.loc[code].values
            ax.plot(range(len(COND_ORDER)), vals, "-o",
                      alpha=0.5, linewidth=1, markersize=4)
            ax.annotate(code[:3], (len(COND_ORDER) - 1, vals[-1]),
                          fontsize=6, alpha=0.7, xytext=(2, 0),
                          textcoords="offset points")

        med = pivot.median()
        ax.plot(range(len(COND_ORDER)), med.values, "k-o",
                  linewidth=2.5, markersize=8, label="median", zorder=5)

        ax.axhline(ref_line, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
        ax.set_xticks(range(len(COND_ORDER)))
        ax.set_xticklabels([c[:3] for c in COND_ORDER])
        ax.set_title(label, fontsize=11, fontweight="bold")
        ax.set_ylabel(label, fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    plt.suptitle("Individual trajectories (1 line = 1 participant)",
                 fontweight="bold", y=1.02)
    plt.tight_layout()
    if save_to is not None:
        save_fig(fig, save_to)
    return fig
