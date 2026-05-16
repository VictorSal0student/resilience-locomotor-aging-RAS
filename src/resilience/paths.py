"""
paths.py — Single source of truth for disk paths.

Reads the .env file at the project root and exposes paths as pathlib.Path
objects. Use everywhere in the code via:

    from resilience.paths import RAW_DIR, PROCESSED_DIR
"""

from __future__ import annotations
from pathlib import Path
import os
from dotenv import load_dotenv


# ────────────────────────────────────────────────────────────────────────────
# .env location: walk up from this file until a .env is found
# ────────────────────────────────────────────────────────────────────────────

def _find_env_file() -> Path:
    """Search for a .env file walking up from this script."""
    current = Path(__file__).resolve().parent
    for _ in range(5):  # max 5 levels up
        candidate = current / ".env"
        if candidate.exists():
            return candidate
        current = current.parent
    raise FileNotFoundError(
        "No .env file found. Copy .env.example to .env at the project root."
    )


_env_path = _find_env_file()
load_dotenv(_env_path)


def _get_path(key: str) -> Path:
    """Read an environment variable and convert it to a Path."""
    value = os.getenv(key)
    if value is None:
        raise KeyError(f"Variable {key} missing from .env. See .env.example.")
    return Path(value).expanduser().resolve()


# ────────────────────────────────────────────────────────────────────────────
# Exposed paths
# ────────────────────────────────────────────────────────────────────────────

PROJECT_ROOT      = _get_path("PROJECT_ROOT")
RAW_DIR           = _get_path("RAW_DIR")
PROCESSED_DIR     = _get_path("PROCESSED_DIR")
BEATMOVE_DIR      = _get_path("BEATMOVE_DIR")
EXTERNAL_DIR      = _get_path("EXTERNAL_DIR")
OUTPUTS_DIR       = _get_path("OUTPUTS_DIR")

FIGURES_DIR       = OUTPUTS_DIR / "figures"
TABLES_DIR        = OUTPUTS_DIR / "tables"
REPORTS_DIR       = OUTPUTS_DIR / "reports"


# ────────────────────────────────────────────────────────────────────────────
# Path-building helpers (no hardcoded paths anywhere else)
# ────────────────────────────────────────────────────────────────────────────
#
# Naming convention for raw .mat and processed .npz files:
#     {participant}_{index:02d}_{condition}.{ext}
# Examples:
#     001CrMa_01_silence.mat
#     001CrMa_02_tempo_random.mat
#     001CrMa_03_beatmove_adaptatif.mat
#
# The index reflects the acquisition order in the Odin session and varies
# with the participant randomisation. The functions below resolve it
# automatically via glob, so callers do not need to know it.
# ────────────────────────────────────────────────────────────────────────────


def _find_unique_match(folder: Path, pattern: str, kind: str) -> Path:
    """
    Find a single file matching `pattern` in `folder`.

    Raises FileNotFoundError if 0 match, RuntimeError if multiple matches.
    `kind` is only used to format error messages ('raw' / 'processed').
    """
    if not folder.exists():
        raise FileNotFoundError(f"{kind} folder not found: {folder}")
    matches = sorted(folder.glob(pattern))
    if len(matches) == 0:
        raise FileNotFoundError(
            f"No {kind} file matches '{pattern}' in {folder}"
        )
    if len(matches) > 1:
        names = "\n    ".join(m.name for m in matches)
        raise RuntimeError(
            f"Multiple {kind} files match '{pattern}' in {folder}:\n"
            f"    {names}\n"
            f"→ Fix manually (one expected per (participant, condition))."
        )
    return matches[0]


def raw_file(participant: str, condition: str) -> Path:
    """
    Path to the raw .mat for (participant, condition).

    The acquisition index is recovered automatically via glob.

    Convention: RAW_DIR/{participant}/{participant}_{NN}_{condition}.mat

    Example: raw_file('001CrMa', 'silence')
             → RAW_DIR/001CrMa/001CrMa_01_silence.mat
    """
    folder = RAW_DIR / participant
    pattern = f"{participant}_*_{condition}.mat"
    return _find_unique_match(folder, pattern, kind="raw")


def processed_file(participant: str, condition: str) -> Path:
    """
    Path to the preprocessed .npz for (participant, condition).

    Read mode: index is recovered via glob.

    Convention: PROCESSED_DIR/{participant}/{participant}_{NN}_{condition}.npz
    """
    folder = PROCESSED_DIR / participant
    pattern = f"{participant}_*_{condition}.npz"
    return _find_unique_match(folder, pattern, kind="processed")


def processed_file_for_writing(participant: str, condition: str) -> Path:
    """
    Build the output .npz path from the matching raw .mat.

    The .npz keeps the exact stem of the .mat (same Odin index), which
    guarantees raw ↔ processed traceability.

    Example: processed_file_for_writing('001CrMa', 'silence')
             → PROCESSED_DIR/001CrMa/001CrMa_01_silence.npz
             (if raw_file = .../001CrMa_01_silence.mat)
    """
    raw = raw_file(participant, condition)
    npz_name = raw.stem + ".npz"
    return PROCESSED_DIR / participant / npz_name


def beatmove_dir(participant: str, condition: str) -> Path:
    """beatMove folder for a given trial."""
    return BEATMOVE_DIR / participant / condition


def ensure_dir(path: Path) -> Path:
    """Create a folder (and parents) if missing. Return the path."""
    path.mkdir(parents=True, exist_ok=True)
    return path


# ────────────────────────────────────────────────────────────────────────────
# Sanity check on load (warning only, no error)
# ────────────────────────────────────────────────────────────────────────────

def check_dirs(verbose: bool = False) -> dict:
    """Check that all main directories exist. Return a report dict."""
    dirs = {
        "PROJECT_ROOT":    PROJECT_ROOT,
        "RAW_DIR":         RAW_DIR,
        "PROCESSED_DIR":   PROCESSED_DIR,
        "BEATMOVE_DIR":    BEATMOVE_DIR,
        "EXTERNAL_DIR":    EXTERNAL_DIR,
        "OUTPUTS_DIR":     OUTPUTS_DIR,
    }
    report = {name: p.exists() for name, p in dirs.items()}
    if verbose:
        for name, exists in report.items():
            flag = "✅" if exists else "❌"
            print(f"  {flag} {name:18s} → {dirs[name]}")
    return report
