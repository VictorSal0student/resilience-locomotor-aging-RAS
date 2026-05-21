"""
config.py — Scientific constants of the project.

⚠️  THIS FILE CONTAINS NO DISK PATHS.
    Paths live in paths.py (which reads .env).

Contents
--------
- CODAMOTION acquisition parameters (sampling rate, trial duration, markers)
- Experimental conditions and beatMove configuration codes
- Preprocessing parameters (EMD walking-band selection, Butterworth, decimation)
- Time-Delay Embedding (TDE) parameters
- Sand bed and baseline zone coordinates (BBOX detection)
- Resilience dynamics parameters (Antoine Dufourneau method, adapted)
"""


# ════════════════════════════════════════════════════════════════════════════
# CODAMOTION ACQUISITION
# ════════════════════════════════════════════════════════════════════════════

FS_RAW   = 400              # Hz, CODAMOTION acquisition rate
FS_CLEAN = 100              # Hz, after decimation

SESSION_DURATION_S = 480    # 8 minutes per trial
TRIM_START_S       = 60     # first minute systematically dropped


# ════════════════════════════════════════════════════════════════════════════
# MARKERS
# ════════════════════════════════════════════════════════════════════════════

MARKERS = {
    "dos":    ["Dos01", "Dos02", "Dos03", "Dos04"],
    "pied_G": ["GLacet", "GMT1", "GMT5", "GTalon"],
    "pied_D": ["DLacet", "DMT1", "DMT5", "DTalon"],
}

ALL_MARKERS = [m for group in MARKERS.values() for m in group]


# ════════════════════════════════════════════════════════════════════════════
# EXPERIMENTAL CONDITIONS
# ════════════════════════════════════════════════════════════════════════════

CONDITIONS = ["silence", "tempo_random", "beatmove_adaptatif"]

BEATMOVE_CODE_TO_CONDITION = {
    "1195": "silence",
    "2220": "tempo_random",
    "9999": "beatmove_adaptatif",
}


# ════════════════════════════════════════════════════════════════════════════
# PREPROCESSING — EMD
# ════════════════════════════════════════════════════════════════════════════

WALKING_BAND_HZ = (0.5, 2.5)
EMD_IMF_INDICES = (3, 4, 5, 6, 7, 8)


# ════════════════════════════════════════════════════════════════════════════
# PREPROCESSING — BUTTERWORTH + DECIMATION
# ════════════════════════════════════════════════════════════════════════════

BUTTER_CUTOFF_HZ = 5.0
BUTTER_ORDER     = 4
DECIMATE_FACTOR  = 4

DEFAULT_SOURCE = "Marker"
DEFAULT_AXIS   = "Z"


# ════════════════════════════════════════════════════════════════════════════
# TIME-DELAY EMBEDDING (TDE)
# ════════════════════════════════════════════════════════════════════════════

TDE_MAX_LAG       = 100
TDE_MAX_DIM       = 10
TDE_FNN_THRESHOLD = 1.0
TDE_FNN_R         = 15.0
TDE_VIZ_DIM       = 3


# ════════════════════════════════════════════════════════════════════════════
# BBOX DETECTION
# ════════════════════════════════════════════════════════════════════════════

DETECT_MIN_DURATION_S = 1.5

BAC_SAND_CORNERS_MM = {
    'A': ( 1070.7,   978.9),
    'B': (  283.9,  1606.6),
    'C': (-2452.0, -1869.2),
    'D': (-1654.5, -2492.2),
}
BAC_SAND_MARGIN_MM = 100

BAC_BASELINE_LEFT_CORNERS_MM = {
    'A': ( 1849.7,   354.1),
    'B': ( 1068.0,   977.7),
    'C': (-1657.2, -2493.4),
    'D': ( -875.5, -3117.0),
}
BAC_BASELINE_LEFT_MARGIN_MM = 0


# ════════════════════════════════════════════════════════════════════════════
# RESILIENCE DYNAMICS (Antoine Dufourneau MATLAB port, adapted)
# ════════════════════════════════════════════════════════════════════════════
#
# Method (notebook 04b): port of the original MATLAB pipeline.
#
# Reference trajectory: pre-perturbation phase-space trajectory smoothed
# by Fourier expansion (order 8) on 360° phase angles.
#
# Stability zones (ellipsoid σ): T1 (1σ, "very stable"), T2 (2σ, "stable"),
# T3 (3σ, "unstable"), beyond T3 ("very unstable"). Scaling factor calibrated
# on pre-perturbation distance distribution (97.5% quantile).
#
# Resilience metrics per trial:
#   T_L (lag time)      : t_entry → first significant deviation
#                          (10 consecutive samples in T3 or >T3)
#   T_P (peak time)     : lag → peak XR_distances within
#                          [pert_entry, pert_exit + POST_PEAK_BUFFER_S]
#   T_S (recovery time) : peak → stable return, search starts at pert_exit
#                          (10 consecutive heel-strikes in T1/T2, tol 5 NaN)
#   P (persistent)      : mean post-recovery vs pre-perturbation
#                          (4 windows: 15, 30, 45, 60 s)

RESILIENCE_THRESHOLD_DETREND_MM = 0.01
RESILIENCE_FOURIER_ORDER        = 8
RESILIENCE_RESOLUTION_PHASE     = 360
RESILIENCE_GAUSS_SMOOTH_WIN     = 30
RESILIENCE_QUANTILE_BASELINE    = 0.975

RESILIENCE_N_STEPS_WINDOW       = 10
RESILIENCE_N_CONSECUTIVE        = 10
RESILIENCE_TOLERANCE_NAN        = 5

# Peak search window: [pert_entry, pert_exit + POST_PEAK_BUFFER_S]
# Adapted from Antoine's single-instant perturbation (5s search window).
# In our protocol the perturbation lasts ~4s (sand crossing), so the peak
# can occur anywhere from sand entry to a few seconds after sand exit.
RESILIENCE_POST_PEAK_BUFFER_S   = 5.0
RESILIENCE_POST_PEAK_VALID_S    = 7.0    # post-peak validation window

RESILIENCE_PERSISTENT_WINDOWS_S = [15.0, 30.0, 45.0, 60.0]
