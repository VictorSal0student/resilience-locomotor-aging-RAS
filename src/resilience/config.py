"""
config.py — Scientific constants of the project.

⚠️  THIS FILE CONTAINS NO DISK PATHS.
    Paths live in paths.py (which reads .env).

Contains:
    - CODAMOTION parameters (sampling rate, duration, markers)
    - Pipeline parameters (filters, EMD, TDE, detection)
    - Experimental conditions and beatMove codes
    - Sand bed coordinates (measured 07/05/2026)
"""


# ════════════════════════════════════════════════════════════════════════════
# CODAMOTION PARAMETERS
# ════════════════════════════════════════════════════════════════════════════

FS_RAW   = 400              # Hz, CODAMOTION acquisition rate
FS_CLEAN = 100              # Hz, after decimation

SESSION_DURATION_S = 480    # 8 minutes per trial
TRIM_START_S       = 60     # first minute systematically dropped
                            # (beatMove setup + phone in pocket)


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

# beatMove codes (app config) → condition.
# See recap_marqueurs: 22/178/1195=silence, 23/888/9999=adaptatif, 22/180/2220=random
BEATMOVE_CODE_TO_CONDITION = {
    "1195": "silence",
    "2220": "tempo_random",
    "9999": "beatmove_adaptatif",
}


# ════════════════════════════════════════════════════════════════════════════
# PREPROCESSING — EMD
# ════════════════════════════════════════════════════════════════════════════
#
# IMF selection strategy for walking signal reconstruction:
#
# 1. AUTO method (default): selects IMFs whose central frequency
#    (FFT/Welch dominant peak) falls within WALKING_BAND_HZ.
#    Adaptive per trial, justified by the cadence/stride observed in 65+
#    (Tudor-Locke et al., 2009; CADENCE-Adults 2021).
#
# 2. HEURISTIC method (fallback): fixed indices (EMD_IMF_INDICES).
#    Used if auto selection returns 0 IMF (pathological case).
#
# Band [0.5–2.5 Hz]:
#   - 0.5 Hz: stride cycle (2 steps) for slow walkers
#   - 2.5 Hz: margin above max observed cadence in fast 65+ walkers

WALKING_BAND_HZ      = (0.5, 2.5)                    # auto IMF selection band
EMD_IMF_INDICES      = (3, 4, 5, 6, 7, 8)            # heuristic fallback (Antoine's choice)


# ════════════════════════════════════════════════════════════════════════════
# PREPROCESSING — BUTTERWORTH + DECIMATION
# ════════════════════════════════════════════════════════════════════════════

BUTTER_CUTOFF_HZ = 5.0
BUTTER_ORDER     = 4

DECIMATE_FACTOR = 4         # 400 Hz / 4 = 100 Hz

# Default source in the .mat file
DEFAULT_SOURCE = "Marker"   # 'Marker' (raw) or 'MFilter' (Odin-filtered, not recommended)
DEFAULT_AXIS   = "Z"


# ════════════════════════════════════════════════════════════════════════════
# TDE
# ════════════════════════════════════════════════════════════════════════════

TDE_MAX_LAG       = 100
TDE_MAX_DIM       = 10
TDE_FNN_THRESHOLD = 1.0      # %
TDE_FNN_R         = 15.0     # ratio Kennel et al. 1992
TDE_VIZ_DIM       = 3


# ════════════════════════════════════════════════════════════════════════════
# SAND DETECTION — PCA METHOD (legacy)
# ════════════════════════════════════════════════════════════════════════════

DETECT_PCA_PERCENTILE_LOW   = 5
DETECT_PCA_PERCENTILE_HIGH  = 95
DETECT_PCA_MARGIN_S_MM      = 200
DETECT_PCA_MARGIN_D_MM      = 200
DETECT_MIN_DURATION_S       = 1.5


# ════════════════════════════════════════════════════════════════════════════
# SAND DETECTION — ABSOLUTE COORDINATES (07/05/2026)
# ════════════════════════════════════════════════════════════════════════════
#
# Sand bed corners in CODAMOTION coordinates (mm), measured directly on
# 07/05/2026 with 4 markers (Marker04 to Marker10).
#
# Convention:
#   A = front-left   (left exit)
#   B = front-right  (right exit)
#   C = rear-right   (right entry)
#   D = rear-left    (left entry)
#
# Sanity check:
#   A-D ≈ 4413 mm  |  B-C ≈ 4423 mm  (measured length 4.4 m)
#   A-B ≈ 1006 mm  |  C-D ≈ 1012 mm  (width ~1.0 m)
#   Diagonals A-C and B-D nearly equal → rectangle.

BAC_SAND_CORNERS_MM = {
    'A': (1070.7,   978.9),    # front-left
    'B': ( 283.9,  1606.6),    # front-right
    'C': (-2452.0, -1869.2),   # rear-right
    'D': (-1654.5, -2492.2),   # rear-left
}

# Tolerance margin around the rectangle (the sacrum is not exactly above
# the centre of the bed but slightly above ground level).
BAC_SAND_MARGIN_MM = 100
