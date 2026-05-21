"""
participants.py — Single registry of all participants.

Single source of truth for:
    - Acquisition date
    - CRF perturbation timestamps (manual annotations)
    - Manual start/end crops (battery loss, end/start-of-trial artefacts)

The matching raw .mat file is resolved automatically by
paths.raw_file(code, condition) via glob — no need to list it here.

Adding a participant = adding ONE Participant block below.

────────────────────────────────────────────────────────────────────────
crop_start_s RULE — decision 14/05/2026
────────────────────────────────────────────────────────────────────────
crop_start_s = 60 s by default on EVERY trial.

Rationale:
    The first 60 s of each session are a silent period for ALL conditions,
    corresponding to the time needed for the beatMove app to synchronise
    with the ankle sensors and compute the target cadence (adaptive tempo).
    This zone is not representative of the experimental condition.

    → Systematically trim the first 60 s to homogenise trials across
      conditions and across participants.

If an artefact is detected AFTER 60 s, crop_start_s is increased
(e.g. 72 s for 003BrLu/silence, 120 s for 002CrPa/beatmove).
────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional


# ────────────────────────────────────────────────────────────────────────────
# DATA CLASSES
# ────────────────────────────────────────────────────────────────────────────

@dataclass
class Trial:
    """One trial = one participant × one condition."""
    crf_entry_s:  Optional[float] = None   # sand entry timestamp (CRF, in s)
    crf_exit_s:   Optional[float] = None   # sand exit timestamp  (CRF, in s)
    n_pas:        Optional[int]   = None   # number of steps in the sand
    pied:         Optional[str]   = None   # 'G' or 'D' (lead foot)
    crop_start_s: float           = 60.0   # start trim: 60 s default (see rule)
    crop_end_s:   Optional[float] = None   # if not None: trim AFTER this timestamp
    excluded:     bool            = False  # True = do not process at all
    note:         Optional[str]   = None   # free comment (reason for crop/exclusion)


@dataclass
class Participant:
    code: str                               # e.g. '001CrMa'
    date: str                               # 'YYYY-MM-DD'
    trials: dict[str, Trial] = field(default_factory=dict)


# ────────────────────────────────────────────────────────────────────────────
# REGISTRY
# ────────────────────────────────────────────────────────────────────────────
# crop_start_s = 60 by default (see rule above).
# Only specified explicitly if > 60 (artefact detected later in the signal).
# ────────────────────────────────────────────────────────────────────────────

PARTICIPANTS: dict[str, Participant] = {

    "001CrMa": Participant(
        code="001CrMa",
        date="2026-04-23",
        trials={
            "silence":            Trial(crf_entry_s=254, crf_exit_s=263, n_pas=7, pied="G"),
            "tempo_random":       Trial(crf_entry_s=248, crf_exit_s=253, n_pas=7, pied="D"),
            "beatmove_adaptatif": Trial(crf_entry_s=253, crf_exit_s=257, n_pas=7, pied="D",
                                          crop_end_s=358,
                                          note="back markers OUT ~3/4 into session"),
        },
    ),

    "002CrPa": Participant(
        code="002CrPa",
        date="2026-04-23",
        trials={
            "silence":            Trial(crf_entry_s=277, crf_exit_s=282, n_pas=7, pied="D"),
            "beatmove_adaptatif": Trial(crf_entry_s=243, crf_exit_s=248, n_pas=7, pied="G",
                                          crop_start_s=120,
                                          note="artefact 100-120s at start"),
            "tempo_random":       Trial(crf_entry_s=242, crf_exit_s=247, n_pas=7, pied="D",
                                          crop_end_s=460,
                                          note="last 30 s of back markers lost"),
        },
    ),

    "003BrLu": Participant(
        code="003BrLu",
        date="2026-05-05",
        trials={
            "silence":            Trial(crf_entry_s=235.42, crf_exit_s=239.27, n_pas=7,
                                          crop_start_s=72),
            "tempo_random":       Trial(crf_entry_s=251.22, crf_exit_s=254.75, n_pas=7,
                                          crop_end_s=400,
                                          note="back markers OUT ~40 s before end"),
            "beatmove_adaptatif": Trial(crf_entry_s=238, crf_exit_s=244, n_pas=7),
        },
    ),

    "004CaGe": Participant(
        code="004CaGe",
        date="2026-05-06",
        trials={
            "silence":            Trial(crf_entry_s=254.5, crf_exit_s=258.27,
                                          crop_start_s=67),
            "tempo_random":       Trial(crf_entry_s=251, crf_exit_s=255,
                                          crop_end_s=440),
            "beatmove_adaptatif": Trial(crf_entry_s=255.34, crf_exit_s=259),
        },
    ),

    "005DeJe": Participant(
        code="005DeJe",
        date="2026-05-07",
        trials={
            "silence":            Trial(crop_end_s=375,
                                          note="Recovered via cluster fill (V2, 17/05/2026); crop_end_s=375 removes final aberrant plateau. CRF not annotated."),
            "tempo_random":       Trial(crf_entry_s=248.51, crf_exit_s=251.89,
                                          crop_start_s=80, crop_end_s=440),
            "beatmove_adaptatif": Trial(crf_entry_s=251, crf_exit_s=254.26,
                                          crop_end_s=330,
                                          note="WATCH: sporadic spikes but overall signal acceptable"),
        },
    ),

    "006MoCa": Participant(
        code="006MoCa",
        date="2026-05-08",
        trials={
            "silence":            Trial(crf_entry_s=253.92, crf_exit_s=257.53,
                                          crop_end_s=426),
            "tempo_random":       Trial(crf_entry_s=244, crf_exit_s=247,
                                          crop_end_s=260,
                                          note="crop_end lowered 305→260 (massive artefact after 260s)"),
            "beatmove_adaptatif": Trial(),  # CRF S2 inverted in original Excel, to re-annotate
        },
    ),

    "007MoMa": Participant(
        code="007MoMa",
        date="2026-05-08",
        trials={
            "silence":            Trial(crf_entry_s=270, crf_exit_s=273.56,
                                          excluded=True,
                                          note="EXCLUDE: broken back markers, cluster fill FOPT=28mm (non-rigid), amplitude 180mm, f_dom 0.75Hz. V2 confirms exclusion."),
            "tempo_random":       Trial(crf_entry_s=243, crf_exit_s=247.5,
                                          crop_end_s=374),
            "beatmove_adaptatif": Trial(crf_entry_s=240, crf_exit_s=243,
                                          excluded=True,
                                          note="EXCLUDE: amplitude 213mm, f_dom 0.75Hz (not walking). V2 confirms exclusion."),
        },
    ),

    "008RiMo": Participant(
        code="008RiMo",
        date="2026-05-11",
        trials={
            # silence: crop_start raised from 45 to 60 (minimum 60s rule applied)
            "silence":            Trial(crf_entry_s=237.35, crf_exit_s=241.09, n_pas=6, pied="G"),
            "tempo_random":       Trial(crf_entry_s=235.59, crf_exit_s=238.87, n_pas=6, pied="G",
                                          note="Recovered via cluster fill (V2, 17/05/2026): 5.0% → 1.7% occluded, FOPT=1.13mm, 2780 marker spikes corrected."),
            "beatmove_adaptatif": Trial(crf_entry_s=241.44, crf_exit_s=245.12, n_pas=6, pied="G",
                                          crop_start_s=63, crop_end_s=324,
                                          note="WATCH: artefacts after 324s"),
        },
    ),

    "009DeFr": Participant(
        code="009DeFr",
        date="2026-05-11",
        trials={
            "silence":            Trial(crf_entry_s=242.23, crf_exit_s=245.77),
            "tempo_random":       Trial(crf_entry_s=234.49, crf_exit_s=237.61,
                                          crop_end_s=245,
                                          note="little post-CRF (~7s), pre-CRF baseline OK"),
            "beatmove_adaptatif": Trial(crf_entry_s=250.59, crf_exit_s=253.98,
                                          crop_end_s=364),
        },
    ),

    "010DeYv": Participant(
        code="010DeYv",
        date="2026-05-11",
        trials={
            "silence":            Trial(crf_entry_s=242.2, crf_exit_s=244.86),
            "tempo_random":       Trial(crf_entry_s=244.54, crf_exit_s=247.22),
            "beatmove_adaptatif": Trial(crf_entry_s=243.7, crf_exit_s=246.39),
        },
    ),

    "011RiJo": Participant(
        code="011RiJo",
        date="2026-05-12",
        trials={
            "silence":            Trial(crf_entry_s=275.34, crf_exit_s=279.99),
            "tempo_random":       Trial(crf_entry_s=235, crf_exit_s=260,
                                          note="CRF not timed, broad estimate (true range 220-270s)"),
            "beatmove_adaptatif": Trial(crf_entry_s=248.6, crf_exit_s=252.54),
        },
    ),

    "012WaCh": Participant(
        code="012WaCh",
        date="2026-05-12",
        trials={
            "silence":            Trial(crf_entry_s=246.62, crf_exit_s=250.56),
            "tempo_random":       Trial(crf_entry_s=246.73, crf_exit_s=250.27),
            "beatmove_adaptatif": Trial(crf_entry_s=257.91, crf_exit_s=261.88),
        },
    ),

    "013WaJe": Participant(
        code="013WaJe",
        date="2026-05-12",
        trials={
            "silence":            Trial(crf_entry_s=244.06, crf_exit_s=248.26,
                                          crop_end_s=308,
                                          note="WATCH: artefacts after 308s"),
            "tempo_random":       Trial(crf_entry_s=240.33, crf_exit_s=244.49),
            "beatmove_adaptatif": Trial(crf_entry_s=249.9, crf_exit_s=253.81),
        },
    ),

    "014DeAn": Participant(
        code="014DeAn",
        date="2026-05-13",
        trials={
            "silence":            Trial(crf_entry_s=255.88, crf_exit_s=259.44),
            "tempo_random":       Trial(crf_entry_s=239.61, crf_exit_s=242.87),
            "beatmove_adaptatif": Trial(crf_entry_s=285.18, crf_exit_s=288.39),
        },
    ),

    "015LaOd": Participant(
        code="015LaOd",
        date="2026-05-18",
        trials={
            "silence":            Trial(crf_entry_s=245.72, crf_exit_s=248.92, n_pas=6, pied="G"),
            "tempo_random":       Trial(crf_entry_s=252.15, crf_exit_s=255.97, n_pas=7, pied="G",
                                          note="participant stopped walking before sand entry (CRF still valid)"),
            "beatmove_adaptatif": Trial(crf_entry_s=241.76, crf_exit_s=245.02, n_pas=6, pied="G"),
        },
    ),

    "016LeSi": Participant(
        code="016LeSi",
        date="2026-05-18",
        trials={
            "silence":            Trial(crf_entry_s=242.74, crf_exit_s=247.89, n_pas=7, pied="G",
                                          note="beatMove 'silence' config failed → ran 'random' with audio muted (effectively silence)"),
            "tempo_random":       Trial(crf_entry_s=244.67, crf_exit_s=249.43, n_pas=7, pied="D"),
            "beatmove_adaptatif": Trial(crf_entry_s=241.80, crf_exit_s=247.21, n_pas=8, pied="D"),
        },
    ),

}


# ────────────────────────────────────────────────────────────────────────────
# HELPERS
# ────────────────────────────────────────────────────────────────────────────

def list_codes() -> list[str]:
    """List all registered participant codes."""
    return sorted(PARTICIPANTS.keys())


def get_trial(participant_code: str, condition: str) -> Trial:
    """Return a trial. Raises KeyError if participant or condition is missing."""
    p = PARTICIPANTS[participant_code]
    if condition not in p.trials:
        raise KeyError(f"Condition '{condition}' missing for {participant_code}")
    return p.trials[condition]


def is_complete(participant_code: str) -> bool:
    """True if all CRF (entry + exit) timestamps are annotated."""
    p = PARTICIPANTS[participant_code]
    return all(
        t.crf_entry_s is not None and t.crf_exit_s is not None
        for t in p.trials.values()
    )


def list_complete() -> list[str]:
    """List participants ready for analysis (complete CRF annotations)."""
    return [code for code in list_codes() if is_complete(code)]


def list_trials_complete() -> list[tuple[str, str]]:
    """All individual trials with complete CRF (and not excluded)."""
    out = []
    for code, p in PARTICIPANTS.items():
        for cond, t in p.trials.items():
            if t.excluded:
                continue
            if t.crf_entry_s is not None and t.crf_exit_s is not None:
                out.append((code, cond))
    return sorted(out)


def list_trials_to_process() -> list[tuple[str, str]]:
    """
    All trials to process in the preprocess batch (not excluded, .mat available).

    Also includes trials without CRF (preprocessing does not depend on CRF);
    only `excluded=True` filters them out.
    """
    out = []
    for code, p in PARTICIPANTS.items():
        for cond, t in p.trials.items():
            if not t.excluded:
                out.append((code, cond))
    return sorted(out)
