"""
preprocess_participant.py — CLI pour preprocesser un participant complet.

Usage:
    python scripts/preprocess_participant.py 003CrLu

Lit les .mat depuis RAW_DIR (renommés proprement), sort les .npz dans
PROCESSED_DIR. Pour chaque condition du participant, applique :
    raw .mat → trajectoires → preprocess.run() → .npz
"""

import sys
from pathlib import Path

from resilience import paths, participants
from resilience.io import loader, writer
from resilience.processing import preprocess


def preprocess_one(participant_code: str, condition: str) -> Path | None:
    """Preprocess un essai. Retourne le chemin du .npz, ou None en cas d'échec."""
    trial = participants.get_trial(participant_code, condition)

    raw_path = paths.raw_file(participant_code, condition)
    if not raw_path.exists():
        print(f"  ❌ {condition:25s} → fichier absent : {raw_path}")
        return None

    print(f"  → {condition:25s} ({raw_path.name})")
    data, fmt = loader.load_mat(raw_path)
    trajectories = loader.extract_marker_trajectories(data, fmt)

    if trajectories is None:
        print("     ❌ Extraction des trajectoires impossible")
        return None

    out = preprocess.run(trajectories, verbose=False)
    npz_path = writer.save_processed(participant_code, condition, out,
                                       trajectories_raw=trajectories)
    print(f"     ✅ {npz_path.relative_to(paths.PROJECT_ROOT)}")
    return npz_path


def main():
    if len(sys.argv) != 2:
        print("Usage : python scripts/preprocess_participant.py <code_participant>")
        print(f"\nParticipants connus : {participants.list_codes()}")
        sys.exit(1)

    code = sys.argv[1]
    if code not in participants.PARTICIPANTS:
        print(f"❌ Participant inconnu : {code}")
        print(f"Participants connus : {participants.list_codes()}")
        sys.exit(1)

    print(f"\n=== Preprocessing {code} ===\n")
    p = participants.PARTICIPANTS[code]
    for condition in p.trials:
        preprocess_one(code, condition)
    print("\n✅ Terminé.\n")


if __name__ == "__main__":
    main()
