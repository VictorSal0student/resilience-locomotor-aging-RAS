# Data

This folder is gitignored — datasets are not versioned in this repository.

Contact Victor Salvat or the EuroMov DHM Lab to request access to the dataset.

## Expected structure

```
data/
├── raw/                    CODAMOTION .mat files
│   ├── 001CrMa/
│   │   ├── 001CrMa_01_silence.mat
│   │   ├── 001CrMa_02_tempo_random.mat
│   │   └── 001CrMa_03_beatmove_adaptatif.mat
│   ├── 002CrPa/
│   │   └── ...
│   └── ...
│
├── processed/              .npz produced by notebooks/01_batch_preprocess.ipynb
│   └── {participant}/{participant}_{NN}_{condition}.npz
│
├── beatmove/               beatMove session logs (.csv)
│   └── {participant}/{condition}/
│
└── external/               Manual CRF annotations, recap files
```

## File naming convention

```
{participant_code}_{acquisition_index:02d}_{condition}.{ext}
```

- `participant_code`: e.g. `001CrMa`, `012WaCh` (3-digit ID + 2-letter surname + 2-letter first name)
- `acquisition_index`: 01, 02 or 03 — order in the Odin session
- `condition`: `silence`, `tempo_random`, or `beatmove_adaptatif`

The acquisition index reflects the randomised condition order and varies per participant.

## Format details

- **raw `.mat`**: CODAMOTION Odin export, MATLAB v5/v6 or v7.3 (HDF5). 400 Hz, 12 markers (Dos01–04, GLacet/GMT1/GMT5/GTalon, DLacet/DMT1/DMT5/DTalon).
- **processed `.npz`**: outputs of the preprocessing pipeline (see `src/resilience/io/writer.py` for the full payload).

## Privacy

Participant codes are anonymous identifiers. No personal information (name, address, date of birth) is stored in this repository or its outputs. The internal mapping code ↔ identity is held by the EuroMov DHM Lab under the RAS protocol.
