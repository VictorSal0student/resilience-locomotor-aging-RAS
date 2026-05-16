"""
Pipeline de traitement des données — Résilience Locomotrice.

EuroMov DHM | Victor Salvat | Master IEAP DigiMove-IDIL (2025-2026)

Modules
-------
paths        — Chemins disque (lit .env)
config       — Constantes scientifiques (FS, marqueurs, paramètres pipeline)
participants — Registre unique des participants (CRF, ordre, crops)

io.loader    — Chargement .mat CODAMOTION
io.writer    — Écriture .npz preprocessed

processing.preprocess — Interpolation, EMD, Butterworth, décimation
processing.detection  — Détection PCA de la transition sol/sable
processing.tde        — AMI, FNN, phase space (Takens)

analysis.resilience   — Métriques de résilience (Option C)

viz.plots             — Visualisations standards

Usage minimal
-------------
>>> from resilience import paths, config, participants
>>> from resilience.io import loader, writer
>>> from resilience.processing import preprocess, detection, tde
>>> from resilience.analysis import resilience
"""

__version__ = "0.1.0"
__author__  = "Victor Salvat"
