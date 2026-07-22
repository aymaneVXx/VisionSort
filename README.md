# VisionSort

VisionSort est une plateforme locale Python + Streamlit pour piloter un cycle complet Replay/vision industrielle autour du suivi de colis:

- configuration de `CaptureSession` C1/C2/C3
- acquisition et enregistrement
- inférence par modèle sélectionnable
- tracking local et association multicaméra
- génération de datasets
- pseudo-annotation et review
- entraînement, évaluation, promotion et rollback de modèles

## Architecture

- Streamlit ne lance aucun traitement persistant: il écrit des commandes dans SQLite et lit les états.
- `python -m visionsort.runtime.supervisor` est l’orchestrateur unique des workers et jobs.
- SQLite stocke uniquement commandes, sessions, états, jobs, événements, tracklets, datasets, modèles et trackers.
- Les images, previews, enregistrements, observations détaillées et rapports restent sur disque.
- Les observations détaillées sont stockées en `JSONL`, avec export `Parquet` possible via un step pipeline dédié.
- L’inférence est conçue autour d’un worker partagé par modèle sélectionné.
- Chaque caméra conserve son tracker local indépendant.
- Le mode simulé est explicite: aucun résultat démo ne doit être utilisé silencieusement hors `DEMO_MODE=1`.

## Modules Principaux

- `app.py` : point d’entrée Streamlit
- `visionsort/runtime/supervisor.py` : supervisor persistant et gestion des commandes
- `visionsort/runtime/pipeline_worker.py` : steps pipeline (`PROCESS_SESSION`, `SAMPLE`, `AUTO_ANNOTATE`, `FINALIZE_DATASET`, `EXPORT_OBSERVATIONS_PARQUET`)
- `visionsort/acquisition/worker.py` : boucle caméra/source, previews, enregistrement, observations JSONL
- `visionsort/inference/engine.py` : backends de modèles et provenance modèle/version
- `visionsort/tracking/engine.py` : trackers locaux, tracklets, matching multicaméra
- `visionsort/events/engine.py` : événements métier prise/transport/dépôt
- `visionsort/datasets/pipeline.py` : création dataset, split stable, déduplication, provenance
- `visionsort/training/pipeline.py` : training, évaluation, candidat, rapport
- `visionsort/deployment/registry.py` : activation, promotion, rejet, archivage, rollback
- `visionsort/observations/export.py` : export `JSONL -> Parquet`
- `visionsort/ui/pages/` : pages Dashboard, Cameras, Live Tracking, Recordings, Dataset Studio, Training, Models, Events, Settings

## Pré-Requis

- Python `3.10+`
- cible de projet demandée: `Python 3.12`
- sur cet environnement, les commandes validées utilisent `python -m ...`

## Installation

```powershell
python -m pip install -U pip
python -m pip install -e .
```

## Initialisation

Mode standard :

```powershell
python scripts/init_project.py
```

Mode Replay démo explicite :

```powershell
$env:DEMO_MODE="1"
python scripts/init_project.py
```

## Démarrer L’Application

Ouvrir **2 terminaux** dans le dossier du projet.

Terminal 1, supervisor :

```powershell
$env:DEMO_MODE="1"
python -m visionsort.runtime.supervisor
```

Terminal 2, Streamlit :

```powershell
$env:DEMO_MODE="1"
python -m streamlit run app.py
```

Ensuite ouvrir :

- [http://localhost:8501](http://localhost:8501)

## Arrêter L’Application

Pour arrêter proprement :

1. Dans Streamlit, arrêter d’abord les sessions/sources si elles tournent encore.
2. Dans chaque terminal, appuyer sur `Ctrl+C`.

Ordre recommandé :

1. arrêter Streamlit avec `Ctrl+C`
2. arrêter le supervisor avec `Ctrl+C`

Si un worker caméra reste bloqué anormalement, relancer le supervisor puis arrêter la session depuis l’UI avant de quitter.

## Workflow Replay Recommandé

1. Activer `DEMO_MODE=1`
2. Initialiser le projet
3. Lancer le supervisor
4. Lancer Streamlit
5. Aller dans `Cameras`
6. Enregistrer ou bootstrapper les sources Replay
7. Créer une `CaptureSession` avec C1/C2/C3 et offsets si nécessaire
8. Démarrer la session
9. Consulter `Dashboard`, `Live Tracking`, `Events`, `Recordings`
10. Arrêter la session
11. Aller dans `Dataset Studio`
12. Lancer `SAMPLE`
13. Lancer `AUTO_ANNOTATE`
14. Revoir les items `NEEDS_REVIEW`
15. Lancer `FINALIZE_DATASET`
16. Optionnel: lancer `EXPORT_OBSERVATIONS_PARQUET`
17. Aller dans `Training`
18. Lancer un entraînement
19. Aller dans `Models`
20. Comparer, promouvoir, activer ou rollbacker le modèle

## Pipeline Runtime

Le cycle persistant actuellement câblé autour des sessions/datasets couvre notamment :

- `CAPTURED`
- `PROCESSED`
- `SAMPLED`
- `AUTO_ANNOTATED`
- `REVIEW_PENDING`
- `DATASET_READY`
- `TRAINING`
- `EVALUATED`
- `CANDIDATE`
- `DEPLOYED`
- `REJECTED`

Des rapports JSON machine-readable sont produits dans `data/runtime/reports/`.

## Fonctionnalités Opérationnelles

- `CaptureSession` avec C1/C2/C3 et offsets Replay
- sources `Replay`, `VideoFileSource`, `RTSPSource`
- timestamps `local` et `global`
- observations détaillées sur disque en `JSONL`
- export `Parquet` via pipeline si dépendances disponibles
- previews JPEG et enregistrements segmentés
- tracking local par caméra
- tracklets persistés
- matching multicam `MATCHED / AMBIGUOUS / UNMATCHED`
- événements prise/transport/dépôt en logique Replay
- génération de datasets YOLO depuis tracklets
- split stable et déduplication basique d’images
- pseudo-annotation et review `NEEDS_REVIEW`
- training hors Streamlit
- évaluation post-training
- registre modèles avec `CANDIDATE / CHAMPION / REJECTED / ARCHIVED`
- activation, promotion et rollback

## Limites Connues

- Les règles multicaméra, prise et dépôt sont testables en Replay mais non validées sur site.
- Le backend `demo_synth_det` reste réservé à `DEMO_MODE`.
- Les modèles Ultralytics peuvent nécessiter le téléchargement des poids au premier lancement.
- ByteTrack et BoT-SORT sont explicitement limités par l’environnement runtime disponible.
- L’export Parquet dépend de `pandas` + `pyarrow`.
- La validation RTSP réelle, la calibration géométrique et les réglages métier nécessitent encore les vraies caméras.

## Tests

Exécution complète validée récemment :

```powershell
python -m pytest tests/test_supervisor_stop_session.py tests/test_supervisor_commands.py tests/test_pipeline_guardrails.py tests/test_dataset_pipeline.py tests/test_pipeline_worker.py tests/test_training_pipeline.py tests/test_training_registry_cycle.py tests/test_model_registry.py tests/test_tracking_events.py tests/test_database.py
```

Run rapide :

```powershell
python -m pytest
```
