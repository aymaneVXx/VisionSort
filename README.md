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
- L’inférence utilise un scheduler central et un cache partagé par `model_id`;
  une source peut combiner les rôles colis et pose.
- Les pipelines `use_active=true` lisent un routage partagé par tâche et
  changent de version sans redémarrer la source; les pipelines
  `use_active=false` conservent leur modèle fixe.
- Chaque résultat d’inférence conserve le modèle réellement utilisé et la
  génération de routage. La bascule charge et vérifie la nouvelle version
  avant de router les frames suivantes, puis décharge l’ancienne version
  après drainage de ses requêtes.
- Chaque caméra conserve son tracker local indépendant.
- `bytetrack_cpu` et `botsort_cpu` utilisent les implémentations natives Ultralytics; `greedy_iou` reste une option de démonstration explicite.
- L'acquisition utilise un buffer borné `latest frame wins` et ne bloque plus sur le temps d'inférence.
- Le mode simulé est explicite: aucun résultat démo ne doit être utilisé silencieusement hors `DEMO_MODE=1`.

### Relations SQLite ajoutées

- `recordings` représente les segments immuables d’une session et
  `recording_frames` indexe précisément chaque
  `(session_id, source_id, stream_epoch, frame_index)`.
- `session_media_coverage` conserve le bilan d’archivage par source.
- `source_model_assignments` relie une source à ses pipelines
  `parcel_detection`, `parcel_segmentation` et `operator_pose`;
  `capture_session_sources.model_pipeline_json` en garde le snapshot.
- `model_registry.is_active` est unique par tâche, et non plus global.
- `model_activation_history` conserve les activations, échecs, remplacements
  et rollbacks réellement appliqués au runtime, par tâche.
- `handoff_hypotheses` conserve les ambiguïtés;
  `handoff_resolution_audit` journalise chaque résolution ou refus avec
  l’ancienne et la nouvelle chaîne.

Les migrations incrémentales SQLite v6, v7, v8 et v9 ajoutent ces structures
sans recréer les bases existantes.

## Modules Principaux

- `app.py` : point d’entrée Streamlit
- `visionsort/runtime/supervisor.py` : supervisor persistant et gestion des commandes
- `visionsort/runtime/pipeline_worker.py` : steps pipeline (`PROCESS_SESSION`, `SAMPLE`, `AUTO_ANNOTATE`, `FINALIZE_DATASET`, `EXPORT_OBSERVATIONS_PARQUET`)
- `visionsort/runtime/e2e.py` : validation CPU complète avec backends simulés explicitement
- `visionsort/runtime/supervisor_e2e.py` : validation multiprocessus de l'archive immuable, du dataset et du déploiement
- `visionsort/runtime/multimodel_e2e.py` : validation multiprocessus des pipelines parcelle + pose et du rechargement sélectif
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
- regroupement de toutes les instances par frame et groupes synchronisés C1/C2/C3
- split immuable par session, déduplication et contrôles anti-fuite
- annotateurs séparés détection, segmentation et pose, plus manifests tracking/ReID
- pseudo-annotation et review `NEEDS_REVIEW`
- training hors Streamlit
- évaluation post-training
- registre modèles avec `CANDIDATE / CHAMPION / REJECTED / ARCHIVED`
- activation, promotion et rollback
- activation à chaud transactionnelle, isolée par tâche, avec rollback fondé
  uniquement sur un ancien déploiement vérifié
- jobs idempotents et reprenables, verrou anti-doublon, annulation persistée
- artefacts `best.pt` copiés dans un répertoire de version immuable
- activation suivie d'un rechargement contrôlé du worker d'inférence
- diagnostic Models comparant registre SQLite, routage runtime, modèles
  chargés, références et requêtes en vol

## Archive média et `latest frame wins`

L’archive actuelle contient les frames retenues par le runtime après le
mécanisme de buffer borné `latest frame wins`. Elle garantit la
reproductibilité des observations, la génération du dataset et la traçabilité
des décisions prises sur ces frames.

Elle ne garantit pas l’enregistrement brut de toutes les frames reçues par un
flux RTSP. Un enregistreur brut intégral reste une fonctionnalité optionnelle
future si le cahier des charges terrain l’exige; il n’est pas ajouté par cette
correction.

## Smoke test local RTX 4050

Ce test est volontairement hors CI et exige trois fichiers de poids
Ultralytics locaux: deux versions détection (ou segmentation) et un modèle
Pose. Il refuse les backends `demo`, vérifie CUDA, exécute de vraies
inférences, bascule parcel v1 vers v2, protège puis décharge v1, et mesure la
mémoire dans le processus GPU.

```powershell
$env:DEMO_MODE="0"
$env:YOLO_CONFIG_DIR="$PWD\data\ultralytics"
nvidia-smi
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
python -m visionsort.runtime.gpu_smoke_test `
  --parcel-v1 "C:\models\parcel-v1.pt" `
  --parcel-v2 "C:\models\parcel-v2.pt" `
  --pose "C:\models\yolo11n-pose.pt" `
  --parcel-task detection `
  --iterations 30 `
  --report "data\runtime\reports\gpu-smoke.json"
```

Le rapport compare `before_activation`, `after_activation` et
`after_unload`, donne les échantillons de mémoire du benchmark, le FPS
approximatif, la confirmation `MODEL_UNLOADED`, les modèles encore chargés et
l’arrêt propre. Par défaut, une croissance persistante supérieure à
128 MiB fait échouer le test; le seuil est ajustable avec
`--memory-growth-limit-mb`. Sans CUDA, le script retourne `SKIPPED` et ne fait
pas échouer la CI.

## Limites Connues

- Les règles multicaméra, prise et dépôt sont testables en Replay mais non validées sur site.
- Le backend `demo_synth_det` reste réservé à `DEMO_MODE`.
- Les poids Ultralytics doivent être présents localement et leur empreinte vérifiable avant chargement; le runtime ne masque pas de téléchargement automatique.
- ByteTrack et BoT-SORT exigent `lap` et restent à valider sur les flux réels du site.
- Un dataset mono-session appartient volontairement à un seul split; plusieurs sessions sont nécessaires pour un entraînement réel train/val/test sans fuite.
- Le checkpoint produit par le scénario E2E démo est explicitement simulé; seul le chemin Ultralytics produit de vrais poids.
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

Scénario end-to-end CPU explicite :

```powershell
$env:DEMO_MODE="1"
python -m visionsort.runtime.e2e --db data/runtime/e2e.db --report data/runtime/reports/e2e.json
```

Ce scénario traite les trois Replay, construit et revoit le dataset, lance
l'entraînement démo, crée/active un candidat et vérifie son utilisation lors
d'une seconde session. Le rapport conserve `NON_VALIDÉ_SUR_SITE` pour tout ce
qui dépend encore des vraies caméras.

Scénario end-to-end multiprocessus via le superviseur :

```powershell
$env:DEMO_MODE="1"
python -m visionsort.runtime.supervisor_e2e --db data/runtime/supervisor-e2e.db --report data/runtime/reports/supervisor-e2e.json
```

Il enregistre des `VideoFileSource` en segments immuables, exécute trois
sessions isolées pour les splits train/val/test, modifie ensuite les URI
courantes, puis vérifie que sampling, validation stricte, fingerprint,
entraînement, promotion et activation utilisent encore l'archive de capture.

Scénario end-to-end multi-modèle via le superviseur :

```powershell
$env:DEMO_MODE="1"
python -m visionsort.runtime.multimodel_e2e --db data/runtime/multimodel-e2e.db --report data/runtime/reports/multimodel-e2e.json
```

Il maintient les Replay actifs en boucle pendant les bascules. Il vérifie
parcel v1 + pose v1, active pose v2 puis parcel v2 sur des frames ultérieures,
confirme l’isolation par tâche et les déchargements, simule l’échec parcel v3,
puis rollbacke explicitement vers parcel v1 réellement déployé. Le rapport
contient la provenance par frame, la timeline, les références, les in-flight,
les confirmations `MODEL_UNLOADED` et la cohérence finale. La CI exécute
installation, compilation et tests sous Python 3.10 et 3.12; les trois
scénarios E2E sont lancés sous Python 3.12.
