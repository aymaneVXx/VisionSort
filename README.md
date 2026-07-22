# VisionSort

VisionSort est une application locale Python + Streamlit pour piloter un pipeline multicaméra de suivi de colis, handoff entre caméras, détection de prise/dépôt, génération de datasets et déclenchement d'entraînements.

## Principes d'architecture

- Streamlit n'exécute aucun traitement long: il écrit uniquement des commandes dans SQLite et lit les statuts.
- `python -m visionsort.runtime.supervisor` est l'unique orchestrateur des processus.
- Les images, previews, vidéos enregistrées et détails frame-level sont stockés sur disque.
- SQLite stocke les états, commandes, événements, tracklets, jobs, modèles, trackers et métadonnées.
- Les détails d'observation sont stockés en JSONL.
- L'inférence passe par un worker GPU partagé, avec un modèle chargé une seule fois pour toutes les caméras actives.
- Chaque caméra conserve son tracker local indépendant.
- Le mode simulé n'est autorisé que si `DEMO_MODE=1`.

## Arborescence principale

- `app.py` : interface Streamlit.
- `visionsort/runtime/supervisor.py` : supervisor persistant.
- `visionsort/acquisition/worker.py` : worker caméra.
- `visionsort/inference/engine.py` : worker GPU partagé et backends.
- `visionsort/tracking/engine.py` : tracker local, tracklets, tracking multicaméra.
- `visionsort/events/engine.py` : logique prise/transport/dépôt.
- `visionsort/datasets/pipeline.py` : création dataset YOLO.
- `visionsort/training/pipeline.py` : jobs d'entraînement.
- `scripts/init_project.py` : bootstrap DB et assets démo.
- `config/default.yaml` : configuration d'exemple.

## Pré-requis

- Python 3.10+ dans cet environnement de travail.
- Le projet a été écrit pour rester compatible avec Python 3.10, même si la cible demandée est Python 3.12.

## Installation

```powershell
python -m pip install -U pip
python -m pip install -e .
```

## Initialisation

Sans démo :

```powershell
python scripts/init_project.py
```

Avec démo Replay explicite :

```powershell
$env:DEMO_MODE="1"
python scripts/init_project.py
```

## Lancement

Terminal 1 :

```powershell
$env:DEMO_MODE="1"
python -m visionsort.runtime.supervisor
```

Terminal 2 :

```powershell
$env:DEMO_MODE="1"
streamlit run app.py
```

## Utilisation rapide en démo

1. Activer `DEMO_MODE=1`.
2. Lancer `scripts/init_project.py`.
3. Lancer le supervisor.
4. Ouvrir Streamlit.
5. Aller sur `Cameras`.
6. Utiliser `Bootstrap démo` si nécessaire.
7. Démarrer les sources `C1`, `C2`, `C3`.
8. Visualiser les previews et les événements dans `Live Tracking` et `Events`.
9. Activer l'enregistrement puis consulter `Recordings`.
10. Générer un dataset depuis `Dataset Studio`.
11. Lancer un entraînement depuis `Training`.

## Ce qui est implémenté

- Sources `Replay`, `VideoFileSource`, `RTSPSource`.
- Supervisor de processus indépendant de Streamlit.
- Worker GPU partagé avec chargement unique du modèle actif.
- Trackers locaux indépendants par caméra.
- Tracklets et suivi multicaméra prudents avec `MATCHED`, `AMBIGUOUS`, `UNMATCHED`.
- Machine d'états colis avec `ON_CONVEYOR`, `PICK_CANDIDATE`, `PICKED`, `CARRIED`, `DROP_CANDIDATE`, `DROPPED`.
- Génération d'un dataset YOLO à partir des tracklets Replay.
- Registre modèles et trackers avec identifiants SQLite.
- Entraînement séparé avec journalisation et gestion d'échec.

## Limites connues

- Les règles multicaméra, prise et dépôt sont testables en Replay mais restent non validées sur site sans données réelles.
- Le backend `demo_synth_det` est réservé à `DEMO_MODE`.
- Les modèles Ultralytics préentraînés peuvent nécessiter le téléchargement des poids au premier lancement.
- Les wrappers ByteTrack et BoT-SORT sont enregistrés dans SQLite; leur exploitation réelle dépend de la disponibilité runtime Ultralytics dans l'environnement.

## Tests

```powershell
pytest
```
