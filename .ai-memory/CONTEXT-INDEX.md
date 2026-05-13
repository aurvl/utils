# CONTEXT-INDEX.md

## Active Memory Files

- `PROJECT-MEMORY-SUMMARY.md` — lire au début de chaque session sérieuse.
- `CURRENT-TASK.md` — lire avant d'exécuter la tâche courante.
- `DECISIONS.md` — lire si on change de direction ou révise une décision passée.
- `TASK-LOG.md` — mettre à jour après chaque tâche significative.
- `LEARNING-LOG.md` — mettre à jour après un insight d'apprentissage important.
- `OPEN-QUESTIONS.md` — lire/mettre à jour si un bloqueur persiste.

## Stable Global Context Files (C:\Users\aurel\.ais\)

- `AGENTS.md` — guide opérationnel central de l'agent.
- `MEMORY.md` — règles de gestion de la mémoire.
- `IDENTITY.md` — identité et comportement de l'agent (si besoin).
- `COLLABORATION CHARTER.md` — principes de collaboration (si besoin).
- `DELEGATION PROTOCOLS.md` — règles d'autonomie (si besoin).
- `METHOD SELECTION PROTOCOL.md` — sélection de méthode/architecture (si besoin).
- `REVIEW AND DILIGENCE.md` — revue, validation, publication (si besoin).

## Project Source Files

- `timeseries_model.py` — modèle principal ODE.
- `timeseries_model_sde.py` — variante SDE (moins prioritaire).
- `const.py` — CATEGORY_MAP des 110+ processus.
- `data_generation.py` — génération des données.
- `ood.py` — évaluation out-of-distribution.
- `train_timeseries.ipynb` — entraînement principal.
- `diagnostic.md` — rapport diagnostic complet (2026-05-13).
- `outputs/history_df.csv` — historique d'entraînement (101 epochs).
