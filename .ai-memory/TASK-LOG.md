# TASK-LOG.md

## 2026-05-13 — Diagnostic ODE instabilities

### Task

Analyse complète des instabilités du modèle ODE après ~500 epochs d'entraînement.

### Agent / Tool Used

Claude Code (claude-sonnet-4-6)

### Files Modified

- `timeseries_model.py` — fixes dt, tanh, init, augmentation, temperature, lambda_kl
- `diagnostic.md` — créé (rapport complet)

### What Was Done

- Analyse de `outputs/history_df.csv` (101 epochs) et `outputs/figures/training_diagnostics.png` (~500 epochs)
- Identification de 5 causes racines (ODE blowup, KL collapse soft, attracteur f≈0, NT-Xent dégénéré, Identifier collapse)
- Application des corrections dans `timeseries_model.py`
- Création du rapport diagnostic.md

### Checks / Tests Run

Analyse statique uniquement. Pas de re-run d'entraînement effectué.

### Result

Fixes appliqués. Rapport documenté. Re-run nécessaire pour validation.

### Issues / Risks

- beta_kl du notebook non corrigé (plafond 0.0001 trop bas)
- Fixes non validés par entraînement

### Next Action

Re-run entraînement avec fixes. Vérifier z_std, theta_std, t-SNE Identifier.

---

## 2026-05-13 — Initialisation .ai-memory/

### Task

Créer le système de mémoire locale du projet selon MEMORY.md et AGENTS.md.

### Agent / Tool Used

Claude Code (claude-sonnet-4-6)

### Files Modified

- `.ai-memory/` (créé) : tous les fichiers mémoire
- `.gitignore` — ajout de `.ai-memory/`

### What Was Done

Lecture de AGENTS.md, MEMORY.md, exploration du projet, création de tous les fichiers mémoire avec contenu réel.

### Result

Système .ai-memory/ opérationnel.

### Next Action

Utiliser `PROJECT-MEMORY-SUMMARY.md` en début de session pour éviter de re-scanner le projet.
