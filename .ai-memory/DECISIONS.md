# DECISIONS.md

## 2026-05-13 — Fixes ODE post-diagnostic

### Decision

Appliquer 5 corrections à `timeseries_model.py` suite à l'analyse diagnostique.

### Reason

Diagnostic complet (voir `diagnostic.md`) identifiant 5 causes racines : ODE blowup, KL collapse, attracteur f≈0, NT-Xent dégénéré, Identifier non-discriminant.

### Corrections appliquées

1. `dt = 1/n_steps` dans EulerODE.forward (+ tanh sur f(z)) — stabilise la forward pass
2. `nn.init.normal_(weight_head[-1].weight, std=0.01)` — initialisation stable
3. Augmentation noise 5% → 20% — force Identifier à capturer la structure dynamique
4. Température NT-Xent 0.07 → 0.15 — optimisation plus stable
5. lambda_kl 0.1 → 0.5 — meilleure régularisation (affecte train() standalone)

### Alternatives Considered

- Gradient clipping plus agressif (déjà à 1.0, suffisant)
- Changer le scheduler ODE (Runge-Kutta) — rejeté pour complexité inutile à ce stade

### Alternatives Rejected

- Augmenter le nombre de steps ODE — n'aide pas si f(z)≈0

### Consequences

- Le beta_kl du notebook doit être corrigé séparément (plafond 0.0001 → ≥0.001)
- Re-run nécessaire pour vérifier que les fixes fonctionnent

### Reversible?

Oui — les paramètres sont dans `Config` et peuvent être rétablis.

---

## 2026-05-13 — Architecture : ODE vs SDE

### Decision

Priorité donnée à la version ODE (`timeseries_model.py`). Version SDE (`timeseries_model_sde.py`) mise de côté jusqu'à stabilisation.

### Reason

L'ODE doit d'abord être stable et discriminant avant d'ajouter la complexité stochastique.

### Reversible?

Oui.
