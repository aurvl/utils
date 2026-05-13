# OPEN-QUESTIONS.md

## Question 1 — beta_kl du notebook trop bas

### Question

Le notebook `train_timeseries.ipynb` utilise un beta_kl custom qui plafonne à 0.0001 — beaucoup trop bas pour contraindre le posterior. Faut-il modifier le notebook ou ajouter un argument à TimeSeriesModel ?

### Why It Matters

Avec beta_kl=0.0001, la KL est ignorée → z_std grossit librement → l'espace latent n'est pas régularisé → l'ODE doit gérer des z arbitrairement grands.

### Current Hypothesis

Modifier le notebook pour plafonner beta_kl à 0.001 minimum. Ou retirer l'annealing custom et utiliser Config.lambda_kl directement.

### Status

Open

### Next Action

Ouvrir `train_timeseries.ipynb` et chercher la cellule qui définit beta_kl annealing. Changer le plafond.

---

## Question 2 — Identifier discrimine-t-il vraiment les types de processus ?

### Question

Après les fixes (augmentation 20%, température 0.15), le t-SNE des embeddings Identifier montre-t-il une séparation par catégorie (oscillators, stochastic, chaotic...) ?

### Why It Matters

C'est le critère central de succès du module Identifier. Si non, les poids dynamiques f ne sont pas conditionnés sur le type de processus → le modèle est équivalent à un ODE non conditionné.

### Current Hypothesis

Les fixes devraient aider, mais la quantité de données par processus et la durée d'entraînement peuvent encore limiter la discrimination.

### Status

Open

### Next Action

Re-run entraînement avec fixes. Après convergence, extraire embeddings Identifier pour 200-500 samples et visualiser avec t-SNE coloré par catégorie.

---

## Question 3 — f(z)≈0 attracteur est-il brisé après dt+tanh ?

### Question

Même avec dt=1/n_steps et tanh, est-ce que l'ODE apprend des dynamics non triviales (trajectoires non constantes dans l'espace latent) ?

### Why It Matters

Si f(z)≈0 persiste, les prédictions restent plates indépendamment des autres fixes.

### Status

Open

### Next Action

Après re-run, logger la norme de f(z) (ou theta_std en contexte) et vérifier que les trajectoires z_0 → z_H varient entre samples.
