# LEARNING-LOG.md

## 2026-05-13 — ODE blowup : le dt manquant

### Topic

Stabilité des Neural ODEs avec Euler discret.

### What I Understood

Sans `dt = 1/n_steps`, l'ODE Euler accumule `n_steps × f(z)` non normalisé. Avec n_steps=24 et des poids Xavier standard, le signal explose en forward pass avant même le calcul du gradient. Le modèle répond en apprenant f(z)≈0, ce qui produit des prédictions plates.

### Mistake or Friction

Le dt était absent de l'implémentation initiale. L'erreur n'était pas visible dans le code sans analyser l'ordre de grandeur des activations.

### Rule for Next Time

Toujours vérifier l'ordre de grandeur de la sortie d'une ODE discrète sur z aléatoire avant de lancer l'entraînement.

---

## 2026-05-13 — NT-Xent : importance de la température et de l'augmentation

### Topic

Apprentissage contrastif pour la discrimination de processus.

### What I Understood

Une augmentation trop faible (5%) crée des paires positives trivialement similaires → le modèle n'apprend pas de features process-type. Une température trop basse (0.07) sature le softmax → instabilité d'optimisation. Les deux problèmes simultanément font que l'Identifier apprend l'identité individuelle, pas la catégorie de dynamique.

### Rule for Next Time

Pour NT-Xent : augmentation suffisamment forte pour forcer l'invariance (≥15-20% du std), température autour de 0.1-0.2 pour des embeddings de dim 64-128.
