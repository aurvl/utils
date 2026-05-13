# Diagnostic — LatentDEWM ODE : instabilités et effondrements

**Date** : 2026-05-13  
**Modèle analysé** : `timeseries_model.py` (ODE uniquement)  
**Données** : `outputs/history_df.csv` (101 epochs), `outputs/figures/training_diagnostics.png` (~500 epochs)

---

## 1. Analyse des courbes

### 1.1 Colonnes du CSV

| Colonne | Signification |
|---|---|
| `train_loss_total` | Perte totale = pred + β·KL + jepa + id |
| `train_loss_pred` | Perte prédiction dominante (≈ dec_loss NLL) |
| `train_loss_kl` | KL brut non pondéré |
| `train_beta_kl` | Coefficient KL du notebook (annealing) |
| `train_theta_std` | Std des sorties de l'Identifier (poids dynamiques θ) |
| `train_z_std` | Std du vecteur latent z post-reparamétrage |

### 1.2 Total loss

- Diminue de ~0.096 (epoch 1) à ~0.046 (epoch 100) : descente régulière.
- Le gap train/val est faible aux premières epochs (~0.005 à epoch 10) mais croît légèrement après epoch 100 → overfitting modéré sur longue durée.
- **Aucun spike à 750 visible dans le CSV actuel.** Le spike décrit visuellement provient d'une run antérieure. Voir §2.1 pour l'hypothèse causale.

### 1.3 KL divergence

- Epoch 1 : KL_val = 12.1 (pic initial très élevé, encoder non régularisé).
- Decreasing rapidly: KL ≈ 1.0 after epoch 20, puis bruit autour de 1.0 avec des pics à ~3.
- **Beta KL** : le notebook anneal de 1e-5 à 0.0001 sur 10 epochs, puis reste constant.
- À partir d'epoch 10 : contribution KL dans la loss totale ≈ 0.0001 × 0.83 ≈ **0.000083** — totalement négligeable.

### 1.4 z_std (variabilité latente)

- Epoch 1 : z_std = 3.15 → Epoch 100 : z_std = 4.70 → Epoch 500 : plateau ~5.0.
- **z_std qui grandit en continu = signe fort de KL collapse soft** : le posterior N(μ, σ²) diverge de N(0,1) sans contrainte effective.
- Pour z_std=4.7 et z_mean≈-0.25 : exp(log_var) ≈ 22, KL ≈ 8.95 par dimension × 64 dims ≈ **573 par sample**. Avec β=0.0001, contribution = 0.057 — encore absorbable, mais le latent n'est pas régularisé.

### 1.5 Theta_std (variabilité des poids dynamiques)

- Epoch 1 : theta_std = 0.35 → Epoch 100 : theta_std = 1.81 → plateau ~2.0.
- Les poids de f augmentent en variabilité, mais train ≈ val exactement → la diversité ne dépend pas du type de processus, elle est subie (pas apprise).

---

## 2. Hypothèses causales

### 2.1 Spike massif epoch ~13 (run antérieure) — ODE explosion

**Cause : blowup exponentiel de la trajectoire latente sans normalisation du pas.**

L'Euler ODE implémenté est : `z_{t+1} = z_t + f(z_t)`, sans facteur `dt`.

Avec `n_steps = 24` et des poids initiaux du `weight_head` de variance standard (Linear Xavier):
- std(W) ≈ sqrt(2/256) ≈ 0.088  
- z_std ≈ 3 (epoch 1)
- std(f(z)) ≈ sqrt(64) × 0.088 × 3 ≈ **2.1 par dimension**

Après 24 pas sans dt : `z_24 ≈ z_0 + 24 × 2.1 = z_0 + 50`.  
L'espace latent explose. Le décodeur reçoit z₂₄ ~50× plus grand que z₀ → NLL explose (spike).

Le `grad_clip = 1.0` (déjà dans le code) limite les explosions de gradient mais pas la forward pass. Le modèle sort du spike en apprenant rapidement `f(z) ≈ 0` (voir §2.3).

### 2.2 KL collapse soft — z_std trop grand

**Cause : beta_kl = 0.0001 → KL effectivement ignoré.**

Le notebook utilise un beta_kl custom (visible dans la colonne `train_beta_kl`) qui plafonne à 0.0001. La `lambda_kl = 0.1` dans Config est contournée.

Conséquence : le posterior apprend une variance de 22× celle d'une N(0,1). L'espace latent est très large et non structuré, ce qui aggrave l'instabilité de l'ODE (f doit gérer des z grand et hétérogènes).

### 2.3 Prédictions plates — attracteur dégénéré de l'ODE

**Cause : f(z) ≈ 0 est le minimum local le plus stable disponible.**

Après le spike initial, le modèle converge vers la solution dégénérée `f(z) → 0` car :
1. Sans dt, l'ODE doit apprendre des poids très petits pour éviter l'explosion.
2. Le décodeur peut minimiser le NLL en ajustant σ plutôt qu'en améliorant μ.
3. Aucune régularisation ne force la trajectoire à être non-triviale.

Résultat : z_{t+1} ≈ z_t pour tout t → trajectoire constante → μ_pred constant sur H=24 → prédiction plate.

Evidence : `train_loss_pred ≈ train_loss_total` (dec_loss domine tout) et les poids dynamiques produisent des norms identiques (~4.5) quel que soit le processus.

### 2.4 Identifier collapse — t-SNE aléatoire

**Cause : augmentation trop faible (5%) + température NT-Xent trop basse (0.07).**

Le `_id_contrastive` utilise 5% de bruit gaussien pour créer des paires positives.

Problèmes :
1. Paires positives trivialement similaires → le modèle n'a pas besoin d'apprendre de features process-type pour satisfaire NT-Xent.
2. Les négatifs incluent des samples du MÊME type de processus → faux signal discriminant.
3. Température = 0.07 = distribution très peakée → optimisation instable (softmax saturée).

Conséquence : l'Identifier LSTM apprend à représenter l'identité individuelle des trajectoires, pas leur type de dynamique. Theta_std croît mais de façon non structurée → t-SNE aléatoire.

### 2.5 Weight norms uniformes (~4.5)

**Cause : conséquence directe de §2.3 et §2.4.**

Si f(z) ≈ 0 pour tous les samples, alors les poids W de chaque couche ont des norms similaires (proches de l'initialisation Xavier). L'Identifier ne différencie pas les processus car NT-Xent ne le force pas à le faire.

---

## 3. Recommandations concrètes et prioritisées

### Priorité 1 — Stabiliser l'ODE (critique)

**3.1 Ajouter `dt = 1/n_steps` à chaque pas Euler**

```python
# Dans EulerODE.forward :
dt = 1.0 / n_steps
z = z + dt * torch.tanh(_apply_dynamic_mlp(z, dyn_layers))
```

Effet :
- Le pas max par dimension est borné à `1/24 ≈ 0.04` (avec tanh).
- Élimine l'explosion exponentielle en forward pass.
- Brise l'attracteur `f ≈ 0` : le modèle peut maintenant apprendre des dynamics sans craindre le blowup.

**3.2 Initialiser le weight_head avec de petites sorties**

```python
# Dans Identifier.__init__, après la définition de weight_head :
nn.init.normal_(self.weight_head[-1].weight, std=0.01)
nn.init.zeros_(self.weight_head[-1].bias)
```

Effet : f(z) ≈ 0 au début de l'entraînement (initialisation stable), puis le modèle peut apprendre des dynamics non triviales sans spike préliminaire.

### Priorité 2 — Corriger le signal de l'Identifier

**3.3 Augmenter le bruit d'augmentation de 5% à 20%**

```python
# Dans TimeSeriesModel._id_contrastive :
scale = x.std(dim=1, keepdim=True) * 0.20   # était 0.05
```

Effet : force le modèle à apprendre des features invariantes à des perturbations plus larges → le LSTM doit capturer la structure dynamique (type de processus) et non l'identité individuelle de la trajectoire.

**3.4 Augmenter la température NT-Xent de 0.07 à 0.15**

```python
temperature: float = 0.15   # était 0.07
```

Effet : softmax moins saturée → gradients plus exploitables → convergence plus stable du NT-Xent.

### Priorité 3 — Renforcer la régularisation KL

**3.5 Augmenter lambda_kl (pour le train() standalone)**

```python
lambda_kl: float = 0.5   # était 0.1
```

Note : si le notebook utilise son propre beta_kl annealing, ce changement n'affectera pas la run du notebook. Il reste utile pour tout usage du train() intégré. Pour le notebook, le plafond beta_kl = 0.0001 est trop bas — viser 0.001 minimum.

---

## 4. Ce qui n'est PAS le problème principal

- Le `grad_clip = 1.0` est déjà en place et est correct.
- Le JEPA loss (Smooth L1 sur patches masqués) fonctionne normalement.
- L'EMA decay = 0.996 est acceptable.
- La sigma_clamp du décodeur `(-4, 2)` est large mais pas la cause première des prédictions plates.

---

## 5. Résumé exécutif

| Problème | Cause | Fix |
|---|---|---|
| Spike epoch ~13 | ODE blowup sans dt (f(z) × 24 pas) | dt=1/n_steps + tanh |
| Prédictions plates | Attracteur f≈0, décodeur σ-dominant | dt + tanh + init petite |
| t-SNE aléatoire | NT-Xent avec augmentation 5% trop faible | bruit 20% + temp 0.15 |
| Weight norms égales | Identifier ne discrimine pas | idem ci-dessus |
| z_std trop grand | beta_kl = 0.0001 négligeable | augmenter lambda_kl dans Config |
