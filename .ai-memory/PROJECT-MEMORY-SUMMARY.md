# PROJECT-MEMORY-SUMMARY.md

## Project Goal

Build **LatentDEWM** — a universal time series forecasting model capable of identifying and adapting to different generative processes (oscillators, stochastic, chaotic, regime-switching, physical ODE, etc.) via a structured latent space and learned dynamics.

The model is a research/portfolio project exploring latent ODE + JEPA + hypernetwork architectures on synthetic multi-process data.

---

## Current State

- Model (ODE version) implemented and training. 101 epochs logged in `outputs/history_df.csv`.
- Diagnostic analysis completed on 2026-05-13 (`diagnostic.md`): 5 root causes identified and fixes applied.
- Fixes applied to `timeseries_model.py`: dt=1/n_steps, tanh on f(z), small weight_head init, augmentation noise 5%→20%, temperature 0.07→0.15, lambda_kl 0.1→0.5.
- SDE variant exists in `timeseries_model_sde.py` (less developed).
- OOD evaluation logic in `ood.py`.

---

## Architecture / File Map

- `timeseries_model.py`: Main ODE model — PatchEmbedding, JEPAPredictor, Encoder (online+target EMA), Identifier (LSTM hypernetwork), EulerODE, Decoder, TimeSeriesModel.
- `timeseries_model_sde.py`: SDE variant of the model.
- `const.py`: CATEGORY_MAP — 110+ process types in 8 categories.
- `data_generation.py`: Data generation for all process types.
- `ood.py`: Out-of-distribution evaluation.
- `questions.py`: Probably analysis / diagnostic queries.
- `train_timeseries.ipynb`: Main training notebook (uses custom beta_kl annealing).
- `ts-jepa.ipynb`: JEPA-specific experiments.
- `data/`: Raw generated data (gitignored).
- `outputs/`: Training logs, figures, models.
- `figures/`: Saved plots.
- `models/`: Saved model checkpoints.

---

## Main Data / Inputs

- Synthetic multi-process time series: 8 categories, ~110 process types.
- Context window: Q=96 timesteps. Horizon: H=24 timesteps. Input dim: C=1 (univariate).
- Train/val split done in notebook.
- Data in `data/` (gitignored). Preprocessed arrays: `X_past.npy`, `X_future.npy`, `labels.npy`.

---

## Main Decisions

- **Encoder**: Bootstrapped JEPA with online/EMA target encoder. Mask ratio=0.75. EMA decay=0.996.
- **Latent**: VAE-style reparameterization, z_dim=64. lambda_kl=0.5 (raised from 0.1 post-diagnostic).
- **Dynamics**: Euler ODE with `dt=1/n_steps` (fix applied), tanh on f(z). Identifier (LSTM) generates per-sample MLP weights.
- **Loss**: JEPA (Smooth L1) + KL + NT-Xent ID (temp=0.15) + Decoder NLL (Gaussian).
- **Augmentation**: 20% noise for NT-Xent pairs (raised from 5% post-diagnostic).
- **ODE blowup fix**: Added `dt=1/n_steps` and `tanh` — eliminates forward pass explosion.
- **KL regularization**: lambda_kl=0.5, but notebook uses its own beta_kl (plateaus at 0.0001 — too low; should target ≥0.001).

---

## Current Blockers

- Notebook's custom beta_kl annealing plateaus at 0.0001 — this overrides Config.lambda_kl for KL regularization during notebook training. z_std may still grow unconstrained.
- No verification yet that Identifier actually separates process types (t-SNE not re-run post-fix).
- ODE dynamics post-fix: unknown if f(z)≈0 attracteur is broken or still dominant.

---

## Last Completed Task

Diagnostic analysis (2026-05-13): full root-cause analysis of ODE instabilities, flat predictions, Identifier collapse. Fixes applied to `timeseries_model.py`.

## Current Task

Set up `.ai-memory/` system for this project.

## Next Small Action

Re-run training with fixes applied. Check z_std, theta_std evolution and run t-SNE on Identifier embeddings to verify process-type separation.

---

## Important Constraints

- Data is gitignored (synthetic generation takes time).
- Model is research/exploration — not production.
- Notebook beta_kl must be corrected separately from Config.lambda_kl.

## Do Not Forget

- The `diagnostic.md` file at project root is the reference for all known issues and fixes.
- `train_timeseries.ipynb` uses its own beta_kl annealing — verify its ceiling is raised to ≥0.001.
- `timeseries_model_sde.py` is less developed; don't prioritize unless ODE version is stable.
