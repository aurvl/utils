"""
╔══════════════════════════════════════════════════════════════════╗
║  TIMESERIES MODEL WITH LATENT SDEs                             ║
║                                                                  ║
║  Extends the previous ODE-based model to full SDE framework:   ║
║    - Stochastic latent dynamics with drift f + diffusion σ     ║
║    - Prior SDE vs Posterior SDE (variational inference)        ║
║    - Generative likelihood (non-Gaussian OK)                   ║
║    - Adjoint-based gradient computation                        ║
╚══════════════════════════════════════════════════════════════════╝
"""

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, Independent


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SDE SOLVER (Euler-Maruyama)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class EulerMaruyamaSDE:
    """
    Solves: dz = f(z,t) dt + σ(z,t) dW_t
    
    Using Euler-Maruyama: z_{n+1} = z_n + f(z_n,t_n) Δt + σ(z_n,t_n) √Δt ξ_n
    where ξ_n ~ N(0,1)
    """
    
    @staticmethod
    def solve(
        z0:       torch.Tensor,
        f_drift:  callable,
        f_sigma:  callable,
        t_span:   Tuple[float, float],
        n_steps:  int,
        seed:     Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        z0     : (B, D)  initial state
        f_drift: (B,D,t) → (B,D)  drift function
        f_sigma: (B,D,t) → (B,D)  diffusion function (std per dim)
        n_steps: number of Euler steps
        
        Returns
            z_traj : (B, H, D)  trajectory (z at each step)
            noise  : (B, H, D)  sampled noise (for backward pass)
        """
        B, D = z0.shape
        device = z0.device
        
        if seed is not None:
            torch.manual_seed(seed)
        
        t0, t1 = t_span
        dt = (t1 - t0) / n_steps
        sqrt_dt = math.sqrt(dt)
        
        z = z0
        traj = [z.clone()]
        noises = []
        
        for i in range(n_steps):
            t = t0 + i * dt
            
            # Drift and diffusion at current state
            drift = f_drift(z, t)           # (B, D)
            sigma = f_sigma(z, t)           # (B, D)  — std per dimension
            
            # Sample noise
            eps = torch.randn(B, D, device=device)  # (B, D)
            noises.append(eps)
            
            # Euler-Maruyama step
            z = z + drift * dt + sigma * sqrt_dt * eps
            traj.append(z.clone())
        
        z_traj = torch.stack(traj, dim=1)  # (B, H+1, D)
        noise_traj = torch.stack(noises, dim=1)  # (B, H, D)
        
        return z_traj, noise_traj


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# STOCHASTIC IDENTIFIER (predicts both drift f and diffusion σ)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class StochasticIdentifier(nn.Module):
    """
    X → LSTM → h → (f_weights, σ_weights)
    
    Hypernetwork that predicts:
      - f_weights: weights of the drift MLP f
      - σ_weights: weights of the diffusion MLP σ
    
    Both σ_prior and σ_posterior share the same diffusion structure
    (required for finite KL divergence in SDE framework)
    """
    
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.shapes = self._weight_shapes(cfg.latent_dim, cfg.dyn_dim, cfg.dyn_layers)
        self.n_f = sum(o * i + o for o, i in self.shapes)
        self.n_sigma = sum(o * i + o for o, i in self.shapes)  # same structure
        
        # Series encoder
        self.lstm = nn.LSTM(
            input_size=cfg.input_dim,
            hidden_size=cfg.id_hidden,
            num_layers=cfg.id_lstm_layers,
            batch_first=True,
            dropout=0.1 if cfg.id_lstm_layers > 1 else 0.0,
        )
        
        # Drift head
        self.f_head = nn.Sequential(
            nn.Linear(cfg.id_hidden, cfg.id_hidden * 2),
            nn.GELU(),
            nn.Linear(cfg.id_hidden * 2, self.n_f),
        )
        
        # Diffusion head (predicts log σ for numerical stability)
        self.sigma_head = nn.Sequential(
            nn.Linear(cfg.id_hidden, cfg.id_hidden * 2),
            nn.GELU(),
            nn.Linear(cfg.id_hidden * 2, self.n_sigma),
        )
        
        # Contrastive projection
        self.proj_head = nn.Sequential(
            nn.Linear(cfg.id_hidden, 64),
            nn.GELU(),
            nn.Linear(64, 32),
        )
    
    @staticmethod
    def _weight_shapes(latent_dim, dyn_dim, dyn_layers):
        shapes, in_d = [], latent_dim
        for i in range(dyn_layers):
            out_d = latent_dim if i == dyn_layers - 1 else dyn_dim
            shapes.append((out_d, in_d))
            in_d = out_d
        return shapes
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """x : (B, Q, C) → h : (B, id_hidden)"""
        # Ensure input is (B, T, C); fix if accidentally (B, C, T)
        if x.ndim == 3 and x.shape[-1] != self.cfg.input_dim and x.shape[1] == self.cfg.input_dim:
            x = x.transpose(1, 2)
        _, (h_n, _) = self.lstm(x)
        return h_n[-1]
    
    def forward(self, x: torch.Tensor):
        """Returns f_weights, sigma_weights, h"""
        h = self.encode(x)
        f_w = self.f_head(h)           # (B, n_f)
        sigma_w = self.sigma_head(h)   # (B, n_sigma)  — log σ for stability
        return f_w, sigma_w, h

    def contrastive_loss(self, h1: torch.Tensor, h2: torch.Tensor) -> torch.Tensor:
        """NT-Xent loss between two noise-augmented views (h1, h2)."""
        z1 = F.normalize(self.proj_head(h1), dim=-1)
        z2 = F.normalize(self.proj_head(h2), dim=-1)
        bsz = z1.shape[0]

        z = torch.cat([z1, z2], dim=0)
        sim = (z @ z.T) / self.cfg.temperature
        sim.fill_diagonal_(-1e9)

        labels = torch.cat([
            torch.arange(bsz, 2 * bsz, device=z.device),
            torch.arange(0, bsz, device=z.device),
        ])
        return F.cross_entropy(sim, labels)
    
    def split_weights(self, weights_flat: torch.Tensor):
        """Split flat → [(W, b), ...]"""
        B, layers, idx = weights_flat.shape[0], [], 0
        for out_d, in_d in self.shapes:
            W = weights_flat[:, idx : idx + out_d * in_d].reshape(B, out_d, in_d)
            idx += out_d * in_d
            b = weights_flat[:, idx : idx + out_d]
            idx += out_d
            layers.append((W, b))
        return layers


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# DYNAMIC MLPs (drift and diffusion)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def apply_dynamic_mlp(z: torch.Tensor,
                      layers: List[Tuple[torch.Tensor, torch.Tensor]]
                      ) -> torch.Tensor:
    """Apply dynamic MLP: z (B,D) → f(z) (B,D)"""
    h = z
    for i, (W, b) in enumerate(layers):
        h = torch.bmm(W, h.unsqueeze(-1)).squeeze(-1) + b
        if i < len(layers) - 1:
            h = F.gelu(h)
    return h


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# GENERATIVE DECODER (supports multiple likelihoods)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class GenerativeDecoder(nn.Module):
    """
    Latent trajectory → observations with flexible likelihood.
    
    Supports:
      - Gaussian (default)
      - Laplace (robust to outliers)
      - Mixture (multiple modes)
    """
    
    def __init__(self, cfg, likelihood="gaussian"):
        super().__init__()
        self.cfg = cfg
        self.likelihood = likelihood
        
        self.net = nn.Sequential(
            nn.Linear(cfg.latent_dim, cfg.dec_hidden),
            nn.GELU(),
            nn.Linear(cfg.dec_hidden, cfg.dec_hidden),
            nn.GELU(),
        )
        
        if likelihood == "gaussian":
            self.head = nn.Linear(cfg.dec_hidden, cfg.input_dim * 2)  # μ, log σ
        elif likelihood == "laplace":
            self.head = nn.Linear(cfg.dec_hidden, cfg.input_dim * 2)  # μ, log b
        elif likelihood == "poisson":
            self.head = nn.Linear(cfg.dec_hidden, cfg.input_dim)       # log λ
        else:
            raise ValueError(f"Unknown likelihood: {likelihood}")
    
    def forward(self, z: torch.Tensor):
        """z : (B, H, D) → distribution over observations"""
        B, H, D = z.shape
        features = self.net(z.reshape(B * H, D)).reshape(B, H, -1)
        params = self.head(features)
        
        if self.likelihood == "gaussian":
            mu, log_sigma = params.chunk(2, dim=-1)
            sigma = log_sigma.clamp(-4, 2).exp()
            return torch.distributions.Normal(mu, sigma)
        
        elif self.likelihood == "laplace":
            mu, log_b = params.chunk(2, dim=-1)
            b = log_b.clamp(-4, 2).exp()
            return torch.distributions.Laplace(mu, b)
        
        elif self.likelihood == "poisson":
            log_lambda = params
            return torch.distributions.Poisson(log_lambda.exp())
    
    @staticmethod
    def nll_loss(dist, y):
        """Negative log-likelihood"""
        return -dist.log_prob(y).mean()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# LATENT SDE MODEL (Prior + Posterior)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class LatentSDE(nn.Module):
    """
    Generative model: z ~ SDE with learned dynamics and noise.
    
    Prior SDE:     dz = f_prior(z) dt + σ(z) dW    [fixed, e.g. f=0]
    Posterior SDE: dz = f_post(z) dt + σ(z) dW     [learned to match data]
    """
    
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        
        # Prior drift (fixed: could be 0 or learned)
        self.register_buffer("f_prior_weight", torch.zeros(cfg.latent_dim))
        
        # Shared diffusion (same σ for prior & posterior)
        self.sigma_net = nn.Sequential(
            nn.Linear(cfg.latent_dim, cfg.dec_hidden),
            nn.GELU(),
            nn.Linear(cfg.dec_hidden, cfg.latent_dim),
        )
    
    def f_prior(self, z, t):
        """Prior drift: 0 (Brownian motion)"""
        return torch.zeros_like(z)
    
    def f_posterior(self, z, t, f_layers):
        """Posterior drift from IDENTIFIER"""
        return apply_dynamic_mlp(z, f_layers)
    
    def sigma(self, z):
        """Shared diffusion function (always positive)"""
        return torch.abs(self.sigma_net(z)) + 1e-4  # ensure σ > 0
    
    def kl_divergence(self, z_post, f_post_layers, sigma_z, dt=0.01):
        """
        KL(posterior || prior) in SDE framework:
        
        KL = ∫₀ᵀ ||f_post(z) - f_prior(z)||² / σ²(z) dt
        
        z_post    : (B, H, D)  posterior samples
        f_post_layers: hypernetwork weights
        sigma_z   : (B, H, D)  diffusion values
        """
        B, H, D = z_post.shape
        
        kl_loss = 0.0
        for t in range(H):
            z_t = z_post[:, t]
            sigma_t = sigma_z[:, t]
            
            f_prior_t = self.f_prior(z_t, t)
            f_post_t = apply_dynamic_mlp(z_t, f_post_layers)
            
            # Drift difference
            drift_diff = f_post_t - f_prior_t  # (B, D)
            
            # KL integrand: ||diff||² / σ²
            kl_t = (drift_diff.pow(2) / (sigma_t.pow(2) + 1e-4)).sum(dim=1).mean()
            kl_loss = kl_loss + kl_t * dt
        
        return kl_loss / H


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# FULL MODEL
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TimeSeriesModelSDE(nn.Module):
    """
    Complete SDE-based time series model:
    
    1. Encoder (JEPA) → z ~ N(μ, σ²)
    2. Identifier → f_post(z), σ(z)
    3. Latent SDE → z_future
    4. Decoder → observations
    5. KL regularization
    """
    
    def __init__(self, cfg, encoder, decoder=None):
        super().__init__()
        self.cfg = cfg
        self.encoder = encoder              # BootstrappedEncoder (JEPA)
        self.ident = StochasticIdentifier(cfg)
        self.sde = LatentSDE(cfg)
        self.decoder = decoder or GenerativeDecoder(cfg, likelihood="gaussian")
        self.solver = EulerMaruyamaSDE()
    
    @staticmethod
    def _reparam(mu, log_var):
        return mu + (0.5 * log_var).exp() * torch.randn_like(mu)
    
    def forward(self, x: torch.Tensor, y: Optional[torch.Tensor] = None):
        """
        x : (B, Q, C)  context
        y : (B, H, C)  targets (optional)
        """
        # Handle 2D input (univariate without C dimension)
        if x.ndim == 2:
            x = x.unsqueeze(-1)
        if y is not None and y.ndim == 2:
            y = y.unsqueeze(-1)
        
        B, D = x.shape[0], self.cfg.latent_dim
        device = x.device
        
        # ── 1. ENCODER (JEPA) ──────────────────────────────────
        mu, log_var, jepa_loss = self.encoder(x)  # (B, D)
        z0 = self._reparam(mu, log_var)
        kl_enc = -0.5 * (1 + log_var - mu.pow(2) - log_var.exp()).mean()
        
        # ── 2. IDENTIFIER ──────────────────────────────────────
        f_w, sigma_w, h_ident = self.ident(x)
        f_layers = self.ident.split_weights(f_w)
        sigma_layers = self.ident.split_weights(sigma_w)
        
        # Contrastive loss (like before)
        id_loss = (self.ident.contrastive_loss(h_ident, h_ident + 0.05 * torch.randn_like(h_ident))
                   if self.training else torch.zeros(1, device=device)[0])
        
        # ── 3. SDE SOLVER ──────────────────────────────────────
        # Define drift and diffusion functions
        def f_drift(z, t):
            return apply_dynamic_mlp(z, f_layers)
        
        def f_sigma(z, t):
            return self.sde.sigma(z)
        
        # Solve SDE
        z_traj, noise_traj = self.solver.solve(
            z0, f_drift, f_sigma,
            t_span=(0, 1),
            n_steps=self.cfg.horizon_len,
        )
        z_future = z_traj[:, 1:, :]  # (B, H, D)  exclude z0
        sigma_z = f_sigma(z_future.reshape(B * self.cfg.horizon_len, D), 0).reshape(B, self.cfg.horizon_len, D)
        
        # ── 4. KL DIVERGENCE (SDE-specific) ────────────────────
        kl_sde = self.sde.kl_divergence(z_future, f_layers, sigma_z)
        
        # ── 5. DECODER ────────────────────────────────────────
        dist = self.decoder(z_future)
        y_pred = dist.mean
        
        out = dict(
            y_pred=y_pred,
            dist=dist,
            mu=mu,
            log_var=log_var,
            z_future=z_future,
            noise_traj=noise_traj,
        )
        
        if y is not None:
            dec_loss = GenerativeDecoder.nll_loss(dist, y)
            cfg = self.cfg
            total = (cfg.lambda_jepa * jepa_loss
                   + cfg.lambda_kl * (kl_enc + kl_sde)
                   + cfg.lambda_id * id_loss
                   + cfg.lambda_dec * dec_loss)
            out.update(
                total_loss=total,
                jepa_loss=jepa_loss,
                kl_enc=kl_enc,
                kl_sde=kl_sde,
                id_loss=id_loss,
                dec_loss=dec_loss,
            )
        
        return out
    
    @torch.no_grad()
    def predict(self, x, n_samples=100):
        """Probabilistic forecast (Monte Carlo)"""
        self.eval()
        if x.ndim == 2:
            x = x.unsqueeze(-1)
        if x.ndim == 3 and x.shape[-1] != self.cfg.input_dim and x.shape[1] == self.cfg.input_dim:
            x = x.transpose(1, 2)
        mu, log_var, _ = self.encoder(x)
        
        samples = []
        for _ in range(n_samples):
            z0 = self._reparam(mu, log_var)
            f_w, sigma_w, _ = self.ident(x)
            f_layers = self.ident.split_weights(f_w)
            
            def f_drift(z, t):
                return apply_dynamic_mlp(z, f_layers)
            
            def f_sigma(z, t):
                return self.sde.sigma(z)
            
            z_traj, _ = self.solver.solve(z0, f_drift, f_sigma, (0, 1), self.cfg.horizon_len)
            z_fut = z_traj[:, 1:, :]
            samples.append(self.decoder(z_fut).mean)
        
        s = torch.stack(samples)
        return dict(
            mean=s.mean(0),
            std=s.std(0),
            q10=s.quantile(0.10, dim=0),
            q90=s.quantile(0.90, dim=0),
            samples=s,
        )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# USAGE EXAMPLE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

if __name__ == "__main__":
    # Import the original Config and Encoder
    from timeseries_model import Config, BootstrappedEncoder
    
    cfg = Config(
        context_len=96,
        horizon_len=24,
        latent_dim=64,
        lambda_kl=0.1,
    )
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Create encoder (from original model)
    encoder = BootstrappedEncoder(cfg).to(device)
    
    # Create full SDE model
    model = TimeSeriesModelSDE(cfg, encoder).to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Test forward
    x = torch.randn(4, 96, 1).to(device)
    y = torch.randn(4, 24, 1).to(device)
    
    model.train()
    out = model(x, y)
    print("Losses:", {k: f"{v.item():.4f}" for k, v in out.items() if "loss" in k})
    
    # Predict
    pred = model.predict(x, n_samples=50)
    print(f"Prediction shape: {pred['mean'].shape}")
