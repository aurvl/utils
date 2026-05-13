"""
╔══════════════════════════════════════════════════════════════════╗
║         INTELLIGENT TIME SERIES MODEL                            ║
║                                                                  ║
║  Architecture:                                                   ║
║    X ──┬──► Encoder (Bootstrapped JEPA) ──► z ~ N(μ,σ)           ║
║        │                                       │                 ║
║        └──► Identifier (Hypernetwork) ──► f weights              ║
║                                          │     │                 ║
║                                          └──►  f(z)              ║
║                                                │                 ║
║                                           Euler ODE              ║
║                                                │                 ║
║                                           Decoder ──► ŷ dist     ║
║                                                                  ║
║  Losses:                                                         ║
║    1. JEPA loss   — latent-space patch prediction                ║
║    2. KL loss     — regularize z posterior                       ║
║    3. ID loss     — NT-Xent contrastive on Identifier            ║
║    4. Decoder NLL — Gaussian log-likelihood                      ║
╚══════════════════════════════════════════════════════════════════╝
"""

import math #noqa
from dataclasses import dataclass, field #noqa
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CONFIG
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@dataclass
class Config:
    # ── Data ──────────────────────────────────────────────────────
    input_dim:      int   = 1     # C  number of variates
    context_len:    int   = 96    # Q  past timesteps (context)
    horizon_len:    int   = 24    # H  future timesteps to predict

    # ── Encoder (Bootstrapped JEPA) ───────────────────────────────
    patch_size:       int   = 8      # tokens of size patch_size × C
    d_model:          int   = 64     # transformer width
    n_heads:          int   = 4      # attention heads
    enc_layers:       int   = 4      # transformer depth
    dropout:          float = 0.1
    latent_dim:       int   = 64     # z dimension
    bootstrap_b:      int   = 8      # b  bootstrap runs per batch
    mask_ratio:       float = 0.75   # fraction of patches masked
    ema_decay:        float = 0.996  # momentum for EMA target encoder

    # ── Identifier (Hypernetwork) ─────────────────────────────────
    id_hidden:        int   = 128    # LSTM hidden size
    id_lstm_layers:   int   = 2
    dyn_layers:       int   = 2      # R  depth of dynamics MLP f
    dyn_dim:          int   = 64     # D  width of dynamics MLP f

    # ── Decoder ───────────────────────────────────────────────────
    dec_hidden:       int   = 128

    # ── Loss weights ──────────────────────────────────────────────
    lambda_jepa:  float = 1.0
    lambda_kl:    float = 0.1     # start small, can anneal up
    lambda_id:    float = 0.5
    lambda_dec:   float = 1.0
    temperature:  float = 0.07   # NT-Xent temperature

    # ── Training ──────────────────────────────────────────────────
    lr:           float = 1e-4
    weight_decay: float = 1e-5
    batch_size:   int   = 32
    grad_clip:    float = 1.0


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# HELPERS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _weight_shapes(latent_dim: int, dyn_dim: int, dyn_layers: int
                   ) -> List[Tuple[int, int]]:
    """(out, in) per layer of the dynamics MLP f."""
    shapes, in_d = [], latent_dim
    for i in range(dyn_layers):
        out_d = latent_dim if i == dyn_layers - 1 else dyn_dim
        shapes.append((out_d, in_d))
        in_d = out_d
    return shapes


def _apply_dynamic_mlp(z: torch.Tensor,
                       layers: List[Tuple[torch.Tensor, torch.Tensor]]
                       ) -> torch.Tensor:
    """
    z      : (B, D)
    layers : [(W:(B,out,in), b:(B,out)), ...]
    Returns: (B, D)  — same shape as z
    """
    h = z
    for i, (W, b) in enumerate(layers):
        h = torch.bmm(W, h.unsqueeze(-1)).squeeze(-1) + b
        if i < len(layers) - 1:
            h = F.gelu(h)
    return h


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PATCH EMBEDDING
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class PatchEmbedding(nn.Module):
    """(B, T, C) → (B, n_patches, d_model)"""

    def __init__(self, input_dim: int, patch_size: int, d_model: int):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Linear(input_dim * patch_size, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        n = T // self.patch_size
        x = x[:, :n * self.patch_size].reshape(B, n, self.patch_size * C)
        return self.norm(self.proj(x))


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# JEPA PREDICTOR
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class JEPAPredictor(nn.Module):
    """
    Lightweight Transformer decoder.
      Memory  = visible patch embeddings from online encoder.
      Queries = positional embeddings of masked positions.
      Output  = predicted EMA representations of masked patches.
    """

    def __init__(self, d_model: int, n_heads: int = 4, n_layers: int = 2):
        super().__init__()
        layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=n_heads,
            dim_feedforward=d_model * 2,
            dropout=0.0, batch_first=True, norm_first=True,
        )
        self.decoder   = nn.TransformerDecoder(layer, num_layers=n_layers)
        self.out_proj  = nn.Linear(d_model, d_model)

    def forward(self, context: torch.Tensor,
                query_pos: torch.Tensor) -> torch.Tensor:
        # context  : (B, n_visible, d)
        # query_pos: (B, n_masked,  d)  positional embeddings of masked spots
        return self.out_proj(self.decoder(query_pos, context))


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# BOOTSTRAPPED JEPA ENCODER
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class BootstrappedEncoder(nn.Module):
    """
    For each of b bootstrap iterations:
      1. Draw a random mask (mask_ratio of patches hidden).
      2. Online encoder processes only the VISIBLE patches.
      3. JEPA predictor reconstructs the MASKED patch representations.
      4. EMA encoder (no-grad, momentum-updated) provides targets.
      5. JEPA loss = MSE(predicted, EMA_target) on masked positions.

    Final z distribution = mean of (μ_k, log_var_k) across b runs.
    """

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        P = cfg.context_len // cfg.patch_size   # number of patches

        # ── Online encoder ────────────────────────────────────────
        self.patch_embed = PatchEmbedding(cfg.input_dim, cfg.patch_size, cfg.d_model)
        self.pos_embed   = nn.Parameter(torch.randn(1, P, cfg.d_model) * 0.02)
        enc_layer        = nn.TransformerEncoderLayer(
            d_model=cfg.d_model, nhead=cfg.n_heads,
            dim_feedforward=cfg.d_model * 4,
            dropout=cfg.dropout, batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=cfg.enc_layers)
        self.predictor   = JEPAPredictor(cfg.d_model)

        # ── EMA (target) encoder ─────────────────────────────────
        ema_enc_layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model, nhead=cfg.n_heads,
            dim_feedforward=cfg.d_model * 4,
            dropout=0.0, batch_first=True, norm_first=True,
        )
        self.ema_patch_embed = PatchEmbedding(cfg.input_dim, cfg.patch_size, cfg.d_model)
        self.ema_transformer = nn.TransformerEncoder(ema_enc_layer, num_layers=cfg.enc_layers)
        self._no_grad_ema()
        self._sync_ema()            # initialise EMA = online

        # ── z projection ─────────────────────────────────────────
        self.to_mu      = nn.Linear(cfg.d_model, cfg.latent_dim)
        self.to_log_var = nn.Linear(cfg.d_model, cfg.latent_dim)

        self.P = P

    # ── EMA helpers ──────────────────────────────────────────────

    def _no_grad_ema(self):
        for p in (*self.ema_patch_embed.parameters(),
                  *self.ema_transformer.parameters()):
            p.requires_grad_(False)

    @torch.no_grad()
    def _sync_ema(self):
        for src, dst in zip(
            (*self.patch_embed.parameters(), *self.transformer.parameters()),
            (*self.ema_patch_embed.parameters(), *self.ema_transformer.parameters()),
        ):
            dst.data.copy_(src.data)

    @torch.no_grad()
    def update_ema(self):
        """Call once per optimiser step."""
        tau = self.cfg.ema_decay
        for src, dst in zip(
            (*self.patch_embed.parameters(), *self.transformer.parameters()),
            (*self.ema_patch_embed.parameters(), *self.ema_transformer.parameters()),
        ):
            dst.data.mul_(tau).add_(src.data, alpha=1.0 - tau)

    # ── Masking ──────────────────────────────────────────────────

    def _sample_mask(self, device: torch.device) -> torch.BoolTensor:
        n_mask = int(self.P * self.cfg.mask_ratio)
        ids    = torch.randperm(self.P, device=device)
        mask   = torch.zeros(self.P, dtype=torch.bool, device=device)
        mask[ids[:n_mask]] = True
        return mask   # True = masked / hidden

    # ── Forward ──────────────────────────────────────────────────

    def forward(self, x: torch.Tensor
                ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        x : (B, Q, C)
        Returns
            mu        (B, latent_dim)
            log_var   (B, latent_dim)
            jepa_loss scalar
        """
        B, device = x.shape[0], x.device

        # EMA targets — computed once for all bootstrap runs
        with torch.no_grad():
            ema_tok = self.ema_patch_embed(x) + self.pos_embed   # (B, P, d)
            ema_h   = self.ema_transformer(ema_tok)               # (B, P, d)
            ema_h   = F.layer_norm(ema_h, [self.cfg.d_model])     # normalize targets

        all_mu, all_lv, jepa_terms = [], [], []

        for _ in range(self.cfg.bootstrap_b):
            mask    = self._sample_mask(device)          # (P,)
            visible = ~mask

            # Online encoder on visible patches
            tok     = self.patch_embed(x) + self.pos_embed  # (B, P, d)
            h_vis   = self.transformer(tok[:, visible])      # (B, n_vis, d)

            # JEPA predictor: predict masked patch representations
            q_pos  = self.pos_embed.expand(B, -1, -1)[:, mask]   # (B, n_mask, d)
            h_pred = self.predictor(h_vis, q_pos)                  # (B, n_mask, d)

            # JEPA loss (smooth L1 is more robust than MSE to outliers)
            jepa_terms.append(
                F.smooth_l1_loss(h_pred, ema_h[:, mask].detach())
            )

            # Latent distribution from visible tokens
            h_agg = h_vis.mean(dim=1)                  # (B, d_model)
            all_mu.append(self.to_mu(h_agg))
            all_lv.append(self.to_log_var(h_agg))

        mu        = torch.stack(all_mu).mean(0)        # (B, latent_dim)
        log_var   = torch.stack(all_lv).mean(0)
        jepa_loss = torch.stack(jepa_terms).mean()

        return mu, log_var, jepa_loss


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# IDENTIFIER  (Hypernetwork)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class Identifier(nn.Module):
    """
    X  →  LSTM  →  h  →  weights of the dynamics MLP f
                   h  →  (proj head)  →  contrastive representation

    Contrastive loss:  NT-Xent between two noise-augmented views of the
    same series.  The intuition: two windows from the same process should
    produce similar dynamic weights; two different processes should not.
    """

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg    = cfg
        self.shapes = _weight_shapes(cfg.latent_dim, cfg.dyn_dim, cfg.dyn_layers)
        self.n_w    = sum(o * i + o for o, i in self.shapes)   # total weight count

        # ── Series encoder ───────────────────────────────────────
        self.lstm = nn.LSTM(
            input_size=cfg.input_dim,
            hidden_size=cfg.id_hidden,
            num_layers=cfg.id_lstm_layers,
            batch_first=True,
            dropout=0.1 if cfg.id_lstm_layers > 1 else 0.0,
        )

        # ── Weight generator ─────────────────────────────────────
        self.weight_head = nn.Sequential(
            nn.Linear(cfg.id_hidden, cfg.id_hidden * 2),
            nn.GELU(),
            nn.Linear(cfg.id_hidden * 2, self.n_w),
        )

        # ── Contrastive projection head ──────────────────────────
        self.proj_head = nn.Sequential(
            nn.Linear(cfg.id_hidden, 64),
            nn.GELU(),
            nn.Linear(64, 32),
        )

    # ── Encode series ────────────────────────────────────────────

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """x : (B, Q, C) → h : (B, id_hidden)"""
        # Ensure input is (B, T, C); fix if accidentally (B, C, T)
        if x.ndim == 3 and x.shape[-1] != self.cfg.input_dim and x.shape[1] == self.cfg.input_dim:
            x = x.transpose(1, 2)  # (B, C, T) → (B, T, C)
        _, (h_n, _) = self.lstm(x)
        return h_n[-1]

    # ── Forward ──────────────────────────────────────────────────

    def forward(self, x: torch.Tensor
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns
            weights_flat  (B, n_w)        flat weights for f
            h             (B, id_hidden)  series embedding
        """
        h = self.encode(x)
        return self.weight_head(h), h

    def split_weights(self, weights_flat: torch.Tensor
                      ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """Reshape flat weight vector → list of (W, b) per layer."""
        B, layers, idx = weights_flat.shape[0], [], 0
        for out_d, in_d in self.shapes:
            W    = weights_flat[:, idx : idx + out_d * in_d].reshape(B, out_d, in_d)
            idx += out_d * in_d
            b    = weights_flat[:, idx : idx + out_d]
            idx += out_d
            layers.append((W, b))
        return layers

    # ── NT-Xent contrastive loss ─────────────────────────────────

    def contrastive_loss(self, h1: torch.Tensor,
                         h2: torch.Tensor) -> torch.Tensor:
        """
        h1, h2 : (B, id_hidden) — two views of the same batch.
        Positive pairs: (i, i+B).  Negatives: everything else in 2B.
        """
        z1 = F.normalize(self.proj_head(h1), dim=-1)   # (B, 32)
        z2 = F.normalize(self.proj_head(h2), dim=-1)
        B  = z1.shape[0]

        z      = torch.cat([z1, z2], dim=0)             # (2B, 32)
        sim    = (z @ z.T) / self.cfg.temperature       # (2B, 2B)
        sim.fill_diagonal_(-1e9)                        # mask self-similarity

        labels = torch.cat([
            torch.arange(B, 2 * B, device=z1.device),
            torch.arange(0, B,     device=z1.device),
        ])
        return F.cross_entropy(sim, labels)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# EULER ODE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class EulerODE(nn.Module):
    """
    Discrete Euler integration of the latent dynamics:
        ẑ_{t+1} = ẑ_t + f(ẑ_t)
    where f is the dynamic MLP whose weights come from the Identifier.
    """

    def forward(self,
                z0:         torch.Tensor,
                dyn_layers: List[Tuple[torch.Tensor, torch.Tensor]],
                n_steps:    int) -> torch.Tensor:
        """
        z0 : (B, latent_dim)
        Returns (B, H, latent_dim)
        """
        z, traj = z0, []
        for _ in range(n_steps):
            z = z + _apply_dynamic_mlp(z, dyn_layers)
            traj.append(z)
        return torch.stack(traj, dim=1)   # (B, H, D)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PROBABILISTIC DECODER
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class ProbabilisticDecoder(nn.Module):
    """
    Latent trajectory → Gaussian(μ, σ) over future values.
    Training loss: NLL.   Evaluation metric: CRPS (see predict()).
    """

    def __init__(self, cfg: Config):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cfg.latent_dim, cfg.dec_hidden),
            nn.GELU(),
            nn.Linear(cfg.dec_hidden, cfg.dec_hidden),
            nn.GELU(),
            nn.Linear(cfg.dec_hidden, cfg.input_dim * 2),  # μ + log σ
        )

    def forward(self, z: torch.Tensor) -> Normal:
        """z : (B, H, D) → Normal over (B, H, C)"""
        B, H, D = z.shape
        out         = self.net(z.reshape(B * H, D)).reshape(B, H, -1)
        mu, log_sig = out.chunk(2, dim=-1)
        sigma       = log_sig.clamp(-4, 2).exp()
        return Normal(mu, sigma)

    @staticmethod
    def nll_loss(dist: Normal, y: torch.Tensor) -> torch.Tensor:
        return -dist.log_prob(y).mean()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# FULL MODEL
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TimeSeriesModel(nn.Module):

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg     = cfg
        self.encoder = BootstrappedEncoder(cfg)
        self.ident   = Identifier(cfg)
        self.ode     = EulerODE()
        self.decoder = ProbabilisticDecoder(cfg)

    # ── Reparameterisation trick ──────────────────────────────────

    @staticmethod
    def _reparam(mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        return mu + (0.5 * log_var).exp() * torch.randn_like(mu)

    @staticmethod
    def _kl(mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        """KL( N(μ,σ²) || N(0,1) )"""
        return -0.5 * (1 + log_var - mu.pow(2) - log_var.exp()).mean()

    # ── Identifier contrastive step ───────────────────────────────

    def _id_contrastive(self, x: torch.Tensor) -> torch.Tensor:
        """Two 5%-noise augmentations → NT-Xent."""
        scale = x.std(dim=1, keepdim=True) * 0.05
        _, h1 = self.ident(x + torch.randn_like(x) * scale)
        _, h2 = self.ident(x + torch.randn_like(x) * scale)
        return self.ident.contrastive_loss(h1, h2)

    # ── Forward ──────────────────────────────────────────────────

    def forward(self,
                x: torch.Tensor,
                y: Optional[torch.Tensor] = None) -> dict:
        """
        x : (B, Q, C)  context window
        y : (B, H, C)  ground truth (optional; needed for losses)

        Returns dict with at minimum:
            y_pred, dist, mu, log_var, weights_flat
        If y is provided, adds:
            total_loss, jepa_loss, kl_loss, id_loss, dec_loss
        """
        if x.ndim == 2:
            x = x.unsqueeze(-1)  # (B, Q) → (B, Q, 1)
        if y is not None and y.ndim == 2:
            y = y.unsqueeze(-1)  # (B, H) → (B, H, 1)
        
        # ── 1. ENCODER ───────────────────────────────────────────
        enc_out = self.encoder(x)                        # returns (mu, log_var, jepa_loss)
        if isinstance(enc_out, tuple) and len(enc_out) == 3:
            mu, log_var, jepa_loss = enc_out
        else:
            mu, log_var = enc_out[:2] if isinstance(enc_out, tuple) else enc_out
            jepa_loss = x.new_zeros(())
        z0 = self._reparam(mu, log_var)                  # (B, D)
        kl = self._kl(mu, log_var)

        # ── 2. IDENTIFIER ────────────────────────────────────────
        ident_out = self.ident(x)                        # returns (weights_flat, h)
        if isinstance(ident_out, tuple) and len(ident_out) >= 2:
            weights_flat = ident_out[0]
        else:
            weights_flat = ident_out
        dyn_layers      = self.ident.split_weights(weights_flat)
        id_loss = (self._id_contrastive(x)
                   if self.training
                   else x.new_zeros(()))

        # ── 3. ODE ───────────────────────────────────────────────
        z_fut = self.ode(z0, dyn_layers, self.cfg.horizon_len)  # (B, H, D)

        # ── 4. DECODER ───────────────────────────────────────────
        dist   = self.decoder(z_fut)
        y_pred = dist.mean                               # point forecast

        out = dict(
            y_pred=y_pred, dist=dist,
            mu=mu, log_var=log_var, weights_flat=weights_flat,
        )

        if y is not None:
            dec_loss  = ProbabilisticDecoder.nll_loss(dist, y)
            cfg       = self.cfg
            total     = (cfg.lambda_jepa * jepa_loss
                       + cfg.lambda_kl   * kl
                       + cfg.lambda_id   * id_loss
                       + cfg.lambda_dec  * dec_loss)
            out.update(
                total_loss=total,
                jepa_loss=jepa_loss,
                kl_loss=kl,
                id_loss=id_loss,
                dec_loss=dec_loss,
            )

        return out

    # ── Probabilistic forecast ────────────────────────────────────

    @torch.no_grad()
    def predict(self, x: torch.Tensor, n_samples: int = 200) -> dict:
        """
        Monte-Carlo probabilistic forecast.
        Returns mean, std, 10th/90th quantiles, and raw samples.
        """
        self.eval()
        if x.ndim == 2:
            x = x.unsqueeze(-1)  # (B, Q) → (B, Q, 1)
        
        # Robust unpacking: encoder may return 2 or 3 values
        enc_out = self.encoder(x)
        if isinstance(enc_out, tuple) and len(enc_out) == 3:
            mu, log_var, _ = enc_out
        else:
            mu, log_var = enc_out[:2] if isinstance(enc_out, tuple) else (enc_out, torch.zeros_like(enc_out))
        
        w_flat, _ = self.ident(x)
        dyn_layers = self.ident.split_weights(w_flat)

        samples = []
        for _ in range(n_samples):
            z0 = self._reparam(mu, log_var)
            zf = self.ode(z0, dyn_layers, self.cfg.horizon_len)
            samples.append(self.decoder(zf).mean)

        s = torch.stack(samples)        # (S, B, H, C)
        return dict(
            mean  = s.mean(0),
            std   = s.std(0),
            q10   = s.quantile(0.10, dim=0),
            q90   = s.quantile(0.90, dim=0),
            samples = s,
        )

    # ── Explain: decode Identifier weights ───────────────────────

    @torch.no_grad()
    def explain(self, x: torch.Tensor) -> dict:
        """
        Return the predicted dynamics weights and their L2 norms
        as a simple proxy for 'how active is each dynamic layer'.
        """
        self.eval()
        w_flat, h = self.ident(x)
        layers    = self.ident.split_weights(w_flat)
        norms = {f"layer_{i}_W_norm": W.norm(dim=(1, 2)).mean().item()
                 for i, (W, _) in enumerate(layers)}
        return dict(embedding=h.cpu(), weight_norms=norms)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TRAINING LOOP
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def train(
    model:        TimeSeriesModel,
    train_loader: torch.utils.data.DataLoader,
    val_loader:   torch.utils.data.DataLoader,
    cfg:          Config,
    n_epochs:     int = 50,
    device:       str = "cpu",
) -> None:

    model = model.to(device)
    opt   = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs)

    best_val = float("inf")

    for epoch in range(1, n_epochs + 1):
        model.train()
        logs: dict = {k: [] for k in ["total", "jepa", "kl", "id", "dec"]}

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)

            opt.zero_grad()
            out = model(x, y)
            out["total_loss"].backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()
            model.encoder.update_ema()   # momentum update of target encoder

            logs["total"].append(out["total_loss"].item())
            logs["jepa"].append(out["jepa_loss"].item())
            logs["kl"].append(out["kl_loss"].item())
            logs["id"].append(out["id_loss"].item()
                              if torch.is_tensor(out["id_loss"])
                              else float(out["id_loss"]))
            logs["dec"].append(out["dec_loss"].item())

        sched.step()

        # ── Validation ───────────────────────────────────────────
        model.eval()
        val_total = []
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                val_total.append(model(x, y)["total_loss"].item())

        val_mean = float(np.mean(val_total))
        flag     = "  ◀ best" if val_mean < best_val else ""
        best_val = min(best_val, val_mean)

        print(
            f"Ep {epoch:3d}/{n_epochs} │ "
            f"total={np.mean(logs['total']):.4f} │ "
            f"jepa={np.mean(logs['jepa']):.4f} │ "
            f"kl={np.mean(logs['kl']):.4f} │ "
            f"id={np.mean(logs['id']):.4f} │ "
            f"dec={np.mean(logs['dec']):.4f} │ "
            f"val={val_mean:.4f}{flag}"
        )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# QUICK SANITY CHECK
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

if __name__ == "__main__":
    cfg    = Config()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = TimeSeriesModel(cfg).to(device)
    n_p   = sum(p.numel() for p in model.parameters())
    print(f"Device     : {device}")
    print(f"Parameters : {n_p:,}")
    print(f"  Encoder  : {sum(p.numel() for p in model.encoder.parameters()):,}")
    print(f"  Identifier: {sum(p.numel() for p in model.ident.parameters()):,}")
    print(f"  Decoder  : {sum(p.numel() for p in model.decoder.parameters()):,}")
    print()

    # ── Single forward-pass test ─────────────────────────────────
    B, Q, H, C = 50, cfg.context_len, cfg.horizon_len, cfg.input_dim
    x_t = torch.randn(B, Q, C).to(device)
    y_t = torch.randn(B, H, C).to(device)

    model.train()
    out = model(x_t, y_t)
    print("Train losses:")
    for k, v in out.items():
        if "loss" in k:
            print(f"  {k:15s}: {v.item():.5f}")

    # Probabilistic prediction
    pred = model.predict(x_t, n_samples=50)
    print(f"\nPrediction shape : {pred['mean'].shape}")   # (B, H, C)
    print(f"Uncertainty std  : {pred['std'].mean().item():.4f}")

    # Explainability
    expl = model.explain(x_t)
    print(f"\nIdentifier weight norms : {expl['weight_norms']}")

    # ── Mini training run ────────────────────────────────────────
    N  = 500
    ds = torch.utils.data.TensorDataset(
        torch.randn(N, Q, C),
        torch.randn(N, H, C),
    )
    tr, vl = torch.utils.data.random_split(ds, [400, 100])
    tr_dl  = torch.utils.data.DataLoader(tr, batch_size=cfg.batch_size, shuffle=True)
    vl_dl  = torch.utils.data.DataLoader(vl, batch_size=cfg.batch_size)

    print("\n── Training (5 epochs) ──")
    train(model, tr_dl, vl_dl, cfg, n_epochs=5, device=device)
