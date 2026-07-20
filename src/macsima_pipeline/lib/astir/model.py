"""Clean-room Astir cell-type model (PyTorch), GPU-resident and minibatched.

Implements the generative model of Geuenich et al. (Cell Systems 2021) from the
published equations. Discrete cell-type latent is marginalized exactly (a weighted
sum over C+1 classes), so there is no reparameterization here.

Model (per cell n, marker g, class c; class C is the all-zero "Other"):
    loc[n,c,g] = exp( baseline_g(design_n) + rho[g,c] * exp(log_delta[g,c]) )
    y[n] | c   ~ LowRankMultivariateNormal(loc[n,c],
                     cov_factor = sigma_g * rho[g,c] * sigmoid(p[g,c]),   (rank 1)
                     cov_diag   = sigma_g^2 * (1 - (rho*sigmoid(p))^2) + eps)
where baseline = design @ mu.T and sigma = exp(log_sigma). A recognition network
gives q(c | z-scored y); the loss is the negative variational-mixture ELBO plus a
Dirichlet prior on the global mixture weights.
"""

from __future__ import annotations

import logging

import numpy as np
import torch
from torch import nn
from torch.distributions import Dirichlet, LowRankMultivariateNormal

from . import data as _data
from .io.marker_yaml import build_rho
from .recognition import TypeRecognitionNet

log = logging.getLogger(__name__)

_VAR_EPS = 1e-6


def _resolve_device(device: str | None) -> str:
    if device in (None, "auto"):
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


class _Model(nn.Module):
    """Learnable parameters + generative density + ELBO."""

    def __init__(self, rho: np.ndarray, n_design: int, mu0: np.ndarray, log_sigma0: np.ndarray,
                 recog_hidden: int, seed: int) -> None:
        super().__init__()
        g, cp1 = rho.shape
        self.register_buffer("rho", torch.tensor(rho, dtype=torch.float32))
        torch.manual_seed(seed)
        self.mu = nn.Parameter(torch.tensor(np.repeat(mu0[:, None], n_design, axis=1), dtype=torch.float32))
        self.log_sigma = nn.Parameter(torch.tensor(log_sigma0, dtype=torch.float32))
        self.log_delta = nn.Parameter(0.1 * torch.randn(g, cp1))
        self.p = nn.Parameter(torch.zeros(g, cp1))
        self.alpha_logits = nn.Parameter(torch.ones(cp1))
        self.recog = TypeRecognitionNet(g, cp1, recog_hidden)

    def _dist(self, design_batch: torch.Tensor) -> LowRankMultivariateNormal:
        delta = torch.exp(self.log_delta)               # (G, C+1)
        mean1 = delta * self.rho                          # (G, C+1)
        mean2 = design_batch @ self.mu.t()               # (B, G)
        mean_total = mean2.unsqueeze(-1) + mean1.unsqueeze(0)     # (B, G, C+1)
        loc = torch.exp(mean_total).permute(0, 2, 1)             # (B, C+1, G)
        p_sig = torch.sigmoid(self.p)                    # (G, C+1)
        sigma = torch.exp(self.log_sigma)                # (G,)
        cov_factor = (sigma.unsqueeze(-1) * self.rho * p_sig).t().unsqueeze(-1)  # (C+1, G, 1)
        cov_diag = (sigma.unsqueeze(-1) ** 2 * (1.0 - (self.rho * p_sig) ** 2)).t() + _VAR_EPS  # (C+1, G)
        return LowRankMultivariateNormal(loc, cov_factor=cov_factor, cov_diag=cov_diag)

    def neg_elbo(self, y_batch: torch.Tensor, x_batch: torch.Tensor, design_batch: torch.Tensor,
                 n_total: int) -> torch.Tensor:
        dist = self._dist(design_batch)
        log_p = dist.log_prob(y_batch.unsqueeze(1))              # (B, C+1)
        gamma, log_gamma = self.recog(x_batch)                   # (B, C+1)
        log_alpha = torch.log_softmax(self.alpha_logits, dim=0)  # (C+1,)
        recon = (gamma * (log_p + log_alpha.unsqueeze(0) - log_gamma)).sum(dim=1)  # (B,)
        alpha = torch.softmax(self.alpha_logits, dim=0)
        conc = torch.full_like(alpha, float(alpha.shape[0]))
        mix_prior = Dirichlet(conc).log_prob(alpha)
        return -(recon.mean() + mix_prior / n_total)

    @torch.no_grad()
    def proba(self, x: torch.Tensor, batch_size: int) -> np.ndarray:
        self.eval()
        out = [self.recog(x[s:s + batch_size])[0].cpu().numpy() for s in range(0, x.shape[0], batch_size)]
        return np.concatenate(out, axis=0)


class AstirCellType:
    """Fit the Astir model and assign per-cell type probabilities.

    Feed RAW intensities to `fit`; normalization is internal. `predict_proba` returns
    (N, C+1) probabilities (last column "Other"); `predict` returns hard labels with
    "Unknown" where the max probability is below a threshold.
    """

    def __init__(self, rho: np.ndarray, class_names: list[str], feature_names: list[str], *,
                 recog_hidden: int = 20, device: str | None = None, seed: int = 0) -> None:
        self.rho = np.asarray(rho, dtype=np.float32)
        self.class_names = list(class_names)
        self.feature_names = list(feature_names)
        self.recog_hidden = recog_hidden
        self.device = _resolve_device(device)
        self.seed = seed
        self._proba: np.ndarray | None = None
        self.losses: list[float] = []
        self.converged = False
        self.batch_names: list[str] = []

    @classmethod
    def from_marker_dict(cls, signature: dict[str, list[str]], feature_names: list[str], **kw) -> "AstirCellType":
        rho, class_names = build_rho(signature, feature_names, include_other=True)
        return cls(rho, class_names, feature_names, **kw)

    # ---- training ----

    def fit(self, x_raw: np.ndarray, *, batch: np.ndarray | None = None, cofactor: float = 5.0,
            winsorize: tuple[float, float] = (0.0, 99.9), max_epochs: int = 50, lr: float = 1e-3,
            batch_size: int = 16384, n_init: int = 5, n_init_epochs: int = 5, delta_loss: float = 1e-3,
            patience: int = 5, include_batch: bool = True, precision: str = "fp32") -> "AstirCellType":
        raw = np.asarray(x_raw, dtype=np.float32)
        n = raw.shape[0]
        if precision == "bf16":  # keep the Gaussian density in fp32; bf16 recog not yet wired
            log.warning("astir precision='bf16' not yet enabled; using fp32")

        y_np = _data.normalize_for_astir(raw, cofactor, winsorize)
        x_np = _data.zscore(y_np)
        design_np, self.batch_names = _data.build_design(n, batch, include_batch)
        n_design = design_np.shape[1]
        mu0, log_sigma0 = _data.mu_sigma_init(y_np)

        y = torch.tensor(y_np, device=self.device)
        x = torch.tensor(x_np, device=self.device)
        design = torch.tensor(design_np, device=self.device)

        # Short multi-init warmup: keep the lowest-loss basin.
        best_state, best_loss = None, np.inf
        for i in range(max(1, n_init)):
            model = _Model(self.rho, n_design, mu0, log_sigma0, self.recog_hidden, self.seed + i).to(self.device)
            opt = torch.optim.Adam(model.parameters(), lr=lr)
            loss = self._train(model, opt, y, x, design, n, n_init_epochs, batch_size, delta_loss, patience=2)
            if loss < best_loss:
                best_loss = loss
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

        model = _Model(self.rho, n_design, mu0, log_sigma0, self.recog_hidden, self.seed).to(self.device)
        if best_state is not None:
            model.load_state_dict(best_state)
        opt = torch.optim.Adam(model.parameters(), lr=lr)
        self.losses = []
        self._train(model, opt, y, x, design, n, max_epochs, batch_size, delta_loss, patience, track=self.losses)

        self._model = model
        self._proba = model.proba(x, batch_size)
        return self

    def _train(self, model: _Model, opt: torch.optim.Optimizer, y: torch.Tensor, x: torch.Tensor,
               design: torch.Tensor, n: int, epochs: int, batch_size: int, delta_loss: float,
               patience: int, track: list[float] | None = None) -> float:
        model.train()
        cpu_gen = torch.Generator().manual_seed(self.seed)
        prev: float | None = None
        bad = 0
        last = float("inf")
        for _ in range(epochs):
            perm = torch.randperm(n, generator=cpu_gen).to(self.device)
            total, nb = 0.0, 0
            for s in range(0, n, batch_size):
                idx = perm[s:s + batch_size]
                opt.zero_grad()
                loss = model.neg_elbo(y[idx], x[idx], design[idx], n)
                loss.backward()
                opt.step()
                total += float(loss.item())
                nb += 1
            last = total / max(nb, 1)
            if track is not None:
                track.append(last)
            if prev is not None and abs((last - prev) / (abs(prev) + 1e-8)) < delta_loss:
                bad += 1
                if bad >= patience:
                    self.converged = True
                    break
            else:
                bad = 0
            prev = last
        return last

    # ---- prediction ----

    def predict_proba(self) -> np.ndarray:
        if self._proba is None:
            raise RuntimeError("call fit() before predict_proba()")
        return self._proba

    def predict(self, threshold: float = 0.7) -> np.ndarray:
        proba = self.predict_proba()
        labels = np.array([self.class_names[i] for i in proba.argmax(axis=1)], dtype=object)
        labels[proba.max(axis=1) < threshold] = "Unknown"
        return labels

    @property
    def classes_(self) -> list[str]:
        return list(self.class_names)


def fit(
    expression: np.ndarray,
    marker_names: list[str],
    signature: dict[str, list[str]],
    *,
    batch: np.ndarray | None = None,
    max_epochs: int = 50,
    learning_rate: float = 1e-3,
    batch_size: int = 16384,
    n_init: int = 5,
    random_seed: int = 0,
    device: str | None = "auto",
    cofactor: float = 5.0,
    winsorize: tuple[float, float] = (0.0, 99.9),
    include_batch: bool = True,
    precision: str = "fp32",
) -> AstirCellType:
    """Functional entry point: fit Astir on a raw expression matrix.

    Returns a fitted `AstirCellType` (use `.predict_proba()`, `.predict()`, `.classes_`).
    """
    model = AstirCellType.from_marker_dict(signature, marker_names, device=device, seed=random_seed)
    return model.fit(
        expression, batch=batch, cofactor=cofactor, winsorize=winsorize, max_epochs=max_epochs,
        lr=learning_rate, batch_size=batch_size, n_init=n_init, include_batch=include_batch, precision=precision,
    )
