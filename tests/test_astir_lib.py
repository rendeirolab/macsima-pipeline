"""Clean-room Astir implementation tests."""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from macsima_pipeline.lib.astir import fit  # noqa: E402
from macsima_pipeline.lib.astir import data as adata_mod  # noqa: E402
from macsima_pipeline.lib.astir.io.marker_yaml import build_rho  # noqa: E402


def test_build_rho_shape_and_other_column() -> None:
    markers = ["CD3", "CD8", "CD19", "CD68"]
    sig = {"T cell": ["CD3"], "CD8 T": ["CD3", "CD8"], "B cell": ["CD19"]}
    rho, names = build_rho(sig, markers, include_other=True)
    assert rho.shape == (4, 4)  # G=4, C=3 (+Other)
    assert names == ["T cell", "CD8 T", "B cell", "Other"]
    assert (rho[:, -1] == 0).all()  # Other column all zero
    assert rho[markers.index("CD3"), 0] == 1.0
    assert rho[markers.index("CD8"), 1] == 1.0
    assert rho[markers.index("CD19"), 2] == 1.0
    # a marker not listed for a class stays 0
    assert rho[markers.index("CD68"), :].sum() == 0.0


def test_normalize_and_design_deterministic() -> None:
    rng = np.random.default_rng(0)
    raw = rng.gamma(2.0, 50.0, size=(100, 5)).astype(np.float32)
    y1 = adata_mod.normalize_for_astir(raw, cofactor=5.0)
    y2 = adata_mod.normalize_for_astir(raw, cofactor=5.0)
    np.testing.assert_array_equal(y1, y2)
    z = adata_mod.zscore(y1)
    np.testing.assert_allclose(z.mean(axis=0), 0.0, atol=1e-4)
    # one-hot design over a batch vector
    batch = np.array(["A", "B", "A", "C"])
    design, names = adata_mod.build_design(4, batch, include_batch=True)
    assert names == ["A", "B", "C"]
    assert design.shape == (4, 3)
    np.testing.assert_array_equal(design.sum(axis=1), np.ones(4))
    # default intercept
    d0, n0 = adata_mod.build_design(4, None)
    assert d0.shape == (4, 1) and n0 == ["intercept"]


def test_lowrank_matches_full_covariance() -> None:
    """The rank-1 + diagonal density must equal the equivalent full-covariance MVN."""
    from torch.distributions import MultivariateNormal

    from macsima_pipeline.lib.astir.model import _Model

    markers = ["CD3", "CD8", "CD19", "CD68"]
    sig = {"T cell": ["CD3"], "CD8 T": ["CD3", "CD8"], "B cell": ["CD19"]}
    rho, _ = build_rho(sig, markers)
    g = len(markers)
    mu0 = np.zeros(g, dtype=np.float32)
    ls0 = np.zeros(g, dtype=np.float32)
    model = _Model(rho, n_design=1, mu0=mu0, log_sigma0=ls0, recog_hidden=8, seed=0)
    with torch.no_grad():
        model.p.copy_(torch.randn_like(model.p))
        model.log_delta.copy_(0.3 * torch.randn_like(model.log_delta))
        design = torch.ones(3, 1)
        dist = model._dist(design)
        full = dist.cov_factor @ dist.cov_factor.transpose(-1, -2) + torch.diag_embed(dist.cov_diag)
        mvn = MultivariateNormal(dist.loc, covariance_matrix=full)
        y = torch.randn(3, 1, g)
        assert torch.allclose(dist.log_prob(y), mvn.log_prob(y), atol=1e-4)


def _synthetic_typed(n_per: int = 60, seed: int = 0):
    """Cells of 4 well-separated types, each defined by one high marker (raw scale)."""
    rng = np.random.default_rng(seed)
    markers = ["M0", "M1", "M2", "M3"]
    sig = {"T0": ["M0"], "T1": ["M1"], "T2": ["M2"], "T3": ["M3"]}
    rows, truth = [], []
    for t in range(4):
        base = rng.gamma(2.0, 3.0, size=(n_per, 4)).astype(np.float32)  # low background
        base[:, t] += rng.gamma(4.0, 40.0, size=n_per)  # high on the type's marker
        rows.append(base)
        truth += [f"T{t}"] * n_per
    return np.vstack(rows), markers, sig, np.array(truth)


def test_seed_reproducible() -> None:
    x, markers, sig, _ = _synthetic_typed(n_per=40)
    m1 = fit(x, markers, sig, max_epochs=8, n_init=2, batch_size=64, random_seed=0, device="cpu")
    m2 = fit(x, markers, sig, max_epochs=8, n_init=2, batch_size=64, random_seed=0, device="cpu")
    np.testing.assert_allclose(m1.predict_proba(), m2.predict_proba(), atol=1e-5)


def test_synthetic_ground_truth_recovery() -> None:
    x, markers, sig, truth = _synthetic_typed(n_per=80, seed=1)
    model = fit(x, markers, sig, max_epochs=60, n_init=3, batch_size=128, random_seed=0, device="cpu")
    pred = model.predict(threshold=0.0)  # force a hard label per cell
    assert model.classes_[-1] == "Other"
    acc = float(np.mean(pred == truth))
    assert acc >= 0.9, f"recovery accuracy too low: {acc:.3f}"
