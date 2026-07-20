"""Phenotype stage tests: signature, normalization, engines, spatial QC, write-back."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from macsima_pipeline.phenotype import signature as sig_mod

# --------------------------------------------------------------------------- #
#  Signature matrix                                                           #
# --------------------------------------------------------------------------- #

_SIGNATURE_YAML = """\
version: 1
cell_types:
  T cell:     {positive: [CD3, CD45], negative: [CD19, CD68], parent: Immune}
  CD8 T cell: {positive: [CD3, CD8],  negative: [CD4],        parent: T cell}
  CD4 T cell: {positive: [CD3, CD4],  negative: [CD8],        parent: T cell}
  B cell:     {positive: [CD19],      negative: [CD3, CD68],  parent: Immune}
  Macrophage: [CD68, CD45]
"""


def _write_signature(tmp_path: Path, text: str = _SIGNATURE_YAML) -> Path:
    p = tmp_path / "signature.yaml"
    p.write_text(text)
    return p


def test_signature_load_and_shorthand(tmp_path: Path) -> None:
    sig = sig_mod.load_signature(_write_signature(tmp_path))
    assert sig.version == 1
    assert set(sig.cell_type_names()) == {"T cell", "CD8 T cell", "CD4 T cell", "B cell", "Macrophage"}
    # positive-only shorthand parsed with empty negatives
    mac = sig.cell_types["Macrophage"]
    assert mac.positive == ("CD68", "CD45")
    assert mac.negative == ()
    assert mac.parent is None
    assert sig.to_marker_dict()["CD8 T cell"] == ["CD3", "CD8"]


def test_signature_coarse_map(tmp_path: Path) -> None:
    sig = sig_mod.load_signature(_write_signature(tmp_path))
    coarse = sig.coarse_map()
    # leaf -> root ancestor; "Immune" is a lineage label (not a defined cell type) and terminates
    assert coarse["CD8 T cell"] == "Immune"
    assert coarse["CD4 T cell"] == "Immune"
    assert coarse["T cell"] == "Immune"
    assert coarse["B cell"] == "Immune"
    # no parent -> maps to itself
    assert coarse["Macrophage"] == "Macrophage"


def test_signature_score_matrix(tmp_path: Path) -> None:
    sig = sig_mod.load_signature(_write_signature(tmp_path))
    markers = ["CD3", "CD8", "CD4", "CD19", "CD68", "CD45"]
    mat = sig.score_matrix(markers)
    assert mat.shape == (5, 6)
    names = sig.cell_type_names()
    b = mat[names.index("B cell")]
    assert b[markers.index("CD19")] == 1.0
    assert b[markers.index("CD3")] == -1.0
    assert b[markers.index("CD68")] == -1.0
    assert b[markers.index("CD8")] == 0.0


def test_signature_validate_warns_on_missing_but_keeps_usable(tmp_path: Path, caplog) -> None:
    sig = sig_mod.load_signature(_write_signature(tmp_path))
    panel = ["DAPI", "CD3", "CD8", "CD4", "CD19", "CD68", "CD45"]
    with caplog.at_level("WARNING"):
        usable = sig_mod.load_signature(_write_signature(tmp_path)).validate_against(panel)
    # every referenced marker is present here -> no missing warning, usable in panel order
    assert usable == ["CD3", "CD8", "CD4", "CD19", "CD68", "CD45"]
    assert sig.cell_type_names()  # sanity


def test_signature_validate_raises_when_type_loses_all_positives(tmp_path: Path) -> None:
    sig = sig_mod.load_signature(_write_signature(tmp_path))
    # panel lacks CD68 and CD45 -> Macrophage (positives CD68, CD45) has no usable positive
    with pytest.raises(ValueError, match="Macrophage"):
        sig.validate_against(["CD3", "CD8", "CD4", "CD19"])


def test_signature_requires_cell_types(tmp_path: Path) -> None:
    p = tmp_path / "bad.yaml"
    p.write_text("version: 1\n")
    with pytest.raises(ValueError, match="cell_types"):
        sig_mod.load_signature(p)


# --------------------------------------------------------------------------- #
#  Normalization                                                              #
# --------------------------------------------------------------------------- #

MARKERS = ["DAPI", "CD3", "CD8", "CD4", "CD19", "CD68", "CD45"]


def _make_adata(n: int = 240, rois: int = 3, seed: int = 0):
    """Synthetic cells with block-structured, ROI-shifted raw intensities."""
    ad = pytest.importorskip("anndata")
    import pandas as pd

    rng = np.random.default_rng(seed)
    g = len(MARKERS)
    x = rng.gamma(shape=2.0, scale=50.0, size=(n, g)).astype(np.float32)
    # a few bright outliers to exercise winsorization
    x[rng.integers(0, n, 5), rng.integers(0, g, 5)] *= 40.0
    roi = np.array([f"ROI{i % rois + 1}" for i in range(n)])
    # per-ROI intensity shift (batch effect)
    for i, r in enumerate(sorted(set(roi))):
        x[roi == r] += (i + 1) * 20.0
    obs = pd.DataFrame(
        {
            "ROI": roi,
            "centroid_x": rng.uniform(0, 1000, n),
            "centroid_y": rng.uniform(0, 1000, n),
        }
    )
    var = pd.DataFrame(index=MARKERS)
    return ad.AnnData(X=x, obs=obs, var=var)


def test_normalize_determinism_and_raw_preserved() -> None:
    from macsima_pipeline.config import PhenotypeNormalizeCfg
    from macsima_pipeline.phenotype import normalize as norm

    cfg = PhenotypeNormalizeCfg()
    a1 = _make_adata()
    raw = a1.X.copy()
    norm.stash_raw(a1, cfg.store_raw_layer)
    norm.normalize(a1, cfg)
    # raw preserved untouched
    np.testing.assert_allclose(a1.layers["counts"], raw, rtol=0, atol=0)
    # deterministic: a fresh run gives identical output
    a2 = _make_adata()
    norm.normalize(a2, cfg)
    np.testing.assert_allclose(a1.X, a2.X, rtol=1e-6, atol=1e-6)
    # finite, and per-marker standardized
    assert np.isfinite(a1.X).all()
    np.testing.assert_allclose(a1.X.mean(axis=0), 0.0, atol=1e-4)
    np.testing.assert_allclose(a1.X.std(axis=0), 1.0, atol=1e-4)


def test_normalize_sparse_equals_dense() -> None:
    from scipy import sparse

    from macsima_pipeline.config import PhenotypeNormalizeCfg
    from macsima_pipeline.phenotype import normalize as norm

    cfg = PhenotypeNormalizeCfg()
    dense = _make_adata()
    sp = _make_adata()
    sp.X = sparse.csr_matrix(sp.X)
    norm.normalize(dense, cfg)
    norm.normalize(sp, cfg)
    np.testing.assert_allclose(dense.X, sp.X, rtol=1e-5, atol=1e-5)


def test_batch_zscore_small_roi_fallback() -> None:
    from macsima_pipeline.config import PhenotypeBatchCfg, PhenotypeNormalizeCfg
    from macsima_pipeline.phenotype import normalize as norm

    a = _make_adata(n=120, rois=3)
    # force one ROI to be tiny so it falls back to global stats
    a.obs.loc[a.obs.index[:2], "ROI"] = "ROI_tiny"
    norm.normalize(a, PhenotypeNormalizeCfg())
    norm.apply_batch(a, PhenotypeBatchCfg(method="zscore_per_roi", min_cells_per_batch=50))
    assert np.isfinite(a.X).all()


def test_batch_none_is_noop() -> None:
    from macsima_pipeline.config import PhenotypeBatchCfg, PhenotypeNormalizeCfg
    from macsima_pipeline.phenotype import normalize as norm

    a = _make_adata()
    norm.normalize(a, PhenotypeNormalizeCfg())
    before = a.X.copy()
    norm.apply_batch(a, PhenotypeBatchCfg(method="none"))
    np.testing.assert_array_equal(a.X, before)


# --------------------------------------------------------------------------- #
#  FlowSOM metacluster labeling (pure numpy; no flowsom package needed)       #
# --------------------------------------------------------------------------- #


def test_label_metaclusters_assigns_by_enrichment(tmp_path: Path) -> None:
    from macsima_pipeline.phenotype.engines.flowsom import _label_metaclusters

    sig = sig_mod.load_signature(_write_signature(tmp_path))
    markers = MARKERS  # DAPI CD3 CD8 CD4 CD19 CD68 CD45
    score = sig.score_matrix(markers)
    names = sig.cell_type_names()

    # three metaclusters, each enriched for a distinct type's positives
    def cell(**vals):
        row = np.zeros(len(markers))
        for k, v in vals.items():
            row[markers.index(k)] = v
        return row

    rows, ids = [], []
    for _ in range(20):  # CD8 T: CD3+ CD8+ CD45+ CD4-
        rows.append(cell(CD3=2.0, CD8=2.0, CD45=1.5, CD4=-1.0))
        ids.append(0)
    for _ in range(20):  # B cell: CD19+ CD45+ CD3-
        rows.append(cell(CD19=2.5, CD45=1.5, CD3=-1.0, CD68=-1.0))
        ids.append(1)
    for _ in range(20):  # Macrophage: CD68+ CD45+
        rows.append(cell(CD68=2.5, CD45=1.5))
        ids.append(2)
    z = np.array(rows)
    cluster_ids = np.array(ids)

    labels, conf, label_map, scores = _label_metaclusters(z, cluster_ids, score, names, tau=0.0)
    assert label_map[0] == "CD8 T cell"
    assert label_map[1] == "B cell"
    assert label_map[2] == "Macrophage"
    assert (conf > 0).all()
    assert scores.shape == (3, len(names))


def test_run_flowsom_end_to_end(tmp_path: Path) -> None:
    pytest.importorskip("flowsom")
    import pandas as pd

    from macsima_pipeline.config import PhenotypeFlowsomCfg
    from macsima_pipeline.phenotype.engines import flowsom as fe

    ad = pytest.importorskip("anndata")
    rng = np.random.default_rng(0)
    markers = MARKERS  # DAPI CD3 CD8 CD4 CD19 CD68 CD45

    def block(n, highs):
        m = rng.normal(0, 0.3, size=(n, len(markers))).astype(np.float32)
        for name in highs:
            m[:, markers.index(name)] += 4.0
        return m

    z = np.vstack([block(60, ["CD3", "CD45"]), block(60, ["CD19", "CD45"]), block(60, ["CD68", "CD45"])])
    obs = pd.DataFrame({"ROI": ["ROI1"] * z.shape[0]})
    adata = ad.AnnData(X=z.copy(), obs=obs, var=pd.DataFrame(index=markers))
    adata.layers["zscore"] = z

    sig = sig_mod.load_signature(_write_signature(tmp_path))
    cfg = PhenotypeFlowsomCfg(grid_size=(6, 6), n_metaclusters=6, som_iterations=20, train_subsample=None)
    res = fe.run_flowsom(adata, sig, cfg, batch_key="ROI")

    assert res.cluster is not None
    assert len(res.labels) == adata.n_obs
    assert set(res.labels.astype(str)) <= set(sig.cell_type_names()) | {"Unknown"}
    # the CD19 block should be dominated by B cell labels
    b_block = res.labels.iloc[60:120].astype(str)
    assert (b_block == "B cell").mean() > 0.5


def test_label_metaclusters_tau_yields_unknown(tmp_path: Path) -> None:
    from macsima_pipeline.phenotype.engines.flowsom import _label_metaclusters

    sig = sig_mod.load_signature(_write_signature(tmp_path))
    z = np.zeros((10, len(MARKERS)))  # no enrichment
    cluster_ids = np.zeros(10, dtype=int)
    labels, _, label_map, _ = _label_metaclusters(
        z, cluster_ids, sig.score_matrix(MARKERS), sig.cell_type_names(), tau=0.1
    )
    assert label_map[0] == "Unknown"
    assert set(labels) == {"Unknown"}


# --------------------------------------------------------------------------- #
#  IO: atomic write + resume check                                            #
# --------------------------------------------------------------------------- #


class _StubCfg:
    def __init__(self, dest: Path) -> None:
        self._dest = dest

    def phenotype_h5ad_path(self, bg=None) -> Path:
        return self._dest


def test_write_atomic_and_phenotype_done(tmp_path: Path) -> None:
    from macsima_pipeline.phenotype import io as pio

    a = _make_adata(n=30)
    dest = tmp_path / "out" / "phenotyped.h5ad"
    cfg = _StubCfg(dest)

    assert pio.phenotype_done(cfg) is False  # nothing written yet
    pio.write_cells_atomic(a, dest)  # no cell_type column
    assert dest.is_file()
    assert pio.phenotype_done(cfg) is False  # present but no cell_type

    import pandas as pd

    a.obs["cell_type"] = pd.Categorical(["T cell"] * a.n_obs)
    pio.write_cells_atomic(a, dest)
    assert pio.phenotype_done(cfg) is True
    # no leftover temp files
    assert not list(dest.parent.glob("*.tmp"))


# --------------------------------------------------------------------------- #
#  Astir adapter (fake lib — no torch training)                               #
# --------------------------------------------------------------------------- #

_ADAPTER_SIG = """\
version: 1
cell_types:
  T cell: [CD3, CD45]
  B cell: [CD19]
  Macrophage: [CD68]
"""


def test_astir_adapter_thresholds_low_confidence(tmp_path: Path, monkeypatch) -> None:
    from macsima_pipeline.config import PhenotypeAstirCfg
    from macsima_pipeline.phenotype.engines import astir as astir_engine

    (tmp_path / "s.yaml").write_text(_ADAPTER_SIG)
    sig = sig_mod.load_signature(tmp_path / "s.yaml")

    a = _make_adata(n=6)
    a.layers["counts"] = a.X.copy()  # adapter reads raw counts layer

    classes = [*sig.cell_type_names(), "Other"]  # T cell, B cell, Macrophage, Other

    def fake_fit(expression, marker_names, signature, **kw):
        n = expression.shape[0]
        proba = np.full((n, len(classes)), 0.02, dtype=np.float32)
        proba[: n // 2, 0] = 0.9  # confident T cell
        proba[n // 2 :, :] = 1.0 / len(classes)  # flat -> below threshold
        proba /= proba.sum(axis=1, keepdims=True)

        class _Fake:
            classes_ = classes
            losses = [1.0, 0.5]
            converged = True

            def predict_proba(self_inner):
                return proba

        return _Fake()

    monkeypatch.setattr("macsima_pipeline.lib.astir.fit", fake_fit)

    cfg = PhenotypeAstirCfg(min_confidence=0.7)
    res = astir_engine.run_astir(a, sig, cfg, batch_key="ROI")
    assert list(res.probabilities.columns) == classes
    assert (res.labels.iloc[: 3] == "T cell").all()
    assert (res.labels.iloc[3:] == "Unknown").all()
    assert res.uns["engine"] == "astir"
    assert res.uns["converged"] is True


# --------------------------------------------------------------------------- #
#  Spatial QC + cross-engine agreement                                        #
# --------------------------------------------------------------------------- #


def test_cross_engine_agreement() -> None:
    import pandas as pd

    from macsima_pipeline.phenotype.spatial_qc import cross_engine_agreement

    idx = [f"c{i}" for i in range(6)]
    a = pd.Series(["T", "T", "B", "B", "M", "M"], index=idx)
    b = pd.Series(["T", "T", "B", "B", "M", "T"], index=idx)  # last disagrees
    agree, metrics = cross_engine_agreement(a, b)
    assert agree.tolist() == [True, True, True, True, True, False]
    assert metrics["accuracy"] == pytest.approx(5 / 6)
    assert metrics["n_cells"] == 6
    assert -1.0 <= metrics["cohen_kappa"] <= 1.0


def test_composition_table_sums_to_one() -> None:
    import pandas as pd

    from macsima_pipeline.phenotype.spatial_qc import composition_table

    a = _make_adata(n=90, rois=3)
    a.obs["cell_type"] = pd.Categorical(
        ["T cell", "B cell", "Macrophage"] * (a.n_obs // 3)
    )
    comp = composition_table(a, "cell_type", "ROI")
    np.testing.assert_allclose(comp.sum(axis=1).to_numpy(), 1.0, atol=1e-6)
    assert set(comp.index) == {"ROI1", "ROI2", "ROI3"}


def _make_grid_adata(labels: np.ndarray):
    """Cells on a square grid, single ROI, with a given cell_type per cell."""
    ad = pytest.importorskip("anndata")
    import pandas as pd

    n = len(labels)
    side = int(np.ceil(np.sqrt(n)))
    xs, ys = np.meshgrid(np.arange(side), np.arange(side))
    coords = np.c_[xs.ravel(), ys.ravel()][:n].astype(float)
    obs = pd.DataFrame(
        {
            "ROI": ["ROI1"] * n,
            "centroid_x": coords[:, 0],
            "centroid_y": coords[:, 1],
            "cell_type": pd.Categorical(labels),
        }
    )
    x = np.random.default_rng(0).normal(size=(n, len(MARKERS))).astype(np.float32)
    return ad.AnnData(X=x, obs=obs, var=pd.DataFrame(index=MARKERS))


def test_spatial_qc_homophily_high_when_coherent() -> None:
    pytest.importorskip("squidpy")
    from macsima_pipeline.config import PhenotypeSpatialQCCfg
    from macsima_pipeline.phenotype import spatial_qc

    n = 100
    side = 10
    # coherent: top half type A, bottom half type B (spatial blocks)
    coherent = np.array(["A" if (i // side) < side // 2 else "B" for i in range(n)])
    a = _make_grid_adata(coherent)
    spatial_qc.build_spatial(a, "centroid_x", "centroid_y")
    assert a.obsm["spatial"].shape == (n, 2)
    # homophily only here (skip nhood_enrichment's numba JIT to keep the test fast)
    cfg = PhenotypeSpatialQCCfg(nhood_enrichment=False, n_neighs=4)
    qc = spatial_qc.compute_spatial_qc(a, cfg, "cell_type", batch_key="ROI")
    coh_homo = qc["homophily"]["overall"]

    # scrambled labels -> lower homophily
    rng = np.random.default_rng(0)
    scrambled = coherent.copy()
    rng.shuffle(scrambled)
    b = _make_grid_adata(scrambled)
    spatial_qc.build_spatial(b, "centroid_x", "centroid_y")
    qc_b = spatial_qc.compute_spatial_qc(b, cfg, "cell_type", batch_key="ROI")
    assert coh_homo > qc_b["homophily"]["overall"]
    assert coh_homo > 0.7


# --------------------------------------------------------------------------- #
#  Orchestrator: write-back contract + resume + graceful skip                 #
# --------------------------------------------------------------------------- #

_MINIMAL_DEFAULT = """\
experiment:
  name: REQUIRED
  raw_root: REQUIRED
containers:
  macsima2mc_sif: macsima2mc.sif
mcmicro:
  params_yaml: configs/mcmicro_params.yaml
  background_subtraction: false
slurm:
  staging:    {partition: tinyq,  qos: tinyq,  cpus: 8,  mem: 32G,  time: "2:00:00"}
  mcmicro:    {partition: shortq, qos: shortq, cpus: 16, mem: 64G,  time: "8:00:00"}
  preprocess: {partition: gpu,    qos: gpu,    cpus: 16, mem: 100G, time: "6:00:00", gres: "gpu:1"}
  viz:        {partition: shortq, qos: shortq, cpus: 8,  mem: 40G,  time: "4:00:00"}
"""


def _build_cfg(tmp_path: Path):
    from macsima_pipeline.config import load_config

    (tmp_path / "default.yaml").write_text(_MINIMAL_DEFAULT)
    (tmp_path / "exp.yaml").write_text(
        "extends: default.yaml\nexperiment:\n  name: pheno\n  raw_root: /tmp/raw\n"
    )
    (tmp_path / "signature.yaml").write_text(_ADAPTER_SIG)
    cfg = load_config(tmp_path / "exp.yaml")
    cfg.paths.work_dir = tmp_path
    cfg.phenotype.signature_matrix = Path("signature.yaml")
    cfg.phenotype.spatial_qc.enabled = False  # skip squidpy in the contract test
    return cfg


def _fake_engine_result(adata, sig, name: str, with_cluster: bool):
    import pandas as pd

    from macsima_pipeline.phenotype.engines.base import EngineResult

    names = sig.cell_type_names()
    labels = pd.Series([names[i % len(names)] for i in range(adata.n_obs)], index=adata.obs_names)
    conf = pd.Series(np.full(adata.n_obs, 0.9), index=adata.obs_names)
    cluster = None
    if with_cluster:
        cluster = pd.Series([str(i % 5) for i in range(adata.n_obs)], index=adata.obs_names)
    return EngineResult(labels=labels, confidence=conf, cluster=cluster, uns={"engine": name})


def test_run_inproc_write_back_contract(tmp_path: Path, monkeypatch) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import anndata as ad

    from macsima_pipeline.phenotype import io as pio
    from macsima_pipeline.phenotype import workers

    cfg = _build_cfg(tmp_path)
    raw = _make_adata(n=60, rois=2)
    cfg.h5ad_path(False).parent.mkdir(parents=True, exist_ok=True)
    raw.write_h5ad(cfg.h5ad_path(False))

    monkeypatch.setattr(
        workers.astir_engine, "run_astir",
        lambda adata, sig, c, batch_key=None: _fake_engine_result(adata, sig, "astir", False),
    )
    monkeypatch.setattr(
        workers.flowsom_engine, "run_flowsom",
        lambda adata, sig, c, batch_key=None: _fake_engine_result(adata, sig, "flowsom", True),
    )

    workers.run_inproc(cfg)

    dest = cfg.phenotype_h5ad_path(False)
    assert dest.is_file()
    assert pio.phenotype_done(cfg, False) is True
    out = ad.read_h5ad(dest)
    assert "counts" in out.layers and "zscore" in out.layers
    assert "spatial" in out.obsm and out.obsm["spatial"].shape == (out.n_obs, 2)
    for col in ("cell_type", "cell_type_coarse", "cell_type_confidence",
                "astir_celltype", "flowsom", "flowsom_celltype", "pheno_agree"):
        assert col in out.obs.columns, col
    assert "phenotype" in out.uns
    assert "composition" in out.uns["phenotype"]
    # coarse label maps CD3/CD45-based "T cell" -> itself (no parent in this signature)
    assert set(out.obs["cell_type"].astype(str)) <= {"T cell", "B cell", "Macrophage"}

    # QC PDF was produced
    assert (cfg.figures_dir() / "phenotype" / "pheno_phenotype_summary_no_bs.pdf").is_file()


def test_run_inproc_skips_without_signature(tmp_path: Path, monkeypatch) -> None:
    from macsima_pipeline.phenotype import workers

    cfg = _build_cfg(tmp_path)
    cfg.phenotype.signature_matrix = None  # -> graceful skip

    called = {"astir": False}

    def _boom(*a, **k):
        called["astir"] = True
        raise AssertionError("engine must not run when signature is unset")

    monkeypatch.setattr(workers.astir_engine, "run_astir", _boom)
    workers.run_inproc(cfg)  # returns cleanly
    assert called["astir"] is False
    assert not cfg.phenotype_h5ad_path(False).exists()
