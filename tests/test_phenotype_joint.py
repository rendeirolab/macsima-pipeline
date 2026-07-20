"""Joint (cross-experiment) phenotyping orchestration.

The engine core `_phenotype_adata` and the QC report are stubbed so these tests
exercise ONLY the joint-specific logic (concat, experiment tagging, combined write,
per-experiment split-back) without pulling torch / matplotlib.
"""

from __future__ import annotations

from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

from macsima_pipeline.config import Config
from macsima_pipeline.phenotype import workers


def _cfg(tmp_path: Path, name: str) -> Config:
    return Config.model_validate(
        {
            "experiment": {"name": name, "raw_root": str(tmp_path / "raw")},
            "paths": {"work_dir": str(tmp_path)},
            "containers": {"macsima2mc_sif": "macsima2mc.sif"},
            "mcmicro": {"params_yaml": "configs/mcmicro_params.yaml", "background_subtraction": False},
            "phenotype": {"signature_matrix": "sig.yaml"},
            "slurm": {
                "staging": {"partition": "t", "qos": "t", "cpus": 1, "mem": "1G", "time": "1:00:00"},
                "mcmicro": {"partition": "t", "qos": "t", "cpus": 1, "mem": "1G", "time": "1:00:00"},
                "preprocess": {"partition": "t", "qos": "t", "cpus": 1, "mem": "1G", "time": "1:00:00"},
                "viz": {"partition": "t", "qos": "t", "cpus": 1, "mem": "1G", "time": "1:00:00"},
            },
        }
    )


def _write_cells(cfg: Config, n: int, rois: int, seed: int) -> None:
    rng = np.random.default_rng(seed)
    obs = pd.DataFrame(
        {
            "ROI": [f"ROI{1 + (i % rois)}" for i in range(n)],
            "centroid_x": rng.random(n),
            "centroid_y": rng.random(n),
        },
        index=[f"{cfg.experiment.name}_cell{i}" for i in range(n)],
    )
    var = pd.DataFrame(index=["DAPI", "CD3", "CD8", "PanCK"])
    a = ad.AnnData(X=rng.random((n, 4)).astype("float32"), obs=obs, var=var)
    dest = cfg.h5ad_path(False)
    dest.parent.mkdir(parents=True, exist_ok=True)
    a.write_h5ad(dest)


def _stub_phenotype_adata(cfg, adata, *, batch=None):
    """Minimal stand-in: assign a deterministic cell_type, no engines."""
    labels = ["T cell" if i % 2 else "B cell" for i in range(adata.n_obs)]
    adata.obs["cell_type"] = pd.Categorical(labels)
    adata.obs["cell_type_confidence"] = np.ones(adata.n_obs, dtype=float)
    adata.uns["phenotype"] = {"engines": ["stub"], "batch_key": batch.batch_key}
    return adata, {}, {}, pd.DataFrame(), {}


def test_run_joint_concats_and_splits_back(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(workers, "_phenotype_adata", _stub_phenotype_adata)
    monkeypatch.setattr(workers, "_write_report", lambda *a, **k: None)

    cfg_a = _cfg(tmp_path, "expA")
    cfg_b = _cfg(tmp_path, "expB")
    _write_cells(cfg_a, n=20, rois=3, seed=1)
    _write_cells(cfg_b, n=12, rois=2, seed=2)

    outputs = workers.run_joint([cfg_a, cfg_b], name="joint_ds", batch_key="sample")

    # combined output
    assert len(outputs) == 1
    combined = ad.read_h5ad(outputs[0])
    assert combined.n_obs == 32
    assert set(combined.obs["experiment"]) == {"expA", "expB"}
    assert "sample" in combined.obs.columns
    assert "cell_type" in combined.obs.columns
    # batch key threaded into the joint run
    assert combined.uns["phenotype"]["batch_key"] == "sample"

    # per-experiment split-back written where viz expects it, with joint labels + original ids
    sub_a = ad.read_h5ad(cfg_a.phenotype_h5ad_path(False))
    sub_b = ad.read_h5ad(cfg_b.phenotype_h5ad_path(False))
    assert sub_a.n_obs == 20 and sub_b.n_obs == 12
    assert "cell_type" in sub_a.obs.columns
    assert list(sub_a.obs_names) == [f"expA_cell{i}" for i in range(20)]
    assert set(sub_a.obs["experiment"]) == {"expA"}


def test_run_joint_skips_when_signature_unset(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(workers, "_phenotype_adata", _stub_phenotype_adata)
    cfg = _cfg(tmp_path, "expA")
    cfg.phenotype.signature_matrix = None
    assert workers.run_joint([cfg], name="joint_ds") == []
