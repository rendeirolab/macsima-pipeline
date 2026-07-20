"""Config loader + interpolation tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from macsima_pipeline.config import Config, load_config

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _write(p: Path, text: str) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text)


def _minimal_default() -> str:
    return """\
experiment:
  name: REQUIRED
  raw_root: REQUIRED
mcmicro:
  params_yaml: configs/mcmicro_params.yaml
  nextflow_config: configs/cemm.nextflow.config
slurm:
  staging:    {partition: tinyq,  qos: tinyq,  cpus: 8,  mem: 32G,  time: "2:00:00"}
  mcmicro:    {partition: shortq, qos: shortq, cpus: 16, mem: 64G,  time: "8:00:00"}
  preprocess: {partition: gpu,    qos: gpu,    cpus: 16, mem: 100G, time: "6:00:00", gres: "gpu:1"}
  viz:        {partition: shortq, qos: shortq, cpus: 8,  mem: 40G,  time: "4:00:00"}
"""


def test_extends_and_interpolation(tmp_path: Path) -> None:
    _write(tmp_path / "default.yaml", _minimal_default())
    _write(
        tmp_path / "exp.yaml",
        """\
extends: "default.yaml"
experiment:
  name: expt99
  raw_root: /tmp/raw
""",
    )

    cfg = load_config(tmp_path / "exp.yaml")
    assert isinstance(cfg, Config)
    assert cfg.experiment.name == "expt99"
    assert cfg.suffix == "_no_bs"
    assert str(cfg.zarr_path()).endswith("expt99_mcmicro_no_bs.zarr")
    assert str(cfg.h5ad_path()).endswith("expt99_cell_expression_mcmicro_no_bs.h5ad")
    assert str(cfg.preprocess_parts_path()).endswith("artifacts/expt99/preprocess_parts_no_bs")
    assert str(cfg.figures_dir()).endswith("figures/expt99")
    assert cfg.jobs_csv("staging").name == "staging_expt99.csv"
    assert cfg.sbatch_path("mcmicro").name == "mcmicro_expt99.sbatch"
    assert cfg.preprocess.parallel.max_workers == 4
    assert cfg.slurm.stage("preprocess_merge").partition == "shortq"
    assert cfg.slurm.stage("preprocess_merge").gres is None
    assert cfg.viz.cell_maps is True
    assert cfg.viz.cell_map_marker_top_n == 20
    assert cfg.viz.channel_qc.enabled is True
    assert cfg.viz.channel_qc.tile_grid == (8, 8)
    assert cfg.viz.channel_qc.workers is None
    assert cfg.viz.channel_qc.report_top_n == 30
    # phenotype (Stage 4) defaults apply even when the config omits the block
    assert cfg.phenotype.enabled is True
    assert cfg.phenotype.signature_matrix is None
    assert cfg.phenotype.engines == ["scyan", "leiden"]
    assert cfg.phenotype.primary_engine == "scyan"
    assert cfg.phenotype.normalize.transform == "arcsinh"
    assert cfg.phenotype.normalize.store_raw_layer == "counts"
    assert cfg.phenotype.normalize.normalized_layer == "zscore"
    assert cfg.phenotype.batch.method == "zscore_per_roi"
    assert cfg.phenotype.scyan.use_layer == "zscore"
    assert cfg.phenotype.leiden.use_layer == "zscore"
    assert cfg.phenotype.scyan.prior_std == 0.25
    assert cfg.phenotype.leiden.n_neighbors == 15
    # phenotype SLURM stage falls back to the GPU preprocess stage when unset
    assert cfg.slurm.stage("phenotype").gres == "gpu:1"
    assert cfg.slurm.stage("phenotype").partition == "gpu"
    assert str(cfg.phenotype_h5ad_path()).endswith("expt99_phenotyped_mcmicro_no_bs.h5ad")


def test_suffix_with_background_subtraction(tmp_path: Path) -> None:
    _write(tmp_path / "default.yaml", _minimal_default())
    _write(
        tmp_path / "exp.yaml",
        """\
extends: "default.yaml"
experiment:
  name: bs_run
  raw_root: /tmp/raw
mcmicro:
  background_subtraction: true
""",
    )
    cfg = load_config(tmp_path / "exp.yaml")
    assert cfg.suffix == ""
    assert str(cfg.zarr_path()).endswith("bs_run_mcmicro.zarr")


def test_missing_required_field_raises(tmp_path: Path) -> None:
    _write(tmp_path / "default.yaml", _minimal_default())
    _write(tmp_path / "exp.yaml", 'extends: "default.yaml"\n')
    with pytest.raises(Exception):
        load_config(tmp_path / "exp.yaml")


def test_shipped_default_loads() -> None:
    """The shipped configs/default.yaml has REQUIRED sentinels; loading directly should error."""
    default = PROJECT_ROOT / "configs" / "default.yaml"
    if not default.exists():
        pytest.skip("default.yaml not shipped in this checkout")
    with pytest.raises(Exception):
        load_config(default)
