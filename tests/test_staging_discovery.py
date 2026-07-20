"""Staging discovery + jobs.csv writer tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from macsima_pipeline.config import load_config
from macsima_pipeline import staging


def _write_default(p: Path) -> None:
    p.write_text(
        """\
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
    )


def _make_raw_tree(root: Path, names: list[str]) -> None:
    for name in names:
        (root / name).mkdir(parents=True, exist_ok=True)


def test_discover_rois_filters(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    _make_raw_tree(raw, ["ROI0", "ROI1", "ROI2", "ROI3", "junk"])

    _write_default(tmp_path / "default.yaml")
    (tmp_path / "exp.yaml").write_text(
        f"""\
extends: "default.yaml"
experiment:
  name: tx
  raw_root: {raw}
  roi_exclude: ["ROI0"]
"""
    )
    cfg = load_config(tmp_path / "exp.yaml")
    cfg.paths.work_dir = tmp_path  # write outputs into tmp_path

    rois = staging.discover_rois(cfg)
    names = [p.name for p in rois]
    assert names == ["ROI1", "ROI2", "ROI3"]


def test_discover_rois_include(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    _make_raw_tree(raw, ["ROI1", "ROI2", "ROI3"])
    _write_default(tmp_path / "default.yaml")
    (tmp_path / "exp.yaml").write_text(
        f"""\
extends: "default.yaml"
experiment:
  name: tx
  raw_root: {raw}
  roi_include: ["ROI2"]
"""
    )
    cfg = load_config(tmp_path / "exp.yaml")
    cfg.paths.work_dir = tmp_path
    assert [p.name for p in staging.discover_rois(cfg)] == ["ROI2"]


def test_write_jobs_csv(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    _make_raw_tree(raw, ["ROI1", "ROI2"])
    _write_default(tmp_path / "default.yaml")
    (tmp_path / "exp.yaml").write_text(
        f"""\
extends: "default.yaml"
experiment:
  name: tx
  raw_root: {raw}
"""
    )
    cfg = load_config(tmp_path / "exp.yaml")
    cfg.paths.work_dir = tmp_path

    rois = staging.discover_rois(cfg)
    csv_path = staging.write_jobs_csv(cfg, rois)
    assert csv_path.exists()
    lines = csv_path.read_text().splitlines()
    assert lines[0] == "job_id,roi_path,roi_name,sample_id,output_dir"
    assert len(lines) == 3
    assert lines[1].startswith("1,")
    assert ",ROI1," in lines[1]


def test_discover_rois_missing_root(tmp_path: Path) -> None:
    _write_default(tmp_path / "default.yaml")
    (tmp_path / "exp.yaml").write_text(
        f"""\
extends: "default.yaml"
experiment:
  name: tx
  raw_root: {tmp_path / 'nope'}
"""
    )
    cfg = load_config(tmp_path / "exp.yaml")
    with pytest.raises(FileNotFoundError):
        staging.discover_rois(cfg)
