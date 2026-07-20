from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from macsima_pipeline import preprocess, slurm
from macsima_pipeline.config import load_config


def _bg_yaml(bg: bool | str) -> str:
    if bg is True:
        return "true"
    if bg is False:
        return "false"
    return '"auto"'


def _write_cfg(tmp_path: Path, *, bg: bool | str = "auto", max_workers: int = 4):
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(
        f"""\
experiment:
  name: tx
  raw_root: {tmp_path / "raw"}
mcmicro:
  params_yaml: configs/mcmicro_params.yaml
  background_subtraction: {_bg_yaml(bg)}
preprocess:
  parallel:
    max_workers: {max_workers}
slurm:
  staging:    {{partition: tinyq,  qos: tinyq,  cpus: 8,  mem: 32G,  time: "2:00:00"}}
  mcmicro:    {{partition: shortq, qos: shortq, cpus: 16, mem: 64G,  time: "8:00:00"}}
  preprocess: {{partition: gpu,    qos: gpu,    cpus: 16, mem: 100G, time: "6:00:00", gres: "gpu:1"}}
  viz:        {{partition: shortq, qos: shortq, cpus: 8,  mem: 40G,  time: "4:00:00"}}
"""
    )
    cfg = load_config(cfg_path)
    cfg.paths.work_dir = tmp_path
    return cfg, cfg_path


def _sample_name(roi: int) -> str:
    return f"rack-01-well-A01-roi-{roi:03d}-exp-2"


def _touch_registration(tmp_path: Path, roi: int) -> Path:
    sample = _sample_name(roi)
    sample_dir = tmp_path / "mcmicro_output" / "tx" / sample
    path = sample_dir / "registration" / f"{sample}.ome.tif"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch()
    (sample_dir / "markers.csv").write_text("marker_name,remove\nDAPI,False\nCD3,True\n")
    return path


def _touch_background(tmp_path: Path, roi: int) -> Path:
    sample = _sample_name(roi)
    sample_dir = tmp_path / "mcmicro_output" / "tx" / sample
    path = sample_dir / "background" / f"{sample}_backsub.ome.tif"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch()
    (sample_dir / "background" / "markers_bs.csv").write_text("marker_name\nDAPI\n")
    return path


def test_discover_jobs_for_explicit_background_modes(tmp_path: Path) -> None:
    cfg, _ = _write_cfg(tmp_path, bg=True)
    _touch_background(tmp_path, 1)
    _touch_background(tmp_path, 2)

    jobs = preprocess.discover_jobs(cfg)

    assert [j.job_id for j in jobs] == [1, 2]
    assert {j.bg for j in jobs} == {True}
    assert [j.roi_name for j in jobs] == ["001", "002"]

    tmp_path_false = tmp_path / "false_case"
    tmp_path_false.mkdir()
    cfg, _ = _write_cfg(tmp_path_false, bg=False)
    _touch_registration(tmp_path_false, 3)

    jobs = preprocess.discover_jobs(cfg)

    assert len(jobs) == 1
    assert jobs[0].bg is False
    assert jobs[0].variant == "no-bg-sub"
    assert jobs[0].roi_name == "003"


def test_discover_jobs_auto_uses_available_variants(tmp_path: Path) -> None:
    cfg, _ = _write_cfg(tmp_path, bg="auto")
    _touch_background(tmp_path, 1)
    _touch_registration(tmp_path, 1)

    jobs = preprocess.discover_jobs(cfg)

    assert [(j.bg, j.variant, j.roi_name) for j in jobs] == [
        (True, "bg-sub", "001"),
        (False, "no-bg-sub", "001"),
    ]


def test_discover_jobs_deferred_before_mcmicro_outputs(tmp_path: Path) -> None:
    cfg, _ = _write_cfg(tmp_path, bg="auto")

    with pytest.raises(preprocess.PreprocessPlanningDeferred, match="deferred until mcmicro"):
        preprocess.discover_jobs(cfg)


def test_jobs_csv_contains_variant_and_part_paths(tmp_path: Path) -> None:
    cfg, _ = _write_cfg(tmp_path, bg="auto")
    img = _touch_registration(tmp_path, 7)

    jobs = preprocess.discover_jobs(cfg)
    csv_path = preprocess.write_jobs_csv(cfg, jobs)
    roundtrip = preprocess.read_jobs_csv(csv_path)

    assert csv_path == tmp_path / "jobs" / "preprocess_tx.csv"
    assert len(roundtrip) == 1
    assert roundtrip[0].image_path == img
    assert roundtrip[0].part_zarr == tmp_path / "artifacts" / "tx" / "preprocess_parts_no_bs" / "007" / "007.zarr"
    assert roundtrip[0].part_h5ad.name == "007_cell_expression.h5ad"


def test_slurm_submit_supports_array_throttle(monkeypatch, tmp_path: Path) -> None:
    calls = []

    monkeypatch.setattr(slurm.shutil, "which", lambda _name: "/usr/bin/sbatch")

    class Result:
        stdout = "Submitted batch job 123\n"

    def fake_run(cmd, **_kwargs):
        calls.append(cmd)
        return Result()

    monkeypatch.setattr(slurm.subprocess, "run", fake_run)

    job_id = slurm.submit(tmp_path / "job.sbatch", array_size=8, array_limit=3)

    assert job_id == 123
    assert calls[0] == ["sbatch", "--array=1-8%3", str(tmp_path / "job.sbatch")]


def test_preprocess_run_submits_worker_array_then_merge(monkeypatch, tmp_path: Path) -> None:
    cfg, cfg_path = _write_cfg(tmp_path, bg=False, max_workers=2)
    _touch_registration(tmp_path, 1)
    _touch_registration(tmp_path, 2)
    calls = []

    def fake_submit(sbatch_path, *, array_size=None, array_limit=None, dependency=None, wait=False):
        calls.append(
            {
                "sbatch": Path(sbatch_path).name,
                "array_size": array_size,
                "array_limit": array_limit,
                "dependency": dependency,
                "wait": wait,
            }
        )
        return 700 + len(calls)

    monkeypatch.setattr(preprocess, "submit", fake_submit)

    job_id = preprocess.run(cfg, cfg_path, do_submit=True, dependency="101")

    assert job_id == 702
    assert calls == [
        {
            "sbatch": "preprocess_tx.sbatch",
            "array_size": 2,
            "array_limit": 2,
            "dependency": "101",
            "wait": False,
        },
        {
            "sbatch": "preprocess_merge_tx.sbatch",
            "array_size": None,
            "array_limit": None,
            "dependency": "701",
            "wait": False,
        },
    ]


def test_merge_preprocess_parts_concatenates_anndata_and_writes_spatialdata(
    monkeypatch,
    tmp_path: Path,
) -> None:
    cfg, _ = _write_cfg(tmp_path, bg=False)
    cfg.experiment.roi_metadata_csv = Path("roi_metadata.csv")
    (tmp_path / "roi_metadata.csv").write_text("ROI,Sample\nROI1,Tumor\n")

    part_zarr = tmp_path / "parts" / "001.zarr"
    part_h5ad = tmp_path / "parts" / "001.h5ad"
    part_zarr.mkdir(parents=True)
    part_h5ad.touch()
    h5ad_writes = {}

    class FakeAnnData:
        def __init__(self, obs: pd.DataFrame):
            self.obs = obs

        def write_h5ad(self, path: str) -> None:
            path = Path(path)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.touch()
            h5ad_writes[path] = self.obs.copy()

    fake_anndata = SimpleNamespace(
        read_h5ad=lambda _path: FakeAnnData(pd.DataFrame({"ROI": ["ROI1"]}, index=["cell1"])),
        concat=lambda adatas: FakeAnnData(pd.concat([a.obs for a in adatas])),
    )
    monkeypatch.setitem(sys.modules, "anndata", fake_anndata)

    jobs_csv = preprocess.write_jobs_csv(
        cfg,
        [
            preprocess.PreprocessJob(
                job_id=1,
                bg=False,
                variant="no-bg-sub",
                roi_name="001",
                image_path=tmp_path / "image.ome.tif",
                part_zarr=part_zarr,
                part_h5ad=part_h5ad,
            )
        ],
    )
    written = []

    class FakeSpatialData:
        def write(self, path: str) -> None:
            written.append(Path(path))
            Path(path).mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(preprocess, "_merge_spatialdata_parts", lambda _paths: FakeSpatialData())

    outputs = preprocess.merge_preprocess_parts(cfg, jobs_csv)

    assert outputs == [(cfg.zarr_path(False), cfg.h5ad_path(False))]
    # _write_sdata_atomic writes to a temp store then renames it to the dest.
    assert len(written) == 1
    assert cfg.zarr_path(False).is_dir()
    tmp = cfg.zarr_path(False).parent / (cfg.zarr_path(False).name + ".tmp")
    assert not tmp.exists()
    assert list(h5ad_writes[cfg.h5ad_path(False)]["ROI"]) == ["ROI1"]
    assert list(h5ad_writes[cfg.h5ad_path(False)]["Sample"]) == ["Tumor"]


def test_merge_replaces_existing_zarr_store(
    monkeypatch,
    tmp_path: Path,
) -> None:
    """A stale destination zarr is atomically replaced by the fresh merge output."""
    cfg, _ = _write_cfg(tmp_path, bg=False)
    cfg.experiment.roi_metadata_csv = Path("roi_metadata.csv")
    (tmp_path / "roi_metadata.csv").write_text("ROI,Sample\nROI1,Tumor\n")

    part_zarr = tmp_path / "parts" / "001.zarr"
    part_h5ad = tmp_path / "parts" / "001.h5ad"
    part_zarr.mkdir(parents=True)
    part_h5ad.touch()

    class FakeAnnData:
        def __init__(self, obs: pd.DataFrame):
            self.obs = obs

        def write_h5ad(self, path: str) -> None:
            path = Path(path)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.touch()

    fake_anndata = SimpleNamespace(
        read_h5ad=lambda _path: FakeAnnData(pd.DataFrame({"ROI": ["ROI1"]}, index=["cell1"])),
        concat=lambda adatas: FakeAnnData(pd.concat([a.obs for a in adatas])),
    )
    monkeypatch.setitem(sys.modules, "anndata", fake_anndata)

    jobs_csv = preprocess.write_jobs_csv(
        cfg,
        [
            preprocess.PreprocessJob(
                job_id=1,
                bg=False,
                variant="no-bg-sub",
                roi_name="001",
                image_path=tmp_path / "image.ome.tif",
                part_zarr=part_zarr,
                part_h5ad=part_h5ad,
            )
        ],
    )

    # Simulate a stale store left behind by a previous run.
    stale = cfg.zarr_path(False)
    stale.mkdir(parents=True)
    (stale / "stale.txt").write_text("old")

    class FakeSpatialData:
        def write(self, path: str) -> None:
            Path(path).mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(preprocess, "_merge_spatialdata_parts", lambda _paths: FakeSpatialData())

    preprocess.merge_preprocess_parts(cfg, jobs_csv)

    assert cfg.zarr_path(False).is_dir()
    assert not (cfg.zarr_path(False) / "stale.txt").exists()
    tmp = cfg.zarr_path(False).parent / (cfg.zarr_path(False).name + ".tmp")
    assert not tmp.exists()
