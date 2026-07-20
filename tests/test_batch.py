"""Batch (multiple experiments in one config) expansion + materialization tests."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from macsima_pipeline.config import (
    Config,
    expand_config,
    load_config,
    materialize_configs,
)

DEFAULT = """\
experiment:
  name: REQUIRED
  raw_root: REQUIRED
  roi_exclude: ["ROI0"]
mcmicro:
  params_yaml: configs/mcmicro_params.yaml
slurm:
  staging:    {partition: tinyq,  qos: tinyq,  cpus: 8,  mem: 32G,  time: "2:00:00"}
  mcmicro:    {partition: shortq, qos: shortq, cpus: 16, mem: 64G,  time: "8:00:00"}
  preprocess: {partition: gpu,    qos: gpu,    cpus: 16, mem: 100G, time: "6:00:00", gres: "gpu:1"}
  viz:        {partition: shortq, qos: shortq, cpus: 8,  mem: 40G,  time: "4:00:00"}
"""


def _write(p: Path, text: str) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text)


def _batch_cfg(tmp_path: Path, body: str) -> Path:
    """Write default.yaml + a batch config whose work_dir is tmp_path."""
    _write(tmp_path / "default.yaml", DEFAULT)
    cfg_path = tmp_path / "batch.yaml"
    _write(cfg_path, f'extends: "default.yaml"\npaths:\n  work_dir: "{tmp_path}"\n{body}')
    return cfg_path


def _single_cfg(tmp_path: Path) -> Path:
    _write(tmp_path / "default.yaml", DEFAULT)
    cfg_path = tmp_path / "single.yaml"
    _write(
        cfg_path,
        f'extends: "default.yaml"\npaths:\n  work_dir: "{tmp_path}"\n'
        "experiment:\n  name: solo\n  raw_root: /tmp/rawS\n",
    )
    return cfg_path


TWO = """\
experiments:
  - name: expA
    raw_root: /tmp/rawA
  - name: expB
    raw_root: /tmp/rawB
    roi_exclude: ["ROI0", "ROI7"]
"""


# ---- expand_config (in-memory) --------------------------------------------


def test_expand_single_returns_one(tmp_path: Path) -> None:
    cfgs = expand_config(_single_cfg(tmp_path))
    assert len(cfgs) == 1
    assert isinstance(cfgs[0], Config)
    assert cfgs[0].experiment.name == "solo"


def test_expand_batch_count_names_and_inherited_defaults(tmp_path: Path) -> None:
    cfgs = expand_config(_batch_cfg(tmp_path, TWO))
    assert {c.experiment.name for c in cfgs} == {"expA", "expB"}
    by_name = {c.experiment.name: c for c in cfgs}
    assert str(by_name["expA"].experiment.raw_root) == "/tmp/rawA"
    assert str(by_name["expB"].experiment.raw_root) == "/tmp/rawB"
    # expA inherits roi_exclude from the shared experiment defaults block...
    assert by_name["expA"].experiment.roi_exclude == ["ROI0"]
    # ...expB overrides it in its own entry.
    assert by_name["expB"].experiment.roi_exclude == ["ROI0", "ROI7"]
    # shared sections apply to every experiment
    assert by_name["expA"].slurm.staging.partition == "tinyq"


def test_only_filter(tmp_path: Path) -> None:
    cfgs = expand_config(_batch_cfg(tmp_path, TWO), only=["expB"])
    assert [c.experiment.name for c in cfgs] == ["expB"]


def test_only_unknown_name_errors(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="not found"):
        expand_config(_batch_cfg(tmp_path, TWO), only=["nope"])


def test_duplicate_names_error(tmp_path: Path) -> None:
    body = "experiments:\n  - {name: dup, raw_root: /tmp/a}\n  - {name: dup, raw_root: /tmp/b}\n"
    with pytest.raises(ValueError, match="duplicate"):
        expand_config(_batch_cfg(tmp_path, body))


def test_empty_experiments_error(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="non-empty"):
        expand_config(_batch_cfg(tmp_path, "experiments: []\n"))


def test_both_keys_error(tmp_path: Path) -> None:
    cfg_path = tmp_path / "both.yaml"
    _write(
        cfg_path,
        "experiment:\n  name: x\n  raw_root: /tmp/x\n"
        "experiments:\n  - {name: a, raw_root: /tmp/a}\n",
    )
    with pytest.raises(ValueError, match="not both"):
        expand_config(cfg_path)


# ---- materialize_configs (submit path) ------------------------------------


def test_materialize_single_writes_nothing(tmp_path: Path) -> None:
    cfg_path = _single_cfg(tmp_path)
    pairs = materialize_configs(cfg_path)
    assert pairs == [("solo", cfg_path.resolve())]
    assert not (tmp_path / "jobs" / "batch").exists()


def test_materialize_batch_writes_loadable_flat_files(tmp_path: Path) -> None:
    pairs = materialize_configs(_batch_cfg(tmp_path, TWO))
    assert {name for name, _ in pairs} == {"expA", "expB"}
    for name, p in pairs:
        assert p.is_file()
        assert p.parent == (tmp_path / "jobs" / "batch")
        # flattened: no extends / no experiments list left in the written file
        raw = yaml.safe_load(p.read_text())
        assert "extends" not in raw
        assert "experiments" not in raw
        assert raw["experiment"]["name"] == name
        # re-loads as a valid single-experiment Config
        loaded = load_config(p)
        assert isinstance(loaded, Config)
        assert loaded.experiment.name == name
    by_name = dict(pairs)
    assert str(load_config(by_name["expA"]).experiment.raw_root) == "/tmp/rawA"


def test_materialize_only_filters(tmp_path: Path) -> None:
    pairs = materialize_configs(_batch_cfg(tmp_path, TWO), only=["expA"])
    assert [name for name, _ in pairs] == ["expA"]
    assert (tmp_path / "jobs" / "batch" / "expA.yaml").is_file()
    assert not (tmp_path / "jobs" / "batch" / "expB.yaml").exists()
