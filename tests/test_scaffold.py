"""Scaffold utility tests: gen_roi_metadata (pre-staging) + gen_markers (post-staging)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from macsima_pipeline import scaffold
from macsima_pipeline.config import Config


def _cfg(tmp_path: Path, raw_root: Path, roi_metadata_csv: str | None = None) -> Config:
    return Config.model_validate(
        {
            "experiment": {
                "name": "texp",
                "raw_root": str(raw_root),
                "roi_exclude": ["ROI0"],
                "roi_metadata_csv": roi_metadata_csv,
            },
            "paths": {"work_dir": str(tmp_path)},
            "mcmicro": {"params_yaml": "configs/mcmicro_params.yaml"},
            "slurm": {
                "staging": {"partition": "t", "qos": "t", "cpus": 1, "mem": "1G", "time": "1:00:00"},
                "mcmicro": {"partition": "t", "qos": "t", "cpus": 1, "mem": "1G", "time": "1:00:00"},
                "preprocess": {"partition": "t", "qos": "t", "cpus": 1, "mem": "1G", "time": "1:00:00"},
                "viz": {"partition": "t", "qos": "t", "cpus": 1, "mem": "1G", "time": "1:00:00"},
            },
        }
    )


def _make_raw(tmp_path: Path) -> Path:
    raw = tmp_path / "raw"
    for n in ["ROI0", "ROI1", "ROI2", "ROI10"]:
        (raw / n).mkdir(parents=True)
    (raw / "not_a_roi_file.txt").write_text("x")
    return raw


# ---- gen_roi_metadata ------------------------------------------------------


def test_roi_metadata_excludes_roi0_and_sorts_numerically(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path, _make_raw(tmp_path))
    out = scaffold.gen_roi_metadata(cfg, config_path=tmp_path / "config.yaml")
    assert out == tmp_path / "roi_metadata_texp.csv"
    lines = out.read_text().splitlines()
    assert lines[0] == "ROI"
    assert lines[1:] == ["ROI1", "ROI2", "ROI10"]  # ROI0 excluded, numeric order


def test_roi_metadata_extra_columns(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path, _make_raw(tmp_path))
    out = scaffold.gen_roi_metadata(
        cfg, config_path=tmp_path / "config.yaml", extra_columns=["Sample", "Patient_ID"]
    )
    lines = out.read_text().splitlines()
    assert lines[0] == "ROI,Sample,Patient_ID"
    assert lines[1] == "ROI1,,"


def test_roi_metadata_writes_to_configured_path(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path, _make_raw(tmp_path), roi_metadata_csv="meta/roi.csv")
    out = scaffold.gen_roi_metadata(cfg, config_path=tmp_path / "config.yaml")
    assert out == tmp_path / "meta" / "roi.csv"
    assert out.is_file()


def test_roi_metadata_refuses_overwrite_without_force(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path, _make_raw(tmp_path))
    dest = tmp_path / "roi_metadata_texp.csv"
    dest.write_text("KEEP ME\n")
    assert scaffold.gen_roi_metadata(cfg, config_path=tmp_path / "config.yaml") is None
    assert dest.read_text() == "KEEP ME\n"
    # --force overwrites
    out = scaffold.gen_roi_metadata(cfg, config_path=tmp_path / "config.yaml", force=True)
    assert out == dest
    assert dest.read_text().splitlines()[0] == "ROI"


def test_roi_metadata_missing_raw_root_returns_none(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path, tmp_path / "does_not_exist")
    assert scaffold.gen_roi_metadata(cfg, config_path=tmp_path / "config.yaml") is None


# ---- gen_markers -----------------------------------------------------------

MARKERS = """\
channel_number,cycle_number,marker_name,Filter,background,exposure,remove
1,1,bg_001_DAPI,DAPI,,633.6,TRUE
2,1,DAPI,DAPI,,1207.5,
3,1,CD31,FITC,bg_001,633.6,
"""


def _make_staged(tmp_path: Path, markers: str = MARKERS) -> None:
    sample = tmp_path / "mcmicro_output" / "texp" / "rack-01-well-C01-roi-003-exp-2"
    (sample / "registration").mkdir(parents=True)
    (sample / "registration" / "img-exp-2.ome.tif").write_bytes(b"")
    (sample / "markers.csv").write_text(markers)


def test_markers_normalizes_remove_and_keeps_all_rows(tmp_path: Path) -> None:
    _make_staged(tmp_path)
    cfg = _cfg(tmp_path, tmp_path / "raw")
    out = scaffold.gen_markers(cfg, config_path=tmp_path / "config.yaml")
    assert out == tmp_path / "markers_texp.csv"
    df = pd.read_csv(out)
    assert len(df) == 3  # bg_* row retained (channel_index alignment)
    assert df["remove"].tolist() == [True, False, False]
    assert list(df.columns) == [
        "channel_number", "cycle_number", "marker_name", "Filter", "background", "exposure", "remove",
    ]


def test_markers_no_staged_output_returns_none(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path, tmp_path / "raw")  # nothing staged
    assert scaffold.gen_markers(cfg, config_path=tmp_path / "config.yaml") is None


def test_markers_refuses_overwrite_without_force(tmp_path: Path) -> None:
    _make_staged(tmp_path)
    cfg = _cfg(tmp_path, tmp_path / "raw")
    dest = tmp_path / "markers_texp.csv"
    dest.write_text("KEEP\n")
    assert scaffold.gen_markers(cfg, config_path=tmp_path / "config.yaml") is None
    assert dest.read_text() == "KEEP\n"
    out = scaffold.gen_markers(cfg, config_path=tmp_path / "config.yaml", force=True)
    assert out == dest


# ---- gen_signature ---------------------------------------------------------

SIG_MARKERS = """\
channel_number,marker_name,remove
1,bg_001_DAPI,TRUE
2,DAPI,
3,CD3,
4,CD45,
5,CD8,
6,CD4,
7,CD19,
8,CD20,
9,PanCK,
"""


def test_gen_signature_builds_loadable_template(tmp_path: Path) -> None:
    _make_staged(tmp_path, markers=SIG_MARKERS)
    cfg = _cfg(tmp_path, tmp_path / "raw")
    out = scaffold.gen_signature(cfg, config_path=tmp_path / "config.yaml")
    assert out == tmp_path / "signature_texp.yaml"

    from macsima_pipeline.phenotype import signature as sig_mod

    sig = sig_mod.load_signature(out)
    names = set(sig.cell_type_names())
    assert {"T cell", "CD4 T cell", "CD8 T cell", "B cell", "Epithelial"} <= names
    assert list(sig.cell_types["T cell"].positive) == ["CD3", "CD45"]
    # bg_* / removed markers excluded from the panel; usable markers appear in the header
    txt = out.read_text()
    assert "PanCK" in txt and "bg_001_DAPI" not in txt and "version: 1" in txt


def test_gen_signature_no_staged_returns_none(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path, tmp_path / "raw")
    assert scaffold.gen_signature(cfg, config_path=tmp_path / "config.yaml") is None


def test_gen_signature_refuses_overwrite(tmp_path: Path) -> None:
    _make_staged(tmp_path, markers=SIG_MARKERS)
    cfg = _cfg(tmp_path, tmp_path / "raw")
    dest = tmp_path / "signature_texp.yaml"
    dest.write_text("KEEP\n")
    assert scaffold.gen_signature(cfg, config_path=tmp_path / "config.yaml") is None
    assert dest.read_text() == "KEEP\n"
