"""Scaffold utility tests: gen_roi_metadata (pre-staging) + write_signature_template."""

from __future__ import annotations

from pathlib import Path

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


# ---- write_signature_template ----------------------------------------------


def test_write_signature_template_builds_loadable(tmp_path: Path) -> None:
    markers = ["DAPI", "CD3", "CD45", "CD8", "CD4", "CD19", "CD20", "PanCK"]
    dest = tmp_path / "signature.yaml"
    out = scaffold.write_signature_template("tx", markers, dest)
    assert out == dest

    from macsima_pipeline.phenotype import signature as sig_mod

    sig = sig_mod.load_signature(out)
    names = set(sig.cell_type_names())
    assert {"T cell", "CD4 T cell", "CD8 T cell", "B cell", "Epithelial"} <= names
    assert list(sig.cell_types["T cell"].positive) == ["CD3", "CD45"]  # CD3e absent from panel
    txt = out.read_text()
    assert "PanCK" in txt and "version: 1" in txt


def test_write_signature_template_empty_markers_returns_none(tmp_path: Path) -> None:
    assert scaffold.write_signature_template("tx", [], tmp_path / "signature.yaml") is None


def test_write_signature_template_skips_existing_unless_force(tmp_path: Path) -> None:
    markers = ["DAPI", "CD3", "CD45"]
    dest = tmp_path / "signature.yaml"
    dest.write_text("KEEP\n")
    assert scaffold.write_signature_template("tx", markers, dest) is None
    assert dest.read_text() == "KEEP\n"  # curated edits never clobbered
    out = scaffold.write_signature_template("tx", markers, dest, force=True)
    assert out == dest
    assert "version: 1" in dest.read_text()
