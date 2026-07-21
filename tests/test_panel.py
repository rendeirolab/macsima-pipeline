"""Tests for the pre-staging marker panel sanity check.

The panel step parses filenames only, so these build raw trees with empty (touched) tiles.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from macsima_pipeline import panel
from macsima_pipeline.config import load_config

_DEFAULT = """\
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

# per cycle: list of (marker, filter, [exposures]); DAPI has no A- token
_CYCLES = {
    1: [("DAPI", "DAPI", [10]), ("CD3", "FITC", [100, 200]), ("CD8", "APC", [50, 300])],
    2: [("DAPI", "DAPI", [10]), ("Keratin_5", "FITC", [100, 200])],
}


def _touch_cycle(cycle_dir: Path, cycle: int, spec) -> None:
    cycle_dir.mkdir(parents=True, exist_ok=True)
    for src in ("S", "B"):
        for marker, filt, exps in spec:
            for exp in exps:
                a = "" if marker == "DAPI" else f"_A-{marker}_C-CL"
                name = (
                    f"CYC-{cycle:03d}_SCN-001_ST-{src}_R-01_W-A01_ROI-001_F-001"
                    f"{a}_D-{filt}_EXP-{exp:g}.tif"
                )
                (cycle_dir / name).touch()


def _make_raw(root: Path, cycles=_CYCLES, rois=("ROI1", "ROI2")) -> None:
    for roi in rois:
        for c, spec in cycles.items():
            _touch_cycle(root / roi / f"{c}_Cycle{c}", c, spec)


def _cfg(tmp_path: Path, raw: Path):
    (tmp_path / "default.yaml").write_text(_DEFAULT)
    (tmp_path / "exp.yaml").write_text(
        f'extends: "default.yaml"\nexperiment:\n  name: tx\n  raw_root: {raw}\n'
    )
    cfg = load_config(tmp_path / "exp.yaml")
    cfg.paths.work_dir = tmp_path
    return cfg


def test_scan_experiment_covers_all_rois_and_cycles(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    _make_raw(raw)
    scan = panel.scan_experiment(_cfg(tmp_path, raw))
    assert set(scan["roi_name"]) == {"ROI1", "ROI2"}
    assert set(scan["cycle"]) == {1, 2}


def test_marker_panel_summary(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    _make_raw(raw)
    cfg = _cfg(tmp_path, raw)
    p = panel.write_marker_panel(cfg, panel.scan_experiment(cfg))
    assert cfg.marker_panel_path().exists()
    # markers spelled as in markers.csv (underscore -> dash); reference flagged; n_rois counted
    row = p[(p.cycle_number == 2) & (p.marker_name == "Keratin-5")].iloc[0]
    assert row.Filter == "FITC" and row.exposure_levels == 2 and row.n_rois == 2
    assert bool(p[p.marker_name == "DAPI"].is_reference.all())


def test_sanity_check_passes_on_good_panel(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    _make_raw(raw)
    cfg = _cfg(tmp_path, raw)
    warns = panel.sanity_check(cfg, panel.scan_experiment(cfg))
    assert warns == []


def test_sanity_check_requires_reference_each_cycle(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    # cycle 2 has no DAPI -> must fail
    bad = {1: _CYCLES[1], 2: [("CD8", "APC", [50, 300])]}
    _make_raw(raw, cycles=bad)
    cfg = _cfg(tmp_path, raw)
    with pytest.raises(RuntimeError, match="reference marker"):
        panel.sanity_check(cfg, panel.scan_experiment(cfg))


def test_generate_returns_frame_and_scaffolds_signature(tmp_path: Path) -> None:
    from macsima_pipeline import scaffold
    from macsima_pipeline.phenotype import signature as sig_mod

    raw = tmp_path / "raw"
    _make_raw(raw)
    cfg = _cfg(tmp_path, raw)
    df = panel.generate(cfg)  # returns the marker-panel summary frame
    assert cfg.marker_panel_path().exists()
    assert "marker_name" in df.columns

    # the CLI unions marker_name across experiments, then scaffolds one shared signature
    dest = tmp_path / "signature.yaml"
    out = scaffold.write_signature_template("tx", list(dict.fromkeys(df["marker_name"])), dest)
    assert out == dest
    sig = sig_mod.load_signature(out)
    assert {"T cell", "CD8 T cell"} <= set(sig.cell_type_names())  # CD3 / CD8 in the panel
    # never clobbers a curated file
    assert scaffold.write_signature_template("tx", ["DAPI"], dest) is None
