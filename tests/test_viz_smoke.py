"""Viz smoke tests using a small synthetic multi-channel TIFF."""

# ruff: noqa: E402 - pytest.importorskip must run before optional heavy imports.

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("tifffile")
pytest.importorskip("matplotlib")
ad = pytest.importorskip("anndata")

import matplotlib

matplotlib.use("Agg")

import tifffile
import pandas as pd

from macsima_pipeline.config import load_config
from macsima_pipeline.viz import cell_maps, channel_qc, io, render, workers


def _write_default(p: Path) -> None:
    p.write_text(
        """\
experiment:
  name: REQUIRED
  raw_root: REQUIRED
containers:
  macsima2mc_sif: macsima2mc.sif
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


def _make_pyramid_tiff(path: Path, c: int = 3, y: int = 1024, x: int = 1024) -> None:
    """Write a simple multi-channel pyramidal OME-TIFF.

    Layout: 3 levels, halving each downsample. Uses the bigtiff-style series
    write pattern that tifffile supports.
    """
    rng = np.random.default_rng(42)
    full = rng.integers(0, 4096, size=(c, y, x), dtype=np.uint16)
    half = full[:, ::2, ::2]
    quarter = full[:, ::4, ::4]
    with tifffile.TiffWriter(str(path), bigtiff=True) as tw:
        tw.write(full, photometric="minisblack", subifds=2)
        tw.write(half, photometric="minisblack")
        tw.write(quarter, photometric="minisblack")


def _make_channel_qc_tiff(path: Path) -> None:
    full = np.zeros((4, 64, 64), dtype=np.uint16)
    full[0] = 5
    full[1] = 10
    full[1, 16:48, 16:48] = 100
    full[2] = np.iinfo(np.uint16).max
    full[3] = 10
    full[3, :8, :] = 200
    full[3, -8:, :] = 200
    full[3, :, :8] = 200
    full[3, :, -8:] = 200
    with tifffile.TiffWriter(str(path), bigtiff=True) as tw:
        tw.write(full, photometric="minisblack")


def test_pyramid_pick(tmp_path: Path) -> None:
    p = tmp_path / "img.ome.tif"
    _make_pyramid_tiff(p, c=2, y=1024, x=1024)
    lvl = io.pick_level(p, target_max_dim=300)
    # We want the coarsest level still >= target. Pyramid: 1024, 512, 256. target=300 -> level 1 (512).
    assert lvl.level == 1
    assert max(lvl.shape[-2:]) == 512


def test_read_channel_shape(tmp_path: Path) -> None:
    p = tmp_path / "img.ome.tif"
    _make_pyramid_tiff(p, c=3, y=512, x=512)
    arr = io.read_channel(p, level=0, channel_ix=1)
    assert arr.shape == (512, 512)
    assert arr.dtype == np.uint16


def test_plot_marker_across_rois(tmp_path: Path) -> None:
    _write_default(tmp_path / "default.yaml")
    (tmp_path / "exp.yaml").write_text(
        f"""\
extends: "default.yaml"
experiment:
  name: smoke
  raw_root: {tmp_path}
viz:
  target_max_dim: 256
  grid_ncols: 2
  fig_size_per_panel: [2, 2]
  dpi: 72
  output_format: png
  rasterized: true
  parallel:
    workers: 1
"""
    )
    cfg = load_config(tmp_path / "exp.yaml")
    cfg.paths.work_dir = tmp_path

    # Two fake ROIs with mcmicro-style stems
    p1 = tmp_path / "rack-01-well-C01-roi-001-exp-2.ome.tif"
    p2 = tmp_path / "rack-01-well-C01-roi-002-exp-2.ome.tif"
    _make_pyramid_tiff(p1, c=2, y=512, x=512)
    _make_pyramid_tiff(p2, c=2, y=512, x=512)

    rois = [render.resolve_roi(p1, cfg), render.resolve_roi(p2, cfg)]
    out = tmp_path / "marker0.png"
    render.plot_marker_across_rois(rois, marker_ix=0, marker_name="DAPI", cfg=cfg, out_path=out)
    assert out.exists() and out.stat().st_size > 0


def test_plot_all_markers_for_roi(tmp_path: Path) -> None:
    _write_default(tmp_path / "default.yaml")
    (tmp_path / "exp.yaml").write_text(
        f"""\
extends: "default.yaml"
experiment:
  name: smoke
  raw_root: {tmp_path}
viz:
  target_max_dim: 256
  grid_ncols: 2
  fig_size_per_panel: [2, 2]
  dpi: 72
  output_format: png
"""
    )
    cfg = load_config(tmp_path / "exp.yaml")
    cfg.paths.work_dir = tmp_path

    p1 = tmp_path / "rack-01-well-C01-roi-003-exp-2.ome.tif"
    _make_pyramid_tiff(p1, c=3, y=512, x=512)
    roi = render.resolve_roi(p1, cfg)
    ci = pd.DataFrame({"marker_name": ["DAPI", "CD8", "CD4"]})

    out = tmp_path / "roi.png"
    render.plot_all_markers_for_roi(roi, ci, cfg, out)
    assert out.exists() and out.stat().st_size > 0


def _cfg(tmp_path: Path, *, output_format: str = "png"):
    _write_default(tmp_path / "default.yaml")
    (tmp_path / "exp.yaml").write_text(
        f'''extends: "default.yaml"
experiment:
  name: smoke
  raw_root: {tmp_path}
viz:
  target_max_dim: 64
  grid_ncols: 5
  fig_size_per_panel: [2, 2]
  dpi: 36
  output_format: {output_format}
'''
    )
    cfg = load_config(tmp_path / "exp.yaml")
    cfg.paths.work_dir = tmp_path
    return cfg


@pytest.mark.parametrize(
    ("n_panels", "shape", "remaining"),
    [(1, (1, 1), 1), (7, (2, 5), 7), (12, (3, 5), 12)],
)
def test_compact_grid_dimensions_and_unused_axes(
    tmp_path: Path, n_panels: int, shape: tuple[int, int], remaining: int
) -> None:
    cfg = _cfg(tmp_path)
    fig, axes = render._new_grid(n_panels, cfg)
    try:
        assert axes.shape == shape
        assert tuple(fig.get_size_inches()) == (shape[1] * 2, shape[0] * 2)
        render._trim_axes(axes, n_panels)
        render._layout_grid(fig, cfg)
        assert len(fig.axes) == remaining
        pars = fig.subplotpars
        assert pars.wspace == cfg.viz.grid_wspace
        assert pars.hspace == cfg.viz.grid_hspace
        assert pars.left <= 0.01 and pars.right >= 0.99
    finally:
        render.plt.close(fig)


def test_empty_grid_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="no panels"):
        render._new_grid(0, _cfg(tmp_path))


def test_filtered_markers_keep_physical_tiff_indices(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    sample = tmp_path / "sample"
    sample.mkdir()
    pd.DataFrame(
        {
            "marker_name": ["background", "CD8", "duplicate CD8", "CD4"],
            "remove": [True, False, True, False],
        }
    ).to_csv(sample / "markers.csv", index=False)
    ci = workers._load_channel_info(cfg, sample, bg=False)
    assert ci["marker_name"].tolist() == ["CD8", "CD4"]
    assert ci["channel_index"].tolist() == [1, 3]


def test_channel_qc_metric_cases(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    qc = cfg.viz.channel_qc
    bg = np.full((32, 32), 10, dtype=np.uint16)
    bright = np.full((32, 32), 10, dtype=np.uint16)
    bright[8:24, 8:24] = 100
    metrics = channel_qc.compute_channel_metrics(bright, qc, background_img=bg)
    assert metrics["snr_p99"] > qc.min_snr
    assert metrics["positive_fraction"] == pytest.approx(0.25)

    weak = np.full((32, 32), 10, dtype=np.uint16)
    metrics = channel_qc.compute_channel_metrics(weak, qc)
    assert metrics["snr_p99"] == pytest.approx(0.0)
    assert metrics["positive_fraction"] == pytest.approx(0.0)

    saturated = np.full((32, 32), np.iinfo(np.uint16).max, dtype=np.uint16)
    metrics = channel_qc.compute_channel_metrics(saturated, qc)
    assert metrics["saturated_fraction"] == pytest.approx(1.0)

    uneven = np.full((32, 32), 10, dtype=np.uint16)
    uneven[:4, :] = 200
    uneven[-4:, :] = 200
    uneven[:, :4] = 200
    uneven[:, -4:] = 200
    metrics = channel_qc.compute_channel_metrics(uneven, qc)
    assert metrics["edge_center_ratio"] > 2.5


def test_channel_qc_variant_outputs_and_bg_comparison(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path, output_format="pdf")
    cfg.viz.parallel.workers = 1
    cfg.viz.channel_qc.workers = 1
    sample = tmp_path / "sample"
    sample.mkdir()
    pd.DataFrame(
        {
            "marker_name": ["bg_001_CD8-FITC", "CD8", "SAT", "UNEVEN"],
            "cycle_number": [1, 1, 1, 1],
            "Filter": ["FITC", "FITC", "APC", "PE"],
            "background": [np.nan, "bg_001_CD8-FITC", np.nan, np.nan],
            "exposure": [100, 100, 50, 40],
            "remove": [True, False, False, False],
        }
    ).to_csv(sample / "markers.csv", index=False)
    path = sample / "rack-01-well-C01-roi-001-exp-2.ome.tif"
    _make_channel_qc_tiff(path)
    roi = render.resolve_roi(path, cfg)
    channel_info = workers._load_channel_info(cfg, sample, bg=False)

    df = channel_qc.run_variant_qc(cfg, [roi], channel_info, sample, bg=False)
    assert df is not None
    assert channel_qc.is_valid_qc_csv(channel_qc.qc_csv_path(cfg, bg=False))
    assert render.is_valid_output(channel_qc.qc_pdf_path(cfg, bg=False), "pdf")
    assert set(["mean", "p99", "snr_p99", "positive_fraction", "flags"]).issubset(df.columns)
    cd8 = df[df["marker_name"].eq("CD8")].iloc[0]
    assert cd8["channel_index"] == 1
    assert cd8["background_channel_index"] == 0
    assert bool(df[df["marker_name"].eq("SAT")]["saturated"].iloc[0])
    assert bool(df[df["marker_name"].eq("UNEVEN")]["uneven"].iloc[0])

    bg_df = df.copy()
    bg_df["variant"] = "bg-sub"
    bg_df["bg_subtracted"] = True
    bg_df["median"] = bg_df["median"] * 0.8
    bg_df["p99"] = bg_df["p99"] * 0.8
    bg_df.to_csv(channel_qc.qc_csv_path(cfg, bg=True), index=False)
    comparison = channel_qc.write_bg_comparison(cfg)
    assert comparison is not None
    assert channel_qc.comparison_csv_path(cfg).is_file()
    assert render.is_valid_output(channel_qc.comparison_pdf_path(cfg), "pdf")


def test_roi_grid_reads_physical_channels_and_keeps_titles(tmp_path: Path, monkeypatch) -> None:
    cfg = _cfg(tmp_path)
    roi = render.RoiImage("ROI9", tmp_path / "unused.tif", io.PyramidLevel(0, (4, 8, 8)))
    ci = pd.DataFrame({"marker_name": ["CD8", "CD4"], "channel_index": [1, 3]})
    seen: list[int] = []
    captured = {}

    def fake_read(path, level, channel_ix):
        seen.append(channel_ix)
        return np.full((8, 8), channel_ix, dtype=np.uint16)

    def capture(fig, out, cfg):
        captured["figure_title"] = fig._suptitle.get_text()
        captured["panel_titles"] = [ax.get_title() for ax in fig.axes]

    monkeypatch.setattr(render, "read_channel", fake_read)
    monkeypatch.setattr(render, "_save_figure_atomic", capture)
    render.plot_all_markers_for_roi(roi, ci, cfg, tmp_path / "unused.png")
    assert seen == [1, 3]
    assert captured["figure_title"] == "ROI: ROI9"
    assert [title.split(" | ")[0] for title in captured["panel_titles"]] == ["CD8", "CD4"]


def test_output_validation_and_atomic_save_failure(tmp_path: Path, monkeypatch) -> None:
    cfg = _cfg(tmp_path, output_format="pdf")
    out = tmp_path / "grid.pdf"
    out.write_bytes(b"%PDF-old\n%%EOF\n")
    assert render.is_valid_output(out, "pdf")
    invalid = tmp_path / "invalid.pdf"
    invalid.write_bytes(b"")
    assert not render.is_valid_output(invalid, "pdf")

    class BrokenFigure:
        def savefig(self, path, **kwargs):
            Path(path).write_bytes(b"%PDF-incomplete")
            raise RuntimeError("interrupted")

    with pytest.raises(RuntimeError, match="interrupted"):
        render._save_figure_atomic(BrokenFigure(), out, cfg)
    assert out.read_bytes() == b"%PDF-old\n%%EOF\n"
    assert not list(tmp_path.glob(".*.tmp.pdf"))


def _synthetic_adata(*, with_xy: bool = True, xy_names: tuple[str, str] = ("x", "y")):
    obs = pd.DataFrame(
        {
            "ROI": ["ROI1", "ROI1", "ROI2", "ROI2"],
            "Sample": ["A", "A", "B", "B"],
            "Patient_ID": ["P1", "P1", "P2", "P2"],
        },
        index=["cell1", "cell2", "cell3", "cell4"],
    )
    if with_xy:
        obs[xy_names[0]] = [10.0, 40.0, 8.0, 30.0]
        obs[xy_names[1]] = [5.0, 45.0, 12.0, 35.0]
    var = pd.DataFrame(index=["DAPI", "CD8", "CD4", "Ki67"])
    x = np.array(
        [
            [120.0, 5.0, 0.0, 2.0],
            [90.0, 0.0, 12.0, 0.0],
            [140.0, 18.0, 4.0, 7.0],
            [80.0, 3.0, 0.0, 1.0],
        ]
    )
    return ad.AnnData(X=x, obs=obs, var=var)


def test_cell_map_xy_column_detection() -> None:
    assert cell_maps.resolve_xy_columns(_synthetic_adata().obs) == ("x", "y")
    assert cell_maps.resolve_xy_columns(
        _synthetic_adata(xy_names=("centroid_x", "centroid_y")).obs
    ) == ("centroid_x", "centroid_y")


def test_plot_cell_map_qc_summary_pdf(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path, output_format="pdf")
    out = tmp_path / "cell_maps.pdf"
    cell_maps.plot_cell_map_qc_summary(_synthetic_adata(), cfg, out, bg=False, h5ad_path=tmp_path / "cells.h5ad")

    assert out.read_bytes().startswith(b"%PDF-")
    assert render.is_valid_output(out, "pdf")


def test_plot_cell_map_qc_summary_without_coordinates(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path, output_format="pdf")
    out = tmp_path / "cell_maps_missing_xy.pdf"
    cell_maps.plot_cell_map_qc_summary(_synthetic_adata(with_xy=False), cfg, out, bg=True)

    assert render.is_valid_output(out, "pdf")


def test_plot_cell_map_colored_by_cell_type(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path, output_format="pdf")
    adata = _synthetic_adata()
    adata.obs["cell_type"] = pd.Categorical(["T cell", "B cell", "T cell", "Macrophage"])
    out = tmp_path / "cell_maps_typed.pdf"
    cell_maps.plot_cell_map_qc_summary(adata, cfg, out, bg=False)
    assert render.is_valid_output(out, "pdf")
    # cell_type must not be mistaken for ROI-level metadata
    assert "cell_type" in cell_maps._INTERNAL_OBS_COLUMNS
    assert "cell_type" not in cell_maps._metadata_columns(cell_maps._obs_with_roi(adata), ("x", "y"))


def test_plot_cell_map_qc_summary_phenotype_page(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path, output_format="pdf")
    adata = _synthetic_adata()
    adata.obs["cell_type"] = pd.Categorical(["T cell", "B cell", "T cell", "Macrophage"])
    adata.uns["phenotype"] = {
        "engines": ["astir", "flowsom"],
        "primary_engine": "astir",
        "agreement": {"accuracy": 0.9, "cohen_kappa": 0.8, "adjusted_rand": 0.75},
        "spatial_qc": {
            "labels": ["T cell", "B cell", "Macrophage"],
            "homophily": {"overall": 0.6, "per_type": {"T cell": 0.7}},
            "nhood_zscore": np.eye(3, dtype="float32"),
        },
        "composition": pd.crosstab(adata.obs["ROI"], adata.obs["cell_type"], normalize="index"),
    }
    out = tmp_path / "cell_maps_pheno.pdf"
    cell_maps.plot_cell_map_qc_summary(adata, cfg, out, bg=False)
    assert render.is_valid_output(out, "pdf")
