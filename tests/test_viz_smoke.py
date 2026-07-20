"""Viz smoke tests using a small synthetic multi-channel TIFF."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("tifffile")
pytest.importorskip("matplotlib")

import matplotlib

matplotlib.use("Agg")

import tifffile
import pandas as pd

from macsima_pipeline.config import load_config
from macsima_pipeline.viz import io, render, workers


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
