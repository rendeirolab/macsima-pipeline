"""Unit tests for the native staging port (staging_core).

Uses tiny synthetic raw tiles carrying minimal OME-XML (StageLabel + Pixels + Channel), so the
tests are self-contained and CI-safe. Illumination correction is off (BaSiCPy exercised
separately) — these cover parsing, exposure-level ranking, grouping/backfill, OME structure,
and markers.csv content.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("tifffile")

import tifffile as tf
from macsima_pipeline import staging_core as sc

# filter, (emission, excitation)
_WAVE = {"DAPI": (461.0, 358.0), "FITC": (530.0, 470.0), "APC": (660.0, 650.0)}


def _ome_xml(name: str, filt: str, x: float, y: float, size: int = 4) -> str:
    em, ex = _WAVE[filt]
    return (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<OME xmlns="http://www.openmicroscopy.org/Schemas/OME/2016-06">'
        f'<Image ID="Image:0" Name="{name}">'
        f'<Pixels ID="Pixels:0" DimensionOrder="XYZCT" Type="uint16" '
        f'SizeX="{size}" SizeY="{size}" SizeC="1" SizeT="1" SizeZ="1" '
        f'PhysicalSizeX="0.00017" PhysicalSizeXUnit="mm" '
        f'PhysicalSizeY="0.00017" PhysicalSizeYUnit="mm" SignificantBits="16">'
        f'<Channel ID="Channel:0" Name="{filt}" SamplesPerPixel="1" '
        f'EmissionWavelength="{em}" EmissionWavelengthUnit="nm" '
        f'ExcitationWavelength="{ex}" ExcitationWavelengthUnit="nm"/>'
        "<TiffData/>"
        "</Pixels>"
        f'<StageLabel Name="StageLabel" X="{x}" XUnit="mm" Y="{y}" YUnit="mm" Z="0" ZUnit="mm"/>'
        "</Image></OME>"
    )


def _write_tile(path: Path, filt: str, x: float, y: float, fill: int, size: int = 4) -> None:
    arr = np.full((size, size), fill, dtype=np.uint16)
    tf.imwrite(str(path), arr, description=_ome_xml(path.name, filt, x, y, size), photometric="minisblack")


# marker, filter, exposures (one exposure -> single level; two -> exp-1/exp-2)
_PANEL = [
    ("DAPI", "DAPI", [10.0]),
    ("Keratin_5", "FITC", [100.0, 200.0]),  # underscore marker on purpose
    ("CD3", "APC", [50.0, 300.0]),
]


def _make_cycle(cycle_dir: Path, *, tiles=(1, 2), cycle: int = 1) -> None:
    cycle_dir.mkdir(parents=True, exist_ok=True)
    fill = 100
    for t in tiles:
        for src in ("S", "B"):
            for marker, filt, exps in _PANEL:
                for exp in exps:
                    a = "" if marker == "DAPI" else f"_A-{marker}_C-CL{marker[:2]}"
                    name = (
                        f"CYC-{cycle:03d}_SCN-001_ST-{src}_R-01_W-A01_ROI-001_F-{t:03d}"
                        f"{a}_D-{filt}_EXP-{exp:g}.tif"
                    )
                    # distinct per-tile position so we can assert; fill varies per plane
                    _write_tile(cycle_dir / name, filt, x=float(t), y=float(t) + 0.5, fill=fill)
                    fill = (fill + 7) % 500 + 1


def test_parse_name_underscore_marker() -> None:
    name = "CYC-005_SCN-001_ST-S_R-01_W-B01_ROI-003_F-002_A-Keratin_5_C-EPR23_D-APC_EXP-96.tif"
    p = sc.parse_name(name)
    assert p is not None
    # regex captures are raw strings (scan_cycle casts to int later)
    assert p["cycle"] == "005" and p["source"] == "S" and p["roi"] == "003" and p["tile"] == "002"
    assert p["marker"] == "Keratin_5"  # underscore preserved (not split)
    assert p["filter"] == "APC" and p["exposure_time"] == "96"


def test_scan_cycle_exposure_levels(tmp_path: Path) -> None:
    cyc = tmp_path / "6_Cycle1"
    _make_cycle(cyc)
    df = sc.scan_cycle(cyc)
    # DAPI has a single exposure -> level 1 only; the others rank low->high as 1,2
    dapi = df[df.marker == "DAPI"]
    assert set(dapi.exposure_level) == {1}
    ker = df[df.marker == "Keratin_5"]
    assert dict(zip(ker.exposure_time, ker.exposure_level)) == {100.0: 1, 200.0: 2}


def test_stage_cycle_outputs_and_ome(tmp_path: Path) -> None:
    cyc = tmp_path / "6_Cycle1"
    _make_cycle(cyc)
    root = tmp_path / "out"
    contrib = sc.stage_cycle(cyc, root, illumination_correction=False)

    # two exposure levels -> two sample dirs
    names = sorted(d.name for d in contrib)
    assert names == ["rack-01-well-A01-roi-001-exp-1", "rack-01-well-A01-roi-001-exp-2"]

    # inspect the exp-2 stain stack
    raw = root / "rack-01-well-A01-roi-001-exp-2" / "raw"
    stain = sorted(raw.glob("*src-S*.ome.tiff"))
    assert len(stain) == 1
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        from ome_types import from_xml

        with tf.TiffFile(stain[0]) as t:
            assert len(t.series) == 2  # one series per tile
            assert t.series[0].shape == (3, 4, 4)  # DAPI + Keratin_5 + CD3
            ome = from_xml(t.ome_metadata)
    px = ome.images[0].pixels
    # reference marker first, order consistent with planes; underscore kept in channel id
    ch_markers = [c.id.split(":")[-1] for c in px.channels]
    assert ch_markers[0] == "DAPI"
    assert "Keratin_5" in ch_markers
    # DAPI (backfilled) exposure is its own (10), not scrambled
    dapi_plane = next(pl for pl, m in zip(px.planes, ch_markers) if m == "DAPI")
    assert dapi_plane.exposure_time == 10.0


def test_stage_cycle_appends_reference_to_exp2(tmp_path: Path) -> None:
    cyc = tmp_path / "6_Cycle1"
    _make_cycle(cyc)
    root = tmp_path / "out"
    sc.stage_cycle(cyc, root, illumination_correction=False)
    # exp-2 group had no DAPI acquisition; append_reference must add it -> filename lists DAPI
    f = sorted((root / "rack-01-well-A01-roi-001-exp-2" / "raw").glob("*src-S*.ome.tiff"))[0]
    assert "markers-DAPI__" in f.name  # DAPI first in the marker list


def test_markers_csv_content(tmp_path: Path) -> None:
    cyc = tmp_path / "6_Cycle1"
    _make_cycle(cyc)
    root = tmp_path / "out"
    samples = sc.stage_roi(
        tmp_path,  # roi_dir: contains the single cycle folder
        root,
        cycle_glob="*Cycle*",
        illumination_correction=False,
    )
    exp2 = next(s for s in samples if s.name.endswith("exp-2"))
    m = pd.read_csv(exp2 / "markers.csv")
    assert list(m.columns) == sc.MARKERS_COLUMNS
    # background (src-B) rows marked remove=TRUE and renamed bg_...
    bg = m[m.remove == True]  # noqa: E712
    assert all(name.startswith("bg_001_") for name in bg.marker_name)
    # stain rows: dash spelling, DAPI has empty background, others reference their bg row
    stain = m[m.remove != True]  # noqa: E712
    assert "Keratin-5" in set(stain.marker_name)  # underscore -> dash in markers.csv
    ker = stain[stain.marker_name == "Keratin-5"].iloc[0]
    assert ker.background == "bg_001_Keratin-5-FITC"
    dapi = stain[stain.marker_name == "DAPI"].iloc[0]
    assert pd.isna(dapi.background) or dapi.background == ""
    # channel_number is a contiguous 1..N
    assert m.channel_number.tolist() == list(range(1, len(m) + 1))
