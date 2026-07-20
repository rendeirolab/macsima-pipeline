"""Native reimplementation of the macsima2mc staging step (drops the Docker container).

Ports SchapiroLabor/macsima2mc v1.3.1 (``tools.py`` / ``ome_schema.py`` / ``mc_tools.py`` /
``illumination_corr.py``) to run in-process with ``tifffile`` + ``ome-types``. It reproduces
the exact output contract that mcmicro (ashlar) and the downstream stages consume:

    <sample>/raw/<corr_>cycle-NNN-src-{S,B}-...-markers-A__B-filters-X__Y.ome.tiff
    <sample>/markers.csv

Each OME-TIFF is multi-series: one OME ``Image`` per field/tile, ``C Y X`` uint16, with each
``Plane`` carrying the tile stage position (from the raw tile's ``StageLabel``) and exposure
time so ashlar can stitch the mosaic.

Intentional deviations from macsima2mc (each strictly more correct and invisible to the
downstream stages, which key off ``marker_name`` + physical channel position, never the
``exposure`` column or the embedded OME channel names):

* The embedded OME channel metadata order matches the physical plane order (reference marker
  first). macsima2mc built the embedded channel block from an unordered grouping, so its
  channel *names* could be permuted relative to the actual pixel planes.
* Because ordering is consistent, ``markers.csv`` ``exposure`` values line up with their marker
  (macsima2mc's were shuffled by the ordering mismatch above).
"""

from __future__ import annotations

import logging
import re
import warnings
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile as tf

log = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
#  Raw MACSima filename parsing (macsima2mc templates.macsima_pattern v2)      #
# --------------------------------------------------------------------------- #

# NOTE: `marker` uses `_A-(.*?)_C-` — marker tokens may themselves contain underscores
# (e.g. `A-Cytokeratin_5_C-...`), so never split the name on "_".
RAW_PATTERNS: dict[str, str] = {
    "cycle": r"CYC-(\d+)",
    "source": r"_ST-(.*?)_",
    "rack": r"_R-(\d+)",
    "well": r"_W-(.*?\d+)",
    "roi": r"_ROI-(\d+)",
    "tile": r"_F-(\d+)",
    "exposure_time": r"_EXP-(\d+(?:\.\d+)?)",
    "marker": r"_A-(.*?)_C-",
    "filter": r"_D-(.*?)_",
}

OME_NS = {"ome": "http://www.openmicroscopy.org/Schemas/OME/2016-06"}


def parse_name(name: str) -> dict[str, str] | None:
    """Extract MACSima acquisition tokens from a raw tile filename.

    Returns None if the mandatory `cycle`/`tile`/`filter` tokens are absent (e.g. a stray
    non-image file). `marker` is left empty for DAPI-style files that carry no `A-` token;
    callers backfill it from the filter (see :func:`scan_cycle`).
    """
    out: dict[str, str] = {}
    for key, pat in RAW_PATTERNS.items():
        m = re.search(pat, name)
        out[key] = m.group(1) if m else ""
    if not (out["cycle"] and out["tile"] and out["filter"]):
        return None
    return out


def scan_cycle(cycle_dir: Path, ref_marker: str = "DAPI") -> pd.DataFrame:
    """Parse every ``*.tif`` in one MACSima cycle folder into an acquisition table.

    Mirrors ``tools.cycle_info``: DAPI files have no ``A-`` token, so ``marker`` is set to the
    reference marker where ``filter == ref_marker``; ``exposure_level`` is the dense rank of
    ``exposure_time`` within each ``(source, marker, filter)`` group (so the lowest exposure is
    level 1 -> ``exp-1``, the next is level 2 -> ``exp-2``, ...).
    """
    rows: list[dict] = []
    for p in sorted(cycle_dir.glob("*.tif")):
        parsed = parse_name(p.name)
        if parsed is None:
            continue
        parsed["full_path"] = str(p)
        parsed["img_name"] = p.name
        rows.append(parsed)
    if not rows:
        raise FileNotFoundError(f"No parsable MACSima tiles in {cycle_dir}")

    df = pd.DataFrame(rows)
    df["tile"] = df["tile"].astype(int)
    df["cycle"] = df["cycle"].astype(int)
    df["exposure_time"] = df["exposure_time"].astype(float)
    # DAPI files carry no A- token -> derive marker from the reference filter.
    df.loc[df["filter"] == ref_marker, "marker"] = ref_marker
    df["exposure_level"] = (
        df.groupby(["source", "marker", "filter"])["exposure_time"].rank(method="dense").astype(int)
    )
    return df


# --------------------------------------------------------------------------- #
#  Raw tile OME metadata (StageLabel position, pixel size, wavelengths)        #
# --------------------------------------------------------------------------- #


def read_tile_meta(path: str | Path) -> dict:
    """Read the handful of OME-XML attributes needed to build the output OME.

    Raw MACSima tiles embed OME-XML with a ``StageLabel`` (stage position — what ashlar needs),
    ``Pixels`` (physical size, dimensions, dtype, significant bits) and a ``Channel``
    (excitation/emission wavelengths). We parse the XML directly (fast, and avoids ome-types'
    noisy pydantic ID-cast warnings on every tile).
    """
    with tf.TiffFile(path) as t:
        xml = t.ome_metadata
    root = ET.fromstring(xml)
    img = root.find("ome:Image", OME_NS)
    px = img.find("ome:Pixels", OME_NS)
    ch = px.find("ome:Channel", OME_NS)
    sl = img.find("ome:StageLabel", OME_NS)
    if sl is None:
        raise ValueError(f"raw tile has no StageLabel (stage position): {path}")

    def _f(x):
        return None if x is None else float(x)

    return {
        "position_x": _f(sl.get("X")),
        "position_y": _f(sl.get("Y")),
        "position_x_unit": sl.get("XUnit"),
        "position_y_unit": sl.get("YUnit"),
        "physical_size_x": _f(px.get("PhysicalSizeX")),
        "physical_size_x_unit": px.get("PhysicalSizeXUnit"),
        "physical_size_y": _f(px.get("PhysicalSizeY")),
        "physical_size_y_unit": px.get("PhysicalSizeYUnit"),
        "size_x": int(px.get("SizeX")),
        "size_y": int(px.get("SizeY")),
        "type": px.get("Type"),
        "significant_bits": int(px.get("SignificantBits")),
        "emission_wavelength": _f(ch.get("EmissionWavelength")) if ch is not None else None,
        "emission_wavelength_unit": ch.get("EmissionWavelengthUnit") if ch is not None else None,
        "excitation_wavelength": _f(ch.get("ExcitationWavelength")) if ch is not None else None,
        "excitation_wavelength_unit": ch.get("ExcitationWavelengthUnit") if ch is not None else None,
    }


# --------------------------------------------------------------------------- #
#  Channel ordering + reference-marker backfill                               #
# --------------------------------------------------------------------------- #

GroupIndex = tuple[str, str, str, str, int]  # (source, rack, well, roi, exposure_level)
GROUP_KEYS = ["source", "rack", "well", "roi", "exposure_level"]


def conform_markers(pairs: list[tuple[str, str]], ref_marker: str) -> list[tuple[str, str]]:
    """Order (marker, filter) pairs with the reference marker first, others in first-seen order.

    Mirrors ``tools.conform_markers`` but with a deterministic, insertion-ordered tail (rather
    than pandas' tie-broken ``value_counts`` order) so the physical channel order is stable.
    """
    rest = [mf for mf in pairs if mf[0] != ref_marker]
    return [(ref_marker, ref_marker)] + rest


def _unique_pairs(group: pd.DataFrame) -> list[tuple[str, str]]:
    seen: dict[tuple[str, str], None] = {}
    for m, f in zip(group["marker"], group["filter"]):
        seen.setdefault((m, f), None)
    return list(seen)


def append_reference(
    group: pd.DataFrame, index: GroupIndex, cycle_df: pd.DataFrame, ref_marker: str
) -> pd.DataFrame:
    """Backfill the reference marker into a group that lacks it.

    A cycle typically acquires DAPI at a single exposure, so the higher exposure-level group
    (``exp-2``) has no DAPI. macsima2mc pulls the reference rows from another exposure level of
    the same ``(source, rack, well, roi)``; we do the same so every staged stack has a DAPI
    channel for registration.
    """
    src, rack, well, roi, _ = index
    ref_rows = cycle_df[
        (cycle_df["source"] == src)
        & (cycle_df["rack"] == rack)
        & (cycle_df["well"] == well)
        & (cycle_df["roi"] == roi)
        & (cycle_df["marker"] == ref_marker)
    ]
    if ref_rows.empty:
        raise ValueError(
            f"reference marker {ref_marker!r} absent from group {index} and its "
            f"(source,rack,well,roi) — cannot build a reference channel"
        )
    # Use the lowest exposure level of the reference (it usually has only one).
    lvl = ref_rows["exposure_level"].min()
    ref_rows = ref_rows[ref_rows["exposure_level"] == lvl]
    return pd.concat([group, ref_rows], ignore_index=True)


# --------------------------------------------------------------------------- #
#  Output naming (macsima2mc tools.cast_*)                                    #
# --------------------------------------------------------------------------- #


def _fname_token(s: str) -> str:
    """macsima2mc replaces '_' with '-' in marker/filter names inside file/dir names."""
    return s.replace("_", "-")


def sample_dir_name(index: GroupIndex) -> str:
    src, rack, well, roi, exp = index
    return f"rack-{rack}-well-{well}-roi-{roi}-exp-{exp}"


def stack_file_name(cycle: int, index: GroupIndex, conformed: list[tuple[str, str]]) -> str:
    src, rack, well, roi, exp = index
    markers = "__".join(_fname_token(m) for m, _ in conformed)
    filters = "__".join(_fname_token(f) for _, f in conformed)
    return (
        f"cycle-{int(cycle):03d}-src-{src}-rack-{rack}-well-{well}-roi-{roi}-exp-{exp}"
        f"-markers-{markers}-filters-{filters}.ome.tiff"
    )


# --------------------------------------------------------------------------- #
#  OME-XML builder (mirror of macsima2mc ome_schema, channel order fixed)      #
# --------------------------------------------------------------------------- #


def _build_ome_xml(
    tiles: list[int],
    conformed: list[tuple[str, str]],
    meta_by_tile: dict[int, list[dict]],
    exposures: list[float],
) -> str:
    """Build multi-series OME-XML: one Image per tile, channels in `conformed` order.

    ``meta_by_tile[tile]`` is the per-channel metadata list (same order as ``conformed``);
    ``exposures`` is the per-channel exposure time (constant across tiles).
    """
    import platform

    import ome_types
    from ome_types.model import OME, Channel, Image, Pixels, Plane, TiffData

    n_ch = len(conformed)
    images = []
    for i, tile in enumerate(tiles):
        cmeta = meta_by_tile[tile]
        base = cmeta[0]
        channels = [
            Channel(
                id=f"Channel:{100 + int(tile)}:{ch}:{conformed[ch][0]}",
                color=-1,  # white (0xFFFFFFFF); downstream ignores channel color
                emission_wavelength=cmeta[ch]["emission_wavelength"],
                emission_wavelength_unit=cmeta[ch]["emission_wavelength_unit"],
                excitation_wavelength=cmeta[ch]["excitation_wavelength"],
                excitation_wavelength_unit=cmeta[ch]["excitation_wavelength_unit"],
            )
            for ch in range(n_ch)
        ]
        planes = [
            Plane(
                the_c=ch,
                the_t=0,
                the_z=0,
                position_x=cmeta[ch]["position_x"],
                position_y=cmeta[ch]["position_y"],
                position_z=0,
                exposure_time=exposures[ch],
                position_x_unit=cmeta[ch]["position_x_unit"],
                position_y_unit=cmeta[ch]["position_y_unit"],
            )
            for ch in range(n_ch)
        ]
        tiff = [TiffData(first_c=ch, ifd=n_ch * i + ch, plane_count=1) for ch in range(n_ch)]
        pixels = Pixels(
            id=f"Pixels:{tile}",
            dimension_order="XYCZT",
            size_c=n_ch,
            size_t=1,
            size_x=base["size_x"],
            size_y=base["size_y"],
            size_z=1,
            type=base["type"],
            big_endian=False,
            channels=channels,
            interleaved=False,
            physical_size_x=base["physical_size_x"],
            physical_size_x_unit=base["physical_size_x_unit"],
            physical_size_y=base["physical_size_y"],
            physical_size_y_unit=base["physical_size_y_unit"],
            physical_size_z=1.0,
            planes=planes,
            significant_bits=base["significant_bits"],
            tiff_data_blocks=tiff,
        )
        images.append(Image(id=f"Image:{i}", pixels=pixels))

    ome = OME()
    ome.creator = f"{ome_types.__name__} {ome_types.__version__} / python version- {platform.python_version()}"
    ome.images = images
    return ome_types.to_xml(ome)


# --------------------------------------------------------------------------- #
#  Illumination correction (BaSiCPy)                                          #
# --------------------------------------------------------------------------- #


def apply_illumination_correction(stack: np.ndarray, n_channels: int) -> np.ndarray:
    """Per-channel BaSiCPy flatfield correction (mirror of ``illumination_corr.apply_corr``).

    ``stack`` is ``(n_tiles * n_channels, Y, X)`` laid out tile-major, so channel ``c`` lives at
    indices ``c, c + n_channels, c + 2*n_channels, ...``. Uses the pytorch BaSiCPy build
    (peng-lab/BaSiCPy); imported lazily so dry-runs / metadata-only paths don't need it.
    """
    from basicpy import BaSiC

    corr = np.zeros(stack.shape, dtype=stack.dtype)
    total = stack.shape[0]
    for c in range(n_channels):
        idx = list(range(c, total, n_channels))
        imgs = stack[idx, :, :]
        signal = imgs[[i for i in range(imgs.shape[0]) if imgs[i].sum() > 0], :, :]
        basic = BaSiC(
            get_darkfield=False,
            smoothness_flatfield=1.0,
            fitting_mode="approximate",
            sort_intensity=True,
        )
        basic.fit(signal)
        ffp = basic.flatfield
        corr[idx, :, :] = np.uint16(np.clip(imgs.astype(float) / ffp, 0, 65535))
    return corr


# --------------------------------------------------------------------------- #
#  Stack a single cycle folder                                                #
# --------------------------------------------------------------------------- #


def stage_cycle(
    cycle_dir: Path,
    sample_root: Path,
    *,
    ref_marker: str = "DAPI",
    illumination_correction: bool = True,
    hi_exposure_only: bool = False,
    out_subdir: str = "raw",
) -> dict[Path, list[dict]]:
    """Stage one raw cycle folder into per-sample OME-TIFF stacks.

    ``sample_root`` is the experiment-level output dir (``mcmicro_output/<exp>``); sample subdirs
    ``rack-..-well-..-roi-..-exp-..`` are created beneath it. Returns, per sample dir, the list
    of channel rows (for :func:`write_markers_csv`) contributed by this cycle.
    """
    df = scan_cycle(cycle_dir, ref_marker=ref_marker)
    marker_rows: dict[Path, list[dict]] = {}

    indices: list[GroupIndex] = [tuple(k) for k, _ in df.groupby(GROUP_KEYS)]
    if hi_exposure_only:
        # keep only the max exposure level per (source, rack, well, roi)
        keep_lvl: dict[tuple, int] = {}
        for src, rack, well, roi, lvl in indices:
            key = (src, rack, well, roi)
            keep_lvl[key] = max(keep_lvl.get(key, lvl), lvl)
        indices = [ix for ix in indices if keep_lvl[(ix[0], ix[1], ix[2], ix[3])] == ix[4]]

    meta_cache: dict[str, dict] = {}

    def _meta(path: str) -> dict:
        if path not in meta_cache:
            meta_cache[path] = read_tile_meta(path)
        return meta_cache[path]

    for index in indices:
        mask = np.logical_and.reduce([df[k] == v for k, v in zip(GROUP_KEYS, index)])
        group = df[mask].copy()

        if ref_marker not in set(group["marker"]):
            group = append_reference(group, index, df, ref_marker)

        conformed = conform_markers(_unique_pairs(group), ref_marker)
        n_ch = len(conformed)
        tiles = sorted(group["tile"].unique().tolist())

        # per-channel exposure (constant across tiles for a given marker/filter)
        exposures: list[float] = []
        for m, f in conformed:
            sel = group[(group["marker"] == m) & (group["filter"] == f)]
            exposures.append(float(sel["exposure_time"].iloc[0]))

        base_meta = _meta(str(group["full_path"].iloc[0]))
        height, width = base_meta["size_y"], base_meta["size_x"]
        stack = np.zeros((len(tiles) * n_ch, int(height), int(width)), dtype=base_meta["type"])

        meta_by_tile: dict[int, list[dict]] = {}
        counter = 0
        for tile in tiles:
            tile_rows = group[group["tile"] == tile]
            per_ch_meta: list[dict] = []
            for m, f in conformed:
                sel = tile_rows[(tile_rows["marker"] == m) & (tile_rows["filter"] == f)]
                if sel.empty:
                    raise ValueError(
                        f"{cycle_dir.name}: tile {tile} group {index} missing channel "
                        f"({m},{f}); incomplete acquisition"
                    )
                path = str(sel["full_path"].iloc[0])
                stack[counter, :, :] = tf.imread(path)
                per_ch_meta.append(_meta(path))
                counter += 1
            meta_by_tile[tile] = per_ch_meta

        ome_xml = _build_ome_xml(tiles, conformed, meta_by_tile, exposures)

        tag = ""
        if illumination_correction:
            stack = apply_illumination_correction(stack, n_ch)
            tag = "corr_"

        sample_dir = sample_root / sample_dir_name(index)
        out_dir = sample_dir / out_subdir
        out_dir.mkdir(parents=True, exist_ok=True)
        cycle_no = int(group["cycle"].iloc[0])
        out_path = out_dir / f"{tag}{stack_file_name(cycle_no, index, conformed)}"
        tf.imwrite(out_path, stack, photometric="minisblack")
        tf.tiffcomment(out_path, ome_xml)
        log.info("  wrote [path]%s[/]", out_path.name)

        # accumulate markers.csv rows for this cycle's contribution to the sample
        src = index[0]
        rows = marker_rows.setdefault(sample_dir, [])
        for (m, f), exp in zip(conformed, exposures):
            fm, ff = _fname_token(m), _fname_token(f)
            if src == "B":
                rows.append(
                    {
                        "cycle_number": cycle_no,
                        "src": src,
                        "marker_name": f"bg_{cycle_no:03d}_{fm}-{ff}",
                        "Filter": ff,
                        "background": "",
                        "exposure": exp,
                        "remove": "TRUE",
                    }
                )
            else:
                rows.append(
                    {
                        "cycle_number": cycle_no,
                        "src": src,
                        "marker_name": fm,
                        "Filter": ff,
                        "background": "" if m == ref_marker else f"bg_{cycle_no:03d}_{fm}-{ff}",
                        "exposure": exp,
                        "remove": "",
                    }
                )
    return marker_rows


# --------------------------------------------------------------------------- #
#  markers.csv writer (mirror of mc_tools.write_markers_file)                 #
# --------------------------------------------------------------------------- #

MARKERS_COLUMNS = [
    "channel_number",
    "cycle_number",
    "marker_name",
    "Filter",
    "background",
    "exposure",
    "remove",
]


def write_markers_csv(
    sample_dir: Path,
    rows: list[dict],
    *,
    ref_marker: str = "DAPI",
    remove_reference_marker: bool = False,
) -> Path:
    """Write ``<sample_dir>/markers.csv`` from accumulated channel rows.

    Rows are ordered by ``(cycle_number, src)`` with background (``B``) before stain (``S``) —
    matching macsima2mc's sorted-filename order — then numbered 1..N.
    """
    ordered = sorted(rows, key=lambda r: (r["cycle_number"], 0 if r["src"] == "B" else 1))
    df = pd.DataFrame(ordered)
    df["channel_number"] = range(1, len(df) + 1)
    if remove_reference_marker and not df.empty:
        earliest = df["cycle_number"].min()
        cond = (df["marker_name"] == ref_marker) & (df["cycle_number"] > earliest)
        df.loc[cond, "remove"] = "TRUE"
    df = df[MARKERS_COLUMNS]
    out = sample_dir / "markers.csv"
    df.to_csv(out, index=False)
    log.info("wrote [path]%s[/] ([count]%d[/] channels)", out, len(df))
    return out


def stage_roi(
    roi_dir: Path,
    sample_root: Path,
    *,
    cycle_glob: str = "*Cycle*",
    ref_marker: str = "DAPI",
    illumination_correction: bool = True,
    hi_exposure_only: bool = False,
    out_subdir: str = "raw",
    remove_reference_marker: bool = False,
) -> list[Path]:
    """Stage every cycle folder of one ROI, then write one ``markers.csv`` per sample dir.

    Returns the list of sample dirs written.
    """
    cycles = sorted(p for p in roi_dir.glob(cycle_glob) if p.is_dir())
    if not cycles:
        raise FileNotFoundError(f"No cycle folders matching {cycle_glob!r} under {roi_dir}")

    accumulated: dict[Path, list[dict]] = {}
    for cyc in cycles:
        log.info("staging cycle [stage]%s[/]", cyc.name)
        contrib = stage_cycle(
            cyc,
            sample_root,
            ref_marker=ref_marker,
            illumination_correction=illumination_correction,
            hi_exposure_only=hi_exposure_only,
            out_subdir=out_subdir,
        )
        for sample_dir, rows in contrib.items():
            accumulated.setdefault(sample_dir, []).extend(rows)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for sample_dir, rows in accumulated.items():
            write_markers_csv(
                sample_dir,
                rows,
                ref_marker=ref_marker,
                remove_reference_marker=remove_reference_marker,
            )
    return sorted(accumulated)
