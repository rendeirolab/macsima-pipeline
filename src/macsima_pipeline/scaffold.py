"""Scaffold utilities.

Two generators that turn a config into the CSV inputs the pipeline consumes:

- `gen_roi_metadata`  — PRE-staging. Reuses `staging.discover_rois` to list the
  exact ROIs the pipeline will process and writes a template `roi_metadata.csv`
  (an `ROI` column + empty user columns) for the user to fill in.
- `gen_markers`       — POST-staging. Reads the `macsima2mc`-generated `markers.csv`
  from the first staged sample and writes a canonical per-experiment panel with a
  normalized `remove` column. A review/curation artifact: it does NOT change the
  preprocess read path (which still reads each sample's own markers.csv).

Both operate on a single-experiment `Config`; the CLI loops them over a batch via
`config.expand_config`.
"""

from __future__ import annotations

import csv
import logging
from pathlib import Path

import pandas as pd

from . import preprocess as preprocess_stage
from . import staging as staging_stage
from .config import Config
from .utils import ensure_dir, roi_index_from_name

log = logging.getLogger(__name__)


def _roi_sort_key(name: str) -> tuple[int, object]:
    """Sort ROI1, ROI2, ..., ROI10 numerically; push unparseable names to the end."""
    try:
        return (0, roi_index_from_name(name))
    except ValueError:
        return (1, name)


def _roi_output_path(cfg: Config, config_path: Path, output: Path | None) -> Path:
    if output is not None:
        return output
    if cfg.experiment.roi_metadata_csv is not None:
        # Exactly where preprocess._join_roi_metadata reads it.
        return cfg.paths.work_dir / cfg.experiment.roi_metadata_csv
    return config_path.parent / f"roi_metadata_{cfg.experiment.name}.csv"


def gen_roi_metadata(
    cfg: Config,
    *,
    config_path: Path,
    output: Path | None = None,
    extra_columns: list[str] | None = None,
    force: bool = False,
) -> Path | None:
    """Write a template roi_metadata.csv for one experiment.

    Returns the written path, or None if skipped (raw_root unreachable, no ROIs, or
    the destination already exists without --force).
    """
    name = cfg.experiment.name
    try:
        rois = staging_stage.discover_rois(cfg)
    except FileNotFoundError as e:
        log.warning(
            "[bad]raw_root unreachable[/] for [stage]%s[/]: %s — run where RawData is mounted",
            name,
            e,
        )
        return None
    if not rois:
        log.warning(
            "[bad]no ROIs[/] for [stage]%s[/] under [path]%s[/] (glob %s)",
            name,
            cfg.experiment.raw_root,
            cfg.experiment.roi_glob,
        )
        return None

    labels = sorted((p.name for p in rois), key=_roi_sort_key)
    extra = list(extra_columns or [])
    header = ["ROI", *extra]

    dest = _roi_output_path(cfg, config_path, output)
    if dest.exists() and not force:
        log.warning("[warn]exists[/] [path]%s[/] — pass --force to overwrite", dest)
        return None

    ensure_dir(dest.parent)
    # lineterminator="\n": match the repo convention (avoid CRLF; see staging.write_jobs_csv).
    with dest.open("w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(header)
        for label in labels:
            w.writerow([label, *([""] * len(extra))])
    log.info(
        "wrote ROI metadata template: [path]%s[/] ([count]%d[/] ROIs)",
        dest,
        len(labels),
    )
    if cfg.experiment.roi_metadata_csv is None and output is None:
        log.info("  set experiment.roi_metadata_csv to [path]%s[/] to have preprocess join it", dest)
    return dest


def _to_bool(v: object) -> bool:
    """Normalize a markers.csv `remove` cell to a clean bool.

    Handles bool, blank/NaN -> False, and string forms (TRUE/False/1/yes/t).
    """
    if pd.isna(v):
        return False
    if isinstance(v, bool):
        return v
    return str(v).strip().lower() in ("true", "1", "yes", "t")


def _markers_output_path(cfg: Config, config_path: Path, output: Path | None) -> Path:
    if output is not None:
        return output
    return config_path.parent / f"markers_{cfg.experiment.name}.csv"


def _find_markers_csv(cfg: Config, bg: bool) -> Path | None:
    """Locate the staged markers.csv (first mcmicro sample). None if not staged yet."""
    try:
        images = preprocess_stage._find_images(cfg, bg)
    except FileNotFoundError:
        return None
    src = images[0].parent.parent / (cfg.mcmicro.markers_bs_csv if bg else cfg.mcmicro.markers_csv)
    return src if src.is_file() else None


def _panel_markers(cfg: Config, bg: bool = False) -> list[str] | None:
    """Usable marker names for the panel (as they appear in adata.var_names).

    Mirrors preprocess._load_channel_info: drop `remove == True` rows, dedup on
    marker_name, preserve order. Returns None if the panel isn't staged yet.
    """
    src = _find_markers_csv(cfg, bg)
    if src is None:
        return None
    df = pd.read_csv(src)
    if "remove" in df.columns:
        df = df[~df["remove"].map(_to_bool)]
    names = list(dict.fromkeys(df["marker_name"].astype(str)))
    return names


def gen_markers(
    cfg: Config,
    *,
    config_path: Path,
    output: Path | None = None,
    bg: bool = False,
    force: bool = False,
) -> Path | None:
    """Consolidate one experiment's staged markers.csv into a canonical panel.

    Keeps all rows and column/row order; only normalizes the `remove` column
    (preserving bg_* rows is required — channel_index is the physical TIFF position).
    Returns the written path, or None if skipped.
    """
    name = cfg.experiment.name
    src = _find_markers_csv(cfg, bg)
    if src is None:
        log.warning(
            "[warn]no staged markers.csv[/] for [stage]%s[/] — it is produced during staging "
            "(stage 1); run gen-markers after staging completes",
            name,
        )
        return None

    df = pd.read_csv(src)
    if "remove" in df.columns:
        df["remove"] = df["remove"].map(_to_bool)

    dest = _markers_output_path(cfg, config_path, output)
    if dest.exists() and not force:
        log.warning("[warn]exists[/] [path]%s[/] — pass --force to overwrite", dest)
        return None

    ensure_dir(dest.parent)
    df.to_csv(dest, index=False)
    log.info(
        "wrote markers panel: [path]%s[/] ([count]%d[/] rows) from [path]%s[/]",
        dest,
        len(df),
        src,
    )
    return dest


# Common lineage markers -> (positive, negative, parent). Only cell types with >=1
# positive marker PRESENT in the panel are emitted, as EXAMPLES to edit. Purely a
# scaffold: the user must curate the actual biology.
_COMMON_SIGNATURE: list[tuple[str, list[str], list[str], str]] = [
    ("T cell",        ["CD3", "CD3e", "CD45"],                        ["CD19", "CD68"], "Immune"),
    ("CD4 T cell",    ["CD4"],                                        ["CD8", "CD8a"],  "T cell"),
    ("CD8 T cell",    ["CD8", "CD8a"],                                ["CD4"],          "T cell"),
    ("B cell",        ["CD19", "CD20"],                               ["CD3"],          "Immune"),
    ("NK cell",       ["CD56"],                                       ["CD3"],          "Immune"),
    ("Macrophage",    ["CD68", "CD163"],                              ["CD3"],          "Myeloid"),
    ("Dendritic",     ["CD11c", "HLADR", "HLA-DR"],                   ["CD3"],          "Myeloid"),
    ("Endothelial",   ["CD31", "CD34"],                               ["CD45"],         "Stroma"),
    ("Fibroblast",    ["aSMA", "Vimentin", "PDGFRb"],                 ["CD45"],         "Stroma"),
    ("Epithelial",    ["PanCK", "EpCAM", "CK", "Cytokeratin", "E-cadherin"], ["CD45"],  "Epithelium"),
    ("Proliferating", ["Ki67"],                                       [],               ""),
]


def _signature_output_path(cfg: Config, config_path: Path, output: Path | None) -> Path:
    if output is not None:
        return output
    return config_path.parent / f"signature_{cfg.experiment.name}.yaml"


def _render_signature_yaml(name: str, panel: list[str], examples: list[tuple]) -> str:
    lines = [
        f"# Signature-matrix template for '{name}' (generated by gen-signature).",
        "# EDIT THIS. The cell types below are GENERIC guesses matched from marker names —",
        "# curate positive/negative markers for YOUR panel + tissue. Names must match the panel.",
        "# scyan builds a table from these; Leiden scores clusters. `parent` builds the coarse label.",
        "#",
        "# Panel markers available:",
    ]
    for i in range(0, len(panel), 8):
        lines.append("#   " + ", ".join(panel[i : i + 8]))
    lines += ["", "version: 1", "", "cell_types:"]
    if not examples:
        lines.append("  # No common lineage markers auto-detected. Define your own, e.g.:")
        lines.append(f"  Cell type A: {{positive: [{panel[0]}], negative: []}}")
    for nm, pos, neg, parent in examples:
        parent_s = f", parent: {parent}" if parent else ""
        lines.append(f"  {nm}: {{positive: [{', '.join(pos)}], negative: [{', '.join(neg)}]{parent_s}}}")
    return "\n".join(lines) + "\n"


def gen_signature(
    cfg: Config,
    *,
    config_path: Path,
    output: Path | None = None,
    bg: bool = False,
    force: bool = False,
) -> Path | None:
    """Write a signature-matrix YAML template from the staged marker panel.

    Lists every usable panel marker (as a reference comment) and pre-fills example
    cell types built from markers present in the panel, for the user to curate.
    Post-staging (needs markers.csv). Returns the written path, or None if skipped.
    """
    name = cfg.experiment.name
    panel = _panel_markers(cfg, bg)
    if panel is None:
        log.warning(
            "[warn]no staged markers.csv[/] for [stage]%s[/] — run gen-signature after staging completes",
            name,
        )
        return None
    if not panel:
        log.warning("[bad]empty marker panel[/] for [stage]%s[/]", name)
        return None

    present = set(panel)
    examples = []
    for nm, pos, neg, parent in _COMMON_SIGNATURE:
        pos_p = [m for m in pos if m in present]
        if not pos_p:
            continue
        examples.append((nm, pos_p, [m for m in neg if m in present], parent))

    dest = _signature_output_path(cfg, config_path, output)
    if dest.exists() and not force:
        log.warning("[warn]exists[/] [path]%s[/] — pass --force to overwrite", dest)
        return None

    ensure_dir(dest.parent)
    dest.write_text(_render_signature_yaml(name, panel, examples))
    log.info(
        "wrote signature template: [path]%s[/] ([count]%d[/] panel markers, [count]%d[/] example types)",
        dest,
        len(panel),
        len(examples),
    )
    log.info("  edit it, then set [stage]phenotype.signature_matrix[/] to this path")
    return dest
