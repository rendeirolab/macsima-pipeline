"""Pre-staging marker panel sanity check.

Runs *before* the (expensive) staging compute — it only parses raw filenames, so it is cheap
enough to run at submit time on a login node. It writes ``artifacts/<exp>/marker_panel.csv``
(one row per cycle x marker x filter: which cycles, filters, exposures and how many ROIs carry
each marker) and validates the acquired panel, so you can catch acquisition problems before
committing compute. The `panel` CLI command additionally scaffolds a shared cell-type
signature template (``signature.yaml``) next to the config for the user to curate.
"""

from __future__ import annotations

import logging

import pandas as pd

from . import staging, staging_core
from .config import Config
from .utils import ensure_dir

log = logging.getLogger(__name__)


def _fname(s: str) -> str:
    """Match the marker/filter spelling used in the staged markers.csv ('_' -> '-')."""
    return s.replace("_", "-")


def scan_experiment(cfg: Config) -> pd.DataFrame:
    """Parse every raw tile filename across all ROIs into one acquisition table.

    Filename-only (no pixel/OME reads), so this is fast. Adds a ``roi_name`` column.
    """
    rois = staging.discover_rois(cfg)
    if not rois:
        raise RuntimeError(f"No ROIs under {cfg.experiment.raw_root} matching {cfg.experiment.roi_glob}")
    frames: list[pd.DataFrame] = []
    for roi in rois:
        cycles = sorted(p for p in roi.glob(cfg.staging.cycle_glob) if p.is_dir())
        for cyc in cycles:
            df = staging_core.scan_cycle(cyc, ref_marker=cfg.staging.reference_marker)
            df["roi_name"] = roi.name
            frames.append(df)
    if not frames:
        raise RuntimeError(f"No cycle folders matching {cfg.staging.cycle_glob!r} under any ROI")
    scan = pd.concat(frames, ignore_index=True)
    log.info(
        "panel scan: [count]%d[/] tiles, [count]%d[/] ROIs, [count]%d[/] cycles",
        len(scan), scan["roi_name"].nunique(), scan["cycle"].nunique(),
    )
    return scan


def sanity_check(cfg: Config, scan: pd.DataFrame) -> list[str]:
    """Validate the panel before staging. Raises RuntimeError on hard errors; returns warnings."""
    ref = cfg.staging.reference_marker
    warns: list[str] = []
    errors: list[str] = []

    stain = scan[scan["source"] == "S"]

    # 1) reference marker present in every cycle
    for c in sorted(scan["cycle"].unique()):
        if ref not in set(scan.loc[scan["cycle"] == c, "marker"]):
            errors.append(f"cycle {c}: reference marker {ref!r} not found (needed for registration)")

    # 2) consistent (marker, filter) set across ROIs
    per_roi = {roi: frozenset(zip(g["marker"], g["filter"])) for roi, g in stain.groupby("roi_name")}
    if per_roi:
        ref_set = max(per_roi.values(), key=len)
        for roi, s in per_roi.items():
            missing = ref_set - s
            if missing:
                warns.append(f"ROI {roi}: missing markers vs fullest ROI: {sorted(m for m, _ in missing)}")

    # 3) each stain marker has a matching background (S and B)
    s_pairs = set(zip(stain["cycle"], stain["marker"], stain["filter"]))
    b = scan[scan["source"] == "B"]
    b_pairs = set(zip(b["cycle"], b["marker"], b["filter"]))
    no_bg = s_pairs - b_pairs
    if no_bg:
        warns.append(f"{len(no_bg)} stain acquisitions have no matching background (src-B) image")

    for w in warns:
        log.warning("[warn]panel sanity[/]: %s", w)
    if errors:
        for e in errors:
            log.error("[bad]panel sanity[/]: %s", e)
        raise RuntimeError("panel sanity check failed:\n  - " + "\n  - ".join(errors))
    log.info("[ok]panel sanity check passed[/] (%d warning(s))", len(warns))
    return warns


def write_marker_panel(cfg: Config, scan: pd.DataFrame) -> pd.DataFrame:
    """Write the marker-panel summary CSV; return the summary frame."""
    stain = scan[scan["source"] == "S"]
    recs: list[dict] = []
    for (cyc, marker, filt), g in stain.groupby(["cycle", "marker", "filter"], sort=True):
        recs.append(
            {
                "cycle_number": int(cyc),
                "marker_name": _fname(marker),
                "Filter": _fname(filt),
                "exposure_levels": g["exposure_level"].nunique(),
                "exposures": ";".join(str(x) for x in sorted(g["exposure_time"].unique())),
                "n_rois": g["roi_name"].nunique(),
                "is_reference": marker == cfg.staging.reference_marker,
            }
        )
    panel = pd.DataFrame(recs).sort_values(["cycle_number", "marker_name"]).reset_index(drop=True)
    out = cfg.marker_panel_path()
    ensure_dir(out.parent)
    panel.to_csv(out, index=False)
    log.info("wrote marker panel [path]%s[/] ([count]%d[/] rows)", out, len(panel))
    return panel


def generate(cfg: Config) -> pd.DataFrame:
    """Full pre-staging step: scan -> sanity check -> write marker panel.

    Returns the marker-panel summary frame (the CLI unions its ``marker_name`` column
    across experiments to scaffold a shared phenotyping signature template).
    """
    scan = scan_experiment(cfg)
    sanity_check(cfg, scan)
    return write_marker_panel(cfg, scan)
