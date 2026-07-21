"""Consolidate mcmicro outputs into the unified ``results/<exp>/images`` tree.

mcmicro publishes per-sample OME-TIFFs under
``mcmicro_output/<exp>/<sample>/{registration,background}/``. This step hardlinks
them (zero extra bytes — same lustre filesystem) into a flat, human-readable
per-ROI layout::

    results/<exp>/images/registration/<roi>.ome.tif   (no-bg variant)
    results/<exp>/images/backsub/<roi>.ome.tif        (bg-sub variant)
    results/<exp>/images/markers.csv  markers_bs.csv

and copies the markers CSVs once per experiment (the panel is constant across an
experiment's ROIs). It is idempotent — an existing link to the same inode is left
alone — so it is safe to call before every preprocess/viz run.
"""

from __future__ import annotations

import logging
import os
import shutil
from pathlib import Path

from .config import IMAGE_VARIANT_SUBDIR, Config
from .utils import ensure_dir, roi_name_from_mcmicro_stem

log = logging.getLogger(__name__)


def _discover_samples(cfg: Config) -> list[Path]:
    """Staged-sample dirs under mcmicro_output/<exp> (tolerant: [] if none yet)."""
    base = cfg.paths.work_dir / cfg.paths.mcmicro_out / cfg.experiment.name
    if not base.is_dir():
        return []
    return sorted(p for p in base.glob(cfg.mcmicro.sample_pattern) if p.is_dir())


def _hardlink(src: Path, dest: Path) -> bool:
    """Hardlink ``src`` -> ``dest`` idempotently. Returns True if a link was made."""
    ensure_dir(dest.parent)
    if dest.exists():
        try:
            if dest.samefile(src):
                return False  # already linked to the same inode
        except OSError:
            pass
        dest.unlink()
    try:
        os.link(src, dest)
    except OSError as e:
        # Different filesystem or a backend without hardlink support: copy instead.
        log.warning("hardlink failed (%s); copying %s -> %s", e, src, dest)
        shutil.copy2(src, dest)
    return True


def consolidate_images(cfg: Config) -> int:
    """Hardlink mcmicro per-sample OME-TIFFs + markers into results/<exp>/images/.

    Returns the number of image links created (0 if nothing new, or if mcmicro has
    not produced any samples yet).
    """
    samples = _discover_samples(cfg)
    if not samples:
        return 0

    images_dir = cfg.images_dir()
    linked = 0
    markers_done = {True: False, False: False}
    for sample in samples:
        for bg in (False, True):
            pattern = cfg.mcmicro.background_pattern if bg else cfg.mcmicro.registration_pattern
            matches = sorted(sample.glob(pattern))
            if not matches:
                continue
            roi_name = roi_name_from_mcmicro_stem(matches[0].stem)
            dest = images_dir / IMAGE_VARIANT_SUBDIR[bg] / f"{roi_name}.ome.tif"
            if _hardlink(matches[0], dest):
                linked += 1
            if not markers_done[bg]:
                # Preserve the source-relative subpath ("markers.csv" /
                # "background/markers_bs.csv") so downstream readers resolve markers
                # under images_dir() with the same cfg.mcmicro.markers_* keys.
                rel = cfg.mcmicro.markers_bs_csv if bg else cfg.mcmicro.markers_csv
                src_markers = sample / rel
                if src_markers.is_file():
                    dst_markers = images_dir / rel
                    ensure_dir(dst_markers.parent)
                    shutil.copy2(src_markers, dst_markers)
                    markers_done[bg] = True

    if linked:
        log.info("[ok]consolidated[/] [count]%d[/] image(s) -> [path]%s[/]", linked, images_dir)
    else:
        log.info("images already consolidated -> [path]%s[/]", images_dir)
    return linked
