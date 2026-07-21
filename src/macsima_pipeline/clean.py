"""Reclaim pipeline scratch space — explicit, opt-in (the "balanced" policy).

Nothing in the pipeline auto-deletes scratch. This is the deliberate tool for it.
Every function is dry-run by default (reports what it *would* remove); pass
``do_delete=True`` to actually delete. Each target is verified before removal so
the command is safe and idempotent.

Targets:
  * work/ + .nextflow*  — Nextflow scratch (removing it disables ``-resume``).
  * staged raw/ tiles   — only for samples already consolidated into results/.
  * orphaned zarr        — pre-refactor artifacts/<exp>/*.zarr + preprocess_parts*.
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

from .config import Config
from .finalize import _discover_samples
from .utils import roi_name_from_mcmicro_stem

log = logging.getLogger(__name__)


def _remove(path: Path, *, do_delete: bool) -> None:
    if not path.exists() and not path.is_symlink():
        return
    log.info("  %s [path]%s[/]", "removing" if do_delete else "would remove", path)
    if not do_delete:
        return
    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path, ignore_errors=True)
    else:
        path.unlink(missing_ok=True)


def clean_work(cfg: Config, *, do_delete: bool) -> None:
    """Nextflow work/ + .nextflow* at the work_dir root (disables -resume)."""
    wd = cfg.paths.work_dir
    log.info("[stage]work/ + .nextflow*[/] under [path]%s[/]", wd)
    _remove(wd / "work", do_delete=do_delete)
    _remove(wd / ".nextflow", do_delete=do_delete)
    for nf in sorted(wd.glob(".nextflow.log*")):
        _remove(nf, do_delete=do_delete)


def clean_raw(cfg: Config, *, do_delete: bool) -> None:
    """Staged raw/ input tiles for samples whose registration image is consolidated."""
    log.info("[stage]staged raw/ tiles[/] for [stage]%s[/]", cfg.experiment.name)
    for sample in _discover_samples(cfg):
        roi = roi_name_from_mcmicro_stem(sample.name)
        deliverable = cfg.variant_images_dir(False) / f"{roi}.ome.tif"
        raw_dir = sample / cfg.staging.output_subdir
        if not raw_dir.is_dir():
            continue
        if deliverable.is_file() and deliverable.stat().st_size > 0:
            _remove(raw_dir, do_delete=do_delete)
        else:
            log.warning("  keeping [path]%s[/] — deliverable not confirmed (%s)", raw_dir, deliverable)


def clean_orphaned_zarr(cfg: Config, *, do_delete: bool) -> None:
    """Pre-refactor artifacts/<exp>/*.zarr + preprocess_parts* stores (dropped layout)."""
    artifacts = cfg.paths.work_dir / "artifacts" / cfg.experiment.name
    log.info("[stage]orphaned zarr[/] under [path]%s[/]", artifacts)
    if not artifacts.is_dir():
        return
    for z in sorted(artifacts.glob("*.zarr")):
        _remove(z, do_delete=do_delete)
    for parts in sorted(artifacts.glob("preprocess_parts*")):
        _remove(parts, do_delete=do_delete)
