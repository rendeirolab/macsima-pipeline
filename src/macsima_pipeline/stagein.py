"""Stage-in: copy a task's inputs to fast node-local storage before processing.

IO-bound SLURM array tasks (GPU segmentation, raw staging) otherwise read large image
data straight off network Lustre. Copying the inputs once to a fast local filesystem —
a RAM disk (tmpfs, e.g. ``/dev/shm``) or node-local SSD/scratch — turns many random /
small-file network reads into a single sequential copy plus fast local reads.

Enabled purely by config: set ``stage_in.dir`` and staging is on; leave it ``None`` and
every :func:`staged` is a no-op. Nothing here is cluster-specific — the target path (and
any ``$VAR`` in it) is expanded at runtime.

On SLURM, files written to tmpfs count against the job's ``--mem`` cgroup limit, so the
planners bump ``--mem`` by the staged size when ``mem_charged`` is true (see
:func:`plan_mem`). This module removes the staged copy in a ``finally`` block; the sbatch
templates add a ``trap`` as a second-layer cleanup for hard crashes.
"""

from __future__ import annotations

import logging
import math
import os
import shutil
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Iterator

from .utils import ensure_dir

if TYPE_CHECKING:
    from .config import StageInCfg

log = logging.getLogger(__name__)


def staged_size(path: Path) -> int:
    """Total bytes of ``path`` — the file's size, or the sum of files under a directory."""
    if path.is_dir():
        return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
    return path.stat().st_size


def plan_mem(sizes: list[int], *, working_gb: int, safety: float, cap_gb: int) -> str:
    """SLURM ``--mem`` string covering the largest staged item plus working headroom.

    ``mem_GB = min(cap_gb, ceil(max(sizes) / 1e9 * safety) + working_gb)``. Sized for the
    largest item because ``--mem`` is uniform across a SLURM array.
    """
    max_bytes = max(sizes) if sizes else 0
    staged_gb = math.ceil(max_bytes / 1e9 * safety)
    return f"{min(cap_gb, staged_gb + working_gb)}G"


def _job_scope() -> str:
    """Unique-per-task subdir component so concurrent array tasks never collide."""
    job = os.environ.get("SLURM_JOB_ID")
    task = os.environ.get("SLURM_ARRAY_TASK_ID")
    if job:
        return f"{job}.{task}" if task else job
    return f"pid{os.getpid()}"


@contextmanager
def staged(src: Path, cfg: StageInCfg | None) -> Iterator[Path]:
    """Yield a fast-local copy of ``src`` (file or directory), or ``src`` unchanged.

    A no-op (yields ``src``) when staging is disabled (``cfg`` is ``None`` or ``cfg.dir``
    is ``None``), when the target dir is missing/unwritable, or when ``src`` would not fit
    the memory/space budget — the caller then reads directly from the original path. The
    staged copy is always removed on exit.
    """
    if cfg is None or cfg.dir is None:
        yield src
        return

    base = Path(os.path.expandvars(str(cfg.dir)))
    dest_dir = base / f"macsima.{_job_scope()}"
    dest = dest_dir / src.name

    try:
        src_bytes = staged_size(src)
        ensure_dir(dest_dir)
        free = shutil.disk_usage(dest_dir).free
    except OSError as e:
        log.warning("[warn]stage-in unavailable[/] (%s); reading [path]%s[/] in place", e, src)
        shutil.rmtree(dest_dir, ignore_errors=True)
        yield src
        return

    # Budget: never exceed free space; for mem-charged tmpfs also never exceed the --mem
    # headroom the planner sized for (cap_gb - working_gb), matching plan_mem's clamp.
    budget = free * 0.9
    if cfg.mem_charged:
        budget = min(budget, (cfg.cap_gb - cfg.working_gb) * 1e9)
    if src_bytes > budget:
        log.warning(
            "[warn]stage-in skipped[/]: %s (%.1f GB) exceeds budget %.1f GB for [path]%s[/]; reading in place",
            src.name, src_bytes / 1e9, budget / 1e9, base,
        )
        shutil.rmtree(dest_dir, ignore_errors=True)
        yield src
        return

    log.info("[ok]stage-in[/]: copying %s (%.1f GB) -> [path]%s[/]", src.name, src_bytes / 1e9, dest)
    try:
        if src.is_dir():
            shutil.copytree(src, dest)
        else:
            shutil.copyfile(src, dest)
    except OSError as e:
        log.warning("[warn]stage-in copy failed[/] (%s); reading [path]%s[/] in place", e, src)
        shutil.rmtree(dest_dir, ignore_errors=True)
        yield src
        return

    try:
        yield dest
    finally:
        shutil.rmtree(dest_dir, ignore_errors=True)
        log.info("stage-in: removed [path]%s[/]", dest_dir)
