"""Stage 1: stage raw MACSima cycles into mcmicro-ingestable per-ROI folders.

Native Python reimplementation of the former macsima2mc Docker/apptainer step (see
:mod:`macsima_pipeline.staging_core`). One SLURM array task per ROI re-invokes the CLI
(``stage --stage-roi``) which stages every cycle of that ROI in-process — the same
callback pattern used by the preprocess and viz stages.

The marker-panel + cell-type signature template are generated at plan time (before the array
is submitted) by :mod:`macsima_pipeline.panel`.
"""

from __future__ import annotations

import csv
import logging
import shlex
import sys
from pathlib import Path

from .config import Config
from .slurm import render_sbatch, submit, write_sbatch
from .utils import ensure_dir

log = logging.getLogger(__name__)


def discover_rois(cfg: Config) -> list[Path]:
    """List ROI dirs under raw_root matching roi_glob, applying include/exclude filters."""
    root = cfg.experiment.raw_root
    if not root.is_dir():
        raise FileNotFoundError(f"raw_root not found: {root}")

    matches = sorted(p for p in root.glob(cfg.experiment.roi_glob) if p.is_dir())
    if cfg.experiment.roi_include is not None:
        keep = set(cfg.experiment.roi_include)
        matches = [p for p in matches if p.name in keep]
    excl = set(cfg.experiment.roi_exclude or [])
    matches = [p for p in matches if p.name not in excl]
    return matches


def write_jobs_csv(cfg: Config, rois: list[Path]) -> Path:
    path = cfg.jobs_csv("staging")
    ensure_dir(path.parent)
    output_dir = cfg.paths.work_dir / cfg.paths.staging_out
    with path.open("w", newline="") as f:
        # lineterminator="\n" — default csv.writer emits \r\n per RFC 4180; a trailing \r once
        # broke path handling in the bash body. Harmless to keep now that we read it in Python.
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["job_id", "roi_path", "roi_name", "sample_id", "output_dir"])
        for i, roi in enumerate(rois, start=1):
            w.writerow([i, str(roi), roi.name, cfg.experiment.name, str(output_dir)])
    log.info("wrote [path]%s[/] ([count]%d[/] rows)", path, len(rois))
    return path


def _read_job_row(jobs_csv: Path, task_id: int) -> dict[str, str]:
    with jobs_csv.open(newline="") as f:
        for row in csv.DictReader(f):
            if int(row["job_id"]) == task_id:
                return row
    raise KeyError(f"no row for job_id={task_id} in {jobs_csv}")


def _body_cmd(cfg: Config, config_path: Path) -> str:
    """Bash body for each SLURM array task: re-invoke the CLI to stage one ROI natively."""
    python = Path(sys.executable)
    work_dir = cfg.paths.work_dir.resolve()
    return (
        f"cd {shlex.quote(str(work_dir))}\n"
        f"uv run --frozen --no-sync "
        f"{shlex.quote(str(python))} -m macsima_pipeline.cli stage "
        f"--config {shlex.quote(str(config_path))} --stage-roi --task-id $SLURM_ARRAY_TASK_ID"
    )


def stage_roi_inproc(cfg: Config, task_id: int) -> list[Path]:
    """Stage one ROI (all its cycle folders) in the current process. Called by the SLURM task."""
    from . import staging_core

    row = _read_job_row(cfg.jobs_csv("staging"), task_id)
    roi_dir = Path(row["roi_path"])
    sample_root = cfg.staging_root()
    st = cfg.staging
    log.info("staging ROI [stage]%s[/] (task %d) -> [path]%s[/]", roi_dir.name, task_id, sample_root)
    samples = staging_core.stage_roi(
        roi_dir,
        sample_root,
        cycle_glob=st.cycle_glob,
        ref_marker=st.reference_marker,
        illumination_correction=st.illumination_correction,
        hi_exposure_only=st.hi_exposure_only,
        out_subdir=st.output_subdir,
        remove_reference_marker=st.remove_reference_marker,
    )
    log.info("[ok]staged[/] ROI [stage]%s[/]: [count]%d[/] sample dir(s)", roi_dir.name, len(samples))
    return samples


def plan(cfg: Config, config_path: Path) -> tuple[Path, Path, int]:
    """Generate the marker panel, discover ROIs, write jobs CSV, render sbatch."""
    from . import panel

    # Marker panel sanity check BEFORE staging (the signature template is scaffolded by the panel command).
    panel.generate(cfg)

    rois = discover_rois(cfg)
    if not rois:
        raise RuntimeError(f"No ROIs found under {cfg.experiment.raw_root} matching {cfg.experiment.roi_glob}")
    csv_path = write_jobs_csv(cfg, rois)
    ensure_dir(cfg.staging_root())
    ensure_dir(cfg.paths.work_dir / cfg.paths.logs_dir)
    content = render_sbatch(cfg, "staging", array_size=len(rois), body_cmd=_body_cmd(cfg, config_path))
    sbatch = write_sbatch(cfg, "staging", content)
    return csv_path, sbatch, len(rois)


def run(
    cfg: Config,
    config_path: Path,
    *,
    do_submit: bool,
    dependency: str | None = None,
) -> int | None:
    csv_path, sbatch, n = plan(cfg, config_path)
    log.info(
        "staging plan: [count]%d[/] ROIs, csv=[path]%s[/] sbatch=[path]%s[/]",
        n, csv_path, sbatch,
    )
    if not do_submit:
        log.warning("[warn](dry-run)[/] sbatch --array=1-[count]%d[/] [path]%s[/]", n, sbatch)
        return None
    return submit(sbatch, array_size=n, dependency=dependency)
