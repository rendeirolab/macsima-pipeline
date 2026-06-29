"""Stage 1: stage raw MACSima cycles into mcmicro-ingestable per-ROI folders.

Port of `scripts/create_staging_jobs.sh` + `scripts/staging.sh` from the original
`metpredict-macsima` repo.
"""

from __future__ import annotations

import csv
import logging
import shlex
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
        # lineterminator="\n" — default csv.writer emits \r\n per RFC 4180 which
        # leaves a literal \r in the last field when bash `read` parses the row.
        # That \r broke apptainer mounts ("mount source …/mcmicro_output\r doesn't exist").
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["job_id", "roi_path", "roi_name", "sample_id", "output_dir"])
        for i, roi in enumerate(rois, start=1):
            w.writerow([i, str(roi), roi.name, cfg.experiment.name, str(output_dir)])
    log.info("wrote [path]%s[/] ([count]%d[/] rows)", path, len(rois))
    return path


def _body_cmd(cfg: Config) -> str:
    """Bash body executed inside each SLURM array task."""
    sif = cfg.containers.macsima2mc_sif
    jobs_csv = cfg.jobs_csv("staging")
    # Read CSV row matching $SLURM_ARRAY_TASK_ID then run apptainer per cycle.
    return rf"""jobs_file={shlex.quote(str(jobs_csv))}
job_line=$(awk -F',' -v task_id="$SLURM_ARRAY_TASK_ID" 'NR > 1 && $1 == task_id {{print; exit}}' "$jobs_file")
if [ -z "$job_line" ]; then
    echo "Error: no row for SLURM_ARRAY_TASK_ID=${{SLURM_ARRAY_TASK_ID}} in $jobs_file" >&2
    exit 1
fi
job_line=${{job_line%$'\r'}}   # defensive: strip trailing CR if CSV ever has CRLF
IFS=',' read -r job_id roi roi_name sample_id output_dir <<< "$job_line"

mkdir -p "$output_dir"          # apptainer --bind requires source dir to exist
output_dir=$(readlink -f "$output_dir")   # absolute path so --bind doesn't depend on cwd

shopt -s nullglob
echo "Staging ROI: $roi_name (task $SLURM_ARRAY_TASK_ID)"
echo "  output_dir: $output_dir"

for cycle in "$roi"/*Cycle*; do
    cycle_folder=$(basename "${{cycle}}")
    echo "  -> $cycle_folder"
    apptainer exec \
        --bind "${{roi}}:/mnt","${{output_dir}}:/media" --no-home \
        {shlex.quote(str(sif))} \
        macsima2mc -i "/mnt/${{cycle_folder}}" -o "/media/${{sample_id}}" -ic
done

echo "Done: $roi_name"
"""


def plan(cfg: Config) -> tuple[Path, Path, int]:
    """Discover ROIs, write jobs CSV, render sbatch. Return (csv, sbatch, n_jobs)."""
    rois = discover_rois(cfg)
    if not rois:
        raise RuntimeError(f"No ROIs found under {cfg.experiment.raw_root} matching {cfg.experiment.roi_glob}")
    csv_path = write_jobs_csv(cfg, rois)
    ensure_dir(cfg.paths.work_dir / cfg.paths.staging_out)
    ensure_dir(cfg.paths.work_dir / cfg.paths.logs_dir)
    content = render_sbatch(cfg, "staging", array_size=len(rois), body_cmd=_body_cmd(cfg))
    sbatch = write_sbatch(cfg, "staging", content)
    return csv_path, sbatch, len(rois)


def run(cfg: Config, *, do_submit: bool, dependency: str | None = None) -> int | None:
    csv_path, sbatch, n = plan(cfg)
    log.info(
        "staging plan: [count]%d[/] ROIs, csv=[path]%s[/] sbatch=[path]%s[/]",
        n, csv_path, sbatch,
    )
    if not do_submit:
        log.warning("[warn](dry-run)[/] sbatch --array=1-[count]%d[/] [path]%s[/]", n, sbatch)
        return None
    return submit(sbatch, array_size=n, dependency=dependency)
