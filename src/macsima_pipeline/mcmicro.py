"""Stage 2: run mcmicro (Nextflow + Singularity) per staged sample.

Port of `scripts/create_mcmicro_jobs.sh` + `scripts/run_mcmicro.sh`.
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


def discover_samples(cfg: Config) -> list[Path]:
    """List staged-sample dirs under {mcmicro_out}/{experiment.name} matching mcmicro.sample_pattern."""
    base = cfg.paths.work_dir / cfg.paths.mcmicro_out / cfg.experiment.name
    if not base.is_dir():
        raise FileNotFoundError(
            f"Staged output dir not found: {base}. Run staging first or check paths.mcmicro_out."
        )
    return sorted(p for p in base.glob(cfg.mcmicro.sample_pattern) if p.is_dir())


def write_jobs_csv(cfg: Config, samples: list[Path]) -> Path:
    path = cfg.jobs_csv("mcmicro")
    ensure_dir(path.parent)
    params = cfg.mcmicro.params_yaml
    with path.open("w", newline="") as f:
        # lineterminator="\n" — avoid CRLF leaving \r in last field after `bash read`.
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["job_id", "sample_path", "sample_name", "params_file"])
        for i, sample in enumerate(samples, start=1):
            w.writerow([i, str(sample), sample.name, str(params)])
    log.info("wrote [path]%s[/] ([count]%d[/] rows)", path, len(samples))
    return path


def _body_cmd(cfg: Config) -> str:
    nxf_config = cfg.mcmicro.nextflow_config
    jobs_csv = cfg.jobs_csv("mcmicro")
    work_dir = cfg.paths.work_dir.resolve()
    nxf_config_flag = f"-c {shlex.quote(str(nxf_config))} " if nxf_config else ""
    nxf_work = shlex.quote(str(work_dir / "work"))
    cd = f"cd {shlex.quote(str(work_dir))}"
    return rf"""{cd}
jobs_file={shlex.quote(str(jobs_csv))}
job_line=$(awk -F',' -v task_id="$SLURM_ARRAY_TASK_ID" 'NR > 1 && $1 == task_id {{print; exit}}' "$jobs_file")
if [ -z "$job_line" ]; then
    echo "Error: no row for SLURM_ARRAY_TASK_ID=${{SLURM_ARRAY_TASK_ID}} in $jobs_file" >&2
    exit 1
fi
job_line=${{job_line%$'\r'}}   # defensive: strip trailing CR if CSV ever has CRLF
IFS=',' read -r job_id sample_path sample_name params_file <<< "$job_line"

echo "=========================================="
echo "mcmicro on $sample_name  (task $SLURM_ARRAY_TASK_ID)"
echo "sample_path: $sample_path"
echo "params:      $params_file"
echo "=========================================="

# publish_dir_mode=link publishes mcmicro's OME-TIFFs as HARDLINKS from the task
# work/ dir (one inode, zero duplication); -work-dir pins work/ on the same
# filesystem as the outputs — mode 'link' has no copy fallback, so they must share
# a filesystem for the publish to succeed.
nextflow {nxf_config_flag}run labsyspharm/mcmicro \
    -profile singularity \
    -work-dir {nxf_work} \
    --in "$sample_path" \
    --params "$params_file" \
    --publish_dir_mode link

rc=$?
echo "Finished $sample_name (rc=$rc)"
exit $rc
"""


def plan(cfg: Config) -> tuple[Path, Path, int]:
    samples = discover_samples(cfg)
    if not samples:
        raise RuntimeError(
            f"No mcmicro samples found under {cfg.paths.mcmicro_out}/{cfg.experiment.name} "
            f"matching {cfg.mcmicro.sample_pattern}"
        )
    csv_path = write_jobs_csv(cfg, samples)
    ensure_dir(cfg.paths.work_dir / cfg.paths.logs_dir)
    content = render_sbatch(cfg, "mcmicro", array_size=len(samples), body_cmd=_body_cmd(cfg))
    sbatch = write_sbatch(cfg, "mcmicro", content)
    return csv_path, sbatch, len(samples)


def run(
    cfg: Config,
    *,
    do_submit: bool,
    dependency: str | None = None,
    wait: bool = False,
) -> int | None:
    csv_path, sbatch, n = plan(cfg)
    log.info(
        "mcmicro plan: [count]%d[/] samples, csv=[path]%s[/] sbatch=[path]%s[/]",
        n, csv_path, sbatch,
    )
    if not do_submit:
        log.warning("[warn](dry-run)[/] sbatch --array=1-[count]%d[/] [path]%s[/]", n, sbatch)
        return None
    return submit(sbatch, array_size=n, dependency=dependency, wait=wait)
