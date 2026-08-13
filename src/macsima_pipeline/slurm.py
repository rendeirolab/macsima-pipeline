"""Render sbatch scripts from Jinja2 templates + submit via `sbatch`."""

from __future__ import annotations

import logging
import re
import shutil
import subprocess
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, StrictUndefined

from .config import Config, SlurmStage

log = logging.getLogger(__name__)

TEMPLATES_DIR = Path(__file__).resolve().parent.parent.parent / "templates"


def _env() -> Environment:
    return Environment(
        loader=FileSystemLoader(str(TEMPLATES_DIR)),
        undefined=StrictUndefined,
        trim_blocks=True,
        lstrip_blocks=True,
    )


def render_sbatch(
    cfg: Config,
    stage: str,
    *,
    array_size: int | None,
    body_cmd: str,
    extra_ctx: dict | None = None,
    template_stage: str | None = None,
    slurm_stage: str | None = None,
    output_stage: str | None = None,
    mem_override: str | None = None,
    stage_in_dir: str | None = None,
) -> str:
    """Render `templates/{stage}.sbatch.j2` with stage SLURM block + body command.

    `mem_override` replaces the stage's `--mem` (used when staging inputs to a RAM disk
    inflates the memory footprint). `stage_in_dir` is the raw, un-expanded staging path
    passed to the template's cleanup `trap`; None (the default) leaves both untouched.
    """
    template_stage = template_stage or stage
    slurm_stage = slurm_stage or stage
    output_stage = output_stage or stage

    slurm: SlurmStage = cfg.slurm.stage(slurm_stage)
    env = _env()
    tmpl = env.get_template(f"{template_stage}.sbatch.j2")
    slurm_ctx = slurm.model_dump()
    if mem_override:
        slurm_ctx["mem"] = mem_override
    ctx = {
        "stage": stage,
        "experiment": cfg.experiment.name,
        "slurm": slurm_ctx,
        "account": cfg.slurm.account,
        "array_size": array_size,
        "log_path": str(cfg.log_path(output_stage)),
        "body_cmd": body_cmd,
        "jobs_csv": str(cfg.jobs_csv(output_stage)),
        "stage_in_dir": stage_in_dir,
    }
    if extra_ctx:
        ctx.update(extra_ctx)
    return tmpl.render(**ctx)


def write_sbatch(cfg: Config, stage: str, content: str) -> Path:
    path = cfg.sbatch_path(stage)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
    path.chmod(0o755)
    return path


def submit(
    sbatch_path: Path,
    *,
    array_size: int | None = None,
    array_limit: int | None = None,
    dependency: str | None = None,
    wait: bool = False,
) -> int:
    """Run `sbatch [--array=1-N[%M]] [--dependency=afterok:JOBID] PATH` and return job id."""
    if shutil.which("sbatch") is None:
        raise RuntimeError("sbatch not found on PATH; cannot submit")
    cmd = ["sbatch"]
    if array_size:
        array = f"1-{array_size}"
        if array_limit:
            array = f"{array}%{array_limit}"
        cmd += [f"--array={array}"]
    if dependency:
        cmd += [f"--dependency=afterok:{dependency}"]
    if wait:
        cmd += ["--wait"]
    cmd += [str(sbatch_path)]
    log.info("[stage]$[/] %s", " ".join(cmd))
    res = subprocess.run(cmd, check=True, capture_output=True, text=True)
    # "Submitted batch job 12345"
    m = re.search(r"Submitted batch job (\d+)", res.stdout)
    if not m:
        raise RuntimeError(f"Could not parse sbatch output: {res.stdout!r}")
    job_id = int(m.group(1))
    log.info("[ok]submitted job[/] [count]%d[/]", job_id)
    return job_id
