"""Cyclopts CLI: stage | mcmicro | preprocess | viz | all."""

from __future__ import annotations

import logging
import shlex
import sys
from pathlib import Path

from cyclopts import App

from . import mcmicro as mcmicro_stage
from . import panel as panel_stage
from . import preprocess as preprocess_stage
from . import staging as staging_stage
from .config import Config, load_config
from .slurm import render_sbatch, submit, write_sbatch
from .utils import banner, ensure_dir, setup_logging

log = setup_logging()

app = App(name="macsima-pipeline", help=__doc__)


def _load(cfg_path: Path) -> Config:
    cfg = load_config(cfg_path)
    log.info(
        "loaded config: experiment=[stage]%s[/] suffix=[path]%s[/]",
        cfg.experiment.name,
        cfg.suffix or "(none)",
    )
    return cfg


# --------------------------------------------------------------------------- #
#  Stage commands — sort_key preserves pipeline order in --help               #
# --------------------------------------------------------------------------- #


@app.command(sort_key=0)
def panel(config: Path) -> None:
    """Pre-staging: sanity-check the acquired marker panel (writes artifacts/<exp>/marker_panel.csv)."""
    banner("Marker panel — sanity check", subtitle=str(config))
    cfg = _load(config)
    mp = panel_stage.generate(cfg)
    log.info("marker panel -> [path]%s[/]", mp)


@app.command(sort_key=1)
def stage(
    config: Path,
    submit: bool = False,
    dependency: int | None = None,
    stage_roi: bool = False,
    task_id: int | None = None,
) -> int | None:
    """Stage 1: generate the marker panel, then stage raw MACSima cycles natively (per-ROI array).

    With --stage-roi --task-id N, stage a single ROI in the current process — this is what each
    SLURM array task re-invokes.
    """
    cfg = _load(config)
    if stage_roi:
        if task_id is None:
            raise ValueError("--stage-roi requires --task-id")
        staging_stage.stage_roi_inproc(cfg, task_id)
        return None
    banner("Stage 1 — staging (native)", subtitle=str(config))
    dep = str(dependency) if dependency else None
    return staging_stage.run(cfg, config.resolve(), do_submit=submit, dependency=dep)


@app.command(sort_key=2)
def mcmicro(
    config: Path,
    submit: bool = False,
    dependency: int | None = None,
    wait: bool = False,
) -> int | None:
    """Stage 2: discover staged samples and submit mcmicro array job."""
    banner("Stage 2 — mcmicro (Nextflow + Singularity)", subtitle=str(config))
    cfg = _load(config)
    dep = str(dependency) if dependency else None
    return mcmicro_stage.run(cfg, do_submit=submit, dependency=dep, wait=wait)


@app.command(sort_key=3)
def preprocess(
    config: Path,
    submit: bool = False,
    dependency: int | None = None,
    inproc: bool = False,
) -> int | None:
    """Stage 3: build SpatialData + segment + export AnnData.

    With --inproc, runs in the current process (this is what the SLURM script
    re-invokes). Without --inproc and --submit, submits the sbatch.
    """
    banner("Stage 3 — preprocess (SpatialData + Cellpose4)", subtitle=str(config))
    cfg = _load(config)
    if inproc:
        outputs = preprocess_stage.run_inproc(cfg)
        for zarr, h5ad in outputs:
            log.info("preprocess inproc done: zarr=[path]%s[/] h5ad=[path]%s[/]", zarr, h5ad)
        return None
    dep = str(dependency) if dependency else None
    return preprocess_stage.run(cfg, config.resolve(), do_submit=submit, dependency=dep)


@app.command(sort_key=4)
def viz(
    config: Path,
    submit: bool = False,
    dependency: int | None = None,
    inproc: bool = False,
) -> int | None:
    """Stage 4: render marker grids, ROI grids, and RGB combinations."""
    banner("Stage 4 — viz (PDF grids)", subtitle=str(config))
    cfg = _load(config)
    if inproc:
        # Import lazily so dry-run + sbatch generation don't pull matplotlib/tifffile
        from .viz import workers

        workers.run_inproc(cfg)
        return None

    ensure_dir(cfg.paths.work_dir / cfg.paths.logs_dir)
    python = Path(sys.executable)
    work_dir = cfg.paths.work_dir.resolve()
    body = (
        f"cd {shlex.quote(str(work_dir))}\n"
        f"uv run --frozen --no-sync "
        f"{shlex.quote(str(python))} -m macsima_pipeline.cli viz "
        f"--config {shlex.quote(str(config.resolve()))} --inproc"
    )
    content = render_sbatch(cfg, "viz", array_size=None, body_cmd=body)
    sbatch = write_sbatch(cfg, "viz", content)
    log.info("viz plan: sbatch=[path]%s[/]", sbatch)
    if not submit:
        log.warning("[warn](dry-run)[/] sbatch [path]%s[/]", sbatch)
        return None
    dep = str(dependency) if dependency else None
    return submit_helper(sbatch, dep)


def submit_helper(sbatch_path: Path, dep: str | None) -> int:
    return submit(sbatch_path, array_size=None, dependency=dep)


def submit_mcmicro_launcher(cfg: Config, config_path: Path, dependency: str) -> int:
    """Submit a barrier job that plans mcmicro after staging has produced samples."""
    ensure_dir(cfg.paths.work_dir / cfg.paths.logs_dir)
    python = Path(sys.executable)
    work_dir = cfg.paths.work_dir.resolve()
    body = (
        f"cd {shlex.quote(str(work_dir))}\n"
        f"uv run --frozen --no-sync "
        f"{shlex.quote(str(python))} -m macsima_pipeline.cli mcmicro "
        f"--config {shlex.quote(str(config_path))} --submit --wait"
    )
    content = render_sbatch(
        cfg,
        "mcmicro_launcher",
        array_size=None,
        body_cmd=body,
        template_stage="mcmicro",
        slurm_stage="mcmicro",
        output_stage="mcmicro_launcher",
    )
    sbatch = write_sbatch(cfg, "mcmicro_launcher", content)
    log.info("mcmicro launcher plan: sbatch=[path]%s[/]", sbatch)
    return submit(sbatch, array_size=None, dependency=dependency)


@app.command(name="all", sort_key=5)
def run_all(config: Path, submit: bool = False) -> None:
    """Submit the full chain: stage -> mcmicro -> preprocess -> viz with afterok deps."""
    banner("Full chain — stage → mcmicro → preprocess → viz", subtitle=str(config))
    cfg = _load(config)
    if not submit:
        # Dry-run: just emit each plan
        staging_stage.run(cfg, config.resolve(), do_submit=False)
        mcmicro_stage.run(cfg, do_submit=False)
        preprocess_stage.run(cfg, config.resolve(), do_submit=False)
        log.warning("[warn](dry-run)[/] viz plan would be rendered after preprocess submit")
        return

    config_path = config.resolve()

    j1 = staging_stage.run(cfg, config_path, do_submit=True)
    j2 = submit_mcmicro_launcher(cfg, config_path, dependency=str(j1))
    j3 = preprocess_stage.run(cfg, config_path, do_submit=True, dependency=str(j2))

    # Viz plan + submit with dep
    python = Path(sys.executable)
    body = (
        f"uv run --frozen --no-sync "
        f"{shlex.quote(str(python))} -m macsima_pipeline.cli viz "
        f"--config {shlex.quote(str(config_path))} --inproc"
    )
    content = render_sbatch(cfg, "viz", array_size=None, body_cmd=body)
    sbatch = write_sbatch(cfg, "viz", content)
    j4 = submit_helper(sbatch, str(j3))
    log.info(
        "[ok]chain submitted[/]: stage=[count]%d[/] mcmicro=[count]%d[/] "
        "preprocess=[count]%d[/] viz=[count]%d[/]",
        j1, j2, j3, j4,
    )


def main() -> None:
    """Entry point for `macsima-pipeline` console script."""
    logging.getLogger().setLevel(logging.INFO)
    app()


if __name__ == "__main__":
    main()
