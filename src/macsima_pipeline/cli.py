"""Cyclopts CLI: stage | mcmicro | preprocess | phenotype | viz | all."""

from __future__ import annotations

import logging
import shlex
import sys
from pathlib import Path

from cyclopts import App

from . import mcmicro as mcmicro_stage
from . import preprocess as preprocess_stage
from . import scaffold
from . import staging as staging_stage
from .config import Config, expand_config, load_config, materialize_configs
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


def _expand(config: Path, only: list[str] | None = None) -> list[Path]:
    """Resolve a config into one path per experiment for a stage command to iterate.

    Single-experiment config -> `[config]` (nothing written). Batch config -> one
    materialized, fully-flattened per-experiment config file under `jobs/batch/`
    (planners re-load these by path on a compute node). `only` filters by name.
    """
    pairs = materialize_configs(config, only=only)
    if len(pairs) > 1:
        log.info(
            "batch: [count]%d[/] experiments %s",
            len(pairs),
            [name for name, _ in pairs],
        )
    return [path for _, path in pairs]


# --------------------------------------------------------------------------- #
#  Stage commands — sort_key preserves pipeline order in --help               #
# --------------------------------------------------------------------------- #


@app.command(sort_key=1)
def stage(
    config: Path,
    submit: bool = False,
    dependency: int | None = None,
    only: list[str] | None = None,
) -> int | None:
    """Stage 1: discover ROIs and submit `macsima2mc` array job.

    A batch config (top-level `experiments:`) submits one staging job per experiment;
    `--only NAME` restricts to the named experiment(s).
    """
    results = [_stage_one(p, submit, dependency) for p in _expand(config, only)]
    return results[0] if len(results) == 1 else None


def _stage_one(config: Path, submit: bool, dependency: int | None) -> int | None:
    banner("Stage 1 — staging (macsima2mc)", subtitle=str(config))
    cfg = _load(config)
    dep = str(dependency) if dependency else None
    return staging_stage.run(cfg, do_submit=submit, dependency=dep)


@app.command(sort_key=2)
def mcmicro(
    config: Path,
    submit: bool = False,
    dependency: int | None = None,
    wait: bool = False,
    only: list[str] | None = None,
) -> int | None:
    """Stage 2: discover staged samples and submit mcmicro array job."""
    results = [_mcmicro_one(p, submit, dependency, wait) for p in _expand(config, only)]
    return results[0] if len(results) == 1 else None


def _mcmicro_one(config: Path, submit: bool, dependency: int | None, wait: bool) -> int | None:
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
    only: list[str] | None = None,
) -> int | None:
    """Stage 3: build SpatialData + segment + export AnnData.

    With --inproc, runs all ROIs in the current process for local/debug use.
    Without --inproc, plans or submits the SLURM worker array plus merge job.
    """
    results = [_preprocess_one(p, submit, dependency, inproc) for p in _expand(config, only)]
    return results[0] if len(results) == 1 else None


def _preprocess_one(
    config: Path, submit: bool, dependency: int | None, inproc: bool
) -> int | None:
    banner("Stage 3 — preprocess (SpatialData + Cellpose4)", subtitle=str(config))
    cfg = _load(config)
    if inproc:
        outputs = preprocess_stage.run_inproc(cfg)
        for zarr, h5ad in outputs:
            log.info("preprocess inproc done: zarr=[path]%s[/] h5ad=[path]%s[/]", zarr, h5ad)
        return None
    dep = str(dependency) if dependency else None
    return preprocess_stage.run(cfg, config.resolve(), do_submit=submit, dependency=dep)


@app.command(name="preprocess-worker", sort_key=96)
def preprocess_worker(config: Path, jobs_csv: Path, task_id: int) -> None:
    """Internal: run one SLURM-array preprocessing worker."""
    banner("Stage 3 worker — preprocess ROI", subtitle=f"{config} :: task {task_id}")
    cfg = _load(config)
    zarr, h5ad = preprocess_stage.run_worker(cfg, jobs_csv, task_id)
    log.info("preprocess worker done: zarr=[path]%s[/] h5ad=[path]%s[/]", zarr, h5ad)


@app.command(name="preprocess-merge", sort_key=97)
def preprocess_merge(config: Path, jobs_csv: Path) -> None:
    """Internal: merge preprocessing worker outputs into final artifacts."""
    banner("Stage 3 merge — preprocess parts", subtitle=str(config))
    cfg = _load(config)
    outputs = preprocess_stage.merge_preprocess_parts(cfg, jobs_csv)
    for zarr, h5ad in outputs:
        log.info("preprocess merge done: zarr=[path]%s[/] h5ad=[path]%s[/]", zarr, h5ad)


@app.command(sort_key=4)
def phenotype(
    config: Path,
    submit: bool = False,
    dependency: int | None = None,
    inproc: bool = False,
    only: list[str] | None = None,
) -> int | None:
    """Stage 4: normalize, phenotype (Astir + FlowSOM), spatial QC; write back AnnData.

    With --inproc, runs in the current process. Otherwise plans or submits a single
    SLURM job. Skips gracefully (exit 0) when phenotype is disabled or no signature
    matrix is configured, so the downstream chain still runs.
    """
    results = [_phenotype_one(p, submit, dependency, inproc) for p in _expand(config, only)]
    return results[0] if len(results) == 1 else None


def _phenotype_one(
    config: Path, submit: bool, dependency: int | None, inproc: bool
) -> int | None:
    banner("Stage 4 — phenotype (normalize + Astir/FlowSOM + spatial QC)", subtitle=str(config))
    cfg = _load(config)
    if inproc:
        from .phenotype import workers

        workers.run_inproc(cfg)
        return None

    ensure_dir(cfg.paths.work_dir / cfg.paths.logs_dir)
    python = Path(sys.executable)
    work_dir = cfg.paths.work_dir.resolve()
    body = (
        f"cd {shlex.quote(str(work_dir))}\n"
        f"uv run --frozen --no-sync "
        f"{shlex.quote(str(python))} -m macsima_pipeline.cli phenotype "
        f"--config {shlex.quote(str(config.resolve()))} --inproc"
    )
    content = render_sbatch(cfg, "phenotype", array_size=None, body_cmd=body)
    sbatch = write_sbatch(cfg, "phenotype", content)
    log.info("phenotype plan: sbatch=[path]%s[/]", sbatch)
    if not submit:
        log.warning("[warn](dry-run)[/] sbatch [path]%s[/]", sbatch)
        return None
    dep = str(dependency) if dependency else None
    return submit_helper(sbatch, dep)


@app.command(sort_key=5)
def viz(
    config: Path,
    submit: bool = False,
    dependency: int | None = None,
    inproc: bool = False,
    only: list[str] | None = None,
) -> int | None:
    """Stage 5: render marker grids, ROI grids, RGB combinations, and cell-map QC."""
    results = [_viz_one(p, submit, dependency, inproc) for p in _expand(config, only)]
    return results[0] if len(results) == 1 else None


def _viz_one(config: Path, submit: bool, dependency: int | None, inproc: bool) -> int | None:
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


def _submit_viz(cfg: Config, config_path: Path, dependency: str | None) -> int:
    """Render and submit the viz stage with an optional SLURM dependency."""
    python = Path(sys.executable)
    work_dir = cfg.paths.work_dir.resolve()
    body = (
        f"cd {shlex.quote(str(work_dir))}\n"
        f"uv run --frozen --no-sync "
        f"{shlex.quote(str(python))} -m macsima_pipeline.cli viz "
        f"--config {shlex.quote(str(config_path))} --inproc"
    )
    content = render_sbatch(cfg, "viz", array_size=None, body_cmd=body)
    sbatch = write_sbatch(cfg, "viz", content)
    return submit_helper(sbatch, dependency)


def _submit_phenotype(cfg: Config, config_path: Path, dependency: str | None) -> int:
    """Render and submit the phenotype stage with an optional SLURM dependency."""
    python = Path(sys.executable)
    work_dir = cfg.paths.work_dir.resolve()
    body = (
        f"cd {shlex.quote(str(work_dir))}\n"
        f"uv run --frozen --no-sync "
        f"{shlex.quote(str(python))} -m macsima_pipeline.cli phenotype "
        f"--config {shlex.quote(str(config_path))} --inproc"
    )
    content = render_sbatch(cfg, "phenotype", array_size=None, body_cmd=body)
    sbatch = write_sbatch(cfg, "phenotype", content)
    return submit_helper(sbatch, dependency)


def submit_mcmicro_planner(cfg: Config, config_path: Path, dependency: str) -> int:
    """Submit a short continuation job after staging.

    The planner submits the real mcmicro array, then submits downstream stages
    against the real array/preprocess job IDs. It does not wait for mcmicro.
    """
    ensure_dir(cfg.paths.work_dir / cfg.paths.logs_dir)
    python = Path(sys.executable)
    work_dir = cfg.paths.work_dir.resolve()
    body = (
        f"cd {shlex.quote(str(work_dir))}\n"
        f"uv run --frozen --no-sync "
        f"{shlex.quote(str(python))} -m macsima_pipeline.cli all-after-stage "
        f"--config {shlex.quote(str(config_path))}"
    )
    content = render_sbatch(
        cfg,
        "mcmicro_planner",
        array_size=None,
        body_cmd=body,
        template_stage="planner",
        slurm_stage="mcmicro",
        output_stage="mcmicro_planner",
    )
    sbatch = write_sbatch(cfg, "mcmicro_planner", content)
    log.info("mcmicro planner plan: sbatch=[path]%s[/]", sbatch)
    return submit(sbatch, array_size=None, dependency=dependency)


def submit_preprocess_viz_planner(cfg: Config, config_path: Path, dependency: str) -> int:
    """Submit a short continuation after mcmicro finishes.

    The planner discovers concrete mcmicro image outputs, submits the preprocess
    worker array and merge job, then submits viz against the merge job id.
    """
    ensure_dir(cfg.paths.work_dir / cfg.paths.logs_dir)
    python = Path(sys.executable)
    work_dir = cfg.paths.work_dir.resolve()
    body = (
        f"cd {shlex.quote(str(work_dir))}\n"
        f"uv run --frozen --no-sync "
        f"{shlex.quote(str(python))} -m macsima_pipeline.cli all-after-mcmicro "
        f"--config {shlex.quote(str(config_path))}"
    )
    content = render_sbatch(
        cfg,
        "preprocess_planner",
        array_size=None,
        body_cmd=body,
        template_stage="planner",
        slurm_stage="preprocess_merge",
        output_stage="preprocess_planner",
    )
    sbatch = write_sbatch(cfg, "preprocess_planner", content)
    log.info("preprocess planner plan: sbatch=[path]%s[/]", sbatch)
    return submit(sbatch, array_size=None, dependency=dependency)


@app.command(name="all-after-stage", sort_key=99)
def run_all_after_stage(config: Path) -> None:
    """Internal continuation: submit mcmicro, then a preprocess/viz planner."""
    banner("Full chain continuation — mcmicro → preprocess planner", subtitle=str(config))
    cfg = _load(config)
    config_path = config.resolve()

    j2 = mcmicro_stage.run(cfg, do_submit=True)
    if j2 is None:
        raise RuntimeError("mcmicro submission did not return a job id")
    j3 = submit_preprocess_viz_planner(cfg, config_path, dependency=str(j2))
    if j3 is None:
        raise RuntimeError("preprocess planner submission did not return a job id")

    log.info(
        "[ok]continuation submitted[/]: mcmicro=[count]%d[/] preprocess_planner=[count]%d[/]",
        j2,
        j3,
    )


@app.command(name="all-after-mcmicro", sort_key=100)
def run_all_after_mcmicro(config: Path) -> None:
    """Internal continuation: submit preprocess workers/merge and viz after mcmicro."""
    banner("Full chain continuation — preprocess → viz", subtitle=str(config))
    cfg = _load(config)
    config_path = config.resolve()

    j3 = preprocess_stage.run(cfg, config_path, do_submit=True)
    if j3 is None:
        raise RuntimeError("preprocess submission did not return a merge job id")
    j3b = _submit_phenotype(cfg, config_path, dependency=str(j3))
    j4 = _submit_viz(cfg, config_path, dependency=str(j3b))

    log.info(
        "[ok]continuation submitted[/]: preprocess_merge=[count]%d[/] "
        "phenotype=[count]%d[/] viz=[count]%d[/]",
        j3,
        j3b,
        j4,
    )


@app.command(name="all", sort_key=6)
def run_all(config: Path, submit: bool = False, only: list[str] | None = None) -> None:
    """Submit the full chain: stage -> mcmicro -> preprocess -> phenotype -> viz.

    A batch config (top-level `experiments:`) submits one independent chain per
    experiment; `--only NAME` restricts to the named experiment(s).
    """
    for path in _expand(config, only):
        _run_all_one(path, submit)


def _run_all_one(config: Path, submit: bool) -> None:
    banner("Full chain — stage → mcmicro → preprocess → viz", subtitle=str(config))
    cfg = _load(config)
    if not submit:
        # Dry-run: just emit each plan
        staging_stage.run(cfg, do_submit=False)
        mcmicro_stage.run(cfg, do_submit=False)
        preprocess_stage.run(cfg, config.resolve(), do_submit=False)
        log.warning("[warn](dry-run)[/] phenotype + viz plans would be rendered after preprocess submit")
        return

    config_path = config.resolve()

    j1 = staging_stage.run(cfg, do_submit=True)
    j2 = submit_mcmicro_planner(cfg, config_path, dependency=str(j1))
    log.info(
        "[ok]chain submitted[/]: stage=[count]%d[/] planner=[count]%d[/]. "
        "The planner will submit mcmicro, preprocess, and viz with real job dependencies.",
        j1, j2,
    )


# --------------------------------------------------------------------------- #
#  Scaffold utilities — generate the CSV inputs the pipeline consumes          #
# --------------------------------------------------------------------------- #


@app.command(name="gen-roi-metadata", sort_key=7)
def gen_roi_metadata(
    config: Path,
    output: Path | None = None,
    columns: list[str] | None = None,
    experiment: str | None = None,
    force: bool = False,
) -> None:
    """Write a template roi_metadata.csv (ROI column pre-filled + empty user columns).

    Pre-staging: reads raw_root to list the exact ROIs the pipeline will process. For a
    batch config, writes one file per experiment. --columns adds empty user columns;
    --experiment NAME targets one experiment; --force overwrites an existing file.
    """
    banner("Utility — gen roi_metadata template", subtitle=str(config))
    cfgs = expand_config(config, only=[experiment] if experiment else None)
    if output is not None and len(cfgs) > 1:
        raise ValueError("--output applies to a single experiment; use --experiment NAME to pick one")
    for cfg in cfgs:
        scaffold.gen_roi_metadata(
            cfg, config_path=config, output=output, extra_columns=columns, force=force
        )


@app.command(name="gen-markers", sort_key=8)
def gen_markers(
    config: Path,
    output: Path | None = None,
    experiment: str | None = None,
    bg: bool = False,
    force: bool = False,
) -> None:
    """Consolidate the macsima2mc-generated markers.csv into a canonical panel.

    Post-staging: reads the markers.csv from the first staged sample, normalizes the
    `remove` column, and writes a per-experiment panel (a review/curation artifact — it
    does NOT change what preprocess reads). --bg uses markers_bs.csv; --force overwrites.
    """
    banner("Utility — gen markers panel", subtitle=str(config))
    cfgs = expand_config(config, only=[experiment] if experiment else None)
    if output is not None and len(cfgs) > 1:
        raise ValueError("--output applies to a single experiment; use --experiment NAME to pick one")
    for cfg in cfgs:
        scaffold.gen_markers(cfg, config_path=config, output=output, bg=bg, force=force)


@app.command(name="gen-signature", sort_key=9)
def gen_signature(
    config: Path,
    output: Path | None = None,
    experiment: str | None = None,
    bg: bool = False,
    force: bool = False,
) -> None:
    """Scaffold the Astir/FlowSOM signature (marker->cell-type table) from the panel.

    Post-staging: reads the marker panel and writes a signature YAML template listing
    all panel markers + example cell types to curate. Set phenotype.signature_matrix at
    the result (otherwise the phenotype stage skips). --experiment NAME targets one.
    """
    banner("Utility — gen signature template", subtitle=str(config))
    cfgs = expand_config(config, only=[experiment] if experiment else None)
    if output is not None and len(cfgs) > 1:
        raise ValueError("--output applies to a single experiment; use --experiment NAME to pick one")
    for cfg in cfgs:
        scaffold.gen_signature(cfg, config_path=config, output=output, bg=bg, force=force)


def _joint_name(config: Path) -> str:
    """Dataset name for joint outputs: the batch folder name, else the config file stem."""
    p = Path(config)
    return p.parent.name if p.name in ("config.yaml", "config.yml") else p.stem


@app.command(name="phenotype-joint", sort_key=10)
def phenotype_joint(
    config: Path,
    name: str | None = None,
    batch_key: str = "sample",
    only: list[str] | None = None,
    submit: bool = False,
    dependency: int | None = None,
    inproc: bool = False,
) -> int | None:
    """Stage 4 (joint): phenotype ALL experiments in a batch together (one model).

    Concatenates every experiment's cell-expression h5ad (inner-join on shared markers),
    fits Astir/FlowSOM ONCE, writes a combined phenotyped h5ad, and splits the joint
    labels back into each experiment's phenotyped h5ad (so per-experiment viz uses them).
    Run after preprocess. --inproc runs locally; otherwise submits one SLURM job.
    `--batch-key` (default `sample` = experiment|ROI) sets the batch-correction unit.
    """
    banner("Stage 4 (joint) — phenotype across experiments", subtitle=str(config))
    cfgs = expand_config(config, only=only)
    joint_name = name or _joint_name(config)
    if len(cfgs) < 2:
        log.warning("joint phenotyping expects a batch (>=2 experiments); got [count]%d[/]", len(cfgs))

    if inproc:
        from .phenotype import workers

        workers.run_joint(cfgs, name=joint_name, batch_key=batch_key)
        return None

    base = cfgs[0]
    cfg_joint = base.model_copy(
        update={"experiment": base.experiment.model_copy(update={"name": joint_name})}
    )
    ensure_dir(cfg_joint.paths.work_dir / cfg_joint.paths.logs_dir)
    python = Path(sys.executable)
    work_dir = cfg_joint.paths.work_dir.resolve()
    only_flag = "".join(f" --only {shlex.quote(n)}" for n in (only or []))
    body = (
        f"cd {shlex.quote(str(work_dir))}\n"
        f"uv run --frozen --no-sync "
        f"{shlex.quote(str(python))} -m macsima_pipeline.cli phenotype-joint "
        f"--config {shlex.quote(str(config.resolve()))} --inproc "
        f"--name {shlex.quote(joint_name)} --batch-key {shlex.quote(batch_key)}{only_flag}"
    )
    content = render_sbatch(cfg_joint, "phenotype", array_size=None, body_cmd=body)
    sbatch = write_sbatch(cfg_joint, "phenotype", content)
    log.info("joint phenotype plan: sbatch=[path]%s[/]", sbatch)
    if not submit:
        log.warning("[warn](dry-run)[/] sbatch [path]%s[/]", sbatch)
        return None
    dep = str(dependency) if dependency else None
    return submit_helper(sbatch, dep)


def main() -> None:
    """Entry point for `macsima-pipeline` console script."""
    logging.getLogger().setLevel(logging.INFO)
    app()


if __name__ == "__main__":
    main()
