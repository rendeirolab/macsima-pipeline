"""Config schema (pydantic) + YAML loader with `extends:` inheritance + placeholder interpolation."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field

from .utils import ensure_dir


# ---------- Sub-models -------------------------------------------------------


class ExperimentCfg(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str
    raw_root: Path
    roi_glob: str = "ROI*"
    roi_exclude: list[str] = Field(default_factory=list)
    roi_include: list[str] | None = None
    roi_metadata_csv: Path | None = None


class PathsCfg(BaseModel):
    model_config = ConfigDict(extra="forbid")
    work_dir: Path = Path(".")
    staging_out: Path = Path("mcmicro_output")
    mcmicro_out: Path = Path("mcmicro_output")
    zarr_out: str = "{experiment_name}_mcmicro{suffix}.zarr"
    h5ad_out: str = "{experiment_name}_cell_expression_mcmicro{suffix}.h5ad"
    phenotype_h5ad_out: str = "{experiment_name}_phenotyped_mcmicro{suffix}.h5ad"
    preprocess_parts_dir: str = "artifacts/{experiment_name}/preprocess_parts{suffix}"
    figures_dir: str = "figures/{experiment_name}"
    jobs_dir: Path = Path("jobs")
    logs_dir: Path = Path("logs")


class ContainersCfg(BaseModel):
    model_config = ConfigDict(extra="forbid")
    macsima2mc_sif: Path
    multiplex_macsima_sif: Path | None = None


class McmicroCfg(BaseModel):
    model_config = ConfigDict(extra="forbid")
    params_yaml: Path
    nextflow_config: Path | None = None
    sample_pattern: str = "rack-*-well-*-roi-*-exp-2"
    registration_pattern: str = "registration/*exp-2.ome.tif"
    background_pattern: str = "background/*exp-2_backsub.ome.tif"
    # true  -> use background-subtracted images only
    # false -> use registration images only
    # "auto" -> use whichever exists; if both exist, run BOTH variants
    # Preserve the historical programmatic default. The shipped YAML opts into
    # "auto" explicitly so it can render both available variants.
    background_subtraction: bool | Literal["auto"] = False
    markers_csv: str = "markers.csv"
    markers_bs_csv: str = "background/markers_bs.csv"


class SegmentationCfg(BaseModel):
    model_config = ConfigDict(extra="forbid")
    method: Literal["cellpose4"] = "cellpose4"
    model: str = "cpsam"
    channels: list[str] = Field(default_factory=lambda: ["DAPI"])
    min_area: int = 15
    gpu: bool = True


class PatchesCfg(BaseModel):
    model_config = ConfigDict(extra="forbid")
    patch_width: int | None = 2048


class PreprocessParallelCfg(BaseModel):
    model_config = ConfigDict(extra="forbid")
    max_workers: int = 4


class PreprocessCfg(BaseModel):
    model_config = ConfigDict(extra="forbid")
    segmentation: SegmentationCfg = SegmentationCfg()
    patches: PatchesCfg = PatchesCfg()
    parallel: PreprocessParallelCfg = PreprocessParallelCfg()
    scale_factors: list[int] = Field(default_factory=lambda: [2, 4])


class VizParallelCfg(BaseModel):
    model_config = ConfigDict(extra="forbid")
    workers: int = 8
    # Pass B (ROI grids) holds n_markers panels in one figure and is much
    # heavier per worker than Pass A. Cap it separately to avoid OOM. If None,
    # falls back to `workers`. Sequential rendering is the safe default.
    roi_workers: int | None = 1
    backend: str = "loky"


class VizCombination(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str
    markers: list[str]


class VizChannelQCCfg(BaseModel):
    model_config = ConfigDict(extra="forbid")
    enabled: bool = True
    sigma: float = 3.0
    outlier_z: float = 3.5
    min_snr: float = 3.0
    min_positive_fraction: float = 0.001
    max_saturated_fraction: float = 0.001
    tile_grid: tuple[int, int] = (8, 8)
    workers: int | None = None
    report_top_n: int = 30


class VizCfg(BaseModel):
    model_config = ConfigDict(extra="forbid")
    target_max_dim: int = 2048
    percentile_clip: tuple[float, float] = (1.0, 99.0)
    grid_ncols: int = 5
    fig_size_per_panel: tuple[float, float] = (5.0, 5.0)
    grid_wspace: float = 0.04
    grid_hspace: float = 0.12
    panel_title_size: float = 7.0
    figure_title_size: float = 12.0
    output_pad_inches: float = 0.02
    dpi: int = 300
    output_format: Literal["pdf", "png"] = "pdf"
    cmap: str = "gray"
    rasterized: bool = True
    pdf_compression: int = 9
    combinations: list[VizCombination] = Field(default_factory=list)
    cell_maps: bool = True
    cell_map_marker_top_n: int = 20
    cell_map_point_size: float = 1.0
    cell_map_marker_columns: int = 3
    channel_qc: VizChannelQCCfg = VizChannelQCCfg()
    parallel: VizParallelCfg = VizParallelCfg()
    cache_percentiles: bool = True


# ---------- Phenotype (Stage 4) ---------------------------------------------


class PhenotypeNormalizeCfg(BaseModel):
    """Per-marker intensity normalization applied before phenotyping.

    Pipeline: stash raw -> winsorize (clip outliers) -> transform -> z-score.
    Produces `layers[store_raw_layer]` (raw) and `layers[normalized_layer]`
    (normalized); `X` is set to the normalized matrix.
    """

    model_config = ConfigDict(extra="forbid")
    clip_percentile: float | None = 99.9
    clip_lower_percentile: float | None = None
    transform: Literal["arcsinh", "percentile", "none"] = "arcsinh"
    cofactor: float = 5.0
    cofactors: dict[str, float] = Field(default_factory=dict)
    percentile_norm_p: float = 99.0
    zscore: bool = True
    store_raw_layer: str = "counts"
    normalized_layer: str | None = "zscore"


class PhenotypeBatchCfg(BaseModel):
    """Batch handling at the intensity stage (preserves marker interpretability)."""

    model_config = ConfigDict(extra="forbid")
    method: Literal["none", "zscore_per_roi", "quantile_reference", "combat"] = "zscore_per_roi"
    batch_key: str = "ROI"
    reference: str | None = None
    min_cells_per_batch: int = 50


class PhenotypeAstirCfg(BaseModel):
    """Clean-room Astir engine (probabilistic per-cell assignment).

    Consumes RAW intensities (`use_layer`): the model does its own
    arcsinh/winsorize/z-score internally, so it must NOT receive pre-normalized
    input. `cofactor` must match the MACSima fluorescence scale.
    """

    model_config = ConfigDict(extra="forbid")
    enabled: bool = True
    cofactor: float = 5.0
    winsorize: tuple[float, float] = (0.0, 99.9)
    max_epochs: int = 50
    n_init: int = 5
    learning_rate: float = 1e-3
    batch_size: int = 16384
    random_seed: int = 0
    device: Literal["auto", "cpu", "cuda"] = "auto"
    precision: Literal["fp32", "bf16"] = "fp32"
    min_confidence: float = 0.7
    use_layer: str = "counts"
    include_batch_covariate: bool = True


class PhenotypeFlowsomCfg(BaseModel):
    """FlowSOM engine (SOM + consensus metaclustering, then signature labeling).

    Consumes the arcsinh + per-marker z-scored layer (`use_layer`).
    """

    model_config = ConfigDict(extra="forbid")
    enabled: bool = True
    grid_size: tuple[int, int] = (10, 10)
    n_metaclusters: int = 30
    som_iterations: int = 100
    train_subsample: int | None = 200_000
    scoring: Literal["signature_zscore", "signature_cosine"] = "signature_zscore"
    tau: float = 0.0
    random_seed: int = 0
    use_layer: str = "zscore"


class PhenotypeSpatialQCCfg(BaseModel):
    """Spatial-coherence QC: formalizes the "labels on the map make sense" check."""

    model_config = ConfigDict(extra="forbid")
    enabled: bool = True
    coord_type: Literal["generic"] = "generic"
    n_neighs: int = 6
    nhood_enrichment: bool = True
    homophily: bool = True
    n_perms: int = 1000
    min_cells_per_roi: int = 100
    random_seed: int = 0


class PhenotypeCfg(BaseModel):
    """Stage 4: normalize + phenotype (Astir + FlowSOM) + spatial QC."""

    model_config = ConfigDict(extra="forbid")
    enabled: bool = True
    signature_matrix: Path | None = None
    engines: list[Literal["astir", "flowsom"]] = Field(default_factory=lambda: ["astir", "flowsom"])
    primary_engine: Literal["astir", "flowsom"] = "astir"
    coarse_label_key: str = "cell_type_coarse"
    normalize: PhenotypeNormalizeCfg = PhenotypeNormalizeCfg()
    batch: PhenotypeBatchCfg = PhenotypeBatchCfg()
    astir: PhenotypeAstirCfg = PhenotypeAstirCfg()
    flowsom: PhenotypeFlowsomCfg = PhenotypeFlowsomCfg()
    spatial_qc: PhenotypeSpatialQCCfg = PhenotypeSpatialQCCfg()


class SlurmStage(BaseModel):
    model_config = ConfigDict(extra="forbid")
    partition: str
    qos: str
    cpus: int
    mem: str
    time: str
    gres: str | None = None
    comment: str | None = None


class SlurmCfg(BaseModel):
    model_config = ConfigDict(extra="forbid")
    account: str | None = None
    staging: SlurmStage
    mcmicro: SlurmStage
    preprocess: SlurmStage
    preprocess_merge: SlurmStage | None = None
    phenotype: SlurmStage | None = None
    viz: SlurmStage

    def stage(self, name: str) -> SlurmStage:
        """Return a configured stage, with sensible defaults for optional stages."""
        if name == "preprocess_merge" and self.preprocess_merge is None:
            return SlurmStage(
                partition=self.viz.partition,
                qos=self.viz.qos,
                cpus=self.viz.cpus,
                mem=self.preprocess.mem,
                time=self.preprocess.time,
            )
        if name == "phenotype" and self.phenotype is None:
            # Default to the GPU preprocess stage — the Astir engine uses torch.
            p = self.preprocess
            return SlurmStage(
                partition=p.partition,
                qos=p.qos,
                gres=p.gres,
                cpus=p.cpus,
                mem=p.mem,
                time=p.time,
                comment=p.comment,
            )
        stage = getattr(self, name)
        if stage is None:
            raise ValueError(f"SLURM stage {name!r} is not configured")
        return stage


class Config(BaseModel):
    """Top-level pipeline config."""

    model_config = ConfigDict(extra="forbid")

    experiment: ExperimentCfg
    paths: PathsCfg = PathsCfg()
    containers: ContainersCfg
    mcmicro: McmicroCfg
    preprocess: PreprocessCfg = PreprocessCfg()
    viz: VizCfg = VizCfg()
    phenotype: PhenotypeCfg = PhenotypeCfg()
    slurm: SlurmCfg

    # ---- derived helpers ----

    @staticmethod
    def suffix_for(bg: bool) -> str:
        """Output-path suffix per variant. True (bg-sub) -> "", False -> "_no_bs"."""
        return "" if bg else "_no_bs"

    @property
    def suffix(self) -> str:
        """Back-compat: only valid when background_subtraction is a bool.

        For `"auto"`, callers must iterate variants via `bg_modes()` and use
        `suffix_for(bg)` instead.
        """
        bg = self.mcmicro.background_subtraction
        if isinstance(bg, bool):
            return self.suffix_for(bg)
        # auto: pick the bg-sub suffix as a stable default for log/display.
        # Path-producing helpers should NOT call this in auto mode.
        return ""

    def bg_modes(self) -> list[bool]:
        """Which background-subtraction variants to run.

        - bool -> single-element list with that bool.
        - "auto" -> probe the staged mcmicro outputs and return [True, False],
          [True], or [False] depending on which image sets exist. If neither
          exists yet (e.g. dry-run before mcmicro has produced output), assume
          both variants will be produced -> [True, False].
        """
        bg = self.mcmicro.background_subtraction
        if isinstance(bg, bool):
            return [bg]
        base = self.paths.work_dir / self.paths.mcmicro_out / self.experiment.name
        has_bs = bool(list(base.rglob(self.mcmicro.background_pattern))) if base.exists() else False
        has_reg = bool(list(base.rglob(self.mcmicro.registration_pattern))) if base.exists() else False
        if not base.exists() or (not has_bs and not has_reg):
            return [True, False]
        modes: list[bool] = []
        if has_bs:
            modes.append(True)
        if has_reg:
            modes.append(False)
        return modes

    def _ctx(self, bg: bool | None = None) -> dict[str, str]:
        suf = self.suffix if bg is None else self.suffix_for(bg)
        return {"experiment_name": self.experiment.name, "suffix": suf}

    def resolve(self, template: str, bg: bool | None = None) -> str:
        """Expand `{experiment_name}` / `{suffix}` placeholders for a given variant."""
        return template.format(**self._ctx(bg))

    def zarr_path(self, bg: bool | None = None) -> Path:
        return self.paths.work_dir / self.resolve(self.paths.zarr_out, bg)

    def h5ad_path(self, bg: bool | None = None) -> Path:
        return self.paths.work_dir / self.resolve(self.paths.h5ad_out, bg)

    def phenotype_h5ad_path(self, bg: bool | None = None) -> Path:
        return self.paths.work_dir / self.resolve(self.paths.phenotype_h5ad_out, bg)

    def preprocess_parts_path(self, bg: bool | None = None) -> Path:
        return self.paths.work_dir / self.resolve(self.paths.preprocess_parts_dir, bg)

    def figures_dir(self) -> Path:
        return self.paths.work_dir / self.resolve(self.paths.figures_dir)

    def jobs_csv(self, stage: str) -> Path:
        return self.paths.work_dir / self.paths.jobs_dir / f"{stage}_{self.experiment.name}.csv"

    def sbatch_path(self, stage: str) -> Path:
        return self.paths.work_dir / self.paths.jobs_dir / f"{stage}_{self.experiment.name}.sbatch"

    def log_path(self, stage: str) -> Path:
        return self.paths.work_dir / self.paths.logs_dir / f"{stage}_{self.experiment.name}_%A_%a.out"


# ---------- YAML loader with `extends:` -------------------------------------


def _deep_merge(base: dict[str, Any], over: dict[str, Any]) -> dict[str, Any]:
    """Right-biased deep merge of plain dicts.

    None in `over` does NOT clobber a non-None value in `base` — this avoids
    YAML footguns like `combinations:` (with only commented-out items) parsing
    to None and overriding the default `[]`.
    """
    out = dict(base)
    for k, v in over.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _deep_merge(out[k], v)
        elif v is None and k in out and out[k] is not None:
            continue
        else:
            out[k] = v
    return out


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open() as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{path}: top-level must be a mapping")
    return data


def _resolve_merged(path: str | Path) -> dict[str, Any]:
    """Load a config YAML and recursively resolve its `extends:` chain.

    Returns the merged dict *before* pydantic validation, with REQUIRED sentinels
    stripped. Shared by `load_config` (single) and the batch expanders below.
    """
    path = Path(path)
    raw = _load_yaml(path)

    chain: list[dict[str, Any]] = []
    cur = raw
    visited: set[Path] = set()
    while "extends" in cur:
        parent_rel = cur.pop("extends")
        parent = (path.parent / parent_rel).resolve()
        if parent in visited:
            raise ValueError(f"Circular `extends` involving {parent}")
        visited.add(parent)
        parent_raw = _load_yaml(parent)
        chain.append(cur)
        cur = parent_raw

    merged = cur
    for layer in reversed(chain):
        merged = _deep_merge(merged, layer)

    # Strip REQUIRED sentinels so missing required keys produce pydantic errors, not literal "REQUIRED" values.
    _strip_required(merged)
    return merged


def load_config(path: str | Path) -> Config:
    """Load a single-experiment config YAML, recursively resolving `extends:` chains."""
    return Config.model_validate(_resolve_merged(path))


def _strip_required(d: Any) -> None:
    if isinstance(d, dict):
        for k, v in list(d.items()):
            if isinstance(v, str) and v == "REQUIRED":
                del d[k]
            else:
                _strip_required(v)
    elif isinstance(d, list):
        for v in d:
            _strip_required(v)


# ---------- Batch (multiple experiments in one config) ----------------------
#
# A batch config carries a top-level `experiments:` list instead of a single
# `experiment:` mapping. Each entry is expanded into a standalone
# single-experiment `Config` sharing every other section (containers, slurm,
# mcmicro, segmentation, viz, phenotype). `Config` itself stays single-experiment
# so all stage code, path helpers, and SLURM templates work unchanged.


def _is_batch(path: str | Path) -> bool:
    """True if the *top-level* config file declares `experiments:` (a batch).

    Detection reads the raw top file, not the merged dict: an inherited
    `experiment:` defaults block from `extends` must not be mistaken for a
    conflicting single-experiment declaration.
    """
    raw = _load_yaml(Path(path))
    if "experiment" in raw and "experiments" in raw:
        raise ValueError(
            f"{path}: define either `experiment` (single) or `experiments` (batch), not both"
        )
    return "experiments" in raw


def _split_batch(merged: dict[str, Any]) -> list[dict[str, Any]]:
    """Expand a merged batch dict into one single-experiment dict per entry.

    Each entry inherits the shared `experiment:` defaults block (e.g. `roi_glob`,
    `roi_exclude`) exactly as a single-experiment child would via `extends`.
    Raises on an empty list, a non-mapping entry, or duplicate names.
    """
    entries = merged.get("experiments")
    if not isinstance(entries, list) or not entries:
        raise ValueError("`experiments` must be a non-empty list")

    base_exp = merged.get("experiment") or {}
    shared = {k: v for k, v in merged.items() if k not in ("experiment", "experiments")}

    out: list[dict[str, Any]] = []
    names: list[Any] = []
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError(f"each `experiments` entry must be a mapping, got {type(entry).__name__}")
        exp = _deep_merge(base_exp, entry)
        names.append(exp.get("name"))
        out.append(_deep_merge(shared, {"experiment": exp}))

    dupes = sorted({n for n in names if names.count(n) > 1})
    if dupes:
        raise ValueError(f"duplicate experiment name(s) in batch: {dupes}")
    return out


def _check_only(names: set[str], only: list[str] | None) -> None:
    if only:
        missing = set(only) - names
        if missing:
            raise ValueError(f"--only names not found in config: {sorted(missing)}")


def expand_config(path: str | Path, only: list[str] | None = None) -> list[Config]:
    """Return one validated single-experiment `Config` per experiment.

    Single-experiment config -> `[Config]`. Batch config -> N `Config`s. In-memory
    only (writes nothing) — used by utilities and dry-run/logging. `only` filters by
    `experiment.name` (error if a requested name is absent).
    """
    merged = _resolve_merged(path)
    if _is_batch(path):
        cfgs = [Config.model_validate(d) for d in _split_batch(merged)]
    else:
        cfgs = [Config.model_validate(merged)]
    _check_only({c.experiment.name for c in cfgs}, only)
    if only:
        cfgs = [c for c in cfgs if c.experiment.name in set(only)]
    return cfgs


def materialize_configs(
    path: str | Path,
    *,
    only: list[str] | None = None,
    dest: str = "jobs/batch",
) -> list[tuple[str, Path]]:
    """Return `[(experiment_name, config_path)]` for the submit path to iterate.

    Single-experiment config: validate and return `[(name, path)]` — writes nothing
      (exact back-compat; the original file is used as-is).
    Batch config: for each entry build a fully-flattened standalone single-experiment
      config (no `extends`), validate it (fail fast), write it to
      `{work_dir}/{dest}/{name}.yaml`, and return `(name, absolute_path)`. Flattened
      files survive the SLURM planner boundary (they are re-loaded on a compute node).
    """
    path = Path(path)
    merged = _resolve_merged(path)

    if not _is_batch(path):
        cfg = Config.model_validate(merged)
        _check_only({cfg.experiment.name}, only)
        return [(cfg.experiment.name, path.resolve())]

    dicts = _split_batch(merged)
    _check_only({d["experiment"]["name"] for d in dicts}, only)
    if only:
        dicts = [d for d in dicts if d["experiment"]["name"] in set(only)]

    out: list[tuple[str, Path]] = []
    for d in dicts:
        cfg = Config.model_validate(d)  # fail fast; also confirms a valid single config
        name = cfg.experiment.name
        dest_dir = ensure_dir(cfg.paths.work_dir / dest)
        target = (dest_dir / f"{name}.yaml").resolve()
        with target.open("w") as f:
            yaml.safe_dump(d, f, sort_keys=False)
        out.append((name, target))
    return out
