"""Config schema (pydantic) + YAML loader with `extends:` inheritance + placeholder interpolation."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field


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
    patch_width: int | None = None


class PreprocessCfg(BaseModel):
    model_config = ConfigDict(extra="forbid")
    segmentation: SegmentationCfg = SegmentationCfg()
    patches: PatchesCfg = PatchesCfg()
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
    parallel: VizParallelCfg = VizParallelCfg()
    cache_percentiles: bool = True


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
    viz: SlurmStage


class Config(BaseModel):
    """Top-level pipeline config."""

    model_config = ConfigDict(extra="forbid")

    experiment: ExperimentCfg
    paths: PathsCfg = PathsCfg()
    containers: ContainersCfg
    mcmicro: McmicroCfg
    preprocess: PreprocessCfg = PreprocessCfg()
    viz: VizCfg = VizCfg()
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


def load_config(path: str | Path) -> Config:
    """Load a config YAML, recursively resolving `extends:` chains."""
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

    return Config.model_validate(merged)


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
