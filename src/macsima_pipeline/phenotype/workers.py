"""Stage 4 orchestrator: normalize -> phenotype (Astir + FlowSOM) -> spatial QC.

Mirrors `viz/workers.run_inproc`: iterates background-subtraction variants, is tolerant
of missing inputs in "auto" mode, and skips already-completed outputs (resume).
Runs on the merged per-variant h5ad (all ROIs jointly), a single job per variant.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import Config
from . import io, normalize, signature, spatial_qc
from .engines import astir as astir_engine
from .engines import flowsom as flowsom_engine

log = logging.getLogger(__name__)

_XY_CANDIDATES = [
    ("centroid_x", "centroid_y"),
    ("x", "y"),
    ("X", "Y"),
    ("x_centroid", "y_centroid"),
    ("X_centroid", "Y_centroid"),
    ("center_x", "center_y"),
]


def _resolve_xy(obs) -> tuple[str, str] | None:
    for x, y in _XY_CANDIDATES:
        if x in obs.columns and y in obs.columns:
            return x, y
    return None


def _sanitize_uns(value):
    """Recursively coerce to h5ad-safe types (None -> "", tuple -> list)."""
    if value is None:
        return ""
    if isinstance(value, tuple):
        return [_sanitize_uns(v) for v in value]
    if isinstance(value, list):
        return [_sanitize_uns(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _sanitize_uns(v) for k, v in value.items()}
    return value


def _coarse_labels(labels: pd.Series, coarse_map: dict[str, str]) -> pd.Series:
    return labels.map(lambda lab: coarse_map.get(lab, lab))


def _phenotype_adata(cfg: Config, adata, *, batch=None):
    """Run the full phenotype pipeline on an in-memory AnnData, in place.

    normalize -> batch-correct -> engines -> write labels to obs -> spatial QC ->
    populate uns['phenotype']. Shared by the per-experiment worker (`_run_variant`)
    and the cross-experiment joint runner (`run_joint`). `batch` overrides
    `cfg.phenotype.batch` (the joint runner passes an experiment-aware batch key).
    Returns (adata, results, qc, comp, agreement).
    """
    pcfg = cfg.phenotype
    batch = batch or pcfg.batch
    sig = signature.load_signature(cfg.paths.work_dir / pcfg.signature_matrix)
    sig.validate_against(list(adata.var_names))

    # --- normalization (Astir keeps raw; FlowSOM reads the normalized layer) ---
    normalize.stash_raw(adata, pcfg.normalize.store_raw_layer)
    normalize.normalize(adata, pcfg.normalize)
    normalize.apply_batch(adata, batch)
    if pcfg.normalize.normalized_layer:
        adata.layers[pcfg.normalize.normalized_layer] = np.asarray(adata.X, dtype=np.float32)

    # --- engines ---
    results = {}
    if "astir" in pcfg.engines and pcfg.astir.enabled:
        log.info("running astir engine")
        results["astir"] = astir_engine.run_astir(adata, sig, pcfg.astir, batch.batch_key)
    if "flowsom" in pcfg.engines and pcfg.flowsom.enabled:
        log.info("running flowsom engine")
        results["flowsom"] = flowsom_engine.run_flowsom(adata, sig, pcfg.flowsom, batch.batch_key)
    if not results:
        raise RuntimeError("no phenotype engines enabled")

    # --- write labels back to obs ---
    primary = results.get(pcfg.primary_engine) or next(iter(results.values()))
    coarse_map = sig.coarse_map()
    adata.obs["cell_type"] = pd.Categorical(primary.labels.astype(str))
    adata.obs["cell_type_confidence"] = primary.confidence.to_numpy(dtype=float)
    adata.obs[pcfg.coarse_label_key] = pd.Categorical(_coarse_labels(primary.labels.astype(str), coarse_map))
    if "astir" in results:
        adata.obs["astir_celltype"] = pd.Categorical(results["astir"].labels.astype(str))
    if "flowsom" in results:
        adata.obs["flowsom"] = pd.Categorical(results["flowsom"].cluster.astype(str))
        adata.obs["flowsom_celltype"] = pd.Categorical(results["flowsom"].labels.astype(str))

    agreement: dict = {}
    if "astir" in results and "flowsom" in results:
        agree, agreement = spatial_qc.cross_engine_agreement(
            results["astir"].labels.astype(str), results["flowsom"].labels.astype(str)
        )
        adata.obs["pheno_agree"] = agree.to_numpy()

    # --- spatial coherence QC ---
    qc: dict = {}
    comp = pd.DataFrame()
    xy = _resolve_xy(adata.obs)
    if xy is not None:
        spatial_qc.build_spatial(adata, *xy)
        if pcfg.spatial_qc.enabled:
            qc = spatial_qc.compute_spatial_qc(adata, pcfg.spatial_qc, "cell_type", batch.batch_key)
        comp = spatial_qc.composition_table(adata, "cell_type", batch.batch_key)
    else:
        log.warning("no centroid columns found; skipping spatial QC")

    adata.uns["phenotype"] = _sanitize_uns(
        {
            "engines": list(results.keys()),
            "primary_engine": pcfg.primary_engine,
            "signature": {
                name: {"positive": list(ct.positive), "negative": list(ct.negative), "parent": ct.parent}
                for name, ct in sig.cell_types.items()
            },
            "normalize": pcfg.normalize.model_dump(),
            "batch": batch.model_dump(),
            "astir": results["astir"].uns if "astir" in results else {},
            "flowsom": results["flowsom"].uns if "flowsom" in results else {},
            "agreement": agreement,
            "spatial_qc": qc,
        }
    )
    if not comp.empty:
        adata.uns["phenotype"]["composition"] = comp
    return adata, results, qc, comp, agreement


def _write_report(cfg: Config, adata, bg: bool, results: dict, qc: dict, comp, agreement: dict) -> None:
    """Best-effort QC report; a plotting failure must never lose the written h5ad."""
    try:
        from . import report

        report.write_phenotype_report(cfg, adata, bg, results, qc, comp, agreement)
    except Exception as e:  # noqa: BLE001
        log.warning("phenotype report failed (%s); h5ad already written", e)


def _run_variant(cfg: Config, bg: bool) -> Path:
    dest = cfg.phenotype_h5ad_path(bg)
    if io.phenotype_done(cfg, bg):
        log.info("skipping completed phenotype output: [path]%s[/]", dest)
        return dest

    src = cfg.h5ad_path(bg)
    if not src.is_file():
        raise FileNotFoundError(src)

    adata = io.read_cells(cfg, bg)
    adata, results, qc, comp, agreement = _phenotype_adata(cfg, adata)

    io.write_cells_atomic(adata, dest)
    log.info("phenotype done: [path]%s[/] (%d cells, %d types)", dest, adata.n_obs,
             len(adata.obs["cell_type"].cat.categories))
    _write_report(cfg, adata, bg, results, qc, comp, agreement)
    return dest


def run_inproc(cfg: Config) -> None:
    """Run the phenotype stage for every background variant, in-process."""
    if not cfg.phenotype.enabled:
        log.info("phenotype disabled; skipping stage")
        return
    if cfg.phenotype.signature_matrix is None:
        log.warning("phenotype enabled but signature_matrix is unset; skipping stage")
        return

    single_variant = isinstance(cfg.mcmicro.background_subtraction, bool)
    ran = 0
    for bg in cfg.bg_modes():
        try:
            _run_variant(cfg, bg)
            ran += 1
        except FileNotFoundError as e:
            if single_variant:
                raise
            log.warning("skipping phenotype variant bg=%s: input missing (%s)", bg, e)
    if ran == 0 and single_variant:
        raise FileNotFoundError("no phenotype variants produced output")


def _joint_cfg(base: Config, name: str) -> Config:
    """A single-experiment Config clone renamed to the joint dataset (drives output paths)."""
    return base.model_copy(
        update={"experiment": base.experiment.model_copy(update={"name": name, "roi_metadata_csv": None})}
    )


def run_joint(cfgs: list[Config], *, name: str, batch_key: str = "sample") -> list[Path]:
    """Phenotype every experiment JOINTLY: one model over the concatenated cells.

    Concatenates each experiment's preprocess-merge AnnData (inner-join on shared
    markers), tags each cell with `experiment` and a globally-unique `sample`
    (experiment|ROI), fits the engines ONCE, writes a combined phenotyped h5ad, then
    splits the joint labels back into each experiment's phenotype_h5ad (restoring
    original cell ids) so per-experiment viz picks them up. Returns combined path(s).
    """
    import anndata as ad

    if not cfgs:
        raise ValueError("run_joint requires at least one experiment")
    base = cfgs[0]
    if not base.phenotype.enabled:
        log.info("phenotype disabled; skipping joint stage")
        return []
    if base.phenotype.signature_matrix is None:
        log.warning("phenotype enabled but signature_matrix is unset; skipping joint stage")
        return []

    cfg_joint = _joint_cfg(base, name)
    batch = base.phenotype.batch.model_copy(update={"batch_key": batch_key})
    variants = sorted({bg for c in cfgs for bg in c.bg_modes()}, reverse=True)  # bg-sub first
    outputs: list[Path] = []

    for bg in variants:
        parts: list[tuple[Config, object]] = []
        for c in cfgs:
            src = c.h5ad_path(bg)
            if not src.is_file():
                log.warning("joint: skipping %s (no input for bg=%s): [path]%s[/]", c.experiment.name, bg, src)
                continue
            a = ad.read_h5ad(src)
            a.obs["experiment"] = c.experiment.name
            parts.append((c, a))
        if not parts:
            log.warning("joint: no inputs for variant bg=%s; skipping", bg)
            continue

        keys = [c.experiment.name for c, _ in parts]
        combined = ad.concat([a for _, a in parts], join="inner", keys=keys, index_unique="-")
        if "ROI" in combined.obs.columns:
            combined.obs["sample"] = combined.obs["experiment"].astype(str) + "|" + combined.obs["ROI"].astype(str)
        else:
            combined.obs["sample"] = combined.obs["experiment"].astype(str)
        log.info(
            "joint phenotype bg=%s: [count]%d[/] experiments, [count]%d[/] cells, [count]%d[/] common markers",
            bg, len(parts), combined.n_obs, combined.n_vars,
        )

        combined, results, qc, comp, agreement = _phenotype_adata(cfg_joint, combined, batch=batch)

        dest = cfg_joint.phenotype_h5ad_path(bg)
        io.write_cells_atomic(combined, dest)
        outputs.append(dest)
        log.info("joint phenotype done: [path]%s[/] (%d cells, %d types)", dest, combined.n_obs,
                 len(combined.obs["cell_type"].cat.categories))

        # split joint labels back to each experiment, restoring original cell ids
        offset = 0
        for c, a in parts:
            n = a.n_obs
            sub = combined[offset:offset + n].copy()
            sub.obs_names = a.obs_names
            io.write_cells_atomic(sub, c.phenotype_h5ad_path(bg))
            log.info("  joint split -> [path]%s[/]", c.phenotype_h5ad_path(bg))
            offset += n

        _write_report(cfg_joint, combined, bg, results, qc, comp, agreement)

    return outputs
