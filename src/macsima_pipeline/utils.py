"""Small helpers shared across stages."""

from __future__ import annotations

import logging
from pathlib import Path

from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel
from rich.text import Text
from rich.theme import Theme

# Shared console — markup + soft wrapping. Stderr by default so stdout stays clean
# for piping (e.g. capturing a job id from `sbatch`).
_THEME = Theme(
    {
        "logging.level.info": "bold cyan",
        "logging.level.warning": "bold yellow",
        "logging.level.error": "bold red",
        "logging.level.debug": "dim",
        "stage": "bold magenta",
        "path": "cyan",
        "count": "bold green",
        "warn": "yellow",
        "bad": "bold red",
        "ok": "bold green",
    }
)
console = Console(theme=_THEME, stderr=True, soft_wrap=False, highlight=True)


def setup_logging(level: int = logging.INFO) -> logging.Logger:
    """Configure root logging with a RichHandler. Idempotent."""
    root = logging.getLogger()
    # Idempotent — re-running setup_logging shouldn't add duplicate handlers.
    if any(isinstance(h, RichHandler) for h in root.handlers):
        root.setLevel(level)
        return logging.getLogger("macsima_pipeline")

    # Drop any default handlers basicConfig may have installed previously.
    for h in list(root.handlers):
        root.removeHandler(h)

    handler = RichHandler(
        console=console,
        show_time=True,
        show_level=True,
        show_path=False,
        markup=True,
        rich_tracebacks=True,
        omit_repeated_times=False,
        log_time_format="[%H:%M:%S]",
    )
    handler.setFormatter(logging.Formatter("%(name)s — %(message)s"))
    root.addHandler(handler)
    root.setLevel(level)
    return logging.getLogger("macsima_pipeline")


def banner(title: str, subtitle: str | None = None) -> None:
    """Print a highlighted panel — use at the top of each CLI command."""
    text = Text(title, style="bold white on magenta")
    console.print(Panel(text, subtitle=subtitle, border_style="magenta", padding=(0, 2)))


def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def roi_index_from_name(name: str) -> int:
    """ROI1 -> 1, ROI23 -> 23. Raises ValueError on malformed names."""
    if not name.startswith("ROI"):
        raise ValueError(f"ROI name must start with 'ROI': {name!r}")
    return int(name[3:])


def roi_name_from_mcmicro_stem(stem: str) -> str:
    """Mirror of original logic: e.g. 'rack-01-well-C01-roi-003-exp-2' -> '003'."""
    parts = stem.split("-")
    if len(parts) < 6:
        raise ValueError(f"Unexpected mcmicro filename stem: {stem!r}")
    return parts[5]


def roi_name_from_results_stem(stem: str) -> str:
    """Results-tree image stem IS the roi name (e.g. '003').

    Consolidated images are named ``<roi>.ome.tif``; ``Path.stem`` strips only the
    final ``.tif`` and leaves a trailing ``.ome``, so drop it here.
    """
    return stem[:-4] if stem.endswith(".ome") else stem
