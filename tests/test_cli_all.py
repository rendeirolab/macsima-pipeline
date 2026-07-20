from pathlib import Path
from types import SimpleNamespace

from macsima_pipeline import cli


def test_run_all_submit_defers_downstream_planning(monkeypatch, tmp_path: Path) -> None:
    cfg = SimpleNamespace(
        experiment=SimpleNamespace(name="expt99"),
        suffix="",
        paths=SimpleNamespace(work_dir=tmp_path, logs_dir=Path("logs")),
    )
    calls: list[tuple[str, object]] = []

    monkeypatch.setattr(cli, "_load", lambda _path: cfg)
    # Batch expansion is exercised in test_batch.py; here keep the single fake path.
    monkeypatch.setattr(cli, "_expand", lambda config, only=None: [config])
    monkeypatch.setattr(cli.staging_stage, "run", lambda *_args, **_kwargs: 101)
    monkeypatch.setattr(
        cli.mcmicro_stage,
        "run",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("mcmicro planned too early")
        ),
    )
    monkeypatch.setattr(
        cli,
        "submit_mcmicro_planner",
        lambda *_args, **kwargs: calls.append(("planner", kwargs["dependency"])) or 202,
    )
    monkeypatch.setattr(
        cli.preprocess_stage,
        "run",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("preprocess planned too early")
        ),
    )
    monkeypatch.setattr(
        cli,
        "_submit_viz",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("viz planned too early")
        ),
    )

    cli.run_all(Path("config.yaml"), submit=True)

    assert calls == [
        ("planner", "101"),
    ]


def test_run_all_after_stage_defers_preprocess_until_mcmicro_done(
    monkeypatch,
    tmp_path: Path,
) -> None:
    cfg = SimpleNamespace(
        experiment=SimpleNamespace(name="expt99"),
        suffix="",
        paths=SimpleNamespace(work_dir=tmp_path, logs_dir=Path("logs")),
    )
    calls: list[tuple[str, object]] = []

    monkeypatch.setattr(cli, "_load", lambda _path: cfg)
    monkeypatch.setattr(
        cli.mcmicro_stage,
        "run",
        lambda *_args, **kwargs: calls.append(("mcmicro", kwargs.get("wait"))) or 202,
    )
    monkeypatch.setattr(
        cli,
        "submit_preprocess_viz_planner",
        lambda *_args, **kwargs: calls.append(("preprocess_planner", kwargs["dependency"])) or 303,
    )
    monkeypatch.setattr(
        cli.preprocess_stage,
        "run",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("preprocess planned before mcmicro finished")
        ),
    )
    monkeypatch.setattr(
        cli,
        "_submit_viz",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("viz planned before preprocess merge exists")
        ),
    )

    cli.run_all_after_stage(tmp_path / "config.yaml")

    assert calls == [
        ("mcmicro", None),
        ("preprocess_planner", "202"),
    ]


def test_run_all_after_mcmicro_viz_depends_on_preprocess_merge(
    monkeypatch,
    tmp_path: Path,
) -> None:
    cfg = SimpleNamespace(
        experiment=SimpleNamespace(name="expt99"),
        suffix="",
        paths=SimpleNamespace(work_dir=tmp_path, logs_dir=Path("logs")),
    )
    calls: list[tuple[str, object]] = []

    monkeypatch.setattr(cli, "_load", lambda _path: cfg)
    monkeypatch.setattr(
        cli.preprocess_stage,
        "run",
        lambda *_args, **kwargs: calls.append(("preprocess", kwargs.get("dependency"))) or 303,
    )
    monkeypatch.setattr(
        cli,
        "_submit_phenotype",
        lambda *_args, **kwargs: calls.append(("phenotype", kwargs["dependency"])) or 350,
    )
    monkeypatch.setattr(
        cli,
        "_submit_viz",
        lambda *_args, **kwargs: calls.append(("viz", kwargs["dependency"])) or 404,
    )

    cli.run_all_after_mcmicro(tmp_path / "config.yaml")

    assert calls == [
        ("preprocess", None),
        ("phenotype", "303"),
        ("viz", "350"),
    ]
