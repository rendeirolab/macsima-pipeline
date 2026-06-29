from pathlib import Path
from types import SimpleNamespace

from macsima_pipeline import cli


def test_run_all_submit_defers_mcmicro_planning(monkeypatch, tmp_path: Path) -> None:
    cfg = SimpleNamespace(
        experiment=SimpleNamespace(name="expt99"),
        suffix="",
        paths=SimpleNamespace(work_dir=tmp_path, logs_dir=Path("logs")),
    )
    calls: list[tuple[str, object]] = []

    monkeypatch.setattr(cli, "_load", lambda _path: cfg)
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
        "submit_mcmicro_launcher",
        lambda *_args, **kwargs: calls.append(("launcher", kwargs["dependency"])) or 202,
    )
    monkeypatch.setattr(
        cli.preprocess_stage,
        "run",
        lambda *_args, **kwargs: calls.append(("preprocess", kwargs["dependency"])) or 303,
    )
    monkeypatch.setattr(cli, "render_sbatch", lambda *_args, **_kwargs: "viz sbatch")
    monkeypatch.setattr(cli, "write_sbatch", lambda *_args, **_kwargs: tmp_path / "viz.sbatch")
    monkeypatch.setattr(
        cli,
        "submit_helper",
        lambda _sbatch, dep: calls.append(("viz", dep)) or 404,
    )

    cli.run_all(Path("config.yaml"), submit=True)

    assert calls == [
        ("launcher", "101"),
        ("preprocess", "202"),
        ("viz", "303"),
    ]
