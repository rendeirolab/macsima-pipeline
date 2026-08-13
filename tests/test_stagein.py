"""Unit tests for stage-in (copy inputs to fast local storage) + --mem sizing."""

from __future__ import annotations

from pathlib import Path

from macsima_pipeline.config import StageInCfg
from macsima_pipeline.stagein import plan_mem, staged, staged_size


def _write(p: Path, nbytes: int) -> Path:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(b"\0" * nbytes)
    return p


# --------------------------------------------------------------------------- #
#  plan_mem                                                                    #
# --------------------------------------------------------------------------- #


def test_plan_mem_empty_is_working_only() -> None:
    assert plan_mem([], working_gb=32, safety=1.15, cap_gb=200) == "32G"


def test_plan_mem_adds_staged_plus_working() -> None:
    # 10 GB * 1.15 = 11.5 -> ceil 12; + 100 = 112
    assert plan_mem([10_000_000_000], working_gb=100, safety=1.15, cap_gb=200) == "112G"


def test_plan_mem_sizes_for_largest() -> None:
    sizes = [7_000_000_000, 81_000_000_000, 20_000_000_000]
    # 81 * 1.15 = 93.15 -> ceil 94; + 100 = 194
    assert plan_mem(sizes, working_gb=100, safety=1.15, cap_gb=200) == "194G"


def test_plan_mem_clamped_by_cap() -> None:
    assert plan_mem([200_000_000_000], working_gb=100, safety=1.15, cap_gb=200) == "200G"


# --------------------------------------------------------------------------- #
#  staged_size                                                                 #
# --------------------------------------------------------------------------- #


def test_staged_size_file_and_dir(tmp_path: Path) -> None:
    f = _write(tmp_path / "a.bin", 1234)
    assert staged_size(f) == 1234

    d = tmp_path / "folder"
    _write(d / "x.bin", 100)
    _write(d / "sub" / "y.bin", 250)
    assert staged_size(d) == 350


# --------------------------------------------------------------------------- #
#  staged()                                                                    #
# --------------------------------------------------------------------------- #


def test_staged_disabled_yields_original(tmp_path: Path) -> None:
    src = _write(tmp_path / "img.tif", 500)
    for cfg in (None, StageInCfg(dir=None, working_gb=1)):
        with staged(src, cfg) as p:
            assert p == src


def test_staged_copies_file_and_cleans_up(tmp_path: Path) -> None:
    src = _write(tmp_path / "img.tif", 4096)
    ram = tmp_path / "ram"
    ram.mkdir()
    # mem_charged=False -> only free-space gates the copy (plenty here)
    cfg = StageInCfg(dir=ram, mem_charged=False, working_gb=1)

    with staged(src, cfg) as p:
        assert p != src
        assert ram in p.parents
        assert p.read_bytes() == src.read_bytes()
        staged_dir = p.parent

    assert not staged_dir.exists()  # removed on exit
    assert list(ram.glob("macsima.*")) == []  # no leaked staging dirs
    assert src.exists()  # original untouched


def test_staged_copies_directory(tmp_path: Path) -> None:
    src = tmp_path / "cycle"
    _write(src / "t1.tif", 100)
    _write(src / "t2.tif", 100)
    ram = tmp_path / "ram"
    ram.mkdir()
    cfg = StageInCfg(dir=ram, mem_charged=False, working_gb=1)

    with staged(src, cfg) as p:
        assert p.is_dir()
        assert (p / "t1.tif").exists() and (p / "t2.tif").exists()
        staged_dir = p.parent

    assert not staged_dir.exists()


def test_staged_skips_when_over_budget(tmp_path: Path) -> None:
    src = _write(tmp_path / "big.tif", 4096)
    ram = tmp_path / "ram"
    ram.mkdir()
    # mem_charged with cap == working -> budget 0 -> any non-empty src is skipped
    cfg = StageInCfg(dir=ram, mem_charged=True, working_gb=1, cap_gb=1)

    with staged(src, cfg) as p:
        assert p == src  # read in place
    assert list(ram.glob("macsima.*")) == []


def test_staged_falls_back_when_dir_unusable(tmp_path: Path) -> None:
    src = _write(tmp_path / "img.tif", 100)
    blocker = _write(tmp_path / "not_a_dir", 1)  # a file where a dir is expected
    cfg = StageInCfg(dir=blocker / "ram", mem_charged=False, working_gb=1)

    with staged(src, cfg) as p:
        assert p == src  # ensure_dir fails -> read in place, no raise


def test_staged_expands_env_vars(tmp_path: Path, monkeypatch) -> None:
    ram = tmp_path / "ram"
    ram.mkdir()
    monkeypatch.setenv("MACSIMA_TEST_RAMDIR", str(ram))
    src = _write(tmp_path / "img.tif", 128)
    cfg = StageInCfg(dir="$MACSIMA_TEST_RAMDIR", mem_charged=False, working_gb=1)

    with staged(src, cfg) as p:
        assert ram in p.parents
        assert p.read_bytes() == src.read_bytes()
