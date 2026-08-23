"""REQ-ARC-WMTE-6700: the induction-quality scorer must refuse non-live
populations and must report distinct counts.

Origin (2026-08-23): the scorer's default root held two clones of this repo, so
`rglob` swept the committed `results/` evidence tree twice. 1138 of 1248 files
were that tree, emitted by a generator retired 2026-07-28. Two operator-facing
rates were wrong for hours as a result.

Every test here builds its own tree under `tmp_path`. Nothing reads or writes
the real corpus, per the Test-Run Record Integrity Discipline.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

_SPEC = importlib.util.spec_from_file_location(
    "arc_induction_quality",
    Path(__file__).resolve().parents[2] / "scripts" / "arc_induction_quality.py",
)
assert _SPEC and _SPEC.loader
aiq = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(aiq)


# A model with a goal predicate that names no structure term, so the goal
# classification is "flat" and cannot be confused with the guard's own verdict.
MODEL_SRC = """
import numpy as np


def engine(grid, action, data):
    g = np.asarray(grid).copy()
    if action == 1:
        g[0, 0] = 1
    return g


def is_level_complete(grid):
    return bool(np.asarray(grid)[0, 0] == 1)
"""


def _write_model(path: Path, src: str = MODEL_SRC) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(src, encoding="utf-8")
    return path


# --- SCENARIO-ARC-WMTE-6700-1: a nested repo clone is excluded ---------------


def test_nested_repo_clone_is_excluded(tmp_path: Path) -> None:
    (tmp_path / "clone" / ".git").mkdir(parents=True)
    _write_model(tmp_path / "clone" / "e3" / "g" / "world_model.py")

    kept, excluded = aiq.find_models([tmp_path])

    assert kept == []
    assert excluded == {"nested_repo_clone": 1}


def test_clone_is_excluded_even_when_git_is_a_file(tmp_path: Path) -> None:
    # A git worktree or submodule has `.git` as a FILE, not a directory. Keying
    # the guard on is_dir() would let a worktree through, and the abpin/headwt
    # pair that caused the incident were exactly two checkouts of one repo.
    (tmp_path / "wt").mkdir()
    (tmp_path / "wt" / ".git").write_text("gitdir: /elsewhere\n", encoding="utf-8")
    _write_model(tmp_path / "wt" / "e3" / "g" / "world_model.py")

    kept, excluded = aiq.find_models([tmp_path])

    assert kept == []
    assert excluded == {"nested_repo_clone": 1}


# --- SCENARIO-ARC-WMTE-6700-2: a committed results tree is excluded ----------


def test_committed_results_tree_is_excluded(tmp_path: Path) -> None:
    _write_model(tmp_path / "run" / "results" / "arc_e3" / "g" / "world_model.py")

    kept, excluded = aiq.find_models([tmp_path])

    assert kept == []
    assert excluded == {"committed_results_tree": 1}


def test_results_must_be_a_whole_path_part(tmp_path: Path) -> None:
    # `my_results_run` is not `results`. A substring test here would silently
    # discard live runs, which is the opposite failure and just as bad.
    _write_model(tmp_path / "my_results_run" / "e3" / "g" / "world_model.py")

    kept, excluded = aiq.find_models([tmp_path])

    assert len(kept) == 1
    assert excluded == {}


# --- SCENARIO-ARC-WMTE-6700-3: a live run directory is kept -----------------


def test_live_run_directory_is_kept(tmp_path: Path) -> None:
    model = _write_model(tmp_path / "run" / "e3" / "g" / "world_model.py")

    kept, excluded = aiq.find_models([tmp_path])

    assert kept == [model]
    assert excluded == {}


def test_archived_attempts_are_still_swept(tmp_path: Path) -> None:
    # REQ-ARC-WMTE-6690's attempt population must survive the new guard.
    attempt = _write_model(tmp_path / "run" / "e3" / "g" / "attempts" / "wm_x__abc.py")

    kept, excluded = aiq.find_models([tmp_path])

    assert attempt in kept
    assert excluded == {}


# --- SCENARIO-ARC-WMTE-6700-4: opt-in restores the old population -----------


def test_include_non_live_scores_excluded_paths(tmp_path: Path) -> None:
    (tmp_path / "clone" / ".git").mkdir(parents=True)
    model = _write_model(tmp_path / "clone" / "e3" / "g" / "world_model.py")

    kept, excluded = aiq.find_models([tmp_path], include_non_live=True)

    assert kept == [model]
    assert excluded == {}


def test_include_non_live_says_the_rates_are_not_live(tmp_path: Path, capsys) -> None:
    (tmp_path / "clone" / ".git").mkdir(parents=True)
    _write_model(tmp_path / "clone" / "e3" / "g" / "world_model.py")

    rc = aiq.main(["--roots", str(tmp_path), "--include-non-live"])
    out = capsys.readouterr().out

    assert rc == 0
    assert "NOT live-generator rates" in out


# --- SCENARIO-ARC-WMTE-6700-5: an all-excluded sweep fails closed -----------


def test_all_excluded_exits_non_zero_and_names_the_class(tmp_path: Path, capsys) -> None:
    _write_model(tmp_path / "run" / "results" / "g" / "world_model.py")

    rc = aiq.main(["--roots", str(tmp_path)])
    out = capsys.readouterr().out

    assert rc == 1
    assert "no LIVE world_model.py found" in out
    assert "committed_results_tree: 1" in out


def test_exclusions_are_named_in_a_mixed_sweep(tmp_path: Path, capsys) -> None:
    # The guard must never silently filter: a live model alongside an excluded
    # one still has to say what was dropped.
    _write_model(tmp_path / "run" / "e3" / "g" / "world_model.py")
    _write_model(tmp_path / "run" / "results" / "h" / "world_model.py")

    rc = aiq.main(["--roots", str(tmp_path)])
    out = capsys.readouterr().out

    assert rc == 0
    assert "EXCLUDED committed_results_tree: 1" in out


# --- SCENARIO-ARC-WMTE-6700-6: duplication is visible in the counts ---------


def test_duplicate_models_report_distinct_counts(tmp_path: Path, capsys) -> None:
    _write_model(tmp_path / "runA" / "e3" / "g" / "world_model.py")
    _write_model(tmp_path / "runB" / "e3" / "g" / "world_model.py")

    rc = aiq.main(["--roots", str(tmp_path)])
    out = capsys.readouterr().out

    assert rc == 0
    assert "2 model(s)" in out
    assert "distinct: 1 file(s) / 1 goal predicate(s) of 2 scored" in out


def test_distinct_predicate_count_ignores_comment_only_differences(tmp_path: Path) -> None:
    # Two files differing only outside the predicate are two distinct FILES but
    # one distinct PREDICATE. That is the count the structural rate is about.
    _write_model(tmp_path / "runA" / "e3" / "g" / "world_model.py")
    _write_model(
        tmp_path / "runB" / "e3" / "g" / "world_model.py",
        MODEL_SRC + "\n# a trailing comment outside the predicate\n",
    )

    kept, _ = aiq.find_models([tmp_path])
    scored = [aiq.score_model(p) for p in kept]

    assert len({s["file_sha16"] for s in scored}) == 2
    assert len({s["goal_predicate_sha16"] for s in scored}) == 1


# --- classify_path directly -------------------------------------------------


def test_classify_path_outside_root_fails_closed(tmp_path: Path) -> None:
    outside = _write_model(tmp_path / "outside" / "world_model.py")
    root = tmp_path / "root"
    root.mkdir()

    assert aiq.classify_path(outside, root) == "outside_root"
