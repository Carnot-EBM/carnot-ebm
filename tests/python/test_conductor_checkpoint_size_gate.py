"""REQ-INFRA-7092: the interrupted-run checkpoint applies the file-size gate.

INCIDENT 2026-10-02. The conductor's size gate lives in `_stage_all_except_claimed`. The
interrupted-run checkpoint in `research_step` stages files one at a time with
`git add -- <file>` and never called it, so five files of 147-335 MB were committed three days
after the cap dropped to 50 MB. GitHub hard-rejects over 100 MB and fell 68 commits behind gitea.

The behavior tests use a tiny threshold and sparse files, so nothing large is written. The wiring
test reads the source, because the bug was a MISSING call: a filter that works but is not called
protects nothing (see test_conductor_oversized_file_gate.py for the main-path gate).
"""

from __future__ import annotations

import ast
import logging
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))

import research_conductor as rc  # noqa: E402

CAP = 1000


def _make(tmp_path: Path, name: str, size: int) -> str:
    """Create a sparse file of `size` bytes under tmp_path and return its repo-relative name."""
    with open(tmp_path / name, "wb") as handle:
        handle.truncate(size)
    return name


@pytest.fixture
def root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(rc, "PROJECT_ROOT", tmp_path)
    return tmp_path


def test_an_oversized_path_is_dropped_and_the_small_one_kept(root: Path) -> None:
    """SCENARIO-INFRA-7092-DROP."""
    small = _make(root, "small.json", 10)
    big = _make(root, "big.json", CAP + 1)
    assert rc._drop_oversized_paths([small, big], CAP) == [small]


def test_a_file_exactly_at_the_cap_is_kept(root: Path) -> None:
    """The rule is strictly greater-than, same as the main gate."""
    at_cap = _make(root, "at_cap.json", CAP)
    assert rc._drop_oversized_paths([at_cap], CAP) == [at_cap]


def test_a_drop_is_logged_by_name_with_its_size(
    root: Path, caplog: pytest.LogCaptureFixture
) -> None:
    big = _make(root, "huge.json", CAP + 500)
    with caplog.at_level(logging.WARNING):
        rc._drop_oversized_paths([big], CAP)
    assert any(
        "BLOCKED_OVERSIZED_FILE" in r.message and "huge.json" in r.message for r in caplog.records
    )
    assert any(str(CAP + 500) in r.getMessage() for r in caplog.records)


def test_a_dropped_file_stays_on_disk_untouched(root: Path) -> None:
    """Nothing is lost: the file is only kept out of the commit."""
    big = _make(root, "keepme.bin", CAP + 1)
    rc._drop_oversized_paths([big], CAP)
    assert (root / big).exists()
    assert (root / big).stat().st_size == CAP + 1


def test_a_vanished_path_is_kept_not_crashed_on(root: Path) -> None:
    """A path deleted between listing and stat has no size, so it is never oversized."""
    assert rc._drop_oversized_paths(["gone.json"], CAP) == ["gone.json"]


def test_order_of_the_kept_paths_is_preserved(root: Path) -> None:
    a = _make(root, "a.json", 1)
    b = _make(root, "b.json", CAP + 1)
    c = _make(root, "c.json", 2)
    assert rc._drop_oversized_paths([c, b, a], CAP) == [c, a]


def test_a_filter_error_returns_every_path_unchanged(
    root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SCENARIO-INFRA-7092-FAILOPEN: refusing to checkpoint is the failure the gate must avoid."""

    def boom(*args: object, **kwargs: object) -> list[str]:
        raise RuntimeError("unexpected")

    monkeypatch.setattr(rc, "oversized_staged_files", boom)
    paths = ["a.json", "b.json"]
    assert rc._drop_oversized_paths(paths, CAP) == paths


def test_the_default_threshold_is_the_shared_cap(root: Path) -> None:
    """The filter uses the same cap as the main gate, not a second number that can drift."""
    over = _make(root, "over.bin", rc.OVERSIZED_FILE_THRESHOLD_BYTES + 1)
    assert rc._drop_oversized_paths([over]) == []


def _checkpoint_function() -> ast.FunctionDef:
    tree = ast.parse((REPO / "scripts" / "research_conductor.py").read_text(encoding="utf-8"))
    marker = "preserve uncommitted work from interrupted run"
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            if any(
                isinstance(n, ast.Constant) and isinstance(n.value, str) and marker in n.value
                for n in ast.walk(node)
            ):
                return node
    raise AssertionError("interrupted-run checkpoint function not found")


def test_the_checkpoint_calls_the_filter_before_its_first_git_add() -> None:
    """SCENARIO-INFRA-7092-WIRED: the bug was a missing call, so check the call site itself."""
    func = _checkpoint_function()
    filter_lines = [
        n.lineno
        for n in ast.walk(func)
        if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Name)
        and n.func.id == "_drop_oversized_paths"
    ]
    add_lines = [
        n.lineno
        for n in ast.walk(func)
        if isinstance(n, ast.Call)
        and n.args
        and isinstance(n.args[0], ast.List)
        and [e.value for e in n.args[0].elts if isinstance(e, ast.Constant)][:2] == ["git", "add"]
    ]
    assert filter_lines, "the checkpoint never calls _drop_oversized_paths"
    assert add_lines, "no git add found in the checkpoint function"
    assert min(filter_lines) < min(add_lines)
