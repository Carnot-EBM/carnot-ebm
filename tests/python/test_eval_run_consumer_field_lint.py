"""The eval-run consumer/producer field join (REQ-ARC-WMTE-6642).

SCENARIO-ARC-WMTE-6642-DECLARE: a consumer without EVAL_RUN_FIELDS_READ fails.
SCENARIO-ARC-WMTE-6642-JOIN: observed passes; wired-unobserved notices; neither fails.
SCENARIO-ARC-WMTE-6642-FAIL-CLOSED: no runs dir / no artifacts fails; populations print.

Origin: three consumers in three days ran against fields no artifact recorded
(trajectory_supervisor, generator_channels, induction_attempts provenance).
"""

from __future__ import annotations

import importlib.util
import json
import sys

import pytest
from pathlib import Path

_SPEC = importlib.util.spec_from_file_location(
    "eval_run_consumer_field_lint",
    Path(__file__).resolve().parents[2] / "scripts" / "eval_run_consumer_field_lint.py",
)
lint = importlib.util.module_from_spec(_SPEC)
sys.modules["eval_run_consumer_field_lint"] = lint
_SPEC.loader.exec_module(lint)

REPO = Path(__file__).resolve().parents[2]


def _fake_repo(tmp_path: Path, *, consumer_body: str, artifact: dict | None) -> Path:
    root = tmp_path / "repo"
    (root / "scripts").mkdir(parents=True)
    (root / "python" / "carnot" / "agentic").mkdir(parents=True)
    runs = root / "results" / "arc_leaderboard_eval_runs"
    runs.mkdir(parents=True)
    (root / "scripts" / "consumer.py").write_text(consumer_body)
    # The producer surface: the eval harness emits "wired_field" as a literal.
    (root / "scripts" / "arc_leaderboard_eval.py").write_text('ROW = {"wired_field": 1}\n')
    (root / "python" / "carnot" / "agentic" / "runtime.py").write_text("x = 1\n")
    if artifact is not None:
        (runs / "run.json").write_text(json.dumps(artifact))
    return root


def _run(root: Path):
    return lint.run_lint(
        root,
        root / "results" / "arc_leaderboard_eval_runs",
        root / "results" / "arc_leaderboard_eval.json",
        (root / "scripts" / "arc_leaderboard_eval.py", root / "python" / "carnot" / "agentic"),
    )


def test_consumer_without_declaration_fails(tmp_path: Path) -> None:
    """SCENARIO-ARC-WMTE-6642-DECLARE."""

    root = _fake_repo(
        tmp_path,
        consumer_body='d = "arc_leaderboard_eval_runs"\n',
        artifact={"per_game": []},
    )
    failures, _ = _run(root)
    assert any("EVAL_RUN_FIELDS_READ" in f for f in failures)


def test_join_observed_wired_and_missing(tmp_path: Path) -> None:
    """SCENARIO-ARC-WMTE-6642-JOIN: all three verdicts from one consumer."""

    body = (
        'd = "arc_leaderboard_eval_runs"\n'
        'EVAL_RUN_FIELDS_READ = ("observed_field", "wired_field", "ghost_field")\n'
    )
    root = _fake_repo(
        tmp_path,
        consumer_body=body,
        artifact={"per_game": [{"observed_field": 3}]},
    )
    failures, notices = _run(root)
    assert any("ghost_field" in f and "NO eval-run artifact" in f for f in failures)
    assert not any("observed_field" in f for f in failures)
    assert not any("wired_field" in f for f in failures)
    assert any("wired_field" in n and "wired" in n for n in notices)


def test_nested_keys_count_as_observed(tmp_path: Path) -> None:
    """SCENARIO-ARC-WMTE-6642-JOIN: containment is any-depth, matching real rows."""

    body = 'd = "arc_leaderboard_eval_runs"\nEVAL_RUN_FIELDS_READ = ("deep_field",)\n'
    root = _fake_repo(
        tmp_path,
        consumer_body=body,
        artifact={"per_game": [{"diag": {"attempts": [{"deep_field": 1}]}}]},
    )
    failures, _ = _run(root)
    assert failures == []


def test_fails_closed_without_artifacts(tmp_path: Path) -> None:
    """SCENARIO-ARC-WMTE-6642-FAIL-CLOSED, both halves."""

    body = 'd = "arc_leaderboard_eval_runs"\nEVAL_RUN_FIELDS_READ = ("f",)\n'
    root = _fake_repo(tmp_path, consumer_body=body, artifact=None)
    failures, _ = _run(root)
    assert any("join cannot run" in f for f in failures)

    import shutil

    shutil.rmtree(root / "results" / "arc_leaderboard_eval_runs")
    failures, _ = _run(root)
    assert any("runs directory missing" in f for f in failures)


def test_population_note_always_printed(tmp_path: Path) -> None:
    """SCENARIO-ARC-WMTE-6642-FAIL-CLOSED: a zero-findings run names its populations."""

    body = 'd = "arc_leaderboard_eval_runs"\nEVAL_RUN_FIELDS_READ = ("per_game",)\n'
    root = _fake_repo(tmp_path, consumer_body=body, artifact={"per_game": []})
    failures, notices = _run(root)
    assert failures == []
    assert any("population:" in n for n in notices)


def test_a_heartbeat_key_does_not_count_as_observed(tmp_path: Path) -> None:
    """SCENARIO-ARC-WMTE-7012-D (lint half): the REQ-ARC-WMTE-7010 heartbeat shares the runs
    directory. A key that exists only in `*.progress.json` is not a recorded field, so a
    consumer declaring it must still fail the join."""

    body = 'd = "arc_leaderboard_eval_runs"\nEVAL_RUN_FIELDS_READ = ("induction_in_flight",)\n'
    root = _fake_repo(tmp_path, consumer_body=body, artifact={"per_game": [{"game": "x"}]})
    runs = root / "results" / "arc_leaderboard_eval_runs"
    (runs / "x-1.progress.json").write_text(json.dumps({"induction_in_flight": None}))
    failures, _ = _run(root)
    assert failures, "a heartbeat-only key satisfied the observed join"
    assert any("induction_in_flight" in f for f in failures)


def test_real_repo_contract_holds(capsys) -> None:
    """The live checkout passes: every seeded declaration is emitted somewhere real.

    This is the wiring test — if a consumer is added without a declaration, or a
    declared field stops being emitted, this test goes RED with the suite.
    """

    assert lint.main([]) == 0
    assert "population:" in capsys.readouterr().out


@pytest.mark.parametrize("mode", ["env", "cli", "missing", "empty", "ghost", "unset", "empty_env"])
def test_worktree_evidence_selection(monkeypatch, tmp_path, capsys, mode):
    """SCENARIO-ARC-WMTE-6642-WORKTREE: exercise the same CLI the commit hook runs."""
    required = "ghost_field" if mode == "ghost" else "observed_field"
    root = _fake_repo(
        tmp_path,
        consumer_body=f'd = "arc_leaderboard_eval_runs"\nEVAL_RUN_FIELDS_READ = ("{required}",)\n',
        artifact={"observed_field": 1} if mode in ("unset", "empty_env") else None,
    )
    external = tmp_path / "external"
    if mode != "missing":
        external.mkdir()
        if mode != "empty":
            (external / "real-shape.json").write_text(
                json.dumps({"per_game": [{"observed_field": 1}]})
            )
    args = ["--repo-root", str(root)]
    monkeypatch.setenv("CARNOT_ARC_EVAL_RUNS_DIR", str(external))
    if mode == "cli":
        monkeypatch.setenv("CARNOT_ARC_EVAL_RUNS_DIR", str(tmp_path / "missing-env"))
        args += ["--runs-dir", str(external)]
    elif mode == "unset":
        monkeypatch.delenv("CARNOT_ARC_EVAL_RUNS_DIR")
    elif mode == "empty_env":
        monkeypatch.setenv("CARNOT_ARC_EVAL_RUNS_DIR", "")
    selected = (
        root / "results" / "arc_leaderboard_eval_runs"
        if mode in ("unset", "empty_env")
        else external
    )
    before = {p: p.read_bytes() for p in tmp_path.rglob("*.json")}
    assert lint.main(args) == (1 if mode in ("missing", "empty", "ghost") else 0)
    output = capsys.readouterr().out
    assert str(selected) in output
    if mode == "ghost":
        assert "NO producer source" in output
    assert before == {p: p.read_bytes() for p in tmp_path.rglob("*.json")}
