"""Tests for scripts/experiment_629_interwhen_diagnostic.py — InterWhen Diagnostic Gate.

Coverage targets (targeted coverage of code added in this experiment only):
- load_corpus_pairs: missing files, valid files, mixed files
- run_monitor_on_set: all correct (fp=0), all incorrect detected, partial detection
- main: gate_open=True, gate_open=False, extended set included, artifact schema

Spec: REQ-VERIFY-132, SCENARIO-VERIFY-171, SCENARIO-VERIFY-172
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import scripts.experiment_629_interwhen_diagnostic as exp629
from scripts.experiment_629_interwhen_diagnostic import load_corpus_pairs, run_monitor_on_set
from carnot.pipeline.interwhen_monitor import InterWhenMonitor
from carnot.pipeline.symcode_verifier import SymCodeVerifier


# ---------------------------------------------------------------------------
# load_corpus_pairs
# ---------------------------------------------------------------------------


def test_load_corpus_pairs_skips_missing_file(tmp_path: Path) -> None:
    """A missing corpus file is skipped without raising; returns empty lists."""
    incorrect, correct = load_corpus_pairs([tmp_path / "nonexistent.json"])
    assert incorrect == []
    assert correct == []


def test_load_corpus_pairs_loads_incorrect_and_correct(tmp_path: Path) -> None:
    """Responses are split correctly by is_correct field."""
    data = [
        {"response": "A", "is_correct": False},
        {"response": "B", "is_correct": True},
        {"response": "C", "is_correct": False},
    ]
    p = tmp_path / "corpus.json"
    p.write_text(json.dumps(data))
    incorrect, correct = load_corpus_pairs([p])
    assert incorrect == ["A", "C"]
    assert correct == ["B"]


def test_load_corpus_pairs_treats_missing_is_correct_as_correct(tmp_path: Path) -> None:
    """Responses without is_correct default to correct (is_correct=True)."""
    data = [{"response": "X"}]
    p = tmp_path / "corpus.json"
    p.write_text(json.dumps(data))
    incorrect, correct = load_corpus_pairs([p])
    assert incorrect == []
    assert correct == ["X"]


def test_load_corpus_pairs_concatenates_multiple_files(tmp_path: Path) -> None:
    """Responses from multiple files are concatenated in order."""
    data1 = [{"response": "A", "is_correct": False}]
    data2 = [{"response": "B", "is_correct": False}, {"response": "C", "is_correct": True}]
    p1 = tmp_path / "a.json"
    p2 = tmp_path / "b.json"
    p1.write_text(json.dumps(data1))
    p2.write_text(json.dumps(data2))
    incorrect, correct = load_corpus_pairs([p1, p2])
    assert incorrect == ["A", "B"]
    assert correct == ["C"]


def test_load_corpus_pairs_second_file_missing_skips_gracefully(tmp_path: Path) -> None:
    """If only the second file is missing, the first file's data is returned."""
    data = [{"response": "A", "is_correct": False}]
    p1 = tmp_path / "present.json"
    p1.write_text(json.dumps(data))
    incorrect, correct = load_corpus_pairs([p1, tmp_path / "missing.json"])
    assert incorrect == ["A"]
    assert correct == []


# ---------------------------------------------------------------------------
# run_monitor_on_set
# ---------------------------------------------------------------------------


def _make_monitor_with_patch(
    monkeypatch: pytest.MonkeyPatch,
    detect_responses: set[str],
) -> InterWhenMonitor:
    """Build an InterWhenMonitor and class-patch any_violation to detect_responses set.

    run_monitor_on_set creates a fresh InterWhenMonitor internally, so we must
    patch the CLASS method rather than an instance method for the mock to take effect.
    """
    verifier = SymCodeVerifier(llm_caller=None)
    monitor = InterWhenMonitor(verifier)

    def _any_violation(self: InterWhenMonitor, response: str) -> bool:  # noqa: ANN001
        return response in detect_responses

    monkeypatch.setattr(InterWhenMonitor, "any_violation", _any_violation)
    return monitor


def test_run_monitor_on_set_all_detected(monkeypatch: pytest.MonkeyPatch) -> None:
    """run_monitor_on_set returns tp=3, fp=0 when all 3 incorrect are detected."""
    incorrect = ["wrong1", "wrong2", "wrong3"]
    correct = ["right1", "right2"]
    monitor = _make_monitor_with_patch(monkeypatch, {"wrong1", "wrong2", "wrong3"})
    tp, fp = run_monitor_on_set(monitor, incorrect, correct)
    assert tp == 3
    assert fp == 0


def test_run_monitor_on_set_none_detected(monkeypatch: pytest.MonkeyPatch) -> None:
    """run_monitor_on_set returns tp=0, fp=0 when nothing is detected."""
    incorrect = ["wrong1", "wrong2"]
    correct = ["right1"]
    monitor = _make_monitor_with_patch(monkeypatch, set())
    tp, fp = run_monitor_on_set(monitor, incorrect, correct)
    assert tp == 0
    assert fp == 0


def test_run_monitor_on_set_false_positives(monkeypatch: pytest.MonkeyPatch) -> None:
    """run_monitor_on_set counts fp correctly when correct responses trigger violations."""
    incorrect = ["wrong1"]
    correct = ["right1", "right2"]
    monitor = _make_monitor_with_patch(monkeypatch, {"right1"})
    tp, fp = run_monitor_on_set(monitor, incorrect, correct)
    assert tp == 0
    assert fp == 1


def test_run_monitor_on_set_partial_detection(monkeypatch: pytest.MonkeyPatch) -> None:
    """run_monitor_on_set handles partial detection: tp=1 of 3 incorrect."""
    incorrect = ["wrong1", "wrong2", "wrong3"]
    correct = ["right1"]
    monitor = _make_monitor_with_patch(monkeypatch, {"wrong2"})
    tp, fp = run_monitor_on_set(monitor, incorrect, correct)
    assert tp == 1
    assert fp == 0


def test_run_monitor_on_set_uses_fresh_monitor(monkeypatch: pytest.MonkeyPatch) -> None:
    """run_monitor_on_set creates a fresh InterWhenMonitor so prior state does not bleed.

    We poison the original monitor's violations_detected list and verify that
    run_monitor_on_set returns deterministic integer results unaffected by it.
    """
    verifier = SymCodeVerifier(llm_caller=None)
    monitor = InterWhenMonitor(verifier)
    from carnot.pipeline.interwhen_monitor import InterWhenViolation
    monitor.violations_detected.append(InterWhenViolation(0, "old", True, 0.5))
    # Patch to detect nothing so the return values are deterministic.
    monkeypatch.setattr(InterWhenMonitor, "any_violation", lambda self, r: False)
    tp, fp = run_monitor_on_set(monitor, ["a simple sentence"], ["a correct one"])
    assert tp == 0
    assert fp == 0


# ---------------------------------------------------------------------------
# main() integration — gate_open=False (below 0.20 threshold)
# ---------------------------------------------------------------------------


def _write_corpus(path: Path, n_incorrect: int, n_correct: int) -> None:
    """Write a minimal corpus JSON file with synthetic incorrect/correct pairs."""
    pairs = [
        {"response": f"wrong_{i}", "is_correct": False} for i in range(n_incorrect)
    ] + [
        {"response": f"right_{i}", "is_correct": True} for i in range(n_correct)
    ]
    path.write_text(json.dumps(pairs))


def test_main_gate_closed_when_recall_below_threshold(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """main() writes gate_open=False when interwhen_recall_primary < 0.20 (0 of 25 detected)."""
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    _write_corpus(results_dir / "live_pairs_578.json", n_incorrect=80, n_correct=20)

    monkeypatch.setattr(exp629, "_REPO_ROOT", tmp_path)
    monkeypatch.setenv("CARNOT_REPO_ROOT", str(tmp_path))
    monkeypatch.setenv("CARNOT_FORCE_LIVE", "0")
    # apply_env_autofix, assert_live_or_ci_skip, ExperimentTimeoutWatchdog are live-mode
    # guards; stub them out so the unit test runs cleanly without a GPU.
    monkeypatch.setattr(exp629, "apply_env_autofix", lambda: None)
    monkeypatch.setattr(exp629, "assert_live_or_ci_skip", lambda: None)
    monkeypatch.setattr(
        exp629,
        "ExperimentTimeoutWatchdog",
        lambda exp_id, timeout_minutes=40: None,
    )

    exp629.main()

    out = json.loads((results_dir / "experiment_629_interwhen_diagnostic.json").read_text())
    assert out["result_schema"] == "carnot.interwhen_diagnostic.v1"
    assert out["n_primary_incorrect"] == 25
    assert out["n_primary_correct"] == 10
    assert isinstance(out["interwhen_tp_primary"], int)
    assert isinstance(out["interwhen_fp_primary"], int)
    assert isinstance(out["interwhen_recall_primary"], float)
    assert isinstance(out["interwhen_fp_rate_primary"], float)
    assert out["prior_best_recall"] == pytest.approx(0.04, abs=1e-6)
    assert out["gate_open"] is False
    assert "Exp 630 GATED" in out["gate_note"]
    assert out["honest_verdict"] == "gate_closed_do_not_retry"
    assert out["retro_070_partial"] is False
    assert out["retro_070_resolved"] is False
    assert out["status"] == "success"


def test_main_gate_open_when_recall_meets_threshold(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """main() writes gate_open=True when run_monitor_on_set detects >= 5 of 25 incorrect."""
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    _write_corpus(results_dir / "live_pairs_578.json", n_incorrect=80, n_correct=20)

    monkeypatch.setattr(exp629, "_REPO_ROOT", tmp_path)
    monkeypatch.setenv("CARNOT_FORCE_LIVE", "0")
    monkeypatch.setattr(exp629, "apply_env_autofix", lambda: None)
    monkeypatch.setattr(exp629, "assert_live_or_ci_skip", lambda: None)
    monkeypatch.setattr(
        exp629,
        "ExperimentTimeoutWatchdog",
        lambda exp_id, timeout_minutes=40: None,
    )

    # Patch run_monitor_on_set to return 5 tp (exactly 0.20 recall) on primary,
    # and 25 tp on extended, so gate flips open.
    call_count = [0]

    def _fake_run_monitor(monitor, incorrect, correct):
        call_count[0] += 1
        if len(incorrect) == 25:  # primary set
            return 5, 0
        return 10, 0  # extended set

    monkeypatch.setattr(exp629, "run_monitor_on_set", _fake_run_monitor)

    exp629.main()

    out = json.loads((results_dir / "experiment_629_interwhen_diagnostic.json").read_text())
    assert out["gate_open"] is True
    assert "UNBLOCKED" in out["gate_note"]
    assert out["honest_verdict"] == "gate_open_vr_unblocked"
    assert out["retro_070_resolved"] is True
    assert out["retro_070_partial"] is True
    assert out["interwhen_recall_primary"] == pytest.approx(0.20, abs=1e-6)


def test_main_extended_set_none_when_insufficient_corpus(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """interwhen_recall_extended is None when fewer than 50 incorrect responses exist."""
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    # Only 30 incorrect — not enough for extended set (needs 50).
    _write_corpus(results_dir / "live_pairs_578.json", n_incorrect=30, n_correct=20)

    monkeypatch.setattr(exp629, "_REPO_ROOT", tmp_path)
    monkeypatch.setenv("CARNOT_FORCE_LIVE", "0")
    monkeypatch.setattr(exp629, "apply_env_autofix", lambda: None)
    monkeypatch.setattr(exp629, "assert_live_or_ci_skip", lambda: None)
    monkeypatch.setattr(
        exp629,
        "ExperimentTimeoutWatchdog",
        lambda exp_id, timeout_minutes=40: None,
    )

    exp629.main()

    out = json.loads((results_dir / "experiment_629_interwhen_diagnostic.json").read_text())
    assert out["interwhen_recall_extended"] is None


def test_main_extended_set_computed_when_50_incorrect_available(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """interwhen_recall_extended is a float when >= 50 incorrect responses exist."""
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    _write_corpus(results_dir / "live_pairs_578.json", n_incorrect=80, n_correct=20)

    monkeypatch.setattr(exp629, "_REPO_ROOT", tmp_path)
    monkeypatch.setenv("CARNOT_FORCE_LIVE", "0")
    monkeypatch.setattr(exp629, "apply_env_autofix", lambda: None)
    monkeypatch.setattr(exp629, "assert_live_or_ci_skip", lambda: None)
    monkeypatch.setattr(
        exp629,
        "ExperimentTimeoutWatchdog",
        lambda exp_id, timeout_minutes=40: None,
    )

    exp629.main()

    out = json.loads((results_dir / "experiment_629_interwhen_diagnostic.json").read_text())
    assert isinstance(out["interwhen_recall_extended"], float)


def test_main_required_schema_fields_present(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """All required schema fields are present in the artifact."""
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    _write_corpus(results_dir / "live_pairs_578.json", n_incorrect=80, n_correct=20)

    monkeypatch.setattr(exp629, "_REPO_ROOT", tmp_path)
    monkeypatch.setenv("CARNOT_FORCE_LIVE", "0")
    monkeypatch.setattr(exp629, "apply_env_autofix", lambda: None)
    monkeypatch.setattr(exp629, "assert_live_or_ci_skip", lambda: None)
    monkeypatch.setattr(
        exp629,
        "ExperimentTimeoutWatchdog",
        lambda exp_id, timeout_minutes=40: None,
    )

    exp629.main()

    out = json.loads((results_dir / "experiment_629_interwhen_diagnostic.json").read_text())
    required = [
        "experiment",
        "title",
        "run_date",
        "started_at",
        "finished_at",
        "duration_s",
        "status",
        "result_schema",
        "n_primary_incorrect",
        "n_primary_correct",
        "interwhen_tp_primary",
        "interwhen_fp_primary",
        "interwhen_recall_primary",
        "interwhen_fp_rate_primary",
        "interwhen_recall_extended",
        "prior_best_recall",
        "gate_open",
        "gate_note",
        "retro_070_partial",
        "retro_070_resolved",
        "honest_verdict",
    ]
    for field in required:
        assert field in out, f"Missing required schema field: {field}"

    assert out["experiment"] == 629
    assert out["result_schema"] == "carnot.interwhen_diagnostic.v1"
    assert isinstance(out["schema"], list)  # build_result auto-generates schema as sorted key list
