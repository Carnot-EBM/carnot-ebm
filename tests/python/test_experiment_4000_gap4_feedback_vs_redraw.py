"""Tests for Exp 4000 GAP-4 feedback-vs-redraw paired control.

Spec refs: REQ-VERIFY-4000, SCENARIO-VERIFY-4000.
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import Any

import pytest

import experiment_4000_gap4_feedback_vs_redraw as exp


JsonDict = dict[str, Any]


def _write_json(path: Path, payload: JsonDict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_gzip_json(path: Path, payload: JsonDict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        json.dump(payload, handle)


def _entry(task: str, demo_value: int, test_value: int) -> JsonDict:
    return {
        "task": task,
        "demos": [{"input": [[demo_value]], "output": [[demo_value]]}],
        "test_input": [[test_value]],
        "candidates": [],
    }


def _arc_gold(entries: list[JsonDict], gold_by_task: dict[str, int]) -> tuple[JsonDict, JsonDict]:
    challenges: JsonDict = {}
    solutions: JsonDict = {}
    for entry in entries:
        task = str(entry["task"])
        challenges[task] = {"test": [{"input": entry["test_input"]}]}
        solutions[task] = [[[gold_by_task[task]]]]
    return challenges, solutions


def _constant_code(demo_value: int, pred_value: int) -> str:
    return (
        "def transform(grid):\n"
        "    if int(grid[0, 0]) == %d:\n"
        "        return np.array([[%d]])\n"
        "    return np.array([[%d]])\n"
    ) % (demo_value, demo_value, pred_value)


def test_req_verify_4000_spec_anchor_exists() -> None:
    """REQ-VERIFY-4000: OpenSpec declares the paired mechanism control."""

    spec = Path("openspec/capabilities/verification/spec.md").read_text(encoding="utf-8")

    assert "REQ-VERIFY-4000" in spec
    assert "SCENARIO-VERIFY-4000" in spec
    assert "same_run_interleaved" in spec
    assert "mcnemar_p" in spec
    assert "blocked_eval_pool_unreadable" in spec


def test_req_verify_4000_selects_real_chain_feasible_tasks() -> None:
    """REQ-VERIFY-4000: the paired control uses the prior chain-feasible task set."""

    chain_artifact = json.loads(
        Path("results/arc3_gap4_arc2_chain_ensemble.json").read_text(encoding="utf-8")
    )

    assert exp.selected_tasks_from_chain_artifact(chain_artifact) == [
        "13e47133",
        "2b83f449",
        "2d0172a1",
        "446ef5d2",
        "58490d8a",
        "58f5dbd5",
        "6e453dd6",
        "6ffbe589",
        "7b80bb43",
        "9aaea919",
        "b10624e5",
        "d8e07eb2",
    ]


def test_req_verify_4000_exact_mcnemar_is_two_sided_binomial() -> None:
    """REQ-VERIFY-4000: paired significance is exact McNemar on discordant pairs."""

    assert exp.exact_mcnemar_p(a_only=0, b_only=0) == 1.0
    assert exp.exact_mcnemar_p(a_only=3, b_only=0) == pytest.approx(0.25)
    assert exp.exact_mcnemar_p(a_only=5, b_only=1) == pytest.approx(0.21875)


def test_scenario_verify_4000_mocked_same_run_interleaved_paired_control(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SCENARIO-VERIFY-4000: mocked A/B arms emit paired rates, McNemar, and provenance."""

    entries = [
        _entry("a_only", 0, 8),
        _entry("b_only", 2, 8),
        _entry("both", 3, 8),
        _entry("neither", 4, 8),
    ]
    pool_path = tmp_path / "results" / "arc3_gap4_arc2_eval_pool.json.gz"
    chain_path = tmp_path / "results" / "arc3_gap4_arc2_chain_ensemble.json"
    output_path = tmp_path / "results" / "experiment_4000_gap4_feedback_vs_redraw.json"
    transcript_dir = tmp_path / "results" / "experiment_4000_gap4_feedback_vs_redraw_transcripts"
    challenges_path = tmp_path / "arc-agi_evaluation2_challenges.json"
    solutions_path = tmp_path / "arc-agi_evaluation2_solutions.json"

    _write_gzip_json(pool_path, {"entries": entries})
    _write_json(chain_path, {"preregistration": {"tasks": [row["task"] for row in entries]}})
    challenges, solutions = _arc_gold(
        entries,
        {"a_only": 1, "b_only": 9, "both": 7, "neither": 6},
    )
    _write_json(challenges_path, challenges)
    _write_json(solutions_path, solutions)

    calls: list[str] = []

    def fake_ask_codex(prompt: str, timeout: int, transcript_path: str | None = None) -> tuple[str, float]:
        assert timeout == 600
        assert transcript_path is not None
        rel = Path(transcript_path).relative_to(transcript_dir)
        calls.append(str(rel))
        task = rel.parts[0]
        stem = rel.stem
        demo_value = int(entries[[row["task"] for row in entries].index(task)]["demos"][0]["input"][0][0])
        pred = 5
        if task == "a_only" and stem == "arm_a_feedback_iter0":
            code = "def transform(grid):\n    return np.array([[9]])\n"
        elif task == "a_only" and stem == "arm_a_feedback_iter1":
            assert "PREVIOUS function failed" in prompt
            code = _constant_code(demo_value, 1)
            pred = 1
        elif task == "b_only" and stem == "arm_b_redraw2_iter0":
            code = _constant_code(demo_value, 9)
            pred = 9
        elif task == "both" and (
            stem == "arm_a_feedback_iter0" or stem == "arm_b_redraw3_iter0"
        ):
            code = _constant_code(demo_value, 7)
            pred = 7
        else:
            code = _constant_code(demo_value, pred)
        return "```python\n" + code + "```\n", 1.25

    monkeypatch.setattr(exp, "ask_codex", fake_ask_codex)

    artifact = exp.run(
        root=tmp_path,
        pool_path=pool_path,
        chain_artifact_path=chain_path,
        output_path=output_path,
        transcripts_dir=transcript_dir,
        challenges_path=challenges_path,
        solutions_path=solutions_path,
        codex_available_override=True,
        workers=1,
    )

    assert output_path.exists()
    assert artifact["same_run_interleaved"] is True
    assert artifact["feedback_beats_redraw"] is False
    assert artifact["paired_contingency"] == {
        "a_correct_b_correct": 1,
        "a_correct_b_wrong": 1,
        "a_wrong_b_correct": 1,
        "a_wrong_b_wrong": 1,
    }
    assert artifact["n_discordant_pairs"] == 2
    assert artifact["mcnemar_p"] == 1.0
    assert artifact["arm_a_gold_rate"] == 0.5
    assert artifact["arm_b_gold_rate"] == 0.5
    assert artifact["total_codex_calls"] == 17
    assert artifact["total_codex_seconds"] == 21.25
    assert artifact["leak_clean"] is True
    assert artifact["honest_verdict"] == "complete: feedback_no_better_than_redraw_p1.0_FALSE_NEGATIVE_RISK"
    assert calls[:6] == [
        "a_only/arm_a_feedback_iter0.txt",
        "a_only/arm_b_redraw1_iter0.txt",
        "a_only/arm_a_feedback_iter1.txt",
        "a_only/arm_b_redraw2_iter0.txt",
        "a_only/arm_b_redraw3_iter0.txt",
        "b_only/arm_a_feedback_iter0.txt",
    ]
    exp.validate_artifact(artifact)


def test_scenario_verify_4000_blocks_without_codex_or_pool(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-4000: blocked preconditions do not fabricate paired results."""

    pool_path = tmp_path / "results" / "arc3_gap4_arc2_eval_pool.json.gz"
    _write_gzip_json(pool_path, {"entries": [_entry("n0", 1, 2)]})
    written_output = tmp_path / "results" / "blocked_codex_written.json"

    written_blocked = exp.run(
        root=tmp_path,
        pool_path=pool_path,
        output_path=written_output,
        codex_available_override=False,
    )

    assert written_output.exists()
    assert written_blocked["honest_verdict"] == "blocked_codex_unavailable"

    blocked_codex = exp.run(
        root=tmp_path,
        pool_path=pool_path,
        output_path=tmp_path / "results" / "blocked_codex.json",
        codex_available_override=False,
        write=False,
    )

    assert blocked_codex["honest_verdict"] == "blocked_codex_unavailable"
    assert blocked_codex["same_run_interleaved"] is False
    assert blocked_codex["feedback_beats_redraw"] is False
    assert blocked_codex["total_codex_calls"] == 0

    blocked_pool = exp.run(
        root=tmp_path,
        pool_path=tmp_path / "results" / "missing.json.gz",
        output_path=tmp_path / "results" / "blocked_pool.json",
        codex_available_override=True,
        write=False,
    )

    assert blocked_pool["honest_verdict"] == "blocked_eval_pool_unreadable"
    assert blocked_pool["mcnemar_p"] == 1.0
    assert blocked_pool["n_discordant_pairs"] == 0


def test_req_verify_4000_defensive_helpers_cover_fallback_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-VERIFY-4000: helper fallbacks preserve honest paired accounting."""

    assert exp.selected_tasks_from_chain_artifact(
        {"per_task": [{"task": "selected_b"}, {"task": "selected_a"}]}
    ) == ["selected_a", "selected_b"]
    assert exp._format_p(0.21875) == "0.21875"

    fallback_entry = {
        "task": "fallback",
        "test_input": [[8]],
        "candidates": [{"correct": True, "grid": [[4]]}],
    }
    assert exp.gold_for_entry(fallback_entry, {}, {}).tolist() == [[4]]
    assert exp.gold_for_entry(
        {"task": "missing", "test_input": [[9]], "candidates": []},
        {"missing": {"test": [{"input": [[1]]}]}},
        {"missing": [[[1]]]},
    ) is None
    assert exp._arm_task_correct({"demo_perfect": False, "predictions": []}, [], {}, {}) is False

    demos = [{"input": [[1]], "output": [[1]]}]

    def no_code(prompt: str, timeout: int, transcript_path: str | None = None) -> tuple[str, float]:
        del prompt, timeout, transcript_path
        return "no python block", 0.2

    monkeypatch.setattr(exp, "ask_codex", no_code)
    code, fn, fit, row = exp._call_and_grade("prompt", tmp_path / "no_code.txt", 600, 0, demos)
    assert (code, fn, fit, row["status"]) == (None, None, 0.0, "no_code")

    def unsafe(prompt: str, timeout: int, transcript_path: str | None = None) -> tuple[str, float]:
        del prompt, timeout, transcript_path
        return "```python\ndef transform(grid):\n    return type(grid)\n```\n", 0.3

    monkeypatch.setattr(exp, "ask_codex", unsafe)
    code, fn, fit, row = exp._call_and_grade("prompt", tmp_path / "unsafe.txt", 600, 1, demos)
    assert (code, fn, fit, row["status"]) == (None, None, 0.0, "unsafe_or_uncompilable")

    pool_path = tmp_path / "results" / "arc3_gap4_arc2_eval_pool.json.gz"
    chain_path = tmp_path / "results" / "arc3_gap4_arc2_chain_ensemble.json"
    _write_gzip_json(pool_path, {"entries": [_entry("present", 1, 2)]})
    _write_json(chain_path, {"preregistration": {"tasks": ["missing"]}})
    with pytest.raises(ValueError, match="missing from eval pool"):
        exp.run(
            root=tmp_path,
            pool_path=pool_path,
            chain_artifact_path=chain_path,
            output_path=tmp_path / "results" / "unused.json",
            codex_available_override=True,
            write=False,
        )


def test_req_verify_4000_validation_rejects_non_bare_schema_fields() -> None:
    """REQ-VERIFY-4000: required artifact fields stay bare scalars."""

    artifact = exp.blocked_artifact(
        "blocked_codex_unavailable",
        [{"resource": "codex", "available": False}, {"resource": "eval_pool", "available": True}],
        0.5,
    )
    exp.validate_artifact(artifact)

    missing = dict(artifact)
    missing.pop("honest_verdict")
    with pytest.raises(ValueError, match="missing required"):
        exp.validate_artifact(missing)

    bad = dict(artifact, honest_verdict="maybe")
    with pytest.raises(ValueError, match="terminal prefix"):
        exp.validate_artifact(bad)
    bad = dict(artifact, same_run_interleaved="true")
    with pytest.raises(ValueError, match="same_run_interleaved"):
        exp.validate_artifact(bad)
    bad = dict(artifact, n_discordant_pairs=True)
    with pytest.raises(ValueError, match="n_discordant_pairs"):
        exp.validate_artifact(bad)
    bad = dict(artifact, mcnemar_p=True)
    with pytest.raises(ValueError, match="mcnemar_p"):
        exp.validate_artifact(bad)
    bad = dict(artifact, arm_a_gold_rate=None)
    with pytest.raises(ValueError, match="arm_a_gold_rate"):
        exp.validate_artifact(bad)
    bad = dict(artifact, leak_clean="yes")
    with pytest.raises(ValueError, match="leak_clean"):
        exp.validate_artifact(bad)
    bad = dict(artifact, inference_substrate=3)
    with pytest.raises(ValueError, match="inference_substrate"):
        exp.validate_artifact(bad)
