"""Tests for Exp 4011 GAP-4 feedback-vs-redraw powered paired control.

Spec refs: REQ-VERIFY-4011, SCENARIO-VERIFY-4011.
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import Any

import pytest

import experiment_4011_gap4_feedback_vs_redraw_v2 as exp


JsonDict = dict[str, Any]


def _write_json(path: Path, payload: JsonDict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_gzip_json(path: Path, payload: JsonDict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        json.dump(payload, handle)


def _entry(task: str) -> JsonDict:
    return {
        "task": task,
        "demos": [{"input": [[1]], "output": [[1]]}],
        "test_input": [[2]],
        "candidates": [],
    }


def _record(task: str, arm_a: bool, arm_b: bool) -> JsonDict:
    return {
        "task": task,
        "n_entries": 1,
        "arm_a_feedback": {"demo_perfect": arm_a},
        "arm_b_redraws": [{"demo_perfect": arm_b}],
        "arm_a_correct": arm_a,
        "arm_b_correct": arm_b,
        "arm_b_correct_sources": ["arm_b_redraw1"] if arm_b else [],
        "n_calls": 4,
        "codex_seconds": 10.0,
    }


def test_req_verify_4011_spec_anchor_exists() -> None:
    """REQ-VERIFY-4011: OpenSpec declares the powered paired control v2."""

    spec = Path("openspec/capabilities/verification/spec.md").read_text(encoding="utf-8")

    assert "REQ-VERIFY-4011" in spec
    assert "SCENARIO-VERIFY-4011" in spec
    assert "n_discordant_pairs>=10" in spec
    assert "achieved_power" in spec
    assert "complete: feedback_vs_redraw_underpowered_n<discordant>" in spec


def test_req_verify_4011_uses_eval_pool_task_stream() -> None:
    """REQ-VERIFY-4011: the powered run scales over unique eval-pool tasks."""

    pool = {
        "entries": [
            _entry("task_a"),
            _entry("task_b"),
            _entry("task_a"),
            _entry("task_c"),
        ]
    }

    assert exp.tasks_from_eval_pool(pool) == ["task_a", "task_b", "task_c"]


def test_req_verify_4011_exact_power_helpers_are_deterministic() -> None:
    """REQ-VERIFY-4011: power and MDE are exact-binomial quantities."""

    assert exp.exact_mcnemar_p(0, 0) == 1.0
    assert exp.exact_mcnemar_p(9, 1) == pytest.approx(0.021484375)
    assert exp._power_at_probability(0, 0.9) == 0.0
    assert exp.achieved_power(0) == 0.0

    target_effect = exp.target_effect_for_discordant_target()
    assert 0.4 < target_effect < 0.5
    assert exp.achieved_power(10) >= 0.8
    assert exp.achieved_power(3) < 0.5
    assert exp.min_detectable_effect(10) == pytest.approx(target_effect, abs=1e-4)
    assert exp.min_detectable_effect(0) == 1.0


def test_scenario_verify_4011_stops_at_powered_discordant_target(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SCENARIO-VERIFY-4011: mocked paired arms stop once 10 discordants exist."""

    tasks = [f"task_{idx:02d}" for idx in range(12)]
    pool_path = tmp_path / "results" / "arc3_gap4_arc2_eval_pool.json.gz"
    output_path = tmp_path / "results" / "experiment_4011_gap4_feedback_vs_redraw_v2.json"
    transcript_dir = tmp_path / "results" / "experiment_4011_gap4_feedback_vs_redraw_v2_transcripts"
    pilot_path = tmp_path / "results" / "experiment_4000_gap4_feedback_vs_redraw.json"

    _write_gzip_json(pool_path, {"entries": [_entry(task) for task in tasks]})
    _write_json(pilot_path, {"n_discordant_pairs": 3, "mcnemar_p": 1.0})

    outcomes = {
        task: (idx % 2 == 0, idx % 2 == 1)
        for idx, task in enumerate(tasks)
    }
    called: list[str] = []

    def fake_paired_task(
        task: str,
        entries: list[JsonDict],
        transcripts_dir: Path,
        challenges: JsonDict,
        solutions: JsonDict,
        iters: int,
        timeout: int,
    ) -> JsonDict:
        del entries, challenges, solutions
        assert transcripts_dir == transcript_dir
        assert iters == 3
        assert timeout == 600
        called.append(task)
        return _record(task, *outcomes[task])

    monkeypatch.setattr(exp, "_paired_task", fake_paired_task)
    monkeypatch.setattr(exp, "audit_transcripts", lambda paths: {"clean": True, "n_transcripts": len(paths), "violations": []})

    artifact = exp.run(
        root=tmp_path,
        pool_path=pool_path,
        pilot_artifact_path=pilot_path,
        output_path=output_path,
        transcripts_dir=transcript_dir,
        challenges_path=tmp_path / "missing_challenges.json",
        solutions_path=tmp_path / "missing_solutions.json",
        codex_available_override=True,
        workers=1,
    )

    assert output_path.exists()
    assert called == tasks[:10]
    assert artifact["same_run_interleaved"] is True
    assert artifact["pool_exhausted"] is False
    assert artifact["stop_reason"] == "discordant_target_met"
    assert artifact["pilot_exp4000"]["n_discordant_pairs"] == 3
    assert artifact["n_discordant_pairs"] == 10
    assert artifact["paired_contingency"] == {
        "a_correct_b_correct": 0,
        "a_correct_b_wrong": 5,
        "a_wrong_b_correct": 5,
        "a_wrong_b_wrong": 0,
    }
    assert artifact["mcnemar_p"] == 1.0
    assert artifact["achieved_power"] >= 0.8
    assert artifact["honest_verdict"] == "complete: feedback_no_better_than_redraw_powered_p1.0"
    exp.validate_artifact(artifact)


def test_scenario_verify_4011_exhausted_pool_underpowered_verdict(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SCENARIO-VERIFY-4011: exhausted pools below 10 discordants are labeled underpowered."""

    tasks = ["both", "a_only", "neither"]
    pool_path = tmp_path / "pool.json.gz"
    pilot_path = tmp_path / "pilot.json"
    _write_gzip_json(pool_path, {"entries": [_entry(task) for task in tasks]})
    _write_json(pilot_path, {"n_discordant_pairs": 3})

    outcomes = {"both": (True, True), "a_only": (True, False), "neither": (False, False)}

    monkeypatch.setattr(
        exp,
        "_paired_task",
        lambda task, entries, transcripts_dir, challenges, solutions, iters, timeout: _record(
            task, *outcomes[task]
        ),
    )
    monkeypatch.setattr(exp, "audit_transcripts", lambda paths: {"clean": True, "n_transcripts": 0, "violations": []})

    artifact = exp.run(
        pool_path=pool_path,
        pilot_artifact_path=pilot_path,
        output_path=tmp_path / "out.json",
        transcripts_dir=tmp_path / "tx",
        challenges_path=tmp_path / "missing_challenges.json",
        solutions_path=tmp_path / "missing_solutions.json",
        codex_available_override=True,
        workers=1,
    )

    assert artifact["pool_exhausted"] is True
    assert artifact["n_discordant_pairs"] == 1
    assert artifact["honest_verdict"] == "complete: feedback_vs_redraw_underpowered_n1"
    assert "FALSE_NEGATIVE_RISK" not in artifact["honest_verdict"]


def test_scenario_verify_4011_parallel_batches_and_pilot_fallbacks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SCENARIO-VERIFY-4011: parallel batches remain same-run and pilot reads stay honest."""

    assert exp.load_pilot_context(tmp_path / "missing_pilot.json") == {"available": False}
    corrupt_pilot = tmp_path / "corrupt_pilot.json"
    corrupt_pilot.write_text("{not-json", encoding="utf-8")
    corrupt_context = exp.load_pilot_context(corrupt_pilot)
    assert corrupt_context["available"] is False
    assert corrupt_context["error"] == "JSONDecodeError"
    assert exp._verdict(True, 0.03125, 10, False) == "success: feedback_beats_redraw_p0.03125"
    assert exp._verdict(False, 1.0, 2, False) == "blocked_discordant_target_unmet_n2"

    tasks = ["same", "a_only", "b_only"]
    pool_path = tmp_path / "pool.json.gz"
    pilot_path = tmp_path / "pilot.json"
    tx_dir = tmp_path / "tx"
    tx_dir.mkdir()
    (tx_dir / "stale.txt").write_text("stale", encoding="utf-8")
    _write_gzip_json(pool_path, {"entries": [_entry(task) for task in tasks]})
    _write_json(pilot_path, {"n_discordant_pairs": 3})
    outcomes = {"same": (True, True), "a_only": (True, False), "b_only": (False, True)}

    monkeypatch.setattr(
        exp,
        "_paired_task",
        lambda task, entries, transcripts_dir, challenges, solutions, iters, timeout: _record(
            task, *outcomes[task]
        ),
    )
    monkeypatch.setattr(exp, "audit_transcripts", lambda paths: {"clean": True, "n_transcripts": 0, "violations": []})

    artifact = exp.run(
        pool_path=pool_path,
        pilot_artifact_path=pilot_path,
        output_path=tmp_path / "out.json",
        transcripts_dir=tx_dir,
        challenges_path=tmp_path / "missing_challenges.json",
        solutions_path=tmp_path / "missing_solutions.json",
        codex_available_override=True,
        workers=2,
    )

    assert not (tx_dir / "stale.txt").exists()
    assert artifact["task_set"] == tasks
    assert artifact["pool_exhausted"] is True
    assert artifact["n_discordant_pairs"] == 2


def test_scenario_verify_4011_parallel_batch_stops_at_target(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SCENARIO-VERIFY-4011: parallel mode stops after a batch reaches target power."""

    tasks = [f"task_{idx:02d}" for idx in range(10)]
    pool_path = tmp_path / "pool.json.gz"
    pilot_path = tmp_path / "pilot.json"
    _write_gzip_json(pool_path, {"entries": [_entry(task) for task in tasks]})
    _write_json(pilot_path, {"n_discordant_pairs": 3})
    monkeypatch.setattr(
        exp,
        "_paired_task",
        lambda task, entries, transcripts_dir, challenges, solutions, iters, timeout: _record(
            task, True, False
        ),
    )
    monkeypatch.setattr(exp, "audit_transcripts", lambda paths: {"clean": True, "n_transcripts": 0, "violations": []})

    artifact = exp.run(
        pool_path=pool_path,
        pilot_artifact_path=pilot_path,
        output_path=tmp_path / "out.json",
        transcripts_dir=tmp_path / "tx",
        challenges_path=tmp_path / "missing_challenges.json",
        solutions_path=tmp_path / "missing_solutions.json",
        codex_available_override=True,
        workers=4,
    )

    assert artifact["pool_exhausted"] is False
    assert artifact["stop_reason"] == "discordant_target_met"
    assert artifact["n_discordant_pairs"] == 10


def test_req_verify_4011_blocks_and_validates_bare_schema(tmp_path: Path) -> None:
    """REQ-VERIFY-4011: blockers do not fabricate wins and schema fields stay bare."""

    pool_path = tmp_path / "pool.json.gz"
    _write_gzip_json(pool_path, {"entries": [_entry("task")]})

    blocked = exp.run(
        pool_path=pool_path,
        output_path=tmp_path / "blocked.json",
        codex_available_override=False,
    )

    assert blocked["honest_verdict"] == "blocked_codex_unavailable"
    assert (tmp_path / "blocked.json").exists()
    assert blocked["same_run_interleaved"] is False
    assert blocked["total_codex_calls"] == 0
    exp.validate_artifact(blocked)

    missing = dict(blocked)
    missing.pop("honest_verdict")
    with pytest.raises(ValueError, match="missing required"):
        exp.validate_artifact(missing)

    bad = dict(blocked, honest_verdict="maybe")
    with pytest.raises(ValueError, match="terminal prefix"):
        exp.validate_artifact(bad)

    bad = dict(blocked, honest_verdict="complete: feedback_no_better_than_redraw_p1.0")
    with pytest.raises(ValueError, match="powered"):
        exp.validate_artifact(bad)

    bad = dict(blocked, honest_verdict="complete: x_FALSE_NEGATIVE_RISK")
    with pytest.raises(ValueError, match="FALSE_NEGATIVE_RISK"):
        exp.validate_artifact(bad)

    bad = dict(blocked, same_run_interleaved="true")
    with pytest.raises(ValueError, match="same_run_interleaved"):
        exp.validate_artifact(bad)

    bad = dict(blocked, n_discordant_pairs=True)
    with pytest.raises(ValueError, match="n_discordant_pairs"):
        exp.validate_artifact(bad)

    bad = dict(blocked, achieved_power=True)
    with pytest.raises(ValueError, match="achieved_power"):
        exp.validate_artifact(bad)

    bad = dict(blocked, achieved_power=1.1)
    with pytest.raises(ValueError, match="achieved_power"):
        exp.validate_artifact(bad)

    bad = dict(blocked, inference_substrate=3)
    with pytest.raises(ValueError, match="inference_substrate"):
        exp.validate_artifact(bad)

    bad = dict(
        blocked,
        honest_verdict="complete: feedback_no_better_than_redraw_powered_p1.0",
        n_discordant_pairs=9,
    )
    with pytest.raises(ValueError, match="powered null"):
        exp.validate_artifact(bad)

    bad = dict(
        blocked,
        honest_verdict="complete: feedback_vs_redraw_underpowered_n10",
        n_discordant_pairs=10,
        pool_exhausted=True,
    )
    with pytest.raises(ValueError, match="underpowered"):
        exp.validate_artifact(bad)

    bad = dict(
        blocked,
        honest_verdict="complete: feedback_vs_redraw_underpowered_n1",
        n_discordant_pairs=1,
        pool_exhausted=False,
    )
    with pytest.raises(ValueError, match="pool_exhausted"):
        exp.validate_artifact(bad)

    bad = dict(blocked, feedback_beats_redraw=True)
    with pytest.raises(ValueError, match="success"):
        exp.validate_artifact(bad)
