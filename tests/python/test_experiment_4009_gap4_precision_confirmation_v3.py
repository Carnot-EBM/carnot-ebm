"""Tests for Exp 4009 GAP-4 precision confirmation v3.

Spec refs: REQ-VERIFY-4009, SCENARIO-VERIFY-4009.
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import Any

import pytest

import experiment_4009_gap4_precision_confirmation_v3 as exp


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


def test_req_verify_4009_spec_anchor_exists() -> None:
    """REQ-VERIFY-4009: OpenSpec declares the v3 execution-floor contract."""

    spec = Path("openspec/capabilities/verification/spec.md").read_text(encoding="utf-8")

    assert "REQ-VERIFY-4009" in spec
    assert "SCENARIO-VERIFY-4009" in spec
    assert "execution_floor_met" in spec
    assert "pending_execution" in spec
    assert "n_agreement_events>=19" in spec


def test_req_verify_4009_real_pool_selects_new_clean_nonprior_tasks() -> None:
    """REQ-VERIFY-4009: the committed ARC-2 task set excludes prior and reuse tasks."""

    pool = exp.load_eval_pool(Path("results/arc3_gap4_arc2_eval_pool.json.gz"))
    chain_artifact = exp.load_json(Path("results/arc3_gap4_arc2_chain_ensemble.json"))
    tasks = exp.clean_new_tasks(pool["entries"], chain_artifact)
    entries_by_task = exp.group_entries_by_task(pool["entries"])

    assert tasks == [
        "21897d95",
        "269e22fb",
        "28a6681f",
        "2c181942",
        "3a25b0d8",
        "3dc255db",
        "a6f40cea",
        "b9e38dc0",
        "dd6b8c4b",
    ]
    assert "aa4ec2a5" not in tasks
    assert "16b78196" not in tasks
    assert sum(len(entries_by_task[task]) for task in tasks) == 13


def test_req_verify_4009_primary_gate_and_verdicts() -> None:
    """REQ-VERIFY-4009: the primary gate and terminal verdicts match the preregistration."""

    assert exp.primary_gate_passed(14, 19) is True
    assert exp.primary_gate_passed(13, 19) is False
    assert exp.primary_gate_passed(14, 18) is False
    assert exp.execution_floor_met(1, 1) is True
    assert exp.execution_floor_met(0, 1) is False
    assert exp.execution_floor_met(1, 0) is False

    assert exp.verdict_for(14, 19, True) == "success: gap4_precision_confirmed_14of19_gold"
    assert (
        exp.verdict_for(13, 19, False)
        == "complete: gap4_agreement_confidence_label_only_13of19"
    )


def test_scenario_verify_4009_preregisters_then_executes_and_scores_agreement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SCENARIO-VERIFY-4009: mocked k=3 fresh chains meet the execution floor."""

    entries = [
        _entry("gold_agree", 0, 8),
        _entry("wrong_majority", 2, 8),
        _entry("no_agreement", 3, 8),
    ]
    pool_path = tmp_path / "results" / "arc3_gap4_arc2_eval_pool.json.gz"
    chain_path = tmp_path / "results" / "arc3_gap4_arc2_chain_ensemble.json"
    output_path = tmp_path / "results" / "experiment_4009_gap4_precision_confirmation_v3.json"
    transcript_dir = tmp_path / "results" / "experiment_4009_gap4_precision_transcripts"
    challenges_path = tmp_path / "arc-agi_evaluation2_challenges.json"
    solutions_path = tmp_path / "arc-agi_evaluation2_solutions.json"

    _write_gzip_json(pool_path, {"entries": entries})
    _write_json(chain_path, {"preregistration": {"tasks": ["prior"]}})
    challenges, solutions = _arc_gold(
        entries,
        {"gold_agree": 1, "wrong_majority": 9, "no_agreement": 2},
    )
    _write_json(challenges_path, challenges)
    _write_json(solutions_path, solutions)
    transcript_dir.mkdir(parents=True)
    (transcript_dir / "stale.txt").write_text("old transcript", encoding="utf-8")

    arm_preds = {
        "gold_agree": [1, 1, 1],
        "wrong_majority": [5, 5, 6],
        "no_agreement": [2, 3, 4],
    }
    seen_prereg: list[JsonDict] = []

    def fake_induce(
        task_name: str,
        demos: list[JsonDict],
        test_input: list[list[int]],
        iters: int,
        timeout: int,
        transcripts_dir: str,
    ) -> JsonDict:
        del test_input, iters, timeout
        assert output_path.exists()
        prereg = json.loads(output_path.read_text(encoding="utf-8"))
        assert prereg["protocol_preregistered"] is True
        assert prereg["execution_floor_met"] is False
        assert prereg["total_codex_calls"] == 0
        assert "pending_execution" not in prereg["honest_verdict"]
        seen_prereg.append(prereg)

        arm_idx = int(Path(transcripts_dir).name.replace("arm", "")) - 1
        Path(transcripts_dir).mkdir(parents=True, exist_ok=True)
        (Path(transcripts_dir) / f"{task_name}_iter0.txt").write_text(
            "===== PROMPT =====\nDemo pairs and test input only\n===== RAW OUTPUT =====\n"
            "```python\n# clean transcript\n```\n",
            encoding="utf-8",
        )
        pred = arm_preds[task_name][arm_idx]
        code = _constant_code(int(demos[0]["input"][0][0]), pred)
        return {
            "task": task_name,
            "demo_fit": 1.0,
            "demo_perfect": True,
            "pred_hash": "unused",
            "pred_grid": [[pred]],
            "n_calls": 1,
            "codex_seconds": 1.0,
            "history": [{"iter": 0, "status": "graded", "demo_fit": 1.0, "codex_s": 1.0}],
            "code": code,
        }

    monkeypatch.setattr(exp.v2, "induce_program", fake_induce)

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

    assert seen_prereg
    assert artifact["protocol_preregistered"] is True
    assert artifact["execution_floor_met"] is True
    assert artifact["n_agreement_events"] == 2
    assert artifact["n_gold_given_agreement"] == 1
    assert artifact["primary_gate_passed"] is False
    assert artifact["agreement_is_selector_not_label"] is False
    assert artifact["fresh_arm_base_rate"] == pytest.approx(0.4444)
    assert artifact["precision_vs_fresharm_base"] == pytest.approx(0.0556)
    assert artifact["total_codex_calls"] == 9
    assert artifact["total_codex_seconds"] == 9.0
    assert artifact["leak_clean"] is True
    assert artifact["draw_stop_reason"] == "pool_exhausted"
    assert artifact["honest_verdict"] == "complete: gap4_agreement_confidence_label_only_1of2"
    assert artifact["missing_verifier_gaps"] == [
        {
            "task": "wrong_majority",
            "input_idx": 0,
            "failure_mode": "agreement_wrong_but_unanimity_tripwire_abstains",
            "missing_discriminator": "GAP-5 demo-underdetermination sibling-input tripwire",
        }
    ]
    exp.validate_artifact(artifact)


def test_scenario_verify_4009_blocks_without_codex_or_pool(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-4009: blocked preconditions do not fabricate execution."""

    pool_path = tmp_path / "results" / "arc3_gap4_arc2_eval_pool.json.gz"
    _write_gzip_json(pool_path, {"entries": [_entry("n0", 1, 2)]})

    written_blocked = exp.run(
        root=tmp_path,
        pool_path=pool_path,
        output_path=tmp_path / "results" / "written_blocked_codex.json",
        codex_available_override=False,
    )

    assert written_blocked["honest_verdict"] == "blocked_codex_unavailable"

    blocked_codex = exp.run(
        root=tmp_path,
        pool_path=pool_path,
        output_path=tmp_path / "results" / "blocked_codex.json",
        codex_available_override=False,
        write=False,
    )

    assert blocked_codex["honest_verdict"] == "blocked_codex_unavailable"
    assert blocked_codex["execution_floor_met"] is False
    assert blocked_codex["protocol_preregistered"] is False
    assert blocked_codex["total_codex_calls"] == 0

    blocked_pool = exp.run(
        root=tmp_path,
        pool_path=tmp_path / "results" / "missing.json.gz",
        output_path=tmp_path / "results" / "blocked_pool.json",
        codex_available_override=True,
        write=False,
    )

    assert blocked_pool["honest_verdict"] == "blocked_eval_pool_unreadable"
    assert blocked_pool["execution_floor_met"] is False
    assert blocked_pool["n_gold_given_agreement"] == 0


def test_scenario_verify_4009_execution_floor_blocks_no_agreement_completion() -> None:
    """SCENARIO-VERIFY-4009: real calls with zero agreement events are blocked, not complete."""

    entries = [_entry("split_vote", 0, 8)]
    arms = [
        {
            "source": "fresh_chain1",
            "demo_perfect": True,
            "predictions": [{"pred_hash": exp.grid_hash([[1]]), "pred_grid": [[1]]}],
        },
        {
            "source": "fresh_chain2",
            "demo_perfect": True,
            "predictions": [{"pred_hash": exp.grid_hash([[2]]), "pred_grid": [[2]]}],
        },
        {
            "source": "fresh_chain3",
            "demo_perfect": True,
            "predictions": [{"pred_hash": exp.grid_hash([[3]]), "pred_grid": [[3]]}],
        },
    ]
    prereg = exp.preregistered_artifact(
        tasks=["split_vote"],
        entries_by_task={"split_vote": entries},
        preconditions=[{"resource": "codex", "available": True}, {"resource": "eval_pool", "available": True}],
        started_s=0.0,
        now_s=0.1,
        n_fresh=3,
        timeout=600,
    )
    artifact = exp.build_complete_artifact(
        records=[{"task": "split_vote", "arms": arms, "n_calls": 3, "codex_seconds": 3.0}],
        entries_by_task={"split_vote": entries},
        preregistration=prereg["preregistration"],
        preconditions=prereg["preconditions_checked"],
        transcript_audit={"clean": True, "n_transcripts": 3, "violations": []},
        challenges={"split_vote": {"test": [{"input": [[8]]}]}},
        solutions={"split_vote": [[[9]]]},
        started_s=0.0,
        now_s=1.0,
        pool_exhausted=True,
    )

    assert artifact["total_codex_calls"] == 3
    assert artifact["n_agreement_events"] == 0
    assert artifact["execution_floor_met"] is False
    assert artifact["honest_verdict"] == "blocked_execution_floor_unmet"
    exp.validate_artifact(artifact)


def test_scenario_verify_4009_stops_sequential_when_target_met(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SCENARIO-VERIFY-4009: sequential drawing stops once the agreement target is met."""

    entries = [_entry("t1", 0, 8), _entry("t2", 1, 8)]
    pool_path = tmp_path / "results" / "pool.json.gz"
    chain_path = tmp_path / "results" / "chain.json"
    challenges_path, solutions_path = tmp_path / "challenges.json", tmp_path / "solutions.json"
    _write_gzip_json(pool_path, {"entries": entries})
    _write_json(chain_path, {"preregistration": {"tasks": []}})
    challenges, solutions = _arc_gold(entries, {"t1": 1, "t2": 1})
    _write_json(challenges_path, challenges)
    _write_json(solutions_path, solutions)
    monkeypatch.setattr(exp, "AGREEMENT_EVENT_TARGET", 1)

    def fake_induce(
        task_name: str,
        demos: list[JsonDict],
        test_input: list[list[int]],
        iters: int,
        timeout: int,
        transcripts_dir: str,
    ) -> JsonDict:
        del test_input, iters, timeout
        transcript_path = Path(transcripts_dir)
        transcript_path.mkdir(parents=True, exist_ok=True)
        (transcript_path / f"{task_name}_iter0.txt").write_text(
            "===== PROMPT =====\nDemo pairs and test input only\n===== RAW OUTPUT =====\n",
            encoding="utf-8",
        )
        return {
            "n_calls": 1,
            "codex_seconds": 1.0,
            "history": [{"iter": 0, "status": "graded", "demo_fit": 1.0, "codex_s": 1.0}],
            "code": _constant_code(int(demos[0]["input"][0][0]), 1),
        }

    monkeypatch.setattr(exp.v2, "induce_program", fake_induce)

    artifact = exp.run(
        pool_path=pool_path,
        chain_artifact_path=chain_path,
        output_path=tmp_path / "results" / "out.json",
        transcripts_dir=tmp_path / "results" / "transcripts",
        challenges_path=challenges_path,
        solutions_path=solutions_path,
        workers=1,
        codex_available_override=True,
    )

    assert len(artifact["per_task"]) == 1
    assert artifact["draw_stop_reason"] == "powered_target_met"
    assert artifact["execution_floor_met"] is True


def test_scenario_verify_4009_parallel_batch_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SCENARIO-VERIFY-4009: the batched worker path records target completion."""

    entries = [_entry("t1", 0, 8), _entry("t2", 1, 8)]
    pool_path = tmp_path / "results" / "pool.json.gz"
    chain_path = tmp_path / "results" / "chain.json"
    challenges_path, solutions_path = tmp_path / "challenges.json", tmp_path / "solutions.json"
    _write_gzip_json(pool_path, {"entries": entries})
    _write_json(chain_path, {"preregistration": {"tasks": []}})
    challenges, solutions = _arc_gold(entries, {"t1": 1, "t2": 1})
    _write_json(challenges_path, challenges)
    _write_json(solutions_path, solutions)
    monkeypatch.setattr(exp, "AGREEMENT_EVENT_TARGET", 1)

    def fake_induce(
        task_name: str,
        demos: list[JsonDict],
        test_input: list[list[int]],
        iters: int,
        timeout: int,
        transcripts_dir: str,
    ) -> JsonDict:
        del test_input, iters, timeout
        transcript_path = Path(transcripts_dir)
        transcript_path.mkdir(parents=True, exist_ok=True)
        (transcript_path / f"{task_name}_iter0.txt").write_text(
            "===== PROMPT =====\nDemo pairs and test input only\n===== RAW OUTPUT =====\n",
            encoding="utf-8",
        )
        return {
            "n_calls": 1,
            "codex_seconds": 1.0,
            "history": [{"iter": 0, "status": "graded", "demo_fit": 1.0, "codex_s": 1.0}],
            "code": _constant_code(int(demos[0]["input"][0][0]), 1),
        }

    monkeypatch.setattr(exp.v2, "induce_program", fake_induce)

    artifact = exp.run(
        pool_path=pool_path,
        chain_artifact_path=chain_path,
        output_path=tmp_path / "results" / "out.json",
        transcripts_dir=tmp_path / "results" / "transcripts",
        challenges_path=challenges_path,
        solutions_path=solutions_path,
        workers=2,
        codex_available_override=True,
    )

    assert len(artifact["per_task"]) == 2
    assert artifact["draw_stop_reason"] == "powered_target_met"
    assert artifact["execution_floor_met"] is True


def test_req_verify_4009_validation_rejects_pending_and_non_bare_fields() -> None:
    """REQ-VERIFY-4009: schema validation rejects pending or zero-call complete artifacts."""

    artifact = exp.blocked_artifact(
        "blocked_codex_unavailable",
        [{"resource": "codex", "available": False}, {"resource": "eval_pool", "available": True}],
        0.5,
    )
    exp.validate_artifact(artifact)

    pending = dict(artifact, honest_verdict="complete: protocol_preregistered_pending_execution")
    with pytest.raises(ValueError, match="pending_execution"):
        exp.validate_artifact(pending)

    missing = dict(artifact)
    missing.pop("honest_verdict")
    with pytest.raises(ValueError, match="missing required"):
        exp.validate_artifact(missing)

    bad_prefix = dict(artifact, honest_verdict="maybe")
    with pytest.raises(ValueError, match="terminal prefix"):
        exp.validate_artifact(bad_prefix)

    complete_zero = dict(artifact, honest_verdict="complete: gap4_agreement_confidence_label_only_0of0")
    with pytest.raises(ValueError, match="execution floor"):
        exp.validate_artifact(complete_zero)

    bad_floor = dict(artifact, execution_floor_met="false")
    with pytest.raises(ValueError, match="execution_floor_met"):
        exp.validate_artifact(bad_floor)

    bad_calls = dict(artifact, total_codex_calls=True)
    with pytest.raises(ValueError, match="total_codex_calls"):
        exp.validate_artifact(bad_calls)

    bad_delta = dict(artifact, precision_vs_fresharm_base=True)
    with pytest.raises(ValueError, match="precision_vs_fresharm_base"):
        exp.validate_artifact(bad_delta)

    bad_gaps = dict(artifact, missing_verifier_gaps={})
    with pytest.raises(ValueError, match="missing_verifier_gaps"):
        exp.validate_artifact(bad_gaps)

    bad_substrate = dict(artifact, inference_substrate=3)
    with pytest.raises(ValueError, match="inference_substrate"):
        exp.validate_artifact(bad_substrate)

    bad_formula = dict(artifact, execution_floor_met=True, total_codex_calls=1, n_agreement_events=0)
    with pytest.raises(ValueError, match="execution_floor_met must equal"):
        exp.validate_artifact(bad_formula)

    bad_primary = dict(
        artifact,
        execution_floor_met=True,
        total_codex_calls=1,
        n_agreement_events=1,
        n_gold_given_agreement=1,
        primary_gate_passed=True,
        honest_verdict="complete: gap4_agreement_confidence_label_only_1of1",
    )
    with pytest.raises(ValueError, match="primary_gate_passed"):
        exp.validate_artifact(bad_primary)
