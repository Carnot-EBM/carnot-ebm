"""Tests for Exp 1317 GRPO/VPRM v11 headline gate.

Spec: REQ-LEARN-1317, SCENARIO-LEARN-1317.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from carnot.reporting import grpo_vprm_v11_headline_gate as exp


QWEN = "unsloth/Qwen3.6-35B-A3B-GGUF"
GEMMA = "unsloth/gemma-4-31B-it-GGUF"


def _attempt(
    *,
    hf_id: str,
    item_id: str,
    path: str,
    truthful: bool,
    parseable: bool = True,
    compact: bool = False,
) -> dict[str, object]:
    return {
        "hf_id": hf_id,
        "item_id": item_id,
        "path": path,
        "parseable": parseable,
        "truthful": truthful,
        "compact_encoding": compact,
        "prompt_chars": 100 if path != "dccd_compact" else 40,
        "errors": [] if parseable else ["no_json_object"],
    }


def _exp1312_payload(*, headline: bool = True, status: str = "complete") -> dict[str, object]:
    attempts: list[dict[str, object]] = []
    cases = [
        (QWEN, "case_repair_from_bad_grammar", False, True, True),
        (QWEN, "case_already_good", True, True, None),
        (GEMMA, "case_token_mask_repairs_dccd", False, False, True),
        (GEMMA, "case_still_bad", False, False, None),
    ]
    for hf_id, item_id, gbnf_truth, dccd_truth, repaired_truth in cases:
        attempts.append(
            _attempt(
                hf_id=hf_id,
                item_id=item_id,
                path="raw_trigger",
                truthful=False,
                parseable=False,
            )
        )
        attempts.append(
            _attempt(hf_id=hf_id, item_id=item_id, path="gbnf_constrained", truthful=gbnf_truth)
        )
        attempts.append(
            _attempt(hf_id=hf_id, item_id=item_id, path="dccd_compact", truthful=dccd_truth)
        )
        if repaired_truth is not None:
            attempts.append(
                _attempt(
                    hf_id=hf_id,
                    item_id=item_id,
                    path="repaired_certificate",
                    truthful=repaired_truth,
                )
            )
    return {
        "artifact": "experiment_1312_triggered_certificate_extraction_dccd_gbnf",
        "status": status,
        "run_date": "20260505",
        "headline_result_allowed": headline,
        "honest_verdict": "triggered_certificate_dccd_gbnf_comparison_complete",
        "models_used": [QWEN, GEMMA],
        "certificate_parse_rate": 0.75,
        "certificate_truthfulness_rate": 0.75,
        "attempts": attempts,
    }


def _exp1315_payload(
    *,
    delta: float = 0.25,
    nonforgetting_rate: float = 1.0,
    memory_regressions: int = 0,
    penalty: float = 0.0,
    status: str = "complete",
) -> dict[str, object]:
    return {
        "experiment": "1315_continuous_self_learning_cerce_nonforgetting_audit",
        "status": status,
        "run_date": "20260505",
        "headline_result_allowed": False,
        "honest_verdict": "cerce_nonforgetting_preserved_improved_non_headline",
        "self_learning_delta_overall": delta,
        "nonforgetting_certificate_rate": nonforgetting_rate,
        "memory_regression_count": memory_regressions,
        "lagrangian_violation_penalty": penalty,
    }


def _cached_specs() -> list[dict[str, object]]:
    return [
        {"name": "Qwen3.6-35B-A3B", "hf_id": QWEN, "gpu": 0, "model_path": "/models/qwen.gguf"},
        {"name": "Gemma4-31B-it", "hf_id": GEMMA, "gpu": 1, "model_path": "/models/gemma.gguf"},
    ]


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_req_learn_1317_writes_in_progress_artifact_first(tmp_path: Path) -> None:
    """REQ-LEARN-1317-1: the workflow starts with an in-progress artifact."""

    out_path = tmp_path / exp.OUTPUT_FILE

    artifact = exp.write_in_progress_artifact(
        out_path,
        project_root="/home/ianblenke/github.com/ianblenke/carnot",
    )

    assert json.loads(out_path.read_text(encoding="utf-8")) == artifact
    assert artifact["status"] == "in_progress"
    assert artifact["run_date"] == "20260505"
    assert artifact["artifact_metadata"]["run_date"] == "20260505"
    assert artifact["grpo_vprm_delta"] is None
    assert artifact["headline_result_allowed"] is False
    assert artifact["honest_verdict"] == "in_progress"


def test_req_learn_1317_missing_inputs_block_before_activation(tmp_path: Path) -> None:
    """REQ-LEARN-1317-2: absent Exp 1312/1315 inputs produce a terminal blocker."""

    results_dir = tmp_path / "results"
    results_dir.mkdir()
    _write_json(results_dir / exp.EXP1312_FILE, _exp1312_payload())
    out_path = results_dir / exp.OUTPUT_FILE

    artifact = exp.run(
        results_dir=results_dir,
        out_path=out_path,
        cached_pair_fn=lambda gpu_indices=(0, 1): _cached_specs(),
    )

    assert artifact["status"] == "blocked"
    assert artifact["honest_verdict"] == "blocked_missing_inputs"
    assert artifact["missing_inputs"] == [f"results/{exp.EXP1315_FILE}"]
    assert artifact["headline_result_allowed"] is False
    assert json.loads(out_path.read_text(encoding="utf-8")) == artifact


def test_req_learn_1317_structured_gates_block_bad_sources(tmp_path: Path) -> None:
    """REQ-LEARN-1317-2/3: non-headline certs or regressed learning block."""

    blockers = exp.structured_gate_blockers(
        _exp1312_payload(headline=False),
        _exp1315_payload(delta=0.0, nonforgetting_rate=0.8, memory_regressions=1, penalty=0.1),
        model_resolution=exp.resolve_model_specs(
            cached_pair_fn=lambda gpu_indices=(0, 1): [],
            exp1312_models=[QWEN, GEMMA],
        ),
    )

    blocker_names = {blocker["gate"] for blocker in blockers}
    assert "exp1312_headline_result_allowed" in blocker_names
    assert "exp1315_positive_self_learning_delta" in blocker_names
    assert "exp1315_nonforgetting_preserved" in blocker_names
    assert "cached_sota_pair_available" in blocker_names

    out_path = tmp_path / exp.OUTPUT_FILE
    artifact = exp.write_terminal_blocker(
        out_path,
        blockers,
        missing_inputs=[],
        project_root="/home/ianblenke/github.com/ianblenke/carnot",
    )

    assert artifact["status"] == "blocked"
    assert artifact["honest_verdict"] == "blocked_structured_gate_failed"
    assert artifact["grpo_vprm_delta"] == 0.0
    assert artifact["nonforgetting_preserved"] is False
    exp.validate_artifact(artifact)

    results_dir = tmp_path / "results"
    results_dir.mkdir()
    _write_json(results_dir / exp.EXP1312_FILE, _exp1312_payload(headline=False))
    _write_json(results_dir / exp.EXP1315_FILE, _exp1315_payload())
    run_artifact = exp.run(
        results_dir=results_dir,
        out_path=results_dir / exp.OUTPUT_FILE,
        cached_pair_fn=lambda gpu_indices=(0, 1): _cached_specs(),
    )
    assert run_artifact["status"] == "blocked"
    assert run_artifact["honest_verdict"] == "blocked_structured_gate_failed"


def test_scenario_learn_1317_certificate_replay_improves_policy_behavior() -> None:
    """SCENARIO-LEARN-1317: token-mask replay improves without forgetting."""

    model_resolution = exp.resolve_model_specs(
        cached_pair_fn=lambda gpu_indices=(0, 1): _cached_specs(),
        exp1312_models=[QWEN, GEMMA],
    )
    artifact = exp.build_artifact(
        _exp1312_payload(),
        _exp1315_payload(delta=0.25),
        model_resolution=model_resolution,
        project_root="/home/ianblenke/github.com/ianblenke/carnot",
    )

    exp.validate_artifact(artifact)
    assert artifact["status"] == "complete"
    assert artifact["n_certificate_cases"] == 4
    assert artifact["baseline_policy_score"] == 0.25
    assert artifact["dccd_policy_score"] == 0.5
    assert artifact["verifier_feedback_token_mask_score"] == 0.75
    assert artifact["grpo_vprm_delta"] == 0.5
    assert artifact["verifier_feedback_token_mask_delta"] == 0.25
    assert artifact["self_verification_gain"] == 0.5
    assert artifact["nonforgetting_preserved"] is True
    assert artifact["headline_result_allowed"] is True
    assert artifact["honest_verdict"] == "grpo_vprm_v11_positive_headline_gate"
    assert artifact["update_budget"] == {"train_steps": 0, "replay_cases": 4, "deterministic": True}
    assert {model["hf_id"] for model in artifact["models_used"]} == {QWEN, GEMMA}


def test_req_learn_1317_run_writes_final_artifact_from_sources(tmp_path: Path) -> None:
    """REQ-LEARN-1317-4/5/6: run loads sources and writes required fields."""

    results_dir = tmp_path / "results"
    results_dir.mkdir()
    _write_json(results_dir / exp.EXP1312_FILE, _exp1312_payload())
    _write_json(results_dir / exp.EXP1315_FILE, _exp1315_payload())
    out_path = results_dir / exp.OUTPUT_FILE

    artifact = exp.run(
        results_dir=results_dir,
        out_path=out_path,
        cached_pair_fn=lambda gpu_indices=(0, 1): _cached_specs(),
        project_root="/home/ianblenke/github.com/ianblenke/carnot",
    )

    assert json.loads(out_path.read_text(encoding="utf-8")) == artifact
    assert artifact["status"] == "complete"
    assert artifact["source_artifacts"] == {
        "exp1312": f"results/{exp.EXP1312_FILE}",
        "exp1315": f"results/{exp.EXP1315_FILE}",
    }
    assert artifact["artifact_metadata"]["run_date"] == "20260505"


def test_req_learn_1317_validation_and_metric_edges_are_strict() -> None:
    """REQ-LEARN-1317-4/5/6: schema and replay metric edges stay strict."""

    assert exp._mean([]) == 0.0
    assert exp._attempt_reward({}) == 0.0
    assert exp._gate_summary([], []) == "all structured gates passed"
    assert exp.derive_honest_verdict(delta=0.1, headline_result_allowed=False) == (
        "grpo_vprm_v11_positive_non_headline"
    )
    assert exp.derive_honest_verdict(delta=-0.1, headline_result_allowed=True) == (
        "grpo_vprm_v11_regression"
    )
    assert exp.derive_honest_verdict(delta=0.0, headline_result_allowed=True) == (
        "grpo_vprm_v11_neutral"
    )
    empty_metrics = exp.audit_certificate_policy([])
    assert empty_metrics["grpo_vprm_delta"] == 0.0
    noisy_payload = _exp1312_payload()
    noisy_payload["attempts"] = ["skip", *noisy_payload["attempts"]]
    assert len(exp.build_certificate_corpus(noisy_payload)) == 4
    path, selected = exp._select_token_mask_attempt(
        {"gbnf_constrained": {"parseable": False, "truthful": False}}
    )
    assert path == "gbnf_constrained"
    assert selected == {"parseable": False, "truthful": False}

    with pytest.raises(AssertionError, match="missing required fields"):
        exp.validate_artifact({"status": "complete"})

    artifact = exp.build_artifact(
        _exp1312_payload(),
        _exp1315_payload(),
        model_resolution=exp.resolve_model_specs(
            cached_pair_fn=lambda gpu_indices=(0, 1): _cached_specs(),
            exp1312_models=[QWEN, GEMMA],
        ),
    )
    for key, value, message in [
        ("status", "in_progress", "status must be complete or blocked"),
        ("grpo_vprm_delta", "bad", "grpo_vprm_delta must be numeric"),
        (
            "verifier_feedback_token_mask_delta",
            None,
            "verifier_feedback_token_mask_delta must be numeric",
        ),
        ("nonforgetting_preserved", "yes", "nonforgetting_preserved must be boolean"),
        ("self_verification_gain", "bad", "self_verification_gain must be numeric"),
        ("models_used", [], "models_used must include"),
        ("headline_result_allowed", "yes", "headline_result_allowed must be boolean"),
        ("honest_verdict", "unexpected", "honest_verdict is unsupported"),
    ]:
        bad = dict(artifact)
        bad[key] = value
        with pytest.raises(AssertionError, match=message):
            exp.validate_artifact(bad)

    dishonest = dict(artifact)
    dishonest["headline_result_allowed"] = True
    dishonest["structured_gates"] = [
        {"gate": "exp1312_status_complete", "passed": False, "reason": "no"}
    ]
    with pytest.raises(AssertionError, match="headline artifacts require all gates"):
        exp.validate_artifact(dishonest)

    legacy_headline = dict(artifact)
    legacy_headline["models_used"] = [{"hf_id": "Qwen/Qwen3.5-0.8B"}]
    with pytest.raises(AssertionError, match="mandated SOTA"):
        exp.validate_artifact(legacy_headline)
