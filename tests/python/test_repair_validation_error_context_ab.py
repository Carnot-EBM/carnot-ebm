"""Tests for Exp 1464 repair validation-error-context A/B evaluation.

Spec: REQ-VERIFY-1464, SCENARIO-VERIFY-1464
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from carnot.reporting.repair_validation_error_context_ab import (
    REQUIRED_ARTIFACT_FIELDS,
    run_experiment,
)


def _exp1397_fixture() -> dict[str, Any]:
    return {
        "generation_rows": [
            {
                "case_id": "156",
                "reasoning_text": "Question: What is 2 + 2? Step: 2 + 2 = 5.",
                "full_certificate_text": "<CARNOT_CERT_STATE:REPAIR_HINT>\nREPAIR_HINT",
            }
        ],
        "certificate_rows": [{"case_id": "156"}],
        "semantic_validation_rows": [
            {
                "case_id": "156",
                "semantic_result": "REPAIR_HINT",
                "failure_reason": "FoVer step contradicts arithmetic.",
            }
        ],
        "scheduler_rows": [
            {
                "case_id": "156",
                "scheduler_action": "repair",
                "repair_required": True,
                "full_pipeline_pass": False,
            }
        ],
        "repair_localization_rows": [
            {
                "case_id": "156",
                "accepted": False,
                "localized_constraint": "fover_incorrect_reasoning_step",
                "minimal_local_change": "repair_or_remove_incorrect_arithmetic_step",
                "repair_hint": "Repair the localized FoVer reasoning step before accepting.",
            }
        ],
    }


def _cached_pair(**_kwargs: Any) -> list[dict[str, Any]]:
    return [
        {
            "name": "Qwen3.6-35B-A3B",
            "hf_id": "unsloth/Qwen3.6-35B-A3B-GGUF",
            "gpu": 0,
            "model_path": "/models/qwen.gguf",
        },
        {
            "name": "Gemma4-31B-it",
            "hf_id": "unsloth/gemma-4-31B-it-GGUF",
            "gpu": 1,
            "model_path": "/models/gemma31.gguf",
        },
    ]


def _valid_dccd_payload() -> str:
    return json.dumps(
        {
            "draft_certificate": {
                "certificate_text": "<CARNOT_CERT_STATE:REPAIR_HINT>\nREPAIR_HINT",
                "state": "REPAIR_HINT",
            },
            "repair_action": {
                "action_type": "STEP_REWRITE",
                "target": "localized FoVer reasoning step",
                "rationale": "The validator identified the missing SAT final state.",
            },
            "final_certificate": {
                "certificate_text": "<CARNOT_CERT_STATE:SAT>\nSAT",
                "state": "SAT",
            },
            "validator_metadata": {"expected_semantic_result": "SAT"},
        }
    )


def test_scenario1464_context_improvement_preserves_lineage(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-1464: exact validation error context can improve retry metrics."""

    exp1397_path = tmp_path / "exp1397.json"
    output_path = tmp_path / "experiment_1464.json"
    exp1397_path.write_text(json.dumps(_exp1397_fixture()), encoding="utf-8")
    writes: list[str] = []

    def generator_factory(_model_spec: dict[str, Any]):
        def generator(prompt: str) -> str:
            if "Retry contract" in prompt and "validation_error_message" in prompt:
                return _valid_dccd_payload()
            return "not json"

        return generator

    def validator(_request: Any, _candidate: Any) -> dict[str, Any]:
        return {
            "constraint_passed": True,
            "semantic_result": "SAT",
            "repair_required": False,
            "false_acceptance": False,
        }

    artifact = run_experiment(
        project_root=tmp_path,
        run_date="20260507",
        exp1397_path=exp1397_path,
        output_path=output_path,
        cached_pair_fn=_cached_pair,
        generator_factory=generator_factory,
        validator=validator,
        executor_runtime_mode="unit_test_injected_generator",
        max_cases=1,
        commands_run=["pytest tests/python/test_repair_validation_error_context_ab.py"],
        write_observer=lambda _path, payload: writes.append(payload["status"]),
    )

    assert set(REQUIRED_ARTIFACT_FIELDS).issubset(artifact)
    assert writes == ["in_progress", "complete"]
    assert artifact["live_sota_model_inference_used"] is False
    assert artifact["validation_error_context_enabled"] is True
    assert artifact["cases_evaluated"] == 1
    assert artifact["baseline_acceptance_rate"] == 0.0
    assert artifact["context_acceptance_rate"] == 1.0
    assert artifact["acceptance_delta_pp"] == 100.0
    assert artifact["schema_validity_delta_pp"] == 100.0
    assert artifact["semantic_correctness_delta_pp"] == 100.0
    assert artifact["repair_executor_lineage_preserved"] is True
    assert artifact["repair_executor_lineage_retired"] is False
    assert artifact["selected_model"]["hf_id"] == "unsloth/Qwen3.6-35B-A3B-GGUF"
    assert artifact["per_case_results"][0]["baseline"]["schema_valid"] is False
    assert artifact["per_case_results"][0]["context"]["accepted"] is True


def test_req1464_no_metric_improvement_retires_lineage(tmp_path: Path) -> None:
    """REQ-VERIFY-1464: no metric improvement sets repair_executor_lineage_retired."""

    exp1397_path = tmp_path / "exp1397.json"
    output_path = tmp_path / "experiment_1464.json"
    exp1397_path.write_text(json.dumps(_exp1397_fixture()), encoding="utf-8")

    artifact = run_experiment(
        project_root=tmp_path,
        run_date="20260507",
        exp1397_path=exp1397_path,
        output_path=output_path,
        cached_pair_fn=_cached_pair,
        generator_factory=lambda _model_spec: (lambda _prompt: _valid_dccd_payload()),
        executor_runtime_mode="unit_test_injected_generator",
        max_cases=1,
    )

    assert artifact["status"] == "complete"
    assert artifact["acceptance_delta_pp"] == 0.0
    assert artifact["schema_validity_delta_pp"] == 0.0
    assert artifact["semantic_correctness_delta_pp"] == 0.0
    assert artifact["repair_executor_lineage_preserved"] is False
    assert artifact["repair_executor_lineage_retired"] is True
    assert artifact["honest_verdict"] == "complete_no_retry_context_improvement_repair_executor_retired"


def test_req1464_empty_repair_subset_completes_without_retiring(tmp_path: Path) -> None:
    """REQ-VERIFY-1464: an empty FoVer repair subset is reported separately."""

    fixture = _exp1397_fixture()
    fixture["repair_localization_rows"] = []
    exp1397_path = tmp_path / "exp1397.json"
    output_path = tmp_path / "experiment_1464.json"
    exp1397_path.write_text(json.dumps(fixture), encoding="utf-8")

    artifact = run_experiment(
        project_root=tmp_path,
        run_date="20260507",
        exp1397_path=exp1397_path,
        output_path=output_path,
        cached_pair_fn=_cached_pair,
        generator_factory=lambda _model_spec: (lambda _prompt: "should not run"),
        executor_runtime_mode="unit_test_injected_generator",
        max_cases=1,
    )

    assert artifact["status"] == "complete"
    assert artifact["cases_evaluated"] == 0
    assert artifact["honest_verdict"] == "complete_no_repair_hint_cases_available"


def test_req1464_missing_sota_cache_blocks_headline_claim(tmp_path: Path) -> None:
    """REQ-VERIFY-1464: missing cached_sota_pair() prevents headline A/B evidence."""

    exp1397_path = tmp_path / "exp1397.json"
    output_path = tmp_path / "experiment_1464.json"
    exp1397_path.write_text(json.dumps(_exp1397_fixture()), encoding="utf-8")

    artifact = run_experiment(
        project_root=tmp_path,
        run_date="20260507",
        exp1397_path=exp1397_path,
        output_path=output_path,
        cached_pair_fn=lambda **_kwargs: None,
        max_cases=1,
    )

    assert artifact["status"] == "blocked"
    assert artifact["live_sota_model_inference_used"] is False
    assert artifact["cases_evaluated"] == 0
    assert artifact["repair_executor_lineage_retired"] is False
    assert artifact["honest_verdict"] == "blocked_sota_model_cache_unavailable"
