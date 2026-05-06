"""Tests for Exp 1428 DCCD schema-constrained repair v2 artifact.

Spec: REQ-VERIFY-1428, SCENARIO-VERIFY-1428
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from carnot.reporting import dccd_schema_constrained_repair_v2 as mod


QWEN_SPEC = {
    "name": "Qwen3.6-35B-A3B",
    "hf_id": "unsloth/Qwen3.6-35B-A3B-GGUF",
    "gpu": 0,
    "model_path": "/cache/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf",
}
GEMMA_SPEC = {
    "name": "Gemma4-31B-it",
    "hf_id": "unsloth/gemma-4-31B-it-GGUF",
    "gpu": 1,
    "model_path": "/cache/gemma-4-31B-it-Q4_K_M.gguf",
}


def _exp1397_artifact(n: int = 25) -> dict[str, Any]:
    generation_rows = []
    certificate_rows = []
    semantic_rows = []
    repair_rows = []
    scheduler_rows = []
    for index in range(n):
        case_id = f"repair_{index}"
        generation_rows.append(
            {
                "case_id": case_id,
                "reasoning_text": f"Question: compute {index}. Reasoning step is incorrect.",
                "full_certificate_text": "<CARNOT_CERT_STATE:REPAIR_HINT>\nREPAIR_HINT: add bound.",
                "model_hf_id": QWEN_SPEC["hf_id"],
                "generation_source": "live_sota_llamacpp",
            }
        )
        certificate_rows.append(
            {
                "case_id": case_id,
                "parseable": True,
                "tag_state": "REPAIR_HINT",
                "dispatched_state": "REPAIR_HINT",
            }
        )
        semantic_rows.append(
            {
                "case_id": case_id,
                "constraint_passed": True,
                "semantic_result": "REPAIR_HINT",
                "expected_state": "REPAIR_HINT",
                "failure_reason": None,
            }
        )
        repair_rows.append(
            {
                "case_id": case_id,
                "localized": True,
                "localized_constraint": "fover_incorrect_reasoning_step",
                "minimal_local_change": "repair_or_remove_incorrect_arithmetic_step",
                "repair_hint": "Repair the localized FoVer reasoning step before accepting.",
            }
        )
        scheduler_rows.append(
            {
                "case_id": case_id,
                "repair_required": True,
                "false_acceptance": False,
                "full_pipeline_pass": False,
            }
        )
    return {
        "status": "complete",
        "cases_evaluated": n,
        "generation_rows": generation_rows,
        "certificate_rows": certificate_rows,
        "semantic_validation_rows": semantic_rows,
        "repair_localization_rows": repair_rows,
        "scheduler_rows": scheduler_rows,
        "full_pipeline_pass_rate": 0.0,
    }


def _dccd_json(case_id: str = "repair_0") -> str:
    return json.dumps(
        {
            "draft_certificate": {
                "certificate_text": "<CARNOT_CERT_STATE:REPAIR_HINT>\nREPAIR_HINT: add bound.",
                "state": "REPAIR_HINT",
            },
            "repair_action": {
                "action_type": "STEP_REWRITE",
                "target": "localized FoVer reasoning step",
                "rationale": "Repair the localized incorrect step before accepting.",
            },
            "final_certificate": {
                "certificate_text": "<CARNOT_CERT_STATE:SAT>\nSAT",
                "state": "SAT",
            },
            "validator_metadata": {
                "expected_semantic_result": "SAT",
                "repair_hint_case_id": case_id,
            },
        }
    )


def test_req1428_run_blocks_when_sota_cache_is_unavailable(tmp_path: Path) -> None:
    """REQ-VERIFY-1428: unavailable SOTA GGUF cache writes a blocked artifact."""

    source = tmp_path / "exp1397.json"
    output = tmp_path / "exp1428.json"
    source.write_text(json.dumps(_exp1397_artifact()), encoding="utf-8")
    writes: list[dict[str, Any]] = []

    artifact = mod.run_experiment(
        project_root=tmp_path,
        run_date="20260506",
        exp1397_path=source,
        output_path=output,
        cached_pair_fn=lambda **_kwargs: None,
        write_observer=lambda _path, payload: writes.append(dict(payload)),
    )

    assert [payload["status"] for payload in writes] == ["in_progress", "blocked"]
    assert set(mod.REQUIRED_ARTIFACT_FIELDS) <= set(artifact)
    assert artifact["status"] == "blocked"
    assert artifact["repair_executor_v2_deployed"] is True
    assert artifact["repair_hint_cases_tested"] == 0
    assert artifact["local_sota_model_used"] is None
    assert artifact["honest_verdict"] == "blocked_sota_model_cache_unavailable"
    assert artifact["cache_diagnostics"]["cached_pair_available"] is False


def test_scenario1428_run_executes_twenty_schema_first_repairs(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-1428: twenty repair hints are schema- and semantic-validated."""

    source = tmp_path / "exp1397.json"
    output = tmp_path / "exp1428.json"
    source.write_text(json.dumps(_exp1397_artifact(25)), encoding="utf-8")

    def generator_factory(_spec: dict[str, Any]) -> Any:
        return lambda _prompt: _dccd_json()

    def validator(_request: Any, _candidate: Any) -> dict[str, Any]:
        return {
            "constraint_passed": True,
            "semantic_result": "SAT",
            "repair_required": False,
            "false_acceptance": False,
            "full_pipeline_pass": True,
        }

    artifact = mod.run_experiment(
        project_root=tmp_path,
        run_date="20260506",
        exp1397_path=source,
        output_path=output,
        cached_pair_fn=lambda **_kwargs: [QWEN_SPEC, GEMMA_SPEC],
        generator_factory=generator_factory,
        validator=validator,
        executor_runtime_mode="unit_test_injected_generator",
        tests_run=[".venv/bin/pytest tests/python -q"],
    )

    assert json.loads(output.read_text(encoding="utf-8")) == artifact
    assert artifact["status"] == "complete"
    assert artifact["repair_executor_v2_deployed"] is True
    assert artifact["repair_hint_cases_available"] == 25
    assert artifact["repair_hint_cases_tested"] == 20
    assert artifact["repaired_cases_successful"] == 20
    assert artifact["repaired_case_success_rate"] == pytest.approx(1.0)
    assert artifact["schema_valid_rate"] == pytest.approx(1.0)
    assert artifact["semantic_acceptance_rate"] == pytest.approx(1.0)
    assert artifact["local_sota_model_used"] == QWEN_SPEC["hf_id"]
    assert artifact["local_sota_model_inference_used"] is False
    assert artifact["rejection_reason_counts"]["schema_validation_failed"] == 0
    assert artifact["honest_verdict"] == (
        "complete_dccd_schema_constrained_repair_v2_nonzero_repairs_"
        "prototype_no_headline_sota_claim"
    )


def test_req1428_artifact_counts_rejection_reasons(tmp_path: Path) -> None:
    """REQ-VERIFY-1428: every rejected repair candidate has one rejection reason."""

    source = tmp_path / "exp1397.json"
    output = tmp_path / "exp1428.json"
    source.write_text(json.dumps(_exp1397_artifact(3)), encoding="utf-8")
    outputs = iter(["not json", _dccd_json(), _dccd_json()])

    def generator_factory(_spec: dict[str, Any]) -> Any:
        return lambda _prompt: next(outputs)

    def validator(request: Any, _candidate: Any) -> dict[str, Any]:
        if request.case_id == "repair_1":
            return {
                "constraint_passed": False,
                "semantic_result": "SAT",
                "repair_required": False,
                "false_acceptance": False,
            }
        return {
            "constraint_passed": True,
            "semantic_result": "SAT",
            "repair_required": False,
            "false_acceptance": False,
        }

    artifact = mod.run_experiment(
        project_root=tmp_path,
        run_date="20260506",
        exp1397_path=source,
        output_path=output,
        cached_pair_fn=lambda **_kwargs: [QWEN_SPEC, GEMMA_SPEC],
        generator_factory=generator_factory,
        validator=validator,
    )

    assert artifact["repair_hint_cases_tested"] == 3
    assert artifact["repaired_cases_successful"] == 1
    assert artifact["schema_valid_rate"] == pytest.approx(2 / 3)
    assert artifact["semantic_acceptance_rate"] == pytest.approx(1 / 3)
    assert artifact["rejection_reason_counts"]["schema_validation_failed"] == 1
    assert artifact["rejection_reason_counts"]["constraint_failed"] == 1
    assert sum(artifact["rejection_reason_counts"].values()) == 2


def test_req1428_complete_verdict_variants() -> None:
    """REQ-VERIFY-1428: terminal verdicts distinguish empty, short, failed, and positive runs."""

    assert mod._complete_verdict(available=0, tested=0, successful=0) == (
        "complete_no_repair_hint_cases_available"
    )
    assert mod._complete_verdict(available=3, tested=3, successful=0) == (
        "complete_dccd_schema_constrained_repair_v2_short_sample"
    )
    assert mod._complete_verdict(available=20, tested=20, successful=0) == (
        "complete_dccd_schema_constrained_repair_v2_no_successful_repairs"
    )
    assert mod._complete_verdict(available=20, tested=20, successful=1) == (
        "complete_dccd_schema_constrained_repair_v2_nonzero_repairs"
    )
    assert mod._complete_verdict(
        available=20,
        tested=20,
        successful=1,
        executor_runtime_mode="prototype_injected_generator",
    ) == (
        "complete_dccd_schema_constrained_repair_v2_nonzero_repairs_"
        "prototype_no_headline_sota_claim"
    )
