"""Tests for Exp 1414 certificate LLM repair executor artifact.

Spec: REQ-VERIFY-1414, SCENARIO-VERIFY-1414
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from carnot.reporting import certificate_llm_repair_executor_v1 as mod


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


def test_req1414_run_blocks_when_sota_cache_is_unavailable(tmp_path: Path) -> None:
    """REQ-VERIFY-1414: unavailable SOTA GGUF cache writes a blocked artifact."""

    source = tmp_path / "exp1397.json"
    output = tmp_path / "exp1414.json"
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
    assert artifact["repair_executor_deployed"] is True
    assert artifact["repair_hint_cases_tested"] == 0
    assert artifact["local_sota_model_used"] is None
    assert artifact["honest_verdict"] == "blocked_sota_model_cache_unavailable"
    assert artifact["cache_diagnostics"]["cached_pair_available"] is False


def test_scenario1414_run_executes_twenty_repair_hint_cases(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-1414: twenty repair hints are executed and validated."""

    source = tmp_path / "exp1397.json"
    output = tmp_path / "exp1414.json"
    source.write_text(json.dumps(_exp1397_artifact(25)), encoding="utf-8")

    def generator_factory(_spec: dict[str, Any]) -> Any:
        def generate(_prompt: str) -> str:
            return json.dumps(
                {
                    "corrected_certificate": "<CARNOT_CERT_STATE:SAT>\nSAT",
                    "corrected_reasoning_step": "corrected local step",
                    "metadata": {"repair": "accepted"},
                }
            )

        return generate

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
        tests_run=[".venv/bin/pytest tests/python -q"],
    )

    assert json.loads(output.read_text(encoding="utf-8")) == artifact
    assert artifact["status"] == "complete"
    assert artifact["repair_hint_cases_available"] == 25
    assert artifact["repair_hint_cases_tested"] == 20
    assert artifact["repaired_cases_successful"] == 20
    assert artifact["repaired_case_success_rate"] == pytest.approx(1.0)
    assert artifact["semantic_equivalence_pass_rate_after_repair"] == pytest.approx(1.0)
    assert artifact["local_sota_model_used"] == QWEN_SPEC["hf_id"]
    assert artifact["honest_verdict"] == "complete_repair_executor_validated_on_sample"
    assert [spec["role"] for spec in artifact["model_specs"]] == [
        "primary_repair_model",
        "dense_fallback",
        "moe_fallback",
    ]


def test_req1414_resolver_records_cache_exceptions() -> None:
    """REQ-VERIFY-1414: cache resolver failures are surfaced in diagnostics."""

    def broken_resolver(**_kwargs: Any) -> None:
        raise RuntimeError("cache offline")

    specs, diagnostics = mod.resolve_model_specs(broken_resolver)

    assert all(spec["cache_status"] == "missing" for spec in specs)
    assert diagnostics["cached_pair_available"] is False
    assert diagnostics["resolver_error"] == "RuntimeError: cache offline"


def test_req1414_request_builder_handles_sparse_rows() -> None:
    """REQ-VERIFY-1414: request construction tolerates partial Exp 1397 rows."""

    requests = mod.repair_requests_from_exp1397(
        {
            "generation_rows": "not-a-list",
            "certificate_rows": [
                {
                    "case_id": "fallback_cert",
                    "certificate_prefix": "<CARNOT_CERT_STATE:REPAIR_HINT>\n",
                    "certificate_body": "REPAIR_HINT: add bound.",
                    "reasoning_text": "fallback prompt",
                }
            ],
            "semantic_validation_rows": "not-a-list",
            "scheduler_rows": "not-a-list",
            "repair_localization_rows": [
                {"case_id": "ignored_without_hint"},
                {
                    "case_id": "fallback_cert",
                    "localized_constraint": "fover_incorrect_reasoning_step",
                    "minimal_local_change": "repair_or_remove_incorrect_arithmetic_step",
                    "repair_hint": "Repair this row.",
                },
            ],
        }
    )

    assert len(requests) == 1
    assert requests[0].case_id == "fallback_cert"
    assert requests[0].current_certificate == (
        "<CARNOT_CERT_STATE:REPAIR_HINT>\nREPAIR_HINT: add bound."
    )
    assert requests[0].original_prompt == "fallback prompt"


def test_req1414_complete_verdict_variants() -> None:
    """REQ-VERIFY-1414: terminal verdicts distinguish empty, short, and failed samples."""

    assert mod._complete_verdict(tested=0, successful=0, available=0) == (
        "complete_no_repair_hint_cases_available"
    )
    assert mod._complete_verdict(tested=3, successful=0, available=3) == (
        "complete_repair_executor_validated_on_available_short_sample"
    )
    assert mod._complete_verdict(tested=20, successful=0, available=20) == (
        "complete_repair_executor_no_successful_repairs"
    )
