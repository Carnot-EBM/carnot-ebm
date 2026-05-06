"""Tests for Exp 1429 constrained repair candidate-search artifact.

Spec: REQ-VERIFY-1429, SCENARIO-VERIFY-1429
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from carnot.reporting import mcmc_constrained_repair_candidate_search as mod


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


def _exp1428_artifact(*, deployed: bool = True) -> dict[str, Any]:
    return {
        "status": "complete" if deployed else "blocked",
        "repair_executor_v2_deployed": deployed,
        "local_sota_model_used": QWEN_SPEC["hf_id"] if deployed else None,
        "honest_verdict": "complete_dccd_schema_constrained_repair_v2_nonzero_repairs",
    }


def _exp1397_artifact(n: int = 3) -> dict[str, Any]:
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
                "full_certificate_text": "<CARNOT_CERT_STATE:REPAIR_HINT>\nREPAIR_HINT.",
            }
        )
        certificate_rows.append({"case_id": case_id, "tag_state": "REPAIR_HINT"})
        semantic_rows.append(
            {
                "case_id": case_id,
                "constraint_passed": True,
                "semantic_result": "REPAIR_HINT",
            }
        )
        repair_rows.append(
            {
                "case_id": case_id,
                "localized": True,
                "localized_constraint": "fover_incorrect_reasoning_step",
                "minimal_local_change": "repair arithmetic step",
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
    }


def _dccd_json(*, variant: str) -> str:
    return json.dumps(
        {
            "draft_certificate": {
                "certificate_text": "<CARNOT_CERT_STATE:REPAIR_HINT>\nREPAIR_HINT.",
                "state": "REPAIR_HINT",
            },
            "repair_action": {
                "action_type": "STEP_REWRITE",
                "target": f"{variant} localized step",
                "rationale": "Bounded candidate search proposes a concrete repair.",
            },
            "final_certificate": {
                "certificate_text": "<CARNOT_CERT_STATE:SAT>\nSAT",
                "state": "SAT",
            },
            "validator_metadata": {"variant": variant},
        }
    )


def test_req1429_run_blocks_when_repair_v2_is_not_deployed(tmp_path: Path) -> None:
    """REQ-VERIFY-1429: Exp 1428 deployment gates candidate search."""

    exp1397 = tmp_path / "exp1397.json"
    exp1428 = tmp_path / "exp1428.json"
    output = tmp_path / "exp1429.json"
    exp1397.write_text(json.dumps(_exp1397_artifact()), encoding="utf-8")
    exp1428.write_text(json.dumps(_exp1428_artifact(deployed=False)), encoding="utf-8")
    writes: list[dict[str, Any]] = []

    artifact = mod.run_experiment(
        project_root=tmp_path,
        run_date="20260506",
        exp1397_path=exp1397,
        exp1428_path=exp1428,
        output_path=output,
        write_observer=lambda _path, payload: writes.append(dict(payload)),
    )

    assert [payload["status"] for payload in writes] == ["in_progress", "blocked"]
    assert set(mod.REQUIRED_ARTIFACT_FIELDS) <= set(artifact)
    assert artifact["status"] == "blocked"
    assert artifact["candidate_search_complete"] is False
    assert artifact["cases_evaluated"] == 0
    assert artifact["honest_verdict"] == "blocked_repair_v2_not_deployed"


def test_scenario1429_reporting_compares_one_candidate_to_best_of_n(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-1429: artifact reports best-of-N improvement over first candidate."""

    exp1397 = tmp_path / "exp1397.json"
    exp1428 = tmp_path / "exp1428.json"
    output = tmp_path / "exp1429.json"
    exp1397.write_text(json.dumps(_exp1397_artifact(3)), encoding="utf-8")
    exp1428.write_text(json.dumps(_exp1428_artifact()), encoding="utf-8")
    outputs = iter(
        [
            _dccd_json(variant="bad"),
            _dccd_json(variant="good"),
            _dccd_json(variant="bad"),
            _dccd_json(variant="good"),
            _dccd_json(variant="bad"),
            _dccd_json(variant="good"),
        ]
    )

    def generator_factory(_spec: dict[str, Any]) -> Any:
        return lambda _prompt: next(outputs)

    def validator(_request: Any, candidate: Any) -> dict[str, Any]:
        accepted = candidate.validator_metadata["variant"] == "good"
        return {
            "constraint_passed": accepted,
            "semantic_result": "SAT" if accepted else "REPAIR_HINT",
            "repair_required": not accepted,
            "false_acceptance": False,
            "energy": 0.25 if accepted else 7.0,
        }

    artifact = mod.run_experiment(
        project_root=tmp_path,
        run_date="20260506",
        exp1397_path=exp1397,
        exp1428_path=exp1428,
        output_path=output,
        cached_pair_fn=lambda **_kwargs: [QWEN_SPEC, GEMMA_SPEC],
        generator_factory=generator_factory,
        validator=validator,
        candidates_per_case=2,
        executor_runtime_mode="unit_test_injected_generator",
        tests_run=[".venv/bin/pytest tests/python -q"],
    )

    assert json.loads(output.read_text(encoding="utf-8")) == artifact
    assert artifact["status"] == "complete"
    assert artifact["candidate_search_complete"] is True
    assert artifact["cases_evaluated"] == 3
    assert artifact["candidates_per_case"] == 2
    assert artifact["mcmc_acceptance_rate"] == pytest.approx(0.5)
    assert artifact["repair_success_rate_one_candidate"] == pytest.approx(0.0)
    assert artifact["repair_success_rate_best_of_n"] == pytest.approx(1.0)
    assert artifact["energy_rerank_improved"] is True
    assert artifact["local_sota_model_used"] == QWEN_SPEC["hf_id"]
    assert artifact["local_sota_model_inference_used"] is False
    assert artifact["honest_verdict"] == (
        "complete_mcmc_constrained_repair_candidate_search_improved_"
        "prototype_no_headline_sota_claim"
    )


def test_req1429_run_blocks_when_sota_cache_is_unavailable(tmp_path: Path) -> None:
    """REQ-VERIFY-1429: missing mandated GGUF cache blocks headline candidate search."""

    exp1397 = tmp_path / "exp1397.json"
    exp1428 = tmp_path / "exp1428.json"
    output = tmp_path / "exp1429.json"
    exp1397.write_text(json.dumps(_exp1397_artifact()), encoding="utf-8")
    exp1428.write_text(json.dumps(_exp1428_artifact()), encoding="utf-8")

    artifact = mod.run_experiment(
        project_root=tmp_path,
        run_date="20260506",
        exp1397_path=exp1397,
        exp1428_path=exp1428,
        output_path=output,
        cached_pair_fn=lambda **_kwargs: None,
    )

    assert artifact["status"] == "blocked"
    assert artifact["candidate_search_complete"] is False
    assert artifact["local_sota_model_used"] is None
    assert artifact["honest_verdict"] == "blocked_sota_model_cache_unavailable"
    assert artifact["cache_diagnostics"]["cached_pair_available"] is False


def test_req1429_complete_verdict_variants() -> None:
    """REQ-VERIFY-1429: terminal verdict distinguishes empty, flat, and failed runs."""

    assert (
        mod._complete_verdict(
            cases_evaluated=0,
            one_candidate_successes=0,
            best_of_n_successes=0,
            executor_runtime_mode="live_local_sota_gguf",
        )
        == "complete_mcmc_constrained_repair_candidate_search_no_cases"
    )
    assert (
        mod._complete_verdict(
            cases_evaluated=3,
            one_candidate_successes=2,
            best_of_n_successes=2,
            executor_runtime_mode="live_local_sota_gguf",
        )
        == "complete_mcmc_constrained_repair_candidate_search_no_rate_improvement"
    )
    assert mod._complete_verdict(
        cases_evaluated=3,
        one_candidate_successes=0,
        best_of_n_successes=0,
        executor_runtime_mode="unit_test_injected_generator",
    ) == (
        "complete_mcmc_constrained_repair_candidate_search_no_successful_repairs_"
        "prototype_no_headline_sota_claim"
    )
