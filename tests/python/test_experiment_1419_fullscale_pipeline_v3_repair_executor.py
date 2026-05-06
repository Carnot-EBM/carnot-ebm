"""Tests for Exp 1419 full-scale pipeline v3 repair-executor rerun.

Spec: REQ-VERIFY-1419, SCENARIO-VERIFY-1419
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from carnot.reporting import fullscale_pipeline_v3_repair_executor as mod


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


def _exp1414(deployed: bool = True) -> dict[str, Any]:
    return {
        "status": "complete",
        "repair_executor_deployed": deployed,
        "model_specs": [QWEN_SPEC, GEMMA_SPEC],
    }


def _exp1397(
    *,
    cases: int = 200,
    baseline_passes: int = 61,
    repair_hints: int = 100,
) -> dict[str, Any]:
    generation_rows = []
    certificate_rows = []
    semantic_rows = []
    repair_rows = []
    scheduler_rows = []
    for index in range(cases):
        case_id = f"case_{index}"
        has_repair = index < repair_hints
        passed = repair_hints <= index < repair_hints + baseline_passes
        expected_state = "REPAIR_HINT" if has_repair else "SAT"
        generation_rows.append(
            {
                "case_id": case_id,
                "reasoning_text": f"Reasoning row {index}",
                "full_certificate_text": f"<CARNOT_CERT_STATE:{expected_state}>\n{expected_state}",
                "model_hf_id": QWEN_SPEC["hf_id"],
                "generation_source": "live_sota_llamacpp",
            }
        )
        certificate_rows.append(
            {
                "case_id": case_id,
                "parseable": True,
                "tag_state": expected_state,
                "dispatched_state": expected_state,
            }
        )
        semantic_rows.append(
            {
                "case_id": case_id,
                "constraint_passed": True,
                "semantic_result": expected_state,
                "expected_state": expected_state,
                "failure_reason": None,
            }
        )
        if has_repair:
            repair_rows.append(
                {
                    "case_id": case_id,
                    "localized": True,
                    "localized_constraint": "fover_incorrect_reasoning_step",
                    "minimal_local_change": "repair_or_remove_incorrect_arithmetic_step",
                    "repair_hint": "Repair this localized FoVer reasoning step.",
                }
            )
        scheduler_rows.append(
            {
                "case_id": case_id,
                "repair_required": has_repair,
                "false_acceptance": False,
                "full_pipeline_pass": passed,
                "scheduler_action": "proxy_accept" if passed else "escalate_full_verifier",
            }
        )
    return {
        "status": "complete",
        "cases_evaluated": cases,
        "certificate_parse_rate": 1.0,
        "semantic_validation_pass_rate": 1.0,
        "full_pipeline_pass_rate": round(baseline_passes / cases, 6) if cases else 0.0,
        "generation_rows": generation_rows,
        "certificate_rows": certificate_rows,
        "semantic_validation_rows": semantic_rows,
        "repair_localization_rows": repair_rows,
        "scheduler_rows": scheduler_rows,
    }


def _write(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_req1419_blocks_when_exp1414_executor_is_not_deployed(tmp_path: Path) -> None:
    """REQ-VERIFY-1419: Exp 1414 deployment is a hard prerequisite gate."""

    exp1397_path = tmp_path / "exp1397.json"
    exp1414_path = tmp_path / "exp1414.json"
    output_path = tmp_path / "exp1419.json"
    _write(exp1397_path, _exp1397())
    _write(exp1414_path, _exp1414(deployed=False))
    writes: list[dict[str, Any]] = []

    artifact = mod.run_experiment(
        project_root=tmp_path,
        run_date="20260506",
        exp1397_path=exp1397_path,
        exp1414_path=exp1414_path,
        output_path=output_path,
        cached_pair_fn=lambda **_kwargs: [QWEN_SPEC, GEMMA_SPEC],
        write_observer=lambda _path, payload: writes.append(dict(payload)),
    )

    assert [payload["status"] for payload in writes] == ["in_progress", "blocked"]
    assert set(mod.REQUIRED_ARTIFACT_FIELDS) <= set(artifact)
    assert artifact["status"] == "blocked"
    assert artifact["blocker"] == "exp1414_repair_executor_not_deployed"
    assert artifact["cases_evaluated"] == 200
    assert artifact["full_pipeline_headline_gate_met"] is False


def test_req1419_blocks_when_case_count_is_below_200(tmp_path: Path) -> None:
    """REQ-VERIFY-1419: non-blocked artifacts require at least 200 source cases."""

    exp1397_path = tmp_path / "exp1397.json"
    exp1414_path = tmp_path / "exp1414.json"
    output_path = tmp_path / "exp1419.json"
    _write(exp1397_path, _exp1397(cases=199, baseline_passes=60, repair_hints=99))
    _write(exp1414_path, _exp1414())

    artifact = mod.run_experiment(
        project_root=tmp_path,
        run_date="20260506",
        exp1397_path=exp1397_path,
        exp1414_path=exp1414_path,
        output_path=output_path,
        cached_pair_fn=lambda **_kwargs: [QWEN_SPEC, GEMMA_SPEC],
    )

    assert artifact["status"] == "blocked"
    assert artifact["blocker"] == "source_case_count_below_200"
    assert artifact["cases_evaluated"] == 199
    assert artifact["repair_hint_cases_total"] == 99


def test_req1419_blocks_when_sota_cache_is_unavailable(tmp_path: Path) -> None:
    """REQ-VERIFY-1419: missing local SOTA GGUF cache is reported exactly."""

    exp1397_path = tmp_path / "exp1397.json"
    exp1414_path = tmp_path / "exp1414.json"
    output_path = tmp_path / "exp1419.json"
    _write(exp1397_path, _exp1397())
    _write(exp1414_path, _exp1414())

    artifact = mod.run_experiment(
        project_root=tmp_path,
        run_date="20260506",
        exp1397_path=exp1397_path,
        exp1414_path=exp1414_path,
        output_path=output_path,
        cached_pair_fn=lambda **_kwargs: None,
    )

    assert artifact["status"] == "blocked"
    assert artifact["blocker"] == "sota_model_cache_unavailable"
    assert artifact["actual_model_used"] is None
    assert artifact["repair_success_rate"] == pytest.approx(0.0)


def test_req1419_blocks_when_local_runtime_cannot_create_generator(tmp_path: Path) -> None:
    """REQ-VERIFY-1419: local model or GPU runtime failures preserve exact error text."""

    exp1397_path = tmp_path / "exp1397.json"
    exp1414_path = tmp_path / "exp1414.json"
    output_path = tmp_path / "exp1419.json"
    _write(exp1397_path, _exp1397())
    _write(exp1414_path, _exp1414())

    def broken_generator_factory(_spec: dict[str, Any]) -> Any:
        raise RuntimeError("libcudart.so.12 missing")

    artifact = mod.run_experiment(
        project_root=tmp_path,
        run_date="20260506",
        exp1397_path=exp1397_path,
        exp1414_path=exp1414_path,
        output_path=output_path,
        cached_pair_fn=lambda **_kwargs: [QWEN_SPEC, GEMMA_SPEC],
        generator_factory=broken_generator_factory,
    )

    assert artifact["status"] == "blocked"
    assert artifact["blocker"] == "local_model_runtime_unavailable"
    assert "libcudart.so.12 missing" in artifact["blocker_detail"]
    assert artifact["actual_model_used"] == QWEN_SPEC["hf_id"]


def test_scenario1419_repairs_update_full_pipeline_headline_gate(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-1419: accepted repairs lift the final 200-case pass rate."""

    exp1397_path = tmp_path / "exp1397.json"
    exp1414_path = tmp_path / "exp1414.json"
    output_path = tmp_path / "exp1419.json"
    _write(exp1397_path, _exp1397())
    _write(exp1414_path, _exp1414())

    def generator_factory(_spec: dict[str, Any]) -> Any:
        def generate(_prompt: str) -> str:
            return json.dumps(
                {
                    "corrected_certificate": "<CARNOT_CERT_STATE:SAT>\nSAT",
                    "corrected_reasoning_step": "corrected local reasoning step",
                    "metadata": {"source": "test"},
                }
            )

        return generate

    def validator(request: Any, _candidate: Any) -> dict[str, Any]:
        accepted_index = int(request.case_id.rsplit("_", 1)[-1])
        accepted = accepted_index < 40
        return {
            "constraint_passed": accepted,
            "semantic_result": "SAT" if accepted else "REPAIR_HINT",
            "repair_required": not accepted,
            "false_acceptance": False,
            "full_pipeline_pass": accepted,
        }

    artifact = mod.run_experiment(
        project_root=tmp_path,
        run_date="20260506",
        exp1397_path=exp1397_path,
        exp1414_path=exp1414_path,
        output_path=output_path,
        cached_pair_fn=lambda **_kwargs: [QWEN_SPEC, GEMMA_SPEC],
        generator_factory=generator_factory,
        validator=validator,
        tests_run=[".venv/bin/pytest tests/python -q"],
    )

    assert json.loads(output_path.read_text(encoding="utf-8")) == artifact
    assert artifact["status"] == "complete"
    assert artifact["cases_evaluated"] == 200
    assert artifact["certificate_parse_rate"] == pytest.approx(1.0)
    assert artifact["semantic_validation_pass_rate"] == pytest.approx(1.0)
    assert artifact["repair_hint_cases_total"] == 100
    assert artifact["repaired_cases_successful"] == 40
    assert artifact["repair_success_rate"] == pytest.approx(0.4)
    assert artifact["full_pipeline_pass_rate"] == pytest.approx(0.505)
    assert artifact["full_pipeline_delta_vs_exp1397"] == pytest.approx(0.2)
    assert artifact["full_pipeline_headline_gate_met"] is True
    assert artifact["actual_model_used"] == QWEN_SPEC["hf_id"]
    assert artifact["model_specs"][0]["role"] == "primary_pipeline_repair_model"
    assert artifact["honest_verdict"] == "headline_full_pipeline_gate_met"


def test_req1419_source_metric_fallbacks_are_deterministic() -> None:
    """REQ-VERIFY-1419: missing source rates are recomputed from audited rows."""

    metrics = mod._source_metrics(
        {
            "scheduler_rows": [
                {"case_id": "a", "full_pipeline_pass": True},
                {"case_id": "b", "full_pipeline_pass": False},
                {"case_id": "ignored_without_id", "full_pipeline_pass": True},
            ],
            "certificate_rows": [
                {"case_id": "a", "parseable": True},
                {"case_id": "b", "parseable": False},
            ],
            "semantic_validation_rows": [
                {"case_id": "a", "constraint_passed": True},
                {"case_id": "b", "constraint_passed": False},
            ],
        }
    )

    assert metrics["cases_evaluated"] == 3
    assert metrics["certificate_parse_rate"] == pytest.approx(1 / 3)
    assert metrics["semantic_validation_pass_rate"] == pytest.approx(1 / 3)
    assert metrics["full_pipeline_pass_rate"] == pytest.approx(2 / 3)
    assert mod._rows("not rows") == []
    assert mod._float("not a float") is None
    assert mod._complete_verdict(headline_met=False) == "not_headline_full_pipeline_below_0_40"
