"""Tests for Exp 1431 full-pipeline v4 micro-gated validation.

Spec: REQ-VERIFY-1431, SCENARIO-VERIFY-1431
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from carnot.reporting import fullscale_pipeline_v4_micro_gated as mod


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


def _exp1397_source(*, repair_cases: int = 20, original_passes: int = 61) -> dict[str, Any]:
    generation_rows = []
    certificate_rows = []
    semantic_rows = []
    scheduler_rows = []
    case_ids = [f"repair_{index:02d}" for index in range(repair_cases)]
    case_ids += [f"pass_{index:03d}" for index in range(original_passes)]
    case_ids += [f"fail_{index:03d}" for index in range(200 - len(case_ids))]
    pass_ids = {case_id for case_id in case_ids if case_id.startswith("pass_")}
    repair_ids = {case_id for case_id in case_ids if case_id.startswith("repair_")}
    for case_id in case_ids:
        is_repair = case_id in repair_ids
        passed = case_id in pass_ids
        state = "REPAIR_HINT" if is_repair else "SAT"
        generation_rows.append(
            {
                "case_id": case_id,
                "reasoning_text": f"Reasoning for {case_id}",
                "full_certificate_text": f"<CARNOT_CERT_STATE:{state}>\n{state}",
                "model_hf_id": QWEN_SPEC["hf_id"],
                "generation_source": "live_sota_llamacpp",
            }
        )
        certificate_rows.append(
            {
                "case_id": case_id,
                "parseable": True,
                "tag_state": state,
                "dispatched_state": state,
            }
        )
        semantic_rows.append(
            {
                "case_id": case_id,
                "constraint_passed": True,
                "semantic_result": state,
                "expected_state": state,
            }
        )
        scheduler_rows.append(
            {
                "case_id": case_id,
                "repair_required": is_repair,
                "false_acceptance": False,
                "full_pipeline_pass": passed,
                "scheduler_action": "proxy_accept" if passed else "escalate_full_verifier",
                "semantic_result": state,
            }
        )
    return {
        "status": "complete",
        "cases_evaluated": 200,
        "certificate_parse_rate": 1.0,
        "semantic_validation_pass_rate": 1.0,
        "full_pipeline_pass_rate": 0.305,
        "generation_rows": generation_rows,
        "certificate_rows": certificate_rows,
        "semantic_validation_rows": semantic_rows,
        "scheduler_rows": scheduler_rows,
    }


def _exp1428(
    *,
    success_rate: float = 1.0,
    runtime_mode: str = "live_local_sota_gguf",
) -> dict[str, Any]:
    return {
        "status": "complete",
        "model_specs": [QWEN_SPEC, GEMMA_SPEC],
        "repair_executor_v2_deployed": True,
        "repaired_cases_successful": int(success_rate * 20),
        "repaired_case_success_rate": success_rate,
        "local_sota_model_used": QWEN_SPEC["hf_id"],
        "executor_runtime_mode": runtime_mode,
        "honest_verdict": "complete_dccd_schema_constrained_repair_v2_nonzero_repairs",
    }


def _exp1430(*, ready: bool = True, accepted_cases: int = 20) -> dict[str, Any]:
    return {
        "status": "complete",
        "prm_guided_selection_ready": ready,
        "cases_evaluated": accepted_cases,
        "selected_repair_success_rate": 1.0 if accepted_cases else 0.0,
        "selector_scoring_mode": "prmv1_checkpoint",
        "prmv1_artifact_used": True,
        "honest_verdict": "complete_prm_guided_selector_no_improvement",
        "case_selections": [
            {
                "case_id": f"repair_{index:02d}",
                "selected_candidate_index": 1,
                "selected_accepted": True,
                "raw_best_of_n_success": True,
            }
            for index in range(accepted_cases)
        ],
    }


def _write(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")


def test_req1431_writes_in_progress_artifact_first(tmp_path: Path) -> None:
    """REQ-VERIFY-1431: bootstrap artifact is written before source loading."""

    output = tmp_path / "exp1431.json"

    artifact = mod.write_in_progress_artifact(output, project_root=tmp_path)

    assert json.loads(output.read_text(encoding="utf-8")) == artifact
    assert artifact["status"] == "in_progress"
    assert set(mod.REQUIRED_ARTIFACT_FIELDS) <= set(artifact)
    assert artifact["eligible_for_200_case_scaleup"] is False
    assert artifact["honest_verdict"] == "in_progress"


def test_req1431_blocks_when_repair_v2_gate_is_not_satisfied(tmp_path: Path) -> None:
    """REQ-VERIFY-1431: Exp 1428 nonzero repair acceptance is a hard gate."""

    exp1397 = tmp_path / "exp1397.json"
    exp1428 = tmp_path / "exp1428.json"
    exp1430 = tmp_path / "exp1430.json"
    output = tmp_path / "exp1431.json"
    _write(exp1397, _exp1397_source())
    _write(exp1428, _exp1428(success_rate=0.0))
    _write(exp1430, _exp1430())
    writes: list[dict[str, Any]] = []

    artifact = mod.run_experiment(
        project_root=tmp_path,
        run_date="20260506",
        exp1397_path=exp1397,
        exp1428_path=exp1428,
        exp1430_path=exp1430,
        output_path=output,
        write_observer=lambda _path, payload: writes.append(dict(payload)),
    )

    assert [payload["status"] for payload in writes] == ["in_progress", "blocked"]
    assert artifact["status"] == "blocked"
    assert artifact["structured_gates_satisfied"] is False
    assert artifact["blocker"] == "exp1428_repair_v2_nonzero_acceptance_missing"
    assert artifact["eligible_for_200_case_scaleup"] is False


def test_req1431_blocks_when_prm_selector_is_not_ready(tmp_path: Path) -> None:
    """REQ-VERIFY-1431: Exp 1430 PRM-guided selection readiness is required."""

    exp1397 = tmp_path / "exp1397.json"
    exp1428 = tmp_path / "exp1428.json"
    exp1430 = tmp_path / "exp1430.json"
    output = tmp_path / "exp1431.json"
    _write(exp1397, _exp1397_source())
    _write(exp1428, _exp1428())
    _write(exp1430, _exp1430(ready=False))

    artifact = mod.run_experiment(
        project_root=tmp_path,
        run_date="20260506",
        exp1397_path=exp1397,
        exp1428_path=exp1428,
        exp1430_path=exp1430,
        output_path=output,
    )

    assert artifact["status"] == "blocked"
    assert artifact["blocker"] == "exp1430_prm_guided_selection_not_ready"
    assert artifact["prm_guided_selection_enabled"] is True
    assert artifact["eligible_for_200_case_scaleup"] is False


def test_scenario1431_micro_gate_counts_selected_repairs_and_beats_baseline(
    tmp_path: Path,
) -> None:
    """SCENARIO-VERIFY-1431: selected accepted repairs lift the 50-case rate."""

    exp1397 = tmp_path / "exp1397.json"
    exp1428 = tmp_path / "exp1428.json"
    exp1430 = tmp_path / "exp1430.json"
    output = tmp_path / "exp1431.json"
    _write(exp1397, _exp1397_source())
    _write(exp1428, _exp1428())
    _write(exp1430, _exp1430())

    artifact = mod.run_experiment(
        project_root=tmp_path,
        run_date="20260506",
        exp1397_path=exp1397,
        exp1428_path=exp1428,
        exp1430_path=exp1430,
        output_path=output,
        tests_run=[".venv/bin/pytest tests/python -q"],
    )

    assert json.loads(output.read_text(encoding="utf-8")) == artifact
    mod.validate_artifact(artifact)
    assert artifact["status"] == "complete"
    assert artifact["structured_gates_satisfied"] is True
    assert artifact["cases_evaluated"] == 50
    assert artifact["sample_case_ids"] != [
        row["case_id"] for row in artifact["source_exp1397_first_50_rows"]
    ]
    assert set(artifact["sample_case_ids"][:20]) == {f"repair_{index:02d}" for index in range(20)}
    assert artifact["repair_hint_cases_total"] == 20
    assert artifact["repaired_cases_successful"] == 20
    assert artifact["repair_success_rate"] == pytest.approx(1.0)
    assert artifact["full_pipeline_pass_rate"] >= 0.4
    assert artifact["beats_exp1419_baseline"] is True
    assert artifact["eligible_for_200_case_scaleup"] is True
    assert artifact["local_sota_model_used"] == QWEN_SPEC["hf_id"]
    assert artifact["honest_verdict"] == "complete_micro_validation_eligible_for_200_case_scaleup"


def test_req1431_prototype_runtime_can_beat_baseline_without_scaleup_eligibility(
    tmp_path: Path,
) -> None:
    """REQ-VERIFY-1431: prototype-only runtime evidence is not scale-up eligible."""

    exp1397 = tmp_path / "exp1397.json"
    exp1428 = tmp_path / "exp1428.json"
    exp1430 = tmp_path / "exp1430.json"
    output = tmp_path / "exp1431.json"
    _write(exp1397, _exp1397_source())
    _write(exp1428, _exp1428(runtime_mode="prototype_injected_schema_generator_no_live_sota_inference"))
    _write(exp1430, _exp1430())

    artifact = mod.run_experiment(
        project_root=tmp_path,
        run_date="20260506",
        exp1397_path=exp1397,
        exp1428_path=exp1428,
        exp1430_path=exp1430,
        output_path=output,
    )

    assert artifact["status"] == "complete"
    assert artifact["beats_exp1419_baseline"] is True
    assert artifact["runtime_evidence_allows_headline_scaleup"] is False
    assert artifact["eligible_for_200_case_scaleup"] is False
    assert artifact["honest_verdict"] == (
        "complete_micro_validation_beats_exp1419_baseline_prototype_no_headline_scaleup"
    )


def test_req1431_blocks_when_source_has_too_few_rows(tmp_path: Path) -> None:
    """REQ-VERIFY-1431: a complete micro validation requires 50 source cases."""

    exp1397 = tmp_path / "exp1397.json"
    exp1428 = tmp_path / "exp1428.json"
    exp1430 = tmp_path / "exp1430.json"
    output = tmp_path / "exp1431.json"
    source = _exp1397_source()
    source["scheduler_rows"] = source["scheduler_rows"][:49]
    _write(exp1397, source)
    _write(exp1428, _exp1428())
    _write(exp1430, _exp1430())

    artifact = mod.run_experiment(
        project_root=tmp_path,
        run_date="20260506",
        exp1397_path=exp1397,
        exp1428_path=exp1428,
        exp1430_path=exp1430,
        output_path=output,
    )

    assert artifact["status"] == "blocked"
    assert artifact["blocker"] == "source_case_count_below_50"
    assert artifact["cases_evaluated"] == 49
    assert artifact["eligible_for_200_case_scaleup"] is False


def test_req1431_validate_artifact_rejects_bad_terminal_shapes() -> None:
    """REQ-VERIFY-1431: artifact schema validates required fields and statuses."""

    with pytest.raises(AssertionError, match="missing required fields"):
        mod.validate_artifact({})

    bad_status = dict.fromkeys(mod.REQUIRED_ARTIFACT_FIELDS, None)
    bad_status.update({"status": "weird", "eligible_for_200_case_scaleup": False})
    with pytest.raises(AssertionError, match="unsupported status"):
        mod.validate_artifact(bad_status)

    blocked = dict.fromkeys(mod.REQUIRED_ARTIFACT_FIELDS, None)
    blocked.update(
        {
            "status": "blocked",
            "eligible_for_200_case_scaleup": True,
            "honest_verdict": "bad",
        }
    )
    with pytest.raises(AssertionError, match="blocked artifact cannot be scale-up eligible"):
        mod.validate_artifact(blocked)


def test_req1431_helper_edges_are_deterministic() -> None:
    """REQ-VERIFY-1431: low-level helpers handle absent or malformed data."""

    assert mod._rate(1, 0) == pytest.approx(0.0)
    assert mod._float_or_zero("bad") == pytest.approx(0.0)
    assert mod._rows("not rows") == []
    assert mod._selected_repair_case_ids({"case_selections": "not rows"}) == set()
    assert mod._runtime_evidence_allows_headline_scaleup(
        {"executor_runtime_mode": "live_local_sota_gguf"},
        {"selector_scoring_mode": "prmv1_checkpoint"},
    )
    assert not mod._runtime_evidence_allows_headline_scaleup(
        {"executor_runtime_mode": "cpu_smoke"},
        {"honest_verdict": "complete"},
    )
    assert mod._sample_boolean_rate("not rows", {"case"}, key="parseable", fallback=0.75) == (
        pytest.approx(0.75)
    )
    assert mod._complete_verdict(beats_baseline=True, eligible=False, runtime_ready=True) == (
        "complete_micro_validation_beats_exp1419_baseline_not_scaleup_eligible"
    )
    assert mod._complete_verdict(beats_baseline=False, eligible=False, runtime_ready=True) == (
        "complete_micro_validation_does_not_beat_exp1419_baseline"
    )
    assert mod._model_specs({}) == mod.MODEL_SPECS


def test_req1431_gate_helper_reports_each_blocker() -> None:
    """REQ-VERIFY-1431: structured gate diagnostics identify exact blockers."""

    assert mod._structured_gate_status(
        _exp1428(),
        _exp1430(),
        repair_v2_enabled=False,
        prm_guided_selection_enabled=True,
    )["blocker"] == "repair_v2_flag_disabled"
    assert mod._structured_gate_status(
        _exp1428(),
        _exp1430(),
        repair_v2_enabled=True,
        prm_guided_selection_enabled=False,
    )["blocker"] == "prm_guided_selection_flag_disabled"
    assert mod._structured_gate_status(
        {"status": "blocked"},
        _exp1430(),
        repair_v2_enabled=True,
        prm_guided_selection_enabled=True,
    )["blocker"] == "exp1428_not_complete"
    assert mod._structured_gate_status(
        {"status": "complete", "repair_executor_v2_deployed": False},
        _exp1430(),
        repair_v2_enabled=True,
        prm_guided_selection_enabled=True,
    )["blocker"] == "exp1428_repair_v2_not_deployed"
    assert mod._structured_gate_status(
        _exp1428(),
        {"status": "blocked"},
        repair_v2_enabled=True,
        prm_guided_selection_enabled=True,
    )["blocker"] == "exp1430_not_complete"


def test_req1431_read_json_rejects_non_object_artifact(tmp_path: Path) -> None:
    """REQ-VERIFY-1431: source artifact loading requires JSON objects."""

    bad = tmp_path / "bad.json"
    bad.write_text("[]", encoding="utf-8")

    with pytest.raises(ValueError, match="artifact must be a JSON object"):
        mod._read_json(bad)
