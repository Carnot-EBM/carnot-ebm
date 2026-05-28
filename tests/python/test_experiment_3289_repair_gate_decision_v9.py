"""Tests for Exp 3289 repair gate decision v9.

Spec refs: REQ-VERIFY-3289, SCENARIO-VERIFY-3289.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from carnot.verify import repair_gate_decision_v9 as mod


GEMMA26 = "unsloth/gemma-4-26B-A4B-it-GGUF"
QWEN = "unsloth/Qwen3.6-35B-A3B-GGUF"
GEMMA31 = "unsloth/gemma-4-31B-it-GGUF"

REQUIRED_FIELDS = {
    "repair_gate_decision_v9_ready",
    "repair_gate_open",
    "garak_redteam_eval_ready",
    "clean_verifier_rerun_ready",
    "repair_gate_input_clean_enough",
    "kan_boundary_decision_ready",
    "gate_inputs",
    "blocked_reasons",
    "permitted_repair_scope",
    "random_seed",
    "reproducibility_checksum",
    "duration_s",
    "honest_verdict",
}


def _write_json(root: Path, rel_path: Path | str, payload: dict[str, Any]) -> None:
    path = root / Path(rel_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _exp3285_payload(*, ready: bool = True) -> dict[str, Any]:
    return {
        "schema_version": "carnot.full_garak_dataflip_redteam_eval.v2",
        "experiment_id": "exp3285",
        "garak_dataflip_redteam_eval_v2_ready": True,
        "garak_redteam_eval_ready": ready,
        "garak_probe_count": 90,
        "attack_success_rate": 0.311111,
        "garak_gate_passed": False,
        "dataflip_gate_passed": True,
        "blocked_reasons": ["garak_attack_success_or_error_gate_failed"],
        "model_specs": {
            "mandated_model_ids": [QWEN, GEMMA31, GEMMA26],
            "runtime": "llama_cpp_openai_compatible_rest",
        },
        "models_used": [
            {
                "model_id": GEMMA26,
                "model_path": "/models/gemma26.gguf",
                "live_target_call": True,
            }
        ],
        "honest_verdict": "complete: garak evidence ready",
    }


def _exp3287_payload(
    *,
    ready: bool = True,
    input_clean: bool = True,
    false_accept_rate: float = 0.0,
    abstention_rate: float = 0.0,
    coverage_rate: float = 1.0,
) -> dict[str, Any]:
    return {
        "schema_version": "carnot.abstention_calibrated_clean_verifier.v15",
        "experiment_id": "exp3287",
        "abstention_calibrated_clean_verifier_v15_ready": ready,
        "clean_verifier_rerun_ready": ready,
        "repair_gate_input_clean_enough": input_clean,
        "false_accept_rate": false_accept_rate,
        "false_reject_rate": 0.0,
        "abstention_rate": abstention_rate,
        "coverage_rate": coverage_rate,
        "n_eval": 20,
        "exact_checkable_row_count": 20,
        "model_specs": {
            "mandated_model_ids": [QWEN, GEMMA31, GEMMA26],
            "calibrated_abstention_policy": {
                "target_false_accept_rate": 0.0,
                "minimum_decision_coverage": 0.5,
            },
        },
        "models_used": [
            {
                "model_id": GEMMA26,
                "hf_id": GEMMA26,
                "model_path": "/models/gemma26.gguf",
                "legacy_small_model": False,
            }
        ],
        "missing_model_specs": [{"model_id": QWEN}, {"model_id": GEMMA31}],
        "gate_reasons": [],
        "honest_verdict": "complete: clean verifier ready",
    }


def _exp3288_payload(*, ready: bool = True, bounded: bool = True) -> dict[str, Any]:
    permitted = [
        "offline_failure_autopsy",
        "negative_control_regression_fixture",
        "future_kan_work_prerequisite_evidence_only",
    ]
    prohibited = [
        "prompt_injection_headline_detector",
        "repair_gate_authority",
        "standalone_garak_success_evidence",
    ]
    if not bounded:
        permitted = ["repair_gate_authority"]
        prohibited = []
    return {
        "schema_version": "carnot.kan_sidecar_failure_autopsy_boundary.v1",
        "experiment_id": "exp3288",
        "kan_failure_autopsy_ready": ready,
        "kan_boundary_decision_ready": ready,
        "prior_full_corpus_auroc": 0.475326,
        "prior_delong_noninferiority_passed": False,
        "kan_boundary_decision": "retire_from_prompt_injection_headline",
        "permitted_downstream_use": permitted,
        "prohibited_downstream_use": prohibited,
        "no_new_kan_training_or_scoring": True,
        "honest_verdict": "complete: KAN boundary ready",
    }


def _write_sources(
    root: Path,
    *,
    garak_ready: bool = True,
    clean_ready: bool = True,
    input_clean: bool = True,
    false_accept_rate: float = 0.0,
    abstention_rate: float = 0.0,
    coverage_rate: float = 1.0,
    kan_ready: bool = True,
    kan_bounded: bool = True,
) -> None:
    _write_json(root, mod.EXP3285_REL_PATH, _exp3285_payload(ready=garak_ready))
    _write_json(
        root,
        mod.EXP3287_REL_PATH,
        _exp3287_payload(
            ready=clean_ready,
            input_clean=input_clean,
            false_accept_rate=false_accept_rate,
            abstention_rate=abstention_rate,
            coverage_rate=coverage_rate,
        ),
    )
    _write_json(root, mod.EXP3288_REL_PATH, _exp3288_payload(ready=kan_ready, bounded=kan_bounded))


def test_req_verify_3289_spec_anchor_declares_gate_schema() -> None:
    """REQ-VERIFY-3289: OpenSpec declares the v9 repair gate fields."""

    spec = (mod.REPO_ROOT / mod.SPEC_REL_PATH).read_text(encoding="utf-8")

    assert "REQ-VERIFY-3289" in spec
    assert "SCENARIO-VERIFY-3289" in spec
    assert mod.OUTPUT_REL_PATH.as_posix() in spec
    for field in REQUIRED_FIELDS:
        assert field in spec


def test_scenario_verify_3289_opens_with_ready_clean_bounded_inputs(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-3289: ready Garak, clean verifier, and bounded KAN open."""

    _write_sources(tmp_path)
    _write_json(
        tmp_path,
        mod.EXP3276_REL_PATH,
        {
            "schema": "blocked_gate_check_v1",
            "experiment": 3276,
            "status": "blocked",
            "gate_check_summary": "prior garak and clean verifier failed",
        },
    )

    artifact = mod.build_artifact(
        tmp_path,
        started_s=10.0,
        now_s=13.25,
        tests_run=["SCENARIO-VERIFY-3289"],
    )

    assert REQUIRED_FIELDS <= set(artifact)
    assert artifact["repair_gate_decision_v9_ready"] is True
    assert artifact["repair_gate_open"] is True
    assert artifact["garak_redteam_eval_ready"] is True
    assert artifact["clean_verifier_rerun_ready"] is True
    assert artifact["repair_gate_input_clean_enough"] is True
    assert artifact["kan_boundary_decision_ready"] is True
    assert artifact["blocked_reasons"] == []
    assert artifact["duration_s"] == pytest.approx(3.25)
    assert artifact["tests_run"] == ["SCENARIO-VERIFY-3289"]
    assert len(artifact["reproducibility_checksum"]) == 64
    assert artifact["honest_verdict"].startswith("complete:")

    gate_inputs = artifact["gate_inputs"]
    assert gate_inputs["exp3285_garak"]["garak_redteam_eval_ready"] is True
    assert gate_inputs["exp3285_garak"]["garak_gate_passed"] is False
    assert gate_inputs["exp3285_garak"]["blocked_reasons"] == [
        "garak_attack_success_or_error_gate_failed"
    ]
    assert gate_inputs["exp3287_clean_verifier"]["false_accept_rate"] == 0.0
    assert gate_inputs["exp3288_kan_boundary"]["kan_downstream_use_bounded"] is True
    assert gate_inputs["exp3276_prior_gate"]["present"] is True

    scope = artifact["permitted_repair_scope"]
    assert scope["repair_task_id"] == "exp3290-gated-sota-repair-micro-panel-v10"
    assert scope["repair_generation_allowed"] is True
    assert scope["max_panel_cases"] == 8
    assert scope["sample_size"] == {"min_cases": 4, "max_cases": 8}
    assert scope["selected_model_ids"] == [GEMMA26]
    assert scope["model_specs"]["mandated_model_ids"] == [QWEN, GEMMA31, GEMMA26]
    assert scope["exact_verification_requirements"]["false_accept_count"] == 0
    assert scope["exact_verification_requirements"]["abstentions_recorded_separately"] is True
    assert scope["kan_boundary"]["kan_as_repair_gate_authority"] is False
    assert scope["claim_boundary"]["headline_claim_allowed"] is False
    mod.validate_artifact(artifact)

    output = mod.write_artifact(
        tmp_path,
        output_path=Path("results/out.json"),
        started_s=1.0,
        now_s=2.5,
        tests_run=["writer"],
    )
    saved = json.loads(output.read_text(encoding="utf-8"))
    assert output == tmp_path / "results/out.json"
    assert saved["repair_gate_open"] is True
    assert saved["duration_s"] == pytest.approx(1.5)
    assert saved["tests_run"] == ["writer"]


def test_req_verify_3289_blocks_precisely_on_failed_conditions(tmp_path: Path) -> None:
    """REQ-VERIFY-3289: any failed conservative condition keeps repair closed."""

    _write_sources(
        tmp_path,
        garak_ready=False,
        clean_ready=False,
        input_clean=False,
        false_accept_rate=0.25,
        abstention_rate=1.0,
        coverage_rate=0.0,
        kan_ready=False,
        kan_bounded=False,
    )

    artifact = mod.build_artifact(tmp_path)

    assert artifact["repair_gate_open"] is False
    assert artifact["permitted_repair_scope"]["repair_generation_allowed"] is False
    codes = {row["code"] for row in artifact["blocked_reasons"]}
    assert {
        "garak_redteam_eval_not_ready",
        "clean_verifier_rerun_not_ready",
        "repair_gate_input_not_clean_enough",
        "clean_verifier_false_accept_relaxation",
        "clean_verifier_abstain_all",
        "clean_verifier_no_coverage",
        "kan_boundary_decision_not_ready",
        "kan_downstream_use_unbounded",
    } <= codes
    assert artifact["honest_verdict"].startswith("complete:")
    mod.validate_artifact(artifact)


def test_req_verify_3289_missing_and_malformed_inputs_fail_closed(tmp_path: Path) -> None:
    """REQ-VERIFY-3289: missing or malformed upstream JSON is diagnosable."""

    bad_path = tmp_path / mod.EXP3285_REL_PATH
    bad_path.parent.mkdir(parents=True, exist_ok=True)
    bad_path.write_text("{not-json\n", encoding="utf-8")
    _write_json(tmp_path, mod.EXP3287_REL_PATH, _exp3287_payload())

    artifact = mod.build_artifact(tmp_path, started_s=5.0, now_s=2.0)

    assert artifact["duration_s"] == 0.0
    assert artifact["repair_gate_open"] is False
    codes = {row["code"] for row in artifact["blocked_reasons"]}
    assert "malformed_artifact" in codes
    assert "missing_artifact" in codes
    assert artifact["gate_inputs"]["exp3285_garak"]["readable"] is False
    assert artifact["gate_inputs"]["exp3288_kan_boundary"]["present"] is False
    assert mod.read_json_object(tmp_path / "missing.json").payload == {}
    list_json = tmp_path / "list.json"
    list_json.write_text("[]\n", encoding="utf-8")
    assert mod.read_json_object(list_json).error == "json root is not an object"
    assert mod.sha256_file(tmp_path / "missing.json") is None
    assert mod.bool_field({"x": True}, "x") is True
    assert mod.bool_field({"x": 1}, "x") is False
    assert mod.rate_field({"x": 0.5}, "x", default=0.0) == 0.5
    assert mod.rate_field({"x": "bad"}, "x", default=0.25) == 0.25
    assert mod.rate_field({"x": 2.0}, "x", default=0.25) == 0.25
    assert len(mod.stable_hash({"b": 1})) == 64
    mod.validate_artifact(artifact)


def test_req_verify_3289_validation_rejects_inconsistent_artifacts(tmp_path: Path) -> None:
    """REQ-VERIFY-3289: validator enforces gate and schema invariants."""

    _write_sources(tmp_path)
    artifact = mod.build_artifact(tmp_path)

    missing_required = dict(artifact)
    missing_required.pop("gate_inputs")
    with pytest.raises(ValueError, match="missing required fields"):
        mod.validate_artifact(missing_required)

    with pytest.raises(ValueError, match="repair_gate_decision_v9_ready"):
        mod.validate_artifact(artifact | {"repair_gate_decision_v9_ready": False})
    with pytest.raises(ValueError, match="gate bool"):
        mod.validate_artifact(artifact | {"repair_gate_open": "yes"})
    with pytest.raises(ValueError, match="gate_inputs"):
        mod.validate_artifact(artifact | {"gate_inputs": []})
    with pytest.raises(ValueError, match="blocked_reasons"):
        mod.validate_artifact(artifact | {"blocked_reasons": {}})
    with pytest.raises(ValueError, match="permitted_repair_scope"):
        mod.validate_artifact(artifact | {"permitted_repair_scope": []})
    with pytest.raises(ValueError, match="duration_s"):
        mod.validate_artifact(artifact | {"duration_s": -1.0})
    with pytest.raises(ValueError, match="reproducibility_checksum"):
        mod.validate_artifact(artifact | {"reproducibility_checksum": "bad"})
    with pytest.raises(ValueError, match="honest_verdict"):
        mod.validate_artifact(artifact | {"honest_verdict": "blocked"})
    with pytest.raises(ValueError, match="open gate"):
        mod.validate_artifact(artifact | {"repair_gate_open": True, "blocked_reasons": [{"x": 1}]})
    with pytest.raises(ValueError, match="open gate"):
        mod.validate_artifact(
            artifact
            | {
                "repair_gate_open": True,
                "permitted_repair_scope": {"repair_generation_allowed": False},
            }
        )
    with pytest.raises(ValueError, match="closed gate"):
        mod.validate_artifact(
            artifact
            | {
                "repair_gate_open": False,
                "blocked_reasons": [],
                "permitted_repair_scope": {"repair_generation_allowed": False},
            }
        )
    with pytest.raises(ValueError, match="closed gate"):
        mod.validate_artifact(
            artifact
            | {
                "repair_gate_open": False,
                "blocked_reasons": [{"code": "x"}],
                "permitted_repair_scope": {"repair_generation_allowed": True},
            }
        )
