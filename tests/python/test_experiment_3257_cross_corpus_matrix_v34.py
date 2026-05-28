"""Tests for Exp 3257 cross-corpus matrix v34.

Spec refs: REQ-REPORT-3257, SCENARIO-REPORT-3257.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from carnot.reporting import cross_corpus_matrix_v34_3257 as mod


REQUIRED_FIELDS = {
    "experiment_id",
    "task_id",
    "milestone",
    "inference_substrate",
    "principle_annotations",
    "matrix_v34_ready",
    "artifacts_expected",
    "artifacts_found",
    "artifacts_missing",
    "gate_blocked_artifacts",
    "runtime_receipt_status",
    "prompt_injection_status",
    "fr11_lifelong_status",
    "pdit_potts_status",
    "publication_blocker_count",
    "paper_ready",
    "next_top_gap",
    "honest_verdict",
}


def _write_json(root: Path, rel_path: Path, payload: dict[str, Any]) -> None:
    path = root / rel_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_text(root: Path, rel_path: Path, text: str) -> None:
    path = root / rel_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _gate_payload(experiment: int, title: str, summary: str) -> dict[str, Any]:
    return {
        "experiment": experiment,
        "schema": "blocked_gate_check_v1",
        "status": "blocked",
        "title": title,
        "blocked_at_layer": "conductor_pre_gate",
        "gate_check_summary": summary,
        "gates_evaluated": [
            {
                "upstream": "upstream-task",
                "artifact_field": "ready",
                "expected": True,
                "actual": None,
                "passed": False,
            }
        ],
        "honest_verdict": "blocked_gate_check_failed",
    }


def _write_dot301_sources(root: Path) -> None:
    _write_json(
        root,
        mod.EXP3246_REL_PATH,
        {
            "experiment_id": "exp3246",
            "task_id": "exp3246-archive-v300-activate-v301",
            "archive_v300_activate_v301_ready": True,
            "prior_paper_ready": False,
            "prior_publication_blocker_count": 106,
            "next_top_gap": "repair_selected_python_torch_cuda_before_exp3237",
            "honest_verdict": "complete: archive_v300_activate_v301_ready=true",
        },
    )
    _write_json(
        root,
        mod.EXP3247_REL_PATH,
        {
            "experiment_id": "exp3247",
            "task_id": "exp3247-selected-python-cuda-root-cause-surgery-v1",
            "preconditions_checked": True,
            "cuda_root_cause_class": "cuda_bindings_runtime_failure",
            "selected_python_cuda_repaired_candidate": False,
            "next_smoke_allowed": False,
            "selected_python_torch_cuda_available_after": False,
            "cuda_bindings_device_count_after": 0,
            "recommended_next_task": "keep_exp3248_blocked_repair_cuda_runtime",
            "honest_verdict": "complete: next_smoke_allowed=false",
        },
    )
    _write_json(
        root,
        mod.EXP3248_REL_PATH,
        _gate_payload(
            3248,
            "Isolated selected-Python CUDA smoke v2 gated on root-cause surgery",
            "first failure: exp3247.next_smoke_allowed (actual=False == expected=True)",
        ),
    )
    _write_json(
        root,
        mod.EXP3250_REL_PATH,
        _gate_payload(
            3250,
            "Mandated SOTA GGUF receipt v8 gated on llama.cpp CUDA",
            "upstream artifact not found for exp3249",
        ),
    )
    _write_json(
        root,
        mod.EXP3251_REL_PATH,
        {
            "experiment_id": "exp3251",
            "task_id": "exp3251-prompt-injection-v4-constraint-tax-manifest-v2",
            "v4_manifest_v2_ready": True,
            "constraint_tax_control_plan_ready": True,
            "garak_config_ready": True,
            "no_llm_invoked": True,
            "no_new_teacher_labeling": True,
            "no_kan_training": True,
            "teacher_label_shard_contract": {
                "same_examples_required": True,
                "paired_arm_ids": ["free_reasoning", "schema_constrained"],
            },
            "honest_verdict": "complete: v4_manifest_v2_ready=true",
        },
    )
    _write_json(
        root,
        mod.EXP3253_REL_PATH,
        _gate_payload(
            3253,
            "Prompt-injection KAN shard train/eval v2 with constraint-tax guardrail",
            "upstream artifact not found for exp3252",
        ),
    )
    _write_json(
        root,
        mod.EXP3255_REL_PATH,
        {
            "experiment_id": "exp3255",
            "task_id": "exp3255-fr11-lifelong-failure-memory-retention-audit-v1",
            "continuous_self_learning_task": True,
            "fr11_controller_update_ready": True,
            "lifelong_eval_ready": True,
            "failure_trace_count": 44,
            "heldout_replay_count": 9,
            "retention_score": 1.0,
            "adaptation_score": 1.0,
            "forgetting_score": 1.0,
            "negative_control_regression_count": 0,
            "doomed_rerun_avoidance_count": 43,
            "model_weight_update_claimed": False,
            "no_new_llm_invoked": True,
            "honest_verdict": "complete: fr11 lifelong audit ready=true",
        },
    )
    _write_json(
        root,
        mod.EXP3256_REL_PATH,
        {
            "experiment_id": "exp3256",
            "task_id": "exp3256-pdit-potts-multistate-sampler-diagnostic-v1",
            "pdit_potts_mapping_ready": True,
            "exact_fallback_preserved": True,
            "hardware_speedup_claim_allowed": False,
            "retired_pimi_scope_reopened": False,
            "thrml_scaling_sweep_reopened": False,
            "candidate_verifier_row_types": [{"row_type": "opencomputer", "q": 4}],
            "q_state_energy_mapping": [{"row_type": "opencomputer", "q": 4}],
            "honest_verdict": "complete: CPU/simulation-only mapping ready",
        },
    )
    _write_text(
        root,
        mod.CONDUCTOR_LOG_REL_PATH,
        "\n".join(
            [
                "| 2026-05-28 07:18 UTC | llama.cpp CUDA receipt smoke v3 gated on selected- | GATE_BLOCK | Pre-emptive skip: upstream retired (exp3248-isolated-cuda-selected-python-smoke-v2) |",
                "| 2026-05-28 07:51 UTC | Prompt-injection teacher-label shard v2 gated on S | GATE_BLOCK | Pre-emptive skip: upstream retired (exp3250-sota-gguf-receipt-v8) |",
                "| 2026-05-28 07:57 UTC | DCCD/SEVerA structured proposal preflight v2 gated | GATE_BLOCK | Pre-emptive skip: upstream retired (exp3250-sota-gguf-receipt-v8) |",
            ]
        )
        + "\n",
    )


def test_req_report_3257_spec_anchor_exists() -> None:
    """REQ-REPORT-3257: OpenSpec declares matrix v34 before implementation."""

    spec = (mod.REPO_ROOT / "openspec/capabilities/research-reporting/spec.md").read_text(
        encoding="utf-8"
    )

    assert "REQ-REPORT-3257" in spec
    assert "SCENARIO-REPORT-3257" in spec
    assert mod.OUTPUT_REL_PATH.as_posix() in spec
    assert Path(mod.__file__).exists()


def test_scenario_report_3257_builds_v34_from_dot301_artifacts(tmp_path: Path) -> None:
    """SCENARIO-REPORT-3257: v34 records found, missing, and gate-blocked evidence."""

    _write_dot301_sources(tmp_path)

    artifact = mod.build_artifact(tmp_path, started_s=4.0, now_s=7.5)
    found = {row["experiment_id"] for row in artifact["artifacts_found"]}
    missing = {row["experiment_id"] for row in artifact["artifacts_missing"]}
    gated = {row["experiment_id"] for row in artifact["gate_blocked_artifacts"]}

    assert REQUIRED_FIELDS <= artifact.keys()
    assert artifact["experiment_id"] == "exp3257"
    assert artifact["task_id"] == "exp3257-cross-corpus-matrix-v34"
    assert artifact["milestone"] == "2026.05.301"
    assert artifact["inference_substrate"] == "aggregation_from_upstream_artifacts"
    assert artifact["matrix_v34_ready"] is True
    assert artifact["duration_s"] == pytest.approx(3.5)
    assert len(artifact["artifacts_expected"]) == 11
    assert found == {"exp3246", "exp3247", "exp3248", "exp3250", "exp3251", "exp3253", "exp3255", "exp3256"}
    assert missing == {"exp3249", "exp3252", "exp3254"}
    assert gated == {"exp3248", "exp3249", "exp3250", "exp3252", "exp3253", "exp3254"}

    runtime = artifact["runtime_receipt_status"]
    assert runtime["selected_python_cuda"]["next_smoke_allowed"] is False
    assert runtime["selected_python_cuda"]["state"] == "blocked_root_cause_surgery_next_smoke_not_allowed"
    assert runtime["llama_cpp_cuda"]["artifact_status"] == "gate_blocked"
    assert runtime["llama_cpp_cuda"]["llama_cpp_cuda_receipt_ready"] is False
    assert runtime["sota_gguf_receipt"]["artifact_status"] == "gate_blocked"
    assert runtime["sota_gguf_receipt"]["sota_gguf_receipt_ready"] is False
    assert runtime["clean_rerun_allowed"] is False
    assert runtime["receipt_chain_ready"] is False

    prompt = artifact["prompt_injection_status"]
    assert prompt["manifest_v2"]["v4_manifest_v2_ready"] is True
    assert prompt["constraint_tax_diagnostic"]["status"] == "plan_ready_no_measurement"
    assert prompt["teacher_labels"]["artifact_status"] == "gate_blocked"
    assert prompt["kan_shard"]["artifact_status"] == "gate_blocked"
    assert prompt["repair_proposal_preflight"]["artifact_status"] == "gate_blocked"
    assert prompt["publication_evidence_ready"] is False

    fr11 = artifact["fr11_lifelong_status"]
    assert fr11["lifelong_eval_ready"] is True
    assert fr11["retention_score"] == 1.0
    assert fr11["adaptation_score"] == 1.0
    assert fr11["forgetting_score"] == 1.0
    assert fr11["model_weight_update_claimed"] is False

    pdit = artifact["pdit_potts_status"]
    assert pdit["pdit_potts_mapping_ready"] is True
    assert pdit["exact_fallback_preserved"] is True
    assert pdit["hardware_speedup_claim_allowed"] is False

    assert artifact["publication_blocker_count"] == 106
    assert artifact["paper_ready"] is False
    assert artifact["next_top_gap"] == "keep_exp3248_blocked_repair_cuda_runtime"
    assert artifact["honest_verdict"].startswith("complete:")
    assert "paper_ready=false" in artifact["honest_verdict"]
    mod.validate_artifact(artifact)


def test_req_report_3257_write_artifact_and_fail_closed_empty_sources(tmp_path: Path) -> None:
    """REQ-REPORT-3257: writer persists output and empty evidence is not paper-ready."""

    _write_dot301_sources(tmp_path)
    output = mod.write_artifact(tmp_path, started_s=1.0, now_s=2.0)
    saved = json.loads(output.read_text(encoding="utf-8"))

    assert output == tmp_path / mod.OUTPUT_REL_PATH
    assert saved["matrix_v34_ready"] is True

    empty = mod.build_artifact(tmp_path / "empty", started_s=1.0, now_s=1.0)
    assert empty["matrix_v34_ready"] is False
    assert len(empty["artifacts_found"]) == 0
    assert len(empty["artifacts_missing"]) == 11
    assert empty["publication_blocker_count"] == 106
    assert empty["paper_ready"] is False
    assert "exp3246 archive/activation artifact is missing or not ready" in empty["invariant_violations"]
    mod.validate_artifact(empty)


def test_req_report_3257_defensive_helpers_and_validation(tmp_path: Path) -> None:
    """REQ-REPORT-3257: helper paths classify malformed evidence without overclaiming."""

    assert mod.read_json_object(tmp_path / "missing.json") == {}
    bad = tmp_path / "bad.json"
    bad.write_text("{bad", encoding="utf-8")
    assert mod.read_json_object(bad) == {}
    non_object = tmp_path / "list.json"
    non_object.write_text("[]", encoding="utf-8")
    assert mod.read_json_object(non_object) == {}
    assert mod.sha256_file(tmp_path / "missing.json") is None
    assert mod._read_text(tmp_path / "missing.log") == ""
    assert mod._duration(3.0, 2.0) == 0.0
    assert mod._as_mapping({"a": 1}) == {"a": 1}
    assert mod._as_mapping([]) == {}
    assert mod._bool_value(True) is True
    assert mod._bool_value("true") is False
    assert mod._int_value(3) == 3
    assert mod._int_value(True) == 0
    assert mod._normal_status("gate-blocked") == "gate_blocked"
    assert mod._normal_status("weird") == "missing"

    gate = mod._gate_evidence(
        mod.EXPECTED_SOURCES[3],
        "| llama.cpp CUDA receipt smoke v3 gated on selected- | GATE_BLOCK | skip |\n",
        {},
    )
    assert gate["status"] == "gate_blocked"
    assert gate["source"] == mod.CONDUCTOR_LOG_REL_PATH.as_posix()
    assert mod._gate_evidence(mod.EXPECTED_SOURCES[3], "", {}) == {"status": "absent"}
    assert mod._gate_evidence(mod.EXPECTED_SOURCES[3], "unrelated GATE_BLOCK line", {}) == {
        "status": "absent"
    }
    assert mod._is_gate_blocked({"schema": "blocked_gate_check_v1"}) is True
    assert mod._is_gate_blocked({"blocked_at_layer": "conductor_pre_gate"}) is True
    assert mod._is_gate_blocked({"honest_verdict": "blocked_gate_check_failed"}) is True

    assert mod._status_for_source(mod.EXPECTED_SOURCES[0], {"archive_v300_activate_v301_ready": True}) == "complete"
    assert mod._status_for_source(mod.EXPECTED_SOURCES[1], {"next_smoke_allowed": False}) == "blocked"
    assert mod._status_for_source(mod.EXPECTED_SOURCES[2], {"schema": "blocked_gate_check_v1"}) == "gate_blocked"
    assert mod._status_for_source(mod.EXPECTED_SOURCES[3], {}) == "missing"

    rows = [
        {"experiment_id": "exp3248", "status": "complete"},
        {"experiment_id": "exp3249", "status": "blocked"},
    ]
    assert mod._row_status(rows, "exp3249") == "blocked"
    assert mod._row_status(rows, "exp9999") == "missing"
    assert mod._required_evidence_exists(
        {"receipt_chain_ready": True},
        {
            "teacher_labels": {"teacher_label_shard_ready": True},
            "kan_shard": {"train_eval_completed": True},
            "repair_proposal_preflight": {"structured_proposal_preflight_ready": True},
        },
    ) is True
    assert mod._required_evidence_exists({"receipt_chain_ready": False}, {}) is False

    assert mod._next_top_gap(
        {"receipt_chain_ready": False, "next_action": "runtime_gap"},
        {},
        {},
        {},
    ) == "runtime_gap"
    assert mod._next_top_gap(
        {"receipt_chain_ready": True},
        {"publication_evidence_ready": False},
        {},
        {},
    ) == "prompt_injection_teacher_labels_and_kan_shard_after_sota_receipt"
    assert mod._next_top_gap(
        {"receipt_chain_ready": True},
        {
            "publication_evidence_ready": True,
            "repair_proposal_preflight": {"structured_proposal_preflight_ready": False},
        },
        {},
        {},
    ) == "dccd_severa_preflight_after_clean_sota_receipt"
    assert mod._next_top_gap(
        {"receipt_chain_ready": True},
        {
            "publication_evidence_ready": True,
            "repair_proposal_preflight": {"structured_proposal_preflight_ready": True},
        },
        {"lifelong_eval_ready": False},
        {},
    ) == "fr11_lifelong_retention_audit"
    assert mod._next_top_gap(
        {"receipt_chain_ready": True},
        {
            "publication_evidence_ready": True,
            "repair_proposal_preflight": {"structured_proposal_preflight_ready": True},
        },
        {"lifelong_eval_ready": True},
        {"pdit_potts_mapping_ready": False},
    ) == "pdit_potts_diagnostic_mapping"
    assert mod._next_top_gap(
        {"receipt_chain_ready": True},
        {
            "publication_evidence_ready": True,
            "repair_proposal_preflight": {"structured_proposal_preflight_ready": True},
        },
        {"lifelong_eval_ready": True},
        {"pdit_potts_mapping_ready": True},
    ) == "publication_blocker_retirement_review"

    artifact = mod.build_artifact(tmp_path, started_s=1.0, now_s=1.0)
    with pytest.raises(ValueError, match="missing required fields"):
        mod.validate_artifact({})
    with pytest.raises(ValueError, match="experiment_id"):
        mod.validate_artifact(artifact | {"experiment_id": "bad"})
    with pytest.raises(ValueError, match="task_id"):
        mod.validate_artifact(artifact | {"task_id": "bad"})
    with pytest.raises(ValueError, match="milestone"):
        mod.validate_artifact(artifact | {"milestone": "bad"})
    with pytest.raises(ValueError, match="inference_substrate"):
        mod.validate_artifact(artifact | {"inference_substrate": "bad"})
    with pytest.raises(ValueError, match="honest_verdict"):
        mod.validate_artifact(artifact | {"honest_verdict": "blocked"})
    with pytest.raises(ValueError, match="publication_blocker_count"):
        mod.validate_artifact(artifact | {"publication_blocker_count": -1})
    with pytest.raises(ValueError, match="paper_ready cannot be true"):
        mod.validate_artifact(artifact | {"paper_ready": True, "publication_blocker_count": 1})
