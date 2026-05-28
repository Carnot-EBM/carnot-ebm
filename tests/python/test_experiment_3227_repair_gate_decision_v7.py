"""Tests for Exp 3227 repair-gate decision v7.

Spec refs: REQ-VERIFY-3227, SCENARIO-VERIFY-3227.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from carnot.verify import repair_gate_decision_v7 as mod


REQUIRED_FIELDS = {
    "schema_version",
    "experiment_id",
    "milestone",
    "input_artifacts",
    "receipt_ok",
    "clean_verifier_ok",
    "structured_preflight_ok",
    "blocker_list",
    "blocker_count",
    "repair_gate_state",
    "repair_ladder_allowed",
    "inference_substrate",
    "conductor_file_modified",
    "active_roadmap_modified",
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


def _write_instruction_sources(root: Path) -> None:
    _write_text(root, Path("AGENTS.md"), "Read CODEX.md before non-trivial changes.\n")
    _write_text(root, Path("CODEX.md"), "Spec First\nWrite Tests First\n")
    _write_text(root, Path("CLAUDE.md"), "Verifier authenticity discipline.\n")
    _write_text(
        root,
        mod.SPEC_REL_PATH,
        "REQ-VERIFY-3227\nSCENARIO-VERIFY-3227\n"
        "results/experiment_3227_repair_gate_decision_v7.json\n",
    )
    _write_text(root, Path("scripts/conductor_gates.py"), "def evaluate_gates(): pass\n")
    _write_json(
        root,
        mod.EXP3213_REL_PATH,
        {
            "schema_version": "carnot.repair_gate_decision.v6",
            "experiment_id": "exp3213",
            "repair_gate_state": "blocked",
            "repair_ladder_allowed": False,
            "honest_verdict": "complete: prior gate blocked",
        },
    )


def _write_clean_upstreams(
    root: Path,
    *,
    receipt: dict[str, Any] | None = None,
    verifier: dict[str, Any] | None = None,
    preflight: dict[str, Any] | None = None,
) -> None:
    _write_instruction_sources(root)
    receipt_payload = {
        "schema_version": "carnot.full_local_sota_receipt.v6",
        "experiment_id": "exp3222",
        "status": "complete",
        "clean_rerun_allowed": True,
        "cpu_fallback_detected": False,
        "substrate_classification": "full_local_sota_receipt",
        "inference_substrate": "local_sota_gguf_cuda_receipt",
        "honest_verdict": "complete: clean local SOTA receipt",
    }
    if receipt:
        receipt_payload.update(receipt)
    _write_json(root, mod.EXP3222_REL_PATH, receipt_payload)

    verifier_payload = {
        "schema_version": "carnot.clean_live_sota_verifier_rerun.v13",
        "experiment_id": "exp3225",
        "status": "complete",
        "clean_verifier_ready": True,
        "clean_verifier_state": "clean",
        "exact_labels_authoritative": True,
        "exact_verifier_types": ["context_exact_checker"],
        "row_count_scored": 4,
        "cpu_fallback_detected": False,
        "inference_substrate": "local_sota_gguf_exact_verifier",
        "honest_verdict": "complete: clean exact verifier",
    }
    if verifier:
        verifier_payload.update(verifier)
    _write_json(root, mod.EXP3225_REL_PATH, verifier_payload)

    preflight_payload = {
        "schema_version": "carnot.structured_repair_proposal_preflight.v2",
        "experiment_id": "exp3226",
        "status": "complete",
        "ready_for_repair_gate": True,
        "exact_verifier_handoff_ready": True,
        "repair_correctness_claimed": False,
        "schema_only_repair_limitation": False,
        "proposal_schema": {"type": "object"},
        "honest_verdict": "complete: structured proposals ready for exact handoff",
    }
    if preflight:
        preflight_payload.update(preflight)
    _write_json(root, mod.EXP3226_REL_PATH, preflight_payload)


def test_req_verify_3227_spec_anchor_exists() -> None:
    """REQ-VERIFY-3227: OpenSpec declares the v7 repair gate artifact."""

    spec = (mod.REPO_ROOT / mod.SPEC_REL_PATH).read_text(encoding="utf-8")

    assert "REQ-VERIFY-3227" in spec
    assert "SCENARIO-VERIFY-3227" in spec
    assert mod.OUTPUT_REL_PATH.as_posix() in spec
    assert "clean_rerun_allowed=true" in spec
    assert "clean_verifier_ready=true" in spec
    assert "ready_for_repair_gate=true" in spec


def test_scenario_verify_3227_unblocks_only_with_all_clean_inputs(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-3227: all three mandatory gates must pass to unblock."""

    _write_clean_upstreams(tmp_path)

    output = mod.write_artifact(tmp_path, output_path=Path("results/out.json"), tests_run=["focused"])
    artifact = json.loads(output.read_text(encoding="utf-8"))

    assert output == tmp_path / "results/out.json"
    assert REQUIRED_FIELDS <= set(artifact)
    assert artifact["schema_version"] == mod.SCHEMA_VERSION
    assert artifact["experiment_id"] == "exp3227"
    assert artifact["milestone"] == "2026.05.298"
    assert artifact["receipt_ok"] is True
    assert artifact["clean_verifier_ok"] is True
    assert artifact["structured_preflight_ok"] is True
    assert artifact["blocker_list"] == []
    assert artifact["blocker_count"] == 0
    assert artifact["repair_gate_state"] == "unblocked"
    assert artifact["repair_ladder_allowed"] is True
    assert artifact["inference_substrate"] == "artifact_gate_aggregation"
    assert artifact["conductor_file_modified"] is False
    assert artifact["active_roadmap_modified"] is False
    assert artifact["honest_verdict"].startswith("complete:")
    assert artifact["tests_run"] == ["focused"]
    assert {
        "exp3222_full_local_sota_receipt_v6",
        "exp3225_clean_live_sota_verifier_rerun_v13",
        "exp3226_structured_repair_proposal_preflight_v2",
    } <= {row["role"] for row in artifact["input_artifacts"]}

    mod.validate_artifact(artifact)


def test_req_verify_3227_current_missing_and_gate_skipped_inputs_stay_blocked(
    tmp_path: Path,
) -> None:
    """REQ-VERIFY-3227: missing and conductor-pre-gate-skipped inputs fail closed."""

    _write_instruction_sources(tmp_path)
    _write_json(
        tmp_path,
        mod.EXP3225_REL_PATH,
        {
            "experiment": 3225,
            "schema": "blocked_gate_check_v1",
            "status": "blocked",
            "honest_verdict": "blocked_gate_check_failed",
            "gate_check_summary": (
                "1 of 1 gate(s) failed; first failure: "
                "exp3222-full-local-sota-receipt-v6.clean_rerun_allowed"
            ),
            "blocked_at_layer": "conductor_pre_gate",
        },
    )

    artifact = mod.build_artifact(tmp_path)

    assert artifact["receipt_ok"] is False
    assert artifact["clean_verifier_ok"] is False
    assert artifact["structured_preflight_ok"] is False
    assert artifact["repair_gate_state"] == "blocked"
    assert artifact["repair_ladder_allowed"] is False
    assert artifact["blocker_count"] == len(artifact["blocker_list"])

    codes = {row["code"] for row in artifact["blocker_list"]}
    assert "missing_artifact" in codes
    assert "gate_skipped_artifact" in codes
    assert "clean_rerun_not_allowed" in codes
    assert "clean_verifier_not_ready" in codes
    assert "structured_preflight_not_ready" in codes
    assert artifact["honest_verdict"].startswith("complete:")


def test_req_verify_3227_cpu_fallback_absent_exact_verifier_and_schema_only_block(
    tmp_path: Path,
) -> None:
    """REQ-VERIFY-3227: CPU fallback, absent exact verifier, and schema-only limits block."""

    _write_clean_upstreams(
        tmp_path,
        receipt={
            "cpu_fallback_detected": True,
            "substrate_classification": "cpu_fallback_receipt_only",
        },
        verifier={
            "exact_labels_authoritative": False,
            "exact_verifier_types": [],
        },
        preflight={
            "exact_verifier_handoff_ready": False,
            "schema_only_repair_limitation": True,
            "structured_decoding_backend": "schema_only_parser",
        },
    )

    artifact = mod.build_artifact(tmp_path)

    assert artifact["receipt_ok"] is False
    assert artifact["clean_verifier_ok"] is False
    assert artifact["structured_preflight_ok"] is False
    assert artifact["repair_gate_state"] == "blocked"

    codes = {row["code"] for row in artifact["blocker_list"]}
    assert "cpu_fallback_detected" in codes
    assert "absent_exact_verifier" in codes
    assert "exact_verifier_handoff_not_ready" in codes
    assert "schema_only_repair_limitation" in codes


def test_req_verify_3227_malformed_input_is_recorded_as_blocker(tmp_path: Path) -> None:
    """REQ-VERIFY-3227: malformed mandatory JSON is recorded distinctly."""

    _write_clean_upstreams(tmp_path)
    malformed = tmp_path / mod.EXP3222_REL_PATH
    malformed.write_text("{not json\n", encoding="utf-8")

    artifact = mod.build_artifact(tmp_path)

    assert artifact["receipt_ok"] is False
    assert artifact["repair_gate_state"] == "blocked"
    assert any(row["code"] == "malformed_artifact" for row in artifact["blocker_list"])
    assert any(
        row["path"] == mod.EXP3222_REL_PATH.as_posix()
        and row["readable_json_object"] is False
        and row["error"]
        for row in artifact["input_artifacts"]
    )


def test_scenario_verify_3227_diagnostic_only_when_no_mandatory_json_readable(
    tmp_path: Path,
) -> None:
    """SCENARIO-VERIFY-3227: no readable mandatory inputs yields diagnostic-only."""

    _write_instruction_sources(tmp_path)

    artifact = mod.build_artifact(tmp_path)

    assert artifact["repair_gate_state"] == "diagnostic_only"
    assert artifact["repair_ladder_allowed"] is False
    assert artifact["receipt_ok"] is False
    assert artifact["clean_verifier_ok"] is False
    assert artifact["structured_preflight_ok"] is False
    assert {row["code"] for row in artifact["blocker_list"]} >= {
        "missing_artifact",
        "no_mandatory_gate_artifacts_readable",
    }


def test_req_verify_3227_helper_detection_variants() -> None:
    """REQ-VERIFY-3227: helper predicates catch alternate blocked evidence shapes."""

    assert mod.gate_skipped({"gated_skip": True}) is True
    assert mod.gate_skipped({"honest_verdict": "complete: no skip"}) is False
    assert mod.cpu_fallback_detected({"inference_substrate": "partial CPU fallback"}) is True
    assert mod.cpu_fallback_detected({"model_specs": [{"backend": "cuda"}]}) is False
    assert mod.exact_verifier_available({"exact_verifier_invocation_count": 1}) is True
    assert mod.exact_verifier_available({"exact_verifier_available": {"ok": ["solver"]}}) is True
    assert mod.exact_verifier_available({"exact_verifier_results": [{"row": "r1"}]}) is True
    assert mod.exact_verifier_available({"exact_verifier_types": []}) is False
    assert mod.schema_only_repair_limitation({"repair_mode": "schema_only"}) is True
    assert mod.schema_only_repair_limitation({"exact_verifier_handoff_ready": True}) is False


def test_req_verify_3227_validation_rejects_contradictory_artifacts() -> None:
    """REQ-VERIFY-3227: validation rejects impossible gate outputs."""

    artifact = {
        "schema_version": mod.SCHEMA_VERSION,
        "experiment_id": "exp3227",
        "milestone": "2026.05.298",
        "input_artifacts": [],
        "receipt_ok": True,
        "clean_verifier_ok": True,
        "structured_preflight_ok": True,
        "blocker_list": [],
        "blocker_count": 0,
        "repair_gate_state": "unblocked",
        "repair_ladder_allowed": True,
        "inference_substrate": "artifact_gate_aggregation",
        "conductor_file_modified": False,
        "active_roadmap_modified": False,
        "honest_verdict": "complete: valid",
    }

    mod.validate_artifact(artifact)
    with pytest.raises(ValueError, match="missing required fields"):
        mod.validate_artifact({k: v for k, v in artifact.items() if k != "honest_verdict"})
    with pytest.raises(ValueError, match="repair_gate_state"):
        mod.validate_artifact(artifact | {"repair_gate_state": "maybe"})
    with pytest.raises(ValueError, match="gate booleans"):
        mod.validate_artifact(artifact | {"receipt_ok": "yes"})
    with pytest.raises(ValueError, match="input_artifacts"):
        mod.validate_artifact(artifact | {"input_artifacts": "none"})
    with pytest.raises(ValueError, match="blocker_list"):
        mod.validate_artifact(artifact | {"blocker_list": "none"})
    with pytest.raises(ValueError, match="blocker_count"):
        mod.validate_artifact(artifact | {"blocker_count": 1})
    with pytest.raises(ValueError, match="unblocked"):
        mod.validate_artifact(artifact | {"repair_ladder_allowed": False})
    with pytest.raises(ValueError, match="unblocked state must not include blockers"):
        mod.validate_artifact(
            artifact | {"blocker_list": [{"code": "x"}], "blocker_count": 1}
        )
    with pytest.raises(ValueError, match="unblocked state requires all mandatory gates"):
        mod.validate_artifact(artifact | {"structured_preflight_ok": False})
    with pytest.raises(ValueError, match="blocked or diagnostic_only"):
        mod.validate_artifact(
            artifact
            | {
                "repair_gate_state": "blocked",
                "repair_ladder_allowed": True,
                "blocker_list": [{"code": "x"}],
                "blocker_count": 1,
            }
        )
    with pytest.raises(ValueError, match="must include blockers"):
        mod.validate_artifact(
            artifact
            | {
                "repair_gate_state": "blocked",
                "repair_ladder_allowed": False,
            }
        )
    with pytest.raises(ValueError, match="inference_substrate"):
        mod.validate_artifact(artifact | {"inference_substrate": "live_model"})
    with pytest.raises(ValueError, match="conductor_file_modified"):
        mod.validate_artifact(artifact | {"conductor_file_modified": True})
    with pytest.raises(ValueError, match="active_roadmap_modified"):
        mod.validate_artifact(artifact | {"active_roadmap_modified": True})
    with pytest.raises(ValueError, match="honest_verdict"):
        mod.validate_artifact(artifact | {"honest_verdict": "blocked"})
