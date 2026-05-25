"""Tests for Exp 3042 repair promotion reconciliation.

Spec refs: REQ-REPORT-3042, SCENARIO-REPORT-3042.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from carnot.reporting import repair_promotion_reconciliation_3042 as mod


REQUIRED_FIELDS = {
    "repair_reconciliation_ready",
    "repair_promotion_candidate",
    "repair_claim_status",
    "accepted_source_artifacts",
    "remaining_blockers",
    "aggregation_false_positives_removed",
    "repair_delta_summary",
    "inference_substrate",
    "honest_verdict",
}

FORBIDDEN_TOP_LEVEL = {
    "model_specs",
    "target_model",
    "cuda",
    "CUDA",
    "gguf",
    "GGUF",
    "gpu_inventory",
    "headline_models_used",
    "live_model_metadata",
}


def _write_json(root: Path, rel_path: Path, payload: dict[str, Any]) -> None:
    path = root / rel_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _flag(kind: str, severity: str = "critical") -> dict[str, str]:
    return {"kind": kind, "severity": severity, "detail": f"{kind} fixture detail"}


def _row(
    row_id: str,
    classification: str,
    source_artifact: str,
    source_field: str,
    *,
    experiment_id: str = "exp3028",
    blocking: bool = True,
    rationale: str = "repair fixture blocker",
    evidence: Any | None = None,
) -> dict[str, Any]:
    return {
        "row_id": row_id,
        "classification": classification,
        "blocking": blocking,
        "source_artifact": source_artifact,
        "source_field": source_field,
        "experiment_id": experiment_id,
        "rationale": rationale,
        "evidence": evidence if evidence is not None else {"fixture": row_id},
    }


def _exp3028(**overrides: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "artifact": "experiment_3028_sota_repair_clean_methodology_rerun_v2",
        "clean_repair_rerun_ready": True,
        "clean_repair_claim_promotable_candidate": True,
        "repair_controller_clean": True,
        "n_tasks": 24,
        "n_live_transcripts": 24,
        "pass_at_1_delta": 0.375,
        "pass_at_k_delta": 0.375,
        "syntax_failure_rate_delta": 0.0,
        "schema_failure_rate_delta": 0.0,
        "false_accept_delta": 0.0,
        "tautology_gate_clean": True,
        "intent_drift_count": 0,
        "candidate_intent_drift_count": 14,
        "legacy_smoke_only_used": False,
        "reproducibility_checksum": "f" * 64,
        "model_specs": [
            {
                "hf_id": "unsloth/gemma-4-26B-A4B-it-GGUF",
                "model_path": "/models/gemma.gguf",
                "checksum": "a" * 64,
            }
        ],
        "headline_models_used": ["unsloth/gemma-4-26B-A4B-it-GGUF"],
        "precondition_checks": [
            {"resource": "exp3015_acceptance_controller", "available": True},
            {"resource": "complete_transcript_reconstruction", "available": True},
        ],
        "inference_substrate": {
            "kind": "clean_repair_reconstruction",
            "live_repair_generation_run": False,
            "model_load_attempted": False,
            "recorded_before_model_load": True,
        },
        "source_artifacts": [
            {"path": "results/experiment_3015.json", "present": True, "sha256": "b" * 64}
        ],
        "honest_verdict": "complete: clean_repair_rerun_ready=true; n_tasks=24",
    }
    payload.update(overrides)
    return payload


def _exp3029(status: str = "bounded") -> dict[str, Any]:
    bounded_claims = []
    promotable_claims = []
    if status == "bounded":
        bounded_claims = [
            {
                "claim_id": "exp3028_clean_repair_candidate",
                "classification": "bounded",
                "blockers": [
                    "matrix repair row is flagged",
                    "capstone repair decision is not promotable",
                ],
            }
        ]
    if status == "clean_candidate":
        promotable_claims = [
            {
                "claim_id": "exp3028_clean_repair_candidate",
                "classification": "promotable",
                "blockers": [],
            }
        ]
    return {
        "artifact": "experiment_3029_repair_promotion_boundary_audit_v2",
        "repair_promotion_boundary_ready": True,
        "repair_claim_status": status,
        "promotable_claims": promotable_claims,
        "bounded_claims": bounded_claims,
        "retired_or_blocked_claims": [],
        "inference_substrate": {
            "kind": "aggregation_from_upstream_artifacts",
            "no_live_llm_inference": True,
        },
        "honest_verdict": f"complete: repair_claim_status={status}",
    }


def _matrix_v18(exp3028_status: str = "flagged", exp3029_status: str = "flagged") -> dict[str, Any]:
    return {
        "artifact": "experiment_3038_cross_corpus_matrix_v18",
        "matrix_v18_ready": True,
        "rows_total": 14,
        "matrix_rows": [
            {
                "experiment_id": "exp3028",
                "status": exp3028_status,
                "task_class": "repair_rerun",
                "repair_claim_status": "clean_candidate_flagged",
                "upstream_flags": ["TAUTOLOGY:critical", "METHODOLOGY_MISSING:warn"],
                "summary": {"clean_repair_rerun_ready": True},
            },
            {
                "experiment_id": "exp3029",
                "status": exp3029_status,
                "task_class": "repair_boundary_audit",
                "repair_claim_status": "bounded",
                "upstream_flags": ["DURATION_TOO_SHORT:critical"],
                "summary": {"repair_claim_status": "bounded"},
            },
        ],
        "inference_substrate": {"kind": "aggregation_from_upstream_artifacts"},
        "honest_verdict": "complete: matrix_v18_ready=true",
    }


def _capstone_v284(repair_status: str = "bounded") -> dict[str, Any]:
    passed = repair_status == "clean_candidate"
    return {
        "artifact": "experiment_3039_capstone_v284",
        "capstone_ready": True,
        "paper_ready": False,
        "repair_claim_status": repair_status,
        "blockers_remaining": [
            {"area": "repair", "status": repair_status, "next_action": "reconcile repair"}
        ]
        if not passed
        else [],
        "paper_ready_checks": [
            {
                "check": "repair_promotable",
                "passed": passed,
                "reason": f"repair_claim_status={repair_status}",
            }
        ],
        "inference_substrate": {"kind": "aggregation_from_upstream_artifacts"},
        "honest_verdict": f"complete: capstone_ready=true; repair_claim_status={repair_status}",
    }


def _exp3041(with_repair_blockers: bool = True) -> dict[str, Any]:
    true_rows = []
    missing_rows = []
    unresolved_rows = []
    if with_repair_blockers:
        true_rows.append(
            _row(
                "exp3028:adversarial_flags",
                "true_blocker",
                mod.EXP3028_REL_PATH.as_posix(),
                "corrigendum_pending",
                evidence=[_flag("TAUTOLOGY"), _flag("DURATION_TOO_SHORT")],
            )
        )
        missing_rows.append(
            _row(
                "exp3028:methodology_missing",
                "missing_metadata",
                mod.EXP3028_REL_PATH.as_posix(),
                "corrigendum_pending[METHODOLOGY_MISSING]",
                evidence=[_flag("METHODOLOGY_MISSING", "warn")],
            )
        )
        unresolved_rows.extend(
            [
                _row(
                    "exp3029:exp3028_clean_repair_candidate",
                    "unresolved_bound",
                    mod.EXP3029_REL_PATH.as_posix(),
                    "bounded_claims[exp3028_clean_repair_candidate]",
                    experiment_id="exp3029",
                ),
                _row(
                    "capstone:repair_bounded",
                    "unresolved_bound",
                    mod.CAPSTONE_V284_REL_PATH.as_posix(),
                    "blockers_remaining[repair]",
                    experiment_id="exp3039",
                    rationale="Capstone repair status remains bounded.",
                ),
                _row(
                    "capstone:repair_promotable_check_failed",
                    "unresolved_bound",
                    mod.CAPSTONE_V284_REL_PATH.as_posix(),
                    "paper_ready_checks[repair_promotable]",
                    experiment_id="exp3039",
                ),
            ]
        )
    return {
        "artifact": "experiment_3041_matrix_capstone_adversarial_flag_hygiene_v1",
        "flag_hygiene_ready": True,
        "aggregation_false_positive_rows": [
            _row(
                "exp3029:top_level_aggregation_flags",
                "aggregation_false_positive",
                mod.EXP3029_REL_PATH.as_posix(),
                "corrigendum_pending",
                experiment_id="exp3029",
                blocking=False,
                evidence={"flags": [_flag("DURATION_TOO_SHORT")]},
            ),
            _row(
                "exp3038:top_level_aggregation_flags",
                "aggregation_false_positive",
                mod.MATRIX_V18_REL_PATH.as_posix(),
                "corrigendum_pending",
                experiment_id="exp3038",
                blocking=False,
                evidence={"flags": [_flag("METHODOLOGY_MISSING", "warn")]},
            ),
        ],
        "true_blocker_rows": true_rows,
        "missing_metadata_rows": missing_rows,
        "unresolved_bound_rows": unresolved_rows,
        "hardware_blocked_rows": [
            _row(
                "exp3034:hardware_blocked",
                "hardware_blocked",
                mod.MATRIX_V18_REL_PATH.as_posix(),
                "matrix_rows[exp3034].status",
                experiment_id="exp3034",
                rationale="hardware fixture unrelated to repair",
            )
        ],
        "gate_skipped_rows": [
            _row(
                "exp3036:gate_skipped",
                "gate_skipped",
                mod.MATRIX_V18_REL_PATH.as_posix(),
                "matrix_rows[exp3036].status",
                experiment_id="exp3036",
                rationale="gate fixture unrelated to repair",
            )
        ],
        "inference_substrate": {
            "kind": "aggregation_from_upstream_artifacts",
            "executes_models": False,
            "executes_hardware": False,
        },
        "honest_verdict": "complete: flag_hygiene_ready=true",
    }


def _write_sources(
    root: Path,
    *,
    exp3028: dict[str, Any] | None = None,
    exp3029: dict[str, Any] | None = None,
    matrix: dict[str, Any] | None = None,
    capstone: dict[str, Any] | None = None,
    hygiene: dict[str, Any] | None = None,
) -> None:
    _write_json(root, mod.EXP3028_REL_PATH, exp3028 or _exp3028())
    _write_json(root, mod.EXP3029_REL_PATH, exp3029 or _exp3029())
    _write_json(root, mod.MATRIX_V18_REL_PATH, matrix or _matrix_v18())
    _write_json(root, mod.CAPSTONE_V284_REL_PATH, capstone or _capstone_v284())
    _write_json(root, mod.EXP3041_REL_PATH, hygiene or _exp3041())


def test_req_report_3042_spec_anchor_and_script_exist() -> None:
    """REQ-REPORT-3042: OpenSpec declares the reconciliation artifact first."""

    spec = (mod.REPO_ROOT / "openspec/capabilities/research-reporting/spec.md").read_text(
        encoding="utf-8"
    )

    assert "REQ-REPORT-3042" in spec
    assert "SCENARIO-REPORT-3042" in spec
    assert mod.OUTPUT_REL_PATH.as_posix() in spec
    assert mod.SCRIPT_REL_PATH.exists()


def test_scenario_report_3042_keeps_repair_bounded_after_confirmed_false_positive_cleanup(
    tmp_path: Path,
) -> None:
    """SCENARIO-REPORT-3042: repair blockers survive aggregation flag cleanup."""

    _write_sources(tmp_path)

    artifact = mod.build_artifact(tmp_path, started_s=4.0, now_s=7.25)
    blocker_ids = {row["row_id"] for row in artifact["remaining_blockers"]}
    removed_ids = {row["row_id"] for row in artifact["aggregation_false_positives_removed"]}

    assert REQUIRED_FIELDS <= artifact.keys()
    assert artifact["repair_reconciliation_ready"] is True
    assert artifact["repair_promotion_candidate"] is False
    assert artifact["repair_claim_status"] == "bounded"
    assert artifact["duration_s"] == pytest.approx(3.25)
    assert artifact["honest_verdict"].startswith("complete: repair_claim_status=bounded")
    assert FORBIDDEN_TOP_LEVEL.isdisjoint(artifact)
    assert artifact["inference_substrate"] == {
        "kind": "aggregation_from_upstream_artifacts",
        "source": "checked_in_artifacts",
        "executes_models": False,
        "executes_hardware": False,
        "executes_conductor": False,
        "no_live_llm_inference": True,
    }

    assert removed_ids == {
        "exp3029:top_level_aggregation_flags",
        "exp3038:top_level_aggregation_flags",
    }
    assert "exp3029:top_level_aggregation_flags" not in blocker_ids
    assert {
        "exp3028:adversarial_flags",
        "exp3028:methodology_missing",
        "exp3029:exp3028_clean_repair_candidate",
        "capstone:repair_bounded",
        "capstone:repair_promotable_check_failed",
    } <= blocker_ids
    assert "exp3034:hardware_blocked" not in blocker_ids
    assert artifact["repair_delta_summary"] == {
        "n_tasks": 24,
        "n_live_transcripts": 24,
        "pass_at_1_delta": 0.375,
        "pass_at_k_delta": 0.375,
        "syntax_failure_rate_delta": 0.0,
        "schema_failure_rate_delta": 0.0,
        "false_accept_delta": 0.0,
        "intent_drift_count": 0,
        "candidate_intent_drift_count": 14,
    }
    assert {row["path"] for row in artifact["accepted_source_artifacts"]} == {
        mod.EXP3028_REL_PATH.as_posix(),
        mod.EXP3029_REL_PATH.as_posix(),
        mod.MATRIX_V18_REL_PATH.as_posix(),
        mod.CAPSTONE_V284_REL_PATH.as_posix(),
        mod.EXP3041_REL_PATH.as_posix(),
    }
    assert all(row["source_artifact"] and row["source_field"] for row in artifact["remaining_blockers"])


def test_req_report_3042_can_mark_clean_candidate_without_final_capstone_promotion(
    tmp_path: Path,
) -> None:
    """REQ-REPORT-3042: candidate promotion is separate from paper promotion."""

    _write_sources(
        tmp_path,
        exp3029=_exp3029(status="clean_candidate"),
        matrix=_matrix_v18(exp3028_status="clean", exp3029_status="clean"),
        capstone=_capstone_v284(repair_status="clean_candidate"),
        hygiene=_exp3041(with_repair_blockers=False),
    )

    artifact = mod.build_artifact(tmp_path)

    assert artifact["repair_reconciliation_ready"] is True
    assert artifact["repair_claim_status"] == "clean_candidate"
    assert artifact["repair_promotion_candidate"] is True
    assert artifact["remaining_blockers"] == []
    assert artifact["prior_repair_status"] == {
        "exp3029_repair_claim_status": "clean_candidate",
        "matrix_v18_exp3028_status": "clean",
        "matrix_v18_exp3029_status": "clean",
        "capstone_repair_claim_status": "clean_candidate",
        "capstone_paper_ready": False,
    }


def test_req_report_3042_blocks_on_missing_required_sources(tmp_path: Path) -> None:
    """REQ-REPORT-3042: absent source artifacts block the reconciliation precondition."""

    _write_json(tmp_path, mod.EXP3028_REL_PATH, _exp3028())

    artifact = mod.build_artifact(tmp_path)

    assert artifact["repair_reconciliation_ready"] is False
    assert artifact["repair_claim_status"] == "blocked"
    assert artifact["repair_promotion_candidate"] is False
    assert artifact["honest_verdict"].startswith("blocked_required_source_missing:")
    assert {row["experiment_id"] for row in artifact["remaining_blockers"]} == {
        "exp3029",
        "exp3038",
        "exp3039",
        "exp3041",
    }


def test_req_report_3042_blocks_when_exp3028_clean_evidence_is_incomplete(
    tmp_path: Path,
) -> None:
    """REQ-REPORT-3042: Exp 3028 source gaps and regressions stay blocking."""

    _write_sources(
        tmp_path,
        exp3028=_exp3028(
            clean_repair_rerun_ready=False,
            repair_controller_clean=False,
            clean_repair_claim_promotable_candidate=False,
            model_specs=[],
            n_live_transcripts=12,
            pass_at_1_delta=None,
            pass_at_k_delta=-0.25,
            false_accept_delta=0.125,
            syntax_failure_rate_delta=0.25,
            schema_failure_rate_delta=0.5,
            intent_drift_count=1,
            tautology_gate_clean=False,
            legacy_smoke_only_used=True,
            reproducibility_checksum="",
            precondition_checks=[],
            inference_substrate={"live_repair_generation_run": True},
        ),
        hygiene=_exp3041(with_repair_blockers=False),
    )

    artifact = mod.build_artifact(tmp_path)
    blocker_ids = {row["row_id"] for row in artifact["remaining_blockers"]}

    assert artifact["repair_claim_status"] == "blocked"
    assert artifact["repair_reconciliation_ready"] is True
    assert {
        "exp3028:clean_repair_rerun_not_ready",
        "exp3028:repair_controller_not_clean",
        "exp3028:clean_repair_claim_candidate_missing",
        "exp3028:model_specs_missing",
        "exp3028:transcript_count_gap",
        "exp3028:delta_fields_missing",
        "exp3028:pass_at_1_delta_not_positive",
        "exp3028:pass_at_k_delta_negative",
        "exp3028:false_accept_regression",
        "exp3028:syntax_schema_regression",
        "exp3028:intent_drift",
        "exp3028:tautology_gate_not_clean",
        "exp3028:legacy_smoke_only_used",
        "exp3028:reproducibility_checksum_missing",
        "exp3028:acceptance_controller_evidence_missing",
    } <= blocker_ids


def test_req_report_3042_write_artifact_persists_stable_json(tmp_path: Path) -> None:
    """REQ-REPORT-3042: write_artifact emits the deliverable JSON."""

    _write_sources(tmp_path)

    output = mod.write_artifact(tmp_path, started_s=1.0, now_s=1.5)
    saved = json.loads(output.read_text(encoding="utf-8"))

    assert output == tmp_path / mod.OUTPUT_REL_PATH
    assert saved["repair_claim_status"] == "bounded"
    assert saved["duration_s"] == pytest.approx(0.5)
    checksums = {row["path"]: row["sha256"] for row in saved["accepted_source_artifacts"]}
    assert checksums[mod.EXP3041_REL_PATH.as_posix()] == _sha256(tmp_path / mod.EXP3041_REL_PATH)


def test_req_report_3042_helper_edges_remain_honest(tmp_path: Path) -> None:
    """REQ-REPORT-3042: helper edges preserve malformed and unusual states."""

    malformed = tmp_path / "bad.json"
    list_json = tmp_path / "list.json"
    malformed.write_text("{bad-json}\n", encoding="utf-8")
    list_json.write_text("[1, 2, 3]\n", encoding="utf-8")

    repair_row = _row(
        "x",
        "true_blocker",
        "a.json",
        "field",
        rationale="repair source gap",
        evidence={"nested": ["repair"]},
    )
    unrelated_row = _row(
        "exp3034:hardware_blocked",
        "hardware_blocked",
        "a.json",
        "field",
        experiment_id="exp3034",
        rationale="hardware only",
    )

    assert mod.read_json_object(tmp_path / "missing.json") == {}
    assert mod.read_json_object(malformed) == {}
    assert mod.read_json_object(list_json) == {}
    assert mod.sha256_file(tmp_path / "missing.txt") is None
    assert mod._as_mapping({"x": 1}) == {"x": 1}
    assert mod._as_mapping(None) == {}
    assert mod._as_list(["x"]) == ["x"]
    assert mod._as_list("x") == []
    assert mod._float_or_none("1.25") == pytest.approx(1.25)
    assert mod._float_or_none(True) is None
    assert mod._float_or_none("bad") is None
    assert mod._int_or_none("4") == 4
    assert mod._int_or_none(False) is None
    assert mod._int_or_none("bad") is None
    assert mod._compact_row(repair_row)["row_id"] == "x"
    assert mod._repair_relevant_blocker(repair_row) is True
    assert mod._repair_relevant_blocker(unrelated_row) is False
    assert mod._source_checksums([{"path": "a", "sha256": "1"}]) == {"a": "1"}
    assert mod._unique_rows([repair_row, dict(repair_row)]) == [repair_row]
