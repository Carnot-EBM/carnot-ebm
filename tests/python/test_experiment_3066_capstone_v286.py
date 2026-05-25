"""Tests for Exp 3066 milestone .286 capstone.

Spec refs: REQ-REPORT-3066, SCENARIO-REPORT-3066.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from carnot.reporting import capstone_v286_3066 as mod


REQUIRED_FIELDS = {
    "capstone_ready",
    "paper_ready",
    "repair_claim_status",
    "solver_grounding_status",
    "fr11_self_learning_status",
    "kan_pwa_status",
    "gatemate_status",
    "ssqa_status",
    "publication_blockers",
    "next_milestone_recommendation",
    "source_artifacts",
    "inference_substrate",
    "honest_verdict",
}
CLASS_FIELDS = (
    "clean_rows",
    "flagged_rows",
    "bounded_rows",
    "blocked_rows",
    "gated_skipped_rows",
    "projection_only_rows",
    "missing_rows",
    "retired_rows",
)
EXPECTED_RECOMMENDATION = (
    "2026.05.287: keep repair headline wording retired or bounded until repair "
    "disqualifiers, adversarial flags, and the gated repair rerun clear; repair "
    "solver-grounded verification by producing positive local verifier gain and "
    "non-flagged SMT guidance over solver-only authority; carry FR-11 only as "
    "controller-side self-learning unless a source artifact explicitly trains and "
    "verifies model weights; keep KAN/PWA bounded to controller-anchor locality; "
    "unblock GateMate with host-visible output-contract and smoke transcript "
    "evidence before any SSQA, readback, or hardware-speedup claim; drive "
    "publication_blocker_count from 10 to 0."
)
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


def _row(row_id: str, status: str, evidence_class: str, source: str | None = None) -> dict[str, Any]:
    return {
        "row_id": row_id,
        "status": status,
        "source_artifact": source or f"results/{row_id.replace(':', '_')}.json",
        "source_field": "status",
        "evidence_class": evidence_class,
        "blocker_class": mod.blocker_class(status),
        "claim_scope": evidence_class,
        "summary": {"status": status},
    }


def _source_artifact(exp_id: str, path: Path, *, required: bool = False) -> dict[str, Any]:
    return {
        "experiment_id": exp_id,
        "path": path.as_posix(),
        "role": f"{exp_id}_role",
        "required": required,
        "present": True,
        "readable_json_object": True,
        "sha256": f"upstream-{exp_id}",
    }


def _status_summaries(*, clean: bool = False) -> dict[str, Any]:
    if clean:
        return {
            "repair": {"status": "clean_promotable", "citations": []},
            "solver_grounded_verification": {"status": "clean_solver_authority", "citations": []},
            "fr11": {"status": "controller_only_clean", "citations": []},
            "kan_pwa": {"status": "clean_controller_anchor", "citations": []},
            "gatemate": {"status": "host_visible_output_ready", "citations": []},
            "ssqa": {"status": "host_visible_readback_ready", "citations": []},
        }
    return {
        "repair": {"status": "bounded_and_gated_skipped", "citations": []},
        "solver_grounded_verification": {
            "status": "flagged_solver_grounded_no_gain",
            "citations": [],
        },
        "fr11": {"status": "controller_only_delayed_regression_ready_flagged", "citations": []},
        "kan_pwa": {"status": "bounded_controller_anchor_audit_not_promoted", "citations": []},
        "gatemate": {"status": "blocked_no_rerun_operator_actions_required", "citations": []},
        "ssqa": {"status": "gated_skipped_host_visible_smoke_missing", "citations": []},
    }


def _matrix_v20(*, clean: bool = False) -> dict[str, Any]:
    if clean:
        clean_rows = [
            _row("repair:headline_status", "clean", "repair_claim_boundary"),
            _row("solver:local_sota_solution_verifier_gain_panel", "clean", "solver_grounded"),
            _row("fr11:delayed_regression", "clean", "fr11_controller_self_learning"),
            _row("kan:pwa_locality_audit", "clean", "kan_pwa_controller_anchor_audit"),
            _row("gatemate:host_visible_output", "clean", "gatemate_output_contract"),
            _row("ssqa:host_visible_readback", "clean", "ssqa_readback_boundary"),
        ]
        clean_rows[2]["summary"] = {
            "model_weight_training": False,
            "model_weight_mutation": False,
        }
        clean_rows[4]["summary"] = {"host_visible_output_evidence": True}
        clean_rows[5]["summary"] = {"host_visible_smoke_present": True}
        classes = {field: [] for field in CLASS_FIELDS}
        classes["clean_rows"] = clean_rows
        return {
            "artifact": "experiment_3065_cross_corpus_matrix_v20",
            "matrix_v20_ready": True,
            "paper_ready": True,
            "rows_total": len(clean_rows),
            "publication_blocker_count": 0,
            "publication_blockers": [],
            "status_summaries": _status_summaries(clean=True),
            "source_artifacts": _required_sources(),
            "required_source_errors": [],
            "honest_verdict": "complete: matrix_v20_ready=true",
            **classes,
        }

    classes = {field: [] for field in CLASS_FIELDS}
    classes["clean_rows"] = [
        _row("methodology:flag_hygiene", "clean", "methodology_boundary"),
        _row("archive:v286_activation", "clean", "archive_activation"),
    ]
    classes["flagged_rows"] = [
        _row("solver:local_sota_solution_verifier_gain_panel", "flagged", "solver_grounded"),
        _row("solver:aquaforte_smt_pilot", "flagged", "solver_grounded"),
        _row("fr11:delayed_regression", "flagged", "fr11_controller_self_learning"),
    ]
    classes["bounded_rows"] = [
        _row("repair:headline_status", "bounded", "repair_claim_boundary"),
        _row("kan:pwa_locality_audit", "bounded", "kan_pwa_controller_anchor_audit"),
    ]
    classes["blocked_rows"] = [
        _row("repair:de_tautology_disqualifiers", "blocked", "repair_claim_boundary"),
        _row("gatemate:no_rerun_ledger", "blocked", "gatemate_output_contract"),
    ]
    classes["gated_skipped_rows"] = [
        _row("repair:gated_sota_rerun", "gated_skipped", "repair_live_rerun"),
        _row("ssqa:host_visible_readback_boundary", "gated_skipped", "ssqa_readback_boundary"),
    ]
    classes["projection_only_rows"] = []
    classes["missing_rows"] = [
        _row(
            "source:exp3059_requested_v1_alias",
            "missing",
            "source_artifact_presence",
            "results/experiment_3059_gated_sota_repair_de_tautology_rerun_v1.json",
        )
    ]
    classes["retired_rows"] = [
        _row("repair:headline_sota_repair_clean_methodology", "retired", "repair_retired_claim")
    ]
    all_rows = [row for field in CLASS_FIELDS for row in classes[field]]
    blockers = [
        {
            "row_id": row["row_id"],
            "status": row["status"],
            "blocker_class": row["blocker_class"],
            "source_artifact": row["source_artifact"],
            "source_field": row["source_field"],
            "claim_scope": row["claim_scope"],
        }
        for row in all_rows
        if row["status"] != "clean" and row["status"] != "retired"
    ]
    return {
        "artifact": "experiment_3065_cross_corpus_matrix_v20",
        "matrix_v20_ready": True,
        "paper_ready": False,
        "rows_total": len(all_rows),
        "publication_blocker_count": len(blockers),
        "publication_blockers": blockers,
        "status_summaries": _status_summaries(),
        "source_artifacts": _required_sources(),
        "required_source_errors": [],
        "honest_verdict": "complete: matrix_v20_ready=true",
        **classes,
    }


def _required_sources() -> list[dict[str, Any]]:
    return [
        _source_artifact("exp3053", Path("results/experiment_3053_capstone_v285.json"), required=True),
        _source_artifact("exp3057", Path("results/experiment_3057_local_sota_solution_verifier_gain_panel_v1.json")),
        _source_artifact("exp3058", Path("results/experiment_3058_aquaforte_style_llm_guided_smt_pilot_v1.json")),
        _source_artifact("exp3061", Path("results/experiment_3061_fr11_delayed_regression_solver_self_model_pilot_v1.json")),
        _source_artifact("exp3062", Path("results/experiment_3062_kan_pwa_locality_verification_audit_v1.json")),
        _source_artifact("exp3063", Path("results/experiment_3063_gatemate_no_rerun_operator_action_ledger_v1.json")),
        _source_artifact("exp3064", Path("results/experiment_3064_ssqa_host_visible_readback_boundary_ledger_v1.json")),
    ]


def _write_sources(root: Path, *, clean: bool = False, omit: set[Path] | None = None) -> dict[str, Any]:
    omit = omit or set()
    matrix = _matrix_v20(clean=clean)
    _write_json(root, mod.MATRIX_V20_REL_PATH, matrix)
    for source in matrix["source_artifacts"]:
        rel_path = Path(source["path"])
        if rel_path not in omit:
            _write_json(root, rel_path, {"artifact": source["experiment_id"]})
    for field in CLASS_FIELDS:
        for row in matrix[field]:
            rel_path = Path(row["source_artifact"])
            if rel_path not in omit and rel_path.as_posix().endswith(".json"):
                _write_json(root, rel_path, {"artifact": row["row_id"], "status": row["status"]})
    return matrix


def _rows_by_id(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(row["row_id"]): row for row in rows}


def test_req_report_3066_spec_anchor_and_script_exist() -> None:
    """REQ-REPORT-3066: OpenSpec declares the .286 capstone contract first."""

    spec = (mod.REPO_ROOT / "openspec/capabilities/research-reporting/spec.md").read_text(
        encoding="utf-8"
    )

    assert "REQ-REPORT-3066" in spec
    assert "SCENARIO-REPORT-3066" in spec
    assert mod.OUTPUT_REL_PATH.as_posix() in spec
    assert mod.SCRIPT_REL_PATH.exists()


def test_scenario_report_3066_builds_capstone_from_matrix_v20(tmp_path: Path) -> None:
    """SCENARIO-REPORT-3066: non-clean matrix v20 rows keep paper readiness false."""

    matrix = _write_sources(tmp_path)

    artifact = mod.build_artifact(tmp_path, started_s=2.0, now_s=4.5)
    summary = artifact["matrix_v20_summary"]
    checks = {row["check"]: row for row in artifact["paper_ready_checks"]}

    assert REQUIRED_FIELDS <= artifact.keys()
    assert FORBIDDEN_TOP_LEVEL.isdisjoint(artifact)
    assert artifact["capstone_ready"] is True
    assert artifact["paper_ready"] is False
    assert artifact["duration_s"] == pytest.approx(2.5)
    assert artifact["honest_verdict"].startswith("complete:")
    assert artifact["repair_claim_status"] == "bounded_and_gated_skipped"
    assert artifact["solver_grounding_status"] == "flagged_solver_grounded_no_gain"
    assert artifact["fr11_self_learning_status"] == "controller_only_delayed_regression_ready_flagged"
    assert artifact["kan_pwa_status"] == "bounded_controller_anchor_audit_not_promoted"
    assert artifact["gatemate_status"] == "blocked_no_rerun_operator_actions_required"
    assert artifact["ssqa_status"] == "gated_skipped_host_visible_smoke_missing"
    assert artifact["publication_blockers"] == matrix["publication_blockers"]
    assert artifact["next_milestone_recommendation"] == EXPECTED_RECOMMENDATION

    assert summary["matrix_v20_ready"] is True
    assert summary["rows_total"] == matrix["rows_total"]
    assert summary["row_count_observed"] == matrix["rows_total"]
    assert summary["counts_match_rows"] is True
    assert summary["publication_blocker_count_matches"] is True
    assert summary["required_source_artifacts_readable"] is True
    assert summary["promoted_claim_count"] == 2
    assert summary["publication_blocker_count"] == 10

    promoted = _rows_by_id(artifact["promoted_claims"])
    assert set(promoted) == {"methodology:flag_hygiene", "archive:v286_activation"}
    assert artifact["promoted_claim_source_coverage"]["all_promoted_claims_have_sources"] is True
    assert checks["capstone_ready"]["passed"] is True
    assert checks["matrix_has_no_publication_blockers"]["passed"] is False
    assert checks["promoted_claims_have_source_artifacts"]["passed"] is True
    assert checks["fr11_model_weight_boundary"]["passed"] is True
    assert checks["hardware_host_visible_output"]["passed"] is True

    source_by_path = {row["path"]: row for row in artifact["source_artifacts"]}
    assert source_by_path[mod.MATRIX_V20_REL_PATH.as_posix()]["sha256"] == _sha256(
        tmp_path / mod.MATRIX_V20_REL_PATH
    )
    assert source_by_path[mod.MATRIX_V20_REL_PATH.as_posix()]["required"] is True
    assert artifact["inference_substrate"] == {
        "kind": "aggregation_from_upstream_artifacts",
        "source": "checked_in_artifacts",
        "executes_models": False,
        "executes_hardware": False,
        "executes_conductor": False,
        "executes_live_repair": False,
        "no_live_llm_inference": True,
    }


def test_req_report_3066_sets_paper_ready_only_with_zero_blockers_and_sources(
    tmp_path: Path,
) -> None:
    """REQ-REPORT-3066: paper readiness requires zero blockers and sourced clean claims."""

    _write_sources(tmp_path, clean=True)

    artifact = mod.build_artifact(tmp_path)
    checks = {row["check"]: row for row in artifact["paper_ready_checks"]}

    assert artifact["capstone_ready"] is True
    assert artifact["paper_ready"] is True
    assert artifact["publication_blockers"] == []
    assert artifact["repair_claim_status"] == "clean_promotable"
    assert artifact["solver_grounding_status"] == "clean_solver_authority"
    assert artifact["gatemate_status"] == "host_visible_output_ready"
    assert artifact["ssqa_status"] == "host_visible_readback_ready"
    assert all(row["passed"] is True for row in checks.values())
    assert artifact["honest_verdict"].startswith("complete:")


def test_req_report_3066_blocks_missing_matrix_and_required_sources(tmp_path: Path) -> None:
    """REQ-REPORT-3066: absent matrix v20 or required source artifacts fail closed."""

    blocked_without_matrix = mod.build_artifact(tmp_path)
    assert blocked_without_matrix["capstone_ready"] is False
    assert blocked_without_matrix["paper_ready"] is False
    assert blocked_without_matrix["honest_verdict"] == "blocked_required_matrix_v20_missing"

    _write_sources(tmp_path, omit={Path("results/experiment_3053_capstone_v285.json")})
    blocked_missing_required_source = mod.build_artifact(tmp_path)

    assert blocked_missing_required_source["capstone_ready"] is False
    assert blocked_missing_required_source["paper_ready"] is False
    assert blocked_missing_required_source["required_source_errors"] == [
        {"experiment_id": "exp3053", "reason": "missing_or_malformed_required_artifact"}
    ]
    assert blocked_missing_required_source["honest_verdict"].startswith(
        "blocked_capstone_preconditions"
    )


def test_req_report_3066_keeps_fr11_and_hardware_claim_boundaries(tmp_path: Path) -> None:
    """REQ-REPORT-3066: model-weight and hardware claims need explicit evidence."""

    matrix = _matrix_v20(clean=True)
    matrix["clean_rows"][2]["summary"] = {
        "model_weight_training": True,
        "model_weight_training_verified": False,
    }
    matrix["clean_rows"][4]["summary"] = {"host_visible_output_evidence": False}
    _write_json(tmp_path, mod.MATRIX_V20_REL_PATH, matrix)
    for source in matrix["source_artifacts"]:
        _write_json(tmp_path, Path(source["path"]), {"artifact": source["experiment_id"]})
    for row in matrix["clean_rows"][1:]:
        _write_json(tmp_path, Path(row["source_artifact"]), {"artifact": row["row_id"]})

    artifact = mod.build_artifact(tmp_path)
    checks = {row["check"]: row for row in artifact["paper_ready_checks"]}

    assert artifact["capstone_ready"] is True
    assert artifact["paper_ready"] is False
    assert artifact["fr11_self_learning_status"] == "blocked_model_weight_learning_unverified"
    assert artifact["gatemate_status"] == "blocked_host_visible_output_missing"
    assert artifact["promoted_claim_source_coverage"]["missing_promoted_claim_sources"] == [
        {
            "row_id": "repair:headline_status",
            "source_artifact": "results/repair_headline_status.json",
            "source_field": "status",
        }
    ]
    assert checks["promoted_claims_have_source_artifacts"]["passed"] is False
    assert checks["fr11_model_weight_boundary"]["passed"] is False
    assert checks["hardware_host_visible_output"]["passed"] is False


def test_req_report_3066_write_artifact_and_helper_edges(tmp_path: Path) -> None:
    """REQ-REPORT-3066: writing and malformed input handling stay deterministic."""

    _write_sources(tmp_path)
    malformed = tmp_path / "bad.json"
    list_json = tmp_path / "list.json"
    malformed.write_text("{bad-json}\n", encoding="utf-8")
    list_json.write_text("[1, 2, 3]\n", encoding="utf-8")

    output = mod.write_artifact(tmp_path, started_s=4.0, now_s=6.0)
    saved = json.loads(output.read_text(encoding="utf-8"))

    assert output == tmp_path / mod.OUTPUT_REL_PATH
    assert saved["capstone_ready"] is True
    assert saved["duration_s"] == pytest.approx(2.0)
    assert saved["source_checksums"][mod.MATRIX_V20_REL_PATH.as_posix()] == _sha256(
        tmp_path / mod.MATRIX_V20_REL_PATH
    )
    assert mod.read_json_object(tmp_path / "missing.json") == {}
    assert mod.read_json_object(malformed) == {}
    assert mod.read_json_object(list_json) == {}
    assert mod.sha256_file(tmp_path / "missing.txt") is None
    assert mod.normal_status("gate-skipped") == "gated_skipped"
    assert mod.normal_status("gate_skipped") == "gated_skipped"
    assert mod.normal_status("pilot_only") == "bounded"
    assert mod.normal_status("unknown") == "missing"
    assert mod.blocker_class("clean") == "none"
    assert mod.blocker_class("missing") == "missing_artifact"
    assert mod._as_mapping({"x": 1}) == {"x": 1}
    assert mod._as_mapping(None) == {}
    assert mod._as_list([1]) == [1]
    assert mod._as_list("x") == []
    assert mod._int_or_none(True) is None
    assert mod._int_or_none("bad") is None
    assert mod._claim_entry({"row_id": "x", "status": "bad"})["status"] == "missing"
