"""Tests for Exp 3041 matrix/capstone adversarial flag hygiene.

Spec refs: REQ-REPORT-3041, SCENARIO-REPORT-3041.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from carnot.reporting import matrix_capstone_adversarial_flag_hygiene_3041 as mod


REQUIRED_FIELDS = {
    "flag_hygiene_ready",
    "rows_reviewed",
    "true_blocker_rows",
    "aggregation_false_positive_rows",
    "missing_metadata_rows",
    "unresolved_bound_rows",
    "hardware_blocked_rows",
    "gate_skipped_rows",
    "source_artifacts",
    "inference_substrate",
    "honest_verdict",
}


def _write_json(root: Path, rel_path: Path, payload: dict[str, Any]) -> None:
    path = root / rel_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _field_rows(rows: list[dict[str, Any]]) -> set[tuple[str, str]]:
    return {(str(row["row_id"]), str(row["source_field"])) for row in rows}


def _row_ids(rows: list[dict[str, Any]]) -> set[str]:
    return {str(row["row_id"]) for row in rows}


def _flag(kind: str, severity: str = "critical") -> dict[str, str]:
    return {"kind": kind, "severity": severity, "detail": f"{kind} fixture detail"}


def _exp3027() -> dict[str, Any]:
    return {
        "artifact": "experiment_3027_adversarial_flag_methodology_corrigendum_v1",
        "methodology_corrigendum_ready": True,
        "flagged_adversarial": True,
        "corrigendum_pending": [_flag("DURATION_TOO_SHORT"), _flag("METHODOLOGY_MISSING", "warn")],
        "inference_substrate": {
            "kind": "aggregation_from_upstream_artifacts",
            "no_live_llm_inference": True,
        },
        "true_methodology_blockers": [
            {
                "row_id": "exp3013_sota_logprob_telemetry",
                "classification": "true_methodology_blocker",
                "matrix_status": "flagged",
                "source_artifact_path": "results/experiment_3013.json",
                "source_experiment_id": "exp3013",
                "supporting_fields": [{"field": "rows[].status", "value": "flagged"}],
            }
        ],
        "missing_metadata_rows": [
            {
                "row_id": "exp3016_repair_acceptance_controller",
                "classification": "missing_metadata",
                "matrix_status": "flagged",
                "source_artifact_path": "results/experiment_3016.json",
                "source_experiment_id": "exp3016",
                "supporting_fields": [{"field": "random_seed", "value": None}],
            }
        ],
        "unresolved_bound_rows": [
            {
                "row_id": "exp3018_beaver_frontier_certificate",
                "classification": "unresolved_bound",
                "matrix_status": "flagged",
                "source_artifact_path": "results/experiment_3018.json",
                "source_experiment_id": "exp3018",
                "supporting_fields": [{"field": "unresolved_count", "value": 2}],
            }
        ],
        "hardware_blocked_rows": [
            {
                "row_id": "exp3021_gatemate_transport_shim",
                "classification": "hardware_blocked",
                "matrix_status": "blocked",
                "source_artifact_path": "results/experiment_3024_cross_corpus_matrix_v17.json",
                "source_experiment_id": "exp3021",
                "supporting_fields": [{"field": "rows[].status", "value": "blocked"}],
            }
        ],
        "honest_verdict": "complete: methodology_corrigendum_ready=true",
    }


def _exp3028() -> dict[str, Any]:
    return {
        "artifact": "experiment_3028_sota_repair_clean_methodology_rerun_v2",
        "flagged_adversarial": True,
        "corrigendum_pending": [
            _flag("TAUTOLOGY"),
            _flag("DURATION_TOO_SHORT"),
            _flag("METHODOLOGY_MISSING", "warn"),
        ],
        "inference_substrate": {
            "kind": "clean_repair_reconstruction",
            "live_repair_generation_run": False,
            "model_load_attempted": False,
        },
        "clean_repair_rerun_ready": True,
        "clean_repair_claim_promotable_candidate": True,
        "repair_controller_clean": True,
        "n_tasks": 24,
        "n_live_transcripts": 24,
        "pass_at_1_delta": 0.375,
        "pass_at_k_delta": 0.375,
        "false_accept_delta": 0.0,
        "tautology_gate_clean": True,
        "honest_verdict": "complete: clean_repair_rerun_ready=true; n_tasks=24",
    }


def _exp3029() -> dict[str, Any]:
    return {
        "artifact": "experiment_3029_repair_promotion_boundary_audit_v2",
        "flagged_adversarial": True,
        "corrigendum_pending": [_flag("DURATION_TOO_SHORT"), _flag("METHODOLOGY_MISSING", "warn")],
        "inference_substrate": {
            "kind": "aggregation_from_upstream_artifacts",
            "no_live_llm_inference": True,
        },
        "repair_claim_status": "bounded",
        "bounded_claims": [
            {
                "claim_id": "exp3028_clean_repair_candidate",
                "classification": "bounded",
                "blockers": [
                    "matrix repair row is flagged",
                    "capstone repair decision is not promotable",
                ],
            }
        ],
        "retired_or_blocked_claims": [
            {
                "claim_id": "unsupported_exp3016_headline_repair_promotion",
                "classification": "retired",
                "blockers": ["Exp 3016 random_seed missing"],
            }
        ],
        "honest_verdict": "complete: repair_claim_status=bounded",
    }


def _matrix_v18() -> dict[str, Any]:
    rows = [
        {"experiment_id": "exp3027", "status": "flagged", "upstream_flags": ["flagged_adversarial=true"]},
        {
            "experiment_id": "exp3028",
            "status": "flagged",
            "upstream_flags": ["TAUTOLOGY:critical", "METHODOLOGY_MISSING:warn"],
        },
        {
            "experiment_id": "exp3029",
            "status": "flagged",
            "repair_claim_status": "bounded",
            "upstream_flags": ["DURATION_TOO_SHORT:critical"],
        },
        {
            "experiment_id": "exp3031",
            "status": "flagged",
            "upstream_flags": ["DURATION_TOO_SHORT:critical"],
        },
        {
            "experiment_id": "exp3034",
            "status": "blocked",
            "task_class": "gatemate_output_contract",
            "gatemate_output_contract_ready": False,
            "host_visible_output_observed": False,
            "summary": {"selected_output_path": "explicit_no_ready_contract"},
        },
        {
            "experiment_id": "exp3035",
            "status": "gated_skipped",
            "task_class": "gatemate_output_shim",
            "summary": {"gate_check_summary": "exp3034 gate failed"},
        },
        {
            "experiment_id": "exp3036",
            "status": "gated_skipped",
            "task_class": "gatemate_host_visible_flash_smoke",
            "actual_path_present": False,
            "summary": {"gate_source": "exp3035"},
        },
        {
            "experiment_id": "exp3037",
            "status": "gated_skipped",
            "task_class": "ssqa_boundary",
            "ssqa_gate_status": "gate_skipped",
        },
        {
            "experiment_id": "exp3039",
            "status": "missing",
            "task_class": "capstone",
            "actual_path_present": False,
        },
    ]
    return {
        "artifact": "experiment_3038_cross_corpus_matrix_v18",
        "matrix_v18_ready": True,
        "rows_total": 14,
        "flagged": 4,
        "blocked": 1,
        "gated_skipped": 3,
        "missing": 1,
        "matrix_rows": rows,
        "flagged_adversarial": True,
        "corrigendum_pending": [_flag("DURATION_TOO_SHORT"), _flag("METHODOLOGY_MISSING", "warn")],
        "inference_substrate": {"kind": "aggregation_from_upstream_artifacts", "executes_models": False},
        "honest_verdict": "complete: matrix_v18_ready=true",
    }


def _capstone_v284() -> dict[str, Any]:
    return {
        "artifact": "experiment_3039_capstone_v284",
        "capstone_ready": True,
        "paper_ready": False,
        "repair_claim_status": "bounded",
        "gatemate_status": "blocked_pinout_missing_bounded",
        "ssqa_status": "gate_skipped_bounded_no_performance_claim",
        "flagged_adversarial": True,
        "corrigendum_pending": [_flag("DURATION_TOO_SHORT"), _flag("METHODOLOGY_MISSING", "warn")],
        "inference_substrate": {
            "kind": "aggregation_from_upstream_artifacts",
            "executes_models": False,
            "executes_hardware": False,
        },
        "blockers_remaining": [
            {"area": "matrix_nonclean", "status": "publication_blocking"},
            {"area": "repair", "status": "bounded"},
            {"area": "gatemate", "status": "blocked_pinout_missing_bounded"},
            {"area": "ssqa", "status": "gate_skipped_bounded_no_performance_claim"},
        ],
        "paper_ready_checks": [
            {"check": "repair_promotable", "passed": False, "reason": "repair_claim_status=bounded"}
        ],
        "honest_verdict": "complete: capstone_ready=true; paper_ready=false",
    }


def _write_sources(root: Path) -> None:
    _write_json(root, mod.EXP3027_REL_PATH, _exp3027())
    _write_json(root, mod.EXP3028_REL_PATH, _exp3028())
    _write_json(root, mod.EXP3029_REL_PATH, _exp3029())
    _write_json(root, mod.MATRIX_V18_REL_PATH, _matrix_v18())
    _write_json(root, mod.CAPSTONE_V284_REL_PATH, _capstone_v284())


def test_req_report_3041_spec_anchor_and_script_exist() -> None:
    """REQ-REPORT-3041: OpenSpec declares the flag-hygiene contract first."""

    spec = (mod.REPO_ROOT / "openspec/capabilities/research-reporting/spec.md").read_text(
        encoding="utf-8"
    )

    assert "REQ-REPORT-3041" in spec
    assert "SCENARIO-REPORT-3041" in spec
    assert mod.OUTPUT_REL_PATH.as_posix() in spec
    assert mod.SCRIPT_REL_PATH.exists()


def test_scenario_report_3041_builds_mechanically_consumable_classifications(
    tmp_path: Path,
) -> None:
    """SCENARIO-REPORT-3041: false positives and real blockers stay separate."""

    _write_sources(tmp_path)

    artifact = mod.build_artifact(tmp_path, started_s=10.0, now_s=12.5)
    all_rows = mod.classification_rows(artifact)

    assert REQUIRED_FIELDS <= artifact.keys()
    assert artifact["flag_hygiene_ready"] is True
    assert artifact["rows_reviewed"] == len(all_rows)
    assert artifact["duration_s"] == pytest.approx(2.5)
    assert artifact["honest_verdict"].startswith("complete: flag_hygiene_ready=true")
    assert artifact["downstream_consumers"] == ["exp3042", "exp3043"]
    assert artifact["inference_substrate"] == {
        "kind": "aggregation_from_upstream_artifacts",
        "executes_models": False,
        "executes_hardware": False,
        "executes_conductor": False,
        "source": "checked_in_artifacts",
    }

    assert ("exp3027:top_level_aggregation_flags", "corrigendum_pending") in _field_rows(
        artifact["aggregation_false_positive_rows"]
    )
    assert ("exp3038:top_level_aggregation_flags", "corrigendum_pending") in _field_rows(
        artifact["aggregation_false_positive_rows"]
    )
    assert "exp3028:adversarial_flags" in _row_ids(artifact["true_blocker_rows"])
    assert "exp3031:matrix_flagged" in _row_ids(artifact["true_blocker_rows"])
    assert "exp3028:methodology_missing" in _row_ids(artifact["missing_metadata_rows"])
    assert "exp3039:matrix_missing" in _row_ids(artifact["missing_metadata_rows"])
    assert "exp3018_beaver_frontier_certificate" in _row_ids(
        artifact["unresolved_bound_rows"]
    )
    assert "exp3029:exp3028_clean_repair_candidate" in _row_ids(
        artifact["unresolved_bound_rows"]
    )
    assert "exp3034:hardware_blocked" in _row_ids(artifact["hardware_blocked_rows"])
    assert {"exp3035:gate_skipped", "exp3036:gate_skipped", "exp3037:gate_skipped"} <= _row_ids(
        artifact["gate_skipped_rows"]
    )
    assert all(row["source_artifact"] and row["source_field"] for row in all_rows)
    assert all(row["blocking"] is True for row in artifact["missing_metadata_rows"])
    assert all(row["blocking"] is True for row in artifact["unresolved_bound_rows"])


def test_req_report_3041_does_not_relabel_nonaggregation_flags_as_false_positive(
    tmp_path: Path,
) -> None:
    """REQ-REPORT-3041: only explicit aggregation artifacts can clear duration flags."""

    _write_sources(tmp_path)

    artifact = mod.build_artifact(tmp_path)

    false_positive_ids = _row_ids(artifact["aggregation_false_positive_rows"])
    true_blocker_ids = _row_ids(artifact["true_blocker_rows"])

    assert "exp3028:top_level_aggregation_flags" not in false_positive_ids
    assert "exp3028:adversarial_flags" in true_blocker_ids


def test_req_report_3041_blocks_when_required_source_is_missing(tmp_path: Path) -> None:
    """REQ-REPORT-3041: missing required source artifacts fail closed."""

    _write_json(tmp_path, mod.EXP3027_REL_PATH, _exp3027())
    _write_json(tmp_path, mod.EXP3028_REL_PATH, _exp3028())

    artifact = mod.build_artifact(tmp_path)

    assert artifact["flag_hygiene_ready"] is False
    assert artifact["honest_verdict"].startswith("blocked_required_source_missing:")
    assert artifact["rows_reviewed"] == 0
    assert {row["experiment_id"] for row in artifact["required_source_errors"]} == {
        "exp3029",
        "exp3038",
        "exp3039",
    }


def test_req_report_3041_write_artifact_persists_stable_json(tmp_path: Path) -> None:
    """REQ-REPORT-3041: write_artifact emits the deliverable JSON."""

    _write_sources(tmp_path)

    output = mod.write_artifact(tmp_path, started_s=1.0, now_s=1.125)
    saved = json.loads(output.read_text(encoding="utf-8"))

    assert output == tmp_path / mod.OUTPUT_REL_PATH
    assert saved["flag_hygiene_ready"] is True
    assert saved["duration_s"] == pytest.approx(0.125)
    checksums = {row["path"]: row["sha256"] for row in saved["source_artifacts"]}
    assert checksums[mod.CAPSTONE_V284_REL_PATH.as_posix()] == _sha256(
        tmp_path / mod.CAPSTONE_V284_REL_PATH
    )


def test_req_report_3041_helper_edges_remain_honest(tmp_path: Path) -> None:
    """REQ-REPORT-3041: helper edges preserve malformed and unusual states."""

    malformed = tmp_path / "bad.json"
    list_json = tmp_path / "list.json"
    malformed.write_text("{bad-json}\n", encoding="utf-8")
    list_json.write_text("[1, 2, 3]\n", encoding="utf-8")

    assert mod.read_json_object(tmp_path / "missing.json") == {}
    assert mod.read_json_object(malformed) == {}
    assert mod.read_json_object(list_json) == {}
    assert mod.sha256_file(tmp_path / "missing.txt") is None
    assert mod._as_mapping({"x": 1}) == {"x": 1}
    assert mod._as_mapping(None) == {}
    assert mod._as_list(["x"]) == ["x"]
    assert mod._as_list("x") == []
    assert mod._flag_kinds([{"kind": "A"}, {"kind": ""}, "bad"]) == ["A"]
    assert mod._flag_kinds_from_strings(["flagged_adversarial=true"]) == ["flagged_adversarial"]
    assert mod._is_aggregation_only({"inference_substrate": "aggregation_from_upstream_artifacts"})
    assert not mod._is_aggregation_only({"inference_substrate": {"kind": "live_llm_inference"}})
    assert mod.classification_rows({"true_blocker_rows": [{"row_id": "x"}]}) == [{"row_id": "x"}]
    assert mod._looks_hardware_related({"host_visible_output_observed": False}, "") is True
    assert mod._honest_verdict(False, 0, {}) == "blocked_flag_hygiene_incomplete: rows_reviewed=0"
    assert mod._unique_rows(
        [
            {
                "row_id": "dup",
                "classification": "true_blocker",
                "source_artifact": "a.json",
                "source_field": "field",
            },
            {
                "row_id": "dup",
                "classification": "true_blocker",
                "source_artifact": "a.json",
                "source_field": "field",
            },
        ]
    ) == [
        {
            "row_id": "dup",
            "classification": "true_blocker",
            "source_artifact": "a.json",
            "source_field": "field",
        }
    ]
    assert mod._rows_from_exp3027_list(
        {"true_methodology_blockers": ["bad"]},
        list_name="true_methodology_blockers",
        classification="true_blocker",
        default_rationale="fallback",
        blocking=True,
    ) == []
    assert mod._classification_lists({"exp3027": {"flagged_adversarial": False}})[
        "true_blocker_rows"
    ] == []
