"""Tests for Exp 3011 milestone .282 terminal capstone.

Spec refs: REQ-REPORT-3011, SCENARIO-REPORT-3011.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from carnot.reporting import capstone_v282_3011 as mod


REQUIRED_FIELDS = {
    "capstone_ready",
    "paper_ready",
    "n_tasks_evaluated",
    "repaired_rows",
    "flagged_rows",
    "blocked_rows",
    "gated_skipped_rows",
    "missing_rows",
    "publication_action_allowed",
    "next_milestone_recommendation",
    "honest_verdict",
}


def _write_json(root: Path, rel_path: Path, payload: dict[str, Any]) -> None:
    path = root / rel_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _row(
    row_id: str,
    source_experiment_id: str,
    status: str,
    *,
    claim_class: str = "claim",
    summary: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "row_id": row_id,
        "source_experiment_id": source_experiment_id,
        "status": status,
        "claim_class": claim_class,
        "claim_boundary": "test boundary",
        "claim_boundary_guard_passed": True,
        "claim_boundary_violations": [],
        "source_honest_verdict": f"{status}: row",
        "summary": summary or {},
        "headline_eligible": status == "clean",
        "paper_claim_eligible": status == "clean",
    }


def _task_rows(*, all_clean: bool = False) -> list[dict[str, Any]]:
    if all_clean:
        return [
            _row("exp3000_archive_activation", "exp3000", "projection-only"),
            _row("exp3001_sota_cache", "exp3001", "clean"),
            _row("exp3002_metamorphic_oracle", "exp3002", "clean"),
            _row("exp3003_metamorphic_repair", "exp3003", "clean"),
            _row("exp3004_aquaforte_beaver_provenance", "exp3004", "clean"),
            _row("exp3005_validator_tree_expansion", "exp3005", "clean"),
            _row("exp3006_fixed_point_diagnostic", "exp3006", "clean"),
            _row("exp3007_fr11_trace_memory_stability", "exp3007", "clean"),
            _row("exp3008_gatemate_host_visible_io", "exp3008", "clean"),
            _row("exp3009_ssqa_dual_bram_report", "exp3009", "clean"),
        ]
    return [
        _row("exp3000_archive_activation", "exp3000", "projection-only"),
        _row("exp3001_sota_cache", "exp3001", "clean"),
        _row("exp3002_metamorphic_oracle", "exp3002", "flagged"),
        _row(
            "exp3003_metamorphic_repair",
            "exp3003",
            "flagged",
            summary={
                "repair_rerun_clean": False,
                "syntax_failure_rate_delta": 0.5,
                "schema_failure_rate_delta": 0.5,
                "false_accept_delta": 0.0,
            },
        ),
        _row(
            "exp3004_aquaforte_beaver_provenance",
            "exp3004",
            "clean",
            summary={
                "substrate_corrigendum_promotable": True,
                "live_retry_provenance_clean": True,
                "enumerator_fallback_separated": True,
            },
        ),
        _row("exp3005_validator_tree_expansion", "exp3005", "clean"),
        _row("exp3006_fixed_point_diagnostic", "exp3006", "clean"),
        _row(
            "exp3007_fr11_trace_memory_stability",
            "exp3007",
            "flagged",
            summary={"trace_memory_stability_ready": True, "heldout_task_count": 4},
        ),
        _row(
            "exp3008_gatemate_host_visible_io",
            "exp3008",
            "blocked",
            summary={"host_visible_io_ready": False, "flash_succeeded": False},
        ),
        _row(
            "exp3009_ssqa_dual_bram_report",
            "exp3009",
            "gated-skipped",
            summary={
                "missing_artifact_present": False,
                "upstream_exp3008_host_visible_io_ready": False,
            },
        ),
    ]


def _matrix_v16(*, all_clean: bool = False, boundary_violation: bool = False) -> dict[str, Any]:
    rows = _task_rows(all_clean=all_clean)
    if not all_clean:
        rows.extend(
            [
                _row("carry_forward_v15:prior_flagged", "exp2998", "flagged"),
                _row("carry_forward_v15:prior_blocked", "exp2998", "blocked"),
                _row("carry_forward_v15:prior_missing", "exp2998", "missing"),
            ]
        )
    violations = (
        [{"row_id": "exp3008_gatemate_host_visible_io", "violation": "unsupported_hardware_claim"}]
        if boundary_violation
        else []
    )
    return {
        "artifact": "experiment_3010_cross_corpus_matrix_v16",
        "matrix_v16_ready": True,
        "honest_verdict": "complete: matrix_v16_ready=true",
        "rows": rows,
        "row_count": len(rows),
        "repaired_claims": [
            "exp3001_sota_headline_cache_ready",
            "exp3004_aquaforte_beaver_substrate_provenance",
            "exp3005_validator_tree_expansion",
            "exp3006_fixed_point_energy_diagnostic",
            *(
                [
                    "exp3003_metamorphic_repair",
                    "exp3007_fr11_trace_memory_stability",
                    "exp3008_gatemate_host_visible_io",
                    "exp3009_ssqa_dual_bram_report",
                ]
                if all_clean
                else []
            ),
        ],
        "still_blocked_claims": [] if all_clean else ["exp3003_metamorphic_repair_flagged"],
        "missing_artifacts": []
        if all_clean
        else [mod.EXP3009_REL_PATH.as_posix()],
        "claim_boundary_violations": violations,
        "recommended_next_actions": [
            "Exp3003: rerun hard-set repair only after removing tautology flags.",
            "Exp3008: add a host-visible GateMate transport.",
            "Exp3009: after Exp3008 reports host_visible_io_ready=true, emit RTL/PnR/resource evidence.",
        ],
        "roadmap_acceptance_summary": {
            "exp3003_repair_promotable": all_clean,
            "exp3004_substrate_promotable": True,
            "exp3007_fr11_promotable": all_clean,
            "exp3008_gatemate_io_promotable": all_clean,
            "exp3009_ssqa_promotable": all_clean,
        },
        "paper_v6_boundary_summary": {"forbidden_claims_absent": not boundary_violation},
        "hardware_boundary_summary": {"forbidden_claims_absent": not boundary_violation},
    }


def _write_ready_sources(
    root: Path,
    *,
    all_clean: bool = False,
    boundary_violation: bool = False,
) -> None:
    _write_json(root, mod.MATRIX_V16_REL_PATH, _matrix_v16(all_clean=all_clean, boundary_violation=boundary_violation))
    _write_json(
        root,
        mod.CAPSTONE_V281_REL_PATH,
        {
            "artifact": "experiment_2999_capstone_v281",
            "capstone_ready": True,
            "paper_ready": False,
            "honest_verdict": "complete: capstone_ready=true; paper_ready=false",
        },
    )
    for row in _task_rows(all_clean=all_clean):
        if row["source_experiment_id"] == "exp3010" or (
            row["source_experiment_id"] == "exp3009" and not all_clean
        ):
            continue
        rel_path = mod.EXP_SOURCE_PATHS[row["source_experiment_id"]]
        _write_json(root, rel_path, {"honest_verdict": row["source_honest_verdict"]})


def test_req_report_3011_spec_anchor_exists() -> None:
    """REQ-REPORT-3011: OpenSpec declares the capstone contract first."""

    spec = Path("openspec/capabilities/research-reporting/spec.md").read_text(encoding="utf-8")

    assert "REQ-REPORT-3011" in spec
    assert "SCENARIO-REPORT-3011" in spec
    assert mod.OUTPUT_REL_PATH.as_posix() in spec


def test_scenario_report_3011_builds_terminal_go_no_go(tmp_path: Path) -> None:
    """SCENARIO-REPORT-3011: .282 capstone reports repair go/no-go honestly."""

    _write_ready_sources(tmp_path)

    artifact = mod.build_artifact(tmp_path, started_s=10.0, now_s=12.25)

    assert REQUIRED_FIELDS <= artifact.keys()
    assert artifact["capstone_ready"] is True
    assert artifact["paper_ready"] is False
    assert artifact["publication_action_allowed"] is False
    assert artifact["n_tasks_evaluated"] == 11
    assert artifact["duration_s"] == pytest.approx(2.25)
    assert artifact["honest_verdict"].startswith("complete: capstone_ready=true; paper_ready=false")
    assert artifact["next_milestone_recommendation"] == mod.NEXT_MILESTONE_RECOMMENDATION

    assert artifact["task_classification_counts"] == {
        "clean": 5,
        "flagged": 3,
        "blocked": 1,
        "gated-skipped": 1,
        "pilot-only": 0,
        "projection-only": 1,
        "missing": 0,
    }
    assert artifact["clean_task_rows"] == [
        "exp3001_sota_cache",
        "exp3004_aquaforte_beaver_provenance",
        "exp3005_validator_tree_expansion",
        "exp3006_fixed_point_diagnostic",
        "exp3010_cross_corpus_matrix_v16",
    ]
    assert artifact["flagged_task_rows"] == [
        "exp3002_metamorphic_oracle",
        "exp3003_metamorphic_repair",
        "exp3007_fr11_trace_memory_stability",
    ]
    assert artifact["blocked_task_rows"] == ["exp3008_gatemate_host_visible_io"]
    assert artifact["gated_skipped_task_rows"] == ["exp3009_ssqa_dual_bram_report"]
    assert artifact["projection_only_task_rows"] == ["exp3000_archive_activation"]

    assert "exp3004_aquaforte_beaver_provenance" in artifact["repaired_rows"]
    assert "exp3003_metamorphic_repair" not in artifact["repaired_rows"]
    assert artifact["flagged_rows"] == [
        "exp3002_metamorphic_oracle",
        "exp3003_metamorphic_repair",
        "exp3007_fr11_trace_memory_stability",
        "carry_forward_v15:prior_flagged",
    ]
    assert artifact["blocked_rows"] == [
        "exp3008_gatemate_host_visible_io",
        "carry_forward_v15:prior_blocked",
    ]
    assert artifact["gated_skipped_rows"] == ["exp3009_ssqa_dual_bram_report"]
    assert artifact["missing_rows"] == ["carry_forward_v15:prior_missing"]
    assert mod.EXP3009_REL_PATH.as_posix() in artifact["missing_artifacts"]

    decisions = artifact["claim_repair_decisions"]
    assert decisions["repair"]["promotable"] is False
    assert decisions["substrate_provenance"]["promotable"] is True
    assert decisions["substrate_provenance"]["repaired_281_blocker"] is True
    assert "not a BEAVER-task solution" in decisions["substrate_provenance"]["claim_boundary"]
    assert decisions["fr11_stability"]["promotable"] is False
    assert decisions["gatemate_io"]["promotable"] is False
    assert decisions["ssqa"]["promotable"] is False
    assert artifact["repaired_281_blockers"] == [
        "exp2993_provenance_repaired_by_exp3004_substrate_provenance"
    ]
    assert artifact["unrepaired_281_blockers"] == [
        "exp2991_methodology_still_flagged_by_exp3003",
        "fr11_stability_carry_forward_still_flagged_by_exp3007",
        "exp2996_hardware_still_blocked_by_exp3008",
        "exp2997_ssqa_still_gated_skipped_by_exp3009",
    ]
    assert "repair/metamorphic row is flagged" in artifact["paper_ready_blockers"]
    assert artifact["source_checksums"][mod.MATRIX_V16_REL_PATH.as_posix()] == _sha256(
        tmp_path / mod.MATRIX_V16_REL_PATH
    )


def test_req_report_3011_blocks_when_required_matrix_missing(tmp_path: Path) -> None:
    """REQ-REPORT-3011: missing matrix v16 fails closed."""

    _write_json(
        tmp_path,
        mod.CAPSTONE_V281_REL_PATH,
        {"honest_verdict": "complete: capstone_ready=true", "capstone_ready": True},
    )

    artifact = mod.build_artifact(tmp_path, started_s=1.0, now_s=1.5)

    assert artifact["capstone_ready"] is False
    assert artifact["paper_ready"] is False
    assert artifact["publication_action_allowed"] is False
    assert artifact["honest_verdict"] == "blocked_required_upstream_missing"
    assert artifact["required_upstream_errors"] == [
        {
            "experiment_id": "exp3010",
            "path": mod.MATRIX_V16_REL_PATH.as_posix(),
            "reason": "missing_or_malformed_artifact",
        }
    ]


def test_req_report_3011_paper_ready_requires_clean_matrix_and_boundaries(tmp_path: Path) -> None:
    """REQ-REPORT-3011: clean synthetic evidence can be ready, publication still cannot."""

    _write_ready_sources(tmp_path, all_clean=True)

    artifact = mod.build_artifact(tmp_path, started_s=2.0, now_s=2.125)

    assert artifact["capstone_ready"] is True
    assert artifact["paper_ready"] is True
    assert artifact["publication_action_allowed"] is False
    assert artifact["paper_ready_blockers"] == []
    assert artifact["flagged_rows"] == []
    assert artifact["blocked_rows"] == []
    assert artifact["gated_skipped_rows"] == []
    assert artifact["missing_rows"] == []
    assert artifact["honest_verdict"].startswith("complete: capstone_ready=true; paper_ready=true")

    _write_ready_sources(tmp_path, all_clean=True, boundary_violation=True)
    blocked = mod.build_artifact(tmp_path, started_s=2.0, now_s=2.25)
    assert blocked["paper_ready"] is False
    assert "matrix_v16 claim_boundary_violations is non-empty" in blocked["paper_ready_blockers"]


def test_req_report_3011_write_artifact_and_main_persist_json(tmp_path: Path) -> None:
    """REQ-REPORT-3011: write_artifact emits the deliverable JSON."""

    _write_ready_sources(tmp_path)

    output = mod.write_artifact(tmp_path, started_s=3.0, now_s=3.5)
    saved = json.loads(output.read_text(encoding="utf-8"))

    assert output == tmp_path / mod.OUTPUT_REL_PATH
    assert saved["capstone_ready"] is True
    assert saved["duration_s"] == pytest.approx(0.5)
    assert mod.main(tmp_path) == 0


def test_req_report_3011_helper_edges_keep_closeout_honest(tmp_path: Path) -> None:
    """REQ-REPORT-3011: helpers keep malformed inputs and unknown statuses honest."""

    missing = tmp_path / "missing.json"
    malformed = tmp_path / "malformed.json"
    list_payload = tmp_path / "list.json"
    malformed.write_text("{bad-json}\n", encoding="utf-8")
    list_payload.write_text("[1, 2]\n", encoding="utf-8")

    assert mod.read_json_object(missing) == {}
    assert mod.read_json_object(malformed) == {}
    assert mod.read_json_object(list_payload) == {}
    assert mod.sha256_file(missing) is None
    assert mod._status_rows([{"row_id": "x", "status": "clean"}, {"status": "clean"}], "clean") == [
        "x"
    ]
    assert mod._status_rows([{"row_id": "x", "status": "unknown"}], "missing") == ["x"]
    assert mod._row_by_id([{"row_id": "x", "status": "clean"}, []]) == {
        "x": {"row_id": "x", "status": "clean"}
    }
    assert mod._task_row_from_matrix("exp3010", {}, {})["status"] == "missing"
    assert mod._task_row_from_matrix("unknown", {}, {})["status"] == "missing"
    assert mod._matrix_wide_status_rows({}, "clean") == []
    assert mod._paper_ready_blockers({}, {}, [], [], []) == [
        "matrix_v16_ready is not true",
        "matrix_v16 claim_boundary_violations is non-empty",
        "repair/metamorphic row is missing",
        "substrate-provenance row is missing",
        "FR-11 stability row is missing",
        "GateMate IO row is missing",
        "SSQA row is missing",
    ]
