"""Tests for Exp 2999 milestone .281 capstone.

Spec refs: REQ-REPORT-2999, SCENARIO-REPORT-2999.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from carnot.reporting import capstone_v281_2999 as mod


REQUIRED_FIELDS = {
    "capstone_ready",
    "paper_ready",
    "clean_artifacts",
    "flagged_artifacts",
    "blocked_artifacts",
    "missing_artifacts",
    "gated_skipped_artifacts",
    "gaps_closed",
    "gaps_remaining",
    "next_milestone_recommendations",
    "external_publication_triggered",
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
    status: str,
    claim_class: str,
    *,
    source: str = "exp2998",
    verdict: str = "complete: row",
    summary: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "row_id": row_id,
        "source_experiment_id": source,
        "milestone": "2026.05.281",
        "status": status,
        "claim_class": claim_class,
        "evidence_type": claim_class,
        "source_honest_verdict": verdict,
        "summary": summary or {},
        "headline_eligible": status == "clean",
        "paper_claim_eligible": status == "clean",
        "claim_boundary_guard_passed": True,
        "claim_boundary_violations": [],
    }


def _base_capstone_v280() -> dict[str, Any]:
    return {
        "artifact": "experiment_2987_capstone_v280",
        "honest_verdict": (
            "complete: milestone_280_capstone; paper_ready=false; clean=3; "
            "flagged=5; blocked=2; missing=0"
        ),
        "paper_ready": False,
        "gaps_remaining": [
            "Repair is not paper-ready: the intent-preserving rerun did not clear cached-SOTA gates.",
            "Solver feedback is not paper-ready: numeric Z3 gains need durable provenance.",
            "GateMate is not hardware-ready: readback hash or passed host-visible smoke vector is absent.",
            "SSQA dual-BRAM register-map plan landed as projection-only evidence.",
        ],
    }


def _base_matrix_v15(*, all_clean: bool = False) -> dict[str, Any]:
    statuses = {
        "sota_cache": "clean",
        "hard_code_manifest": "clean",
        "repair": "clean" if all_clean else "flagged",
        "solver_provenance": "clean",
        "aquaforte_beaver_substrate": "clean" if all_clean else "flagged",
        "prompt_validator_protocol": "clean",
        "fr11_self_learning": "clean",
        "gatemate": "clean" if all_clean else "blocked",
        "ssqa": "clean" if all_clean else "missing",
    }
    rows = [
        _row(
            "exp2989_sota_cache",
            statuses["sota_cache"],
            "sota_cache_provenance",
            source="exp2989",
            summary={"sota_headline_ready": True, "n_live_transcripts": 1},
        ),
        _row(
            "exp2990_hard_code_manifest",
            statuses["hard_code_manifest"],
            "hard_code_manifest",
            source="exp2990",
            summary={"hard_code_stress_set_ready": True, "n_items": 24},
        ),
        _row(
            "exp2991_intent_preserving_repair",
            statuses["repair"],
            "repair_eval",
            source="exp2991",
            verdict="flagged: hard-set repair did not clear promotion gates",
            summary={"pass_at_1_delta": 0.4166666666666667, "repair_rerun_clean": all_clean},
        ),
        _row(
            "exp2992_solver_provenance",
            statuses["solver_provenance"],
            "solver_provenance",
            source="exp2992",
            summary={"solver_provenance_reproduced": True, "z3_execution_rate": 1.0},
        ),
        _row(
            "exp2993_aquaforte_beaver_substrate",
            statuses["aquaforte_beaver_substrate"],
            "aquaforte_beaver_substrate",
            source="exp2993",
            summary={"live_llm_retry_measured": True, "enumerator_only_fallback_measured": True},
        ),
        _row(
            "exp2994_prompt_validator_protocol",
            statuses["prompt_validator_protocol"],
            "prompt_validator_protocol",
            source="exp2994",
            summary={"prompt_validator_protocol_ready": True, "exact_verifier_authority_preserved": True},
        ),
        _row(
            "exp2995_fr11_trace_memory",
            statuses["fr11_self_learning"],
            "fr11_self_learning",
            source="exp2995",
            summary={"trace_memory_ready": True, "forgetting_guard_passed": True},
        ),
        _row(
            "exp2996_gatemate_readback_smoke",
            statuses["gatemate"],
            "hardware_readback_smoke",
            source="exp2996",
            verdict="blocked_flash_failed",
            summary={"hardware_smoke_boundary_recorded": True, "flash_succeeded": all_clean},
        ),
        _row(
            "exp2997_ssqa_dual_bram_rtl_pnr",
            statuses["ssqa"],
            "hardware_ssqa_rtl_pnr",
            source="exp2997",
            verdict="" if not all_clean else "complete: ssqa ready",
            summary={"ssqa_rtl_pnr_report_ready": all_clean},
        ),
    ]
    if not all_clean:
        rows.extend(
            [
                _row("carry_forward_v14:pilot", "pilot-only", "prior_v14_carry_forward"),
                _row("carry_forward_v14:projection", "projection-only", "prior_v14_carry_forward"),
                _row("exp2991_gate_skip_shadow", "gated-skipped", "repair_eval"),
            ]
        )
    claim_rows = {
        "sota_cache": rows[0],
        "hard_code_manifest": rows[1],
        "repair": rows[2],
        "solver_provenance": rows[3],
        "aquaforte_beaver_substrate": rows[4],
        "prompt_validator_protocol": rows[5],
        "fr11_self_learning": rows[6],
        "gatemate": rows[7],
        "ssqa": rows[8],
    }
    non_clean = [
        {"row_id": row["row_id"], "status": row["status"], "reason": row["source_honest_verdict"]}
        for row in rows
        if row["status"] in {"flagged", "blocked", "missing", "gated-skipped"}
    ]
    return {
        "artifact": "experiment_2998_cross_corpus_matrix_v15",
        "honest_verdict": "complete: matrix_v15_ready=true",
        "milestone": "2026.05.281",
        "matrix_v15_ready": True,
        "rows": rows,
        "claim_rows": claim_rows,
        "hardware_claim_boundary": {"forbidden_claims_absent": True},
        "self_learning_claim_boundary": {"status": "clean"},
        "paper_v6_claim_boundary": {"forbidden_claims_absent": True, "unsafe_headline_rows": []},
        "claim_boundary_violations": [],
        "unresolved_blockers": non_clean,
        "next_milestone_recommendations": [
            "Repair: resolve Exp 2991 artifact flags before promotion.",
            "GateMate: add host-visible readback before sampler-facing claims.",
        ],
    }


def _write_ready_sources(root: Path, *, all_clean: bool = False) -> None:
    _write_json(root, mod.MATRIX_V15_REL_PATH, _base_matrix_v15(all_clean=all_clean))
    _write_json(root, mod.CAPSTONE_V280_REL_PATH, _base_capstone_v280())
    for path in (
        mod.EXP2988_REL_PATH,
        mod.EXP2989_REL_PATH,
        mod.EXP2990_REL_PATH,
        mod.EXP2991_REL_PATH,
        mod.EXP2992_REL_PATH,
        mod.EXP2993_REL_PATH,
        mod.EXP2994_REL_PATH,
        mod.EXP2995_REL_PATH,
        mod.EXP2996_REL_PATH,
    ):
        _write_json(root, path, {"honest_verdict": "complete: source present"})


def test_req_report_2999_spec_anchor_exists() -> None:
    """REQ-REPORT-2999: OpenSpec declares the capstone contract first."""

    spec = Path("openspec/capabilities/research-reporting/spec.md").read_text(encoding="utf-8")

    assert "REQ-REPORT-2999" in spec
    assert "SCENARIO-REPORT-2999" in spec
    assert mod.OUTPUT_REL_PATH.as_posix() in spec


def test_scenario_report_2999_builds_capstone_without_publication(tmp_path: Path) -> None:
    """SCENARIO-REPORT-2999: .281 capstone reports readiness without publishing."""

    _write_ready_sources(tmp_path)

    artifact = mod.build_artifact(tmp_path, started_s=1.0, now_s=2.25)

    assert REQUIRED_FIELDS <= artifact.keys()
    assert artifact["capstone_ready"] is True
    assert artifact["paper_ready"] is False
    assert artifact["paper_v6_closer_to_readiness"] is True
    assert artifact["external_publication_triggered"] is False
    assert artifact["honest_verdict"].startswith("complete: capstone_ready=true; paper_ready=false")
    assert artifact["duration_s"] == pytest.approx(1.25)
    assert artifact["clean_artifacts"] == [
        "exp2989_sota_cache",
        "exp2990_hard_code_manifest",
        "exp2992_solver_provenance",
        "exp2994_prompt_validator_protocol",
        "exp2995_fr11_trace_memory",
    ]
    assert artifact["flagged_artifacts"] == [
        "exp2991_intent_preserving_repair",
        "exp2993_aquaforte_beaver_substrate",
    ]
    assert artifact["blocked_artifacts"] == ["exp2996_gatemate_readback_smoke"]
    assert artifact["missing_artifacts"] == ["exp2997_ssqa_dual_bram_rtl_pnr"]
    assert artifact["gated_skipped_artifacts"] == ["exp2991_gate_skip_shadow"]
    assert artifact["pilot_only_artifacts"] == ["carry_forward_v14:pilot"]
    assert artifact["projection_only_artifacts"] == ["carry_forward_v14:projection"]

    proofs = artifact["milestone_proof_summary"]
    assert proofs["sota_cache"]["status"] == "clean"
    assert proofs["repair"]["status"] == "flagged"
    assert proofs["solver_provenance"]["status"] == "clean"
    assert proofs["fr11_self_learning"]["status"] == "clean"
    assert proofs["gatemate"]["status"] == "blocked"
    assert proofs["ssqa"]["status"] == "missing"

    closed_text = " ".join(artifact["gaps_closed"])
    remaining_text = " ".join(artifact["gaps_remaining"]).lower()
    assert "SOTA cache" in closed_text
    assert "solver provenance" in closed_text.lower()
    assert "FR-11" in closed_text
    assert "repair" in remaining_text
    assert "gatemate" in remaining_text
    assert "ssqa" in remaining_text
    assert artifact["source_checksums"][mod.MATRIX_V15_REL_PATH.as_posix()] == _sha256(
        tmp_path / mod.MATRIX_V15_REL_PATH
    )
    assert artifact["source_artifacts_read"][-1]["experiment_id"] == "exp2997"
    assert artifact["source_artifacts_read"][-1]["present"] is False


def test_req_report_2999_blocks_when_required_matrix_missing(tmp_path: Path) -> None:
    """REQ-REPORT-2999: required matrix and prior capstone fail closed."""

    _write_json(tmp_path, mod.CAPSTONE_V280_REL_PATH, _base_capstone_v280())

    artifact = mod.build_artifact(tmp_path, started_s=3.0, now_s=3.5)

    assert artifact["capstone_ready"] is False
    assert artifact["paper_ready"] is False
    assert artifact["honest_verdict"] == "blocked_required_upstream_missing"
    assert artifact["required_upstream_errors"] == [
        {
            "experiment_id": "exp2998",
            "path": mod.MATRIX_V15_REL_PATH.as_posix(),
            "reason": "missing_or_malformed_artifact",
        }
    ]
    assert artifact["external_publication_triggered"] is False


def test_req_report_2999_paper_ready_requires_all_claim_rows_clean(tmp_path: Path) -> None:
    """REQ-REPORT-2999: paper_ready is true only from clean local evidence."""

    _write_ready_sources(tmp_path, all_clean=True)

    artifact = mod.build_artifact(tmp_path, started_s=4.0, now_s=4.125)

    assert artifact["capstone_ready"] is True
    assert artifact["paper_ready"] is True
    assert artifact["paper_ready_blockers"] == []
    assert artifact["honest_verdict"].startswith("complete: capstone_ready=true; paper_ready=true")
    assert artifact["external_publication_triggered"] is False


def test_req_report_2999_boundary_failures_and_carry_forward_blockers(tmp_path: Path) -> None:
    """REQ-REPORT-2999: paper and hardware boundary failures remain blockers."""

    matrix = _base_matrix_v15(all_clean=True)
    matrix["rows"].append(_row("carry_forward_v14:old_flag", "flagged", "prior_v14_carry_forward"))
    matrix["claim_boundary_violations"] = [{"row_id": "x", "violation": "unsafe_claim"}]
    matrix["paper_v6_claim_boundary"] = {
        "forbidden_claims_absent": False,
        "unsafe_headline_rows": ["x"],
    }
    matrix["hardware_claim_boundary"] = {"forbidden_claims_absent": False}
    _write_json(tmp_path, mod.MATRIX_V15_REL_PATH, matrix)
    _write_json(tmp_path, mod.CAPSTONE_V280_REL_PATH, _base_capstone_v280())

    artifact = mod.build_artifact(tmp_path, started_s=4.0, now_s=4.25)

    assert artifact["paper_ready"] is False
    assert "matrix_v15 claim_boundary_violations is non-empty" in artifact["paper_ready_blockers"]
    assert "paper-v6 forbidden-claim boundary failed" in artifact["paper_ready_blockers"]
    assert "hardware forbidden-claim boundary failed" in artifact["paper_ready_blockers"]
    assert "Prior matrix carry-forward blockers remain: 1 rows." in artifact["gaps_remaining"]


def test_req_report_2999_write_artifact_and_main_persist_json(tmp_path: Path) -> None:
    """REQ-REPORT-2999: write_artifact emits the deliverable JSON."""

    _write_ready_sources(tmp_path)

    output = mod.write_artifact(tmp_path, started_s=5.0, now_s=5.5)
    saved = json.loads(output.read_text(encoding="utf-8"))

    assert output == tmp_path / mod.OUTPUT_REL_PATH
    assert saved["capstone_ready"] is True
    assert saved["duration_s"] == pytest.approx(0.5)
    assert mod.main(tmp_path) == 0


def test_req_report_2999_helper_edges_keep_sources_honest(tmp_path: Path) -> None:
    """REQ-REPORT-2999: helpers keep absent and malformed inputs honest."""

    missing = tmp_path / "missing.json"
    malformed = tmp_path / "malformed.json"
    list_payload = tmp_path / "list.json"
    malformed.write_text("{bad-json}\n", encoding="utf-8")
    list_payload.write_text("[1, 2, 3]\n", encoding="utf-8")

    assert mod.read_json_object(missing) == {}
    assert mod.read_json_object(malformed) == {}
    assert mod.read_json_object(list_payload) == {}
    assert mod.sha256_file(missing) is None
    assert mod._row_ids_by_status([{"row_id": "x", "status": "clean"}, {"status": "clean"}], "clean") == [
        "x"
    ]
    assert mod._claim_statuses({"claim_rows": {"x": {"status": "clean"}, "y": []}}) == {
        "x": "clean"
    }
