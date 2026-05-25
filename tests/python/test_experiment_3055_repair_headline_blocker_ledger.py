"""Tests for Exp 3055 repair headline retirement and blocker ledger.

Spec refs: REQ-REPORT-3055, SCENARIO-REPORT-3055.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from carnot.reporting import repair_headline_blocker_ledger_3055 as mod


REQUIRED_FIELDS = {
    "repair_headline_retirement_ready",
    "retired_repair_claims",
    "still_bounded_repair_claims",
    "rerun_prerequisites",
    "manifest_updates",
    "source_artifacts",
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


def _write_text(root: Path, rel_path: Path, text: str) -> None:
    path = root / rel_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _claim(
    claim_id: str,
    proposed: str,
    source_artifact: str,
    source_field: str,
    *,
    row_id: str | None = None,
    status: str = "retired",
) -> dict[str, Any]:
    return {
        "row_id": row_id or f"exp3029:{claim_id}",
        "status": status,
        "source_artifact": source_artifact,
        "source_field": source_field,
        "blocker_class": "retired_claim" if status == "retired" else "bounded_claim",
        "claim_scope": "retired_repair_claim"
        if status == "retired"
        else "repair_headline_boundary",
        "evidence_class": "repair_retirement_boundary"
        if status == "retired"
        else "repair_reconciliation",
        "summary": {
            "evidence": {
                "claim_id": claim_id,
                "classification": status,
                "proposed_repair_claim": proposed,
                "allowed_wording": "Use bounded wording only.",
                "blockers": ["matrix .285 does not support headline promotion"],
                "required_support_fields": ["matrix repair status clean"],
                "observed_support_fields": {"matrix_repair_status": "bounded"},
            },
            "repair_claim_status": "bounded",
            "remaining_blocker_count": 9,
            "repair_promotion_candidate": False,
        },
    }


def _blocker(row_id: str, classification: str, source_field: str) -> dict[str, Any]:
    return {
        "row_id": row_id,
        "classification": classification,
        "blocking": True,
        "source_artifact": mod.EXP3041_REL_PATH.as_posix(),
        "source_field": source_field,
        "experiment_id": "exp3028",
        "rationale": "repair blocker fixture",
        "evidence": [{"kind": "TAUTOLOGY", "severity": "critical", "detail": "fixture"}],
    }


def _exp3041() -> dict[str, Any]:
    return {
        "artifact": "experiment_3041_matrix_capstone_adversarial_flag_hygiene_v1",
        "flag_hygiene_ready": True,
        "true_blocker_rows": [
            _blocker("exp3028:adversarial_flags", "true_blocker", "true_blocker_rows[0]")
        ],
        "missing_metadata_rows": [
            _blocker("exp3028:methodology_missing", "missing_metadata", "missing_metadata_rows[0]")
        ],
        "unresolved_bound_rows": [
            _blocker("capstone:repair_bounded", "unresolved_bound", "unresolved_bound_rows[0]")
        ],
        "aggregation_false_positive_rows": [],
        "hardware_blocked_rows": [],
        "gate_skipped_rows": [],
        "consumer_contract": {
            "required_row_fields": [
                "row_id",
                "classification",
                "blocking",
                "source_artifact",
                "source_field",
            ]
        },
        "honest_verdict": "complete: flag_hygiene_ready=true",
    }


def _exp3042() -> dict[str, Any]:
    retired = [
        _claim(
            "headline_sota_repair_clean_methodology",
            "The repair result is promotable as a headline SOTA repair claim.",
            "results/experiment_3029_repair_promotion_boundary_audit_v2.json",
            "retired_or_blocked_claims[headline_sota_repair_clean_methodology]",
        ),
        _claim(
            "unsupported_exp3016_headline_repair_promotion",
            "The original Exp 3016 headline repair claim is promotable without the Exp 3028 boundary.",
            "results/experiment_3029_repair_promotion_boundary_audit_v2.json",
            "retired_or_blocked_claims[unsupported_exp3016_headline_repair_promotion]",
        ),
    ]
    bounded = _claim(
        "exp3028_clean_repair_candidate",
        "Exp 3028 supplies clean acceptance-controlled SOTA repair evidence.",
        "results/experiment_3029_repair_promotion_boundary_audit_v2.json",
        "bounded_claims[exp3028_clean_repair_candidate]",
        status="bounded",
    )
    return {
        "artifact": "experiment_3042_repair_promotion_reconciliation_v3",
        "repair_reconciliation_ready": True,
        "repair_claim_status": "bounded",
        "repair_promotion_candidate": False,
        "remaining_blockers": [
            _blocker("exp3028:adversarial_flags", "true_blocker", "remaining_blockers[0]"),
            _blocker("exp3028:methodology_missing", "missing_metadata", "remaining_blockers[1]"),
        ],
        "repair_delta_summary": {
            "n_tasks": 24,
            "n_live_transcripts": 24,
            "pass_at_1_delta": 0.375,
            "pass_at_k_delta": 0.375,
            "false_accept_delta": 0.0,
        },
        "prior_repair_status": {"matrix_v18_exp3028_status": "flagged"},
        "retired_or_blocked_claims": retired,
        "bounded_claims": [bounded],
        "honest_verdict": "complete: repair_claim_status=bounded",
    }


def _matrix_v19() -> dict[str, Any]:
    retired = [
        _claim(
            "headline_sota_repair_clean_methodology",
            "The repair result is promotable as a headline SOTA repair claim.",
            "results/experiment_3029_repair_promotion_boundary_audit_v2.json",
            "retired_or_blocked_claims[headline_sota_repair_clean_methodology]",
        ),
        _claim(
            "unsupported_exp3016_headline_repair_promotion",
            "The original Exp 3016 headline repair claim is promotable without the Exp 3028 boundary.",
            "results/experiment_3029_repair_promotion_boundary_audit_v2.json",
            "retired_or_blocked_claims[unsupported_exp3016_headline_repair_promotion]",
        ),
    ]
    repair_row = _claim(
        "repair_headline_status",
        "Repair headline status remains bounded.",
        mod.EXP3042_REL_PATH.as_posix(),
        "repair_claim_status",
        row_id="repair:headline_status",
        status="bounded",
    )
    return {
        "artifact": "experiment_3052_cross_corpus_matrix_v19",
        "matrix_v19_ready": True,
        "rows_total": 3,
        "repair_claim_status": "bounded",
        "rows": [repair_row, *retired],
        "honest_verdict": "complete: matrix_v19_ready=true",
    }


def _capstone_v285() -> dict[str, Any]:
    retired = _matrix_v19()["rows"][1:]
    bounded = [_matrix_v19()["rows"][0]]
    return {
        "artifact": "experiment_3053_capstone_v285",
        "capstone_ready": True,
        "paper_ready": False,
        "repair_claim_status": "bounded",
        "bounded_claims": bounded,
        "retired_claims": retired,
        "blocked_claims": [
            _claim(
                "repair_blocker",
                "Repair rerun is blocked until methodology evidence exists.",
                mod.EXP3042_REL_PATH.as_posix(),
                "remaining_blockers",
                row_id="v18:exp3028",
                status="flagged",
            )
        ],
        "next_milestone_recommendation": (
            "2026.05.286: retire unsupported repair headline wording; rerun repair "
            "promotion only after repair_status=bounded clears blockers."
        ),
        "honest_verdict": "complete: capstone_ready=true; repair_claim_status=bounded",
    }


def _manifest_text(include_entry: bool = True) -> str:
    extra = ""
    if include_entry:
        extra = f"""
  - id: {mod.MANIFEST_ENTRY_ID}
    reason: |
      Retire unsupported repair headline wording until deterministic evidence
      clears the Exp 3055 rerun prerequisites.
    blocked_patterns:
      - "headline SOTA repair"
      - "original Exp 3016 headline repair claim"
    retired_milestone: "2026.05.285"
    retired_by_artifact: "{mod.OUTPUT_REL_PATH.as_posix()}"
    operator_reopen_required: true
    retire_if_same_verdict: true
"""
    return "retired: []\nretired_experiments: []\nretired_extras:\n" + extra


def _write_sources(root: Path, *, manifest_entry: bool = True) -> None:
    _write_json(root, mod.EXP3041_REL_PATH, _exp3041())
    _write_json(root, mod.EXP3042_REL_PATH, _exp3042())
    _write_json(root, mod.MATRIX_V19_REL_PATH, _matrix_v19())
    _write_json(root, mod.CAPSTONE_V285_REL_PATH, _capstone_v285())
    _write_text(root, mod.MANIFEST_REL_PATH, _manifest_text(include_entry=manifest_entry))


def test_req_report_3055_spec_anchor_and_script_exist() -> None:
    """REQ-REPORT-3055: OpenSpec declares the ledger contract first."""

    spec = (mod.REPO_ROOT / "openspec/capabilities/research-reporting/spec.md").read_text(
        encoding="utf-8"
    )

    assert "REQ-REPORT-3055" in spec
    assert "SCENARIO-REPORT-3055" in spec
    assert mod.OUTPUT_REL_PATH.as_posix() in spec
    assert mod.SCRIPT_REL_PATH.exists()


def test_scenario_report_3055_builds_ready_repair_blocker_ledger(tmp_path: Path) -> None:
    """SCENARIO-REPORT-3055: retired wording and bounded evidence stay separated."""

    _write_sources(tmp_path)

    artifact = mod.build_artifact(tmp_path, started_s=2.0, now_s=4.5)
    retired = {row["claim_id"]: row for row in artifact["retired_repair_claims"]}
    bounded = {row["row_id"]: row for row in artifact["still_bounded_repair_claims"]}
    prereqs = {row["gate"] for row in artifact["rerun_prerequisites"]}

    assert REQUIRED_FIELDS <= artifact.keys()
    assert FORBIDDEN_TOP_LEVEL.isdisjoint(artifact)
    assert artifact["repair_headline_retirement_ready"] is True
    assert artifact["duration_s"] == pytest.approx(2.5)
    assert artifact["honest_verdict"].startswith("complete:")

    assert set(retired) == {
        "headline_sota_repair_clean_methodology",
        "unsupported_exp3016_headline_repair_promotion",
    }
    assert all(row["status"] == "retired_for_headline_use" for row in retired.values())
    assert all(row["matrix_v20_consumable"] is True for row in retired.values())
    assert (
        "headline SOTA repair"
        in retired["headline_sota_repair_clean_methodology"]["unsupported_headline_wording"]
    )

    assert "repair:headline_status" in bounded
    assert bounded["repair:headline_status"]["status"] == "bounded_evidence_not_headline"
    assert bounded["repair:headline_status"]["repair_promotion_candidate"] is False
    assert artifact["repair_claim_status"] == "bounded"
    assert artifact["extracted_repair_blockers"]

    assert prereqs == {
        "deterministic_fingerprint",
        "seed",
        "duration_sanity",
        "de_tautology_metrics",
        "verifier_gain",
        "exact_checker_authority",
    }
    assert all(row["required"] is True for row in artifact["rerun_prerequisites"])
    assert artifact["manifest_updates"] == [
        {
            "id": mod.MANIFEST_ENTRY_ID,
            "path": mod.MANIFEST_REL_PATH.as_posix(),
            "applied": True,
            "reason": "CLAUDE.md failed-rerun and exclusion-manifest discipline requires retired headline scope to be traceable.",
            "retired_by_artifact": mod.OUTPUT_REL_PATH.as_posix(),
        }
    ]
    assert artifact["inference_substrate"] == {
        "kind": "aggregation_from_upstream_artifacts",
        "source": "checked_in_artifacts",
        "executes_models": False,
        "executes_hardware": False,
        "executes_conductor": False,
        "no_live_llm_inference": True,
    }
    source_by_path = {row["path"]: row for row in artifact["source_artifacts"]}
    assert source_by_path[mod.EXP3041_REL_PATH.as_posix()]["sha256"] == _sha256(
        tmp_path / mod.EXP3041_REL_PATH
    )


def test_req_report_3055_blocks_ready_when_manifest_entry_is_absent(tmp_path: Path) -> None:
    """REQ-REPORT-3055: manifest-readiness is explicit when retirement is required."""

    _write_sources(tmp_path, manifest_entry=False)

    artifact = mod.build_artifact(tmp_path)

    assert artifact["repair_headline_retirement_ready"] is False
    assert artifact["manifest_updates"][0]["applied"] is False
    assert artifact["honest_verdict"].startswith("blocked_manifest_update_missing:")
    assert "manifest update missing" in artifact["blocked_reasons"]


def test_req_report_3055_write_artifact_emits_deliverable(tmp_path: Path) -> None:
    """REQ-REPORT-3055: write_artifact writes the stable JSON deliverable."""

    _write_sources(tmp_path)

    output = mod.write_artifact(tmp_path, started_s=1.0, now_s=1.75)
    payload = json.loads(output.read_text(encoding="utf-8"))

    assert output == tmp_path / mod.OUTPUT_REL_PATH
    assert payload["repair_headline_retirement_ready"] is True
    assert payload["duration_s"] == pytest.approx(0.75)


def test_req_report_3055_fails_closed_when_matrix_cannot_consume_decisions(
    tmp_path: Path,
) -> None:
    """REQ-REPORT-3055: matrix v20 readiness requires row ids and source fields."""

    _write_sources(tmp_path)
    matrix = _matrix_v19()
    del matrix["rows"][1]["source_field"]
    _write_json(tmp_path, mod.MATRIX_V19_REL_PATH, matrix)

    artifact = mod.build_artifact(tmp_path)

    assert artifact["repair_headline_retirement_ready"] is False
    assert artifact["honest_verdict"].startswith("blocked_matrix_v20_not_consumable:")
    assert "matrix v20 cannot consume every repair decision" in artifact["blocked_reasons"]


def test_req_report_3055_helper_edges_fail_closed(tmp_path: Path) -> None:
    """REQ-REPORT-3055: malformed sources and irrelevant rows do not fabricate evidence."""

    bad_json = tmp_path / "bad.json"
    bad_json.write_text("{", encoding="utf-8")
    missing = tmp_path / "missing.json"

    assert mod.read_json_object(missing) == {}
    assert mod.read_json_object(bad_json) == {}
    assert mod.sha256_file(missing) is None
    assert mod._source_errors(
        [
            {"required": False, "present": False, "readable_json_object": False},
            {"required": True, "present": False, "readable_json_object": False, "path": "x"},
            {"required": True, "present": True, "readable_json_object": False, "path": "y"},
        ]
    ) == [
        {"path": "x", "reason": "missing_required_source"},
        {"path": "y", "reason": "malformed_required_json"},
    ]

    payloads = {
        "exp3041": {"true_blocker_rows": [{"row_id": "unrelated", "classification": "noise"}]},
        "exp3042": {"remaining_blockers": [{"row_id": "unrelated", "classification": "noise"}]},
        "exp3052": {
            "rows": [
                {
                    "row_id": "noise:claim",
                    "status": "retired",
                    "source_artifact": "results/noise.json",
                    "source_field": "rows[0]",
                    "summary": {"evidence": {"claim_id": "noise_claim"}},
                },
                {
                    "row_id": "noise:bounded",
                    "status": "bounded",
                    "source_artifact": "results/noise.json",
                    "source_field": "rows[1]",
                },
                {
                    "row_id": "noise:blocked",
                    "status": "blocked",
                    "source_artifact": "results/noise.json",
                    "source_field": "rows[2]",
                },
            ]
        },
        "exp3053": {"retired_claims": [], "bounded_claims": [], "blocked_claims": []},
    }

    assert mod._retired_repair_claims(payloads) == []
    assert mod._bounded_repair_claims(payloads) == []
    assert mod._repair_blockers(payloads) == []
    assert mod._retired_from_blockers(
        {
            "true_blocker_rows": [
                {
                    "row_id": "exp3029:headline",
                    "source_field": "retired_or_blocked_claims[headline]",
                    "evidence": {"classification": "retired", "claim_id": "headline"},
                }
            ]
        }
    )

    assert mod._evidence_mapping({"evidence": {"claim_id": "repair_claim"}}) == {
        "claim_id": "repair_claim"
    }
    assert mod._evidence({}) == {}
    assert (
        mod._source_artifact_from_row(
            {"evidence": {"source_artifact_path": "results/repair_source.json"}}
        )
        == "results/repair_source.json"
    )
    assert mod._source_artifact_from_row({"evidence": {}}) == ""
    assert mod._manifest_updates(tmp_path, []) == []
    assert mod._manifest_updates(tmp_path, [{"claim_id": "repair_claim"}])[0]["applied"] is False

    blocked = mod._blocked_reasons(
        source_errors=[{"path": "x", "reason": "missing"}],
        manifest_updates=[{"applied": True}],
        retired_claims=[],
        bounded_claims=[],
        consumability_errors=[],
        matrix={},
        capstone={},
    )
    assert blocked == [
        "required source artifacts missing or malformed",
        "matrix v19 is not ready",
        "capstone v285 is not ready",
        "no retired repair headline claims found",
        "no bounded repair claims found",
    ]
    assert (
        mod._honest_verdict(
            {"repair_headline_retirement_ready": False, "blocked_reasons": ["other"]}
        )
        == "blocked_precondition: repair headline retirement ledger incomplete"
    )
