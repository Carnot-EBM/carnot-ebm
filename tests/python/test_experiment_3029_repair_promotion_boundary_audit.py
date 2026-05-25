"""Tests for Exp 3029 repair promotion boundary audit.

Spec refs: REQ-REPORT-3029, SCENARIO-REPORT-3029.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from carnot.reporting import repair_promotion_boundary_audit_3029 as mod


REQUIRED_FIELDS = {
    "repair_promotion_boundary_ready",
    "repair_claim_status",
    "promotable_claims",
    "bounded_claims",
    "retired_or_blocked_claims",
    "cited_upstream_artifacts",
    "inference_substrate",
    "honest_verdict",
}

HEADLINE_MODEL = "unsloth/gemma-4-26B-A4B-it-GGUF"
FORBIDDEN_TOP_LEVEL = {
    "model_specs",
    "target_model",
    "cuda",
    "gpu",
    "gpu_inventory",
    "gguf",
    "gguf_cache_paths",
    "headline_models_used",
    "model_checksums",
}


def _write_json(root: Path, rel_path: Path, payload: dict[str, Any]) -> None:
    path = root / rel_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _exp3027() -> dict[str, Any]:
    return {
        "artifact": "experiment_3027_adversarial_flag_methodology_corrigendum",
        "methodology_corrigendum_ready": True,
        "sota_headline_ready": True,
        "repair_rerun_required": True,
        "repair_rerun_decision": {
            "decision": "live_rerun_required",
            "reason": "Exp 3016 is missing required reconstruction metadata",
        },
        "missing_metadata_rows": [
            {
                "row_id": "exp3016_repair_acceptance_controller",
                "classification": "missing_metadata",
                "supporting_fields": [
                    {"field": "random_seed", "value": None},
                    {"field": "transcript_sha256s", "value": None},
                ],
            }
        ],
        "honest_verdict": "complete: methodology_corrigendum_ready=true",
        "inference_substrate": {
            "kind": "aggregation_from_upstream_artifacts",
            "no_top_level_live_model_metadata": True,
        },
    }


def _exp3028(
    *,
    clean: bool = True,
    false_accept_delta: float = 0.0,
    tautology_gate_clean: bool = True,
    n_tasks: int = 24,
    n_live_transcripts: int = 24,
    legacy_smoke_only_used: bool = False,
) -> dict[str, Any]:
    return {
        "artifact": "experiment_3028_sota_repair_clean_methodology_rerun_v2",
        "clean_repair_rerun_ready": clean,
        "repair_controller_clean": clean,
        "clean_repair_claim_promotable_candidate": clean,
        "n_tasks": n_tasks,
        "n_live_transcripts": n_live_transcripts,
        "accepted_candidate_count": 9 if clean else 0,
        "rejected_candidate_count": 15,
        "legacy_smoke_only_used": legacy_smoke_only_used,
        "pass_at_1_delta": 0.375 if clean else 0.0,
        "pass_at_k_delta": 0.375 if clean else 0.0,
        "syntax_failure_rate_delta": 0.0,
        "schema_failure_rate_delta": 0.0,
        "false_accept_delta": false_accept_delta,
        "tautology_gate_clean": tautology_gate_clean,
        "intent_drift_count": 0,
        "reproducibility_checksum": "exp3028-checksum",
        "model_specs": [
            {
                "hf_id": HEADLINE_MODEL,
                "model_path": "/models/gemma.gguf",
                "checksum": "bounded-checksum",
            }
        ],
        "headline_models_used": [HEADLINE_MODEL],
        "inference_substrate": {
            "kind": "clean_repair_reconstruction",
            "live_repair_generation_run": False,
            "model_load_attempted": False,
            "reconstruction_mode": "exp3016_nested_live_transcripts",
            "gguf_cache_paths": {HEADLINE_MODEL: "/models/gemma.gguf"},
        },
        "honest_verdict": "complete: clean_repair_rerun_ready=true; n_tasks=24"
        if clean
        else "complete_flagged: clean_repair_rerun_ready=false",
    }


def _exp3016() -> dict[str, Any]:
    return {
        "artifact": "experiment_3016_sota_repair_rerun_with_acceptance_controller",
        "repair_controller_clean": True,
        "headline_result": True,
        "n_tasks": 24,
        "n_live_transcripts": 24,
        "pass_at_1_delta": 0.375,
        "pass_at_k_delta": 0.375,
        "false_accept_delta": 0.0,
        "tautology_gate_clean": True,
        "syntax_failure_rate_delta": 0.0,
        "schema_failure_rate_delta": 0.0,
        "model_specs": {"headline_models": [HEADLINE_MODEL]},
        "headline_models_used": [HEADLINE_MODEL],
        "model_checksums": {HEADLINE_MODEL: {"bounded_sha256": "bounded-checksum"}},
        "honest_verdict": "complete: acceptance-controlled SOTA repair rerun gates passed",
        "inference_substrate": "live_sota_gguf_repair_with_acceptance_controller",
    }


def _repair_row(status: str) -> dict[str, Any]:
    upstream_flags = [] if status == "clean" else ["TAUTOLOGY:critical", "METHODOLOGY_MISSING:warn"]
    return {
        "row_id": "exp3016_repair_acceptance_controller",
        "source_experiment_id": "exp3016",
        "status": status,
        "claim_class": "repair_eval",
        "evidence_type": "live_llm_repair_source",
        "inference_substrate": "live_sota_gguf_repair_with_acceptance_controller",
        "headline_eligible": status == "clean",
        "paper_claim_eligible": status == "clean",
        "claim_boundary": "Repair promotion requires clean deltas and clean methodology flags.",
        "claim_boundary_guard_passed": True,
        "claim_boundary_violations": [],
        "source_honest_verdict": "complete: acceptance-controlled SOTA repair rerun gates passed",
        "summary": {
            "repair_controller_clean": True,
            "headline_result": True,
            "n_tasks": 24,
            "n_live_transcripts": 24,
            "pass_at_1_delta": 0.375,
            "pass_at_k_delta": 0.375,
            "false_accept_delta": 0.0,
            "syntax_failure_rate_delta": 0.0,
            "schema_failure_rate_delta": 0.0,
            "tautology_gate_clean": True,
        },
        "upstream_flags": upstream_flags,
    }


def _matrix(*, repair_status: str = "flagged") -> dict[str, Any]:
    return {
        "artifact": "experiment_3024_cross_corpus_matrix_v17",
        "matrix_v17_ready": True,
        "inference_substrate": mod.INFERENCE_SUBSTRATE_KIND,
        "claim_rows": {"exp3016_repair": _repair_row(repair_status)},
        "rows": [_repair_row(repair_status)],
        "repaired_claims": ["exp3020_fr11_verifier_feedback_controller"],
        "still_blocked_claims": []
        if repair_status == "clean"
        else ["exp3016_repair_acceptance_controller_flagged"],
        "claim_boundary_violations": [],
        "honest_verdict": f"complete: matrix_v17_ready=true; repair_status={repair_status}",
    }


def _capstone(*, repair_status: str = "flagged", repair_promotable: bool = False) -> dict[str, Any]:
    return {
        "artifact": "experiment_3025_capstone_v283",
        "capstone_ready": True,
        "paper_ready": repair_promotable,
        "claim_promotion_decisions": {
            "repair": {
                "row_id": "exp3016_repair_acceptance_controller",
                "status": repair_status,
                "promotable": repair_promotable,
                "claim_boundary": "Exp 3016 repair is promotable only when flags are absent.",
                "summary": _repair_row(repair_status)["summary"],
                "upstream_flags": _repair_row(repair_status)["upstream_flags"],
                "source_honest_verdict": "complete: acceptance-controlled SOTA repair rerun gates passed",
            }
        },
        "paper_ready_blockers": []
        if repair_promotable
        else ["repair row exp3016_repair_acceptance_controller is flagged"],
        "honest_verdict": "complete: capstone_ready=true; paper_ready=false",
        "inference_substrate": mod.INFERENCE_SUBSTRATE_KIND,
    }


def _write_sources(
    root: Path,
    *,
    repair_status: str = "flagged",
    repair_promotable: bool = False,
    exp3028: dict[str, Any] | None = None,
) -> None:
    _write_json(root, mod.EXP3027_REL_PATH, _exp3027())
    _write_json(root, mod.EXP3028_REL_PATH, exp3028 or _exp3028())
    _write_json(root, mod.EXP3016_REL_PATH, _exp3016())
    _write_json(root, mod.MATRIX_V17_REL_PATH, _matrix(repair_status=repair_status))
    _write_json(
        root,
        mod.CAPSTONE_V283_REL_PATH,
        _capstone(repair_status=repair_status, repair_promotable=repair_promotable),
    )


def _claim_by_id(claims: list[dict[str, Any]], claim_id: str) -> dict[str, Any]:
    return next(row for row in claims if row["claim_id"] == claim_id)


def test_req_report_3029_spec_anchor_and_script_exist() -> None:
    """REQ-REPORT-3029: OpenSpec declares the repair boundary artifact."""

    spec = Path("openspec/capabilities/research-reporting/spec.md").read_text(encoding="utf-8")
    script = Path("scripts/experiment_3029_repair_promotion_boundary_audit_v2.py")

    assert "REQ-REPORT-3029" in spec
    assert "SCENARIO-REPORT-3029" in spec
    assert mod.OUTPUT_REL_PATH.as_posix() in spec
    assert script.is_file()


def test_scenario_report_3029_bounds_clean_evidence_when_matrix_is_stale(tmp_path: Path) -> None:
    """SCENARIO-REPORT-3029: clean Exp 3028 evidence is bounded by stale matrix rows."""

    _write_sources(tmp_path)

    artifact = mod.build_artifact(tmp_path, started_s=10.0, now_s=12.25)

    assert REQUIRED_FIELDS <= artifact.keys()
    assert artifact["repair_promotion_boundary_ready"] is True
    assert artifact["repair_claim_status"] == "bounded"
    assert artifact["promotable_claims"] == []
    assert artifact["duration_s"] == pytest.approx(2.25)
    assert artifact["honest_verdict"].startswith("complete:")
    assert artifact["inference_substrate"] == {
        "kind": mod.INFERENCE_SUBSTRATE_KIND,
        "no_live_llm_inference": True,
        "no_new_repair_generation": True,
        "no_verifier_scoring_run": True,
        "no_top_level_live_model_metadata": True,
        "source_model_metadata_location": "cited_upstream_artifacts[].model_provenance",
    }

    bounded = _claim_by_id(artifact["bounded_claims"], "exp3028_clean_repair_candidate")
    assert "Exp 3028" in bounded["allowed_wording"]
    assert "bounded" in bounded["allowed_wording"]
    assert bounded["blockers"] == [
        "matrix repair row is flagged",
        "capstone repair decision is not promotable",
        "capstone still lists repair paper-ready blocker",
    ]

    retired = _claim_by_id(
        artifact["retired_or_blocked_claims"],
        "unsupported_exp3016_headline_repair_promotion",
    )
    assert "retire" in retired["allowed_wording"].lower()
    assert "Exp 3027 requires repair rerun or reconstruction" in retired["blockers"]

    assert len(artifact["claim_boundary_table"]) == 3
    for row in artifact["claim_boundary_table"]:
        assert {
            "claim_id",
            "proposed_repair_claim",
            "required_support_fields",
            "observed_support_fields",
            "blockers",
            "allowed_wording",
        } <= row.keys()

    assert not FORBIDDEN_TOP_LEVEL.intersection(artifact)
    serialized_citations = json.dumps(artifact["cited_upstream_artifacts"], sort_keys=True)
    assert HEADLINE_MODEL in serialized_citations
    artifact_without_citations = dict(artifact)
    artifact_without_citations.pop("cited_upstream_artifacts")
    assert HEADLINE_MODEL not in json.dumps(artifact_without_citations, sort_keys=True)
    assert artifact["source_checksums"][mod.EXP3028_REL_PATH.as_posix()] == _sha256(
        tmp_path / mod.EXP3028_REL_PATH
    )
    assert artifact["status_updates_written"] is False


def test_req_report_3029_promotes_when_all_sources_are_clean(tmp_path: Path) -> None:
    """REQ-REPORT-3029: clean source, matrix, and capstone support promote wording."""

    _write_sources(tmp_path, repair_status="clean", repair_promotable=True)

    artifact = mod.build_artifact(tmp_path, started_s=1.0, now_s=1.5)

    assert artifact["repair_promotion_boundary_ready"] is True
    assert artifact["repair_claim_status"] == "clean"
    promoted = _claim_by_id(artifact["promotable_claims"], "headline_sota_repair_clean_methodology")
    assert "may state" in promoted["allowed_wording"]
    assert promoted["blockers"] == []
    assert (
        _claim_by_id(artifact["claim_boundary_table"], "exp3028_clean_repair_candidate")[
            "classification"
        ]
        == "promotable"
    )


def test_req_report_3029_retires_unsafe_repair_claim(tmp_path: Path) -> None:
    """REQ-REPORT-3029: unsafe rerun evidence retires repair promotion language."""

    unsafe = _exp3028(clean=False, false_accept_delta=0.25, tautology_gate_clean=False)
    _write_sources(tmp_path, exp3028=unsafe)

    artifact = mod.build_artifact(tmp_path, started_s=2.0, now_s=2.5)

    assert artifact["repair_promotion_boundary_ready"] is True
    assert artifact["repair_claim_status"] == "retired"
    assert artifact["bounded_claims"] == []
    unsafe_claim = _claim_by_id(
        artifact["retired_or_blocked_claims"], "exp3028_clean_repair_candidate"
    )
    assert "false_accept_delta is positive" in unsafe_claim["blockers"]
    assert "tautology_gate_clean is not true" in unsafe_claim["blockers"]
    assert "retire" in unsafe_claim["allowed_wording"].lower()


def test_req_report_3029_blocks_when_required_source_missing(tmp_path: Path) -> None:
    """REQ-REPORT-3029: missing required upstream artifacts fail closed."""

    _write_json(tmp_path, mod.EXP3027_REL_PATH, _exp3027())

    artifact = mod.build_artifact(tmp_path, started_s=3.0, now_s=3.125)

    assert artifact["repair_promotion_boundary_ready"] is False
    assert artifact["repair_claim_status"] == "blocked"
    assert artifact["honest_verdict"] == "blocked_required_upstream_missing"
    assert artifact["required_source_errors"] == [
        {
            "experiment_id": "exp3028",
            "path": mod.EXP3028_REL_PATH.as_posix(),
            "reason": "missing_or_malformed_artifact",
        },
        {
            "experiment_id": "exp3016",
            "path": mod.EXP3016_REL_PATH.as_posix(),
            "reason": "missing_or_malformed_artifact",
        },
        {
            "experiment_id": "exp3024",
            "path": mod.MATRIX_V17_REL_PATH.as_posix(),
            "reason": "missing_or_malformed_artifact",
        },
        {
            "experiment_id": "exp3025",
            "path": mod.CAPSTONE_V283_REL_PATH.as_posix(),
            "reason": "missing_or_malformed_artifact",
        },
    ]


def test_req_report_3029_write_main_and_helper_edges(tmp_path: Path) -> None:
    """REQ-REPORT-3029: persistence and helper edges stay deterministic."""

    _write_sources(tmp_path)
    output = mod.write_artifact(tmp_path, started_s=4.0, now_s=5.0)
    saved = json.loads(output.read_text(encoding="utf-8"))

    assert output == tmp_path / mod.OUTPUT_REL_PATH
    assert saved["repair_claim_status"] == "bounded"
    assert mod.main(tmp_path) == 0

    malformed = tmp_path / "malformed.json"
    list_payload = tmp_path / "list.json"
    malformed.write_text("{bad-json}\n", encoding="utf-8")
    list_payload.write_text("[1, 2]\n", encoding="utf-8")
    assert mod.read_json_object(tmp_path / "missing.json") == {}
    assert mod.read_json_object(malformed) == {}
    assert mod.read_json_object(list_payload) == {}
    assert mod.sha256_file(tmp_path / "missing.json") is None
    assert mod._as_float(None) is None
    assert mod._as_float(True) is None
    assert mod._as_float("bad") is None
    assert mod._as_float("1.25") == pytest.approx(1.25)
    assert mod._mapping_list([{"a": 1}, "bad"]) == [{"a": 1}]
    assert mod._count({"a": 1}) == 1
    assert mod._repair_claim_status([]) == "blocked"
    assert mod._matrix_repair_row(
        {"rows": [{"row_id": "exp3016_repair_acceptance_controller", "status": "clean"}]}
    ) == {"row_id": "exp3016_repair_acceptance_controller", "status": "clean"}
    assert mod._matrix_capstone_boundary_blockers(
        {"claim_boundary_violations": [{"violation": "x"}]},
        {"capstone_ready": False, "paper_ready_blockers": ["repair blocker"]},
    ) == [
        "matrix_v17_ready is not true",
        "matrix repair row is missing",
        "capstone_ready is not true",
        "capstone repair decision is not promotable",
        "capstone still lists repair paper-ready blocker",
        "matrix claim boundary violations are present",
    ]
    assert mod._required_source_errors({"exp3027": {}, "exp3028": {}})[:2] == [
        {
            "experiment_id": "exp3027",
            "path": mod.EXP3027_REL_PATH.as_posix(),
            "reason": "missing_or_malformed_artifact",
        },
        {
            "experiment_id": "exp3028",
            "path": mod.EXP3028_REL_PATH.as_posix(),
            "reason": "missing_or_malformed_artifact",
        },
    ]
