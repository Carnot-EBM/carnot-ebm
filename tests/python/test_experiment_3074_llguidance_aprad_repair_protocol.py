"""Tests for Exp 3074 LLGuidance/AprAD repair protocol.

Spec refs: REQ-REPORT-3074, SCENARIO-REPORT-3074.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from carnot.reporting import llguidance_aprad_repair_protocol_3074 as mod


REPO_ROOT = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO_ROOT / "openspec" / "capabilities" / "research-reporting" / "spec.md"
SCRIPT_PATH = REPO_ROOT / "scripts" / mod.SCRIPT_FILENAME

REQUIRED_ARTIFACT_FIELDS = {
    "grammar_constrained_repair_protocol_ready",
    "schema_syntax_failure_targets",
    "exact_semantic_validation_required",
    "aprad_intent_preservation_rules",
    "llguidance_runtime_plan",
    "de_tautology_disqualifiers",
    "exp3075_required_fields",
    "inference_substrate",
    "honest_verdict",
}

EXP3056_DISQUALIFIER_IDS = {
    "prior_tautology_not_cleared",
    "pass_at_1_pass_at_k_delta_bit_identical_without_k1_declaration",
    "implausible_perfect_delta_without_per_case_evidence",
    "duration_too_short_for_live_compute",
    "missing_random_seed_or_seed_log",
    "missing_model_specs_identity",
    "missing_transcript_fingerprints",
    "checker_authority_missing_or_self_graded",
    "intent_drift_detected",
    "false_accept_or_syntax_schema_regression",
    "legacy_smoke_or_non_headline_model_used",
    "unresolved_methodology_blocker_present",
}

EXP3075_CONSUMER_FIELDS = {
    "grammar_constrained_repair_protocol_ready",
    "schema_syntax_failure_targets",
    "exact_semantic_validation_required",
    "aprad_intent_preservation_rules",
    "llguidance_runtime_plan",
    "de_tautology_disqualifiers",
    "repair_generation_blocked",
    "clean_blocked_outcome",
    "verifier_gain_gate_passed",
    "gate_check_summary",
    "inference_substrate",
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


def _disqualifiers() -> list[dict[str, str]]:
    return [
        {"id": disqualifier_id, "required_clearance": f"clear {disqualifier_id}"}
        for disqualifier_id in sorted(EXP3056_DISQUALIFIER_IDS)
    ]


def _write_sources(root: Path, *, omit: set[Path] | None = None) -> None:
    omit = omit or set()
    payloads: dict[Path, dict[str, Any]] = {
        mod.EXP3056_REL_PATH: {
            "artifact": "experiment_3056_repair_de_tautology_protocol_v1",
            "repair_de_tautology_protocol_ready": True,
            "promotion_disqualifiers": _disqualifiers(),
            "required_live_run_fields": [
                "schema",
                "random_seed",
                "pass_at_1_derivation",
                "pass_at_k_derivation",
                "checker_authority",
                "honest_verdict",
            ],
            "blocked_prior_fields": {
                "tautology": {"kind": "TAUTOLOGY", "observed_fields": ["pass_at_1_delta"]},
                "duration": {"kind": "DURATION_TOO_SHORT", "observed_fields": ["duration_s"]},
            },
            "inference_substrate": {"no_live_llm_inference": True, "executes_models": False},
            "honest_verdict": "complete: repair_de_tautology_protocol_ready=true",
        },
        mod.EXP3059_REL_PATH: {
            "experiment": 3059,
            "status": "blocked",
            "honest_verdict": "blocked_gate_check_failed",
            "blocked_at_layer": "conductor_pre_gate",
            "gate_check_summary": (
                "1 of 2 gate(s) failed; first failure: "
                "exp3057-local-sota-solution-verifier-gain-panel.verifier_gain_delta "
                "(actual=-0.125 > expected=0.0)"
            ),
            "gates_evaluated": [
                {
                    "upstream": "exp3056-repair-de-tautology-protocol-v1",
                    "artifact_field": "repair_de_tautology_protocol_ready",
                    "expected": True,
                    "actual": True,
                    "passed": True,
                },
                {
                    "upstream": "exp3057-local-sota-solution-verifier-gain-panel",
                    "artifact_field": "verifier_gain_delta",
                    "op": ">",
                    "expected": 0.0,
                    "actual": -0.125,
                    "passed": False,
                },
            ],
        },
        mod.CAPSTONE_V286_REL_PATH: {
            "artifact": "experiment_3066_capstone_v286",
            "capstone_ready": True,
            "repair_claim_status": "bounded_and_gated_skipped",
            "source_artifacts": [
                {"experiment_id": "exp3056", "path": mod.EXP3056_REL_PATH.as_posix()}
            ],
            "honest_verdict": "complete: capstone_ready=true",
        },
    }
    for rel_path, payload in payloads.items():
        if rel_path not in omit:
            _write_json(root, rel_path, payload)
    if mod.RESEARCH_REFERENCES_REL_PATH not in omit:
        _write_text(
            root,
            mod.RESEARCH_REFERENCES_REL_PATH,
            (
                "LLGuidance supports JSON schema and grammar-constrained masks. "
                "AprAD-style intent preservation separates syntax from semantics."
            ),
        )


def test_req_report_3074_spec_and_script_anchor_exists() -> None:
    """REQ-REPORT-3074: the protocol is OpenSpec anchored."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-REPORT-3074" in spec
    assert "SCENARIO-REPORT-3074" in spec
    assert mod.ARTIFACT_FILENAME in spec
    assert "grammar_constrained_repair_protocol_ready" in spec
    assert "exact_semantic_validation_required" in spec
    assert SCRIPT_PATH.exists()


def test_scenario_report_3074_builds_consumable_protocol(tmp_path: Path) -> None:
    """SCENARIO-REPORT-3074: grammar and intent gates precede live repair."""

    _write_sources(tmp_path)
    artifact = mod.build_artifact(tmp_path, started_s=10.0, now_s=12.25)

    assert REQUIRED_ARTIFACT_FIELDS <= set(artifact)
    assert artifact["grammar_constrained_repair_protocol_ready"] is True
    assert artifact["exact_semantic_validation_required"] is True
    assert artifact["duration_s"] == pytest.approx(2.25)
    assert artifact["honest_verdict"].startswith("complete:")

    disqualifier_ids = {row["id"] for row in artifact["de_tautology_disqualifiers"]}
    assert disqualifier_ids == EXP3056_DISQUALIFIER_IDS
    assert artifact["clean_blocked_outcome"]["triggered_by_exp3059"] is True
    assert artifact["clean_blocked_outcome"]["verifier_gain_gate"]["passed"] is False
    assert "actual=-0.125" in artifact["clean_blocked_outcome"]["gate_check_summary"]

    failure_targets = artifact["schema_syntax_failure_targets"]
    assert failure_targets["syntax_errors"]["measured"] is True
    assert failure_targets["schema_errors"]["artifact_field"] == "schema_failure_rate_delta"
    assert failure_targets["parse_failures"]["fallback_behavior"] == "deterministic_schema_validation"

    runtime_plan = artifact["llguidance_runtime_plan"]
    assert runtime_plan["grammar_source"] == "exp3074_json_schema_to_llguidance_or_gbnf"
    assert runtime_plan["constrained_syntax_target"] == "single_repair_candidate_json"
    assert runtime_plan["live_generation_allowed_by_this_protocol"] is False
    assert runtime_plan["fallback_behavior"]["on_backend_unavailable"] == (
        "emit unconstrained draft to deterministic JSON-schema validator only; do not promote"
    )

    rule_ids = {row["id"] for row in artifact["aprad_intent_preservation_rules"]}
    assert {
        "task_intent_hash_required",
        "behavioral_tests_required",
        "semantic_drift_checks_required",
        "independent_verifier_authority_required",
    } <= rule_ids
    assert EXP3075_CONSUMER_FIELDS <= set(artifact["exp3075_required_fields"])
    assert artifact["exp3075_consumer_contract"]["matrix_version"] == "v21"
    assert artifact["inference_substrate"]["live_llm_inference"] is False
    assert artifact["inference_substrate"]["protocol_only"] is True

    mod.validate_artifact(artifact)


def test_req_report_3074_blocks_cleanly_and_validates_edges(tmp_path: Path) -> None:
    """REQ-REPORT-3074: missing sources block readiness without live inference."""

    bad_json = tmp_path / "results" / "bad.json"
    bad_json.parent.mkdir(parents=True, exist_ok=True)
    bad_json.write_text("{bad", encoding="utf-8")
    assert mod.read_json_object(bad_json) == {}
    assert mod.read_text(tmp_path / "missing.md") == ""
    assert mod._de_tautology_disqualifiers({}) == []
    assert mod._duration(0.0, None) >= 0.0

    _write_sources(tmp_path, omit={mod.EXP3059_REL_PATH})
    artifact = mod.build_artifact(tmp_path, started_s=1.0, now_s=2.0)

    assert artifact["grammar_constrained_repair_protocol_ready"] is False
    assert artifact["honest_verdict"].startswith("blocked_missing_source:")
    assert artifact["clean_blocked_outcome"]["outcome"] == "blocked_missing_source"
    assert any(
        row["path"] == mod.EXP3059_REL_PATH.as_posix() and row["present"] is False
        for row in artifact["source_artifacts"]
    )
    mod.validate_artifact(artifact)

    _write_sources(tmp_path)
    ready = mod.build_artifact(tmp_path, started_s=1.0, now_s=2.0)
    written = mod.write_artifact(
        tmp_path,
        output_path=Path("results") / "exp3074-copy.json",
        started_s=1.0,
        now_s=2.0,
    )
    assert written == tmp_path / "results" / "exp3074-copy.json"
    assert (
        json.loads(written.read_text(encoding="utf-8"))[
            "grammar_constrained_repair_protocol_ready"
        ]
        is True
    )

    with pytest.raises(ValueError, match="missing required fields"):
        mod.validate_artifact({"honest_verdict": "complete: incomplete"})
    with pytest.raises(ValueError, match="live_llm_inference"):
        mod.validate_artifact(
            ready
            | {"inference_substrate": ready["inference_substrate"] | {"live_llm_inference": True}}
        )
    with pytest.raises(ValueError, match="de_tautology_disqualifiers"):
        mod.validate_artifact(ready | {"de_tautology_disqualifiers": []})
    with pytest.raises(ValueError, match="exact_semantic_validation_required"):
        mod.validate_artifact(ready | {"exact_semantic_validation_required": False})
    with pytest.raises(ValueError, match="required_fields"):
        mod.validate_artifact(ready | {"exp3075_required_fields": []})
    with pytest.raises(ValueError, match="honest_verdict"):
        mod.validate_artifact(ready | {"honest_verdict": "ready"})
    with pytest.raises(ValueError, match="blocked_missing_source"):
        mod.validate_artifact(artifact | {"honest_verdict": "complete: wrong"})
