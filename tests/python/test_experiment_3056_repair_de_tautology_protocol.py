"""Tests for Exp 3056 repair de-tautology protocol.

Spec refs: REQ-REPORT-3056, SCENARIO-REPORT-3056.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from carnot.reporting import repair_de_tautology_protocol_3056 as mod


REQUIRED_ARTIFACT_FIELDS = {
    "repair_de_tautology_protocol_ready",
    "blocked_prior_fields",
    "required_live_run_fields",
    "intent_preservation_checks",
    "duration_sanity_rule",
    "fingerprint_requirements",
    "promotion_disqualifiers",
    "inference_substrate",
    "honest_verdict",
}

EXP3059_REQUIRED_FIELDS = {
    "schema",
    "artifact",
    "run_date",
    "started_at",
    "finished_at",
    "duration_s",
    "wall_clock_start_utc",
    "wall_clock_end_utc",
    "random_seed",
    "seed_log",
    "model_specs",
    "models_used",
    "decode_config",
    "inference_substrate",
    "n_tasks",
    "candidate_count",
    "baseline_metrics",
    "repair_metrics",
    "pass_at_1_delta",
    "pass_at_k_delta",
    "pass_at_1_derivation",
    "pass_at_k_derivation",
    "per_task_results",
    "non_vacuous_outcome_summary",
    "accepted_candidate_count",
    "rejected_candidate_count",
    "false_accept_delta",
    "syntax_failure_rate_delta",
    "schema_failure_rate_delta",
    "intent_drift_count",
    "intent_preservation_checks",
    "checker_authority",
    "transcript_fingerprints",
    "raw_transcript_paths",
    "fingerprint_linkage",
    "reproducibility_checksum",
    "duration_sanity_check",
    "blocked_prior_fields_checked",
    "promotion_disqualifiers",
    "adversarial_verify_flags",
    "tests_run",
    "honest_verdict",
}

PROMOTION_DISQUALIFIER_IDS = {
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


def _corrigendum_flags() -> list[dict[str, str]]:
    return [
        {
            "kind": "TAUTOLOGY",
            "severity": "critical",
            "detail": (
                "pass_at_1_delta=0.375 and pass_at_k_delta=0.375 agree to >5 sig figs. "
                "Two distinct metrics matching this precisely is more likely a bug than a finding."
            ),
        },
        {
            "kind": "IMPLAUSIBLE_PERFECT",
            "severity": "info",
            "detail": "false_accept_delta=0.0 (exactly zero). Confirm this is not a stub default.",
        },
        {
            "kind": "IMPLAUSIBLE_PERFECT",
            "severity": "info",
            "detail": (
                "schema_failure_rate_delta=0.0 (exactly zero). Confirm this is not a stub default."
            ),
        },
        {
            "kind": "IMPLAUSIBLE_PERFECT",
            "severity": "info",
            "detail": (
                "syntax_failure_rate_delta=0.0 (exactly zero). Confirm this is not a stub default."
            ),
        },
        {
            "kind": "DURATION_TOO_SHORT",
            "severity": "critical",
            "detail": (
                "duration_s=0.103766 but artifact references compute-bound markers "
                "(GGUF / CUDA / live model). Loading and running a real model takes "
                ">=60.0s minimum; this completed too fast to have invoked the model."
            ),
        },
        {
            "kind": "METHODOLOGY_MISSING",
            "severity": "warn",
            "detail": "Compute-bound artifact missing: random_seed. Methodology unverifiable.",
        },
    ]


def _exp3028() -> dict[str, Any]:
    return {
        "artifact": "experiment_3028_sota_repair_clean_methodology_rerun_v2",
        "run_date": "20260525",
        "duration_s": 0.103766,
        "random_seed": None,
        "n_tasks": 24,
        "candidate_count": 24,
        "accepted_candidate_count": 9,
        "rejected_candidate_count": 15,
        "baseline_metrics": {"pass_at_1": 0.0, "pass_at_k": 0.0, "candidate_count": 24},
        "repair_metrics": {"pass_at_1": 0.375, "pass_at_k": 0.375, "candidate_count": 24},
        "pass_at_1_delta": 0.375,
        "pass_at_k_delta": 0.375,
        "false_accept_delta": 0.0,
        "schema_failure_rate_delta": 0.0,
        "syntax_failure_rate_delta": 0.0,
        "intent_drift_count": 0,
        "candidate_intent_drift_count": 14,
        "tautology_gate_clean": True,
        "corrigendum_pending": _corrigendum_flags(),
        "model_specs": [
            {
                "hf_id": "unsloth/gemma-4-26B-A4B-it-GGUF",
                "model_path": "/models/gemma-4-26B-A4B-it-UD-Q4_K_M.gguf",
                "checksum": "3098fa8e9a753392f81c5d9a409a24704c743208874064816856bc8aca6080e0",
            }
        ],
        "reproducibility_checksum": (
            "f7c34d69b95aba0b8553c04739345553f046eb952b966831a7f2a31df02276aa"
        ),
        "inference_substrate": {
            "cuda_available": True,
            "kind": "clean_repair_reconstruction",
            "live_repair_generation_run": False,
            "model_load_attempted": False,
            "recorded_before_model_load": True,
        },
        "honest_verdict": "complete: clean_repair_rerun_ready=true; n_tasks=24",
    }


def _blocker(
    row_id: str,
    classification: str,
    source_field: str,
    evidence: Any,
) -> dict[str, Any]:
    return {
        "row_id": row_id,
        "classification": classification,
        "blocking": True,
        "source_artifact": mod.EXP3028_REL_PATH.as_posix(),
        "source_field": source_field,
        "rationale": "fixture repair blocker",
        "evidence": evidence,
    }


def _exp3042() -> dict[str, Any]:
    return {
        "artifact": "experiment_3042_repair_promotion_reconciliation_v3",
        "repair_reconciliation_ready": True,
        "repair_claim_status": "bounded",
        "repair_promotion_candidate": False,
        "remaining_blockers": [
            _blocker(
                "exp3028:adversarial_flags",
                "true_blocker",
                "corrigendum_pending",
                _corrigendum_flags()[:5],
            ),
            _blocker(
                "exp3028:methodology_missing",
                "missing_metadata",
                "corrigendum_pending[METHODOLOGY_MISSING]",
                [_corrigendum_flags()[5]],
            ),
            _blocker(
                "capstone:repair_bounded",
                "unresolved_bound",
                "paper_ready_checks[repair_promotable]",
                {"check": "repair_promotable", "passed": False, "reason": "bounded"},
            ),
        ],
        "exp3028_evidence_checks": {
            "positive_pass_at_1_delta": True,
            "nonnegative_pass_at_k_delta": True,
            "tautology_gate_clean": True,
            "reproducibility_checksum_present": True,
        },
        "inference_substrate": {
            "kind": "aggregation_from_upstream_artifacts",
            "source": "checked_in_artifacts",
            "executes_models": False,
            "executes_hardware": False,
            "executes_conductor": False,
            "no_live_llm_inference": True,
        },
        "honest_verdict": "complete: repair_claim_status=bounded",
    }


def _exp3043() -> dict[str, Any]:
    return {
        "artifact": "experiment_3043_verified_speculation_transcript_fingerprint_v1",
        "fingerprint_live_ready": True,
        "deterministic_replay_passed": True,
        "random_seed": 304300,
        "model_specs": [
            {
                "hf_id": "unsloth/gemma-4-26B-A4B-it-GGUF",
                "model_hash_or_cache_path": "bounded_sha256:abc",
            }
        ],
        "raw_transcript_paths": ["results/raw/experiment_3043/prompt_00_run_01.json"],
        "transcript_fingerprints": [
            {
                "prompt_hash": "f607e3b37c57fca1a905cce59462f6f681d7599f2a8de5c161705f844cd64ee3",
                "raw_output_hash": (
                    "e94438c4dcfb854b71c3ec4681289ba0dc5956f794d0c39728e9aa264e622542"
                ),
                "normalized_output_hash": (
                    "088227a88edc8c3bc399e2cf4ca0c54ef6a1ae01b53340b58f1a1f1764636331"
                ),
                "model_hash_or_cache_path": "bounded_sha256:abc",
                "seed": 304300,
                "run_index": 1,
                "tokens_observed": 10,
            }
        ],
        "reproducibility_checksum": (
            "eee23eb2aa1d4d1d043b795ef8c4a04c3249423e715f06b4180008116b8bf00d"
        ),
        "honest_verdict": "complete: fingerprint_live_ready=true",
    }


def _exp3055() -> dict[str, Any]:
    return {
        "artifact": "experiment_3055_repair_headline_retirement_and_blocker_ledger_v1",
        "repair_headline_retirement_ready": True,
        "repair_claim_status": "bounded",
        "extracted_repair_blockers": [
            {
                "row_id": "exp3028:adversarial_flags",
                "status": "true_blocker",
                "classification": "true_blocker",
                "source_artifact": mod.EXP3028_REL_PATH.as_posix(),
                "source_field": "corrigendum_pending",
                "matrix_v20_consumable": True,
            }
        ],
        "rerun_prerequisites": [
            {"gate": "deterministic_fingerprint", "required": True},
            {"gate": "seed", "required": True},
            {"gate": "duration_sanity", "required": True},
            {"gate": "de_tautology_metrics", "required": True},
            {"gate": "verifier_gain", "required": True},
            {"gate": "exact_checker_authority", "required": True},
        ],
        "inference_substrate": {
            "kind": "aggregation_from_upstream_artifacts",
            "source": "checked_in_artifacts",
            "executes_models": False,
            "executes_hardware": False,
            "executes_conductor": False,
            "no_live_llm_inference": True,
        },
        "honest_verdict": "complete: repair_headline_retirement_ready=true",
    }


def _research_refs(include_aprad: bool = True) -> str:
    if not include_aprad:
        return "No aligned-decoding repair reference in this fixture.\n"
    return (
        "- Approximately Aligned Decoding / AprAD balances constraint satisfaction, "
        "compute, and output-distribution distortion. Carnot use: preserve draft "
        "intent while applying hard verifier gates; no implementation claim.\n"
    )


def _write_sources(root: Path, *, include_aprad: bool = True) -> None:
    _write_json(root, mod.EXP3028_REL_PATH, _exp3028())
    _write_json(root, mod.EXP3042_REL_PATH, _exp3042())
    _write_json(root, mod.EXP3043_REL_PATH, _exp3043())
    _write_json(root, mod.EXP3055_REL_PATH, _exp3055())
    _write_text(root, mod.RESEARCH_REFERENCES_REL_PATH, _research_refs(include_aprad))


def test_req_report_3056_spec_anchor_and_script_exist() -> None:
    """REQ-REPORT-3056: OpenSpec declares the protocol contract first."""

    spec = (mod.REPO_ROOT / "openspec/capabilities/research-reporting/spec.md").read_text(
        encoding="utf-8"
    )

    assert "REQ-REPORT-3056" in spec
    assert "SCENARIO-REPORT-3056" in spec
    assert mod.OUTPUT_REL_PATH.as_posix() in spec
    assert mod.SCRIPT_REL_PATH.exists()


def test_scenario_report_3056_builds_ready_de_tautology_protocol(tmp_path: Path) -> None:
    """SCENARIO-REPORT-3056: prior blockers map to future live-run checks."""

    _write_sources(tmp_path)

    artifact = mod.build_artifact(tmp_path, started_s=1.0, now_s=3.25)
    blocked = artifact["blocked_prior_fields"]
    live_fields = set(artifact["required_live_run_fields"])
    disqualifier_ids = {row["id"] for row in artifact["promotion_disqualifiers"]}
    fingerprint_fields = {row["field"] for row in artifact["fingerprint_requirements"]}

    assert REQUIRED_ARTIFACT_FIELDS <= artifact.keys()
    assert artifact["repair_de_tautology_protocol_ready"] is True
    assert artifact["duration_s"] == pytest.approx(2.25)
    assert artifact["honest_verdict"].startswith("complete:")
    assert artifact["exp3059_matrix_v20_required_fields"] == artifact["required_live_run_fields"]
    assert EXP3059_REQUIRED_FIELDS <= live_fields

    assert set(blocked) == {
        "tautology",
        "implausible_perfect_results",
        "duration",
        "missing_seed",
        "unresolved_methodology",
    }
    assert blocked["tautology"]["source_artifact"] == mod.EXP3028_REL_PATH.as_posix()
    assert blocked["tautology"]["source_field"] == "corrigendum_pending[0]"
    assert blocked["tautology"]["observed_fields"] == ["pass_at_1_delta", "pass_at_k_delta"]
    assert blocked["implausible_perfect_results"]["observed_fields"] == [
        "false_accept_delta",
        "schema_failure_rate_delta",
        "syntax_failure_rate_delta",
    ]
    assert blocked["duration"]["observed_fields"] == ["duration_s"]
    assert blocked["missing_seed"]["observed_fields"] == ["random_seed"]
    assert blocked["unresolved_methodology"]["source_artifact"] == mod.EXP3042_REL_PATH.as_posix()
    assert {
        "exp3028:methodology_missing",
        "capstone:repair_bounded",
    } <= {row["row_id"] for row in blocked["unresolved_methodology"]["rows"]}

    assert artifact["duration_sanity_rule"] == {
        "applies_to": "live local SOTA repair runs",
        "minimum_live_compute_duration_s": 60.0,
        "applies_when_any_marker_present": ["GGUF", "CUDA", "live model", "model_specs"],
        "requires_monotonic_wall_clock": True,
        "failure_action": "disqualify_matrix_v20_promotion",
        "source_blocker": "exp3028.corrigendum_pending[DURATION_TOO_SHORT]",
    }
    assert PROMOTION_DISQUALIFIER_IDS <= disqualifier_ids
    assert {
        "transcript_fingerprints[].prompt_hash",
        "transcript_fingerprints[].raw_output_hash",
        "transcript_fingerprints[].normalized_output_hash",
        "transcript_fingerprints[].model_hash_or_cache_path",
        "transcript_fingerprints[].seed",
        "raw_transcript_paths[]",
        "reproducibility_checksum",
    } <= fingerprint_fields

    aprad_checks = [
        row
        for row in artifact["intent_preservation_checks"]
        if row["id"] == "aprad_inspired_distribution_distortion_guard"
    ]
    assert aprad_checks
    assert aprad_checks[0]["claims_aprad_implementation"] is False
    assert "AprAD-inspired" in aprad_checks[0]["requirement"]
    assert all(row["required"] is True for row in artifact["intent_preservation_checks"])

    assert artifact["inference_substrate"] == {
        "kind": "aggregation_from_upstream_artifacts",
        "source": "checked_in_artifacts",
        "executes_models": False,
        "executes_hardware": False,
        "executes_conductor": False,
        "no_live_llm_inference": True,
    }
    source_by_path = {row["path"]: row for row in artifact["source_artifacts"]}
    assert source_by_path[mod.EXP3028_REL_PATH.as_posix()]["sha256"] == _sha256(
        tmp_path / mod.EXP3028_REL_PATH
    )


def test_req_report_3056_write_artifact_emits_deliverable(tmp_path: Path) -> None:
    """REQ-REPORT-3056: write_artifact writes the stable JSON deliverable."""

    _write_sources(tmp_path)

    output = mod.write_artifact(tmp_path, started_s=5.0, now_s=6.5)
    payload = json.loads(output.read_text(encoding="utf-8"))

    assert output == tmp_path / mod.OUTPUT_REL_PATH
    assert payload["repair_de_tautology_protocol_ready"] is True
    assert payload["duration_s"] == pytest.approx(1.5)


def test_req_report_3056_blocks_when_required_sources_are_absent(tmp_path: Path) -> None:
    """REQ-REPORT-3056: Exp 3059 cannot consume a protocol with missing sources."""

    _write_sources(tmp_path)
    (tmp_path / mod.EXP3042_REL_PATH).unlink()

    artifact = mod.build_artifact(tmp_path)

    assert artifact["repair_de_tautology_protocol_ready"] is False
    assert "required source artifacts missing or malformed" in artifact["blocked_reasons"]
    assert artifact["honest_verdict"].startswith("blocked_precondition:")


def test_req_report_3056_blocks_when_prior_blocker_category_is_missing(tmp_path: Path) -> None:
    """REQ-REPORT-3056: all prior blocker categories must map to explicit checks."""

    _write_sources(tmp_path)
    exp3028 = _exp3028()
    exp3028["corrigendum_pending"] = [
        row for row in exp3028["corrigendum_pending"] if row["kind"] != "TAUTOLOGY"
    ]
    _write_json(tmp_path, mod.EXP3028_REL_PATH, exp3028)

    artifact = mod.build_artifact(tmp_path)

    assert artifact["repair_de_tautology_protocol_ready"] is False
    assert "missing blocked_prior_fields categories: tautology" in artifact["blocked_reasons"]
    assert artifact["honest_verdict"].startswith("blocked_precondition:")


def test_req_report_3056_blocks_when_aprad_reference_is_absent(tmp_path: Path) -> None:
    """REQ-REPORT-3056: AprAD-inspired intent language must come from references."""

    _write_sources(tmp_path, include_aprad=False)

    artifact = mod.build_artifact(tmp_path)

    assert artifact["repair_de_tautology_protocol_ready"] is False
    assert "AprAD reference unavailable" in artifact["blocked_reasons"]


def test_req_report_3056_helper_edges_fail_closed(tmp_path: Path) -> None:
    """REQ-REPORT-3056: malformed sources and missing data do not fabricate readiness."""

    bad_json = tmp_path / "bad.json"
    array_json = tmp_path / "array.json"
    missing = tmp_path / "missing.json"
    bad_json.write_text("{", encoding="utf-8")
    array_json.write_text("[]", encoding="utf-8")

    assert mod.read_json_object(missing) == {}
    assert mod.read_json_object(bad_json) == {}
    assert mod.read_json_object(array_json) == {}
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

    assert mod._rows_by_kind([], "TAUTOLOGY") == []
    assert mod._rows_by_kind([{"kind": "TAUTOLOGY"}], "METHODOLOGY_MISSING") == []
    assert mod._read_text(missing) == ""
    assert mod._range_source_field("corrigendum_pending", []) == "corrigendum_pending"
    assert mod._range_source_field(
        "corrigendum_pending",
        [(1, {"kind": "IMPLAUSIBLE_PERFECT"}), (3, {"kind": "IMPLAUSIBLE_PERFECT"})],
    ) == "corrigendum_pending[1],corrigendum_pending[3]"
    assert mod._observed_fields_from_detail("no matching fields here") == []
    assert mod._unique(["random_seed", "random_seed", "duration_s"]) == [
        "random_seed",
        "duration_s",
    ]
    assert mod._as_list({"not": "a-list"}) == []
    assert mod._as_mapping(["not", "a", "mapping"]) == {}
    assert mod._duration(3.0, 2.0) == 0.0
    assert (
        mod._honest_verdict(
            {"repair_de_tautology_protocol_ready": False, "blocked_reasons": ["x", "y"]}
        )
        == "blocked_precondition: repair de-tautology protocol incomplete; reasons=2"
    )
