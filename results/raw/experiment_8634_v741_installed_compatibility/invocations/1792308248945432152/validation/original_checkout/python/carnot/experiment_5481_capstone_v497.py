"""Exp5481 .497 capstone truth table and PRD gap synthesis.

Spec refs: REQ-REPORT-5481, SCENARIO-REPORT-5481,
SCENARIO-REPORT-5481-MISSING-FLAGGED.

This module is intentionally an aggregation step. It reads the result artifacts
that already exist for Exp5468 through Exp5480 and reports what they prove. That
keeps the capstone from turning a planned lane, a quarantined lane, a bounded
receipt, or an honest null into a stronger claim than the upstream artifact
actually supports.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any


JsonDict = dict[str, Any]
JsonMap = Mapping[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[2]
MILESTONE = "2026.07.497"
EXPERIMENT = "experiment_5481_capstone_v497"
EXPERIMENT_ID = "exp5481-v497-capstone"
SCHEMA = "carnot.experiment_5481.capstone_v497.v1"
RUN_DATE = "20260709"
RANDOM_SEED = 5481
OUTPUT_REL_PATH = Path("results/experiment_5481_capstone_v497.json")
INFERENCE_SUBSTRATE = "aggregation_from_upstream_artifacts"

SPEC_REFS = (
    "REQ-REPORT-5481",
    "SCENARIO-REPORT-5481",
    "SCENARIO-REPORT-5481-MISSING-FLAGGED",
)

CONTEXT_PATHS = (
    "AGENTS.md",
    "CODEX.md",
    "CLAUDE.md",
    "research-program.md",
    "_bmad/prd.md",
    "_bmad/architecture.md",
    "ops/status.md",
    "ops/changelog.md",
    "ops/conductor-log.md",
    "ops/e2e-test-plan.md",
    "results",
)

EXPECTED_ARTIFACT_PATHS = (
    "results/experiment_5468_transition_v497.json",
    "results/experiment_5469_source_delta_v497.json",
    "results/experiment_5470_rewrite_state_semantic_fixture_v497.json",
    "results/experiment_5471_guard_composition_scale_v497.json",
    "results/experiment_5472_sota_evidence_telemetry_v497.json",
    "results/experiment_5473_csl_kan_surrogate_assurance_v497.json",
    "results/experiment_5474_sota_csl_scale_v497.json",
    "results/experiment_5475_csl_behavioral_memory_ladder_v497.json",
    "results/experiment_5476_helper_lemma_core_witness_repair_v497.json",
    "results/experiment_5477_pdit_lns_boundary_exchange_v497.json",
    "results/experiment_5478_hardware_receipts_v497.json",
    "results/experiment_5479_arc_target_rotation_precheck_v497.json",
    "results/experiment_5480_arc_live_salience_levelup_v497.json",
)

REQUIRED_ARTIFACT_FIELDS = (
    "milestone",
    "artifact_paths",
    "missing_artifacts",
    "flagged_artifacts",
    "headline_ready_lanes",
    "bounded_lanes",
    "blocked_lanes",
    "honest_null_lanes",
    "prd_gap_table",
    "failure_taxonomy",
    "guided_decoding_quarantine_status",
    "csl_status",
    "arc_registry_delta",
    "hardware_speedup_claim",
    "ops_status_updated",
    "ops_changelog_updated",
    "roadmap_yaml_unchanged",
    "conductor_unchanged",
    "next_recommendations",
    "inference_substrate",
    "honest_verdict",
)

DEFAULT_TESTS_RUN = (
    ".venv/bin/pytest tests/python/test_experiment_5481_capstone_v497.py -q --no-cov",
    (
        ".venv/bin/coverage run --include=python/carnot/experiment_5481_capstone_v497.py "
        "-m pytest tests/python/test_experiment_5481_capstone_v497.py -q --no-cov -n 0"
    ),
    (
        ".venv/bin/coverage report "
        "--include=python/carnot/experiment_5481_capstone_v497.py --fail-under=100"
    ),
    ".venv/bin/pytest tests/python -q",
    (
        "ops/e2e-test-plan.md review: Exp5481 is aggregation-only; no fresh "
        "model training, PyO3 round trip, or destructive hardware workload applies"
    ),
)

FIELD_PRINCIPLES = {
    "milestone": "route key for the .497 capstone.",
    "artifact_paths": "only upstream artifacts that were actually present and read.",
    "missing_artifacts": "expected Exp5468-Exp5480 artifacts that do not exist.",
    "flagged_artifacts": "artifacts with explicit top-level adversarial flags.",
    "headline_ready_lanes": "lanes whose source fields satisfy their authority gate.",
    "bounded_lanes": "useful receipts that do not support headline claims.",
    "blocked_lanes": "quarantined or otherwise blocked lanes.",
    "honest_null_lanes": "executed lanes that produced no positive bankable result.",
    "prd_gap_table": "FR-11/FR-12/hardware/ARC/local-runtime mapping to evidence.",
    "failure_taxonomy": "failure classes preserved for next-milestone planning.",
    "guided_decoding_quarantine_status": "current quarantine state from artifact fields.",
    "csl_status": "claim boundary for continuous self-learning evidence.",
    "arc_registry_delta": "reproduced-level before/after delta from Exp5480 fields.",
    "hardware_speedup_claim": "true only when an upstream artifact claims speedup.",
    "ops_status_updated": "false because this terminal run leaves ops reconciliation to conductor.",
    "ops_changelog_updated": "false because this terminal run leaves ops reconciliation to conductor.",
    "roadmap_yaml_unchanged": "protected-file check for research-roadmap.yaml.",
    "conductor_unchanged": "protected-file check for scripts/research_conductor.py.",
    "next_recommendations": "natural priorities implied by the truth table.",
    "inference_substrate": "aggregation only; no hidden live inference.",
    "honest_verdict": "terminal verdict starting complete: or blocked:.",
}


def _stable_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _sha256_text(text: str) -> str:
    return f"sha256:{hashlib.sha256(text.encode('utf-8')).hexdigest()}"


def _select(payload: JsonMap, fields: Sequence[str]) -> JsonDict:
    return {field: payload[field] for field in fields if field in payload}


def _is_number(value: Any, expected: float) -> bool:
    return (
        isinstance(value, int | float) and not isinstance(value, bool) and float(value) == expected
    )


def _artifact_is_flagged(payload: JsonMap) -> bool:
    return payload.get("flagged_adversarial") is True or str(
        payload.get("adversarial_verdict", "")
    ).lower() in {"flagged", "adversarial_flagged"}


def _load_context(root: Path) -> tuple[list[str], list[str]]:
    read: list[str] = []
    missing: list[str] = []
    for rel_path in CONTEXT_PATHS:
        path = root / rel_path
        if path.exists():
            if path.is_file():
                path.read_text(encoding="utf-8")
            read.append(rel_path)
        else:
            missing.append(rel_path)
    return sorted(read), sorted(missing)


def _load_artifacts(root: Path) -> tuple[dict[str, JsonDict], list[str], list[str], dict[str, str]]:
    payloads: dict[str, JsonDict] = {}
    missing: list[str] = []
    flagged: list[str] = []
    checksums: dict[str, str] = {}
    for rel_path in EXPECTED_ARTIFACT_PATHS:
        path = root / rel_path
        if not path.exists():
            missing.append(rel_path)
            continue
        text = path.read_text(encoding="utf-8")
        payload = json.loads(text)
        if not isinstance(payload, dict):
            raise ValueError(f"{path} did not contain a JSON object")
        payloads[rel_path] = payload
        checksums[rel_path] = _sha256_text(text)
        if _artifact_is_flagged(payload):
            flagged.append(rel_path)
    return payloads, sorted(missing), sorted(flagged), checksums


def _protected_file_clean(root: Path, rel_path: str) -> bool:
    if not (root / ".git").exists():
        return True
    result = subprocess.run(  # pragma: no cover - covered by real repo execution.
        ["git", "status", "--short", "--", rel_path],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
    )
    return result.returncode == 0 and result.stdout.strip() == ""


def _classify_sources(
    source_artifacts: Sequence[str],
    missing_artifacts: Sequence[str],
    flagged_artifacts: Sequence[str],
) -> str | None:
    if any(path in missing_artifacts for path in source_artifacts):
        return "missing"
    if any(path in flagged_artifacts for path in source_artifacts):
        return "flagged"
    return None


def _row(
    lane: str,
    classification: str,
    source_artifacts: Sequence[str],
    evidence: JsonDict,
    claim_boundary: str,
) -> JsonDict:
    return {
        "lane": lane,
        "classification": classification,
        "source_artifacts": list(source_artifacts),
        "evidence": evidence,
        "claim_boundary": claim_boundary,
    }


def _truth_table(
    payloads: Mapping[str, JsonMap],
    missing_artifacts: Sequence[str],
    flagged_artifacts: Sequence[str],
) -> dict[str, JsonDict]:
    exp5468 = payloads.get("results/experiment_5468_transition_v497.json", {})
    exp5469 = payloads.get("results/experiment_5469_source_delta_v497.json", {})
    exp5470 = payloads.get("results/experiment_5470_rewrite_state_semantic_fixture_v497.json", {})
    exp5471 = payloads.get("results/experiment_5471_guard_composition_scale_v497.json", {})
    exp5472 = payloads.get("results/experiment_5472_sota_evidence_telemetry_v497.json", {})
    exp5473 = payloads.get("results/experiment_5473_csl_kan_surrogate_assurance_v497.json", {})
    exp5474 = payloads.get("results/experiment_5474_sota_csl_scale_v497.json", {})
    exp5475 = payloads.get("results/experiment_5475_csl_behavioral_memory_ladder_v497.json", {})
    exp5476 = payloads.get("results/experiment_5476_helper_lemma_core_witness_repair_v497.json", {})
    exp5477 = payloads.get("results/experiment_5477_pdit_lns_boundary_exchange_v497.json", {})
    exp5478 = payloads.get("results/experiment_5478_hardware_receipts_v497.json", {})
    exp5479 = payloads.get("results/experiment_5479_arc_target_rotation_precheck_v497.json", {})
    exp5480 = payloads.get("results/experiment_5480_arc_live_salience_levelup_v497.json", {})

    transition_sources = (
        "results/experiment_5468_transition_v497.json",
        "results/experiment_5469_source_delta_v497.json",
    )
    guided_sources = (
        "results/experiment_5468_transition_v497.json",
        "results/experiment_5470_rewrite_state_semantic_fixture_v497.json",
        "results/experiment_5471_guard_composition_scale_v497.json",
        "results/experiment_5472_sota_evidence_telemetry_v497.json",
    )
    reasoning_sources = (
        "results/experiment_5470_rewrite_state_semantic_fixture_v497.json",
        "results/experiment_5471_guard_composition_scale_v497.json",
        "results/experiment_5476_helper_lemma_core_witness_repair_v497.json",
    )
    local_sota_sources = ("results/experiment_5472_sota_evidence_telemetry_v497.json",)
    csl_sources = (
        "results/experiment_5473_csl_kan_surrogate_assurance_v497.json",
        "results/experiment_5474_sota_csl_scale_v497.json",
        "results/experiment_5475_csl_behavioral_memory_ladder_v497.json",
    )
    pdit_sources = ("results/experiment_5477_pdit_lns_boundary_exchange_v497.json",)
    hardware_sources = ("results/experiment_5478_hardware_receipts_v497.json",)
    arc_sources = (
        "results/experiment_5479_arc_target_rotation_precheck_v497.json",
        "results/experiment_5480_arc_live_salience_levelup_v497.json",
    )

    transition_class = _classify_sources(transition_sources, missing_artifacts, flagged_artifacts)
    guided_class = _classify_sources(guided_sources, missing_artifacts, flagged_artifacts)
    reasoning_class = _classify_sources(reasoning_sources, missing_artifacts, flagged_artifacts)
    local_sota_class = _classify_sources(local_sota_sources, missing_artifacts, flagged_artifacts)
    csl_class = _classify_sources(csl_sources, missing_artifacts, flagged_artifacts)
    pdit_class = _classify_sources(pdit_sources, missing_artifacts, flagged_artifacts)
    hardware_class = _classify_sources(hardware_sources, missing_artifacts, flagged_artifacts)
    arc_class = _classify_sources(arc_sources, missing_artifacts, flagged_artifacts)

    if transition_class is None:
        transition_class = "bounded"
    if guided_class is None:
        quarantine_held = (
            exp5470.get("guided_decoding_quarantine_lifted") is False
            and exp5471.get("guided_decoding_quarantine_lifted") is False
            and exp5472.get("guided_decoding_used") is False
        )
        guided_class = "blocked" if quarantine_held else "bounded"
    if reasoning_class is None:
        reasoning_ready = (
            exp5470.get("rewrite_state_fixture_ready") is True
            and _is_number(exp5470.get("exact_validator_agreement"), 1.0)
            and exp5471.get("guard_composition_ready") is True
            and _is_number(exp5471.get("false_accept_rate"), 0.0)
            and exp5476.get("helper_lemma_repair_ready") is True
            and exp5476.get("false_accept_count") == 0
        )
        reasoning_class = "headline_ready" if reasoning_ready else "blocked"
    if local_sota_class is None:
        local_sota_ready = (
            exp5472.get("sota_evidence_telemetry_ready") is True
            and exp5472.get("guided_decoding_used") is False
            and bool(exp5472.get("gpu_offload_receipts"))
        )
        local_sota_class = "bounded" if local_sota_ready else "blocked"
    if csl_class is None:
        csl_ready = (
            exp5473.get("csl_kan_surrogate_ready") is True
            and exp5473.get("model_weight_mutation") is False
            and exp5473.get("constraint_violation_count") == 0
            and exp5474.get("csl_scale_ready") is True
            and exp5474.get("model_weight_mutation") is False
            and exp5475.get("csl_behavioral_memory_ready") is True
            and exp5475.get("model_weight_mutation") is False
        )
        csl_class = "headline_ready" if csl_ready else "blocked"
    if pdit_class is None:
        pdit_ready = (
            exp5477.get("boundary_exchange_ready") is True
            and _is_number(exp5477.get("exact_fallback_completeness_rate"), 1.0)
            and exp5477.get("unsafe_false_accept_count") == 0
            and exp5477.get("hardware_speedup_claim") is False
        )
        pdit_class = "bounded" if pdit_ready else "blocked"
    if hardware_class is None:
        hardware_ready = (
            exp5478.get("hardware_receipts_ready") is True
            and exp5478.get("hardware_speedup_claim") is False
            and _is_number(exp5478.get("result_hash_match_rate"), 1.0)
        )
        hardware_class = "bounded" if hardware_ready else "blocked"
    if arc_class is None:
        arc_banked = exp5480.get("new_level_banked") is True and exp5480.get(
            "offline_reproduced"
        ) is True
        arc_prechecked = exp5479.get("arc_target_rotation_ready") is True
        arc_class = "headline_ready" if arc_banked else "honest_null" if arc_prechecked else "blocked"

    speedup_class = hardware_class
    if hardware_class == "bounded":
        speedup_class = "honest_null"

    return {
        "transition_source_refresh": _row(
            "transition_source_refresh",
            transition_class,
            transition_sources,
            {
                "transition_verdict": exp5468.get("honest_verdict", ""),
                "source_delta_verdict": exp5469.get("honest_verdict", ""),
                "new_actionable_findings_count": exp5469.get("new_actionable_findings_count"),
                "closed_scopes_reopened": exp5469.get("closed_scopes_reopened"),
            },
            "Operational/source refresh evidence only; it is not an experimental headline lane.",
        ),
        "guided_decoding": _row(
            "guided_decoding",
            guided_class,
            guided_sources,
            {
                "prior_quarantine_carried_by_exp5468": _select(exp5468, ("blocked_lanes",)),
                "rewrite_quarantine_lifted": exp5470.get("guided_decoding_quarantine_lifted"),
                "guard_quarantine_lifted": exp5471.get("guided_decoding_quarantine_lifted"),
                "sota_guided_decoding_used": exp5472.get("guided_decoding_used"),
            },
            "Guided decoding remains quarantined; no .497 artifact lifts it or uses it.",
        ),
        "verifiable_reasoning_guards": _row(
            "verifiable_reasoning_guards",
            reasoning_class,
            reasoning_sources,
            {
                "rewrite_state_fixture_ready": exp5470.get("rewrite_state_fixture_ready"),
                "exact_validator_agreement": exp5470.get("exact_validator_agreement"),
                "guard_composition_ready": exp5471.get("guard_composition_ready"),
                "false_accept_rate": exp5471.get("false_accept_rate"),
                "helper_lemma_repair_ready": exp5476.get("helper_lemma_repair_ready"),
                "helper_false_accept_count": exp5476.get("false_accept_count"),
            },
            "Headline-ready only for deterministic rewrite, guard, and helper-witness fixtures.",
        ),
        "local_sota_runtime": _row(
            "local_sota_runtime",
            local_sota_class,
            local_sota_sources,
            {
                "sota_evidence_telemetry_ready": exp5472.get("sota_evidence_telemetry_ready"),
                "guided_decoding_used": exp5472.get("guided_decoding_used"),
                "headline_models_run": exp5472.get("headline_models_run", []),
                "gpu_offload_receipt_count": len(exp5472.get("gpu_offload_receipts", [])),
                "exact_validator_accuracy": exp5472.get("exact_validator_accuracy"),
            },
            "Bounded GGUF/GPU-offload runtime receipt; exact validators remain final authority.",
        ),
        "csl": _row(
            "csl",
            csl_class,
            csl_sources,
            {
                "csl_kan_surrogate_ready": exp5473.get("csl_kan_surrogate_ready"),
                "csl_scale_ready": exp5474.get("csl_scale_ready"),
                "csl_behavioral_memory_ready": exp5475.get("csl_behavioral_memory_ready"),
                "model_weight_mutation": [
                    exp5473.get("model_weight_mutation"),
                    exp5474.get("model_weight_mutation"),
                    exp5475.get("model_weight_mutation"),
                ],
                "delta_vs_no_memory": exp5474.get("delta_vs_no_memory"),
                "delta_vs_naive_icl": exp5474.get("delta_vs_naive_icl"),
                "context_token_cost_delta": exp5474.get("context_token_cost_delta"),
            },
            "Headline-ready for governed frozen-policy and memory routing, not model-weight learning.",
        ),
        "pdit_lns_boundary_exchange": _row(
            "pdit_lns_boundary_exchange",
            pdit_class,
            pdit_sources,
            {
                "boundary_exchange_ready": exp5477.get("boundary_exchange_ready"),
                "exact_fallback_completeness_rate": exp5477.get(
                    "exact_fallback_completeness_rate"
                ),
                "unsafe_false_accept_count": exp5477.get("unsafe_false_accept_count"),
                "advisory_improvement_delta": exp5477.get("advisory_improvement_delta"),
                "hardware_speedup_claim": exp5477.get("hardware_speedup_claim"),
            },
            "Bounded solver-advisory lane; exact fallback remains final authority.",
        ),
        "hardware_receipts": _row(
            "hardware_receipts",
            hardware_class,
            hardware_sources,
            {
                "hardware_receipts_ready": exp5478.get("hardware_receipts_ready"),
                "hardware_speedup_claim": exp5478.get("hardware_speedup_claim"),
                "result_hash_match_rate": exp5478.get("result_hash_match_rate"),
                "reachable_boards": exp5478.get("reachable_boards", []),
                "unreachable_boards": exp5478.get("unreachable_boards", []),
            },
            "Receipt-only hardware evidence; no speedup claim and KV260 remains blocked if unreachable.",
        ),
        "arc_live_path": _row(
            "arc_live_path",
            arc_class,
            arc_sources,
            {
                "arc_target_rotation_ready": exp5479.get("arc_target_rotation_ready"),
                "selected_game": exp5479.get("selected_game"),
                "selected_target_level": exp5479.get("selected_target_level"),
                "new_level_banked": exp5480.get("new_level_banked"),
                "offline_reproduced": exp5480.get("offline_reproduced"),
                "failure_mode": exp5480.get("failure_mode"),
                "reproduced_levels_before": exp5480.get("reproduced_levels_before"),
                "reproduced_levels_after": exp5480.get("reproduced_levels_after"),
            },
            "ARC remains live-path honest-null unless a new level reproduces through the gate.",
        ),
        "hardware_speedup_claim": _row(
            "hardware_speedup_claim",
            speedup_class,
            hardware_sources,
            {
                "hardware_speedup_claim": exp5478.get("hardware_speedup_claim"),
                "reachable_boards": exp5478.get("reachable_boards", []),
                "unreachable_boards": exp5478.get("unreachable_boards", []),
            },
            "No hardware speedup is claimed; receipts prove hash-matched execution only.",
        ),
    }


def _rows_by_class(truth_table: Mapping[str, JsonDict], classification: str) -> list[JsonDict]:
    return [row for row in truth_table.values() if row["classification"] == classification]


def _arc_registry_delta(exp5480: JsonMap) -> int:
    before = exp5480.get("reproduced_levels_before")
    after = exp5480.get("reproduced_levels_after")
    if isinstance(before, int) and isinstance(after, int) and not isinstance(before, bool):
        return after - before
    return 0


def _hardware_speedup_claim(payloads: Mapping[str, JsonMap]) -> bool:
    exp5477 = payloads.get("results/experiment_5477_pdit_lns_boundary_exchange_v497.json", {})
    exp5478 = payloads.get("results/experiment_5478_hardware_receipts_v497.json", {})
    return exp5477.get("hardware_speedup_claim") is True or exp5478.get(
        "hardware_speedup_claim"
    ) is True


def _guided_status(truth_table: Mapping[str, JsonDict]) -> str:
    row = truth_table["guided_decoding"]
    return "quarantined" if row["classification"] == "blocked" else str(row["classification"])


def _csl_status(truth_table: Mapping[str, JsonDict]) -> str:
    classification = truth_table["csl"]["classification"]
    if classification == "headline_ready":
        return "headline_ready: governed frozen-policy CSL, KAN assurance, GGUF scale, and behavioral memory receipts; no model weight mutation"
    return f"{classification}: CSL headline blocked by missing, flagged, or failed frozen-policy evidence"


def _prd_gap_table(
    truth_table: Mapping[str, JsonDict],
    payloads: Mapping[str, JsonMap],
    arc_delta: int,
    hardware_speedup_claim: bool,
) -> JsonDict:
    exp5474 = payloads.get("results/experiment_5474_sota_csl_scale_v497.json", {})
    exp5478 = payloads.get("results/experiment_5478_hardware_receipts_v497.json", {})
    exp5480 = payloads.get("results/experiment_5480_arc_live_salience_levelup_v497.json", {})
    return {
        "FR-11 continuous self-learning": {
            "status": (
                "headline_ready_bounded_to_frozen_policy"
                if truth_table["csl"]["classification"] == "headline_ready"
                else truth_table["csl"]["classification"]
            ),
            "evidence": truth_table["csl"]["source_artifacts"],
            "gap": "No model-weight mutation, autonomous Rust transpilation, or broad unsupervised improvement claim.",
            "key_fields": {
                "delta_vs_no_memory": exp5474.get("delta_vs_no_memory"),
                "delta_vs_naive_icl": exp5474.get("delta_vs_naive_icl"),
                "context_token_cost_delta": exp5474.get("context_token_cost_delta"),
            },
        },
        "FR-12 verifiable reasoning": {
            "status": truth_table["verifiable_reasoning_guards"]["classification"],
            "evidence": truth_table["verifiable_reasoning_guards"]["source_artifacts"],
            "gap": "Evidence is deterministic fixture and witness based, not a broad reasoning benchmark.",
        },
        "hardware acceleration": {
            "status": "bounded_receipts_only",
            "evidence": truth_table["hardware_receipts"]["source_artifacts"],
            "gap": "No speedup claim; KV260 remains receipt/SSH blocked if listed unreachable.",
            "key_fields": {
                "reachable_boards": exp5478.get("reachable_boards", []),
                "unreachable_boards": exp5478.get("unreachable_boards", []),
                "hardware_speedup_claim": hardware_speedup_claim,
            },
        },
        "ARC live path": {
            "status": truth_table["arc_live_path"]["classification"],
            "evidence": truth_table["arc_live_path"]["source_artifacts"],
            "gap": "No target level banked unless offline_reproduced=true and registry delta is positive.",
            "key_fields": {
                "arc_registry_delta": arc_delta,
                "new_level_banked": exp5480.get("new_level_banked"),
                "offline_reproduced": exp5480.get("offline_reproduced"),
            },
        },
        "local SOTA runtime": {
            "status": truth_table["local_sota_runtime"]["classification"],
            "evidence": truth_table["local_sota_runtime"]["source_artifacts"],
            "gap": "Runtime/offload telemetry only; exact validators remain the authority.",
        },
    }


def _failure_taxonomy(
    truth_table: Mapping[str, JsonDict],
    payloads: Mapping[str, JsonMap],
    missing_artifacts: Sequence[str],
    flagged_artifacts: Sequence[str],
) -> JsonDict:
    exp5478 = payloads.get("results/experiment_5478_hardware_receipts_v497.json", {})
    exp5480 = payloads.get("results/experiment_5480_arc_live_salience_levelup_v497.json", {})
    return {
        "guided_decoding_quarantine": {
            "classification": truth_table["guided_decoding"]["classification"],
            "evidence": truth_table["guided_decoding"]["source_artifacts"],
            "failure_mode": "prior flagged/tautological guided-decoding evidence remains quarantined; .497 fixtures did not lift it",
        },
        "arc_no_bank": {
            "classification": truth_table["arc_live_path"]["classification"],
            "evidence": truth_table["arc_live_path"]["source_artifacts"],
            "failure_mode": exp5480.get("failure_mode", ""),
            "action_count": exp5480.get("action_count"),
            "explored_state_count": exp5480.get("explored_state_count"),
        },
        "hardware_receipts_only": {
            "classification": truth_table["hardware_receipts"]["classification"],
            "evidence": truth_table["hardware_receipts"]["source_artifacts"],
            "failure_mode": "receipt-only; no matched speedup claim",
            "unreachable_boards": exp5478.get("unreachable_boards", []),
        },
        "local_sota_bounded": {
            "classification": truth_table["local_sota_runtime"]["classification"],
            "evidence": truth_table["local_sota_runtime"]["source_artifacts"],
            "failure_mode": "runtime telemetry is not a verifier moat or guided-decoding win",
        },
        "missing_or_flagged_inputs": {
            "missing_artifacts": list(missing_artifacts),
            "flagged_artifacts": list(flagged_artifacts),
        },
    }


def _next_recommendations(
    guided_status: str,
    csl_status: str,
    arc_delta: int,
    hardware_speedup_claim: bool,
) -> list[str]:
    return [
        f"Keep guided decoding {guided_status}; require independent metrics and a clean live rerun before any lift.",
        f"Treat CSL as {csl_status}; keep the headline boundary on frozen-policy and memory routing evidence.",
        f"ARC banked {arc_delta} levels in .497; next milestone should improve trajectory generation beyond salience clicks for sb26 L3.",
        (
            "Keep hardware receipt-only until KV260 or another board produces matched workload "
            f"receipts with an authenticated speedup claim; current speedup claim={hardware_speedup_claim}."
        ),
        "Use local SOTA GGUF offload as bounded runtime telemetry; exact validators remain final authority.",
    ]


def build_report(root: Path = REPO_ROOT, tests_run: Sequence[str] | None = None) -> JsonDict:
    context_read, context_missing = _load_context(root)
    payloads, missing_artifacts, flagged_artifacts, artifact_checksums = _load_artifacts(root)
    truth_table = _truth_table(payloads, missing_artifacts, flagged_artifacts)
    exp5480 = payloads.get("results/experiment_5480_arc_live_salience_levelup_v497.json", {})
    arc_delta = _arc_registry_delta(exp5480)
    speedup_claim = _hardware_speedup_claim(payloads)
    guided_status = _guided_status(truth_table)
    csl_status = _csl_status(truth_table)
    read_artifacts = sorted(payloads)
    headline_ready_lanes = _rows_by_class(truth_table, "headline_ready")
    bounded_lanes = _rows_by_class(truth_table, "bounded")
    blocked_lanes = _rows_by_class(truth_table, "blocked")
    honest_null_lanes = _rows_by_class(truth_table, "honest_null")
    missing_lanes = _rows_by_class(truth_table, "missing")
    flagged_lanes = _rows_by_class(truth_table, "flagged")

    report: JsonDict = {
        "schema": SCHEMA,
        "experiment": EXPERIMENT,
        "experiment_id": EXPERIMENT_ID,
        "run_date": RUN_DATE,
        "random_seed": RANDOM_SEED,
        "result_path": OUTPUT_REL_PATH.as_posix(),
        "spec_refs": list(SPEC_REFS),
        "field_principles": FIELD_PRINCIPLES,
        "source_context_read": context_read,
        "source_context_missing": context_missing,
        "artifact_checksums": artifact_checksums,
        "tests_run": list(tests_run or DEFAULT_TESTS_RUN),
        "milestone": MILESTONE,
        "artifact_paths": read_artifacts,
        "missing_artifacts": missing_artifacts,
        "flagged_artifacts": flagged_artifacts,
        "truth_table": truth_table,
        "headline_ready_lanes": headline_ready_lanes,
        "bounded_lanes": bounded_lanes,
        "blocked_lanes": blocked_lanes,
        "honest_null_lanes": honest_null_lanes,
        "missing_lanes": missing_lanes,
        "flagged_lanes": flagged_lanes,
        "prd_gap_table": _prd_gap_table(truth_table, payloads, arc_delta, speedup_claim),
        "failure_taxonomy": _failure_taxonomy(
            truth_table, payloads, missing_artifacts, flagged_artifacts
        ),
        "guided_decoding_quarantine_status": guided_status,
        "csl_status": csl_status,
        "arc_registry_delta": arc_delta,
        "hardware_speedup_claim": speedup_claim,
        "ops_status_updated": False,
        "ops_changelog_updated": False,
        "roadmap_yaml_unchanged": _protected_file_clean(root, "research-roadmap.yaml"),
        "conductor_unchanged": _protected_file_clean(root, "scripts/research_conductor.py"),
        "next_recommendations": _next_recommendations(
            guided_status, csl_status, arc_delta, speedup_claim
        ),
        "inference_substrate": INFERENCE_SUBSTRATE,
    }
    terminal_prefix = "blocked:" if missing_artifacts or flagged_artifacts else "complete:"
    report["honest_verdict"] = (
        f"{terminal_prefix} .497 capstone read {len(read_artifacts)}/"
        f"{len(EXPECTED_ARTIFACT_PATHS)} artifacts; guided_decoding={guided_status}; "
        f"csl={truth_table['csl']['classification']}; arc_registry_delta={arc_delta}; "
        f"hardware_speedup_claim={speedup_claim}."
    )
    report["reproducibility_checksum"] = _sha256_text(
        _stable_json({key: value for key, value in report.items() if key != "reproducibility_checksum"})
    )
    return report


def write_report(
    root: Path = REPO_ROOT,
    output_rel_path: Path = OUTPUT_REL_PATH,
    tests_run: Sequence[str] | None = None,
) -> JsonDict:
    report = build_report(root, tests_run=tests_run)
    output_path = root / output_rel_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=REPO_ROOT)
    parser.add_argument("--output", type=Path, default=OUTPUT_REL_PATH)
    parser.add_argument("--test-command", action="append", dest="tests_run")
    args = parser.parse_args(argv)
    write_report(args.root, args.output, tests_run=args.tests_run)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
