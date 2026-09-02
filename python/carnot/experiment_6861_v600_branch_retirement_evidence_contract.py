"""Build the immutable V600 evidence root from V599 terminal records.

The reducer reads JSON evidence as data. It never imports a V599 reducer or
runs a model. This keeps procedural completion separate from scientific proof
and lets missing or changed evidence fail closed.

Spec refs: REQ-REPORT-6861 and SCENARIO-REPORT-6861-*.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from collections import defaultdict
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_PATH = Path("results/experiment_6861_v600_branch_retirement_evidence_contract.json")
VERDICT_CLASSES = {
    "positive",
    "circular_positive",
    "null",
    "blocked",
    "disqualified",
    "partial",
}
MODEL_IDS = {
    "unsloth/Qwen3.6-35B-A3B-GGUF",
    "unsloth/gemma-4-31B-it-GGUF",
    "unsloth/gemma-4-26B-A4B-it-GGUF",
}


def _source(
    task_id: str,
    title_prefix: str,
    path: str,
    branch: str,
    frozen_sha256: str,
    *,
    gate_record_allowed: bool = False,
) -> dict[str, object]:
    """Create one source rule with an explicit frozen content identity."""

    return {
        "task_id": task_id,
        "title_prefix": title_prefix,
        "path": path,
        "branch": branch,
        "frozen_sha256": "sha256:" + frozen_sha256,
        "gate_record_allowed": gate_record_allowed,
    }


V599_SOURCES: dict[str, dict[str, object]] = {
    "exp6848": _source(
        "exp6848-v599-method-change-evidence-contract",
        "V599 method-change and evidence contract",
        "results/experiment_6848_v599_method_change_evidence_contract.json",
        "shared_evidence",
        "bac46360f2369727bd19cc774185250860083e82439f1114fcf577bbb34480ef",
    ),
    "exp6849": _source(
        "exp6849-typed-program-isomorphic-authority-audit",
        "Independent typed-program isomorphic authority aud",
        "results/experiment_6849_typed_program_isomorphic_authority_audit.json",
        "typed_authority",
        "3ad1e98df8e95d7c31cf6942d5b6d4c01ab68c7259c2cc88346623012eba6d3a",
    ),
    "exp6850": _source(
        "exp6850-three-family-scoring-admission-canary",
        "Three-family forced-sequence scoring admission can",
        "results/experiment_6850_three_family_scoring_admission_canary.json",
        "local_scoring",
        "0093104ffb1d4f3f22e7ab9369d56561d8d01fa84406faa89d8f0117ced6977d",
    ),
    "exp6851": _source(
        "exp6851-three-family-isomorphic-compatibility-stream",
        "Three-family isomorphic fixed-sequence compatibili",
        "results/experiment_6851_three_family_isomorphic_compatibility_stream.json",
        "compatibility",
        "d9346dfe9d65ff9943f237ffae3e3c0fc4fc74809dece1e36ff55f1bd394841c",
    ),
    "exp6852": _source(
        "exp6852-compatibility-shortcut-authority-audit",
        "Independent compatibility shortcut and claim-autho",
        "results/experiment_6852_compatibility_shortcut_authority_audit.json",
        "compatibility",
        "121dd2640dd50404e081e281953402b2f6e0d36c0aeeb805df049e1debbe07a9",
    ),
    "exp6853": _source(
        "exp6853-risk-sensitive-memory-opportunity-fixture",
        "Risk-sensitive memory opportunity and headroom fix",
        "results/experiment_6853_risk_sensitive_memory_opportunity_fixture.json",
        "self_learning",
        "eba4dc6cd9b369cade53f208b5a2a7cf948ae8179c2a37e5f23c5f66990daf07",
    ),
    "exp6854": _source(
        "exp6854-risk-sensitive-abstention-memory-controller",
        "Risk-sensitive abstention-aware online memory cont",
        "results/experiment_6854_risk_sensitive_abstention_memory_controller.json",
        "self_learning",
        "5cd1d62a7bc75c5a5f9e26bc973ab13522d352ba1a6a60b26afc51d43ed1974c",
    ),
    "exp6855": _source(
        "exp6855-counterfactual-memory-credit-audit",
        "Counterfactual memory-write credit and negative-tr",
        "results/experiment_6855_counterfactual_memory_credit_audit.json",
        "self_learning",
        "613d402ac70ac9b903c868e8e66c64da10367be26d9256fc6e9405bdea5e73e0",
    ),
    "exp6856": _source(
        "exp6856-sealed-risk-sensitive-learning-audit",
        "Sealed risk-sensitive self-learning durability and",
        "results/experiment_6856_sealed_risk_sensitive_learning_audit.json",
        "self_learning",
        "6e6f1856acf0db957ca0427a53422ff8467ba7c04aed296d23eea13e16de7912",
    ),
    "exp6857": _source(
        "exp6857-dynamic-live-arc-receipt-router",
        "Dynamic provenance-qualified live ARC receipt rout",
        "results/experiment_6857_dynamic_live_arc_receipt_router.json",
        "supervisor",
        "822121d5106b7a9f019dd3b092fec607643211f414e2dfecfa9bf98ca08f321f",
    ),
    "exp6858": _source(
        "exp6858-supervisor-counterfactual-credit-audit",
        "Supervisor counterfactual credit audit gated on Ex",
        "results/experiment_6858_supervisor_counterfactual_credit_audit.json",
        "supervisor",
        "6e7b8a9f9da68b96e98ba23be94ed5a38572d573de6ccc4f1535602a44386d9a",
        gate_record_allowed=True,
    ),
    "exp6859": _source(
        "exp6859-first-party-tool-gap-receipt-wiring",
        "First-party tool-gap receipt wiring at the canonic",
        "results/experiment_6859_first_party_tool_gap_receipt_wiring.json",
        "tool_gap",
        "74863bdf181a5f8c4d43565a5cc98ca43a0611228f70a093d89b21621d0c4476",
    ),
    "exp6860": _source(
        "exp6860-v599-independent-capstone",
        "V599 independent adversarial capstone and retireme",
        "results/experiment_6860_v599_independent_capstone.json",
        "v599_capstone",
        "442ad56eeeb9a4351346aa6778d1f7ea28ac968219dcad7f4b6c33537dfd2bd7",
    ),
}


def sha256_path(path: Path) -> str:
    """Hash source bytes so later edits cannot silently change the evidence."""

    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path: Path) -> dict[str, Any] | None:
    """Return one JSON object, or ``None`` when evidence is absent or malformed."""

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def parse_v599_conductor_events(
    text: str, source_specs: dict[str, dict[str, object]]
) -> list[dict[str, str]]:
    """Parse only terminal table rows that match a declared V599 task."""

    events: list[dict[str, str]] = []
    for line in text.splitlines():
        if not line.startswith("|"):
            continue
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if len(cells) < 4:
            continue
        for source_id, spec in source_specs.items():
            title_prefix = str(spec["title_prefix"])
            if cells[1].startswith(title_prefix):
                events.append(
                    {
                        "source_id": source_id,
                        "task_id": str(spec["task_id"]),
                        "timestamp": cells[0],
                        "title": cells[1],
                        "status": cells[2],
                        "detail": cells[3],
                    }
                )
                break
    return events


def inspect_source_inventory(
    repo_root: Path,
    source_specs: dict[str, dict[str, object]] | None = None,
    *,
    conductor_text: str | None = None,
) -> dict[str, Any]:
    """Load every source and preserve missing, gate-skipped, and drift states."""

    specs = source_specs or V599_SOURCES
    if conductor_text is None:
        conductor_path = repo_root / "ops/conductor-log.md"
        conductor_text = conductor_path.read_text(encoding="utf-8")
    events = parse_v599_conductor_events(conductor_text, specs)
    skips = [event for event in events if event["status"] == "GATE_BLOCK"]
    flags = [event for event in events if event["status"] == "FLAGGED"]
    payloads: dict[str, dict[str, Any]] = {}
    source_hashes: dict[str, dict[str, Any]] = {}
    missing: list[str] = []
    drift: list[dict[str, Any]] = []

    for source_id, spec in specs.items():
        relative_path = str(spec["path"])
        path = repo_root / relative_path
        payload = read_json(path)
        gate_skip = next((row for row in skips if row["source_id"] == source_id), None)
        actual_hash = sha256_path(path) if path.is_file() else None
        expected_hash = str(spec["frozen_sha256"])
        if payload is not None:
            payloads[source_id] = payload
        elif gate_skip is None or spec.get("gate_record_allowed") is not True:
            missing.append(relative_path)
        if actual_hash is not None and actual_hash != expected_hash:
            drift.append(
                {
                    "source_id": source_id,
                    "path": relative_path,
                    "expected_sha256": expected_hash,
                    "observed_sha256": actual_hash,
                }
            )
        source_hashes[source_id] = {
            "task_id": spec["task_id"],
            "path": relative_path,
            "expected_sha256": expected_hash,
            "sha256": actual_hash,
            "hash_matches": actual_hash == expected_hash if actual_hash else None,
            "state": "terminal"
            if payload is not None
            else "conductor_gate"
            if gate_skip
            else "missing",
            "immutable": True,
        }

    checks = [
        {
            "check": "terminal_source_inventory",
            "expected": [],
            "observed": missing,
            "passed": not missing,
        },
        {
            "check": "immutable_source_hashes",
            "expected": [],
            "observed": drift,
            "passed": not drift,
        },
    ]
    failed = [check for check in checks if not check["passed"]]
    flagged_ids = {
        source_id
        for source_id, payload in payloads.items()
        if payload.get("flagged_adversarial") is True
    } | {event["source_id"] for event in flags}
    return {
        "payloads": payloads,
        "source_artifact_hashes": source_hashes,
        "missing_artifact_inventory": missing,
        "source_drift_manifest": drift,
        "conductor_skip_manifest": skips,
        "conductor_flag_events": flags,
        "flagged_source_ids": sorted(flagged_ids),
        "preconditions_passed": not failed,
        "gate_check_summary": {
            "passed": not failed,
            "checks": checks,
            "failed_check": failed[0]["check"] if failed else None,
            "failed_checks": failed,
            "observed": failed[0]["observed"] if failed else "all checks pass",
        },
    }


def _mean(values: list[float]) -> float | None:
    """Keep an absent measurement distinct from a measured numeric zero."""

    return sum(values) / len(values) if values else None


def recompute_dispositions(
    payloads: dict[str, dict[str, Any]], *, flagged_source_ids: set[str]
) -> dict[str, dict[str, Any]]:
    """Recompute five branch outcomes from rows without producer aggregates."""

    authority = payloads.get("exp6849", {})
    authority_rows = [row for row in authority.get("rows", []) if isinstance(row, dict)]
    candidates = [
        row for row in authority.get("candidate_identity_manifest", []) if isinstance(row, dict)
    ]
    candidate_ids = [row.get("candidate_id") for row in candidates]
    semantic_ids = [row.get("semantic_identity") for row in candidates]
    typed_ready = bool(authority_rows) and all(
        row.get("parity_passed") is True
        and row.get("atom_identities_match") is True
        and row.get("observed_label") == row.get("expected_label")
        for row in authority_rows
    )
    typed_ready = (
        typed_ready
        and bool(candidate_ids)
        and len(candidate_ids) == len(set(candidate_ids))
        and len(semantic_ids) == len(set(semantic_ids))
    )

    scoring = payloads.get("exp6851", {})
    scoring_rows = [row for row in scoring.get("rows", []) if isinstance(row, dict)]
    scoring_models = {row.get("model_hf_id") for row in scoring_rows}
    local_scoring_ready = len(scoring_rows) == 168 and scoring_models == MODEL_IDS

    compatibility_audit = payloads.get("exp6852", {})
    attack_rows = [
        row
        for row in compatibility_audit.get("rows", [])
        if isinstance(row, dict) and row.get("row_kind") == "shortcut_attack"
    ]
    failed_attacks = [
        row
        for row in attack_rows
        if row.get("passed") is not True or row.get("explains_same_or_greater") is True
    ]
    invariance_rows = [
        row
        for row in compatibility_audit.get("isomorphic_invariance_results", [])
        if isinstance(row, dict)
    ]
    failed_invariance = [
        row for row in invariance_rows if row.get("direction_survives") is not True
    ]
    compatibility_eligible = (
        typed_ready
        and local_scoring_ready
        and bool(attack_rows)
        and not failed_attacks
        and bool(invariance_rows)
        and not failed_invariance
    )

    sealed = payloads.get("exp6856", {})
    held_rows = [
        row
        for row in sealed.get("rows", [])
        if isinstance(row, dict)
        and row.get("arm") == "contextual_bandit"
        and row.get("split") == "held_future"
    ]
    held_effect = _mean(
        [
            float(row["effect_vs_no_memory"])
            for row in held_rows
            if isinstance(row.get("effect_vs_no_memory"), (int, float))
        ]
    )
    abstention_rate = (
        sum(row.get("abstained") is True for row in held_rows) / len(held_rows)
        if held_rows
        else None
    )
    credit_rows = [
        row
        for row in payloads.get("exp6855", {}).get("rows", [])
        if isinstance(row, dict) and row.get("counterfactual_metric") == "marginal_value"
    ]
    credit_by_write: dict[str, float] = defaultdict(float)
    for row in credit_rows:
        write_id = row.get("write_id")
        value = row.get("metric_value")
        if isinstance(write_id, str) and isinstance(value, (int, float)):
            credit_by_write[write_id] += float(value)
    harmful_writes = sum(value < 0 for value in credit_by_write.values())
    memory_flagged = "exp6856" in flagged_source_ids or sealed.get("flagged_adversarial") is True
    restart = sealed.get("restart_results", {})
    rollback = sealed.get("rollback_results", {})
    transaction_reusable = (
        isinstance(restart, dict)
        and restart.get("byte_identical") is True
        and isinstance(rollback, dict)
        and rollback.get("restored_parent_bytes") is True
    )

    router = payloads.get("exp6857", {})
    headroom_rows = [
        row for row in router.get("supervisor_headroom_rows", []) if isinstance(row, dict)
    ]
    eligible_headroom = [
        row
        for row in headroom_rows
        if row.get("valid_headroom") is True
        and row.get("live_reachable") is True
        and row.get("provenance_class") == "authentic_live"
    ]

    tool_rows = [
        row
        for row in payloads.get("exp6859", {}).get("rows", [])
        if isinstance(row, dict) and row.get("row_kind") == "gap_chain"
    ]
    complete_receipts = [
        row
        for row in tool_rows
        if row.get("join_complete") is True and row.get("first_party") is True
    ]
    authentic_receipts = [
        row
        for row in complete_receipts
        if row.get("provenance_class") == "authentic_live" and row.get("live_reachable") is True
    ]

    return {
        "typed_authority": {
            "branch": "typed_authority",
            "verdict_class": "positive" if typed_ready else "null",
            "exact_authority_ready": typed_ready,
            "authority_row_count": len(authority_rows),
            "scope": "reusable_infrastructure",
        },
        "compatibility": {
            "branch": "compatibility",
            "verdict_class": "positive" if compatibility_eligible else "null",
            "scientific_claim_eligible": compatibility_eligible,
            "local_scoring_ready": local_scoring_ready,
            "scoring_row_count": len(scoring_rows),
            "shortcut_failure_count": len(failed_attacks),
            "isomorphic_direction_failure_count": len(failed_invariance),
            "scope": "scientific_claim",
        },
        "self_learning": {
            "branch": "self_learning",
            "verdict_class": "disqualified" if memory_flagged else "null",
            "scientific_claim_eligible": False,
            "held_future_effect": held_effect,
            "abstention_rate": abstention_rate,
            "harmful_write_count": harmful_writes,
            "flagged_controlling_source": memory_flagged,
            "transaction_infrastructure_reusable": transaction_reusable,
            "scope": "scientific_claim_and_reusable_infrastructure",
        },
        "supervisor": {
            "branch": "supervisor",
            "verdict_class": "blocked" if not eligible_headroom else "partial",
            "scientific_claim_eligible": False,
            "eligible_headroom_row_count": len(eligible_headroom),
            "failed_gate_field": "supervisor_headroom_ready_score",
            "observed_gate_value": len(eligible_headroom),
            "scope": "retired_claim_mechanism",
        },
        "tool_gap": {
            "branch": "tool_gap",
            "verdict_class": "partial" if complete_receipts else "blocked",
            "scientific_claim_eligible": bool(authentic_receipts),
            "receipt_contract_ready": bool(complete_receipts),
            "complete_receipt_count": len(complete_receipts),
            "authentic_live_chain_count": len(authentic_receipts),
            "scope": "reusable_infrastructure_without_live_effect",
        },
    }


def retired_mechanism_manifest() -> list[dict[str, Any]]:
    """Name only failed claim mechanisms, not the infrastructure beneath them."""

    return [
        {
            "mechanism_id": "raw_fixed_sequence_margin_claim",
            "source_ids": ["exp6851", "exp6852"],
            "reason": "Nuisance controls and isomorphic transforms explain margins as large as the semantic effect.",
            "retired_scope": "scientific compatibility claim from raw fixed-sequence margins",
        },
        {
            "mechanism_id": "unchanged_supervisor_credit_without_headroom",
            "source_ids": ["exp6857", "exp6858"],
            "reason": "No authentic live row had nonzero pre-action supervisor headroom.",
            "retired_scope": "another unchanged supervisor-credit audit",
        },
        {
            "mechanism_id": "v599_risk_sensitive_memory_policy",
            "source_ids": ["exp6855", "exp6856"],
            "reason": "The sealed policy abstained on every held opportunity and retained harmful writes.",
            "retired_scope": "the same risk-sensitive memory decision policy",
        },
    ]


def changed_mechanism_manifest() -> list[dict[str, Any]]:
    """Describe the exact mechanism changes required before new measurement."""

    return [
        {
            "mechanism_id": "dual_side_nuisance_matched_semantic_contrast",
            "replaces": "raw_fixed_sequence_margin_claim",
            "reuses_retired_mechanism": False,
            "method_delta": "Use separate exact structure and solution authorities on token-matched semantic contrasts frozen before scoring.",
            "falsifiable_gate": "A held paired effect must have a confidence interval above zero in both model families and exceed every matched nuisance effect.",
            "consumers": ["exp6862", "exp6863", "exp6864", "exp6865", "exp6866"],
        },
        {
            "mechanism_id": "decision_time_observability_constrained_memory",
            "replaces": "v599_risk_sensitive_memory_policy",
            "reuses_retired_mechanism": False,
            "method_delta": "Use only decision-time observable inputs, force bounded exploration, and quarantine writes whose pessimistic benefit does not clear harm cost.",
            "falsifiable_gate": "Held actions must be nondegenerate, improve exact later outcomes, reduce harmful writes, survive restart and rollback, and show zero leakage.",
            "consumers": ["exp6867", "exp6868", "exp6869"],
        },
        {
            "mechanism_id": "resumable_authentic_first_party_tool_gap_receipts",
            "replaces": "unchanged_supervisor_credit_without_headroom",
            "reuses_retired_mechanism": False,
            "method_delta": "Capture atomic first-party receipt chains on the canonical adapter-disabled live path before any matched delivery-versus-withholding test.",
            "falsifiable_gate": "At least one authentic chain must replay from the same pre-gap state before exact delivered and withheld outcomes can be compared.",
            "consumers": ["exp6870", "exp6871", "exp6872"],
        },
    ]


def validate_changed_mechanisms(
    rows: list[dict[str, Any]], retired_ids: set[str]
) -> list[dict[str, str]]:
    """Reject an unchanged failed mechanism disguised as a new task name."""

    invalid: list[dict[str, str]] = []
    for row in rows:
        mechanism_id = str(row.get("mechanism_id", ""))
        unchanged_reuse = row.get("reuses_retired_mechanism") is True or mechanism_id in retired_ids
        if unchanged_reuse and (not row.get("method_delta") or not row.get("falsifiable_gate")):
            invalid.append(
                {
                    "mechanism_id": mechanism_id,
                    "reason": "retired_mechanism_reused_without_method_delta_and_gate",
                }
            )
    return invalid


def build_split_and_gate_contract() -> dict[str, Any]:
    """Define one field vocabulary for all downstream V600 evidence."""

    return {
        "schema": "carnot.v600.split_and_gate_contract.v1",
        "calibration_split_hash_field": "calibration_split_sha256",
        "held_split_hash_field": "held_split_sha256",
        "split_disjointness_field": "calibration_held_group_overlap_count",
        "split_rules": {
            "group_identity_field": "contrast_group_identity",
            "chronological_identity_field": "decision_sequence_index",
            "freeze_before_fields": ["model_score", "held_label", "policy_action"],
            "held_open_count_field": "held_split_open_count",
            "required_held_open_count": 1,
        },
        "provenance_schema": {
            "schema": "carnot.v600.evidence_provenance.v1",
            "required_fields": [
                "source_sha256",
                "calibration_split_sha256",
                "held_split_sha256",
                "model_hf_id",
                "model_artifact_sha256",
                "tokenizer_identity",
                "tokenizer_sha256",
                "process_owner_pid",
                "process_owner_start_time",
                "lease_identity",
                "exact_structure_authority",
                "exact_solution_authority",
                "exact_later_outcome_identity",
                "live_receipt_identity",
                "conductor_skip_identity",
            ],
            "null_policy": "Unavailable evidence stays null. It never becomes numeric zero.",
        },
        "claim_fields": {
            "verdict_class": "verdict_class",
            "honest_verdict": "honest_verdict",
            "scientific_claim_eligible": "scientific_claim_eligible",
            "verifier_is_oracle": "verifier_is_oracle",
            "flagged_source_manifest": "flagged_source_manifest",
        },
        "downstream_gate_fields": [
            {
                "consumer": "exp6862-dual-side-semantic-contrast-bank",
                "artifact_field": "v600_evidence_contract_ready_score",
                "op": "==",
                "value": 1,
            },
            {
                "consumer": "exp6867-decision-time-observability-firewall",
                "artifact_field": "v600_evidence_contract_ready_score",
                "op": "==",
                "value": 1,
            },
            {
                "consumer": "exp6870-resumable-live-arc-receipt-checkpoint-harness",
                "artifact_field": "v600_evidence_contract_ready_score",
                "op": "==",
                "value": 1,
            },
        ],
        "branch_gate_fields": {
            "semantic_contrast": [
                "structure_side_exact_passed",
                "solution_side_exact_passed",
                "token_count_matched",
                "calibration_held_group_overlap_count",
                "held_paired_effect",
                "held_paired_effect_ci95_low",
                "max_matched_nuisance_effect",
                "both_model_families_agree",
            ],
            "observability_safe_memory": [
                "decision_time_observability_passed",
                "exploration_floor_met",
                "harmful_write_count",
                "held_future_effect",
                "false_positive_injection_rate",
                "restart_equivalent",
                "rollback_equivalent",
                "leakage_path_count",
            ],
            "live_tool_gap": [
                "authentic_live_tool_gap_chain_count",
                "replayable_chain_count",
                "pre_gap_state_hash_match",
                "delivery_outcome",
                "withheld_outcome",
                "matched_live_effect_eligible_score",
            ],
        },
    }


def reference_verification_rows() -> list[dict[str, Any]]:
    """Return facts checked against primary arXiv pages on 2026-09-02."""

    boundary = (
        "Primary arXiv abstract and metadata page only. No paper code, model weights, "
        "private data, or unreported implementation behavior was treated as available."
    )
    return [
        {
            "arxiv_id": "2606.10616",
            "primary_url": "https://arxiv.org/abs/2606.10616",
            "title": "Learning What to Remember: Observability-Safe Memory Retention via Constrained Optimization for Long-Horizon Language Agents",
            "primary_page_verified": True,
            "verified_on": "2026-09-02",
            "primary_page_facts": {"submitted": "2026-06-09", "current_version": "v6"},
            "method_delta": "OSL-MR separates online-observable decision features from offline supervision and optimizes retention under budget and delayed costs.",
            "carnot_hook": "Exp6867 must label every policy input by decision-time availability and reject outcome-only features before action selection.",
            "access_boundary": boundary,
        },
        {
            "arxiv_id": "2605.29556",
            "primary_url": "https://arxiv.org/abs/2605.29556",
            "title": "Opt-Verifier: Unleashing the Power of LLMs for Optimization Modeling via Dual-Side Verification",
            "primary_page_verified": True,
            "verified_on": "2026-09-02",
            "primary_page_facts": {"submitted": "2026-05-28", "venue": "ICML 2026"},
            "planner_metadata_correction": {
                "recorded_venue": "ICLR 2026",
                "primary_page_venue": "ICML 2026",
            },
            "method_delta": "The method verifies generated optimization models from separate structure and solution perspectives.",
            "carnot_hook": "Exp6862 requires independent exact structure-side and solution-side checks before any contrast reaches scoring.",
            "access_boundary": boundary,
        },
        {
            "arxiv_id": "2604.09459",
            "primary_url": "https://arxiv.org/abs/2604.09459",
            "title": "From Reasoning to Agentic: Credit Assignment in Reinforcement Learning for Large Language Models",
            "primary_page_verified": True,
            "verified_on": "2026-09-02",
            "primary_page_facts": {
                "submitted": "2026-04-10",
                "current_version": "v3",
                "paper_count": 69,
                "core_method_count": 56,
                "boundary_enabler_count": 13,
            },
            "planner_metadata_correction": {
                "recorded_method_count": 47,
                "current_primary_page_core_method_count": 56,
            },
            "method_delta": "The current survey adds restored-state identification limits and a claim card that binds estimand, provenance, and falsification.",
            "carnot_hook": "Freeze credit at an observable decision or receipt, require a valid pre-action alternative, and reveal exact outcomes later.",
            "access_boundary": boundary,
        },
        {
            "arxiv_id": "2608.15008",
            "primary_url": "https://arxiv.org/abs/2608.15008",
            "title": "Harness the Memory: A Holistic Evaluation of Memory Substrates in Memory Agents",
            "primary_page_verified": True,
            "verified_on": "2026-09-02",
            "primary_page_facts": {
                "submitted": "2026-08-15",
                "backbone_model_count": 3,
                "benchmark_suite_count": 4,
                "metric_count": 26,
            },
            "method_delta": "A controlled harness finds no universal memory substrate and reports that excessive retrieval can harm sequential decisions.",
            "carnot_hook": "Compare verified-write, read-only, no-memory, and abstain on the same chronological opportunities instead of assuming retrieval helps.",
            "access_boundary": boundary,
        },
    ]


def _preserved_infrastructure(
    payloads: dict[str, dict[str, Any]], dispositions: dict[str, dict[str, Any]]
) -> list[dict[str, Any]]:
    """Summarize reusable receipts without copying large raw evidence tables."""

    scoring = payloads.get("exp6851", {})
    process_rows = [row for row in scoring.get("process_receipts", []) if isinstance(row, dict)]
    tokenizer_rows = [row for row in scoring.get("tokenizer_receipts", []) if isinstance(row, dict)]
    model_hashes = scoring.get("model_artifact_hashes", {})
    return [
        {
            "infrastructure_id": "typed_exact_authority",
            "source_ids": ["exp6849"],
            "ready": dispositions["typed_authority"]["exact_authority_ready"],
            "preserved_fields": ["typed atoms", "exact labels", "collision-free identities"],
        },
        {
            "infrastructure_id": "three_family_local_scoring",
            "source_ids": ["exp6850", "exp6851"],
            "ready": dispositions["compatibility"]["local_scoring_ready"],
            "model_identities": sorted(str(key) for key in model_hashes),
            "model_artifact_hashes": {
                str(key): value.get("sha256")
                for key, value in model_hashes.items()
                if isinstance(value, dict)
            },
            "tokenizer_identities": [
                {"model_hf_id": row.get("hf_id"), "tokenizer_sha256": row.get("tokenizer_sha256")}
                for row in tokenizer_rows
            ],
            "process_ownership_identities": [
                {
                    "model_hf_id": row.get("hf_id"),
                    "owner_pid": row.get("owner_pid"),
                    "owner_start_time": row.get("owner_start_time_ticks"),
                    "ownership_token_digest": row.get("ownership_token_digest"),
                }
                for row in process_rows
            ],
        },
        {
            "infrastructure_id": "memory_transaction_and_rollback",
            "source_ids": ["exp6854", "exp6856"],
            "ready": dispositions["self_learning"]["transaction_infrastructure_reusable"],
            "preserved_fields": ["transaction identity", "restart bytes", "rollback parent bytes"],
        },
        {
            "infrastructure_id": "first_party_tool_gap_receipts",
            "source_ids": ["exp6859"],
            "ready": dispositions["tool_gap"]["receipt_contract_ready"],
            "preserved_fields": [
                "receipt_identity",
                "hop_identity",
                "previous_hop_identity",
                "exact_outcome",
            ],
        },
    ]


def _field_principles() -> dict[str, str]:
    """Explain why each required field exists in the audit contract."""

    return {
        "preconditions_checked": "A complete inventory prevents missing evidence from becoming a measured zero.",
        "inference_substrate": "The declared CPU replay distinguishes this reduction from model inference.",
        "duration_s": "Measured wall time exposes implausible execution claims.",
        "source_artifact_hashes": "Frozen content hashes detect silent source drift.",
        "reproducibility_checksum": "A canonical digest lets another reducer confirm the same evidence root.",
        "rows": "Per-source and per-decision rows keep every aggregate auditable.",
        "terminal_branch_manifest": "One closed class per branch prevents procedural success from promoting science.",
        "conductor_skip_manifest": "Structured skips preserve blocked states without inventing artifacts.",
        "flagged_source_manifest": "Flags quarantine scientific claims while preserving checkable receipts.",
        "retired_mechanism_manifest": "Narrow retirement prevents failed claims from discarding useful infrastructure.",
        "preserved_infrastructure_manifest": "Reusable exact receipts remain available to changed mechanisms.",
        "changed_mechanism_manifest": "A method delta and falsifiable gate prevent unchanged failed reruns.",
        "split_and_gate_contract": "Exact field names stop downstream split and gate reinterpretation.",
        "source_access_boundaries": "Access limits prevent metadata checks from becoming implementation claims.",
        "reference_verification_rows": "Primary-page checks bind each literature claim to a visible source.",
        "v600_evidence_contract_ready_score": "Downstream tasks share one fail-closed readiness gate.",
        "gate_check_summary": "Blocked output names the first failed check and its observed value.",
        "verifier_is_oracle": "The reducer organizes evidence but does not define scientific truth.",
        "verdict_class": "A closed class keeps positive, null, blocked, partial, and disqualified states distinct.",
        "honest_verdict": "A terminal prefix lets the conductor classify completion without guessing.",
    }


def reproducibility_checksum(artifact: dict[str, Any]) -> str:
    """Hash stable evidence fields while excluding wall time and the digest itself."""

    stable = {
        key: value
        for key, value in artifact.items()
        if key not in {"duration_s", "reproducibility_checksum", "result_path"}
    }
    encoded = json.dumps(stable, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return "sha256:" + hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def build_artifact(
    repo_root: Path,
    run_date: str,
    *,
    source_specs: dict[str, dict[str, object]] | None = None,
    conductor_text: str | None = None,
) -> dict[str, Any]:
    """Build a terminal contract even when a source precondition fails."""

    started = time.monotonic()
    inventory = inspect_source_inventory(
        repo_root,
        source_specs,
        conductor_text=conductor_text,
    )
    payloads = inventory["payloads"]
    flagged_ids = set(inventory["flagged_source_ids"])
    dispositions = recompute_dispositions(payloads, flagged_source_ids=flagged_ids)
    capstone = payloads.get("exp6860", {})
    capstone_branch = {
        "branch": "v599_capstone",
        "verdict_class": "positive"
        if str(capstone.get("honest_verdict", "")).startswith("complete_")
        else "null",
        "scope": "procedural_only",
        "scientific_branch_advance_count": 0,
        "source_verdict": capstone.get("honest_verdict"),
    }
    terminal_branches = [*dispositions.values(), capstone_branch]
    retired = retired_mechanism_manifest()
    changed = changed_mechanism_manifest()
    retired_ids = {row["mechanism_id"] for row in retired}
    invalid_changes = validate_changed_mechanisms(changed, retired_ids)
    references = reference_verification_rows()
    split_contract = build_split_and_gate_contract()
    preserved = _preserved_infrastructure(payloads, dispositions)
    flagged_manifest = [
        {
            "source_id": source_id,
            "task_id": inventory["source_artifact_hashes"][source_id]["task_id"],
            "claim_use": "quarantined",
            "infrastructure_use": "allowed_only_when_independently_checkable",
        }
        for source_id in inventory["flagged_source_ids"]
    ]
    readiness_checks = [
        {
            "check": "source_preconditions",
            "expected": True,
            "observed": inventory["preconditions_passed"],
            "passed": inventory["preconditions_passed"],
        },
        {
            "check": "changed_mechanisms_valid",
            "expected": [],
            "observed": invalid_changes,
            "passed": not invalid_changes,
        },
        {
            "check": "primary_references_verified",
            "expected": 4,
            "observed": sum(row["primary_page_verified"] is True for row in references),
            "passed": all(row["primary_page_verified"] is True for row in references),
        },
        {
            "check": "downstream_gate_consumers_named",
            "expected": ["exp6862", "exp6867", "exp6870"],
            "observed": [
                str(row["consumer"]).split("-", 1)[0]
                for row in split_contract["downstream_gate_fields"]
            ],
            "passed": len(split_contract["downstream_gate_fields"]) == 3,
        },
    ]
    failed = [check for check in readiness_checks if not check["passed"]]
    ready = not failed
    gate_summary = (
        {
            "passed": True,
            "checks": readiness_checks,
            "failed_check": None,
            "failed_checks": [],
            "observed": "all checks pass",
        }
        if ready
        else {
            "passed": False,
            "checks": readiness_checks,
            "failed_check": inventory["gate_check_summary"]["failed_check"] or failed[0]["check"],
            "failed_checks": inventory["gate_check_summary"]["failed_checks"] or failed,
            "observed": inventory["gate_check_summary"]["observed"]
            if not inventory["preconditions_passed"]
            else failed[0]["observed"],
        }
    )
    source_rows = [
        {"row_kind": "source", "source_id": source_id, **row}
        for source_id, row in inventory["source_artifact_hashes"].items()
    ]
    rows = (
        source_rows
        + [{"row_kind": "disposition", **row} for row in terminal_branches]
        + [{"row_kind": "retirement", **row} for row in retired]
        + [{"row_kind": "reference", **row} for row in references]
    )
    artifact: dict[str, Any] = {
        "schema": "carnot.v600.branch_retirement_evidence_contract.v1",
        "experiment_id": "exp6861-v600-branch-retirement-evidence-contract",
        "run_date": run_date,
        "status": "complete",
        "field_principles": _field_principles(),
        "preconditions_checked": [
            {
                "check": "every_v599_terminal_artifact_or_explicit_gate_record",
                "available": not inventory["missing_artifact_inventory"],
                "observed": inventory["missing_artifact_inventory"],
            },
            {
                "check": "immutable_source_hashes_stable",
                "available": not inventory["source_drift_manifest"],
                "observed": inventory["source_drift_manifest"],
            },
            {"check": "no_llm_invoked", "available": True, "observed": True},
        ],
        "inference_substrate": "deterministic CPU evidence replay",
        "duration_s": max(time.monotonic() - started, 0.0001),
        "source_artifact_hashes": inventory["source_artifact_hashes"],
        "reproducibility_checksum": "",
        "rows": rows,
        "terminal_branch_manifest": terminal_branches,
        "conductor_skip_manifest": inventory["conductor_skip_manifest"],
        "flagged_source_manifest": flagged_manifest,
        "retired_mechanism_manifest": retired,
        "preserved_infrastructure_manifest": preserved,
        "changed_mechanism_manifest": changed,
        "split_and_gate_contract": split_contract,
        "source_access_boundaries": [
            {
                "source_class": "V599 local terminal artifacts",
                "boundary": "Read-only JSON replay. No V599 experiment, producer module, or model process was run.",
            },
            {
                "source_class": "promoted research references",
                "boundary": references[0]["access_boundary"],
            },
            {
                "source_class": "ARC live evidence",
                "boundary": "Terminal receipts only. No game source, live environment, adapter, or prior trajectory was accessed.",
            },
        ],
        "reference_verification_rows": references,
        "fresh_reducer_manifest": {
            "producer_modules_imported": [],
            "producer_aggregate_functions_imported": [],
            "llm_invocations": 0,
            "algorithms": [
                "typed exact row and identity replay",
                "raw compatibility shortcut and invariance replay",
                "held-future abstention and per-write marginal credit replay",
                "live headroom and first-party receipt replay",
            ],
        },
        "missing_artifact_inventory": inventory["missing_artifact_inventory"],
        "source_drift_manifest": inventory["source_drift_manifest"],
        "v600_evidence_contract_ready_score": int(ready),
        "gate_check_summary": gate_summary,
        "verifier_is_oracle": False,
        "verdict_class": "positive" if ready else "blocked",
        "honest_verdict": (
            "complete_positive_v600_branch_retirement_evidence_contract_ready"
            if ready
            else "complete_blocked_v600_branch_retirement_evidence_contract"
        ),
        "result_path": str(OUTPUT_PATH),
    }
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    return artifact


def validate_artifact(artifact: dict[str, Any]) -> list[str]:
    """Return all material contract errors instead of hiding later failures."""

    required = {
        "field_principles",
        "preconditions_checked",
        "inference_substrate",
        "duration_s",
        "source_artifact_hashes",
        "reproducibility_checksum",
        "rows",
        "terminal_branch_manifest",
        "conductor_skip_manifest",
        "flagged_source_manifest",
        "retired_mechanism_manifest",
        "preserved_infrastructure_manifest",
        "changed_mechanism_manifest",
        "split_and_gate_contract",
        "source_access_boundaries",
        "reference_verification_rows",
        "v600_evidence_contract_ready_score",
        "gate_check_summary",
        "verifier_is_oracle",
        "verdict_class",
        "honest_verdict",
    }
    missing = sorted(required - artifact.keys())
    errors = ["missing_required_fields:" + ",".join(missing)] if missing else []
    if artifact.get("inference_substrate") != "deterministic CPU evidence replay":
        errors.append("invalid_inference_substrate")
    if artifact.get("verifier_is_oracle") is not False:
        errors.append("verifier_is_oracle_must_be_false")
    if artifact.get("verdict_class") not in VERDICT_CLASSES:
        errors.append("invalid_verdict_class")
    if not str(artifact.get("honest_verdict", "")).startswith("complete_"):
        errors.append("honest_verdict_not_terminal")
    if artifact.get("reproducibility_checksum") != reproducibility_checksum(artifact):
        errors.append("reproducibility_checksum_mismatch")
    return errors


def write_atomic(path: Path, artifact: dict[str, Any]) -> None:
    """Replace the result atomically so interruption cannot leave partial JSON."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def main(argv: list[str] | None = None) -> int:
    """Run deterministic evidence replay and write the requested artifact."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", required=True)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    args = parser.parse_args(argv)
    artifact = build_artifact(REPO_ROOT, args.date)
    errors = validate_artifact(artifact)
    if errors:
        raise ValueError(";".join(errors))
    write_atomic(args.output, artifact)
    print(json.dumps({"output": str(args.output), "honest_verdict": artifact["honest_verdict"]}))
    return 0


if __name__ == "__main__":  # pragma: no cover - the wrapper exercises this path.
    raise SystemExit(main())
