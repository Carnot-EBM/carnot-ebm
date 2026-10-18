#!/usr/bin/env python3
"""Exp5471 deterministic guard-composition scale-up.

Spec refs: REQ-SAFE-5471, SCENARIO-SAFE-5471.

The experiment composes the clean Exp5470 rewrite-state fixture with three
exact guard families.  A repair proposal score is recorded because downstream
repair systems often have one, but the score is deliberately advisory: guard
catches, final acceptance, and readiness are recomputed only from exact local
validator evidence.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

from carnot import experiment_5470_rewrite_state_semantic_fixture_v497 as exp5470


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_RELATIVE_PATH = Path("results/experiment_5471_guard_composition_scale_v497.json")
EXPERIMENT_ID = "experiment_5471_guard_composition_scale_v497"
TASK_ID = "exp5471-v497-guard-composition-scale"
MILESTONE = "2026.07.497"
RUN_DATE = "2026-07-09"
SCHEMA = "carnot.experiment_5471.guard_composition_scale.v497"
SPEC_REFS = ("REQ-SAFE-5471", "SCENARIO-SAFE-5471", "REQ-SAFE-5470")
RANDOM_SEED = 5471
INFERENCE_SUBSTRATE = exp5470.INFERENCE_SUBSTRATE
EXACT_FINAL_AUTHORITY = "exact_guard_composition_validators"
TERMINAL_PREFIXES = ("complete:", "blocked:")

GUARD_IDS = (
    "license_transition_guard",
    "semantic_graph_guard",
    "distortion_guard",
)

FIELD_PRINCIPLES: dict[str, str] = {
    "fixture_count": "scaled deterministic fixture coverage.",
    "minimal_core_count": (
        "distinct unsatisfied-constraint minimal core IDs, separate from semantic graph nodes."
    ),
    "semantic_graph_node_count": "distinct semantic graph node IDs observed in receipts.",
    "guard_overlap_matrix": "per-guard catch and pairwise overlap rates from exact evidence.",
    "false_accept_rate": "invalid candidates accepted by exact final validators.",
    "false_reject_rate": "valid candidates rejected by exact final validators.",
    "exact_final_agreement": "exact final verdicts matching curated expected labels.",
    "guard_composition_ready": "downstream gate for deterministic guard composition.",
    "guided_decoding_quarantine_lifted": "guided-decoding quarantine remains closed.",
    "inference_substrate": "deterministic fixture with no LLM inference.",
    "random_seed": "deterministic fixture seed.",
    "honest_verdict": 'terminal status; start with "complete:" or "blocked:".',
}
REQUIRED_ARTIFACT_FIELDS = tuple(FIELD_PRINCIPLES)


@dataclass(frozen=True)
class GuardScaleCandidate:
    """One Exp5470-shaped transition plus composition-specific expectations."""

    transition: exp5470.TransitionCandidate
    expected_guard_ids: tuple[str, ...]
    repair_proposal_score: float
    repair_action_hint: str

    @property
    def candidate_id(self) -> str:
        """Expose the stable transition ID used by tests and artifacts."""

        return self.transition.candidate_id


def build_candidates() -> list[GuardScaleCandidate]:
    """Return Exp5470 rows plus five deterministic scale-up transitions."""

    expected_guards = {
        "5470-valid-fact-paraphrase": (),
        "5470-valid-arithmetic": (),
        "5470-hidden-premise": ("license_transition_guard",),
        "5470-unlicensed-state-change": ("license_transition_guard",),
        "5470-fabricated-citation": ("license_transition_guard",),
        "5470-json-semantic-invalid": ("semantic_graph_guard",),
        "5470-api-precondition-invalid": ("semantic_graph_guard",),
        "5470-factual-distortion": ("distortion_guard",),
    }
    candidates: list[GuardScaleCandidate] = []
    for index, transition in enumerate(exp5470.build_candidates()):
        candidates.append(
            GuardScaleCandidate(
                transition=transition,
                expected_guard_ids=expected_guards[transition.candidate_id],
                repair_proposal_score=round(0.13 + index * 0.053, 6),
                repair_action_hint="minimal_core_feedback" if not transition.expected_accept else "none",
            )
        )
    candidates.extend(_scaleup_candidates())
    return candidates


def evaluate_candidates(candidates: Sequence[GuardScaleCandidate]) -> list[JsonDict]:
    """Evaluate each candidate with exact guards and attach receipts."""

    rows: list[JsonDict] = []
    for candidate in candidates:
        transition = candidate.transition
        license_result = exp5470.validate_license_transition(transition)
        semantic_result = exp5470.validate_semantic_constraints(transition)
        distortion_result = validate_distortion_guard(transition)
        guard_results = {
            "license_transition_guard": _license_guard_result(license_result),
            "semantic_graph_guard": _semantic_graph_guard_result(transition, semantic_result),
            "distortion_guard": _distortion_guard_result(distortion_result),
        }
        caught_by = [guard_id for guard_id in GUARD_IDS if guard_results[guard_id]["caught"]]
        semantic_receipt = semantic_graph_receipt(transition, semantic_result)
        minimal_core = minimal_core_feedback(transition.candidate_id, guard_results)
        exact = exact_final_validator(candidate, guard_results)
        row = {
            "candidate_id": transition.candidate_id,
            "case_type": transition.case_type,
            "domain": transition.domain,
            "description": transition.description,
            "expected_accept": transition.expected_accept,
            "expected_guard_ids": list(candidate.expected_guard_ids),
            "source_state_id": transition.source_state.state_id,
            "target_state_id": transition.target_state.state_id,
            "local_syntax_valid": transition.local_syntax_valid,
            "lcd_advisory_accept": transition.lcd_advisory_accept,
            "repair_proposal": {
                "proposal_score": float(candidate.repair_proposal_score),
                "score_purpose": "repair_ranking_only",
                "selected_repair_action": candidate.repair_action_hint,
                "used_for_guard_success": False,
            },
            "guard_results": guard_results,
            "caught_by_guards": caught_by,
            "minimal_core_feedback": minimal_core,
            "semantic_graph_receipt": semantic_receipt,
            "exact_final_verdict": exact,
        }
        row["row_checksum"] = row_checksum(row)
        rows.append(row)
    return rows


def validate_distortion_guard(candidate: exp5470.TransitionCandidate) -> JsonDict:
    """Detect fact rewrites against problem-fact licenses in any domain."""

    licenses = {license_row.license_id: license_row for license_row in candidate.licenses}
    distorted_keys: list[str] = []
    for key, target_atom in candidate.target_state.facts.items():
        license_row = licenses.get(target_atom.license_id)
        if (
            license_row is not None
            and license_row.license_type == "problem_fact"
            and license_row.subject == key
            and license_row.value != target_atom.value
        ):
            distorted_keys.append(key)
    return {
        "authority": "exact_problem_fact_distortion_guard",
        "distorted_fact_keys": sorted(distorted_keys),
    }


def semantic_graph_receipt(
    candidate: exp5470.TransitionCandidate,
    semantic_result: Mapping[str, Any],
) -> JsonDict:
    """Build semantic graph nodes separately from repair-core IDs."""

    node_ids = {f"sem:{candidate.candidate_id}:domain:{candidate.domain}"}
    edge_ids: set[str] = set()
    atoms_by_domain = _mapping(semantic_result.get("answer_set_atoms"))
    for domain, atom_payload in atoms_by_domain.items():
        atom_map = _mapping(atom_payload)
        for bucket in ("required", "present", "missing_atoms", "forbidden_atoms"):
            for atom in atom_map.get(bucket, []):
                node_ids.add(f"sem:{candidate.candidate_id}:{domain}:{bucket}:{atom}")
                edge_ids.add(f"edge:{candidate.candidate_id}:{domain}:{bucket}")
    for atom in semantic_result.get("missing_atoms", []):
        node_ids.add(f"sem:{candidate.candidate_id}:missing:{atom}")
        edge_ids.add(f"edge:{candidate.candidate_id}:missing")
    for atom in semantic_result.get("forbidden_atoms", []):
        node_ids.add(f"sem:{candidate.candidate_id}:forbidden:{atom}")
        edge_ids.add(f"edge:{candidate.candidate_id}:forbidden")
    return {
        "authority": "exact_answer_set_semantic_validator",
        "node_ids": sorted(node_ids),
        "edge_ids": sorted(edge_ids),
        "node_count": len(node_ids),
    }


def minimal_core_feedback(
    candidate_id: str,
    guard_results: Mapping[str, Mapping[str, Any]],
) -> JsonDict:
    """Return stable minimal core IDs for each unsatisfied guard constraint."""

    core_ids: list[str] = []
    constraint_ids: list[str] = []
    guard_to_core_ids: dict[str, list[str]] = {}
    for guard_id in GUARD_IDS:
        result = guard_results[guard_id]
        if result.get("caught") is not True:
            continue
        guard_core_ids: list[str] = []
        for violation in result.get("violation_kinds", []):
            violation_id = str(violation)
            core_id = f"core:{candidate_id}:{guard_id}:{violation_id}"
            constraint_id = f"constraint:{guard_id}:{violation_id}"
            core_ids.append(core_id)
            constraint_ids.append(constraint_id)
            guard_core_ids.append(core_id)
        guard_to_core_ids[guard_id] = guard_core_ids
    return {
        "authority": "deterministic_minimal_core_feedback",
        "minimal_core_ids": sorted(core_ids),
        "unsatisfied_constraint_ids": sorted(set(constraint_ids)),
        "guard_to_core_ids": {
            guard_id: sorted(ids) for guard_id, ids in sorted(guard_to_core_ids.items())
        },
        "generated_from": "exact_guard_violations_not_repair_score",
    }


def exact_final_validator(
    candidate: GuardScaleCandidate,
    guard_results: Mapping[str, Mapping[str, Any]],
) -> JsonDict:
    """Combine exact guard results without consulting the repair proposal score."""

    caught_by = [guard_id for guard_id in GUARD_IDS if guard_results[guard_id]["caught"]]
    violation_kinds: list[str] = []
    for guard_id in caught_by:
        violation_kinds.extend(str(item) for item in guard_results[guard_id]["violation_kinds"])
    accepted = not caught_by
    expected_guards = tuple(candidate.expected_guard_ids)
    matches_expected = (
        accepted == candidate.transition.expected_accept
        and tuple(caught_by) == expected_guards
    )
    return {
        "accepted": accepted,
        "caught_by_guards": caught_by,
        "violation_kinds": sorted(set(violation_kinds)),
        "expected_accept": candidate.transition.expected_accept,
        "expected_guard_ids": list(expected_guards),
        "matches_expected": matches_expected,
        "final_authority": EXACT_FINAL_AUTHORITY,
        "computed_from_repair_score": False,
    }


def derive_metrics(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Compute guard, overlap, and final-validator metrics from row evidence."""

    row_list = [row for row in rows if isinstance(row, Mapping)]
    invalid_rows = [row for row in row_list if row.get("expected_accept") is False]
    valid_rows = [row for row in row_list if row.get("expected_accept") is True]
    guard_catch_counts = {
        guard_id: sum(1 for row in row_list if _guard_caught(row, guard_id))
        for guard_id in GUARD_IDS
    }
    guard_catch_rates = {
        guard_id: _rate(count, len(invalid_rows))
        for guard_id, count in guard_catch_counts.items()
    }
    false_accepts = [
        row
        for row in invalid_rows
        if _exact_verdict(row).get("accepted") is True
    ]
    false_rejects = [
        row
        for row in valid_rows
        if _exact_verdict(row).get("accepted") is False
    ]
    exact_matches = [
        row
        for row in row_list
        if _exact_verdict(row).get("accepted") == row.get("expected_accept")
    ]
    core_ids = {
        str(core_id)
        for row in row_list
        for core_id in _mapping(row.get("minimal_core_feedback")).get(
            "minimal_core_ids", []
        )
    }
    semantic_node_ids = {
        str(node_id)
        for row in row_list
        for node_id in _mapping(row.get("semantic_graph_receipt")).get("node_ids", [])
    }
    return {
        "fixture_count": len(row_list),
        "minimal_core_count": len(core_ids),
        "semantic_graph_node_count": len(semantic_node_ids),
        "guard_catch_counts": guard_catch_counts,
        "guard_catch_rates": guard_catch_rates,
        "guard_overlap_matrix": _guard_overlap_matrix(row_list, len(invalid_rows)),
        "false_accept_rate": _rate(len(false_accepts), len(invalid_rows)),
        "false_reject_rate": _rate(len(false_rejects), len(valid_rows)),
        "exact_final_agreement": _rate(len(exact_matches), len(row_list)),
        "row_checksums_match": all(row.get("row_checksum") == row_checksum(row) for row in row_list),
        "single_guard_failure_count": sum(
            1
            for row in invalid_rows
            if len(row.get("caught_by_guards", [])) == 1
        ),
        "composed_guard_failure_count": sum(
            1
            for row in invalid_rows
            if len(row.get("caught_by_guards", [])) >= 2
        ),
        "guard_evidence_independent": all(_guard_evidence_independent(row) for row in row_list),
        "core_ids_separate_from_semantic_nodes": core_ids.isdisjoint(semantic_node_ids),
        "exact_authority_rows": sum(
            1
            for row in row_list
            if _exact_verdict(row).get("final_authority") == EXACT_FINAL_AUTHORITY
            and _exact_verdict(row).get("computed_from_repair_score") is False
        ),
    }


def build_artifact(
    *,
    tests_run: Sequence[str | Mapping[str, Any]] = (),
) -> JsonDict:
    """Build the terminal Exp5471 artifact from exact row evidence."""

    rows = evaluate_candidates(build_candidates())
    metrics = derive_metrics(rows)
    ready = bool(
        metrics["fixture_count"] > len(exp5470.build_candidates())
        and all(metrics["guard_catch_counts"][guard_id] > 0 for guard_id in GUARD_IDS)
        and metrics["single_guard_failure_count"] >= len(GUARD_IDS)
        and metrics["composed_guard_failure_count"] >= 2
        and metrics["false_accept_rate"] == 0.0
        and metrics["false_reject_rate"] == 0.0
        and metrics["exact_final_agreement"] == 1.0
        and metrics["row_checksums_match"] is True
        and metrics["guard_evidence_independent"] is True
        and metrics["core_ids_separate_from_semantic_nodes"] is True
        and metrics["exact_authority_rows"] == metrics["fixture_count"]
    )
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "task_id": TASK_ID,
        "milestone": MILESTONE,
        "run_date": RUN_DATE,
        "spec_refs": list(SPEC_REFS),
        "fixture_count": metrics["fixture_count"],
        "minimal_core_count": metrics["minimal_core_count"],
        "semantic_graph_node_count": metrics["semantic_graph_node_count"],
        "guard_overlap_matrix": metrics["guard_overlap_matrix"],
        "false_accept_rate": metrics["false_accept_rate"],
        "false_reject_rate": metrics["false_reject_rate"],
        "exact_final_agreement": metrics["exact_final_agreement"],
        "guard_composition_ready": ready,
        "guided_decoding_quarantine_lifted": False,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "random_seed": RANDOM_SEED,
        "honest_verdict": (
            "complete: deterministic guard composition ready; guided decoding remains quarantined"
            if ready
            else "blocked: deterministic guard composition checks failed"
        ),
        "guard_ids": list(GUARD_IDS),
        "guard_catch_counts": metrics["guard_catch_counts"],
        "guard_catch_rates": metrics["guard_catch_rates"],
        "single_guard_failure_count": metrics["single_guard_failure_count"],
        "composed_guard_failure_count": metrics["composed_guard_failure_count"],
        "row_results": rows,
        "row_provenance_checksum": row_provenance_checksum(rows),
        "metric_details": metrics,
        "field_principles": dict(FIELD_PRINCIPLES),
        "tests_run": _normalise_tests_run(tests_run),
        "research_conductor_modified": False,
    }
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    validate_artifact(artifact)
    return artifact


def run(
    *,
    result_path: Path | str = REPO_ROOT / RESULT_RELATIVE_PATH,
    tests_run: Sequence[str | Mapping[str, Any]] = (),
    write: bool = True,
) -> JsonDict:
    """Build and optionally write the Exp5471 deliverable JSON."""

    artifact = build_artifact(tests_run=tests_run)
    if write:
        output_path = Path(result_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(artifact, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
            encoding="utf-8",
        )
    return artifact


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """Raise when the artifact no longer supports the Exp5471 claim."""

    errors = artifact_schema_errors(artifact)
    if errors:
        raise ValueError("; ".join(errors))


def artifact_schema_errors(artifact: Mapping[str, Any]) -> list[str]:
    """Return schema, metric, authority, and row-integrity errors."""

    errors: list[str] = []
    missing = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact]
    if missing:
        errors.append(f"missing required fields: {missing}")
    rows_value = artifact.get("row_results")
    if not isinstance(rows_value, list):
        errors.append("row_results must be a list")
        rows: list[Mapping[str, Any]] = []
    else:
        rows = [row for row in rows_value if isinstance(row, Mapping)]
    metrics = derive_metrics(rows)
    for field_name in (
        "fixture_count",
        "minimal_core_count",
        "semantic_graph_node_count",
        "guard_overlap_matrix",
        "false_accept_rate",
        "false_reject_rate",
        "exact_final_agreement",
    ):
        if artifact.get(field_name) != metrics[field_name]:
            errors.append(f"{field_name} must match row recomputation")
    for field_name in ("guard_catch_counts", "guard_catch_rates"):
        if artifact.get(field_name) != metrics[field_name]:
            errors.append(f"{field_name} must match row recomputation")
    if artifact.get("field_principles") != FIELD_PRINCIPLES:
        errors.append("field_principles mismatch")
    if artifact.get("guided_decoding_quarantine_lifted") is not False:
        errors.append("guided decoding quarantine must remain lifted=false")
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        errors.append("inference_substrate mismatch")
    if artifact.get("random_seed") != RANDOM_SEED:
        errors.append("random_seed mismatch")
    verdict = artifact.get("honest_verdict")
    if not isinstance(verdict, str) or "\n" in verdict or not verdict.startswith(TERMINAL_PREFIXES):
        errors.append("honest_verdict must start with complete: or blocked:")
    if artifact.get("research_conductor_modified") is not False:
        errors.append("scripts/research_conductor.py must not be modified")
    if artifact.get("row_provenance_checksum") != row_provenance_checksum(rows):
        errors.append("row_provenance_checksum mismatch")
    if artifact.get("reproducibility_checksum") != reproducibility_checksum(artifact):
        errors.append("reproducibility_checksum mismatch")
    errors.extend(_row_integrity_errors(rows))

    ready = artifact.get("guard_composition_ready")
    if type(ready) is not bool:
        errors.append("guard_composition_ready must be boolean")
    if ready is True:
        if metrics["fixture_count"] <= len(exp5470.build_candidates()):
            errors.append("guard_composition_ready requires scaled fixture_count")
        for guard_id in GUARD_IDS:
            if metrics["guard_catch_counts"][guard_id] <= 0:
                errors.append(f"guard_composition_ready requires {guard_id} catches")
        if metrics["single_guard_failure_count"] < len(GUARD_IDS):
            errors.append("guard_composition_ready requires single-guard failures")
        if metrics["composed_guard_failure_count"] < 2:
            errors.append("guard_composition_ready requires composed-guard failures")
        if metrics["false_accept_rate"] != 0.0:
            errors.append("guard_composition_ready requires false_accept_rate=0.0")
        if metrics["false_reject_rate"] != 0.0:
            errors.append("guard_composition_ready requires false_reject_rate=0.0")
        if metrics["exact_final_agreement"] != 1.0:
            errors.append("guard_composition_ready requires exact_final_agreement=1.0")
        if metrics["row_checksums_match"] is not True:
            errors.append("guard_composition_ready requires valid row checksums")
        if metrics["guard_evidence_independent"] is not True:
            errors.append("guard success must not use repair proposal score")
        if metrics["core_ids_separate_from_semantic_nodes"] is not True:
            errors.append("minimal core IDs must be separate from semantic graph node IDs")
        if metrics["exact_authority_rows"] != metrics["fixture_count"]:
            errors.append("guard_composition_ready requires exact final authority on every row")
    return errors


def row_checksum(row: Mapping[str, Any]) -> str:
    """Hash one evaluated row while excluding its own checksum."""

    payload = {key: value for key, value in row.items() if key != "row_checksum"}
    return _sha256_json(payload)


def row_provenance_checksum(rows: Sequence[Mapping[str, Any]]) -> str:
    """Hash stable row IDs, guard catches, exact verdicts, and row checksums."""

    payload = [
        {
            "candidate_id": row.get("candidate_id"),
            "caught_by_guards": row.get("caught_by_guards"),
            "minimal_core_ids": _mapping(row.get("minimal_core_feedback")).get(
                "minimal_core_ids", []
            ),
            "semantic_node_ids": _mapping(row.get("semantic_graph_receipt")).get(
                "node_ids", []
            ),
            "exact_final_verdict": row.get("exact_final_verdict"),
            "row_checksum": row.get("row_checksum"),
        }
        for row in rows
    ]
    return _sha256_json(payload)


def reproducibility_checksum(artifact: Mapping[str, Any]) -> str:
    """Hash the artifact payload without the self-referential checksum."""

    payload = {key: value for key, value in artifact.items() if key != "reproducibility_checksum"}
    return _sha256_json(payload)


def _scaleup_candidates() -> list[GuardScaleCandidate]:
    stock_fact = exp5470.LicenseRecord(
        "PF:bolt-stock-3",
        "problem_fact",
        "inventory.bolt.stock",
        3,
        "WIT:stock-row-17",
    )
    order_locked = exp5470.LicenseRecord(
        "SF:order-o17-locked",
        "state_fact",
        "order.locked",
        True,
        "WIT:order-o17",
    )
    order_unlocked = exp5470.LicenseRecord(
        "SF:order-o21-unlocked",
        "state_fact",
        "order.locked",
        False,
        "WIT:order-o21",
    )
    cancel_precondition = exp5470.LicenseRecord(
        "WIT:cancel-precondition-5471",
        "witness_row",
        "api.cancel_order.precondition",
        {"locked": False, "requires_override_token": False},
        "WIT:cancel-precondition-5471",
    )
    budget_fact = exp5470.LicenseRecord(
        "PF:budget-cap-2400-5471",
        "problem_fact",
        "budget.max_usd",
        2400,
        "WIT:budget-row-5471",
    )
    return [
        GuardScaleCandidate(
            transition=exp5470.TransitionCandidate(
                candidate_id="5471-valid-api-unlocked",
                case_type="valid_api_rewrite",
                domain="api_preconditions",
                description="Valid unlocked-order cancellation row in the scaled fixture.",
                source_state=exp5470.TypedState(
                    state_id="source-5471-valid-api",
                    facts={"order.locked": exp5470.StateAtom(False, "SF:order-o21-unlocked")},
                ),
                target_state=exp5470.TypedState(
                    state_id="target-5471-valid-api",
                    facts={"order.locked": exp5470.StateAtom(False, "SF:order-o21-unlocked")},
                    api_call={"name": "cancel_order", "args": {"order_id": "O-21"}},
                ),
                licenses=(order_unlocked, cancel_precondition),
                local_syntax_valid=True,
                lcd_advisory_accept=True,
                expected_accept=True,
                expected_violation_kinds=(),
            ),
            expected_guard_ids=(),
            repair_proposal_score=0.044,
            repair_action_hint="none",
        ),
        GuardScaleCandidate(
            transition=exp5470.TransitionCandidate(
                candidate_id="5471-json-api-composed",
                case_type="composed_license_semantic",
                domain="json_constraints",
                description="Adds an unlicensed premise and exceeds stock in one JSON row.",
                source_state=exp5470.TypedState(
                    state_id="source-5471-json-composed",
                    facts={
                        "inventory.bolt.stock": exp5470.StateAtom(3, "PF:bolt-stock-3")
                    },
                ),
                target_state=exp5470.TypedState(
                    state_id="target-5471-json-composed",
                    facts={
                        "inventory.bolt.stock": exp5470.StateAtom(3, "PF:bolt-stock-3"),
                        "clinic.has_backup_generator": exp5470.StateAtom(
                            True, "UNLICENSED:backup-generator-5471"
                        ),
                    },
                    json_payload={"sku": "bolt", "quantity": 7},
                ),
                licenses=(stock_fact,),
                local_syntax_valid=True,
                lcd_advisory_accept=True,
                expected_accept=False,
                expected_violation_kinds=("hidden_premise", "semantic_invalid"),
            ),
            expected_guard_ids=("license_transition_guard", "semantic_graph_guard"),
            repair_proposal_score=0.992,
            repair_action_hint="repair_json_and_remove_hidden_fact",
        ),
        GuardScaleCandidate(
            transition=exp5470.TransitionCandidate(
                candidate_id="5471-distortion-api-composed",
                case_type="composed_semantic_distortion",
                domain="api_preconditions",
                description="Violates API preconditions while distorting a budget fact.",
                source_state=exp5470.TypedState(
                    state_id="source-5471-api-distortion",
                    facts={
                        "order.locked": exp5470.StateAtom(True, "SF:order-o17-locked"),
                        "budget.max_usd": exp5470.StateAtom(
                            2400, "PF:budget-cap-2400-5471"
                        ),
                    },
                ),
                target_state=exp5470.TypedState(
                    state_id="target-5471-api-distortion",
                    facts={
                        "order.locked": exp5470.StateAtom(True, "SF:order-o17-locked"),
                        "budget.max_usd": exp5470.StateAtom(
                            2500, "PF:budget-cap-2400-5471"
                        ),
                    },
                    api_call={"name": "cancel_order", "args": {"order_id": "O-17"}},
                ),
                licenses=(order_locked, cancel_precondition, budget_fact),
                local_syntax_valid=True,
                lcd_advisory_accept=True,
                expected_accept=False,
                expected_violation_kinds=("api_precondition_failed", "factual_distortion"),
            ),
            expected_guard_ids=("semantic_graph_guard", "distortion_guard"),
            repair_proposal_score=0.881,
            repair_action_hint="unlock_order_and_restore_budget",
        ),
        GuardScaleCandidate(
            transition=exp5470.TransitionCandidate(
                candidate_id="5471-license-distortion-composed",
                case_type="composed_license_distortion",
                domain="fact_anchor",
                description="Distorts a budget fact and adds an unlicensed premise.",
                source_state=exp5470.TypedState(
                    state_id="source-5471-license-distortion",
                    facts={
                        "budget.max_usd": exp5470.StateAtom(
                            2400, "PF:budget-cap-2400-5471"
                        )
                    },
                ),
                target_state=exp5470.TypedState(
                    state_id="target-5471-license-distortion",
                    facts={
                        "budget.max_usd": exp5470.StateAtom(
                            2600, "PF:budget-cap-2400-5471"
                        ),
                        "budget.has_exception": exp5470.StateAtom(
                            True, "UNLICENSED:budget-exception"
                        ),
                    },
                ),
                licenses=(budget_fact,),
                local_syntax_valid=True,
                lcd_advisory_accept=True,
                expected_accept=False,
                expected_violation_kinds=("hidden_premise", "factual_distortion"),
            ),
            expected_guard_ids=("license_transition_guard", "distortion_guard"),
            repair_proposal_score=0.927,
            repair_action_hint="remove_exception_and_restore_budget",
        ),
        GuardScaleCandidate(
            transition=exp5470.TransitionCandidate(
                candidate_id="5471-all-guards-composed",
                case_type="composed_all_guards",
                domain="api_preconditions",
                description="Combines hidden premise, API precondition failure, and fact distortion.",
                source_state=exp5470.TypedState(
                    state_id="source-5471-all",
                    facts={
                        "order.locked": exp5470.StateAtom(True, "SF:order-o17-locked"),
                        "budget.max_usd": exp5470.StateAtom(
                            2400, "PF:budget-cap-2400-5471"
                        ),
                    },
                ),
                target_state=exp5470.TypedState(
                    state_id="target-5471-all",
                    facts={
                        "order.locked": exp5470.StateAtom(True, "SF:order-o17-locked"),
                        "budget.max_usd": exp5470.StateAtom(
                            2700, "PF:budget-cap-2400-5471"
                        ),
                        "clinic.has_backup_generator": exp5470.StateAtom(
                            True, "UNLICENSED:backup-generator-all"
                        ),
                    },
                    api_call={"name": "cancel_order", "args": {"order_id": "O-17"}},
                ),
                licenses=(order_locked, cancel_precondition, budget_fact),
                local_syntax_valid=True,
                lcd_advisory_accept=True,
                expected_accept=False,
                expected_violation_kinds=(
                    "hidden_premise",
                    "api_precondition_failed",
                    "factual_distortion",
                ),
            ),
            expected_guard_ids=GUARD_IDS,
            repair_proposal_score=0.997,
            repair_action_hint="compose_all_exact_repairs",
        ),
    ]


def _license_guard_result(license_result: Mapping[str, Any]) -> JsonDict:
    violation_kinds: list[str] = []
    if license_result.get("hidden_premise_keys"):
        violation_kinds.append("hidden_premise")
    if license_result.get("unlicensed_mutation_keys"):
        violation_kinds.append("unlicensed_mutation")
    if license_result.get("fabricated_citation_ids"):
        violation_kinds.append("fabricated_evidence")
    return {
        "authority": "exact_license_transition_validator",
        "evidence_source": "exact_license_transition_validator",
        "caught": bool(violation_kinds),
        "violation_kinds": violation_kinds,
        "details": dict(license_result),
    }


def _semantic_graph_guard_result(
    candidate: exp5470.TransitionCandidate,
    semantic_result: Mapping[str, Any],
) -> JsonDict:
    violation_kinds = _semantic_graph_violation_kinds(candidate, semantic_result)
    return {
        "authority": "exact_answer_set_semantic_validator",
        "evidence_source": "exact_answer_set_semantic_validator",
        "caught": bool(violation_kinds),
        "violation_kinds": violation_kinds,
        "details": dict(semantic_result),
    }


def _distortion_guard_result(distortion_result: Mapping[str, Any]) -> JsonDict:
    violation_kinds = ["factual_distortion"] if distortion_result.get("distorted_fact_keys") else []
    return {
        "authority": "exact_problem_fact_distortion_guard",
        "evidence_source": "exact_problem_fact_distortion_guard",
        "caught": bool(violation_kinds),
        "violation_kinds": violation_kinds,
        "details": dict(distortion_result),
    }


def _semantic_graph_violation_kinds(
    candidate: exp5470.TransitionCandidate,
    semantic_result: Mapping[str, Any],
) -> list[str]:
    missing = set(str(item) for item in semantic_result.get("missing_atoms", []))
    forbidden = set(str(item) for item in semantic_result.get("forbidden_atoms", []))
    if not missing and not forbidden:
        return []
    if candidate.domain == "api_preconditions" and "api_preconditions_met" in missing:
        return ["api_precondition_failed"]
    if candidate.domain == "fact_anchor" and (
        "fact_anchor_supported" in missing
        or any(atom.startswith("fact_supported:") for atom in missing)
    ):
        return []
    return ["semantic_invalid"]


def _guard_overlap_matrix(
    rows: Sequence[Mapping[str, Any]],
    invalid_count: int,
) -> dict[str, dict[str, JsonDict]]:
    matrix: dict[str, dict[str, JsonDict]] = {}
    for left in GUARD_IDS:
        matrix[left] = {}
        for right in GUARD_IDS:
            count = sum(
                1
                for row in rows
                if _guard_caught(row, left) and _guard_caught(row, right)
            )
            matrix[left][right] = {
                "count": count,
                "rate": _rate(count, invalid_count),
            }
    return matrix


def _row_integrity_errors(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    errors: list[str] = []
    for row in rows:
        row_id = row.get("candidate_id")
        guard_results = row.get("guard_results")
        if not isinstance(guard_results, Mapping):
            errors.append(f"{row_id} guard_results must be a mapping")
            continue
        caught_from_results = [
            guard_id for guard_id in GUARD_IDS if _mapping(guard_results.get(guard_id)).get("caught")
        ]
        if row.get("caught_by_guards") != caught_from_results:
            errors.append(f"{row_id} caught_by_guards must match exact guard results")
        for guard_id in GUARD_IDS:
            result = _mapping(guard_results.get(guard_id))
            if result.get("evidence_source") == "repair_proposal_score":
                errors.append(f"{row_id} guard success must not use repair proposal score")
            if "caught" not in result:
                errors.append(f"{row_id} {guard_id} missing caught field")
        verdict = _exact_verdict(row)
        if verdict.get("final_authority") != EXACT_FINAL_AUTHORITY:
            errors.append(f"{row_id} final authority must be exact guard validators")
        if verdict.get("computed_from_repair_score") is not False:
            errors.append(f"{row_id} final verdict must not use repair proposal score")
        if verdict.get("accepted") != row.get("expected_accept"):
            errors.append(f"{row_id} exact validator did not match expected label")
        if _core_node_overlap(row):
            errors.append(f"{row_id} minimal core IDs must be separate from semantic graph node IDs")
        if row.get("row_checksum") != row_checksum(row):
            errors.append(f"{row_id} row checksum mismatch")
    return errors


def _guard_caught(row: Mapping[str, Any], guard_id: str) -> bool:
    return _mapping(_mapping(row.get("guard_results")).get(guard_id)).get("caught") is True


def _guard_evidence_independent(row: Mapping[str, Any]) -> bool:
    guard_results = _mapping(row.get("guard_results"))
    return all(
        _mapping(guard_results.get(guard_id)).get("evidence_source")
        != "repair_proposal_score"
        for guard_id in GUARD_IDS
    )


def _core_node_overlap(row: Mapping[str, Any]) -> set[str]:
    core_ids = set(_mapping(row.get("minimal_core_feedback")).get("minimal_core_ids", []))
    node_ids = set(_mapping(row.get("semantic_graph_receipt")).get("node_ids", []))
    return core_ids.intersection(node_ids)


def _exact_verdict(row: Mapping[str, Any]) -> Mapping[str, Any]:
    return _mapping(row.get("exact_final_verdict"))


def _mapping(value: Any) -> JsonDict:
    return dict(value) if isinstance(value, Mapping) else {}


def _rate(numerator: int, denominator: int) -> float:
    return round(numerator / denominator, 6) if denominator else 0.0


def _normalise_tests_run(tests_run: Sequence[str | Mapping[str, Any]]) -> list[JsonDict]:
    rows: list[JsonDict] = []
    for row in tests_run:
        if isinstance(row, Mapping):
            rows.append(
                {
                    "command": str(row.get("command", "")),
                    "outcome": str(row.get("outcome", "recorded")),
                }
            )
        else:
            rows.append({"command": str(row), "outcome": "recorded"})
    return rows or [{"command": "not_recorded", "outcome": "not_recorded"}]


def _sha256_json(payload: Any) -> str:
    data = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-path", type=Path, default=REPO_ROOT / RESULT_RELATIVE_PATH)
    args = parser.parse_args(argv)
    artifact = run(result_path=args.result_path, write=True)
    print(json.dumps(artifact, indent=2, sort_keys=True, ensure_ascii=True))
    return 0 if artifact["guard_composition_ready"] else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
