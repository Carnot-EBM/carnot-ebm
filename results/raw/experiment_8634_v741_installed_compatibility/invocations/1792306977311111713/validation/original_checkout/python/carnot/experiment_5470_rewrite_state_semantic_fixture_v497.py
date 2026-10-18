#!/usr/bin/env python3
"""Exp5470 deterministic rewrite-state and semantic-constraint fixture.

Spec refs: REQ-SAFE-5470, SCENARIO-SAFE-5470.

The fixture is intentionally small and exact.  Each row is a typed transition
from a source state to a target state, and every fact, computation, citation,
or witness row must carry a license ID.  This makes the local validator the
final authority: a locally parseable or LCD-friendly rewrite is still rejected
when it adds a hidden premise, changes state without a transition license,
fabricates evidence, violates a semantic constraint, or distorts an anchored
fact.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
from typing import Any


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_RELATIVE_PATH = Path(
    "results/experiment_5470_rewrite_state_semantic_fixture_v497.json"
)
EXPERIMENT_ID = "experiment_5470_rewrite_state_semantic_fixture_v497"
TASK_ID = "exp5470-v497-rewrite-state-semantic-fixture"
MILESTONE = "2026.07.497"
RUN_DATE = "2026-07-09"
SCHEMA = "carnot.experiment_5470.rewrite_state_semantic_fixture.v497"
SPEC_REFS = ("REQ-SAFE-5470", "SCENARIO-SAFE-5470")
RANDOM_SEED = 5470
INFERENCE_SUBSTRATE = "deterministic_fixture_no_llm"
TERMINAL_PREFIXES = ("complete:", "blocked:")

REQUIRED_CASE_TYPES = (
    "valid_rewrite",
    "arithmetic_valid",
    "hidden_premise_mutation",
    "unlicensed_state_change",
    "fabricated_evidence",
    "semantic_json_invalid",
    "api_precondition_invalid",
    "factual_distortion",
)
REQUIRED_DOMAINS = (
    "fact_anchor",
    "arithmetic",
    "json_constraints",
    "api_preconditions",
)

FIELD_PRINCIPLES: dict[str, str] = {
    "fixture_count": "bounded fixture coverage.",
    "transition_count": "typed source-to-target state transition denominator.",
    "hidden_premise_catch_rate": "hidden premises rejected by license validation.",
    "unlicensed_mutation_catch_rate": (
        "unlicensed state changes rejected before semantic scoring."
    ),
    "semantic_false_accept_rate": (
        "exact-validator false accepts on locally syntax-valid semantic traps."
    ),
    "factual_distortion_rate": "factual-distortion false accepts after exact validation.",
    "lcd_bias_probe_passed": (
        "LCD-style advisory accepts are exposed and rejected by exact validators."
    ),
    "exact_validator_agreement": (
        "exact validators agree with curated expected transition labels."
    ),
    "rewrite_state_fixture_ready": "downstream deterministic fixture gate.",
    "guided_decoding_quarantine_lifted": "guided-decoding quarantine remains closed.",
    "inference_substrate": "deterministic fixture with no LLM inference.",
    "random_seed": "deterministic fixture seed.",
    "honest_verdict": 'terminal status; start with "complete:" or "blocked:".',
}
REQUIRED_ARTIFACT_FIELDS = tuple(FIELD_PRINCIPLES)


@dataclass(frozen=True)
class LicenseRecord:
    """A local authority that allows one typed atom, citation, or witness row."""

    license_id: str
    license_type: str
    subject: str
    value: Any
    witness_row_id: str = ""


@dataclass(frozen=True)
class StateAtom:
    """One typed state value plus the license ID claimed for that value."""

    value: Any
    license_id: str


@dataclass(frozen=True)
class CitationRef:
    """A citation claim that must resolve to a licensed witness row."""

    citation_id: str
    license_id: str
    witness_row_id: str


@dataclass(frozen=True)
class TypedState:
    """The verifier-visible state before or after a rewrite."""

    state_id: str
    facts: dict[str, StateAtom] = field(default_factory=dict)
    citations: tuple[CitationRef, ...] = ()
    computations: dict[str, StateAtom] = field(default_factory=dict)
    json_payload: Mapping[str, Any] | None = None
    api_call: Mapping[str, Any] | None = None


@dataclass(frozen=True)
class TransitionCandidate:
    """One candidate rewrite represented as a typed state transition."""

    candidate_id: str
    case_type: str
    domain: str
    description: str
    source_state: TypedState
    target_state: TypedState
    licenses: tuple[LicenseRecord, ...]
    local_syntax_valid: bool
    lcd_advisory_accept: bool
    expected_accept: bool
    expected_violation_kinds: tuple[str, ...]


def build_candidates() -> list[TransitionCandidate]:
    """Return the bounded deterministic fixture used by Exp5470."""

    fact_opened = LicenseRecord(
        "PF:clinic-opened-2019",
        "problem_fact",
        "clinic.opened_year",
        2019,
        "WIT:riverton-opened",
    )
    cite_opened = LicenseRecord(
        "CITE:riverton-register",
        "citation",
        "clinic.opened_year",
        "WIT:riverton-opened",
        "WIT:riverton-opened",
    )
    witness_opened = LicenseRecord(
        "WIT:riverton-opened",
        "witness_row",
        "clinic.opened_year",
        "The Riverton clinic opened to patients in 2019.",
        "WIT:riverton-opened",
    )
    compute_sum = LicenseRecord(
        "COMP:add-2-3-5",
        "computation",
        "sum.total",
        {"left": 2, "right": 3, "result": 5},
    )
    stock_fact = LicenseRecord(
        "PF:bolt-stock-3",
        "problem_fact",
        "inventory.bolt.stock",
        3,
        "WIT:stock-row-17",
    )
    json_schema = LicenseRecord(
        "WIT:json-order-schema",
        "witness_row",
        "json.order.schema",
        {"required": ["sku", "quantity"], "allow_extra": False},
        "WIT:json-order-schema",
    )
    order_locked = LicenseRecord(
        "SF:order-o17-locked",
        "state_fact",
        "order.locked",
        True,
        "WIT:order-o17",
    )
    cancel_precondition = LicenseRecord(
        "WIT:cancel-precondition",
        "witness_row",
        "api.cancel_order.precondition",
        {"locked": False, "requires_override_token": False},
        "WIT:cancel-precondition",
    )
    budget_fact = LicenseRecord(
        "PF:budget-cap-2400",
        "problem_fact",
        "budget.max_usd",
        2400,
        "WIT:budget-row-4",
    )
    cite_budget = LicenseRecord(
        "CITE:budget-row-4",
        "citation",
        "budget.max_usd",
        "WIT:budget-row-4",
        "WIT:budget-row-4",
    )
    witness_budget = LicenseRecord(
        "WIT:budget-row-4",
        "witness_row",
        "budget.max_usd",
        "The approved maximum budget is 2400 USD.",
        "WIT:budget-row-4",
    )
    common_fact_licenses = (fact_opened, cite_opened, witness_opened)
    return [
        TransitionCandidate(
            candidate_id="5470-valid-fact-paraphrase",
            case_type="valid_rewrite",
            domain="fact_anchor",
            description="Preserves the opened-year fact and its witness row.",
            source_state=TypedState(
                state_id="source-valid-fact",
                facts={
                    "clinic.opened_year": StateAtom(2019, "PF:clinic-opened-2019")
                },
                citations=(CitationRef("riverton-register", "CITE:riverton-register", "WIT:riverton-opened"),),
            ),
            target_state=TypedState(
                state_id="target-valid-fact",
                facts={
                    "clinic.opened_year": StateAtom(2019, "PF:clinic-opened-2019")
                },
                citations=(CitationRef("riverton-register", "CITE:riverton-register", "WIT:riverton-opened"),),
            ),
            licenses=common_fact_licenses,
            local_syntax_valid=True,
            lcd_advisory_accept=True,
            expected_accept=True,
            expected_violation_kinds=(),
        ),
        TransitionCandidate(
            candidate_id="5470-valid-arithmetic",
            case_type="arithmetic_valid",
            domain="arithmetic",
            description="Computes 2 + 3 = 5 with an explicit computation license.",
            source_state=TypedState(
                state_id="source-valid-arithmetic",
                facts={
                    "sum.left": StateAtom(2, "COMP:add-2-3-5"),
                    "sum.right": StateAtom(3, "COMP:add-2-3-5"),
                },
            ),
            target_state=TypedState(
                state_id="target-valid-arithmetic",
                facts={
                    "sum.left": StateAtom(2, "COMP:add-2-3-5"),
                    "sum.right": StateAtom(3, "COMP:add-2-3-5"),
                },
                computations={"sum.total": StateAtom(5, "COMP:add-2-3-5")},
            ),
            licenses=(compute_sum,),
            local_syntax_valid=True,
            lcd_advisory_accept=True,
            expected_accept=True,
            expected_violation_kinds=(),
        ),
        TransitionCandidate(
            candidate_id="5470-hidden-premise",
            case_type="hidden_premise_mutation",
            domain="fact_anchor",
            description="Adds an unlicensed backup-generator premise.",
            source_state=TypedState(
                state_id="source-hidden-premise",
                facts={
                    "clinic.opened_year": StateAtom(2019, "PF:clinic-opened-2019")
                },
                citations=(CitationRef("riverton-register", "CITE:riverton-register", "WIT:riverton-opened"),),
            ),
            target_state=TypedState(
                state_id="target-hidden-premise",
                facts={
                    "clinic.opened_year": StateAtom(2019, "PF:clinic-opened-2019"),
                    "clinic.has_backup_generator": StateAtom(
                        True, "UNLICENSED:backup-generator"
                    ),
                },
                citations=(CitationRef("riverton-register", "CITE:riverton-register", "WIT:riverton-opened"),),
            ),
            licenses=common_fact_licenses,
            local_syntax_valid=True,
            lcd_advisory_accept=True,
            expected_accept=False,
            expected_violation_kinds=("hidden_premise",),
        ),
        TransitionCandidate(
            candidate_id="5470-unlicensed-state-change",
            case_type="unlicensed_state_change",
            domain="fact_anchor",
            description="Flips a locked order to unlocked without a transition license.",
            source_state=TypedState(
                state_id="source-unlicensed-state",
                facts={"order.locked": StateAtom(True, "SF:order-o17-locked")},
            ),
            target_state=TypedState(
                state_id="target-unlicensed-state",
                facts={"order.locked": StateAtom(False, "SF:order-o17-locked")},
            ),
            licenses=(order_locked,),
            local_syntax_valid=True,
            lcd_advisory_accept=True,
            expected_accept=False,
            expected_violation_kinds=("unlicensed_mutation",),
        ),
        TransitionCandidate(
            candidate_id="5470-fabricated-citation",
            case_type="fabricated_evidence",
            domain="fact_anchor",
            description="Keeps the fact but cites a non-existent witness row.",
            source_state=TypedState(
                state_id="source-fabricated-citation",
                facts={
                    "clinic.opened_year": StateAtom(2019, "PF:clinic-opened-2019")
                },
            ),
            target_state=TypedState(
                state_id="target-fabricated-citation",
                facts={
                    "clinic.opened_year": StateAtom(2019, "PF:clinic-opened-2019")
                },
                citations=(CitationRef("phantom-report", "CITE:phantom-report", "WIT:phantom-report"),),
            ),
            licenses=common_fact_licenses,
            local_syntax_valid=True,
            lcd_advisory_accept=True,
            expected_accept=False,
            expected_violation_kinds=("fabricated_evidence",),
        ),
        TransitionCandidate(
            candidate_id="5470-json-semantic-invalid",
            case_type="semantic_json_invalid",
            domain="json_constraints",
            description="JSON shape is valid but quantity exceeds licensed stock.",
            source_state=TypedState(
                state_id="source-json",
                facts={"inventory.bolt.stock": StateAtom(3, "PF:bolt-stock-3")},
            ),
            target_state=TypedState(
                state_id="target-json",
                facts={"inventory.bolt.stock": StateAtom(3, "PF:bolt-stock-3")},
                json_payload={"sku": "bolt", "quantity": 4},
            ),
            licenses=(stock_fact, json_schema),
            local_syntax_valid=True,
            lcd_advisory_accept=True,
            expected_accept=False,
            expected_violation_kinds=("semantic_invalid",),
        ),
        TransitionCandidate(
            candidate_id="5470-api-precondition-invalid",
            case_type="api_precondition_invalid",
            domain="api_preconditions",
            description="API call is well-formed but violates the locked-order precondition.",
            source_state=TypedState(
                state_id="source-api",
                facts={"order.locked": StateAtom(True, "SF:order-o17-locked")},
            ),
            target_state=TypedState(
                state_id="target-api",
                facts={"order.locked": StateAtom(True, "SF:order-o17-locked")},
                api_call={"name": "cancel_order", "args": {"order_id": "O-17"}},
            ),
            licenses=(order_locked, cancel_precondition),
            local_syntax_valid=True,
            lcd_advisory_accept=True,
            expected_accept=False,
            expected_violation_kinds=("api_precondition_failed",),
        ),
        TransitionCandidate(
            candidate_id="5470-factual-distortion",
            case_type="factual_distortion",
            domain="fact_anchor",
            description="Raises an anchored budget fact while reusing the old citation.",
            source_state=TypedState(
                state_id="source-distortion",
                facts={"budget.max_usd": StateAtom(2400, "PF:budget-cap-2400")},
                citations=(CitationRef("budget-row-4", "CITE:budget-row-4", "WIT:budget-row-4"),),
            ),
            target_state=TypedState(
                state_id="target-distortion",
                facts={"budget.max_usd": StateAtom(2500, "PF:budget-cap-2400")},
                citations=(CitationRef("budget-row-4", "CITE:budget-row-4", "WIT:budget-row-4"),),
            ),
            licenses=(budget_fact, cite_budget, witness_budget),
            local_syntax_valid=True,
            lcd_advisory_accept=True,
            expected_accept=False,
            expected_violation_kinds=("factual_distortion",),
        ),
    ]


def candidate_by_id(
    candidates: Sequence[TransitionCandidate], candidate_id: str
) -> TransitionCandidate:
    """Return one fixture candidate by stable ID."""

    return next(candidate for candidate in candidates if candidate.candidate_id == candidate_id)


def missing_or_bad_license_ids(candidate: TransitionCandidate) -> list[str]:
    """List license references that are absent or explicitly marked unlicensed."""

    known = {license_row.license_id for license_row in candidate.licenses}
    referenced: list[str] = []
    for state in (candidate.source_state, candidate.target_state):
        referenced.extend(atom.license_id for atom in state.facts.values())
        referenced.extend(atom.license_id for atom in state.computations.values())
        referenced.extend(citation.license_id for citation in state.citations)
    return sorted(
        {
            license_id
            for license_id in referenced
            if license_id.startswith("UNLICENSED:") or license_id not in known
        }
    )


def evaluate_candidates(candidates: Sequence[TransitionCandidate]) -> list[JsonDict]:
    """Evaluate each transition and attach exact validator evidence."""

    rows: list[JsonDict] = []
    for candidate in candidates:
        exact = exact_validate_candidate(candidate)
        row = _candidate_payload(candidate)
        row.update(
            {
                "license_result": exact["license_result"],
                "semantic_result": exact["semantic_result"],
                "answer_set_atoms": exact["answer_set_atoms"],
                "exact_final_verdict": {
                    "accepted": exact["accepted"],
                    "violation_kinds": exact["violation_kinds"],
                    "final_authority": "exact_rewrite_state_semantic_validators",
                    "expected_accept": candidate.expected_accept,
                    "expected_violation_kinds": list(candidate.expected_violation_kinds),
                    "matches_expected": exact["matches_expected"],
                },
            }
        )
        row["row_checksum"] = row_checksum(row)
        rows.append(row)
    return rows


def exact_validate_candidate(candidate: TransitionCandidate) -> JsonDict:
    """Run license and semantic validators, then combine their exact verdicts."""

    license_result = validate_license_transition(candidate)
    semantic_result = validate_semantic_constraints(candidate)
    violation_kinds: list[str] = []
    if license_result["hidden_premise_keys"]:
        violation_kinds.append("hidden_premise")
    if license_result["unlicensed_mutation_keys"]:
        violation_kinds.append("unlicensed_mutation")
    if license_result["fabricated_citation_ids"]:
        violation_kinds.append("fabricated_evidence")
    if semantic_result["missing_atoms"] or semantic_result["forbidden_atoms"]:
        if candidate.domain == "api_preconditions" and "api_preconditions_met" in semantic_result[
            "missing_atoms"
        ]:
            violation_kinds.append("api_precondition_failed")
        elif candidate.domain == "fact_anchor" and "fact_anchor_supported" in semantic_result[
            "missing_atoms"
        ]:
            violation_kinds.append("factual_distortion")
        else:
            violation_kinds.append("semantic_invalid")
    accepted = not violation_kinds
    expected_violations = set(candidate.expected_violation_kinds)
    return {
        "accepted": accepted,
        "violation_kinds": violation_kinds,
        "license_result": license_result,
        "semantic_result": semantic_result,
        "answer_set_atoms": semantic_result["answer_set_atoms"],
        "matches_expected": (
            accepted == candidate.expected_accept
            and set(violation_kinds) == expected_violations
        ),
    }


def validate_license_transition(candidate: TransitionCandidate) -> JsonDict:
    """Check license presence, hidden premises, state changes, and citations."""

    licenses = {license_row.license_id: license_row for license_row in candidate.licenses}
    hidden_premises: list[str] = []
    unlicensed_mutations: list[str] = []
    fabricated_citations: list[str] = []
    missing_ids = missing_or_bad_license_ids(candidate)

    for key, target_atom in candidate.target_state.facts.items():
        license_row = licenses.get(target_atom.license_id)
        source_atom = candidate.source_state.facts.get(key)
        if source_atom is None and license_row is None:
            hidden_premises.append(key)
        elif source_atom is None and license_row and license_row.license_type not in {
            "problem_fact",
            "computation",
        }:
            hidden_premises.append(key)
        elif (
            source_atom is not None
            and source_atom.value != target_atom.value
            and license_row is not None
            and license_row.license_type == "state_fact"
        ):
            transition_licensed = any(
                row.license_type == "state_transition"
                and row.subject == key
                and row.value == target_atom.value
                for row in candidate.licenses
            )
            if not transition_licensed:
                unlicensed_mutations.append(key)

    witness_ids = {
        row.witness_row_id
        for row in candidate.licenses
        if row.license_type == "witness_row" and row.witness_row_id
    }
    for citation in candidate.target_state.citations:
        citation_license = licenses.get(citation.license_id)
        if (
            citation_license is None
            or citation_license.license_type != "citation"
            or citation.witness_row_id not in witness_ids
        ):
            fabricated_citations.append(citation.license_id)

    return {
        "authority": "exact_license_transition_validator",
        "missing_or_bad_license_ids": missing_ids,
        "hidden_premise_keys": sorted(hidden_premises),
        "unlicensed_mutation_keys": sorted(unlicensed_mutations),
        "fabricated_citation_ids": sorted(fabricated_citations),
    }


def validate_semantic_constraints(candidate: TransitionCandidate) -> JsonDict:
    """Validate small-domain answer-set-like constraints."""

    atoms = answer_set_atoms(candidate)
    domain_atoms = atoms.get(candidate.domain)
    if domain_atoms is None:
        return {
            "authority": "exact_answer_set_semantic_validator",
            "accepted": False,
            "answer_set_atoms": atoms,
            "missing_atoms": ["semantic_domain_supported"],
            "forbidden_atoms": [],
            "failure_reasons": [f"unsupported_domain:{candidate.domain}"],
        }
    required = list(domain_atoms["required"])
    present = set(domain_atoms["present"])
    forbidden_present = list(domain_atoms["forbidden_present"])
    missing = [atom for atom in required if atom not in present]
    return {
        "authority": "exact_answer_set_semantic_validator",
        "accepted": not missing and not forbidden_present,
        "answer_set_atoms": atoms,
        "missing_atoms": missing,
        "forbidden_atoms": forbidden_present,
        "failure_reasons": [],
    }


def answer_set_atoms(candidate: TransitionCandidate) -> JsonDict:
    """Compute finite-domain required and present atoms for one candidate."""

    if candidate.domain == "fact_anchor":
        return {"fact_anchor": _fact_anchor_atoms(candidate)}
    if candidate.domain == "arithmetic":
        return {"arithmetic": _arithmetic_atoms(candidate)}
    if candidate.domain == "json_constraints":
        return {"json_constraints": _json_constraint_atoms(candidate)}
    if candidate.domain == "api_preconditions":
        return {"api_preconditions": _api_precondition_atoms(candidate)}
    return {}


def derive_metrics(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Recompute fixture metrics only from evaluated row evidence."""

    row_list = [dict(row) for row in rows if isinstance(row, Mapping)]
    hidden_expected = _rows_with_expected(row_list, "hidden_premise")
    unlicensed_expected = _rows_with_expected(row_list, "unlicensed_mutation")
    semantic_traps = [
        row
        for row in row_list
        if row.get("local_syntax_valid") is True
        and row.get("lcd_advisory_accept") is True
        and set(_expected_violations(row)).intersection(
            {"semantic_invalid", "api_precondition_failed"}
        )
    ]
    factual_traps = _rows_with_expected(row_list, "factual_distortion")
    lcd_semantic_rejections = [
        row
        for row in semantic_traps
        if _verdict(row).get("accepted") is False
    ]
    lcd_factual_rejections = [
        row
        for row in factual_traps
        if row.get("lcd_advisory_accept") is True
        and _verdict(row).get("accepted") is False
    ]
    exact_matches = [row for row in row_list if _verdict(row).get("matches_expected") is True]
    return {
        "fixture_count": len(row_list),
        "transition_count": len(row_list),
        "hidden_premise_catch_rate": _catch_rate(hidden_expected, "hidden_premise"),
        "unlicensed_mutation_catch_rate": _catch_rate(
            unlicensed_expected, "unlicensed_mutation"
        ),
        "semantic_false_accept_rate": _false_accept_rate(semantic_traps),
        "factual_distortion_rate": _false_accept_rate(factual_traps),
        "lcd_bias_probe_passed": bool(
            semantic_traps
            and factual_traps
            and lcd_semantic_rejections
            and lcd_factual_rejections
            and _false_accept_rate(semantic_traps) == 0.0
            and _false_accept_rate(factual_traps) == 0.0
        ),
        "exact_validator_agreement": _rate(len(exact_matches), len(row_list)),
        "row_checksums_match": all(row.get("row_checksum") == row_checksum(row) for row in row_list),
        "case_type_counts": _counts(row_list, "case_type"),
        "domain_counts": _counts(row_list, "domain"),
        "semantic_trap_count": len(semantic_traps),
        "factual_trap_count": len(factual_traps),
    }


def build_artifact(
    *,
    tests_run: Sequence[str | Mapping[str, Any]] = (),
) -> JsonDict:
    """Build the Exp5470 terminal artifact from deterministic row evidence."""

    rows = evaluate_candidates(build_candidates())
    metrics = derive_metrics(rows)
    ready = bool(
        metrics["fixture_count"] == len(REQUIRED_CASE_TYPES)
        and set(REQUIRED_CASE_TYPES).issubset(metrics["case_type_counts"])
        and set(REQUIRED_DOMAINS).issubset(metrics["domain_counts"])
        and metrics["hidden_premise_catch_rate"] == 1.0
        and metrics["unlicensed_mutation_catch_rate"] == 1.0
        and metrics["semantic_false_accept_rate"] == 0.0
        and metrics["factual_distortion_rate"] == 0.0
        and metrics["lcd_bias_probe_passed"] is True
        and metrics["exact_validator_agreement"] == 1.0
        and metrics["row_checksums_match"] is True
    )
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "task_id": TASK_ID,
        "milestone": MILESTONE,
        "run_date": RUN_DATE,
        "spec_refs": list(SPEC_REFS),
        "fixture_count": metrics["fixture_count"],
        "transition_count": metrics["transition_count"],
        "hidden_premise_catch_rate": metrics["hidden_premise_catch_rate"],
        "unlicensed_mutation_catch_rate": metrics["unlicensed_mutation_catch_rate"],
        "semantic_false_accept_rate": metrics["semantic_false_accept_rate"],
        "factual_distortion_rate": metrics["factual_distortion_rate"],
        "lcd_bias_probe_passed": metrics["lcd_bias_probe_passed"],
        "exact_validator_agreement": metrics["exact_validator_agreement"],
        "rewrite_state_fixture_ready": ready,
        "guided_decoding_quarantine_lifted": False,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "random_seed": RANDOM_SEED,
        "honest_verdict": (
            "complete: deterministic rewrite-state semantic fixture ready; guided decoding remains quarantined"
            if ready
            else "blocked: deterministic rewrite-state semantic fixture not ready"
        ),
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
    """Build and optionally write the Exp5470 deliverable JSON."""

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
    """Raise when the artifact no longer supports the Exp5470 fixture claim."""

    errors = artifact_schema_errors(artifact)
    if errors:
        raise ValueError("; ".join(errors))


def artifact_schema_errors(artifact: Mapping[str, Any]) -> list[str]:
    """Return schema, metric, quarantine, and row-integrity errors."""

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
        "transition_count",
        "hidden_premise_catch_rate",
        "unlicensed_mutation_catch_rate",
        "semantic_false_accept_rate",
        "factual_distortion_rate",
        "lcd_bias_probe_passed",
        "exact_validator_agreement",
    ):
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
    ready = artifact.get("rewrite_state_fixture_ready")
    if type(ready) is not bool:
        errors.append("rewrite_state_fixture_ready must be boolean")
    if ready is True:
        if metrics["fixture_count"] != len(REQUIRED_CASE_TYPES):
            errors.append("rewrite_state_fixture_ready requires all fixture cases")
        if not set(REQUIRED_CASE_TYPES).issubset(metrics["case_type_counts"]):
            errors.append("rewrite_state_fixture_ready requires all case types")
        if not set(REQUIRED_DOMAINS).issubset(metrics["domain_counts"]):
            errors.append("rewrite_state_fixture_ready requires all validator domains")
        if metrics["hidden_premise_catch_rate"] != 1.0:
            errors.append("rewrite_state_fixture_ready requires hidden_premise_catch_rate=1.0")
        if metrics["unlicensed_mutation_catch_rate"] != 1.0:
            errors.append(
                "rewrite_state_fixture_ready requires unlicensed_mutation_catch_rate=1.0"
            )
        if metrics["semantic_false_accept_rate"] != 0.0:
            errors.append("rewrite_state_fixture_ready requires semantic_false_accept_rate=0.0")
        if metrics["factual_distortion_rate"] != 0.0:
            errors.append("rewrite_state_fixture_ready requires factual_distortion_rate=0.0")
        if artifact.get("lcd_bias_probe_passed") is not True:
            errors.append("rewrite_state_fixture_ready requires top-level lcd_bias_probe_passed")
        if metrics["lcd_bias_probe_passed"] is not True:
            errors.append("rewrite_state_fixture_ready requires lcd_bias_probe_passed")
        if metrics["exact_validator_agreement"] != 1.0:
            errors.append("rewrite_state_fixture_ready requires exact_validator_agreement=1.0")
        if metrics["row_checksums_match"] is not True:
            errors.append("rewrite_state_fixture_ready requires valid row checksums")
    return errors


def row_checksum(row: Mapping[str, Any]) -> str:
    """Hash one evaluated row while excluding its own checksum."""

    payload = {key: value for key, value in row.items() if key != "row_checksum"}
    return _sha256_json(payload)


def row_provenance_checksum(rows: Sequence[Mapping[str, Any]]) -> str:
    """Hash stable row IDs, exact verdicts, and row checksums."""

    payload = [
        {
            "candidate_id": row.get("candidate_id"),
            "case_type": row.get("case_type"),
            "domain": row.get("domain"),
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


def _fact_anchor_atoms(candidate: TransitionCandidate) -> JsonDict:
    required: list[str] = []
    present: list[str] = []
    forbidden_present: list[str] = []
    licenses = {license_row.license_id: license_row for license_row in candidate.licenses}
    for key, atom in candidate.target_state.facts.items():
        license_row = licenses.get(atom.license_id)
        if license_row and license_row.license_type == "problem_fact":
            atom_name = f"fact_supported:{key}"
            required.append(atom_name)
            if license_row.subject == key and license_row.value == atom.value:
                present.append(atom_name)
            else:
                forbidden_present.append("fact_distorted")
    if forbidden_present:
        required.append("fact_anchor_supported")
    return {
        "required": required,
        "present": present,
        "forbidden_present": sorted(set(forbidden_present)),
    }


def _arithmetic_atoms(candidate: TransitionCandidate) -> JsonDict:
    required = ["arithmetic_result_matches"]
    computation = candidate.target_state.computations.get("sum.total")
    licenses = {license_row.license_id: license_row for license_row in candidate.licenses}
    present: list[str] = []
    if computation is not None:
        license_row = licenses.get(computation.license_id)
        left = candidate.target_state.facts.get("sum.left")
        right = candidate.target_state.facts.get("sum.right")
        if (
            license_row
            and license_row.license_type == "computation"
            and left is not None
            and right is not None
            and left.value + right.value == computation.value
            and license_row.value.get("result") == computation.value
        ):
            present.append("arithmetic_result_matches")
    return {"required": required, "present": present, "forbidden_present": []}


def _json_constraint_atoms(candidate: TransitionCandidate) -> JsonDict:
    payload = candidate.target_state.json_payload or {}
    stock = candidate.source_state.facts.get("inventory.bolt.stock")
    shape_valid = set(payload) == {"sku", "quantity"} and payload.get("sku") == "bolt"
    quantity = payload.get("quantity")
    present: list[str] = []
    if shape_valid and isinstance(quantity, int):
        present.append("json_shape_valid")
    if stock is not None and isinstance(quantity, int) and quantity <= stock.value:
        present.append("json_semantic_valid")
    return {
        "required": ["json_shape_valid", "json_semantic_valid"],
        "present": present,
        "forbidden_present": [],
    }


def _api_precondition_atoms(candidate: TransitionCandidate) -> JsonDict:
    call = candidate.target_state.api_call or {}
    locked = candidate.source_state.facts.get("order.locked")
    present: list[str] = []
    if call.get("name") == "cancel_order" and "order_id" in dict(call.get("args", {})):
        present.append("api_shape_valid")
    if locked is not None and locked.value is False:
        present.append("api_preconditions_met")
    return {
        "required": ["api_shape_valid", "api_preconditions_met"],
        "present": present,
        "forbidden_present": [],
    }


def _row_integrity_errors(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    errors: list[str] = []
    for row in rows:
        row_id = row.get("candidate_id")
        if row.get("case_type") not in REQUIRED_CASE_TYPES:
            errors.append(f"{row_id} case_type is unknown")
        if row.get("domain") not in REQUIRED_DOMAINS:
            errors.append(f"{row_id} domain is unknown")
        verdict = row.get("exact_final_verdict")
        if not isinstance(verdict, Mapping):
            errors.append(f"{row_id} exact_final_verdict must be a mapping")
            continue
        if verdict.get("final_authority") != "exact_rewrite_state_semantic_validators":
            errors.append(f"{row_id} final authority must be exact validators")
        if verdict.get("matches_expected") is not True:
            errors.append(f"{row_id} exact validator did not match expected label")
        if row.get("row_checksum") != row_checksum(row):
            errors.append(f"{row_id} row checksum mismatch")
    return errors


def _candidate_payload(candidate: TransitionCandidate) -> JsonDict:
    return {
        "candidate_id": candidate.candidate_id,
        "case_type": candidate.case_type,
        "domain": candidate.domain,
        "description": candidate.description,
        "source_state": _state_payload(candidate.source_state),
        "target_state": _state_payload(candidate.target_state),
        "licenses": [_license_payload(license_row) for license_row in candidate.licenses],
        "local_syntax_valid": candidate.local_syntax_valid,
        "lcd_advisory_accept": candidate.lcd_advisory_accept,
        "expected_accept": candidate.expected_accept,
        "expected_violation_kinds": list(candidate.expected_violation_kinds),
        "license_ids": [license_row.license_id for license_row in candidate.licenses],
    }


def _state_payload(state: TypedState) -> JsonDict:
    return {
        "state_id": state.state_id,
        "facts": {
            key: {"value": atom.value, "license_id": atom.license_id}
            for key, atom in sorted(state.facts.items())
        },
        "citations": [
            {
                "citation_id": citation.citation_id,
                "license_id": citation.license_id,
                "witness_row_id": citation.witness_row_id,
            }
            for citation in state.citations
        ],
        "computations": {
            key: {"value": atom.value, "license_id": atom.license_id}
            for key, atom in sorted(state.computations.items())
        },
        "json_payload": dict(state.json_payload) if state.json_payload is not None else None,
        "api_call": dict(state.api_call) if state.api_call is not None else None,
    }


def _license_payload(license_row: LicenseRecord) -> JsonDict:
    return {
        "license_id": license_row.license_id,
        "license_type": license_row.license_type,
        "subject": license_row.subject,
        "value": license_row.value,
        "witness_row_id": license_row.witness_row_id,
    }


def _rows_with_expected(rows: Sequence[Mapping[str, Any]], violation: str) -> list[Mapping[str, Any]]:
    return [row for row in rows if violation in _expected_violations(row)]


def _expected_violations(row: Mapping[str, Any]) -> list[str]:
    verdict = _verdict(row)
    value = verdict.get("expected_violation_kinds", row.get("expected_violation_kinds", []))
    return [str(item) for item in value if isinstance(item, str)]


def _catch_rate(rows: Sequence[Mapping[str, Any]], violation: str) -> float:
    caught = [
        row
        for row in rows
        if violation in _verdict(row).get("violation_kinds", [])
    ]
    return _rate(len(caught), len(rows))


def _false_accept_rate(rows: Sequence[Mapping[str, Any]]) -> float:
    false_accepts = [
        row for row in rows if _verdict(row).get("accepted") is True
    ]
    return _rate(len(false_accepts), len(rows)) if rows else 0.0


def _verdict(row: Mapping[str, Any]) -> Mapping[str, Any]:
    verdict = row.get("exact_final_verdict", {})
    return verdict if isinstance(verdict, Mapping) else {}


def _rate(numerator: int, denominator: int) -> float:
    return 0.0 if denominator == 0 else numerator / denominator


def _counts(rows: Sequence[Mapping[str, Any]], field_name: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        key = str(row.get(field_name))
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))


def _normalise_tests_run(tests_run: Sequence[str | Mapping[str, Any]]) -> list[JsonDict]:
    normalised: list[JsonDict] = []
    for item in tests_run:
        if isinstance(item, str):
            normalised.append({"command": item, "outcome": "recorded"})
        else:
            normalised.append(dict(item))
    return normalised


def _sha256_json(payload: Any) -> str:
    data = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def main(argv: list[str] | None = None) -> int:  # pragma: no cover
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-path", type=Path, default=REPO_ROOT / RESULT_RELATIVE_PATH)
    args = parser.parse_args(argv)
    artifact = run(result_path=args.result_path)
    print(json.dumps(artifact, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
