"""Build an exact error fixture and unused-pair slices for later experiments.

Spec refs: REQ-VERIFY-6967 and SCENARIO-VERIFY-6967-*.

This module does not repair a proposal or estimate model headroom. It reruns
the frozen parser and two exact authorities. It then freezes new exact pairs
whose IDs were absent from every prior model prompt.
"""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Mapping, Sequence
from copy import deepcopy
import hashlib
from itertools import combinations
import json
from pathlib import Path
import time
from typing import Any

from carnot import experiment_6955_reformulation_fixture as fixture_exp
from carnot import experiment_6956_three_family_reformulation_bank as bank_exp
from carnot import experiment_6957_smt_mapping_certification as certificate_exp


JsonDict = dict[str, Any]

EXPERIMENT_ID = 6967
EXPECTED_PROPOSAL_COUNT = 162
EXPECTED_FIXTURE_PAIR_COUNT = 120
RANDOM_SEED = 6_967_202_609_04
SCHEMA_VERSION = "carnot.exp6967.certified_error_headroom_fixture.v1"
INFERENCE_SUBSTRATE = "deterministic_z3_and_bounded_enumeration_reducer"
SPEC_PATH = Path("openspec/capabilities/verification/spec.md")
MODULE_PATH = Path("python/carnot/experiment_6967_certified_error_headroom_fixture.py")
TEST_PATH = Path("tests/python/test_experiment_6967_certified_error_headroom_fixture.py")
WRAPPER_PATH = Path("scripts/experiments/experiment_6967_certified_error_headroom_fixture.py")
OUTPUT_PATH = Path("results/experiment_6967_certified_error_headroom_fixture.json")
SOURCE_PATHS = {
    "fixture": Path("results/experiment_6955_reformulation_fixture.json"),
    "bank": Path("results/experiment_6956_three_family_reformulation_bank.json"),
    "certificate": Path("results/experiment_6957_smt_mapping_certification.json"),
    "selection": Path("results/experiment_6959_certified_energy_selection.json"),
    "fixture_checkpoint": Path(
        "results/checkpoints/experiment_6955_reformulation_fixture_corpus.json"
    ),
}
FAMILIES = tuple(fixture_exp.FAMILIES)
SLICE_POLICY = {
    "calibration": {"equivalent": 4, "non_equivalent": 2},
    "heldout": {"equivalent": 4, "non_equivalent": 2},
    "chronological": {"equivalent": 5, "non_equivalent": 3},
}
FORBIDDEN_PROMPT_FIELDS = {
    "canonical_relation",
    "certificate",
    "counterexample",
    "counterexamples",
    "difficulty",
    "enumeration_label",
    "exact_label",
    "expected_label",
    "hard_negative_edit",
    "label",
    "outcome",
    "sealed_label",
    "solver_status",
    "source_certificate_hash",
    "witness",
    "witnesses",
    "z3_label",
}

REQUIRED_ARTIFACT_FIELDS = (
    "field_principles",
    "preconditions_checked",
    "inference_substrate",
    "duration_s",
    "source_artifact_hashes",
    "rows",
    "recomputed_proposal_rows",
    "error_cluster_rows",
    "cluster_example_rows",
    "used_pair_exclusion_rows",
    "calibration_rows",
    "heldout_rows",
    "chronological_event_rows",
    "family_balance_rows",
    "exact_witness_rows",
    "solver_agreement_rows",
    "split_disjointness_rows",
    "prompt_visible_rows",
    "sealed_label_hashes",
    "split_hashes",
    "headroom_opportunity_rows",
    "error_fixture_ready_score",
    "chronological_event_stream_ready_score",
    "random_seed",
    "reproducibility_checksum",
    "gate_check_summary",
    "verifier_is_oracle",
    "verdict_class",
    "honest_verdict",
)

FIELD_PRINCIPLES = {
    "field_principles": "A reason per field makes the evidence contract auditable.",
    "preconditions_checked": "Fail-closed checks stop incomplete frozen inputs from producing slices.",
    "inference_substrate": "The fixed declaration separates exact reduction from model inference.",
    "duration_s": "Measured wall time shows that parser and solver work executed.",
    "source_artifact_hashes": "Content hashes bind the fixture to its frozen evidence and code.",
    "rows": "The complete proposal surface lets later audits ignore aggregates.",
    "recomputed_proposal_rows": "One raw-byte recomputation per attempt preserves the fixed denominator.",
    "error_cluster_rows": "Exclusive cluster counts expose each repair target without overlap.",
    "cluster_example_rows": "Stable examples connect every cluster to raw and exact evidence.",
    "used_pair_exclusion_rows": "Hashed used IDs prevent prior prompt pairs from entering new slices.",
    "calibration_rows": "A fixed calibration manifest prevents policy-driven sample selection.",
    "heldout_rows": "A separate held-out manifest supports later causal prompt evaluation.",
    "chronological_event_rows": "Ordered dependency records support later leakage-safe learning tests.",
    "family_balance_rows": "Per-family counts prevent one formulation family from dominating a slice.",
    "exact_witness_rows": "Separate exact evidence keeps solver labels outside prompt records.",
    "solver_agreement_rows": "Per-pair parity prevents one exact engine from certifying itself.",
    "split_disjointness_rows": "Pairwise intersections make exact ID leakage directly auditable.",
    "prompt_visible_rows": "Public-only records define what a later model may inspect.",
    "sealed_label_hashes": "Seals bind hidden labels without placing them in model-visible records.",
    "split_hashes": "Content-addressed manifests detect later membership or order drift.",
    "headroom_opportunity_rows": "Null claims distinguish diagnosed errors from tested repair gains.",
    "error_fixture_ready_score": "The bare gate opens only after complete recomputation and isolation.",
    "chronological_event_stream_ready_score": "The bare gate opens only for immutable ordered dependencies.",
    "random_seed": "One fixed seed makes selection and event order reproducible.",
    "reproducibility_checksum": "A timing-free digest detects any scientific-content change.",
    "gate_check_summary": "Expected and observed values make every blocked run actionable.",
    "verifier_is_oracle": "True states that exact engines construct this fixture's authority labels.",
    "verdict_class": "A closed class prevents exact fixture conformance from becoming model evidence.",
    "honest_verdict": "A stable prefix lets automation classify the terminal outcome.",
}


def canonical_json(value: Any) -> bytes:
    """Return stable UTF-8 JSON bytes for every scientific identity."""

    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def sha256_bytes(value: bytes) -> str:
    """Return the repository spelling of one SHA-256 content digest."""

    return "sha256:" + hashlib.sha256(value).hexdigest()


def sha256_path(path: Path) -> str | None:
    """Hash a file, while preserving a missing source as an explicit null."""

    return sha256_bytes(path.read_bytes()) if path.is_file() else None


def gate_check(check: str, expected: Any, observed: Any) -> JsonDict:
    """Record both sides of one exact precondition comparison."""

    return {
        "check": check,
        "expected_value": expected,
        "observed_value": observed,
        "passed": expected == observed,
    }


def gate_summary(checks: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    """Keep each failed check with the values needed to diagnose it."""

    return [
        {
            "failed_check": row.get("check"),
            "expected_value": row.get("expected_value"),
            "observed_value": row.get("observed_value"),
        }
        for row in checks
        if row.get("passed") is not True
    ]


def write_json_atomic(path: Path, value: Any) -> None:
    """Replace JSON atomically so readers never observe a partial fixture."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _read_object(path: Path) -> JsonDict:
    """Load one required JSON object through a strict type boundary."""

    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"json_object_required:{path}")
    return value


def load_inputs(repo_root: Path) -> JsonDict:
    """Load all frozen artifacts and index exact fixture pairs by unique ID."""

    inputs = {name: _read_object(repo_root / path) for name, path in SOURCE_PATHS.items()}
    pairs = inputs["fixture_checkpoint"].get("pairs", [])
    inputs["fixture_pairs"] = {
        str(row.get("pair_id")): deepcopy(row) for row in pairs if isinstance(row, Mapping)
    }
    return inputs


def _raw_roster_observation(bank: Mapping[str, Any]) -> JsonDict:
    """Recompute raw identities instead of trusting the bank's aggregate counts."""

    attempts = bank.get("attempt_rows", [])
    raw_rows = bank.get("raw_output_rows", [])
    if not isinstance(attempts, list) or not isinstance(raw_rows, list):
        return {"attempt_count": 0, "raw_count": 0, "unique": False, "hashes_match": False}
    raw_by_key = {str(row.get("attempt_key")): row for row in raw_rows if isinstance(row, Mapping)}
    keys = [str(row.get("attempt_key")) for row in attempts if isinstance(row, Mapping)]
    hashes_match = len(raw_by_key) == len(raw_rows)
    for attempt in attempts:
        if not isinstance(attempt, Mapping):
            hashes_match = False
            continue
        raw = raw_by_key.get(str(attempt.get("attempt_key")))
        text = str(attempt.get("raw_text", ""))
        expected_hash = bank_exp.sha256_text(text)
        if (
            raw is None
            or raw.get("raw_text") != attempt.get("raw_text")
            or raw.get("raw_sha256") != expected_hash
            or attempt.get("raw_sha256") != expected_hash
        ):
            hashes_match = False
    return {
        "attempt_count": len(attempts),
        "raw_count": len(raw_rows),
        "unique": len(keys) == len(set(keys)) == len(raw_by_key),
        "hashes_match": hashes_match,
    }


def check_preconditions(inputs: Mapping[str, Any]) -> list[JsonDict]:
    """Check every frozen input before raw output or exact labels are reduced."""

    fixture = inputs.get("fixture", {})
    bank = inputs.get("bank", {})
    certificate = inputs.get("certificate", {})
    selection = inputs.get("selection", {})
    pairs = inputs.get("fixture_pairs", {})
    fixture = fixture if isinstance(fixture, Mapping) else {}
    bank = bank if isinstance(bank, Mapping) else {}
    certificate = certificate if isinstance(certificate, Mapping) else {}
    selection = selection if isinstance(selection, Mapping) else {}
    pairs = pairs if isinstance(pairs, Mapping) else {}
    pair_ids = [str(row.get("pair_id")) for row in pairs.values() if isinstance(row, Mapping)]
    fixture_agreement = fixture.get("authority_agreement_rows", [])
    fixture_witnesses = fixture.get("feasibility_witness_rows", [])
    certificate_rows = certificate.get("proposal_rows", [])
    headroom_rows = selection.get("headroom_rows", [])
    used_ids = used_pair_ids(bank)
    return [
        gate_check(
            "reformulation_fixture_ready_score", 1, fixture.get("reformulation_fixture_ready_score")
        ),
        gate_check(
            "reformulation_bank_complete_score", 1, bank.get("reformulation_bank_complete_score")
        ),
        gate_check(
            "smt_certification_run_complete_score",
            1,
            certificate.get("smt_certification_run_complete_score"),
        ),
        gate_check(
            "certified_selection_run_complete_score",
            1,
            selection.get("certified_selection_run_complete_score"),
        ),
        gate_check(
            "raw_proposal_roster",
            {
                "attempt_count": EXPECTED_PROPOSAL_COUNT,
                "raw_count": EXPECTED_PROPOSAL_COUNT,
                "unique": True,
                "hashes_match": True,
            },
            _raw_roster_observation(bank),
        ),
        gate_check(
            "certificate_proposal_count",
            EXPECTED_PROPOSAL_COUNT,
            len(certificate_rows) if isinstance(certificate_rows, list) else 0,
        ),
        gate_check("exact_fixture_pair_count", EXPECTED_FIXTURE_PAIR_COUNT, len(pair_ids)),
        gate_check("unique_exact_pair_ids", EXPECTED_FIXTURE_PAIR_COUNT, len(set(pair_ids))),
        gate_check(
            "exact_fixture_witness_count",
            EXPECTED_FIXTURE_PAIR_COUNT,
            len(fixture_witnesses) if isinstance(fixture_witnesses, list) else 0,
        ),
        gate_check(
            "exact_fixture_solver_agreement",
            True,
            isinstance(fixture_agreement, list)
            and len(fixture_agreement) == EXPECTED_FIXTURE_PAIR_COUNT
            and all(row.get("authorities_agree") is True for row in fixture_agreement),
        ),
        gate_check("v609_used_pair_count", 18, len(used_ids)),
        gate_check(
            "v609_zero_candidate_headroom",
            {"row_count": 54, "values": [0]},
            {
                "row_count": len(headroom_rows) if isinstance(headroom_rows, list) else 0,
                "values": sorted({row.get("available_headroom") for row in headroom_rows})
                if isinstance(headroom_rows, list)
                else [],
            },
        ),
        gate_check("z3_available", True, fixture_exp.z3 is not None),
    ]


def _attempt_for_recomputation(attempt: Mapping[str, Any], pair: Mapping[str, Any]) -> JsonDict:
    """Rebuild one certificate input from raw text and the frozen parser only."""

    if attempt.get("call_status") == "complete":
        parse = bank_exp.parse_candidate(
            str(attempt.get("raw_text", "")),
            {
                "source_formulation": attempt["source_formulation"],
                "target_formulation": attempt["target_formulation"],
            },
        )
    else:
        parse = {
            "json_valid": False,
            "schema_valid": False,
            "failure_reason": attempt.get("failure_reason") or attempt.get("call_status"),
            "parsed_candidate": None,
            "confidence": None,
            "rationale": None,
        }
    return {
        "attempt_key": attempt["attempt_key"],
        "hf_id": attempt.get("hf_id"),
        "model_family": attempt.get("model_family"),
        "pair_id": attempt.get("pair_id"),
        "problem_family": attempt.get("problem_family"),
        "prompt_variant_id": attempt.get("prompt_variant_id"),
        "raw_sha256": attempt.get("raw_sha256"),
        "source_formulation": deepcopy(attempt["source_formulation"]),
        "target_formulation": deepcopy(attempt["target_formulation"]),
        "parse": parse,
        "canonical_relation": pair["expected_label"],
        "difficulty": pair["difficulty"],
    }


def _error_signature(bundle: Mapping[str, Any]) -> list[str]:
    """Return one ordered signature that preserves every failed obligation."""

    proposal = bundle["proposal_row"]
    if proposal.get("exact_mapping_correct") is True:
        return []
    parse = bundle["parse_row"]
    schema = bundle["schema_row"]
    agreement = bundle["authority_agreement_row"]
    enumeration = bundle["enumeration_row"]
    z3_row = bundle["z3_row"]
    if parse.get("parse_failure") is True:
        return [f"parse:{parse.get('failure_reason') or 'parse_rejected'}"]
    if schema.get("schema_valid") is not True:
        return [f"schema:{schema.get('failure_reason') or 'schema_rejected'}"]
    signature: list[str] = []
    if proposal.get("timeout") is True:
        signature.append("solver:timeout")
    if proposal.get("unknown") is True:
        signature.append("solver:unknown")
    if agreement.get("authorities_agree") is not True:
        signature.append("solver:authority_disagreement")
    checks = (
        ("forward_feasible", "domain_correspondence:forward"),
        ("reverse_feasible", "domain_correspondence:reverse"),
        ("variable_coverage_complete", "domain_correspondence:variable_coverage"),
        ("objective_direction_valid", "objective:direction"),
        ("objective_affine_preserved", "objective:affine_value"),
        ("objective_order_preserved", "objective:order"),
    )
    for field, label in checks:
        if enumeration.get(field) is False or z3_row.get(field) is False:
            signature.append(label)
    if not signature:
        signature.append(
            "relation_mismatch:"
            f"{proposal.get('certified_relation')}_vs_{proposal.get('canonical_relation')}"
        )
    return signature


def _repair_stage(signature: Sequence[str]) -> str | None:
    """Map an exact failure signature to the earliest applicable repair stage."""

    if not signature:
        return None
    if signature[0].startswith("parse:"):
        return "parse_repair"
    if signature[0].startswith("schema:"):
        return "schema_repair"
    return "semantic_repair"


def recompute_proposal(
    attempt: Mapping[str, Any], pair: Mapping[str, Any]
) -> tuple[JsonDict, JsonDict]:
    """Reparse and recertify one unedited raw proposal with both authorities."""

    certificate_input = _attempt_for_recomputation(attempt, pair)
    bundle = certificate_exp.certify_proposal(certificate_input)
    signature = _error_signature(bundle)
    enumeration = bundle["enumeration_row"]
    z3_row = bundle["z3_row"]
    agreement = bundle["authority_agreement_row"]
    exact_payload = {
        "attempt_key": attempt["attempt_key"],
        "enumeration": enumeration,
        "z3": z3_row,
        "agreement": agreement,
    }
    evidence_hash = sha256_bytes(canonical_json(exact_payload))
    witness_row = {
        "subject_kind": "proposal",
        "attempt_key": attempt["attempt_key"],
        "pair_id": attempt["pair_id"],
        "enumeration_status": enumeration["status"],
        "enumeration_label": enumeration.get("label"),
        "enumeration_witnesses": deepcopy(enumeration.get("witnesses", {})),
        "enumeration_counterexamples": deepcopy(enumeration.get("counterexamples", {})),
        "z3_status": z3_row["status"],
        "z3_label": z3_row.get("label"),
        "z3_witnesses": deepcopy(z3_row.get("witnesses", {})),
        "z3_counterexamples": deepcopy(z3_row.get("counterexamples", {})),
        "authorities_agree": agreement["authorities_agree"],
        "exact_evidence_hash": evidence_hash,
    }
    proposal = bundle["proposal_row"]
    parse = bundle["parse_row"]
    schema = bundle["schema_row"]
    cross = bundle["cross_feasibility_row"]
    direction = bundle["objective_direction_row"]
    order = bundle["objective_order_row"]
    row = {
        "attempt_key": attempt["attempt_key"],
        "ordinal": attempt.get("ordinal"),
        "pair_id": attempt["pair_id"],
        "model_family": attempt.get("model_family"),
        "formulation_family": attempt.get("problem_family"),
        "prompt_variant_id": attempt.get("prompt_variant_id"),
        "raw_sha256": attempt.get("raw_sha256"),
        "raw_hash_recomputed": bank_exp.sha256_text(str(attempt.get("raw_text", ""))),
        "json_valid": parse["json_valid"],
        "parse_failure": proposal["parse_failure"],
        "parse_reason": parse.get("failure_reason") if proposal["parse_failure"] else None,
        "schema_valid": schema["schema_valid"],
        "schema_failure": proposal["schema_failure"],
        "schema_reason": schema.get("failure_reason") if proposal["schema_failure"] else None,
        "variable_coverage_complete": bundle["variable_coverage_row"]["coverage_complete"],
        "domain_correspondence": {
            "enumeration_forward": cross["enumeration_forward_feasible"],
            "enumeration_reverse": cross["enumeration_reverse_feasible"],
            "z3_forward": cross["z3_forward_feasible"],
            "z3_reverse": cross["z3_reverse_feasible"],
        },
        "objective_direction": {
            "enumeration": direction["enumeration_direction_valid"],
            "z3": direction["z3_direction_valid"],
        },
        "objective_affine": {
            "enumeration": order["enumeration_objective_affine_preserved"],
            "z3": order["z3_objective_affine_preserved"],
        },
        "objective_order": {
            "enumeration": order["enumeration_objective_order_preserved"],
            "z3": order["z3_objective_order_preserved"],
            "tie_count": order["tie_count"],
        },
        "enumeration_status": enumeration["status"],
        "z3_status": z3_row["status"],
        "timeout": proposal["timeout"],
        "unknown": proposal["unknown"],
        "authorities_agree": agreement["authorities_agree"],
        "quarantined": proposal["quarantined"],
        "canonical_relation": proposal["canonical_relation"],
        "certified_relation": proposal["certified_relation"],
        "exact_mapping_correct": proposal["exact_mapping_correct"],
        "error_signature": signature,
        "repair_stage": _repair_stage(signature),
        "descriptive_confidence": proposal["confidence"],
        "descriptive_rationale_present": proposal["rationale_present"],
        "self_report_is_authority": False,
        "exact_evidence_hash": evidence_hash,
        "terminal": proposal["terminal"],
    }
    return row, witness_row


def _prior_certificate_match(row: Mapping[str, Any], prior: Mapping[str, Any] | None) -> bool:
    """Compare only outcome fields that the predecessor also recorded."""

    if prior is None:
        return False
    fields = (
        "parse_failure",
        "schema_failure",
        "timeout",
        "unknown",
        "quarantined",
        "canonical_relation",
        "certified_relation",
        "exact_mapping_correct",
        "terminal",
    )
    return all(row.get(field) == prior.get(field) for field in fields)


def recompute_all(
    bank: Mapping[str, Any],
    pairs: Mapping[str, Mapping[str, Any]],
    prior_certificate: Mapping[str, Any],
) -> tuple[list[JsonDict], list[JsonDict]]:
    """Recompute the full proposal denominator in frozen ordinal order."""

    prior_rows = {
        str(row.get("attempt_key")): row
        for row in prior_certificate.get("proposal_rows", [])
        if isinstance(row, Mapping)
    }
    rows: list[JsonDict] = []
    witnesses: list[JsonDict] = []
    attempts = sorted(bank["attempt_rows"], key=lambda row: int(row.get("ordinal", -1)))
    for attempt in attempts:
        pair = pairs[str(attempt["pair_id"])]
        row, witness = recompute_proposal(attempt, pair)
        row["prior_certificate_match"] = _prior_certificate_match(
            row, prior_rows.get(str(row["attempt_key"]))
        )
        rows.append(row)
        witnesses.append(witness)
    return rows, witnesses


def _count_by(rows: Sequence[Mapping[str, Any]], field: str) -> JsonDict:
    """Return a lexical count map for one descriptive cluster dimension."""

    return dict(sorted(Counter(str(row.get(field)) for row in rows).items()))


def cluster_error_rows(
    rows: Sequence[Mapping[str, Any]],
    witness_rows: Sequence[Mapping[str, Any]] | None = None,
) -> JsonDict:
    """Group every incorrect proposal by its complete deterministic signature."""

    witnesses = {str(row.get("attempt_key")): row for row in (witness_rows or [])}
    grouped: dict[tuple[str, ...], list[Mapping[str, Any]]] = {}
    for row in sorted(rows, key=lambda item: str(item.get("attempt_key"))):
        if row.get("exact_mapping_correct") is True:
            continue
        signature = tuple(str(value) for value in row.get("error_signature", []))
        grouped.setdefault(signature, []).append(row)
    cluster_rows: list[JsonDict] = []
    example_rows: list[JsonDict] = []
    for signature in sorted(grouped):
        members = grouped[signature]
        cluster_id = "error_cluster_" + sha256_bytes(canonical_json(signature))[7:23]
        member_keys = [str(row["attempt_key"]) for row in members]
        cluster_rows.append(
            {
                "cluster_id": cluster_id,
                "error_signature": list(signature),
                "repair_stage": members[0].get("repair_stage"),
                "count": len(members),
                "member_attempt_keys_hash": sha256_bytes(canonical_json(member_keys)),
                "model_family_counts": _count_by(members, "model_family"),
                "formulation_family_counts": _count_by(members, "formulation_family"),
                "prompt_variant_counts": _count_by(members, "prompt_variant_id"),
                "example_attempt_keys": member_keys[:3],
            }
        )
        for row in members[:3]:
            key = str(row["attempt_key"])
            example_rows.append(
                {
                    "cluster_id": cluster_id,
                    "attempt_key": key,
                    "raw_sha256": row.get("raw_sha256"),
                    "exact_evidence_hash": row.get("exact_evidence_hash"),
                    "exact_witness": deepcopy(witnesses.get(key)),
                }
            )
    cluster_by_signature = {
        tuple(row["error_signature"]): row["cluster_id"] for row in cluster_rows
    }
    for row in rows:
        if row.get("exact_mapping_correct") is not True:
            row["error_cluster_id"] = cluster_by_signature[tuple(row["error_signature"])]
        else:
            row["error_cluster_id"] = None
    return {"error_cluster_rows": cluster_rows, "cluster_example_rows": example_rows}


def used_pair_ids(bank: Mapping[str, Any]) -> set[str]:
    """Return every exact pair ID that appeared in any prior model prompt."""

    rows = bank.get("attempt_rows", [])
    return {
        str(row.get("pair_id"))
        for row in rows
        if isinstance(row, Mapping) and row.get("pair_id") is not None
    }


def _selection_key(pair: Mapping[str, Any], seed: int) -> tuple[str, str]:
    """Rank one unused pair without consulting a later model outcome."""

    pair_id = str(pair["pair_id"])
    digest = sha256_bytes(canonical_json({"seed": seed, "pair_id": pair_id}))
    return digest, pair_id


def freeze_slices(
    pairs: Sequence[Mapping[str, Any]], used_ids: set[str], *, seed: int
) -> dict[str, list[JsonDict]]:
    """Freeze balanced disjoint slices from exact pairs absent from V609 prompts."""

    available = [deepcopy(dict(row)) for row in pairs if str(row.get("pair_id")) not in used_ids]
    cells: dict[tuple[str, str], list[JsonDict]] = {}
    for family in FAMILIES:
        for label in ("equivalent", "non_equivalent"):
            members = [
                row
                for row in available
                if row.get("family") == family and row.get("expected_label") == label
            ]
            cells[(family, label)] = sorted(members, key=lambda row: _selection_key(row, seed))
    cursors = {cell: 0 for cell in cells}
    result: dict[str, list[JsonDict]] = {}
    for split, policy in SLICE_POLICY.items():
        selected: list[JsonDict] = []
        for family in FAMILIES:
            for label, count in policy.items():
                cell = (family, label)
                start = cursors[cell]
                stop = start + count
                members = cells[cell][start:stop]
                if len(members) != count:
                    raise ValueError(f"insufficient_unused_pairs:{family}:{label}")
                selected.extend(deepcopy(members))
                cursors[cell] = stop
        result[split] = sorted(selected, key=lambda row: _selection_key(row, seed))
    return result


def pairwise_sets(
    split_ids: Mapping[str, set[str]],
) -> dict[tuple[str, str], tuple[set[str], set[str]]]:
    """Return every pair of sets so tests and artifacts use the same isolation roster."""

    names = sorted(split_ids)
    return {
        (left, right): (set(split_ids[left]), set(split_ids[right]))
        for left, right in combinations(names, 2)
    }


def _used_pair_exclusion_rows(bank: Mapping[str, Any]) -> list[JsonDict]:
    """Hash every used pair ID and prove that each one is excluded."""

    attempts = bank.get("attempt_rows", [])
    counts = Counter(str(row.get("pair_id")) for row in attempts if isinstance(row, Mapping))
    return [
        {
            "pair_id": pair_id,
            "pair_id_hash": sha256_bytes(pair_id.encode()),
            "v609_attempt_count": counts[pair_id],
            "excluded_from_all_new_slices": True,
        }
        for pair_id in sorted(counts)
    ]


def _fixture_certificate(pair: Mapping[str, Any], split: str) -> tuple[JsonDict, JsonDict]:
    """Recompute one selected pair label and retain both exact engine records."""

    enumeration = fixture_exp.prove_pair_with_enumerator(pair)
    z3_row = fixture_exp.prove_pair_with_z3(pair)
    agreement = fixture_exp.authority_agreement_row(pair, enumeration, z3_row)
    payload = {
        "pair_id": pair["pair_id"],
        "split": split,
        "expected_label": pair["expected_label"],
        "difficulty": pair["difficulty"],
        "hard_negative_edit": pair.get("hard_negative_edit"),
        "enumeration": enumeration,
        "z3": z3_row,
        "agreement": agreement,
    }
    certificate_hash = sha256_bytes(canonical_json(payload))
    witness = {
        "subject_kind": "slice_pair",
        **payload,
        "source_certificate_hash": certificate_hash,
    }
    solver_row = {
        "split": split,
        "pair_id": pair["pair_id"],
        "expected_label": pair["expected_label"],
        "enumeration_label": enumeration.get("label"),
        "z3_label": z3_row.get("label"),
        "enumeration_status": enumeration.get("status"),
        "z3_status": z3_row.get("status"),
        "authorities_agree": agreement.get("authorities_agree") is True,
        "labels_match_expected": enumeration.get("label")
        == z3_row.get("label")
        == pair["expected_label"],
        "source_certificate_hash": certificate_hash,
    }
    return witness, solver_row


def _prompt_record(pair: Mapping[str, Any], split: str) -> JsonDict:
    """Build the exact public record that a later model may inspect."""

    row = {
        "record_id": f"{split}:{pair['pair_id']}",
        "split": split,
        "pair_id": pair["pair_id"],
        "formulation_family": pair["family"],
        "source_formulation": deepcopy(pair["source"]),
        "target_formulation": deepcopy(pair["target"]),
    }
    row["prompt_record_hash"] = sha256_bytes(canonical_json(row))
    return row


def find_forbidden_prompt_paths(value: Any, prefix: str = "") -> list[str]:
    """Find label-bearing field names at every depth of a model-visible payload."""

    paths: list[str] = []
    if isinstance(value, Mapping):
        for key, item in value.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            if str(key) in FORBIDDEN_PROMPT_FIELDS:
                paths.append(path)
            paths.extend(find_forbidden_prompt_paths(item, path))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            path = f"{prefix}[{index}]"
            paths.extend(find_forbidden_prompt_paths(item, path))
    return paths


def _family_balance_rows(slices: Mapping[str, Sequence[Mapping[str, Any]]]) -> list[JsonDict]:
    """Record family and label counts for every frozen slice."""

    rows: list[JsonDict] = []
    for split in SLICE_POLICY:
        expected_count = sum(SLICE_POLICY[split].values())
        for family in FAMILIES:
            members = [row for row in slices[split] if row.get("family") == family]
            labels = Counter(str(row.get("expected_label")) for row in members)
            rows.append(
                {
                    "split": split,
                    "formulation_family": family,
                    "pair_count": len(members),
                    "expected_pair_count": expected_count,
                    "equivalent_count": labels["equivalent"],
                    "hard_negative_count": labels["non_equivalent"],
                    "balance_passed": len(members) == expected_count
                    and labels["equivalent"] == SLICE_POLICY[split]["equivalent"]
                    and labels["non_equivalent"] == SLICE_POLICY[split]["non_equivalent"],
                }
            )
    return rows


def _split_disjointness_rows(
    used_ids: set[str], slices: Mapping[str, Sequence[Mapping[str, Any]]]
) -> list[JsonDict]:
    """Record every pairwise intersection across the used and new sets."""

    split_ids = {name: {str(row["pair_id"]) for row in rows} for name, rows in slices.items()}
    split_ids["v609_used"] = set(used_ids)
    rows: list[JsonDict] = []
    for (left, right), (left_ids, right_ids) in pairwise_sets(split_ids).items():
        overlap = sorted(left_ids & right_ids)
        rows.append(
            {
                "left_split": left,
                "right_split": right,
                "left_count": len(left_ids),
                "right_count": len(right_ids),
                "overlap_pair_ids": overlap,
                "disjoint": not overlap,
            }
        )
    return rows


def _build_slice_surfaces(
    slices: Mapping[str, Sequence[Mapping[str, Any]]],
    proposal_witnesses: Sequence[Mapping[str, Any]],
) -> JsonDict:
    """Build separate public prompts, sealed labels, and chronological records."""

    prompt_rows: list[JsonDict] = []
    slice_witnesses: list[JsonDict] = []
    agreement_rows: list[JsonDict] = []
    manifests: dict[str, list[JsonDict]] = {"calibration": [], "heldout": []}
    prompts_by_key: dict[tuple[str, str], JsonDict] = {}
    witnesses_by_key: dict[tuple[str, str], JsonDict] = {}
    for split in SLICE_POLICY:
        for pair in slices[split]:
            prompt = _prompt_record(pair, split)
            witness, agreement = _fixture_certificate(pair, split)
            prompt_rows.append(prompt)
            slice_witnesses.append(witness)
            agreement_rows.append(agreement)
            key = (split, str(pair["pair_id"]))
            prompts_by_key[key] = prompt
            witnesses_by_key[key] = witness
            if split in manifests:
                manifests[split].append(
                    {
                        "pair_id": pair["pair_id"],
                        "formulation_family": pair["family"],
                        "source_fixture_split": pair["split"],
                        "prompt_record_hash": prompt["prompt_record_hash"],
                        "source_certificate_hash": witness["source_certificate_hash"],
                    }
                )
    chronological_rows: list[JsonDict] = []
    previous_id: str | None = None
    previous_hash = sha256_bytes(canonical_json({"chronological_genesis": RANDOM_SEED}))
    for ordinal, pair in enumerate(slices["chronological"]):
        pair_id = str(pair["pair_id"])
        prompt = prompts_by_key[("chronological", pair_id)]
        witness = witnesses_by_key[("chronological", pair_id)]
        event_id = (
            "chronological_event_"
            + sha256_bytes(
                canonical_json({"seed": RANDOM_SEED, "ordinal": ordinal, "pair_id": pair_id})
            )[7:23]
        )
        dependency_ids = [f"exp6955:pair:{pair_id}"]
        if previous_id is not None:
            dependency_ids.append(f"exp6967:event:{previous_id}")
        dependency_record = {
            "dependency_ids": dependency_ids,
            "predecessor_event_id": previous_id,
            "predecessor_record_hash": previous_hash,
        }
        row = {
            "event_id": event_id,
            "event_ordinal": ordinal,
            "pair_id": pair_id,
            "formulation_family": pair["family"],
            "predecessor_event_id": previous_id,
            "predecessor_record_hash": previous_hash,
            "dependency_ids": dependency_ids,
            "dependency_record_hash": sha256_bytes(canonical_json(dependency_record)),
            "source_certificate_hash": witness["source_certificate_hash"],
            "prompt_record_hash": prompt["prompt_record_hash"],
            "later_outcome_exists": False,
        }
        row["event_hash"] = sha256_bytes(canonical_json(row))
        chronological_rows.append(row)
        previous_id = event_id
        previous_hash = row["event_hash"]
    sealed_hashes = {
        split: sha256_bytes(
            canonical_json([row for row in slice_witnesses if row.get("split") == split])
        )
        for split in SLICE_POLICY
    }
    split_hashes = {
        split: sha256_bytes(
            canonical_json(
                {
                    "split": split,
                    "prompt_record_hashes": [
                        row["prompt_record_hash"]
                        for row in prompt_rows
                        if row.get("split") == split
                    ],
                    "sealed_label_hash": sealed_hashes[split],
                    "event_hashes": [row["event_hash"] for row in chronological_rows]
                    if split == "chronological"
                    else [],
                }
            )
        )
        for split in SLICE_POLICY
    }
    return {
        "calibration_rows": manifests["calibration"],
        "heldout_rows": manifests["heldout"],
        "chronological_event_rows": chronological_rows,
        "prompt_visible_rows": prompt_rows,
        "exact_witness_rows": [deepcopy(row) for row in proposal_witnesses] + slice_witnesses,
        "solver_agreement_rows": agreement_rows,
        "sealed_label_hashes": sealed_hashes,
        "split_hashes": split_hashes,
    }


def _headroom_opportunity_rows(rows: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    """Describe repair opportunities while refusing an untested model claim."""

    return [
        {
            "opportunity_id": stage,
            "repair_stage": stage,
            "diagnosed_failure_count": sum(row.get("repair_stage") == stage for row in rows),
            "live_candidate_count": 0,
            "live_candidates_exist": False,
            "model_headroom_claim": None,
            "claim_status": "not_tested_no_live_candidates",
        }
        for stage in ("parse_repair", "schema_repair", "semantic_repair")
    ]


def _chronology_ready(rows: Sequence[Mapping[str, Any]]) -> bool:
    """Verify order, predecessor hashes, dependencies, and absent outcomes."""

    if len(rows) != 24:
        return False
    genesis = sha256_bytes(canonical_json({"chronological_genesis": RANDOM_SEED}))
    previous_id: str | None = None
    previous_hash = genesis
    for ordinal, row in enumerate(rows):
        dependency_record = {
            "dependency_ids": row.get("dependency_ids"),
            "predecessor_event_id": previous_id,
            "predecessor_record_hash": previous_hash,
        }
        stored_hash = row.get("event_hash")
        event_payload = {key: value for key, value in row.items() if key != "event_hash"}
        if not (
            row.get("event_ordinal") == ordinal
            and row.get("predecessor_event_id") == previous_id
            and row.get("predecessor_record_hash") == previous_hash
            and isinstance(row.get("dependency_ids"), list)
            and bool(row.get("dependency_ids"))
            and row.get("dependency_record_hash") == sha256_bytes(canonical_json(dependency_record))
            and isinstance(row.get("source_certificate_hash"), str)
            and str(row.get("source_certificate_hash")).startswith("sha256:")
            and isinstance(row.get("prompt_record_hash"), str)
            and str(row.get("prompt_record_hash")).startswith("sha256:")
            and row.get("later_outcome_exists") is False
            and "outcome" not in row
            and stored_hash == sha256_bytes(canonical_json(event_payload))
        ):
            return False
        previous_id = str(row.get("event_id"))
        previous_hash = str(stored_hash)
    return True


def source_artifact_hashes(repo_root: Path) -> JsonDict:
    """Bind output to every frozen input and source that interprets it."""

    paths = {
        **SOURCE_PATHS,
        "module": MODULE_PATH,
        "test": TEST_PATH,
        "spec": SPEC_PATH,
        "wrapper": WRAPPER_PATH,
    }
    return {
        name: {"path": str(path), "sha256": sha256_path(repo_root / path)}
        for name, path in paths.items()
    }


def _empty_surfaces() -> JsonDict:
    """Return every required row surface for a schema-complete blocked result."""

    fields = (
        "rows",
        "recomputed_proposal_rows",
        "error_cluster_rows",
        "cluster_example_rows",
        "used_pair_exclusion_rows",
        "calibration_rows",
        "heldout_rows",
        "chronological_event_rows",
        "family_balance_rows",
        "exact_witness_rows",
        "solver_agreement_rows",
        "split_disjointness_rows",
        "prompt_visible_rows",
        "headroom_opportunity_rows",
    )
    return {field: [] for field in fields} | {"sealed_label_hashes": {}, "split_hashes": {}}


def payload_checksum(artifact: Mapping[str, Any]) -> str:
    """Hash scientific content while excluding wall time and the digest itself."""

    payload = deepcopy(dict(artifact))
    payload.pop("duration_s", None)
    payload.pop("reproducibility_checksum", None)
    return sha256_bytes(canonical_json(payload))


def build_blocked_artifact(
    *,
    date: str,
    duration_s: float,
    checks: Sequence[Mapping[str, Any]],
    source_hashes: Mapping[str, Any],
) -> JsonDict:
    """Build the full diagnostic schema when an input precondition fails."""

    artifact: JsonDict = {
        "schema": SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "run_date": date,
        "field_principles": dict(FIELD_PRINCIPLES),
        "preconditions_checked": [deepcopy(dict(row)) for row in checks],
        "inference_substrate": INFERENCE_SUBSTRATE,
        "duration_s": round(duration_s, 9),
        "source_artifact_hashes": deepcopy(dict(source_hashes)),
        **_empty_surfaces(),
        "error_fixture_ready_score": 0,
        "chronological_event_stream_ready_score": 0,
        "random_seed": RANDOM_SEED,
        "reproducibility_checksum": "",
        "gate_check_summary": gate_summary(checks),
        "verifier_is_oracle": True,
        "verdict_class": "blocked",
        "honest_verdict": "blocked_certified_error_headroom_fixture",
    }
    artifact["reproducibility_checksum"] = payload_checksum(artifact)
    return artifact


def _readiness(
    proposal_rows: Sequence[Mapping[str, Any]],
    cluster_rows: Sequence[Mapping[str, Any]],
    slice_surfaces: Mapping[str, Any],
    balance_rows: Sequence[Mapping[str, Any]],
    disjointness_rows: Sequence[Mapping[str, Any]],
) -> tuple[int, int]:
    """Reduce both bare gates only from terminal row evidence."""

    failures = [row for row in proposal_rows if row.get("exact_mapping_correct") is not True]
    error_ready = int(
        len(proposal_rows) == EXPECTED_PROPOSAL_COUNT
        and len({str(row.get("attempt_key")) for row in proposal_rows}) == EXPECTED_PROPOSAL_COUNT
        and all(row.get("terminal") is True for row in proposal_rows)
        and all(row.get("prior_certificate_match") is True for row in proposal_rows)
        and all(isinstance(row.get("error_cluster_id"), str) for row in failures)
        and sum(int(row.get("count", 0)) for row in cluster_rows) == len(failures)
        and all(row.get("disjoint") is True for row in disjointness_rows)
        and all(row.get("balance_passed") is True for row in balance_rows)
        and len(slice_surfaces["calibration_rows"]) == 18
        and len(slice_surfaces["heldout_rows"]) == 18
        and len(slice_surfaces["chronological_event_rows"]) == 24
        and len(slice_surfaces["prompt_visible_rows"]) == 60
        and not find_forbidden_prompt_paths(slice_surfaces["prompt_visible_rows"])
        and len(slice_surfaces["solver_agreement_rows"]) == 60
        and all(
            row.get("authorities_agree") is True and row.get("labels_match_expected") is True
            for row in slice_surfaces["solver_agreement_rows"]
        )
    )
    chronology_ready = int(_chronology_ready(slice_surfaces["chronological_event_rows"]))
    return error_ready, chronology_ready


def build_artifact(
    *, date: str, repo_root: Path, inputs: Mapping[str, Any] | None = None
) -> JsonDict:
    """Recompute errors, freeze unused exact pairs, and derive both gates."""

    started = time.monotonic()
    loaded = dict(inputs) if inputs is not None else load_inputs(repo_root)
    checks = check_preconditions(loaded)
    hashes = source_artifact_hashes(repo_root)
    if any(row["passed"] is not True for row in checks):
        return build_blocked_artifact(
            date=date,
            duration_s=time.monotonic() - started,
            checks=checks,
            source_hashes=hashes,
        )
    bank = loaded["bank"]
    pairs = loaded["fixture_pairs"]
    proposal_rows, proposal_witnesses = recompute_all(bank, pairs, loaded["certificate"])
    clusters = cluster_error_rows(proposal_rows, proposal_witnesses)
    used_ids = used_pair_ids(bank)
    slices = freeze_slices(list(pairs.values()), used_ids, seed=RANDOM_SEED)
    slice_surfaces = _build_slice_surfaces(slices, proposal_witnesses)
    balance_rows = _family_balance_rows(slices)
    disjointness_rows = _split_disjointness_rows(used_ids, slices)
    opportunities = _headroom_opportunity_rows(proposal_rows)
    error_ready, chronology_ready = _readiness(
        proposal_rows,
        clusters["error_cluster_rows"],
        slice_surfaces,
        balance_rows,
        disjointness_rows,
    )
    complete = error_ready == chronology_ready == 1
    artifact: JsonDict = {
        "schema": SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "run_date": date,
        "field_principles": dict(FIELD_PRINCIPLES),
        "preconditions_checked": checks,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "duration_s": round(time.monotonic() - started, 9),
        "source_artifact_hashes": hashes,
        "rows": deepcopy(proposal_rows),
        "recomputed_proposal_rows": proposal_rows,
        "error_cluster_rows": clusters["error_cluster_rows"],
        "cluster_example_rows": clusters["cluster_example_rows"],
        "used_pair_exclusion_rows": _used_pair_exclusion_rows(bank),
        "calibration_rows": slice_surfaces["calibration_rows"],
        "heldout_rows": slice_surfaces["heldout_rows"],
        "chronological_event_rows": slice_surfaces["chronological_event_rows"],
        "family_balance_rows": balance_rows,
        "exact_witness_rows": slice_surfaces["exact_witness_rows"],
        "solver_agreement_rows": slice_surfaces["solver_agreement_rows"],
        "split_disjointness_rows": disjointness_rows,
        "prompt_visible_rows": slice_surfaces["prompt_visible_rows"],
        "sealed_label_hashes": slice_surfaces["sealed_label_hashes"],
        "split_hashes": slice_surfaces["split_hashes"],
        "headroom_opportunity_rows": opportunities,
        "error_fixture_ready_score": error_ready,
        "chronological_event_stream_ready_score": chronology_ready,
        "random_seed": RANDOM_SEED,
        "reproducibility_checksum": "",
        "gate_check_summary": [],
        "verifier_is_oracle": True,
        "verdict_class": "circular_positive" if complete else "partial",
        "honest_verdict": (
            "complete_circular_certified_error_headroom_fixture"
            if complete
            else "partial_certified_error_headroom_fixture"
        ),
    }
    artifact["reproducibility_checksum"] = payload_checksum(artifact)
    return artifact


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """Reject any terminal claim that is not supported by its row evidence."""

    missing = sorted(set(REQUIRED_ARTIFACT_FIELDS) - set(artifact))
    if missing:
        raise ValueError(f"required_artifact_fields:{missing}")
    if set(REQUIRED_ARTIFACT_FIELDS) - set(artifact.get("field_principles", {})):
        raise ValueError("field_principles")
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        raise ValueError("inference_substrate")
    if artifact.get("verifier_is_oracle") is not True:
        raise ValueError("verifier_is_oracle")
    for field in ("error_fixture_ready_score", "chronological_event_stream_ready_score"):
        if type(artifact.get(field)) is not int or artifact[field] not in {0, 1}:
            raise ValueError(f"bare_gate_field:{field}")
    if artifact.get("verdict_class") == "blocked":
        if not artifact.get("gate_check_summary"):
            raise ValueError("blocked_gate_check_summary")
    else:
        proposals = artifact.get("recomputed_proposal_rows", [])
        if len(proposals) != EXPECTED_PROPOSAL_COUNT:
            raise ValueError("recomputed_proposal_count")
        clusters = artifact.get("error_cluster_rows", [])
        balance = artifact.get("family_balance_rows", [])
        disjointness = artifact.get("split_disjointness_rows", [])
        surfaces = {
            field: artifact.get(field)
            for field in (
                "calibration_rows",
                "heldout_rows",
                "chronological_event_rows",
                "prompt_visible_rows",
                "solver_agreement_rows",
            )
        }
        expected_error, expected_chronology = _readiness(
            proposals, clusters, surfaces, balance, disjointness
        )
        if artifact.get("error_fixture_ready_score") != expected_error:
            raise ValueError("error_fixture_ready_score")
        if artifact.get("chronological_event_stream_ready_score") != expected_chronology:
            raise ValueError("chronological_event_stream_ready_score")
        if artifact.get("verdict_class") != (
            "circular_positive" if expected_error == expected_chronology == 1 else "partial"
        ):
            raise ValueError("verdict_class")
        if any(
            row.get("model_headroom_claim") is not None or row.get("live_candidate_count") != 0
            for row in artifact.get("headroom_opportunity_rows", [])
        ):
            raise ValueError("fabricated_headroom")
    if artifact.get("reproducibility_checksum") != payload_checksum(artifact):
        raise ValueError("reproducibility_checksum")


def run(*, date: str, repo_root: Path, output_path: Path | None = None) -> JsonDict:
    """Build, validate, and atomically publish the dated terminal artifact."""

    target = output_path or repo_root / OUTPUT_PATH
    started = time.monotonic()
    try:
        inputs = load_inputs(repo_root)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        checks = [
            gate_check(
                "source_artifacts_loadable",
                True,
                {"loadable": False, "error": f"{type(exc).__name__}:{exc}"},
            )
        ]
        artifact = build_blocked_artifact(
            date=date,
            duration_s=time.monotonic() - started,
            checks=checks,
            source_hashes=source_artifact_hashes(repo_root),
        )
    else:
        artifact = build_artifact(date=date, repo_root=repo_root, inputs=inputs)
    validate_artifact(artifact)
    write_json_atomic(target, artifact)
    return artifact


def main(argv: Sequence[str] | None = None) -> int:
    """Expose the required dated command without any model-inference option."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", required=True)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    artifact = run(date=args.date, repo_root=args.repo_root, output_path=args.output)
    print(
        json.dumps(
            {
                "artifact": str(args.output or args.repo_root / OUTPUT_PATH),
                "error_fixture_ready_score": artifact["error_fixture_ready_score"],
                "chronological_event_stream_ready_score": artifact[
                    "chronological_event_stream_ready_score"
                ],
                "verdict_class": artifact["verdict_class"],
                "honest_verdict": artifact["honest_verdict"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - the wrapper is the required command surface.
    raise SystemExit(main())
