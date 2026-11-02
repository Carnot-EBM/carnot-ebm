"""Build a sealed chronological stream for later constraint learning.

The stream changes fault families across four blocks. It also keeps tied
groups and a final recurrence block. Exact labels live in a separate file, so
an earlier decision view cannot read a current or future outcome.

Spec refs: REQ-LEARN-6985 and SCENARIO-LEARN-6985-*.
"""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Mapping, Sequence
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time
from typing import Any

from carnot import experiment_6984_exact_contrast_fixture as contrast_exp


JsonDict = dict[str, Any]

EXPERIMENT_ID = 6985
SCHEMA_VERSION = "carnot.exp6985.chronological_constraint_stream.v1"
RUN_DATE = "20260904"
RANDOM_SEED = 6_985_202_609_04
EXPECTED_EVENT_COUNT = 24
INFERENCE_SUBSTRATE = "deterministic_z3_chronological_stream_no_llm"
RESULT_PATH = Path("results/experiment_6985_chronological_constraint_stream.json")
RAW_ROOT = Path("results/raw/experiment_6985_chronological_constraint_stream")
EXP6984_PATH = Path("results/experiment_6984_exact_contrast_fixture.json")
EXP6955_PATH = Path("results/experiment_6955_reformulation_fixture.json")
EXP6961_PATH = Path("results/experiment_6961_certified_event_sequence.json")
EXP6967_PATH = Path("results/experiment_6967_certified_error_headroom_fixture.json")
EXP6978_PATH = Path("results/experiment_6978_transactional_constraint_self_learning.json")
CONTRAST_MODULE_PATH = Path("python/carnot/experiment_6984_exact_contrast_fixture.py")

EXPECTED_EXP6984_HASH = "sha256:15f9a9bb58ca7793966f2fbac548f6879a64417b50e31b0504078cfe6ea46a3f"
EXPECTED_FAULT_MODULE_HASH = (
    "sha256:c7f2861a6d1364157cca6ca13d16f5b3ec1c818d326d1608b7e92de38b8fe4a7"
)
FAULT_FAMILIES = tuple(contrast_exp.FAULT_FAMILIES)
SHIFT_IDS = ("shift_0", "shift_1", "shift_2", "shift_3_recurrence")
BLOCK_HEADROOM_FAULTS = (
    ("bound_change", "coefficient_swap", "bound_change", "coefficient_swap"),
    (
        "objective_direction_reversal",
        "constraint_omission",
        "objective_direction_reversal",
        "constraint_omission",
    ),
    ("coefficient_swap", "constraint_omission", "coefficient_swap", "constraint_omission"),
    ("bound_change", "coefficient_swap", "bound_change", "coefficient_swap"),
)
BLOCK_INVALID_FAULTS = (
    ("bound_change", "coefficient_swap"),
    ("objective_direction_reversal", "constraint_omission"),
    ("coefficient_swap", "constraint_omission"),
    ("bound_change", "coefficient_swap"),
)
EVENT_TYPES = ("headroom", "headroom", "headroom", "headroom", "all_valid", "all_invalid")
VERDICT_CLASSES = {
    "positive",
    "circular_positive",
    "null",
    "blocked",
    "disqualified",
    "partial",
}
EXP6984_USED_SOURCE_PAIR_IDS = frozenset(
    f"{family}-{template}-{ordinal}"
    for family in range(3)
    for template in range(4)
    for ordinal in range(3)
)

FORBIDDEN_STREAM_LABEL_FIELDS = {
    "exact_label",
    "exact_labels",
    "expected_label",
    "expected_relation",
    "certified_relation",
    "authority_outcome",
    "future_labels",
    "current_label",
}
FORBIDDEN_VISIBILITY_FIELDS = FORBIDDEN_STREAM_LABEL_FIELDS | {
    "enumeration_authority",
    "z3_authority",
}

REQUIRED_ARTIFACT_FIELDS = (
    "schema",
    "experiment_id",
    "run_date",
    "field_principles",
    "preconditions_checked",
    "inference_substrate",
    "duration_s",
    "source_artifact_hashes",
    "source_disjointness_rows",
    "stream_manifest",
    "stream_hash",
    "sealed_label_path",
    "sealed_label_hash",
    "rows",
    "per_event_results",
    "per_candidate_rows",
    "shift_block_rows",
    "fault_family_rows",
    "tie_group_rows",
    "recurrence_rows",
    "retention_anchor_rows",
    "held_future_window_rows",
    "visibility_manifest_rows",
    "future_label_leakage_rows",
    "authority_agreement_rows",
    "expected_event_count",
    "observed_event_count",
    "headroom_event_count",
    "all_valid_event_count",
    "all_invalid_event_count",
    "chronological_stream_ready_score",
    "continuous_self_learning_fixture",
    "random_seed",
    "reproducibility_checksum",
    "gate_check_summary",
    "verifier_is_oracle",
    "verdict_class",
    "honest_verdict",
)

FIELD_PRINCIPLES = {
    "schema": "A schema version makes incompatible stream evidence fail replay.",
    "experiment_id": "A stable identity prevents another experiment from owning these rows.",
    "run_date": "A fixed date prevents silent changes to the execution boundary.",
    "field_principles": "A reason for every field makes the evidence contract reviewable.",
    "preconditions_checked": "Fail-closed checks stop missing exact resources from being invented.",
    "inference_substrate": "The substrate states that local exact solvers, not an LLM, made labels.",
    "duration_s": "Measured wall time proves that stream construction ran.",
    "source_artifact_hashes": "Source hashes bind the stream to the evidence it consumes.",
    "source_disjointness_rows": "Disjointness rows prevent static contrast sources from recurring.",
    "stream_manifest": "The manifest freezes file paths, orders, blocks, and support commitments.",
    "stream_hash": "A byte hash makes any later stream edit visible.",
    "sealed_label_path": "A separate path keeps exact outcomes outside public stream rows.",
    "sealed_label_hash": "A byte hash freezes all exact outcomes without opening them early.",
    "rows": "Attempt rows preserve accepted, rejected, and unknown construction outcomes.",
    "per_event_results": "Event rows expose the complete chronological denominator.",
    "per_candidate_rows": "Candidate rows freeze public payloads without exact labels.",
    "shift_block_rows": "Block rows prove that the fault distribution changes over time.",
    "fault_family_rows": "Fault rows retain zero and nonzero family cells for each shift.",
    "tie_group_rows": "Tie rows make the zero-advantage no-update control explicit.",
    "recurrence_rows": "Recurrence rows make later forgetting measurable on new sources.",
    "retention_anchor_rows": "Anchors identify the early support that a later learner must retain.",
    "held_future_window_rows": "Frozen future windows prevent support checks from moving after use.",
    "visibility_manifest_rows": "Manifest receipts prove each decision view contains predecessors only.",
    "future_label_leakage_rows": "Mutation receipts prove current and future labels fail closed.",
    "authority_agreement_rows": "Agreement rows require two exact decisions for each candidate.",
    "expected_event_count": "The preregistered count prevents quiet event replacement.",
    "observed_event_count": "The observed count exposes missing or added events.",
    "headroom_event_count": "The headroom count fixes the discriminative evaluation mass.",
    "all_valid_event_count": "The all-valid count fixes one zero-advantage control class.",
    "all_invalid_event_count": "The all-invalid count fixes the other zero-advantage control class.",
    "chronological_stream_ready_score": "One requires all exact, chronology, sealing, and replay gates.",
    "continuous_self_learning_fixture": "True marks the stream as later learning input, not an update.",
    "random_seed": "One seed fixes identifiers and label-blind candidate order.",
    "reproducibility_checksum": "A timing-free digest detects scientific-content drift.",
    "gate_check_summary": "Failures keep their expected and observed values for diagnosis.",
    "verifier_is_oracle": "True states that exact authorities supplied the fixture labels.",
    "verdict_class": "A closed class prevents oracle fixture evidence from becoming a learned win.",
    "honest_verdict": "A stable prefix lets automation classify the terminal state.",
}


class ImmutableStreamError(RuntimeError):
    """Report an attempt to change bytes at a sealed stream path."""


def canonical_json(value: Any) -> bytes:
    """Return the single UTF-8 JSON form used by every content hash."""

    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def sha256_bytes(value: bytes) -> str:
    """Return a repository-style SHA-256 digest for exact bytes."""

    return "sha256:" + hashlib.sha256(value).hexdigest()


def sha256_json(value: Any) -> str:
    """Hash JSON content without formatting or dictionary-order effects."""

    return sha256_bytes(canonical_json(value))


def sha256_path(path: Path) -> str | None:
    """Hash one file while preserving absence as a null value."""

    return sha256_bytes(path.read_bytes()) if path.is_file() else None


def _jsonl_bytes(rows: Sequence[Mapping[str, Any]]) -> bytes:
    """Serialize rows with one canonical JSON object on each line."""

    return b"".join(canonical_json(dict(row)) + b"\n" for row in rows)


def _write_immutable_bytes(path: Path, payload: bytes) -> str:
    """Create sealed bytes once and permit only byte-identical replay."""

    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("xb") as handle:
            handle.write(payload)
    except FileExistsError:
        if path.read_bytes() != payload:
            raise ImmutableStreamError(f"immutable_stream_mismatch:{path}") from None
    return sha256_bytes(payload)


def write_immutable_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> str:
    """Seal one JSONL file and return its exact byte hash."""

    return _write_immutable_bytes(path, _jsonl_bytes(rows))


def write_immutable_json(path: Path, value: Mapping[str, Any]) -> str:
    """Seal one canonical JSON object with a final newline."""

    return _write_immutable_bytes(path, canonical_json(dict(value)) + b"\n")


def read_jsonl(path: Path) -> list[JsonDict]:
    """Read a sealed JSONL file as object rows."""

    return [dict(json.loads(line)) for line in path.read_text().splitlines() if line]


def write_json_atomic(path: Path, value: Mapping[str, Any]) -> None:
    """Replace the aggregate only after complete JSON bytes exist beside it."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=path.name, delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(json.dumps(value, indent=2, sort_keys=True).encode() + b"\n")
    os.replace(temporary, path)


def _read_json(path: Path) -> JsonDict:
    """Read an object artifact or return an empty object at a failed boundary."""

    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}
    return dict(value) if isinstance(value, Mapping) else {}


def _storage_writable(raw_root: Path) -> bool:
    """Probe exclusive creation without touching a requested evidence path."""

    try:
        raw_root.mkdir(parents=True, exist_ok=True)
        descriptor, name = tempfile.mkstemp(prefix=".write-probe-", dir=raw_root)
        os.close(descriptor)
        Path(name).unlink()
    except OSError:  # pragma: no cover - the unit suite cannot revoke host permissions portably.
        return False
    return True


def gate_check(check: str, expected: Any, observed: Any, *, passed: bool | None = None) -> JsonDict:
    """Record one gate with exact diagnostic values."""

    return {
        "check": check,
        "expected_value": deepcopy(expected),
        "observed_value": deepcopy(observed),
        "passed": observed == expected if passed is None else passed,
    }


def gate_summary(checks: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    """Keep failed checks with the values needed to diagnose them."""

    return [
        {
            "check": row.get("check"),
            "expected_value": deepcopy(row.get("expected_value")),
            "observed_value": deepcopy(row.get("observed_value")),
        }
        for row in checks
        if row.get("passed") is not True
    ]


def exp6984_source_ids(repo_root: Path) -> set[str]:
    """Read the exact source IDs already consumed by the static contrast fixture."""

    artifact = _read_json(repo_root / EXP6984_PATH)
    return {
        str(row["source_pair_id"])
        for row in artifact.get("source_manifest_rows", [])
        if isinstance(row, Mapping) and row.get("source_pair_id") is not None
    }


def unused_source_pairs(repo_root: Path | None = None) -> dict[str, JsonDict]:
    """Return generated source pairs after an optional Exp6984 exclusion."""

    pairs = contrast_exp.frozen_source_pairs()
    if repo_root is None:
        return pairs
    used = exp6984_source_ids(repo_root)
    if not used:
        return {}
    return {pair_id: pair for pair_id, pair in pairs.items() if pair_id not in used}


def selected_source_rows(repo_root: Path) -> list[JsonDict]:
    """Select 24 clean unused bases without inspecting candidate outcomes."""

    rows: list[JsonDict] = []
    for pair_id, pair in unused_source_pairs(repo_root).items():
        family, template, ordinal = (int(part) for part in pair_id.split("-"))
        if template >= 4 or ordinal not in {3, 4} or pair.get("expected_label") != "equivalent":
            continue
        source_group_id = sha256_json(
            {
                "source_pair_id": pair_id,
                "source_hash": contrast_exp.fixture_exp.formulation_hash(pair["source"]),
                "target_hash": contrast_exp.fixture_exp.formulation_hash(pair["target"]),
            }
        )
        rows.append(
            {
                "source_pair_id": pair_id,
                "source_group_id": source_group_id,
                "formulation_family": pair["family"],
                "family_index": family,
                "template_index": template,
                "ordinal_in_template": ordinal,
            }
        )
    selected = sorted(
        rows,
        key=lambda row: tuple(int(part) for part in row["source_pair_id"].split("-")),
    )[:EXPECTED_EVENT_COUNT]
    # The fourth block uses a source whose objective has real coefficient headroom
    # for its fourth event. This order is fixed before the stream hash exists.
    selected[21], selected[22] = selected[22], selected[21]
    return selected


def source_artifact_hashes(repo_root: Path) -> JsonDict:
    """Bind the fixture to upstream stream patterns and exact source evidence."""

    return {
        "experiment_6984_exact_contrast_fixture": sha256_path(repo_root / EXP6984_PATH),
        "experiment_6955_reformulation_fixture": sha256_path(repo_root / EXP6955_PATH),
        "experiment_6961_certified_event_sequence": sha256_path(repo_root / EXP6961_PATH),
        "experiment_6967_certified_error_headroom_fixture": sha256_path(repo_root / EXP6967_PATH),
        "experiment_6978_transactional_constraint_self_learning": sha256_path(
            repo_root / EXP6978_PATH
        ),
        "frozen_fault_operator_module": sha256_path(repo_root / CONTRAST_MODULE_PATH),
    }


def collect_preconditions(repo_root: Path, raw_root: Path) -> list[JsonDict]:
    """Check the upstream score, frozen mutators, exact engines, and source floor."""

    upstream = _read_json(repo_root / EXP6984_PATH)
    score = upstream.get("contrast_fixture_complete_score")
    unused_count = len(unused_source_pairs(repo_root))
    fault_observed = {
        "fault_families": list(FAULT_FAMILIES),
        "module_hash": sha256_path(repo_root / CONTRAST_MODULE_PATH),
    }
    return [
        gate_check(
            "experiment_6984_artifact_hash",
            EXPECTED_EXP6984_HASH,
            sha256_path(repo_root / EXP6984_PATH),
        ),
        gate_check(
            "contrast_fixture_complete_score",
            1,
            score,
            passed=type(score) is int and score == 1,
        ),
        gate_check(
            "frozen_fault_operators",
            {
                "fault_families": list(FAULT_FAMILIES),
                "module_hash": EXPECTED_FAULT_MODULE_HASH,
            },
            fault_observed,
        ),
        gate_check("z3_available", True, contrast_exp.fixture_exp.z3 is not None),
        gate_check(
            "bounded_enumeration_available",
            True,
            callable(contrast_exp.certificate_exp.certify_with_enumerator),
        ),
        gate_check(
            "minimum_unused_source_groups",
            ">=24",
            unused_count,
            passed=unused_count >= EXPECTED_EVENT_COUNT,
        ),
        gate_check("immutable_stream_paths_writable", True, _storage_writable(raw_root)),
    ]


def build_event_plan(source_rows: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    """Freeze event types and fault-family changes before exact labels open."""

    if len(source_rows) < EXPECTED_EVENT_COUNT:
        raise ValueError("insufficient_unused_source_groups")
    rows: list[JsonDict] = []
    for event_ordinal, source in enumerate(source_rows[:EXPECTED_EVENT_COUNT]):
        block = event_ordinal // 6
        local = event_ordinal % 6
        event_type = EVENT_TYPES[local]
        if event_type == "headroom":
            candidate_faults: list[str | None] = [None, BLOCK_HEADROOM_FAULTS[block][local]]
        elif event_type == "all_valid":
            candidate_faults = [None, None]
        else:
            candidate_faults = list(BLOCK_INVALID_FAULTS[block])
        event_id = (
            "event_"
            + sha256_json(
                {
                    "seed": RANDOM_SEED,
                    "event_ordinal": event_ordinal,
                    "source_group_id": source["source_group_id"],
                }
            ).removeprefix("sha256:")[:20]
        )
        rows.append(
            {
                **deepcopy(dict(source)),
                "event_id": event_id,
                "event_ordinal": event_ordinal,
                "shift_id": SHIFT_IDS[block],
                "shift_ordinal": block,
                "block_event_ordinal": local,
                "event_type": event_type,
                "candidate_fault_families": candidate_faults,
                "plan_frozen_before_labels": True,
            }
        )
    return rows


def _candidate_payload(pair: Mapping[str, Any]) -> JsonDict:
    """Project executable content without source metadata or expected labels."""

    return {
        "source_formulation": deepcopy(pair["source"]),
        "target_formulation": deepcopy(pair["target"]),
        "mapping_candidate": deepcopy(pair["mapping"]),
    }


def _candidate_specs(event: Mapping[str, Any], base: Mapping[str, Any]) -> list[JsonDict]:
    """Construct two preregistered candidates for one event type."""

    event_type = str(event["event_type"])
    faults = list(event["candidate_fault_families"])
    if event_type == "headroom":
        negative, mutation = contrast_exp.apply_fault(base, str(faults[1]))
        return [
            {"pair": deepcopy(base), "fault_family": None, "relation": "equivalent"},
            {
                "pair": negative,
                "fault_family": faults[1],
                "relation": "non_equivalent",
                "mutation": mutation,
            },
        ]
    if event_type == "all_valid":
        renamed, receipt = contrast_exp.alpha_rename_pair(base, str(event["event_id"]))
        return [
            {"pair": deepcopy(base), "fault_family": None, "relation": "equivalent"},
            {
                "pair": renamed,
                "fault_family": None,
                "relation": "equivalent",
                "alpha_receipt": receipt,
            },
        ]
    return [
        {
            "pair": pair,
            "fault_family": fault,
            "relation": "non_equivalent",
            "mutation": mutation,
        }
        for fault in faults
        for pair, mutation in [contrast_exp.apply_fault(base, str(fault))]
    ]


def certify_candidate_attempt(
    pair: Mapping[str, Any], candidate_id: str, *, expected_relation: str
) -> JsonDict:
    """Keep a complete receipt for accepted, unknown, or rejected certification."""

    try:
        certificate = contrast_exp.certify_pair(pair, candidate_id)
    except Exception as error:  # Exact construction errors are evidence, not replacement signals.
        return {
            "candidate_id": candidate_id,
            "attempt_status": "rejected",
            "accepted": False,
            "terminal": True,
            "replacement_used": False,
            "rejection_reason": f"{type(error).__name__}:{error}",
            "expected_relation": expected_relation,
            "certified_relation": None,
            "enumeration_authority": {"status": "exception", "label": None},
            "z3_authority": {"status": "exception", "label": None},
            "agreement": {"all_required_agreement": False, "terminal": True},
        }
    enumeration = dict(certificate["enumeration"])
    z3_row = dict(certificate["z3"])
    agreement = dict(certificate["agreement"])
    decided = (
        enumeration.get("status") in contrast_exp.DECISION_STATUSES
        and z3_row.get("status") in contrast_exp.DECISION_STATUSES
    )
    relation = agreement.get("certified_relation")
    accepted = (
        decided
        and agreement.get("all_required_agreement") is True
        and relation == expected_relation
    )
    status = "accepted" if accepted else ("rejected" if decided else "unknown")
    return {
        "candidate_id": candidate_id,
        "attempt_status": status,
        "accepted": accepted,
        "terminal": decided,
        "replacement_used": False,
        "rejection_reason": None if accepted else f"authority_{status}",
        "expected_relation": expected_relation,
        "certified_relation": relation,
        "enumeration_authority": enumeration,
        "z3_authority": z3_row,
        "agreement": agreement,
    }


def _attempt_evidence_row(attempt: Mapping[str, Any], event: Mapping[str, Any]) -> JsonDict:
    """Preserve attempt status and receipts while keeping exact labels sealed."""

    enumeration = attempt["enumeration_authority"]
    z3_row = attempt["z3_authority"]
    agreement = attempt["agreement"]
    return {
        "event_id": event["event_id"],
        "event_ordinal": event["event_ordinal"],
        "candidate_id": attempt["candidate_id"],
        "attempt_status": attempt["attempt_status"],
        "accepted": attempt["accepted"],
        "terminal": attempt["terminal"],
        "replacement_used": attempt["replacement_used"],
        "rejection_reason": attempt["rejection_reason"],
        "enumeration_status": enumeration.get("status"),
        "z3_status": z3_row.get("status"),
        "enumeration_receipt_hash": sha256_json(enumeration),
        "z3_receipt_hash": sha256_json(z3_row),
        "agreement_receipt_hash": sha256_json(agreement),
    }


def _authority_agreement_row(attempt: Mapping[str, Any], event: Mapping[str, Any]) -> JsonDict:
    """Expose exact-engine parity without copying the sealed relation value."""

    agreement = attempt["agreement"]
    return {
        "event_id": event["event_id"],
        "event_ordinal": event["event_ordinal"],
        "candidate_id": attempt["candidate_id"],
        "enumeration_status": attempt["enumeration_authority"].get("status"),
        "z3_status": attempt["z3_authority"].get("status"),
        "all_required_agreement": agreement.get("all_required_agreement") is True,
        "terminal": attempt["terminal"],
        "enumeration_receipt_hash": sha256_json(attempt["enumeration_authority"]),
        "z3_receipt_hash": sha256_json(attempt["z3_authority"]),
    }


def _nested_keys(value: Any) -> set[str]:
    """Collect nested object keys for exact label-leakage checks."""

    if isinstance(value, Mapping):
        return set(map(str, value)) | set().union(*(_nested_keys(item) for item in value.values()))
    if isinstance(value, list):
        return set().union(*(_nested_keys(item) for item in value)) if value else set()
    return set()


def _source_disjointness_rows(stream_rows: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    """Compare the frozen new source IDs with the pinned Exp6984 source roster."""

    current = {str(row["source_pair_id"]) for row in stream_rows}
    overlap = sorted(current & EXP6984_USED_SOURCE_PAIR_IDS)
    return [
        {
            "left_source": "experiment_6984",
            "right_source": "experiment_6985",
            "left_source_count": len(EXP6984_USED_SOURCE_PAIR_IDS),
            "right_source_count": len(current),
            "overlap_count": len(overlap),
            "overlap_source_pair_ids": overlap,
            "terminal": True,
        }
    ]


def _shift_block_rows(stream_rows: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    """Reduce event and fault counts for all four chronological blocks."""

    rows: list[JsonDict] = []
    for shift_ordinal, shift_id in enumerate(SHIFT_IDS):
        block = [row for row in stream_rows if row["shift_id"] == shift_id]
        faults = Counter(
            fault for row in block for fault in row["candidate_fault_families"] if fault is not None
        )
        rows.append(
            {
                "shift_id": shift_id,
                "shift_ordinal": shift_ordinal,
                "event_ordinals": [row["event_ordinal"] for row in block],
                "event_count": len(block),
                "headroom_event_count": sum(row["event_type"] == "headroom" for row in block),
                "all_valid_event_count": sum(row["event_type"] == "all_valid" for row in block),
                "all_invalid_event_count": sum(row["event_type"] == "all_invalid" for row in block),
                "fault_family_counts": {fault: faults.get(fault, 0) for fault in FAULT_FAMILIES},
                "fault_mix_hash": sha256_json(
                    sorted(
                        fault
                        for row in block
                        for fault in row["candidate_fault_families"]
                        if fault is not None
                    )
                ),
                "terminal": len(block) == 6,
            }
        )
    return rows


def _fault_family_rows(stream_rows: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    """Keep every shift and fault cell, including required zero counts."""

    return [
        {
            "shift_id": shift_id,
            "fault_family": fault,
            "candidate_count": sum(
                candidate_fault == fault
                for row in stream_rows
                if row["shift_id"] == shift_id
                for candidate_fault in row["candidate_fault_families"]
            ),
            "terminal": True,
        }
        for shift_id in SHIFT_IDS
        for fault in FAULT_FAMILIES
    ]


def _recurrence_rows(stream_rows: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    """Link each final-block event to the matching early event on a new source."""

    first = sorted(
        (row for row in stream_rows if row["shift_id"] == SHIFT_IDS[0]),
        key=lambda row: row["block_event_ordinal"],
    )
    final = sorted(
        (row for row in stream_rows if row["shift_id"] == SHIFT_IDS[-1]),
        key=lambda row: row["block_event_ordinal"],
    )
    return [
        {
            "recurrence_id": f"recurrence_{index}",
            "anchor_event_id": anchor["event_id"],
            "anchor_event_ordinal": anchor["event_ordinal"],
            "anchor_shift_id": anchor["shift_id"],
            "revisit_event_id": revisit["event_id"],
            "revisit_event_ordinal": revisit["event_ordinal"],
            "revisit_shift_id": revisit["shift_id"],
            "source_group_is_new": anchor["source_group_id"] != revisit["source_group_id"],
            "fault_family_signature_matches": anchor["candidate_fault_families"]
            == revisit["candidate_fault_families"],
            "event_type_matches": anchor["event_type"] == revisit["event_type"],
            "terminal": True,
        }
        for index, (anchor, revisit) in enumerate(zip(first, final, strict=True))
    ]


def _retention_anchor_rows(recurrence_rows: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    """Freeze the six early events that later evaluation must retain."""

    return [
        {
            "retention_anchor_id": f"retention_anchor_{index}",
            "anchor_event_id": row["anchor_event_id"],
            "anchor_event_ordinal": row["anchor_event_ordinal"],
            "check_at_event_id": row["revisit_event_id"],
            "check_at_event_ordinal": row["revisit_event_ordinal"],
            "frozen_before_stream_release": True,
            "terminal": True,
        }
        for index, row in enumerate(recurrence_rows)
    ]


def _held_future_window_rows(
    stream_rows: Sequence[Mapping[str, Any]], label_rows: Sequence[Mapping[str, Any]]
) -> list[JsonDict]:
    """Freeze every future-support denominator and its sealed-label commitment."""

    labels_by_ordinal = {int(row["event_ordinal"]): dict(row) for row in label_rows}
    rows: list[JsonDict] = []
    for event in stream_rows:
        ordinal = int(event["event_ordinal"])
        future = [row for row in stream_rows if int(row["event_ordinal"]) > ordinal]
        future_labels = [labels_by_ordinal[int(row["event_ordinal"])] for row in future]
        faults = Counter(
            fault
            for row in future
            for fault in row["candidate_fault_families"]
            if fault is not None
        )
        rows.append(
            {
                "checkpoint_id": f"held_future_after_{ordinal:03d}",
                "after_event_id": event["event_id"],
                "after_event_ordinal": ordinal,
                "held_future_event_ids": [row["event_id"] for row in future],
                "held_future_event_ordinals": [row["event_ordinal"] for row in future],
                "support_metrics": {
                    "event_count": len(future),
                    "candidate_count": 2 * len(future),
                    "headroom_event_count": sum(row["event_type"] == "headroom" for row in future),
                    "tie_event_count": sum(row["event_type"] != "headroom" for row in future),
                    "fault_family_counts": {
                        fault: faults.get(fault, 0) for fault in FAULT_FAMILIES
                    },
                },
                "sealed_support_hash": sha256_json(future_labels),
                "frozen_before_stream_release": True,
                "terminal": True,
            }
        )
    return rows


def _decision_projection(row: Mapping[str, Any]) -> JsonDict:
    """Keep public predecessor data and omit exact-authority evidence."""

    allowed = (
        "event_id",
        "event_ordinal",
        "shift_id",
        "shift_ordinal",
        "block_event_ordinal",
        "source_group_id",
        "source_pair_id",
        "formulation_family",
        "event_type",
        "candidate_ids",
        "candidate_fault_families",
        "label_commitment_hash",
        "held_future_checkpoint_id",
    )
    return {key: deepcopy(row[key]) for key in allowed if key in row}


def build_visibility_manifest(
    public_rows: Sequence[Mapping[str, Any]], current_ordinal: int
) -> JsonDict:
    """Build an event view from strict predecessors and no exact labels."""

    visible = [
        _decision_projection(row)
        for row in public_rows
        if int(row["event_ordinal"]) < current_ordinal
    ]
    payload: JsonDict = {
        "current_event_ordinal": current_ordinal,
        "visible_event_ordinals": [row["event_ordinal"] for row in visible],
        "visible_events": visible,
        "decision_view_is_read_only": True,
    }
    payload["manifest_hash"] = sha256_json(payload)
    return payload


def validate_visibility_manifest(manifest: Mapping[str, Any], *, current_ordinal: int) -> list[str]:
    """Reject label fields, changed hashes, and non-predecessor ordinals."""

    errors: list[str] = []
    visible = list(manifest.get("visible_events", []))
    ordinals = [row.get("event_ordinal") for row in visible if isinstance(row, Mapping)]
    if manifest.get("current_event_ordinal") != current_ordinal:
        errors.append("current_ordinal_mismatch")
    if manifest.get("visible_event_ordinals") != list(range(current_ordinal)):
        errors.append("visible_ordinal_frontier")
    if ordinals != list(range(current_ordinal)) or any(
        not isinstance(ordinal, int) or ordinal >= current_ordinal for ordinal in ordinals
    ):
        errors.append("current_or_future_ordinal")
    forbidden = sorted(_nested_keys(manifest) & FORBIDDEN_VISIBILITY_FIELDS)
    if forbidden:
        errors.append(f"label_field_visible:{forbidden}")
    payload = {key: deepcopy(value) for key, value in manifest.items() if key != "manifest_hash"}
    if manifest.get("manifest_hash") != sha256_json(payload):
        errors.append("manifest_hash")
    return errors


def public_event_rows_for_tests() -> list[JsonDict]:
    """Return a small public chronology used by direct leakage mutations."""

    return [
        {
            "event_id": f"test_event_{ordinal:03d}",
            "event_ordinal": ordinal,
            "shift_id": SHIFT_IDS[ordinal // 6],
            "shift_ordinal": ordinal // 6,
            "block_event_ordinal": ordinal % 6,
        }
        for ordinal in range(EXPECTED_EVENT_COUNT)
    ]


def _visibility_rows(
    stream_rows: Sequence[Mapping[str, Any]], raw_root: Path
) -> tuple[list[JsonDict], list[JsonDict]]:
    """Seal all predecessor views and run current/future label mutations."""

    receipts: list[JsonDict] = []
    leakage_rows: list[JsonDict] = []
    for ordinal in range(EXPECTED_EVENT_COUNT):
        manifest = build_visibility_manifest(stream_rows, ordinal)
        relative = Path("visibility") / f"event_{ordinal:03d}.json"
        manifest_sha = write_immutable_json(raw_root / relative, manifest)
        clean_errors = validate_visibility_manifest(manifest, current_ordinal=ordinal)
        current_mutation = deepcopy(manifest)
        current_mutation["visible_events"].append(
            {"event_ordinal": ordinal, "current_label": "equivalent"}
        )
        future_ordinal = min(ordinal + 1, EXPECTED_EVENT_COUNT)
        future_mutation = deepcopy(manifest)
        future_mutation["visible_events"].append(
            {"event_ordinal": future_ordinal, "future_labels": ["non_equivalent"]}
        )
        current_errors = validate_visibility_manifest(current_mutation, current_ordinal=ordinal)
        future_errors = validate_visibility_manifest(future_mutation, current_ordinal=ordinal)
        visible_keys = _nested_keys(manifest) & FORBIDDEN_VISIBILITY_FIELDS
        receipts.append(
            {
                "event_ordinal": ordinal,
                "manifest_path": str(relative),
                "manifest_sha256": manifest_sha,
                "manifest_hash": manifest["manifest_hash"],
                "visible_event_ordinals": list(range(ordinal)),
                "current_event_visible": ordinal in manifest["visible_event_ordinals"],
                "future_event_visible": any(
                    value > ordinal for value in manifest["visible_event_ordinals"]
                ),
                "label_fields_visible": sorted(visible_keys),
                "manifest_replays": not clean_errors,
                "terminal": True,
            }
        )
        leakage_rows.append(
            {
                "event_ordinal": ordinal,
                "check": "current_and_future_labels_rejected",
                "clean_manifest_errors": clean_errors,
                "current_label_mutation_rejected": bool(current_errors),
                "future_label_mutation_rejected": bool(future_errors),
                "passed": not clean_errors and bool(current_errors) and bool(future_errors),
            }
        )
    return receipts, leakage_rows


def payload_checksum(artifact: Mapping[str, Any]) -> str:
    """Hash scientific content while excluding wall time and this digest."""

    payload = {
        key: deepcopy(value)
        for key, value in artifact.items()
        if key not in {"duration_s", "reproducibility_checksum"}
    }
    return sha256_json(payload)


def _empty_rows() -> JsonDict:
    """Return every required row family for a schema-complete blocked result."""

    return {
        field: []
        for field in (
            "source_disjointness_rows",
            "rows",
            "per_event_results",
            "per_candidate_rows",
            "shift_block_rows",
            "fault_family_rows",
            "tie_group_rows",
            "recurrence_rows",
            "retention_anchor_rows",
            "held_future_window_rows",
            "visibility_manifest_rows",
            "future_label_leakage_rows",
            "authority_agreement_rows",
        )
    }


def build_blocked_artifact(
    *,
    repo_root: Path,
    raw_root: Path,
    preconditions: Sequence[Mapping[str, Any]],
    duration_s: float,
) -> JsonDict:
    """Build a full blocked schema without writing stream or label files."""

    del raw_root
    artifact: JsonDict = {
        "schema": SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "run_date": RUN_DATE,
        "field_principles": deepcopy(FIELD_PRINCIPLES),
        "preconditions_checked": deepcopy(list(preconditions)),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "duration_s": duration_s,
        "source_artifact_hashes": source_artifact_hashes(repo_root),
        **_empty_rows(),
        "stream_manifest": {},
        "stream_hash": None,
        "sealed_label_path": "sealed_labels.jsonl",
        "sealed_label_hash": None,
        "expected_event_count": EXPECTED_EVENT_COUNT,
        "observed_event_count": 0,
        "headroom_event_count": 0,
        "all_valid_event_count": 0,
        "all_invalid_event_count": 0,
        "chronological_stream_ready_score": 0,
        "continuous_self_learning_fixture": True,
        "random_seed": RANDOM_SEED,
        "reproducibility_checksum": "",
        "gate_check_summary": gate_summary(preconditions),
        "verifier_is_oracle": True,
        "verdict_class": "blocked",
        "honest_verdict": "blocked_chronological_constraint_stream",
    }
    artifact["reproducibility_checksum"] = payload_checksum(artifact)
    return artifact


def build_artifact(repo_root: Path, raw_root: Path, *, duration_s: float) -> JsonDict:
    """Construct, certify, seal, and reduce the complete 24-event stream."""

    preconditions = collect_preconditions(repo_root, raw_root)
    if gate_summary(preconditions):
        return build_blocked_artifact(
            repo_root=repo_root,
            raw_root=raw_root,
            preconditions=preconditions,
            duration_s=duration_s,
        )
    source_pairs = unused_source_pairs(repo_root)
    event_plan = build_event_plan(selected_source_rows(repo_root))
    attempt_rows: list[JsonDict] = []
    candidate_rows: list[JsonDict] = []
    agreement_rows: list[JsonDict] = []
    event_rows: list[JsonDict] = []
    label_rows: list[JsonDict] = []
    stream_rows: list[JsonDict] = []

    for event in event_plan:
        base = source_pairs[str(event["source_pair_id"])]
        specs = _candidate_specs(event, base)
        prepared: list[tuple[str, JsonDict, JsonDict]] = []
        for variant_ordinal, spec in enumerate(specs):
            payload = _candidate_payload(spec["pair"])
            payload_hash = sha256_json(payload)
            candidate_id = (
                "candidate_"
                + sha256_json(
                    {
                        "seed": RANDOM_SEED,
                        "event_id": event["event_id"],
                        "variant_ordinal": variant_ordinal,
                        "payload_hash": payload_hash,
                    }
                ).removeprefix("sha256:")[:20]
            )
            prepared.append(
                (
                    payload_hash,
                    {
                        "candidate_id": candidate_id,
                        "payload": payload,
                        "payload_hash": payload_hash,
                        "fault_family": spec.get("fault_family"),
                        "relation": spec["relation"],
                    },
                    spec,
                )
            )
        prepared.sort(key=lambda item: sha256_json({"seed": RANDOM_SEED, "hash": item[0]}))
        labels: list[str | None] = []
        event_candidate_rows: list[JsonDict] = []
        for pair_position, (_, prepared_row, spec) in enumerate(prepared):
            candidate_id = str(prepared_row["candidate_id"])
            attempt = certify_candidate_attempt(
                spec["pair"], candidate_id, expected_relation=str(spec["relation"])
            )
            attempt_rows.append(_attempt_evidence_row(attempt, event))
            agreement_rows.append(_authority_agreement_row(attempt, event))
            labels.append(attempt["certified_relation"])
            public_candidate = {
                "event_id": event["event_id"],
                "event_ordinal": event["event_ordinal"],
                "candidate_id": candidate_id,
                "pair_position": pair_position,
                "fault_family": prepared_row["fault_family"],
                "serialized_candidate": canonical_json(prepared_row["payload"]).decode(),
                "candidate_payload_hash": prepared_row["payload_hash"],
            }
            event_candidate_rows.append(public_candidate)
            candidate_rows.append(public_candidate)
        label_payload = {
            "event_id": event["event_id"],
            "event_ordinal": event["event_ordinal"],
            "candidate_ids": [row["candidate_id"] for row in event_candidate_rows],
            "exact_labels": labels,
        }
        label_payload["label_commitment_hash"] = sha256_json(label_payload)
        label_rows.append(label_payload)
        stream_row = {
            key: deepcopy(event[key])
            for key in (
                "event_id",
                "event_ordinal",
                "shift_id",
                "shift_ordinal",
                "block_event_ordinal",
                "source_group_id",
                "source_pair_id",
                "formulation_family",
                "event_type",
                "candidate_fault_families",
            )
        }
        stream_row.update(
            {
                "candidate_ids": [row["candidate_id"] for row in event_candidate_rows],
                "candidates": event_candidate_rows,
                "label_commitment_hash": label_payload["label_commitment_hash"],
                "held_future_checkpoint_id": f"held_future_after_{event['event_ordinal']:03d}",
                "terminal": all(row["terminal"] for row in attempt_rows[-2:]),
            }
        )
        stream_rows.append(stream_row)
        event_rows.append(
            {
                "event_id": event["event_id"],
                "event_ordinal": event["event_ordinal"],
                "shift_id": event["shift_id"],
                "source_group_id": event["source_group_id"],
                "source_pair_id": event["source_pair_id"],
                "event_type": event["event_type"],
                "candidate_ids": stream_row["candidate_ids"],
                "candidate_labels_tied": len(set(labels)) == 1,
                "exact_group_advantage": int(len(set(labels)) > 1),
                "label_commitment_hash": label_payload["label_commitment_hash"],
                "terminal": stream_row["terminal"],
            }
        )

    source_disjointness = _source_disjointness_rows(stream_rows)
    shift_rows = _shift_block_rows(stream_rows)
    fault_rows = _fault_family_rows(stream_rows)
    recurrence_rows = _recurrence_rows(stream_rows)
    retention_rows = _retention_anchor_rows(recurrence_rows)
    held_rows = _held_future_window_rows(stream_rows, label_rows)
    tie_rows = [
        {
            "event_id": row["event_id"],
            "event_ordinal": row["event_ordinal"],
            "event_type": row["event_type"],
            "exact_group_advantage": row["exact_group_advantage"],
            "required_later_action": "no_update",
            "terminal": row["terminal"],
        }
        for row in event_rows
        if row["event_type"] in {"all_valid", "all_invalid"}
    ]
    stream_path = raw_root / "chronological_stream.jsonl"
    label_path = raw_root / "sealed_labels.jsonl"
    stream_hash = write_immutable_jsonl(stream_path, stream_rows)
    sealed_label_hash = write_immutable_jsonl(label_path, label_rows)
    visibility_rows, leakage_rows = _visibility_rows(stream_rows, raw_root)
    stream_manifest = {
        "stream_path": "chronological_stream.jsonl",
        "stream_format": "canonical_jsonl",
        "immutable": True,
        "labels_stored_separately": True,
        "event_count": len(stream_rows),
        "candidate_count": len(candidate_rows),
        "shift_ids": list(SHIFT_IDS),
        "event_order_hash": sha256_json([row["event_id"] for row in stream_rows]),
        "candidate_order_hash": sha256_json(
            [
                {"event_id": row["event_id"], "candidate_ids": row["candidate_ids"]}
                for row in stream_rows
            ]
        ),
        "shift_block_hash": sha256_json(shift_rows),
        "retention_anchor_hash": sha256_json(retention_rows),
        "held_future_window_hash": sha256_json(held_rows),
    }
    headroom_count = sum(row["event_type"] == "headroom" for row in event_rows)
    all_valid_count = sum(row["event_type"] == "all_valid" for row in event_rows)
    all_invalid_count = sum(row["event_type"] == "all_invalid" for row in event_rows)
    scientific_checks = [
        gate_check("observed_event_count", EXPECTED_EVENT_COUNT, len(event_rows)),
        gate_check("headroom_event_count", 16, headroom_count),
        gate_check("all_valid_event_count", 4, all_valid_count),
        gate_check("all_invalid_event_count", 4, all_invalid_count),
        gate_check(
            "all_candidates_accepted",
            True,
            all(row["accepted"] and row["terminal"] for row in attempt_rows),
        ),
        gate_check(
            "all_authorities_agree",
            True,
            all(row["all_required_agreement"] and row["terminal"] for row in agreement_rows),
        ),
        gate_check(
            "source_overlap_zero",
            0,
            source_disjointness[0]["overlap_count"],
        ),
        gate_check(
            "visibility_manifests_replay",
            True,
            all(row["manifest_replays"] for row in visibility_rows),
        ),
        gate_check(
            "future_labels_rejected",
            True,
            all(row["passed"] for row in leakage_rows),
        ),
        gate_check(
            "tie_groups_zero_advantage",
            True,
            len(tie_rows) == 8
            and all(
                row["exact_group_advantage"] == 0 and row["required_later_action"] == "no_update"
                for row in tie_rows
            ),
        ),
        gate_check(
            "recurrence_complete",
            True,
            len(recurrence_rows) == 6
            and all(
                row["source_group_is_new"]
                and row["fault_family_signature_matches"]
                and row["event_type_matches"]
                for row in recurrence_rows
            ),
        ),
    ]
    ready = not gate_summary([*preconditions, *scientific_checks])
    artifact: JsonDict = {
        "schema": SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "run_date": RUN_DATE,
        "field_principles": deepcopy(FIELD_PRINCIPLES),
        "preconditions_checked": preconditions,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "duration_s": duration_s,
        "source_artifact_hashes": source_artifact_hashes(repo_root),
        "source_disjointness_rows": source_disjointness,
        "stream_manifest": stream_manifest,
        "stream_hash": stream_hash,
        "sealed_label_path": "sealed_labels.jsonl",
        "sealed_label_hash": sealed_label_hash,
        "rows": attempt_rows,
        "per_event_results": event_rows,
        "per_candidate_rows": candidate_rows,
        "shift_block_rows": shift_rows,
        "fault_family_rows": fault_rows,
        "tie_group_rows": tie_rows,
        "recurrence_rows": recurrence_rows,
        "retention_anchor_rows": retention_rows,
        "held_future_window_rows": held_rows,
        "visibility_manifest_rows": visibility_rows,
        "future_label_leakage_rows": leakage_rows,
        "authority_agreement_rows": agreement_rows,
        "expected_event_count": EXPECTED_EVENT_COUNT,
        "observed_event_count": len(event_rows),
        "headroom_event_count": headroom_count,
        "all_valid_event_count": all_valid_count,
        "all_invalid_event_count": all_invalid_count,
        "chronological_stream_ready_score": int(ready),
        "continuous_self_learning_fixture": True,
        "random_seed": RANDOM_SEED,
        "reproducibility_checksum": "",
        "gate_check_summary": gate_summary([*preconditions, *scientific_checks]),
        "verifier_is_oracle": True,
        "verdict_class": "circular_positive" if ready else "disqualified",
        "honest_verdict": (
            "complete_circular_chronological_constraint_stream"
            if ready
            else "complete_disqualified_chronological_constraint_stream"
        ),
    }
    artifact["reproducibility_checksum"] = payload_checksum(artifact)
    return artifact


def _files_replay(artifact: Mapping[str, Any], raw_root: Path) -> tuple[bool, list[JsonDict]]:
    """Replay stream, labels, and all visibility manifest bytes."""

    stream_path = raw_root / str(artifact["stream_manifest"].get("stream_path", ""))
    label_path = raw_root / str(artifact["sealed_label_path"])
    if not stream_path.is_file() or not label_path.is_file():
        return False, []
    stream_rows = read_jsonl(stream_path)
    if sha256_path(stream_path) != artifact["stream_hash"]:
        return False, stream_rows
    if sha256_path(label_path) != artifact["sealed_label_hash"]:
        return False, stream_rows
    for receipt in artifact["visibility_manifest_rows"]:
        path = raw_root / str(receipt["manifest_path"])
        if not path.is_file() or sha256_path(path) != receipt["manifest_sha256"]:
            return False, stream_rows
        manifest = _read_json(path)
        if validate_visibility_manifest(manifest, current_ordinal=int(receipt["event_ordinal"])):
            return False, stream_rows
    return True, stream_rows


def validate_artifact(artifact: Mapping[str, Any], *, raw_root: Path) -> list[str]:
    """Recompute the schema, sealed files, row gates, score, and verdict."""

    errors: list[str] = []
    missing = sorted(set(REQUIRED_ARTIFACT_FIELDS) - set(artifact))
    if missing:
        return [f"missing_required_fields:{missing}"]
    if set(artifact["field_principles"]) != set(REQUIRED_ARTIFACT_FIELDS):
        errors.append("field_principles")
    score = artifact["chronological_stream_ready_score"]
    if type(score) is not int or score not in {0, 1}:
        errors.append("bare_score")
    if artifact["inference_substrate"] != INFERENCE_SUBSTRATE:
        errors.append("inference_substrate")
    if artifact["continuous_self_learning_fixture"] is not True:
        errors.append("fixture_declaration")
    if artifact["verifier_is_oracle"] is not True:
        errors.append("oracle_declaration")
    if artifact["verdict_class"] not in VERDICT_CLASSES:
        errors.append("verdict_class")
    if artifact["verdict_class"] == "blocked":
        if score != 0:
            errors.append("blocked_score")
        if not artifact["gate_check_summary"]:
            errors.append("blocked_gate_summary")
        if not str(artifact["honest_verdict"]).startswith(
            "blocked_chronological_constraint_stream"
        ):
            errors.append("blocked_verdict")
    else:
        files_replay, stream_rows = _files_replay(artifact, raw_root)
        if not files_replay:
            errors.append("immutable_file_replay")
        if _nested_keys(stream_rows) & FORBIDDEN_STREAM_LABEL_FIELDS:
            errors.append("stream_label_sealing")
        label_path = raw_root / str(artifact["sealed_label_path"])
        label_rows = read_jsonl(label_path) if label_path.is_file() else []
        event_rows = list(artifact["per_event_results"])
        attempt_rows = list(artifact["rows"])
        agreement_rows = list(artifact["authority_agreement_rows"])
        if (
            artifact["expected_event_count"] != EXPECTED_EVENT_COUNT
            or artifact["observed_event_count"] != len(event_rows)
            or len(event_rows) != EXPECTED_EVENT_COUNT
            or len(stream_rows) != EXPECTED_EVENT_COUNT
            or len(label_rows) != EXPECTED_EVENT_COUNT
        ):
            errors.append("event_count")
        counts = {
            event_type: sum(row.get("event_type") == event_type for row in event_rows)
            for event_type in {"headroom", "all_valid", "all_invalid"}
        }
        if (
            artifact["headroom_event_count"] != counts["headroom"]
            or artifact["all_valid_event_count"] != counts["all_valid"]
            or artifact["all_invalid_event_count"] != counts["all_invalid"]
            or counts != {"headroom": 16, "all_valid": 4, "all_invalid": 4}
        ):
            errors.append("event_type_counts")
        if len(attempt_rows) != 48 or not all(
            row.get("accepted") is True and row.get("terminal") is True for row in attempt_rows
        ):
            errors.append("candidate_attempts")
        if len(artifact["per_candidate_rows"]) != 48:
            errors.append("candidate_count")
        if len(agreement_rows) != 48 or not all(
            row.get("all_required_agreement") is True and row.get("terminal") is True
            for row in agreement_rows
        ):
            errors.append("authority_agreement")
        disjointness = _source_disjointness_rows(stream_rows)
        if artifact["source_disjointness_rows"] != disjointness or any(
            row["overlap_count"] for row in disjointness
        ):
            errors.append("source_disjointness")
        shift_rows = _shift_block_rows(stream_rows)
        if artifact["shift_block_rows"] != shift_rows or any(
            row["event_count"] != 6 for row in shift_rows
        ):
            errors.append("shift_blocks")
        if artifact["fault_family_rows"] != _fault_family_rows(stream_rows):
            errors.append("fault_family_rows")
        recurrence_rows = _recurrence_rows(stream_rows)
        recurrence_valid = len(recurrence_rows) == 6 and all(
            row["source_group_is_new"]
            and row["fault_family_signature_matches"]
            and row["event_type_matches"]
            for row in recurrence_rows
        )
        if artifact["recurrence_rows"] != recurrence_rows or not recurrence_valid:
            errors.append("recurrence")
        if artifact["retention_anchor_rows"] != _retention_anchor_rows(recurrence_rows):
            errors.append("retention_anchors")
        held_rows = _held_future_window_rows(stream_rows, label_rows)
        if artifact["held_future_window_rows"] != held_rows:
            errors.append("held_future_windows")
        tie_rows = list(artifact["tie_group_rows"])
        tie_valid = len(tie_rows) == 8 and all(
            row.get("exact_group_advantage") == 0
            and row.get("required_later_action") == "no_update"
            and row.get("terminal") is True
            for row in tie_rows
        )
        if not tie_valid:
            errors.append("tie_group")
        visibility_valid = len(
            artifact["visibility_manifest_rows"]
        ) == EXPECTED_EVENT_COUNT and all(
            row.get("manifest_replays") is True
            and row.get("visible_event_ordinals") == list(range(row["event_ordinal"]))
            and row.get("current_event_visible") is False
            and row.get("future_event_visible") is False
            and row.get("label_fields_visible") == []
            for row in artifact["visibility_manifest_rows"]
        )
        if not visibility_valid or not files_replay:
            errors.append("visibility_manifest")
        leakage_valid = len(artifact["future_label_leakage_rows"]) == EXPECTED_EVENT_COUNT and all(
            row.get("passed") is True
            and row.get("current_label_mutation_rejected") is True
            and row.get("future_label_mutation_rejected") is True
            for row in artifact["future_label_leakage_rows"]
        )
        if not leakage_valid:
            errors.append("future_label_leakage")
        manifest = artifact["stream_manifest"]
        expected_manifest = {
            "stream_path": "chronological_stream.jsonl",
            "stream_format": "canonical_jsonl",
            "immutable": True,
            "labels_stored_separately": True,
            "event_count": len(stream_rows),
            "candidate_count": len(artifact["per_candidate_rows"]),
            "shift_ids": list(SHIFT_IDS),
            "event_order_hash": sha256_json([row["event_id"] for row in stream_rows]),
            "candidate_order_hash": sha256_json(
                [
                    {"event_id": row["event_id"], "candidate_ids": row["candidate_ids"]}
                    for row in stream_rows
                ]
            ),
            "shift_block_hash": sha256_json(shift_rows),
            "retention_anchor_hash": sha256_json(artifact["retention_anchor_rows"]),
            "held_future_window_hash": sha256_json(held_rows),
        }
        if manifest != expected_manifest:
            errors.append("stream_manifest")
        ready = files_replay and not any(
            name in errors
            for name in (
                "event_count",
                "event_type_counts",
                "candidate_attempts",
                "candidate_count",
                "authority_agreement",
                "source_disjointness",
                "shift_blocks",
                "fault_family_rows",
                "recurrence",
                "retention_anchors",
                "held_future_windows",
                "tie_group",
                "visibility_manifest",
                "future_label_leakage",
                "stream_manifest",
                "stream_label_sealing",
            )
        )
        if score != int(ready):
            errors.append("chronological_stream_ready_score")
        expected_class = "circular_positive" if ready else "disqualified"
        if artifact["verdict_class"] != expected_class:
            errors.append("verdict_class_consistency")
        expected_prefix = "complete_circular_" if ready else "complete_disqualified_"
        if not str(artifact["honest_verdict"]).startswith(expected_prefix):
            errors.append("honest_verdict_consistency")
    if artifact["reproducibility_checksum"] != payload_checksum(artifact):
        errors.append("checksum")
    return errors


def run(*, repo_root: Path, result_path: Path, raw_root: Path, run_date: str) -> JsonDict:
    """Build, validate, and write one fixed-date terminal artifact."""

    if run_date != RUN_DATE:
        raise ValueError(f"run_date_mismatch:expected={RUN_DATE}:observed={run_date}")
    started = time.perf_counter()
    artifact = build_artifact(repo_root, raw_root, duration_s=0.0)
    artifact["duration_s"] = round(time.perf_counter() - started, 6)
    artifact["reproducibility_checksum"] = payload_checksum(artifact)
    errors = validate_artifact(artifact, raw_root=raw_root)
    if errors:
        raise ValueError("artifact_validation_failed:" + ",".join(errors))
    write_json_atomic(result_path, artifact)
    return artifact


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - command wrapper.
    """Expose the fixed-date command used by the research conductor."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", required=True)
    args = parser.parse_args(argv)
    repo_root = Path(__file__).resolve().parents[2]
    artifact = run(
        repo_root=repo_root,
        result_path=repo_root / RESULT_PATH,
        raw_root=repo_root / RAW_ROOT,
        run_date=args.date,
    )
    print(json.dumps({"verdict": artifact["honest_verdict"]}, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover - module command surface.
    raise SystemExit(main())
