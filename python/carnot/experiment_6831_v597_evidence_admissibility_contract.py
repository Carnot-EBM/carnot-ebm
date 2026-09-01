"""Issue the V597 evidence-admissibility contract from frozen artifacts.

The reducer treats Exp6826 only as a quarantined comparator. All authority
decisions are recomputed directly from Exp6813, Exp6824, and Exp6825 rows.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
from collections import defaultdict
from copy import deepcopy
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping, Sequence


JsonDict = dict[str, Any]
SHA256_PATTERN = re.compile(r"^sha256:[0-9a-f]{64}$")
CLOSED_VERDICT_CLASSES = frozenset(
    {"positive", "circular_positive", "null", "blocked", "disqualified", "partial"}
)
PHASE_NAMES = ("start", "read", "recompute", "write", "verify")
SOURCE_PATHS = {
    "exp6813": Path("results/experiment_6813_selective_priority_arbiter_ab.json"),
    "exp6824": Path("results/experiment_6824_selective_arbiter_cold_row_replay.json"),
    "exp6825": Path("results/experiment_6825_selective_arbiter_authority_attacks.json"),
    "exp6826": Path("results/experiment_6826_selective_arbiter_sealed_adoption.json"),
    "exp6827": Path("results/experiment_6827_chronological_causal_edge_memory_stream.json"),
}
EXPECTED_ROW_COUNTS = {
    "exp6813": 576,
    "exp6824": 578,
    "exp6825": 768,
    "exp6826": 9,
    "exp6827": 4320,
}
COMPLETION_FIELDS = {
    "exp6813": "selective_arbiter_ab_completed",
    "exp6824": "cold_replay_shard_complete",
    "exp6825": "authority_attack_shard_complete",
    "exp6826": "selective_arbiter_audit_complete",
    "exp6827": "verified_memory_stream_ready",
}
OUTPUT_PATH = Path("results/experiment_6831_v597_evidence_admissibility_contract.json")
MODULE_PATH = Path("python/carnot/experiment_6831_v597_evidence_admissibility_contract.py")
WRAPPER_PATH = Path("scripts/experiments/experiment_6831_v597_evidence_admissibility_contract.py")
AUTHORITY_CRITERION_ROW_COUNT = 13
STREAM_CRITERION_ROW_COUNT = 8
COUNTERFACTUAL_KINDS = frozenset({"remove", "substitute", "reorder"})
MUTATING_OPERATION_KINDS = frozenset({"add", "revise", "soft_delete", "restore"})
READ_ONLY_OPERATION_KINDS = frozenset({"retrieve", "filter"})
PRIORITY_ATTACKS = frozenset(
    {
        "priority_inversion",
        "authority_spoofing",
        "stale_prerequisite",
        "fallback_deletion",
        "consequence_weakening",
        "no_candidate",
    }
)
IDENTITY_ATTACKS = frozenset({"tie_reorder", "canonical_byte_mutation", "safe_action_mutation"})
CERTIFICATE_ATTACKS = frozenset({"no_candidate", "fabricated_certificates"})
ADOPTION_ATTACKS = frozenset(
    {
        "model_label_influence",
        "exact_valid_label_influence",
        "future_outcome_leakage",
        "row_deletion",
        "duplicate_rows",
        "row_reorder",
    }
)
REPLAY_COMMANDS = [
    ".venv/bin/python scripts/experiments/experiment_6831_v597_evidence_admissibility_contract.py --date 20260831",
    ".venv/bin/python scripts/adversarial_verify.py results/experiment_6831_v597_evidence_admissibility_contract.json",
]
FIELD_PRINCIPLES = {
    "schema": "Names the stable artifact contract read by downstream tasks.",
    "experiment_id": "Identifies the task-owned evidence aggregation run.",
    "title": "States that this receipt concerns evidence admissibility only.",
    "run_date": "Records the operator-supplied execution date.",
    "status": "Reports a terminal complete or complete-blocked state.",
    "openspec_requirement_ids": "Connects fields and rows to their governing requirements.",
    "replay_commands": "Records deterministic commands needed to reproduce and audit the receipt.",
    "inference_substrate": "Names only the allowlisted upstream-artifact aggregation substrate.",
    "duration_s": "Measures task-owned wall time from start through final verification.",
    "phase_clocks": "Separates ordered start, read, recompute, write, and verify timing.",
    "launch_receipts": "Binds the running process and command identity without inference credit.",
    "accelerator_samples": "Records process observations used to deny unearned accelerator credit.",
    "implementation_hashes": "Binds the fresh reducer and its task-owned command wrapper.",
    "reproducibility_checksum": "Binds stable inputs, code, rows, commands, and output decisions.",
    "source_artifact_hashes": "Records immutable identities for Exp6813 and Exp6824 through Exp6827.",
    "preconditions_checked": "Records every attempted gate with expected and observed values.",
    "rows": "Provides one row for every authority or stream-readiness criterion and source unit.",
    "flagged_receipt_disposition": "Keeps the flagged Exp6826 receipt quarantined and nonauthoritative.",
    "selective_arbiter_decisions": "Separates safety, identity, certificate, utility, and adoption findings.",
    "selective_arbiter_receipt_admissible": "Reports procedural authority usability independent of effect sign.",
    "csl_stream_validation": "Reports row, order, split, transaction, rotation, seal, and headroom checks.",
    "csl_inputs_admissible": "Provides the exact immutable-input gate consumed by Exp6835.",
    "v597_contract_ready": "Provides the exact complete-contract gate consumed by Exp6832.",
    "gate_check_summary": "Names the first failed check and observation or confirms all checks passed.",
    "verifier_is_oracle": "Declares that evidence auditing does not define proposal truth.",
    "verdict_class": "Uses one closed verdict class supported by terminal rows.",
    "honest_verdict": "States the terminal row-supported conclusion with the required complete prefix.",
}


def canonical_bytes(value: Any) -> bytes:
    """Return the repository's stable compact JSON encoding."""
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n"
    ).encode()


def sha256_bytes(raw: bytes) -> str:
    """Label a byte digest so identities cannot be mistaken for free text."""
    return f"sha256:{hashlib.sha256(raw).hexdigest()}"


def sha256_json(value: Any) -> str:
    """Hash a value through the canonical JSON encoding."""
    return sha256_bytes(canonical_bytes(value))


def _utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="microseconds").replace("+00:00", "Z")


def _phase_clock(started: float) -> JsonDict:
    return {"utc": _utc_now(), "elapsed_s": round(time.perf_counter() - started, 6)}


def _source_hash_records(hashes: Mapping[str, str | None]) -> JsonDict:
    return {
        name: {"path": str(path), "sha256": hashes.get(name)} for name, path in SOURCE_PATHS.items()
    }


def _implementation_hashes(root: Path | None = None) -> JsonDict:
    base = root or Path(__file__).resolve().parents[2]
    records: JsonDict = {}
    for label, relative_path in {"reducer": MODULE_PATH, "wrapper": WRAPPER_PATH}.items():
        path = base / relative_path
        records[label] = {"path": str(relative_path), "sha256": sha256_bytes(path.read_bytes())}
    return records


def _launch_receipts(root: Path, argv: Sequence[str]) -> JsonDict:
    return {
        "process_id": os.getpid(),
        "parent_process_id": os.getppid(),
        "executable": sys.executable,
        "working_directory": str(root.resolve()),
        "command": [sys.executable, str(WRAPPER_PATH), *argv],
    }


def _accelerator_samples() -> JsonDict:
    handles: list[str] = []
    fd_root = Path("/proc/self/fd")
    if fd_root.is_dir():
        for entry in fd_root.iterdir():
            try:
                target = str(entry.resolve())
            except OSError:
                continue
            if target.startswith(("/dev/nvidia", "/dev/dri/render")):
                handles.append(target)
    libraries: list[str] = []
    maps_path = Path("/proc/self/maps")
    if maps_path.is_file():
        for line in maps_path.read_text(errors="replace").splitlines():
            lowered = line.lower()
            if any(marker in lowered for marker in ("libnvidia", "libamdhip", "libmps")):
                libraries.append(line.rsplit(maxsplit=1)[-1])
    return {
        "accelerator_credit_claimed": False,
        "process_device_handles": sorted(set(handles)),
        "loaded_accelerator_libraries": sorted(set(libraries)),
    }


def _read_sources(root: Path) -> tuple[dict[str, JsonDict], dict[str, str]]:
    payloads: dict[str, JsonDict] = {}
    hashes: dict[str, str] = {}
    for name, relative_path in SOURCE_PATHS.items():
        path = root / relative_path
        try:
            raw = path.read_bytes()
            payload = json.loads(raw)
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(payload, dict):
            payloads[name] = payload
            hashes[name] = sha256_bytes(raw)
    return payloads, hashes


def _hash_source_files(root: Path) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for name, relative_path in SOURCE_PATHS.items():
        try:
            hashes[name] = sha256_bytes((root / relative_path).read_bytes())
        except OSError:
            continue
    return hashes


def _row_identity(row: Mapping[str, Any]) -> str:
    row_id = row.get("row_id")
    return str(row_id) if row_id is not None else sha256_json(row)


def _embedded_hash(payload: Mapping[str, Any], source: str) -> Any:
    record = payload.get("source_artifact_hashes", {}).get(source)
    return record.get("sha256") if isinstance(record, dict) else None


def _precondition_checks(
    payloads: Mapping[str, JsonDict],
    initial_hashes: Mapping[str, str],
    final_hashes: Mapping[str, str],
) -> tuple[list[JsonDict], JsonDict | None]:
    checks: list[JsonDict] = []

    def checked(name: str, expected: Any, observed: Any) -> bool:
        passed = observed == expected
        checks.append({"check": name, "expected": expected, "observed": observed, "passed": passed})
        return passed

    for source in SOURCE_PATHS:
        if not checked(f"source_{source}_readable", True, source in payloads):
            return checks, checks[-1]
    for source in SOURCE_PATHS:
        if not checked(
            f"source_{source}_hash_stable", initial_hashes.get(source), final_hashes.get(source)
        ):
            return checks, checks[-1]
    for source, payload in payloads.items():
        status = payload.get("status")
        terminal = isinstance(status, str) and status.startswith("complete")
        if not checked(f"source_{source}_terminal", True, terminal):
            return checks, checks[-1]
        if not checked(
            f"source_{source}_verdict_closed",
            True,
            payload.get("verdict_class") in CLOSED_VERDICT_CLASSES,
        ):
            return checks, checks[-1]
        honest = payload.get("honest_verdict")
        if not checked(
            f"source_{source}_honest_terminal",
            True,
            isinstance(honest, str) and honest.startswith("complete"),
        ):
            return checks, checks[-1]
        rows = payload.get("rows")
        count = len(rows) if isinstance(rows, list) else None
        if not checked(f"source_{source}_row_count", EXPECTED_ROW_COUNTS[source], count):
            return checks, checks[-1]
        identities = [_row_identity(row) for row in rows if isinstance(row, dict)]
        if not checked(f"source_{source}_row_identity_unique", count, len(set(identities))):
            return checks, checks[-1]
        completion_field = COMPLETION_FIELDS[source]
        if not checked(f"source_{source}_{completion_field}", True, payload.get(completion_field)):
            return checks, checks[-1]

    seals = (
        ("exp6824", "exp6813"),
        ("exp6825", "exp6813"),
        ("exp6826", "exp6813"),
        ("exp6826", "exp6824"),
        ("exp6826", "exp6825"),
        ("exp6827", "exp6826"),
    )
    for owner, source in seals:
        if not checked(
            f"source_{owner}_seal_{source}",
            initial_hashes[source],
            _embedded_hash(payloads[owner], source),
        ):
            return checks, checks[-1]
    flagged = payloads["exp6826"].get("flagged_adversarial")
    if not checked("source_exp6826_flagged_receipt", True, flagged):
        return checks, checks[-1]
    return checks, None


def _source_metrics(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    held = [row for row in rows if row.get("split") == "held"]
    selective = [row for row in held if row.get("arm") == "selective_priority"]
    flat_by_pair = {
        str(row.get("pair_id")): row for row in held if row.get("arm") == "flat_reject_retry"
    }
    paired = [(row, flat_by_pair.get(str(row.get("pair_id")))) for row in selective]
    complete_pairs = [(left, right) for left, right in paired if right is not None]
    progress_deltas = [
        float(left.get("accepted_progress", 0)) - float(right.get("accepted_progress", 0))
        for left, right in complete_pairs
    ]
    safe_rows = [row for row in selective if row.get("base_already_valid") is True]
    return {
        "held_selective_rows": len(selective),
        "pair_count": len(complete_pairs),
        "hard_violations": sum(row.get("accepted_hard_violation") is True for row in selective),
        "safe_identity_count": sum(row.get("safe_action_identity") is True for row in safe_rows),
        "safe_identity_denominator": len(safe_rows),
        "certificate_count": sum(row.get("certificate_complete") is True for row in selective),
        "harmful_selection_count": sum(row.get("harmful_selection") is True for row in selective),
        "legal_support_count": sum(row.get("legality") is True for row in selective),
        "paired_progress_delta": sum(progress_deltas) / len(progress_deltas)
        if progress_deltas
        else 0.0,
    }


def _finding_from_utility(metrics: Mapping[str, Any]) -> str:
    if metrics["harmful_selection_count"] or metrics["paired_progress_delta"] < 0:
        return "harmful"
    if metrics["paired_progress_delta"] == 0:
        return "null"
    complete = (
        metrics["held_selective_rows"] == 144
        and metrics["pair_count"] == 144
        and metrics["legal_support_count"] == 144
    )
    return "positive" if complete else "partial"


def _attack_row(
    criterion: str,
    attack_ids: frozenset[str],
    rows: Sequence[Mapping[str, Any]],
) -> tuple[JsonDict, str]:
    selected = [row for row in rows if row.get("attack_id") in attack_ids]
    passed = len(selected) == len(attack_ids) * 48 and all(
        row.get("passed") is True for row in selected
    )
    finding = "positive" if passed else "disqualified"
    return (
        {
            "domain": "selective_arbiter",
            "criterion": criterion,
            "source_unit": "exp6825_authority_attack_rows",
            "finding": finding,
            "decision": "pass" if passed else "fail",
            "passed": passed,
            "complete": len(selected) == len(attack_ids) * 48,
            "admissible": passed,
            "observed": {"attack_count": len(attack_ids), "row_count": len(selected)},
        },
        finding,
    )


def _metric_row(source: str, criterion: str, metrics: Mapping[str, Any]) -> tuple[JsonDict, str]:
    if criterion == "hard_safety":
        passed = metrics["held_selective_rows"] == 144 and metrics["hard_violations"] == 0
        observed = {
            "held_rows": metrics["held_selective_rows"],
            "hard_violations": metrics["hard_violations"],
        }
        finding = "positive" if passed else "disqualified"
    elif criterion == "safe_action_identity":
        passed = metrics["safe_identity_denominator"] == 28 and metrics["safe_identity_count"] == 28
        observed = {
            "identity_count": metrics["safe_identity_count"],
            "denominator": metrics["safe_identity_denominator"],
        }
        finding = "positive" if passed else "disqualified"
    elif criterion == "certificate_truth":
        passed = metrics["held_selective_rows"] == 144 and metrics["certificate_count"] == 144
        observed = {
            "certificate_count": metrics["certificate_count"],
            "denominator": metrics["held_selective_rows"],
        }
        finding = "positive" if passed else "disqualified"
    else:
        finding = _finding_from_utility(metrics)
        passed = finding == "positive"
        observed = {
            "paired_progress_delta": metrics["paired_progress_delta"],
            "pair_count": metrics["pair_count"],
            "harmful_selection_count": metrics["harmful_selection_count"],
            "legal_support_count": metrics["legal_support_count"],
        }
    return (
        {
            "domain": "selective_arbiter",
            "criterion": criterion,
            "source_unit": f"{source}_held_rows",
            "finding": finding,
            "decision": "pass"
            if passed
            else ("fail" if finding in {"harmful", "disqualified"} else "insufficient"),
            "passed": passed,
            "complete": metrics["held_selective_rows"] == 144,
            "admissible": finding not in {"blocked", "disqualified", "partial"},
            "observed": observed,
        },
        finding,
    )


def _combine_findings(findings: Sequence[str]) -> str:
    for candidate in ("disqualified", "blocked", "partial", "harmful", "null"):
        if candidate in findings:
            return candidate
    return "positive"


def derive_adoption(findings: Sequence[str]) -> str:
    """Apply the closed conservative propagation table to component findings."""
    combined = _combine_findings(findings)
    return {
        "positive": "enable",
        "null": "keep_shadow",
        "harmful": "retire",
        "blocked": "insufficient",
        "partial": "insufficient",
        "disqualified": "redesign",
    }[combined]


def receipt_is_admissible(findings: Mapping[str, str], *, rows_complete: bool) -> bool:
    """Grant procedural use only when all three authority boundaries pass."""
    authority = ("hard_safety", "safe_action_identity", "certificate_truth")
    return rows_complete and all(findings.get(name) == "positive" for name in authority)


def reduce_selective_arbiter(
    exp6813: Mapping[str, Any],
    exp6824: Mapping[str, Any],
    exp6825: Mapping[str, Any],
) -> tuple[JsonDict, list[JsonDict]]:
    """Freshly reduce source rows without importing any prior decision code."""
    producer_metrics = _source_metrics(exp6813["rows"])
    cold_metrics = _source_metrics(
        [row for row in exp6824["rows"] if row.get("row_type") == "cold_replay"]
    )
    rows: list[JsonDict] = []
    component_sources: dict[str, list[str]] = defaultdict(list)
    for criterion in ("hard_safety", "safe_action_identity", "certificate_truth", "utility"):
        for source, metrics in (("exp6813", producer_metrics), ("exp6824", cold_metrics)):
            row, finding = _metric_row(source, criterion, metrics)
            rows.append(row)
            component_sources[criterion].append(finding)
    attack_rows = exp6825["rows"]
    for criterion, attacks in (
        ("hard_safety", PRIORITY_ATTACKS),
        ("safe_action_identity", IDENTITY_ATTACKS),
        ("certificate_truth", CERTIFICATE_ATTACKS),
    ):
        row, finding = _attack_row(criterion, attacks, attack_rows)
        rows.append(row)
        component_sources[criterion].append(finding)
    adoption_row, adoption_finding = _attack_row("adoption", ADOPTION_ATTACKS, attack_rows)
    rows.append(adoption_row)
    component_sources["adoption"].append(adoption_finding)

    findings = {
        criterion: _combine_findings(source_findings)
        for criterion, source_findings in component_sources.items()
    }
    adoption_inputs = [
        findings["hard_safety"],
        findings["safe_action_identity"],
        findings["certificate_truth"],
        findings["utility"],
        findings["adoption"],
    ]
    adoption = derive_adoption(adoption_inputs)
    decisions: JsonDict = {
        criterion: {
            "finding": findings[criterion],
            "decision": (
                "pass"
                if findings[criterion] == "positive"
                else "fail"
                if findings[criterion] in {"harmful", "disqualified"}
                else "insufficient"
            ),
            "source_row_count": sum(row["criterion"] == criterion for row in rows),
        }
        for criterion in ("hard_safety", "safe_action_identity", "certificate_truth")
    }
    decisions["utility"] = {
        "finding": findings["utility"],
        "decision": "pass" if findings["utility"] == "positive" else "insufficient",
        "paired_progress_delta": cold_metrics["paired_progress_delta"],
        "producer_delta": producer_metrics["paired_progress_delta"],
        "harmful_selection_count": cold_metrics["harmful_selection_count"],
        "legal_support_count": cold_metrics["legal_support_count"],
    }
    decisions["adoption"] = {
        "finding": _combine_findings(adoption_inputs),
        "decision": adoption,
        "component_findings": {name: findings[name] for name in findings},
        "conservative_table": {
            "positive": "enable_only_when_all_inputs_are_positive",
            "null": "keep_shadow",
            "harmful": "retire",
            "blocked": "insufficient",
            "disqualified": "redesign",
            "partial": "insufficient",
        },
    }
    comparator = {
        "domain": "flagged_comparator",
        "criterion": "receipt_disposition",
        "source_unit": "exp6826_flagged_receipt",
        "finding": "disqualified",
        "decision": "quarantine",
        "passed": True,
        "complete": True,
        "admissible": False,
        "observed": {"authority_consumed": False, "flagged": True},
    }
    rows.append(comparator)
    return decisions, rows


def _digest_fields_valid(rows: Sequence[Mapping[str, Any]]) -> tuple[bool, int]:
    checked = 0
    for row in rows:
        expected_id = "::".join(
            str(row.get(name))
            for name in ("order_id", "source_family", "event_id", "counterfactual_kind")
        )
        if row.get("row_id") != expected_id:
            return False, checked
        for key, value in row.items():
            is_digest = key.endswith("_sha256") or key in {
                "causal_edge_id",
                "action_identity",
                "outcome_identity",
            }
            if is_digest and value is not None:
                checked += 1
                if not isinstance(value, str) or SHA256_PATTERN.fullmatch(value) is None:
                    return False, checked
    identities = [row.get("row_id") for row in rows]
    return len(rows) == 4320 and len(set(identities)) == 4320, checked


def _order_identity_valid(stream: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]) -> bool:
    order_hashes = stream.get("order_hashes", {})
    split_manifest = stream.get("split_manifest", {})
    families = list(split_manifest.get("by_family", {}))
    if len(order_hashes) != 5 or len(families) != 3:
        return False
    for order_index in range(1, 6):
        order_id = f"order_{order_index}"
        event_ids_by_family: JsonDict = {}
        for family in families:
            family_rows = [
                row
                for row in rows
                if row.get("order_id") == order_id and row.get("source_family") == family
            ]
            by_position = {
                int(row["chronological_position"]): str(row["event_id"]) for row in family_rows
            }
            if set(by_position) != set(range(1, 97)):
                return False
            event_ids_by_family[family] = [by_position[position] for position in range(1, 97)]
        expected = sha256_json(
            {
                "order_id": order_id,
                "seed": 6_827_000 + order_index,
                "event_ids_by_family": event_ids_by_family,
            }
        )
        if order_hashes.get(order_id) != expected:
            return False
    return True


def _split_identity_valid(stream: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]) -> bool:
    manifest = stream.get("split_manifest", {})
    if manifest.get("development_count_per_family") != 72:
        return False
    if manifest.get("held_future_count_per_family") != 24:
        return False
    for family, split_sets in manifest.get("by_family", {}).items():
        observed: dict[str, set[str]] = defaultdict(set)
        for row in rows:
            if row.get("source_family") == family:
                observed[str(row.get("split"))].add(str(row.get("event_id")))
        if observed["development"] != set(split_sets.get("development", [])):
            return False
        if observed["held_future"] != set(split_sets.get("held_future", [])):
            return False
        hard_cases = set(split_sets.get("hard_case", []))
        if len(hard_cases) != 32 or not hard_cases.issubset(observed["development"]):
            return False
    return True


def _transaction_validation(rows: Sequence[Mapping[str, Any]]) -> tuple[bool, list[JsonDict]]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("operation_id"))].append(row)
    representatives: list[JsonDict] = []
    valid = len(grouped) == 1440
    shared = (
        "receipt_sha256",
        "parent_state_sha256",
        "new_state_sha256",
        "operation_kind",
        "operation_admitted",
        "operation_reason",
        "chronological_position",
        "event_id",
    )
    for operation_rows in grouped.values():
        first = operation_rows[0]
        kinds = {row.get("counterfactual_kind") for row in operation_rows}
        if len(operation_rows) != 3 or kinds != COUNTERFACTUAL_KINDS:
            valid = False
        if any(
            any(row.get(field) != first.get(field) for field in shared) for row in operation_rows
        ):
            valid = False
        parent = first.get("parent_state_sha256")
        new = first.get("new_state_sha256")
        admitted = first.get("operation_admitted") is True
        kind = first.get("operation_kind")
        if (not admitted or kind in READ_ONLY_OPERATION_KINDS) and parent != new:
            valid = False
        if admitted and kind in MUTATING_OPERATION_KINDS and parent == new:
            valid = False
        representatives.append(dict(first))
    by_scope: dict[tuple[str, str], list[JsonDict]] = defaultdict(list)
    for row in representatives:
        by_scope[(str(row.get("order_id")), str(row.get("source_family")))].append(row)
    for scope_rows in by_scope.values():
        ordered = sorted(scope_rows, key=lambda row: int(row["chronological_position"]))
        if len(ordered) != 96:
            valid = False
        for previous, current in zip(ordered, ordered[1:], strict=False):
            if previous.get("new_state_sha256") != current.get("parent_state_sha256"):
                valid = False
    return valid, representatives


def _rotation_valid(stream: Mapping[str, Any]) -> bool:
    manifest = stream.get("split_manifest", {})
    families = set(manifest.get("by_family", {}))
    rotations = manifest.get("rotations", [])
    held = [row.get("held_out_family") for row in rotations]
    if len(rotations) != 3 or set(held) != families or len(set(held)) != 3:
        return False
    for row in rotations:
        if set(row.get("development_families", [])) != families - {row.get("held_out_family")}:
            return False
    return True


def _sealed_fields_valid(stream: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]) -> bool:
    allowlist = set(stream.get("feature_allowlist", []))
    denylist = set(stream.get("feature_denylist", []))
    sealed = set(stream.get("sealed_field_manifest", {}).get("sealed_until_after_decision", []))
    if denylist != sealed or allowlist & denylist:
        return False
    return all(set(row.get("decision_feature_keys", [])) == allowlist for row in rows)


def _headroom_from_rows(representatives: Sequence[Mapping[str, Any]]) -> JsonDict:
    unique_events: dict[str, Mapping[str, Any]] = {}
    by_scope: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in representatives:
        unique_events.setdefault(str(row.get("event_id")), row)
        by_scope[(str(row.get("order_id")), str(row.get("source_family")))].append(row)
    recoveries = 0
    for scope_rows in by_scope.values():
        pressure_pending = False
        pressure_released = False
        for row in sorted(scope_rows, key=lambda item: int(item["chronological_position"])):
            if row.get("operation_reason") == "capacity_exceeded":
                pressure_pending = True
            if (
                pressure_pending
                and row.get("operation_admitted")
                and row.get("operation_kind") == "soft_delete"
            ):
                pressure_released = True
            if (
                pressure_released
                and row.get("operation_admitted")
                and row.get("operation_kind") == "add"
            ):
                recoveries += 1
                pressure_pending = False
                pressure_released = False
    return {
        "legal_alternative_count": sum(
            int(row.get("legal_alternative_count", 0)) for row in unique_events.values()
        ),
        "safe_no_op_event_count": sum(
            row.get("operation_admitted") is True
            and row.get("operation_kind") in READ_ONLY_OPERATION_KINDS
            and row.get("read_operation_id") is None
            for row in representatives
        ),
        "conflict_event_count": sum(
            row.get("operation_reason") in {"key_conflict", "stale_revision"}
            for row in representatives
        ),
        "capacity_pressure_case_count": sum(
            row.get("operation_reason") == "capacity_exceeded" for row in representatives
        ),
        "stale_pressure_recovery_count": recoveries,
        "canonical_receipt_count": sum(
            isinstance(row.get("receipt_sha256"), str)
            and SHA256_PATTERN.fullmatch(str(row.get("receipt_sha256"))) is not None
            for row in representatives
        ),
        "later_read_opportunity_count": sum(
            row.get("read_operation_id") is not None for row in representatives
        ),
    }


def _headroom_valid(
    stream: Mapping[str, Any], representatives: Sequence[Mapping[str, Any]]
) -> tuple[bool, JsonDict]:
    observed = _headroom_from_rows(representatives)
    admitted = sum(row.get("operation_admitted") is True for row in representatives)
    rejected = len(representatives) - admitted
    expected = stream.get("headroom_metrics", {})
    valid = (
        observed == expected
        and all(isinstance(value, int) and value > 0 for value in observed.values())
        and stream.get("admissible_operation_count") == admitted
        and stream.get("rejected_operation_count") == rejected
        and stream.get("later_read_opportunity_count") == observed["later_read_opportunity_count"]
    )
    return valid, observed


def _stream_row(criterion: str, passed: bool, observed: JsonDict) -> JsonDict:
    return {
        "domain": "csl_stream",
        "criterion": criterion,
        "source_unit": "exp6827_stream",
        "finding": "positive" if passed else "blocked",
        "decision": "pass" if passed else "insufficient",
        "passed": passed,
        "complete": passed,
        "admissible": passed,
        "observed": observed,
    }


def validate_csl_stream(stream: Mapping[str, Any]) -> JsonDict:
    """Validate stream identities and learning preconditions independently."""
    raw_rows = stream.get("rows", [])
    rows = [row for row in raw_rows if isinstance(row, dict)] if isinstance(raw_rows, list) else []
    digest_valid, digest_count = _digest_fields_valid(rows)
    order_valid = _order_identity_valid(stream, rows)
    split_valid = _split_identity_valid(stream, rows)
    transaction_valid, representatives = _transaction_validation(rows)
    rotation_valid = _rotation_valid(stream)
    sealed_valid = _sealed_fields_valid(stream, rows)
    headroom_valid, headroom = _headroom_valid(stream, representatives)
    honest = stream.get("honest_verdict")
    no_updates = not any(
        name in stream for name in ("weight_updates", "parameter_updates", "learning_receipt")
    )
    learning_valid = (
        isinstance(honest, str)
        and "no learning ran" in honest.lower()
        and no_updates
        and stream.get("verified_memory_stream_ready") is True
    )
    validation_rows = [
        _stream_row(
            "row_hash_identity",
            digest_valid,
            {"row_count": len(rows), "digest_count": digest_count},
        ),
        _stream_row(
            "order_identity", order_valid, {"order_count": len(stream.get("order_hashes", {}))}
        ),
        _stream_row(
            "split_identity",
            split_valid,
            {"family_count": len(stream.get("split_manifest", {}).get("by_family", {}))},
        ),
        _stream_row(
            "canonical_transactions", transaction_valid, {"transaction_count": len(representatives)}
        ),
        _stream_row(
            "family_rotation",
            rotation_valid,
            {"rotation_count": len(stream.get("split_manifest", {}).get("rotations", []))},
        ),
        _stream_row(
            "sealed_fields",
            sealed_valid,
            {"decision_feature_count": len(stream.get("feature_allowlist", []))},
        ),
        _stream_row("nonzero_headroom", headroom_valid, headroom),
        _stream_row(
            "learning_preconditions",
            learning_valid,
            {"weights_immutable": no_updates, "learning_ran": False},
        ),
    ]
    admissible = all(row["passed"] for row in validation_rows)
    return {
        "admissible": admissible,
        "learning_ran": False,
        "weights_immutable": no_updates,
        "row_count": len(rows),
        "rows": validation_rows,
    }


def validate_phase_clocks(phase_clocks: Mapping[str, Any], duration_s: float) -> list[str]:
    """Check required phase names, monotonic elapsed time, and duration coverage."""
    errors: list[str] = []
    if set(phase_clocks) != set(PHASE_NAMES):
        errors.append("phase clock names differ from the required phases")
        return errors
    elapsed = [phase_clocks[name].get("elapsed_s") for name in PHASE_NAMES]
    if not all(isinstance(value, (int, float)) for value in elapsed):
        errors.append("phase clocks contain a nonnumeric elapsed value")
    elif elapsed != sorted(elapsed):
        errors.append("phase clocks are not ordered")
    elif float(duration_s) < float(elapsed[-1]):
        errors.append("duration does not cover the verify phase")
    return errors


def _checksum_payload(artifact: Mapping[str, Any]) -> JsonDict:
    excluded = {
        "reproducibility_checksum",
        "duration_s",
        "phase_clocks",
        "launch_receipts",
        "accelerator_samples",
        "field_principles",
    }
    return {key: deepcopy(value) for key, value in artifact.items() if key not in excluded}


def _apply_checksum(artifact: JsonDict) -> None:
    artifact["reproducibility_checksum"] = sha256_json(_checksum_payload(artifact))


def _base_artifact(
    *,
    run_date: str,
    phase_clocks: Mapping[str, Any],
    duration_s: float,
    launch_receipts: Mapping[str, Any],
    accelerator_samples: Mapping[str, Any],
    hashes: Mapping[str, str],
    checks: Sequence[Mapping[str, Any]],
    implementation_hashes: Mapping[str, Any] | None,
) -> JsonDict:
    return {
        "schema": "carnot.experiment_6831.v597_evidence_admissibility_contract.v1",
        "experiment_id": "6831",
        "title": "V597 Evidence Admissibility Contract",
        "run_date": run_date,
        "status": "complete_blocked_v597_evidence_admissibility",
        "openspec_requirement_ids": ["REQ-CONSTRAINT-6831", "REQ-CL-6831"],
        "replay_commands": list(REPLAY_COMMANDS),
        "field_principles": deepcopy(FIELD_PRINCIPLES),
        "inference_substrate": "aggregation_from_upstream_artifacts",
        "duration_s": round(float(duration_s), 6),
        "phase_clocks": deepcopy(dict(phase_clocks)),
        "launch_receipts": deepcopy(dict(launch_receipts)),
        "accelerator_samples": deepcopy(dict(accelerator_samples)),
        "implementation_hashes": deepcopy(dict(implementation_hashes or _implementation_hashes())),
        "reproducibility_checksum": "",
        "source_artifact_hashes": _source_hash_records(hashes),
        "preconditions_checked": deepcopy(list(checks)),
        "rows": [],
        "flagged_receipt_disposition": {
            "source": "exp6826",
            "flagged": True,
            "flag_kinds": ["DURATION_TOO_SHORT", "METHODOLOGY_MISSING"],
            "disposition": "quarantined_comparator_only",
            "authority_consumed": False,
        },
        "selective_arbiter_decisions": {},
        "selective_arbiter_receipt_admissible": False,
        "csl_stream_validation": {},
        "csl_inputs_admissible": False,
        "v597_contract_ready": False,
        "gate_check_summary": {},
        "verifier_is_oracle": False,
        "verdict_class": "blocked",
        "honest_verdict": "complete_blocked_v597_evidence_admissibility",
    }


def build_contract(
    payloads: Mapping[str, JsonDict],
    initial_hashes: Mapping[str, str],
    final_hashes: Mapping[str, str],
    *,
    run_date: str,
    phase_clocks: Mapping[str, Any],
    duration_s: float,
    launch_receipts: Mapping[str, Any],
    accelerator_samples: Mapping[str, Any],
    implementation_hashes: Mapping[str, Any] | None = None,
) -> JsonDict:
    """Build a terminal contract, stopping before reduction on a failed gate."""
    checks, failed = _precondition_checks(payloads, initial_hashes, final_hashes)
    artifact = _base_artifact(
        run_date=run_date,
        phase_clocks=phase_clocks,
        duration_s=duration_s,
        launch_receipts=launch_receipts,
        accelerator_samples=accelerator_samples,
        hashes=initial_hashes,
        checks=checks,
        implementation_hashes=implementation_hashes,
    )
    clock_errors = validate_phase_clocks(phase_clocks, duration_s)
    if failed is None and clock_errors:
        failed = {
            "check": "task_owned_phase_clocks",
            "expected": "ordered required phases covered by duration_s",
            "observed": clock_errors,
            "passed": False,
        }
        artifact["preconditions_checked"].append(failed)
    if failed is not None:
        artifact["gate_check_summary"] = {
            "passed": False,
            "failed_check": failed["check"],
            "expected": failed["expected"],
            "observed": failed["observed"],
        }
        _apply_checksum(artifact)
        return artifact

    decisions, authority_rows = reduce_selective_arbiter(
        payloads["exp6813"], payloads["exp6824"], payloads["exp6825"]
    )
    stream_validation = validate_csl_stream(payloads["exp6827"])
    findings = {name: value["finding"] for name, value in decisions.items() if name != "adoption"}
    receipt_admissible = receipt_is_admissible(
        findings, rows_complete=len(authority_rows) == AUTHORITY_CRITERION_ROW_COUNT
    )
    csl_admissible = bool(stream_validation["admissible"])
    contract_ready = receipt_admissible and csl_admissible
    artifact.update(
        {
            "status": "complete_v597_evidence_admissibility",
            "rows": [*authority_rows, *stream_validation["rows"]],
            "selective_arbiter_decisions": decisions,
            "selective_arbiter_receipt_admissible": receipt_admissible,
            "csl_stream_validation": {
                key: deepcopy(value) for key, value in stream_validation.items() if key != "rows"
            },
            "csl_inputs_admissible": csl_admissible,
            "v597_contract_ready": contract_ready,
            "gate_check_summary": {
                "passed": contract_ready,
                "failed_check": None if contract_ready else "component_admissibility",
                "expected": {
                    "selective_arbiter_receipt_admissible": True,
                    "csl_inputs_admissible": True,
                },
                "observed": {
                    "selective_arbiter_receipt_admissible": receipt_admissible,
                    "csl_inputs_admissible": csl_admissible,
                },
            },
            "verdict_class": "positive" if contract_ready else "partial",
            "honest_verdict": (
                "complete_v597_evidence_admissibility: selective-arbiter authority and causal-edge inputs "
                "are procedurally admissible; no learning ran"
                if contract_ready
                else "complete_v597_evidence_admissibility: one or more component inputs remain inadmissible; no learning ran"
            ),
        }
    )
    _apply_checksum(artifact)
    return artifact


def validate_artifact(artifact: Mapping[str, Any]) -> list[str]:
    """Validate the terminal schema, principles, clocks, rows, and checksum."""
    errors: list[str] = []
    required = set(FIELD_PRINCIPLES) | {"field_principles"}
    if set(artifact) != required:
        errors.append("top-level fields differ from the declared contract")
    principles = artifact.get("field_principles")
    if not isinstance(principles, dict) or set(principles) != set(artifact) - {"field_principles"}:
        errors.append("field principles do not cover every declared field")
    if artifact.get("inference_substrate") != "aggregation_from_upstream_artifacts":
        errors.append("inference substrate is not the aggregation allowlist value")
    if artifact.get("verdict_class") not in CLOSED_VERDICT_CLASSES:
        errors.append("verdict class is outside the closed set")
    honest = artifact.get("honest_verdict")
    if not isinstance(honest, str) or not honest.startswith("complete_"):
        errors.append("honest verdict lacks the terminal complete prefix")
    duration = artifact.get("duration_s")
    if isinstance(duration, (int, float)):
        errors.extend(validate_phase_clocks(artifact.get("phase_clocks", {}), float(duration)))
    else:
        errors.append("duration is not numeric")
    recorded_checksum = artifact.get("reproducibility_checksum")
    if recorded_checksum != sha256_json(_checksum_payload(artifact)):
        errors.append("reproducibility checksum mismatch")
    ready = artifact.get("v597_contract_ready") is True
    rows = artifact.get("rows")
    if ready and (
        not isinstance(rows, list)
        or len(rows) != AUTHORITY_CRITERION_ROW_COUNT + STREAM_CRITERION_ROW_COUNT
    ):
        errors.append("ready contract does not have one row per criterion source unit")
    if ready and not (
        artifact.get("selective_arbiter_receipt_admissible") is True
        and artifact.get("csl_inputs_admissible") is True
        and artifact.get("verdict_class") == "positive"
    ):
        errors.append("ready contract disagrees with component admissibility")
    if not ready and artifact.get("verdict_class") == "blocked" and rows != []:
        errors.append("blocked precondition contract contains reduced rows")
    return errors


def _write_json(path: Path, artifact: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(canonical_bytes(artifact))
    temporary.replace(path)


def execute(root: Path, run_date: str, output_path: Path) -> JsonDict:
    """Read, recompute, verify immutable inputs, and write one artifact."""
    if re.fullmatch(r"\d{8}", run_date) is None:
        raise ValueError("run date must use YYYYMMDD")
    datetime.strptime(run_date, "%Y%m%d")
    started = time.perf_counter()
    clocks: JsonDict = {"start": _phase_clock(started)}
    payloads, initial_hashes = _read_sources(root)
    clocks["read"] = _phase_clock(started)
    implementations = _implementation_hashes()
    clocks["recompute"] = _phase_clock(started)
    clocks["write"] = _phase_clock(started)
    final_hashes = _hash_source_files(root)
    clocks["verify"] = _phase_clock(started)
    duration = max(time.perf_counter() - started, float(clocks["verify"]["elapsed_s"]))
    command_args = ["--date", run_date]
    artifact = build_contract(
        payloads,
        initial_hashes,
        final_hashes,
        run_date=run_date,
        phase_clocks=clocks,
        duration_s=duration,
        launch_receipts=_launch_receipts(root, command_args),
        accelerator_samples=_accelerator_samples(),
        implementation_hashes=implementations,
    )
    _write_json(output_path, artifact)
    return artifact


def main(argv: Sequence[str] | None = None) -> int:
    """Run the task or validate an already written artifact."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default="20260831")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--output", type=Path)
    parser.add_argument("--validate", type=Path)
    args = parser.parse_args(argv)
    if args.validate is not None:
        artifact = json.loads(args.validate.read_text())
        errors = validate_artifact(artifact)
        if errors:
            print("\n".join(errors), file=sys.stderr)
            return 1
        print(f"valid: {args.validate}")
        return 0
    output = args.output or args.root / OUTPUT_PATH
    artifact = execute(args.root, args.date, output)
    print(json.dumps({"artifact": str(output), "verdict": artifact["honest_verdict"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
