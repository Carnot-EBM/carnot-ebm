"""Cold-audit the Exp6791 self-learning claim before causal replay.

Spec refs: REQ-CL-6792 and SCENARIO-CL-6792-*.

The audit reads checked-in JSON in a fresh CPU process. It stops before replay
when source evidence cannot support an independent byte-level audit. This rule
prevents producer booleans from replacing the transaction bytes they describe.
"""

from __future__ import annotations

import argparse
import base64
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


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[2]
RUN_DATE = "20260830"
EXPERIMENT_ID = "experiment_6792_csl_causal_safety_cold_audit"
SCHEMA = "carnot.experiment_6792.csl_causal_safety_cold_audit.v1"
INFERENCE_SUBSTRATE = "fresh-process CPU causal and safety replay, no LLM"
RANDOM_SEED = 6_792_030
MODULE_RELATIVE_PATH = Path("python/carnot/experiment_6792_csl_causal_safety_cold_audit.py")
SCRIPT_RELATIVE_PATH = Path("scripts/experiments/experiment_6792_csl_causal_safety_cold_audit.py")
RESULT_RELATIVE_PATH = Path("results/experiment_6792_csl_causal_safety_cold_audit.json")
SPEC_RELATIVE_PATH = Path("openspec/capabilities/continuous-learning/spec.md")
SOURCE_RELATIVE_PATHS = {
    "experiment_6790": Path("results/experiment_6790_chronological_constraint_routing_stream.json"),
    "experiment_6791": Path(
        "results/experiment_6791_compositional_online_constraint_routing_ab.json"
    ),
    "experiment_6750": Path("results/experiment_6750_csl_durability_support_poison_audit.json"),
    "experiment_6763": Path("results/experiment_6763_csl_hard_case_forgetting_audit.json"),
}
EXPECTED_SOURCE_HASHES = {
    "experiment_6790": "sha256:2f2cf984ac9d4dcf4be0fc211329022bc773d3cfd88bb861aaec474e9c53aeb4",
    "experiment_6791": "sha256:bf07395629a10ec9ec434c2f8bac809ac9729e921ebf1e55f10268ea10e27d99",
    "experiment_6750": "sha256:0982dc35c3fe4ba084d5b64b2909750bedc5c33602922d4eb52d8ac9adc9d0d0",
    "experiment_6763": "sha256:134025c66c4b08b57be2b73013a8b35d3c0a4c9b33d9b77ea3542b3c2260a7e0",
}
EXPECTED_ORDER_HASHES = {
    "order_1": "sha256:eb1b5e209eb223964c68e0b479ed5f0c339a41c40fdbb7bf743c831722b55183",
    "order_2": "sha256:58c7ecf2ce649ad058c96d192c8667945f11ebe185516855f504417e67f08ddd",
    "order_3": "sha256:259b36180d68a5c7214078f3659c42b55bb663ee4ab82ec1b81226c7c815ccf4",
    "order_4": "sha256:b3a2752bf52c4ca72ed084b21ada328a154b4504979f369f7cc6a9682704cf39",
    "order_5": "sha256:ac41cb8ac831556cca9e2fb2a0ebebd966cb567be7e294ca8201d0f81f32d14e",
}
EXPECTED_ARMS = {
    "frozen_controller",
    "compositional_online",
    "random_update_placebo",
    "retrieval_disabled_online",
}
EXPECTED_EVENT_COUNT = 240
EXPECTED_ROW_COUNT = len(EXPECTED_ORDER_HASHES) * EXPECTED_EVENT_COUNT * len(EXPECTED_ARMS)
VERDICT_CLASSES = {
    "positive",
    "circular_positive",
    "null",
    "blocked",
    "disqualified",
    "partial",
}
TERMINAL_PREFIXES = (
    "complete:",
    "complete_",
    "success:",
    "success_",
    "passed:",
    "passed_",
    "shipped:",
    "shipped_",
)
REQUIRED_AUDIT_FIELDS = (
    "field_principles",
    "inference_substrate",
    "duration_s",
    "random_seed",
    "reproducibility_checksum",
    "source_artifact_hashes",
    "rows",
    "cold_recomputed_metrics",
    "headline_differences",
    "credited_factor_count",
    "factors_with_changed_action_witness",
    "retrieval_disable_effects",
    "poison_attack_results",
    "admitted_poison_count",
    "influenced_poison_count",
    "restart_byte_identity",
    "restart_action_identity",
    "capacity_eviction_receipts",
    "retention_after_phase",
    "hard_case_harm_after_phase",
    "rollback_byte_identity",
    "rollback_action_identity",
    "source_verdict_supported",
    "csl_causal_audit_completed",
    "gate_check_summary",
    "verifier_is_oracle",
    "verdict_class",
    "honest_verdict",
)
REQUIRED_ARTIFACT_FIELDS = (
    "schema",
    "experiment_id",
    "run_date",
    "status",
    *REQUIRED_AUDIT_FIELDS,
)
FIELD_PRINCIPLES = {
    "schema": "The version makes incompatible audit payloads fail closed.",
    "experiment_id": "The stable ID binds the artifact to this cold audit.",
    "run_date": "The fixed date prevents silent protocol drift.",
    "status": "The status separates a source block from a complete replay.",
    "field_principles": "Each field states the evidence boundary it protects.",
    "inference_substrate": "The audit uses a fresh CPU process and no LLM.",
    "duration_s": "Measured wall time shows that prerequisite checks ran.",
    "random_seed": "The fixed seed owns later replay and confidence sampling.",
    "reproducibility_checksum": "The checksum detects drift in stable audit evidence.",
    "source_artifact_hashes": "Exact file hashes bind every checked-in source.",
    "rows": "Rows hold source replays, causal controls, and attacks after admission.",
    "cold_recomputed_metrics": "Raw rows must own metrics after prerequisites pass.",
    "headline_differences": "Cold values must be compared with source headlines.",
    "credited_factor_count": "The count includes only action and utility witnesses.",
    "factors_with_changed_action_witness": "Each credited factor needs an action witness.",
    "retrieval_disable_effects": "The global control isolates retrieval effects.",
    "poison_attack_results": "Attack rows must remain visible after rejection.",
    "admitted_poison_count": "No poison can enter durable memory.",
    "influenced_poison_count": "No poison can change a later action.",
    "restart_byte_identity": "Restarted state must match exact prior bytes.",
    "restart_action_identity": "Restarted state must choose the same next action.",
    "capacity_eviction_receipts": "Eviction receipts expose every removed record.",
    "retention_after_phase": "Old-family retention is checked after each phase.",
    "hard_case_harm_after_phase": "Hard cases are checked after each phase.",
    "rollback_byte_identity": "Rollback must restore exact prior bytes.",
    "rollback_action_identity": "Rollback must restore retrieval and action behavior.",
    "source_verdict_supported": "Positive support needs every causal and safety gate.",
    "csl_causal_audit_completed": "Completion depends on work, not effect sign.",
    "gate_check_summary": "Each failure keeps its expected and observed value.",
    "verifier_is_oracle": "False keeps the audit separate from outcome authority.",
    "verdict_class": "A closed class prevents an ambiguous terminal claim.",
    "honest_verdict": "A terminal prefix lets automation classify the result.",
}


def canonical_json_bytes(value: Any) -> bytes:
    """Return one stable JSON byte form for audit checksums."""

    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n"
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    """Return a project-style SHA-256 digest."""

    return "sha256:" + hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str | None:
    """Hash exact source bytes, or return no hash for an absent source."""

    return sha256_bytes(path.read_bytes()) if path.is_file() else None


def read_json_object(path: Path) -> JsonDict:
    """Read one JSON object and reject arrays or scalar roots."""

    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {path}")
    return value


def _gate(check: str, expected: Any, observed: Any) -> JsonDict:
    """Keep exact values so a failed prerequisite remains diagnosable."""

    return {
        "check": check,
        "expected": deepcopy(expected),
        "observed": deepcopy(observed),
        "passed": observed == expected,
    }


def _gate_summary(checks: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Collect failed prerequisites without discarding their evidence."""

    copied = [deepcopy(dict(row)) for row in checks]
    failures = [row for row in copied if row["passed"] is not True]
    return {
        "all_passed": not failures,
        "checks": copied,
        "failed_checks": [row["check"] for row in failures],
        "failures": failures,
    }


def _complete_row_observation(source: Mapping[str, Any]) -> JsonDict:
    """Count complete order-event-arm cells without trusting source totals."""

    rows = source.get("rows", [])
    required = {
        "arm",
        "event_id",
        "hidden_receipt_hash",
        "order_id",
        "position",
        "revealed_post_action_receipt",
        "row_key",
        "snapshot",
        "transaction",
    }
    row_keys = [row.get("row_key") for row in rows if isinstance(row, Mapping)]
    cells = Counter(
        (
            str(row.get("order_id")),
            str(row.get("event_id")),
            str(row.get("arm")),
        )
        for row in rows
        if isinstance(row, Mapping)
    )
    complete_rows = sum(
        required <= set(row)
        and row.get("arm") in EXPECTED_ARMS
        and row.get("order_id") in EXPECTED_ORDER_HASHES
        for row in rows
        if isinstance(row, Mapping)
    )
    return {
        "row_count": len(rows),
        "complete_rows": complete_rows,
        "unique_row_keys": len(set(row_keys)),
        "unique_cells": len(cells),
        "singleton_cells": sum(count == 1 for count in cells.values()),
        "transaction_receipt_count": len(source.get("transaction_receipts", [])),
    }


def _transaction_byte_observation(source: Mapping[str, Any]) -> JsonDict:
    """Verify raw parent and new bytes against each committed receipt hash."""

    receipts = [
        row
        for row in source.get("transaction_receipts", [])
        if isinstance(row, Mapping) and row.get("committed") is True
    ]
    parent_count = sum(isinstance(row.get("parent_bytes_b64"), str) for row in receipts)
    new_count = sum(isinstance(row.get("new_state_bytes_b64"), str) for row in receipts)
    hash_matches = 0
    for row in receipts:
        parent = row.get("parent_bytes_b64")
        new = row.get("new_state_bytes_b64")
        if isinstance(parent, str) and isinstance(new, str):
            try:
                parent_bytes = base64.b64decode(parent.encode("ascii"), validate=True)
                new_bytes = base64.b64decode(new.encode("ascii"), validate=True)
            except (ValueError, UnicodeEncodeError):
                continue
            hash_matches += int(
                sha256_bytes(parent_bytes) == row.get("parent_hash")
                and sha256_bytes(new_bytes) == row.get("new_state_hash")
            )
    return {
        "committed_receipts": len(receipts),
        "parent_byte_snapshots": parent_count,
        "new_state_byte_snapshots": new_count,
        "byte_hash_matches": hash_matches,
    }


def _exact_receipt_observation(
    source: Mapping[str, Any], routing_source: Mapping[str, Any]
) -> JsonDict:
    """Match each arm receipt hash to its independent Exp6790 event receipt."""

    expected = {
        (str(row.get("order_id")), str(row.get("event_id"))): row.get("hidden_receipt_hash")
        for row in routing_source.get("rows", [])
        if isinstance(row, Mapping)
    }
    rows = source.get("rows", [])
    matching = sum(
        row.get("hidden_receipt_hash")
        == expected.get((str(row.get("order_id")), str(row.get("event_id"))))
        == row.get("revealed_post_action_receipt", {}).get("source_receipt_hash")
        for row in rows
        if isinstance(row, Mapping)
    )
    return {"expected_rows": EXPECTED_ROW_COUNT, "matching_rows": matching}


def evaluate_preconditions(
    sources: Mapping[str, Mapping[str, Any]], source_paths: Mapping[str, Path]
) -> JsonDict:
    """Check all frozen inputs before causal or safety code can execute."""

    source = sources.get("experiment_6791", {})
    routing_source = sources.get("experiment_6790", {})
    complete_rows = _complete_row_observation(source)
    expected_rows = {
        "row_count": EXPECTED_ROW_COUNT,
        "complete_rows": EXPECTED_ROW_COUNT,
        "unique_row_keys": EXPECTED_ROW_COUNT,
        "unique_cells": EXPECTED_ROW_COUNT,
        "singleton_cells": EXPECTED_ROW_COUNT,
        "transaction_receipt_count": EXPECTED_ROW_COUNT,
    }
    observed_hashes = {name: sha256_file(Path(path)) for name, path in source_paths.items()}
    expected_hash_evidence = {
        "artifact_hashes": EXPECTED_SOURCE_HASHES,
        "exp6791_source_artifact_hash": EXPECTED_SOURCE_HASHES["experiment_6790"],
    }
    observed_hash_evidence = {
        "artifact_hashes": observed_hashes,
        "exp6791_source_artifact_hash": source.get("source_artifact_hash"),
    }
    byte_observation = _transaction_byte_observation(source)
    expected_bytes = {
        "committed_receipts": byte_observation["committed_receipts"],
        "parent_byte_snapshots": byte_observation["committed_receipts"],
        "new_state_byte_snapshots": byte_observation["committed_receipts"],
        "byte_hash_matches": byte_observation["committed_receipts"],
    }
    order_hashes = source.get("frozen_manifest", {}).get("order_hashes", {})
    observed_orders = {
        "order_count": len(order_hashes),
        "order_hashes": order_hashes,
        "rows_per_order": dict(
            sorted(
                Counter(
                    str(row.get("order_id"))
                    for row in source.get("rows", [])
                    if isinstance(row, Mapping)
                ).items()
            )
        ),
    }
    expected_orders = {
        "order_count": len(EXPECTED_ORDER_HASHES),
        "order_hashes": EXPECTED_ORDER_HASHES,
        "rows_per_order": {
            order_id: EXPECTED_EVENT_COUNT * len(EXPECTED_ARMS)
            for order_id in EXPECTED_ORDER_HASHES
        },
    }
    checks = [
        _gate("compositional_csl_completed", True, source.get("compositional_csl_completed")),
        _gate("complete_per_event_rows", expected_rows, complete_rows),
        _gate("source_hashes", expected_hash_evidence, observed_hash_evidence),
        _gate("transaction_byte_snapshots", expected_bytes, byte_observation),
        _gate(
            "exact_receipt_hashes",
            {"expected_rows": EXPECTED_ROW_COUNT, "matching_rows": EXPECTED_ROW_COUNT},
            _exact_receipt_observation(source, routing_source),
        ),
        _gate("all_five_orders", expected_orders, observed_orders),
    ]
    return _gate_summary(checks)


def _source_hashes(source_paths: Mapping[str, Path]) -> JsonDict:
    """Record exact input hashes even when one prerequisite blocks replay."""

    return {name: sha256_file(Path(path)) for name, path in source_paths.items()}


def reproducibility_checksum(artifact: Mapping[str, Any]) -> str:
    """Hash stable evidence while excluding elapsed wall time and this hash."""

    material = {
        key: artifact.get(key)
        for key in REQUIRED_ARTIFACT_FIELDS
        if key not in {"duration_s", "reproducibility_checksum"}
    }
    return sha256_bytes(canonical_json_bytes(material))


def build_artifact(
    *,
    source_paths: Mapping[str, Path] | None = None,
    run_date: str = RUN_DATE,
    duration_s: float | None = None,
) -> JsonDict:
    """Build the terminal block and never enter replay after a failed gate."""

    if len(run_date) != 8 or not run_date.isdigit():
        raise ValueError("date must use YYYYMMDD")
    started = time.monotonic()
    paths = {
        name: Path(path)
        for name, path in (
            source_paths
            or {name: REPO_ROOT / relative for name, relative in SOURCE_RELATIVE_PATHS.items()}
        ).items()
    }
    sources = {name: read_json_object(path) for name, path in paths.items()}
    summary = evaluate_preconditions(sources, paths)
    if summary["all_passed"]:
        raise RuntimeError("source prerequisites passed; a complete replay is required")
    failed = summary["failures"][0]
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "run_date": run_date,
        "status": "complete_blocked_csl_causal_audit",
        "field_principles": deepcopy(FIELD_PRINCIPLES),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "duration_s": round(
            float(duration_s) if duration_s is not None else time.monotonic() - started,
            6,
        ),
        "random_seed": RANDOM_SEED,
        "reproducibility_checksum": "",
        "source_artifact_hashes": _source_hashes(paths),
        "rows": [],
        "cold_recomputed_metrics": {},
        "headline_differences": {},
        "credited_factor_count": 0,
        "factors_with_changed_action_witness": [],
        "retrieval_disable_effects": [],
        "poison_attack_results": [],
        "admitted_poison_count": 0,
        "influenced_poison_count": 0,
        "restart_byte_identity": None,
        "restart_action_identity": None,
        "capacity_eviction_receipts": [],
        "retention_after_phase": {},
        "hard_case_harm_after_phase": {},
        "rollback_byte_identity": None,
        "rollback_action_identity": None,
        "source_verdict_supported": False,
        "csl_causal_audit_completed": False,
        "gate_check_summary": summary,
        "verifier_is_oracle": False,
        "verdict_class": "blocked",
        "honest_verdict": (
            "complete_blocked_csl_causal_audit: "
            f"{failed['check']} expected {failed['expected']!r}, observed {failed['observed']!r}"
        ),
    }
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    errors = validate_artifact(artifact)
    if errors:
        raise ValueError("; ".join(errors))
    return artifact


def validate_artifact(artifact: Mapping[str, Any]) -> list[str]:
    """Return closed schema errors without changing the audit evidence."""

    errors: list[str] = []
    if set(artifact) != set(REQUIRED_ARTIFACT_FIELDS):
        errors.append("required field set mismatch")
    if set(artifact.get("field_principles", {})) != set(REQUIRED_ARTIFACT_FIELDS):
        errors.append("field principle coverage mismatch")
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        errors.append("inference substrate mismatch")
    if artifact.get("random_seed") != RANDOM_SEED:
        errors.append("random seed mismatch")
    if artifact.get("verifier_is_oracle") is not False:
        errors.append("verifier_is_oracle must be false")
    if artifact.get("verdict_class") not in VERDICT_CLASSES:
        errors.append("verdict class is outside the closed enum")
    if not str(artifact.get("honest_verdict", "")).startswith(TERMINAL_PREFIXES):
        errors.append("honest verdict lacks a terminal prefix")
    if artifact.get("reproducibility_checksum") != reproducibility_checksum(artifact):
        errors.append("reproducibility checksum mismatch")
    if artifact.get("status") == "complete_blocked_csl_causal_audit":
        if artifact.get("verdict_class") != "blocked":
            errors.append("blocked verdict_class mismatch")
        if artifact.get("csl_causal_audit_completed") is not False:
            errors.append("blocked audit cannot be complete")
        if artifact.get("rows") != []:
            errors.append("blocked audit contains replay rows")
        summary = artifact.get("gate_check_summary", {})
        if summary.get("all_passed") is not False or not summary.get("failures"):
            errors.append("blocked audit lacks a failed gate")
    return errors


def write_artifact(path: Path | str, artifact: Mapping[str, Any]) -> JsonDict:
    """Validate and publish one complete artifact with an atomic rename."""

    errors = validate_artifact(artifact)
    if errors:
        raise ValueError("; ".join(errors))
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(artifact, indent=2, sort_keys=True, ensure_ascii=True) + "\n"
    descriptor, name = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=target.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
    return {"path": str(target), "atomic_rename": True, "sha256": sha256_file(target)}


def main(argv: Sequence[str] | None = None) -> int:
    """Run the prerequisite audit or validate its stored terminal artifact."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default=RUN_DATE)
    parser.add_argument("--output", default=str(REPO_ROOT / RESULT_RELATIVE_PATH))
    parser.add_argument("--validate", action="store_true")
    args = parser.parse_args(argv)
    output = Path(args.output)
    if args.validate:
        artifact = read_json_object(output)
        errors = validate_artifact(artifact)
        if errors:
            raise ValueError("; ".join(errors))
    else:
        artifact = build_artifact(run_date=args.date)
        write_artifact(output, artifact)
    print(artifact["honest_verdict"])
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
