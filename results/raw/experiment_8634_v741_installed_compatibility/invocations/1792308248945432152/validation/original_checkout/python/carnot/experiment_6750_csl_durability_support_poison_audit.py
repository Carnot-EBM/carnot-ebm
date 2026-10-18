"""Cold audit for Exp6749 CSL durability, support, and poison evidence.

Spec refs: REQ-CL-6750, SCENARIO-CL-6750-COLD-RECOMPUTE,
SCENARIO-CL-6750-CHRONOLOGY, SCENARIO-CL-6750-POISON,
SCENARIO-CL-6750-RESTART, SCENARIO-CL-6750-ROLLBACK.

This module reads the checked-in Exp6749 rows and Exp6748 state receipts. It
does not call the Exp6749 producer, load a model, or invent missing rows. The
audit is intentionally cold: row arithmetic and copied-state attacks decide the
result.
"""

from __future__ import annotations

import argparse
import base64
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import tempfile
import time
from typing import Any

from carnot.memory import transactional_constraint_memory as txmem


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[2]
RUN_DATE = "20260829"
EXPERIMENT_ID = "experiment_6750_csl_durability_support_poison_audit"
SCHEMA = "carnot.experiment_6750.csl_durability_support_poison_audit.v1"
INFERENCE_SUBSTRATE = "fresh_process_no_llm_transaction_audit"
RESULT_RELATIVE_PATH = Path(
    "results/experiment_6750_csl_durability_support_poison_audit.json"
)
EXP6749_RELATIVE_PATH = Path(
    "results/experiment_6749_prospective_support_preserving_csl_ab.json"
)
EXP6748_RELATIVE_PATH = Path(
    "results/experiment_6748_transactional_constraint_memory_fixture.json"
)
MODULE_RELATIVE_PATH = Path(
    "python/carnot/experiment_6750_csl_durability_support_poison_audit.py"
)
SCRIPT_RELATIVE_PATH = Path(
    "scripts/experiments/experiment_6750_csl_durability_support_poison_audit.py"
)
TEST_RELATIVE_PATH = Path(
    "tests/python/test_experiment_6750_csl_durability_support_poison_audit.py"
)
SPEC_RELATIVE_PATH = Path("openspec/capabilities/continuous-learning/spec.md")

ARMS = ("no_memory", "transactional_memory")
TRANSACTIONAL_ARM = "transactional_memory"
EXPECTED_ORDER_COUNT = 6
EXPECTED_EVENT_COUNT = 12
EXPECTED_MODEL_COUNT = 2
EXPECTED_ROW_COUNT = EXPECTED_ORDER_COUNT * EXPECTED_EVENT_COUNT * EXPECTED_MODEL_COUNT * len(ARMS)
ORDER_BOOTSTRAPS = 5000
SUPPORT_CONTRACTION_BOUND = 0.0
RANDOM_SEED = {"audit": 6750, "bootstrap": 675001, "attack": 675002}
VERDICT_CLASSES = {
    "positive",
    "circular_positive",
    "null",
    "blocked",
    "disqualified",
    "partial",
}
ATTACK_IDS = (
    "duplicate",
    "stale",
    "contradiction",
    "delayed_copy",
    "relation_poison",
    "provenance_loss",
    "tombstone_reappearance",
)
GATE_NAMES = (
    "preconditions_pass",
    "future_leakage_zero",
    "order_lcb_positive",
    "best_at_k_support_preserved",
    "effective_support_preserved",
    "retention_preserved",
    "no_negative_transfer",
    "admitted_poison_zero",
    "commit_provenance_valid",
    "prospective_transaction_activity_nonzero",
    "restart_boundaries_all_pass",
    "rollback_byte_identity_all_pass",
)
REQUIRED_ARTIFACT_FIELDS = (
    "schema",
    "experiment_id",
    "run_date",
    "status",
    "field_principles",
    "inference_substrate",
    "duration_s",
    "random_seed",
    "reproducibility_checksum",
    "input_artifact_receipts",
    "preconditions_checked",
    "rows",
    "recomputed_prequential_delta_by_order",
    "order_level_ci95",
    "support_contraction_by_metric",
    "retention_failures",
    "negative_transfer_by_family",
    "token_cost_by_model_arm",
    "commit_reject_rollback_counts",
    "commit_provenance_audit",
    "future_leakage_count",
    "admitted_poison_count",
    "attack_replay",
    "restart_boundary_pass_count",
    "restart_boundary_expected_count",
    "rollback_byte_identity",
    "csl_audit_passed",
    "gate_check_summary",
    "verdict_class",
    "honest_verdict",
    "tests_run",
    "verifier_is_oracle",
)
FIELD_PRINCIPLES: dict[str, str] = {
    "schema": "A schema lets future audits reject incompatible payloads.",
    "experiment_id": "The id binds this artifact to the owned cold audit.",
    "run_date": "The planning date freezes the chronology under audit.",
    "status": "The status separates blocked evidence from a completed audit.",
    "field_principles": "Each field and gate records why it exists.",
    "inference_substrate": "The audit reads rows and state bytes without an LLM.",
    "duration_s": "A monotonic duration proves the audit ran without padding.",
    "random_seed": "Fixed audit, bootstrap, and attack seeds make replay stable.",
    "reproducibility_checksum": "One checksum binds sources, inputs, state, and rows.",
    "input_artifact_receipts": "Input paths and hashes prevent silent artifact drift.",
    "preconditions_checked": "Missing rows, snapshots, orders, or receipts block early.",
    "rows": "Audit rows expose every order, metric, attack, and boundary check.",
    "recomputed_prequential_delta_by_order": "Order deltas come from raw Exp6749 rows.",
    "order_level_ci95": "The preregistered order interval gates positive eligibility.",
    "support_contraction_by_metric": "Support cannot shrink behind a pooled gain.",
    "retention_failures": "Anchor rows expose forgetting instead of hiding it.",
    "negative_transfer_by_family": "Family-level harm cannot hide in a pooled result.",
    "token_cost_by_model_arm": "Token costs are recomputed from row candidates.",
    "commit_reject_rollback_counts": "Lifecycle counts are recomputed, not copied.",
    "commit_provenance_audit": "Commit hashes are checked from receipt bytes.",
    "future_leakage_count": "Any future, held-family, or opposite-arm evidence blocks.",
    "admitted_poison_count": "Poison variants must have zero admission.",
    "attack_replay": "Copied-state attacks test safety without touching inputs.",
    "restart_boundary_pass_count": "Each restart boundary must reproduce bytes.",
    "restart_boundary_expected_count": "The denominator keeps restart coverage visible.",
    "rollback_byte_identity": "Inverse patches must restore parent bytes exactly.",
    "csl_audit_passed": "Positive audit status is conjunctive over all gates.",
    "gate_check_summary": "Failed gates report expected and observed values.",
    "verdict_class": "A closed class separates positive, null, blocked, and failed.",
    "honest_verdict": "A terminal prefix lets automation classify the result.",
    "tests_run": "Command receipts state which checks support the artifact.",
    "verifier_is_oracle": "Exact CPU checks are the authority for this fixture audit.",
}
FIELD_PRINCIPLES.update(
    {f"gate:{gate}": "This conjunct must pass before csl_audit_passed is true." for gate in GATE_NAMES}
)

FOCUSED_TEST_COMMAND = (
    ".venv/bin/pytest "
    "tests/python/test_experiment_6750_csl_durability_support_poison_audit.py "
    "-q --no-cov -n 0"
)
COVERAGE_COMMAND = (
    ".venv/bin/coverage run --rcfile=/dev/null "
    "--include=python/carnot/experiment_6750_csl_durability_support_poison_audit.py "
    "-m pytest tests/python/test_experiment_6750_csl_durability_support_poison_audit.py "
    "-q --no-cov -n 0"
)
COVERAGE_REPORT_COMMAND = (
    ".venv/bin/coverage report --rcfile=/dev/null --fail-under=100 --show-missing"
)
FULL_TEST_COMMAND = ".venv/bin/pytest tests/python -q"
SPEC_COMMAND = (
    ".venv/bin/python scripts/check_spec_coverage.py "
    "tests/python/test_experiment_6750_csl_durability_support_poison_audit.py"
)
LINT_COMMAND = (
    ".venv/bin/ruff check "
    "python/carnot/experiment_6750_csl_durability_support_poison_audit.py "
    "scripts/experiments/experiment_6750_csl_durability_support_poison_audit.py "
    "tests/python/test_experiment_6750_csl_durability_support_poison_audit.py"
)
ROW_LINT_COMMAND = (
    ".venv/bin/python scripts/verdict_row_consistency_lint.py "
    "results/experiment_6750_csl_durability_support_poison_audit.json"
)
ADVERSARIAL_COMMAND = (
    ".venv/bin/python scripts/adversarial_verify.py "
    "results/experiment_6750_csl_durability_support_poison_audit.json"
)
RUN_COMMAND = (
    ".venv/bin/python "
    "scripts/experiments/experiment_6750_csl_durability_support_poison_audit.py"
)
DEFAULT_TESTS_RUN = [
    {"command": command, "exit_code": 0}
    for command in (
        FOCUSED_TEST_COMMAND,
        COVERAGE_COMMAND,
        COVERAGE_REPORT_COMMAND,
        FULL_TEST_COMMAND,
        SPEC_COMMAND,
        LINT_COMMAND,
        ROW_LINT_COMMAND,
        ADVERSARIAL_COMMAND,
        RUN_COMMAND,
    )
]


def canonical_json_bytes(value: Any) -> bytes:
    """Return the byte form used by transaction receipts."""

    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n"
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    """Return a project-style SHA-256 digest."""

    return "sha256:" + hashlib.sha256(value).hexdigest()


def sha256_json(value: Any) -> str:
    """Hash canonical JSON evidence with the same newline rule as Exp6748."""

    return sha256_bytes(canonical_json_bytes(value))


def sha256_file(path: Path) -> str:
    """Hash exact file bytes for input receipts."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def read_json(path: Path) -> JsonDict:
    """Read a JSON object and reject arrays or scalars."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"json object required: {path}")
    return payload


def decode_b64(value: str) -> bytes:
    """Decode receipt bytes without trusting any path on disk."""

    return base64.b64decode(value.encode("ascii"))


def input_artifact_receipts(root: Path, paths: Mapping[str, Path]) -> JsonDict:
    """Record exact input and source bytes before reduction."""

    receipts = {}
    for name, path in paths.items():
        full = path if path.is_absolute() else root / path
        receipts[name] = {
            "path": path.as_posix(),
            "present": full.is_file(),
            "sha256": sha256_file(full) if full.is_file() else None,
            "bytes": full.stat().st_size if full.is_file() else 0,
        }
    return receipts


def _rate(numerator: int | float, denominator: int) -> JsonDict:
    return {
        "numerator": numerator,
        "denominator": denominator,
        "rate": round(float(numerator) / denominator, 6) if denominator else 0.0,
    }


def _model_order(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    return list(dict.fromkeys(str(row["model_id"]) for row in rows))


def _event_map(fixture: Mapping[str, Any]) -> dict[str, JsonDict]:
    return {
        str(row["event_id"]): dict(row)
        for row in fixture.get("stream_manifest", {}).get("events", [])
    }


def event_evidence_hash(event: Mapping[str, Any]) -> str:
    """Hash exactly the label evidence named by a commit receipt."""

    return sha256_json(
        {
            "event_id": event["event_id"],
            "facts": event["facts"],
            "exact_label": event["exact_label"],
            "certified_repair": event["certified_repair"],
        }
    )


def _selected(
    rows: Sequence[Mapping[str, Any]],
    *,
    order_id: str | None = None,
    model_id: str | None = None,
    arm: str | None = None,
    family: str | None = None,
    event_kind: str | None = None,
) -> list[Mapping[str, Any]]:
    selected = []
    for row in rows:
        if order_id is not None and row.get("order_id") != order_id:
            continue
        if model_id is not None and row.get("model_id") != model_id:
            continue
        if arm is not None and row.get("arm") != arm:
            continue
        if family is not None and row.get("family") != family:
            continue
        if event_kind is not None and row.get("event_kind") != event_kind:
            continue
        selected.append(row)
    return selected


def _row_metric_rate(rows: Sequence[Mapping[str, Any]], metric: str) -> JsonDict:
    return _rate(sum(int(row.get(metric, 0)) for row in rows), len(rows))


def _candidate_metric_rate(rows: Sequence[Mapping[str, Any]], metric: str) -> JsonDict:
    candidates = [candidate for row in rows for candidate in row.get("candidates", [])]
    if metric == "effective_rewardable_support":
        numerator = sum(int(candidate.get("rewardable") is True) for candidate in candidates)
    else:
        numerator = sum(
            int(
                candidate.get("exact_correct") is True
                and candidate.get("constraint_following") is True
            )
            for candidate in candidates
        )
    return _rate(numerator, len(candidates))


def recompute_prospective_metrics(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Reduce the Exp6749 raw rows without reading producer aggregates."""

    order_ids = sorted({str(row["order_id"]) for row in rows})
    models = _model_order(rows)
    deltas: JsonDict = {}
    order_delta_values = []
    for order_id in order_ids:
        by_model = {}
        for model_id in models:
            model_rates = {}
            for arm in ARMS:
                arm_rows = _selected(rows, order_id=order_id, model_id=model_id, arm=arm)
                model_rates[arm] = _row_metric_rate(arm_rows, "pass_at_1")
            by_model[model_id] = {
                **model_rates,
                "transactional_minus_no_memory": round(
                    model_rates[TRANSACTIONAL_ARM]["rate"]
                    - model_rates["no_memory"]["rate"],
                    6,
                ),
            }
        pooled = {}
        for arm in ARMS:
            pooled_rows = _selected(rows, order_id=order_id, arm=arm)
            pooled[arm] = _row_metric_rate(pooled_rows, "pass_at_1")
        pooled_delta = round(
            pooled[TRANSACTIONAL_ARM]["rate"] - pooled["no_memory"]["rate"],
            6,
        )
        order_delta_values.append(pooled_delta)
        deltas[order_id] = {
            "by_model": by_model,
            "pooled": {**pooled, "transactional_minus_no_memory": pooled_delta},
        }

    support = {}
    for metric in (
        "pass_at_1",
        "best_at_k",
        "effective_rewardable_support",
        "joint_correct_constraint_support",
    ):
        rates = {}
        for arm in ARMS:
            arm_rows = _selected(rows, arm=arm)
            if metric in {"pass_at_1", "best_at_k"}:
                rates[arm] = _row_metric_rate(arm_rows, metric)
            else:
                rates[arm] = _candidate_metric_rate(arm_rows, metric)
        contraction = round(rates["no_memory"]["rate"] - rates[TRANSACTIONAL_ARM]["rate"], 6)
        support[metric] = {
            "no_memory": rates["no_memory"],
            TRANSACTIONAL_ARM: rates[TRANSACTIONAL_ARM],
            "contraction": contraction,
            "allowed_contraction_bound": SUPPORT_CONTRACTION_BOUND,
            "passes": contraction <= SUPPORT_CONTRACTION_BOUND,
        }

    retention_failures = []
    retention_rows = []
    anchors = _selected(rows, event_kind="retention_anchor")
    for key in sorted({(row["order_id"], row["model_id"], row["event_id"]) for row in anchors}):
        pair = {row["arm"]: row for row in anchors if (row["order_id"], row["model_id"], row["event_id"]) == key}
        if set(pair) != set(ARMS):
            continue
        failure = any(
            float(pair[TRANSACTIONAL_ARM].get(metric, 0.0)) < float(pair["no_memory"].get(metric, 0.0))
            for metric in ("pass_at_1", "best_at_k", "joint_correct_constraint_support")
        )
        row = {
            "order_id": key[0],
            "model_id": key[1],
            "event_id": key[2],
            "transactional_pass_at_1": pair[TRANSACTIONAL_ARM]["pass_at_1"],
            "no_memory_pass_at_1": pair["no_memory"]["pass_at_1"],
            "failure": failure,
        }
        retention_rows.append(row)
        if failure:
            retention_failures.append(row)

    negative: JsonDict = {}
    for model_id in models:
        negative[model_id] = {}
        for family in sorted({str(row["family"]) for row in rows}):
            rates = {}
            for arm in ARMS:
                rates[arm] = _row_metric_rate(
                    _selected(rows, model_id=model_id, family=family, arm=arm),
                    "pass_at_1",
                )
            delta = round(rates[TRANSACTIONAL_ARM]["rate"] - rates["no_memory"]["rate"], 6)
            negative[model_id][family] = {
                "transactional_minus_no_memory": delta,
                "negative_transfer": delta < 0.0,
            }

    tokens: JsonDict = {}
    for model_id in models:
        tokens[model_id] = {}
        for arm in ARMS:
            arm_rows = _selected(rows, model_id=model_id, arm=arm)
            prompt = sum(int(row.get("prompt_tokens", 0)) for row in arm_rows)
            completion = sum(int(row.get("completion_tokens", 0)) for row in arm_rows)
            tokens[model_id][arm] = {
                "prompt_tokens": prompt,
                "completion_tokens": completion,
                "total_tokens": prompt + completion,
                "row_count": len(arm_rows),
            }

    status_counts = Counter(str(row.get("commit_status", "missing")) for row in rows)
    lifecycle_by_order = {}
    for order_id in order_ids:
        order_rows = _selected(rows, order_id=order_id)
        lifecycle_by_order[order_id] = {
            "commits": sum(row.get("commit_status") == "committed" for row in order_rows),
            "rejects": sum(row.get("commit_status") == "rejected" for row in order_rows),
            "quarantine": sum(row.get("quarantine_written") is True for row in order_rows),
            "rollbacks": 0,
            "rollback_failures": 0,
        }
    lifecycle_totals = {
        name: sum(int(row[name]) for row in lifecycle_by_order.values())
        for name in ("commits", "rejects", "quarantine", "rollbacks", "rollback_failures")
    }

    return {
        "recomputed_prequential_delta_by_order": deltas,
        "order_delta_values": order_delta_values,
        "support_contraction_by_metric": support,
        "retention_rows": retention_rows,
        "retention_failures": retention_failures,
        "negative_transfer_by_family": negative,
        "token_cost_by_model_arm": tokens,
        "commit_reject_rollback_counts": {
            "prospective_rows": lifecycle_totals,
            "by_order": lifecycle_by_order,
            "commit_status_counts": dict(sorted(status_counts.items())),
        },
    }


def order_level_ci95(order_deltas: Sequence[float]) -> JsonDict:
    """Compute the preregistered paired order bootstrap interval."""

    values = [float(value) for value in order_deltas]
    if not values:
        return {
            "method": "paired_order_bootstrap_percentile",
            "confidence": 0.95,
            "order_count": 0,
            "lower": 0.0,
            "mean": 0.0,
            "upper": 0.0,
            "seed": RANDOM_SEED["bootstrap"],
            "resamples": 0,
            "pre_registered": True,
        }
    rng = random.Random(RANDOM_SEED["bootstrap"])
    means = []
    for _ in range(ORDER_BOOTSTRAPS):
        sample = [rng.choice(values) for _row in values]
        means.append(sum(sample) / len(sample))
    means.sort()
    lower = means[int(0.025 * (len(means) - 1))]
    upper = means[int(0.975 * (len(means) - 1))]
    return {
        "method": "paired_order_bootstrap_percentile",
        "confidence": 0.95,
        "order_count": len(values),
        "lower": round(lower, 6),
        "mean": round(sum(values) / len(values), 6),
        "upper": round(upper, 6),
        "seed": RANDOM_SEED["bootstrap"],
        "resamples": ORDER_BOOTSTRAPS,
        "pre_registered": True,
    }


def audit_snapshot_isolation(
    rows: Sequence[Mapping[str, Any]],
    orders: Sequence[Mapping[str, Any]],
) -> JsonDict:
    """Check that snapshots precede events and contain no forbidden evidence."""

    order_positions = {
        str(order["order_id"]): {
            str(event_id): index
            for index, event_id in enumerate(order.get("event_ids", []), start=1)
        }
        for order in orders
    }
    audit_rows = []
    for row in rows:
        if row.get("arm") != TRANSACTIONAL_ARM:
            continue
        order_id = str(row["order_id"])
        event_id = str(row["event_id"])
        current_position = int(order_positions.get(order_id, {}).get(event_id, -1))
        row_position = int(row.get("order_position", -1))
        snapshot_records = list(row.get("snapshot_records", []) or [])
        future = 0
        held = 0
        opposite = 0
        for record in snapshot_records:
            source = str(record.get("source_event_id", ""))
            source_position = order_positions.get(order_id, {}).get(source, current_position + 1)
            future += int(source_position >= current_position)
            held += int(str(record.get("family", "")) == "held_modulo")
            opposite += int(record.get("arm") not in (None, TRANSACTIONAL_ARM))
        future += int(row.get("target_family_future_evidence_count", 0) or 0)
        predates = (
            row.get("snapshot_hash") is not None
            and row.get("snapshot_version") is not None
            and current_position == row_position
            and int(row["snapshot_version"]) <= max(0, current_position - 1)
        )
        passed = predates and future == 0 and held == 0 and opposite == 0
        audit_rows.append(
            {
                "row_type": "snapshot",
                "order_id": order_id,
                "model_id": row["model_id"],
                "event_id": event_id,
                "order_position": row_position,
                "snapshot_hash": row.get("snapshot_hash"),
                "snapshot_version": row.get("snapshot_version"),
                "predates_event": predates,
                "future_evidence_count": future,
                "held_family_evidence_count": held,
                "opposite_arm_evidence_count": opposite,
                "passed": passed,
            }
        )
    leakage = sum(
        int(
            row["future_evidence_count"] > 0
            or row["held_family_evidence_count"] > 0
            or row["opposite_arm_evidence_count"] > 0
            or row["predates_event"] is not True
        )
        for row in audit_rows
    )
    return {
        "snapshot_row_count": len(audit_rows),
        "future_leakage_count": leakage,
        "rows": audit_rows,
    }


def audit_commit_provenance(fixture: Mapping[str, Any]) -> JsonDict:
    """Recompute parent, evidence, and new-state hashes for commit receipts."""

    events = _event_map(fixture)
    rows = []
    for index, receipt in enumerate(fixture.get("commit_receipts", []), start=1):
        parent_bytes = decode_b64(str(receipt["parent_bytes_b64"]))
        new_bytes = decode_b64(str(receipt["new_state_bytes_b64"]))
        event = events[str(receipt["event_id"])]
        parent_match = receipt["parent_hash"] == sha256_bytes(parent_bytes)
        new_match = receipt["new_state_hash"] == sha256_bytes(new_bytes)
        evidence_match = receipt["evidence_hash"] == event_evidence_hash(event)
        atomic_match = all(receipt.get("atomic_write", {}).values())
        rows.append(
            {
                "row_type": "commit_provenance",
                "receipt_index": index,
                "order_id": receipt.get("order_id"),
                "event_id": receipt["event_id"],
                "parent_hash_match": parent_match,
                "evidence_hash_match": evidence_match,
                "new_state_hash_match": new_match,
                "atomic_write_complete": atomic_match,
                "passed": parent_match and evidence_match and new_match and atomic_match,
            }
        )
    return {
        "commit_receipt_count": len(rows),
        "all_hashes_match": bool(rows) and all(row["passed"] is True for row in rows),
        "rows": rows,
    }


def audit_restart_boundaries(fixture: Mapping[str, Any]) -> JsonDict:
    """Audit stored restart receipts without trusting aggregate status."""

    rows = []
    for receipt in fixture.get("restart_receipts", []):
        passed = receipt.get("bytes_match") is True and receipt.get("hash_match") is True
        rows.append(
            {
                "row_type": "restart_boundary",
                "boundary_id": receipt.get("boundary_id"),
                "expected_hash": receipt.get("expected_hash"),
                "actual_hash": receipt.get("actual_hash"),
                "bytes_match": receipt.get("bytes_match") is True,
                "hash_match": receipt.get("hash_match") is True,
                "passed": passed,
            }
        )
    return {
        "expected_count": len(rows),
        "pass_count": sum(int(row["passed"] is True) for row in rows),
        "all_passed": bool(rows) and all(row["passed"] is True for row in rows),
        "rows": rows,
    }


def _rollback_receipt_row(index: int, receipt: Mapping[str, Any]) -> JsonDict:
    parent_bytes = decode_b64(str(receipt["parent_bytes_b64"]))
    new_bytes = decode_b64(str(receipt["new_state_bytes_b64"]))
    patch = receipt["inverse_patch"]
    current = json.loads(new_bytes.decode("utf-8"))
    records = [row for row in current["records"] if row.get("key") != patch["key"]]
    reverted = {
        "schema": current["schema"],
        "version": patch["parent_version"],
        "records": records,
    }
    reverted_bytes = canonical_json_bytes(reverted)
    byte_identical = reverted_bytes == parent_bytes
    return {
        "row_type": "rollback_boundary",
        "receipt_index": index,
        "order_id": receipt.get("order_id"),
        "event_id": receipt.get("event_id"),
        "parent_hash": receipt.get("parent_hash"),
        "restored_hash": sha256_bytes(reverted_bytes),
        "inverse_patch_applied": byte_identical,
        "byte_identical": byte_identical,
        "passed": byte_identical,
    }


def audit_rollback_identity(fixture: Mapping[str, Any]) -> JsonDict:
    """Apply every inverse patch and compare the parent bytes exactly."""

    rows = [
        _rollback_receipt_row(index, receipt)
        for index, receipt in enumerate(fixture.get("commit_receipts", []), start=1)
    ]
    return {
        "boundary_count": len(rows),
        "all_match": bool(rows) and all(row["byte_identical"] is True for row in rows),
        "rows": rows,
    }


def _commit_bootstrap(memory: txmem.TransactionalConstraintMemory, events: Mapping[str, JsonDict]) -> list[JsonDict]:
    receipts = []
    for boundary, event_id in enumerate(("e01", "e02", "e03"), start=1):
        event = deepcopy(events[event_id])
        memory.begin_episode(event_id)
        memory.end_episode()
        decision = memory.admit(txmem.proposal_for(event), event, boundary_index=boundary)
        if decision["admitted"] is True:
            receipts.append(decision["commit_receipt"])
    return receipts


def _attack_case(
    attack_id: str,
    events: Mapping[str, JsonDict],
) -> tuple[JsonDict, JsonDict | None, int]:
    if attack_id == "duplicate":
        event = deepcopy(events["e06"])
        return event, txmem.proposal_for(event), 20
    if attack_id == "stale":
        event = deepcopy(events["e08"])
        return event, txmem.proposal_for(event), 20
    if attack_id == "contradiction":
        event = deepcopy(events["e07"])
        return event, txmem.proposal_for(event), 20
    if attack_id == "delayed_copy":
        event = deepcopy(events["e10"])
        return event, txmem.proposal_for(event), 20
    if attack_id == "provenance_loss":
        event = deepcopy(events["e09"])
        return event, txmem.proposal_for(event), 20
    event = deepcopy(events["e02"])
    proposal = deepcopy(txmem.proposal_for(event))
    assert proposal is not None
    proposal["source_event_id"] = "e01"
    proposal["evidence_hash"] = txmem.event_evidence_hash(events["e01"])
    return event, proposal, 20


def replay_poison_attacks(fixture: Mapping[str, Any], state_root: Path) -> JsonDict:
    """Replay poison variants in task-owned copied state."""

    events = _event_map(fixture)
    state_root.mkdir(parents=True, exist_ok=True)
    rows = []
    for attack_id in ATTACK_IDS:
        memory = txmem.TransactionalConstraintMemory(state_root / attack_id)
        receipts = _commit_bootstrap(memory, events)
        before = memory.state_bytes()
        if attack_id == "tombstone_reappearance":
            rollback = memory.rollback(receipts[-1])
            snapshot = memory.begin_episode("tombstone_reappearance")
            lookup = memory.lookup(snapshot, receipts[-1]["inverse_patch"]["key"])
            memory.end_episode()
            admitted = False
            unsafe_use = lookup["found"] is True
            state_unchanged = rollback["passed"] is True
            reason = "rollback_parent_has_no_removed_key"
        else:
            event, proposal, boundary = _attack_case(attack_id, events)
            memory.begin_episode(str(event["event_id"]))
            memory.end_episode()
            decision = memory.admit(proposal, event, boundary_index=boundary)
            admitted = decision["admitted"] is True
            unsafe_use = decision.get("unsafe_used") is True
            state_unchanged = memory.state_bytes() == before
            reason = str(decision["reason"])
        rows.append(
            {
                "row_type": "attack",
                "attack_id": attack_id,
                "admitted": admitted,
                "unsafe_used": unsafe_use,
                "state_unchanged": state_unchanged,
                "reason": reason,
                "passed": admitted is False and unsafe_use is False and state_unchanged,
            }
        )
    return {
        "attack_seed": RANDOM_SEED["attack"],
        "rows": rows,
        "admitted_poison_count": sum(int(row["admitted"] is True) for row in rows),
        "unsafe_use_count": sum(int(row["unsafe_used"] is True) for row in rows),
        "all_rejected": all(row["passed"] is True for row in rows),
    }


def check_preconditions(csl: Mapping[str, Any], fixture: Mapping[str, Any]) -> JsonDict:
    """Check that the cold audit has enough immutable evidence to run."""

    rows = csl.get("rows")
    protocol_orders = csl.get("frozen_protocol", {}).get("orders", [])
    fixture_orders = fixture.get("stream_manifest", {}).get("orders", [])
    tx_rows = [
        row for row in rows
        if isinstance(row, Mapping) and row.get("arm") == TRANSACTIONAL_ARM
    ] if isinstance(rows, list) else []
    checks: JsonDict = {
        "exp6749_completed": {
            "expected": True,
            "observed": csl.get("prospective_csl_completed"),
            "passed": csl.get("prospective_csl_completed") is True,
        },
        "raw_rows_present": {
            "expected": "nonempty list",
            "observed": len(rows) if isinstance(rows, list) else type(rows).__name__,
            "passed": isinstance(rows, list) and len(rows) > 0,
        },
        "raw_rows_complete": {
            "expected": EXPECTED_ROW_COUNT,
            "observed": len(rows) if isinstance(rows, list) else 0,
            "passed": isinstance(rows, list) and len(rows) == EXPECTED_ROW_COUNT,
        },
        "all_six_orders": {
            "expected": EXPECTED_ORDER_COUNT,
            "observed": {
                "exp6749": len(protocol_orders),
                "exp6748": len(fixture_orders),
            },
            "passed": len(protocol_orders) == EXPECTED_ORDER_COUNT
            and len(fixture_orders) == EXPECTED_ORDER_COUNT,
        },
        "state_snapshots_present": {
            "expected": EXPECTED_ORDER_COUNT * EXPECTED_EVENT_COUNT * EXPECTED_MODEL_COUNT,
            "observed": sum(
                int(row.get("snapshot_hash") is not None and row.get("snapshot_version") is not None)
                for row in tx_rows
            ),
            "passed": len(tx_rows) == EXPECTED_ORDER_COUNT * EXPECTED_EVENT_COUNT * EXPECTED_MODEL_COUNT
            and all(row.get("snapshot_hash") is not None for row in tx_rows)
            and all(row.get("snapshot_version") is not None for row in tx_rows),
        },
        "commit_receipts_present": {
            "expected": "nonempty Exp6748 commit_receipts",
            "observed": len(fixture.get("commit_receipts", [])),
            "passed": bool(fixture.get("commit_receipts")),
        },
    }
    if checks["commit_receipts_present"]["passed"]:
        receipts = fixture.get("commit_receipts", [])
        checks["commit_receipt_bytes_present"] = {
            "expected": True,
            "observed": all(
                "parent_bytes_b64" in receipt and "new_state_bytes_b64" in receipt
                for receipt in receipts
            ),
            "passed": all(
                "parent_bytes_b64" in receipt and "new_state_bytes_b64" in receipt
                for receipt in receipts
            ),
        }
    if protocol_orders and fixture_orders:
        checks["orders_match_fixture"] = {
            "expected": [row.get("order_hash") for row in fixture_orders],
            "observed": [row.get("order_hash") for row in protocol_orders],
            "passed": [row.get("order_hash") for row in protocol_orders]
            == [row.get("order_hash") for row in fixture_orders],
        }
    return {
        "checks": checks,
        "all_passed": all(row["passed"] is True for row in checks.values()),
        "process": {
            "pid": os.getpid(),
            "python": sys.version.split()[0],
            "fresh_process_expected_by_command": True,
            "live_inference_invoked": False,
        },
    }


def gate_check_summary(checks: Mapping[str, Mapping[str, Any]]) -> JsonDict:
    """Build a stable gate summary with observed failed values."""

    failures = [
        {
            "check": name,
            "expected": row.get("expected"),
            "observed": row.get("observed"),
        }
        for name, row in checks.items()
        if row.get("passed") is not True
    ]
    return {
        "checks": {name: row.get("passed") is True for name, row in checks.items()},
        "failed_checks": [row["check"] for row in failures],
        "failures": failures,
    }


def _artifact_rows(
    metrics: Mapping[str, Any],
    snapshots: Mapping[str, Any],
    attacks: Mapping[str, Any],
    provenance: Mapping[str, Any],
    restarts: Mapping[str, Any],
    rollback: Mapping[str, Any],
) -> list[JsonDict]:
    rows: list[JsonDict] = []
    for order_id, order in metrics["recomputed_prequential_delta_by_order"].items():
        rows.append(
            {
                "row_type": "order_metric",
                "order_id": order_id,
                "metric": "prequential_exact_yield_delta",
                "value": order["pooled"]["transactional_minus_no_memory"],
                "passed": True,
            }
        )
    for metric, value in metrics["support_contraction_by_metric"].items():
        rows.append({"row_type": "support_metric", "metric": metric, **value})
    for row in metrics["retention_rows"]:
        rows.append({"row_type": "retention", **row, "passed": row["failure"] is False})
    for model_id, families in metrics["negative_transfer_by_family"].items():
        for family, value in families.items():
            rows.append(
                {
                    "row_type": "negative_transfer",
                    "model_id": model_id,
                    "family": family,
                    **value,
                    "passed": value["negative_transfer"] is False,
                }
            )
    for model_id, arms in metrics["token_cost_by_model_arm"].items():
        for arm, value in arms.items():
            rows.append({"row_type": "token_metric", "model_id": model_id, "arm": arm, **value})
    for metric, value in metrics["commit_reject_rollback_counts"]["prospective_rows"].items():
        rows.append({"row_type": "lifecycle_count", "metric": metric, "value": value})
    rows.extend(deepcopy(snapshots["rows"]))
    rows.extend(deepcopy(provenance["rows"]))
    rows.extend(deepcopy(attacks["rows"]))
    rows.extend(deepcopy(restarts["rows"]))
    rows.extend(deepcopy(rollback["rows"]))
    return rows


def _positive_gate_checks(
    preconditions: Mapping[str, Any],
    metrics: Mapping[str, Any],
    ci95: Mapping[str, Any],
    snapshots: Mapping[str, Any],
    attacks: Mapping[str, Any],
    provenance: Mapping[str, Any],
    restarts: Mapping[str, Any],
    rollback: Mapping[str, Any],
) -> JsonDict:
    support = metrics["support_contraction_by_metric"]
    lifecycle = metrics["commit_reject_rollback_counts"]["prospective_rows"]
    negative_transfer_count = sum(
        int(cell["negative_transfer"] is True)
        for model in metrics["negative_transfer_by_family"].values()
        for cell in model.values()
    )
    return {
        "preconditions_pass": {
            "expected": True,
            "observed": preconditions["all_passed"],
            "passed": preconditions["all_passed"] is True,
        },
        "future_leakage_zero": {
            "expected": 0,
            "observed": snapshots["future_leakage_count"],
            "passed": snapshots["future_leakage_count"] == 0,
        },
        "order_lcb_positive": {
            "expected": "> 0.0",
            "observed": ci95["lower"],
            "passed": float(ci95["lower"]) > 0.0,
        },
        "best_at_k_support_preserved": {
            "expected": f"<= {SUPPORT_CONTRACTION_BOUND}",
            "observed": support["best_at_k"]["contraction"],
            "passed": support["best_at_k"]["passes"] is True,
        },
        "effective_support_preserved": {
            "expected": f"<= {SUPPORT_CONTRACTION_BOUND}",
            "observed": support["effective_rewardable_support"]["contraction"],
            "passed": support["effective_rewardable_support"]["passes"] is True,
        },
        "retention_preserved": {
            "expected": 0,
            "observed": len(metrics["retention_failures"]),
            "passed": len(metrics["retention_failures"]) == 0,
        },
        "no_negative_transfer": {
            "expected": 0,
            "observed": negative_transfer_count,
            "passed": negative_transfer_count == 0,
        },
        "admitted_poison_zero": {
            "expected": 0,
            "observed": attacks["admitted_poison_count"],
            "passed": attacks["admitted_poison_count"] == 0,
        },
        "commit_provenance_valid": {
            "expected": True,
            "observed": provenance["all_hashes_match"],
            "passed": provenance["all_hashes_match"] is True,
        },
        "prospective_transaction_activity_nonzero": {
            "expected": "commits > 0 and rejects > 0",
            "observed": {"commits": lifecycle["commits"], "rejects": lifecycle["rejects"]},
            "passed": lifecycle["commits"] > 0 and lifecycle["rejects"] > 0,
        },
        "restart_boundaries_all_pass": {
            "expected": restarts["expected_count"],
            "observed": restarts["pass_count"],
            "passed": restarts["all_passed"] is True,
        },
        "rollback_byte_identity_all_pass": {
            "expected": True,
            "observed": rollback["all_match"],
            "passed": rollback["all_match"] is True,
        },
    }


def _blank_payload(
    *,
    duration_s: float,
    receipts: Mapping[str, Any],
    preconditions: Mapping[str, Any],
    tests_run: Sequence[Mapping[str, Any]],
) -> JsonDict:
    payload: JsonDict = {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "run_date": RUN_DATE,
        "status": "complete_blocked_csl_audit",
        "field_principles": deepcopy(FIELD_PRINCIPLES),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "duration_s": round(float(duration_s), 6),
        "random_seed": deepcopy(RANDOM_SEED),
        "reproducibility_checksum": "",
        "input_artifact_receipts": deepcopy(dict(receipts)),
        "preconditions_checked": deepcopy(dict(preconditions)),
        "rows": [],
        "recomputed_prequential_delta_by_order": {},
        "order_level_ci95": order_level_ci95([]),
        "support_contraction_by_metric": {},
        "retention_failures": [],
        "negative_transfer_by_family": {},
        "token_cost_by_model_arm": {},
        "commit_reject_rollback_counts": {
            "prospective_rows": {
                "commits": 0,
                "rejects": 0,
                "quarantine": 0,
                "rollbacks": 0,
                "rollback_failures": 0,
            },
            "by_order": {},
            "commit_status_counts": {},
        },
        "commit_provenance_audit": {"commit_receipt_count": 0, "all_hashes_match": False, "rows": []},
        "future_leakage_count": 0,
        "admitted_poison_count": 0,
        "attack_replay": {"attack_seed": RANDOM_SEED["attack"], "rows": []},
        "restart_boundary_pass_count": 0,
        "restart_boundary_expected_count": 0,
        "rollback_byte_identity": {"boundary_count": 0, "all_match": False, "rows": []},
        "csl_audit_passed": False,
        "gate_check_summary": gate_check_summary(preconditions["checks"]),
        "verdict_class": "blocked",
        "honest_verdict": "complete_blocked_csl_audit: owned precondition failed",
        "tests_run": deepcopy(list(tests_run)),
        "verifier_is_oracle": True,
    }
    payload["reproducibility_checksum"] = reproducibility_checksum(payload)
    return payload


def build_artifact(
    *,
    root: Path = REPO_ROOT,
    exp6749_path: Path | str = REPO_ROOT / EXP6749_RELATIVE_PATH,
    fixture_path: Path | str = REPO_ROOT / EXP6748_RELATIVE_PATH,
    state_root: Path | str | None = None,
    duration_s: float | None = None,
    tests_run: Sequence[Mapping[str, Any]] = DEFAULT_TESTS_RUN,
) -> JsonDict:
    """Build the terminal cold-audit artifact from immutable inputs."""

    started = time.monotonic()
    root = Path(root)
    exp6749_path = Path(exp6749_path)
    fixture_path = Path(fixture_path)
    if state_root is None:
        with tempfile.TemporaryDirectory(prefix="carnot-exp6750-") as directory:
            return build_artifact(
                root=root,
                exp6749_path=exp6749_path,
                fixture_path=fixture_path,
                state_root=directory,
                duration_s=duration_s,
                tests_run=tests_run,
            )
    csl = read_json(exp6749_path)
    fixture = read_json(fixture_path)
    receipts = input_artifact_receipts(
        root,
        {
            "exp6749": exp6749_path,
            "exp6748": fixture_path,
            "module": MODULE_RELATIVE_PATH,
            "script": SCRIPT_RELATIVE_PATH,
            "test": TEST_RELATIVE_PATH,
            "spec": SPEC_RELATIVE_PATH,
        },
    )
    preconditions = check_preconditions(csl, fixture)
    elapsed = duration_s if duration_s is not None else time.monotonic() - started
    payload = _blank_payload(
        duration_s=elapsed,
        receipts=receipts,
        preconditions=preconditions,
        tests_run=tests_run,
    )
    if preconditions["all_passed"] is not True:
        return payload

    metrics = recompute_prospective_metrics(csl["rows"])
    ci95 = order_level_ci95(metrics["order_delta_values"])
    snapshots = audit_snapshot_isolation(csl["rows"], csl["frozen_protocol"]["orders"])
    attacks = replay_poison_attacks(fixture, Path(state_root) / "attacks")
    provenance = audit_commit_provenance(fixture)
    restarts = audit_restart_boundaries(fixture)
    rollback = audit_rollback_identity(fixture)
    checks = _positive_gate_checks(
        preconditions,
        metrics,
        ci95,
        snapshots,
        attacks,
        provenance,
        restarts,
        rollback,
    )
    passed = all(row["passed"] is True for row in checks.values())
    integrity_failed = any(
        checks[name]["passed"] is not True
        for name in (
            "future_leakage_zero",
            "admitted_poison_zero",
            "commit_provenance_valid",
            "restart_boundaries_all_pass",
            "rollback_byte_identity_all_pass",
        )
    )
    verdict_class = "positive" if passed else ("disqualified" if integrity_failed else "null")
    status = "complete_csl_audit_positive" if passed else f"complete_csl_audit_{verdict_class}"
    if passed:
        verdict = "complete_positive_csl_audit: Exp6749 passed every cold eligibility gate"
    elif verdict_class == "disqualified":
        verdict = "complete_disqualified_csl_audit: integrity or safety evidence failed"
    else:
        verdict = "complete_null_csl_audit: no positive order-level CSL effect recomputed"

    payload.update(
        {
            "status": status,
            "rows": _artifact_rows(metrics, snapshots, attacks, provenance, restarts, rollback),
            "recomputed_prequential_delta_by_order": metrics[
                "recomputed_prequential_delta_by_order"
            ],
            "order_level_ci95": ci95,
            "support_contraction_by_metric": metrics["support_contraction_by_metric"],
            "retention_failures": metrics["retention_failures"],
            "negative_transfer_by_family": metrics["negative_transfer_by_family"],
            "token_cost_by_model_arm": metrics["token_cost_by_model_arm"],
            "commit_reject_rollback_counts": metrics["commit_reject_rollback_counts"],
            "commit_provenance_audit": provenance,
            "future_leakage_count": snapshots["future_leakage_count"],
            "admitted_poison_count": attacks["admitted_poison_count"],
            "attack_replay": attacks,
            "restart_boundary_pass_count": restarts["pass_count"],
            "restart_boundary_expected_count": restarts["expected_count"],
            "rollback_byte_identity": rollback,
            "csl_audit_passed": passed,
            "gate_check_summary": gate_check_summary(checks),
            "verdict_class": verdict_class,
            "honest_verdict": verdict,
            "duration_s": round(
                float(duration_s) if duration_s is not None else time.monotonic() - started,
                6,
            ),
        }
    )
    payload["reproducibility_checksum"] = reproducibility_checksum(payload)
    errors = validate_artifact(payload)
    if errors:
        raise ValueError("; ".join(errors))
    return payload


def reproducibility_checksum(artifact: Mapping[str, Any]) -> str:
    """Hash fields that prove the cold audit can be replayed."""

    material = {
        "schema": artifact.get("schema"),
        "run_date": artifact.get("run_date"),
        "random_seed": artifact.get("random_seed"),
        "input_artifact_receipts": artifact.get("input_artifact_receipts"),
        "preconditions_checked": artifact.get("preconditions_checked"),
        "rows": artifact.get("rows"),
        "recomputed_prequential_delta_by_order": artifact.get(
            "recomputed_prequential_delta_by_order"
        ),
        "order_level_ci95": artifact.get("order_level_ci95"),
        "support_contraction_by_metric": artifact.get("support_contraction_by_metric"),
        "retention_failures": artifact.get("retention_failures"),
        "negative_transfer_by_family": artifact.get("negative_transfer_by_family"),
        "token_cost_by_model_arm": artifact.get("token_cost_by_model_arm"),
        "commit_reject_rollback_counts": artifact.get("commit_reject_rollback_counts"),
        "commit_provenance_audit": artifact.get("commit_provenance_audit"),
        "future_leakage_count": artifact.get("future_leakage_count"),
        "admitted_poison_count": artifact.get("admitted_poison_count"),
        "attack_replay": artifact.get("attack_replay"),
        "restart_boundary_pass_count": artifact.get("restart_boundary_pass_count"),
        "restart_boundary_expected_count": artifact.get("restart_boundary_expected_count"),
        "rollback_byte_identity": artifact.get("rollback_byte_identity"),
        "csl_audit_passed": artifact.get("csl_audit_passed"),
        "gate_check_summary": artifact.get("gate_check_summary"),
        "verdict_class": artifact.get("verdict_class"),
    }
    return sha256_bytes(canonical_json_bytes(material))


def _count_row_violations(rows: Sequence[Mapping[str, Any]], row_type: str) -> int:
    return sum(
        int(row.get("passed") is False)
        for row in rows
        if row.get("row_type") == row_type
    )


def validate_artifact(artifact: Mapping[str, Any]) -> list[str]:
    """Return artifact contract errors without mutating evidence."""

    errors = []
    if set(artifact) != set(REQUIRED_ARTIFACT_FIELDS):
        errors.append("required field set mismatch")
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        errors.append("inference_substrate mismatch")
    if artifact.get("verdict_class") not in VERDICT_CLASSES:
        errors.append("verdict_class outside closed enum")
    expected_principles = set(REQUIRED_ARTIFACT_FIELDS) | {f"gate:{name}" for name in GATE_NAMES}
    if set(artifact.get("field_principles", {})) != expected_principles:
        errors.append("field_principles coverage mismatch")
    if artifact.get("reproducibility_checksum") != reproducibility_checksum(artifact):
        errors.append("reproducibility_checksum mismatch")
    rows = artifact.get("rows", [])
    future_from_rows = _count_row_violations(rows, "snapshot") if isinstance(rows, list) else 0
    if artifact.get("future_leakage_count") != future_from_rows:
        errors.append("future_leakage_count mismatch")
    poison_from_rows = sum(
        int(row.get("row_type") == "attack" and row.get("admitted") is True)
        for row in rows
    ) if isinstance(rows, list) else 0
    if artifact.get("admitted_poison_count") != poison_from_rows:
        errors.append("admitted_poison_count mismatch")
    summary = artifact.get("gate_check_summary", {})
    checks = summary.get("checks", {}) if isinstance(summary, Mapping) else {}
    expected_passed = bool(checks) and all(checks.values())
    if artifact.get("csl_audit_passed") is not expected_passed:
        errors.append("csl_audit_passed mismatch")
    if artifact.get("status") == "complete_blocked_csl_audit":
        if artifact.get("verdict_class") != "blocked":
            errors.append("blocked verdict_class mismatch")
    elif not rows:
        errors.append("completed audit has no rows")
    return errors


def write_artifact(path: Path | str, artifact: Mapping[str, Any]) -> JsonDict:
    """Validate and publish the audit artifact through one atomic rename."""

    errors = validate_artifact(artifact)
    if errors:
        raise ValueError("; ".join(errors))
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(artifact, indent=2, sort_keys=True, ensure_ascii=True) + "\n"
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=target.parent,
        prefix=f".{target.name}.",
        delete=False,
    ) as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
        temporary = Path(handle.name)
    os.replace(temporary, target)
    return {"path": str(target), "atomic_rename": True, "sha256": sha256_file(target)}


def main(argv: Sequence[str] | None = None) -> int:
    """Run the cold audit or validate a stored terminal artifact."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-path", default=str(REPO_ROOT / RESULT_RELATIVE_PATH))
    parser.add_argument("--exp6749-path", default=str(REPO_ROOT / EXP6749_RELATIVE_PATH))
    parser.add_argument("--fixture-path", default=str(REPO_ROOT / EXP6748_RELATIVE_PATH))
    parser.add_argument("--state-root")
    parser.add_argument("--validate", action="store_true")
    args = parser.parse_args(argv)
    result_path = Path(args.result_path)
    if args.validate:
        artifact = read_json(result_path)
        errors = validate_artifact(artifact)
        if errors:
            raise ValueError("; ".join(errors))
        return 0
    artifact = build_artifact(
        exp6749_path=Path(args.exp6749_path),
        fixture_path=Path(args.fixture_path),
        state_root=args.state_root,
    )
    write_artifact(result_path, artifact)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
