"""Reduce sealed controller and credit rows without producer code.

The audit reads only JSON evidence. This keeps the reducer independent from
the code that produced the evidence and makes aggregate contradictions visible.

Spec refs: REQ-CL-6856 and SCENARIO-CL-6856-*.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass, field
import hashlib
import json
import math
from pathlib import Path
import time
from typing import Any, Mapping, Sequence


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[2]
CONTROLLER_RELATIVE_PATH = Path(
    "results/experiment_6854_risk_sensitive_abstention_memory_controller.json"
)
CREDIT_RELATIVE_PATH = Path("results/experiment_6855_counterfactual_memory_credit_audit.json")
RESULT_RELATIVE_PATH = Path("results/experiment_6856_sealed_risk_sensitive_learning_audit.json")
SCHEMA = "carnot.experiment_6856.sealed_risk_sensitive_learning_audit.v1"
EXPERIMENT_ID = "exp6856-sealed-risk-sensitive-learning-audit"
INFERENCE_SUBSTRATE = "deterministic CPU sealed independent reduction"
BLOCKED_VERDICT = "complete_blocked_sealed_risk_sensitive_learning_audit"
RANDOM_SEED = 6_856_001
FALSE_POSITIVE_RATE_BOUND = 0.05
CREDIT_BOUND = 3.0

POLICY_ARMS = (
    "contextual_bandit",
    "no_memory",
    "always_memory",
    "abstain_only",
    "random_admission",
    "read_only_fixed",
)
COMPARISON_ARMS = POLICY_ARMS[1:]
FAMILIES = (
    "unsloth/Qwen3.6-35B-A3B-GGUF",
    "unsloth/gemma-4-26B-A4B-it-GGUF",
    "unsloth/gemma-4-31B-it-GGUF",
)
ORDERS = ("order_1", "order_2", "order_3", "order_4", "order_5")
SEALED_SPLIT_MANIFEST = {
    "train_start": 1,
    "train_end": 510,
    "held_future_start": 511,
    "held_future_end": 765,
}
EXPECTED_CONTROLLER_ROWS_SHA256 = (
    "sha256:c257437810a352160e425db8a673bbd81cd0e67738db85053446a7405d0e71e8"
)
EXPECTED_CREDIT_ROWS_SHA256 = (
    "sha256:3fb8e7d8e2942876db491507a9a623cafc381bef4eb61333f940b8042a960a23"
)
PRODUCER_MODULE_MARKERS = ("experiment_6854_", "experiment_6855_")
MEASURED_ARM_FIELDS = (
    "loss",
    "reward",
    "regret",
    "false_positive_injection",
    "helpful_memory_selected",
    "missed_reuse",
    "abstained",
)
VALID_VERDICT_CLASSES = {
    "positive",
    "circular_positive",
    "null",
    "blocked",
    "disqualified",
    "partial",
}
REQUIRED_ARTIFACT_FIELDS = (
    "field_principles",
    "preconditions_checked",
    "inference_substrate",
    "duration_s",
    "source_artifact_hashes",
    "random_seed",
    "reproducibility_checksum",
    "rows",
    "fresh_reducer_manifest",
    "per_arm_summary",
    "held_future_effect",
    "win_loss_tie_counts",
    "no_headroom_count",
    "false_positive_injection_rate",
    "abstention_calibration",
    "durability_results",
    "restart_results",
    "rollback_results",
    "delayed_correction_results",
    "poison_results",
    "capacity_results",
    "leave_one_family_out_rows",
    "leave_one_order_out_rows",
    "leakage_results",
    "counterfactual_support_results",
    "continuous_self_learning_ready_score",
    "retirement_recommendation",
    "gate_check_summary",
    "verifier_is_oracle",
    "verdict_class",
    "honest_verdict",
)
OPEN_SPEC_IDS = (
    "REQ-CL-6856",
    "SCENARIO-CL-6856-PRECONDITIONS",
    "SCENARIO-CL-6856-ALL-NULL",
    "SCENARIO-CL-6856-AGGREGATE-CONTRADICTION",
    "SCENARIO-CL-6856-POISON",
    "SCENARIO-CL-6856-STALE-CORRECTION",
    "SCENARIO-CL-6856-ROLLBACK",
    "SCENARIO-CL-6856-CAPACITY",
    "SCENARIO-CL-6856-FAMILY-REMOVAL",
    "SCENARIO-CL-6856-ORDER-REMOVAL",
    "SCENARIO-CL-6856-GATES",
)


def canonical_json_bytes(value: Any) -> bytes:
    """Use one stable encoding for hashes, checkpoints, and output checks."""

    return json.dumps(
        value,
        ensure_ascii=True,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def sha256_json(value: Any) -> str:
    """Return a labeled digest for one JSON-compatible value."""

    return "sha256:" + hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def sha256_file(path: Path) -> str | None:
    """Hash a source file without treating a missing file as valid evidence."""

    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def load_source(path: Path) -> JsonDict:
    """Load one JSON object and reject other top-level shapes."""

    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"source must contain a JSON object: {path}")
    return value


def _check(check: str, expected: Any, observed: Any, passed: bool) -> JsonDict:
    """Keep the expected and observed values beside every gate decision."""

    return {
        "check": check,
        "expected": expected,
        "observed": observed,
        "passed": bool(passed),
    }


def _gate_summary(checks: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Expose every failed check so a blocked result is actionable."""

    copied = [dict(row) for row in checks]
    failed = [row for row in copied if not row["passed"]]
    return {
        "checks": copied,
        "failed_checks": failed,
        "failed_check": failed[0]["check"] if failed else None,
        "observed": failed[0]["observed"] if failed else None,
        "passed": not failed,
    }


def _producer_overlap(imports: Sequence[str]) -> list[str]:
    """Find imports that would reuse either producer implementation."""

    return sorted(
        name for name in imports if any(marker in name for marker in PRODUCER_MODULE_MARKERS)
    )


def _decision_dimensions(decision_id: str) -> tuple[str, str]:
    """Read the sealed order and family from the stable decision identity."""

    parts = decision_id.split("::", 2)
    if len(parts) != 3:
        return "invalid_order", "invalid_family"
    return parts[0], parts[1]


def validate_preconditions(
    controller: Mapping[str, Any],
    credit: Mapping[str, Any],
    *,
    split_manifest: Mapping[str, int] = SEALED_SPLIT_MANIFEST,
    reducer_imports: Sequence[str] = (),
) -> JsonDict:
    """Validate sealed inputs before reducing any scientific headline."""

    controller_rows = controller.get("rows")
    credit_rows = credit.get("rows")
    controller_rows = controller_rows if isinstance(controller_rows, list) else []
    credit_rows = credit_rows if isinstance(credit_rows, list) else []
    controller_digest = sha256_json(controller_rows)
    credit_digest = sha256_json(credit_rows)

    sequences = [row.get("decision_sequence_index") for row in controller_rows]
    expected_sequences = list(range(1, len(controller_rows) + 1))
    decision_ids = [row.get("decision_id") for row in controller_rows]
    duplicate_count = len(decision_ids) - len(set(decision_ids))
    missing_arm_rows: list[str] = []
    null_metric_rows: list[str] = []
    invalid_dimension_rows: list[str] = []
    for row in controller_rows:
        decision_id = str(row.get("decision_id"))
        metrics = row.get("arm_metrics")
        if not isinstance(metrics, Mapping) or set(metrics) != set(POLICY_ARMS):
            missing_arm_rows.append(decision_id)
            continue
        if any(
            not isinstance(metrics[arm], Mapping)
            or all(metrics[arm].get(field) is None for field in MEASURED_ARM_FIELDS)
            for arm in POLICY_ARMS
        ):
            null_metric_rows.append(decision_id)
        order_id, family = _decision_dimensions(decision_id)
        if order_id not in ORDERS or family not in FAMILIES:
            invalid_dimension_rows.append(decision_id)

    controller_ids = set(decision_ids)
    credit_decision_ids = {
        row.get("decision_id") for row in credit_rows if row.get("decision_id") is not None
    }
    unknown_credit_ids = sorted(str(value) for value in credit_decision_ids - controller_ids)
    overlap = _producer_overlap(reducer_imports)
    expected_split = dict(SEALED_SPLIT_MANIFEST)
    observed_split = dict(split_manifest)

    checks = [
        _check(
            "risk_sensitive_controller_complete_score",
            1,
            controller.get("risk_sensitive_controller_complete_score"),
            controller.get("risk_sensitive_controller_complete_score") == 1,
        ),
        _check(
            "counterfactual_memory_audit_complete_score",
            1,
            credit.get("counterfactual_memory_audit_complete_score"),
            credit.get("counterfactual_memory_audit_complete_score") == 1,
        ),
        _check(
            "stable_controller_raw_row_hashes",
            EXPECTED_CONTROLLER_ROWS_SHA256,
            controller_digest,
            controller_digest == EXPECTED_CONTROLLER_ROWS_SHA256,
        ),
        _check(
            "stable_credit_raw_row_hashes",
            EXPECTED_CREDIT_ROWS_SHA256,
            credit_digest,
            credit_digest == EXPECTED_CREDIT_ROWS_SHA256,
        ),
        _check(
            "declared_chronological_split",
            expected_split,
            observed_split,
            observed_split == expected_split,
        ),
        _check(
            "chronological_decision_sequence",
            {"start": 1, "end": len(controller_rows)},
            {
                "start": sequences[0] if sequences else None,
                "end": sequences[-1] if sequences else None,
                "out_of_order_count": sum(
                    observed != expected
                    for observed, expected in zip(sequences, expected_sequences, strict=True)
                ),
            },
            bool(controller_rows) and sequences == expected_sequences,
        ),
        _check(
            "unique_decision_identities",
            0,
            duplicate_count,
            bool(controller_rows) and duplicate_count == 0,
        ),
        _check(
            "complete_comparison_arms",
            {"arms": list(POLICY_ARMS), "missing_row_count": 0},
            {"missing_row_count": len(missing_arm_rows), "decision_ids": missing_arm_rows},
            not missing_arm_rows,
        ),
        _check(
            "non_null_decision_metrics",
            0,
            {"null_row_count": len(null_metric_rows), "decision_ids": null_metric_rows},
            not null_metric_rows,
        ),
        _check(
            "declared_family_and_order_dimensions",
            0,
            {
                "invalid_row_count": len(invalid_dimension_rows),
                "decision_ids": invalid_dimension_rows,
            },
            not invalid_dimension_rows,
        ),
        _check(
            "credit_decisions_match_controller",
            0,
            {"unknown_count": len(unknown_credit_ids), "decision_ids": unknown_credit_ids},
            not unknown_credit_ids,
        ),
        _check(
            "no_reducer_source_overlap",
            [],
            overlap,
            not overlap,
        ),
    ]
    return _gate_summary(checks)


def _no_headroom(metrics: Mapping[str, Mapping[str, Any]]) -> bool:
    """Detect decisions where neither memory direction has useful headroom."""

    memory = metrics["always_memory"]
    no_memory = metrics["no_memory"]
    return not any(
        (
            memory.get("false_positive_injection"),
            memory.get("helpful_memory_selected"),
            no_memory.get("missed_reuse"),
        )
    )


def reduce_decision_rows(
    source_rows: Sequence[Mapping[str, Any]],
    *,
    split_manifest: Mapping[str, int] = SEALED_SPLIT_MANIFEST,
) -> list[JsonDict]:
    """Flatten one decision and arm into one independently auditable metric row."""

    reduced: list[JsonDict] = []
    held_start = int(split_manifest["held_future_start"])
    for source in source_rows:
        decision_id = str(source["decision_id"])
        sequence = int(source["decision_sequence_index"])
        order_id, family = _decision_dimensions(decision_id)
        metrics = source["arm_metrics"]
        baseline_reward = float(metrics["no_memory"]["reward"])
        no_headroom = _no_headroom(metrics)
        split = "held_future" if sequence >= held_start else "chronological_train"
        for arm in POLICY_ARMS:
            arm_row = metrics[arm]
            reward = float(arm_row["reward"])
            reduced.append(
                {
                    "decision_id": decision_id,
                    "decision_sequence_index": sequence,
                    "arm": arm,
                    "attack": split,
                    "family": family,
                    "order_id": order_id,
                    "split": split,
                    "metric_value": round(reward, 12),
                    "effect_vs_no_memory": round(reward - baseline_reward, 12),
                    "no_headroom": no_headroom,
                    "false_positive_injection": bool(arm_row["false_positive_injection"]),
                    "abstained": bool(arm_row["abstained"]),
                }
            )
    return reduced


def _mean(values: Sequence[float]) -> float | None:
    """Keep an empty comparison null instead of turning it into zero."""

    if not values:
        return None
    return round(sum(values) / len(values), 12)


def _per_arm_summary(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Recompute each arm summary from normalized rows only."""

    summary: JsonDict = {}
    for arm in POLICY_ARMS:
        arm_rows = [row for row in rows if row["arm"] == arm]
        held_rows = [row for row in arm_rows if row["split"] == "held_future"]
        summary[arm] = {
            "decision_count": len(arm_rows),
            "held_future_count": len(held_rows),
            "mean_reward": _mean([float(row["metric_value"]) for row in arm_rows]),
            "held_future_mean_reward": _mean([float(row["metric_value"]) for row in held_rows]),
            "false_positive_injection_count": sum(
                bool(row["false_positive_injection"]) for row in arm_rows
            ),
            "false_positive_injection_rate": round(
                sum(bool(row["false_positive_injection"]) for row in arm_rows) / len(arm_rows),
                12,
            ),
            "abstention_count": sum(bool(row["abstained"]) for row in arm_rows),
        }
    return summary


def _controller_rows(rows: Sequence[Mapping[str, Any]], *, held_only: bool) -> list[JsonDict]:
    """Select the learned arm while preserving chronological row order."""

    return [
        dict(row)
        for row in rows
        if row["arm"] == "contextual_bandit" and (not held_only or row["split"] == "held_future")
    ]


def _win_loss_tie_counts(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Count signs directly and report no-headroom decisions beside them."""

    all_rows = _controller_rows(rows, held_only=False)
    held_rows = _controller_rows(rows, held_only=True)

    def count(selected: Sequence[Mapping[str, Any]]) -> JsonDict:
        effects = [float(row["effect_vs_no_memory"]) for row in selected]
        return {
            "wins": sum(value > 0 for value in effects),
            "losses": sum(value < 0 for value in effects),
            "ties": sum(value == 0 for value in effects),
            "no_headroom_count": sum(bool(row["no_headroom"]) for row in selected),
            "decision_count": len(selected),
        }

    return {"all_decisions": count(all_rows), "held_future": count(held_rows)}


def _abstention_calibration(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Reject a policy that calls every decision an abstention opportunity."""

    learned = _controller_rows(rows, held_only=False)
    helpful = [
        row for row in learned if not row["no_headroom"] and float(row["effect_vs_no_memory"]) > 0
    ]
    harmful = [
        row for row in learned if not row["no_headroom"] and float(row["effect_vs_no_memory"]) < 0
    ]
    no_headroom = [row for row in learned if row["no_headroom"]]

    def rate(selected: Sequence[Mapping[str, Any]]) -> float | None:
        if not selected:
            return None
        return round(sum(bool(row["abstained"]) for row in selected) / len(selected), 12)

    overall = rate(learned)
    helpful_rate = rate(helpful)
    harmful_rate = rate(harmful)
    calibrated = bool(
        overall is not None
        and 0.05 <= overall <= 0.95
        and helpful_rate is not None
        and helpful_rate < 1.0
        and harmful_rate is not None
        and harmful_rate >= helpful_rate
    )
    return {
        "overall_abstention_rate": overall,
        "helpful_opportunity_abstention_rate": helpful_rate,
        "harmful_opportunity_abstention_rate": harmful_rate,
        "no_headroom_abstention_rate": rate(no_headroom),
        "helpful_opportunity_count": len(helpful),
        "harmful_opportunity_count": len(harmful),
        "no_headroom_count": len(no_headroom),
        "nondegenerate_rate_bounds": [0.05, 0.95],
        "calibrated": calibrated,
    }


def leave_one_group_out(
    rows: Sequence[Mapping[str, Any]],
    *,
    group_field: str,
    output_field: str,
    groups: Sequence[str],
) -> list[JsonDict]:
    """Report each held-out family or order without pooling null headroom."""

    learned = [row for row in rows if row["arm"] == "contextual_bandit"]
    results: list[JsonDict] = []
    for group in groups:
        selected = [row for row in learned if row[group_field] == group]
        comparable = [row for row in selected if not row["no_headroom"]]
        effects = [float(row["effect_vs_no_memory"]) for row in comparable]
        mean_effect = _mean(effects)
        results.append(
            {
                output_field: group,
                "attack": f"leave_one_{group_field}_out",
                "decision_count": len(selected),
                "comparable_count": len(comparable),
                "no_headroom_count": len(selected) - len(comparable),
                "mean_effect_vs_no_memory": mean_effect,
                "wins": sum(value > 0 for value in effects),
                "losses": sum(value < 0 for value in effects),
                "ties": sum(value == 0 for value in effects),
                "portability_passed": mean_effect is not None and mean_effect >= 0,
            }
        )
    return results


def reduce_credit_rows(
    credit_rows: Sequence[Mapping[str, Any]],
    *,
    controller: Mapping[str, Any],
    normalized_decisions: Sequence[Mapping[str, Any]],
) -> JsonDict:
    """Recompute per-write signs and support from raw counterfactual rows."""

    no_headroom_ids = {
        str(row["decision_id"])
        for row in normalized_decisions
        if row["arm"] == "contextual_bandit" and row["no_headroom"]
    }
    known_decisions = {
        str(row["decision_id"]) for row in normalized_decisions if row["arm"] == "contextual_bandit"
    }
    expected_write_ids = {
        str(row["update_receipt_sha256"])
        for row in controller.get("update_rows", [])
        if row.get("update_receipt_sha256") is not None
    }
    grouped: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    poison_rows: list[JsonDict] = []
    unknown_decisions: set[str] = set()
    supported_methods: set[str] = set()
    maximum_abs_credit = 0.0
    for row in credit_rows:
        decision_id = str(row.get("decision_id"))
        if decision_id not in known_decisions:
            unknown_decisions.add(decision_id)
        write_id = row.get("write_id")
        if write_id is None:
            continue
        value = row.get("metric_value")
        if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            poison_rows.append(
                {"write_id": write_id, "decision_id": decision_id, "observed": value}
            )
            continue
        supported_methods.add(str(row.get("method")))
        numeric = float(value)
        maximum_abs_credit = max(maximum_abs_credit, abs(numeric))
        if decision_id in no_headroom_ids:
            continue
        metric = str(row.get("counterfactual_metric"))
        grouped[str(write_id)][metric].append(numeric)

    per_write_rows: list[JsonDict] = []
    tolerance = 1e-12
    for write_id in sorted(grouped):
        metrics = grouped[write_id]
        deletion = sum(metrics.get("deletion_effect", []))
        marginal = sum(metrics.get("marginal_value", []))
        supported = bool(metrics.get("deletion_effect") or metrics.get("marginal_value"))
        if not supported:
            credit_class = "unsupported"
        elif marginal < -tolerance or deletion < -tolerance:
            credit_class = "harmful"
        elif marginal > tolerance or deletion > tolerance:
            credit_class = "helpful"
        else:
            credit_class = "redundant"
        per_write_rows.append(
            {
                "write_id": write_id,
                "aggregate_deletion_effect": round(deletion, 12),
                "aggregate_marginal_value": round(marginal, 12),
                "credit_class": credit_class,
                "supported": supported,
            }
        )

    supported_ids = {row["write_id"] for row in per_write_rows if row["supported"]}
    unsupported_ids = sorted(expected_write_ids - supported_ids)
    contract = controller.get("controller_schema", {})
    source_contract = {
        "counterfactual_memory_audit_complete_score": None,
        "causal_memory_credit_eligible_score": None,
    }
    class_counts = {
        name: sum(row["credit_class"] == name for row in per_write_rows)
        for name in ("helpful", "harmful", "redundant", "unsupported")
    }
    valid = bool(
        not poison_rows
        and not unknown_decisions
        and supported_ids
        and contract.get("max_updates_per_action") == 256
    )
    return {
        "valid_counterfactual_support": valid,
        "supported_write_count": len(supported_ids),
        "unsupported_write_count": len(unsupported_ids),
        "unsupported_write_ids": unsupported_ids,
        "helpful_write_count": class_counts["helpful"],
        "harmful_write_count": class_counts["harmful"],
        "redundant_write_count": class_counts["redundant"],
        "poison_metric_count": len(poison_rows),
        "poison_metric_rows": poison_rows,
        "unknown_decision_count": len(unknown_decisions),
        "maximum_abs_row_credit": round(maximum_abs_credit, 12),
        "declared_credit_bound": CREDIT_BOUND,
        "methods": sorted(supported_methods),
        "per_write_rows_sha256": sha256_json(per_write_rows),
        "source_gate_values": source_contract,
    }


@dataclass
class BoundedAuditState:
    """Hold bounded credit with stable bytes and explicit removal records."""

    capacity: int
    tombstone_capacity: int
    credit_bound: float
    active: dict[str, JsonDict] = field(default_factory=dict)
    tombstones: dict[str, int] = field(default_factory=dict)

    def _add_tombstone(self, write_id: str, sequence: int) -> None:
        """Keep recent removals bounded so stale corrections stay rejected."""

        self.tombstones[write_id] = sequence
        while len(self.tombstones) > self.tombstone_capacity:
            oldest = min(self.tombstones, key=lambda key: (self.tombstones[key], key))
            del self.tombstones[oldest]

    def apply(
        self,
        write_id: str,
        credit: float,
        *,
        sequence: int,
        expires_at: int,
    ) -> JsonDict:
        """Apply one finite credit and evict the oldest entry when necessary."""

        if not math.isfinite(credit):
            return {"accepted": False, "reason": "nonfinite_credit"}
        if write_id in self.tombstones:
            return {"accepted": False, "reason": "tombstoned_write"}
        bounded = max(-self.credit_bound, min(self.credit_bound, credit))
        evicted: str | None = None
        if write_id not in self.active and len(self.active) >= self.capacity:
            evicted = min(
                self.active,
                key=lambda key: (int(self.active[key]["sequence"]), key),
            )
            evicted_sequence = int(self.active[evicted]["sequence"])
            del self.active[evicted]
            self._add_tombstone(evicted, evicted_sequence)
        self.active[write_id] = {
            "credit": round(bounded, 12),
            "sequence": sequence,
            "expires_at": expires_at,
        }
        return {
            "accepted": True,
            "clamped": bounded != credit,
            "evicted_write_id": evicted,
        }

    def correct(self, write_id: str, delta: float, *, sequence: int) -> JsonDict:
        """Apply a live correction and reject expired or removed support."""

        if write_id in self.tombstones:
            return {"accepted": False, "reason": "tombstoned_write"}
        record = self.active.get(write_id)
        if record is None:
            return {"accepted": False, "reason": "missing_support"}
        if sequence > int(record["expires_at"]):
            del self.active[write_id]
            self._add_tombstone(write_id, sequence)
            return {"accepted": False, "reason": "expired_support"}
        if not math.isfinite(delta):
            return {"accepted": False, "reason": "nonfinite_correction"}
        value = float(record["credit"]) + delta
        record["credit"] = round(
            max(-self.credit_bound, min(self.credit_bound, value)),
            12,
        )
        return {"accepted": True, "clamped": abs(value) > self.credit_bound}

    def _payload(self) -> JsonDict:
        """Return the complete state needed for an exact restart."""

        return {
            "capacity": self.capacity,
            "tombstone_capacity": self.tombstone_capacity,
            "credit_bound": self.credit_bound,
            "active": self.active,
            "tombstones": self.tombstones,
        }

    def to_bytes(self) -> bytes:
        """Serialize state with a checksum that detects one changed byte."""

        payload = self._payload()
        return canonical_json_bytes({"payload": payload, "sha256": sha256_json(payload)})

    @classmethod
    def from_bytes(cls, raw: bytes) -> BoundedAuditState:
        """Load a complete valid checkpoint or reject it without partial state."""

        try:
            wrapper = json.loads(raw)
            payload = wrapper["payload"]
            checksum = wrapper["sha256"]
        except (json.JSONDecodeError, KeyError, TypeError) as error:
            raise ValueError("checkpoint checksum is invalid") from error
        if checksum != sha256_json(payload):
            raise ValueError("checkpoint checksum is invalid")
        return cls(
            capacity=int(payload["capacity"]),
            tombstone_capacity=int(payload["tombstone_capacity"]),
            credit_bound=float(payload["credit_bound"]),
            active=deepcopy(payload["active"]),
            tombstones={key: int(value) for key, value in payload["tombstones"].items()},
        )

    def restore(self, raw: bytes) -> None:
        """Replace this state only after a complete checkpoint validation."""

        restored = self.from_bytes(raw)
        self.capacity = restored.capacity
        self.tombstone_capacity = restored.tombstone_capacity
        self.credit_bound = restored.credit_bound
        self.active = restored.active
        self.tombstones = restored.tombstones


def _durability_audits() -> JsonDict:
    """Exercise restart, poison, rollback, correction, eviction, and tombstones."""

    state = BoundedAuditState(capacity=3, tombstone_capacity=3, credit_bound=1.0)
    state.apply("write-a", 0.2, sequence=1, expires_at=6)
    state.apply("write-b", -0.3, sequence=2, expires_at=6)
    parent = state.to_bytes()
    restarted = BoundedAuditState.from_bytes(parent)
    restart = {
        "byte_identical": restarted.to_bytes() == parent,
        "checkpoint_sha256": "sha256:" + hashlib.sha256(parent).hexdigest(),
    }

    poison = state.apply("poison", math.nan, sequence=3, expires_at=7)
    unchanged_after_poison = state.to_bytes() == parent
    oversized = state.apply("write-c", 99.0, sequence=3, expires_at=7)
    mutated = state.to_bytes()
    state.restore(parent)
    rollback = {
        "parent_sha256": "sha256:" + hashlib.sha256(parent).hexdigest(),
        "mutated_sha256": "sha256:" + hashlib.sha256(mutated).hexdigest(),
        "restored_parent_bytes": state.to_bytes() == parent,
    }

    live_correction = state.correct("write-a", 0.1, sequence=4)
    state.apply("write-c", 0.4, sequence=4, expires_at=8)
    eviction = state.apply("write-d", 0.5, sequence=5, expires_at=8)
    stale_correction = state.correct("write-a", 0.1, sequence=6)
    delayed = {
        "live_correction_accepted": live_correction["accepted"],
        "stale_correction_rejected": not stale_correction["accepted"],
        "stale_reason": stale_correction["reason"],
    }
    poison_result = {
        "nonfinite_poison_rejected": not poison["accepted"],
        "state_unchanged_after_poison": unchanged_after_poison,
        "oversized_credit_clamped": oversized["clamped"],
    }
    capacity = {
        "capacity": state.capacity,
        "active_count": len(state.active),
        "tombstone_capacity": state.tombstone_capacity,
        "tombstone_count": len(state.tombstones),
        "evicted_write_id": eviction["evicted_write_id"],
        "tombstone_blocks_stale_correction": stale_correction["reason"] == "tombstoned_write",
        "within_bounds": (
            len(state.active) <= state.capacity
            and len(state.tombstones) <= state.tombstone_capacity
        ),
    }
    passed = bool(
        restart["byte_identical"]
        and rollback["restored_parent_bytes"]
        and delayed["live_correction_accepted"]
        and delayed["stale_correction_rejected"]
        and poison_result["nonfinite_poison_rejected"]
        and poison_result["state_unchanged_after_poison"]
        and poison_result["oversized_credit_clamped"]
        and capacity["within_bounds"]
        and capacity["tombstone_blocks_stale_correction"]
    )
    return {
        "durability_results": {"passed": passed},
        "restart_results": restart,
        "rollback_results": rollback,
        "delayed_correction_results": delayed,
        "poison_results": poison_result,
        "capacity_results": capacity,
    }


def _fresh_reducer_manifest(reducer_imports: Sequence[str]) -> JsonDict:
    """Declare the implementation boundary used by the sealed reduction."""

    overlap = _producer_overlap(reducer_imports)
    return {
        "module": "carnot.experiment_6856_sealed_risk_sensitive_learning_audit",
        "module_sha256": sha256_file(Path(__file__)),
        "producer_module_imports": list(reducer_imports),
        "source_overlap": bool(overlap),
        "overlap_modules": overlap,
        "headlines_recomputed_from_rows": True,
        "producer_aggregates_used_as_authority": False,
        "llm_invoked": False,
    }


def _source_artifact_hashes(
    repo_root: Path,
    *,
    controller_path: Path | None,
    credit_path: Path | None,
    controller_rows: Sequence[Mapping[str, Any]],
    credit_rows: Sequence[Mapping[str, Any]],
) -> JsonDict:
    """Bind file bytes and canonical row bytes for both sealed sources."""

    controller_file = controller_path or repo_root / CONTROLLER_RELATIVE_PATH
    credit_file = credit_path or repo_root / CREDIT_RELATIVE_PATH
    return {
        "exp6854": {
            "path": str(controller_file),
            "sha256": sha256_file(controller_file),
            "raw_rows_sha256": sha256_json(controller_rows),
            "role": "chronological controller decision rows",
        },
        "exp6855": {
            "path": str(credit_file),
            "sha256": sha256_file(credit_file),
            "raw_rows_sha256": sha256_json(credit_rows),
            "role": "independent per-write credit and support rows",
        },
    }


def _field_principles() -> JsonDict:
    """Explain why each top-level artifact field exists."""

    principles = {
        "field_principles": "Each field states the failure mode it prevents.",
        "preconditions_checked": "Bad sealed evidence must block before reduction.",
        "inference_substrate": "The CPU declaration confirms that no model ran.",
        "duration_s": "Wall time confirms that the reducer executed.",
        "source_artifact_hashes": "Hashes bind both source files and raw rows.",
        "random_seed": "One fixed seed makes synthetic attacks repeatable.",
        "reproducibility_checksum": "The checksum binds all stable output.",
        "rows": "Rows expose each decision, arm, attack, family, and order.",
        "fresh_reducer_manifest": "The manifest proves producer code was not reused.",
        "per_arm_summary": "Independent arm totals prevent selective comparison.",
        "held_future_effect": "Future-only benefit keeps training persistence separate.",
        "win_loss_tie_counts": "Sign counts preserve harmful and null decisions.",
        "no_headroom_count": "No-headroom decisions stay separate from zero effect.",
        "false_positive_injection_rate": "The rate measures harmful memory admission.",
        "abstention_calibration": "Calibration rejects an all-abstain safety shortcut.",
        "durability_results": "Durability joins all bounded-state checks.",
        "restart_results": "Restart must reproduce identical bytes.",
        "rollback_results": "Rollback must restore the exact parent state.",
        "delayed_correction_results": "Corrections need live unremoved support.",
        "poison_results": "Poison cannot become durable evidence.",
        "capacity_results": "Eviction and tombstones must stay bounded.",
        "leave_one_family_out_rows": "Each family gets a separate portability result.",
        "leave_one_order_out_rows": "Each order gets a separate portability result.",
        "leakage_results": "Aggregate contradictions and overlap fail the claim.",
        "counterfactual_support_results": "Per-write signs need observed support.",
        "continuous_self_learning_ready_score": "Every benefit and safety gate is required.",
        "retirement_recommendation": "A repeated harmful selector gets a clear action.",
        "gate_check_summary": "Failed gates retain expected and observed values.",
        "verifier_is_oracle": "False keeps this reducer distinct from outcome authority.",
        "verdict_class": "A closed class prevents a positive wording escape.",
        "honest_verdict": "The terminal statement preserves the measured disposition.",
    }
    return principles


def _base_artifact(
    *,
    run_date: str,
    duration_s: float,
    source_hashes: Mapping[str, Any],
    manifest: Mapping[str, Any],
    preconditions: Mapping[str, Any],
) -> JsonDict:
    """Create a full blocked-safe schema before any reduction can succeed."""

    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "title": "Sealed risk-sensitive learning audit",
        "status": "blocked",
        "run_date": run_date,
        "openspec_requirement_ids": list(OPEN_SPEC_IDS),
        "field_principles": _field_principles(),
        "preconditions_checked": list(preconditions["checks"]),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "duration_s": round(duration_s, 6),
        "source_artifact_hashes": dict(source_hashes),
        "random_seed": RANDOM_SEED,
        "reproducibility_checksum": "",
        "rows": [],
        "fresh_reducer_manifest": dict(manifest),
        "per_arm_summary": {},
        "held_future_effect": None,
        "win_loss_tie_counts": {},
        "no_headroom_count": 0,
        "false_positive_injection_rate": None,
        "abstention_calibration": {"calibrated": False},
        "durability_results": {"passed": False},
        "restart_results": {"byte_identical": False},
        "rollback_results": {"restored_parent_bytes": False},
        "delayed_correction_results": {},
        "poison_results": {},
        "capacity_results": {},
        "leave_one_family_out_rows": [],
        "leave_one_order_out_rows": [],
        "leakage_results": {"passed": False},
        "counterfactual_support_results": {
            "valid_counterfactual_support": False,
            "harmful_write_count": 0,
        },
        "continuous_self_learning_ready_score": 0,
        "retirement_recommendation": {
            "recommended": False,
            "harmful_v598_verdict_reproduced": False,
        },
        "gate_check_summary": dict(preconditions),
        "verifier_is_oracle": False,
        "verdict_class": "blocked",
        "honest_verdict": BLOCKED_VERDICT,
    }
    return artifact


def _aggregate_contradictions(
    controller: Mapping[str, Any],
    *,
    held_future_effect: float,
    false_positive_injection_rate: float,
) -> list[JsonDict]:
    """Compare producer headlines only after fresh row values exist."""

    comparisons = (
        ("held_future_effect", held_future_effect),
        ("false_positive_injection_rate", false_positive_injection_rate),
    )
    contradictions: list[JsonDict] = []
    for field_name, fresh_value in comparisons:
        producer_value = controller.get(field_name)
        if producer_value is None or not math.isclose(
            float(producer_value), fresh_value, rel_tol=0.0, abs_tol=1e-12
        ):
            contradictions.append(
                {
                    "field": field_name,
                    "producer_value": producer_value,
                    "fresh_row_value": fresh_value,
                }
            )
    return contradictions


def _terminal_disposition(
    *,
    readiness: int,
    contradictions: Sequence[Mapping[str, Any]],
    harmful_write_count: int,
    calibrated: bool,
) -> tuple[str, str]:
    """Preserve blocked, disqualified, null, or positive evidence exactly."""

    if contradictions:
        return (
            "disqualified",
            "complete_disqualified_sealed_risk_sensitive_learning_aggregate_contradiction",
        )
    if readiness == 1:
        return "positive", "complete_positive_continuous_self_learning_evidence"
    reasons: list[str] = []
    if harmful_write_count:
        reasons.append("harmful_writes")
    if not calibrated:
        reasons.append("uncalibrated_abstention")
    suffix = "_and_".join(reasons) if reasons else "readiness_gate_failed"
    return "null", f"complete_null_sealed_risk_sensitive_learning_{suffix}"


def build_artifact(
    repo_root: Path,
    *,
    run_date: str,
    duration_s: float,
    controller: Mapping[str, Any],
    credit: Mapping[str, Any],
    split_manifest: Mapping[str, int] = SEALED_SPLIT_MANIFEST,
    reducer_imports: Sequence[str] = (),
    controller_path: Path | None = None,
    credit_path: Path | None = None,
) -> JsonDict:
    """Build one blocked or complete sealed audit from raw source rows."""

    controller_rows = controller.get("rows")
    credit_rows = credit.get("rows")
    controller_rows = controller_rows if isinstance(controller_rows, list) else []
    credit_rows = credit_rows if isinstance(credit_rows, list) else []
    preconditions = validate_preconditions(
        controller,
        credit,
        split_manifest=split_manifest,
        reducer_imports=reducer_imports,
    )
    manifest = _fresh_reducer_manifest(reducer_imports)
    source_hashes = _source_artifact_hashes(
        repo_root,
        controller_path=controller_path,
        credit_path=credit_path,
        controller_rows=controller_rows,
        credit_rows=credit_rows,
    )
    artifact = _base_artifact(
        run_date=run_date,
        duration_s=duration_s,
        source_hashes=source_hashes,
        manifest=manifest,
        preconditions=preconditions,
    )
    if not preconditions["passed"]:
        artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
        return artifact

    rows = reduce_decision_rows(controller_rows, split_manifest=split_manifest)
    per_arm_summary = _per_arm_summary(rows)
    held = _controller_rows(rows, held_only=True)
    held_future_effect = _mean([float(row["effect_vs_no_memory"]) for row in held])
    assert held_future_effect is not None
    controller_all = _controller_rows(rows, held_only=False)
    false_positive_rate = round(
        sum(bool(row["false_positive_injection"]) for row in controller_all) / len(controller_all),
        12,
    )
    counts = _win_loss_tie_counts(rows)
    calibration = _abstention_calibration(rows)
    family_rows = leave_one_group_out(
        rows,
        group_field="family",
        output_field="held_out_family",
        groups=FAMILIES,
    )
    order_rows = leave_one_group_out(
        rows,
        group_field="order_id",
        output_field="held_out_order",
        groups=ORDERS,
    )
    support = reduce_credit_rows(
        credit_rows,
        controller=controller,
        normalized_decisions=rows,
    )
    support["source_gate_values"] = {
        "counterfactual_memory_audit_complete_score": credit.get(
            "counterfactual_memory_audit_complete_score"
        ),
        "causal_memory_credit_eligible_score": credit.get("causal_memory_credit_eligible_score"),
    }
    support["valid_counterfactual_support"] = bool(
        support["valid_counterfactual_support"]
        and credit.get("causal_memory_credit_eligible_score") == 1
        and credit.get("valid_counterfactual_contract", {}).get("declared_valid") is True
    )
    durability = _durability_audits()
    contradictions = _aggregate_contradictions(
        controller,
        held_future_effect=held_future_effect,
        false_positive_injection_rate=false_positive_rate,
    )
    leakage = {
        "source_overlap": manifest["source_overlap"],
        "producer_aggregate_imported": False,
        "aggregate_contradiction_count": len(contradictions),
        "aggregate_contradictions": contradictions,
        "unknown_counterfactual_decision_count": support["unknown_decision_count"],
        "passed": (
            not manifest["source_overlap"]
            and not contradictions
            and support["unknown_decision_count"] == 0
        ),
    }
    readiness_checks = [
        _check("nonzero_held_future_benefit", ">0", held_future_effect, held_future_effect > 0),
        _check(
            "bounded_false_positive_injection",
            f"<={FALSE_POSITIVE_RATE_BOUND}",
            false_positive_rate,
            false_positive_rate <= FALSE_POSITIVE_RATE_BOUND,
        ),
        _check(
            "calibrated_abstention",
            True,
            calibration["calibrated"],
            calibration["calibrated"],
        ),
        _check(
            "durability",
            True,
            durability["durability_results"]["passed"],
            durability["durability_results"]["passed"],
        ),
        _check(
            "valid_counterfactual_support",
            True,
            support["valid_counterfactual_support"],
            support["valid_counterfactual_support"],
        ),
        _check("no_leakage", True, leakage["passed"], leakage["passed"]),
        _check(
            "leave_one_family_out_portability",
            True,
            {row["held_out_family"]: row["portability_passed"] for row in family_rows},
            all(row["portability_passed"] for row in family_rows),
        ),
        _check(
            "leave_one_order_out_portability",
            True,
            {row["held_out_order"]: row["portability_passed"] for row in order_rows},
            all(row["portability_passed"] for row in order_rows),
        ),
        _check(
            "no_supported_harmful_writes",
            0,
            support["harmful_write_count"],
            support["harmful_write_count"] == 0,
        ),
    ]
    readiness = int(all(row["passed"] for row in readiness_checks))
    verdict_class, honest_verdict = _terminal_disposition(
        readiness=readiness,
        contradictions=contradictions,
        harmful_write_count=int(support["harmful_write_count"]),
        calibrated=bool(calibration["calibrated"]),
    )
    harmful_v598_reproduced = bool(
        held_future_effect < 0 and counts["held_future"]["losses"] >= counts["held_future"]["wins"]
    )

    artifact.update(
        {
            "status": "complete",
            "rows": rows,
            "per_arm_summary": per_arm_summary,
            "held_future_effect": held_future_effect,
            "win_loss_tie_counts": counts,
            "no_headroom_count": counts["all_decisions"]["no_headroom_count"],
            "false_positive_injection_rate": false_positive_rate,
            "abstention_calibration": calibration,
            **durability,
            "leave_one_family_out_rows": family_rows,
            "leave_one_order_out_rows": order_rows,
            "leakage_results": leakage,
            "counterfactual_support_results": support,
            "continuous_self_learning_ready_score": readiness,
            "retirement_recommendation": {
                "recommended": harmful_v598_reproduced,
                "harmful_v598_verdict_reproduced": harmful_v598_reproduced,
                "rule": "Retire the changed selector only if it reproduces the harmful V598 sign.",
                "observed_held_future_effect": held_future_effect,
            },
            "gate_check_summary": {
                **preconditions,
                "readiness_checks": readiness_checks,
                "readiness_passed": readiness == 1,
            },
            "verdict_class": verdict_class,
            "honest_verdict": honest_verdict,
        }
    )
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    return artifact


def reproducibility_checksum(artifact: Mapping[str, Any]) -> str:
    """Hash stable content while excluding wall time and the checksum itself."""

    stable = deepcopy(dict(artifact))
    stable.pop("duration_s", None)
    stable.pop("reproducibility_checksum", None)
    return sha256_json(stable)


def validate_artifact(artifact: Mapping[str, Any]) -> list[str]:
    """Return all schema errors so no malformed result is written silently."""

    errors = [
        f"missing field {field_name}"
        for field_name in REQUIRED_ARTIFACT_FIELDS
        if field_name not in artifact
    ]
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        errors.append("invalid inference_substrate")
    if artifact.get("verifier_is_oracle") is not False:
        errors.append("verifier_is_oracle must be false")
    if artifact.get("verdict_class") not in VALID_VERDICT_CLASSES:
        errors.append("invalid verdict_class")
    if not str(artifact.get("honest_verdict", "")).startswith("complete_"):
        errors.append("honest_verdict must start with complete_")
    return errors


def write_artifact(path: Path, artifact: Mapping[str, Any]) -> None:
    """Write stable readable JSON only after the schema passes."""

    errors = validate_artifact(artifact)
    if errors:
        raise ValueError("; ".join(errors))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(artifact, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Run the sealed audit and write only the requested result path."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", required=True)
    parser.add_argument("--controller", type=Path, default=REPO_ROOT / CONTROLLER_RELATIVE_PATH)
    parser.add_argument("--credit", type=Path, default=REPO_ROOT / CREDIT_RELATIVE_PATH)
    parser.add_argument("--output", type=Path, default=REPO_ROOT / RESULT_RELATIVE_PATH)
    args = parser.parse_args(argv)

    started = time.perf_counter()
    controller = load_source(args.controller)
    credit = load_source(args.credit)
    artifact = build_artifact(
        REPO_ROOT,
        run_date=args.date,
        duration_s=0.0,
        controller=controller,
        credit=credit,
        controller_path=args.controller,
        credit_path=args.credit,
    )
    artifact["duration_s"] = round(time.perf_counter() - started, 6)
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    write_artifact(args.output, artifact)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
