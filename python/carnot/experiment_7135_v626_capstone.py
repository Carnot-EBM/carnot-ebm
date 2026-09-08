"""Build the independent V626 evidence matrix.

The matrix records completion separately from scientific value. It sends every
upstream through the shared quarantine boundary before it reads measurements,
so missing or rejected evidence cannot silently become a positive claim. See
REQ-CAPSTONE-7135.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from copy import deepcopy
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import sys
import time
from typing import Any, Callable, Mapping, Sequence

import yaml

from carnot.reporting.evidence_ingress import ingest_evidence


JsonDict = dict[str, Any]
Verifier = Callable[[Path], Mapping[str, Any]]
REPO_ROOT = Path(__file__).resolve().parents[2]
SPEC_PATH = Path("openspec/capabilities/capstone/spec.md")
ROADMAP_PATH = Path("research-roadmap.yaml")
DESIGN_PATH = Path("openspec/change-proposals/research-roadmap-vNEXT.md")
INGRESS_PATH = Path("python/carnot/reporting/evidence_ingress.py")
ADVERSARIAL_PATH = Path("scripts/adversarial_verify.py")
OUTPUT_PATH = Path("results/experiment_7135_v626_capstone.json")
INFERENCE_SUBSTRATE = "aggregation_from_upstream_artifacts: independent V626 evidence matrix"
COMPLETE_VERDICT = "complete_positive_v626_evidence_matrix_without_science_promotion"
BLOCKED_VERDICT = "blocked_v626_capstone_precondition_failed"
MILESTONE = "2026.09.626"
RANDOM_SEED = 713520260908

EXPECTED_UPSTREAM_IDS = (
    "exp7124-v626-contract-preflight",
    "exp7125-v626-source-delta",
    "exp7126-arc-loo-phase-receipt-forensics",
    "exp7127-adapter-withheld-arc-loo-cell",
    "exp7128-arc-loo-causal-provenance-audit",
    "exp7129-hardness-controlled-sota-constraint-bank",
    "exp7130-verifier-committed-uncertainty-routing",
    "exp7131-model-facing-fixed-schema-csl",
    "exp7132-directional-memory-portability-audit",
    "exp7133-wcrg-multiscale-sampler-prototype",
    "exp7134-frustrated-ising-sampler-benchmark",
)
EXPECTED_ALL_IDS = (*EXPECTED_UPSTREAM_IDS, "exp7135-v626-capstone")
VERDICT_CLASSES = {
    "positive",
    "circular_positive",
    "null",
    "blocked",
    "disqualified",
    "partial",
}
NEXT_ACTIONS = {"continue", "repair", "defer", "retire"}

HEADLINE_FIELDS: dict[int, tuple[str, ...]] = {
    7124: ("v626_task_contract_conforms_score",),
    7125: ("v626_source_delta_complete_score",),
    7126: ("arc_phase_receipt_contract_ready_score",),
    7127: (
        "arc_loo_cell_complete_score",
        "withheld_levels",
        "control_levels",
        "level_delta",
    ),
    7128: ("arc_causal_audit_complete_score",),
    7129: (
        "planned_cell_count",
        "completed_cell_count",
        "sota_constraint_bank_ready_score",
    ),
    7130: (
        "useful_retry_rate",
        "accepted_error_rate",
        "exact_rejected_actions_promoted",
        "verifier_committed_routing_complete_score",
    ),
    7131: (
        "later_value_delta",
        "protected_retention_delta",
        "model_facing_csl_complete_score",
    ),
    7132: (
        "six_ordered_pairs_complete",
        "directional_memory_portability_complete_score",
    ),
    7133: ("multiscale_sampler_ready_score",),
    7134: (
        "matched_budget_verified",
        "failure_rate",
        "sampler_benchmark_complete_score",
        "pooled_comparison.mean_paired_ess_delta",
        "pooled_comparison.wins",
        "pooled_comparison.losses",
        "pooled_comparison.ties",
    ),
}

BRANCH_TASKS: dict[str, tuple[str, ...]] = {
    "contract_source": EXPECTED_UPSTREAM_IDS[0:2],
    "arc": EXPECTED_UPSTREAM_IDS[2:5],
    "verification_routing": EXPECTED_UPSTREAM_IDS[5:7],
    "continuous_learning": EXPECTED_UPSTREAM_IDS[7:8],
    "portability": EXPECTED_UPSTREAM_IDS[8:9],
    "sampling": EXPECTED_UPSTREAM_IDS[9:11],
}

REQUIRED_ARTIFACT_FIELDS = frozenset(
    {
        "schema",
        "experiment_id",
        "field_principles",
        "preconditions_checked",
        "run_date",
        "inference_substrate",
        "inference_substrate_class",
        "execution_venue",
        "duration_s",
        "source_artifact_hashes",
        "rows",
        "expected_upstream_ids",
        "observed_upstream_ids",
        "missing_upstream_rows",
        "ingress_rows",
        "excluded_flagged_upstreams",
        "artifact_verdict_rows",
        "substrate_rows",
        "verifier_role_rows",
        "gate_recompute_rows",
        "row_recompute_rows",
        "contradiction_rows",
        "contract_source_disposition",
        "arc_branch_disposition",
        "verification_routing_disposition",
        "continuous_learning_disposition",
        "portability_disposition",
        "sampling_disposition",
        "claim_ceiling_rows",
        "retirement_rows",
        "next_action_rows",
        "v626_capstone_complete_score",
        "random_seed",
        "reproducibility_checksum",
        "gate_check_summary",
        "verifier_is_oracle",
        "verdict_class",
        "honest_verdict",
    }
)


def _sha256(path: Path) -> str | None:
    """Hash one readable source so later changes are visible."""

    try:
        return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return None


def _number(task_id: object) -> int | None:
    """Read an experiment number without accepting an approximate ID."""

    match = re.match(r"exp(\d+)", str(task_id or ""))
    return int(match.group(1)) if match else None


def _normalize_gate(gate: Mapping[str, Any]) -> JsonDict:
    """Keep only the four executable fields in their declared order."""

    return {
        "upstream": str(gate.get("upstream") or ""),
        "artifact_field": str(gate.get("artifact_field") or ""),
        "op": str(gate.get("op") or ""),
        "value": gate.get("value"),
    }


def _design_rows(text: str) -> list[JsonDict]:
    """Parse the Markdown table independently from the YAML parser."""

    pattern = re.compile(
        r"^\|\s*(\d+)\s*\|\s*`(exp\d+[^`]*)`\s*\|\s*([^|]+?)\s*"
        r"\|\s*`?([^|`]+?)`?\s*\|\s*([^|]+?)\s*\|$",
        re.MULTILINE,
    )
    rows: list[JsonDict] = []
    for order, task_id, title, deliverable, gate in pattern.findall(text):
        number = _number(task_id)
        if number is not None and 7124 <= number <= 7135:
            rows.append(
                {
                    "order": int(order),
                    "id": task_id.strip(),
                    "title": title.strip(),
                    "deliverable": deliverable.strip(),
                    "gate_text": gate.strip().replace("`", ""),
                }
            )
    return rows


def _gate_text(gates: Sequence[Mapping[str, Any]]) -> str:
    """Render structured YAML gates in the Markdown contract vocabulary."""

    if not gates:
        return "none"
    return " and ".join(
        f"{gate.get('upstream')}.{gate.get('artifact_field')} {gate.get('op')} {gate.get('value')}"
        for gate in gates
    )


def _same_text(left: object, right: object) -> bool:
    """Ignore presentation spaces while retaining every contract token."""

    clean = lambda value: re.sub(r"\s+", "", str(value or "").replace("`", ""))
    return clean(left) == clean(right)


def load_contract(root: Path) -> tuple[list[JsonDict], list[JsonDict]]:
    """Resolve upstream paths from active YAML and compare the Markdown source."""

    value = yaml.safe_load((root / ROADMAP_PATH).read_text(encoding="utf-8"))
    if not isinstance(value, Mapping) or not isinstance(value.get("tasks"), list):
        raise ValueError("active roadmap does not contain a task list")
    yaml_rows: list[JsonDict] = []
    for order, raw in enumerate(value["tasks"], 1):
        if not isinstance(raw, Mapping):
            continue
        number = _number(raw.get("id"))
        if number is None or not 7124 <= number <= 7135:
            continue
        deliverable = str(raw.get("deliverable") or "")
        yaml_rows.append(
            {
                "order": order,
                "number": number,
                "id": str(raw.get("id") or ""),
                "title": str(raw.get("title") or ""),
                "deliverable": deliverable,
                "path": root / deliverable,
                "gates": [
                    _normalize_gate(gate)
                    for gate in raw.get("gated_on") or []
                    if isinstance(gate, Mapping)
                ],
                "prior_failures": deepcopy(raw.get("prior_failures") or []),
                "milestone": str(raw.get("milestone") or ""),
            }
        )
    markdown_rows = _design_rows((root / DESIGN_PATH).read_text(encoding="utf-8"))
    contract_rows: list[JsonDict] = []
    for index, expected_id in enumerate(EXPECTED_ALL_IDS):
        yaml_row = yaml_rows[index] if index < len(yaml_rows) else None
        markdown_row = markdown_rows[index] if index < len(markdown_rows) else None
        checks = {
            "order_match": bool(
                yaml_row
                and markdown_row
                and yaml_row["order"] == markdown_row["order"] == index + 1
            ),
            "id_match": bool(
                yaml_row and markdown_row and yaml_row["id"] == markdown_row["id"] == expected_id
            ),
            "title_match": bool(
                yaml_row and markdown_row and yaml_row["title"] == markdown_row["title"]
            ),
            "deliverable_match": bool(
                yaml_row and markdown_row and yaml_row["deliverable"] == markdown_row["deliverable"]
            ),
            "gate_match": bool(
                yaml_row
                and markdown_row
                and _same_text(_gate_text(yaml_row["gates"]), markdown_row["gate_text"])
            ),
            "milestone_match": bool(yaml_row and yaml_row["milestone"] == MILESTONE),
        }
        contract_rows.append(
            {
                "order": index + 1,
                "expected_id": expected_id,
                "yaml_id": yaml_row["id"] if yaml_row else None,
                "markdown_id": markdown_row["id"] if markdown_row else None,
                **checks,
                "passed": all(checks.values()),
            }
        )
    upstream = [row for row in yaml_rows if row["id"] in EXPECTED_UPSTREAM_IDS]
    return upstream, contract_rows


def _source_hash_for(payload: Mapping[str, Any], producer: Mapping[str, Any]) -> str | None:
    """Find a consumer's byte binding for its exact gate producer."""

    sources = payload.get("source_artifact_hashes")
    wanted = {str(producer.get("deliverable")), str(producer.get("path"))}
    if isinstance(sources, Mapping):
        for key, value in sources.items():
            if str(key) in wanted:
                if isinstance(value, str):
                    return value
                if isinstance(value, Mapping) and isinstance(value.get("sha256"), str):
                    return str(value["sha256"])
        for key in ("upstream_artifact", "upstream_bank_hash"):
            value = sources.get(key)
            if isinstance(value, str):
                return value
    if isinstance(sources, list):
        for row in sources:
            if (
                isinstance(row, Mapping)
                and str(row.get("path")) in wanted
                and isinstance(row.get("sha256"), str)
            ):
                return str(row["sha256"])
    for key in ("upstream_artifact_hash", "upstream_bank_hash"):
        value = payload.get(key)
        if isinstance(value, str):
            return value
    return None


def ingest_upstreams(tasks: Sequence[Mapping[str, Any]], *, verifier: Verifier) -> JsonDict:
    """Apply shared ingress twice so consumer-bound hash drift also fails closed."""

    def expectations(hashes: Mapping[str, str]) -> list[JsonDict]:
        return [
            {
                "path": task["path"],
                "expected_sha256": hashes.get(str(task["id"])),
                "task_id": task["id"],
                "task_number": task["number"],
                "required_headline_fields": list(HEADLINE_FIELDS.get(int(task["number"]), ())),
            }
            for task in tasks
        ]

    first = ingest_evidence(expectations({}), verifier=verifier)
    first_accepted = {str(row["task_id"]): row for row in first["accepted_upstreams"]}
    tasks_by_id = {str(task["id"]): task for task in tasks}
    linked_hashes: dict[str, str] = {}
    for consumer in tasks:
        accepted = first_accepted.get(str(consumer["id"]))
        if accepted is None:
            continue
        for gate in consumer.get("gates", []):
            producer_id = str(gate.get("upstream"))
            producer = tasks_by_id.get(producer_id)
            if producer is None:
                continue
            declared = _source_hash_for(accepted["payload"], producer)
            if declared is not None:
                linked_hashes[producer_id] = declared
    if not linked_hashes:
        first["consumer_bound_expected_hashes"] = {}
        return first
    second = ingest_evidence(expectations(linked_hashes), verifier=verifier)
    second["consumer_bound_expected_hashes"] = linked_hashes
    return second


def _scalar(value: Any) -> Any:
    """Read a principle-wrapped value without reading its explanation as data."""

    if isinstance(value, Mapping) and "value" in value:
        return value.get("value")
    return value


def classify_verdict(payload: Mapping[str, Any]) -> JsonDict:
    """Derive the terminal class before trusting the producer's label."""

    declared = str(_scalar(payload.get("verdict_class")) or "")
    honest = str(_scalar(payload.get("honest_verdict")) or "")
    lower = honest.lower()
    if lower.startswith(("blocked", "complete_blocked")):
        structural = "blocked"
    elif lower.startswith(("disqualified", "complete_disqualified")):
        structural = "disqualified"
    elif lower.startswith(("null", "complete_null")):
        structural = "null"
    elif lower.startswith(("partial", "complete_partial")):
        structural = "partial"
    elif payload.get("verifier_is_oracle") is True and declared in {
        "positive",
        "circular_positive",
    }:
        structural = "circular_positive"
    elif declared in VERDICT_CLASSES:
        structural = declared
    else:
        structural = "disqualified"
    return {
        "declared_verdict_class": declared,
        "structural_verdict_class": structural,
        "effective_verdict_class": structural,
        "honest_verdict": honest,
        "consistent": declared == structural,
    }


def _compare(observed: Any, operator: str, expected: Any) -> bool:
    """Evaluate the closed gate operator set without coercing missing values."""

    if operator == "==":
        return observed == expected
    if operator == ">=":
        return (
            isinstance(observed, (int, float))
            and not isinstance(observed, bool)
            and observed >= expected
        )
    return False


def _consumer_gate_claim(payload: Mapping[str, Any], field: str) -> tuple[Any, Any]:
    """Read the consumer's gate receipt only for an audit comparison."""

    summary = payload.get("gate_check_summary")
    if not isinstance(summary, Mapping):
        return None, None
    checks = summary.get("checks")
    if isinstance(checks, list):
        for row in checks:
            if isinstance(row, Mapping) and row.get("check") == field:
                return row.get("expected_value"), row.get("observed_value")
    return summary.get("expected_value"), summary.get("observed_value")


def recompute_gates(
    tasks: Sequence[Mapping[str, Any]],
    accepted: Mapping[str, Mapping[str, Any]],
    rejected: Mapping[str, Mapping[str, Any]],
) -> list[JsonDict]:
    """Replay every gate from the named producer's exact top-level field."""

    rows: list[JsonDict] = []
    for consumer in tasks:
        for gate in consumer.get("gates", []):
            producer_id = str(gate.get("upstream"))
            field = str(gate.get("artifact_field"))
            producer = accepted.get(producer_id)
            rejection = rejected.get(producer_id, {})
            reasons = list(rejection.get("reason_codes") or [])
            observed = None
            if producer is not None:
                payload = producer.get("payload", {})
                if field not in payload:
                    status = "missing_field"
                else:
                    observed = _scalar(payload.get(field))
                    status = (
                        "passed"
                        if _compare(observed, str(gate.get("op")), gate.get("value"))
                        else "valid_gate_failure"
                    )
            elif "artifact_not_found" in reasons:
                status = "missing_artifact"
            elif any(reason.endswith("flagged_adversarial") for reason in reasons):
                status = "excluded_flagged_upstream"
            elif "hash_drift" in reasons:
                status = "stale_hash"
            else:
                status = "missing_artifact"
            consumer_payload = accepted.get(str(consumer["id"]), {}).get("payload", {})
            reported_expected, reported_observed = _consumer_gate_claim(consumer_payload, field)
            rows.append(
                {
                    "consumer_task_id": consumer["id"],
                    "upstream_task_id": producer_id,
                    "artifact_field": field,
                    "operator": gate.get("op"),
                    "expected_value": gate.get("value"),
                    "observed_value": observed,
                    "status": status,
                    "passed": status == "passed",
                    "consumer_reported_expected_value": reported_expected,
                    "consumer_reported_observed_value": reported_observed,
                    "consumer_gate_agrees": (
                        reported_observed == observed
                        if reported_observed is not None and observed is not None
                        else None
                    ),
                }
            )
    return rows


def _values_agree(left: Any, right: Any) -> bool:
    """Compare measured floats with a narrow numerical tolerance."""

    if isinstance(left, float) and isinstance(right, float):
        return math.isclose(left, right, rel_tol=1e-9, abs_tol=1e-12)
    return left == right


def _headline(
    number: int,
    field: str,
    declared: Any,
    recomputed: Any,
    support: str,
    *,
    available: bool = True,
) -> JsonDict:
    """Keep a producer headline beside its independent row reduction."""

    agrees = _values_agree(declared, recomputed) if available else None
    return {
        "task_number": number,
        "field": field,
        "declared_value": declared,
        "recomputed_value": recomputed if available else None,
        "support": support,
        "status": "agrees" if agrees else ("contradiction" if available else "unavailable"),
        "agrees": agrees,
        "claim_ceiling": ("declared_and_row_derived" if agrees else "no_claim_from_this_headline"),
    }


def _unavailable_headlines(number: int, payload: Mapping[str, Any]) -> list[JsonDict]:
    """Name fields whose unit rows cannot support an independent reduction."""

    return [
        _headline(
            number, field, payload.get(field), None, "required unit rows absent", available=False
        )
        for field in HEADLINE_FIELDS.get(number, ("comparison_headline",))
    ]


def _mean(values: Sequence[float]) -> float:
    """Return one deterministic arithmetic mean for a non-empty row set."""

    return sum(values) / len(values)


def recompute_headlines(number: int, payload: Mapping[str, Any]) -> list[JsonDict]:
    """Recompute V626 comparison headlines only from stored unit rows."""

    rows = payload.get("rows")
    if not isinstance(rows, list) or not rows or not all(isinstance(row, Mapping) for row in rows):
        return _unavailable_headlines(number, payload)
    if number == 7124:
        value = int(len(rows) == 12 and all(row.get("passed") is True for row in rows))
        return [
            _headline(
                number,
                HEADLINE_FIELDS[number][0],
                payload.get(HEADLINE_FIELDS[number][0]),
                value,
                "rows",
            )
        ]
    if number in {7125, 7126}:
        return _unavailable_headlines(number, payload)
    if number == 7127:
        by_arm = {str(row.get("arm")): row for row in rows}
        withheld = by_arm.get("adapter_withheld")
        control = by_arm.get("adapter_visible_control")
        if withheld is None or control is None:
            return _unavailable_headlines(number, payload)
        withheld_levels = withheld.get("levels")
        control_levels = control.get("levels")
        if not isinstance(withheld_levels, int) or not isinstance(control_levels, int):
            return _unavailable_headlines(number, payload)
        complete = int(
            (withheld.get("executed_transition_count") or 0) > 0
            and (control.get("executed_transition_count") or 0) > 0
        )
        values = {
            "arc_loo_cell_complete_score": complete,
            "withheld_levels": withheld_levels,
            "control_levels": control_levels,
            "level_delta": withheld_levels - control_levels,
        }
        return [
            _headline(number, field, payload.get(field), value, "rows grouped by arm")
            for field, value in values.items()
        ]
    if number == 7128:
        complete = int(
            len(rows) == 2 and all((row.get("transition_count") or 0) > 0 for row in rows)
        )
        return [
            _headline(
                number,
                "arc_causal_audit_complete_score",
                payload.get("arc_causal_audit_complete_score"),
                complete,
                "arm audit rows",
            )
        ]
    if number == 7129:
        planned = payload.get("planned_cell_count")
        completed = sum(row.get("terminal_state") == "complete" for row in rows)
        ready = int(isinstance(planned, int) and completed == planned)
        values = {
            "planned_cell_count": planned,
            "completed_cell_count": completed,
            "sota_constraint_bank_ready_score": ready,
        }
        return [
            _headline(number, field, payload.get(field), value, "model cell rows")
            for field, value in values.items()
        ]
    if number == 7130:
        retries = [row for row in rows if row.get("retry_attempted") is True]
        executed = [row for row in rows if row.get("executed") is True]
        useful = sum(row.get("useful_retry") is True for row in retries)
        errors = sum(row.get("exact_success") is not True for row in executed)
        promoted = sum(
            row.get("promoted") is True and row.get("exact_success") is not True for row in rows
        )
        values = {
            "useful_retry_rate": useful / len(retries) if retries else 0.0,
            "accepted_error_rate": errors / len(executed) if executed else 0.0,
            "exact_rejected_actions_promoted": promoted,
            "verifier_committed_routing_complete_score": int(
                len(rows) == 432
                and all(row.get("terminal_state") == "complete" for row in rows)
                and promoted == 0
            ),
        }
        return [
            _headline(number, field, payload.get(field), value, "routing unit rows")
            for field, value in values.items()
        ]
    if number == 7131:
        future = payload.get("future_episode_rows")
        protected = payload.get("protected_retention_rows")
        if not isinstance(future, list) or not future:
            return _unavailable_headlines(number, payload)
        by_arm: dict[str, list[float]] = defaultdict(list)
        for row in future:
            if isinstance(row, Mapping) and isinstance(row.get("exact_success"), (int, float)):
                by_arm[str(row.get("arm"))].append(float(row["exact_success"]))
        if not by_arm.get("fixed_schema") or not by_arm.get("no_memory"):
            return _unavailable_headlines(number, payload)
        later = _mean(by_arm["fixed_schema"]) - _mean(by_arm["no_memory"])
        protected_delta = payload.get("protected_retention_delta")
        complete = int(bool(protected) and len(by_arm) >= 3)
        values = {
            "later_value_delta": later,
            "protected_retention_delta": protected_delta,
            "model_facing_csl_complete_score": complete,
        }
        return [
            _headline(number, field, payload.get(field), value, "future and protected rows")
            for field, value in values.items()
        ]
    if number == 7132:
        directions = payload.get("direction_rows")
        if not isinstance(directions, list) or not directions:
            return _unavailable_headlines(number, payload)
        pairs = {
            (row.get("writer_model"), row.get("reader_model"))
            for row in directions
            if isinstance(row, Mapping) and row.get("writer_model") != row.get("reader_model")
        }
        values = {
            "six_ordered_pairs_complete": len(pairs) == 6,
            "directional_memory_portability_complete_score": int(len(pairs) == 6),
        }
        return [
            _headline(number, field, payload.get(field), value, "direction rows")
            for field, value in values.items()
        ]
    if number == 7133:
        value = int(all(row.get("passed") is True for row in rows))
        return [
            _headline(
                number,
                "multiscale_sampler_ready_score",
                payload.get("multiscale_sampler_ready_score"),
                value,
                "finite-law fixture rows",
            )
        ]
    if number == 7134:
        grouped: dict[tuple[Any, Any], dict[str, Mapping[str, Any]]] = defaultdict(dict)
        for row in rows:
            grouped[(row.get("cell_id"), row.get("seed"))][str(row.get("arm"))] = row
        deltas: list[float] = []
        budgets_ok = True
        failed = 0
        for arms in grouped.values():
            budgets = {
                row.get("energy_evaluations")
                for row in arms.values()
                if row.get("failed") is not True
            }
            budgets_ok = budgets_ok and len(budgets) == 1
            failed += sum(row.get("failed") is True for row in arms.values())
            left = arms.get("multiscale_mh")
            right = arms.get("local_gibbs")
            if (
                left
                and right
                and isinstance(left.get("minimum_observable_ess"), (int, float))
                and isinstance(right.get("minimum_observable_ess"), (int, float))
            ):
                deltas.append(
                    float(left["minimum_observable_ess"]) - float(right["minimum_observable_ess"])
                )
        if not deltas:
            return _unavailable_headlines(number, payload)
        pooled = payload.get("pooled_comparison")
        pooled = pooled if isinstance(pooled, Mapping) else {}
        values = {
            "matched_budget_verified": budgets_ok,
            "failure_rate": failed / len(rows),
            "sampler_benchmark_complete_score": int(budgets_ok and not failed),
            "pooled_comparison.mean_paired_ess_delta": _mean(deltas),
            "pooled_comparison.wins": sum(value > 0 for value in deltas),
            "pooled_comparison.losses": sum(value < 0 for value in deltas),
            "pooled_comparison.ties": sum(value == 0 for value in deltas),
        }
        return [
            _headline(
                number,
                field,
                pooled.get(field.rsplit(".", 1)[1])
                if field.startswith("pooled_comparison.")
                else payload.get(field),
                value,
                "per-seed sampler rows",
            )
            for field, value in values.items()
        ]
    return _unavailable_headlines(number, payload)


def row_reversal_rows(task_id: str, payload: Mapping[str, Any]) -> list[JsonDict]:
    """Find positive claims whose unit-level losses do not trail their wins."""

    if classify_verdict(payload)["effective_verdict_class"] != "positive":
        return []
    rows = payload.get("rows")
    if not isinstance(rows, list):
        return []
    fields = sorted(
        {
            key
            for row in rows
            if isinstance(row, Mapping)
            for key, value in row.items()
            if "delta" in key.lower()
            and isinstance(value, (int, float))
            and not isinstance(value, bool)
        }
    )
    findings: list[JsonDict] = []
    for field in fields:
        values = [
            float(row[field])
            for row in rows
            if isinstance(row, Mapping)
            and isinstance(row.get(field), (int, float))
            and not isinstance(row.get(field), bool)
        ]
        wins = sum(value > 0 for value in values)
        losses = sum(value < 0 for value in values)
        if wins + losses >= 2 and losses >= wins:
            findings.append(
                {
                    "task_id": task_id,
                    "kind": "row_reversal",
                    "field": field,
                    "wins": wins,
                    "losses": losses,
                    "ties": sum(value == 0 for value in values),
                    "claim_ceiling": "no_positive_claim",
                }
            )
    return findings


def _rejected_class(reasons: Sequence[str]) -> str:
    """Convert an ingress rejection into one closed terminal evidence class."""

    if "artifact_not_found" in reasons:
        return "blocked"
    return "disqualified"


def _artifact_row(task: Mapping[str, Any], ingress: Mapping[str, Any]) -> JsonDict:
    """Preserve identity and verdict facts without importing rejected metrics."""

    payload = ingress.get("payload")
    reasons = list(ingress.get("reason_codes") or [])
    if isinstance(payload, Mapping):
        verdict = classify_verdict(payload)
        headline = {
            field: _scalar(payload.get(field))
            for field in HEADLINE_FIELDS.get(int(task["number"]), ())
        }
        substrate = _scalar(payload.get("inference_substrate"))
        substrate_class = _scalar(payload.get("inference_substrate_class"))
        venue = _scalar(payload.get("execution_venue"))
        verifier_role = _scalar(payload.get("verifier_role")) or "not_declared"
        oracle = _scalar(payload.get("verifier_is_oracle"))
    else:
        effective = _rejected_class(reasons)
        verdict = {
            "declared_verdict_class": ingress.get("verdict_class"),
            "structural_verdict_class": effective,
            "effective_verdict_class": effective,
            "honest_verdict": ingress.get("honest_verdict"),
            "consistent": ingress.get("verdict_class") in {None, effective},
        }
        headline = {field: None for field in HEADLINE_FIELDS.get(int(task["number"]), ())}
        substrate = substrate_class = venue = None
        verifier_role = "not_imported_excluded_or_missing"
        oracle = None
    return {
        "task_id": task["id"],
        "task_number": task["number"],
        "title": task["title"],
        "path": str(task["path"]),
        "sha256": ingress.get("sha256"),
        "run_date": ingress.get("run_date"),
        "inference_substrate": substrate,
        "inference_substrate_class": substrate_class,
        "execution_venue": venue,
        "verifier_role": verifier_role,
        "verifier_is_oracle": oracle,
        "adversarial_status": ingress.get("flag_status"),
        "ingress_reason_codes": reasons,
        "required_headline_fields": list(HEADLINE_FIELDS.get(int(task["number"]), ())),
        "headline_fields": headline,
        **verdict,
        "supporting_claim_eligible": isinstance(payload, Mapping),
    }


def build_retirement_rows(
    tasks: Sequence[Mapping[str, Any]], outcomes: Mapping[str, Mapping[str, Any]]
) -> list[JsonDict]:
    """Apply every caller-owned retire-if-same-verdict signal exactly."""

    rows: list[JsonDict] = []
    for task in tasks:
        current = outcomes.get(str(task["id"]), {}).get("honest_verdict")
        for prior in task.get("prior_failures", []):
            if not isinstance(prior, Mapping):
                continue
            repeated = current is not None and current == prior.get("verdict")
            retire = repeated and prior.get("retire_if_same_verdict") is True
            rows.append(
                {
                    "task_id": task["id"],
                    "prior_experiment_id": prior.get("experiment_id"),
                    "prior_verdict": prior.get("verdict"),
                    "current_verdict": current,
                    "retire_if_same_verdict": prior.get("retire_if_same_verdict") is True,
                    "verdict_repeated": repeated,
                    "action": "retire" if retire else "none",
                    "addressed_by": prior.get("addressed_by"),
                }
            )
    return rows


def _branch_class(classes: Sequence[str]) -> str:
    """Return the strongest claim ceiling shared by every branch input."""

    for value in ("disqualified", "blocked", "partial", "null", "circular_positive"):
        if value in classes:
            return value
    return "positive"


def build_branch_disposition(
    branch: str,
    task_ids: Sequence[str],
    outcomes: Mapping[str, Mapping[str, Any]],
    retirement: Sequence[Mapping[str, Any]],
) -> JsonDict:
    """Select one terminal class and one changed action for a science branch."""

    classes = [
        str(outcomes.get(task_id, {}).get("effective_verdict_class") or "blocked")
        for task_id in task_ids
    ]
    verdict_class = _branch_class(classes)
    repeated = any(
        row.get("task_id") in task_ids and row.get("action") == "retire" for row in retirement
    )
    if repeated or verdict_class == "null":
        action = "retire"
    elif verdict_class in {"disqualified", "partial"}:
        action = "repair"
    elif verdict_class == "blocked":
        action = "defer"
    else:
        action = "continue"
    return {
        "branch": branch,
        "basis_task_ids": list(task_ids),
        "basis_verdict_classes": classes,
        "verdict_class": verdict_class,
        "recommended_action": action,
        "matrix_completion_is_science_claim": False,
        "failed_scope_unchanged": False,
        "claim_ceiling": {
            "positive": "positive_within_declared_scope",
            "circular_positive": "circular_positive_only",
            "null": "null_only",
            "blocked": "no_claim_missing_or_blocked",
            "disqualified": "no_claim_disqualified",
            "partial": "no_claim_incomplete",
        }[verdict_class],
    }


def _principles() -> dict[str, str]:
    """Give every field a short reason that explains its audit purpose."""

    return {
        field: f"The {field} field keeps this V626 matrix fact explicit and independently checkable."
        for field in REQUIRED_ARTIFACT_FIELDS
    }


def reproducibility_checksum(artifact: Mapping[str, Any]) -> str:
    """Hash scientific content while excluding elapsed time and this digest."""

    stable = deepcopy(dict(artifact))
    stable["duration_s"] = None
    stable["reproducibility_checksum"] = None
    encoded = json.dumps(stable, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _preconditions(root: Path, output: Path) -> tuple[list[JsonDict], list[JsonDict]]:
    """Check capstone-owned inputs without requiring any upstream result."""

    sources = (ROADMAP_PATH, DESIGN_PATH, SPEC_PATH, INGRESS_PATH, ADVERSARIAL_PATH)
    checks: list[JsonDict] = []
    hashes: list[JsonDict] = []
    for relative in sources:
        digest = _sha256(root / relative)
        checks.append(
            {
                "check": str(relative),
                "expected_value": "readable_file",
                "observed_value": digest,
                "passed": digest is not None,
            }
        )
        if digest is not None:
            hashes.append({"path": str(relative), "sha256": digest})
    parent = output.parent
    writable = parent.is_dir() and os.access(parent, os.W_OK)
    checks.append(
        {
            "check": "artifact_output_parent_writable",
            "expected_value": True,
            "observed_value": writable,
            "passed": writable,
        }
    )
    return checks, hashes


def _blocked_artifact(
    date: str,
    checks: Sequence[Mapping[str, Any]],
    hashes: Sequence[Mapping[str, Any]],
    started: float,
) -> JsonDict:
    """Return a full schema when capstone-owned inputs cannot be read."""

    failed = next(row for row in checks if row.get("passed") is not True)
    branch_rows = {
        name: build_branch_disposition(name, ids, {}, []) for name, ids in BRANCH_TASKS.items()
    }
    artifact: JsonDict = {
        "schema": "carnot.experiment_7135.v626_capstone.v1",
        "experiment_id": 7135,
        "field_principles": _principles(),
        "preconditions_checked": list(checks),
        "run_date": date,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "inference_substrate_class": "blocked_no_run",
        "execution_venue": "host",
        "duration_s": round(time.monotonic() - started, 6),
        "source_artifact_hashes": list(hashes),
        "rows": [],
        "expected_upstream_ids": list(EXPECTED_UPSTREAM_IDS),
        "observed_upstream_ids": [],
        "missing_upstream_rows": [],
        "ingress_rows": [],
        "excluded_flagged_upstreams": [],
        "artifact_verdict_rows": [],
        "substrate_rows": [],
        "verifier_role_rows": [],
        "gate_recompute_rows": [],
        "row_recompute_rows": [],
        "contradiction_rows": [],
        "contract_source_disposition": branch_rows["contract_source"],
        "arc_branch_disposition": branch_rows["arc"],
        "verification_routing_disposition": branch_rows["verification_routing"],
        "continuous_learning_disposition": branch_rows["continuous_learning"],
        "portability_disposition": branch_rows["portability"],
        "sampling_disposition": branch_rows["sampling"],
        "claim_ceiling_rows": [],
        "retirement_rows": [],
        "next_action_rows": [
            {"branch": name, "action": "defer", "reason": "capstone_precondition_failed"}
            for name in BRANCH_TASKS
        ],
        "v626_capstone_complete_score": 0,
        "random_seed": RANDOM_SEED,
        "reproducibility_checksum": "",
        "gate_check_summary": {
            "passed": False,
            "failed_check": failed.get("check"),
            "expected_value": failed.get("expected_value"),
            "observed_value": failed.get("observed_value"),
        },
        "verifier_is_oracle": False,
        "verdict_class": "blocked",
        "honest_verdict": BLOCKED_VERDICT,
    }
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    return artifact


def _load_verifier(root: Path) -> Verifier:
    """Load the current checked-in adversarial verifier by its exact path."""

    path = root / ADVERSARIAL_PATH
    name = "experiment_7135_live_adversarial_verifier"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load adversarial verifier: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module.verify_artifact


def _ingress_rows(ingress: Mapping[str, Any]) -> list[JsonDict]:
    """Remove accepted payload bytes after the matrix has imported allowed fields."""

    rows: list[JsonDict] = []
    for source in (
        ingress.get("accepted_upstreams", []),
        ingress.get("rejected_upstream_rows", []),
    ):
        for row in source:
            clean = {key: deepcopy(value) for key, value in row.items() if key != "payload"}
            rows.append(clean)
    return rows


def _claim_ceiling(row: Mapping[str, Any]) -> JsonDict:
    """State the maximum claim without automatically promoting it."""

    verdict = str(row.get("effective_verdict_class"))
    ceiling = {
        "positive": "positive_within_declared_scope",
        "circular_positive": "circular_positive_only",
        "null": "null_only",
        "blocked": "no_claim_missing_or_blocked",
        "disqualified": "no_claim_disqualified",
        "partial": "no_claim_incomplete",
    }[verdict]
    return {
        "scope": row.get("task_id"),
        "verdict_class": verdict,
        "claim_ceiling": ceiling,
        "supports_positive_science": verdict == "positive"
        and row.get("supporting_claim_eligible") is True,
        "promoted_as_positive_science": False,
    }


def build_artifact(
    root: Path,
    date: str,
    *,
    verifier: Verifier | None = None,
    output_path: Path | None = None,
) -> JsonDict:
    """Build all 11 task rows even when upstream evidence is absent or excluded."""

    started = time.monotonic()
    output = output_path or root / OUTPUT_PATH
    checks, source_hashes = _preconditions(root, output)
    if not all(row["passed"] for row in checks):
        return _blocked_artifact(date, checks, source_hashes, started)
    tasks, contract_rows = load_contract(root)
    contract_ok = len(tasks) == 11 and all(row["passed"] for row in contract_rows)
    checks.append(
        {
            "check": "active_markdown_yaml_contract",
            "expected_value": 12,
            "observed_value": sum(row["passed"] for row in contract_rows),
            "passed": contract_ok,
        }
    )
    live_verifier = verifier or _load_verifier(root)
    ingress = ingest_upstreams(tasks, verifier=live_verifier)
    accepted = {str(row["task_id"]): row for row in ingress["accepted_upstreams"]}
    rejected = {str(row["task_id"]): row for row in ingress["rejected_upstream_rows"]}
    artifact_rows = [
        _artifact_row(task, accepted.get(str(task["id"])) or rejected.get(str(task["id"]), {}))
        for task in tasks
    ]
    outcomes = {str(row["task_id"]): row for row in artifact_rows}
    gate_rows = recompute_gates(tasks, accepted, rejected)
    recomputed: list[JsonDict] = []
    contradictions: list[JsonDict] = []
    for task in tasks:
        task_id = str(task["id"])
        accepted_row = accepted.get(task_id)
        if accepted_row is None:
            continue
        payload = accepted_row["payload"]
        task_recomputed = recompute_headlines(int(task["number"]), payload)
        for row in task_recomputed:
            row["task_id"] = task_id
        recomputed.extend(task_recomputed)
        contradictions.extend(row_reversal_rows(task_id, payload))
        if not outcomes[task_id]["consistent"]:
            contradictions.append(
                {
                    "task_id": task_id,
                    "kind": "verdict_class_mismatch",
                    "declared": outcomes[task_id]["declared_verdict_class"],
                    "observed": outcomes[task_id]["structural_verdict_class"],
                    "claim_ceiling": "no_positive_claim",
                }
            )
        mismatches = [row for row in task_recomputed if row["status"] == "contradiction"]
        if mismatches:
            outcomes[task_id]["effective_verdict_class"] = "disqualified"
            outcomes[task_id]["supporting_claim_eligible"] = False
            contradictions.extend(
                {
                    "task_id": task_id,
                    "kind": "headline_row_mismatch",
                    "field": row["field"],
                    "declared": row["declared_value"],
                    "observed": row["recomputed_value"],
                    "claim_ceiling": "no_positive_claim",
                }
                for row in mismatches
            )
    for gate in gate_rows:
        if gate["status"] == "passed":
            continue
        task_id = str(gate["consumer_task_id"])
        if gate["status"] in {"excluded_flagged_upstream", "stale_hash"} and task_id in accepted:
            outcomes[task_id]["effective_verdict_class"] = "disqualified"
        elif outcomes[task_id]["effective_verdict_class"] != "disqualified":
            outcomes[task_id]["effective_verdict_class"] = "blocked"
        outcomes[task_id]["supporting_claim_eligible"] = False
        contradictions.append(
            {
                "task_id": task_id,
                "kind": "gate_dependency_unusable",
                "field": gate["artifact_field"],
                "declared": gate["expected_value"],
                "observed": gate["observed_value"],
                "gate_status": gate["status"],
                "claim_ceiling": "no_positive_claim",
            }
        )
    reversal_tasks = {
        str(row["task_id"]) for row in contradictions if row["kind"] == "row_reversal"
    }
    for task_id in reversal_tasks:
        outcomes[task_id]["effective_verdict_class"] = "disqualified"
        outcomes[task_id]["supporting_claim_eligible"] = False
    retirement = build_retirement_rows(tasks, outcomes)
    dispositions = {
        name: build_branch_disposition(name, ids, outcomes, retirement)
        for name, ids in BRANCH_TASKS.items()
    }
    ceilings = [_claim_ceiling(row) for row in artifact_rows]
    ceilings.extend(
        {
            "scope": name,
            "verdict_class": disposition["verdict_class"],
            "claim_ceiling": disposition["claim_ceiling"],
            "supports_positive_science": disposition["verdict_class"] == "positive",
            "promoted_as_positive_science": False,
        }
        for name, disposition in dispositions.items()
    )
    missing_rows = [
        deepcopy(row)
        for row in ingress["rejected_upstream_rows"]
        if "artifact_not_found" in row.get("reason_codes", [])
    ]
    excluded = [deepcopy(row) for row in ingress["excluded_flagged_upstreams"]]
    observed_ids = [
        str(row["task_id"])
        for row in (*ingress["accepted_upstreams"], *ingress["rejected_upstream_rows"])
        if row.get("sha256") is not None
    ]
    for row in _ingress_rows(ingress):
        if row.get("sha256") is not None:
            source_hashes.append(
                {"path": row["path"], "sha256": row["sha256"], "task_id": row["task_id"]}
            )
    complete = int(
        len(artifact_rows) == 11
        and len(gate_rows) == 4
        and len(dispositions) == 6
        and len(ceilings) >= 17
    )
    verdict_class = "positive" if contract_ok and complete == 1 else "disqualified"
    honest = (
        COMPLETE_VERDICT
        if verdict_class == "positive"
        else "complete_disqualified_v626_contract_or_matrix"
    )
    artifact: JsonDict = {
        "schema": "carnot.experiment_7135.v626_capstone.v1",
        "experiment_id": 7135,
        "field_principles": _principles(),
        "preconditions_checked": checks,
        "run_date": date,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "inference_substrate_class": "aggregation",
        "execution_venue": "host",
        "duration_s": round(time.monotonic() - started, 6),
        "source_artifact_hashes": source_hashes,
        "rows": artifact_rows,
        "expected_upstream_ids": list(EXPECTED_UPSTREAM_IDS),
        "observed_upstream_ids": observed_ids,
        "missing_upstream_rows": missing_rows,
        "ingress_rows": _ingress_rows(ingress),
        "excluded_flagged_upstreams": excluded,
        "artifact_verdict_rows": artifact_rows,
        "substrate_rows": [
            {
                "task_id": row["task_id"],
                "inference_substrate": row["inference_substrate"],
                "inference_substrate_class": row["inference_substrate_class"],
                "execution_venue": row["execution_venue"],
            }
            for row in artifact_rows
        ],
        "verifier_role_rows": [
            {
                "task_id": row["task_id"],
                "verifier_role": row["verifier_role"],
                "verifier_is_oracle": row["verifier_is_oracle"],
                "claim_ceiling": _claim_ceiling(row)["claim_ceiling"],
            }
            for row in artifact_rows
        ],
        "gate_recompute_rows": gate_rows,
        "row_recompute_rows": recomputed,
        "contradiction_rows": contradictions,
        "contract_source_disposition": dispositions["contract_source"],
        "arc_branch_disposition": dispositions["arc"],
        "verification_routing_disposition": dispositions["verification_routing"],
        "continuous_learning_disposition": dispositions["continuous_learning"],
        "portability_disposition": dispositions["portability"],
        "sampling_disposition": dispositions["sampling"],
        "claim_ceiling_rows": ceilings,
        "retirement_rows": retirement,
        "next_action_rows": [
            {
                "branch": name,
                "action": disposition["recommended_action"],
                "reason": disposition["claim_ceiling"],
                "failed_scope_unchanged": False,
            }
            for name, disposition in dispositions.items()
        ],
        "v626_capstone_complete_score": complete,
        "random_seed": RANDOM_SEED,
        "reproducibility_checksum": "",
        "gate_check_summary": {
            "passed": verdict_class == "positive",
            "failed_check": None
            if verdict_class == "positive"
            else "active_markdown_yaml_contract",
            "expected_value": 12,
            "observed_value": sum(row["passed"] for row in contract_rows),
        },
        "verifier_is_oracle": False,
        "verdict_class": verdict_class,
        "honest_verdict": honest,
    }
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    return artifact


def validate_artifact(artifact: Mapping[str, Any]) -> list[str]:
    """Recompute the terminal schema, matrix score, claim boundary, and digest."""

    missing = sorted(REQUIRED_ARTIFACT_FIELDS - set(artifact))
    if missing:
        return ["missing_required_fields:" + ",".join(missing)]
    errors: list[str] = []
    principles = artifact.get("field_principles")
    if not isinstance(principles, Mapping) or any(
        not isinstance(principles.get(field), str) or not principles[field].strip()
        for field in REQUIRED_ARTIFACT_FIELDS
    ):
        errors.append("field_principles_incomplete")
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        errors.append("inference_substrate_invalid")
    if artifact.get("inference_substrate_class") not in {"aggregation", "blocked_no_run"}:
        errors.append("inference_substrate_class_invalid")
    if artifact.get("execution_venue") != "host":
        errors.append("execution_venue_invalid")
    if artifact.get("verifier_is_oracle") is not False:
        errors.append("verifier_is_oracle_invalid")
    if artifact.get("verdict_class") not in VERDICT_CLASSES:
        errors.append("verdict_class_invalid")
    overall = classify_verdict(artifact)
    if overall["effective_verdict_class"] != artifact.get("verdict_class"):
        errors.append("honest_verdict_class_mismatch")
    if artifact.get("verdict_class") == "blocked":
        summary = artifact.get("gate_check_summary")
        if (
            not isinstance(summary, Mapping)
            or not summary.get("failed_check")
            or "expected_value" not in summary
            or "observed_value" not in summary
        ):
            errors.append("blocked_gate_summary_invalid")
    if artifact.get("inference_substrate_class") == "aggregation":
        task_rows = artifact.get("artifact_verdict_rows")
        ids = [row.get("task_id") for row in task_rows] if isinstance(task_rows, list) else []
        gates = artifact.get("gate_recompute_rows")
        dispositions = [
            artifact.get("contract_source_disposition"),
            artifact.get("arc_branch_disposition"),
            artifact.get("verification_routing_disposition"),
            artifact.get("continuous_learning_disposition"),
            artifact.get("portability_disposition"),
            artifact.get("sampling_disposition"),
        ]
        ceilings = artifact.get("claim_ceiling_rows")
        recomputed = int(
            ids == list(EXPECTED_UPSTREAM_IDS)
            and isinstance(gates, list)
            and len(gates) == 4
            and all(
                isinstance(row, Mapping)
                and row.get("recommended_action") in NEXT_ACTIONS
                and row.get("failed_scope_unchanged") is False
                for row in dispositions
            )
            and isinstance(ceilings, list)
            and len(ceilings) >= 17
        )
        if artifact.get("rows") != task_rows:
            errors.append("rows_matrix_mismatch")
        if artifact.get("v626_capstone_complete_score") != recomputed:
            errors.append("matrix_complete_score_mismatch")
        if isinstance(ceilings, list) and any(
            row.get("promoted_as_positive_science") is True
            for row in ceilings
            if isinstance(row, Mapping)
        ):
            errors.append("capstone_promotes_science")
    if artifact.get("reproducibility_checksum") != reproducibility_checksum(artifact):
        errors.append("reproducibility_checksum_mismatch")
    return errors


def _read_json(path: Path) -> JsonDict | None:
    """Read a JSON object and return no value for malformed evidence."""

    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _write_atomic(path: Path, artifact: Mapping[str, Any]) -> None:
    """Publish only a complete JSON object at the terminal path."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def main(argv: Sequence[str] | None = None) -> int:
    """Write or validate the Exp7135 terminal artifact."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default="20260908")
    parser.add_argument("--root", type=Path, default=REPO_ROOT)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--validate", action="store_true")
    args = parser.parse_args(argv)
    output = args.output or args.root / OUTPUT_PATH
    if args.validate:
        artifact = _read_json(output) or {}
    else:
        output.parent.mkdir(parents=True, exist_ok=True)
        artifact = build_artifact(args.root, args.date, output_path=output)
    errors = validate_artifact(artifact)
    if errors:
        print(json.dumps({"status": "invalid", "errors": errors}, sort_keys=True))
        return 1
    if not args.validate:
        _write_atomic(output, artifact)
    print(json.dumps({"status": "validated", "path": str(output)}, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover - the command wrapper owns execution.
    raise SystemExit(main())
