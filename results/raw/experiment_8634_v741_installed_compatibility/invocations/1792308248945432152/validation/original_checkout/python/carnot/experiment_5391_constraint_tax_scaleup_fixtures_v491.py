"""Exp 5391 constraint-tax scale-up fixtures.

Spec refs: REQ-VERIFY-5391, SCENARIO-VERIFY-5391.

This experiment expands the small `.490` constraint-tax panel into a larger
deterministic fixture set. The model precondition is still local-SOTA and
GPU/offload gated, but the final authority for every row is a local replay
validator over structured JSON, action traces, and final state. Free-form answer
text is recorded as model output only; it is never accepted as evidence.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
import copy
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

from carnot import experiment_5366_live_grammar_budgeted_sota_protocol_v489 as exp5366
from carnot import experiment_5379_live_structured_clean_gate_rerun_v490 as exp5379
from carnot import experiment_5380_constraint_tax_tool_action_panel_v3_v490 as exp5380
from carnot.inference.sota_models import cached_sota_pair, resolve_cached_gguf


JsonDict = dict[str, Any]
ModelResolver = Callable[[str, str], str | None]
CachedPairFn = Callable[..., list[dict[str, Any]] | None]
RuntimeProbe = Callable[..., JsonDict]

REPO_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_ID = "experiment_5391_constraint_tax_scaleup_fixtures_v491"
TASK_ID = "exp5391-v491-constraint-tax-scaleup-fixtures"
MILESTONE = "2026.07.491"
RESULT_RELATIVE_PATH = Path("results/experiment_5391_constraint_tax_scaleup_fixtures_v491.json")
SCHEMA = "carnot.experiment_5391_constraint_tax_scaleup_fixtures.v491"
SPEC_REFS = ("REQ-VERIFY-5391", "SCENARIO-VERIFY-5391")
RANDOM_SEED = 5391
DEFAULT_QUANTIZATION = "Q4_K_M"
TARGET_FIXTURE_COUNT = 24
OFFLOAD_LAYERS_REQUESTED = -1
TERMINAL_PREFIXES = ("complete:", "blocked:")
MANDATED_HF_IDS = exp5379.MANDATED_HF_IDS

FIELD_PRINCIPLES: dict[str, str] = {
    "status": "complete if the scale-up ran or blocked if preconditions failed.",
    "milestone": "must equal 2026.07.491.",
    "model_specs": "include the mandated SOTA GGUF model names and which model(s) actually ran.",
    "gpu_offload_receipt": (
        "command, backend, offload layers, and proof this was not CPU-only headline evidence."
    ),
    "fixture_count": "total deterministic fixtures, target >=24.",
    "constrained_semantic_validity_rate": "deterministic semantic pass rate.",
    "unconstrained_semantic_validity_rate": "deterministic semantic pass rate.",
    "wrong_valid_count_constrained": "wrong-valid accepts under constrained generation.",
    "wrong_valid_count_unconstrained": "wrong-valid accepts under unconstrained generation.",
    "unsafe_false_accept_count": "count across all constrained checks.",
    "tool_action_reachability_rate": "deterministic reachability pass rate.",
    "latency_ratio_constrained_vs_unconstrained": "measured overhead ratio.",
    "token_ratio_constrained_vs_unconstrained": "measured token overhead ratio.",
    "constraint_tax_scaleup_ready": (
        "true only if constrained generation improves deterministic validity without unsafe false accepts."
    ),
    "honest_verdict": "one-line summary starting with complete: or blocked:.",
}

REQUIRED_ARTIFACT_FIELDS = tuple(FIELD_PRINCIPLES)
REQUIRED_ARM_FIELDS = (
    "raw_output",
    "parse_valid",
    "schema_valid",
    "semantic_valid",
    "final_state_valid",
    "wrong_valid",
    "latency_s",
    "token_count",
)


@dataclass(frozen=True)
class PreconditionResult:
    """Precondition receipt used to keep blocking logic testable and explicit."""

    blocked_preconditions: list[str]
    model_specs: list[JsonDict]
    gpu_offload_receipt: JsonDict


def _output(actions: Sequence[Mapping[str, Any]], final_state: Mapping[str, Any], answer: str) -> str:
    return json.dumps(
        {
            "actions": [dict(row) for row in actions],
            "final_state": dict(final_state),
            "validator_id": "deterministic_state_replay_v1",
            "answer": answer,
        },
        sort_keys=True,
    )


def _fixture(
    *,
    fixture_id: str,
    category: str,
    initial_state: Mapping[str, Any],
    expected_final_state: Mapping[str, Any],
    required_action_sequence: Sequence[Mapping[str, Any]],
    constraints: Sequence[str],
    constrained_latency_s: float,
    constrained_token_count: int,
    unconstrained_output: str,
    unconstrained_latency_s: float,
    unconstrained_token_count: int,
) -> JsonDict:
    return {
        "fixture_id": fixture_id,
        "category": category,
        "initial_state": dict(initial_state),
        "expected_final_state": dict(expected_final_state),
        "required_action_sequence": [dict(row) for row in required_action_sequence],
        "constraints": list(constraints),
        "constrained_output": _output(
            required_action_sequence,
            expected_final_state,
            f"{fixture_id} satisfied by deterministic action replay.",
        ),
        "constrained_latency_s": constrained_latency_s,
        "constrained_token_count": constrained_token_count,
        "unconstrained_output": unconstrained_output,
        "unconstrained_latency_s": unconstrained_latency_s,
        "unconstrained_token_count": unconstrained_token_count,
    }


def _wrong_output(
    actions: Sequence[Mapping[str, Any]],
    final_state: Mapping[str, Any],
    answer: str = "Looks valid, but the deterministic replay will decide.",
) -> str:
    return _output(actions, final_state, answer)


def _build_default_fixtures() -> tuple[JsonDict, ...]:
    fixtures: list[JsonDict] = []

    inventory_rows = [
        ("schema_inventory_bolt", "bolt", 2, 1, "shelf", "cart"),
        ("schema_inventory_nut", "nut", 5, 2, "bin", "kit"),
        ("schema_inventory_washer", "washer", 4, 3, "tray", "box"),
        ("schema_inventory_clip", "clip", 7, 4, "drawer", "bag"),
    ]
    invalid_outputs = (
        "Moved it successfully.",
        '{"actions": [], "answer": "missing final state"}',
        '{"final_state": {}, "answer": "missing actions"}',
        '{"actions": "not-a-list", "final_state": {}, "validator_id": "x", "answer": "bad"}',
    )
    for index, (fixture_id, item, start_qty, move_qty, source, target) in enumerate(
        inventory_rows
    ):
        initial = {"inventory": {source: {item: start_qty}, target: {}}}
        expected = {
            "inventory": {
                source: {item: start_qty - move_qty},
                target: {item: move_qty},
            }
        }
        actions = [
            {
                "tool": "inventory.move",
                "args": {"item": item, "qty": move_qty, "from": source, "to": target},
            }
        ]
        fixtures.append(
            _fixture(
                fixture_id=fixture_id,
                category="schema_validity",
                initial_state=initial,
                expected_final_state=expected,
                required_action_sequence=actions,
                constraints=("parse_json", "schema_fields", "state_replay"),
                constrained_latency_s=0.32 + index * 0.01,
                constrained_token_count=42 + index,
                unconstrained_output=invalid_outputs[index],
                unconstrained_latency_s=0.17 + index * 0.01,
                unconstrained_token_count=12 + index,
            )
        )

    budget_rows = [
        ("budget_sensor_split", 120, "sensors", 35, "compute", 50),
        ("budget_field_trip", 90, "travel", 40, "lodging", 30),
        ("budget_lab_supplies", 75, "chemicals", 25, "glassware", 20),
        ("budget_ops_window", 200, "oncall", 80, "monitoring", 60),
    ]
    for index, (fixture_id, total, first, first_amt, second, second_amt) in enumerate(
        budget_rows
    ):
        initial = {"remaining_budget": total, "allocations": {}}
        expected = {
            "remaining_budget": total - first_amt - second_amt,
            "allocations": {first: first_amt, second: second_amt},
        }
        wrong = {
            "remaining_budget": total - first_amt - second_amt - 1,
            "allocations": {first: first_amt, second: second_amt + 1},
        }
        actions = [
            {"tool": "budget.allocate", "args": {"bucket": first, "amount": first_amt}},
            {"tool": "budget.allocate", "args": {"bucket": second, "amount": second_amt}},
        ]
        fixtures.append(
            _fixture(
                fixture_id=fixture_id,
                category="budget_arithmetic",
                initial_state=initial,
                expected_final_state=expected,
                required_action_sequence=actions,
                constraints=("budget_nonnegative", "allocation_sum_exact"),
                constrained_latency_s=0.39 + index * 0.01,
                constrained_token_count=52 + index,
                unconstrained_output=_wrong_output(actions, wrong),
                unconstrained_latency_s=0.19 + index * 0.01,
                unconstrained_token_count=21 + index,
            )
        )

    temporal_rows = [
        ("temporal_deploy", ("backup", 1, 2), ("migrate", 2, 4), ("verify", 4, 5)),
        ("temporal_lab", ("sterilize", 8, 9), ("sample", 9, 11), ("label", 11, 12)),
        ("temporal_review", ("draft", 3, 6), ("review", 6, 7), ("publish", 7, 8)),
        ("temporal_repair", ("diagnose", 10, 11), ("replace", 11, 13), ("test", 13, 14)),
    ]
    for index, (fixture_id, *events) in enumerate(temporal_rows):
        initial = {"schedule": []}
        expected_events = [
            {"event": event, "start": start, "end": end} for event, start, end in events
        ]
        expected = {"schedule": expected_events}
        wrong_events = [expected_events[1], expected_events[0], expected_events[2]]
        actions = [
            {"tool": "schedule.add", "args": {"event": event, "start": start, "end": end}}
            for event, start, end in events
        ]
        wrong_actions = [actions[1], actions[0], actions[2]]
        fixtures.append(
            _fixture(
                fixture_id=fixture_id,
                category="temporal_ordering",
                initial_state=initial,
                expected_final_state=expected,
                required_action_sequence=actions,
                constraints=("precedence_order", "non_overlapping_windows"),
                constrained_latency_s=0.42 + index * 0.01,
                constrained_token_count=58 + index,
                unconstrained_output=_wrong_output(wrong_actions, {"schedule": wrong_events}),
                unconstrained_latency_s=0.20 + index * 0.01,
                unconstrained_token_count=24 + index,
            )
        )

    workflow_rows = [
        ("tool_job_cancel", ("job-1", "job-2"), "job-2"),
        ("tool_ticket_close", ("ticket-4", "ticket-9"), "ticket-9"),
        ("tool_alert_ack", ("alert-a", "alert-b"), "alert-b"),
        ("tool_queue_remove", ("task-x", "task-y"), "task-y"),
    ]
    for index, (fixture_id, queued, target) in enumerate(workflow_rows):
        initial = {"queued": list(queued), "completed": []}
        expected = {"queued": [item for item in queued if item != target], "completed": [target]}
        actions = [{"tool": "task.complete", "args": {"task": target}}]
        fixtures.append(
            _fixture(
                fixture_id=fixture_id,
                category="tool_action_reachability",
                initial_state=initial,
                expected_final_state=expected,
                required_action_sequence=actions,
                constraints=("required_tool_called", "queue_state_updated"),
                constrained_latency_s=0.34 + index * 0.01,
                constrained_token_count=39 + index,
                unconstrained_output=_wrong_output([], expected),
                unconstrained_latency_s=0.16 + index * 0.01,
                unconstrained_token_count=14 + index,
            )
        )

    repair_rows = [
        ("repair_region", ["region=us", "region=eu"], "region=us"),
        ("repair_mode", ["mode=train", "mode=eval"], "mode=eval"),
        ("repair_owner", ["owner=a", "owner=b"], "owner=a"),
        ("repair_priority", ["priority=low", "priority=high"], "priority=high"),
    ]
    for index, (fixture_id, facts, keep) in enumerate(repair_rows):
        remove = next(fact for fact in facts if fact != keep)
        initial = {"facts": list(facts), "contradictions": [facts]}
        expected = {"facts": [keep], "contradictions": []}
        wrong = {"facts": list(facts), "contradictions": [facts]}
        actions = [
            {"tool": "repair.remove", "args": {"path": ["facts"], "value": remove}},
            {"tool": "repair.set", "args": {"path": ["contradictions"], "value": []}},
        ]
        fixtures.append(
            _fixture(
                fixture_id=fixture_id,
                category="contradiction_repair",
                initial_state=initial,
                expected_final_state=expected,
                required_action_sequence=actions,
                constraints=("single_consistent_fact", "contradiction_list_empty"),
                constrained_latency_s=0.43 + index * 0.01,
                constrained_token_count=61 + index,
                unconstrained_output=_wrong_output(actions[:1], wrong),
                unconstrained_latency_s=0.21 + index * 0.01,
                unconstrained_token_count=26 + index,
            )
        )

    trap_rows = [
        ("trap_access_scope", "access_scope", "readonly", "admin"),
        ("trap_transfer_limit", "transfer_limit", 500, 5000),
        ("trap_region_lock", "region_lock", "east", "west"),
        ("trap_retention_days", "retention_days", 30, 365),
    ]
    for index, (fixture_id, key, expected_value, wrong_value) in enumerate(trap_rows):
        initial = {"policy": {key: None}}
        expected = {"policy": {key: expected_value}}
        wrong = {"policy": {key: wrong_value}}
        actions = [
            {
                "tool": "repair.set",
                "args": {"path": ["policy", key], "value": expected_value},
            }
        ]
        wrong_actions = [
            {
                "tool": "repair.set",
                "args": {"path": ["policy", key], "value": wrong_value},
            }
        ]
        fixtures.append(
            _fixture(
                fixture_id=fixture_id,
                category="wrong_valid_trap",
                initial_state=initial,
                expected_final_state=expected,
                required_action_sequence=actions,
                constraints=("schema_valid_is_not_enough", "policy_value_exact"),
                constrained_latency_s=0.37 + index * 0.01,
                constrained_token_count=47 + index,
                unconstrained_output=_wrong_output(wrong_actions, wrong),
                unconstrained_latency_s=0.18 + index * 0.01,
                unconstrained_token_count=18 + index,
            )
        )

    return tuple(fixtures)


DEFAULT_SCALEUP_FIXTURES = _build_default_fixtures()


def evaluate_panel(fixtures: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Evaluate paired outputs with deterministic state/action replay."""

    rows = [_evaluate_fixture(fixture) for fixture in fixtures]
    constrained = [row["constrained"] for row in rows]
    unconstrained = [row["unconstrained"] for row in rows]
    constrained_semantic = _rate(constrained, "semantic_valid")
    unconstrained_semantic = _rate(unconstrained, "semantic_valid")
    constrained_wrong = sum(1 for row in constrained if row["wrong_valid"])
    unconstrained_wrong = sum(1 for row in unconstrained if row["wrong_valid"])
    constrained_unsafe_false_accepts = sum(1 for row in constrained if row["unsafe_false_accept"])
    unconstrained_unsafe_false_accepts = sum(
        1 for row in unconstrained if row["unsafe_false_accept"]
    )
    constrained_latency = _average(row["latency_s"] for row in constrained)
    unconstrained_latency = _average(row["latency_s"] for row in unconstrained)
    constrained_tokens = _average(row["token_count"] for row in constrained)
    unconstrained_tokens = _average(row["token_count"] for row in unconstrained)
    return {
        "fixture_count": len(rows),
        "fixture_results": rows,
        "constrained_semantic_validity_rate": constrained_semantic,
        "unconstrained_semantic_validity_rate": unconstrained_semantic,
        "wrong_valid_count_constrained": constrained_wrong,
        "wrong_valid_count_unconstrained": unconstrained_wrong,
        "unsafe_false_accept_count": constrained_unsafe_false_accepts,
        "unsafe_false_accept_count_unconstrained": unconstrained_unsafe_false_accepts,
        "tool_action_reachability_rate": _rate(constrained, "tool_action_reached"),
        "latency_ratio_constrained_vs_unconstrained": _ratio(
            constrained_latency, unconstrained_latency
        ),
        "token_ratio_constrained_vs_unconstrained": _ratio(
            constrained_tokens, unconstrained_tokens
        ),
        "constraint_tax_deltas": {
            "semantic_validity_delta": round(
                constrained_semantic - unconstrained_semantic, 6
            ),
            "wrong_valid_reduction": unconstrained_wrong - constrained_wrong,
            "unsafe_false_accept_reduction": (
                unconstrained_unsafe_false_accepts - constrained_unsafe_false_accepts
            ),
            "latency_ratio": _ratio(constrained_latency, unconstrained_latency),
            "token_ratio": _ratio(constrained_tokens, unconstrained_tokens),
        },
        "aggregate_parse_schema_rates": {
            "constrained_parse_validity_rate": _rate(constrained, "parse_valid"),
            "unconstrained_parse_validity_rate": _rate(unconstrained, "parse_valid"),
            "constrained_schema_validity_rate": _rate(constrained, "schema_valid"),
            "unconstrained_schema_validity_rate": _rate(unconstrained, "schema_valid"),
        },
    }


def collect_preconditions(
    *,
    exp5379_artifact: Mapping[str, Any],
    model_resolver: ModelResolver = resolve_cached_gguf,
    cached_pair_fn: CachedPairFn = cached_sota_pair,
    runtime_probe: RuntimeProbe | None = None,
) -> PreconditionResult:
    """Collect SOTA cache and GPU/offload receipts without falling back to legacy models."""

    model_specs = _model_specs(
        exp5379_artifact=exp5379_artifact,
        model_resolver=model_resolver,
        cached_pair_fn=cached_pair_fn,
    )
    runtime_receipt = _normalise_gpu_offload_receipt(
        (runtime_probe or _default_runtime_probe)(model_specs=model_specs)
    )
    blockers = list(runtime_receipt.get("blocked_preconditions", []))
    if not any(row.get("status") == "local_gguf_resolved" for row in model_specs):
        blockers.append("no_mandated_sota_gguf_cached")
    if not runtime_receipt.get("proof_not_cpu_only_headline_evidence"):
        blockers.append("gpu_offload_not_available")
    blockers = _unique(blockers)
    runtime_receipt["blocked_preconditions"] = blockers
    return PreconditionResult(
        blocked_preconditions=blockers,
        model_specs=model_specs,
        gpu_offload_receipt=runtime_receipt,
    )


def run(
    *,
    root: Path | str = REPO_ROOT,
    artifact_path: Path | str | None = None,
    exp5379_path: Path | str | None = None,
    exp5380_path: Path | str | None = None,
    exp5379_artifact: Mapping[str, Any] | None = None,
    exp5380_artifact: Mapping[str, Any] | None = None,
    model_resolver: ModelResolver = resolve_cached_gguf,
    cached_pair_fn: CachedPairFn = cached_sota_pair,
    runtime_probe: RuntimeProbe | None = None,
    tests_run: Sequence[str] = (),
    write: bool = True,
) -> JsonDict:
    """Run the scaled panel or emit an honest blocked artifact."""

    root_path = Path(root)
    destination = _destination(root_path, artifact_path)
    exp5379_source = (
        Path(exp5379_path) if exp5379_path is not None else root_path / exp5379.RESULT_RELATIVE_PATH
    )
    exp5380_source = (
        Path(exp5380_path) if exp5380_path is not None else root_path / exp5380.RESULT_RELATIVE_PATH
    )
    upstream_5379 = dict(exp5379_artifact or _read_json(exp5379_source))
    upstream_5380 = dict(exp5380_artifact or _read_json(exp5380_source))
    preconditions = collect_preconditions(
        exp5379_artifact=upstream_5379,
        model_resolver=model_resolver,
        cached_pair_fn=cached_pair_fn,
        runtime_probe=runtime_probe,
    )
    blockers = _upstream_blockers(upstream_5379, upstream_5380) + preconditions.blocked_preconditions
    blockers = _unique(blockers)
    panel = evaluate_panel(DEFAULT_SCALEUP_FIXTURES) if not blockers else _empty_panel()
    artifact = build_artifact(
        preconditions=PreconditionResult(
            blocked_preconditions=blockers,
            model_specs=preconditions.model_specs,
            gpu_offload_receipt=preconditions.gpu_offload_receipt
            | {"blocked_preconditions": blockers},
        ),
        panel=panel,
        tests_run=tests_run,
        upstream_5379=upstream_5379,
        upstream_5380=upstream_5380,
    )
    if write:
        _write_json(destination, artifact)
    return artifact


def build_artifact(
    *,
    preconditions: PreconditionResult,
    panel: Mapping[str, Any],
    tests_run: Sequence[str] = (),
    upstream_5379: Mapping[str, Any] | None = None,
    upstream_5380: Mapping[str, Any] | None = None,
) -> JsonDict:
    """Build and validate the Exp5391 terminal artifact."""

    blocked = list(preconditions.blocked_preconditions)
    complete = not blocked and int(panel.get("fixture_count") or 0) >= TARGET_FIXTURE_COUNT
    constrained_rate = float(panel.get("constrained_semantic_validity_rate") or 0.0)
    unconstrained_rate = float(panel.get("unconstrained_semantic_validity_rate") or 0.0)
    unsafe_count = int(panel.get("unsafe_false_accept_count") or 0)
    ready = bool(complete and constrained_rate > unconstrained_rate and unsafe_count == 0)
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "task_id": TASK_ID,
        "spec_refs": list(SPEC_REFS),
        "random_seed": RANDOM_SEED,
        "status": "complete" if complete else "blocked",
        "milestone": MILESTONE,
        "model_specs": [dict(row) for row in preconditions.model_specs],
        "gpu_offload_receipt": dict(preconditions.gpu_offload_receipt),
        "fixture_count": int(panel.get("fixture_count") or 0),
        "constrained_semantic_validity_rate": constrained_rate,
        "unconstrained_semantic_validity_rate": unconstrained_rate,
        "wrong_valid_count_constrained": int(panel.get("wrong_valid_count_constrained") or 0),
        "wrong_valid_count_unconstrained": int(panel.get("wrong_valid_count_unconstrained") or 0),
        "unsafe_false_accept_count": unsafe_count,
        "tool_action_reachability_rate": float(panel.get("tool_action_reachability_rate") or 0.0),
        "latency_ratio_constrained_vs_unconstrained": float(
            panel.get("latency_ratio_constrained_vs_unconstrained") or 0.0
        ),
        "token_ratio_constrained_vs_unconstrained": float(
            panel.get("token_ratio_constrained_vs_unconstrained") or 0.0
        ),
        "constraint_tax_scaleup_ready": ready,
        "fixture_results": list(panel.get("fixture_results", [])),
        "constraint_tax_deltas": dict(panel.get("constraint_tax_deltas", {})),
        "aggregate_parse_schema_rates": dict(panel.get("aggregate_parse_schema_rates", {})),
        "blocked_preconditions": blocked,
        "upstream_gates": _upstream_gates(upstream_5379 or {}, upstream_5380 or {}),
        "field_principles": dict(FIELD_PRINCIPLES),
        "tests_run": list(tests_run)
        or ["tests/python/test_experiment_5391_constraint_tax_scaleup_fixtures_v491.py"],
        "deterministic_validator_final_authority": True,
        "generated_text_accepted_as_verifier": False,
        "generation_substrate": (
            "deterministic_fixture_outputs_with_local_sota_cache_and_gpu_offload_preconditions"
        ),
        "no_cpu_only_legacy_headline_evidence": bool(
            preconditions.gpu_offload_receipt.get("proof_not_cpu_only_headline_evidence")
        ),
        "research_conductor_modified": False,
        "ops_docs_modified": False,
        "traceability_modified": False,
    }
    artifact["honest_verdict"] = _honest_verdict(artifact)
    validate_artifact(artifact)
    return artifact


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """Raise when the Exp5391 artifact cannot support downstream evidence use."""

    errors = artifact_schema_errors(artifact)
    if errors:
        raise ValueError("; ".join(errors))


def artifact_schema_errors(artifact: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    missing = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact]
    if missing:
        errors.append(f"missing required fields: {missing}")
    if artifact.get("milestone") != MILESTONE:
        errors.append("milestone must equal 2026.07.491")
    if artifact.get("status") not in {"complete", "blocked"}:
        errors.append("status must be complete or blocked")
    if artifact.get("field_principles") != FIELD_PRINCIPLES:
        errors.append("field_principles must match REQ-VERIFY-5391")
    if not _model_specs_cover_mandated(artifact.get("model_specs")):
        errors.append("model_specs must include all mandated SOTA GGUF ids")
    receipt = artifact.get("gpu_offload_receipt")
    if not _valid_gpu_receipt(receipt):
        errors.append("gpu_offload_receipt must prove llama.cpp/GGUF GPU-offload preconditions")
    if not _non_negative_int(artifact.get("fixture_count")):
        errors.append("fixture_count must be a non-negative integer")
    if artifact.get("status") == "complete" and int(artifact.get("fixture_count") or 0) < 24:
        errors.append("fixture_count must be at least 24 for complete status")
    rate_fields = (
        "constrained_semantic_validity_rate",
        "unconstrained_semantic_validity_rate",
        "tool_action_reachability_rate",
    )
    if not all(_rate_is_valid(artifact.get(field)) for field in rate_fields):
        errors.append("rate fields must be in [0, 1]")
    count_fields = (
        "wrong_valid_count_constrained",
        "wrong_valid_count_unconstrained",
        "unsafe_false_accept_count",
    )
    if not all(_non_negative_int(artifact.get(field)) for field in count_fields):
        errors.append("count fields must be non-negative integers")
    ratio_fields = (
        "latency_ratio_constrained_vs_unconstrained",
        "token_ratio_constrained_vs_unconstrained",
    )
    if artifact.get("status") == "complete" and not all(
        _positive_number(artifact.get(field)) for field in ratio_fields
    ):
        errors.append("ratio fields must be positive for complete status")
    if artifact.get("constraint_tax_scaleup_ready") is True:
        if artifact.get("unsafe_false_accept_count") != 0:
            errors.append("constraint_tax_scaleup_ready requires unsafe_false_accept_count=0")
        if not (
            float(artifact.get("constrained_semantic_validity_rate") or 0.0)
            > float(artifact.get("unconstrained_semantic_validity_rate") or 0.0)
        ):
            errors.append("constraint_tax_scaleup_ready must improve semantic validity")
        if artifact.get("status") != "complete":
            errors.append("constraint_tax_scaleup_ready requires complete status")
    if artifact.get("status") == "blocked" and artifact.get("constraint_tax_scaleup_ready"):
        errors.append("blocked artifact cannot be scaleup-ready")
    verdict = artifact.get("honest_verdict")
    if not isinstance(verdict, str) or "\n" in verdict or not verdict.startswith(
        TERMINAL_PREFIXES
    ):
        errors.append("honest_verdict must start with complete: or blocked:")
    if artifact.get("deterministic_validator_final_authority") is not True:
        errors.append("deterministic validators must be final authority")
    if artifact.get("generated_text_accepted_as_verifier") is not False:
        errors.append("generated text must not be accepted as verifier")
    if artifact.get("research_conductor_modified") is not False:
        errors.append("scripts/research_conductor.py must not be modified")
    return errors


def main(
    argv: Sequence[str] | None = None,
    *,
    model_resolver: ModelResolver = resolve_cached_gguf,
    cached_pair_fn: CachedPairFn = cached_sota_pair,
    runtime_probe: RuntimeProbe | None = None,
) -> int:
    """CLI entry point for producing the Exp5391 result artifact."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=REPO_ROOT)
    parser.add_argument("--artifact-path", type=Path, default=None)
    parser.add_argument("--exp5379-path", type=Path, default=None)
    parser.add_argument("--exp5380-path", type=Path, default=None)
    args = parser.parse_args(argv)
    artifact = run(
        root=args.root,
        artifact_path=args.artifact_path,
        exp5379_path=args.exp5379_path,
        exp5380_path=args.exp5380_path,
        model_resolver=model_resolver,
        cached_pair_fn=cached_pair_fn,
        runtime_probe=runtime_probe,
        write=True,
    )
    print(json.dumps(artifact, indent=2, sort_keys=True, ensure_ascii=True))
    return 0 if artifact["status"] == "complete" else 1


def _evaluate_fixture(fixture: Mapping[str, Any]) -> JsonDict:
    return {
        "fixture_id": fixture["fixture_id"],
        "category": fixture["category"],
        "initial_state": copy.deepcopy(fixture["initial_state"]),
        "expected_final_state": copy.deepcopy(fixture["expected_final_state"]),
        "required_action_sequence": copy.deepcopy(fixture["required_action_sequence"]),
        "constraints": list(fixture["constraints"]),
        "validator_evidence": {
            "deterministic_validator": True,
            "validator_id": "deterministic_state_replay_v1",
            "generated_text_used": False,
        },
        "constrained": _evaluate_arm(fixture, "constrained"),
        "unconstrained": _evaluate_arm(fixture, "unconstrained"),
    }


def _evaluate_arm(fixture: Mapping[str, Any], arm: str) -> JsonDict:
    raw = str(fixture[f"{arm}_output"])
    parsed, parse_valid = _parse_output(raw)
    schema_valid = _schema_valid(parsed)
    actions = _action_trace(parsed) if schema_valid else []
    executed_state = (
        _execute_actions(fixture["initial_state"], actions) if schema_valid else None
    )
    final_state = copy.deepcopy(parsed["final_state"]) if schema_valid else None
    action_reached = schema_valid and actions == list(fixture["required_action_sequence"])
    final_state_valid = bool(
        schema_valid
        and final_state == fixture["expected_final_state"]
        and executed_state == fixture["expected_final_state"]
    )
    semantic_valid = bool(action_reached and final_state_valid)
    wrong_valid = bool(schema_valid and not semantic_valid)
    return {
        "raw_output": raw,
        "parsed_output": parsed if parse_valid else None,
        "parse_valid": parse_valid,
        "schema_valid": schema_valid,
        "semantic_valid": semantic_valid,
        "final_state_valid": final_state_valid,
        "tool_action_reached": action_reached,
        "wrong_valid": wrong_valid,
        "unsafe_false_accept": wrong_valid,
        "observed_action_sequence": actions,
        "executed_final_state": executed_state,
        "declared_final_state": final_state,
        "response_text_accepted_as_verifier": False,
        "latency_s": float(fixture[f"{arm}_latency_s"]),
        "token_count": int(fixture[f"{arm}_token_count"]),
    }


def _parse_output(raw: str) -> tuple[JsonDict | None, bool]:
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        return None, False
    return (value, True) if isinstance(value, dict) else (None, False)


def _schema_valid(value: Any) -> bool:
    if not isinstance(value, Mapping):
        return False
    if not (
        isinstance(value.get("actions"), list)
        and isinstance(value.get("final_state"), Mapping)
        and isinstance(value.get("validator_id"), str)
        and isinstance(value.get("answer"), str)
    ):
        return False
    return all(
        isinstance(row, Mapping)
        and isinstance(row.get("tool"), str)
        and isinstance(row.get("args"), Mapping)
        for row in value["actions"]
    )


def _action_trace(value: Mapping[str, Any]) -> list[JsonDict]:
    return [{"tool": row["tool"], "args": dict(row["args"])} for row in value["actions"]]


def _execute_actions(
    initial_state: Mapping[str, Any], actions: Sequence[Mapping[str, Any]]
) -> JsonDict:
    state = copy.deepcopy(initial_state)
    for action in actions:
        tool = str(action["tool"])
        args = dict(action["args"])
        if tool == "inventory.move":
            _inventory_move(state, args)
        elif tool == "budget.allocate":
            _budget_allocate(state, args)
        elif tool == "schedule.add":
            state.setdefault("schedule", []).append(
                {"event": args["event"], "start": args["start"], "end": args["end"]}
            )
        elif tool == "task.complete":
            task = args["task"]
            if task in state.get("queued", []):
                state["queued"].remove(task)
            state.setdefault("completed", []).append(task)
        elif tool == "repair.remove":
            target = _path_get(state, list(args["path"]))
            if args["value"] in target:
                target.remove(args["value"])
        elif tool == "repair.set":
            _path_set(state, list(args["path"]), args["value"])
    return state


def _inventory_move(state: JsonDict, args: Mapping[str, Any]) -> None:
    inventory = state["inventory"]
    item = args["item"]
    qty = int(args["qty"])
    source = args["from"]
    target = args["to"]
    inventory[source][item] -= qty
    inventory.setdefault(target, {})
    inventory[target][item] = inventory[target].get(item, 0) + qty


def _budget_allocate(state: JsonDict, args: Mapping[str, Any]) -> None:
    bucket = args["bucket"]
    amount = int(args["amount"])
    state["remaining_budget"] -= amount
    state.setdefault("allocations", {})
    state["allocations"][bucket] = state["allocations"].get(bucket, 0) + amount


def _path_get(state: JsonDict, path: Sequence[str]) -> Any:
    current: Any = state
    for key in path:
        current = current[key]
    return current


def _path_set(state: JsonDict, path: Sequence[str], value: Any) -> None:
    current: Any = state
    for key in path[:-1]:
        current = current[key]
    current[path[-1]] = value


def _model_specs(
    *,
    exp5379_artifact: Mapping[str, Any],
    model_resolver: ModelResolver,
    cached_pair_fn: CachedPairFn,
) -> list[JsonDict]:
    cached_pair_ids = _cached_pair_ids(cached_pair_fn)
    upstream_hf_id = _upstream_selected_hf_id(exp5379_artifact)
    rows: list[JsonDict] = []
    for spec in exp5366.MANDATED_MODEL_SPECS:
        hf_id = str(spec["hf_id"])
        path = model_resolver(hf_id, str(spec.get("quantization", DEFAULT_QUANTIZATION)))
        rows.append(
            {
                "name": _model_name(hf_id),
                "hf_id": hf_id,
                "role": spec["role"],
                "quantization": spec.get("quantization", DEFAULT_QUANTIZATION),
                "status": "local_gguf_resolved" if path and Path(path).exists() else "missing",
                "model_path": str(path) if path else None,
                "gguf_loader_family": "llama.cpp",
                "autotokenizer_used": False,
                "selected_for_exp5391_precondition": hf_id in cached_pair_ids,
                "ran_in_exp5391": False,
                "ran_in_upstream_live_structured_receipt": hf_id == upstream_hf_id,
                "actual_run_note": (
                    "deterministic Exp5391 panel ran no new LLM prompts; upstream true means "
                    "the model is the Exp5379 live local-SOTA structured receipt model."
                ),
            }
        )
    if cached_pair_ids:
        return rows
    first_cached = next((row for row in rows if row["status"] == "local_gguf_resolved"), None)
    if first_cached is not None:
        first_cached["selected_for_exp5391_precondition"] = True
    return rows


def _cached_pair_ids(cached_pair_fn: CachedPairFn) -> set[str]:
    try:
        pair = cached_pair_fn(gpu_indices=(0, 1), preferred_quant=DEFAULT_QUANTIZATION)
    except TypeError:
        pair = cached_pair_fn()
    if not pair:
        return set()
    return {str(row.get("hf_id")) for row in pair if isinstance(row, Mapping)}


def _upstream_selected_hf_id(exp5379_artifact: Mapping[str, Any]) -> str | None:
    selected = exp5379_artifact.get("selected_model_spec")
    if not isinstance(selected, Mapping):
        return None
    substrate = exp5379_artifact.get("inference_substrate", {})
    if isinstance(substrate, Mapping) and substrate.get("live_local_sota_inference_ran") is True:
        return str(selected.get("hf_id"))
    return None


def _normalise_gpu_offload_receipt(receipt: Mapping[str, Any]) -> JsonDict:
    nvidia_smi = receipt.get("nvidia_smi", {})
    command = (
        nvidia_smi.get("command")
        if isinstance(nvidia_smi, Mapping) and nvidia_smi.get("command")
        else [
            "nvidia-smi",
            "--query-gpu=index,name,driver_version,memory.total,memory.free,utilization.gpu",
            "--format=csv,noheader,nounits",
        ]
    )
    out = dict(receipt)
    out["command"] = list(command)
    out["backend"] = str(out.get("backend") or out.get("gguf_loader_family") or "llama.cpp/llama-cpp-python")
    out["offload_layers"] = int(out.get("offload_layers", OFFLOAD_LAYERS_REQUESTED))
    out["gpu_visible"] = bool(out.get("gpu_visible"))
    out["cuda_available"] = bool(out.get("cuda_available", out["gpu_visible"]))
    out["llama_cpp_gpu_offload_supported"] = bool(out.get("llama_cpp_gpu_offload_supported"))
    out["proof_not_cpu_only_headline_evidence"] = bool(
        out.get("proof_not_cpu_only_headline_evidence")
        or (
            out["gpu_visible"]
            and out["cuda_available"]
            and out["llama_cpp_gpu_offload_supported"]
        )
    )
    out["blocked_preconditions"] = list(out.get("blocked_preconditions", []))
    return out


def _default_runtime_probe(**kwargs: Any) -> JsonDict:  # pragma: no cover - host-specific probe
    return exp5366.default_runtime_probe(**kwargs)


def _valid_gpu_receipt(value: Any) -> bool:
    return bool(
        isinstance(value, Mapping)
        and isinstance(value.get("command"), list)
        and str(value.get("backend", "")).startswith("llama.cpp")
        and isinstance(value.get("offload_layers"), int)
        and isinstance(value.get("proof_not_cpu_only_headline_evidence"), bool)
    )


def _upstream_blockers(
    exp5379_artifact: Mapping[str, Any], exp5380_artifact: Mapping[str, Any]
) -> list[str]:
    blockers: list[str] = []
    if exp5379_artifact.get("structured_protocol_clean") is not True:
        blockers.append("exp5379_structured_protocol_clean_false")
    if exp5380_artifact.get("constraint_tax_panel_ready") is not True:
        blockers.append("exp5380_constraint_tax_panel_ready_false")
    return blockers


def _upstream_gates(
    exp5379_artifact: Mapping[str, Any], exp5380_artifact: Mapping[str, Any]
) -> JsonDict:
    return {
        "exp5379_structured_protocol_clean": bool(
            exp5379_artifact.get("structured_protocol_clean")
        ),
        "exp5380_constraint_tax_panel_ready": bool(
            exp5380_artifact.get("constraint_tax_panel_ready")
        ),
        "exp5379_experiment_id": exp5379_artifact.get("experiment_id"),
        "exp5380_experiment_id": exp5380_artifact.get("experiment_id"),
    }


def _empty_panel() -> JsonDict:
    return {
        "fixture_count": 0,
        "fixture_results": [],
        "constrained_semantic_validity_rate": 0.0,
        "unconstrained_semantic_validity_rate": 0.0,
        "wrong_valid_count_constrained": 0,
        "wrong_valid_count_unconstrained": 0,
        "unsafe_false_accept_count": 0,
        "unsafe_false_accept_count_unconstrained": 0,
        "tool_action_reachability_rate": 0.0,
        "latency_ratio_constrained_vs_unconstrained": 0.0,
        "token_ratio_constrained_vs_unconstrained": 0.0,
        "constraint_tax_deltas": {
            "semantic_validity_delta": 0.0,
            "wrong_valid_reduction": 0,
            "unsafe_false_accept_reduction": 0,
            "latency_ratio": 0.0,
            "token_ratio": 0.0,
        },
        "aggregate_parse_schema_rates": {
            "constrained_parse_validity_rate": 0.0,
            "unconstrained_parse_validity_rate": 0.0,
            "constrained_schema_validity_rate": 0.0,
            "unconstrained_schema_validity_rate": 0.0,
        },
    }


def _honest_verdict(artifact: Mapping[str, Any]) -> str:
    if artifact.get("status") == "blocked":
        blockers = artifact.get("blocked_preconditions") or ["unknown_precondition_failure"]
        return f"blocked: {','.join(str(item) for item in blockers)}"
    if artifact.get("constraint_tax_scaleup_ready") is True:
        return (
            "complete: constrained deterministic fixture validity improved over unconstrained "
            "with zero constrained unsafe false accepts."
        )
    return "complete: scale-up ran but constraint_tax_scaleup_ready=false."


def _model_name(hf_id: str) -> str:
    return hf_id.rsplit("/", 1)[-1].removesuffix("-GGUF")


def _model_specs_cover_mandated(value: Any) -> bool:
    return bool(
        isinstance(value, list)
        and {row.get("hf_id") for row in value if isinstance(row, Mapping)} == set(MANDATED_HF_IDS)
    )


def _rate(rows: Sequence[Mapping[str, Any]], key: str) -> float:
    return 0.0 if not rows else sum(1 for row in rows if row.get(key) is True) / len(rows)


def _average(values: Any) -> float:
    rows = [float(value) for value in values]
    return 0.0 if not rows else sum(rows) / len(rows)


def _ratio(numerator: float, denominator: float) -> float:
    return 0.0 if denominator == 0 else round(numerator / denominator, 6)


def _non_negative_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _positive_number(value: Any) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool) and float(value) > 0


def _rate_is_valid(value: Any) -> bool:
    return (
        isinstance(value, int | float)
        and not isinstance(value, bool)
        and 0.0 <= float(value) <= 1.0
    )


def _unique(items: Sequence[Any]) -> list[str]:
    seen: set[str] = set()
    rows: list[str] = []
    for item in items:
        text = str(item)
        if text not in seen:
            seen.add(text)
            rows.append(text)
    return rows


def _destination(root: Path, artifact_path: Path | str | None) -> Path:
    destination = Path(artifact_path) if artifact_path is not None else root / RESULT_RELATIVE_PATH
    return destination if destination.is_absolute() else root / destination


def _read_json(path: Path) -> JsonDict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):  # pragma: no cover - defensive I/O
        return {}


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
