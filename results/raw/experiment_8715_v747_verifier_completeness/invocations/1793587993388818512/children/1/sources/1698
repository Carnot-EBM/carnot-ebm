#!/usr/bin/env python3
"""Exp 5380 constraint-tax tool/action reachability panel.

Spec refs: REQ-VERIFY-5380, SCENARIO-VERIFY-5380.

This experiment consumes the clean Exp 5379 structured-output gate and then
checks a different question: whether constraints buy deterministic tool/action
reachability and correct state transitions, rather than just well-formed JSON.
The default path is a deterministic replay panel with fixed paired fixtures, so
no new model-quality claim is introduced after the Exp 5379 live GGUF receipt.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import copy
import json
from pathlib import Path
from typing import Any

from carnot import experiment_5379_live_structured_clean_gate_rerun_v490 as exp5379


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_ID = "experiment_5380_constraint_tax_tool_action_panel_v3_v490"
MILESTONE = "2026.07.490"
RESULT_RELATIVE_PATH = Path("results/experiment_5380_constraint_tax_tool_action_panel_v3_v490.json")
SCHEMA = "carnot.experiment_5380.constraint_tax_tool_action_panel.v490"
SPEC_REFS = ("REQ-VERIFY-5380", "SCENARIO-VERIFY-5380")
RANDOM_SEED = 5380
MANDATED_HF_IDS = exp5379.MANDATED_HF_IDS
TERMINAL_PREFIXES = ("complete:", "blocked_")

REQUIRED_ARTIFACT_FIELDS = (
    "status",
    "upstream_structured_protocol_clean",
    "constraint_tax_panel_ready",
    "MODEL_SPECS",
    "selected_model_spec",
    "inference_substrate",
    "fixture_count",
    "constrained_schema_validity_rate",
    "unconstrained_schema_validity_rate",
    "constrained_semantic_success_rate",
    "unconstrained_semantic_success_rate",
    "wrong_valid_count",
    "deterministic_state_evidence_count",
    "tool_action_reachability_rate",
    "latency_or_token_overhead",
    "unsafe_false_accepts",
    "honest_verdict",
)

FIELD_PRINCIPLES: dict[str, str] = {
    "status": "complete only if the upstream clean structured gate is true and the panel ran.",
    "upstream_structured_protocol_clean": "copied from Exp5379.",
    "constraint_tax_panel_ready": (
        "true only if deterministic state/tool-call evidence exists and unsafe_false_accepts=0."
    ),
    "MODEL_SPECS": (
        "list containing all mandated local GGUF model specs considered for live LLM panel results."
    ),
    "selected_model_spec": "exact model spec used if live LLM calls ran.",
    "inference_substrate": "concrete runtime path or explicit no-live-LLM explanation.",
    "fixture_count": "number of paired fixtures.",
    "constrained_schema_validity_rate": "schema-valid fraction under constraints.",
    "unconstrained_schema_validity_rate": "schema-valid fraction without constraints.",
    "constrained_semantic_success_rate": "semantic success fraction under constraints.",
    "unconstrained_semantic_success_rate": "semantic success fraction without constraints.",
    "wrong_valid_count": "schema-valid but semantically wrong outputs.",
    "deterministic_state_evidence_count": (
        "fixtures with initial state, final state, and verifier predicate evidence."
    ),
    "tool_action_reachability_rate": "fraction of required tool/action sequences reached.",
    "latency_or_token_overhead": "measured constraint overhead.",
    "unsafe_false_accepts": "count of invalid/unsafe outputs accepted as valid.",
    "honest_verdict": "one-line result or block reason.",
}

DEFAULT_PANEL_FIXTURES: tuple[JsonDict, ...] = (
    {
        "fixture_id": "inventory_move_one_bolt",
        "initial_state": {"shelf": {"bolt": 2}, "cart": {}},
        "expected_final_state": {"shelf": {"bolt": 1}, "cart": {"bolt": 1}},
        "required_tool_sequence": [
            {
                "tool": "inventory.move",
                "args": {"item": "bolt", "qty": 1, "from": "shelf", "to": "cart"},
            }
        ],
        "verifier_predicate": "shelf.bolt == 1 and cart.bolt == 1",
        "unconstrained": {
            "response": "Moved one bolt to the cart.",
            "latency_s": 0.18,
            "token_count": 8,
            "unsafe_response": False,
        },
        "constrained": {
            "response": {
                "tool_calls": [
                    {
                        "tool": "inventory.move",
                        "args": {"item": "bolt", "qty": 1, "from": "shelf", "to": "cart"},
                    }
                ],
                "final_state": {"shelf": {"bolt": 1}, "cart": {"bolt": 1}},
                "verifier_predicate": "shelf.bolt == 1 and cart.bolt == 1",
                "response_text": "inventory.move transferred exactly one bolt.",
            },
            "latency_s": 0.31,
            "token_count": 30,
            "unsafe_response": False,
        },
    },
    {
        "fixture_id": "drawer_key_unlock",
        "initial_state": {
            "drawer_open": False,
            "agent_inventory": [],
            "door_locked": True,
        },
        "expected_final_state": {
            "drawer_open": True,
            "agent_inventory": ["brass_key"],
            "door_locked": False,
        },
        "required_tool_sequence": [
            {"tool": "drawer.open", "args": {"drawer": "desk"}},
            {"tool": "inventory.add", "args": {"item": "brass_key"}},
            {"tool": "lock.unlock", "args": {"target": "door", "key": "brass_key"}},
        ],
        "verifier_predicate": "drawer_open and brass_key in inventory and door_locked is false",
        "unconstrained": {
            "response": {
                "tool_calls": [
                    {"tool": "drawer.open", "args": {"drawer": "desk"}},
                    {"tool": "inventory.add", "args": {"item": "brass_key"}},
                ],
                "final_state": {
                    "drawer_open": True,
                    "agent_inventory": ["brass_key"],
                    "door_locked": False,
                },
                "verifier_predicate": (
                    "drawer_open and brass_key in inventory and door_locked is false"
                ),
                "response_text": "The door is unlocked after taking the key.",
            },
            "latency_s": 0.22,
            "token_count": 24,
            "unsafe_response": False,
        },
        "constrained": {
            "response": {
                "tool_calls": [
                    {"tool": "drawer.open", "args": {"drawer": "desk"}},
                    {"tool": "inventory.add", "args": {"item": "brass_key"}},
                    {"tool": "lock.unlock", "args": {"target": "door", "key": "brass_key"}},
                ],
                "final_state": {
                    "drawer_open": True,
                    "agent_inventory": ["brass_key"],
                    "door_locked": False,
                },
                "verifier_predicate": (
                    "drawer_open and brass_key in inventory and door_locked is false"
                ),
                "response_text": "drawer.open, inventory.add, and lock.unlock all completed.",
            },
            "latency_s": 0.44,
            "token_count": 45,
            "unsafe_response": False,
        },
    },
    {
        "fixture_id": "cancel_duplicate_job",
        "initial_state": {"queued_jobs": ["job-17", "job-42"], "canceled_jobs": []},
        "expected_final_state": {
            "queued_jobs": ["job-17"],
            "canceled_jobs": ["job-42"],
            "last_cancel_reason": "duplicate",
        },
        "required_tool_sequence": [
            {"tool": "job.cancel", "args": {"job_id": "job-42", "reason": "duplicate"}}
        ],
        "verifier_predicate": "job-42 not in queued_jobs and job-42 in canceled_jobs",
        "unconstrained": {
            "response": "The duplicate job should be canceled.",
            "latency_s": 0.20,
            "token_count": 11,
            "unsafe_response": False,
        },
        "constrained": {
            "response": {
                "tool_calls": [
                    {"tool": "job.cancel", "args": {"job_id": "job-42", "reason": "duplicate"}}
                ],
                "final_state": {
                    "queued_jobs": ["job-17"],
                    "canceled_jobs": ["job-42"],
                    "last_cancel_reason": "duplicate",
                },
                "verifier_predicate": "job-42 not in queued_jobs and job-42 in canceled_jobs",
                "response_text": "job.cancel removed the duplicate queued job.",
            },
            "latency_s": 0.33,
            "token_count": 28,
            "unsafe_response": False,
        },
    },
)


def field_provenance() -> dict[str, JsonDict]:
    """Return the principle annotation required for every Exp 5380 field."""

    return {
        field: {
            "principle": principle,
            "satisfied_by": "Exp 5380 constraint-tax tool/action reachability panel",
        }
        for field, principle in FIELD_PRINCIPLES.items()
    }


def run(
    *,
    root: Path | str = REPO_ROOT,
    artifact_path: Path | str | None = None,
    exp5379_path: Path | str | None = None,
    exp5379_artifact: Mapping[str, Any] | None = None,
    panel_fixtures: Sequence[Mapping[str, Any]] = DEFAULT_PANEL_FIXTURES,
    tests_run: Sequence[Any] | None = None,
    write: bool = True,
) -> JsonDict:
    """Run the Exp 5380 gate and optionally write the terminal result artifact."""

    root_path = Path(root)
    destination = _destination(root_path, artifact_path)
    upstream_path = (
        Path(exp5379_path) if exp5379_path is not None else root_path / exp5379.RESULT_RELATIVE_PATH
    )
    upstream = dict(exp5379_artifact or _read_json(upstream_path))
    upstream_clean = bool(upstream.get("structured_protocol_clean"))
    model_specs = _model_specs_from_upstream(upstream)
    panel = evaluate_panel(panel_fixtures) if upstream_clean else _empty_panel_summary()
    panel_ran = bool(upstream_clean and panel["fixture_count"] > 0)
    ready = bool(
        panel_ran
        and panel["deterministic_state_evidence_count"] == panel["fixture_count"]
        and panel["unsafe_false_accepts"] == 0
    )
    status = "complete" if panel_ran else "blocked"
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "spec_refs": list(SPEC_REFS),
        "random_seed": RANDOM_SEED,
        "status": status,
        "upstream_structured_protocol_clean": upstream_clean,
        "constraint_tax_panel_ready": ready,
        "MODEL_SPECS": model_specs,
        "selected_model_spec": None,
        "inference_substrate": _inference_substrate(upstream=upstream, panel_ran=panel_ran),
        "fixture_count": panel["fixture_count"],
        "constrained_schema_validity_rate": panel["constrained_schema_validity_rate"],
        "unconstrained_schema_validity_rate": panel["unconstrained_schema_validity_rate"],
        "constrained_semantic_success_rate": panel["constrained_semantic_success_rate"],
        "unconstrained_semantic_success_rate": panel["unconstrained_semantic_success_rate"],
        "wrong_valid_count": panel["wrong_valid_count"],
        "deterministic_state_evidence_count": panel["deterministic_state_evidence_count"],
        "tool_action_reachability_rate": panel["tool_action_reachability_rate"],
        "latency_or_token_overhead": panel["latency_or_token_overhead"],
        "unsafe_false_accepts": panel["unsafe_false_accepts"],
        "paired_fixture_results": panel["paired_fixture_results"],
        "constraint_benefit": panel["constraint_benefit"],
        "source_artifacts": _source_artifacts(upstream=upstream, upstream_path=upstream_path),
        "field_provenance": field_provenance(),
        "tests_run": list(tests_run or []),
        "no_autotokenizer_used": True,
    }
    artifact["honest_verdict"] = _honest_verdict(artifact)
    validate_artifact(artifact)
    if write:
        _write_json(destination, artifact)
    return artifact


def evaluate_panel(panel_fixtures: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Evaluate paired fixtures and return aggregate constraint-tax metrics."""

    rows = [_fixture_result(fixture) for fixture in panel_fixtures]
    fixture_count = len(rows)
    constrained = [row["constrained"] for row in rows]
    unconstrained = [row["unconstrained"] for row in rows]
    constrained_schema = _rate(constrained, "schema_valid")
    unconstrained_schema = _rate(unconstrained, "schema_valid")
    constrained_semantic = _rate(constrained, "semantic_success")
    unconstrained_semantic = _rate(unconstrained, "semantic_success")
    constrained_reachability = _rate(constrained, "tool_action_reached")
    evidence_count = sum(1 for row in rows if _has_state_evidence(row))
    wrong_valid_count = sum(
        1
        for row in (*constrained, *unconstrained)
        if row["schema_valid"] and not row["semantic_success"]
    )
    unsafe_false_accepts = sum(
        1 for row in (*constrained, *unconstrained) if row["unsafe_false_accept"]
    )
    return {
        "fixture_count": fixture_count,
        "constrained_schema_validity_rate": constrained_schema,
        "unconstrained_schema_validity_rate": unconstrained_schema,
        "constrained_semantic_success_rate": constrained_semantic,
        "unconstrained_semantic_success_rate": unconstrained_semantic,
        "wrong_valid_count": wrong_valid_count,
        "deterministic_state_evidence_count": evidence_count,
        "tool_action_reachability_rate": constrained_reachability,
        "latency_or_token_overhead": _latency_or_token_overhead(rows),
        "unsafe_false_accepts": unsafe_false_accepts,
        "paired_fixture_results": rows,
        "constraint_benefit": {
            "schema_validity_delta": round(constrained_schema - unconstrained_schema, 6),
            "semantic_success_delta": round(constrained_semantic - unconstrained_semantic, 6),
            "tool_action_reachability_delta": round(
                constrained_reachability - _rate(unconstrained, "tool_action_reached"), 6
            ),
        },
    }


def artifact_schema_errors(artifact: Mapping[str, Any]) -> list[str]:
    """Return contract errors that would make the Exp 5380 artifact unusable."""

    missing = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact]
    status = artifact.get("status")
    provenance = artifact.get("field_provenance")
    checks = (
        (bool(missing), f"missing required fields: {missing}"),
        (status not in {"complete", "blocked"}, "status must be complete or blocked"),
        (
            not isinstance(artifact.get("upstream_structured_protocol_clean"), bool),
            "upstream_structured_protocol_clean must be boolean",
        ),
        (
            not isinstance(artifact.get("constraint_tax_panel_ready"), bool),
            "constraint_tax_panel_ready must be boolean",
        ),
        (
            not _model_specs_cover_mandated(artifact.get("MODEL_SPECS")),
            "MODEL_SPECS must cover mandated GGUF ids",
        ),
        (
            artifact.get("selected_model_spec") is not None
            and not isinstance(artifact.get("selected_model_spec"), Mapping),
            "selected_model_spec must be null or object",
        ),
        (
            not isinstance(artifact.get("inference_substrate"), Mapping),
            "inference_substrate must be object",
        ),
        (
            not _non_negative_int(artifact.get("fixture_count")),
            "fixture_count must be non-negative integer",
        ),
        (
            not all(
                _rate_is_valid(artifact.get(field))
                for field in (
                    "constrained_schema_validity_rate",
                    "unconstrained_schema_validity_rate",
                    "constrained_semantic_success_rate",
                    "unconstrained_semantic_success_rate",
                    "tool_action_reachability_rate",
                )
            ),
            "rate fields must be in [0, 1]",
        ),
        (
            not all(
                _non_negative_int(artifact.get(field))
                for field in (
                    "wrong_valid_count",
                    "deterministic_state_evidence_count",
                    "unsafe_false_accepts",
                )
            ),
            "count fields must be non-negative integers",
        ),
        (
            not isinstance(artifact.get("latency_or_token_overhead"), Mapping),
            "latency_or_token_overhead must be object",
        ),
        (
            artifact.get("constraint_tax_panel_ready") is True
            and artifact.get("unsafe_false_accepts") != 0,
            "constraint_tax_panel_ready requires unsafe_false_accepts=0",
        ),
        (
            artifact.get("constraint_tax_panel_ready") is True
            and int(artifact.get("deterministic_state_evidence_count") or 0) <= 0,
            "constraint_tax_panel_ready requires deterministic state evidence",
        ),
        (
            status == "complete"
            and (
                artifact.get("upstream_structured_protocol_clean") is not True
                or int(artifact.get("fixture_count") or 0) <= 0
            ),
            "complete status requires upstream clean gate and panel fixtures",
        ),
        (
            status == "blocked" and artifact.get("constraint_tax_panel_ready") is True,
            "blocked status cannot be panel-ready",
        ),
        (
            not isinstance(artifact.get("honest_verdict"), str)
            or not artifact.get("honest_verdict", "").startswith(TERMINAL_PREFIXES),
            "honest_verdict must start with complete: or blocked_",
        ),
        (
            not isinstance(provenance, Mapping)
            or any(field not in provenance for field in REQUIRED_ARTIFACT_FIELDS),
            "field_provenance must cover required fields",
        ),
    )
    return [message for failed, message in checks if failed]


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """Raise when an Exp 5380 artifact cannot support downstream gating."""

    errors = artifact_schema_errors(artifact)
    if errors:
        raise ValueError(f"Exp 5380 artifact validation failed: {'; '.join(errors)}")


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point for producing the Exp 5380 result artifact."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=REPO_ROOT)
    parser.add_argument("--artifact-path", type=Path, default=None)
    parser.add_argument("--exp5379-path", type=Path, default=None)
    parser.add_argument("--no-write", action="store_true")
    args = parser.parse_args(argv)
    artifact = run(
        root=args.root,
        artifact_path=args.artifact_path,
        exp5379_path=args.exp5379_path,
        write=not args.no_write,
    )
    print(json.dumps(artifact, indent=2, sort_keys=True, ensure_ascii=True))
    return 0 if artifact["status"] == "complete" else 1


def _fixture_result(fixture: Mapping[str, Any]) -> JsonDict:
    initial = copy.deepcopy(fixture["initial_state"])
    expected = copy.deepcopy(fixture["expected_final_state"])
    required = copy.deepcopy(fixture["required_tool_sequence"])
    return {
        "fixture_id": fixture["fixture_id"],
        "initial_state": initial,
        "expected_final_state": expected,
        "required_tool_sequence": required,
        "verifier_predicate": fixture["verifier_predicate"],
        "unconstrained": _arm_result(fixture, "unconstrained"),
        "constrained": _arm_result(fixture, "constrained"),
    }


def _arm_result(fixture: Mapping[str, Any], arm: str) -> JsonDict:
    candidate = fixture[arm]
    response = candidate["response"]
    schema_valid = _schema_valid(response)
    tool_calls = _tool_trace(response) if schema_valid else []
    executed_final_state = (
        _apply_tool_sequence(fixture["initial_state"], tool_calls) if schema_valid else None
    )
    final_state = copy.deepcopy(response["final_state"]) if schema_valid else None
    tool_action_reached = schema_valid and _tool_sequence_reached(
        tool_calls, fixture["required_tool_sequence"]
    )
    state_verified = bool(
        schema_valid
        and final_state == fixture["expected_final_state"]
        and executed_final_state == fixture["expected_final_state"]
    )
    semantic_success = bool(tool_action_reached and state_verified)
    unsafe_false_accept = bool(candidate.get("unsafe_response") and semantic_success)
    return {
        "schema_valid": schema_valid,
        "semantic_success": semantic_success,
        "wrong_valid": bool(schema_valid and not semantic_success),
        "tool_action_reached": tool_action_reached,
        "state_verified": state_verified,
        "unsafe_false_accept": unsafe_false_accept,
        "tool_call_trace": tool_calls,
        "executed_final_state": executed_final_state,
        "final_state": final_state,
        "verifier_predicate": fixture["verifier_predicate"],
        "response_text_fallback": _response_text_fallback(response, schema_valid),
        "latency_s": float(candidate.get("latency_s", 0.0)),
        "token_count": int(candidate.get("token_count", 0)),
    }


def _apply_tool_sequence(
    initial_state: Mapping[str, Any], tool_calls: Sequence[Mapping[str, Any]]
) -> JsonDict:
    state = copy.deepcopy(initial_state)
    for call in tool_calls:
        tool = call["tool"]
        args = call.get("args", {})
        if tool == "inventory.move":
            item = args["item"]
            qty = int(args["qty"])
            source = args["from"]
            destination = args["to"]
            state[source][item] -= qty
            state.setdefault(destination, {})
            state[destination][item] = state[destination].get(item, 0) + qty
        elif tool == "drawer.open":
            state["drawer_open"] = True
        elif tool == "inventory.add":
            inventory = state.setdefault("agent_inventory", [])
            if args["item"] not in inventory:
                inventory.append(args["item"])
        elif tool == "lock.unlock":
            if args["key"] in state.get("agent_inventory", []):
                state["door_locked"] = False
        elif tool == "job.cancel":
            job_id = args["job_id"]
            if job_id in state.get("queued_jobs", []):
                state["queued_jobs"].remove(job_id)
            state.setdefault("canceled_jobs", []).append(job_id)
            state["last_cancel_reason"] = args["reason"]
        else:
            raise ValueError(f"unknown fixture tool: {tool}")  # pragma: no cover - fixture guard
    return state


def _schema_valid(response: Any) -> bool:
    return bool(
        isinstance(response, Mapping)
        and isinstance(response.get("tool_calls"), list)
        and isinstance(response.get("final_state"), Mapping)
        and isinstance(response.get("verifier_predicate"), str)
        and isinstance(response.get("response_text"), str)
    )


def _tool_trace(response: Mapping[str, Any]) -> list[JsonDict]:
    return [dict(call) for call in response.get("tool_calls", []) if isinstance(call, Mapping)]


def _tool_sequence_reached(
    actual: Sequence[Mapping[str, Any]], required: Sequence[Mapping[str, Any]]
) -> bool:
    return [dict(call) for call in actual] == [dict(call) for call in required]


def _response_text_fallback(response: Any, schema_valid: bool) -> JsonDict:
    text = response if isinstance(response, str) else ""
    attempted = bool(not schema_valid and text)
    return {
        "attempted": attempted,
        "text": text if attempted else "",
        "accepted_as_semantic_evidence": False,
    }


def _latency_or_token_overhead(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    constrained_latency = _average(row["constrained"]["latency_s"] for row in rows)
    unconstrained_latency = _average(row["unconstrained"]["latency_s"] for row in rows)
    constrained_tokens = _average(row["constrained"]["token_count"] for row in rows)
    unconstrained_tokens = _average(row["unconstrained"]["token_count"] for row in rows)
    return {
        "measurement_basis": "deterministic_fixture_receipts",
        "constrained_avg_latency_s": round(constrained_latency, 6),
        "unconstrained_avg_latency_s": round(unconstrained_latency, 6),
        "latency_s_delta": round(constrained_latency - unconstrained_latency, 6),
        "latency_ratio": _ratio(constrained_latency, unconstrained_latency),
        "constrained_avg_tokens": round(constrained_tokens, 6),
        "unconstrained_avg_tokens": round(unconstrained_tokens, 6),
        "token_delta": round(constrained_tokens - unconstrained_tokens, 6),
        "token_ratio": _ratio(constrained_tokens, unconstrained_tokens),
    }


def _has_state_evidence(row: Mapping[str, Any]) -> bool:
    return bool(
        row.get("initial_state")
        and row.get("expected_final_state")
        and row.get("verifier_predicate")
        and row.get("constrained", {}).get("final_state") == row.get("expected_final_state")
        and row.get("constrained", {}).get("tool_call_trace")
    )


def _empty_panel_summary() -> JsonDict:
    return {
        "fixture_count": 0,
        "constrained_schema_validity_rate": 0.0,
        "unconstrained_schema_validity_rate": 0.0,
        "constrained_semantic_success_rate": 0.0,
        "unconstrained_semantic_success_rate": 0.0,
        "wrong_valid_count": 0,
        "deterministic_state_evidence_count": 0,
        "tool_action_reachability_rate": 0.0,
        "latency_or_token_overhead": {
            "measurement_basis": "not_run_upstream_gate_blocked",
            "constrained_avg_latency_s": 0.0,
            "unconstrained_avg_latency_s": 0.0,
            "latency_s_delta": 0.0,
            "latency_ratio": 0.0,
            "constrained_avg_tokens": 0.0,
            "unconstrained_avg_tokens": 0.0,
            "token_delta": 0.0,
            "token_ratio": 0.0,
        },
        "unsafe_false_accepts": 0,
        "paired_fixture_results": [],
        "constraint_benefit": {
            "schema_validity_delta": 0.0,
            "semantic_success_delta": 0.0,
            "tool_action_reachability_delta": 0.0,
        },
    }


def _inference_substrate(*, upstream: Mapping[str, Any], panel_ran: bool) -> JsonDict:
    return {
        "kind": "deterministic_fixture_replay" if panel_ran else "blocked_upstream_gate",
        "live_llm_calls_ran": False,
        "explanation": (
            "No new live LLM generation ran; Exp5380 replays deterministic paired "
            "state/tool fixtures after checking the Exp5379 clean structured gate."
        ),
        "allowed_loader_family": "llama.cpp/GGUF only; no AutoTokenizer or AutoModel on -GGUF repositories.",
        "upstream_inference_substrate": dict(upstream.get("inference_substrate", {}))
        if isinstance(upstream.get("inference_substrate"), Mapping)
        else {},
    }


def _source_artifacts(*, upstream: Mapping[str, Any], upstream_path: Path) -> list[JsonDict]:
    return [
        {
            "path": upstream_path.as_posix(),
            "experiment_id": upstream.get("experiment_id"),
            "structured_protocol_clean": bool(upstream.get("structured_protocol_clean")),
            "used_as_gate": True,
        }
    ]


def _honest_verdict(artifact: Mapping[str, Any]) -> str:
    if artifact.get("upstream_structured_protocol_clean") is not True:
        return "blocked_exp5379_structured_protocol_clean_false"
    if int(artifact.get("fixture_count") or 0) <= 0:
        return "blocked_no_paired_fixtures"
    if artifact.get("constraint_tax_panel_ready") is True:
        return "complete: constraint-tax panel ready with deterministic state/tool evidence"
    return (
        "complete: panel ran but constraint_tax_panel_ready=false because "
        f"unsafe_false_accepts={artifact.get('unsafe_false_accepts')}"
    )


def _model_specs_from_upstream(upstream: Mapping[str, Any]) -> list[JsonDict]:
    value = upstream.get("MODEL_SPECS")
    if _model_specs_cover_mandated(value):
        return [dict(row) for row in value]  # type: ignore[union-attr]
    return [
        {
            "hf_id": hf_id,
            "status": "missing_from_exp5379_artifact",
            "gguf_loader_family": "llama.cpp",
            "autotokenizer_used": False,
        }
        for hf_id in MANDATED_HF_IDS
    ]


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


def _rate_is_valid(value: Any) -> bool:
    return (
        isinstance(value, int | float)
        and not isinstance(value, bool)
        and 0.0 <= float(value) <= 1.0
    )


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
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":  # pragma: no cover - direct CLI execution
    raise SystemExit(main())
