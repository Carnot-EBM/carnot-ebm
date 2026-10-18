"""Audit tool-gap causal support from frozen ARC transcript receipts.

Spec refs: REQ-ARC-6845 and SCENARIO-ARC-6845-*.

This reducer reads existing files only. It separates tool request transport
from later utility, because a response reaching the agent is not proof that the
agent used it or made progress from it.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
import hashlib
import json
import os
from pathlib import Path
import statistics
import sys
import time
from typing import Any

from carnot.agentic.arc_solve_artifact_discipline import (
    DETERMINISTIC_CPU_LIVE_RECEIPT_AUDIT_NO_LLM_SUBSTRATE,
)


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[2]
SPEC_PATH = Path("openspec/capabilities/arc-agi/spec.md")
MODULE_PATH = Path("python/carnot/experiment_6845_tool_gap_causal_support_audit.py")
# REQ-ARC-WMTE-6642: the eval-run fields this module requires. Checked by
# scripts/eval_run_consumer_field_lint.py against real artifacts + producer source.
EVAL_RUN_FIELDS_READ = ("honest_verdict",)
WRAPPER_PATH = Path("scripts/experiments/experiment_6845_tool_gap_causal_support_audit.py")
OUTPUT_PATH = Path("results/experiment_6845_tool_gap_causal_support_audit.json")
SCHEMA = "carnot.experiment_6845.tool_gap_causal_support_audit.v1"
INFERENCE_SUBSTRATE = DETERMINISTIC_CPU_LIVE_RECEIPT_AUDIT_NO_LLM_SUBSTRATE
VERDICT_CLASSES = {"positive", "circular_positive", "null", "blocked", "disqualified", "partial"}
EXPECTED_BATCH_CELLS: tuple[tuple[str, str], ...] = (
    ("r11l", "off_or_unobserved"),
    ("lp85", "off_or_unobserved"),
    ("ls20", "off_or_unobserved"),
    ("wa30", "off_or_unobserved"),
    ("sp80", "selfparse"),
    ("su15", "selfparse"),
    ("tu93", "selfparse"),
    ("cn04", "selfparse"),
    ("m0r0", "selfparse"),
    ("sk48", "selfparse"),
)

REQUIRED_ARTIFACT_FIELDS = (
    "field_principles",
    "preconditions_checked",
    "inference_substrate",
    "duration_s",
    "source_artifact_hashes",
    "reproducibility_checksum",
    "per_game_results",
    "configuration_strata",
    "obligation_ledger",
    "request_receipt_joins",
    "agent_visibility_results",
    "next_action_results",
    "later_outcome_results",
    "transport_results",
    "utility_results",
    "headroom_results",
    "unmatched_cell_results",
    "tool_gap_audit_complete_score",
    "tool_gap_effect_eligible_score",
    "solve_claim",
    "gate_check_summary",
    "verifier_is_oracle",
    "verdict_class",
    "honest_verdict",
)
TOP_LEVEL_FIELDS = ("schema", "experiment_id", "run_date", "status", *REQUIRED_ARTIFACT_FIELDS)

FIELD_PRINCIPLES: JsonDict = {
    "schema": "The schema lets later reducers reject incompatible audit files.",
    "experiment_id": "The identifier binds this artifact to REQ-ARC-6845.",
    "run_date": "The date states when the frozen transcript audit ran.",
    "status": "The status separates effect-eligible audits from blocked audits.",
    "field_principles": "Each top-level field states why an auditor needs it.",
    "preconditions_checked": "The gates show which evidence was required before eligibility.",
    "inference_substrate": "This is deterministic CPU transcript audit work, not a live run.",
    "duration_s": "Wall time exposes skipped or fabricated reducer execution.",
    "source_artifact_hashes": "Source hashes bind the frozen artifacts used as inputs.",
    "reproducibility_checksum": "One digest binds the audit content except wall time.",
    "per_game_results": "One obligation row keeps the request, response, action, and outcome local.",
    "configuration_strata": "Strata prevent pooling across unlike games or configurations.",
    "obligation_ledger": "The fresh ledger lists every reduced tool-gap obligation.",
    "request_receipt_joins": "Join counts show whether request and response identities exist.",
    "agent_visibility_results": "Visibility rates show whether responses reached the agent text.",
    "next_action_results": "Next-action rates show whether the agent acted after the response.",
    "later_outcome_results": "Outcome rates show whether exact later progress was available.",
    "transport_results": "Transport metrics stop receipt success from becoming utility.",
    "utility_results": "Utility metrics measure use, changed action, progress, and invalid actions.",
    "headroom_results": "Headroom shows whether rows could distinguish useful actions.",
    "unmatched_cell_results": "Unmatched, loop-off, and no-obligation cells stay visible.",
    "tool_gap_audit_complete_score": "Completeness is about audit shape, not effect size.",
    "tool_gap_effect_eligible_score": "Eligibility is 1 only when joins, timing, matching, outcomes, and headroom pass.",
    "solve_claim": "False prevents an audit artifact from becoming an ARC solve claim.",
    "gate_check_summary": "A blocked verdict names the first failed check and observed value.",
    "verifier_is_oracle": "False because exact later outcomes are external receipts.",
    "verdict_class": "A closed verdict class prevents unsupported positives.",
    "honest_verdict": "The complete_ prefix states the terminal evidence boundary.",
}


def canonical_json(value: Any) -> str:
    """Encode JSON in one stable form for row and artifact hashes."""

    return json.dumps(value, ensure_ascii=True, separators=(",", ":"), sort_keys=True)


def sha256_bytes(value: bytes) -> str:
    return "sha256:" + hashlib.sha256(value).hexdigest()


def sha256_json(value: Any) -> str:
    return sha256_bytes(canonical_json(value).encode("utf-8"))


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and value.startswith("sha256:") and len(value) > 20


def _read_bytes(path: Path) -> tuple[bytes, str | None]:
    try:
        return path.read_bytes(), None
    except OSError as exc:
        return b"", type(exc).__name__


def _load_json(raw: bytes) -> JsonDict:
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _relative(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path)


def _common_root(source_paths: Mapping[str, Path] | None) -> Path:
    if not source_paths:
        return REPO_ROOT
    existing = [path.resolve() for path in source_paths.values() if path.exists()]
    if not existing:
        return REPO_ROOT
    return Path(os.path.commonpath([str(path) for path in existing]))


def collect_default_source_paths(root: Path = REPO_ROOT) -> dict[str, Path]:
    """Collect bounded source files and terminal rows named by Exp6843."""

    paths: dict[str, Path] = {
        "agents_md": root / "AGENTS.md",
        "claude_md": root / "CLAUDE.md",
        "codex_md": root / "CODEX.md",
        "north_star": root / "ops/north-star.md",
        "ops_status": root / "ops/status.md",
        "spec": root / SPEC_PATH,
        "module_source": root / MODULE_PATH,
        "wrapper_source": root / WRAPPER_PATH,
        "agent_runtime_missing": root / "python/carnot/agent_runtime.py",
        "experiment_6473": root / "results/experiment_6473_tool_loop_compaction_pilot_ab.json",
        "experiment_6777": root / "results/experiment_6777_arc_tool_gap_transport.json",
        "experiment_6820_missing": root
        / "results/experiment_6820_arc_tool_gap_obligation_transport_v2.json",
        "experiment_6843": root / "results/experiment_6843_live_arc_evidence_stratum_freeze.json",
    }
    exp6843_path = paths["experiment_6843"]
    raw, error = _read_bytes(exp6843_path)
    payload = _load_json(raw) if error is None else {}
    for row in payload.get("rows", []) if isinstance(payload.get("rows"), list) else []:
        if isinstance(row, Mapping) and isinstance(row.get("source_path"), str):
            source_path = root / row["source_path"]
            if source_path.suffix == ".json":
                paths[f"leaderboard:{source_path.name}"] = source_path
    run_dir = root / "results/arc_leaderboard_eval_runs"
    if not any(key.startswith("leaderboard:") for key in paths) and run_dir.exists():
        for path in sorted(run_dir.glob("*.json")):
            paths[f"leaderboard:{path.name}"] = path
    return paths


def _source_record(name: str, path: Path, root: Path) -> JsonDict:
    raw, error = _read_bytes(path)
    payload = _load_json(raw) if error is None else {}
    return {
        "name": name,
        "path": _relative(path, root),
        "exists": error is None,
        "read_error": error,
        "file_sha256": sha256_bytes(raw) if error is None else None,
        "size_bytes": len(raw) if error is None else None,
        "status": payload.get("status"),
        "honest_verdict": payload.get("honest_verdict"),
        "verdict_class": payload.get("verdict_class"),
        "top_level_keys": sorted(payload) if payload else [],
        "_payload": payload,
        "_raw_text": raw.decode("utf-8", errors="replace") if error is None else "",
    }


def _source_records(source_paths: Mapping[str, Path], root: Path) -> dict[str, JsonDict]:
    return {name: _source_record(name, path, root) for name, path in sorted(source_paths.items())}


def source_artifact_hashes(records: Mapping[str, Mapping[str, Any]]) -> dict[str, JsonDict]:
    return {
        name: {key: value for key, value in record.items() if not key.startswith("_")}
        for name, record in records.items()
    }


def _payload(records: Mapping[str, Mapping[str, Any]], name: str) -> JsonDict:
    value = records.get(name, {}).get("_payload", {})
    return value if isinstance(value, dict) else {}


def _finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _mean(values: Sequence[float]) -> float | None:
    if not values:
        return None
    return float(statistics.fmean(values))


def _rate_interval(count: int, total: int) -> JsonDict:
    if total <= 0:
        return {"lower": None, "upper": None, "method": "normal_approximation_95"}
    rate = count / total
    spread = 1.96 * ((rate * (1 - rate) / total) ** 0.5)
    return {
        "lower": max(0.0, rate - spread),
        "upper": min(1.0, rate + spread),
        "method": "normal_approximation_95",
    }


def _check(check: str, expected: Any, observed: Any, passed: bool | None = None) -> JsonDict:
    return {
        "check": check,
        "expected": expected,
        "observed": observed,
        "passed": observed == expected if passed is None else bool(passed),
    }


def _gate_summary(checks: Sequence[Mapping[str, Any]]) -> JsonDict:
    failed = next((row for row in checks if row.get("passed") is not True), None)
    if failed is None:
        return {
            "passed": True,
            "failed_check": None,
            "expected": "all gates pass",
            "observed": "all gates pass",
        }
    return {
        "passed": False,
        "failed_check": failed.get("check"),
        "expected": failed.get("expected"),
        "observed": failed.get("observed"),
    }


def _inventory_cells(records: Mapping[str, Mapping[str, Any]]) -> list[JsonDict]:
    inventory = _payload(records, "experiment_6843")
    cells: list[JsonDict] = []
    for row in inventory.get("rows", []) if isinstance(inventory.get("rows"), list) else []:
        if not isinstance(row, Mapping):
            continue
        cells.append(
            {
                "source_path": row.get("source_path"),
                "source_artifact_sha256": row.get("source_artifact_sha256"),
                "run": row.get("run_id"),
                "game": row.get("game"),
                "model": row.get("model_id") or "unknown",
                "policy": row.get("policy") or "unknown",
                "budget": row.get("budget"),
                "tool_loop_state": row.get("tool_loop_state") or "unknown",
                "supervisor_state": row.get("supervisor_state") or "unknown",
                "tool_gap_receipt_complete": row.get("tool_gap_receipt_complete") is True,
                "tool_gap_receipt_count": row.get("tool_gap_receipt_count") or 0,
                "tool_gap_calls_total": row.get("tool_gap_calls_total") or 0,
                "tool_gap_event_count": row.get("tool_gap_event_count") or 0,
                "inventory_stratum_identity": row.get("stratum_identity"),
                "inventory_row_sha256": row.get("row_sha256"),
            }
        )
    return cells


def _records_by_path(records: Mapping[str, Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
    return {str(record.get("path")): record for record in records.values() if record.get("path")}


def _game_rows(payload: Mapping[str, Any], game: Any) -> list[Mapping[str, Any]]:
    rows = payload.get("per_game", [])
    if not isinstance(rows, list):
        return []
    return [row for row in rows if isinstance(row, Mapping) and row.get("game") == game]


def _tool_gap_stats(game_row: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    diagnostics = game_row.get("policy_diagnostics", {})
    attempts = diagnostics.get("induction_attempts", []) if isinstance(diagnostics, Mapping) else []
    stats: list[Mapping[str, Any]] = []
    for attempt in attempts if isinstance(attempts, list) else []:
        if isinstance(attempt, Mapping) and isinstance(attempt.get("tool_gap"), Mapping):
            stats.append(attempt["tool_gap"])
    return stats


def _event_requested_tool(event: Mapping[str, Any]) -> str:
    if event.get("kind") == "bad_arguments":
        return str(event.get("tool") or event.get("requested_tool") or "unknown")
    return str(event.get("requested_tool") or event.get("tool") or "unknown")


def _event_missing_fact(event: Mapping[str, Any]) -> str:
    if isinstance(event.get("missing_fact"), str) and event.get("missing_fact"):
        return str(event["missing_fact"])
    if event.get("kind") == "bad_arguments":
        return f"signature mismatch for tool {_event_requested_tool(event)}"
    return f"tool unavailable: {_event_requested_tool(event)}"


def _outcome_score(outcome: Mapping[str, Any]) -> float:
    reward = outcome.get("reward")
    if isinstance(reward, Mapping) and reward.get("present") is True:
        value = reward.get("value")
        if _finite_number(value):
            return float(value)
    before = outcome.get("levels_completed_before", 0)
    after = outcome.get("levels_completed_after", 0)
    level_delta = (
        float(after) - float(before) if _finite_number(before) and _finite_number(after) else 0.0
    )
    termination = outcome.get("termination")
    state = str(termination.get("state", "") if isinstance(termination, Mapping) else "").upper()
    if state == "GAME_OVER" and level_delta <= 0:
        return -1.0
    return level_delta


def _direction(outcome: Mapping[str, Any]) -> str:
    score = _outcome_score(outcome)
    if score > 0:
        return "progress"
    if score < 0 or outcome.get("error") is not None or outcome.get("outcome_status") != "returned":
        return "regression"
    return "abstention"


def _headroom_nonzero(outcome: Mapping[str, Any]) -> bool:
    headroom = outcome.get("headroom")
    if not isinstance(headroom, Mapping):
        return False
    if headroom.get("nonzero_headroom") is True:
        return True
    baseline = headroom.get("baseline_score")
    best = headroom.get("best_available_score")
    return _finite_number(baseline) and _finite_number(best) and float(best) != float(baseline)


def _action_validity(row: Mapping[str, Any]) -> JsonDict:
    receipt = row.get("next_action_receipt")
    if isinstance(receipt, Mapping) and isinstance(receipt.get("proposal_validity"), Mapping):
        validity = dict(receipt["proposal_validity"])
        if "valid" in validity:
            return validity
    action = row.get("next_action")
    if not isinstance(action, Mapping):
        return {"valid": False, "reason": "missing_next_action", "available_action_ids": []}
    kind = action.get("kind")
    available = receipt.get("available_actions", []) if isinstance(receipt, Mapping) else []
    available_list = list(available) if isinstance(available, list) else []
    if kind == "RESET":
        return {"valid": True, "reason": "reset_control", "available_action_ids": available_list}
    if available_list and kind not in available_list:
        return {
            "valid": False,
            "reason": "kind_not_in_available_actions",
            "action_kind": kind,
            "available_action_ids": available_list,
        }
    data = action.get("data")
    if kind == 6 and not (
        isinstance(data, Mapping)
        and isinstance(data.get("x"), int)
        and isinstance(data.get("y"), int)
    ):
        return {
            "valid": False,
            "reason": "action6_requires_integer_xy",
            "action_kind": kind,
            "available_action_ids": available_list,
        }
    return {
        "valid": True,
        "reason": "kind_available_and_payload_well_formed",
        "action_kind": kind,
        "available_action_ids": available_list,
    }


def _event_sequence_ok(event: Mapping[str, Any]) -> bool:
    keys = (
        "request_sequence",
        "receipt_sequence",
        "visibility_sequence",
        "next_action_sequence",
        "outcome_sequence",
    )
    values = [event.get(key) for key in keys]
    if not all(_finite_number(value) for value in values):
        return False
    return all(float(left) < float(right) for left, right in zip(values, values[1:]))


def _stratum_identity(row: Mapping[str, Any]) -> str:
    return "|".join(
        str(row.get(key))
        for key in (
            "game",
            "run",
            "model",
            "policy",
            "budget",
            "tool_loop_state",
            "supervisor_state",
            "requested_tool",
        )
    )


def _row_with_hash(row: JsonDict) -> JsonDict:
    material = {key: value for key, value in row.items() if key != "row_sha256"}
    return {**row, "row_sha256": sha256_json(material)}


def _reduce_event(
    *,
    cell: Mapping[str, Any],
    source_record: Mapping[str, Any],
    event: Mapping[str, Any],
    event_index: int,
) -> JsonDict:
    requested_tool = _event_requested_tool(event)
    receipt = event.get("next_action_receipt")
    later = event.get("later_exact_outcome")
    receipt = receipt if isinstance(receipt, Mapping) else {}
    later = later if isinstance(later, Mapping) else {}
    action_row = {"next_action": event.get("next_action"), "next_action_receipt": receipt}
    validity = _action_validity(action_row)
    request_joined = (
        _is_sha256(event.get("request_id"))
        and _is_sha256(event.get("tool_receipt_id"))
        and _is_sha256(event.get("response_id"))
        and isinstance(event.get("actual_call"), Mapping)
        and isinstance(event.get("exact_response"), Mapping)
        and str(event.get("actual_call", {}).get("name")) == requested_tool
    )
    agent_visible = _is_sha256(event.get("agent_visible_response_id")) and bool(
        str(event.get("agent_visible_text") or "").strip()
    )
    next_action_receipted = (
        bool(receipt)
        and _is_sha256(receipt.get("next_action_id") or event.get("next_action_id"))
        and isinstance(event.get("next_action"), Mapping)
    )
    exact_outcome = (
        _is_sha256(later.get("outcome_id"))
        and later.get("outcome_status") == "returned"
        and later.get("live_return") is True
        and later.get("fully_joined") is True
    )
    reduced: JsonDict = {
        "row_kind": "eligible_tool_gap_obligation",
        "source_artifact": source_record.get("name"),
        "source_path": source_record.get("path"),
        "source_artifact_sha256": source_record.get("file_sha256"),
        "game": cell.get("game"),
        "run": cell.get("run"),
        "model": cell.get("model"),
        "policy": cell.get("policy"),
        "budget": cell.get("budget"),
        "tool_loop_state": cell.get("tool_loop_state"),
        "supervisor_state": cell.get("supervisor_state"),
        "event_index": event_index,
        "missing_fact": _event_missing_fact(event),
        "requested_tool": requested_tool,
        "actual_call": event.get("actual_call"),
        "exact_response": event.get("exact_response"),
        "agent_visible_text": event.get("agent_visible_text"),
        "next_action": event.get("next_action"),
        "next_action_receipt": dict(receipt),
        "later_exact_outcome": {
            **dict(later),
            "level_delta": _outcome_score(later),
            "direction": _direction(later),
        },
        "request": {
            "request_id": event.get("request_id"),
            "turn": event.get("turn"),
            "request_sequence": event.get("request_sequence"),
        },
        "receipt": {
            "tool_receipt_id": event.get("tool_receipt_id"),
            "response_id": event.get("response_id"),
            "receipt_sequence": event.get("receipt_sequence"),
        },
        "agent_visible_response_id": event.get("agent_visible_response_id"),
        "raw_transcript_sha256": event.get("raw_transcript_sha256"),
        "next_action_id": event.get("next_action_id") or receipt.get("next_action_id"),
        "request_receipt_joined": request_joined,
        "agent_visible": agent_visible,
        "next_action_receipted": next_action_receipted,
        "exact_later_outcome_joined": exact_outcome,
        "temporal_order_verified": _event_sequence_ok(event),
        "transport_success": bool(request_joined and agent_visible),
        "response_used": receipt.get("used_tool_response") is True,
        "action_changed": receipt.get("changed_after_response") is True,
        "proposal_validity": validity,
        "invalid_action": validity.get("valid") is not True,
        "transition_progress": 1 if _direction(later) == "progress" else 0,
        "regression": _direction(later) == "regression",
        "abstention": _direction(later) == "abstention",
        "headroom": {
            "nonzero_headroom": _headroom_nonzero(later),
            "source": "later_exact_outcome.headroom",
        },
        "matched_cell": True,
        "inventory_stratum_identity": cell.get("inventory_stratum_identity"),
    }
    reduced["matched_stratum"] = _stratum_identity(reduced)
    reduced["obligation_identity"] = sha256_json(
        {
            "request_id": event.get("request_id"),
            "tool_receipt_id": event.get("tool_receipt_id"),
            "response_id": event.get("response_id"),
            "visible_id": event.get("agent_visible_response_id"),
            "next_action_id": reduced["next_action_id"],
            "outcome_id": later.get("outcome_id"),
        }
    )
    return _row_with_hash(reduced)


def reduce_obligation_rows(
    records: Mapping[str, Mapping[str, Any]],
    cells: Sequence[Mapping[str, Any]],
) -> tuple[list[JsonDict], list[JsonDict]]:
    """Reduce terminal tool-gap cells into obligation rows and unmatched cells."""

    by_path = _records_by_path(records)
    rows: list[JsonDict] = []
    unmatched: list[JsonDict] = []
    for cell in cells:
        if cell.get("tool_gap_receipt_complete") is not True:
            unmatched.append(
                {
                    "source_artifact": "experiment_6843",
                    "matched_stratum": cell.get("inventory_stratum_identity"),
                    "game": cell.get("game"),
                    "reason": "tool_gap_receipt_missing_or_loop_off",
                    "row_count": 1,
                }
            )
            continue
        source_path = str(cell.get("source_path"))
        source_record = by_path.get(source_path)
        if source_record is None or source_record.get("exists") is not True:
            unmatched.append(
                {
                    "source_artifact": "experiment_6843",
                    "matched_stratum": cell.get("inventory_stratum_identity"),
                    "game": cell.get("game"),
                    "reason": "source_artifact_missing",
                    "row_count": 1,
                    "observed": source_path,
                }
            )
            continue
        event_count = 0
        for game_row in _game_rows(source_record.get("_payload", {}), cell.get("game")):
            for stats in _tool_gap_stats(game_row):
                events = stats.get("tool_gap_events", [])
                if not isinstance(events, list):
                    continue
                for event in events:
                    if isinstance(event, Mapping):
                        event_count += 1
                        rows.append(
                            _reduce_event(
                                cell=cell,
                                source_record=source_record,
                                event=event,
                                event_index=event_count,
                            )
                        )
        if event_count == 0:
            unmatched.append(
                {
                    "source_artifact": str(source_record.get("name")),
                    "matched_stratum": cell.get("inventory_stratum_identity"),
                    "game": cell.get("game"),
                    "reason": "no_tool_gap_obligation_events",
                    "row_count": 1,
                    "observed": {
                        "tool_gap_calls_total": cell.get("tool_gap_calls_total"),
                        "tool_gap_event_count": cell.get("tool_gap_event_count"),
                    },
                }
            )
    return sorted(rows, key=lambda row: (row["matched_stratum"], row["event_index"])), unmatched


def _group_by_stratum(rows: Sequence[Mapping[str, Any]]) -> dict[str, list[Mapping[str, Any]]]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["matched_stratum"])].append(row)
    return dict(grouped)


def _cell_stratum(cell: Mapping[str, Any]) -> str:
    return "|".join(
        str(cell.get(key))
        for key in (
            "game",
            "run",
            "model",
            "policy",
            "budget",
            "tool_loop_state",
            "supervisor_state",
        )
    )


def configuration_strata(
    rows: Sequence[Mapping[str, Any]],
    cells: Sequence[Mapping[str, Any]],
    expected_missing: Sequence[Mapping[str, Any]] = (),
) -> JsonDict:
    groups = _group_by_stratum(rows)
    strata: dict[str, JsonDict] = {}
    for cell in cells:
        key = _cell_stratum(cell) + "|none_observed"
        strata[key] = {
            "matched_stratum": key,
            "game": cell.get("game"),
            "run": cell.get("run"),
            "model": cell.get("model"),
            "policy": cell.get("policy"),
            "budget": cell.get("budget"),
            "tool_loop_state": cell.get("tool_loop_state"),
            "supervisor_state": cell.get("supervisor_state"),
            "requested_tool": "none_observed",
            "terminal_tool_gap_cell": cell.get("tool_gap_receipt_complete") is True,
            "obligation_count": 0,
            "transport_success_count": 0,
            "response_used_count": 0,
            "nonzero_headroom": False,
        }
    for stratum, group in groups.items():
        first = group[0]
        strata[stratum] = {
            "matched_stratum": stratum,
            "game": first.get("game"),
            "run": first.get("run"),
            "model": first.get("model"),
            "policy": first.get("policy"),
            "budget": first.get("budget"),
            "tool_loop_state": first.get("tool_loop_state"),
            "supervisor_state": first.get("supervisor_state"),
            "requested_tool": first.get("requested_tool"),
            "terminal_tool_gap_cell": True,
            "obligation_count": len(group),
            "transport_success_count": sum(int(row.get("transport_success")) for row in group),
            "response_used_count": sum(int(row.get("response_used")) for row in group),
            "nonzero_headroom": any(
                row.get("headroom", {}).get("nonzero_headroom") for row in group
            ),
        }
        cell_key = (
            "|".join(
                str(first.get(key))
                for key in (
                    "game",
                    "run",
                    "model",
                    "policy",
                    "budget",
                    "tool_loop_state",
                    "supervisor_state",
                )
            )
            + "|none_observed"
        )
        strata.pop(cell_key, None)
    values = sorted(strata.values(), key=lambda row: str(row["matched_stratum"]))
    return {
        "stratum_count": len(values),
        "obligation_stratum_count": sum(int(row["obligation_count"] > 0) for row in values),
        "unmatched_stratum_count": sum(int(row["obligation_count"] == 0) for row in values),
        "strata": values,
        "expected_missing_configurations": list(expected_missing),
        "pooling_rule": (
            "game, run, model, policy, budget, tool-loop state, supervisor state, "
            "and requested tool must match"
        ),
    }


def request_receipt_joins(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    request_ids = [row.get("request", {}).get("request_id") for row in rows]
    response_ids = [row.get("receipt", {}).get("response_id") for row in rows]
    obligation_ids = [row.get("obligation_identity") for row in rows]
    duplicate_requests = sorted(
        item for item, count in Counter(request_ids).items() if item and count > 1
    )
    duplicate_responses = sorted(
        item for item, count in Counter(response_ids).items() if item and count > 1
    )
    duplicate_obligations = sorted(
        item for item, count in Counter(obligation_ids).items() if item and count > 1
    )
    missing = [
        row["row_sha256"]
        for row in rows
        if not (
            _is_sha256(row.get("request", {}).get("request_id"))
            and _is_sha256(row.get("receipt", {}).get("tool_receipt_id"))
            and _is_sha256(row.get("receipt", {}).get("response_id"))
        )
    ]
    mismatch = [
        row["row_sha256"]
        for row in rows
        if not (
            isinstance(row.get("actual_call"), Mapping)
            and str(row.get("actual_call", {}).get("name")) == str(row.get("requested_tool"))
        )
    ]
    return {
        "rows_reduced": len(rows),
        "joined_count": sum(int(row.get("request_receipt_joined") is True) for row in rows),
        "missing_identity_row_hashes": missing,
        "missing_identity_count": len(missing),
        "request_response_mismatch_row_hashes": mismatch,
        "request_response_mismatch_count": len(mismatch),
        "duplicate_request_ids": duplicate_requests,
        "duplicate_response_ids": duplicate_responses,
        "duplicate_obligation_identities": duplicate_obligations,
        "raw_transcript_hash_missing_count": sum(
            int(not _is_sha256(row.get("raw_transcript_sha256"))) for row in rows
        ),
    }


def _stratum_rate_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    counters: Mapping[str, str],
) -> list[JsonDict]:
    results: list[JsonDict] = []
    for stratum, group in sorted(_group_by_stratum(rows).items()):
        total = len(group)
        base: JsonDict = {
            "matched_stratum": stratum,
            "game": group[0]["game"],
            "run": group[0]["run"],
            "model": group[0]["model"],
            "policy": group[0]["policy"],
            "budget": group[0]["budget"],
            "tool_loop_state": group[0]["tool_loop_state"],
            "supervisor_state": group[0]["supervisor_state"],
            "requested_tool": group[0]["requested_tool"],
            "obligation_count": total,
        }
        for output_name, row_key in counters.items():
            count = sum(int(row.get(row_key) is True or row.get(row_key) == 1) for row in group)
            base[f"{output_name}_count"] = count
            base[f"{output_name}_rate"] = count / total if total else None
            base[f"{output_name}_uncertainty"] = _rate_interval(count, total)
        results.append(base)
    return results


def agent_visibility_results(rows: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    return _stratum_rate_rows(rows, counters={"agent_visible": "agent_visible"})


def next_action_results(rows: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    return _stratum_rate_rows(
        rows,
        counters={
            "next_action_receipted": "next_action_receipted",
            "action_change": "action_changed",
        },
    )


def later_outcome_results(rows: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    return _stratum_rate_rows(
        rows,
        counters={
            "exact_later_outcome": "exact_later_outcome_joined",
            "transition_progress": "transition_progress",
        },
    )


def transport_results(rows: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    return _stratum_rate_rows(
        rows,
        counters={
            "request_receipt_joined": "request_receipt_joined",
            "transport_success": "transport_success",
        },
    )


def utility_results(rows: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    results: list[JsonDict] = []
    for stratum, group in sorted(_group_by_stratum(rows).items()):
        total = len(group)
        used = sum(int(row.get("response_used") is True) for row in group)
        changed = sum(int(row.get("action_changed") is True) for row in group)
        progress = sum(int(row.get("transition_progress") == 1) for row in group)
        invalid = sum(int(row.get("invalid_action") is True) for row in group)
        scores = [float(row["later_exact_outcome"]["level_delta"]) for row in group]
        results.append(
            {
                "matched_stratum": stratum,
                "game": group[0]["game"],
                "run": group[0]["run"],
                "model": group[0]["model"],
                "policy": group[0]["policy"],
                "budget": group[0]["budget"],
                "tool_loop_state": group[0]["tool_loop_state"],
                "supervisor_state": group[0]["supervisor_state"],
                "requested_tool": group[0]["requested_tool"],
                "obligation_count": total,
                "use_count": used,
                "use_rate": used / total if total else None,
                "action_change_count": changed,
                "action_change_rate": changed / total if total else None,
                "transition_progress_count": progress,
                "transition_progress_rate": progress / total if total else None,
                "invalid_action_count": invalid,
                "invalid_action_rate": invalid / total if total else None,
                "mean_later_outcome_score": _mean(scores),
                "uncertainty": {
                    "use_rate_interval_95": _rate_interval(used, total),
                    "action_change_rate_interval_95": _rate_interval(changed, total),
                    "transition_progress_rate_interval_95": _rate_interval(progress, total),
                    "invalid_action_rate_interval_95": _rate_interval(invalid, total),
                    "small_n_descriptive_only": total < 30,
                },
                "effect_eligible": all(
                    row.get("transport_success")
                    and row.get("exact_later_outcome_joined")
                    and row.get("temporal_order_verified")
                    and row.get("matched_cell")
                    for row in group
                )
                and any(row.get("headroom", {}).get("nonzero_headroom") for row in group),
            }
        )
    return results


def headroom_results(rows: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    results: list[JsonDict] = []
    for stratum, group in sorted(_group_by_stratum(rows).items()):
        outcome_scores = [float(row["later_exact_outcome"]["level_delta"]) for row in group]
        nonzero = any(row.get("headroom", {}).get("nonzero_headroom") for row in group)
        results.append(
            {
                "matched_stratum": stratum,
                "game": group[0]["game"],
                "run": group[0]["run"],
                "model": group[0]["model"],
                "policy": group[0]["policy"],
                "budget": group[0]["budget"],
                "tool_loop_state": group[0]["tool_loop_state"],
                "supervisor_state": group[0]["supervisor_state"],
                "requested_tool": group[0]["requested_tool"],
                "obligation_count": len(group),
                "later_outcome_scores": outcome_scores,
                "nonzero_headroom": nonzero,
                "no_headroom_row_hashes": [
                    row["row_sha256"]
                    for row in group
                    if row.get("headroom", {}).get("nonzero_headroom") is not True
                ],
                "rule": "later_exact_outcome.headroom must show a score difference",
            }
        )
    return results


def _blocked_source_diagnostics(records: Mapping[str, Mapping[str, Any]]) -> list[JsonDict]:
    diagnostics: list[JsonDict] = []
    for name in ("experiment_6777", "experiment_6820_missing"):
        record = records.get(name, {})
        if record.get("exists") is not True or record.get("status") == "blocked":
            diagnostics.append(
                {
                    "source_artifact": name,
                    "matched_stratum": None,
                    "game": None,
                    "reason": "missing_source_artifact"
                    if record.get("exists") is not True
                    else "blocked_source_artifact",
                    "row_count": 0,
                    "observed": {
                        "path": record.get("path"),
                        "read_error": record.get("read_error"),
                        "status": record.get("status"),
                        "honest_verdict": record.get("honest_verdict"),
                    },
                }
            )
    return diagnostics


def _status_declares_expected_batches(records: Mapping[str, Mapping[str, Any]]) -> bool:
    text = str(records.get("ops_status", {}).get("_raw_text", ""))
    return "sp80,su15" in text and "tu93,cn04" in text and "m0r0,sk48" in text


def expected_configuration_diagnostics(
    records: Mapping[str, Mapping[str, Any]],
    cells: Sequence[Mapping[str, Any]],
) -> list[JsonDict]:
    if not _status_declares_expected_batches(records):
        return []
    present = {(str(cell.get("game")), str(cell.get("tool_loop_state"))) for cell in cells}
    diagnostics: list[JsonDict] = []
    for game, tool_loop_state in EXPECTED_BATCH_CELLS:
        if (game, tool_loop_state) in present:
            continue
        diagnostics.append(
            {
                "source_artifact": "ops_status",
                "matched_stratum": None,
                "game": game,
                "reason": "expected_configuration_missing_terminal_receipt",
                "row_count": 0,
                "observed": {"expected_tool_loop_state": tool_loop_state},
            }
        )
    return diagnostics


def obligation_ledger(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    return {
        "row_count": len(rows),
        "rows": list(rows),
        "row_rule": (
            "one row per tool_gap_events entry with request, receipt, visible text, "
            "next action, and later outcome fields preserved"
        ),
    }


def _all_visible(
    visibility: Sequence[Mapping[str, Any]], rows: Sequence[Mapping[str, Any]]
) -> bool:
    return bool(rows) and all(row.get("agent_visible") is True for row in rows)


def _all_next_actions(rows: Sequence[Mapping[str, Any]]) -> bool:
    return bool(rows) and all(row.get("next_action_receipted") is True for row in rows)


def _all_exact_outcomes(rows: Sequence[Mapping[str, Any]]) -> bool:
    return bool(rows) and all(row.get("exact_later_outcome_joined") is True for row in rows)


def _all_temporal(rows: Sequence[Mapping[str, Any]]) -> bool:
    return bool(rows) and all(row.get("temporal_order_verified") is True for row in rows)


def _all_matched(rows: Sequence[Mapping[str, Any]]) -> bool:
    return bool(rows) and all(row.get("matched_cell") is True for row in rows)


def _any_headroom(headroom: Sequence[Mapping[str, Any]]) -> bool:
    return any(row.get("nonzero_headroom") is True for row in headroom)


def _duplicate_transport_identities(join: Mapping[str, Any]) -> list[str]:
    return sorted(
        set(join.get("duplicate_request_ids", []))
        | set(join.get("duplicate_response_ids", []))
        | set(join.get("duplicate_obligation_identities", []))
    )


def evaluate_preconditions(
    records: Mapping[str, Mapping[str, Any]],
    cells: Sequence[Mapping[str, Any]],
    rows: Sequence[Mapping[str, Any]],
    join: Mapping[str, Any],
    visibility: Sequence[Mapping[str, Any]],
    next_actions: Sequence[Mapping[str, Any]],
    later_outcomes: Sequence[Mapping[str, Any]],
    strata: Mapping[str, Any],
    headroom: Sequence[Mapping[str, Any]],
) -> list[JsonDict]:
    inventory = _payload(records, "experiment_6843")
    terminal_cells = [cell for cell in cells if cell.get("tool_gap_receipt_complete") is True]
    missing_visibility = sum(int(row.get("agent_visible") is not True) for row in rows)
    missing_next = sum(int(row.get("next_action_receipted") is not True) for row in rows)
    missing_outcomes = sum(int(row.get("exact_later_outcome_joined") is not True) for row in rows)
    temporal_failures = sum(int(row.get("temporal_order_verified") is not True) for row in rows)
    duplicate_ids = _duplicate_transport_identities(join)
    return [
        _check("arc_inventory_complete_score", 1, inventory.get("arc_inventory_complete_score")),
        _check("terminal_tool_gap_cells", ">0", len(terminal_cells), len(terminal_cells) > 0),
        _check("tool_gap_obligations", ">0", len(rows), len(rows) > 0),
        _check(
            "immutable_raw_transcript_hashes",
            True,
            {
                "rows_checked": len(rows),
                "missing": join.get("raw_transcript_hash_missing_count"),
            },
            len(rows) > 0 and join.get("raw_transcript_hash_missing_count") == 0,
        ),
        _check(
            "request_response_identities",
            True,
            {
                "missing_identity_count": join.get("missing_identity_count"),
                "request_response_mismatch_count": join.get("request_response_mismatch_count"),
            },
            join.get("missing_identity_count") == 0
            and join.get("request_response_mismatch_count") == 0
            and bool(rows),
        ),
        _check(
            "agent_visible_responses",
            True,
            {"missing_agent_visible_count": missing_visibility, "visibility_results": visibility},
            _all_visible(visibility, rows),
        ),
        _check(
            "next_action_receipts",
            True,
            {"missing_next_action_count": missing_next, "next_action_results": next_actions},
            _all_next_actions(rows),
        ),
        _check(
            "exact_later_outcomes",
            True,
            {
                "missing_exact_later_outcome_count": missing_outcomes,
                "later_outcome_results": later_outcomes,
            },
            _all_exact_outcomes(rows),
        ),
        _check(
            "temporal_order",
            True,
            {"temporal_failure_count": temporal_failures},
            _all_temporal(rows),
        ),
        _check("duplicate_transport_identities", [], duplicate_ids),
        _check(
            "matched_cells",
            True,
            {
                "all_obligations_matched": _all_matched(rows),
                "configuration_strata": strata,
            },
            _all_matched(rows) and int(strata.get("obligation_stratum_count", 0) or 0) > 0,
        ),
        _check(
            "headroom_nonzero",
            True,
            {"headroom_results": headroom},
            _any_headroom(headroom),
        ),
    ]


def reproducibility_checksum(artifact: Mapping[str, Any]) -> str:
    """Hash all stable artifact content except wall time and this digest."""

    return sha256_json(
        {
            key: value
            for key, value in artifact.items()
            if key not in {"duration_s", "reproducibility_checksum"}
        }
    )


def build_artifact(
    *,
    run_date: str,
    duration_s: float,
    source_paths: Mapping[str, Path] | None = None,
    root: Path | None = None,
) -> JsonDict:
    """Build the tool-gap audit from frozen source artifacts."""

    if root is None:
        root = _common_root(source_paths) if source_paths is not None else REPO_ROOT
    paths = dict(source_paths or collect_default_source_paths(root))
    records = _source_records(paths, root)
    cells = _inventory_cells(records)
    rows, unmatched = reduce_obligation_rows(records, cells)
    unmatched.extend(_blocked_source_diagnostics(records))
    expected_missing = expected_configuration_diagnostics(records, cells)
    unmatched.extend(expected_missing)
    strata = configuration_strata(rows, cells, expected_missing)
    join = request_receipt_joins(rows)
    visibility = agent_visibility_results(rows)
    next_actions = next_action_results(rows)
    later_outcomes = later_outcome_results(rows)
    transport = transport_results(rows)
    utility = utility_results(rows)
    headroom = headroom_results(rows)
    preconditions = evaluate_preconditions(
        records,
        cells,
        rows,
        join,
        visibility,
        next_actions,
        later_outcomes,
        strata,
        headroom,
    )
    gate_summary = _gate_summary(preconditions)
    effect_eligible = int(gate_summary["passed"])
    status = (
        "complete_tool_gap_causal_support_audit"
        if effect_eligible
        else "complete_blocked_tool_gap_causal_support_audit"
    )
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment_id": "exp6845-tool-gap-causal-support-audit",
        "run_date": run_date,
        "status": status,
        "field_principles": dict(FIELD_PRINCIPLES),
        "preconditions_checked": preconditions,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "duration_s": float(duration_s),
        "source_artifact_hashes": source_artifact_hashes(records),
        "reproducibility_checksum": "",
        "per_game_results": rows,
        "configuration_strata": strata,
        "obligation_ledger": obligation_ledger(rows),
        "request_receipt_joins": join,
        "agent_visibility_results": visibility,
        "next_action_results": next_actions,
        "later_outcome_results": later_outcomes,
        "transport_results": transport,
        "utility_results": utility,
        "headroom_results": headroom,
        "unmatched_cell_results": unmatched,
        "tool_gap_audit_complete_score": 1,
        "tool_gap_effect_eligible_score": effect_eligible,
        "solve_claim": False,
        "gate_check_summary": gate_summary,
        "verifier_is_oracle": False,
        "verdict_class": "null" if effect_eligible else "blocked",
        "honest_verdict": (
            "complete_tool_gap_causal_support_audit_effect_eligible_no_solve_claim"
            if effect_eligible
            else "complete_blocked_tool_gap_causal_support_audit"
        ),
    }
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    return artifact


def validate_artifact(artifact: Mapping[str, Any]) -> list[str]:
    """Return schema errors. Claim strength is validated by gate rows."""

    errors: list[str] = []
    missing = [field for field in TOP_LEVEL_FIELDS if field not in artifact]
    if missing:
        errors.append("required artifact fields are missing")
    if set(artifact.get("field_principles", {})) != set(artifact):
        errors.append("field principles do not cover every top-level field")
    if artifact.get("schema") != SCHEMA:
        errors.append("schema mismatch")
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        errors.append("inference substrate mismatch")
    if artifact.get("solve_claim") is not False:
        errors.append("solve_claim must be false")
    if artifact.get("verifier_is_oracle") is not False:
        errors.append("verifier_is_oracle must be false")
    if artifact.get("verdict_class") not in VERDICT_CLASSES:
        errors.append("verdict class is outside the closed set")
    if not str(artifact.get("honest_verdict", "")).startswith("complete_"):
        errors.append("honest verdict lacks complete_ terminal prefix")
    if artifact.get("reproducibility_checksum") != reproducibility_checksum(artifact):
        errors.append("reproducibility checksum mismatch")
    if artifact.get("tool_gap_audit_complete_score") != 1:
        errors.append("tool-gap audit complete score mismatch")
    if artifact.get("tool_gap_effect_eligible_score") not in {0, 1}:
        errors.append("effect eligible score must be binary")

    status = artifact.get("status")
    gate = artifact.get("gate_check_summary", {})
    if status == "complete_blocked_tool_gap_causal_support_audit":
        if artifact.get("verdict_class") != "blocked":
            errors.append("blocked verdict_class mismatch")
        if artifact.get("tool_gap_effect_eligible_score") != 0:
            errors.append("blocked artifact marked effect eligible")
        if not isinstance(gate, Mapping) or not gate.get("failed_check"):
            errors.append("blocked artifact lacks failed check")
    elif status == "complete_tool_gap_causal_support_audit":
        if artifact.get("verdict_class") != "null":
            errors.append("complete audit verdict_class mismatch")
        if artifact.get("tool_gap_effect_eligible_score") != 1:
            errors.append("complete artifact lacks effect eligibility")
        if not isinstance(gate, Mapping) or gate.get("passed") is not True:
            errors.append("complete artifact gate summary mismatch")
        if not artifact.get("per_game_results"):
            errors.append("complete audit lacks per_game rows")
    else:
        errors.append("status mismatch")

    for row in artifact.get("per_game_results", []):
        if not str(row.get("row_sha256", "")).startswith("sha256:"):
            errors.append("per_game row missing hash")
            break
    return errors


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Audit tool-gap causal support.")
    parser.add_argument("--date", required=True, help="execution date as YYYYMMDD")
    parser.add_argument("--output", default=str(REPO_ROOT / OUTPUT_PATH), help="artifact path")
    args = parser.parse_args(argv)

    start = time.monotonic()
    paths = collect_default_source_paths(REPO_ROOT)
    artifact = build_artifact(
        run_date=args.date,
        duration_s=time.monotonic() - start,
        source_paths=paths,
        root=REPO_ROOT,
    )
    errors = validate_artifact(artifact)
    if errors:
        for error in errors:
            print(f"validation error: {error}", file=sys.stderr)
        return 1
    _write_json_atomic(Path(args.output), artifact)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
