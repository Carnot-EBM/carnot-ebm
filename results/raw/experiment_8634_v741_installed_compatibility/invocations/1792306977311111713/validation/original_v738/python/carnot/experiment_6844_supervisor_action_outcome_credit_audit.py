"""Audit supervisor action credit from exact frozen ARC outcome receipts.

Spec refs: REQ-ARC-6844 and SCENARIO-ARC-6844-*.

This reducer reads existing artifacts only. It does not launch ARC, load a
model, or import earlier benefit aggregates. Each action keeps its own receipt,
dose, and later outcome so a run-level claim cannot hide row-level evidence.
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
MODULE_PATH = Path("python/carnot/experiment_6844_supervisor_action_outcome_credit_audit.py")
WRAPPER_PATH = Path("scripts/experiments/experiment_6844_supervisor_action_outcome_credit_audit.py")
OUTPUT_PATH = Path("results/experiment_6844_supervisor_action_outcome_credit_audit.json")
SCHEMA = "carnot.experiment_6844.supervisor_action_outcome_credit_audit.v1"
INFERENCE_SUBSTRATE = DETERMINISTIC_CPU_LIVE_RECEIPT_AUDIT_NO_LLM_SUBSTRATE
VERDICT_CLASSES = {"positive", "circular_positive", "null", "blocked", "disqualified", "partial"}

REQUIRED_ARTIFACT_FIELDS = (
    "field_principles",
    "preconditions_checked",
    "inference_substrate",
    "duration_s",
    "source_artifact_hashes",
    "reproducibility_checksum",
    "per_game_results",
    "configuration_strata",
    "exact_outcome_join_results",
    "action_credit_results",
    "headroom_results",
    "unmatched_cell_results",
    "transition_progress_results",
    "invalid_action_results",
    "supervisor_causal_audit_complete_score",
    "supervisor_effect_eligible_score",
    "solve_claim",
    "gate_check_summary",
    "verifier_is_oracle",
    "verdict_class",
    "honest_verdict",
)
TOP_LEVEL_FIELDS = ("schema", "experiment_id", "run_date", "status", *REQUIRED_ARTIFACT_FIELDS)

FIELD_PRINCIPLES: JsonDict = {
    "schema": "The schema lets later reducers reject incompatible audit files.",
    "experiment_id": "The identifier binds this artifact to REQ-ARC-6844.",
    "run_date": "The date states when the frozen receipt audit ran.",
    "status": "The status separates effect-eligible audits from blocked audits.",
    "field_principles": "Each top-level field states why an auditor needs it.",
    "preconditions_checked": "The gates show which evidence was required before eligibility.",
    "inference_substrate": "This is deterministic CPU live-receipt audit work, not a live run.",
    "duration_s": "Wall time exposes skipped or fabricated reducer execution.",
    "source_artifact_hashes": "Source hashes bind the frozen artifacts used as inputs.",
    "reproducibility_checksum": "One digest binds the audit content except wall time.",
    "per_game_results": "One action or matched control row keeps dose and outcome local.",
    "configuration_strata": "Strata prevent pooling across unlike games or configurations.",
    "exact_outcome_join_results": "Join counts show whether later outcomes are exact receipts.",
    "action_credit_results": "Credit summaries are recomputed from rows, not prior aggregates.",
    "headroom_results": "Headroom shows whether matched cells could distinguish actions.",
    "unmatched_cell_results": "Unmatched and blocked sources remain visible as diagnostics.",
    "transition_progress_results": "Progress, regression, and abstention are reported by stratum.",
    "invalid_action_results": "Invalid-action rates separate bad actions from neutral outcomes.",
    "supervisor_causal_audit_complete_score": "Completeness is about audit shape, not effect size.",
    "supervisor_effect_eligible_score": "Eligibility is 1 only when matching, timing, headroom, and exact outcomes pass.",
    "solve_claim": "False prevents an audit artifact from becoming an ARC solve claim.",
    "gate_check_summary": "A blocked verdict names the first failed gate and observed value.",
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


def collect_default_source_paths(root: Path = REPO_ROOT) -> dict[str, Path]:
    """Collect only frozen inputs and source files for this reducer."""

    return {
        "claude_md": root / "CLAUDE.md",
        "codex_md": root / "CODEX.md",
        "north_star": root / "ops/north-star.md",
        "arc_trajectory_supervisor": root / "python/carnot/agentic/arc_trajectory_supervisor.py",
        "spec": root / SPEC_PATH,
        "module_source": root / MODULE_PATH,
        "wrapper_source": root / WRAPPER_PATH,
        "experiment_6524": root
        / "results/experiment_6524_arc_supervisor_redirect_generalization.json",
        "experiment_6681": root / "results/experiment_6681_arc_post_redirect_outcomes.json",
        "experiment_6682": root / "results/experiment_6682_arc_held_family_supervisor_ab.json",
        "experiment_6776": root / "results/experiment_6776_arc_shadow_supervisor_accrual.json",
        "experiment_6843": root / "results/experiment_6843_live_arc_evidence_stratum_freeze.json",
    }


def _source_record(name: str, path: Path, root: Path) -> JsonDict:
    raw, error = _read_bytes(path)
    payload = _load_json(raw)
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
    }


def _source_records(source_paths: Mapping[str, Path], root: Path) -> dict[str, JsonDict]:
    return {name: _source_record(name, path, root) for name, path in sorted(source_paths.items())}


def source_artifact_hashes(records: Mapping[str, Mapping[str, Any]]) -> dict[str, JsonDict]:
    return {
        name: {key: value for key, value in record.items() if key != "_payload"}
        for name, record in records.items()
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


def _payload(records: Mapping[str, Mapping[str, Any]], name: str) -> JsonDict:
    value = records.get(name, {}).get("_payload", {})
    return value if isinstance(value, dict) else {}


def _finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _mean(values: Sequence[float]) -> float | None:
    if not values:
        return None
    return float(statistics.fmean(values))


def _outcome_score(row: Mapping[str, Any]) -> float:
    reward = row.get("reward")
    if isinstance(reward, Mapping) and reward.get("present") is True:
        value = reward.get("value")
        if _finite_number(value):
            return float(value)
    before = row.get("levels_completed_before", 0)
    after = row.get("levels_completed_after", 0)
    level_delta = float(after) - float(before) if _finite_number(before) and _finite_number(after) else 0.0
    termination = row.get("termination")
    state = str(termination.get("state", "") if isinstance(termination, Mapping) else "").upper()
    if state == "GAME_OVER" and level_delta <= 0:
        return -1.0
    return level_delta


def _direction(row: Mapping[str, Any]) -> str:
    score = _outcome_score(row)
    if score > 0:
        return "progress"
    if score < 0 or row.get("error") is not None or row.get("outcome_status") != "returned":
        return "regression"
    return "abstention"


def _lineage(row: Mapping[str, Any]) -> JsonDict:
    lineage = row.get("lineage")
    if isinstance(lineage, Mapping):
        return {
            key: row[key] if key in row else lineage.get(key)
            for key in _LINEAGE_KEYS
        }
    return {key: row.get(key) for key in _LINEAGE_KEYS}


_LINEAGE_KEYS = ("proposal_id", "application_id", "environment_step_id", "outcome_id")


def _receipt_complete(row: Mapping[str, Any]) -> bool:
    lineage = _lineage(row)
    if not all(isinstance(lineage.get(key), str) and lineage[key] for key in _LINEAGE_KEYS):
        return False
    return bool(row.get("fully_joined") is True and row.get("live_return") is True)


def _temporal_order_ok(row: Mapping[str, Any]) -> bool:
    decision = row.get("decision_sequence")
    outcome = row.get("outcome_sequence")
    if _finite_number(decision) and _finite_number(outcome):
        return float(outcome) > float(decision)
    return _receipt_complete(row) and row.get("outcome_status") == "returned"


def _action_validity(row: Mapping[str, Any]) -> JsonDict:
    action = row.get("proposed_action")
    if not isinstance(action, Mapping):
        return {"valid": False, "reason": "missing_proposed_action", "available_action_ids": []}
    kind = action.get("kind")
    before = row.get("observation_before")
    available = before.get("available_actions", []) if isinstance(before, Mapping) else []
    available_list = list(available) if isinstance(available, list) else []
    if kind == "RESET":
        return {"valid": True, "reason": "reset_control", "available_action_ids": available_list}
    if kind not in available_list:
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


def _row_kind(row: Mapping[str, Any]) -> str:
    return "eligible_redirect" if row.get("redirect_applied") is True else "matched_control"


def _episode_budget(payload: Mapping[str, Any], row: Mapping[str, Any]) -> Any:
    if row.get("budget") is not None:
        return row.get("budget")
    for episode in payload.get("canonical_path_receipt", {}).get("live_metadata", {}).get(
        "episode_rows", []
    ):
        if not isinstance(episode, Mapping):
            continue
        row_episode_id = row.get("episode_id")
        row_family = row.get("family")
        episode_matches = row_episode_id is not None and episode.get("episode_id") == row_episode_id
        family_matches = row_family is not None and episode.get("family") == row_family
        if episode_matches or family_matches:
            return episode.get("actions")
    return None


def _policy(payload: Mapping[str, Any], row: Mapping[str, Any]) -> str:
    value = row.get("policy") or payload.get("policy")
    if value is None:
        value = payload.get("canonical_path_receipt", {}).get("policy")
    return str(value or "unknown")


def _model(row: Mapping[str, Any]) -> str:
    return str(row.get("model_id") or "unknown")


def _supervisor_mode(row: Mapping[str, Any]) -> str:
    return str(row.get("supervisor_mode") or "applied")


def _tool_loop_state(row: Mapping[str, Any]) -> str:
    return str(row.get("tool_loop_state") or "off_or_unobserved")


def _game(row: Mapping[str, Any]) -> str:
    return str(row.get("game") or row.get("family") or "unknown")


def _run(row: Mapping[str, Any]) -> str:
    if row.get("episode_id"):
        return str(row["episode_id"])
    return f"{_game(row)}:{row.get('attempt')}:{row.get('episode_seed')}"


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
            "supervisor_mode",
        )
    )


def _row_with_hash(row: JsonDict) -> JsonDict:
    material = {key: value for key, value in row.items() if key != "row_sha256"}
    return {**row, "row_sha256": sha256_json(material)}


def _reduce_outcome_row(payload: Mapping[str, Any], row: Mapping[str, Any]) -> JsonDict:
    lineage = _lineage(row)
    action = row.get("applied_action")
    exact_ok = (
        _receipt_complete(row)
        and row.get("outcome_status") == "returned"
        and isinstance(row.get("return_hash"), str)
        and isinstance(row.get("state_hash"), str)
    )
    temporal_ok = _temporal_order_ok(row)
    validity = _action_validity(row)
    reduced: JsonDict = {
        "row_kind": _row_kind(row),
        "source_artifact": "experiment_6681",
        "game": _game(row),
        "run": _run(row),
        "model": _model(row),
        "policy": _policy(payload, row),
        "budget": _episode_budget(payload, row),
        "supervisor_mode": _supervisor_mode(row),
        "tool_loop_state": _tool_loop_state(row),
        "action_index": row.get("action_index"),
        "action": action,
        "proposed_action": row.get("proposed_action"),
        "receipt": lineage,
        "next_state": {
            "levels_completed_before": row.get("levels_completed_before"),
            "levels_completed_after": row.get("levels_completed_after"),
            "termination": row.get("termination"),
            "state_hash": row.get("state_hash"),
            "return_hash": row.get("return_hash"),
        },
        "later_exact_outcome": {
            "source": "experiment_6681.redirect_outcome_rows"
            if row.get("redirect_applied") is True
            else "experiment_6681.non_redirect_control_rows",
            "outcome_id": lineage.get("outcome_id"),
            "outcome_status": row.get("outcome_status"),
            "fully_joined": row.get("fully_joined") is True,
            "live_return": row.get("live_return") is True,
            "level_delta": _outcome_score(row),
            "direction": _direction(row),
        },
        "dose": {
            "value": float(row.get("action_cost", 1) or 0),
            "unit": "environment_action",
            "applied_to_action_identity": None,
        },
        "headroom": {"nonzero_headroom": False, "stratum_score_values": []},
        "transition_progress": 1 if _direction(row) == "progress" else 0,
        "regression": _direction(row) == "regression",
        "abstention": _direction(row) == "abstention",
        "proposal_validity": validity,
        "invalid_action": validity.get("valid") is not True or row.get("error") is not None,
        "exact_outcome_joined": exact_ok,
        "temporal_order_verified": temporal_ok,
    }
    reduced["matched_stratum"] = _stratum_identity(reduced)
    reduced["action_identity"] = sha256_json(
        {
            "run": reduced["run"],
            "action_index": reduced["action_index"],
            "action": reduced["action"],
            "receipt": reduced["receipt"],
        }
    )
    reduced["dose"]["applied_to_action_identity"] = reduced["action_identity"]
    return _row_with_hash(reduced)


def reduce_exact_outcome_rows(payload: Mapping[str, Any]) -> list[JsonDict]:
    """Build row-level evidence from exact outcome rows only."""

    rows: list[JsonDict] = []
    for source_key in ("redirect_outcome_rows", "non_redirect_control_rows"):
        values = payload.get(source_key, [])
        if not isinstance(values, list):
            continue
        for row in values:
            if isinstance(row, Mapping):
                rows.append(_reduce_outcome_row(payload, row))
    return sorted(rows, key=lambda row: (row["matched_stratum"], row["action_index"], row["row_kind"]))


def _group_by_stratum(rows: Sequence[Mapping[str, Any]]) -> dict[str, list[Mapping[str, Any]]]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["matched_stratum"])].append(row)
    return dict(grouped)


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


def _with_headroom(rows: Sequence[Mapping[str, Any]]) -> tuple[list[JsonDict], list[JsonDict]]:
    groups = _group_by_stratum(rows)
    headroom_results: list[JsonDict] = []
    row_results: list[JsonDict] = []
    for stratum, group in sorted(groups.items()):
        scores = [float(row["later_exact_outcome"]["level_delta"]) for row in group]
        redirects = [row for row in group if row["row_kind"] == "eligible_redirect"]
        controls = [row for row in group if row["row_kind"] == "matched_control"]
        nonzero = bool(redirects and controls and len(set(scores)) > 1)
        headroom = {
            "matched_stratum": stratum,
            "game": group[0]["game"],
            "run": group[0]["run"],
            "model": group[0]["model"],
            "policy": group[0]["policy"],
            "budget": group[0]["budget"],
            "tool_loop_state": group[0]["tool_loop_state"],
            "supervisor_mode": group[0]["supervisor_mode"],
            "redirect_count": len(redirects),
            "matched_control_count": len(controls),
            "redirect_scores": [row["later_exact_outcome"]["level_delta"] for row in redirects],
            "control_scores": [row["later_exact_outcome"]["level_delta"] for row in controls],
            "nonzero_headroom": nonzero,
            "rule": "same-stratum redirect/control exact outcome scores are not all equal",
        }
        headroom_results.append(headroom)
        for row in group:
            changed = dict(row)
            changed["headroom"] = {
                "nonzero_headroom": nonzero,
                "stratum_score_values": scores,
                "matched_control_count": len(controls),
            }
            changed["row_sha256"] = sha256_json(
                {key: value for key, value in changed.items() if key != "row_sha256"}
            )
            row_results.append(changed)
    return sorted(
        row_results,
        key=lambda row: (row["matched_stratum"], row["action_index"], row["row_kind"]),
    ), headroom_results


def configuration_strata(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    groups = _group_by_stratum(rows)
    strata: list[JsonDict] = []
    for stratum, group in sorted(groups.items()):
        redirects = [row for row in group if row["row_kind"] == "eligible_redirect"]
        controls = [row for row in group if row["row_kind"] == "matched_control"]
        strata.append(
            {
                "matched_stratum": stratum,
                "game": group[0]["game"],
                "run": group[0]["run"],
                "model": group[0]["model"],
                "policy": group[0]["policy"],
                "budget": group[0]["budget"],
                "tool_loop_state": group[0]["tool_loop_state"],
                "supervisor_mode": group[0]["supervisor_mode"],
                "redirect_count": len(redirects),
                "matched_control_count": len(controls),
                "matched": bool(redirects and controls),
                "nonzero_headroom": bool(
                    redirects
                    and controls
                    and len(
                        {
                            float(row["later_exact_outcome"]["level_delta"])
                            for row in group
                        }
                    )
                    > 1
                ),
            }
        )
    return {
        "stratum_count": len(strata),
        "matched_stratum_count": sum(int(row["matched"]) for row in strata),
        "unmatched_stratum_count": sum(int(not row["matched"]) for row in strata),
        "strata": strata,
        "pooling_rule": "game, run, model, policy, budget, tool-loop state, and supervisor mode must match",
    }


def exact_outcome_join_results(
    rows: Sequence[Mapping[str, Any]],
    outcome_payload: Mapping[str, Any],
) -> JsonDict:
    outcome_ids = [row["receipt"].get("outcome_id") for row in rows]
    duplicate_outcomes = sorted(
        item for item, count in Counter(outcome_ids).items() if item and count > 1
    )
    action_ids = [row.get("action_identity") for row in rows]
    duplicate_actions = sorted(
        item for item, count in Counter(action_ids).items() if item and count > 1
    )
    missing_receipts = [
        row["row_sha256"]
        for row in rows
        if not row.get("exact_outcome_joined")
    ]
    temporal_failures = [
        row["row_sha256"]
        for row in rows
        if not row.get("temporal_order_verified")
    ]
    return {
        "source_artifact": "experiment_6681",
        "source_ready": outcome_payload.get("arc_outcome_transport_ready") is True,
        "rows_reduced": len(rows),
        "redirect_row_count": sum(int(row["row_kind"] == "eligible_redirect") for row in rows),
        "matched_control_row_count": sum(int(row["row_kind"] == "matched_control") for row in rows),
        "exact_later_outcome_count": sum(int(row.get("exact_outcome_joined") is True) for row in rows),
        "missing_receipt_row_hashes": missing_receipts,
        "missing_receipt_count": len(missing_receipts),
        "temporal_failure_row_hashes": temporal_failures,
        "temporal_failure_count": len(temporal_failures),
        "duplicate_outcome_ids": duplicate_outcomes,
        "duplicate_action_identities": duplicate_actions,
        "action_identity_missing_count": sum(
            int(not str(row.get("action_identity", "")).startswith("sha256:")) for row in rows
        ),
    }


def _count_direction(rows: Sequence[Mapping[str, Any]], kind: str, direction: str) -> int:
    return sum(
        int(row["row_kind"] == kind and row["later_exact_outcome"]["direction"] == direction)
        for row in rows
    )


def action_credit_results(rows: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    results: list[JsonDict] = []
    for stratum, group in sorted(_group_by_stratum(rows).items()):
        redirects = [row for row in group if row["row_kind"] == "eligible_redirect"]
        controls = [row for row in group if row["row_kind"] == "matched_control"]
        redirect_scores = [float(row["later_exact_outcome"]["level_delta"]) for row in redirects]
        control_scores = [float(row["later_exact_outcome"]["level_delta"]) for row in controls]
        redirect_mean = _mean(redirect_scores)
        control_mean = _mean(control_scores)
        effect_delta = (
            None if redirect_mean is None or control_mean is None else redirect_mean - control_mean
        )
        nonzero_headroom = bool(redirects and controls and len(set(redirect_scores + control_scores)) > 1)
        total = len(group)
        results.append(
            {
                "matched_stratum": stratum,
                "game": group[0]["game"],
                "run": group[0]["run"],
                "model": group[0]["model"],
                "policy": group[0]["policy"],
                "budget": group[0]["budget"],
                "tool_loop_state": group[0]["tool_loop_state"],
                "supervisor_mode": group[0]["supervisor_mode"],
                "matched_cells_compared": bool(redirects and controls),
                "redirect_count": len(redirects),
                "matched_control_count": len(controls),
                "redirect_progress_count": _count_direction(group, "eligible_redirect", "progress"),
                "control_progress_count": _count_direction(group, "matched_control", "progress"),
                "redirect_regression_count": _count_direction(group, "eligible_redirect", "regression"),
                "control_regression_count": _count_direction(group, "matched_control", "regression"),
                "redirect_abstention_count": _count_direction(group, "eligible_redirect", "abstention"),
                "control_abstention_count": _count_direction(group, "matched_control", "abstention"),
                "redirect_invalid_action_rate": (
                    sum(int(row["invalid_action"]) for row in redirects) / len(redirects)
                    if redirects
                    else None
                ),
                "control_invalid_action_rate": (
                    sum(int(row["invalid_action"]) for row in controls) / len(controls)
                    if controls
                    else None
                ),
                "redirect_mean_outcome_score": redirect_mean,
                "control_mean_outcome_score": control_mean,
                "effect_delta": effect_delta,
                "nonzero_headroom": nonzero_headroom,
                "uncertainty": {
                    "method": "descriptive_normal_approximation_by_stratum",
                    "sample_size": total,
                    "progress_rate_interval_95": _rate_interval(
                        sum(int(row["transition_progress"]) for row in group), total
                    ),
                    "small_n_descriptive_only": total < 30,
                },
                "effect_eligible": bool(redirects and controls and nonzero_headroom),
            }
        )
    return results


def transition_progress_results(rows: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    results: list[JsonDict] = []
    for stratum, group in sorted(_group_by_stratum(rows).items()):
        total = len(group)
        progress = sum(int(row["transition_progress"]) for row in group)
        regression = sum(int(row["regression"]) for row in group)
        abstention = sum(int(row["abstention"]) for row in group)
        results.append(
            {
                "matched_stratum": stratum,
                "game": group[0]["game"],
                "total_actions": total,
                "transition_progress_count": progress,
                "regression_count": regression,
                "abstention_count": abstention,
                "transition_progress_rate": progress / total if total else None,
                "regression_rate": regression / total if total else None,
                "abstention_rate": abstention / total if total else None,
                "uncertainty": {
                    "progress_rate_interval_95": _rate_interval(progress, total),
                    "regression_rate_interval_95": _rate_interval(regression, total),
                    "abstention_rate_interval_95": _rate_interval(abstention, total),
                },
            }
        )
    return results


def invalid_action_results(rows: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    results: list[JsonDict] = []
    for stratum, group in sorted(_group_by_stratum(rows).items()):
        total = len(group)
        invalid = sum(int(row["invalid_action"]) for row in group)
        results.append(
            {
                "matched_stratum": stratum,
                "game": group[0]["game"],
                "total_actions": total,
                "invalid_action_count": invalid,
                "invalid_action_rate": invalid / total if total else None,
                "uncertainty": {"invalid_action_rate_interval_95": _rate_interval(invalid, total)},
            }
        )
    return results


def unmatched_cell_results(
    rows: Sequence[Mapping[str, Any]],
    records: Mapping[str, Mapping[str, Any]],
) -> list[JsonDict]:
    unmatched: list[JsonDict] = []
    for stratum, group in sorted(_group_by_stratum(rows).items()):
        redirects = [row for row in group if row["row_kind"] == "eligible_redirect"]
        controls = [row for row in group if row["row_kind"] == "matched_control"]
        reason = None
        if redirects and not controls:
            reason = "no_matched_control_in_same_configuration"
        elif controls and not redirects:
            reason = "control_without_redirect_in_same_configuration"
        if reason is not None:
            unmatched.append(
                {
                    "source_artifact": "experiment_6681",
                    "matched_stratum": stratum,
                    "game": group[0]["game"],
                    "reason": reason,
                    "row_count": len(group),
                }
            )

    inventory = _payload(records, "experiment_6843")
    for item in inventory.get("unmatched_cell_reasons", []) if isinstance(inventory, Mapping) else []:
        if isinstance(item, Mapping):
            unmatched.append(
                {
                    "source_artifact": "experiment_6843",
                    "matched_stratum": item.get("stratum_identity"),
                    "game": item.get("game"),
                    "reason": ",".join(item.get("reasons", []))
                    if isinstance(item.get("reasons"), list)
                    else "inventory_unmatched",
                    "row_count": 1,
                }
            )

    exp6524 = _payload(records, "experiment_6524")
    if "missing_outcome" in str(exp6524.get("status", "")) or exp6524.get("redirect_outcome_rows") == []:
        unmatched.append(
            {
                "source_artifact": "experiment_6524",
                "matched_stratum": None,
                "game": None,
                "reason": "missing_outcome_bearing_receipts",
                "row_count": len(exp6524.get("redirect_outcome_rows", []) or []),
                "observed": exp6524.get("gate_check_summary"),
            }
        )

    exp6682 = _payload(records, "experiment_6682")
    gate6682 = exp6682.get("gate_check_summary", {})
    if exp6682.get("verdict_class") == "partial" or (
        isinstance(gate6682, Mapping) and gate6682.get("failed_check")
    ):
        unmatched.append(
            {
                "source_artifact": "experiment_6682",
                "matched_stratum": None,
                "game": None,
                "reason": str(gate6682.get("failed_check") or "partial_verification"),
                "row_count": len(exp6682.get("paired_episode_rows", []) or []),
                "observed": gate6682.get("observed") if isinstance(gate6682, Mapping) else None,
            }
        )

    exp6776 = _payload(records, "experiment_6776")
    gate6776 = exp6776.get("gate_check_summary", {})
    if exp6776.get("verdict_class") == "blocked" or exp6776.get("shadow_supervisor_transport_ready") is False:
        failed = gate6776.get("failed_check") if isinstance(gate6776, Mapping) else None
        unmatched.append(
            {
                "source_artifact": "experiment_6776",
                "matched_stratum": None,
                "game": None,
                "reason": "resource_block" if failed == "exclusive_gpu_without_unrelated_compute" else str(failed or "blocked"),
                "row_count": len(exp6776.get("rows", []) or []),
                "observed": gate6776.get("observed") if isinstance(gate6776, Mapping) else None,
            }
        )
    return unmatched


def _headroom_nonzero(headroom: Sequence[Mapping[str, Any]]) -> bool:
    return any(row.get("nonzero_headroom") is True for row in headroom)


def _matched_cells_present(strata: Mapping[str, Any]) -> bool:
    return int(strata.get("matched_stratum_count", 0) or 0) > 0


def _all_exact(join: Mapping[str, Any]) -> bool:
    return (
        join.get("source_ready") is True
        and join.get("rows_reduced", 0) > 0
        and join.get("missing_receipt_count") == 0
    )


def _all_temporal(join: Mapping[str, Any]) -> bool:
    return join.get("temporal_failure_count") == 0


def _all_action_identities(join: Mapping[str, Any]) -> bool:
    return not join.get("duplicate_action_identities") and join.get("action_identity_missing_count") == 0


def evaluate_preconditions(
    records: Mapping[str, Mapping[str, Any]],
    rows: Sequence[Mapping[str, Any]],
    strata: Mapping[str, Any],
    join: Mapping[str, Any],
    headroom: Sequence[Mapping[str, Any]],
) -> list[JsonDict]:
    inventory = _payload(records, "experiment_6843")
    source_hashes = source_artifact_hashes(records)
    required_sources = ("experiment_6843", "experiment_6681")
    immutable = all(
        str(source_hashes.get(name, {}).get("file_sha256", "")).startswith("sha256:")
        for name in required_sources
    )
    eligible = (
        inventory.get("supervisor_eligible_cells", {}).get("count", 0)
        if isinstance(inventory.get("supervisor_eligible_cells"), Mapping)
        else 0
    )
    duplicate_outcomes = list(join.get("duplicate_outcome_ids", []) or [])
    return [
        _check("arc_inventory_complete_score", 1, inventory.get("arc_inventory_complete_score")),
        _check("eligible_supervisor_cells", ">0", eligible, eligible > 0),
        _check("immutable_attempts", True, immutable),
        _check("exact_later_outcomes", True, _all_exact(join)),
        _check("temporal_order", True, _all_temporal(join)),
        _check("action_identities", True, _all_action_identities(join)),
        _check("duplicate_exact_outcome_identities", [], duplicate_outcomes),
        _check("matched_cells", True, _matched_cells_present(strata)),
        _check("headroom_nonzero", True, _headroom_nonzero(headroom)),
        _check("per_action_rows_present", True, bool(rows)),
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


def _common_root(source_paths: Mapping[str, Path] | None) -> Path:
    if not source_paths:
        return REPO_ROOT
    existing = [path.resolve() for path in source_paths.values() if path.exists()]
    if not existing:
        return REPO_ROOT
    return Path(os.path.commonpath([str(path) for path in existing]))


def build_artifact(
    *,
    run_date: str,
    duration_s: float,
    source_paths: Mapping[str, Path] | None = None,
    root: Path | None = None,
) -> JsonDict:
    """Build the action-credit audit from frozen source artifacts."""

    if root is None:
        root = _common_root(source_paths) if source_paths is not None else REPO_ROOT
    paths = dict(source_paths or collect_default_source_paths(root))
    records = _source_records(paths, root)
    outcome_payload = _payload(records, "experiment_6681")
    rows_without_headroom = reduce_exact_outcome_rows(outcome_payload)
    rows, headroom = _with_headroom(rows_without_headroom)
    strata = configuration_strata(rows)
    join = exact_outcome_join_results(rows, outcome_payload)
    unmatched = unmatched_cell_results(rows, records)
    preconditions = evaluate_preconditions(records, rows, strata, join, headroom)
    gate_summary = _gate_summary(preconditions)
    effect_eligible = int(gate_summary["passed"])
    status = (
        "complete_supervisor_outcome_credit_audit"
        if effect_eligible
        else "complete_blocked_supervisor_outcome_credit_audit"
    )
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment_id": "exp6844-supervisor-action-outcome-credit-audit",
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
        "exact_outcome_join_results": join,
        "action_credit_results": action_credit_results(rows),
        "headroom_results": headroom,
        "unmatched_cell_results": unmatched,
        "transition_progress_results": transition_progress_results(rows),
        "invalid_action_results": invalid_action_results(rows),
        "supervisor_causal_audit_complete_score": 1,
        "supervisor_effect_eligible_score": effect_eligible,
        "solve_claim": False,
        "gate_check_summary": gate_summary,
        "verifier_is_oracle": False,
        "verdict_class": "null" if effect_eligible else "blocked",
        "honest_verdict": (
            "complete_supervisor_outcome_credit_audit_effect_eligible_no_solve_claim"
            if effect_eligible
            else "complete_blocked_supervisor_outcome_credit_audit"
        ),
    }
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    return artifact


def validate_artifact(artifact: Mapping[str, Any]) -> list[str]:
    """Return schema errors. Claim strength is validated by the gate rows."""

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
    if artifact.get("supervisor_causal_audit_complete_score") != 1:
        errors.append("causal audit complete score mismatch")
    if artifact.get("supervisor_effect_eligible_score") not in {0, 1}:
        errors.append("effect eligible score must be binary")

    status = artifact.get("status")
    gate = artifact.get("gate_check_summary", {})
    if status == "complete_blocked_supervisor_outcome_credit_audit":
        if artifact.get("verdict_class") != "blocked":
            errors.append("blocked verdict_class mismatch")
        if artifact.get("supervisor_effect_eligible_score") != 0:
            errors.append("blocked artifact marked effect eligible")
        if not isinstance(gate, Mapping) or not gate.get("failed_check"):
            errors.append("blocked artifact lacks failed check")
    elif status == "complete_supervisor_outcome_credit_audit":
        if artifact.get("verdict_class") != "null":
            errors.append("complete audit verdict_class mismatch")
        if artifact.get("supervisor_effect_eligible_score") != 1:
            errors.append("complete artifact lacks effect eligibility")
        if not isinstance(gate, Mapping) or gate.get("passed") is not True:
            errors.append("complete artifact gate summary mismatch")
    else:
        errors.append("status mismatch")

    if not artifact.get("per_game_results"):
        errors.append("per_game_results are empty")
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
    parser = argparse.ArgumentParser(description="Audit supervisor action outcome credit.")
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
