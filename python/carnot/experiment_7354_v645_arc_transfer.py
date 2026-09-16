"""Run one bounded matched live ARC result-resume transfer pilot.

The module reuses the shipped E3 policy, local Qwen runner, GPU lease, request
capture, and scoped validator. It adds only dependency gating, the two-game
matched schedule, causal reduction, and terminal accounting.

Spec refs: REQ-ARC-WMTE-7354 and SCENARIO-ARC-WMTE-7354-*.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from copy import deepcopy
from datetime import UTC, datetime
import hashlib
import json
import os
from pathlib import Path
import socket
import tempfile
import time
from typing import Any

import yaml

from carnot import experiment_7305_v642_arc_selfparse as live_base
from carnot.reporting import experiment_7303_validation_scope as validation_scope


JsonDict = dict[str, Any]
REPO_ROOT = Path(__file__).resolve().parents[2]
RUN_DATE = "20260916"
MILESTONE = "2026.09.645"
EXPERIMENT_ID = "exp7354-arc-transfer"
SCHEMA = "carnot.experiment_7354.v645.arc_transfer.v1"

MODEL_SPECS = [{"hf_id": "unsloth/Qwen3.8-27B-GGUF", "quantization": "Q4_K_M"}]
MODEL_ID = MODEL_SPECS[0]["hf_id"]
QUANTIZATION = MODEL_SPECS[0]["quantization"]
DEVELOPMENT_SEED = 7_354_202_609_16
EVALUATION_SEED = 17_354_202_609_16
RESAMPLING_SEED = 27_354_202_609_16
TARGET_ROTATION = ("r11l", "re86")
ACTION_LIMIT = 192
COMPLETION_LIMIT = 2
GENERATED_TOKEN_LIMIT = 4096
TOKENS_PER_CALL = GENERATED_TOKEN_LIMIT // COMPLETION_LIMIT
MODEL_LOAD_LIMIT_S = 600
SESSION_LIMIT_S = 2400

SPEC_PATH = Path("openspec/capabilities/arc-world-model-trust-energy/spec.md")
EXP7345_PATH = Path("results/experiment_7345_v645_arc_resume_check.json")
EXCLUSION_PATH = Path("ops/exclusion_manifest.yaml")
REGISTRY_PATH = Path("ops/arc_solve_registry.yaml")
RESULT_PATH = Path("results/experiment_7354_v645_arc_transfer.json")
RAW_DIR = Path("results/raw/experiment_7354_v645_arc_transfer")
CHECKPOINT_PATH = Path("results/checkpoints/experiment_7354_v645_arc_transfer.json")
SCHEDULE_PATH = RAW_DIR / "frozen_schedule.json"
SESSION_PATH = RAW_DIR / "live_session.json"
EVIDENCE_PATH = RAW_DIR / "causal_evidence.jsonl"
RAW_PANEL_PATH = RAW_DIR / "independent_reduction_input.json"
TERMINAL_CANDIDATE_PATH = RAW_DIR / "measured_terminal_candidate.json"
MODULE_PATH = Path("python/carnot/experiment_7354_v645_arc_transfer.py")
WRAPPER_PATH = Path("scripts/experiments/experiment_7354_v645_arc_transfer.py")
TEST_PATH = Path("tests/python/test_experiment_7354_v645_arc_transfer.py")

REQUIRED_VALIDATION_NAMES = validation_scope.REQUIRED_CHECK_NAMES
REQUIRED_E2E_NAMES = (
    "e2e_009",
    "e2e_010",
    "e2e_offline_smoke",
    "full_python_suite",
)
REQUIRED_TERMINAL_NAMES = (
    "independent_reducer",
    "adversarial_verify",
    "verdict_row_consistency_strict",
)
ZERO_INVOCATION_COUNTS = deepcopy(live_base.ZERO_INVOCATION_COUNTS)

sha256_file = live_base.sha256_file
atomic_write = live_base.atomic_write
prior_session_environment = live_base.session_environment


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()


def artifact_checksum(value: Mapping[str, Any]) -> str:
    payload = deepcopy(dict(value))
    payload["reproducibility_checksum"] = ""
    return "sha256:" + hashlib.sha256(_canonical_bytes(payload)).hexdigest()


def gate_check(
    check: str,
    upstream: str,
    artifact_field: str,
    expected: Any,
    observed: Any,
    principle: str = "Reject unsafe evidence before it affects the current run.",
) -> JsonDict:
    return {
        "check": check,
        "upstream": upstream,
        "artifact_field": artifact_field,
        "expected": expected,
        "observed": observed,
        "passed": observed == expected,
        "principle": principle,
    }


def gate_summary(checks: Sequence[Mapping[str, Any]]) -> JsonDict:
    failed = [deepcopy(dict(row)) for row in checks if row.get("passed") is not True]
    return {
        "all_passed": not failed,
        "failed_count": len(failed),
        "first_failure": failed[0] if failed else None,
    }


def check_dependency(artifact: Mapping[str, Any] | None) -> JsonDict:
    """Reject the first unsafe Exp7345 state before reading its readiness as authority."""

    upstream = EXP7345_PATH.as_posix()
    checks: list[JsonDict] = []
    if artifact is None:
        checks.append(
            gate_check("exp7345_dependency", upstream, "artifact", "available", "missing")
        )
    else:
        checks.extend(
            (
                gate_check(
                    "exp7345_dependency",
                    upstream,
                    "flagged_adversarial",
                    False,
                    bool(artifact.get("flagged_adversarial") or artifact.get("quarantined")),
                ),
                gate_check(
                    "exp7345_dependency",
                    upstream,
                    "verdict_class",
                    "terminal_safe",
                    (
                        artifact.get("verdict_class")
                        if artifact.get("verdict_class") in {"blocked", "disqualified", "partial"}
                        else "terminal_safe"
                    ),
                ),
                gate_check(
                    "exp7345_dependency", upstream, "status", "complete", artifact.get("status")
                ),
                gate_check(
                    "exp7345_dependency",
                    upstream,
                    "seal_for_exp7354",
                    True,
                    artifact.get("seal_for_exp7354"),
                ),
                gate_check(
                    "exp7345_dependency",
                    upstream,
                    "arc_resume_ready_score",
                    1,
                    artifact.get("arc_resume_ready_score"),
                ),
                gate_check(
                    "exp7345_dependency",
                    upstream,
                    "gate_check_summary.all_passed",
                    True,
                    (artifact.get("gate_check_summary") or {}).get("all_passed"),
                ),
            )
        )
    first = next((row for row in checks if row["passed"] is not True), None)
    return {"passed": first is None, "checks": checks, "first_failure": first}


def freeze_panel(
    registry: Mapping[str, Any], *, adaptered_games: set[str] | frozenset[str]
) -> JsonDict:
    """Seal the existing two-game generalization rotation before outcomes exist."""

    indexed = {
        str(row.get("game")): dict(row)
        for row in registry.get("games", [])
        if isinstance(row, Mapping) and row.get("game")
    }
    rows = []
    for position, game in enumerate(TARGET_ROTATION):
        source = indexed.get(game, {})
        rows.append(
            {
                "game": game,
                "eligible": bool(source),
                "least_recently_measured_rank": position,
                "last_measurement": "experiment_7305" if game == "r11l" else "experiment_7319",
                "registry_levels_before_attempt": int(source.get("levels_reproduced") or 0),
                "registry_reproducibility": source.get("reproducibility"),
                "adapter_available_but_withheld": game in adaptered_games,
                "adapter_disabled": True,
                "banked_solution_disabled": True,
                "off_path_engines_disabled": True,
            }
        )
    return {
        "passed": len(rows) == 2 and all(row["eligible"] for row in rows),
        "games": [row["game"] for row in rows],
        "game_rows": rows,
        "selection_basis": "existing_r11l_re86_least_recently_measured_rotation",
        "selection_used_current_outcomes": False,
        "outcomes_seen_before_freeze": False,
        "game_source_read": False,
        "offline_ground_truth_search_used": False,
    }


def matched_schedule(games: Sequence[str]) -> list[JsonDict]:
    """Counterbalance the only treatment difference across two fresh game episodes."""

    rows: list[JsonDict] = []
    arms = ("result_resume", "result_withheld")
    for game_index, game in enumerate(games[:2]):
        ordered = arms if game_index == 0 else tuple(reversed(arms))
        for arm in ordered:
            rows.append(
                {
                    "episode_id": f"{game}:{arm}",
                    "game": str(game),
                    "arm": arm,
                    "seed": EVALUATION_SEED,
                    "execution_order": len(rows),
                    "action_limit": ACTION_LIMIT,
                    "completion_limit": COMPLETION_LIMIT,
                    "generated_token_limit": GENERATED_TOKEN_LIMIT,
                    "fresh_store": True,
                    "adapter_disabled": True,
                    "banked_solution_disabled": True,
                    "off_path_engines_disabled": True,
                }
            )
    return rows


def session_environment(
    base_env: Mapping[str, str],
    *,
    arm: str,
    episode_dir: Path,
    gpu_index: int,
    port: int,
    boundary_path: Path | None = None,
) -> dict[str, str]:
    """Use the qualified resume flag as the only matched treatment difference."""

    env = prior_session_environment(
        base_env,
        episode_dir=episode_dir,
        gpu_index=gpu_index,
        port=port,
        boundary_path=boundary_path,
        arm=arm,
    )
    env.pop("CARNOT_ARC_SELFPARSE_RESULT_RESUME", None)
    env["CARNOT_ARC_RANDOM_SEED"] = str(EVALUATION_SEED)
    env["CARNOT_ARC_GENERATOR_SEED"] = str(EVALUATION_SEED)
    env["CARNOT_7354_EPISODE_ID"] = episode_dir.name.replace("__", ":")
    if arm == "result_resume":
        env["CARNOT_ARC_SELFPARSE_RESULT_RESUME"] = "1"
        env["CARNOT_ARC_INDUCE_TOOL_TURNS"] = str(COMPLETION_LIMIT)
    elif arm in {"result_withheld", "current_feedback"}:
        env["CARNOT_ARC_INDUCE_TOOL_TURNS"] = "1"
    else:
        raise ValueError(f"unknown matched arm: {arm}")
    return env


def _request_bytes(row: Mapping[str, Any]) -> bytes | None:
    if isinstance(row.get("request_body"), str):
        return str(row["request_body"]).encode()
    path_text = row.get("request_path")
    if not path_text:
        return None
    try:
        return Path(str(path_text)).read_bytes()
    except OSError:
        return None


def _evidence_receipts(episode: Mapping[str, Any]) -> list[JsonDict]:
    receipts = episode.get("evidence_receipts")
    if isinstance(receipts, list):
        return [dict(row) for row in receipts if isinstance(row, Mapping)]
    receipt = episode.get("evidence_receipt")
    return [dict(receipt)] if isinstance(receipt, Mapping) else []


def _request_occurrences(body: bytes | None, text: str) -> int:
    if not body or not text:
        return 0
    try:
        payload = json.loads(body)
    except (UnicodeDecodeError, json.JSONDecodeError):
        return 0
    return sum(
        str(message.get("content") or "").count(text)
        for message in payload.get("messages", [])
        if isinstance(message, Mapping)
    )


def _causal_chain(episode: Mapping[str, Any]) -> JsonDict:
    """Join one result receipt to request bytes, trust, plan, and a later action."""

    requests = [
        dict(row) for row in episode.get("raw_request_manifest", []) if isinstance(row, Mapping)
    ]
    inductions = [
        dict(row) for row in episode.get("induction_rows", []) if isinstance(row, Mapping)
    ]
    consumptions = [
        dict(row) for row in episode.get("policy_consumption_rows", []) if isinstance(row, Mapping)
    ]
    for evidence_index, receipt in enumerate(_evidence_receipts(episode)):
        if receipt.get("enabled") is not True:
            continue
        attempt_id = receipt.get("attempt_id")
        for result in receipt.get("result_rows", []):
            if not isinstance(result, Mapping):
                continue
            next_id = str(result.get("next_request_id") or "")
            try:
                next_index = int(next_id.rsplit(":", 1)[1])
            except (IndexError, ValueError):
                next_index = -1
            next_request = next(
                (row for row in requests if int(row.get("call_index", -1)) == next_index), None
            )
            body = _request_bytes(next_request or {})
            bounded = str(result.get("bounded_response") or "")
            induction = inductions[evidence_index] if evidence_index < len(inductions) else {}
            trust = any(
                row.get("accepted_by_heldout_verifier") is True
                for row in induction.get("refinement_rounds", [])
                if isinstance(row, Mapping)
            )
            consumed = next(
                (
                    row
                    for row in consumptions
                    if int(row.get("attempt_index", -1)) == evidence_index
                    and row.get("policy_action_executed") is True
                ),
                None,
            )
            source_index = str(result.get("source_request_id") or "").rsplit(":", 1)[-1]
            source_request = next(
                (row for row in requests if str(row.get("call_index")) == source_index), None
            )
            same_identity = bool(
                attempt_id
                and result.get("attempt_id") == attempt_id
                and result.get("episode_id") == episode.get("game")
            )
            links = {
                "source_request": source_request is not None,
                "successful_tool_result": any(
                    item.get("ok") is True
                    for item in result.get("dispatch_results", [])
                    if isinstance(item, Mapping)
                ),
                "same_episode_and_attempt": same_identity,
                "next_request": next_request is not None,
                "result_in_next_request_once": _request_occurrences(body, bounded) == 1,
                "receipt_captured": result.get("receipt_captured") is True,
                "engine_trusted": trust,
                "plan_installed": induction.get("planned") is True,
                "later_policy_action": consumed is not None,
            }
            return {
                "episode_id": episode.get("episode_id"),
                "game": episode.get("game"),
                "arm": episode.get("arm"),
                "evidence_index": evidence_index,
                "attempt_id": attempt_id,
                "result_id": result.get("result_id"),
                "tool_names": deepcopy(result.get("tool_names") or []),
                "source_request_sha256": (source_request or {}).get("request_sha256"),
                "result_sha256": (
                    "sha256:" + hashlib.sha256(bounded.encode()).hexdigest() if bounded else None
                ),
                "next_request_sha256": (next_request or {}).get("request_sha256"),
                "engine_sha256": induction.get("engine_source_sha256"),
                "plan_sha256": (consumed or {}).get("plan_sha256"),
                "later_action": {
                    "action_index": (consumed or {}).get("action_index"),
                    "action": (consumed or {}).get("action"),
                    "data": deepcopy((consumed or {}).get("data")),
                },
                "links": links,
                "absent_links": [name for name, present in links.items() if not present],
                "chain_complete": all(links.values()),
            }
    return {
        "episode_id": episode.get("episode_id"),
        "game": episode.get("game"),
        "arm": episode.get("arm"),
        "links": {},
        "absent_links": ["successful_bound_result"],
        "chain_complete": False,
    }


def _summary_row(episode: Mapping[str, Any]) -> JsonDict:
    requests = [
        dict(row) for row in episode.get("raw_request_manifest", []) if isinstance(row, Mapping)
    ]
    action_rows = [dict(row) for row in episode.get("action_rows", []) if isinstance(row, Mapping)]
    request_hashes_valid = True
    for request in requests:
        body = _request_bytes(request)
        expected = request.get("request_sha256")
        if body is None or expected != "sha256:" + hashlib.sha256(body).hexdigest():
            request_hashes_valid = False
    raw_generated = sum(int(row.get("completion_tokens") or 0) for row in requests)
    raw_completed = sum(row.get("transport_completed") is True for row in requests)
    raw_actions = len(action_rows)
    raw_environment_actions = sum(row.get("action") != "RESET" for row in action_rows)
    action_receipts_valid = all("i" in row and "action" in row for row in action_rows)
    failures = [str(episode.get("error"))] if episode.get("error") else []
    terminal = episode.get("disposition") in {"complete", "censored_timeout"}
    factory = episode.get("factory_receipt") or {}
    authentic = bool(
        terminal
        and episode.get("fresh_store") is True
        and episode.get("adapter_disabled") is True
        and episode.get("banked_solution_disabled") is True
        and episode.get("off_path_engines_disabled") is True
        and factory.get("factory") == "make_carnot_agent"
        and factory.get("policy_class") == "E3AgentPolicy"
        and request_hashes_valid
        and action_receipts_valid
        and raw_environment_actions == int(episode.get("action_count") or 0)
        and len(requests) == int(episode.get("generation_calls_attempted") or 0)
        and raw_completed == int(episode.get("generation_calls_completed") or 0)
        and raw_generated == int(episode.get("generated_tokens") or 0)
    )
    within_budget = bool(
        raw_actions <= ACTION_LIMIT
        and len(requests) <= COMPLETION_LIMIT
        and raw_generated <= GENERATED_TOKEN_LIMIT
        and int(episode.get("action_limit") or 0) == ACTION_LIMIT
        and int(episode.get("completion_limit") or 0) == COMPLETION_LIMIT
        and int(episode.get("generated_token_limit") or 0) == GENERATED_TOKEN_LIMIT
    )
    return {
        "unit_id": episode.get("episode_id"),
        "game": episode.get("game"),
        "arm": episode.get("arm"),
        "seed": episode.get("seed"),
        "disposition": episode.get("disposition"),
        "censored": bool(episode.get("censored")),
        "authentic_terminal": authentic,
        "within_budget": within_budget,
        "request_hashes_valid": request_hashes_valid,
        "action_receipts_valid": action_receipts_valid,
        "raw_action_count": raw_actions,
        "raw_environment_action_count": raw_environment_actions,
        "raw_generation_calls_attempted": len(requests),
        "raw_generation_calls_completed": raw_completed,
        "raw_generated_tokens": raw_generated,
        "levels": int(episode.get("levels") or 0),
        "metrics": {"levels": int(episode.get("levels") or 0)},
        "costs": {
            "actions": raw_actions,
            "generation_calls": len(requests),
            "output_tokens": raw_generated,
            "wall_s": float((episode.get("compute_cost") or {}).get("wall_s") or 0.0),
        },
        "failures": failures,
        "abstentions": int(not episode.get("policy_consumption_rows")),
    }


def reduce_raw_panel(payload: Mapping[str, Any]) -> JsonDict:
    """Recompute all budgets and causal links from raw request and action records."""

    schedule = [dict(row) for row in payload.get("schedule", []) if isinstance(row, Mapping)]
    episodes = [dict(row) for row in payload.get("episodes", []) if isinstance(row, Mapping)]
    summaries = [_summary_row(row) for row in episodes]
    expected_ids = {row["episode_id"] for row in schedule}
    observed_ids = [row.get("unit_id") for row in summaries]
    capture = int(
        len(schedule) == 4
        and len(episodes) == 4
        and len(set(observed_ids)) == 4
        and set(observed_ids) == expected_ids
        and all(row["authentic_terminal"] and row["within_budget"] for row in summaries)
    )
    chain_rows = [_causal_chain(row) for row in episodes if row.get("arm") == "result_resume"]
    treatment_games = sorted(
        str(row.get("game")) for row in chain_rows if row.get("chain_complete") is True
    )
    controls_valid = True
    no_regression = True
    for game in TARGET_ROTATION:
        treatment = next(
            (
                row
                for row in summaries
                if row.get("game") == game and row.get("arm") == "result_resume"
            ),
            None,
        )
        control = next(
            (
                row
                for row in summaries
                if row.get("game") == game and row.get("arm") == "result_withheld"
            ),
            None,
        )
        raw_control = next(
            (
                row
                for row in episodes
                if row.get("game") == game and row.get("arm") == "result_withheld"
            ),
            {},
        )
        control_receipts = _evidence_receipts(raw_control)
        controls_valid = controls_valid and bool(
            control
            and control["authentic_terminal"]
            and control["within_budget"]
            and all(receipt.get("enabled") is not True for receipt in control_receipts)
        )
        no_regression = no_regression and bool(
            treatment
            and control
            and treatment["levels"] >= control["levels"]
            and treatment["raw_action_count"] <= control["raw_action_count"]
        )
    feedback = int(
        capture == 1
        and treatment_games == sorted(TARGET_ROTATION)
        and controls_valid
        and no_regression
    )
    return {
        "arc_capture_complete_score": capture,
        "arc_feedback_value_score": feedback,
        "planned_units": len(schedule),
        "attempted_units": len(episodes),
        "completed_units": sum(row["disposition"] == "complete" for row in summaries),
        "censored_units": sum(row["censored"] for row in summaries),
        "causal_treatment_games": treatment_games,
        "matched_controls_valid": controls_valid,
        "no_progress_or_action_regression": no_regression,
        "per_game_results": summaries,
        "causal_chain_rows": chain_rows,
    }


def independent_reduce(path: Path) -> JsonDict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        value = {}
    return reduce_raw_panel(value if isinstance(value, Mapping) else {})


def run_trace_mutations(payload: Mapping[str, Any]) -> JsonDict:
    """Prove that withheld and stale result traces cannot retain causal credit."""

    baseline = reduce_raw_panel(payload)
    withheld = deepcopy(dict(payload))
    withheld_episode = next(
        row for row in withheld.get("episodes", []) if row.get("arm") == "result_resume"
    )
    receipts = _evidence_receipts(withheld_episode)
    receipts[0]["enabled"] = False
    withheld_episode["evidence_receipt"] = receipts[0]
    withheld_episode.pop("evidence_receipts", None)
    stale = deepcopy(dict(payload))
    stale_episode = next(
        row for row in stale.get("episodes", []) if row.get("arm") == "result_resume"
    )
    receipts = _evidence_receipts(stale_episode)
    result_rows = receipts[0].setdefault("result_rows", [])
    if result_rows:
        result_rows[0]["attempt_id"] = "stale:attempt"
    else:
        result_rows.append(
            {
                "episode_id": stale_episode.get("game"),
                "attempt_id": "stale:attempt",
                "source_request_id": "request:0",
                "next_request_id": "request:1",
                "result_id": "stale:synthetic-mutation",
                "bounded_response": '<tool_response>{"ok": true}</tool_response>',
                "dispatch_results": [{"ok": True}],
                "tool_names": ["stale_result_mutation"],
                "delivery_count": 1,
                "receipt_captured": True,
            }
        )
    stale_episode["evidence_receipt"] = receipts[0]
    stale_episode.pop("evidence_receipts", None)
    withheld_score = reduce_raw_panel(withheld)["arc_feedback_value_score"]
    stale_score = reduce_raw_panel(stale)["arc_feedback_value_score"]
    return {
        "result_withheld_detected": withheld_score == 0,
        "stale_result_detected": stale_score == 0,
        "baseline_feedback_value_score": baseline["arc_feedback_value_score"],
        "result_withheld_feedback_value_score": withheld_score,
        "stale_result_feedback_value_score": stale_score,
    }


def _receipts_pass(receipts: Sequence[Mapping[str, Any]], names: Sequence[str]) -> bool:
    return all(
        sum(
            row.get("name") == name and row.get("passed") is True and row.get("exit_code") == 0
            for row in receipts
        )
        == 1
        for name in names
    )


def _acceptance_gates(
    preconditions: Sequence[Mapping[str, Any]],
    reduction: Mapping[str, Any],
    mutations: Mapping[str, Any],
    receipts: Sequence[Mapping[str, Any]],
    runtime_identity: Mapping[str, Any],
) -> list[JsonDict]:
    validations = _receipts_pass(
        receipts,
        (*REQUIRED_VALIDATION_NAMES, *REQUIRED_E2E_NAMES, *REQUIRED_TERMINAL_NAMES),
    )
    observations = (
        (
            "preconditions",
            True,
            all(row.get("passed") is True for row in preconditions),
            "Every producer and runtime check passes before dependent work.",
        ),
        (
            "authentic_terminal_panel",
            1,
            reduction.get("arc_capture_complete_score"),
            "All four matched episode rows are authentic, bounded, and terminal.",
        ),
        (
            "causal_feedback_value",
            1,
            reduction.get("arc_feedback_value_score"),
            "Both games need the full result-to-action chain and valid controls.",
        ),
        (
            "trace_mutations",
            True,
            bool(
                mutations.get("result_withheld_detected") and mutations.get("stale_result_detected")
            ),
            "Withheld and stale results must remove causal credit.",
        ),
        (
            "task_owned_cuda",
            True,
            runtime_identity.get("task_linked_cuda_execution") is True,
            "Live GPU mode needs task-owned process evidence on the leased device.",
        ),
        (
            "required_validation",
            True,
            validations,
            "Scoped, E2E, full-suite, reduction, and terminal checks all pass.",
        ),
    )
    return [
        {
            "check": name,
            "expected": expected,
            "observed": observed,
            "passed": observed == expected,
            "principle": principle,
        }
        for name, expected, observed, principle in observations
    ]


def _field_principles(artifact: Mapping[str, Any]) -> JsonDict:
    specific = {
        "schema": "Version the record and retain ordinary top-level experiment_id and milestone.",
        "status": "Write a terminal result only after actual work and affected checks.",
        "run_date": "Use 20260916 and record real UTC timestamps separately.",
        "preconditions_checked": "Record every actual input and resource check before dependent work.",
        "MODEL_SPECS": "List the intended unsloth/Qwen3.8-27B-GGUF Q4_K_M model identity.",
        "model_invoked": "True means a current model load or generation was attempted.",
        "invocation_counts": "Separate attempted, complete, failed, cancelled, and active operations.",
        "inference_substrate": "Declare current computation, never historical model receipts.",
        "inference_substrate_class": "Use the closed duration class that matches current work.",
        "execution_venue": "Use host because this task makes no board claim.",
        "duration_s": "Measure monotonic elapsed time without adding delay.",
        "phase_spans": "Measure disjoint load, generation, evaluation, test, and write work.",
        "random_seed": "Freeze development, evaluation, and resampling seeds before outcomes.",
        "reproducibility_checksum": "Bind code, settings, inputs, evaluator identity, and raw evidence.",
        "source_artifact_hashes": "Authenticate exact producers and same-milestone inputs.",
        "rows": "Keep each comparative unit, metric, cost, failure, and censoring disposition.",
        "sample_size_budget": "Record planned, attempted, complete, censored units and stopping rules.",
        "acceptance_gate_results": "Each gate records expected, observed, passed, and its principle.",
        "gate_check_summary": "Name the first upstream field mismatch with expected and observed values.",
        "verifier_is_oracle": "The executor defines correctness, so separate code does not remove circularity.",
        "honest_verdict": "Complete work starts complete_; external absence starts blocked_.",
        "verdict_class": "Use the closed terminal verdict enum.",
        "flagged_adversarial": "False requires current verification; a critical finding blocks value.",
        "validation_receipts": "Retain exact commands, scopes, exits, elapsed times, and log hashes.",
        "repository_health": "Keep dated unrelated failures separate from current required checks.",
        "field_principles": "Explain each field without wrapping its executable value.",
        "arc_capture_complete_score": "Complete live accounting includes null and censored episodes.",
        "arc_feedback_value_score": "Only the declared two-game causal pilot can pass this field.",
        "solve_provenance": "New credit would require live discovery and reproduction.",
        "per_game_results": "One row per matched game and arm keeps budgets and outcomes visible.",
        "causal_chain_rows": "A tool log alone cannot prove that a result affected a later action.",
        "runtime_identity_receipt": "Bind the served model, owned process, lease, and request evidence.",
    }
    return {
        key: specific.get(key, f"Retain the ordinary {key} evidence for this pilot.")
        for key in artifact
    }


def build_blocked_artifact(
    *,
    checks: Sequence[Mapping[str, Any]],
    started_at_utc: str,
    ended_at_utc: str,
    duration_s: float,
    source_hashes: Mapping[str, Any] | None = None,
) -> JsonDict:
    summary = gate_summary(checks)
    first = summary["first_failure"] or gate_check(
        "unknown_precondition", EXPERIMENT_ID, "state", "available", "unknown"
    )
    artifact: JsonDict = {
        "schema": SCHEMA,
        "status": f"blocked_{first['check']}",
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "run_date": RUN_DATE,
        "started_at_utc": started_at_utc,
        "ended_at_utc": ended_at_utc,
        "preconditions_checked": [deepcopy(dict(row)) for row in checks],
        "MODEL_SPECS": deepcopy(MODEL_SPECS),
        "resolved_model_specs": [],
        "model_invoked": False,
        "invocation_counts": deepcopy(ZERO_INVOCATION_COUNTS),
        "inference_substrate": "blocked_no_run",
        "inference_substrate_class": "blocked_no_run",
        "inference_mode": "not_run",
        "execution_venue": "host",
        "execution_host": socket.gethostname(),
        "duration_s": round(max(float(duration_s), 0.000001), 6),
        "phase_spans": [],
        "random_seed": {
            "development": DEVELOPMENT_SEED,
            "evaluation": EVALUATION_SEED,
            "resampling": RESAMPLING_SEED,
            "sealed_before_outcomes": True,
        },
        "reproducibility_checksum": "",
        "source_artifact_hashes": deepcopy(dict(source_hashes or {})),
        "rows": [],
        "sample_size_budget": {
            "planned_units": 4,
            "attempted_units": 0,
            "completed_units": 0,
            "censored_units": 4,
            "stopping_rule": "stop before model work on the first unsafe external precondition",
        },
        "acceptance_gate_results": [],
        "gate_check_summary": summary,
        "verifier_is_oracle": True,
        "honest_verdict": f"blocked_{first['check']}",
        "verdict_class": "blocked",
        "flagged_adversarial": True,
        "validation_receipts": [],
        "repository_health": {"status": "not_evaluated", "affects_required_checks": False},
        "field_principles": {},
        "arc_capture_complete_score": 0,
        "arc_feedback_value_score": 0,
        "promotion_value": 0,
        "solve_provenance": "no_run",
        "per_game_results": [],
        "causal_chain_rows": [],
        "selection_receipt": {},
        "trace_mutation_results": {},
        "runtime_identity_receipt": {},
        "production_default_changed": False,
        "competition_run_submitted": False,
        "population_improvement_claimed": False,
    }
    artifact["field_principles"] = _field_principles(artifact)
    artifact["reproducibility_checksum"] = artifact_checksum(artifact)
    return artifact


def build_terminal_artifact(
    *,
    preconditions: Sequence[Mapping[str, Any]],
    source_hashes: Mapping[str, Any],
    selection: Mapping[str, Any],
    raw_panel: Mapping[str, Any],
    reduction: Mapping[str, Any],
    mutations: Mapping[str, Any],
    runtime_identity: Mapping[str, Any],
    model_specs: Sequence[Mapping[str, Any]],
    invocation_counts: Mapping[str, Any],
    validation_receipts: Sequence[Mapping[str, Any]],
    repository_health: Mapping[str, Any],
    started_at_utc: str,
    ended_at_utc: str,
    duration_s: float,
    phase_spans: Sequence[Mapping[str, Any]],
) -> JsonDict:
    counts = deepcopy(dict(invocation_counts))
    model_invoked = int(counts.get("model_loads_attempted") or 0) > 0
    generation_attempted = int(counts.get("generation_calls_attempted") or 0) > 0
    substrate_class = (
        "model_bounded_generation" if generation_attempted else "model_load_no_generation"
    )
    gates = _acceptance_gates(
        preconditions, reduction, mutations, validation_receipts, runtime_identity
    )
    checks_passed = all(row["passed"] for row in gates)
    capture = int(checks_passed and reduction.get("arc_capture_complete_score") == 1)
    feedback = int(checks_passed and reduction.get("arc_feedback_value_score") == 1)
    verdict = "circular_positive" if feedback else "null"
    status = "complete" if checks_passed else "disqualified"
    if not checks_passed:
        verdict = "disqualified"
    artifact: JsonDict = {
        "schema": SCHEMA,
        "status": status,
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "run_date": RUN_DATE,
        "started_at_utc": started_at_utc,
        "ended_at_utc": ended_at_utc,
        "preconditions_checked": [deepcopy(dict(row)) for row in preconditions],
        "MODEL_SPECS": deepcopy(MODEL_SPECS),
        "resolved_model_specs": [deepcopy(dict(row)) for row in model_specs],
        "model_invoked": model_invoked,
        "invocation_counts": counts,
        "inference_substrate": "live_llm_inference",
        "inference_substrate_class": substrate_class,
        "inference_mode": (
            "live_gpu"
            if runtime_identity.get("task_linked_cuda_execution") is True
            else "not_verified"
        ),
        "execution_venue": "host",
        "execution_host": socket.gethostname(),
        "duration_s": round(max(float(duration_s), 0.000001), 6),
        "phase_spans": [deepcopy(dict(row)) for row in phase_spans],
        "random_seed": {
            "development": DEVELOPMENT_SEED,
            "evaluation": EVALUATION_SEED,
            "resampling": RESAMPLING_SEED,
            "sealed_before_outcomes": True,
        },
        "reproducibility_checksum": "",
        "source_artifact_hashes": deepcopy(dict(source_hashes)),
        "rows": deepcopy(reduction.get("per_game_results") or []),
        "sample_size_budget": {
            "planned_units": 4,
            "attempted_units": reduction.get("attempted_units"),
            "completed_units": reduction.get("completed_units"),
            "censored_units": reduction.get("censored_units"),
            "games": 2,
            "arms_per_game": 2,
            "action_limit_per_episode": ACTION_LIMIT,
            "completion_limit_per_episode": COMPLETION_LIMIT,
            "generated_token_limit_per_episode": GENERATED_TOKEN_LIMIT,
            "model_load_limit_s": MODEL_LOAD_LIMIT_S,
            "aggregate_live_limit_s": SESSION_LIMIT_S,
            "stopping_rule": "four sealed episodes or the first fixed resource ceiling; no outcome extension",
        },
        "acceptance_gate_results": gates,
        "gate_check_summary": gate_summary([*preconditions, *gates]),
        "verifier_is_oracle": True,
        "honest_verdict": (
            "complete_disqualified_required_check_failed"
            if not checks_passed
            else "complete_circular_positive_result_resume_chain_on_both_games"
            if feedback
            else "complete_null_result_resume_chain_or_non_regression_gate_not_met"
        ),
        "verdict_class": verdict,
        "flagged_adversarial": not checks_passed,
        "validation_receipts": [deepcopy(dict(row)) for row in validation_receipts],
        "repository_health": deepcopy(dict(repository_health)),
        "field_principles": {},
        "arc_capture_complete_score": capture,
        "arc_feedback_value_score": feedback,
        "promotion_value": 0,
        "solve_provenance": "live_agent_self_discovery",
        "per_game_results": deepcopy(reduction.get("per_game_results") or []),
        "causal_chain_rows": deepcopy(reduction.get("causal_chain_rows") or []),
        "selection_receipt": deepcopy(dict(selection)),
        "trace_mutation_results": deepcopy(dict(mutations)),
        "runtime_identity_receipt": deepcopy(dict(runtime_identity)),
        "independent_reduction": deepcopy(dict(reduction)),
        "raw_panel_identity": {
            "schema": "carnot.exp7354.raw_panel.v1",
            "sha256": "sha256:" + hashlib.sha256(_canonical_bytes(raw_panel)).hexdigest(),
            "episode_ids": [row.get("episode_id") for row in raw_panel.get("episodes", [])],
        },
        "production_default_changed": False,
        "competition_run_submitted": False,
        "population_improvement_claimed": False,
    }
    artifact["field_principles"] = _field_principles(artifact)
    artifact["reproducibility_checksum"] = artifact_checksum(artifact)
    return artifact


def validate_artifact(value: object) -> list[str]:
    if not isinstance(value, Mapping):
        return ["artifact_not_object"]
    artifact = dict(value)
    errors: list[str] = []
    if (artifact.get("schema"), artifact.get("experiment_id"), artifact.get("milestone")) != (
        SCHEMA,
        EXPERIMENT_ID,
        MILESTONE,
    ):
        errors.append("identity_mismatch")
    if artifact.get("run_date") != RUN_DATE:
        errors.append("run_date_mismatch")
    if artifact.get("execution_venue") != "host":
        errors.append("execution_venue_mismatch")
    if artifact.get("verdict_class") not in {
        "positive",
        "circular_positive",
        "null",
        "blocked",
        "disqualified",
        "partial",
    }:
        errors.append("verdict_class_invalid")
    blocked = artifact.get("verdict_class") == "blocked"
    prefix = "blocked_" if blocked else "complete_"
    if not str(artifact.get("honest_verdict") or "").startswith(prefix):
        errors.append("honest_verdict_prefix_invalid")
    if blocked:
        if (
            artifact.get("model_invoked") is not False
            or artifact.get("invocation_counts") != ZERO_INVOCATION_COUNTS
        ):
            errors.append("blocked_invocation_mismatch")
        if artifact.get("inference_substrate_class") != "blocked_no_run":
            errors.append("blocked_substrate_mismatch")
        if (artifact.get("gate_check_summary") or {}).get("first_failure") is None:
            errors.append("blocked_gate_summary_missing")
    else:
        counts = artifact.get("invocation_counts") or {}
        if artifact.get("model_invoked") is not (int(counts.get("model_loads_attempted") or 0) > 0):
            errors.append("model_invoked_mismatch")
        expected_class = (
            "model_bounded_generation"
            if int(counts.get("generation_calls_attempted") or 0) > 0
            else "model_load_no_generation"
        )
        if artifact.get("inference_substrate_class") != expected_class:
            errors.append("inference_substrate_class_mismatch")
        floor = 10.0 if expected_class == "model_bounded_generation" else 2.0
        if float(artifact.get("duration_s") or 0.0) < floor:
            errors.append("duration_floor_failed")
        if (
            artifact.get("inference_mode") == "live_gpu"
            and (artifact.get("runtime_identity_receipt") or {}).get("task_linked_cuda_execution")
            is not True
        ):
            errors.append("live_gpu_receipt_missing")
    if artifact.get("verdict_class") in {"blocked", "disqualified"} and any(
        artifact.get(field) != 0
        for field in ("arc_capture_complete_score", "arc_feedback_value_score", "promotion_value")
    ):
        errors.append("unsafe_scores_on_failed_artifact")
    principles = artifact.get("field_principles")
    if not isinstance(principles, Mapping) or set(principles) != set(artifact):
        errors.append("field_principles_incomplete")
    if (
        artifact.get("production_default_changed") is not False
        or artifact.get("competition_run_submitted") is not False
    ):
        errors.append("forbidden_state_change")
    if artifact.get("reproducibility_checksum") != artifact_checksum(artifact):
        errors.append("reproducibility_checksum_mismatch")
    return list(dict.fromkeys(errors))


def run_scoped_validation(root: Path, private: Path) -> list[JsonDict]:  # pragma: no cover
    basetemp = private / "scoped"
    coverage_file = private / "coverage" / ".coverage"
    basetemp.mkdir(parents=True, exist_ok=True)
    coverage_file.parent.mkdir(parents=True, exist_ok=True)
    outcome = validation_scope.run_scoped_validation(
        root,
        test_paths=[str(TEST_PATH)],
        changed_modules=[str(MODULE_PATH)],
        static_paths=[str(WRAPPER_PATH)],
        basetemp=basetemp,
        coverage_file=coverage_file,
        log_dir=root / RAW_DIR / "validation/scoped",
        historical_failures=[],
    )
    return [dict(row) for row in outcome.get("validation_receipts", [])]


def e2e_command_specs(root: Path, private: Path) -> list[validation_scope.CommandSpec]:
    python = str(root / ".venv/bin/python")
    pytest = str(root / ".venv/bin/pytest")
    return [
        validation_scope.CommandSpec(
            "e2e_009",
            (
                pytest,
                "-n",
                "0",
                "-o",
                "addopts=",
                "--no-cov",
                f"--basetemp={private / 'e2e009'}",
                "tests/python/test_arc_induction_state_persistence.py",
                "-q",
            ),
            "E2E-009",
        ),
        validation_scope.CommandSpec(
            "e2e_010",
            (
                pytest,
                "-n",
                "0",
                "-o",
                "addopts=",
                "--no-cov",
                f"--basetemp={private / 'e2e010'}",
                "tests/python/test_arc_tool_grammar_transport.py",
                "-q",
            ),
            "E2E-010",
        ),
        validation_scope.CommandSpec(
            "e2e_offline_smoke",
            (
                "/usr/bin/env",
                "CARNOT_ARC_DISABLE_INDUCTION=1",
                python,
                "-u",
                "scripts/arc_loop_solve.py",
                "--mechanism",
                "e3",
                "--game",
                "r11l",
                "--max-actions",
                "12",
                "--output",
                str(private / "offline-smoke.json"),
            ),
            "E2E-009 offline twin",
        ),
        validation_scope.CommandSpec(
            "full_python_suite",
            (pytest, "tests/python", "-q"),
            "mandatory full Python suite",
            timeout_s=1800.0,
        ),
    ]


def terminal_command_specs(
    root: Path, candidate: Path, raw_path: Path | None = None
) -> list[validation_scope.CommandSpec]:
    python = str(root / ".venv/bin/python")
    raw = raw_path or root / RAW_PANEL_PATH
    return [
        validation_scope.CommandSpec(
            "independent_reducer",
            (python, "-u", str(root / WRAPPER_PATH), "--reduce-raw", str(raw)),
            "raw episode and request records",
        ),
        validation_scope.CommandSpec(
            "adversarial_verify",
            (python, "-u", "scripts/adversarial_verify.py", str(candidate)),
            "measured terminal candidate",
        ),
        validation_scope.CommandSpec(
            "verdict_row_consistency_strict",
            (
                python,
                "-u",
                "scripts/verdict_row_consistency_lint.py",
                "--strict",
                str(candidate),
            ),
            "measured terminal candidate",
        ),
    ]


def progress(started: float, phase: str, event: str, **fields: Any) -> None:
    print(
        " ".join(
            (
                f"phase={phase}",
                f"event={event}",
                f"elapsed_s={max(0.0, time.monotonic() - started):.3f}",
                *(f"{key}={value}" for key, value in sorted(fields.items())),
            )
        ),
        flush=True,
    )


def _utc_now() -> str:  # pragma: no cover
    return datetime.now(UTC).isoformat()


def _load_json(path: Path) -> JsonDict | None:  # pragma: no cover
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return dict(value) if isinstance(value, Mapping) else None


def _static_preconditions(
    root: Path,
) -> tuple[list[JsonDict], JsonDict, JsonDict | None]:  # pragma: no cover
    checks: list[JsonDict] = []
    hashes: JsonDict = {}
    required = (
        Path("AGENTS.md"),
        Path("CODEX.md"),
        Path("CLAUDE.md"),
        Path("research-program.md"),
        Path("research-references.md"),
        EXCLUSION_PATH,
        Path("ops/e2e-test-plan.md"),
        REGISTRY_PATH,
        SPEC_PATH,
        EXP7345_PATH,
        Path("python/carnot/agentic/arc_competition_agent.py"),
        Path("python/carnot/agentic/arc_induction_tool_loop.py"),
        Path("python/carnot/agentic/arc_selfparse_result_resume.py"),
        Path("python/carnot/reporting/experiment_7303_validation_scope.py"),
        MODULE_PATH,
        WRAPPER_PATH,
        TEST_PATH,
    )
    for relative in required:
        path = root / relative
        checks.append(
            gate_check("required_input", relative.as_posix(), "exists", True, path.is_file())
        )
        if path.is_file():
            hashes[relative.as_posix()] = {
                "sha256": sha256_file(path),
                "role": "current_input",
                "authorizes_readiness": relative == EXP7345_PATH,
            }
    spec = (root / SPEC_PATH).read_text(encoding="utf-8") if (root / SPEC_PATH).is_file() else ""
    checks.append(
        gate_check(
            "driving_requirement",
            SPEC_PATH.as_posix(),
            "REQ-ARC-WMTE-7354",
            True,
            "## REQ-ARC-WMTE-7354:" in spec,
        )
    )
    upstream = _load_json(root / EXP7345_PATH)
    dependency = check_dependency(upstream)
    checks.extend(dependency["checks"])
    try:
        manifest = yaml.safe_load((root / EXCLUSION_PATH).read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError):
        manifest = {}
    checks.append(
        gate_check(
            "exclusion_manifest",
            EXCLUSION_PATH.as_posix(),
            "exp7354_not_retired",
            False,
            live_base._manifest_rejects(manifest, EXPERIMENT_ID),
        )
    )
    return checks, hashes, upstream


def _runtime_preconditions(
    root: Path, *, gpu_wait_s: float, started: float
) -> tuple[list[JsonDict], JsonDict, JsonDict]:  # pragma: no cover
    checks: list[JsonDict] = []
    hashes: JsonDict = {}
    idle: list[JsonDict] = []
    deadline = time.monotonic() + max(0.0, gpu_wait_s)
    while True:
        idle = [
            row
            for row in live_base._gpu_inventory()
            if not row.get("compute_apps") and int(row.get("free_memory_mb") or 0) >= 20_000
        ]
        if idle or time.monotonic() >= deadline:
            break
        progress(
            started,
            "runtime_preflight",
            "pending_idle_gpu",
            remaining_s=round(deadline - time.monotonic(), 1),
        )
        time.sleep(min(30.0, max(0.0, deadline - time.monotonic())))
    gpu = idle[0] if idle else None
    checks.append(
        gate_check("gpu_reservation", "nvidia-smi", "idle_gpu_with_20GB", True, gpu is not None)
    )

    from carnot.inference.sota_models import cached_current_model, gguf_tokenizer_loadable

    model = cached_current_model(
        gpu_index=int(gpu.get("index", 0)) if gpu else 0, preferred_quant=QUANTIZATION
    )
    model_path = Path(str(model.get("model_path"))) if model else None
    model_ok = bool(
        model
        and model.get("hf_id") == MODEL_ID
        and model_path
        and model_path.is_file()
        and QUANTIZATION in model_path.name
    )
    checks.append(gate_check("cached_current_model", MODEL_ID, "Q4_K_M_path", True, model_ok))
    progress(started, "runtime_preflight", "before_tokenizer_load")
    tokenizer_ok, tokenizer_detail = gguf_tokenizer_loadable(str(model_path) if model_ok else None)
    progress(started, "runtime_preflight", "after_tokenizer_load", passed=tokenizer_ok)
    checks.append(
        gate_check(
            "embedded_tokenizer",
            str(model_path),
            "loadable",
            True,
            tokenizer_ok,
            tokenizer_detail,
        )
    )
    server_candidates = (
        os.environ.get("CARNOT_LLAMA_SERVER"),
        str(Path.home() / ".cache/llama.cpp-master/build/bin/llama-server"),
    )
    server = next(
        (Path(value) for value in server_candidates if value and Path(value).is_file()), None
    )
    checks.append(
        gate_check("native_runtime", "llama.cpp", "server_binary", True, server is not None)
    )
    model_hash = None
    if model_ok and model_path is not None:
        progress(started, "runtime_preflight", "before_model_hash", path=model_path)
        model_hash = sha256_file(model_path)
        progress(started, "runtime_preflight", "after_model_hash", sha256=model_hash)
        hashes[str(model_path)] = {
            "sha256": model_hash,
            "bytes": model_path.stat().st_size,
            "role": "current_model",
            "authorizes_readiness": True,
        }
    if server is not None:
        hashes[str(server)] = {
            "sha256": sha256_file(server),
            "bytes": server.stat().st_size,
            "role": "native_runtime_binary",
            "authorizes_readiness": True,
        }
    resolved = {
        **deepcopy(dict(model or {})),
        "hf_id": MODEL_ID,
        "quantization": QUANTIZATION,
        "model_path": str(model_path) if model_path else None,
        "sha256": model_hash,
        "bytes": model_path.stat().st_size if model_ok and model_path else None,
        "runtime_settings": {
            "runner": "LocalGGUFProposer_native_llama.cpp",
            "context_tokens": 49152,
            "kv_quantization": "q8_0",
            "offload_layers_requested": 999,
            "completion_limit_per_episode": COMPLETION_LIMIT,
            "generated_token_limit_per_episode": GENERATED_TOKEN_LIMIT,
            "tokens_per_call": TOKENS_PER_CALL,
        },
    }
    return (
        checks,
        hashes,
        {
            "gpu": gpu,
            "model_path": str(model_path) if model_ok and model_path else None,
            "model_hash": model_hash,
            "model_spec": resolved,
            "server": str(server) if server else None,
        },
    )


@contextmanager
def _configured_runtime() -> Any:  # pragma: no cover
    updates = {
        "RUN_DATE": RUN_DATE,
        "MILESTONE": MILESTONE,
        "EXPERIMENT_ID": EXPERIMENT_ID,
        "MODEL_SPECS": MODEL_SPECS,
        "TARGET_GAME": "matched_panel",
        "DEVELOPMENT_SEED": DEVELOPMENT_SEED,
        "EVALUATION_SEED": EVALUATION_SEED,
        "ACTION_LIMIT": ACTION_LIMIT,
        "COMPLETION_LIMIT": COMPLETION_LIMIT,
        "GENERATED_TOKEN_LIMIT": GENERATED_TOKEN_LIMIT,
        "TOKENS_PER_CALL": TOKENS_PER_CALL,
        "SESSION_LIMIT_S": SESSION_LIMIT_S,
        "MODEL_LOAD_LIMIT_S": MODEL_LOAD_LIMIT_S,
        "RESULT_PATH": RESULT_PATH,
        "RAW_DIR": RAW_DIR,
        "CHECKPOINT_PATH": CHECKPOINT_PATH,
        "BOUNDARY_PATH": RAW_DIR / "receipt_events.jsonl",
        "TOOL_EVENT_PATH": EVIDENCE_PATH,
        "SESSION_PATH": SESSION_PATH,
        "RAW_ROW_PATH": RAW_PANEL_PATH,
        "TERMINAL_CANDIDATE_PATH": TERMINAL_CANDIDATE_PATH,
        "MODULE_PATH": MODULE_PATH,
        "WRAPPER_PATH": WRAPPER_PATH,
        "TEST_PATH": TEST_PATH,
        "session_environment": session_environment,
    }
    old = {name: getattr(live_base, name) for name in updates}
    try:
        for name, value in updates.items():
            setattr(live_base, name, value)
        with live_base._configured_runtime() as reused:
            yield reused
    finally:
        for name, value in old.items():
            setattr(live_base, name, value)


class _EvidenceSink(list[JsonDict]):  # pragma: no cover
    """Flush each tool result and final resume receipt before later work starts."""

    def __init__(self, path: Path) -> None:
        super().__init__()
        self.path = path
        self.episode_id = "unassigned"
        self.induction_index = 0

    def _write(self, row: Mapping[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        descriptor = os.open(self.path, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o600)
        try:
            os.write(descriptor, _canonical_bytes(row) + b"\n")
            os.fsync(descriptor)
        finally:
            os.close(descriptor)

    def append(self, value: JsonDict) -> None:
        row = {
            "kind": "tool_event",
            "episode_id": self.episode_id,
            "induction_index": self.induction_index,
            **deepcopy(value),
        }
        super().append(row)
        self._write(row)

    def record_attempt(self, stats: Mapping[str, Any]) -> None:
        self._write(
            {
                "kind": "attempt_summary",
                "episode_id": self.episode_id,
                "induction_index": self.induction_index,
                "result_resume": deepcopy(stats.get("result_resume") or {"enabled": False}),
                "turns": stats.get("turns"),
                "decode_tokens_total": stats.get("decode_tokens_total"),
                "terminated_by": stats.get("terminated_by"),
                "tool_calls_total": stats.get("tool_calls_total"),
            }
        )


def run_live_session(args: argparse.Namespace) -> int:  # pragma: no cover
    from carnot.agentic import arc_induction_tool_loop as tool_loop

    sink = _EvidenceSink(Path(args.raw_dir) / EVIDENCE_PATH.name)
    original = tool_loop.induce_with_tool_loop
    counters: dict[str, int] = {}

    def instrumented(*call_args: Any, **kwargs: Any) -> Any:
        episode_id = os.environ.get("CARNOT_7354_EPISODE_ID", "unassigned")
        sink.episode_id = episode_id
        sink.induction_index = counters.get(episode_id, 0)
        kwargs["tool_event_sink"] = sink
        try:
            return original(*call_args, **kwargs)
        finally:
            proposer = call_args[0] if call_args else None
            stats = getattr(proposer, "last_tool_loop_stats", {}) or {}
            sink.record_attempt(stats)
            counters[episode_id] = sink.induction_index + 1

    tool_loop.induce_with_tool_loop = instrumented
    try:
        with _configured_runtime() as reused:
            return int(reused.run_live_session(args))
    finally:
        tool_loop.induce_with_tool_loop = original


def _read_evidence(path: Path) -> list[JsonDict]:  # pragma: no cover
    if not path.is_file():
        return []
    rows: list[JsonDict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, Mapping):
            rows.append(dict(value))
    return rows


def _complete_episode_rows(
    schedule: Sequence[Mapping[str, Any]],
    session: Mapping[str, Any],
    evidence: Sequence[Mapping[str, Any]],
) -> list[JsonDict]:  # pragma: no cover
    existing = {
        str(row.get("episode_id")): deepcopy(dict(row))
        for row in session.get("episodes", [])
        if isinstance(row, Mapping)
    }
    rows: list[JsonDict] = []
    for scheduled in schedule:
        episode_id = str(scheduled["episode_id"])
        row = existing.get(
            episode_id,
            {
                **deepcopy(dict(scheduled)),
                "disposition": "censored_timeout",
                "censored": True,
                "model_invoked": False,
                "model_loaded": bool(session.get("model_loaded")),
                "generation_calls_attempted": 0,
                "generation_calls_completed": 0,
                "generated_tokens": 0,
                "action_count": 0,
                "levels": 0,
                "induction_rows": [],
                "policy_consumption_rows": [],
                "action_rows": [],
                "raw_request_manifest": [],
                "factory_receipt": {
                    "factory": "make_carnot_agent",
                    "policy_class": "E3AgentPolicy",
                    "adapter_disabled": True,
                    "denied_inputs": ["banked_solutions", "game_adapter", "game_source"],
                },
                "error": "aggregate_live_window_timeout",
            },
        )
        row.update(
            {
                "action_limit": ACTION_LIMIT,
                "completion_limit": COMPLETION_LIMIT,
                "generated_token_limit": GENERATED_TOKEN_LIMIT,
                "fresh_store": True,
                "adapter_disabled": True,
                "banked_solution_disabled": True,
                "off_path_engines_disabled": True,
            }
        )
        row["evidence_receipts"] = [
            deepcopy(dict(item.get("result_resume") or {}))
            for item in evidence
            if item.get("kind") == "attempt_summary" and item.get("episode_id") == episode_id
        ]
        rows.append(row)
    return rows


def load_completed_live_checkpoint(
    path: Path, schedule: Sequence[Mapping[str, Any]]
) -> JsonDict | None:
    """Return only a fully authenticated live window for this exact sealed schedule."""

    value = _load_json(path)
    if value is None:
        return None
    episodes = [row for row in value.get("episodes", []) if isinstance(row, Mapping)]
    runtime = value.get("runtime_receipt") or {}
    model = value.get("model_spec") or {}
    identity_fields = (
        "episode_id",
        "game",
        "arm",
        "seed",
        "action_limit",
        "completion_limit",
        "generated_token_limit",
    )
    identities_match = len(episodes) == len(schedule) == 4 and all(
        all(episode.get(field) == sealed.get(field) for field in identity_fields)
        for episode, sealed in zip(episodes, schedule, strict=True)
    )
    complete = bool(
        identities_match
        and value.get("model_loaded") is True
        and value.get("timed_out") is False
        and not value.get("error")
        and model.get("hf_id") == MODEL_ID
        and model.get("quantization") == QUANTIZATION
        and runtime.get("task_linked_cuda_execution") is True
        and runtime.get("identity_authentication_valid") is True
        and all(
            episode.get("disposition") == "complete"
            and episode.get("censored") is False
            and not episode.get("error")
            and 0 < int(episode.get("generation_calls_attempted") or 0) <= COMPLETION_LIMIT
            for episode in episodes
        )
    )
    return deepcopy(value) if complete else None


def _invocation_counts(
    session: Mapping[str, Any], episodes: Sequence[Mapping[str, Any]]
) -> JsonDict:  # pragma: no cover
    requests = [
        row
        for episode in episodes
        for row in episode.get("raw_request_manifest", [])
        if isinstance(row, Mapping)
    ]
    timed_out = bool(session.get("timed_out"))
    load_complete = bool(session.get("model_loaded"))
    return {
        "model_loads_attempted": 1,
        "model_loads_completed": int(load_complete),
        "model_loads_failed": int(not load_complete and not timed_out),
        "model_loads_cancelled": int(not load_complete and timed_out),
        "model_loads_in_flight": 0,
        "generation_calls_attempted": len(requests),
        "generation_calls_completed": sum(
            row.get("transport_completed") is True for row in requests
        ),
        "generation_calls_failed": sum(bool(row.get("error")) for row in requests),
        "generation_calls_cancelled": 0,
        "generation_calls_in_flight": 0,
        "usable_answers": sum(row.get("usable_answer") is True for row in requests),
    }


def _phase(name: str, started: float, origin: float, units: int) -> JsonDict:  # pragma: no cover
    return {
        "phase": name,
        "started_elapsed_s": round(started - origin, 6),
        "duration_s": round(time.monotonic() - started, 6),
        "completed_units": units,
    }


def run_experiment(args: argparse.Namespace) -> JsonDict:  # pragma: no cover
    started = time.monotonic()
    started_utc = _utc_now()
    phases: list[JsonDict] = []
    progress(started, "startup", "paths_and_date_authenticated", run_date=args.date)
    phase_start = time.monotonic()
    progress(started, "preconditions", "before_static_checks")
    checks, hashes, _upstream = _static_preconditions(REPO_ROOT)
    phases.append(_phase("static_preconditions", phase_start, started, len(checks)))
    progress(
        started, "preconditions", "after_static_checks", passed=all(row["passed"] for row in checks)
    )
    if not all(row["passed"] for row in checks):
        artifact = build_blocked_artifact(
            checks=checks,
            source_hashes=hashes,
            started_at_utc=started_utc,
            ended_at_utc=_utc_now(),
            duration_s=time.monotonic() - started,
        )
        errors = validate_artifact(artifact)
        if errors:
            raise RuntimeError(f"blocked artifact validation failed: {errors}")
        atomic_write(REPO_ROOT / RESULT_PATH, artifact)
        progress(started, "write", "terminal_blocked_artifact_written", path=RESULT_PATH)
        return artifact

    phase_start = time.monotonic()
    progress(started, "runtime_preflight", "before_runner_resolution")
    runtime_checks, runtime_hashes, resources = _runtime_preconditions(
        REPO_ROOT, gpu_wait_s=args.gpu_wait_s, started=started
    )
    checks.extend(runtime_checks)
    hashes.update(runtime_hashes)
    phases.append(_phase("runtime_preconditions", phase_start, started, len(runtime_checks)))
    progress(
        started,
        "runtime_preflight",
        "after_runner_resolution",
        passed=all(row["passed"] for row in checks),
    )
    if not all(row["passed"] for row in checks):
        artifact = build_blocked_artifact(
            checks=checks,
            source_hashes=hashes,
            started_at_utc=started_utc,
            ended_at_utc=_utc_now(),
            duration_s=time.monotonic() - started,
        )
        atomic_write(REPO_ROOT / RESULT_PATH, artifact)
        progress(started, "write", "terminal_runtime_block_written", path=RESULT_PATH)
        return artifact

    phase_start = time.monotonic()
    progress(started, "selection", "before_registry_precheck")
    registry = yaml.safe_load((REPO_ROOT / REGISTRY_PATH).read_text(encoding="utf-8")) or {}
    from carnot.agentic.arc_game_adapters import adaptered_games

    selection = freeze_panel(registry, adaptered_games=set(adaptered_games()))
    selection_gate = gate_check(
        "registry_precheck_and_freeze",
        REGISTRY_PATH.as_posix(),
        "two_eligible_games",
        True,
        selection["passed"],
    )
    checks.append(selection_gate)
    schedule = matched_schedule(selection["games"])
    atomic_write(REPO_ROOT / SCHEDULE_PATH, {"selection_receipt": selection, "rows": schedule})
    phases.append(_phase("selection", phase_start, started, len(schedule)))
    progress(
        started,
        "selection",
        "after_registry_precheck",
        games=selection["games"],
        passed=selection["passed"],
    )
    if not selection["passed"]:
        artifact = build_blocked_artifact(
            checks=checks,
            source_hashes=hashes,
            started_at_utc=started_utc,
            ended_at_utc=_utc_now(),
            duration_s=time.monotonic() - started,
        )
        atomic_write(REPO_ROOT / RESULT_PATH, artifact)
        return artifact

    phase_start = time.monotonic()
    progress(started, "live_window", "before_model_load_generation_benchmark", planned_units=4)
    session = load_completed_live_checkpoint(REPO_ROOT / SESSION_PATH, schedule)
    if session is None:
        with _configured_runtime() as reused:
            session = reused.run_child_with_lease(
                resources=resources,
                schedule_path=REPO_ROOT / SCHEDULE_PATH,
                raw_dir=REPO_ROOT / RAW_DIR,
                checkpoint_path=REPO_ROOT / CHECKPOINT_PATH,
                session_path=REPO_ROOT / SESSION_PATH,
                remaining_s=SESSION_LIMIT_S,
            )
    else:
        progress(
            started,
            "live_window",
            "authenticated_completed_checkpoint_reused",
            completed_units=len(session["episodes"]),
        )
    phases.append(
        _phase(
            "model_load_and_generation", phase_start, started, len(session.get("episodes") or [])
        )
    )
    progress(
        started,
        "live_window",
        "after_model_load_generation_benchmark",
        completed_units=len(session.get("episodes") or []),
    )

    phase_start = time.monotonic()
    evidence = _read_evidence(REPO_ROOT / EVIDENCE_PATH)
    episodes = _complete_episode_rows(schedule, session, evidence)
    raw_panel = {
        "schema": "carnot.exp7354.raw_panel.v1",
        "schedule": schedule,
        "episodes": episodes,
        "evidence_rows": evidence,
    }
    atomic_write(REPO_ROOT / RAW_PANEL_PATH, raw_panel)
    reduction = reduce_raw_panel(raw_panel)
    mutations = run_trace_mutations(raw_panel)
    phases.append(_phase("evaluation", phase_start, started, len(episodes)))
    progress(
        started,
        "evaluation",
        "independent_reduction_complete",
        capture=reduction["arc_capture_complete_score"],
        value=reduction["arc_feedback_value_score"],
    )

    for path in sorted(item for item in (REPO_ROOT / RAW_DIR).rglob("*") if item.is_file()):
        hashes[str(path.relative_to(REPO_ROOT))] = {
            "sha256": sha256_file(path),
            "bytes": path.stat().st_size,
            "role": "current_raw_evidence",
            "authorizes_readiness": False,
        }

    private = Path(tempfile.mkdtemp(prefix="exp7354-validation-", dir="/tmp"))
    for parent in (private / "e2e009", private / "e2e010"):
        parent.mkdir(parents=True, exist_ok=True)
    phase_start = time.monotonic()
    progress(started, "validation", "before_scoped_checks")
    scoped = run_scoped_validation(REPO_ROOT, private)
    progress(started, "validation", "after_scoped_checks", completed_units=len(scoped))
    progress(started, "e2e", "before_affected_checks")
    e2e = validation_scope.run_commands(
        REPO_ROOT,
        e2e_command_specs(REPO_ROOT, private),
        log_dir=REPO_ROOT / RAW_DIR / "validation/e2e",
        heartbeat_s=60.0,
    )
    progress(started, "e2e", "after_affected_checks", completed_units=len(e2e))
    receipts = [*scoped, *e2e]
    phases.append(_phase("tests", phase_start, started, len(receipts)))

    runtime = deepcopy(dict(session.get("runtime_receipt") or {}))
    runtime["cumulative_ledger_identity"] = {
        "episode_ids": [row["episode_id"] for row in episodes],
        "request_sha256s": [
            request.get("request_sha256")
            for row in episodes
            for request in row.get("raw_request_manifest", [])
            if isinstance(request, Mapping)
        ],
        "raw_panel_sha256": "sha256:" + hashlib.sha256(_canonical_bytes(raw_panel)).hexdigest(),
    }
    counts = _invocation_counts(session, episodes)
    repository_health = validation_scope.build_repository_health([])
    candidate = build_terminal_artifact(
        preconditions=checks,
        source_hashes=hashes,
        selection=selection,
        raw_panel=raw_panel,
        reduction=reduction,
        mutations=mutations,
        runtime_identity=runtime,
        model_specs=[resources["model_spec"]],
        invocation_counts=counts,
        validation_receipts=receipts,
        repository_health=repository_health,
        started_at_utc=started_utc,
        ended_at_utc=_utc_now(),
        duration_s=time.monotonic() - started,
        phase_spans=phases,
    )
    atomic_write(REPO_ROOT / TERMINAL_CANDIDATE_PATH, candidate)

    phase_start = time.monotonic()
    progress(started, "terminal_validation", "before_independent_and_strict_checks")
    terminal = validation_scope.run_commands(
        REPO_ROOT,
        terminal_command_specs(
            REPO_ROOT, REPO_ROOT / TERMINAL_CANDIDATE_PATH, REPO_ROOT / RAW_PANEL_PATH
        ),
        log_dir=REPO_ROOT / RAW_DIR / "validation/terminal",
        heartbeat_s=60.0,
    )
    receipts.extend(terminal)
    phases.append(_phase("terminal_validation", phase_start, started, len(terminal)))
    progress(
        started,
        "terminal_validation",
        "after_independent_and_strict_checks",
        completed_units=len(terminal),
    )

    artifact = build_terminal_artifact(
        preconditions=checks,
        source_hashes=hashes,
        selection=selection,
        raw_panel=raw_panel,
        reduction=reduction,
        mutations=mutations,
        runtime_identity=runtime,
        model_specs=[resources["model_spec"]],
        invocation_counts=counts,
        validation_receipts=receipts,
        repository_health=repository_health,
        started_at_utc=started_utc,
        ended_at_utc=_utc_now(),
        duration_s=time.monotonic() - started,
        phase_spans=phases,
    )
    errors = validate_artifact(artifact)
    if errors:
        raise RuntimeError(f"terminal artifact validation failed: {errors}")
    progress(started, "write", "before_atomic_terminal_write", path=RESULT_PATH)
    atomic_write(REPO_ROOT / TERMINAL_CANDIDATE_PATH, artifact)
    atomic_write(REPO_ROOT / RESULT_PATH, artifact)
    atomic_write(
        REPO_ROOT / CHECKPOINT_PATH,
        {
            "status": "complete",
            "completed_units": 4,
            "result_path": RESULT_PATH.as_posix(),
            "arc_capture_complete_score": artifact["arc_capture_complete_score"],
            "arc_feedback_value_score": artifact["arc_feedback_value_score"],
        },
    )
    progress(started, "write", "after_atomic_terminal_write", path=RESULT_PATH)
    return artifact


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:  # pragma: no cover
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", default=RUN_DATE)
    parser.add_argument("--role", choices=("experiment", "live-session"), default="experiment")
    parser.add_argument("--reduce-raw", type=Path)
    parser.add_argument("--model-path")
    parser.add_argument("--model-hash")
    parser.add_argument("--gpu-index", type=int)
    parser.add_argument("--port", type=int)
    parser.add_argument("--schedule-path", type=Path)
    parser.add_argument("--raw-dir", type=Path, default=RAW_DIR)
    parser.add_argument("--checkpoint-path", type=Path, default=CHECKPOINT_PATH)
    parser.add_argument("--session-path", type=Path, default=SESSION_PATH)
    parser.add_argument("--gpu-wait-s", type=float, default=120.0)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover
    started = time.monotonic()
    progress(started, "startup", "entrypoint")
    args = parse_args(argv)
    if args.reduce_raw is not None:
        print(json.dumps(independent_reduce(args.reduce_raw), sort_keys=True), flush=True)
        return 0
    if args.date != RUN_DATE:
        raise SystemExit(f"--date must be {RUN_DATE}")
    if args.role == "live-session":
        return run_live_session(args)
    run_experiment(args)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
