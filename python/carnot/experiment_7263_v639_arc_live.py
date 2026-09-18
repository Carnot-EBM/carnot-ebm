"""Run the V639 live ARC transition-witness case study.

The experiment uses the submitted policy factory and one cached native llama.cpp
server. Small pure helpers own selection, reduction, and terminal validation so the
live run cannot silently turn missing evidence into a positive result.

Spec: REQ-ARC-WMTE-7263 and SCENARIO-ARC-WMTE-7263-*.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from copy import deepcopy
from datetime import UTC, datetime
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
from typing import Any

import numpy as np
import yaml

from carnot import experiment_7234_v637_arc_scored_dryrun as scored


JsonDict = dict[str, Any]
REPO_ROOT = Path(__file__).resolve().parents[2]
TASK_ID = "exp7263-arc-live"
EXPERIMENT_ID = TASK_ID
MILESTONE = "2026.09.639"
RUN_DATE = "20260913"
RANDOM_SEED = 7_263_001
MODEL_ID = "unsloth/Qwen3.8-27B-GGUF"
QUANTIZATION = "Q4_K_M"
MODEL_SPECS = ({"hf_id": MODEL_ID, "quantization": QUANTIZATION},)
SCHEMA = "carnot.experiment_7263.v639.arc_live.v1"

ACTION_LIMIT = 192
COMPLETION_LIMIT = 2
GENERATED_TOKEN_LIMIT = 4096
TOKENS_PER_CALL = GENERATED_TOKEN_LIMIT // COMPLETION_LIMIT
LIVE_WINDOW_S = 2400
STARTUP_TIMEOUT_S = 600
N_CTX = 49_152

SPEC_PATH = Path("openspec/capabilities/arc-world-model-trust-energy/spec.md")
UPSTREAM_PATH = Path("results/experiment_7262_v639_arc_witness_receipt.json")
REGISTRY_PATH = Path("ops/arc_solve_registry.yaml")
RESULT_PATH = Path("results/experiment_7263_v639_arc_live.json")
CHECKPOINT_PATH = Path("results/checkpoints/experiment_7263_v639_arc_live.json")
RAW_DIR = Path("results/raw/experiment_7263_v639_arc_live")
RAW_ROWS_PATH = RAW_DIR / "episode_rows.json"
TERMINAL_CANDIDATE_PATH = RAW_DIR / "terminal_candidate.json"
SESSION_PATH = RAW_DIR / "live_session.json"
MODULE_PATH = Path("python/carnot/experiment_7263_v639_arc_live.py")
WRAPPER_PATH = Path("scripts/experiments/experiment_7263_v639_arc_live.py")
TEST_PATH = Path("tests/python/test_experiment_7263_v639_arc_live.py")

REQUIRED_INPUTS = (
    Path("AGENTS.md"),
    Path("CLAUDE.md"),
    Path("CODEX.md"),
    Path("research-program.md"),
    Path("ops/exclusion_manifest.yaml"),
    Path("ops/e2e-test-plan.md"),
    Path("python/carnot/agentic/arc_competition_agent.py"),
    Path("python/carnot/agentic/arc_induction_tool_loop.py"),
    Path("python/carnot/agentic/arc_eval_provenance.py"),
    Path("python/carnot/agentic/arc_transition_witness_exp7248.py"),
    Path("scripts/arc_loop_solve.py"),
    Path("scripts/kaggle/submission_kernel/main.py"),
    REGISTRY_PATH,
    SPEC_PATH,
    Path("scripts/experiment_template.py"),
    Path("python/carnot/inference/sota_models.py"),
    Path("python/carnot/experiment_7209_v635_span_canary.py"),
    Path("python/carnot/experiment_7234_v637_arc_scored_dryrun.py"),
    UPSTREAM_PATH,
    MODULE_PATH,
    WRAPPER_PATH,
    TEST_PATH,
)

FIELD_PRINCIPLES = {
    "schema": "Version the result; retain ordinary top-level experiment_id and milestone.",
    "experiment_id": "Bind the terminal result to Exp7263 without a nested value wrapper.",
    "milestone": "Bind the result to milestone 2026.09.639.",
    "status": "Use complete or blocked only for terminal work; checkpoints hold unfinished work.",
    "run_date": "Use 20260913 and actual UTC bounds so the dated evidence is auditable.",
    "started_at_utc": "Record the actual UTC start of this invocation.",
    "ended_at_utc": "Record the actual UTC end of terminal construction.",
    "field_principles": "Store explanations here; consumers read ordinary top-level values.",
    "preconditions_checked": "Retain observed input hashes, resource ownership, and failures before model work.",
    "MODEL_SPECS": "Declare models executable in this invocation; historical metadata stays in sidecars.",
    "model_invoked": "Derive model use from actual calls; parse failure does not erase an invocation.",
    "invocation_counts": "Separate attempted and completed loads and generations from usable answers.",
    "inference_substrate": "Use the recognized literal that describes the computation that ran.",
    "inference_substrate_class": "Declare actual compute; never pad time to satisfy a class floor.",
    "inference_mode": "Use live_gpu only when the task-owned runtime has CUDA evidence.",
    "execution_venue": "Use host for host orchestration and identify hardware in runtime receipts.",
    "execution_host": "Record the host separately from the execution venue.",
    "duration_s": "Measure monotonic time and retain disjoint phase spans.",
    "phase_spans": "Keep measured load, generation, scoring, and validation spans separate.",
    "random_seed": "Freeze independent-unit seeds before outcomes are visible.",
    "reproducibility_checksum": "Bind code, input manifests, configuration, and raw evidence.",
    "source_artifact_hashes": "Authenticate exact inputs and preserve quarantine and retirement state.",
    "rows": "Retain each game-arm unit, metric, error, abstention, and censoring state.",
    "sample_size_budget": "Record planned, attempted, completed, censored units and the fixed stop rule.",
    "acceptance_gate_results": "Keep expected, observed, pass, and principle for every criterion.",
    "gate_check_summary": "For blocked work, name each upstream check and exact observed and expected value.",
    "verifier_is_oracle": "Expose shared evaluator authority; exact conformance is not learned correctness.",
    "honest_verdict": "Use terminal complete_ or blocked_ prefixes and state the measured finding.",
    "verdict_class": "Use the fixed verdict vocabulary; oracle authority forbids a positive verdict.",
    "validation_receipts": "Record commands, exits, and log hashes without suppressing failures.",
    "arc_capture_complete_score": "One accounts for all four episodes; it is not an efficacy score.",
    "arc_method_value_score": "One requires prediction lift, policy use, and no matched-game regression.",
    "per_game_results": "Keep every game, seed, arm outcome, and cost for paired recomputation.",
    "solve_provenance": "Use live_agent_self_discovery; registered public solves receive no new credit.",
    "policy_consumption_rows": "Join engine and plan hashes to actual executed policy actions.",
    "selection_receipt": "Prove registry precheck preceded metadata-only roster freezing.",
    "runtime_receipt": "Bind GPU, native process, offload, decoding, and lease evidence to this run.",
    "raw_request_manifest": "Persist every dispatched request and reply with tokens and timestamps.",
    "submission_configuration_diff": "Keep local and submitted runtime differences explicit.",
    "claim_scope": "Two public games permit a case study only, not a population interval.",
    "population_confidence_interval": "Remain null because two games cannot support a population CI.",
    "new_solve_claimed": "Require registry precheck and reproduction before any new level credit.",
    "reproduced_levels": "Count only newly reproduced levels from this live attempt.",
}


def _canonical_bytes(value: Any) -> bytes:
    """Return stable JSON bytes for hashes shared by tests and reducers."""

    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()


def _sha256_bytes(value: bytes) -> str:
    """Return a standard prefixed SHA-256 digest."""

    return "sha256:" + hashlib.sha256(value).hexdigest()


def artifact_checksum(value: Mapping[str, Any]) -> str:
    """Hash an artifact while excluding the field that carries this digest."""

    payload = deepcopy(dict(value))
    payload["reproducibility_checksum"] = ""
    return _sha256_bytes(_canonical_bytes(payload))


def gate_check(check: str, upstream: str, field: str, expected: Any, observed: Any) -> JsonDict:
    """Build one ordinary fail-closed precondition row."""

    return {
        "check": check,
        "upstream": upstream,
        "field": field,
        "expected": expected,
        "observed": observed,
        "passed": observed == expected,
    }


def authenticate_upstream(path: Path, *, quarantined: bool) -> tuple[list[JsonDict], JsonDict]:
    """Authenticate Exp7262 and retain byte cleanliness as its own check."""

    try:
        raw = path.read_bytes()
    except OSError:
        raw = b""
    try:
        payload = json.loads(raw.rstrip(b"\x00").decode()) if raw else {}
    except (UnicodeDecodeError, json.JSONDecodeError):
        payload = {}
    clean = bool(raw and raw.endswith(b"\n") and b"\x00" not in raw)
    checks = [
        gate_check("upstream", str(path), "exists", True, path.is_file()),
        gate_check(
            "upstream",
            str(path),
            "experiment_id",
            "exp7262-arc-witness-receipt",
            payload.get("experiment_id"),
        ),
        gate_check("upstream", str(path), "status", "complete", payload.get("status")),
        gate_check(
            "upstream",
            str(path),
            "arc_witness_ready_score",
            1,
            payload.get("arc_witness_ready_score"),
        ),
        gate_check("upstream", str(path), "quarantine_state", False, quarantined),
        gate_check("upstream", str(path), "clean_terminal_bytes", True, clean),
    ]
    return checks, dict(payload)


def _count_scalar(value: Any, target: str) -> int:
    """Count exact metadata references without interpreting narrative outcome text."""

    if isinstance(value, Mapping):
        return sum(_count_scalar(item, target) for item in value.values())
    if isinstance(value, list | tuple):
        return sum(_count_scalar(item, target) for item in value)
    return int(value == target)


def select_frozen_roster(
    registry: Mapping[str, Any], *, adaptered_games: set[str] | frozenset[str]
) -> JsonDict:
    """Freeze the two least-referenced public games after a registry precheck."""

    games = [dict(row) for row in registry.get("games", []) if isinstance(row, Mapping)]
    ranking: list[JsonDict] = []
    for row in games:
        game = str(row.get("game") or "")
        if not game:
            continue
        ranking.append(
            {
                "game": game,
                "metadata_exposure_count": max(0, _count_scalar(registry, game) - 1),
                "registry_levels_before_attempt": int(row.get("levels_reproduced") or 0),
                "adapter_available_but_withheld": game in adaptered_games,
                "adapter_disabled": True,
                "registry_precheck_passed": True,
            }
        )
    ranking.sort(
        key=lambda row: (
            int(row["metadata_exposure_count"]),
            int(row["registry_levels_before_attempt"]),
            str(row["game"]),
        )
    )
    selected = ranking[:2]
    return {
        "selection_steps": ["registry_precheck", "roster_freeze"],
        "selection_basis": "fewest_exact_metadata_references_then_prior_levels_then_game_id",
        "selection_used_outcome_labels": False,
        "games": [row["game"] for row in selected],
        "game_rows": selected,
        "ranking": ranking,
    }


def counterbalanced_schedule(games: Sequence[str], seed: int) -> list[JsonDict]:
    """Assign both arms to two games and reverse their order on the second game."""

    arms = ("current_feedback", "typed_witness_feedback")
    rows: list[JsonDict] = []
    for game_index, game in enumerate(games[:2]):
        ordered = arms if game_index % 2 == 0 else tuple(reversed(arms))
        for arm in ordered:
            rows.append(
                {
                    "episode_id": f"{game}:{arm}",
                    "game": str(game),
                    "arm": arm,
                    "seed": int(seed),
                    "execution_order": len(rows),
                    "adapter_disabled": True,
                }
            )
    return rows


def episode_environment(
    base_env: Mapping[str, str],
    *,
    arm: str,
    episode_dir: Path,
    gpu_index: int,
    port: int,
) -> dict[str, str]:
    """Build one equal-budget environment whose only arm change is typed feedback."""

    env = dict(base_env)
    env.pop("CARNOT_ARC_TRANSITION_WITNESS", None)
    env.update(
        {
            "PYTHONUNBUFFERED": "1",
            "PYTHONPATH": f"{REPO_ROOT / 'python'}:{REPO_ROOT}",
            "CARNOT_FORCE_LIVE": "1",
            "CARNOT_ARC_LLM_BACKEND": "llamacpp",
            "CARNOT_ARC_INDUCE_TOOL_LOOP": "selfparse",
            "CARNOT_ARC_CEGIS_TOOL_LOOP": "1",
            "CARNOT_ARC_CEGIS_ACCEPT_SPLIT": "1",
            "CARNOT_ARC_MAX_REFINEMENT_ROUNDS": str(COMPLETION_LIMIT),
            "CARNOT_ARC_INDUCE_TOOL_TURNS": "1",
            "CARNOT_ARC_INDUCE_MAX_TOKENS": str(TOKENS_PER_CALL),
            "CARNOT_ARC_INDUCE_N_CTX": str(N_CTX),
            "CARNOT_ARC_INDUCE_TIMEOUT": "600",
            "CARNOT_ARC_LLAMA_SERVER_PARALLEL": "1",
            "CARNOT_ARC_GENERATOR_REQUIRE_CUDA": "1",
            "CARNOT_ARC_GENERATOR_CUDA_GPU": str(gpu_index),
            "CARNOT_ARC_RANDOM_SEED": str(RANDOM_SEED),
            "CARNOT_ARC_GENERATOR_SEED": str(RANDOM_SEED),
            "CARNOT_ARC_PROPOSER_PORT": str(port),
            "CARNOT_ARC_MTP": "0",
            "CARNOT_ARC_ACTION_PROVENANCE": "1",
            "CARNOT_ARC_ACTION_PROVENANCE_DIR": str(episode_dir / "action_provenance"),
            "CARNOT_ARC_E3_DIR": str(episode_dir / "e3"),
            "CARNOT_ARC_SERVER_LOG_DIR": str(episode_dir.parent / "server_logs"),
            "CUDA_VISIBLE_DEVICES": str(gpu_index),
        }
    )
    if arm == "typed_witness_feedback":
        env["CARNOT_ARC_TRANSITION_WITNESS"] = "1"
    return env


def identity_baseline_from_transition_source(payload: Mapping[str, Any]) -> JsonDict:
    """Score identity on the acceptance rows that refinement could not inspect."""

    from carnot.agentic.arc_executable_world_model import Transition
    from carnot.agentic.arc_world_model_trust_energy import split_refinement_acceptance

    transitions = [
        Transition(
            grid=np.asarray(row["grid"]),
            action=int(row.get("action", -1)),
            data=None,
            next_grid=np.asarray(row["next_grid"]),
            level_before=int(row.get("level_before", 0)),
            level_after=int(row.get("level_after", 0)),
        )
        for row in payload.get("rows", [])
        if isinstance(row, Mapping) and "grid" in row and "next_grid" in row
    ]
    split = split_refinement_acceptance(transitions)
    acceptance_ids = {id(row) for row in split.acceptance}
    acceptance_indices = [
        index for index, transition in enumerate(transitions) if id(transition) in acceptance_ids
    ]
    gradeable = [row for row in split.acceptance if int(row.level_after) <= int(row.level_before)]
    correct = sum(np.array_equal(row.grid, row.next_grid) for row in gradeable)
    total = len(gradeable)
    return {
        "reserved_row_indices": acceptance_indices,
        "identity_correct": int(correct),
        "identity_total": total,
        "identity_baseline_accuracy": (float(correct) / total if total else None),
        "reserved_before_refinement": True,
        "acceptance_decidable": bool(split.decidable),
        "acceptance_reason": split.reason,
    }


def _normalise_digest(value: Any) -> str | None:
    """Read the standard engine fingerprint shapes without inventing a digest."""

    if isinstance(value, Mapping):
        value = value.get("sha256_full") or value.get("sha256")
    if not value:
        return None
    text = str(value)
    return text if text.startswith("sha256:") else "sha256:" + text


def _attempt_engine_digest(attempt: Mapping[str, Any]) -> str | None:
    """Select the retained round's engine digest for action provenance joins."""

    rounds = [row for row in attempt.get("refinement_rounds", []) if isinstance(row, Mapping)]
    preferred = [row for row in rounds if row.get("retained_as_best_engine")]
    for row in reversed(preferred or rounds):
        digest = _normalise_digest(row.get("engine_source_sha256"))
        if digest:
            return digest
    return _normalise_digest(attempt.get("engine_source_sha256"))


def build_policy_consumption_rows(
    episode_id: str,
    attempts: Sequence[Mapping[str, Any]],
    action_rows: Sequence[Mapping[str, Any]],
) -> list[JsonDict]:
    """Join a planned engine to the real plan-step actions the policy executed."""

    consumed = [
        dict(row)
        for row in action_rows
        if row.get("top_branch") in {"execute.plan_step", "induce.plan_from_current"}
    ]
    groups: dict[tuple[int, int], list[JsonDict]] = {}
    for row in consumed:
        attempt_index = row.get("plan_installed_by_attempt")
        if attempt_index is None:
            attempt_index = row.get("induction_attempt_index")
        if not isinstance(attempt_index, int) or not 0 <= attempt_index < len(attempts):
            continue
        if not attempts[attempt_index].get("planned"):
            continue
        groups.setdefault((attempt_index, int(row.get("plan_epoch") or 0)), []).append(row)
    output: list[JsonDict] = []
    for (attempt_index, plan_epoch), group in sorted(groups.items()):
        engine_digest = _attempt_engine_digest(attempts[attempt_index])
        if engine_digest is None:
            continue
        action_projection = [
            {"action": row.get("action"), "data": row.get("data")} for row in group
        ]
        plan_digest = _sha256_bytes(_canonical_bytes(action_projection))
        for row in group:
            output.append(
                {
                    "episode_id": episode_id,
                    "attempt_index": attempt_index,
                    "plan_epoch": plan_epoch,
                    "engine_sha256": engine_digest,
                    "plan_sha256": plan_digest,
                    "action_index": int(row.get("i") or 0),
                    "action": row.get("action"),
                    "data": deepcopy(row.get("data")),
                    "policy_action_executed": True,
                    "top_branch": row.get("top_branch"),
                }
            )
    return output


def validate_episode_row(row: Mapping[str, Any]) -> list[str]:
    """Refuse a row whose live budget or adapter isolation drifted."""

    errors: list[str] = []
    if int(row.get("action_count") or 0) > ACTION_LIMIT:
        errors.append("action_limit_exceeded")
    if int(row.get("generation_calls_attempted") or 0) > COMPLETION_LIMIT:
        errors.append("generation_call_limit_exceeded")
    if int(row.get("generation_calls_completed") or 0) > int(
        row.get("generation_calls_attempted") or 0
    ):
        errors.append("generation_completion_count_invalid")
    if int(row.get("generated_tokens") or 0) > GENERATED_TOKEN_LIMIT:
        errors.append("generated_token_limit_exceeded")
    if row.get("adapter_disabled") is not True:
        errors.append("adapter_not_disabled")
    if row.get("arm") not in {"current_feedback", "typed_witness_feedback"}:
        errors.append("arm_invalid")
    if row.get("disposition") not in {"complete", "censored_timeout"}:
        errors.append("disposition_not_terminal")
    return errors


def _mean(values: Sequence[float]) -> float | None:
    """Return a mean only when at least one measured value exists."""

    return sum(values) / len(values) if values else None


def reduce_episode_rows(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Independently reduce the four frozen episodes into capture and value gates."""

    episodes = [deepcopy(dict(row)) for row in rows]
    unique_units = {(str(row.get("game")), str(row.get("arm"))) for row in episodes}
    games = sorted({str(row.get("game")) for row in episodes})
    expected_units = {
        (game, arm) for game in games for arm in ("current_feedback", "typed_witness_feedback")
    }
    row_errors = [error for row in episodes for error in validate_episode_row(row)]
    accounted = all(
        int(row.get("generation_calls_completed") or 0)
        <= int(row.get("generation_calls_attempted") or 0)
        for row in episodes
    )
    capture = int(
        len(episodes) == 4
        and len(games) == 2
        and unique_units == expected_units
        and not row_errors
        and accounted
    )
    current = [
        float(row["heldout_accuracy"])
        for row in episodes
        if row.get("arm") == "current_feedback" and row.get("heldout_accuracy") is not None
    ]
    treatment = [
        float(row["heldout_accuracy"])
        for row in episodes
        if row.get("arm") == "typed_witness_feedback" and row.get("heldout_accuracy") is not None
    ]
    identity = [
        float(row["identity_baseline_accuracy"])
        for row in episodes
        if row.get("arm") == "typed_witness_feedback"
        and row.get("identity_baseline_accuracy") is not None
    ]
    current_mean = _mean(current)
    treatment_mean = _mean(treatment)
    identity_mean = _mean(identity)
    lift_current = (
        treatment_mean - current_mean
        if treatment_mean is not None and current_mean is not None
        else None
    )
    lift_identity = (
        treatment_mean - identity_mean
        if treatment_mean is not None and identity_mean is not None
        else None
    )
    treatment_consumption = sum(
        len(row.get("policy_consumption_rows", []))
        for row in episodes
        if row.get("arm") == "typed_witness_feedback"
    )
    regressions: list[str] = []
    for game in games:
        levels = {
            str(row.get("arm")): int(row.get("levels") or 0)
            for row in episodes
            if row.get("game") == game
        }
        if levels.get("typed_witness_feedback", 0) < levels.get("current_feedback", 0):
            regressions.append(game)
    useful_candidate = any(
        row.get("arm") == "typed_witness_feedback"
        and row.get("candidate_valid") is True
        and row.get("trust_accepted") is True
        and int(row.get("non_identity_predictions") or 0) > 0
        for row in episodes
    )
    method_value = int(
        capture == 1
        and lift_current is not None
        and lift_current > 0.0
        and lift_identity is not None
        and lift_identity > 0.0
        and treatment_consumption > 0
        and not regressions
        and useful_candidate
    )
    attempted_calls = sum(int(row.get("generation_calls_attempted") or 0) for row in episodes)
    completed_calls = sum(int(row.get("generation_calls_completed") or 0) for row in episodes)
    usable_answers = sum(int(row.get("usable_answers") or 0) for row in episodes)
    any_load_attempt = any(row.get("model_invoked") or row.get("model_loaded") for row in episodes)
    any_loaded = any(row.get("model_loaded") is True for row in episodes)
    return {
        "arc_capture_complete_score": capture,
        "arc_method_value_score": method_value,
        "typed_witness_mean_heldout_accuracy": treatment_mean,
        "current_feedback_mean_heldout_accuracy": current_mean,
        "identity_baseline_mean_accuracy": identity_mean,
        "typed_witness_prediction_lift_vs_current": lift_current,
        "typed_witness_prediction_lift_vs_identity": lift_identity,
        "treatment_policy_consumed_plans": treatment_consumption,
        "matched_game_level_regression": bool(regressions),
        "regressed_games": regressions,
        "useful_treatment_candidate": useful_candidate,
        "row_errors": row_errors,
        "claim_scope": "two_game_case_study_only",
        "population_confidence_interval": None,
        "invocation_counts": {
            "model_loads_attempted": int(any_load_attempt),
            "model_loads_completed": int(any_loaded),
            "generation_calls_attempted": attempted_calls,
            "generation_calls_completed": completed_calls,
            "usable_answers": usable_answers,
        },
    }


def _acceptance_results(reduction: Mapping[str, Any]) -> list[JsonDict]:
    """Render each frozen criterion with expected and observed values."""

    return [
        {
            "criterion": "all_four_episodes_accounted",
            "expected": 1,
            "observed": reduction["arc_capture_complete_score"],
            "passed": reduction["arc_capture_complete_score"] == 1,
            "principle": "Capture completeness is separate from method value.",
        },
        {
            "criterion": "typed_witness_prediction_lift_above_current",
            "expected": ">0",
            "observed": reduction["typed_witness_prediction_lift_vs_current"],
            "passed": bool((reduction["typed_witness_prediction_lift_vs_current"] or 0) > 0),
            "principle": "Treatment must improve the paired current-feedback case study.",
        },
        {
            "criterion": "typed_witness_prediction_above_identity",
            "expected": ">0",
            "observed": reduction["typed_witness_prediction_lift_vs_identity"],
            "passed": bool((reduction["typed_witness_prediction_lift_vs_identity"] or 0) > 0),
            "principle": "A learned candidate must beat predicting no state change.",
        },
        {
            "criterion": "treatment_plan_consumed_by_policy",
            "expected": ">0",
            "observed": reduction["treatment_policy_consumed_plans"],
            "passed": int(reduction["treatment_policy_consumed_plans"]) > 0,
            "principle": "A valid engine without an executed plan has no policy value.",
        },
        {
            "criterion": "no_matched_game_level_regression",
            "expected": False,
            "observed": reduction["matched_game_level_regression"],
            "passed": reduction["matched_game_level_regression"] is False,
            "principle": "Prediction gains cannot hide a policy-level regression.",
        },
    ]


def _base_artifact_fields(*, started_at_utc: str, ended_at_utc: str, duration_s: float) -> JsonDict:
    """Create shared terminal identity and timing fields."""

    return {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "run_date": RUN_DATE,
        "started_at_utc": started_at_utc,
        "ended_at_utc": ended_at_utc,
        "duration_s": round(float(duration_s), 6),
        "execution_venue": "host",
        "execution_host": socket.gethostname(),
        "random_seed": RANDOM_SEED,
        "reproducibility_checksum": "",
    }


def build_terminal_artifact(
    *,
    started_at_utc: str,
    ended_at_utc: str,
    duration_s: float,
    preconditions: Sequence[Mapping[str, Any]],
    source_hashes: Mapping[str, Any],
    selection_receipt: Mapping[str, Any],
    episode_rows: Sequence[Mapping[str, Any]],
    model_spec: Mapping[str, Any],
    runtime_receipt: Mapping[str, Any],
    validation_receipts: Sequence[Mapping[str, Any]],
    phase_spans: Sequence[Mapping[str, Any]],
) -> JsonDict:
    """Build a complete measured artifact without promoting a failed scientific gate."""

    rows = [deepcopy(dict(row)) for row in episode_rows]
    reduction = reduce_episode_rows(rows)
    validation_passed = all(row.get("passed") is True for row in validation_receipts)
    method_value = int(reduction["arc_method_value_score"] == 1 and validation_passed)
    invoked = reduction["invocation_counts"]["generation_calls_attempted"] > 0
    loaded = reduction["invocation_counts"]["model_loads_completed"] > 0
    treatment_consumed = int(reduction["treatment_policy_consumed_plans"])
    if method_value:
        verdict = "circular_positive"
        honest = "complete_circular_positive_typed_witness_case_study_value"
    elif treatment_consumed == 0:
        verdict = "null"
        honest = "complete_null_no_policy_consumed_plan"
    elif not validation_passed:
        verdict = "null"
        honest = "complete_null_validation_failed"
    else:
        verdict = "null"
        honest = "complete_null_typed_witness_case_study_gates_not_met"
    if invoked:
        substrate = "live_llm_inference"
        substrate_class = "model_full_generation"
    elif loaded:
        substrate = "model_load_no_generation"
        substrate_class = "model_load_no_generation"
    else:
        substrate = "blocked_no_run"
        substrate_class = "blocked_no_run"
    artifact = _base_artifact_fields(
        started_at_utc=started_at_utc,
        ended_at_utc=ended_at_utc,
        duration_s=duration_s,
    )
    artifact.update(
        {
            "status": "complete",
            "field_principles": {},
            "preconditions_checked": [deepcopy(dict(row)) for row in preconditions],
            "MODEL_SPECS": [deepcopy(dict(model_spec))] if (invoked or loaded) else [],
            "model_invoked": invoked,
            "invocation_counts": deepcopy(reduction["invocation_counts"]),
            "inference_substrate": substrate,
            "inference_substrate_class": substrate_class,
            "inference_mode": "live_gpu" if loaded else "not_invoked",
            "phase_spans": [deepcopy(dict(row)) for row in phase_spans],
            "source_artifact_hashes": deepcopy(dict(source_hashes)),
            "rows": rows,
            "sample_size_budget": {
                "planned_units": 4,
                "attempted_units": len(rows),
                "completed_units": sum(row.get("disposition") == "complete" for row in rows),
                "censored_units": sum(bool(row.get("censored")) for row in rows),
                "games": 2,
                "seeds_per_game": 1,
                "actions_per_episode": ACTION_LIMIT,
                "completion_limit_per_episode": COMPLETION_LIMIT,
                "generated_token_limit_per_episode": GENERATED_TOKEN_LIMIT,
                "live_window_s": LIVE_WINDOW_S,
                "stopping_rule": "four frozen counterbalanced episodes; timeouts stay censored",
            },
            "acceptance_gate_results": _acceptance_results(reduction),
            "gate_check_summary": [],
            "verifier_is_oracle": True,
            "honest_verdict": honest,
            "verdict_class": verdict,
            "validation_receipts": [deepcopy(dict(row)) for row in validation_receipts],
            "arc_capture_complete_score": reduction["arc_capture_complete_score"],
            "arc_method_value_score": method_value,
            "per_game_results": rows,
            "solve_provenance": "live_agent_self_discovery",
            "policy_consumption_rows": [
                deepcopy(item) for row in rows for item in row.get("policy_consumption_rows", [])
            ],
            "selection_receipt": deepcopy(dict(selection_receipt)),
            "runtime_receipt": deepcopy(dict(runtime_receipt)),
            "raw_request_manifest": [
                deepcopy(item) for row in rows for item in row.get("raw_request_manifest", [])
            ],
            "submission_configuration_diff": {
                "gateway": "local_public_arcade_not_hidden_competition_gateway",
                "gpu": "host_GPU_not_Kaggle_Blackwell_96GB",
                "weights": "Q4_K_M_GGUF_not_submitted_NVFP4_safetensors",
                "runtime": "native_llama.cpp_single_server_not_submitted_vLLM",
                "parallelism": "four_sequential_episodes_not_submitted_concurrent_games",
                "completion_cap": f"{GENERATED_TOKEN_LIMIT}_per_episode",
                "competition_equivalent_claim": False,
            },
            "claim_scope": reduction["claim_scope"],
            "population_confidence_interval": reduction["population_confidence_interval"],
            "new_solve_claimed": False,
            "reproduced_levels": 0,
        }
    )
    artifact["field_principles"] = {key: FIELD_PRINCIPLES[key] for key in artifact}
    artifact["reproducibility_checksum"] = artifact_checksum(artifact)
    return artifact


def build_blocked_artifact(
    *,
    started_at_utc: str,
    ended_at_utc: str,
    duration_s: float,
    preconditions: Sequence[Mapping[str, Any]],
    source_hashes: Mapping[str, Any],
) -> JsonDict:
    """Build a terminal external block with no success-shaped measurement fields."""

    failures = [deepcopy(dict(row)) for row in preconditions if row.get("passed") is not True]
    artifact = _base_artifact_fields(
        started_at_utc=started_at_utc,
        ended_at_utc=ended_at_utc,
        duration_s=duration_s,
    )
    artifact.update(
        {
            "status": "blocked",
            "field_principles": {},
            "preconditions_checked": [deepcopy(dict(row)) for row in preconditions],
            "MODEL_SPECS": [],
            "model_invoked": False,
            "invocation_counts": {
                "model_loads_attempted": 0,
                "model_loads_completed": 0,
                "generation_calls_attempted": 0,
                "generation_calls_completed": 0,
                "usable_answers": 0,
            },
            "inference_substrate": "blocked_no_run",
            "inference_substrate_class": "blocked_no_run",
            "inference_mode": "not_invoked",
            "phase_spans": [],
            "source_artifact_hashes": deepcopy(dict(source_hashes)),
            "rows": [],
            "sample_size_budget": {
                "planned_units": 4,
                "attempted_units": 0,
                "completed_units": 0,
                "censored_units": 0,
                "stopping_rule": "external precondition failure stops before model load",
            },
            "acceptance_gate_results": [
                {
                    "criterion": row.get("field"),
                    "expected": row.get("expected"),
                    "observed": row.get("observed"),
                    "passed": False,
                    "principle": "An external prerequisite must pass before expensive work.",
                }
                for row in failures
            ],
            "gate_check_summary": failures,
            "verifier_is_oracle": True,
            "honest_verdict": "blocked_required_external_precondition",
            "verdict_class": "blocked",
            "validation_receipts": [],
            "arc_capture_complete_score": 0,
            "arc_method_value_score": 0,
            "per_game_results": [],
            "solve_provenance": "live_agent_self_discovery",
            "policy_consumption_rows": [],
            "selection_receipt": {},
            "runtime_receipt": {},
            "raw_request_manifest": [],
            "submission_configuration_diff": {},
            "claim_scope": "blocked_no_case_study",
            "population_confidence_interval": None,
            "new_solve_claimed": False,
            "reproduced_levels": 0,
        }
    )
    artifact["field_principles"] = {key: FIELD_PRINCIPLES[key] for key in artifact}
    artifact["reproducibility_checksum"] = artifact_checksum(artifact)
    return artifact


def validate_artifact(value: Mapping[str, Any] | Path) -> list[str]:
    """Cold-check terminal structure, recomputed rows, budgets, and checksum."""

    if isinstance(value, Path):
        try:
            artifact = json.loads(value.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            return [f"artifact_unreadable:{type(exc).__name__}"]
    else:
        artifact = deepcopy(dict(value))
    errors: list[str] = []
    if artifact.get("schema") != SCHEMA or artifact.get("experiment_id") != EXPERIMENT_ID:
        errors.append("schema_or_experiment_identity_mismatch")
    if artifact.get("milestone") != MILESTONE or artifact.get("run_date") != RUN_DATE:
        errors.append("milestone_or_run_date_mismatch")
    if artifact.get("status") not in {"complete", "blocked"}:
        errors.append("status_not_terminal")
    if set(artifact.get("field_principles", {})) != set(artifact):
        errors.append("field_principles_must_cover_every_top_level_field")
    if artifact.get("verdict_class") not in {
        "positive",
        "circular_positive",
        "null",
        "blocked",
        "disqualified",
        "partial",
    }:
        errors.append("verdict_class_invalid")
    if artifact.get("verifier_is_oracle") is True and artifact.get("verdict_class") == "positive":
        errors.append("oracle_forbids_positive")
    status = artifact.get("status")
    if status == "blocked":
        if not str(artifact.get("honest_verdict") or "").startswith("blocked_"):
            errors.append("blocked_verdict_prefix_invalid")
        if artifact.get("rows") or artifact.get("model_invoked"):
            errors.append("blocked_artifact_contains_measurement")
        if artifact.get("inference_substrate") != "blocked_no_run":
            errors.append("blocked_substrate_invalid")
        if not artifact.get("gate_check_summary"):
            errors.append("blocked_gate_summary_missing")
    elif status == "complete":
        if not str(artifact.get("honest_verdict") or "").startswith("complete_"):
            errors.append("complete_verdict_prefix_invalid")
        reduction = reduce_episode_rows(artifact.get("rows", []))
        validations_pass = all(
            row.get("passed") is True for row in artifact.get("validation_receipts", [])
        )
        expected_value = int(reduction["arc_method_value_score"] == 1 and validations_pass)
        if artifact.get("arc_capture_complete_score") != reduction["arc_capture_complete_score"]:
            errors.append("arc_capture_complete_score_inconsistent")
        if artifact.get("arc_method_value_score") != expected_value:
            errors.append("arc_method_value_score_inconsistent")
        if artifact.get("invocation_counts") != reduction["invocation_counts"]:
            errors.append("invocation_counts_inconsistent")
        invoked = reduction["invocation_counts"]["generation_calls_attempted"] > 0
        loaded = reduction["invocation_counts"]["model_loads_completed"] > 0
        expected_specs = [deepcopy(dict(MODEL_SPECS[0]))] if (invoked or loaded) else []
        actual_specs = artifact.get("MODEL_SPECS", [])
        if expected_specs and (
            len(actual_specs) != 1
            or actual_specs[0].get("hf_id") != MODEL_ID
            or actual_specs[0].get("quantization", QUANTIZATION) != QUANTIZATION
        ):
            errors.append("model_specs_inconsistent")
        if not expected_specs and actual_specs:
            errors.append("model_specs_inconsistent")
        if artifact.get("model_invoked") != invoked:
            errors.append("model_invoked_inconsistent")
        if invoked and artifact.get("inference_substrate") != "live_llm_inference":
            errors.append("live_substrate_inconsistent")
        if invoked and artifact.get("duration_s", 0) < 60:
            errors.append("model_full_generation_duration_floor_failed")
        if (
            reduction["treatment_policy_consumed_plans"] == 0
            and artifact.get("honest_verdict") != "complete_null_no_policy_consumed_plan"
        ):
            errors.append("no_consumption_verdict_inconsistent")
        if artifact.get("population_confidence_interval") is not None:
            errors.append("two_game_population_ci_forbidden")
        for row in artifact.get("rows", []):
            errors.extend(validate_episode_row(row))
    if artifact.get("reproducibility_checksum") != artifact_checksum(artifact):
        errors.append("reproducibility_checksum_mismatch")
    return errors


def independent_reduce(path: Path) -> JsonDict:
    """Read raw episode rows and recompute the scientific gates from scratch."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    return reduce_episode_rows(payload.get("rows", []))


def build_validation_commands(*, terminal_candidate: Path, raw_rows: Path) -> list[JsonDict]:
    """Return the exact scoped validation plan; the live loop itself is the E2E."""

    python = str(REPO_ROOT / ".venv/bin/python")
    pytest = str(REPO_ROOT / ".venv/bin/pytest")
    coverage = str(REPO_ROOT / ".venv/bin/coverage")
    ruff = str(REPO_ROOT / ".venv/bin/ruff")
    mypy = str(REPO_ROOT / ".venv/bin/mypy")
    tests = {
        "focused_exp7263": TEST_PATH,
        "affected_exp7262": Path("tests/python/test_experiment_7262_v639_arc_witness_receipt.py"),
        "affected_transition_witness": Path(
            "tests/python/test_experiment_7248_v638_arc_witness.py"
        ),
        "affected_induction_state": Path("tests/python/test_arc_induction_state_persistence.py"),
    }
    rows = [
        {
            "name": name,
            "command": [
                pytest,
                str(path),
                "-q",
                "--no-cov",
                "-n",
                "0",
                f"--basetemp=/tmp/exp7263-{name}",
            ],
        }
        for name, path in tests.items()
    ]
    rows.extend(
        [
            {
                "name": "scoped_coverage_run",
                "command": [
                    coverage,
                    "run",
                    "--data-file=/tmp/exp7263-v639.coverage",
                    "--include=*/experiment_7263_v639_arc_live.py",
                    "-m",
                    "pytest",
                    "-o",
                    "addopts=",
                    str(TEST_PATH),
                    "-q",
                    "-n",
                    "0",
                    "--basetemp=/tmp/exp7263-coverage",
                ],
            },
            {
                "name": "scoped_coverage_report",
                "command": [
                    coverage,
                    "report",
                    "--data-file=/tmp/exp7263-v639.coverage",
                    "--include=*/experiment_7263_v639_arc_live.py",
                    "--show-missing",
                    "--fail-under=100",
                ],
            },
            {
                "name": "ruff_check",
                "command": [ruff, "check", str(MODULE_PATH), str(WRAPPER_PATH), str(TEST_PATH)],
            },
            {
                "name": "ruff_format",
                "command": [
                    ruff,
                    "format",
                    "--check",
                    str(MODULE_PATH),
                    str(WRAPPER_PATH),
                    str(TEST_PATH),
                ],
            },
            {
                "name": "changed_module_mypy",
                "command": [mypy, str(MODULE_PATH), str(WRAPPER_PATH)],
            },
            {
                "name": "scoped_spec_coverage",
                "command": [
                    python,
                    "-u",
                    "scripts/check_spec_coverage.py",
                    str(TEST_PATH),
                    *[str(path) for path in list(tests.values())[1:]],
                ],
            },
            {
                "name": "independent_raw_row_reducer",
                "command": [
                    python,
                    "-u",
                    str(WRAPPER_PATH),
                    "--reduce-raw",
                    str(raw_rows),
                ],
            },
            {
                "name": "terminal_candidate_adversarial_verify",
                "command": [python, "-u", "scripts/adversarial_verify.py", str(terminal_candidate)],
            },
            {
                "name": "terminal_candidate_row_consistency",
                "command": [
                    python,
                    "-u",
                    "scripts/verdict_row_consistency_lint.py",
                    str(terminal_candidate),
                ],
            },
        ]
    )
    return rows


def _progress(phase: str, event: str, **fields: Any) -> None:  # pragma: no cover - live output.
    """Print and flush one truthful phase or long-call boundary."""

    print(
        json.dumps(
            {"experiment": TASK_ID, "phase": phase, "event": event, **fields}, sort_keys=True
        ),
        flush=True,
    )


def _iso_now() -> str:  # pragma: no cover - live timestamp.
    """Return an auditable UTC timestamp."""

    return datetime.now(UTC).isoformat(timespec="seconds")


def atomic_write(path: Path, value: Any) -> None:
    """Publish one JSON file through a same-directory atomic rename."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _sha256_file(path: Path) -> str:  # pragma: no cover - source authentication.
    """Hash one file and emit heartbeats if large model bytes take time."""

    digest = hashlib.sha256()
    started = time.monotonic()
    next_beat = started + 45
    total = 0
    with path.open("rb") as handle:
        while chunk := handle.read(16 * 1024 * 1024):
            digest.update(chunk)
            total += len(chunk)
            now = time.monotonic()
            if now >= next_beat:
                _progress(
                    "preconditions",
                    "hash_heartbeat",
                    path=str(path),
                    bytes_read=total,
                    elapsed_s=round(now - started, 1),
                )
                next_beat = now + 45
    return "sha256:" + digest.hexdigest()


def _load_registry(path: Path) -> JsonDict:  # pragma: no cover - live input.
    """Read only registry metadata before selecting games."""

    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError):
        return {}
    return dict(value) if isinstance(value, Mapping) else {}


def _quarantined_upstream(value: Mapping[str, Any]) -> bool:  # pragma: no cover - live input.
    """Reuse the shipped quarantine classifier for the upstream receipt."""

    return bool(scored.is_quarantined(value))


def collect_preconditions(
    root: Path,
) -> tuple[list[JsonDict], JsonDict, JsonDict]:  # pragma: no cover
    """Authenticate every listed input, writable path, model, runtime, and GPU."""

    checks: list[JsonDict] = []
    hashes: JsonDict = {}
    for relative in REQUIRED_INPUTS:
        path = root / relative
        exists = path.is_file()
        checks.append(gate_check("required_input", relative.as_posix(), "exists", True, exists))
        if exists:
            hashes[relative.as_posix()] = _sha256_file(path)
    spec_text = (
        (root / SPEC_PATH).read_text(encoding="utf-8") if (root / SPEC_PATH).is_file() else ""
    )
    checks.append(
        gate_check(
            "driving_capability",
            SPEC_PATH.as_posix(),
            "REQ-ARC-WMTE-7263",
            True,
            "REQ-ARC-WMTE-7263" in spec_text,
        )
    )
    try:
        raw_upstream = json.loads((root / UPSTREAM_PATH).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        raw_upstream = {}
    upstream_checks, upstream = authenticate_upstream(
        root / UPSTREAM_PATH, quarantined=_quarantined_upstream(raw_upstream)
    )
    checks.extend(upstream_checks)
    for relative in (RESULT_PATH.parent, CHECKPOINT_PATH.parent, RAW_DIR):
        path = root / relative
        path.mkdir(parents=True, exist_ok=True)
        checks.append(
            gate_check(
                "writable_output_directory",
                relative.as_posix(),
                "owner_and_writable",
                True,
                path.is_dir() and os.access(path, os.W_OK) and path.stat().st_uid == os.getuid(),
            )
        )
    from carnot.inference.sota_models import cached_current_model

    current = cached_current_model(preferred_quant=QUANTIZATION)
    model_path = Path(str(current.get("model_path"))) if current else None
    observed_quantization = (current.get("quantization") if current else None) or (
        QUANTIZATION if model_path and QUANTIZATION in model_path.name else None
    )
    model_ok = bool(
        current
        and current.get("hf_id") == MODEL_ID
        and observed_quantization == QUANTIZATION
        and model_path
        and model_path.is_file()
    )
    checks.append(gate_check("cached_current_model", MODEL_ID, QUANTIZATION, True, model_ok))
    server_candidates = (
        os.environ.get("CARNOT_LLAMA_SERVER"),
        str(Path.home() / ".cache/llama.cpp-master/build/bin/llama-server"),
    )
    server = next(
        (Path(value) for value in server_candidates if value and Path(value).is_file()), None
    )
    checks.append(gate_check("native_runtime", "llama.cpp", "server_binary", True, bool(server)))
    gpus = scored._gpu_inventory()
    idle = [
        row
        for row in gpus
        if not row.get("compute_apps") and int(row.get("free_memory_mb") or 0) >= 20_000
    ]
    checks.append(gate_check("gpu", "nvidia-smi", "idle_gpu_with_20GB", True, bool(idle)))
    model_hash = None
    if model_ok and model_path is not None:
        _progress("preconditions", "BEFORE model hash", path=str(model_path))
        model_hash = _sha256_file(model_path)
        hashes[str(model_path)] = model_hash
        _progress("preconditions", "AFTER model hash", sha256=model_hash)
    return (
        checks,
        hashes,
        {
            "upstream": upstream,
            "model_path": str(model_path) if model_ok and model_path else None,
            "model_hash": model_hash,
            "server": str(server) if server else None,
            "gpu": idle[0] if idle else None,
        },
    )


class RequestCapture:  # pragma: no cover - live HTTP seam.
    """Persist every model request and response while enforcing per-episode caps."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.rows: list[JsonDict] = []
        self.episode_id = "unassigned"
        self.original: Any = None

    def install(self) -> None:
        """Install the narrow urllib seam used by LocalGGUFProposer."""

        import urllib.request

        self.original = urllib.request.urlopen
        urllib.request.urlopen = self._open

    def restore(self) -> None:
        """Restore urllib after the live session."""

        import urllib.request

        if self.original is not None:
            urllib.request.urlopen = self.original

    def begin_episode(self, episode_id: str) -> None:
        """Assign later generation rows to one frozen unit."""

        self.episode_id = episode_id

    def episode_rows(self, episode_id: str) -> list[JsonDict]:
        """Return generation calls dispatched for one episode."""

        return [row for row in self.rows if row.get("episode_id") == episode_id]

    def _open(self, request: Any, *args: Any, **kwargs: Any) -> Any:
        import io
        import urllib.request

        if not isinstance(request, urllib.request.Request):
            return self.original(request, *args, **kwargs)
        url = str(request.full_url)
        if not url.endswith(("/completion", "/v1/chat/completions", "/v1/completions")):
            return self.original(request, *args, **kwargs)
        prior = self.episode_rows(self.episode_id)
        if len(prior) >= COMPLETION_LIMIT:
            raise RuntimeError("episode_generation_call_limit_reached")
        body = bytes(request.data or b"")
        payload = json.loads(body) if body else {}
        requested = int(payload.get("max_tokens") or payload.get("n_predict") or 0)
        if requested > TOKENS_PER_CALL:
            raise RuntimeError("episode_generation_token_request_exceeded")
        index = len(prior)
        directory = self.root / self.episode_id.replace(":", "__") / "requests"
        directory.mkdir(parents=True, exist_ok=True)
        request_path = directory / f"{index:02d}_request.json"
        response_path = directory / f"{index:02d}_response.json"
        request_path.write_bytes(body)
        started = time.monotonic()
        started_utc = _iso_now()
        row: JsonDict = {
            "episode_id": self.episode_id,
            "call_index": index,
            "url": url,
            "request_path": str(request_path),
            "request_sha256": _sha256_bytes(body),
            "request_bytes": len(body),
            "requested_max_tokens": requested,
            "decoding_parameters": {
                key: payload.get(key)
                for key in ("temperature", "top_p", "top_k", "seed", "max_tokens", "n_predict")
                if key in payload
            },
            "started_at_utc": started_utc,
            "ended_at_utc": None,
            "duration_s": None,
            "transport_completed": False,
            "usable_answer": False,
            "prompt_tokens": None,
            "completion_tokens": None,
            "finish_reason": None,
            "response_path": str(response_path),
            "response_sha256": None,
            "error": None,
        }
        self.rows.append(row)
        _progress("generation", "BEFORE model generation", episode_id=self.episode_id, call=index)
        try:
            response = self.original(request, *args, **kwargs)
            response_bytes = response.read()
            response.close()
            response_path.write_bytes(response_bytes)
            output = json.loads(response_bytes) if response_bytes else {}
            choice = (output.get("choices") or [{}])[0] if isinstance(output, Mapping) else {}
            message = choice.get("message") if isinstance(choice, Mapping) else {}
            content = message.get("content") if isinstance(message, Mapping) else choice.get("text")
            usage = output.get("usage", {}) if isinstance(output, Mapping) else {}
            timings = output.get("timings", {}) if isinstance(output, Mapping) else {}
            row.update(
                {
                    "transport_completed": True,
                    "usable_answer": bool(content),
                    "prompt_tokens": usage.get("prompt_tokens", timings.get("prompt_n")),
                    "completion_tokens": usage.get("completion_tokens", timings.get("predicted_n")),
                    "finish_reason": choice.get("finish_reason", output.get("stop_type")),
                    "response_sha256": _sha256_bytes(response_bytes),
                }
            )
            return io.BytesIO(response_bytes)
        except Exception as exc:
            row["error"] = f"{type(exc).__name__}: {exc}"[:300]
            raise
        finally:
            row["ended_at_utc"] = _iso_now()
            row["duration_s"] = round(time.monotonic() - started, 6)
            _progress(
                "generation",
                "AFTER model generation",
                episode_id=self.episode_id,
                call=index,
                completed=row["transport_completed"],
                elapsed_s=row["duration_s"],
            )


def _identity_from_attempts(attempts: Sequence[Mapping[str, Any]]) -> JsonDict:  # pragma: no cover
    """Load the archived pre-refinement transition source for identity scoring."""

    for attempt in reversed(attempts):
        for row in attempt.get("refinement_rounds", []):
            path_text = row.get("transition_source_path") if isinstance(row, Mapping) else None
            if not path_text:
                continue
            path = Path(str(path_text))
            if not path.is_absolute():
                path = REPO_ROOT / path
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            result = identity_baseline_from_transition_source(payload)
            result["transition_source_path"] = str(path)
            result["transition_source_sha256"] = _sha256_file(path)
            return result
    return {
        "reserved_row_indices": [],
        "identity_correct": 0,
        "identity_total": 0,
        "identity_baseline_accuracy": None,
        "reserved_before_refinement": True,
        "acceptance_decidable": False,
        "acceptance_reason": "transition_source_absent",
    }


def _episode_result(
    schedule_row: Mapping[str, Any],
    *,
    policy: Any,
    run_row: Mapping[str, Any],
    requests: Sequence[Mapping[str, Any]],
    action_rows: Sequence[Mapping[str, Any]],
    wall_s: float,
) -> JsonDict:  # pragma: no cover - live row projection.
    """Project one real policy episode into the independent row schema."""

    attempts = [deepcopy(dict(row)) for row in getattr(policy, "induction_attempts", [])]
    identity = _identity_from_attempts(attempts)
    round_rows = [
        dict(round_row)
        for attempt in attempts
        for round_row in attempt.get("refinement_rounds", [])
        if isinstance(round_row, Mapping)
    ]
    accuracies = [
        float(row["heldout_accuracy"])
        for row in round_rows
        if row.get("heldout_accuracy") is not None
    ]
    if not accuracies:
        accuracies = [
            float(row["heldout_accuracy"])
            for row in attempts
            if row.get("heldout_accuracy") is not None
        ]
    consumption = build_policy_consumption_rows(
        str(schedule_row["episode_id"]), attempts, action_rows
    )
    generated_tokens = sum(int(row.get("completion_tokens") or 0) for row in requests)
    prompt_tokens = sum(int(row.get("prompt_tokens") or 0) for row in requests)
    return {
        **deepcopy(dict(schedule_row)),
        "disposition": "complete",
        "censored": False,
        "model_invoked": bool(requests),
        "model_loaded": True,
        "generation_calls_attempted": len(requests),
        "generation_calls_completed": sum(
            row.get("transport_completed") is True for row in requests
        ),
        "usable_answers": sum(row.get("usable_answer") is True for row in requests),
        "generated_tokens": generated_tokens,
        "action_count": int(run_row.get("actions") or len(action_rows)),
        "action_limit": ACTION_LIMIT,
        "identity_baseline_accuracy": identity["identity_baseline_accuracy"],
        "identity_baseline_receipt": identity,
        "heldout_accuracy": max(accuracies) if accuracies else None,
        "non_identity_predictions": sum(
            row.get("engine_functionally_identity") is False for row in round_rows
        ),
        "candidate_valid": bool(any(_attempt_engine_digest(row) for row in attempts)),
        "trust_accepted": any(
            row.get("accepted_by_heldout_verifier") is True for row in round_rows
        ),
        "installed_plans": sum(row.get("planned") is True for row in attempts),
        "model_planned_actions": len(consumption),
        "levels": int(run_row.get("levels") or run_row.get("reached") or 0),
        "action_cost": int(run_row.get("actions") or len(action_rows)),
        "compute_cost": {
            "wall_s": round(float(wall_s), 6),
            "prompt_tokens": prompt_tokens,
            "generated_tokens": generated_tokens,
        },
        "policy_consumption_rows": consumption,
        "adapter_disabled": True,
        "typed_witness_deliveries": sum(
            bool((row.get("tool_loop") or {}).get("transition_witness", {}).get("delivered"))
            for row in round_rows
        ),
        "induction_rows": attempts,
        "action_rows": [deepcopy(dict(row)) for row in action_rows],
        "raw_request_manifest": [deepcopy(dict(row)) for row in requests],
        "error": None,
    }


def _offload_layers(log_path: Path | None) -> int | None:  # pragma: no cover - native log.
    """Read the runtime's layer-offload count from its own log."""

    if log_path is None or not log_path.is_file():
        return None
    import re

    text = log_path.read_text(errors="replace")
    matches = re.findall(r"offload(?:ed|ing)[^\n]*?(\d+)(?:/\d+)?[^\n]*layers", text, re.I)
    return int(matches[-1]) if matches else None


def run_live_session(args: argparse.Namespace) -> int:  # pragma: no cover - live CUDA E2E.
    """Load one Qwen server and run all four submitted-policy episodes."""

    started = time.monotonic()
    raw_dir = Path(args.raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    schedule = json.loads(Path(args.schedule_path).read_text(encoding="utf-8"))["rows"]
    authority_model_identity = {
        "hf_id": MODEL_ID,
        "model_path": str(Path(args.model_path).absolute()),
        "model_hash": str(args.model_hash),
        "quantization": QUANTIZATION,
    }
    from carnot.experiment_7318_v643_arc_authority import (
        attach_episode_authority,
        select_episode_authority,
        validate_authority_bundle_before_model_load,
    )

    authority_preflight = validate_authority_bundle_before_model_load(
        os.environ, schedule, authority_model_identity
    )
    if authority_preflight.get("allowed") is not True:
        raise RuntimeError(
            f"ARC episode authority rejected before model load: {authority_preflight.get('reason')}"
        )
    checkpoint = Path(args.checkpoint_path)
    session_path = Path(args.session_path)
    capture = RequestCapture(raw_dir)
    proposer: Any = None
    episode_rows: list[JsonDict] = []
    phase_spans: list[JsonDict] = []
    session: JsonDict = {
        "model_loaded": False,
        "model_invoked": False,
        "episodes": episode_rows,
        "requests": capture.rows,
        "phase_spans": phase_spans,
        "error": None,
    }
    try:
        capture.install()
        from carnot.agentic.arc_executable_world_model import LocalGGUFProposer

        load_start = time.monotonic()
        _progress("model_load", "BEFORE model load", model_path=args.model_path)
        proposer = LocalGGUFProposer(
            repo_substr="Qwen3.8-27B",
            model_path=str(Path(args.model_path).absolute()),
            port=int(args.port),
            mtp=False,
            kv_quant="q8_0",
            use_chat_template=True,
            n_gpu_layers=999,
            n_ctx=N_CTX,
            max_tokens=TOKENS_PER_CALL,
            timeout=600,
            tries=1,
        )
        if not proposer._ensure_server():
            raise RuntimeError("llama_server_startup_failed")
        load_duration = time.monotonic() - load_start
        _progress("model_load", "AFTER model load", elapsed_s=round(load_duration, 6))
        phase_spans.append({"phase": "model_load", "duration_s": load_duration})
        server = proposer._proc
        server_pid = getattr(server, "pid", None)
        log_path = getattr(proposer, "_stderr_log_path", None)
        session["model_loaded"] = True
        session["model_spec"] = {
            "hf_id": MODEL_ID,
            "quantization": QUANTIZATION,
            "model_path": str(Path(args.model_path).absolute()),
            "model_file_hash": args.model_hash,
            "revision": getattr(proposer, "model_revision", None),
        }
        session["runtime_receipt"] = {
            "runner": "LocalGGUFProposer_native_llama.cpp",
            "native_binary": proposer.last_launch_argv[0],
            "server_command": list(proposer.last_launch_argv),
            "server_pid": server_pid,
            "server_pid_start_tick": scored._process_start_tick(server_pid) if server_pid else None,
            "offload_layers_requested": 999,
            "offload_layers_observed": _offload_layers(Path(log_path) if log_path else None),
            "server_log_path": str(log_path) if log_path else None,
            "embedded_chat_template": True,
            "embedded_tokenizer": True,
            "context_tokens": N_CTX,
            "kv_quantization": "q8_0",
            "parallel_slots": 1,
            "tokens_per_call": TOKENS_PER_CALL,
            "completion_limit_per_episode": COMPLETION_LIMIT,
            "generated_token_limit_per_episode": GENERATED_TOKEN_LIMIT,
        }
        atomic_write(
            checkpoint,
            {
                "status": "running",
                "stage": "model_loaded",
                "model_loaded": True,
                "server_pid": server_pid,
                "completed_units": 0,
            },
        )
        if str(REPO_ROOT / "scripts") not in sys.path:
            sys.path.insert(0, str(REPO_ROOT / "scripts"))
        from arc_leaderboard_eval import ProgressWriter, run_game
        from carnot.agentic import arc_executable_world_model as e3

        for index, schedule_row in enumerate(schedule):
            episode_id = str(schedule_row["episode_id"])
            episode_dir = raw_dir / episode_id.replace(":", "__")
            episode_dir.mkdir(parents=True, exist_ok=True)
            capture.begin_episode(episode_id)
            environment = episode_environment(
                os.environ,
                arm=str(schedule_row["arm"]),
                episode_dir=episode_dir,
                gpu_index=int(args.gpu_index),
                port=int(args.port),
            )
            episode_authority, environment = select_episode_authority(
                environment, schedule_row, authority_model_identity
            )
            old_environment = dict(os.environ)
            old_e3_dir = e3.E3_DIR
            episode_start = time.monotonic()
            _progress(
                "episode",
                "BEFORE benchmark",
                episode_id=episode_id,
                completed_units=index,
                total_units=len(schedule),
            )
            try:
                os.environ.clear()
                os.environ.update(environment)
                e3.E3_DIR = episode_dir / "e3"
                policy, factory = scored.build_disposable_submitted_policy(
                    str(schedule_row["game"]), proposer
                )
                attach_episode_authority(policy.proposer, episode_authority)
                policy.think_arm_fallback_enabled = False
                policy.induction_progress_hook = lambda kind, payload: _progress(
                    "episode_generation", kind, episode_id=episode_id, **payload
                )
                progress = ProgressWriter(
                    episode_dir / "run_progress.json",
                    game=str(schedule_row["game"]),
                    game_index=index,
                    games_planned=len(schedule),
                    policy=policy,
                )
                run_row = run_game(
                    str(schedule_row["game"]), policy, budget=ACTION_LIMIT, progress=progress
                )
                recorder = policy.action_provenance()
                action_rows = deepcopy(recorder.rows if recorder is not None else [])
                if recorder is not None:
                    recorder.flush()
                row = _episode_result(
                    schedule_row,
                    policy=policy,
                    run_row=run_row,
                    requests=capture.episode_rows(episode_id),
                    action_rows=action_rows,
                    wall_s=time.monotonic() - episode_start,
                )
                row["factory_receipt"] = factory
            except Exception as exc:
                requests = capture.episode_rows(episode_id)
                row = {
                    **deepcopy(dict(schedule_row)),
                    "disposition": "complete",
                    "censored": False,
                    "model_invoked": bool(requests),
                    "model_loaded": True,
                    "generation_calls_attempted": len(requests),
                    "generation_calls_completed": sum(
                        item.get("transport_completed") is True for item in requests
                    ),
                    "usable_answers": sum(item.get("usable_answer") is True for item in requests),
                    "generated_tokens": sum(
                        int(item.get("completion_tokens") or 0) for item in requests
                    ),
                    "action_count": 0,
                    "action_limit": ACTION_LIMIT,
                    "identity_baseline_accuracy": None,
                    "heldout_accuracy": None,
                    "non_identity_predictions": 0,
                    "candidate_valid": False,
                    "trust_accepted": False,
                    "installed_plans": 0,
                    "model_planned_actions": 0,
                    "levels": 0,
                    "action_cost": 0,
                    "compute_cost": {"wall_s": time.monotonic() - episode_start},
                    "policy_consumption_rows": [],
                    "adapter_disabled": True,
                    "raw_request_manifest": deepcopy(requests),
                    "error": f"{type(exc).__name__}: {exc}"[:500],
                }
            finally:
                e3.E3_DIR = old_e3_dir
                os.environ.clear()
                os.environ.update(old_environment)
            episode_rows.append(row)
            atomic_write(raw_dir / "episode_rows.json", {"rows": episode_rows})
            atomic_write(
                checkpoint,
                {
                    "status": "running",
                    "stage": "episodes",
                    "model_loaded": True,
                    "server_pid": server_pid,
                    "completed_units": len(episode_rows),
                    "total_units": len(schedule),
                },
            )
            _progress(
                "episode",
                "AFTER benchmark",
                episode_id=episode_id,
                completed_units=len(episode_rows),
                total_units=len(schedule),
                elapsed_s=round(time.monotonic() - episode_start, 6),
            )
    except Exception as exc:
        session["error"] = f"{type(exc).__name__}: {exc}"[:500]
        _progress("live_session", "error", error=session["error"])
    finally:
        capture.restore()
        session["model_invoked"] = bool(capture.rows)
        _progress(
            "model_unload",
            "BEFORE model unload",
            pid=getattr(getattr(proposer, "_proc", None), "pid", None),
        )
        if proposer is not None:
            proposer.stop()
        _progress("model_unload", "AFTER model unload")
        session["duration_s"] = time.monotonic() - started
        atomic_write(session_path, session)
        atomic_write(
            checkpoint,
            {
                "status": "running",
                "stage": "child_terminal",
                "model_loaded": bool(session.get("model_loaded")),
                "completed_units": len(episode_rows),
                "terminal_child": True,
            },
        )
    return 0


def _owned_gpu_sample(gpu_index: int, process_group: int) -> JsonDict:  # pragma: no cover
    """Record only CUDA processes owned by this task's process group."""

    value = scored._owned_gpu_samples(gpu_index, process_group)
    value["sampled_at_utc"] = _iso_now()
    return value


def run_child_with_lease(
    *,
    resources: Mapping[str, Any],
    schedule_path: Path,
    raw_dir: Path,
    checkpoint_path: Path,
    session_path: Path,
    remaining_s: float,
) -> JsonDict:  # pragma: no cover - live process and lease.
    """Own one GPU lease and one process group for the complete live window."""

    from carnot.experiment_7318_v643_arc_authority import authority_bundle_environment
    from carnot.gpu_lease_phase_journal import GpuLease

    gpu = dict(resources["gpu"])
    lease = GpuLease.acquire(
        runtime_dir=checkpoint_path.parent / "gpu_leases",
        task_id=TASK_ID,
        device_uuid=str(gpu["uuid"]),
        expected_model=str(resources["model_path"]),
        vram_before_mb=int(gpu["total_memory_mb"]) - int(gpu["free_memory_mb"]),
        ttl_s=90,
    )
    lease.transition("admitted")
    lease.transition("loading")
    port = scored._free_port()
    command = [
        sys.executable,
        "-u",
        str(REPO_ROOT / WRAPPER_PATH),
        "--role",
        "live-session",
        "--date",
        RUN_DATE,
        "--model-path",
        str(resources["model_path"]),
        "--model-hash",
        str(resources["model_hash"]),
        "--gpu-index",
        str(gpu["index"]),
        "--port",
        str(port),
        "--schedule-path",
        str(schedule_path),
        "--raw-dir",
        str(raw_dir),
        "--checkpoint-path",
        str(checkpoint_path),
        "--session-path",
        str(session_path),
    ]
    env = episode_environment(
        os.environ,
        arm="current_feedback",
        episode_dir=raw_dir / "bootstrap",
        gpu_index=int(gpu["index"]),
        port=port,
    )
    env["CARNOT_ARC_GGUF_PATH"] = str(resources["model_path"])
    env["CARNOT_LLAMA_SERVER"] = str(resources["server"])
    schedule = json.loads(schedule_path.read_text(encoding="utf-8"))["rows"]
    model_identity = {
        "hf_id": MODEL_ID,
        "model_path": str(Path(str(resources["model_path"])).absolute()),
        "model_hash": str(resources["model_hash"]),
        "quantization": QUANTIZATION,
    }
    authority_valid_for_s = min(float(remaining_s), float(LIVE_WINDOW_S)) + 60.0
    authorities = [
        lease.issue_arc_episode_authority(
            episode_id=str(row["episode_id"]),
            game=str(row["game"]),
            model_identity=model_identity,
            resource_bounds={
                "action_limit": ACTION_LIMIT,
                "completion_limit": COMPLETION_LIMIT,
                "generated_token_limit": GENERATED_TOKEN_LIMIT,
                "session_limit_s": LIVE_WINDOW_S,
            },
            nonce_ledger_path=(
                raw_dir / "authority_nonces" / str(row["episode_id"]).replace(":", "__")
            ),
            valid_for_s=authority_valid_for_s,
        )
        for row in schedule
    ]
    env = authority_bundle_environment(env, authorities)
    _progress("live_subprocess", "BEFORE subprocess", command=command)
    process = subprocess.Popen(
        command,
        cwd=REPO_ROOT,
        env=env,
        start_new_session=True,
    )
    started = time.monotonic()
    next_beat = started
    samples: list[JsonDict] = []
    resident = False
    timed_out = False
    cap = min(float(remaining_s), float(LIVE_WINDOW_S))
    while process.poll() is None:
        now = time.monotonic()
        if now - started >= cap:
            timed_out = True
            break
        progress: JsonDict = {}
        if checkpoint_path.is_file():
            try:
                progress = json.loads(checkpoint_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                progress = {}
        if progress.get("model_loaded") and not resident:
            sample = _owned_gpu_sample(int(gpu["index"]), process.pid)
            samples.append(sample)
            owned = sample["owned_compute_apps"]
            resident = bool(owned)
            if resident:
                lease.transition(
                    "resident", vram_mb=sum(int(row["used_memory_mb"]) for row in owned)
                )
                lease.transition("inferencing")
        if not resident and now - started >= STARTUP_TIMEOUT_S:
            timed_out = True
            break
        if now >= next_beat:
            lease.heartbeat()
            sample = _owned_gpu_sample(int(gpu["index"]), process.pid)
            samples.append(sample)
            _progress(
                "live_subprocess",
                "heartbeat",
                elapsed_s=round(now - started, 1),
                completed_units=int(progress.get("completed_units") or 0),
                model_loaded=bool(progress.get("model_loaded")),
                owned_gpu_processes=len(sample["owned_compute_apps"]),
            )
            next_beat = now + 45
        time.sleep(1)
    signals_sent: list[str] = []
    if process.poll() is None:
        os.killpg(process.pid, signal.SIGTERM)
        signals_sent.append("SIGTERM:task_process_group")
        try:
            process.wait(timeout=20)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            signals_sent.append("SIGKILL:task_process_group")
            process.wait(timeout=10)
    _progress(
        "live_subprocess",
        "AFTER subprocess",
        returncode=process.returncode,
        timed_out=timed_out,
        elapsed_s=round(time.monotonic() - started, 6),
    )
    try:
        session = json.loads(session_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        # A bounded parent can stop the child after an episode checkpoint but before
        # the final session write. Keep those durable rows so downstream accounting
        # does not replace real actions and generations with synthetic zeroes.
        durable = {}
        try:
            durable = json.loads((raw_dir / "episode_rows.json").read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            pass
        progress_rows = {}
        for sealed in schedule:
            episode_id = str(sealed.get("episode_id"))
            progress_path = raw_dir / episode_id.replace(":", "__") / "run_progress.json"
            try:
                progress_rows[episode_id] = json.loads(progress_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                pass
        from carnot.experiment_7384_v648_arc_invocation_boundary import (
            recover_timeout_accounting,
        )

        recovered = recover_timeout_accounting(
            schedule=schedule,
            terminal_session=None,
            durable_episode_rows=durable.get("rows", []),
            progress_rows=progress_rows,
            timed_out=timed_out,
        )
        session = {
            "model_loaded": resident,
            "model_invoked": any(
                row.get("model_invoked") is True for row in durable.get("rows", [])
            ),
            "episodes": recovered["rows"],
            "requests": [],
            "phase_spans": [],
            "runtime_receipt": {},
            "timeout_accounting": recovered["accounting"],
            "last_confirmed_event": recovered["last_confirmed_event"],
            "first_missing_event": recovered["first_missing_event"],
            "error": (
                "live_child_did_not_write_session_recovered_episode_checkpoints"
                if durable.get("rows")
                else "live_child_did_not_write_session"
            ),
        }
    if resident:
        lease.transition("unloading")
        lease.transition(
            "validating",
            vram_mb=0,
            exit_code=int(process.returncode or 0),
            unload_observed=True,
        )
        lease.transition("terminal_complete")
    else:
        lease.transition("terminal_blocked")
    release = lease.release()
    runtime = dict(session.get("runtime_receipt", {}))
    owned = [row for sample in samples for row in sample["owned_compute_apps"]]
    runtime.update(
        {
            "gpu_uuid": gpu["uuid"],
            "gpu_index": gpu["index"],
            "gpu_name": gpu["name"],
            "vram_total_mb": gpu["total_memory_mb"],
            "vram_free_before_mb": gpu["free_memory_mb"],
            "task_linked_cuda_execution": bool(owned),
            "gpu_samples": samples,
            "lease_owner": lease.owner_receipt(),
            "lease_release": release,
            "signals_sent": signals_sent,
            "timed_out": timed_out,
        }
    )
    session["runtime_receipt"] = runtime
    session["model_loaded"] = bool(session.get("model_loaded") and resident)
    session["model_invoked"] = bool(session.get("model_invoked") and owned)
    session["timed_out"] = timed_out
    atomic_write(session_path, session)
    return session


def _censored_rows(
    schedule: Sequence[Mapping[str, Any]], existing: Sequence[Mapping[str, Any]]
) -> list[JsonDict]:  # pragma: no cover - live timeout reduction.
    """Account for each unfinished scheduled episode as a censored row."""

    rows = [deepcopy(dict(row)) for row in existing]
    done = {str(row.get("episode_id")) for row in rows}
    for schedule_row in schedule:
        if str(schedule_row["episode_id"]) in done:
            continue
        rows.append(
            {
                **deepcopy(dict(schedule_row)),
                "disposition": "censored_timeout",
                "censored": True,
                "model_invoked": False,
                "model_loaded": True,
                "generation_calls_attempted": 0,
                "generation_calls_completed": 0,
                "usable_answers": 0,
                "generated_tokens": 0,
                "action_count": 0,
                "action_limit": ACTION_LIMIT,
                "identity_baseline_accuracy": None,
                "heldout_accuracy": None,
                "non_identity_predictions": 0,
                "candidate_valid": False,
                "trust_accepted": False,
                "installed_plans": 0,
                "model_planned_actions": 0,
                "levels": 0,
                "action_cost": 0,
                "compute_cost": {"wall_s": 0.0},
                "policy_consumption_rows": [],
                "adapter_disabled": True,
                "raw_request_manifest": [],
                "error": "whole_live_window_timeout",
            }
        )
    return rows


def _run_validation_rows(
    commands: Sequence[Mapping[str, Any]], *, raw_dir: Path
) -> list[JsonDict]:  # pragma: no cover - validation subprocesses.
    """Stream scoped validation with truthful external heartbeats and log hashes."""

    from carnot.experiment_7246_v638_source_map import _run_streaming_command

    rows: list[JsonDict] = []
    validation_dir = raw_dir / "validation"
    validation_dir.mkdir(parents=True, exist_ok=True)
    for index, command_row in enumerate(commands):
        name = str(command_row["name"])
        command = [str(part) for part in command_row["command"]]
        _progress(
            "validation",
            "BEFORE subprocess",
            name=name,
            completed_units=index,
            total_units=len(commands),
        )
        result = _run_streaming_command(
            command,
            cwd=REPO_ROOT,
            timeout_s=900,
            heartbeat_s=45,
            operation=f"exp7263:{name}",
        )
        output = str(result.get("output") or result.get("stdout") or "")
        log_path = validation_dir / f"{index:02d}_{name}.log"
        log_path.write_text(output, encoding="utf-8")
        row = {
            "name": name,
            "command": " ".join(command),
            "exit_code": int(result["exit_code"]),
            "expected_exit_code": 0,
            "passed": int(result["exit_code"]) == 0,
            "timed_out": bool(result.get("timed_out")),
            "duration_s": float(result["duration_s"]),
            "log_path": str(log_path.relative_to(REPO_ROOT)),
            "log_sha256": _sha256_file(log_path),
            "output_tail": output[-4000:],
        }
        rows.append(row)
        _progress(
            "validation",
            "AFTER subprocess",
            name=name,
            passed=row["passed"],
            completed_units=len(rows),
            total_units=len(commands),
        )
    return rows


def run_experiment(args: argparse.Namespace) -> JsonDict:  # pragma: no cover - live task.
    """Run preflight, four live episodes, validation, and atomic publication."""

    started = time.monotonic()
    started_utc = _iso_now()
    root = REPO_ROOT
    result_path = args.result_path if args.result_path.is_absolute() else root / args.result_path
    checkpoint_path = (
        args.checkpoint_path if args.checkpoint_path.is_absolute() else root / args.checkpoint_path
    )
    raw_dir = args.raw_dir if args.raw_dir.is_absolute() else root / args.raw_dir
    raw_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    _progress("startup", "BEGIN", run_date=args.date, elapsed_s=0.0)
    _progress("preconditions", "BEGIN")
    checks, hashes, resources = collect_preconditions(root)
    _progress(
        "preconditions",
        "END",
        passed=all(row["passed"] for row in checks),
        completed_units=len(checks),
    )
    if not all(row["passed"] for row in checks):
        artifact = build_blocked_artifact(
            started_at_utc=started_utc,
            ended_at_utc=_iso_now(),
            duration_s=time.monotonic() - started,
            preconditions=checks,
            source_hashes=hashes,
        )
        internal_errors = validate_artifact(artifact)
        if internal_errors:
            raise RuntimeError(f"blocked artifact failed validation: {internal_errors}")
        atomic_write(result_path, artifact)
        _progress("terminal", "published_blocked", path=str(result_path))
        return artifact
    _progress("selection", "BEGIN registry precheck before roster selection")
    registry = _load_registry(root / REGISTRY_PATH)
    from carnot.agentic.arc_game_adapters import adaptered_games

    selection = select_frozen_roster(registry, adaptered_games=set(adaptered_games()))
    schedule = counterbalanced_schedule(selection["games"], RANDOM_SEED)
    selection_ok = len(selection["games"]) == 2 and len(schedule) == 4
    selection_check = gate_check(
        "registry_precheck_before_roster",
        REGISTRY_PATH.as_posix(),
        "four_counterbalanced_units",
        True,
        selection_ok,
    )
    checks.append(selection_check)
    schedule_path = raw_dir / "frozen_schedule.json"
    atomic_write(schedule_path, {"selection_receipt": selection, "rows": schedule})
    hashes[str(schedule_path.relative_to(root))] = _sha256_file(schedule_path)
    _progress("selection", "END", games=selection["games"], completed_units=4)
    if not selection_ok:
        artifact = build_blocked_artifact(
            started_at_utc=started_utc,
            ended_at_utc=_iso_now(),
            duration_s=time.monotonic() - started,
            preconditions=checks,
            source_hashes=hashes,
        )
        atomic_write(result_path, artifact)
        return artifact
    atomic_write(
        checkpoint_path,
        {
            "status": "running",
            "stage": "live_window",
            "terminal": False,
            "selection_receipt": selection,
            "completed_units": 0,
        },
    )
    _progress("live_window", "BEGIN", max_seconds=LIVE_WINDOW_S, planned_units=4)
    live_start = time.monotonic()
    session = run_child_with_lease(
        resources=resources,
        schedule_path=schedule_path,
        raw_dir=raw_dir,
        checkpoint_path=checkpoint_path,
        session_path=raw_dir / "live_session.json",
        remaining_s=max(1.0, LIVE_WINDOW_S - (time.monotonic() - live_start)),
    )
    _progress(
        "live_window",
        "END",
        model_loaded=session.get("model_loaded"),
        model_invoked=session.get("model_invoked"),
        completed_units=len(session.get("episodes", [])),
        elapsed_s=round(time.monotonic() - live_start, 6),
    )
    if not session.get("model_loaded"):
        load_failure = gate_check(
            "live_model_load",
            MODEL_ID,
            "task_owned_gpu_resident_model",
            True,
            False,
        )
        checks.append(load_failure)
        artifact = build_blocked_artifact(
            started_at_utc=started_utc,
            ended_at_utc=_iso_now(),
            duration_s=time.monotonic() - started,
            preconditions=checks,
            source_hashes=hashes,
        )
        atomic_write(result_path, artifact)
        _progress("terminal", "published_blocked_no_run", path=str(result_path))
        return artifact
    episode_rows = _censored_rows(schedule, session.get("episodes", []))
    atomic_write(raw_dir / "episode_rows.json", {"rows": episode_rows})
    for path in raw_dir.rglob("*"):
        if path.is_file() and path != raw_dir / "terminal_candidate.json":
            hashes[str(path.relative_to(root))] = _sha256_file(path)
    model_spec = dict(session.get("model_spec", {}))
    runtime_receipt = dict(session.get("runtime_receipt", {}))
    phase_spans = list(session.get("phase_spans", []))
    preliminary = build_terminal_artifact(
        started_at_utc=started_utc,
        ended_at_utc=_iso_now(),
        duration_s=time.monotonic() - started,
        preconditions=checks,
        source_hashes=hashes,
        selection_receipt=selection,
        episode_rows=episode_rows,
        model_spec=model_spec,
        runtime_receipt=runtime_receipt,
        validation_receipts=[],
        phase_spans=phase_spans,
    )
    atomic_write(raw_dir / "terminal_candidate.json", preliminary)
    commands = build_validation_commands(
        terminal_candidate=raw_dir / "terminal_candidate.json",
        raw_rows=raw_dir / "episode_rows.json",
    )
    _progress("validation", "BEGIN", total_units=len(commands))
    validation_rows = _run_validation_rows(commands[:-2], raw_dir=raw_dir)
    candidate = build_terminal_artifact(
        started_at_utc=started_utc,
        ended_at_utc=_iso_now(),
        duration_s=time.monotonic() - started,
        preconditions=checks,
        source_hashes=hashes,
        selection_receipt=selection,
        episode_rows=episode_rows,
        model_spec=model_spec,
        runtime_receipt=runtime_receipt,
        validation_receipts=validation_rows,
        phase_spans=phase_spans,
    )
    atomic_write(raw_dir / "terminal_candidate.json", candidate)
    checker_rows = _run_validation_rows(commands[-2:], raw_dir=raw_dir)
    validation_rows.extend(checker_rows)
    artifact = build_terminal_artifact(
        started_at_utc=started_utc,
        ended_at_utc=_iso_now(),
        duration_s=time.monotonic() - started,
        preconditions=checks,
        source_hashes=hashes,
        selection_receipt=selection,
        episode_rows=episode_rows,
        model_spec=model_spec,
        runtime_receipt=runtime_receipt,
        validation_receipts=validation_rows,
        phase_spans=phase_spans,
    )
    internal_errors = validate_artifact(artifact)
    if internal_errors:
        raise RuntimeError(f"terminal artifact failed internal validation: {internal_errors}")
    atomic_write(raw_dir / "terminal_candidate.json", artifact)
    atomic_write(result_path, artifact)
    atomic_write(
        checkpoint_path,
        {
            "status": "complete",
            "stage": "terminal_published",
            "terminal": True,
            "completed_units": 4,
            "result_path": str(result_path),
        },
    )
    _progress(
        "validation",
        "END",
        passed=all(row["passed"] for row in validation_rows),
        completed_units=len(validation_rows),
    )
    _progress(
        "terminal",
        "published",
        path=str(result_path),
        honest_verdict=artifact["honest_verdict"],
    )
    return artifact


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:  # pragma: no cover
    """Parse the thin experiment and child-session command line."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--date", default=RUN_DATE)
    parser.add_argument("--role", choices=("experiment", "live-session"), default="experiment")
    parser.add_argument("--result-path", type=Path, default=RESULT_PATH)
    parser.add_argument("--checkpoint-path", type=Path, default=CHECKPOINT_PATH)
    parser.add_argument("--raw-dir", type=Path, default=RAW_DIR)
    parser.add_argument("--reduce-raw", type=Path)
    parser.add_argument("--model-path")
    parser.add_argument("--model-hash")
    parser.add_argument("--gpu-index", type=int)
    parser.add_argument("--port", type=int)
    parser.add_argument("--schedule-path", type=Path)
    parser.add_argument("--session-path", type=Path, default=SESSION_PATH)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - thin CLI.
    """Dispatch raw reduction, the live child, or the complete experiment."""

    _progress("startup", "entrypoint")
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


if __name__ == "__main__":  # pragma: no cover - module execution.
    raise SystemExit(main())
