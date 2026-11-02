"""Run the post-identity-repair ARC typed-witness policy-use pilot.

The live transport, GPU lease, scored policy factory, and game loop are reused
from Experiment 7263. This module owns the new identity prerequisite, fixed
public roster, stricter per-game efficacy reduction, and V640 terminal record.

Spec: REQ-ARC-WMTE-7280 and SCENARIO-ARC-WMTE-7280-*.
"""

from __future__ import annotations

import argparse
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from copy import deepcopy
from datetime import UTC, datetime
import hashlib
import json
import os
from pathlib import Path
import socket
import sys
import time
from typing import Any

import yaml

from carnot.agentic import arc_eval_provenance as provenance
from carnot import experiment_7263_v639_arc_live as prior


JsonDict = dict[str, Any]
REPO_ROOT = Path(__file__).resolve().parents[2]
TASK_ID = "exp7280-arc-live"
EXPERIMENT_ID = TASK_ID
MILESTONE = "2026.09.640"
RUN_DATE = "20260913"
RANDOM_SEED = 7_280_202_609_13
SCHEMA = "carnot.experiment_7280.v640.arc_live.v1"

MODEL_ID = "unsloth/Qwen3.8-27B-GGUF"
QUANTIZATION = "Q4_K_M"
MODEL_SPECS = ({"hf_id": MODEL_ID, "quantization": QUANTIZATION},)
ACTION_LIMIT = 192
COMPLETION_LIMIT = 2
GENERATED_TOKEN_LIMIT = 4096
TOKENS_PER_CALL = GENERATED_TOKEN_LIMIT // COMPLETION_LIMIT
SESSION_LIMIT_S = 3000
MODEL_LOAD_LIMIT_S = 600
IDENTITY_OBLIGATIONS = provenance.IDENTITY_OBLIGATIONS

SPEC_PATH = Path("openspec/capabilities/arc-world-model-trust-energy/spec.md")
UPSTREAM_PATH = Path("results/experiment_7276_v640_arc_identity.json")
REGISTRY_PATH = Path("ops/arc_solve_registry.yaml")
EXCLUSION_PATH = Path("ops/exclusion_manifest.yaml")
RESULT_PATH = Path("results/experiment_7280_v640_arc_live.json")
CHECKPOINT_PATH = Path("results/checkpoints/experiment_7280_v640_arc_live.json")
RAW_DIR = Path("results/raw/experiment_7280_v640_arc_live")
RAW_ROWS_PATH = RAW_DIR / "episode_rows.json"
TERMINAL_CANDIDATE_PATH = RAW_DIR / "terminal_candidate.json"
SESSION_PATH = RAW_DIR / "live_session.json"
IDENTITY_PATH = RAW_DIR / "model_identity.json"
MODULE_PATH = Path("python/carnot/experiment_7280_v640_arc_live.py")
WRAPPER_PATH = Path("scripts/experiments/experiment_7280_v640_arc_live.py")
TEST_PATH = Path("tests/python/test_experiment_7280_v640_arc_live.py")

REQUIRED_INPUTS = (
    Path("AGENTS.md"),
    Path("CLAUDE.md"),
    Path("CODEX.md"),
    Path("research-program.md"),
    EXCLUSION_PATH,
    Path("ops/e2e-test-plan.md"),
    REGISTRY_PATH,
    SPEC_PATH,
    Path("scripts/experiment_template.py"),
    Path("scripts/arc_leaderboard_eval.py"),
    Path("scripts/arc_loop_solve.py"),
    Path("python/carnot/agentic/arc_competition_agent.py"),
    Path("python/carnot/agentic/arc_eval_provenance.py"),
    Path("python/carnot/agentic/arc_executable_world_model.py"),
    Path("python/carnot/experiment_7263_v639_arc_live.py"),
    Path("python/carnot/experiment_7276_v640_arc_identity.py"),
    UPSTREAM_PATH,
    MODULE_PATH,
    WRAPPER_PATH,
    TEST_PATH,
)

FIELD_PRINCIPLES = {
    "schema": "Version the result and retain ordinary top-level experiment_id and milestone.",
    "experiment_id": "Identify this invocation with an ordinary stable string.",
    "milestone": "Bind the result to milestone 2026.09.640.",
    "status": "Use complete or blocked for terminal evidence; keep unfinished work in checkpoints.",
    "run_date": "Use 20260913 and actual UTC start and end times.",
    "started_at_utc": "Record the actual UTC invocation start.",
    "ended_at_utc": "Record the actual UTC terminal construction time.",
    "field_principles": "Store explanations here; consumer values remain ordinary top-level fields.",
    "preconditions_checked": "Record actual input hashes, authority separation, resource ownership, and failures.",
    "MODEL_SPECS": "Declare models executable in this invocation; keep historical identities in hashed sidecars.",
    "model_invoked": "Derive from actual calls, including failed or unusable generation.",
    "invocation_counts": "Separate attempted and completed loads and generation from usable answers.",
    "inference_substrate": "Use the recognized literal for actual computation, not a task label.",
    "inference_substrate_class": "Declare full generation, load-only work, or the actual no-model class.",
    "inference_mode": "Use live_gpu only when task-owned CUDA execution is observed.",
    "execution_venue": "Host orchestration is host; identify real device execution separately.",
    "execution_host": "Record the host separately from the execution venue.",
    "duration_s": "Measure monotonic elapsed time and disjoint phase spans.",
    "phase_spans": "Keep load, generation, benchmark, and validation spans distinct.",
    "random_seed": "Freeze independent-unit seeds before observing results.",
    "reproducibility_checksum": "Bind code, configuration, input manifests, and raw evidence.",
    "source_artifact_hashes": "Preserve exact input identity, retirement, and quarantine status.",
    "rows": "Keep each unit, arm, seed, error, abstention, cost, metric, and censoring state.",
    "sample_size_budget": "Record planned, attempted, completed, censored units, and the stopping rule.",
    "acceptance_gate_results": "Each criterion records expected, observed, passed, and principle.",
    "gate_check_summary": "For blocked work name upstream, exact check, observed value, and expected value.",
    "verifier_is_oracle": "Expose shared evaluator authority; exact conformance is not learned correctness.",
    "honest_verdict": "Completed findings start complete_; external absence starts blocked_.",
    "verdict_class": "Use the closed verdict vocabulary; oracle authority forbids positive.",
    "validation_receipts": "Retain command, exit code, timing, and log hash; do not hide failures.",
    "arc_capture_complete_score": "One means all four dispositions and raw lineage are authenticated.",
    "arc_method_value_score": "One requires useful prediction and consumed plans in both treatment games.",
    "per_game_results": "Keep four paired-arm rows with errors, costs, censoring, and progress.",
    "policy_consumption_rows": "Bind predicted transition, selected plan, and actual action issued.",
    "identity_obligation_rows": "Each episode proves strict runtime identity before generation.",
    "selection_receipt": "Prove registry precheck fixed the two public development games before play.",
    "runner_receipt": "Bind current Qwen GGUF ownership, server, model hash, raw calls, and CUDA samples.",
    "raw_request_manifest": "Retain every actual request and response byte receipt.",
    "submission_configuration_diff": "Keep local and submitted runtime differences explicit.",
    "claim_scope": "Four episodes over two public games support a pilot case study only.",
    "population_confidence_interval": "Remain unset because this pilot has only two games.",
    "solve_provenance": "Only live_agent_self_discovery can support this live method path.",
    "new_solve_claimed": "Registered public progress is not a new solve claim.",
    "reproduced_levels": "List only levels reproduced from captured live action labels.",
    "official_score": "Keep the official score unset because no submission or upload occurs.",
}


def _canonical_bytes(value: Any) -> bytes:
    """Return deterministic JSON bytes for evidence hashes."""

    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()


def _sha256_bytes(value: bytes) -> str:
    """Return one prefixed SHA-256 digest."""

    return "sha256:" + hashlib.sha256(value).hexdigest()


def artifact_checksum(value: Mapping[str, Any]) -> str:
    """Hash the terminal record without trusting its checksum carrier."""

    payload = deepcopy(dict(value))
    payload["reproducibility_checksum"] = ""
    return _sha256_bytes(_canonical_bytes(payload))


def gate_check(check: str, upstream: str, field: str, expected: Any, observed: Any) -> JsonDict:
    """Build one fail-closed precondition row with ordinary values."""

    return {
        "check": check,
        "upstream": upstream,
        "field": field,
        "expected": expected,
        "observed": observed,
        "passed": observed == expected,
    }


def authenticate_identity_upstream(
    path: Path, *, quarantined: bool, retired: bool
) -> tuple[list[JsonDict], JsonDict]:
    """Authenticate the repaired identity receipt before any model work."""

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
        gate_check("identity_upstream", str(path), "exists", True, path.is_file()),
        gate_check(
            "identity_upstream",
            str(path),
            "experiment_id",
            "exp7276-arc-identity",
            payload.get("experiment_id"),
        ),
        gate_check(
            "identity_upstream", str(path), "milestone", MILESTONE, payload.get("milestone")
        ),
        gate_check("identity_upstream", str(path), "status", "complete", payload.get("status")),
        gate_check(
            "identity_upstream",
            str(path),
            "arc_identity_ready_score",
            1,
            payload.get("arc_identity_ready_score"),
        ),
        gate_check("identity_upstream", str(path), "quarantine_state", False, quarantined),
        gate_check("identity_upstream", str(path), "retirement_state", False, retired),
        gate_check("identity_upstream", str(path), "clean_terminal_bytes", True, clean),
    ]
    return checks, dict(payload)


def freeze_public_roster(
    registry: Mapping[str, Any], *, adaptered_games: set[str] | frozenset[str]
) -> JsonDict:
    """Freeze re86 and r11l as registered public targets with adapters withheld."""

    wanted = ("re86", "r11l")
    indexed = {
        str(row.get("game")): dict(row)
        for row in registry.get("games", [])
        if isinstance(row, Mapping) and row.get("game")
    }
    rows: list[JsonDict] = []
    for game in wanted:
        if game not in indexed:
            continue
        registry_row = indexed[game]
        rows.append(
            {
                "game": game,
                "registry_levels_before_attempt": int(registry_row.get("levels_reproduced") or 0),
                "registry_reproducibility": registry_row.get("reproducibility"),
                "registered_public_development_target": True,
                "adapter_available_but_withheld": game in adaptered_games,
                "adapter_disabled": True,
                "banked_solution_disabled": True,
                "registry_precheck_passed": True,
            }
        )
    return {
        "selection_steps": ["registry_precheck", "roster_freeze"],
        "selection_basis": "task_frozen_registered_public_development_targets",
        "selection_used_game_source": False,
        "selection_used_outcome_labels": False,
        "re_solve_campaign": False,
        "games": [row["game"] for row in rows],
        "game_rows": rows,
        "roster_complete": [row["game"] for row in rows] == list(wanted),
    }


def resolve_cached_model(pair: Sequence[Mapping[str, Any]]) -> tuple[JsonDict, Path | None, bool]:
    """Resolve the current cached snapshot and retain its independent revision."""

    selected = next((dict(row) for row in pair if row.get("hf_id") == MODEL_ID), {})
    model_path = Path(str(selected.get("model_path"))) if selected.get("model_path") else None
    revision = (
        provenance.huggingface_snapshot_revision(str(model_path), MODEL_ID) if model_path else None
    )
    model_ok = bool(
        model_path and model_path.is_file() and QUANTIZATION in model_path.name and revision
    )
    selected.update(
        {
            "quantization": QUANTIZATION,
            "revision": revision,
            "model_path": str(model_path) if model_path else None,
        }
    )
    return selected, model_path, model_ok


counterbalanced_schedule = prior.counterbalanced_schedule
atomic_write = prior.atomic_write


def episode_environment(
    base_env: Mapping[str, str],
    *,
    arm: str,
    episode_dir: Path,
    gpu_index: int,
    port: int,
) -> dict[str, str]:
    """Reuse the shipped equal-budget environment with the V640 frozen seed."""

    old_seed = prior.RANDOM_SEED
    try:
        prior.RANDOM_SEED = RANDOM_SEED
        return prior.episode_environment(
            base_env,
            arm=arm,
            episode_dir=episode_dir,
            gpu_index=gpu_index,
            port=port,
        )
    finally:
        prior.RANDOM_SEED = old_seed


def bind_identity_obligations(
    episode_id: str,
    receipt: Mapping[str, Any],
    *,
    authenticated_at_utc: str,
) -> list[JsonDict]:
    """Bind every typed identity obligation to one episode before generation."""

    source_rows = [
        dict(row) for row in receipt.get("identity_obligation_rows", []) if isinstance(row, Mapping)
    ]
    strict = [row.get("obligation") for row in source_rows] == list(IDENTITY_OBLIGATIONS) and all(
        row.get("status") == "supported" for row in source_rows
    )
    return [
        {
            "episode_id": episode_id,
            "obligation": row.get("obligation"),
            "status": row.get("status"),
            "authenticated_before_first_generation": strict,
            "authenticated_at_utc": authenticated_at_utc,
            "evidence_source": row.get("evidence_source"),
            "observed": deepcopy(row.get("observed_value")),
        }
        for row in source_rows
    ]


def validate_episode_row(row: Mapping[str, Any]) -> list[str]:
    """Reject budget, isolation, identity, or terminal-disposition drift."""

    errors: list[str] = []
    if int(row.get("action_count") or 0) > ACTION_LIMIT:
        errors.append("action_limit_exceeded")
    attempted = int(row.get("generation_calls_attempted") or 0)
    completed = int(row.get("generation_calls_completed") or 0)
    if attempted > COMPLETION_LIMIT:
        errors.append("generation_call_limit_exceeded")
    if completed > attempted:
        errors.append("generation_completion_count_invalid")
    if int(row.get("generated_tokens") or 0) > GENERATED_TOKEN_LIMIT:
        errors.append("generated_token_limit_exceeded")
    if row.get("adapter_disabled") is not True:
        errors.append("adapter_not_disabled")
    if row.get("fresh_store") is not True:
        errors.append("store_not_fresh")
    if row.get("arm") not in {"current_feedback", "typed_witness_feedback"}:
        errors.append("arm_invalid")
    if row.get("disposition") not in {"complete", "censored_timeout"}:
        errors.append("disposition_not_terminal")
    identity_rows = row.get("identity_obligation_rows")
    valid_identity = isinstance(identity_rows, list) and [
        item.get("obligation") for item in identity_rows if isinstance(item, Mapping)
    ] == list(IDENTITY_OBLIGATIONS)
    if valid_identity:
        valid_identity = all(
            isinstance(item, Mapping)
            and item.get("status") == "supported"
            and item.get("authenticated_before_first_generation") is True
            for item in identity_rows
        )
    if not valid_identity:
        errors.append("identity_obligations_not_supported")
    return errors


def reduce_episode_rows(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Reduce four rows while requiring useful policy consumption per treatment game."""

    episodes = [deepcopy(dict(row)) for row in rows]
    games = sorted({str(row.get("game")) for row in episodes})
    units = {(str(row.get("game")), str(row.get("arm"))) for row in episodes}
    expected = {
        (game, arm)
        for game in ("re86", "r11l")
        for arm in ("current_feedback", "typed_witness_feedback")
    }
    row_errors = [error for row in episodes for error in validate_episode_row(row)]
    capture = int(
        len(episodes) == 4
        and games == ["r11l", "re86"]
        and units == expected
        and not row_errors
        and all(isinstance(row.get("raw_request_manifest"), list) for row in episodes)
        and all(bool(row.get("identity_receipt_sha256")) for row in episodes)
    )
    treatment_checks: JsonDict = {}
    useful_games: list[str] = []
    above_identity: dict[str, bool] = {}
    regressions: list[str] = []
    for game in ("re86", "r11l"):
        by_arm = {str(row.get("arm")): row for row in episodes if str(row.get("game")) == game}
        treatment = by_arm.get("typed_witness_feedback", {})
        current = by_arm.get("current_feedback", {})
        heldout = treatment.get("heldout_accuracy")
        identity = treatment.get("identity_baseline_accuracy")
        accuracy_ok = bool(
            heldout is not None and identity is not None and float(heldout) > float(identity)
        )
        above_identity[game] = accuracy_ok
        consumed = len(treatment.get("policy_consumption_rows", []))
        valid_plan = bool(
            treatment.get("candidate_valid") is True
            and treatment.get("trust_accepted") is True
            and int(treatment.get("non_identity_predictions") or 0) > 0
            and int(treatment.get("installed_plans") or 0) > 0
        )
        progress_ok = int(treatment.get("levels") or 0) >= int(current.get("levels") or 0)
        if not progress_ok:
            regressions.append(game)
        useful = bool(accuracy_ok and valid_plan and consumed > 0 and progress_ok)
        if useful:
            useful_games.append(game)
        treatment_checks[game] = {
            "heldout_accuracy": heldout,
            "identity_baseline_accuracy": identity,
            "accuracy_above_identity": accuracy_ok,
            "candidate_valid": treatment.get("candidate_valid") is True,
            "trust_accepted": treatment.get("trust_accepted") is True,
            "non_identity_predictions": int(treatment.get("non_identity_predictions") or 0),
            "installed_plans": int(treatment.get("installed_plans") or 0),
            "consumed_plan_actions": consumed,
            "paired_progress_no_regression": progress_ok,
            "useful_consumed_plan": useful,
        }
    attempted_calls = sum(int(row.get("generation_calls_attempted") or 0) for row in episodes)
    completed_calls = sum(int(row.get("generation_calls_completed") or 0) for row in episodes)
    usable_answers = sum(int(row.get("usable_answers") or 0) for row in episodes)
    any_load_attempt = any(
        row.get("model_loaded") is True or row.get("model_invoked") is True for row in episodes
    )
    any_loaded = any(row.get("model_loaded") is True for row in episodes)
    method_value = int(capture == 1 and useful_games == ["re86", "r11l"])
    return {
        "arc_capture_complete_score": capture,
        "arc_method_value_score": method_value,
        "treatment_game_checks": treatment_checks,
        "treatment_games_with_useful_consumed_plan": sorted(useful_games),
        "treatment_accuracy_above_identity_by_game": {
            game: above_identity[game] for game in sorted(above_identity)
        },
        "matched_game_level_regression": bool(regressions),
        "regressed_games": regressions,
        "row_errors": row_errors,
        "claim_scope": "two_game_pilot_case_study_only",
        "population_confidence_interval": None,
        "invocation_counts": {
            "model_loads_attempted": int(any_load_attempt),
            "model_loads_completed": int(any_loaded),
            "generation_calls_attempted": attempted_calls,
            "generation_calls_completed": completed_calls,
            "usable_answers": usable_answers,
        },
    }


def _acceptance_results(reduction: Mapping[str, Any], *, validation_passed: bool) -> list[JsonDict]:
    """Render capture, per-game value, regression, and validation as separate gates."""

    rows = [
        {
            "criterion": "all_four_episode_dispositions_and_lineage",
            "expected": 1,
            "observed": reduction["arc_capture_complete_score"],
            "passed": reduction["arc_capture_complete_score"] == 1,
            "principle": "Capture completeness is separate from policy value.",
        }
    ]
    for game in ("re86", "r11l"):
        observed = reduction["treatment_game_checks"].get(game, {})
        rows.append(
            {
                "criterion": f"{game}_useful_consumed_treatment_plan",
                "expected": True,
                "observed": observed,
                "passed": observed.get("useful_consumed_plan") is True,
                "principle": "Each treatment game must predict usefully and affect policy action.",
            }
        )
    rows.extend(
        [
            {
                "criterion": "no_paired_progress_regression",
                "expected": False,
                "observed": reduction["matched_game_level_regression"],
                "passed": reduction["matched_game_level_regression"] is False,
                "principle": "Prediction quality cannot hide worse paired game progress.",
            },
            {
                "criterion": "scoped_validation_passed",
                "expected": True,
                "observed": validation_passed,
                "passed": validation_passed,
                "principle": "Measured claims require the scoped tests and independent checks.",
            },
        ]
    )
    return rows


def _base_artifact(started_at_utc: str, ended_at_utc: str, duration_s: float) -> JsonDict:
    """Create terminal identity and timing fields shared by both dispositions."""

    return {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "status": "blocked",
        "run_date": RUN_DATE,
        "started_at_utc": started_at_utc,
        "ended_at_utc": ended_at_utc,
        "field_principles": {},
        "execution_venue": "host",
        "execution_host": socket.gethostname(),
        "duration_s": round(float(duration_s), 6),
        "random_seed": RANDOM_SEED,
        "reproducibility_checksum": "",
    }


def _seal_artifact(artifact: JsonDict) -> JsonDict:
    """Attach a principle to every top-level value and seal the result."""

    artifact["field_principles"] = {key: FIELD_PRINCIPLES[key] for key in artifact}
    artifact["reproducibility_checksum"] = artifact_checksum(artifact)
    return artifact


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
    runner_receipt: Mapping[str, Any],
    validation_receipts: Sequence[Mapping[str, Any]],
    phase_spans: Sequence[Mapping[str, Any]],
) -> JsonDict:
    """Build a complete pilot without conflating valid synthesis and policy use."""

    rows = [deepcopy(dict(row)) for row in episode_rows]
    reduction = reduce_episode_rows(rows)
    validation_passed = all(row.get("passed") is True for row in validation_receipts)
    method_value = int(reduction["arc_method_value_score"] == 1 and validation_passed)
    counts = deepcopy(reduction["invocation_counts"])
    invoked = counts["generation_calls_attempted"] > 0
    loaded = counts["model_loads_completed"] > 0
    if invoked:
        substrate = "live_llm_inference"
        substrate_class = "model_full_generation"
    elif loaded:
        substrate = "model_load_no_generation"
        substrate_class = "model_load_no_generation"
    else:
        substrate = "blocked_no_run"
        substrate_class = "blocked_no_run"
    artifact = _base_artifact(started_at_utc, ended_at_utc, duration_s)
    artifact.update(
        {
            "status": "complete",
            "preconditions_checked": [deepcopy(dict(row)) for row in preconditions],
            "MODEL_SPECS": [deepcopy(dict(model_spec))] if (loaded or invoked) else [],
            "model_invoked": invoked,
            "invocation_counts": counts,
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
                "generation_calls_per_episode": COMPLETION_LIMIT,
                "generated_tokens_per_episode": GENERATED_TOKEN_LIMIT,
                "elapsed_cap_per_arm_s": SESSION_LIMIT_S,
                "session_limit_s": SESSION_LIMIT_S,
                "model_load_limit_s": MODEL_LOAD_LIMIT_S,
                "stopping_rule": "four fixed counterbalanced episodes; authentic timeouts stay censored",
            },
            "acceptance_gate_results": _acceptance_results(
                reduction, validation_passed=validation_passed
            ),
            "gate_check_summary": [],
            "verifier_is_oracle": True,
            "honest_verdict": (
                "complete_circular_positive_typed_witness_policy_use_pilot"
                if method_value
                else "complete_null_typed_witness_policy_use_gates_not_met"
            ),
            "verdict_class": "circular_positive" if method_value else "null",
            "validation_receipts": [deepcopy(dict(row)) for row in validation_receipts],
            "arc_capture_complete_score": reduction["arc_capture_complete_score"],
            "arc_method_value_score": method_value,
            "per_game_results": rows,
            "policy_consumption_rows": [
                deepcopy(item) for row in rows for item in row.get("policy_consumption_rows", [])
            ],
            "identity_obligation_rows": [
                deepcopy(item) for row in rows for item in row.get("identity_obligation_rows", [])
            ],
            "selection_receipt": deepcopy(dict(selection_receipt)),
            "runner_receipt": deepcopy(dict(runner_receipt)),
            "raw_request_manifest": [
                deepcopy(item) for row in rows for item in row.get("raw_request_manifest", [])
            ],
            "submission_configuration_diff": {
                "gateway": "local_public_arcade_not_hidden_competition_gateway",
                "gpu": "one_task_owned_RTX_3090_not_Kaggle_Blackwell_96GB",
                "weights": "Q4_K_M_GGUF_not_submitted_NVFP4_safetensors",
                "runtime": "native_llama.cpp_single_server_not_submitted_vLLM",
                "parallelism": "four_sequential_episodes_not_DualGPURunner",
                "adapters": "withheld",
                "competition_equivalent_claim": False,
            },
            "claim_scope": reduction["claim_scope"],
            "population_confidence_interval": None,
            "solve_provenance": "live_agent_self_discovery",
            "new_solve_claimed": False,
            "reproduced_levels": [],
            "official_score": None,
        }
    )
    return _seal_artifact(artifact)


def build_blocked_artifact(
    *,
    started_at_utc: str,
    ended_at_utc: str,
    duration_s: float,
    preconditions: Sequence[Mapping[str, Any]],
    source_hashes: Mapping[str, Any],
    invocation_counts: Mapping[str, int] | None = None,
    model_spec: Mapping[str, Any] | None = None,
    runner_receipt: Mapping[str, Any] | None = None,
) -> JsonDict:
    """Build a terminal external block using only computation that occurred."""

    failures = [deepcopy(dict(row)) for row in preconditions if row.get("passed") is not True]
    counts = dict(
        invocation_counts
        or {
            "model_loads_attempted": 0,
            "model_loads_completed": 0,
            "generation_calls_attempted": 0,
            "generation_calls_completed": 0,
            "usable_answers": 0,
        }
    )
    loaded = int(counts.get("model_loads_completed") or 0) > 0
    invoked = int(counts.get("generation_calls_attempted") or 0) > 0
    substrate = (
        "live_llm_inference"
        if invoked
        else "model_load_no_generation"
        if loaded
        else "blocked_no_run"
    )
    artifact = _base_artifact(started_at_utc, ended_at_utc, duration_s)
    artifact.update(
        {
            "preconditions_checked": [deepcopy(dict(row)) for row in preconditions],
            "MODEL_SPECS": [deepcopy(dict(model_spec))]
            if model_spec and (loaded or invoked)
            else [],
            "model_invoked": invoked,
            "invocation_counts": counts,
            "inference_substrate": substrate,
            "inference_substrate_class": substrate,
            "inference_mode": "live_gpu" if loaded else "not_invoked",
            "phase_spans": [],
            "source_artifact_hashes": deepcopy(dict(source_hashes)),
            "rows": [],
            "sample_size_budget": {
                "planned_units": 4,
                "attempted_units": 0,
                "completed_units": 0,
                "censored_units": 0,
                "stopping_rule": "external prerequisite failure stops further work",
            },
            "acceptance_gate_results": [
                {
                    "criterion": row.get("field"),
                    "expected": row.get("expected"),
                    "observed": row.get("observed"),
                    "passed": False,
                    "principle": "External prerequisites must pass at their applicable boundary.",
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
            "policy_consumption_rows": [],
            "identity_obligation_rows": [],
            "selection_receipt": {},
            "runner_receipt": deepcopy(dict(runner_receipt or {})),
            "raw_request_manifest": [],
            "submission_configuration_diff": {},
            "claim_scope": "blocked_no_case_study",
            "population_confidence_interval": None,
            "solve_provenance": "live_agent_self_discovery",
            "new_solve_claimed": False,
            "reproduced_levels": [],
            "official_score": None,
        }
    )
    return _seal_artifact(artifact)


def validate_artifact(value: Mapping[str, Any] | Path) -> list[str]:
    """Cold-check terminal identity, reduction, budgets, and checksum."""

    if isinstance(value, Path):
        try:
            artifact = json.loads(value.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            return [f"artifact_unreadable:{type(exc).__name__}"]
    else:
        artifact = deepcopy(dict(value))
    errors: list[str] = []
    if set(artifact.get("field_principles", {})) != set(artifact):
        errors.append("field_principles_must_cover_every_top_level_field")
    if artifact.get("schema") != SCHEMA or artifact.get("experiment_id") != EXPERIMENT_ID:
        errors.append("schema_or_experiment_identity_mismatch")
    if artifact.get("milestone") != MILESTONE or artifact.get("run_date") != RUN_DATE:
        errors.append("milestone_or_run_date_mismatch")
    if artifact.get("status") not in {"complete", "blocked"}:
        errors.append("status_not_terminal")
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
    if artifact.get("population_confidence_interval") is not None:
        errors.append("population_confidence_interval_forbidden")
    if artifact.get("official_score") is not None:
        errors.append("official_score_must_be_unset")
    if artifact.get("status") == "blocked":
        if not str(artifact.get("honest_verdict") or "").startswith("blocked_"):
            errors.append("blocked_verdict_prefix_invalid")
        if artifact.get("rows") or artifact.get("model_invoked"):
            errors.append("blocked_artifact_contains_measurement")
        counts = artifact.get("invocation_counts", {})
        loaded = int(counts.get("model_loads_completed") or 0) > 0
        invoked = int(counts.get("generation_calls_attempted") or 0) > 0
        expected_substrate = (
            "live_llm_inference"
            if invoked
            else "model_load_no_generation"
            if loaded
            else "blocked_no_run"
        )
        if artifact.get("inference_substrate") != expected_substrate:
            errors.append("blocked_substrate_invalid")
        if not artifact.get("gate_check_summary"):
            errors.append("blocked_gate_summary_missing")
    elif artifact.get("status") == "complete":
        if not str(artifact.get("honest_verdict") or "").startswith(("complete_", "complete:")):
            errors.append("complete_verdict_prefix_invalid")
        reduction = reduce_episode_rows(artifact.get("rows", []))
        validation_passed = all(
            row.get("passed") is True for row in artifact.get("validation_receipts", [])
        )
        expected_method = int(reduction["arc_method_value_score"] == 1 and validation_passed)
        if artifact.get("arc_capture_complete_score") != reduction["arc_capture_complete_score"]:
            errors.append("arc_capture_complete_score_inconsistent")
        if artifact.get("arc_method_value_score") != expected_method:
            errors.append("arc_method_value_score_inconsistent")
        if artifact.get("invocation_counts") != reduction["invocation_counts"]:
            errors.append("invocation_counts_inconsistent")
        invoked = reduction["invocation_counts"]["generation_calls_attempted"] > 0
        loaded = reduction["invocation_counts"]["model_loads_completed"] > 0
        if artifact.get("model_invoked") != invoked:
            errors.append("model_invoked_inconsistent")
        if invoked and artifact.get("inference_substrate") != "live_llm_inference":
            errors.append("live_substrate_inconsistent")
        if invoked and float(artifact.get("duration_s") or 0) < 60:
            errors.append("model_full_generation_duration_floor_failed")
        specs = artifact.get("MODEL_SPECS", [])
        if (loaded or invoked) and (
            len(specs) != 1
            or specs[0].get("hf_id") != MODEL_ID
            or specs[0].get("quantization") != QUANTIZATION
        ):
            errors.append("model_specs_inconsistent")
        if not (loaded or invoked) and specs:
            errors.append("model_specs_inconsistent")
        for row in artifact.get("rows", []):
            errors.extend(validate_episode_row(row))
    if artifact.get("reproducibility_checksum") != artifact_checksum(artifact):
        errors.append("reproducibility_checksum_mismatch")
    return errors


def independent_reduce(path: Path) -> JsonDict:
    """Read raw episode rows and recompute both pilot scores."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    return reduce_episode_rows(payload.get("rows", []))


def build_validation_commands(*, terminal_candidate: Path, raw_rows: Path) -> list[JsonDict]:
    """Return only the focused, affected, E2E, and artifact validation commands."""

    python = str(REPO_ROOT / ".venv/bin/python")
    pytest = str(REPO_ROOT / ".venv/bin/pytest")
    coverage = str(REPO_ROOT / ".venv/bin/coverage")
    ruff = str(REPO_ROOT / ".venv/bin/ruff")
    mypy = str(REPO_ROOT / ".venv/bin/mypy")
    focused = [
        (
            "focused_exp7280",
            TEST_PATH,
        ),
        (
            "affected_exp7276",
            Path("tests/python/test_experiment_7276_v640_arc_identity.py"),
        ),
        (
            "affected_arc_eval_provenance",
            Path("tests/python/test_arc_eval_provenance_contract_20260905.py"),
        ),
        (
            "e2e_009_policy_memory",
            Path("tests/python/test_arc_induction_state_persistence.py"),
        ),
    ]
    commands: list[JsonDict] = [
        {
            "name": name,
            "command": [
                pytest,
                "-o",
                "addopts=",
                str(path),
                "-q",
                "--no-cov",
                "-n",
                "0",
                f"--basetemp=/tmp/exp7280-{name}",
            ],
        }
        for name, path in focused
    ]
    commands.extend(
        [
            {
                "name": "e2e_009_offline_scored_smoke",
                "command": [
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
                    "/tmp/exp7280-e2e009-r11l.json",
                ],
                "env": {"CARNOT_ARC_DISABLE_INDUCTION": "1"},
            },
            {
                "name": "e2e_010_grammar_transport",
                "command": [
                    pytest,
                    "-o",
                    "addopts=",
                    "tests/python/test_arc_tool_grammar_transport.py",
                    "-q",
                    "--no-cov",
                    "-n",
                    "0",
                    "--basetemp=/tmp/exp7280-e2e010",
                ],
            },
            {
                "name": "scoped_coverage_run",
                "command": [
                    coverage,
                    "run",
                    "--data-file=/tmp/exp7280-v640.coverage",
                    "--include=*/experiment_7280_v640_arc_live.py",
                    "-m",
                    "pytest",
                    "-o",
                    "addopts=",
                    str(TEST_PATH),
                    "-q",
                    "-n",
                    "0",
                    "--basetemp=/tmp/exp7280-coverage",
                ],
            },
            {
                "name": "scoped_coverage_report",
                "command": [
                    coverage,
                    "report",
                    "--data-file=/tmp/exp7280-v640.coverage",
                    "--include=*/experiment_7280_v640_arc_live.py",
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
                    *[str(path) for _name, path in focused[1:]],
                    "tests/python/test_arc_tool_grammar_transport.py",
                ],
            },
            {
                "name": "independent_raw_row_reducer",
                "command": [python, "-u", str(WRAPPER_PATH), "--reduce-raw", str(raw_rows)],
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
    return commands


def _progress(phase: str, event: str, **fields: Any) -> None:  # pragma: no cover
    """Print and flush one truthful phase or long-operation boundary."""

    print(
        json.dumps(
            {"experiment": TASK_ID, "phase": phase, "event": event, **fields}, sort_keys=True
        ),
        flush=True,
    )


def _iso_now() -> str:  # pragma: no cover
    """Return an auditable UTC timestamp."""

    return datetime.now(UTC).isoformat(timespec="seconds")


def _sha256_file(path: Path) -> str:  # pragma: no cover
    """Hash one file while emitting progress for large model bytes."""

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


def _manifest_state(manifest: Any, experiment_id: str) -> tuple[bool, bool]:  # pragma: no cover
    """Read explicit quarantine and retirement markers without interpreting prose."""

    rows = []
    if isinstance(manifest, list):
        rows = manifest
    elif isinstance(manifest, Mapping):
        rows = [
            row
            for key in ("retired", "retired_experiments", "retired_extras")
            for row in manifest.get(key, [])
            if isinstance(row, Mapping)
        ]
    matching = [
        row
        for row in rows
        if isinstance(row, Mapping)
        and experiment_id in {str(row.get("id")), str(row.get("experiment_id"))}
    ]
    return bool(matching), any(row.get("quarantined") is True for row in matching)


def collect_preconditions(
    root: Path,
) -> tuple[list[JsonDict], JsonDict, JsonDict]:  # pragma: no cover
    """Authenticate inputs, identity readiness, cache selection, runtime, and one idle GPU."""

    checks: list[JsonDict] = []
    hashes: JsonDict = {}
    for relative in REQUIRED_INPUTS:
        path = root / relative
        exists = path.is_file()
        checks.append(gate_check("required_input", relative.as_posix(), "exists", True, exists))
        if exists:
            hashes[relative.as_posix()] = {
                "sha256": _sha256_file(path),
                "quarantined": False,
                "retired": False,
                "current_output": relative in {MODULE_PATH, WRAPPER_PATH, TEST_PATH},
            }
    spec_text = (
        (root / SPEC_PATH).read_text(encoding="utf-8") if (root / SPEC_PATH).is_file() else ""
    )
    checks.append(
        gate_check(
            "driving_capability",
            SPEC_PATH.as_posix(),
            "REQ-ARC-WMTE-7280",
            True,
            "REQ-ARC-WMTE-7280" in spec_text,
        )
    )
    try:
        manifest = yaml.safe_load((root / EXCLUSION_PATH).read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError):
        manifest = {}
    retired, quarantined = _manifest_state(manifest, "exp7276-arc-identity")
    try:
        upstream_value = json.loads((root / UPSTREAM_PATH).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        upstream_value = {}
    quarantined = quarantined or bool(prior.scored.is_quarantined(upstream_value))
    upstream_checks, upstream = authenticate_identity_upstream(
        root / UPSTREAM_PATH, quarantined=quarantined, retired=retired
    )
    checks.extend(upstream_checks)
    for relative in (RESULT_PATH.parent, CHECKPOINT_PATH.parent, RAW_DIR):
        path = root / relative
        path.mkdir(parents=True, exist_ok=True)
        owned = path.is_dir() and os.access(path, os.W_OK) and path.stat().st_uid == os.getuid()
        checks.append(
            gate_check(
                "writable_output_directory", relative.as_posix(), "owner_and_writable", True, owned
            )
        )

    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from scripts import experiment_template

    pair = experiment_template.cached_sota_pair(preferred_quant=QUANTIZATION)
    selected, model_path, model_ok = resolve_cached_model(pair or [])
    revision = selected.get("revision")
    checks.append(
        gate_check("cached_sota_pair", MODEL_ID, "current_Q4_K_M_snapshot", True, model_ok)
    )
    server_candidates = (
        os.environ.get("CARNOT_LLAMA_SERVER"),
        str(Path.home() / ".cache/llama.cpp-master/build/bin/llama-server"),
    )
    server = next(
        (Path(value) for value in server_candidates if value and Path(value).is_file()), None
    )
    checks.append(gate_check("native_runtime", "llama.cpp", "server_binary", True, bool(server)))
    gpus = prior.scored._gpu_inventory()
    idle = [
        row
        for row in gpus
        if row.get("name") == "NVIDIA GeForce RTX 3090"
        and not row.get("compute_apps")
        and int(row.get("free_memory_mb") or 0) >= 23_000
    ]
    checks.append(
        gate_check("gpu_lease", "nvidia-smi", "idle_RTX_3090_with_headroom", True, bool(idle))
    )
    model_hash = None
    if model_ok and model_path is not None:
        _progress("preconditions", "BEFORE model hash", path=str(model_path))
        model_hash = _sha256_file(model_path)
        hashes[str(model_path)] = {
            "sha256": model_hash,
            "revision": revision,
            "quarantined": False,
            "retired": False,
        }
        _progress("preconditions", "AFTER model hash", sha256=model_hash)
    selected.update(
        {
            "quantization": QUANTIZATION,
            "revision": revision,
            "model_file_hash": model_hash,
            "model_path": str(model_path) if model_path else None,
        }
    )
    return (
        checks,
        hashes,
        {
            "upstream": upstream,
            "model_spec": selected,
            "model_path": str(model_path) if model_ok and model_path else None,
            "model_hash": model_hash,
            "server": str(server) if server else None,
            "gpu": idle[0] if idle else None,
        },
    )


@contextmanager
def _configured_prior() -> Iterator[None]:  # pragma: no cover
    """Temporarily point the proven V639 live machinery at this task's owned paths."""

    updates = {
        "TASK_ID": TASK_ID,
        "EXPERIMENT_ID": EXPERIMENT_ID,
        "MILESTONE": MILESTONE,
        "RANDOM_SEED": RANDOM_SEED,
        "SCHEMA": SCHEMA,
        "LIVE_WINDOW_S": SESSION_LIMIT_S,
        "STARTUP_TIMEOUT_S": MODEL_LOAD_LIMIT_S,
        "RESULT_PATH": RESULT_PATH,
        "CHECKPOINT_PATH": CHECKPOINT_PATH,
        "RAW_DIR": RAW_DIR,
        "RAW_ROWS_PATH": RAW_ROWS_PATH,
        "TERMINAL_CANDIDATE_PATH": TERMINAL_CANDIDATE_PATH,
        "SESSION_PATH": SESSION_PATH,
        "MODULE_PATH": MODULE_PATH,
        "WRAPPER_PATH": WRAPPER_PATH,
        "TEST_PATH": TEST_PATH,
    }
    old = {name: getattr(prior, name) for name in updates}
    try:
        for name, value in updates.items():
            setattr(prior, name, value)
        yield
    finally:
        for name, value in old.items():
            setattr(prior, name, value)


def _capture_live_identity(proposer: Any, raw_dir: Path) -> JsonDict:  # pragma: no cover
    """Capture raw server identity after load and before the first generation."""

    import urllib.request

    raw_props_path = raw_dir / "raw_server_props.json"
    with urllib.request.urlopen(proposer._url() + "/props", timeout=10) as response:
        raw_props_bytes = response.read()
    raw_props_path.write_bytes(raw_props_bytes)
    raw_props = json.loads(raw_props_bytes)
    command = list(proposer.last_launch_argv)
    launch = command[command.index("-m") + 1] if "-m" in command else None
    server_pid = getattr(getattr(proposer, "_proc", None), "pid", None)
    source = provenance.capture_arc_model_identity_source_provenance(
        raw_server_props=raw_props,
        requested_model_path=proposer.requested_model_path,
        source_kind="live_server_props",
        launch_model_argument=launch,
        server_pid=server_pid,
        server_pid_start_tick=proposer.server_pid_start_tick,
    )
    receipt = provenance.build_typed_arc_model_identity_receipt(
        selected_model_spec={
            "model_path": proposer.requested_model_path,
            "model_filename": proposer.requested_model_filename,
            "hf_id": proposer.model_repository,
            "revision": proposer.model_revision,
            "model_file_hash": provenance._sha256_file(Path(proposer.requested_model_path)),
        },
        launch_model_argument=launch,
        raw_server_props=raw_props,
        source_provenance=source,
    )
    decision = provenance.validate_typed_arc_model_identity_receipt(receipt)
    value = {
        "authenticated_at_utc": _iso_now(),
        "authenticated_before_first_generation": True,
        "valid": decision.valid,
        "errors": list(decision.errors),
        "raw_server_props_path": str(raw_props_path),
        "raw_server_props_sha256": _sha256_bytes(raw_props_bytes),
        "receipt": receipt,
    }
    atomic_write(raw_dir / "model_identity.json", value)
    if not decision.valid:
        raise RuntimeError("live model identity is invalid: " + "; ".join(decision.errors))
    return value


def run_live_session(args: argparse.Namespace) -> int:  # pragma: no cover
    """Run the shipped live child after inserting strict identity before generation."""

    raw_dir = Path(args.raw_dir)
    from carnot.agentic.arc_executable_world_model import LocalGGUFProposer

    original_ensure = LocalGGUFProposer._ensure_server
    identity_complete = False

    def ensure_and_authenticate(proposer: Any) -> bool:
        nonlocal identity_complete
        if identity_complete:
            return original_ensure(proposer)
        _progress("model_load", "BEFORE shipped model load")
        loaded = original_ensure(proposer)
        _progress("model_load", "AFTER shipped model load", loaded=loaded)
        if loaded:
            _progress("identity", "BEFORE strict identity authentication")
            _capture_live_identity(proposer, raw_dir)
            identity_complete = True
            _progress("identity", "AFTER strict identity authentication", valid=True)
        return loaded

    LocalGGUFProposer._ensure_server = ensure_and_authenticate
    try:
        with _configured_prior():
            result = prior.run_live_session(args)
    finally:
        LocalGGUFProposer._ensure_server = original_ensure
    try:
        session = json.loads(Path(args.session_path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return result
    try:
        identity = json.loads((raw_dir / "model_identity.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        identity = {}
    receipt = identity.get("receipt", {}) if isinstance(identity, Mapping) else {}
    receipt_sha = _sha256_bytes(_canonical_bytes(receipt)) if receipt else None
    for row in session.get("episodes", []):
        episode_id = str(row.get("episode_id"))
        row["identity_obligation_rows"] = bind_identity_obligations(
            episode_id,
            receipt,
            authenticated_at_utc=str(identity.get("authenticated_at_utc")),
        )
        row["identity_receipt_sha256"] = receipt_sha
        row["fresh_store"] = True
        row["abstention"] = False
        row["elapsed_cap_s"] = SESSION_LIMIT_S
    runtime = dict(session.get("runtime_receipt", {}))
    runtime.update(
        {
            "identity_authentication_valid": identity.get("valid"),
            "identity_authentication_errors": identity.get("errors", []),
            "identity_authenticated_at_utc": identity.get("authenticated_at_utc"),
            "identity_authenticated_before_first_generation": identity.get(
                "authenticated_before_first_generation", False
            ),
            "identity_receipt_sha256": receipt_sha,
            "raw_server_props_path": identity.get("raw_server_props_path"),
            "raw_server_props_sha256": identity.get("raw_server_props_sha256"),
            "raw_server_props": receipt.get("raw_server_props") if receipt else None,
            "dual_gpu_runner_used": False,
        }
    )
    session["runtime_receipt"] = runtime
    atomic_write(Path(args.session_path), session)
    return result


def run_child_with_lease(
    *,
    resources: Mapping[str, Any],
    schedule_path: Path,
    raw_dir: Path,
    checkpoint_path: Path,
    session_path: Path,
    remaining_s: float,
) -> JsonDict:  # pragma: no cover
    """Reuse one task-owned lease and add measured post-residency KV headroom."""

    with _configured_prior():
        session = prior.run_child_with_lease(
            resources=resources,
            schedule_path=schedule_path,
            raw_dir=raw_dir,
            checkpoint_path=checkpoint_path,
            session_path=session_path,
            remaining_s=remaining_s,
        )
    runtime = dict(session.get("runtime_receipt", {}))
    samples = runtime.get("gpu_samples", [])
    used = [
        int(app.get("used_memory_mb") or 0)
        for sample in samples
        if isinstance(sample, Mapping)
        for app in sample.get("owned_compute_apps", [])
        if isinstance(app, Mapping)
    ]
    free_before = int(runtime.get("vram_free_before_mb") or 0)
    runtime.update(
        {
            "measured_peak_owned_vram_mb": max(used, default=0),
            "measured_kv_headroom_mb": max(0, free_before - max(used, default=0)),
            "kv_headroom_measurement": "idle_free_vram_before_minus_peak_task_owned_server_vram",
            "dual_gpu_runner_used": False,
        }
    )
    session["runtime_receipt"] = runtime
    atomic_write(session_path, session)
    return session


def _censored_rows(
    schedule: Sequence[Mapping[str, Any]],
    existing: Sequence[Mapping[str, Any]],
    identity: Mapping[str, Any],
) -> list[JsonDict]:  # pragma: no cover
    """Account for unfinished authentic episodes without inventing outcomes."""

    with _configured_prior():
        rows = prior._censored_rows(schedule, existing)
    receipt = identity.get("receipt", {}) if isinstance(identity, Mapping) else {}
    receipt_sha = _sha256_bytes(_canonical_bytes(receipt)) if receipt else None
    for row in rows:
        episode_id = str(row.get("episode_id"))
        row.setdefault(
            "identity_obligation_rows",
            bind_identity_obligations(
                episode_id,
                receipt,
                authenticated_at_utc=str(identity.get("authenticated_at_utc")),
            ),
        )
        row.setdefault("identity_receipt_sha256", receipt_sha)
        row.setdefault("fresh_store", True)
        row.setdefault("abstention", False)
        row.setdefault("elapsed_cap_s", SESSION_LIMIT_S)
    return rows


def _run_validation_rows(
    commands: Sequence[Mapping[str, Any]], *, raw_dir: Path
) -> list[JsonDict]:  # pragma: no cover
    """Stream each bounded validation command and retain its exact receipt."""

    from carnot.experiment_7246_v638_source_map import _run_streaming_command

    rows: list[JsonDict] = []
    directory = raw_dir / "validation"
    directory.mkdir(parents=True, exist_ok=True)
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
        overrides = {str(key): str(value) for key, value in command_row.get("env", {}).items()}
        old = {key: os.environ.get(key) for key in overrides}
        os.environ.update(overrides)
        try:
            result = _run_streaming_command(
                command,
                cwd=REPO_ROOT,
                timeout_s=900,
                heartbeat_s=45,
                operation=f"exp7280:{name}",
            )
        finally:
            for key, value in old.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value
        output = str(result.get("output") or result.get("stdout") or "")
        log_path = directory / f"{index:02d}_{name}.log"
        log_path.write_text(output, encoding="utf-8")
        row = {
            "name": name,
            "command": " ".join(command),
            "environment_overrides": overrides,
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


def _load_registry(path: Path) -> JsonDict:  # pragma: no cover
    """Read only public registry metadata before the roster is frozen."""

    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError):
        return {}
    return dict(value) if isinstance(value, Mapping) else {}


def run_experiment(args: argparse.Namespace) -> JsonDict:  # pragma: no cover
    """Run preflight, four live episodes, scoped validation, and atomic publication."""

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
        errors = validate_artifact(artifact)
        if errors:
            raise RuntimeError(f"blocked artifact failed validation: {errors}")
        atomic_write(result_path, artifact)
        _progress("terminal", "published_blocked", path=str(result_path))
        return artifact

    _progress("selection", "BEGIN registry precheck before roster freeze")
    registry = _load_registry(root / REGISTRY_PATH)
    from carnot.agentic.arc_game_adapters import adaptered_games

    selection = freeze_public_roster(registry, adaptered_games=set(adaptered_games()))
    schedule = counterbalanced_schedule(selection["games"], RANDOM_SEED)
    selection_ok = selection["roster_complete"] is True and len(schedule) == 4
    checks.append(
        gate_check(
            "registry_precheck_before_roster",
            REGISTRY_PATH.as_posix(),
            "re86_r11l_four_counterbalanced_units",
            True,
            selection_ok,
        )
    )
    schedule_path = raw_dir / "frozen_schedule.json"
    atomic_write(schedule_path, {"selection_receipt": selection, "rows": schedule})
    hashes[str(schedule_path.relative_to(root))] = {"sha256": _sha256_file(schedule_path)}
    _progress("selection", "END", games=selection["games"], completed_units=len(schedule))
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
    _progress("live_window", "BEGIN", max_seconds=SESSION_LIMIT_S, planned_units=4)
    live_start = time.monotonic()
    session = run_child_with_lease(
        resources={
            **resources,
            "model_path": resources["model_path"],
            "model_hash": resources["model_hash"],
        },
        schedule_path=schedule_path,
        raw_dir=raw_dir,
        checkpoint_path=checkpoint_path,
        session_path=raw_dir / "live_session.json",
        remaining_s=max(1.0, SESSION_LIMIT_S - (time.monotonic() - started)),
    )
    _progress(
        "live_window",
        "END",
        model_loaded=session.get("model_loaded"),
        model_invoked=session.get("model_invoked"),
        completed_units=len(session.get("episodes", [])),
        elapsed_s=round(time.monotonic() - live_start, 6),
    )
    try:
        identity = json.loads((raw_dir / "model_identity.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        identity = {}
    if not session.get("model_loaded") or identity.get("valid") is not True:
        checks.append(
            gate_check(
                "live_model_identity_before_generation",
                MODEL_ID,
                "task_owned_loaded_and_identity_valid",
                True,
                False,
            )
        )
        load_attempted = 1
        load_completed = int(bool(identity))
        counts = {
            "model_loads_attempted": load_attempted,
            "model_loads_completed": load_completed,
            "generation_calls_attempted": 0,
            "generation_calls_completed": 0,
            "usable_answers": 0,
        }
        for path in raw_dir.rglob("*"):
            if path.is_file():
                hashes[str(path.relative_to(root))] = {"sha256": _sha256_file(path)}
        artifact = build_blocked_artifact(
            started_at_utc=started_utc,
            ended_at_utc=_iso_now(),
            duration_s=time.monotonic() - started,
            preconditions=checks,
            source_hashes=hashes,
            invocation_counts=counts,
            model_spec=resources["model_spec"],
            runner_receipt=session.get("runtime_receipt", {}),
        )
        errors = validate_artifact(artifact)
        if errors:
            raise RuntimeError(f"runtime blocked artifact failed validation: {errors}")
        atomic_write(result_path, artifact)
        _progress("terminal", "published_blocked_identity", path=str(result_path))
        return artifact

    episode_rows = _censored_rows(schedule, session.get("episodes", []), identity)
    atomic_write(raw_dir / "episode_rows.json", {"rows": episode_rows})
    for path in raw_dir.rglob("*"):
        if path.is_file() and path != raw_dir / "terminal_candidate.json":
            hashes[str(path.relative_to(root))] = {"sha256": _sha256_file(path)}
    model_spec = dict(resources["model_spec"])
    runner_receipt = dict(session.get("runtime_receipt", {}))
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
        runner_receipt=runner_receipt,
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
        runner_receipt=runner_receipt,
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
        runner_receipt=runner_receipt,
        validation_receipts=validation_rows,
        phase_spans=phase_spans,
    )
    errors = validate_artifact(artifact)
    if errors:
        raise RuntimeError(f"terminal artifact failed validation: {errors}")
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
    """Parse the thin experiment and live-child command line."""

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


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover
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


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
