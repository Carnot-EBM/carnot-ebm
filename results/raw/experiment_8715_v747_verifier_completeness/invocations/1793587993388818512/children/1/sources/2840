"""Measure whether direct selfparse evidence reaches a later policy action.

The live child reuses the shipped ARC policy, llama.cpp runner, GPU lease, and
request capture. This module adds fail-closed dependency checks, durable tool
event capture, one frozen transfer session, and an independent terminal reducer.

Spec refs: REQ-ARC-WMTE-7305 and SCENARIO-ARC-WMTE-7305-*.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
from contextlib import contextmanager
from copy import deepcopy
from datetime import UTC, datetime
import hashlib
import json
import os
from pathlib import Path
import socket
import sys
import tempfile
import time
from typing import Any

import yaml


JsonDict = dict[str, Any]
REPO_ROOT = Path(__file__).resolve().parents[2]
RUN_DATE = "20260914"
MILESTONE = "2026.09.642"
EXPERIMENT_ID = "exp7305-arc-selfparse"
SCHEMA = "carnot.experiment_7305.v642.arc_selfparse.v1"

MODEL_ID = "unsloth/Qwen3.8-27B-GGUF"
QUANTIZATION = "Q4_K_M"
MODEL_SPECS = [{"hf_id": MODEL_ID, "quantization": QUANTIZATION}]
TARGET_GAME = "r11l"
DEVELOPMENT_SEED = 7_305_202_609_14
EVALUATION_SEED = 17_305_202_609_14
ACTION_LIMIT = 192
COMPLETION_LIMIT = 2
GENERATED_TOKEN_LIMIT = 4096
TOKENS_PER_CALL = GENERATED_TOKEN_LIMIT // COMPLETION_LIMIT
SESSION_LIMIT_S = 3000
MODEL_LOAD_LIMIT_S = 600
EVIDENCE_GOAL = 10

SPEC_PATH = Path("openspec/capabilities/arc-world-model-trust-energy/spec.md")
EXP7304_PATH = Path("results/experiment_7304_v642_arc_receipt.json")
EXCLUSION_PATH = Path("ops/exclusion_manifest.yaml")
REGISTRY_PATH = Path("ops/arc_solve_registry.yaml")
RESULT_PATH = Path("results/experiment_7305_v642_arc_selfparse.json")
RAW_DIR = Path("results/raw/experiment_7305_v642_arc_selfparse")
CHECKPOINT_PATH = Path("results/checkpoints/experiment_7305_v642_arc_selfparse.json")
BOUNDARY_PATH = RAW_DIR / "receipt_events.jsonl"
TOOL_EVENT_PATH = RAW_DIR / "tool_events.jsonl"
SESSION_PATH = RAW_DIR / "live_session.json"
RAW_ROW_PATH = RAW_DIR / "independent_reduction_input.json"
TERMINAL_CANDIDATE_PATH = RAW_DIR / "measured_terminal_candidate.json"
MODULE_PATH = Path("python/carnot/experiment_7305_v642_arc_selfparse.py")
WRAPPER_PATH = Path("scripts/experiments/experiment_7305_v642_arc_selfparse.py")
TEST_PATH = Path("tests/python/test_experiment_7305_v642_arc_selfparse.py")

HISTORICAL_PATHS = (
    Path("results/experiment_7206_v635_arc_volume_a.json"),
    Path("results/experiment_7207_v635_arc_volume_b.json"),
    Path("results/experiment_7221_v636_arc_session.json"),
)
HISTORICAL_REPOSITORY_HEALTH_LOG = Path(
    "results/raw/experiment_7289_v641_arc_boundary/validation/04_full_python_suite.log"
)
FAILED_DATE_OVERRIDE_ATTEMPT_DIR = Path(
    "results/raw/experiment_7305_v642_arc_selfparse_failed_date_override_20260914_1124"
)

ZERO_INVOCATION_COUNTS = {
    "model_loads_attempted": 0,
    "model_loads_completed": 0,
    "model_loads_failed": 0,
    "model_loads_cancelled": 0,
    "model_loads_in_flight": 0,
    "generation_calls_attempted": 0,
    "generation_calls_completed": 0,
    "generation_calls_failed": 0,
    "generation_calls_cancelled": 0,
    "generation_calls_in_flight": 0,
    "usable_answers": 0,
}

FIELD_PRINCIPLES = {
    "schema": "Version the record while keeping ordinary top-level experiment_id and milestone.",
    "status": "Write the terminal result only after the work and checks; checkpoints remain separate.",
    "run_date": "Use 20260914 with actual UTC start/end and monotonic phase timing.",
    "preconditions_checked": "Hash real inputs and record actual availability and failed checks.",
    "MODEL_SPECS": "Actual current executable model identities; historical identities remain in sidecars.",
    "model_invoked": "True for any attempted model load or generation, even when output is unusable.",
    "invocation_counts": "Separate attempted, completed, failed, cancelled, and in-flight loads and generations.",
    "inference_substrate": "Describe actual computation with a recognized literal, not intended work.",
    "inference_substrate_class": "Full generation has a 60s floor; bounded generation 10s; load-only 2s. Never pad time.",
    "execution_venue": "Record actual host/device work; a CPU replay is not GPU or FPGA execution.",
    "duration_s": "Measure total elapsed and disjoint phase spans including failed work.",
    "random_seed": "Seal development and independent evaluation seeds before seeing outcomes.",
    "reproducibility_checksum": "Bind code, inputs, config, model when used, and raw evidence.",
    "source_artifact_hashes": "Authenticate producer identity, terminal class, and quarantine state.",
    "rows": "Record every comparative unit with metrics, costs, errors, abstentions, and censoring.",
    "sample_size_budget": "Keep planned, attempted, complete, and censored counts with the frozen stopping rule.",
    "acceptance_gate_results": "Every check has expected, observed, passed, and a principle explaining its purpose.",
    "gate_check_summary": "Every blocked result names upstream, check, field, observed value, and expected value.",
    "verifier_is_oracle": "Shared evaluator authority permits circular_positive only, not positive scientific value.",
    "honest_verdict": "Completed findings start complete_; external failure starts blocked_. State the finding.",
    "verdict_class": "Use the closed verdict enum; partial is only for unfinished work owned by this task.",
    "validation_receipts": "Record exact commands, scope, exit codes, elapsed time, and log hashes; retain failures.",
    "arc_capture_complete_score": "One requires authenticated current evidence and complete censoring, independent of tool success.",
    "arc_tool_use_score": "One requires a new tool-result-to-policy-action chain; it is not a solve claim.",
    "solve_provenance": "Use live_agent_self_discovery only for the live path and exclude transfer solves from headline credit.",
    "per_game_results": "Keep target, seed, actions, policy use, censoring, and runtime reachability per session.",
    "cumulative_induction_ledger": "Keep hash-bound historical inductions separate from current counters.",
    "offline_reproduced": "True only after an actual replay for an incidental novel solve.",
    "reproduced_levels": "Count only replayed novel levels, never a copied registry total.",
    "official_score": "Remain null unless an actual official evaluation exists.",
}


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def artifact_checksum(value: Mapping[str, Any]) -> str:
    payload = deepcopy(dict(value))
    payload["reproducibility_checksum"] = ""
    return "sha256:" + hashlib.sha256(_canonical_bytes(payload)).hexdigest()


def _historical_checksum(value: Mapping[str, Any]) -> str:
    payload = {key: item for key, item in value.items() if key != "reproducibility_checksum"}
    return "sha256:" + hashlib.sha256(_canonical_bytes(payload)).hexdigest()


def atomic_write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, indent=2, sort_keys=True, default=str) + "\n"
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False, encoding="utf-8") as tmp:
        tmp.write(payload)
        tmp.flush()
        os.fsync(tmp.fileno())
        temporary = Path(tmp.name)
    temporary.replace(path)


def gate_check(
    check: str,
    upstream: str,
    field: str,
    expected: Any,
    observed: Any,
    principle: str = "Reject unsafe evidence before it contributes to a score.",
) -> JsonDict:
    return {
        "check": check,
        "upstream": upstream,
        "field": field,
        "expected": expected,
        "observed": observed,
        "expected_value": expected,
        "observed_value": observed,
        "passed": observed == expected,
        "principle": principle,
    }


def gate_summary(checks: Sequence[Mapping[str, Any]]) -> JsonDict:
    failed = [deepcopy(dict(row)) for row in checks if row.get("passed") is not True]
    return {
        "passed": not failed,
        "failed_count": len(failed),
        "first_failure": failed[0] if failed else None,
        "failures": failed,
    }


def _quarantined(value: Mapping[str, Any]) -> bool:
    return bool(value.get("flagged_adversarial") or value.get("quarantined"))


def check_dependency(artifact: Mapping[str, Any] | None) -> JsonDict:
    """Return the first unsafe Experiment 7304 state without reading its score early."""

    upstream = EXP7304_PATH.as_posix()
    if artifact is None:
        return gate_check(
            "exp7304_dependency", upstream, "artifact", "complete", "missing_artifact"
        )
    if _quarantined(artifact):
        return gate_check(
            "exp7304_dependency", upstream, "quarantine_state", "unquarantined", "quarantined"
        )
    if artifact.get("verdict_class") == "disqualified":
        return gate_check(
            "exp7304_dependency", upstream, "verdict_class", "not_disqualified", "disqualified"
        )
    if artifact.get("status") != "complete":
        return gate_check(
            "exp7304_dependency", upstream, "status", "complete", artifact.get("status")
        )
    if artifact.get("arc_receipt_ready_score") != 1:
        return gate_check(
            "exp7304_dependency",
            upstream,
            "arc_receipt_ready_score",
            1,
            artifact.get("arc_receipt_ready_score"),
        )
    checksum = artifact.get("reproducibility_checksum")
    observed = checksum == artifact_checksum(artifact) if isinstance(checksum, str) else "missing"
    return gate_check(
        "exp7304_dependency", upstream, "reproducibility_checksum_valid", True, observed
    )


def authenticate_caller_hashes(
    root: Path,
    handoff: Mapping[str, Any],
    *,
    hasher: Callable[[Path], str] = sha256_file,
) -> list[JsonDict]:
    hashes = handoff.get("caller_code_hashes")
    if not isinstance(hashes, Mapping):
        return [
            gate_check(
                "exp7304_caller_hash",
                EXP7304_PATH.as_posix(),
                "caller_code_hashes",
                "mapping",
                "missing",
            )
        ]
    return [
        gate_check(
            "exp7304_caller_hash",
            str(relative),
            "sha256",
            expected,
            hasher(root / str(relative)),
        )
        for relative, expected in hashes.items()
    ]


def freeze_target(
    registry: Mapping[str, Any], *, adaptered_games: set[str] | frozenset[str]
) -> JsonDict:
    indexed = {
        str(row.get("game")): row
        for row in registry.get("games", [])
        if isinstance(row, Mapping) and row.get("game")
    }
    row = indexed.get(TARGET_GAME)
    return {
        "passed": row is not None,
        "target": TARGET_GAME,
        "seed": EVALUATION_SEED,
        "selection_basis": "predeclared_registered_public_transfer_probe",
        "registry_prechecked": row is not None,
        "registry_levels_before_attempt": int((row or {}).get("levels_reproduced") or 0),
        "registry_reproducibility": (row or {}).get("reproducibility"),
        "adapter_available_but_withheld": TARGET_GAME in adaptered_games,
        "adapter_disabled": True,
        "banked_solution_disabled": True,
        "outcomes_seen_before_freeze": False,
        "registered_levels_are_transfer_only": True,
        "game_source_read": False,
        "offline_ground_truth_search_used": False,
    }


def session_environment(
    base_env: Mapping[str, str],
    *,
    episode_dir: Path,
    gpu_index: int,
    port: int,
    boundary_path: Path | None = None,
    arm: str = "direct_selfparse",
) -> dict[str, str]:
    """Build the isolated direct-selfparse environment with fixed ceilings."""

    del arm
    env = dict(base_env)
    for retired in (
        "CARNOT_ARC_SUPERVISOR_TOOL_ARM",
        "CARNOT_ARC_SUPERVISOR_TOOL_LOOP_REINDUCTION",
        "CARNOT_ARC_TRANSITION_WITNESS",
    ):
        env.pop(retired, None)
    receipt_path = boundary_path or Path(
        env.get("CARNOT_ARC_BOUNDARY_LEDGER_PATH", str(REPO_ROOT / BOUNDARY_PATH))
    )
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
            "CARNOT_ARC_INDUCE_TOOL_TURNS": str(COMPLETION_LIMIT),
            "CARNOT_ARC_INDUCE_MAX_TOKENS": str(TOKENS_PER_CALL),
            "CARNOT_ARC_INDUCE_N_CTX": "49152",
            "CARNOT_ARC_INDUCE_TIMEOUT": str(MODEL_LOAD_LIMIT_S),
            "CARNOT_ARC_LLAMA_SERVER_PARALLEL": "1",
            "CARNOT_ARC_GENERATOR_REQUIRE_CUDA": "1",
            "CARNOT_ARC_GENERATOR_CUDA_GPU": str(gpu_index),
            "CARNOT_ARC_RANDOM_SEED": str(EVALUATION_SEED),
            "CARNOT_ARC_GENERATOR_SEED": str(EVALUATION_SEED),
            "CARNOT_ARC_PROPOSER_PORT": str(port),
            "CARNOT_ARC_MTP": "0",
            "CARNOT_ARC_ACTION_PROVENANCE": "1",
            "CARNOT_ARC_ACTION_PROVENANCE_DIR": str(episode_dir / "action_provenance"),
            "CARNOT_ARC_E3_DIR": str(episode_dir / "e3"),
            "CARNOT_ARC_SERVER_LOG_DIR": str(episode_dir.parent / "server_logs"),
            "CARNOT_ARC_BOUNDARY_LEDGER_PATH": str(receipt_path),
            "CUDA_VISIBLE_DEVICES": str(gpu_index),
        }
    )
    return env


def reduce_boundary_ledger(path: Path) -> JsonDict:
    from carnot.agentic.arc_inference_boundary import (
        InvocationBoundaryLedger,
        reduce_boundary_events,
    )

    reduced = reduce_boundary_events(InvocationBoundaryLedger(path).read_events())
    counts = reduced.get("invocation_counts")
    if isinstance(counts, dict):
        counts.setdefault("model_loads_cancelled", 0)
        counts.setdefault("generation_calls_cancelled", 0)
    return reduced


def _request_contains(path: str, bounded_response: str) -> bool:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False

    def contains(value: Any) -> bool:
        if isinstance(value, str):
            return bounded_response in value
        if isinstance(value, Mapping):
            return any(contains(item) for item in value.values())
        if isinstance(value, list):
            return any(contains(item) for item in value)
        return False

    return contains(payload)


def reduce_tool_use_chain(
    episode: Mapping[str, Any], tool_events: Sequence[Mapping[str, Any]]
) -> JsonDict:
    """Require tool success, later delivery, installed plan, and executed action."""

    requests = [row for row in episode.get("raw_request_manifest", []) if isinstance(row, Mapping)]
    inductions = [row for row in episode.get("induction_rows", []) if isinstance(row, Mapping)]
    consumptions = [
        row for row in episode.get("policy_consumption_rows", []) if isinstance(row, Mapping)
    ]
    actions = [row for row in episode.get("action_rows", []) if isinstance(row, Mapping)]
    rows: list[JsonDict] = []
    for event_index, event in enumerate(tool_events):
        induction_index = int(event.get("induction_index", event_index) or 0)
        turn = int(event.get("turn", 0) or 0)
        dispatch = event.get("dispatch_result")
        successful = isinstance(dispatch, Mapping) and dispatch.get("ok") is True
        bounded = str(event.get("bounded_response") or "")
        delivered = successful and any(
            int(request.get("call_index", -1) or -1) > turn
            and _request_contains(str(request.get("request_path") or ""), bounded)
            for request in requests
        )
        planned = bool(
            delivered
            and induction_index < len(inductions)
            and inductions[induction_index].get("planned") is True
        )
        matching = [
            row
            for row in consumptions
            if row.get("attempt_index") == induction_index
            and row.get("policy_action_executed") is True
        ]
        consumed = bool(planned and matching)
        acted = bool(
            consumed
            and any(
                int(action.get("i", -1) or -1) >= int(use.get("action_index", 0) or 0)
                for use in matching
                for action in actions
            )
        )
        rows.append(
            {
                "event_index": event_index,
                "induction_index": induction_index,
                "tool": event.get("parsed_tool"),
                "successful_tool_result": successful,
                "result_in_later_request": delivered,
                "induction_installed_plan": planned,
                "policy_consumed_result": consumed,
                "subsequent_environment_action": acted,
                "passed": bool(successful and delivered and planned and consumed and acted),
            }
        )
    return {
        "arc_tool_use_score": int(any(row["passed"] for row in rows)),
        "successful_tool_results": sum(row["successful_tool_result"] for row in rows),
        "results_in_later_request": sum(row["result_in_later_request"] for row in rows),
        "policy_consumed_results": sum(row["policy_consumed_result"] for row in rows),
        "subsequent_environment_actions": sum(row["subsequent_environment_action"] for row in rows),
        "rows": rows,
    }


def build_cumulative_ledger(
    output_path: Path,
    sources: Sequence[Path],
    *,
    quarantine_check: Callable[[Mapping[str, Any]], bool] = _quarantined,
) -> JsonDict:
    """Hash, reject, and deduplicate historical direct-selfparse inductions."""

    unique: dict[str, JsonDict] = {}
    receipts: list[JsonDict] = []
    for path in sources:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(payload, dict):
                raise ValueError("not_object")
            digest = sha256_file(path)
            read_error = None
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            payload = {}
            digest = "missing"
            read_error = f"{type(exc).__name__}: {exc}"[:200]
        observed = "available"
        completion = payload.get("arc_session_complete_score")
        checksum = payload.get("reproducibility_checksum")
        checksum_valid = (
            checksum in {artifact_checksum(payload), _historical_checksum(payload)}
            if isinstance(checksum, str)
            else None
        )
        rows = payload.get("cumulative_induction_rows")
        consumed = True
        if read_error:
            observed, consumed = "missing_or_unreadable", False
        elif quarantine_check(payload):
            observed, consumed = "quarantined", False
        elif payload.get("status") != "complete" or completion != 1:
            observed, consumed = "nonterminal_or_incomplete", False
        elif checksum_valid is False:
            observed, consumed = "checksum_invalid", False
        elif not isinstance(rows, list):
            observed, consumed = "induction_rows_missing", False
        if consumed:
            for raw in rows:
                if not isinstance(raw, Mapping) or raw.get("engaged") is not True:
                    continue
                identity = str(raw.get("induction_id") or "")
                if identity.startswith("sha256:") and len(identity) == 71:
                    unique.setdefault(identity, deepcopy(dict(raw)))
        receipts.append(
            {
                "path": str(path),
                "sha256": digest,
                "status": payload.get("status"),
                "arc_session_complete_score": completion,
                "artifact_checksum_valid": checksum_valid,
                "observed": observed,
                "consumed": consumed,
                "accepted_inductions": len(rows) if consumed and isinstance(rows, list) else 0,
            }
        )
    sidecar = {
        "schema": "carnot.experiment_7305.cumulative_induction_ledger.v1",
        "historical_only": True,
        "source_receipts": receipts,
        "rows": list(unique.values()),
        "authenticated_unique_inductions": len(unique),
        "evidence_goal": EVIDENCE_GOAL,
        "remaining_gap": max(0, EVIDENCE_GOAL - len(unique)),
    }
    atomic_write(output_path, sidecar)
    return {
        **sidecar,
        "path": str(output_path),
        "sha256": sha256_file(output_path),
    }


def _episode_accounted(episode: Mapping[str, Any]) -> bool:
    return bool(
        episode.get("disposition") in {"complete", "censored_timeout"}
        and episode.get("censored") in {True, False}
        and int(episode.get("action_count") or 0) <= ACTION_LIMIT
        and int(episode.get("generation_calls_attempted") or 0) <= COMPLETION_LIMIT
        and int(episode.get("generation_calls_completed") or 0)
        <= int(episode.get("generation_calls_attempted") or 0)
        and int(episode.get("generated_tokens") or 0) <= GENERATED_TOKEN_LIMIT
        and episode.get("adapter_disabled") is True
        and episode.get("banked_solution_disabled", True) is True
    )


def reduce_current_session(
    episode: Mapping[str, Any],
    boundary: Mapping[str, Any],
    tool_events: Sequence[Mapping[str, Any]],
) -> JsonDict:
    counts = boundary.get("invocation_counts")
    counts = (
        deepcopy(dict(counts)) if isinstance(counts, Mapping) else deepcopy(ZERO_INVOCATION_COUNTS)
    )
    chain = reduce_tool_use_chain(episode, tool_events)
    evidence = bool(
        boundary.get("activity_known") is True
        and boundary.get("disqualified") is False
        and boundary.get("model_invoked") is True
        and (counts.get("model_loads_attempted", 0) or counts.get("generation_calls_attempted", 0))
    )
    capture = int(evidence and _episode_accounted(episode))
    tool_score = int(capture == 1 and chain["arc_tool_use_score"] == 1)
    return {
        "arc_capture_complete_score": capture,
        "arc_tool_use_score": tool_score,
        "verdict_class": "circular_positive" if tool_score else "null",
        "invocation_counts": counts,
        "tool_use_chain": chain,
        "episode_accounted": _episode_accounted(episode),
        "boundary_evidence_authenticated": evidence,
    }


def _model_specs_from_boundary(boundary: Mapping[str, Any]) -> list[JsonDict]:
    rows = boundary.get("call_rows")
    identities = [
        row.get("model_identity")
        for row in rows
        if isinstance(rows, list) and isinstance(row, Mapping)
        if isinstance(row.get("model_identity"), Mapping)
    ]
    if not identities:
        return deepcopy(MODEL_SPECS)
    identity = dict(identities[0])
    return [
        {
            "hf_id": identity.get("model_repository") or MODEL_ID,
            "quantization": QUANTIZATION,
            "model_filename": identity.get("model_filename"),
            "model_revision": identity.get("model_revision"),
            "model_path": identity.get("model_path"),
        }
    ]


def _acceptance_results(
    reduction: Mapping[str, Any], validation_receipts: Sequence[Mapping[str, Any]]
) -> list[JsonDict]:
    validation_passed = all(row.get("passed") is True for row in validation_receipts)
    return [
        {
            "criterion": "authenticated_current_capture",
            "expected": 1,
            "observed": reduction["arc_capture_complete_score"],
            "passed": reduction["arc_capture_complete_score"] == 1,
            "principle": "A mechanism result needs current model evidence and complete censoring.",
        },
        {
            "criterion": "tool_result_to_policy_action_chain",
            "expected": 1,
            "observed": reduction["arc_tool_use_score"],
            "passed": reduction["arc_tool_use_score"] == 1,
            "principle": "Dispatch alone is not policy use; every later chain link must exist.",
        },
        {
            "criterion": "scoped_validation_passed",
            "expected": True,
            "observed": validation_passed,
            "passed": validation_passed,
            "principle": "A failing affected check disqualifies the result without erasing evidence.",
        },
    ]


def _field_principles(artifact: Mapping[str, Any]) -> JsonDict:
    return {
        key: FIELD_PRINCIPLES.get(
            key, "Keep this value explicit so the terminal record is auditable."
        )
        for key in artifact
    }


def _repository_health() -> JsonDict:
    path = REPO_ROOT / HISTORICAL_REPOSITORY_HEALTH_LOG
    return {
        "affects_required_checks": False,
        "historical_failures": [
            {
                "source_experiment": "exp7289-arc-boundary",
                "command": ".venv/bin/pytest -o addopts= tests/python -q --no-cov -n 0",
                "exit_code": 2,
                "resolved": False,
                "log_path": HISTORICAL_REPOSITORY_HEALTH_LOG.as_posix(),
                "log_sha256": sha256_file(path) if path.is_file() else None,
            }
        ],
    }


def build_terminal_artifact(
    *,
    started_at_utc: str,
    ended_at_utc: str,
    duration_s: float,
    preconditions: Sequence[Mapping[str, Any]],
    source_hashes: Mapping[str, Any],
    selection: Mapping[str, Any],
    episode: Mapping[str, Any],
    boundary: Mapping[str, Any],
    runner: Mapping[str, Any],
    tool_events: Sequence[Mapping[str, Any]],
    cumulative_ledger: Mapping[str, Any],
    validation_receipts: Sequence[Mapping[str, Any]],
    phase_spans: Sequence[Mapping[str, Any]],
) -> JsonDict:
    reduction = reduce_current_session(episode, boundary, tool_events)
    counts = reduction["invocation_counts"]
    generation_attempted = int(counts.get("generation_calls_attempted", 0) or 0) > 0
    model_invoked = bool(boundary.get("model_invoked"))
    substrate_class = (
        "model_full_generation"
        if generation_attempted
        else "model_load_no_generation"
        if model_invoked
        else "no_model_load"
    )
    substrate = (
        "live_llm_inference"
        if generation_attempted
        else "model_load_no_generation"
        if model_invoked
        else "offline_arcade_live_agent_runtime_self_discovery_no_llm"
    )
    tool_score = reduction["arc_tool_use_score"]
    validation_passed = all(row.get("passed") is True for row in validation_receipts)
    verdict_class = reduction["verdict_class"] if validation_passed else "disqualified"
    artifact: JsonDict = {
        "schema": SCHEMA,
        "status": "complete",
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "run_date": RUN_DATE,
        "started_at_utc": started_at_utc,
        "ended_at_utc": ended_at_utc,
        "duration_s": round(max(float(duration_s), 0.000001), 6),
        "phase_spans": [deepcopy(dict(row)) for row in phase_spans],
        "preconditions_checked": [deepcopy(dict(row)) for row in preconditions],
        "MODEL_SPECS": _model_specs_from_boundary(boundary) if model_invoked else [],
        "model_invoked": model_invoked,
        "invocation_counts": deepcopy(counts),
        "inference_substrate": substrate,
        "inference_substrate_class": substrate_class,
        "inference_mode": "live_gpu"
        if runner.get("task_linked_cuda_execution")
        else "not_verified",
        "execution_venue": "host",
        "execution_host": socket.gethostname(),
        "random_seed": {
            "development": DEVELOPMENT_SEED,
            "independent_evaluation": EVALUATION_SEED,
            "sealed_before_outcomes": True,
        },
        "source_artifact_hashes": deepcopy(dict(source_hashes)),
        "rows": [
            {
                "unit_id": episode.get("episode_id"),
                "arm": "direct_selfparse",
                "seed": episode.get("seed"),
                "metrics": {
                    "levels": int(episode.get("levels") or 0),
                    "arc_tool_use_score": tool_score,
                },
                "costs": {
                    "actions": int(episode.get("action_count") or 0),
                    "generated_tokens": int(episode.get("generated_tokens") or 0),
                },
                "errors": [str(episode.get("error"))] if episode.get("error") else [],
                "abstentions": int(tool_score == 0),
                "censored": bool(episode.get("censored")),
            }
        ],
        "sample_size_budget": {
            "planned_units": 1,
            "attempted_units": 1,
            "complete_units": int(episode.get("disposition") == "complete"),
            "censored_units": int(bool(episode.get("censored"))),
            "action_limit": ACTION_LIMIT,
            "completion_limit": COMPLETION_LIMIT,
            "generated_token_limit": GENERATED_TOKEN_LIMIT,
            "session_limit_s": SESSION_LIMIT_S,
            "model_load_limit_s": MODEL_LOAD_LIMIT_S,
            "stopping_rule": "one frozen target and seed; stop at the first inherited ceiling",
            "outcome_based_extension": False,
        },
        "acceptance_gate_results": _acceptance_results(reduction, validation_receipts),
        "gate_check_summary": gate_summary(preconditions),
        "verifier_is_oracle": True,
        "honest_verdict": (
            "complete_disqualified_scoped_validation_failed"
            if not validation_passed
            else "complete_circular_positive_runtime_tool_result_consumed_before_policy_action"
            if tool_score
            else "complete_null_no_runtime_tool_result_to_policy_action_chain"
        ),
        "verdict_class": verdict_class,
        "validation_receipts": [deepcopy(dict(row)) for row in validation_receipts],
        "arc_capture_complete_score": reduction["arc_capture_complete_score"],
        "arc_tool_use_score": tool_score,
        "tool_use_chain": reduction["tool_use_chain"],
        "solve_provenance": "live_agent_self_discovery",
        "per_game_results": [deepcopy(dict(episode))],
        "cumulative_induction_ledger": deepcopy(dict(cumulative_ledger)),
        "offline_reproduced": False,
        "reproduced_levels": 0,
        "official_score": None,
        "new_solve_claimed": False,
        "registry_modified": False,
        "production_default_changed": False,
        "runner_receipt": deepcopy(dict(runner)),
        "repository_health": _repository_health(),
        "reproducibility_checksum": "",
    }
    artifact["field_principles"] = _field_principles(artifact)
    artifact["field_principles"]["field_principles"] = (
        "Explain why each top-level field exists without changing its ordinary value."
    )
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
    summary = gate_summary(preconditions)
    first = summary["first_failure"] or gate_check(
        "unknown_external_precondition", EXPERIMENT_ID, "state", "available", "unknown"
    )
    artifact: JsonDict = {
        "schema": SCHEMA,
        "status": "blocked",
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "run_date": RUN_DATE,
        "started_at_utc": started_at_utc,
        "ended_at_utc": ended_at_utc,
        "duration_s": round(max(float(duration_s), 0.000001), 6),
        "phase_spans": [],
        "preconditions_checked": [deepcopy(dict(row)) for row in preconditions],
        "MODEL_SPECS": [],
        "model_invoked": False,
        "invocation_counts": deepcopy(ZERO_INVOCATION_COUNTS),
        "inference_substrate": "aggregation_from_upstream_artifacts",
        "inference_substrate_class": "blocked_no_run",
        "inference_mode": "not_invoked",
        "execution_venue": "host",
        "execution_host": socket.gethostname(),
        "random_seed": {
            "development": DEVELOPMENT_SEED,
            "independent_evaluation": EVALUATION_SEED,
            "sealed_before_outcomes": True,
        },
        "source_artifact_hashes": deepcopy(dict(source_hashes)),
        "rows": [],
        "sample_size_budget": {
            "planned_units": 1,
            "attempted_units": 0,
            "complete_units": 0,
            "censored_units": 1,
            "stopping_rule": "block before model work when an external prerequisite fails",
        },
        "acceptance_gate_results": [],
        "gate_check_summary": summary,
        "verifier_is_oracle": True,
        "honest_verdict": (
            "blocked_external_precondition:"
            f"{first['upstream']}:{first['check']}:{first['field']}:"
            f"observed={first['observed']!r}:expected={first['expected']!r}"
        ),
        "verdict_class": "blocked",
        "validation_receipts": [],
        "arc_capture_complete_score": 0,
        "arc_tool_use_score": 0,
        "tool_use_chain": {"arc_tool_use_score": 0, "rows": []},
        "solve_provenance": "development_proxy",
        "per_game_results": [],
        "cumulative_induction_ledger": {
            "authenticated_unique_inductions": 0,
            "evidence_goal": EVIDENCE_GOAL,
            "remaining_gap": EVIDENCE_GOAL,
        },
        "offline_reproduced": False,
        "reproduced_levels": 0,
        "official_score": None,
        "new_solve_claimed": False,
        "registry_modified": False,
        "production_default_changed": False,
        "runner_receipt": {},
        "repository_health": _repository_health(),
        "reproducibility_checksum": "",
    }
    artifact["field_principles"] = _field_principles(artifact)
    artifact["field_principles"]["field_principles"] = (
        "Explain why each top-level field exists without changing its ordinary value."
    )
    artifact["reproducibility_checksum"] = artifact_checksum(artifact)
    return artifact


def validate_artifact(value: Mapping[str, Any]) -> list[str]:
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
    prefix = "blocked_" if artifact.get("status") == "blocked" else "complete_"
    if not str(artifact.get("honest_verdict") or "").startswith(prefix):
        errors.append("honest_verdict_prefix_invalid")
    if artifact.get("reproducibility_checksum") != artifact_checksum(artifact):
        errors.append("reproducibility_checksum_mismatch")
    principles = artifact.get("field_principles")
    if not isinstance(principles, Mapping) or set(principles) != set(artifact):
        errors.append("field_principles_incomplete")
    if artifact.get("official_score") is not None:
        errors.append("official_score_must_be_null")
    if (
        artifact.get("registry_modified") is not False
        or artifact.get("production_default_changed") is not False
    ):
        errors.append("forbidden_state_change")
    if artifact.get("status") == "blocked":
        if (
            artifact.get("model_invoked") is not False
            or artifact.get("invocation_counts") != ZERO_INVOCATION_COUNTS
        ):
            errors.append("blocked_invocation_evidence_invalid")
        if artifact.get("gate_check_summary", {}).get("first_failure") is None:
            errors.append("blocked_failure_summary_missing")
    else:
        per_game = artifact.get("per_game_results")
        episode = per_game[0] if isinstance(per_game, list) and len(per_game) == 1 else {}
        boundary = {
            "activity_known": True,
            "disqualified": False,
            "model_invoked": artifact.get("model_invoked"),
            "invocation_counts": artifact.get("invocation_counts"),
        }
        events = artifact.get("tool_use_chain", {}).get("rows", [])
        if not isinstance(episode, Mapping) or not _episode_accounted(episode):
            errors.append("episode_accounting_invalid")
        if artifact.get("arc_capture_complete_score") != int(
            boundary["model_invoked"] is True and _episode_accounted(episode)
        ):
            errors.append("capture_score_inconsistent")
        if artifact.get("arc_tool_use_score") != int(
            any(isinstance(row, Mapping) and row.get("passed") is True for row in events)
        ):
            errors.append("tool_score_inconsistent")
        if artifact.get("model_invoked") and artifact.get("duration_s", 0) < (
            60 if artifact.get("inference_substrate_class") == "model_full_generation" else 2
        ):
            errors.append("substrate_duration_floor_failed")
    return errors


def build_validation_commands(
    *, terminal_candidate: Path, raw_row: Path, private_root: Path
) -> list[JsonDict]:
    root = REPO_ROOT
    python = str(root / ".venv/bin/python")
    pytest = str(root / ".venv/bin/pytest")
    coverage = str(root / ".venv/bin/coverage")
    ruff = str(root / ".venv/bin/ruff")
    mypy = str(root / ".venv/bin/mypy")
    common = ["-n", "0", "-o", "addopts=", "--no-cov"]
    static = [MODULE_PATH.as_posix(), WRAPPER_PATH.as_posix(), TEST_PATH.as_posix()]
    coverage_file = private_root / ".coverage"
    include = "*/experiment_7305_v642_arc_selfparse.py"

    def command(name: str, argv: Sequence[str], scope: str) -> JsonDict:
        return {"name": name, "argv": list(argv), "scope": scope}

    return [
        command(
            "focused_exp7305",
            [pytest, *common, f"--basetemp={private_root / 'focused'}", TEST_PATH.as_posix(), "-q"],
            "explicit_test",
        ),
        command(
            "affected_boundary_receipts",
            [
                pytest,
                *common,
                f"--basetemp={private_root / 'boundary'}",
                "tests/python/test_experiment_7289_v641_arc_boundary.py",
                "-q",
            ],
            "affected_boundary",
        ),
        command(
            "affected_live_policy",
            [
                pytest,
                *common,
                f"--basetemp={private_root / 'policy'}",
                "tests/python/test_experiment_7234_v637_arc_scored_dryrun.py",
                "-q",
            ],
            "affected_policy",
        ),
        command(
            "e2e_009_cross_call_persistence",
            [
                pytest,
                *common,
                f"--basetemp={private_root / 'e2e009'}",
                "tests/python/test_arc_induction_state_persistence.py",
                "-q",
            ],
            "e2e_009",
        ),
        command(
            "e2e_010_tool_transport",
            [
                pytest,
                *common,
                f"--basetemp={private_root / 'e2e010'}",
                "tests/python/test_arc_tool_grammar_transport.py",
                "-q",
            ],
            "e2e_010",
        ),
        command(
            "e2e_009_llm_off_environment_smoke",
            [
                python,
                "-u",
                "scripts/arc_loop_solve.py",
                "--mechanism",
                "e3",
                "--game",
                TARGET_GAME,
                "--max-actions",
                "12",
                "--output",
                str(private_root / "e2e009-llm-off.json"),
            ],
            "e2e_009_llm_off_environment",
        ),
        command(
            "changed_module_coverage",
            [
                coverage,
                "run",
                f"--data-file={coverage_file}",
                f"--include={include}",
                "-m",
                "pytest",
                *common,
                f"--basetemp={private_root / 'coverage'}",
                TEST_PATH.as_posix(),
                "-q",
            ],
            "changed_module_and_test",
        ),
        command(
            "changed_module_coverage_report",
            [
                coverage,
                "report",
                f"--data-file={coverage_file}",
                f"--include={include}",
                "--show-missing",
                "--fail-under=100",
            ],
            "changed_module",
        ),
        command("ruff_check", [ruff, "check", *static], "changed_files"),
        command("ruff_format", [ruff, "format", "--check", *static], "changed_files"),
        command("changed_module_mypy", [mypy, MODULE_PATH.as_posix()], "changed_module"),
        command(
            "scoped_spec_coverage",
            [
                python,
                "-u",
                "scripts/check_spec_coverage.py",
                TEST_PATH.as_posix(),
                "tests/python/test_experiment_7289_v641_arc_boundary.py",
                "tests/python/test_experiment_7234_v637_arc_scored_dryrun.py",
                "tests/python/test_arc_induction_state_persistence.py",
                "tests/python/test_arc_tool_grammar_transport.py",
            ],
            "exact_tests",
        ),
        command(
            "independent_raw_reducer",
            [python, "-u", str(root / WRAPPER_PATH), "--reduce-raw", str(raw_row)],
            "raw_current_session",
        ),
        command(
            "terminal_candidate_adversarial_verify",
            [python, "-u", "scripts/adversarial_verify.py", str(terminal_candidate)],
            "measured_terminal_candidate",
        ),
        command(
            "terminal_candidate_row_consistency_strict",
            [
                python,
                "-u",
                "scripts/verdict_row_consistency_lint.py",
                "--strict",
                str(terminal_candidate),
            ],
            "measured_terminal_candidate",
        ),
    ]


class _DurableToolSink(list[JsonDict]):  # pragma: no cover - live child only.
    """Flush each successful dispatch before the next blocking generation starts."""

    def __init__(self, path: Path) -> None:
        super().__init__()
        self.path = path
        self.induction_index = 0

    def append(self, value: JsonDict) -> None:
        row = {
            "episode_id": f"{TARGET_GAME}:direct_selfparse",
            "induction_index": self.induction_index,
            **deepcopy(value),
        }
        super().append(row)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(self.path, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o600)
        try:
            os.write(fd, _canonical_bytes(row) + b"\n")
            os.fsync(fd)
        finally:
            os.close(fd)


@contextmanager
def _configured_runtime() -> Any:  # pragma: no cover - live process integration.
    from carnot import experiment_7280_v640_arc_live as reused
    from carnot import experiment_7263_v639_arc_live as live

    updates = {
        "TASK_ID": EXPERIMENT_ID,
        "EXPERIMENT_ID": EXPERIMENT_ID,
        "MILESTONE": MILESTONE,
        "RUN_DATE": RUN_DATE,
        "RANDOM_SEED": EVALUATION_SEED,
        "ACTION_LIMIT": ACTION_LIMIT,
        "COMPLETION_LIMIT": COMPLETION_LIMIT,
        "GENERATED_TOKEN_LIMIT": GENERATED_TOKEN_LIMIT,
        "TOKENS_PER_CALL": TOKENS_PER_CALL,
        "SESSION_LIMIT_S": SESSION_LIMIT_S,
        "MODEL_LOAD_LIMIT_S": MODEL_LOAD_LIMIT_S,
        "RESULT_PATH": RESULT_PATH,
        "CHECKPOINT_PATH": CHECKPOINT_PATH,
        "RAW_DIR": RAW_DIR,
        "RAW_ROWS_PATH": RAW_ROW_PATH,
        "TERMINAL_CANDIDATE_PATH": TERMINAL_CANDIDATE_PATH,
        "SESSION_PATH": SESSION_PATH,
        "MODULE_PATH": MODULE_PATH,
        "WRAPPER_PATH": WRAPPER_PATH,
        "TEST_PATH": TEST_PATH,
    }
    old = {name: getattr(reused, name) for name in updates}
    old_environment = live.episode_environment
    old_live_run_date = live.RUN_DATE
    try:
        for name, value in updates.items():
            setattr(reused, name, value)
        live.RUN_DATE = RUN_DATE
        live.episode_environment = session_environment
        yield reused
    finally:
        live.episode_environment = old_environment
        live.RUN_DATE = old_live_run_date
        for name, value in old.items():
            setattr(reused, name, value)


def run_live_session(args: argparse.Namespace) -> int:  # pragma: no cover - live child only.
    from carnot.agentic import arc_induction_tool_loop as tool_loop

    sink = _DurableToolSink(Path(args.raw_dir) / TOOL_EVENT_PATH.name)
    original = tool_loop.induce_with_tool_loop
    induction_index = 0

    def instrumented(*call_args: Any, **kwargs: Any) -> Any:
        nonlocal induction_index
        sink.induction_index = induction_index
        kwargs["tool_event_sink"] = sink
        induction_index += 1
        return original(*call_args, **kwargs)

    tool_loop.induce_with_tool_loop = instrumented
    try:
        with _configured_runtime() as reused:
            return int(reused.run_live_session(args))
    finally:
        tool_loop.induce_with_tool_loop = original


def _utc_now() -> str:  # pragma: no cover - integration timestamp.
    return datetime.now(UTC).isoformat()


def _progress(started: float, phase: str, event: str, **fields: Any) -> None:  # pragma: no cover
    print(
        json.dumps(
            {
                "phase": phase,
                "event": event,
                "elapsed_s": round(time.monotonic() - started, 3),
                **fields,
            },
            sort_keys=True,
            default=str,
        ),
        flush=True,
    )


def _load_json(path: Path) -> JsonDict | None:  # pragma: no cover - integration I/O.
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _manifest_rejects(manifest: Mapping[str, Any], experiment_id: str) -> bool:  # pragma: no cover
    for key in ("retired", "retired_experiments", "retired_extras"):
        rows = manifest.get(key, [])
        if any(
            experiment_id in {str(row), str(row.get("id")), str(row.get("experiment_id"))}
            for row in rows
            if not isinstance(rows, (str, bytes))
            if isinstance(row, (str, Mapping))
        ):
            return True
    return False


def _gpu_inventory() -> list[JsonDict]:  # pragma: no cover - external resource.
    from carnot import experiment_7234_v637_arc_scored_dryrun as scored

    return [dict(row) for row in scored._gpu_inventory()]


def collect_preconditions(
    root: Path, *, gpu_wait_s: float = 120.0
) -> tuple[list[JsonDict], JsonDict, JsonDict]:  # pragma: no cover - integration preflight.
    started = time.monotonic()
    checks: list[JsonDict] = []
    hashes: JsonDict = {}
    required = (
        SPEC_PATH,
        EXP7304_PATH,
        EXCLUSION_PATH,
        REGISTRY_PATH,
        Path("ops/e2e-test-plan.md"),
        Path("ops/known-issues.md"),
        Path("ops/north-star.md"),
        Path("research-program.md"),
        Path("python/carnot/agentic/arc_competition_agent.py"),
        Path("python/carnot/agentic/arc_executable_world_model.py"),
        Path("python/carnot/agentic/arc_induction_tool_loop.py"),
        Path("python/carnot/agentic/arc_inference_boundary.py"),
        Path("python/carnot/inference/sota_models.py"),
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
                "terminal_class": "input",
                "quarantined": False,
                "retired": False,
            }
    failed_attempt = root / FAILED_DATE_OVERRIDE_ATTEMPT_DIR
    if failed_attempt.is_dir():
        for path in sorted(item for item in failed_attempt.rglob("*") if item.is_file()):
            hashes[str(path.relative_to(root))] = {
                "sha256": sha256_file(path),
                "terminal_class": "failed_pre_model_child_date_override_attempt",
                "quarantined": False,
                "retired": False,
            }
    spec = (root / SPEC_PATH).read_text(encoding="utf-8") if (root / SPEC_PATH).is_file() else ""
    checks.append(
        gate_check(
            "driving_capability",
            SPEC_PATH.as_posix(),
            "REQ-ARC-WMTE-7305",
            True,
            "REQ-ARC-WMTE-7305" in spec,
        )
    )
    dependency = _load_json(root / EXP7304_PATH)
    dependency_gate = check_dependency(dependency)
    checks.append(dependency_gate)
    if dependency is not None:
        hashes[EXP7304_PATH.as_posix()] = {
            "sha256": sha256_file(root / EXP7304_PATH),
            "producer_identity": dependency.get("experiment_id"),
            "terminal_class": dependency.get("verdict_class"),
            "quarantined": _quarantined(dependency),
            "retired": False,
        }
        handoff = dependency.get("caller_handoff")
        if isinstance(handoff, Mapping):
            checks.extend(authenticate_caller_hashes(root, handoff))
            sealed = dict(handoff)
            expected_handoff = sealed.pop("handoff_sha256", None)
            observed_handoff = "sha256:" + hashlib.sha256(_canonical_bytes(sealed)).hexdigest()
            checks.append(
                gate_check(
                    "exp7304_handoff",
                    EXP7304_PATH.as_posix(),
                    "handoff_sha256",
                    expected_handoff,
                    observed_handoff,
                )
            )
        else:
            checks.append(
                gate_check(
                    "exp7304_handoff",
                    EXP7304_PATH.as_posix(),
                    "caller_handoff",
                    "mapping",
                    "missing",
                )
            )
    try:
        manifest = yaml.safe_load((root / EXCLUSION_PATH).read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError):
        manifest = {}
    checks.append(
        gate_check(
            "exclusion_manifest",
            EXCLUSION_PATH.as_posix(),
            "exp7304_not_retired",
            False,
            _manifest_rejects(manifest, "exp7304-arc-receipt"),
        )
    )
    for relative in (RESULT_PATH.parent, RAW_DIR, CHECKPOINT_PATH.parent):
        path = root / relative
        path.mkdir(parents=True, exist_ok=True)
        checks.append(
            gate_check(
                "output_path",
                relative.as_posix(),
                "owner_and_writable",
                True,
                path.is_dir() and os.access(path, os.W_OK) and path.stat().st_uid == os.getuid(),
            )
        )

    idle: list[JsonDict] = []
    deadline = time.monotonic() + max(0.0, gpu_wait_s)
    while True:
        idle = [
            row
            for row in _gpu_inventory()
            if not row.get("compute_apps") and int(row.get("free_memory_mb") or 0) >= 20_000
        ]
        if idle or time.monotonic() >= deadline:
            break
        _progress(
            started,
            "gpu_reservation",
            "waiting_for_idle_gpu",
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
    tokenizer_ok, tokenizer_detail = gguf_tokenizer_loadable(str(model_path) if model_ok else None)
    checks.append(
        gate_check(
            "embedded_tokenizer", str(model_path), "loadable", True, tokenizer_ok, tokenizer_detail
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
        _progress(started, "model_hash", "before_model_hash", path=str(model_path))
        model_hash = sha256_file(model_path)
        hashes[str(model_path)] = {
            "sha256": model_hash,
            "terminal_class": "current_model",
            "quarantined": False,
            "retired": False,
        }
        _progress(started, "model_hash", "after_model_hash", sha256=model_hash)
    return (
        checks,
        hashes,
        {
            "model_path": str(model_path) if model_ok and model_path else None,
            "model_hash": model_hash,
            "model_spec": model,
            "server": str(server) if server else None,
            "gpu": gpu,
        },
    )


def _read_tool_events(path: Path) -> list[JsonDict]:  # pragma: no cover - integration I/O.
    if not path.is_file():
        return []
    rows: list[JsonDict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            rows.append(value)
    return rows


def _censored_episode(session: Mapping[str, Any]) -> JsonDict:  # pragma: no cover
    existing = session.get("episodes")
    if isinstance(existing, list) and existing and isinstance(existing[0], Mapping):
        row = deepcopy(dict(existing[0]))
        row.setdefault("banked_solution_disabled", True)
        return row
    requests = session.get("requests") if isinstance(session.get("requests"), list) else []
    return {
        "episode_id": f"{TARGET_GAME}:direct_selfparse",
        "game": TARGET_GAME,
        "seed": EVALUATION_SEED,
        "arm": "direct_selfparse",
        "disposition": "censored_timeout",
        "censored": True,
        "model_loaded": bool(session.get("model_loaded")),
        "model_invoked": bool(session.get("model_invoked")),
        "generation_calls_attempted": len(requests),
        "generation_calls_completed": sum(
            isinstance(row, Mapping) and row.get("transport_completed") is True for row in requests
        ),
        "generated_tokens": sum(
            int(row.get("completion_tokens") or 0) for row in requests if isinstance(row, Mapping)
        ),
        "action_count": 0,
        "action_limit": ACTION_LIMIT,
        "levels": 0,
        "adapter_disabled": True,
        "banked_solution_disabled": True,
        "induction_rows": [],
        "policy_consumption_rows": [],
        "raw_request_manifest": deepcopy(requests),
        "action_rows": [],
        "error": session.get("error") or "session_censored_before_episode_row",
    }


def _run_commands(
    commands: Sequence[Mapping[str, Any]], log_dir: Path, started: float
) -> list[JsonDict]:  # pragma: no cover - subprocess validation.
    from carnot.reporting.experiment_7303_validation_scope import CommandSpec, run_commands

    specs = [
        CommandSpec(str(row["name"]), tuple(str(item) for item in row["argv"]), str(row["scope"]))
        for row in commands
    ]
    _progress(started, "validation", "before_subprocess_group", units=len(specs))
    rows = run_commands(REPO_ROOT, specs, log_dir=log_dir, heartbeat_s=60.0)
    _progress(
        started,
        "validation",
        "after_subprocess_group",
        units=len(rows),
        passed=all(row.get("passed") is True for row in rows),
    )
    return rows


def independent_reduce(path: Path) -> JsonDict:  # pragma: no cover - CLI validation path.
    payload = json.loads(path.read_text(encoding="utf-8"))
    return reduce_current_session(payload["episode"], payload["boundary"], payload["tool_events"])


def run_experiment(args: argparse.Namespace) -> JsonDict:  # pragma: no cover - live orchestration.
    started = time.monotonic()
    started_utc = _utc_now()
    phase_spans: list[JsonDict] = []
    _progress(started, "startup", "entrypoint_and_paths_authenticated", run_date=args.date)
    _progress(started, "preconditions", "begin")
    phase_started = time.monotonic()
    checks, hashes, resources = collect_preconditions(REPO_ROOT, gpu_wait_s=args.gpu_wait_s)
    phase_spans.append({"phase": "preconditions", "duration_s": time.monotonic() - phase_started})
    _progress(
        started,
        "preconditions",
        "end",
        passed=all(row["passed"] for row in checks),
        units=len(checks),
    )
    if not all(row["passed"] for row in checks):
        artifact = build_blocked_artifact(
            started_at_utc=started_utc,
            ended_at_utc=_utc_now(),
            duration_s=time.monotonic() - started,
            preconditions=checks,
            source_hashes=hashes,
        )
        errors = validate_artifact(artifact)
        if errors:
            raise RuntimeError(f"blocked artifact validation failed: {errors}")
        atomic_write(REPO_ROOT / RESULT_PATH, artifact)
        _progress(started, "publication", "terminal_blocked_artifact_written", path=RESULT_PATH)
        return artifact

    _progress(started, "target_freeze", "begin")
    phase_started = time.monotonic()
    registry = yaml.safe_load((REPO_ROOT / REGISTRY_PATH).read_text(encoding="utf-8")) or {}
    from carnot.agentic.arc_game_adapters import adaptered_games

    selection = freeze_target(registry, adaptered_games=set(adaptered_games()))
    target_gate = gate_check(
        "registry_precheck_and_freeze",
        REGISTRY_PATH.as_posix(),
        TARGET_GAME,
        True,
        selection["passed"],
    )
    checks.append(target_gate)
    atomic_write(REPO_ROOT / RAW_DIR / "frozen_target.json", selection)
    _progress(started, "target_freeze", "end", passed=selection["passed"], target=TARGET_GAME)
    if not target_gate["passed"]:
        artifact = build_blocked_artifact(
            started_at_utc=started_utc,
            ended_at_utc=_utc_now(),
            duration_s=time.monotonic() - started,
            preconditions=checks,
            source_hashes=hashes,
        )
        atomic_write(REPO_ROOT / RESULT_PATH, artifact)
        return artifact

    cumulative = build_cumulative_ledger(
        REPO_ROOT / RAW_DIR / "cumulative_inductions.json",
        [REPO_ROOT / path for path in HISTORICAL_PATHS],
    )
    _progress(
        started,
        "historical_ledger",
        "complete",
        unique=cumulative["authenticated_unique_inductions"],
        remaining_gap=cumulative["remaining_gap"],
    )
    schedule_path = REPO_ROOT / RAW_DIR / "frozen_schedule.json"
    schedule = [
        {
            "episode_id": f"{TARGET_GAME}:direct_selfparse",
            "game": TARGET_GAME,
            "arm": "direct_selfparse",
            "seed": EVALUATION_SEED,
            "execution_order": 0,
            "adapter_disabled": True,
        }
    ]
    atomic_write(schedule_path, {"selection_receipt": selection, "rows": schedule})
    atomic_write(
        REPO_ROOT / CHECKPOINT_PATH,
        {"status": "running", "phase": "live_window", "completed_units": 0},
    )
    phase_spans.append(
        {"phase": "target_freeze_and_history", "duration_s": time.monotonic() - phase_started}
    )

    _progress(
        started, "live_window", "before_model_load_generation_benchmark", cap_s=SESSION_LIMIT_S
    )
    live_started = time.monotonic()
    with _configured_runtime() as reused:
        session = reused.run_child_with_lease(
            resources=resources,
            schedule_path=schedule_path,
            raw_dir=REPO_ROOT / RAW_DIR,
            checkpoint_path=REPO_ROOT / CHECKPOINT_PATH,
            session_path=REPO_ROOT / SESSION_PATH,
            remaining_s=max(1.0, SESSION_LIMIT_S - (time.monotonic() - started)),
        )
    live_duration = time.monotonic() - live_started
    phase_spans.append({"phase": "live_window", "duration_s": live_duration})
    _progress(
        started,
        "live_window",
        "after_model_load_generation_benchmark",
        elapsed_s=round(live_duration, 3),
        model_invoked=session.get("model_invoked"),
    )
    episode = _censored_episode(session)
    boundary = reduce_boundary_ledger(REPO_ROOT / BOUNDARY_PATH)
    tool_events = _read_tool_events(REPO_ROOT / TOOL_EVENT_PATH)
    raw_input = {"episode": episode, "boundary": boundary, "tool_events": tool_events}
    atomic_write(REPO_ROOT / RAW_ROW_PATH, raw_input)
    for path in (
        REPO_ROOT / BOUNDARY_PATH,
        REPO_ROOT / TOOL_EVENT_PATH,
        REPO_ROOT / SESSION_PATH,
        REPO_ROOT / RAW_ROW_PATH,
        REPO_ROOT / RAW_DIR / "cumulative_inductions.json",
    ):
        if path.is_file():
            hashes[str(path.relative_to(REPO_ROOT))] = {
                "sha256": sha256_file(path),
                "terminal_class": "current_raw_sidecar",
                "quarantined": False,
                "retired": False,
            }

    private = Path(tempfile.mkdtemp(prefix="exp7305-validation-", dir="/tmp"))
    commands = build_validation_commands(
        terminal_candidate=REPO_ROOT / TERMINAL_CANDIDATE_PATH,
        raw_row=REPO_ROOT / RAW_ROW_PATH,
        private_root=private,
    )
    nonterminal = commands[:-2]
    validation_started = time.monotonic()
    validation = _run_commands(nonterminal, REPO_ROOT / RAW_DIR / "validation", started)
    phase_spans.append(
        {
            "phase": "scoped_validation",
            "duration_s": time.monotonic() - validation_started,
        }
    )
    candidate = build_terminal_artifact(
        started_at_utc=started_utc,
        ended_at_utc=_utc_now(),
        duration_s=time.monotonic() - started,
        preconditions=checks,
        source_hashes=hashes,
        selection=selection,
        episode=episode,
        boundary=boundary,
        runner=session.get("runtime_receipt", {}),
        tool_events=tool_events,
        cumulative_ledger={key: value for key, value in cumulative.items() if key != "rows"},
        validation_receipts=validation,
        phase_spans=phase_spans,
    )
    atomic_write(REPO_ROOT / TERMINAL_CANDIDATE_PATH, candidate)
    terminal_validation_started = time.monotonic()
    validation.extend(
        _run_commands(commands[-2:], REPO_ROOT / RAW_DIR / "terminal_validation", started)
    )
    phase_spans.append(
        {
            "phase": "terminal_validation",
            "duration_s": time.monotonic() - terminal_validation_started,
        }
    )
    artifact = build_terminal_artifact(
        started_at_utc=started_utc,
        ended_at_utc=_utc_now(),
        duration_s=time.monotonic() - started,
        preconditions=checks,
        source_hashes=hashes,
        selection=selection,
        episode=episode,
        boundary=boundary,
        runner=session.get("runtime_receipt", {}),
        tool_events=tool_events,
        cumulative_ledger={key: value for key, value in cumulative.items() if key != "rows"},
        validation_receipts=validation,
        phase_spans=phase_spans,
    )
    errors = validate_artifact(artifact)
    if errors:
        raise RuntimeError(f"terminal artifact validation failed: {errors}")
    atomic_write(REPO_ROOT / TERMINAL_CANDIDATE_PATH, artifact)
    atomic_write(REPO_ROOT / RESULT_PATH, artifact)
    atomic_write(
        REPO_ROOT / CHECKPOINT_PATH,
        {
            "status": "complete",
            "phase": "terminal_published",
            "completed_units": 1,
            "result_path": RESULT_PATH.as_posix(),
        },
    )
    _progress(
        started,
        "publication",
        "terminal_artifact_atomically_written",
        path=RESULT_PATH,
        arc_capture_complete_score=artifact["arc_capture_complete_score"],
        arc_tool_use_score=artifact["arc_tool_use_score"],
    )
    return artifact


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:  # pragma: no cover - CLI.
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


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - CLI.
    started = time.monotonic()
    _progress(started, "startup", "entrypoint")
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
