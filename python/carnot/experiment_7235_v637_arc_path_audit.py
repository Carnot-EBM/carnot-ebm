"""Audit whether an archived scored-path engine was authentic, useful, and used.

The producer already proved that model traffic and engine writes occurred. This
module checks the stronger claim independently: the saved bytes must join to one
session, predict later observations, and affect a later policy action. The audit
executes only archived Python engines on archived public transitions. It never
loads or invokes an LLM and never reads game source.

Spec refs: REQ-ARC-WMTE-7235 and SCENARIO-ARC-WMTE-7235-*.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from copy import deepcopy
from datetime import UTC, datetime
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import random
import socket
import time
from typing import Any, Callable

import numpy as np

from carnot import experiment_7206_v635_arc_volume_a as base
from carnot import experiment_7234_v637_arc_scored_dryrun as producer
from carnot.agentic.arc_executable_world_model import Transition, WorldModelVerifier
from carnot.agentic.arc_induction_tools import parse_xml_tool_calls


JsonDict = dict[str, Any]
Engine = Callable[[np.ndarray, int, Any], np.ndarray]

REPO_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_ID = 7235
MILESTONE = "2026.09.637"
RUN_DATE = "20260912"
RANDOM_SEED = 7_235_001
MODEL_SPECS: list[JsonDict] = []

SCHEMA = "carnot.experiment_7235.v637_arc_path_audit.v1"
SPEC_PATH = Path("openspec/capabilities/arc-world-model-trust-energy/spec.md")
UPSTREAM_PATH = Path("results/experiment_7234_v637_arc_scored_dryrun.json")
SESSION_PATH = Path("results/raw/experiment_7234/llamacpp/session.json")
MANIFEST_PATH = Path("results/raw/experiment_7234/llamacpp/e3/lp85/attempts/manifest.jsonl")
ACTION_ROWS_PATH = Path("results/raw/experiment_7234/llamacpp/action_rows.json")
INDUCTION_ROWS_PATH = Path("results/raw/experiment_7234/llamacpp/induction_rows.json")
ACTION_PROVENANCE_PATH = Path(
    "results/raw/experiment_7234/llamacpp/action_provenance/action_provenance_lp85_lp85.json"
)
RESULT_PATH = Path("results/experiment_7235_v637_arc_path_audit.json")
CHECKPOINT_PATH = Path("results/checkpoints/experiment_7235_v637_arc_path_audit/running.json")
RAW_DIR = Path("results/raw/experiment_7235")
MODULE_PATH = Path("python/carnot/experiment_7235_v637_arc_path_audit.py")
WRAPPER_PATH = Path("scripts/experiments/experiment_7235_v637_arc_path_audit.py")
TEST_PATH = Path("tests/python/test_experiment_7235_v637_arc_path_audit.py")

REQUIRED_SOURCE_PATHS = (
    Path("AGENTS.md"),
    Path("CLAUDE.md"),
    Path("CODEX.md"),
    Path("research-program.md"),
    Path("research-references.md"),
    Path("ops/exclusion_manifest.yaml"),
    Path("ops/e2e-test-plan.md"),
    Path("python/carnot/agentic/arc_eval_provenance.py"),
    Path("python/carnot/agentic/arc_competition_agent.py"),
    Path("python/carnot/agentic/arc_induction_tool_loop.py"),
    Path("python/carnot/agentic/arc_tool_gap_receipt.py"),
    Path("scripts/arc_loop_solve.py"),
    Path("ops/arc_solve_registry.yaml"),
    Path("results/experiment_7221_v636_arc_session.json"),
    SPEC_PATH,
    MODULE_PATH,
    WRAPPER_PATH,
    TEST_PATH,
)

FIELD_PRINCIPLES: JsonDict = {
    "schema": "Version the artifact and bind experiment_id and milestone to this task.",
    "experiment_id": "Bind this receipt to Experiment 7235.",
    "milestone": "Bind this receipt to milestone 2026.09.637.",
    "status": "Terminal complete or blocked only; unfinished work uses a separate checkpoint path.",
    "run_date": "Use 20260912; retain actual UTC start and end timestamps.",
    "started_at_utc": "Retain the actual UTC audit start.",
    "ended_at_utc": "Retain the actual UTC terminal construction time.",
    "field_principles": "Keep ordinary values at top level; put their explanations in this map.",
    "preconditions_checked": "Observed paths, resources, model identity and upstream checks before expensive work.",
    "inference_substrate": "Use the recognized literal for the operation actually executed.",
    "inference_substrate_class": "Actual class determines the duration floor; never pad duration or relabel to pass.",
    "execution_venue": "Top-level orchestration is host; board rows name kv260, gatemate or polarfire.",
    "execution_host": "Actual hostname, distinct from execution_venue.",
    "duration_s": "Measured monotonic elapsed work; record phase spans separately.",
    "phase_spans": "Measured preflight, replay, scoring, and validation spans.",
    "MODEL_SPECS": "Models actually invoked; [] for tasks with no LLM.",
    "model_invoked": "Current task execution only; historical sources and injected fixtures are separate.",
    "source_artifact_hashes": "Hash source code, public inputs, private evaluator inputs and raw output files.",
    "rows": "Every comparison retains one row per independent unit and arm, with errors and abstentions.",
    "audit_rows": "Per transition, tool, and action evidence keeps source hashes, useful prediction, and policy use separate.",
    "sample_size_budget": "Predeclared independent units, attempted/completed/censored units, and stopping rule.",
    "random_seed": "Freeze seeds and schedules before observing evaluation labels.",
    "reproducibility_checksum": "Hash the exact settings, inputs and raw rows supporting the result.",
    "gate_check_summary": "Every blocked_* verdict names check, upstream, artifact_field, expected and observed value.",
    "verifier_is_oracle": "True when the verification authority also defines correctness; separate code is insufficient independence.",
    "verdict_class": "Exactly positive | circular_positive | null | blocked | disqualified | partial. External incompleteness is blocked.",
    "honest_verdict": "Completed findings start complete_ or complete:. External absence starts blocked_. Failed acceptance forbids positive.",
    "acceptance_gate_results": "Preserve each frozen criterion, actual value and pass/fail independently of task completion.",
    "arc_path_audit_complete_score": "Complete independently checked provenance, reachability and backend-difference matrix.",
    "backend_dispositions": "GGUF and optional native-vLLM outcomes remain separate.",
    "solve_provenance": "Preserve the producer authority for any audited solve; never upgrade a development proxy.",
    "next_mechanism_condition": "A concrete observed blocker or completion; no invented tool demand.",
    "runner_receipt": "Current CPU audit counters only; historical producer execution lives in a sidecar.",
    "sidecar_receipts": "Hashes retain historical invocation facts and synthetic attacks outside current invocation fields.",
    "reducer_replay": "The saved session must reduce the same way under the currently shipped producer reducer.",
    "factory_reachability_receipt": "A live construction proves the current factory still reaches E3AgentPolicy without an adapter.",
    "game_source_access_receipt": "The audit records its input allowlist and never opens executable game source.",
    "world_model_efficacy_claim": "Remain null unless a useful prediction changed policy action and produced progress.",
    "independent_game_generalization_claimed": "Correlated frames from one session cannot establish cross-game generalization.",
    "local_versus_scored_differences": "Preserve every producer-recorded difference from the scored competition runtime.",
    "transport_validation_question_retired": "The completed ten-induction collection is not scheduled again.",
    "new_solve_claimed": "An audit of a known public session does not create a new solve.",
    "validation_receipts": "Retain exact validation commands, exit states, and unrelated baseline failures.",
}

atomic_write = base.atomic_write
artifact_checksum = base.artifact_checksum
gate_check = base.gate_check
gate_summary = base.gate_summary
is_quarantined = base.is_quarantined
sha256_file = base.sha256_file
unwrap_evidence_value = base.unwrap_evidence_value


def _progress(phase: int, event: str, **fields: Any) -> None:  # pragma: no cover - live output
    """Emit a truthful unbuffered phase record for long-running conductors."""

    print(json.dumps({"phase": phase, "event": event, **fields}, sort_keys=True), flush=True)


def _iso_now() -> str:  # pragma: no cover - wall clock receipt
    return datetime.now(UTC).isoformat(timespec="seconds")


def _canonical_sha256(value: Any) -> str:
    data = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    return "sha256:" + hashlib.sha256(data).hexdigest()


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def authenticate_upstream(path: Path) -> tuple[JsonDict | None, list[JsonDict]]:
    """Authenticate the exact Exp7234 path before reading any result field."""

    checks = [gate_check("exact_upstream_present", str(path), "path", True, path.is_file())]
    if not path.is_file():
        return None, checks
    try:
        value = _read_json(path)
    except (OSError, json.JSONDecodeError) as exc:
        checks.append(
            gate_check(
                "exact_upstream_readable",
                str(path),
                "json",
                "readable_mapping",
                type(exc).__name__,
                False,
            )
        )
        return None, checks
    if not isinstance(value, Mapping):
        checks.append(
            gate_check(
                "exact_upstream_readable",
                str(path),
                "json",
                "readable_mapping",
                type(value).__name__,
                False,
            )
        )
        return None, checks
    payload = dict(value)
    quarantined = is_quarantined(payload)
    checks.append(
        gate_check(
            "upstream_not_quarantined",
            str(path),
            "flagged_adversarial",
            False,
            quarantined,
        )
    )
    if quarantined:
        return None, checks
    checks.extend(
        [
            gate_check(
                "upstream_schema",
                str(path),
                "schema",
                producer.SCHEMA,
                unwrap_evidence_value(payload.get("schema")),
            ),
            gate_check(
                "upstream_experiment_identity",
                str(path),
                "experiment_id",
                producer.EXPERIMENT_ID,
                unwrap_evidence_value(payload.get("experiment_id")),
            ),
        ]
    )
    return (payload if all(row["passed"] for row in checks) else None), checks


def _recorded_hash(upstream: Mapping[str, Any], path: Path, root: Path) -> str | None:
    hashes = upstream.get("source_artifact_hashes")
    if not isinstance(hashes, Mapping):
        return None
    try:
        relative = path.relative_to(root).as_posix()
    except ValueError:
        relative = str(path)
    return hashes.get(relative) or hashes.get(str(path))


def _authenticate_path(
    root: Path,
    upstream: Mapping[str, Any],
    path: Path,
    *,
    expected_hash: str | None = None,
) -> tuple[str, JsonDict]:
    absolute = path if path.is_absolute() else root / path
    if not absolute.is_file():
        return "", gate_check(
            "evidence_path_and_hash", str(absolute), "sha256", expected_hash, "absent", False
        )
    actual = sha256_file(absolute)
    expected = expected_hash or _recorded_hash(upstream, absolute, root)
    return actual, gate_check(
        "evidence_path_and_hash", str(absolute), "sha256", expected, actual, expected == actual
    )


def _json_strings(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, Mapping):
        return [text for item in value.values() for text in _json_strings(item)]
    if isinstance(value, list):
        return [text for item in value for text in _json_strings(item)]
    return []


def _engine_calls(response: Mapping[str, Any]) -> list[str]:
    calls: list[str] = []
    for text in _json_strings(response):
        if "<tool_call" not in text or "run_engine_on_transitions" not in text:
            continue
        parsed, _seen, _unparsed = parse_xml_tool_calls(text)
        for call in parsed:
            function = call.get("function", {})
            if function.get("name") != "run_engine_on_transitions":
                continue
            arguments = function.get("arguments")
            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments)
                except json.JSONDecodeError:
                    arguments = {}
            if isinstance(arguments, Mapping) and isinstance(arguments.get("code"), str):
                calls.append(str(arguments["code"]))
    return calls


def _load_engine(path: Path) -> Engine:
    """Import one hash-checked engine without importing an environment package."""

    name = "arc_path_audit_" + hashlib.sha256(str(path).encode()).hexdigest()[:16]
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"engine_import_spec_unavailable:{path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    engine = getattr(module, "engine", None)
    if not callable(engine):
        raise TypeError("engine_not_callable")
    return engine


def _load_transitions(path: Path) -> tuple[list[Transition], list[int]]:
    transitions: list[Transition] = []
    indices: list[int] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        transitions.append(
            Transition(
                np.asarray(row["grid"], dtype=np.int64),
                int(row["action"]),
                deepcopy(row.get("data")),
                np.asarray(row["next_grid"], dtype=np.int64),
                int(row.get("level_before", 0)),
                int(row.get("level_after", 0)),
            )
        )
        indices.append(int(row["index"]))
    return transitions, indices


def _score_one(engine: Engine, transition: Transition) -> JsonDict:
    score = WorldModelVerifier([transition], hud_mask_enabled=False).score(engine)
    return {
        "exact_transition_correct": score.n == 1 and score.n_correct == 1,
        "value": float(score.accuracy),
        "change_fidelity": float(score.change_fidelity),
        "cell_recall": float(score.cell_recall),
        "engine_functionally_identity": bool(score.functionally_identity),
        "engine_identity_measurable": bool(score.identity_measurable),
        "engine_raise_count": int(score.n_engine_raised),
        "error": score.error,
    }


def score_future_controls(
    *,
    engine: Engine,
    transitions: Sequence[Transition],
    transition_indices: Sequence[int],
    engine_sha256: str,
    transition_source_sha256: str,
    session_hash: str,
    seed: int,
) -> list[JsonDict]:
    """Score the archived engine and two frozen controls on later session rows."""

    order = list(range(len(transitions)))
    random.Random(seed).shuffle(order)

    def noop(grid: np.ndarray, action: int, data: Any) -> np.ndarray:
        del action, data
        return grid.copy()

    rows: list[JsonDict] = []
    for position, (transition, transition_index) in enumerate(zip(transitions, transition_indices)):
        shuffled_source = transitions[order[position]]
        shuffled = Transition(
            transition.grid,
            shuffled_source.action,
            deepcopy(shuffled_source.data),
            transition.next_grid,
            transition.level_before,
            transition.level_after,
        )
        for arm, arm_engine, scored_transition in (
            ("recorded_engine", engine, transition),
            ("noop_control", noop, transition),
            ("shuffled_action_control", engine, shuffled),
        ):
            result = _score_one(arm_engine, scored_transition)
            rows.append(
                {
                    "row_type": "future_transition",
                    "unit_id": f"{session_hash}:{engine_sha256}:{transition_index}",
                    "cluster_id": session_hash,
                    "arm": arm,
                    "seed": seed,
                    "transition_index": int(transition_index),
                    "source_sha256": transition_source_sha256,
                    "engine_sha256": engine_sha256,
                    "useful_prediction": bool(
                        arm == "recorded_engine"
                        and result["exact_transition_correct"]
                        and not result["engine_functionally_identity"]
                    ),
                    "policy_used_engine": False,
                    "abstention": False,
                    **result,
                }
            )
    return rows


def evaluate_path_receipt(receipt: Mapping[str, Any], expected: Mapping[str, Any]) -> JsonDict:
    """Reject receipt shapes that confuse parsing, execution, and policy use."""

    reasons: list[str] = []
    if receipt.get("parsed_tool") is not True:
        reasons.append("tool_not_parsed")
    if not receipt.get("engine_sha256"):
        reasons.append("parsed_tool_without_engine")
    for field, reason in (
        ("session_hash", "stale_session_hash"),
        ("request_sha256", "request_hash_mismatch"),
        ("engine_sha256", "engine_hash_mismatch"),
        ("transition_sha256", "transition_hash_mismatch"),
        ("induction_action_index", "action_index_mismatch"),
    ):
        if receipt.get(field) != expected.get(field) and reason not in reasons:
            reasons.append(reason)
    if receipt.get("engine_functionally_identity") is True:
        reasons.append("identity_engine")
    if receipt.get("policy_used_engine") is not True:
        reasons.append("engine_never_consumed")
    return {"admitted": not reasons, "rejection_reasons": reasons}


def _producer_projection(row: Mapping[str, Any]) -> JsonDict:
    keys = (
        "backend",
        "disposition",
        "model_invoked",
        "transport_completed",
        "semantic_usable",
        "tool_dispatches",
        "engine_writes",
        "policy_engine_consumptions",
        "valid_engine_count",
        "non_identity_prediction_count",
        "held_future_prediction_accuracy",
        "trust_acceptance_count",
        "model_planned_actions",
        "actions",
        "levels",
        "progress",
        "error",
    )
    return {key: deepcopy(row.get(key)) for key in keys}


def _read_manifest(path: Path) -> list[JsonDict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _policy_used(action_rows: Sequence[Mapping[str, Any]], action_index: int) -> bool:
    return any(
        int(row.get("i", -1)) > action_index
        and row.get("top_branch") == "execute.plan_step"
        and row.get("plan_installed_by_attempt") is not None
        for row in action_rows
    )


def replay_saved_receipts(root: Path, upstream: Mapping[str, Any]) -> JsonDict:
    """Replay authenticated Exp7234 evidence through current independent reducers."""

    checks: list[JsonDict] = []
    source_hashes: JsonDict = {}
    authenticated: dict[Path, str] = {}
    for relative in (
        SESSION_PATH,
        MANIFEST_PATH,
        ACTION_ROWS_PATH,
        INDUCTION_ROWS_PATH,
        ACTION_PROVENANCE_PATH,
    ):
        actual, check = _authenticate_path(root, upstream, relative)
        checks.append(check)
        if check["passed"]:
            authenticated[relative] = actual
            source_hashes[relative.as_posix()] = actual
    if not all(row["passed"] for row in checks):
        return {
            "checks": checks,
            "source_hashes": source_hashes,
            "error": "required_saved_receipt_failed",
        }

    session = _read_json(root / SESSION_PATH)
    action_document = _read_json(root / ACTION_ROWS_PATH)
    action_rows = [dict(row) for row in action_document.get("rows", []) if isinstance(row, Mapping)]
    reduced = producer.reduce_backend_session(session)
    upstream_backend = next(
        (
            dict(row)
            for row in upstream.get("per_game_results", [])
            if isinstance(row, Mapping) and row.get("backend") == "llamacpp"
        ),
        {},
    )
    reducer_replay = {
        "current_reducer": "carnot.experiment_7234_v637_arc_scored_dryrun.reduce_backend_session",
        "observed": _producer_projection(reduced),
        "upstream": _producer_projection(upstream_backend),
    }
    reducer_replay["matches_upstream"] = reducer_replay["observed"] == reducer_replay["upstream"]

    game = str(upstream.get("selection_receipt", {}).get("selected_game") or "lp85")
    policy, factory = producer.build_disposable_submitted_policy(game, proposer=object())
    factory_receipt = {
        **{
            key: deepcopy(factory.get(key))
            for key in (
                "factory",
                "policy_class",
                "cascade",
                "adapter_disabled",
                "denied_inputs",
                "policy_inputs",
            )
        },
        "constructed_policy_class": type(policy).__name__,
        "reachable": type(policy).__name__ == "E3AgentPolicy",
    }

    session_hash = _canonical_sha256(
        {
            "upstream_sha256": sha256_file(root / UPSTREAM_PATH),
            "session_sha256": authenticated[SESSION_PATH],
            "action_rows_sha256": authenticated[ACTION_ROWS_PATH],
            "backend": reduced.get("backend"),
            "game": game,
            "seed": upstream.get("random_seed"),
        }
    )

    completions = [dict(row) for row in session.get("completions", []) if isinstance(row, Mapping)]
    code_calls: dict[str, JsonDict] = {}
    for completion in completions:
        request_path = Path(str(completion.get("request_path", "")))
        response_path = Path(str(completion.get("response_path", "")))
        request_actual, request_check = _authenticate_path(
            root,
            upstream,
            request_path,
            expected_hash=str(completion.get("request_sha256") or ""),
        )
        response_actual, response_check = _authenticate_path(
            root,
            upstream,
            response_path,
            expected_hash=str(completion.get("response_sha256") or ""),
        )
        checks.extend((request_check, response_check))
        if request_check["passed"]:
            source_hashes[str(request_path)] = request_actual
        if not response_check["passed"]:
            continue
        source_hashes[str(response_path)] = response_actual
        response = _read_json(response_path)
        if not isinstance(response, Mapping):
            continue
        for code in _engine_calls(response):
            code_hash = "sha256:" + hashlib.sha256(code.encode()).hexdigest()
            code_calls[code_hash] = {
                "request_sequence": completion.get("sequence"),
                "request_sha256": request_actual,
                "response_sha256": response_actual,
                "parsed_tool": True,
            }

    induction_action_indices = [
        int(row["i"]) for row in action_rows if row.get("induction_ran_this_action") is True
    ]
    induction_action_index = (
        induction_action_indices[0] if len(induction_action_indices) == 1 else -1
    )
    manifest_rows = _read_manifest(root / MANIFEST_PATH)
    archive_root = (root / MANIFEST_PATH).parents[2]
    snapshots: list[JsonDict] = []
    for manifest in manifest_rows:
        engine_path = archive_root / str(manifest["engine"]["path"])
        transition_path = archive_root / str(manifest["transitions"]["path"])
        engine_hash, engine_check = _authenticate_path(
            root, upstream, engine_path, expected_hash=str(manifest["engine"]["sha256"])
        )
        transition_hash, transition_check = _authenticate_path(
            root,
            upstream,
            transition_path,
            expected_hash=str(manifest["transitions"]["sha256"]),
        )
        checks.extend((engine_check, transition_check))
        if not engine_check["passed"] or not transition_check["passed"]:
            continue
        source_hashes[str(engine_path.relative_to(root))] = engine_hash
        source_hashes[str(transition_path.relative_to(root))] = transition_hash
        transitions, indices = _load_transitions(transition_path)
        engine = _load_engine(engine_path)
        snapshot_score = WorldModelVerifier(transitions, hud_mask_enabled=False).score(engine)
        snapshots.append(
            {
                "manifest": manifest,
                "engine_path": engine_path,
                "transition_path": transition_path,
                "engine_sha256": engine_hash,
                "transition_sha256": transition_hash,
                "transitions": transitions,
                "indices": indices,
                "engine": engine,
                "snapshot_score": snapshot_score,
            }
        )

    latest = max(snapshots, key=lambda row: max(row["indices"], default=-1), default=None)
    engine_receipts: list[JsonDict] = []
    audit_rows: list[JsonDict] = []
    for snapshot in snapshots:
        engine_hash = snapshot["engine_sha256"]
        request = code_calls.get(engine_hash, {})
        consumed = _policy_used(action_rows, induction_action_index)
        expected = {
            "session_hash": session_hash,
            "request_sha256": request.get("request_sha256"),
            "engine_sha256": engine_hash,
            "transition_sha256": snapshot["transition_sha256"],
            "induction_action_index": induction_action_index,
        }
        receipt = {
            **expected,
            **request,
            "run_id": snapshot["manifest"].get("run_id"),
            "engine_functionally_identity": bool(snapshot["snapshot_score"].functionally_identity),
            "policy_used_engine": consumed,
            "request_hash_matches": bool(request.get("request_sha256")),
            "engine_hash_matches": True,
            "transition_hash_matches": True,
        }
        receipt["admission"] = evaluate_path_receipt(receipt, expected)
        engine_receipts.append(receipt)
        audit_rows.append(
            {
                "row_type": "engine_action_join",
                "unit_id": str(snapshot["manifest"].get("run_id")),
                "cluster_id": session_hash,
                "arm": "recorded_engine",
                "seed": RANDOM_SEED,
                "transition_index": None,
                "source_sha256": snapshot["transition_sha256"],
                "engine_sha256": engine_hash,
                "request_sha256": request.get("request_sha256"),
                "induction_action_index": induction_action_index,
                "useful_prediction": False,
                "policy_used_engine": consumed,
                "value": int(receipt["admission"]["admitted"]),
                "error": None,
                "abstention": False,
            }
        )
        max_seen = max(snapshot["indices"], default=-1)
        future_positions = (
            [i for i, index in enumerate(latest["indices"]) if index > max_seen]
            if latest is not None
            else []
        )
        if future_positions:
            future = [latest["transitions"][i] for i in future_positions]
            future_indices = [latest["indices"][i] for i in future_positions]
            audit_rows.extend(
                score_future_controls(
                    engine=snapshot["engine"],
                    transitions=future,
                    transition_indices=future_indices,
                    engine_sha256=engine_hash,
                    transition_source_sha256=latest["transition_sha256"],
                    session_hash=session_hash,
                    seed=RANDOM_SEED,
                )
            )
        else:
            audit_rows.append(
                {
                    "row_type": "future_transition",
                    "unit_id": f"{session_hash}:{engine_hash}:censored",
                    "cluster_id": session_hash,
                    "arm": "recorded_engine",
                    "seed": RANDOM_SEED,
                    "transition_index": None,
                    "source_sha256": snapshot["transition_sha256"],
                    "engine_sha256": engine_hash,
                    "useful_prediction": False,
                    "policy_used_engine": consumed,
                    "value": None,
                    "error": "no_saved_held_future_transition",
                    "abstention": True,
                }
            )

    backend_dispositions = []
    for row in upstream.get("per_game_results", []):
        if not isinstance(row, Mapping):
            continue
        runner = row.get("runner_receipt") if isinstance(row.get("runner_receipt"), Mapping) else {}
        backend_dispositions.append(
            {
                "backend": row.get("backend"),
                "upstream_disposition": row.get("disposition"),
                "actual_backend_identity": runner.get("runner"),
                "model_invoked_by_producer": row.get("model_invoked"),
                "audit_disposition": (
                    "complete_replay"
                    if row.get("backend") == "llamacpp" and snapshots
                    else "complete_no_engine_rows"
                ),
                "error": row.get("error"),
            }
        )
    return {
        "checks": checks,
        "source_hashes": source_hashes,
        "session_hash": session_hash,
        "reducer_replay": reducer_replay,
        "factory_receipt": factory_receipt,
        "game_source_access_receipt": {
            "game_source_read": False,
            "adapter_used": False,
            "policy_input_allowlist": [
                "archived_public_frames",
                "archived_actions",
                "archived_own_transitions",
            ],
            "denied_inputs": deepcopy(factory.get("denied_inputs")),
        },
        "engine_receipts": engine_receipts,
        "audit_rows": audit_rows,
        "backend_dispositions": backend_dispositions,
    }


def _comparison_rows(audit_rows: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    return [
        {
            "unit_id": row.get("unit_id"),
            "arm": row.get("arm"),
            "seed": row.get("seed"),
            "metric": row.get("row_type"),
            "value": row.get("value"),
            "error": row.get("error"),
            "abstention": bool(row.get("abstention")),
        }
        for row in audit_rows
    ]


def build_terminal_artifact(
    *,
    run_date: str,
    duration_s: float,
    started_at_utc: str,
    ended_at_utc: str,
    checks: Sequence[Mapping[str, Any]],
    source_hashes: Mapping[str, Any],
    audit_rows: Sequence[Mapping[str, Any]],
    backend_dispositions: Sequence[Mapping[str, Any]],
    sidecars: Mapping[str, Any],
    reducer_replay: Mapping[str, Any],
    factory_receipt: Mapping[str, Any],
    local_scored_differences: Mapping[str, Any],
    sample_size: Mapping[str, Any],
    game_source_access_receipt: Mapping[str, Any] | None = None,
    phase_spans: Sequence[Mapping[str, Any]] = (),
    validation_receipts: Sequence[Mapping[str, Any]] = (),
) -> JsonDict:
    """Build a terminal result while keeping audit completion distinct from efficacy."""

    summary = gate_summary(checks)
    blocked = not summary["passed"]
    complete = not blocked
    useful_action = any(
        row.get("useful_prediction") is True and row.get("policy_used_engine") is True
        for row in audit_rows
    )
    progress = any(
        float(row.get("progress") or 0) > 0
        for row in backend_dispositions
        if isinstance(row, Mapping)
    )
    efficacy = (
        {"useful_policy_action": True, "progress": True} if useful_action and progress else None
    )
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "status": "blocked" if blocked else "complete",
        "run_date": run_date,
        "started_at_utc": started_at_utc,
        "ended_at_utc": ended_at_utc,
        "field_principles": {},
        "preconditions_checked": [deepcopy(dict(row)) for row in checks],
        "inference_substrate": "blocked_no_run" if blocked else "cpu_exact_solver_or_simulator",
        "inference_substrate_class": "blocked_no_run"
        if blocked
        else "cpu_exact_solver_or_simulator",
        "execution_venue": "host",
        "execution_host": socket.gethostname(),
        "duration_s": float(duration_s),
        "phase_spans": [deepcopy(dict(row)) for row in phase_spans],
        "MODEL_SPECS": [],
        "model_invoked": False,
        "source_artifact_hashes": deepcopy(dict(source_hashes)),
        "rows": _comparison_rows(audit_rows),
        "audit_rows": [deepcopy(dict(row)) for row in audit_rows],
        "sample_size_budget": deepcopy(dict(sample_size)),
        "random_seed": RANDOM_SEED,
        "reproducibility_checksum": "",
        "gate_check_summary": summary,
        "verifier_is_oracle": True,
        "verdict_class": "blocked" if blocked else "positive" if efficacy else "null",
        "honest_verdict": (
            f"blocked_{summary.get('failed_check') or 'required_preconditions'}"
            if blocked
            else "complete_positive_useful_consumed_engine_with_progress"
            if efficacy
            else "complete_null_no_useful_model_action_or_progress"
        ),
        "acceptance_gate_results": [
            {
                "criterion": "authentic_replay_complete",
                "actual_value": complete,
                "passed": complete,
            },
            {
                "criterion": "useful_prediction_consumed_by_policy",
                "actual_value": useful_action,
                "passed": useful_action,
            },
            {"criterion": "observed_progress", "actual_value": progress, "passed": progress},
        ],
        "arc_path_audit_complete_score": int(complete),
        "backend_dispositions": [deepcopy(dict(row)) for row in backend_dispositions],
        "solve_provenance": "live_agent_self_discovery",
        "next_mechanism_condition": (
            "Record the consumed engine SHA-256 on each policy-visible action and preserve later full-grid transitions; do not collect another ten inductions."
            if not useful_action
            else "Use a fresh session cluster before any broader efficacy claim."
        ),
        "runner_receipt": {
            "runner": "local_cpu_independent_receipt_reducer",
            "model_request_count": 0,
            "model_completion_count": 0,
            "model_error_count": 0,
            "engine_replay_executed": not blocked,
        },
        "sidecar_receipts": deepcopy(dict(sidecars)),
        "reducer_replay": deepcopy(dict(reducer_replay)),
        "factory_reachability_receipt": deepcopy(dict(factory_receipt)),
        "game_source_access_receipt": deepcopy(
            dict(game_source_access_receipt or {"game_source_read": False, "adapter_used": False})
        ),
        "world_model_efficacy_claim": efficacy,
        "independent_game_generalization_claimed": False,
        "local_versus_scored_differences": deepcopy(dict(local_scored_differences)),
        "transport_validation_question_retired": True,
        "new_solve_claimed": False,
        "validation_receipts": [deepcopy(dict(row)) for row in validation_receipts],
    }
    artifact["field_principles"] = {key: FIELD_PRINCIPLES[key] for key in artifact}
    artifact["reproducibility_checksum"] = artifact_checksum(artifact)
    return artifact


def validate_artifact(value: Mapping[str, Any] | Path) -> list[str]:
    """Cold-validate identity, no-LLM fields, row schema, and checksum."""

    if isinstance(value, Path):
        try:
            artifact = _read_json(value)
        except (OSError, json.JSONDecodeError) as exc:
            return [f"artifact_unreadable:{type(exc).__name__}"]
    else:
        artifact = dict(value)
    errors: list[str] = []
    if set(artifact.get("field_principles", {})) != set(artifact):
        errors.append("field_principles_must_cover_every_top_level_field")
    if (
        artifact.get("schema") != SCHEMA
        or artifact.get("experiment_id") != EXPERIMENT_ID
        or artifact.get("milestone") != MILESTONE
    ):
        errors.append("schema_or_experiment_identity_mismatch")
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
    honest = str(artifact.get("honest_verdict"))
    if artifact.get("status") == "complete" and not honest.startswith(("complete_", "complete:")):
        errors.append("complete_honest_verdict_prefix_invalid")
    if artifact.get("status") == "blocked" and not honest.startswith("blocked_"):
        errors.append("blocked_honest_verdict_prefix_invalid")
    runner = (
        artifact.get("runner_receipt")
        if isinstance(artifact.get("runner_receipt"), Mapping)
        else {}
    )
    if (
        artifact.get("MODEL_SPECS") != []
        or artifact.get("model_invoked") is not False
        or any(
            int(runner.get(key, 0) or 0) != 0
            for key in ("model_request_count", "model_completion_count", "model_error_count")
        )
    ):
        errors.append("no_llm_contract_violated")
    expected_substrate = (
        "blocked_no_run" if artifact.get("status") == "blocked" else "cpu_exact_solver_or_simulator"
    )
    if (
        artifact.get("inference_substrate") != expected_substrate
        or artifact.get("inference_substrate_class") != expected_substrate
    ):
        errors.append("inference_substrate_inconsistent")
    required_row_keys = {"unit_id", "arm", "seed", "metric", "value", "error", "abstention"}
    for index, row in enumerate(artifact.get("rows", [])):
        if not isinstance(row, Mapping) or not required_row_keys.issubset(row):
            errors.append(f"row_schema_invalid:{index}")
    if artifact.get("arc_path_audit_complete_score") != int(artifact.get("status") == "complete"):
        errors.append("arc_path_audit_complete_score_inconsistent")
    if artifact.get("reproducibility_checksum") != artifact_checksum(artifact):
        errors.append("reproducibility_checksum_mismatch")
    return errors


def _negative_receipts(engine_receipts: Sequence[Mapping[str, Any]], session_hash: str) -> JsonDict:
    if engine_receipts:
        source = dict(engine_receipts[0])
        expected = {
            key: source.get(key)
            for key in (
                "session_hash",
                "request_sha256",
                "engine_sha256",
                "transition_sha256",
                "induction_action_index",
            )
        }
    else:
        expected = {
            "session_hash": session_hash,
            "request_sha256": "unavailable",
            "engine_sha256": "unavailable",
            "transition_sha256": "unavailable",
            "induction_action_index": -1,
        }
    baseline = {
        **expected,
        "parsed_tool": True,
        "engine_functionally_identity": False,
        "policy_used_engine": True,
    }
    attacks = []
    for name, mutation in (
        ("parsed_tool_without_engine", {"engine_sha256": None}),
        ("identity_engine", {"engine_functionally_identity": True}),
        ("engine_never_consumed", {"policy_used_engine": False}),
        ("stale_session_hash", {"session_hash": "sha256:" + "0" * 64}),
    ):
        receipt = {**baseline, **mutation}
        attacks.append(
            {"attack": name, "receipt": receipt, "result": evaluate_path_receipt(receipt, expected)}
        )
    return {
        "schema": "carnot.experiment_7235.synthetic_negative_receipts.v1",
        "current_task_model_invoked": False,
        "expected": expected,
        "attacks": attacks,
    }


def _write_sidecars(
    raw_dir: Path, upstream: Mapping[str, Any], evidence: Mapping[str, Any]
) -> JsonDict:
    raw_dir.mkdir(parents=True, exist_ok=True)
    historical_path = raw_dir / "historical_source_receipts.json"
    negative_path = raw_dir / "synthetic_negative_receipts.json"
    historical = {
        "schema": "carnot.experiment_7235.historical_source_receipts.v1",
        "upstream_path": str(REPO_ROOT / UPSTREAM_PATH),
        "upstream_sha256": sha256_file(REPO_ROOT / UPSTREAM_PATH),
        "producer_model_specs": deepcopy(upstream.get("MODEL_SPECS", [])),
        "producer_model_invoked": upstream.get("model_invoked"),
        "producer_runner_receipt": deepcopy(upstream.get("runner_receipt", {})),
        "engine_receipts": deepcopy(evidence.get("engine_receipts", [])),
    }
    negative = _negative_receipts(
        evidence.get("engine_receipts", []), str(evidence.get("session_hash", "unavailable"))
    )
    atomic_write(historical_path, historical)
    atomic_write(negative_path, negative)
    return {
        "historical_source_receipts": {
            "path": str(historical_path),
            "sha256": sha256_file(historical_path),
        },
        "synthetic_negative_receipts": {
            "path": str(negative_path),
            "sha256": sha256_file(negative_path),
        },
    }


def run_experiment(args: argparse.Namespace) -> JsonDict:
    """Authenticate, replay, score controls, validate, and atomically publish."""

    started = time.monotonic()
    started_utc = _iso_now()
    root = REPO_ROOT
    result_path = args.result_path if args.result_path.is_absolute() else root / args.result_path
    raw_dir = args.raw_dir if args.raw_dir.is_absolute() else root / args.raw_dir
    checkpoint = (
        args.checkpoint_path if args.checkpoint_path.is_absolute() else root / args.checkpoint_path
    )
    phase_spans: list[JsonDict] = []
    source_hashes: JsonDict = {}
    checks: list[JsonDict] = []

    _progress(0, "phase_start", name="preconditions")
    phase = time.monotonic()
    for relative in REQUIRED_SOURCE_PATHS:
        path = root / relative
        checks.append(
            gate_check("required_source", str(path), relative.as_posix(), True, path.is_file())
        )
        if path.is_file():
            source_hashes[relative.as_posix()] = sha256_file(path)
    raw_dir.mkdir(parents=True, exist_ok=True)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    checks.extend(
        [
            gate_check(
                "raw_path_writable", str(raw_dir), "writable", True, os.access(raw_dir, os.W_OK)
            ),
            gate_check(
                "checkpoint_path_writable",
                str(checkpoint.parent),
                "writable",
                True,
                os.access(checkpoint.parent, os.W_OK),
            ),
            gate_check(
                "current_reducer_import",
                producer.__name__,
                "reduce_backend_session",
                True,
                callable(producer.reduce_backend_session),
            ),
            gate_check(
                "current_verifier_import",
                WorldModelVerifier.__module__,
                "WorldModelVerifier",
                True,
                callable(WorldModelVerifier),
            ),
        ]
    )
    upstream, upstream_checks = authenticate_upstream(root / UPSTREAM_PATH)
    checks.extend(upstream_checks)
    if (root / UPSTREAM_PATH).is_file():
        source_hashes[UPSTREAM_PATH.as_posix()] = sha256_file(root / UPSTREAM_PATH)
    phase_spans.append({"phase": "preconditions", "duration_s": time.monotonic() - phase})
    atomic_write(checkpoint, {"status": "running", "phase": "preconditions", "checks": checks})
    _progress(0, "phase_end", name="preconditions", passed=upstream is not None)

    evidence: JsonDict = {}
    sidecars: JsonDict = {}
    if upstream is not None and all(row["passed"] for row in checks):
        _progress(1, "phase_start", name="receipt_replay")
        phase = time.monotonic()
        _progress(1, "benchmark_start", operation="current_reducer_and_engine_replay")
        evidence = replay_saved_receipts(root, upstream)
        _progress(
            1,
            "benchmark_end",
            operation="current_reducer_and_engine_replay",
            rows=len(evidence.get("audit_rows", [])),
        )
        checks.extend(evidence.get("checks", []))
        checks.append(
            gate_check(
                "current_reducer_matches_upstream",
                str(SESSION_PATH),
                "reduced_projection",
                True,
                evidence.get("reducer_replay", {}).get("matches_upstream"),
            )
        )
        checks.append(
            gate_check(
                "current_factory_reachable",
                "make_carnot_agent",
                "E3AgentPolicy",
                True,
                evidence.get("factory_receipt", {}).get("reachable"),
            )
        )
        source_hashes.update(evidence.get("source_hashes", {}))
        phase_spans.append({"phase": "receipt_replay", "duration_s": time.monotonic() - phase})
        _progress(1, "phase_end", name="receipt_replay")

        _progress(2, "phase_start", name="sidecar_write")
        sidecars = _write_sidecars(raw_dir, upstream, evidence)
        for receipt in sidecars.values():
            source_hashes[str(receipt["path"])] = receipt["sha256"]
        _progress(2, "phase_end", name="sidecar_write")

    audit_rows = evidence.get("audit_rows", [])
    engine_receipts = evidence.get("engine_receipts", [])
    completed = len(engine_receipts)
    censored = sum(
        row.get("abstention") is True and row.get("arm") == "recorded_engine" for row in audit_rows
    )
    sample_size = {
        "planned_independent_units": 1,
        "attempted_units": int(bool(completed)),
        "completed_units": int(bool(completed)),
        "censored_units": int(not completed),
        "cluster_count": 1 if completed else 0,
        "archived_engine_count": completed,
        "censored_engine_count": censored,
        "cluster_definition": "one Exp7234 backend game session; frames are correlated",
        "stopping_rule": "audit every authenticated archived engine once; do not collect more inductions",
    }
    validation_path = checkpoint.parent / "validation_receipts.json"
    validation_receipts = _read_json(validation_path) if validation_path.is_file() else []
    if validation_path.is_file():
        source_hashes[str(validation_path)] = sha256_file(validation_path)
    _progress(3, "phase_start", name="final_validation_and_atomic_write")
    artifact = build_terminal_artifact(
        run_date=args.date,
        duration_s=time.monotonic() - started,
        started_at_utc=started_utc,
        ended_at_utc=_iso_now(),
        checks=checks,
        source_hashes=source_hashes,
        audit_rows=audit_rows,
        backend_dispositions=evidence.get("backend_dispositions", []),
        sidecars=sidecars,
        reducer_replay=evidence.get("reducer_replay", {}),
        factory_receipt=evidence.get("factory_receipt", {}),
        game_source_access_receipt=evidence.get("game_source_access_receipt"),
        local_scored_differences=(upstream or {}).get("submission_configuration_diff", {}),
        sample_size=sample_size,
        phase_spans=phase_spans,
        validation_receipts=validation_receipts,
    )
    _progress(3, "validation_start", operation="cold_artifact_validation")
    errors = validate_artifact(artifact)
    _progress(3, "validation_end", operation="cold_artifact_validation", errors=errors)
    if errors:
        raise ValueError("terminal_artifact_invalid:" + ",".join(errors))
    _progress(3, "artifact_write_start", path=str(result_path))
    atomic_write(result_path, artifact)
    _progress(3, "artifact_write_end", path=str(result_path))
    _progress(3, "phase_end", name="final_validation_and_atomic_write")
    return artifact


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default=RUN_DATE)
    parser.add_argument("--result-path", type=Path, default=RESULT_PATH)
    parser.add_argument("--raw-dir", type=Path, default=RAW_DIR)
    parser.add_argument("--checkpoint-path", type=Path, default=CHECKPOINT_PATH)
    parser.add_argument("--validate", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    _progress(0, "phase_start", name="entrypoint")
    args = parse_args(argv)
    if args.validate is not None:
        return int(bool(validate_artifact(args.validate)))
    artifact = run_experiment(args)
    print(
        json.dumps(
            {
                "artifact": str(args.result_path),
                "status": artifact["status"],
                "verdict": artifact["honest_verdict"],
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - thin module execution
    raise SystemExit(main())
