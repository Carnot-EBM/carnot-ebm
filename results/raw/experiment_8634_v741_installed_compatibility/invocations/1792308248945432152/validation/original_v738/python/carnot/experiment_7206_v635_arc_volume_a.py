"""Run one V635 ARC selfparse session and merge unique induction evidence.

The expensive CUDA, llama.cpp, lease, heartbeat, and ARC environment machinery
is reused from Experiment 7193. This module owns the V635 source gates, the
per-name receipt repair, cumulative deduplication, and terminal artifact.

Spec refs: REQ-ARC-WMTE-7206 and SCENARIO-ARC-WMTE-7206-*.
"""

from __future__ import annotations

import argparse
import ast
from collections import Counter
from collections.abc import Mapping, Sequence
from copy import deepcopy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import socket
import sys
import tempfile
import time
from typing import Any, Callable

import yaml

from carnot.agentic.arc_induction_tools import parse_xml_tool_calls
from carnot import experiment_7193_v634_arc_direct_tool as reused


JsonDict = dict[str, Any]
REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_ROOT = REPO_ROOT / "scripts"

TASK_ID = "exp7206-arc-volume-a"
EXPERIMENT_ID = 7206
MILESTONE = "2026.09.635"
RUN_DATE = "20260911"
GAME = "r11l"
RANDOM_SEED = 7_206_001
ACTION_BUDGET = 4000
SESSION_TIMEOUT_S = 3600
INDUCTION_TIMEOUT_S = 2400
HARD_CAP_S = 4800
N_CTX = 49152
COMPLETION_BUDGET = 4096
EVIDENCE_TARGET = 10
MODEL_ID = "unsloth/Qwen3.8-27B-GGUF"
QUANTIZATION = "Q4_K_M"
MODEL_SPECS = [{"hf_id": MODEL_ID, "quantization": QUANTIZATION}]
EXPECTED_PRIOR_VERDICT = "blocked_required_source_bytes"

SCHEMA = "carnot.experiment_7206.arc_volume_a.v1"
DRIVING_REQUIREMENT = "REQ-ARC-WMTE-7206"
SPEC_PATH = Path("openspec/capabilities/arc-world-model-trust-energy/spec.md")
ROADMAP_PATH = Path("research-roadmap.yaml")
EXCLUSION_PATH = Path("ops/exclusion_manifest.yaml")
REGISTRY_PATH = Path("ops/arc_solve_registry.yaml")
EVAL_PATH = Path("scripts/arc_leaderboard_eval.py")
POLICY_PATH = Path("python/carnot/agentic/arc_competition_agent.py")
MODULE_PATH = Path("python/carnot/experiment_7206_v635_arc_volume_a.py")
WRAPPER_PATH = Path("scripts/experiments/experiment_7206_v635_arc_volume_a.py")
TEST_PATH = Path("tests/python/test_experiment_7206_v635_arc_volume_a.py")
RESULT_PATH = Path("results/experiment_7206_v635_arc_volume_a.json")
CHECKPOINT_PATH = Path("results/checkpoints/experiment_7206_v635_arc_volume_a/running.json")
CHECKPOINT_SCHEMA = "carnot.experiment_7206.checkpoint.v1"
RAW_DIR = Path("results/raw/experiment_7206")

PRIOR_ARTIFACT_PATH = Path("results/experiment_7186_v633_arc_withheld_transfer.json")
EXP7193_PATH = Path("results/experiment_7193_v634_arc_direct_tool.json")
EXP7194_PATH = Path("results/experiment_7194_v634_arc_gap_audit.json")
EXP7193_SESSION_PATH = Path("results/raw/experiment_7193/session_receipt.json")
EXP7193_RUN_ROW_PATH = Path("results/raw/experiment_7193/run_game_row.json")
EXP7193_COMPLETION_PATH = Path("results/raw/experiment_7193/completion_manifest.json")
HISTORICAL_PATH = Path("results/arc_leaderboard_eval_runs/r11l-1594772.json")
SIBLING_PATH = Path("results/experiment_7207_v635_arc_volume_b.json")
SIBLING_TASK_ID = "exp7207-arc-volume-b"
# REQ-ARC-WMTE-6642: declare every eval-run field consumed by the historical
# authentication and projection path. The producer/consumer lint verifies these
# names against the shipped evaluator surface even when no local corpus exists.
EVAL_RUN_FIELDS_READ = (
    "complete",
    "flagged_adversarial",
    "random_seed",
    "per_game",
    "game",
    "policy_diagnostics",
    "induction_attempts",
    "tool_gap",
    "tool_calls_total",
    "terminated_by",
    "tool_gap_events",
    "tool_gap_events_dropped",
    "started_at",
    "reason",
    "wall_s",
    "skipped",
    "refinement_rounds",
    "counterexamples",
    "kind",
    "engine_identity_measurable",
    "engine_functionally_identity",
    "goal_predicate_satisfiable",
    "selected_candidate_name",
)

EXPECTED_TASK_CONTRACT = {
    "id": TASK_ID,
    "milestone": MILESTONE,
    "deliverable": RESULT_PATH.as_posix(),
    "gated_on": None,
    "prior_failures": [
        {
            "experiment_id": "exp7186-arc-withheld-transfer",
            "verdict": EXPECTED_PRIOR_VERDICT,
            "addressed_by": (
                "Use the shipped direct run_game path and the successful Exp7193 runtime; "
                "no invented runner is required."
            ),
            "retire_if_same_verdict": True,
        }
    ],
    "operator_override": (
        "2026-09-11 operator directive in ops/known-issues.md: accumulate separate selfparse "
        "sessions after Exp7193; each new seed contributes unique induction evidence under "
        "its own wall cap."
    ),
}

REQUIRED_SOURCE_PATHS = (
    Path("AGENTS.md"),
    Path("CLAUDE.md"),
    Path("CODEX.md"),
    Path("research-program.md"),
    Path("research-references.md"),
    ROADMAP_PATH,
    EXCLUSION_PATH,
    Path("ops/e2e-test-plan.md"),
    Path("ops/known-issues.md"),
    REGISTRY_PATH,
    EVAL_PATH,
    POLICY_PATH,
    Path("python/carnot/agentic/arc_executable_world_model.py"),
    Path("python/carnot/agentic/arc_induction_tool_loop.py"),
    Path("python/carnot/agentic/arc_tool_gap_receipt.py"),
    Path("python/carnot/agentic/arc_eval_provenance.py"),
    Path("python/carnot/experiment_7193_v634_arc_direct_tool.py"),
    EXP7193_PATH,
    EXP7194_PATH,
    EXP7193_SESSION_PATH,
    EXP7193_RUN_ROW_PATH,
    EXP7193_COMPLETION_PATH,
    HISTORICAL_PATH,
    Path("python/carnot/inference/sota_models.py"),
    Path("python/carnot/inference/llama_server_supervisor.py"),
    Path("python/carnot/gpu_lease_phase_journal.py"),
    SPEC_PATH,
    PRIOR_ARTIFACT_PATH,
    MODULE_PATH,
    WRAPPER_PATH,
    TEST_PATH,
)

FIELD_PRINCIPLES: JsonDict = {
    "field_principles": "Echo the reason for each field beside its actual evidence.",
    "status": "Write a terminal artifact only after completion or a diagnosed external block.",
    "run_date": "Use 20260911; do not substitute an upstream experiment date.",
    "preconditions_checked": "Record the actual resource, code and gate observations.",
    "inference_substrate": "Describe executed computation, not the intended workload.",
    "inference_substrate_class": "The actual operation determines its duration floor.",
    "execution_venue": (
        "Use exactly host, kv260, gatemate or polarfire; these tasks execute on host."
    ),
    "execution_host": "Put the actual hostname here, never inside execution_venue.",
    "duration_s": "Measure monotonic work time; do not pad it to pass a floor.",
    "source_artifact_hashes": "Bind source code, input data and frozen contracts to the claim.",
    "rows": (
        "Keep unit_id, arm, seed, metric, error and abstention for each comparison; "
        "do not replace numeric rows with a task roster."
    ),
    "sample_size_budget": (
        "Retain planned, attempted, completed, censored and independent-unit counts."
    ),
    "random_seed": "Freeze stochastic choices before held-out outcomes are read.",
    "reproducibility_checksum": "Hash the inputs, code, settings and raw rows.",
    "gate_check_summary": (
        "Every blocked_* verdict names failed check, upstream, field, expected and observed value."
    ),
    "verifier_is_oracle": (
        "Same correctness authority remains circular even with a separate implementation."
    ),
    "verdict_class": (
        "Use exactly positive | circular_positive | null | blocked | disqualified | partial; "
        "only incomplete own work can be partial."
    ),
    "honest_verdict": (
        "Use complete_ or complete: for completed findings; blocked_* for external absence. "
        "Readiness is not scientific value."
    ),
    "arc_session_complete_score": (
        "A terminal scheduled session can honestly contain zero useful inductions."
    ),
    "solve_provenance": "Use live_agent_self_discovery for the reachable policy path.",
    "per_game_results": "Keep the full session denominator, banked levels and action count.",
    "tool_induction_rows": "Unique IDs and per-name events prevent duplicated or lost tool evidence.",
    "cumulative_induction_rows": (
        "Only authenticated distinct inductions count toward the cumulative target."
    ),
    "adapter_isolation_receipt": (
        "A withheld-adapter claim requires an observed policy access boundary."
    ),
    "new_solve_claimed": "False because r11l has already been reproduced.",
    "paired_efficacy_reported": (
        "False because the two new sessions are replication units, not causal arms."
    ),
    "MODEL_SPECS": "Include the mandated Qwen3.8 GGUF for every LLM call.",
    "model_invoked": "Record whether generation actually occurred.",
    "phase_spans": "Separate loading, prefill, generation, parsing and verification costs.",
    "gpu_receipts": "Task-owned CUDA evidence must overlap actual model work.",
    "model_identity_receipt": ("Record repository ID, revision, GGUF hash and actual runtime."),
    "runner_receipt": "Model count and execution runner explain compute allocation.",
    "schema": "A versioned schema lets cumulative consumers reject incompatible rows.",
    "experiment_id": "Bind the terminal receipt to the V635 task identity.",
    "inference_mode": "Use live_gpu only when task-owned CUDA work was sampled.",
    "model_specs": "Keep the actual resolved model beside the mandated model declaration.",
    "completion_receipts": "Raw completion hashes preserve every actual model response.",
    "session_receipt": "The bounded child terminal state distinguishes zero evidence from no run.",
    "registry_reproduction_receipt": "The evaluator confirms r11l was reproduced before launch.",
    "optional_source_receipts": "Missing sibling output is visible and nonblocking.",
    "demand_observation": "Report observed demand and limitations over the exact denominator.",
    "official_leaderboard_score_reported": "This cumulative session is not an official submission.",
    "registered_level_increment": "A repeated known game cannot increment the solve registry.",
    "submitted": "The experiment must not alter or submit the live default.",
}
REQUIRED_ARTIFACT_FIELDS = frozenset(
    {
        "field_principles",
        "status",
        "run_date",
        "preconditions_checked",
        "inference_substrate",
        "inference_substrate_class",
        "execution_venue",
        "execution_host",
        "duration_s",
        "source_artifact_hashes",
        "rows",
        "sample_size_budget",
        "random_seed",
        "reproducibility_checksum",
        "gate_check_summary",
        "verifier_is_oracle",
        "verdict_class",
        "honest_verdict",
        "arc_session_complete_score",
        "solve_provenance",
        "per_game_results",
        "tool_induction_rows",
        "cumulative_induction_rows",
        "adapter_isolation_receipt",
        "new_solve_claimed",
        "paired_efficacy_reported",
        "MODEL_SPECS",
        "model_invoked",
        "phase_spans",
        "gpu_receipts",
        "model_identity_receipt",
        "runner_receipt",
    }
)
HASH_RE = re.compile(r"sha256:[0-9a-f]{64}")
DURATION_FLOORS = {
    "model_load_no_generation": 2.0,
    "model_bounded_generation": 10.0,
    "model_full_generation": 60.0,
}
VERDICT_CLASSES = {"positive", "circular_positive", "null", "blocked", "disqualified", "partial"}


def canonical_json(value: Any) -> str:
    """Serialize a receipt in one stable form."""

    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, default=str)


def sha256_bytes(value: bytes) -> str:
    """Return a labeled SHA-256 digest."""

    return "sha256:" + hashlib.sha256(value).hexdigest()


def sha256_file(path: str | Path) -> str:
    """Hash one regular file without loading large artifacts into memory."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def atomic_write(path: str | Path, payload: Mapping[str, Any]) -> None:
    """Publish a complete JSON document with one atomic replacement."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    data = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
    with tempfile.NamedTemporaryFile(dir=target.parent, delete=False) as handle:
        handle.write(data)
        temporary = Path(handle.name)
    os.replace(temporary, target)


def artifact_checksum(artifact: Mapping[str, Any]) -> str:
    """Bind every terminal field except the checksum itself."""

    body = {key: value for key, value in artifact.items() if key != "reproducibility_checksum"}
    return sha256_bytes(canonical_json(body).encode())


def gate_check(
    check: str,
    upstream: str,
    field: str,
    expected: Any,
    observed: Any,
    passed: bool | None = None,
    **extra: Any,
) -> JsonDict:
    """Keep a gate's expected and observed evidence beside its decision."""

    row = {
        "check": str(check),
        "upstream": str(upstream),
        "field": str(field),
        "expected_value": deepcopy(expected),
        "observed_value": deepcopy(observed),
        "passed": bool(expected == observed) if passed is None else bool(passed),
    }
    row.update(deepcopy(extra))
    return row


def gate_summary(checks: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Retain all gates and expose the first failure as the terminal diagnosis."""

    rows = [deepcopy(dict(row)) for row in checks]
    failed = next((row for row in rows if row.get("passed") is not True), None)
    return {
        "passed": failed is None,
        "failed_check": None if failed is None else failed.get("check"),
        "upstream": None if failed is None else failed.get("upstream"),
        "field": None if failed is None else failed.get("field"),
        "expected_value": None if failed is None else failed.get("expected_value"),
        "observed_value": None if failed is None else failed.get("observed_value"),
        "checks": rows,
    }


def unwrap_evidence_value(value: Any) -> Any:
    """Unwrap only an explicit two-key principle/value evidence wrapper."""

    if (
        isinstance(value, Mapping)
        and set(value) == {"principle", "value"}
        and isinstance(value.get("principle"), str)
    ):
        return value.get("value")
    return value


def is_quarantined(payload: Mapping[str, Any]) -> bool:
    """Use the repository authority, with a strict local fallback."""

    try:
        if str(SCRIPTS_ROOT) not in sys.path:
            sys.path.insert(0, str(SCRIPTS_ROOT))
        from conductor_gates import _is_quarantined

        return bool(_is_quarantined(dict(payload)))
    except Exception:
        return unwrap_evidence_value(payload.get("flagged_adversarial")) is True


def authenticated_field_gate(
    payload: Mapping[str, Any], *, upstream: str, field: str, expected: Any
) -> JsonDict:
    """Reject quarantine before reading or unwrapping a producer field."""

    quarantined = is_quarantined(payload)
    if quarantined:
        observed = {"value": "not_consumed", "quarantined": True, "consumed": False}
    else:
        observed = {
            "value": unwrap_evidence_value(payload.get(field)),
            "quarantined": False,
            "consumed": True,
        }
    return gate_check(
        "upstream_authenticated_field",
        upstream,
        field,
        {"value": expected, "quarantined": False, "consumed": True},
        observed,
        not quarantined and observed["value"] == expected,
    )


def configure_reused_driver() -> Any:
    """Give the shipped Exp7193 runtime this experiment's immutable identity."""

    reused.TASK_ID = TASK_ID
    reused.MILESTONE = MILESTONE
    reused.RUN_DATE = RUN_DATE
    reused.GAME = GAME
    reused.RANDOM_SEED = RANDOM_SEED
    reused.ACTION_BUDGET = ACTION_BUDGET
    reused.SESSION_TIMEOUT_S = SESSION_TIMEOUT_S
    reused.INDUCTION_TIMEOUT_S = INDUCTION_TIMEOUT_S
    reused.N_CTX = N_CTX
    reused.COMPLETION_BUDGET = COMPLETION_BUDGET
    reused.RESULT_PATH = RESULT_PATH
    reused.CHECKPOINT_PATH = CHECKPOINT_PATH
    reused.RAW_DIR = RAW_DIR
    reused.WRAPPER_PATH = WRAPPER_PATH
    return reused


def _resolved_path(root: Path, raw: Any) -> Path:
    path = Path(str(raw))
    return path if path.is_absolute() else root / path


def terminal_tool_event_receipt(
    *,
    root: Path,
    completions: Sequence[Mapping[str, Any]],
    attempt_index: int,
    recorded_total: int | None,
) -> JsonDict:
    """Derive a terminal loop's total and name map from one ordered event suffix.

    The policy receipt's recorded total is used only to delimit the final loop
    when a policy attempt performed multiple refinement loops. Both published
    aggregates are then derived from the selected strict-parser events.
    """

    if recorded_total is not None and (
        not isinstance(recorded_total, int)
        or isinstance(recorded_total, bool)
        or recorded_total < 0
    ):
        return {
            "aggregation_consistent": False,
            "recorded_tool_calls_total": recorded_total,
            "parsed_event_stream_total": 0,
            "terminal_suffix_rule": "recorded_terminal_dispatch_total",
            "tool_calls_total": None,
            "tool_calls_by_name": {},
            "tool_call_events": [],
            "parser_blocks_seen": 0,
            "parser_blocks_unparsed": 0,
            "error": "recorded_total_invalid",
        }
    events: list[JsonDict] = []
    blocks_seen = 0
    blocks_unparsed = 0
    matching = sorted(
        (
            dict(row)
            for row in completions
            if row.get("stage") == "environment"
            and row.get("induction_attempt_index") == attempt_index
        ),
        key=lambda row: int(row.get("index", 0) or 0),
    )
    for completion in matching:
        path = _resolved_path(root, completion.get("content_path"))
        raw = path.read_bytes()
        observed_hash = sha256_bytes(raw)
        expected_hash = completion.get("content_sha256")
        if expected_hash and observed_hash != expected_hash:
            raise ValueError(f"completion_hash_mismatch:{path}")
        parsed, seen, unparsed = parse_xml_tool_calls(raw.decode("utf-8", "strict"))
        blocks_seen += int(seen)
        blocks_unparsed += int(unparsed)
        for call_index, call in enumerate(parsed):
            function = call.get("function") if isinstance(call, Mapping) else None
            function = function if isinstance(function, Mapping) else {}
            name = str(function.get("name") or "")
            arguments = function.get("arguments")
            events.append(
                {
                    "completion_id": completion.get("completion_id"),
                    "completion_sha256": observed_hash,
                    "call_index": call_index,
                    "tool_name": name,
                    "arguments_sha256": sha256_bytes(str(arguments or "").encode()),
                }
            )
    if recorded_total is None and (blocks_unparsed or not events):
        return {
            "aggregation_consistent": False,
            "recorded_tool_calls_total": None,
            "parsed_event_stream_total": len(events),
            "terminal_suffix_rule": "complete_attempt_event_stream",
            "tool_calls_total": None,
            "tool_calls_by_name": {},
            "tool_call_events": [],
            "parser_blocks_seen": blocks_seen,
            "parser_blocks_unparsed": blocks_unparsed,
            "error": (
                "attempt_event_stream_contains_unparsed_blocks"
                if blocks_unparsed
                else "attempt_event_stream_contains_no_tool_calls"
            ),
        }
    if recorded_total is not None and len(events) < recorded_total:
        return {
            "aggregation_consistent": False,
            "recorded_tool_calls_total": recorded_total,
            "parsed_event_stream_total": len(events),
            "terminal_suffix_rule": "recorded_terminal_dispatch_total",
            "tool_calls_total": None,
            "tool_calls_by_name": {},
            "tool_call_events": [],
            "parser_blocks_seen": blocks_seen,
            "parser_blocks_unparsed": blocks_unparsed,
            "error": "recorded_total_exceeds_parsed_event_stream",
        }
    selected = (
        events if recorded_total is None else events[-recorded_total:] if recorded_total else []
    )
    counts = Counter(str(row["tool_name"]) for row in selected)
    return {
        "aggregation_consistent": True,
        "recorded_tool_calls_total": recorded_total,
        "parsed_event_stream_total": len(events),
        "terminal_suffix_rule": (
            "complete_attempt_event_stream"
            if recorded_total is None
            else "recorded_terminal_dispatch_total"
        ),
        "tool_calls_total": len(selected),
        "tool_calls_by_name": dict(sorted(counts.items())),
        "tool_call_events": selected,
        "parser_blocks_seen": blocks_seen,
        "parser_blocks_unparsed": blocks_unparsed,
        "error": None,
    }


def _returned_tool_loop_without_gap(attempt: Mapping[str, Any]) -> str | None:
    """Recover a returned loop marker when the policy omitted its gap attachment.

    Some level-up reinductions return through model validation before the policy
    attaches ``tool_gap``. Their refinement receipt still records proposer success
    and the loop's terminal reason. Counts remain grounded in completion bytes.
    """

    rounds = attempt.get("refinement_rounds")
    for raw in reversed(rounds if isinstance(rounds, list) else []):
        if not isinstance(raw, Mapping) or raw.get("proposer_ok") is not True:
            continue
        message = str(raw.get("message") or "")
        match = re.search(r"\btool loop:\s*([^,()]+)", message)
        if match:
            return match.group(1).strip()
    return None


def _induction_id(attempt: Mapping[str, Any], index: int, seed: int) -> str:
    identity = {
        "game": GAME,
        "seed": seed,
        "attempt_index": index,
        "started_at": attempt.get("started_at"),
        "reason": attempt.get("reason"),
    }
    return sha256_bytes(canonical_json(identity).encode())


def _model_validity_errors(attempt: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    if attempt.get("skipped"):
        errors.append(str(attempt["skipped"]))
    rounds = attempt.get("refinement_rounds")
    for row in rounds if isinstance(rounds, list) else []:
        if isinstance(row, Mapping) and row.get("skipped"):
            errors.append(str(row["skipped"]))
    counterexamples = attempt.get("counterexamples")
    for row in counterexamples if isinstance(counterexamples, list) else []:
        if isinstance(row, Mapping) and row.get("kind"):
            errors.append(str(row["kind"]))
    return list(dict.fromkeys(errors))


def project_tool_inductions(
    root: Path,
    run_row: Mapping[str, Any] | None,
    completions: Sequence[Mapping[str, Any]],
    *,
    seed: int | None = None,
    session_id: str | None = None,
    source_hash: str | None = None,
    upstream_rows: Sequence[Mapping[str, Any]] = (),
) -> list[JsonDict]:
    """Project every gap-capable policy attempt with same-stream call counts."""

    seed = RANDOM_SEED if seed is None else seed
    session_id = TASK_ID if session_id is None else session_id
    diagnostics = run_row.get("policy_diagnostics", {}) if isinstance(run_row, Mapping) else {}
    attempts = diagnostics.get("induction_attempts", []) if isinstance(diagnostics, Mapping) else []
    attempts = attempts if isinstance(attempts, list) else []
    upstream_by_index = {
        int(row["attempt_index"]): row
        for row in upstream_rows
        if isinstance(row, Mapping) and isinstance(row.get("attempt_index"), int)
    }
    rows: list[JsonDict] = []
    for index, raw_attempt in enumerate(attempts):
        if not isinstance(raw_attempt, Mapping):
            continue
        attempt = dict(raw_attempt)
        raw_gap = attempt.get("tool_gap")
        policy_attached = isinstance(raw_gap, Mapping)
        inferred_outcome = None
        if not policy_attached:
            inferred_outcome = _returned_tool_loop_without_gap(attempt)
        if not policy_attached and inferred_outcome is None:
            continue
        gap = raw_gap if policy_attached else {}
        recorded_total = gap.get("tool_calls_total", 0) if policy_attached else None
        receipt = terminal_tool_event_receipt(
            root=root,
            completions=completions,
            attempt_index=index,
            recorded_total=(
                int(recorded_total)
                if isinstance(recorded_total, int) and not isinstance(recorded_total, bool)
                else -1
                if recorded_total is not None
                else None
            ),
        )
        terminal_outcome = str(gap.get("terminated_by") or inferred_outcome or "")
        terminal = bool(terminal_outcome)
        calls = receipt.get("tool_calls_total")
        engaged = bool(receipt.get("aggregation_consistent") and calls and terminal)
        matching = [
            dict(row)
            for row in completions
            if row.get("stage") == "environment" and row.get("induction_attempt_index") == index
        ]
        upstream = upstream_by_index.get(index, {})
        rows.append(
            {
                "unit_id": f"{GAME}:{seed}:induction:{index}",
                "source_session_id": session_id,
                "source_artifact_hash": source_hash,
                "source_authenticated": True,
                "game": GAME,
                "seed": seed,
                "attempt_index": index,
                "induction_id": str(
                    upstream.get("induction_id") or _induction_id(attempt, index, seed)
                ),
                "reason": attempt.get("reason"),
                "started_at": attempt.get("started_at"),
                "elapsed_s": float(attempt.get("wall_s", 0.0) or 0.0),
                "selfparse": bool(gap.get("selfparse", True)),
                "tool_calls_total": calls,
                "recorded_tool_calls_total": receipt["recorded_tool_calls_total"],
                "tool_calls_by_name": deepcopy(receipt["tool_calls_by_name"]),
                "tool_call_events": deepcopy(receipt["tool_call_events"]),
                "per_name_evidence_state": (
                    "same_event_stream"
                    if receipt["aggregation_consistent"]
                    else "unavailable_do_not_invent"
                ),
                "aggregation_consistent": receipt["aggregation_consistent"],
                "aggregation_error": receipt["error"],
                "terminal_outcome": terminal_outcome,
                "terminal_result_returned": terminal,
                "engaged": engaged,
                "tool_gap_events": deepcopy(list(gap.get("tool_gap_events", []) or [])),
                "tool_gap_events_dropped": int(gap.get("tool_gap_events_dropped", 0) or 0),
                "tool_gap_attachment_state": (
                    "policy_attached" if policy_attached else "recovered_return"
                ),
                "completion_ids": [row.get("completion_id") for row in matching],
                "raw_completion_hashes": [row.get("content_sha256") for row in matching],
                "model_validity_errors": _model_validity_errors(attempt),
                "world_model_nondegeneracy": {
                    "engine_identity_measurable": attempt.get("engine_identity_measurable"),
                    "engine_functionally_identity": attempt.get("engine_functionally_identity"),
                    "goal_predicate_satisfiable": attempt.get("goal_predicate_satisfiable"),
                    "selected_candidate_name": attempt.get("selected_candidate_name"),
                },
                "error": attempt.get("skipped") or receipt["error"],
                "abstention": not engaged,
            }
        )
    return rows


def historical_induction_rows(payload: Mapping[str, Any], *, source_hash: str) -> list[JsonDict]:
    """Project authentic old loops while leaving unavailable names explicitly unknown."""

    seed = int(payload.get("random_seed", 0) or 0)
    result: list[JsonDict] = []
    games = payload.get("per_game", [])
    for game_row in games if isinstance(games, list) else []:
        if not isinstance(game_row, Mapping) or game_row.get("game") != GAME:
            continue
        diagnostics = game_row.get("policy_diagnostics", {})
        attempts = (
            diagnostics.get("induction_attempts", []) if isinstance(diagnostics, Mapping) else []
        )
        for index, raw_attempt in enumerate(attempts if isinstance(attempts, list) else []):
            if not isinstance(raw_attempt, Mapping):
                continue
            gap = raw_attempt.get("tool_gap")
            if not isinstance(gap, Mapping):
                continue
            calls = int(gap.get("tool_calls_total", 0) or 0)
            terminal = bool(gap.get("terminated_by"))
            if calls <= 0 or not terminal:
                continue
            result.append(
                {
                    "unit_id": f"{GAME}:{seed}:induction:{index}",
                    "source_session_id": "r11l-1594772",
                    "source_artifact_hash": source_hash,
                    "source_authenticated": True,
                    "game": GAME,
                    "seed": seed,
                    "attempt_index": index,
                    "induction_id": _induction_id(raw_attempt, index, seed),
                    "reason": raw_attempt.get("reason"),
                    "started_at": raw_attempt.get("started_at"),
                    "elapsed_s": float(raw_attempt.get("wall_s", 0.0) or 0.0),
                    "selfparse": True,
                    "tool_calls_total": calls,
                    "recorded_tool_calls_total": calls,
                    "tool_calls_by_name": {},
                    "tool_call_events": [],
                    "per_name_evidence_state": "not_recorded_do_not_invent",
                    "aggregation_consistent": None,
                    "aggregation_error": "raw_per_name_event_stream_not_recorded",
                    "terminal_outcome": str(gap.get("terminated_by")),
                    "terminal_result_returned": True,
                    "engaged": True,
                    "tool_gap_events": deepcopy(list(gap.get("tool_gap_events", []) or [])),
                    "tool_gap_events_dropped": int(gap.get("tool_gap_events_dropped", 0) or 0),
                    "completion_ids": [],
                    "raw_completion_hashes": [],
                    "model_validity_errors": _model_validity_errors(raw_attempt),
                    "world_model_nondegeneracy": {
                        "engine_identity_measurable": raw_attempt.get("engine_identity_measurable"),
                        "engine_functionally_identity": raw_attempt.get(
                            "engine_functionally_identity"
                        ),
                        "goal_predicate_satisfiable": raw_attempt.get("goal_predicate_satisfiable"),
                        "selected_candidate_name": raw_attempt.get("selected_candidate_name"),
                    },
                    "error": raw_attempt.get("skipped"),
                    "abstention": False,
                }
            )
    return result


def merge_cumulative_rows(
    source_groups: Sequence[tuple[str, Sequence[Mapping[str, Any]]]],
) -> tuple[list[JsonDict], JsonDict]:
    """Merge authenticated engaged rows by induction ID without double counting."""

    merged: list[JsonDict] = []
    by_id: dict[str, JsonDict] = {}
    for source, rows in source_groups:
        for raw in rows:
            if raw.get("source_authenticated") is not True or raw.get("engaged") is not True:
                continue
            induction_id = str(raw.get("induction_id") or "")
            if not HASH_RE.fullmatch(induction_id):
                continue
            if induction_id in by_id:
                duplicates = by_id[induction_id].setdefault("duplicate_sources", [])
                if source not in duplicates:
                    duplicates.append(source)
                continue
            row = deepcopy(dict(raw))
            row["duplicate_sources"] = []
            by_id[induction_id] = row
            merged.append(row)
    sessions = {str(row.get("source_session_id")) for row in merged}
    seeds = {int(row.get("seed", 0) or 0) for row in merged}
    return merged, {
        "cumulative_unique_inductions": len(merged),
        "distinct_session_count": len(sessions),
        "distinct_seed_count": len(seeds),
        "observed_tool_calls": sum(int(row.get("tool_calls_total", 0) or 0) for row in merged),
        "observed_gap_events": sum(len(row.get("tool_gap_events", []) or []) for row in merged),
    }


def _function_names(path: Path) -> tuple[set[str], JsonDict]:
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    names = {
        node.name for node in ast.walk(tree) if isinstance(node, (ast.FunctionDef, ast.ClassDef))
    }
    constructors: JsonDict = {}
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == "E3AgentPolicy":
            init = next(
                (
                    child
                    for child in node.body
                    if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef))
                    and child.name == "__init__"
                ),
                None,
            )
            if init is not None:
                constructors["E3AgentPolicy"] = [arg.arg for arg in init.args.args]
    return names, constructors


def adapter_isolation_receipt(eval_path: Path, policy_path: Path) -> JsonDict:
    """Verify the shipped evaluator chooses E3 without a solution constructor channel."""

    try:
        eval_names, _ = _function_names(eval_path)
        policy_names, constructors = _function_names(policy_path)
        eval_source = eval_path.read_text(encoding="utf-8")
        args = constructors.get("E3AgentPolicy", [])
    except (OSError, SyntaxError):
        eval_names, policy_names, eval_source, args = set(), set(), "", []
    observed = {
        "run_game_defined": "run_game" in eval_names,
        "evaluator_policy_builder_defined": "_build_policy" in eval_names,
        "evaluator_builds_e3": "E3AgentPolicy(" in eval_source,
        "e3_policy_defined": "E3AgentPolicy" in policy_names,
        "make_carnot_agent_defined": "make_carnot_agent" in policy_names,
        "observed_policy_constructor_has_solutions": "solutions" in args,
    }
    passed = (
        all(
            value is True
            for key, value in observed.items()
            if key != "observed_policy_constructor_has_solutions"
        )
        and observed["observed_policy_constructor_has_solutions"] is False
    )
    return {
        "passed": passed,
        **observed,
        "verified_before_model_work": True,
        "policy_class": "E3AgentPolicy",
        "policy_inputs": ["public_frames", "available_actions", "own_transitions"],
        "policy_denied": [
            "per_game_adapter",
            "game_source",
            "registry_contents",
            "solved_trajectories",
            "historical_world_models",
            "banked_solutions",
        ],
        "environment_executable_source_allowed": True,
        "registry_read_role": "evaluator_prelaunch_reproduction_confirmation_only",
        "entrypoint": "arc_leaderboard_eval.run_game -> E3AgentPolicy",
    }


def registry_reproduction_receipt(path: Path) -> JsonDict:
    """Read the evaluator-owned registry and confirm r11l is already reproduced."""

    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        return {"passed": False, "error": f"{type(exc).__name__}:{exc}"}
    games = payload.get("games", []) if isinstance(payload, Mapping) else []
    row = next(
        (dict(item) for item in games if isinstance(item, Mapping) and item.get("game") == GAME),
        {},
    )
    reproduced = row.get("reproducibility") == "reproduced"
    levels = int(row.get("levels_reproduced", 0) or 0)
    return {
        "passed": bool(reproduced and levels > 0),
        "game": GAME,
        "reproducibility": row.get("reproducibility"),
        "levels_reproduced": levels,
        "full_game_clear": row.get("full_game_clear"),
        "policy_access": "withheld",
    }


def snapshot_sources(root: Path, paths: Sequence[Path]) -> tuple[JsonDict, JsonDict]:
    """Record byte lengths and hashes without raising on a missing source."""

    sizes: JsonDict = {}
    hashes: JsonDict = {}
    for relative in paths:
        path = root / relative
        try:
            size = path.stat().st_size if path.is_file() else 0
            sizes[relative.as_posix()] = size
            hashes[relative.as_posix()] = sha256_file(path) if size else "missing"
        except OSError:
            sizes[relative.as_posix()] = 0
            hashes[relative.as_posix()] = "missing"
    return sizes, hashes


def _load_json(path: Path) -> tuple[JsonDict, str | None]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return {}, f"{type(exc).__name__}:{exc}"
    return (dict(value), None) if isinstance(value, Mapping) else ({}, "not_json_object")


def _task_contract(path: Path) -> JsonDict:
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError):
        return {}
    tasks = (
        payload
        if isinstance(payload, list)
        else payload.get("tasks", [])
        if isinstance(payload, Mapping)
        else []
    )
    task = next(
        (dict(row) for row in tasks if isinstance(row, Mapping) and row.get("id") == TASK_ID),
        {},
    )
    return {key: task.get(key) for key in EXPECTED_TASK_CONTRACT}


def _optional_sibling(root: Path, source_hashes: JsonDict) -> tuple[JsonDict, list[JsonDict]]:
    path = root / SIBLING_PATH
    if not path.is_file():
        return {"source": SIBLING_TASK_ID, "state": "absent_nonblocking"}, []
    payload, error = _load_json(path)
    source_hashes[SIBLING_PATH.as_posix()] = sha256_file(path)
    quarantined = is_quarantined(payload) if error is None else None
    complete = (
        unwrap_evidence_value(payload.get("arc_session_complete_score"))
        if error is None and quarantined is False
        else "not_consumed"
    )
    accepted = error is None and quarantined is False and complete == 1
    receipt = {
        "source": SIBLING_TASK_ID,
        "path": SIBLING_PATH.as_posix(),
        "source_hash": source_hashes[SIBLING_PATH.as_posix()],
        "state": "authenticated" if accepted else "rejected_nonblocking",
        "read_error": error,
        "quarantined": quarantined,
        "field": "arc_session_complete_score",
        "expected_value": 1,
        "observed_value": complete,
        "consumed": accepted,
    }
    raw_rows = payload.get("tool_induction_rows", []) if accepted else []
    rows = [
        deepcopy(dict(row))
        for row in raw_rows
        if isinstance(raw_rows, list) and isinstance(row, Mapping)
    ]
    return receipt, rows


def collect_static_preconditions(
    *, root: Path, result_path: Path, checkpoint_path: Path, raw_dir: Path
) -> tuple[list[JsonDict], JsonDict, JsonDict]:
    """Authenticate all required bytes and producer gates before model work."""

    sizes, hashes = snapshot_sources(root, REQUIRED_SOURCE_PATHS)
    spec = (root / SPEC_PATH).read_text(encoding="utf-8") if (root / SPEC_PATH).is_file() else ""
    isolation = adapter_isolation_receipt(root / EVAL_PATH, root / POLICY_PATH)
    registry = registry_reproduction_receipt(root / REGISTRY_PATH)
    prior, prior_error = _load_json(root / PRIOR_ARTIFACT_PATH)
    exp7193, exp7193_error = _load_json(root / EXP7193_PATH)
    session7193, session_error = _load_json(root / EXP7193_SESSION_PATH)
    row7193, row_error = _load_json(root / EXP7193_RUN_ROW_PATH)
    completion7193, completion_error = _load_json(root / EXP7193_COMPLETION_PATH)
    historical, historical_error = _load_json(root / HISTORICAL_PATH)
    exp7194, exp7194_error = _load_json(root / EXP7194_PATH)
    source_ok = all(size > 0 for size in sizes.values())
    hash_ok = all(isinstance(value, str) and HASH_RE.fullmatch(value) for value in hashes.values())
    tools = {
        "python": Path(sys.executable).is_file(),
        "nvidia-smi": shutil.which("nvidia-smi") is not None,
        "sha256sum": shutil.which("sha256sum") is not None,
        "native_llama_module": importlib.util.find_spec("carnot.agentic.arc_executable_world_model")
        is not None,
        "eval_script": (root / EVAL_PATH).is_file(),
    }
    storage = {
        "result_parent_writable": result_path.parent.is_dir()
        and os.access(result_path.parent, os.W_OK),
        "checkpoint_parent_writable": checkpoint_path.parent.is_dir()
        and os.access(checkpoint_path.parent, os.W_OK),
        "raw_dir_writable": raw_dir.is_dir() and os.access(raw_dir, os.W_OK),
    }
    checks = [
        gate_check(
            "driving_capability_spec",
            SPEC_PATH.as_posix(),
            DRIVING_REQUIREMENT,
            True,
            f"## {DRIVING_REQUIREMENT}:" in spec,
        ),
        gate_check(
            "required_source_bytes",
            "repository",
            "REQUIRED_SOURCE_PATHS",
            {path.as_posix(): "nonempty" for path in REQUIRED_SOURCE_PATHS},
            sizes,
            source_ok,
        ),
        gate_check(
            "required_source_hashes",
            "repository",
            "source_artifact_hashes",
            "sha256:<64 hex> for every required source",
            hashes,
            hash_ok,
        ),
        gate_check(
            "exact_v635_task_contract",
            ROADMAP_PATH.as_posix(),
            ",".join(EXPECTED_TASK_CONTRACT),
            EXPECTED_TASK_CONTRACT,
            _task_contract(root / ROADMAP_PATH),
        ),
    ]
    if prior_error is None:
        prior_gate = authenticated_field_gate(
            prior,
            upstream=PRIOR_ARTIFACT_PATH.as_posix(),
            field="honest_verdict",
            expected=EXPECTED_PRIOR_VERDICT,
        )
        prior_gate["observed_value"]["consumed"] = False
        prior_gate["expected_value"]["consumed"] = False
        prior_gate["evidence_role"] = "addressed_prior_failure_only"
    else:
        prior_gate = gate_check(
            "upstream_authenticated_field",
            PRIOR_ARTIFACT_PATH.as_posix(),
            "honest_verdict",
            EXPECTED_PRIOR_VERDICT,
            prior_error,
            False,
        )
    checks.append(prior_gate)
    if exp7193_error is None:
        checks.append(
            authenticated_field_gate(
                exp7193,
                upstream=EXP7193_PATH.as_posix(),
                field="arc_tool_measurement_complete_score",
                expected=1,
            )
        )
    else:
        checks.append(
            gate_check(
                "upstream_authenticated_field",
                EXP7193_PATH.as_posix(),
                "arc_tool_measurement_complete_score",
                1,
                exp7193_error,
                False,
            )
        )
    checks.extend(
        [
            gate_check(
                "exp7193_terminal_session",
                EXP7193_SESSION_PATH.as_posix(),
                "terminal_receipt,status",
                {"terminal_receipt": True, "status": "complete"},
                session_error
                or {
                    "terminal_receipt": session7193.get("terminal_receipt"),
                    "status": session7193.get("status"),
                },
                session_error is None
                and session7193.get("terminal_receipt") is True
                and session7193.get("status") == "complete",
            ),
            gate_check(
                "exp7193_run_row_hash_join",
                EXP7193_SESSION_PATH.as_posix(),
                "run_row_sha256",
                session7193.get("run_row_sha256") if session_error is None else None,
                hashes.get(EXP7193_RUN_ROW_PATH.as_posix()) if row_error is None else row_error,
                session_error is None
                and row_error is None
                and session7193.get("run_row_sha256")
                == hashes.get(EXP7193_RUN_ROW_PATH.as_posix()),
            ),
            gate_check(
                "exp7193_completion_manifest_join",
                EXP7193_COMPLETION_PATH.as_posix(),
                "completions",
                session7193.get("completions") if session_error is None else None,
                completion7193.get("completions") if completion_error is None else completion_error,
                session_error is None
                and completion_error is None
                and session7193.get("completions") == completion7193.get("completions"),
            ),
            gate_check(
                "exp7193_model_identity",
                EXP7193_SESSION_PATH.as_posix(),
                "model_identity_validation.valid",
                True,
                (
                    session7193.get("model_identity_validation", {}).get("valid")
                    if isinstance(session7193.get("model_identity_validation"), Mapping)
                    else None
                ),
            ),
            gate_check(
                "historical_receipt_authenticated",
                HISTORICAL_PATH.as_posix(),
                "complete,quarantine,r11l",
                {"complete": True, "quarantined": False, "r11l": True},
                {
                    "complete": historical.get("complete") if historical_error is None else None,
                    "quarantined": is_quarantined(historical) if historical_error is None else None,
                    "r11l": any(
                        isinstance(row, Mapping) and row.get("game") == GAME
                        for row in historical.get("per_game", [])
                    )
                    if historical_error is None and isinstance(historical.get("per_game"), list)
                    else False,
                    "error": historical_error,
                },
                historical_error is None
                and historical.get("complete") is True
                and not is_quarantined(historical)
                and any(
                    isinstance(row, Mapping) and row.get("game") == GAME
                    for row in historical.get("per_game", [])
                ),
            ),
            gate_check(
                "shipped_policy_access_boundary",
                f"{EVAL_PATH.as_posix()},{POLICY_PATH.as_posix()}",
                "run_game,E3AgentPolicy,make_carnot_agent,no_solutions_constructor",
                True,
                isolation.get("passed"),
            ),
            gate_check(
                "registry_reproduced_before_launch",
                REGISTRY_PATH.as_posix(),
                "r11l.reproducibility,levels_reproduced",
                {"reproducibility": "reproduced", "levels_reproduced": ">0"},
                {
                    "reproducibility": registry.get("reproducibility"),
                    "levels_reproduced": registry.get("levels_reproduced"),
                },
                registry.get("passed") is True,
            ),
            gate_check("required_tools", "host", "tools", {key: True for key in tools}, tools),
            gate_check(
                "output_directories",
                "host_filesystem",
                "result,checkpoint,raw",
                {key: True for key in storage},
                storage,
            ),
        ]
    )
    optional = [
        {
            "source": "exp7194-arc-gap-audit",
            "path": EXP7194_PATH.as_posix(),
            "state": "read_only_diagnostic_not_cumulative_authority",
            "read_error": exp7194_error,
            "quarantined": is_quarantined(exp7194) if exp7194_error is None else None,
            "structured_fields_consumed": False,
        }
    ]
    sibling_receipt, sibling_rows = _optional_sibling(root, hashes)
    optional.append(sibling_receipt)
    return (
        checks,
        hashes,
        {
            "isolation": isolation,
            "registry": registry,
            "historical": historical,
            "exp7193": exp7193,
            "exp7193_session": session7193,
            "exp7193_run_row": row7193,
            "optional_source_receipts": optional,
            "sibling_rows": sibling_rows,
        },
    )


def _transient_level_transitions(run_row: Mapping[str, Any] | None) -> list[JsonDict]:
    if not isinstance(run_row, Mapping):
        return []
    transitions: list[JsonDict] = []
    previous: int | None = None
    frames = run_row.get("frame_sequence", [])
    for row in frames if isinstance(frames, list) else []:
        if not isinstance(row, Mapping) or row.get("levels_completed") is None:
            continue
        level = int(row["levels_completed"])
        if previous is not None and level > previous:
            transitions.append(
                {
                    "from_level": previous,
                    "to_level": level,
                    "frame_index": row.get("frame_index"),
                    "action_count": row.get("action_count"),
                }
            )
        previous = level
    return transitions


def _timing_spans(session: Mapping[str, Any] | None, parsing_duration_s: float) -> list[JsonDict]:
    spans = (
        deepcopy(list(session.get("phase_spans", []) or [])) if isinstance(session, Mapping) else []
    )
    completions = session.get("completions", []) if isinstance(session, Mapping) else []
    for phase, key in (("prefill", "prompt_ms"), ("generation", "predicted_ms")):
        values = [
            float(row.get("timings", {}).get(key))
            for row in completions
            if isinstance(row, Mapping)
            and isinstance(row.get("timings"), Mapping)
            and isinstance(row.get("timings", {}).get(key), (int, float))
        ]
        spans.append(
            {
                "phase": phase,
                "duration_s": sum(values) / 1000.0 if values else None,
                "measured": bool(values),
                "completed_units": len(values),
            }
        )
    spans.append({"phase": "parsing", "duration_s": float(parsing_duration_s), "measured": True})
    return spans


def build_terminal_artifact(
    *,
    duration_s: float,
    checks: Sequence[Mapping[str, Any]],
    source_hashes: Mapping[str, Any],
    session: Mapping[str, Any] | None,
    current_rows: Sequence[Mapping[str, Any]],
    cumulative_rows: Sequence[Mapping[str, Any]],
    cumulative_summary: Mapping[str, Any],
    historical_count: int,
    optional_source_receipts: Sequence[Mapping[str, Any]],
    isolation: Mapping[str, Any],
    registry: Mapping[str, Any],
    parsing_duration_s: float,
) -> JsonDict:
    """Build one evidence-linked blocked or complete cumulative-session artifact."""

    check_rows = [deepcopy(dict(row)) for row in checks]
    summary = gate_summary(check_rows)
    blocked = summary["passed"] is not True or session is None
    terminal = bool(isinstance(session, Mapping) and session.get("terminal_receipt"))
    run_row = reused._session_run_row(session)
    completions = (
        [deepcopy(dict(row)) for row in session.get("completions", [])]
        if isinstance(session, Mapping) and isinstance(session.get("completions"), list)
        else []
    )
    inference_substrate, substrate_class, inference_mode = reused.classify_inference_work(session)
    if session is None:
        inference_substrate, substrate_class, inference_mode = (
            "preflight_only_no_model_load",
            "blocked_no_run",
            "not_run",
        )
    rows = [deepcopy(dict(row)) for row in current_rows]
    cumulative = [deepcopy(dict(row)) for row in cumulative_rows]
    gap_events = sum(len(row.get("tool_gap_events", []) or []) for row in cumulative)
    new_count = sum(row.get("engaged") is True for row in rows)
    cumulative_count = len(cumulative)
    if blocked:
        status = "blocked"
        verdict_class = "blocked"
        honest_verdict = "blocked_" + str(summary.get("failed_check") or "live_prerequisite")
        complete_score = 0
    else:
        status = "complete"
        verdict_class = "positive" if gap_events else "null"
        honest_verdict = (
            f"complete_observed_missing_tool_demand_cumulative_n_{cumulative_count}"
            if gap_events
            else f"complete_null_no_observed_missing_tool_demand_cumulative_n_{cumulative_count}"
        )
        complete_score = int(terminal)
    comparison_rows = []
    if not blocked:
        comparison_rows.append(
            {
                "unit_id": f"{GAME}:{RANDOM_SEED}:direct_selfparse",
                "arm": "direct_selfparse",
                "seed": RANDOM_SEED,
                "metric": "real_returned_tool_loop_inductions",
                "metric_value": new_count,
                "error": session.get("error") if isinstance(session, Mapping) else None,
                "abstention": run_row is None,
            }
        )
    actions = int(run_row.get("actions", 0) or 0) if isinstance(run_row, Mapping) else 0
    levels = int(run_row.get("levels", 0) or 0) if isinstance(run_row, Mapping) else 0
    deepest = (
        int(run_row.get("deepest_level_reached", 0) or 0) if isinstance(run_row, Mapping) else 0
    )
    registry_levels = int(registry.get("levels_reproduced", 0) or 0)
    per_game = []
    if not blocked:
        per_game.append(
            {
                "game": GAME,
                "seed": RANDOM_SEED,
                "action_cap": ACTION_BUDGET,
                "session_timeout_s": SESSION_TIMEOUT_S,
                "induction_timeout_s": INDUCTION_TIMEOUT_S,
                "actions": actions,
                "registry_banked_levels_before_run": registry_levels,
                "banked_levels": registry_levels,
                "new_banked_levels": 0,
                "session_completed_levels": levels,
                "deepest_transient_level": deepest,
                "transient_level_transitions": _transient_level_transitions(run_row),
                "world_model_nondegeneracy": [
                    deepcopy(row.get("world_model_nondegeneracy", {})) for row in rows
                ],
                "new_tool_loop_inductions": new_count,
                "tool_calls": sum(int(row.get("tool_calls_total", 0) or 0) for row in rows),
                "timed_out": bool(session.get("timed_out"))
                if isinstance(session, Mapping)
                else False,
                "terminal_outcome": (
                    "timed_out"
                    if isinstance(session, Mapping) and session.get("timed_out")
                    else "returned"
                ),
                "error": session.get("error") if isinstance(session, Mapping) else None,
            }
        )
    actual_models: list[JsonDict] = []
    if isinstance(session, Mapping):
        model = session.get("model_spec") or session.get("model_identity")
        if isinstance(model, Mapping):
            actual_models.append(deepcopy(dict(model)))
    model_invoked = bool(completions)
    attempted = int(session is not None)
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "field_principles": {},
        "status": status,
        "run_date": RUN_DATE,
        "preconditions_checked": check_rows,
        "inference_substrate": inference_substrate,
        "inference_substrate_class": substrate_class,
        "inference_mode": inference_mode,
        "execution_venue": "host",
        "execution_host": socket.gethostname(),
        "duration_s": float(duration_s),
        "source_artifact_hashes": deepcopy(dict(source_hashes)),
        "rows": comparison_rows,
        "sample_size_budget": {
            "planned_sessions": 1,
            "attempted_sessions": attempted,
            "completed_sessions": complete_score,
            "censored_sessions": int(bool(session and session.get("timed_out"))),
            "independent_units": 1,
            "planned_action_cap": ACTION_BUDGET,
            "completed_actions": actions,
            "historical_tool_loop_inductions": int(historical_count),
            "new_tool_loop_inductions": int(new_count),
            "cumulative_unique_inductions": cumulative_count,
            "distinct_session_count": int(cumulative_summary.get("distinct_session_count", 0) or 0),
            "distinct_seed_count": int(cumulative_summary.get("distinct_seed_count", 0) or 0),
            "evidence_target": EVIDENCE_TARGET,
            "evidence_target_role": "operational_collection_target_not_statistical_proof",
            "exclusions": [summary.get("failed_check")] if blocked else [],
        },
        "random_seed": RANDOM_SEED,
        "reproducibility_checksum": "",
        "gate_check_summary": summary,
        "verifier_is_oracle": False,
        "verdict_class": verdict_class,
        "honest_verdict": honest_verdict,
        "arc_session_complete_score": complete_score,
        "solve_provenance": "live_agent_self_discovery",
        "per_game_results": per_game,
        "tool_induction_rows": rows,
        "cumulative_induction_rows": cumulative,
        "adapter_isolation_receipt": deepcopy(dict(isolation)),
        "registry_reproduction_receipt": deepcopy(dict(registry)),
        "optional_source_receipts": [deepcopy(dict(row)) for row in optional_source_receipts],
        "demand_observation": {
            "finding": (
                "observed_missing_tool_demand" if gap_events else "no_observed_missing_tool_demand"
            ),
            "observed_induction_denominator": cumulative_count,
            "observed_tool_call_denominator": int(
                cumulative_summary.get("observed_tool_calls", 0) or 0
            ),
            "observed_gap_events": gap_events,
            "evidence_target": EVIDENCE_TARGET,
            "statistical_absence_claimed": False,
            "limitations": [
                "inductions_within_a_session_are_dependent",
                "public_r11l_sessions_do_not_establish_cross_game_independence",
                "ten_is_an_operational_target_not_a_statistical_floor",
                "zero_events_means_no_observed_demand_not_absent_demand",
            ],
        },
        "MODEL_SPECS": deepcopy(MODEL_SPECS),
        "model_invoked": model_invoked,
        "model_specs": actual_models,
        "phase_spans": _timing_spans(session, parsing_duration_s),
        "gpu_receipts": (
            deepcopy(dict(session.get("gpu_receipts", {}))) if isinstance(session, Mapping) else {}
        ),
        "model_identity_receipt": (
            deepcopy(dict(session.get("model_identity", {})))
            if isinstance(session, Mapping)
            else {}
        ),
        "runner_receipt": (
            deepcopy(dict(session.get("runner_receipt", {})))
            if isinstance(session, Mapping)
            else {}
        ),
        "completion_receipts": completions,
        "session_receipt": {
            "terminal_receipt": terminal,
            "status": session.get("status") if isinstance(session, Mapping) else None,
            "timed_out": bool(session.get("timed_out")) if isinstance(session, Mapping) else False,
            "error": session.get("error") if isinstance(session, Mapping) else None,
            "run_row_path": session.get("run_row_path") if isinstance(session, Mapping) else None,
            "run_row_sha256": session.get("run_row_sha256")
            if isinstance(session, Mapping)
            else None,
        },
        "new_solve_claimed": False,
        "paired_efficacy_reported": False,
        "official_leaderboard_score_reported": False,
        "registered_level_increment": 0,
        "submitted": False,
    }
    artifact["field_principles"] = {key: FIELD_PRINCIPLES[key] for key in artifact}
    artifact["reproducibility_checksum"] = artifact_checksum(artifact)
    return artifact


def validate_artifact(value: Mapping[str, Any] | str | Path) -> list[str]:
    """Recompute the terminal schema, counts, classifications, and non-claims."""

    if isinstance(value, (str, Path)):
        try:
            loaded = json.loads(Path(value).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return ["artifact_unreadable"]
        artifact = dict(loaded) if isinstance(loaded, Mapping) else {}
    else:
        artifact = dict(value)
    errors: list[str] = []
    missing = sorted(REQUIRED_ARTIFACT_FIELDS - set(artifact))
    if missing:
        errors.append("missing_fields:" + ",".join(missing))
    principles = artifact.get("field_principles", {})
    if not isinstance(principles, Mapping) or set(principles) != set(artifact):
        errors.append("field_principles_must_cover_every_top_level_field")
    elif any(principles.get(key) != FIELD_PRINCIPLES.get(key) for key in artifact):
        errors.append("field_principles_mismatch")
    if artifact.get("run_date") != RUN_DATE:
        errors.append("run_date_mismatch")
    if artifact.get("MODEL_SPECS") != MODEL_SPECS:
        errors.append("model_specs_declaration_mismatch")
    if artifact.get("execution_venue") != "host":
        errors.append("execution_venue_invalid")
    if artifact.get("verdict_class") not in VERDICT_CLASSES:
        errors.append("verdict_class_invalid")
    checks = artifact.get("preconditions_checked", [])
    check_rows = checks if isinstance(checks, list) else []
    if artifact.get("gate_check_summary") != gate_summary(check_rows):
        errors.append("gate_check_summary_mismatch")
    blocked = artifact.get("verdict_class") == "blocked"
    if blocked:
        if artifact.get("status") != "blocked" or artifact.get("arc_session_complete_score") != 0:
            errors.append("blocked_terminal_inconsistent")
        if not str(artifact.get("honest_verdict", "")).startswith("blocked_"):
            errors.append("blocked_verdict_prefix_missing")
    else:
        if artifact.get("status") != "complete" or artifact.get("arc_session_complete_score") != 1:
            errors.append("complete_terminal_inconsistent")
        if not str(artifact.get("honest_verdict", "")).startswith("complete_"):
            errors.append("complete_verdict_prefix_missing")
    for row in artifact.get("rows", []) if isinstance(artifact.get("rows"), list) else []:
        if not isinstance(row, Mapping) or not {
            "unit_id",
            "arm",
            "seed",
            "metric",
            "error",
            "abstention",
        } <= set(row):
            errors.append("comparison_row_contract_invalid")
            break
    cumulative = artifact.get("cumulative_induction_rows", [])
    cumulative_rows = cumulative if isinstance(cumulative, list) else []
    ids = [row.get("induction_id") for row in cumulative_rows if isinstance(row, Mapping)]
    if len(ids) != len(set(ids)):
        errors.append("cumulative_induction_ids_not_unique")
    sample = artifact.get("sample_size_budget", {})
    if not isinstance(sample, Mapping) or sample.get("cumulative_unique_inductions") != len(
        cumulative_rows
    ):
        errors.append("cumulative_count_inconsistent")
    current = artifact.get("tool_induction_rows", [])
    current_rows = current if isinstance(current, list) else []
    expected_new = sum(
        row.get("engaged") is True for row in current_rows if isinstance(row, Mapping)
    )
    if isinstance(sample, Mapping) and sample.get("new_tool_loop_inductions") != expected_new:
        errors.append("new_count_inconsistent")
    demand = artifact.get("demand_observation", {})
    if not isinstance(demand, Mapping) or demand.get("observed_induction_denominator") != len(
        cumulative_rows
    ):
        errors.append("demand_denominator_inconsistent")
    if isinstance(demand, Mapping) and demand.get("statistical_absence_claimed") is not False:
        errors.append("statistical_absence_claim_forbidden")
    if artifact.get("solve_provenance") != "live_agent_self_discovery":
        errors.append("solve_provenance_invalid")
    if artifact.get("new_solve_claimed") is not False:
        errors.append("new_solve_claim_forbidden")
    if artifact.get("paired_efficacy_reported") is not False:
        errors.append("paired_efficacy_claim_forbidden")
    if artifact.get("official_leaderboard_score_reported") is not False:
        errors.append("official_score_claim_forbidden")
    if artifact.get("registered_level_increment") != 0:
        errors.append("registry_increment_forbidden")
    if artifact.get("submitted") is not False:
        errors.append("submission_forbidden")
    substrate_class = str(artifact.get("inference_substrate_class"))
    floor = DURATION_FLOORS.get(substrate_class)
    if floor is not None and float(artifact.get("duration_s", 0.0) or 0.0) < floor:
        errors.append(f"duration_floor_not_met:{substrate_class}")
    gpu = artifact.get("gpu_receipts", {})
    if artifact.get("inference_mode") == "live_gpu" and (
        not isinstance(gpu, Mapping)
        or gpu.get("provenance_ok") is not True
        or gpu.get("task_linked_cuda_execution") is not True
    ):
        errors.append("live_gpu_without_task_linked_cuda_receipt")
    checksum = artifact.get("reproducibility_checksum")
    if not isinstance(checksum, str) or not HASH_RE.fullmatch(checksum):
        errors.append("reproducibility_checksum_invalid")
    elif checksum != artifact_checksum(artifact):
        errors.append("reproducibility_checksum_mismatch")
    return errors


def _progress(phase: int, event: str, **fields: Any) -> None:
    """Print one flushed, machine-readable observed progress record."""

    print(canonical_json({"phase": phase, "event": event, **fields}), flush=True)


def _historical_groups(
    root: Path, context: Mapping[str, Any]
) -> tuple[list[tuple[str, list[JsonDict]]], int]:
    historical = context.get("historical", {})
    historical = historical if isinstance(historical, Mapping) else {}
    history_rows = historical_induction_rows(
        historical, source_hash=sha256_file(root / HISTORICAL_PATH)
    )
    session = context.get("exp7193_session", {})
    run_row = context.get("exp7193_run_row", {})
    exp7193 = context.get("exp7193", {})
    session = session if isinstance(session, Mapping) else {}
    run_row = run_row if isinstance(run_row, Mapping) else {}
    exp7193 = exp7193 if isinstance(exp7193, Mapping) else {}
    completions = session.get("completions", [])
    upstream_rows = exp7193.get("tool_induction_rows", [])
    exp7193_rows = project_tool_inductions(
        root,
        run_row,
        completions if isinstance(completions, list) else [],
        seed=7_193_001,
        session_id="exp7193-arc-direct-tool",
        source_hash=sha256_file(root / EXP7193_PATH),
        upstream_rows=upstream_rows if isinstance(upstream_rows, list) else [],
    )
    authorized_indices = {
        int(row["attempt_index"])
        for row in upstream_rows
        if isinstance(upstream_rows, list)
        and isinstance(row, Mapping)
        and isinstance(row.get("attempt_index"), int)
        and not isinstance(row.get("attempt_index"), bool)
    }
    exp7193_rows = [row for row in exp7193_rows if row.get("attempt_index") in authorized_indices]
    groups = [("r11l-1594772", history_rows), ("exp7193-arc-direct-tool", exp7193_rows)]
    return groups, sum(row.get("engaged") is True for _, rows in groups for row in rows)


def run_experiment(
    *,
    root: Path,
    run_date: str,
    result_path: Path,
    checkpoint_path: Path,
    raw_dir: Path,
    live_runner: Callable[..., Any] | None = None,
) -> JsonDict:
    """Run preflight, one bounded reused live session, merge, validate, and write."""

    started = time.monotonic()
    _progress(0, "phase_start", name="checkpoint_before_checks")
    result_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)
    atomic_write(
        checkpoint_path,
        {
            "schema": CHECKPOINT_SCHEMA,
            "task_id": TASK_ID,
            "status": "preflight_pending",
            "terminal": False,
        },
    )
    _progress(0, "phase_end", name="checkpoint_before_checks")

    _progress(1, "phase_start", name="static_preconditions")
    _progress(1, "benchmark_start", operation="source_and_producer_gates")
    checks, source_hashes, context = collect_static_preconditions(
        root=root,
        result_path=result_path,
        checkpoint_path=checkpoint_path,
        raw_dir=raw_dir,
    )
    checks.insert(
        0,
        gate_check(
            "execution_date",
            TASK_ID,
            "run_date",
            RUN_DATE,
            run_date,
        ),
    )
    static_ready = all(row.get("passed") is True for row in checks)
    _progress(1, "benchmark_end", operation="source_and_producer_gates", passed=static_ready)
    _progress(1, "phase_end", name="static_preconditions", passed=static_ready)

    configure_reused_driver()
    session: JsonDict | None = None
    if static_ready and live_runner is not None:
        _progress(2, "phase_start", name="injected_live_session")
        value = live_runner(
            root=root,
            model={},
            gpu={},
            server="",
            raw_dir=raw_dir,
            checkpoint_path=checkpoint_path,
        )
        if isinstance(value, tuple):
            session, extra_checks = value
            checks.extend(extra_checks)
        else:
            session = value
        _progress(2, "phase_end", name="injected_live_session", returned=session is not None)
    elif static_ready:
        _progress(2, "phase_start", name="cache_cuda_conflict_checks")
        live_checks, model, gpu, server = reused._live_resource_preconditions(root)
        checks.extend(live_checks)
        live_ready = all(row.get("passed") is True for row in live_checks)
        _progress(2, "phase_end", name="cache_cuda_conflict_checks", passed=live_ready)
        if live_ready and model is not None and gpu is not None and server is not None:
            _progress(3, "phase_start", name="leased_live_session")
            session, runtime_checks = reused._run_live_session(
                root=root,
                model=model,
                gpu=gpu,
                server=server,
                raw_dir=raw_dir,
                checkpoint_path=checkpoint_path,
            )
            checks.extend(runtime_checks)
            _progress(3, "phase_end", name="leased_live_session", returned=session is not None)
    if static_ready and session is None and all(row.get("passed") is True for row in checks):
        checks.append(
            gate_check(
                "live_session_receipt",
                TASK_ID,
                "terminal_receipt",
                True,
                None,
                False,
            )
        )

    _progress(7, "phase_start", name="receipt_parsing_and_cumulative_merge")
    parse_started = time.monotonic()
    run_row = reused._session_run_row(session)
    completions = (
        session.get("completions", [])
        if isinstance(session, Mapping) and isinstance(session.get("completions"), list)
        else []
    )
    current_rows = project_tool_inductions(root, run_row, completions)
    historical_groups, historical_count = _historical_groups(root, context)
    sibling_rows = context.get("sibling_rows", [])
    sibling_rows = sibling_rows if isinstance(sibling_rows, list) else []
    source_groups = [
        *historical_groups,
        (SIBLING_TASK_ID, sibling_rows),
        (TASK_ID, current_rows),
    ]
    cumulative_rows, cumulative_summary = merge_cumulative_rows(source_groups)
    parsing_duration = time.monotonic() - parse_started
    _progress(
        7,
        "phase_end",
        name="receipt_parsing_and_cumulative_merge",
        completed_units=len(current_rows),
        cumulative_unique=len(cumulative_rows),
        elapsed_s=round(parsing_duration, 3),
    )

    artifact = build_terminal_artifact(
        duration_s=time.monotonic() - started,
        checks=checks,
        source_hashes=source_hashes,
        session=session,
        current_rows=current_rows,
        cumulative_rows=cumulative_rows,
        cumulative_summary=cumulative_summary,
        historical_count=historical_count,
        optional_source_receipts=context.get("optional_source_receipts", []),
        isolation=context.get("isolation", {}),
        registry=context.get("registry", {}),
        parsing_duration_s=parsing_duration,
    )
    _progress(8, "phase_start", name="terminal_validation_and_atomic_write")
    _progress(8, "validation_start", operation="cold_artifact_validation")
    validation_started = time.monotonic()
    errors = validate_artifact(artifact)
    validation_duration = time.monotonic() - validation_started
    artifact["phase_spans"].append(
        {"phase": "verification", "duration_s": validation_duration, "measured": True}
    )
    artifact["duration_s"] = time.monotonic() - started
    artifact["reproducibility_checksum"] = artifact_checksum(artifact)
    errors.extend(validate_artifact(artifact))
    errors = list(dict.fromkeys(errors))
    _progress(8, "validation_end", operation="cold_artifact_validation", errors=errors)
    if errors:
        raise ValueError("terminal_artifact_invalid:" + ",".join(errors))
    _progress(8, "artifact_write_start", path=str(result_path), status=artifact["status"])
    atomic_write(result_path, artifact)
    _progress(8, "artifact_write_end", path=str(result_path), status=artifact["status"])
    _progress(8, "phase_end", name="terminal_validation_and_atomic_write")
    return artifact


def run_session_child(args: argparse.Namespace) -> int:
    """Delegate the isolated CUDA child to the configured shipped runtime."""

    configure_reused_driver()
    return int(reused.run_session_child(args))


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse driver, isolated-session, and cold-validation modes."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--role", choices=("driver", "session"), default="driver")
    parser.add_argument("--date", default=RUN_DATE)
    parser.add_argument("--result-path", type=Path, default=RESULT_PATH)
    parser.add_argument("--checkpoint-path", type=Path, default=CHECKPOINT_PATH)
    parser.add_argument("--raw-dir", type=Path, default=RAW_DIR)
    parser.add_argument("--validate", type=Path)
    parser.add_argument("--model-path")
    parser.add_argument("--gpu-index", type=int)
    parser.add_argument("--port", type=int)
    parser.add_argument("--session-output", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Execute one bounded cumulative session, its child, or validation."""

    _progress(0, "phase_start", name="entrypoint_before_checks")
    args = parse_args(argv)
    if args.validate is not None:
        _progress(8, "validation_start", path=str(args.validate))
        errors = validate_artifact(args.validate)
        _progress(8, "validation_end", path=str(args.validate), errors=errors)
        return int(bool(errors))
    if args.role == "session":
        return run_session_child(args)
    root = Path(os.environ.get("CARNOT_REPO_ROOT", REPO_ROOT)).resolve()
    result_path = args.result_path if args.result_path.is_absolute() else root / args.result_path
    checkpoint_path = (
        args.checkpoint_path if args.checkpoint_path.is_absolute() else root / args.checkpoint_path
    )
    raw_dir = args.raw_dir if args.raw_dir.is_absolute() else root / args.raw_dir
    artifact = run_experiment(
        root=root,
        run_date=str(args.date),
        result_path=result_path,
        checkpoint_path=checkpoint_path,
        raw_dir=raw_dir,
    )
    print(
        canonical_json(
            {
                "artifact": str(result_path),
                "status": artifact["status"],
                "verdict": artifact["honest_verdict"],
            }
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
