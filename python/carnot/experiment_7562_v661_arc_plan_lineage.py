"""Measure default-off ARC plan lineage without loading a model or playing a game."""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Mapping, Sequence
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import random
from types import SimpleNamespace
import tempfile
import time
from typing import Any

from carnot.agentic import arc_decision_telemetry as telemetry
from carnot.agentic.arc_competition_agent import E3AgentPolicy
from carnot.experiment_7358_v646_validation_contract import AffectedManifest
from carnot.reporting import experiment_7303_validation_scope as validation_scope


Json = dict[str, Any]
REPO_ROOT = Path(__file__).resolve().parents[2]
RUN_DATE = "20260923"
MILESTONE = "2026.09.661"
EXPERIMENT_ID = "exp7562-arc-plan-lineage"
SCHEMA = "carnot.exp7562.v661.arc_plan_lineage.v1"
RESULT_PATH = Path("results/experiment_7562_v661_arc_plan_lineage.json")
UPSTREAM_PATH = Path("results/experiment_7557_v660_arc_generalization.json")
SPEC_PATH = Path("openspec/capabilities/arc-world-model-trust-energy/spec.md")
MODULE_PATH = Path("python/carnot/experiment_7562_v661_arc_plan_lineage.py")
WRAPPER_PATH = Path("scripts/experiments/experiment_7562_v661_arc_plan_lineage.py")
TEST_PATH = Path("tests/python/test_experiment_7562_v661_arc_plan_lineage.py")
TELEMETRY_PATH = Path("python/carnot/agentic/arc_decision_telemetry.py")
POLICY_PATH = Path("python/carnot/agentic/arc_competition_agent.py")
MODEL_SPECS: list[Json] = []
FROZEN_GAMES = ("sb26", "vc33", "su15", "g50t", "m0r0", "dc22")
FROZEN_SEEDS = (7570001, 7570002)
ZERO_INVOCATION_COUNTS = {
    f"{kind}_{state}": 0
    for kind in ("model_loads", "forward_calls", "generation_calls")
    for state in ("attempted", "completed", "failed", "cancelled", "in_flight")
}
TERMINAL_COMMAND_NAMES = (
    "declared_entrypoint",
    "fresh_process_cold_replay",
    "independent_reduction",
    "adversarial_verify",
    "verdict_row_consistency_strict",
)
E2E_COMMAND_NAMES = ("e2e_011", "e2e_012", "e2e_013", "llm_off_environment_smoke")
REQUIRED_RECEIPT_NAMES = (
    *validation_scope.REQUIRED_CHECK_NAMES,
    *E2E_COMMAND_NAMES,
    *TERMINAL_COMMAND_NAMES,
)
AFFECTED_MANIFEST = AffectedManifest(
    experiment_id=EXPERIMENT_ID,
    test_paths=(TEST_PATH.as_posix(),),
    changed_modules=(MODULE_PATH.as_posix(),),
    static_paths=(
        WRAPPER_PATH.as_posix(),
        TELEMETRY_PATH.as_posix(),
        POLICY_PATH.as_posix(),
    ),
)


def progress(started: float, phase: str, event: str, **details: Any) -> None:
    """Print every slow boundary so the conductor can distinguish work from a stall."""

    suffix = " ".join(f"{key}={value}" for key, value in sorted(details.items()))
    print(
        f"[exp7562] phase={phase} event={event} elapsed_s={time.monotonic() - started:.3f}"
        + (f" {suffix}" if suffix else ""),
        flush=True,
    )


def sha256_file(path: Path) -> str:
    """Hash exact bytes so the artifact detects later source drift."""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def canonical_hash(value: Any) -> str:
    """Hash stable JSON so a removed row changes the run identity."""

    data = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()
    return "sha256:" + hashlib.sha256(data).hexdigest()


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    """Publish one complete object after its temporary bytes reach storage."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def load_json(path: Path) -> Json:
    """Load one required JSON object and reject every other top-level type."""

    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"json_object_required:{path}")
    return value


def precondition_row(
    check: str, upstream: str, artifact_field: str, expected: Any, observed: Any
) -> Json:
    """Keep the exact dependency comparison in blocked artifacts."""

    return {
        "check": check,
        "upstream": upstream,
        "path": upstream,
        "artifact_field": artifact_field,
        "expected": expected,
        "observed": observed,
        "op": "==",
        "passed": observed == expected,
        "principle": "Dependent evidence must exist before measurement.",
    }


def collect_preconditions(root: Path) -> list[Json]:
    """Authenticate Exp7557, the requirement, resources, and owned worktree paths."""

    upstream_path = root / UPSTREAM_PATH
    available = upstream_path.is_file()
    checks = [
        precondition_row(
            "exp7557_upstream_available",
            UPSTREAM_PATH.as_posix(),
            "path",
            "readable_file",
            "readable_file" if available else None,
        )
    ]
    upstream: Json = {}
    if available:
        try:
            upstream = load_json(upstream_path)
        except (OSError, ValueError, json.JSONDecodeError):
            checks[0].update(observed="malformed_json", passed=False)
    for check, field, expected in (
        ("exp7557_identity", "experiment_id", "exp7557-arc-generalization"),
        (
            "exp7557_literal_verdict",
            "honest_verdict",
            "complete_null_feasibility_only_causal_endpoint_unavailable",
        ),
        ("exp7557_verdict_class", "verdict_class", "null"),
        ("exp7557_ready", "arc_analysis_complete_score", 1),
        ("exp7557_not_flagged", "flagged_adversarial", False),
    ):
        if upstream:
            checks.append(
                precondition_row(
                    check, UPSTREAM_PATH.as_posix(), field, expected, upstream.get(field)
                )
            )
    required = (
        Path("AGENTS.md"),
        Path("CODEX.md"),
        Path("CLAUDE.md"),
        Path("research-program.md"),
        Path("ops/exclusion_manifest.yaml"),
        Path("ops/e2e-test-plan.md"),
        Path("ops/arc_solve_registry.yaml"),
        SPEC_PATH,
        MODULE_PATH,
        WRAPPER_PATH,
        TEST_PATH,
        TELEMETRY_PATH,
        POLICY_PATH,
    )
    for relative in required:
        present = (root / relative).is_file()
        checks.append(
            precondition_row(
                f"required_input:{relative.as_posix()}",
                relative.as_posix(),
                "path",
                "readable_file",
                "readable_file" if present else None,
            )
        )
    try:
        spec = (root / SPEC_PATH).read_text(encoding="utf-8")
    except OSError:
        spec = ""
    checks.append(
        precondition_row(
            "requirement_declared",
            SPEC_PATH.as_posix(),
            "REQ-ARC-WMTE-7562",
            True,
            "REQ-ARC-WMTE-7562" in spec,
        )
    )
    try:
        registry = (root / "ops/arc_solve_registry.yaml").read_text(encoding="utf-8")
    except OSError:
        registry = ""
    for game in FROZEN_GAMES:
        checks.append(
            precondition_row(
                f"registry_precheck:{game}",
                "ops/arc_solve_registry.yaml",
                game,
                True,
                game in registry,
            )
        )
    return checks


def first_failed_precondition(checks: Sequence[Mapping[str, Any]]) -> Json | None:
    """Return the first exact failure while the artifact retains every check."""

    return next((deepcopy(dict(row)) for row in checks if row.get("passed") is not True), None)


def freeze_arc_roster() -> list[Json]:
    """Seal the successor episodes without starting any environment work."""

    return [
        {
            "episode_id": f"exp7570:{game}:seed:{seed}",
            "game": game,
            "seed": seed,
            "adapter_withheld": True,
            "policy_action_limit": 600,
            "induction_limit": 2,
            "request_token_ceiling": 4096,
            "sampler": "existing_live_default",
            "checkpoint_scope": "whole_episode",
            "supervisor_redirect_outcome": None,
            "disposition": "unstarted_exp7570_only",
        }
        for game in FROZEN_GAMES
        for seed in FROZEN_SEEDS
    ]


class _ControlVerdict:
    accuracy = 1.0
    cell_recall = 1.0
    change_accuracy = 1.0
    change_fidelity = 1.0
    correct_changed_cells = 1
    spurious_changed_cells = 0
    noop_hallucination_rate = 0.0


class _ControlVerifier:
    def score(self, _engine: Any) -> _ControlVerdict:
        return _ControlVerdict()


def _policy_shell(recorder: Any) -> E3AgentPolicy:
    policy = E3AgentPolicy.__new__(E3AgentPolicy)
    policy._decision_telemetry = recorder
    policy.two_sided_goal_contract = None
    policy.structured_evidence_memory = None
    policy.phase = "execute"
    policy.plan = []
    policy.pi = 0
    policy.transitions = []
    policy.induction_attempts = []
    policy.proposer = SimpleNamespace(last_generated_tokens=7, last_prompt_tokens=11)
    return policy


def _arm(recorder: telemetry.DecisionTelemetryRecorder, policy: E3AgentPolicy) -> str:
    recorder.begin_step(level_before=0, phase="induce")
    recorder.record_induction_decision(
        policy,
        stalled=True,
        won=False,
        decision=(True, "stall"),
        wall_time_s=0.001,
    )
    return recorder._armed_induction_attempt_ids[-1]


def _install_accepted_plan(
    recorder: telemetry.DecisionTelemetryRecorder,
    policy: E3AgentPolicy,
    engine_name: str,
    *,
    execute: bool,
) -> str:
    attempt_id = _arm(recorder, policy)
    engine = SimpleNamespace(name=engine_name)
    recorder.time_world_model_verification(_ControlVerifier(), engine, candidate_source=engine_name)

    def planner(_engine: Any, _done: Any, _start: Any, **_kwargs: Any) -> list[Json]:
        return [{"action": 1, "data": None}]

    plan = E3AgentPolicy._call_plan_in_model(
        policy,
        planner,
        engine,
        lambda _grid: False,
        [[0]],
        goal_energy_override=lambda _grid: 0.0,
    )
    policy.plan = list(plan)
    policy.pi = 0
    recorder.complete_induction(
        policy,
        {"reason": "stall", "planned": True, "plan_length": len(plan)},
        0.01,
    )
    if execute:
        move = E3AgentPolicy._next_plan_move(policy)
        recorder.record_policy_action(
            policy,
            proposed_move=move,
            selected_move=move,
            level_before=0,
            provenance="execute.plan_step",
        )
    return attempt_id


def _read_jsonl(path: Path) -> list[Json]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _parity_trace(recorder: Any) -> Json:
    policy = _policy_shell(recorder)
    calls = {"model": 0, "environment": 0}
    rng = random.Random(7562001)

    def planner(_engine: Any, _done: Any, _start: Any, **_kwargs: Any) -> list[Json]:
        calls["model"] += 1
        return [{"action": 2, "data": None}]

    plan = E3AgentPolicy._call_plan_in_model(
        policy,
        planner,
        SimpleNamespace(name="parity"),
        lambda _grid: False,
        [[0]],
        goal_energy_override=lambda _grid: 0.0,
    )
    policy.plan = list(plan)
    move = E3AgentPolicy._next_plan_move(policy)
    calls["environment"] += 1
    provenance = [{"branch": "execute.plan_step", "move": list(move)}]
    return {
        "actions": [list(move)],
        "model_calls": calls["model"],
        "environment_calls": calls["environment"],
        "existing_provenance": provenance,
        "rng_state_after": canonical_hash(rng.getstate()),
        "rng_draw_after": rng.random(),
    }


def run_scripted_controls(output_dir: Path) -> Json:
    """Drive accepted, rejected, replaced, parse, and censored real policy seams."""

    output_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []

    positive_path = output_dir / "positive.jsonl"
    positive = telemetry.DecisionTelemetryRecorder(
        "sb26", path=positive_path, episode_id="positive"
    )
    positive_policy = _policy_shell(positive)
    _install_accepted_plan(positive, positive_policy, "accepted-positive", execute=True)
    positive.begin_policy_step(positive_policy, SimpleNamespace(levels_completed=1))
    positive.observe_induction_progress(positive_policy, SimpleNamespace(levels_completed=1))
    positive.finish_episode(level_end=1, actions_used=1)
    paths.append(positive_path)

    rejected_path = output_dir / "rejected.jsonl"
    rejected = telemetry.DecisionTelemetryRecorder(
        "vc33", path=rejected_path, episode_id="rejected"
    )
    rejected_policy = _policy_shell(rejected)
    _arm(rejected, rejected_policy)
    rejected.time_world_model_verification(
        _ControlVerifier(), SimpleNamespace(name="rejected"), candidate_source="rejected"
    )
    rejected.complete_induction(
        rejected_policy,
        {"planned": False, "skipped": "world_model_accuracy_below_threshold"},
        0.01,
    )
    rejected.finish_episode(level_end=0)
    paths.append(rejected_path)

    replaced_path = output_dir / "replaced.jsonl"
    replaced = telemetry.DecisionTelemetryRecorder(
        "su15", path=replaced_path, episode_id="replaced"
    )
    replaced_policy = _policy_shell(replaced)
    _install_accepted_plan(replaced, replaced_policy, "replacement-a", execute=False)
    _install_accepted_plan(replaced, replaced_policy, "replacement-b", execute=False)
    replaced.finish_episode(level_end=0)
    paths.append(replaced_path)

    parse_path = output_dir / "parse.jsonl"
    parse = telemetry.DecisionTelemetryRecorder("g50t", path=parse_path, episode_id="parse")
    parse_policy = _policy_shell(parse)
    _arm(parse, parse_policy)
    parse.complete_induction(
        parse_policy,
        {"planned": False, "skipped": "proposer_failed", "proposer_note": "invalid JSON"},
        0.01,
    )
    parse.finish_episode(level_end=0)
    paths.append(parse_path)

    censored_path = output_dir / "censored.jsonl"
    censored = telemetry.DecisionTelemetryRecorder(
        "m0r0", path=censored_path, episode_id="censored"
    )
    censored_policy = _policy_shell(censored)
    _install_accepted_plan(censored, censored_policy, "accepted-censored", execute=True)
    censored.finish_episode(level_end=0, actions_used=1)
    paths.append(censored_path)

    rows = [row for path in paths for row in _read_jsonl(path)]
    parity_path = output_dir / "parity.jsonl"
    parity_enabled = telemetry.DecisionTelemetryRecorder(
        "dc22", path=parity_path, episode_id="parity"
    )
    trace_off = _parity_trace(telemetry.NOOP_RECORDER)
    trace_on = _parity_trace(parity_enabled)
    parity_enabled.finish_episode(level_end=0, actions_used=1)
    parity = {
        "actions_equal": trace_off["actions"] == trace_on["actions"],
        "model_calls_equal": trace_off["model_calls"] == trace_on["model_calls"],
        "environment_calls_equal": (
            trace_off["environment_calls"] == trace_on["environment_calls"]
        ),
        "existing_provenance_equal": (
            trace_off["existing_provenance"] == trace_on["existing_provenance"]
        ),
        "rng_state_equal": trace_off["rng_state_after"] == trace_on["rng_state_after"],
        "rng_draw_equal": trace_off["rng_draw_after"] == trace_on["rng_draw_after"],
        "off": trace_off,
        "on": trace_on,
    }
    parity["passed"] = all(
        parity[key]
        for key in (
            "actions_equal",
            "model_calls_equal",
            "environment_calls_equal",
            "existing_provenance_equal",
            "rng_state_equal",
            "rng_draw_equal",
        )
    )
    reduction = reduce_lineage_rows(rows)
    positive_terminal = next(
        row
        for row in rows
        if row.get("record_type") == "plan_lineage_terminal"
        and row.get("terminal_stage") == "executed_with_level_progress"
    )
    corruption_rejections: dict[str, bool] = {}
    expected = {
        "induction_attempt_id": "terminal_attempt_join_missing",
        "model_version": "terminal_model_join_missing",
        "plan_id": "terminal_plan_join_missing",
    }
    for field, error in expected.items():
        mutated = deepcopy(rows)
        target = next(
            row
            for row in mutated
            if row.get("record_type") == "plan_lineage_terminal"
            and row.get("induction_attempt_id") == positive_terminal.get("induction_attempt_id")
        )
        target[field] = f"corrupt:{field}"
        corruption_rejections[field] = error in reduce_lineage_rows(mutated)["errors"]
    return {
        "rows": rows,
        "row_hash": canonical_hash(rows),
        "case_paths": [path.as_posix() for path in paths],
        "raw_sidecars": [
            {"path": path.as_posix(), "sha256": sha256_file(path), "bytes": path.stat().st_size}
            for path in paths
        ],
        "case_count": 5,
        "attempt_count": len(reduction["terminal_rows"]),
        "real_call_sites": [
            "E3AgentPolicy._call_plan_in_model",
            "E3AgentPolicy._next_plan_move",
            "DecisionTelemetryRecorder.record_policy_action",
            "DecisionTelemetryRecorder.begin_policy_step",
        ],
        "policy_parity": parity,
        "corruption_rejections": corruption_rejections,
    }


def reduce_lineage_rows(rows: Sequence[Mapping[str, Any]]) -> Json:
    """Validate immutable joins and reduce mutually exclusive terminal stages."""

    values = [deepcopy(dict(row)) for row in rows]
    opportunities = {
        str(row.get("attempt_id"))
        for row in values
        if row.get("seam") == "induction_timing"
        and row.get("gate_decision") in {"induce_now", "reinduce_now"}
        and row.get("attempt_id")
    }
    accepted_models = {
        (
            str(row.get("attempt_id")),
            str(row.get("model_version")),
        )
        for row in values
        if row.get("seam") == "world_model_hypothesis_gate"
        and row.get("outcome") == "accept"
        and row.get("model_version")
    }
    plans = {
        (
            str(row.get("induction_attempt_id") or row.get("attempt_id")),
            str(row.get("model_version")),
            str(row.get("plan_id")),
        )
        for row in values
        if row.get("seam") in {"planner_invocation", "plan_registration"} and row.get("plan_id")
    }
    actions = {
        str(row.get("action_id")): row
        for row in values
        if row.get("seam") == "policy_action" and row.get("action_id")
    }
    transitions = {
        str(row.get("action_id")): row
        for row in values
        if row.get("seam") == "level_transition" and row.get("action_id")
    }
    terminals = [row for row in values if row.get("record_type") == "plan_lineage_terminal"]
    errors: list[str] = []
    terminal_ids: list[str] = []
    for terminal in terminals:
        attempt_id = str(terminal.get("induction_attempt_id") or "")
        terminal_ids.append(attempt_id)
        stage = terminal.get("terminal_stage")
        if stage not in telemetry.PLAN_LINEAGE_TERMINAL_STAGES:
            errors.append("terminal_stage_invalid")
        if attempt_id not in opportunities:
            errors.append("terminal_attempt_join_missing")
        model_version = terminal.get("model_version")
        if (
            model_version is not None
            and (
                attempt_id,
                str(model_version),
            )
            not in accepted_models
        ):
            errors.append("terminal_model_join_missing")
        plan_id = terminal.get("plan_id")
        if (
            plan_id is not None
            and (
                attempt_id,
                str(model_version),
                str(plan_id),
            )
            not in plans
        ):
            errors.append("terminal_plan_join_missing")
        for action_id in terminal.get("executed_action_ids") or []:
            action = actions.get(str(action_id))
            if action is None or (
                str(action.get("induction_attempt_id")),
                str(action.get("model_version")),
                str(action.get("plan_id")),
            ) != (attempt_id, str(model_version), str(plan_id)):
                errors.append("terminal_action_join_missing")
        if stage == "executed_with_level_progress":
            action_id = str(terminal.get("level_progress_action_id") or "")
            transition = transitions.get(action_id)
            if (
                action_id not in set(map(str, terminal.get("executed_action_ids") or []))
                or transition is None
                or int(transition.get("level_delta") or 0) <= 0
            ):
                errors.append("terminal_level_transition_join_missing")
        if str(stage).startswith("executed_") and not terminal.get("executed_action_ids"):
            errors.append("executed_stage_without_action")
        if stage == "planned_not_executed" and terminal.get("executed_action_ids"):
            errors.append("not_executed_stage_with_action")
    duplicate_ids = [key for key, count in Counter(terminal_ids).items() if key and count != 1]
    if duplicate_ids:
        errors.append("terminal_stage_not_mutually_exclusive")
    unknown = sorted(opportunities - set(terminal_ids))
    if unknown:
        errors.append("missing_instrumentation_unknown")
    stage_counts = dict(
        sorted(Counter(str(row.get("terminal_stage")) for row in terminals).items())
    )
    positive = sum(row.get("terminal_stage") == "executed_with_level_progress" for row in terminals)
    return {
        "valid": not errors and not unknown,
        "errors": sorted(set(errors)),
        "terminal_rows": terminals,
        "terminal_stage_counts": stage_counts,
        "positive_join_count": positive,
        "unknown_attempt_ids": unknown,
        "opportunity_count": len(opportunities),
        "terminal_count": len(terminals),
        "accepted_model_join_count": len(accepted_models),
        "plan_join_count": len(plans),
        "plan_action_join_count": sum(
            row.get("plan_linked") is True for row in values if row.get("seam") == "policy_action"
        ),
        "level_transition_join_count": sum(
            int(row.get("level_delta") or 0) > 0
            for row in values
            if row.get("seam") == "level_transition" and row.get("plan_linked") is True
        ),
    }


def _gate(
    check: str,
    category: str,
    expected: Any,
    observed: Any,
    passed: bool,
    principle: str,
    op: str = "==",
) -> Json:
    return {
        "check": check,
        "category": category,
        "expected": expected,
        "observed": observed,
        "op": op,
        "passed": bool(passed),
        "principle": principle,
    }


def _gate_summary(gates: Sequence[Mapping[str, Any]]) -> Json:
    failures = [deepcopy(dict(row)) for row in gates if row.get("passed") is not True]
    return {
        "all_passed": not failures,
        "failed_count": len(failures),
        "first_failure": failures[0] if failures else None,
        "failures": failures,
    }


def _receipts_pass(receipts: Sequence[Mapping[str, Any]]) -> bool:
    counts = Counter(str(row.get("name")) for row in receipts)
    return all(
        counts[name] == 1
        and next(row for row in receipts if row.get("name") == name).get("passed") is True
        and next(row for row in receipts if row.get("name") == name).get("exit_code") == 0
        for name in REQUIRED_RECEIPT_NAMES
    )


def _field_principles(keys: Sequence[str]) -> dict[str, Json]:
    specific = {
        "experiment_id": "Binds the exact task, milestone, and run date.",
        "preconditions_checked": "Prevents measurement over missing upstream evidence.",
        "MODEL_SPECS": "An empty list prevents a no-load run from implying model work.",
        "model_specs": "Resolved models are absent because this task loads none.",
        "model_invoked": "Typed zero calls prevent historical work from becoming current work.",
        "inference_substrate_class": "The actual substrate determines the duration floor.",
        "inference_substrate": "The runtime kind prevents false model-inference claims.",
        "execution_venue": "A legal venue value stays separate from CPU identity.",
        "duration_s": "Monotonic timing exposes implausibly short or padded work.",
        "random_seed": "Frozen seeds make ordering and parity controls reproducible.",
        "reproducibility_checksum": "The checksum binds sources, settings, code, and evidence.",
        "rows": "Absolute terminal rows keep missing outcomes from becoming zeros.",
        "sample_size_budget": "All attempted, censored, and unstarted units remain countable.",
        "acceptance_gate_results": "Completion cannot replace support or validity.",
        "gate_check_summary": "The first exact failure remains actionable.",
        "honest_verdict": "A terminal prefix gives the conductor an unambiguous state.",
        "verdict_class": "The closed class keeps fixture readiness separate from causal benefit.",
        "verifier_is_oracle": "Oracle fixtures cannot establish oracle-distinct benefit.",
        "flagged_adversarial": "A prior or current quarantine cannot be erased.",
        "validation_receipts": "Exact commands and log hashes make checks independently auditable.",
        "plan_lineage_ready_score": "Readiness needs reachability, joins, parity, and validation.",
        "join_schema": "Immutable IDs distinguish plan execution from incidental movement.",
        "per_game_results": "Scripted fixtures cannot become live game outcomes.",
        "solve_provenance": "Fixtures and repeated public levels claim no new solve.",
        "policy_parity": "Observation must preserve actions, calls, provenance, and RNG.",
        "frozen_arc_roster": "The future twelve episodes are fixed before outcomes exist.",
    }
    return {
        key: {
            "principle": specific.get(
                key, "This field keeps one terminal claim explicit and independently checkable."
            )
        }
        for key in keys
        if key != "field_principles"
    }


def reproducibility_checksum(artifact: Mapping[str, Any]) -> str:
    value = deepcopy(dict(artifact))
    value.pop("reproducibility_checksum", None)
    value.pop("field_principles", None)
    return canonical_hash(value)


def build_artifact(
    run_date: str,
    *,
    checks: Sequence[Mapping[str, Any]],
    controls: Mapping[str, Any],
    reduction: Mapping[str, Any],
    validation_receipts: Sequence[Mapping[str, Any]],
    duration_s: float,
    phase_spans: Sequence[Mapping[str, Any]],
    source_hashes: Mapping[str, str],
    terminal: bool,
) -> Json:
    """Build candidate or terminal evidence from independently reducible rows."""

    parity = deepcopy(dict(controls.get("policy_parity") or {}))
    corruptions = deepcopy(dict(controls.get("corruption_rejections") or {}))
    roster = freeze_arc_roster()
    receipts_ok = _receipts_pass(validation_receipts) if terminal else False
    expected_stages = {
        "censored": 2,
        "executed_with_level_progress": 1,
        "parse_rejected": 1,
        "planned_not_executed": 1,
        "verifier_rejected": 1,
    }
    gates = [
        _gate(
            "preconditions",
            "validity",
            True,
            all(row.get("passed") is True for row in checks),
            all(row.get("passed") is True for row in checks),
            "Missing upstream evidence must stop measurement.",
        ),
        _gate(
            "lineage_reduction",
            "validity",
            True,
            reduction.get("valid"),
            reduction.get("valid") is True,
            "Malformed joins cannot support readiness.",
        ),
        _gate(
            "positive_join_control",
            "readiness",
            1,
            reduction.get("positive_join_count"),
            reduction.get("positive_join_count") == 1,
            "A known executed plan must reach a joined level transition.",
        ),
        _gate(
            "scripted_terminal_stages",
            "readiness",
            expected_stages,
            reduction.get("terminal_stage_counts"),
            reduction.get("terminal_stage_counts") == expected_stages,
            "Accepted, rejected, replaced, parse, and censored controls must stay distinct.",
        ),
        _gate(
            "identifier_corruption_rejected",
            "validity",
            {"induction_attempt_id": True, "model_version": True, "plan_id": True},
            corruptions,
            corruptions == {"induction_attempt_id": True, "model_version": True, "plan_id": True},
            "A copied or corrupted ID must not create an attribution edge.",
        ),
        _gate(
            "policy_parity",
            "readiness",
            True,
            parity.get("passed"),
            parity.get("passed") is True,
            "Observation must preserve actions, calls, provenance, and RNG.",
        ),
        _gate(
            "frozen_roster",
            "readiness",
            12,
            len(roster),
            len(roster) == 12 and all(row["disposition"].startswith("unstarted") for row in roster),
            "Future outcomes must not alter the pre-registered roster.",
        ),
        _gate(
            "required_validation",
            "validity",
            True,
            receipts_ok if terminal else "pending",
            receipts_ok,
            "Only scoped checks and capability E2E can promote readiness.",
        ),
        _gate(
            "causal_benefit_not_claimed",
            "benefit",
            False,
            False,
            True,
            "Temporal lineage bookkeeping is not a causal treatment comparison.",
        ),
    ]
    ready = int(terminal and all(row["passed"] for row in gates))
    scripted = [
        {
            **deepcopy(dict(row)),
            "row_type": "scripted_lineage_attempt",
            "disposition": str(row.get("terminal_stage")),
        }
        for row in reduction.get("terminal_rows") or []
    ]
    frozen_rows = [{**deepcopy(row), "row_type": "frozen_future_episode"} for row in roster]
    artifact: Json = {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "run_date": run_date,
        "status": (
            "complete_circular_positive_plan_lineage_ready_fixture_only"
            if ready
            else "partial_terminal_validation_pending"
        ),
        "honest_verdict": (
            "complete_circular_positive_plan_lineage_ready_fixture_only"
            if ready
            else "partial_terminal_validation_pending"
        ),
        "verdict_class": "circular_positive" if ready else "partial",
        "positive_claim": False,
        "causal_benefit_claimed": False,
        "plan_lineage_ready_score": ready,
        "MODEL_SPECS": [],
        "model_specs": [],
        "model_invoked": False,
        "new_model_calls": 0,
        "invocation_counts": deepcopy(ZERO_INVOCATION_COUNTS),
        "historical_model_calls": {"exp7557": "preserved_as_upstream_not_current"},
        "planned_inference_substrate_class": "no_model_load",
        "inference_substrate_class": "no_model_load",
        "inference_substrate": "offline_arcade_live_agent_runtime_self_discovery_no_llm",
        "execution_venue": "host",
        "device_identity": {"kind": "cpu", "platform": os.uname().machine},
        "duration_s": float(duration_s),
        "phase_spans": [deepcopy(dict(row)) for row in phase_spans],
        "process_identity": {"pid": os.getpid(), "worktree_root": str(REPO_ROOT)},
        "random_seed": {
            "control_seed": 7562001,
            "ordering_seed": 7562002,
            "bootstrap_seed": 7562003,
        },
        "preconditions_checked": [deepcopy(dict(row)) for row in checks],
        "source_artifact_hashes": deepcopy(dict(source_hashes)),
        "code_hashes": {},
        "join_schema": {
            "keys": [
                "episode_id",
                "induction_attempt_id",
                "model_version",
                "plan_id",
                "action_id",
            ],
            "observation_window_policy_actions": 32,
            "terminal_stages": sorted(telemetry.PLAN_LINEAGE_TERMINAL_STAGES),
            "missing_instrumentation": "unknown",
            "causal_interpretation": "temporal_attribution_only",
        },
        "raw_control_rows": deepcopy(list(controls.get("rows") or [])),
        "raw_control_row_hash": controls.get("row_hash"),
        "raw_control_sidecars": deepcopy(list(controls.get("raw_sidecars") or [])),
        "scripted_results": scripted,
        "per_game_results": [],
        "frozen_arc_roster": roster,
        "rows": [*scripted, *frozen_rows],
        "sample_size_budget": {
            "planned": 6,
            "attempted": reduction.get("opportunity_count", 0),
            "completed": reduction.get("terminal_count", 0),
            "excluded": 0,
            "failed": 0,
            "censored": (reduction.get("terminal_stage_counts") or {}).get("censored", 0),
            "unstarted": 0,
        },
        "policy_parity": parity,
        "identifier_corruption_controls": corruptions,
        "real_call_site_reachability": deepcopy(list(controls.get("real_call_sites") or [])),
        "verifier_is_oracle": True,
        "flagged_adversarial": False,
        "solve_provenance": {
            "required_for_future_credit": "live_agent_self_discovery",
            "fixture_claim": "none",
            "new_solve_claimed": False,
            "credited_levels": 0,
            "registry_prechecked_games": list(FROZEN_GAMES),
        },
        "production_defaults_changed": False,
        "induction_eligibility_changed": False,
        "sampling_changed": False,
        "request_budget_changed": False,
        "model_selection_changed": False,
        "action_priority_changed": False,
        "generator_weights_changed": False,
        "game_source_read": False,
        "kernel_submitted": False,
        "roster_executed": False,
        "acceptance_gate_results": gates,
        "gate_check_summary": _gate_summary(gates),
        "validation_receipts": [deepcopy(dict(row)) for row in validation_receipts],
    }
    for path in (MODULE_PATH, WRAPPER_PATH, TELEMETRY_PATH, POLICY_PATH, TEST_PATH, SPEC_PATH):
        absolute = REPO_ROOT / path
        if absolute.is_file():
            artifact["code_hashes"][path.as_posix()] = sha256_file(absolute)
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    artifact["field_principles"] = _field_principles(tuple(artifact))
    return artifact


def build_blocked_artifact(
    run_date: str, checks: Sequence[Mapping[str, Any]], *, duration_s: float
) -> Json:
    """Publish external absence as blocked without inventing control evidence."""

    failure = first_failed_precondition(checks) or precondition_row(
        "exp7557_upstream_available",
        UPSTREAM_PATH.as_posix(),
        "path",
        "readable_file",
        None,
    )
    failure["category"] = "validity"
    artifact: Json = {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "run_date": run_date,
        "status": "complete_blocked_exp7557_not_ready",
        "honest_verdict": "complete_blocked_exp7557_not_ready",
        "verdict_class": "blocked",
        "positive_claim": False,
        "causal_benefit_claimed": False,
        "plan_lineage_ready_score": 0,
        "MODEL_SPECS": [],
        "model_specs": [],
        "model_invoked": False,
        "new_model_calls": 0,
        "invocation_counts": deepcopy(ZERO_INVOCATION_COUNTS),
        "historical_model_calls": {},
        "planned_inference_substrate_class": "no_model_load",
        "inference_substrate_class": "blocked_no_run",
        "inference_substrate": "blocked_no_run",
        "execution_venue": "host",
        "device_identity": {"kind": "cpu", "platform": os.uname().machine},
        "duration_s": float(duration_s),
        "phase_spans": [],
        "process_identity": {"pid": os.getpid(), "worktree_root": str(REPO_ROOT)},
        "random_seed": {
            "control_seed": 7562001,
            "ordering_seed": 7562002,
            "bootstrap_seed": 7562003,
        },
        "preconditions_checked": [deepcopy(dict(row)) for row in checks],
        "source_artifact_hashes": {},
        "code_hashes": {},
        "join_schema": {},
        "raw_control_rows": [],
        "raw_control_row_hash": None,
        "raw_control_sidecars": [],
        "scripted_results": [],
        "per_game_results": [],
        "frozen_arc_roster": freeze_arc_roster(),
        "rows": [],
        "sample_size_budget": {
            "planned": 6,
            "attempted": 0,
            "completed": 0,
            "excluded": 0,
            "failed": 0,
            "censored": 0,
            "unstarted": 6,
        },
        "policy_parity": {},
        "identifier_corruption_controls": {},
        "real_call_site_reachability": [],
        "verifier_is_oracle": True,
        "flagged_adversarial": False,
        "solve_provenance": {
            "required_for_future_credit": "live_agent_self_discovery",
            "fixture_claim": "none",
            "new_solve_claimed": False,
            "credited_levels": 0,
        },
        "production_defaults_changed": False,
        "roster_executed": False,
        "acceptance_gate_results": [failure],
        "gate_check_summary": {
            "all_passed": False,
            "failed_count": 1,
            "first_failure": deepcopy(failure),
            "failures": [deepcopy(failure)],
        },
        "validation_receipts": [],
    }
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    artifact["field_principles"] = _field_principles(tuple(artifact))
    return artifact


def independent_reduce(artifact: Mapping[str, Any]) -> Json:
    """Recompute readiness from raw control rows and validation receipts."""

    reduction = reduce_lineage_rows(artifact.get("raw_control_rows") or [])
    receipts_ok = _receipts_pass(artifact.get("validation_receipts") or [])
    parity = artifact.get("policy_parity") or {}
    corruptions = artifact.get("identifier_corruption_controls") or {}
    roster = artifact.get("frozen_arc_roster") or []
    ready = int(
        reduction["valid"]
        and reduction["positive_join_count"] == 1
        and reduction["terminal_stage_counts"]
        == {
            "censored": 2,
            "executed_with_level_progress": 1,
            "parse_rejected": 1,
            "planned_not_executed": 1,
            "verifier_rejected": 1,
        }
        and parity.get("passed") is True
        and corruptions == {"induction_attempt_id": True, "model_version": True, "plan_id": True}
        and len(roster) == 12
        and all(row.get("disposition") == "unstarted_exp7570_only" for row in roster)
        and receipts_ok
    )
    return {
        "plan_lineage_ready_score": ready,
        "required_validation_passed": receipts_ok,
        "lineage_valid": reduction["valid"],
        "lineage_errors": reduction["errors"],
        "positive_join_count": reduction["positive_join_count"],
        "terminal_stage_counts": reduction["terminal_stage_counts"],
        "terminal_count": reduction["terminal_count"],
        "unknown_attempt_ids": reduction["unknown_attempt_ids"],
        "policy_parity_passed": parity.get("passed") is True,
        "corruption_controls_passed": all(corruptions.values()) and len(corruptions) == 3,
        "frozen_roster_count": len(roster),
    }


def validate_artifact(artifact: Mapping[str, Any], *, require_terminal: bool) -> list[str]:
    """Reject identity, no-load accounting, join, roster, or receipt mutations."""

    errors: list[str] = []
    if artifact.get("experiment_id") != EXPERIMENT_ID:
        errors.append("experiment_id_mismatch")
    if artifact.get("milestone") != MILESTONE or artifact.get("run_date") != RUN_DATE:
        errors.append("experiment_identity_mismatch")
    honest_verdict = str(artifact.get("honest_verdict") or "")
    verdict_class = artifact.get("verdict_class")
    if verdict_class == "partial":
        if honest_verdict.startswith("complete_"):
            errors.append("partial_success_prefix_contradiction")
    elif not honest_verdict.startswith("complete_"):
        errors.append("terminal_prefix_missing")
    if artifact.get("verdict_class") not in {
        "positive",
        "circular_positive",
        "null",
        "blocked",
        "disqualified",
        "partial",
    }:
        errors.append("verdict_class_invalid")
    score = artifact.get("plan_lineage_ready_score")
    if type(score) is not int or score not in {0, 1}:
        errors.append("plan_lineage_ready_score_invalid")
    if artifact.get("MODEL_SPECS") != [] or artifact.get("model_specs") != []:
        errors.append("current_model_specs_not_empty")
    if artifact.get("model_invoked") is not False or artifact.get("new_model_calls") != 0:
        errors.append("current_model_invoked")
    if artifact.get("invocation_counts") != ZERO_INVOCATION_COUNTS:
        errors.append("current_invocation_counts_nonzero")
    if artifact.get("planned_inference_substrate_class") != "no_model_load":
        errors.append("planned_substrate_class_mismatch")
    if artifact.get("execution_venue") != "host":
        errors.append("execution_venue_invalid")
    if artifact.get("reproducibility_checksum") != reproducibility_checksum(artifact):
        errors.append("reproducibility_checksum_mismatch")
    principles = artifact.get("field_principles") or {}
    if set(artifact) - {"field_principles"} > set(principles):
        errors.append("field_principles_incomplete")
    if any(not row.get("principle") for row in artifact.get("acceptance_gate_results") or []):
        errors.append("gate_principles_incomplete")
    if artifact.get("verdict_class") == "blocked":
        if score != 0 or artifact.get("inference_substrate_class") != "blocked_no_run":
            errors.append("blocked_classification_mismatch")
        failure = (artifact.get("gate_check_summary") or {}).get("first_failure")
        if not isinstance(failure, Mapping):
            errors.append("blocked_gate_summary_missing")
        return sorted(set(errors))
    if artifact.get("inference_substrate_class") != "no_model_load":
        errors.append("inference_substrate_class_mismatch")
    if artifact.get("inference_substrate") != (
        "offline_arcade_live_agent_runtime_self_discovery_no_llm"
    ):
        errors.append("inference_substrate_mismatch")
    if artifact.get("verifier_is_oracle") is not True:
        errors.append("fixture_oracle_declaration_missing")
    if (
        artifact.get("positive_claim") is not False
        or artifact.get("causal_benefit_claimed") is not False
    ):
        errors.append("causal_benefit_overclaim")
    if artifact.get("raw_control_row_hash") != canonical_hash(
        artifact.get("raw_control_rows") or []
    ):
        errors.append("raw_control_hash_mismatch")
    replay = independent_reduce(artifact)
    if replay["lineage_valid"] is not True:
        errors.append("lineage_reduction_invalid")
    if replay["terminal_count"] != 6:
        errors.append("scripted_attempt_count_mismatch")
    if artifact.get("per_game_results") != [] or artifact.get("roster_executed") is not False:
        errors.append("fixture_promoted_to_game_outcome")
    if (artifact.get("solve_provenance") or {}).get("new_solve_claimed") is not False:
        errors.append("fixture_promoted_to_solve")
    if len(artifact.get("frozen_arc_roster") or []) != 12:
        errors.append("frozen_roster_mismatch")
    if require_terminal:
        if replay["required_validation_passed"] is not True:
            errors.append("required_validation_failed")
        if score != replay["plan_lineage_ready_score"] or score != 1:
            errors.append("terminal_readiness_mismatch")
        if artifact.get("verdict_class") != "circular_positive":
            errors.append("terminal_verdict_class_mismatch")
    elif score != 0 or artifact.get("verdict_class") != "partial":
        errors.append("candidate_classification_mismatch")
    return sorted(set(errors))


def cold_replay(path: Path) -> Json:
    """Reload a candidate in a new process and repeat its graph reduction."""

    artifact = load_json(path)
    require_terminal = artifact.get("plan_lineage_ready_score") == 1
    errors = validate_artifact(artifact, require_terminal=require_terminal)
    if errors:
        raise ValueError("cold_replay_invalid:" + ",".join(errors))
    return independent_reduce(artifact)


def e2e_commands(root: Path, private: Path) -> list[validation_scope.CommandSpec]:
    """Declare E2E-011/012/013 and the private LLM-off environment smoke."""

    private.mkdir(parents=True, exist_ok=True)
    pytest = str(root / ".venv/bin/pytest")
    python = str(root / ".venv/bin/python")
    common = ("-n", "0", "-o", "addopts=", "--no-cov")
    return [
        validation_scope.CommandSpec(
            "e2e_011",
            (
                pytest,
                *common,
                f"--basetemp={private / 'e2e011'}",
                "tests/python/test_arc_decision_telemetry.py",
                "-q",
            ),
            "E2E-011 ARC decision telemetry parity",
        ),
        validation_scope.CommandSpec(
            "e2e_012",
            (
                pytest,
                *common,
                f"--basetemp={private / 'e2e012'}",
                "tests/python/test_experiment_7491_e6_timed_live_profile.py",
                "tests/python/test_experiment_7492_e6_timed_cost_profile.py",
                "tests/python/test_arc_decision_telemetry.py",
                "-q",
            ),
            "E2E-012 exclusive timing parity",
        ),
        validation_scope.CommandSpec(
            "e2e_013",
            (
                pytest,
                *common,
                f"--basetemp={private / 'e2e013'}",
                "tests/python/test_arc_decision_telemetry.py",
                "tests/python/test_experiment_7491_e6_timed_live_profile.py",
                "tests/python/test_experiment_7531_b2_induction_gate_measurement.py",
                "tests/python/test_semif_arc_readout_eval.py",
                "-q",
            ),
            "E2E-013 induction-attempt outcome telemetry",
        ),
        validation_scope.CommandSpec(
            "llm_off_environment_smoke",
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
                str(private / "r11l-smoke.json"),
            ),
            "private LLM-off real-environment smoke",
        ),
    ]


def terminal_commands(root: Path, candidate: Path) -> list[validation_scope.CommandSpec]:
    """Declare the thin entrypoint, cold replay, reducer, and strict readers."""

    python = str(root / ".venv/bin/python")
    wrapper = WRAPPER_PATH.as_posix()
    return [
        validation_scope.CommandSpec(
            "declared_entrypoint",
            (python, "-u", wrapper, "--date", RUN_DATE, "--validate", str(candidate)),
            "declared capability entrypoint read-only mode",
        ),
        validation_scope.CommandSpec(
            "fresh_process_cold_replay",
            (python, "-u", wrapper, "--date", RUN_DATE, "--cold-replay", str(candidate)),
            "fresh-process cold replay",
        ),
        validation_scope.CommandSpec(
            "independent_reduction",
            (
                python,
                "-u",
                "-c",
                (
                    "import json,sys; "
                    "from carnot.experiment_7562_v661_arc_plan_lineage import "
                    "independent_reduce,load_json; "
                    "print(json.dumps(independent_reduce(load_json(__import__('pathlib').Path(sys.argv[1]))),sort_keys=True))"
                ),
                str(candidate),
            ),
            "independent raw-row reduction",
        ),
        validation_scope.CommandSpec(
            "adversarial_verify",
            (python, "-u", "scripts/adversarial_verify.py", str(candidate)),
            "exact terminal candidate",
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
            "exact terminal candidate",
        ),
    ]


def _span(phase: str, phase_started: float, run_started: float) -> Json:
    ended = time.monotonic()
    return {
        "phase": phase,
        "start_s": phase_started - run_started,
        "end_s": ended - run_started,
        "duration_s": ended - phase_started,
    }


def run_experiment(  # pragma: no cover - exercised by the declared entrypoint.
    root: Path, run_date: str, *, output_path: Path = RESULT_PATH
) -> Json:
    """Check inputs, run controls and scoped validation, then publish atomically."""

    if run_date != RUN_DATE:
        raise ValueError(f"run_date_mismatch:{run_date}")
    started = time.monotonic()
    spans: list[Json] = []
    progress(started, "preconditions", "start")
    phase_started = time.monotonic()
    checks = collect_preconditions(root)
    spans.append(_span("preconditions", phase_started, started))
    failure = first_failed_precondition(checks)
    progress(started, "preconditions", "end", passed=failure is None)
    if failure is not None:
        blocked = build_blocked_artifact(run_date, checks, duration_s=time.monotonic() - started)
        progress(started, "publication", "before_atomic_blocked", path=output_path)
        atomic_json(root / output_path, blocked)
        progress(started, "publication", "after_atomic_blocked", path=output_path)
        return blocked

    raw_root = root / "results/raw/experiment_7562_v661_arc_plan_lineage"
    run_raw = raw_root / f"run-{os.getpid()}"
    progress(started, "scripted_controls", "start", planned_attempts=6)
    phase_started = time.monotonic()
    controls = run_scripted_controls(run_raw / "scripted")
    reduction = reduce_lineage_rows(controls["rows"])
    spans.append(_span("scripted_controls", phase_started, started))
    progress(
        started,
        "scripted_controls",
        "end",
        completed_attempts=reduction["terminal_count"],
        errors=len(reduction["errors"]),
    )
    if reduction["errors"]:
        raise RuntimeError("lineage_control_reduction_failed:" + ",".join(reduction["errors"]))

    private_root = Path(tempfile.mkdtemp(prefix="exp7562-validation-", dir="/tmp"))
    basetemp = private_root / "pytest"
    basetemp.mkdir(parents=True, exist_ok=True)
    coverage_file = private_root / ".coverage.exp7562"
    affected_commands = validation_scope.build_scoped_commands(
        root,
        AFFECTED_MANIFEST.test_paths,
        AFFECTED_MANIFEST.changed_modules,
        static_paths=AFFECTED_MANIFEST.static_paths,
        basetemp=basetemp,
        coverage_file=coverage_file,
    )
    progress(started, "affected_validation", "before_subprocesses", units=len(affected_commands))
    phase_started = time.monotonic()
    affected = validation_scope.run_commands(
        root,
        affected_commands,
        log_dir=run_raw / "validation/affected",
        extra_env={"COVERAGE_FILE": str(coverage_file)},
    )
    spans.append(_span("affected_validation", phase_started, started))
    progress(
        started,
        "affected_validation",
        "after_subprocesses",
        passed=all(row["passed"] for row in affected),
    )
    if not all(row["passed"] for row in affected):
        raise RuntimeError("scoped_validation_failed")

    capability_commands = e2e_commands(root, private_root / "capability")
    progress(started, "capability_e2e", "before_subprocesses", units=len(capability_commands))
    phase_started = time.monotonic()
    capability = validation_scope.run_commands(
        root,
        capability_commands,
        log_dir=run_raw / "validation/capability",
    )
    spans.append(_span("capability_e2e", phase_started, started))
    progress(
        started,
        "capability_e2e",
        "after_subprocesses",
        passed=all(row["passed"] for row in capability),
    )
    if not all(row["passed"] for row in capability):
        raise RuntimeError("capability_e2e_failed")

    source_hashes = {
        UPSTREAM_PATH.as_posix(): sha256_file(root / UPSTREAM_PATH),
        SPEC_PATH.as_posix(): sha256_file(root / SPEC_PATH),
        "ops/arc_solve_registry.yaml": sha256_file(root / "ops/arc_solve_registry.yaml"),
    }
    candidate = build_artifact(
        run_date,
        checks=checks,
        controls=controls,
        reduction=reduction,
        validation_receipts=[*affected, *capability],
        duration_s=time.monotonic() - started,
        phase_spans=spans,
        source_hashes=source_hashes,
        terminal=False,
    )
    candidate_errors = validate_artifact(candidate, require_terminal=False)
    if candidate_errors:
        raise RuntimeError("candidate_invalid:" + ",".join(candidate_errors))
    candidate_path = run_raw / "terminal_candidate.json"
    atomic_json(candidate_path, candidate)

    readers = terminal_commands(root, candidate_path)
    progress(started, "terminal_validation", "before_subprocesses", units=len(readers))
    phase_started = time.monotonic()
    terminal = validation_scope.run_commands(
        root,
        readers,
        log_dir=run_raw / "validation/terminal",
    )
    spans.append(_span("terminal_validation", phase_started, started))
    progress(
        started,
        "terminal_validation",
        "after_subprocesses",
        passed=all(row["passed"] for row in terminal),
    )
    if not all(row["passed"] for row in terminal):
        raise RuntimeError("terminal_validation_failed")

    final = build_artifact(
        run_date,
        checks=checks,
        controls=controls,
        reduction=reduction,
        validation_receipts=[*affected, *capability, *terminal],
        duration_s=time.monotonic() - started,
        phase_spans=spans,
        source_hashes=source_hashes,
        terminal=True,
    )
    final_errors = validate_artifact(final, require_terminal=True)
    if final_errors:
        raise RuntimeError("terminal_artifact_invalid:" + ",".join(final_errors))
    encoded_size = len(json.dumps(final, sort_keys=True).encode())
    if encoded_size >= 20 * 1024 * 1024:
        raise RuntimeError(f"terminal_artifact_too_large:{encoded_size}")
    progress(started, "publication", "before_atomic_terminal", path=output_path)
    atomic_json(candidate_path, final)
    atomic_json(root / output_path, final)
    progress(started, "publication", "after_atomic_terminal", path=output_path)
    return final


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the thin public entrypoint and its read-only replay modes."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", required=True)
    parser.add_argument("--output", type=Path, default=RESULT_PATH)
    parser.add_argument("--cold-replay", type=Path)
    parser.add_argument("--validate", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - thin CLI.
    """Dispatch the experiment or one read-only fresh-process reducer."""

    args = parse_args(argv)
    if args.cold_replay is not None:
        print(json.dumps(cold_replay(args.cold_replay), sort_keys=True), flush=True)
        return 0
    if args.validate is not None:
        artifact = load_json(args.validate)
        require_terminal = artifact.get("plan_lineage_ready_score") == 1
        errors = validate_artifact(artifact, require_terminal=require_terminal)
        if errors:
            raise ValueError("artifact_invalid:" + ",".join(errors))
        print(json.dumps(independent_reduce(artifact), sort_keys=True), flush=True)
        return 0
    run_experiment(REPO_ROOT, args.date, output_path=args.output)
    return 0


if __name__ == "__main__":  # pragma: no cover - module execution convenience.
    raise SystemExit(main())
