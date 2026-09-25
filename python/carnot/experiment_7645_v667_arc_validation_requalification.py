"""REQ-REPORT-7645: CPU requalification of the live ARC goal guard."""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import socket
import tempfile
import time
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

import numpy as np

from carnot import experiment_7639_v666_arc_goal_dedup as prior
from carnot.agentic import arc_competition_agent as agent
from carnot.agentic import arc_executable_world_model as e3
from carnot.reporting.current_work_receipt import (
    ZERO_INVOCATION_COUNTS,
    atomic_json,
    canonical_hash,
    sha256_file,
)
from carnot.reporting.experiment_7303_validation_scope import (
    CommandSpec,
    build_scoped_commands,
    run_commands,
)

RUN_DATE = "20260925"
RESULT = Path("results/experiment_7645_v667_arc_validation_requalification.json")
RAW = Path("results/raw/experiment_7645_v667_arc_validation_requalification")
MODULE = Path("python/carnot/experiment_7645_v667_arc_validation_requalification.py")
TEST = Path("tests/python/test_experiment_7645_v667_arc_validation_requalification.py")
WRAPPER = Path("scripts/experiments/experiment_7645_v667_arc_validation_requalification.py")
SPEC = Path("openspec/capabilities/research-reporting/spec.md")
OLD_RESULT = Path("results/experiment_7639_v666_arc_goal_dedup.json")
MODEL_SPECS: list[dict[str, Any]] = []
FIELD_PRINCIPLES = {
    **prior.FIELD_PRINCIPLES,
    "goal_guard_rows": "Exact source behavior is readiness evidence, not game benefit.",
    "execution_venue_details": "Host and PID metadata cannot expand the venue enum.",
    "sample_size_budget": "One constructed state pair is one independent fixture group.",
}


def progress(started: float, phase: str, event: str, **detail: Any) -> None:
    """Print a flushed boundary with elapsed monotonic time."""
    suffix = " ".join(f"{key}={value}" for key, value in sorted(detail.items()))
    print(
        f"[exp7645] phase={phase} event={event} elapsed_s={time.monotonic() - started:.3f} {suffix}".rstrip(),
        flush=True,
    )


def _row(
    unit: str,
    passed: bool,
    goal_calls: list[Any],
    diagnostics: Mapping[str, Any],
    mask: np.ndarray,
    route: str,
) -> dict[str, Any]:
    """Retain absolute operands for one independently constructed state pair."""
    return {
        "unit_id": unit,
        "arm": "CPU_GOAL_GUARD",
        "unit_kind": "exact_regression_fixture",
        "route": route,
        "passed": bool(passed),
        "goal_evaluations": len(goal_calls),
        "goal_inputs": goal_calls,
        "dedup_decisions": int(diagnostics.get("hud_dedup_states_merged", 0)),
        "mask_shape": list(mask.shape),
        "mask_cells": int(np.count_nonzero(mask)),
        "numerator": int(passed),
        "denominator": 1,
        "exclusion": None,
        "censored": False,
        "raw_provenance": "current_source_exact_constructed_state_pair",
        "diagnostics": dict(diagnostics),
    }


def measure_goal_guard_rows() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Exercise the shipped planner and live Stage-2 mask without a model call."""
    cases = (
        ("empty_mask", [[False, False]], [[0, 7]], True),
        ("full_mask", [[True, True]], [[0, 7]], True),
        ("terminal_alias", [[False, True]], [[0, 1], [0, 7]], True),
        ("true_alias", [[False, True]], [[0, 1], [0, 2]], False),
        ("ordinary_duplicate", [[False, False]], [[0, 1], [0, 1]], False),
    )
    rows: list[dict[str, Any]] = []
    original = e3._model_candidates
    try:
        for name, raw_mask, states, expected in cases:
            count = len(states)
            e3._model_candidates = lambda _grid: [  # type: ignore[assignment]
                {"action": index + 1, "data": None} for index in range(count)
            ]
            mask = np.asarray(raw_mask, dtype=bool)
            calls: list[Any] = []
            diagnostics: dict[str, Any] = {}

            def engine(_grid: np.ndarray, action: int, _data: Any) -> np.ndarray:
                return np.asarray(states[action - 1], dtype=np.int16).reshape(1, 2)

            def goal(grid: np.ndarray) -> bool:
                calls.append(np.asarray(grid).tolist())
                return int(grid[0, 1]) == 7

            plan = e3.plan_in_model(
                engine,
                goal,
                np.zeros((1, 2), dtype=np.int16),
                dedup_mask=mask,
                diagnostics=diagnostics,
                max_depth=1,
            )
            passed = bool(plan) is expected and (
                name != "terminal_alias" or calls[:2] == [[[0, 1]], [[0, 7]]]
            )
            rows.append(_row(name, passed, calls, diagnostics, mask, "plan_in_model"))
    finally:
        e3._model_candidates = original  # type: ignore[assignment]

    explorer = agent.StepwiseExplorer(edge_bar_hud_mask=True)
    for index in range(30):
        grid = np.full((64, 64), 3, dtype=np.int16)
        grid[:, 0] = 0
        grid[:index, 0] = 5
        grid[30:34, 30:34] = 7
        explorer.awaiting = None
        explorer._ingest(SimpleNamespace(frame=grid, state="NOT_FINISHED", levels_completed=0))
    frame_mask = explorer.hud_mask
    if frame_mask is None:
        raise RuntimeError("stage2_mask_unresolved")
    policy = object.__new__(agent.E3AgentPolicy)
    policy.two_sided_goal_contract = None
    policy.explorer = explorer
    policy.cell = 1
    start = np.zeros((64, 64), dtype=np.int16)
    observed = start.copy()
    observed[20, 20] = 1
    policy.transitions = [prior._transition(start, observed)]
    policy._episode_transition_start = 0
    goal_calls: list[Any] = []
    diagnostics = {}
    old_flag = os.environ.get("CARNOT_ARC_PLAN_HUD_DEDUP")
    os.environ["CARNOT_ARC_PLAN_HUD_DEDUP"] = "1"
    try:

        def wrapper_goal(grid: np.ndarray) -> bool:
            goal_calls.append(int(grid[0, 0]))
            return int(grid[0, 0]) == 7

        def wrapper_engine(grid: np.ndarray, _action: int, _data: Any) -> np.ndarray:
            after = grid.copy()
            after[0, 0] = 7
            return after

        e3._model_candidates = lambda _grid: [{"action": 1, "data": None}]  # type: ignore[assignment]
        plan = policy._call_plan_in_model(
            e3.plan_in_model,
            wrapper_engine,
            wrapper_goal,
            start,
            diagnostics=diagnostics,
            goal_energy_override=lambda _grid: 1.0,
        )
    finally:
        e3._model_candidates = original  # type: ignore[assignment]
        if old_flag is None:
            os.environ.pop("CARNOT_ARC_PLAN_HUD_DEDUP", None)
        else:
            os.environ["CARNOT_ARC_PLAN_HUD_DEDUP"] = old_flag
    logical = e3.logical_hud_mask(frame_mask, 1)
    assert logical is not None
    passed = (
        bool(plan)
        and goal_calls == [7]
        and diagnostics.get("planner_hud_dedup_mask_status") == "applied"
    )
    rows.append(
        _row(
            "live_stage2_mask",
            passed,
            goal_calls,
            diagnostics,
            logical,
            "E3AgentPolicy._call_plan_in_model",
        )
    )
    stage2 = {
        "frame_shape": list(frame_mask.shape),
        "logical_shape": list(logical.shape),
        "mask_cells": int(np.count_nonzero(logical)),
        "source": explorer._hud_mask_source,
        "stage2_verdict": explorer.hud_mask_diagnostics()["stage2"]["stage2_verdict"],
    }
    return rows, stage2


def independent_reduce(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Count constructed cases once each, irrespective of views or arms."""
    units = {str(row["unit_id"]) for row in rows}
    return {
        "independent_groups": len(units),
        "rows": len(rows),
        "passed_groups": len({str(row["unit_id"]) for row in rows if row.get("passed") is True}),
        "excluded_groups": len({str(row["unit_id"]) for row in rows if row.get("exclusion")}),
        "censored_groups": len({str(row["unit_id"]) for row in rows if row.get("censored")}),
    }


def _gate(passed: bool, principle: str, operands: Any) -> dict[str, Any]:
    return {"passed": bool(passed), "principle": principle, "measured_operands": operands}


def build_artifact(
    rows: Sequence[Mapping[str, Any]],
    stage2: Mapping[str, Any],
    preconditions: Sequence[Mapping[str, Any]],
    spans: Sequence[Mapping[str, Any]],
    source_hashes: Mapping[str, Any],
    duration_s: float,
    receipts: Sequence[Mapping[str, Any]],
    *,
    flagged_adversarial: bool = False,
) -> dict[str, Any]:
    """Separate terminal completion, validation readiness, and scientific benefit."""
    failed = next((check for check in preconditions if check.get("passed") is not True), None)
    measured = list(rows)
    reduction = independent_reduce(measured)
    valid = bool(measured) and all(row.get("passed") is True for row in measured)
    validated = bool(receipts) and all(row.get("passed") is True for row in receipts)
    ready = valid and validated and not flagged_adversarial and failed is None
    if failed is not None:
        verdict_class = "blocked"
        verdict = f"complete_blocked_{failed['check']}"
        gate_summary = {
            key: failed.get(key)
            for key in ("check", "upstream", "path", "field", "operator", "expected", "observed")
        }
    elif not validated or not valid or flagged_adversarial:
        verdict_class = "disqualified"
        verdict = "complete_disqualified_arc_validation_failed"
        gate_summary = {
            "failed_validation": [
                row.get("name") for row in receipts if row.get("passed") is not True
            ],
            "failed_fixtures": [
                row.get("unit_id") for row in measured if row.get("passed") is not True
            ],
            "flagged_adversarial": flagged_adversarial,
        }
    else:
        verdict_class = "null"
        verdict = "complete_null_arc_goal_guard_ready_no_hidden_game_benefit"
        gate_summary = {"readiness_only": True, "independent_hidden_game_groups": 0}
    gates = {
        "validity": _gate(
            valid and failed is None and not flagged_adversarial,
            "Exact source and authentic inputs are necessary for validity.",
            {
                "passing_fixtures": sum(row.get("passed") is True for row in measured),
                "fixtures": len(measured),
            },
        ),
        "readiness": _gate(
            ready,
            "All scoped checks and terminal readers must pass.",
            {
                "validation_exits": [row.get("exit_code") for row in receipts],
                "flagged": flagged_adversarial,
            },
        ),
        "probability_benefit": _gate(
            False,
            "Only independent hidden-game outcomes estimate probability benefit.",
            {"hidden_game_groups": 0},
        ),
        "utility": _gate(
            False,
            "Measured game benefit and cost are required for utility.",
            {"benefit_measured": False, "game_cost_measured": False},
        ),
        "retention": _gate(
            False,
            "Fixture replay cannot show retained learned improvement.",
            {"retained_learning_groups": 0},
        ),
        "freshness": _gate(
            bool(source_hashes) and validated,
            "Current code and command receipts bind fresh qualification.",
            {
                "source_files": len(source_hashes.get("producer_files", {})),
                "receipts": len(receipts),
            },
        ),
    }
    value: dict[str, Any] = {
        "schema": "carnot.experiment_7645_v667_arc_validation_requalification.v1",
        "experiment_id": 7645,
        "milestone": "2026.09.667",
        "run_date": RUN_DATE,
        "requirement_id": "REQ-REPORT-7645",
        "honest_verdict": verdict,
        "verdict_class": verdict_class,
        "flagged_adversarial": flagged_adversarial,
        "gate_check_summary": gate_summary,
        "acceptance_gate_results": gates,
        "rows": deepcopy(measured),
        "goal_guard_rows": deepcopy(measured),
        "stage2_mask": dict(stage2),
        "independent_reduction": reduction,
        "sample_size_budget": {
            "intended_independent_groups": 6,
            "observed_independent_groups": reduction["independent_groups"],
            "eligible_groups": reduction["independent_groups"] - reduction["excluded_groups"],
            "excluded_groups": reduction["excluded_groups"],
            "censored_groups": reduction["censored_groups"],
            "exposure_limit": "constructed CPU fixtures only; no live game exposure",
            "repeated_seeds_views_and_orderings_increase_sample_size": False,
        },
        "preconditions_checked": deepcopy(list(preconditions)),
        "inference_substrate": "current_host_cpu_planner_and_stage2_replay",
        "inference_substrate_class": "no_model_load",
        "planned_inference_substrate_class": "no_model_load",
        "MODEL_SPECS": MODEL_SPECS,
        "planned_MODEL_SPECS": [],
        "historical_models": ["Qwen3.8-27B-GGUF"],
        "model_invoked": False,
        "invocation_counts": {
            **ZERO_INVOCATION_COUNTS,
            "forward_calls_attempted": 0,
            "forward_calls_completed": 0,
            "input_tokens": 0,
            "output_tokens": 0,
        },
        "execution_venue": "host",
        "execution_venue_details": {
            "host": socket.gethostname(),
            "owned_pid": os.getpid(),
            "device_uuid": None,
        },
        "execution_host": socket.gethostname(),
        "execution_owned_pid": os.getpid(),
        "execution_device_uuid": None,
        "phase_spans": deepcopy(list(spans)),
        "duration_s": round(duration_s, 6),
        "random_seed": {"fixture_order": 7645, "stochastic_operations": "none"},
        "source_artifact_hashes": deepcopy(dict(source_hashes)),
        "validation_receipts": deepcopy(list(receipts)),
        "verifier_is_oracle": True,
        "field_principles": FIELD_PRINCIPLES,
        "planner_goal_guard_ready_score": int(ready),
        "solve_provenance": "development_proxy",
        "new_game_level_solve_credit": False,
        "production_defaults_changed": False,
        "external_publication_authorized": False,
        "generator_training_attempted": False,
        "reproducibility_checksum": "",
    }
    value["reproducibility_checksum"] = canonical_hash(value)
    return value


def collect_preconditions(root: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Authenticate actual inputs; the planned result is never an input."""
    files = (
        Path("AGENTS.md"),
        Path("CLAUDE.md"),
        Path("CODEX.md"),
        Path("research-program.md"),
        Path("ops/exclusion_manifest.yaml"),
        Path("ops/e2e-test-plan.md"),
        Path("scripts/experiment_template.py"),
        Path("python/carnot/reporting/current_work_receipt.py"),
        Path("python/carnot/reporting/experiment_7303_validation_scope.py"),
        Path("python/carnot/agentic/arc_executable_world_model.py"),
        Path("python/carnot/agentic/arc_competition_agent.py"),
        Path("ops/arc_solve_registry.yaml"),
        MODULE,
        TEST,
        WRAPPER,
        SPEC,
        OLD_RESULT,
    )
    checks: list[dict[str, Any]] = []
    hashes: dict[str, Any] = {
        "producer_files": {},
        "pre_gate_receipts": {},
        "missing_inputs": [],
        "planned_outputs_not_inputs": [RESULT.as_posix()],
    }
    for relative in files:
        path = root / relative
        passed = path.is_file() and path.stat().st_uid == os.getuid()
        checks.append(
            {
                "check": "named_input_authentication",
                "upstream": "worktree",
                "path": relative.as_posix(),
                "field": "is_file_and_current_user_owned",
                "operator": "==",
                "expected": True,
                "observed": passed,
                "passed": passed,
            }
        )
        if passed:
            hashes["producer_files"][relative.as_posix()] = sha256_file(path)
        else:
            hashes["missing_inputs"].append(relative.as_posix())
    requirement = (root / SPEC).is_file() and "REQ-REPORT-7645" in (root / SPEC).read_text(
        encoding="utf-8"
    )
    checks.append(
        {
            "check": "capability_requirement",
            "upstream": "OpenSpec",
            "path": SPEC.as_posix(),
            "field": "REQ-REPORT-7645",
            "operator": "contains",
            "expected": True,
            "observed": requirement,
            "passed": requirement,
        }
    )
    checks.append(
        {
            "check": "model_call_declaration",
            "upstream": "current_task",
            "path": MODULE.as_posix(),
            "field": "MODEL_SPECS",
            "operator": "==",
            "expected": [],
            "observed": MODEL_SPECS,
            "passed": MODEL_SPECS == [],
        }
    )
    return checks, hashes


def affected_validation_manifest() -> dict[str, Any]:
    """Freeze affected files before validation starts."""
    return {
        "tests": [TEST.as_posix()],
        "changed_modules": [MODULE.as_posix()],
        "static_paths": [WRAPPER.as_posix()],
        "coverage_required_percent": 100,
        "serial": True,
        "repository_addopts_disabled": True,
    }


def build_validation_commands(root: Path, private: Path) -> list[CommandSpec]:
    """Create pytest's parent before the scoped builder emits child paths."""
    (private / "pytest").mkdir(parents=True, exist_ok=True)
    (private / "coverage").mkdir(parents=True, exist_ok=True)
    return build_scoped_commands(
        root,
        [TEST.as_posix()],
        [MODULE.as_posix()],
        static_paths=[WRAPPER.as_posix()],
        basetemp=private / "pytest",
        coverage_file=private / "coverage/.coverage",
    )


def validate_artifact(value: Mapping[str, Any]) -> list[str]:
    """Check the closed venue, independent rows, and immutable checksum."""
    required = {
        "honest_verdict",
        "verdict_class",
        "goal_guard_rows",
        "rows",
        "acceptance_gate_results",
        "gate_check_summary",
        "sample_size_budget",
        "preconditions_checked",
        "inference_substrate",
        "inference_substrate_class",
        "MODEL_SPECS",
        "phase_spans",
        "source_artifact_hashes",
        "validation_receipts",
        "field_principles",
        "planner_goal_guard_ready_score",
        "solve_provenance",
        "reproducibility_checksum",
    }
    errors = sorted(required - value.keys())
    if not str(value.get("honest_verdict", "")).startswith("complete_"):
        errors.append("honest_verdict")
    if value.get("execution_venue") != "host":
        errors.append("execution_venue")
    if value.get("MODEL_SPECS") != [] or value.get("model_invoked") is not False:
        errors.append("model_declaration")
    if value.get("rows") != value.get("goal_guard_rows"):
        errors.append("goal_guard_rows")
    if independent_reduce(value.get("rows", [])) != value.get("independent_reduction"):
        errors.append("independent_reduction")
    copied = deepcopy(dict(value))
    observed = copied.get("reproducibility_checksum")
    copied["reproducibility_checksum"] = ""
    if observed != canonical_hash(copied):
        errors.append("reproducibility_checksum")
    return errors


def cold_replay(path: Path) -> list[str]:
    """Read exact candidate bytes and recompute independent row totals."""
    return validate_artifact(json.loads(path.read_text(encoding="utf-8")))


def build_terminal_commands(root: Path, candidate: Path) -> list[CommandSpec]:
    """Bind every reader to the same immutable private candidate."""
    python = str(root / ".venv/bin/python")
    wrapper = str(root / WRAPPER)
    return [
        CommandSpec(
            "declared_entrypoint_validate",
            (python, "-u", wrapper, "--date", RUN_DATE, "--validate", str(candidate)),
            "exact_candidate",
            300,
        ),
        CommandSpec(
            "fresh_process_cold_replay",
            (python, "-u", wrapper, "--date", RUN_DATE, "--cold-replay", str(candidate)),
            "exact_candidate",
            300,
        ),
        CommandSpec(
            "independent_reduction",
            (python, "-u", wrapper, "--date", RUN_DATE, "--independent-reduce", str(candidate)),
            "exact_candidate",
            300,
        ),
        CommandSpec(
            "adversarial_verify",
            (python, "-u", str(root / "scripts/adversarial_verify.py"), str(candidate)),
            "exact_candidate",
            300,
        ),
        CommandSpec(
            "verdict_row_consistency_strict",
            (
                python,
                "-u",
                str(root / "scripts/verdict_row_consistency_lint.py"),
                "--strict",
                str(candidate),
            ),
            "exact_candidate",
            300,
        ),
    ]


def run_experiment(
    root: Path, run_date: str, output: Path
) -> dict[str, Any]:  # pragma: no cover - full entrypoint integration
    """Measure, validate, cold-read, and atomically publish current CPU evidence."""
    started = time.monotonic()
    progress(started, "startup", "begin", root=root.resolve())
    repo = Path(__file__).resolve().parents[2]
    if root.resolve() != repo or run_date != RUN_DATE:
        raise ValueError("root_or_date_mismatch")
    private = Path(tempfile.mkdtemp(prefix="carnot-exp7645-v667-")).resolve()
    spans: list[dict[str, Any]] = []
    phase = time.monotonic()
    progress(started, "preconditions", "before")
    checks, hashes = collect_preconditions(repo)
    prior_hash = sha256_file(repo / OLD_RESULT) if (repo / OLD_RESULT).is_file() else None
    progress(started, "preconditions", "after", checked=len(checks))
    prior._record_phase(spans, "preconditions", phase, started, completed_units=len(checks))
    if any(check["passed"] is not True for check in checks):
        artifact = build_artifact([], {}, checks, spans, hashes, time.monotonic() - started, [])
        progress(started, "publish", "before_atomic_blocked")
        atomic_json(output, artifact)
        progress(started, "publish", "after_atomic_blocked")
        return artifact

    raw = repo / RAW
    raw.mkdir(parents=True, exist_ok=True)
    atomic_json(raw / "affected_validation_manifest.json", affected_validation_manifest())
    phase = time.monotonic()
    progress(started, "measurement", "before_benchmark")
    rows, stage2 = measure_goal_guard_rows()
    for index, row in enumerate(rows):
        atomic_json(raw / "checkpoints" / f"{index:02d}_{row['unit_id']}.json", row)
        progress(started, "measurement", "unit_complete", completed=index + 1, total=len(rows))
    progress(started, "measurement", "after_benchmark", passed=all(row["passed"] for row in rows))
    prior._record_phase(
        spans,
        "measurement",
        phase,
        started,
        completed_units=len(rows),
        checkpoint_position=str(raw / "checkpoints"),
    )

    phase = time.monotonic()
    progress(started, "scoped_validation", "before_subprocess")
    validation = run_commands(
        repo,
        build_validation_commands(repo, private),
        log_dir=private / "logs/affected",
        extra_env={"JAX_PLATFORMS": "cpu"},
        heartbeat_s=60,
    )
    progress(
        started,
        "scoped_validation",
        "after_subprocess",
        passed=all(row["passed"] for row in validation),
    )
    prior._record_phase(spans, "scoped_validation", phase, started, completed_units=len(validation))

    phase = time.monotonic()
    progress(started, "arc_e2e", "before_subprocess")
    e2e = run_commands(
        repo,
        prior.build_e2e_commands(repo, private),
        log_dir=private / "logs/e2e",
        extra_env={"JAX_PLATFORMS": "cpu"},
        heartbeat_s=60,
    )
    progress(started, "arc_e2e", "after_subprocess", passed=all(row["passed"] for row in e2e))
    prior._record_phase(spans, "arc_e2e", phase, started, completed_units=len(e2e))
    receipts = [
        *prior._copy_receipts_after_exit(repo, validation, raw, "affected"),
        *prior._copy_receipts_after_exit(repo, e2e, raw, "e2e"),
    ]
    candidate = build_artifact(
        rows, stage2, checks, spans, hashes, time.monotonic() - started, receipts
    )
    candidate_path = private / "terminal_candidate.json"
    atomic_json(candidate_path, candidate)
    phase = time.monotonic()
    progress(
        started,
        "terminal_readers",
        "before_subprocess",
        candidate_sha256=sha256_file(candidate_path),
    )
    terminal_private = run_commands(
        repo,
        build_terminal_commands(repo, candidate_path),
        log_dir=private / "logs/terminal",
        extra_env={"JAX_PLATFORMS": "cpu"},
        heartbeat_s=60,
    )
    progress(
        started,
        "terminal_readers",
        "after_subprocess",
        passed=all(row["passed"] for row in terminal_private),
    )
    prior._record_phase(
        spans, "terminal_readers", phase, started, completed_units=len(terminal_private)
    )
    terminal = prior._copy_receipts_after_exit(repo, terminal_private, raw, "terminal")
    atomic_json(raw / "terminal_validation_receipts.json", {"receipts": terminal})
    flagged = (
        next(row for row in terminal if row["name"] == "adversarial_verify")["passed"] is not True
    )
    final = build_artifact(
        rows,
        stage2,
        checks,
        spans,
        hashes,
        time.monotonic() - started,
        [*receipts, *terminal],
        flagged_adversarial=flagged,
    )
    errors = validate_artifact(final)
    if errors:
        raise ValueError("terminal_artifact_invalid:" + ",".join(errors))
    if sha256_file(repo / OLD_RESULT) != prior_hash:
        raise ValueError("v666_artifact_changed")
    progress(started, "publish", "before_atomic", output=output)
    atomic_json(output, final)
    progress(started, "publish", "after_atomic", verdict=final["honest_verdict"])
    return final


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", required=True)
    parser.add_argument("--output", type=Path, default=RESULT)
    parser.add_argument("--validate", type=Path)
    parser.add_argument("--cold-replay", type=Path)
    parser.add_argument("--independent-reduce", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - thin CLI
    args = parse_args(argv)
    reader = args.validate or args.cold_replay or args.independent_reduce
    if reader is not None:
        errors = cold_replay(reader)
        print(json.dumps({"errors": errors}, sort_keys=True), flush=True)
        return int(bool(errors))
    root = Path(__file__).resolve().parents[2]
    output = args.output if args.output.is_absolute() else root / args.output
    run_experiment(root, args.date, output)
    return 0
