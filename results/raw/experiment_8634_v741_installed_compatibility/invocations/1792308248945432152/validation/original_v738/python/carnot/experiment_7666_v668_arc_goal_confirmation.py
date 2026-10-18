"""Run REQ-REPORT-7666's CPU goal-confirmation fixture protocol."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import socket
import tempfile
import time
from types import SimpleNamespace
from typing import Any, Sequence

from carnot.agentic.arc_goal_confirmation import GoalConfirmation
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash, sha256_file
from carnot.reporting.experiment_7303_validation_scope import (
    CommandSpec,
    build_scoped_commands,
    run_commands,
)

ROOT = Path(__file__).resolve().parents[2]
RAW = Path("results/raw/experiment_7666_v668_arc_goal_confirmation")
RESULT = Path("results/experiment_7666_v668_arc_goal_confirmation.json")
MODEL_SPECS: list[dict[str, Any]] = []
CASES = (
    ("wrong_goal", 0, "NOT_FINISHED", 1, "contradiction"),
    ("true_goal", 1, "NOT_FINISHED", 1, "confirmed"),
    ("final_action_level_change", 2, "WIN", 1, "confirmed"),
    ("delayed_animation", 0, "NOT_FINISHED", 3, "unknown"),
    ("unknown_terminal", 0, "MYSTERY", 1, "unknown"),
    ("loss", 0, "GAME_OVER", 1, "contradiction"),
)
MANIFEST = {
    "test_paths": ["tests/python/test_experiment_7666_v668_arc_goal_confirmation.py"],
    "changed_modules": ["python/carnot/agentic/arc_goal_confirmation.py"],
    "static_paths": [
        "python/carnot/agentic/arc_competition_agent.py",
        "python/carnot/experiment_7666_v668_arc_goal_confirmation.py",
        "scripts/experiments/experiment_7666_v668_arc_goal_confirmation.py",
    ],
    "e2e_paths": [
        "tests/python/test_experiment_7666_v668_arc_goal_confirmation.py",
        "tests/python/test_arc_decision_telemetry.py",
        "tests/python/test_experiment_7491_e6_timed_live_profile.py",
        "tests/python/test_experiment_7531_b2_induction_gate_measurement.py",
        "tests/python/test_semif_arc_readout_eval.py",
    ],
}


def progress(started: float, phase: str, event: str, **details: Any) -> None:
    """Emit a flushed boundary, including each completed independent unit."""
    print(
        f"[exp7666] phase={phase} event={event} elapsed_s={time.monotonic() - started:.3f} {details}",
        flush=True,
    )


def sdk_frame(seed: int, level: int, state: str, layers: int) -> SimpleNamespace:
    """Make a scripted SDK-shaped frame with independent level and state fields."""
    grid = [[seed % 4 for _ in range(8)] for _ in range(8)]
    return SimpleNamespace(frame=[grid for _ in range(layers)], levels_completed=level, state=state)


def measure(raw: Path, started: float) -> list[dict[str, Any]]:
    """Checkpoint every arm of 48 separate, oracle-labelled SDK fixtures."""
    rows: list[dict[str, Any]] = []
    for case, level, state, layers, expected in CASES:
        for seed in range(8):
            before = sdk_frame(seed, 0, "NOT_FINISHED", 1)
            after = sdk_frame(seed + 1, level, state, layers)
            guard = GoalConfirmation()
            guard.arm(before, frames_seen=1, level=0, predicted_goal=True, plan_length=1)
            observed = guard.observe(after, frames_seen=2)
            group = f"{case}-{seed}"
            for arm in ("OFF", "ON"):
                row = {
                    "group_id": group,
                    "arm": arm,
                    "case": case,
                    "seed": seed,
                    "expected": expected,
                    "goal_row": observed if arm == "ON" else None,
                    "raw_metrics": {
                        "correct": int(observed["status"] == expected) if arm == "ON" else None
                    },
                    "counts": {"sdk_frames": 2, "executed_plan_endpoints": 1},
                    "exclusions": [],
                    "censored": observed["status"] == "unknown" if arm == "ON" else False,
                    "provenance": "scripted_sdk_fixture_oracle",
                }
                atomic_json(raw / "rows" / f"{group}-{arm}.json", row)
                rows.append(row)
            progress(started, "measurement", "unit_checkpoint", completed_groups=len(rows) // 2)
    return rows


def cold_reduce(rows_path: Path) -> dict[str, Any]:
    """Recompute fixture labels from raw rows in a fresh interpreter."""
    rows = json.loads(rows_path.read_text())
    groups = {row["group_id"] for row in rows}
    on = [row for row in rows if row["arm"] == "ON"]
    return {
        "groups": len(groups),
        "rows": len(rows),
        "oracle_matches": sum(row["goal_row"]["status"] == row["expected"] for row in on),
        "contradictions": sum(row["goal_row"]["contradiction"] for row in on),
        "unknown": sum(row["goal_row"]["status"] == "unknown" for row in on),
    }


def validation_commands(root: Path, private: Path) -> list[CommandSpec]:
    """Freeze file-scoped checks before any fixture is measured."""
    commands = build_scoped_commands(
        root,
        MANIFEST["test_paths"],
        MANIFEST["changed_modules"],
        static_paths=[*MANIFEST["static_paths"], "tests/python/coverage_experiment_7666.py"],
        basetemp=private / "basetemp",
        coverage_file=private / "coverage.data",
    )
    include = "*/arc_goal_confirmation.py"
    coverage = str(root / ".venv/bin/coverage")
    for index, command in enumerate(commands):
        if command.name == "changed_module_coverage":
            commands[index] = CommandSpec(
                command.name,
                (
                    "/usr/bin/env",
                    f"PYTHONPATH={private / 'no_jax'}:{root / 'python'}:{root}",
                    coverage,
                    "run",
                    "--rcfile=/dev/null",
                    f"--data-file={private / 'coverage.data'}",
                    f"--include={include}",
                    "tests/python/coverage_experiment_7666.py",
                ),
                "new_guard_behavior",
            )
        elif command.name == "changed_module_coverage_report":
            commands[index] = CommandSpec(
                command.name,
                (
                    coverage,
                    "report",
                    "--rcfile=/dev/null",
                    f"--data-file={private / 'coverage.data'}",
                    f"--include={include}",
                    "--show-missing",
                    "--fail-under=100",
                ),
                "new_guard_behavior",
            )
    commands.append(
        CommandSpec(
            "changed_orchestrator_mypy",
            (
                str(root / ".venv/bin/mypy"),
                "python/carnot/experiment_7666_v668_arc_goal_confirmation.py",
            ),
            "changed_orchestrator",
        )
    )
    return commands


def gate_check(
    check: str, upstream: str, path: Path, field: str, expected: Any, observed: Any
) -> dict[str, Any]:
    """Name every operand of a missing-input block."""
    return {
        "check": check,
        "upstream": upstream,
        "path": str(path.resolve()),
        "field": field,
        "operator": "==",
        "expected": expected,
        "observed": observed,
    }


def preconditions(root: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Authenticate real source bytes and declare the CPU-only resource checks."""
    required = {
        "v667_receipt": root / "results/experiment_7652_v667_arc_wrapper_measurement.json",
        "spec": root / "openspec/capabilities/research-reporting/spec.md",
        "policy": root / "python/carnot/agentic/arc_competition_agent.py",
        "guard": root / "python/carnot/agentic/arc_goal_confirmation.py",
        "reducer": Path(__file__).resolve(),
        "tests": root / MANIFEST["test_paths"][0],
        "e2e_plan": root / "ops/e2e-test-plan.md",
    }
    failed: list[dict[str, Any]] = []
    hashes: dict[str, Any] = {"producers": {}, "pre_gate_receipts": {}, "missing_inputs": []}
    for label, path in required.items():
        if not path.is_file():
            failed.append(gate_check("input_exists", label, path, "exists", True, False))
            hashes["missing_inputs"].append(str(path))
        else:
            hashes["producers"][str(path.relative_to(root))] = sha256_file(path)
    if (root / ".venv/bin/python").is_file() is False:
        failed.append(
            gate_check(
                "python_exists", "local_venv", root / ".venv/bin/python", "exists", True, False
            )
        )
    return failed, hashes


def artifact(
    *,
    rows: list[dict[str, Any]],
    reduced: dict[str, Any],
    failed: list[dict[str, Any]],
    hashes: dict[str, Any],
    receipts: list[dict[str, Any]],
    terminal: list[dict[str, Any]],
    spans: list[dict[str, Any]],
    duration: float,
) -> dict[str, Any]:
    """Reduce validity separately from fixture readiness and scientific benefit."""
    required_ok = not failed and all(row.get("passed") is True for row in receipts + terminal)
    flagged = any(row["name"] == "adversarial_verify" and not row["passed"] for row in terminal)
    fixture_ok = reduced.get("groups") == 48 and reduced.get("oracle_matches") == 48
    ready = int(required_ok and fixture_ok and not flagged)
    verdict_class = (
        "blocked"
        if failed
        else "disqualified"
        if not required_ok or flagged
        else "circular_positive"
        if fixture_ok
        else "null"
    )
    honest_verdict = {
        "blocked": "complete_blocked_missing_input",
        "disqualified": "complete_disqualified_goal_confirmation_validation_failed",
        "circular_positive": "complete_circular_positive_goal_confirmation_fixture_ready",
        "null": "complete_null_goal_confirmation_fixture_failed",
    }[verdict_class]
    principles = {
        "validity": "Every frozen required check and terminal reader must pass.",
        "readiness": "Reachability, independent fixtures, and full new-guard coverage are required.",
        "coverage": "Distinct scripted SDK outcomes, not views or arms, form the denominator.",
        "probability_benefit": "Fixture labels cannot establish hidden-game probability gain.",
        "utility": "No scored action-efficiency benefit was measured.",
        "retention": "No cross-level retained model was evaluated.",
        "freshness": "Scripted exposed fixtures cannot support a new hidden-game claim.",
    }
    gates = {
        name: {"passed": value, "operands": operands, "principle": principles[name]}
        for name, value, operands in (
            (
                "validity",
                required_ok,
                {
                    "failed_preconditions": len(failed),
                    "failed_checks": sum(not r.get("passed", False) for r in receipts + terminal),
                },
            ),
            (
                "readiness",
                bool(ready),
                {
                    "independent_groups": reduced.get("groups", 0),
                    "oracle_matches": reduced.get("oracle_matches", 0),
                },
            ),
            (
                "coverage",
                reduced.get("groups") == 48,
                {"observed": reduced.get("groups", 0), "minimum": 48},
            ),
            ("probability_benefit", False, {"hidden_game_trials": 0, "estimated_delta": None}),
            ("utility", False, {"scored_efficiency_trials": 0, "estimated_delta": None}),
            ("retention", False, {"cross_level_trials": 0}),
            ("freshness", False, {"new_hidden_games": 0, "prior_exposure": "fixture_protocol"}),
        )
    }
    return {
        "experiment_id": 7666,
        "milestone": "2026.09.668",
        "run_date": "20260925",
        "honest_verdict": honest_verdict,
        "verdict_class": verdict_class,
        "flagged_adversarial": flagged,
        "gate_check_summary": failed
        if failed
        else {"required_checks_passed": required_ok, "flagged_adversarial": flagged},
        "acceptance_gate_results": gates,
        "goal_confirmation_ready_score": ready,
        "goal_confirmation_protocol_path": str(RAW / "protocol.json"),
        "rows": rows,
        "goal_rows": [row["goal_row"] for row in rows if row["arm"] == "ON"],
        "sample_size_budget": {
            "intended_independent_groups": 48,
            "observed_independent_groups": reduced.get("groups", 0),
            "eligible_independent_groups": reduced.get("groups", 0),
            "excluded_independent_groups": 0,
            "censored_independent_groups": reduced.get("unknown", 0),
            "prior_exposure": "scripted fixtures and known failure classes",
            "claim_limit": "development_proxy; no hidden-game quality or new solve credit",
        },
        "inference_substrate": "CPU scripted SDK observation and Python source validation",
        "inference_substrate_class": "no_model_load",
        "planned_inference_substrate_class": "no_model_load",
        "MODEL_SPECS": MODEL_SPECS,
        "planned_MODEL_SPECS": [],
        "model_specs": [],
        "model_specs_declaration": "no current model loaded or invoked",
        "model_invoked": False,
        "invocation_counts": {
            "loads_attempted": 0,
            "loads_completed": 0,
            "loads_cancelled": 0,
            "forward_calls_attempted": 0,
            "forward_calls_completed": 0,
            "generations_attempted": 0,
            "generations_completed": 0,
            "input_tokens": 0,
            "output_tokens": 0,
        },
        "historical_model_provenance": "V667 artifact cites inherited model-produced engines; none invoked here",
        "execution_venue": "host",
        "execution_venue_details": {
            "host": socket.gethostname(),
            "owned_pid": os.getpid(),
            "gpu_uuid": None,
        },
        "phase_spans": spans,
        "duration_s": duration,
        "random_seed": {
            "fixture_seeds": list(range(8)),
            "purpose": "distinct scripted grids; no stochastic inference",
        },
        "reproducibility_checksum": canonical_hash(
            {
                "hashes": hashes,
                "cases": CASES,
                "reducer": hashes["producers"].get(
                    "python/carnot/experiment_7666_v668_arc_goal_confirmation.py"
                ),
            }
        ),
        "source_artifact_hashes": hashes,
        "preconditions_checked": {
            "resource_checks": {
                "python_exists": True,
                "host_cpu_available": True,
                "model_load_required": False,
            },
            "input_failures": failed,
        },
        "validation_receipts": {
            "affected_file_manifest": MANIFEST,
            "required_checks": receipts,
            "terminal_readers": terminal,
            "required_checks_passed": required_ok,
            "cold_reduction": reduced,
            "unrelated_full_suite_debt": "recorded separately; not an acceptance gate",
        },
        "verifier_is_oracle": True,
        "field_principles": {
            **principles,
            "rows": "One group and arm per raw SDK fixture; repeated views do not enlarge N.",
            "goal_rows": "Only SDK level and terminal observations can confirm a solve.",
            "honest_verdict": "Completion, receipt validity, and scientific benefit are separate.",
            "inference_substrate": "Current CPU work is not historical model generation.",
            "source_artifact_hashes": "Only immutable input bytes bind the reducer.",
            "phase_spans": "Monotonic disjoint spans describe actual current work.",
        },
        "solve_provenance": "development_proxy",
        "new_game_level_solve_credit": False,
        "same_verdict_retirement": {
            "mechanism": "V667 source and claim grammar",
            "decision": "retire_zero-check_mechanism",
            "basis": "V667 checked zero real source predicates; fixture guard success does not revive that mechanism",
        },
    }


def _span(
    spans: list[dict[str, Any]], name: str, start: float, phase_start: float, completed: int
) -> None:
    """Checkpoint a disjoint monotonic stage interval."""
    end = time.monotonic()
    spans.append(
        {
            "phase": name,
            "start_s": phase_start - start,
            "end_s": end - start,
            "duration_s": end - phase_start,
            "completed_units": completed,
            "heartbeat_s": end - start,
            "checkpoint": completed,
        }
    )


def _no_jax_shim(path: Path) -> None:
    """Keep optional JAX outside the CPU coverage process on this host."""
    path.mkdir(parents=True, exist_ok=True)
    (path / "sitecustomize.py").write_text(
        "import builtins\n"
        "_real_import = builtins.__import__\n"
        "def _no_jax(name, *args, **kwargs):\n"
        "    if name == 'jax' or name.startswith('jax.'):\n"
        "        error = ModuleNotFoundError('optional JAX disabled for CPU coverage')\n"
        "        error.name = 'jax'\n"
        "        raise error\n"
        "    return _real_import(name, *args, **kwargs)\n"
        "builtins.__import__ = _no_jax\n",
        encoding="utf-8",
    )


def _persist_receipts(
    root: Path, raw: Path, family: str, receipts: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Move streamed logs into durable raw evidence before publication."""
    durable: list[dict[str, Any]] = []
    for index, receipt in enumerate(receipts):
        source = Path(receipt["log_path"])
        if not source.is_absolute():
            source = root / source
        target = raw / "logs" / family / f"{index:02d}_{receipt['name']}.log"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(source.read_bytes())
        durable.append(
            {
                **receipt,
                "log_path": str(target.relative_to(root)),
                "log_sha256": sha256_file(target),
            }
        )
    return durable


def run_experiment(root: Path, run_date: str, output: Path) -> dict[str, Any]:
    """Freeze, measure, validate, cold-reduce, read the candidate, publish."""
    started = time.monotonic()
    root = root.resolve()
    progress(started, "startup", "begin", root=root)
    if root != ROOT or run_date != "20260925":
        raise ValueError("run_contract_mismatch")
    raw = root / RAW
    raw.mkdir(parents=True, exist_ok=True)
    private = Path(tempfile.mkdtemp(prefix="carnot-exp7666-")).resolve()
    (private / "basetemp").mkdir(parents=True, exist_ok=True)
    _no_jax_shim(private / "no_jax")
    spans: list[dict[str, Any]] = []

    phase = time.monotonic()
    progress(started, "preconditions", "before")
    failed, hashes = preconditions(root)
    progress(started, "preconditions", "after", failed=len(failed))
    _span(spans, "preconditions", started, phase, len(hashes["producers"]))
    if failed:
        final = artifact(
            rows=[],
            reduced={},
            failed=failed,
            hashes=hashes,
            receipts=[],
            terminal=[],
            spans=spans,
            duration=time.monotonic() - started,
        )
        progress(started, "publication", "before_atomic", verdict=final["honest_verdict"])
        atomic_json(output, final)
        progress(started, "publication", "after_atomic", output=output)
        return final

    phase = time.monotonic()
    progress(started, "freeze", "before")
    commands = validation_commands(root, private)
    frozen = {
        "affected_files": MANIFEST,
        "commands": [
            {"name": c.name, "argv": list(c.argv), "timeout_s": c.timeout_s} for c in commands
        ],
    }
    atomic_json(raw / "affected_validation_manifest.json", frozen)
    protocol = {
        "requirement": "REQ-REPORT-7666",
        "cases": CASES,
        "fixture_seeds": list(range(8)),
        "independent_groups": 48,
        "oracle_truth": "scripted_sdk_fixture",
        "MODEL_SPECS": [],
        "inference_substrate_class": "no_model_load",
    }
    atomic_json(raw / "protocol.json", protocol)
    hashes["pre_gate_receipts"][str(RAW / "affected_validation_manifest.json")] = sha256_file(
        raw / "affected_validation_manifest.json"
    )
    hashes["pre_gate_receipts"][str(RAW / "protocol.json")] = sha256_file(raw / "protocol.json")
    progress(started, "freeze", "after", commands=len(commands), groups=48)
    _span(spans, "freeze", started, phase, 2)

    phase = time.monotonic()
    progress(started, "measurement", "before_benchmark")
    rows = measure(raw, started)
    atomic_json(raw / "rows.json", rows)
    progress(started, "measurement", "after_benchmark", rows=len(rows))
    _span(spans, "measurement", started, phase, len(rows) // 2)

    phase = time.monotonic()
    progress(started, "validation", "before_subprocess")
    validation = run_commands(
        root,
        commands,
        log_dir=private / "logs" / "affected",
        extra_env={"JAX_PLATFORMS": "cpu", "COVERAGE_FILE": str(private / "coverage.data")},
        heartbeat_s=60,
    )
    validation = _persist_receipts(root, raw, "affected", validation)
    progress(
        started, "validation", "after_subprocess", failed=sum(not r["passed"] for r in validation)
    )
    _span(spans, "validation", started, phase, len(validation))

    phase = time.monotonic()
    progress(started, "e2e", "before_subprocess")
    e2e_command = CommandSpec(
        "e2e_011_013_and_goal",
        (
            str(root / ".venv/bin/pytest"),
            "-n",
            "0",
            "-o",
            "addopts=",
            "--no-cov",
            f"--basetemp={private / 'basetemp' / 'e2e'}",
            *MANIFEST["e2e_paths"],
            "-q",
        ),
        "declared_task_E2E",
    )
    e2e = run_commands(
        root,
        [e2e_command],
        log_dir=private / "logs" / "e2e",
        extra_env={"JAX_PLATFORMS": "cpu"},
        heartbeat_s=60,
    )
    e2e = _persist_receipts(root, raw, "e2e", e2e)
    progress(started, "e2e", "after_subprocess", failed=sum(not r["passed"] for r in e2e))
    _span(spans, "e2e", started, phase, len(e2e))

    phase = time.monotonic()
    progress(started, "cold_reduction", "before_subprocess")
    cold_command = CommandSpec(
        "cold_reduction",
        (
            str(root / ".venv/bin/python"),
            "-u",
            "-m",
            "carnot.experiment_7666_v668_arc_goal_confirmation",
            "--cold-reduce",
            str(raw / "rows.json"),
        ),
        "exact_raw_rows",
    )
    cold = run_commands(
        root,
        [cold_command],
        log_dir=private / "logs" / "cold",
        extra_env={"JAX_PLATFORMS": "cpu"},
        heartbeat_s=60,
    )
    cold = _persist_receipts(root, raw, "cold", cold)
    try:
        reduced = json.loads((root / cold[0]["log_path"]).read_text().strip().splitlines()[-1])
    except (IndexError, json.JSONDecodeError):
        reduced = {}
    if reduced != cold_reduce(raw / "rows.json"):
        cold[0]["passed"] = False
    progress(
        started,
        "cold_reduction",
        "after_subprocess",
        groups=reduced.get("groups", 0),
        exit=cold[0]["exit_code"],
    )
    _span(spans, "cold_reduction", started, phase, len(rows) // 2)
    validation.extend(e2e + cold)

    phase = time.monotonic()
    candidate_path = raw / "terminal_candidate.json"
    candidate = artifact(
        rows=rows,
        reduced=reduced,
        failed=[],
        hashes=hashes,
        receipts=validation,
        terminal=[],
        spans=spans,
        duration=time.monotonic() - started,
    )
    atomic_json(candidate_path, candidate)
    candidate_hash = sha256_file(candidate_path)
    progress(started, "terminal_readers", "before_subprocess", candidate_sha256=candidate_hash)
    terminal_commands = [
        CommandSpec(
            "adversarial_verify",
            (
                str(root / ".venv/bin/python"),
                "-u",
                "scripts/adversarial_verify.py",
                str(candidate_path),
            ),
            "exact_terminal_candidate",
            300.0,
        ),
        CommandSpec(
            "verdict_row_consistency_lint",
            (
                str(root / ".venv/bin/python"),
                "-u",
                "scripts/verdict_row_consistency_lint.py",
                "--strict",
                str(candidate_path),
            ),
            "exact_terminal_candidate",
            300.0,
        ),
    ]
    terminal = run_commands(
        root,
        terminal_commands,
        log_dir=private / "logs" / "terminal",
        extra_env={"JAX_PLATFORMS": "cpu"},
        heartbeat_s=60,
    )
    terminal = _persist_receipts(root, raw, "terminal", terminal)
    atomic_json(
        raw / "terminal_validation_receipts.json",
        {"candidate_sha256": candidate_hash, "readers": terminal},
    )
    progress(
        started,
        "terminal_readers",
        "after_subprocess",
        failed=sum(not r["passed"] for r in terminal),
    )
    _span(spans, "terminal_readers", started, phase, len(terminal))

    final = artifact(
        rows=rows,
        reduced=reduced,
        failed=[],
        hashes=hashes,
        receipts=validation,
        terminal=terminal,
        spans=spans,
        duration=time.monotonic() - started,
    )
    final["validation_receipts"]["terminal_candidate_path"] = str(candidate_path.relative_to(root))
    final["validation_receipts"]["terminal_candidate_sha256"] = candidate_hash
    progress(started, "publication", "before_atomic", verdict=final["honest_verdict"])
    atomic_json(output, final)
    progress(started, "publication", "after_atomic", output=output)
    return final


def main(argv: Sequence[str] | None = None) -> int:
    """Keep the public CLI thin and make cold reduction a separate process."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date")
    parser.add_argument("--output", type=Path, default=RESULT)
    parser.add_argument("--cold-reduce", type=Path)
    args = parser.parse_args(argv)
    if args.cold_reduce is not None:
        print(json.dumps(cold_reduce(args.cold_reduce), sort_keys=True), flush=True)
        return 0
    if args.date is None:
        parser.error("--date is required")
    output = args.output if args.output.is_absolute() else ROOT / args.output
    run_experiment(ROOT, args.date, output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
