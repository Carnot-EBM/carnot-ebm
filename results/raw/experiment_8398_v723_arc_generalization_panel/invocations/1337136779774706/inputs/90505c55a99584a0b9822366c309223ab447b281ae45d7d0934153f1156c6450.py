"""CPU qualification and terminal custody for the V671 ARC runner.

REQ-REPORT-7708. Public games supply a frozen schedule for the next live task;
this task scores only scripted SDK transport with no model load.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from contextlib import nullcontext
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

from carnot.agentic.arc_generalization_runtime import freeze_schedule, run_episode
from carnot.reporting.current_work_receipt import atomic_json, sha256_file

ROOT = Path(__file__).resolve().parents[2]
RESULT = Path("results/experiment_7708_v671_arc_generalization_runner.json")
RAW = Path("results/raw/experiment_7708_v671_arc_generalization_runner")
REQUIRED = (
    "AGENTS.md",
    "CODEX.md",
    "CLAUDE.md",
    "research-program.md",
    "ops/exclusion_manifest.yaml",
    "ops/e2e-test-plan.md",
    "ops/arc_solve_registry.yaml",
    "scripts/experiment_template.py",
    "python/carnot/reporting/current_work_receipt.py",
    "python/carnot/reporting/experiment_7303_validation_scope.py",
    "openspec/capabilities/research-reporting/spec.md",
    "openspec/capabilities/arc-world-model-trust-energy/spec.md",
    "python/carnot/agentic/arc_competition_agent.py",
    "python/carnot/agentic/arc_decision_telemetry.py",
    "python/carnot/agentic/arc_generalization_runtime.py",
    "python/carnot/experiment_7708_v671_arc_generalization_runner.py",
    "scripts/experiments/experiment_7708_v671_arc_generalization_runner.py",
    "tests/python/test_experiment_7708_v671_arc_generalization_runner.py",
)
SCOPE = {
    "tests": ["tests/python/test_experiment_7708_v671_arc_generalization_runner.py"],
    "changed_modules": [
        "python/carnot/agentic/arc_generalization_runtime.py",
        "python/carnot/experiment_7708_v671_arc_generalization_runner.py",
    ],
    "static_paths": ["scripts/experiments/experiment_7708_v671_arc_generalization_runner.py"],
    "e2e": ["E2E-009", "E2E-011", "E2E-013"],
}
PRINCIPLES = {
    "honest_verdict": "A terminal disposition prevents retries of unchanged external blocks.",
    "verdict_class": "A closed enum carries claim eligibility into downstream readers.",
    "flagged_adversarial": "Disqualified evidence must not pass a downstream readiness gate.",
    "gate_check_summary": "Exact upstream operands distinguish a false gate from missing evidence.",
    "acceptance_gate_results": "Measured operands make each claim independently checkable.",
    "rows": "Unit observations let another reader recompute every comparison.",
    "sample_size_budget": "Seeds and transformed views never enlarge independent n.",
    "inference_substrate": "The declared path must match real computation and its duration floor.",
    "inference_substrate_class": "No generation has no model duration floor.",
    "MODEL_SPECS": "Model identity must match actual invocations, not the coding agent.",
    "model_invoked": "Current invocation counters distinguish calls from plans.",
    "execution_venue": "Host and device claims need measured custody.",
    "phase_spans": "Disjoint spans and checkpoints locate current work.",
    "random_seed": "Independent replay uses the same declared random inputs.",
    "reproducibility_checksum": "Immutable inputs and reducer code bind the replay identity.",
    "source_artifact_hashes": "Producer and missing custody stay separate from planned outputs.",
    "preconditions_checked": "Required inputs and resources are measured before execution.",
    "validation_receipts": "Frozen commands, exits, and logs control readiness.",
    "verifier_is_oracle": "Fixture truth forbids an oracle-distinct positive claim.",
    "arc_runner_ready_score": "Readiness requires scored fixture execution, withholding, parity, and validation.",
    "arc_schedule_path": "The fixed schedule lets Exp7709 replay the same public selection.",
    "registry_precheck": "Historical levels remain visible without opening solve credit.",
    "solve_provenance": "Only reachable runtime self-discovery can receive live solve credit.",
    "new_solve_credit": "A scripted fixture cannot add a registry solve.",
    "terminal_reader_receipts_path": "The sidecar binds exact candidate bytes to fresh reader exits and log hashes.",
}


def progress(started: float, phase: str, event: str, **detail: Any) -> None:
    """Keep the owner visible at every phase and long child boundary."""
    print(
        f"[exp7708] {phase} {event} elapsed_s={time.monotonic() - started:.3f} {detail}", flush=True
    )


def check(
    name: str, upstream: str, path: str, field: str, expected: Any, observed: Any
) -> dict[str, Any]:
    """Carry the exact external operand into blocked terminal evidence."""
    return {
        "check": name,
        "upstream": upstream,
        "path": path,
        "field": field,
        "operator": "==",
        "expected": expected,
        "observed": observed,
        "passed": expected == observed,
    }


def collect_preconditions(
    root: Path, *, sdk_roster: Sequence[str] | None
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Authenticate current bytes and the SDK catalogue before fixtures run."""
    root = root.resolve()
    checks: list[dict[str, Any]] = []
    hashes: dict[str, Any] = {"producer_files": {}, "pre_gate_receipts": {}, "missing_custody": []}
    for relative in REQUIRED:
        path = root / relative
        present = path.is_file() and path.stat().st_size > 0
        checks.append(check("input_bytes", relative, str(path), "readable_nonempty", True, present))
        if present:
            hashes["producer_files"][relative] = sha256_file(path)
        else:
            hashes["missing_custody"].append(relative)
    checks.append(
        check(
            "sdk_catalogue",
            "arc_agi_offline_sdk",
            str(root / "environment_files"),
            "at_least_two_runnable_games",
            True,
            sdk_roster is not None and len(set(sdk_roster)) >= 2,
        )
    )
    checks.append(
        check(
            "offline_runtime",
            "local_python_environment",
            str(root / ".venv/bin/python"),
            "python_executable",
            True,
            (root / ".venv/bin/python").is_file(),
        )
    )
    for spec, requirement in (
        ("openspec/capabilities/research-reporting/spec.md", "REQ-REPORT-7708"),
        ("openspec/capabilities/arc-world-model-trust-energy/spec.md", "REQ-ARC-WMTE-7708"),
    ):
        path = root / spec
        found = path.is_file() and requirement in path.read_text(encoding="utf-8")
        checks.append(check("driving_requirement", spec, str(path), requirement, True, found))
    return checks, hashes


def _gate(passed: bool, principle: str, **operands: Any) -> dict[str, Any]:
    return {"passed": passed, "principle": principle, "measured_operands": operands}


def build_artifact(
    *,
    checks: Sequence[Mapping[str, Any]],
    hashes: Mapping[str, Any],
    schedule: Mapping[str, Any] | None,
    rows: Sequence[Mapping[str, Any]],
    run_date: str,
    validation_receipts: Sequence[Mapping[str, Any]],
    duration_s: float,
    phase_spans: Sequence[Mapping[str, Any]] = (),
    flagged_adversarial: bool = False,
) -> dict[str, Any]:
    """Reduce only raw fixtures; missing inputs and invalid checks stay distinct."""
    from carnot.reporting.experiment_7303_validation_scope import REQUIRED_CHECK_NAMES

    failures = [dict(row) for row in checks if row.get("passed") is not True]
    receipt_names = {
        str(row.get("name")) for row in validation_receipts if row.get("exit_code") == 0
    }
    expected_names = set(REQUIRED_CHECK_NAMES) | {"e2e_009", "e2e_009_smoke", "e2e_011", "e2e_013"}
    required_ok = expected_names <= receipt_names and all(
        row.get("exit_code") == 0 for row in validation_receipts
    )
    row_ok = len(rows) == 2 and all(
        row.get("counts", {}).get("choose_action", 0) > 0
        and row.get("counts", {}).get("observations", 0) > 0
        and row.get("censoring") is None
        and row.get("policy_entry", {}).get("policy_class") == "E3AgentPolicy"
        for row in rows
    )
    if failures:
        verdict = "blocked"
        reason = str(failures[0]["check"])
    elif flagged_adversarial or validation_receipts and not required_ok:
        verdict = "disqualified"
        reason = "required_validation"
    elif not row_ok:
        verdict = "disqualified"
        reason = "fixture_execution"
    elif not validation_receipts:
        verdict = "partial"
        reason = "validation_unfinished"
    else:
        verdict = "circular_positive"
        reason = "fixture_runner_ready"
    ready = int(verdict == "circular_positive")
    observed = len(
        {str(row.get("game")) for row in rows if row.get("counts", {}).get("observations", 0)}
    )
    eligible = len(schedule.get("rows", [])) if schedule else 0
    gates = {
        "validity": _gate(
            not failures and row_ok,
            "Required bytes and factual SDK actions must exist.",
            failed=len(failures),
            runner_rows=len(rows),
        ),
        "readiness": _gate(
            bool(ready),
            "Readiness prevents invalid evidence propagation.",
            required_checks=len(expected_names),
            passing_checks=len(receipt_names & expected_names),
        ),
        "coverage": _gate(
            required_ok,
            "Every frozen changed-code and E2E check must pass.",
            passing_checks=len(receipt_names & expected_names),
            required_checks=len(expected_names),
        ),
        "freshness": _gate(
            row_ok,
            "Current fixture transitions must be observed in this invocation.",
            observed_games=observed,
        ),
        "probability": _gate(
            False, "Public fixtures cannot estimate hidden-game probability.", hidden_games=0
        ),
        "utility": _gate(
            False, "Quality thresholds prevent effects inferred from plumbing.", paired_live_arms=0
        ),
        "retention": _gate(
            False, "Retention bounds prevent improvement by forgetting.", retained_live_games=0
        ),
        "efficiency": _gate(
            False, "Action-cost benefit needs paired live episodes.", paired_live_arms=0
        ),
    }
    result: dict[str, Any] = {
        "schema": "carnot.exp7708.v671.arc_generalization_runner.v1",
        "experiment_id": 7708,
        "milestone": "2026.09.671",
        "run_date": run_date,
        "status": "complete" if verdict != "partial" else "partial",
        "honest_verdict": f"complete_{verdict}_{reason}"
        if verdict != "partial"
        else "partial_validation_unfinished",
        "verdict_class": verdict,
        "flagged_adversarial": flagged_adversarial,
        "gate_check_summary": {"failed_checks": failures, "failed_count": len(failures)},
        "acceptance_gate_results": gates,
        "rows": [dict(row) for row in rows],
        "sample_size_budget": {
            "intended_independent_games": 2,
            "observed_independent_games": observed,
            "eligible_independent_games": eligible,
            "excluded_independent_games": len(schedule.get("sdk_exclusions", []))
            if schedule
            else 0,
            "censored_independent_games": sum(bool(row.get("censoring")) for row in rows),
            "effective_blocks": observed,
            "prior_exposure": "public_games_historically_exposed",
            "inference_limits": {"model_loads": 0, "generations": 0, "max_actions_per_fixture": 3},
        },
        "inference_substrate": "host_scripted_sdk_cpu_fixture_no_model_call",
        "inference_substrate_class": "no_model_load",
        "planned_inference_substrate_class": "no_model_load",
        "MODEL_SPECS": [],
        "planned_MODEL_SPECS": [],
        "model_specs": [
            {
                "no_model_invoked": True,
                "current_llm_provenance": "unsloth/Qwen3.8-27B-GGUF; not loaded",
            }
        ],
        "model_invoked": False,
        "invocation_counts": {
            "loads": 0,
            "forwards": 0,
            "generations": 0,
            "tokens": 0,
            "failures": 0,
            "cancellations": 0,
        },
        "execution_venue": "host",
        "execution_venue_details": {
            "host": socket.gethostname(),
            "owned_pid": os.getpid(),
            "gpu_uuid": None,
        },
        "duration_s": duration_s,
        "phase_spans": [dict(span) for span in phase_spans],
        "random_seed": {"selection_salt": "v671-arc-20260926", "fixture_seed": 7708},
        "source_artifact_hashes": dict(hashes),
        "preconditions_checked": [dict(row) for row in checks],
        "validation_receipts": [dict(row) for row in validation_receipts],
        "terminal_reader_receipts_path": str(RAW / "validation_receipts.json"),
        "frozen_validation_scope": SCOPE,
        "verifier_is_oracle": True,
        "arc_runner_ready_score": ready,
        "arc_schedule_path": str(RAW / "schedule.json"),
        "registry_precheck": schedule.get("registry_precheck", {}) if schedule else {},
        "solve_provenance": "development_proxy" if rows else "no_live_attempt",
        "new_solve_credit": False,
        "prior_failures": [
            {
                "experiment_id": "exp7694",
                "custody": "not_emitted_usage_limit_three_attempts",
                "same_verdict_retirement": "not_applicable_no_producer",
            },
            {
                "experiment_id": "exp7681",
                "verdict": "complete_blocked_eligible_novel_target",
                "mechanism": "registry full clear bars new solve credit, not public fixture selection",
                "same_verdict_retirement": "retired_by_adapter_withheld_schedule"
                if schedule
                else "pending",
            },
            {
                "experiment_id": "exp7653",
                "verdict": "complete_disqualified_required_validation",
                "mechanism": "frozen affected scope and scored fixture entry",
                "same_verdict_retirement": "retired_by_required_validation" if ready else "pending",
            },
        ],
        "effective_agent_backend": {
            "requested": "codex",
            "effective": "codex",
            "CODEX_FORCE_EXPERIMENTS": os.environ.get("CODEX_FORCE_EXPERIMENTS"),
            "session_id": os.environ.get("CODEX_SESSION_ID"),
        },
        "current_agent_invocation_receipt": {
            "session_id": os.environ.get("CODEX_SESSION_ID"),
            "backend": "codex",
            "current_work_completed_units": len(rows),
            "successful_current_process": True,
        },
    }
    result["reproducibility_checksum"] = (
        "sha256:"
        + hashlib.sha256(
            json.dumps(
                [hashes, schedule, SCOPE, result["random_seed"]], sort_keys=True, default=str
            ).encode()
        ).hexdigest()
    )
    result["field_principles"] = {
        key: PRINCIPLES.get(key, "This field limits the current claim to recorded work.")
        for key in result
    }
    result["field_principles"].update({name: value["principle"] for name, value in gates.items()})
    return result


class _FixtureFrame:
    """A visible SDK frame with no hidden label or source object."""

    def __init__(self, level: int) -> None:
        self.levels_completed = level
        self.frame = [[[0] * 8 for _ in range(8)]]
        self.available_actions = [1, 2, 3, 4, 5, 6]
        self.state = "NOT_FINISHED"


class _FixtureEnvironment:
    def __init__(self) -> None:
        self.actions = 0

    def reset(self) -> _FixtureFrame:
        return _FixtureFrame(0)

    def step(self, action: Any, *, data: Any = None) -> _FixtureFrame:
        self.actions += 1
        return _FixtureFrame(int(self.actions >= 1))


class _FixtureArcade:
    def open_scorecard(self) -> str:
        return "exp7708-scripted-scorecard"

    def make(self, game: str, *, scorecard_id: str) -> _FixtureEnvironment:
        if not game or scorecard_id != "exp7708-scripted-scorecard":
            raise ValueError("invalid_fixture_transport")
        return _FixtureEnvironment()


def _catalogue(started: float) -> tuple[list[str] | None, str | None, list[dict[str, str]]]:
    progress(started, "sdk_catalogue", "before")
    excluded: list[dict[str, str]] = []
    try:
        from carnot.agentic.arc_solver_kit import offline_arcade

        arcade = offline_arcade()
        listed = sorted({str(item.game_id).split("-")[0] for item in arcade.available_environments})
        games = []
        for index, game in enumerate(listed):
            progress(started, "sdk_reset", "before", game=game, completed=index)
            try:
                env = arcade.make(game, scorecard_id=arcade.open_scorecard())
                if env.reset() is None:
                    raise RuntimeError("sdk_null_observation")
                games.append(game)
            except Exception as exc:
                excluded.append({"game": game, "reason": f"{type(exc).__name__}: {exc}"[:200]})
            progress(started, "sdk_reset", "after", game=game, completed=index + 1)
        error = None
    except Exception as exc:
        games = None
        error = f"{type(exc).__name__}: {exc}"[:300]
    progress(started, "sdk_catalogue", "after", games=len(games or []), error=error)
    return games, error, excluded


def _validate(root: Path, private: Path, started: float) -> list[dict[str, Any]]:
    """Use the pre-frozen scope and private paths for every required child."""
    from carnot.reporting.experiment_7303_validation_scope import (
        CommandSpec,
        build_scoped_commands,
        run_commands,
    )

    private.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="exp7708-validation-", dir="/tmp") as scratch_dir:
        scratch = Path(scratch_dir)
        (scratch / "basetemp").mkdir()
        commands = build_scoped_commands(
            root,
            SCOPE["tests"],
            SCOPE["changed_modules"],
            static_paths=SCOPE["static_paths"],
            basetemp=scratch / "basetemp",
            coverage_file=private / ".coverage",
        )
        test_cmd = str(root / ".venv/bin/pytest")
        common = ("-n", "0", "-o", "addopts=", "--no-cov")
        e2e_tests = {
            "e2e_009": ("tests/python/test_arc_induction_state_persistence.py",),
            "e2e_011": ("tests/python/test_arc_decision_telemetry.py",),
            "e2e_013": (
                "tests/python/test_arc_decision_telemetry.py",
                "tests/python/test_experiment_7491_e6_timed_live_profile.py",
                "tests/python/test_experiment_7531_b2_induction_gate_measurement.py",
                "tests/python/test_semif_arc_readout_eval.py",
            ),
        }
        for name, paths in e2e_tests.items():
            commands.append(
                CommandSpec(
                    name,
                    (test_cmd, *common, f"--basetemp={scratch / 'basetemp' / name}", *paths, "-q"),
                    "numbered_e2e_cpu",
                    900.0,
                )
            )
        commands.append(
            CommandSpec(
                "e2e_009_smoke",
                (
                    str(root / ".venv/bin/python"),
                    "-u",
                    "scripts/arc_loop_solve.py",
                    "--mechanism",
                    "e3",
                    "--game",
                    "r11l",
                    "--max-actions",
                    "12",
                    "--output",
                    str(scratch / "e2e_009_smoke.json"),
                ),
                "numbered_e2e_cpu",
                300.0,
            )
        )
        progress(started, "validation", "before", commands=len(commands))
        receipts = run_commands(
            root,
            commands,
            log_dir=private / "logs",
            extra_env={"JAX_PLATFORMS": "cpu", "CARNOT_ARC_DISABLE_INDUCTION": "1"},
            heartbeat_s=45.0,
        )
    progress(started, "validation", "after", passed=sum(row["exit_code"] == 0 for row in receipts))
    return receipts


def cold_read(path: Path) -> dict[str, Any]:
    """Recount persisted raw observations in a fresh interpreter."""
    value = json.loads(path.read_text(encoding="utf-8"))
    rows = value.get("rows", [])
    games = {str(row["game"]) for row in rows if row.get("counts", {}).get("observations", 0)}
    counts = {
        "independent_games": len(games),
        "rows": len(rows),
        "choose_action": sum(int(row["counts"]["choose_action"]) for row in rows),
        "observations": sum(int(row["counts"]["observations"]) for row in rows),
    }
    if counts["independent_games"] != value["sample_size_budget"]["observed_independent_games"]:
        raise ValueError("raw_group_count_mismatch")
    if any(row.get("new_solve_credit") is not False for row in rows):
        raise ValueError("fixture_solve_credit")
    return counts


def _terminal_readers(
    root: Path, candidate: Path, private: Path, started: float
) -> list[dict[str, Any]]:
    from carnot.reporting.experiment_7303_validation_scope import CommandSpec, run_commands

    python = str(root / ".venv/bin/python")
    commands = [
        CommandSpec(
            "cold_reduction",
            (
                python,
                "-u",
                str(root / "scripts/experiments/experiment_7708_v671_arc_generalization_runner.py"),
                "--cold-read",
                str(candidate),
            ),
            "exact_terminal_candidate",
            120.0,
        ),
        CommandSpec(
            "adversarial_verify",
            (python, "-u", "scripts/adversarial_verify.py", str(candidate)),
            "exact_terminal_candidate",
            300.0,
        ),
        CommandSpec(
            "verdict_row_consistency_strict",
            (python, "-u", "scripts/verdict_row_consistency_lint.py", "--strict", str(candidate)),
            "exact_terminal_candidate",
            300.0,
        ),
    ]
    progress(started, "terminal_readers", "before", candidate=str(candidate))
    receipts = run_commands(root, commands, log_dir=private / "terminal_logs", heartbeat_s=45.0)
    progress(started, "terminal_readers", "after", exits=[row["exit_code"] for row in receipts])
    return receipts


def run_experiment(root: Path, run_date: str, output: Path) -> dict[str, Any]:
    """Freeze, execute, validate, cold-read, and atomically publish one receipt."""
    root = root.resolve()
    started = time.monotonic()
    spans: list[dict[str, Any]] = []
    progress(started, "preflight", "before", root=str(root))
    phase_start = time.monotonic()
    roster, sdk_error, sdk_exclusions = _catalogue(started)
    checks, hashes = collect_preconditions(root, sdk_roster=roster)
    if sdk_error:
        checks.append(
            check(
                "sdk_error",
                "arc_agi_offline_sdk",
                str(root / "environment_files"),
                "catalogue_error",
                None,
                sdk_error,
            )
        )
    spans.append(
        {
            "phase": "preflight",
            "start_monotonic_s": phase_start,
            "end_monotonic_s": time.monotonic(),
            "completed_units": 0,
        }
    )
    progress(started, "preflight", "after", failed=sum(not row["passed"] for row in checks))

    raw = root / RAW
    raw.mkdir(parents=True, exist_ok=True)
    schedule = None
    rows: list[dict[str, Any]] = []
    if all(row["passed"] for row in checks):
        registry = (
            yaml.safe_load((root / "ops/arc_solve_registry.yaml").read_text(encoding="utf-8")) or {}
        )
        schedule = freeze_schedule(roster or [], registry.get("games", []))
        schedule["sdk_exclusions"] = sdk_exclusions
        atomic_json(raw / "schedule.json", schedule)
        progress(started, "fixture", "before", games=[unit["game"] for unit in schedule["rows"]])
        phase_start = time.monotonic()
        for unit in schedule["rows"]:
            progress(started, "fixture", "before_benchmark", episode=unit["episode_id"])
            row = run_episode(unit, _FixtureArcade())
            rows.append(row)
            atomic_json(raw / "checkpoint_rows.json", rows)
            progress(
                started,
                "fixture",
                "after_benchmark",
                episode=unit["episode_id"],
                actions=row["counts"]["choose_action"],
            )
        spans.append(
            {
                "phase": "fixture",
                "start_monotonic_s": phase_start,
                "end_monotonic_s": time.monotonic(),
                "completed_units": len(rows),
                "checkpoint": str(RAW / "checkpoint_rows.json"),
            }
        )
        progress(started, "fixture", "after", completed=len(rows))

    with nullcontext(raw / "private_validation") as private:
        private.mkdir(parents=True, exist_ok=True)
        phase_start = time.monotonic()
        receipts = _validate(root, private, started)
        spans.append(
            {
                "phase": "validation",
                "start_monotonic_s": phase_start,
                "end_monotonic_s": time.monotonic(),
                "completed_units": len(receipts),
            }
        )
        candidate = build_artifact(
            checks=checks,
            hashes=hashes,
            schedule=schedule,
            rows=rows,
            run_date=run_date,
            validation_receipts=receipts,
            duration_s=time.monotonic() - started,
            phase_spans=spans,
        )
        candidate_path = raw / "terminal_candidate.json"
        atomic_json(candidate_path, candidate)
        terminal = _terminal_readers(root, candidate_path, private, started)
        if any(row["exit_code"] != 0 for row in terminal):
            failed_receipts = [*receipts, *terminal]
            candidate = build_artifact(
                checks=checks,
                hashes=hashes,
                schedule=schedule,
                rows=rows,
                run_date=run_date,
                validation_receipts=failed_receipts,
                duration_s=time.monotonic() - started,
                phase_spans=spans,
                flagged_adversarial=any(
                    row["name"] == "adversarial_verify" and row["exit_code"] != 0
                    for row in terminal
                ),
            )
            atomic_json(candidate_path, candidate)
            terminal = _terminal_readers(root, candidate_path, private, started)
        sidecar = {
            "candidate_sha256": sha256_file(candidate_path),
            "validation_receipts": receipts,
            "terminal_reader_receipts": terminal,
        }
        atomic_json(raw / "validation_receipts.json", sidecar)
        atomic_json(output, candidate)
    progress(
        started, "publication", "after", verdict=candidate["honest_verdict"], output=str(output)
    )
    return candidate


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", default="20260926")
    parser.add_argument("--output", default=str(RESULT))
    parser.add_argument("--cold-read", type=Path)
    args = parser.parse_args(argv)
    if args.cold_read:
        print(json.dumps(cold_read(args.cold_read), sort_keys=True), flush=True)
        return 0
    run_experiment(ROOT, args.date, ROOT / args.output)
    return 0
