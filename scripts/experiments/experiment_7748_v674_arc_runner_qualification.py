"""Thin no-model ARC runner qualification for REQ-REPORT-7748."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import socket
import time
from typing import Any

from carnot.agentic.arc_solver_kit import offline_arcade
from carnot.experiment_7708_v671_arc_generalization_runner import _FixtureArcade
from carnot.experiment_7748_v674_arc_runner_qualification import (
    cold_reduce,
    run_qualification_episode,
)
from carnot.reporting.current_work_receipt import atomic_json, sha256_file
from carnot.reporting.experiment_7303_validation_scope import (
    CommandSpec,
    build_scoped_commands,
    run_commands,
)

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "results/raw/experiment_7748_v674_arc_runner_qualification"
RESULT = ROOT / "results/experiment_7748_v674_arc_runner_qualification.json"
GAMES = ("cd82", "dc22", "lf52", "m0r0", "sk48", "tn36", "sb26", "sc25")
SEEDS = (674901, 674902)
ARMS = ("off", "total", "organic")
INPUTS = (
    "AGENTS.md",
    "CODEX.md",
    "CLAUDE.md",
    "research-program.md",
    "ops/exclusion_manifest.yaml",
    "ops/e2e-test-plan.md",
    "ops/arc_solve_registry.yaml",
    "ops/arc_supervisor_refinement_ledger.json",
    "python/carnot/agentic/arc_go_explore.py",
    "python/carnot/agentic/arc_competition_agent.py",
    "python/carnot/agentic/arc_generalization_runtime.py",
    "python/carnot/agentic/arc_solver_kit.py",
    "python/carnot/experiment_7748_v674_arc_runner_qualification.py",
    "scripts/experiments/experiment_7748_v674_arc_runner_qualification.py",
    "openspec/capabilities/research-reporting/spec.md",
    "openspec/capabilities/arc-world-model-trust-energy/spec.md",
)
TESTS = (
    "tests/python/test_experiment_7748_v674_arc_runner_qualification.py",
    "tests/python/test_experiment_7735_v673_arc_organic_visits.py",
    "tests/python/test_arc_go_explore_seen_contamination_10024.py",
    "tests/python/test_experiment_4701_amortized_exploration_prior_go_explore_live.py",
    "tests/python/test_arc_competition_agent_adapter.py",
    "tests/python/test_experiment_7708_v671_arc_generalization_runner.py",
)
CONTROLS = {
    "spacing_fresh_actions": 20,
    "replay_cap": 400,
    "max_prefix": 30,
    "cell_bins": 6,
    "max_cells": 256,
    "max_actions": 12,
    "max_seconds": 45,
}
REQUIRED = {
    "worktree_imports",
    "focused_pytest",
    "changed_module_coverage",
    "changed_module_coverage_report",
    "ruff_check",
    "ruff_format",
    "changed_module_mypy",
    "scoped_spec_coverage",
    "e2e_009",
    "e2e_009_smoke",
    "e2e_011",
    "e2e_013",
}


def progress(start: float, phase: str, event: str, **detail: Any) -> None:
    print(
        f"[exp7748] {phase} {event} elapsed_s={time.monotonic() - start:.2f} {detail}", flush=True
    )


def check(name: str, path: Path, field: str, expected: Any, observed: Any) -> dict[str, Any]:
    return {
        "check": name,
        "upstream_id": name,
        "artifact_path": str(path),
        "field": field,
        "op": "==",
        "expected": expected,
        "observed": observed,
        "passed": expected == observed,
    }


def inputs() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    hashes: dict[str, Any] = {
        "eligible_producers": {},
        "historical_disqualified_sources": {},
        "missing_inputs": [],
        "pre_gate_receipts": {},
    }
    for relative in INPUTS:
        path = ROOT / relative
        present = path.is_file() and path.stat().st_size > 0
        checks.append(check("required_input", path, "readable_nonempty", True, present))
        if present:
            hashes["eligible_producers"][relative] = sha256_file(path)
        else:
            hashes["missing_inputs"].append(relative)
    for relative, marker in ((INPUTS[-2], "REQ-REPORT-7748"), (INPUTS[-1], "REQ-ARC-WMTE-7748")):
        path = ROOT / relative
        checks.append(
            check("requirement", path, marker, True, path.is_file() and marker in path.read_text())
        )
    checks.append(
        check(
            "python",
            ROOT / ".venv/bin/python",
            "executable",
            True,
            os.access(ROOT / ".venv/bin/python", os.X_OK),
        )
    )
    prior = ROOT / "results/experiment_7735_v673_arc_organic_visits.json"
    if prior.is_file():
        hashes["historical_disqualified_sources"][str(prior.relative_to(ROOT))] = sha256_file(prior)
    return checks, hashes


def manifest(arcade: Any) -> dict[str, Any]:
    roster = {str(item.game_id).split("-")[0]: item for item in arcade.available_environments}
    rows = []
    for game in GAMES:
        item = roster[game]
        baseline = [
            getattr(action, "model_dump", lambda: str(action))() for action in item.baseline_actions
        ]
        for seed in SEEDS:
            for arm in ARMS:
                rows.append(
                    {
                        "game": game,
                        "seed": seed,
                        "arm": arm,
                        "episode_id": f"{game}:{seed}:{arm}",
                        "status": "unstarted",
                    }
                )
        roster[game] = {"game_id": str(item.game_id), "baseline_actions": baseline}
    return {
        "schema": "carnot.exp7749.frozen_panel.v1",
        "games": list(GAMES),
        "seeds": list(SEEDS),
        "arms": list(ARMS),
        "controls": CONTROLS,
        "sdk_version": importlib.metadata.version("arc-agi"),
        "environment_info": {game: roster[game] for game in GAMES},
        "rows": rows,
    }


def validate(start: float) -> list[dict[str, Any]]:
    private = Path("/tmp/exp7748-validation")
    (private / "basetemp").mkdir(parents=True, exist_ok=True)
    commands = build_scoped_commands(
        ROOT,
        TESTS,
        ("python/carnot/experiment_7748_v674_arc_runner_qualification.py",),
        static_paths=("scripts/experiments/experiment_7748_v674_arc_runner_qualification.py",),
        basetemp=private / "basetemp",
        coverage_file=private / ".coverage",
    )
    pytest = str(ROOT / ".venv/bin/pytest")
    common = ("-n", "0", "-o", "addopts=", "--no-cov")
    for name, tests in (
        ("e2e_009", ("tests/python/test_arc_induction_state_persistence.py",)),
        ("e2e_011", ("tests/python/test_arc_decision_telemetry.py",)),
        (
            "e2e_013",
            (
                "tests/python/test_arc_decision_telemetry.py",
                "tests/python/test_experiment_7491_e6_timed_live_profile.py",
                "tests/python/test_experiment_7531_b2_induction_gate_measurement.py",
                "tests/python/test_semif_arc_readout_eval.py",
            ),
        ),
    ):
        commands.append(
            CommandSpec(
                name,
                (pytest, *common, f"--basetemp={private / 'basetemp' / name}", *tests, "-q"),
                "numbered_e2e_cpu",
                900,
            )
        )
    commands.append(
        CommandSpec(
            "e2e_009_smoke",
            (
                str(ROOT / ".venv/bin/python"),
                "-u",
                "scripts/arc_loop_solve.py",
                "--mechanism",
                "e3",
                "--game",
                "r11l",
                "--max-actions",
                "12",
                "--output",
                str(private / "offline_smoke.json"),
            ),
            "numbered_e2e_cpu",
            300,
        )
    )
    progress(start, "validation", "before", commands=len(commands))
    receipts = run_commands(
        ROOT,
        commands,
        log_dir=RAW / "validation",
        heartbeat_s=30,
        extra_env={"JAX_PLATFORMS": "", "CARNOT_ARC_DISABLE_INDUCTION": "1"},
    )
    progress(start, "validation", "after", passed=sum(row["passed"] for row in receipts))
    return receipts


def span(
    name: str, first: float, last: float, run_date: str, units: int, checkpoint: str | None = None
) -> dict[str, Any]:
    return {
        "name": name,
        "start_monotonic": first,
        "end_monotonic": last,
        "duration_s": last - first,
        "run_date": run_date,
        "heartbeat_times": [last],
        "completed_units": units,
        "checkpoint_hash": checkpoint,
    }


def artifact(
    rows: list[dict[str, Any]],
    checks: list[dict[str, Any]],
    hashes: dict[str, Any],
    receipts: list[dict[str, Any]],
    spans: list[dict[str, Any]],
    run_date: str,
    panel_path: Path,
    panel_hash: str | None,
    *,
    flagged: bool = False,
) -> dict[str, Any]:
    failed = [row for row in checks if not row["passed"]]
    passed = {row["name"] for row in receipts if row.get("passed") and row["exit_code"] == 0}
    validation_ok = REQUIRED <= passed and all(row.get("passed") for row in receipts)
    sdk_rows = [row for row in rows if row["claim_scope"] == "adapter_withheld_public"]
    sdk_ok = len(sdk_rows) == 3 and all(
        row["error"] is None
        and row["counts"]["sdk_transitions"] > 0
        and row["policy_entry"]["policy_class"] == "E3AgentPolicy"
        and all(action["induction_attempt_count"] == 0 for action in row["actions"])
        for row in sdk_rows
    )
    distinct_ok = any(row["arm"] == "total" for row in rows) and any(
        row["arm"] == "organic" for row in rows
    )
    ready = not failed and validation_ok and sdk_ok and distinct_ok and not flagged
    if failed:
        verdict, reason = "blocked", str(failed[0]["check"])
    elif flagged or not validation_ok or not sdk_ok or not distinct_ok:
        verdict, reason = "disqualified", "required_runner_validation"
    else:
        verdict, reason = "null", "runner_qualified_no_benefit_measurement"
    gates = {
        "validity": not failed and sdk_ok,
        "readiness": ready,
        "probability_quality": None,
        "decision_benefit": None,
        "retention": None,
        "efficiency": None,
    }
    summary = cold_reduce(rows, len(rows))
    result: dict[str, Any] = {
        "schema": "carnot.exp7748.v674.arc_runner_qualification.v1",
        "experiment_id": 7748,
        "milestone": "2026.09.674",
        "run_date": run_date,
        "honest_verdict": f"complete_{verdict}_{reason}",
        "verdict_class": verdict,
        "flagged_adversarial": flagged,
        "gate_check_summary": {"failed_count": len(failed), "failed_checks": failed},
        "acceptance_gate_results": gates,
        "rows": rows,
        "sample_size_budget": {
            "intended": 6,
            "started": summary["started"],
            "completed": summary["completed"],
            "eligible": sum(row["error"] is None for row in rows),
            "excluded": sum(bool(row["exclusions"]) for row in rows),
            "censored": sum(bool(row["censoring"]) for row in rows),
            "effective_independent_n": int(sdk_ok),
            "seeds_or_actions_multiply_n": False,
        },
        "claim_scope": ["fixture_only", "adapter_withheld_public"],
        "fresh_generalization_eligible": False,
        "inference_substrate": "host_cpu_sdk_scored_agent_no_model_load",
        "inference_substrate_class": "no_model_load",
        "planned_inference_substrate_class": "no_model_load",
        "MODEL_SPECS": [],
        "planned_MODEL_SPECS": [],
        "model_specs": [],
        "model_invoked": False,
        "invocation_counts": {
            key: 0
            for key in (
                "loads",
                "forwards",
                "generations",
                "input_tokens",
                "output_tokens",
                "failures",
                "cancellations",
            )
        },
        "execution_venue": "host",
        "execution_venue_details": {
            "host": socket.gethostname(),
            "pid": os.getpid(),
            "gpu_uuid": None,
        },
        "phase_spans": spans,
        "random_seed": {"panel_seeds": list(SEEDS), "fixture_seed": 7748, "smoke_seed": 674901},
        "source_artifact_hashes": hashes,
        "preconditions_checked": checks,
        "validation_receipts": receipts,
        "frozen_validation_scope": {
            "tests": list(TESTS),
            "changed_modules": ["python/carnot/experiment_7748_v674_arc_runner_qualification.py"],
            "unchanged_large_policy_coverage": "historical Exp7735 38% whole-policy; unchanged",
        },
        "verifier_is_oracle": True,
        "organic_runner_ready_score": int(ready),
        "arc_schedule_manifest_path": str(panel_path.relative_to(ROOT)),
        "arc_schedule_manifest_sha256": panel_hash,
        "counter_event_rows": [
            dict(event, episode_id=row["episode_id"])
            for row in rows
            for event in row["counter_event_rows"]
        ],
        "solve_provenance": {
            "fixture": "development_proxy",
            "sdk": "live_agent_self_discovery",
            "new_solve_credit": False,
        },
        "raw_reduction": summary,
        "raw_rows_sha256": sha256_file(RAW / "rows.json"),
        "administrative_readiness": None,
        "supervisor_outcome_receipts": {
            "source": "ops/arc_supervisor_refinement_ledger.json",
            "source_sha256": hashes["eligible_producers"].get(
                "ops/arc_supervisor_refinement_ledger.json"
            ),
            "historical_entries": len(
                json.loads((ROOT / "ops/arc_supervisor_refinement_ledger.json").read_text()).get(
                    "entries", {}
                )
            ),
            "current_firings": 0,
            "current_refinement_candidates": 0,
        },
        "effective_coding_backend": "codex",
        "reset_charging_interpretations": ["charged", "uncharged"],
    }
    result["reproducibility_checksum"] = (
        "sha256:"
        + hashlib.sha256(
            json.dumps(
                [hashes, CONTROLS, list(TESTS), panel_hash, result["raw_rows_sha256"], run_date],
                sort_keys=True,
            ).encode()
        ).hexdigest()
    )
    principles = {key: "Owned raw evidence bounds downstream use." for key in result}
    principles["honest_verdict"] = "Completion and scientific benefit are different facts."
    principles["organic_runner_ready_score"] = (
        "Real SDK reachability and matched controls are required."
    )
    principles["sample_size_budget"] = "Actions and seeds do not multiply independent games."
    for key in gates:
        principles[f"acceptance_gate_results.{key}"] = "Unmeasured quality or benefit stays null."
    result["field_principles"] = principles
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", default="20260927")
    parser.add_argument("--cold", type=Path)
    args = parser.parse_args(argv)
    started = time.monotonic()
    progress(started, "startup", "before", root=str(ROOT.resolve()), pid=os.getpid())
    RAW.mkdir(parents=True, exist_ok=True)
    panel_path = RAW / "arc_schedule_manifest.json"
    if args.cold:
        progress(started, "cold_reduce", "before", candidate=str(args.cold))
        candidate = json.loads(args.cold.read_text())
        raw = RAW / "rows.json"
        if sha256_file(raw) != candidate["raw_rows_sha256"]:
            raise ValueError("raw_rows_hash_mismatch")
        rows = json.loads(raw.read_text())["rows"]
        if rows != candidate["rows"]:
            raise ValueError("candidate_rows_mismatch")
        if candidate["arc_schedule_manifest_sha256"] is not None:
            if sha256_file(panel_path) != candidate["arc_schedule_manifest_sha256"]:
                raise ValueError("schedule_hash_mismatch")
            if len(json.loads(panel_path.read_text())["rows"]) != 48:
                raise ValueError("schedule_row_count")
        summary = cold_reduce(rows, len(rows))
        if summary != candidate["raw_reduction"]:
            raise ValueError("raw_reduction_mismatch")
        progress(started, "cold_reduce", "after", **summary)
        return 0

    spans: list[dict[str, Any]] = []
    progress(started, "preconditions", "before")
    first = time.monotonic()
    checks, hashes = inputs()
    arcade = None
    try:
        arcade = offline_arcade()
        roster = {str(item.game_id).split("-")[0] for item in arcade.available_environments}
        for game in GAMES:
            checks.append(check("sdk_game", ROOT / "environment_files", game, True, game in roster))
        if all(game in roster for game in GAMES):
            panel = manifest(arcade)
            if panel_path.exists() and json.loads(panel_path.read_text()) != panel:
                raise ValueError("frozen_manifest_changed")
            atomic_json(panel_path, panel)
    except Exception as exc:
        checks.append(
            check(
                "sdk_catalogue",
                ROOT / "environment_files",
                "runnable_catalogue",
                True,
                f"{type(exc).__name__}: {exc}"[:200],
            )
        )
    panel_hash = sha256_file(panel_path) if panel_path.is_file() else None
    scope_path = RAW / "frozen_affected_scope.json"
    scope = {
        "tests": list(TESTS),
        "changed_modules": ["python/carnot/experiment_7748_v674_arc_runner_qualification.py"],
        "static_paths": ["scripts/experiments/experiment_7748_v674_arc_runner_qualification.py"],
        "required_checks": sorted(REQUIRED),
        "controls": CONTROLS,
    }
    if scope_path.exists() and json.loads(scope_path.read_text()) != scope:
        raise ValueError("frozen_validation_scope_changed")
    atomic_json(scope_path, scope)
    spans.append(span("preconditions", first, time.monotonic(), args.date, len(checks), panel_hash))
    progress(started, "preconditions", "after", failed=sum(not row["passed"] for row in checks))

    rows: list[dict[str, Any]] = []
    first = time.monotonic()
    if all(row["passed"] for row in checks):
        for game, seed, source in (("fixture", 7748, _FixtureArcade()), ("r11l", 674901, arcade)):
            for arm in ARMS:
                progress(started, "episode", "before", completed=len(rows), game=game, arm=arm)
                row = run_qualification_episode(
                    game, seed, arm, source, 3 if game == "fixture" else 12
                )
                row["input_hashes"] = hashes["eligible_producers"]
                row["status"] = "completed" if row["error"] is None else "censored"
                rows.append(row)
                checkpoint = RAW / f"episode_{len(rows) - 1:02d}.json"
                atomic_json(checkpoint, row)
                progress(
                    started,
                    "episode",
                    "after",
                    completed=len(rows),
                    game=game,
                    arm=arm,
                    checkpoint_sha256=sha256_file(checkpoint),
                )
    atomic_json(RAW / "rows.json", {"rows": rows})
    spans.append(
        span(
            "episodes",
            first,
            time.monotonic(),
            args.date,
            len(rows),
            sha256_file(RAW / "rows.json"),
        )
    )
    progress(started, "episodes", "after", completed=len(rows))

    first = time.monotonic()
    receipts = validate(started) if not any(not row["passed"] for row in checks) else []
    spans.append(
        span(
            "validation",
            first,
            time.monotonic(),
            args.date,
            sum(row.get("passed", False) for row in receipts),
        )
    )
    candidate_path = RAW / "terminal_candidate.json"
    value = artifact(rows, checks, hashes, receipts, spans, args.date, panel_path, panel_hash)
    value["terminal_reader_receipts_path"] = str((RAW / "terminal_receipts.json").relative_to(ROOT))
    value["field_principles"]["terminal_reader_receipts_path"] = (
        "The exact candidate is read before publication."
    )
    atomic_json(candidate_path, value)

    python = str(ROOT / ".venv/bin/python")
    commands = [
        CommandSpec(
            "cold_reduce",
            (python, "-u", __file__, "--cold", str(candidate_path)),
            "terminal_reader",
            120,
        ),
        CommandSpec(
            "adversarial_verify",
            (python, "-u", "scripts/adversarial_verify.py", str(candidate_path)),
            "terminal_reader",
            180,
        ),
        CommandSpec(
            "strict_row_lint",
            (
                python,
                "-u",
                "scripts/verdict_row_consistency_lint.py",
                "--strict",
                str(candidate_path),
            ),
            "terminal_reader",
            180,
        ),
    ]
    progress(started, "terminal_readers", "before", candidate_sha256=sha256_file(candidate_path))
    terminal = run_commands(
        ROOT,
        commands,
        log_dir=RAW / "terminal_logs",
        heartbeat_s=30,
        extra_env={"JAX_PLATFORMS": ""},
    )
    if any(not row["passed"] for row in terminal):
        value = artifact(
            rows,
            checks,
            hashes,
            receipts,
            spans,
            args.date,
            panel_path,
            panel_hash,
            flagged=any(
                row["name"] == "adversarial_verify" and not row["passed"] for row in terminal
            ),
        )
        value["terminal_reader_receipts_path"] = str(
            (RAW / "terminal_receipts.json").relative_to(ROOT)
        )
        value["field_principles"]["terminal_reader_receipts_path"] = (
            "The exact candidate is read before publication."
        )
        atomic_json(candidate_path, value)
        progress(started, "terminal_readers", "retry_disqualified_candidate")
        terminal = run_commands(
            ROOT,
            commands,
            log_dir=RAW / "terminal_logs",
            heartbeat_s=30,
            extra_env={"JAX_PLATFORMS": ""},
        )
    atomic_json(
        RAW / "terminal_receipts.json",
        {"candidate_sha256": sha256_file(candidate_path), "commands": terminal},
    )
    progress(started, "terminal_readers", "after", passed=sum(row["passed"] for row in terminal))
    atomic_json(RESULT, value)
    progress(
        started,
        "publish",
        "after",
        result=str(RESULT),
        sha256=sha256_file(RESULT),
        verdict=value["honest_verdict"],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
