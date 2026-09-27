"""Requalify the scored ARC archive path under REQ-REPORT-7763."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import socket
import time
from typing import Any

import yaml

from carnot.agentic.arc_solver_kit import offline_arcade
from carnot.experiment_7708_v671_arc_generalization_runner import _FixtureArcade
from carnot.experiment_7763_v675_arc_runner_qualification import (
    ARMS,
    GAMES,
    SEEDS,
    cold_reduce,
    run_probe,
    schedule_rows,
)
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash, sha256_file
from carnot.reporting.experiment_7303_validation_scope import (
    CommandSpec,
    build_scoped_commands,
    run_commands,
)

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "results/raw/experiment_7763_v675_arc_runner_qualification"
RESULT = ROOT / "results/experiment_7763_v675_arc_runner_qualification.json"
SCOPE = RAW / "frozen_affected_scope.json"
PRIVATE = Path("/tmp/exp7763-v675-validation")
SOURCE_PATHS = (
    "AGENTS.md",
    "CODEX.md",
    "CLAUDE.md",
    "research-program.md",
    "ops/exclusion_manifest.yaml",
    "ops/e2e-test-plan.md",
    "ops/arc_solve_registry.yaml",
    "ops/arc_supervisor_refinement_ledger.json",
    "openspec/capabilities/research-reporting/spec.md",
    "openspec/capabilities/arc-world-model-trust-energy/spec.md",
    "python/carnot/agentic/arc_competition_agent.py",
    "python/carnot/agentic/arc_go_explore.py",
    "python/carnot/agentic/arc_generalization_runtime.py",
    "python/carnot/agentic/arc_solver_kit.py",
    "python/carnot/experiment_7748_v674_arc_runner_qualification.py",
    "python/carnot/experiment_7763_v675_arc_runner_qualification.py",
    "scripts/experiments/experiment_7763_v675_arc_runner_qualification.py",
    "results/experiment_7748_v674_arc_runner_qualification.json",
    "results/experiment_7749_arc_organic_measurement.json",
)


def progress(start: float, phase: str, event: str, units: int = 0, **detail: Any) -> None:
    """Print a flushed boundary with measured time and completed units."""
    print(
        f"[exp7763] phase={phase} event={event} elapsed_s={time.monotonic() - start:.2f} completed_units={units} {detail}",
        flush=True,
    )


def check(path: Path, field: str, expected: Any, observed: Any, upstream: str) -> dict[str, Any]:
    """Retain an exact precondition operand and the source byte hash."""
    return {
        "upstream_id": upstream,
        "artifact_path": str(path),
        "artifact_sha256": sha256_file(path) if path.is_file() else None,
        "field": field,
        "expected": expected,
        "observed": observed,
        "operator": "==",
        "passed": expected == observed,
    }


def preflight(start: float) -> tuple[list[dict[str, Any]], dict[str, Any], Any, dict[str, Any]]:
    """Check custody and catalogue before any scored action."""
    progress(start, "preconditions", "before")
    checks = [
        check(
            ROOT / p,
            "readable_nonempty",
            True,
            (ROOT / p).is_file() and (ROOT / p).stat().st_size > 0,
            p,
        )
        for p in SOURCE_PATHS
    ]
    checks += [
        check(
            ROOT / "openspec/capabilities/research-reporting/spec.md",
            "REQ-REPORT-7763",
            True,
            "REQ-REPORT-7763"
            in (ROOT / "openspec/capabilities/research-reporting/spec.md").read_text(),
            "spec",
        ),
        check(
            ROOT / ".venv/bin/python",
            "executable",
            True,
            os.access(ROOT / ".venv/bin/python", os.X_OK),
            "python",
        ),
    ]
    prior = json.loads(
        (ROOT / "results/experiment_7748_v674_arc_runner_qualification.json").read_text()
    )
    checks.append(
        check(
            ROOT / "results/experiment_7748_v674_arc_runner_qualification.json",
            "organic_runner_ready_score",
            0,
            prior.get("organic_runner_ready_score"),
            "exp7748_historical_diagnostic",
        )
    )
    pregate = ROOT / "results/experiment_7749_arc_organic_measurement.json"
    pregate_value = json.loads(pregate.read_text()) if pregate.is_file() else None
    arcade = offline_arcade()
    roster = {str(item.game_id).split("-")[0]: item for item in arcade.available_environments}
    registry_value = yaml.safe_load((ROOT / "ops/arc_solve_registry.yaml").read_text())
    registry_rows = (
        registry_value if isinstance(registry_value, list) else registry_value.get("games", [])
    )
    registry = {row.get("game"): row for row in registry_rows if isinstance(row, dict)}
    for game in GAMES:
        checks.append(
            check(
                ROOT / "ops/arc_solve_registry.yaml",
                f"registry_game:{game}",
                True,
                game in registry,
                "arc_registry",
            )
        )
        checks.append(
            check(ROOT / "environment_files", f"sdk_game:{game}", True, game in roster, "arc_sdk")
        )
    hashes = {
        p: {
            "sha256": sha256_file(ROOT / p),
            "date": "20260927",
            "imported_fields": ["organic_runner_ready_score"] if "7748" in p else [],
            "eligible": "7748" not in p,
        }
        for p in SOURCE_PATHS
        if (ROOT / p).is_file()
    }
    hashes["results/experiment_7749_v674_arc_generalization.json"] = {
        "sha256": None,
        "date": None,
        "imported_fields": [],
        "eligible": False,
        "role": "absent_downstream_producer",
    }
    if pregate_value is not None:
        hashes[str(pregate.relative_to(ROOT))]["imported_fields"] = [
            "gate_check_summary",
            "honest_verdict",
        ]
    resource = {
        "disk_free_bytes": shutil.disk_usage(ROOT).free,
        "cpu_count": os.cpu_count(),
        "host": socket.gethostname(),
        "sdk_version": importlib.metadata.version("arc-agi"),
        "backend": "offline_arcade_cpu",
        "pre_gate_receipt": str(pregate.relative_to(ROOT)) if pregate_value else None,
        "pre_gate_summary": pregate_value.get("gate_check_summary") if pregate_value else None,
        "historical_runner_verdict": prior.get("honest_verdict"),
    }
    progress(
        start, "preconditions", "after", len(checks), failed=sum(not c["passed"] for c in checks)
    )
    return checks, hashes, arcade, {"roster": roster, "registry": registry, "resource": resource}


def panel(context: dict[str, Any]) -> dict[str, Any]:
    """Freeze baseline actions and all intended comparisons before results."""
    return {
        "schema": "carnot.exp7763.frozen_arc_panel.v1",
        "games": list(GAMES),
        "seeds": list(SEEDS),
        "arms": list(ARMS),
        "round_robin_order": "game_seed_arm",
        "max_actions_charged": 2000,
        "max_seconds_per_episode": 75,
        "reset_charging_conventions": ["charged", "uncharged"],
        "controls": {
            "spacing_fresh_actions": 20,
            "replay_cap": 400,
            "max_prefix": 30,
            "cell_bins": 6,
            "max_cells": 256,
        },
        "sdk_version": context["resource"]["sdk_version"],
        "environment_info": {
            game: {
                "game_id": str(context["roster"][game].game_id),
                "baseline_actions": [
                    str(action) for action in context["roster"][game].baseline_actions
                ],
            }
            for game in GAMES
        },
        "registry_precheck": {
            game: {
                "levels_reproduced": context["registry"][game].get("levels_reproduced"),
                "full_game_clear": context["registry"][game].get("full_game_clear"),
                "historically_public": True,
            }
            for game in GAMES
        },
        "rows": schedule_rows(),
    }


def validation(start: float) -> list[dict[str, Any]]:
    """Run the frozen affected scope and every applicable CPU E2E."""
    scope = json.loads(SCOPE.read_text())
    base = PRIVATE / "basetemp"
    base.mkdir(parents=True, exist_ok=True)
    commands = build_scoped_commands(
        ROOT,
        scope["tests"],
        scope["changed_modules"],
        static_paths=scope["static_paths"],
        basetemp=base,
        coverage_file=PRIVATE / ".coverage",
    )
    python = str(ROOT / ".venv/bin/python")
    pytest = str(ROOT / ".venv/bin/pytest")
    common = ("-n", "0", "-o", "addopts=", "--no-cov")
    commands.append(
        CommandSpec(
            "coverage_path_subprocess",
            (
                pytest,
                *common,
                f"--basetemp={base / 'coverage_path'}",
                "tests/python/test_experiment_4701_amortized_exploration_prior_go_explore_live.py::test_scenario_arc_wmte_4701_stepwise_orders_prior_and_exposes_archive",
                "-q",
            ),
            "real_unskipped_subprocess",
            180,
        )
    )
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
                (pytest, *common, f"--basetemp={base / name}", *tests, "-q"),
                "numbered_e2e_cpu",
                900,
            )
        )
    commands.append(
        CommandSpec(
            "e2e_009_smoke",
            (
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
                str(PRIVATE / "offline_smoke.json"),
            ),
            "actual_offline_sdk_smoke",
            300,
        )
    )
    commands.append(
        CommandSpec(
            "full_pytest",
            (pytest, *common, f"--basetemp={base / 'full'}", "tests/python", "-q"),
            "all_python_tests",
            1800,
        )
    )
    progress(start, "validation", "before", 0, commands=len(commands))
    receipts = run_commands(
        ROOT,
        commands,
        log_dir=RAW / "validation",
        heartbeat_s=30,
        extra_env={"JAX_PLATFORMS": "cpu", "CARNOT_ARC_DISABLE_INDUCTION": "1"},
    )
    progress(
        start, "validation", "after", len(receipts), passed=sum(row["passed"] for row in receipts)
    )
    return receipts


def artifact(
    schedule: list[dict[str, Any]],
    probes: list[dict[str, Any]],
    checks: list[dict[str, Any]],
    hashes: dict[str, Any],
    resource: dict[str, Any],
    receipts: list[dict[str, Any]],
    spans: list[dict[str, Any]],
    started: float,
    run_date: str,
    panel_path: Path,
) -> dict[str, Any]:
    """Assemble a terminal record whose readiness follows every required receipt."""
    scope = json.loads(SCOPE.read_text())
    failed = [row for row in checks if not row["passed"]]
    receipt_names = {
        row["name"] for row in receipts if row.get("passed") and row.get("exit_code") == 0
    }
    missing = sorted(
        set(scope["required_checks"])
        - {"cold_reduce", "adversarial_verify", "strict_row_lint"}
        - receipt_names
    )
    failed += [
        check(
            ROOT / "results/raw/experiment_7763_v675_arc_runner_qualification/validation",
            name,
            0,
            next((row["exit_code"] for row in receipts if row["name"] == name), "missing"),
            "current_validation",
        )
        for name in missing
    ]
    sdk = [row for row in probes if row["claim_scope"] == "adapter_withheld_public"]
    sdk_ok = len(sdk) == 3 and all(
        row["error"] is None
        and row["counts"]["sdk_transitions"] > 0
        and row["policy_entry"]["policy_class"] == "E3AgentPolicy"
        for row in sdk
    )
    summary = cold_reduce(schedule, probes)
    ready = not failed and sdk_ok
    verdict_class = (
        "blocked"
        if any(row["upstream_id"] != "current_validation" for row in failed)
        else "null"
        if ready
        else "disqualified"
    )
    value: dict[str, Any] = {
        "schema": "carnot.exp7763.v675.arc_runner_qualification.v1",
        "experiment_id": 7763,
        "milestone": "2026.09.675",
        "run_date": run_date,
        "honest_verdict": "complete_null_runner_qualified_no_benefit_measurement"
        if ready
        else "complete_blocked_preconditions"
        if verdict_class == "blocked"
        else "complete_disqualified_required_runner_validation",
        "verdict_class": verdict_class,
        "flagged_adversarial": False,
        "gate_check_summary": failed,
        "rows": schedule,
        "probe_rows": probes,
        "acceptance_gate_results": {
            "validity": not failed and sdk_ok,
            "readiness": ready,
            "probability_quality": None,
            "decision_benefit": None,
            "retention": None,
            "efficiency": None,
        },
        "duration_s": time.monotonic() - started,
        "phase_spans": spans,
        "random_seed": {"panel_seeds": list(SEEDS), "fixture_seed": 67500, "sdk_probe_seed": 67501},
        "reproducibility_checksum": canonical_hash(
            {
                "source_artifact_hashes": hashes,
                "scope": json.loads(SCOPE.read_text()),
                "panel_sha256": sha256_file(panel_path),
                "raw_sha256": sha256_file(RAW / "raw_probes.json"),
                "code_sha256": sha256_file(Path(__file__)),
            }
        ),
        "sample_size_budget": {
            "intended": 48,
            "eligible": 48,
            "started": 0,
            "completed": 0,
            "excluded": 0,
            "censored": 0,
            "effective_independent_n": 0,
            "qualification_probes_started": summary["started"],
            "qualification_probes_completed": summary["completed"],
            "seeds_or_actions_multiply_n": False,
        },
        "source_artifact_hashes": hashes,
        "preconditions_checked": {"checks": checks, "resources": resource},
        "validation_receipts": receipts,
        "frozen_validation_scope": json.loads(SCOPE.read_text()),
        "verifier_is_oracle": True,
        "claim_scope": [
            "circular_positive_fixture_mechanics",
            "adapter_withheld_public_transport_only",
        ],
        "inference_substrate": "offline_arcade_live_agent_runtime_self_discovery_no_llm",
        "inference_substrate_class": "no_model_load",
        "planned_inference_substrate_class": "no_model_load",
        "MODEL_SPECS": [],
        "model_specs": [],
        "model_invocation_counts": {
            "loads": 0,
            "generations": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "loaded_files": [],
        },
        "model_invoked": False,
        "organic_runner_ready_score": int(ready),
        "arc_panel_manifest_path": str(panel_path.relative_to(ROOT)),
        "arc_panel_manifest_sha256": sha256_file(panel_path),
        "solve_provenance": {
            "fixture": "development_proxy",
            "sdk": "live_agent_self_discovery",
            "new_solve_credit": False,
        },
        "reset_charging_interpretations": {
            "charged": "RESET counts toward the 2000 action cap",
            "uncharged": "RESET is excluded from the alternate reported count",
        },
        "raw_reduction": summary,
        "raw_probes_sha256": sha256_file(RAW / "raw_probes.json"),
        "terminal_reader_receipts_path": str((RAW / "terminal_receipts.json").relative_to(ROOT)),
        "execution_venue": "host",
        "execution_venue_details": {
            "host": socket.gethostname(),
            "pid": os.getpid(),
            "gpu_uuid": None,
        },
    }
    principles = {
        key: "Owned evidence and exact inputs bound downstream interpretation." for key in value
    }
    principles.update(
        {
            "honest_verdict": "Terminal records do not retry unchanged inputs.",
            "verdict_class": "The claim class travels with evidence.",
            "gate_check_summary": "Missing producers and failed thresholds are different causes.",
            "rows": "Aggregates must be recomputable without rerunning science.",
            "sample_size_budget": "Seeds and actions do not increase independent game count.",
            "organic_runner_ready_score": "The next panel needs a qualified scored path.",
        }
    )
    for gate in value["acceptance_gate_results"]:
        principles[f"acceptance_gate_results.{gate}"] = (
            "Unmeasured quality or benefit remains null."
        )
    value["field_principles"] = principles
    return value


def main(argv: list[str] | None = None) -> int:
    """Run qualification or independently reduce raw evidence in a fresh process."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", default="20260927")
    parser.add_argument("--cold", type=Path)
    args = parser.parse_args(argv)
    start = time.monotonic()
    progress(start, "startup", "before", root=str(ROOT), pid=os.getpid())
    if args.cold:
        progress(start, "cold_reduce", "before")
        raw = json.loads(args.cold.read_text())
        observed = cold_reduce(raw["schedule"], raw["probes"])
        if observed != raw["summary"]:
            raise ValueError("raw_reduction_mismatch")
        progress(start, "cold_reduce", "after", observed["started"], **observed)
        return 0
    RAW.mkdir(parents=True, exist_ok=True)
    spans: list[dict[str, Any]] = []
    phase = time.monotonic()
    checks, hashes, arcade, context = preflight(start)
    panel_path = RAW / "arc_panel_manifest.json"
    frozen = panel(context)
    if panel_path.exists() and json.loads(panel_path.read_text()) != frozen:
        raise ValueError("frozen_panel_changed")
    atomic_json(panel_path, frozen)
    spans.append(
        {
            "phase": "preconditions",
            "duration_s": time.monotonic() - phase,
            "completed_units": len(checks),
        }
    )
    schedule = frozen["rows"]
    probes: list[dict[str, Any]] = []
    phase = time.monotonic()
    if all(row["passed"] for row in checks):
        for game, seed, source in (("fixture", 67500, _FixtureArcade()), ("r11l", 67501, arcade)):
            for arm in ARMS:
                progress(
                    start,
                    "scored_probe",
                    "before",
                    len(probes),
                    game=game,
                    arm=arm,
                    model_loads=0,
                    generations=0,
                )
                row = run_probe(game, seed, arm, source, 3 if game == "fixture" else 12)
                raw_path = RAW / f"probe_{len(probes):02d}.json"
                atomic_json(raw_path, row)
                row["raw_path"] = str(raw_path.relative_to(ROOT))
                row["raw_sha256"] = sha256_file(raw_path)
                probes.append(row)
                progress(
                    start,
                    "scored_probe",
                    "after",
                    len(probes),
                    game=game,
                    arm=arm,
                    error=row["error"],
                )
    summary = cold_reduce(schedule, probes)
    raw_path = RAW / "raw_probes.json"
    atomic_json(raw_path, {"schedule": schedule, "probes": probes, "summary": summary})
    spans.append(
        {
            "phase": "scored_probes",
            "duration_s": time.monotonic() - phase,
            "completed_units": len(probes),
        }
    )
    python = str(ROOT / ".venv/bin/python")
    progress(start, "cold_reduce", "before", len(probes))
    phase = time.monotonic()
    cold = run_commands(
        ROOT,
        [
            CommandSpec(
                "cold_reduce",
                (python, "-u", __file__, "--cold", str(raw_path)),
                "fresh_process_raw_reduction",
                120,
            )
        ],
        log_dir=RAW / "cold_logs",
        heartbeat_s=30,
        extra_env={"JAX_PLATFORMS": "cpu"},
    )
    spans.append(
        {
            "phase": "cold_reduce",
            "duration_s": time.monotonic() - phase,
            "completed_units": int(cold[0]["passed"]),
        }
    )
    progress(start, "cold_reduce", "after", int(cold[0]["passed"]))
    phase = time.monotonic()
    receipts = validation(start) if all(row["passed"] for row in checks) else []
    receipts.extend(cold)
    spans.append(
        {
            "phase": "validation",
            "duration_s": time.monotonic() - phase,
            "completed_units": sum(row["passed"] for row in receipts),
        }
    )
    value = artifact(
        schedule,
        probes,
        checks,
        hashes,
        context["resource"],
        receipts,
        spans,
        start,
        args.date,
        panel_path,
    )
    candidate = RAW / "terminal_candidate.json"
    atomic_json(candidate, value)
    progress(start, "terminal_readers", "before", 0, candidate_sha256=sha256_file(candidate))
    phase = time.monotonic()
    terminal = run_commands(
        ROOT,
        [
            CommandSpec(
                "adversarial_verify",
                (python, "-u", "scripts/adversarial_verify.py", str(candidate)),
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
                    str(candidate),
                ),
                "terminal_reader",
                180,
            ),
        ],
        log_dir=RAW / "terminal_logs",
        heartbeat_s=30,
        extra_env={"JAX_PLATFORMS": "cpu"},
    )
    atomic_json(
        RAW / "terminal_receipts.json",
        {
            "candidate_sha256": sha256_file(candidate),
            "commands": terminal,
            "duration_s": time.monotonic() - phase,
        },
    )
    progress(
        start,
        "terminal_readers",
        "after",
        len(terminal),
        passed=sum(row["passed"] for row in terminal),
    )
    if any(not row["passed"] for row in terminal):
        value["flagged_adversarial"] = any(
            row["name"] == "adversarial_verify" and not row["passed"] for row in terminal
        )
        value["organic_runner_ready_score"] = 0
        value["acceptance_gate_results"]["readiness"] = False
        value["verdict_class"] = "disqualified"
        value["honest_verdict"] = "complete_disqualified_terminal_reader"
        for row in terminal:
            if not row["passed"]:
                value["gate_check_summary"].append(
                    check(
                        ROOT / row["log_path"], row["name"], 0, row["exit_code"], "terminal_reader"
                    )
                )
    progress(start, "publish", "before", 0)
    atomic_json(RESULT, value)
    progress(
        start,
        "publish",
        "after",
        1,
        result=str(RESULT),
        sha256=sha256_file(RESULT),
        verdict=value["honest_verdict"],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
