"""Publish V678 scored ARC runner evidence for REQ-REPORT-7803."""

from __future__ import annotations

import argparse
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
from carnot.experiment_7763_v675_arc_runner_qualification import GAMES, run_probe
from carnot.experiment_7776_v676_arc_runner_qualification import reduce_probe_evidence
from carnot.experiment_7803_v678_arc_runner_qualification import (
    classify_runner,
    freeze_panel,
    positive_selector_fixture,
)
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash, sha256_file
from carnot.reporting.experiment_7303_validation_scope import (
    CommandSpec,
    build_scoped_commands,
    run_commands,
)

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "results/raw/experiment_7803_v678_arc_runner_qualification"
RESULT = ROOT / "results/experiment_7803_v678_arc_runner_qualification.json"
SCOPE = RAW / "frozen_affected_scope.json"
PRIVATE = Path("/tmp/exp7803-v678-validation")
OLD = ROOT / "results/experiment_7790_v677_arc_runner_qualification.json"
OLD_PANEL = (
    ROOT / "results/raw/experiment_7790_v677_arc_runner_qualification/arc_panel_manifest.json"
)
SOURCE_PATHS = (
    "AGENTS.md",
    "CODEX.md",
    "CLAUDE.md",
    "research-program.md",
    "ops/exclusion_manifest.yaml",
    "ops/e2e-test-plan.md",
    "ops/arc_solve_registry.yaml",
    "openspec/capabilities/research-reporting/spec.md",
    "openspec/capabilities/arc-world-model-trust-energy/spec.md",
    "python/carnot/agentic/arc_competition_agent.py",
    "python/carnot/agentic/arc_graph_explore.py",
    "python/carnot/agentic/arc_solver_kit.py",
    "python/carnot/experiment_7803_v678_arc_runner_qualification.py",
    "scripts/experiments/experiment_7803_v678_arc_runner_qualification.py",
    "results/experiment_7790_v677_arc_runner_qualification.json",
    "results/raw/experiment_7790_v677_arc_runner_qualification/arc_panel_manifest.json",
    "results/raw/experiment_7803_v678_arc_runner_qualification/frozen_affected_scope.json",
)


def progress(start: float, phase: str, event: str, units: int = 0, **detail: Any) -> None:
    """Show real elapsed time and completed units at each task boundary."""
    print(
        f"[exp7803] phase={phase} event={event} elapsed_s={time.monotonic() - start:.2f} completed_units={units} {detail}",
        flush=True,
    )


def check(path: Path, field: str, expected: Any, observed: Any, upstream: str) -> dict[str, Any]:
    """Keep the failed operand and exact source bytes for a cold reader."""
    return {
        "upstream_id": upstream,
        "artifact_path": str(path),
        "artifact_sha256": sha256_file(path) if path.is_file() else None,
        "field": field,
        "operator": "==",
        "expected": expected,
        "observed": observed,
        "passed": expected == observed,
    }


def preflight(start: float) -> tuple[list[dict[str, Any]], dict[str, Any], Any, dict[str, Any]]:
    """Check named inputs, SDK roster, registry exposure, backend, and resources first."""
    progress(start, "preconditions", "before")
    checks = [
        check(
            ROOT / name,
            "readable_nonempty",
            True,
            (ROOT / name).is_file() and (ROOT / name).stat().st_size > 0,
            name,
        )
        for name in SOURCE_PATHS
    ]
    checks.append(
        check(
            ROOT / ".venv/bin/python",
            "executable",
            True,
            os.access(ROOT / ".venv/bin/python", os.X_OK),
            "python",
        )
    )
    prior = json.loads(OLD.read_text()) if OLD.is_file() else {}
    checks.append(
        check(
            OLD,
            "honest_verdict",
            "complete_disqualified_required_runner_validation",
            prior.get("honest_verdict"),
            "exp7790_historical",
        )
    )
    checks.append(
        check(
            OLD,
            "organic_runner_ready_score",
            0,
            prior.get("organic_runner_ready_score"),
            "exp7790_historical",
        )
    )
    panel = freeze_panel(OLD_PANEL) if OLD_PANEL.is_file() else None
    checks.append(check(OLD_PANEL, "frozen_panel", True, panel is not None, "exp7790_panel"))
    arcade = offline_arcade()
    roster = {str(item.game_id).split("-")[0] for item in arcade.available_environments}
    registry_data = yaml.safe_load((ROOT / "ops/arc_solve_registry.yaml").read_text())
    registry_rows = (
        registry_data if isinstance(registry_data, list) else registry_data.get("games", [])
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
    for game in ("r11l", "cd82"):
        checks.append(
            check(ROOT / "environment_files", f"probe_game:{game}", True, game in roster, "arc_sdk")
        )
    checks.append(
        check(
            ROOT,
            "free_bytes_at_least_1GiB",
            True,
            shutil.disk_usage(ROOT).free >= 1024**3,
            "resources",
        )
    )
    checks.append(
        check(
            ROOT / ".venv/bin/python",
            "cpu_backend",
            "cpu",
            os.environ.get("JAX_PLATFORMS"),
            "backend",
        )
    )
    hashes = {
        name: {
            "sha256": sha256_file(ROOT / name),
            "date": "20260928",
            "imported_fields": [],
            "eligible": True,
        }
        for name in SOURCE_PATHS
        if (ROOT / name).is_file()
    }
    producer = ROOT / "results/experiment_7749_v674_arc_generalization.json"
    pregate = ROOT / "results/experiment_7749_arc_organic_measurement.json"
    hashes[str(producer.relative_to(ROOT))] = {
        "sha256": sha256_file(producer) if producer.is_file() else None,
        "date": None,
        "imported_fields": [],
        "eligible": False,
        "role": "downstream_science_producer",
    }
    hashes[str(pregate.relative_to(ROOT))] = {
        "sha256": sha256_file(pregate) if pregate.is_file() else None,
        "date": None,
        "imported_fields": ["gate_check_summary"] if pregate.is_file() else [],
        "eligible": False,
        "role": "conductor_pre_gate_receipt",
    }
    resources = {
        "disk_free_bytes": shutil.disk_usage(ROOT).free,
        "cpu_count": os.cpu_count(),
        "host": socket.gethostname(),
        "sdk_version": importlib.metadata.version("arc-agi"),
        "backend": "offline_arcade_cpu",
        "science_producer_present": producer.is_file(),
        "conductor_pre_gate_present": pregate.is_file(),
        "historical_runner_verdict": prior.get("honest_verdict"),
    }
    progress(
        start,
        "preconditions",
        "after",
        len(checks),
        failed=sum(not row["passed"] for row in checks),
    )
    return checks, hashes, arcade, resources


def validation(start: float, scope: dict[str, Any]) -> list[dict[str, Any]]:
    """Run exactly the frozen affected commands plus named E2E and one full-suite check."""
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
    frozen = {row["name"]: row["command_argv"] for row in scope["command_argv"]}
    for spec in commands:
        if list(spec.argv) != frozen[spec.name]:
            raise ValueError(f"validation_argv_changed:{spec.name}")
    for name in ("e2e_009", "e2e_011", "e2e_013", "e2e_009_smoke", "full_python_suite"):
        commands.append(CommandSpec(name, tuple(frozen[name]), "frozen_required", 1800))
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


def terminal_readers(start: float, candidate: Path, label: str) -> list[dict[str, Any]]:
    """Use both cold readers on the exact candidate bytes before publication."""
    python = str(ROOT / ".venv/bin/python")
    commands = [
        CommandSpec(
            "adversarial_verify",
            (python, "-u", "scripts/adversarial_verify.py", str(candidate)),
            "terminal_reader",
            180,
        ),
        CommandSpec(
            "strict_row_lint",
            (python, "-u", "scripts/verdict_row_consistency_lint.py", "--strict", str(candidate)),
            "terminal_reader",
            180,
        ),
    ]
    progress(start, "terminal_readers", "before", 0, label=label, sha256=sha256_file(candidate))
    rows = run_commands(
        ROOT,
        commands,
        log_dir=RAW / f"terminal_{label}",
        heartbeat_s=30,
        extra_env={"JAX_PLATFORMS": "cpu"},
    )
    progress(
        start, "terminal_readers", "after", len(rows), passed=sum(row["passed"] for row in rows)
    )
    return rows


def build_artifact(
    start: float,
    spans: list[dict[str, Any]],
    checks: list[dict[str, Any]],
    hashes: dict[str, Any],
    resources: dict[str, Any],
    panel_path: Path,
    probes: list[dict[str, Any]],
    fixture: dict[str, Any],
    summary: dict[str, int],
    receipts: list[dict[str, Any]],
    date: str,
) -> dict[str, Any]:
    """Preserve all scheduled units and separate transport from unmeasured benefit."""
    scope = json.loads(SCOPE.read_text())
    panel = json.loads(panel_path.read_text())
    budget = {
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
    }
    gates = {
        "validity": False,
        "readiness": False,
        "probability_quality": None,
        "decision_benefit": None,
        "retention": None,
        "efficiency": None,
    }
    value: dict[str, Any] = {
        "schema": "carnot.exp7803.v678.arc_runner_qualification.v1",
        "experiment_id": 7803,
        "milestone": "2026.09.678",
        "run_date": date,
        "honest_verdict": "complete_null_runner_qualification_pending_terminal",
        "verdict_class": "null",
        "flagged_adversarial": False,
        "gate_check_summary": [row for row in checks if not row["passed"]],
        "rows": panel["rows"],
        "probe_rows": probes,
        "transition_rows": [
            {
                "episode_id": row["episode_id"],
                "game": row["game"],
                "arm": row["arm"],
                "actions": row["actions"],
                "sdk_score_inputs": {
                    "actions_charged": row["actions_charged"],
                    "peak_level": row["raw_metrics"]["peak_level"],
                },
            }
            for row in probes
        ],
        "selector_rows": fixture,
        "acceptance_gate_results": gates,
        "duration_s": time.monotonic() - start,
        "phase_spans": spans,
        "random_seed": {
            "panel_seeds": panel["seeds"],
            "probe_seed": 67501,
            "selector_fixture": "deterministic",
        },
        "reproducibility_checksum": canonical_hash(
            {
                "sources": hashes,
                "scope": scope,
                "panel_sha256": sha256_file(panel_path),
                "raw_sha256": sha256_file(RAW / "raw_probes.json"),
                "roles": panel["arms"],
                "controls": panel["controls"],
                "seeds": panel["seeds"],
            }
        ),
        "sample_size_budget": budget,
        "source_artifact_hashes": hashes,
        "preconditions_checked": {"checks": checks, "resources": resources},
        "validation_receipts": receipts,
        "frozen_validation_scope": scope,
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
        "organic_runner_ready_score": 0,
        "arc_panel_manifest_path": str(panel_path.relative_to(ROOT)),
        "arc_panel_manifest_sha256": sha256_file(panel_path),
        "solve_provenance": {
            "fixture": "development_proxy",
            "sdk": "live_agent_self_discovery",
            "new_solve_credit": False,
        },
        "raw_reduction": summary,
        "raw_probes_sha256": sha256_file(RAW / "raw_probes.json"),
        "execution_venue": "host",
        "execution_venue_details": {
            "host": socket.gethostname(),
            "pid": os.getpid(),
            "gpu_uuid": None,
        },
        "repository_health": {
            "historical_exp7790_verdict": resources["historical_runner_verdict"],
            "broad_suite_current": "required_full_python_suite_receipt",
        },
        "reset_charging_interpretations": {
            "charged": {row["episode_id"]: row["actions_charged"] for row in probes},
            "uncharged": {
                row["episode_id"]: row["actions_charged"]
                - sum(action["action"] == "RESET" for action in row["actions"])
                for row in probes
            },
            "budget_interpretation": "charged",
        },
        "supervisor_outcome_receipts": [
            {
                "episode_id": row["episode_id"],
                "observed_goal_firings": sum(
                    action["goal_firing"] is not None for action in row["actions"]
                ),
                "arm_changed_by_supervisor": False,
                "source": "raw_action_telemetry",
            }
            for row in probes
        ],
    }
    value["field_principles"] = {key: "Exact current evidence bounds this field." for key in value}
    value["field_principles"].update(
        {
            "honest_verdict": "A terminal result has one owner.",
            "gate_check_summary": "Missing evidence differs from a scientific null.",
            "rows": "Recompute comparisons from independent units.",
            "organic_runner_ready_score": "A scored transport fixture proves no benefit.",
            "sample_size_budget": "Views and seeds do not add independent game families.",
        }
    )
    for gate in gates:
        value["field_principles"][f"acceptance_gate_results.{gate}"] = "Unmeasured gain stays null."
    return value


def main(argv: list[str] | None = None) -> int:
    """Qualify the current scored path and atomically publish its final gate state."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", default="20260928")
    parser.add_argument("--cold", type=Path)
    args = parser.parse_args(argv)
    start = time.monotonic()
    progress(start, "startup", "before", pid=os.getpid())
    if args.cold:
        progress(start, "cold_reduce", "before")
        raw = json.loads(args.cold.read_text())
        if reduce_probe_evidence(raw["schedule"], raw["probes"]) != raw["summary"]:
            raise ValueError("raw_reduction_mismatch")
        progress(start, "cold_reduce", "after", raw["summary"]["started"])
        return 0
    if args.date != "20260928":
        raise ValueError("run_date_must_match_milestone")
    RAW.mkdir(parents=True, exist_ok=True)
    PRIVATE.mkdir(parents=True, exist_ok=True)
    os.environ["CARNOT_ARC_DISABLE_INDUCTION"] = "1"
    for phase in ("model_load", "generation"):
        progress(start, phase, "before", 0)
        progress(start, phase, "after", 0)
    spans: list[dict[str, Any]] = []
    phase = time.monotonic()
    checks, hashes, arcade, resources = preflight(start)
    spans.append(
        {
            "phase": "preconditions",
            "duration_s": time.monotonic() - phase,
            "completed_units": len(checks),
        }
    )
    progress(start, "panel_freeze", "before")
    phase = time.monotonic()
    panel = freeze_panel(OLD_PANEL)
    panel["schema"] = "carnot.exp7803.frozen_arc_panel.v1"
    panel["source_panel_sha256"] = sha256_file(OLD_PANEL)
    panel_path = RAW / "arc_panel_manifest.json"
    atomic_json(panel_path, panel)
    spans.append(
        {
            "phase": "panel_freeze",
            "duration_s": time.monotonic() - phase,
            "completed_units": len(panel["rows"]),
        }
    )
    progress(start, "panel_freeze", "after", len(panel["rows"]), sha256=sha256_file(panel_path))
    fixture: dict[str, Any] = {}
    probes: list[dict[str, Any]] = []
    phase = time.monotonic()
    if all(row["passed"] for row in checks):
        progress(start, "selector_fixture", "before")
        fixture = positive_selector_fixture()
        atomic_json(RAW / "selector_fixture.json", fixture)
        progress(start, "selector_fixture", "after", 1)
        for game in ("r11l", "cd82"):
            for arm in ("off", "total", "organic"):
                progress(start, "scored_probe", "before", len(probes), game=game, arm=arm)
                row = run_probe(game, 67501, arm, arcade, 12)
                path = RAW / f"probe_{len(probes):02d}.json"
                atomic_json(path, row)
                row["raw_path"] = str(path.relative_to(ROOT))
                row["raw_sha256"] = sha256_file(path)
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
    summary = reduce_probe_evidence(panel["rows"], probes)
    raw_path = RAW / "raw_probes.json"
    atomic_json(raw_path, {"schedule": panel["rows"], "probes": probes, "summary": summary})
    spans.append(
        {
            "phase": "scored_probes",
            "duration_s": time.monotonic() - phase,
            "completed_units": len(probes),
        }
    )
    progress(start, "cold_reduce", "before", len(probes))
    phase = time.monotonic()
    frozen = json.loads(SCOPE.read_text())
    cold_argv = next(
        row["command_argv"] for row in frozen["command_argv"] if row["name"] == "cold_reduce"
    )
    cold = run_commands(
        ROOT,
        [CommandSpec("cold_reduce", tuple(cold_argv), "fresh_process_raw_reduction", 120)],
        log_dir=RAW / "cold_logs",
        heartbeat_s=30,
        extra_env={"JAX_PLATFORMS": "cpu", "CARNOT_ARC_DISABLE_INDUCTION": "1"},
    )
    spans.append(
        {
            "phase": "cold_reduce",
            "duration_s": time.monotonic() - phase,
            "completed_units": int(cold[0]["passed"]),
        }
    )
    progress(start, "cold_reduce", "after", int(cold[0]["passed"]))
    receipts = list(cold)
    if all(row["passed"] for row in checks):
        phase = time.monotonic()
        receipts.extend(validation(start, frozen))
        spans.append(
            {
                "phase": "validation",
                "duration_s": time.monotonic() - phase,
                "completed_units": sum(row["passed"] for row in receipts),
            }
        )
    value = build_artifact(
        start,
        spans,
        checks,
        hashes,
        resources,
        panel_path,
        probes,
        fixture,
        summary,
        receipts,
        args.date,
    )
    if any(not row["passed"] for row in checks):
        value["honest_verdict"] = "complete_blocked_preconditions"
        value["verdict_class"] = "blocked"
    candidate = RAW / "terminal_candidate.json"
    progress(start, "candidate", "before")
    atomic_json(candidate, value)
    progress(start, "candidate", "after", 1, sha256=sha256_file(candidate))
    phase = time.monotonic()
    first = terminal_readers(start, candidate, "preliminary")
    ready, failed = classify_runner([*receipts, *first], frozen["required_checks"], probes)
    readers = first
    if ready and all(row["passed"] for row in checks):
        value["organic_runner_ready_score"] = 1
        value["acceptance_gate_results"].update(validity=True, readiness=True)
        value["honest_verdict"] = "complete_null_runner_qualified_no_benefit_measurement"
        ready_path = RAW / "terminal_candidate_ready.json"
        atomic_json(ready_path, value)
        readers = terminal_readers(start, ready_path, "ready")
        ready, failed = classify_runner([*receipts, *readers], frozen["required_checks"], probes)
    if not ready:
        blocked = any(not row["passed"] for row in checks)
        value["organic_runner_ready_score"] = 0
        value["acceptance_gate_results"].update(validity=False, readiness=False)
        value["honest_verdict"] = (
            "complete_blocked_preconditions"
            if blocked
            else "complete_disqualified_required_runner_validation"
        )
        value["verdict_class"] = "blocked" if blocked else "disqualified"
        for name in failed:
            receipt = next((row for row in [*receipts, *readers] if row["name"] == name), None)
            path = ROOT / receipt["log_path"] if receipt else RAW / "validation"
            value["gate_check_summary"].append(
                check(
                    path,
                    name,
                    0,
                    receipt["exit_code"] if receipt else "missing",
                    "current_validation",
                )
            )
    value["flagged_adversarial"] = any(
        row["name"] == "adversarial_verify" and not row["passed"] for row in readers
    )
    value["validation_receipts"] = [*receipts, *readers]
    spans.append(
        {
            "phase": "terminal_readers",
            "duration_s": time.monotonic() - phase,
            "completed_units": sum(row["passed"] for row in readers),
        }
    )
    atomic_json(
        RAW / "terminal_receipts.json",
        {
            "preliminary_candidate_sha256": sha256_file(candidate),
            "preliminary_commands": first,
            "final_commands": readers,
        },
    )
    value["phase_spans"] = spans
    value["duration_s"] = time.monotonic() - start
    progress(start, "publish", "before", verdict=value["honest_verdict"])
    atomic_json(RESULT, value)
    progress(start, "publish", "after", 1, sha256=sha256_file(RESULT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
