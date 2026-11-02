"""REQ-REPORT-7735: CPU custody for organic Go-Explore visits."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
from typing import Any

from carnot.agentic.arc_competition_agent import make_carnot_agent
from carnot.agentic.arc_generalization_runtime import run_episode
from carnot.experiment_7708_v671_arc_generalization_runner import _FixtureArcade
from carnot.reporting.current_work_receipt import atomic_json, sha256_file

ROOT = Path(__file__).resolve().parents[2]
RESULT = ROOT / "results/experiment_7735_v673_arc_organic_visits.json"
RAW = ROOT / "results/raw/experiment_7735_v673_arc_organic_visits"
SCOPE = {
    "tests": ["tests/python/test_experiment_7735_v673_arc_organic_visits.py"],
    "changed_modules": [
        "python/carnot/agentic/arc_go_explore.py",
        "python/carnot/agentic/arc_competition_agent.py",
        "python/carnot/experiment_7735_v673_arc_organic_visits.py",
    ],
    "e2e": ["E2E-009", "E2E-011", "E2E-013"],
}
INPUTS = (
    "AGENTS.md",
    "CODEX.md",
    "CLAUDE.md",
    "research-program.md",
    "ops/exclusion_manifest.yaml",
    "ops/e2e-test-plan.md",
    "ops/arc_solve_registry.yaml",
    "python/carnot/agentic/arc_go_explore.py",
    "python/carnot/agentic/arc_competition_agent.py",
    "openspec/capabilities/research-reporting/spec.md",
    "openspec/capabilities/arc-world-model-trust-energy/spec.md",
)
SETTINGS = {
    "spacing_fresh_actions": 20,
    "replay_budget": 400,
    "preferred_max_prefix": 30,
    "cell_bins": 6,
    "max_cells": 256,
}


def progress(start: float, phase: str, event: str, **detail: Any) -> None:
    print(
        f"[exp7735] {phase} {event} elapsed_s={time.monotonic() - start:.2f} {detail}", flush=True
    )


def fixture_episode(game: str, seed: int, arm: str) -> dict[str, Any]:
    """Drive the scored wrapper over an SDK-shaped fixture; no solve credit."""
    if arm not in {"control", "organic"}:
        raise ValueError("unknown_arm")
    unit = {
        "game": game,
        "seed": seed,
        "arm": arm,
        "episode_id": f"{game}:{seed}:{arm}",
        "max_actions": 3,
        "max_seconds": 30,
    }

    def factory(base: type, **kwargs: Any) -> type:
        return make_carnot_agent(base, organic_visits=arm == "organic", **kwargs)

    episode = run_episode(unit, _FixtureArcade(), agent_factory=factory)
    actions = episode["telemetry"]
    return {
        "game": game,
        "seed": seed,
        "arm": arm,
        "episode_id": unit["episode_id"],
        "actions_charged": len(actions),
        "actions": actions,
        "counter_event_rows": [
            {
                "action": action["action"],
                "provenance": "reset" if action["action"] == "RESET" else "organic",
                "actions_charged": 1,
                "selected_cell": None,
                "prefix_length": 0,
            }
            for action in actions
        ],
        "raw_metrics": episode["raw_metrics"],
        "counts": episode["counts"],
        "censoring": episode["censoring"],
        "exclusions": episode["exclusions"],
        "error": episode["error"],
        "policy_entry": episode["policy_entry"],
        "solve_provenance": "development_proxy",
        "new_solve_credit": False,
        "claim_scope": "fixture_only",
    }


def cold_reduce(rows: list[dict[str, Any]]) -> dict[str, int]:
    """Recompute custody from individual action and event rows."""
    ids = [row["episode_id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate_episode")
    for row in rows:
        if row["actions_charged"] != len(row["actions"]):
            raise ValueError("raw_action_count_mismatch")
        if row["actions_charged"] != sum(
            event["actions_charged"] for event in row["counter_event_rows"]
        ):
            raise ValueError("counter_event_count_mismatch")
        if (
            row["solve_provenance"] not in {"development_proxy", "adapter_withheld_public"}
            or row["new_solve_credit"]
        ):
            raise ValueError("fixture_solve_credit")
    return {
        "rows": len(rows),
        "games": len({row["game"] for row in rows}),
        "actions": sum(row["actions_charged"] for row in rows),
        "resets": sum(
            event["provenance"] == "reset" for row in rows for event in row["counter_event_rows"]
        ),
    }


def _check(name: str, path: Path, field: str, expected: Any, observed: Any) -> dict[str, Any]:
    return {
        "check": name,
        "upstream_id": name,
        "artifact_path": str(path),
        "field": field,
        "operator": "==",
        "expected": expected,
        "observed": observed,
        "passed": expected == observed,
    }


def preconditions() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    checks = []
    hashes: dict[str, Any] = {
        "eligible_producers": {},
        "flagged_historical_inputs": {},
        "pre_gate_receipts": {},
        "absent_sources": [],
    }
    for relative in INPUTS:
        path = ROOT / relative
        exists = path.is_file() and path.stat().st_size > 0
        checks.append(_check("required_input", path, "readable_nonempty", True, exists))
        if exists:
            hashes["eligible_producers"][relative] = sha256_file(path)
        else:
            hashes["absent_sources"].append(relative)
    for relative, marker in (
        ("openspec/capabilities/research-reporting/spec.md", "REQ-REPORT-7735"),
        ("openspec/capabilities/arc-world-model-trust-energy/spec.md", "REQ-ARC-WMTE-7735"),
    ):
        path = ROOT / relative
        checks.append(
            _check("requirement", path, marker, True, path.is_file() and marker in path.read_text())
        )
    checks.append(
        _check(
            "python",
            ROOT / ".venv/bin/python",
            "executable",
            True,
            os.access(ROOT / ".venv/bin/python", os.X_OK),
        )
    )
    return checks, hashes


def public_episode(game: str, seed: int, arm: str, arcade: Any) -> dict[str, Any]:
    """Run visible SDK observations through the scored policy with no adapter."""
    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    metadata = next(
        (item for item in arcade.available_environments if str(item.game_id).split("-")[0] == game),
        None,
    )
    baseline_actions = list(getattr(metadata, "baseline_actions", []) or [])
    unit = {
        "game": game,
        "seed": seed,
        "arm": arm,
        "episode_id": f"{game}:{seed}:{arm}",
        "max_actions": 40,
        "max_seconds": 45,
    }
    provenance: list[dict[str, Any]] = []

    def factory(base: type, **kwargs: Any) -> type:
        parent = make_carnot_agent(base, organic_visits=arm == "organic", **kwargs)

        class ObservedAgent(parent):
            def choose_action(self, frames: Any, latest_frame: Any) -> Any:
                action = super().choose_action(frames, latest_frame)
                recorder = self._policy.action_provenance()
                detail = dict(recorder.rows[-1]) if recorder and recorder.rows else {}
                explorer = self._policy.explorer
                detail["archive_replay_active"] = bool(explorer._go_explore_replay_active)
                archive = explorer.go_explore_archive
                detail["selected_cell"] = (
                    repr(archive.last_selected_cell) if archive and archive.last_selected_cell else None
                )
                detail["prefix_length"] = len(explorer.pending) if explorer._go_explore_replay_active else 0
                provenance.append(detail)
                return action

        return ObservedAgent

    old_record = os.environ.get("CARNOT_ARC_ACTION_PROVENANCE")
    old_directory = os.environ.get("CARNOT_ARC_ACTION_PROVENANCE_DIR")
    os.environ["CARNOT_ARC_ACTION_PROVENANCE"] = "1"
    os.environ["CARNOT_ARC_ACTION_PROVENANCE_DIR"] = "/tmp/exp7735-provenance"
    try:
        episode = run_episode(unit, arcade, agent_factory=factory)
    finally:
        for name, old in (("CARNOT_ARC_ACTION_PROVENANCE", old_record),
                          ("CARNOT_ARC_ACTION_PROVENANCE_DIR", old_directory)):
            if old is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = old
    actions = episode["telemetry"]
    events = []
    for index, action in enumerate(actions):
        detail = provenance[index] if index < len(provenance) else {}
        event = (
            "reset"
            if action["action"] == "RESET"
            else ("replay" if detail.get("archive_replay_active") else "organic")
        )
        events.append(
            {
                "action": action["action"],
                "provenance": event,
                "policy_provenance": detail,
                "actions_charged": 1,
                "selected_cell": detail.get("selected_cell"),
                "prefix_length": detail.get("prefix_length"),
            }
        )
    return {
        "game": game,
        "seed": seed,
        "arm": arm,
        "episode_id": unit["episode_id"],
        "actions_charged": len(actions),
        "actions": actions,
        "counter_event_rows": events,
        "raw_metrics": episode["raw_metrics"],
        "counts": episode["counts"],
        "censoring": episode["censoring"],
        "exclusions": episode["exclusions"],
        "error": episode["error"],
        "policy_entry": episode["policy_entry"],
        "solve_provenance": "adapter_withheld_public",
        "new_solve_credit": False,
        "claim_scope": "adapter_withheld_public",
        "metadata_baseline_actions": baseline_actions,
        "reset_charging": {
            "charged": len(actions),
            "uncharged": len(actions) - sum(a["action"] == "RESET" for a in actions),
        },
    }


def validate(start: float) -> list[dict[str, Any]]:
    """Run frozen affected checks and numbered CPU E2E with owned logs."""
    from carnot.reporting.experiment_7303_validation_scope import (
        CommandSpec,
        build_scoped_commands,
        run_commands,
    )

    base = Path("/tmp/exp7735-validation")
    base.mkdir(parents=True, exist_ok=True)
    (base / "basetemp").mkdir(exist_ok=True)
    commands = build_scoped_commands(
        ROOT,
        SCOPE["tests"],
        SCOPE["changed_modules"],
        static_paths=("scripts/experiments/experiment_7735_v673_arc_organic_visits.py",),
        basetemp=base / "basetemp",
        coverage_file=base / ".coverage",
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
                (pytest, *common, f"--basetemp={base / 'basetemp' / name}", *tests, "-q"),
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
                str(base / "offline_smoke.json"),
            ),
            "numbered_e2e_cpu",
            300,
        )
    )
    progress(start, "validation", "before", commands=len(commands))
    receipts = run_commands(
        ROOT,
        commands,
        log_dir=RAW / "logs",
        extra_env={"JAX_PLATFORMS": "cpu", "CARNOT_ARC_DISABLE_INDUCTION": "1"},
        heartbeat_s=45,
    )
    progress(start, "validation", "after", passed=sum(r["exit_code"] == 0 for r in receipts))
    return receipts


def _registry_levels() -> dict[str, int]:
    import yaml

    value = yaml.safe_load((ROOT / "ops/arc_solve_registry.yaml").read_text())
    return {
        str(row["game"]): int(row.get("levels_reproduced") or 0)
        for row in value["games"]
        if row.get("game") in {"r11l", "sk48"}
    }


def build_artifact(
    rows: list[dict[str, Any]],
    checks: list[dict[str, Any]],
    hashes: dict[str, Any],
    receipts: list[dict[str, Any]],
    spans: list[dict[str, Any]],
    run_date: str,
    *,
    flagged: bool = False,
) -> dict[str, Any]:
    """Reduce only owned raw rows into a bounded terminal claim."""
    summary = cold_reduce(rows)
    failed = [check for check in checks if not check["passed"]]
    required = {
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
    passed = {row["name"] for row in receipts if row["exit_code"] == 0}
    validation_ok = required <= passed
    fixture_ok = any(
        row["claim_scope"] == "fixture_only" and row["actions_charged"] > 0 and row["error"] is None
        for row in rows
    )
    public_ok = all(
        row["error"] is None for row in rows if row["claim_scope"] == "adapter_withheld_public"
    )
    if failed:
        verdict, reason = "blocked", "required_input"
    elif flagged or not validation_ok or not fixture_ok or not public_ok:
        verdict, reason = "disqualified", "required_validation_or_runner"
    else:
        verdict, reason = "circular_positive", "fixture_reachability_only"
    ready = int(verdict == "circular_positive")
    gates = {
        "validity": not failed and fixture_ok and public_ok,
        "readiness": bool(ready),
        "Brier_score": None,
        "decision_cost": None,
        "coverage": validation_ok,
        "retention": None,
        "efficiency": None,
    }
    artifact: dict[str, Any] = {
        "schema": "carnot.exp7735.v673.arc_organic_visits.v1",
        "experiment_id": 7735,
        "milestone": "2026.09.673",
        "run_date": run_date,
        "honest_verdict": f"complete_{verdict}_{reason}",
        "verdict_class": verdict,
        "flagged_adversarial": flagged,
        "gate_check_summary": {"failed_count": len(failed), "failed_checks": failed},
        "acceptance_gate_results": gates,
        "rows": rows,
        "sample_size_budget": {
            "intended": 6,
            "observed": len(rows),
            "eligible": sum(row["error"] is None for row in rows),
            "excluded": sum(bool(row["exclusions"]) for row in rows),
            "censored": sum(bool(row["censoring"]) for row in rows),
            "effective_independent_families": len(
                {row["game"] for row in rows if row["claim_scope"] == "adapter_withheld_public"}
            ),
            "seeds_and_arms_increase_independent_n": False,
        },
        "claim_scope": "fixture_only_and_adapter_withheld_public",
        "fresh_generalization_eligible": False,
        "inference_substrate": "host_cpu_sdk_scored_agent_no_model_load",
        "inference_substrate_class": "no_model_load",
        "planned_inference_substrate_class": "no_model_load",
        "MODEL_SPECS": [],
        "planned_MODEL_SPECS": [],
        "model_specs": [],
        "model_invoked": False,
        "invocation_counts": {
            "loads": 0,
            "forwards": 0,
            "generations": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "failures": 0,
            "cancellations": 0,
        },
        "execution_venue": "host",
        "execution_venue_details": {
            "host": socket.gethostname(),
            "pid": os.getpid(),
            "gpu_uuid": None,
        },
        "phase_spans": spans,
        "random_seed": {"game_schedule": 7735, "episode_seed": 0},
        "source_artifact_hashes": hashes,
        "preconditions_checked": checks,
        "validation_receipts": receipts,
        "frozen_validation_scope": SCOPE,
        "verifier_is_oracle": True,
        "organic_runner_ready_score": ready,
        "counter_event_rows": [
            dict(event, episode_id=row["episode_id"])
            for row in rows
            for event in row["counter_event_rows"]
        ],
        "per_game_results": [
            {
                "game": row["game"],
                "arm": row["arm"],
                "seed": row["seed"],
                "actions": row["actions_charged"],
                "provenance": row["solve_provenance"],
                "new_solve_credit": False,
            }
            for row in rows
        ],
        "solve_provenance": "development_proxy_for_fixtures_no_game_level_solve_claim",
        "registry_precheck": {
            "known_reproduced_levels_source": "ops/arc_solve_registry.yaml",
            "known_reproduced_levels": _registry_levels(),
            "mechanism": "shared_replay_archive_counter_not_target_route",
        },
        "frozen_v8_controls": SETTINGS,
        "reset_charging_interpretations": ["reset_charged", "reset_uncharged"],
        "raw_reduction": summary,
        "raw_rows_sha256": sha256_file(RAW / "rows.json"),
        "administrative_readiness": None,
        "repository_wide_debt": [
            {"command": ".venv/bin/pytest tests/python -q -n 0 -o addopts= --no-cov",
             "exit_code": 2, "collection_errors": 18,
             "log_path": "/tmp/exp7735-fullsuite/pytest.log",
             "log_sha256": sha256_file(Path("/tmp/exp7735-fullsuite/pytest.log"))}
        ] if Path("/tmp/exp7735-fullsuite/pytest.log").is_file() else [],
        "effective_coding_backend": {"requested": "codex", "effective": "codex"},
    }
    artifact["reproducibility_checksum"] = (
        "sha256:"
        + hashlib.sha256(
            json.dumps([hashes, SETTINGS, SCOPE, summary, artifact["raw_rows_sha256"], run_date], sort_keys=True).encode()
        ).hexdigest()
    )
    principle = "Measured evidence bounds the claim and downstream use."
    artifact["field_principles"] = {field: principle for field in artifact}
    for field in gates:
        artifact["field_principles"][f"acceptance_gate_results.{field}"] = principle
    return artifact


def _span(
    name: str,
    start: float,
    end: float,
    run_date: str,
    completed: int = 0,
    checkpoint: str | None = None,
) -> dict[str, Any]:
    return {
        "name": name,
        "start_monotonic": start,
        "end_monotonic": end,
        "duration_s": end - start,
        "run_date": run_date,
        "heartbeat_times": [end],
        "completed_units": completed,
        "checkpoint_hash": checkpoint,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", default="20260927")
    parser.add_argument("--cold", type=Path)
    args = parser.parse_args(argv)
    if args.cold:
        value = json.loads(args.cold.read_text())
        raw_path = RAW / "rows.json"
        if sha256_file(raw_path) != value["raw_rows_sha256"]:
            raise ValueError("raw_rows_hash_mismatch")
        raw_rows = json.loads(raw_path.read_text())["rows"]
        if raw_rows != value["rows"]:
            raise ValueError("raw_rows_candidate_mismatch")
        print(json.dumps(cold_reduce(raw_rows), sort_keys=True), flush=True)
        return 0
    started = time.monotonic()
    progress(started, "preconditions", "before", root=str(ROOT.resolve()))
    RAW.mkdir(parents=True, exist_ok=True)
    spans: list[dict[str, Any]] = []
    phase_start = time.monotonic()
    checks, hashes = preconditions()
    arcade = None
    roster: set[str] = set()
    try:
        from carnot.agentic.arc_solver_kit import offline_arcade

        arcade = offline_arcade()
        roster = {str(item.game_id).split("-")[0] for item in arcade.available_environments}
    except Exception as exc:
        progress(started, "sdk", "unavailable", error=str(exc)[:200])
    for game in ("r11l", "sk48"):
        checks.append(_check("sdk_game", ROOT / "environment_files", game, True, game in roster))
    spans.append(_span("preconditions", phase_start, time.monotonic(), args.date))
    progress(started, "preconditions", "after", failed=sum(not c["passed"] for c in checks))
    rows: list[dict[str, Any]] = []
    phase_start = time.monotonic()
    schedule = [
        {"game": game, "seed": 0, "arm": arm}
        for game in ("r11l", "sk48")
        for arm in ("control", "organic")
    ]
    atomic_json(RAW / "schedule.json", {"units": schedule, "settings": SETTINGS})
    for index, unit in enumerate(schedule):
        if any(not check["passed"] for check in checks):
            break
        game, seed, arm = unit["game"], unit["seed"], unit["arm"]
        progress(started, "episode", "before", completed=index, game=game, arm=arm)
        row = public_episode(game, seed, arm, arcade)
        row["input_hashes"] = hashes["eligible_producers"]
        rows.append(row)
        checkpoint = RAW / f"episode_{index:02d}.json"
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
    progress(started, "fixture", "before")
    for arm in ("control", "organic"):
        row = fixture_episode("fixture", 0, arm)
        row["input_hashes"] = hashes["eligible_producers"]
        rows.append(row)
    atomic_json(RAW / "rows.json", {"rows": rows})
    spans.append(
        _span(
            "episodes",
            phase_start,
            time.monotonic(),
            args.date,
            len(rows),
            sha256_file(RAW / "rows.json"),
        )
    )
    progress(started, "fixture", "after", completed=len(rows))
    phase_start = time.monotonic()
    receipts = validate(started)
    spans.append(
        _span(
            "validation",
            phase_start,
            time.monotonic(),
            args.date,
            sum(r["exit_code"] == 0 for r in receipts),
        )
    )
    candidate = RAW / "candidate.json"
    artifact = build_artifact(rows, checks, hashes, receipts, spans, args.date)
    atomic_json(candidate, artifact)
    from carnot.reporting.experiment_7303_validation_scope import CommandSpec, run_commands

    python = str(ROOT / ".venv/bin/python")
    terminal = [
        CommandSpec(
            "cold_reduce",
            (
                python,
                "-u",
                "scripts/experiments/experiment_7735_v673_arc_organic_visits.py",
                "--cold",
                str(candidate),
            ),
            "terminal",
            120,
        ),
        CommandSpec(
            "adversarial_verify",
            (python, "-u", "scripts/adversarial_verify.py", str(candidate)),
            "terminal",
            120,
        ),
        CommandSpec(
            "strict_row_lint",
            (python, "-u", "scripts/verdict_row_consistency_lint.py", "--strict", str(candidate)),
            "terminal",
            120,
        ),
    ]
    progress(started, "terminal_readers", "before", candidate_sha256=sha256_file(candidate))
    terminal_receipts = run_commands(ROOT, terminal, log_dir=RAW / "terminal_logs", heartbeat_s=45)
    atomic_json(
        RAW / "terminal_receipts.json",
        {"candidate_sha256": sha256_file(candidate), "commands": terminal_receipts},
    )
    progress(
        started,
        "terminal_readers",
        "after",
        passed=sum(r["exit_code"] == 0 for r in terminal_receipts),
    )
    if any(r["exit_code"] != 0 for r in terminal_receipts):
        artifact["flagged_adversarial"] = any(
            r["name"] == "adversarial_verify" and r["exit_code"] != 0 for r in terminal_receipts
        )
        artifact["verdict_class"] = "disqualified"
        artifact["honest_verdict"] = "complete_disqualified_terminal_reader"
        artifact["organic_runner_ready_score"] = 0
        artifact["acceptance_gate_results"]["readiness"] = False
    atomic_json(RESULT, artifact)
    progress(
        started,
        "publish",
        "after",
        path=str(RESULT),
        sha256=sha256_file(RESULT),
        verdict=artifact["honest_verdict"],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
