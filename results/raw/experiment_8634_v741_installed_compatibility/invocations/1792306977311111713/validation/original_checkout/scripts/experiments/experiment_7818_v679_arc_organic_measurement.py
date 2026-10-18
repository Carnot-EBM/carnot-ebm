"""Run the public, adapter-withheld scored ARC panel for REQ-REPORT-7818."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import signal
import socket
import subprocess
import sys
import time
import uuid
from typing import Any

import numpy as np
import yaml

from carnot.agentic.arc_generalization_runtime import run_episode
from carnot.agentic.arc_solver_kit import offline_arcade
from carnot.experiment_7817_v679_arc_runner_qualification import make_agent_factory
from carnot.experiment_7818_v679_arc_organic_measurement import (
    command_plan,
    reduce_rows,
    score_episode,
    seal_log,
    sha256,
    verify_log,
)
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "results/raw/experiment_7818_v679_arc_organic_measurement"
MANIFEST = RAW / "validation_command_manifest.json"
RESULT = ROOT / "results/experiment_7818_v679_arc_organic_measurement.json"
PRIOR = ROOT / "results/experiment_7817_v679_arc_runner_qualification.json"
PANEL = ROOT / "results/raw/experiment_7790_v677_arc_runner_qualification/arc_panel_manifest.json"
INPUTS = (
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
    "python/carnot/agentic/arc_competition_agent.py",
    "python/carnot/agentic/arc_graph_explore.py",
    "python/carnot/agentic/arc_solver_kit.py",
    "tests/python/test_arc_ige_cell_selector.py",
    "openspec/capabilities/research-reporting/spec.md",
    "openspec/capabilities/arc-world-model-trust-energy/spec.md",
    "openspec/change-proposals/research-roadmap-v678-preserved-20260928.md",
)


def progress(start: float, phase: str, event: str, units: int = 0, **detail: Any) -> None:
    """Flush real elapsed work at every phase and long child boundary."""
    print(
        f"[exp7818] phase={phase} event={event} elapsed_s={time.monotonic() - start:.2f} completed_units={units} {detail}",
        flush=True,
    )


def failed_operand(
    path: Path, field: str, expected: Any, observed: Any, upstream: str
) -> dict[str, Any]:
    """Name the exact external byte source and field behind a blocked gate."""
    return {
        "upstream_id": upstream,
        "artifact_path": str(path),
        "artifact_sha256": sha256(path) if path.is_file() else None,
        "field": field,
        "operator": "==",
        "expected": expected,
        "observed": observed,
        "passed": expected == observed,
    }


def preflight(start: float) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    """Check named bytes, readiness and SDK resources before an episode starts."""
    progress(start, "preconditions", "before")
    checks = [
        failed_operand(
            ROOT / name,
            "readable_nonempty",
            True,
            (ROOT / name).is_file() and (ROOT / name).stat().st_size > 0,
            name,
        )
        for name in INPUTS
    ]
    for path in (PRIOR, PANEL, MANIFEST):
        checks.append(
            failed_operand(
                path,
                "readable_nonempty",
                True,
                path.is_file() and path.stat().st_size > 0,
                path.stem,
            )
        )
    prior = json.loads(PRIOR.read_text()) if PRIOR.is_file() else {}
    checks.append(
        failed_operand(
            PRIOR,
            "organic_runner_ready_score",
            1,
            prior.get("organic_runner_ready_score"),
            "exp7817_science_producer",
        )
    )
    checks.append(
        failed_operand(
            PRIOR,
            "verdict_class",
            "circular_positive",
            prior.get("verdict_class"),
            "exp7817_science_producer",
        )
    )
    checks.append(
        failed_operand(
            ROOT / ".venv/bin/python",
            "executable",
            True,
            os.access(ROOT / ".venv/bin/python", os.X_OK),
            "python",
        )
    )
    checks.append(
        failed_operand(
            ROOT,
            "disk_free_at_least_1GiB",
            True,
            shutil.disk_usage(ROOT).free >= 1024**3,
            "resources",
        )
    )
    checks.append(
        failed_operand(ROOT, "cpu_backend", "cpu", os.environ.get("JAX_PLATFORMS"), "backend")
    )
    panel = json.loads(PANEL.read_text()) if PANEL.is_file() else {}
    manifest = json.loads(MANIFEST.read_text()) if MANIFEST.is_file() else {}
    checks.append(
        failed_operand(
            PANEL,
            "games",
            ["cd82", "dc22", "lf52", "m0r0", "sk48", "tn36", "sb26", "sc25"],
            panel.get("games"),
            "exp7790_frozen_panel",
        )
    )
    checks.append(
        failed_operand(
            MANIFEST, "command_count", 66, len(manifest.get("commands", [])), "exp7818_manifest"
        )
    )
    arcade = offline_arcade()
    info = {str(item.game_id).split("-")[0]: item for item in arcade.available_environments}
    registry_data = yaml.safe_load((ROOT / "ops/arc_solve_registry.yaml").read_text())
    registry_rows = (
        registry_data if isinstance(registry_data, list) else registry_data.get("games", [])
    )
    registry = {row.get("game") for row in registry_rows if isinstance(row, dict)}
    for game in panel.get("games", []):
        checks.append(
            failed_operand(
                ROOT / "environment_files", f"sdk_game:{game}", True, game in info, "arc_sdk"
            )
        )
        checks.append(
            failed_operand(
                ROOT / "ops/arc_solve_registry.yaml",
                f"registry_game:{game}",
                True,
                game in registry,
                "arc_registry",
            )
        )
        checks.append(
            failed_operand(
                ROOT / "environment_files",
                f"human_baseline:{game}",
                True,
                bool(getattr(info.get(game), "baseline_actions", None)),
                "arc_sdk",
            )
        )
    sources = {
        name: {
            "path": name,
            "sha256": sha256(ROOT / name),
            "date": "20260928",
            "imported_fields": [],
            "eligible": True,
        }
        for name in INPUTS
        if (ROOT / name).is_file()
    }
    for path, role in (
        (PRIOR, "science_producer"),
        (PANEL, "historical_panel"),
        (
            ROOT / "results/experiment_7804_arc_organic_measurement.json",
            "conductor_pre_gate_receipt",
        ),
    ):
        label = str(path.relative_to(ROOT))
        sources[label] = {
            "path": label,
            "sha256": sha256(path) if path.is_file() else None,
            "date": "20260928",
            "imported_fields": ["organic_runner_ready_score"] if path == PRIOR else [],
            "eligible": path == PRIOR,
            "role": role,
        }
    resources = {
        "host": socket.gethostname(),
        "disk_free_bytes": shutil.disk_usage(ROOT).free,
        "cpu_count": os.cpu_count(),
        "backend": "offline_arcade_cpu",
    }
    progress(
        start, "preconditions", "after", len(checks), failed=sum(not c["passed"] for c in checks)
    )
    return checks, sources, resources


def episode(game: str, seed: int, arm: str, start: float) -> dict[str, Any]:
    """Run one real SDK episode with zero model tiers and local archive choices."""
    random.seed(seed)
    np.random.seed(seed)
    arcade = offline_arcade()
    info = next(
        item for item in arcade.available_environments if str(item.game_id).startswith(game + "-")
    )
    baseline = list(info.baseline_actions or [])
    events: list[dict[str, Any]] = []
    unit = {
        "episode_id": f"{game}:{seed}:{arm}",
        "game": game,
        "seed": seed,
        "arm": arm,
        "max_actions": 2000,
        "max_seconds": 75,
    }
    progress(start, "model_load", "before")
    progress(start, "model_load", "after")
    progress(start, "generation", "before")
    progress(start, "generation", "after")
    progress(start, "sdk_episode", "before", game=game, seed=seed, arm=arm)
    value = run_episode(unit, arcade, agent_factory=make_agent_factory(arm, events))
    metrics = score_episode(value["telemetry"], baseline)
    counter = {
        provenance: sum(e.get("provenance") == provenance for e in events)
        for provenance in ("organic", "replay", "reset")
    }
    row = {
        **unit,
        **metrics,
        "status": "completed" if value["error"] is None else "error",
        "censoring": value["censoring"],
        "error": value["error"],
        "policy_entry": value["policy_entry"],
        "raw_metrics": value["raw_metrics"],
        "counts": value["counts"],
        "actions": value["telemetry"],
        "counter_event_rows": events,
        "observation_counts": counter,
        "organic_visits": sum(
            e.get("organic_seen", 0) for e in events if e.get("event") == "observation"
        ),
        "supervisor_outcomes": [
            a.get("goal_firing") for a in value["telemetry"] if a.get("goal_firing") is not None
        ],
        "solve_provenance": "live_agent_self_discovery",
        "new_solve_credit": False,
        "human_baseline_source": "EnvironmentInfo",
        "model_invocation_counts": {"loads": 0, "generations": 0},
    }
    progress(start, "sdk_episode", "after", len(value["telemetry"]), error=value["error"])
    return row


def run_child(
    spec: dict[str, Any],
    plan: list[dict[str, Any]],
    private: Path,
    durable: Path,
    attempt: int,
    start: float,
    units: int,
) -> dict[str, Any]:
    """Supervise an owned process and seal its output only after handle close."""
    expected = {key: spec[key] for key in ("name", "argv", "classification")}
    if expected not in plan:
        raise ValueError(f"undeclared_child:{spec['name']}")
    name = spec["name"]
    child_root = private / f"{attempt:04d}_{name}"
    child_root.mkdir(parents=True, exist_ok=False)
    for arg in spec["argv"]:
        if arg.startswith("--basetemp=") or arg.startswith("--data-file="):
            Path(arg.split("=", 1)[1]).parent.mkdir(parents=True, exist_ok=True)
    if name == "e2e_009_smoke":
        Path(spec["argv"][-1]).parent.mkdir(parents=True, exist_ok=True)
    durable.mkdir(parents=True, exist_ok=True)
    live = child_root / "live.log"
    env = {
        **os.environ,
        "PYTHONPATH": "python:.",
        "PYTHONUNBUFFERED": "1",
        "JAX_PLATFORMS": "cpu",
        "CARNOT_ARC_DISABLE_INDUCTION": "1",
    }
    progress(start, "subprocess", "before", units, name=name, classification=spec["classification"])
    began = time.monotonic()
    timed_out = False
    with live.open("wb") as stream:
        child = subprocess.Popen(
            spec["argv"],
            cwd=ROOT,
            env=env,
            stdout=stream,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        while child.poll() is None:
            if time.monotonic() - began > float(spec["timeout_s"]):
                timed_out = True
                os.killpg(child.pid, signal.SIGTERM)
                try:
                    child.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL)
                    child.wait()
                break
            try:
                child.wait(timeout=30)
            except subprocess.TimeoutExpired:
                progress(
                    start,
                    "subprocess",
                    "heartbeat",
                    units,
                    name=name,
                    child_elapsed_s=round(time.monotonic() - began, 2),
                )
    data = live.read_bytes()
    sealed = seal_log(durable, name, attempt, data)
    receipt = {
        "name": name,
        "classification": spec["classification"],
        "command_argv": spec["argv"],
        "exit_code": child.returncode,
        "timed_out": timed_out,
        "passed": child.returncode == 0 and not timed_out,
        "duration_s": time.monotonic() - began,
        "log_path": str(sealed.relative_to(ROOT)),
        "log_sha256": sha256(sealed),
    }
    progress(
        start,
        "subprocess",
        "after",
        units + 1,
        name=name,
        exit_code=child.returncode,
        timed_out=timed_out,
    )
    return receipt


def raw_reduction(path: Path) -> dict[str, Any]:
    """Fresh process recomputes game pairs from immutable raw episode rows."""
    value = json.loads(path.read_text())
    if len(value) != 48:
        raise ValueError("raw_row_count")
    keys = [(row["game"], row["seed"], row["arm"]) for row in value]
    if len(set(keys)) != 48:
        raise ValueError("duplicate_episode")
    return reduce_rows(value)


def cold_replay(path: Path) -> None:
    """Check raw bytes, manifest bytes and every already sealed child log."""
    value = json.loads(path.read_text())
    if sha256(MANIFEST) != value["validation_command_manifest_sha256"]:
        raise ValueError("manifest_mutated")
    raw = ROOT / value["raw_rows_path"]
    if not verify_log(raw, value["raw_rows_sha256"]):
        raise ValueError("raw_rows_mutated")
    if raw_reduction(raw) != value["raw_reduction"]:
        raise ValueError("raw_reduction_changed")
    for receipt in value["validation_receipts"]:
        if not verify_log(ROOT / receipt["log_path"], receipt["log_sha256"]):
            raise ValueError(f"sealed_log_mutated:{receipt['name']}")


def build_artifact(
    start: float,
    checks: list[dict[str, Any]],
    sources: dict[str, Any],
    resources: dict[str, Any],
    rows: list[dict[str, Any]],
    receipts: list[dict[str, Any]],
    reduction: dict[str, Any],
    date: str,
    spans: list[dict[str, Any]],
) -> dict[str, Any]:
    """Keep a complete terminal record even when evidence is blocked or null."""
    failed = [row for row in checks if not row["passed"]]
    required = [
        row
        for row in json.loads(MANIFEST.read_text())["commands"]
        if row["classification"] == "required" and not row["name"].startswith("episode_")
    ]
    failed_required = [
        row["name"]
        for row in required
        if len(matches := [r for r in receipts if r["name"] == row["name"]]) != 1
        or not matches[0]["passed"]
    ]
    started = sum(row["status"] != "unstarted" for row in rows)
    completed = sum(row["status"] == "completed" for row in rows)
    censored = sum(bool(row.get("censoring")) or row["status"] == "timed_out" for row in rows)
    if failed:
        verdict = "complete_blocked_external_preconditions"
        verdict_class = "blocked"
    elif failed_required:
        verdict = "complete_disqualified_required_validation"
        verdict_class = "disqualified"
    elif started < 48:
        verdict = (
            "complete_null_budget_censored_public_panel"
            if reduction["independent_n"]
            else "complete_blocked_zero_valid_pairs"
        )
        verdict_class = "null" if reduction["independent_n"] else "blocked"
    elif reduction["independent_n"] == 0:
        verdict = "complete_blocked_zero_valid_pairs"
        verdict_class = "blocked"
    elif reduction["organic_benefit_score"]:
        verdict = "complete_positive_public_organic_benefit"
        verdict_class = "positive"
    else:
        verdict = "complete_null_public_organic_benefit"
        verdict_class = "null"
    ready = int(not failed and not failed_required and reduction["independent_n"] == 8)
    benefit = int(ready and reduction["organic_benefit_score"])
    gates = {
        "validity": bool(ready),
        "readiness": bool(ready),
        "probability_quality": None,
        "decision_benefit": bool(benefit) if ready else None,
        "retention": not reduction["lost_baseline_winning_seed"] if ready else None,
        "efficiency": reduction["shared_win_action_regression"] if ready else None,
    }
    budget = {
        "intended": 48,
        "eligible": 48 - len(failed),
        "started": started,
        "completed": completed,
        "excluded": sum(row["status"] == "error" for row in rows),
        "censored": censored,
        "independent_n": reduction["independent_n"],
        "seeds_or_actions_multiply_n": False,
    }
    value: dict[str, Any] = {
        "schema": "carnot.exp7818.v679.arc_organic_measurement.v1",
        "experiment_id": 7818,
        "milestone": "2026.09.679",
        "run_date": date,
        "honest_verdict": verdict,
        "verdict_class": verdict_class,
        "flagged_adversarial": bool(failed or failed_required),
        "gate_check_summary": [
            *failed,
            *[
                failed_operand(MANIFEST, name, True, False, "required_validation")
                for name in failed_required
            ],
        ],
        "rows": rows,
        "per_game_results": reduction,
        "acceptance_gate_results": gates,
        "arc_measurement_ready_score": ready,
        "organic_benefit_score": benefit,
        "duration_s": time.monotonic() - start,
        "phase_spans": spans,
        "random_seed": {"game_order": 67815, "episode_seeds": [67815, 67816]},
        "sample_size_budget": budget,
        "source_artifact_hashes": sources,
        "preconditions_checked": {"checks": checks, "resources": resources},
        "validation_receipts": receipts,
        "verifier_is_oracle": False,
        "claim_scope": [
            "adapter_withheld_exposed_public_generalization_proxy",
            "all_640_source_families_exposed_development_data",
            "no_hidden_leaderboard_claim",
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
        "solve_provenance": "live_agent_self_discovery",
        "score_formula_version": "installed_arc_agi_EnvironmentScoreCalculator_weighted_level_v1",
        "reset_charge_sensitivity": ["charged", "uncharged"],
        "validation_command_manifest_path": str(MANIFEST.relative_to(ROOT)),
        "validation_command_manifest_sha256": sha256(MANIFEST),
        "observed_child_commands": [
            {"name": r["name"], "argv": r["command_argv"], "classification": r["classification"]}
            for r in receipts
        ],
        "repository_health": next(
            (r for r in receipts if r["name"] == "repository_health_full_python_suite"), None
        ),
        "raw_rows_path": str((RAW / "raw_rows.json").relative_to(ROOT)),
        "raw_rows_sha256": sha256(RAW / "raw_rows.json"),
        "raw_reduction": reduction,
        "execution_venue": "host",
        "execution_venue_details": {"host": socket.gethostname(), "pid": os.getpid()},
    }
    value["reproducibility_checksum"] = canonical_hash(
        {
            "sources": sources,
            "manifest": value["validation_command_manifest_sha256"],
            "raw": value["raw_rows_sha256"],
            "seed": value["random_seed"],
            "roles": ["off", "total", "organic"],
        }
    )
    value["field_principles"] = {key: "Exact current evidence bounds this field." for key in value}
    return value


def main(argv: list[str] | None = None) -> int:
    """Dispatch the frozen panel, validate it, and atomically publish one result."""
    start = time.monotonic()
    progress(start, "startup", "before", pid=os.getpid())
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", default="20260928")
    parser.add_argument("--episode", nargs=3, metavar=("GAME", "SEED", "ARM"))
    parser.add_argument("--record-dispatch", type=Path)
    parser.add_argument("--cold-reduce", type=Path)
    parser.add_argument("--cold-replay", type=Path)
    args = parser.parse_args(argv)
    if args.date != "20260928":
        raise ValueError("run_date_mismatch")
    if args.record_dispatch:
        atomic_json(args.record_dispatch, command_plan(MANIFEST))
        progress(start, "dispatch_record", "after", 66)
        return 0
    if args.cold_reduce:
        progress(start, "cold_reduce", "before")
        value = raw_reduction(args.cold_reduce)
        print("REDUCTION_JSON=" + json.dumps(value, sort_keys=True), flush=True)
        progress(start, "cold_reduce", "after", value["independent_n"])
        return 0
    if args.cold_replay:
        progress(start, "cold_replay", "before")
        cold_replay(args.cold_replay)
        progress(start, "cold_replay", "after", 1)
        return 0
    if args.episode:
        game, seed, arm = args.episode
        row = episode(game, int(seed), arm, start)
        print("EPISODE_JSON=" + json.dumps(row, separators=(",", ":"), sort_keys=True), flush=True)
        return int(row["status"] == "error")
    RAW.mkdir(parents=True, exist_ok=True)
    plan = command_plan(MANIFEST)
    specs = json.loads(MANIFEST.read_text())["commands"]
    run_id = uuid.uuid4().hex
    private = Path("/tmp") / f"exp7818-v679-{run_id}"
    private.mkdir(parents=True)
    durable = RAW / "validation_logs" / run_id
    spans: list[dict[str, Any]] = []
    phase = time.monotonic()
    checks, sources, resources = preflight(start)
    spans.append(
        {
            "phase": "preconditions",
            "duration_s": time.monotonic() - phase,
            "completed_units": len(checks),
        }
    )
    rows: list[dict[str, Any]] = []
    for spec in specs[:48]:
        _, game, seed, arm = spec["name"].split("_")
        rows.append(
            {
                "episode_id": f"{game}:{seed}:{arm}",
                "game": game,
                "seed": int(seed),
                "arm": arm,
                "status": "unstarted",
                "censoring": None,
                "score_charged": None,
                "score_uncharged": None,
                "solve_provenance": "live_agent_self_discovery",
                "new_solve_credit": False,
            }
        )
    receipts: list[dict[str, Any]] = []
    if all(row["passed"] for row in checks):
        phase = time.monotonic()
        for index, spec in enumerate(specs[:48]):
            if time.monotonic() - start >= 3900:
                progress(start, "launch_budget", "stop", index)
                break
            receipt = run_child(spec, plan, private, durable, index + 1, start, index)
            receipts.append(receipt)
            row = rows[index]
            for line in reversed((ROOT / receipt["log_path"]).read_text().splitlines()):
                if line.startswith("EPISODE_JSON="):
                    row.update(json.loads(line.removeprefix("EPISODE_JSON=")))
                    break
            if row["status"] == "unstarted":
                row["status"] = "timed_out" if receipt["timed_out"] else "error"
                row["censoring"] = "child_timeout" if receipt["timed_out"] else "child_exit"
                row["error"] = f"exit_code:{receipt['exit_code']}"
            row["raw_path"] = receipt["log_path"]
            row["raw_sha256"] = receipt["log_sha256"]
        spans.append(
            {
                "phase": "sdk_episodes",
                "duration_s": time.monotonic() - phase,
                "completed_units": len(receipts),
            }
        )
    atomic_json(RAW / "raw_rows.json", rows)
    progress(start, "raw_rows", "after", len(rows), sha256=sha256(RAW / "raw_rows.json"))
    reduction = raw_reduction(RAW / "raw_rows.json")
    progress(start, "reduction", "after", reduction["independent_n"])
    candidate = RAW / "terminal_candidate.json"
    if all(row["passed"] for row in checks):
        phase = time.monotonic()
        for index, spec in enumerate(specs[48:-3], 49):
            receipt = run_child(spec, plan, private, durable, index, start, len(receipts))
            receipts.append(receipt)
        spans.append(
            {
                "phase": "validation",
                "duration_s": time.monotonic() - phase,
                "completed_units": len(receipts) - 48,
            }
        )
    value = build_artifact(
        start, checks, sources, resources, rows, receipts, reduction, args.date, spans
    )
    atomic_json(candidate, value)
    if all(row["passed"] for row in checks):
        for index, spec in enumerate(specs[-3:], len(specs) - 2):
            if spec["name"] == "cold_replay":
                value = build_artifact(
                    start, checks, sources, resources, rows, receipts, reduction, args.date, spans
                )
                atomic_json(candidate, value)
            receipt = run_child(spec, plan, private, durable, index, start, len(receipts))
            receipts.append(receipt)
    value = build_artifact(
        start, checks, sources, resources, rows, receipts, reduction, args.date, spans
    )
    atomic_json(RESULT, value)
    progress(
        start,
        "terminal",
        "after",
        len(receipts),
        verdict=value["honest_verdict"],
        independent_n=reduction["independent_n"],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
