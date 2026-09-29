"""Publish direct scored SDK qualification for REQ-REPORT-7817."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import signal
import socket
import subprocess
import sys
import time
import uuid
from typing import Any

import yaml

from carnot.agentic.arc_solver_kit import offline_arcade
from carnot.experiment_7776_v676_arc_runner_qualification import reduce_probe_evidence
from carnot.experiment_7817_v679_arc_runner_qualification import (
    command_plan,
    freeze_panel,
    positive_selector_fixture,
    run_probe,
    sdk_probe_ok,
    seal_log,
    sha256,
    validate_receipts,
    verify_log,
)
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash


SOURCE_ROOT = Path(__file__).resolve().parents[2]
ROOT = SOURCE_ROOT
RAW = ROOT / "results/raw/experiment_7817_v679_arc_runner_qualification"
RESULT = ROOT / "results/experiment_7817_v679_arc_runner_qualification.json"
MANIFEST = RAW / "validation_command_manifest.json"
OLD_PANEL = (
    ROOT / "results/raw/experiment_7790_v677_arc_runner_qualification/arc_panel_manifest.json"
)
OLD = ROOT / "results/experiment_7803_v678_arc_runner_qualification.json"
INPUTS = (
    "AGENTS.md",
    "CLAUDE.md",
    "CODEX.md",
    "research-program.md",
    "ops/exclusion_manifest.yaml",
    "ops/e2e-test-plan.md",
    "ops/arc_solve_registry.yaml",
    "scripts/experiment_template.py",
    "python/carnot/reporting/current_work_receipt.py",
    "python/carnot/reporting/experiment_7303_validation_scope.py",
    "openspec/capabilities/research-reporting/spec.md",
    "openspec/capabilities/arc-world-model-trust-energy/spec.md",
    "openspec/change-proposals/research-roadmap-v678-preserved-20260928.md",
    "python/carnot/agentic/arc_competition_agent.py",
    "python/carnot/agentic/arc_graph_explore.py",
    "python/carnot/agentic/arc_solver_kit.py",
    "tests/python/test_arc_ige_cell_selector.py",
    "python/carnot/experiment_7790_v677_arc_runner_qualification.py",
    "python/carnot/experiment_7748_v674_arc_runner_qualification.py",
    "python/carnot/experiment_7803_v678_arc_runner_qualification.py",
    "results/experiment_7790_v677_arc_runner_qualification.json",
    "results/experiment_7803_v678_arc_runner_qualification.json",
    "results/raw/experiment_7790_v677_arc_runner_qualification/arc_panel_manifest.json",
    "results/raw/experiment_7817_v679_arc_runner_qualification/validation_command_manifest.json",
)


def progress(start: float, phase: str, event: str, units: int = 0, **details: Any) -> None:
    """Keep every phase and long child visible with elapsed work and completed units."""
    print(
        f"[exp7817] phase={phase} event={event} elapsed_s={time.monotonic() - start:.2f} completed_units={units} {details}",
        flush=True,
    )


def operand(path: Path, field: str, expected: Any, observed: Any, upstream: str) -> dict[str, Any]:
    """Keep an exact failed input rather than a generic blocked label."""
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
    """Check cheap named inputs and exact gates before any scored transition."""
    progress(start, "preconditions", "before")
    checks = [
        operand(
            ROOT / name,
            "readable_nonempty",
            True,
            (ROOT / name).is_file() and (ROOT / name).stat().st_size > 0,
            name,
        )
        for name in INPUTS
    ]
    checks.append(
        operand(
            ROOT / ".venv/bin/python",
            "executable",
            True,
            os.access(ROOT / ".venv/bin/python", os.X_OK),
            "python",
        )
    )
    prior = json.loads(OLD.read_text()) if OLD.is_file() else {}
    checks.append(
        operand(
            OLD,
            "honest_verdict",
            "complete_disqualified_required_runner_validation",
            prior.get("honest_verdict"),
            "exp7803_historical",
        )
    )
    checks.append(
        operand(
            OLD,
            "organic_runner_ready_score",
            0,
            prior.get("organic_runner_ready_score"),
            "exp7803_historical",
        )
    )
    panel = freeze_panel(OLD_PANEL) if OLD_PANEL.is_file() else None
    checks.append(operand(OLD_PANEL, "frozen_panel", True, panel is not None, "exp7790_panel"))
    arcade = offline_arcade()
    roster = {str(item.game_id).split("-")[0] for item in arcade.available_environments}
    data = yaml.safe_load((ROOT / "ops/arc_solve_registry.yaml").read_text())
    registry_rows = data if isinstance(data, list) else data.get("games", [])
    registry = {row.get("game"): row for row in registry_rows if isinstance(row, dict)}
    for game in (*panel["games"], "r11l") if panel else ("r11l",):
        checks.append(
            operand(
                ROOT / "ops/arc_solve_registry.yaml",
                f"registry_game:{game}",
                True,
                game in registry,
                "arc_registry",
            )
        )
        checks.append(
            operand(ROOT / "environment_files", f"sdk_game:{game}", True, game in roster, "arc_sdk")
        )
    checks.append(
        operand(
            ROOT,
            "free_bytes_at_least_1GiB",
            True,
            shutil.disk_usage(ROOT).free >= 1024**3,
            "resources",
        )
    )
    checks.append(
        operand(
            ROOT / ".venv/bin/python",
            "cpu_backend",
            "cpu",
            os.environ.get("JAX_PLATFORMS"),
            "backend",
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
    producer = ROOT / "results/experiment_7749_v674_arc_generalization.json"
    pregate = ROOT / "results/experiment_7749_arc_organic_measurement.json"
    for path, role in (
        (producer, "downstream_science_producer"),
        (pregate, "conductor_pre_gate_receipt"),
    ):
        sources[str(path.relative_to(ROOT))] = {
            "path": str(path.relative_to(ROOT)),
            "sha256": sha256(path) if path.is_file() else None,
            "date": None,
            "imported_fields": [],
            "eligible": False,
            "role": role,
        }
    resources = {
        "disk_free_bytes": shutil.disk_usage(ROOT).free,
        "cpu_count": os.cpu_count(),
        "host": socket.gethostname(),
        "sdk_version": importlib.metadata.version("arc-agi"),
        "backend": "offline_arcade_cpu",
        "science_producer_present": producer.is_file(),
        "conductor_pre_gate_present": pregate.is_file(),
    }
    progress(
        start, "preconditions", "after", len(checks), failed=sum(not x["passed"] for x in checks)
    )
    return checks, sources, resources


def run_child(
    spec: dict[str, Any],
    plan: list[dict[str, Any]],
    private: Path,
    durable: Path,
    attempt: int,
    start: float,
    completed: int,
) -> dict[str, Any]:
    """Supervise only this child, then seal exact output after process and handle close."""
    declared = next((row for row in plan if row["name"] == spec["name"]), None)
    if declared != {key: spec[key] for key in ("name", "argv", "classification")}:
        raise ValueError(f"undeclared_child:{spec['name']}")
    name = spec["name"]
    child_root = private / f"{attempt:04d}_{name}"
    child_root.mkdir(parents=True, exist_ok=False)
    for arg in spec["argv"]:
        if arg.startswith("--basetemp="):
            Path(arg.split("=", 1)[1]).parent.mkdir(parents=True, exist_ok=True)
        if arg.startswith("--data-file="):
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
    progress(
        start, "subprocess", "before", completed, name=name, classification=spec["classification"]
    )
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
            if time.monotonic() - began >= spec["timeout_s"]:
                timed_out = True
                os.killpg(child.pid, signal.SIGTERM)
                try:
                    child.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL)
                    child.wait()
                break
            progress(
                start,
                "subprocess",
                "outstanding",
                completed,
                name=name,
                child_elapsed_s=round(time.monotonic() - began, 2),
            )
            try:
                remaining = max(0.01, spec["timeout_s"] - (time.monotonic() - began))
                child.wait(timeout=min(30, remaining))
            except subprocess.TimeoutExpired:
                pass
        exit_code = child.wait()
    data = live.read_bytes()
    sealed = seal_log(durable, name, attempt, data)
    digest = sha256(sealed)
    receipt = {
        "name": name,
        "command_argv": spec["argv"],
        "classification": spec["classification"],
        "exit_code": exit_code,
        "timed_out": timed_out,
        "passed": exit_code == 0 and not timed_out,
        "duration_s": time.monotonic() - began,
        "log_path": str(sealed.relative_to(ROOT)),
        "log_sha256": digest,
        "output_tail": data[-3000:].decode(errors="replace"),
        "private_root": str(child_root),
        "owned_pid": child.pid,
    }
    progress(
        start,
        "subprocess",
        "after",
        completed + 1,
        name=name,
        exit_code=exit_code,
        timed_out=timed_out,
        log_sha256=digest,
    )
    return receipt


def cold_reduce(path: Path) -> dict[str, int]:
    """Recount immutable raw probes in a fresh process."""
    raw = json.loads(path.read_text())
    summary = reduce_probe_evidence(raw["schedule"], raw["probes"])
    if summary != raw["summary"]:
        raise ValueError("raw_reduction_mismatch")
    return summary


def cold_replay(path: Path) -> None:
    """Verify sealed logs and raw bytes after all earlier children have exited."""
    value = json.loads(path.read_text())
    raw = ROOT / value["raw_probes_path"]
    if not verify_log(raw, value["raw_probes_sha256"]):
        raise ValueError("raw_probes_mutated")
    for receipt in value["validation_receipts"]:
        if not verify_log(ROOT / receipt["log_path"], receipt["log_sha256"]):
            raise ValueError(f"sealed_log_mutated:{receipt['name']}")
    if cold_reduce(raw) != value["raw_reduction"]:
        raise ValueError("cold_reduction_changed")


def extract_probe(receipt: dict[str, Any]) -> dict[str, Any]:
    """Read the raw row from the already sealed child log."""
    for line in reversed((ROOT / receipt["log_path"]).read_text().splitlines()):
        if line.startswith("PROBE_JSON="):
            return json.loads(line.removeprefix("PROBE_JSON="))
    raise ValueError(f"probe_row_missing:{receipt['name']}")


def build_artifact(
    start: float,
    spans: list[dict[str, Any]],
    checks: list[dict[str, Any]],
    sources: dict[str, Any],
    resources: dict[str, Any],
    panel_path: Path,
    probes: list[dict[str, Any]],
    fixture: dict[str, Any],
    summary: dict[str, int],
    receipts: list[dict[str, Any]],
    date: str,
) -> dict[str, Any]:
    """Keep transport evidence separate from science measurements and prior failures."""
    panel = json.loads(panel_path.read_text())
    raw_path = RAW / "raw_probes.json"
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
        "schema": "carnot.exp7817.v679.arc_runner_qualification.v1",
        "experiment_id": 7817,
        "milestone": "2026.09.679",
        "run_date": date,
        "honest_verdict": "complete_null_runner_qualification_pending_terminal",
        "verdict_class": "null",
        "flagged_adversarial": False,
        "gate_check_summary": [row for row in checks if not row["passed"]],
        "rows": panel["rows"],
        "probe_rows": probes,
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
                "sources": sources,
                "manifest_sha256": sha256(MANIFEST),
                "panel_sha256": sha256(panel_path),
                "raw_sha256": sha256(raw_path),
                "roles": panel["arms"],
                "controls": panel["controls"],
                "seeds": panel["seeds"],
            }
        ),
        "sample_size_budget": budget,
        "source_artifact_hashes": sources,
        "preconditions_checked": {"checks": checks, "resources": resources},
        "validation_receipts": receipts,
        "verifier_is_oracle": True,
        "claim_scope": [
            "circular_positive_fixture_mechanics",
            "adapter_withheld_public_transport_only",
            "all_640_source_families_exposed_development_data",
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
        "arc_panel_manifest_sha256": sha256(panel_path),
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
        "solve_provenance": {
            "fixture": "development_proxy",
            "sdk": "live_agent_self_discovery",
            "new_solve_credit": False,
        },
        "raw_reduction": summary,
        "raw_probes_path": str(raw_path.relative_to(ROOT)),
        "raw_probes_sha256": sha256(raw_path),
        "execution_venue": "host",
        "execution_venue_details": {
            "host": socket.gethostname(),
            "pid": os.getpid(),
            "gpu_uuid": None,
        },
        "repository_health": {
            "historical_exp7803_verdict": "complete_disqualified_required_runner_validation",
            "broad_suite_current": None,
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
                "source": "raw_action_telemetry",
            }
            for row in probes
        ],
        "validation_command_manifest_path": str(MANIFEST.relative_to(SOURCE_ROOT)),
        "validation_command_manifest_sha256": sha256(MANIFEST),
        "observed_child_commands": [
            {
                "name": row["name"],
                "argv": row["command_argv"],
                "classification": row["classification"],
            }
            for row in receipts
        ],
    }
    value["field_principles"] = {key: "Exact current evidence bounds this field." for key in value}
    value["field_principles"].update(
        {
            "honest_verdict": "A terminal result has one owner.",
            "verdict_class": "Claim strength travels with the record.",
            "flagged_adversarial": "Invalid evidence cannot open a gate.",
            "gate_check_summary": "Missing evidence differs from a scientific null.",
            "rows": "Recompute comparisons from independent units.",
            "organic_runner_ready_score": "A scored transport fixture proves no benefit.",
            "sample_size_budget": "Views and seeds do not add independent game families.",
            "validation_command_manifest_path": "The prospective command list cannot change silently.",
            "repository_health": "Broad failures remain visible without changing affected scope.",
        }
    )
    for gate in gates:
        value["field_principles"][f"acceptance_gate_results.{gate}"] = "Unmeasured gain stays null."
    return value


def main(argv: list[str] | None = None) -> int:
    """Execute only the frozen current commands and publish one terminal result."""
    start = time.monotonic()
    progress(start, "startup", "before", pid=os.getpid())
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", default="20260928")
    parser.add_argument("--record-dispatch", type=Path)
    parser.add_argument("--list-commands", action="store_true")
    parser.add_argument("--probe", nargs=2, metavar=("GAME", "ARM"))
    parser.add_argument("--cold-reduce", type=Path)
    parser.add_argument("--cold-replay", type=Path)
    args = parser.parse_args(argv)
    if args.date != "20260928":
        raise ValueError("run_date_must_match_milestone")
    if args.record_dispatch or args.list_commands:
        plan = command_plan(MANIFEST)
        if args.record_dispatch:
            atomic_json(args.record_dispatch, plan)
        else:
            print(json.dumps(plan, sort_keys=True), flush=True)
        progress(start, "dispatch_plan", "after", len(plan))
        return 0
    if args.cold_reduce:
        progress(start, "cold_reduce", "before")
        result = cold_reduce(args.cold_reduce)
        progress(start, "cold_reduce", "after", result["started"])
        return 0
    if args.cold_replay:
        progress(start, "cold_replay", "before")
        cold_replay(args.cold_replay)
        progress(start, "cold_replay", "after", 1)
        return 0
    if args.probe:
        game, arm = args.probe
        progress(start, "model_load", "before")
        progress(start, "model_load", "after")
        progress(start, "generation", "before")
        progress(start, "generation", "after")
        progress(start, "scored_probe", "before", game=game, arm=arm)
        row = run_probe(game, arm, offline_arcade())
        print("PROBE_JSON=" + json.dumps(row, sort_keys=True, separators=(",", ":")), flush=True)
        progress(start, "scored_probe", "after", len(row["actions"]), error=row["error"])
        return int(row["error"] is not None)
    RAW.mkdir(parents=True, exist_ok=True)
    plan = command_plan(MANIFEST)
    specs = json.loads(MANIFEST.read_text())["commands"]
    run_id = uuid.uuid4().hex
    private = Path("/tmp") / f"exp7817-v679-{run_id}"
    private.mkdir(parents=True)
    durable = RAW / "validation_logs" / run_id
    for phase in ("model_load", "generation"):
        progress(start, phase, "before")
        progress(start, phase, "after")
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
    panel = freeze_panel(OLD_PANEL)
    panel["schema"] = "carnot.exp7817.frozen_arc_panel.v1"
    panel["source_panel_sha256"] = sha256(OLD_PANEL)
    panel_path = RAW / "arc_panel_manifest.json"
    atomic_json(panel_path, panel)
    progress(start, "panel_freeze", "after", len(panel["rows"]), sha256=sha256(panel_path))
    fixture: dict[str, Any] = {}
    probes: list[dict[str, Any]] = []
    receipts: list[dict[str, Any]] = []
    if all(row["passed"] for row in checks):
        progress(start, "selector_fixture", "before")
        fixture = positive_selector_fixture()
        atomic_json(RAW / "selector_fixture.json", fixture)
        progress(start, "selector_fixture", "after", 1)
        phase = time.monotonic()
        for index, spec in enumerate(specs[:6]):
            receipt = run_child(spec, plan, private, durable, index + 1, start, len(receipts))
            receipts.append(receipt)
            try:
                row = extract_probe(receipt)
            except ValueError:
                continue
            row["raw_path"] = receipt["log_path"]
            row["raw_sha256"] = receipt["log_sha256"]
            probes.append(row)
        spans.append(
            {
                "phase": "scored_probes",
                "duration_s": time.monotonic() - phase,
                "completed_units": len(probes),
            }
        )
    summary = reduce_probe_evidence(panel["rows"], probes)
    raw_path = RAW / "raw_probes.json"
    atomic_json(raw_path, {"schedule": panel["rows"], "probes": probes, "summary": summary})
    if all(row["passed"] for row in checks):
        phase = time.monotonic()
        for index, spec in enumerate(specs[6:-3], start=7):
            receipt = run_child(spec, plan, private, durable, index, start, len(receipts))
            receipts.append(receipt)
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
        sources,
        resources,
        panel_path,
        probes,
        fixture,
        summary,
        receipts,
        args.date,
    )
    diagnostic = next(
        (row for row in receipts if row["name"] == "repository_health_full_python_suite"), None
    )
    value["repository_health"]["broad_suite_current"] = diagnostic
    if any(not row["passed"] for row in checks):
        value["honest_verdict"] = "complete_blocked_preconditions"
        value["verdict_class"] = "blocked"
    candidate = RAW / "terminal_candidate.json"
    progress(start, "candidate", "before")
    atomic_json(candidate, value)
    progress(start, "candidate", "after", 1, sha256=sha256(candidate))
    if all(row["passed"] for row in checks):
        for index, spec in enumerate(specs[-3:], start=len(receipts) + 1):
            receipts.append(run_child(spec, plan, private, durable, index, start, len(receipts)))
    failed = validate_receipts(MANIFEST, receipts, sdk_ok=sdk_probe_ok(probes))
    if any(not row["passed"] for row in checks):
        value["honest_verdict"] = "complete_blocked_preconditions"
        value["verdict_class"] = "blocked"
    elif failed:
        value["honest_verdict"] = "complete_disqualified_required_runner_validation"
        value["verdict_class"] = "disqualified"
    else:
        value["organic_runner_ready_score"] = 1
        value["acceptance_gate_results"].update(validity=True, readiness=True)
        value["honest_verdict"] = "complete_circular_positive_runner_qualified_no_benefit"
        value["verdict_class"] = "circular_positive"
    for name in failed:
        receipt = next((row for row in receipts if row["name"] == name), None)
        path = ROOT / receipt["log_path"] if receipt else MANIFEST
        value["gate_check_summary"].append(
            operand(
                path, name, 0, receipt["exit_code"] if receipt else "missing", "current_validation"
            )
        )
    value["flagged_adversarial"] = any(
        row["name"] == "adversarial_verify" and not row["passed"] for row in receipts
    )
    value["validation_receipts"] = receipts
    value["observed_child_commands"] = [
        {"name": row["name"], "argv": row["command_argv"], "classification": row["classification"]}
        for row in receipts
    ]
    value["phase_spans"] = spans
    value["duration_s"] = time.monotonic() - start
    progress(start, "publish", "before", verdict=value["honest_verdict"])
    atomic_json(RESULT, value)
    progress(start, "publish", "after", 1, sha256=sha256(RESULT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
