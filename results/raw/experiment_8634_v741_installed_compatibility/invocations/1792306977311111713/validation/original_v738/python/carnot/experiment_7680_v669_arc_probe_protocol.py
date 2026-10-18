"""REQ-REPORT-7680: measure a default-off scored ARC probe seam on CPU."""

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

import numpy as np

from carnot.agentic.arc_competition_agent import make_carnot_agent
from carnot.agentic.arc_probe_protocol import ArcProbeProtocol
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash, sha256_file
from carnot.reporting.experiment_7303_validation_scope import (
    CommandSpec,
    build_scoped_commands,
    run_commands,
)

ROOT = Path(__file__).resolve().parents[2]
RAW = Path("results/raw/experiment_7680_v669_arc_probe_protocol")
RESULT = Path("results/experiment_7680_v669_arc_probe_protocol.json")
MODEL_SPECS: list[dict[str, Any]] = []
CASES = ("ordinary", "no_engine", "all_refuted", "irreversible", "hud", "stale", "false_terminal")
ARMS = ("current", "novelty", "guided")
MANIFEST = {
    "test_paths": ["tests/python/test_experiment_7680_v669_arc_probe_protocol.py"],
    "changed_modules": ["python/carnot/agentic/arc_probe_protocol.py"],
    "static_paths": [
        "python/carnot/agentic/arc_competition_agent.py",
        "python/carnot/experiment_7680_v669_arc_probe_protocol.py",
        "scripts/experiments/experiment_7680_v669_arc_probe_protocol.py",
    ],
    "e2e_paths": [
        "tests/python/test_arc_induction_state_persistence.py",
        "tests/python/test_arc_decision_telemetry.py",
        "tests/python/test_experiment_7666_v668_arc_goal_confirmation.py",
    ],
}


def progress(started: float, phase: str, event: str, **details: Any) -> None:
    """Flush every phase and long unit so a slow child remains observable."""
    print(
        f"[exp7680] {phase} {event} elapsed_s={time.monotonic() - started:.3f} {details}",
        flush=True,
    )


def fixture_frame(seed: int, step: int, case: str) -> SimpleNamespace:
    """Build public SDK-shaped frames without hidden state or game adapters."""
    grid = np.zeros((8, 8), dtype=int)
    if case == "all_refuted":
        grid[7, seed] = 9
        grid.flat[:step] = 1
    else:
        grid[2:6, 2:6] = (seed + step) % 4
        grid[0, :] = (seed + step) % 8
    actions = [1, 2, 3, 4, 5]
    if case == "irreversible" and step >= 2:
        actions = [1, 3, 4, 5]
    return SimpleNamespace(
        frame=[grid.tolist()],
        levels_completed=int(case == "irreversible" and step >= 3),
        state="NOT_FINISHED",
        available_actions=actions,
        score=0,
    )


def cold_reduce_rows(rows: Sequence[dict[str, Any]]) -> dict[str, int]:
    """Reduce only raw arm rows; reject a group that lacks a control."""
    groups: dict[str, set[str]] = {}
    for row in rows:
        group = str(row["group_id"])
        arm = str(row["arm"])
        if arm in groups.setdefault(group, set()):
            raise ValueError("duplicate arms in group")
        groups[group].add(arm)
    if any(arms != set(ARMS) for arms in groups.values()):
        raise ValueError("missing arms in group")
    return {
        "independent_groups": len(groups),
        "rows": len(rows),
        "admitted_probes": sum(int(row["raw_metrics"]["admitted"]) for row in rows),
        "false_goal_confirmations": sum(
            int(row["raw_metrics"]["false_confirmation"]) for row in rows
        ),
        "rejected_probes": sum(int(row["raw_metrics"].get("rejected", 0)) for row in rows),
    }


def measure(root: Path, raw: Path, started: float) -> list[dict[str, Any]]:
    """Execute three actual scored wrapper arms for each independent fixture."""
    os.environ["CARNOT_ARC_DISABLE_INDUCTION"] = "1"

    class Base:
        game_id = "adapter_withheld_fixture"

        def __init__(self) -> None:
            pass

    rows: list[dict[str, Any]] = []
    for case in CASES:
        for seed in range(8):
            group = f"{case}-{seed}"
            for arm in ARMS:
                agent = make_carnot_agent(
                    Base,
                    arc_probe_protocol=None if arm == "current" else arm,
                    goal_confirmation=True,
                )()
                policy = agent._policy
                frames: list[Any] = []
                actions: list[str] = []
                for step in range(4):
                    latest = (
                        frames[-1]
                        if case == "stale" and step == 2
                        else fixture_frame(seed, step, case)
                    )
                    if case == "false_terminal" and step == 3:
                        policy._goal_confirmation.arm(
                            frames[-1],
                            frames_seen=len(frames) - 1,
                            level=0,
                            predicted_goal=True,
                            plan_length=1,
                        )
                    action = agent.choose_action(frames, latest)
                    actions.append(str(action))
                    frames.append(latest)
                    if arm != "current" and step == 1:
                        checkpoint = policy._arc_probe_protocol.checkpoint()
                        atomic_json(raw / "checkpoints" / f"{group}-{arm}.json", checkpoint)
                        policy._arc_probe_protocol = ArcProbeProtocol.from_checkpoint(checkpoint)
                diagnostics = policy.arc_probe_diagnostics()
                decisions = diagnostics.get("decisions", [])
                hypotheses = diagnostics.get("goal_hypotheses", [])
                all_refuted = bool(hypotheses) and all(
                    item["state"] == "refuted" for item in hypotheses
                )
                if case == "all_refuted" and arm != "current" and not all_refuted:
                    raise AssertionError("all_refuted fixture retained a live hypothesis")
                admitted = sum(int(item["admitted"]) for item in decisions)
                rejected = sum(int(not item["admitted"]) for item in decisions)
                guard = policy.goal_confirmation_receipts()
                row = {
                    "group_id": group,
                    "arm": arm,
                    "case": case,
                    "seed": seed,
                    "actions": actions,
                    "decisions": decisions,
                    "goal_hypotheses": hypotheses,
                    "goal_guard": guard,
                    "raw_metrics": {
                        "admitted": admitted,
                        "rejected": rejected,
                        "false_confirmation": sum(
                            item["status"] == "confirmed"
                            for item in guard
                            if case == "false_terminal"
                        ),
                        "sdk_progress": diagnostics.get("sdk_progress", 0),
                        "all_refuted": all_refuted,
                    },
                    "counts": {"wrapper_calls": 4, "model_calls": 0, "virtual_engine_calls": 0},
                    "exclusions": [],
                    "censored": case == "stale",
                    "provenance": "development_proxy_scored_wrapper_scripted_frames",
                }
                atomic_json(raw / "rows" / f"{group}-{arm}.json", row)
                rows.append(row)
            progress(
                started, "measurement", "unit_checkpoint", groups=len(rows) // 3, rows=len(rows)
            )
    return rows


def gate_check(
    check: str, upstream: str, path: Path, field: str, expected: Any, observed: Any
) -> dict[str, Any]:
    """Make a missing upstream operand readable without guessing its meaning."""
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
    """Authenticate exact input bytes and separate missing inherited live science."""
    required = {
        "v668_fixture": root / "results/experiment_7666_v668_arc_goal_confirmation.json",
        "registry": root / "ops/arc_solve_registry.yaml",
        "report_spec": root / "openspec/capabilities/research-reporting/spec.md",
        "capability_spec": root / "openspec/capabilities/arc-agi/spec.md",
        "policy": root / "python/carnot/agentic/arc_competition_agent.py",
        "frontier": root / "python/carnot/agentic/arc_active_reward_machine_frontier.py",
        "contract": root / "python/carnot/agentic/arc_two_sided_goal_contract.py",
        "scheduler": root / "python/carnot/agentic/arc_probe_protocol.py",
        "reducer": Path(__file__).resolve(),
        "tests": root / MANIFEST["test_paths"][0],
    }
    failed: list[dict[str, Any]] = []
    hashes: dict[str, Any] = {"producers": {}, "pre_gate_receipts": {}, "missing_evidence": []}
    for label, path in required.items():
        if not path.is_file():
            failed.append(gate_check("input_exists", label, path, "exists", True, False))
        else:
            hashes["producers"][str(path.relative_to(root))] = sha256_file(path)
    inherited_live = root / "results/experiment_7667_v668_arc_goal_observation.json"
    if not inherited_live.is_file():
        hashes["missing_evidence"].append(
            gate_check(
                "upstream_live_exists",
                "v668_live_producer",
                inherited_live,
                "exists",
                True,
                False,
            )
        )
    if not failed:
        try:
            inherited = json.loads(required["v668_fixture"].read_text())
            if inherited.get("verdict_class") != "circular_positive":
                failed.append(
                    gate_check(
                        "fixture_class",
                        "v668_fixture",
                        required["v668_fixture"],
                        "verdict_class",
                        "circular_positive",
                        inherited.get("verdict_class"),
                    )
                )
        except (OSError, ValueError) as exc:
            failed.append(
                gate_check(
                    "fixture_parse",
                    "v668_fixture",
                    required["v668_fixture"],
                    "json_parse",
                    True,
                    repr(exc),
                )
            )
    return failed, hashes


def validation_commands(root: Path, private: Path) -> list[CommandSpec]:
    """Freeze affected files and private coverage paths before fixture outcomes."""
    return build_scoped_commands(
        root,
        MANIFEST["test_paths"],
        MANIFEST["changed_modules"],
        static_paths=MANIFEST["static_paths"],
        basetemp=private / "basetemp",
        coverage_file=private / "coverage.data",
    )


def span(
    spans: list[dict[str, Any]], start: float, phase_start: float, name: str, completed: int
) -> None:
    """Record one disjoint monotonic phase with its completed checkpoint count."""
    end = time.monotonic()
    spans.append(
        {
            "phase": name,
            "start_s": phase_start - start,
            "end_s": end - start,
            "duration_s": end - phase_start,
            "heartbeat_s": end - start,
            "completed_units": completed,
            "checkpoint": completed,
        }
    )


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
    """Separate work completion, fixture validity, and absent live efficacy."""
    required_ok = bool(receipts) and all(row["passed"] for row in receipts)
    terminal_ok = len(terminal) == 2 and all(row["passed"] for row in terminal)
    n = reduced.get("independent_groups", 0)
    false = reduced.get("false_goal_confirmations", 0)
    ready = int(not failed and n >= 48 and false == 0 and required_ok and terminal_ok)
    verdict_class = (
        "blocked"
        if failed
        else "disqualified"
        if receipts and not required_ok or terminal and not terminal_ok
        else "circular_positive"
    )
    verdict = {
        "blocked": "complete_blocked_required_input",
        "disqualified": "complete_disqualified_required_validation",
        "circular_positive": "complete_circular_positive_probe_fixture",
    }[verdict_class]
    gate = {
        "validity": {
            "passed": not failed and required_ok and terminal_ok,
            "input_failures": len(failed),
            "failed_checks": sum(not r["passed"] for r in receipts + terminal),
        },
        "readiness": {
            "passed": bool(ready),
            "score": ready,
            "scored_wrapper_groups": n,
            "minimum_groups": 48,
        },
        "coverage": {
            "passed": required_ok,
            "changed_behavior_target_percent": 100,
            "coverage_receipt": next(
                (r for r in receipts if r["name"] == "changed_module_coverage_report"), None
            ),
        },
        "freshness": {"passed": False, "live_producer_available": not hashes["missing_evidence"]},
        "probability": {"passed": False, "independent_hidden_games": 0},
        "decision_utility": {
            "passed": False,
            "new_hidden_game_wins": 0,
            "admitted_probes": reduced.get("admitted_probes", 0),
        },
        "retention": {"passed": False, "live_retest_groups": 0},
        "efficiency": {"passed": False, "live_action_cost_measured": False},
    }
    return {
        "experiment_id": 7680,
        "milestone": "2026.09.669",
        "honest_verdict": verdict,
        "verdict_class": verdict_class,
        "flagged_adversarial": not terminal_ok if terminal else False,
        "gate_check_summary": failed,
        "acceptance_gate_results": gate,
        "arc_probe_protocol_ready_score": ready,
        "arc_generalization_task": True,
        "rows": rows,
        "cold_reduction": reduced,
        "sample_size_budget": {
            "intended": 56,
            "observed": n,
            "eligible": n,
            "excluded": 0,
            "censored": sum(r["censored"] for r in rows if r["arm"] == "guided"),
            "independent_groups": n,
            "effective_blocks": n,
            "prior_exposure": "scripted development proxy; no hidden-game inference",
            "limits": "eight constructed configurations in each of seven fixture families",
        },
        "inference_substrate": "cpu_scripted_scored_wrapper_no_model_load",
        "inference_substrate_class": "no_model_load",
        "planned_inference_substrate_class": "no_model_load",
        "MODEL_SPECS": MODEL_SPECS,
        "planned_MODEL_SPECS": [],
        "model_specs": [{"no_model": True, "reason": "CPU scripted fixture protocol"}],
        "model_invoked": False,
        "invocation_counts": {
            key: 0
            for key in (
                "loads_attempted",
                "loads_completed",
                "loads_cancelled",
                "forward_calls_attempted",
                "forward_calls_completed",
                "generations_attempted",
                "generations_completed",
                "input_tokens",
                "output_tokens",
            )
        },
        "historical_model_provenance": "V668 fixture is inherited; no current model calls",
        "execution_venue": "host",
        "execution_venue_details": {
            "host": socket.gethostname(),
            "gpu_uuid": None,
            "owned_pid": os.getpid(),
        },
        "phase_spans": spans,
        "duration_s": duration,
        "random_seed": {
            "fixture_seeds": list(range(8)),
            "purpose": "distinct scripted initial grids, not repeated views",
        },
        "reproducibility_checksum": canonical_hash(
            {
                "inputs": hashes["producers"],
                "cases": CASES,
                "arms": ARMS,
                "reducer": hashes["producers"].get(
                    "python/carnot/experiment_7680_v669_arc_probe_protocol.py"
                ),
            }
        ),
        "source_artifact_hashes": hashes,
        "preconditions_checked": {
            "root": str(ROOT),
            "host_cpu_available": True,
            "model_load_required": False,
            "required_input_failures": failed,
            "inherited_live_evidence_missing": hashes["missing_evidence"],
        },
        "validation_receipts": {
            "affected_file_manifest": MANIFEST,
            "required_checks": receipts,
            "terminal_readers": terminal,
            "required_checks_passed": required_ok and terminal_ok,
            "cold_reduction": reduced,
            "unrelated_full_suite_debt": "separate from affected acceptance gates",
        },
        "verifier_is_oracle": True,
        "field_principles": {
            "rows": "One constructed fixture configuration counts once across three scored arms.",
            "honest_verdict": "Completion is separate from a scientific benefit or hidden-game solve.",
            "gate_check_summary": "A blocked check names exact upstream operands.",
            "acceptance_gate_results": "Validity, readiness, and live benefit use distinct evidence.",
            "inference_substrate": "Only current CPU work determines the substrate floor.",
            "MODEL_SPECS": "No current LLM is invoked.",
            "sample_size_budget": "Views and arms do not multiply independent groups.",
            "arc_probe_protocol_ready_score": "Requires scored reachability, bounds, guard, telemetry, and validation.",
            "goal_guard_results": "Only actual SDK progress can confirm a terminal goal.",
            "source_artifact_hashes": "Immutable inputs and pre-gate receipts bind replay.",
            "phase_spans": "Disjoint monotonic intervals account for current elapsed time.",
        },
        "solve_provenance": "development_proxy",
        "new_game_level_solve_credit": False,
        "probe_contract_path": str(RAW / "contract.json"),
        "wrapper_route_rows": [
            {
                "group_id": row["group_id"],
                "arm": row["arm"],
                "actions": row["actions"],
                "admitted": row["raw_metrics"]["admitted"],
                "rejected": row["raw_metrics"]["rejected"],
                "path": "make_carnot_agent.choose_action -> E3AgentPolicy.next_move",
            }
            for row in rows
        ],
        "goal_guard_results": {
            "false_terminal_confirmations": false,
            "ambiguous_goals": sum(
                item["goal_guard"] == "unverifiable" for row in rows for item in row["decisions"]
            ),
            "next_level_frame_handling": "new episode opening; prior feature bank reset",
            "authentic_sdk_terminal_authority": True,
        },
        "same_verdict_retirement": {
            "mechanism": "negative-only goal identification",
            "decision": "retire_unique_goal_claim_without_positive_evidence",
            "basis": "bounded probes can test effects; no-win histories remain ambiguous",
            "resource_absence": "V668 live producer absent; fixture path is current CPU work",
        },
    }


def persist_receipts(
    root: Path, raw: Path, family: str, receipts: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Keep exact child output bytes next to the rows before publication."""
    durable = []
    for index, row in enumerate(receipts):
        source = Path(row["log_path"])
        if not source.is_absolute():
            source = root / source
        target = raw / "logs" / family / f"{index:02d}_{row['name']}.log"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(source.read_bytes())
        durable.append(
            {**row, "log_path": str(target.relative_to(root)), "log_sha256": sha256_file(target)}
        )
    return durable


def run_experiment(root: Path, run_date: str, output: Path) -> dict[str, Any]:
    """Freeze, measure, validate, cold-reduce, and atomically publish once."""
    started = time.monotonic()
    root = root.resolve()
    progress(started, "startup", "begin", root=root)
    if root != ROOT or run_date != "20260926":
        raise ValueError("run_contract_mismatch")
    raw = root / RAW
    raw.mkdir(parents=True, exist_ok=True)
    private = Path(tempfile.mkdtemp(prefix="carnot-exp7680-")).resolve()
    (private / "basetemp").mkdir(parents=True, exist_ok=True)
    spans: list[dict[str, Any]] = []

    phase = time.monotonic()
    progress(started, "preconditions", "before")
    failed, hashes = preconditions(root)
    progress(started, "preconditions", "after", failures=len(failed))
    span(spans, started, phase, "preconditions", len(hashes["producers"]))
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
        atomic_json(output, final)
        return final

    phase = time.monotonic()
    progress(started, "freeze", "before")
    commands = validation_commands(root, private)
    contract = {
        "requirement": "REQ-REPORT-7680",
        "capability": "REQ-ARC-PROBE-7680",
        "cases": CASES,
        "fixture_seeds": list(range(8)),
        "arms": ARMS,
        "independent_groups": 56,
        "action_limit": 4,
        "hypothesis_limit": 16,
        "candidate_limit": 5,
        "probe_decision_limit": 8,
        "virtual_engine_call_limit": 0,
        "decision_wall_time_s": 0.01,
        "oracle_truth": "scripted_sdk_fixture",
        "MODEL_SPECS": MODEL_SPECS,
        "inference_substrate_class": "no_model_load",
        "affected_validation_manifest": MANIFEST,
        "commands": [{"name": c.name, "argv": list(c.argv)} for c in commands],
    }
    atomic_json(raw / "contract.json", contract)
    hashes["pre_gate_receipts"][str(RAW / "contract.json")] = sha256_file(raw / "contract.json")
    progress(started, "freeze", "after", groups=56, commands=len(commands))
    span(spans, started, phase, "freeze", 1)

    phase = time.monotonic()
    progress(started, "measurement", "before_benchmark")
    rows = measure(root, raw, started)
    atomic_json(raw / "rows.json", rows)
    progress(started, "measurement", "after_benchmark", groups=len(rows) // 3)
    span(spans, started, phase, "measurement", len(rows) // 3)

    phase = time.monotonic()
    progress(started, "validation", "before_subprocess")
    receipts = run_commands(
        root,
        commands,
        log_dir=private / "logs" / "affected",
        extra_env={"JAX_PLATFORMS": "cpu", "COVERAGE_FILE": str(private / "coverage.data")},
        heartbeat_s=30,
    )
    receipts = persist_receipts(root, raw, "affected", receipts)
    progress(
        started, "validation", "after_subprocess", failures=sum(not r["passed"] for r in receipts)
    )
    span(spans, started, phase, "validation", len(receipts))

    phase = time.monotonic()
    progress(started, "e2e", "before_subprocess")
    e2e = run_commands(
        root,
        [
            CommandSpec(
                "e2e_009_011_goal",
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
                "E2E-009_E2E-011_and_goal_guard",
                900.0,
            )
        ],
        log_dir=private / "logs" / "e2e",
        extra_env={"JAX_PLATFORMS": "cpu"},
        heartbeat_s=30,
    )
    e2e = persist_receipts(root, raw, "e2e", e2e)
    receipts.extend(e2e)
    progress(started, "e2e", "after_subprocess", failures=sum(not r["passed"] for r in e2e))
    span(spans, started, phase, "e2e", len(e2e))

    phase = time.monotonic()
    progress(started, "cold_reduction", "before_subprocess")
    cold = run_commands(
        root,
        [
            CommandSpec(
                "cold_reduction",
                (
                    str(root / ".venv/bin/python"),
                    "-u",
                    "-m",
                    "carnot.experiment_7680_v669_arc_probe_protocol",
                    "--cold-reduce",
                    str(raw / "rows.json"),
                ),
                "exact_raw_rows",
                300.0,
            )
        ],
        log_dir=private / "logs" / "cold",
        extra_env={"JAX_PLATFORMS": "cpu"},
        heartbeat_s=30,
    )
    cold = persist_receipts(root, raw, "cold", cold)
    try:
        reduced = json.loads((root / cold[0]["log_path"]).read_text().strip().splitlines()[-1])
    except (IndexError, ValueError):
        reduced = {}
    if reduced != cold_reduce_rows(rows):
        cold[0]["passed"] = False
    receipts.extend(cold)
    progress(
        started,
        "cold_reduction",
        "after_subprocess",
        groups=reduced.get("independent_groups", 0),
        exit=cold[0]["exit_code"],
    )
    span(spans, started, phase, "cold_reduction", len(rows) // 3)

    phase = time.monotonic()
    candidate_path = raw / "terminal_candidate.json"
    candidate = artifact(
        rows=rows,
        reduced=reduced,
        failed=[],
        hashes=hashes,
        receipts=receipts,
        terminal=[],
        spans=spans,
        duration=time.monotonic() - started,
    )
    atomic_json(candidate_path, candidate)
    candidate_hash = sha256_file(candidate_path)
    progress(started, "terminal_readers", "before_subprocess", candidate_sha256=candidate_hash)
    terminal = run_commands(
        root,
        [
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
        ],
        log_dir=private / "logs" / "terminal",
        extra_env={"JAX_PLATFORMS": "cpu"},
        heartbeat_s=30,
    )
    terminal = persist_receipts(root, raw, "terminal", terminal)
    atomic_json(
        raw / "terminal_validation_receipts.json",
        {
            "candidate_sha256": candidate_hash,
            "readers": terminal,
        },
    )
    progress(
        started,
        "terminal_readers",
        "after_subprocess",
        failures=sum(not r["passed"] for r in terminal),
    )
    span(spans, started, phase, "terminal_readers", len(terminal))

    final = artifact(
        rows=rows,
        reduced=reduced,
        failed=[],
        hashes=hashes,
        receipts=receipts,
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
    """Parse CLI arguments and keep the scored protocol in the package module."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date")
    parser.add_argument("--output", type=Path, default=RESULT)
    parser.add_argument("--cold-reduce", type=Path)
    args = parser.parse_args(argv)
    if args.cold_reduce is not None:
        print(
            json.dumps(cold_reduce_rows(json.loads(args.cold_reduce.read_text())), sort_keys=True),
            flush=True,
        )
        return 0
    if args.date is None:
        parser.error("--date is required")
    output = args.output if args.output.is_absolute() else ROOT / args.output
    run_experiment(ROOT, args.date, output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
