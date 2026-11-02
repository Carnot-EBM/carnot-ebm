"""Terminal V672 ARC evidence recovery (REQ-REPORT-7722)."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import socket
import tempfile
import time
from typing import Any

from carnot.agentic.arc_evidence_recovery import recover_history
from carnot.reporting.current_work_receipt import atomic_json, sha256_file

Json = dict[str, Any]
ROOT = Path(__file__).resolve().parents[2]
RAW = Path("results/raw/experiment_7722_v672_arc_evidence_recovery")
OLD_RAW = Path("results/raw/experiment_7709_v671_arc_first_contact")
RESULT = Path("results/experiment_7722_v672_arc_evidence_recovery.json")
SCOPE = {
    "tests": [
        "tests/python/test_experiment_7708_v671_arc_generalization_runner.py",
        "tests/python/test_experiment_7709_v671_arc_first_contact.py",
        "tests/python/test_experiment_7722_v672_arc_evidence_recovery.py",
    ],
    "changed_modules": [
        "python/carnot/agentic/arc_evidence_recovery.py",
        "python/carnot/experiment_7722_v672_arc_evidence_recovery.py",
    ],
    "static_paths": ["scripts/experiments/experiment_7722_v672_arc_evidence_recovery.py"],
    "e2e": ["E2E-009", "E2E-011", "E2E-013"],
}
REQUIRED = (
    "AGENTS.md",
    "CODEX.md",
    "CLAUDE.md",
    "research-program.md",
    "ops/exclusion_manifest.yaml",
    "ops/e2e-test-plan.md",
    "scripts/experiment_template.py",
    "python/carnot/reporting/current_work_receipt.py",
    "python/carnot/reporting/experiment_7303_validation_scope.py",
    "openspec/capabilities/research-reporting/spec.md",
    "openspec/capabilities/arc-world-model-trust-energy/spec.md",
    "python/carnot/agentic/arc_competition_agent.py",
    "python/carnot/agentic/arc_solver_kit.py",
    "python/carnot/agentic/arc_evidence_recovery.py",
    "python/carnot/experiment_7722_v672_arc_evidence_recovery.py",
    "scripts/experiments/experiment_7722_v672_arc_evidence_recovery.py",
    *SCOPE["tests"],
)
GATE_NAMES = (
    "validity",
    "readiness",
    "probability",
    "utility",
    "coverage",
    "source_dependence",
    "retention",
    "efficiency",
)


def progress(started: float, phase: str, event: str, **detail: Any) -> None:
    """Print a flushed owner boundary with elapsed time and completed units."""
    print(
        f"[exp7722] {phase} {event} elapsed_s={time.monotonic() - started:.3f} {detail}", flush=True
    )


def _gate(passed: bool | None, principle: str, **operands: Any) -> Json:
    return {"passed": passed, "principle": principle, "measured_operands": operands}


def read_suite_debt(path: Path, root: Path) -> Json:
    """Reuse only the exact failed global-suite command and its hashed owned log."""
    receipts = json.loads(path.read_text(encoding="utf-8"))
    matches = [row for row in receipts if row.get("name") == "full_python_suite"]
    if len(matches) != 1:
        raise ValueError("suite_debt_receipt_invalid")
    receipt = matches[0]
    log = root / receipt.get("log_path", "")
    valid = (
        receipt.get("command_argv") == [str(root / ".venv/bin/pytest"), "tests/python", "-q"]
        and receipt.get("scope") == "global_debt"
        and receipt.get("exit_code") != 0
        and log.is_file()
        and sha256_file(log) == receipt.get("log_sha256")
    )
    if not valid:
        raise ValueError("suite_debt_receipt_invalid")
    return dict(receipt)


def preflight(root: Path) -> tuple[list[Json], Json]:
    """Authenticate the current source and requirements without requiring own output."""
    checks: list[Json] = []
    hashes: Json = {}
    for relative in REQUIRED:
        path = root / relative
        observed = sha256_file(path) if path.is_file() else None
        checks.append(
            {
                "check": "current_input_bytes",
                "upstream": relative,
                "path": str(path),
                "field": "sha256",
                "operator": "!=",
                "expected": None,
                "observed": observed,
                "passed": observed is not None,
            }
        )
        if observed is not None:
            hashes[relative] = observed
    for relative, requirement in (
        ("openspec/capabilities/research-reporting/spec.md", "REQ-REPORT-7722"),
        ("openspec/capabilities/arc-world-model-trust-energy/spec.md", "REQ-ARC-WMTE-7722"),
    ):
        path = root / relative
        observed = requirement in path.read_text(encoding="utf-8") if path.is_file() else False
        checks.append(
            {
                "check": "driving_requirement",
                "upstream": relative,
                "path": str(path),
                "field": requirement,
                "operator": "==",
                "expected": True,
                "observed": observed,
                "passed": observed,
            }
        )
    return checks, hashes


def build_artifact(
    recovered: Json,
    receipts: list[Json],
    run_date: str,
    duration_s: float,
    *,
    current_checks: list[Json] | None = None,
    current_hashes: Json | None = None,
    spans: list[Json] | None = None,
) -> Json:
    """Keep scientific support separate from validation and old disqualification."""
    checks = [*(current_checks or []), *recovered.get("checks", [])]
    failed = [row for row in checks if row.get("passed") is not True]
    from carnot.reporting.experiment_7303_validation_scope import REQUIRED_CHECK_NAMES

    names = {
        row.get("name")
        for row in receipts
        if row.get("exit_code") == 0 and row.get("passed") is True
    }
    required = set(REQUIRED_CHECK_NAMES) | {"e2e_009", "e2e_011", "e2e_013", "e2e_009_smoke"}
    validation_ok = required <= names and all(
        row.get("exit_code") == 0 for row in receipts if row.get("scope") != "global_debt"
    )
    rows = recovered.get("rows", [])
    authentic = not failed and len(rows) == 2 and recovered.get("joined_actions") == 256
    if failed:
        verdict, reason = "blocked", str(failed[0]["check"])
    elif not validation_ok:
        verdict, reason = "disqualified", "required_validation"
    else:
        verdict, reason = "null", "two_public_game_historical_generalization"
    ready = int(authentic and validation_ok and verdict == "null")
    gates = {
        "validity": _gate(
            authentic,
            "Original bytes and joins bound evidence validity.",
            joined_actions=recovered.get("joined_actions"),
            failed_checks=len(failed),
        ),
        "readiness": _gate(
            bool(ready),
            "Required validation bounds measurement readiness.",
            required_checks=len(required),
            passed_checks=len(names & required),
        ),
        "coverage": _gate(
            validation_ok,
            "Every frozen check must pass.",
            required_checks=sorted(required),
            passed_checks=sorted(names & required),
        ),
        "probability": _gate(
            None, "Two known public games cannot estimate hidden-game probability.", hidden_games=0
        ),
        "utility": _gate(
            None, "Unpaired null progress cannot establish decision benefit.", paired_arms=0
        ),
        "source_dependence": _gate(
            False,
            "Original history is identified and never new solve credit.",
            original_v671_verdict=recovered.get("prior_verdict"),
        ),
        "retention": _gate(None, "No independent retention comparison was run.", groups=0),
        "efficiency": _gate(None, "No paired action-cost comparison was run.", paired_arms=0),
    }
    source_hashes = dict(recovered.get("hashes", {}))
    source_hashes.setdefault("valid_producers", {}).update(current_hashes or {})
    seed = {"selection_salt": "v671-arc-20260926", "reduction_seed": 7722}
    checksum = (
        "sha256:"
        + hashlib.sha256(
            json.dumps([source_hashes, SCOPE, seed], sort_keys=True).encode()
        ).hexdigest()
    )
    supervisor = recovered.get("supervisor", {})
    per_game = [
        {**row, "supervisor": supervisor.get("per_game", {}).get(row["game"])} for row in rows
    ]
    artifact: Json = {
        "schema": "carnot.exp7722.v672.arc_evidence_recovery.v1",
        "experiment_id": 7722,
        "milestone": "2026.09.672",
        "run_date": run_date,
        "status": "complete",
        "honest_verdict": f"complete_{verdict}_{reason}",
        "verdict_class": verdict,
        "flagged_adversarial": False,
        "gate_check_summary": {"failed_checks": failed, "failed_count": len(failed)},
        "acceptance_gate_results": gates,
        "rows": per_game,
        "per_game_results": per_game,
        "sample_size_budget": {
            "intended_families": 2,
            "observed_families": len(rows),
            "eligible_families": len(rows) if authentic else 0,
            "excluded_families": 0,
            "censored_families": sum(bool(row.get("censoring")) for row in rows),
            "effective_blocks": len(rows) if authentic else 0,
            "roles": {"wa30": "known_public_game", "lf52": "known_public_game"},
            "exposure": "previously_cleared_public_games",
            "arms": ["adapter_withheld"],
            "independent_unit": "public_game",
            "action_limit_per_game": 128,
        },
        "inference_substrate": "host_cpu_historical_byte_reduction_no_model_call",
        "inference_substrate_class": "no_model_load",
        "planned_inference_substrate_class": "no_model_load",
        "MODEL_SPECS": [],
        "planned_MODEL_SPECS": [],
        "model_specs": [{"no_model_invoked": True, "historical_model": "unsloth/Qwen3.8-27B-GGUF"}],
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
        "phase_spans": list(spans or []),
        "duration_s": duration_s,
        "random_seed": seed,
        "reproducibility_checksum": checksum,
        "source_artifact_hashes": source_hashes,
        "preconditions_checked": checks,
        "validation_receipts": receipts,
        "repository_health": {
            "full_suite_debt": [
                row
                for row in receipts
                if row.get("scope") == "global_debt" and row.get("exit_code") != 0
            ]
        },
        "frozen_validation_scope": SCOPE,
        "verifier_is_oracle": True,
        "arc_evidence_ready_score": ready,
        "solve_provenance": {
            "historical": "live_agent_self_discovery",
            "current_fixtures": "development_proxy",
        },
        "historical_model_provenance": recovered.get("historical_model_provenance"),
        "registry_precheck": {
            "games": recovered.get("registry_precheck", {}),
            "adapter_withheld": True,
            "withheld_inputs": [
                "per_game_adapter",
                "stored_route",
                "banked_solution",
                "game_source",
                "expert_goal",
                "offline_bfs_truth",
            ],
        },
        "supervisor_outcomes": supervisor,
        "prior_verdict": recovered.get("prior_verdict"),
        "prior_result_immutable": True,
        "new_solve_credit": False,
        "historical_generalization_report": {
            "observed_level_ups": sum(row["observed_level_ups"] for row in rows),
            "accepted_engines": sum(row["accepted_engines"] for row in rows),
            "goal_recall": "unknown",
            "hidden_game_solve_rate": None,
            "exp10015_goal_probe": "closed_null",
            "search_budget": "closed_null",
            "recent_explorer": "closed_null",
        },
        "terminal_reader_receipts_path": str(RAW / "validation_receipts.json"),
        "effective_agent_backend": {
            "requested": "codex",
            "effective": os.environ.get("CODEX_AGENT_BACKEND"),
            "session_id": os.environ.get("CODEX_SESSION_ID"),
        },
    }
    artifact["field_principles"] = {
        name: (
            gates[name]["principle"]
            if name in gates
            else "Measured evidence bounds this claim and prevents invalid downstream use."
        )
        for name in (*artifact.keys(), *GATE_NAMES)
    }
    return artifact


def _validation(
    root: Path, private: Path, started: float, suite_debt: Json | None = None
) -> list[Json]:
    """Run the frozen affected checks, full Python suite, and applicable CPU E2E."""
    from carnot.reporting.experiment_7303_validation_scope import (
        CommandSpec,
        build_scoped_commands,
        run_commands,
    )

    private.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="exp7722-validation-", dir="/tmp") as scratch_dir:
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
        pytest = str(root / ".venv/bin/pytest")
        common = ("-n", "0", "-o", "addopts=", "--no-cov")
        e2e = {
            "e2e_009": ("tests/python/test_arc_induction_state_persistence.py",),
            "e2e_011": ("tests/python/test_arc_decision_telemetry.py",),
            "e2e_013": (
                "tests/python/test_arc_decision_telemetry.py",
                "tests/python/test_experiment_7491_e6_timed_live_profile.py",
                "tests/python/test_experiment_7531_b2_induction_gate_measurement.py",
                "tests/python/test_semif_arc_readout_eval.py",
            ),
        }
        for name, paths in e2e.items():
            commands.append(
                CommandSpec(
                    name,
                    (pytest, *common, f"--basetemp={scratch / name}", *paths, "-q"),
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
        if suite_debt is None:
            commands.append(
                CommandSpec(
                    "full_python_suite",
                    (pytest, "tests/python", "-q"),
                    "global_debt",
                    1800.0,
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
        if suite_debt is not None:
            receipts.append(dict(suite_debt))
    progress(started, "validation", "after", completed=len(receipts))
    return receipts


def cold_read(path: Path, root: Path = ROOT) -> Json:
    """Reopen original bytes in a fresh interpreter and compare candidate rows."""
    candidate = json.loads(path.read_text(encoding="utf-8"))
    recovered = recover_history(root, root / OLD_RAW)
    if candidate["rows"] != [
        {**row, "supervisor": recovered["supervisor"]["per_game"].get(row["game"])}
        for row in recovered["rows"]
    ]:
        raise ValueError("cold_historical_rows_mismatch")
    if recovered["failed_checks"]:
        raise ValueError("cold_historical_inputs_missing")
    return {
        "joined_actions": recovered["joined_actions"],
        "model_calls": recovered["historical_model_provenance"]["calls"],
        "source_hashes": recovered["hashes"],
    }


def _terminal(root: Path, candidate: Path, private: Path, started: float) -> list[Json]:
    """Run all terminal readers on one exact candidate and hash their logs."""
    from carnot.reporting.experiment_7303_validation_scope import CommandSpec, run_commands

    python = str(root / ".venv/bin/python")
    commands = [
        CommandSpec(
            "cold_reduction",
            (
                python,
                "-u",
                "scripts/experiments/experiment_7722_v672_arc_evidence_recovery.py",
                "--cold-read",
                str(candidate),
            ),
            "exact_candidate",
            300.0,
        ),
        CommandSpec(
            "adversarial_verify",
            (python, "-u", "scripts/adversarial_verify.py", str(candidate)),
            "exact_candidate",
            300.0,
        ),
        CommandSpec(
            "verdict_row_consistency_strict",
            (python, "-u", "scripts/verdict_row_consistency_lint.py", "--strict", str(candidate)),
            "exact_candidate",
            300.0,
        ),
    ]
    progress(started, "terminal_readers", "before", commands=len(commands))
    receipts = run_commands(
        root,
        commands,
        log_dir=private / "terminal_logs",
        extra_env={"JAX_PLATFORMS": "cpu"},
        heartbeat_s=45.0,
    )
    progress(started, "terminal_readers", "after", completed=len(receipts))
    return receipts


def run_experiment(
    root: Path, run_date: str, output: Path, *, suite_debt_path: Path | None = None
) -> Json:
    """Freeze scope, measure old bytes, validate, cold replay, and publish once."""
    started = time.monotonic()
    root = root.resolve()
    suite_debt = read_suite_debt(suite_debt_path, root) if suite_debt_path else None
    progress(started, "start", "before", root=root, completed=0)
    private = root / RAW
    private.mkdir(parents=True, exist_ok=True)
    atomic_json(private / "frozen_scope.json", SCOPE)
    spans: list[Json] = []

    phase_start = time.monotonic()
    progress(started, "preconditions", "before", completed=0)
    checks, current_hashes = preflight(root)
    try:
        recovered = recover_history(root, root / OLD_RAW)
    except (ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
        check = {
            "check": "historical_schema_valid",
            "upstream": "experiment_7709",
            "path": str(root / OLD_RAW),
            "field": "raw_rows_and_joins",
            "operator": "==",
            "expected": "valid_original_schema",
            "observed": f"{type(exc).__name__}: {exc}",
            "passed": False,
        }
        recovered = {
            "checks": [check],
            "failed_checks": [check],
            "rows": [],
            "hashes": {
                "valid_producers": {},
                "flagged_historical_evidence": {},
                "pre_gate_receipts": {},
                "missing_custody": [str(root / OLD_RAW)],
            },
        }
    spans.append(
        {
            "phase": "preconditions_and_reduction",
            "start_monotonic": phase_start,
            "end_monotonic": time.monotonic(),
            "duration_s": time.monotonic() - phase_start,
            "heartbeat_timestamps": [phase_start, time.monotonic()],
            "completed_units": len(recovered.get("rows", [])),
            "checkpoint": str(private / "recovery_checkpoint.json"),
        }
    )
    atomic_json(private / "recovery_checkpoint.json", recovered)
    progress(
        started,
        "preconditions",
        "after",
        completed=len(recovered.get("rows", [])),
        failed_checks=len(recovered.get("failed_checks", [])),
    )

    failed_preflight = any(row.get("passed") is not True for row in checks)
    missing_history = bool(recovered.get("failed_checks"))
    receipts: list[Json] = []
    if not failed_preflight and not missing_history:
        phase_start = time.monotonic()
        receipts = _validation(root, private, started, suite_debt)
        spans.append(
            {
                "phase": "validation",
                "start_monotonic": phase_start,
                "end_monotonic": time.monotonic(),
                "duration_s": time.monotonic() - phase_start,
                "heartbeat_timestamps": [phase_start, time.monotonic()],
                "completed_units": len(receipts),
                "checkpoint": str(private / "validation_receipts_preterminal.json"),
            }
        )
        atomic_json(private / "validation_receipts_preterminal.json", receipts)
    artifact = build_artifact(
        recovered,
        receipts,
        run_date,
        time.monotonic() - started,
        current_checks=checks,
        current_hashes=current_hashes,
        spans=spans,
    )
    candidate = private / "terminal_candidate.json"
    atomic_json(candidate, artifact)
    if artifact["verdict_class"] != "blocked":
        terminal = _terminal(root, candidate, private, started)
        atomic_json(
            private / "validation_receipts.json",
            {
                "candidate_sha256": sha256_file(candidate),
                "terminal_reader_receipts": terminal,
            },
        )
        if any(row["exit_code"] != 0 for row in terminal):
            artifact = build_artifact(
                recovered,
                [*receipts, *terminal],
                run_date,
                time.monotonic() - started,
                current_checks=checks,
                current_hashes=current_hashes,
                spans=spans,
            )
            artifact["honest_verdict"] = "complete_disqualified_terminal_reader"
            artifact["verdict_class"] = "disqualified"
            artifact["arc_evidence_ready_score"] = 0
            artifact["flagged_adversarial"] = any(
                row["name"] == "adversarial_verify" and row["exit_code"] != 0 for row in terminal
            )
            atomic_json(candidate, artifact)
    else:
        atomic_json(
            private / "validation_receipts.json",
            {
                "candidate_sha256": sha256_file(candidate),
                "terminal_reader_receipts": [],
                "reason": "blocked_input_bytes",
            },
        )
    atomic_json(output, artifact)
    progress(
        started,
        "publication",
        "after",
        completed=len(artifact["rows"]),
        verdict=artifact["honest_verdict"],
    )
    return artifact


def main(argv: list[str] | None = None) -> int:
    """Keep the CLI limited to one date, a cold replay mode, and the owned run."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default="20260926")
    parser.add_argument("--cold-read", type=Path)
    parser.add_argument("--reuse-full-suite-receipt", type=Path)
    args = parser.parse_args(argv)
    if args.cold_read:
        print(json.dumps(cold_read(args.cold_read), sort_keys=True), flush=True)
        return 0
    if args.reuse_full_suite_receipt:
        run_experiment(
            ROOT, args.date, ROOT / RESULT, suite_debt_path=args.reuse_full_suite_receipt
        )
    else:
        run_experiment(ROOT, args.date, ROOT / RESULT)
    return 0
