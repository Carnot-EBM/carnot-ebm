"""Run V676 scored ARC qualification with the frozen affected validation scope."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import socket
import time
from typing import Any

from scripts.experiments import experiment_7763_v675_arc_runner_qualification as prior

from carnot.experiment_7776_v676_arc_runner_qualification import (
    gate_decision,
    reduce_probe_evidence,
)
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash, sha256_file
from carnot.reporting.experiment_7303_validation_scope import (
    CommandSpec,
    build_scoped_commands,
    run_commands,
)

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "results/raw/experiment_7776_v676_arc_runner_qualification"
RESULT = ROOT / "results/experiment_7776_v676_arc_runner_qualification.json"
SCOPE = RAW / "frozen_affected_scope.json"
PRIVATE = Path("/tmp/exp7776-v676-validation")
OLD_RESULT = ROOT / "results/experiment_7763_v675_arc_runner_qualification.json"
OLD_PANEL = (
    ROOT / "results/raw/experiment_7763_v675_arc_runner_qualification/arc_panel_manifest.json"
)


def progress(start: float, phase: str, event: str, units: int = 0, **detail: Any) -> None:
    """Show phase boundaries and completed work while a supervisor owns children."""
    print(
        f"[exp7776] phase={phase} event={event} elapsed_s={time.monotonic() - start:.2f} "
        f"completed_units={units} {detail}",
        flush=True,
    )


def _configure_reused_runner() -> None:
    """Point the qualified V675 helpers at only this run's owned files."""
    prior.RAW = RAW
    prior.SCOPE = SCOPE
    prior.PRIVATE = PRIVATE


def preflight(start: float) -> tuple[list[dict[str, Any]], dict[str, Any], Any, dict[str, Any]]:
    """Bind the old disqualification and inspect every current input before action."""
    _configure_reused_runner()
    checks, hashes, arcade, context = prior.preflight(start)
    old = json.loads(OLD_RESULT.read_text()) if OLD_RESULT.is_file() else {}
    checks.extend(
        [
            prior.check(
                OLD_RESULT,
                "honest_verdict",
                "complete_disqualified_required_runner_validation",
                old.get("honest_verdict"),
                "exp7763_historical_diagnostic",
            ),
            prior.check(
                OLD_RESULT,
                "organic_runner_ready_score",
                0,
                old.get("organic_runner_ready_score"),
                "exp7763_historical_diagnostic",
            ),
            prior.check(
                ROOT / "openspec/capabilities/research-reporting/spec.md",
                "REQ-REPORT-7776",
                True,
                "REQ-REPORT-7776"
                in (ROOT / "openspec/capabilities/research-reporting/spec.md").read_text(),
                "v676_spec",
            ),
            prior.check(
                ROOT / "openspec/capabilities/arc-world-model-trust-energy/spec.md",
                "REQ-ARC-WMTE-7776",
                True,
                "REQ-ARC-WMTE-7776"
                in (
                    ROOT / "openspec/capabilities/arc-world-model-trust-energy/spec.md"
                ).read_text(),
                "v676_spec",
            ),
            prior.check(
                SCOPE,
                "readable_nonempty",
                True,
                SCOPE.is_file() and SCOPE.stat().st_size > 0,
                "v676_scope",
            ),
            prior.check(
                OLD_PANEL,
                "readable_nonempty",
                True,
                OLD_PANEL.is_file() and OLD_PANEL.stat().st_size > 0,
                "exp7763_panel",
            ),
        ]
    )
    for relative, fields in (
        (
            "results/experiment_7763_v675_arc_runner_qualification.json",
            ["honest_verdict", "organic_runner_ready_score"],
        ),
        (
            "results/raw/experiment_7763_v675_arc_runner_qualification/arc_panel_manifest.json",
            ["games", "seeds", "arms", "controls", "rows"],
        ),
        ("python/carnot/experiment_7776_v676_arc_runner_qualification.py", []),
        ("scripts/experiments/experiment_7776_v676_arc_runner_qualification.py", []),
        (
            "results/raw/experiment_7776_v676_arc_runner_qualification/frozen_affected_scope.json",
            [],
        ),
    ):
        path = ROOT / relative
        hashes[relative] = {
            "sha256": sha256_file(path) if path.is_file() else None,
            "date": "20260927" if path.is_file() else None,
            "imported_fields": fields,
            "eligible": path.is_file(),
        }
    snapshot_dir = RAW / "input_snapshots"
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    for relative in (
        "CLAUDE.md",
        "CODEX.md",
        "research-program.md",
        "ops/e2e-test-plan.md",
        "openspec/capabilities/research-reporting/spec.md",
        "openspec/capabilities/arc-world-model-trust-energy/spec.md",
    ):
        source = ROOT / relative
        target = snapshot_dir / relative.replace("/", "__")
        shutil.copyfile(source, target)
        hashes[relative]["snapshot_path"] = str(target.relative_to(ROOT))
        hashes[relative]["sha256"] = sha256_file(target)
    context["resource"]["historical_v675_verdict"] = old.get("honest_verdict")
    context["resource"]["historical_full_suite_collection_open"] = True
    context["resource"]["missing_downstream_producer"] = (
        "results/experiment_7749_v674_arc_generalization.json"
    )
    context["resource"]["distinct_conductor_pre_gate"] = (
        "results/experiment_7749_arc_organic_measurement.json"
    )
    progress(
        start,
        "preconditions_v676",
        "after",
        len(checks),
        failed=sum(not row["passed"] for row in checks),
    )
    return checks, hashes, arcade, context


def frozen_panel(context: dict[str, Any]) -> tuple[dict[str, Any], Path]:
    """Use the old panel's order and bounds before any probe can run."""
    panel = prior.panel(context)
    old = json.loads(OLD_PANEL.read_text())
    for key in (
        "games",
        "seeds",
        "arms",
        "controls",
        "round_robin_order",
        "max_actions_charged",
        "max_seconds_per_episode",
        "reset_charging_conventions",
        "rows",
    ):
        if panel[key] != old[key]:
            raise ValueError(f"old_panel_changed:{key}")
    panel["schema"] = "carnot.exp7776.frozen_arc_panel.v1"
    panel["source_panel_sha256"] = sha256_file(OLD_PANEL)
    path = RAW / "arc_panel_manifest.json"
    if path.exists() and json.loads(path.read_text()) != panel:
        raise ValueError("frozen_panel_changed")
    atomic_json(path, panel)
    return panel, path


def validation(start: float) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Run the affected closure; keep broad collection as independent debt."""
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
    commands.append(
        CommandSpec(
            "child_kill",
            (
                pytest,
                *common,
                f"--basetemp={base / 'child_kill'}",
                "tests/python/test_experiment_7776_v676_arc_runner_qualification.py::test_scenario_report_7776_child_owned_timeout_and_private_basetemp",
                "-q",
            ),
            "owned_timeout_and_private_pytest",
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
    diagnostic_path = RAW / "repository_diagnostic/receipt.json"
    if diagnostic_path.is_file():
        diagnostic = json.loads(diagnostic_path.read_text())
        if sha256_file(ROOT / diagnostic["log_path"]) != diagnostic["log_sha256"]:
            raise ValueError("repository_diagnostic_log_changed")
        progress(start, "repository_diagnostic", "reused", 1, exit=diagnostic["exit_code"])
    else:
        diagnostic = run_commands(
            ROOT,
            [
                CommandSpec(
                    "repository_collection_diagnostic",
                    (
                        pytest,
                        *common,
                        f"--basetemp={base / 'repository_diagnostic'}",
                        "tests/python",
                        "-q",
                    ),
                    "historical_repository_health_only",
                    1800,
                )
            ],
            log_dir=RAW / "repository_diagnostic",
            heartbeat_s=30,
            extra_env={"JAX_PLATFORMS": "cpu", "CARNOT_ARC_DISABLE_INDUCTION": "1"},
        )[0]
        atomic_json(diagnostic_path, diagnostic)
    return receipts, diagnostic


def terminal_readers(start: float, candidate: Path, label: str) -> list[dict[str, Any]]:
    """Run both readers on one exact candidate, retaining exits and log hashes."""
    python = str(ROOT / ".venv/bin/python")
    progress(
        start, "terminal_readers", "before", 0, label=label, candidate_sha256=sha256_file(candidate)
    )
    receipts = run_commands(
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
        log_dir=RAW / f"terminal_{label}",
        heartbeat_s=30,
        extra_env={"JAX_PLATFORMS": "cpu"},
    )
    progress(
        start,
        "terminal_readers",
        "after",
        len(receipts),
        label=label,
        passed=sum(row["passed"] for row in receipts),
    )
    return receipts


def main(argv: list[str] | None = None) -> int:
    """Publish one terminal result after current checks and cold evidence reduction."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", default="20260927")
    parser.add_argument("--cold", type=Path)
    args = parser.parse_args(argv)
    start = time.monotonic()
    progress(start, "startup", "before", 0, pid=os.getpid(), host=socket.gethostname())
    if args.cold:
        progress(start, "cold_reduce", "before")
        raw = json.loads(args.cold.read_text())
        observed = reduce_probe_evidence(raw["schedule"], raw["probes"])
        if observed != raw["summary"]:
            raise ValueError("raw_reduction_mismatch")
        progress(start, "cold_reduce", "after", observed["started"], **observed)
        return 0
    RAW.mkdir(parents=True, exist_ok=True)
    PRIVATE.mkdir(parents=True, exist_ok=True)
    os.environ["CARNOT_ARC_DISABLE_INDUCTION"] = "1"
    progress(start, "model_load", "before", 0, planned_models=0)
    progress(start, "model_load", "after", 0, actual_models=0)
    progress(start, "generation", "before", 0, planned_calls=0)
    progress(start, "generation", "after", 0, actual_calls=0)
    spans: list[dict[str, Any]] = []
    phase = time.monotonic()
    checks, hashes, arcade, context = preflight(start)
    spans.append(
        {
            "phase": "preconditions",
            "duration_s": time.monotonic() - phase,
            "completed_units": len(checks),
        }
    )
    progress(start, "panel_freeze", "before", 0)
    phase = time.monotonic()
    panel, panel_path = frozen_panel(context)
    schedule = panel["rows"]
    spans.append(
        {
            "phase": "panel_freeze",
            "duration_s": time.monotonic() - phase,
            "completed_units": len(schedule),
        }
    )
    progress(start, "panel_freeze", "after", len(schedule), panel_sha256=sha256_file(panel_path))
    probes: list[dict[str, Any]] = []
    phase = time.monotonic()
    if all(row["passed"] for row in checks):
        for game, seed, source in (
            ("fixture", 67500, prior._FixtureArcade()),
            ("r11l", 67501, arcade),
        ):
            for arm in prior.ARMS:
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
                row = prior.run_probe(game, seed, arm, source, 3 if game == "fixture" else 12)
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
    summary = reduce_probe_evidence(schedule, probes)
    raw_path = RAW / "raw_probes.json"
    atomic_json(raw_path, {"schedule": schedule, "probes": probes, "summary": summary})
    spans.append(
        {
            "phase": "scored_probes",
            "duration_s": time.monotonic() - phase,
            "completed_units": len(probes),
        }
    )
    progress(start, "cold_reduce", "before", len(probes))
    phase = time.monotonic()
    cold = run_commands(
        ROOT,
        [
            CommandSpec(
                "cold_reduce",
                (str(ROOT / ".venv/bin/python"), "-u", __file__, "--cold", str(raw_path)),
                "fresh_process_raw_reduction",
                120,
            )
        ],
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
    diagnostic: dict[str, Any] | None = None
    if all(row["passed"] for row in checks):
        phase = time.monotonic()
        current, diagnostic = validation(start)
        receipts.extend(current)
        spans.append(
            {
                "phase": "validation",
                "duration_s": time.monotonic() - phase,
                "completed_units": sum(row["passed"] for row in current),
            }
        )
    progress(start, "artifact", "before", len(probes))
    value = prior.artifact(
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
    value.update(
        {
            "schema": "carnot.exp7776.v676.arc_runner_qualification.v1",
            "experiment_id": 7776,
            "milestone": "2026.09.676",
            "run_date": args.date,
            "organic_runner_ready_score": 0,
            "repository_health": {
                "historical_exp7763_verdict": context["resource"]["historical_v675_verdict"],
                "historical_full_suite_collection_debt_open": True,
                "current_collection_diagnostic": diagnostic,
                "affects_required_checks": False,
            },
            "prior_attempt_receipt": {
                "path": "results/raw/experiment_7776_v676_arc_runner_qualification/prior_attempt_artifact.json",
                "sha256": sha256_file(RAW / "prior_attempt_artifact.json"),
            }
            if (RAW / "prior_attempt_artifact.json").is_file()
            else None,
            "frozen_validation_scope": json.loads(SCOPE.read_text()),
            "arc_panel_manifest_path": str(panel_path.relative_to(ROOT)),
            "arc_panel_manifest_sha256": sha256_file(panel_path),
            "source_artifact_hashes": hashes,
            "raw_reduction": summary,
            "raw_probes_sha256": sha256_file(raw_path),
            "supervisor_outcome_receipts": [
                {
                    "episode_id": row["episode_id"],
                    "observed_goal_firings": sum(
                        action.get("goal_firing") is not None for action in row["actions"]
                    ),
                    "arm_changed_by_supervisor": False,
                    "source": "raw_action_telemetry",
                }
                for row in probes
            ],
            "reset_charging_interpretations": {
                "charged": {row["episode_id"]: row["actions_charged"] for row in probes},
                "uncharged": {
                    row["episode_id"]: row["actions_charged"]
                    - sum(action.get("action") == "RESET" for action in row["actions"])
                    for row in probes
                },
                "budget_interpretation": "charged",
            },
        }
    )
    value["reproducibility_checksum"] = canonical_hash(
        {
            "seeds": list(prior.SEEDS),
            "source_artifact_hashes": hashes,
            "scope_sha256": sha256_file(SCOPE),
            "panel_sha256": sha256_file(panel_path),
            "raw_sha256": sha256_file(raw_path),
            "roles": ["off", "total", "organic"],
            "parameters": panel["controls"],
        }
    )
    value["field_principles"].update(
        {
            "experiment_id": "One current artifact has one unique experiment owner.",
            "repository_health": "Historical collection debt stays visible outside current affected gates.",
            "prior_attempt_receipt": "A failed owned validation remains inspectable after correction.",
            "supervisor_outcome_receipts": "Observed firings do not invent a changed arm.",
            "reset_charging_interpretations": "Both registered RESET charging readings remain inspectable.",
            "source_artifact_hashes": "Exact current inputs and historical eligibility remain distinct.",
            "reproducibility_checksum": "Inputs, roles, seeds, and controls bind the run.",
        }
    )
    value["acceptance_gate_results"]["readiness"] = False
    value["honest_verdict"] = (
        "complete_blocked_preconditions"
        if any(not row["passed"] for row in checks)
        else "complete_null_runner_qualification_pending_terminal"
    )
    progress(start, "artifact", "after", 1)
    candidate = RAW / "terminal_candidate.json"
    atomic_json(candidate, value)
    phase = time.monotonic()
    first = terminal_readers(start, candidate, "preliminary")
    sdk = [row for row in probes if row["claim_scope"] == "adapter_withheld_public"]
    sdk_ok = len(sdk) == 3 and all(
        row["error"] is None
        and row["counts"]["sdk_transitions"] > 0
        and row["policy_entry"]["policy_class"] == "E3AgentPolicy"
        and all(action.get("induction_attempt_count", 0) == 0 for action in row["actions"])
        for row in sdk
    )
    required = value["frozen_validation_scope"]["required_checks"]
    ready, failed_names = gate_decision([*receipts, *first], required, sdk_ok=sdk_ok)
    final_readers = first
    if ready:
        value["organic_runner_ready_score"] = 1
        value["acceptance_gate_results"]["readiness"] = True
        value["acceptance_gate_results"]["validity"] = True
        value["honest_verdict"] = "complete_null_runner_qualified_no_benefit_measurement"
        value["verdict_class"] = "null"
        final_candidate = RAW / "terminal_candidate_ready.json"
        atomic_json(final_candidate, value)
        final_readers = terminal_readers(start, final_candidate, "ready")
        ready, failed_names = gate_decision([*receipts, *final_readers], required, sdk_ok=sdk_ok)
    if not ready:
        value["organic_runner_ready_score"] = 0
        value["acceptance_gate_results"]["readiness"] = False
        value["acceptance_gate_results"]["validity"] = False
        value["verdict_class"] = (
            "blocked" if any(not row["passed"] for row in checks) else "disqualified"
        )
        value["honest_verdict"] = (
            "complete_blocked_preconditions"
            if value["verdict_class"] == "blocked"
            else "complete_disqualified_required_runner_validation"
        )
        for name in failed_names:
            receipt = next(
                (row for row in [*receipts, *final_readers] if row["name"] == name), None
            )
            value["gate_check_summary"].append(
                prior.check(
                    ROOT / receipt["log_path"] if receipt else RAW / "validation",
                    name,
                    0,
                    receipt["exit_code"] if receipt else "missing",
                    "current_validation",
                )
            )
    value["flagged_adversarial"] = any(
        row["name"] == "adversarial_verify" and not row["passed"] for row in final_readers
    )
    value["validation_receipts"] = [*receipts, *final_readers]
    spans.append(
        {
            "phase": "terminal_readers",
            "duration_s": time.monotonic() - phase,
            "completed_units": sum(row["passed"] for row in final_readers),
        }
    )
    atomic_json(
        RAW / "terminal_receipts.json",
        {
            "preliminary_candidate_sha256": sha256_file(candidate),
            "preliminary_commands": first,
            "final_candidate_sha256": sha256_file(RAW / "terminal_candidate_ready.json")
            if (RAW / "terminal_candidate_ready.json").is_file()
            else None,
            "final_commands": final_readers,
        },
    )
    value["phase_spans"] = spans
    value["duration_s"] = time.monotonic() - start
    progress(start, "publish", "before", 0, verdict=value["honest_verdict"])
    atomic_json(RESULT, value)
    progress(start, "publish", "after", 1, result=str(RESULT), sha256=sha256_file(RESULT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
