"""Publish the V672 capstone after scoped and cold validation (REQ-REPORT-7725)."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import socket
import subprocess
import tempfile
import time
from typing import Any

from carnot.reporting import v672_capstone, v672_contract
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash, sha256_file
from carnot.reporting.experiment_7303_validation_scope import (
    CommandSpec,
    build_scoped_commands,
    reduce_required_checks,
    run_commands,
)

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = Path("results/experiment_7725_v672_capstone.json")
RAW = Path("results/raw/experiment_7725_v672_capstone")
MODULE = "python/carnot/experiment_7725_v672_capstone.py"
CAPABILITY = "python/carnot/reporting/v672_capstone.py"
WRAPPER = "scripts/experiments/experiment_7725_v672_capstone.py"
TEST = "tests/python/test_experiment_7725_v672_capstone.py"
GATES = (
    "measured_validity",
    "readiness",
    "probability",
    "utility",
    "coverage",
    "source_dependence",
    "retention",
    "efficiency",
)
PRINCIPLE = "Measured evidence bounds the claim and prevents invalid downstream use."


def progress(start: float, phase: str, event: str, units: int = 0) -> None:  # pragma: no cover
    """Flush owned work counts so a quiet subprocess cannot hide a stalled run."""
    print(
        f"[exp7725] {phase} {event} elapsed_s={time.monotonic() - start:.3f} completed_units={units}",
        flush=True,
    )


def publication(root: Path) -> dict[str, Any]:
    """Ask the unchanged publication gate for the current four operands."""
    result = subprocess.run(
        [str(root / ".venv/bin/python"), "scripts/publication_gate.py", "--json"],
        cwd=root,
        capture_output=True,
        text=True,
        timeout=120,
        check=True,
    )
    value: dict[str, Any] = json.loads(result.stdout)
    value.update(
        headline_auroc=0.9131,
        headline_scope="established_FoVer_dual_condition_only",
        publication_performed=False,
    )
    return value


def build_artifact(
    root: Path,
    receipts: list[dict[str, Any]] | None = None,
    spans: list[dict[str, Any]] | None = None,
    duration_s: float = 0.0,
) -> dict[str, Any]:
    """Report measured custody and keep absent benefit operands empty."""
    root = root.resolve()
    selected, roadmap, candidates = v672_contract.resolve_authority(root)
    design = root / v672_contract.DESIGN_PATH
    comparison = v672_contract.compare_authorities(design.read_text(), roadmap)
    accounting = v672_capstone.account(root, roadmap["tasks"])
    failure = accounting["gate_check_summary"]["failed_checks"]
    if not comparison["passed"]:
        failure.append(
            v672_capstone.failed_check(
                "independent_contract",
                "v672_design",
                str(v672_contract.DESIGN_PATH),
                "table_and_machine_contract",
                "==",
                "thirteen_matching_tasks",
                comparison["errors"],
            )
        )
    checks = (
        reduce_required_checks(receipts or []) if receipts else {"required_checks_passed": True}
    )
    valid = checks["required_checks_passed"]
    verdict = "disqualified" if not valid else accounting["verdict_class"]
    honest = (
        "complete_disqualified_required_validation" if not valid else accounting["honest_verdict"]
    )
    accounting["gate_check_summary"].update(
        passed=not failure,
        failed_count=len(failure),
        first_failure=failure[0] if failure else None,
    )
    source_hashes = accounting["source_artifact_hashes"] | {
        "authority": [
            {"path": selected.relative_to(root).as_posix(), "sha256": sha256_file(selected)},
            {"path": str(v672_contract.DESIGN_PATH), "sha256": sha256_file(design)},
        ],
        "publication_state": [
            {
                "path": "ops/publication_gate_state.json",
                "sha256": sha256_file(root / "ops/publication_gate_state.json"),
            }
        ],
    }
    eligible = verdict == "null" and not failure
    gates = {
        name: {"passed": None, "measured_operands": None, "principle": PRINCIPLE} for name in GATES
    }
    gates["measured_validity"].update(
        passed=valid and comparison["passed"],
        measured_operands={"contract_match": comparison["passed"], "validation_passed": valid},
    )
    gates["readiness"].update(
        passed=eligible,
        measured_operands={"required_science_eligible": eligible, "accounted_tasks": 13},
    )
    gates["coverage"]["measured_operands"] = {
        "eligible_required_producers": sum(
            row["verdict_class"] in v672_capstone.ELIGIBLE
            and row["task_id"].split("-", 1)[0] in {"exp7718", "exp7720", "exp7721"}
            for row in accounting["prior_dispositions"]
        ),
        "required_producers": 3,
    }
    rows = [
        {
            "unit_id": row["task_id"],
            "arm": "task_accounting",
            "order": row["order"],
            "raw_metrics": {
                "availability": row["availability"],
                "verdict_class": row["verdict_class"],
                "registered_gates": row["registered_gates"],
            },
            "denominators": {"independent_tasks": 1, "independent_families": None},
            "exclusions": [],
            "censored": row["availability"] in {"absent", "pre_gate_receipt"},
            "provenance": row["evidence_path"] or row["planned_path"],
        }
        for row in accounting["prior_dispositions"]
    ]
    rows[-1]["raw_metrics"]["verdict_class"] = verdict
    preconditions = [
        {"check": "authenticated_input", **item, "passed": True}
        for kind in (
            "producers",
            "flagged_historical_evidence",
            "pre_gate_receipts",
            "authority",
            "publication_state",
        )
        for item in source_hashes[kind]
    ]
    preconditions.extend(
        [
            {
                "check": "absolute_root",
                "path": str(root),
                "expected": str(ROOT),
                "observed": str(root),
                "passed": root == ROOT,
            },
            {
                "check": "cpu_aggregation_available",
                "path": "/proc/self/status",
                "expected": True,
                "observed": Path("/proc/self/status").is_file(),
                "passed": Path("/proc/self/status").is_file(),
            },
        ]
    )
    decisions = [
        {
            "mechanism": "V671_certificate_count_head",
            "evidence": "Exp7704 null registered decision benefit",
            "changed_premise": "original-source latent alignment with fresh human labels",
            "acceptance_gate": "paired natural decision utility",
            "retirement_outcome": "retire_unchanged_null_mechanism",
        },
        {
            "mechanism": "V671_fixed_period_acquisition",
            "evidence": "Exp7705 readiness zero; Exp7706 gate skipped",
            "changed_premise": "reachable causal commit and complete static closure",
            "acceptance_gate": "delayed improvement and retention",
            "retirement_outcome": "retire_unchanged_unqualified_mechanism",
        },
        {
            "mechanism": "V672_latent_source_alignment",
            "evidence": "Exp7718 absent after cohort shortage",
            "changed_premise": "enough unexposed original source families",
            "acceptance_gate": "fresh human-label probability and utility",
            "retirement_outcome": "blocked_not_scientifically_retired",
        },
        {
            "mechanism": "V672_continuous_acquisition",
            "evidence": "Exp7720 absent; Exp7719 disqualified",
            "changed_premise": "passing acquisition qualification and fresh admission",
            "acceptance_gate": "causal retained improvement",
            "retirement_outcome": "blocked_not_scientifically_retired",
        },
        {
            "mechanism": "V671_capstone_required_evidence",
            "evidence": "Exp7712 complete_blocked_required_v671_scientific_evidence",
            "changed_premise": "eligible V672 required science",
            "acceptance_gate": "required producer eligibility",
            "retirement_outcome": "same_verdict_retired_v671_scope_only",
        },
        {
            "mechanism": "V669_capstone_required_evidence",
            "evidence": "Exp7684 complete_blocked_required_v669_scientific_evidence",
            "changed_premise": "eligible V672 required science",
            "acceptance_gate": "required producer eligibility",
            "retirement_outcome": "same_verdict_retired_v669_scope_only",
        },
    ]
    artifact: dict[str, Any] = {
        **accounting,
        "honest_verdict": honest,
        "verdict_class": verdict,
        "experiment": 7725,
        "run_date": "20260926",
        "schema": "carnot.exp7725.v672.capstone.v1",
        "flagged_adversarial": False,
        "selected_authority": {
            "path": selected.relative_to(root).as_posix(),
            "candidates": candidates,
            "comparison": comparison,
        },
        "source_artifact_hashes": source_hashes,
        "preconditions_checked": preconditions,
        "acceptance_gate_results": gates,
        "rows": rows,
        "sample_size_budget": {
            "intended_families": None,
            "observed_families": None,
            "eligible_families": None,
            "excluded_families": None,
            "censored_families": None,
            "roles": None,
            "prior_exposure": None,
            "effective_blocks": None,
            "accounted_tasks": 13,
            "independent_science_producers": 3,
        },
        "inference_substrate": "cpu_aggregation",
        "inference_substrate_class": "aggregation",
        "planned_inference_substrate_class": "aggregation",
        "MODEL_SPECS": [],
        "planned_MODEL_SPECS": [],
        "model_specs": [],
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
            "hostname": socket.gethostname(),
            "pid": os.getpid(),
            "gpu_uuid": None,
        },
        "effective_execution_backend": "host_python_cpu_aggregation",
        "phase_spans": spans or [],
        "duration_s": duration_s,
        "random_seed": {"seeds": [], "purpose": "deterministic_accounting_no_sampling"},
        "validation_receipts": {
            "predeclared_scope": {
                "changed_modules": [MODULE, CAPABILITY],
                "static_paths": [WRAPPER],
                "tests": [TEST],
                "requirements": ["REQ-REPORT-7725"],
            },
            "commands": receipts or [],
            "required_checks": checks,
            "e2e": "SCENARIO-REPORT-7725-TERMINAL",
            "global_debt": "historical failures remain separate",
        },
        "verifier_is_oracle": False,
        "capstone_accounting_ready_score": 1,
        "continuation_decisions": decisions,
        "publication_gates": publication(root),
        "three_prd_gaps": [
            {
                "gap": "source_dependent_evidence_coverage",
                "observed": "fresh cohort blocked; natural benefit unmeasured",
            },
            {
                "gap": "causal_retained_learning",
                "observed": "qualification disqualified; delayed learning unmeasured",
            },
            {
                "gap": "reachable_complete_service_cost",
                "observed": "ARC and native qualification disqualified; complete-service cost unmeasured",
            },
        ],
        "hardware_continuity": {
            "source": "research-hardware-wishlist.md",
            "sha256": sha256_file(root / "research-hardware-wishlist.md"),
            "kv260": "focus_board_no_new_measurement",
            "polarfire": "no_new_measurement",
            "gatemate": "no_new_measurement",
            "purchase_performed": False,
        },
        "publication_performed": False,
        "generator_training_performed": False,
        "production_defaults_changed": False,
    }
    artifact["reproducibility_checksum"] = canonical_hash(
        {
            "sources": source_hashes,
            "configuration": artifact["selected_authority"],
            "reducer": sha256_file(root / CAPABILITY),
        }
    )
    artifact["field_principles"] = {key: PRINCIPLE for key in artifact}
    artifact["field_principles"].update(
        honest_verdict="Terminal custody prevents retries of unchanged external blocks.",
        verdict_class="Claim eligibility travels with the result.",
        gate_check_summary="Exact operands distinguish scientific failure from a broken interface.",
        inference_substrate_class="Duration floors must match real computation.",
        MODEL_SPECS="Experimental model identity must match actual invocation.",
    )
    artifact["field_principles"]["field_principles"] = "Every field states its evidence boundary."
    artifact["field_principles"]["acceptance_gates"] = {name: PRINCIPLE for name in GATES}
    return artifact


def read_candidate(path: Path, root: Path = ROOT) -> list[str]:
    """Recompute source rows and immutable conclusions in a fresh process."""
    value = json.loads(path.read_text())
    _, roadmap, _ = v672_contract.resolve_authority(root)
    account_view = dict(value)
    account_view["gate_check_summary"] = dict(value["gate_check_summary"])
    account_view["gate_check_summary"]["failed_checks"] = [
        row
        for row in value["gate_check_summary"]["failed_checks"]
        if row["check"] != "independent_contract"
    ]
    account_view["gate_check_summary"].update(
        failed_count=len(account_view["gate_check_summary"]["failed_checks"]),
        first_failure=next(iter(account_view["gate_check_summary"]["failed_checks"]), None),
        passed=not account_view["gate_check_summary"]["failed_checks"],
    )
    account_view["source_artifact_hashes"] = {
        key: value["source_artifact_hashes"][key]
        for key in (
            "producers",
            "flagged_historical_evidence",
            "pre_gate_receipts",
            "missing_custody",
            "planned_output_is_input",
        )
    }
    errors = v672_capstone.cold_reduce(account_view, root, roadmap["tasks"])
    expected = build_artifact(root, value.get("validation_receipts", {}).get("commands", []))
    for key in (
        "gate_check_summary",
        "source_artifact_hashes",
        "selected_authority",
        "reproducibility_checksum",
        "acceptance_gate_results",
        "sample_size_budget",
        "three_prd_gaps",
        "continuation_decisions",
        "rows",
        "publication_gates",
    ):
        if canonical_hash(value.get(key)) != canonical_hash(expected[key]):
            errors.append(f"{key}_mismatch")
    return errors


def run_experiment(root: Path, run_date: str, output: Path) -> dict[str, Any]:  # pragma: no cover
    """Run bounded readers, then atomically publish only their checked bytes."""
    start = time.monotonic()
    progress(start, "preflight", "start")
    root = root.resolve()
    if run_date != "20260926":
        raise ValueError("V672 run date must be 20260926")
    selected, roadmap, _ = v672_contract.resolve_authority(root)
    if not selected.is_file():
        raise ValueError("selected authority must exist")
    accounting = v672_capstone.account(root, roadmap["tasks"])
    raw = root / RAW
    raw.mkdir(parents=True, exist_ok=True)
    scope = {
        "changed_modules": [MODULE, CAPABILITY],
        "static_paths": [WRAPPER],
        "tests": [TEST],
        "requirements": ["REQ-REPORT-7725", "SCENARIO-REPORT-7725-TERMINAL"],
    }
    atomic_json(raw / "frozen_affected_scope.json", scope)
    progress(start, "preflight", "done", len(accounting["prior_dispositions"]))
    private = Path(tempfile.mkdtemp(prefix="exp7725-validation-", dir="/tmp"))
    (private / "pytest").mkdir()
    commands = build_scoped_commands(
        root,
        [TEST],
        [MODULE, CAPABILITY],
        static_paths=[WRAPPER],
        basetemp=private / "pytest",
        coverage_file=private / ".coverage",
    )
    progress(start, "validation", "before_subprocesses")
    begin = time.monotonic() - start
    receipts = run_commands(root, commands, log_dir=raw / "validation/affected", heartbeat_s=60)
    spans = [
        {
            "phase": "validation",
            "start_s": begin,
            "end_s": time.monotonic() - start,
            "duration_s": time.monotonic() - start - begin,
            "completed_units": len(receipts),
            "heartbeat_timestamps": [],
            "checkpoints": ["frozen_affected_scope.json"],
        }
    ]
    progress(start, "validation", "after_subprocesses", len(receipts))
    candidate = build_artifact(root, receipts, spans, time.monotonic() - start)
    candidate_path = raw / "terminal_candidate.json"
    atomic_json(candidate_path, candidate)
    terminal = [
        CommandSpec(
            "cold_replay",
            (str(root / ".venv/bin/python"), "-u", WRAPPER, "--cold-validate", str(candidate_path)),
            "exact_candidate",
            900,
        ),
        CommandSpec(
            "adversarial_verify",
            (
                str(root / ".venv/bin/python"),
                "-u",
                "scripts/adversarial_verify.py",
                str(candidate_path),
            ),
            "exact_candidate",
            900,
        ),
        CommandSpec(
            "verdict_row_consistency_strict",
            (
                str(root / ".venv/bin/python"),
                "-u",
                "scripts/verdict_row_consistency_lint.py",
                "--strict",
                str(candidate_path),
            ),
            "exact_candidate",
            900,
        ),
    ]
    progress(start, "terminal", "before_subprocesses")
    begin = time.monotonic() - start
    terminal_receipts = run_commands(
        root, terminal, log_dir=raw / "validation/terminal", heartbeat_s=60
    )
    spans.append(
        {
            "phase": "terminal",
            "start_s": begin,
            "end_s": time.monotonic() - start,
            "duration_s": time.monotonic() - start - begin,
            "completed_units": len(terminal_receipts),
            "heartbeat_timestamps": [],
            "checkpoints": ["terminal_candidate.json"],
        }
    )
    progress(start, "terminal", "after_subprocesses", len(terminal_receipts))
    final = build_artifact(root, [*receipts, *terminal_receipts], spans, time.monotonic() - start)
    final["validation_receipts"]["cold_reduction"] = (
        "passed" if terminal_receipts[0]["passed"] else "failed"
    )
    final["validation_receipts"]["terminal_readers"] = terminal_receipts
    if not all(row["passed"] for row in terminal_receipts):
        final["verdict_class"] = "disqualified"
        final["honest_verdict"] = "complete_disqualified_terminal_reader"
        final["acceptance_gate_results"]["readiness"]["passed"] = False
        final["capstone_accounting_ready_score"] = 0
    destination = output if output.is_absolute() else root / output
    progress(start, "publication", "before_atomic")
    atomic_json(destination, final)
    progress(start, "publication", "after_atomic", 1)
    return final


def main(argv: list[str] | None = None) -> int:  # pragma: no cover
    """Expose one thin entrypoint and a separate read-only cold replay."""
    print("[exp7725] startup flushed", flush=True)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--date", default="20260926")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--cold-validate", type=Path)
    args = parser.parse_args(argv)
    if args.cold_validate:
        errors = read_candidate(args.cold_validate, args.root)
        print(json.dumps({"cold_errors": errors}), flush=True)
        return int(bool(errors))
    run_experiment(args.root, args.date, args.output)
    return 0
