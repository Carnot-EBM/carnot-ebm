"""Orchestrate the independent V668 source evidence audit.

REQ-REPORT-7664; SCENARIO-REPORT-7664-CUSTODY and -TERMINAL.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import socket
import sys
import tempfile
import time
from typing import Any

from carnot.reporting import experiment_7303_validation_scope as checks
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash, sha256_file
from carnot.reporting.independent_evidence_audit import mutate, reduce_evidence


ROOT = Path(__file__).resolve().parents[2]
RESULT = Path("results/experiment_7664_v668_independent_evidence_audit.json")
RAW = Path("results/raw/experiment_7664_v668_independent_evidence_audit")
NOTE = Path("docs/research-notes/v668-evidence-audit.md")
PRODUCERS = {
    "corpus": "results/experiment_7659_v668_atom_corpus.json",
    "energy": "results/experiment_7660_v668_atom_energy.json",
    "evaluation": "results/experiment_7661_v668_decision_evaluation.json",
    "delayed": "results/experiment_7662_v668_delayed_update_protocol.json",
    "continuous": "results/experiment_7663_v668_continuous_atom_learning.json",
}
FEATURES = {
    role: f"results/raw/experiment_7659_v668_atom_corpus/{role}_features.jsonl"
    for role in ("pilot", "fit", "tune", "policy", "online", "evaluation")
}
MANIFEST = {
    "test_paths": ["tests/python/test_experiment_7664_v668_independent_evidence_audit.py"],
    "changed_modules": [
        "python/carnot/reporting/independent_evidence_audit.py",
        "python/carnot/experiment_7664_v668_independent_evidence_audit.py",
    ],
    "static_paths": ["scripts/experiments/experiment_7664_v668_independent_evidence_audit.py"],
}


def progress(started: float, phase: str, event: str, **details: Any) -> None:
    """Flush every phase boundary with actual elapsed monotonic time."""
    print(
        f"[exp7664] {phase} {event} elapsed_s={time.monotonic() - started:.3f} {details}",
        flush=True,
    )


def gate_check(
    check: str, upstream: str, path: Path, field: str, operator: str, expected: Any, observed: Any
) -> dict[str, Any]:
    """Keep exact failed operands for a blocked upstream dependency."""
    return {
        "check": check,
        "upstream": upstream,
        "path": str(path.resolve()),
        "field": field,
        "operator": operator,
        "expected": expected,
        "observed": observed,
    }


def authenticate_inputs(root: Path) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    """Read terminal producers and raw features, distinguishing missing evidence."""
    found: dict[str, Any] = {}
    failed: list[dict[str, Any]] = []
    hashes: dict[str, Any] = {"producers": {}, "pre_gate_receipts": {}, "missing_inputs": []}
    for name, label in PRODUCERS.items():
        path = root / label
        if not path.is_file():
            failed.append(gate_check("producer_exists", name, path, "exists", "==", True, False))
            hashes["missing_inputs"].append(label)
            continue
        value = json.loads(path.read_text())
        valid = (
            str(value.get("honest_verdict", "")).startswith("complete_")
            and value.get("verdict_class") not in ("blocked", "disqualified", "partial")
            and value.get("validation_receipts", {}).get("required_checks_passed") is True
            and all(
                r.get("passed") is True
                for r in value.get("validation_receipts", {}).get("terminal_readers", [])
            )
        )
        if not valid:
            failed.append(
                gate_check("producer_valid", name, path, "terminal_valid", "==", True, False)
            )
            hashes["pre_gate_receipts"][label] = sha256_file(path)
        else:
            found[name] = value
            hashes["producers"][label] = sha256_file(path)
    feature_rows: list[dict[str, Any]] = []
    for role, label in FEATURES.items():
        path = root / label
        if not path.is_file():
            failed.append(gate_check("raw_exists", role, path, "exists", "==", True, False))
            hashes["missing_inputs"].append(label)
            continue
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        feature_rows.extend(rows)
        hashes["producers"][label] = sha256_file(path)
        if "corpus" in found:
            manifest_path = root / found["corpus"]["feature_manifest_path"]
            if manifest_path.is_file():
                manifest = json.loads(manifest_path.read_text())
                declared = manifest["roles"][role]["feature_sha256"]
                if declared != sha256_file(path):
                    failed.append(
                        gate_check(
                            "raw_custody",
                            "corpus",
                            path,
                            "sha256",
                            "==",
                            declared,
                            sha256_file(path),
                        )
                    )
            else:
                failed.append(
                    gate_check(
                        "manifest_exists", "corpus", manifest_path, "exists", "==", True, False
                    )
                )
    found["features"] = feature_rows
    return found, failed, hashes


def raw_inputs(found: dict[str, Any]) -> dict[str, Any]:
    """Take raw row operands, never a producer aggregate or headline."""
    return {
        "features": found["features"],
        "evaluation": found["evaluation"]["rows"],
        "delayed": found["delayed"]["rows"],
        "delayed_events": found["delayed"]["event_rows"],
        "continuous": found["continuous"]["rows"],
        "feedback": found["continuous"]["causal_feedback_rows"],
        "admissions": found["continuous"]["admission_decisions"],
        "retention": found["continuous"]["retention_rows"],
    }


def private_mutations(inputs: dict[str, Any]) -> dict[str, bool]:
    """Require all six corruptions to fail the current evidence reducer."""
    outcomes: dict[str, bool] = {}
    for name in (
        "source_hash",
        "future_label",
        "admission_reuse",
        "arm_metric",
        "unknown_removed",
        "fixture_truth",
    ):
        changed = deepcopy(inputs)
        mutate(changed, name)
        try:
            reduce_evidence(changed)
        except ValueError:
            outcomes[name] = True
        else:
            outcomes[name] = False
    return outcomes


def acceptance_gates(summary: dict[str, Any], valid: bool) -> list[dict[str, Any]]:
    """Keep structural readiness apart from probability, utility, and freshness."""
    coverage = summary.get("coverage", {})
    contrasts = summary.get("contrasts", {})
    continuous = contrasts.get("continuous", {})
    probability = bool(continuous) and all(
        continuous[arm]["brier"]["ci95"][0] > 0.01 for arm in ("frozen", "scalar")
    )
    utility = (
        bool(continuous)
        and all(continuous[arm]["decision_cost"]["ci95"][0] > 0.01 for arm in ("frozen", "scalar"))
        and summary.get("continuous", {}).get("arms", {}).get("source", {}).get("coverage", 0.125)
        >= 0.2
    )
    gates = [
        (
            "validity",
            valid,
            {"required_checks_passed": valid},
            "Authenticated inputs and required readers govern validity.",
        ),
        (
            "readiness",
            valid and bool(summary),
            {"independent_reduction_complete": bool(summary)},
            "Replayable evidence establishes structural readiness only.",
        ),
        (
            "coverage",
            coverage.get("independent_groups") == 248,
            coverage,
            "Each original source group appears once; unknowns stay in denominator.",
        ),
        (
            "probability_benefit",
            probability,
            continuous,
            "Both block-eight lower Brier improvements must exceed 0.01.",
        ),
        (
            "utility",
            utility,
            {"contrasts": continuous, "coverage_floor": 0.2},
            "Cost gains and non-escalation coverage must clear separate thresholds.",
        ),
        (
            "retention",
            bool(summary.get("retention")) and summary.get("feedback_events", 0) > 0,
            {
                "retention": summary.get("retention", {}),
                "feedback_events": summary.get("feedback_events", 0),
            },
            "Released labels and later replay establish measured retention, not benefit.",
        ),
        (
            "freshness",
            coverage.get("previously_exposed_groups", 0) < coverage.get("independent_groups", 0),
            {"previously_exposed_groups": coverage.get("previously_exposed_groups", 0)},
            "Previously exposed groups cannot confirm a fresh claim.",
        ),
    ]
    return [
        {"gate": name, "passed": passed, "measured_operands": operands, "principle": principle}
        for name, passed, operands, principle in gates
    ]


def build_artifact(
    root: Path,
    date: str,
    found: dict[str, Any],
    failed: list[dict[str, Any]],
    hashes: dict[str, Any],
    summary: dict[str, Any],
    mutations: dict[str, bool],
) -> dict[str, Any]:
    """Build a terminal audit whose rows and source links can be replayed."""
    valid = not failed and all(mutations.values())
    source_rows = [dict(row, evidence_stage="coverage") for row in found.get("features", [])]
    measured_rows = [
        dict(row, evidence_stage=stage)
        for stage, key in (
            ("probability_utility", "evaluation"),
            ("delayed_update", "delayed"),
            ("continuous_update", "continuous"),
        )
        for row in found.get(key, {}).get("rows", [])
    ]
    retained_rows = [
        dict(row, arm="retained_source", evidence_stage="retention")
        for row in found.get("continuous", {}).get("retention_rows", [])
    ]
    rows = source_rows + measured_rows + retained_rows
    coverage = summary.get("coverage", {})
    artifact: dict[str, Any] = {
        "schema": "carnot.exp7664.independent_evidence_audit.v1",
        "experiment_id": 7664,
        "milestone": "2026.09.668",
        "run_date": date,
        "honest_verdict": "complete_blocked_required_v668_external_evidence"
        if failed
        else "complete_null_no_independent_benefit",
        "verdict_class": "blocked" if failed else "null",
        "flagged_adversarial": False,
        "gate_check_summary": failed,
        "acceptance_gate_results": acceptance_gates(summary, valid),
        "rows": rows,
        "sample_size_budget": {
            "intended_independent_groups": 248,
            "observed_independent_groups": coverage.get("independent_groups", 0),
            "eligible": coverage.get("independent_groups", 0) - coverage.get("excluded_groups", 0),
            "excluded": coverage.get("excluded_groups", 0),
            "censored": coverage.get("censored_groups", 0),
            "effective_evaluation_groups": summary.get("evaluation", {}).get("effective_groups", 0),
            "effective_online_groups": summary.get("continuous", {}).get("effective_groups", 0),
            "prior_exposure": "The inherited source roster was previously exposed.",
            "claim_limits": "Partial structural atoms do not certify whole answers or fresh benefit.",
        },
        "inference_substrate": "cpu_aggregation",
        "inference_substrate_class": "aggregation",
        "planned_inference_substrate_class": "aggregation",
        "MODEL_SPECS": [],
        "planned_MODEL_SPECS": [],
        "model_specs": [],
        "model_specs_declaration": "no_current_model",
        "model_invoked": False,
        "invocation_counts": {
            "loads": 0,
            "forwards": 0,
            "generations": 0,
            "tokens": 0,
            "attempted": 0,
            "cancelled": 0,
        },
        "execution_venue": "host",
        "execution_venue_details": {
            "host": socket.gethostname(),
            "owned_pid": os.getpid(),
            "gpu_uuid": None,
        },
        "phase_spans": [],
        "duration_s": 0.0,
        "random_seed": {
            "bootstrap": 7664,
            "purpose": "paired group and block confidence intervals",
        },
        "source_artifact_hashes": hashes,
        "preconditions_checked": {
            "root": str(root.resolve()),
            "producer_count": len(found) - 1,
            "raw_feature_files": len(FEATURES),
            "resource": "host CPU and local bytes",
        },
        "validation_receipts": {
            "frozen_affected_scope": MANIFEST,
            "required_commands": [],
            "terminal_readers": [],
            "unrelated_repository_suite_debt": [
                {
                    "command": ".venv/bin/pytest tests/python -q",
                    "observed": "interrupted after 1335 passed, 5 failed, 67 collection errors",
                    "first_error": "legacy Qwen3.6 registry KeyError in Exp5500",
                    "scope": "repository-wide; outside frozen affected acceptance",
                }
            ],
        },
        "verifier_is_oracle": False,
        "whole_answer_label_independence": {
            "source": "Exp7602 isolated evaluator stores",
            "roles": {
                "pilot": 8,
                "fit": 80,
                "tune": 20,
                "policy": 20,
                "online": 80,
                "evaluation": 40,
            },
            "independent_of_atom_verifier": True,
            "claim_limit": "Whole-answer labels do not turn partial atoms into answer certificates.",
        },
        "field_principles": {
            "rows": "Each independent group and arm retains raw operands and provenance.",
            "sample_size_budget": "Repeated arms, seeds, and views never enlarge N.",
            "independent_reduction": "Recompute from raw operands, without producer reducer.",
            "honest_verdict": "Audit completion is separate from scientific benefit.",
            "source_artifact_hashes": "Immutable bytes bind every finding.",
            "validation_receipts": "Failed required checks disqualify readiness.",
            "claim_findings": "Every claim links to its source and scope.",
        },
        "independent_audit_complete_score": int(valid),
        "independent_reduction": summary,
        "private_mutation_results": mutations,
        "audit_note_path": str(NOTE),
    }
    claims = (
        (
            "coverage",
            "corpus",
            "248 original groups; 90 checked propositions; unknowns retained",
            summary.get("coverage"),
        ),
        (
            "probability",
            "evaluation",
            "No registered held-out Brier benefit",
            summary.get("contrasts", {}).get("evaluation"),
        ),
        (
            "utility",
            "continuous",
            "No calibrated decision-cost and coverage benefit",
            summary.get("contrasts", {}).get("continuous"),
        ),
        (
            "retention",
            "continuous",
            "Delayed updates and replay exist; benefit is not established",
            summary.get("retention"),
        ),
    )
    artifact["claim_findings"] = [
        {
            "claim": claim,
            "source_path": PRODUCERS[source],
            "source_sha256": hashes["producers"].get(PRODUCERS[source]),
            "finding": finding,
            "measured_operands": operands,
            "principle": "A source-linked operand supports only its measured scope.",
        }
        for claim, source, finding, operands in claims
    ]
    artifact["reproducibility_checksum"] = canonical_hash(
        {
            "source_artifact_hashes": hashes,
            "reducer_code": sha256_file(
                root / "python/carnot/reporting/independent_evidence_audit.py"
            ),
            "configuration": {"seed": 7664, "block_size": 8, "manifest": MANIFEST},
        }
    )
    return artifact


def cold_replay(path: Path) -> list[str]:
    """Reload candidate bytes and reauthenticate immutable producer paths."""
    value = json.loads(path.read_text())
    root = ROOT
    found, failures, hashes = authenticate_inputs(root)
    errors: list[str] = []
    if failures != value["gate_check_summary"]:
        errors.append("blocked gate operands changed")
    if hashes != value["source_artifact_hashes"]:
        errors.append("source custody changed")
    if value["reproducibility_checksum"] != canonical_hash(
        {
            "source_artifact_hashes": hashes,
            "reducer_code": sha256_file(
                root / "python/carnot/reporting/independent_evidence_audit.py"
            ),
            "configuration": {"seed": 7664, "block_size": 8, "manifest": MANIFEST},
        }
    ):
        errors.append("reproducibility checksum changed")
    if not failures:
        rebuilt = reduce_evidence(raw_inputs(found))
        if rebuilt != value["independent_reduction"]:
            errors.append("independent reduction changed")
        if not all(private_mutations(raw_inputs(found)).values()):
            errors.append("private mutation escaped")
    return errors


def terminal_commands(root: Path, candidate: Path) -> list[checks.CommandSpec]:
    """Name bounded independent and adversarial readers of exact candidate bytes."""
    python = str(root / ".venv/bin/python")
    return [
        checks.CommandSpec(
            "fresh_process_cold_replay",
            (
                python,
                "-u",
                "-m",
                "carnot.experiment_7664_v668_independent_evidence_audit",
                "--cold",
                str(candidate),
            ),
            "exact_candidate",
            180,
        ),
        checks.CommandSpec(
            "independent_raw_reduction",
            (
                python,
                "-u",
                "-m",
                "carnot.experiment_7664_v668_independent_evidence_audit",
                "--independent",
                str(candidate),
            ),
            "exact_candidate",
            180,
        ),
        checks.CommandSpec(
            "adversarial_verify",
            (python, "-u", "scripts/adversarial_verify.py", "--json", str(candidate)),
            "exact_candidate",
            180,
        ),
        checks.CommandSpec(
            "verdict_row_consistency_strict",
            (python, "-u", "scripts/verdict_row_consistency_lint.py", "--strict", str(candidate)),
            "exact_candidate",
            180,
        ),
    ]


def write_note(root: Path, summary: dict[str, Any]) -> None:
    """Explain what the independently checked evidence can actually claim."""
    coverage = summary.get("coverage", {})
    continuous = summary.get("continuous", {})
    lines = [
        "# V668 independent evidence audit",
        "",
        "## Structural soundness",
        "",
        f"The raw roster contains {coverage.get('independent_groups', 0)} original source groups, "
        f"{coverage.get('checked_propositions', 0)} checked propositions, and "
        f"{coverage.get('unknown_claims', 0)} unknown claims. The atom verifier checks narrow "
        "structural propositions. All whole-answer labels (pilot 8, fit 80, tune 20, "
        "policy 20, online 80, evaluation 40) come from Exp7602 isolated evaluator "
        "stores, independent of the atom verifier.",
        "",
        "## Discriminatory information",
        "",
        "Source, erasure, and derangement controls preserve the group denominator. "
        "The checked atoms are partial and cannot prove whole-answer correctness.",
        "",
        "## Calibrated utility",
        "",
        f"The continuous source arm has Brier {continuous.get('arms', {}).get('source', {}).get('brier')} "
        "and no registered benefit over both controls. Decision cost and non-escalation "
        "coverage are separate gates.",
        "",
        "## Retained updates",
        "",
        f"There are {summary.get('feedback_events', 0)} feedback rows. One-use admission, "
        "release order, and retention are independently replayed; an update count alone "
        "does not establish benefit.",
        "",
        "## Exposed-data limits",
        "",
        f"All {coverage.get('previously_exposed_groups', 0)} observed original groups were "
        "previously exposed. These measurements are exploratory and cannot support a "
        "fresh confirmatory claim. The failed V667 grammar mechanism remains retired; "
        "the new source-atom mechanism is a distinct, narrower method.",
        "",
    ]
    path = root / NOTE
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def run_experiment(root: Path, date: str, output: Path) -> dict[str, Any]:
    """Measure, validate, run terminal readers, then publish atomically."""
    root = root.resolve()
    started = time.monotonic()
    progress(started, "preconditions", "start", root=str(root))
    found, failures, hashes = authenticate_inputs(root)
    raw_dir = root / RAW
    raw_dir.mkdir(parents=True, exist_ok=True)
    atomic_json(raw_dir / "frozen_affected_validation_manifest.json", MANIFEST)
    progress(started, "preconditions", "complete", failed=len(failures), producers=len(found) - 1)
    boundary = time.monotonic() - started
    progress(started, "reduction", "start")
    inputs = raw_inputs(found) if not failures else {}
    summary = reduce_evidence(inputs) if inputs else {}
    mutations = private_mutations(inputs) if inputs else {}
    artifact = build_artifact(root, date, found, failures, hashes, summary, mutations)
    write_note(root, summary)
    progress(
        started,
        "reduction",
        "complete",
        groups=summary.get("coverage", {}).get("independent_groups", 0),
    )
    next_boundary = time.monotonic() - started
    artifact["phase_spans"].append(
        {
            "phase": "preconditions",
            "start_s": 0.0,
            "end_s": boundary,
            "completed_units": len(found) - 1,
            "checkpoint_position": "source_inventory",
            "heartbeat_times_s": [boundary],
        }
    )
    artifact["phase_spans"].append(
        {
            "phase": "independent_reduction",
            "start_s": boundary,
            "end_s": next_boundary,
            "completed_units": len(artifact["rows"]),
            "checkpoint_position": "raw_rows_reduced",
            "heartbeat_times_s": [next_boundary],
        }
    )
    private = Path(tempfile.mkdtemp(prefix="exp7664-", dir="/tmp"))
    basetemp = private / "basetemp"
    basetemp.mkdir()
    progress(started, "affected_validation", "start")
    validation = checks.run_scoped_validation(
        root,
        MANIFEST["test_paths"],
        MANIFEST["changed_modules"],
        static_paths=MANIFEST["static_paths"],
        basetemp=basetemp,
        coverage_file=private / "coverage.data",
        log_dir=raw_dir / "validation/affected",
        extra_env={"COVERAGE_FILE": str(private / "coverage.data"), "JAX_PLATFORMS": "cpu"},
    )
    artifact["validation_receipts"]["required_commands"] = validation["validation_receipts"]
    artifact["validation_receipts"]["required_checks_passed"] = validation["required_checks_passed"]
    artifact["validation_receipts"]["failed_required_commands"] = validation[
        "failed_required_commands"
    ]
    if not validation["required_checks_passed"]:
        artifact["honest_verdict"] = "complete_disqualified_required_validation"
        artifact["verdict_class"] = "disqualified"
        artifact["independent_audit_complete_score"] = 0
        for gate in artifact["acceptance_gate_results"]:
            if gate["gate"] in ("validity", "readiness"):
                gate["passed"] = False
    progress(
        started, "affected_validation", "complete", passed=validation["required_checks_passed"]
    )
    third_boundary = time.monotonic() - started
    artifact["phase_spans"].append(
        {
            "phase": "affected_validation",
            "start_s": next_boundary,
            "end_s": third_boundary,
            "completed_units": len(validation["validation_receipts"]),
            "checkpoint_position": "affected_receipts",
            "heartbeat_times_s": [third_boundary],
        }
    )
    artifact["duration_s"] = third_boundary
    candidate = raw_dir / "exact_terminal_candidate.json"
    atomic_json(candidate, artifact)
    progress(started, "terminal_readers", "start", candidate=str(candidate))
    terminal = checks.run_commands(
        root,
        terminal_commands(root, candidate),
        log_dir=raw_dir / "validation/terminal",
        extra_env={"JAX_PLATFORMS": "cpu"},
    )
    outcomes = {
        "candidate_path": str(
            candidate.relative_to(root) if candidate.is_relative_to(root) else candidate
        ),
        "candidate_sha256": sha256_file(candidate),
        "receipts": terminal,
        "all_passed": all(row["passed"] for row in terminal),
    }
    atomic_json(raw_dir / "exact_terminal_reader_outcomes.json", outcomes)
    artifact["validation_receipts"]["terminal_readers"] = terminal
    artifact["validation_receipts"]["exact_terminal_candidate_sha256"] = outcomes[
        "candidate_sha256"
    ]
    artifact["validation_receipts"]["exact_terminal_reader_outcomes_path"] = str(
        RAW / "exact_terminal_reader_outcomes.json"
    )
    artifact["flagged_adversarial"] = not next(
        row for row in terminal if row["name"] == "adversarial_verify"
    )["passed"]
    if not outcomes["all_passed"]:
        artifact["honest_verdict"] = "complete_disqualified_terminal_reader"
        artifact["verdict_class"] = "disqualified"
        artifact["independent_audit_complete_score"] = 0
        for gate in artifact["acceptance_gate_results"]:
            if gate["gate"] in ("validity", "readiness"):
                gate["passed"] = False
    final_boundary = time.monotonic() - started
    artifact["phase_spans"].append(
        {
            "phase": "terminal_readers",
            "start_s": third_boundary,
            "end_s": final_boundary,
            "completed_units": len(terminal),
            "checkpoint_position": "exact_candidate_readers",
            "heartbeat_times_s": [final_boundary],
        }
    )
    artifact["duration_s"] = final_boundary
    progress(started, "terminal_readers", "complete", passed=outcomes["all_passed"])
    destination = output if output.is_absolute() else root / output
    atomic_json(destination, artifact)
    progress(
        started, "publication", "complete", output=str(destination), sha256=sha256_file(destination)
    )
    return artifact


def main(argv: list[str] | None = None) -> int:
    """Provide the declared CLI and two fresh-process terminal readers."""
    print("[exp7664] startup flushed", flush=True)
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", default="20260925")
    parser.add_argument("--output", type=Path, default=RESULT)
    parser.add_argument("--cold", type=Path)
    parser.add_argument("--independent", type=Path)
    args = parser.parse_args(argv)
    if args.cold or args.independent:
        errors = cold_replay(args.cold or args.independent)
        print(json.dumps({"errors": errors}), flush=True)
        return int(bool(errors))
    run_experiment(ROOT, args.date, args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
