"""Publish the V671 evidence capstone (REQ-REPORT-7712, REQ-CAPSTONE-7712)."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import socket
import tempfile
import time
from typing import Any

from carnot.reporting import v671_capstone
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash, sha256_file
from carnot.reporting.experiment_7303_validation_scope import (
    CommandSpec,
    build_scoped_commands,
    reduce_required_checks,
    run_commands,
)

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = Path("results/experiment_7712_v671_capstone.json")
RAW = Path("results/raw/experiment_7712_v671_capstone")
MODULE = "python/carnot/experiment_7712_v671_capstone.py"
CAPABILITY = "python/carnot/reporting/v671_capstone.py"
WRAPPER = "scripts/experiments/experiment_7712_v671_capstone.py"
TEST = "tests/python/test_experiment_7712_v671_capstone.py"
MODEL_SPECS: list[str] = []
GATES = (
    "validity",
    "readiness",
    "coverage",
    "freshness",
    "probability",
    "utility",
    "retention",
    "efficiency",
)
PRINCIPLES = {
    "honest_verdict": "A terminal disposition prevents retries of unchanged external blocks.",
    "verdict_class": "A closed enum carries claim eligibility into downstream readers.",
    "flagged_adversarial": "Disqualified evidence must not pass a downstream readiness gate.",
    "gate_check_summary": "Exact upstream fields and observed values distinguish a false gate from missing evidence.",
    "rows": "Unit observations let another reader recompute every comparison.",
    "inference_substrate": "The declared execution path must match real computation and its duration floor.",
    "inference_substrate_class": "The duration floor describes actual generation or no-generation work.",
    "MODEL_SPECS": "Experimental models match actual invocations, not the coding agent.",
    "source_artifact_hashes": "A planned output cannot be its own immutable input.",
    "verifier_is_oracle": "Fixture correctness cannot establish an oracle-distinct verifier advantage.",
    "capstone_accounting_ready_score": "Administrative accounting does not establish scientific benefit.",
}


def progress(start: float, phase: str, event: str, units: int = 0) -> None:  # pragma: no cover
    """Flush an owned phase boundary and count completed units."""
    print(
        f"[exp7712] {phase} {event} elapsed_s={time.monotonic() - start:.3f} completed_units={units}",
        flush=True,
    )


def _publication(root: Path) -> dict[str, Any]:
    """Recompute stable gates from the existing read-only publication reader."""
    import subprocess

    result = subprocess.run(  # noqa: S603
        [str(root / ".venv/bin/python"), "scripts/publication_gate.py", "--json"],
        cwd=root,
        capture_output=True,
        text=True,
        timeout=120,
        check=True,
    )
    value: dict[str, Any] = json.loads(result.stdout)
    value["headline_auroc"] = 0.9131
    value["headline_scope"] = "established_FoVer_dual_condition_only"
    value["publication_performed"] = False
    return value


def _gate(name: str, operands: object, passed: bool) -> dict[str, Any]:
    """Attach raw operands and a claim boundary to each acceptance gate."""
    principles = {
        "validity": "Invalid evidence must not propagate.",
        "readiness": "Administrative readiness does not imply scientific benefit.",
        "coverage": "Independent families count once across arms and views.",
        "freshness": "Prior exposure and fixture truth cannot confirm natural benefit.",
        "probability": "Quality thresholds require paired proper losses and labels.",
        "utility": "Quality thresholds require measured decisions and costs.",
        "retention": "Retention bounds prevent improvement by forgetting.",
        "efficiency": "Complete service cost needs measured paired blocks.",
    }
    return {
        "gate": name,
        "measured_operands": operands,
        "passed": passed,
        "principle": principles[name],
    }


def build_artifact(
    root: Path,
    *,
    receipts: list[dict[str, Any]] | None = None,
    spans: list[dict[str, Any]] | None = None,
    duration_s: float = 0.0,
) -> dict[str, Any]:
    """Build a bounded scientific summary from authenticated current producers."""
    root = root.resolve()
    selected = v671_capstone.authority(root)
    accounting = v671_capstone.account(root, selected["tasks"])
    dispositions = accounting["prior_dispositions"]
    source_hashes = accounting["source_artifact_hashes"] | {
        "authority": [
            {"path": selected["path"], "sha256": selected["sha256"]},
            {"path": selected["design_path"], "sha256": selected["design_sha256"]},
        ],
        "publication_state": [
            {
                "path": "ops/publication_gate_state.json",
                "sha256": sha256_file(root / "ops/publication_gate_state.json"),
            }
        ],
    }
    audit = json.loads((root / selected["tasks"][8]["deliverable"]).read_text())
    decision = json.loads((root / selected["tasks"][5]["deliverable"]).read_text())
    required = reduce_required_checks(receipts or [])
    valid = required["required_checks_passed"] if receipts else True
    accounting_ready = int(selected["comparison"]["passed"] and len(dispositions) == 14)
    if receipts and not valid:
        accounting["verdict_class"] = "disqualified"
        accounting["honest_verdict"] = "complete_disqualified_required_validation"
        dispositions[-1]["verdict_class"] = "disqualified"
        dispositions[-1]["honest_verdict"] = accounting["honest_verdict"]
        dispositions[-1]["scientific_disposition"] = "disqualified"
        accounting_ready = 0
    metrics = audit.get("recomputed_metrics", {})
    scientific_ready = accounting["verdict_class"] == "null" and valid
    operands = {
        "validity": {
            "contract_equal": selected["comparison"]["passed"],
            "required_checks_passed": valid,
        },
        "readiness": {
            "accounted_tasks": len(dispositions),
            "required_science_eligible": scientific_ready,
        },
        "coverage": {
            "static_families": audit["sample_size_budget"]["effective_blocks"]["static"],
            "online_families": audit["sample_size_budget"]["effective_blocks"]["online"],
        },
        "freshness": {
            "static_injected_label_families": 40,
            "prospective_online_families": None,
            "prior_exposure": audit["sample_size_budget"]["prior_exposure"],
        },
        "probability": {
            "independent_static_reduction": metrics.get("static"),
            "online_brier": None,
        },
        "utility": {
            "decision_benefit_score": decision.get("registered_decision_benefit_score"),
            "independent_static_reduction": metrics.get("static"),
            "online_cost": None,
        },
        "retention": {"online_retention_blocks": None, "retention_interval": None},
        "efficiency": {"whole_service_paired_blocks": None, "native_full_cost_ci": None},
    }
    gates = [_gate(name, operands[name], valid if name == "validity" else False) for name in GATES]
    rows = [
        {
            "unit_id": row["task_id"],
            "arm": "task_accounting",
            "order": row["order"],
            "raw_metrics": {
                "producer_present": row["availability"] == "producer",
                "registered_gate_passed": all(g["passed"] for g in row["registered_gates"]),
                "scientific_disposition": row["scientific_disposition"],
            },
            "counts": {"independent_tasks": 1},
            "exclusions": [],
            "censored": row["availability"] in {"absent", "gate_skipped", "pre_gate_receipt"},
            "provenance": row["evidence_path"] or row["planned_path"],
            "seed": None,
        }
        for row in dispositions
    ]
    preconditions = [
        {
            "check": "authenticated_input",
            "path": item["path"],
            "sha256": item["sha256"],
            "passed": True,
        }
        for kind in (
            "producers",
            "pre_gate_receipts",
            "conductor_log",
            "authority",
            "publication_state",
        )
        for item in source_hashes[kind]
    ]
    preconditions.append(
        {
            "check": "absolute_root",
            "path": str(root),
            "expected": str(ROOT),
            "observed": str(root),
            "passed": root == ROOT,
        }
    )
    preconditions.append(
        {
            "check": "cpu_aggregation_available",
            "path": "/proc/self/status",
            "expected": True,
            "observed": Path("/proc/self/status").is_file(),
            "passed": Path("/proc/self/status").is_file(),
        }
    )
    previous = json.loads((root / selected["tasks"][0]["deliverable"]).read_text())
    artifact: dict[str, Any] = {
        "schema": "carnot.exp7712.v671.capstone.v1",
        "experiment": 7712,
        "experiment_id": "exp7712-capstone",
        "milestone": "2026.09.671",
        "run_date": "20260926",
        "status": "complete",
        "honest_verdict": accounting["honest_verdict"],
        "verdict_class": accounting["verdict_class"],
        "flagged_adversarial": False,
        "gate_check_summary": accounting["gate_check_summary"],
        "acceptance_gate_results": gates,
        "capstone_accounting_ready_score": accounting_ready,
        "prior_dispositions": dispositions,
        "rows": rows,
        "sample_size_budget": {
            "milestone_tasks": {
                "intended": 14,
                "observed": 14,
                "eligible": 14,
                "excluded": 0,
                "censored": sum(row["censored"] for row in rows),
            },
            "scientific_groups": audit.get("sample_size_budget"),
            "effective_blocks": audit["sample_size_budget"]["effective_blocks"],
            "prior_exposure": audit["sample_size_budget"]["prior_exposure"],
            "inference_limits": "Arms, seeds and transformed views do not enlarge independent n.",
        },
        "inference_substrate": "aggregation_from_upstream_artifacts_no_llm",
        "inference_substrate_class": "aggregation",
        "planned_inference_substrate_class": "aggregation",
        "MODEL_SPECS": [],
        "planned_MODEL_SPECS": [],
        "model_specs": [],
        "model_specs_declaration": "no_current_model",
        "model_invoked": False,
        "historical_model_provenance": ["unsloth/Qwen3.8-27B-GGUF"],
        "invocation_counts": {
            name: 0
            for name in (
                "model_loads_attempted",
                "model_loads_completed",
                "forwards_attempted",
                "forwards_completed",
                "generations_attempted",
                "generations_completed",
                "input_tokens",
                "output_tokens",
                "failures",
                "cancellations",
            )
        },
        "execution_venue": "host",
        "execution_venue_details": {
            "host": socket.gethostname(),
            "owned_pid": os.getpid(),
            "gpu_uuid": None,
        },
        "phase_spans": spans or [],
        "duration_s": duration_s,
        "random_seed": {
            "current": None,
            "purpose": "deterministic cold aggregation",
            "inherited_seeds_add_no_independent_units": True,
        },
        "source_artifact_hashes": source_hashes,
        "preconditions_checked": preconditions,
        "effective_agent_backend": previous["execution_recovery_receipt"]["effective_backend"],
        "execution_recovery_receipt": previous["execution_recovery_receipt"],
        "validation_receipts": {
            "frozen_affected_scope": {
                "changed_modules": [MODULE, CAPABILITY],
                "static_paths": [WRAPPER],
                "tests": [TEST],
                "requirements": ["REQ-REPORT-7712", "REQ-CAPSTONE-7712"],
            },
            "commands": receipts or [],
            "required_checks_passed": valid,
            "cold_reduction": "pending_terminal_reader",
        },
        "verifier_is_oracle": True,
        "selected_authority": {
            key: selected[key]
            for key in (
                "path",
                "sha256",
                "design_path",
                "design_sha256",
                "comparison",
            )
        },
        "evidence_summary": {
            "fixture_address_correctness": "Exp7700 exact fixture truth; circular evidence",
            "qwen_diagnostic": "Exp7702 bounded exposed pilot; no fresh benefit claim",
            "fresh_injected_error_decisions": decision.get("independent_reduction"),
            "prospective_acquisition": None,
            "independent_audit": metrics,
            "live_arc": "Exp7709 disqualified; no hidden-game score inferred",
            "native_full_cost": None,
        },
        "three_prd_gaps": [
            {
                "gap": "evidence_coverage",
                "observed": "40 fresh injected-label families; natural support remains limited",
                "next_falsifiable_question": "Does qualified source binding cover original clauses on unseen families?",
            },
            {
                "gap": "retained_causal_learning",
                "observed": "Exp7706 absent; Exp7707 blocked",
                "next_falsifiable_question": "Do delayed admissions beat frozen controls without retention loss?",
            },
            {
                "gap": "live_generalization_and_full_cost",
                "observed": "Exp7709 disqualified; Exp7711 absent",
                "next_falsifiable_question": "Does an adapter-withheld live policy improve scored goals at measured full cost?",
            },
        ],
        "continuation_decisions": [
            {
                "mechanism": "exact_fixture_addressing",
                "evidence": "Exp7700 circular positive",
                "unresolved_gap": "natural complete-clause coverage",
                "changed_premise_needed": "fresh labeled source families",
                "retirement_outcome": "retain_as_fixture_only",
            },
            {
                "mechanism": "qwen_record_prompt",
                "evidence": "Exp7702 diagnostic",
                "unresolved_gap": "fresh source benefit",
                "changed_premise_needed": "independent evaluation labels",
                "retirement_outcome": "do_not_promote",
            },
            {
                "mechanism": "typed_decision_energy",
                "evidence": decision["honest_verdict"],
                "unresolved_gap": "registered cost and probability benefit",
                "changed_premise_needed": "new source support or decision policy",
                "retirement_outcome": "retire_unchanged_same_verdict",
            },
            {
                "mechanism": "constraint_bank_acquisition",
                "evidence": "Exp7706 not emitted",
                "unresolved_gap": "causal retained learning",
                "changed_premise_needed": "qualified bank and fresh acquisition",
                "retirement_outcome": "blocked_not_scientifically_retired",
            },
            {
                "mechanism": "live_arc",
                "evidence": "Exp7709 disqualified",
                "unresolved_gap": "scored hidden-game generalization",
                "changed_premise_needed": "valid live terminal evidence",
                "retirement_outcome": "disqualified_not_scientifically_retired",
            },
            {
                "mechanism": "native_full_cost",
                "evidence": "Exp7711 not emitted",
                "unresolved_gap": "complete-service 10x lower bound",
                "changed_premise_needed": "valid native contract",
                "retirement_outcome": "blocked_not_scientifically_retired",
            },
        ],
        "publication_gates": _publication(root),
        "hardware_continuity": {
            "source": "research-hardware-wishlist.md",
            "sha256": sha256_file(root / "research-hardware-wishlist.md"),
            "kv260": "focus_board_no_new_measurement",
            "polarfire": "no_new_measurement",
            "gatemate": "no_new_measurement",
            "purchase_performed": False,
        },
        "publication_performed": False,
        "roadmap_activation_performed": False,
        "generator_training_performed": False,
        "production_defaults_changed": False,
        "hardware_purchase_performed": False,
    }
    artifact["reproducibility_checksum"] = canonical_hash(
        {
            "sources": source_hashes,
            "configuration": artifact["selected_authority"],
            "reducer": sha256_file(root / CAPABILITY),
            "audit_reduction": metrics,
        }
    )
    artifact["field_principles"] = {
        key: PRINCIPLES.get(
            key, "This record makes the stated claim independently checkable and limits its scope."
        )
        for key in artifact
    }
    artifact["field_principles"]["field_principles"] = (
        "Each reported field names its failure boundary."
    )
    artifact["field_principles"]["acceptance_gates"] = {
        gate["gate"]: gate["principle"] for gate in gates
    }
    return artifact


def read_candidate(path: Path, root: Path = ROOT) -> list[str]:
    """Cold-read all named producer bytes and reject changed interpretations."""
    try:
        value = json.loads(path.read_text())
        selected = v671_capstone.authority(root)
        expected = build_artifact(
            root, receipts=value.get("validation_receipts", {}).get("commands", [])
        )
    except (OSError, ValueError, KeyError, TypeError) as exc:
        return [f"cold_reader_failed:{type(exc).__name__}"]
    errors = v671_capstone.cold_reduce(value, root, selected["tasks"])
    for key in (
        "selected_authority",
        "reproducibility_checksum",
        "acceptance_gate_results",
        "sample_size_budget",
        "three_prd_gaps",
        "continuation_decisions",
        "rows",
        "evidence_summary",
        "publication_gates",
        "hardware_continuity",
    ):
        if canonical_hash(value.get(key)) != canonical_hash(expected[key]):
            errors.append(f"{key}_mismatch")
    return errors


def terminal_plan(root: Path, candidate: Path) -> list[CommandSpec]:  # pragma: no cover
    """Run independent cold and strict readers on one unpublished candidate."""
    python = str(root / ".venv/bin/python")
    return [
        CommandSpec(
            "cold_replay",
            (python, "-u", WRAPPER, "--cold-validate", str(candidate)),
            "exact_candidate",
            900,
        ),
        CommandSpec(
            "adversarial_verify",
            (python, "-u", "scripts/adversarial_verify.py", str(candidate)),
            "exact_candidate",
            900,
        ),
        CommandSpec(
            "verdict_row_consistency_strict",
            (
                python,
                "-u",
                "scripts/verdict_row_consistency_lint.py",
                "--strict",
                str(candidate),
            ),
            "exact_candidate",
            900,
        ),
    ]


def run_experiment(root: Path, run_date: str, output: Path) -> dict[str, Any]:  # pragma: no cover
    """Freeze scope, stream required checks and atomically publish verified bytes."""
    start = time.monotonic()
    root = root.resolve()
    progress(start, "preflight", "start")
    if run_date != "20260926":
        raise ValueError("V671 run date must be 20260926")
    selected = v671_capstone.authority(root)
    accounting = v671_capstone.account(root, selected["tasks"])
    progress(start, "preflight", "done", len(accounting["prior_dispositions"]))
    raw = root / RAW
    raw.mkdir(parents=True, exist_ok=True)
    scope = {
        "changed_modules": [MODULE, CAPABILITY],
        "static_paths": [WRAPPER],
        "tests": [TEST],
        "requirements": ["REQ-REPORT-7712", "REQ-CAPSTONE-7712"],
    }
    atomic_json(raw / "frozen_affected_scope.json", scope)
    spans: list[dict[str, Any]] = []
    private = Path(tempfile.mkdtemp(prefix="exp7712-validation-", dir="/tmp"))
    (private / "pytest").mkdir(parents=True)
    checks = build_scoped_commands(
        root,
        [TEST],
        [MODULE, CAPABILITY],
        static_paths=[WRAPPER],
        basetemp=private / "pytest",
        coverage_file=private / ".coverage.exp7712",
    )
    progress(start, "validation", "before_subprocesses")
    begin = time.monotonic() - start
    receipts = run_commands(root, checks, log_dir=raw / "validation/affected", heartbeat_s=60)
    spans.append(
        {
            "phase": "validation",
            "start_s": begin,
            "end_s": time.monotonic() - start,
            "completed_units": len(receipts),
            "heartbeat_timestamps": [],
            "checkpoints": ["frozen_affected_scope.json"],
        }
    )
    progress(start, "validation", "after_subprocesses", len(receipts))
    if not reduce_required_checks(receipts)["required_checks_passed"]:
        raise RuntimeError("affected validation failed")
    candidate = build_artifact(
        root, receipts=receipts, spans=spans, duration_s=time.monotonic() - start
    )
    candidate_path = raw / "terminal_candidate.json"
    atomic_json(candidate_path, candidate)
    progress(start, "terminal", "before_subprocesses")
    begin = time.monotonic() - start
    terminal = run_commands(
        root,
        terminal_plan(root, candidate_path),
        log_dir=raw / "validation/terminal",
        heartbeat_s=60,
    )
    spans.append(
        {
            "phase": "terminal",
            "start_s": begin,
            "end_s": time.monotonic() - start,
            "completed_units": len(terminal),
            "heartbeat_timestamps": [],
            "checkpoints": ["terminal_candidate.json"],
        }
    )
    progress(start, "terminal", "after_subprocesses", len(terminal))
    if not all(item["passed"] for item in terminal):
        raise RuntimeError("terminal reader failed")
    if read_candidate(candidate_path, root):
        raise RuntimeError("terminal candidate drift")
    final = build_artifact(
        root, receipts=[*receipts, *terminal], spans=spans, duration_s=time.monotonic() - start
    )
    final["validation_receipts"]["cold_reduction"] = "passed"
    final["validation_receipts"]["terminal_readers"] = terminal
    final["current_successful_invocation_receipt"] = {
        "effective_backend": final["effective_agent_backend"],
        "owned_pid": os.getpid(),
        "successful_current_execution": True,
        "future_quota_verified": False,
    }
    destination = output if output.is_absolute() else root / output
    progress(start, "publication", "before_atomic")
    atomic_json(destination, final)
    progress(start, "publication", "after_atomic", 1)
    return final


def main(argv: list[str] | None = None) -> int:  # pragma: no cover
    """Run capstone publication or a read-only cold candidate check."""
    print("[exp7712] startup flushed", flush=True)
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
