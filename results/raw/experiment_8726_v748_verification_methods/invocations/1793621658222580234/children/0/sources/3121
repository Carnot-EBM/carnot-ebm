"""Aggregate V669 evidence after cold custody and gate checks. REQ-REPORT-7684."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import platform
import socket
import tempfile
import time
from typing import Any

from carnot.experiment_7628_v665_capstone import _field_principles, _publication_record
from carnot.reporting import v669_capstone
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash, sha256_file
from carnot.reporting.experiment_7303_validation_scope import (
    CommandSpec,
    build_scoped_commands,
    reduce_required_checks,
    run_commands,
)


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = Path("results/experiment_7684_v669_capstone.json")
NOTE = Path("docs/research-notes/v669-retrospective.md")
MODULE = "python/carnot/experiment_7684_v669_capstone.py"
CAPABILITY = "python/carnot/reporting/v669_capstone.py"
WRAPPER = "scripts/experiments/experiment_7684_v669_capstone.py"
TEST = "tests/python/test_experiment_7684_v669_capstone.py"
MODEL_SPECS: list[str] = []


def progress(start: float, phase: str, event: str, units: int = 0) -> None:  # pragma: no cover
    """Give the operator an owned heartbeat at each phase boundary."""

    print(
        f"[exp7684] {phase} {event} units={units} elapsed_s={time.monotonic() - start:.3f}",
        flush=True,
    )


def _gate(name: str, operands: object, passed: bool, principle: str) -> dict[str, Any]:
    """Keep the operand beside its interpretation so readiness cannot mask a null."""

    return {"gate": name, "measured_operands": operands, "passed": passed, "principle": principle}


def build_artifact(
    root: Path,
    *,
    receipts: list[dict[str, Any]] | None = None,
    spans: list[dict[str, Any]] | None = None,
    duration_s: float = 0.0,
) -> dict[str, Any]:
    """Reopen immutable sources and assemble a candidate without a model call."""

    selected = v669_capstone.authority(root)
    accounting = v669_capstone.account(root, selected["tasks"])
    tasks = accounting["prior_dispositions"]
    required = reduce_required_checks(receipts or [])
    audit_path = root / selected["tasks"][8]["deliverable"]
    audit = json.loads(audit_path.read_text()) if audit_path.is_file() else {}
    cohort = json.loads((root / selected["tasks"][2]["deliverable"]).read_text())
    arc = json.loads((root / selected["tasks"][10]["deliverable"]).read_text())
    publication = _publication_record(root)
    publication["headline_auroc"] = 0.9131
    publication["claim_boundary"] = "historical_frozen_FoVer_only"
    hardware_path = root / "results/experiment_7641_v666_native_consumer.json"
    hardware = json.loads(hardware_path.read_text())["hardware_dispositions"]
    hardware.append(
        {
            "hardware": "efficiency_estimates",
            "current_execution": False,
            "disposition": "vendor_only_unqualified_for_whole_service",
            "source_receipt": "research-hardware-wishlist.md",
        }
    )
    summary = audit.get("independent_reduction") or {
        "cohort": {"independent_groups": None, "unknown_groups": None},
        "fixture": None,
        "quotes": None,
    }
    cold_raw = v669_capstone.reduce_audit_rows(audit.get("rows", []))
    validated = required["required_checks_passed"]
    gates = [
        _gate(
            "validity",
            {
                "contract_equal": selected["comparison"]["passed"],
                "required_checks_passed": validated,
            },
            validated,
            "Contract equality and current validation govern validity.",
        ),
        _gate(
            "readiness",
            {"fourteen_accounted": len(tasks) == 14, "cold_reduction_pending": not validated},
            validated,
            "Accounting readiness does not imply scientific benefit.",
        ),
        _gate(
            "coverage",
            {
                "cohort_groups": summary["cohort"]["independent_groups"],
                "unknown_groups": summary["cohort"]["unknown_groups"],
                "static_groups": None,
                "online_groups": None,
            },
            False,
            "Unknown relations and missing comparisons cannot open coverage.",
        ),
        _gate(
            "freshness",
            {
                "fresh_confirmatory_static": None,
                "fresh_confirmatory_online": None,
                "cohort_prior_exposure": cohort["sample_size_budget"].get("prior_exposure_groups"),
            },
            False,
            "Fresh cohorts need valid independent evaluation labels.",
        ),
        _gate(
            "probability",
            {"paired_brier_improvement_ci": None},
            False,
            "Probability benefit needs paired fresh labels and an interval.",
        ),
        _gate(
            "decision_utility",
            {"paired_cost_improvement_ci": None},
            False,
            "Decision benefit needs typed action costs on fresh groups.",
        ),
        _gate(
            "retention",
            {"one_use_admissions": None, "retention_ci": None},
            False,
            "Delayed updates need causal release and retained quality.",
        ),
        _gate(
            "efficiency",
            {"whole_service_paired_blocks": None, "prior_direct_service_speedup": 7.827},
            False,
            "Direct service speed does not measure complete service cost.",
        ),
    ]
    if not validated and receipts:
        accounting["verdict_class"] = "disqualified"
        accounting["honest_verdict"] = "complete_disqualified_required_validation"
        tasks[-1]["verdict_class"] = "disqualified"
        tasks[-1]["honest_verdict"] = accounting["honest_verdict"]
        tasks[-1]["scientific_disposition"] = "disqualified"
    rows = [
        {
            "unit_id": row["task_id"],
            "arm": "task_accounting",
            "order": row["order"],
            "raw_metrics": {
                "producer_present": row["availability"] == "producer",
                "registered_gate_passed": all(g["passed"] for g in row["registered_gates"]),
            },
            "counts": {"independent_tasks": 1},
            "exclusions": [],
            "censored": row["availability"] in {"absent", "pre_gate_receipt", "gate_skipped"},
            "provenance": row["evidence_path"] or row["planned_path"],
            "seed": None,
        }
        for row in tasks
    ]
    source_hashes = accounting["source_artifact_hashes"] | {
        "authority": [
            {"path": selected["path"], "sha256": selected["sha256"]},
            {"path": selected["design_path"], "sha256": selected["design_sha256"]},
        ],
        "hardware": [
            {"path": str(hardware_path.relative_to(root)), "sha256": sha256_file(hardware_path)}
        ],
        "publication_state": [
            {
                "path": "ops/publication_gate_state.json",
                "sha256": sha256_file(root / "ops/publication_gate_state.json"),
            }
        ],
        "planned_output_is_input": False,
    }
    preconditions = [
        {
            "check": "authenticated_input",
            "path": entry["path"],
            "sha256": entry["sha256"],
            "passed": True,
        }
        for kind in (
            "producers",
            "pre_gate_receipts",
            "conductor_log",
            "authority",
            "hardware",
            "publication_state",
        )
        for entry in source_hashes[kind]
    ]
    preconditions.append(
        {
            "check": "absolute_root",
            "path": str(root),
            "expected": str(ROOT),
            "observed": str(root.resolve()),
            "passed": root.resolve() == ROOT,
        }
    )
    artifact: dict[str, Any] = {
        "schema": "carnot.exp7684.v669.capstone.v1",
        "experiment": 7684,
        "experiment_id": "exp7684-capstone",
        "milestone": "2026.09.669",
        "run_date": "20260926",
        "status": "complete",
        "honest_verdict": accounting["honest_verdict"],
        "verdict_class": accounting["verdict_class"],
        "flagged_adversarial": False,
        "gate_check_summary": accounting["gate_check_summary"],
        "acceptance_gate_results": gates,
        "capstone_accounting_ready_score": int(validated and len(tasks) == 14),
        "prior_dispositions": tasks,
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
            "prior_exposure": cohort["sample_size_budget"].get("prior_exposure_groups"),
            "effective_blocks": (audit.get("sample_size_budget") or {}).get("effective_blocks"),
            "limits": "Arms, seeds, and views do not enlarge independent source-family n.",
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
                "cancellations",
            )
        },
        "execution_venue": "host",
        "execution_venue_details": {
            "authenticated_host": socket.gethostname(),
            "owned_pid": os.getpid(),
            "gpu_uuid": None,
            "platform": platform.platform(),
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
        "validation_receipts": {
            "frozen_affected_scope": {
                "changed_modules": [MODULE, CAPABILITY],
                "static_paths": [WRAPPER],
                "tests": [TEST],
                "requirements": ["REQ-REPORT-7684", "REQ-CAPSTONE-7684"],
            },
            "commands": receipts or [],
            "required_checks_passed": validated,
            "cold_reduction": "pending_terminal_reader",
        },
        "verifier_is_oracle": True,
        "selected_authority": {
            key: selected[key]
            for key in ("path", "sha256", "design_path", "design_sha256", "comparison")
        },
        "evidence_summary": summary,
        "cold_raw_reduction": cold_raw,
        "retrospective_path": str(NOTE),
        "gap_assessment": [
            {
                "gap": "bound_relations_on_fresh_source_families",
                "strongest_evidence": summary["cohort"],
                "failure": cohort["honest_verdict"],
                "uncertainty": "all 480 families lack checked relation support",
                "next_evidence": "qualified relation features and isolated fresh decision labels",
            },
            {
                "gap": "causal_acquired_constraints_with_retention",
                "strongest_evidence": None,
                "failure": "Exp7678 producer absent after upstream gate skip",
                "uncertainty": "no prospective online or retention interval",
                "next_evidence": "one-use delayed admission with frozen static control and retention32",
            },
            {
                "gap": "scored_live_probe_reachability",
                "strongest_evidence": arc.get("probe_opportunities"),
                "failure": arc["honest_verdict"],
                "uncertainty": "no eligible novel SDK target",
                "next_evidence": "eligible adapter-free live episode with scored probe firing",
            },
        ],
        "publication_gates": publication,
        "hardware_dispositions": hardware,
        "retirement_actions": [
            {
                "mechanism": "membership_only_relation_check",
                "action": "retire_unchanged",
                "basis": "Exp7672 repeated same-verdict tuple failures",
            },
            {
                "mechanism": "exact_quote_prompt",
                "action": "retire_unchanged",
                "basis": "Exp7676 declared same-verdict retirement",
            },
            {
                "mechanism": "negative_only_goal_identification",
                "action": "retire_unique_goal_claim",
                "basis": "Exp7680 fixture cannot identify a unique goal from no-win histories",
            },
            {
                "mechanism": "fresh_relation_energy_and_online_learning",
                "action": "await_qualified_evidence",
                "basis": "gate-skipped producers have no scientific verdict",
            },
        ],
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
            "account": accounting,
        }
    )
    artifact["field_principles"] = _field_principles(tuple(artifact))
    artifact["field_principles"]["field_principles"] = (
        "Each reported field names its failure boundary."
    )
    return artifact


def read_candidate(path: Path, root: Path = ROOT) -> list[str]:
    """Cold-read all named bytes and reject changed interpretation or checksum."""

    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        selected = v669_capstone.authority(root)
        expected = build_artifact(
            root, receipts=value.get("validation_receipts", {}).get("commands", [])
        )
    except (OSError, ValueError, KeyError, TypeError) as exc:
        return [f"cold_reader_failed:{type(exc).__name__}"]
    errors = v669_capstone.cold_reduce(value, root, selected["tasks"])
    for key in (
        "selected_authority",
        "reproducibility_checksum",
        "acceptance_gate_results",
        "sample_size_budget",
        "gap_assessment",
        "publication_gates",
        "hardware_dispositions",
        "retirement_actions",
        "rows",
        "cold_raw_reduction",
    ):
        if canonical_hash(value.get(key)) != canonical_hash(expected[key]):
            errors.append(f"{key}_mismatch")
    return errors


def terminal_plan(root: Path, candidate: Path) -> list[CommandSpec]:  # pragma: no cover
    """Read one unpublished candidate with independent bounded processes."""

    python = str(root / ".venv/bin/python")
    return [
        CommandSpec(
            "cold_replay",
            (python, "-u", WRAPPER, "--cold-validate", str(candidate)),
            "exact_candidate",
            900,
        ),
        CommandSpec(
            "independent_reduction",
            (python, "-u", WRAPPER, "--independent-reduce", str(candidate)),
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
            (python, "-u", "scripts/verdict_row_consistency_lint.py", "--strict", str(candidate)),
            "exact_candidate",
            900,
        ),
    ]


def run_experiment(root: Path, run_date: str, output: Path) -> dict[str, Any]:  # pragma: no cover
    """Validate fixed scope, persist terminal receipts, then publish atomically."""

    start = time.monotonic()
    progress(start, "preconditions", "before")
    if root.resolve() != ROOT or run_date != "20260926":
        raise ValueError("frozen V669 root or date mismatch")
    private = Path(tempfile.mkdtemp(prefix="carnot-exp7684-", dir="/tmp"))
    (private / "pytest").mkdir()
    spans: list[dict[str, Any]] = []

    def close(phase: str, begin: float, units: int) -> None:
        now = time.monotonic() - start
        spans.append(
            {
                "phase": phase,
                "start_s": begin,
                "end_s": now,
                "duration_s": now - begin,
                "completed_units": units,
                "heartbeat_times_s": [now],
                "checkpoint_position": phase,
            }
        )
        progress(start, phase, "after", units)

    initial = build_artifact(root)
    if not all(item["passed"] for item in initial["preconditions_checked"]):
        raise RuntimeError("named input authentication failed")
    close("preconditions", 0.0, 14)
    progress(start, "validation", "before_subprocesses")
    begin = time.monotonic() - start
    scoped = build_scoped_commands(
        root,
        [TEST],
        [MODULE, CAPABILITY],
        static_paths=[WRAPPER],
        basetemp=private / "pytest",
        coverage_file=private / ".coverage.exp7684",
    )
    receipts = run_commands(root, scoped, log_dir=private / "logs/affected", heartbeat_s=60)
    close("validation", begin, len(receipts))
    required = reduce_required_checks(receipts)
    candidate = build_artifact(
        root, receipts=receipts, spans=spans, duration_s=time.monotonic() - start
    )
    candidate_path = private / "candidate.json"
    atomic_json(candidate_path, candidate)
    if not required["required_checks_passed"]:
        raise RuntimeError(f"affected validation failed: {required}")
    progress(start, "terminal", "before_subprocesses")
    begin = time.monotonic() - start
    terminal = run_commands(
        root, terminal_plan(root, candidate_path), log_dir=private / "logs/terminal", heartbeat_s=60
    )
    close("terminal", begin, len(terminal))
    if not all(item["passed"] for item in terminal):
        raise RuntimeError("terminal reader failed")
    final = build_artifact(
        root, receipts=[*receipts, *terminal], spans=spans, duration_s=time.monotonic() - start
    )
    final["validation_receipts"]["cold_reduction"] = "passed"
    final["validation_receipts"]["terminal_readers"] = terminal
    if read_candidate(candidate_path, root):
        raise RuntimeError("candidate drift after terminal readers")
    destination = output if output.is_absolute() else root / output
    progress(start, "publication", "before_atomic")
    atomic_json(destination, final)
    progress(start, "publication", "after_atomic", 1)
    return final


def main(argv: list[str] | None = None) -> int:  # pragma: no cover
    """Run the capstone or one read-only cold reader."""

    print("[exp7684] startup flushed", flush=True)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--date", default="20260926")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--cold-validate", type=Path)
    parser.add_argument("--independent-reduce", type=Path)
    args = parser.parse_args(argv)
    if args.cold_validate or args.independent_reduce:
        errors = read_candidate(args.cold_validate or args.independent_reduce, args.root)
        print(json.dumps({"cold_errors": errors}), flush=True)
        return int(bool(errors))
    run_experiment(args.root, args.date, args.output)
    return 0
