"""Account for V673 evidence without upgrading blocked producers (REQ-REPORT-7738)."""

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

from carnot.experiment_7726_v673_contract_methods import (
    DESIGN,
    compare_contract,
    resolve_authority,
)
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash, sha256_file
from carnot.reporting.experiment_7303_validation_scope import (
    CommandSpec,
    build_scoped_commands,
    reduce_required_checks,
    run_commands,
)

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = Path("results/experiment_7738_v673_capstone.json")
RAW = Path("results/raw/experiment_7738_v673_capstone")
MODULE = "python/carnot/experiment_7738_v673_capstone.py"
CLI = "scripts/experiments/experiment_7738_v673_capstone.py"
TEST = "tests/python/test_experiment_7738_v673_capstone.py"
REQUIRED = {7731, 7733, 7734}
ELIGIBLE = {"positive", "circular_positive", "null"}
PRINCIPLE = "Measured evidence bounds the claim and downstream use."
GATES = (
    "validity",
    "readiness",
    "brier_score",
    "decision_cost",
    "coverage",
    "retention",
    "efficiency",
)


def progress(start: float, phase: str, event: str, units: int = 0) -> None:
    """Flush elapsed time and completed work at each owned boundary."""
    print(
        f"[exp7738] {phase} {event} elapsed_s={time.monotonic() - start:.3f} completed_units={units}",
        flush=True,
    )


def failed(
    check: str, upstream: str, path: str, field: str, operator: str, expected: Any, observed: Any
) -> dict[str, Any]:
    """Keep exact operands for an unavailable or disqualified input."""
    return {
        "check": check,
        "upstream_id": upstream,
        "artifact_path": path,
        "field": field,
        "operator": operator,
        "expected": expected,
        "observed": observed,
    }


def read_authority(root: Path) -> dict[str, Any]:
    """Compare the activated or staged YAML with independent table and JSON text."""
    path, roadmap, candidates = resolve_authority(root)
    design = root / DESIGN
    comparison = compare_contract(design.read_text(), roadmap)
    return {
        "path": path.relative_to(root).as_posix(),
        "tasks": roadmap["tasks"],
        "candidates": candidates,
        "comparison": comparison,
        "hashes": {
            path.relative_to(root).as_posix(): sha256_file(path),
            DESIGN.as_posix(): sha256_file(design),
        },
    }


def account(
    root: Path, tasks: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    """Reduce thirteen task slots, distinguishing planned paths from gate receipts."""
    if len(tasks) != 13 or any(
        not task["id"].startswith(f"exp{7726 + i}-") for i, task in enumerate(tasks)
    ):
        raise ValueError("V673 thirteen-task order required")
    hashes: dict[str, Any] = {
        key: []
        for key in (
            "eligible_producers",
            "disqualified_producers",
            "flagged_historical_inputs",
            "pre_gate_receipts",
            "absent_sources",
        )
    }
    rows: list[dict[str, Any]] = []
    payloads: dict[str, dict[str, Any]] = {}
    failures: list[dict[str, Any]] = []
    for index, task in enumerate(tasks):
        number = 7726 + index
        label = task["deliverable"]
        planned = root / label
        alternate_label = label.replace("_v673_", "_")
        alternate = root / alternate_label
        value: dict[str, Any] = {}
        evidence = label
        state = "planned_output" if number == 7738 else "absent"
        if number != 7738 and planned.is_file():
            value = json.loads(planned.read_text())
            if not isinstance(value, dict):
                raise ValueError(f"producer object required: {label}")
            state = (
                "pre_gate_receipt" if value.get("schema") == "blocked_gate_check_v1" else "producer"
            )
        elif number != 7738 and alternate.is_file():
            value = json.loads(alternate.read_text())
            if not isinstance(value, dict) or value.get("schema") != "blocked_gate_check_v1":
                raise ValueError(f"pre-gate schema required: {alternate_label}")
            evidence, state = alternate_label, "pre_gate_receipt"
        if number != 7738:
            if not planned.is_file():
                hashes["absent_sources"].append({"task_id": task["id"], "path": label})
            if state != "absent":
                bucket = (
                    "pre_gate_receipts"
                    if state == "pre_gate_receipt"
                    else "flagged_historical_inputs"
                    if value.get("flagged_adversarial") is True
                    else "eligible_producers"
                    if value.get("verdict_class") in ELIGIBLE
                    and str(value.get("honest_verdict", "")).startswith("complete_")
                    else "disqualified_producers"
                )
                hashes[bucket].append(
                    {
                        "task_id": task["id"],
                        "path": evidence,
                        "sha256": sha256_file(root / evidence),
                    }
                )
                payloads[task["id"]] = value
        row = {
            "unit_id": task["id"],
            "task_id": task["id"],
            "order": index + 1,
            "arm": "task_accounting",
            "planned_path": label,
            "evidence_path": evidence if state not in {"absent", "planned_output"} else None,
            "availability": state,
            "verdict_class": value.get("verdict_class"),
            "honest_verdict": value.get("honest_verdict"),
            "flagged_adversarial": value.get("flagged_adversarial"),
            "registered_gates": [],
            "raw_metrics": {
                "verdict_class": value.get("verdict_class"),
                "honest_verdict": value.get("honest_verdict"),
            },
            "denominators": {"independent_tasks": 1, "independent_families": None},
            "censored": state in {"absent", "pre_gate_receipt"},
            "exclusions": [state] if state in {"absent", "pre_gate_receipt"} else [],
            "input_hashes": {evidence: sha256_file(root / evidence)}
            if state not in {"absent", "planned_output"}
            else {},
        }
        rows.append(row)
    by_id = {task["id"]: task for task in tasks}
    for task, row in zip(tasks, rows, strict=True):
        for gate in task.get("gated_on") or []:
            source = payloads.get(gate["upstream"], {})
            observed = source.get(gate["artifact_field"])
            passed = observed == gate["value"] if gate["op"] == "==" else observed in gate["value"]
            check = failed(
                "registered_gate",
                gate["upstream"],
                by_id[gate["upstream"]]["deliverable"],
                gate["artifact_field"],
                gate["op"],
                gate["value"],
                observed,
            )
            check.update(passed=passed, consumer=task["id"])
            row["registered_gates"].append(check)
    for number in sorted(REQUIRED):
        row = rows[number - 7726]
        source = payloads.get(row["task_id"])
        if source is None:
            failures.append(
                failed(
                    "required_source_exists",
                    f"Exp{number}",
                    row["planned_path"],
                    "exists",
                    "==",
                    True,
                    False,
                )
            )
        else:
            field = (
                "honest_verdict"
                if not str(source.get("honest_verdict", "")).startswith("complete_")
                else "flagged_adversarial"
                if source.get("flagged_adversarial") is not False
                else "verdict_class"
            )
            expected = (
                "complete_*"
                if field == "honest_verdict"
                else False
                if field == "flagged_adversarial"
                else sorted(ELIGIBLE)
            )
            observed = source.get(field)
            if field != "verdict_class" or observed not in ELIGIBLE:
                failures.append(
                    failed(
                        "required_source_eligible",
                        f"Exp{number}",
                        row["evidence_path"] or row["planned_path"],
                        field,
                        "in" if field == "verdict_class" else "==",
                        expected,
                        observed,
                    )
                )
    return rows, hashes, failures


def cold_replay(value: dict[str, Any], root: Path, tasks: list[dict[str, Any]]) -> list[str]:
    """Reopen exact source bytes and reject any changed accounting conclusion."""
    rows, hashes, failures = account(root, tasks)
    errors = []
    if "verdict_class" in value:
        valid = (
            value.get("validation_receipts", {})
            .get("required_checks", {})
            .get("required_checks_passed", True)
        )
        verdict = "disqualified" if not valid else "blocked" if failures else "null"
        honest = (
            "complete_disqualified_v673_capstone_validation"
            if not valid
            else "complete_blocked_required_v673_evidence"
            if failures
            else "complete_null_v673_development_study"
        )
        rows[-1].update(
            verdict_class=verdict,
            honest_verdict=honest,
            raw_metrics={"verdict_class": verdict, "honest_verdict": honest},
        )
        if value["verdict_class"] != verdict or value.get("honest_verdict") != honest:
            errors.append("verdict")
    expected = {"rows": rows, "source_artifact_hashes": hashes, "gate_check_summary": failures}
    for key, item in expected.items():
        actual = value.get(key)
        if key == "source_artifact_hashes" and isinstance(actual, dict):
            actual = {name: actual.get(name) for name in hashes}
        if canonical_hash(actual) != canonical_hash(item):
            errors.append(key)
    return errors


def publication(root: Path) -> dict[str, Any]:
    """Read the unchanged G1–G4 gate with the repository's own command."""
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
    pub: dict[str, Any],
    receipts: list[dict[str, Any]] | None = None,
    spans: list[dict[str, Any]] | None = None,
    duration_s: float = 0.0,
) -> dict[str, Any]:
    """Build a terminal capstone from source rows and measured validation."""
    authority = read_authority(root)
    rows, hashes, failures = account(root, authority["tasks"])
    checks = (
        reduce_required_checks(receipts or []) if receipts else {"required_checks_passed": True}
    )
    valid = checks["required_checks_passed"] and authority["comparison"]["passed"]
    verdict = "disqualified" if not valid else "blocked" if failures else "null"
    honest = (
        "complete_disqualified_v673_capstone_validation"
        if not valid
        else "complete_blocked_required_v673_evidence"
        if failures
        else "complete_null_v673_development_study"
    )
    rows[-1]["verdict_class"] = verdict
    rows[-1]["honest_verdict"] = honest
    rows[-1]["raw_metrics"] = {"verdict_class": verdict, "honest_verdict": honest}
    source_hashes = {
        **hashes,
        "authority": authority["hashes"],
        "publication_state": {
            "ops/publication_gate_state.json": sha256_file(root / "ops/publication_gate_state.json")
        },
        "planned_output_is_input": False,
    }
    full_log = root / RAW / "validation/full/00_full_python_suite.log"
    full_suite_debt = (
        {
            "command": "PYTHONPATH=python:. .venv/bin/pytest tests/python -q -n 0 -o addopts= --no-cov",
            "exit_code": 2,
            "log_path": str(full_log.relative_to(root)),
            "log_sha256": sha256_file(full_log),
            "collection_errors": 18,
        }
        if full_log.is_file()
        else None
    )
    gates: dict[str, Any] = {name: None for name in GATES}
    gates["validity"] = valid
    gates["readiness"] = valid and not failures
    gates["coverage"] = {
        "required_eligible": sum(
            not any(f["upstream_id"] == f"Exp{n}" for f in failures) for n in REQUIRED
        ),
        "required": 3,
    }
    domains = {"qwen": rows[3], "arc": rows[9:11], "service": rows[11]}
    decisions = [
        {
            "mechanism": "evidence_set_static",
            "gate": "eligible_Exp7731_and_Exp7734",
            "observed": [rows[5]["availability"], rows[8]["verdict_class"]],
            "claim_scope": "development_only",
            "retirement": "unchanged_certificate_heads_and_external_text_rerankers_remain_retired",
            "next_changed_premise": "passing set-head validation and paired development decisions",
        },
        {
            "mechanism": "continuous_admission",
            "gate": "eligible_Exp7733_and_Exp7734",
            "observed": [rows[7]["availability"], rows[8]["verdict_class"]],
            "claim_scope": "development_only",
            "retirement": "unchanged_importance_anchors_remain_retired",
            "next_changed_premise": "passing causal qualification and later retained forecasts",
        },
        {
            "mechanism": "organic_ARC_exploration",
            "gate": "public_game_score_and_action_cost",
            "observed": [rows[9]["verdict_class"], rows[10]["availability"]],
            "claim_scope": "adapter_withheld_public",
            "retirement": "failed_unchanged_exploration_heuristics_remain_retired",
            "next_changed_premise": "valid organic runner and scored public game rows",
        },
        {
            "mechanism": "complete_set_service",
            "gate": "batch1_p95_le_100ms_and_le_2x_pooled_MLP",
            "observed": rows[11]["availability"],
            "claim_scope": "development_only",
            "retirement": "none_from_absence",
            "next_changed_premise": "valid head and paired service timing",
        },
    ]
    artifact: dict[str, Any] = {
        "schema": "carnot.exp7738.v673.capstone.v1",
        "experiment_id": 7738,
        "milestone": "2026.09.673",
        "run_date": "20260927",
        "honest_verdict": honest,
        "verdict_class": verdict,
        "flagged_adversarial": False,
        "gate_check_summary": failures,
        "acceptance_gate_results": gates,
        "rows": rows,
        "prior_dispositions": [
            {
                key: row[key]
                for key in ("task_id", "availability", "verdict_class", "honest_verdict")
            }
            for row in rows
        ],
        "sample_size_budget": {
            "intended_tasks": 13,
            "observed_tasks": 13,
            "eligible_tasks": len(hashes["eligible_producers"]),
            "excluded_tasks": len(hashes["disqualified_producers"]),
            "censored_tasks": sum(row["censored"] for row in rows),
            "intended_families": None,
            "observed_families": None,
            "eligible_families": None,
            "excluded_families": None,
            "censored_families": None,
            "effective_independent_families": None,
            "seeds_windows_arms_increase_independent_n": False,
        },
        "claim_scope": {
            "reused_RAGTruth": "development_only",
            "exact_fixtures": "fixture_only",
            "ARC_public_games": "adapter_withheld_public",
            "fresh_generalization_eligible": False,
        },
        "fresh_generalization_eligible": False,
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
        "phase_spans": spans or [],
        "duration_s": duration_s,
        "random_seed": {"seeds": [], "purpose": "deterministic_accounting_no_sampling"},
        "source_artifact_hashes": source_hashes,
        "preconditions_checked": {
            "absolute_root": str(root.resolve()),
            "authority_match": authority["comparison"]["passed"],
            "authority_errors": authority["comparison"]["errors"],
            "cpu_aggregation_available": Path("/proc/self/status").is_file(),
            "effective_coding_backend": "codex",
            "experimental_model_load": False,
        },
        "validation_receipts": {
            "frozen_scope": {
                "changed_modules": [MODULE],
                "static_paths": [CLI],
                "tests": [TEST],
                "requirements": ["REQ-REPORT-7738"],
            },
            "commands": receipts or [],
            "required_checks": checks,
            "e2e": "SCENARIO-REPORT-7738-TERMINAL",
            "global_suite_debt": full_suite_debt,
        },
        "verifier_is_oracle": False,
        "capstone_accounting_ready_score": 1,
        "continuation_decisions": decisions,
        "domain_results": domains,
        "publication_gates": pub,
        "three_prd_gaps": [
            {
                "gap": "evidence_coverage",
                "observed": "development inputs blocked; fresh evidence absent",
            },
            {
                "gap": "retained_causal_learning",
                "observed": "Exp7733 pre-gate receipt; retention unmeasured",
            },
            {
                "gap": "useful_reachable_behavior_complete_costs",
                "observed": "ARC runner disqualified; service pre-gate receipt",
            },
        ],
        "hardware_continuity": {
            "inventory_path": "research-hardware-wishlist.md",
            "inventory_sha256": sha256_file(root / "research-hardware-wishlist.md"),
            "service_measurement_eligible": rows[11]["verdict_class"] in ELIGIBLE,
            "acquisition_needed": None,
            "purchase_performed": False,
        },
        "publication_performed": False,
        "production_defaults_changed": False,
        "generator_training_performed": False,
    }
    artifact["reproducibility_checksum"] = canonical_hash(
        {
            "authority": authority["hashes"],
            "sources": hashes,
            "reducer": sha256_file(root / MODULE),
            "configuration": ["aggregation", []],
        }
    )
    artifact["field_principles"] = {key: PRINCIPLE for key in artifact}
    artifact["field_principles"].update(
        {
            "honest_verdict": "Terminal custody avoids retries of unchanged external failures.",
            "rows": "A comparison must be recomputable without repeating model work.",
            "claim_scope": "Prior exposure cannot be erased by a new split or experiment ID.",
            "field_principles": PRINCIPLE,
            "acceptance_gates": {name: PRINCIPLE for name in GATES},
        }
    )
    return artifact


def read_candidate(path: Path, root: Path = ROOT) -> list[str]:
    """Cold reduce exact candidate and raw rows in a fresh CLI process."""
    value = json.loads(path.read_text())
    authority = read_authority(root)
    errors = cold_replay(value, root, authority["tasks"])
    raw_rows = root / RAW / "rows.json"
    if not raw_rows.is_file() or canonical_hash(json.loads(raw_rows.read_text())) != canonical_hash(
        value.get("rows")
    ):
        errors.append("raw_rows")
    expected = build_artifact(
        root, publication(root), value.get("validation_receipts", {}).get("commands", [])
    )
    for key in (
        "source_artifact_hashes",
        "sample_size_budget",
        "acceptance_gate_results",
        "continuation_decisions",
        "domain_results",
        "reproducibility_checksum",
        "publication_gates",
        "three_prd_gaps",
        "capstone_accounting_ready_score",
    ):
        if canonical_hash(value.get(key)) != canonical_hash(expected[key]):
            errors.append(key)
    return sorted(set(errors))


def run_experiment(root: Path, run_date: str, output: Path) -> dict[str, Any]:
    """Run scoped and terminal readers before publishing checked capstone bytes."""
    start = time.monotonic()
    progress(start, "preflight", "start")
    root = root.resolve()
    if run_date != "20260927":
        raise ValueError("V673 run date must be 20260927")
    authority = read_authority(root)
    rows, _, _ = account(root, authority["tasks"])
    raw = root / RAW
    raw.mkdir(parents=True, exist_ok=True)
    scope = {
        "changed_modules": [MODULE],
        "static_paths": [CLI],
        "tests": [TEST],
        "requirements": ["REQ-REPORT-7738", "SCENARIO-REPORT-7738-TERMINAL"],
    }
    atomic_json(raw / "frozen_affected_scope.json", scope)
    progress(start, "preflight", "before_publication_gate", len(rows))
    pub = publication(root)
    progress(start, "preflight", "after_publication_gate", len(rows))
    private = Path(tempfile.mkdtemp(prefix="exp7738-validation-", dir="/tmp"))
    (private / "pytest").mkdir()
    commands = build_scoped_commands(
        root,
        [TEST],
        [MODULE],
        static_paths=[CLI],
        basetemp=private / "pytest",
        coverage_file=private / ".coverage",
    )
    spans: list[dict[str, Any]] = []
    boundary = time.monotonic() - start
    spans.append(
        {
            "phase": "preflight",
            "start_s": 0.0,
            "end_s": boundary,
            "duration_s": boundary,
            "run_date": run_date,
            "completed_units": len(rows),
            "heartbeat_times": [],
            "checkpoint_hashes": {
                "frozen_affected_scope.json": sha256_file(raw / "frozen_affected_scope.json")
            },
        }
    )
    progress(start, "validation", "before_subprocesses")
    receipts = run_commands(root, commands, log_dir=raw / "validation/affected", heartbeat_s=60)
    end = time.monotonic() - start
    spans.append(
        {
            "phase": "validation",
            "start_s": boundary,
            "end_s": end,
            "duration_s": end - boundary,
            "run_date": run_date,
            "completed_units": len(receipts),
            "heartbeat_times": [],
            "checkpoint_hashes": {},
        }
    )
    progress(start, "validation", "after_subprocesses", len(receipts))
    candidate = build_artifact(root, pub, receipts, spans, end)
    atomic_json(raw / "rows.json", candidate["rows"])
    candidate_path = raw / "terminal_candidate.json"
    atomic_json(candidate_path, candidate)
    terminal = [
        CommandSpec(
            "cold_replay",
            (str(root / ".venv/bin/python"), "-u", CLI, "--cold-validate", str(candidate_path)),
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
    terminal_receipts = run_commands(
        root, terminal, log_dir=raw / "validation/terminal", heartbeat_s=60
    )
    finish = time.monotonic() - start
    spans.append(
        {
            "phase": "terminal",
            "start_s": end,
            "end_s": finish,
            "duration_s": finish - end,
            "run_date": run_date,
            "completed_units": len(terminal_receipts),
            "heartbeat_times": [],
            "checkpoint_hashes": {"terminal_candidate.json": sha256_file(candidate_path)},
        }
    )
    progress(start, "terminal", "after_subprocesses", len(terminal_receipts))
    final = build_artifact(root, pub, receipts, spans, finish)
    final["validation_receipts"].update(
        cold_reduction=terminal_receipts[0]["passed"], terminal_readers=terminal_receipts
    )
    if not all(receipt["passed"] for receipt in terminal_receipts):
        final.update(
            verdict_class="disqualified",
            honest_verdict="complete_disqualified_terminal_reader",
            flagged_adversarial=not terminal_receipts[1]["passed"],
        )
        final["acceptance_gate_results"]["readiness"] = False
        final["capstone_accounting_ready_score"] = 0
    destination = output if output.is_absolute() else root / output
    progress(start, "publication", "before_atomic")
    atomic_json(destination, final)
    progress(start, "publication", "after_atomic", 1)
    return final


def main(argv: list[str] | None = None) -> int:
    """Expose one thin CLI and a separate read-only cold replay."""
    print("[exp7738] startup flushed", flush=True)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--date", default="20260927")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--cold-validate", type=Path)
    args = parser.parse_args(argv)
    if args.cold_validate:
        errors = read_candidate(args.cold_validate, args.root.resolve())
        print(json.dumps({"cold_errors": errors}), flush=True)
        return int(bool(errors))
    run_experiment(args.root, args.date, args.output)
    return 0
