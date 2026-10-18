"""Publish an independent V669 audit while keeping absent science visible.

REQ-REPORT-7679; SCENARIO-REPORT-7679-CUSTODY/ROWS/TERMINAL.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import socket
import tempfile
import time
from typing import Any

from carnot.reporting import experiment_7303_validation_scope as checks
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash, sha256_file
from carnot.reporting.independent_evidence_audit_v669 import (
    PRODUCERS,
    audit_cohort,
    audit_fixture,
    audit_quotes,
    check_private_corruptions,
    failed_check,
    inventory,
)


ROOT = Path(__file__).resolve().parents[2]
RAW = Path("results/raw/experiment_7679_v669_independent_evidence_audit")
RESULT = Path("results/experiment_7679_v669_independent_evidence_audit.json")
MANIFEST = {
    "test_paths": ["tests/python/test_experiment_7679_v669_independent_evidence_audit.py"],
    "changed_modules": [
        "python/carnot/reporting/independent_evidence_audit_v669.py",
        "python/carnot/experiment_7679_v669_independent_evidence_audit.py",
    ],
    "static_paths": ["scripts/experiments/experiment_7679_v669_independent_evidence_audit.py"],
    "requirements": ["REQ-REPORT-7679", "REQ-CONTINUOUS-7679"],
}
ROLES = ("fit", "retention", "tune", "policy", "online_update", "online_admission", "evaluation")
ROW_KEYS = (
    "unit_id",
    "independent_unit",
    "arm",
    "evidence_stage",
    "role",
    "dialect",
    "split",
    "population",
    "truth",
    "observed",
    "source_sha256",
    "answer_sha256",
    "source_group_id",
    "original_source_sha256",
    "provenance",
    "raw_metrics",
    "metrics",
    "denominator",
    "numerator",
    "unknown_claims",
    "checked_relations",
    "output_tokens",
    "generation_s",
    "censored",
    "excluded",
    "prior_exposure",
    "claim_limit",
)


def progress(started: float, phase: str, event: str, **details: Any) -> None:
    """A flushed boundary makes owned work visible during long validation."""
    print(
        f"[exp7679] {phase} {event} elapsed_s={time.monotonic() - started:.3f} {details}",
        flush=True,
    )


def _read_json(
    root: Path, label: str, upstream: str, hashes: dict[str, Any], blocked: list[dict[str, Any]]
) -> Any:
    """Hash exact raw bytes, and name a missing or malformed store precisely."""
    path = root / label
    if not path.is_file():
        blocked.append(failed_check("raw_exists", upstream, path, "exists", True, False))
        hashes["missing_evidence"].append(label)
        return None
    hashes["raw_stores"][label] = sha256_file(path)
    try:
        if path.suffix == ".jsonl":
            return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        return json.loads(path.read_text())
    except ValueError as error:
        blocked.append(
            failed_check("raw_json", upstream, path, "json_valid", True, type(error).__name__)
        )
        return None


def reduce_available(
    root: Path, hashes: dict[str, Any], blocked: list[dict[str, Any]], started: float
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, bool]]:
    """Audit each present source independently so a gap cannot erase other rows."""
    all_rows: list[dict[str, Any]] = []
    summary: dict[str, Any] = {}
    raw = _read_json(
        root,
        "results/raw/experiment_7672_v669_bound_relations/rows.json",
        "Exp7672",
        hashes,
        blocked,
    )
    fixture: list[dict[str, Any]] = []
    if raw is not None:
        try:
            fixture, summary["fixture"] = audit_fixture(raw)
            all_rows.extend(fixture)
        except (KeyError, TypeError, ValueError) as error:
            blocked.append(
                failed_check(
                    "raw_contract",
                    "Exp7672",
                    root / "results/raw/experiment_7672_v669_bound_relations/rows.json",
                    "paired_rows",
                    "valid",
                    str(error),
                )
            )
    progress(started, "reduction", "unit_complete", unit="fixture", rows=len(fixture))
    base = "results/raw/experiment_7673_v669_fresh_relation_cohort"
    protocol = _read_json(root, f"{base}/protocol.json", "Exp7673", hashes, blocked)
    by_role: dict[str, Any] = {}
    rosters: dict[str, Any] = {}
    model_inputs: dict[str, Any] = {}
    cohort: list[dict[str, Any]] = []
    for role in ROLES:
        features = _read_json(root, f"{base}/{role}_features.jsonl", "Exp7673", hashes, blocked)
        evaluator = _read_json(
            root, f"{base}/{role}_evaluator_store.jsonl", "Exp7673", hashes, blocked
        )
        model = _read_json(root, f"{base}/{role}_model_inputs.jsonl", "Exp7673", hashes, blocked)
        if protocol is None or features is None or evaluator is None or model is None:
            continue
        try:
            declared = protocol["roles"][role]["families"]
            observed = [r["family_id"] for r in evaluator]
            if declared != observed:
                raise ValueError("evaluator family roster differs from frozen protocol")
            units = list(dict.fromkeys(r["unit_id"] for r in features))
            if len(units) != len(declared) or len(model) != len(declared):
                raise ValueError("source view or model input count differs from family roster")
            if any(item.get("role") != role for item in evaluator + model):
                raise ValueError("evaluator or model input role changed")
            by_role[role], rosters[role], model_inputs[role] = features, units, model
        except (KeyError, TypeError, ValueError) as error:
            blocked.append(
                failed_check(
                    "role_custody",
                    "Exp7673",
                    root / f"{base}/{role}_features.jsonl",
                    "family_role_roster",
                    "frozen",
                    str(error),
                )
            )
        progress(
            started, "reduction", "unit_complete", unit=role, groups=len(rosters.get(role, []))
        )
    if by_role:
        try:
            cohort, summary["cohort"] = audit_cohort(by_role, rosters, model_inputs)
            all_rows.extend(cohort)
        except (KeyError, TypeError, ValueError) as error:
            blocked.append(
                failed_check(
                    "raw_contract",
                    "Exp7673",
                    root / base / "protocol.json",
                    "paired_roles",
                    "valid",
                    str(error),
                )
            )
    quote_label = "results/raw/experiment_7676_v669_qwen_quote_relations/rows.jsonl"
    raw_quotes = _read_json(root, quote_label, "Exp7676", hashes, blocked)
    quotes: list[dict[str, Any]] = []
    if raw_quotes is not None:
        try:
            for row in raw_quotes:
                for field in ("request_path", "raw_response_path"):
                    path = Path(row[field])
                    declared = row[
                        "request_sha256" if field == "request_path" else "raw_response_sha256"
                    ]
                    actual = sha256_file(path) if path.is_file() else None
                    if actual != declared:
                        blocked.append(
                            failed_check("raw_custody", "Exp7676", path, "sha256", declared, actual)
                        )
            quotes, summary["quotes"] = audit_quotes(raw_quotes)
            all_rows.extend(quotes)
        except (KeyError, TypeError, ValueError) as error:
            blocked.append(
                failed_check(
                    "raw_contract",
                    "Exp7676",
                    root / quote_label,
                    "paired_rows",
                    "valid",
                    str(error),
                )
            )
    progress(started, "reduction", "unit_complete", unit="quotes", rows=len(quotes))
    compact = [{k: row[k] for k in ROW_KEYS if k in row} for row in all_rows]
    mutations = (
        check_private_corruptions(raw, by_role["online_admission"], raw_quotes)
        if raw and by_role.get("online_admission") and raw_quotes
        else {"not_run_missing_input": False}
    )
    return compact, summary, mutations


def _gates(summary: dict[str, Any], blocked: list[dict[str, Any]]) -> dict[str, Any]:
    """Keep protocol diagnostics apart from unobserved quality and cost gates."""
    fixture = summary.get("fixture", {})
    cohort = summary.get("cohort", {})
    quotes = summary.get("quotes", {})
    operands = {
        "validity": {"failed_contract_checks": len(blocked)},
        "readiness": {
            "fixture_correct_groups": fixture.get("fixture_correct_groups"),
            "cohort_groups": cohort.get("independent_groups"),
        },
        "coverage": {
            "fixture_groups": fixture.get("independent_groups"),
            "cohort_groups": cohort.get("independent_groups"),
            "qwen_groups": quotes.get("independent_groups"),
            "required_static_and_online_groups": None,
        },
        "freshness": {
            "fresh_confirmatory_static_groups": None,
            "fresh_confirmatory_online_groups": None,
        },
        "probability": {
            "paired_fresh_probabilities": None,
            "proper_loss": None,
            "simultaneous_lower_bound": None,
        },
        "decision_utility": {
            "paired_actions": None,
            "source_controls": None,
            "cost_difference": None,
        },
        "retention": {
            "release_order_events": None,
            "one_use_admissions": None,
            "restart_replays": None,
        },
        "efficiency": {
            "online_comparison_tokens": None,
            "online_comparison_duration_s": None,
            "paired_retention_confirmed": None,
        },
    }
    principles = {
        "validity": "Only authenticated producers and raw rows can support an audit gate.",
        "readiness": "Fixture and cohort diagnostics do not establish learned benefit.",
        "coverage": "One source family counts once across arms, seeds, and views.",
        "freshness": "Exposed pilots and exact fixtures cannot confirm fresh natural benefit.",
        "probability": "Proper loss requires paired fresh probabilities and independent labels.",
        "decision_utility": "Measured actions, error costs, and source controls govern utility.",
        "retention": "Release order, one-use admission, and replay govern retention.",
        "efficiency": "Online cost and retained quality must both be reproduced.",
    }
    return {
        name: {"passed": False, "measured_operands": value, "principle": principles[name]}
        for name, value in operands.items()
    }


def build_artifact(
    root: Path,
    date: str,
    rows: list[dict[str, Any]],
    summary: dict[str, Any],
    hashes: dict[str, Any],
    blocked: list[dict[str, Any]],
    mutations: dict[str, bool],
) -> dict[str, Any]:
    """Expose available cells without turning an absent producer into a null."""
    complete = (
        not blocked
        and all(mutations.values())
        and all(key in summary for key in ("fixture", "cohort", "quotes", "static", "online"))
    )
    source_classes = {
        number: (
            "terminal"
            if label in hashes["producers"]
            else "pre_gate"
            if label in hashes["pre_gate_receipts"]
            else "absent"
        )
        for number, label in PRODUCERS.items()
    }
    intended = {"fixture": 80, "cohort": 480, "quotes": 24, "static": None, "online": None}
    observed = {key: summary.get(key, {}).get("independent_groups") for key in intended}
    health_path = root / RAW / "validation/repository_health/full_suite_debt.json"
    health = [json.loads(health_path.read_text())] if health_path.is_file() else []
    artifact = {
        "schema": "carnot.exp7679.independent_evidence_audit.v1",
        "experiment_id": 7679,
        "milestone": "2026.09.669",
        "run_date": date,
        "honest_verdict": (
            "complete_null_no_independent_benefit"
            if complete
            else "complete_blocked_required_static_online_evidence"
        ),
        "verdict_class": "null" if complete else "blocked",
        "flagged_adversarial": False,
        "gate_check_summary": blocked,
        "acceptance_gate_results": _gates(summary, blocked),
        "rows": rows,
        "audit_rows_path": str(RAW / "rows.jsonl"),
        "sample_size_budget": {
            "intended_independent_groups": intended,
            "observed_independent_groups": observed,
            "eligible": {
                key: (
                    None
                    if observed[key] is None
                    else observed[key] - summary.get(key, {}).get("excluded_groups", 0)
                )
                for key in intended
            },
            "excluded": {key: summary.get(key, {}).get("excluded_groups") for key in intended},
            "censored": {key: summary.get(key, {}).get("censored_groups") for key in intended},
            "prior_exposure": {
                "fixture_oracle": 72,
                "pilot_fixture": 8,
                "qwen_exposed": summary.get("quotes", {}).get("exposed_groups"),
            },
            "effective_blocks": {
                "fixture": observed["fixture"],
                "cohort": observed["cohort"],
                "quotes": observed["quotes"],
                "static": None,
                "online": None,
            },
            "limits": "Views, arms, and seeds never increase the independent family count.",
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
            "authenticated_host": socket.gethostname(),
            "host": socket.gethostname(),
            "gpu_uuid": None,
            "owned_pid": os.getpid(),
        },
        "phase_spans": [],
        "duration_s": 0.0,
        "random_seed": {"seed": 7679, "purpose": "deterministic source order; no resampling"},
        "source_artifact_hashes": hashes,
        "preconditions_checked": {
            "root": str(root.resolve()),
            "producer_slots": len(PRODUCERS),
            "producer_classes": source_classes,
            "resource": "authenticated local bytes and host CPU",
            "external_absence_is_blocked": True,
        },
        "validation_receipts": {
            "frozen_affected_scope": MANIFEST,
            "required_commands": [],
            "terminal_readers": [],
            "required_checks_passed": False,
            "unrelated_full_suite_debt": health,
        },
        "verifier_is_oracle": True,
        "field_principles": {
            name: gate["principle"] for name, gate in _gates(summary, blocked).items()
        }
        | {
            "honest_verdict": "Completion is separate from measured scientific benefit.",
            "rows": "Raw independent units retain arm, source, exclusion, and censoring.",
            "source_artifact_hashes": "Immutable source bytes bind all dispositions.",
            "inference_substrate": "Only current CPU aggregation defines this task's substrate.",
        },
        "independent_audit_complete_score": int(complete),
        "independent_quality_confirmed_score": 0,
        "independent_efficiency_confirmed_score": 0,
        "independent_quality_confirmed_gates": [],
        "independent_reduction": summary,
        "private_mutation_results": mutations,
        "evidence_dispositions": [
            {
                "task": f"Exp{number}",
                "path": label,
                "availability": source_classes[number],
                "custody": "byte_hashed" if source_classes[number] != "absent" else "missing",
                "circularity": "exact_fixture_oracle"
                if number == 7672
                else "injected_answer_labels"
                if number == 7673
                else "none_claimed",
                "freshness": "exposed_or_fixture" if number in (7672, 7676) else "unconfirmed",
                "row_consistency": "cold_reduced"
                if number in (7672, 7673, 7676)
                else "unavailable",
                "claim_limit": "No independent static or online benefit confirmed.",
            }
            for number, label in PRODUCERS.items()
        ],
        "retirement_decision": {
            "mechanism": "static or online relation-energy decision",
            "same_verdict_recurrence": False,
            "resource_absence_distinct_from_science": True,
            "retired": False,
        },
        "activation": False,
        "production_promotion": False,
    }
    artifact["reproducibility_checksum"] = canonical_hash(
        {
            "inputs": hashes,
            "config": {"manifest": MANIFEST, "seed": 7679},
            "reducer_code": sha256_file(
                root / "python/carnot/reporting/independent_evidence_audit_v669.py"
            ),
        }
    )
    return artifact


def cold_replay(candidate: Path) -> list[str]:
    """A fresh process reauthenticates raw bytes and recomputes every available cell."""
    value = json.loads(candidate.read_text())
    root = Path(value["preconditions_checked"]["root"])
    found, blocked, hashes = inventory(root)
    rows, summary, mutations = reduce_available(root, hashes, blocked, time.monotonic())
    errors = []
    if blocked != value["gate_check_summary"]:
        errors.append("gate operands changed")
    if hashes != value["source_artifact_hashes"]:
        errors.append("source bytes changed")
    if summary != value["independent_reduction"]:
        errors.append("raw reduction changed")
    if rows != value["rows"]:
        errors.append("independent rows changed")
    if mutations != value["private_mutation_results"]:
        errors.append("private corruption result changed")
    if len(found) != sum(
        1
        for label in PRODUCERS.values()
        if label in hashes["producers"] or label in hashes["pre_gate_receipts"]
    ):
        errors.append("producer roster changed")
    expected = canonical_hash(
        {
            "inputs": hashes,
            "config": {"manifest": MANIFEST, "seed": 7679},
            "reducer_code": sha256_file(
                root / "python/carnot/reporting/independent_evidence_audit_v669.py"
            ),
        }
    )
    if value["reproducibility_checksum"] != expected:
        errors.append("reproducibility checksum changed")
    return errors


def terminal_commands(root: Path, candidate: Path) -> list[checks.CommandSpec]:
    """All readers consume the same immutable candidate bytes."""
    python = str(root / ".venv/bin/python")
    module = "carnot.experiment_7679_v669_independent_evidence_audit"
    return [
        checks.CommandSpec(
            "fresh_process_cold_replay",
            (python, "-u", "-m", module, "--cold", str(candidate)),
            "exact_candidate",
            180,
        ),
        checks.CommandSpec(
            "independent_raw_reduction",
            (python, "-u", "-m", module, "--independent", str(candidate)),
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


def run_experiment(root: Path, date: str, output: Path) -> dict[str, Any]:
    """Run bounded local checks and publish only after exact-candidate readers."""
    root = root.resolve()
    started = time.monotonic()
    spans: list[dict[str, Any]] = []
    previous = 0.0

    def span(name: str, units: int, checkpoint: str) -> None:
        nonlocal previous
        now = time.monotonic() - started
        spans.append(
            {
                "phase": name,
                "start_s": previous,
                "end_s": now,
                "completed_units": units,
                "checkpoint_position": checkpoint,
                "heartbeat_times_s": [now],
            }
        )
        previous = now

    progress(started, "preconditions", "start", root=str(root))
    found, blocked, hashes = inventory(root)
    raw_dir = root / RAW
    raw_dir.mkdir(parents=True, exist_ok=True)
    atomic_json(raw_dir / "frozen_affected_scope.json", MANIFEST)
    progress(started, "preconditions", "complete", producers=len(found), blocked=len(blocked))
    span("preconditions", len(found), "source_inventory")
    progress(started, "reduction", "start")
    rows, summary, mutations = reduce_available(root, hashes, blocked, started)
    with (raw_dir / "rows.jsonl").open("w", encoding="utf-8") as stream:
        for index, row in enumerate(rows):
            stream.write(json.dumps(row, sort_keys=True) + "\n")
            if (index + 1) % 100 == 0:
                progress(started, "reduction", "heartbeat", completed_units=index + 1)
        stream.flush()
        os.fsync(stream.fileno())
    atomic_json(
        raw_dir / "checkpoint.json",
        {
            "completed_units": len(rows),
            "summary": summary,
            "row_sha256": sha256_file(raw_dir / "rows.jsonl"),
        },
    )
    artifact = build_artifact(root, date, rows, summary, hashes, blocked, mutations)
    progress(started, "reduction", "complete", rows=len(rows), blocked=len(blocked))
    span("reduction", len(rows), "rows_reduced")
    private = Path(tempfile.mkdtemp(prefix="exp7679-", dir="/tmp"))
    basetemp = private / "basetemp"
    basetemp.mkdir()
    progress(started, "affected_validation", "start")
    commands = checks.build_scoped_commands(
        root,
        MANIFEST["test_paths"],
        MANIFEST["changed_modules"],
        static_paths=MANIFEST["static_paths"],
        basetemp=basetemp,
        coverage_file=private / "coverage.data",
    )
    receipts = checks.run_commands(
        root,
        commands,
        log_dir=raw_dir / "validation/affected",
        extra_env={"COVERAGE_FILE": str(private / "coverage.data"), "JAX_PLATFORMS": "cpu"},
    )
    result = checks.reduce_required_checks(receipts)
    artifact["validation_receipts"].update(result)
    artifact["validation_receipts"]["required_commands"] = receipts
    progress(started, "affected_validation", "complete", passed=result["required_checks_passed"])
    span("affected_validation", len(receipts), "affected_receipts")
    if not result["required_checks_passed"]:
        artifact["honest_verdict"] = "complete_disqualified_required_validation"
        artifact["verdict_class"] = "disqualified"
        artifact["independent_audit_complete_score"] = 0
    artifact["phase_spans"] = spans
    artifact["duration_s"] = previous
    candidate = raw_dir / "exact_terminal_candidate.json"
    atomic_json(candidate, artifact)
    progress(started, "terminal_readers", "start", candidate=str(candidate))
    terminal = checks.run_commands(
        root,
        terminal_commands(root, candidate),
        log_dir=raw_dir / "validation/terminal",
        extra_env={"JAX_PLATFORMS": "cpu"},
    )
    artifact["validation_receipts"]["terminal_readers"] = terminal
    artifact["validation_receipts"]["exact_terminal_candidate_sha256"] = sha256_file(candidate)
    artifact["flagged_adversarial"] = not next(
        row for row in terminal if row["name"] == "adversarial_verify"
    )["passed"]
    if not all(row["passed"] for row in terminal):
        artifact["honest_verdict"] = "complete_disqualified_terminal_reader"
        artifact["verdict_class"] = "disqualified"
        artifact["independent_audit_complete_score"] = 0
    progress(started, "terminal_readers", "complete", passed=all(row["passed"] for row in terminal))
    span("terminal_readers", len(terminal), "exact_candidate_readers")
    artifact["phase_spans"] = spans
    artifact["duration_s"] = previous
    destination = output if output.is_absolute() else root / output
    atomic_json(destination, artifact)
    progress(
        started, "publication", "complete", output=str(destination), sha256=sha256_file(destination)
    )
    return artifact


def main(argv: list[str] | None = None) -> int:
    """Keep the entrypoint thin and expose fresh-process replay switches."""
    print("[exp7679] startup flushed", flush=True)
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", default="20260926")
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
    raise SystemExit(main())
