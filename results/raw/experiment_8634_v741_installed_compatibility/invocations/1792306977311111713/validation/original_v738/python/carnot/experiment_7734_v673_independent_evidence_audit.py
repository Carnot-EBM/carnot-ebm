"""Cold V673 evidence audit for REQ-REPORT-7734 and REQ-CL-7734-AUDIT."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import math
import os
from pathlib import Path
import socket
import tempfile
import time
from typing import Any

from carnot.reporting.current_work_receipt import atomic_json, canonical_hash, sha256_file
from carnot.reporting import experiment_7303_validation_scope as checks


ROOT = Path(__file__).resolve().parents[2]
RAW = Path("results/raw/experiment_7734_v673_independent_evidence_audit")
OUTPUT = Path("results/experiment_7734_v673_independent_evidence_audit.json")
PLAN = {
    7727: "results/experiment_7727_v673_development_corpus.json",
    7728: "results/experiment_7728_v673_set_energy_protocol.json",
    7729: "results/experiment_7729_v673_qwen_draft_pilot.json",
    7730: "results/experiment_7730_v673_set_energy_fit.json",
    7731: "results/experiment_7731_development_decisions.json",
    7732: "results/experiment_7732_v673_causal_admission.json",
    7733: "results/experiment_7733_continuous_set_learning.json",
}
REQUIRED = {7731, 7733}
SCOPE = {
    "test_paths": ["tests/python/test_experiment_7734_v673_independent_evidence_audit.py"],
    "changed_modules": ["python/carnot/experiment_7734_v673_independent_evidence_audit.py"],
    "static_paths": ["scripts/experiments/experiment_7734_v673_independent_evidence_audit.py"],
    "requirements": ["REQ-REPORT-7734", "REQ-CL-7734-AUDIT"],
}
PRINCIPLE = "Measured evidence bounds the claim and downstream use."


def progress(start: float, phase: str, event: str, units: int = 0) -> None:
    """Print real monotonic progress at each phase boundary."""
    print(
        f"[exp7734] {phase} {event} elapsed_s={time.monotonic() - start:.3f} completed_units={units}",
        flush=True,
    )


def failure(
    check: str, upstream_id: str, path: Path, field: str, expected: Any, observed: Any
) -> dict[str, Any]:
    """Retain enough operands to reproduce one failed gate."""
    return {
        "check": check,
        "upstream_id": upstream_id,
        "artifact_path": str(path.resolve()),
        "field": field,
        "operator": "==",
        "expected": expected,
        "observed": observed,
    }


def inspect_sources(
    root: Path, plan: dict[int, str], required: set[int]
) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    """Hash planned inputs and distinguish complete null from pre-gate receipts."""
    custody: list[dict[str, Any]] = []
    hashes: dict[str, Any] = {
        key: {}
        for key in (
            "eligible_producers",
            "flagged_historical_inputs",
            "pre_gate_receipts",
            "absent_sources",
        )
    }
    failures: list[dict[str, Any]] = []
    for number, relative in plan.items():
        path = root / relative
        source = f"Exp{number}"
        if not path.is_file():
            hashes["absent_sources"][relative] = None
            custody.append({"upstream_id": source, "artifact_path": relative, "state": "absent"})
            if number in required:
                failures.append(
                    failure("required_source_exists", source, path, "exists", True, False)
                )
            continue
        digest = sha256_file(path)
        try:
            value = json.loads(path.read_bytes())
        except (ValueError, UnicodeError):
            value = None
        if not isinstance(value, dict):
            state, field, expected, observed = "pre_gate", "json_object", True, False
        else:
            verdict = value.get("honest_verdict")
            flagged = value.get("flagged_adversarial")
            state = (
                "eligible"
                if isinstance(verdict, str)
                and verdict.startswith("complete_")
                and value.get("verdict_class") in {"positive", "null", "circular_positive"}
                and flagged is False
                else "pre_gate"
            )
            field, expected, observed = (
                "verdict_class",
                "positive|null|circular_positive",
                value.get("verdict_class"),
            )
            if not isinstance(verdict, str) or not verdict.startswith("complete_"):
                field, expected, observed = "honest_verdict", "complete_*", verdict
            if flagged is True:
                state, field, expected, observed = "flagged", "flagged_adversarial", False, True
        bucket = (
            "eligible_producers"
            if state == "eligible"
            else "flagged_historical_inputs"
            if state == "flagged"
            else "pre_gate_receipts"
        )
        hashes[bucket][relative] = digest
        custody.append(
            {
                "upstream_id": source,
                "artifact_path": relative,
                "state": state,
                "sha256": digest,
                "verdict_class": value.get("verdict_class") if isinstance(value, dict) else None,
            }
        )
        if number in required and state != "eligible":
            failures.append(
                failure("required_source_eligible", source, path, field, expected, observed)
            )
    return custody, hashes, failures


def reduce_raw(
    rows: list[dict[str, Any]],
    events: list[dict[str, Any]],
    summary: dict[str, Any],
    bank: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Recompute family metrics, pairing, chronology, admission and static closure."""
    failed: list[dict[str, Any]] = []

    def fail(check: str, observed: Any) -> None:
        failed.append({"check": check, "observed": observed})

    by_unit: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_arm: dict[str, list[float]] = defaultdict(list)
    reduced_rows: list[dict[str, Any]] = []
    for row in rows:
        unit, arm = row["unit_id"], row["arm"]
        by_unit[unit].append(row)
        if row["prediction_tick"] >= row["feedback_tick"]:
            fail("label_leakage", [unit, arm])
        label, probability = row["label"], row["probability"]
        if (
            type(label) is not int
            or label not in (0, 1)
            or not isinstance(probability, (int, float))
            or not math.isfinite(probability)
            or not 0 <= probability <= 1
        ):
            fail("corrected_confidence", [unit, arm])
            brier = None
        else:
            brier = (probability - label) ** 2
            by_arm[arm].append(brier)
            if not math.isclose(brier, row["brier"], abs_tol=1e-9):
                fail("aggregate_contradiction", [unit, arm, row["brier"], brier])
        reduced_rows.append(
            {
                "family_id": unit,
                "arm": arm,
                "role": row["role"],
                "raw_metrics": {"label": label, "probability": probability, "brier": brier},
                "denominators": row.get("denominators", {"independent_family": 1}),
                "censored": row["censored"],
                "exclusions": row["exclusions"],
                "input_hash": row["input_hash"],
                "source_sha256": row["source_sha256"],
            }
        )
    for unit, group in by_unit.items():
        if len({r["source_sha256"] for r in group}) != 1:
            fail("source_identity", unit)
        if (
            len({r["input_hash"] for r in group}) != 1
            or len({r["arm"] for r in group}) != len(group)
            or len(group) != 3
        ):
            fail("paired_input_support", unit)
        if len({r["role"] for r in group}) != 1:
            fail("retained_disjoint", unit)
    roles: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        roles[row["role"]].add(row["unit_id"])
    if any(roles[a] & roles[b] for a in roles for b in roles if a != b):
        fail("retained_disjoint", sorted(roles))
    admissions = [(e["unit_id"], e["arm"]) for e in events if e["kind"] == "one_use_admission"]
    if len(admissions) != len(set(admissions)):
        fail("admission_one_use", admissions)
    event_keys = Counter((e["unit_id"], e["arm"], e["kind"]) for e in events)
    for row in rows:
        for kind in ("prediction", "feedback_arrival"):
            if event_keys[row["unit_id"], row["arm"], kind] != 1:
                fail("event_chronology", [row["unit_id"], row["arm"], kind])
    for event in events:
        if event["kind"] == "feedback_arrival":
            matching = [r for r in by_unit.get(event["unit_id"], []) if r["arm"] == event["arm"]]
            if not matching or event["tick"] <= matching[0]["prediction_tick"]:
                fail("event_chronology", event)
    if summary.get("restart_exact_parity") is not True:
        fail("restart_replay", summary.get("restart_exact_parity"))
    closure = summary.get("static_closure_complete", {})
    dictionary = closure.get("dictionary", [])
    primitives = [name for name in dictionary if "&" not in name]
    pairs = [name for name in dictionary if "&" in name]
    if (
        not closure.get("equality_check")
        or len(primitives) != 8
        or len(pairs) != 28
        or set(dictionary) != set(closure.get("weights", {}))
    ):
        fail("static_closure", closure)
    if bank is not None:
        grammar = bank["config"]["grammar"]
        expected = list(grammar["primitives"]) + ["&".join(pair) for pair in grammar["pairs"]]
        if (
            dictionary != expected
            or bank["templates"]
            or bank["used_admissions"]
            or bank["proposal"] is not None
        ):
            fail(
                "static_closure",
                {
                    "expected_dictionary": expected,
                    "observed_dictionary": dictionary,
                    "templates": bank["templates"],
                    "used_admissions": bank["used_admissions"],
                },
            )
    if summary.get("exactly_once") is not True:
        fail("admission_one_use", summary.get("exactly_once"))
    decision_costs = []
    for decision in summary.get("decisions", []):
        frozen = decision["frozen"]
        labels = decision["labels"]
        if len(frozen) != len(labels) or len({row["unit_id"] for row in frozen}) != len(frozen):
            fail("admission_family_support", len(frozen))
            continue
        gains = [
            (row["base"] - label) ** 2 - (row["candidate"] - label) ** 2
            for row, label in zip(frozen, labels, strict=True)
        ]
        gain = sum(gains) / len(gains)
        if not math.isclose(gain, decision["mean_brier_reduction"], abs_tol=1e-9):
            fail("decision_aggregate_contradiction", gain)
        base_false = decision["base_false_accepts"]
        candidate_false = decision["candidate_false_accepts"]
        if not 0 <= base_false <= len(frozen) or not 0 <= candidate_false <= len(frozen):
            fail("false_accept_cost", [base_false, candidate_false])
        if decision["accepted"] and (gain < 0.01 or candidate_false > base_false):
            fail("false_accept_cost", [gain, base_false, candidate_false])
        decision_costs.append(
            {
                "gain": gain,
                "base_false_accepts": base_false,
                "candidate_false_accepts": candidate_false,
                "denominator": len(frozen),
            }
        )
    means = {
        arm: {"brier": sum(losses) / len(losses), "denominator": len(losses)}
        for arm, losses in by_arm.items()
        if losses
    }
    return {
        "failed_checks": failed,
        "rows": reduced_rows,
        "by_arm": means,
        "sample_size": {
            "intended": 96,
            "observed": len(by_unit),
            "eligible": len(by_unit),
            "excluded": 0,
            "censored": len({r["unit_id"] for r in rows if r["censored"]}),
            "effective_independent_families": len(by_unit),
            "roles": {key: len(value) for key, value in roles.items()},
        },
        "event_count": len(events),
        "admission_count": len(admissions),
        "decision_costs": decision_costs,
    }


def make_artifact(
    root: Path,
    date: str,
    custody: list[dict[str, Any]],
    hashes: dict[str, Any],
    failures: list[dict[str, Any]],
    reduction: dict[str, Any] | None,
) -> dict[str, Any]:
    """Keep terminal work completion distinct from upstream science readiness."""
    states = {row["upstream_id"]: row["state"] for row in custody}
    static = states.get("Exp7731") == "eligible"
    online = states.get("Exp7733") == "eligible"
    audit_valid = reduction is None or not reduction["failed_checks"]
    eligible = static and online and audit_valid and not failures
    gate_names = (
        "validity",
        "readiness",
        "brier_score",
        "decision_cost",
        "coverage",
        "retention",
        "efficiency",
    )
    gates: dict[str, Any] = {name: None for name in gate_names}
    gates["validity"] = audit_valid
    if eligible:
        gates["readiness"] = True
        gates["brier_score"] = reduction["by_arm"] if reduction else None
        gates["coverage"] = reduction["sample_size"] if reduction else None
    rows = (
        reduction["rows"]
        if reduction
        else [
            {
                "family_id": row["upstream_id"],
                "arm": "custody",
                "role": "source",
                "raw_metrics": None,
                "denominators": None,
                "censored": False,
                "exclusions": [row["state"]],
                "input_hash": row.get("sha256"),
            }
            for row in custody
        ]
    )
    counts = {
        key: 0
        for key in (
            "model_loads",
            "forward_calls",
            "generation_calls",
            "input_tokens",
            "output_tokens",
            "failures",
            "cancellations",
        )
    }
    artifact: dict[str, Any] = {
        "schema": "carnot.exp7734.v673.independent_evidence_audit.v1",
        "experiment_id": 7734,
        "milestone": "2026.09.673",
        "run_date": date,
        "honest_verdict": "complete_null_no_new_benefit"
        if eligible
        else "complete_blocked_required_v673_evidence",
        "verdict_class": "null" if eligible else "blocked",
        "flagged_adversarial": False,
        "gate_check_summary": failures,
        "acceptance_gate_results": gates,
        "rows": rows,
        "sample_size_budget": reduction["sample_size"]
        if reduction
        else {
            "intended": None,
            "observed": 0,
            "eligible": 0,
            "excluded": 0,
            "censored": 0,
            "effective_independent_families": 0,
        },
        "claim_scope": {
            "reused_RAGTruth": "development_only",
            "exact_fixtures": "fixture_only",
            "ARC_public_games": "adapter_withheld_public",
            "fresh_generalization_eligible": False,
        },
        "fresh_generalization_eligible": False,
        "inference_substrate": "aggregation_from_upstream_artifacts",
        "inference_substrate_class": "aggregation",
        "planned_inference_substrate_class": "aggregation",
        "MODEL_SPECS": [],
        "planned_MODEL_SPECS": [],
        "model_specs": [],
        "model_invoked": False,
        "invocation_counts": counts,
        "execution_venue": "host",
        "execution_venue_details": {
            "host": socket.gethostname(),
            "owned_pid": os.getpid(),
        },
        "phase_spans": [],
        "duration_s": 0.0,
        "random_seed": {
            "seed": 7734,
            "purpose": "fixed audit ordering; no experimental randomization",
        },
        "source_artifact_hashes": hashes,
        "preconditions_checked": {
            "root": str(root.resolve()),
            "cpu_count": os.cpu_count(),
            "effective_coding_backend": "Codex GPT-6 API session",
            "required_static_exists": states.get("Exp7731") != "absent",
            "required_online_exists": states.get("Exp7733") != "absent",
            "schemas_checked": {row["upstream_id"]: row["state"] for row in custody},
        },
        "validation_receipts": {
            "frozen_affected_scope": SCOPE,
            "required_commands": [],
            "terminal_readers": [],
            "e2e_checks": [],
            "global_suite_debt": [],
        },
        "verifier_is_oracle": True,
        "independent_audit_complete_score": int(audit_valid),
        "independent_static_eligible": static and audit_valid,
        "independent_online_eligible": online and audit_valid,
        "claim_matrix": {
            row["upstream_id"]: {
                "eligible": row["state"] == "eligible",
                "state": row["state"],
                "artifact_path": row["artifact_path"],
                "exposure_limit": "development_only"
                if row["upstream_id"] != "Exp7728"
                else "fixture_only",
                "observed_metrics": reduction["by_arm"]
                if row["upstream_id"] == "Exp7732" and reduction
                else None,
            }
            for row in custody
        },
        "upstream_dispositions": custody,
        "recomputed_metrics": reduction,
        "source_plan": {str(number): path for number, path in PLAN.items()},
        "activation": False,
        "pilot_qwen_receipts": {
            "upstream_id": "Exp7729",
            "state": states.get("Exp7729"),
            "current_model_invocations": 0,
        },
    }
    artifact["field_principles"] = {key: PRINCIPLE for key in artifact}
    artifact["field_principles"].update({key: PRINCIPLE for key in gate_names})
    artifact["reproducibility_checksum"] = canonical_hash(
        {
            "sources": hashes,
            "scope": SCOPE,
            "seed": 7734,
            "reducer_code": sha256_file(
                root / "python/carnot/experiment_7734_v673_independent_evidence_audit.py"
            )
            if (root / "python/carnot/experiment_7734_v673_independent_evidence_audit.py").is_file()
            else None,
        }
    )
    artifact["field_principles"]["reproducibility_checksum"] = PRINCIPLE
    return artifact


def cold_replay(candidate: Path) -> list[str]:
    """In a fresh process, rehash source bytes and recompute raw reductions."""
    value = json.loads(candidate.read_bytes())
    root = Path(value["preconditions_checked"]["root"])
    plan = {int(number): path for number, path in value["source_plan"].items()}
    custody, hashes, failures = inspect_sources(root, plan, REQUIRED & set(plan))
    errors = []
    for key, current in (
        ("upstream_dispositions", custody),
        ("source_artifact_hashes", hashes),
        ("gate_check_summary", failures),
    ):
        if current != value[key]:
            errors.append(f"{key}_changed")
    raw_files = value.get("raw_input_hashes", {})
    for relative, digest in raw_files.items():
        path = root / relative
        if not path.is_file() or sha256_file(path) != digest:
            errors.append(f"raw_input_changed:{relative}")
    if raw_files:
        folder = root / "results/raw/experiment_7732_v673_causal_admission"
        producer = json.loads((root / PLAN[7732]).read_bytes())
        reduced = reduce_raw(
            json.loads((folder / "rows.json").read_bytes()),
            json.loads((folder / "event_rows.json").read_bytes()),
            producer,
            json.loads((folder / "bank_complete_static.json").read_bytes())
            if (folder / "bank_complete_static.json").is_file()
            else None,
        )
        if reduced != value["recomputed_metrics"]:
            errors.append("raw_reduction_changed")
    checksum = canonical_hash(
        {
            "sources": hashes,
            "scope": SCOPE,
            "seed": 7734,
            "reducer_code": sha256_file(
                root / "python/carnot/experiment_7734_v673_independent_evidence_audit.py"
            )
            if (root / "python/carnot/experiment_7734_v673_independent_evidence_audit.py").is_file()
            else None,
        }
    )
    if checksum != value["reproducibility_checksum"]:
        errors.append("reproducibility_checksum_changed")
    return errors


def run_experiment(root: Path, date: str, output: Path) -> dict[str, Any]:
    """Run bounded readers, preserve exact logs, then publish one terminal JSON."""
    root = root.resolve()
    started = time.monotonic()
    spans: list[dict[str, Any]] = []
    last = 0.0

    def span(name: str, units: int) -> None:
        nonlocal last
        now = time.monotonic() - started
        spans.append(
            {
                "phase": name,
                "run_date": date,
                "start_s": last,
                "end_s": now,
                "duration_s": now - last,
                "heartbeat_timestamps_s": [now],
                "completed_units": units,
                "checkpoint_hash": canonical_hash({"phase": name, "units": units}),
            }
        )
        last = now

    progress(started, "preconditions", "start")
    raw = root / RAW
    raw.mkdir(parents=True, exist_ok=True)
    atomic_json(raw / "frozen_affected_scope.json", SCOPE)
    custody, hashes, failures = inspect_sources(root, PLAN, REQUIRED)
    raw_inputs: dict[str, str] = {}
    reduction = None
    folder = root / "results/raw/experiment_7732_v673_causal_admission"
    files = [folder / "rows.json", folder / "event_rows.json", folder / "bank_complete_static.json"]
    if all(path.is_file() for path in files) and (root / PLAN[7732]).is_file():
        for path in files:
            raw_inputs[path.relative_to(root).as_posix()] = sha256_file(path)
        producer = json.loads((root / PLAN[7732]).read_bytes())
        reduction = reduce_raw(
            json.loads(files[0].read_bytes()),
            json.loads(files[1].read_bytes()),
            producer,
            json.loads(files[2].read_bytes()),
        )
        for item in reduction["failed_checks"]:
            failures.append(
                failure(
                    item["check"], "Exp7732", files[0], "raw_reduction", "valid", item["observed"]
                )
            )
    atomic_json(
        raw / "checkpoint.json",
        {"completed_units": len(custody), "source_hashes": hashes, "raw_input_hashes": raw_inputs},
    )
    progress(started, "preconditions", "complete", len(custody))
    span("preconditions", len(custody))
    progress(started, "reduction", "start")
    artifact = make_artifact(root, date, custody, hashes, failures, reduction)
    artifact["raw_input_hashes"] = raw_inputs
    artifact["field_principles"]["raw_input_hashes"] = PRINCIPLE
    atomic_json(raw / "independent_reduction.json", reduction)
    progress(started, "reduction", "complete", len(artifact["rows"]))
    span("reduction", len(artifact["rows"]))
    private = Path(tempfile.mkdtemp(prefix="exp7734-", dir="/tmp"))
    (private / "basetemp").mkdir()
    progress(started, "affected_validation", "start")
    commands = checks.build_scoped_commands(
        root,
        SCOPE["test_paths"],
        SCOPE["changed_modules"],
        static_paths=SCOPE["static_paths"],
        basetemp=private / "basetemp",
        coverage_file=private / "coverage.data",
    )
    receipts = checks.run_commands(
        root,
        commands,
        log_dir=raw / "validation/affected",
        extra_env={"COVERAGE_FILE": str(private / "coverage.data"), "JAX_PLATFORMS": "cpu"},
        heartbeat_s=30.0,
    )
    artifact["validation_receipts"]["required_commands"] = receipts
    artifact["validation_receipts"].update(checks.reduce_required_checks(receipts))
    if not artifact["validation_receipts"]["required_checks_passed"]:
        artifact["honest_verdict"] = "complete_disqualified_required_validation"
        artifact["verdict_class"] = "disqualified"
        artifact["independent_audit_complete_score"] = 0
        artifact["acceptance_gate_results"]["readiness"] = False
    progress(started, "affected_validation", "complete", len(receipts))
    span("affected_validation", len(receipts))
    progress(started, "repository_health", "start")
    full = [
        checks.CommandSpec(
            "full_python_suite",
            (
                str(root / ".venv/bin/pytest"),
                "-n",
                "0",
                "-o",
                "addopts=",
                "--no-cov",
                f"--basetemp={private / 'full'}",
                "tests/python",
                "-q",
            ),
            "repository_health",
            900,
        )
    ]
    artifact["validation_receipts"]["global_suite_debt"] = checks.run_commands(
        root,
        full,
        log_dir=raw / "validation/full",
        extra_env={"JAX_PLATFORMS": "cpu"},
        heartbeat_s=30.0,
    )
    progress(started, "repository_health", "complete", 1)
    span("repository_health", 1)
    artifact["phase_spans"] = spans.copy()
    artifact["duration_s"] = last
    candidate = raw / "terminal_candidate.json"
    atomic_json(candidate, artifact)
    python = str(root / ".venv/bin/python")
    terminal = [
        checks.CommandSpec(
            "fresh_process_cold_replay",
            (
                python,
                "-u",
                "-m",
                "carnot.experiment_7734_v673_independent_evidence_audit",
                "--cold",
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
    progress(started, "terminal_readers", "start")
    readers = checks.run_commands(
        root,
        terminal,
        log_dir=raw / "validation/terminal",
        extra_env={"JAX_PLATFORMS": "cpu"},
        heartbeat_s=30.0,
    )
    artifact["validation_receipts"]["terminal_readers"] = readers
    artifact["validation_receipts"]["exact_candidate_sha256"] = sha256_file(candidate)
    artifact["validation_receipts"]["e2e_checks"] = [
        {
            "name": "cold_cli_replay_from_raw",
            "passed": readers[0]["passed"],
            "receipt": readers[0]["log_sha256"],
        }
    ]
    artifact["flagged_adversarial"] = not readers[1]["passed"]
    if not all(row["passed"] for row in readers):
        artifact["honest_verdict"] = "complete_disqualified_terminal_reader"
        artifact["verdict_class"] = "disqualified"
        artifact["independent_audit_complete_score"] = 0
        artifact["acceptance_gate_results"]["readiness"] = False
    progress(started, "terminal_readers", "complete", len(readers))
    span("terminal_readers", len(readers))
    progress(started, "publication", "start")
    span("publication", 1)
    artifact["phase_spans"] = spans
    artifact["duration_s"] = last
    atomic_json(output, artifact)
    progress(started, "publication", "complete", 1)
    return artifact


def main(argv: list[str] | None = None) -> int:
    """Expose the bounded task and fresh-process cold reader through one CLI."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", default="20260927")
    parser.add_argument("--output", type=Path, default=ROOT / OUTPUT)
    parser.add_argument("--cold", type=Path)
    args = parser.parse_args(argv)
    if args.cold:
        errors = cold_replay(args.cold)
        print(json.dumps({"cold_replay_errors": errors}), flush=True)
        return int(bool(errors))
    run_experiment(ROOT, args.date, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
