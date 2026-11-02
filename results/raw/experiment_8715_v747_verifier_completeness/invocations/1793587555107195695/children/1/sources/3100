"""Audit V667 source witnesses and keep blocked science explicit.

The source rows are read again here because producer headlines can be wrong.
Spec: REQ-REPORT-7650 and SCENARIO-REPORT-7650-*.
"""

from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import json
import os
from pathlib import Path
import socket
import sys
import tempfile
import time
from typing import Any

from carnot.experiment_7644_v667_source_witness_prototype import (
    build_fixture_rows,
    validate_predictor_input,
)
from carnot.reporting import experiment_7303_validation_scope as checks
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash, sha256_file
from carnot.verify.source_claim_witness import verify_claim


ROOT = Path(__file__).resolve().parents[2]
RESULT = Path("results/experiment_7650_v667_independent_source_audit.json")
RAW = Path("results/raw/experiment_7650_v667_independent_source_audit")
NOTE = Path("docs/research-notes/v667-source-learning-audit.md")
SOURCES = (
    ("exp7644", "results/experiment_7644_v667_source_witness_prototype.json"),
    ("exp7646", "results/experiment_7646_v667_source_feature_corpus.json"),
    ("exp7647", "results/experiment_7647_v667_witness_energy.json"),
    ("exp7648", "results/experiment_7648_v667_decision_evaluation.json"),
    ("exp7649", "results/experiment_7649_v667_continuous_witness_learning.json"),
)
NAMED_INPUTS = (
    "AGENTS.md",
    "CODEX.md",
    "CLAUDE.md",
    "research-program.md",
    "ops/exclusion_manifest.yaml",
    "ops/e2e-test-plan.md",
    "scripts/experiment_template.py",
    "python/carnot/reporting/current_work_receipt.py",
    "python/carnot/reporting/experiment_7303_validation_scope.py",
    "python/carnot/experiment_7638_v666_evidence_audit.py",
    "python/carnot/experiment_7579_v662_decision_learning_audit.py",
    "openspec/capabilities/research-reporting/spec.md",
)
MANIFEST = {
    "test_paths": ["tests/python/test_experiment_7650_v667_independent_source_audit.py"],
    "changed_modules": ["python/carnot/experiment_7650_v667_independent_source_audit.py"],
    "static_paths": ["scripts/experiments/experiment_7650_v667_independent_source_audit.py"],
}


def progress(started: float, phase: str, event: str, **details: Any) -> None:
    """Show owned work as it proceeds so a long check remains observable."""

    print(
        f"[exp7650] {phase} {event} elapsed_s={time.monotonic() - started:.3f} {details}",
        flush=True,
    )


def gate_check(
    check: str, upstream: str, path: Path, field: str, operator: str, expected: Any, observed: Any
) -> dict[str, Any]:
    """Keep both sides of every failed external condition."""

    return {
        "check": check,
        "upstream": upstream,
        "path": str(path.resolve()),
        "field": field,
        "operator": operator,
        "expected": expected,
        "observed": observed,
    }


def collect_sources(
    root: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Read actual producer bytes and retain a conductor receipt as a receipt."""

    sources: list[dict[str, Any]] = []
    failed: list[dict[str, Any]] = []
    hashes: dict[str, Any] = {
        "producer_files": [],
        "pre_gate_receipts": [],
        "missing_inputs": [],
        "planned_outputs": [],
        "raw_files": [],
    }
    for upstream, relative in SOURCES:
        path = (root / relative).resolve()
        if not path.is_file():
            if upstream == "exp7647":
                receipt_path = (root / "results/experiment_7647_witness_energy.json").resolve()
                if receipt_path.is_file():
                    receipt_value = json.loads(receipt_path.read_text(encoding="utf-8"))
                    if receipt_value.get("blocked_at_layer") == "conductor_pre_gate":
                        receipt = {
                            "upstream": upstream,
                            "path": str(receipt_path),
                            "sha256": sha256_file(receipt_path),
                            "bytes": receipt_path.stat().st_size,
                        }
                        hashes["pre_gate_receipts"].append(receipt)
                        sources.append(
                            {
                                **receipt,
                                "disposition": "pre_gate_blocked",
                                "honest_verdict": receipt_value.get("honest_verdict"),
                            }
                        )
                        failed.append(
                            gate_check(
                                "producer_eligible",
                                upstream,
                                receipt_path,
                                "blocked_at_layer",
                                "==",
                                None,
                                "conductor_pre_gate",
                            )
                        )
                        continue
            sources.append({"upstream": upstream, "path": str(path), "disposition": "missing"})
            hashes["missing_inputs"].append({"upstream": upstream, "path": str(path)})
            failed.append(
                gate_check("producer_exists", upstream, path, "path.is_file", "==", True, False)
            )
            continue
        value = json.loads(path.read_text(encoding="utf-8"))
        is_receipt = value.get("blocked_at_layer") == "conductor_pre_gate"
        disposition = "pre_gate_blocked" if is_receipt else value.get("verdict_class", "unknown")
        receipt = {
            "upstream": upstream,
            "path": str(path),
            "sha256": sha256_file(path),
            "bytes": path.stat().st_size,
        }
        hashes["pre_gate_receipts" if is_receipt else "producer_files"].append(receipt)
        sources.append(
            {
                **receipt,
                "disposition": disposition,
                "honest_verdict": value.get("honest_verdict"),
                "flagged_adversarial": value.get("flagged_adversarial"),
            }
        )
        if disposition not in {"positive", "circular_positive", "null"}:
            failed.append(
                gate_check(
                    "producer_eligible",
                    upstream,
                    path,
                    "verdict_class",
                    "in",
                    ["positive", "circular_positive", "null"],
                    disposition,
                )
            )
        if value.get("flagged_adversarial") is True:
            failed.append(
                gate_check(
                    "producer_unflagged", upstream, path, "flagged_adversarial", "==", False, True
                )
            )
    raw_paths = ["tests/python/fixtures/experiment_7644_source_witness.jsonl"] + [
        f"results/raw/experiment_7646_v667_source_feature_corpus/{role}_features.jsonl"
        for role in ("fit", "tune", "policy", "online", "evaluation", "pilot")
    ]
    for relative in raw_paths:
        path = (root / relative).resolve()
        hashes["raw_files"].append({"path": str(path), "sha256": sha256_file(path)})
    return sources, failed, hashes


def reduce_available(
    root: Path, sources: list[dict[str, Any]]
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    """Recompute only metrics supported by raw source and fixture rows."""

    findings: dict[str, Any] = {
        item["upstream"]: {"disposition": item["disposition"], "reader_result": "unavailable"}
        for item in sources
    }
    fixture_path = root / "tests/python/fixtures/experiment_7644_source_witness.jsonl"
    cases = [json.loads(line) for line in fixture_path.read_text().splitlines()]
    fixture_rows = build_fixture_rows(cases)
    matches = sum(row["numerator"] for row in fixture_rows)
    findings["exp7644"].update(
        reader_result="raw_fixture_recomputed",
        fixture_matches=matches,
        fixture_denominator=len(cases),
        oracle_distinct_benefit=False,
    )
    feature_base = root / "results/raw/experiment_7646_v667_source_feature_corpus"
    feature_rows = [
        {**json.loads(line), "raw_feature_path": str(feature_base / f"{role}_features.jsonl")}
        for role in ("fit", "tune", "policy", "online", "evaluation", "pilot")
        for line in (feature_base / f"{role}_features.jsonl").read_text().splitlines()
    ]
    scored = [row for row in feature_rows if row["role"] != "pilot"]
    groups = {row["unit_id"] for row in scored}
    arms = Counter(row["arm"] for row in scored)
    unknown = sum(int(row["unchecked_prose"]) for row in scored)
    checked = sum(int(row["checked_predicates"]) for row in scored)
    denominator = sum(int(row["denominator"]) for row in scored)
    findings["exp7646"].update(
        reader_result="raw_features_recomputed_disqualified",
        independent_groups=len(groups),
        arm_rows=len(scored),
        arms=dict(arms),
        checked_predicates=checked,
        unknown_prose=unknown,
        coverage_denominator=denominator,
        checked_predicate_coverage=checked / denominator if denominator else None,
        benefit_eligible=False,
    )
    rows = fixture_rows + feature_rows
    budget = {
        "intended_independent_groups": 240,
        "observed_independent_groups": len(groups),
        "fixture_groups": len(cases),
        "eligible": 0,
        "excluded": len(groups),
        "censored": len({row["unit_id"] for row in scored if row["censored"]}),
        "exposure_limits": "Three source arms share each group; all inherited groups were exposed.",
    }
    return findings, rows, budget


def causal_errors(events: list[dict[str, Any]]) -> list[str]:
    """A released label may update once, after its prediction and release."""

    errors: list[str] = []
    admissions: set[str] = set()
    for row in events:
        if (
            row["release_step"] >= row["update_step"]
            or row["prediction_step"] >= row["release_step"]
        ):
            errors.append("future_label")
        if row["id"] in admissions:
            errors.append("duplicate_admission")
        admissions.add(row["id"])
    return errors


def run_private_mutations() -> dict[str, dict[str, Any]]:
    """Change one source or lifecycle operand, then require its reader to reject."""

    source = "```python file=a.py\n1 | class A:\n2 |     def f(self): pass\n3 | class B:\n4 |     def f(self): pass\n```"
    baseline = {
        "source": source,
        "claim": "In `a.py`, `A.f` is defined at line 2.",
        "closed_files": ["a.py"],
    }
    bad: dict[str, Any] = {
        "label_in_predictor_input": {**baseline, "dataset_label": 1},
        "evidence_wrong_file": {**baseline, "claim": "In `b.py`, `A.f` is defined at line 2."},
        "same_name_wrong_scope": {**baseline, "claim": "In `a.py`, `A.f` is defined at line 4."},
        "semantic_overclaim": {
            **baseline,
            "claim": "In `a.py`, `A.f` is defined at line 2. Therefore it is secure.",
        },
        "future_label_shuffle": [
            {"id": "x", "prediction_step": 1, "release_step": 4, "update_step": 3}
        ],
        "duplicate_admission": [
            {"id": "x", "prediction_step": 1, "release_step": 2, "update_step": 3},
            {"id": "x", "prediction_step": 4, "release_step": 5, "update_step": 6},
        ],
        "permuted_role": {"expected": "evaluation", "observed": "fit"},
        "aggregate_disagreement": {"rows": [1, 0], "claimed_total": 2},
    }
    clean_events = [{"id": "x", "prediction_step": 1, "release_step": 2, "update_step": 3}]
    clean_values: dict[str, Any] = {name: baseline for name in bad}
    clean_values["future_label_shuffle"] = clean_events
    clean_values["duplicate_admission"] = clean_events
    clean_values["permuted_role"] = {"expected": "evaluation", "observed": "evaluation"}
    clean_values["aggregate_disagreement"] = {"rows": [1, 0], "claimed_total": 1}
    results: dict[str, dict[str, Any]] = {}
    for name, changed in bad.items():
        if name == "label_in_predictor_input":
            try:
                validate_predictor_input(changed)
            except ValueError:
                rejected = True
            else:
                raise AssertionError("label_mutation_not_rejected")
        elif name in {"evidence_wrong_file", "same_name_wrong_scope", "semantic_overclaim"}:
            witness = verify_claim(**changed)
            rejected = witness["status"] != "supported" or witness["residual_unverified_span"]
        elif name in {"future_label_shuffle", "duplicate_admission"}:
            rejected = bool(causal_errors(changed))
        elif name == "permuted_role":
            rejected = changed["observed"] != changed["expected"]
        else:
            rejected = sum(changed["rows"]) != changed["claimed_total"]
        results[name] = {
            "rejected": rejected,
            "baseline_sha256": canonical_hash(clean_values[name]),
            "changed_sha256": canonical_hash(changed),
        }
    return results


def validate_terminal(value: dict[str, Any], root: Path) -> list[str]:
    """Rebuild custody and rows before accepting one terminal candidate."""

    errors: list[str] = []
    sources, failed, hashes = collect_sources(root)
    findings, rows, budget = reduce_available(root, sources)
    validity = next(
        (
            gate.get("passed")
            for gate in value.get("acceptance_gate_results", [])
            if gate.get("gate") == "validity"
        ),
        True,
    )
    expected_verdict = (
        "complete_blocked_source_producers_unavailable"
        if validity
        else "complete_disqualified_required_validation"
    )
    expected_class = "blocked" if validity else "disqualified"
    if (
        value.get("honest_verdict") != expected_verdict
        or value.get("verdict_class") != expected_class
    ):
        errors.append("blocked_verdict_mismatch")
    if value.get("gate_check_summary") != failed or value.get("source_artifact_hashes") != hashes:
        errors.append("source_custody_mismatch")
    if value.get("rows") != rows or value.get("sample_size_budget") != budget:
        errors.append("row_reduction_mismatch")
    if value.get("independent_findings") != findings:
        errors.append("finding_mismatch")
    if value.get(
        "acceptance_gate_results", acceptance_gates(findings, validity)
    ) != acceptance_gates(findings, validity):
        errors.append("acceptance_gate_mismatch")
    if value.get("private_mutations", run_private_mutations()) != run_private_mutations():
        errors.append("mutation_mismatch")
    if value.get("flagged_adversarial") is not False or value.get(
        "independent_audit_complete_score", int(validity)
    ) != int(validity):
        errors.append("terminal_claim_mismatch")
    if value.get("MODEL_SPECS") != [] or value.get("model_invoked") is not False:
        errors.append("current_inference_mismatch")
    return errors


def acceptance_gates(findings: dict[str, Any], valid: bool) -> list[dict[str, Any]]:
    """A coverage count cannot answer a probability or decision question."""

    operands = {
        "validity": {"required_checks_passed": valid},
        "readiness": {
            "fixture_matches": findings["exp7644"]["fixture_matches"],
            "corpus_disposition": findings["exp7646"]["disposition"],
        },
        "probability_benefit": {"eligible_paired_probabilities": 0, "eligible_labels": 0},
        "utility": {"eligible_typed_decisions": 0, "eligible_costs": 0},
        "retention": {"eligible_delayed_updates": 0, "eligible_holdout_rows": 0},
        "freshness": {"fresh_unexposed_groups": 0},
    }
    principles = {
        "validity": "Owned checks and terminal readers govern audit validity.",
        "readiness": "Fixture agreement and valid source rows govern readiness.",
        "probability_benefit": "Proper loss needs eligible paired probabilities and independent labels.",
        "utility": "Decision value needs typed actions, costs, and independent outcomes.",
        "retention": "Retained learning needs ordered feedback and held-out replay.",
        "freshness": "Fresh benefit needs a previously unexposed independent roster.",
    }
    return [
        {
            "gate": name,
            "principle": principles[name],
            "measured_operands": operands[name],
            "passed": valid if name == "validity" else (False if name == "readiness" else None),
        }
        for name in principles
    ]


def build_artifact(root: Path, date: str, *, valid: bool) -> dict[str, Any]:
    """Assemble current CPU evidence with no inherited model activity."""

    sources, failed, hashes = collect_sources(root)
    findings, rows, budget = reduce_available(root, sources)
    mutations = run_private_mutations()
    inputs = [
        {
            "path": str((root / relative).resolve()),
            "exists": (root / relative).is_file(),
            "sha256": sha256_file(root / relative) if (root / relative).is_file() else None,
        }
        for relative in NAMED_INPUTS
    ]
    value: dict[str, Any] = {
        "experiment_id": "exp7650-independent-source-audit",
        "milestone": "2026.09.667",
        "run_date": date,
        "honest_verdict": (
            "complete_blocked_source_producers_unavailable"
            if valid
            else "complete_disqualified_required_validation"
        ),
        "verdict_class": "blocked" if valid else "disqualified",
        "flagged_adversarial": False,
        "gate_check_summary": failed,
        "acceptance_gate_results": acceptance_gates(findings, valid),
        "rows": rows,
        "sample_size_budget": budget,
        "preconditions_checked": inputs,
        "inference_substrate": "CPU aggregation of authenticated source rows; no current model inference",
        "inference_substrate_class": "aggregation",
        "planned_inference_substrate_class": "aggregation",
        "MODEL_SPECS": [],
        "planned_MODEL_SPECS": [],
        "model_invoked": False,
        "invocation_counts": {
            name: 0 for name in ("model_loads", "forward_calls", "generations", "tokens")
        },
        "historical_model_identity": "Inherited producer model records only; not current inference",
        "execution_venue": "host",
        "execution_venue_details": {"hostname": socket.gethostname(), "owned_pid": os.getpid()},
        "phase_spans": [],
        "duration_s": 0.0,
        "random_seed": {"fixture_order": "frozen source order", "bootstrap": None},
        "source_artifact_hashes": hashes,
        "validation_receipts": {"affected": [], "terminal": [], "terminal_outcomes_path": None},
        "verifier_is_oracle": True,
        "field_principles": {
            "honest_verdict": "Completion and scientific benefit are separate claims.",
            "verdict_class": "Missing or disqualified external evidence blocks benefit.",
            "flagged_adversarial": "A flagged candidate cannot open a downstream gate.",
            "gate_check_summary": "Failed conditions retain both literal operands.",
            "acceptance_gate_results": "Coverage, proper loss, decisions, and retention have different evidence.",
            "rows": "Independent source groups own denominators; views do not multiply them.",
            "sample_size_budget": "Exposure and censoring bound inferential sample size.",
            "source_artifact_hashes": "Exact bytes bind source custody.",
            "validation_receipts": "Actual exits and logs govern audit validity.",
            "independent_audit_complete_score": "An audit may complete while science remains blocked.",
        },
        "independent_audit_complete_score": int(valid),
        "independent_findings": findings,
        "claim_findings": {
            "coverage": findings["exp7646"],
            "probability": {
                "reader_result": "unavailable",
                "reason": "7647 pre-gate and 7648 absent",
            },
            "decision": {"reader_result": "unavailable", "reason": "7648 absent"},
            "source_sensitivity": {"reader_result": "unavailable", "reason": "7648 absent"},
            "update_ordering": {"reader_result": "unavailable", "reason": "7649 absent"},
            "retention": {"reader_result": "unavailable", "reason": "7649 absent"},
        },
        "private_mutations": mutations,
        "audit_note_path": NOTE.as_posix(),
    }
    value["reproducibility_checksum"] = canonical_hash(
        {
            "sources": hashes,
            "config": {"class": "aggregation", "model_specs": []},
            "reduction_code": sha256_file(Path(__file__)),
            "findings": findings,
        }
    )
    return value


def cold_replay(candidate: Path) -> list[str]:
    """Reload exact candidate bytes and recalculate source custody."""

    value = json.loads(candidate.read_text(encoding="utf-8"))
    return validate_terminal(value, ROOT)


def independent_replay(candidate: Path) -> list[str]:
    """Recompute the immutable source and reduction checksum in another process."""

    value = json.loads(candidate.read_text(encoding="utf-8"))
    errors = cold_replay(candidate)
    expected = canonical_hash(
        {
            "sources": value["source_artifact_hashes"],
            "config": {"class": "aggregation", "model_specs": []},
            "reduction_code": sha256_file(Path(__file__)),
            "findings": value["independent_findings"],
        }
    )
    if expected != value.get("reproducibility_checksum"):
        errors.append("reproducibility_checksum_mismatch")
    if not all(
        item["rejected"] and item["baseline_sha256"] != item["changed_sha256"]
        for item in run_private_mutations().values()
    ):
        errors.append("private_mutation_failure")
    return errors


def terminal_commands(root: Path, candidate: Path) -> list[checks.CommandSpec]:
    """Name four bounded readers for one immutable candidate path."""

    python = str(root / ".venv/bin/python")
    module = "carnot.experiment_7650_v667_independent_source_audit"
    return [
        checks.CommandSpec(
            "fresh_process_cold_replay",
            (python, "-m", module, "--cold", str(candidate)),
            "exact_candidate",
            180,
        ),
        checks.CommandSpec(
            "independent_raw_reduction",
            (python, "-m", module, "--independent", str(candidate)),
            "exact_candidate",
            180,
        ),
        checks.CommandSpec(
            "adversarial_verify",
            (python, "scripts/adversarial_verify.py", "--json", str(candidate)),
            "exact_candidate",
            180,
        ),
        checks.CommandSpec(
            "verdict_row_consistency_strict",
            (python, "scripts/verdict_row_consistency_lint.py", "--strict", str(candidate)),
            "exact_candidate",
            180,
        ),
    ]


def write_note(root: Path, findings: dict[str, Any]) -> None:
    """Record a mechanism decision without rewriting prior verdicts."""

    path = root / NOTE
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "# V667 source and learning audit\n\n"
        "Experiment 7644 matches 48 exact structural fixtures. This is oracle agreement. "
        "It gives no probability or decision benefit.\n\n"
        f"Experiment 7646 has {findings['exp7646']['independent_groups']} scored groups and "
        f"{findings['exp7646']['unknown_prose']} unchecked prose spans across source arms. "
        "Its required validation failed and its final artifact is flagged. "
        "The corpus cannot open a benefit gate.\n\n"
        "Experiment 7647 has a conductor pre-gate receipt, not a trained energy result. "
        "Experiments 7648 and 7649 have no producer artifact. Probability, typed utility, "
        "source sensitivity, ordered updates, and retention have no eligible raw rows.\n\n"
        "Keep the byte-bound structural witness and explicit unknown status. "
        "Change the 7646 validation defects before treating its rows as eligible. "
        "Retire only the unchanged historical-only audit mechanism. Preserve the V666 "
        "and V667 blocked records; resource absence does not falsify the scientific idea.\n",
        encoding="utf-8",
    )


def run_experiment(root: Path, date: str, output: Path) -> dict[str, Any]:
    """Validate owned work, run exact readers, then publish one blocked audit."""

    root = root.resolve()
    started = time.monotonic()
    progress(started, "preconditions", "start", root=str(root))
    sources, _, _ = collect_sources(root)
    progress(started, "preconditions", "complete", dispositions=[x["disposition"] for x in sources])
    preconditions_end = time.monotonic() - started
    raw_dir = root / RAW
    raw_dir.mkdir(parents=True, exist_ok=True)
    atomic_json(raw_dir / "affected_validation_manifest.json", MANIFEST)
    private = Path(tempfile.mkdtemp(prefix="exp7650-", dir="/tmp"))
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
    progress(
        started, "affected_validation", "complete", passed=validation["required_checks_passed"]
    )
    valid = bool(validation["required_checks_passed"])
    artifact = build_artifact(root, date, valid=valid)
    artifact["validation_receipts"]["affected"] = validation["validation_receipts"]
    write_note(root, artifact["independent_findings"])
    artifact["phase_spans"] = [
        {
            "phase": "preconditions_and_reduction",
            "start_s": 0.0,
            "end_s": preconditions_end,
            "completed_units": 5,
            "checkpoint_position": "producer_inventory",
        },
        {
            "phase": "affected_validation",
            "start_s": preconditions_end,
            "end_s": time.monotonic() - started,
            "completed_units": len(validation["validation_receipts"]),
            "checkpoint_position": "affected_receipts",
        },
    ]
    artifact["duration_s"] = time.monotonic() - started
    candidate = raw_dir / "terminal_candidate.json"
    atomic_json(candidate, artifact)
    progress(started, "terminal_readers", "start", candidate=str(candidate))
    provisional = checks.run_commands(
        root,
        terminal_commands(root, candidate),
        log_dir=raw_dir / "validation/terminal_provisional",
        extra_env={"JAX_PLATFORMS": "cpu"},
    )
    artifact["validation_receipts"]["terminal"] = provisional
    artifact["validation_receipts"]["terminal_outcomes_path"] = str(
        raw_dir / "terminal_reader_outcomes.json"
    )
    artifact["duration_s"] = time.monotonic() - started
    atomic_json(candidate, artifact)
    exact = checks.run_commands(
        root,
        terminal_commands(root, candidate),
        log_dir=raw_dir / "validation/terminal_exact",
        extra_env={"JAX_PLATFORMS": "cpu"},
    )
    outcomes = {
        "candidate_sha256": sha256_file(candidate),
        "receipts": exact,
        "all_passed": all(row["passed"] for row in exact),
    }
    atomic_json(raw_dir / "terminal_reader_outcomes.json", outcomes)
    progress(started, "terminal_readers", "complete", passed=outcomes["all_passed"])
    if not outcomes["all_passed"]:
        raise RuntimeError("terminal_reader_failed")
    output = output if output.is_absolute() else root / output
    atomic_json(output, artifact)
    progress(started, "publication", "complete", output=str(output), sha256=sha256_file(output))
    return artifact


def main(argv: list[str] | None = None) -> int:
    """Provide the declared run and two fresh-process terminal readers."""

    print("[exp7650] startup flushed", flush=True)
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", default="20260925")
    parser.add_argument("--output", type=Path, default=RESULT)
    parser.add_argument("--cold", type=Path)
    parser.add_argument("--independent", type=Path)
    args = parser.parse_args(argv)
    if args.cold:
        errors = cold_replay(args.cold)
        print(json.dumps({"errors": errors}), flush=True)
        return int(bool(errors))
    if args.independent:
        errors = independent_replay(args.independent)
        print(json.dumps({"errors": errors}), flush=True)
        return int(bool(errors))
    run_experiment(ROOT, args.date, args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
