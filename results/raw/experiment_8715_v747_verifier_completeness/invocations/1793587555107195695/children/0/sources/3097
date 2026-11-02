"""Orchestrate the CPU source-witness fixture and pilot custody experiment.

The verifier lives in ``carnot.verify.source_claim_witness``. This module
freezes inputs, runs bounded checks and publishes an evidence-only artifact.
Spec: REQ-REPORT-7644, SCENARIO-REPORT-7644-TERMINAL.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time
from typing import Any

from carnot.reporting import experiment_7303_validation_scope as checks
from carnot.reporting.current_work_receipt import atomic_json, sha256_file
from carnot.verify.source_claim_witness import FEATURE_NAMES, SCHEMA_VERSION, verify_claim


ROOT = Path(__file__).resolve().parents[2]
RESULT = Path("results/experiment_7644_v667_source_witness_prototype.json")
RAW = Path("results/raw/experiment_7644_v667_source_witness_prototype")
FIXTURES = Path("tests/python/fixtures/experiment_7644_source_witness.jsonl")
PILOT = Path("results/raw/experiment_7602_v664_evidence_requalification/pilot_model_inputs.jsonl")
SCHEMA_PATH = RAW / "schema.json"
MANIFEST = {
    "test_paths": ["tests/python/test_experiment_7644_v667_source_witness_prototype.py"],
    "changed_modules": [
        "python/carnot/verify/source_claim_witness.py",
        "python/carnot/experiment_7644_v667_source_witness_prototype.py",
    ],
    "static_paths": ["scripts/experiments/experiment_7644_v667_source_witness_prototype.py"],
}
NAMED_INPUTS = (
    "CLAUDE.md",
    "CODEX.md",
    "research-program.md",
    "ops/exclusion_manifest.yaml",
    "ops/e2e-test-plan.md",
    "scripts/experiment_template.py",
    "python/carnot/reporting/current_work_receipt.py",
    "python/carnot/reporting/experiment_7303_validation_scope.py",
    "openspec/capabilities/research-reporting/spec.md",
    "python/carnot/verify/code_structural_dependency_verifier.py",
    "python/carnot/experiment_7602_v664_evidence_requalification.py",
    "python/carnot/experiment_7616_v665_evidence_schema.py",
    "python/carnot/verify/source_claim_witness.py",
    "python/carnot/experiment_7644_v667_source_witness_prototype.py",
    "scripts/experiments/experiment_7644_v667_source_witness_prototype.py",
    PILOT.as_posix(),
    "research-references.md",
    FIXTURES.as_posix(),
)


def progress(started: float, phase: str, event: str, **details: Any) -> None:
    """Flush phase boundaries and elapsed time for the task owner."""

    print(
        f"[exp7644] {phase} {event} elapsed_s={time.monotonic() - started:.3f} {details}",
        flush=True,
    )


def validate_predictor_input(record: dict[str, Any]) -> None:
    """Reject label-sidecar handles before source data reaches the verifier."""

    if any("label" in key.lower() or "sidecar" in key.lower() for key in record):
        raise ValueError("label_or_sidecar_access_rejected")
    if set(record) != {"source", "claim", "closed_files"}:
        raise ValueError("predictor_input_fields_invalid")


def build_fixture_rows(cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Reduce each independently labeled fixture without letting labels into input."""

    rows = []
    for case in cases:
        model_input = {key: case[key] for key in ("source", "claim", "closed_files")}
        validate_predictor_input(model_input)
        witness = verify_claim(**model_input)
        rows.append(
            {
                "unit_id": case["id"],
                "arm": "structural_witness",
                "predicate": case["claim"],
                "independent_truth": case["expected"],
                "witness": witness,
                "abstention_reason": witness["reason"] if witness["status"] == "unknown" else None,
                "exact_bytes": case["source"].encode().hex(),
                "absolute_metric": int(witness["status"] == case["expected"]),
                "numerator": int(witness["status"] == case["expected"]),
                "denominator": 1,
                "raw_provenance": FIXTURES.as_posix(),
                "excluded": False,
                "censored": False,
            }
        )
    return rows


def cold_reduce_rows(path: Path, cases: list[dict[str, Any]]) -> dict[str, Any]:
    """Recompute fixture statuses from source bytes and independent annotations."""

    rows = json.loads(path.read_text(encoding="utf-8"))
    if len(rows) != len(cases):
        return {"passed": False, "matched": 0, "denominator": len(cases)}
    matched = 0
    for row, case in zip(rows, cases, strict=True):
        expected = verify_claim(case["source"], case["claim"], closed_files=case["closed_files"])
        if (
            row["unit_id"] == case["id"]
            and row["independent_truth"] == case["expected"]
            and row["witness"] == expected
            and expected["status"] == case["expected"]
        ):
            matched += 1
    return {"passed": matched == len(cases), "matched": matched, "denominator": len(cases)}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _source_hashes(root: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    inspected = []
    hashes: dict[str, Any] = {
        "producer_files": {},
        "pre_gate_receipts": {},
        "missing_inputs": [],
        "planned_outputs": [RESULT.as_posix(), SCHEMA_PATH.as_posix()],
    }
    for relative in NAMED_INPUTS:
        path = root / relative
        exists = path.is_file()
        inspected.append(
            {
                "check": "input_file_exists",
                "upstream": "named_input",
                "path": str(path),
                "field": "is_file",
                "operator": "eq",
                "expected": True,
                "observed": exists,
                "passed": exists,
            }
        )
        if exists:
            hashes["producer_files"][relative] = sha256_file(path)
        else:
            hashes["missing_inputs"].append(relative)
    return inspected, hashes


def _gate(
    name: str, value: bool | None, principle: str, operands: dict[str, Any]
) -> dict[str, Any]:
    return {"gate": name, "passed": value, "principle": principle, "measured_operands": operands}


def _pilot_rows(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Measure coverage on inherited source and answer bytes without labels."""

    rows = []
    for number, record in enumerate(records):
        source = record["complete_source"]
        claim = record["answer_sentences"][0]["text"]
        witness = verify_claim(source, claim)
        rows.append(
            {
                "unit_id": f"pilot-{number}",
                "arm": "structural_witness",
                "predicate": claim,
                "witness": witness,
                "absolute_metric": int(witness["status"] != "unknown"),
                "numerator": int(witness["status"] != "unknown"),
                "denominator": 1,
                "raw_provenance": PILOT.as_posix(),
                "source_sha256": record["source_sha256"],
                "excluded": False,
                "censored": False,
                "independent_truth": None,
            }
        )
    return rows


def _checksum(value: dict[str, Any]) -> str:
    copy = {key: item for key, item in value.items() if key != "reproducibility_checksum"}
    return "sha256:" + hashlib.sha256(json.dumps(copy, sort_keys=True).encode()).hexdigest()


def build_artifact(
    *,
    date: str,
    root: Path,
    fixture_rows: list[dict[str, Any]],
    pilot_rows: list[dict[str, Any]],
    preconditions: list[dict[str, Any]],
    hashes: dict[str, Any],
    receipts: list[dict[str, Any]],
    spans: list[dict[str, Any]],
    duration: float,
    terminal_ready: bool,
) -> dict[str, Any]:
    """Build a terminal report with separate fixture and scientific gates."""

    fixture_ok = bool(fixture_rows) and all(row["numerator"] == 1 for row in fixture_rows)
    validation_ok = all(row["passed"] for row in receipts) and bool(receipts)
    blocked = [row for row in preconditions if not row["passed"]]
    terminal_ok = terminal_ready and validation_ok
    ready = fixture_ok and terminal_ok and not blocked
    flagged = any(row["name"] == "adversarial_verify" and not row["passed"] for row in receipts)
    if blocked:
        verdict_class, verdict = "blocked", "complete_blocked_missing_named_input"
    elif not fixture_ok or (receipts and not validation_ok):
        verdict_class, verdict = "disqualified", "complete_disqualified_required_validation"
    elif ready:
        verdict_class, verdict = "circular_positive", "complete_circular_positive_fixture_witness"
    else:
        verdict_class, verdict = "null", "complete_null_pending_terminal_readers"
    gates = [
        _gate(
            "validity",
            validation_ok if receipts else None,
            "Current scoped checks and exact readers govern validity.",
            {
                "passed_receipts": sum(row["passed"] for row in receipts),
                "receipt_count": len(receipts),
            },
        ),
        _gate(
            "readiness",
            ready,
            "Complete independent fixtures and terminal readers govern readiness.",
            {
                "fixture_matches": sum(row["numerator"] for row in fixture_rows),
                "fixture_count": len(fixture_rows),
                "terminal_ready": terminal_ready,
            },
        ),
        _gate(
            "probability_benefit",
            None,
            "Proper loss requires oracle-distinct corpus labels and paired probabilities.",
            {"labeled_corpus_groups": 0, "paired_probabilities": 0},
        ),
        _gate(
            "utility",
            None,
            "Decision value requires typed actions, costs and independent outcomes.",
            {"typed_decisions": 0},
        ),
        _gate(
            "retention",
            None,
            "Retention requires delayed feedback and replay on independent groups.",
            {"delayed_feedback_events": 0},
        ),
        _gate(
            "freshness",
            None,
            "Freshness requires a newly held-out independent roster.",
            {"fresh_independent_groups": 0},
        ),
    ]
    artifact = {
        "experiment_id": "exp7644-source-witness-prototype",
        "milestone": "2026.09.667",
        "run_date": date,
        "honest_verdict": verdict,
        "verdict_class": verdict_class,
        "flagged_adversarial": flagged,
        "gate_check_summary": blocked,
        "acceptance_gate_results": gates,
        "rows": [*fixture_rows, *pilot_rows],
        "fixture_rows": fixture_rows,
        "sample_size_budget": {
            "intended_independent_groups": len(fixture_rows) + len(pilot_rows),
            "observed_independent_groups": len(fixture_rows) + len(pilot_rows),
            "eligible": len(fixture_rows) + len(pilot_rows),
            "excluded": 0,
            "censored": 0,
            "exposure_limits": "48 fixed fixtures and eight inherited pilot inputs; no repeated views",
        },
        "preconditions_checked": preconditions,
        "inference_substrate": "CPU AST parsing and fixture/pilot reduction only",
        "inference_substrate_class": "no_model_load",
        "planned_inference_substrate_class": "no_model_load",
        "MODEL_SPECS": [],
        "planned_MODEL_SPECS": [],
        "model_invoked": False,
        "invocation_counts": {"loads": 0, "forwards": 0, "generations": 0, "tokens": 0},
        "execution_venue": "host",
        "execution_venue_details": {"hostname": os.uname().nodename, "owned_pid": os.getpid()},
        "phase_spans": spans,
        "duration_s": duration,
        "random_seed": {
            "value": 7644001,
            "purpose": "fixture roster is fixed; no stochastic model calls",
        },
        "source_artifact_hashes": hashes,
        "validation_receipts": receipts,
        "verifier_is_oracle": True,
        "witness_ready_score": int(ready),
        "witness_schema_path": SCHEMA_PATH.as_posix(),
        "semantic_entailment_claim": False,
        "field_principles": {
            "honest_verdict": "Terminal completion and scientific benefit are distinct.",
            "verdict_class": "Fixed verdict enum, with fixture agreement circular positive.",
            "rows": "Each independent source group counts once; repeated views do not add sample size.",
            "witness_ready_score": "Exact fixtures, offset mapping and terminal readers govern readiness.",
            "flagged_adversarial": "A flagged reader cannot open a downstream gate.",
            "probability_benefit": "Proper loss needs independent labels and probabilities.",
            "utility": "Decision value needs typed costs and actions.",
            "retention": "Learning retention needs delayed replay.",
            "freshness": "Reused pilot inputs are not fresh evidence.",
            "semantic_entailment_claim": "Structural facts never certify surrounding prose.",
        },
    }
    artifact["reproducibility_checksum"] = _checksum(artifact)
    return artifact


def cold_replay(path: Path, root: Path = ROOT) -> dict[str, Any]:
    """Read an unpublished artifact afresh and verify immutable input hashes."""

    artifact = json.loads(path.read_text(encoding="utf-8"))
    original = artifact.get("reproducibility_checksum")
    hashes = artifact["source_artifact_hashes"]["producer_files"]
    hash_ok = all(sha256_file(root / name) == value for name, value in hashes.items())
    cases = _read_jsonl(root / FIXTURES)
    rows_path = root / RAW / "fixture_rows.json"
    fixture = cold_reduce_rows(rows_path, cases)
    schema = json.loads((root / SCHEMA_PATH).read_text(encoding="utf-8"))
    passed = (
        original == _checksum(artifact)
        and hash_ok
        and fixture["passed"]
        and artifact["fixture_rows"] == json.loads(rows_path.read_text())
        and schema["schema_version"] == SCHEMA_VERSION
        and schema["feature_names"] == list(FEATURE_NAMES)
    )
    return {
        "passed": passed,
        "checksum_valid": original == _checksum(artifact),
        "input_hashes_valid": hash_ok,
        "fixture_reduction": fixture,
    }


def independent_reduce(path: Path, root: Path = ROOT) -> dict[str, Any]:
    """Recompute coverage and fixture comparisons from raw input, not summaries."""

    artifact = json.loads(path.read_text(encoding="utf-8"))
    cases = _read_jsonl(root / FIXTURES)
    pilot = _read_jsonl(root / PILOT)
    expected_fixture = build_fixture_rows(cases)
    expected_pilot = _pilot_rows(pilot)
    rows = artifact["rows"]
    passed = (
        artifact["fixture_rows"] == expected_fixture
        and rows == [*expected_fixture, *expected_pilot]
        and artifact["sample_size_budget"]["observed_independent_groups"] == len(rows)
    )
    return {
        "passed": passed,
        "fixture_matches": sum(row["numerator"] for row in expected_fixture),
        "fixture_denominator": len(expected_fixture),
        "pilot_coverage_numerator": sum(row["numerator"] for row in expected_pilot),
        "pilot_coverage_denominator": len(expected_pilot),
    }


def _span(
    name: str, started: float, done: float, completed: int, checkpoint: str
) -> dict[str, Any]:
    return {
        "phase": name,
        "start_offset_s": started,
        "end_offset_s": done,
        "duration_s": done - started,
        "completed_units": completed,
        "checkpoint": checkpoint,
    }


def _terminal_commands(root: Path, candidate: Path) -> list[checks.CommandSpec]:
    python = str(root / ".venv/bin/python")
    module = "carnot.experiment_7644_v667_source_witness_prototype"
    return [
        checks.CommandSpec(
            "cold_replay",
            (python, "-m", module, "--cold-replay", str(candidate)),
            "exact_candidate",
            180,
        ),
        checks.CommandSpec(
            "independent_reduce",
            (python, "-m", module, "--independent-reduce", str(candidate)),
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
            "verdict_row_consistency",
            (python, "scripts/verdict_row_consistency_lint.py", "--strict", str(candidate)),
            "exact_candidate",
            180,
        ),
    ]


def run_experiment(root: Path, date: str, output: Path) -> dict[str, Any]:
    """Authenticate, freeze, reduce, validate and atomically publish the task."""

    root = root.resolve()
    started = time.monotonic()
    spans: list[dict[str, Any]] = []
    progress(started, "preconditions", "begin")
    checks_done, hashes = _source_hashes(root)
    spans.append(
        _span("preconditions", 0, time.monotonic() - started, len(checks_done), "input_hashes")
    )
    progress(started, "preconditions", "end", checked=len(checks_done))
    raw = root / RAW
    raw.mkdir(parents=True, exist_ok=True)
    progress(started, "schema_freeze", "begin")
    schema = {
        "schema_version": SCHEMA_VERSION,
        "feature_names": list(FEATURE_NAMES),
        "status_enum": ["supported", "contradicted", "unknown"],
        "fields": [
            "proposition_checked",
            "source_offset",
            "source_sha256",
            "reason",
            "parser_completeness",
            "residual_unverified_span",
        ],
    }
    schema_path = root / SCHEMA_PATH
    if schema_path.exists() and json.loads(schema_path.read_text()) != schema:
        raise ValueError("frozen_witness_schema_changed")
    atomic_json(schema_path, schema)
    spans.append(
        _span(
            "schema_freeze",
            spans[-1]["end_offset_s"],
            time.monotonic() - started,
            1,
            SCHEMA_PATH.as_posix(),
        )
    )
    progress(started, "schema_freeze", "end")
    if hashes["missing_inputs"]:
        blocked = build_artifact(
            date=date,
            root=root,
            fixture_rows=[],
            pilot_rows=[],
            preconditions=checks_done,
            hashes=hashes,
            receipts=[],
            spans=spans,
            duration=time.monotonic() - started,
            terminal_ready=False,
        )
        progress(started, "publication", "before_atomic", verdict=blocked["honest_verdict"])
        atomic_json(output, blocked)
        progress(started, "publication", "after_atomic")
        return blocked
    progress(started, "fixture_and_pilot", "begin")
    cases = _read_jsonl(root / FIXTURES)
    records = _read_jsonl(root / PILOT)
    fixture_rows = build_fixture_rows(cases)
    pilot_rows = _pilot_rows(records)
    atomic_json(raw / "fixture_rows.json", fixture_rows)
    atomic_json(raw / "pilot_rows.json", pilot_rows)
    spans.append(
        _span(
            "fixture_and_pilot",
            spans[-1]["end_offset_s"],
            time.monotonic() - started,
            len(fixture_rows) + len(pilot_rows),
            "fixture_rows.json,pilot_rows.json",
        )
    )
    progress(started, "fixture_and_pilot", "end", completed=len(fixture_rows) + len(pilot_rows))
    progress(started, "affected_validation", "before_subprocess")
    validation_start = time.monotonic() - started
    with tempfile.TemporaryDirectory(prefix="exp7644-validation-") as private:
        private_root = Path(private)
        (private_root / "basetemp").mkdir()
        command_plan = checks.build_scoped_commands(
            root,
            MANIFEST["test_paths"],
            MANIFEST["changed_modules"],
            static_paths=MANIFEST["static_paths"],
            basetemp=private_root / "basetemp",
            coverage_file=private_root / "coverage.data",
        )
        # Keep the temporary parent alive for the entire scoped pytest run.
        receipts = checks.run_commands(
            root,
            command_plan,
            log_dir=raw / "validation" / "affected",
            extra_env={
                "JAX_PLATFORMS": "cpu",
                "COVERAGE_FILE": str(private_root / "coverage.data"),
            },
            heartbeat_s=30.0,
        )
    spans.append(
        _span(
            "affected_validation",
            validation_start,
            time.monotonic() - started,
            len(receipts),
            "validation/affected",
        )
    )
    progress(started, "affected_validation", "after_subprocess", completed=len(receipts))
    candidate_path = raw / "exact_terminal_candidate.json"
    artifact = build_artifact(
        date=date,
        root=root,
        fixture_rows=fixture_rows,
        pilot_rows=pilot_rows,
        preconditions=checks_done,
        hashes=hashes,
        receipts=receipts,
        spans=spans,
        duration=time.monotonic() - started,
        terminal_ready=True,
    )
    atomic_json(candidate_path, artifact)
    progress(started, "terminal_readers", "before_subprocess")
    terminal_start = time.monotonic() - started
    terminal_plan = _terminal_commands(root, candidate_path)
    terminal = checks.run_commands(
        root,
        terminal_plan,
        log_dir=raw / "validation" / "terminal",
        extra_env={"JAX_PLATFORMS": "cpu"},
        heartbeat_s=30.0,
    )
    spans.append(
        _span(
            "terminal_readers",
            terminal_start,
            time.monotonic() - started,
            len(terminal),
            "validation/terminal",
        )
    )
    progress(started, "terminal_readers", "after_subprocess", completed=len(terminal))
    final = build_artifact(
        date=date,
        root=root,
        fixture_rows=fixture_rows,
        pilot_rows=pilot_rows,
        preconditions=checks_done,
        hashes=hashes,
        receipts=[*receipts, *terminal],
        spans=spans,
        duration=time.monotonic() - started,
        terminal_ready=all(item["passed"] for item in terminal),
    )
    atomic_json(candidate_path, final)
    progress(started, "exact_terminal_replay", "before_subprocess")
    exact = checks.run_commands(
        root,
        terminal_plan,
        log_dir=raw / "validation" / "terminal_exact",
        extra_env={"JAX_PLATFORMS": "cpu"},
        heartbeat_s=30.0,
    )
    atomic_json(raw / "exact_terminal_reader_outcomes.json", {"outcomes": exact})
    progress(started, "exact_terminal_replay", "after_subprocess", completed=len(exact))
    spans.append(
        _span(
            "exact_terminal_replay",
            spans[-1]["end_offset_s"],
            time.monotonic() - started,
            len(exact),
            "exact_terminal_reader_outcomes.json",
        )
    )
    if [(r["name"], r["passed"]) for r in exact] != [(r["name"], r["passed"]) for r in terminal]:
        raise RuntimeError("exact_terminal_reader_outcomes_changed")
    if not all(r["passed"] for r in exact):
        final = build_artifact(
            date=date,
            root=root,
            fixture_rows=fixture_rows,
            pilot_rows=pilot_rows,
            preconditions=checks_done,
            hashes=hashes,
            receipts=[*receipts, *terminal, *exact],
            spans=spans,
            duration=time.monotonic() - started,
            terminal_ready=False,
        )
    final["terminal_reader_outcomes_path"] = (
        RAW / "exact_terminal_reader_outcomes.json"
    ).as_posix()
    final["phase_spans"] = spans
    final["duration_s"] = time.monotonic() - started
    final["reproducibility_checksum"] = _checksum(final)
    progress(started, "publication", "before_atomic")
    atomic_json(output, final)
    progress(started, "publication", "after_atomic", verdict=final["honest_verdict"])
    return final


def main(argv: list[str] | None = None) -> int:
    """Run the declared producer or one read-only fresh-process reader."""

    print("[exp7644] startup flushed", flush=True)
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--date")
    modes.add_argument("--cold-replay", type=Path)
    modes.add_argument("--independent-reduce", type=Path)
    parser.add_argument("--output", type=Path, default=RESULT)
    args = parser.parse_args(argv)
    if args.cold_replay:
        value = cold_replay(args.cold_replay)
        print(json.dumps(value, sort_keys=True), flush=True)
        return int(not value["passed"])
    if args.independent_reduce:
        value = independent_reduce(args.independent_reduce)
        print(json.dumps(value, sort_keys=True), flush=True)
        return int(not value["passed"])
    if args.date != "20260925" or args.output != RESULT:
        parser.error("date or output differs from the declared task")
    value = run_experiment(ROOT, args.date, ROOT / args.output)
    print(
        json.dumps({"honest_verdict": value["honest_verdict"], "result": str(args.output)}),
        flush=True,
    )
    return int(value["verdict_class"] == "disqualified")


if __name__ == "__main__":
    raise SystemExit(main())
