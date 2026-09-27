#!/usr/bin/env python3
"""Run REQ-REPORT-7768 from frozen inputs and publish one terminal receipt."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import time
from typing import Any

from carnot import experiment_7768_v676_source_view_qualification as exp
from carnot.reporting.current_work_receipt import atomic_json, sha256_file
from carnot.reporting.experiment_7303_validation_scope import (
    CommandSpec,
    build_scoped_commands,
    run_commands,
)


def _span(name: str, began: float, units: int) -> dict[str, Any]:
    return {"phase": name, "duration_s": time.monotonic() - began, "completed_units": units}


def _failure(receipt: dict[str, Any]) -> dict[str, Any]:
    return {
        "upstream_id": "exp7768",
        "artifact_path": receipt["log_path"],
        "artifact_hash": receipt["log_sha256"],
        "field": f"{receipt['name']}.exit_code",
        "expected": 0,
        "observed": receipt["exit_code"],
        "operator": "==",
    }


def _sources(root: Path) -> list[dict[str, Any]]:
    items = (
        (
            "exp7727",
            "results/experiment_7727_v673_development_corpus.json",
            ["development_cohort_ready_score", "development_manifest_sha256"],
            True,
        ),
        (
            "exp7727_manifest",
            "results/raw/experiment_7727_v673_development_corpus/development_manifest.json",
            ["counts", "roles"],
            True,
        ),
        (
            "exp7753_conductor_pre_gate",
            "results/experiment_7753_v675_contract_methods.json",
            ["contract_ready_score"],
            True,
        ),
        (
            "exp7754_historical",
            "results/experiment_7754_v675_sentence_protocol.json",
            ["honest_verdict", "sentence_protocol_ready_score"],
            False,
        ),
        (
            "exp7756_historical",
            "results/experiment_7756_v675_evidence_view_protocol.json",
            ["honest_verdict", "evidence_view_ready_score"],
            False,
        ),
    )
    return [
        {
            "upstream_id": owner,
            "path": str(root / path),
            "sha256": sha256_file(root / path) if (root / path).is_file() else None,
            "date": "20260927",
            "imported_fields": fields,
            "eligible": eligible,
        }
        for owner, path, fields, eligible in items
    ]


def validation_commands(scope: dict[str, Any], private: Path) -> list[CommandSpec]:
    """Test the full closure while tracing only the code this task adds."""
    commands = build_scoped_commands(
        exp.ROOT,
        scope["test_paths"],
        scope["changed_modules"],
        static_paths=scope["static_paths"],
        basetemp=private / "pytest",
        coverage_file=private / ".coverage",
    )
    coverage = next(
        index for index, spec in enumerate(commands) if spec.name == "changed_module_coverage"
    )
    spec = commands[coverage]
    argv = tuple(part for part in spec.argv if part not in scope["test_paths"][1:])
    commands[coverage] = CommandSpec(spec.name, argv, "direct_new_module_tests", spec.timeout_s)
    return commands


def _artifact(
    start: float,
    date: str,
    checks: list[dict[str, Any]],
    sources: list[dict[str, Any]],
    prepared: dict[str, Any] | None,
    scope: dict[str, Any],
    spans: list[dict[str, Any]],
) -> dict[str, Any]:
    rows = []
    coverage = []
    if prepared is not None:
        coverage = prepared["annotation_coverage_rows"]
        labels = {row["family_id"]: row for row in prepared["targets"]}
        for row in prepared["rows"]:
            label = labels[row["family_id"]]
            rows.append(
                {
                    "family_id": row["family_id"],
                    "role": row["role"],
                    "raw_path": str(exp.RAW / "rows.jsonl"),
                    "source_sha256": row["source_sha256"],
                    "answer_sha256": row["response_sha256"],
                    "source_windows_a": len(row["view_a"]["windows"]),
                    "source_windows_b": len(row["view_b"]["windows"]),
                    "answer_units": len(row["view_a"]["answer_units"]),
                    "null_windows": {"a": 1, "b": 1},
                    "prior_exposure": row["prior_exposure"],
                    "fresh_generalization_eligible": row["fresh_generalization_eligible"],
                    "label": label["response_label"],
                    "label_authority": "evaluator_only_after_public_seal",
                    "arms": row["arms"],
                    "abstention": row["abstention"],
                    "excluded": False,
                    "censored": row["abstention"] is not None,
                }
            )
    failures = [check for check in checks if not check["passed"]]
    bound = {
        "sources": sources,
        "manifest": sha256_file(exp.RAW / "source_view_manifest.json") if prepared else None,
        "code": sha256_file(Path(exp.__file__)),
        "wrapper": sha256_file(Path(__file__)),
        "roles": exp.COUNTS,
        "arms": sorted(exp.views.ARMS),
        "seed": 67601,
    }
    return {
        "experiment_id": "exp7768-source-view-qualification",
        "milestone": "2026.09.676",
        "run_date": date,
        "honest_verdict": "complete_blocked_external_prerequisite"
        if failures
        else "complete_disqualified_required_validation",
        "verdict_class": "blocked" if failures else "disqualified",
        "flagged_adversarial": False,
        "gate_check_summary": failures,
        "rows": rows,
        "annotation_coverage_rows": coverage,
        "acceptance_gate_results": {
            "validity": bool(prepared),
            "readiness": 0,
            "probability_quality": None,
            "decision_benefit": None,
            "retention": None,
            "efficiency": None,
        },
        "duration_s": time.monotonic() - start,
        "phase_spans": spans,
        "random_seed": 67601,
        "reproducibility_checksum": "sha256:"
        + hashlib.sha256(json.dumps(bound, sort_keys=True).encode()).hexdigest(),
        "sample_size_budget": {
            "intended": 640,
            "eligible": len(rows),
            "started": len(rows),
            "completed": len(rows),
            "excluded": 0,
            "censored": sum(row["censored"] for row in rows),
            "effective_independent_N": len(rows),
        },
        "source_artifact_hashes": sources,
        "preconditions_checked": {
            "checks": checks,
            "backend": "cpu",
            "host": platform.node(),
            "cpu_count": os.cpu_count(),
            "scope_path": str(exp.SCOPE),
            "scope_sha256": sha256_file(exp.SCOPE),
        },
        "validation_receipts": {
            "frozen_affected_scope": scope,
            "commands": [],
            "broad_collection_diagnostic": None,
            "cold_replay": None,
            "terminal_readers": None,
            "real_entrypoint_e2e": None,
        },
        "verifier_is_oracle": False,
        "claim_scope": {
            "value": "exposed_development_only",
            "fresh_generalization_eligible": False,
        },
        "field_principles": exp.PRINCIPLES,
        "inference_substrate": "verifier_ensemble_against_cached_candidates",
        "inference_substrate_class": "no_model_load",
        "planned_inference_substrate_class": "no_model_load",
        "MODEL_SPECS": [],
        "model_specs": [],
        "model_invocation_counts": {
            "loads": 0,
            "generation_calls": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "loaded_files": [],
        },
        "sentence_protocol_ready_score": 0,
        "evidence_view_ready_score": 0,
        "sentence_protocol_manifest_path": {
            "path": str(exp.RAW / "source_view_manifest.json"),
            "sha256": sha256_file(exp.RAW / "source_view_manifest.json"),
        }
        if prepared
        else None,
        "source_view_manifest_path": {
            "path": str(exp.RAW / "source_view_manifest.json"),
            "sha256": sha256_file(exp.RAW / "source_view_manifest.json"),
        }
        if prepared
        else None,
    }


def run(date: str) -> dict[str, Any]:
    """Execute one bounded qualification and publish only terminal bytes."""
    if date != "20260927":
        raise ValueError("run_date")
    start = time.monotonic()
    exp.RAW.mkdir(parents=True, exist_ok=True)
    scope = json.loads(exp.SCOPE.read_text())
    private = Path("/tmp") / f"carnot-7768-{os.getpid()}"
    exp.prepare_basetemp(private / "pytest")
    spans: list[dict[str, Any]] = []
    exp.progress(start, "preflight", "begin")
    began = time.monotonic()
    checks = exp.preflight(exp.ROOT)
    sources = _sources(exp.ROOT)
    spans.append(_span("preflight", began, len(checks)))
    exp.progress(start, "preflight", "complete", len(checks))
    prepared = None
    if all(check["passed"] for check in checks):
        exp.progress(start, "prepare", "begin")
        began = time.monotonic()
        prepared = exp.prepare_corpus(exp.MANIFEST, exp.RAW)
        spans.append(_span("prepare", began, len(prepared["rows"])))
        exp.progress(start, "prepare", "complete", len(prepared["rows"]))
    result = _artifact(start, date, checks, sources, prepared, scope, spans)
    candidate = exp.RAW / "candidate.json"
    if prepared is not None:
        exp.progress(start, "broad_collection", "before_subprocess")
        began = time.monotonic()
        diagnostic = run_commands(
            exp.ROOT,
            [
                CommandSpec(
                    "broad_python_collection",
                    (
                        str(exp.ROOT / ".venv/bin/pytest"),
                        "tests/python",
                        "-q",
                        "-n",
                        "0",
                        "-o",
                        "addopts=",
                        "--no-cov",
                        f"--basetemp={private / 'pytest/broad'}",
                    ),
                    "repository_health_only",
                    1800,
                )
            ],
            log_dir=exp.RAW / "diagnostic_logs",
            heartbeat_s=30,
        )
        result["validation_receipts"]["broad_collection_diagnostic"] = diagnostic[0]
        spans.append(_span("broad_collection", began, 1))
        exp.progress(start, "broad_collection", "after_subprocess", 1)
        exp.progress(start, "affected_validation", "begin")
        began = time.monotonic()
        commands = validation_commands(scope, private)
        receipts = run_commands(
            exp.ROOT,
            commands,
            log_dir=exp.RAW / "validation_logs",
            extra_env={"JAX_PLATFORMS": "cpu", "COVERAGE_FILE": str(private / ".coverage")},
            heartbeat_s=30,
        )
        result["validation_receipts"]["commands"] = receipts
        result["gate_check_summary"].extend(_failure(row) for row in receipts if not row["passed"])
        spans.append(_span("affected_validation", began, len(receipts)))
        exp.progress(start, "affected_validation", "complete", len(receipts))
        exp.progress(start, "cold_replay", "before_subprocess")
        began = time.monotonic()
        cold = run_commands(
            exp.ROOT,
            [
                CommandSpec(
                    "cold_replay",
                    (
                        str(exp.ROOT / ".venv/bin/python"),
                        "-u",
                        exp.WRAPPER,
                        "--cold-replay",
                        str(exp.RAW),
                    ),
                    "all_640_records",
                    900,
                )
            ],
            log_dir=exp.RAW / "cold_logs",
            heartbeat_s=30,
        )
        result["validation_receipts"]["cold_replay"] = cold[0]
        result["validation_receipts"]["real_entrypoint_e2e"] = {
            "direct_preparation_families": len(prepared["rows"]),
            "fresh_process_cold_replay": cold[0]["passed"],
        }
        result["gate_check_summary"].extend(_failure(row) for row in cold if not row["passed"])
        spans.append(_span("cold_replay", began, 1))
        exp.progress(start, "cold_replay", "after_subprocess", 1)
        if not result["gate_check_summary"]:
            result["verdict_class"] = "null"
            result["honest_verdict"] = "complete_null_source_view_qualification"
            result["sentence_protocol_ready_score"] = 1
            result["evidence_view_ready_score"] = 1
            result["acceptance_gate_results"]["readiness"] = 1
        atomic_json(candidate, result)
        exp.progress(start, "terminal", "before_subprocess")
        began = time.monotonic()
        terminal = run_commands(
            exp.ROOT,
            [
                CommandSpec(
                    "adversarial_verify",
                    (
                        str(exp.ROOT / ".venv/bin/python"),
                        "-u",
                        "scripts/adversarial_verify.py",
                        "--json",
                        str(candidate),
                    ),
                    "exact_candidate",
                    300,
                ),
                CommandSpec(
                    "verdict_row_consistency_strict",
                    (
                        str(exp.ROOT / ".venv/bin/python"),
                        "-u",
                        "scripts/verdict_row_consistency_lint.py",
                        "--strict",
                        str(candidate),
                    ),
                    "exact_candidate",
                    300,
                ),
            ],
            log_dir=exp.RAW / "terminal_logs",
            heartbeat_s=30,
        )
        result["validation_receipts"]["terminal_readers"] = terminal
        result["flagged_adversarial"] = not terminal[0]["passed"]
        result["gate_check_summary"].extend(_failure(row) for row in terminal if not row["passed"])
        spans.append(_span("terminal", began, len(terminal)))
        exp.progress(start, "terminal", "after_subprocess", len(terminal))
        if result["gate_check_summary"]:
            result["verdict_class"] = "disqualified"
            result["honest_verdict"] = "complete_disqualified_required_validation"
            result["sentence_protocol_ready_score"] = 0
            result["evidence_view_ready_score"] = 0
            result["acceptance_gate_results"]["readiness"] = 0
    result["phase_spans"] = spans
    result["duration_s"] = time.monotonic() - start
    atomic_json(candidate, result)
    exp.OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    temporary = exp.OUTPUT.with_suffix(".json.tmp")
    temporary.write_bytes(candidate.read_bytes())
    os.replace(temporary, exp.OUTPUT)
    exp.progress(start, "publish", "complete", 1)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", default="20260927")
    parser.add_argument("--cold-replay", type=Path)
    args = parser.parse_args(argv)
    if args.cold_replay:
        exp.replay_corpus(exp.MANIFEST, args.cold_replay)
        print("cold replay passed", flush=True)
        return 0
    result = run(args.date)
    return 0 if result["verdict_class"] in {"null", "blocked"} else 1


if __name__ == "__main__":
    sys.exit(main())
