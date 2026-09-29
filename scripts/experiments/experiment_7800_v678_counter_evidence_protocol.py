#!/usr/bin/env python3
"""Freeze the CPU source-deletion protocol; REQ-REPORT-7800."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))

from carnot import experiment_7800_v678_counter_evidence_protocol as study  # noqa: E402
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash, sha256_file  # noqa: E402
from carnot.reporting import experiment_7303_validation_scope as validation  # noqa: E402

RAW = Path("results/raw/experiment_7800_v678_counter_evidence_protocol")
OUTPUT = Path("results/experiment_7800_v678_counter_evidence_protocol.json")
PRINCIPLES = {
    "experiment_id": "Each result has one owner.",
    "honest_verdict": "External incompleteness cannot be fixed by retrying owned work.",
    "verdict_class": "Claim strength travels with the record.",
    "flagged_adversarial": "Invalid evidence cannot open a gate.",
    "gate_check_summary": "Missing evidence differs from a scientific null.",
    "rows": "Recompute every comparison from its units.",
    "acceptance_gate_results": "A working fixture proves no scientific gain.",
    "duration_s": "Duration reflects actual work.",
    "phase_spans": "Duration reflects actual work.",
    "random_seed": "Replay requires identical inputs.",
    "reproducibility_checksum": "Replay requires identical inputs.",
    "sample_size_budget": "Views and seeds are not new families.",
    "source_artifact_hashes": "Old files cannot replace missing current producers.",
    "preconditions_checked": "Cheap failures precede compute.",
    "validation_receipts": "Every required check must pass.",
    "verifier_is_oracle": "Fixture success is circular evidence.",
    "claim_scope": "Exposed data provide no hidden generalization.",
    "inference_substrate": "Floors follow invoked work.",
    "inference_substrate_class": "Floors follow invoked work.",
    "MODEL_SPECS": "Citing a model is not invoking it.",
    "model_specs": "Actual loaded bytes determine model identity.",
    "model_invocation_counts": "A call count must describe current calls.",
    "counter_evidence_ready_score": "Semantic interventions need known input changes.",
    "counter_evidence_protocol_path": "Outcome-dependent redesign invalidates pairing.",
    "family_manifest_path": "Outcome-dependent redesign invalidates pairing.",
    "intervention_fixture_rows": "Syntax-valid payloads can remove the wrong evidence.",
}


def progress(start: float, phase: str, event: str, units: int = 0) -> None:
    """Flush a measured phase boundary and completed units."""
    print(
        f"[exp7800] phase={phase} event={event} elapsed_s={time.monotonic() - start:.3f} completed_units={units}",
        flush=True,
    )


def fixture_rows(protocol: dict[str, Any]) -> list[dict[str, Any]]:
    """Exercise actual chat payload and parser with deterministic scripted replies."""
    source = "Café one here. Other two now. Third nice here."
    answer = "Café is here. Keep this answer."
    family = {
        "family_id": "fixture-utf8",
        "complete_source": source,
        "complete_response": answer,
        "source_sha256": study.digest(source.encode()),
        "response_sha256": study.digest(answer.encode()),
    }

    def reply(_: dict[str, Any]) -> dict[str, Any]:
        return {
            "choices": [
                {
                    "message": {
                        "content": json.dumps(
                            {"unsupported_probability": 0.4, "source_sentence_id": 0}
                        )
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 40, "completion_tokens": 12},
        }

    rows = study.capture_fixture(family, protocol, reply)
    source2 = "Only one sentence. A very long unrelated sentence with many extra tokens and names."
    other = {
        **family,
        "family_id": "fixture-unmatched",
        "complete_source": source2,
        "source_sha256": study.digest(source2.encode()),
    }
    rows.extend(study.capture_fixture(other, protocol, reply))
    rows.extend(
        study.capture_fixture(
            {
                **family,
                "family_id": "fixture-empty",
                "complete_source": "",
                "source_sha256": study.digest(b""),
            },
            protocol,
            reply,
        )
    )
    return rows


def cold_reduce(raw_path: Path) -> dict[str, Any]:
    """Reopen exact fixture rows in a fresh process without modified-source labels."""
    rows = json.loads(raw_path.read_text())
    reduced = study.reduce_pilot(rows, {"fixture-utf8": 0})
    if reduced["modified_source_labels_applied"] != 0 or reduced["matched_n"] != 1:
        raise ValueError("cold_reduction_mismatch")
    return reduced


def build_record(
    date: str,
    started: float,
    checks: list[dict[str, Any]],
    hashes: dict[str, Any],
    rows: list[dict[str, Any]],
    protocol_path: Path | None,
    manifest_path: Path | None,
    fixtures: list[dict[str, Any]],
    receipts: dict[str, Any],
    spans: list[dict[str, Any]],
    verdict: str,
) -> dict[str, Any]:
    """Keep readiness, blocked inputs, and CPU fixture claims separate."""
    ready = int(verdict == "complete_circular_positive_counter_evidence_protocol")
    cls = "circular_positive" if ready else "blocked" if "blocked" in verdict else "disqualified"
    source_hashes = {
        **hashes,
        "protocol": {
            "path": str(protocol_path) if protocol_path else None,
            "sha256": sha256_file(protocol_path) if protocol_path else None,
        },
        "family_manifest": {
            "path": str(manifest_path) if manifest_path else None,
            "sha256": sha256_file(manifest_path) if manifest_path else None,
        },
    }
    budget = {
        "intended": 48,
        "eligible": len(rows),
        "started": 0,
        "completed": 0,
        "excluded": 0,
        "censored": 0,
        "independent_n": len(rows),
        "planned_panel_calls": 144,
        "planned_canary_calls": 2,
    }
    return {
        "schema": "carnot.exp7800.counter_evidence.v1",
        "experiment_id": "exp7800-counter-evidence-protocol",
        "milestone": "2026.09.678",
        "run_date": date,
        "honest_verdict": verdict,
        "verdict_class": cls,
        "flagged_adversarial": False,
        "gate_check_summary": [item for item in checks if not item["passed"]],
        "rows": rows,
        "acceptance_gate_results": {
            "validity": bool(ready),
            "readiness": bool(ready),
            "probability_quality": None,
            "decision_benefit": None,
            "retention": None,
            "efficiency": None,
        },
        "duration_s": time.monotonic() - started,
        "phase_spans": spans,
        "random_seed": study.SEED,
        "reproducibility_checksum": canonical_hash(
            {
                "code": sha256_file(
                    ROOT / "python/carnot/experiment_7800_v678_counter_evidence_protocol.py"
                ),
                "entrypoint": sha256_file(
                    ROOT / "scripts/experiments/experiment_7800_v678_counter_evidence_protocol.py"
                ),
                "inputs": hashes,
                "protocol": sha256_file(protocol_path) if protocol_path else None,
                "seed": study.SEED,
            }
        ),
        "sample_size_budget": budget,
        "source_artifact_hashes": source_hashes,
        "preconditions_checked": checks,
        "validation_receipts": receipts,
        "verifier_is_oracle": True,
        "claim_scope": "Exposed RAGTruth evaluation development families; deterministic CPU fixture only; no semantic certificate or GPU measurement.",
        "inference_substrate": "verifier_ensemble_against_cached_candidates",
        "inference_substrate_class": "no_model_load",
        "planned_inference_substrate_class": "no_model_load",
        "MODEL_SPECS": [],
        "model_specs": [],
        "model_invocation_counts": {"loads": 0, "calls": 0, "input_tokens": 0, "output_tokens": 0},
        "counter_evidence_ready_score": ready,
        "counter_evidence_protocol_path": str(protocol_path) if protocol_path else None,
        "family_manifest_path": str(manifest_path) if manifest_path else None,
        "intervention_fixture_rows": fixtures,
        "field_principles": PRINCIPLES,
        "prior_retirement": {"same_verdict_matches": [], "retired": []},
        "production_promotion": False,
    }


def main(argv: list[str] | None = None) -> int:
    """Run one dated CPU protocol production or a separate cold raw reader."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", default="20260928")
    parser.add_argument("--reduce-raw", type=Path)
    args = parser.parse_args(argv)
    if args.reduce_raw:
        print(json.dumps(cold_reduce(args.reduce_raw), sort_keys=True), flush=True)
        return 0
    started = time.monotonic()
    progress(started, "startup", "start")
    raw = ROOT / RAW
    raw.mkdir(parents=True, exist_ok=True)
    spans: list[dict[str, Any]] = []
    begun = time.monotonic()
    progress(started, "preconditions", "before")
    families, checks, hashes = study.preflight(ROOT)
    for command in ("pytest", "coverage", "ruff", "mypy", "python"):
        path = ROOT / ".venv/bin" / command
        checks.append(study.check("exp7800_runtime", path, "exists", True, path.is_file()))
    checks.append(
        study.check(
            "exp7800_runtime",
            ROOT / "scripts/check_spec_coverage.py",
            "exists",
            True,
            (ROOT / "scripts/check_spec_coverage.py").is_file(),
        )
    )
    progress(started, "preconditions", "after", len(checks))
    spans.append(
        {
            "phase": "preconditions",
            "duration_s": time.monotonic() - begun,
            "completed_units": len(checks),
        }
    )
    rows: list[dict[str, Any]] = []
    fixtures: list[dict[str, Any]] = []
    protocol_path: Path | None = None
    manifest_path: Path | None = None
    receipts: dict[str, Any] = {}
    if not all(item["passed"] for item in checks):
        verdict = "complete_blocked_external_input"
    else:
        begun = time.monotonic()
        progress(started, "freeze", "before", len(families))
        selected = study.freeze_families(families)
        protocol = study.make_protocol(selected)
        protocol_path = raw / "counter_evidence_protocol.json"
        manifest_path = raw / "family_manifest.json"
        atomic_json(protocol_path, protocol)
        atomic_json(
            manifest_path,
            {
                "schema": "carnot.exp7800.family_manifest.v1",
                "selection": "seeded SHA-256(source_sha256), lowest 48 of evaluation64",
                "source_role": "evaluation",
                "prior_exposure": True,
                "families": selected,
            },
        )
        for family in selected:
            rows.append(
                {
                    "family_id": family["family_id"],
                    "role": "evaluation",
                    "source_sha256": family["source_sha256"],
                    "answer_sha256": family["response_sha256"],
                    "prior_exposure": True,
                    "official_split": "test",
                    "source_bytes": len(family["complete_source"].encode()),
                    "answer_bytes": len(family["complete_response"].encode()),
                    "source_sentence_count": len(
                        study.sentence_offsets(family["complete_source"].encode())
                    ),
                    "arms": {arm: "unstarted_no_model_load" for arm in study.ARMS},
                    "excluded": False,
                    "censored": False,
                    "denominator": 1,
                }
            )
        fixtures = fixture_rows(protocol)
        atomic_json(raw / "intervention_fixture_rows.json", fixtures)
        progress(started, "freeze", "after", len(rows))
        spans.append(
            {
                "phase": "freeze",
                "duration_s": time.monotonic() - begun,
                "completed_units": len(rows),
            }
        )
        begun = time.monotonic()
        progress(started, "validation", "before", 0)
        plan = json.loads((raw / "validation_plan.json").read_text())
        Path("/tmp/carnot-exp7800-pytest").mkdir(parents=True, exist_ok=True)
        Path("/tmp/carnot-exp7800-coverage").mkdir(parents=True, exist_ok=True)
        scoped = validation.run_scoped_validation(
            ROOT,
            plan["tests"],
            plan["changed_modules"],
            static_paths=plan["static_paths"],
            basetemp=Path("/tmp/carnot-exp7800-pytest"),
            coverage_file=Path("/tmp/carnot-exp7800-coverage/.coverage"),
            log_dir=raw / "validation",
        )
        receipts["scoped"] = scoped
        full = validation.run_commands(
            ROOT,
            [
                validation.CommandSpec(
                    "full_python_suite",
                    (
                        str(ROOT / ".venv/bin/pytest"),
                        "tests/python",
                        "-q",
                    ),
                    "repository_python",
                    timeout_s=1800,
                )
            ],
            log_dir=raw / "full_validation",
        )
        receipts["full_python_suite"] = full
        cold = validation.run_commands(
            ROOT,
            [
                validation.CommandSpec(
                    "cold_raw_reduction",
                    (
                        str(ROOT / ".venv/bin/python"),
                        "-u",
                        "scripts/experiments/experiment_7800_v678_counter_evidence_protocol.py",
                        "--reduce-raw",
                        str(raw / "intervention_fixture_rows.json"),
                    ),
                    "task_owned_e2e",
                    timeout_s=60,
                )
            ],
            log_dir=raw / "cold_validation",
        )
        receipts["cold_reduction"] = cold
        for item in [*scoped["validation_receipts"], *full, *cold]:
            if not item["passed"]:
                log_path = ROOT / item["log_path"]
                checks.append(
                    study.check(
                        "exp7800_validation",
                        log_path,
                        f"{item['name']}.exit_code",
                        0,
                        item["exit_code"],
                    )
                )
        valid = scoped["required_checks_passed"] and all(item["passed"] for item in [*full, *cold])
        verdict = (
            "complete_circular_positive_counter_evidence_protocol"
            if valid
            else "complete_disqualified_required_checks"
        )
        progress(
            started,
            "validation",
            "after",
            len(scoped["validation_receipts"]) + len(full) + len(cold),
        )
        spans.append(
            {
                "phase": "validation",
                "duration_s": time.monotonic() - begun,
                "completed_units": len(scoped["validation_receipts"]) + len(full) + len(cold),
            }
        )
    candidate = build_record(
        args.date,
        started,
        checks,
        hashes,
        rows,
        protocol_path,
        manifest_path,
        fixtures,
        receipts,
        spans,
        verdict,
    )
    progress(started, "candidate", "before_write", len(rows))
    atomic_json(ROOT / OUTPUT, candidate)
    progress(started, "candidate", "after_write", len(rows))
    readers = validation.run_commands(
        ROOT,
        [
            validation.CommandSpec(
                "adversarial_verify",
                (
                    str(ROOT / ".venv/bin/python"),
                    "-u",
                    "scripts/adversarial_verify.py",
                    str(ROOT / OUTPUT),
                ),
                "exact_candidate",
                180,
            ),
            validation.CommandSpec(
                "verdict_row_consistency_strict",
                (
                    str(ROOT / ".venv/bin/python"),
                    "-u",
                    "scripts/verdict_row_consistency_lint.py",
                    "--strict",
                    str(ROOT / OUTPUT),
                ),
                "exact_candidate",
                180,
            ),
        ],
        log_dir=raw / "terminal_readers",
    )
    atomic_json(raw / "terminal_reader_receipts.json", readers)
    progress(started, "terminal_readers", "after", len(readers))
    if not all(item["passed"] for item in readers):
        candidate["honest_verdict"] = "complete_disqualified_terminal_reader"
        candidate["verdict_class"] = "disqualified"
        candidate["counter_evidence_ready_score"] = 0
        candidate["acceptance_gate_results"]["validity"] = False
        candidate["acceptance_gate_results"]["readiness"] = False
        candidate["flagged_adversarial"] = not readers[0]["passed"]
        atomic_json(ROOT / OUTPUT, candidate)
    progress(started, "finish", "complete", len(rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
