#!/usr/bin/env python3
"""Run the Exp7755 fixture training and exact terminal checks."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
from typing import Any

from carnot import experiment_7755_v675_training_runtime as experiment
from carnot.reporting.current_work_receipt import atomic_json, sha256_file
from carnot.reporting.experiment_7303_validation_scope import (
    CommandSpec,
    build_scoped_commands,
    run_commands,
)
from carnot.verify import training_runtime as runtime

ROOT = Path(__file__).resolve().parents[2]
NAME = "experiment_7755_v675_training_runtime"
RAW = ROOT / "results/raw" / NAME
OUTPUT = ROOT / "results" / f"{NAME}.json"
SCOPE = Path("/tmp/exp7755_validation_scope.json")
PRINCIPLES = {
    "experiment_id": "An artifact must have a unique current owner.",
    "honest_verdict": "A terminal record must not waste attempts on unchanged inputs.",
    "verdict_class": "The claim class travels with the evidence.",
    "flagged_adversarial": "Invalid evidence must not open downstream gates.",
    "gate_check_summary": "Missing producers and failed scientific thresholds are different causes.",
    "rows": "Aggregates must be recomputable without rerunning science.",
    "acceptance_gate_results": "A working protocol is not evidence of benefit.",
    "duration_s": "Duration must describe actual work without padding.",
    "reproducibility_checksum": "A third party needs the same experiment inputs.",
    "sample_size_budget": "Repeated views and seeds do not increase independent family count.",
    "source_artifact_hashes": "A missing producer cannot be replaced with a convenient old result.",
    "preconditions_checked": "Access and validity must be established before expensive work.",
    "validation_receipts": "All registered checks must pass before readiness opens.",
    "verifier_is_oracle": "Execution truth and independent semantic verification are distinct claims.",
    "inference_substrate_class": "Duration floors must match the invoked substrate.",
    "MODEL_SPECS": "A cited upstream model is not a current model invocation.",
    "training_runtime_ready_score": "A fixture learner must execute before a natural comparison.",
    "training_protocol_path": "Outcomes cannot select the training recipe.",
    "fixture_training_rows": "A no-op or numerically invalid optimizer must fail.",
}


def _span(name: str, began: float, units: int) -> dict[str, Any]:
    return {"phase": name, "duration_s": time.monotonic() - began, "completed_units": units}


def _source_checks() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    sources = []
    observations = []
    for identifier, path, field, expected in (
        (
            "exp7740-sentence-label-protocol",
            ROOT / "results/experiment_7740_v674_sentence_label_protocol.json",
            "sentence_protocol_ready_score",
            1,
        ),
        (
            "exp7741-training-qualification",
            ROOT / "results/experiment_7741_v674_training_qualification.json",
            "training_runtime_ready_score",
            1,
        ),
        (
            "exp7741-conductor-pre-gate",
            ROOT / "results/experiment_7741_training_qualification.json",
            "honest_verdict",
            "complete_qualified_training_runtime",
        ),
        (
            "exp7754-sentence-protocol",
            ROOT / "results/experiment_7754_v675_sentence_protocol.json",
            "sentence_protocol_ready_score",
            1,
        ),
    ):
        exists = path.is_file()
        sha = sha256_file(path) if exists else None
        data = json.loads(path.read_text()) if exists else {}
        observed = data.get(field)
        eligible = exists and observed == expected
        sources.append(
            {
                "upstream_id": identifier,
                "artifact_path": str(path.relative_to(ROOT)),
                "artifact_hash": sha,
                "run_date": data.get("run_date"),
                "imported_fields": {field: observed},
                "eligibility": eligible,
            }
        )
        if not eligible:
            observations.append(
                {
                    "upstream_id": identifier,
                    "artifact_path": str(path.relative_to(ROOT)),
                    "artifact_hash": sha,
                    "field": field,
                    "expected": expected,
                    "observed": observed,
                    "operator": "==",
                    "role": "historical_natural_input_not_fixture_gate",
                }
            )
    return sources, observations


def _command(name: str, argv: tuple[str, ...], timeout: float = 900.0) -> CommandSpec:
    return CommandSpec(name, argv, "exp7755_exact_candidate", timeout)


def _run(name: str, argv: tuple[str, ...], timeout: float = 900.0) -> dict[str, Any]:
    return run_commands(
        ROOT, [_command(name, argv, timeout)], log_dir=RAW / "validation_logs", heartbeat_s=30.0
    )[0]


def _candidate(
    fixture: dict[str, Any],
    sources: list[dict[str, Any]],
    observations: list[dict[str, Any]],
    protocol_path: Path,
    started: float,
    spans: list[dict[str, Any]],
    receipts: list[dict[str, Any]],
) -> dict[str, Any]:
    scope = json.loads(SCOPE.read_text())
    input_hashes = {
        str(path.relative_to(ROOT)): sha256_file(path)
        for path in (
            protocol_path,
            ROOT / "python/carnot/verify/training_runtime.py",
            ROOT / "python/carnot/experiment_7755_v675_training_runtime.py",
            ROOT / "scripts/experiments/experiment_7755_v675_training_runtime.py",
        )
    }
    input_hashes.update({item["artifact_path"]: item["artifact_hash"] for item in sources})
    checksum = hashlib.sha256(
        json.dumps(
            {
                "inputs": input_hashes,
                "seeds": runtime.SEEDS,
                "rates": runtime.LEARNING_RATES,
                "roles": "synthetic_fixture_only",
            },
            sort_keys=True,
        ).encode()
    ).hexdigest()
    return {
        "experiment_id": "exp7755-training-runtime",
        "milestone": "2026.09.675",
        "run_date": "20260927",
        "honest_verdict": "complete_disqualified_required_validation",
        "verdict_class": "disqualified",
        "flagged_adversarial": False,
        "gate_check_summary": [],
        "rows": fixture["decision_rows"],
        "fixture_training_rows": fixture["trial_rows"],
        "acceptance_gate_results": {
            "validity": None,
            "readiness": 0,
            "probability_quality": None,
            "decision_benefit": None,
            "retention": None,
            "efficiency": None,
        },
        "duration_s": time.monotonic() - started,
        "phase_spans": spans,
        "random_seed": list(runtime.SEEDS),
        "reproducibility_checksum": "sha256:" + checksum,
        "sample_size_budget": {
            "intended": 4,
            "eligible": 4,
            "started": 4,
            "completed": 4,
            "excluded": 0,
            "censored": 0,
            "effective_independent_n": 4,
        },
        "source_artifact_hashes": sources,
        "preconditions_checked": {
            "fixture_gate": "open_independent_of_natural_corpus",
            "natural_input_observations": observations,
            "backend": "jax_cpu",
            "cpu_count": __import__("os").cpu_count(),
            "scope_path": str(SCOPE),
            "scope_sha256": sha256_file(SCOPE),
        },
        "validation_receipts": {"frozen_affected_scope": scope, "commands": receipts},
        "verifier_is_oracle": True,
        "claim_scope": "synthetic_fixture_execution_only; natural_performance_unmeasured",
        "field_principles": PRINCIPLES,
        "inference_substrate": "verifier_ensemble_against_cached_candidates",
        "inference_substrate_class": "no_model_load",
        "planned_inference_substrate_class": "no_model_load",
        "MODEL_SPECS": [],
        "model_specs": [],
        "model_invocation_counts": {
            "model_loads": 0,
            "generation_calls": 0,
            "tokens": 0,
            "loaded_files": [],
        },
        "training_runtime_ready_score": 0,
        "training_protocol_path": str(protocol_path.relative_to(ROOT)),
    }


def _independent_reduce(raw: Path) -> dict[str, Any]:
    """Use plain JSON and arithmetic to check persisted risk and action rows."""
    fixtures = [json.loads(line) for line in (raw / "fixtures.jsonl").read_text().splitlines()]
    trials = [json.loads(line) for line in (raw / "trials.jsonl").read_text().splitlines()]
    decisions = [json.loads(line) for line in (raw / "decisions.jsonl").read_text().splitlines()]
    family_ids = {row["id"] for row in fixtures}
    for row in decisions:
        p = row["probability_unsupported"]
        costs = {"accept": 5 * p, "reject": 1 - p, "escalate": 0.25}
        expected = min(("escalate", "accept", "reject"), key=lambda key: costs[key])
        if row["family"] not in family_ids or row["action"] != expected or not 0 <= p <= 1:
            raise ValueError("independent row mismatch")
    if len(decisions) != 4 * len({(row["arm"], row["mode"]) for row in trials}):
        raise ValueError("independent denominator mismatch")
    return {
        "trial_count": len(trials),
        "decision_count": len(decisions),
        "effective_independent_n": len(family_ids),
    }


def run(date: str) -> dict[str, Any]:
    """Fit fixtures, run every frozen check and publish one terminal record."""
    started = time.monotonic()
    spans: list[dict[str, Any]] = []
    phase = time.monotonic()
    experiment.progress(started, "preconditions", "start", 0)
    if date != "20260927":
        raise ValueError("wrong run date")
    scope = json.loads(SCOPE.read_text())
    RAW.mkdir(parents=True, exist_ok=True)
    sources, observations = _source_checks()
    protocol_path = RAW / "training_protocol.json"
    atomic_json(
        protocol_path,
        {
            "architecture": {
                "energy": "two_state_per_sentence_with_null",
                "logistic": "prior_pooled_per_sentence_shared_head",
                "mlp": "prior_pooled_per_sentence_width_16_shared_head",
            },
            "losses": {
                "primary": "response_NLL_of_mean_raw_view_risk",
                "local": "mean_known_sentence_NLL",
                "response_set_local_coefficient": 0,
                "other_local_coefficient": 1,
            },
            "two_view": {
                "ordinary": "mean_raw_risk",
                "constrained": "mean_raw_risk_plus_duals",
                "J": "symmetric_bernoulli_KL_divided_by_two",
                "J_max": 0.01,
                "second_view_CE_max": 0.70,
                "dual_eta": 0.01,
                "dual_range": [0, 10],
            },
            "seeds": runtime.SEEDS,
            "learning_rates": runtime.LEARNING_RATES,
            "epochs_max": runtime.EPOCHS_MAX,
            "fixture_epochs": 12,
            "parameter_max_including_duals": runtime.PARAMETER_MAX,
            "temperature_grid": runtime.temperature_grid(),
            "selection": "uncalibrated_deployed_response_NLL",
            "risk_order": "view_risk_then_average_then_one_response_temperature_then_one_decision",
        },
    )
    spans.append(_span("preconditions", phase, len(sources)))
    experiment.progress(started, "preconditions", "complete", len(sources))

    phase = time.monotonic()
    fixture = experiment.run_fixtures(RAW, extra_modes=("ordinary", "constrained"))
    spans.append(_span("fixture_fit", phase, fixture["trial_count"]))
    numerical = all(
        row["final_hash"] != row["initial_hash"]
        and row["final_loss"] < row["initial_loss"]
        and row["gradient_error"] <= 1e-6
        and row["parameter_count"] <= runtime.PARAMETER_MAX
        for row in fixture["trial_rows"]
    )
    receipts: list[dict[str, Any]] = []
    e2e_log = RAW / "validation_logs" / "real_entrypoint_e2e.log"
    e2e_log.parent.mkdir(parents=True, exist_ok=True)
    e2e_log.write_text(
        json.dumps(
            {
                "trials": fixture["trial_count"],
                "decisions": fixture["decision_count"],
                "numerical": numerical,
            }
        )
    )
    receipts.append(
        {
            "name": "real_entrypoint_e2e",
            "command_argv": sys.argv,
            "exit_code": 0 if numerical else 1,
            "passed": numerical,
            "log_path": str(e2e_log.relative_to(ROOT)),
            "log_sha256": sha256_file(e2e_log),
        }
    )

    phase = time.monotonic()
    basetemp = RAW / "private_pytest"
    basetemp.mkdir(parents=True, exist_ok=True)
    commands = build_scoped_commands(
        ROOT,
        scope["tests"],
        scope["changed_modules"],
        static_paths=scope["static_paths"],
        basetemp=basetemp,
        coverage_file=RAW / ".coverage",
    )
    experiment.progress(started, "validation", "before_scoped_subprocesses", 0)
    receipts.extend(run_commands(ROOT, commands, log_dir=RAW / "validation_logs", heartbeat_s=30.0))
    experiment.progress(started, "validation", "after_scoped_subprocesses", len(commands))
    full = (
        str(ROOT / ".venv/bin/pytest"),
        "tests/python",
        "-q",
        "-n",
        "0",
        "-o",
        "addopts=",
        "--no-cov",
        f"--basetemp={basetemp / 'full'}",
    )
    receipts.append(_run("full_python_suite", full, 1800.0))
    spans.append(_span("validation", phase, len(receipts)))

    phase = time.monotonic()
    script = str(ROOT / "scripts/experiments/experiment_7755_v675_training_runtime.py")
    python = str(ROOT / ".venv/bin/python")
    receipts.append(_run("cold_replay", (python, "-u", script, "--cold-reduce", str(RAW))))
    receipts.append(
        _run("independent_reduction", (python, "-u", script, "--independent-reduce", str(RAW)))
    )
    spans.append(_span("cold_replay", phase, 2))

    candidate = _candidate(fixture, sources, observations, protocol_path, started, spans, receipts)
    failures = [row for row in receipts if not row["passed"]]
    if not failures:
        candidate["honest_verdict"] = "complete_circular_positive_fixture_training"
        candidate["verdict_class"] = "circular_positive"
        candidate["training_runtime_ready_score"] = 1
        candidate["acceptance_gate_results"]["validity"] = 1
        candidate["acceptance_gate_results"]["readiness"] = 1
    candidate_path = RAW / "terminal_candidate.json"
    atomic_json(candidate_path, candidate)
    phase = time.monotonic()
    receipts.append(
        _run(
            "adversarial_verify",
            (python, "-u", "scripts/adversarial_verify.py", "--json", str(candidate_path)),
        )
    )
    receipts.append(
        _run(
            "strict_row_consistency",
            (
                python,
                "-u",
                "scripts/verdict_row_consistency_lint.py",
                "--strict",
                str(candidate_path),
            ),
        )
    )
    spans.append(_span("terminal_readers", phase, 2))
    report = json.loads((RAW / "validation_logs" / "00_adversarial_verify.log").read_text())
    flagged = report["flagged_count"] > 0
    failures = [row for row in receipts if not row["passed"]]
    candidate["flagged_adversarial"] = flagged
    candidate["duration_s"] = time.monotonic() - started
    candidate["phase_spans"] = spans
    candidate["validation_receipts"]["commands"] = receipts
    candidate["validation_receipts"]["candidate_sha256"] = sha256_file(candidate_path)
    candidate["validation_receipts"]["coverage"] = next(
        (
            row.get("output_tail")
            for row in receipts
            if row["name"] == "changed_module_coverage_report"
        ),
        None,
    )
    candidate["gate_check_summary"] = [
        {
            "upstream_id": NAME,
            "artifact_path": row.get("log_path"),
            "artifact_hash": row.get("log_sha256"),
            "field": row["name"] + ".exit_code",
            "expected": 0,
            "observed": row["exit_code"],
            "operator": "==",
        }
        for row in failures
    ]
    if failures or flagged:
        candidate["honest_verdict"] = "complete_disqualified_required_validation"
        candidate["verdict_class"] = "disqualified"
        candidate["training_runtime_ready_score"] = 0
        candidate["acceptance_gate_results"]["validity"] = 0
        candidate["acceptance_gate_results"]["readiness"] = 0
    atomic_json(OUTPUT, candidate)
    experiment.progress(started, "publication", "complete", fixture["trial_count"])
    return candidate


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--date")
    parser.add_argument("--cold-reduce", type=Path)
    parser.add_argument("--independent-reduce", type=Path)
    args = parser.parse_args()
    if args.cold_reduce:
        print(json.dumps(experiment.cold_reduce(args.cold_reduce), sort_keys=True), flush=True)
        return 0
    if args.independent_reduce:
        print(json.dumps(_independent_reduce(args.independent_reduce), sort_keys=True), flush=True)
        return 0
    run(args.date)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
