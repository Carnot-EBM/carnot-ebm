"""Fixture qualification for REQ-VERIFY-7769; no natural labels are read."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from typing import Any

import numpy as np

from carnot import experiment_7760_v675_online_runner as online
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash, sha256_file
from carnot.reporting.experiment_7303_validation_scope import (
    CommandSpec,
    build_scoped_commands,
    reduce_required_checks,
    run_commands,
)
from carnot.verify import training_qualification as qualification
from carnot.verify import training_runtime as runtime

ROOT = Path(__file__).resolve().parents[2]
SCOPE = ROOT / "results/raw/experiment_7769_v676_training_qualification/frozen_scope.json"
SOURCE_PATHS = (
    "results/experiment_7755_v675_training_runtime.json",
    "results/experiment_7760_v675_online_runner.json",
    "results/experiment_7768_v676_source_view_qualification.json",
    online.SOURCE,
    online.MANIFEST,
)
PRINCIPLES = {
    "experiment_id": "An artifact must have a unique current owner.",
    "honest_verdict": "A terminal record must not waste attempts on unchanged inputs.",
    "verdict_class": "The claim class travels with the evidence.",
    "flagged_adversarial": "Invalid evidence must not open downstream gates.",
    "gate_check_summary": "Missing producers and failed scientific thresholds are different causes.",
    "rows": "Aggregates must be recomputable without rerunning science.",
    "acceptance_gate_results": "A working protocol is not evidence of benefit.",
    "duration_s": "Duration must describe actual work without padding.",
    "phase_spans": "Measured phases include validation and cold replay.",
    "random_seed": "A third party needs the same experiment inputs.",
    "reproducibility_checksum": "A third party needs the same experiment inputs.",
    "reproducibility_inputs": "Exact code, source, fixture, protocol and parameter bytes identify a replay.",
    "sample_size_budget": "Repeated views and seeds do not increase independent family count.",
    "source_artifact_hashes": "A missing producer cannot be replaced with a convenient old result.",
    "preconditions_checked": "Access and validity must be established before expensive work.",
    "validation_receipts": "All registered checks must pass before readiness opens.",
    "verifier_is_oracle": "Execution truth and independent semantic verification are distinct claims.",
    "claim_scope": "Fixture execution does not measure natural benefit.",
    "inference_substrate_class": "Duration floors must match the invoked substrate.",
    "MODEL_SPECS": "A cited upstream model is not a current model invocation.",
    "model_specs": "An empty list records zero current model specifications.",
    "model_invocation_counts": "Actual calls, tokens and loaded files must be counted.",
    "planned_inference_substrate_class": "The planned and actual substrates must agree.",
    "training_runtime_ready_score": "A fixture learner must execute before a natural comparison.",
    "training_protocol_path": "Outcomes cannot select the training recipe.",
    "fixture_training_rows": "A no-op or numerically invalid optimizer must fail.",
    "online_runtime_ready_score": "Historical readiness cannot bypass a failed original requirement.",
    "online_protocol_path": "Natural learning needs one current compatible runner.",
    "online_fixture": "Current queue and restart execution must precede readiness.",
    "historical_spec_mismatch": "A new fixture pass cannot revise a failed original requirement.",
    "raw_paths": "A fresh process must be able to reopen every raw table.",
}


def progress(start: float, phase: str, event: str, units: int) -> None:
    """Flush measured progress at each phase and completed fixture unit."""
    print(
        f"[exp7769] phase={phase} event={event} elapsed_s={time.monotonic() - start:.3f} completed_units={units}",
        flush=True,
    )


def qualify_online(folder: Path) -> dict[str, Any]:
    """Rerun the V675 queue under the current V676 fixture contract."""
    start = time.monotonic()
    checks, _, names = online.preflight(ROOT)
    if not all(row["passed"] for row in checks):
        raise ValueError("online bank preflight failed")
    folder.mkdir(parents=True, exist_ok=True)
    progress(start, "online", "before_fixture", 0)
    raw = online.run_fixture(folder / "fixture", names)
    reduced = online.cold_reduce(Path(raw["raw_path"]))
    progress(start, "online", "after_fixture", reduced["row_count"])
    arms = raw["arms"]
    adaptive = arms["adaptive"]
    static = arms["complete_static"]
    shuffled = arms["shuffled"]
    stage = online.OnlineRunner(
        folder / "restart.json", folder / "restart_bank.json", names, "adaptive"
    )
    for block in range(4):
        if block:
            stage.release_block(block - 1, block)
        stage.predict_block(block)
        progress(start, "restart", "saved_block", block + 1)
    argv = [
        sys.executable,
        "-m",
        "carnot.experiment_7760_v675_online_runner",
        "--resume-private",
        str(stage.state_path),
        str(stage.bank_path),
    ]
    progress(start, "restart", "before_subprocess", 4)
    child = subprocess.run(argv, capture_output=True, text=True, check=False, timeout=90)
    progress(start, "restart", "after_subprocess", 5)
    log = folder / "restart.log"
    log.write_text(child.stdout + child.stderr)
    resumed = json.loads(child.stdout.splitlines()[-1]) if child.returncode == 0 else {}
    parity = all(
        resumed.get(key) == adaptive.get(key)
        for key in (
            "decision_hash",
            "bank_hash",
            "queue_hash",
            "credits",
            "rng_counter",
            "model_hash",
        )
    )
    progress(start, "hard_exit", "before_subprocess", 5)
    hard_exit = online.hard_exit_commit_check(folder / "hard_exit", names)
    progress(start, "hard_exit", "after_subprocess", 6)
    result = {
        "valid": bool(reduced["valid"] and child.returncode == 0 and parity and hard_exit),
        "prediction_before_feedback": all(
            row["label_arrival_block"] == row["block"] + 1 for row in adaptive["rows"]
        ),
        "rejected_admissions": any(
            row["event"] == "admission" and not row["accepted"] for row in adaptive["lifecycle"]
        ),
        "applied_updates": adaptive["update_calls"] == 96,
        "shuffled_delayed_labels": all(
            row["label_arrival_block"] == row["block"] + 1 for row in shuffled["rows"]
        ),
        "complete_static_initial_coefficients": len(static["initial_active"]),
        "hard_exit": hard_exit,
        "cold_restart_parity": parity,
        "raw_path": str(raw["raw_path"]),
        "raw_hash": reduced["raw_sha256"],
        "restart_command_argv": argv,
        "restart_exit_code": child.returncode,
        "restart_log_hash": sha256_file(log),
        "hard_exit_receipt_path": str(folder / "hard_exit" / "receipt.json"),
    }
    result["valid"] = bool(
        result["valid"]
        and all(
            result[key]
            for key in (
                "prediction_before_feedback",
                "rejected_admissions",
                "applied_updates",
                "shuffled_delayed_labels",
            )
        )
        and result["complete_static_initial_coefficients"] == 16
    )
    return result


def run_fixture(folder: Path, date: str) -> dict[str, Any]:
    """Train every registered fixture arm and retain exact raw decisions."""
    start = time.monotonic()
    folder.mkdir(parents=True, exist_ok=True)
    scope = json.loads(SCOPE.read_text())
    source_hashes = []
    for relative in SOURCE_PATHS:
        path = ROOT / relative
        value = json.loads(path.read_text()) if path.is_file() else {}
        source_hashes.append(
            {
                "upstream_id": Path(relative).stem,
                "artifact_path": relative,
                "artifact_hash": sha256_file(path) if path.is_file() else None,
                "run_date": value.get("run_date"),
                "imported_fields": {
                    key: value.get(key)
                    for key in (
                        "honest_verdict",
                        "verdict_class",
                        "flagged_adversarial",
                        "training_runtime_ready_score",
                        "online_runtime_ready_score",
                        "evidence_view_ready_score",
                    )
                },
                "eligibility": path.is_file()
                and value.get("verdict_class") != "disqualified"
                and (
                    relative != SOURCE_PATHS[1]
                    or (
                        bool(value.get("validation_receipts", {}).get("full_python_suite"))
                        and all(
                            row.get("passed")
                            for row in value.get("validation_receipts", {}).get(
                                "full_python_suite", []
                            )
                        )
                    )
                ),
            }
        )
    checks, _, names = online.preflight(ROOT)
    failed = [row for row in checks if not row["passed"]]
    if failed:
        raise ValueError(f"required online source failed: {failed}")
    protocol = {
        "arms": list(qualification.ARMS),
        "architecture": "normalized_binary_sentence_energy",
        "seed": runtime.SEEDS[0],
        "learning_rate": runtime.LEARNING_RATES[1],
        "epochs": 2,
        "optimizer": "full_batch_gradient_descent",
        "temperature_grid": runtime.temperature_grid(),
        "loss": "response_NLL_plus_mean_known_sentence_NLL",
        "constraint": {
            "symmetric_kl_half_tolerance": 0.01,
            "alternate_ce_tolerance": 0.70,
            "dual_step": 0.01,
            "dual_clip": [0, 10],
        },
        "probability_clip": [1e-6, 1 - 1e-6],
        "parameter_max": 4096,
        "complete_static_initial_coefficients": {name: 0.0 for name in names},
    }
    online_protocol = {
        "runner": online.MODULE,
        "bank": online.BANK_MODULE,
        "queue_capacity": 12,
        "blocks": 8,
        "labels_delayed_blocks": 1,
        "complete_static_predicates": names,
    }
    atomic_json(folder / "training_protocol.json", protocol)
    atomic_json(folder / "online_protocol.json", online_protocol)
    records = qualification.fixture_records() + [
        {"id": "empty", "source": "", "answer": "", "label": None, "known": []},
        {"id": "over_budget", "source": "A.", "answer": "A. " * 17, "label": None, "known": []},
    ]
    atomic_json(folder / "fixtures.json", records)
    training_rows = []
    rows = []
    spans = []
    phase = time.monotonic()
    progress(start, "numerical", "start", 0)
    for arm in qualification.ARMS:
        batch, excluded = qualification.make_batch(records, arm, names)
        progress(start, "numerical", f"before_benchmark_{arm}", len(training_rows))
        head = qualification.fit_arm(
            arm, batch, static_names=names if arm == "complete_static_constrained_set" else None
        )
        progress(start, "numerical", f"after_benchmark_{arm}", len(training_rows) + 1)
        head_path = folder / f"{arm}.json"
        progress(start, "numerical", f"before_model_save_{arm}", len(training_rows))
        runtime.save(head_path, head)
        progress(start, "numerical", f"after_model_save_{arm}", len(training_rows))
        progress(start, "numerical", f"before_model_load_{arm}", len(training_rows))
        loaded = runtime.load(head_path)
        progress(start, "numerical", f"after_model_load_{arm}", len(training_rows))
        adapter = qualification.NaturalHeadAdapter(loaded)
        training_rows.append(
            {
                "arm": arm,
                "head_arm": head["head_arm"],
                "mode": head["mode"],
                "seed": head["seed"],
                "initial_hash": head["initial_hash"],
                "final_hash": head["final_hash"],
                "initial_loss": head["curve"][0]["loss"],
                "final_loss": head["curve"][-1]["loss"],
                "gradient_error": head["gradient_error"],
                "gradient_norm": head["curve"][-1]["gradient_norm"],
                "parameter_count": head["parameter_count"],
                "dual_variables": head["duals"],
                "temperature": head["temperature"],
                "static_predicates": head["static_predicates"],
                "static_coefficients": head["static_coefficients"],
                "head_path": str(head_path),
                "head_hash": sha256_file(head_path),
                "label_scope": "synthetic_fixture",
            }
        )
        eligible = [
            record for record in records if record["id"] not in {item["id"] for item in excluded}
        ]
        for index, record in enumerate(eligible):
            rows.append(
                {
                    "family": record["id"],
                    "arm": arm,
                    "seed": head["seed"],
                    "label": record["label"],
                    "known": record["known"],
                    "raw_path": str(folder / "fixtures.json"),
                    "excluded": False,
                    "censored": False,
                    **adapter.decide(batch, index),
                }
            )
        for item in excluded:
            rows.append(
                {
                    "family": item["id"],
                    "arm": arm,
                    "seed": head["seed"],
                    "label": None,
                    "known": [],
                    "raw_path": str(folder / "fixtures.json"),
                    "excluded": True,
                    "censored": False,
                    "reason": item["reason"],
                    "raw_risk": None,
                    "probability_unsupported": 0.5,
                    "action": "escalate",
                }
            )
        training_rows[-1]["policy_outcomes"] = [
            {
                "family": row["family"],
                "action": row["action"],
                "probability_unsupported": row["probability_unsupported"],
                "excluded": row["excluded"],
            }
            for row in rows
            if row["arm"] == arm
        ]
        progress(start, "numerical", f"completed_{arm}", len(training_rows))
    spans.append(
        {
            "phase": "numerical",
            "duration_s": time.monotonic() - phase,
            "completed_units": len(training_rows),
        }
    )
    phase = time.monotonic()
    progress(start, "online", "start", 0)
    online_result = qualify_online(folder / "online")
    spans.append(
        {"phase": "online", "duration_s": time.monotonic() - phase, "completed_units": 640}
    )
    atomic_json(folder / "fixture_rows.json", rows)
    atomic_json(folder / "training_rows.json", training_rows)
    passed = online_result["valid"] and all(
        row["initial_hash"] != row["final_hash"]
        and row["gradient_error"] < 1e-4
        and row["parameter_count"] <= 4096
        for row in training_rows
    )
    input_hashes = {row["artifact_path"]: row["artifact_hash"] for row in source_hashes}
    input_hashes.update(scope["frozen_hashes"])
    input_hashes.update(
        {path: sha256_file(ROOT / path) for path in (*scope["changed_modules"], *scope["cli"])}
    )
    input_hashes[str(SCOPE.relative_to(ROOT))] = sha256_file(SCOPE)
    input_hashes.update({row["head_path"]: row["head_hash"] for row in training_rows})
    input_hashes.update(
        {
            str(folder / name): sha256_file(folder / name)
            for name in ("training_protocol.json", "online_protocol.json", "fixtures.json")
        }
    )
    prior_online = json.loads((ROOT / SOURCE_PATHS[1]).read_text())
    prior_full_suite = prior_online.get("validation_receipts", {}).get("full_python_suite", [])
    artifact = {
        "experiment_id": 7769,
        "milestone": "2026.09.676",
        "run_date": date,
        "honest_verdict": "complete_circular_positive_training_fixture"
        if passed
        else "complete_disqualified_fixture",
        "verdict_class": "circular_positive" if passed else "disqualified",
        "flagged_adversarial": False,
        "gate_check_summary": [],
        "rows": rows,
        "acceptance_gate_results": {
            "validity": passed,
            "readiness": 0,
            "probability_quality": None,
            "decision_benefit": None,
            "retention": None,
            "efficiency": None,
        },
        "duration_s": time.monotonic() - start,
        "phase_spans": spans,
        "random_seed": runtime.SEEDS[0],
        "reproducibility_checksum": canonical_hash(
            {"inputs": input_hashes, "protocol": protocol, "roles": "synthetic_fixture_only"}
        ),
        "reproducibility_inputs": input_hashes,
        "sample_size_budget": {
            "intended": 6,
            "eligible": 4,
            "started": 4,
            "completed": 4,
            "excluded": 2,
            "censored": 0,
            "effective_independent_n": 4,
        },
        "source_artifact_hashes": source_hashes,
        "preconditions_checked": {
            "online_bank_checks": checks,
            "source_view_v676_disqualified": source_hashes[2]["imported_fields"]["verdict_class"]
            == "disqualified",
            "natural_fitting_authorized": False,
            "backend": os.environ.get("JAX_PLATFORMS", "default"),
            "free_bytes": os.statvfs(ROOT).f_bavail * os.statvfs(ROOT).f_frsize,
            "scope_path": str(SCOPE),
            "scope_hash": sha256_file(SCOPE),
        },
        "validation_receipts": {
            "frozen_affected_scope": scope,
            "commands": [],
            "real_entrypoint_e2e": None,
            "cold_replay": None,
            "terminal_readers": None,
        },
        "verifier_is_oracle": True,
        "claim_scope": "synthetic_fixture_only_no_natural_benefit",
        "field_principles": PRINCIPLES,
        "inference_substrate": "verifier_ensemble_against_cached_candidates",
        "inference_substrate_class": "no_model_load",
        "planned_inference_substrate_class": "no_model_load",
        "MODEL_SPECS": [],
        "model_specs": [],
        "model_invocation_counts": {"calls": 0, "tokens": 0, "loaded_files": []},
        "training_runtime_ready_score": 0,
        "training_protocol_path": str(folder / "training_protocol.json"),
        "fixture_training_rows": training_rows,
        "online_runtime_ready_score": 0,
        "online_protocol_path": str(folder / "online_protocol.json"),
        "online_fixture": online_result,
        "historical_spec_mismatch": {
            "exp7755_verdict": source_hashes[0]["imported_fields"]["honest_verdict"],
            "exp7760_reported_ready": source_hashes[1]["imported_fields"][
                "online_runtime_ready_score"
            ],
            "exp7760_full_suite_exit": prior_full_suite[0]["exit_code"]
            if prior_full_suite
            else None,
            "exp7760_full_suite_log_hash": prior_full_suite[0]["log_sha256"]
            if prior_full_suite
            else None,
        },
        "raw_paths": {
            "fixtures": str(folder / "fixtures.json"),
            "rows": str(folder / "fixture_rows.json"),
            "training": str(folder / "training_rows.json"),
        },
    }
    return artifact


def cold_reduce(candidate: Path) -> dict[str, Any]:
    """Reopen immutable raw rows and saved heads in a fresh process."""
    start = time.monotonic()
    progress(start, "cold_replay", "start", 0)
    artifact = json.loads(candidate.read_text())
    raw = artifact["raw_paths"]
    records = json.loads(Path(raw["fixtures"]).read_text())
    rows = json.loads(Path(raw["rows"]).read_text())
    training = json.loads(Path(raw["training"]).read_text())
    if rows != artifact["rows"] or training != artifact["fixture_training_rows"]:
        raise ValueError("candidate_raw_rows_invalid")
    names = json.loads(Path(artifact["online_protocol_path"]).read_text())[
        "complete_static_predicates"
    ]
    checked = 0
    for fitted in training:
        arm = fitted["arm"]
        progress(start, "cold_replay", f"before_model_load_{arm}", checked)
        head_path = Path(fitted["head_path"])
        if sha256_file(head_path) != fitted["head_hash"]:
            raise ValueError("head_hash_invalid")
        head = runtime.load(head_path)
        progress(start, "cold_replay", f"after_model_load_{arm}", checked)
        batch, excluded = qualification.make_batch(records, arm, names)
        wanted = [row for row in rows if row["arm"] == arm]
        if len(wanted) != len(records) or len(excluded) != 2:
            raise ValueError("unit_count_invalid")
        adapter = qualification.NaturalHeadAdapter(head)
        eligible = [row for row in wanted if not row["excluded"]]
        for index, row in enumerate(eligible):
            expected = adapter.decide(batch, index)
            if any(row[key] != expected[key] for key in expected):
                raise ValueError("cold_decision_invalid")
            checked += 1
        progress(start, "cold_replay", f"verified_{arm}", checked)
    online_result = online.cold_reduce(Path(artifact["online_fixture"]["raw_path"]))
    if (
        not online_result["valid"]
        or online_result["raw_sha256"] != artifact["online_fixture"]["raw_hash"]
    ):
        raise ValueError("online_raw_invalid")
    return {
        "valid": True,
        "decision_count": checked,
        "arm_count": len(training),
        "online_row_count": online_result["row_count"],
        "raw_row_hash": sha256_file(Path(raw["rows"])),
    }


def blocked_artifact(date: str, checks: list[dict[str, Any]], started: float) -> dict[str, Any]:
    """Retain every planned arm when an external bank input fails preflight."""
    failed = [
        {
            "upstream_id": row.get("upstream_id"),
            "artifact_path": row.get("artifact_path"),
            "artifact_hash": row.get("artifact_sha256"),
            "field": row.get("field"),
            "expected": row.get("expected"),
            "observed": row.get("observed"),
            "operator": row.get("operator", "=="),
        }
        for row in checks
        if not row["passed"]
    ]
    sources = []
    for relative in SOURCE_PATHS:
        path = ROOT / relative
        sources.append(
            {
                "upstream_id": Path(relative).stem,
                "artifact_path": relative,
                "artifact_hash": sha256_file(path) if path.is_file() else None,
                "run_date": None,
                "imported_fields": {},
                "eligibility": False,
            }
        )
    rows = [
        {
            "family": f"fixture-{index}",
            "arm": arm,
            "label": None,
            "excluded": True,
            "censored": False,
            "status": "unstarted_external_precondition",
            "raw_path": None,
        }
        for arm in qualification.ARMS
        for index in range(6)
    ]
    return {
        "experiment_id": 7769,
        "milestone": "2026.09.676",
        "run_date": date,
        "honest_verdict": "complete_blocked_external_precondition",
        "verdict_class": "blocked",
        "flagged_adversarial": False,
        "gate_check_summary": failed,
        "rows": rows,
        "acceptance_gate_results": {
            "validity": False,
            "readiness": 0,
            "probability_quality": None,
            "decision_benefit": None,
            "retention": None,
            "efficiency": None,
        },
        "duration_s": time.monotonic() - started,
        "phase_spans": [
            {
                "phase": "preflight",
                "duration_s": time.monotonic() - started,
                "completed_units": len(checks),
            }
        ],
        "random_seed": runtime.SEEDS[0],
        "reproducibility_checksum": canonical_hash(
            {"checks": checks, "scope_hash": sha256_file(SCOPE)}
        ),
        "sample_size_budget": {
            "intended": 6,
            "eligible": 0,
            "started": 0,
            "completed": 0,
            "excluded": 6,
            "censored": 0,
            "effective_independent_n": 0,
        },
        "source_artifact_hashes": sources,
        "preconditions_checked": {
            "online_bank_checks": checks,
            "natural_fitting_authorized": False,
        },
        "validation_receipts": {
            "frozen_affected_scope": json.loads(SCOPE.read_text()),
            "commands": [],
            "real_entrypoint_e2e": None,
            "cold_replay": None,
            "terminal_readers": None,
        },
        "verifier_is_oracle": True,
        "claim_scope": "unstarted_fixture_external_block",
        "field_principles": PRINCIPLES,
        "inference_substrate": "verifier_ensemble_against_cached_candidates",
        "inference_substrate_class": "no_model_load",
        "planned_inference_substrate_class": "no_model_load",
        "MODEL_SPECS": [],
        "model_specs": [],
        "model_invocation_counts": {"calls": 0, "tokens": 0, "loaded_files": []},
        "training_runtime_ready_score": 0,
        "training_protocol_path": None,
        "fixture_training_rows": [
            {"arm": arm, "status": "unstarted_external_precondition"} for arm in qualification.ARMS
        ],
        "online_runtime_ready_score": 0,
        "online_protocol_path": None,
    }


def launch(date: str) -> dict[str, Any]:
    """Run the real fixture, affected validation and terminal artifact readers."""
    start = time.monotonic()
    raw = ROOT / "results/raw/experiment_7769_v676_training_qualification"
    raw.mkdir(parents=True, exist_ok=True)
    preflight_checks, _, _ = online.preflight(ROOT)
    if any(not row["passed"] for row in preflight_checks):
        artifact = blocked_artifact(date, preflight_checks, start)
        candidate = raw / "terminal_candidate.json"
        atomic_json(candidate, artifact)
        terminal_start = time.monotonic()
        readers = run_commands(
            ROOT,
            [
                CommandSpec(
                    "adversarial_verify",
                    (
                        sys.executable,
                        "-u",
                        "scripts/adversarial_verify.py",
                        "--json",
                        str(candidate),
                    ),
                    "blocked_candidate",
                    60,
                ),
                CommandSpec(
                    "strict_row_consistency",
                    (
                        sys.executable,
                        "-u",
                        "scripts/verdict_row_consistency_lint.py",
                        "--strict",
                        str(candidate),
                    ),
                    "blocked_candidate",
                    60,
                ),
            ],
            log_dir=raw / "validation_logs" / "terminal",
            heartbeat_s=30,
        )
        artifact["validation_receipts"]["terminal_readers"] = readers
        artifact["phase_spans"].append(
            {
                "phase": "terminal_readers",
                "duration_s": time.monotonic() - terminal_start,
                "completed_units": len(readers),
            }
        )
        report = (
            json.loads((ROOT / readers[0]["log_path"]).read_text()) if readers[0]["passed"] else {}
        )
        artifact["flagged_adversarial"] = bool(report.get("flagged_count", 1))
        for row in readers:
            if not row["passed"]:
                artifact["gate_check_summary"].append(
                    {
                        "upstream_id": "exp7769",
                        "artifact_path": row["log_path"],
                        "artifact_hash": row["log_sha256"],
                        "field": f"{row['name']}.exit_code",
                        "expected": 0,
                        "observed": row["exit_code"],
                        "operator": "==",
                    }
                )
        artifact["duration_s"] = time.monotonic() - start
        atomic_json(ROOT / "results/experiment_7769_v676_training_qualification.json", artifact)
        progress(start, "publish", "blocked_atomic_write", len(artifact["rows"]))
        return artifact
    artifact = run_fixture(raw / "fixture", date)
    phase = time.monotonic()
    candidate = raw / "terminal_candidate.json"
    atomic_json(candidate, artifact)
    logs = raw / "validation_logs"
    commands = [
        CommandSpec(
            "cold_replay",
            (
                sys.executable,
                "-u",
                "-m",
                "carnot.experiment_7769_v676_training_qualification",
                "--cold-reduce",
                str(candidate),
            ),
            "exact_candidate",
            120,
        )
    ]
    progress(start, "cold_replay", "before_subprocess", 0)
    cold = run_commands(ROOT, commands, log_dir=logs, heartbeat_s=30)[0]
    progress(start, "cold_replay", "after_subprocess", 1)
    artifact["validation_receipts"]["cold_replay"] = cold
    artifact["phase_spans"].append(
        {
            "phase": "cold_replay",
            "duration_s": time.monotonic() - phase,
            "completed_units": 36 if cold["passed"] else 0,
        }
    )
    phase = time.monotonic()
    with tempfile.TemporaryDirectory(prefix="exp7769-validation-") as private:
        private_path = Path(private)
        basetemp = private_path / "basetemp"
        for name in ("focused", "coverage", "full"):
            (basetemp / name).parent.mkdir(parents=True, exist_ok=True)
        scope = json.loads(SCOPE.read_text())
        tests = [*scope["direct_tests"], *scope["transitive_tests"]]
        scoped = build_scoped_commands(
            ROOT,
            tests,
            scope["changed_modules"],
            static_paths=scope["cli"],
            basetemp=basetemp,
            coverage_file=private_path / ".coverage",
        )
        scoped = [
            CommandSpec(
                item.name,
                item.argv,
                item.scope,
                1800 if item.name == "changed_module_coverage" else item.timeout_s,
            )
            for item in scoped
        ]
        progress(start, "affected_validation", "before_subprocesses", 0)
        receipts = run_commands(
            ROOT,
            scoped,
            log_dir=logs / "affected",
            extra_env={"COVERAGE_CORE": "sysmon"},
            heartbeat_s=30,
        )
        progress(start, "affected_validation", "after_subprocesses", len(receipts))
        artifact["validation_receipts"]["commands"] = receipts
        artifact["validation_receipts"]["affected_result"] = reduce_required_checks(receipts)
        full = run_commands(
            ROOT,
            [
                CommandSpec(
                    "full_python_suite",
                    (
                        str(ROOT / ".venv/bin/pytest"),
                        "tests/python",
                        "-q",
                        "-n",
                        "0",
                        "-o",
                        "addopts=",
                        "--no-cov",
                        f"--basetemp={basetemp / 'full'}",
                    ),
                    "broad_collection_diagnostic",
                    300,
                )
            ],
            log_dir=logs / "diagnostic",
            heartbeat_s=30,
        )[0]
        artifact["validation_receipts"]["broad_collection_diagnostic"] = full
        e2e = run_commands(
            ROOT,
            [
                CommandSpec(
                    "real_entrypoint_e2e",
                    (
                        sys.executable,
                        "-u",
                        "scripts/experiments/experiment_7769_v676_training_qualification.py",
                        "--private-e2e",
                        "--date",
                        date,
                    ),
                    "private_real_entrypoint",
                    180,
                )
            ],
            log_dir=logs / "e2e",
            heartbeat_s=30,
        )[0]
        artifact["validation_receipts"]["real_entrypoint_e2e"] = e2e
    failures = []
    for row in [cold, *receipts, e2e]:
        if not row["passed"]:
            failures.append(
                {
                    "upstream_id": "exp7769",
                    "artifact_path": row["log_path"],
                    "artifact_hash": row["log_sha256"],
                    "field": f"{row['name']}.exit_code",
                    "expected": 0,
                    "observed": row["exit_code"],
                    "operator": "==",
                }
            )
    artifact["gate_check_summary"] = failures
    if failures:
        artifact["honest_verdict"] = "complete_disqualified_required_validation"
        artifact["verdict_class"] = "disqualified"
        artifact["training_runtime_ready_score"] = 0
        artifact["online_runtime_ready_score"] = 0
        artifact["acceptance_gate_results"]["readiness"] = 0
        artifact["acceptance_gate_results"]["validity"] = False
    artifact["duration_s"] = time.monotonic() - start
    artifact["phase_spans"].append(
        {
            "phase": "validation",
            "duration_s": time.monotonic() - phase,
            "completed_units": len(receipts) + 3,
        }
    )
    atomic_json(candidate, artifact)
    terminal_start = time.monotonic()
    readers = run_commands(
        ROOT,
        [
            CommandSpec(
                "adversarial_verify",
                (sys.executable, "-u", "scripts/adversarial_verify.py", "--json", str(candidate)),
                "exact_candidate",
                60,
            ),
            CommandSpec(
                "strict_row_consistency",
                (
                    sys.executable,
                    "-u",
                    "scripts/verdict_row_consistency_lint.py",
                    "--strict",
                    str(candidate),
                ),
                "exact_candidate",
                60,
            ),
        ],
        log_dir=logs / "terminal",
        heartbeat_s=30,
    )
    artifact["validation_receipts"]["terminal_readers"] = readers
    artifact["phase_spans"].append(
        {
            "phase": "terminal_readers",
            "duration_s": time.monotonic() - terminal_start,
            "completed_units": len(readers),
        }
    )
    report = json.loads((ROOT / readers[0]["log_path"]).read_text()) if readers[0]["passed"] else {}
    artifact["flagged_adversarial"] = bool(report.get("flagged_count", 1))
    for row in readers:
        if not row["passed"]:
            artifact["gate_check_summary"].append(
                {
                    "upstream_id": "exp7769",
                    "artifact_path": row["log_path"],
                    "artifact_hash": row["log_sha256"],
                    "field": f"{row['name']}.exit_code",
                    "expected": 0,
                    "observed": row["exit_code"],
                    "operator": "==",
                }
            )
    if artifact["flagged_adversarial"] or artifact["gate_check_summary"]:
        artifact["honest_verdict"] = "complete_disqualified_required_validation"
        artifact["verdict_class"] = "disqualified"
        artifact["training_runtime_ready_score"] = 0
        artifact["online_runtime_ready_score"] = 0
        artifact["acceptance_gate_results"]["readiness"] = 0
        artifact["acceptance_gate_results"]["validity"] = False
    else:
        artifact["training_runtime_ready_score"] = 1
        artifact["online_runtime_ready_score"] = 1
        artifact["acceptance_gate_results"]["readiness"] = 1
    artifact["duration_s"] = time.monotonic() - start
    output = ROOT / "results/experiment_7769_v676_training_qualification.json"
    atomic_json(output, artifact)
    progress(start, "publish", "after_atomic_write", len(artifact["rows"]))
    return artifact


def main(argv: list[str] | None = None) -> int:
    """Expose the cold reducer for an owned independent child process."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--cold-reduce", type=Path, required=True)
    args = parser.parse_args(argv)
    print(json.dumps(cold_reduce(args.cold_reduce), sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":  # pragma: no cover - real child entrypoint exercises this.
    raise SystemExit(main())
