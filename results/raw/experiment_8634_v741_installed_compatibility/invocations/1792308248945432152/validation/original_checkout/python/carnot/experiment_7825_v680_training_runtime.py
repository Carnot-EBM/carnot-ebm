"""Qualify current fixture and online runtime evidence for REQ-VERIFY-7825."""

from __future__ import annotations

import json
from pathlib import Path
import time
from typing import Any

import numpy as np

from carnot import experiment_7811_v679_training_runtime as prior
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash, sha256_file

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "results/raw/experiment_7825_v680_training_runtime"
OUTPUT = ROOT / "results/experiment_7825_v680_training_runtime.json"
SCOPE = RAW / "frozen_validation_scope.json"
SEEDS = (68001, 68002, 68003)
ARMS = prior.ARMS
COMMAND_NAMES = [
    "coverage_shard_0",
    "coverage_shard_1",
    "coverage_combine",
    "coverage_report",
    "affected_pytest",
    "ruff_check",
    "ruff_format",
    "mypy",
    "spec_coverage",
    "task_e2e",
    "cold_replay",
    "adversarial_verify",
    "strict_row_consistency",
    "repository_health",
]
MODEL_SPECS: list[dict[str, Any]] = []

seal_log = prior.seal_log
read_sealed_log = prior.read_sealed_log
reduce_validation = prior.reduce_validation
cold_reduce = prior.cold_reduce
failed_operand = prior.failed_operand
historical_receipt_byte_audit = prior.historical_receipt_byte_audit
fit_one = prior.fit_one


def progress(start: float, phase: str, event: str, units: int) -> None:
    """Print a flushed task boundary with elapsed time and completed units."""
    print(
        f"[exp7825] phase={phase} event={event} elapsed_s={time.monotonic() - start:.3f} completed_units={units}",
        flush=True,
    )


def preflight(root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str]]:
    """Check producer bytes, old failure custody, current scope and executables."""
    checks, sources, names = prior.preflight(root)
    scope = root / "results/raw/experiment_7825_v680_training_runtime/frozen_validation_scope.json"
    old = root / "results/experiment_7811_v679_training_runtime.json"
    for path, upstream in ((scope, "7825_scope"), (old, "7811")):
        exists = path.is_file()
        digest = sha256_file(path) if exists else None
        checks.append(
            {
                "upstream_id": upstream,
                "artifact_path": str(path),
                "artifact_sha256": digest,
                "field": "exists",
                "operator": "==",
                "expected": True,
                "observed": exists,
                "passed": exists,
            }
        )
        sources.append(
            {
                "upstream_id": upstream,
                "artifact_path": str(path),
                "artifact_hash": digest,
                "run_date": None,
                "source_kind": "historical_context" if upstream == "7811" else "validation_scope",
                "eligibility": False,
            }
        )
    if scope.is_file():
        value = json.loads(scope.read_text())
        actual = [row["name"] for row in value.get("commands", [])]
        checks.append(
            {
                "upstream_id": "7825_scope",
                "artifact_path": str(scope),
                "artifact_sha256": sha256_file(scope),
                "field": "command_names",
                "operator": "==",
                "expected": COMMAND_NAMES,
                "observed": actual,
                "passed": actual == COMMAND_NAMES,
            }
        )
        for command in value.get("commands", []):
            executable = Path(command["argv"][0])
            exists = executable.is_file()
            checks.append(
                {
                    "upstream_id": "7825_scope",
                    "artifact_path": str(executable),
                    "artifact_sha256": sha256_file(executable) if exists else None,
                    "field": "exists",
                    "operator": "==",
                    "expected": True,
                    "observed": exists,
                    "passed": exists,
                }
            )
    if old.is_file():
        value = json.loads(old.read_text())
        actual = value.get("verdict_class")
        checks.append(
            {
                "upstream_id": "7811",
                "artifact_path": str(old),
                "artifact_sha256": sha256_file(old),
                "field": "verdict_class",
                "operator": "==",
                "expected": "disqualified",
                "observed": actual,
                "passed": actual == "disqualified",
            }
        )
        failures = value.get("gate_check_summary", [])
        observed = any(
            row.get("field") == "ruff_format.exit_code" and row.get("observed") == 1
            for row in failures
        )
        checks.append(
            {
                "upstream_id": "7811",
                "artifact_path": str(old),
                "artifact_sha256": sha256_file(old),
                "field": "ruff_format.exit_code",
                "operator": "==",
                "expected": True,
                "observed": observed,
                "passed": observed,
            }
        )
    return checks, sources, names


def base_record(
    date: str, checks: list[dict[str, Any]], sources: list[dict[str, Any]], duration: float
) -> dict[str, Any]:
    """Create a complete current owner record, including blocked unstarted rows."""
    record = prior._base_record(date, checks, sources, duration)
    record.update(experiment_id=7825, milestone="2026.09.680", random_seed=list(SEEDS))
    if record["verdict_class"] == "blocked":
        record["rows"] = [
            {
                "family": item["family"],
                "arm": arm,
                "seed": seed,
                "status": "unstarted_external_precondition",
                "excluded": True,
                "censored": False,
                "raw_path": None,
            }
            for item in prior.fixture_records()
            for arm in ARMS
            for seed in SEEDS
        ]
    record["claim_scope"] = (
        "six_public_fixture_families_only; no natural fitting or held-out benefit"
    )
    record["field_principles"] = {
        **record["field_principles"],
        "experiment_id": "One result has one task owner.",
        "training_runtime_ready_score": "Required checks and fixtures gate readiness.",
        "online_runtime_ready_score": "Bank lifecycle and required checks gate readiness.",
    }
    record["fixture_training_rows"] = []
    return record


def protocols(names: list[str]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Freeze current fixture seeds and an independent natural-fit recipe."""
    training, online = prior.protocols(names)
    training.update(schema="exp7825-training-protocol-v1", seeds=list(SEEDS))
    training["natural_recipe"]["epochs"] = 16
    training["natural_recipe"]["arms"] = list(ARMS)
    training["natural_recipe"]["fixture_weights_allowed"] = False
    online.update(schema="exp7825-online-protocol-v1", proposal_budget_per_block=1)
    return training, online


def measure(date: str, raw: Path) -> dict[str, Any]:
    """Fit all current fixture units and exercise the qualified online bank."""
    started = time.monotonic()
    progress(started, "preflight", "before", 0)
    checks, sources, names = preflight(ROOT)
    progress(started, "preflight", "after", len(checks))
    record = base_record(date, checks, sources, time.monotonic() - started)
    if record["verdict_class"] == "blocked":
        return record
    raw.mkdir(parents=True, exist_ok=True)
    fixtures = prior.fixture_records()
    prior.validate_roles(fixtures)
    training, online = protocols(names)
    training_path, online_path = raw / "training_protocol.json", raw / "online_protocol.json"
    fixture_path = raw / "fixture_records.json"
    atomic_json(training_path, training)
    atomic_json(online_path, online)
    atomic_json(fixture_path, fixtures)
    record["training_protocol_path"] = str(training_path)
    record["online_protocol_path"] = str(online_path)
    rows: list[dict[str, Any]] = []
    fits: list[dict[str, Any]] = []
    phase = time.monotonic()
    code_hash = sha256_file(Path(__file__))
    for arm in ARMS:
        for seed in SEEDS:
            checkpoint = raw / "units" / f"{arm}_{seed}.json"
            progress(started, "numerical", f"before_benchmark_{arm}_{seed}", len(fits))
            saved = json.loads(checkpoint.read_text()) if checkpoint.is_file() else None
            if (
                saved is not None
                and saved.get("code_sha256") == code_hash
                and Path(saved["head_path"]).is_file()
                and sha256_file(Path(saved["head_path"])) == saved["head_hash"]
            ):
                result = saved
            else:
                result = fit_one(arm, seed, fixtures, names, raw / "heads")
                result["code_sha256"] = code_hash
                atomic_json(checkpoint, result)
            rows.extend(result["rows"])
            fits.append({key: value for key, value in result.items() if key != "rows"})
            progress(started, "numerical", f"after_benchmark_{arm}_{seed}", len(fits))
    record["phase_spans"].append(
        {"phase": "numerical", "duration_s": time.monotonic() - phase, "completed_units": len(fits)}
    )
    phase = time.monotonic()
    progress(started, "online", "before_benchmark", 0)
    bank = prior.prior.exercise_online(raw / "online", names)
    progress(started, "online", "after_benchmark", len(bank["query_rows"]))
    record["phase_spans"].append(
        {
            "phase": "online",
            "duration_s": time.monotonic() - phase,
            "completed_units": len(bank["query_rows"]),
        }
    )
    rows_path, fits_path = raw / "rows.json", raw / "training_rows.json"
    atomic_json(rows_path, rows)
    atomic_json(fits_path, fits)
    valid = (
        len(fits) == 27
        and len(rows) == 162
        and bank["valid"]
        and all(
            item["initial_hash"] != item["final_hash"]
            and item["gradient_norm"] > 0
            and item["gradient_error"] < 1e-4
            and item["normalization_error"] < 1e-8
            and item["reload_decision_equal"]
            and item["parameter_count"] <= 4096
            and np.isfinite(item["final_loss"])
            and all(np.isfinite(item["dual_variables"]))
            and (item["dual_effect_nonzero"] if item["mode"] == "constrained" else True)
            for item in fits
        )
    )
    if not valid:
        record["gate_check_summary"].append(
            failed_operand(
                "exp7825", str(fits_path), "fixture_valid", True, False, sha256_file(fits_path)
            )
        )
    record["rows"] = rows
    record["fixture_training_rows"] = fits
    record["online_fixture"] = bank
    record["acceptance_gate_results"]["validity"] = bool(valid)
    record["sample_size_budget"].update(
        eligible=6, started=6, completed=6, excluded=0, censored=0, effective_independent_n=6
    )
    record["raw_paths"] = {
        "rows": str(rows_path),
        "training": str(fits_path),
        "fixtures": str(fixture_path),
    }
    record["reproducibility_inputs"] = {
        "code": code_hash,
        "fixture_data": sha256_file(fixture_path),
        "training_protocol": sha256_file(training_path),
        "online_protocol": sha256_file(online_path),
        "seeds": list(SEEDS),
        "sources": {item["artifact_path"]: item["artifact_hash"] for item in sources},
    }
    record["reproducibility_checksum"] = canonical_hash(record["reproducibility_inputs"])
    record["duration_s"] = time.monotonic() - started
    return record
