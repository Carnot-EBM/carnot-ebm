#!/usr/bin/env python3
"""Callable byte-fixture qualification for REQ-VERIFY-7867."""

from __future__ import annotations

import argparse
import inspect
import json
import os
from pathlib import Path
import tempfile
import time
from typing import Any

from carnot.reporting.current_work_receipt import (
    atomic_json,
    build_current_work_receipt,
    canonical_hash,
    sha256_file,
)
from carnot.verify import evidence_views, natural_bank, natural_predicates, natural_training
from carnot.verify import training_runtime

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "results/experiment_7867_v683_natural_runtime.json"
MODEL_SPECS: list[dict[str, Any]] = []
SEED = natural_training.SEEDS[0]
SCHEMA = "carnot.exp7867.natural_runtime.v1"
MODULES = (evidence_views, natural_training, natural_predicates, natural_bank, training_runtime)


def progress(start: float, phase: str, event: str, units: int = 0) -> None:
    """Report actual elapsed work at each phase and arm boundary."""
    print(
        f"[exp7867] phase={phase} event={event} elapsed_s={time.monotonic() - start:.3f} completed_units={units}",
        flush=True,
    )


def fixture_records() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Use two visible byte families; names never enter the numeric features."""
    fit = [
        {
            "id": "opaque-fit",
            "group": "fixture-fit",
            "role": "fit",
            "source": "Lumen has 12 apples. Orion has four pears. Vega stores the fruit. The archive stays open.",
            "answer": "Lumen has 12 apples.",
            "label": 0,
            "known": [1],
        }
    ]
    tune = [
        {
            "id": "opaque-tune",
            "group": "fixture-tune",
            "role": "tune",
            "source": "Nova has 8 stones. Mira records each stone. The shelf is blue. Nobody moves the case.",
            "answer": "Nova has 9 stones.",
            "label": 1,
            "known": [0],
        }
    ]
    return fit, tune


def _check(
    source: str, path: Path, digest: str | None, field: str, expected: Any, observed: Any
) -> dict[str, Any]:
    return {
        "upstream_id": source,
        "path": str(path),
        "sha256": digest,
        "artifact_field": field,
        "op": "==",
        "expected": expected,
        "observed": observed,
        "passed": expected == observed,
    }


def split_token(group: str, role: str, digest: str) -> str:
    """Bind one family role to exact source bytes before labels can move."""
    return canonical_hash({"group": group, "role": role, "sha256": digest})


def science_gate(root: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Record source authority even when only fixture mechanics run."""
    path = root / "results/experiment_7866_v683_source_boundary.json"
    digest = sha256_file(path) if path.is_file() else None
    try:
        producer = json.loads(path.read_text())
    except (ValueError, OSError):
        producer = {}
    checks = [
        _check("exp7866-source-boundary", path, digest, field, expected, producer.get(field))
        for field, expected in (
            ("experiment_id", 7866),
            ("milestone", "2026.09.683"),
            ("source_boundary_ready_score", 1),
            ("flagged_adversarial", False),
        )
    ]
    checks.append(
        _check(
            "exp7866-source-boundary",
            path,
            digest,
            "qualified_verdict",
            True,
            producer.get("verdict_class") in {"positive", "circular_positive", "null"},
        )
    )
    for row in checks:
        row["scope"] = "scientific_measurement_only"
    source = {
        "upstream_id": "exp7866-source-boundary",
        "path": str(path),
        "sha256": digest,
        "role": "science_producer",
        "exposure": "exposed_development",
    }
    return checks, source


def preflight(
    manifest: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Bind each declared byte file and role before any numerical fitting."""
    exists = manifest.is_file()
    digest = sha256_file(manifest) if exists else None
    checks = [_check("qualified-manifest", manifest, digest, "exists", True, exists)]
    sources = [
        {
            "upstream_id": "qualified-manifest",
            "path": str(manifest),
            "sha256": digest,
            "role": "qualified_manifest",
            "exposure": "exposed_development",
        }
    ]
    if not exists:
        return checks, sources, []
    try:
        value = json.loads(manifest.read_text())
    except (ValueError, OSError):
        value = {}
    if not isinstance(value, dict):
        value = {}
    checks.append(
        _check(
            "qualified-manifest",
            manifest,
            digest,
            "schema",
            "carnot.natural_runtime.qualified.v1",
            value.get("schema"),
        )
    )
    oracle = value.get("fixture_oracle") is True
    if not oracle:
        producer_path = Path(value.get("producer_path", ""))
        checks.append(
            _check(
                "exp7866-source-boundary",
                producer_path,
                None,
                "producer_path",
                str(ROOT / "results/experiment_7866_v683_source_boundary.json"),
                str(producer_path),
            )
        )
        producer_hash = sha256_file(producer_path) if producer_path.is_file() else None
        sources.append(
            {
                "upstream_id": "exp7866-source-boundary",
                "path": str(producer_path),
                "sha256": producer_hash,
                "role": "science_producer",
                "exposure": "exposed_development",
            }
        )
        checks.append(
            _check(
                "exp7866-source-boundary",
                producer_path,
                producer_hash,
                "sha256",
                value.get("producer_sha256"),
                producer_hash,
            )
        )
        try:
            producer = json.loads(producer_path.read_text())
        except (ValueError, OSError):
            producer = {}
        for field, expected in (
            ("experiment_id", 7866),
            ("milestone", "2026.09.683"),
            ("source_boundary_ready_score", 1),
            ("flagged_adversarial", False),
        ):
            checks.append(
                _check(
                    "exp7866-source-boundary",
                    producer_path,
                    producer_hash,
                    field,
                    expected,
                    producer.get(field),
                )
            )
        checks.append(
            _check(
                "exp7866-source-boundary",
                producer_path,
                producer_hash,
                "cohort_manifest_sha256",
                value.get("cohort_manifest_sha256"),
                producer.get("cohort_manifest_sha256"),
            )
        )
        checks.append(
            _check(
                "exp7866-source-boundary",
                producer_path,
                producer_hash,
                "qualified_verdict",
                True,
                producer.get("verdict_class") in {"positive", "circular_positive", "null"},
            )
        )
    records: list[dict[str, Any]] = []
    seen: dict[str, str] = {}
    entries = value.get("records", [])
    if not isinstance(entries, list):
        entries = []
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            entry = {}
        role = entry.get("role")
        path = Path(entry.get("path", ""))
        actual = sha256_file(path) if path.is_file() else None
        declared = entry.get("sha256")
        upstream = f"qualified-row-{index}"
        sources.append(
            {
                "upstream_id": upstream,
                "path": str(path),
                "sha256": actual,
                "role": role,
                "exposure": "exposed_development",
            }
        )
        checks.append(_check(upstream, path, actual, "sha256", declared, actual))
        checks.append(
            _check(
                upstream,
                path,
                actual,
                "role_allowed",
                True,
                role in {"fit", "tune", "evaluation", "retention"},
            )
        )
        group = entry.get("group")
        checks.append(_check(upstream, path, actual, "group_unique_role", None, seen.get(group)))
        expected_token = (
            split_token(group, role, actual)
            if (isinstance(group, str) and isinstance(role, str) and actual is not None)
            else None
        )
        checks.append(
            _check(upstream, path, actual, "split_token", expected_token, entry.get("split_token"))
        )
        if isinstance(group, str):
            seen[group] = role
        if actual is None:
            continue
        try:
            row = json.loads(path.read_text())
        except (ValueError, OSError):
            row = {}
        checks.append(
            _check(
                upstream,
                path,
                actual,
                "record_schema",
                True,
                all(key in row for key in ("source", "answer", "label")),
            )
        )
        records.append({**row, "id": entry.get("id", group), "group": group, "role": role})
    for role in ("fit", "tune"):
        checks.append(
            _check(
                "qualified-manifest",
                manifest,
                digest,
                f"has_{role}",
                True,
                any(row.get("role") == role for row in records),
            )
        )
    return checks, sources, records


def _manifest(private: Path, sources: list[dict[str, Any]]) -> Path:
    """Publish the callable signatures with the bytes that define this run."""
    path = private / "runtime_interface_v1.json"
    calls = (
        natural_training.prepare,
        natural_training.fit,
        natural_training.predict,
        natural_predicates.features,
        natural_bank.NaturalBank.predict,
        natural_bank.NaturalBank.release,
        natural_bank.NaturalBank.admit,
    )
    atomic_json(
        path,
        {
            "schema": "carnot.natural_runtime.interface.v1",
            "signatures": {
                f"{fn.__module__}.{fn.__qualname__}": str(inspect.signature(fn)) for fn in calls
            },
            "source_hashes": sources,
            "module_hashes": {
                module.__name__: sha256_file(Path(module.__file__)) for module in MODULES
            },
        },
    )
    return path


def _base(
    date: str,
    started_ns: int,
    checks: list[dict[str, Any]],
    sources: list[dict[str, Any]],
    private: Path,
) -> dict[str, Any]:
    """Keep blocked and fixture outputs on one explicit artifact schema."""
    failures = [
        {key: val for key, val in check.items() if key != "passed"}
        for check in checks
        if not check["passed"] and check.get("scope") != "scientific_measurement_only"
    ]
    return {
        "schema": SCHEMA,
        "experiment_id": 7867,
        "task_id": "exp7867-natural-runtime",
        "milestone": "2026.09.683",
        "run_date": date,
        "honest_verdict": "complete_blocked_unqualified_source"
        if failures
        else "complete_circular_positive_fixture_runtime",
        "verdict_class": "blocked" if failures else "circular_positive",
        "flagged_adversarial": False,
        "gate_check_summary": failures,
        "scientific_gate_check_summary": [
            {key: val for key, val in check.items() if key != "passed"}
            for check in checks
            if not check["passed"] and check.get("scope") == "scientific_measurement_only"
        ],
        "rows": [],
        "fixture_rows": [],
        "sample_size_budget": {
            key: 0
            for key in (
                "intended",
                "eligible",
                "started",
                "completed",
                "censored",
                "excluded",
                "independent",
            )
        },
        "acceptance_gate_results": {
            "validity": not failures,
            "readiness": not failures,
            "probability_quality": None,
            "decision_benefit": None,
            "retention": None,
            "efficiency": None,
        },
        "duration_s": 0.0,
        "phase_spans": [],
        "random_seed": [SEED],
        "source_artifact_hashes": sources,
        "preconditions_checked": checks,
        "validation_receipts": [],
        "validation_command_manifest_path": None,
        "observed_child_commands": [],
        "repository_health": {"status": "not_run"},
        "verifier_is_oracle": not failures,
        "claim_scope": "fixture_mechanics_only" if not failures else "blocked_before_measurement",
        "inference_substrate": "cpu_no_pretrained_model" if not failures else "blocked_no_run",
        "inference_substrate_class": "no_model_load" if not failures else "blocked_no_run",
        "planned_inference_substrate_class": "no_model_load",
        "execution_venue": "host",
        "actual_compute": "verifier_ensemble_against_cached_candidates",
        "MODEL_SPECS": [],
        "model_specs": [],
        "target_model": "none (no pretrained model)",
        "model_invocation_counts": {"loads": 0, "calls": 0, "tokens": 0, "model_file_hashes": []},
        "natural_training_ready_score": 0 if failures else 1,
        "natural_online_ready_score": 0 if failures else 1,
        "natural_measurement_performed": False,
        "trained_head_specs": [],
        "runtime_manifest_path": None,
        "private_root": str(private),
        "started_monotonic_ns": started_ns,
        "historical_obligations": [
            {
                "experiment_id": prior,
                "path": str(ROOT / relative),
                "sha256": sha256_file(ROOT / relative),
            }
            for prior, relative in (
                (7825, "results/experiment_7825_v680_training_runtime.json"),
                (7853, "results/experiment_7853_v682_natural_runtime.json"),
            )
            if (ROOT / relative).is_file()
        ],
    }


def _finish(record: dict[str, Any], started_ns: int, start: float) -> dict[str, Any]:
    """Bind the observed duration and exact files after all owned work."""
    ended_ns = time.monotonic_ns()
    record["duration_s"] = (ended_ns - started_ns) / 1e9
    record["phase_spans"].append(
        {
            "phase": "total",
            "duration_s": record["duration_s"],
            "completed_units": record["sample_size_budget"]["completed"],
        }
    )
    record["code_hashes"] = [
        {"path": str(path), "sha256": sha256_file(path)}
        for path in [Path(__file__), *(Path(module.__file__) for module in MODULES)]
    ]
    record["reproducibility_checksum"] = canonical_hash(
        {
            "schema": SCHEMA,
            "sources": record["source_artifact_hashes"],
            "code": record["code_hashes"],
            "seed": record["random_seed"],
            "configuration": {"arms": list(natural_training.ARMS), "epochs": 1},
        }
    )
    record["current_work_receipt"] = build_current_work_receipt(
        run_id=f"exp7867-{record['run_date']}-{os.getpid()}",
        owner_pid=os.getpid(),
        events=[],
        inference_substrate=record["inference_substrate"],
        inference_substrate_details={"model_files": []},
        inference_substrate_class=record["inference_substrate_class"],
        execution_venue="host",
        started_monotonic_ns=started_ns,
        ended_monotonic_ns=ended_ns,
        phase_spans=record["phase_spans"],
    )
    record["resolved_imports"] = {
        module.__name__: str(Path(module.__file__).resolve()) for module in MODULES
    }
    record["field_principles"] = {
        key: (
            "Keep validity, component readiness and scientific benefit separate."
            if key == "acceptance_gate_results"
            else "Bind this field to current bytes and observed work; fixture outcomes have no natural authority."
        )
        for key in record
    }
    for gate in record["acceptance_gate_results"]:
        record["field_principles"][f"acceptance_gate_results.{gate}"] = (
            "Keep unmeasured benefit null and failed readiness explicit."
        )
    progress(start, "finalize", "after", record["sample_size_budget"]["completed"])
    return record


def run_fixture(
    private: Path, date: str, *, qualified_manifest: Path | None = None
) -> dict[str, Any]:
    """Fit real byte-derived heads and report only the scope that ran."""
    start = time.monotonic()
    started_ns = time.monotonic_ns()
    progress(start, "start", "entered")
    private.mkdir(parents=True, exist_ok=True)
    progress(start, "preflight", "before")
    if qualified_manifest is None:
        fit, tune = fixture_records()
        sources = [
            {
                "upstream_id": row["group"],
                "path": None,
                "sha256": evidence_views.digest(row["source"].encode() + row["answer"].encode()),
                "role": row["role"],
                "exposure": "deterministic_fixture",
            }
            for row in [*fit, *tune]
        ]
        checks = [
            _check(
                "fixture",
                Path(__file__),
                sha256_file(Path(__file__)),
                "fixture_groups_disjoint",
                True,
                fit[0]["group"] != tune[0]["group"],
            )
        ]
        science_checks, science_source = science_gate(ROOT)
        checks.extend(science_checks)
        sources.append(science_source)
    else:
        checks, sources, records = preflight(qualified_manifest)
        fit = [row for row in records if row.get("role") == "fit"]
        tune = [row for row in records if row.get("role") == "tune"]
    record = _base(date, started_ns, checks, sources, private)
    progress(start, "preflight", "after")
    if record["gate_check_summary"]:
        return _finish(record, started_ns, start)
    if qualified_manifest is not None:
        qualified = json.loads(qualified_manifest.read_text())
        if qualified.get("fixture_oracle") is not True:
            record["honest_verdict"] = "complete_null_exposed_natural_runtime"
            record["verdict_class"] = "null"
            record["verifier_is_oracle"] = False
            record["claim_scope"] = "exposed_development_natural_measurement"
            record["natural_measurement_performed"] = True
    manifest_path = _manifest(private, sources)
    record["runtime_manifest_path"] = str(manifest_path)
    record["runtime_manifest_sha256"] = sha256_file(manifest_path)
    record["sample_size_budget"]["intended"] = len(natural_training.ARMS)
    for index, arm in enumerate(natural_training.ARMS):
        progress(start, "fit", f"before_{arm}", index)
        head = natural_training.fit(fit, tune, arm, SEED, 0.01, 1)
        predictions = natural_training.predict(head, tune)
        batch, _ = natural_training.prepare(tune, arm)
        left = float(training_runtime.predict(head["params"], batch["a"], head["arm"])[0])
        right = float(training_runtime.predict(head["params"], batch["b"], head["arm"])[0])
        probability = predictions[0]["probability_unsupported"]
        label = int(tune[0]["label"])
        head_path = private / f"head_{arm}_{SEED}.json"
        training_runtime.save(head_path, head)
        record["trained_head_specs"].append(
            {
                "arm": arm,
                "seed": SEED,
                "path": str(head_path),
                "sha256": sha256_file(head_path),
                "parameter_count": head["parameter_count"],
                "temperature": head["temperature"],
            }
        )
        row = {
            "arm": arm,
            "family": tune[0]["group"],
            "seed": SEED,
            "status": "completed",
            "intended": 1,
            "eligible": 1,
            "started": 1,
            "completed": 1,
            "censored": 0,
            "excluded": 0,
            "independent": 0,
            "feature_dim": 132,
            "temperature": head["temperature"],
            "view_divergence": float(
                training_runtime.constraints(head["params"], batch, head["arm"])[0]
            ),
            "risk_a": left,
            "risk_b": right,
            "probability_unsupported": probability,
            "label": label,
            "action": predictions[0]["action"],
            "brier": (probability - label) ** 2,
            "gradient_error": head["gradient_error"],
        }
        record["rows"].append(row)
        record["fixture_rows"].append(
            {
                "arm": arm,
                "family": row["family"],
                "probability_unsupported": probability,
                "action": row["action"],
                "oracle_label": label,
            }
        )
        progress(start, "fit", f"after_{arm}", index + 1)
    record["sample_size_budget"].update(
        {
            "eligible": len(record["rows"]),
            "started": len(record["rows"]),
            "completed": len(record["rows"]),
            "independent": len({row["group"] for row in [*fit, *tune]}),
        }
    )
    progress(start, "online", "before", len(record["rows"]))
    bank_path = private / "natural_bank.json"
    bank = natural_bank.NaturalBank(bank_path)
    feature = natural_predicates.features(fit[0]["source"].encode(), b"Lumen has 13 apples.")
    bank.predict("coefficient-training", 0, feature, 0.1)
    bank.release("coefficient-training", 1, 1)
    before = bank.predict("probe", 2, feature, 0.04, read_only=True)
    bank.predict("admission", 3, feature, 0.04)
    coefficients = list(bank.state["coefficients"])
    bank.release("admission", 4, 1, admission_only=True)
    bank.admit("admission", "and_unmatched_decimal")
    after = natural_bank.NaturalBank(bank_path).predict("later", 5, feature, 0.04, read_only=True)
    if bank.state["coefficients"] != coefficients or after["probability"] <= before["probability"]:
        raise ValueError("causal admission qualification failed")
    record["online_probe"] = {
        "before": before,
        "after": after,
        "bank_path": str(bank_path),
        "bank_sha256": sha256_file(bank_path),
    }
    progress(start, "online", "after", len(record["rows"]) + 1)
    record["phase_spans"].append(
        {
            "phase": "fit_and_online",
            "duration_s": time.monotonic() - start,
            "completed_units": len(record["rows"]) + 1,
        }
    )
    return _finish(record, started_ns, start)


def cold_replay(path: Path) -> dict[str, Any]:
    """Check exact source, code and output row arithmetic in a cold process."""
    value = json.loads(path.read_text())
    valid = all(sha256_file(Path(item["path"])) == item["sha256"] for item in value["code_hashes"])
    valid = valid and all(
        item["path"] is None
        or (Path(item["path"]).is_file() and sha256_file(Path(item["path"])) == item["sha256"])
        for item in value["source_artifact_hashes"]
    )
    valid = valid and value["reproducibility_checksum"] == canonical_hash(
        {
            "schema": SCHEMA,
            "sources": value["source_artifact_hashes"],
            "code": value["code_hashes"],
            "seed": value["random_seed"],
            "configuration": {"arms": list(natural_training.ARMS), "epochs": 1},
        }
    )
    rows = value["rows"]
    valid = valid and value["sample_size_budget"]["completed"] == sum(
        row["completed"] for row in rows
    )
    valid = valid and all(
        abs(row["brier"] - (row["probability_unsupported"] - row["label"]) ** 2) < 1e-12
        for row in rows
    )
    manifest = value.get("runtime_manifest_path")
    valid = valid and (
        manifest is None or sha256_file(Path(manifest)) == value["runtime_manifest_sha256"]
    )
    return {"valid": bool(valid), "row_count": len(rows)}


def main(argv: list[str] | None = None) -> int:
    """Run a private fixture or hash-bound source and publish one terminal file."""
    start = time.monotonic()
    progress(start, "cli", "entered")
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", default="20260929")
    parser.add_argument("--fixture", action="store_true")
    parser.add_argument("--qualified-manifest", type=Path)
    parser.add_argument("--private-root", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--cold-replay", type=Path)
    args = parser.parse_args(argv)
    if args.cold_replay:
        progress(start, "cold_replay", "before")
        report = cold_replay(args.cold_replay)
        print(json.dumps(report, sort_keys=True), flush=True)
        progress(start, "cold_replay", "after", report["row_count"])
        return 0 if report["valid"] else 1
    if args.date != "20260929" or (args.fixture and args.qualified_manifest):
        parser.error("date or source mode is invalid")
    private = args.private_root or Path(tempfile.mkdtemp(prefix="exp7867-", dir="/tmp"))
    record = run_fixture(private, args.date, qualified_manifest=args.qualified_manifest)
    output = args.output or OUTPUT
    atomic_json(output, record)
    progress(start, "publish", "after", record["sample_size_budget"]["completed"])
    print(
        json.dumps({"output": str(output), "honest_verdict": record["honest_verdict"]}), flush=True
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
