"""Reduce historical compatibility checks without invoking old models.

REQ-REPORT-7782 and SCENARIO-REPORT-7782-RECEIPT.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import argparse
from pathlib import Path
import signal
import subprocess
import sys
import time
from typing import Any

JsonDict = dict[str, Any]
MANIFEST = Path("tests/python/fixtures/experiment_7782_frozen_scope.json")
HISTORICAL_FILES = (
    Path("tests/python/fixtures/roadmap_2026_09_675.yaml"),
    Path("tests/python/fixtures/roadmap_design_2026_09_675.md"),
)
EXPECTED_DIGESTS = (
    "0fc9604cda91cc5198a95aad47ea0f48f700d5f5cd18cb5ce9b3fa3bc7f87171",
    "bef49b376955937ca54df4651063dd4e117b192bce18ee3f4a8a1f2103175bef",
)


def sha256_file(path: Path) -> str:
    """Hash exact bytes so a receipt cannot survive a changed source."""
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def checksum(value: JsonDict) -> str:
    """Bind code, evidence, roles, commands, and seed without self reference."""
    encoded = json.dumps(
        {key: item for key, item in value.items() if key != "reproducibility_checksum"},
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def load_manifest(root: Path) -> JsonDict:
    """Require the frozen list rather than discovering nodes after repair."""
    value = json.loads((root / MANIFEST).read_text())
    if len(value.get("diagnosed_nodes", [])) != 23:
        raise ValueError("frozen_node_count_invalid")
    return value


def command_receipt(
    name: str, argv: list[str], exit_code: int, log_path: Path, duration_s: float
) -> JsonDict:
    """Keep the child command, real exit, and immutable log hash together."""
    return {
        "name": name,
        "argv": argv,
        "exit_code": exit_code,
        "passed": exit_code == 0,
        "log_path": str(log_path),
        "log_sha256": sha256_file(log_path),
        "duration_s": duration_s,
    }


def _source(root: Path, path: Path, imported_fields: list[str], eligibility: str) -> JsonDict:
    absolute = path if path.is_absolute() else root / path
    exists = absolute.is_file()
    observed: JsonDict = {}
    run_date = None
    if exists and path.suffix == ".json":
        try:
            data = json.loads(absolute.read_text())
            if isinstance(data, dict):
                observed = {field: data.get(field) for field in imported_fields}
                run_date = data.get("run_date")
        except (OSError, ValueError):
            observed = {"parse_error": True}
    return {
        "path": str(path),
        "sha256": sha256_file(absolute) if exists else None,
        "date": run_date or "20260927" if exists else None,
        "imported_fields": imported_fields,
        "observed_fields": observed,
        "eligibility": eligibility,
        "exists": exists,
    }


def _fixture_ready(root: Path) -> bool:
    return all(
        (root / path).is_file() and sha256_file(root / path) == "sha256:" + digest
        for path, digest in zip(HISTORICAL_FILES, EXPECTED_DIGESTS, strict=True)
    )


def _rows(manifest: JsonDict, *, affected: bool, collection: bool) -> list[JsonDict]:
    """Retain each old node even when a current child has not started."""
    return [
        {
            **node,
            "disposition": "resolved"
            if affected and collection
            else "unstarted"
            if not affected
            else "collection_failed",
            "metrics": {"affected_passed": affected, "collection_passed": collection},
        }
        for node in manifest["diagnosed_nodes"]
    ]


def build_candidate(
    root: Path,
    manifest: JsonDict,
    receipts: list[JsonDict],
    spans: list[JsonDict],
    *,
    flagged_adversarial: bool,
) -> JsonDict:
    """Build one terminal administrative record from actual child exits."""
    by_name = {row["name"]: row for row in receipts}
    affected = by_name.get("affected", {}).get("passed") is True
    collection = by_name.get("collection", {}).get("passed") is True
    fixture_ready = _fixture_ready(root) and affected
    failed = [row for row in receipts if not row["passed"]]
    required_names = {"affected", "collection"}
    missing = sorted(required_names - by_name.keys())
    gate_checks = [
        {
            "upstream_id": "Exp7782:validation",
            "artifact_path": row["log_path"],
            "artifact_hash": row["log_sha256"],
            "field": f"{row['name']}.exit_code",
            "operator": "==",
            "expected": 0,
            "observed": row["exit_code"],
        }
        for row in failed
    ]
    gate_checks.extend(
        {
            "upstream_id": "Exp7782:validation",
            "artifact_path": None,
            "artifact_hash": None,
            "field": f"{name}.exit_code",
            "operator": "==",
            "expected": 0,
            "observed": "not_started",
        }
        for name in missing
    )
    if not _fixture_ready(root):
        gate_checks.append(
            {
                "upstream_id": "V675:fixture",
                "artifact_path": str(HISTORICAL_FILES[0]),
                "artifact_hash": None,
                "field": "immutable_fixture_hash",
                "operator": "==",
                "expected": list(EXPECTED_DIGESTS),
                "observed": [
                    sha256_file(root / path).removeprefix("sha256:")
                    if (root / path).is_file()
                    else None
                    for path in HISTORICAL_FILES
                ],
            }
        )
    if flagged_adversarial:
        gate_checks.append(
            {
                "upstream_id": "Exp7782:terminal_reader",
                "artifact_path": None,
                "artifact_hash": None,
                "field": "flagged_adversarial",
                "operator": "==",
                "expected": False,
                "observed": True,
            }
        )
    rows = _rows(manifest, affected=affected, collection=collection)
    sources = [_source(root, MANIFEST, ["diagnosed_nodes"], "frozen_scope")]
    sources.extend(
        _source(
            root,
            path,
            ["milestone", "tasks"] if path.suffix == ".yaml" else ["Exact Task Contract"],
            "historical_fixture",
        )
        for path in HISTORICAL_FILES
    )
    for name in (
        "7768_v676_source_view_qualification",
        "7770_v676_qwen_runner_qualification",
        "7779_v676_hardware_evidence",
    ):
        sources.append(
            _source(
                root,
                Path(f"results/experiment_{name}.json"),
                ["honest_verdict", "gate_check_summary", "validation_receipts"],
                "diagnostic_only",
            )
        )
    sources.append(
        _source(
            root,
            Path(
                "results/raw/experiment_7782_v677_historical_compatibility/v676_full_python_suite.log"
            ),
            ["collection_errors"],
            "historical_diagnostic",
        )
    )
    sources.append(
        _source(
            root,
            Path(
                "results/raw/experiment_7768_v676_source_view_qualification/validation_logs/01_focused_pytest.log"
            ),
            ["capstone_failures"],
            "historical_diagnostic",
        )
    )
    for path in (
        "python/carnot/experiment_7131_v626_model_facing_csl.py",
        "python/carnot/experiment_7495_v656_window_calibration.py",
        "python/carnot/experiment_7496_v656_causal_update_fixture.py",
        "python/carnot/experiment_7782_v677_historical_compatibility.py",
        "tests/python/test_experiment_7782_v677_historical_compatibility.py",
    ):
        sources.append(_source(root, Path(path), ["implementation"], "current_code"))
    missing_sources = [source for source in sources if not source["exists"]]
    gate_checks.extend(
        {
            "upstream_id": "Exp7782:source",
            "artifact_path": source["path"],
            "artifact_hash": source["sha256"],
            "field": "exists",
            "operator": "==",
            "expected": True,
            "observed": False,
        }
        for source in missing_sources
    )
    qualified = not gate_checks and not flagged_adversarial
    collection_errors = []
    collection_log = by_name.get("collection", {}).get("log_path")
    if not collection and collection_log and Path(collection_log).is_file():
        collection_errors = re.findall(
            r"^ERROR (tests/python/\S+?\.py)", Path(collection_log).read_text(), re.M
        )
    if not collection_errors and not collection:
        collection_errors = [row["node"] for row in rows if row["disposition"] != "resolved"]
    value: JsonDict = {
        "experiment_id": "experiment_7782_v677_historical_compatibility",
        "milestone": "2026.09.677",
        "run_date": "20260927",
        "honest_verdict": "complete_blocked_missing_historical_input"
        if missing_sources
        else "complete_circular_positive_historical_compatibility"
        if qualified
        else "complete_disqualified_required_validation",
        "verdict_class": "blocked"
        if missing_sources
        else "circular_positive"
        if qualified
        else "disqualified",
        "flagged_adversarial": flagged_adversarial,
        "gate_check_summary": gate_checks,
        "rows": rows,
        "compatibility_rows": rows,
        "remaining_collection_errors": collection_errors,
        "collection_ready_score": int(collection and qualified),
        "historical_fixture_ready_score": int(fixture_ready and qualified),
        "acceptance_gate_results": {
            "validity": int(qualified),
            "readiness": int(qualified and affected and collection and fixture_ready),
            "probability_quality": None,
            "decision_benefit": None,
            "retention": None,
            "efficiency": None,
        },
        "duration_s": sum(float(span["duration_s"]) for span in spans),
        "phase_spans": spans,
        "random_seed": 7782,
        "sample_size_budget": {
            "intended": 23,
            "eligible": 23,
            "started": 23 if affected else 0,
            "completed": 23 if affected else 0,
            "excluded": 0,
            "censored": 0,
            "independent_n": 23,
        },
        "source_artifact_hashes": sources,
        "preconditions_checked": {
            "input_paths_present": all(row["exists"] for row in sources),
            "custody_hashes_present": all(row["sha256"] for row in sources),
            "backend": "host_cpu_aggregation",
            "gpu_used": False,
            "cpu_count": os.cpu_count(),
        },
        "validation_receipts": receipts,
        "verifier_is_oracle": True,
        "claim_scope": {
            "kind": "historical_fixture_compatibility",
            "science": "unmeasured",
            "fresh_generalization_eligible": False,
        },
        "inference_substrate": "verifier_ensemble_against_cached_candidates",
        "inference_substrate_class": "no_model_load",
        "planned_inference_substrate_class": "no_model_load",
        "MODEL_SPECS": [],
        "model_specs": [],
        "model_invocation_counts": {
            "loads": 0,
            "generations": 0,
            "forwards": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "loaded_files": [],
        },
    }
    value["field_principles"] = {
        key: "This field prevents an unsupported compatibility claim." for key in value
    }
    value["field_principles"]["field_principles"] = "Each field states the failure it prevents."
    value["field_principles"]["acceptance_gate_results"] = {
        key: "A working fixture cannot establish scientific benefit."
        for key in value["acceptance_gate_results"]
    }
    value["field_principles"]["honest_verdict"] = (
        "A complete record prevents retries on unchanged evidence."
    )
    value["field_principles"]["source_artifact_hashes"] = (
        "Exact bytes prevent an older receipt from replacing a missing producer."
    )
    value["field_principles"]["duration_s"] = (
        "Measured time prevents a named substrate from inflating work."
    )
    value["field_principles"]["reproducibility_checksum"] = (
        "This checksum binds code, data, roles, parameters, and seed."
    )
    value["reproducibility_checksum"] = checksum(value)
    return value


def cold_validate(value: JsonDict, root: Path) -> list[str]:
    """Reopen sources and recompute every named compatibility disposition."""
    errors = []
    if value.get("reproducibility_checksum") != checksum(value):
        errors.append("reproducibility_checksum_mismatch")
    if set(value.get("field_principles", {})) != set(value):
        errors.append("field_principles_incomplete")
    try:
        manifest = load_manifest(root)
    except (OSError, ValueError, TypeError):
        return [*errors, "frozen_manifest_invalid"]
    receipts = value.get("validation_receipts", [])
    by_name = {row.get("name"): row for row in receipts}
    expected_rows = _rows(
        manifest,
        affected=by_name.get("affected", {}).get("passed") is True,
        collection=by_name.get("collection", {}).get("passed") is True,
    )
    if value.get("compatibility_rows") != expected_rows or value.get("rows") != expected_rows:
        errors.append("compatibility_rows_mismatch")
    for source in value.get("source_artifact_hashes", []):
        path = Path(source["path"])
        absolute = path if path.is_absolute() else root / path
        if not absolute.is_file() or sha256_file(absolute) != source.get("sha256"):
            errors.append(f"source_hash_mismatch:{source['path']}")
    for receipt in receipts:
        path = Path(receipt["log_path"])
        if not path.is_file() or sha256_file(path) != receipt.get("log_sha256"):
            errors.append(f"validation_log_mismatch:{receipt['name']}")
        if receipt.get("passed") != (receipt.get("exit_code") == 0):
            errors.append(f"validation_exit_mismatch:{receipt['name']}")
    if value.get("duration_s") != sum(
        float(span["duration_s"]) for span in value.get("phase_spans", [])
    ):
        errors.append("duration_mismatch")
    if (
        value.get("MODEL_SPECS") != []
        or value.get("model_specs") != []
        or value.get("model_invocation_counts", {}).get("loads") != 0
    ):
        errors.append("model_invocation_mismatch")
    if value.get("historical_fixture_ready_score") and not _fixture_ready(root):
        errors.append("historical_fixture_mismatch")
    return errors


def atomic_json(path: Path, value: JsonDict) -> None:
    """Replace one complete file only after its bytes reach local storage."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as handle:
        json.dump(value, handle, sort_keys=True, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def run_child(
    root: Path,
    name: str,
    argv: list[str],
    log_path: Path,
    *,
    timeout_s: float,
    env_extra: dict[str, str] | None = None,
) -> JsonDict:
    """Supervise only the process group started for this check."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    environment = {
        **os.environ,
        "PYTHONPATH": "python:.",
        "CARNOT_FORCE_LIVE": "1",
        "JAX_PLATFORMS": "cpu",
        **(env_extra or {}),
    }
    started = time.monotonic()
    print(
        f"phase={name} event=before_subprocess elapsed_s=0 completed_units=0 argv={argv}",
        flush=True,
    )
    with log_path.open("w") as log:
        child = subprocess.Popen(
            argv,
            cwd=root,
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        while child.poll() is None:
            try:
                child.wait(timeout=30)
            except subprocess.TimeoutExpired:
                elapsed = time.monotonic() - started
                print(
                    f"phase={name} event=supervised_child elapsed_s={elapsed:.1f} completed_units=0 pid={child.pid}",
                    flush=True,
                )
                if elapsed >= timeout_s:
                    os.killpg(child.pid, signal.SIGTERM)
                    try:
                        child.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        os.killpg(child.pid, signal.SIGKILL)
                        child.wait()
    elapsed = time.monotonic() - started
    print(
        f"phase={name} event=after_subprocess elapsed_s={elapsed:.1f} completed_units=1 exit={child.returncode}",
        flush=True,
    )
    return command_receipt(name, argv, int(child.returncode), log_path, elapsed)


def validation_plan(
    root: Path, manifest: JsonDict, private: Path
) -> list[tuple[str, list[str], float, dict[str, str]]]:
    """Freeze explicit file targets before the first validation child starts."""
    affected = sorted(
        set(
            [
                *manifest["collection_nodes"],
                "tests/python/test_experiment_7766_v675_capstone.py",
                "tests/python/test_experiment_7782_v677_historical_compatibility.py",
                "tests/python/test_inference_sota_models.py",
            ]
        )
    )
    modules = [
        "python/carnot/experiment_7131_v626_model_facing_csl.py",
        "python/carnot/experiment_7495_v656_window_calibration.py",
        "python/carnot/experiment_7496_v656_causal_update_fixture.py",
        "python/carnot/experiment_7782_v677_historical_compatibility.py",
    ]
    changed = [
        *modules,
        "python/carnot/experiment_5512_structured_output_positive_control.py",
        "python/carnot/experiment_5759_sota_exact_proposal_utility_panel.py",
        "python/carnot/experiment_5786_sota_constraint_stream.py",
        "python/carnot/experiment_5799_sota_answer_channel_canary.py",
        "python/carnot/experiment_6607_gemma4_26b_direct_headroom.py",
        "tests/python/test_experiment_7766_v675_capstone.py",
        "tests/python/test_experiment_7782_v677_historical_compatibility.py",
    ]
    pytest_args = ["-q", "-n", "0", "-o", "addopts=", "--no-cov"]
    return [
        (
            "affected",
            [
                ".venv/bin/pytest",
                *affected,
                *pytest_args,
                f"--basetemp={private / 'basetemp/affected'}",
            ],
            180,
            {},
        ),
        (
            "collection",
            [
                ".venv/bin/pytest",
                "--collect-only",
                "tests/python",
                *pytest_args,
                f"--basetemp={private / 'basetemp/collection'}",
            ],
            180,
            {},
        ),
        (
            "coverage",
            [
                ".venv/bin/coverage",
                "run",
                "--include=*/experiment_7131_v626_model_facing_csl.py,*/experiment_7495_v656_window_calibration.py,*/experiment_7496_v656_causal_update_fixture.py",
                "-m",
                "pytest",
                "tests/python/test_experiment_7131_v626_model_facing_csl.py",
                "tests/python/test_experiment_7495_v656_window_calibration.py",
                "tests/python/test_experiment_7496_v656_causal_update_fixture.py",
                "tests/python/test_experiment_7782_v677_historical_compatibility.py",
                *pytest_args,
                f"--basetemp={private / 'basetemp/coverage'}",
            ],
            90,
            {"COVERAGE_FILE": str(private / "coverage.7782")},
        ),
        (
            "coverage_report",
            [".venv/bin/coverage", "report", "-m", "--fail-under=100"],
            20,
            {"COVERAGE_FILE": str(private / "coverage.7782")},
        ),
        ("ruff_check", [".venv/bin/ruff", "check", *changed], 20, {}),
        ("ruff_format", [".venv/bin/ruff", "format", "--check", *changed], 20, {}),
        ("mypy", [".venv/bin/mypy", *modules], 60, {}),
        (
            "spec_coverage",
            [".venv/bin/python", "scripts/check_spec_coverage.py", *affected],
            30,
            {},
        ),
        (
            "e2e_014",
            [
                ".venv/bin/pytest",
                "tests/python/test_experiment_7770_v676_qwen_runner_qualification.py",
                *pytest_args,
                f"--basetemp={private / 'basetemp/e2e'}",
            ],
            180,
            {},
        ),
        (
            "e2e_014_cold_replay",
            [
                ".venv/bin/python",
                "scripts/experiments/experiment_7770_v676_qwen_runner_qualification.py",
                "--cold-replay",
                "results/experiment_7770_v676_qwen_runner_qualification.json",
            ],
            40,
            {},
        ),
        (
            "full_python_suite",
            [
                ".venv/bin/pytest",
                "tests/python",
                *pytest_args,
                f"--basetemp={private / 'basetemp/full'}",
            ],
            240,
            {},
        ),
    ]


def main(argv: list[str] | None = None) -> int:
    """Run bounded CPU validations, replay readers, then publish the receipt."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", required=True)
    parser.add_argument("--cold-replay")
    args = parser.parse_args(argv)
    root = Path(__file__).resolve().parents[2]
    if args.date != "20260927":
        parser.error("--date must be 20260927")
    if args.cold_replay:
        candidate = json.loads(Path(args.cold_replay).read_text())
        errors = cold_validate(candidate, root)
        print(json.dumps({"errors": errors}, sort_keys=True), flush=True)
        return int(bool(errors))
    total_started = time.monotonic()
    private = Path("/tmp/carnot-7782/run")
    (private / "basetemp").mkdir(parents=True, exist_ok=True)
    logs = root / "results/raw/experiment_7782_v677_historical_compatibility/validation_logs"
    logs.mkdir(parents=True, exist_ok=True)
    manifest = load_manifest(root)
    preflight = build_candidate(root, manifest, [], [], flagged_adversarial=False)
    missing_input = any(not source["exists"] for source in preflight["source_artifact_hashes"])
    if missing_input or not _fixture_ready(root):
        deliverable = root / "results/experiment_7782_v677_historical_compatibility.json"
        atomic_json(deliverable, preflight)
        print("phase=preflight event=terminal_block elapsed_s=0 completed_units=0", flush=True)
        return 0
    print("phase=preflight event=complete elapsed_s=0 completed_units=23", flush=True)
    receipts = []
    spans = []
    for index, (name, command, timeout_s, environment) in enumerate(
        validation_plan(root, manifest, private), start=1
    ):
        receipt = run_child(
            root,
            name,
            command,
            logs / f"{index:02d}_{name}.log",
            timeout_s=timeout_s,
            env_extra=environment,
        )
        receipts.append(receipt)
        spans.append({"phase": name, "duration_s": receipt["duration_s"]})
    candidate_path = private / "candidate.json"
    candidate = build_candidate(root, manifest, receipts, spans, flagged_adversarial=False)
    atomic_json(candidate_path, candidate)
    terminal = [
        (
            "cold_replay",
            [
                ".venv/bin/python",
                "scripts/experiments/experiment_7782_v677_historical_compatibility.py",
                "--date",
                args.date,
                "--cold-replay",
                str(candidate_path),
            ],
        ),
        (
            "adversarial_verify",
            [".venv/bin/python", "scripts/adversarial_verify.py", "--json", str(candidate_path)],
        ),
        (
            "strict_row_consistency",
            [
                ".venv/bin/python",
                "scripts/verdict_row_consistency_lint.py",
                "--strict",
                str(candidate_path),
            ],
        ),
    ]
    for index, (name, command) in enumerate(terminal, start=len(receipts) + 1):
        receipt = run_child(root, name, command, logs / f"{index:02d}_{name}.log", timeout_s=30)
        receipts.append(receipt)
        spans.append({"phase": name, "duration_s": receipt["duration_s"]})
    adversarial_log = Path(
        next(row["log_path"] for row in receipts if row["name"] == "adversarial_verify")
    )
    try:
        flagged = json.loads(adversarial_log.read_text()).get("flagged_count", 1) > 0
    except (ValueError, OSError):
        flagged = True
    final = build_candidate(root, manifest, receipts, spans, flagged_adversarial=flagged)
    if cold_validate(final, root):
        final["gate_check_summary"].append(
            {
                "upstream_id": "Exp7782:cold_reader",
                "artifact_path": str(candidate_path),
                "artifact_hash": sha256_file(candidate_path),
                "field": "cold_validate.errors",
                "operator": "==",
                "expected": [],
                "observed": cold_validate(final, root),
            }
        )
        final["honest_verdict"] = "complete_disqualified_cold_replay"
        final["verdict_class"] = "disqualified"
        final["collection_ready_score"] = 0
        final["historical_fixture_ready_score"] = 0
        final["acceptance_gate_results"]["validity"] = 0
        final["acceptance_gate_results"]["readiness"] = 0
        final["reproducibility_checksum"] = checksum(final)
    deliverable = root / "results/experiment_7782_v677_historical_compatibility.json"
    atomic_json(deliverable, final)
    print(
        f"phase=publish event=after_atomic_write elapsed_s={time.monotonic() - total_started:.1f} completed_units={len(final['compatibility_rows'])} path={deliverable}",
        flush=True,
    )
    return 0 if final["verdict_class"] != "disqualified" else 1
