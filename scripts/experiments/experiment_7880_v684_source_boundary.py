"""Qualify original exposed source custody with measured validation (REQ-REPORT-7880)."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import shutil
import sys
import time
from typing import Any

from carnot.reporting import source_boundary_7866 as acquisition
from carnot.reporting import source_boundary_7880 as controls
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash, sha256_file
from carnot.reporting.experiment_7303_validation_scope import CommandSpec, run_commands
from carnot.verify import source_projection


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "results/experiment_7880_v684_source_boundary.json"
RAW = ROOT / "results/raw/experiment_7880_v684_source_boundary"
UPSTREAM = ROOT / "results/experiment_7810_v679_source_view_qualification.json"
HISTORY = ROOT / "results/experiment_7866_v683_source_boundary.json"
LICENSE = ROOT / "results/raw/experiment_7423_v651_annotated_protocol/corpus_manifest.json"
SEED = 68480
TASK = "exp7880-v684-source-boundary"


def progress(start: float, phase: str, event: str, units: int = 0) -> None:
    """Give the supervisor real elapsed time and completed work at every boundary."""
    print(
        f"[exp7880] phase={phase} event={event} elapsed_s={time.monotonic() - start:.3f} completed={units}",
        flush=True,
    )


def operand(upstream: str, path: Path, field: str, expected: Any, observed: Any) -> dict[str, Any]:
    """Record the exact failed source or required child operand."""
    return {
        "upstream_id": upstream,
        "path": str(path),
        "hash": sha256_file(path) if path.is_file() else None,
        "artifact_field": field,
        "op": "==",
        "expected": expected,
        "observed": observed,
    }


def preflight(
    start: float,
) -> tuple[dict[str, Any] | None, list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Authenticate original manifests and programs before current projection."""
    failures: list[dict[str, Any]] = []
    sources: list[dict[str, Any]] = []
    for name, path, role in (
        ("exp7810", UPSTREAM, "science_producer"),
        ("exp7866", HISTORY, "cached_candidate_only"),
        ("exp7423", LICENSE, "license_authority"),
    ):
        digest = sha256_file(path) if path.is_file() else None
        sources.append(
            {
                "upstream_id": name,
                "path": str(path),
                "sha256": digest,
                "date": "20260928" if name == "exp7810" else "20260929",
                "role": role,
                "eligibility": "exposed_development",
            }
        )
        if digest is None:
            failures.append(operand(name, path, "sha256", "present", None))
    for name in ("python", "pytest", "coverage", "ruff", "mypy"):
        path = ROOT / ".venv/bin" / name
        if not path.is_file():
            failures.append(operand("exp7880", path, "executable", "present", None))
    if failures:
        return None, sources, failures, {}
    upstream, history, license_data = (
        json.loads(path.read_text()) for path in (UPSTREAM, HISTORY, LICENSE)
    )
    for field, expected in (
        ("verdict_class", "circular_positive"),
        ("evidence_view_ready_score", 1),
        ("run_date", "20260928"),
        ("flagged_adversarial", False),
    ):
        if upstream.get(field) != expected:
            failures.append(operand("exp7810", UPSTREAM, field, expected, upstream.get(field)))
    if history.get("verdict_class") != "disqualified":
        failures.append(
            operand(
                "exp7866", HISTORY, "verdict_class", "disqualified", history.get("verdict_class")
            )
        )
    if license_data.get("license") != "MIT":
        failures.append(operand("exp7423", LICENSE, "license", "MIT", license_data.get("license")))
    for entry in upstream.get("source_artifact_hashes", []):
        path = Path(entry["path"])
        observed = sha256_file(path) if path.is_file() else None
        sources.append({**entry, "sha256": observed, "role": "original_source_or_evaluator"})
        if observed != entry["sha256"]:
            failures.append(
                operand(entry["upstream_id"], path, "sha256", entry["sha256"], observed)
            )
    manifest_path = Path(upstream.get("source_view_manifest_path") or "/missing")
    if not manifest_path.is_file():
        failures.append(operand("exp7810", manifest_path, "sha256", "present", None))
        return None, sources, failures, license_data
    manifest = json.loads(manifest_path.read_text())
    for field, expected in (
        ("schema", "carnot.exp7768.source_view_manifest.v1"),
        ("role_counts", acquisition.ROLES),
    ):
        if manifest.get(field) != expected:
            failures.append(operand("exp7810", manifest_path, field, expected, manifest.get(field)))
    for field, hash_field in (
        ("development_manifest_path", "development_manifest_sha256"),
        ("rows_path", "rows_sha256"),
    ):
        path = Path(manifest.get(field) or "/missing")
        observed = sha256_file(path) if path.is_file() else None
        sources.append(
            {
                "upstream_id": "exp7810",
                "path": str(path),
                "sha256": observed,
                "role": field,
                "date": "20260928",
                "eligibility": "exposed_development",
            }
        )
        if observed != manifest.get(hash_field):
            failures.append(operand("exp7810", path, "sha256", manifest.get(hash_field), observed))
    progress(start, "preflight", "checked", len(sources))
    return (manifest if not failures else None), sources, failures, license_data


def base(
    start: float, sources: list[dict[str, Any]], failures: list[dict[str, Any]]
) -> dict[str, Any]:
    """Populate the complete contract even for a terminal external block."""
    return {
        "schema": "carnot.exp7880.source_boundary.v1",
        "experiment_id": 7880,
        "task_id": TASK,
        "milestone": "2026.09.684",
        "run_date": "20260929",
        "honest_verdict": "complete_blocked_source_evidence",
        "verdict_class": "blocked",
        "flagged_adversarial": False,
        "gate_check_summary": failures,
        "rows": [],
        "sample_size_budget": {
            key: 640 if key == "intended" else 0
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
            "validity": False,
            "readiness": 0,
            "probability_quality": None,
            "decision_benefit": None,
            "retention": None,
            "efficiency": None,
        },
        "duration_s": 0.0,
        "phase_spans": [],
        "random_seed": SEED,
        "reproducibility_checksum": None,
        "source_artifact_hashes": sources,
        "preconditions_checked": {"failed": failures},
        "resolved_imports": {},
        "validation_receipts": [],
        "validation_command_manifest_path": None,
        "observed_child_commands": [],
        "historical_required_failures": [],
        "repository_health": {},
        "verifier_is_oracle": True,
        "claim_scope": "exposed_development_custody_only",
        "field_principles": {
            "identity": "Bind current evidence to its own producer.",
            "rows": "Count primitive families without adding views or seeds.",
            "coverage": "Only executed statements in exact files support readiness.",
            "roles": "Public hash fixes disjoint policy use before labels open.",
            "gates": "Custody readiness does not show decision benefit.",
        },
        "inference_substrate": "verifier_ensemble_against_cached_candidates",
        "inference_substrate_class": "blocked_no_run" if failures else "no_model_load",
        "planned_inference_substrate_class": "no_model_load",
        "execution_venue": "host",
        "MODEL_SPECS": [],
        "model_specs": [],
        "target_model": "none (no pretrained model)",
        "model_invocation_counts": {
            "model_loads_attempted": 0,
            "model_loads_completed": 0,
            "generation_calls_attempted": 0,
            "generation_calls_completed": 0,
        },
        "trained_head_specs": [],
        "source_boundary_ready_score": 0,
        "cohort_manifest_path": None,
        "cohort_manifest_sha256": None,
        "role_counts": acquisition.ROLES,
        "policy_subroles": {"policy_design": 32, "calibration_replay": 32},
        "source_license_receipts": [],
        "coverage_measured_files": [],
        "coverage_statement_counts": {},
    }


def freeze(private: Path) -> dict[str, Any]:
    """Name the dependency closure and all child commands before results exist."""
    python = str(ROOT / ".venv/bin/python")
    pytest = str(ROOT / ".venv/bin/pytest")
    coverage = str(ROOT / ".venv/bin/coverage")
    ruff = str(ROOT / ".venv/bin/ruff")
    mypy = str(ROOT / ".venv/bin/mypy")
    module = "python/carnot/reporting/source_boundary_7880.py"
    script = "scripts/experiments/experiment_7880_v684_source_boundary.py"
    owned = [module, script]
    tests = [
        "tests/python/test_source_boundary_7880.py",
        "tests/python/test_source_boundary_7866.py",
        "tests/python/test_source_boundary_7852.py",
        "tests/python/test_source_projection_7838.py",
    ]
    closure = [
        *owned,
        "python/carnot/reporting/source_boundary_7866.py",
        "python/carnot/reporting/natural_source_cohort.py",
        "python/carnot/reporting/current_work_receipt.py",
        "python/carnot/reporting/experiment_7303_validation_scope.py",
        "python/carnot/verify/source_projection.py",
        "python/carnot/verify/source_alignment.py",
        *tests,
    ]
    include = ",".join(str((ROOT / path).resolve()) for path in owned)
    common = ["-n", "0", "-o", "addopts=", "--no-cov"]
    fixture = private / "fixture with spaces" / "public.jsonl"
    feature = private / "fixture with spaces" / "features.jsonl"

    def item(
        name: str, argv: list[str], timeout: int = 300, expected: str = "zero"
    ) -> dict[str, Any]:
        return {
            "name": name,
            "argv": argv,
            "timeout_s": timeout,
            "classification": "required",
            "expected_exit": expected,
        }

    commands = [
        item(
            "worktree_imports",
            [
                python,
                "-c",
                "import json; from carnot.reporting.source_boundary_7866 import qualified_imports; print(json.dumps({'resolved_imports':qualified_imports()}))",
            ],
        ),
        item(
            "affected_pytest",
            [pytest, *common, f"--basetemp={private / 'pytest'}", *tests, "-q"],
            600,
        ),
        item(
            "coverage_unit",
            [
                coverage,
                "run",
                f"--data-file={private / 'unit.coverage'}",
                f"--include={include}",
                "-m",
                "pytest",
                *common,
                f"--basetemp={private / 'unit_pytest'}",
                *tests,
                "-q",
            ],
            600,
        ),
        item(
            "coverage_cli_success",
            [
                coverage,
                "run",
                f"--data-file={private / 'cli_success.coverage'}",
                f"--include={include}",
                script,
                "--fixture-public",
                str(fixture),
                "--fixture-output",
                str(feature),
            ],
        ),
        item(
            "coverage_cli_failure",
            [
                coverage,
                "run",
                f"--data-file={private / 'cli_failure.coverage'}",
                f"--include={include}",
                script,
                "--fixture-public",
                str(fixture.with_name("missing.jsonl")),
                "--fixture-output",
                str(feature),
            ],
            expected="nonzero",
        ),
        item("ruff_check", [ruff, "check", *owned, tests[0]]),
        item("ruff_format", [ruff, "format", "--check", *owned, tests[0]]),
        item("mypy", [mypy, "--strict", *owned], 600),
        item("scoped_spec_coverage", [python, "scripts/check_spec_coverage.py", *tests]),
        item(
            "e2e_015", [pytest, *common, f"--basetemp={private / 'e2e_015'}", "-q", tests[2]], 300
        ),
        item(
            "cold_replay",
            [python, "-m", "carnot.verify.source_projection", "replay", str(fixture), str(feature)],
        ),
    ]
    return {
        "schema": "carnot.exp7880.validation.v1",
        "task_id": TASK,
        "affected_source_closure": [
            {"path": path, "sha256": sha256_file(ROOT / path)} for path in closure
        ],
        "affected_tests": tests,
        "owned_files": owned,
        "coverage_include": include,
        "commands": commands,
        "inapplicable_e2e": [
            {
                "ids": "E2E-001-through-014,E2E-016-through-017",
                "reason": "This CPU source-custody run changes no model, device, service, or ARC agent.",
            }
        ],
    }


def child(spec: dict[str, Any], index: int, private: Path, start: float) -> dict[str, Any]:
    """Supervise a bounded child and seal its closed log at a hash path."""
    progress(start, spec["name"], "before_subprocess", index)
    command = CommandSpec(
        spec["name"], tuple(spec["argv"]), spec["classification"], float(spec["timeout_s"])
    )
    receipt = run_commands(
        ROOT,
        [command],
        log_dir=private / "child_logs" / f"{index:02d}",
        extra_env={"CARNOT_FORCE_LIVE": "1", "JAX_PLATFORMS": "cpu"},
        heartbeat_s=30,
    )[0]
    source = ROOT / receipt["log_path"]
    digest = sha256_file(source)
    sealed = RAW / "validation_logs" / f"{index:02d}_{spec['name']}_{digest[7:]}.log"
    sealed.parent.mkdir(parents=True, exist_ok=True)
    if sealed.exists() and sha256_file(sealed) != digest:
        raise ValueError("sealed_log_collision")
    if not sealed.exists():
        shutil.copyfile(source, sealed)
    receipt.update(
        {
            "log_path": str(sealed),
            "log_sha256": digest,
            "classification": spec["classification"],
            "expected_exit": spec.get("expected_exit", "zero"),
        }
    )
    if spec.get("expected_exit") == "nonzero":
        receipt["passed"] = receipt["exit_code"] != 0 and not receipt["timed_out"]
    progress(start, spec["name"], "after_subprocess", index + 1)
    return receipt


def fixture_cli(public: Path, output: Path) -> None:
    """Exercise the real public-only CLI on success and failure paths."""
    source_projection.extract_file(public, output)
    source_projection.replay_file(public, output)


def custody(
    artifact: dict[str, Any],
    manifest: dict[str, Any],
    license_data: dict[str, Any],
    private: Path,
    start: float,
) -> None:
    """Rebuild public bytes and evaluator joins from original authenticated files."""
    rows, public = acquisition.acquire(manifest, lambda n: progress(start, "custody", "rows", n))
    if len(rows) != 640 or len({row["family_id"] for row in rows}) != 640:
        raise ValueError(f"source_family_count:{len(rows)}")
    subroles = controls.policy_subroles(rows)
    for row in rows:
        if row["role"] == "policy":
            row["policy_subrole"] = subroles[row["family_id"]]
    if not acquisition.imports_valid(acquisition.qualified_imports()):
        raise ValueError("resolved_imports_invalid")
    artifact["resolved_imports"] = acquisition.qualified_imports()
    artifact["source_license_receipts"] = [
        {
            "path": str(LICENSE),
            "sha256": sha256_file(LICENSE),
            "license": license_data["license"],
            "url": license_data["repository"],
            "revision": license_data["commit"],
            "label_authority": license_data["label_authority"],
        }
    ]
    RAW.mkdir(parents=True, exist_ok=True)
    public_path = RAW / "public.jsonl"
    evaluator_path = RAW / "evaluator.jsonl"
    source_projection.write_jsonl(public_path, public)
    source_projection.write_jsonl(
        evaluator_path,
        [
            {
                "family_id": row["family_id"],
                "role": row["role"],
                "policy_subrole": row.get("policy_subrole"),
                "label_provenance": row["label_provenance"],
            }
            for row in rows
        ],
    )
    features: list[dict[str, Any]] = []
    for index, row in enumerate(public, 1):
        feature = source_projection.extract_row(row)
        features.append(feature)
        original = rows[index - 1]
        original["feature_hash"] = feature["feature_hash"]
        original["view_a_windows"] = feature["view_a_windows"]
        original["view_b_windows"] = feature["view_b_windows"]
        original["status"] = "excluded" if feature["abstention"] else "completed"
        original["exclusion_reasons"] = [feature["abstention"]] if feature["abstention"] else []
        if index % 32 == 0:
            progress(start, "projection", "rows", index)
    source_projection.validate_features(public, features, [row["family_id"] for row in rows])
    feature_path = RAW / "features.jsonl"
    source_projection.write_jsonl(feature_path, features)
    checksum = canonical_hash(
        {
            "rows": rows,
            "public": sha256_file(public_path),
            "evaluator": sha256_file(evaluator_path),
            "features": sha256_file(feature_path),
        }
    )
    cohort_path = RAW / f"cohort-{checksum[7:]}.json"
    if cohort_path.exists() and json.loads(cohort_path.read_text()).get("rows") != rows:
        raise ValueError("immutable_cohort_collision")
    if not cohort_path.exists():
        atomic_json(
            cohort_path,
            {
                "schema": "carnot.exp7880.cohort.v1",
                "rows": rows,
                "public_shard": {"path": str(public_path), "sha256": sha256_file(public_path)},
                "evaluator_shard": {
                    "path": str(evaluator_path),
                    "sha256": sha256_file(evaluator_path),
                },
                "feature_shard": {"path": str(feature_path), "sha256": sha256_file(feature_path)},
                "source_license": artifact["source_license_receipts"],
            },
        )
    artifact["cohort_manifest_path"] = str(cohort_path)
    artifact["cohort_manifest_sha256"] = sha256_file(cohort_path)
    artifact["rows"] = [
        {
            key: value
            for key, value in row.items()
            if key not in {"complete_source", "complete_response", "label_provenance"}
        }
        for row in rows
    ]
    budget = acquisition.reduce_budget(rows, 640)
    budget["eligible"] = 640
    artifact["sample_size_budget"] = budget
    artifact["role_counts"] = dict(Counter(row["role"] for row in rows))
    artifact["policy_subroles"] = dict(Counter(subroles.values()))
    history = json.loads(HISTORY.read_text())
    candidate = history.get("cached_candidate_receipt", {})
    cached = Path(candidate.get("path") or "/missing")
    artifact["cached_candidate_receipt"] = {
        "path": str(cached),
        "sha256": sha256_file(cached) if cached.is_file() else None,
        "authority": "cached_candidate_only",
        "comparison": "recomputed_from_authenticated_public_bytes",
    }
    artifact["preconditions_checked"].update(
        {
            "manifest_schema": manifest["schema"],
            "role_counts": manifest["role_counts"],
            "source_families": len(rows),
            "public_shard_sha256": sha256_file(public_path),
            "evaluator_shard_sha256": sha256_file(evaluator_path),
        }
    )


def validate(artifact: dict[str, Any], scope: dict[str, Any], private: Path, start: float) -> None:
    """Run frozen children and reject empty per-file coverage before combining."""
    fixture = private / "fixture with spaces" / "public.jsonl"
    fixture.parent.mkdir(parents=True, exist_ok=True)
    source_projection.write_jsonl(
        fixture,
        [{"family_id": "fixture", "source_bytes": b"A. B.".hex(), "answer_bytes": b"A.".hex()}],
    )
    expected = [ROOT / path for path in scope["owned_files"]]
    for index, spec in enumerate(scope["commands"]):
        receipt = child(spec, index, private, start)
        artifact["validation_receipts"].append(receipt)
        artifact["observed_child_commands"].append(receipt)
        if not receipt["passed"]:
            artifact["gate_check_summary"].append(
                operand(
                    "exp7880",
                    Path(receipt["log_path"]),
                    spec["name"],
                    {"passed": True},
                    {"exit_code": receipt["exit_code"], "passed": False},
                )
            )
        if spec["name"] == "worktree_imports" and not acquisition.imports_valid(
            receipt.get("resolved_imports", {})
        ):
            artifact["gate_check_summary"].append(
                operand(
                    "exp7880",
                    Path(receipt["log_path"]),
                    "resolved_imports",
                    acquisition.qualified_imports(),
                    receipt.get("resolved_imports"),
                )
            )
        if spec["name"] in {"coverage_unit", "coverage_cli_success", "coverage_cli_failure"}:
            name = {
                "coverage_unit": "unit",
                "coverage_cli_success": "cli_success",
                "coverage_cli_failure": "cli_failure",
            }[spec["name"]]
            required = expected[:1] if name == "unit" else expected[1:]
            try:
                counts = controls.measured_counts(private / f"{name}.coverage", required)
                artifact["coverage_statement_counts"][name] = counts
                artifact["coverage_measured_files"].extend(counts)
            except ValueError as exc:
                artifact["gate_check_summary"].append(
                    operand(
                        "exp7880",
                        private / f"{name}.coverage",
                        "coverage_statement_counts",
                        "nonzero for explicit files",
                        str(exc),
                    )
                )
    if artifact["gate_check_summary"]:
        return
    combine = {
        "name": "coverage_combine",
        "argv": [
            str(ROOT / ".venv/bin/coverage"),
            "combine",
            f"--data-file={private / 'combined.coverage'}",
            str(private / "unit.coverage"),
            str(private / "cli_success.coverage"),
            str(private / "cli_failure.coverage"),
        ],
        "classification": "required",
        "timeout_s": 300,
    }
    report = {
        "name": "coverage_report",
        "argv": [
            str(ROOT / ".venv/bin/coverage"),
            "report",
            f"--data-file={private / 'combined.coverage'}",
            f"--include={scope['coverage_include']}",
            "--show-missing",
            "--fail-under=100",
        ],
        "classification": "required",
        "timeout_s": 300,
    }
    for index, spec in enumerate((combine, report), len(scope["commands"])):
        receipt = child(spec, index, private, start)
        artifact["validation_receipts"].append(receipt)
        artifact["observed_child_commands"].append(receipt)
        if not receipt["passed"]:
            artifact["gate_check_summary"].append(
                operand(
                    "exp7880",
                    Path(receipt["log_path"]),
                    spec["name"],
                    {"passed": True},
                    {"exit_code": receipt["exit_code"], "passed": False},
                )
            )
    if not artifact["gate_check_summary"]:
        artifact["coverage_statement_counts"]["combined"] = controls.measured_counts(
            private / "combined.coverage", expected
        )


def terminal(artifact: dict[str, Any], start: float, private: Path) -> dict[str, Any]:
    """Validate exact final bytes, keep reports outside the candidate, and publish atomically."""
    verdict = (
        "blocked"
        if artifact["inference_substrate_class"] == "blocked_no_run"
        else ("disqualified" if artifact["gate_check_summary"] else "circular_positive")
    )
    artifact["verdict_class"] = verdict
    artifact["honest_verdict"] = {
        "blocked": "complete_blocked_source_evidence",
        "disqualified": "complete_disqualified_required_checks",
        "circular_positive": "complete_circular_positive_source_boundary_readiness",
    }[verdict]
    artifact["source_boundary_ready_score"] = int(verdict == "circular_positive")
    artifact["acceptance_gate_results"]["validity"] = verdict == "circular_positive"
    artifact["acceptance_gate_results"]["readiness"] = int(verdict == "circular_positive")
    artifact["duration_s"] = time.monotonic() - start
    candidate = private / "terminal_candidate.json"
    atomic_json(candidate, artifact)
    reports = []
    for index, (name, argv) in enumerate(
        (
            (
                "adversarial_verify",
                [
                    str(ROOT / ".venv/bin/python"),
                    "scripts/adversarial_verify.py",
                    "--json",
                    str(candidate),
                ],
            ),
            (
                "strict_rows",
                [
                    str(ROOT / ".venv/bin/python"),
                    "scripts/verdict_row_consistency_lint.py",
                    "--strict",
                    str(candidate),
                ],
            ),
        )
    ):
        report = child(
            {"name": name, "argv": argv, "timeout_s": 180, "classification": "required_terminal"},
            index + 100,
            private,
            start,
        )
        reports.append(report)
    flagged = False
    for report in reports:
        if report["name"] == "adversarial_verify":
            try:
                parsed = json.loads(report["output_tail"])
                flagged = bool(parsed["flagged_count"])
            except (ValueError, KeyError, TypeError):
                flagged = True
    if flagged != artifact["flagged_adversarial"] or not all(item["passed"] for item in reports):
        artifact["flagged_adversarial"] = flagged
        for report in reports:
            if not report["passed"]:
                artifact["gate_check_summary"].append(
                    operand(
                        "exp7880",
                        Path(report["log_path"]),
                        report["name"],
                        {"passed": True},
                        {"exit_code": report["exit_code"], "passed": False},
                    )
                )
        artifact["verdict_class"] = "disqualified"
        artifact["honest_verdict"] = "complete_disqualified_required_checks"
        artifact["source_boundary_ready_score"] = 0
        artifact["acceptance_gate_results"]["validity"] = False
        artifact["acceptance_gate_results"]["readiness"] = 0
        artifact["duration_s"] = time.monotonic() - start
        atomic_json(candidate, artifact)
        reports = []
        for index, (name, argv) in enumerate(
            (
                (
                    "adversarial_verify",
                    [
                        str(ROOT / ".venv/bin/python"),
                        "scripts/adversarial_verify.py",
                        "--json",
                        str(candidate),
                    ],
                ),
                (
                    "strict_rows",
                    [
                        str(ROOT / ".venv/bin/python"),
                        "scripts/verdict_row_consistency_lint.py",
                        "--strict",
                        str(candidate),
                    ],
                ),
            )
        ):
            reports.append(
                child(
                    {
                        "name": f"final_{name}",
                        "argv": argv,
                        "timeout_s": 180,
                        "classification": "required_terminal",
                    },
                    index + 102,
                    private,
                    start,
                )
            )
        if not all(item["passed"] for item in reports):
            raise ValueError("final_candidate_verification_failed")
    RAW.mkdir(parents=True, exist_ok=True)
    atomic_json(
        RAW / "terminal_validation_receipts.json",
        {
            "candidate_sha256": sha256_file(candidate),
            "candidate_path": str(candidate),
            "reports": reports,
        },
    )
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    temporary = OUTPUT.with_name(f".{OUTPUT.name}.tmp-{os.getpid()}")
    shutil.copyfile(candidate, temporary)
    os.replace(temporary, OUTPUT)
    progress(start, "terminal", "published", len(artifact["rows"]))
    return artifact


def run_experiment(date: str) -> dict[str, Any]:
    """Acquire originals, validate measured scope, and close the current artifact."""
    start = time.monotonic()
    progress(start, "start", "begin")
    if date != "20260929":
        raise ValueError("run_date_mismatch")
    manifest, sources, failures, license_data = preflight(start)
    artifact = base(start, sources, failures)
    artifact["phase_spans"].append({"phase": "preflight", "duration_s": time.monotonic() - start})
    history = json.loads(HISTORY.read_text()) if HISTORY.is_file() else {}
    historical = [
        {
            "experiment_id": 7866,
            "name": receipt["name"],
            "exit_code": receipt["exit_code"],
            "log_path": receipt["log_path"],
            "log_sha256": receipt["log_sha256"],
        }
        for receipt in history.get("validation_receipts", [])
        if not receipt.get("passed")
    ]
    artifact["historical_required_failures"] = historical
    artifact["repository_health"] = {
        "historical_exp7866_verdict": history.get("honest_verdict"),
        "historical_required_failures": historical,
        "affects_required_checks": False,
        "full_suite": history.get("repository_health", {}).get("historical_full_suite"),
    }
    private = Path("/tmp") / f"exp7880-{canonical_hash({'sources': sources, 'seed': SEED})[7:23]}"
    private.mkdir(parents=True, exist_ok=True)
    if manifest is None:
        return terminal(artifact, start, private)
    scope = freeze(private)
    scope_path = RAW / "validation_command_manifest.json"
    atomic_json(scope_path, scope)
    artifact["validation_command_manifest_path"] = str(scope_path)
    artifact["validation_command_manifest_sha256"] = sha256_file(scope_path)
    artifact["reproducibility_checksum"] = canonical_hash(
        {
            "sources": sources,
            "closure": scope["affected_source_closure"],
            "config": artifact["validation_command_manifest_sha256"],
            "seed": SEED,
        }
    )
    progress(start, "custody", "begin")
    try:
        custody(artifact, manifest, license_data, private, start)
    except (OSError, KeyError, ValueError, TypeError) as exc:
        field = (
            "source_family_count"
            if str(exc).startswith("source_family_count")
            else "owned_conversion"
        )
        artifact["gate_check_summary"].append(
            operand(
                "exp7810" if field == "source_family_count" else "exp7880",
                scope_path,
                field,
                "640 unique" if field == "source_family_count" else "valid",
                str(exc),
            )
        )
        if field == "source_family_count":
            artifact["inference_substrate_class"] = "blocked_no_run"
        return terminal(artifact, start, private)
    artifact["phase_spans"].append(
        {"phase": "custody_projection", "duration_s": time.monotonic() - start}
    )
    progress(start, "validation", "begin")
    validate(artifact, scope, private, start)
    artifact["phase_spans"].append({"phase": "validation", "duration_s": time.monotonic() - start})
    return terminal(artifact, start, private)


def main(argv: list[str] | None = None) -> int:
    """Expose a private public fixture route and the dated science run."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--date")
    parser.add_argument("--fixture-public", type=Path)
    parser.add_argument("--fixture-output", type=Path)
    args = parser.parse_args(argv)
    if args.fixture_public is not None and args.fixture_output is not None:
        fixture_cli(args.fixture_public, args.fixture_output)
        return 0
    if args.date is None:
        parser.error("--date is required")
    run_experiment(args.date)
    return 0


if __name__ == "__main__":
    sys.exit(main())
