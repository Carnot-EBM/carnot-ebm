"""Publish current, exposed source custody after measured checks (REQ-REPORT-7892-V685)."""

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

from coverage import CoverageData

from carnot.reporting import source_boundary_7866 as acquisition
from carnot.reporting import source_boundary_7892 as boundary
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash, sha256_file
from carnot.reporting.experiment_7303_validation_scope import CommandSpec, run_commands
from carnot.verify import source_projection

# A file-path CLI launch puts this directory, rather than the repository root,
# on sys.path. Preserve the qualified prior-producer import in that mode.
if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.experiments import experiment_7880_v684_source_boundary as prior  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "results/experiment_7892_v685_source_boundary.json"
RAW = ROOT / "results/raw/experiment_7892_v685_source_boundary"
SEED = 68592
TESTS = [
    "tests/python/test_source_boundary_7892.py",
    "tests/python/test_source_boundary_7880.py",
    "tests/python/test_source_boundary_7866.py",
    "tests/python/test_source_boundary_7852.py",
    "tests/python/test_source_projection_7838.py",
]
OWNED = [
    "python/carnot/reporting/source_boundary_7892.py",
    "scripts/experiments/experiment_7892_v685_source_boundary.py",
    "python/carnot/reporting/source_boundary_7880.py",
    "scripts/experiments/experiment_7880_v684_source_boundary.py",
]


def progress(start: float, phase: str, event: str, units: int = 0) -> None:
    """Keep the owned process observable even while a child is supervised."""
    print(
        f"[exp7892] phase={phase} event={event} elapsed_s={time.monotonic() - start:.3f} "
        f"completed={units}",
        flush=True,
    )


def operand(upstream: str, path: Path, field: str, expected: Any, observed: Any) -> dict[str, Any]:
    """Name the actual file and field that prevented a gate from opening."""
    return {
        "upstream_id": upstream,
        "path": str(path),
        "hash": sha256_file(path) if path.is_file() else None,
        "artifact_field": field,
        "op": "==",
        "expected": expected,
        "observed": observed,
    }


def base(
    start: float, sources: list[dict[str, Any]], failures: list[dict[str, Any]]
) -> dict[str, Any]:
    """Emit a full schema for an external block as well as a complete run."""
    history = json.loads(prior.OUTPUT.read_text()) if prior.OUTPUT.is_file() else {}
    return {
        "schema": "carnot.exp7892.source_boundary.v1",
        **boundary.IDENTITY,
        "run_date": "20260929",
        "honest_verdict": "complete_blocked_source_evidence",
        "verdict_class": "blocked",
        "flagged_adversarial": False,
        "gate_check_summary": failures,
        "rows": [],
        "sample_size_budget": boundary.budget([], 640),
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
        "historical_required_failures": history.get("historical_required_failures", [])
        + [
            {
                "experiment_id": 7880,
                "name": item["name"],
                "exit_code": item["exit_code"],
                "log_path": item["log_path"],
                "log_sha256": item["log_sha256"],
            }
            for item in history.get("validation_receipts", [])
            if not item.get("passed")
        ],
        "repository_health": {
            "affects_required_checks": False,
            "historical": history.get("repository_health", {}),
        },
        "verifier_is_oracle": True,
        "claim_scope": "exposed_development_custody_only",
        "inference_substrate": "deterministic_cpu",
        "inference_substrate_class": "blocked_no_run" if failures else "no_model_load",
        "planned_inference_substrate_class": "no_model_load",
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
        "public_shards": [],
        "evaluator_shards": [],
        "feature_shards": [],
        "role_counts": {},
        "source_license_receipts": [],
        "coverage_statement_counts": {},
        "identity_checks": {},
        "field_principles": {
            "identity": "Current evidence belongs to its current producer.",
            "roles": "Public family hashes freeze policy use before labels open.",
            "shards": "Original bytes and response labels stay separate.",
            "coverage": "Only fully measured current paths support readiness.",
            "gates": "Custody mechanics cannot prove decision benefit.",
        },
    }


def freeze(private: Path, raw: Path) -> dict[str, Any]:
    """Pin every affected command and include before producing an outcome."""
    py = str(ROOT / ".venv/bin/python")
    pytest = str(ROOT / ".venv/bin/pytest")
    cov = str(ROOT / ".venv/bin/coverage")
    include = ",".join(str((ROOT / name).resolve()) for name in OWNED)
    common = ["-n", "0", "-o", "addopts=", "--no-cov"]
    fixture = private / "fixture with spaces" / "public.jsonl"
    feature = fixture.with_name("features.jsonl")

    def item(
        name: str,
        argv: list[str],
        expected: str = "zero",
        deadline: int = 600,
        reason: str | None = None,
    ) -> dict[str, Any]:
        return {
            "name": name,
            "argv": argv,
            "classification": "required",
            "expected_exit": expected,
            "expected_reason": reason,
            "timeout_s": deadline,
        }

    commands = [
        item(
            "affected_pytest", [pytest, *common, f"--basetemp={private / 'pytest'}", *TESTS, "-q"]
        ),
        item(
            "coverage_unit",
            [
                cov,
                "run",
                f"--data-file={private / 'unit.coverage'}",
                f"--include={include}",
                "-m",
                "pytest",
                *common,
                f"--basetemp={private / 'unit_pytest'}",
                *TESTS,
                "-q",
            ],
        ),
        item(
            "coverage_cli_success",
            [
                cov,
                "run",
                f"--data-file={private / 'success.coverage'}",
                f"--include={include}",
                __file__,
                "--fixture-public",
                str(fixture),
                "--fixture-output",
                str(feature),
            ],
        ),
        item(
            "coverage_cli_failure",
            [
                cov,
                "run",
                f"--data-file={private / 'failure.coverage'}",
                f"--include={include}",
                __file__,
                "--fixture-public",
                str(fixture.with_name("missing.jsonl")),
                "--fixture-output",
                str(feature),
            ],
            "nonzero",
            reason="FileNotFoundError",
        ),
        item(
            "coverage_cold_replay",
            [
                cov,
                "run",
                f"--data-file={private / 'replay.coverage'}",
                f"--include={include}",
                __file__,
                "--fixture-public",
                str(fixture),
                "--fixture-output",
                str(feature),
                "--cold-replay",
            ],
        ),
        item(
            "coverage_historical_cli",
            [
                cov,
                "run",
                f"--data-file={private / 'historical.coverage'}",
                f"--include={include}",
                str(ROOT / OWNED[3]),
                "--fixture-public",
                str(fixture),
                "--fixture-output",
                str(private / "historical_features.jsonl"),
            ],
        ),
        item("ruff_check", [str(ROOT / ".venv/bin/ruff"), "check", *OWNED[:2], *TESTS[:2]]),
        item(
            "ruff_format",
            [str(ROOT / ".venv/bin/ruff"), "format", "--check", *OWNED[:2], *TESTS[:2]],
        ),
        item("mypy", [str(ROOT / ".venv/bin/mypy"), "--strict", *OWNED[:2]]),
        item("scoped_spec_coverage", [py, "scripts/check_spec_coverage.py", *TESTS]),
        item(
            "e2e_015",
            [
                pytest,
                *common,
                f"--basetemp={private / 'e2e_015'}",
                "-q",
                "tests/python/test_source_boundary_7852.py",
            ],
        ),
        item(
            "e2e_016_fixture",
            [
                py,
                "scripts/experiments/experiment_7868_v683_intervention_protocol.py",
                "--date",
                "20260929",
                "--fixture-e2e",
                str(private / "e2e_016.json"),
            ],
        ),
        item(
            "e2e_016_replay",
            [
                py,
                "scripts/experiments/experiment_7868_v683_intervention_protocol.py",
                "--date",
                "20260929",
                "--cold-replay",
                str(private / "e2e_016.json"),
            ],
        ),
        *[
            item(
                f"cold_feature_replay_{index}",
                [
                    py,
                    "-m",
                    "carnot.verify.source_projection",
                    "replay",
                    str(raw / f"public_{index}.jsonl"),
                    str(raw / f"features_{index}.jsonl"),
                ],
            )
            for index in range(2)
        ],
    ]
    return {
        "schema": "carnot.exp7892.validation.v1",
        "task_id": boundary.IDENTITY["task_id"],
        "affected_source_closure": [
            {"path": p, "sha256": sha256_file(ROOT / p)}
            for p in [
                *OWNED,
                "python/carnot/reporting/source_boundary_7866.py",
                "python/carnot/reporting/current_work_receipt.py",
                "python/carnot/reporting/experiment_7303_validation_scope.py",
                "python/carnot/verify/source_projection.py",
                *TESTS,
            ]
        ],
        "affected_tests": TESTS,
        "owned_files": OWNED,
        "coverage_include": include,
        "commands": commands,
        "inapplicable_e2e": [
            {
                "ids": "E2E-001-through-014,E2E-017",
                "reason": "This CPU source producer changes no model, device, service or ARC agent.",
            }
        ],
    }


def child(
    spec: dict[str, Any], index: int, raw: Path, private: Path, start: float
) -> dict[str, Any]:
    """Run one declared child with heartbeats and seal its completed log."""
    progress(start, spec["name"], "before_subprocess", index)
    command = CommandSpec(
        spec["name"], tuple(spec["argv"]), spec["classification"], float(spec["timeout_s"])
    )
    receipt = run_commands(
        ROOT,
        [command],
        log_dir=private / "child_logs" / f"{index:02d}",
        extra_env={"CARNOT_FORCE_LIVE": "1", "JAX_PLATFORMS": "cpu", "PYTHONPATH": "python:."},
        heartbeat_s=30,
    )[0]
    source = ROOT / receipt["log_path"]
    digest = sha256_file(source)
    sealed = raw / "validation_logs" / f"{index:02d}_{spec['name']}_{digest[7:]}.log"
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
            "deadline_s": spec["timeout_s"],
        }
    )
    if spec.get("expected_exit") == "nonzero":
        reason = spec.get("expected_reason")
        receipt["passed"] = (
            receipt["exit_code"] != 0
            and not receipt["timed_out"]
            and reason is not None
            and reason in source.read_text()
        )
    progress(start, spec["name"], "after_subprocess", index + 1)
    return receipt


def custody(
    artifact: dict[str, Any],
    manifest: dict[str, Any],
    license_data: dict[str, Any],
    raw: Path,
    start: float,
) -> None:
    """Reopen originals and publish separate public and evaluator evidence."""
    rows, public = acquisition.acquire(manifest, lambda n: progress(start, "acquire", "rows", n))
    qualified, evaluators, counts = boundary.qualify(rows, public)
    if not acquisition.imports_valid(acquisition.qualified_imports()):
        raise ValueError("resolved_imports_invalid")
    artifact["resolved_imports"] = acquisition.qualified_imports()
    raw.mkdir(parents=True, exist_ok=True)
    features = []
    for index, row in enumerate(public, 1):
        feature = source_projection.extract_row(row)
        features.append(feature)
        qualified[index - 1]["feature_hash"] = feature["feature_hash"]
        qualified[index - 1]["view_a_windows"] = feature["view_a_windows"]
        qualified[index - 1]["view_b_windows"] = feature["view_b_windows"]
        if feature["abstention"]:
            qualified[index - 1]["status"] = "excluded"
            qualified[index - 1]["exclusion_reasons"] = [feature["abstention"]]
        else:
            qualified[index - 1]["exclusion_reasons"] = []
        if index % 32 == 0:
            progress(start, "features", "rows", index)
    source_projection.validate_features(public, features, [row["family_id"] for row in qualified])
    shards: dict[str, list[dict[str, str]]] = {"public": [], "evaluator": [], "feature": []}
    for index in range(2):
        start_index, end_index = index * 320, (index + 1) * 320
        public_part = public[start_index:end_index]
        evaluator_part = evaluators[start_index:end_index]
        feature_part = features[start_index:end_index]
        paths = {
            kind: raw / f"{name}_{index}.jsonl"
            for kind, name in (
                ("public", "public"),
                ("evaluator", "evaluator"),
                ("feature", "features"),
            )
        }
        for kind, part in (
            ("public", public_part),
            ("evaluator", evaluator_part),
            ("feature", feature_part),
        ):
            source_projection.write_jsonl(paths[kind], part)
            if paths[kind].stat().st_size >= 50 * 1024 * 1024:
                raise ValueError(f"raw_shard_too_large:{paths[kind]}")
            shards[kind].append({"path": str(paths[kind]), "sha256": sha256_file(paths[kind])})
        atomic_json(
            paths["feature"].with_suffix(".manifest.json"),
            {
                "public_sha256": source_projection.digest(public_part),
                "features_sha256": source_projection.digest(feature_part),
                "family_ids": [row["family_id"] for row in qualified[start_index:end_index]],
            },
        )
    cohort = {
        "schema": "carnot.exp7892.cohort.v1",
        "rows": qualified,
        "public_shards": shards["public"],
        "evaluator_shards": shards["evaluator"],
        "feature_shards": shards["feature"],
    }
    cohort_path = raw / f"cohort-{canonical_hash(cohort)[7:]}.json"
    if cohort_path.is_file() and json.loads(cohort_path.read_text()) != cohort:
        raise ValueError("cohort_hash_collision")
    if not cohort_path.is_file():
        atomic_json(cohort_path, cohort)
    artifact["cohort_manifest_path"] = str(cohort_path)
    artifact["cohort_manifest_sha256"] = sha256_file(cohort_path)
    artifact["public_shards"] = shards["public"]
    artifact["evaluator_shards"] = shards["evaluator"]
    artifact["feature_shards"] = shards["feature"]
    artifact["rows"] = qualified
    artifact["sample_size_budget"] = boundary.budget(qualified, 640)
    artifact["role_counts"] = counts
    artifact["source_license_receipts"] = [
        {
            "path": str(prior.LICENSE),
            "sha256": sha256_file(prior.LICENSE),
            "license": license_data["license"],
            "url": license_data["repository"],
            "revision": license_data["commit"],
            "label_authority": license_data["label_authority"],
        }
    ]
    artifact["preconditions_checked"].update(
        {
            "source_families": 640,
            "role_counts": counts,
            "original_manifest_sha256": sha256_file(Path(manifest["development_manifest_path"])),
            "public_shard_hashes": [item["sha256"] for item in shards["public"]],
            "evaluator_shard_hashes": [item["sha256"] for item in shards["evaluator"]],
        }
    )


def coverage_counts(private: Path, scope: dict[str, Any]) -> dict[str, dict[str, int]]:
    """Demand measured lines in every declared file before a coverage claim."""
    result: dict[str, dict[str, int]] = {}
    required = [str((ROOT / path).resolve()) for path in scope["owned_files"]]
    for name in ("unit", "success", "failure", "replay", "historical"):
        path = private / f"{name}.coverage"
        if not path.is_file():
            raise ValueError(f"coverage_missing:{name}")
        data = CoverageData(basename=str(path))
        data.read()
        counts = {name: len(data.lines(name) or ()) for name in required}
        if not any(counts.values()):
            raise ValueError(f"coverage_empty:{name}")
        result[name] = counts
    return result


def validate(
    artifact: dict[str, Any], scope: dict[str, Any], raw: Path, private: Path, start: float
) -> None:
    """Run the frozen affected checks and measure the real producer paths."""
    fixture = private / "fixture with spaces" / "public.jsonl"
    fixture.parent.mkdir(parents=True, exist_ok=True)
    source_projection.write_jsonl(
        fixture,
        [{"family_id": "fixture", "source_bytes": b"A. B.".hex(), "answer_bytes": b"A.".hex()}],
    )
    for index, spec in enumerate(scope["commands"]):
        receipt = child(spec, index, raw, private, start)
        artifact["validation_receipts"].append(receipt)
        artifact["observed_child_commands"].append(receipt)
        if not receipt["passed"]:
            artifact["gate_check_summary"].append(
                operand(
                    "exp7892",
                    Path(receipt["log_path"]),
                    spec["name"],
                    {
                        "passed": True,
                        "expected_exit": spec["expected_exit"],
                        "reason": spec.get("expected_reason"),
                    },
                    {"passed": False, "exit_code": receipt["exit_code"]},
                )
            )
    try:
        artifact["coverage_statement_counts"] = coverage_counts(private, scope)
    except ValueError as exc:
        artifact["gate_check_summary"].append(
            operand(
                "exp7892", private, "coverage_statement_counts", "nonempty measured files", str(exc)
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
            *[
                str(private / f"{name}.coverage")
                for name in ("unit", "success", "failure", "replay", "historical")
            ],
        ],
        "classification": "required",
        "expected_exit": "zero",
        "timeout_s": 300,
    }
    report = {
        "name": "coverage_report",
        "argv": [
            str(ROOT / ".venv/bin/coverage"),
            "json",
            f"--data-file={private / 'combined.coverage'}",
            f"--include={scope['coverage_include']}",
            "--fail-under=100",
            "-o",
            str(private / "coverage.json"),
        ],
        "classification": "required",
        "expected_exit": "zero",
        "timeout_s": 300,
    }
    for index, spec in enumerate((combine, report), len(scope["commands"])):
        receipt = child(spec, index, raw, private, start)
        artifact["validation_receipts"].append(receipt)
        artifact["observed_child_commands"].append(receipt)
        if not receipt["passed"]:
            artifact["gate_check_summary"].append(
                operand(
                    "exp7892",
                    Path(receipt["log_path"]),
                    spec["name"],
                    {"passed": True},
                    {"passed": False, "exit_code": receipt["exit_code"]},
                )
            )
    path = private / "coverage.json"
    if path.is_file():
        report_data = json.loads(path.read_text())
        files = report_data["files"]
        artifact["coverage_statement_counts"]["combined"] = {
            str((ROOT / name).resolve()): files.get(name, {}).get("summary", {})
            for name in scope["owned_files"]
        }
        for name in scope["owned_files"]:
            summary = files.get(name, {}).get("summary", {})
            if summary.get("num_statements", 0) == 0 or summary.get("missing_lines", 1):
                artifact["gate_check_summary"].append(
                    operand("exp7892", path, f"coverage:{name}", {"missing_lines": 0}, summary)
                )
    else:
        artifact["gate_check_summary"].append(
            operand("exp7892", path, "coverage_report", "present", None)
        )


def terminal(
    artifact: dict[str, Any], output: Path, raw: Path, private: Path, start: float
) -> dict[str, Any]:
    """Validate exact final bytes and publish only the checked candidate."""

    def classify() -> None:
        if artifact["inference_substrate_class"] == "blocked_no_run":
            verdict, honest = "blocked", "complete_blocked_source_evidence"
        elif artifact["gate_check_summary"]:
            verdict, honest = "disqualified", "complete_disqualified_required_checks"
        else:
            verdict, honest = "circular_positive", "complete_circular_positive_source_boundary"
        artifact["verdict_class"] = verdict
        artifact["honest_verdict"] = honest
        ready = int(verdict == "circular_positive")
        artifact["source_boundary_ready_score"] = ready
        artifact["acceptance_gate_results"]["validity"] = bool(ready)
        artifact["acceptance_gate_results"]["readiness"] = ready
        artifact["identity_checks"] = {
            key: artifact.get(key) == expected for key, expected in boundary.IDENTITY.items()
        }
        boundary.require_identity(artifact)
        artifact["duration_s"] = time.monotonic() - start

    def verify(candidate: Path, offset: int) -> list[dict[str, Any]]:
        checks = [
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
        ]
        return [
            child(
                {
                    "name": name,
                    "argv": argv,
                    "classification": "required_terminal",
                    "timeout_s": 180,
                    "expected_exit": "zero",
                },
                offset + i,
                raw,
                private,
                start,
            )
            for i, (name, argv) in enumerate(checks)
        ]

    classify()
    candidate = private / "terminal_candidate.json"
    atomic_json(candidate, artifact)
    reports = verify(candidate, 100)
    try:
        adversarial = json.loads(reports[0]["output_tail"])
        flagged = bool(adversarial["flagged_count"])
    except (ValueError, KeyError, TypeError):
        flagged = True
    if flagged or not all(report["passed"] for report in reports):
        artifact["flagged_adversarial"] = flagged
        if flagged:
            artifact["gate_check_summary"].append(
                operand("exp7892", Path(reports[0]["log_path"]), "flagged_adversarial", False, True)
            )
        for report in reports:
            if not report["passed"]:
                artifact["gate_check_summary"].append(
                    operand(
                        "exp7892",
                        Path(report["log_path"]),
                        report["name"],
                        {"passed": True},
                        {"passed": False, "exit_code": report["exit_code"]},
                    )
                )
        artifact["inference_substrate_class"] = "no_model_load"
        classify()
        atomic_json(candidate, artifact)
        reports = verify(candidate, 102)
        if not all(report["passed"] for report in reports):
            raise ValueError("final_candidate_verification_failed")
    raw.mkdir(parents=True, exist_ok=True)
    atomic_json(
        raw / "terminal_validation_receipts.json",
        {
            "candidate_path": str(candidate),
            "candidate_sha256": sha256_file(candidate),
            "reports": reports,
        },
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp-{os.getpid()}")
    shutil.copyfile(candidate, temporary)
    os.replace(temporary, output)
    progress(start, "terminal", "published", len(artifact["rows"]))
    return artifact


def run_experiment(date: str, output: Path = OUTPUT, raw: Path = RAW) -> dict[str, Any]:
    """Run current custody and only its frozen, independently checked paths."""
    start = time.monotonic()
    progress(start, "start", "begin")
    if date != "20260929":
        raise ValueError("run_date_mismatch")
    manifest, sources, failures, license_data = prior.preflight(start)
    if prior.OUTPUT.is_file():
        sources.append(
            {
                "upstream_id": "exp7880",
                "path": str(prior.OUTPUT),
                "sha256": sha256_file(prior.OUTPUT),
                "role": "historical_failure_only",
                "eligibility": "historical",
                "date": "20260929",
            }
        )
    else:
        failures.append(operand("exp7880", prior.OUTPUT, "sha256", "present", None))
    artifact = base(start, sources, failures)
    artifact["phase_spans"].append({"phase": "preflight", "duration_s": time.monotonic() - start})
    private = Path("/tmp") / f"exp7892-{canonical_hash({'sources': sources, 'seed': SEED})[7:23]}"
    private.mkdir(parents=True, exist_ok=True)
    if manifest is None or failures:
        return terminal(artifact, output, raw, private, start)
    scope = freeze(private, raw)
    scope_path = raw / "validation_command_manifest.json"
    atomic_json(scope_path, scope)
    artifact["validation_command_manifest_path"] = str(scope_path)
    artifact["validation_command_manifest_sha256"] = sha256_file(scope_path)
    artifact["reproducibility_checksum"] = canonical_hash(
        {
            "sources": sources,
            "closure": scope["affected_source_closure"],
            "validation": sha256_file(scope_path),
            "seed": SEED,
        }
    )
    progress(start, "custody", "begin")
    phase = time.monotonic()
    try:
        custody(artifact, manifest, license_data, raw, start)
    except (OSError, KeyError, ValueError, TypeError) as exc:
        field = (
            "source_family_count"
            if str(exc).startswith(("family_count", "role_count", "missing_human_label"))
            else "owned_conversion"
        )
        artifact["gate_check_summary"].append(
            operand(
                "exp7810" if field == "source_family_count" else "exp7892",
                scope_path,
                field,
                "640 eligible labeled families" if field == "source_family_count" else "valid",
                str(exc),
            )
        )
        if field == "source_family_count":
            artifact["inference_substrate_class"] = "blocked_no_run"
        return terminal(artifact, output, raw, private, start)
    artifact["phase_spans"].append({"phase": "custody", "duration_s": time.monotonic() - phase})
    progress(start, "validation", "begin")
    phase = time.monotonic()
    validate(artifact, scope, raw, private, start)
    artifact["phase_spans"].append({"phase": "validation", "duration_s": time.monotonic() - phase})
    return terminal(artifact, output, raw, private, start)


def main(argv: list[str] | None = None) -> int:
    """Expose explicit private fixture paths and one dated producer command."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--date")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--raw-root", type=Path, default=RAW)
    parser.add_argument("--fixture-public", type=Path)
    parser.add_argument("--fixture-output", type=Path)
    parser.add_argument("--cold-replay", action="store_true")
    args = parser.parse_args(argv)
    if args.fixture_public is not None and args.fixture_output is not None:
        if args.cold_replay:
            source_projection.replay_file(args.fixture_public, args.fixture_output)
        else:
            source_projection.extract_file(args.fixture_public, args.fixture_output)
        return 0
    if args.date is None:
        parser.error("--date is required")
    run_experiment(args.date, args.output, args.raw_root)
    return 0


if __name__ == "__main__":
    sys.exit(main())
