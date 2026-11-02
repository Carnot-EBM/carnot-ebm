"""Requalify exposed public source custody and exact current imports (REQ-REPORT-7866)."""

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

from carnot.reporting.current_work_receipt import (
    atomic_json,
    build_current_work_receipt,
    canonical_hash,
    sha256_file,
)
from carnot.reporting.experiment_7303_validation_scope import CommandSpec, run_commands
from carnot.reporting import source_boundary_7866 as boundary
from carnot.verify import source_projection

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "results/experiment_7866_v683_source_boundary.json"
RAW = ROOT / "results/raw/experiment_7866_v683_source_boundary"
UPSTREAM = ROOT / "results/experiment_7810_v679_source_view_qualification.json"
HISTORY = ROOT / "results/experiment_7852_v682_source_boundary.json"
LICENSE = ROOT / "results/raw/experiment_7423_v651_annotated_protocol/corpus_manifest.json"
SEED = 68366


def progress(start: float, phase: str, event: str, units: int = 0) -> None:
    """Show each real phase edge and completed count while the task runs."""
    print(
        f"[exp7866] phase={phase} event={event} elapsed_s={time.monotonic() - start:.3f} completed={units}",
        flush=True,
    )


def gate(upstream: str, path: Path, field: str, expected: Any, observed: Any) -> dict[str, Any]:
    """Retain the exact missing or mismatched operand for a terminal block."""
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
    """Authenticate source bytes before any owned projection or child starts."""
    failed: list[dict[str, Any]] = []
    sources: list[dict[str, Any]] = []
    for name, path in (("exp7810", UPSTREAM), ("exp7852", HISTORY), ("exp7423", LICENSE)):
        digest = sha256_file(path) if path.is_file() else None
        sources.append(
            {
                "upstream_id": name,
                "path": str(path),
                "sha256": digest,
                "date": "20260928"
                if name == "exp7810"
                else "20260929"
                if name == "exp7852"
                else "20260801",
                "role": "science_producer"
                if name == "exp7810"
                else "historical_diagnostic"
                if name == "exp7852"
                else "license_authority",
                "eligibility": "exposed_development",
            }
        )
        if digest is None:
            failed.append(gate(name, path, "sha256", "present", None))
    if failed:
        return None, sources, failed, {}
    upstream = json.loads(UPSTREAM.read_text())
    history = json.loads(HISTORY.read_text())
    license_receipt = json.loads(LICENSE.read_text())
    for field, expected in (
        ("verdict_class", "circular_positive"),
        ("evidence_view_ready_score", 1),
        ("run_date", "20260928"),
        ("flagged_adversarial", False),
    ):
        if upstream.get(field) != expected:
            failed.append(gate("exp7810", UPSTREAM, field, expected, upstream.get(field)))
    for field, expected in (("verdict_class", "disqualified"), ("source_boundary_ready_score", 0)):
        if history.get(field) != expected:
            failed.append(gate("exp7852", HISTORY, field, expected, history.get(field)))
    if license_receipt.get("license") != "MIT":
        failed.append(gate("exp7423", LICENSE, "license", "MIT", license_receipt.get("license")))
    for item in history.get("source_artifact_hashes", []):
        path = Path(item["path"])
        observed = sha256_file(path) if path.is_file() else None
        sources.append({**item, "sha256": observed})
        if observed != item["sha256"]:
            failed.append(gate(item["upstream_id"], path, "sha256", item["sha256"], observed))
    public_manifest_path = Path(history.get("public_manifest_path") or "/missing")
    if not public_manifest_path.is_file():
        failed.append(gate("exp7852", public_manifest_path, "sha256", "present", None))
    else:
        expected_manifest_hash = history.get("public_manifest_sha256")
        observed_manifest_hash = sha256_file(public_manifest_path)
        if observed_manifest_hash != expected_manifest_hash:
            failed.append(
                gate(
                    "exp7852",
                    public_manifest_path,
                    "sha256",
                    expected_manifest_hash,
                    observed_manifest_hash,
                )
            )
        cached = json.loads(public_manifest_path.read_text())
        for field, expected_field in (
            ("public_path", "public_sha256"),
            ("features_path", "features_sha256"),
        ):
            path = Path(cached.get(field) or "/missing")
            expected = cached.get(expected_field)
            observed = sha256_file(path) if path.is_file() else None
            sources.append(
                {
                    "upstream_id": "exp7852",
                    "path": str(path),
                    "sha256": observed,
                    "date": "20260929",
                    "role": "cached_candidate_only",
                    "eligibility": "exposed_development",
                }
            )
            if observed != expected:
                failed.append(gate("exp7852", path, "sha256", expected, observed))
    for program in ("python", "pytest", "coverage", "ruff", "mypy"):
        path = ROOT / ".venv/bin" / program
        if not path.is_file():
            failed.append(gate("exp7866", path, "executable", "present", None))
    manifest_path = Path(upstream.get("source_view_manifest_path") or "/missing")
    if not manifest_path.is_file():
        failed.append(gate("exp7810", manifest_path, "sha256", "present", None))
        return None, sources, failed, license_receipt
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema") != "carnot.exp7768.source_view_manifest.v1":
        failed.append(
            gate(
                "exp7810",
                manifest_path,
                "schema",
                "carnot.exp7768.source_view_manifest.v1",
                manifest.get("schema"),
            )
        )
    if manifest.get("role_counts") != boundary.ROLES:
        failed.append(
            gate(
                "exp7810", manifest_path, "role_counts", boundary.ROLES, manifest.get("role_counts")
            )
        )
    progress(start, "preflight", "checked", len(sources))
    return (manifest if not failed else None), sources, failed, license_receipt


def base(
    start: float, sources: list[dict[str, Any]], failed: list[dict[str, Any]]
) -> dict[str, Any]:
    """Keep every required field present even when external evidence blocks work."""
    receipt = build_current_work_receipt(
        run_id="exp7866-current",
        owner_pid=os.getpid(),
        events=[],
        inference_substrate="verifier_ensemble_against_cached_candidates",
        inference_substrate_details={"model_calls": 0, "tokens": 0, "model_file_hashes": []},
        inference_substrate_class="no_model_load",
        execution_venue="host",
        started_monotonic_ns=int(start * 1_000_000_000),
        ended_monotonic_ns=time.monotonic_ns(),
    )
    return {
        "schema": "carnot.exp7866.source_boundary.v1",
        "experiment_id": 7866,
        "task_id": "exp7866-v683-source-boundary",
        "milestone": "2026.09.683",
        "run_date": "20260929",
        "honest_verdict": "complete_blocked_source_evidence"
        if failed
        else "partial_pending_qualification",
        "verdict_class": "blocked" if failed else "partial",
        "flagged_adversarial": False,
        "gate_check_summary": failed,
        "rows": [],
        "sample_size_budget": {
            key: (640 if key == "intended" else 0)
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
        "duration_s": 0,
        "phase_spans": [],
        "random_seed": SEED,
        "reproducibility_checksum": None,
        "source_artifact_hashes": sources,
        "preconditions_checked": {"failed": failed, "source_count": len(sources)},
        "validation_receipts": [],
        "validation_command_manifest_path": None,
        "observed_child_commands": [],
        "repository_health": {},
        "verifier_is_oracle": True,
        "claim_scope": "exposed_development_boundary_only; no fresh generalization or benefit",
        "field_principles": {
            "identity": "Current measurements cannot reuse historical aliases.",
            "rows": "Primitive family outcomes permit independent recounting.",
            "sample_size_budget": "Views and seeds do not add families.",
            "acceptance_gate_results": "Data readiness does not imply decision benefit.",
            "validation_receipts": "Failed inherited checks remain visible.",
            "source_boundary_ready_score": "Custody, imports and required children all gate readiness.",
        },
        "inference_substrate": "verifier_ensemble_against_cached_candidates",
        "inference_substrate_class": "blocked_no_run" if failed else "no_model_load",
        "planned_inference_substrate_class": "no_model_load",
        "execution_venue": "host",
        "MODEL_SPECS": [],
        "model_specs": [],
        "target_model": "none (no pretrained model)",
        "model_invocation_counts": receipt["invocation_counts"],
        "current_work_receipt": receipt,
        "trained_head_specs": [],
        "source_boundary_ready_score": 0,
        "cohort_manifest_path": None,
        "cohort_manifest_sha256": None,
        "role_counts": boundary.ROLES,
        "source_license_receipts": [],
        "resolved_imports": {},
    }


def freeze(private: Path) -> dict[str, Any]:
    """Name exact affected files, argv, classes and deadlines before measuring."""
    python = str(ROOT / ".venv/bin/python")
    pytest = str(ROOT / ".venv/bin/pytest")
    coverage = str(ROOT / ".venv/bin/coverage")
    ruff = str(ROOT / ".venv/bin/ruff")
    mypy = str(ROOT / ".venv/bin/mypy")
    script = "scripts/experiments/experiment_7866_v683_source_boundary.py"
    module = "python/carnot/reporting/source_boundary_7866.py"
    tests = [
        "tests/python/test_source_boundary_7866.py",
        "tests/python/test_source_boundary_7852.py",
        "tests/python/test_source_projection_7838.py",
    ]
    common = ["-n", "0", "-o", "addopts=", "--no-cov"]
    fixture = private / "fixture with spaces" / "public.jsonl"
    feature = private / "fixture with spaces" / "features.jsonl"
    include = "*/source_boundary_7866.py,*/experiment_7866_v683_source_boundary.py"

    def item(
        name: str, argv: list[str], timeout: int = 180, classification: str = "required"
    ) -> dict[str, Any]:
        return {"name": name, "argv": argv, "timeout_s": timeout, "classification": classification}

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
            300,
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
            300,
        ),
        item(
            "coverage_cli",
            [
                coverage,
                "run",
                f"--data-file={private / 'cli.coverage'}",
                f"--include={include}",
                script,
                "--fixture-public",
                str(fixture),
                "--fixture-output",
                str(feature),
            ],
        ),
        item(
            "coverage_combine",
            [
                coverage,
                "combine",
                f"--data-file={private / 'combined.coverage'}",
                str(private / "unit.coverage"),
                str(private / "cli.coverage"),
            ],
        ),
        item(
            "coverage_report",
            [
                coverage,
                "report",
                f"--data-file={private / 'combined.coverage'}",
                f"--include={include}",
                "--show-missing",
                "--fail-under=100",
            ],
        ),
        item("ruff_check", [ruff, "check", module, script, *tests]),
        item("ruff_format", [ruff, "format", "--check", module, script, *tests]),
        item("mypy", [mypy, "--strict", module, script], 180),
        item("scoped_spec", [python, "scripts/check_spec_coverage.py", *tests]),
        item(
            "cli_e2e",
            [
                python,
                "-u",
                script,
                "--fixture-public",
                str(fixture),
                "--fixture-output",
                str(feature),
            ],
        ),
        item(
            "cold_replay",
            [python, "-m", "carnot.verify.source_projection", "replay", str(fixture), str(feature)],
        ),
        item("full_pytest", [pytest, "tests/python", "-q"], 600),
    ]
    closure = [
        module,
        script,
        "python/carnot/verify/source_projection.py",
        "python/carnot/verify/source_alignment.py",
        "python/carnot/reporting/current_work_receipt.py",
        "python/carnot/reporting/experiment_7303_validation_scope.py",
        "scripts/experiments/experiment_7852_v682_source_boundary.py",
        *tests,
    ]
    historical = json.loads(
        (
            ROOT
            / "results/raw/experiment_7852_v682_source_boundary/validation_command_manifest.json"
        ).read_text()
    )
    return {
        "schema": "carnot.exp7866.validation.v1",
        "task_id": "exp7866-v683-source-boundary",
        "affected_source_closure": [
            {"path": path, "sha256": sha256_file(ROOT / path)} for path in closure
        ],
        "affected_tests": tests,
        "commands": commands,
        "historical_affected_source_closure": historical["affected_source_closure"],
        "historical_affected_tests": historical["affected_tests"],
        "historical_aborted_command": next(
            item for item in historical["commands"] if item["name"] == "changed_coverage"
        ),
        "inapplicable_e2e": [
            {
                "ids": "E2E-001-through-014",
                "reason": "No model, hardware, ARC agent or service changed.",
            }
        ],
    }


def child(spec: dict[str, Any], index: int, private: Path, start: float) -> dict[str, Any]:
    """Supervise one child and seal its log only after process and handles close."""
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
    original = Path(receipt["log_path"])
    digest = sha256_file(original)
    sealed = RAW / "validation_logs" / f"{index:02d}_{spec['name']}_{digest[7:]}.log"
    sealed.parent.mkdir(parents=True, exist_ok=True)
    if sealed.exists() and sha256_file(sealed) != digest:
        raise ValueError("sealed_log_collision")
    if not sealed.exists():
        shutil.copyfile(original, sealed)
    receipt["log_path"], receipt["log_sha256"] = str(sealed), digest
    receipt["classification"] = spec["classification"]
    progress(start, spec["name"], "after_subprocess", index + 1)
    return receipt


def fixture_cli(public: Path, output: Path) -> None:
    """The same public-only route serves private success and failure fixtures."""
    source_projection.extract_file(public, output)
    source_projection.replay_file(public, output)


def terminal(artifact: dict[str, Any], start: float, verdict: str) -> dict[str, Any]:
    """Publish one closed terminal state with measured time and honest gates."""
    artifact["duration_s"] = time.monotonic() - start
    artifact["verdict_class"] = verdict
    artifact["honest_verdict"] = {
        "blocked": "complete_blocked_source_evidence",
        "disqualified": "complete_disqualified_required_checks",
        "circular_positive": "complete_circular_positive_source_boundary_readiness",
    }[verdict]
    artifact["source_boundary_ready_score"] = int(verdict == "circular_positive")
    artifact["acceptance_gate_results"]["validity"] = verdict == "circular_positive"
    artifact["acceptance_gate_results"]["readiness"] = int(verdict == "circular_positive")
    atomic_json(OUTPUT, artifact)
    progress(start, "terminal", "published", len(artifact["rows"]))
    return artifact


def run_experiment(date: str) -> dict[str, Any]:
    """Requalify the original corpus, then reduce every frozen child exit."""
    start = time.monotonic()
    progress(start, "start", "begin")
    if date != "20260929":
        raise ValueError("run_date_mismatch")
    manifest, sources, failed, license_receipt = preflight(start)
    artifact = base(start, sources, failed)
    artifact["phase_spans"].append({"phase": "preflight", "duration_s": time.monotonic() - start})
    history = json.loads(HISTORY.read_text()) if HISTORY.is_file() else {}
    artifact["repository_health"] = {
        "historical_exp7852_verdict": history.get("honest_verdict"),
        "historical_failed_required": [
            {
                "name": receipt["name"],
                "exit_code": receipt["exit_code"],
                "log_path": receipt["log_path"],
                "log_sha256": receipt["log_sha256"],
            }
            for receipt in history.get("validation_receipts", [])
            if not receipt.get("passed")
        ],
        "historical_full_suite": history.get("repository_health", {}).get("repository_health_180s"),
        "v683_first_attempt": {
            "path": str(RAW / "attempts/attempt-6c115e8469574640.json"),
            "sha256": "sha256:6c115e84695746403c9d28ce9cccb8513fa28cbccceb6d6e539f5c03d0d45405",
            "reason": "Coverage source selection collected no data; the required broad suite timed out after failures.",
        },
    }
    if manifest is None:
        return terminal(artifact, start, "blocked")
    artifact["source_license_receipts"] = [
        {
            "path": str(LICENSE),
            "sha256": sha256_file(LICENSE),
            "license": license_receipt["license"],
            "url": license_receipt["repository"],
            "revision": license_receipt["commit"],
            "label_authority": license_receipt["label_authority"],
        }
    ]
    config = canonical_hash({"sources": sources, "code": sha256_file(Path(__file__)), "seed": SEED})
    private = Path("/tmp") / f"exp7866-{config[7:23]}"
    private.mkdir(parents=True, exist_ok=True)
    scope = freeze(private)
    scope_path = RAW / "validation_command_manifest.json"
    atomic_json(scope_path, scope)
    artifact["validation_command_manifest_path"] = str(scope_path)
    artifact["validation_command_manifest_sha256"] = sha256_file(scope_path)
    artifact["reproducibility_checksum"] = canonical_hash(
        {
            "sources": sources,
            "closure": scope["affected_source_closure"],
            "configuration": sha256_file(scope_path),
            "seed": SEED,
        }
    )
    artifact["preconditions_checked"]["manifest_schema"] = manifest["schema"]
    artifact["preconditions_checked"]["role_counts"] = manifest["role_counts"]
    artifact["preconditions_checked"]["historical_disqualification_preserved"] = True
    progress(start, "custody", "begin")
    try:
        cohort, public = boundary.acquire(manifest, lambda n: progress(start, "custody", "rows", n))
        public_path = private / "public.jsonl"
        if public_path.exists() and source_projection.read_jsonl(public_path) != public:
            raise ValueError("public_checkpoint_drift")
        if not public_path.exists():
            source_projection.write_jsonl(public_path, public)
        old_public_manifest = json.loads(Path(history["public_manifest_path"]).read_text())
        cached_public = Path(old_public_manifest["public_path"])
        features_path = Path(old_public_manifest["features_path"])
        if sha256_file(public_path) != old_public_manifest["public_sha256"] or sha256_file(
            cached_public
        ) != sha256_file(public_path):
            raise ValueError("cached_public_identity_drift")
        if sha256_file(features_path) != old_public_manifest["features_sha256"]:
            raise ValueError("cached_feature_identity_drift")
        progress(start, "projection", "before_cached_replay", 0)
        source_projection.replay_file(public_path, features_path)
        progress(start, "projection", "after_cached_replay", 640)
        features = source_projection.read_jsonl(features_path)
        if len(features) != 640:
            raise ValueError("cached_feature_roster_drift")
        for index, (row, feature) in enumerate(zip(cohort, features, strict=True), 1):
            if row["family_id"] != feature["family_id"]:
                raise ValueError(f"feature_join_drift:{index}")
            row["feature_hash"] = feature["feature_hash"]
            row["status"] = "excluded" if feature["abstention"] else "completed"
            row["exclusion_reasons"] = [feature["abstention"]] if feature["abstention"] else []
            row["view_a_windows"] = feature["view_a_windows"]
            row["view_b_windows"] = feature["view_b_windows"]
            if index % 32 == 0:
                progress(start, "reduction", "rows", index)
        budget = boundary.reduce_budget(cohort, 640)
        if budget["independent"] != 640:
            raise ValueError("independent_family_count_drift")
        role_counts = Counter(row["role"] for row in cohort)
        if dict(role_counts) != boundary.ROLES:
            raise ValueError("role_count_drift")
        cohort_hash = canonical_hash(cohort)
        cohort_path = RAW / f"cohort-{cohort_hash[7:]}.json"
        if cohort_path.exists() and json.loads(cohort_path.read_text()).get("rows") != cohort:
            raise ValueError("immutable_cohort_collision")
        if not cohort_path.exists():
            atomic_json(
                cohort_path, {"rows": cohort, "source_license": artifact["source_license_receipts"]}
            )
        artifact["cohort_manifest_path"] = str(cohort_path)
        artifact["cohort_manifest_sha256"] = sha256_file(cohort_path)
        artifact["rows"] = [
            {
                key: value
                for key, value in row.items()
                if key not in {"complete_source", "complete_response", "label_provenance"}
            }
            for row in cohort
        ]
        artifact["sample_size_budget"] = budget
        artifact["resolved_imports"] = boundary.qualified_imports()
        if not boundary.imports_valid(artifact["resolved_imports"]):
            raise ValueError("import_resolution_drift")
        artifact["cached_candidate_receipt"] = {
            "path": str(features_path),
            "sha256": sha256_file(features_path),
            "authority": "cached_projection_only",
        }
        artifact["phase_spans"].append(
            {"phase": "custody_projection", "duration_s": time.monotonic() - start}
        )
    except (OSError, KeyError, ValueError, TypeError) as exc:
        artifact["gate_check_summary"].append(
            gate("exp7866", scope_path, "owned_conversion", "valid", str(exc))
        )
        return terminal(artifact, start, "disqualified")
    fixture = private / "fixture with spaces" / "public.jsonl"
    fixture.parent.mkdir(parents=True, exist_ok=True)
    source_projection.write_jsonl(
        fixture,
        [{"family_id": "fixture", "source_bytes": b"A. B.".hex(), "answer_bytes": b"A.".hex()}],
    )
    progress(start, "validation", "begin")
    for index, spec in enumerate(scope["commands"]):
        if spec["name"] == "full_pytest":
            prior = json.loads((RAW / "attempts/attempt-6c115e8469574640.json").read_text())
            receipt = next(
                item for item in prior["validation_receipts"] if item["name"] == "full_pytest"
            )
            receipt = {
                **receipt,
                "reused_from": str(RAW / "attempts/attempt-6c115e8469574640.json"),
            }
            artifact["validation_receipts"].append(receipt)
            artifact["observed_child_commands"].append(receipt)
            artifact["gate_check_summary"].append(
                gate(
                    "exp7866",
                    Path(receipt["log_path"]),
                    "full_pytest",
                    {"exit_code": 0, "passed": True},
                    {"exit_code": receipt["exit_code"], "passed": False},
                )
            )
            progress(start, "full_pytest", "prior_failed_receipt_reused", index + 1)
            continue
        if spec["name"] == "coverage_combine" and any(
            not receipt["passed"]
            for receipt in artifact["validation_receipts"]
            if receipt["name"] in {"coverage_unit", "coverage_cli"}
        ):
            artifact["gate_check_summary"].append(
                gate(
                    "exp7866",
                    scope_path,
                    "coverage_combine",
                    "both completed child files",
                    "aborted child",
                )
            )
            continue
        receipt = child(spec, index, private, start)
        artifact["validation_receipts"].append(receipt)
        artifact["observed_child_commands"].append(receipt)
        if not receipt["passed"]:
            artifact["gate_check_summary"].append(
                gate(
                    "exp7866",
                    Path(receipt["log_path"]),
                    spec["name"],
                    {"exit_code": 0, "passed": True},
                    {"exit_code": receipt["exit_code"], "passed": False},
                )
            )
        if spec["name"] == "worktree_imports" and not boundary.imports_valid(
            receipt.get("resolved_imports", {})
        ):
            artifact["gate_check_summary"].append(
                gate(
                    "exp7866",
                    Path(receipt["log_path"]),
                    "resolved_imports",
                    boundary.qualified_imports(),
                    receipt.get("resolved_imports"),
                )
            )
        progress(start, "validation", "completed", index + 1)
    artifact["flagged_adversarial"] = False
    verdict = "disqualified" if artifact["gate_check_summary"] else "circular_positive"
    artifact["phase_spans"].append(
        {
            "phase": "validation",
            "duration_s": time.monotonic() - start - artifact["phase_spans"][0]["duration_s"],
        }
    )
    artifact["duration_s"] = time.monotonic() - start
    artifact["verdict_class"] = verdict
    artifact["honest_verdict"] = (
        "complete_disqualified_required_checks"
        if verdict == "disqualified"
        else "complete_circular_positive_source_boundary_readiness"
    )
    artifact["source_boundary_ready_score"] = int(verdict == "circular_positive")
    artifact["acceptance_gate_results"]["validity"] = verdict == "circular_positive"
    artifact["acceptance_gate_results"]["readiness"] = int(verdict == "circular_positive")
    candidate = private / "terminal_candidate.json"
    atomic_json(candidate, artifact)
    reports = []
    for offset, name, argv in (
        (
            0,
            "adversarial_verify",
            [
                str(ROOT / ".venv/bin/python"),
                "scripts/adversarial_verify.py",
                "--json",
                str(candidate),
            ],
        ),
        (
            1,
            "strict_rows",
            [
                str(ROOT / ".venv/bin/python"),
                "scripts/verdict_row_consistency_lint.py",
                "--strict",
                str(candidate),
            ],
        ),
    ):
        reports.append(
            child(
                {
                    "name": name,
                    "argv": argv,
                    "timeout_s": 180,
                    "classification": "required_terminal",
                },
                len(scope["commands"]) + offset,
                private,
                start,
            )
        )
    artifact["flagged_adversarial"] = not reports[0]["passed"]
    if not all(report["passed"] for report in reports):
        artifact["gate_check_summary"].extend(
            gate(
                "exp7866",
                Path(report["log_path"]),
                report["name"],
                "exit_code=0",
                report["exit_code"],
            )
            for report in reports
            if not report["passed"]
        )
        verdict = "disqualified"
    artifact["terminal_verification_receipts"] = reports
    artifact["terminal_validation_manifest_path"] = str(RAW / "terminal_validation_receipts.json")
    artifact["duration_s"] = time.monotonic() - start
    artifact["verdict_class"] = verdict
    artifact["honest_verdict"] = (
        "complete_disqualified_required_checks"
        if verdict == "disqualified"
        else "complete_circular_positive_source_boundary_readiness"
    )
    artifact["source_boundary_ready_score"] = int(verdict == "circular_positive")
    artifact["acceptance_gate_results"]["validity"] = verdict == "circular_positive"
    artifact["acceptance_gate_results"]["readiness"] = int(verdict == "circular_positive")
    atomic_json(candidate, artifact)
    final_reports = []
    for offset, name, argv in (
        (
            0,
            "adversarial_verify",
            [
                str(ROOT / ".venv/bin/python"),
                "scripts/adversarial_verify.py",
                "--json",
                str(candidate),
            ],
        ),
        (
            1,
            "strict_rows",
            [
                str(ROOT / ".venv/bin/python"),
                "scripts/verdict_row_consistency_lint.py",
                "--strict",
                str(candidate),
            ],
        ),
    ):
        final_reports.append(
            child(
                {
                    "name": f"final_{name}",
                    "argv": argv,
                    "timeout_s": 180,
                    "classification": "required_terminal",
                },
                len(scope["commands"]) + 2 + offset,
                private,
                start,
            )
        )
    atomic_json(
        RAW / "terminal_validation_receipts.json",
        {
            "candidate_path": str(candidate),
            "candidate_sha256": sha256_file(candidate),
            "reports": final_reports,
        },
    )
    if not all(report["passed"] for report in final_reports):
        raise ValueError("final_candidate_verification_failed")
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    temporary = OUTPUT.with_name(f".{OUTPUT.name}.tmp-{os.getpid()}")
    shutil.copyfile(candidate, temporary)
    os.replace(temporary, OUTPUT)
    progress(start, "terminal", "published", len(artifact["rows"]))
    return artifact


def main(argv: list[str] | None = None) -> int:
    """Run a dated qualification or the inherited public-only fixture route."""
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
