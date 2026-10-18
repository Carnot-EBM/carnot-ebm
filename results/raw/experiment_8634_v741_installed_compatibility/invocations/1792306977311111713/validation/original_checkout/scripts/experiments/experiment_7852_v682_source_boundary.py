"""Qualify archived source identity and current public bytes (REQ-REPORT-7852)."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
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
from carnot.reporting import source_identity_7852 as identity
from carnot.verify import evidence_views, source_projection

if __package__:
    from . import experiment_7838_v681_source_boundary as previous
else:
    import experiment_7838_v681_source_boundary as previous

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "results/experiment_7852_v682_source_boundary.json"
RAW = ROOT / "results/raw/experiment_7852_v682_source_boundary"
ROLES = previous.ROLES
SEED = 68252


def progress(start: float, phase: str, event: str, units: int = 0) -> None:
    """Expose every phase edge and long-loop advance to the operator."""
    print(
        f"[exp7852] phase={phase} event={event} elapsed_s={time.monotonic() - start:.3f} completed={units}",
        flush=True,
    )


def gate(path: Path, field: str, expected: Any, observed: Any) -> dict[str, Any]:
    """Keep an absent upstream field distinct from a wrong value."""
    return {
        "upstream_id": "exp7810",
        "path": str(path),
        "hash": sha256_file(path) if path.is_file() else None,
        "artifact_field": field,
        "op": "==",
        "expected": expected,
        "observed": observed,
    }


def preflight(
    start: float,
) -> tuple[dict[str, Any] | None, list[dict[str, Any]], list[dict[str, Any]]]:
    """Check exact legacy bytes, all shards, and closed historical obligations."""
    source = identity.LEGACY_PATH
    sources: list[dict[str, Any]] = []
    failed: list[dict[str, Any]] = []
    digest = sha256_file(source) if source.is_file() else None
    if digest != identity.LEGACY_HASH:
        failed.append(gate(source, "sha256", identity.LEGACY_HASH, digest))
        return None, sources, failed
    archived = json.loads(source.read_text())
    try:
        identity.resolve(source, digest, archived)
    except ValueError as exc:
        field = str(exc).split(":", 2)[1]
        expected = {
            "experiment_id": identity.LEGACY_SLUG,
            "milestone": identity.LEGACY_MILESTONE,
            "run_date": "20260928",
            "task_id": identity.LEGACY_SLUG,
        }[field]
        failed.append(gate(source, field, expected, archived.get(field)))
    manifest, checked, prior_failures = previous.preflight(start)
    sources.extend(checked)
    failed.extend(prior_failures)
    for field, expected in (
        ("schema", "carnot.exp7810.source_view_qualification.v1"),
        ("milestone", identity.LEGACY_MILESTONE),
    ):
        if archived.get(field) != expected:
            failed.append(gate(source, field, expected, archived.get(field)))
    for receipt in archived.get("validation_receipts", []):
        if (
            receipt.get("classification") == "diagnostic"
            or receipt.get("name") == "repository_health"
        ):
            continue
        path = Path(receipt.get("log_path") or "missing")
        expected_hash = receipt.get("log_sha256")
        observed_hash = sha256_file(path) if path.is_file() else None
        if observed_hash != expected_hash or not receipt.get("passed"):
            failed.append(
                gate(
                    path,
                    "validation_receipt",
                    {"hash": expected_hash, "passed": True},
                    {"hash": observed_hash, "passed": receipt.get("passed")},
                )
            )
    if manifest is not None and manifest.get("schema") != "carnot.exp7768.source_view_manifest.v1":
        failed.append(
            gate(
                Path(archived["source_view_manifest_path"]),
                "schema",
                "carnot.exp7768.source_view_manifest.v1",
                manifest.get("schema"),
            )
        )
    progress(start, "preflight", "checked", len(sources))
    return (manifest if not failed else None), sources, failed


def seal(receipt: dict[str, Any], index: int) -> dict[str, Any]:
    """Copy a closed child log to an immutable path named by its bytes."""
    source = Path(receipt["log_path"])
    digest = sha256_file(source)
    destination = RAW / "validation_logs" / f"{index:02d}_{receipt['name']}_{digest[7:]}.log"
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and sha256_file(destination) != digest:
        raise ValueError("sealed_log_collision")
    if not destination.exists():
        shutil.copyfile(source, destination)
    receipt["log_path"] = str(destination)
    receipt["log_sha256"] = digest
    return receipt


def child(spec: dict[str, Any], index: int, private: Path, start: float) -> dict[str, Any]:
    """Supervise one owned child and retain its exit and sealed output."""
    progress(start, spec["name"], "before_subprocess", index)
    command = CommandSpec(
        spec["name"], tuple(spec["argv"]), spec["classification"], float(spec["timeout_s"])
    )
    result = run_commands(
        ROOT,
        [command],
        log_dir=private / "child_logs" / f"{index:02d}",
        extra_env={"CARNOT_FORCE_LIVE": "1", "JAX_PLATFORMS": "cpu"},
        heartbeat_s=30,
    )[0]
    progress(start, spec["name"], "after_subprocess", index + 1)
    return seal(result, index)


def frozen_commands(private: Path) -> dict[str, Any]:
    """Name each required child before reading natural labels or metrics."""
    python = str(ROOT / ".venv/bin/python")
    pytest = str(ROOT / ".venv/bin/pytest")
    coverage = str(ROOT / ".venv/bin/coverage")
    ruff = str(ROOT / ".venv/bin/ruff")
    mypy = str(ROOT / ".venv/bin/mypy")
    test = "tests/python/test_source_boundary_7852.py"
    module = "python/carnot/reporting/source_identity_7852.py"
    script = "scripts/experiments/experiment_7852_v682_source_boundary.py"
    fixture = private / "fixture_public.jsonl"
    feature = private / "fixture_features.jsonl"
    public = private / "public.jsonl"
    output = private / "features.jsonl"
    candidate = private / "candidate.json"
    common = ["-n", "0", "-o", "addopts=", "--no-cov"]

    def item(
        name: str, argv: list[str], timeout: int = 90, classification: str = "required"
    ) -> dict[str, Any]:
        return {"name": name, "argv": argv, "timeout_s": timeout, "classification": classification}

    commands = [
        item(
            "worktree_imports",
            [
                python,
                "-c",
                "import json,pathlib,carnot.reporting.source_identity_7852 as i,carnot.verify.source_projection as p; print(json.dumps({'resolved_imports':{'carnot.reporting.source_identity_7852':str(pathlib.Path(i.__file__).resolve()),'carnot.verify.source_projection':str(pathlib.Path(p.__file__).resolve())}}))",
            ],
        ),
        item(
            "affected_pytest",
            [
                pytest,
                *common,
                f"--basetemp={private / 'pytest'}",
                test,
                "tests/python/test_source_projection_7838.py",
                "-q",
            ],
            150,
        ),
        item(
            "changed_coverage",
            [
                coverage,
                "run",
                f"--data-file={private / 'unit.coverage'}",
                "-m",
                "pytest",
                *common,
                f"--basetemp={private / 'coverage_pytest'}",
                test,
                "-q",
            ],
            150,
        ),
        item(
            "changed_coverage_report",
            [
                coverage,
                "report",
                f"--data-file={private / 'unit.coverage'}",
                "--include=*/source_identity_7852.py",
                "--fail-under=100",
            ],
            30,
        ),
        item("ruff_check", [ruff, "check", module, script, test]),
        item("ruff_format", [ruff, "format", "--check", module, script, test]),
        item("mypy", [mypy, "--strict", module, script], 120),
        item("scoped_spec", [python, "scripts/check_spec_coverage.py", test], 90),
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
            [python, "-m", "carnot.verify.source_projection", "replay", str(public), str(output)],
            180,
        ),
        item(
            "adversarial_verify",
            [python, "scripts/adversarial_verify.py", "--json", str(candidate)],
        ),
        item(
            "strict_rows",
            [python, "scripts/verdict_row_consistency_lint.py", "--strict", str(candidate)],
        ),
    ]
    closure = [
        module,
        script,
        "python/carnot/verify/source_projection.py",
        "python/carnot/verify/evidence_views.py",
        "python/carnot/verify/source_alignment.py",
        "python/carnot/reporting/current_work_receipt.py",
        "python/carnot/reporting/experiment_7303_validation_scope.py",
        "scripts/experiments/experiment_7838_v681_source_boundary.py",
    ]
    return {
        "schema": "carnot.exp7852.validation.v1",
        "task_id": "exp7852-source-boundary",
        "affected_tests": [test, "tests/python/test_source_projection_7838.py"],
        "affected_source_closure": [
            {"path": path, "sha256": sha256_file(ROOT / path)} for path in closure
        ],
        "commands": commands,
        "repository_health": item(
            "repository_health_180s",
            [pytest, *common, f"--basetemp={private / 'repository_health'}", "tests/python", "-q"],
            180,
            "diagnostic",
        ),
        "inapplicable_e2e": [
            {
                "ids": "E2E-001-through-014",
                "reason": "No hardware, ARC, or model service changed; the task-specific private CLI and cold replay cover this CPU boundary.",
            }
        ],
    }


def artifact_base(
    start: float, sources: list[dict[str, Any]], failed: list[dict[str, Any]]
) -> dict[str, Any]:
    """Give every terminal path the same explicit source and validation fields."""
    receipt = build_current_work_receipt(
        run_id="exp7852-current",
        owner_pid=os.getpid(),
        events=[],
        inference_substrate="cpu_no_pretrained_model",
        inference_substrate_details={"model_calls": 0, "tokens": 0, "model_file_hashes": []},
        inference_substrate_class="no_model_load",
        execution_venue="host",
        started_monotonic_ns=int(start * 1_000_000_000),
        ended_monotonic_ns=time.monotonic_ns(),
    )
    return {
        "schema": "carnot.exp7852.source_boundary.v1",
        "experiment_id": 7852,
        "task_id": "exp7852-source-boundary",
        "milestone": "2026.09.682",
        "run_date": "20260929",
        "honest_verdict": "complete_blocked_required_source_evidence"
        if failed
        else "complete_disqualified_required_checks",
        "verdict_class": "blocked" if failed else "disqualified",
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
        "duration_s": time.monotonic() - start,
        "phase_spans": [],
        "random_seed": SEED,
        "reproducibility_checksum": None,
        "source_artifact_hashes": sources,
        "preconditions_checked": {"failed": failed, "source_count": len(sources)},
        "validation_receipts": [],
        "validation_command_manifest_path": None,
        "observed_child_commands": [],
        "repository_health": {
            "historical_exp7824": {
                "path": str(ROOT / "results/experiment_7824_v680_source_feature_isolation.json"),
                "required_full_suite": "failed",
                "coverage_percent": 27,
            }
        },
        "verifier_is_oracle": True,
        "claim_scope": "exposed_development_boundary_only",
        "field_principles": {
            "identity": "Current producers use numbers; one archived alias is hash bound.",
            "source_artifact_hashes": "Source custody limits authority to exact bytes.",
            "rows": "Primitive family outcomes allow cold reduction.",
            "sample_size_budget": "Source views are repeated measures.",
            "acceptance_gate_results": "Source readiness does not imply prediction benefit.",
            "validation_receipts": "A new wrapper cannot erase failed checks.",
            "source_boundary_ready_score": "Training opens only after source, separation, and checks pass.",
        },
        "inference_substrate": "cpu_no_pretrained_model",
        "inference_substrate_class": "no_model_load",
        "planned_inference_substrate_class": "no_model_load",
        "MODEL_SPECS": [],
        "model_specs": [],
        "model_invocation_counts": receipt["invocation_counts"],
        "current_work_receipt": receipt,
        "source_boundary_ready_score": 0,
        "public_manifest_path": None,
        "evaluator_manifest_path": None,
        "role_hashes": {},
        "identity_compatibility_rows": [],
        "resolved_imports": {
            "identity": str(Path(identity.__file__).resolve()),
            "source_projection": str(Path(source_projection.__file__).resolve()),
        },
    }


def run_experiment(date: str) -> dict[str, Any]:
    """Qualify source custody, run the public child, then reduce frozen checks."""
    start = time.monotonic()
    progress(start, "start", "begin")
    if date != "20260929":
        raise ValueError("run_date_mismatch")
    manifest, sources, failed = preflight(start)
    artifact = artifact_base(start, sources, failed)
    previous_result = json.loads(OUTPUT.read_text()) if OUTPUT.is_file() else {}
    artifact["prior_required_failures"] = previous_result.get("prior_required_failures", []) + [
        {
            "name": item["name"],
            "exit_code": item["exit_code"],
            "log_path": item["log_path"],
            "log_sha256": item["log_sha256"],
        }
        for item in previous_result.get("validation_receipts", [])
        if not item["passed"]
    ]
    artifact["phase_spans"].append({"phase": "preflight", "duration_s": time.monotonic() - start})
    archived = (
        json.loads(identity.LEGACY_PATH.read_text()) if identity.LEGACY_PATH.is_file() else {}
    )
    alias = {
        "path": str(identity.LEGACY_PATH),
        "hash": identity.LEGACY_HASH,
        "observed": archived.get("experiment_id"),
        "normalized_experiment_id": 7810,
        "accepted": manifest is not None,
    }
    artifact["identity_compatibility_rows"] = [alias] + [
        {"mutation": name, "accepted": False}
        for name in ("suffix", "milestone", "path", "hash", "conflicting_task_id")
    ]
    if manifest is None:
        artifact["duration_s"] = time.monotonic() - start
        atomic_json(OUTPUT, artifact)
        progress(start, "terminal", "blocked")
        return artifact
    config_hash = canonical_hash(
        {"source": identity.LEGACY_HASH, "code": sha256_file(Path(__file__)), "seed": SEED}
    )
    private = Path("/tmp") / f"exp7852-{config_hash[7:23]}"
    private.mkdir(parents=True, exist_ok=True)
    scope = frozen_commands(private)
    scope_path = RAW / "validation_command_manifest.json"
    atomic_json(scope_path, scope)
    artifact["validation_command_manifest_path"] = str(scope_path)
    artifact["validation_command_manifest_sha256"] = sha256_file(scope_path)
    artifact["reproducibility_checksum"] = canonical_hash(
        {
            "closure": scope["affected_source_closure"],
            "inputs": sources,
            "configuration": sha256_file(scope_path),
            "seed": SEED,
        }
    )
    source_projection.write_jsonl(
        private / "fixture_public.jsonl",
        [{"family_id": "fixture", "source_bytes": b"A. B.".hex(), "answer_bytes": b"A.".hex()}],
    )
    progress(start, "custody", "begin")
    public, sidecar, evidence = previous.custody(manifest, start)
    development = json.loads(Path(manifest["development_manifest_path"]).read_text())
    if development.get("schema") != "carnot.exp7727.development_manifest.v1":
        raise ValueError("development_schema_drift")
    groups: dict[str, set[str]] = defaultdict(set)
    for row in evidence:
        groups[row["source_sha256"]].add(row["role"])
    if any(len(roles) != 1 for roles in groups.values()):
        raise ValueError("duplicate_source_role_overlap")
    artifact["duplicate_source_groups"] = {"groups": len(groups), "cross_role_overlap": 0}
    public_path = private / "public.jsonl"
    sidecar_path = private / "evaluator.jsonl"
    features_path = private / "features.jsonl"
    for path, records in ((public_path, public), (sidecar_path, sidecar)):
        if path.exists() and source_projection.read_jsonl(path) != records:
            raise ValueError(f"checkpoint_input_changed:{path}")
        if not path.exists():
            source_projection.write_jsonl(path, records)
    if features_path.exists():
        source_projection.replay_file(public_path, features_path)
        progress(start, "projection", "checkpoint_reused", 640)
    else:
        extraction = {
            "name": "public_extraction",
            "argv": [
                str(ROOT / ".venv/bin/python"),
                "-u",
                "-m",
                "carnot.verify.source_projection",
                "extract",
                str(public_path),
                str(features_path),
            ],
            "classification": "owned_compute",
            "timeout_s": 600,
        }
        receipt = child(extraction, 98, private, start)
        artifact["observed_child_commands"].append(receipt)
        if not receipt["passed"]:
            raise ValueError("public_extraction_failed")
    source_projection.replay_file(public_path, features_path)
    features = source_projection.read_jsonl(features_path)
    source_projection.validate_features(public, features, [row["family_id"] for row in evidence])
    valid: Counter[str] = Counter()
    excluded = 0
    metadata_mutations = 0
    for index, (row, private_row, feature) in enumerate(
        zip(evidence, sidecar, features, strict=True), 1
    ):
        reason = feature["abstention"]
        if reason is None:
            valid[row["role"]] += 1
            original = public[index - 1]
            for field, value in (
                ("label", 999),
                ("role", "mutated"),
                ("confidence", -1),
                ("annotations", [{"start": 999}]),
                ("generator_identity", "mutated"),
            ):
                if source_projection.public_row({**original, field: value}) != original:
                    raise ValueError(f"metadata_leak:{field}:{index}")
                metadata_mutations += 1
            changed_join = source_projection.extract_row(
                {**original, "family_id": f"opaque-{index}"}
            )
            if changed_join["feature_hash"] != feature[
                "feature_hash"
            ] or source_projection.canonical_bytes(
                changed_join["views"]
            ) != source_projection.canonical_bytes(feature["views"]):
                raise ValueError(f"opaque_join_feature_leak:{index}")
            metadata_mutations += 1
            for arm in ("a", "b"):
                view = evidence_views.deserialize_pair(feature["views"])[arm]
                prior = evidence_views.location_prior(view)
                if abs(sum(prior) - 1) > 1e-12 or len(prior) != len(view["windows"]) + 1:
                    raise ValueError("duplicate_group_weight_drift")
        else:
            excluded += 1
        row.update(
            status="completed" if reason is None else "excluded",
            eligible=reason is None,
            abstention=reason,
            arm="public_projection",
            seed=SEED,
            source_family=row["source_sha256"],
            feature_hash=feature["feature_hash"],
            view_a_windows=feature["view_a_windows"],
            view_b_windows=feature["view_b_windows"],
        )
        if private_row["family_id"] != row["family_id"]:
            raise ValueError("evaluator_join_drift")
        if index % 32 == 0:
            progress(start, "reduction", "rows", index)
    if valid["fit"] < 128 or valid["tune"] < 32:
        artifact["gate_check_summary"].append(
            gate(
                Path(manifest["rows_path"]),
                "valid_fit_tune_groups",
                {"fit": 128, "tune": 32},
                dict(valid),
            )
        )
    artifact["rows"] = evidence
    artifact["sample_size_budget"] = {
        "intended": 640,
        "eligible": 640 - excluded,
        "started": 640,
        "completed": 640 - excluded,
        "censored": 0,
        "excluded": excluded,
        "independent": len(groups),
    }
    artifact["valid_role_groups"] = dict(valid)
    artifact["metadata_mutation_count"] = metadata_mutations
    unknown_loss = source_projection.masked_loss([0.2, -0.4], [1, -1], [1, 0])
    if (
        unknown_loss != source_projection.masked_loss([0.2, -0.4], [1, 999], [1, 0])
        or unknown_loss[1][0] == 0
    ):
        raise ValueError("unknown_label_mask_drift")
    artifact["unknown_label_mask_check"] = {
        "loss": unknown_loss[0],
        "gradient": unknown_loss[1],
        "unknown_gradient": unknown_loss[1][1],
    }
    artifact["role_hashes"] = manifest["role_hashes"]
    public_manifest = RAW / "public_manifest.json"
    evaluator_manifest = RAW / "evaluator_manifest.json"
    atomic_json(
        public_manifest,
        {
            "schema": "carnot.exp7852.public.v1",
            "field_map": ["family_id", "source_bytes", "answer_bytes"],
            "public_path": str(public_path),
            "public_sha256": sha256_file(public_path),
            "features_path": str(features_path),
            "features_sha256": sha256_file(features_path),
            "projection_manifest_sha256": sha256_file(features_path.with_suffix(".manifest.json")),
        },
    )
    atomic_json(
        evaluator_manifest,
        {
            "schema": "carnot.exp7852.evaluator.v1",
            "field_map": [
                "family_id",
                "role",
                "evaluator",
                "target",
                "generator_identity",
                "confidence",
            ],
            "path": str(sidecar_path),
            "sha256": sha256_file(sidecar_path),
            "role_hashes": manifest["role_hashes"],
        },
    )
    artifact["public_manifest_path"] = str(public_manifest)
    artifact["public_manifest_sha256"] = sha256_file(public_manifest)
    artifact["evaluator_manifest_path"] = str(evaluator_manifest)
    artifact["evaluator_manifest_sha256"] = sha256_file(evaluator_manifest)
    artifact["phase_spans"].append(
        {
            "phase": "custody_projection_reduction",
            "duration_s": time.monotonic() - start - artifact["phase_spans"][0]["duration_s"],
        }
    )
    progress(start, "validation", "begin")
    candidate = private / "candidate.json"
    for index, spec in enumerate(scope["commands"]):
        if spec["name"] in {"adversarial_verify", "strict_rows"}:
            passing = not artifact["gate_check_summary"] and all(
                item["passed"] for item in artifact["validation_receipts"]
            )
            artifact["verdict_class"] = "circular_positive" if passing else "disqualified"
            artifact["honest_verdict"] = (
                "complete_circular_positive_source_boundary_readiness"
                if passing
                else "complete_disqualified_required_checks"
            )
            artifact["acceptance_gate_results"]["validity"] = passing
            artifact["acceptance_gate_results"]["readiness"] = int(passing)
            artifact["source_boundary_ready_score"] = int(passing)
            atomic_json(candidate, artifact)
        receipt = child(spec, index, private, start)
        artifact["validation_receipts"].append(receipt)
        artifact["observed_child_commands"].append(receipt)
        if spec["name"] == "worktree_imports":
            artifact["resolved_imports"] = receipt.get(
                "resolved_imports", artifact["resolved_imports"]
            )
        progress(start, "validation", "completed", index + 1)
    health = previous_result.get("repository_health", {}).get("repository_health_180s")
    if health is None:
        health = child(scope["repository_health"], len(scope["commands"]), private, start)
    else:
        progress(start, "repository_health_180s", "prior_closed_receipt_reused", 1)
    artifact["repository_health"]["repository_health_180s"] = health
    artifact["observed_child_commands"].append(health)
    artifact["flagged_adversarial"] = any(
        item["name"] == "adversarial_verify" and not item["passed"]
        for item in artifact["validation_receipts"]
    )
    passing = not artifact["gate_check_summary"] and all(
        item["passed"] for item in artifact["validation_receipts"]
    )
    artifact["acceptance_gate_results"]["validity"] = passing
    artifact["acceptance_gate_results"]["readiness"] = int(passing)
    artifact["source_boundary_ready_score"] = int(passing)
    artifact["verdict_class"] = "circular_positive" if passing else "disqualified"
    artifact["honest_verdict"] = (
        "complete_circular_positive_source_boundary_readiness"
        if passing
        else "complete_disqualified_required_checks"
    )
    artifact["duration_s"] = time.monotonic() - start
    artifact["candidate_path"] = str(candidate)
    artifact["candidate_sha256"] = sha256_file(candidate)
    atomic_json(OUTPUT, artifact)
    progress(start, "terminal", "published", len(evidence))
    return artifact


def main(argv: list[str] | None = None) -> int:
    """Run dated natural custody or one private public-only CLI fixture."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--date")
    parser.add_argument("--fixture-public", type=Path)
    parser.add_argument("--fixture-output", type=Path)
    args = parser.parse_args(argv)
    if args.fixture_public is not None and args.fixture_output is not None:
        source_projection.extract_file(args.fixture_public, args.fixture_output)
        source_projection.replay_file(args.fixture_public, args.fixture_output)
        return 0
    if args.date is None:
        parser.error("--date is required")
    run_experiment(args.date)
    return 0


if __name__ == "__main__":
    sys.exit(main())
