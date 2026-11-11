"""REQ-REPORT-9015-V770: retain terminal source blocks with independently replayable evidence."""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import time
from typing import Any

from carnot.reporting import human_disagreement_source_9015 as e
from carnot.reporting import v770_branch_evidence as history
from carnot.reporting.current_work_receipt import (
    atomic_json,
    build_current_work_receipt,
    canonical_hash,
)
from carnot.reporting.natural_label_runner_8975 import read_ref
from carnot.reporting.primary_publication import publish_primary
from carnot.reporting.v709_execution import child
from carnot.reporting.v756_branch_qualification import verify_measured

Json = dict[str, Any]
RAW = e.ROOT / "results/raw" / e.NAME
CHECKS = [
    "owned_tests",
    "direct_consumers",
    "e2e018",
    "e2e021",
    "e2e020",
    "ruff_check",
    "ruff_format",
    "strict_mypy",
    "spec_coverage",
    "owned_coverage",
]
DIRECT = [
    "tests/python/test_human_source_authority_9001.py",
    "tests/python/test_natural_criterion_audit_8908.py",
    "tests/python/test_primary_publication_7928.py",
    "tests/python/test_current_work_receipt.py",
    "tests/python/test_experiment_7891_v685_authority_lifecycle.py",
    "tests/python/test_restricted_decision_audit_8210.py",
    "tests/python/test_hard_exit_learning_qualification_8206.py",
]
REUSED = [
    "python/carnot/reporting/" + name + ".py"
    for name in [
        "v770_branch_evidence",
        "v769_branch_evidence",
        "v709_execution",
        "current_work_receipt",
        "natural_label_runner_8975",
        "natural_label_eligibility_8975",
        "human_source_authority_9001",
        "natural_criterion_audit_8908",
        "v756_branch_qualification",
        "primary_publication",
    ]
] + ["scripts/adversarial_verify.py", "scripts/verdict_row_consistency_lint.py"]


def scope_manifest(commands: list[Json]) -> Json:
    """Seal explicit paths and commands before source qualification or test execution."""
    return dict(
        commands=commands,
        owned_paths=e.OWNED,
        code_test_hashes=[e.ref(e.ROOT / p) for p in [*e.OWNED, e.TEST, *DIRECT, *REUSED]],
        frozen_ns=time.monotonic_ns(),
        required_check_names=CHECKS,
        statement_percent=100,
        excluded_statements=0,
        repository_health_separate=True,
        measurement_cap_s=3300,
        validation_reserve_s=1200,
        hard_cap_s=4800,
    )


def receipts_pass(scope: Json, validation: Json) -> bool:
    """Reject missing, stale, failed, timed-out, changed-command or changed-log receipts."""
    try:
        expected = {c["name"]: c["argv"] for c in scope["commands"]}
        checks = validation["checks"]
        return bool(
            sorted(expected) == sorted(CHECKS)
            and sorted(c["name"] for c in checks) == sorted(CHECKS)
            and all(e.ref(Path(r["path"])) == r for r in scope["code_test_hashes"])
            and all(
                c["passed"] is True
                and c["normal_exit"] is True
                and c["timed_out"] is False
                and c["exit_code"] == 0
                and c["started_monotonic_ns"] >= scope["frozen_ns"]
                and c["ended_monotonic_ns"] >= c["started_monotonic_ns"]
                and c["scope_sha256"] == canonical_hash(scope)
                and c["argv"] == expected[c["name"]]
                and all(
                    e.ref(Path(c[s + "_path"]))["sha256"] == c[s + "_sha256"]
                    and c[s + "_sha256"] is not None
                    for s in ["stdout", "stderr"]
                )
                for c in checks
            )
            and verify_measured(validation["owned_statement_coverage"], e.OWNED)
        )
    except (KeyError, ValueError, TypeError, OSError):
        return False


def measure(manifest: Path, output: Path, date: str, scope: Json) -> Json:
    """Authenticate source, environment and failed history without loading any model."""
    e.progress("preconditions_before")
    start = time.monotonic_ns()
    raw = output.parent / "raw" / e.NAME / "invocations" / str(time.time_ns())
    raw.mkdir(parents=True)
    atomic_json(raw / "validation_scope_manifest.json", scope)
    disk = shutil.disk_usage(raw).free
    memory = os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
    try:
        if (
            min(disk, memory) < 300_000_000
            or Path(sys.prefix).resolve() != (e.ROOT / ".venv").resolve()
        ):
            raise OSError("virtualenv_or_resources")
        reduced = e.inspect(manifest, raw / "audit")
    except (OSError, ValueError, KeyError, TypeError) as error:
        reduced = e.audit([], set(), {}, raw / "audit")
        reduced.update(
            input_scope="external_unavailable",
            release_revision=e.REVISION,
            license_receipt=None,
            annotation_policy_hash=None,
            exposure_inventory=None,
            source_resolution_receipts=[],
            source_access_receipts=[],
            source_artifact_hashes=[e.ref(manifest)],
        )
        observed = (
            "missing_field:" + str(error)
            if isinstance(error, KeyError)
            else "missing_file:" + str(error)
            if isinstance(error, FileNotFoundError)
            else str(error)
        )
        reduced["gate_check_summary"].insert(
            0,
            e.gate(
                "authenticated_inputs", "authenticated pinned release", observed, e.ref(manifest)
            ),
        )
    health = history.health(e.ROOT)
    for reference in health["references"]:
        if e.ref(Path(reference["path"])) != reference:
            raise ValueError("historical_health_hash")
    end = time.monotonic_ns()
    receipt = build_current_work_receipt(
        run_id=raw.name,
        owner_pid=os.getpid(),
        events=[],
        inference_substrate="aggregation_from_upstream_artifacts",
        inference_substrate_class="no_model_load",
        inference_substrate_details=dict(
            operation="original human disagreement source qualification", current_LLM_calls=0
        ),
        execution_venue="host",
        started_monotonic_ns=start,
        ended_monotonic_ns=end,
        phase_spans=[dict(phase="source_qualification", start_s=0, end_s=(end - start) / 1e9)],
    )
    primitive = dict(
        manifest=e.ref(manifest),
        reduction=reduced,
        run_date=date,
        current_work_receipt=receipt,
        repository_health=health,
        validation_scope_manifest=scope,
        code_config_hashes=[e.ref(e.ROOT / p) for p in [*e.OWNED, e.TEST, *DIRECT, *REUSED]],
        preconditions_checked=dict(
            python_executable=sys.executable,
            repository_virtualenv=str(e.ROOT / ".venv"),
            dependencies="stdlib, pyarrow and existing reporting helpers",
            available_memory_bytes=memory,
            writable_disk_bytes=disk,
            MODEL_SPECS=[],
            current_model_calls=0,
            network_total_cap_s=600,
            network_total_cap_bytes=268435456,
            measurement_cap_s=3300,
            validation_reserve_s=1200,
            hard_cap_s=4800,
            declared_gate_fields=[
                "release_authority",
                "network_budget",
                "source_resolution_access",
                "full_role_class_label_contract",
                "all_fourteen_languages",
            ],
        ),
    )
    atomic_json(raw / "primitive.json", primitive)
    e.progress("measurement_complete", reduced["completed_count"], reduced["censored_count"])
    return e.ref(raw / "primitive.json")


def build(reference: Json, validation: Json) -> Json:
    """Current normal exits qualify source custody only; failed repository history stays failed."""
    primitive = read_ref(reference)
    value: Json = deepcopy(primitive["reduction"])
    value["gate_check_summary"].extend(primitive["repository_health"].get("checks", []))
    value["source_artifact_hashes"].extend(e.ref(e.ROOT / path) for path in list(history.PINS)[-3:])
    scope = primitive["validation_scope_manifest"]
    passed = receipts_pass(scope, validation)
    blockers = [g for g in value["gate_check_summary"] if not g["passed"]]
    cls = "disqualified" if not passed else "blocked" if blockers else "null"
    value.update(
        experiment_id=9015,
        task_id=e.TASK,
        milestone="2026.11.770",
        run_date=primitive["run_date"],
        honest_verdict="complete_disqualified_owned_validation"
        if not passed
        else "complete_blocked_" + blockers[0]["check"]
        if blockers
        else "complete_null_human_source_qualified",
        verdict_class=cls,
        verifier_is_oracle=False,
        claim_scope="Original human disagreement source qualification only; no detector result, factual oracle or span leaderboard claim.",
        exposure_scope="Historical source identities are excluded before labels; unresolved cross-language source identities remain ineligible.",
        inference_substrate="aggregation_from_upstream_artifacts",
        inference_substrate_class="no_model_load",
        MODEL_SPECS=[],
        model_specs=[],
        trained_head_specs=[],
        model_invocation_counts=primitive["current_work_receipt"]["invocation_counts"],
        current_work_receipt=primitive["current_work_receipt"],
        duration_s=primitive["current_work_receipt"]["duration_s"],
        phase_spans=primitive["current_work_receipt"]["phase_spans"],
        random_seed=7709015,
        primitive_reference=reference,
        raw_shard_hashes=[reference, *value["label_sidecar_hashes"], *value["public_view_hashes"]],
        code_config_hashes=primitive["code_config_hashes"],
        preconditions_checked=primitive["preconditions_checked"],
        validation_scope_manifest=scope,
        validation_receipts=validation.get("checks", []),
        owned_statement_coverage=validation.get("owned_statement_coverage", {}),
        required_checks_passed=passed,
        flagged_adversarial=not passed,
        repository_health=primitive["repository_health"],
        repository_health_passed=False,
        cited_upstream_artifacts=[
            dict(
                experiment_id=int(Path(p).name.split("_")[1]),
                sha256=pin,
                fields_imported=["verdict_class", "repository_health"],
                historical_only=True,
            )
            for p, pin in list(history.PINS.items())[-3:]
        ],
        methodology_note="Pinned author validation and test parquet retain original multi-annotator spans. Roles precede label reduction. Required reference metadata access failures terminate qualification. No new model calls or alternative dataset. Historical full-suite failures remain disqualified, separately from current scoped checks.",
    )
    if not passed or blockers:
        value.update(natural_pilot_ready_score=0, independent_label_ready_score=0)
    if not passed:
        value["gate_check_summary"].append(e.gate("owned_validation", True, False, reference))
    value["acceptance_gates"] = [
        dict(criterion=k, expected=v, observed=value[k], passed=value[k] == v)
        for k, v in [
            ("natural_pilot_ready_score", 1),
            ("independent_label_ready_score", 1),
            ("required_checks_passed", True),
        ]
    ]
    value["field_principles"] = dict.fromkeys(
        [*value, "reproducibility_checksum"],
        "Retained original bytes and independent source identities bound this source audit; normal scoped checks cannot repair failed historical results or establish detector benefit.",
    )
    value["reproducibility_checksum"] = canonical_hash(value)
    return value


def replay(path: Path) -> Json:
    """Recompute from original source metadata so rehashed aggregates cannot grant readiness."""
    value = json.loads(path.read_bytes())
    primitive = read_ref(value["primitive_reference"])
    for reference in (
        value["source_artifact_hashes"] + value["raw_shard_hashes"] + value["code_config_hashes"]
    ):
        if e.ref(Path(reference["path"])) != reference:
            raise ValueError("operand_hash")
    if primitive["reduction"]["input_scope"] != "external_unavailable":
        with tempfile.TemporaryDirectory(prefix="carnot9015-replay-") as temporary:
            fresh = e.inspect(Path(primitive["manifest"]["path"]), Path(temporary))
            for key in ["label_sidecar_hashes", "public_view_hashes"]:
                if [r["sha256"] for r in fresh[key]] != [
                    r["sha256"] for r in primitive["reduction"][key]
                ]:
                    raise ValueError("sidecar_drift")
                fresh[key] = primitive["reduction"][key]
            if fresh != primitive["reduction"]:
                raise ValueError("source_reduction_drift")
    for reference in primitive["repository_health"]["references"]:
        if e.ref(Path(reference["path"])) != reference:
            raise ValueError("historical_health_hash")
    expected = build(
        value["primitive_reference"],
        dict(
            checks=value["validation_receipts"],
            owned_statement_coverage=value["owned_statement_coverage"],
        ),
    )
    if expected != value:
        raise ValueError("reduction_drift")
    return dict(passed=True, replay_passed=True, primary_sha256=e.ref(path)["sha256"])


def validate(candidate: Path, logs: Path) -> Json:
    """Run unchanged terminal scanners and an actual foreign-directory cold replay."""
    checks = []
    for name, args in [
        (
            "cold_replay",
            [
                "env",
                "-u",
                "PYTHONPATH",
                "-C",
                tempfile.gettempdir(),
                str(e.ROOT / ".venv/bin/python"),
                "-u",
                str(e.ROOT / e.CLI),
                "--cold-replay",
                str(candidate),
            ],
        ),
        (
            "adversarial",
            [
                str(e.ROOT / ".venv/bin/python"),
                "-u",
                str(e.ROOT / "scripts/adversarial_verify.py"),
                "--json",
                str(candidate),
            ],
        ),
        (
            "strict_rows",
            [
                str(e.ROOT / ".venv/bin/python"),
                "-u",
                str(e.ROOT / "scripts/verdict_row_consistency_lint.py"),
                "--strict",
                str(candidate),
            ],
        ),
    ]:
        checks.append(child(name, args, logs, deadline=120, heartbeat=20))
    return dict(passed=all(c["passed"] and c["normal_exit"] for c in checks), checks=checks)


def main(argv: list[str] | None = None) -> int:
    """Publish a checked terminal result; unavailable external evidence never triggers a fallback."""
    os.environ["PYTHONUNBUFFERED"] = "1"
    e.progress("start_no_model_load")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default="20261111")
    parser.add_argument("--inputs", type=Path, default=RAW / "release/manifest.json")
    parser.add_argument("--receipts", type=Path)
    parser.add_argument("--scope", type=Path)
    parser.add_argument("--output", type=Path, default=e.ROOT / "results" / (e.NAME + ".json"))
    parser.add_argument("--cold-replay", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.date != "20261111":
            raise ValueError("run_date")
        if args.cold_replay:
            print(json.dumps(replay(args.cold_replay)), flush=True)
            return 0
        raw = args.output.parent / "raw" / e.NAME
        scope_path = args.scope or raw / "validation_scope_manifest.json"
        receipts_path = args.receipts or raw / "validation.json"
        scope = json.loads(scope_path.read_bytes()) if scope_path.is_file() else scope_manifest([])
        reference = measure(args.inputs, args.output, args.date, scope)
        validation = json.loads(receipts_path.read_bytes()) if receipts_path.is_file() else {}
        result = build(reference, validation)
        raw = Path(reference["path"]).parent
        publication = publish_primary(args.output, result, lambda p: validate(p, raw / "terminal"))
        print(json.dumps(publication), flush=True)
        e.progress("complete", result["completed_count"], result["censored_count"])
        return 0
    except (OSError, ValueError, KeyError, TypeError) as error:
        e.progress("refused:" + str(error))
        return 1
