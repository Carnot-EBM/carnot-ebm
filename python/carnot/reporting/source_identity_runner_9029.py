"""REQ-REPORT-9029-V771: publish bounded metadata evidence with independent cold replay."""

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

from carnot.reporting import source_identity_9029 as e
from carnot.reporting.current_work_receipt import (
    atomic_json,
    build_current_work_receipt,
    canonical_hash,
)
from carnot.reporting.human_disagreement_runner_9015 import DIRECT, REUSED
from carnot.reporting.natural_label_runner_8975 import read_ref
from carnot.reporting.primary_publication import publish_primary
from carnot.reporting.v709_execution import child
from carnot.reporting.v756_branch_qualification import verify_measured

Json = dict[str, Any]
CHECKS = [
    "owned_tests",
    "direct_consumers",
    "e2e018",
    "e2e021",
    "ruff_check",
    "ruff_format",
    "strict_mypy",
    "spec_coverage",
    "owned_coverage",
]


def scope_manifest(commands: list[Json]) -> Json:
    """Freeze explicit consumers and file hashes before any measurement starts."""
    return dict(
        commands=commands,
        owned_paths=e.OWNED,
        required_check_names=CHECKS,
        code_test_hashes=[
            e.ref(e.ROOT / p)
            for p in [
                *e.OWNED,
                e.TEST,
                *DIRECT,
                *REUSED,
                "tests/python/test_human_disagreement_source_9015.py",
            ]
        ],
        frozen_ns=time.monotonic_ns(),
        statement_percent=100,
        excluded_statements=0,
        measurement_cap_s=3300,
        validation_reserve_s=1200,
        hard_cap_s=4800,
        repository_health_separate=True,
    )


def receipts_pass(scope: Json, validation: Json) -> bool:
    """Authenticate normal exits, frozen commands, logs and measured coverage SQLite."""
    try:
        expected = {c["name"]: c["argv"] for c in scope["commands"]}
        checks = validation["checks"]
        return bool(
            sorted(expected) == sorted(CHECKS)
            and sorted(c["name"] for c in checks) == sorted(CHECKS)
            and all(
                e.ref(Path(r["path"]))["sha256"] == r["sha256"] for r in scope["code_test_hashes"]
            )
            and all(
                c["passed"] is True
                and c["normal_exit"] is True
                and not c["timed_out"]
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
    """Check inputs and resources before making any bounded metadata requests."""
    e.progress("preconditions_before")
    start = time.monotonic_ns()
    raw = output.parent / "raw" / e.NAME / "invocations" / str(time.time_ns())
    raw.mkdir(parents=True)
    disk = shutil.disk_usage(raw).free
    memory = os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
    atomic_json(raw / "validation_scope_manifest.json", scope)
    state: Json = dict(
        resolutions={},
        receipts=[],
        unresolved=[],
        blocked=None,
        cache_reuse_count=0,
        retry_counts=0,
        bytes_downloaded=0,
        network_duration_s=0,
    )
    inputs: Json = dict(
        records=[],
        inventory=dict(exposed_keys=[], historical_reference_urls=[], rows=[]),
        references=[e.ref(manifest)],
        inherited_receipts=[],
    )
    failure = None
    try:
        if (
            min(disk, memory) < 300_000_000
            or Path(sys.prefix).resolve() != (e.ROOT / ".venv").resolve()
        ):
            raise OSError("virtualenv_or_resources")
        inputs = e.load_inputs(manifest)
        e.progress("preconditions_authenticated", len(inputs["records"]), 0)
        urls = [r["wikipedia_url"] for r in inputs["records"]] + inputs["inventory"][
            "historical_reference_urls"
        ]
        state = e.acquire(
            urls, inputs["inherited_receipts"], output.parent / "raw" / e.NAME / "metadata_cache"
        )
    except (OSError, ValueError, KeyError, TypeError) as error:
        failure = (
            "missing_field:" + str(error)
            if isinstance(error, KeyError)
            else "missing_file:" + str(error)
            if isinstance(error, FileNotFoundError)
            else str(error)
        )
    reduced = e.reduce_metadata(inputs["records"], inputs["inventory"], state["resolutions"])
    if failure:
        reduced["gate_check_summary"].insert(
            0,
            e.gate(
                "authenticated_inputs",
                "pinned release and authenticated cache",
                failure,
                e.ref(manifest),
            ),
        )
    if state["blocked"]:
        operand = state["blocked"]
        reduced["gate_check_summary"].insert(
            0,
            e.gate(
                "source_resolution_access",
                200,
                operand["status"],
                dict(
                    e.ref(raw / "metadata_manifest.json"),
                    artifact_field="blocked_request.status",
                    request_url=operand["url"],
                    access_error=operand["error"],
                ),
            ),
        )
    reduced["source_identity_ready_score"] = int(
        not failure and not state["blocked"] and reduced["source_identity_ready_score"] == 1
    )
    metadata = dict(
        experiment_id=9029,
        successor_experiment_id=9033,
        release_revision=e.REVISION,
        records=inputs["records"],
        inventory=inputs["inventory"],
        request_rows=state["receipts"],
        source_group_rows=reduced["source_group_rows"],
        source_identity_ready_score=reduced["source_identity_ready_score"],
        unresolved_source_groups=reduced["unresolved_source_groups"],
        unresolved_requests=state["unresolved"],
        blocked_request=state["blocked"],
        source_artifact_hashes=inputs["references"],
        labels_accessed=False,
        historical_requests_preserved=True,
        new_network_budget_s=600,
        new_network_budget_bytes=268435456,
    )
    atomic_json(raw / "metadata_manifest.json", metadata)
    metadata_ref = e.ref(raw / "metadata_manifest.json")
    for operand in reduced["gate_check_summary"]:
        if not operand["path"] or operand["path"] == metadata_ref["path"]:
            operand.update(metadata_ref)
    health = json.loads(e.UPSTREAM.read_bytes())["repository_health"]
    health["current_full_suite"] = [
        dict(json.loads(p.read_bytes()), receipt_reference=e.ref(p))
        for p in (e.ROOT / "results/raw" / e.NAME / "repository_health").glob(
            "full_python_suite_once.receipt.json"
        )
    ]
    health["references"] += [
        e.ref(e.ROOT / r[k])
        for r in health["current_full_suite"]
        for k in ["stdout_path", "stderr_path"]
    ]
    health["references"] += [r["receipt_reference"] for r in health["current_full_suite"]]
    for reference in health["references"]:
        e.authenticate(reference)
    end = time.monotonic_ns()
    receipt = build_current_work_receipt(
        run_id=raw.name,
        owner_pid=os.getpid(),
        events=[],
        inference_substrate="aggregation_from_upstream_artifacts",
        inference_substrate_class="no_model_load",
        inference_substrate_details=dict(
            operation="public source metadata and historical exposure joins", current_LLM_calls=0
        ),
        execution_venue="host",
        started_monotonic_ns=start,
        ended_monotonic_ns=end,
        phase_spans=[dict(phase="metadata_resolution", start_s=0, end_s=(end - start) / 1e9)],
    )
    primitive = dict(
        manifest=e.ref(manifest),
        metadata_manifest=metadata_ref,
        reduction=reduced,
        acquisition=state,
        run_date=date,
        current_work_receipt=receipt,
        repository_health=health,
        validation_scope_manifest=scope,
        preconditions_checked=dict(
            python_executable=sys.executable,
            repository_virtualenv=str(e.ROOT / ".venv"),
            available_memory_bytes=memory,
            writable_disk_bytes=disk,
            dependencies="stdlib, pyarrow, coverage and retained reporting helpers",
            writable_probe=metadata_ref,
            current_model_calls=0,
            declared_gate_fields=[g["check"] for g in reduced["gate_check_summary"]],
            network_cap_s=600,
            network_cap_bytes=268435456,
            authenticated_inputs=failure is None,
        ),
        source_artifact_hashes=inputs["references"],
        code_config_hashes=scope["code_test_hashes"],
    )
    atomic_json(raw / "primitive.json", primitive)
    e.progress("measurement_complete", reduced["completed_count"], reduced["censored_count"])
    return e.ref(raw / "primitive.json")


def build(reference: Json, validation: Json) -> Json:
    """Qualify current metadata custody without claiming label or verifier performance."""
    primitive = read_ref(reference)
    value: Json = deepcopy(primitive["reduction"])
    scope = primitive["validation_scope_manifest"]
    passed = receipts_pass(scope, validation)
    blockers = [g for g in value["gate_check_summary"] if not g["passed"]]
    state = primitive["acquisition"]
    value.update(
        experiment_id=9029,
        task_id="exp9029-source-identity-resolution",
        milestone="2026.11.771",
        run_date=primitive["run_date"],
        verdict_class="disqualified" if not passed else "blocked" if blockers else "null",
        honest_verdict="complete_disqualified_owned_validation"
        if not passed
        else "complete_blocked_" + blockers[0]["check"]
        if blockers
        else "complete_null_source_identity_resolved",
        source_identity_ready_score=int(
            passed and not blockers and value["source_identity_ready_score"] == 1
        ),
        verifier_is_oracle=False,
        independent_generalization_score=0,
        claim_scope="Public source identities and prior exposure joins only. No label, training, factual verification or detector claim.",
        exposure_scope="Exact pinned Exp9015 historical inventory; actual item/source overlap excluded, planning prose alone is not an exposed row; unresolved joins never establish unseen sources.",
        inference_substrate="aggregation_from_upstream_artifacts",
        inference_substrate_class="no_model_load",
        MODEL_SPECS=[],
        model_specs=[],
        trained_head_specs=[],
        model_invocation_counts=primitive["current_work_receipt"]["invocation_counts"],
        current_work_receipt=primitive["current_work_receipt"],
        duration_s=primitive["current_work_receipt"]["duration_s"],
        phase_spans=primitive["current_work_receipt"]["phase_spans"],
        random_seed=7719029,
        metadata_manifest=primitive["metadata_manifest"],
        release_revision=e.REVISION,
        request_rows=state["receipts"],
        source_resolution_receipts=[
            dict(path=r["path"], sha256=r["sha256"]) for r in state["receipts"]
        ],
        cache_reuse_count=state["cache_reuse_count"],
        retry_counts=state["retry_counts"],
        bytes_downloaded=state["bytes_downloaded"],
        unresolved_request_count=len(state["unresolved"]),
        network_duration_s=state["network_duration_s"],
        labels_accessed=False,
        primitive_reference=reference,
        source_artifact_hashes=primitive["source_artifact_hashes"],
        raw_shard_hashes=[
            reference,
            primitive["metadata_manifest"],
            *[dict(path=r["path"], sha256=r["sha256"]) for r in state["receipts"]],
        ],
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
                experiment_id=9015,
                sha256=e.UPSTREAM_PIN,
                fields_imported=[
                    "release_revision",
                    "source_access_receipts",
                    "exposure_inventory",
                    "repository_health",
                ],
                historical_only=True,
            )
        ],
        methodology_note="Authenticate the pinned author release and retained HTTP responses. Deduplicate normalized Wikipedia requests across all release languages and historical sources, then resolve serial batches under fixed budgets. Retain redirects and QIDs; group public identity hashes with the frozen document parser. Atomic checkpoints preserve successes through later blocks. No annotations or models are opened. Cold replay uses retained bytes only; current normal exits and zero-exclusion owned coverage qualify publication, separately from failed historical health.",
    )
    value["acceptance_gates"] = [
        dict(criterion=k, expected=v, observed=value[k], passed=value[k] == v)
        for k, v in [("source_identity_ready_score", 1), ("required_checks_passed", True)]
    ]
    value["field_principles"] = dict.fromkeys(
        [*value, "reproducibility_checksum"],
        "Authenticated public metadata and complete historical joins bound readiness. External incompleteness is terminal blocked; current failed checks disqualify. Retained bytes and primitive group denominators support replay, never a label or verifier win.",
    )
    value["reproducibility_checksum"] = canonical_hash(value)
    return value


def replay(path: Path) -> Json:
    """Rebuild group components from retained response bytes without network or labels."""
    value = json.loads(path.read_bytes())
    primitive = read_ref(value["primitive_reference"])
    for reference in (
        value["source_artifact_hashes"] + value["raw_shard_hashes"] + value["code_config_hashes"]
    ):
        if e.ref(Path(reference["path"]))["sha256"] != reference["sha256"]:
            raise ValueError("operand_hash:" + reference["path"])
    metadata = read_ref(primitive["metadata_manifest"])
    if primitive["preconditions_checked"]["authenticated_inputs"]:
        original = e.load_inputs(Path(primitive["manifest"]["path"]))
        if (
            metadata["records"] != original["records"]
            or metadata["inventory"] != original["inventory"]
        ):
            raise ValueError("public_metadata_drift")
    resolutions = {}
    for receipt in metadata["request_rows"]:
        resolutions.update(e.parse_receipt(receipt))
    if resolutions != primitive["acquisition"]["resolutions"]:
        raise ValueError("resolution_drift")
    fresh = e.reduce_metadata(metadata["records"], metadata["inventory"], resolutions)
    expected_ready = int(
        fresh["source_identity_ready_score"] == 1
        and not primitive["acquisition"]["blocked"]
        and primitive["preconditions_checked"]["authenticated_inputs"]
    )
    if primitive["reduction"]["source_identity_ready_score"] != expected_ready:
        raise ValueError("readiness_drift")
    expected_gates = {
        g["check"]: [g[k] for k in ["expected", "observed", "passed"]]
        for g in fresh["gate_check_summary"]
    }
    retained_gates = {
        g["check"]: [g[k] for k in ["expected", "observed", "passed"]]
        for g in primitive["reduction"]["gate_check_summary"]
    }
    if any(retained_gates.get(k) != v for k, v in expected_gates.items()):
        raise ValueError("gate_drift")
    for key in [
        "rows",
        "source_group_rows",
        "exposure_inventory",
        "unresolved_source_groups",
        "intended_count",
        "independent_count",
        "completed_count",
        "excluded_count",
        "censored_count",
        "failed_count",
    ]:
        if fresh[key] != primitive["reduction"][key]:
            raise ValueError("component_drift:" + key)
    for reference in primitive["repository_health"]["references"]:
        e.authenticate(reference)
    if value != build(
        value["primitive_reference"],
        dict(
            checks=value["validation_receipts"],
            owned_statement_coverage=value["owned_statement_coverage"],
        ),
    ):
        raise ValueError("reduction_drift")
    return dict(passed=True, replay_passed=True, primary_sha256=e.ref(path)["sha256"])


def validate(candidate: Path, logs: Path) -> Json:
    """Run unchanged scanners and the actual CLI from a foreign working directory."""
    checks = []
    for name, argv in [
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
        checks.append(child(name, argv, logs, deadline=120, heartbeat=20))
    return dict(passed=all(c["passed"] and c["normal_exit"] for c in checks), checks=checks)


def main(argv: list[str] | None = None) -> int:
    """Publish only scanner-accepted terminal evidence; source blocks never trigger substitutes."""
    os.environ["PYTHONUNBUFFERED"] = "1"
    e.progress("start_no_model_load")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default="20261112")
    parser.add_argument("--inputs", type=Path, default=e.DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=e.ROOT / "results" / (e.NAME + ".json"))
    parser.add_argument("--scope", type=Path)
    parser.add_argument("--receipts", type=Path)
    parser.add_argument("--cold-replay", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.date != "20261112":
            raise ValueError("run_date")
        if args.cold_replay:
            print(json.dumps(replay(args.cold_replay)), flush=True)
            return 0
        raw = args.output.parent / "raw" / e.NAME
        scope_path = args.scope or raw / "validation_scope_manifest.json"
        validation_path = args.receipts or raw / "validation.json"
        scope = json.loads(scope_path.read_bytes()) if scope_path.exists() else scope_manifest([])
        reference = measure(args.inputs, args.output, args.date, scope)
        validation = json.loads(validation_path.read_bytes()) if validation_path.exists() else {}
        result = build(reference, validation)
        publication = publish_primary(
            args.output, result, lambda p: validate(p, Path(reference["path"]).parent / "terminal")
        )
        print(json.dumps(publication), flush=True)
        e.progress("complete", result["completed_count"], result["censored_count"])
        return 0
    except (OSError, ValueError, KeyError, TypeError) as error:
        e.progress("refused:" + str(error))
        return 1
