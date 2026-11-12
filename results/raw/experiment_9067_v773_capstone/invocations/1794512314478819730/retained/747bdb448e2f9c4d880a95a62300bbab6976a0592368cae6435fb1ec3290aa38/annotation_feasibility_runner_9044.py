"""REQ-REPORT-9044-V772: publish only replayable metadata with current scoped checks."""

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

from carnot.reporting import annotation_feasibility_9044 as e
from carnot.reporting import v770_branch_evidence as history
from carnot.reporting.current_work_receipt import (
    atomic_json,
    build_current_work_receipt,
    canonical_hash,
)
from carnot.reporting.human_disagreement_source_9015 import gate
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
DIRECT = [
    "tests/python/test_human_disagreement_source_9015.py",
    "tests/python/test_primary_publication_7928.py",
    "tests/python/test_current_work_receipt.py",
]
REUSED = [
    "python/carnot/reporting/" + n + ".py"
    for n in [
        "human_disagreement_source_9015",
        "natural_label_runner_8975",
        "natural_label_eligibility_8975",
        "v770_branch_evidence",
        "v769_branch_evidence",
        "primary_publication",
        "current_work_receipt",
        "v709_execution",
        "v756_branch_qualification",
    ]
] + [
    "scripts/adversarial_verify.py",
    "scripts/verdict_row_consistency_lint.py",
    "tests/python/test_experiment_7891_v685_authority_lifecycle.py",
    "tests/python/test_restricted_decision_audit_8210.py",
]


def scope_manifest(commands: list[Json]) -> Json:
    """Bind code and commands before measuring; unrelated repository failures stay separate."""
    return dict(
        commands=commands,
        owned_paths=e.OWNED,
        code_test_hashes=[e.ref(e.ROOT / p) for p in [*e.OWNED, e.TEST, *DIRECT, *REUSED]],
        frozen_ns=time.monotonic_ns(),
        required_check_names=CHECKS,
        statement_percent=100,
        excluded_statements=0,
        measurement_cap_s=3300,
        validation_reserve_s=1200,
        hard_cap_s=4800,
    )


def receipts_pass(scope: Json, validation: Json) -> bool:
    """Require normal exits and actual coverage, with immutable code, logs and command vectors."""
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
    """Authenticate resource and upstream operands before any ID-only record access."""
    e.progress("preconditions_before")
    start = time.monotonic_ns()
    raw = output.parent / "raw" / e.NAME / "invocations" / str(time.time_ns())
    raw.mkdir(parents=True)
    atomic_json(raw / "validation_scope_manifest.json", scope)
    disk = shutil.disk_usage(raw).free
    memory = os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
    resources = (
        min(disk, memory) >= 300_000_000
        and Path(sys.prefix).resolve() == (e.ROOT / ".venv").resolve()
    )
    authority = [
        gate(
            "authority_" + Path(p).stem,
            "sha256:" + pin,
            e.ref(e.ROOT / p)["sha256"],
            e.ref(e.ROOT / p),
        )
        for p, pin in e.AUTHORITY.items()
    ]
    for operand in authority:
        operand["artifact_field"] = "sha256"
    authority.append(gate("virtualenv_and_resources", True, resources, e.ref(manifest)))
    source = e.ROOT / "results/experiment_9029_v771_source_identity_resolution.json"
    upstream = json.loads(source.read_bytes()) if source.is_file() else {}
    current_gate = gate(
        "source_identity_ready_score",
        1,
        upstream.get(
            "source_identity_ready_score", "missing_field" if source.is_file() else "missing_file"
        ),
        e.ref(source),
    )
    current_gate["artifact_field"] = "source_identity_ready_score"
    e.progress(
        "preconditions_after",
        sum(g["passed"] for g in authority),
        sum(not g["passed"] for g in authority),
    )
    # Failed authority prevents release access; an unavailable path records the terminal operand.
    reduced = e.inspect(
        manifest if all(g["passed"] for g in authority) else raw / "unavailable_inputs"
    )
    end = time.monotonic_ns()
    receipt = build_current_work_receipt(
        run_id=raw.name,
        owner_pid=os.getpid(),
        events=[],
        inference_substrate="aggregation_from_upstream_artifacts",
        inference_substrate_class="no_model_load",
        inference_substrate_details=dict(
            operation="original annotation structure only", current_LLM_calls=0
        ),
        execution_venue="host",
        started_monotonic_ns=start,
        ended_monotonic_ns=end,
        phase_spans=[dict(phase="annotation_structure", start_s=0, end_s=(end - start) / 1e9)],
    )
    health = history.health(e.ROOT)
    for reference in health["references"]:
        if e.ref(Path(reference["path"])) != reference:
            raise ValueError("historical_health_hash")
    atomic_json(raw / "annotation_schema_rows.json", dict(rows=reduced["annotation_schema_rows"]))
    atomic_json(raw / "access_receipts.json", dict(rows=reduced["access_receipts"]))
    primitive = dict(
        manifest=e.ref(manifest),
        inspected_manifest=e.ref(
            manifest if all(g["passed"] for g in authority) else raw / "unavailable_inputs"
        ),
        reduction=reduced,
        authority_gates=authority,
        current_gate=current_gate,
        run_date=date,
        current_work_receipt=receipt,
        repository_health=health,
        validation_scope_manifest=scope,
        code_config_hashes=scope["code_test_hashes"],
        raw_rows=[e.ref(raw / p) for p in ["annotation_schema_rows.json", "access_receipts.json"]],
        preconditions_checked=dict(
            python_executable=sys.executable,
            virtualenv=sys.prefix,
            available_memory_bytes=memory,
            free_disk_bytes=disk,
            resource_floor_bytes=300_000_000,
            current_model_calls=0,
            retained_local_bytes_only=True,
            test_projection=e.COLUMNS,
            model_specs=[],
            upstream_hashes=authority,
            current_gate_fields=["source_identity_ready_score", "annotation_structure_ready_score"],
            current_gate_operands=[current_gate],
        ),
    )
    atomic_json(raw / "primitive.json", primitive)
    e.progress("measurement_complete", reduced["completed_count"], 0)
    return e.ref(raw / "primitive.json")


def build(reference: Json, validation: Json) -> Json:
    """Qualify structure separately while preserving the blocked natural pilot and failed history."""
    p = read_ref(reference)
    value = deepcopy(p["reduction"])
    gates = p["authority_gates"] + value["gate_check_summary"]
    gates += [
        gate(
            "original_individual_judgments_available",
            1,
            value["annotation_structure_ready_score"],
            p["manifest"],
        ),
        p["current_gate"],
    ]
    gates[-1]["artifact_field"] = "source_identity_ready_score"
    passed = receipts_pass(p["validation_scope_manifest"], validation)
    blockers = [g for g in gates if not g["passed"]]
    value.update(
        experiment_id=9044,
        task_id="exp9044-annotation-feasibility",
        milestone="2026.11.772",
        honest_verdict="complete_disqualified_owned_validation"
        if not passed
        else "complete_blocked_" + blockers[0]["check"]
        if blockers
        else "complete_null_structure_only",
        verdict_class="disqualified" if not passed else "blocked" if blockers else "null",
        gate_check_summary=gates,
        required_checks_passed=passed,
        flagged_adversarial=not passed,
        verifier_is_oracle=False,
        independent_generalization_score=0,
        claim_scope="Original individual annotation availability only; no fitted detector or factual-correctness claim.",
        exposure_scope="Every record enumerated in access_receipts: IDs, language and annotator IDs only; no cohort selected or outcomes reduced.",
        inference_substrate="aggregation_from_upstream_artifacts",
        inference_substrate_class="no_model_load",
        MODEL_SPECS=[],
        model_specs=[],
        trained_head_specs=[],
        model_invocation_counts=p["current_work_receipt"]["invocation_counts"],
        current_work_receipt=p["current_work_receipt"],
        run_date=p["run_date"],
        duration_s=p["current_work_receipt"]["duration_s"],
        phase_spans=p["current_work_receipt"]["phase_spans"],
        random_seed=7729044,
        primitive_reference=reference,
        raw_shard_hashes=[reference, *p["raw_rows"]],
        code_config_hashes=p["code_config_hashes"],
        validation_scope_manifest=p["validation_scope_manifest"],
        validation_receipts=validation.get("checks", []),
        owned_statement_coverage=validation.get("owned_statement_coverage", {}),
        repository_health=p["repository_health"],
        repository_health_passed=False,
        preconditions_checked=p["preconditions_checked"],
        cited_upstream_artifacts=[
            dict(
                experiment_id=int(Path(path).name.split("_")[1]),
                sha256="sha256:" + pin,
                fields_imported=[
                    "honest_verdict",
                    "gate_check_summary",
                    "source_identity_ready_score",
                ],
                historical_model_receipts="reference only; no current invocation inheritance",
            )
            for path, pin in e.AUTHORITY.items()
            if path.startswith("results/experiment_")
        ],
        methodology_note="Pinned author release; nested Parquet leaf projection reads distinct original people without reading annotation values. All official records are inventoried. Span types come from schema metadata only. Context support maps to original person-level risk, never factual truth. No model or numerical fit, no new split and no source eligibility claim.",
    )
    value["source_artifact_hashes"] += [e.ref(e.ROOT / path) for path in e.AUTHORITY]
    value["missing_authority"] = blockers + [
        dict(check="frozen_cohort_and_label_support", observed="unknown; not accessed")
    ]
    if not passed:
        value["annotation_structure_ready_score"] = 0
    value["acceptance_gates"] = [
        dict(criterion=k, expected=1, observed=value[k], passed=value[k] == 1)
        for k in [
            "annotation_structure_ready_score",
            "natural_pilot_ready_score",
            "required_checks_passed",
        ]
    ]
    value["field_principles"] = dict.fromkeys(
        [*value, "reproducibility_checksum"],
        "Structural metadata cannot establish outcome support or unseen eligibility; authenticated primitive bytes and current normal-exit checks bound this claim.",
    )
    value["reproducibility_checksum"] = canonical_hash(value)
    return value


def replay(path: Path) -> Json:
    """Re-read only original IDs and rederive the full artifact to reject rehashed summaries."""
    value = json.loads(path.read_bytes())
    primitive = read_ref(value["primitive_reference"])
    for reference in (
        value["source_artifact_hashes"] + value["raw_shard_hashes"] + value["code_config_hashes"]
    ):
        if e.ref(Path(reference["path"])) != reference:
            raise ValueError("operand_hash")
    fresh = e.inspect(Path(primitive["inspected_manifest"]["path"]))
    if fresh != primitive["reduction"]:
        raise ValueError("source_reduction_drift")
    for reference, rows in zip(
        primitive["raw_rows"],
        [fresh["annotation_schema_rows"], fresh["access_receipts"]],
        strict=True,
    ):
        if read_ref(reference)["rows"] != rows:
            raise ValueError("primitive_rows_drift")
    if history.health(e.ROOT) != primitive["repository_health"]:
        raise ValueError("historical_health_drift")
    for operand in primitive["authority_gates"]:
        if (
            operand["artifact_field"] == "sha256"
            and operand["expected"]
            != "sha256:" + e.AUTHORITY[str(Path(operand["path"]).relative_to(e.ROOT))]
        ):
            raise ValueError("independent_authority_pin")
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
    """Use unchanged scanners and a real cold child; their bytes stay bound to publication."""
    commands = [
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
    ]
    checks = [child(name, argv, logs, deadline=120, heartbeat=20) for name, argv in commands]
    return dict(passed=all(c["passed"] and c["normal_exit"] for c in checks), checks=checks)


def main(argv: list[str] | None = None) -> int:
    """Publish a validated terminal result; absent external inputs never trigger substitution."""
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
        scope = json.loads(scope_path.read_bytes()) if scope_path.is_file() else scope_manifest([])
        validation = json.loads(validation_path.read_bytes()) if validation_path.is_file() else {}
        reference = measure(args.inputs, args.output, args.date, scope)
        value = build(reference, validation)
        publication = publish_primary(
            args.output, value, lambda p: validate(p, Path(reference["path"]).parent / "terminal")
        )
        print(json.dumps(publication), flush=True)
        e.progress("complete", value["completed_count"], 0)
        return 0
    except (OSError, ValueError, KeyError, TypeError) as error:
        e.progress("refused:" + str(error))
        return 1
