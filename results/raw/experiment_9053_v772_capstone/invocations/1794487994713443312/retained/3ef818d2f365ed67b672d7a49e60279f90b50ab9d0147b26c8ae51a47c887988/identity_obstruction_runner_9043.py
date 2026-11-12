"""REQ-REPORT-9043-V772: publish an offline diagnosis with separately blocked eligibility."""

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

from carnot.reporting import identity_obstruction_9043 as e
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
    "tests/python/test_source_identity_9029.py",
    "tests/python/test_human_disagreement_source_9015.py",
    "tests/python/test_human_source_authority_9001.py",
    "tests/python/test_primary_publication_7928.py",
    "tests/python/test_current_work_receipt.py",
]
E2E = [
    "tests/python/test_experiment_7891_v685_authority_lifecycle.py",
    "tests/python/test_restricted_decision_audit_8210.py",
]
REUSED = [
    "python/carnot/reporting/" + name + ".py"
    for name in [
        "source_identity_9029",
        "current_work_receipt",
        "primary_publication",
        "v709_execution",
        "v756_branch_qualification",
        "roadmap_contract",
        "natural_label_runner_8975",
        "natural_label_eligibility_8975",
        "human_source_authority_9001",
        "human_disagreement_source_9015",
    ]
] + ["scripts/adversarial_verify.py", "scripts/verdict_row_consistency_lint.py"]


def scope_manifest(commands: list[Json]) -> Json:
    """Freeze explicit callers, tests and validators before measuring current work."""
    return dict(
        commands=commands,
        owned_paths=e.OWNED,
        required_check_names=CHECKS,
        code_test_hashes=[
            e.old.ref(e.ROOT / p) for p in [*e.OWNED, e.TEST, *DIRECT, *E2E, *REUSED]
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
    """Authenticate normal exits and Coverage.py SQLite, rather than asserted percentages."""
    try:
        expected = {c["name"]: c["argv"] for c in scope["commands"]}
        checks = validation["checks"]
        return bool(
            sorted(expected) == sorted(CHECKS)
            and sorted(c["name"] for c in checks) == sorted(CHECKS)
            and all(e.old.ref(Path(r["path"])) == r for r in scope["code_test_hashes"])
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
                    e.old.ref(Path(c[s + "_path"]))["sha256"] == c[s + "_sha256"]
                    and c[s + "_sha256"] is not None
                    for s in ["stdout", "stderr"]
                )
                for c in checks
            )
            and verify_measured(validation["owned_statement_coverage"], e.OWNED)
        )
    except (OSError, ValueError, KeyError, TypeError):
        return False


def measure(upstream: Path, output: Path, date: str, scope: Json) -> Json:
    """Record real input and resource checks before processing retained response bytes."""
    e.progress("preconditions_before")
    start = time.monotonic_ns()
    raw = output.parent / "raw" / e.NAME / "invocations" / str(time.time_ns())
    raw.mkdir(parents=True)
    disk = shutil.disk_usage(raw).free
    memory = os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
    atomic_json(raw / "validation_scope_manifest.json", scope)
    inputs: Json = dict(references=[e.old.ref(upstream)], primary={})
    reduction: Json = dict(
        rows=[],
        obstruction_rows=[],
        request_rows=[],
        release_group_rows=[],
        category_counts=dict.fromkeys(e.CATEGORIES, 0),
        parser_regression_evidence={},
        reopening_evidence={},
        identity_diagnosis_ready_score=0,
        source_identity_ready_score=0,
        intended_count=215,
        completed_count=0,
        independent_count=0,
        excluded_count=0,
        censored_count=215,
        failed_count=0,
        gate_check_summary=[],
    )
    checks: list[Json] = []
    shards = []
    try:
        checks.append(
            e.require(
                Path(sys.executable),
                "virtualenv",
                str(e.ROOT / ".venv"),
                str(Path(sys.prefix).resolve()),
            )
        )
        checks.append(e.require(raw, "resources", True, min(disk, memory) >= 300_000_000))
        inputs = e.load(upstream)
        checks += inputs["gate_check_summary"]
        e.progress("preconditions_after", len(inputs["references"]), 0)
        reduction = e.diagnose(inputs["metadata"])
        checks.append(
            e.require(
                upstream,
                "strict_gate_operands",
                inputs["primary"]["gate_check_summary"],
                [
                    g | {k: inputs["primary"]["metadata_manifest"][k] for k in ["path", "sha256"]}
                    for g in reduction["gate_check_summary"]
                ],
            )
        )
        reduction["gate_check_summary"] = deepcopy(inputs["primary"]["gate_check_summary"])
        for index, receipt in enumerate(reduction["request_rows"]):
            path = raw / (f"response-{index:03d}.body")
            shutil.copyfile(receipt["path"], path)
            shards.append(
                dict(e.old.ref(path), source=dict(path=receipt["path"], sha256=receipt["sha256"]))
            )
        atomic_json(raw / "request_receipts.json", reduction["request_rows"])
        shards.append(e.old.ref(raw / "request_receipts.json"))
    except e.Blocked as error:
        checks.append(error.operand)
    reduction["gate_check_summary"] = checks + reduction["gate_check_summary"]
    end = time.monotonic_ns()
    work = build_current_work_receipt(
        run_id=raw.name,
        owner_pid=os.getpid(),
        events=[],
        inference_substrate="aggregation_from_upstream_artifacts",
        inference_substrate_class="no_model_load",
        inference_substrate_details=dict(
            operation="offline retained metadata classification", current_LLM_calls=0
        ),
        execution_venue="host",
        started_monotonic_ns=start,
        ended_monotonic_ns=end,
        phase_spans=[
            dict(phase="identity_obstruction_audit", start_s=0, end_s=(end - start) / 1e9)
        ],
    )
    primitive = dict(
        upstream=e.old.ref(upstream),
        reduction=reduction,
        run_date=date,
        current_work_receipt=work,
        source_artifact_hashes=inputs["references"],
        raw_shard_hashes=shards,
        repository_health=inputs["primary"].get(
            "repository_health", dict(passed=False, status="upstream_unavailable")
        ),
        validation_scope_manifest=scope,
        code_config_hashes=scope["code_test_hashes"],
        preconditions_checked=dict(
            python_executable=sys.executable,
            available_memory_bytes=memory,
            writable_disk_bytes=disk,
            writable_probe=e.old.ref(raw / "validation_scope_manifest.json"),
            current_model_calls=0,
            network_requests=0,
            labels_accessed=False,
            authenticated_inputs=bool(inputs.get("metadata")),
            checks=checks,
        ),
    )
    atomic_json(raw / "primitive.json", primitive)
    e.progress("measurement_after", reduction["completed_count"], reduction["censored_count"])
    return e.old.ref(raw / "primitive.json")


def build(reference: Json, validation: Json) -> Json:
    """A complete diagnosis does not turn missing cross-language identities into eligible sources."""
    primitive = read_ref(reference)
    value: Json = deepcopy(primitive["reduction"])
    passed = receipts_pass(primitive["validation_scope_manifest"], validation)
    blockers = [g for g in value["gate_check_summary"] if not g["passed"]]
    value.update(
        experiment_id=9043,
        task_id="exp9043-identity-obstruction-audit",
        milestone="2026.11.772",
        run_date=primitive["run_date"],
        verdict_class="disqualified" if not passed else "blocked" if blockers else "null",
        honest_verdict="complete_disqualified_owned_validation"
        if not passed
        else "complete_blocked_" + blockers[0]["check"]
        if blockers
        else "complete_null_identity_diagnosis",
        identity_diagnosis_ready_score=int(passed and value["identity_diagnosis_ready_score"] == 1),
        source_identity_ready_score=int(
            passed and not blockers and value["source_identity_ready_score"] == 1
        ),
        verifier_is_oracle=False,
        independent_generalization_score=0,
        claim_scope="Complete classification of retained identity obstructions only; no human-pilot, factual verifier, learned benefit or north-star result.",
        exposure_scope="All 215 unresolved host/title identities from the frozen release and historical inventory. Missing pages and absent QIDs cannot establish cross-language nonexposure.",
        inference_substrate="aggregation_from_upstream_artifacts",
        inference_substrate_class="no_model_load",
        MODEL_SPECS=[],
        model_specs=[],
        trained_head_specs=[],
        labels_accessed=False,
        network_requests=0,
        model_invocation_counts=primitive["current_work_receipt"]["invocation_counts"],
        current_work_receipt=primitive["current_work_receipt"],
        duration_s=primitive["current_work_receipt"]["duration_s"],
        phase_spans=primitive["current_work_receipt"]["phase_spans"],
        random_seed=7729043,
        primitive_reference=reference,
        source_artifact_hashes=primitive["source_artifact_hashes"],
        raw_shard_hashes=[reference, *primitive["raw_shard_hashes"]],
        code_config_hashes=primitive["code_config_hashes"],
        preconditions_checked=primitive["preconditions_checked"],
        validation_scope_manifest=primitive["validation_scope_manifest"],
        validation_receipts=validation.get("checks", []),
        owned_statement_coverage=validation.get("owned_statement_coverage", {}),
        required_checks_passed=passed,
        flagged_adversarial=not passed,
        repository_health=primitive["repository_health"],
        repository_health_passed=False,
        cited_upstream_artifacts=[
            dict(
                experiment_id=9029,
                sha256=e.UPSTREAM_PIN,
                fields_imported=["metadata_manifest", "gate_check_summary", "repository_health"],
                historical_only=True,
            )
        ],
        methodology_note="Authenticate independent primary, manifest, response and preserved V771 authority pins. Reconstruct original unresolved requests with the unchanged parser and strict grouping rules. Inspect each original response for aliases, markers, local page identity and QID presence. Cold replay recomputes classifications from immutable source bytes. All current model counters are zero; historical model receipts are reference-only.",
    )
    value["acceptance_gates"] = [
        dict(criterion=k, expected=v, observed=value[k], passed=value[k] == v)
        for k, v in [
            ("identity_diagnosis_ready_score", 1),
            ("source_identity_ready_score", 1),
            ("required_checks_passed", True),
        ]
    ]
    value["field_principles"] = dict.fromkeys(
        [*value, "reproducibility_checksum"],
        "Immutable public metadata supports diagnosis only. Complete classification and strict source eligibility are separate claims. Current owned failure disqualifies; external identity blocks are terminal. Primitive bytes and measured scoped checks support replay; historical health stays failed.",
    )
    value["reproducibility_checksum"] = canonical_hash(value)
    return value


def replay(path: Path) -> Json:
    """Recompute classifications from independently pinned originals, even after aggregate rehashing."""
    value = json.loads(path.read_bytes())
    primitive = read_ref(value["primitive_reference"])
    if not primitive["preconditions_checked"]["authenticated_inputs"] and (
        any(
            primitive["reduction"][key]
            for key in [
                "rows",
                "obstruction_rows",
                "request_rows",
                "release_group_rows",
                "identity_diagnosis_ready_score",
                "source_identity_ready_score",
                "completed_count",
            ]
        )
        or any(primitive["reduction"]["category_counts"].values())
    ):
        raise ValueError("diagnosis_drift")
    for reference in (
        value["source_artifact_hashes"] + value["raw_shard_hashes"] + value["code_config_hashes"]
    ):
        if e.old.ref(Path(reference["path"]))["sha256"] != reference["sha256"]:
            raise ValueError("operand_hash")
    if primitive["preconditions_checked"]["authenticated_inputs"]:
        inputs = e.load(Path(primitive["upstream"]["path"]))
        fresh = e.diagnose(inputs["metadata"])
        retained = primitive["reduction"]
        if any(fresh[k] != retained[k] for k in fresh if k != "gate_check_summary"):
            raise ValueError("diagnosis_drift")
        original_gates = inputs["primary"]["gate_check_summary"]
        if retained["gate_check_summary"][-4:] != original_gates:
            raise ValueError("gate_drift")
    if value != build(
        value["primitive_reference"],
        dict(
            checks=value["validation_receipts"],
            owned_statement_coverage=value["owned_statement_coverage"],
        ),
    ):
        raise ValueError("reduction_drift")
    return dict(passed=True, replay_passed=True, primary_sha256=e.old.ref(path)["sha256"])


def validate(candidate: Path, logs: Path) -> Json:
    """Run the real foreign-directory replay and unchanged artifact scanners with bounded waits."""
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
    checks = [child(name, argv, logs, deadline=180, heartbeat=30) for name, argv in commands]
    return dict(passed=all(c["passed"] and c["normal_exit"] for c in checks), checks=checks)


def main(argv: list[str] | None = None) -> int:
    """Keep failed candidates under raw; publish only after current scoped checks pass."""
    os.environ["PYTHONUNBUFFERED"] = "1"
    e.progress("start_no_model_load")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default="20261112")
    parser.add_argument("--upstream", type=Path, default=e.UPSTREAM)
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
        reference = measure(args.upstream, args.output, args.date, scope)
        validation = json.loads(validation_path.read_bytes()) if validation_path.exists() else {}
        result = build(reference, validation)
        if not result["required_checks_passed"]:
            atomic_json(raw / "rejected_primary.json", result)
            e.progress(
                "refused_owned_validation", result["completed_count"], result["censored_count"]
            )
            return 1
        publication = publish_primary(
            args.output, result, lambda p: validate(p, Path(reference["path"]).parent / "terminal")
        )
        print(json.dumps(publication), flush=True)
        e.progress("complete", result["completed_count"], result["censored_count"])
        return 0
    except (OSError, ValueError, KeyError, TypeError):
        e.progress("refused_input_or_replay")
        return 1
