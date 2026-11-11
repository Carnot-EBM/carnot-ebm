"""REQ-VERIFY-9001: publish only a replayable original-release qualification."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import time
from typing import Any

from carnot.reporting import human_source_authority_9001 as e
from carnot.reporting.current_work_receipt import (
    atomic_json,
    build_current_work_receipt,
    canonical_hash,
)
from carnot.reporting.natural_label_runner_8975 import read_ref
from carnot.reporting.primary_publication import publish_primary
from carnot.reporting.v709_execution import child

Json = dict[str, Any]
RAW = e.ROOT / "results/raw" / e.NAME


def measure(manifest: Path, output: Path, date: str) -> Json:
    """Retain measured operands and clocks even when external eligibility is blocked."""
    e.progress("preconditions_no_model_load")
    start = time.monotonic_ns()
    raw = output.parent / "raw" / e.NAME / "invocations" / str(time.time_ns())
    raw.mkdir(parents=True)
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
        reduced = e.audit([], [], set(), False, raw / "audit")
        reduced.update(
            input_scope="external_unavailable",
            source_artifact_hashes=[e.ref(manifest)],
            cited_upstream_artifacts=[],
            exposure_ledger=[],
            source_access_receipts=[],
        )
        reduced["gate_check_summary"].insert(
            0,
            dict(
                check="authenticated_inputs",
                upstream="original_RAGTruth",
                **e.ref(manifest),
                artifact_field="original_release_and_exposure_bytes",
                op="==",
                expected="authenticated pinned original release",
                observed="missing_field:" + str(error)
                if isinstance(error, KeyError)
                else str(error),
                passed=False,
            ),
        )
    end = time.monotonic_ns()
    receipt = build_current_work_receipt(
        run_id=raw.name,
        owner_pid=os.getpid(),
        events=[],
        inference_substrate="aggregation_from_upstream_artifacts",
        inference_substrate_class="no_model_load",
        inference_substrate_details=dict(
            operation="original human annotation and source exposure audit", current_LLM_calls=0
        ),
        execution_venue="host",
        started_monotonic_ns=start,
        ended_monotonic_ns=end,
        phase_spans=[
            dict(phase="preconditions_original_source_audit", start_s=0, end_s=(end - start) / 1e9)
        ],
    )
    primitive = dict(
        manifest=e.ref(manifest),
        reduction=reduced,
        run_date=date,
        current_work_receipt=receipt,
        code_config_hashes=[
            e.ref(e.ROOT / p)
            for p in e.OWNED
            + [
                "tests/python/test_human_source_authority_9001.py",
                "python/carnot/reporting/primary_publication.py",
                "scripts/adversarial_verify.py",
                "scripts/verdict_row_consistency_lint.py",
                "python/carnot/reporting/current_work_receipt.py",
                "python/carnot/reporting/natural_criterion_audit_8908.py",
                "python/carnot/reporting/natural_label_eligibility_8975.py",
                "python/carnot/reporting/natural_label_runner_8975.py",
                "python/carnot/reporting/v709_execution.py",
            ]
        ],
        preconditions_checked=dict(
            python_executable=sys.executable,
            repository_virtualenv=str(e.ROOT / ".venv"),
            dependencies="stdlib and imported existing reporting helpers",
            available_memory_bytes=memory,
            writable_disk_bytes=disk,
            MODEL_SPECS=[],
            current_model_calls=0,
            declared_gate_fields=[
                "documented_manual",
                "human_authority_group_count",
                "unseen_source_group_count",
                "full_role_class_contract",
            ],
            network_total_cap_s=600,
            network_total_cap_bytes=268435456,
            network_policy="Retained bounded public curl requests only; local CLI authenticates their pinned bytes.",
            measurement_cap_s=3600,
            checks_reserve_s=900,
            invocation_cap_s=4800,
        ),
    )
    atomic_json(raw / "primitive.json", primitive)
    e.progress("measurement_complete", reduced["completed_count"], reduced["censored_count"])
    return e.ref(raw / "primitive.json")


def build(reference: Json, validation: Json) -> Json:
    """Readiness requires actual normal exits; external shortages remain terminal blocks."""
    primitive = read_ref(reference)
    checks = validation.get("checks", [])
    coverage = validation.get("owned_statement_coverage", {})
    passed = (
        e.CHECKS <= {c.get("name") for c in checks}
        and all(
            c.get("passed") is True and c.get("normal_exit") is True and c.get("exit_code") == 0
            for c in checks
        )
        and coverage.get("percent_covered") == 100
        and coverage.get("missing_lines") == []
        and coverage.get("excluded_lines") == []
    )
    reduced = primitive["reduction"]
    manual = reduced["annotation_origin_rows"][0]["documented_manual"]
    blocker = (
        "authenticated_inputs"
        if reduced["input_scope"] == "external_unavailable"
        else "documented_human_authority"
        if not manual
        else "unseen_source_groups"
        if not reduced["natural_pilot_ready_score"]
        else None
    )
    verdict = "disqualified" if not passed else "blocked" if blocker else "null"
    value = dict(reduced)
    receipt = primitive["current_work_receipt"]
    value.update(
        experiment_id=9001,
        task_id=e.TASK,
        milestone="2026.11.769",
        run_date=primitive["run_date"],
        honest_verdict="complete_"
        + (
            "disqualified_owned_validation"
            if not passed
            else "blocked_" + blocker
            if blocker
            else "null_human_source_authority_qualified"
        ),
        verdict_class=verdict,
        verifier_is_oracle=False,
        claim_scope="Original human annotation authority and source custody only; no verifier benefit or generalization.",
        exposure_scope="All historically exposed groups are development-only; only unseen original source groups can enter new roles.",
        inference_substrate="aggregation_from_upstream_artifacts",
        inference_substrate_class="no_model_load",
        MODEL_SPECS=[],
        model_specs=[],
        trained_head_specs=[],
        model_invocation_counts=receipt["invocation_counts"],
        current_work_receipt=receipt,
        duration_s=receipt["duration_s"],
        phase_spans=receipt["phase_spans"],
        random_seed=9001,
        primitive_reference=reference,
        raw_shard_hashes=[
            reference,
            *reduced["label_sidecar_hashes"],
            *reduced["public_view_hashes"],
        ],
        code_config_hashes=primitive["code_config_hashes"],
        preconditions_checked=primitive["preconditions_checked"],
        validation_receipts=checks,
        owned_statement_coverage=coverage,
        required_checks_passed=passed,
        flagged_adversarial=not passed,
        repository_health=validation.get("repository_health", []),
    )
    if not passed:
        value.update(independent_label_ready_score=0, natural_pilot_ready_score=0)
        value["gate_check_summary"] = [
            *value["gate_check_summary"],
            dict(
                check="owned_validation",
                upstream=e.TASK,
                **reference,
                artifact_field="required_checks_passed",
                op="==",
                expected=True,
                observed=False,
                passed=False,
            ),
        ]
    value["acceptance_gates"] = [
        dict(criterion=k, expected=v, observed=value[k], passed=value[k] == v)
        for k, v in [
            ("independent_label_ready_score", 1),
            ("natural_pilot_ready_score", 1),
            ("required_checks_passed", True),
        ]
    ]
    value["field_principles"] = {
        k: "Original bytes, independent source groups and actual validation bound this audit; exposed evidence cannot establish new generalization."
        for k in value
    }
    value["field_principles"].update(
        independent_count="Unseen eligible source components, never repeated responses or independent annotators.",
        intended_count="All audited source components plus unfilled600-role slots; completed/excluded/censored reconcile this denominator.",
    )
    value["reproducibility_checksum"] = canonical_hash(value)
    return value


def replay(path: Path) -> Json:
    """Recompute from the release so rehashed summaries cannot conceal changed evidence."""
    value = json.loads(path.read_bytes())
    primitive = read_ref(value["primitive_reference"])
    for reference in primitive["code_config_hashes"]:
        if e.ref(Path(reference["path"])) != reference:
            raise ValueError("executed_source_hash")
    manifest = primitive["manifest"]
    if e.ref(Path(manifest["path"])) != manifest:
        raise ValueError("manifest_hash")
    for reference in value["raw_shard_hashes"]:
        if e.ref(Path(reference["path"])) != reference:
            raise ValueError("raw_shard_hash")
    if primitive["reduction"]["input_scope"] != "external_unavailable":
        with tempfile.TemporaryDirectory(prefix="carnot9001-replay-") as temporary:
            fresh = e.inspect(Path(manifest["path"]), Path(temporary))
            for key in ("label_sidecar_hashes", "public_view_hashes"):
                if [r["sha256"] for r in fresh[key]] != [
                    r["sha256"] for r in primitive["reduction"][key]
                ]:
                    raise ValueError("original_sidecar_drift")
                fresh[key] = primitive["reduction"][key]
            if fresh != primitive["reduction"]:
                raise ValueError("original_reduction_drift")
    for check in value["validation_receipts"]:
        for stream in ("stdout", "stderr"):
            if (
                stream + "_path" in check
                and e.ref(Path(check[stream + "_path"]))["sha256"] != check[stream + "_sha256"]
            ):
                raise ValueError("validation_log_hash")
    expected = build(
        value["primitive_reference"],
        dict(
            checks=value["validation_receipts"],
            owned_statement_coverage=value["owned_statement_coverage"],
            repository_health=value["repository_health"],
        ),
    )
    if expected != value:
        raise ValueError("reduction_drift")
    return dict(passed=True, replay_passed=True, primary_sha256=e.ref(path)["sha256"])


def validate(candidate: Path, logs: Path) -> Json:
    """Run the unchanged validators on private candidate bytes before publication."""
    checks = []
    for name, args in [
        ("cold_replay", [str(e.ROOT / e.CLI), "--cold-replay", str(candidate)]),
        ("adversarial", [str(e.ROOT / "scripts/adversarial_verify.py"), "--json", str(candidate)]),
        (
            "strict_rows",
            [str(e.ROOT / "scripts/verdict_row_consistency_lint.py"), "--strict", str(candidate)],
        ),
    ]:
        check = child(
            name, [str(e.ROOT / ".venv/bin/python"), "-u", *args], logs, deadline=180, heartbeat=20
        )
        for stream in ("stdout_path", "stderr_path"):
            print(Path(check[stream]).read_text(), end="", flush=True)
        checks.append(check)
    return dict(passed=all(c["passed"] for c in checks), checks=checks)


def main(argv: list[str] | None = None) -> int:
    """Use the same retained-byte audit from the requested command or foreign directory."""
    os.environ["PYTHONUNBUFFERED"] = "1"
    e.progress("start_no_model_load")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default="20261111")
    parser.add_argument("--inputs", type=Path, default=RAW / "release/manifest.json")
    parser.add_argument("--receipts", type=Path, default=RAW / "validation_receipts.json")
    parser.add_argument("--output", type=Path, default=e.ROOT / "results" / (e.NAME + ".json"))
    parser.add_argument("--cold-replay", type=Path)
    args = parser.parse_args(argv)
    try:
        datetime.strptime(args.date, "%Y%m%d")
        if len(args.date) != 8:
            raise ValueError("run_date")
        if args.cold_replay:
            print(json.dumps(replay(args.cold_replay)), flush=True)
            return 0
        reference = measure(args.inputs, args.output, args.date)
        validation = json.loads(args.receipts.read_bytes()) if args.receipts.is_file() else {}
        result = build(reference, validation)
        raw = Path(reference["path"]).parent
        candidate = raw / "preflight_candidate.json"
        atomic_json(candidate, result)
        report = validate(candidate, raw / "preflight")
        validation.setdefault("checks", []).extend(report["checks"])
        result = build(reference, validation)
        publication = publish_primary(args.output, result, lambda p: validate(p, raw / "terminal"))
        print(json.dumps(publication), flush=True)
        e.progress("complete", result["completed_count"], result["censored_count"])
        return 0
    except (OSError, ValueError, KeyError, TypeError) as error:
        e.progress("refused:" + str(error))
        return 1
