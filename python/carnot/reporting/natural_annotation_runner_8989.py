"""REQ-REPORT-8989: bind the annotation audit to retained bytes and normal-exit checks."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import shutil
import sys
import time
from typing import Any

from carnot.reporting import natural_annotation_authority_8989 as e
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
    """Retain original operands and actual clocks before validation can grant readiness."""
    e.progress("preconditions")
    start = time.monotonic_ns()
    raw = output.parent / "raw" / e.NAME / "invocations" / str(time.time_ns())
    raw.mkdir(parents=True)
    disk = shutil.disk_usage(raw).free
    memory = os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
    try:
        if min(disk, memory) < 300_000_000:
            raise OSError("insufficient_disk_or_memory")
        reduced = e.inspect(manifest)
    except (OSError, ValueError, KeyError, TypeError) as error:
        reduced = e.audit([], {})
        reduced.update(
            input_scope="external_unavailable",
            release_revision=e.REVISIONS,
            source_artifact_hashes=[e.ref(manifest)],
            license_references={},
            source_urls={},
        )
        reduced["gate_check_summary"].append(
            dict(
                check="authenticated_release",
                upstream="RAGognize",
                **e.ref(manifest),
                artifact_field="pinned_release_bytes",
                op="==",
                expected="retained pinned release",
                observed=str(error),
                passed=False,
            )
        )
    end = time.monotonic_ns()
    receipt = build_current_work_receipt(
        run_id=raw.name,
        owner_pid=os.getpid(),
        events=[],
        inference_substrate="aggregation_from_upstream_artifacts",
        inference_substrate_class="no_model_load",
        inference_substrate_details=dict(
            operation="read-only original release and code audit", current_LLM_calls=0
        ),
        execution_venue="host",
        started_monotonic_ns=start,
        ended_monotonic_ns=end,
        phase_spans=[
            dict(phase="preconditions_and_release_audit", start_s=0, end_s=(end - start) / 1e9)
        ],
    )
    prior = e.ROOT / "results/experiment_8975_v767_natural_label_eligibility.json"
    saved = raw / prior.name
    saved.write_bytes(prior.read_bytes())
    old = json.loads(saved.read_bytes())
    primitive = dict(
        manifest=e.ref(manifest),
        reduction=reduced,
        current_work_receipt=receipt,
        run_date=date,
        code_config_hashes=[
            e.ref(e.ROOT / p)
            for p in e.OWNED
            + [
                "tests/python/test_natural_annotation_authority_8989.py",
                "python/carnot/reporting/primary_publication.py",
                "scripts/adversarial_verify.py",
                "scripts/verdict_row_consistency_lint.py",
            ]
        ],
        cited_upstream_artifacts=[
            dict(
                experiment_id=8975,
                fields_imported=["honest_verdict", "verdict_class"],
                **e.ref(saved),
                imported_values={k: old[k] for k in ["honest_verdict", "verdict_class"]},
                scope="historical_context_only; unchanged AuthenHallu source access not retried",
            )
        ],
        preconditions_checked=dict(
            python_executable=sys.executable,
            repository_virtualenv=str(e.ROOT / ".venv"),
            dependencies="stdlib, pyarrow and existing reporting helpers imported",
            writable_disk_bytes=disk,
            available_memory_bytes=memory,
            declared_upstream_gates=[],
            historical_authority=e.ref(saved),
            MODEL_SPECS=[],
            current_model_calls=0,
            network_total_cap_s=600,
            network_total_cap_bytes=268435456,
            network_request_timeout_s=30,
            network_policy="bounded read-only acquisition retained in release/retrievals.json; this CLI reduces authenticated local bytes",
            new_agreements_accepted=False,
            author_pipeline_executed=False,
            measurement_cap_s=3600,
            checks_reserve_s=900,
            invocation_cap_s=4800,
        ),
    )
    atomic_json(raw / "primitive.json", primitive)
    e.progress("measurement_complete", reduced["completed_count"], 0)
    return e.ref(raw / "primitive.json")


def build(reference: Json, validation: Json) -> Json:
    """Failed owned checks zero readiness even when the annotation audit is complete."""
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
    reduced, receipt = primitive["reduction"], primitive["current_work_receipt"]
    verdict = (
        "disqualified"
        if not passed
        else "blocked"
        if reduced["input_scope"] == "external_unavailable"
        else "null"
    )
    value = dict(reduced)
    value.update(
        experiment_id=8989,
        task_id=e.TASK,
        milestone="2026.11.768",
        run_date=primitive["run_date"],
        honest_verdict="complete_"
        + (
            "null_machine_annotation_only"
            if verdict == "null"
            else verdict
            + "_"
            + ("authenticated_release" if verdict == "blocked" else "owned_validation")
        ),
        verdict_class=verdict,
        verifier_is_oracle=False,
        claim_scope="Public release/pipeline audit only; no verifier benefit or independent factuality evaluation.",
        exposure_scope="Entire pinned public release, both splits; no blind test or new labels.",
        inference_substrate="aggregation_from_upstream_artifacts",
        inference_substrate_class="no_model_load",
        MODEL_SPECS=[],
        model_specs=[],
        trained_head_specs=[],
        model_invocation_counts=receipt["invocation_counts"],
        current_work_receipt=receipt,
        duration_s=receipt["duration_s"],
        phase_spans=receipt["phase_spans"],
        random_seed=8989,
        primitive_reference=reference,
        raw_shard_hashes=[reference],
        code_config_hashes=primitive["code_config_hashes"],
        cited_upstream_artifacts=primitive["cited_upstream_artifacts"],
        preconditions_checked=primitive["preconditions_checked"],
        validation_receipts=checks,
        owned_statement_coverage=coverage,
        required_checks_passed=passed,
        flagged_adversarial=not passed,
        repository_health=validation.get("repository_health", []),
        future_split_criteria="Cluster original article URL/revision, question, context/reference and exact-response duplicates before any fit. Keep all answerability/model variants together; fit no thresholds on exposed evaluation groups.",
        downstream_natural_model_task=False,
        graph_science_gate_dependency=False,
    )
    if not passed:
        value.update(
            source_access_ready_score=0,
            annotation_audit_ready_score=0,
            independent_label_ready_score=0,
        )
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
        dict(criterion=k, expected=w, observed=value[k], passed=value[k] == w)
        for k, w in [
            ("source_access_ready_score", 1),
            ("annotation_audit_ready_score", 1),
            ("independent_label_ready_score", 1),
            ("required_checks_passed", True),
        ]
    ]
    value["field_principles"] = {
        key: "Retain exact bytes, source groups and measured normal-exit validation; machine evidence cannot establish independent human truth."
        for key in value
    }
    value["field_principles"].update(
        independent_count="Distinct article/question groups, not independent human decisions; human label groups are separately zero.",
        completed_count="Response variants inspected, including invalid offset observations; no evidence is discarded.",
    )
    value["reproducibility_checksum"] = canonical_hash(value)
    return value


def replay(path: Path) -> Json:
    """Rebuild from pinned parquet bytes so repaired summary hashes cannot hide drift."""
    value = json.loads(path.read_bytes())
    primitive = read_ref(value["primitive_reference"])
    for reference in primitive["code_config_hashes"]:
        if e.ref(Path(reference["path"])) != reference:
            raise ValueError("executed_source_hash")
    manifest = primitive["manifest"]
    if e.ref(Path(manifest["path"])) != manifest:
        raise ValueError("manifest_hash")
    if primitive["reduction"]["input_scope"] != "external_unavailable":
        if e.inspect(Path(manifest["path"])) != primitive["reduction"]:
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
    """Use unmodified validators on private candidate bytes before readers see them."""
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
    """Offer the same retained-byte audit and cold replay from any working directory."""
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
        e.progress("complete", result["completed_count"], 0)
        return 0
    except (OSError, ValueError, KeyError, TypeError) as error:
        e.progress("refused:" + str(error))
        return 1
