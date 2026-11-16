"""REQ-REPORT-9141-V779: qualify and publish descriptive human evidence with no inference."""

import argparse
import json
import os
from pathlib import Path
import resource
import shutil
import time
from typing import Any
from unittest.mock import patch

from carnot.reporting import human_contrast_9141 as d
from carnot.reporting import logit_methods_runner_9126 as lifecycle
from carnot.reporting.current_work_receipt import atomic_json as atomic_json
from carnot.reporting.current_work_receipt import (
    build_current_work_receipt,
    canonical_hash,
    validate_current_work_receipt,
)
from carnot.reporting.primary_publication import publish_primary

Json = dict[str, Any]
DIRECT = [
    *lifecycle.DIRECT,
    "tests/python/test_experiment_7891_v685_authority_lifecycle.py",
    "tests/python/test_restricted_decision_audit_8210.py",
]


def checksum(value: Json) -> str:
    """A digest detects changed bytes; independent reconstruction also detects changed claims."""
    return canonical_hash({k: v for k, v in value.items() if k != "reproducibility_checksum"})


def measure(root: Path, raw: Path, date: str) -> Json:
    """Seal protocol, exclusions and validation scope before reading annotation outcomes."""
    start = time.monotonic_ns()
    raw.mkdir(parents=True, exist_ok=True, mode=0o700)
    raw.chmod(0o700)
    with patch.multiple(lifecycle, d=d, DIRECT=DIRECT):
        scope = lifecycle.scope(raw)
    scope["commands"][0]["argv"].insert(
        2, "COVERAGE_RCFILE=" + scope["coverage_config_reference"]["path"]
    )
    atomic_json(raw / "diagnostic_protocol.json", d.protocol())
    protocol_ref = d.ref(raw / "diagnostic_protocol.json")
    scope.update(random_seed=d.SEED, exclusions=[], diagnostic_protocol_reference=protocol_ref)
    atomic_json(raw / "frozen_scope.json", scope)
    d.progress("resources_before")
    limit = resource.getrlimit(resource.RLIMIT_AS)[0]
    resources = [
        d.check(
            raw,
            "free_disk_bytes>=2GiB",
            True,
            shutil.disk_usage(raw).free >= 2 * 1024**3,
            "resource_disk",
        ),
        d.check(
            raw,
            "address_space_at_least_256MiB",
            True,
            limit == resource.RLIM_INFINITY or limit >= 256 * 1024**2,
            "resource_memory",
        ),
        d.check(raw, "numpy_float64_bytes", 8, lifecycle.np_itemsize(), "resource_import"),
    ]
    d.progress("resources_after")
    inspected = d.inspect(root)
    inspected["checks"].extend(resources)
    d.progress("paired_qid_bootstrap_before")
    d.reduce(inspected)
    d.progress("paired_qid_bootstrap_after")
    masks = dict(
        rows=[
            dict(
                qid=r["qid"],
                group_id=r["group_id"],
                role=r["role"],
                static_slice="mixed" if r["slice"] == "mixed" else "homogeneous",
            )
            for r in inspected["rows"]
        ]
    )
    atomic_json(raw / "evaluator_masks.json", masks)
    (raw / "evaluator_masks.json").chmod(0o600)
    ended = time.monotonic_ns()
    receipt = build_current_work_receipt(
        run_id=raw.name,
        owner_pid=os.getpid(),
        events=[],
        inference_substrate="cpu_cached_human_annotation_reduction",
        inference_substrate_details=dict(
            work="SHA256 authentication, original annotation counts and paired-qid bootstrap; no fit, model load, forward or generation"
        ),
        inference_substrate_class="no_model_load",
        execution_venue="host",
        started_monotonic_ns=start,
        ended_monotonic_ns=ended,
    )
    atomic_json(raw / "current_work.json", receipt)
    work = dict(
        root=str(root),
        raw=str(raw),
        run_date=date,
        scope=scope,
        inspection=inspected,
        current_work_reference=d.ref(raw / "current_work.json"),
        diagnostic_protocol_reference=protocol_ref,
        evaluator_mask_reference=dict(
            d.ref(raw / "evaluator_masks.json"), role="evaluator_only_after_prediction_seal"
        ),
        start_ns=start,
        duration_s=(ended - start) / 1e9,
    )
    atomic_json(raw / "work.json", work)
    d.progress("measurement_after")
    return work


def receipts_pass(work: Json, validation: Json) -> bool:
    """Existing receipt validation requires actual normal exits, log hashes and SQLite coverage."""
    with patch.object(lifecycle, "d", d):
        return bool(lifecycle.receipts_pass(work, validation))


def qualify(work: Json) -> Json:
    """Reuse supervised checks and one broad-suite attempt with durable heartbeat logs."""
    with patch.object(lifecycle, "d", d):
        return dict(lifecycle.qualify(work))


def build(work: Json, validation: Json) -> Json:
    """Readiness records protocol custody and owned checks, never favorable agreement."""
    inspected = work["inspection"]
    blocked = not all(r["passed"] for r in inspected["checks"])
    passed = receipts_pass(work, validation)
    cls = "blocked" if blocked else "null" if passed else "disqualified"
    rows = inspected["rows"]
    reduced = d.reduce(inspected)
    complete = len(rows)
    value: Json = dict(
        schema="carnot.v779_human_contrast.v1",
        experiment_id=9141,
        task_id=d.TASK,
        milestone="2026.11.779",
        run_date=work["run_date"],
        honest_verdict=f"complete_{cls}_v779_human_contrast_audit",
        verdict_class=cls,
        contrast_protocol_ready_score=int(passed and not blocked),
        required_checks_passed=passed,
        flagged_adversarial=False,
        rows=rows,
        agreement_rows=rows,
        **reduced,
        gate_check_summary=inspected["checks"],
        intended_count=300,
        attempted_count=complete,
        completed_count=complete,
        failed_count=0,
        censored_count=300 - complete,
        excluded_count=0,
        sample_size_budget=dict(
            intended=300,
            attempted=complete,
            completed=complete,
            failed=0,
            censored=300 - complete,
            effective_independent_groups=complete,
            intended_siblings=900,
            completed_siblings=3 * complete,
        ),
        verifier_is_oracle=False,
        independent_generalization_score=0,
        claim_scope=dict(
            exposure_scope="Original LPHB900/300 exposed development",
            independent_generalization_score=0,
            scope="Human disagreement and diagnostic protocol only; no truth ceiling, model benefit, ARC solve, GAP-ORACLE-DISTINCT or GAP-DETECTOR-AUROC-4208 closure",
        ),
        exposure_scope="exposed_development",
        MODEL_SPECS=[],
        requested_inference_substrate_class="no_model_load",
        inference_substrate_class="blocked_no_run" if blocked else "no_model_load",
        inference_substrate="cpu_cached_human_annotation_reduction",
        current_llm_invocations=0,
        generated_tokens=0,
        model_invocation_counts=dict(
            current_llm_invocations=0,
            loads_attempted=0,
            loads_completed=0,
            loads_failed=0,
            forward_calls=0,
            generated_tokens=0,
        ),
        duration_s=work["duration_s"],
        random_seed=d.SEED,
        evaluator_mask_reference=work["evaluator_mask_reference"],
        diagnostic_protocol_reference=work["diagnostic_protocol_reference"],
        diagnostic_outcomes=dict(
            static="unavailable: no model predictions consumed",
            online="unavailable: no model predictions consumed",
        ),
        source_references=inspected["source_references"],
        code_config_hashes=work["scope"]["code_test_hashes"],
        current_work_receipts=[work["current_work_reference"]],
        validation_receipts=validation.get("checks", []),
        owned_statement_coverage=validation.get("owned_statement_coverage", {}),
        repository_health=validation.get("repository_health", {}),
        validation_scope_manifest=work["scope"],
        acceptance_gates=dict(
            source="Exact original labels and roles; missing external inputs block",
            diagnostics="Frozen descriptive slices; no training selection or promotion",
            bounds="Original two static/seven online bounds unchanged",
            validation="All affected checks and100 percent owned statements, zero exclusions",
            agreement="Counts never gate readiness",
        ),
        censor_reason="source_contract_failed" if blocked else None,
        work_reference=d.ref(Path(work["raw"]) / "work.json"),
    )
    value["field_principles"] = {
        k: "Original rows and executed checks bind this descriptive field; no independent model benefit."
        for k in value
    }
    value["reproducibility_checksum"] = checksum(value)
    return value


def replay(path: Path) -> Json:
    """Rebuild rows and intervals under original authority, including rehashed tamper refusal."""
    value = json.loads(path.read_bytes())
    reference = value["work_reference"]
    if d.ref(Path(reference["path"])) != reference:
        raise ValueError("work_hash")
    work = json.loads(Path(reference["path"]).read_bytes())
    for reference in [
        work["current_work_reference"],
        work["diagnostic_protocol_reference"],
        work["scope"]["coverage_config_reference"],
    ]:
        if d.ref(Path(reference["path"])) != reference:
            raise ValueError("operand_hash")
    mask_ref = work["evaluator_mask_reference"]
    if d.ref(Path(mask_ref["path"])) != {k: mask_ref[k] for k in ["path", "sha256"]}:
        raise ValueError("mask_hash")
    if json.loads(Path(work["diagnostic_protocol_reference"]["path"]).read_bytes()) != d.protocol():
        raise ValueError("protocol_drift")
    fresh = d.inspect(Path(work["root"]))
    original = work["inspection"]
    if fresh != dict(original, checks=original["checks"][:-3]):
        raise ValueError("source_drift")
    expected_masks = dict(
        rows=[
            dict(
                qid=r["qid"],
                group_id=r["group_id"],
                role=r["role"],
                static_slice="mixed" if r["slice"] == "mixed" else "homogeneous",
            )
            for r in fresh["rows"]
        ]
    )
    if json.loads(Path(mask_ref["path"]).read_bytes()) != expected_masks:
        raise ValueError("mask_drift")
    if validate_current_work_receipt(
        json.loads(Path(work["current_work_reference"]["path"]).read_bytes()), []
    ):
        raise ValueError("current_work_drift")
    validation = dict(
        checks=value["validation_receipts"],
        owned_statement_coverage=value["owned_statement_coverage"],
        repository_health=value["repository_health"],
    )
    if build(work, validation) != value:
        raise ValueError("reduction_drift")
    return dict(passed=True, replay_passed=True)


def terminal(candidate: Path, logs: Path) -> Json:
    """Unchanged real checkers and foreign-directory replay qualify publication bytes."""
    with patch.object(lifecycle, "d", d):
        return dict(lifecycle.terminal(candidate, logs))


def main(argv: list[str] | None = None) -> int:
    """Private controls cannot overwrite the primary or grant unexecuted validation credit."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default="20261116")
    parser.add_argument("--root", type=Path, default=d.ROOT)
    parser.add_argument("--output", type=Path, default=d.ROOT / "results" / (d.NAME + ".json"))
    parser.add_argument("--cold-replay", type=Path)
    parser.add_argument("--from-work", type=Path)
    parser.add_argument("--checks-only", action="store_true")
    args = parser.parse_args(argv)
    d.progress("start_no_model_load")
    try:
        if args.date != "20261116":
            raise ValueError("run_date")
        if args.cold_replay:
            print(json.dumps(replay(args.cold_replay)), flush=True)
            return 0
        if args.from_work and args.output.absolute().is_relative_to(d.ROOT):
            raise ValueError("private_work_requires_external_output")
        raw = args.output.absolute().parent / "raw" / d.NAME / "invocations" / str(time.time_ns())
        work = (
            json.loads(args.from_work.read_bytes())
            if args.from_work
            else measure(args.root, raw, args.date)
        )
        validation: Json = {}
        if not args.from_work and not args.checks_only:
            if (time.monotonic_ns() - work["start_ns"]) / 1e9 >= 3300:
                raise TimeoutError("measurement_deadline")
            d.progress("validation_before")
            validation = qualify(work)
            atomic_json(raw / "validation.json", validation)
            d.progress("validation_after")
            if (time.monotonic_ns() - work["start_ns"]) / 1e9 >= 4800:
                raise TimeoutError("hard_deadline")
        value = build(work, validation)
        atomic_json(raw / "candidate.json", value)
        if args.checks_only:
            print(json.dumps(dict(candidate=str(raw / "candidate.json"))), flush=True)
            return 0
        if not args.from_work and not value["required_checks_passed"]:
            d.progress("owned_validation_failed")
            return 1
        d.progress("publication_before")
        receipt = publish_primary(args.output, value, lambda path: terminal(path, raw / "terminal"))
        atomic_json(raw / "publication.json", receipt)
        print(json.dumps(dict(receipt, honest_verdict=value["honest_verdict"])), flush=True)
        d.progress("publication_after")
        return 0
    except (OSError, ValueError, KeyError, TypeError) as error:
        d.progress("refused:" + str(error))
        return 1
