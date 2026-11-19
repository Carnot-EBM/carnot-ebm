"""REQ-REPORT-9225-V785: qualify and publish selective-deferral masks and spline-support diagnostics."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import resource
import shutil
import time
from typing import Any
from unittest.mock import patch

from carnot.reporting import logit_methods_runner_9126 as lifecycle
from carnot.reporting import selective_deferral_9225 as d
from carnot.reporting.current_work_receipt import (
    atomic_json as atomic_json,
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
    """Digest all fields except reproducibility_checksum; replay also recomputes rows."""
    return canonical_hash({k: v for k, v in value.items() if k != "reproducibility_checksum"})


def measure(root: Path, raw: Path, date: str) -> Json:
    """Freeze scope, selective-deferral protocol, spline-support protocol and preconditions."""
    start = time.monotonic_ns()
    raw.mkdir(parents=True, exist_ok=True, mode=0o700)
    raw.chmod(0o700)
    with patch.multiple(lifecycle, d=d, DIRECT=DIRECT):
        scope = lifecycle.scope(raw)
    scope["commands"][0]["argv"].insert(
        2, "COVERAGE_RCFILE=" + scope["coverage_config_reference"]["path"]
    )
    d.progress("resources_before")
    free_disk = shutil.disk_usage(raw).free
    limit = resource.getrlimit(resource.RLIMIT_AS)[0]
    cpus = sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else [0]
    tools = {
        name: importlib.util.find_spec(name) is not None
        for name in ("pytest", "coverage", "ruff", "mypy", "yaml", "numpy", "scipy")
    }
    preconditions = [
        dict(
            resource="private_disk",
            available=free_disk >= 2 * 1024**3,
            check_receipt=dict(path=str(raw), free_bytes=free_disk, required_bytes=2 * 1024**3),
        ),
        dict(
            resource="memory",
            available=limit == resource.RLIM_INFINITY or limit >= 256 * 1024**2,
            check_receipt=dict(rlimit_as=limit, minimum_bytes=256 * 1024**2),
        ),
        dict(
            resource="cpu",
            available=bool(cpus),
            check_receipt=dict(cpu_count=len(cpus), affinity=cpus[:16]),
        ),
        dict(
            resource="required_tools",
            available=all(tools.values()) and lifecycle.np_itemsize() == 8,
            check_receipt=dict(tools=tools, numpy_float64_bytes=lifecycle.np_itemsize()),
        ),
    ]
    resource_checks = [
        d.check(raw, "free_disk_bytes>=2GiB", True, preconditions[0]["available"], "resource_disk"),
        d.check(
            raw,
            "address_space_at_least_256MiB",
            True,
            preconditions[1]["available"],
            "resource_memory",
        ),
        d.check(raw, "cpu_available", True, preconditions[2]["available"], "resource_cpu"),
        d.check(
            raw,
            "required_tools_and_float64",
            True,
            preconditions[3]["available"],
            "resource_tools",
        ),
    ]
    d.progress("resources_after")
    inspected = d.inspect(root)
    inspected["checks"].extend(resource_checks)

    deferral_proto = d.selective_deferral_protocol()
    atomic_json(raw / "selective_deferral_protocol.json", deferral_proto)
    (raw / "selective_deferral_protocol.json").chmod(0o400)
    proto_ref = dict(
        d.ref(raw / "selective_deferral_protocol.json"),
        version=deferral_proto["version"],
        seed_prefix=deferral_proto["seed_prefix"],
        alpha_grid=deferral_proto["alpha_grid"],
        regimes=deferral_proto["regimes"],
        escalation_interval=deferral_proto["escalation_interval"],
        audit_seeds_sha256=canonical_hash(inspected["audit_seeds"]),
        audit_seed_count=len(inspected["audit_seeds"]),
        audit_seeds=inspected["audit_seeds"],
    )

    spline_proto = d.spline_support_protocol()
    atomic_json(raw / "spline_support_protocol.json", spline_proto)
    (raw / "spline_support_protocol.json").chmod(0o400)
    spline_ref = dict(
        d.ref(raw / "spline_support_protocol.json"),
        version=spline_proto["version"],
        knot_vector=spline_proto["knot_vector"],
        basis_columns=spline_proto["basis_columns"],
        boundary_thresholds=spline_proto["boundary_thresholds"],
        strata=spline_proto["strata"],
        labels_read=False,
    )

    masks_payload = dict(
        version="v785-selective-deferral-masks-v1",
        role="evaluator_only_after_prediction_seal",
        rows=inspected["mask_rows"],
        support_counts=inspected["controls"].get("support_counts", {}),
    )
    atomic_json(raw / "selective_deferral_masks.json", masks_payload)
    (raw / "selective_deferral_masks.json").chmod(0o400)
    masks_ref = dict(
        d.ref(raw / "selective_deferral_masks.json"),
        role="evaluator_only_after_prediction_seal",
        response_count=len(inspected["mask_rows"]),
        qid_count=len({m["qid"] for m in inspected["mask_rows"]}),
        stream_count=len({m["stream_id"] for m in inspected["mask_rows"]}),
        alpha_grid=list(d.ALPHA_GRID),
        regimes=list(d.REGIMES),
    )
    scope.update(
        random_seed=d.SEED,
        exclusions=[],
        owned_paths=d.OWNED,
        selective_deferral_protocol_reference=proto_ref,
        spline_support_protocol_reference=spline_ref,
    )
    atomic_json(raw / "frozen_scope.json", scope)
    ended = time.monotonic_ns()
    receipt = build_current_work_receipt(
        run_id=raw.name,
        owner_pid=os.getpid(),
        events=[],
        inference_substrate="cpu_cached_human_annotation_reduction",
        inference_substrate_details=dict(
            work="Independent SHA256/git-blob authentication, selective-deferral Horvitz-Thompson IPW controls and label-free 57-column cubic B-spline support-density controls; zero LLM loads, forwards or generations"
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
        preconditions_checked=preconditions,
        inspection=inspected,
        current_work_reference=d.ref(raw / "current_work.json"),
        selective_deferral_protocol_reference=proto_ref,
        spline_support_protocol_reference=spline_ref,
        selective_deferral_masks_reference=masks_ref,
        classifier_reference=dict(
            **d.ref(d.ROOT / "python/carnot/reporting/selective_deferral_9225.py"),
            module="carnot.reporting.selective_deferral_9225",
            entrypoints=[
                "audit_draw",
                "observe_feedback",
                "estimate_selective_risk",
                "fit_feature_transform",
                "classify_spline_support",
            ],
        ),
        start_ns=start,
        duration_s=(ended - start) / 1e9,
    )
    atomic_json(raw / "work.json", work)
    d.progress("measurement_after", len(inspected["rows"]), 0)
    return work


def receipts_pass(work: Json, validation: Json) -> bool:
    """Require actual normal exits, log hashes and 100 percent owned statement coverage."""
    with patch.object(lifecycle, "d", d):
        return bool(lifecycle.receipts_pass(work, validation))


def qualify(work: Json) -> Json:
    """Run the frozen check union with supervised children and record repository health."""
    raw = Path(work["raw"])
    checks = [
        lifecycle.child(c["name"], c["argv"], raw / "validation", deadline=600, heartbeat=30)
        for c in work["scope"]["commands"]
    ]
    for item in checks:
        item["scope_sha256"] = canonical_hash(work["scope"])
    coverage = lifecycle.measured(
        sorted(raw.glob(".coverage.*")), d.OWNED, raw / "combined_coverage"
    )
    reused = work["inspection"].get("reused_health_receipt", {})
    health = dict(
        reused,
        scope="repository_health",
        note="Unrelated repository-wide suite recorded separately from owned/affected checks",
    )
    return dict(checks=checks, owned_statement_coverage=coverage, repository_health=health)


def build(work: Json, validation: Json) -> Json:
    """Emit selective-deferral and spline-support diagnostic readiness with zero model work."""
    inspected = work["inspection"]
    blocked = not all(r["passed"] for r in inspected["checks"])
    passed = receipts_pass(work, validation)
    rows = inspected["rows"]
    ctrl = inspected.get("controls", {})
    estimator_rows = ctrl.get("estimator_control_rows", [])
    spline_rows = ctrl.get("spline_control_rows", [])
    estimator_ok = bool(estimator_rows) and all(r["passed"] for r in estimator_rows)
    spline_ok = bool(spline_rows) and all(r["passed"] for r in spline_rows)
    masks_ok = len(rows) == 150 and len({r["qid"] for r in rows}) == 50
    all_ok = masks_ok and estimator_ok and spline_ok
    cls = "blocked" if blocked else "null" if (passed and all_ok) else "disqualified"
    completed_responses = len(rows)
    completed_qids = len({r["qid"] for r in rows})
    protocol_refs = dict(
        inspected.get("protocol_references", {}),
        selective_deferral_protocol_reference=work["selective_deferral_protocol_reference"],
        spline_support_protocol_reference=work["spline_support_protocol_reference"],
        selective_deferral_masks_reference=work["selective_deferral_masks_reference"],
        classifier_reference=work["classifier_reference"],
    )
    value: Json = dict(
        schema="carnot.v785_selective_deferral_protocol.v1",
        experiment_id=9225,
        task_id=d.TASK,
        milestone="2026.11.785",
        run_date=work["run_date"],
        honest_verdict=f"complete_{cls}_v785_selective_deferral_protocol",
        verdict_class=cls,
        selective_deferral_ready_score=int(passed and masks_ok and estimator_ok and not blocked),
        spline_support_diagnostic_ready_score=int(passed and spline_ok and not blocked),
        required_checks_passed=passed,
        flagged_adversarial=False,
        gate_check_summary=inspected["checks"],
        preconditions_checked=work["preconditions_checked"],
        rows=rows,
        intended_count=150,
        attempted_count=completed_responses,
        completed_count=completed_responses,
        failed_count=0,
        censored_count=150 - completed_responses,
        excluded_count=0,
        effective_independent_groups=completed_qids,
        sample_size_budget=dict(
            intended=150,
            attempted=completed_responses,
            completed=completed_responses,
            failed=0,
            censored=150 - completed_responses,
            effective_independent_groups=completed_qids,
            intended_qids=50,
            completed_qids=completed_qids,
            siblings_per_qid=3,
            streams=10,
            qids_per_stream=5,
            total_human_qids=300,
            total_human_responses=900,
            initial_qids=100,
            guard_qids=50,
            retention_qids=100,
        ),
        verifier_is_oracle=False,
        independent_generalization_score=0,
        claim_scope=dict(
            exposure_scope="exposed_development_only",
            independent_generalization_score=0,
            allowed_claims=[
                "authenticated Exp9097 human custody, Exp9126/9132 method receipts and Exp9211 target/ambiguity authority",
                "frozen 10-stream (50 qids / 150 responses) selective-deferral feedback masks and Horvitz-Thompson IPW risk controls",
                "label-free 57-column cubic B-spline support-density classifier and partition-of-unity controls",
            ],
            scope="Selective-deferral feedback masks, inverse-propensity estimators and label-free cubic B-spline support-density diagnostics only; no LLM load, verifier benefit, learning effect or independent generalization.",
        ),
        label_exposure_scope="exposed_development_only",
        exposure_scope="exposed_development_only",
        scientific_effect_measured=False,
        MODEL_SPECS=[],
        model_invocation_status="not_requested_empty_model_specs",
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
        source_references=inspected["source_references"],
        code_config_hashes=work["scope"]["code_test_hashes"],
        current_work_receipts=[work["current_work_reference"]],
        validation_receipts=validation.get("checks", []),
        owned_statement_coverage=validation.get("owned_statement_coverage", {}),
        repository_health=validation.get("repository_health", {}),
        validation_scope_manifest=work["scope"],
        acceptance_gates=dict(
            source_custody=dict(
                condition="Independently authenticated Exp9097 900 responses/300 qids, Exp9126/9132 method receipts and Exp9211 qualified target/ambiguity primary",
                principle="Historical evidence retains its original authority; missing external source blocks.",
            ),
            selective_deferral_method=dict(
                condition="Frozen 10-stream pre-outcome SHA256 audit draws, unbiased Horvitz-Thompson IPW estimators for alpha>0, and explicit alpha=0 unidentifiability",
                principle="Audited-deferral masks and inverse-propensity controls qualify a counterfactual method, not primary learning gain.",
            ),
            spline_support_method=dict(
                condition="Label-free 57-column cubic B-spline support-density classifier preserves exact partition-of-unity and deterministic strata under clipping, constants and monotone transforms",
                principle="Label-free knot-density strata must be sealed before reading TEST outcomes.",
            ),
            validation=dict(
                condition="All 6 owned/affected checks pass with 100 percent owned unit+child statement coverage and zero exclusions",
                principle="Self-reported passing checks do not establish readiness.",
            ),
        ),
        protocol_references=protocol_refs,
        selective_deferral_protocol_reference=work["selective_deferral_protocol_reference"],
        spline_support_protocol_reference=work["spline_support_protocol_reference"],
        selective_deferral_masks_reference=work["selective_deferral_masks_reference"],
        classifier_reference=work["classifier_reference"],
        estimator_control_rows=estimator_rows,
        spline_control_rows=spline_rows,
        support_counts=ctrl.get("support_counts", {}),
        censor_reason="source_custody_blocked" if blocked else None,
        work_reference=d.ref(Path(work["raw"]) / "work.json"),
    )
    value["field_principles"] = {
        k: "Authenticated original human custody, pre-outcome selective-deferral masks and label-free cubic B-spline support controls bind this field; zero current model work or empirical verifier benefit."
        for k in value
    }
    value["reproducibility_checksum"] = checksum(value)
    return value


def replay(path: Path) -> Json:
    """Recompute deferral masks, IPW estimators and spline controls in a fresh process."""
    value = json.loads(path.read_bytes())
    wref = value["work_reference"]
    if d.ref(Path(wref["path"])) != wref:
        raise ValueError("work_hash")
    work = json.loads(Path(wref["path"]).read_bytes())
    for ref_obj in [
        work["current_work_reference"],
        work["scope"]["coverage_config_reference"],
    ]:
        if d.ref(Path(ref_obj["path"])) != ref_obj:
            raise ValueError("operand_hash")
    proto_ref = work["selective_deferral_protocol_reference"]
    if d.ref(Path(proto_ref["path"])) != {k: proto_ref[k] for k in ("path", "sha256")}:
        raise ValueError("protocol_hash")
    spline_ref = work["spline_support_protocol_reference"]
    if d.ref(Path(spline_ref["path"])) != {k: spline_ref[k] for k in ("path", "sha256")}:
        raise ValueError("spline_protocol_hash")
    mask_ref = work["selective_deferral_masks_reference"]
    if d.ref(Path(mask_ref["path"])) != {k: mask_ref[k] for k in ("path", "sha256")}:
        raise ValueError("mask_hash")
    if json.loads(Path(proto_ref["path"]).read_bytes()) != d.selective_deferral_protocol():
        raise ValueError("protocol_drift")
    if json.loads(Path(spline_ref["path"]).read_bytes()) != d.spline_support_protocol():
        raise ValueError("spline_protocol_drift")
    fresh = d.inspect(Path(work["root"]))
    original = work["inspection"]
    if fresh != dict(original, checks=original["checks"][:-4]):
        raise ValueError("source_drift")
    expected_masks = dict(
        version="v785-selective-deferral-masks-v1",
        role="evaluator_only_after_prediction_seal",
        rows=fresh["mask_rows"],
        support_counts=fresh["controls"].get("support_counts", {}),
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
    """Run unchanged adversarial_verify, strict row lint and foreign-directory cold replay."""
    with patch.object(lifecycle, "d", d):
        return dict(lifecycle.terminal(candidate, logs))


def main(argv: list[str] | None = None) -> int:
    """Enforce date, private-work output boundary, monotonic deadlines and checked publication."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default="20261119")
    parser.add_argument("--root", type=Path, default=d.ROOT)
    parser.add_argument("--output", type=Path, default=d.ROOT / "results" / (d.NAME + ".json"))
    parser.add_argument("--cold-replay", type=Path)
    parser.add_argument("--from-work", type=Path)
    parser.add_argument("--checks-only", action="store_true")
    args = parser.parse_args(argv)
    d.progress("start_no_model_load")
    try:
        if args.date != "20261119":
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
