"""REQ-REPORT-9211-V784: qualify and publish human target identifiability and ambiguity bounds."""

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

from carnot.reporting import label_ambiguity_9211 as d
from carnot.reporting import logit_methods_runner_9126 as lifecycle
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
    """Freeze scope, rubric, protocol and preconditions before reducing human source."""
    start = time.monotonic_ns()
    raw.mkdir(parents=True, exist_ok=True, mode=0o700)
    raw.chmod(0o700)
    with patch.multiple(lifecycle, d=d, DIRECT=DIRECT):
        scope = lifecycle.scope(raw)
    scope["commands"][0]["argv"].insert(
        2, "COVERAGE_RCFILE=" + scope["coverage_config_reference"]["path"]
    )
    atomic_json(raw / "label_ambiguity_protocol.json", d.protocol())
    proto_ref = d.ref(raw / "label_ambiguity_protocol.json")
    rubric_payload = d.rubric()
    atomic_json(raw / "target_identifiability_rubric.json", rubric_payload)
    rubric_ref = dict(
        d.ref(raw / "target_identifiability_rubric.json"),
        version=rubric_payload["version"],
        targets=rubric_payload["targets"],
        statuses=rubric_payload["statuses"],
        decisions=rubric_payload["decisions"],
    )
    scope.update(
        random_seed=d.SEED,
        exclusions=[],
        owned_paths=d.OWNED,
        label_ambiguity_protocol_reference=proto_ref,
        target_rubric_reference=rubric_ref,
    )
    atomic_json(raw / "frozen_scope.json", scope)
    d.progress("resources_before")
    free_disk = shutil.disk_usage(raw).free
    limit = resource.getrlimit(resource.RLIMIT_AS)[0]
    cpus = sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else [0]
    tools = {
        name: importlib.util.find_spec(name) is not None
        for name in ("pytest", "coverage", "ruff", "mypy", "yaml", "numpy")
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
    masks_payload = dict(
        version="v784-label-ambiguity-masks-v1",
        role="evaluator_only_after_prediction_seal",
        rows=inspected["mask_rows"],
        support_counts=inspected["controls"].get("support_counts", {}),
    )
    atomic_json(raw / "ambiguity_masks.json", masks_payload)
    (raw / "ambiguity_masks.json").chmod(0o600)
    masks_ref = dict(
        d.ref(raw / "ambiguity_masks.json"),
        role="evaluator_only_after_prediction_seal",
        response_count=len(inspected["mask_rows"]),
        qid_count=len({m["qid"] for m in inspected["mask_rows"]}),
        observed_agreement_count=sum(
            m["a2_status"] == "observed_agreement" for m in inspected["mask_rows"]
        ),
        observed_disagreement_count=sum(
            m["a2_status"] == "observed_disagreement" for m in inspected["mask_rows"]
        ),
        missing_a2_count=sum(m["a2_status"] == "missing_a2" for m in inspected["mask_rows"]),
        scenarios=["observed_disagreement", "missing_annotation_envelope"],
    )
    ended = time.monotonic_ns()
    receipt = build_current_work_receipt(
        run_id=raw.name,
        owner_pid=os.getpid(),
        events=[],
        inference_substrate="cpu_cached_human_annotation_reduction",
        inference_substrate_details=dict(
            work="Independent SHA256/git-blob authentication, v2 taxonomy evidence audit and exact affine Brier endpoint controls; zero LLM loads, forwards or generations"
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
        label_ambiguity_protocol_reference=proto_ref,
        target_rubric_reference=rubric_ref,
        ambiguity_masks_reference=masks_ref,
        reducer_reference=dict(
            **d.ref(d.ROOT / "python/carnot/reporting/label_ambiguity_9211.py"),
            module="carnot.reporting.label_ambiguity_9211",
            entrypoints=["reduce_contrast_endpoints", "reduce_both_scenarios"],
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
    """Emit separate target-identifiability and label-ambiguity readiness with zero model work."""
    inspected = work["inspection"]
    blocked = not all(r["passed"] for r in inspected["checks"])
    passed = receipts_pass(work, validation)
    rows = inspected["rows"]
    trows = inspected["target_identifiability_rows"]
    ctrl = inspected.get("controls", {})
    endpoint_rows = ctrl.get("endpoint_control_rows", [])
    target_ctrl_rows = ctrl.get("target_control_rows", [])
    controls_ok = (
        bool(endpoint_rows)
        and all(r["passed"] for r in endpoint_rows)
        and bool(target_ctrl_rows)
        and all(r["passed"] for r in target_ctrl_rows)
    )
    target_audit_ok = len(rows) == 900 and len(trows) == 2700 and controls_ok
    cls = "blocked" if blocked else "null" if passed and target_audit_ok else "disqualified"
    completed_responses = len(rows)
    completed_qids = len({r["qid"] for r in rows})
    protocol_refs = dict(
        inspected.get("protocol_references", {}),
        label_ambiguity_protocol_reference=work["label_ambiguity_protocol_reference"],
        target_rubric_reference=work["target_rubric_reference"],
        ambiguity_masks_reference=work["ambiguity_masks_reference"],
        reducer_reference=work["reducer_reference"],
    )
    value: Json = dict(
        schema="carnot.v784_label_ambiguity.v1",
        experiment_id=9211,
        task_id=d.TASK,
        milestone="2026.11.784",
        run_date=work["run_date"],
        honest_verdict=f"complete_{cls}_v784_label_ambiguity",
        verdict_class=cls,
        target_identifiability_ready_score=int(passed and target_audit_ok and not blocked),
        label_ambiguity_ready_score=int(passed and controls_ok and not blocked),
        required_checks_passed=passed,
        flagged_adversarial=False,
        gate_check_summary=inspected["checks"],
        preconditions_checked=work["preconditions_checked"],
        rows=rows,
        intended_count=900,
        attempted_count=completed_responses,
        completed_count=completed_responses,
        failed_count=0,
        censored_count=900 - completed_responses,
        excluded_count=0,
        effective_independent_groups=completed_qids,
        sample_size_budget=dict(
            intended=900,
            attempted=completed_responses,
            completed=completed_responses,
            failed=0,
            censored=900 - completed_responses,
            effective_independent_groups=completed_qids,
            intended_qids=300,
            completed_qids=completed_qids,
            siblings_per_qid=3,
            repeated_retention_streams=10,
            repeated_retention_appearances=10 * sum(r["role"] == "TEST" for r in rows),
        ),
        verifier_is_oracle=False,
        independent_generalization_score=0,
        claim_scope=dict(
            exposure_scope="exposed_development_only",
            independent_generalization_score=0,
            allowed_claims=[
                "authenticated original LPHB 900-response custody",
                "v2 taxonomy target-identifiability evidence audit (zero identifiable SI/CF rows)",
                "exact affine Brier label-ambiguity endpoint reducer and 900-response masks",
            ],
            scope="Target identifiability and exact label-ambiguity bounds only; no LLM load, verifier benefit, learning effect or independent generalization.",
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
                condition="Independently authenticated Exp9097 900 responses/300 qids, 7 LPHB release blobs and Exp9126/9132 role/method receipts",
                principle="Historical evidence retains its original authority; missing external source blocks.",
            ),
            target_identifiability=dict(
                condition="900 responses x 3 v2 taxonomy targets audited before predictions; zero identifiable SI/CF rows reported honestly",
                principle="Missing category authority cannot become a measured class or guessed label.",
            ),
            label_ambiguity_method=dict(
                condition="Exact affine Brier contrast endpoints proven on analytic, exhaustive binary, permutation, sign and 10-stream shared-label controls",
                principle="Repeated measurements share one underlying label; fixture probabilities establish no verifier benefit.",
            ),
            validation=dict(
                condition="All 6 owned/affected checks pass with 100 percent owned unit+child statement coverage and zero exclusions",
                principle="Self-reported passing checks do not establish readiness.",
            ),
        ),
        protocol_references=protocol_refs,
        target_rubric_reference=work["target_rubric_reference"],
        target_identifiability_rows=trows,
        target_identifiability_summary=ctrl.get("target_summary", {}),
        target_identifiability_controls=target_ctrl_rows,
        ambiguity_masks_reference=work["ambiguity_masks_reference"],
        label_ambiguity_protocol_reference=work["label_ambiguity_protocol_reference"],
        reducer_reference=work["reducer_reference"],
        endpoint_control_rows=endpoint_rows,
        support_counts=ctrl.get("support_counts", {}),
        censor_reason="source_custody_blocked" if blocked else None,
        work_reference=d.ref(Path(work["raw"]) / "work.json"),
    )
    value["field_principles"] = {
        k: "Authenticated original human custody and exact CPU endpoint controls bind this field; zero current model work or empirical verifier benefit."
        for k in value
    }
    value["reproducibility_checksum"] = checksum(value)
    return value


def replay(path: Path) -> Json:
    """Recompute masks, target rows and controls in a fresh process; reject tampered sidecars."""
    value = json.loads(path.read_bytes())
    wref = value["work_reference"]
    if d.ref(Path(wref["path"])) != wref:
        raise ValueError("work_hash")
    work = json.loads(Path(wref["path"]).read_bytes())
    for ref_obj in [
        work["current_work_reference"],
        work["label_ambiguity_protocol_reference"],
        work["scope"]["coverage_config_reference"],
    ]:
        if d.ref(Path(ref_obj["path"])) != ref_obj:
            raise ValueError("operand_hash")
    rubric_ref = work["target_rubric_reference"]
    if d.ref(Path(rubric_ref["path"])) != {k: rubric_ref[k] for k in ("path", "sha256")}:
        raise ValueError("rubric_hash")
    mask_ref = work["ambiguity_masks_reference"]
    if d.ref(Path(mask_ref["path"])) != {k: mask_ref[k] for k in ("path", "sha256")}:
        raise ValueError("mask_hash")
    if (
        json.loads(Path(work["label_ambiguity_protocol_reference"]["path"]).read_bytes())
        != d.protocol()
    ):
        raise ValueError("protocol_drift")
    if json.loads(Path(rubric_ref["path"]).read_bytes()) != d.rubric():
        raise ValueError("rubric_drift")
    fresh = d.inspect(Path(work["root"]))
    original = work["inspection"]
    if fresh != dict(original, checks=original["checks"][:-4]):
        raise ValueError("source_drift")
    expected_masks = dict(
        version="v784-label-ambiguity-masks-v1",
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
    parser.add_argument("--date", default="20261118")
    parser.add_argument("--root", type=Path, default=d.ROOT)
    parser.add_argument("--output", type=Path, default=d.ROOT / "results" / (d.NAME + ".json"))
    parser.add_argument("--cold-replay", type=Path)
    parser.add_argument("--from-work", type=Path)
    parser.add_argument("--checks-only", action="store_true")
    args = parser.parse_args(argv)
    d.progress("start_no_model_load")
    try:
        if args.date != "20261118":
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
