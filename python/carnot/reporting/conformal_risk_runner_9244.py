"""REQ-REPORT-9244-V787: qualify and publish cluster-valid conformal risk, CA-OCRC tracking and Mycelium provenance."""

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

from carnot.reporting import conformal_risk_9244 as d
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
    """Freeze scope, conformal risk protocol, CA-OCRC protocol, Mycelium ledger and preconditions."""
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

    conformal_proto = d.conformal_risk_protocol()
    atomic_json(raw / "conformal_risk_protocol.json", conformal_proto)
    (raw / "conformal_risk_protocol.json").chmod(0o400)
    conformal_ref = dict(
        d.ref(raw / "conformal_risk_protocol.json"),
        version=conformal_proto["version"],
        delta=conformal_proto["delta"],
        confidence_level=conformal_proto["confidence_level"],
        alpha_risk_grid=conformal_proto["alpha_risk_grid"],
        primary_threshold=conformal_proto["primary_threshold"],
        bound_methods=conformal_proto["bound_methods"],
        cluster_unit=conformal_proto["cluster_unit"],
        cluster_aggregation_rule="aggregate accepted-error and acceptance indicators to independent qid cluster first (bounded in [0,1] per qid), then compute exact Clopper-Pearson on qid-level any-accepted-error and deterministic Waudby-Smith/Ramdas betting e-process upper confidence bounds over independent accepted qids; report unavailable at zero accepted qids",
    )

    ca_ocrc_proto = d.ca_ocrc_protocol()
    ca_ocrc_payload = dict(
        **ca_ocrc_proto,
        proxy_delay_seeds=inspected["ca_ocrc_seeds"],
        mask_rows=inspected["ca_ocrc_mask_rows"],
    )
    atomic_json(raw / "ca_ocrc_protocol.json", ca_ocrc_payload)
    (raw / "ca_ocrc_protocol.json").chmod(0o400)
    ca_ocrc_ref = dict(
        d.ref(raw / "ca_ocrc_protocol.json"),
        version=ca_ocrc_proto["version"],
        capacities=ca_ocrc_proto["capacities"],
        delays=ca_ocrc_proto["delays"],
        ca_ocrc_seed=ca_ocrc_proto["ca_ocrc_seed"],
        seed_prefix=ca_ocrc_proto["seed_prefix"],
        corrupted_feedback_epsilons=ca_ocrc_proto["corrupted_feedback_epsilons"],
        propensity_formula=ca_ocrc_proto["propensity_formula"],
        ipw_estimator_formula=ca_ocrc_proto["ipw_estimator_formula"],
        corrupted_envelope_formula=ca_ocrc_proto["corrupted_envelope_formula"],
        debiased_corrupted_formula=ca_ocrc_proto["debiased_corrupted_formula"],
        proxy_delay_seeds_sha256=canonical_hash(inspected["ca_ocrc_seeds"]),
        proxy_delay_seed_count=len(inspected["ca_ocrc_seeds"]),
        proxy_delay_seeds=inspected["ca_ocrc_seeds"],
    )

    prov_summary = inspected["provenance_summary"]
    ledger_payload = dict(
        version="v787-mycelium-provenance-ledger-v1",
        role="descriptive_and_shadow_only",
        summary=prov_summary,
        rows=inspected["ledger_rows"],
    )
    atomic_json(raw / "provenance_ledger.json", ledger_payload)
    (raw / "provenance_ledger.json").chmod(0o400)
    ledger_ref = dict(
        d.ref(raw / "provenance_ledger.json"),
        version="v787-mycelium-provenance-ledger-v1",
        response_count=len(inspected["ledger_rows"]),
        qid_count=len({r["qid"] for r in inspected["ledger_rows"]}),
        dataset_counts=prov_summary.get("dataset_counts", {}),
        generator_counts=prov_summary.get("generator_counts", {}),
        granularity_counts=prov_summary.get("granularity_counts", {}),
        typology_counts=prov_summary.get("typology_counts", {}),
        role_counts=prov_summary.get("role_counts", {}),
    )
    scope.update(
        random_seed=d.SEED,
        exclusions=[],
        owned_paths=d.OWNED,
        conformal_risk_protocol_reference=conformal_ref,
        ca_ocrc_protocol_reference=ca_ocrc_ref,
        provenance_ledger_reference=ledger_ref,
    )
    atomic_json(raw / "frozen_scope.json", scope)
    ended = time.monotonic_ns()
    receipt = build_current_work_receipt(
        run_id=raw.name,
        owner_pid=os.getpid(),
        events=[],
        inference_substrate="cpu_cached_human_annotation_reduction",
        inference_substrate_details=dict(
            work="Independent SHA256/git-blob authentication, qid-cluster conformal risk bounds, CA-OCRC capacity-constrained delayed-feedback IPW controls and Mycelium 900-response provenance ledger; zero LLM loads, forwards or generations"
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
        conformal_risk_protocol_reference=conformal_ref,
        ca_ocrc_protocol_reference=ca_ocrc_ref,
        provenance_ledger_reference=ledger_ref,
        reducer_reference=dict(
            **d.ref(d.ROOT / "python/carnot/reporting/conformal_risk_9244.py"),
            module="carnot.reporting.conformal_risk_9244",
            entrypoints=[
                "clopper_pearson_upper",
                "betting_e_process_upper",
                "cluster_conformal_risk_bounds",
                "simulate_cluster_coverage",
                "ca_ocrc_draw",
                "ca_ocrc_track_stream",
                "build_provenance_ledger",
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
    """Emit cluster-valid conformal risk, CA-OCRC and Mycelium provenance readiness with zero model work."""
    inspected = work["inspection"]
    blocked = not all(r["passed"] for r in inspected["checks"])
    passed = receipts_pass(work, validation)
    rows = inspected["rows"]
    ctrl = inspected.get("controls", {})
    coverage_rows = ctrl.get("coverage_control_rows", [])
    edge_rows = ctrl.get("edge_control_rows", [])
    ca_ocrc_rows = ctrl.get("ca_ocrc_control_rows", [])
    prov_summary = inspected.get("provenance_summary", {})
    conformal_ok = (
        bool(coverage_rows)
        and all(r["passed"] for r in coverage_rows)
        and bool(edge_rows)
        and all(r["passed"] for r in edge_rows)
    )
    ca_ocrc_ok = (
        len(inspected.get("ca_ocrc_mask_rows", [])) == 150
        and bool(ca_ocrc_rows)
        and all(r["passed"] for r in ca_ocrc_rows)
    )
    provenance_ok = (
        len(rows) == 900
        and len({r["qid"] for r in rows}) == 300
        and prov_summary.get("total_responses") == 900
        and prov_summary.get("total_qids") == 300
        and prov_summary.get("invented_span_labels") == 0
    )
    all_ok = conformal_ok and ca_ocrc_ok and provenance_ok
    cls = "blocked" if blocked else "null" if (passed and all_ok) else "disqualified"
    completed_responses = len(rows)
    completed_qids = len({r["qid"] for r in rows})
    protocol_refs = dict(
        inspected.get("protocol_references", {}),
        conformal_risk_protocol_reference=work["conformal_risk_protocol_reference"],
        ca_ocrc_protocol_reference=work["ca_ocrc_protocol_reference"],
        provenance_ledger_reference=work["provenance_ledger_reference"],
        reducer_reference=work["reducer_reference"],
    )
    value: Json = dict(
        schema="carnot.v787_conformal_risk_protocol.v1",
        experiment_id=9244,
        task_id=d.TASK,
        milestone="2026.11.787",
        run_date=work["run_date"],
        honest_verdict=f"complete_{cls}_v787_conformal_risk_protocol",
        verdict_class=cls,
        conformal_risk_ready_score=int(passed and conformal_ok and not blocked),
        ca_ocrc_ready_score=int(passed and ca_ocrc_ok and not blocked),
        provenance_audit_ready_score=int(passed and provenance_ok and not blocked),
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
            train_qids=150,
            cal_qids=50,
            test_qids=100,
            stream_qids=50,
            stream_responses=len(inspected.get("ca_ocrc_mask_rows", [])),
            streams=10,
            qids_per_stream=5,
        ),
        verifier_is_oracle=False,
        independent_generalization_score=0,
        claim_scope=dict(
            exposure_scope="exposed_development_only",
            independent_generalization_score=0,
            allowed_claims=[
                "authenticated Exp9097 human custody, Exp9126/9132 method receipts, Exp9211 target/ambiguity primary and Exp9225 selective-deferral/spline primary",
                "finite-sample qid-cluster-valid Clopper-Pearson and Waudby-Smith/Ramdas betting e-process accepted-risk bounds over independent accepted qids",
                "CA-OCRC capacity-constrained delayed-feedback IPW risk tracking for capacities C in [1,2,5] and corrupted-feedback flip-rate bounds for epsilon in [0.0,0.05]",
                "Mycelium 900-response provenance-typology-granularity ledger (answer_level present, span_level absent) across TRAIN150/CAL50/TEST100",
            ],
            scope="Qid-cluster conformal risk bounds, CA-OCRC capacity-constrained delayed-feedback IPW tracking and Mycelium provenance accounting only; no LLM load, verifier benefit, learning effect or independent generalization.",
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
                condition="Independently authenticated Exp9097 900 responses/300 qids, Exp9126/9132 method receipts, Exp9211 target/ambiguity primary and Exp9225 selective-deferral/spline primary",
                principle="Historical evidence retains its original authority; missing external source blocks.",
            ),
            conformal_risk_method=dict(
                condition="Qid-cluster-valid Clopper-Pearson and betting e-process upper bounds retain >=0.95 coverage across 2000 trials at rho>=0.7 while naive i.i.d. sibling bounds undercover, with monotone alpha_risk ordering and unavailable at zero accepted qids",
                principle="Finite-sample cluster-valid risk bounds qualify a descriptive audit method, not primary verifier gain.",
            ),
            ca_ocrc_method=dict(
                condition="CA-OCRC randomized proxy-delay IPW tracking for capacities C in [1,2,5] under 1-step and multi-step delays is unbiased vs biased drop-on-full tracking and brackets corrupted feedback for epsilon in [0.0,0.05]",
                principle="Capacity-constrained delayed-feedback IPW tracking qualifies a shadow protocol, not empirical learning gain.",
            ),
            provenance_ledger_method=dict(
                condition="Complete 900-response Mycelium provenance-typology-granularity ledger (answer_level present, span_level absent) across TRAIN150/CAL50/TEST100 with zero invented span labels or model judges",
                principle="Complete provenance-typology-granularity accounting must seal before reading TEST predictions.",
            ),
            validation=dict(
                condition="All 6 owned/affected checks pass with 100 percent owned unit+child statement coverage and zero exclusions",
                principle="Self-reported passing checks do not establish readiness.",
            ),
        ),
        protocol_references=protocol_refs,
        conformal_risk_protocol_reference=work["conformal_risk_protocol_reference"],
        ca_ocrc_protocol_reference=work["ca_ocrc_protocol_reference"],
        provenance_ledger_reference=work["provenance_ledger_reference"],
        reducer_reference=work["reducer_reference"],
        coverage_control_rows=coverage_rows,
        edge_control_rows=edge_rows,
        ca_ocrc_control_rows=ca_ocrc_rows,
        provenance_summary=prov_summary,
        support_counts=ctrl.get("support_counts", {}),
        censor_reason="source_custody_blocked" if blocked else None,
        work_reference=d.ref(Path(work["raw"]) / "work.json"),
    )
    value["field_principles"] = {
        k: "Authenticated original human custody, qid-cluster conformal risk bounds, CA-OCRC delayed-feedback IPW tracking and Mycelium provenance accounting bind this field; zero current model work or empirical verifier benefit."
        for k in value
    }
    value["reproducibility_checksum"] = checksum(value)
    return value


def replay(path: Path) -> Json:
    """Recompute conformal risk bounds, CA-OCRC tracking controls and Mycelium ledger in a fresh process."""
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
    conf_ref = work["conformal_risk_protocol_reference"]
    if d.ref(Path(conf_ref["path"])) != {k: conf_ref[k] for k in ("path", "sha256")}:
        raise ValueError("conformal_protocol_hash")
    ca_ref = work["ca_ocrc_protocol_reference"]
    if d.ref(Path(ca_ref["path"])) != {k: ca_ref[k] for k in ("path", "sha256")}:
        raise ValueError("ca_ocrc_protocol_hash")
    ledger_ref = work["provenance_ledger_reference"]
    if d.ref(Path(ledger_ref["path"])) != {k: ledger_ref[k] for k in ("path", "sha256")}:
        raise ValueError("provenance_ledger_hash")
    if json.loads(Path(conf_ref["path"]).read_bytes()) != d.conformal_risk_protocol():
        raise ValueError("conformal_protocol_drift")
    fresh = d.inspect(Path(work["root"]))
    original = work["inspection"]
    if fresh != dict(original, checks=original["checks"][:-4]):
        raise ValueError("source_drift")
    expected_ca = dict(
        **d.ca_ocrc_protocol(),
        proxy_delay_seeds=fresh["ca_ocrc_seeds"],
        mask_rows=fresh["ca_ocrc_mask_rows"],
    )
    if json.loads(Path(ca_ref["path"]).read_bytes()) != expected_ca:
        raise ValueError("ca_ocrc_protocol_drift")
    expected_ledger = dict(
        version="v787-mycelium-provenance-ledger-v1",
        role="descriptive_and_shadow_only",
        summary=fresh["provenance_summary"],
        rows=fresh["ledger_rows"],
    )
    if json.loads(Path(ledger_ref["path"]).read_bytes()) != expected_ledger:
        raise ValueError("provenance_ledger_drift")
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
    parser.add_argument("--date", default="20261120")
    parser.add_argument("--root", type=Path, default=d.ROOT)
    parser.add_argument("--output", type=Path, default=d.ROOT / "results" / (d.NAME + ".json"))
    parser.add_argument("--cold-replay", type=Path)
    parser.add_argument("--from-work", type=Path)
    parser.add_argument("--checks-only", action="store_true")
    args = parser.parse_args(argv)
    d.progress("start_no_model_load")
    try:
        if args.date != "20261120":
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
