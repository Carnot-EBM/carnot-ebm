"""REQ-REPORT-9132-V778: publish frozen logit methods after measured validation.

Only mechanics readiness is tested here. Human future-benefit bounds stay
unmeasured, and historical model activity never enters the empty current ledger.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import resource
import shutil
import time
from typing import Any
from unittest.mock import patch

import numpy as np

from carnot.memory import human_online_9099 as learner
from carnot.reporting import logit_methods_runner_9126 as base
from carnot.reporting import online_methods_9132 as d
from carnot.reporting import online_methods_runner_9099 as original
from carnot.reporting.current_work_receipt import (
    atomic_json,
    build_current_work_receipt,
    canonical_hash,
    validate_current_work_receipt,
)
from carnot.reporting.primary_publication import publish_primary

Json = dict[str, Any]
CHECKS = [*base.CHECKS, "e2e018", "e2e020", "e2e021"]
DIRECT = [
    *base.DIRECT,
    "tests/python/test_logit_methods_9126.py",
    "tests/python/test_spline_basis_methods_8986.py",
]


def scope(raw: Path) -> Json:
    """Freeze exact checks and instrument real children, including os._exit, without exclusions."""
    with patch.multiple(base, d=d, CHECKS=CHECKS, DIRECT=DIRECT):
        manifest = base.scope(raw)
    cfg = raw / "coverage.ini"
    cfg.write_text(cfg.read_text().replace("patch = subprocess", "patch = subprocess, _exit"))
    manifest["coverage_config_reference"] = d.ref(cfg)
    command = manifest["commands"][0]["argv"]
    command.insert(1, "COVERAGE_RCFILE=" + str(cfg))
    command.append("tests/python/test_online_methods_9099.py")
    py = str(d.ROOT / ".venv/bin/python")
    for name, test in [
        ("e2e018", "test_experiment_7891_v685_authority_lifecycle.py"),
        ("e2e020", "test_hard_exit_learning_qualification_8206.py"),
        ("e2e021", "test_restricted_decision_audit_8210.py"),
    ]:
        manifest["commands"].append(
            dict(
                name=name,
                argv=[
                    py,
                    "-m",
                    "pytest",
                    "-n",
                    "0",
                    "-o",
                    "addopts=",
                    "--no-cov",
                    "-q",
                    "tests/python/" + test,
                ],
            )
        )
        manifest["code_test_hashes"].append(d.ref(d.ROOT / "tests/python" / test))
    manifest["code_test_hashes"].extend(
        d.ref(d.ROOT / path)
        for path in [
            d.PROTOCOL,
            "tests/python/test_online_methods_9099.py",
            "python/carnot/autoresearch/spline_basis_8986.py",
            "python/carnot/reporting/online_methods_9099.py",
            "python/carnot/reporting/primary_publication.py",
            "python/carnot/reporting/current_work_receipt.py",
            "scripts/adversarial_verify.py",
            "scripts/verdict_row_consistency_lint.py",
        ]
    )
    manifest["e2e_scope"] = (
        "E2E-018/020/021 original assertions plus private actual six-arm CLI success/refusal/tamper/cold replay outside checkout without PYTHONPATH"
    )
    return manifest


def receipts_pass(work: Json, validation: Json) -> bool:
    """Require matching real command receipts and complete SQLite statement membership."""
    with patch.multiple(base, d=d, CHECKS=CHECKS):
        return base.receipts_pass(work, validation)


def qualify(work: Json) -> Json:
    """Reuse supervised checks and keep the single full-suite run as separate global health."""
    with patch.object(base, "d", d):
        return base.qualify(work)


def measure(root: Path, raw: Path, date: str) -> Json:
    """Authenticate and freeze protocol, rosters and validation before numerical controls."""
    start = time.monotonic_ns()
    raw.mkdir(parents=True, exist_ok=True, mode=0o700)
    manifest = scope(raw)
    d.progress("authenticate_before")
    evidence = d.authenticate(root)
    disk = shutil.disk_usage(raw).free
    memory = resource.getrlimit(resource.RLIMIT_AS)[0]
    descriptors = resource.getrlimit(resource.RLIMIT_NOFILE)[0]
    checks = [
        d.logits.check(raw, "free_disk_bytes>=2GiB", True, disk >= 2 * 1024**3, "resource_disk"),
        d.logits.check(
            raw,
            "address_space>=256MiB",
            True,
            memory == resource.RLIM_INFINITY or memory >= 256 * 1024**2,
            "resource_memory",
        ),
        d.logits.check(
            raw, "file_descriptors>=128", True, descriptors >= 128, "resource_descriptors"
        ),
        d.logits.check(
            raw, "float64_bytes", 8, int(np.dtype(np.float64).itemsize), "resource_import"
        ),
    ]
    for check in checks:
        check["upstream"] = d.TASK
    evidence["gate_check_summary"].extend(checks)
    evidence["blocker"] = next(
        (c["check"] for c in evidence["gate_check_summary"] if not c["passed"]), None
    )
    references = {}
    for name, payload in [
        ("protocol", d.protocol()),
        ("rosters", evidence["roster"]),
        ("donor_map", evidence["roster"].get("donor_map", {})),
    ]:
        path = raw / (name + ".json")
        atomic_json(path, payload)
        references[name + "_reference"] = d.ref(path)
    d.progress("protocol_rosters_sealed")
    result = {} if evidence["blocker"] else d.evaluate(raw / "fixtures")
    changes = (
        {}
        if evidence["blocker"]
        else d.old.donor_changes(
            json.loads((root / d.old.AUTHORITY).read_bytes())["rows"], evidence["roster"]
        )
    )
    end = time.monotonic_ns()
    current = build_current_work_receipt(
        run_id=raw.name,
        owner_pid=os.getpid(),
        events=[],
        inference_substrate="cpu_numerical_methods_controls",
        inference_substrate_class="no_model_load",
        execution_venue="host",
        inference_substrate_details=dict(
            work="seven-feature coefficient and temperature fixtures; no human benefit measurement"
        ),
        started_monotonic_ns=start,
        ended_monotonic_ns=end,
        small_ebm_training=dict(
            performed=bool(result),
            scope="CPU convex initialization and delayed numerical controls only",
        ),
    )
    atomic_json(raw / "current_work.json", current)
    work = dict(
        root=str(root),
        raw=str(raw),
        run_date=date,
        scope=manifest,
        evidence=evidence,
        result=result,
        donor_changes=changes,
        current_work_reference=d.ref(raw / "current_work.json"),
        duration_s=(end - start) / 1e9,
        **references,
    )
    atomic_json(raw / "work.json", work)
    return work


def build(work: Json, validation: Json) -> Json:
    """A complete methods null requires no scientific win; affected failures disqualify readiness."""
    evidence, result = work["evidence"], work["result"]
    passed = receipts_pass(work, validation)
    blocked = evidence["blocker"] is not None
    mechanics = (
        bool(result)
        and result["exact_resume"]
        and all(row["passed"] for row in result["causal_controls"])
    )
    bad = bool(result) and not mechanics
    cls = (
        "disqualified"
        if bad or validation.get("checks") and not passed
        else "blocked"
        if blocked
        else "null"
    )
    contract = d.protocol()
    roster = evidence["roster"]
    rows = result.get("rows", [])
    value = dict(
        result,
        schema="carnot.online_logit_methods.v1",
        experiment_id=9132,
        task_id=d.TASK,
        milestone="2026.11.778",
        run_date=work["run_date"],
        honest_verdict="complete_" + cls + "_online_logit_methods",
        verdict_class=cls,
        online_methods_ready_score=int(passed and mechanics and not blocked),
        required_checks_passed=passed,
        flagged_adversarial=bool(bad),
        rows=rows,
        gate_check_summary=evidence["gate_check_summary"],
        intended_count=360,
        attempted_count=len(rows),
        completed_count=len(rows),
        failed_count=0,
        censored_count=360 - len(rows),
        excluded_count=0,
        sample_size_budget=dict(
            fixture_conditions=4,
            fixture_qids_per_condition=5,
            arms=6,
            siblings=3,
            initialization_qids=100,
            online_qids=50,
            streams=10,
            guard_qids=50,
            retention_qids=100,
            future_qids=40,
            paired_stream_draws=10000,
        ),
        effective_independent_groups=0,
        independent_generalization_score=0,
        verifier_is_oracle=False,
        claim_scope="Frozen delayed logit methods and exact CPU-fixture recovery only; human benefit unmeasured; no ARC level solve.",
        exposure_scope="Original LPHB300 qids/900 siblings are exposed development; cannot close GAP-ORACLE-DISTINCT or GAP-DETECTOR-AUROC-4208.",
        MODEL_SPECS=[],
        requested_inference_substrate_class="no_model_load",
        inference_substrate_class="no_model_load",
        inference_substrate="cpu_numerical_methods_controls",
        current_llm_invocations=0,
        model_invocation_counts=dict(
            model_loads=0, forward_calls=0, generations=0, embedding_calls=0, total=0
        ),
        generated_tokens=0,
        duration_s=work["duration_s"],
        random_seed=778132,
        online_protocol_reference=work["protocol_reference"],
        initial_roster_reference=work["rosters_reference"],
        donor_map_reference=work["donor_map_reference"],
        feature_basis_schema=contract["initialization"],
        features=contract["features"],
        seven_bound_specification=contract["benefit_bounds"],
        scientific_bounds_status="frozen_unmeasured",
        learning_signal_score=0,
        continuous_self_learning_task=True,
        temperature_gradient_reference_rows=result.get("temperature_gradient_reference_rows", []),
        causal_controls=result.get("causal_controls", []),
        exact_recovery_rows=result.get("exact_recovery_rows", []),
        donor_label_change_rows=work["donor_changes"].get("rows", []),
        donor_changed_label_fraction=work["donor_changes"].get("fraction"),
        source_references=evidence["source_references"],
        current_work_receipts=[work["current_work_reference"]],
        validation_receipts=validation.get("checks", []),
        owned_statement_coverage=validation.get("owned_statement_coverage", {}),
        repository_health=validation.get("repository_health", {}),
        code_config_hashes=work["scope"]["code_test_hashes"],
        validation_scope_manifest=work["scope"],
        acceptance_gates=dict(
            authenticated=not blocked,
            causal_mechanics=mechanics,
            exact_recovery=result.get("exact_resume", False),
            owned_validation=passed,
            human_bounds="unmeasured",
        ),
        censor_reason="upstream authentication or resources blocked" if blocked else None,
        work_reference=d.ref(Path(work["raw"]) / "work.json"),
        **{
            key: roster.get(key, [])
            for key in ["initial_group_roster", "stream_roster", "guard_roster", "retention_roster"]
        },
    )
    value["field_principles"] = original.principles(value)
    value["field_principles"].update(
        temperature_gradient_reference_rows="Independent scalar logloss derivatives fix the direction and optimizer equations for recalibration.",
        feature_basis_schema="Initialization-only imputation and direct fixed cubic features prevent static150 or future leakage.",
        seven_bound_specification="Seven simultaneous directional stream bounds prevent selective comparison against favorable controls.",
        donor_map_reference="Exact original initialization donors preserve causal label availability and sibling pairing.",
        initial_roster_reference="Original initialization100, stream50, CAL50 and retention100 identities prevent cohort substitution.",
        continuous_self_learning_task="Prediction-before-feedback defines future learning, with retention outside operational updates.",
        requested_inference_substrate_class="The authorized no-model substrate is fixed before resource checks or numerical work.",
        generated_tokens="No new language-model output is generated; historical cached answers receive no current token credit.",
        exact_recovery_rows="Complete state and prediction equality after real exit73 proves durable temperature and optimizer transactions.",
        scientific_bounds_status="Frozen bounds remain unmeasured until a separate human stream experiment supplies their operands.",
        random_seed="Seed778132 fixes synthetic controls; original donors keep776099 and paired stream bounds keep778135.",
    )
    value["reproducibility_checksum"] = canonical_hash(value)
    return value


def replay_value(value: Json) -> Json:
    """Rebuild trajectories, sources and all headlines rather than trust a rehashed summary."""
    for reference in [
        value["work_reference"],
        value["online_protocol_reference"],
        value["initial_roster_reference"],
        value["donor_map_reference"],
        *value["source_references"],
        *value["current_work_receipts"],
        *value["code_config_hashes"],
    ]:
        if d.ref(Path(reference["path"])) != reference:
            raise ValueError("operand_hash")
    work = json.loads(Path(value["work_reference"]["path"]).read_bytes())
    if json.loads(Path(work["protocol_reference"]["path"]).read_bytes()) != d.protocol():
        raise ValueError("protocol_drift")
    evidence = d.authenticate(Path(work["root"]))
    if (
        evidence["gate_check_summary"] != work["evidence"]["gate_check_summary"][:-4]
        or evidence["roster"] != work["evidence"]["roster"]
    ):
        raise ValueError("source_drift")
    if json.loads(Path(work["rosters_reference"]["path"]).read_bytes()) != evidence[
        "roster"
    ] or json.loads(Path(work["donor_map_reference"]["path"]).read_bytes()) != evidence[
        "roster"
    ].get("donor_map", {}):
        raise ValueError("roster_drift")
    changes = (
        {}
        if work["evidence"]["blocker"]
        else d.old.donor_changes(
            json.loads((Path(work["root"]) / d.old.AUTHORITY).read_bytes())["rows"],
            evidence["roster"],
        )
    )
    if changes != work["donor_changes"]:
        raise ValueError("donor_change_drift")
    if work["result"] and d.reduce(Path(work["raw"]) / "fixtures") != work["result"]:
        raise ValueError("fixture_reduction_drift")
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
    """Use unchanged artifact scanners and actual cold replay outside the checkout."""
    with patch.object(base, "d", d):
        return base.terminal(candidate, logs)


def main(argv: list[str] | None = None) -> int:
    """The public command needs no external validation files and never loads a model."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default="20261116")
    parser.add_argument("--root", type=Path, default=d.ROOT)
    parser.add_argument("--output", type=Path, default=d.ROOT / "results" / (d.NAME + ".json"))
    parser.add_argument("--cold-replay", type=Path)
    parser.add_argument("--checks-only", action="store_true")
    parser.add_argument("--worker", type=Path, nargs=3)
    parser.add_argument(
        "--crash",
        default="",
        choices=["", "before_prediction_fsync", "after_feedback_receipt", "during_atomic_replace"],
    )
    args = parser.parse_args(argv)
    d.progress("start_no_model_load")
    try:
        if args.date != "20261116":
            raise ValueError("run_date")
        if args.worker:
            request, directory, labels = args.worker
            learner.drive(json.loads(request.read_bytes()), directory, labels, crash=args.crash)
            return 0
        if args.cold_replay:
            print(json.dumps(replay_value(json.loads(args.cold_replay.read_bytes()))), flush=True)
            return 0
        raw = args.output.parent / "raw" / d.NAME / "invocations" / str(time.time_ns())
        work = measure(args.root, raw, args.date)
        validation = {} if args.checks_only else qualify(work)
        value = build(work, validation)
        candidate = raw / "candidate.json"
        atomic_json(candidate, value)
        if args.checks_only:
            print(json.dumps(dict(candidate=str(candidate))), flush=True)
            return 0
        atomic_json(raw / "validation.json", validation)
        if not value["required_checks_passed"]:
            d.progress("owned_validation_failed")
            return 1
        d.progress("publication_before")
        receipt = publish_primary(args.output, value, lambda path: terminal(path, raw / "terminal"))
        print(json.dumps(receipt), flush=True)
        d.progress("complete")
        return 0
    except (OSError, ValueError, KeyError, TypeError) as error:
        d.progress("refused:" + str(error))
        return 1
