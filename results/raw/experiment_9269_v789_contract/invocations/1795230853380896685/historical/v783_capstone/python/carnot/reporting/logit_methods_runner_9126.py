"""REQ-REPORT-9126-V778: publish only replayable controls and measured validation.

A methods null is a completed protocol, not a scientific positive. Historical model
activity stays in source references and never enters this process's empty ledger.
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

from carnot.reporting import logit_methods_9126 as d
from carnot.reporting import mdh_human_release_runner_9056 as prior
from carnot.reporting.current_work_receipt import (
    atomic_json,
    build_current_work_receipt,
    canonical_hash,
)
from carnot.reporting.primary_publication import publish_primary
from carnot.reporting.v709_execution import child
from carnot.reporting.v756_branch_qualification import measured

Json = dict[str, Any]
CHECKS = [
    "owned_tests",
    "direct_consumers",
    "ruff_check",
    "ruff_format",
    "strict_mypy",
    "spec_coverage",
]
DIRECT = [
    "tests/python/test_primary_publication_7928.py",
    "tests/python/test_current_work_receipt.py",
]


def scope(raw: Path) -> Json:
    """Freeze exact commands and a no-exclusion coverage configuration before outcomes."""
    py = str(d.ROOT / ".venv/bin/python")
    cfg = raw / "coverage.ini"
    cfg.write_text(
        "[run]\nparallel = true\npatch = subprocess\ninclude =\n"
        + "".join("    " + str(d.ROOT / p) + "\n" for p in d.OWNED)
        + "[report]\nexclude_lines =\n"
    )
    paths = [*d.OWNED, d.TEST]
    commands = [
        dict(name=name, argv=argv)
        for name, argv in [
            (
                "owned_tests",
                [
                    "env",
                    "COVERAGE_FILE=" + str(raw / ".coverage"),
                    py,
                    "-m",
                    "coverage",
                    "run",
                    "--rcfile=" + str(cfg),
                    "-m",
                    "pytest",
                    "-n",
                    "0",
                    "-o",
                    "addopts=",
                    "--no-cov",
                    "-q",
                    d.TEST,
                ],
            ),
            (
                "direct_consumers",
                [py, "-m", "pytest", "-n", "0", "-o", "addopts=", "--no-cov", "-q", *DIRECT],
            ),
            ("ruff_check", [py, "-m", "ruff", "check", *paths]),
            ("ruff_format", [py, "-m", "ruff", "format", "--check", *paths]),
            ("strict_mypy", [py, "-m", "mypy", "--strict", "--follow-imports=silent", *d.OWNED]),
            ("spec_coverage", [py, "scripts/check_spec_coverage.py", d.TEST]),
        ]
    ]
    return dict(
        commands=commands,
        code_test_hashes=[d.ref(d.ROOT / p) for p in [*paths, *DIRECT]],
        frozen_ns=time.monotonic_ns(),
        required_check_names=CHECKS,
        statement_percent=100,
        excluded_statements=0,
        measurement_cap_s=3300,
        validation_reserve_s=1200,
        hard_cap_s=4800,
        coverage_config_reference=d.ref(cfg),
        e2e_scope="E2E-018/021 private actual CLI success/refusal/tamper/cold replay in owned tests; E2E-020 not applicable: no learning or recovery changes",
    )


def measure(root: Path, raw: Path, date: str) -> Json:
    """Authenticate and freeze label-free positions before running numerical controls."""
    began = time.monotonic_ns()
    raw.mkdir(parents=True, exist_ok=True, mode=0o700)
    manifest = scope(raw)
    d.progress("authenticate_no_model_load")
    inspected = d.inspect(root)
    disk = shutil.disk_usage(raw).free
    inspected["checks"].append(
        d.check(raw, "free_disk_bytes>=2GiB", True, disk >= 2 * 1024**3, "resource_disk")
    )
    inspected["checks"].append(
        d.check(raw, "numpy_float64_bytes", 8, np_itemsize(), "resource_import")
    )
    memory_limit = resource.getrlimit(resource.RLIMIT_AS)[0]
    inspected["checks"].append(
        d.check(
            raw,
            "address_space_at_least_256MiB",
            True,
            memory_limit == resource.RLIM_INFINITY or memory_limit >= 256 * 1024**2,
            "resource_memory",
        )
    )
    references = {}
    for name, payload in [
        ("protocol", d.protocol()),
        ("raw_sample_roster", dict(rows=inspected["raw_roster"])),
    ]:
        path = raw / (name + ".json")
        atomic_json(path, payload)
        references[name + "_reference"] = d.ref(path)
    d.progress("controls_before")
    controls = d.controls() if all(c["passed"] for c in inspected["checks"]) else []
    d.progress("controls_after")
    ended = time.monotonic_ns()
    receipt = build_current_work_receipt(
        run_id=raw.name,
        owner_pid=os.getpid(),
        events=[],
        inference_substrate="cpu_numerical_methods_controls",
        inference_substrate_details=dict(
            work="tiny full-vocabulary formula and gauge fixtures; no human fit or model access"
        ),
        inference_substrate_class="no_model_load",
        execution_venue="host",
        started_monotonic_ns=began,
        ended_monotonic_ns=ended,
    )
    atomic_json(raw / "current_work.json", receipt)
    work = dict(
        root=str(root),
        raw=str(raw),
        run_date=date,
        scope=manifest,
        inspection=inspected,
        controls=controls,
        current_work_reference=d.ref(raw / "current_work.json"),
        duration_s=(ended - began) / 1e9,
        **references,
    )
    atomic_json(raw / "work.json", work)
    return work


def np_itemsize() -> int:
    """Confirm the imported numerical substrate rather than claiming an inferred dtype."""
    import numpy as np

    return int(np.dtype(np.float64).itemsize)


def receipts_pass(work: Json, validation: Json) -> bool:
    """Reuse exact-command, log-hash and SQLite coverage checks from the existing publisher."""
    with patch.multiple(prior, e=d, CHECKS=CHECKS):
        return prior.receipts_pass(work["scope"], validation)


def qualify(work: Json) -> Json:
    """Run the frozen check union with supervised children and real child coverage."""
    raw = Path(work["raw"])
    checks = [
        child(c["name"], c["argv"], raw / "validation", deadline=600, heartbeat=30)
        for c in work["scope"]["commands"]
    ]
    for c in checks:
        c["scope_sha256"] = canonical_hash(work["scope"])
    coverage = measured(sorted(raw.glob(".coverage.*")), d.OWNED, raw / "combined_coverage")
    health = child(
        "global_python_suite",
        [str(d.ROOT / ".venv/bin/pytest"), "tests/python", "-q"],
        raw / "repository_health",
        deadline=600,
        heartbeat=30,
        scope="repository_health",
    )
    return dict(checks=checks, owned_statement_coverage=coverage, repository_health=health)


def build(work: Json, validation: Json) -> Json:
    """Derive readiness from authentication, rejecting controls and complete owned checks."""
    checks = work["inspection"]["checks"]
    blocked = not all(c["passed"] for c in checks)
    rows = work["controls"]
    controls_ok = bool(rows) and all(r["passed"] for r in rows)
    qualified = receipts_pass(work, validation)
    bad = not controls_ok and not blocked
    verdict = (
        "blocked"
        if blocked
        else "disqualified"
        if bad or (validation.get("checks") and not qualified)
        else "null"
    )
    value = dict(
        schema="carnot.logit_methods.v1",
        experiment_id=9126,
        task_id=d.TASK,
        milestone="2026.11.778",
        run_date=work["run_date"],
        honest_verdict="complete_" + verdict + "_whole_answer_methods",
        verdict_class=verdict,
        methods_ready_score=int(qualified and controls_ok and not blocked),
        required_checks_passed=qualified,
        flagged_adversarial=bool(bad),
        rows=rows,
        gate_check_summary=checks,
        intended_count=9,
        attempted_count=len(rows),
        completed_count=sum(r["passed"] for r in rows),
        failed_count=sum(not r["passed"] for r in rows),
        censored_count=9 - len(rows),
        sample_size_budget=dict(
            control_units=9,
            human_qids=300,
            human_responses=900,
            TRAIN_qids=150,
            CAL_qids=50,
            TEST_qids=100,
            raw_sample_qids=20,
            raw_sample_responses=60,
            raw_positions=8,
        ),
        effective_independent_groups=0,
        independent_generalization_score=0,
        verifier_is_oracle=False,
        claim_scope="Methods and synthetic controls only; no human advantage, ARC level solve, paper localization or independent confirmation.",
        exposure_scope="Original900 LPHB responses/300 qids are exposed development; cannot close GAP-ORACLE-DISTINCT or GAP-DETECTOR-AUROC-4208.",
        MODEL_SPECS=[],
        requested_inference_substrate_class="no_model_load",
        inference_substrate_class="no_model_load",
        inference_substrate="cpu_numerical_methods_controls",
        current_llm_invocations=0,
        model_invocation_counts=dict(model_loads=0, forward_calls=0, generations=0, total=0),
        generated_tokens=0,
        duration_s=work["duration_s"],
        random_seed=778126,
        protocol_reference=work["protocol_reference"],
        static_protocol_reference=work["protocol_reference"],
        collection_protocol_reference=work["protocol_reference"],
        token_statistic_schema=d.protocol()["schema"],
        raw_sample_roster_reference=work["raw_sample_roster_reference"],
        formula_reference_rows=[r for r in rows if r["condition"] == "independent_formula"],
        indexing_control_rows=[
            r for r in rows if r["condition"] in ["indexing_rejection", "chunk128"]
        ],
        gauge_control_rows=[r for r in rows if r["condition"] in ["constant", "alternating"]],
        five_arm_specification=d.protocol()["arms"],
        source_references=work["inspection"]["source_references"],
        current_work_receipts=[work["current_work_reference"]],
        validation_receipts=validation.get("checks", []),
        owned_statement_coverage=validation.get("owned_statement_coverage", {}),
        repository_health=validation.get("repository_health", {}),
        code_config_hashes=work["scope"]["code_test_hashes"],
        validation_scope_manifest=work["scope"],
        acceptance_gates=dict(
            authenticated=not blocked,
            controls=controls_ok,
            owned_validation=qualified,
            scientific_bounds=d.protocol()["bounds"],
            accepted_risk=d.protocol()["accepted_risk"],
        ),
        historical_provenance_only=True,
        energy_increment_score=0,
        cross_model_scoring=True,
        exact_span_replication=False,
        retired_null=d.protocol()["retired_null"],
        censor_reason="upstream authentication failed" if blocked else None,
        work_reference=d.ref(Path(work["raw"]) / "work.json"),
    )
    value["field_principles"] = {
        k: "Retained evidence binds this field; fixture methods establish no human benefit."
        for k in value
    }
    value["field_principles"].update(
        rows="One retained control per scheduled unit; no fixture is an independent human question.",
        source_references="Exact historical producer bytes preserve provenance without current inference credit.",
        gauge_control_rows="Separate normalized probability invariance from raw representation gauge sensitivity.",
        raw_sample_roster_reference="Lexical identities and positions precede labels, preventing favorable replay selection.",
        methods_ready_score="Formula correctness plus rejecting controls and owned validation; no favorable human outcome required.",
        current_work_receipts="Empty owned event ledger prevents historical model work from entering current counters.",
        gate_check_summary="Missing bytes, missing fields and observed zero remain distinct operands.",
        validation_receipts="Exact commands, durable logs and real SQLite statements prevent fabricated passes.",
        sample_size_budget="Siblings and bootstrap replicates never increase independent question support.",
        protocol_reference="Immutable prior choices prevent thresholds or denominators moving after outcomes.",
    )
    value["reproducibility_checksum"] = canonical_hash(value)
    return value


def replay_value(value: Json) -> Json:
    """Recompute primitives and gates so even rehashed aggregate tampering is refused."""
    for reference in [
        value["work_reference"],
        value["protocol_reference"],
        value["raw_sample_roster_reference"],
        *value["source_references"],
        *value["current_work_receipts"],
        *value["code_config_hashes"],
    ]:
        if d.ref(Path(reference["path"])) != reference:
            raise ValueError("operand_hash")
    work = json.loads(Path(value["work_reference"]["path"]).read_bytes())
    if json.loads(Path(work["protocol_reference"]["path"]).read_bytes()) != d.protocol():
        raise ValueError("protocol_drift")
    inspected = d.inspect(Path(work["root"]))
    if (
        inspected["checks"] != work["inspection"]["checks"][:-3]
        or inspected["raw_roster"] != work["inspection"]["raw_roster"]
    ):
        raise ValueError("source_drift")
    expected = d.controls() if all(c["passed"] for c in work["inspection"]["checks"]) else []
    if work["controls"] != expected:
        raise ValueError("control_drift")
    validation = dict(
        checks=value["validation_receipts"],
        owned_statement_coverage=value["owned_statement_coverage"],
        repository_health=value["repository_health"],
    )
    if build(work, validation) != value:
        raise ValueError("reduction_drift")
    return dict(passed=True, replay_passed=True)


def terminal(candidate: Path, logs: Path) -> Json:
    """Use the unchanged real scanners and foreign-directory replay with child heartbeats."""
    with patch.object(prior, "e", d):
        return prior.validate(candidate, logs)


def main(argv: list[str] | None = None) -> int:
    """Execute the required no-model command; private controls do not publish a primary."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default="20261116")
    parser.add_argument("--root", type=Path, default=d.ROOT)
    parser.add_argument("--output", type=Path, default=d.ROOT / "results" / (d.NAME + ".json"))
    parser.add_argument("--cold-replay", type=Path)
    parser.add_argument("--checks-only", action="store_true")
    args = parser.parse_args(argv)
    d.progress("start_no_model_load")
    try:
        if args.date != "20261116":
            raise ValueError("run_date")
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
