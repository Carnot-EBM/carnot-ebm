"""REQ-REPORT-8200: freeze and validate observed demand without loading a model."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone, UTC
import json
import os
from pathlib import Path
import tempfile
import time
from typing import Any

from carnot.verify import request_trace_8200 as e
from carnot.reporting import request_trace_inventory_8200 as custody
from carnot.reporting.current_work_receipt import ZERO_INVOCATION_COUNTS, canonical_hash
from carnot.reporting.exact_request_execution_8188 import execute
from carnot.reporting.experiment_7303_validation_scope import CommandSpec, build_scoped_commands
from carnot.reporting.primary_publication import publish_primary

Json = dict[str, Any]
MODULES = [
    "python/carnot/verify/request_trace_8200.py",
    "python/carnot/reporting/request_trace_inventory_8200.py",
    "python/carnot/reporting/request_trace_execution_8200.py",
]
TEST = "tests/python/test_request_trace_8200.py"


def validation_plan(private: Path) -> list[CommandSpec]:
    """Explicit files prevent unrelated coverage debt from changing readiness."""
    plan = build_scoped_commands(
        e.ROOT,
        [
            TEST,
            "tests/python/test_source_boundary_7852.py",
            "tests/python/test_experiment_7942_v689_sentence_labels.py",
        ],
        MODULES,
        static_paths=[e.CLI],
        basetemp=private,
        coverage_file=private / ".coverage",
    )
    result = []
    for spec in plan:
        argv = spec.argv
        argv = tuple(a + ",*/" + e.CLI if a.startswith("--include=") else a for a in argv)
        if spec.name == "changed_module_mypy":
            argv += ("--strict", "--follow-imports=silent")
        result.append(CommandSpec(spec.name, argv, spec.scope, 900))
    return result


def validators(path: Path) -> list[CommandSpec]:
    """Unmodified auditors and a fresh process inspect the same candidate bytes."""
    py = str(e.ROOT / ".venv/bin/python")
    return [
        CommandSpec(
            "cold_replay",
            (py, "-u", str(e.ROOT / e.CLI), "--cold-replay", str(path)),
            "terminal",
            60,
        ),
        CommandSpec(
            "adversarial",
            (py, str(e.ROOT / "scripts/adversarial_verify.py"), "--json", str(path)),
            "terminal",
            60,
        ),
        CommandSpec(
            "strict_rows",
            (py, str(e.ROOT / "scripts/verdict_row_consistency_lint.py"), "--strict", str(path)),
            "terminal",
            60,
        ),
    ]


def run_checks(plan: list[CommandSpec], raw: Path) -> list[Json]:
    """Reuse qualified subprocess handling and seal completed logs once."""
    receipts = execute(plan, raw)
    for row in receipts:
        row.update(
            expected_exit=0,
            actual_exit=row["exit_code"],
            normal_exit=row["exit_code"] >= 0 and not row.get("timed_out", False),
        )
        Path(row["log_path"]).chmod(0o444)
    return receipts


def build(
    data: Json,
    result: Json,
    raw: Path,
    receipts: list[Json],
    date: str,
    duration: float,
    fixture: bool,
) -> Json:
    """Trace readiness measures support, never generalization or production reuse."""
    checks = [
        custody.operand(
            "reconstructable_request_count",
            raw / "inventory.json",
            96,
            result["completed_count"],
            ">=",
        ),
        custody.operand(
            "independent_source_count",
            raw / "inventory.json",
            24,
            result["independent_count"],
            ">=",
        ),
        custody.operand(
            "single_qualified_schema", raw / "inventory.json", True, result["supported_schema"]
        ),
        *data["checks"],
    ]
    passed = all(r["passed"] and r["normal_exit"] for r in receipts)
    ready = result["ready"] and all(c["passed"] for c in checks) and passed
    failed = next((c["check"] for c in checks if not c["passed"]), "none")
    verdict = (
        "complete_circular_positive_request_trace_fixture"
        if fixture
        else "complete_positive_request_trace_frozen"
    )
    cls = "circular_positive" if fixture else "positive"
    if not ready:
        verdict, cls = "complete_blocked_" + failed, "blocked"
    if not passed:
        verdict, cls = "complete_disqualified_owned_validation", "disqualified"
    trace = e.reference(raw / "trace.json")
    now = datetime.now(UTC).isoformat()
    value = dict(
        result,
        experiment=8200,
        experiment_id=8200,
        task_id="exp8200-request-trace-census",
        schema="carnot.request_trace_census.v1",
        run_date=date,
        started_at=now,
        finished_at=now,
        duration_s=duration,
        status="completed",
        title="Observed research request trace census",
        honest_verdict=verdict,
        verdict_class=cls,
        request_trace_ready_score=int(ready),
        verifier_is_oracle=fixture,
        claim_scope="exposed research request census; no service speed claim",
        exposure_scope="private oracle fixtures"
        if fixture
        else "exposed research development ledgers",
        independent_generalization_score=0,
        generalized_learning_benefit_score=0,
        required_checks_passed=passed,
        flagged_adversarial=False,
        validation_receipts=receipts,
        terminal_validation_sidecar_path=str(raw / "terminal_validation.json"),
        preconditions_checked=True,
        gate_check_summary=checks,
        inference_substrate="aggregation_from_upstream_artifacts",
        inference_substrate_class="no_model_load",
        MODEL_SPECS=[],
        model_specs=[],
        trained_head_specs=[],
        model_invocation_counts=dict(ZERO_INVOCATION_COUNTS),
        call_ledger=[],
        cited_upstream_artifacts=data["citations"],
        inventory_rows=data["inventory_rows"],
        trace_path=trace["path"],
        trace_sha256=trace["sha256"],
        primitive_inventory_path=str(raw / "inventory.json"),
        intended_count=96,
        censored_count=0,
        failed_count=0,
        sample_size_budget=dict(requests=96, independent_source_clusters=24, current_model_calls=0),
        random_seed=7088200,
        source_artifact_hashes=data["refs"],
        raw_shard_hashes=[
            e.reference(raw / n)
            for n in ("inventory.json", "trace.json", "validation_commands.json")
        ],
        code_config_hashes=[
            custody.copy_bytes(e.ROOT / p, raw / "code") for p in [*MODULES, e.CLI]
        ],
        phase_spans=[],
        acceptance_gates=dict(requests=96, sources=24, single_schema=e.SCHEMA),
        workload_scope="research_workload",
        deployment_workload=dict(authenticated_sample_available=False),
        benchmark_gate=None,
        benchmark_performed=False,
        nfr01_met=False,
        imported_model_provenance=dict(model="unsloth/Qwen3.8-27B-GGUF", current_calls=0),
        field_principles=dict(
            readiness="Support and normal owned validation are required.",
            identity="Different seeds or runtime bytes are different requests.",
            scope="Controlled runtime replay is not production demand.",
            independence="Requests and sentences are not independent source clusters.",
        ),
    )
    value["rows"] = [
        dict(
            unit_id=r["call_id"],
            source_cluster_id=r["source_cluster_id"],
            arm="observed",
            condition="original",
            metric="exact_repeat",
            numerator=int(d["original_distance"] is not None),
            denominator=1,
            status="completed",
            exclusion_reason=None,
        )
        for r, d in zip(result["requests"], result["reuse_distance_rows"], strict=True)
    ]
    value["reproducibility_checksum"] = canonical_hash(
        dict(trace=trace, code=value["code_config_hashes"])
    )
    return value


def main(argv: list[str] | None = None) -> int:
    """Finish owned work even when upstream chronology blocks service measurement."""
    os.environ["PYTHONUNBUFFERED"] = "1"
    started = time.monotonic()
    started_at = datetime.now(UTC).isoformat()
    e.progress("start", 0, 1)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", choices=["20261006"], default="20261006")
    parser.add_argument("--root", type=Path, default=e.ROOT)
    parser.add_argument("--output", type=Path, default=e.ROOT / "results" / (e.NAME + ".json"))
    parser.add_argument("--private-input", type=Path)
    parser.add_argument("--cold-replay", type=Path)
    parser.add_argument("--repository-health-receipt", type=Path)
    args = parser.parse_args(argv)
    if (
        args.root == e.ROOT
        and not args.repository_health_receipt
        and os.environ.get("CARNOT_8200_REPOSITORY_HEALTH_RECEIPT")
    ):
        args.repository_health_receipt = Path(os.environ["CARNOT_8200_REPOSITORY_HEALTH_RECEIPT"])
    if args.cold_replay:
        passed = e.replay(args.cold_replay)
        e.progress("cold_replay_after", int(passed), 0)
        return 0 if passed else 1
    fixture = args.private_input is not None
    output = args.output.absolute()
    if fixture and output.is_relative_to(e.ROOT / "results"):
        parser.error("private fixture output must be outside results")
    raw = output.parent / "raw" / output.stem / "invocations" / str(time.time_ns())
    raw.mkdir(parents=True, mode=0o700)
    private = Path(tempfile.mkdtemp(prefix="carnot-8200-"))
    os.environ["COVERAGE_FILE"] = str(private / ".coverage.repository")
    plan = validation_plan(private)
    candidate = private / (e.NAME + ".json")
    health = CommandSpec(
        "repository_health_once",
        (str(e.ROOT / ".venv/bin/pytest"), "tests/python", "-q"),
        "repository_health",
        3600,
    )
    e.seal(
        raw / "validation_commands.json",
        dict(
            owned=[asdict(c) for c in plan],
            terminal=[asdict(c) for c in validators(candidate)],
            repository_health=asdict(health),
            frozen_before_measurement=True,
        ),
    )
    e.progress("preconditions_before", 0, 1)
    if fixture:
        source = args.private_input
        checks = [custody.operand("private_input_exists", source, True, source.is_file())]
        data: Json = dict(
            records=[], replay_identity={}, inventory_rows=[], refs=[], checks=checks, citations=[]
        )
        if source.is_file():
            ref = custody.copy_bytes(source, raw)
            data.update(json.loads(Path(ref["frozen_path"]).read_text()))
            data["refs"] = [ref]
        e.seal(
            raw / "inventory.json",
            dict(records=data["records"], replay_identity=data["replay_identity"]),
        )
    else:
        authority = custody.authorities(args.root, raw)
        paths = [
            str(args.root / t["deliverable"])
            for t in authority["tasks"]
            if (args.root / t["deliverable"]).is_file()
            and json.loads((args.root / t["deliverable"]).read_text()).get("call_ledger")
        ]
        summaries = run_checks(
            [
                CommandSpec(
                    "upstream_summaries",
                    (
                        str(e.ROOT / ".venv/bin/python"),
                        str(e.ROOT / "scripts/summarize_artifact.py"),
                        *paths,
                    ),
                    "input_diagnosis",
                    1200,
                )
            ]
            if paths
            else [],
            raw / "summaries",
        )
        data = custody.inventory(args.root, raw, authority)
        e.seal(raw / "upstream_summary_receipts.json", dict(receipts=summaries))
        data["refs"].append(e.reference(raw / "upstream_summary_receipts.json"))
        for receipt in summaries:
            data["checks"].append(
                custody.operand(
                    "upstream_summary_normal_exit",
                    Path(receipt["log_path"]),
                    True,
                    receipt["normal_exit"],
                )
            )
        manifest = args.root / "ops/exclusion_manifest.yaml"
        data["checks"].append(
            custody.operand("exclusion_manifest_exists", manifest, True, manifest.is_file())
        )
        if manifest.is_file():
            data["refs"].append(custody.copy_bytes(manifest, raw))
    data["checks"].append(
        custody.operand("private_writable_storage", raw, True, os.access(raw, os.W_OK))
    )
    e.progress("preconditions_after", 1, 0)
    receipts = [] if fixture else run_checks(plan, raw / "validation")
    e.progress("census_before", 0, len(data["records"]))
    result = e.census(data["records"], data["replay_identity"])
    e.seal(raw / "trace.json", result)
    e.progress("census_after", len(data["records"]), 0)
    value = build(data, result, raw, receipts, args.date, time.monotonic() - started, fixture)
    value["started_at"] = started_at
    if not fixture:
        if args.repository_health_receipt:
            while not args.repository_health_receipt.is_file():
                e.progress("waiting_repository_health", 0, 1)
                time.sleep(30)
            value["repository_health"] = json.loads(args.repository_health_receipt.read_text())[
                "receipts"
            ]
            for receipt in value["repository_health"]:
                ref = custody.copy_bytes(Path(receipt["log_path"]), raw / "health")
                receipt.update(log_path=ref["frozen_path"], log_sha256=ref["sha256"])
        else:
            value["repository_health"] = run_checks([health], raw / "health")
    e.seal(raw / "independent_reduction.json", e.census(data["records"], data["replay_identity"]))
    value["raw_shard_hashes"].append(e.reference(raw / "independent_reduction.json"))
    e.atomic_json(candidate, value)
    terminal = run_checks(validators(candidate), raw / "terminal")
    if not all(r["passed"] and r["normal_exit"] for r in terminal):
        value.update(
            honest_verdict="complete_disqualified_owned_validation",
            verdict_class="disqualified",
            request_trace_ready_score=0,
            required_checks_passed=False,
        )
    value["validation_receipts"] += terminal
    value["duration_s"] = time.monotonic() - started
    value["finished_at"] = datetime.now(UTC).isoformat()
    value["phase_spans"] = [dict(phase="owned_census_execution", duration_s=value["duration_s"])]
    if output.exists():
        custody.copy_bytes(output, raw / "preserved_primaries")

    def checked(path: Path) -> Json:
        """Publication checks exact locked bytes before readers can see them."""
        rows = run_checks(validators(path), raw / "publication")
        return dict(passed=all(r["passed"] and r["normal_exit"] for r in rows), receipts=rows)

    e.progress("publication_before", 0, 1)
    publication = publish_primary(output, value, checked)
    e.seal(
        raw / "terminal_validation.json",
        dict(
            publication=publication,
            normal_process_exit=True,
            required_checks_passed=value["required_checks_passed"],
        ),
    )
    e.progress("complete", value["completed_count"], 0)
    return 0
