"""REQ-REPORT-8968: reuse private validation and publication without historical mutation."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
import tempfile
import time
from pathlib import Path
from typing import Any

from carnot.reporting import complete_service_cost_8968 as d
from carnot.reporting import online_benefit_runner_8967 as shared
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash
from carnot.reporting.experiment_7303_validation_scope import CommandSpec, run_commands
from carnot.reporting.matched_corpus_runner_8949 import validation_plan
from carnot.reporting.primary_publication import publish_primary
from carnot.reporting.v756_branch_qualification import measured
from carnot.reporting.v748_collection_contract import namespace

Json = dict[str, Any]
runtime = namespace(shared, "service8968_runner")
runtime.d = d
runtime.HELPERS = shared.HELPERS + [
    "python/carnot/autoresearch/relational_graph_encoder.py",
    "python/carnot/autoresearch/graph_calibration_benchmark.py",
    "python/carnot/memory/online_calibration_8960.py",
    "python/carnot/reporting/online_benefit_runner_8967.py",
    "research-hardware-wishlist.md",
]


def evaluation(evidence: Json) -> tuple[Json, bool]:
    """Only this measurement path starts services; cold replay reduces retained clocks."""
    if evidence["failures"]:
        return d.empty_result(), False
    try:
        raw = Path(evidence["references"][0]["path"]).parents[2]
        result = d.measure(evidence, raw)
        return result, bool(result["failed_count"] or result["censored_count"])
    except (OSError, ValueError, KeyError, TypeError) as error:
        evidence["failures"].append(
            dict(
                upstream=d.TASK,
                path=str(d.ROOT / d.CLI),
                hash=d.ref(d.ROOT / d.CLI)["sha256"],
                check="owned_measurement",
                artifact_field="complete_process_census",
                expected="96 complete process units",
                observed=str(error),
            )
        )
        return d.empty_result(), True


def replay(path: Path) -> Json:
    """Rebuild headlines and gates from raw process receipts, even after rehashing."""
    d.progress("before_cold_replay")
    try:
        value = json.loads(path.read_bytes())
        unsigned = dict(value)
        checksum = unsigned.pop("reproducibility_checksum")
        d.check(canonical_hash(unsigned) == checksum, "checksum_drift")
        for i, operand in enumerate(
            value["source_artifact_hashes"]
            + value["code_config_hashes"]
            + value["raw_shard_hashes"]
        ):
            d.check(
                d.ref(Path(operand["path"]))["sha256"] == operand["sha256"], "retained_bytes_drift"
            )
            if operand.get("executed_source"):
                d.check(
                    d.ref(Path(operand["source_path"]))["sha256"] == operand["sha256"],
                    "executed_source_drift",
                )
            if i % 500 == 0:
                d.progress("cold_authenticate", i, 0)
        raw = Path(value["raw_shard_hashes"][0]["path"]).parent
        work = json.loads((raw / "work.json").read_bytes())
        validation = json.loads((raw / "validation.json").read_bytes())
        evidence = d.authenticate(raw / "inputs")
        d.check(
            {k: evidence[k] for k in ("failures", "references")} == work["evidence"], "input_drift"
        )
        rebuilt = d.reduce(raw)
        d.check(rebuilt == work["result"], "reduction_drift")
        if value["cost_rows_ready_score"]:
            d.check(shared.verify_measured(validation["coverage"], d.OWNED), "coverage_drift")
        passed = d.build(work, validation, value["fixture_only"]) == value
        d.progress("after_cold_replay", int(passed), int(not passed))
        return dict(passed=passed)
    except (OSError, ValueError, KeyError, TypeError):
        d.progress("cold_replay_refused", 0, 1)
        return dict(passed=False)


runtime.evaluation, runtime.replay = evaluation, replay


def main(argv: list[str] | None = None) -> int:
    """Keep measurements private until owned checks and semantic replay finish."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date")
    parser.add_argument("--private-output", type=Path)
    parser.add_argument("--sources", type=Path)
    parser.add_argument("--cold-replay", type=Path)
    args = parser.parse_args(argv)
    if args.cold_replay is not None:
        report = replay(args.cold_replay)
        print(json.dumps(report), flush=True)
        return 0 if report["passed"] else 1
    if args.date != "20261110":
        parser.error("this producer requires --date 20261110")
    fixture = args.private_output is not None
    output = args.private_output or d.ROOT / "results" / (d.NAME + ".json")
    if fixture and output.resolve().is_relative_to(d.ROOT):
        parser.error("private outputs must remain outside checkout")
    previous = json.loads(output.read_bytes()) if output.is_file() else {}
    raw = output.parent / "raw" / d.NAME / "invocations" / str(time.time_ns())
    work = runtime.freeze(args.sources or d.ROOT, raw)
    for span in work["phase_spans"]:
        if span["phase"] == "read_only_reduction":
            span["phase"] = "complete_service_processes"
    work["prior_terminal"] = {
        k: previous[k]
        for k in ("verdict_class", "honest_verdict", "gate_check_summary")
        if k in previous
    }
    validation: Json = dict(receipts=[], coverage={})
    checked = time.monotonic_ns()
    if not fixture:
        private = Path(tempfile.mkdtemp(prefix="carnot8968-validation-"))
        d.progress("before_owned_checks")
        validation["receipts"] = run_commands(
            d.ROOT,
            [
                replace(c, timeout_s=900) if c.name == "owned_unit_child" else c
                for c in validation_plan(private, producer=d)
            ],
            log_dir=raw / "validation",
            heartbeat_s=15,
        )
        if (private / ".coverage").is_file():
            validation["coverage"] = measured([private / ".coverage"], d.OWNED, raw / "measured")
        d.progress("after_owned_checks")
        d.progress("before_full_python_suite")
        debt = run_commands(
            d.ROOT,
            [
                CommandSpec(
                    "full_python_suite",
                    (str(d.ROOT / ".venv/bin/pytest"), "tests/python", "-q"),
                    "repository_health",
                    300,
                )
            ],
            log_dir=raw / "repository_health",
            heartbeat_s=15,
        )
        work["global_test_debt"] = dict(
            receipts=debt, affects_owned_readiness=False, repeated=False
        )
        d.progress("after_full_python_suite")
    work["ended"] = time.monotonic_ns()
    work["phase_spans"].append(
        dict(phase="validation_and_repository_health", duration_s=(work["ended"] - checked) / 1e9)
    )
    atomic_json(raw / "validation.json", validation)
    work["raw_references"] = [
        d.ref(p)
        for p in sorted(raw.rglob("*"))
        if p.is_file()
        and "sources" not in p.parts
        and "inputs" not in p.parts
        and p.name != "validation.json"
    ]
    atomic_json(raw / "work.json", work)
    value = d.build(work, validation, fixture)
    if not fixture:
        d.progress("before_terminal_checks", 0, 3)
        candidate = raw / "prospective.json"
        atomic_json(candidate, value)
        py = str(d.ROOT / ".venv/bin/python")
        commands = [
            CommandSpec(
                "cold_replay_cli",
                (
                    py,
                    "-u",
                    "-c",
                    "import os,subprocess,sys,tempfile; env=dict(os.environ); env.pop('PYTHONPATH',None); raise SystemExit(subprocess.run(sys.argv[1:],cwd=tempfile.gettempdir(),env=env,timeout=120).returncode)",
                    py,
                    "-u",
                    str(d.ROOT / d.CLI),
                    "--cold-replay",
                    str(candidate),
                ),
                "terminal",
                150,
            ),
            CommandSpec(
                "adversarial",
                (py, "scripts/adversarial_verify.py", "--json", str(candidate)),
                "terminal",
                60,
            ),
            CommandSpec(
                "strict_rows",
                (py, "scripts/verdict_row_consistency_lint.py", "--strict", str(candidate)),
                "terminal",
                60,
            ),
        ]
        terminal = run_commands(
            d.ROOT, commands, log_dir=raw / "terminal_validation", heartbeat_s=15
        )
        validation["receipts"].extend(terminal)
        work["raw_references"].extend(
            d.ref(p) for p in sorted((raw / "terminal_validation").glob("*.log"))
        )
        work["phase_spans"].append(
            dict(phase="terminal_checks", duration_s=(time.monotonic_ns() - work["ended"]) / 1e9)
        )
        work["ended"] = time.monotonic_ns()
        atomic_json(raw / "validation.json", validation)
        atomic_json(raw / "work.json", work)
        value = d.build(work, validation, fixture)
        d.progress("after_terminal_checks", 3, 0)
    d.check((time.monotonic_ns() - work["started"]) / 1e9 < 4200, "total_work_budget")
    d.progress("before_primary_publication")
    published = publish_primary(output, value, replay)
    d.progress("terminal_published", value["completed_count"], value["censored_count"])
    print(
        json.dumps(
            dict(
                **published,
                honest_verdict=value["honest_verdict"],
                cost_rows_ready_score=value["cost_rows_ready_score"],
                deployment_recommendation=value["deployment_recommendation"],
            )
        ),
        flush=True,
    )
    return 0
