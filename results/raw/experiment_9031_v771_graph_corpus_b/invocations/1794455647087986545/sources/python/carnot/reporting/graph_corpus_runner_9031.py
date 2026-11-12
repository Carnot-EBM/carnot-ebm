"""REQ-REPORT-9031-V771: freeze validation before one bounded B collection."""

import argparse
from concurrent.futures import ThreadPoolExecutor, wait
from dataclasses import replace
from datetime import datetime, UTC
import json
import os
from pathlib import Path
import subprocess
import tempfile
import time
from typing import Any
from unittest.mock import patch

from carnot.inference import graph_corpus_runtime_8961 as runtime
from carnot.reporting import graph_corpus_b_9031 as d
from carnot.reporting import graph_corpus_runner_8961 as shared
from carnot.reporting import graph_corpus_runner_8990 as original_runtime
from carnot.reporting import invocation_provenance_9000 as custody
from carnot.reporting import retained_shard_runner_9030 as validation_runner
from carnot.reporting import scoped_provenance_9014 as checks
from carnot.reporting import graph_source_8931 as source
from carnot.reporting import graph_corpus_a_8961 as original_shard
from carnot.reporting.current_work_receipt import atomic_json
from carnot.reporting.experiment_7303_validation_scope import CommandSpec
from carnot.reporting.primary_publication import publish_primary
from carnot.reporting.v756_branch_qualification import measured

Json = dict[str, Any]
DIRECT = checks.DIRECT


def plan(private: Path) -> list[CommandSpec]:
    """Reuse every frozen direct consumer; only new statements need new coverage."""
    with patch.object(validation_runner, "d", d):
        commands: list[CommandSpec] = validation_runner.plan(private)
    return [replace(c, timeout_s=600 if c.name == "owned_tests" else c.timeout_s) for c in commands]


def preflight(raw: Path) -> Json:
    """An unreadable GPU health query is unknown and cannot authorize a load."""
    try:
        return original_runtime.preflight(raw)
    except subprocess.SubprocessError as error:
        raise RuntimeError("gpu_health_unknown:" + str(error)) from error


def collect(units: list[Json], raw: Path) -> Json:
    """The parent reports real ledger counts while native loads or requests wait."""
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(runtime.run, units, raw, task_id=d.TASK)
        while True:
            done, _ = wait([future], timeout=30)
            if done:
                return future.result()
            path = raw / "invocation_ledger.json"
            rows = json.loads(path.read_bytes())["rows"] if path.is_file() else []
            completed = sum(
                r["operation"] == "generation" and r["status"] == "completed" for r in rows
            )
            source.progress("waiting_B_runtime", completed, len(units) - completed)


def freeze(root: Path, raw: Path, fixture: bool) -> Json:
    """Snapshot identities before checking prior responses or touching model weights."""
    source.progress("before_B_snapshot", 0, 512)
    work: Json = shared.freeze(root, raw, True, producer=d)
    work["environment"] = custody.environment(raw)
    work["inventory"] = d.inventory(root, work["units"], raw / "prior_B")
    work["extra_checks"] = [*work["environment"]["checks"], work["inventory"]["check"]]
    failures = [r for r in work["extra_checks"] if not r["passed"]]
    if not fixture and not work["authentication"]["failures"] and not failures:
        with patch.object(runtime, "preflight", preflight):
            work["runtime"] = collect(work["units"], raw / "runtime")
    elif not fixture:
        work["runtime"]["rows"] = [source.censored(u, "external_inputs") for u in work["units"]]
    work["runtime"]["checks"].extend(failures)
    atomic_json(raw / "runtime/runtime.json", work["runtime"])
    reduced = d.reduce(work["units"], work["runtime"]["rows"])
    for filename, key in [
        ("evaluation_labels.json", "label_sidecar"),
        ("detector_view.json", "detector_view"),
    ]:
        atomic_json(raw / filename, dict(rows=reduced[key]))
    work["raw_references"] = (
        [d.reference(p) for p in sorted((raw / "runtime").rglob("*")) if p.is_file()]
        + [
            d.reference(raw / name)
            for name in ["scheduled_units.json", "evaluation_labels.json", "detector_view.json"]
        ]
        + work["inventory"]["references"]
    )
    work["ended"] = time.monotonic()
    work["finished_at"] = datetime.now(UTC).isoformat()
    source.progress("after_B_collection", reduced["completed_count"], reduced["censored_count"])
    return work


def replay(path: Path) -> Json:
    """The established replay reconstructs responses and rejects rehashed summaries."""
    return shared.replay(path, producer=d)


def scanners(path: Path, raw: Path, scope: Json) -> Json:
    """Keep both original scanners and a fresh foreign-directory replay mandatory."""
    with patch.object(validation_runner, "d", d):
        return validation_runner.scanners(path, raw, scope)


def main(argv: list[str] | None = None) -> int:
    """Private controls cannot load a model; live publication requires owned acceptance."""
    os.environ.update(
        PYTHONUNBUFFERED="1", CARNOT_FORCE_LIVE="1", CUDA_VISIBLE_DEVICES="0", JAX_PLATFORMS="cpu"
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", choices=[d.RUN_DATE], default=d.RUN_DATE)
    parser.add_argument("--input-root", type=Path, default=d.ROOT)
    parser.add_argument("--fixture-e2e", type=Path)
    parser.add_argument("--cold-replay", type=Path)
    args = parser.parse_args(argv)
    if args.cold_replay:
        result = replay(args.cold_replay)
        print(json.dumps(result), flush=True)
        return int(not result["passed"])
    fixture = args.fixture_e2e is not None
    output = args.fixture_e2e or d.ROOT / "results" / (d.NAME + ".json")
    if fixture and output.resolve().is_relative_to(d.ROOT):
        parser.error("fixture writes require private scratch")
    raw = output.absolute().parent / "raw" / d.NAME / "invocations" / str(time.time_ns())
    raw.mkdir(parents=True)
    private = Path(tempfile.mkdtemp(prefix="carnot-9031-validation-"))
    commands = plan(private)
    scope = dict(
        frozen_ns=time.monotonic_ns(),
        owned_paths=d.OWNED,
        direct_consumer_tests=DIRECT,
        commands=[dict(name=c.name, argv=list(c.argv), timeout_s=c.timeout_s) for c in commands],
        required_check_names=checks.CHECKS,
        code_test_hashes=[
            d.reference(d.ROOT / p)
            for p in dict.fromkeys(
                [
                    *d.OWNED,
                    d.TEST,
                    *DIRECT,
                    *checks.REUSED,
                    *original_shard.OWNED,
                    original_runtime.__file__,
                    validation_runner.__file__,
                ]
            )
        ],
        excluded_statements=0,
        statement_percent=100,
        measurement_limit_s=3300,
        validation_reserve_s=1200,
        hard_cap_s=4800,
        scope="Current B producer and complete frozen direct consumers; historical failed health stays separate.",
    )
    atomic_json(raw / "validation_scope_manifest.json", scope)
    work = freeze(args.input_root, raw, fixture)
    validation_started = time.monotonic()
    validation: Json = dict(receipts=[], coverage={}, scope=scope)
    if not fixture:
        os.environ["COVERAGE_RCFILE"] = str(private / "coverage.ini")
        for index, command in enumerate(commands):
            source.progress("owned_validation", index, len(commands) - index)
            validation["receipts"].append(checks.run_check(command, raw / "validation", scope))
        validation["coverage"] = measured([private / ".coverage"], d.OWNED, raw / "measured")
        validation["coverage"].pop("rows", None)
    work["ended"] = time.monotonic()
    work["phase_spans"].append(
        dict(phase="owned_validation", duration_s=work["ended"] - validation_started)
    )
    work["finished_at"] = datetime.now(UTC).isoformat()
    atomic_json(raw / "work.json", work)
    atomic_json(raw / "validation.json", validation)
    value = d.build(work, validation, fixture)
    atomic_json(raw / "candidate.json", value)
    if not fixture and not value["required_checks_passed"]:
        source.progress("owned_checks_failed_primary_unchanged", 0, 1)
        return 1
    if fixture:
        publish_primary(output, value, replay)
    else:
        publish_primary(output, value, lambda p: scanners(p, raw / "terminal", scope))
    source.progress("terminal_published", value["completed_count"], value["censored_count"])
    print(
        json.dumps(
            dict(
                path=str(output),
                honest_verdict=value["honest_verdict"],
                shard_ready_score=value["shard_ready_score"],
            )
        ),
        flush=True,
    )
    return 0
