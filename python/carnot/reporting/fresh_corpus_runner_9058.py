"""REQ-REPORT-9058-V773: freeze checks before first-response collection and publication."""

import argparse
from functools import partial
import json
import os
from pathlib import Path
import tempfile
import time
from typing import Any
from unittest.mock import patch

from carnot.reporting import fresh_corpus_a_9058 as d
from carnot.reporting import fresh_seal_runner_9042 as prior
from carnot.reporting import scoped_provenance_9014 as checks
from carnot.reporting import retained_shard_runner_9030 as terminal
from carnot.reporting.current_work_receipt import ZERO_INVOCATION_COUNTS, atomic_json
from carnot.reporting.primary_publication import publish_primary
from carnot.reporting.v709_execution import child
from carnot.reporting.v756_branch_qualification import measured, verify_measured

Json = dict[str, Any]
DIRECT = [
    "tests/python/test_fresh_seal_methods_9042.py",
    "tests/python/test_graph_corpus_a_8961.py",
    "tests/python/test_current_work_receipt.py",
    "tests/python/test_primary_publication_7928.py",
]


def foreign_argv(*args: str) -> tuple[str, ...]:
    """Direct script imports must work without the checkout as the working directory."""
    with patch.multiple(checks, CLI=d.CLI):
        return checks.foreign_argv(*args)


def plan(private: Path) -> tuple[list[Any], Json]:
    """Reuse the qualified scoped harness with exact owned paths and unchanged consumers."""
    with patch.multiple(prior, d=d, DIRECT=DIRECT):
        return prior.plan(private)


def freeze(root: Path, raw: Path, fixture: bool) -> Json:
    """Missing external authority terminates before any model can be loaded."""
    started = time.monotonic()
    d.progress("preconditions_before", 0, 512)
    environment = d.p.environment(raw)
    evidence = d.authenticate(root, raw)
    result: Json = dict(
        counts=dict(ZERO_INVOCATION_COUNTS),
        ledger=[],
        checks=[],
        owned_failure=False,
        model_receipt={},
    )
    rows = []
    if evidence["manifest"]:
        state = d.c.Checkpoint(
            evidence["manifest"], d.canonical_hash(evidence["manifest"]), raw / "custody"
        )
        if all(r["passed"] for r in evidence["checks"] + environment["checks"]) and not fixture:
            pending = [
                u for u in evidence["units"] if state.state["status"][u["unit_id"]] == "pending"
            ]
            with patch.object(d.runtime, "collect", partial(d.collect, checkpoint=state)):
                result = d.runtime.run(pending, raw / "runtime", task_id=d.TASK)
            result["checks"] = [
                d.p.gate(
                    raw / "runtime", r["artifact_field"], r["expected"], r["observed"], r["check"]
                )
                for r in result["checks"]
            ]
        rows = d.public_rows(evidence["units"], state)
    d.progress(
        "measurement_after",
        sum(r["status"] == "completed" for r in rows),
        512 - sum(r["status"] == "completed" for r in rows),
    )
    return dict(
        raw=str(raw),
        input_root=str(root),
        fixture_only=fixture,
        evidence=evidence,
        environment=environment,
        runtime=result,
        rows=rows,
        server_executable_reference=d.p.reference(
            Path.home() / ".cache/llama.cpp-master/build/bin/llama-server"
        ),
        code_references=[d.p.reference(d.ROOT / name) for name in d.OWNED],
        phase_spans=[
            dict(phase="authentication_and_collection", duration_s=time.monotonic() - started)
        ],
        duration_s=time.monotonic() - started,
    )


def replay(path: Path) -> Json:
    """Rebuild summaries from immutable primitives rather than trusting a rehashed headline."""
    try:
        value = json.loads(path.read_bytes())
        for ref in (
            value["raw_shard_hashes"]
            + value["source_artifact_hashes"]
            + value["code_config_hashes"]
        ):
            if d.p.reference(Path(ref["path"]))["sha256"] != ref["sha256"]:
                raise ValueError("primitive_bytes")
        work = json.loads(Path(value["raw_shard_hashes"][0]["path"]).read_bytes())
        validation = json.loads((Path(work["raw"]) / "validation.json").read_bytes())
        if work["rows"]:
            checkpoint = d.c.Checkpoint.__new__(d.c.Checkpoint)
            checkpoint.state = json.loads(
                (Path(work["raw"]) / "custody/checkpoint.json").read_bytes()
            )
            if (
                checkpoint.state["manifest_sha256"]
                != d.canonical_hash(work["evidence"]["manifest"])
                or d.public_rows(work["evidence"]["units"], checkpoint) != work["rows"]
            ):
                raise ValueError("transport_reduction_drift")
            for index, row in enumerate(work["rows"]):
                if row["status"] != "completed":
                    continue
                folder = d.runtime.folder(Path(work["raw"]) / "runtime/collection", index, row)
                envelope = json.loads((folder / "response.json").read_bytes())
                response = envelope["raw_response"]
                call = next(r for r in work["runtime"]["ledger"] if r["call_id"] == row["unit_id"])
                if (
                    row["response_sha256"] != d.p.reference(folder / "response.json")["sha256"]
                    or call["response_sha256"] != d.canonical_hash(response)
                    or row["request_id"] != response["id"]
                    or row["token_count"] != response["usage"]["completion_tokens"]
                    or row["duration_s"] != envelope["duration_s"]
                ):
                    raise ValueError("response_receipt_drift")
        if work["accepted"] and not (
            checks.receipts_pass(
                dict(
                    validation_scope_manifest=validation["scope"],
                    validation=validation,
                    frozen_ns=validation["scope"]["frozen_ns"],
                )
            )
            and verify_measured(validation["coverage"], d.OWNED)
        ):
            raise ValueError("validation_drift")
        work["raw_references"] = value["raw_shard_hashes"]
        if d.build(work, validation, work["accepted"]) != value:
            raise ValueError("aggregate_drift")
        return dict(passed=True, reason="primitive_replay_passed")
    except (OSError, ValueError, KeyError, TypeError, IndexError, StopIteration) as error:
        return dict(passed=False, reason=str(error))


def main(argv: list[str] | None = None) -> int:
    """Publish only a checked terminal while retaining failed candidates and private evidence."""
    os.environ.update(PYTHONUNBUFFERED="1", JAX_PLATFORMS="cpu")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", choices=["20261112"], default="20261112")
    parser.add_argument("--input-root", type=Path, default=d.ROOT)
    parser.add_argument("--fixture-e2e", type=Path)
    parser.add_argument("--cold-replay", type=Path)
    args = parser.parse_args(argv)
    if args.cold_replay:
        report = replay(args.cold_replay)
        print(json.dumps(report), flush=True)
        return int(not report["passed"])
    output = args.fixture_e2e or d.ROOT / "results" / (d.NAME + ".json")
    if args.fixture_e2e and output.resolve().is_relative_to(d.ROOT):
        parser.error("fixture writes require private scratch")
    raw = output.absolute().parent / "raw" / d.NAME / "invocations" / str(time.time_ns())
    raw.mkdir(parents=True, mode=0o700)
    private = Path(tempfile.mkdtemp(prefix="carnot-9058-validation-"))
    commands, scope = plan(private)
    atomic_json(raw / "validation_scope_manifest.json", scope)
    started = time.monotonic()
    work = freeze(args.input_root, raw, args.fixture_e2e is not None)
    validation: Json = dict(receipts=[], coverage={}, scope=scope)
    accepted = False
    if not args.fixture_e2e:
        before = os.environ.get("COVERAGE_RCFILE")
        os.environ["COVERAGE_RCFILE"] = str(private / "coverage.ini")
        try:
            for index, command in enumerate(commands):
                d.progress("owned_validation", index, len(commands) - index)
                validation["receipts"].append(checks.run_check(command, raw / "validation", scope))
        finally:
            if before is None:
                os.environ.pop("COVERAGE_RCFILE", None)
            else:
                os.environ["COVERAGE_RCFILE"] = before
        validation["coverage"] = measured([private / ".coverage"], d.OWNED, raw / "measured")
        validation["coverage"].pop("rows", None)
        accepted = checks.receipts_pass(
            dict(
                validation_scope_manifest=scope, validation=validation, frozen_ns=scope["frozen_ns"]
            )
        ) and verify_measured(validation["coverage"], d.OWNED)
    work.update(accepted=accepted, duration_s=time.monotonic() - started)
    atomic_json(raw / "work.json", work)
    atomic_json(raw / "validation.json", validation)
    work["raw_references"] = [
        d.p.reference(raw / "work.json"),
        d.p.reference(raw / "validation.json"),
        *[
            d.p.reference(path)
            for folder in [raw / "runtime", raw / "custody"]
            for path in sorted(folder.rglob("*"))
            if path.is_file()
        ],
    ]
    value = d.build(work, validation, accepted)
    atomic_json(raw / "candidate.json", value)
    if not args.fixture_e2e and not value["required_checks_passed"]:
        d.progress("owned_checks_failed_primary_preserved", 0, 1)
        return 1
    if args.fixture_e2e:
        publish_primary(output, value, replay)
    else:
        with patch.multiple(terminal, d=d):
            publish_primary(
                output, value, lambda path: terminal.scanners(path, raw / "terminal", scope)
            )
    d.progress("terminal_published", value["completed_count"], value["censored_count"])
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
