"""REQ-REPORT-8966: publish only validated causal evidence and replay its custody."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import time
from typing import Any

from carnot.reporting import online_response_streams_8966 as d
from carnot.reporting.current_work_receipt import atomic_json
from carnot.reporting.experiment_7303_validation_scope import CommandSpec, run_commands
from carnot.reporting.matched_corpus_runner_8949 import validation_plan
from carnot.reporting.primary_publication import publish_primary
from carnot.reporting.v756_branch_qualification import measured

Json = dict[str, Any]


def evaluation(evidence: Json, raw: Path, replay: bool = False) -> tuple[Json, bool]:
    """External absence blocks; an owned causal failure disqualifies the experiment."""
    if evidence["failures"]:
        return d.empty_result(), False
    try:
        return d.evaluate(evidence, raw, replay=replay), False
    except (ValueError, OSError, KeyError, TypeError) as error:
        evidence["failures"].append(
            dict(
                upstream=d.TASK,
                path=str(raw),
                hash=None,
                check="owned_stream_controls",
                artifact_field="causal_streams",
                expected="complete authenticated streams",
                observed=str(error),
                failure_kind="failed_owned_control",
            )
        )
        return d.empty_result(), True


def freeze(root: Path, raw: Path, fixture: bool) -> Json:
    """Snapshot original inputs and executed code before any labels can be released."""
    started = time.monotonic()
    d.progress("before_authentication", 0, 4)
    for name, _, _, _ in d.INPUTS:
        source = root / name
        if not fixture and source.is_file():
            target = raw / "inputs" / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(source.read_bytes())
    evidence = d.authenticate(raw / "inputs")
    authenticated = time.monotonic()
    d.progress("after_authentication", 4, 0)
    result, failed = evaluation(evidence, raw)
    sources = []
    for name in [
        *d.OWNED,
        *d.TESTS,
        "python/carnot/memory/online_calibration_8960.py",
        "python/carnot/reporting/relational_controls_8965.py",
        "python/carnot/autoresearch/relational_controls_worker.py",
        "python/carnot/autoresearch/relational_graph_encoder.py",
        "python/carnot/reporting/current_work_receipt.py",
        "python/carnot/reporting/primary_publication.py",
        "scripts/experiment_template.py",
        "openspec/capabilities/research-reporting/spec.md",
    ]:
        target = raw / "sources" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((d.ROOT / name).read_bytes())
        sources.append(
            dict(
                d.ref(target), source_path=str(d.ROOT / name), executed_source=name.endswith(".py")
            )
        )
    ended = time.monotonic()
    return dict(
        raw=str(raw),
        evidence=evidence,
        result=result,
        owned_failure=failed,
        sources=sources,
        started=started,
        ended=ended,
        phase_spans=[
            dict(phase="authentication", duration_s=authenticated - started),
            dict(phase="sealed_streams", duration_s=ended - authenticated),
        ],
        raw_references=[],
        global_test_debt={},
        prior_terminal={},
    )


def replay(path: Path) -> Json:
    """Recompute streams and typed metrics so changed evidence fails even after rehashing."""
    d.progress("before_cold_replay")
    try:
        value = json.loads(path.read_text())
        for ref in (
            value["source_artifact_hashes"]
            + value["code_config_hashes"]
            + value["raw_shard_hashes"]
        ):
            if (
                d.ref(Path(ref["path"]))["sha256"] != ref["sha256"]
                or ref.get("executed_source")
                and d.ref(Path(ref["source_path"]))["sha256"] != ref["sha256"]
            ):
                raise ValueError("retained_bytes_drift")
        raw = Path(value["raw_shard_hashes"][0]["path"]).parent
        work = json.loads((raw / "work.json").read_text())
        evidence = d.authenticate(raw / "inputs")
        result, failed = evaluation(evidence, raw, True)
        if (
            evidence != work["evidence"]
            or result != work["result"]
            or failed != work["owned_failure"]
        ):
            raise ValueError("measurement_drift")
        validation = json.loads((raw / "validation.json").read_text())
        passed = d.build(work, validation, value["fixture_only"]) == value
        d.progress("after_cold_replay", int(passed), int(not passed))
        return dict(passed=passed)
    except (OSError, ValueError, KeyError, TypeError):
        d.progress("cold_replay_refused", 0, 1)
        return dict(passed=False)


def main(argv: list[str] | None = None) -> int:
    """Keep fixture publication private and bound all validation before primary publication."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date")
    parser.add_argument("--fixture-e2e", type=Path)
    parser.add_argument("--cold-replay", type=Path)
    args = parser.parse_args(argv)
    if args.cold_replay is not None:
        report = replay(args.cold_replay)
        print(json.dumps(report), flush=True)
        return 0 if report["passed"] else 1
    if args.date != "20261110":
        parser.error("this producer requires --date 20261110")
    fixture = args.fixture_e2e is not None
    output = args.fixture_e2e or d.ROOT / "results" / (d.NAME + ".json")
    if fixture and output.resolve().is_relative_to(d.ROOT):
        parser.error("fixture writes require private scratch")
    previous = json.loads(output.read_text()) if output.is_file() else {}
    raw = output.parent / "raw" / d.NAME / "invocations" / str(time.time_ns())
    work = freeze(d.ROOT, raw, fixture)
    work["prior_terminal"] = previous
    validation: Json = dict(receipts=[], coverage={})
    checked = time.monotonic()
    if not fixture:
        private = Path(tempfile.mkdtemp(prefix="carnot-8966-validation-"))
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
    work["ended"] = time.monotonic()
    work["phase_spans"].append(
        dict(phase="validation_and_repository_health", duration_s=work["ended"] - checked)
    )
    atomic_json(raw / "validation.json", validation)
    work["raw_references"] = [
        d.ref(p)
        for p in sorted(raw.rglob("*"))
        if p.is_file() and "sources" not in p.parts and "inputs" not in p.parts
    ]
    atomic_json(raw / "work.json", work)
    value = d.build(work, validation, fixture)
    d.progress("before_primary_publication")
    published = publish_primary(output, value, replay)
    d.progress("terminal_published", value["completed_count"], value["censored_count"])
    print(
        json.dumps(
            dict(
                **published,
                honest_verdict=value["honest_verdict"],
                online_rows_ready_score=value["online_rows_ready_score"],
            )
        ),
        flush=True,
    )
    return 0
