"""REQ-REPORT-8986: retain real validation and publish only checked primary bytes."""

import argparse
from dataclasses import replace
import json
import os
from pathlib import Path
import sys
import tempfile
import time
from typing import Any

from carnot.reporting import spline_basis_methods_8986 as d
from carnot.reporting import shared_feature_methods_8973 as historical
from carnot.reporting.current_work_receipt import atomic_json
from carnot.reporting.experiment_7303_validation_scope import CommandSpec, run_commands
from carnot.reporting.primary_publication import publish_primary

Json = dict[str, Any]


def freeze(root: Path, raw: Path) -> Json:
    """Authenticate and snapshot inputs before fitting any natural coefficient."""
    started = time.monotonic()
    d.progress("before_preconditions", 0, 1)
    evidence = d.authenticate(root, raw)
    preconditions = d.preflight(raw, evidence)
    authenticated = time.monotonic()
    d.progress("after_preconditions", int(not evidence["failures"]), len(evidence["failures"]))
    sources = [
        dict(d.retain(d.reference(d.ROOT / name), raw), executed_source=name in d.OWNED)
        for name in [
            *d.OWNED,
            *d.TESTS,
            "python/carnot/autoresearch/encoder_head_8972.py",
            "python/carnot/autoresearch/relational_graph_encoder.py",
            "python/carnot/reporting/shared_feature_methods_8973.py",
            "python/carnot/reporting/encoder_head_diagnosis_8972.py",
            "python/carnot/reporting/primary_publication.py",
            "python/carnot/reporting/current_work_receipt.py",
            "scripts/experiment_template.py",
            "scripts/adversarial_verify.py",
            "scripts/verdict_row_consistency_lint.py",
        ]
    ]
    result, owned_failure = {}, None
    if not evidence["failures"]:
        try:
            result = d.evaluate(evidence, raw)
        except ValueError as error:
            owned_failure = str(error)
            d.progress("measurement_refused", 0, 1)
    ended = time.monotonic()
    return dict(
        evidence=evidence,
        result=result,
        sources=sources,
        raw=str(raw),
        preconditions=preconditions,
        duration_s=ended - started,
        phase_spans=[
            dict(phase="preconditions", duration_s=authenticated - started),
            dict(phase="numerical_fitting", duration_s=ended - authenticated),
        ],
        global_test_debt={},
        raw_references=[],
        owned_failure=owned_failure,
    )


def artifact(work: Json, validation: Json, fixture: bool) -> Json:
    """Bind work separately so no digest needs to include itself."""
    value = d.build(work, validation, fixture)
    value["work_reference"] = d.reference(Path(work["raw"]) / "work.json")
    value["field_principles"]["work_reference"] = "Immutable work binds exact replay operands."
    value["reproducibility_checksum"] = d.checksum(value)
    return value


def replay(path: Path) -> Json:
    """Recompute fits and graph reductions from independently pinned historical bytes."""
    d.progress("before_cold_replay")
    try:
        value = json.loads(path.read_text())
        for ref in (
            value["source_artifact_hashes"]
            + value["code_config_hashes"]
            + value["raw_shard_hashes"]
        ):
            if d.reference(Path(ref["path"]))["sha256"] != ref["sha256"]:
                raise ValueError("retained_bytes_drift")
            if (
                ref.get("executed_source")
                and d.reference(Path(ref["source_path"]))["sha256"] != ref["sha256"]
            ):
                raise ValueError("executed_bytes_drift")
        raw = Path(value["work_reference"]["path"]).parent
        work = json.loads((raw / "work.json").read_text())
        validation = json.loads((raw / "validation.json").read_text())
        work["raw_references"] = value["raw_shard_hashes"]
        if artifact(work, validation, value["fixture_only"]) != value:
            raise ValueError("aggregate_drift")
        evidence = work["evidence"]
        if not evidence["failures"]:
            producer = evidence["producer"]
            pinned = next(r for r in evidence["references"] if r.get("experiment_id") == 8972)
            if (
                pinned["sha256"] != historical.PIN
                or json.loads(Path(pinned["path"]).read_text()) != producer
                or historical.load(evidence, producer["work_reference"]) != evidence["work"]
            ):
                raise ValueError("pinned_operand_drift")
            for source, pin, payload in [
                (d.PROTOCOL, d.PROTOCOL_PIN, evidence["protocol"]),
                (d.CONTEXT, d.CONTEXT_PIN, evidence["context"]),
                (d.METHODS, d.METHOD_PIN, {"rows": evidence["method_source_rows"]}),
            ]:
                ref = next(
                    r
                    for r in evidence["references"]
                    if r.get("source_path") == str(d.ROOT / source)
                )
                if ref["sha256"] != pin or json.loads(Path(ref["path"]).read_text()) != payload:
                    raise ValueError("protocol_operand_drift")
            if (
                not work["owned_failure"]
                and d.evaluate(evidence, raw, replay=True) != work["result"]
            ):
                raise ValueError("reduction_drift")
        d.progress("after_cold_replay", 1, 0)
        return dict(passed=True, replay_passed=True)
    except (OSError, ValueError, KeyError, TypeError, StopIteration) as error:
        d.progress("cold_replay_refused", 0, 1)
        return dict(passed=False, replay_passed=False, reason=str(error))


def main(argv: list[str] | None = None) -> int:
    """One dated invocation owns its raw evidence; fixtures cannot self-qualify."""
    d.progress("start")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date")
    parser.add_argument("--fixture-e2e", type=Path)
    parser.add_argument("--cold-replay", type=Path)
    args = parser.parse_args(argv)
    if args.cold_replay is not None:
        report = replay(args.cold_replay)
        print(json.dumps(report), flush=True)
        return 0 if report["passed"] else 1
    if args.date != "20261111":
        parser.error("this producer requires --date 20261111")
    fixture = args.fixture_e2e is not None
    output = args.fixture_e2e or d.ROOT / "results" / (d.NAME + ".json")
    if fixture and output.resolve().is_relative_to(d.ROOT):
        parser.error("fixture writes require private scratch")
    raw = output.parent / "raw" / d.NAME / "invocations" / str(time.time_ns())
    private = Path(tempfile.mkdtemp(prefix="carnot-8986-checks-"))
    from carnot.reporting.matched_corpus_runner_8949 import validation_plan
    from carnot.reporting.v756_branch_qualification import measured
    import carnot.reporting.spline_basis_methods_8986 as producer

    commands = [
        replace(c, timeout_s=600) if c.name == "owned_unit_child" else c
        for c in validation_plan(private, producer=producer)
    ]
    atomic_json(
        raw / "validation_plan.json",
        dict(
            commands=[
                dict(name=c.name, argv=list(c.argv), timeout_s=c.timeout_s, scope=c.scope)
                for c in commands
            ]
        ),
    )
    started = time.monotonic()
    work = freeze(Path(os.environ.get("CARNOT_8986_INPUT_ROOT", str(d.ROOT))), raw)
    validation: Json = dict(receipts=[], coverage={})
    if not fixture:
        d.progress("before_owned_checks", 0, len(commands))
        validation["receipts"] = run_commands(
            d.ROOT, commands, log_dir=raw / "validation", heartbeat_s=15
        )
        if (private / ".coverage").is_file():
            validation["coverage"] = measured([private / ".coverage"], d.OWNED, raw / "measured")
        d.progress("after_owned_checks", len(commands), 0)
        work["global_test_debt"] = dict(
            receipts=run_commands(
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
            ),
            affects_owned_readiness=False,
            repeated=False,
        )
    work["duration_s"] = time.monotonic() - started
    work["phase_spans"].append(
        dict(
            phase="validation",
            duration_s=work["duration_s"] - sum(p["duration_s"] for p in work["phase_spans"]),
        )
    )
    atomic_json(raw / "validation.json", validation)
    atomic_json(raw / "work.json", work)
    work["raw_references"] = [
        d.reference(p) for p in sorted(raw.rglob("*")) if p.is_file() and "inputs" not in p.parts
    ]
    value = artifact(work, validation, fixture)

    def validate(candidate: Path) -> Json:
        d.progress("before_terminal_validators")
        specs = [
            CommandSpec(
                "cold_replay",
                (sys.executable, "-u", d.CLI, "--cold-replay", str(candidate)),
                "private_terminal",
                180,
            )
        ]
        if not fixture:
            specs += [
                CommandSpec(
                    "adversarial",
                    (
                        sys.executable,
                        "-u",
                        "scripts/adversarial_verify.py",
                        "--json",
                        str(candidate),
                    ),
                    "private_terminal",
                    60,
                ),
                CommandSpec(
                    "strict_rows",
                    (
                        sys.executable,
                        "-u",
                        "scripts/verdict_row_consistency_lint.py",
                        "--strict",
                        str(candidate),
                    ),
                    "private_terminal",
                    60,
                ),
            ]
        receipts = run_commands(d.ROOT, specs, log_dir=raw / "terminal", heartbeat_s=15)
        report = dict(passed=all(r["passed"] for r in receipts), receipts=receipts)
        atomic_json(raw / "terminal_report.json", report)
        d.progress("after_terminal_validators", len(receipts), 0)
        return report

    d.progress("before_primary_publication")
    atomic_json(raw / "candidate.json", value)
    published = publish_primary(output, value, validate)
    d.progress("completed", value["completed_count"], value["censored_count"])
    print(
        json.dumps(
            dict(
                **published,
                honest_verdict=value["honest_verdict"],
                spline_methods_ready_score=value["spline_methods_ready_score"],
            )
        ),
        flush=True,
    )
    return 0
