"""REQ-REPORT-8949: validate owned work before publishing a terminal shard result."""

from __future__ import annotations

import argparse
from datetime import datetime, UTC
import json
from pathlib import Path
import tempfile
import time
from types import ModuleType
from typing import Any

from carnot.inference import graph_corpus_runtime_8961 as runtime
from carnot.reporting import graph_corpus_a_8961 as d
from carnot.reporting import bfs_error_pilot_8918 as pilot
from carnot.reporting import graph_source_methods_8931 as methods
from carnot.reporting import matched_source_methods_8946 as matched
from carnot.reporting import graph_source_8931 as source
from carnot.reporting.current_work_receipt import ZERO_INVOCATION_COUNTS, atomic_json
from carnot.reporting.experiment_7303_validation_scope import CommandSpec, run_commands
from carnot.reporting.primary_publication import publish_primary
from carnot.reporting.v756_branch_qualification import measured

Json = dict[str, Any]


def validation_plan(private: Path, *, producer: ModuleType | None = None) -> list[CommandSpec]:
    """Reuse the established private check roster with this producer's exact paths."""
    from carnot.reporting.matched_corpus_runner_8949 import validation_plan as shared_plan

    commands: list[CommandSpec] = shared_plan(private, producer=producer or d)
    return commands


def freeze(root: Path, raw: Path, fixture: bool, *, producer: ModuleType | None = None) -> Json:
    """Seal original inputs, execution order and source bytes before live generation."""
    config = producer or d
    started = time.monotonic()
    started_at = datetime.now(UTC).isoformat()
    source.progress("before_input_authentication", 0, 512)
    inputs = raw / "inputs"
    for name in [row[0] for row in config.INPUTS]:
        path = root / name
        if path.is_file():
            target = inputs / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(path.read_bytes())
    evidence = config.authenticate(inputs)
    # The frozen method roster still defines all slots when an external gate is absent.
    protocol = evidence["protocol"] or json.loads((config.ROOT / config.INPUTS[0][0]).read_text())
    units = config.ordered_units(protocol)
    atomic_json(raw / "scheduled_units.json", dict(units=units))
    source.progress("after_input_authentication", 0, 512)
    sources = []
    for name in [
        *config.OWNED,
        *config.TESTS,
        "python/carnot/reporting/graph_corpus_a_8961.py",
        "python/carnot/reporting/graph_corpus_runner_8961.py",
        "python/carnot/inference/graph_corpus_runtime_8961.py",
        "tests/python/test_matched_depth_control_8948.py",
        "python/carnot/reporting/depth_control_8934.py",
        "python/carnot/reporting/matched_depth_control_8948.py",
        "python/carnot/inference/matched_depth_runtime_8948.py",
        "python/carnot/reporting/matched_source_methods_8946.py",
        "python/carnot/reporting/pilot_label_audit_8932.py",
        "python/carnot/reporting/v765_branch_contract.py",
        "python/carnot/inference/sota_models.py",
        "python/carnot/reporting/graph_source_8931.py",
        "python/carnot/reporting/graph_source_methods_8931.py",
        "python/carnot/reporting/pilot_label_audit_runner_8932.py",
        "python/carnot/inference/bfs_pilot_runtime_8918.py",
        "python/carnot/inference/qwen_sufficiency_7920.py",
        "python/carnot/inference/llama_cpp_process.py",
        "python/carnot/gpu_lease_phase_journal.py",
        "python/carnot/verify/qwen_development_capture_7995.py",
        "python/carnot/reporting/current_work_receipt.py",
        "python/carnot/reporting/primary_publication.py",
        "scripts/experiment_template.py",
        "scripts/adversarial_verify.py",
        "scripts/verdict_row_consistency_lint.py",
        "openspec/capabilities/research-reporting/spec.md",
    ]:
        target = raw / "sources" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((config.ROOT / name).read_bytes())
        sources.append(
            dict(
                methods.reference(target),
                source_path=str(config.ROOT / name),
                executed_source=name in config.OWNED,
            )
        )
    auth_duration = time.monotonic() - started
    if fixture or evidence["failures"]:
        result: Json = dict(
            rows=[
                source.censored(u, "private_fixture" if fixture else "external_inputs")
                for u in units
            ],
            counts=dict(ZERO_INVOCATION_COUNTS),
            checks=[],
            live=False,
            model_receipt={},
            resource_lease={},
            phase_spans=[],
            settings=dict(source.SETTINGS),
            owned_failure=False,
        )
        atomic_json(raw / "runtime" / "runtime.json", result)
    else:
        result = runtime.run(
            units, raw / "runtime", **(dict(task_id=config.TASK) if producer else {})
        )
    reduced = config.reduce(units, result["rows"])
    atomic_json(raw / "evaluation_labels.json", dict(rows=reduced["label_sidecar"]))
    atomic_json(raw / "detector_view.json", dict(rows=reduced["detector_view"]))
    return dict(
        template=config.ExperimentTemplate(
            config.EXPERIMENT_ID,
            f"Matched graph corpus shard {config.SHARD}",
            f"results/{config.NAME}.json",
        ).build_result({}, status="success"),
        raw=str(raw),
        started=started,
        started_at=started_at,
        finished_at=datetime.now(UTC).isoformat(),
        ended=time.monotonic(),
        units=units,
        runtime=result,
        authentication=evidence,
        sources=sources,
        input_references=[methods.reference(p) for p in sorted(inputs.rglob("*")) if p.is_file()],
        raw_references=[
            methods.reference(p) for p in sorted((raw / "runtime").rglob("*")) if p.is_file()
        ]
        + [
            methods.reference(raw / name)
            for name in ("scheduled_units.json", "evaluation_labels.json", "detector_view.json")
        ],
        phase_spans=[dict(phase="authenticate_and_snapshot", duration_s=auth_duration)],
        repository_health={},
        prior_terminal={},
    )


def replay(path: Path, *, producer: ModuleType | None = None) -> Json:
    """Rebuild labels, signals and gates from authenticated raw bytes, not headline values."""
    config = producer or d
    try:
        value = json.loads(path.read_text())
        for ref in (
            value["raw_shard_hashes"]
            + value["source_artifact_hashes"]
            + value["code_config_hashes"]
        ):
            if methods.reference(Path(ref["path"]))["sha256"] != ref["sha256"]:
                raise ValueError("retained_bytes_drift")
        raw = Path(value["raw_shard_hashes"][0]["path"]).parent
        work = json.loads((raw / "work.json").read_text())
        validation = json.loads((raw / "validation.json").read_text())
        if (
            config.authenticate(raw / "inputs") != work["authentication"]
            or config.ordered_units(
                work["authentication"]["protocol"]
                or json.loads((config.ROOT / config.INPUTS[0][0]).read_text())
            )
            != work["units"]
        ):
            raise ValueError("input_or_order_drift")
        for index, (unit, row) in enumerate(
            zip(work["units"], work["runtime"]["rows"], strict=True)
        ):
            response_path = (
                runtime.folder(raw / "runtime" / "collection", index, unit) / "response.json"
            )
            if response_path.is_file():
                envelope = json.loads(response_path.read_text())
                rebuilt = source.label_response(unit, envelope["raw_response"])
                rebuilt.update(
                    duration_s=envelope["duration_s"],
                    actual_token_usage=envelope["raw_response"].get("usage"),
                )
                if rebuilt != row:
                    raise ValueError("response_label_drift")
        rebuilt_views = config.reduce(work["units"], work["runtime"]["rows"])
        for filename, key in (
            ("evaluation_labels.json", "label_sidecar"),
            ("detector_view.json", "detector_view"),
        ):
            if json.loads((raw / filename).read_text()) != dict(rows=rebuilt_views[key]):
                raise ValueError("sidecar_reduction_drift")
        passed = config.build(work, validation, value["fixture_only"]) == value
        return dict(passed=passed, replay_passed=passed)
    except (OSError, ValueError, KeyError, TypeError):
        return dict(passed=False, replay_passed=False)


def main(argv: list[str] | None = None, *, producer: ModuleType | None = None) -> int:
    """Private CLI cases cannot load weights or recursively qualify their own fixtures."""
    config = producer or d
    options = dict(producer=producer) if producer else {}
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date")
    parser.add_argument("--fixture-e2e", type=Path)
    parser.add_argument("--cold-replay", type=Path)
    args = parser.parse_args(argv)
    if args.cold_replay is not None:
        report = replay(args.cold_replay, **options)
        print(json.dumps(report), flush=True)
        return 0 if report["passed"] else 1
    run_date = getattr(config, "RUN_DATE", "20261110")
    if args.date != run_date:
        parser.error(f"this producer requires --date {run_date}")
    fixture = args.fixture_e2e is not None
    output = args.fixture_e2e or config.ROOT / "results" / (config.NAME + ".json")
    if fixture and output.resolve().is_relative_to(config.ROOT):
        parser.error("fixture writes require private scratch")
    raw = output.parent / "raw" / config.NAME / "invocations" / str(time.time_ns())
    prior = json.loads(output.read_text()) if output.is_file() else {}
    work = freeze(config.ROOT, raw, fixture, **options)
    work["prior_terminal"] = prior
    validation_started = time.monotonic()
    validation: Json = dict(receipts=[], coverage={})
    if not fixture:
        private = Path(tempfile.mkdtemp(prefix=f"carnot-{config.EXPERIMENT_ID}-validation-"))
        source.progress("before_owned_checks")
        validation["receipts"] = run_commands(
            config.ROOT,
            validation_plan(private, **options),
            log_dir=raw / "validation",
            heartbeat_s=15,
        )
        if (private / ".coverage").is_file():
            validation["coverage"] = measured(
                [private / ".coverage"], config.OWNED, raw / "measured"
            )
        source.progress("after_owned_checks")
        work["repository_health"] = dict(
            receipts=run_commands(
                config.ROOT,
                [
                    CommandSpec(
                        "full_python_suite",
                        (str(config.ROOT / ".venv/bin/pytest"), "tests/python", "-q"),
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
    work["ended"] = time.monotonic()
    work["finished_at"] = datetime.now(UTC).isoformat()
    work["phase_spans"].append(
        dict(
            phase="validation_and_repository_health", duration_s=work["ended"] - validation_started
        )
    )
    atomic_json(raw / "work.json", work)
    atomic_json(raw / "validation.json", validation)
    value = config.build(work, validation, fixture)
    if not fixture and not value["required_checks_passed"]:
        raise ValueError("owned_checks_failed; primary unchanged")
    receipts: list[Json] = []

    def validate(candidate: Path) -> Json:
        report = replay(candidate, **options)
        if not fixture:
            py = str(config.ROOT / ".venv/bin/python")
            commands = [
                CommandSpec(
                    "cold_replay_cli",
                    (
                        py,
                        "-u",
                        "-c",
                        "import os,subprocess,sys,tempfile; env=dict(os.environ); env.pop('PYTHONPATH',None); raise SystemExit(subprocess.run(sys.argv[1:],cwd=tempfile.gettempdir(),env=env,timeout=60).returncode)",
                        py,
                        "-u",
                        str(config.ROOT / config.CLI),
                        "--cold-replay",
                        str(candidate),
                    ),
                    "terminal",
                    75,
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
            receipts.extend(
                run_commands(
                    config.ROOT, commands, log_dir=raw / "terminal_validation", heartbeat_s=15
                )
            )
            report.update(
                passed=report["passed"] and pilot.terminal_passed(receipts), receipts=receipts
            )
        return report

    source.progress("before_terminal_validation")
    try:
        published = publish_primary(output, value, validate)
    except ValueError as error:
        if str(error) != "candidate_rejected":
            raise
        validation["terminal_rejected"] = True
        validation["receipts"].extend(receipts)
        atomic_json(raw / "validation.json", validation)
        value = config.build(work, validation, fixture)
        published = publish_primary(output, value, lambda path: replay(path, **options))
    source.progress("terminal_published", 512, 0)
    print(
        json.dumps(
            dict(
                **published,
                honest_verdict=value["honest_verdict"],
                shard_ready_score=value["shard_ready_score"],
            )
        ),
        flush=True,
    )
    return 0
