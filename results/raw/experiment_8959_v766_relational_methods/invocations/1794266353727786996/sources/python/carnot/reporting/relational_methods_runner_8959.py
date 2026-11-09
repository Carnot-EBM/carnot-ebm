"""REQ-REPORT-8959: publish only after checks and semantic cold replay."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import time
from typing import Any

from carnot.reporting import relational_methods_8959 as d
from carnot.reporting import matched_corpus_runner_8949 as shared
from carnot.reporting.current_work_receipt import atomic_json
from carnot.reporting.experiment_7303_validation_scope import CommandSpec, run_commands
from carnot.reporting.primary_publication import publish_primary
from carnot.reporting.v756_branch_qualification import measured

Json = dict[str, Any]


def freeze(root: Path, raw: Path) -> Json:
    """Save weights and all adaptive choices before fresh graph construction starts."""
    started = time.monotonic()
    d.b.progress("8959_before_authentication")
    evidence = d.authenticate(root)
    d.b.progress("8959_after_authentication", len(evidence["developer"]), len(evidence["failures"]))
    trained: Json = dict(states={}, numerical_fit_count=0, numerical_training_calls=0)
    controls: Json = dict(rows=[], passed=False, numerical_fit_count=0, numerical_training_calls=0)
    if not evidence["failures"]:
        trained = d.train(evidence)
        controls = d.fixture_controls()
    atomic_json(raw / "trained_states.json", trained)
    atomic_json(raw / "frozen_protocol.json", d.protocol())
    sealed = time.monotonic()
    d.b.progress("8959_weights_and_protocol_saved")
    graphs = (
        d.graph_protocol(evidence)
        if not evidence["failures"] and controls["passed"]
        else dict(
            graph_manifest=[],
            proposal_rejection_rows=[],
            complete=False,
            terminal_reason="external_evidence"
            if evidence["failures"]
            else "failed_owned_controls",
            history_graph_count=len(evidence["old_graphs"]),
            history_construction_seed_count=len(evidence["old_seeds"]),
        )
    )
    d.b.progress(
        "8959_after_fresh_construction",
        len(graphs["graph_manifest"]),
        1024 - len(graphs["graph_manifest"]),
    )
    sources = []
    for name in [
        *d.OWNED,
        *d.TESTS,
        "python/carnot/autoresearch/graph_calibration_benchmark.py",
        "python/carnot/models/gibbs/__init__.py",
        "python/carnot/reporting/graph_source_8931.py",
        "python/carnot/reporting/current_work_receipt.py",
        "python/carnot/reporting/primary_publication.py",
        "scripts/experiment_template.py",
        "openspec/capabilities/research-reporting/spec.md",
    ]:
        target = raw / "sources" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((root / name).read_bytes())
        sources.append(
            dict(
                d.calibration.reference(target),
                source_path=str(root / name),
                executed_source=name in d.OWNED,
            )
        )
    ended = time.monotonic()
    work = dict(
        evidence=evidence,
        trained=trained,
        controls=controls,
        graphs=graphs,
        sources=sources,
        trained_reference=d.calibration.reference(raw / "trained_states.json"),
        protocol_reference=d.calibration.reference(raw / "frozen_protocol.json"),
        freeze_order=sealed <= ended,
        duration_s=ended - started,
        phase_spans=[
            dict(phase="authenticate_train_and_seal", duration_s=sealed - started),
            dict(phase="fresh_construction", duration_s=ended - sealed),
        ],
        global_test_debt={},
        raw_references=[],
    )
    atomic_json(raw / "measurement.json", work)
    work["raw_references"] = [
        d.calibration.reference(raw / "measurement.json"),
        work["trained_reference"],
        work["protocol_reference"],
    ]
    return work


def replay(path: Path) -> Json:
    """Rebuild the published object from checked primitive operands, not headline fields."""
    d.b.progress("8959_before_cold_replay")
    try:
        value = json.loads(path.read_text())
        for ref in (
            value["source_artifact_hashes"]
            + value["code_config_hashes"]
            + value["raw_shard_hashes"]
        ):
            if d.calibration.reference(Path(ref["path"]))["sha256"] != ref["sha256"]:
                raise ValueError("retained_bytes_drift")
            if (
                ref.get("executed_source")
                and d.calibration.reference(Path(ref["source_path"]))["sha256"] != ref["sha256"]
            ):
                raise ValueError("executed_bytes_drift")
        raw = Path(value["raw_shard_hashes"][0]["path"]).parent
        work = json.loads((raw / "work.json").read_text())
        validation = json.loads((raw / "validation.json").read_text())
        work["raw_references"].extend(
            [
                d.calibration.reference(raw / "work.json"),
                d.calibration.reference(raw / "validation.json"),
            ]
        )
        if d.build(work, validation, value["fixture_only"]) != value:
            raise ValueError("aggregate_drift")
        evidence = d.authenticate(d.ROOT)
        if (
            evidence != work["evidence"]
            or json.loads((raw / "frozen_protocol.json").read_text()) != d.protocol()
            or json.loads((raw / "trained_states.json").read_text()) != work["trained"]
            or not d.verify_graphs(work["graphs"], evidence)
        ):
            raise ValueError("frozen_operands_drift")
        training_calls = 0
        if not evidence["failures"]:
            recomputed = d.train(evidence)
            training_calls = recomputed["numerical_training_calls"]
            if recomputed != work["trained"]:
                raise ValueError("trained_state_drift")
            if work["graphs"]["complete"]:
                roster = d.graph_protocol(evidence)
                if any(
                    roster[key] != work["graphs"][key]
                    for key in (
                        "graph_manifest",
                        "proposal_rejection_rows",
                        "complete",
                        "terminal_reason",
                    )
                ):
                    raise ValueError("construction_drift")
        passed = True
        d.b.progress("8959_after_cold_replay", int(passed), int(not passed))
        return dict(passed=passed, replay_passed=passed, numerical_training_calls=training_calls)
    except (OSError, ValueError, KeyError, TypeError) as error:
        d.b.progress("8959_cold_replay_refused")
        return dict(passed=False, replay_passed=False, reason=str(error))


def main(argv: list[str] | None = None) -> int:
    """Private CLI runs use scratch; the real producer records one global health run."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date")
    parser.add_argument("--fixture-e2e", type=Path)
    parser.add_argument("--cold-replay", type=Path)
    args = parser.parse_args(argv)
    if args.cold_replay is not None:
        report = replay(args.cold_replay)
        print(json.dumps(report), flush=True)
        return 0 if report["passed"] else 1
    if args.date != "20261109":
        parser.error("this producer requires --date 20261109")
    fixture = args.fixture_e2e is not None
    output = args.fixture_e2e or d.ROOT / "results" / (d.NAME + ".json")
    if fixture and output.resolve().is_relative_to(d.ROOT):
        parser.error("fixture writes require private scratch")
    raw = output.parent / "raw" / d.NAME / "invocations" / str(time.time_ns())
    started = time.monotonic()
    work = freeze(d.ROOT, raw)
    validation: Json = dict(receipts=[], coverage={})
    if not fixture:
        private = Path(tempfile.mkdtemp(prefix="carnot-8959-validation-"))
        d.b.progress("8959_before_owned_checks")
        validation["receipts"] = run_commands(
            d.ROOT,
            [
                replace(command, timeout_s=540)
                for command in shared.validation_plan(private, producer=d)
            ],
            log_dir=raw / "validation",
            heartbeat_s=15,
        )
        if (private / ".coverage").is_file():
            validation["coverage"] = measured([private / ".coverage"], d.OWNED, raw / "measured")
        d.b.progress("8959_after_owned_checks")
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
    work["raw_references"].extend(
        [
            d.calibration.reference(raw / "work.json"),
            d.calibration.reference(raw / "validation.json"),
        ]
    )
    value = d.build(work, validation, fixture)
    if not fixture and not value["required_checks_passed"]:
        raise ValueError("owned_checks_failed; candidate retained below raw; primary unchanged")
    d.b.progress("8959_before_primary_publication")
    published = publish_primary(output, value, replay)
    d.b.progress("8959_terminal_published", value["completed_count"], value["censored_count"])
    print(
        json.dumps(
            dict(
                **published,
                honest_verdict=value["honest_verdict"],
                relational_methods_ready_score=value["relational_methods_ready_score"],
            )
        ),
        flush=True,
    )
    return 0
