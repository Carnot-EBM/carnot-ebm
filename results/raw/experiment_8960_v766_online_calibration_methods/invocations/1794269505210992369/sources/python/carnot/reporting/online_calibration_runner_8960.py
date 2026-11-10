"""REQ-REPORT-8960: retain operands, check owned work, then publish a single primary."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import tempfile
import time
from typing import Any

from carnot.reporting import online_calibration_methods_8960 as m
from carnot.reporting.current_work_receipt import (
    atomic_json,
    build_current_work_receipt,
    validate_current_work_receipt,
)
from carnot.reporting.experiment_7303_validation_scope import CommandSpec, run_commands
from carnot.reporting.primary_publication import publish_primary
from scripts.experiment_template import ExperimentTemplate

Json = dict[str, Any]


def receipt(work: Json) -> Json:
    """Actual clocks include validation without attributing historical model activity."""
    return build_current_work_receipt(
        run_id=str(work["started_ns"]),
        owner_pid=os.getpid(),
        events=[],
        inference_substrate="verifier_ensemble_against_cached_candidates",
        inference_substrate_class="no_model_load",
        execution_venue="host",
        inference_substrate_details=dict(
            scope="private scalar mechanics only; frozen model outputs retained"
        ),
        started_monotonic_ns=work["started_ns"],
        ended_monotonic_ns=time.monotonic_ns(),
        phase_spans=work["phase_spans"],
        small_ebm_training=dict(
            performed=False, scalar_adapter_steps=work["result"].get("numerical_training_calls", 0)
        ),
    )


def freeze(root: Path, raw: Path) -> Json:
    """Authenticate first; original producers and quarantines remain unchanged."""
    started = time.monotonic_ns()
    m.progress("authentication_before")
    evidence = m.authenticate(root)
    m.progress("authentication_after", len(evidence["references"]), len(evidence["failures"]))
    atomic_json(raw / "protocol.json", m.protocol(evidence))
    result = {} if evidence["failures"] else m.development(raw / "private_mechanics")
    sources = []
    for name in [
        *m.OWNED,
        *m.TESTS,
        "python/carnot/autoresearch/relational_graph_encoder.py",
        "python/carnot/reporting/current_work_receipt.py",
        "python/carnot/reporting/primary_publication.py",
        "scripts/experiment_template.py",
        "openspec/capabilities/research-reporting/spec.md",
    ]:
        target = raw / "sources" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((root / name).read_bytes())
        sources.append(
            dict(m.reference(target), source_path=str(root / name), executed_source=name in m.OWNED)
        )
    work: Json = dict(
        evidence=evidence,
        result=result,
        sources=sources,
        raw_references=[],
        template=ExperimentTemplate(
            8960, "Delayed online calibration methods", f"results/{m.NAME}.json", seed=m.SEED
        ).build_result({}, status="success"),
        started_ns=started,
        phase_spans=[
            dict(
                phase="authenticate_and_private_mechanics",
                duration_s=(time.monotonic_ns() - started) / 1e9,
            )
        ],
        global_test_debt={},
    )
    work["current_work_receipt"] = receipt(work)
    atomic_json(raw / "measurement.json", work)
    work["raw_references"] = [
        m.reference(raw / "measurement.json"),
        m.reference(raw / "protocol.json"),
    ]
    return work


def replay(path: Path) -> Json:
    """Recompute mechanics and the full artifact so repaired hashes cannot hide false claims."""
    m.progress("cold_replay_before")
    try:
        value = json.loads(path.read_text())
        if value["reproducibility_checksum"] != m.checksum(value):
            raise ValueError("checksum_drift")
        for ref in (
            value["source_artifact_hashes"]
            + value["code_config_hashes"]
            + value["raw_shard_hashes"]
        ):
            if m.reference(Path(ref["path"]))["sha256"] != ref["sha256"]:
                raise ValueError("retained_bytes_drift")
            if (
                ref.get("executed_source")
                and m.reference(Path(ref["source_path"]))["sha256"] != ref["sha256"]
            ):
                raise ValueError("executed_bytes_drift")
        raw = Path(value["raw_shard_hashes"][0]["path"]).parent
        work = json.loads((raw / "work.json").read_text())
        validation = json.loads((raw / "validation.json").read_text())
        work["raw_references"] += [
            m.reference(raw / "work.json"),
            m.reference(raw / "validation.json"),
        ]
        if validate_current_work_receipt(work["current_work_receipt"]):
            raise ValueError("compute_scope")
        if value != m.build(work, validation, value["fixture_only"]):
            raise ValueError("aggregate_drift")
        with tempfile.TemporaryDirectory(prefix="carnot8960-replay-") as directory:
            result = {} if work["evidence"]["failures"] else m.development(Path(directory))
        if result != work["result"] or json.loads(
            (raw / "protocol.json").read_text()
        ) != m.protocol(work["evidence"]):
            raise ValueError("mechanics_or_protocol_drift")
        m.progress("cold_replay_after", 1, 0)
        return dict(passed=True, replay_passed=True)
    except (OSError, ValueError, KeyError, TypeError) as error:
        m.progress("cold_replay_refused")
        return dict(passed=False, replay_passed=False, reason=str(error))


def main(argv: list[str] | None = None) -> int:
    """Private children bypass recursive validation and never publish research readiness."""
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
    output = args.fixture_e2e or m.ROOT / "results" / (m.NAME + ".json")
    if fixture and output.resolve().is_relative_to(m.ROOT):
        parser.error("fixture writes require private scratch")
    raw = output.parent / "raw" / m.NAME / "invocations" / str(time.time_ns())
    work = freeze(m.ROOT, raw)
    validation: Json = dict(receipts=[], coverage={})
    if not fixture:
        from carnot.reporting import matched_corpus_runner_8949 as shared
        from carnot.reporting.v756_branch_qualification import measured

        private = Path(tempfile.mkdtemp(prefix="carnot8960-validation-"))
        m.progress("owned_checks_before")
        validation["receipts"] = run_commands(
            m.ROOT,
            shared.validation_plan(private, producer=m),
            log_dir=raw / "validation",
            heartbeat_s=15,
        )
        if (private / ".coverage").is_file():
            validation["coverage"] = measured([private / ".coverage"], m.OWNED, raw / "measured")
        m.progress("owned_checks_after")
        work["global_test_debt"] = dict(
            receipts=run_commands(
                m.ROOT,
                [
                    CommandSpec(
                        "full_python_suite",
                        (str(m.ROOT / ".venv/bin/pytest"), "tests/python", "-q"),
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
    elapsed = (time.monotonic_ns() - work["started_ns"]) / 1e9
    work["phase_spans"].append(
        dict(
            phase="validation",
            duration_s=elapsed - sum(p["duration_s"] for p in work["phase_spans"]),
        )
    )
    work["current_work_receipt"] = receipt(work)
    atomic_json(raw / "validation.json", validation)
    atomic_json(raw / "work.json", work)
    work["raw_references"] += [m.reference(raw / "work.json"), m.reference(raw / "validation.json")]
    value = m.build(work, validation, fixture)
    atomic_json(raw / "candidate.json", value)
    if not fixture and not value["required_checks_passed"]:
        raise ValueError("owned_checks_failed; candidate retained; primary unchanged")
    m.progress("primary_publication_before")
    published = publish_primary(output, value, replay)
    m.progress("terminal_published", value["completed_count"], value["censored_count"])
    print(
        json.dumps(
            dict(
                **published,
                honest_verdict=value["honest_verdict"],
                online_methods_ready_score=value["online_methods_ready_score"],
            )
        ),
        flush=True,
    )
    return 0
