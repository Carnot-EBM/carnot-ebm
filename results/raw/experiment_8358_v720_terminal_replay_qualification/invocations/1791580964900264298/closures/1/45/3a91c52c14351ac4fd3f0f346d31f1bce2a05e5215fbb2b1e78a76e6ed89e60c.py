"""REQ-REPORT-8246: publish an evidence ledger after exact private validation.

The qualified supervisor provides deadlines, process-group cleanup and complete
output hashes. The current invocation performs no model or board operation.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import sys
import tempfile
import time
from typing import Any
from unittest.mock import patch

from carnot.reporting import gatemate_change_ledger_8246 as h
from carnot.reporting.current_work_receipt import ZERO_INVOCATION_COUNTS, atomic_json
from carnot.reporting.evidence_features_custody_7980 import checked, reference
from carnot.reporting.experiment_7303_validation_scope import CommandSpec
from carnot.reporting.primary_publication import publish_primary, validate_primary
from scripts.experiment_template import normalize_artifact_for_template_write

Json = dict[str, Any]
TEST = "tests/python/test_gatemate_change_ledger_8246.py"
OWNED = [
    "python/carnot/reporting/gatemate_change_ledger_8246.py",
    "python/carnot/reporting/gatemate_ledger_execution_8246.py",
]
execute = h.previous_cli.execute
checksum = h.previous_cli.checksum


def commands(private: Path) -> list[CommandSpec]:
    """Reuse the qualified plan, including subprocess coverage and private E2E."""
    with (
        patch.object(h.previous_cli, "h", h),
        patch.object(h.previous_cli, "OWNED", OWNED),
        patch.object(h.previous_cli, "TEST", TEST),
    ):
        return list(h.previous_cli.commands(private))


def validators(path: Path) -> list[CommandSpec]:
    """Unchanged auditors check the bytes that the current replay CLI reads."""
    with patch.object(h.previous_cli, "h", h):
        return list(h.previous_cli.validators(path))


def replay(path: Path) -> Json:
    """Authenticate primitives and rebuild the ledger before trusting a terminal summary."""
    value = json.loads(path.read_bytes())
    if value["config"] != h.CONFIG:
        raise ValueError("configuration_drift")
    for ref in (
        value["source_artifact_hashes"] + value["code_config_hashes"] + value["raw_shard_hashes"]
    ):
        checked(ref)
    data = json.loads(checked(value["replay_input_reference"]).read_bytes())
    h.verify_primitives(data)
    reduction = h.reduce(data)
    if value["verdict_class"] == "disqualified":
        reduction["future_probe_contract"]["eligible"] = False
    for key, expected in reduction.items():
        if value["verdict_class"] == "disqualified" and key in {
            "honest_verdict",
            "verdict_class",
            "gatemate_obligation_ready_score",
        }:
            continue
        if value[key] != expected:
            raise ValueError("reduction_drift:" + key)
    if value["reproducibility_checksum"] != checksum(value):
        raise ValueError("checksum_drift")
    return dict(passed=True, replay_passed=True)


def main(argv: list[str] | None = None) -> int:
    """Keep failures reviewable while exposing only normally validated terminal bytes."""
    began, wall = time.monotonic_ns(), time.time_ns()
    h.progress("start_no_model_load", 0, 1)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", choices=["20261007"], default="20261007")
    parser.add_argument("--root", type=Path, default=h.ROOT)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--input", type=Path)
    parser.add_argument("--cold-replay", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.cold_replay:
            print(json.dumps(replay(args.cold_replay)), flush=True)
            return 0
        output = (args.output or args.root / "results" / (h.NAME + ".json")).absolute()
        if args.input and any(
            output.resolve().is_relative_to((r / "results").resolve()) for r in [h.ROOT, args.root]
        ):
            raise ValueError("private_fixture_requires_private_output")
        raw = output.parent / "raw" / h.NAME / "invocations" / str(wall)
        raw.mkdir(parents=True, exist_ok=True)
        private = Path(tempfile.mkdtemp(prefix="carnot8246-"))
        private.chmod(0o700)
        candidate = private / (h.NAME + ".json")
        plan = commands(private / "checks")
        health = CommandSpec(
            "repository_health_once",
            (str(h.ROOT / ".venv/bin/pytest"), "tests/python", "-q"),
            "unrelated_repository_health",
            180,
        )
        preflight = [
            CommandSpec(
                "resources_and_scratch",
                (
                    sys.executable,
                    "-c",
                    'import pathlib,shutil,sys,pytest,coverage,ruff,mypy;p=pathlib.Path(sys.argv[1]);p.write_bytes(b"private scratch");assert p.read_bytes()==b"private scratch";assert sys.version_info>=(3,11);print(sys.version);print(shutil.disk_usage(p.parent))',
                    str(private / "writable"),
                ),
                "preconditions",
                15,
            )
        ]
        atomic_json(
            raw / "validation_commands.json",
            dict(
                commands=[asdict(s) for s in plan],
                preconditions=[asdict(s) for s in preflight],
                terminal=[asdict(s) for s in validators(candidate)],
                repository_health=asdict(health),
                config=h.CONFIG,
                frozen_before_measurement=True,
            ),
        )
        h.progress("preconditions_before", 0, 1)
        pre_receipts = execute(preflight, raw / "preflight")
        data = json.loads(args.input.read_bytes()) if args.input else h.load(args.root, raw)
        if args.input:
            data["fixture"] = True
            h.freeze(args.input, raw, data)
        atomic_json(raw / "replay_inputs.json", data)
        pre_end = time.monotonic_ns()
        h.progress("preconditions_after", 1, 0)
        h.progress("reduction_before", 0, 1)
        value = h.reduce(data)
        reduce_end = time.monotonic_ns()
        h.progress("reduction_after", 1, 0)
        receipts = execute(plan, raw / "validation") if not args.input else []
        passed = all(r["passed"] and r["normal_exit"] for r in pre_receipts + receipts)
        if not passed:
            value.update(
                honest_verdict="complete_disqualified_owned_checks",
                verdict_class="disqualified",
                gatemate_obligation_ready_score=0,
            )
            value["future_probe_contract"]["eligible"] = False
        health_receipts = execute([health], raw / "health") if not args.input else []
        ended = time.monotonic_ns()
        terminal_side = raw / "terminal_validation.json"
        code_paths = [
            *OWNED,
            h.CLI,
            TEST,
            "python/carnot/reporting/primary_publication.py",
            "scripts/experiment_template.py",
            "python/carnot/reporting/gatemate_continuity_8232.py",
            "python/carnot/reporting/gatemate_execution_8232.py",
            "python/carnot/reporting/recorder_execution_8213.py",
            "scripts/experiments/experiment_7146_v627_gatemate_changed_state.py",
            "scripts/adversarial_verify.py",
            "scripts/verdict_row_consistency_lint.py",
        ]
        value.update(
            experiment_id=8246,
            task_id="exp8246-gatemate-change-ledger",
            milestone="2026.10.712",
            run_date=args.date,
            schema="carnot.gatemate_change_ledger.v712.v1",
            config=h.CONFIG,
            random_seed=h.CONFIG["seed"],
            duration_s=(ended - began) / 1e9,
            invocation=dict(
                pid=os.getpid(),
                started_wall_ns=wall,
                started_monotonic_ns=began,
                ended_monotonic_ns=ended,
                argv=list(argv) if argv is not None else sys.argv,
            ),
            phase_spans=[
                dict(
                    phase=name,
                    started_monotonic_ns=start,
                    ended_monotonic_ns=end,
                    duration_s=(end - start) / 1e9,
                )
                for name, start, end in [
                    ("preconditions", began, pre_end),
                    ("reduction", pre_end, reduce_end),
                    ("validation", reduce_end, ended),
                ]
            ],
            inference_substrate="aggregation_from_upstream_artifacts",
            inference_substrate_class="no_model_load",
            MODEL_SPECS=[],
            trained_head_specs=[],
            model_invocation_counts=dict(ZERO_INVOCATION_COUNTS),
            current_model_calls=0,
            model_invoked=False,
            execution_venue="host",
            fixture_mode=bool(data["fixture"]),
            exposure_scope="stored GateMate custody and established operator receipt documents",
            preconditions_checked=data["checks"],
            precondition_receipts=pre_receipts,
            gate_check_summary=data["checks"],
            cited_upstream_artifacts=data["cited"],
            source_artifact_hashes=data["references"],
            code_config_hashes=[reference(h.ROOT / p) for p in code_paths],
            raw_shard_hashes=[
                reference(raw / p) for p in ["replay_inputs.json", "validation_commands.json"]
            ]
            + [
                reference(Path(r[k]))
                for r in pre_receipts + receipts + health_receipts
                for k in ["stdout_path", "stderr_path"]
                if k in r
            ],
            replay_input_reference=reference(raw / "replay_inputs.json"),
            required_checks_passed=passed,
            validation_receipts=receipts,
            repository_health=dict(
                receipts=health_receipts,
                required_for_owned_readiness=False,
                passed=all(r["passed"] for r in health_receipts) if health_receipts else None,
            ),
            flagged_adversarial=False,
            terminal_validation_sidecar_path=str(terminal_side),
            methodology="Authenticated Exp8232 receipt/frontier, immutable document comparison and qualified dry-run parsing of new operator evidence. No model load, JTAG retry, flashing, benchmark or device execution.",
            claim_scope="Evidence ledger; future physical preflight remains unexecuted",
        )
        value = normalize_artifact_for_template_write(value)
        value["field_principles"] = {
            k: "Readiness certifies owned evidence checks and grants no scientific benefit."
            if k.endswith("score")
            else "Each obligation retains its missing status, disposition and numerator/denominator."
            if k.endswith("count") or k in {"rows", "board_rows", "gatemate_obligation"}
            else "Bind the current invocation to exact authenticated inputs, fields, code, clocks and receipts."
            for k in value
        }
        value["field_principles"].update(
            physical_change_frontier="Only previously unseen operator evidence after Exp8232 can qualify future preflight.",
            change_ledger="Compare actual document hashes; unchanged files do not imply or trigger physical work.",
            future_probe_contract="A falsifiable future detection/flash/smoke contract grants no current device execution.",
            model_invocation_counts="Historical model provenance creates no current calls.",
            field_principles="Explain each field's evidence constraint.",
            reproducibility_checksum="Bind all candidate fields without hashing the checksum into itself.",
        )
        value["reproducibility_checksum"] = checksum(value)
        validate_primary(value, output)
        atomic_json(candidate, value)

        def validate(path: Path) -> Json:
            """Replay and unchanged auditors must exit normally before exposing these bytes."""
            results = execute(validators(path), raw / ("terminal-" + str(time.time_ns())))
            return dict(
                passed=all(r["passed"] and r["normal_exit"] for r in results), receipts=results
            )

        report = validate(candidate)
        if not report["passed"]:
            atomic_json(
                raw / "failed_terminal_candidate.json",
                dict(
                    value,
                    honest_verdict="complete_disqualified_terminal_validation",
                    verdict_class="disqualified",
                    required_checks_passed=False,
                    gatemate_obligation_ready_score=0,
                ),
            )
            atomic_json(terminal_side, report)
            return 1
        publication = publish_primary(output, value, validate)
        atomic_json(
            terminal_side, dict(publication=publication, private_candidate_validation=report)
        )
        h.progress("published", 1, 0)
        return 0
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(json.dumps(dict(passed=False, error=str(error))), flush=True)
        return 1
