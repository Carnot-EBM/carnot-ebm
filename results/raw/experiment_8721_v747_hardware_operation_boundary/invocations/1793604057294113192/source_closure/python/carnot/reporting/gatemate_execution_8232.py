"""REQ-REPORT-8232: terminal publication binds a documentation audit to its bytes.

Qualified helpers supervise children and preserve full stream hashes. The current
run neither loads a model nor runs a board command. Private fixture outputs are
kept outside results so parser tests cannot become current hardware evidence.
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

from carnot.reporting import gatemate_continuity_8232 as h
from carnot.reporting import recorder_execution_8213 as qualified
from carnot.reporting.current_work_receipt import (
    ZERO_INVOCATION_COUNTS,
    atomic_json,
    canonical_hash,
)
from carnot.reporting.evidence_features_custody_7980 import checked, reference
from carnot.reporting.experiment_7303_validation_scope import CommandSpec
from carnot.reporting.primary_publication import publish_primary, validate_primary
from carnot.verify import request_recorder_8213 as supervisor_config
from scripts.experiments import experiment_7146_v627_gatemate_changed_state as legacy
from scripts.experiment_template import normalize_artifact_for_template_write

Json = dict[str, Any]
TEST = "tests/python/test_gatemate_continuity_8232.py"
OWNED = [
    "python/carnot/reporting/gatemate_continuity_8232.py",
    "python/carnot/reporting/gatemate_execution_8232.py",
]
execute = qualified.execute


def checksum(value: Json) -> str:
    """Bind all recorded fields without including the digest in its own input."""
    return canonical_hash({k: v for k, v in value.items() if k != "reproducibility_checksum"})


def commands(private: Path) -> list[CommandSpec]:
    """Freeze qualified coverage including actual CLI children and private E2E."""
    private.mkdir(parents=True, exist_ok=True)
    with (
        patch.object(qualified, "MODULES", OWNED),
        patch.object(qualified, "TEST", TEST),
        patch.object(supervisor_config, "CLI", h.CLI),
    ):
        plan = qualified.validation_plan(private)
    config = private / "coverage.ini"
    config.write_text(config.read_text() + "[report]\nexclude_lines =\n")
    return [
        CommandSpec(
            s.name,
            (s.argv[0], h.CLI, *s.argv[1:], "--config-file=/dev/null")
            if s.name == "changed_module_mypy"
            else s.argv,
            s.scope,
            s.timeout_s,
        )
        for s in plan
    ]


def validators(path: Path) -> list[CommandSpec]:
    """Fresh processes use unchanged terminal auditors and this task's replay CLI."""
    with patch.object(supervisor_config, "CLI", h.CLI):
        return qualified.validators(path)


def replay(path: Path) -> Json:
    """Recompute a board obligation from frozen primitives, rejecting summary drift."""
    value = json.loads(path.read_bytes())
    if value["config"] != h.CONFIG:
        raise ValueError("configuration_drift")
    for ref in (
        value["source_artifact_hashes"] + value["code_config_hashes"] + value["raw_shard_hashes"]
    ):
        checked(ref)
    data = json.loads(checked(value["replay_input_reference"]).read_bytes())
    for row in data["receipt_rows"]:
        fresh = legacy.receipt_row(
            row["raw_receipt"],
            source_path=row["source_path"],
            row_index=int(row["row_id"].split("-")[1]),
            cutoff_date=h.CONFIG["cutoff_date"],
            run_date=h.CONFIG["run_date"],
            structured=row["structured_receipt"],
        )
        if fresh != row:
            raise ValueError("receipt_drift")
    valid = [r for r in data["receipt_rows"] if r["valid"]]
    if bool(valid) != bool(data["physical_change"].get("exists")):
        raise ValueError("physical_change_drift")
    for key, expected in h.reduce(data).items():
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
    return dict(passed=True, replay_passed=True, board_checksum=canonical_hash(value["rows"]))


def main(argv: list[str] | None = None) -> int:
    """Publish only a normally validated candidate; external blocking stays terminal."""
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
        if args.input and output.resolve().is_relative_to((h.ROOT / "results").resolve()):
            raise ValueError("private_fixture_requires_private_output")
        raw = output.parent / "raw" / h.NAME / "invocations" / str(wall)
        raw.mkdir(parents=True, exist_ok=True)
        private = Path(tempfile.mkdtemp(prefix="carnot8232-"))
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
        code_refs = [
            reference(h.ROOT / p)
            for p in [
                *OWNED,
                h.CLI,
                TEST,
                "python/carnot/reporting/primary_publication.py",
                "scripts/experiment_template.py",
                "python/carnot/reporting/recorder_execution_8213.py",
                "scripts/experiments/experiment_7146_v627_gatemate_changed_state.py",
                "scripts/adversarial_verify.py",
                "scripts/verdict_row_consistency_lint.py",
            ]
        ]
        h.progress("preconditions_before", 0, 1)
        pre_receipts = execute(preflight, raw / "preflight")
        data = json.loads(args.input.read_bytes()) if args.input else h.load(args.root, raw)
        data["fixture"] = bool(args.input)
        if args.input:
            h.freeze(args.input, raw, data)
        atomic_json(raw / "replay_inputs.json", data)
        pre_end = time.monotonic_ns()
        h.progress("preconditions_after", 1, 0)
        h.progress("reduce_before", 0, 1)
        value = h.reduce(data)
        reduce_end = time.monotonic_ns()
        h.progress("reduce_after", 1, 0)
        receipts = execute(plan, raw / "validation") if not args.input else []
        passed = all(r["passed"] and r["normal_exit"] for r in pre_receipts + receipts)
        if not passed:
            value.update(
                honest_verdict="complete_disqualified_owned_checks",
                verdict_class="disqualified",
                gatemate_obligation_ready_score=0,
            )
        health_receipts = execute([health], raw / "health") if not args.input else []
        ended = time.monotonic_ns()
        terminal_side = raw / "terminal_validation.json"
        value.update(
            experiment_id=8232,
            task_id="exp8232-gatemate-continuity",
            milestone="2026.10.711",
            run_date=args.date,
            schema="carnot.gatemate_continuity.v711.v1",
            config=h.CONFIG,
            random_seed=h.CONFIG["seed"],
            duration_s=(ended - began) / 1e9,
            invocation=dict(
                pid=os.getpid(),
                started_wall_ns=wall,
                started_monotonic_ns=began,
                ended_monotonic_ns=ended,
                argv=list(argv) if argv is not None else sys.argv[1:],
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
            fixture_mode=bool(args.input),
            exposure_scope="historical hardware custody and operator documentation",
            preconditions_checked=data["checks"],
            precondition_receipts=pre_receipts,
            gate_check_summary=data["checks"],
            cited_upstream_artifacts=data["cited"],
            source_artifact_hashes=data["references"],
            code_config_hashes=code_refs,
            raw_shard_hashes=[
                reference(raw / p) for p in ["replay_inputs.json", "validation_commands.json"]
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
            methodology="Authenticated historical GateMate custody and dry-run structured operator receipt audit. No model, benchmark, JTAG retry, flash or device execution.",
            claim_scope="Documentation audit only; recorded changes require future device preflight",
        )
        value = normalize_artifact_for_template_write(value)
        principles = {
            "identity": "Bind this task to the actual invocation and date.",
            "verdict": "Completed external blocking differs from failed owned checks and unfinished work.",
            "evidence": "Preserve exact input bytes, missing status, imported fields and clocks.",
            "models": "Historical model provenance creates no current calls or trained heads.",
            "counts": "Keep the excluded board in its denominator without inventing a failed probe.",
            "readiness": "Audit readiness grants no device acceptance or scientific benefit.",
            "reopen": "Only physical evidence can open future preflight; host files cannot prove device success.",
            "checks": "Required checks bind exact argv, exits and full stream hashes to candidate bytes.",
        }
        value["field_principles"] = {
            k: principles[
                "models"
                if k
                in {
                    "MODEL_SPECS",
                    "trained_head_specs",
                    "model_invocation_counts",
                    "inference_substrate",
                    "inference_substrate_class",
                    "current_model_calls",
                }
                else "counts"
                if k.endswith("count") or k in {"rows", "board_rows"}
                else "verdict"
                if k in {"honest_verdict", "verdict_class"}
                else "checks"
                if k
                in {
                    "validation_receipts",
                    "acceptance_gates",
                    "required_checks_passed",
                    "flagged_adversarial",
                    "terminal_validation_sidecar_path",
                }
                else "readiness"
                if k.endswith("score") or k == "verifier_is_oracle"
                else "reopen"
                if k
                in {
                    "gatemate_obligation",
                    "physical_change_evidence",
                    "physical_change_receipt_rows",
                    "reopen_contract_path",
                }
                else "identity"
                if k in {"experiment_id", "task_id", "milestone", "run_date", "invocation"}
                else "evidence"
            ]
            for k in value
        }
        value["field_principles"].update(
            field_principles="Explain the failure prevented by each recorded field.",
            reproducibility_checksum="Bind all candidate fields to one independently recomputed digest.",
        )
        value["reproducibility_checksum"] = checksum(value)
        validate_primary(value, output)
        atomic_json(candidate, value)

        def validate(path: Path) -> Json:
            """Unchanged auditors and a fresh replay must exit normally on exact bytes."""
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
