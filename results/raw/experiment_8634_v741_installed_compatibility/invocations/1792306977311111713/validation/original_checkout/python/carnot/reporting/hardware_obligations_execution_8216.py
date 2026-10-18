"""REQ-REPORT-8216: qualified supervision and atomic publication bind the report.

The current process reads artifacts and measures host primitives. Cached model
provenance cannot add a current model invocation or a board execution.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
import tempfile
import time
from typing import Any
from unittest.mock import patch

from carnot.reporting import hardware_workload_obligations_8216 as h
from carnot.reporting import recorder_execution_8213 as qualified
from carnot.reporting.current_work_receipt import (
    ZERO_INVOCATION_COUNTS,
    atomic_json,
    canonical_hash,
)
from carnot.reporting.evidence_features_custody_7980 import checked, reference
from carnot.reporting.experiment_7303_validation_scope import CommandSpec
from carnot.reporting.primary_publication import publish_primary
from scripts.experiment_template import normalize_artifact_for_template_write

Json = dict[str, Any]
TEST = "tests/python/test_hardware_workload_obligations_8216.py"
OWNED = [
    "python/carnot/reporting/hardware_workload_obligations_8216.py",
    "python/carnot/reporting/hardware_obligations_execution_8216.py",
]
execute = qualified.execute


def commands(private: Path) -> list[CommandSpec]:
    """Reuse exact owned coverage, types, spec and private E2E command construction."""
    with (
        patch.object(qualified, "MODULES", OWNED),
        patch.object(qualified, "TEST", TEST),
        patch.object(qualified.e, "CLI", h.CLI),
    ):
        plan = qualified.validation_plan(private)
    config = private / "coverage.ini"
    config.write_text(config.read_text() + "[report]\nexclude_lines =\n")
    return plan


def validators(path: Path) -> list[CommandSpec]:
    """The unchanged auditors and a fresh CLI process inspect the checked bytes."""
    with patch.object(qualified.e, "CLI", h.CLI):
        return qualified.validators(path)


def stable(rows: list[Json]) -> list[Json]:
    """Recorded timer values remain observations while decisions replay exactly."""
    return [{k: v for k, v in r.items() if k not in {"elapsed_ns", "fallback_ns"}} for r in rows]


def replay(path: Path) -> Json:
    """Recompute primitive decisions and costs, rejecting even rehashed summaries."""
    value = json.loads(path.read_bytes())
    if value["config"] != h.CONFIG:
        raise ValueError("configuration_drift")
    for ref in (
        value["source_artifact_hashes"] + value["raw_shard_hashes"] + value["code_config_hashes"]
    ):
        checked(ref)
    data = json.loads(checked(value["replay_input_reference"]).read_bytes())
    measured = h.precision(data)
    if stable(measured) != stable(value["precision_rows"]):
        raise ValueError("precision_drift")
    primitive = json.loads(checked(value["primitive_reference"]).read_bytes())
    if primitive["precision_rows"] != value["precision_rows"]:
        raise ValueError("primitive_drift")
    fresh = h.reduce(data, primitive["precision_rows"], primitive["storage_probe_rows"])
    for key, expected in fresh.items():
        if value["verdict_class"] == "disqualified" and key in {
            "honest_verdict",
            "verdict_class",
            "hardware_boundary_ready_score",
        }:
            continue
        if value[key] != expected:
            raise ValueError("reduction_drift:" + key)
    if value["reproducibility_checksum"] != qualified.checksum(value):
        raise ValueError("checksum_drift")
    return dict(passed=True, rows_checksum=canonical_hash(stable(measured)))


def main(argv: list[str] | None = None) -> int:
    """Freeze checks before measurement and publish only validated terminal bytes."""
    began = time.monotonic_ns()
    h.progress("start_no_model_load", 0, 1)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", choices=["20261006"], default="20261006")
    parser.add_argument("--root", type=Path, default=h.ROOT)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--input", type=Path)
    parser.add_argument("--cold-replay", type=Path)
    parser.add_argument("--repository-health-receipt", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.cold_replay:
            print(json.dumps(replay(args.cold_replay)), flush=True)
            return 0
        output = (args.output or args.root / "results" / (h.NAME + ".json")).absolute()
        if args.input and output.is_relative_to(h.ROOT / "results"):
            raise ValueError("private_fixture_requires_private_output")
        raw = output.parent / "raw" / h.NAME / "invocations" / str(time.time_ns())
        raw.mkdir(parents=True, exist_ok=True)
        private = Path(tempfile.mkdtemp(prefix="carnot8216-"))
        private.chmod(0o700)
        plan = commands(private)
        health = CommandSpec(
            "repository_health_once",
            (str(h.ROOT / ".venv/bin/pytest"), "tests/python", "-q"),
            "unrelated_repository_health",
            180,
        )
        atomic_json(
            raw / "validation_commands.json",
            dict(
                commands=[asdict(s) for s in plan],
                terminal=[asdict(s) for s in validators(private / (h.NAME + ".json"))],
                repository_health=asdict(health),
                config=h.CONFIG,
                frozen_before_measurement=True,
            ),
        )
        preconditions = [
            CommandSpec(
                "python_environment",
                (
                    sys.executable,
                    "-c",
                    "import pytest,coverage,ruff,mypy,sys; print(sys.version); assert sys.version_info >= (3,11)",
                ),
                "preconditions",
                15,
            ),
            CommandSpec(
                "private_scratch",
                (
                    sys.executable,
                    "-c",
                    'import pathlib,sys;p=pathlib.Path(sys.argv[1]);p.write_bytes(b"writable");print(p.read_bytes())',
                    str(private / "probe"),
                ),
                "preconditions",
                10,
            ),
        ]
        h.progress("preconditions_before", 0, 1)
        preflight = execute(preconditions, raw / "preflight")
        data = json.loads(args.input.read_bytes()) if args.input else h.load(args.root, raw)
        data["fixture"] = bool(args.input)
        if not all(r["passed"] for r in preflight):
            data["branches"] = dict.fromkeys(h.PRODUCERS, False)
            data["heads"] = []
            data["checks"] += [
                dict(
                    path=r.get("stdout_path", ""),
                    artifact_field=r.get("name"),
                    op="==",
                    expected=0,
                    observed=r.get("exit_code"),
                    passed=False,
                )
                for r in preflight
                if not r["passed"]
            ]
        atomic_json(raw / "replay_inputs.json", data)
        h.progress("preconditions_after", 1, 0)
        measured = h.precision(data)
        probes = []
        if data["branches"]["learning"]:
            payload = checked(data["learning"]["storage_source"]).read_bytes()
            probes.append(h.storage_probe(payload, private / "learning.bin", "learning"))
        primitive = dict(precision_rows=measured, storage_probe_rows=probes)
        atomic_json(raw / "primitive_rows.json", primitive)
        value = h.reduce(data, measured, probes)
        receipts = (
            execute(plan, raw / "validation")
            if not args.input and any(data["branches"].values())
            else []
        )
        passed = all(r["passed"] and r["normal_exit"] for r in receipts)
        if not passed:
            value.update(
                honest_verdict="complete_disqualified_owned_checks",
                verdict_class="disqualified",
                hardware_boundary_ready_score=0,
            )
        health_receipts = []
        if args.repository_health_receipt:
            health_receipts = json.loads(args.repository_health_receipt.read_bytes())["receipts"]
            for receipt in health_receipts:
                for field in ["stdout", "stderr"]:
                    ref = dict(path=receipt[field + "_path"], sha256=receipt[field + "_sha256"])
                    frozen = h.copy_bytes(checked(ref), raw)
                    data["references"].append(frozen)
                    receipt[field + "_path"] = frozen["frozen_path"]
        elif not args.input and any(data["branches"].values()):
            health_receipts = execute([health], raw / "health")
        terminal_side = raw / "terminal_validation.json"
        value.update(
            experiment_id=8216,
            task_id="exp8216-hardware-workload-obligations",
            milestone="2026.10.709",
            run_date=args.date,
            schema="carnot.hardware_obligations.v709.v1",
            config=h.CONFIG,
            random_seed=h.CONFIG["seed"],
            duration_s=(time.monotonic_ns() - began) / 1e9,
            measurement_clocks=dict(
                started_monotonic_ns=began, ended_monotonic_ns=time.monotonic_ns()
            ),
            MODEL_SPECS=[],
            inference_substrate="verifier_ensemble_against_cached_candidates"
            if measured
            else "aggregation_from_upstream_artifacts",
            phase_substrates=dict(
                precision="verifier_ensemble_against_cached_candidates",
                reduction="aggregation_from_upstream_artifacts",
            ),
            inference_substrate_class="no_model_load",
            model_invocation_counts=dict(ZERO_INVOCATION_COUNTS),
            current_model_calls=0,
            call_ledger=[],
            exposure_scope="exposed development evidence",
            preconditions_checked=data["checks"],
            precondition_receipts=preflight,
            gate_check_summary=data["checks"] + value["cost_obligation_checks"],
            cited_upstream_artifacts=data["cited"],
            source_artifact_hashes=[
                dict(
                    r,
                    path=r.get("frozen_path", r["path"]),
                    original_path=r.get("original_path", r["path"]),
                )
                for r in data["references"]
            ],
            code_config_hashes=[
                reference(h.ROOT / p)
                for p in [
                    *OWNED,
                    h.CLI,
                    "python/carnot/verify/restricted_action_rule_8207.py",
                    "python/carnot/verify/prospective_service_8214.py",
                    "python/carnot/reporting/primary_publication.py",
                    "scripts/experiment_template.py",
                ]
            ],
            raw_shard_hashes=[
                reference(raw / p)
                for p in ["replay_inputs.json", "primitive_rows.json", "validation_commands.json"]
            ],
            replay_input_reference=reference(raw / "replay_inputs.json"),
            primitive_reference=reference(raw / "primitive_rows.json"),
            required_checks_passed=passed,
            validation_receipts=receipts,
            repository_health=health_receipts,
            flagged_adversarial=False,
            terminal_validation_sidecar_path=str(terminal_side),
            methodology="Independent cached-branch authentication; current converted CPU multiply/accumulate with original acceptance permission; complete shared-acquisition service accounting. Unknown original learning component clocks remain unavailable.",
            claim_scope="Software reducer and host primitive limits; no current model or board execution",
            deployment_demand_observed=False,
            deployment_claim=False,
            field_principles=dict(
                identity="Exact task and immutable invocation bytes.",
                verdict="Owned failed checks disqualify; optional blocked branches retain their operands.",
                counts="Every measured precision arm and missing branch slot remains in its denominator.",
                provenance="Source copies preserve original scope and byte hashes.",
                substrate="No model load or current board execution.",
                readiness="Qualified reducer only; each branch keeps its own qualification.",
                precision="Acceptance permission follows conversion and CPU fallback.",
                costs="Retain acquisition and durable costs; absent timers stay null.",
                generalization="Reused development data earn zero generalization credit.",
                boards="Historical obligations do not imply current dispatch.",
            ),
        )
        value = normalize_artifact_for_template_write(value)
        value["reproducibility_checksum"] = qualified.checksum(value)
        candidate = private / (h.NAME + ".json")
        atomic_json(candidate, value)

        def validate(path: Path) -> Json:
            """Atomic publication accepts only unchanged auditors' exact candidate bytes."""
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
                    verdict_class="disqualified",
                    honest_verdict="complete_disqualified_terminal_validation",
                    hardware_boundary_ready_score=0,
                ),
            )
            atomic_json(terminal_side, report)
            return 1
        publication = publish_primary(output, value, validate)
        atomic_json(terminal_side, dict(publication=publication))
        h.progress("published", 1, 0)
        return 0
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(json.dumps(dict(passed=False, error=str(error))), flush=True)
        return 1
