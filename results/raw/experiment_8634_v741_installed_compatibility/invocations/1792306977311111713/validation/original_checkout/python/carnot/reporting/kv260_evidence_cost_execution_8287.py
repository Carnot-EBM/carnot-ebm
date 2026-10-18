"""REQ-REPORT-8287: freeze commands and replay primitives before atomic publication.

The qualified supervisor preserves real exits and kills bounded process groups.
Private fixtures execute the same CLI and terminal consumers as natural work.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import shutil
import sys
import tempfile
import time
from typing import Any
from unittest.mock import patch

from carnot.reporting import kv260_boundary_execution_8230 as supervisor
from carnot.reporting import kv260_evidence_cost_boundary_8287 as h
from carnot.reporting.current_work_receipt import ZERO_INVOCATION_COUNTS, atomic_json
from carnot.reporting.evidence_features_custody_7980 import checked, reference
from carnot.reporting.experiment_7303_validation_scope import CommandSpec
from carnot.reporting.primary_publication import publish_primary
from carnot.reporting.request_trace_inventory_8200 import operand
from scripts.experiment_template import normalize_artifact_for_template_write

Json = dict[str, Any]
OWNED = [
    "python/carnot/reporting/kv260_evidence_cost_boundary_8287.py",
    "python/carnot/reporting/kv260_evidence_cost_execution_8287.py",
]
TEST = "tests/python/test_kv260_evidence_cost_boundary_8287.py"
execute = supervisor.execute
checksum = supervisor.checksum


def commands(private: Path) -> list[CommandSpec]:
    """Reuse private child coverage, strict lint, spec references and E2E consumers."""
    with (
        patch.object(supervisor, "OWNED", OWNED),
        patch.object(supervisor, "TEST", TEST),
        patch.object(supervisor.h, "CLI", h.CLI),
    ):
        return supervisor.commands(private)


def validators(path: Path) -> list[CommandSpec]:
    """Keep primary terminal auditors unchanged while selecting this replay CLI."""
    with patch.object(supervisor.h, "CLI", h.CLI):
        return supervisor.validators(path)


def replay(path: Path) -> Json:
    """Rehashed summary tampering still differs from fresh primitive reconstruction."""
    value = json.loads(path.read_bytes())
    if value["config"] != h.CONFIG:
        raise ValueError("configuration_drift")
    for ref in (
        value["source_artifact_hashes"] + value["code_config_hashes"] + value["raw_shard_hashes"]
    ):
        checked(ref)
    data = json.loads(checked(value["replay_input_reference"]).read_bytes())
    measured = h.precision(data) if data.get("resources_available", True) else []
    if (
        measured
        != json.loads(checked(value["primitive_reference"]).read_bytes())["fixed_point_rows"]
    ):
        raise ValueError("primitive_drift")
    for key, expected in h.reduce(data, measured).items():
        if value["verdict_class"] == "disqualified" and key in {
            "honest_verdict",
            "verdict_class",
            "kv260_boundary_ready_score",
        }:
            continue
        if value[key] != expected:
            raise ValueError("reduction_drift:" + key)
    if checksum(value) != value["reproducibility_checksum"]:
        raise ValueError("checksum_drift")
    return dict(passed=True, replay_passed=True)


def main(argv: list[str] | None = None) -> int:
    """Missing external inputs publish a complete blocked record after owned checks."""
    began, wall = time.monotonic_ns(), time.time_ns()
    print("[exp8287] start no_model_load", flush=True)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", choices=["20261008"], default="20261008")
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
        raw = output.parent / "raw" / h.NAME / "invocations" / str(time.time_ns())
        raw.mkdir(parents=True, exist_ok=True)
        private = Path(tempfile.mkdtemp(prefix="carnot8287-"))
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
                    'import pathlib,sys,pytest,coverage,ruff,mypy,numpy; p=pathlib.Path(sys.argv[1]);p.write_bytes(b"private");assert p.read_bytes()==b"private";print(sys.version)',
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
                frozen_before_measurement=True,
            ),
        )
        h.progress("preconditions_before", 0, 1)
        pre_receipts = execute(preflight, raw / "preflight")
        data = json.loads(args.input.read_bytes()) if args.input else h.load(args.root, raw)
        data["fixture"] = bool(args.input)
        data["resources_available"] = all(r["passed"] for r in pre_receipts)
        if not data["resources_available"]:
            data["checks"].append(
                operand("resources_and_scratch", private, "available", pre_receipts)
            )
        if args.input:
            h.freeze(args.input, raw, data)
        atomic_json(raw / "replay_inputs.json", data)
        pre_end = time.monotonic_ns()
        h.progress("preconditions_after", 1, 0)
        h.progress("benchmark_before_fixed_point", 0, 1)
        measured = h.precision(data) if data["resources_available"] else []
        h.progress("benchmark_after_fixed_point", 1, 0)
        measurement_end = time.monotonic_ns()
        atomic_json(raw / "primitive_rows.json", dict(fixed_point_rows=measured))
        value = h.reduce(data, measured)
        h.progress("owned_validation_before", 0, len(plan))
        receipts = execute(plan, raw / "validation") if not args.input else []
        h.progress("owned_validation_after", len(receipts), 0)
        passed = all(r["passed"] and r["normal_exit"] for r in receipts)
        if not passed and data["resources_available"]:
            value.update(
                honest_verdict="complete_disqualified_owned_checks",
                verdict_class="disqualified",
                kv260_boundary_ready_score=0,
            )
        health_receipts = execute([health], raw / "health") if not args.input else []
        for p in (private / "checks").glob("*"):
            if p.is_file() and (p.name.startswith(".coverage") or p.name == "coverage.ini"):
                destination = raw / "coverage_evidence" / p.name
                destination.parent.mkdir(exist_ok=True)
                shutil.copyfile(p, destination)
        code_refs: list[Json] = []
        for p in [
            *OWNED,
            h.CLI,
            TEST,
            "python/carnot/reporting/kv260_decision_boundary_8244.py",
            "python/carnot/reporting/kv260_evidence_cost_boundary_8258.py",
            "python/carnot/reporting/kv260_evidence_cost_boundary_8273.py",
            "python/carnot/reporting/kv260_evidence_cost_execution_8273.py",
            "python/carnot/reporting/kv260_evidence_cost_execution_8258.py",
            "python/carnot/reporting/kv260_boundary_execution_8230.py",
            "python/carnot/reporting/recorder_execution_8213.py",
            "python/carnot/reporting/primary_publication.py",
            "python/carnot/verify/evidence_view_kernel_8249.py",
            "python/carnot/verify/margin_energy_training_8237.py",
            "python/carnot/verify/restricted_action_rule_8207.py",
            "scripts/experiment_template.py",
            "ops/exclusion_manifest.yaml",
            "openspec/capabilities/research-reporting/spec.md",
            "openspec/capabilities/verification/spec.md",
            "openspec/change-proposals/research-roadmap-vNEXT.md",
            "research-hardware-wishlist.md",
            "ops/hardware-bringup-prep.md",
            "docs/research-notes/v715-kv260-evidence-cost.md",
        ]:
            h.freeze(h.ROOT / p, raw, dict(references=code_refs))
        ended = time.monotonic_ns()
        side = raw / "terminal_validation.json"
        value.update(
            experiment_id=8287,
            task_id="exp8287-kv260-evidence-cost-boundary",
            milestone="2026.10.715",
            run_date=args.date,
            schema="carnot.kv260_evidence_cost_boundary.v715.v1",
            config=h.CONFIG,
            random_seed=h.CONFIG["seed"],
            duration_s=(ended - began) / 1e9,
            invocation_clocks=dict(
                started_wall_ns=wall,
                ended_wall_ns=time.time_ns(),
                started_monotonic_ns=began,
                ended_monotonic_ns=ended,
            ),
            invocation_argv=list(argv if argv is not None else sys.argv[1:]),
            phase_spans=[
                dict(
                    phase=name,
                    started_monotonic_ns=start,
                    ended_monotonic_ns=end,
                    duration_s=(end - start) / 1e9,
                )
                for name, start, end in [
                    ("preconditions", began, pre_end),
                    ("numerical_mechanics", pre_end, measurement_end),
                    ("validation", measurement_end, ended),
                ]
            ],
            MODEL_SPECS=[],
            inference_substrate="verifier_ensemble_against_cached_candidates"
            if measured
            else "aggregation_from_upstream_artifacts",
            inference_substrate_class="no_model_load",
            model_invocation_counts=dict(ZERO_INVOCATION_COUNTS),
            current_model_calls=0,
            call_ledger=[],
            exposure_scope="reused exposed development and synthetic numerical fixtures; no independent generalization",
            preconditions_checked=data["checks"],
            precondition_receipts=pre_receipts,
            cited_upstream_artifacts=data["cited"],
            source_artifact_hashes=data["references"],
            code_config_hashes=code_refs,
            raw_shard_hashes=[reference(p) for p in sorted(raw.rglob("*")) if p.is_file()],
            replay_input_reference=reference(raw / "replay_inputs.json"),
            primitive_reference=reference(raw / "primitive_rows.json"),
            required_checks_passed=passed,
            validation_receipts=receipts,
            repository_health=health_receipts,
            fixture_mode=bool(args.input),
            flagged_adversarial=False,
            terminal_validation_sidecar_path=str(side),
            methodology="Authenticate qualified V712 hardware and request custody independently from current V715 capture, head, seal and admitted-state operands. Recompute request work and parallel elapsed costs, including each cold start once. Existing quadratic fabric supports no measured compatible spans, giving f=0 only when clocks exist. Q8.8/Q16.16 CPU coefficient storage and counter arithmetic use exact CPU fallback. Synthetic fixtures remain separate from natural benefit and device timing.",
            field_principles={},
        )
        value["field_principles"] = {
            key: "Bind actual invocation bytes; missing operands remain explicit; readiness grants no scientific benefit."
            for key in value
        }
        value["field_principles"].update(
            phase_cost_rows="Request-work phase shares and separately observed parallel makespan charge each cold server once; overlapping costs are never summed as parallel elapsed time.",
            fixed_point_rows="Signed storage with exact CPU nonlinear bases; overflow, raw errors and fallback remain visible; fixtures grant no natural benefit.",
            source_cost_scope="V712/Exp8258 historical costs cannot replace missing V715 three-view acquisition measurements.",
            ideal_whole_request_bound="Amdahl fraction includes only implemented compatible measured spans; unsupported measured work means zero, absent clocks mean unavailable.",
            repository_health="One bounded full-suite diagnostic; unrelated failures cannot establish a global pass.",
        )
        value = normalize_artifact_for_template_write(value)
        value["reproducibility_checksum"] = checksum(value)
        atomic_json(candidate, value)

        def validate(path: Path) -> Json:
            """Use real replay and unchanged auditor exits to qualify candidate bytes."""
            reports = execute(validators(path), raw / ("terminal-" + str(time.time_ns())))
            return dict(
                passed=all(r["passed"] and r["normal_exit"] for r in reports), receipts=reports
            )

        report = validate(candidate)
        if not report["passed"]:
            value.update(
                honest_verdict="complete_disqualified_terminal_validation",
                verdict_class="disqualified",
                kv260_boundary_ready_score=0,
            )
            value["reproducibility_checksum"] = checksum(value)
            atomic_json(raw / "failed_terminal_candidate.json", value)
            atomic_json(side, report)
            return 1
        if output.exists():
            h.freeze(output, raw / "historical_primary", dict(references=[]))
        publication = publish_primary(output, value, validate)
        atomic_json(side, dict(publication=publication, validation=report))
        print("[exp8287] published", flush=True)
        return 0
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(json.dumps(dict(passed=False, error=str(error))), flush=True)
        return 1
