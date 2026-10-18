"""REQ-REPORT-8231: publish byte-bound state inventory after private validation.

Qualified helpers supply bounded process groups, heartbeats and stream hashes.
Current work loads no model and gives no device or generalization benefit credit.
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

from carnot.reporting import polarfire_state_boundary_8231 as h
from carnot.reporting import recorder_execution_8213 as qualified
from carnot.reporting.current_work_receipt import (
    ZERO_INVOCATION_COUNTS,
    atomic_json,
    canonical_hash,
)
from carnot.reporting.evidence_features_custody_7980 import checked, reference
from carnot.reporting.experiment_7303_validation_scope import CommandSpec
from carnot.reporting.primary_publication import publish_primary
from carnot.verify import request_recorder_8213 as supervisor_config
from scripts.experiment_template import normalize_artifact_for_template_write

Json = dict[str, Any]
TEST = "tests/python/test_polarfire_state_boundary_8231.py"
OWNED = [
    "python/carnot/reporting/polarfire_state_boundary_8231.py",
    "python/carnot/reporting/polarfire_boundary_execution_8231.py",
]
execute = qualified.execute


def checksum(value: Json) -> str:
    """Bind every recorded field except the checksum itself."""
    return canonical_hash({k: v for k, v in value.items() if k != "reproducibility_checksum"})


def commands(private: Path) -> list[CommandSpec]:
    """Scope coverage to added statements while including actual CLI children."""
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
    """Use the unchanged auditors and this task's own fresh replay CLI."""
    with patch.object(supervisor_config, "CLI", h.CLI):
        return qualified.validators(path)


def replay(path: Path) -> Json:
    """Recompute deterministic primitives so a rehashed summary cannot pass."""
    value = json.loads(path.read_bytes())
    if value["config"] != h.CONFIG:
        raise ValueError("configuration_drift")
    for ref in (
        value["source_artifact_hashes"] + value["code_config_hashes"] + value["raw_shard_hashes"]
    ):
        checked(ref)
    data = json.loads(checked(value["replay_input_reference"]).read_bytes())
    primitive = json.loads(checked(value["primitive_reference"]).read_bytes())
    measured = primitive["serialized_state_rows"]
    h.verify_primitives(data, measured)
    if json.loads(checked(value["transfer_contract_reference"]).read_bytes()) != h.contract():
        raise ValueError("contract_drift")
    for key, expected in h.reduce(data, measured).items():
        if value["verdict_class"] == "disqualified" and key in {
            "honest_verdict",
            "verdict_class",
            "polarfire_boundary_ready_score",
        }:
            continue
        if value[key] != expected:
            raise ValueError("reduction_drift:" + key)
    if value["reproducibility_checksum"] != checksum(value):
        raise ValueError("checksum_drift")
    return dict(passed=True, rows_checksum=canonical_hash(measured), host_restart_parity=True)


def main(argv: list[str] | None = None) -> int:
    """Freeze commands before work; publish only after every required terminal exit."""
    began, wall = time.monotonic_ns(), time.time_ns()
    h.progress("start_no_model_load", 0, 1)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", choices=["20261007"], default="20261007")
    parser.add_argument("--root", type=Path, default=h.ROOT)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--input", type=Path)
    parser.add_argument("--health-receipt", type=Path)
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
        private = Path(tempfile.mkdtemp(prefix="carnot8231-"))
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
                    'import pathlib,shutil,sys,pytest,coverage,ruff,mypy; p=pathlib.Path(sys.argv[1]);p.write_bytes(b"private scratch");assert p.read_bytes()==b"private scratch";assert sys.version_info>=(3,11);print(sys.version);print(shutil.disk_usage(p.parent))',
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
                repository_health_source=str(args.health_receipt) if args.health_receipt else None,
                config=h.CONFIG,
                frozen_before_measurement=True,
            ),
        )
        h.progress("preconditions_before", 0, 1)
        pre_receipts = execute(preflight, raw / "preflight")
        data = json.loads(args.input.read_bytes()) if args.input else h.load(args.root, raw)
        data["fixture"] = bool(args.input)
        if args.input:
            h.freeze(args.input, raw, data)
        health_source = None
        reused_health: list[Json] = []
        if args.health_receipt:
            saved_health = h.freeze(args.health_receipt, raw, data)
            health_source = reference(saved_health)
            reused_health = json.loads(saved_health.read_bytes())["repository_health"]
            for receipt in reused_health:
                for stream in ["stdout", "stderr"]:
                    h.freeze(
                        checked(
                            dict(path=receipt[stream + "_path"], sha256=receipt[stream + "_sha256"])
                        ),
                        raw,
                        data,
                    )
        if not all(r["passed"] for r in pre_receipts):
            data["checks"].append(
                h.operand("resources_and_scratch", private / "writable", True, False)
            )
            data["branches"]["resources_and_scratch"] = dict(
                ready=False, verdict="missing_resources", failed_operands=pre_receipts
            )
            data["cases"] = []
        pre_end = time.monotonic_ns()
        atomic_json(raw / "replay_inputs.json", data)
        atomic_json(raw / "transfer_contract.json", h.contract())
        h.progress("preconditions_after", 1, 0)
        measured = h.measure(data, raw)
        measurement_end = time.monotonic_ns()
        atomic_json(raw / "primitive_rows.json", dict(serialized_state_rows=measured))
        value = h.reduce(data, measured)
        h.progress("owned_validation_before", 0, len(plan))
        receipts = execute(plan, raw / "validation") if not args.input else []
        passed = all(r["passed"] and r["normal_exit"] for r in receipts)
        if not passed:
            value.update(
                honest_verdict="complete_disqualified_owned_checks",
                verdict_class="disqualified",
                polarfire_boundary_ready_score=0,
            )
        h.progress("owned_validation_after", len(receipts), 0)
        health_receipts = (
            reused_health
            if args.health_receipt
            else execute([health], raw / "health")
            if not args.input
            else []
        )
        side = raw / "terminal_validation.json"
        code_refs: list[Json] = []
        for p in [
            *OWNED,
            h.CLI,
            TEST,
            "python/carnot/verify/utility_kernel_8221.py",
            "python/carnot/verify/utility_patch_methods_8219.py",
            "python/carnot/reporting/primary_publication.py",
            "python/carnot/reporting/recorder_execution_8213.py",
            "python/carnot/reporting/hardware_workload_obligations_8216.py",
            "scripts/experiment_template.py",
            "ops/exclusion_manifest.yaml",
            "openspec/change-proposals/research-roadmap-vNEXT.md",
        ]:
            h.freeze(h.ROOT / p, raw, dict(references=code_refs))
        ended = time.monotonic_ns()
        value.update(
            experiment_id=8231,
            task_id="exp8231-polarfire-state-boundary",
            milestone="2026.10.711",
            run_date=args.date,
            schema="carnot.polarfire_state_boundary.v711.v1",
            config=h.CONFIG,
            random_seed=h.CONFIG["seed"],
            duration_s=(ended - began) / 1e9,
            invocation_clocks=dict(
                started_wall_ns=wall,
                ended_wall_ns=time.time_ns(),
                started_monotonic_ns=began,
                ended_monotonic_ns=ended,
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
                    ("host_state_measurement", pre_end, measurement_end),
                    ("validation_and_health", measurement_end, ended),
                ]
            ],
            MODEL_SPECS=[],
            trained_head_specs=[
                dict(
                    arm=c["arm"],
                    scope=c["scope"],
                    state_sha256=canonical_hash(c["state"]),
                    current_fit=False,
                )
                for c in data["cases"]
            ],
            inference_substrate="verifier_ensemble_against_cached_candidates"
            if measured
            else "aggregation_from_upstream_artifacts",
            inference_substrate_class="no_model_load",
            model_invocation_counts=dict(ZERO_INVOCATION_COUNTS),
            current_model_calls=0,
            call_ledger=[],
            exposure_scope="frozen development kernel fixtures; no independent evaluation",
            preconditions_checked=data["checks"],
            precondition_receipts=pre_receipts,
            gate_check_summary=data["checks"],
            cited_upstream_artifacts=data["cited"],
            source_artifact_hashes=data["references"],
            code_config_hashes=code_refs,
            raw_shard_hashes=[reference(p) for p in sorted(raw.rglob("*")) if p.is_file()],
            replay_input_reference=reference(raw / "replay_inputs.json"),
            primitive_reference=reference(raw / "primitive_rows.json"),
            transfer_contract_path=str(raw / "transfer_contract.json"),
            transfer_contract_reference=reference(raw / "transfer_contract.json"),
            required_checks_passed=passed,
            validation_receipts=receipts,
            repository_health=health_receipts,
            repository_health_source=health_source,
            fixture_mode=bool(args.input),
            flagged_adversarial=False,
            terminal_validation_sidecar_path=str(side),
            methodology="Versioned canonical JSON preserves full frozen tree and causal pending/release/RNG state. Host scalar prediction replay and local fsync measure host feasibility only. No naturally trained state is substituted for blocked learning. Historical Qwen provenance is not a current invocation.",
            claim_scope="Portable state and historical Linux CPU dispatch boundary; no device or learning benefit",
            field_principles={},
        )
        value["field_principles"] = {
            key: "Bind invocation identity, bytes and costs; retain missing evidence without benefit credit."
            for key in value
        }
        value["field_principles"].update(
            serialized_state_rows="Actual host serialized sizes, ordered tree inventory, exact mixtures, clocks and local durable parity by frozen state scope.",
            polarfire_boundary_ready_score="Qualified host inventory remains ready despite blocked learning; no fabric, reachability or benefit claim.",
            polarfire_obligation="Historical authenticated Linux CPU dispatch is separate from all future device obligations.",
            transfer_contract_path="Version/hash equality and durable restart contract charges every host/network cost symbolically until measured.",
            trained_head_specs="Frozen state provenance only; current fitting and generator calls remain zero.",
            blocked_obligations="Preserve original upstream verdict, failed fields and validation receipts.",
            reproducibility_checksum="Canonical checksum binds every field except itself.",
            repository_health="One bounded global diagnostic is separate from owned validation; no global pass inferred.",
        )
        value = normalize_artifact_for_template_write(value)
        value["reproducibility_checksum"] = checksum(value)
        atomic_json(candidate, value)

        def validate(path: Path) -> Json:
            """Require actual normal auditor exits before bytes become reader-visible."""
            reports = execute(validators(path), raw / ("terminal-" + str(time.time_ns())))
            return dict(
                passed=all(r["passed"] and r["normal_exit"] for r in reports), receipts=reports
            )

        report = validate(candidate)
        if not report["passed"]:
            value.update(
                honest_verdict="complete_disqualified_terminal_validation",
                verdict_class="disqualified",
                polarfire_boundary_ready_score=0,
            )
            value["reproducibility_checksum"] = checksum(value)
            atomic_json(raw / "failed_terminal_candidate.json", value)
            atomic_json(side, report)
            return 1
        publication = publish_primary(output, value, validate)
        atomic_json(side, dict(publication=publication))
        h.progress("published", 1, 0)
        return 0
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(json.dumps(dict(passed=False, error=str(error))), flush=True)
        return 1
