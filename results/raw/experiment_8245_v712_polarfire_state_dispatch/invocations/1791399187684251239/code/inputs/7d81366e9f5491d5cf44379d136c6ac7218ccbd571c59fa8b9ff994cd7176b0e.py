"""REQ-REPORT-8245: owned validation precedes any current PolarFire contact.

Qualified process supervision keeps full streams, normal exits and bounded
process groups. Historical disqualification remains a source fact, not repaired.
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

from carnot.reporting import polarfire_state_dispatch_8245 as d
from carnot.reporting import polarfire_packet_evaluator_8245 as e
from carnot.reporting import recorder_execution_8213 as qualified
from carnot.verify import request_recorder_8213 as supervisor_config
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
TEST = "tests/python/test_polarfire_state_dispatch_8245.py"
MODULES = [
    "python/carnot/reporting/polarfire_state_dispatch_8245.py",
    "python/carnot/reporting/polarfire_dispatch_execution_8245.py",
    d.EVALUATOR,
]
OWNED = [*MODULES, d.CLI]
execute = qualified.execute


def checksum(value: Json) -> str:
    """Bind all report fields except the checksum itself to detect summary changes."""
    return canonical_hash({k: v for k, v in value.items() if k != "reproducibility_checksum"})


def commands(private: Path) -> list[CommandSpec]:
    """Freeze valid file arguments together, including measured real CLI statements."""
    private.mkdir(parents=True, exist_ok=True)
    with (
        patch.object(qualified, "MODULES", MODULES),
        patch.object(qualified, "TEST", TEST),
        patch.object(supervisor_config, "CLI", d.CLI),
    ):
        plan = qualified.validation_plan(private)
    config = private / "coverage.ini"
    config.write_text(config.read_text() + "[report]\nexclude_lines =\n")
    result = []
    for spec in plan:
        argv = spec.argv
        if spec.name == "changed_module_mypy":
            argv = (
                str(d.ROOT / ".venv/bin/mypy"),
                "--config-file=/dev/null",
                "--strict",
                "--follow-imports=silent",
                *OWNED,
            )
        result.append(CommandSpec(spec.name, argv, spec.scope, 180))
    return result


def validators(path: Path) -> list[CommandSpec]:
    """The unchanged auditors and a fresh replay must all exit normally."""
    with patch.object(supervisor_config, "CLI", d.CLI):
        return qualified.validators(path)


def historical(raw: Path) -> Json:
    """Reproduce the recorded defects without replacing the disqualified primary."""
    path = d.ROOT / "results/experiment_8231_v711_polarfire_state_boundary.json"
    if (
        e.digest(path.read_bytes())
        != "sha256:8d211e8c1b1d2ef8d7a7cbe9f4df0bdfd5530f0bd6d9e651d589fe5fc9e9e359"
    ):
        raise ValueError("historical_primary_hash")
    value = json.loads(path.read_bytes())
    archived = {
        str(Path(r["original_path"]).relative_to(d.ROOT)): r
        for r in value["code_config_hashes"]
        if r["original_path"].endswith(".py")
    }
    snapshots = [d.copy_bytes(checked(ref), raw) for ref in archived.values()]
    specs = [
        CommandSpec(
            "historical_" + r["name"],
            tuple(
                archived[a]["path"] if r["name"] == "ruff_format" and a in archived else a
                for a in r["command_argv"]
            ),
            "historical_failure_reproduction",
            30,
        )
        for r in value["validation_receipts"]
        if r["name"] in {"ruff_format", "changed_module_mypy"}
    ]
    receipts = execute(specs, raw / "historical")
    for receipt, expected_exit in zip(receipts, [1, 2], strict=True):
        receipt.update(
            expected_exit=expected_exit,
            passed=receipt["actual_exit"] == expected_exit and receipt["normal_exit"],
        )
    return dict(
        primary=d.copy_bytes(path, raw),
        verdict=value["honest_verdict"],
        receipts=receipts,
        archived_sources=snapshots,
        original_commands=[
            r["command_argv"]
            for r in value["validation_receipts"]
            if r["name"] in {"ruff_format", "changed_module_mypy"}
        ],
        reproduced=all(r["passed"] for r in receipts),
    )


def replay(path: Path) -> Json:
    """Cold disk reads recompute primitives, rows and blocked outcomes independently."""
    value = json.loads(path.read_bytes())
    for ref in (
        value["source_artifact_hashes"] + value["code_config_hashes"] + value["raw_shard_hashes"]
    ):
        checked(ref)
    data = json.loads(checked(value["input_reference"]).read_bytes())
    host = json.loads(checked(value["host_reference"]).read_bytes())["output"]
    device = json.loads(checked(value["board_reference"]).read_bytes())
    if data["state"] is not None:
        packet = json.loads(Path(value["packet_path"]).read_bytes())
        if e.digest(Path(value["packet_path"]).read_bytes()) != value["packet_sha256"]:
            raise ValueError("packet_hash")
        if packet["evaluator_sha256"] != e.digest((d.ROOT / d.EVALUATOR).read_bytes()):
            raise ValueError("evaluator_hash")
        if e.evaluate(packet) != d.expected(data):
            raise ValueError("packet_reference_drift")
    for key, wanted in d.reduce(data, host, device, value["required_checks_passed"]).items():
        if value[key] != wanted:
            raise ValueError("reduction_drift:" + key)
    if device["executed"]:
        receipt = next(r for r in device["receipts"] if r["name"] == "board_evaluate")
        if (
            device["output"] is not None
            and json.loads(
                checked(
                    dict(path=receipt["stdout_path"], sha256=receipt["stdout_sha256"])
                ).read_bytes()
            )
            != device["output"]
        ):
            raise ValueError("board_transcript_drift")
    if value["reproducibility_checksum"] != checksum(value):
        raise ValueError("checksum_drift")
    return dict(
        passed=True,
        host_parity=value["host_parity"],
        device_execution_count=value["current_device_execution_count"],
    )


def main(argv: list[str] | None = None) -> int:
    """Publish checked terminal bytes, leaving unavailable external operands blocked."""
    d.progress("start_no_model_load", 0, 1)
    began, wall = time.monotonic_ns(), time.time_ns()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", choices=["20261007"], default="20261007")
    parser.add_argument("--root", type=Path, default=d.ROOT)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--input", type=Path)
    parser.add_argument("--cold-replay", type=Path)
    parser.add_argument("--evaluate", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.evaluate:
            return e.main([str(args.evaluate)])
        if args.cold_replay:
            print(json.dumps(replay(args.cold_replay)), flush=True)
            return 0
        output = (args.output or args.root / "results" / (d.NAME + ".json")).absolute()
        if args.input and output.resolve().is_relative_to((d.ROOT / "results").resolve()):
            raise ValueError("private_fixture_requires_private_output")
        private = Path(tempfile.mkdtemp(prefix="carnot8245-"))
        private.chmod(0o700)
        raw = output.parent / "raw" / d.NAME / "invocations" / str(time.time_ns())
        raw.mkdir(parents=True, exist_ok=True)
        candidate = private / (d.NAME + ".json")
        plan = commands(private / "checks")
        health = CommandSpec(
            "repository_health_once",
            (str(d.ROOT / ".venv/bin/pytest"), "tests/python", "-q"),
            "repository_health_diagnostic",
            180,
        )
        preflight = [
            CommandSpec(
                "resources_and_scratch",
                (
                    sys.executable,
                    "-c",
                    'import pathlib,sys,shutil,pytest,coverage,ruff,mypy; p=pathlib.Path(sys.argv[1]); p.write_bytes(b"writable"); assert p.read_bytes()==b"writable"; assert sys.version_info >= (3,10); print(shutil.disk_usage(p.parent))',
                    str(private / "probe"),
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
                retries=0,
            ),
        )
        d.progress("preconditions_before")
        pre_receipts = execute(preflight, raw / "preflight")
        history: Json = historical(raw) if not args.input else dict(receipts=[], reproduced=True)
        data = json.loads(args.input.read_bytes()) if args.input else d.load(args.root, raw)
        data["fixture"] = bool(args.input)
        if not args.input:
            data["references"] += [history["primary"], *history["archived_sources"]]
        if args.input:
            data["references"].append(d.copy_bytes(args.input, raw))
        if not all(r["passed"] for r in pre_receipts):
            data["checks"].append(
                d.operand("resources_and_scratch", private / "probe", True, False)
            )
            data.update(state=None, queries=[])
        pre_end = time.monotonic_ns()
        atomic_json(raw / "input.json", data)
        d.progress("preconditions_after", 1, 0)
        host: Json | None = None
        host_receipts: list[Json] = []
        packet_path = None
        if data["state"] is not None:
            d.progress("host_benchmark_before", 0, 1)
            d.packet(data, raw)
            packet_path = str(raw / "packet.json")
            host_receipts = execute(
                [
                    CommandSpec(
                        "host_evaluate",
                        (sys.executable, "-u", str(d.ROOT / d.EVALUATOR), packet_path),
                        "host_parity",
                        60,
                    )
                ],
                raw / "host",
            )
            if host_receipts[0]["passed"]:
                host = json.loads(Path(host_receipts[0]["stdout_path"]).read_bytes())
            d.progress("host_benchmark_after", 1, 0)
        atomic_json(raw / "host.json", dict(output=host))
        measured_end = time.monotonic_ns()
        d.progress("owned_validation_before", 0, len(plan))
        receipts = execute(plan, raw / "validation") if not args.input else []
        owned = all(r["passed"] and r["normal_exit"] for r in receipts + host_receipts) and bool(
            history["reproduced"]
        )
        d.progress("owned_validation_after", len(receipts), 0)
        device: Json = dict(ready=False, executed=False, output=None, block=None, receipts=[])
        d.progress("board_phase_before", 0, 1)
        if owned and data["state"] is not None and host == d.expected(data) and not args.input:
            device = d.board(raw)
        atomic_json(raw / "board_transcript.json", device)
        board_end = time.monotonic_ns()
        d.progress("board_phase_after", 1, 0)
        health_receipts = execute([health], raw / "health") if not args.input else []
        side = raw / "terminal_validation.json"
        code = [
            d.copy_bytes(d.ROOT / p, raw / "code")
            for p in [
                *OWNED,
                TEST,
                "python/carnot/reporting/polarfire_state_boundary_8231.py",
                "python/carnot/reporting/primary_publication.py",
                "scripts/experiment_template.py",
                "ops/exclusion_manifest.yaml",
                "openspec/change-proposals/research-roadmap-vNEXT.md",
            ]
        ]
        ended = time.monotonic_ns()
        value = d.reduce(data, host, device, owned)
        value.update(
            experiment_id=8245,
            task_id="exp8245-polarfire-state-dispatch",
            milestone="2026.10.712",
            run_date=args.date,
            schema="carnot.polarfire_state_dispatch.v1",
            inference_substrate="hardware_smoke"
            if device["executed"]
            else "aggregation_from_upstream_artifacts",
            inference_substrate_class="no_model_load",
            MODEL_SPECS=[],
            model_invocation_counts=dict(ZERO_INVOCATION_COUNTS),
            current_model_calls=0,
            trained_head_specs=[dict(origin=data.get("state_origin"), current_fit=False)]
            if data["state"] is not None
            else [],
            state_origin=data.get("state_origin"),
            packet_path=packet_path,
            packet_sha256=e.digest(Path(packet_path).read_bytes()) if packet_path else None,
            board_transcript_path=str(raw / "board_transcript.json"),
            preconditions_checked=data["checks"],
            precondition_receipts=pre_receipts,
            gate_check_summary=data["checks"] + ([device["block"]] if device["block"] else []),
            required_checks_passed=owned,
            validation_receipts=receipts + host_receipts,
            historical_failure_reproduction=history,
            repository_health=health_receipts,
            flagged_adversarial=False,
            terminal_validation_sidecar_path=str(side),
            exposure_scope="reused exposed development probabilities; no independent evaluation",
            duration_s=(ended - began) / 1e9,
            random_seed=7128245,
            invocation_clocks=dict(
                started_wall_ns=wall,
                ended_wall_ns=time.time_ns(),
                started_monotonic_ns=began,
                ended_monotonic_ns=ended,
            ),
            phase_spans=[
                dict(
                    phase=n, started_monotonic_ns=s, ended_monotonic_ns=t, duration_s=(t - s) / 1e9
                )
                for n, s, t in [
                    ("preconditions", began, pre_end),
                    ("host_parity", pre_end, measured_end),
                    ("owned_validation_and_board", measured_end, board_end),
                    ("repository_health", board_end, ended),
                ]
            ],
            source_artifact_hashes=data["references"],
            code_config_hashes=code,
            raw_shard_hashes=[reference(p) for p in sorted(raw.rglob("*")) if p.is_file()],
            input_reference=reference(raw / "input.json"),
            host_reference=reference(raw / "host.json"),
            board_reference=reference(raw / "board_transcript.json"),
            cited_upstream_artifacts=data["cited"],
            fixture_mode=bool(args.input),
            methodology="Full selected state and every query are hashed. The original host evaluator defines parity. Current board execution is one bounded Linux CPU dispatch. No generator loads, state fitting, FPGA acceleration, speed or learning gain is inferred. Historical Qwen is provenance only.",
            claim_scope="execution mechanics and exact hashes only; independent benefit remains zero",
        )
        value["field_principles"] = {
            k: "Bind current invocation and evidence bytes; missing evidence supplies no benefit."
            for k in value
        }
        value["field_principles"].update(
            polarfire_validation_ready_score="All owned validation and host parity pass; board availability is separate.",
            polarfire_workload_validated="True only after a current board result matches the host state and output hashes.",
            state_origin="Qualified first-seed local_only state, or explicitly static mechanics fallback; no disqualified state.",
            repository_health="One bounded global diagnostic retains actual failures; no global pass is inferred.",
            rows="Each selected original query retains host and board conditions, including missing probabilities and unavailable board results.",
            verifier_is_oracle="Hash parity is reference-defined mechanics; successful device parity is circular_positive.",
            inference_substrate="Actual board dispatch uses hardware_smoke with no_model_load; absent dispatch uses artifact aggregation.",
            historical_failure_reproduction="Original format and argv exits are reproduced without promoting historical Exp8231.",
            reproducibility_checksum="Canonical hash binds every field except this checksum.",
        )
        value = normalize_artifact_for_template_write(value)
        value["reproducibility_checksum"] = checksum(value)
        atomic_json(candidate, value)

        def validate(path: Path) -> Json:
            reports = execute(validators(path), raw / ("terminal-" + str(time.time_ns())))
            return dict(
                passed=all(r["passed"] and r["normal_exit"] for r in reports), receipts=reports
            )

        report = validate(candidate)
        if not report["passed"]:
            atomic_json(raw / "failed_terminal_candidate.json", value)
            atomic_json(side, report)
            return 1
        publication = publish_primary(output, value, validate)
        atomic_json(
            side, dict(publication=publication, normal_process_exit=True, owned_checks_passed=owned)
        )
        d.progress("published", 1, 0)
        return 0
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(json.dumps(dict(passed=False, error=str(error))), flush=True)
        return 1
