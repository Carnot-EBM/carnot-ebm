"""REQ-REPORT-8148: publish authenticated arithmetic bounds after owned validation."""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import json
import os
from pathlib import Path
import tempfile
import time
from typing import Any

from carnot import experiment_8134_v703_hardware_service_boundary as previous
from carnot.reporting import hardware_workload_8148 as h
from carnot.reporting import hardware_workload_inputs_8148 as inputs
from carnot.reporting.current_work_receipt import (
    ZERO_INVOCATION_COUNTS,
    atomic_json,
    canonical_hash,
)
from carnot.reporting.evidence_features_custody_7980 import checked, reference
from carnot.reporting.experiment_7303_validation_scope import CommandSpec, run_commands
from carnot.reporting.primary_publication import publish_primary

Json = dict[str, Any]
ROOT = previous.ROOT
NAME = "experiment_8148_v704_hardware_workload_boundary"
SCRIPT = "scripts/experiments/" + NAME + ".py"
TEST = "tests/python/test_hardware_workload_8148.py"
OWNED = [
    "python/carnot/" + NAME + ".py",
    "python/carnot/reporting/hardware_workload_8148.py",
    "python/carnot/reporting/hardware_workload_inputs_8148.py",
    SCRIPT,
]


def commands(scratch: Path) -> list[CommandSpec]:
    """Reuse the qualified command plan with coverage limited to these new files."""
    specs = previous.commands(scratch)
    mapping = dict(zip([*previous.OWNED, previous.TEST], [*OWNED, TEST], strict=True))
    cfg = scratch / "coverage.ini"
    content = cfg.read_text()
    for old, new in mapping.items():
        content = content.replace(old, new)
    cfg.write_text(content)
    return [replace(s, argv=tuple(mapping.get(a, a) for a in s.argv)) for s in specs]


def replay(path: Path) -> Json:
    """Rehash saved operands and recompute every scientific reduction in a cold process."""
    value = json.loads(path.read_bytes())
    for ref in (
        value["source_artifact_hashes"] + value["raw_shard_hashes"] + value["code_config_hashes"]
    ):
        checked(ref)
    fresh = h.reduce(json.loads(checked(value["replay_input_reference"]).read_bytes()))
    for key, expected in fresh.items():
        if value["verdict_class"] == "disqualified" and key in {
            "honest_verdict",
            "verdict_class",
            "hardware_boundary_ready_score",
            "measured_workload_bound_ready_score",
        }:
            continue
        if value[key] != expected:
            raise ValueError("reduction_drift:" + key)
    if value["required_checks_passed"] != all(
        r["passed"] and r["normal_exit"] for r in value["validation_receipts"]
    ):
        raise ValueError("validation_receipt_drift")
    return dict(passed=True, rows_checksum=canonical_hash(fresh["rows"]))


def terminal_commands(path: Path) -> list[CommandSpec]:
    """Keep unchanged terminal validators bound to exactly one candidate."""
    specs = previous.terminal_commands(path)
    return [
        replace(
            s,
            argv=tuple(
                str(ROOT / SCRIPT) if a == str(ROOT / previous.SCRIPT) else a for a in s.argv
            ),
        )
        for s in specs
    ]


def execute(specs: list[CommandSpec], private: Path, env: Json | None = None) -> list[Json]:
    """Keep validation outputs private and record real normal exits and log bytes."""
    receipts = run_commands(ROOT, specs, log_dir=private, heartbeat_s=30, extra_env=env)
    for receipt in receipts:
        receipt.update(
            expected_exit=0,
            actual_exit=receipt["exit_code"],
            normal_exit=receipt["exit_code"] >= 0 and not receipt.get("timed_out", False),
        )
    return receipts


def main(argv: list[str] | None = None) -> int:
    """Freeze the owned plan before measurement and publish only checked terminal bytes."""
    os.environ["PYTHONUNBUFFERED"] = "1"
    began = time.monotonic()
    spans: list[Json] = []

    def phase(name: str) -> None:
        elapsed = time.monotonic() - began
        if spans:
            spans[-1]["end_s"] = elapsed
        spans.append(dict(phase=name, start_s=elapsed))
        print(f"[exp8148] phase={name} elapsed_s={elapsed:.3f} model_loads=0", flush=True)

    phase("preconditions")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", choices=["20261005"], default="20261005")
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--input", type=Path)
    parser.add_argument("--health-receipt", type=Path)
    parser.add_argument("--cold-replay", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.cold_replay:
            print(json.dumps(replay(args.cold_replay)), flush=True)
            return 0
        output = (args.output or args.root / "results" / (NAME + ".json")).absolute()
        if args.input and output.is_relative_to(ROOT / "results"):
            raise ValueError("private_fixture_requires_private_output")
        raw = output.parent / "raw" / NAME / "invocations" / str(time.time_ns())
        raw.mkdir(parents=True, exist_ok=True)
        prior_candidate = output.parent / "raw" / NAME / "terminal_candidate.json"
        if prior_candidate.exists():
            (raw / "preserved_terminal_candidate.json").write_bytes(prior_candidate.read_bytes())
        private = Path(tempfile.mkdtemp(prefix="carnot8148-", dir="/tmp"))
        specs = [] if args.input else commands(private)
        previous_health = []
        if args.health_receipt:
            previous_health = json.loads(args.health_receipt.read_bytes())["repository_health"]
            specs = [s for s in specs if s.scope != "repository_health"]
        atomic_json(
            raw / "validation_manifest.json",
            dict(
                commands=[asdict(s) for s in specs],
                terminal_commands=[asdict(s) for s in terminal_commands(output)],
                config=h.CONFIG,
                frozen_before_measurement=True,
            ),
        )
        phase("before_owned_validation_subprocesses")
        receipts = execute(
            specs,
            private / "logs",
            dict(
                CARNOT_8148_COVERAGE_CONFIG=str(private / "coverage.ini"),
                CARNOT_8148_CLI_RECEIPTS=str(private / "private_cli_receipts.json"),
                COVERAGE_FILE=str(private / ".coverage.health"),
                JAX_PLATFORMS="cpu",
            ),
        )
        owned = [r for r in receipts if r["scope"] != "repository_health"]
        health = previous_health + [r for r in receipts if r["scope"] == "repository_health"]
        passed = all(r["passed"] and r["normal_exit"] for r in owned)
        phase("after_owned_validation_authenticate_inputs")
        data = json.loads(args.input.read_bytes()) if args.input else inputs.load(args.root, raw)
        if args.input:
            data["fixture"] = True
        if args.health_receipt:
            inputs.custody.seal(
                args.health_receipt,
                reference(args.health_receipt)["sha256"],
                raw,
                data,
                "previous_repository_health",
            )
            for check in data["checks"]:
                check.setdefault("artifact_field", check["field"])
        atomic_json(raw / "replay_inputs.json", data)
        phase("before_cpu_reduction")
        value = h.reduce(data)
        phase("after_cpu_reduction")
        if not passed:
            value.update(
                honest_verdict="complete_disqualified_owned_checks",
                verdict_class="disqualified",
                hardware_boundary_ready_score=0,
                measured_workload_bound_ready_score=0,
            )
        atomic_json(
            raw / "primitive_rows.json", dict(rows=value["rows"], board_rows=value["board_rows"])
        )
        atomic_json(raw / "validation_receipts.json", dict(owned=owned, repository_health=health))
        for name in ["coverage.json", "private_cli_receipts.json"]:
            atomic_json(
                raw / name,
                json.loads((private / name).read_bytes()) if (private / name).exists() else {},
            )
        # Copy transcripts as evidence while their original execution output stays private.
        atomic_json(
            raw / "validation_transcripts.json",
            dict(
                logs=[
                    dict(name=r["name"], transcript=(ROOT / r["log_path"]).read_text())
                    for r in receipts
                ]
            ),
        )
        value.update(
            experiment_id=8148,
            task_id="exp8148-hardware-workload-boundary",
            run_date=args.date,
            schema="carnot.hardware_workload_boundary.v704.v1",
            random_seed=h.CONFIG["seed"],
            inference_substrate="aggregation_from_upstream_artifacts",
            inference_substrate_class="no_model_load",
            MODEL_SPECS=[],
            model_invocation_counts=ZERO_INVOCATION_COUNTS,
            call_ledger=[],
            duration_s=time.monotonic() - began,
            phase_spans=spans,
            claim_scope="Natural exposed host workload ceilings and historical board custody; no measured device acceleration.",
            exposure_scope="exposed natural development inputs and historical receipts",
            methodology_note="Only mixed scoring envelope outer ceilings are known. Memory and crossings inside the envelope remain unknown; exact arithmetic fractions are not claimed. Precision uses measured natural inputs with float64 fallback, no LLM load or training.",
            required_checks_passed=passed,
            flagged_adversarial=False,
            validation_receipts=owned,
            repository_health=health,
            source_artifact_hashes=data["references"],
            code_config_hashes=[reference(ROOT / p) for p in [*OWNED, TEST]],
            replay_input_reference=reference(raw / "replay_inputs.json"),
            raw_shard_hashes=[reference(p) for p in raw.rglob("*") if p.is_file()],
            terminal_validation_sidecar_path=str(raw / "terminal_validation.json"),
            config=h.CONFIG,
        )
        value["reproducibility_checksum"] = canonical_hash(
            dict(
                input=value["replay_input_reference"],
                code=value["code_config_hashes"],
                config=h.CONFIG,
            )
        )
        value["field_principles"] = {
            k: f"{k} binds bounded current evidence; historical execution stays historical."
            for k in value
        }
        value["field_principles"].update(
            field_principles="State the evidential limit of each field.",
            amdahl_bounds="S_max=1/(1-f); mixed scoring envelope is only an optimistic outer bound. Missing arithmetic share stays unknown.",
            natural_workload_rows="Keep dependent repetitions and missing costs; numeric operand bytes are not measured bus traffic.",
            gate_check_summary="Every external block retains its exact requested operand.",
            board_rows="Original dates and hashes remain independent of service qualification.",
        )
        phase("before_terminal_publication")

        def terminal(path: Path) -> Json:
            checks = execute(terminal_commands(path), private / "terminal" / str(time.time_ns()))
            return dict(
                passed=all(r["passed"] and r["normal_exit"] for r in checks), receipts=checks
            )

        if output.exists():
            atomic_json(raw / "preserved_primary.json", json.loads(output.read_bytes()))
        publication = publish_primary(output, value, replay if args.input else terminal)
        atomic_json(
            raw / "terminal_validation.json",
            dict(publication=publication, required_checks_passed=passed, normal_process_exit=True),
        )
        phase("complete")
        return 0
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(f"[exp8148] terminal_error={error}", flush=True)
        return 1
