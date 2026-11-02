"""REQ-REPORT-8162: validate owned code before publishing durable software bounds."""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import json
import os
from pathlib import Path
import tempfile
import time
from typing import Any

from carnot import experiment_8148_v704_hardware_workload_boundary as qualified
from carnot.reporting import hardware_workload_8162 as h
from carnot.reporting import hardware_workload_inputs_8162 as inputs
from carnot.reporting.current_work_receipt import (
    ZERO_INVOCATION_COUNTS,
    atomic_json,
    canonical_hash,
)
from carnot.reporting.evidence_features_custody_7980 import checked, reference
from carnot.reporting.experiment_7303_validation_scope import CommandSpec, build_scoped_commands
from carnot.reporting.primary_publication import publish_primary

Json = dict[str, Any]
ROOT = qualified.ROOT
NAME = "experiment_8162_v705_hardware_workload_boundary"
SCRIPT = "scripts/experiments/" + NAME + ".py"
TEST = "tests/python/test_hardware_workload_8162.py"
OWNED = [
    "python/carnot/reporting/" + n + ".py"
    for n in [
        "hardware_workload_8162",
        "hardware_workload_inputs_8162",
        "hardware_boundary_execution_8162",
    ]
]
execute = qualified.execute


def commands(private: Path) -> list[CommandSpec]:
    """Freeze exact scopes and merge real child coverage before enforcing 100 percent."""
    cfg = private / "coverage.ini"
    cfg.write_text(
        "[run]\nparallel = True\ndata_file = "
        + str(private / ".coverage")
        + "\ninclude =\n"
        + "".join("    */" + p + "\n" for p in [*OWNED, SCRIPT])
    )
    specs = build_scoped_commands(
        ROOT,
        [TEST],
        OWNED,
        static_paths=[SCRIPT],
        basetemp=private,
        coverage_file=private / ".coverage",
    )
    result = []
    for spec in specs:
        argv = spec.argv
        if spec.name == "changed_module_coverage":
            argv = (argv[0], "run", "--rcfile=" + str(cfg), *argv[2:])
        if spec.name == "changed_module_coverage_report":
            result.append(
                CommandSpec(
                    "coverage_combine",
                    (
                        str(ROOT / ".venv/bin/coverage"),
                        "combine",
                        "--rcfile=" + str(cfg),
                        str(private),
                    ),
                    "owned_coverage",
                )
            )
        argv = tuple(a + ",*/" + SCRIPT if a.startswith("--include=") else a for a in argv)
        if spec.name == "changed_module_mypy":
            argv += ("--strict", "--follow-imports=silent")
        result.append(replace(spec, argv=argv))
    return result


def terminal(path: Path) -> list[CommandSpec]:
    """Reuse unchanged auditors, with a cold replay of this producer's candidate."""
    return [
        replace(
            s,
            argv=tuple(
                str(ROOT / SCRIPT) if a == str(ROOT / qualified.SCRIPT) else a for a in s.argv
            ),
        )
        for s in qualified.terminal_commands(path)
    ]


def replay(path: Path) -> Json:
    """Rehash primitive and code bytes, then independently check headline denominators."""
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
    for bound in value["amdahl_bounds"]:
        selected = [
            r
            for r in value["workload_rows"]
            if r["upstream"] == bound["upstream"] and r["arm"] == bound["arm"]
        ]
        if all(r["status"] == "completed" for r in selected):
            total = sum(r["numerator"] for r in selected)
            retained = sum(r["denominator"] for r in selected)
            if bound["outer_ceiling"] != total / retained:
                raise ValueError("independent_ceiling_drift")
    if value["required_checks_passed"] != all(
        r["passed"] and r["normal_exit"] for r in value["validation_receipts"]
    ):
        raise ValueError("validation_receipt_drift")
    return dict(passed=True, rows_checksum=canonical_hash(fresh["rows"]))


def main(argv: list[str] | None = None) -> int:
    """Freeze checks, preserve upstream failures, and atomically publish checked bytes.

    Private fixtures exercise transport without recursive pytest. Production
    runs every owned check and records broad repository health separately. No
    model or board is loaded, flashed, purchased or probed in either route.
    """
    os.environ["PYTHONUNBUFFERED"] = "1"
    began = time.monotonic()
    print("[exp8162] phase=start completed=0 pending=1 model_loads=0", flush=True)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", choices=["20261005"], default="20261005")
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--input", type=Path)
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
        private = Path(tempfile.mkdtemp(prefix="carnot8162-"))
        plan = [] if args.input else commands(private)
        health = CommandSpec(
            "repository_health_once",
            (str(ROOT / ".venv/bin/pytest"), "tests/python", "-q"),
            "repository_health_not_science_gate",
            1800,
        )
        candidate = private / (NAME + ".json")
        atomic_json(
            raw / "validation_commands.json",
            dict(
                commands=[asdict(s) for s in plan],
                terminal=[asdict(s) for s in terminal(candidate)],
                repository_health=asdict(health),
                config=h.CONFIG,
                frozen_before_measurement=True,
            ),
        )
        print("[exp8162] phase=preconditions_before completed=0 pending=1", flush=True)
        data = json.loads(args.input.read_bytes()) if args.input else inputs.load(args.root, raw)
        if args.input:
            data["fixture"] = True
        atomic_json(raw / "replay_inputs.json", data)
        print("[exp8162] phase=preconditions_after completed=1 pending=0", flush=True)
        receipts = execute(
            plan,
            private / "owned",
            dict(
                CARNOT_8162_COVERAGE_CONFIG=str(private / "coverage.ini"),
                COVERAGE_FILE=str(private / ".coverage.health"),
                JAX_PLATFORMS="cpu",
            ),
        )
        repository_health = (
            []
            if args.input
            else execute(
                [health], private / "health", dict(COVERAGE_FILE=str(private / ".coverage.health"))
            )
        )
        print("[exp8162] phase=reduction_before completed=0 pending=1", flush=True)
        reduction_start = time.monotonic()
        value = h.reduce(data)
        reduction_duration = time.monotonic() - reduction_start
        print("[exp8162] phase=reduction_after completed=1 pending=0", flush=True)
        passed = all(r["passed"] and r["normal_exit"] for r in receipts)
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
        atomic_json(raw / "validation_receipts.json", receipts)
        value.update(
            experiment_id=8162,
            task_id="exp8162-hardware-workload-boundary",
            run_date=args.date,
            schema="carnot.hardware_workload_boundary.v705.v1",
            random_seed=h.CONFIG["seed"],
            config=h.CONFIG,
            duration_s=time.monotonic() - began,
            phase_spans=[dict(phase="software_reduction", duration_s=reduction_duration)]
            + [dict(phase=r["name"], duration_s=r["duration_s"]) for r in receipts],
            required_checks_passed=passed,
            flagged_adversarial=False,
            validation_receipts=receipts,
            repository_health=repository_health,
            preconditions_checked=value["gate_check_summary"],
            claim_scope="Durable host batch and composed request software ceilings; historical board custody only",
            exposure_scope="private fixture"
            if data["fixture"]
            else "exposed cached natural development",
            inference_substrate="aggregation_from_upstream_artifacts",
            inference_substrate_class="no_model_load",
            MODEL_SPECS=[],
            model_invocation_counts=ZERO_INVOCATION_COUNTS,
            call_ledger=[],
            source_artifact_hashes=data["references"],
            code_config_hashes=[reference(ROOT / p) for p in [*OWNED, SCRIPT, TEST]],
            raw_shard_hashes=[
                reference(raw / p)
                for p in [
                    "replay_inputs.json",
                    "primitive_rows.json",
                    "validation_commands.json",
                    "validation_receipts.json",
                ]
            ],
            replay_input_reference=reference(raw / "replay_inputs.json"),
            terminal_validation_sidecar_path=str(raw / "terminal_validation.json"),
            field_principles=dict(
                readiness="Normal owned checks; custody is not fabric benefit",
                rows="Original units reconstruct every comparison",
                substrate="Imported provenance never increments current calls",
                costs="Unknown components remain unknown",
            ),
        )
        value["reproducibility_checksum"] = canonical_hash(value["rows"])
        for receipt in receipts:
            if receipt.get("log_path"):
                value["raw_shard_hashes"].append(
                    dict(path=receipt["log_path"], sha256=receipt["log_sha256"])
                )
        atomic_json(candidate, value)
        atomic_json(raw / "independent_reduction.json", replay(candidate))

        def validate(path: Path) -> Json:
            checks = execute(terminal(path), private / ("terminal_" + str(time.time_ns())))
            return dict(
                passed=all(r["passed"] and r["normal_exit"] for r in checks), receipts=checks
            )

        terminal_report = validate(candidate)
        if not terminal_report["passed"]:
            value.update(
                required_checks_passed=False,
                honest_verdict="complete_disqualified_terminal_validation",
                verdict_class="disqualified",
                hardware_boundary_ready_score=0,
                measured_workload_bound_ready_score=0,
            )
            atomic_json(raw / "failed_terminal_candidate.json", value)
            atomic_json(raw / "terminal_validation.json", terminal_report)
            return 1
        value["validation_receipts"] += terminal_report["receipts"]
        value["raw_shard_hashes"].append(reference(raw / "independent_reduction.json"))
        for receipt in terminal_report["receipts"]:
            if receipt.get("log_path"):
                value["raw_shard_hashes"].append(
                    dict(path=receipt["log_path"], sha256=receipt["log_sha256"])
                )
        value["duration_s"] = time.monotonic() - began
        if output.exists():
            (raw / "preserved_primary.json").write_bytes(output.read_bytes())
        print("[exp8162] phase=publication_before completed=0 pending=1", flush=True)
        publication = publish_primary(output, value, validate)
        atomic_json(
            raw / "terminal_validation.json",
            dict(publication=publication, required_checks_passed=passed, normal_process_exit=True),
        )
        print(
            f"[exp8162] phase=complete completed={value['completed_count']} pending=0", flush=True
        )
        return 0 if passed else 1
    except (ValueError, OSError, KeyError) as exc:
        print(f"[exp8162] failed={type(exc).__name__}:{exc}", flush=True)
        return 1
