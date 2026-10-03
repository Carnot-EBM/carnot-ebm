"""REQ-REPORT-7989: publish measured costs for independently qualified branches.

The prior service publisher supplies the validation protocol. Current timings
use frozen multivariate heads and never turn old model activity into new calls.
"""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
import json
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory
import time
from typing import Any

from carnot import experiment_7976_v691_service_cost as previous
from carnot import experiment_7972_v691_qwen_energy_calibration as prior
from carnot.reporting import service_cost_7989 as s
from carnot.reporting.current_work_receipt import atomic_json
from carnot.reporting.experiment_7303_validation_scope import CommandSpec, run_commands
from carnot.reporting.primary_publication import publish_primary, reader_receipt

Json = dict[str, Any]
ROOT = s.ROOT
NAME = "experiment_7989_v692_service_cost"
TASK = "exp7989-service-cost"
MODEL_SPECS: list[str] = []
OWNED = [p.replace("7976", "7989").replace("v691", "v692") for p in previous.OWNED]
TESTS = [p.replace("7976", "7989").replace("v691", "v692") for p in previous.TESTS]
INCLUDE = ",".join(str(ROOT / p) for p in OWNED)


def base(plan: Json) -> Json:
    """Retain the complete schema while binding identity to this invocation."""
    value = previous.base(plan)
    value.update(
        experiment_id=7989,
        task_id=TASK,
        milestone="2026.10.692",
        schema="carnot.exp7989.service_cost.v1",
        random_seed=69289,
        invocation_timestamp=datetime.now(UTC).isoformat(),
        raw_shard_hashes=[],
        replay_inputs=dict(requests=[], heads={}),
        service_summary=[],
        complete_service_cost=[],
        paired_cpu_comparisons=[],
        acquisition_setup=plan.get("acquisition_setup", {}),
        operation_inventory=[],
        compatible_fraction=0.0,
        transfer_bytes=0,
        methodology="One named request sweep and ten randomized paired CPU timing repeats. Source-group mean differences give paired 95% t intervals. Complete service adds only matching authenticated historical acquisition; setup is separately amortized over actual completed producer requests. Null accuracy does not invalidate cost.",
    )
    return value


def terminal_check(candidate: Path) -> Json:
    """Both terminal tools inspect the exact bytes accepted by cold replay."""
    s.replay(json.loads(candidate.read_text()))
    receipts = run_commands(
        ROOT,
        [
            CommandSpec(
                name,
                (str(ROOT / ".venv/bin/python"), "-u", script, flag, str(candidate)),
                "terminal_candidate",
                60,
            )
            for name, script, flag in [
                ("adversarial", "scripts/adversarial_verify.py", "--json"),
                ("strict_rows", "scripts/verdict_row_consistency_lint.py", "--strict"),
            ]
        ],
        log_dir=candidate.parent / "terminal_logs",
        heartbeat_s=10,
    )
    return dict(passed=all(r["passed"] for r in receipts), receipts=receipts)


def publish(output: Path, value: Json) -> None:
    """One checked primary remains visible to both production readers."""
    raw = output.parent / "raw" / output.stem
    value["terminal_validation_sidecar_path"] = str(raw / "terminal_validation.json")
    value["primary_resolution_receipt"] = dict(path=str(raw / "primary_resolution.json"))
    value["field_principles"] = {
        k: "Bind current identity, exact upstream bytes and measured work; cost readiness is not scientific benefit."
        for k in value
    }
    receipt = publish_primary(output, value, terminal_check)
    atomic_json(raw / "terminal_validation.json", receipt)
    readers = reader_receipt(
        TASK,
        output.parent,
        field="service_measurement_ready_score",
        expected=value["service_measurement_ready_score"],
    )
    if not readers["passed"]:
        raise ValueError("primary_resolution")
    atomic_json(raw / "primary_resolution.json", readers)


def freeze_commands(raw: Path, scratch: Path) -> Json:
    """Reuse the qualified command plan, changing only this task's owned scope."""
    manifest = previous.freeze_commands(raw, scratch)
    inputs, heads = s.fixture()
    atomic_json(scratch / "input.json", dict(requests=inputs, heads=heads))
    atomic_json(scratch / "request.json", inputs[0])
    atomic_json(scratch / "heads.json", heads)
    manifest = json.loads(
        json.dumps(manifest)
        .replace("7976", "7989")
        .replace("v691_service_cost", "v692_service_cost")
    )
    manifest["commands"] = [c for c in manifest["commands"] if c["name"] != "private_live_cli"]
    for command in manifest["commands"]:
        if command["name"] == "cold_replay_cli":
            command["argv"] = [
                "/usr/bin/env",
                "-u",
                "PYTHONPATH",
                "-C",
                str(scratch),
                *command["argv"],
            ]
        if command["name"] == "repository_health_full_suite":
            command.update(
                argv=[str(ROOT / ".venv/bin/pytest"), "tests/python", "-q"], deadline_s=300
            )
    libraries = manifest["transitive_consumers"] + [
        "python/carnot/verify/multivariate_energy_7982.py",
        "python/carnot/verify/evidence_features_7980.py",
        "python/carnot/experiment_7976_v691_service_cost.py",
        "python/carnot/reporting/service_cost_7976.py",
    ]
    manifest["code_config_hashes"] = [s.reference(ROOT / p) for p in OWNED + libraries + TESTS]
    atomic_json(raw / "validation_commands.json", manifest)
    return manifest


def main(argv: list[str] | None = None) -> int:
    """Bound each invocation and keep all mutable test output outside the checkout."""
    started, started_at = time.monotonic(), datetime.now(UTC).isoformat()
    print("[exp7989] phase=start model_loads=0 generation_calls=0", flush=True)
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", default="20261001")
    parser.add_argument("--output", type=Path, default=ROOT / "results" / f"{NAME}.json")
    parser.add_argument("--data-root", type=Path, default=ROOT)
    parser.add_argument("--fixture-e2e", type=Path)
    parser.add_argument("--cold-replay", type=Path)
    parser.add_argument("--terminal-recheck", type=Path)
    parser.add_argument("--request", type=Path)
    parser.add_argument("--heads", type=Path)
    parser.add_argument("--skip-validation", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.date != "20261001":
            raise ValueError("run_date")
        if args.cold_replay or args.terminal_recheck:
            path = args.cold_replay or args.terminal_recheck
            s.replay(json.loads(path.read_text()))
            if args.terminal_recheck and not terminal_check(path)["passed"]:
                raise ValueError("terminal_recheck")
            print("[exp7989] replay_passed", flush=True)
            return 0
        if args.request:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            print("[exp7989] before_benchmark private_request", flush=True)
            s.request(args.request, args.heads, args.output, "gibbs", "fsync")
            print("[exp7989] after_benchmark private_request", flush=True)
            return 0
        print("[exp7989] phase=authenticate", flush=True)
        plan = (
            json.loads(args.fixture_e2e.read_text())
            if args.fixture_e2e
            else s.authenticate(args.data_root)
        )
        value = base(plan)
        value["started_at"] = started_at
        output = args.output.absolute()
        raw = output.parent / "raw" / output.stem
        with TemporaryDirectory(prefix="carnot-7989-", dir="/tmp") as folder:
            scratch = Path(folder)
            manifest = freeze_commands(raw, scratch)
            value.update(
                validation_command_manifest_path=str(raw / "validation_commands.json"),
                code_config_hashes=manifest["code_config_hashes"],
                scratch_root_receipt=dict(
                    path=folder, outside_checkout=True, removed_after_exit=True
                ),
            )
            atomic_json(
                raw / "input_checkpoint.json", dict(requests=plan["requests"], heads=plan["heads"])
            )
            value["input_checkpoint"] = s.reference(raw / "input_checkpoint.json")
            value["raw_shard_hashes"] = [value["input_checkpoint"]]
            value["reproducibility_checksum"] = s.canonical_hash(
                dict(
                    code=manifest["code_config_hashes"],
                    input=value["input_checkpoint"],
                    date=args.date,
                    seed=69289,
                )
            )
            if plan["requests"]:
                print("[exp7989] phase=measure_cpu", flush=True)
                value.update(s.measure(plan["requests"], plan["heads"], scratch / "service"))
                value.pop("replay_inputs")
                value.update(
                    honest_verdict="complete_null_service_cost",
                    verdict_class="null",
                    service_measurement_ready_score=1,
                    trained_head_specs=[
                        dict(
                            arm=a,
                            head_seeds=[h.get("seed") for h in hs],
                            parameters_per_seed=len(hs[0]["parameters"]),
                            pretrained=False,
                        )
                        for a, hs in plan["heads"]["heads"].items()
                    ],
                )
                value["acceptance_gate_results"].update(readiness=True)
                if args.fixture_e2e:
                    value.update(
                        honest_verdict="complete_circular_positive_service_fixture",
                        verdict_class="circular_positive",
                    )
            else:
                value["inference_substrate"] = "aggregation_from_upstream_artifacts"
            value["cited_upstream_artifacts"] = [
                dict(
                    r,
                    imported_fields=[
                        "readiness",
                        "producer_identity",
                        "public_bytes",
                        "frozen_heads",
                        "matching_acquisition",
                    ],
                )
                for r in value["source_artifact_hashes"]
            ]
            value["historical_required_failures"] = [
                r
                for u in plan.get("upstream", {}).values()
                for r in u.get("historical_required_failures", [])
            ]
            if not args.fixture_e2e and not args.skip_validation:
                print("[exp7989] phase=owned_validation", flush=True)
                previous.apply_checks(value, prior.execute_commands(manifest, raw, scratch))
                coverage = json.loads((scratch / "coverage.json").read_text())
                value["coverage_statement_counts"] = {
                    k: v["summary"] for k, v in coverage["files"].items()
                }
                for path in scratch.glob(".coverage*"):
                    archive = raw / "coverage_data" / path.name
                    archive.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(path, archive)
                shutil.copy2(scratch / "coverage.json", raw / "coverage.json")
            value.update(
                duration_s=time.monotonic() - started, finished_at=datetime.now(UTC).isoformat()
            )
            value["phase_spans"] = [
                dict(
                    phase="owned_authentication_cpu_measurement_validation",
                    start_s=0.0,
                    end_s=value["duration_s"],
                )
            ]
            print("[exp7989] phase=terminal_publication", flush=True)
            publish(output, value)
            print(
                f"[exp7989] published path={output} elapsed_s={time.monotonic() - started:.3f}",
                flush=True,
            )
            return 0
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(f"[exp7989] rejected={type(error).__name__}:{error}", flush=True)
        return 1
