"""REQ-REPORT-8134: validate CPU software analysis before primary publication."""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import json
import os
from pathlib import Path
import tempfile
import time
from typing import Any

from carnot import experiment_8121_v702_hardware_batch_boundary as previous
from carnot.reporting import hardware_service_8134 as h
from carnot.reporting import hardware_service_inputs_8134 as inputs
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
NAME = "experiment_8134_v703_hardware_service_boundary"
SCRIPT = "scripts/experiments/" + NAME + ".py"
TEST = "tests/python/test_hardware_service_8134.py"
OWNED = [
    "python/carnot/" + NAME + ".py",
    "python/carnot/reporting/hardware_service_8134.py",
    "python/carnot/reporting/hardware_service_inputs_8134.py",
    SCRIPT,
]


def commands(scratch: Path) -> list[CommandSpec]:
    """Reuse the qualified validation plan with coverage limited to these files."""
    specs = previous.commands(scratch)
    mapping = dict(zip([*previous.OWNED, previous.TEST], [*OWNED, TEST], strict=True))
    cfg = scratch / "coverage.ini"
    content = cfg.read_text()
    for old, new in mapping.items():
        content = content.replace(old, new)
    cfg.write_text(content)
    return [replace(s, argv=tuple(mapping.get(a, a) for a in s.argv)) for s in specs]


def replay(path: Path) -> Json:
    """Rehash all saved operands and recompute reductions in a cold process."""
    value = json.loads(path.read_bytes())
    for ref in (
        value["source_artifact_hashes"] + value["code_config_hashes"] + value["raw_shard_hashes"]
    ):
        checked(ref)
    data = json.loads(checked(value["replay_input_reference"]).read_bytes())
    fresh = h.reduce(data)
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
    saved = json.loads(checked(value["validation_reference"]).read_bytes())
    if saved["owned"] != value["validation_receipts"] or value["required_checks_passed"] != all(
        r["passed"] for r in saved["owned"]
    ):
        raise ValueError("validation_receipt_drift")
    return dict(passed=True, rows_checksum=canonical_hash(fresh["rows"]))


def terminal_commands(path: Path) -> list[CommandSpec]:
    """Keep unmodified terminal validators bound to identical candidate bytes."""
    py = str(ROOT / ".venv/bin/python")
    return [
        CommandSpec(
            "cold_replay",
            (py, "-u", str(ROOT / SCRIPT), "--cold-replay", str(path)),
            "terminal",
            60,
        ),
        CommandSpec(
            "adversarial",
            (py, str(ROOT / "scripts/adversarial_verify.py"), "--json", str(path)),
            "terminal",
            60,
        ),
        CommandSpec(
            "strict_rows",
            (py, str(ROOT / "scripts/verdict_row_consistency_lint.py"), "--strict", str(path)),
            "terminal",
            60,
        ),
    ]


def terminal(path: Path) -> Json:
    """Normal exits are required; a timeout cannot grant publication credit."""
    receipts = run_commands(
        ROOT,
        terminal_commands(path),
        log_dir=path.parent / "terminal_logs" / str(time.time_ns()),
        heartbeat_s=10,
    )
    for receipt in receipts:
        receipt.update(
            expected_exit=0,
            normal_exit=receipt["exit_code"] >= 0 and not receipt.get("timed_out", False),
        )
    return dict(passed=all(r["passed"] and r["normal_exit"] for r in receipts), receipts=receipts)


def main(argv: list[str] | None = None) -> int:
    """Freeze validation before evaluation and publish only a checked candidate."""
    os.environ["PYTHONUNBUFFERED"] = "1"
    began = time.monotonic()
    spans: list[Json] = []

    def phase(name: str) -> None:
        elapsed = time.monotonic() - began
        if spans:
            spans[-1]["end_s"] = elapsed
        spans.append(dict(phase=name, start_s=elapsed))
        print(
            f"[exp8134] phase={name} elapsed_s={elapsed:.3f} model_loads=0 device_calls=0",
            flush=True,
        )

    phase("preconditions")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", choices=["20261004"], default="20261004")
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--input", type=Path)
    parser.add_argument("--worker-input", type=Path)
    parser.add_argument("--cold-replay", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.cold_replay:
            print(json.dumps(replay(args.cold_replay)), flush=True)
            return 0
        output = (args.output or args.root / "results" / (NAME + ".json")).absolute()
        if args.worker_input:
            phase("before_cpu_evaluation")
            atomic_json(output, h.reduce(json.loads(args.worker_input.read_bytes())))
            phase("after_cpu_evaluation")
            return 0
        raw = output.parent / "raw" / NAME / "invocations" / str(time.time_ns())
        raw.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="carnot8134-", dir="/tmp") as temporary:
            scratch = Path(temporary)
            specs = [] if args.input else commands(scratch)
            atomic_json(
                raw / "validation_manifest.json",
                dict(
                    commands=[asdict(s) for s in specs],
                    terminal_commands=[
                        asdict(s)
                        for s in terminal_commands(
                            output.parent / "raw" / NAME / "terminal_candidate.json"
                        )
                    ],
                    config=h.CONFIG,
                ),
            )
            phase("authenticate_inputs")
            data = (
                json.loads(args.input.read_bytes()) if args.input else inputs.load(args.root, raw)
            )
            if args.input:
                data["fixture"] = True
            atomic_json(raw / "replay_inputs.json", data)
            phase("before_evaluation_subprocess")
            child = CommandSpec(
                "normal_evaluation_exit",
                (
                    str(ROOT / ".venv/bin/python"),
                    "-u",
                    str(ROOT / SCRIPT),
                    "--worker-input",
                    str(raw / "replay_inputs.json"),
                    "--output",
                    str(raw / "worker.json"),
                ),
                "measurement",
                60,
            )
            measurement = run_commands(
                ROOT, [child], log_dir=raw / "measurement_logs", heartbeat_s=10
            )
            if not measurement[0]["passed"]:
                raise ValueError("evaluation_child_failed")
            phase("after_evaluation_independent_reduction")
            value = json.loads((raw / "worker.json").read_bytes())
            fresh = h.reduce(data)
            if value != fresh:
                raise ValueError("independent_reduction_drift")
            atomic_json(raw / "independent_reduction.json", fresh)
            atomic_json(
                raw / "primitive_rows.json",
                dict(rows=value["rows"], board_rows=value["board_rows"]),
            )
            phase("before_owned_validation")
            receipts = run_commands(
                ROOT,
                specs,
                log_dir=raw / "validation_logs",
                heartbeat_s=10,
                extra_env=dict(
                    CARNOT_8134_COVERAGE_CONFIG=str(scratch / "coverage.ini"),
                    CARNOT_8134_CLI_RECEIPTS=str(scratch / "private_cli_receipts.json"),
                    JAX_PLATFORMS="cpu",
                    OPENBLAS_NUM_THREADS="1",
                ),
            )
            owned = [r for r in receipts if r.get("scope") != "repository_health"]
            health = [r for r in receipts if r.get("scope") == "repository_health"]
            if args.input:
                owned = [
                    dict(
                        measurement[0],
                        name="private_fixture_normal_exit",
                        scope="private_fixture_only",
                    )
                ]
            for receipt in owned + health + measurement:
                receipt.update(
                    expected_exit=0,
                    normal_exit=receipt["exit_code"] >= 0 and not receipt.get("timed_out", False),
                )
            passed = all(r["passed"] and r["normal_exit"] for r in owned)
            if not passed:
                value.update(
                    honest_verdict="complete_disqualified_owned_checks",
                    verdict_class="disqualified",
                    hardware_boundary_ready_score=0,
                    measured_workload_bound_ready_score=0,
                )
            atomic_json(
                raw / "validation_receipts.json", dict(owned=owned, repository_health=health)
            )
            coverage = (
                json.loads((scratch / "coverage.json").read_bytes())
                if (scratch / "coverage.json").is_file()
                else {}
            )
            atomic_json(raw / "coverage.json", coverage)
            private_cli = (
                json.loads((scratch / "private_cli_receipts.json").read_bytes())
                if (scratch / "private_cli_receipts.json").is_file()
                else {}
            )
            atomic_json(raw / "private_cli_receipts.json", private_cli)
            phase("after_validation_freeze_candidate")
            duration = time.monotonic() - began
            spans[-1]["end_s"] = duration
            value.update(
                experiment_id=8134,
                task_id="exp8134-hardware-service-boundary",
                run_date=args.date,
                schema="carnot.hardware_service_boundary.v703.v1",
                random_seed=h.CONFIG["seed"],
                inference_substrate="verifier_ensemble_against_cached_candidates",
                inference_substrate_class="no_model_load",
                MODEL_SPECS=[],
                model_invocation_counts=ZERO_INVOCATION_COUNTS,
                call_ledger=[],
                duration_s=duration,
                phase_spans=[dict(span) for span in spans],
                claim_scope="CPU precision fixtures and historical transaction outer ceilings; independent board custody; no device speedup or independent learning benefit.",
                exposure_scope="exposed development numerical fixtures and historical receipts",
                methodology_note="Hash-authenticated primitive transactions; retain measured non-scoring costs and unknown in-envelope memory/crossings; score-envelope outer ceilings only. Exact pure-arithmetic and natural full-service costs remain unknown.",
                required_checks_passed=passed,
                flagged_adversarial=False,
                validation_receipts=owned,
                repository_health=health,
                coverage_statement_counts=coverage.get("files", {}),
                measurement_exit_receipts=measurement,
                source_artifact_hashes=data["references"],
                code_config_hashes=[reference(ROOT / p) for p in [*OWNED, TEST]],
                replay_input_reference=reference(raw / "replay_inputs.json"),
                validation_reference=reference(raw / "validation_receipts.json"),
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
                k: f"{k} binds current bounded evidence; it grants no natural learning or hardware speedup credit."
                for k in value
            }
            value["field_principles"].update(
                field_principles="Principles preserve the claim boundary of every field.",
                amdahl_bounds="Unknown in-envelope costs prevent exact arithmetic ceilings; an optimistic outer ceiling below 100 can still rule out 100x.",
                board_rows="Each original board date and hash survives independent service failure.",
                fallback_cost_rows="Fallback is required for exact decisions; unknown incremental timing is never zero.",
                gate_check_summary="The actual failed operand explains a terminal external block.",
            )
            phase("before_terminal_publication")
            publication = publish_primary(output, value, replay if args.input else terminal)
            atomic_json(
                raw / "terminal_validation.json",
                dict(
                    publication=publication,
                    required_checks_passed=passed,
                    normal_process_exit=measurement,
                ),
            )
        phase("complete")
        return 0
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(f"[exp8134] terminal_error={error}", flush=True)
        return 1
