"""REQ-REPORT-8081: publish checked read-only custody after normal reduction exit."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import tempfile
import time
from typing import Any

from carnot.reporting import hardware_workload_8081 as h
from carnot.reporting.current_work_receipt import (
    atomic_json,
    build_current_work_receipt,
    canonical_hash,
    sha256_file,
)
from carnot.reporting.evidence_features_custody_7980 import checked, reference
from carnot.reporting.experiment_7303_validation_scope import CommandSpec, run_commands
from carnot.reporting.primary_publication import publish_primary, reader_receipt

Json = dict[str, Any]
ROOT = h.ROOT
NAME = "experiment_8081_v699_hardware_workload_boundary"
TASK = "exp8081-hardware-workload-boundary"
SCRIPT = f"scripts/experiments/{NAME}.py"
TEST = "tests/python/test_hardware_workload_8081.py"
OWNED = [f"python/carnot/{NAME}.py", "python/carnot/reporting/hardware_workload_8081.py", SCRIPT]
MODEL_SPECS: list[str] = []


def commands(scratch: Path) -> list[CommandSpec]:
    """Freeze checks before reduction; all test scratch stays outside results."""
    config = scratch / "coverage.ini"
    config.write_text(
        "[run]\nparallel=True\ndata_file="
        + str(scratch / ".coverage")
        + "\ninclude=\n    "
        + "\n    ".join(str(ROOT / p) for p in OWNED)
        + "\n"
    )
    py = str(ROOT / ".venv/bin/python")
    cov = (py, "-m", "coverage")
    fixture = str(scratch / "e2e016.json")
    specs = [
        ("environment", (py, "-c", "import sys,pytest,coverage,ruff,mypy; print(sys.version)"), 30),
        (
            "owned_tests",
            (
                *cov,
                "run",
                "--rcfile=" + str(config),
                "-m",
                "pytest",
                "-n",
                "0",
                "-o",
                "addopts=",
                "--no-cov",
                "--basetemp=" + str(scratch / "pytest"),
                TEST,
                "tests/python/test_primary_publication_7928.py",
                "tests/python/test_current_work_receipt.py",
                "-q",
            ),
            120,
        ),
        ("coverage_combine", (*cov, "combine", "--rcfile=" + str(config)), 30),
        (
            "coverage_json",
            (*cov, "json", "--rcfile=" + str(config), "-o", str(scratch / "coverage.json")),
            30,
        ),
        ("coverage100", (*cov, "report", "--rcfile=" + str(config), "--fail-under=100"), 30),
        ("ruff_check", (str(ROOT / ".venv/bin/ruff"), "check", *OWNED, TEST), 30),
        ("ruff_format", (str(ROOT / ".venv/bin/ruff"), "format", "--check", *OWNED, TEST), 30),
        (
            "strict_mypy",
            (str(ROOT / ".venv/bin/mypy"), "--strict", "--follow-imports=silent", *OWNED),
            90,
        ),
        ("spec_coverage", (py, "scripts/check_spec_coverage.py", *OWNED, TEST), 30),
        (
            "e2e016_fixture",
            (
                py,
                "scripts/experiments/experiment_7868_v683_intervention_protocol.py",
                "--date",
                "20260929",
                "--fixture-e2e",
                fixture,
            ),
            90,
        ),
        (
            "e2e016_cold_replay",
            (
                py,
                "scripts/experiments/experiment_7868_v683_intervention_protocol.py",
                "--date",
                "20260929",
                "--cold-replay",
                fixture,
            ),
            90,
        ),
    ]
    return [CommandSpec(name, argv, "owned", timeout) for name, argv, timeout in specs] + [
        CommandSpec(
            "full_pytest",
            (str(ROOT / ".venv/bin/pytest"), "tests/python", "-q"),
            "repository_health",
            120,
        )
    ]


def replay(path: Path) -> Json:
    """Verify frozen bytes and recompute every scientific operand in a fresh process."""
    value = json.loads(path.read_bytes())
    for ref in (
        value["raw_shard_hashes"] + value["source_artifact_hashes"] + value["code_config_hashes"]
    ):
        checked(ref)
    data = json.loads(checked(value["replay_input_reference"]).read_bytes())
    fresh = h.reduce(data)
    for key, wanted in fresh.items():
        if value["verdict_class"] == "disqualified" and key in {
            "honest_verdict",
            "verdict_class",
            "hardware_custody_ready_score",
            "combined_workload_bound_ready_score",
        }:
            continue
        if value[key] != wanted:
            raise ValueError("reduction_drift:" + key)
    return dict(passed=True, rows_checksum=canonical_hash(fresh["rows"]))


def terminal_commands(path: Path) -> list[CommandSpec]:
    """Freeze the terminal reader arguments before exposing a candidate."""
    py = str(ROOT / ".venv/bin/python")
    specs = [
        CommandSpec(
            "cold_reduce",
            (py, "-u", str(ROOT / SCRIPT), "--cold-replay", str(path)),
            "terminal",
            90,
        ),
        CommandSpec(
            "adversarial",
            (py, "scripts/adversarial_verify.py", "--json", str(path)),
            "terminal",
            90,
        ),
        CommandSpec(
            "strict_rows",
            (py, "scripts/verdict_row_consistency_lint.py", "--strict", str(path)),
            "terminal",
            60,
        ),
    ]
    return specs


def terminal(path: Path) -> Json:
    """Owned cold readers and strict row lint must accept the exact candidate bytes."""
    value = json.loads(path.read_bytes())
    receipts = run_commands(
        ROOT,
        terminal_commands(path),
        log_dir=Path(value["raw_directory"]) / "terminal_logs" / sha256_file(path).split(":")[1],
        heartbeat_s=10,
    )
    return dict(passed=all(r["passed"] for r in receipts), receipts=receipts)


def main(argv: list[str] | None = None) -> int:
    """No model or board is invoked; the parent publishes only an exited child's data."""
    os.environ["PYTHONUNBUFFERED"] = "1"
    started_ns = time.monotonic_ns()
    spans: list[Json] = []

    def phase(name: str) -> None:
        elapsed = (time.monotonic_ns() - started_ns) / 1e9
        if spans:
            spans[-1]["end_s"] = elapsed
        spans.append(dict(phase=name, start_s=elapsed))
        print(
            f"[exp8081] phase={name} elapsed_s={elapsed:.3f} completed={len(spans)} "
            f"pending={name} model_loads=0 generations=0 device_calls=0",
            flush=True,
        )

    phase("start_preconditions")
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", default="20261003")
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--worker-input", type=Path)
    parser.add_argument("--cold-replay", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.date != "20261003":
            raise ValueError("run_date")
        if args.cold_replay:
            print(json.dumps(replay(args.cold_replay)), flush=True)
            return 0
        if args.worker_input:
            phase("before_primitive_reduction")
            data = json.loads(args.worker_input.read_bytes())
            atomic_json(args.output, h.reduce(data))
            phase("after_primitive_reduction_normal_exit")
            return 0
        output = (args.output or args.root / "results" / (NAME + ".json")).absolute()
        raw = output.parent / "raw" / NAME / "invocations" / str(time.time_ns())
        raw.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="carnot8081-", dir="/tmp") as temporary:
            scratch = Path(temporary)
            specs = commands(scratch)
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
                    scope="frozen before any reduction; no new model or device measurement",
                ),
            )
            phase("authenticate_named_resources_and_original_custody")
            data = h.load(args.root, raw)
            atomic_json(raw / "replay_inputs.json", data)
            phase("before_reduction_subprocess")
            child = CommandSpec(
                "reduction_normal_exit",
                (
                    str(ROOT / ".venv/bin/python"),
                    "-u",
                    str(ROOT / SCRIPT),
                    "--worker-input",
                    str(raw / "replay_inputs.json"),
                    "--output",
                    str(raw / "reduction.json"),
                ),
                "owned",
                90,
            )
            exit_receipts = run_commands(ROOT, [child], log_dir=raw / "child_logs", heartbeat_s=10)
            phase("after_reduction_subprocess")
            if not exit_receipts[0]["passed"]:
                raise ValueError("reduction_child_failed")
            value = json.loads((raw / "reduction.json").read_bytes())
            atomic_json(
                raw / "primitive_rows.json",
                dict(
                    rows=value["rows"],
                    board_rows=value["board_rows"],
                    mode_qualification_rows=value["mode_qualification_rows"],
                    compatible_component_fractions=value["compatible_component_fractions"],
                    acceleration_bounds=value["acceleration_bounds"],
                ),
            )
            phase("before_validation_subprocesses")
            receipts = run_commands(
                ROOT,
                specs,
                log_dir=raw / "validation_logs",
                heartbeat_s=10,
                extra_env=dict(
                    JAX_PLATFORMS="cpu",
                    OPENBLAS_NUM_THREADS="1",
                    CARNOT_EXPERIMENT_ARTIFACT_ROOT=str(scratch),
                    CARNOT_8081_PRIVATE_CLI_RECEIPTS=str(scratch / "private_cli.json"),
                ),
            )
            phase("after_validation_subprocesses")
            health = [r for r in receipts if r["scope"] == "repository_health"]
            receipts = [r for r in receipts if r["scope"] == "owned"]
            counts = (
                json.loads((scratch / "coverage.json").read_bytes())["files"]
                if (scratch / "coverage.json").is_file()
                else {}
            )
            passed = (
                all(r["passed"] for r in receipts)
                and set(counts) == set(OWNED)
                and all(r["summary"]["missing_lines"] == 0 for r in counts.values())
            )
            if not passed:
                value.update(
                    honest_verdict="complete_disqualified_owned_checks",
                    verdict_class="disqualified",
                    hardware_custody_ready_score=0,
                    combined_workload_bound_ready_score=0,
                )
            prior = json.loads((ROOT / h.CUSTODY).read_bytes())
            for label, payload in (
                ("validation_receipts", dict(rows=receipts)),
                ("coverage", dict(files=counts)),
            ):
                atomic_json(raw / (label + ".json"), payload)
            if (scratch / "private_cli.json").is_file():
                atomic_json(
                    raw / "private_cli_receipts.json",
                    json.loads((scratch / "private_cli.json").read_bytes()),
                )
            phase("freeze_candidate_before_publication")
            ended_ns = time.monotonic_ns()
            spans[-1]["end_s"] = (ended_ns - started_ns) / 1e9
            receipt = build_current_work_receipt(
                run_id=str(raw.name),
                owner_pid=os.getpid(),
                events=[],
                inference_substrate="aggregation_from_upstream_artifacts",
                inference_substrate_details=dict(no_model_load=True),
                inference_substrate_class="no_model_load",
                execution_venue="host",
                started_monotonic_ns=started_ns,
                ended_monotonic_ns=ended_ns,
                phase_spans=spans,
            )
            value.update(
                experiment_id=8081,
                task_id=TASK,
                run_date=args.date,
                milestone="2026.10.699",
                schema="carnot.hardware_feature_boundary.v699.v1",
                random_seed=8081,
                inference_substrate="aggregation_from_upstream_artifacts",
                inference_substrate_class="no_model_load",
                MODEL_SPECS=[],
                model_specs=[],
                model_invocation_counts=receipt["invocation_counts"],
                substrate_declaration=dict(
                    reduction="aggregation_from_upstream_artifacts",
                    no_model_load=True,
                    MODEL_SPECS=[],
                ),
                current_work_receipt=receipt,
                duration_s=receipt["duration_s"],
                phase_spans=spans,
                claim_scope="Read-only historical custody and conditional bounds only. No current device execution, speed or purchase claim.",
                methodology="Authenticate byte-bound history independently; reduce only qualified current costs with explicit unknown transfer and queue.",
                required_checks_passed=passed,
                validation_receipts=receipts,
                measurement_exit_receipt=exit_receipts[0],
                coverage_statement_counts=counts,
                title="Historical board custody and corrected workload boundary",
                repository_health=prior["repository_health"] + health,
                source_artifact_hashes=data["references"],
                code_config_hashes=[reference(ROOT / p) for p in [*OWNED, TEST]],
                raw_directory=str(raw),
                replay_input_reference=reference(raw / "replay_inputs.json"),
                raw_shard_hashes=[reference(p) for p in [*raw.glob("*.json"), *raw.rglob("*.log")]],
                terminal_validation_sidecar_path=str(raw / "terminal_validation.json"),
                flagged_adversarial=False,
            )
            value["reproducibility_checksum"] = canonical_hash(
                dict(inputs=value["replay_input_reference"], code=value["code_config_hashes"])
            )
            value["field_principles"] = {
                key: f"Bind {key} to exact current operands; history, controls and estimates grant no device benefit."
                for key in value
            }
            value["field_principles"]["field_principles"] = (
                "Each field states the inference error it prevents."
            )
            publication = publish_primary(output, value, terminal)
            reader = reader_receipt(
                TASK,
                output.parent,
                field="hardware_custody_ready_score",
                expected=value["hardware_custody_ready_score"],
            )
            atomic_json(
                raw / "terminal_validation.json",
                dict(publication=publication, reader=reader, normal_process_exit=exit_receipts[0]),
            )
            if not reader["passed"]:
                raise ValueError("published_reader_failed")
            phase("complete")
            return 0
    except (OSError, ValueError, KeyError, TimeoutError) as error:
        print(f"[exp8081] terminal_error={error}", flush=True)
        return 1
