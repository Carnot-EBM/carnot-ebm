"""REQ-REPORT-8121: publish only checked memory counts and historical boundaries."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import tempfile
import time
from typing import Any

from carnot.reporting import hardware_batch_8121 as h
from carnot.reporting import hardware_batch_inputs_8121 as inputs
from carnot.reporting.current_work_receipt import (
    ZERO_INVOCATION_COUNTS,
    atomic_json,
    canonical_hash,
)
from carnot.reporting.evidence_features_custody_7980 import checked, reference
from carnot.reporting.experiment_7303_validation_scope import CommandSpec, run_commands
from carnot.reporting.primary_publication import publish_primary

Json = dict[str, Any]
ROOT = h.prior.ROOT
NAME = "experiment_8121_v702_hardware_batch_boundary"
SCRIPT = f"scripts/experiments/{NAME}.py"
TEST = "tests/python/test_hardware_batch_8121.py"
OWNED = [
    f"python/carnot/{NAME}.py",
    "python/carnot/reporting/hardware_batch_8121.py",
    "python/carnot/reporting/hardware_batch_inputs_8121.py",
    SCRIPT,
]


def commands(scratch: Path) -> list[CommandSpec]:
    """Freeze exact checks and limit coverage to the newly introduced code."""
    scratch.mkdir(parents=True, exist_ok=True)
    cfg = scratch / "coverage.ini"
    cfg.write_text(
        "[run]\nparallel=True\ndata_file="
        + str(scratch / ".coverage")
        + "\ninclude=\n    "
        + "\n    ".join(str(ROOT / p) for p in OWNED)
        + "\n"
    )
    py = str(ROOT / ".venv/bin/python")
    cov = (py, "-m", "coverage")
    tests = (
        TEST,
        "tests/python/test_primary_publication_7928.py",
        "tests/python/test_current_work_receipt.py",
    )
    plan = [
        (
            "owned_tests_and_private_E2E",
            (
                *cov,
                "run",
                "--parallel-mode",
                "--rcfile=" + str(cfg),
                "-m",
                "pytest",
                "-n",
                "0",
                "-o",
                "addopts=",
                "--no-cov",
                "--basetemp=" + str(scratch / "pytest"),
                *tests,
                "-q",
            ),
            180,
        ),
        ("coverage_combine", (*cov, "combine", "--rcfile=" + str(cfg)), 30),
        (
            "coverage_json",
            (*cov, "json", "--rcfile=" + str(cfg), "-o", str(scratch / "coverage.json")),
            30,
        ),
        (
            "coverage100",
            (*cov, "report", "--rcfile=" + str(cfg), "--show-missing", "--fail-under=100"),
            30,
        ),
        ("ruff_check", (str(ROOT / ".venv/bin/ruff"), "check", *OWNED, TEST), 30),
        ("ruff_format", (str(ROOT / ".venv/bin/ruff"), "format", "--check", *OWNED, TEST), 30),
        (
            "strict_mypy",
            (str(ROOT / ".venv/bin/mypy"), "--strict", "--follow-imports=silent", *OWNED),
            90,
        ),
        ("spec_coverage", (py, "scripts/check_spec_coverage.py", TEST), 30),
    ]
    return [CommandSpec(n, a, "owned", t) for n, a, t in plan] + [
        CommandSpec(
            "full_pytest",
            (str(ROOT / ".venv/bin/pytest"), "tests/python", "-q"),
            "repository_health",
            120,
        )
    ]


def replay(path: Path) -> Json:
    """Rehash saved operands and independently recompute every scientific field.

    Validation receipts and the frozen manifest are themselves sealed shards.
    A changed report cannot borrow the original passing tests or source hashes.
    """
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
    """Bind cold replay and both terminal validators to the same candidate bytes."""
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
    """Only passing current terminal checks permit reader-visible publication."""
    receipts = run_commands(
        ROOT,
        terminal_commands(path),
        log_dir=path.parent / "terminal_logs" / str(time.time_ns()),
        heartbeat_s=10,
    )
    for receipt in receipts:
        receipt["normal_exit"] = receipt["exit_code"] >= 0 and not receipt.get("timed_out", False)
    return dict(passed=all(r["passed"] for r in receipts), receipts=receipts)


def main(argv: list[str] | None = None) -> int:
    """Use a normally exited child for computation and validate before publication."""
    os.environ["PYTHONUNBUFFERED"] = "1"
    began = time.monotonic()
    spans: list[Json] = []

    def phase(name: str) -> None:
        elapsed = time.monotonic() - began
        if spans:
            spans[-1]["end_s"] = elapsed
        spans.append(dict(phase=name, start_s=elapsed))
        print(
            f"[exp8121] phase={name} elapsed_s={elapsed:.3f} model_loads=0 device_calls=0",
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
            phase("before_reduction")
            atomic_json(output, h.reduce(json.loads(args.worker_input.read_bytes())))
            phase("after_reduction")
            return 0
        raw = output.parent / "raw" / NAME / "invocations" / str(time.time_ns())
        raw.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="carnot8121-", dir="/tmp") as temporary:
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
            phase("before_measurement_subprocess")
            child = CommandSpec(
                "normal_measurement_exit",
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
            measurement[0]["normal_exit"] = measurement[0].get("exit_code", -1) == 0
            if not measurement[0]["passed"]:
                raise ValueError("measurement_child_failed")
            phase("after_measurement_independent_reduction")
            value = json.loads((raw / "worker.json").read_bytes())
            independent = h.reduce(data)
            if value != independent:
                raise ValueError("independent_reduction_drift")
            atomic_json(raw / "independent_reduction.json", independent)
            atomic_json(
                raw / "primitive_rows.json",
                dict(
                    operation_count_rows=value["operation_count_rows"],
                    precision_rows=value["precision_rows"],
                    board_rows=value["board_rows"],
                    observed_touch_rows=value["observed_touch_rows"],
                ),
            )
            phase("before_validation_subprocesses")
            receipts = run_commands(
                ROOT,
                specs,
                log_dir=raw / "validation_logs",
                heartbeat_s=10,
                extra_env=dict(
                    CARNOT_8121_COVERAGE_CONFIG=str(scratch / "coverage.ini"),
                    CARNOT_8121_CLI_RECEIPTS=str(scratch / "private_cli_receipts.json"),
                    JAX_PLATFORMS="cpu",
                    OPENBLAS_NUM_THREADS="1",
                ),
            )
            owned = [r for r in receipts if r.get("scope") != "repository_health"]
            health = [r for r in receipts if r.get("scope") == "repository_health"]
            for receipt in receipts:
                receipt["normal_exit"] = receipt.get("exit_code", -1) >= 0 and not receipt.get(
                    "timed_out", False
                )
            if args.input:
                owned = [
                    dict(
                        measurement[0],
                        name="private_fixture_normal_exit",
                        scope="private_fixture_only",
                    )
                ]
            passed = all(r["passed"] for r in owned)
            if not passed:
                value.update(
                    honest_verdict="complete_disqualified_owned_checks",
                    verdict_class="disqualified",
                    hardware_boundary_ready_score=0,
                )
            phase("after_validation_freeze_candidate")
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
            duration = time.monotonic() - began
            spans[-1]["end_s"] = duration
            value.update(
                experiment_id=8121,
                task_id="exp8121-hardware-batch-boundary",
                run_date=args.date,
                schema="carnot.hardware_batch_boundary.v702.v1",
                random_seed=h.CONFIG["seed"],
                inference_substrate="aggregation_from_upstream_artifacts",
                inference_substrate_class="no_model_load",
                MODEL_SPECS=[],
                model_invocation_counts=ZERO_INVOCATION_COUNTS,
                call_ledger=[],
                duration_s=duration,
                phase_spans=[dict(span) for span in spans],
                claim_scope="Analytic batch traffic and replayed storage bounds; independent historical board evidence; no local device speedup.",
                exposure_scope="exposed development numerical fixtures and historical receipts",
                methodology_note="Exact operation and packed-byte counts; Exp8108 analytic storage envelope and float64 fallback; complete-cost arithmetic-free ceilings only for qualified whole-service rows.",
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
                k: f"{k} records bounded evidence and grants no hardware performance or independent learning credit."
                for k in value
            }
            value["field_principles"].update(
                field_principles="Each field names the inference boundary it preserves.",
                conditional_speedup_bound="Unknown acquisition, transfer, fallback or persistence suppresses the ceiling; none is free.",
                operation_count_rows="Analytic fixtures bound memory work; they are not measured board transfers.",
                board_rows="Original dates, workloads and hashes stay separate from current CPU replay.",
                precision_rows="Analytic storage bounds do not certify an unimplemented fixed-point LUT.",
                hardware_boundary_ready_score="Assessment completion cannot authorize hardware or benefit claims.",
                required_checks_passed="Only current owned normal exits support publication; global health stays separate.",
            )
            phase("before_terminal_publication")
            publication = publish_primary(
                output, value, (lambda p: replay(p)) if args.input else terminal
            )
            atomic_json(
                raw / "terminal_validation.json",
                dict(
                    publication=publication,
                    normal_process_exit=measurement,
                    required_checks_passed=passed,
                ),
            )
        phase("complete")
        return 0
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(f"[exp8121] terminal_error={error}", flush=True)
        return 1
