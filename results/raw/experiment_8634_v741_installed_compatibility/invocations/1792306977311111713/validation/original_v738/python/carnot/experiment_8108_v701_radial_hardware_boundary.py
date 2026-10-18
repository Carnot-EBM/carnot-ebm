"""REQ-REPORT-8108: checked CPU assessment with no model or device execution."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import tempfile
import time
from typing import Any

from carnot.reporting import radial_hardware_8108 as h
from carnot.reporting.current_work_receipt import (
    ZERO_INVOCATION_COUNTS,
    atomic_json,
    canonical_hash,
)
from carnot.reporting.evidence_features_custody_7980 import checked, reference
from carnot.reporting.experiment_7303_validation_scope import CommandSpec, run_commands
from carnot.reporting.primary_publication import publish_primary

Json = dict[str, Any]
ROOT = h.ROOT
NAME = "experiment_8108_v701_radial_hardware_boundary"
SCRIPT = f"scripts/experiments/{NAME}.py"
TEST = "tests/python/test_radial_hardware_8108.py"
OWNED = [f"python/carnot/{NAME}.py", "python/carnot/reporting/radial_hardware_8108.py", SCRIPT]


def commands(scratch: Path) -> list[CommandSpec]:
    """Freeze exact commands before measurement, covering only newly introduced files."""
    scratch.mkdir(parents=True, exist_ok=True)
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
    tests = (
        TEST,
        "tests/python/test_primary_publication_7928.py",
        "tests/python/test_current_work_receipt.py",
        "tests/python/test_experiment_7891_v685_authority_lifecycle.py",
    )
    plan = [
        (
            "owned_tests",
            (
                *cov,
                "run",
                "--parallel-mode",
                "--rcfile=" + str(config),
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
            120,
        ),
        ("coverage_combine", (*cov, "combine", "--rcfile=" + str(config)), 30),
        (
            "coverage_json",
            (*cov, "json", "--rcfile=" + str(config), "-o", str(scratch / "coverage.json")),
            30,
        ),
        (
            "coverage100",
            (*cov, "report", "--rcfile=" + str(config), "--show-missing", "--fail-under=100"),
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
    return [CommandSpec(name, argv, "owned", timeout) for name, argv, timeout in plan] + [
        CommandSpec(
            "full_pytest",
            (str(ROOT / ".venv/bin/pytest"), "tests/python", "-q"),
            "repository_health",
            120,
        )
    ]


def replay(path: Path) -> Json:
    """Cold checks rehash sealed bytes and recompute every scientific output."""
    value = json.loads(path.read_bytes())
    for ref in (
        value["source_artifact_hashes"] + value["code_config_hashes"] + value["raw_shard_hashes"]
    ):
        checked(ref)
    data = json.loads(checked(value["replay_input_reference"]).read_bytes())
    fresh = h.reduce(data)
    for key, wanted in fresh.items():
        if value["verdict_class"] == "disqualified" and key in {
            "honest_verdict",
            "verdict_class",
            "hardware_boundary_ready_score",
        }:
            continue
        if value[key] != wanted:
            raise ValueError("reduction_drift:" + key)
    return dict(passed=True, rows_checksum=canonical_hash(fresh["rows"]))


def terminal_commands(path: Path) -> list[CommandSpec]:
    """The terminal checks consume the same candidate bytes that readers will see."""
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
    """A failed current validator prevents publication and leaves its receipt intact."""
    receipts = run_commands(
        ROOT,
        terminal_commands(path),
        log_dir=path.parent / "terminal_logs" / str(time.time_ns()),
        heartbeat_s=10,
    )
    return dict(passed=all(r["passed"] for r in receipts), receipts=receipts)


def main(argv: list[str] | None = None) -> int:
    """The worker exits normally before the parent validates and publishes evidence."""
    os.environ["PYTHONUNBUFFERED"] = "1"
    began = time.monotonic()
    spans: list[Json] = []

    def phase(name: str) -> None:
        elapsed = time.monotonic() - began
        if spans:
            spans[-1]["end_s"] = elapsed
        spans.append(dict(phase=name, start_s=elapsed))
        print(
            f"[exp8108] phase={name} elapsed_s={elapsed:.3f} model_loads=0 device_calls=0",
            flush=True,
        )

    phase("start_preconditions")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", choices=["20261004"], default="20261004")
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--worker-input", type=Path)
    parser.add_argument("--cold-replay", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.cold_replay:
            print(json.dumps(replay(args.cold_replay)), flush=True)
            return 0
        output = (args.output or args.root / "results" / (NAME + ".json")).absolute()
        raw = output.parent / "raw" / NAME / "invocations" / str(time.time_ns())
        raw.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="carnot8108-", dir="/tmp") as temporary:
            scratch = Path(temporary)
            specs = [] if args.worker_input else commands(scratch)
            atomic_json(
                raw / "validation_manifest.json",
                dict(
                    commands=[asdict(s) for s in specs],
                    terminal_commands=[
                        asdict(s) for s in terminal_commands(raw / "terminal_candidate.json")
                    ],
                    config=h.CONFIG,
                    scope="frozen before CPU arithmetic",
                ),
            )
            phase("authenticate_inputs")
            data = (
                json.loads(args.worker_input.read_bytes())
                if args.worker_input
                else h.load(args.root, raw)
            )
            atomic_json(raw / "replay_inputs.json", data)
            phase("before_measurement")
            if args.worker_input:
                value = h.reduce(data)
                measurement: list[Json] = []
            else:
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
                if not measurement[0]["passed"]:
                    raise ValueError("measurement_child_failed")
                value = json.loads((raw / "worker.json").read_bytes())
            phase("after_measurement_before_independent_reduction")
            independent = h.reduce(data)
            if any(value[k] != v for k, v in independent.items()):
                raise ValueError("independent_reduction_drift")
            atomic_json(raw / "independent_reduction.json", independent)
            phase("before_validation_subprocesses")
            receipts = run_commands(
                ROOT,
                specs,
                log_dir=raw / "validation_logs",
                heartbeat_s=10,
                extra_env=dict(
                    CARNOT_8108_COVERAGE_CONFIG=str(scratch / "coverage.ini"),
                    CARNOT_8108_CLI_RECEIPTS=str(scratch / "private_cli_receipts.json"),
                    JAX_PLATFORMS="cpu",
                    OPENBLAS_NUM_THREADS="1",
                ),
            )
            health = [r for r in receipts if r.get("scope") == "repository_health"]
            owned = [r for r in receipts if r.get("scope") != "repository_health"]
            passed = all(r["passed"] for r in owned)
            if not passed:
                value.update(
                    honest_verdict="complete_disqualified_owned_checks",
                    verdict_class="disqualified",
                    hardware_boundary_ready_score=0,
                )
            phase("after_validation_freeze_candidate")
            duration = time.monotonic() - began
            spans[-1]["end_s"] = duration
            atomic_json(
                raw / "validation_receipts.json", dict(owned=owned, repository_health=health)
            )
            coverage = (
                json.loads((scratch / "coverage.json").read_bytes())
                if (scratch / "coverage.json").is_file()
                else {}
            )
            private_cli = (
                json.loads((scratch / "private_cli_receipts.json").read_bytes())
                if (scratch / "private_cli_receipts.json").is_file()
                else {}
            )
            atomic_json(raw / "coverage.json", coverage)
            atomic_json(raw / "private_cli_receipts.json", private_cli)
            value.update(
                experiment_id=8108,
                task_id="exp8108-radial-hardware-boundary",
                run_date=args.date,
                milestone="2026.10.701",
                schema="carnot.radial_hardware_boundary.v701.v1",
                random_seed=h.CONFIG["seed"],
                inference_substrate="aggregation_from_upstream_artifacts",
                inference_substrate_class="no_model_load",
                MODEL_SPECS=[],
                model_invocation_counts=ZERO_INVOCATION_COUNTS,
                duration_s=duration,
                phase_spans=spans,
                claim_scope="CPU fixture storage precision and historical per-board custody; no current hardware acceleration.",
                methodology_note="Gaussian/sigmoid Lipschitz storage envelope with outward rounding and declared host allowance; independently charged complete-service primitives.",
                required_checks_passed=passed,
                coverage_statement_counts=coverage.get("files", {}),
                validation_receipts=owned,
                repository_health=health,
                measurement_exit_receipts=measurement,
                flagged_adversarial=False,
                source_artifact_hashes=data["references"],
                code_config_hashes=[reference(ROOT / p) for p in [*OWNED, TEST]],
                replay_input_reference=reference(raw / "replay_inputs.json"),
                raw_shard_hashes=[reference(p) for p in raw.rglob("*") if p.is_file()],
                terminal_validation_sidecar_path=str(raw / "terminal_validation.json"),
            )
            value["reproducibility_checksum"] = canonical_hash(
                dict(
                    inputs=value["replay_input_reference"],
                    code=value["code_config_hashes"],
                    config=h.CONFIG,
                )
            )
            value["field_principles"] = {
                key: f"{key} records scoped evidence; it grants no hardware or independent learning claim."
                for key in value
            }
            value["field_principles"]["field_principles"] = (
                "Explain the inference error prevented by each field."
            )
            if args.worker_input:
                atomic_json(output, value)
            else:
                publication = publish_primary(output, value, terminal)
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
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f"[exp8108] terminal_error={exc}", flush=True)
        return 1
