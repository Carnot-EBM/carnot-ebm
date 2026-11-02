"""REQ-REPORT-8055: publish bounded guard evidence through existing consumers."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import tempfile
import time
from typing import Any

from carnot.reporting import hardware_guard_8055 as h
from carnot.reporting.current_work_receipt import (
    ZERO_INVOCATION_COUNTS,
    atomic_json,
    canonical_hash,
    sha256_file,
)
from carnot.reporting.evidence_features_custody_7980 import checked, reference
from carnot.reporting.experiment_7303_validation_scope import CommandSpec, run_commands
from carnot.reporting.primary_publication import publish_primary, reader_receipt

Json = dict[str, Any]
ROOT = h.ROOT
NAME = "experiment_8055_v697_hardware_guard_boundary"
TASK = "exp8055-hardware-guard-boundary"
SCRIPT = f"scripts/experiments/{NAME}.py"
TEST = "tests/python/test_hardware_guard_8055.py"
OWNED = [f"python/carnot/{NAME}.py", "python/carnot/reporting/hardware_guard_8055.py", SCRIPT]
MODEL_SPECS: list[str] = []


def commands(scratch: Path) -> list[CommandSpec]:
    """Freeze owned acceptance separately from one bounded repository observation."""
    (scratch / "pytest").mkdir(parents=True, exist_ok=True)
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
    return [
        CommandSpec(
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
                "-q",
            ),
            "owned",
            120,
        ),
        CommandSpec("coverage_combine", (*cov, "combine", "--rcfile=" + str(config)), "owned", 60),
        CommandSpec(
            "coverage_json",
            (*cov, "json", "--rcfile=" + str(config), "-o", str(scratch / "coverage.json")),
            "owned",
            60,
        ),
        CommandSpec(
            "coverage100",
            (*cov, "report", "--rcfile=" + str(config), "--fail-under=100"),
            "owned",
            60,
        ),
        CommandSpec(
            "ruff_check", (str(ROOT / ".venv/bin/ruff"), "check", *OWNED, TEST), "owned", 60
        ),
        CommandSpec(
            "ruff_format",
            (str(ROOT / ".venv/bin/ruff"), "format", "--check", *OWNED, TEST),
            "owned",
            60,
        ),
        CommandSpec(
            "strict_mypy",
            (str(ROOT / ".venv/bin/mypy"), "--strict", "--follow-imports=silent", *OWNED),
            "owned",
            120,
        ),
        CommandSpec(
            "spec_coverage", (py, "scripts/check_spec_coverage.py", *OWNED, TEST), "owned", 60
        ),
        CommandSpec(
            "full_pytest",
            (str(ROOT / ".venv/bin/pytest"), "tests/python", "-q"),
            "repository_health",
            120,
        ),
    ]


def validate(specs: list[CommandSpec], raw: Path, scratch: Path) -> tuple[list[Json], Json]:
    """Keep original failed exits and measure only newly added statements."""
    receipts = run_commands(
        ROOT,
        specs,
        log_dir=raw / "validation_logs",
        heartbeat_s=10,
        extra_env=dict(
            JAX_PLATFORMS="cpu",
            CARNOT_EXPERIMENT_ARTIFACT_ROOT=str(scratch),
            CARNOT_8019_COVERAGE_CONFIG=str(scratch / "coverage.ini"),
            CARNOT_8055_PRIVATE_CLI_RECEIPTS=str(scratch / "private_cli.json"),
            OPENBLAS_NUM_THREADS="1",
        ),
    )
    for row in receipts:
        row.update(expected_exit_code=0, actual_exit_code=row["exit_code"])
    coverage = scratch / "coverage.json"
    counts = (
        {k: v["summary"] for k, v in json.loads(coverage.read_bytes())["files"].items()}
        if coverage.is_file()
        else {}
    )
    atomic_json(raw / "coverage.json", dict(files=counts))
    atomic_json(raw / "validation_receipts.json", dict(rows=receipts))
    private_cli = scratch / "private_cli.json"
    if private_cli.is_file():
        atomic_json(raw / "private_cli_receipts.json", json.loads(private_cli.read_bytes()))
    return receipts, counts


def replay(path: Path) -> Json:
    """Cold reduction checks exact raw and code bytes before recomputing decisions."""
    value = json.loads(path.read_bytes())
    for ref in (
        value["raw_shard_hashes"] + value["code_config_hashes"] + value["cited_upstream_artifacts"]
    ):
        checked(ref)
    manifest = json.loads(checked(value["replay_input_reference"]).read_bytes())
    plan = manifest["meta"]
    plan["cases"] = [
        row
        for ref in manifest["case_shards"]
        for row in json.loads(checked(ref).read_bytes())["rows"]
    ]
    fresh = h.reduce(plan)
    for key in fresh:
        if value["verdict_class"] == "disqualified" and key in {
            "honest_verdict",
            "verdict_class",
            "hardware_custody_ready_score",
            "guard_fallback_ready_score",
        }:
            continue
        actual, expected = value[key], fresh[key]
        if key == "guard_fallback_rows":
            actual = [
                {k: v for k, v in r.items() if k != "guard_scan_and_fallback_ns"} for r in actual
            ]
            expected = [
                {k: v for k, v in r.items() if k != "guard_scan_and_fallback_ns"} for r in expected
            ]
        if actual != expected:
            raise ValueError("reduction_drift:" + key)
    return dict(passed=True, rows_checksum=canonical_hash(fresh["rows"]))


def freeze_data(raw: Path, data: Json) -> None:
    """Small case shards preserve original operands without exceeding reader budgets."""
    refs = []
    began = time.monotonic()
    for index in range(0, len(data["cases"]), 100):
        path = raw / "case_shards" / f"{index:05d}.json"
        atomic_json(path, dict(rows=data["cases"][index : index + 100]))
        refs.append(reference(path))
        print(
            f"[exp8055] freeze_case_shard elapsed_s={time.monotonic() - began:.3f} "
            f"completed={index + len(data['cases'][index : index + 100])} "
            f"pending={max(0, len(data['cases']) - index - 100)}",
            flush=True,
        )
    atomic_json(
        raw / "replay_inputs.json",
        dict(meta={k: v for k, v in data.items() if k != "cases"}, case_shards=refs),
    )


def terminal(path: Path) -> Json:
    """Fresh process reduction and both existing linters qualify the same bytes."""
    value = json.loads(path.read_bytes())
    raw = Path(value["raw_directory"])
    py = str(ROOT / ".venv/bin/python")
    specs = [
        CommandSpec(
            "cold_reduce",
            (py, "-u", str(ROOT / SCRIPT), "--cold-replay", str(path)),
            "terminal",
            120,
        ),
        CommandSpec(
            "adversarial",
            (py, "scripts/adversarial_verify.py", "--json", str(path)),
            "terminal",
            120,
        ),
        CommandSpec(
            "strict_rows",
            (py, "scripts/verdict_row_consistency_lint.py", "--strict", str(path)),
            "terminal",
            60,
        ),
    ]
    receipts = run_commands(
        ROOT,
        specs,
        log_dir=raw / "terminal_logs" / sha256_file(path).split(":")[-1],
        heartbeat_s=10,
    )
    for row in receipts:
        row.update(expected_exit_code=0, actual_exit_code=row["exit_code"])
    return dict(passed=all(r["passed"] for r in receipts), receipts=receipts)


def main(argv: list[str] | None = None) -> int:
    """Freeze resources and publish only checked terminal evidence for this invocation."""
    started = time.monotonic()
    spans: list[Json] = []

    def phase(name: str) -> None:
        elapsed = time.monotonic() - started
        if spans:
            spans[-1]["end_s"] = elapsed
        spans.append(dict(phase=name, start_s=elapsed))
        print(
            f"[exp8055] phase={name} elapsed_s={elapsed:.3f} completed={len(spans)} pending={name} "
            "model_loads=0 generations=0 device_calls=0",
            flush=True,
        )

    parser = argparse.ArgumentParser()
    parser.add_argument("--date", default="20261003")
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--fixture-input", type=Path)
    parser.add_argument("--validation-worker", action="store_true")
    parser.add_argument("--cold-replay", type=Path)
    args = parser.parse_args(argv)
    phase("start_preconditions")
    try:
        if args.date != "20261003":
            raise ValueError("run_date")
        if args.cold_replay:
            print(json.dumps(replay(args.cold_replay)), flush=True)
            return 0
        output = (args.output or args.root / "results" / (NAME + ".json")).absolute()
        raw = output.parent / "raw" / NAME / "invocations" / str(time.time_ns())
        raw.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="carnot-8055-", dir="/tmp") as temporary:
            scratch = Path(temporary)
            specs = commands(scratch)
            historical_health = (
                ROOT / "results/raw" / NAME / "development_failures/repository_health.json"
            )
            if historical_health.is_file():
                specs = [c for c in specs if c.scope != "repository_health"]
            code = [reference(ROOT / p) for p in [*OWNED, TEST]]
            atomic_json(
                raw / "methods.json",
                dict(
                    config=h.CONFIG,
                    code=code,
                    commands=[asdict(c) for c in specs],
                    artifact_guard_enabled=True,
                    substrate="no model load; no device invocation",
                    sources=h.SOURCES,
                ),
            )
            phase("authenticate_inputs_before_small_head_load")
            data = (
                json.loads(args.fixture_input.read_bytes())
                if args.fixture_input
                else h.load(args.root, raw)
            )
            phase("authenticate_inputs_after_small_head_load")
            freeze_data(raw, data)
            phase("before_guard_replay_benchmark")
            value = h.reduce(data)
            data["cpu_guard_cost_rows"] = value["guard_fallback_rows"]
            freeze_data(raw, data)
            phase("after_guard_replay_benchmark")
            atomic_json(raw / "primitive_rows.json", dict(rows=value["rows"]))
            atomic_json(raw / "checkpoints.json", value["final_checkpoints"])
            phase("before_validation_subprocesses")
            receipts, counts = ([], {}) if args.validation_worker else validate(specs, raw, scratch)
            if historical_health.is_file():
                receipts += json.loads(historical_health.read_bytes())["rows"]
            phase("after_validation_subprocesses")
            owned_passed = set(counts) == set(OWNED) and all(
                c["missing_lines"] == 0 for c in counts.values()
            )
            owned_passed = owned_passed and all(
                r["passed"] for r in receipts if r["scope"] == "owned"
            )
            if not args.validation_worker and not owned_passed:
                value.update(
                    honest_verdict="complete_disqualified_owned_checks",
                    verdict_class="disqualified",
                    hardware_custody_ready_score=0,
                    guard_fallback_ready_score=0,
                )
            value.update(
                experiment_id=8055,
                task_id=TASK,
                milestone="2026.10.697",
                run_date=args.date,
                schema="carnot.hardware_guard_boundary.v697.v1",
                config=h.CONFIG,
                claim_scope="This invocation authenticates historical custody and CPU guard emulation on exposed development. No device performance, future safety or service acceleration.",
                methodology="Freeze analytical Q12 input and operation rounding cells; propagate calibrated probabilities and all guard predicates; use CPU shadow before committing any uncertainty; restart exact authoritative state.",
                inference_substrate="aggregation_from_upstream_artifacts",
                inference_substrate_class="no_model_load",
                MODEL_SPECS=[],
                model_specs=[],
                model_invocation_counts=dict(ZERO_INVOCATION_COUNTS),
                substrate_declaration=dict(
                    reduction="aggregation_from_upstream_artifacts",
                    numerical_work="verifier_scoring",
                    inference_substrate_class="no_model_load",
                    MODEL_SPECS=[],
                    pretrained_model_calls=0,
                ),
                trained_head_specs=[
                    dict(
                        parameter_count=len(data["head"]["parameters"]),
                        pretrained=False,
                        current_training=False,
                    )
                ]
                if "head" in data
                else [],
                preconditions_checked=data["checks"],
                random_seed=8055,
                cited_upstream_artifacts=data["references"],
                code_config_hashes=code,
                raw_directory=str(raw),
                replay_input_reference=reference(raw / "replay_inputs.json"),
                checkpoint_references=[reference(raw / "checkpoints.json")],
                validation_receipts=receipts,
                coverage_statement_counts=counts,
                owned_checks_passed=owned_passed,
                repository_health=[r for r in receipts if r["scope"] == "repository_health"],
                terminal_validation_sidecar_path=str(raw / "terminal_validation.json"),
                flagged_adversarial=False,
            )
            value["raw_shard_hashes"] = [
                reference(p) for p in [*raw.glob("*.json"), *(raw / "case_shards").glob("*.json")]
            ]
            value["reproducibility_checksum"] = canonical_hash(
                dict(config=h.CONFIG, code=code, inputs=value["replay_input_reference"])
            )
            phase("freeze_candidate_before_publication")
            value["duration_s"] = time.monotonic() - started
            spans[-1]["end_s"] = value["duration_s"]
            value["phase_spans"] = spans
            value["field_principles"] = {
                k: "Bind "
                + k
                + " to actual current evidence; imported history and controls do not prove device benefit."
                for k in value
            }
            publication = publish_primary(output, value, terminal)
            published = terminal(output)
            reader = reader_receipt(
                TASK,
                output.parent,
                field="guard_fallback_ready_score",
                expected=value["guard_fallback_ready_score"],
            )
            atomic_json(
                raw / "terminal_validation.json",
                dict(publication=publication, published=published, reader=reader),
            )
            if not published["passed"] or not reader["passed"]:
                raise ValueError("published_validation_failed")
            phase("complete")
            return 0
    except (OSError, ValueError, KeyError, TimeoutError) as error:
        print(f"[exp8055] terminal_error={error}", flush=True)
        return 1
