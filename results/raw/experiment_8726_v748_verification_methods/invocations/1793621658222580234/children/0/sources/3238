"""REQ-REPORT-8016: freeze, measure and publish an honest hardware boundary."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import tempfile
import time
from typing import Any

from carnot.reporting import hardware_update_8016 as h
from carnot.reporting.current_work_receipt import (
    ZERO_INVOCATION_COUNTS,
    atomic_json,
    canonical_hash,
    sha256_file,
)
from carnot.reporting.experiment_7303_validation_scope import (
    CommandSpec,
    build_scoped_commands,
    run_commands,
)
from carnot.reporting.primary_publication import publish_primary, reader_receipt
from scripts.experiment_template import normalize_artifact_for_template_write

Json = dict[str, Any]
ROOT = h.ROOT
NAME = "experiment_8016_v694_hardware_update_boundary"
TASK = "exp8016-hardware-update-boundary"
MODEL_SPECS: list[str] = []
OWNED = [
    f"python/carnot/{NAME}.py",
    "python/carnot/reporting/hardware_update_8016.py",
    f"scripts/experiments/{NAME}.py",
]
TESTS = ["tests/python/test_hardware_update_8016.py", f"tests/python/test_{NAME}.py"]


def commands(raw: Path, scratch: Path) -> list[CommandSpec]:
    """Reuse the scoped supervisor and keep all test output outside results."""
    (scratch / "pytest").mkdir(parents=True, exist_ok=True)
    specs = build_scoped_commands(
        ROOT,
        TESTS,
        OWNED[:-1],
        static_paths=[OWNED[-1]],
        basetemp=scratch / "pytest",
        coverage_file=scratch / ".coverage",
    )
    include = ",".join(str(ROOT / p) for p in OWNED)
    specs = [
        CommandSpec(
            r.name,
            tuple("--include=" + include if a.startswith("--include=") else a for a in r.argv),
            r.scope,
            600,
        )
        for r in specs
        if r.name != "changed_module_mypy"
    ]
    specs.extend(
        [
            CommandSpec(
                "strict_mypy",
                (str(ROOT / ".venv/bin/mypy"), "--strict", "--follow-imports=silent", *OWNED),
                "owned",
            ),
            CommandSpec(
                "coverage_json",
                (
                    str(ROOT / ".venv/bin/coverage"),
                    "json",
                    "--data-file=" + str(scratch / ".coverage"),
                    "-o",
                    str(scratch / "coverage.json"),
                ),
                "owned",
            ),
            CommandSpec(
                "consumer_and_board_mutations",
                (
                    str(ROOT / ".venv/bin/pytest"),
                    "-n",
                    "0",
                    "-o",
                    "addopts=",
                    "--no-cov",
                    "--basetemp=" + str(scratch / "consumers"),
                    "tests/python/test_hardware_sparse_8003.py",
                    "tests/python/test_primary_publication_7928.py",
                    "-q",
                ),
                "owned",
                600,
            ),
            CommandSpec(
                "module_spec_coverage",
                (str(ROOT / ".venv/bin/python"), "scripts/check_spec_coverage.py", *OWNED, *TESTS),
                "owned",
            ),
            CommandSpec(
                "full_pytest",
                (str(ROOT / ".venv/bin/pytest"), "tests/python", "-q"),
                "repository_health",
                900,
            ),
        ]
    )
    if (raw / "repository_health.json").is_file():
        specs = [r for r in specs if r.name != "full_pytest"]
    atomic_json(
        raw / "validation_manifest.json",
        dict(
            commands=[asdict(r) for r in specs],
            environment=dict(
                PYTHONUNBUFFERED="1",
                JAX_PLATFORMS="cpu",
                COVERAGE_FILE=str(scratch / ".coverage-health"),
            ),
            owned=OWNED,
        ),
    )
    return specs


def validate(specs: list[CommandSpec], raw: Path, scratch: Path) -> tuple[list[Json], Json]:
    """Actual exits and log bytes remain separate from unrelated health failures."""
    receipts = run_commands(
        ROOT,
        specs,
        log_dir=raw / "validation_logs" / canonical_hash([asdict(r) for r in specs]),
        heartbeat_s=30,
        extra_env=dict(
            PYTHONUNBUFFERED="1",
            JAX_PLATFORMS="cpu",
            COVERAGE_FILE=str(scratch / ".coverage-health"),
            OPENBLAS_NUM_THREADS="1",
        ),
    )
    for r in receipts:
        r.update(required=r["scope"] != "repository_health", log_path=str(ROOT / r["log_path"]))
    health_path = raw / "repository_health.json"
    health = [r for r in receipts if not r["required"]]
    if health:
        atomic_json(health_path, health)
    elif health_path.is_file():
        receipts.extend(
            dict(r, reused_diagnostic=True) for r in json.loads(health_path.read_bytes())
        )
    report = scratch / "coverage.json"
    counts = (
        {p: v["summary"] for p, v in json.loads(report.read_bytes())["files"].items()}
        if report.is_file()
        else {}
    )
    atomic_json(raw / "validation_receipts.json", receipts)
    atomic_json(raw / "coverage_statement_counts.json", counts)
    return receipts, counts


def terminal_check(candidate: Path) -> Json:
    """A fresh process reduces checkpoints before unchanged final-byte readers."""
    py = str(ROOT / ".venv/bin/python")
    specs = [
        CommandSpec(
            "cold_reduce",
            (py, "-u", str(ROOT / OWNED[-1]), "--cold-replay", str(candidate)),
            "terminal",
            60,
        ),
        CommandSpec(
            "adversarial",
            (py, "-u", "scripts/adversarial_verify.py", "--json", str(candidate)),
            "terminal",
            60,
        ),
        CommandSpec(
            "strict_rows",
            (py, "-u", "scripts/verdict_row_consistency_lint.py", "--strict", str(candidate)),
            "terminal",
            60,
        ),
    ]
    receipts = run_commands(
        ROOT,
        specs,
        log_dir=candidate.parent / "terminal_logs" / sha256_file(candidate).split(":")[1],
        heartbeat_s=30,
    )
    for r in receipts:
        r["log_path"] = str(ROOT / r["log_path"])
    flagged = json.loads(Path(receipts[1]["log_path"]).read_bytes())["flagged_count"]
    return dict(
        passed=not flagged and all(r["passed"] for r in receipts),
        candidate_sha256=sha256_file(candidate),
        receipts=receipts,
        flagged_adversarial=bool(flagged),
    )


def publish(output: Path, value: Json) -> None:
    """Expose only checked bytes and require both existing consumers to select them."""
    raw = output.parent / "raw" / output.stem
    value["terminal_validation_sidecar_path"] = str(raw / "terminal_validation.json")
    value["field_principles"] = {
        k: "Keep board custody, CPU controls, natural updates, missing costs and current calls separate."
        for k in value
    }
    receipt = publish_primary(output, normalize_artifact_for_template_write(value), terminal_check)
    atomic_json(raw / "terminal_validation.json", receipt)
    readers = reader_receipt(
        TASK,
        output.parent,
        field="hardware_evidence_ready_score",
        expected=value["hardware_evidence_ready_score"],
    )
    if not readers["passed"] or readers["gate_sha256"] != receipt["primary_sha256"]:
        raise ValueError("primary_resolution")
    atomic_json(raw / "reader_receipt.json", readers)
    final = terminal_check(output)
    atomic_json(raw / "published_validation.json", final)
    if not final["passed"] or final["candidate_sha256"] != receipt["primary_sha256"]:
        raise ValueError("published_validation")


def main(argv: list[str] | None = None) -> int:
    """Freeze method and source hashes before any evaluator label is exposed."""
    started = time.monotonic()
    spans: list[Json] = []

    def phase(name: str) -> None:
        elapsed = time.monotonic() - started
        if spans:
            spans[-1]["end_s"] = elapsed
        spans.append(dict(phase=name, start_s=elapsed))
        print(
            f"[exp8016] phase={name} elapsed_s={elapsed:.3f} model_loads=0 generations=0 device_calls=0",
            flush=True,
        )

    parser = argparse.ArgumentParser()
    parser.add_argument("--date", default="20261002")
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, default=ROOT / "results" / (NAME + ".json"))
    parser.add_argument("--fixture-input", type=Path)
    parser.add_argument("--validation-worker", action="store_true")
    parser.add_argument("--cold-replay", type=Path)
    args = parser.parse_args(argv)
    phase("start")
    try:
        if args.date != "20261002":
            raise ValueError("run_date")
        if args.cold_replay:
            h.replay(json.loads(args.cold_replay.read_bytes()))
            print("[exp8016] replay_passed", flush=True)
            return 0
        output = args.output.absolute()
        raw = output.parent / "raw" / NAME
        raw.mkdir(parents=True, exist_ok=True)
        scratch = Path(tempfile.mkdtemp(prefix="carnot-8016-", dir="/tmp"))
        phase("freeze_methods_roles_budgets_gates")
        helpers = [
            "python/carnot/reporting/hardware_sparse_8003.py",
            "python/carnot/verify/fixedpoint_sparse_8003.py",
            "python/carnot/verify/sparse_energy_7996.py",
            "python/carnot/verify/typed_development_7997.py",
            "python/carnot/reporting/current_work_receipt.py",
            "python/carnot/reporting/primary_publication.py",
            "python/carnot/reporting/experiment_7303_validation_scope.py",
            "scripts/experiment_template.py",
        ]
        refs = [
            dict(path=str(ROOT / p), sha256=sha256_file(ROOT / p)) for p in OWNED + TESTS + helpers
        ]
        role_hashes = {
            str(args.root / "results" / p): sha256_file(args.root / "results" / p)
            if (args.root / "results" / p).is_file()
            else None
            for p in h.UPSTREAM.values()
        }
        atomic_json(
            raw / "configuration.json",
            dict(
                config=h.CONFIG,
                code_hashes=refs,
                role_hashes=role_hashes,
                labels_exposed=False,
                inference_substrate_class="no_model_load",
            ),
        )
        specs = commands(raw, scratch)
        phase("authenticate_board_history_and_update_prerequisites")
        plan = (
            json.loads(args.fixture_input.read_bytes())
            if args.fixture_input
            else h.authenticate(args.root, raw / "custody")
        )
        for ref in plan["cited_upstream_artifacts"]:
            ref["imported_fields"] = ["exact_receipt_bytes_and_hash_bound_declared_fields"]
        atomic_json(raw / "replay_inputs.json", plan)
        phase("before_cpu_update_replay")
        value = h.reduce(plan)
        phase("after_cpu_update_replay")
        atomic_json(raw / "primitive_rows.json", dict(rows=value["rows"]))
        atomic_json(
            raw / "update_checkpoints.json",
            [r["checkpoint"] for r in value["cumulative_error_rows"]],
        )
        phase("owned_validation")
        receipts, counts = ([], {}) if args.validation_worker else validate(specs, raw, scratch)
        covered = set(counts) == set(OWNED) and all(
            r["num_statements"] > 0 and r["covered_lines"] == r["num_statements"]
            for r in counts.values()
        )
        if not args.validation_worker and (
            not covered or any(r["required"] and not r["passed"] for r in receipts)
        ):
            value.update(
                honest_verdict="complete_disqualified_owned_checks",
                verdict_class="disqualified",
                hardware_evidence_ready_score=0,
            )
        for ref in refs:
            if sha256_file(Path(ref["path"])) != ref["sha256"]:
                raise ValueError("frozen_code_drift")
        value.update(
            experiment_id=8016,
            task_id=TASK,
            milestone="2026.10.694",
            run_date=args.date,
            schema="carnot.hardware_update_boundary.v694.v1",
            claim_scope="Current authenticated board custody; optional CPU update replay. No current device execution, pretrained model load, new natural observations or purchase recommendation.",
            inference_substrate="verifier_ensemble_against_cached_candidates"
            if plan["trajectory"]
            else "aggregation_from_upstream_artifacts",
            inference_substrate_class="no_model_load",
            MODEL_SPECS=[],
            model_specs=[],
            model_invocation_counts=dict(ZERO_INVOCATION_COUNTS),
            random_seed=69416,
            methodology="Freeze role hashes, signed 24-bit Q12 storage, signed 32-bit Q12 sums, fit-only lookup tables and exact ordered updates. Compare float64 before/after every update and JSON state restart. Block absent trajectories. Device and acquisition costs remain distinct.",
            methodology_note="CPU fixtures are circular protocol evidence. No inference-only sweep or historical science retry substitutes for a natural trajectory.",
            replay_inputs=plan,
            config=h.CONFIG,
            validation_receipts=receipts,
            coverage_statement_counts=counts,
            code_config_hashes=refs,
            raw_shard_hashes=[
                dict(path=str(raw / p), sha256=sha256_file(raw / p))
                for p in (
                    "configuration.json",
                    "validation_manifest.json",
                    "replay_inputs.json",
                    "primitive_rows.json",
                    "update_checkpoints.json",
                )
            ]
            + [
                dict(path=str(p), sha256=sha256_file(p))
                for p in (raw / "development_failures").glob("*")
            ],
            checkpoints=dict(
                path=str(raw / "update_checkpoints.json"),
                sha256=sha256_file(raw / "update_checkpoints.json"),
            ),
            repository_health=[r for r in receipts if not r["required"]],
            validation_worker=args.validation_worker,
            flagged_adversarial=False,
        )
        value["reproducibility_checksum"] = canonical_hash(
            dict(config=h.CONFIG, inputs=plan, code=refs, rows=value["rows"])
        )
        phase("freeze_candidate")
        value["duration_s"] = time.monotonic() - started
        spans[-1]["end_s"] = value["duration_s"]
        value["phase_spans"] = spans
        publish(output, value)
        phase("published_and_rechecked")
        return 0
    except (OSError, ValueError, KeyError, TypeError, ZeroDivisionError) as error:
        print(f"[exp8016] terminal_error={error}", flush=True)
        return 1
