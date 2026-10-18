"""REQ-REPORT-8029: freeze and publish independent custody and workload decisions."""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path
import tempfile
import time
from typing import Any

from carnot.experiment_8016_v694_hardware_update_boundary import validate
from carnot.reporting import hardware_workload_8029 as h
from carnot.reporting.current_work_receipt import (
    ZERO_INVOCATION_COUNTS,
    atomic_json,
    canonical_hash,
    sha256_file,
)
from carnot.reporting.evidence_features_custody_7980 import checked, reference
from carnot.reporting.experiment_7303_validation_scope import (
    CommandSpec,
    build_scoped_commands,
    run_commands,
)
from carnot.reporting.primary_publication import publish_primary, reader_receipt
from carnot.reporting.publication_qualification_7928 import terminal_checks
from scripts.experiment_template import normalize_artifact_for_template_write

Json = dict[str, Any]
ROOT = h.ROOT
NAME = "experiment_8029_v695_hardware_workload_boundary"
TASK = "exp8029-hardware-workload-boundary"
MODEL_SPECS: list[str] = []
OWNED = [
    f"python/carnot/{NAME}.py",
    "python/carnot/reporting/hardware_workload_8029.py",
    f"scripts/experiments/{NAME}.py",
]
TEST = "tests/python/test_hardware_workload_8029.py"


def commands(scratch: Path) -> list[CommandSpec]:
    """Reuse bounded scoped checks and separate the one broad health diagnostic."""
    (scratch / "pytest").mkdir(parents=True, exist_ok=True)
    include = ",".join(str(ROOT / p) for p in OWNED)
    scoped = build_scoped_commands(
        ROOT,
        [TEST],
        OWNED[:-1],
        static_paths=[OWNED[-1]],
        basetemp=scratch / "pytest",
        coverage_file=scratch / ".coverage",
    )
    scoped = [
        replace(
            r,
            scope="owned",
            timeout_s=120,
            argv=tuple("--include=" + include if a.startswith("--include=") else a for a in r.argv),
        )
        for r in scoped
        if r.name != "changed_module_mypy"
    ]
    py = str(ROOT / ".venv/bin/python")
    return scoped + [
        CommandSpec(
            "strict_mypy",
            (str(ROOT / ".venv/bin/mypy"), "--strict", "--follow-imports=silent", *OWNED),
            "owned",
            120,
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
            60,
        ),
        CommandSpec(
            "consumer_board_mutations",
            (
                str(ROOT / ".venv/bin/pytest"),
                "-n",
                "0",
                "-o",
                "addopts=",
                "--no-cov",
                "--basetemp=" + str(scratch / "consumers"),
                "tests/python/test_hardware_update_8016.py",
                "tests/python/test_primary_publication_7928.py",
                "-q",
            ),
            "owned",
            120,
        ),
        CommandSpec(
            "full_pytest",
            (str(ROOT / ".venv/bin/pytest"), "tests/python", "-q"),
            "repository_health",
            120,
        ),
        CommandSpec(
            "repository_spec_health",
            (py, "scripts/check_spec_coverage.py"),
            "repository_health",
            60,
        ),
    ]


def replay(path: Path) -> Json:
    """Cold readers recompute metrics from durable inputs and verify every state hash."""
    value = json.loads(path.read_bytes())
    for ref in (
        value["raw_shard_hashes"] + value["code_config_hashes"] + value["cited_upstream_artifacts"]
    ):
        checked(ref)
    plan = json.loads(checked(value["replay_input_reference"]).read_bytes())
    fresh = h.reduce(plan)
    for ref, row in zip(
        value["checkpoint_references"], fresh["cumulative_error_rows"], strict=True
    ):
        if json.loads(checked(ref).read_bytes()) != row["checkpoint"]:
            raise ValueError("restart_checkpoint_drift")
    for key, expected in fresh.items():
        if (
            key in {"hardware_custody_ready_score", "honest_verdict", "verdict_class"}
            and value["verdict_class"] == "disqualified"
        ):
            continue
        if value[key] != expected:
            raise ValueError("reduction_drift:" + key)
    for receipt in value["validation_receipts"]:
        checked(dict(path=receipt["log_path"], sha256=receipt["log_sha256"]))
    return dict(passed=True, rows_checksum=canonical_hash(fresh["rows"]))


def terminal(path: Path) -> Json:
    """Use existing adversarial and row readers after a fresh-process raw reduction."""
    raw = Path(json.loads(path.read_bytes())["raw_directory"])
    logs = raw / "cold_logs" / sha256_file(path).split(":")[-1]
    cold = run_commands(
        ROOT,
        [
            CommandSpec(
                "cold_reduce",
                (
                    str(ROOT / ".venv/bin/python"),
                    "-u",
                    str(ROOT / OWNED[-1]),
                    "--cold-replay",
                    str(path),
                ),
                "terminal",
                120,
            )
        ],
        log_dir=logs,
        heartbeat_s=30,
    )
    report = terminal_checks(path, raw)
    report["receipts"] += cold
    report["passed"] = (
        report["passed"] and not report["flagged_adversarial"] and all(r["passed"] for r in cold)
    )
    return report


def main(argv: list[str] | None = None) -> int:
    """Freeze gates before measurement and publish only exact checked candidate bytes."""
    started = time.monotonic()
    spans: list[Json] = []

    def phase(name: str) -> None:
        elapsed = time.monotonic() - started
        if spans:
            spans[-1]["end_s"] = elapsed
        spans.append(dict(phase=name, start_s=elapsed))
        print(
            f"[exp8029] phase={name} elapsed_s={elapsed:.3f} completed={len(spans)} "
            "pending=current_phase model_loads=0 generations=0 device_calls=0",
            flush=True,
        )

    parser = argparse.ArgumentParser()
    parser.add_argument("--date", default="20261002")
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--fixture-input", type=Path)
    parser.add_argument("--validation-worker", action="store_true")
    parser.add_argument("--cold-replay", type=Path)
    args = parser.parse_args(argv)
    phase("start")
    try:
        if args.date != "20261002":
            raise ValueError("run_date")
        if args.cold_replay:
            print(json.dumps(replay(args.cold_replay)), flush=True)
            return 0
        output = (args.output or args.root / "results" / (NAME + ".json")).absolute()
        raw = output.parent / "raw" / NAME
        raw.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="carnot-8029-", dir="/tmp") as temporary:
            scratch = Path(temporary)
            phase("freeze_methods_source_roles_budgets_acceptance")
            code = [
                reference(ROOT / p)
                for p in OWNED
                + [
                    TEST,
                    "python/carnot/verify/causal_online_8025.py",
                    "python/carnot/verify/fixedpoint_sparse_8003.py",
                    "python/carnot/reporting/hardware_update_8016.py",
                    "python/carnot/reporting/current_work_receipt.py",
                    "python/carnot/reporting/primary_publication.py",
                    "scripts/experiment_template.py",
                ]
            ]
            specs = [
                r
                for r in commands(scratch)
                if r.scope != "repository_health" or not (raw / "repository_health.json").is_file()
            ]
            atomic_json(
                raw / "methods.json",
                dict(
                    config=h.CONFIG,
                    code=code,
                    sources=h.SOURCES,
                    source_hashes={
                        str(args.root / "results" / (stem + ".json")): sha256_file(
                            args.root / "results" / (stem + ".json")
                        )
                        if (args.root / "results" / (stem + ".json")).is_file()
                        else None
                        for stem, _ in h.SOURCES.values()
                    },
                    commands=[asdict(r) for r in specs],
                    artifact_guard_enabled=True,
                    roles="authenticated history; exposed-development update replay; independent current costs",
                    exclusions="missing, blocked or disqualified current sources; unmatched operations",
                    acceptance="custody independent; drift<=0.001; zero disagreement/overflow; identical restart",
                ),
            )
            phase("authenticate_original_receipts_without_devices")
            plan = (
                json.loads(args.fixture_input.read_bytes())
                if args.fixture_input
                else h.load(args.root, raw)
            )
            atomic_json(raw / "replay_inputs.json", plan)
            phase("before_cpu_quantized_update_replay")
            value = h.reduce(plan)
            phase("after_cpu_quantized_update_replay")
            checkpoints = []
            for index, row in enumerate(value["cumulative_error_rows"]):
                path = raw / "checkpoints" / f"{index:05d}.json"
                atomic_json(path, row["checkpoint"])
                checkpoints.append(reference(path))
                if index % 256 == 0:
                    print(
                        f"[exp8029] checkpoint elapsed_s={time.monotonic() - started:.3f} "
                        f"completed={index + 1} pending={len(value['cumulative_error_rows']) - index - 1}",
                        flush=True,
                    )
            atomic_json(raw / "primitive_rows.json", dict(rows=value["rows"]))
            phase("before_owned_validation_subprocesses")
            receipts, counts = ([], {}) if args.validation_worker else validate(specs, raw, scratch)
            phase("after_owned_validation_subprocesses")
            covered = set(counts) == set(OWNED) and all(
                r["missing_lines"] == 0 and r["num_statements"] > 0 for r in counts.values()
            )
            good = covered and all(r["passed"] for r in receipts if r["required"])
            if not args.validation_worker and not good:
                value.update(
                    honest_verdict="complete_disqualified_owned_checks",
                    verdict_class="disqualified",
                    hardware_custody_ready_score=0,
                )
            for ref in code:
                checked(ref)
            value.update(
                experiment_id=8029,
                task_id=TASK,
                milestone="2026.10.695",
                run_date=args.date,
                schema="carnot.hardware_workload_boundary.v695.v1",
                config=h.CONFIG,
                claim_scope="Current read-only custody and CPU quantized replay of exposed development. No current device execution, model load, natural deployment benefit or vendor speedup.",
                methodology="Freeze exact source roles and gates; authenticate original board bytes; replay all current sparse updates in Q12 signed24/acc32 against float64; keep independent cost gates and cold-reduce durable evidence.",
                inference_substrate="aggregation_from_upstream_artifacts",
                inference_substrate_class="no_model_load",
                MODEL_SPECS=[],
                model_specs=[],
                model_invocation_counts=dict(ZERO_INVOCATION_COUNTS),
                trained_head_specs=[
                    dict(
                        parameter_count=len(plan["trajectory"]["head"]["parameters"]),
                        pretrained=False,
                        current_fitting=False,
                    )
                ]
                if plan["trajectory"]
                else [],
                random_seed=h.CONFIG["seed"],
                raw_directory=str(raw),
                checkpoint_references=checkpoints,
                replay_input_reference=reference(raw / "replay_inputs.json"),
                raw_shard_hashes=[
                    reference(p)
                    for p in [
                        raw / "methods.json",
                        raw / "replay_inputs.json",
                        raw / "primitive_rows.json",
                    ]
                ]
                + checkpoints
                + [reference(p) for p in (raw / "development_failures").glob("*")],
                code_config_hashes=code,
                validation_receipts=receipts,
                coverage_statement_counts=counts,
                owned_checks_passed=good,
                repository_health=[r for r in receipts if not r["required"]],
                flagged_adversarial=False,
                terminal_validation_sidecar_path=str(raw / "terminal_validation.json"),
            )
            value["reproducibility_checksum"] = canonical_hash(
                dict(config=h.CONFIG, code=code, inputs=plan)
            )
            phase("freeze_candidate")
            value["duration_s"] = time.monotonic() - started
            spans[-1]["end_s"] = value["duration_s"]
            value["phase_spans"] = spans
            value["field_principles"] = {
                k: "Retain "
                + k
                + " to separate custody, numerical loss, current costs and device evidence."
                for k in value
            }
            publication = publish_primary(
                output, normalize_artifact_for_template_write(value), terminal
            )
            atomic_json(raw / "terminal_validation.json", publication)
            final = terminal(output)
            readers = reader_receipt(
                TASK,
                output.parent,
                field="hardware_custody_ready_score",
                expected=value["hardware_custody_ready_score"],
            )
            atomic_json(
                raw / "published_readers.json",
                dict(publication=publication, final=final, readers=readers),
            )
            if (
                not final["passed"]
                or not readers["passed"]
                or readers["gate_sha256"] != publication["primary_sha256"]
            ):
                raise ValueError("published_reader_validation")
            phase("published_and_rechecked")
        return 0
    except (OSError, ValueError, KeyError, TypeError, ZeroDivisionError, StopIteration) as error:
        print(f"[exp8029] terminal_error={error}", flush=True)
        return 1
