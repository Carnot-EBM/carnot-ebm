"""REQ-REPORT-8217: freeze owned validation and publish a complete terminal capstone.

Private scratch stays alive around every child. Separate repository health never
repairs a failed owned check or grants deployment and generalization credit.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import tempfile
import time
from typing import Any
from unittest.mock import patch

from carnot.reporting import v709_capstone_inputs as e
from carnot.reporting import v709_capstone_science as s
from carnot.reporting import v709_execution as x
from carnot.reporting import v709_runner as qualified
from carnot.reporting.current_work_receipt import (
    ZERO_INVOCATION_COUNTS,
    atomic_json,
    canonical_hash,
    sha256_file,
)
from carnot.reporting.primary_publication import publish_primary
from scripts.experiment_template import normalize_artifact_for_template_write

Json = dict[str, Any]
ROOT = e.ROOT
NAME = "experiment_8217_v709_capstone"
CLI = f"scripts/experiments/{NAME}.py"
TEST = "tests/python/test_v709_capstone_8217.py"
OWNED = [
    "python/carnot/reporting/v709_capstone.py",
    "python/carnot/reporting/v709_capstone_inputs.py",
    "python/carnot/reporting/v709_capstone_science.py",
    CLI,
]
REUSED = [
    "python/carnot/reporting/v709_execution.py",
    "python/carnot/reporting/v709_qualification.py",
    "python/carnot/reporting/v709_runner.py",
    "python/carnot/reporting/primary_publication.py",
    "scripts/experiment_template.py",
    "python/carnot/verify/restricted_decision_rule_8210.py",
    "python/carnot/verify/memory_benefit_audit_8212.py",
    "python/carnot/verify/prospective_service_8214.py",
    "python/carnot/reporting/arc_authoritative_frontier_8215.py",
    "python/carnot/reporting/hardware_workload_obligations_8216.py",
]


def commands(private: Path) -> list[Json]:
    """Reuse qualified commands with coverage subprocess hooks for actual CLI statements."""
    with (
        patch.object(qualified, "TEST", TEST),
        patch.object(qualified, "OWNED", OWNED),
        patch.object(qualified, "CLI", CLI),
    ):
        plan = qualified.commands(private)
    config = private / "coverage.ini"
    config.write_text(
        config.read_text().replace("parallel=true", "parallel=true\npatch=subprocess")
        + "[report]\nexclude_lines=\n"
    )
    tests = [
        "test_source_boundary_7852.py",
        "test_experiment_7942_v689_sentence_labels.py",
        "test_v709_qualification_8205.py",
        "test_restricted_decision_audit_8210.py",
        "test_memory_benefit_audit_8212.py",
        "test_prospective_service_measurement_8214.py",
        "test_arc_authoritative_frontier_8215.py",
        "test_hardware_workload_obligations_8216.py",
    ]
    plan.append(
        dict(
            name="affected_reducers_E2E015_019",
            argv=[
                str(ROOT / ".venv/bin/pytest"),
                "-n",
                "0",
                "-o",
                "addopts=",
                "--no-cov",
                "-q",
                *["tests/python/" + t for t in tests],
                "--basetemp=" + str(private / "reducers"),
            ],
            deadline=600,
            expected=0,
            scope="owned",
        )
    )
    plan.append(
        dict(
            name="publication_gate",
            argv=[str(ROOT / ".venv/bin/python"), "scripts/publication_gate.py", "--json"],
            deadline=90,
            expected=0,
            scope="publication",
        )
    )
    return plan


def qualify(value: Json, passed: bool) -> None:
    """An owned failure must disqualify current execution even when science is blocked."""
    value.update(required_checks_passed=passed, capstone_execution_ready_score=int(passed))
    value["acceptance_gates"]["owned_validation"]["passed"] = passed
    row = value["task_dispositions"][-1]
    row.update(qualified=passed, eligible=passed, excluded=not passed, numerator=int(passed))
    if not passed:
        value.update(
            honest_verdict="complete_disqualified_owned_validation", verdict_class="disqualified"
        )
        row.update(
            honest_verdict=value["honest_verdict"], verdict_class="disqualified", failed=True
        )
    value["failed_count"] = sum(r["failed"] for r in value["rows"])
    value["excluded_count"] = sum(r["excluded"] for r in value["rows"])
    value["eligible_count"] = sum(r["eligible"] for r in value["rows"])


def replay(path: Path) -> Json:
    """Rehash complete stream custody and recompute headlines from frozen primitives."""
    value = json.loads(path.read_bytes())
    for ref in [
        value["replay_input_reference"],
        value["primitive_reference"],
        *value["source_artifact_hashes"],
        *value["raw_shard_hashes"],
        *value["code_config_hashes"],
    ]:
        named = Path(ref.get("snapshot_path", ref["path"]))
        if sha256_file(named) != ref["sha256"]:
            raise ValueError("evidence_hash_drift:" + str(named))
    for receipt in value["validation_receipts"]:
        for stream in ("stdout", "stderr"):
            if sha256_file(Path(receipt[stream + "_path"])) != receipt[stream + "_sha256"]:
                raise ValueError("validation_stream_drift")
    data = json.loads(Path(value["replay_input_reference"]["path"]).read_bytes())
    branches = json.loads(Path(value["primitive_reference"]["path"]).read_bytes())
    fresh = s.reduce(data, branches)
    qualify(fresh, value["required_checks_passed"])
    for key, observed in fresh.items():
        if value[key] != observed:
            raise ValueError("primitive_reduction_drift:" + key)
    return dict(passed=True, rows_checksum=canonical_hash(fresh["rows"]))


def terminal_plan(path: Path) -> list[Json]:
    """Freeze the unchanged terminal auditors and the actual cold CLI operands."""
    with patch.object(qualified, "CLI", CLI):
        return list(qualified.terminal_plan(path))


def main(argv: list[str] | None = None) -> int:
    """Preserve literal child evidence and expose only a validated candidate's bytes."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", choices=["20261006"], default="20261006")
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--private-fixture", action="store_true")
    parser.add_argument("--cold-replay", type=Path)
    args = parser.parse_args(argv)
    e.progress("start")
    try:
        if args.cold_replay:
            print(json.dumps(replay(args.cold_replay)), flush=True)
            return 0
        output = (args.output or ROOT / "results" / (NAME + ".json")).absolute()
        if args.private_fixture and (output.is_relative_to(ROOT / "results") or args.root == ROOT):
            raise ValueError("private_fixture_requires_private_root_and_output")
        raw = output.parent / "raw" / NAME / "invocations" / str(time.time_ns())
        raw.mkdir(parents=True, exist_ok=True)
        start, wall = time.monotonic_ns(), time.time_ns()
        with tempfile.TemporaryDirectory(prefix="carnot8217-", dir="/tmp") as directory:
            private = Path(directory)
            probe = private / "write_probe"
            probe.write_bytes(b"actual private writable scratch")
            runtime = dict(
                python=os.sys.version,
                executable=os.sys.executable,
                private_mode=oct(private.stat().st_mode & 0o777),
                private_writable=probe.read_bytes() == b"actual private writable scratch",
            )
            plan = [] if args.private_fixture else commands(private)
            controls = x.pytest_plan(private / "controls")
            terminal_specs = terminal_plan(
                output.parent / "raw" / output.stem / "terminal_candidate.json"
            )
            preflight = qualified.precondition_command()
            named_inputs = [
                str(args.root / p)
                for p in [
                    *e.q.INPUTS,
                    "results/experiment_8205_v709_contract_consumer_qualification.json",
                    "ops/north-star.md",
                    "research-roadmap-next.yaml",
                ]
            ]
            preflight["argv"][-1] = preflight["argv"][-1].replace(
                "raise SystemExit(int",
                'print(json.dumps({"named_input_paths":{n:Path(n).is_file() for n in '
                + repr(named_inputs)
                + "}})); raise SystemExit(int",
            )
            atomic_json(
                raw / "validation_manifest.json",
                dict(
                    commands=plan,
                    controls=controls,
                    terminal_commands=terminal_specs,
                    precondition=preflight,
                    runtime=runtime,
                    frozen_before_measurement_ns=time.monotonic_ns(),
                ),
            )
            precondition = x.execute([preflight], raw / "preconditions")[0]
            e.progress("preconditions_before")
            data = e.load(args.root, raw)
            if not precondition["passed"]:
                data["failures"].append(
                    e.operand(
                        precondition["stderr_path"],
                        "python_environment_exit",
                        0,
                        precondition["exit_code"],
                    )
                )
                for row in data["dispositions"]:
                    row["eligible"] = False
            e.progress("preconditions_after", 12, 0)
            branches = s.measure(data, raw / "science")
            atomic_json(raw / "replay_inputs.json", data)
            atomic_json(raw / "branch_primitives.json", branches)
            value = s.reduce(data, branches)
            e.progress("validation_before", 0, len(plan) + 2)
            receipts = x.execute(controls, raw / "control_logs")
            receipts += x.execute(
                [p for p in plan if p["scope"] == "owned"], raw / "validation_logs"
            )
            qualify(value, bool(receipts) and all(r["passed"] for r in receipts))
            health = x.execute(
                [p for p in plan if p["scope"] == "repository_health"], raw / "health_logs"
            )
            gates = x.execute(
                [
                    dict(
                        name="publication_gate",
                        argv=[
                            str(ROOT / ".venv/bin/python"),
                            "scripts/publication_gate.py",
                            "--json",
                        ],
                        deadline=90,
                        expected=0,
                        scope="publication",
                    )
                ],
                raw / "publication_logs",
            )[0]
            publication_gates = json.loads(Path(gates["stdout_path"]).read_bytes())
            coverage_path = private / "coverage.json"
            coverage = json.loads(coverage_path.read_bytes()) if coverage_path.is_file() else {}
            atomic_json(raw / "coverage.json", coverage)
            value.update(
                experiment_id=8217,
                task_id=e.TASK,
                milestone=e.q.MILESTONE,
                run_date=args.date,
                schema="carnot.v709.capstone.v1",
                random_seed=7098217,
                duration_s=(time.monotonic_ns() - start) / 1e9,
                inference_substrate="aggregation_from_upstream_artifacts",
                inference_substrate_class="no_model_load",
                MODEL_SPECS=[],
                model_invocation_counts=dict(ZERO_INVOCATION_COUNTS),
                call_ledger=[],
                trained_head_specs=[],
                preconditions_checked=data["preconditions"],
                runtime_preconditions=runtime,
                measurement_clocks=dict(
                    started_monotonic_ns=start,
                    ended_monotonic_ns=time.monotonic_ns(),
                    started_wall_ns=wall,
                ),
                validation_receipts=receipts,
                precondition_receipts=[precondition],
                repository_health=dict(owned=False, receipts=health),
                coverage_statement_counts=coverage.get("files", {}),
                coverage_totals=coverage.get("totals", {}),
                source_artifact_hashes=[*data["references"], *branches["references"]],
                code_config_hashes=[
                    dict(path=str(ROOT / p), sha256=sha256_file(ROOT / p))
                    for p in [*OWNED, TEST, *REUSED]
                ],
                replay_input_reference=dict(
                    path=str(raw / "replay_inputs.json"),
                    sha256=sha256_file(raw / "replay_inputs.json"),
                ),
                primitive_reference=dict(
                    path=str(raw / "branch_primitives.json"),
                    sha256=sha256_file(raw / "branch_primitives.json"),
                ),
                raw_shard_hashes=[
                    dict(path=str(p), sha256=sha256_file(p))
                    for p in sorted(raw.rglob("*"))
                    if p.is_file()
                ],
                terminal_validation_sidecar_path=str(raw / "terminal_validation.json"),
                paper_ready=publication_gates["paper_ready"],
                external_publication_authorized=False,
                publication_gate_results=publication_gates,
                unmet_gates=publication_gates["unmet_gates"],
                cited_upstream_artifacts=[
                    dict(
                        experiment_id=r["experiment_id"],
                        path=r["path"],
                        sha256=r["sha256"],
                        fields_imported=list(data["primaries"].get(r["task_id"], {})),
                    )
                    for r in data["dispositions"]
                ],
                methodology_note="Exact thirteen-task accounting and independent cached primitive reductions; no current model load, board operation or independent deployment measurement.",
            )
            value["reproducibility_checksum"] = canonical_hash(
                [
                    value["replay_input_reference"],
                    value["primitive_reference"],
                    value["code_config_hashes"],
                ]
            )
            value["field_principles"] = {
                k: "Preserve exact invocation, source bytes, denominators and scope; owned validity cannot establish independent science."
                for k in value
            }
            value = normalize_artifact_for_template_write(value)
            e.progress("publication_before", 13, 1)

            def validate(candidate: Path) -> Json:
                checks = x.execute(terminal_specs, raw / "terminal_logs")
                return dict(passed=all(r["passed"] for r in checks), checks=checks)

            publication = publish_primary(output, value, validate)
            atomic_json(
                raw / "terminal_validation.json",
                dict(
                    publication=publication, required_checks_passed=value["required_checks_passed"]
                ),
            )
        e.progress("complete", 13, 0)
        return 0
    except (OSError, ValueError, KeyError, TypeError, StopIteration) as error:
        print(json.dumps(dict(passed=False, error=str(error))), flush=True)
        return 1
