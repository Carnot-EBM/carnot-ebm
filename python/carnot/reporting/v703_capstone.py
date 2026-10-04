"""REQ-REPORT-8135: publish a checked terminal capstone without loading a model.

The worker repeats the reduction in a fresh interpreter. Validators must exit
normally before the only reader-visible primary is atomically replaced.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import shutil
import tempfile
import time
from typing import Any

from carnot.reporting import v703_capstone_inputs as inputs
from carnot.reporting import v703_capstone_reduction as reduction
from carnot.reporting.current_work_receipt import (
    ZERO_INVOCATION_COUNTS,
    atomic_json,
    canonical_hash,
)
from carnot.reporting.experiment_7303_validation_scope import (
    CommandSpec,
    build_scoped_commands,
    run_commands,
)
from carnot.reporting.primary_publication import publish_primary

Json = dict[str, Any]
ROOT = inputs.ROOT
NAME = "experiment_8135_v703_capstone"
CLI = f"scripts/experiments/{NAME}.py"
TEST = "tests/python/test_v703_capstone_8135.py"
OWNED = [
    "python/carnot/reporting/v703_capstone.py",
    "python/carnot/reporting/v703_capstone_inputs.py",
    "python/carnot/reporting/v703_capstone_reduction.py",
    CLI,
]
DEPENDENCIES = [
    "python/carnot/reporting/v701_capstone_reduction.py",
    "python/carnot/reporting/v699_capstone_reduction.py",
    "python/carnot/reporting/hardware_service_8134.py",
    "python/carnot/experiment_8132_v703_service_cost.py",
    "python/carnot/reporting/v685_authority_lifecycle.py",
    "python/carnot/reporting/experiment_7303_validation_scope.py",
    "python/carnot/reporting/primary_publication.py",
]
reference = inputs.reference
progress = inputs.progress


def commands(scratch: Path) -> list[CommandSpec]:
    """Freeze explicit validation and cover only statements introduced by this task."""
    (scratch / "pytest").mkdir(parents=True, exist_ok=True)
    py = str(ROOT / ".venv/bin/python")
    plan = build_scoped_commands(
        ROOT,
        [TEST],
        OWNED[:-1],
        static_paths=[CLI],
        basetemp=scratch / "pytest",
        coverage_file=scratch / ".coverage",
    )
    config = scratch / "coverage.ini"
    config.write_text(
        "[run]\ndata_file="
        + str(scratch / ".coverage")
        + "\ninclude=\n    "
        + "\n    ".join(str(ROOT / p) for p in OWNED)
        + "\n"
    )
    for index, spec in enumerate(plan):
        if spec.name == "changed_module_coverage":
            plan[index] = CommandSpec(
                spec.name,
                (
                    str(ROOT / ".venv/bin/coverage"),
                    "run",
                    "--rcfile=" + str(config),
                    "-m",
                    "pytest",
                    "-n",
                    "0",
                    "-o",
                    "addopts=",
                    "--no-cov",
                    "-q",
                    TEST,
                    "--basetemp=" + str(scratch / "covered"),
                ),
                "owned",
                120,
            )
        elif spec.name == "changed_module_coverage_report":
            plan[index] = CommandSpec(
                spec.name,
                (
                    str(ROOT / ".venv/bin/coverage"),
                    "json",
                    "--rcfile=" + str(config),
                    "--fail-under=100",
                    "-o",
                    str(scratch / "coverage.json"),
                ),
                "owned",
                30,
            )
        elif spec.name == "changed_module_mypy":
            plan[index] = CommandSpec(
                spec.name,
                (str(ROOT / ".venv/bin/mypy"), "--strict", "--follow-imports=skip", *OWNED[:-1]),
                "owned",
                60,
            )
        else:
            plan[index] = CommandSpec(spec.name, spec.argv, "owned", 120)
    consumers = [
        "tests/python/test_primary_publication_7928.py",
        "tests/python/test_conductor_gates.py",
        "tests/python/test_experiment_7891_v685_authority_lifecycle.py",
    ]
    plan.append(
        CommandSpec(
            "consumers_E2E018",
            (
                str(ROOT / ".venv/bin/pytest"),
                "-n",
                "0",
                "-o",
                "addopts=",
                "--no-cov",
                "-q",
                *consumers,
                "--basetemp=" + str(scratch / "consumers"),
            ),
            "owned",
            180,
        )
    )
    plan.append(
        CommandSpec(
            "publication_gate", (py, "scripts/publication_gate.py", "--json"), "publication", 60
        )
    )
    paths = sorted(
        {
            p
            for n in [
                8122,
                8123,
                8132,
                8133,
                8134,
                8110,
                8083,
                8112,
                8084,
                8099,
                8113,
                8074,
                8114,
                8115,
                8116,
                8119,
                8103,
                8117,
                8106,
                8120,
                7988,
                8068,
            ]
            for p in (ROOT / "results").glob(f"experiment_{n}_*.json")
        }
    )
    plan.append(
        CommandSpec(
            "upstream_summaries",
            (py, "-u", "scripts/summarize_artifact.py", *map(str, paths)),
            "upstream_inventory",
            90,
        )
    )
    return plan


def execute(plan: list[CommandSpec], raw: Path) -> list[Json]:
    """Capture exact argv, normal exit, duration and log hashes with ten-second heartbeats."""
    receipts = run_commands(
        ROOT,
        plan,
        log_dir=raw,
        heartbeat_s=10,
        extra_env=dict(PYTHONUNBUFFERED="1", JAX_PLATFORMS="cpu"),
    )
    return [
        dict(
            r,
            log_path=str((ROOT / r["log_path"]).resolve()),
            argv=r["command_argv"],
            expected_exit=0,
            actual_exit=r["exit_code"],
            normal_exit=r["exit_code"] >= 0 and not r["timed_out"],
        )
        for r in receipts
    ]


def replay(path: Path) -> Json:
    """Rehash saved bytes and recompute primitive equations and every scientific claim."""
    value = json.loads(path.read_bytes())
    for ref in [
        value["replay_input_reference"],
        *value["source_artifact_hashes"],
        *value["code_config_hashes"],
        *value["raw_shard_hashes"],
    ]:
        if reference(Path(ref["path"]))["sha256"] != ref["sha256"]:
            raise ValueError("input_hash_drift:" + ref["path"])
    for receipt in value["validation_receipts"]:
        if reference(Path(receipt["log_path"]))["sha256"] != receipt["log_sha256"]:
            raise ValueError("validation_log_hash_drift")
    data = json.loads(Path(value["replay_input_reference"]["path"]).read_bytes())
    for index, task in enumerate(data["tasks"][:-1]):
        audit = data["independent_reductions"].get(task["id"], {})
        if audit.get("available"):
            operand = (
                dict(audit["result"], replay_input_reference=audit["primitive_reference"])
                if audit.get("kind") == "hardware"
                else dict(primitive_rows=audit["primitive_reference"], reduction=audit["result"])
            )
            fresh = inputs.primitive_audit(
                operand,
                8123 + index,
            )
            if fresh != audit:
                raise ValueError("primitive_reduction_drift")
    fresh = reduction.reduce(data)
    reduction.qualify(fresh, value["required_checks_passed"])
    for key, expected in fresh.items():
        if value[key] != expected:
            raise ValueError("reduction_drift:" + key)
    return dict(passed=True, rows_checksum=canonical_hash(fresh["rows"]))


def terminal(path: Path) -> Json:
    """Use unchanged validators; a timeout or abnormal exit never counts as a pass."""
    py = str(ROOT / ".venv/bin/python")
    plan = [
        CommandSpec(name, argv, "terminal", 90)
        for name, argv in [
            ("cold_replay", (py, "-u", str(ROOT / CLI), "--cold-replay", str(path))),
            ("adversarial", (py, str(ROOT / "scripts/adversarial_verify.py"), "--json", str(path))),
            (
                "strict_rows",
                (py, str(ROOT / "scripts/verdict_row_consistency_lint.py"), "--strict", str(path)),
            ),
        ]
    ]
    receipts = execute(plan, path.parent / "terminal_logs" / str(time.time_ns()))
    return dict(passed=all(r["passed"] and r["normal_exit"] for r in receipts), receipts=receipts)


def main(argv: list[str] | None = None) -> int:
    """Finish external blocks once while preserving failed owned candidate bytes."""
    os.environ["PYTHONUNBUFFERED"] = "1"
    began = time.monotonic()
    progress("start")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", choices=["20261004"], default="20261004")
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--worker-input", type=Path)
    parser.add_argument("--cold-replay", type=Path)
    parser.add_argument("--fixture-e2e", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.cold_replay:
            print(json.dumps(replay(args.cold_replay)), flush=True)
            return 0
        output = (args.output or args.root / "results" / (NAME + ".json")).absolute()
        if args.worker_input and output.parent == ROOT / "results":
            raise ValueError("worker_cannot_publish_primary")
        raw = output.parent / "raw" / NAME / "invocations" / str(time.time_ns())
        raw.mkdir(parents=True, exist_ok=True)
        if output.is_file():
            shutil.copyfile(output, raw / "previous_primary.json")
        if args.worker_input:
            atomic_json(output, reduction.reduce(json.loads(args.worker_input.read_bytes())))
            progress("worker_complete", 13, 0)
            return 0
        with tempfile.TemporaryDirectory(prefix="carnot8135-", dir="/tmp") as temporary:
            scratch = Path(temporary)
            plan = commands(scratch)
            if args.fixture_e2e:
                if args.root.resolve() == ROOT or output.is_relative_to(ROOT / "results"):
                    raise ValueError("fixture_requires_private_paths")
                plan = [
                    CommandSpec(
                        "fixture_consumers",
                        (
                            str(ROOT / ".venv/bin/pytest"),
                            "-n",
                            "0",
                            "-o",
                            "addopts=",
                            "--no-cov",
                            "-q",
                            "tests/python/test_primary_publication_7928.py",
                            "--basetemp=" + str(scratch / "fixture_consumers"),
                        ),
                        "owned",
                        60,
                    ),
                    *[s for s in plan if s.scope == "publication"],
                ]
            atomic_json(
                raw / "validation_manifest.json",
                dict(commands=[asdict(s) for s in plan], owned=OWNED),
            )
            summaries = execute(
                [s for s in plan if s.scope == "upstream_inventory"], raw / "summary_logs"
            )
            progress("before_evidence_load")
            data = inputs.load(args.root, raw)
            if args.fixture_e2e:
                data["verifier_is_oracle"] = True
            atomic_json(raw / "replay_inputs.json", data)
            progress("after_evidence_load_before_reduction", 12, 1)
            value = reduction.reduce(data)
            measurement = execute(
                [
                    CommandSpec(
                        "independent_reduction",
                        (
                            str(ROOT / ".venv/bin/python"),
                            "-u",
                            str(ROOT / CLI),
                            "--worker-input",
                            str(raw / "replay_inputs.json"),
                            "--output",
                            str(raw / "worker.json"),
                        ),
                        "measurement",
                        90,
                    )
                ],
                raw / "measurement_logs",
            )
            child = json.loads((raw / "worker.json").read_bytes())
            if not measurement[0]["passed"] or child != value:
                raise ValueError("independent_reduction_child_failed")
            atomic_json(raw / "independent_reduction.json", child)
            progress("after_reduction_before_validation", 13, len(plan) - 1)
            receipts = execute(
                [s for s in plan if s.scope != "upstream_inventory"], raw / "validation_logs"
            )
            owned = [r for r in receipts if r["scope"] == "owned"]
            passed = bool(owned) and all(r["passed"] and r["normal_exit"] for r in owned)
            reduction.qualify(value, passed)
            publication = next(r for r in receipts if r["name"] == "publication_gate")
            gates = json.loads(Path(publication["log_path"]).read_bytes())
            coverage_path = scratch / "coverage.json"
            coverage = json.loads(coverage_path.read_bytes()) if coverage_path.is_file() else {}
            atomic_json(raw / "coverage.json", coverage)
            atomic_json(raw / "primitive_rows.json", dict(rows=value["rows"]))
            duration = time.monotonic() - began
            value.update(
                experiment_id=8135,
                task_id="exp8135-capstone",
                run_date=args.date,
                milestone="2026.10.703",
                schema="carnot.v703.capstone.v1",
                random_seed=703,
                inference_substrate="aggregation_from_upstream_artifacts",
                inference_substrate_class="no_model_load",
                MODEL_SPECS=[],
                model_invocation_counts=ZERO_INVOCATION_COUNTS,
                call_ledger=[],
                trained_head_specs=[
                    dict(upstream=t, specs=p["trained_head_specs"])
                    for t, p in data["primaries"].items()
                    if p.get("trained_head_specs")
                ],
                cited_upstream_artifacts=[
                    dict(
                        task_id=t,
                        MODEL_SPECS=p.get("MODEL_SPECS", []),
                        model_invocation_counts=p.get("model_invocation_counts", {}),
                        cited_provenance=p.get("cited_upstream_artifacts", []),
                    )
                    for t, p in data["primaries"].items()
                ],
                duration_s=duration,
                phase_spans=[
                    dict(
                        phase="evidence_accounting",
                        start_s=0,
                        end_s=duration,
                        completed_units=13,
                        pending_units=0,
                    )
                ],
                methodology_note="Read-only hash-bound branch reductions on exposed development; independent child and normal validators; no current inference or service benchmark.",
                flagged_adversarial=False,
                validation_receipts=owned,
                measurement_exit_receipts=measurement,
                upstream_summary_receipts=summaries,
                coverage_statement_counts=coverage.get("files", {}),
                source_artifact_hashes=data["references"],
                code_config_hashes=[reference(ROOT / p) for p in [*OWNED, *DEPENDENCIES, TEST]],
                replay_input_reference=reference(raw / "replay_inputs.json"),
                raw_shard_hashes=[reference(p) for p in raw.rglob("*") if p.is_file()],
                terminal_validation_sidecar_path=str(raw / "terminal_validation.json"),
                publication_gate_results=gates,
                paper_ready=gates["paper_ready"],
                unmet_gates=gates["unmet_gates"],
                external_publication_authorized=False,
                acceptance_gates=dict(
                    owned_validation=passed,
                    science_inputs=value["science_ready_score"],
                    independent_generalization=False,
                    generalized_learning=False,
                ),
            )
            value.update({key: gates["gates"][key] for key in ("G1", "G2", "G3", "G4")})
            health = Path("/tmp/carnot8135-health/receipt.json")
            value["repository_health"] = json.loads(health.read_bytes()) if health.is_file() else []
            value["reproducibility_checksum"] = canonical_hash(
                [value["replay_input_reference"], value["code_config_hashes"]]
            )
            value["field_principles"] = {
                k: f"{k} binds observed branch evidence; exposed data cannot establish independent generalization."
                for k in value
            }
            value["field_principles"].update(
                honest_verdict="Completed nulls and unchanged external blocks are terminal and recorded once.",
                verdict_class="Partial means unfinished owned work; external absence is blocked.",
                rows="Each task has one disposition; every comparative conclusion reduces from primitive source clusters.",
                H1="Source decisions require original sample support and the fixed H1/H2 Holm family.",
                H2="Later benefit also requires retained safety; protocol failure is not a valid scientific null.",
                service_evidence_scope="Host fixture timing and modeled historical acquisition cannot establish complete deployed service.",
                retirement_decisions="Preserve every prior; environmental failures retire no method family.",
                publication_gate_results="Unchanged FoVer G1-G4 and paper readiness authorize no external publication.",
            )
            progress("before_primary_publication", 13, 1)
            atomic_json(raw / "candidate.json", value)
            try:
                pub = publish_primary(output, value, terminal)
            except ValueError:
                atomic_json(raw / "failed_primary.json", value)
                reduction.qualify(value, False)
                value["flagged_adversarial"] = True
                pub = publish_primary(output, value, terminal)
            atomic_json(
                raw / "terminal_validation.json",
                dict(
                    publication=pub,
                    required_checks_passed=value["required_checks_passed"],
                    normal_process_exit=measurement,
                ),
            )
        progress("complete", 13, 0)
        return 0
    except (ValueError, KeyError, OSError, TypeError) as error:
        print(json.dumps(dict(passed=False, error=str(error))), flush=True)
        return 1
