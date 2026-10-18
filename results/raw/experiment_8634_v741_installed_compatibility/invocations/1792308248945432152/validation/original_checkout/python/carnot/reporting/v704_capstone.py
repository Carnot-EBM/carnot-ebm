"""Run and publish V704 capstone accounting after normal owned validation.

REQ-REPORT-8149: commands freeze before measurement and raw evidence stays below
raw/. A blocked research branch does not invalidate completed accounting.
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

from carnot.reporting import v704_capstone_evidence as e
from carnot.reporting.current_work_receipt import (
    ZERO_INVOCATION_COUNTS,
    atomic_json,
    canonical_hash,
)
from carnot.reporting.experiment_7303_validation_scope import CommandSpec, build_scoped_commands
from carnot.reporting.primary_publication import publish_primary
from carnot.reporting.v703_capstone import execute

Json = dict[str, Any]
ROOT = e.ROOT
NAME = "experiment_8149_v704_capstone"
CLI = f"scripts/experiments/{NAME}.py"
TEST = "tests/python/test_v704_capstone_8149.py"
OWNED = [
    "python/carnot/reporting/v704_capstone.py",
    "python/carnot/reporting/v704_capstone_evidence.py",
    CLI,
]
MODEL_SPECS: list[Json] = []


def commands(private: Path) -> list[CommandSpec]:
    """Freeze explicit checks and added-statement coverage without a global fallback."""
    (private / "pytest").mkdir(parents=True, exist_ok=True)
    plan = build_scoped_commands(
        ROOT,
        [TEST],
        OWNED[:-1],
        static_paths=[CLI],
        basetemp=private / "pytest",
        coverage_file=private / ".coverage",
    )
    config = private / "coverage.ini"
    config.write_text(
        "[run]\ndata_file="
        + str(private / ".coverage")
        + "\ninclude=\n    "
        + "\n    ".join(str(ROOT / p) for p in OWNED)
        + "\n"
    )
    replacements = dict(
        changed_module_coverage=(
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
            "--basetemp=" + str(private / "covered"),
        ),
        changed_module_coverage_report=(
            str(ROOT / ".venv/bin/coverage"),
            "json",
            "--rcfile=" + str(config),
            "--fail-under=100",
            "-o",
            str(private / "coverage.json"),
        ),
        changed_module_mypy=(
            str(ROOT / ".venv/bin/mypy"),
            "--strict",
            "--follow-imports=skip",
            *OWNED[:-1],
        ),
    )
    plan = [CommandSpec(s.name, replacements.get(s.name, s.argv), "owned", 180) for s in plan]
    plan.extend(
        [
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
                    "tests/python/test_primary_publication_7928.py",
                    "tests/python/test_conductor_gates.py",
                    "tests/python/test_experiment_7891_v685_authority_lifecycle.py",
                    "--basetemp=" + str(private / "consumers"),
                ),
                "owned",
                180,
            ),
            CommandSpec(
                "publication_gate",
                (str(ROOT / ".venv/bin/python"), "scripts/publication_gate.py", "--json"),
                "publication",
                60,
            ),
        ]
    )
    sources = sorted(
        {
            path
            for number in range(8136, 8149)
            for path in (ROOT / "results").glob(f"experiment_{number}_*.json")
        }
    )
    plan.append(
        CommandSpec(
            "upstream_summaries",
            (
                str(ROOT / ".venv/bin/python"),
                "-u",
                "scripts/summarize_artifact.py",
                *map(str, sources),
            ),
            "upstream_inventory",
            120,
        )
    )
    return plan


def qualify(value: Json, passed: bool) -> None:
    """Validation failure disqualifies this invocation and zeros all readiness."""
    value.update(required_checks_passed=passed, capstone_execution_ready_score=int(passed))
    if not passed:
        value.update(
            honest_verdict="complete_disqualified_owned_validation",
            verdict_class="disqualified",
            science_ready_score=0,
            h1_development_signal_score=0,
            h2_development_signal_score=0,
        )
        value["rows"][-1].update(
            honest_verdict=value["honest_verdict"],
            verdict_class="disqualified",
            failed=True,
            exclusion_reason="owned_validation",
        )
        value["failed_count"] = sum(r["failed"] for r in value["rows"])


def replay(path: Path) -> Json:
    """Rehash immutable snapshots and independently reconstruct each owned reduction."""
    value = json.loads(path.read_bytes())
    for receipt in value["validation_receipts"]:
        if e.reference(Path(receipt["log_path"]))["sha256"] != receipt["log_sha256"]:
            raise ValueError("validation_log_hash_drift")
    for ref in [
        value["replay_input_reference"],
        *value["source_artifact_hashes"],
        *value["raw_shard_hashes"],
        *value["code_config_hashes"],
    ]:
        if e.reference(Path(ref["path"]))["sha256"] != ref["sha256"]:
            raise ValueError("input_hash_drift:" + ref["path"])
    data = json.loads(Path(value["replay_input_reference"]["path"]).read_bytes())
    for index, task in enumerate(data["tasks"][:-1]):
        audit = data["audits"][task["id"]]
        if audit["available"] and e.primitive(data["primaries"][task["id"]], 8136 + index) != audit:
            raise ValueError("primitive_reduction_drift")
    fresh = e.reduce(data)
    qualify(fresh, value["required_checks_passed"])
    for key, expected in fresh.items():
        if value[key] != expected:
            raise ValueError("reduction_drift:" + key)
    return dict(passed=True, rows_checksum=canonical_hash(fresh["rows"]))


def terminal(path: Path) -> Json:
    """Unchanged adversarial and strict row readers must exit normally on exact bytes."""
    py = str(ROOT / ".venv/bin/python")
    plan = [
        CommandSpec(name, argv, "terminal", 120)
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
    """Finish unchanged external blocks once, preserving failed owned candidates."""
    os.environ["PYTHONUNBUFFERED"] = "1"
    began = time.monotonic()
    e.progress("start")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", choices=["20261005"], default="20261005")
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
        if (args.worker_input or args.fixture_e2e) and output.is_relative_to(ROOT / "results"):
            raise ValueError("private_mode_cannot_publish_production")
        raw = output.parent / "raw" / NAME / "invocations" / str(time.time_ns())
        raw.mkdir(parents=True, exist_ok=True)
        if output.is_file():
            shutil.copyfile(output, raw / "previous_primary.json")
        if args.worker_input:
            atomic_json(output, e.reduce(json.loads(args.worker_input.read_bytes())))
            e.progress("worker_complete", 14, 0)
            return 0
        with tempfile.TemporaryDirectory(prefix="carnot8149-", dir="/tmp") as temp:
            private = Path(temp)
            plan = commands(private)
            if args.fixture_e2e:
                if args.root.resolve() == ROOT:
                    raise ValueError("fixture_requires_private_root")
                plan = [s for s in plan if s.name in {"consumers_E2E018", "publication_gate"}]
            atomic_json(
                raw / "validation_manifest.json",
                dict(commands=[asdict(s) for s in plan], owned=OWNED),
            )
            summaries = execute(
                [s for s in plan if s.scope == "upstream_inventory"], raw / "summary_logs"
            )
            e.progress("before_input_load")
            data = e.load(args.root, raw)
            data["verifier_is_oracle"] = args.fixture_e2e
            atomic_json(raw / "replay_inputs.json", data)
            e.progress("after_input_load_before_reduction", 13, 1)
            value = e.reduce(data)
            worker = CommandSpec(
                "independent_reduction",
                (
                    str(ROOT / ".venv/bin/python"),
                    "-u",
                    str(ROOT / CLI),
                    "--worker-input",
                    str(raw / "replay_inputs.json"),
                    "--output",
                    str(private / "worker.json"),
                ),
                "measurement",
                120,
            )
            measurement = execute([worker], raw / "measurement_logs")
            if (
                not measurement[0]["passed"]
                or json.loads((private / "worker.json").read_bytes()) != value
            ):
                raise ValueError("independent_reduction_child_failed")
            atomic_json(raw / "independent_reduction.json", value)
            e.progress("after_reduction_before_validation", 14, len(plan))
            receipts = execute(
                [s for s in plan if s.scope != "upstream_inventory"], raw / "validation_logs"
            )
            owned = [r for r in receipts if r["scope"] == "owned"]
            passed = bool(owned) and all(r["passed"] and r["normal_exit"] for r in owned)
            qualify(value, passed)
            gate_receipt = next(r for r in receipts if r["name"] == "publication_gate")
            gates = json.loads(Path(gate_receipt["log_path"]).read_bytes())
            coverage_path = private / "coverage.json"
            coverage = json.loads(coverage_path.read_bytes()) if coverage_path.is_file() else {}
            atomic_json(raw / "coverage.json", coverage)
            atomic_json(raw / "primitive_rows.json", dict(rows=value["rows"]))
            duration = time.monotonic() - began
            dependencies = [
                "python/carnot/verify/learning_audit_8144.py",
                "python/carnot/experiment_8145_v704_natural_service_cost.py",
                "python/carnot/experiment_8146_v704_live_service_cost.py",
                "python/carnot/reporting/hardware_workload_8148.py",
                "python/carnot/reporting/v685_authority_lifecycle.py",
                "python/carnot/reporting/primary_publication.py",
                "python/carnot/reporting/current_work_receipt.py",
                "python/carnot/reporting/experiment_7303_validation_scope.py",
            ]
            value.update(
                experiment_id=8149,
                task_id="exp8149-capstone",
                run_date=args.date,
                milestone="2026.10.704",
                schema="carnot.v704.capstone.v1",
                random_seed=7048144,
                inference_substrate="aggregation_from_upstream_artifacts",
                inference_substrate_class="no_model_load",
                MODEL_SPECS=[],
                model_invocation_counts=deep_counts(),
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
                        phase="owned_accounting",
                        start_s=0,
                        end_s=duration,
                        completed_units=14,
                        pending_units=0,
                    )
                ],
                methodology_note="Hash-bound independent aggregation of completed, skipped and disqualified development tasks; no current inference, service benchmark or external generalization.",
                flagged_adversarial=False,
                validation_receipts=owned,
                measurement_exit_receipts=measurement,
                upstream_summary_receipts=summaries,
                coverage_statement_counts=coverage.get("files", {}),
                source_artifact_hashes=data["references"],
                code_config_hashes=[e.reference(ROOT / p) for p in [*OWNED, *dependencies, TEST]],
                replay_input_reference=e.reference(raw / "replay_inputs.json"),
                raw_shard_hashes=[e.reference(p) for p in raw.rglob("*") if p.is_file()],
                terminal_validation_sidecar_path=str(raw / "terminal_validation.json"),
                publication_gate_results=gates,
                paper_ready=gates["paper_ready"],
                unmet_gates=gates["unmet_gates"],
                external_publication_authorized=False,
                acceptance_gates=dict(
                    owned_validation=passed,
                    science_inputs=False,
                    independent_generalization=False,
                    generalized_learning=False,
                ),
            )
            value.update({key: gates["gates"][key] for key in ("G1", "G2", "G3", "G4")})
            health = Path("/tmp/carnot8149-health/receipt.json")
            value["repository_health"] = json.loads(health.read_bytes()) if health.is_file() else {}
            value["reproducibility_checksum"] = canonical_hash(
                [value["replay_input_reference"], value["code_config_hashes"]]
            )
            value["field_principles"] = {
                k: "Observed evidence only; completion does not confer natural benefit or independent generalization."
                for k in value
            }
            value["field_principles"].update(
                honest_verdict="Completed nulls and external blocks are terminal and recorded once.",
                verdict_class="Partial means unfinished owned work only; external blocks stay blocked.",
                rows="One disposition per actual YAML task; primitive source masks and repetitions remain separate.",
                inference_substrate="Current invocation owns zero model loads and zero generations.",
                retirement_decisions="Retirement needs authenticated repeated scope; missing input retires no new method family.",
            )
            e.progress("before_primary_publication", 14, 1)
            atomic_json(raw / "candidate.json", value)
            try:
                pub = publish_primary(output, value, terminal)
            except ValueError:
                atomic_json(raw / "failed_primary.json", value)
                qualify(value, False)
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
        e.progress("complete", 14, 0)
        return 0
    except (ValueError, KeyError, OSError, TypeError) as error:
        print(json.dumps(dict(passed=False, error=str(error))), flush=True)
        return 1


def deep_counts() -> Json:
    """Return independent zero counters for current execution, never imported counts."""
    return dict(ZERO_INVOCATION_COUNTS)
