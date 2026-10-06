"""REQ-REPORT-8204: publish terminal accounting after normal owned validation.

The thin CLI loads no model. A fresh interpreter repeats the cached reductions,
and locked publication exposes only bytes accepted by unchanged auditors.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import asdict
import json
import os
from pathlib import Path
import tempfile
import threading
import time
from typing import Any

from carnot.reporting import v708_capstone_inputs as e
from carnot.reporting import v708_capstone_science as science
from carnot.reporting.current_work_receipt import (
    ZERO_INVOCATION_COUNTS,
    atomic_json,
    canonical_hash,
)
from carnot.reporting.experiment_7303_validation_scope import CommandSpec, build_scoped_commands
from carnot.reporting.primary_publication import publish_primary
from carnot.reporting.v703_capstone import execute as run_checks

Json = dict[str, Any]
ROOT = e.ROOT
NAME = "experiment_8204_v708_capstone"
CLI = f"scripts/experiments/{NAME}.py"
TEST = "tests/python/test_v708_capstone_8204.py"
OWNED = [
    "python/carnot/reporting/v708_capstone.py",
    "python/carnot/reporting/v708_capstone_inputs.py",
    "python/carnot/reporting/v708_capstone_science.py",
    CLI,
]
MODEL_SPECS: list[Json] = []


def execute(plan: list[CommandSpec], raw: Path, heartbeat_s: float = 30) -> list[Json]:
    """Report completed child counts even when a child writes only to its log."""
    receipts = []
    for index, spec in enumerate(plan):
        e.progress("before_subprocess_" + spec.name, index, len(plan) - index)
        stop = threading.Event()

        def monitor() -> None:
            while not stop.wait(heartbeat_s):
                e.progress("waiting_" + spec.name, index, len(plan) - index)

        thread = threading.Thread(target=monitor, daemon=True)
        thread.start()
        try:
            current = run_checks([spec], raw / spec.name)
            for receipt in current:
                Path(receipt["log_path"]).chmod(0o444)
            receipts.extend(current)
        finally:
            stop.set()
            thread.join()
        e.progress("after_subprocess_" + spec.name, index + 1, len(plan) - index - 1)
    return receipts


def commands(private: Path) -> list[CommandSpec]:
    """Freeze file-based static checks and coverage limited to the newly added code."""
    private.mkdir(parents=True, exist_ok=True)
    (private / "pytest").mkdir(parents=True, exist_ok=True)
    config = private / "coverage.ini"
    config.write_text(
        "[run]\ndata_file="
        + str(private / ".coverage")
        + "\ninclude=\n    "
        + "\n    ".join(str(ROOT / p) for p in OWNED)
        + "\n"
    )
    plan = build_scoped_commands(
        ROOT,
        [TEST],
        OWNED[:-1],
        static_paths=[CLI],
        basetemp=private / "pytest",
        coverage_file=private / ".coverage",
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
            *OWNED,
        ),
    )
    plan = [CommandSpec(s.name, replacements.get(s.name, s.argv), "owned", 240) for s in plan]
    py = str(ROOT / ".venv/bin/python")
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
                    "tests/python/test_contract_custody_8192.py",
                    "--basetemp=" + str(private / "consumers"),
                ),
                "owned",
                240,
            ),
            CommandSpec(
                "publication_gate", (py, "scripts/publication_gate.py", "--json"), "publication", 90
            ),
        ]
    )
    tasks = json.loads((ROOT / e.INPUT).read_bytes())["task_contract"]
    paths = [
        str(ROOT / t["deliverable"]) for t in tasks[:-1] if (ROOT / t["deliverable"]).is_file()
    ]
    paths.extend(
        str(p) for n in (8174, 8188) for p in (ROOT / "results").glob(f"experiment_{n}_*.json")
    )
    plan.append(
        CommandSpec(
            "upstream_summaries",
            (py, "-u", "scripts/summarize_artifact.py", *paths),
            "upstream_inventory",
            240,
        )
    )
    plan.append(
        CommandSpec(
            "full_python_suite",
            (str(ROOT / ".venv/bin/pytest"), "tests/python", "-q"),
            "repository_health",
            600,
        )
    )
    return plan


def inherit_health(previous: Json, raw: Path) -> list[Json]:
    """Reuse authenticated unrelated health so an owned repair cannot rerun the suite."""
    receipts: list[Json] = deepcopy(previous.get("repository_health", {}).get("receipts", []))
    for receipt in receipts:
        original = Path(receipt["log_path"])
        if e.reference(original)["sha256"] != receipt["log_sha256"]:
            raise ValueError("historical_health_log_hash_drift")
        target = raw / "health_custody" / (receipt["log_sha256"].split(":")[-1] + ".log")
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("xb") as stream:
            stream.write(original.read_bytes())
        target.chmod(0o444)
        receipt.update(source_log_path=str(original), log_path=str(target))
    return receipts


def qualify(value: Json, passed: bool) -> None:
    """Owned failure cannot retain readiness or masquerade as an external block."""
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
    """Rehash immutable custody and recompute headlines rather than trusting scores."""
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
        if (
            audit["available"]
            and science.primitive(data["primaries"][task["id"]], 8192 + index) != audit
        ):
            raise ValueError("primitive_reduction_drift")
    fresh = science.reduce(data)
    qualify(fresh, value["required_checks_passed"])
    for key, expected in fresh.items():
        if value[key] != expected:
            raise ValueError("reduction_drift:" + key)
    return dict(passed=True, rows_checksum=canonical_hash(fresh["rows"]))


def terminal(path: Path) -> Json:
    """Unmodified auditors and a fresh cold CLI must accept the candidate bytes."""
    py = str(ROOT / ".venv/bin/python")
    plan = [
        CommandSpec(name, argv, "terminal", 180)
        for name, argv in (
            ("cold_replay", (py, "-u", str(ROOT / CLI), "--cold-replay", str(path))),
            ("adversarial", (py, str(ROOT / "scripts/adversarial_verify.py"), "--json", str(path))),
            (
                "strict_rows",
                (py, str(ROOT / "scripts/verdict_row_consistency_lint.py"), "--strict", str(path)),
            ),
        )
    ]
    receipts = execute(plan, path.parent / "terminal_logs" / str(time.time_ns()))
    return dict(passed=all(r["passed"] and r["normal_exit"] for r in receipts), receipts=receipts)


def main(argv: list[str] | None = None) -> int:
    """Freeze checks before reduction, then publish one fully validated terminal result."""
    os.environ["PYTHONUNBUFFERED"] = "1"
    began = time.monotonic()
    e.progress("start")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", choices=["20261006"], default="20261006")
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
        if args.worker_input:
            atomic_json(output, science.reduce(json.loads(args.worker_input.read_bytes())))
            e.progress("worker_complete", 13, 0)
            return 0
        if args.fixture_e2e and args.root.resolve() == ROOT:
            raise ValueError("fixture_requires_private_root")
        raw = output.parent / "raw" / NAME / "invocations" / str(time.time_ns())
        raw.mkdir(parents=True, exist_ok=True)
        previous = json.loads(output.read_bytes()) if output.is_file() else {}
        if output.is_file():
            (raw / "previous_primary.json").write_bytes(output.read_bytes())
        inherited_health = inherit_health(previous, raw)
        with tempfile.TemporaryDirectory(prefix="carnot8204-", dir="/tmp") as temp:
            private = Path(temp)
            probe = private / "storage_probe"
            probe.write_bytes(b"actual private writable storage")
            writable = (
                probe.read_bytes() == b"actual private writable storage"
                and private.stat().st_mode & 0o077 == 0
            )
            plan = commands(private)
            if inherited_health:
                plan = [s for s in plan if s.scope != "repository_health"]
            if args.fixture_e2e:
                plan = [s for s in plan if s.name in ("consumers_E2E018", "publication_gate")]
            atomic_json(
                raw / "validation_manifest.json",
                dict(
                    commands=[asdict(s) for s in plan],
                    owned=OWNED,
                    imported_health_receipts=inherited_health,
                ),
            )
            summaries = execute(
                [s for s in plan if s.scope == "upstream_inventory"], raw / "summary_logs"
            )
            e.progress("before_preconditions")
            data = e.load(args.root, raw)
            data["verifier_is_oracle"] = args.fixture_e2e
            data["runtime"] = dict(
                python=os.sys.version,
                executable=os.sys.executable,
                PYTHONUNBUFFERED=os.environ["PYTHONUNBUFFERED"],
                JAX_PLATFORMS=os.environ.get("JAX_PLATFORMS"),
                private_storage_writable=bool(writable),
                private_mode=oct(private.stat().st_mode & 0o777),
            )
            e.progress("after_preconditions", 12, 0)
            atomic_json(raw / "replay_inputs.json", data)
            e.progress("before_reduction", 12, 1)
            value = science.reduce(data)
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
                180,
            )
            measurement = execute([worker], raw / "measurement_logs")
            if (
                not measurement[0]["passed"]
                or json.loads((private / "worker.json").read_bytes()) != value
            ):
                raise ValueError("independent_reduction_child_failed")
            atomic_json(raw / "independent_reduction.json", value)
            e.progress("after_reduction", 13, 0)
            receipts = execute(
                [s for s in plan if s.scope not in ("upstream_inventory", "repository_health")],
                raw / "validation_logs",
            )
            owned = [r for r in receipts if r["scope"] == "owned"]
            passed = bool(writable and owned) and all(
                r["passed"] and r["normal_exit"] for r in owned
            )
            qualify(value, passed)
            gate = next(r for r in receipts if r["name"] == "publication_gate")
            gates = json.loads(Path(gate["log_path"]).read_bytes())
            health = inherited_health or execute(
                [s for s in plan if s.scope == "repository_health"], raw / "health_logs"
            )
            coverage_path = private / "coverage.json"
            coverage = json.loads(coverage_path.read_bytes()) if coverage_path.is_file() else {}
            atomic_json(raw / "coverage.json", coverage)
            atomic_json(
                raw / "primitive_rows.json",
                dict(
                    rows=value["rows"],
                    source_rows={
                        t: a["result"].get("rows", [])
                        for t, a in data["audits"].items()
                        if a["available"]
                    },
                ),
            )
            duration = time.monotonic() - began
            value.update(
                experiment_id=8204,
                task_id="exp8204-capstone",
                run_date=args.date,
                milestone="2026.10.708",
                schema="carnot.v708.capstone.v1",
                random_seed=7088204,
                duration_s=duration,
                inference_substrate="aggregation_from_upstream_artifacts",
                inference_substrate_class="no_model_load",
                MODEL_SPECS=[],
                trained_head_specs=[],
                model_invocation_counts=dict(ZERO_INVOCATION_COUNTS),
                call_ledger=[],
                cited_upstream_artifacts=[
                    dict(
                        experiment_id=8192 + i,
                        fields_imported=list(p),
                        sha256=data["dispositions"][i]["sha256"],
                    )
                    for i, p in enumerate(data["primaries"].values())
                    if p
                ],
                historical_model_provenance=[
                    dict(
                        task_id=t,
                        MODEL_SPECS=p.get("MODEL_SPECS", []),
                        model_invocation_counts=p.get("model_invocation_counts", {}),
                        trained_head_specs=p.get("trained_head_specs", []),
                    )
                    for t, p in data["primaries"].items()
                ],
                phase_spans=[
                    dict(
                        phase="owned_accounting",
                        start_s=0,
                        end_s=duration,
                        completed_units=13,
                        pending_units=0,
                    )
                ],
                methodology_note="Authenticated cached source reductions; no current model, trained head, device dispatch or deployment benchmark.",
                flagged_adversarial=False,
                validation_receipts=owned,
                measurement_exit_receipts=measurement,
                upstream_summary_receipts=summaries,
                coverage_statement_counts=coverage.get("files", {}),
                coverage_totals=coverage.get("totals", {}),
                source_artifact_hashes=data["references"],
                code_config_hashes=[e.reference(ROOT / p) for p in [*OWNED, TEST]],
                replay_input_reference=e.reference(raw / "replay_inputs.json"),
                raw_shard_hashes=[e.reference(p) for p in raw.rglob("*") if p.is_file()],
                terminal_validation_sidecar_path=str(raw / "terminal_validation.json"),
                publication_gate_results=gates,
                paper_ready=gates["paper_ready"],
                unmet_gates=gates["unmet_gates"],
                external_publication_authorized=False,
                repository_health=dict(receipts=health, owned=False, reused=bool(inherited_health)),
                historical_owned_validation_failures=[
                    r for r in previous.get("validation_receipts", []) if not r["passed"]
                ],
                runtime_preconditions=data["runtime"],
                source_comparisons=data["source_comparisons"],
                acceptance_gates=dict(
                    owned_validation=passed,
                    science_inputs=False,
                    independent_generalization=False,
                    alpha_H1=0.025,
                    alpha_H2=0.025,
                    nfr01_complete_service_ratio=10,
                ),
            )
            value.update({k: gates["gates"][k] for k in ("G1", "G2", "G3", "G4")})
            value["reproducibility_checksum"] = canonical_hash(
                [value["replay_input_reference"], value["code_config_hashes"]]
            )
            value["field_principles"] = {
                k: "Completion grants no independent science credit; exact frozen evidence and normal checks determine the claim."
                for k in value
            }
            e.progress("before_primary_publication", 13, 1)
            atomic_json(raw / "candidate.json", value)
            pub = publish_primary(output, value, terminal)
            atomic_json(
                raw / "terminal_validation.json",
                dict(publication=pub, required_checks_passed=passed),
            )
        e.progress("complete", 13, 0)
        return 0
    except (ValueError, KeyError, OSError, TypeError) as error:
        print(json.dumps(dict(passed=False, error=str(error))), flush=True)
        return 1
