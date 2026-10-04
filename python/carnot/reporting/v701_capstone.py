"""REQ-REPORT-8109: checked accounting preserves blocked science and historical bytes."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import importlib
import json
import os
from pathlib import Path
import re
import tempfile
import time
from typing import Any

import yaml

from carnot.reporting import v699_capstone as previous
from carnot.reporting import v701_capstone_reduction as reduction
from carnot.reporting.current_work_receipt import (
    ZERO_INVOCATION_COUNTS,
    atomic_json,
    canonical_hash,
)
from carnot.reporting.evidence_features_custody_7980 import checked
from carnot.reporting.experiment_7303_validation_scope import CommandSpec, run_commands
from carnot.reporting.primary_publication import publish_primary
from carnot.reporting.roadmap_contract import parse_design
from carnot.reporting.v685_authority_lifecycle import tasks_digest

Json = dict[str, Any]
ROOT = previous.ROOT
NAME = "experiment_8109_v701_capstone"
CLI = f"scripts/experiments/{NAME}.py"
TEST = "tests/python/test_v701_capstone_8109.py"
OWNED = [
    "python/carnot/reporting/v701_capstone.py",
    "python/carnot/reporting/v701_capstone_reduction.py",
    CLI,
]
DESIGN = "openspec/change-proposals/research-roadmap-vNEXT.md"
INPUT = "results/experiment_8097_v701_contract_custody.json"
NAMED = [
    "AGENTS.md",
    "CODEX.md",
    "CLAUDE.md",
    "ops/e2e-test-plan.md",
    "openspec/capabilities/research-reporting/spec.md",
    "openspec/capabilities/verification/spec.md",
    "scripts/experiment_template.py",
    "python/carnot/reporting/current_work_receipt.py",
    "python/carnot/reporting/primary_publication.py",
    "ops/exclusion_manifest.yaml",
    DESIGN,
    "python/carnot/reporting/v699_capstone.py",
    "python/carnot/reporting/v699_capstone_reduction.py",
    "python/carnot/reporting/v685_authority_lifecycle.py",
    "scripts/publication_gate.py",
    "ops/north-star.md",
    "ops/verifier_gaps.md",
]
reference = previous.reference
failure = previous.failure


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Flushed real counts let the operator distinguish work from a stalled child."""
    print(
        f"[exp8109] phase={phase} completed_units={completed} pending_units={pending}", flush=True
    )


def authorities(root: Path) -> tuple[list[Json], list[Json]]:
    """Compare whole immutable invocations; a consumed staging file is not a new authority."""
    data = json.loads((root / INPUT).read_bytes())
    snapshots = data["authority_snapshots"]
    active, design = [
        checked(dict(path=snapshots[k]["snapshot_path"], sha256=snapshots[k]["sha256"]))
        for k in ("active", "design")
    ]
    text = design.read_text()
    table, tasks = parse_design(text, milestone="2026.10.701")
    digest = re.search(r"Canonical full-task SHA-256: `([0-9a-f]{64})`", text)
    expected_table = [
        dict(order=i + 1, **{k: t[k] for k in ("id", "title", "phase", "deliverable")})
        for i, t in enumerate(tasks)
    ]
    invocation = yaml.safe_load(active.read_bytes())
    if (
        invocation["tasks"] != tasks
        or invocation["milestone"] != "2026.10.701"
        or table != expected_table
        or digest is None
        or digest[1] != tasks_digest(tasks)
        or data["canonical_tasks_sha256"] != tasks_digest(tasks)
        or [t["id"].split("-")[0] for t in tasks] != [f"exp{n}" for n in range(8097, 8110)]
    ):
        raise ValueError("immutable_authority_drift")
    return tasks, [reference(root / INPUT), reference(active), reference(design)]


def primitive_audit(value: Json, number: int) -> Json:
    """Qualified independent equations read primitives, never upstream headline totals."""
    mapping = {
        8098: ("carnot.experiment_8098_v701_development_methods", "reduction", None),
        8102: ("carnot.experiment_8102_v701_learning_stream_capture", "reduced_adapter", None),
        8105: ("carnot.experiment_8105_v701_native_radial_kernel", "reduction", "evidence.json"),
        8106: (
            "carnot.experiment_8106_v701_radial_service_cost",
            "reduce_rows",
            "primitive_rows.json",
        ),
        8108: ("carnot.reporting.radial_hardware_8108", "reduce", "replay_inputs.json"),
    }
    if number not in mapping:
        return dict(
            available=False,
            reason="No scientific decision primitives in this administrative branch.",
        )
    module, function, filename = mapping[number]
    operand: Any = value["rows"]
    if filename:
        refs = value["raw_shard_hashes"]
        selected = next(ref for ref in refs if Path(ref["path"]).name == filename)
        operand = json.loads(checked(selected).read_bytes())
    return dict(available=True, result=getattr(importlib.import_module(module), function)(operand))


def load(root: Path, raw: Path) -> Json:
    """Record every observed block; an invalid branch cannot silently contribute benefit."""
    failures: list[Json] = []
    refs: list[Json] = []
    try:
        tasks, refs = authorities(root)
    except (ValueError, KeyError, OSError, TypeError) as error:
        _, tasks = parse_design((ROOT / DESIGN).read_text(), milestone="2026.10.701")
        failures.append(failure(root / INPUT, "exp8097", "immutable_authority", True, str(error)))
    preconditions = []
    for path in [
        *(root / p for p in NAMED),
        *(ROOT / ".venv/bin" / p for p in ("python", "pytest", "coverage", "ruff", "mypy")),
    ]:
        ref = reference(path)
        gate = dict(
            check="resource_exists",
            upstream="preconditions",
            path=str(path),
            hash=ref["sha256"],
            field="is_file",
            op="==",
            expected=True,
            observed=path.is_file(),
            passed=path.is_file(),
        )
        preconditions.append(gate)
        refs.append(ref)
        if not gate["passed"]:
            failures.append(gate)
        progress("precondition", len(preconditions), len(NAMED) + 5 - len(preconditions))
    rows, inputs, issues = previous.collect(root, tasks)
    refs.extend(inputs)
    successful_gate_ids = set()
    primaries, audits = {}, {}
    for task, row in zip(tasks[:-1], rows, strict=True):
        value = previous.read(Path(row["path"]))
        primaries[task["id"]] = value
        successful_gate_ids.update(
            canonical_hash(
                dict(g, artifact_field=g.get("field", g.get("artifact_field")), passed=False)
            )
            for g in value.get("gate_check_summary", [])
            if isinstance(g, dict) and g.get("passed") is True
        )
        row["gate_check_summary"] = [
            g
            for g in row.get("gate_check_summary", [])
            if canonical_hash(g) not in successful_gate_ids
        ]
        eligible = (
            row["primary_present"]
            and row["verdict_class"] not in {"blocked", "disqualified"}
            and not row["gate_check_summary"]
        )
        row.update(
            eligible=eligible,
            excluded=not eligible,
            numerator=int(eligible),
            raw_numerator=int(eligible),
            exclusion_reason=None if eligible else row["exclusion_reason"],
        )
        row.update(
            source_id=task["id"], issued_state=row["honest_verdict"], metric="authenticated_task"
        )
        for gate in task["gated_on"]:
            upstream = primaries.get(gate["upstream"], {})
            observed = upstream.get(gate["artifact_field"])
            if observed != gate["value"]:
                failures.append(
                    failure(
                        root / next(t["deliverable"] for t in tasks if t["id"] == gate["upstream"]),
                        gate["upstream"],
                        gate["artifact_field"],
                        gate["value"],
                        observed,
                    )
                )
        progress("before_primitive_reduction", len(audits), 12 - len(audits))
        try:
            audits[task["id"]] = (
                primitive_audit(value, int(task["id"][3:7]))
                if row["primary_present"]
                else dict(available=False, reason="absent primary")
            )
        except (ValueError, KeyError, OSError, TypeError, StopIteration) as error:
            audits[task["id"]] = dict(available=False, reason=str(error))
            failures.append(
                failure(
                    Path(row["path"]),
                    task["id"],
                    "primitive_reduction",
                    "valid equations",
                    str(error),
                )
            )
            row.update(
                eligible=False,
                excluded=True,
                numerator=0,
                exclusion_reason="invalid_primitive_reduction",
            )
        progress("after_primitive_reduction", len(audits), 12 - len(audits))
    failures.extend(g for g in issues if canonical_hash(g) not in successful_gate_ids)
    refs = [dict(ref, sha256=reference(Path(ref["path"]))["sha256"]) for ref in refs]
    # Keep exact source bytes below raw even when excluded from all benefit claims.
    for ref in refs:
        path = Path(ref["path"])
        if path.is_file() and path.suffix in (".json", ".bin", ".md", ".yaml"):
            target = raw / "custody" / (ref["sha256"].split(":")[-1] + path.suffix)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(path.read_bytes())
            ref.update(source_path=str(path), path=str(target))
    return dict(
        tasks=tasks,
        dispositions=rows,
        primaries=primaries,
        references=refs,
        failures=failures,
        preconditions=preconditions,
        independent_reductions=audits,
    )


def qualify(value: Json, passed: bool) -> None:
    """Failed owned checks disqualify execution; external blocks remain terminal blocked."""
    value.update(
        required_checks_passed=passed,
        capstone_execution_ready_score=int(passed),
        capstone_ready_score=int(passed and value["science_ready_score"]),
    )
    if not passed:
        value.update(
            honest_verdict="complete_disqualified_owned_checks",
            verdict_class="disqualified",
            capstone_ready_score=0,
        )


def replay(path: Path) -> Json:
    """Authenticate observed bytes and repeat equations, including absence and receipt identity."""
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
            raise ValueError("validation_receipt_drift")
    data = json.loads(Path(value["replay_input_reference"]["path"]).read_bytes())
    for task in data["tasks"][:-1]:
        audit = data["independent_reductions"].get(task["id"], {})
        if (
            audit.get("available")
            and primitive_audit(data["primaries"][task["id"]], int(task["id"][3:7])) != audit
        ):
            raise ValueError("primitive_reduction_drift:" + task["id"])
    fresh = reduction.reduce(data)
    if "required_checks_passed" in value:
        qualify(fresh, value["required_checks_passed"])
    for key, wanted in fresh.items():
        if value[key] != wanted:
            raise ValueError("reduction_drift:" + key)
    return dict(passed=True, rows_checksum=canonical_hash(fresh["rows"]))


def commands(scratch: Path) -> list[CommandSpec]:
    """Freeze explicit owned and consumer checks; only new statements enter coverage."""
    from carnot.experiment_8108_v701_radial_hardware_boundary import commands as qualified_commands

    specs = qualified_commands(scratch)
    mapping = {
        "tests/python/test_radial_hardware_8108.py": TEST,
        "python/carnot/experiment_8108_v701_radial_hardware_boundary.py": OWNED[0],
        "python/carnot/reporting/radial_hardware_8108.py": OWNED[1],
        "scripts/experiments/experiment_8108_v701_radial_hardware_boundary.py": CLI,
    }
    config = scratch / "coverage.ini"
    config.write_text(
        "[run]\nparallel=True\ndata_file="
        + str(scratch / ".coverage")
        + "\ninclude=\n    "
        + "\n    ".join(str(ROOT / p) for p in OWNED)
        + "\n"
    )
    adapted = [
        CommandSpec(s.name, tuple(mapping.get(a, a) for a in s.argv), s.scope, s.timeout_s)
        for s in specs
    ]
    return [
        CommandSpec(
            "publication_gate",
            (str(ROOT / ".venv/bin/python"), str(ROOT / "scripts/publication_gate.py"), "--json"),
            "publication",
            30,
        ),
        *adapted,
    ]


def terminal(path: Path) -> Json:
    """Only candidate bytes accepted by the cold reader and both auditors become primary."""
    py = str(ROOT / ".venv/bin/python")
    specs = [
        CommandSpec(name, argv, "terminal", 90)
        for name, argv in (
            ("cold_replay", (py, "-u", str(ROOT / CLI), "--cold-replay", str(path))),
            ("adversarial", (py, str(ROOT / "scripts/adversarial_verify.py"), "--json", str(path))),
            (
                "strict_rows",
                (py, str(ROOT / "scripts/verdict_row_consistency_lint.py"), "--strict", str(path)),
            ),
        )
    ]
    receipts = run_commands(
        ROOT, specs, log_dir=path.parent / "terminal_logs" / str(time.time_ns()), heartbeat_s=10
    )
    return dict(passed=all(r["passed"] for r in receipts), receipts=receipts)


def main(argv: list[str] | None = None) -> int:
    """A normally exited child and current validation bind all reader-visible conclusions."""
    os.environ["PYTHONUNBUFFERED"] = "1"
    began = time.monotonic()
    progress("start")
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
        with tempfile.TemporaryDirectory(prefix="carnot8109-", dir="/tmp") as temporary:
            scratch = Path(temporary)
            specs = [] if args.worker_input else commands(scratch)
            atomic_json(
                raw / "validation_manifest.json",
                dict(commands=[asdict(s) for s in specs], owned=OWNED),
            )
            progress("preconditions_before")
            data = (
                json.loads(args.worker_input.read_bytes())
                if args.worker_input
                else load(args.root, raw)
            )
            atomic_json(raw / "replay_inputs.json", data)
            progress("preconditions_after_before_reduction", 12, 1)
            value = reduction.reduce(data)
            measurement = []
            if not args.worker_input:
                child = CommandSpec(
                    "normal_reduction_exit",
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
                measurement = run_commands(
                    ROOT, [child], log_dir=raw / "measurement_logs", heartbeat_s=10
                )
                if not measurement[0]["passed"]:
                    raise ValueError("reduction_child_failed")
                child_value = json.loads((raw / "worker.json").read_bytes())
                if any(child_value[k] != v for k, v in value.items()):
                    raise ValueError("independent_reduction_drift")
            atomic_json(raw / "independent_reduction.json", value)
            progress("after_reduction_before_validation", 13, len(specs))
            receipts = run_commands(
                ROOT,
                specs,
                log_dir=raw / "validation_logs",
                heartbeat_s=10,
                extra_env=dict(
                    CARNOT_8109_COVERAGE_CONFIG=str(scratch / "coverage.ini"),
                    CARNOT_8109_CLI_RECEIPTS=str(scratch / "private_cli_receipts.json"),
                ),
            )
            receipts = [
                dict(
                    r,
                    log_path=str((ROOT / r["log_path"]).resolve()),
                    argv=r.get("command_argv", []),
                    normal_exit=r.get("exit_code") == 0 and not r.get("timed_out", False),
                )
                for r in receipts
            ]
            owned = [r for r in receipts if r["scope"] == "owned"]
            health = [r for r in receipts if r["scope"] == "repository_health"]
            passed = bool(owned) and all(r["passed"] for r in owned)
            if not args.worker_input:
                qualify(value, passed)
            publication = next((r for r in receipts if r["name"] == "publication_gate"), None)
            gates = (
                json.loads(Path(publication["log_path"]).read_bytes())
                if publication
                else dict(paper_ready=False, unmet_gates=["unmeasured_private_fixture"])
            )
            progress("after_validation_before_publication", len(receipts), 1)
            duration = time.monotonic() - began
            coverage = (
                json.loads((scratch / "coverage.json").read_bytes())
                if (scratch / "coverage.json").exists()
                else {}
            )
            atomic_json(raw / "coverage.json", coverage)
            private_cli = scratch / "private_cli_receipts.json"
            atomic_json(
                raw / "private_cli_receipts.json",
                json.loads(private_cli.read_bytes()) if private_cli.exists() else {},
            )
            atomic_json(
                raw / "validation_receipts.json", dict(owned=owned, repository_health=health)
            )
            value.update(
                experiment_id=8109,
                task_id="exp8109-capstone",
                run_date=args.date,
                milestone="2026.10.701",
                schema="carnot.v701.capstone.v1",
                random_seed=701,
                inference_substrate="aggregation_from_upstream_artifacts",
                inference_substrate_class="no_model_load",
                MODEL_SPECS=[],
                model_invocation_counts=ZERO_INVOCATION_COUNTS,
                trained_head_specs=[],
                duration_s=duration,
                phase_spans=[
                    dict(
                        phase="bounded_reduction_validation",
                        start_s=0,
                        end_s=duration,
                        completed_units=13,
                    )
                ],
                methodology_note="Immutable authority, authenticated primitive reductions and source-level development decisions; no current model work.",
                flagged_adversarial=False,
                validation_receipts=owned,
                repository_health=health,
                measurement_exit_receipts=measurement,
                source_artifact_hashes=data["references"],
                code_config_hashes=[reference(ROOT / p) for p in [*OWNED, TEST]],
                replay_input_reference=reference(raw / "replay_inputs.json"),
                raw_shard_hashes=[reference(p) for p in raw.rglob("*") if p.is_file()],
                terminal_validation_sidecar_path=str(raw / "terminal_validation.json"),
                coverage_statement_counts=coverage.get("files", {}),
                publication_gate_results=gates,
                paper_ready=gates["paper_ready"],
                unmet_gates=gates["unmet_gates"],
                external_publication_authorized=False,
                publication_evidence_scope="Historical FoVer verifier-ensemble headline; separate from exposed V701 radial science and existing corrigenda.",
                publication_scope_hashes=[
                    reference(ROOT / p)
                    for p in (
                        "ops/publication_gate_state.json",
                        "docs/technical-report.md",
                        "docs/arxiv-paper/main.tex",
                        "results/experiment_2850_fover_dual_condition_integrity_v4.json",
                    )
                ],
                acceptance_gates=dict(
                    current_owned_validation=passed,
                    all_science_inputs=value["science_ready_score"],
                    independent_generalization=False,
                    generalized_learning=False,
                ),
            )
            value["reproducibility_checksum"] = canonical_hash(
                [value["replay_input_reference"], value["code_config_hashes"]]
            )
            value["field_principles"] = {
                k: f"{k} preserves scoped observations without independent verifier, lifelong learning or deployment credit."
                for k in value
            }
            value["field_principles"].update(
                task_dispositions="All thirteen ordered outcomes include absent outputs and excluded invalid evidence.",
                validation_receipts="Current argv, normal exit, duration and log hashes authenticate owned readiness.",
                capstone_ready_score="Administrative completion cannot conceal blocked science.",
                publication_gate_results="Historical FoVer G1-G4 cannot close independent V701 science gaps.",
                MODEL_SPECS="No current model load or generation borrows credit from historical producers.",
                retirement_candidates="Literal verdict equality and hashes retire only an exact scientific configuration.",
                H1="Source groups, missing masks and issued costs prevent seeds and selected successes from creating benefit.",
                H2="Original release slots and finite retention support cannot establish lifelong learning.",
            )
            if args.worker_input:
                atomic_json(output, value)
            else:
                publication_receipt = publish_primary(output, value, terminal)
                atomic_json(
                    raw / "terminal_validation.json",
                    dict(
                        publication=publication_receipt,
                        normal_process_exit=measurement,
                        required_checks_passed=passed,
                    ),
                )
        progress("complete", 13, 0)
        return 0
    except (ValueError, KeyError, OSError, TypeError) as error:
        print(json.dumps(dict(passed=False, error=str(error))), flush=True)
        return 1
