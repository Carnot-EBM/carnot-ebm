"""REQ-REPORT-8429: bind executable authority without changing the scientific question."""

from __future__ import annotations

import json
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory
import time
from typing import Any

import yaml

from carnot.reporting import v725_contract_methods as legacy
from carnot.reporting.current_work_receipt import atomic_json, sha256_file
from carnot.reporting.roadmap_contract import parse_design
from carnot.reporting.v685_authority_lifecycle import assess_authorities, tasks_digest
from carnot.reporting.v710_contract_replay import snapshot

Json = dict[str, Any]
ROOT = legacy.ROOT
NAME, TASK, MILESTONE = (
    "experiment_8429_v726_contract_methods",
    "exp8429-contract-methods",
    "2026.10.726",
)
DESIGN, ACTIVE, STAGED = legacy.DESIGN, legacy.ACTIVE, legacy.STAGED
PROTOCOL = "openspec/change-proposals/v726-methods-protocol.json"
METHODS = "docs/research-notes/v726-methods.md"
PROTOCOL_PIN = "sha256:bb868af1d6f173c5b98155498e5daadd286e2641ef4aa3c20b7a0671b1b95117"
METHODS_PIN = "sha256:e80cfc4f3ea7f4a5221fef92dd4987d0e587487c8c5e7bbff7b6c40cb3d16c11"
TASKS_PIN = "72f4cca447651e88180fd3196f947615d653b04cc7280ff8f12fd8f5eaedaa88"
SCIENCE = ("partition", "features", "fit", "decision", "feedback", "learning_audit", "cost")
CLI, TEST = f"scripts/experiments/{NAME}.py", "tests/python/test_v726_contract_methods_8429.py"
OWNED = [
    "python/carnot/reporting/v726_contract_methods.py",
    "python/carnot/reporting/v726_contract_runner.py",
    CLI,
]
MODEL_SPECS: list[Json] = []
read, reference, checksum = legacy.read, legacy.reference, legacy.checksum


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Flushed counts make bounded work visible while children hold their logs."""
    print(f"[exp8429] phase={phase} completed={completed} pending={pending}", flush=True)


def failure(path: Path, field: str, expected: Any, observed: Any) -> Json:
    """A missing operand stays null so a measured zero cannot replace it."""
    return dict(legacy.failure(path, field, expected, observed), upstream_id=TASK)


def protocol() -> Json:
    """Fixed independent seals prevent rehashed candidates from moving the split."""
    if sha256_file(ROOT / METHODS) != METHODS_PIN:
        raise ValueError("methods_hash")
    p = dict(read(dict(path=str(ROOT / PROTOCOL), sha256=PROTOCOL_PIN)))
    old = read(p["inherited_protocol_reference"])
    if any(p[key] != old[key] for key in SCIENCE):
        raise ValueError("inherited_scientific_operand")
    return p


def authority(root: Path, raw: Path) -> Json:
    """The shipped reader checks order; full equality also binds unlisted task fields."""
    paths = [root / n for n in [DESIGN, STAGED, ACTIVE]]
    snapshots = {}
    for role, path in zip(["design", "staged", "active"], paths):
        if path.is_file():
            snapshots[role] = snapshot(path, raw, role)
        else:
            snapshots[role] = dict(source_path=str(path), exists=False, sha256=None)
    try:
        table, machine = parse_design(paths[0].read_text(), milestone=MILESTONE)
        result = assess_authorities(
            *paths, raw / "typed", milestone=MILESTONE, first_id=8429, count=14
        )
        active = yaml.safe_load(paths[2].read_bytes()) if paths[2].is_file() else {}
        tasks = active.get("tasks", [])
        for i, row in enumerate(result["contract_rows"]):
            row["checks"]["full_object"] = (
                i < len(tasks) and i < len(machine) and tasks[i] == machine[i]
            )
            row["matched"] = all(row["checks"].values())
            row.update(
                absolute_metric=int(row["matched"]),
                raw_numerator=sum(row["checks"].values()),
                raw_denominator=len(row["checks"]),
                excluded=not row["matched"],
            )
        failures = result["gate_check_summary"]
        if tasks_digest(machine) != TASKS_PIN or result["canonical_tasks_sha256"] != TASKS_PIN:
            failures.append(
                failure(paths[0], "pinned_complete_task_digest", TASKS_PIN, tasks_digest(machine))
            )
        if not all(row["matched"] for row in result["contract_rows"]):
            failures.append(failure(paths[2], "full_task_objects", True, False))
        result.update(
            observed_count=len(tasks),
            tasks=tasks,
            observed_active_tasks_sha256=tasks_digest(tasks),
            activated=not failures,
            gate_check_summary=failures,
        )
    except (OSError, ValueError, KeyError, IndexError, TypeError) as error:
        rows = [
            dict(
                unit_id=f"exp{8429 + i}",
                family=f"exp{8429 + i}",
                order=i + 1,
                arm="active_contract",
                seed=None,
                status="unstarted",
                checks=dict(independent_authority=False),
                matched=False,
                absolute_metric=0,
                raw_numerator=0,
                raw_denominator=1,
                censored=True,
                excluded=False,
                effective_independent_groups=0,
                missing_reason="required_authority_unavailable_or_invalid",
            )
            for i in range(14)
        ]
        result = dict(
            activated=False,
            canonical_tasks_sha256=TASKS_PIN,
            contract_rows=rows,
            observed_count=0,
            tasks=[],
            observed_active_tasks_sha256=None,
            gate_check_summary=[
                failure(
                    paths[0],
                    "required_authority",
                    True,
                    None if isinstance(error, FileNotFoundError) else str(error),
                )
            ],
        )
    for gate in result["gate_check_summary"]:
        gate.update(check=gate["artifact_field"], upstream_id=TASK, passed=False)
    result["authority_snapshots"] = snapshots
    result["staging_disposition"] = "existing" if paths[1].is_file() else "consumed_or_absent"
    return dict(result)


def runtime(root: Path) -> Json:
    """Reuse the qualified typed causal reader; no device inventory or probe runs."""
    result = legacy.runtime(root)
    for gate in result["failures"]:
        gate["upstream_id"] = TASK
    return dict(result)


def measure(root: Path, raw: Path) -> Json:
    """Seal required named operands before administrative readiness is reduced."""
    start = time.monotonic_ns()
    raw.mkdir(parents=True, exist_ok=True, mode=0o700)
    progress("preconditions_before", 0, 14)
    p = protocol()
    resources = dict(
        legacy.resources(raw),
        free_disk_bytes=shutil.disk_usage(raw).free,
        tool_paths={
            n: shutil.which(str(ROOT / ".venv/bin" / n))
            for n in ["python", "pytest", "coverage", "ruff", "mypy"]
        },
        validation_reserve_s=900,
        measurement_cap_s=3900,
    )
    failures = []
    if not (
        resources["private_disk_backed_scratch"]
        and resources["available_memory_bytes"] >= resources["minimum_memory_bytes"]
        and resources["free_disk_bytes"] >= 104857600
        and all(resources["tool_paths"].values())
    ):
        failures.append(failure(raw, "private_disk_memory_tools_deadline", True, resources))
    for ref in [*p["source_references"], p["release_primitive_reference"]]:
        path = Path(ref["path"])
        if not path.is_file() or sha256_file(path) != ref["sha256"]:
            failures.append(
                failure(
                    path,
                    "required_input_sha256",
                    ref["sha256"],
                    sha256_file(path) if path.is_file() else None,
                )
            )
    c = authority(root, raw / "authority")
    progress("authority_after", c["observed_count"], max(0, 14 - c["observed_count"]))
    repair = runtime(root)
    refs = [
        dict(path=s["snapshot_path"], sha256=s["sha256"], source_path=s["source_path"], role=n)
        for n, s in c["authority_snapshots"].items()
        if "snapshot_path" in s
    ]
    work = dict(
        root=str(root),
        started_monotonic_ns=start,
        ended_monotonic_ns=time.monotonic_ns(),
        preconditions=resources,
        contract=c,
        runtime=repair,
        failures=failures,
        protocol=p,
        authority_inputs=refs,
        source_refs=p["source_references"],
        code_refs=[
            reference(ROOT / n)
            for n in [
                *OWNED,
                TEST,
                "python/carnot/reporting/v725_contract_methods.py",
                "python/carnot/reporting/v725_contract_runner.py",
                "python/carnot/reporting/v717_contract_runner.py",
                "python/carnot/reporting/primary_publication.py",
                "python/carnot/reporting/v685_authority_lifecycle.py",
                "python/carnot/reporting/roadmap_contract.py",
                "python/carnot/reporting/v709_execution.py",
                "python/carnot/verify/cuda_reader_8410.py",
                "scripts/adversarial_verify.py",
                "scripts/verdict_row_consistency_lint.py",
                "scripts/experiment_template.py",
            ]
        ],
        phase_spans=[
            dict(
                name="preconditions_authority_frozen_methods",
                duration_s=(time.monotonic_ns() - start) / 1e9,
            )
        ],
    )
    atomic_json(raw / "measurement.json", work)
    progress("methods_frozen_after", 14, 0)
    return work


def build(work: Json, receipts: list[Json], raw: Path, output: Path) -> Json:
    """Reuse the established schema; qualification remains separate from scientific benefit."""
    value = legacy.build(work, receipts, raw, output)
    ready = value["methods_ready_score"]
    value.update(
        experiment_id=8429,
        task_id=TASK,
        milestone=MILESTONE,
        honest_verdict=f"complete_{value['verdict_class']}_v726_contract_methods",
        intended_count=14,
        random_seed=7258429,
        sample_size_budget=dict(
            intended_task_slots=14,
            independent_questions=0,
            retrospective_questions_reserved=300,
            seeds_and_siblings_are_not_independent=True,
        ),
        contract_ready_score=ready,
        canonical_tasks_sha256=TASKS_PIN,
        contract_rows=value["rows"],
        authority_snapshots=work["contract"]["authority_snapshots"],
        protocol_path=str(ROOT / PROTOCOL),
        protocol_sha256=PROTOCOL_PIN,
        protocol_delta=work["protocol"]["protocol_delta"],
        methods_protocol_path=str(ROOT / PROTOCOL),
        methods_protocol_sha256=PROTOCOL_PIN,
        methods_note_path=str(ROOT / METHODS),
        methods_note_sha256=METHODS_PIN,
        runtime_repair_reference=work["runtime"]["receipt"],
        historical_authority_limits=dict(
            v725="four activated tasks; methods score zero; no calibration or learning result",
            v722="missing history remains a separate unresolved branch",
            v717_H1=-0.00390625,
            v717_H2=0,
            public_labels="900 answers, 300 exposed question clusters; older overlap unknown",
        ),
        current_llm_invocations=0,
        numeric_learning_substrate="verifier_ensemble_against_cached_candidates",
        future_dependencies=[
            dict(experiment_id=i, status="future_dependency_not_observation")
            for i in range(8430, 8443)
        ],
        compatibility_audit=work.get("compatibility_audit", []),
        methodology_note="Oracle administrative full-object custody and fixed retrospective methods only. Zero current model execution. Public human labels cannot establish independent factuality, current Qwen skill or hidden-game transfer.",
    )
    value["acceptance_gates"].update(
        complete_fourteen_objects=bool(ready), independent_generalization=False
    )
    principles = value["field_principles"]
    principles.update(
        {
            k: "Bind this field to sealed invocation evidence; administrative readiness does not establish scientific benefit."
            for k in value
            if k not in principles
        }
    )
    principles.update(
        contract_rows="One complete executable object per intended slot; rows are administrative units, not independent questions.",
        authority_snapshots="Separate actual source bytes; consumed staged authority remains absent.",
        protocol_delta="Field-by-field immutable differences identify authority repair and only the three permitted clarifications.",
        compatibility_audit="Private representative consumer findings preserve honest zero metrics; unsupported findings neither alter validators nor establish authority readiness.",
        runtime_repair_reference="Optional typed causal repair evidence; absence cannot lower independent CPU methods readiness.",
        historical_authority_limits="Imported historical failures and exposed labels limit claims without pretending new work occurred.",
        numeric_learning_substrate="Future numerical heads use cached candidates; imported models are historical provenance.",
    )
    value["reproducibility_checksum"] = checksum(value)
    return dict(value)


def replay(path: Path) -> bool:
    """Fresh reduction distrusts summary hashes and reconstructs every authority row."""
    try:
        value = json.loads(path.read_bytes())
        work = read(value["primitive_reference"])
        if work["protocol"] != protocol() or work["source_refs"] != protocol()["source_references"]:
            return False
        for ref in [
            *value["source_artifact_hashes"],
            *value["code_config_hashes"],
            *value["raw_shard_hashes"],
            *work["authority_inputs"],
            work["protocol"]["release_primitive_reference"],
        ]:
            if sha256_file(Path(ref["path"])) != ref["sha256"]:
                return False
        with TemporaryDirectory(prefix="exp8429-cold-", dir="/var/tmp") as directory:
            root = Path(directory)
            for ref in work["authority_inputs"]:
                target = root / {"design": DESIGN, "active": ACTIVE, "staged": STAGED}[ref["role"]]
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(ref["path"], target)
            actual = authority(root, root / "authority")
            actual.pop("authority_snapshots")
            expected = {k: v for k, v in work["contract"].items() if k != "authority_snapshots"}
            if json.loads(json.dumps(actual).replace(str(root), work["root"])) != expected:
                return False
        for receipt in value["validation_receipts"]:
            saved = json.loads(
                Path(receipt["stdout_path"]).with_suffix(".receipt.json").read_bytes()
            )
            if receipt != saved:
                return False
            for stream in ["stdout", "stderr"]:
                if sha256_file(Path(receipt[stream + "_path"])) != receipt[stream + "_sha256"]:
                    return False
        runtime_actual = runtime(Path(work["root"]))
        if {k: v for k, v in runtime_actual.items() if k != "inspection_duration_s"} != {
            k: v for k, v in work["runtime"].items() if k != "inspection_duration_s"
        }:
            return False
        for audit in work.get("compatibility_audit", []):
            for ref in audit["references"]:
                if sha256_file(Path(ref["path"])) != ref["sha256"]:
                    return False
        return bool(
            value
            == build(
                work,
                value["validation_receipts"],
                Path(value["primitive_reference"]["path"]).parent,
                Path(value["publication_output"]),
            )
        )
    except (OSError, ValueError, KeyError, TypeError, IndexError):
        return False
