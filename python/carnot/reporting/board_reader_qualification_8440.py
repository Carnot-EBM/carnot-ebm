"""REQ-REPORT-8440: separate reader correctness from installed board capability.

Independent historical families supply their own immutable authority bytes.
Current costs are reduced outside those aliases so old absences stay historical.
"""

from __future__ import annotations

from contextlib import contextmanager
import json
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory
import time
from typing import Any, Iterator
from unittest.mock import patch

import yaml

from carnot.reporting import board_operation_boundary_8413 as old
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash
from carnot.reporting.primary_publication import read_bound_sidecar, validate_primary
from carnot.reporting.roadmap_contract import parse_design
from carnot.reporting.v685_authority_lifecycle import assess_authorities, tasks_digest

Json = dict[str, Any]
ROOT, base = old.ROOT, old.base
NAME, TASK, MILESTONE = (
    "experiment_8440_v726_board_reader_qualification",
    "exp8440-board-reader-qualification",
    "2026.10.726",
)
CLI, TEST = f"scripts/experiments/{NAME}.py", "tests/python/test_board_reader_qualification_8440.py"
OWNED = [
    "python/carnot/reporting/board_reader_qualification_8440.py",
    "python/carnot/reporting/board_reader_runner_8440.py",
    CLI,
]
MODEL_SPECS: list[Json] = []
TASKS_PIN = "72f4cca447651e88180fd3196f947615d653b04cc7280ff8f12fd8f5eaedaa88"
FAMILIES = {
    723: (old.READER, old.READER_PIN),
    724: (
        "results/experiment_8413_v724_board_operation_boundary.json",
        "sha256:ab4bd6cc49f95cc4c207cb35c0a721b8b2569e3c22d4e3c596dcbfadd887bf50",
    ),
}
OPTIONAL = "results/experiment_8437_v726_acceleration_ceiling.json"
OPTIONAL_PIN = "sha256:01c273a37be2749e94e07589d9224ea2c8bc9a7207420e4a75cb0719d4aa9a6f"
OPERATION_GROUPS = dict(
    gathering=["input_validation", "snapshot_pinning"],
    scoring=["direct_scoring"],
    updates=["coefficient_update", "delayed_release", "scheduling"],
    copies=["python_boundary_conversion"],
    serialization=["temporary_write"],
    file_fsync=["file_fsync"],
    directory_fsync=["directory_fsync"],
    acknowledgment=["acknowledgment", "response", "pointer_publication", "pointer_replace"],
    transport=["transfers"],
    calibration=["calibration"],
    transaction_overhead=["transaction_overhead"],
)
gate = failure = old.gate


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Flush actual counts so a bounded child cannot appear silently stalled."""
    print(f"[exp8440] phase={phase} completed={completed} pending={pending}", flush=True)


def family(version: int, root: Path = ROOT) -> Json:
    """A fixed producer hash, rather than a rehashed manifest, authorizes aliases."""
    name, digest = FAMILIES[version]
    return dict(json.loads(old.authority.checked(root / name, digest).read_bytes()))


@contextmanager
def historical_bindings(version: int) -> Iterator[None]:
    """Each original consumer reads its own design and active bytes without test edits."""
    value = family(version)
    with old.prior.prior.source_aliases(value["source_artifact_hashes"]):
        yield


def authority(root: Path, raw: Path) -> Json:
    """Compare all fourteen full task objects and the independent canonical digest."""
    paths = [root / n for n in [old.authority.DESIGN, old.authority.STAGED, old.authority.ACTIVE]]
    try:
        tasks = parse_design(paths[0].read_text(), milestone=MILESTONE)[1]
        result = dict(assess_authorities(*paths, raw, milestone=MILESTONE, first_id=8429, count=14))
        active = yaml.safe_load(paths[2].read_bytes())["tasks"]
        matched = active == tasks and tasks_digest(tasks) == TASKS_PIN
        result["activated"] = result["activated"] and matched
        result["tasks"] = tasks
        result["gate_check_summary"].append(gate(paths[2], "full_task_objects", True, matched))
        return result
    except (OSError, ValueError, KeyError, TypeError) as error:
        return dict(
            activated=False,
            tasks=[],
            canonical_tasks_sha256=TASKS_PIN,
            gate_check_summary=[
                gate(
                    paths[0],
                    "required_authority",
                    True,
                    None if isinstance(error, FileNotFoundError) else str(error),
                )
            ],
        )


def operation_mapping(rows: list[Json]) -> list[Json]:
    """Missing members keep a grouped clock null, even when other members were measured."""
    old.operation_map(rows)
    clocks = {row["operation"]: row["cost_ns"] for row in rows}
    return [
        dict(
            operation=name,
            primitive_stages=members,
            assigned_substrate="host_CPU",
            kv260_supported=False,
            cost_ns=sum(clocks[m] for m in members) if all(m in clocks for m in members) else None,
            missing_stages=[m for m in members if m not in clocks],
        )
        for name, members in OPERATION_GROUPS.items()
    ]


def optional(root: Path, raw: Path, work: Json) -> Json:
    """Optional current ceilings require terminal and primitive hashes, never a headline alone."""
    path = root / OPTIONAL
    result: Json = dict(path=str(path), present=path.is_file(), qualified=False, evidence=None)
    if not path.is_file():
        return result
    try:
        value = json.loads(
            base.pin(old.authority.checked(path, OPTIONAL_PIN), raw, work["refs"]).read_bytes()
        )
        validate_primary(value, path)
        terminal = json.loads(
            base.pin(
                Path(value["terminal_validation_sidecar_path"]), raw, work["refs"]
            ).read_bytes()
        )
        sidecar = Path(terminal["publication"]["sidecar_path"])
        bound = read_bound_sidecar(path, sidecar)
        base.pin(sidecar, raw, work["refs"])
        for ref in value["raw_shard_hashes"]:
            base.checked(ref)
        if not (
            value["required_checks_passed"]
            and not value["flagged_adversarial"]
            and bound["report"]["passed"]
            and value["acceleration_analysis_ready_score"] == 1
        ):
            raise ValueError("unqualified_optional_ceiling")
        result.update(
            qualified=True,
            evidence={
                k: value[k]
                for k in [
                    "target_comparison",
                    "measured_primary_composite_stages_s",
                    "bound_assumptions",
                ]
            },
        )
    except (OSError, ValueError, KeyError, TypeError) as error:
        result["missing_reason"] = str(error)
    return result


def operands(root: Path, raw: Path) -> Json:
    """Rebuild history with independent family roots, then read current costs separately."""
    current = authority(root, raw / "current-authority")
    try:
        history = family(724, root)
        with old.prior.prior.source_aliases(history["source_artifact_hashes"]):
            work = old.operands(root, raw)
        for ref in history["source_artifact_hashes"]:
            base.pin(base.checked(ref), raw, work["refs"])
    except (OSError, ValueError, KeyError, TypeError):
        work = old.operands(root, raw)
    work["contract"] = dict(
        activated=current["activated"],
        canonical_tasks_sha256=current["canonical_tasks_sha256"],
        task_sha256=canonical_hash(next((t for t in current["tasks"] if t["id"] == TASK), {})),
    )
    work["checks"] = [
        g
        for g in work["checks"]
        if g["artifact_field"]
        not in {
            "independent_design_contract",
            "active_task_authority",
            "cost_primary_present",
            "primitive_cost_reduction",
            "exists",
        }
    ]
    work["checks"].extend(current["gate_check_summary"])
    work["checks"].append(
        gate(root / old.authority.ACTIVE, "V726_activation", True, current["activated"])
    )
    for name in [old.authority.DESIGN, old.authority.ACTIVE, old.authority.STAGED]:
        if (root / name).is_file():
            base.pin(root / name, raw, work["refs"])
    work["branches"] = [old.cost_branch(root / "results" / (old.SOURCES[0] + ".json"), raw, work)]
    work["optional_ceiling"] = optional(root, raw, work)
    work["absent"] = [b["path"] for b in work["branches"] if not b["present"]]
    if not work["optional_ceiling"]["present"]:
        work["absent"].append(str(root / OPTIONAL))
    return dict(work)


def measure(root: Path, raw: Path) -> Json:
    """Authenticate before work and seal logs, resources and source bytes for cold replay."""
    started = time.monotonic_ns()
    raw.mkdir(parents=True, exist_ok=True, mode=0o700)
    progress("preconditions_before")
    mounts = [line.split() for line in Path("/proc/self/mountinfo").read_text().splitlines()]
    mount = max(
        (m for m in mounts if raw.resolve().is_relative_to(Path(m[4]))), key=lambda m: len(m[4])
    )
    memory = dict(line.split(":", 1) for line in Path("/proc/meminfo").read_text().splitlines())
    resources = dict(
        filesystem=mount[mount.index("-") + 1],
        disk_backed=mount[mount.index("-") + 1] not in {"tmpfs", "ramfs"},
        memory_available_bytes=int(memory["MemAvailable"].split()[0]) * 1024,
        free_disk_bytes=shutil.disk_usage(raw).free,
        task_cap_s=4800,
        validation_reserve_s=900,
        tool_paths={
            n: shutil.which(str(ROOT / ".venv/bin" / n))
            for n in ["python", "pytest", "coverage", "ruff", "mypy"]
        },
        no_model_load=True,
    )
    work = operands(root, raw)
    historical = family(724)
    cov = json.loads(
        base.pin(
            base.checked(historical["owned_coverage_reference"]), raw, work["refs"]
        ).read_bytes()
    )
    missing = {p: row["missing_lines"] for p, row in cov["files"].items() if row["missing_lines"]}
    logs = [r for r in historical["validation_receipts"] if not r["passed"]]
    for receipt in logs:
        for stream in ["stdout", "stderr"]:
            base.pin(
                base.checked(
                    dict(path=receipt[stream + "_path"], sha256=receipt[stream + "_sha256"])
                ),
                raw,
                work["refs"],
            )
    work["operand_check_count"] = len(work["checks"])
    work["history_defects"] = dict(
        missing_statements=missing,
        failed_receipts=logs,
        reader_assertion="tests/python/test_board_operation_boundary_8399.py:123",
    )
    work["preconditions"] = resources
    for name, passed in [
        ("disk_backed_scratch", work["preconditions"]["disk_backed"]),
        (
            "available_memory_at_least_1GiB",
            work["preconditions"]["memory_available_bytes"] >= 1024**3,
        ),
        ("available_disk_at_least_1GiB", work["preconditions"]["free_disk_bytes"] >= 1024**3),
        ("tools_available", all(work["preconditions"]["tool_paths"].values())),
    ]:
        work["checks"].append(gate(raw, name, True, bool(passed)))
    work.update(
        pre_gate_reference=None,
        started_monotonic_ns=started,
        ended_monotonic_ns=time.monotonic_ns(),
        code_config_hashes=[
            base.reference(ROOT / p)
            for p in [
                *OWNED,
                TEST,
                *old.OWNED,
                "python/carnot/reporting/primary_publication.py",
                "scripts/adversarial_verify.py",
                "scripts/verdict_row_consistency_lint.py",
                "scripts/experiment_template.py",
                "docs/research-notes/v726-board-operations.md",
                "ops/exclusion_manifest.yaml",
            ]
        ],
    )
    atomic_json(raw / "measurement.json", work)
    progress("measurement_sealed", 1, 0)
    return work


def build(work: Json, receipts: list[Json], raw: Path, output: Path) -> Json:
    """Administrative readiness cannot upgrade unsupported kernels or closed scientific results."""
    with patch.multiple(
        old, NAME=NAME, TASK=TASK, MILESTONE=MILESTONE, CLI=CLI, TEST=TEST, OWNED=OWNED
    ):
        value = old.build(work, receipts, raw, output)
    clocks = work["branches"][0]
    rows = clocks["rows"]
    total = clocks.get("complete_wall_ns")
    arithmetic = sum(
        r["cost_ns"] for r in rows if r["operation"] in {"direct_scoring", "coefficient_update"}
    )
    value.update(
        experiment_id=8440,
        run_date="20261012",
        random_seed=7268440,
        honest_verdict="complete_" + value["verdict_class"] + "_v726_board_reader_qualification",
        inference_substrate_class="no_model_load",
        historical_model_provenance=work["history"].get("historical_model_provenance", []),
        coverage_missing_before_after=dict(
            before=work["history_defects"]["missing_statements"],
            after=work.get("historical_missing_after"),
        ),
        historical_consumer_receipts=[
            r
            for r in receipts
            if "historical" in r.get("name", "") or "original" in r.get("name", "")
        ],
        operation_mapping=operation_mapping(rows),
        measured_end_to_end_cost_ns=total,
        ideal100x_arithmetic_bound=total / (total - arithmetic + arithmetic / 100)
        if total
        else None,
        optional_current_cost_evidence=work["optional_ceiling"],
        kv260_supported_operations=[
            dict(
                operation="quadratic_Ising_energy", k_max=5, scope="authenticated_historical_fabric"
            )
        ],
        device_commands_executed=[],
        hardware_reopening_conditions=[
            "exact_installed_compatible_kernel_and_firmware_hash",
            "matched_input_output_parity",
            "complete_request_response_bytes_and_transport_clocks",
            "equivalent_durable_transaction_clocks",
            "exclusive_board_lease",
            "future_access_ssh_kria_and_xmutil",
        ],
        historical_missing_v722_branch="independent_design_unavailable; not_required_for_this_reader",
        history_defects=work["history_defects"],
        newly_owned_coverage=work.get("owned_coverage"),
        methodology_note="Authenticate independent V723 and V724 roots, retain original logs and assertions, reduce qualified Exp8407 costs through the shipped primitive reducer, and optionally import sealed Exp8437 bounds. CPU costs and the conditional ideal100x bound grant no board acceleration, factuality, current Qwen skill or hidden-game generalization. No model load or device command runs.",
    )
    ceiling = work["optional_ceiling"]
    value["rows"].append(
        dict(
            source_id="experiment_8437_optional_ceiling",
            arm="optional_current_bound",
            intended=1,
            completed=ceiling["qualified"],
            failed=False,
            censored=not ceiling["qualified"],
            excluded=False,
            independent=0,
            numerator=int(ceiling["qualified"]),
            denominator=1,
            missing_reason=None
            if ceiling["qualified"]
            else ceiling.get("missing_reason", "optional_evidence_absent"),
        )
    )
    value.update(
        intended_count=len(value["rows"]),
        completed_count=sum(r["completed"] for r in value["rows"]),
        censored_count=sum(r["censored"] for r in value["rows"]),
        sample_size_budget=dict(
            required_cost_branches=1,
            optional_ceiling_branches=1,
            historical_boards=2,
            independent_deployment_samples=0,
        ),
    )
    value["model_invocation_counts"]["current_llm_invocations"] = 0
    for check in value["gate_check_summary"]:
        check.update(
            check=check["artifact_field"], field=check["artifact_field"], operator=check["op"]
        )
    value["raw_shard_hashes"].extend(work.get("coverage_shards", []))
    value["field_principles"].update(
        {
            k: "Bind exact task, producer hashes and primitive operands; qualification grants no hardware or scientific benefit."
            for k in value
        }
    )
    value["reproducibility_checksum"] = canonical_hash(
        {k: v for k, v in value.items() if k != "reproducibility_checksum"}
    )
    return dict(value)


def replay(path: Path) -> bool:
    """Fresh reductions reject summary and operand changes even after their hashes are repaired."""
    try:
        value = json.loads(path.read_bytes())
        work = json.loads(base.checked(value["work_reference"]).read_bytes())
        for ref in [
            *value["source_artifact_hashes"],
            *value["code_config_hashes"],
            *value["raw_shard_hashes"],
        ]:
            base.checked(ref)
        for receipt in value["validation_receipts"]:
            for stream in ["stdout", "stderr"]:
                if stream + "_path" in receipt:
                    base.checked(
                        dict(path=receipt[stream + "_path"], sha256=receipt[stream + "_sha256"])
                    )
        with TemporaryDirectory(prefix="exp8440-replay-", dir=path.parent) as folder:
            with old.prior.prior.source_aliases(work["refs"], work["absent"]):
                actual = operands(Path(work["root"]), Path(folder))
            if any(
                actual[k] != work[k]
                for k in [
                    "contract",
                    "reader_ready",
                    "history",
                    "branches",
                    "optional_ceiling",
                    "absent",
                ]
            ):
                return False
        if actual["checks"] != work["checks"][: work["operand_check_count"]]:
            return False
        original = family(724)
        coverage = json.loads(base.checked(original["owned_coverage_reference"]).read_bytes())
        missing = {
            p: r["missing_lines"] for p, r in coverage["files"].items() if r["missing_lines"]
        }
        if work["history_defects"]["missing_statements"] != missing:
            return False
        rebuilt = build(
            work,
            value["validation_receipts"],
            Path(value["work_reference"]["path"]).parent,
            Path(value["publication_output"]),
        )
        return bool(rebuilt == value)
    except (OSError, ValueError, KeyError, TypeError):
        return False
