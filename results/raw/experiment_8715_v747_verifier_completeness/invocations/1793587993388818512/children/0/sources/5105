"""REQ-REPORT-8464: sealed service operands support timing, never generalization.

Historical producers are checked through their original authority. The current
contract authorizes this task but cannot rewrite an older experiment's meaning.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory
import time
from typing import Any

from carnot.reporting import v728_contract_methods as authority
from carnot.reporting import v727_historical_inputs as historical
from carnot.reporting import transaction_attribution_8436 as shared
from carnot.reporting.current_work_receipt import (
    atomic_json,
    canonical_hash,
    ZERO_INVOCATION_COUNTS,
)
from carnot.verify import persistence_localization_8464 as k
from carnot.verify import python_transaction_cost_8407 as source

Json = dict[str, Any]
ROOT = authority.ROOT
NAME, TASK = "experiment_8464_v728_persistence_localization", "exp8464-persistence-localization"
CLI, TEST = f"scripts/experiments/{NAME}.py", "tests/python/test_persistence_localization_8464.py"
OWNED = [
    "python/carnot/verify/persistence_localization_8464.py",
    "python/carnot/reporting/persistence_localization_8464.py",
    "python/carnot/reporting/persistence_localization_runner_8464.py",
    CLI,
]
MODEL_SPECS: list[Json] = []
UPSTREAM = "results/experiment_8457_v728_contract_methods.json"
UPSTREAM_PIN = "sha256:7a4d4935da70d990d1821bc23f7b539a9bd854a649fe4c439642ea6680fb5459"
REQUIRED = shared.REQUIRED
reference, read, checksum = shared.reference, shared.read, shared.checksum


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Keep actual measurements and child waits visible to the task supervisor."""
    print(f"[exp8464] phase={phase} completed={completed} pending={pending}", flush=True)


def gate(path: Path, field: str, expected: Any, observed: Any, upstream: str = TASK) -> Json:
    """A missing input is null, while a failed measured check retains its actual value."""
    return dict(
        check=field,
        upstream=upstream,
        path=str(path),
        sha256=shared.sha256_file(path) if path.is_file() else None,
        field=field,
        operator="==",
        expected=expected,
        observed=observed,
        passed=False,
    )


def filesystem(path: Path) -> Json:
    """Choose the longest real mount match so tmpfs cannot masquerade as disk service."""
    mounts = [line.split(" - ") for line in Path("/proc/self/mountinfo").read_text().splitlines()]
    mounted = max(
        (m for m in mounts if path.resolve().is_relative_to(Path(m[0].split()[4]))),
        key=lambda m: len(m[0].split()[4]),
    )
    left, right = mounted[0].split(), mounted[1].split()
    return dict(
        mount_id=left[0],
        device=left[2],
        mount_point=left[4],
        filesystem_type=right[0],
        mount_source=right[1],
        mount_options=left[5],
        stat_device=path.stat().st_dev,
        disk_backed=right[0] not in ("tmpfs", "ramfs"),
    )


def dependencies(root: Path, raw: Path, private: Path) -> Json:
    """Use only Exp8457's independently sealed service root, without Exp8449 admission."""
    primary = read(dict(path=str(root / UPSTREAM), sha256=UPSTREAM_PIN))
    if primary.get("service_inputs_ready_score") != 1 or not primary["required_checks_passed"]:
        raise ValueError("service_inputs_ready_score")
    frozen = read(primary["primitive_reference"])
    family = next(f for f in primary["historical_family_receipts"] if f["experiment_id"] == 8407)
    ref = next(r for r in frozen["history"]["references"] if "experiment_8407_" in r["path"])
    if ref["sha256"] != "sha256:" + historical.PINS[8407]:
        raise ValueError("independent_service_pin")
    progress("historical_service_authentication_before")
    qualified = historical.qualify(8407, ref, [], private / "original_authority")
    progress("historical_service_authentication_after")
    if not qualified["qualified"] or qualified != family:
        raise ValueError("original_service_qualification")
    service = read(dict(path=ref["snapshot_path"], sha256=ref["sha256"]))
    service_work = read(service["work_reference"])
    refs = [
        reference(root / UPSTREAM),
        primary["primitive_reference"],
        dict(path=ref["snapshot_path"], sha256=ref["sha256"]),
        service["work_reference"],
        *service_work["refs"],
        *service["code_config_hashes"],
    ]
    for operand in refs:
        if shared.sha256_file(Path(operand["path"])) != operand["sha256"]:
            raise ValueError("historical_operand_hash")
    return dict(
        bundle=service_work["bundle"],
        refs=refs,
        qualification=qualified,
        historical_model_provenance=service["historical_model_provenance"],
    )


def reduce(work: Json) -> tuple[list[Json], list[Json]]:
    """Retain the registered ten pairs even when individual operands are missing."""
    rows: list[Json] = []
    warmups: list[Json] = []
    indexed = {(r["cell_index"], r["repeat"], r["arm"]): r for r in work["primitives"]}
    for index, cell in enumerate(k.CELLS):
        for repeat in range(6):
            slot: Json = dict(
                cell=cell,
                repeat=repeat,
                unit_id=f"natural-b{cell['batch']}-reuse8-pair{repeat}",
                order=["instrumented", "baseline"] if repeat % 2 else ["baseline", "instrumented"],
            )
            refs = [indexed.get((index, repeat, arm)) for arm in ("baseline", "instrumented")]
            if all(refs):
                b, i = [read(ref) for ref in refs]
                if any(p["cell"] != cell or p["repeat"] != repeat for p in (b, i)):
                    raise ValueError("registered_pair_slot_binding")
                slot.update(k.pair(work["bundle"], b, i), primitive_references=refs)
            else:
                slot.update(
                    status="failed" if slot["unit_id"] in work["owned_failures"] else "censored",
                    missing_reason=work["owned_failures"].get(
                        slot["unit_id"], "required_pair_operands_missing"
                    ),
                    absolute_metric=None,
                    complete_wall_s=None,
                    stage_ns=None,
                    missing_arms=[
                        arm
                        for arm, ref in zip(("baseline", "instrumented"), refs, strict=True)
                        if not ref
                    ],
                    primitive_references=[ref for ref in refs if ref],
                )
            (warmups if repeat == 0 else rows).append(slot)
    return rows, warmups


def measure(root: Path, raw: Path, private: Path) -> Json:
    """Admission precedes the two-cell benchmark; validation has a reserved deadline."""
    progress("preconditions_before")
    start = time.monotonic_ns()
    raw.mkdir(parents=True, exist_ok=True, mode=0o700)
    private.mkdir(parents=True, exist_ok=True)
    probe = private / "disk_probe"
    probe.write_bytes(b"exp8464 real disk readback")
    fs, private_fs = filesystem(raw), filesystem(private)
    memory = int(Path("/proc/meminfo").read_text().split("MemAvailable:")[1].split()[0]) * 1024
    resources = dict(
        disk_backed=fs["disk_backed"] and private_fs["disk_backed"],
        scratch_readback=probe.read_bytes() == b"exp8464 real disk readback",
        available_memory_bytes=memory,
        memory_available=memory > 256_000_000,
        disk_free=shutil.disk_usage(raw).free > 2_000_000_000,
        private_scratch_path=str(private),
        private_filesystem=private_fs,
        task_cap_s=4800,
        validation_reserve_s=900,
        measurement_cap_s=1800,
        tools={
            n: os.access(ROOT / ".venv/bin" / n, os.X_OK)
            for n in ("python", "pytest", "coverage", "ruff", "mypy")
        },
    )
    checked = authority.authority(root, raw / "authority")
    gates = (
        [
            gate(
                root / authority.ACTIVE,
                "activated_complete_V728_authority",
                True,
                checked["activated"],
            )
        ]
        if not checked["activated"]
        else []
    )
    task = [t for t in checked["tasks"] if t["id"] == TASK]
    if len(task) != 1 or task[0].get("MODEL_SPECS") != MODEL_SPECS:
        gates.append(gate(root / authority.ACTIVE, "exact_task_and_model_specs", TASK, task))
    for name in ("disk_backed", "scratch_readback", "memory_available", "disk_free"):
        if not resources[name]:
            gates.append(gate(private, name, True, resources[name]))
    for name, ready in resources["tools"].items():
        if not ready:
            gates.append(gate(ROOT / ".venv/bin" / name, "tool_available", True, None))
    work: Json = dict(
        root=str(root),
        raw=str(raw),
        bundle={},
        refs=[],
        primitives=[],
        owned_failures={},
        gate_check_summary=gates,
        historical_model_provenance=[],
        authority=checked,
        preconditions_checked=resources,
        filesystem_identity=fs,
        started_monotonic_ns=start,
        historical_qualification={},
        phase_spans=[],
    )
    try:
        deps = dependencies(root, raw, private)
        work.update(deps)
    except (OSError, ValueError, KeyError, TypeError, StopIteration) as error:
        gates.append(
            gate(
                root / UPSTREAM,
                "sealed_original_8407_service_family",
                True,
                str(error) if (root / UPSTREAM).exists() else None,
                "exp8457-contract-methods",
            )
        )
    progress("preconditions_after")
    tape = dict(
        cells=k.CELLS,
        event_tape=source.cost_tape(8),
        journal_checkpoint_every_committed_transactions=32,
        final_drain_checkpoint=True,
        delay_ticks=8,
        inputs=5,
        max_state_bytes=source.s.MAX_BYTES,
        max_events=source.s.MAX_EVENTS,
        journal_ceiling_bytes=16 * 1024 * 1024,
        candidate_journal_work_units=[
            dict(
                cell=cell,
                transactions=len(source.batches(source.cost_tape(8), cell["batch"])),
                events=72,
                predictions=64,
                feedback=8,
            )
            for cell in k.CELLS
        ],
    )
    atomic_json(raw / "frozen_workload.json", tape)
    work["workload_reference"] = reference(raw / "frozen_workload.json")
    if not gates:
        for index, cell in enumerate(k.CELLS):
            for repeat in range(6):
                for arm in (
                    ["instrumented", "baseline"] if repeat % 2 else ["baseline", "instrumented"]
                ):
                    progress(
                        "benchmark_before_" + arm,
                        len(work["primitives"]),
                        24 - len(work["primitives"]),
                    )
                    if (time.monotonic_ns() - start) / 1e9 >= 1800:
                        work["owned_failures"][f"natural-b{cell['batch']}-reuse8-pair{repeat}"] = (
                            "measurement_deadline"
                        )
                        continue
                    destination = raw / "repeats" / f"cell{index}-repeat{repeat}-{arm}"
                    try:
                        result = k.run_repeat(
                            work["bundle"], cell, repeat, destination, arm == "instrumented"
                        )
                        k.reduce(work["bundle"], result)
                        atomic_json(destination / "primitive.json", result)
                        work["primitives"].append(
                            dict(
                                reference(destination / "primitive.json"),
                                cell_index=index,
                                repeat=repeat,
                                arm=arm,
                            )
                        )
                    except (OSError, ValueError, KeyError, TypeError, TimeoutError) as error:
                        work["owned_failures"][f"natural-b{cell['batch']}-reuse8-pair{repeat}"] = (
                            str(error)
                        )
                    progress(
                        "benchmark_after_" + arm,
                        len(work["primitives"]),
                        24 - len(work["primitives"]),
                    )
    work["rows"], work["warmups"] = reduce(work)
    expected = []
    for ref in work["primitives"]:
        if ref["arm"] == "baseline":
            original = read(ref)
            expected.append(
                dict(
                    cell=original["cell"],
                    repeat=original["repeat"],
                    repeat_id=original["repeat_id"],
                    state_hash=original["state_hash"],
                    response_hash=original["response_hash"],
                    request_rows=[
                        {
                            key: value
                            for key, value in row.items()
                            if key not in ("start_ns", "end_ns")
                        }
                        for row in original["rows"]
                    ],
                    transactions=[
                        dict(
                            transaction_id=tx["transaction_id"],
                            request_ids=tx["request_ids"],
                            state_hash=tx["state_hash"],
                            response_hash=response_hash,
                            bytes=tx["bytes"],
                            fsync_count=tx["fsync_count"],
                        )
                        for tx, response_hash in zip(
                            original["transactions"], original["response_hashes"], strict=True
                        )
                    ],
                )
            )
    tape["expected_state_response_hashes"] = expected
    atomic_json(raw / "frozen_workload.json", tape)
    work["workload_reference"] = reference(raw / "frozen_workload.json")
    work["code_config_hashes"] = [reference(ROOT / path) for path in [*OWNED, TEST, *shared.GUARDS]]
    work["ended_monotonic_ns"] = time.monotonic_ns()
    work["phase_spans"].append(
        dict(phase="admission_and_measurement", start_ns=start, end_ns=work["ended_monotonic_ns"])
    )
    atomic_json(raw / "measurement.json", work)
    progress("measurement_complete", len(work["primitives"]), 24 - len(work["primitives"]))
    return work


def build(work: Json, receipts: list[Json], output: Path) -> Json:
    """Qualification depends on complete clocks and validation, never a desired bottleneck."""
    rows, warmups = work["rows"], work["warmups"]
    valid = [r for r in rows if r["status"] == "completed"]
    passed = bool(receipts) and all(r["passed"] for r in receipts)
    required = REQUIRED <= {r.get("name") for r in receipts if r["passed"]}
    complete = len(valid) == 10 and len(work["primitives"]) == 24
    report = work.get("owned_coverage", {})
    covered = coverage_ready(report)
    flags = work.get("adversarial_findings", [])
    qualified = passed and required and covered and not flags
    verdict = (
        "disqualified"
        if not passed or flags or work["owned_failures"]
        else "blocked"
        if work["gate_check_summary"]
        else "null"
        if complete and qualified
        else "disqualified"
    )
    stage_rows = [
        dict(
            unit_id=r["unit_id"],
            stage=name,
            duration_s=r["stage_ns"][name] / 1e9,
            complete_wall_s=r["complete_wall_s"],
        )
        for r in valid
        for name in k.STAGES
    ]
    value: Json = dict(
        experiment_id=8464,
        task_id=TASK,
        milestone="2026.10.728",
        run_date="20261012",
        honest_verdict="complete_" + verdict + "_persistence_localization",
        verdict_class=verdict,
        gate_check_summary=work["gate_check_summary"],
        inference_substrate="verifier_ensemble_against_cached_candidates",
        reader_substrate="aggregation_from_upstream_artifacts",
        inference_substrate_class="no_model_load",
        MODEL_SPECS=[],
        model_invocation_counts=dict(ZERO_INVOCATION_COUNTS, current_llm_invocations=0),
        historical_model_provenance=work["historical_model_provenance"],
        rows=rows,
        intended_count=10,
        completed_count=len(valid),
        failed_count=sum(r["status"] == "failed" for r in rows),
        censored_count=sum(r["status"] == "censored" for r in rows),
        excluded_count=0,
        independent_count=0,
        sample_size_budget=dict(
            cells=2,
            measured_pairs_per_cell=5,
            warmups_per_cell=1,
            arms_per_pair=2,
            repeat_source_multiplication=0,
        ),
        verifier_is_oracle=False,
        exposure_scope="retrospective_cached_five_input_engineering_service",
        independent_generalization_score=0,
        generalized_learning_benefit_score=0,
        required_checks_passed=passed and required,
        flagged_adversarial=bool(flags) or not passed,
        acceptance_gates=dict(
            complete_clocks=complete,
            original_causal_semantics=complete,
            exact_stage_nesting=complete,
            scoped_validation=required,
            owned_coverage=covered,
            authentic_operands=not work["gate_check_summary"],
            no_unresolved_findings=not flags,
        ),
        validation_receipts=receipts,
        terminal_validation_sidecar_path=str(Path(work["raw"]) / "terminal_validation.json"),
        adversarial_findings=flags,
        preconditions_checked=work["preconditions_checked"],
        duration_s=(work["ended_monotonic_ns"] - work["started_monotonic_ns"]) / 1e9,
        phase_spans=work["phase_spans"],
        random_seed=7288464,
        source_artifact_hashes=work["refs"],
        code_config_hashes=work["code_config_hashes"],
        raw_shard_hashes=work["primitives"]
        + work.get("coverage_shards", [])
        + [work["workload_reference"]],
        cited_upstream_artifacts=[
            dict(
                ref,
                imported_fields="sealed original authority, head, slots, labels and qualification",
            )
            for ref in work["refs"]
        ],
        persistence_profile_ready_score=int(verdict == "null"),
        stage_rows=stage_rows,
        paired_repeat_rows=rows,
        warmups=warmups,
        instrumentation_overhead=[
            dict(
                unit_id=r["unit_id"],
                seconds=r["instrumentation_overhead_s"],
                fraction=r["instrumentation_overhead_fraction"],
                subtracted=False,
            )
            for r in valid
        ],
        workload_reference=work["workload_reference"],
        filesystem_identity=work["filesystem_identity"],
        complete_wall_s=sum(r["complete_wall_s"] for r in valid) if valid else None,
        work_reference=reference(Path(work["raw"]) / "measurement.json"),
        owned_coverage=report,
        historical_qualification=work.get("qualification", {}),
        closed_exposed_utility=dict(H1=-0.00390625, H2=0, reopened=False),
        preserved_history=dict(
            V727_methods_ready_score=0,
            V727_science_executed=False,
            V726_admission_learning="null",
            V726_kernel_percent=0.093,
            V726_persistence_percent=96.44,
        ),
        frozen_expected_hashes=[
            dict(unit_id=r["unit_id"], state_hash=r["state_hash"], response_hash=r["response_hash"])
            for r in [*warmups, *valid]
            if r["status"] == "completed"
        ],
        future_dependencies=[
            "Exp8465 journal qualification",
            "Exp8451 frozen work units",
            "Exp8466 complete cost",
        ],
        methodology_note="Original five-input natural Exp8407 service; two cells, one warmup and five paired repeats, whole clocks from startup through durable feedback tail. Original baselines retain instrumentation overhead without subtraction. Serialization and filesystem operations have separate disjoint leaves. Gaps and nonpersistence work remain explicit orchestration. No accelerator benefit, generator calls or generalization measured.",
    )
    value["field_principles"] = {
        key: "Bind to immutable invocation operands; timing qualification establishes no benefit or independent generalization."
        for key in value
    }
    value["field_principles"].update(
        persistence_profile_ready_score="Authentic complete paired clocks and unchanged causal semantics with exact stage nesting and all required checks; favorable bottleneck is unnecessary.",
        stage_rows="Disjoint leaf intervals inside each durable commit; outer gaps, Python scoring, snapshot reads and response encoding remain orchestration. State copy count is zero; validation is the original byte-budget check.",
        instrumentation_overhead="Instrumented whole clock minus paired original baseline, including negative host noise; never silently subtracted.",
        filesystem_identity="Actual longest matching Linux mount, device and disk-backed readback for preserved snapshot files.",
    )
    value["reproducibility_checksum"] = checksum(value)
    return value


def coverage_ready(report: Json) -> bool:
    """Read actual Coverage.py bytes, so an asserted percentage cannot grant readiness."""
    if not report:
        return False
    try:
        original = read(report["reference"])
        return bool(
            report == dict(original["totals"], reference=report["reference"])
            and original["totals"]["missing_lines"] == 0
            and original["totals"]["percent_covered"] == 100
            and all(
                original["files"][path]["summary"]["num_statements"] > 0
                and original["files"][path]["summary"]["missing_lines"] == 0
                for path in OWNED
            )
        )
    except (OSError, ValueError, KeyError, TypeError):
        return False


def replay(path: Path) -> bool:
    """Fresh readers reject changed aggregates even when the candidate checksum is repaired."""
    try:
        value = json.loads(path.read_bytes())
        if value["experiment_id"] != 8464 or value["reproducibility_checksum"] != checksum(value):
            return False
        work = read(value["work_reference"])
        if work["authority"]["activated"]:
            if authority.tasks_digest(work["authority"]["tasks"]) != authority.TASKS_PIN:
                return False
        if work["bundle"]:
            with TemporaryDirectory(dir="/var/tmp", prefix="exp8464-cold-") as folder:
                deps = dependencies(Path(work["root"]), Path(folder), Path(folder))
            if deps["bundle"] != work["bundle"]:
                return False
        for ref in [
            *value["source_artifact_hashes"],
            *value["code_config_hashes"],
            *value["raw_shard_hashes"],
        ]:
            if shared.sha256_file(Path(ref["path"])) != ref["sha256"]:
                return False
        rows, warmups = reduce(work)
        if rows != work["rows"] or warmups != work["warmups"]:
            return False
        return bool(value == build(work, value["validation_receipts"], path))
    except (OSError, ValueError, KeyError, TypeError, StopIteration):
        return False
