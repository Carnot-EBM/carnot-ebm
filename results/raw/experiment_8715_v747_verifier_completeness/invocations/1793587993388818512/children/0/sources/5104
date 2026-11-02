"""REQ-REPORT-8449: bind paired persistence evidence to sealed service inputs.

Required authority can block measurement without erasing the intended roster.
This engineering profile measures one host. It makes no model quality claim.
"""

from __future__ import annotations

from copy import deepcopy
import json
import os
from pathlib import Path
import shutil
import sys
import time
from typing import Any

from carnot.reporting import v727_contract_methods as authority
from carnot.reporting import v727_historical_inputs as history
from carnot.reporting.current_work_receipt import (
    atomic_json,
    canonical_hash,
    sha256_file,
    ZERO_INVOCATION_COUNTS,
)
from carnot.reporting.v709_execution import child
from carnot.verify import persistence_localization_8449 as k

Json = dict[str, Any]
ROOT = authority.ROOT
NAME, TASK = "experiment_8449_v727_persistence_localization", "exp8449-persistence-localization"
CLI, TEST = f"scripts/experiments/{NAME}.py", "tests/python/test_persistence_localization_8449.py"
OWNED = [
    "python/carnot/verify/persistence_localization_8449.py",
    "python/carnot/reporting/persistence_localization_8449.py",
    "python/carnot/reporting/persistence_localization_runner_8449.py",
    CLI,
]
PRODUCER = "results/experiment_8407_v724_python_transaction_cost.json"
MODEL_SPECS: list[Json] = []
reference = authority.reference
history_worker = history.worker
REQUIRED = {
    "owned_tests",
    "private_E2E018_consumers",
    "private_E2E021",
    "coverage_combine",
    "coverage_report",
    "coverage_json",
    "ruff_check",
    "ruff_format",
    "strict_mypy",
    "spec_coverage",
}


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Flush phase boundaries and actual counts without any model invocation."""
    print(f"[exp8449] phase={phase} completed={completed} pending={pending}", flush=True)


def checksum(value: Json) -> str:
    return str(
        canonical_hash(
            {key: item for key, item in value.items() if key != "reproducibility_checksum"}
        )
    )


def read(ref: Json) -> Json:
    """Check exact input bytes before interpreting their metrics or state."""
    path = Path(ref["path"])
    if sha256_file(path) != ref["sha256"]:
        raise ValueError("input_hash")
    return dict(json.loads(path.read_bytes()))


def historical(
    root: Path, raw: Path, private: Path
) -> tuple[Json, list[Json], list[Json], list[Json]]:
    """Reuse original complete-cost authentication; mutable V727 is never its authority."""
    path = root / PRODUCER
    refs: list[Json] = []
    try:
        primary = dict(path=str(path), sha256="sha256:" + history.PINS[8407])
        value = read(primary)
        refs.append(primary)
        frozen = history.cap.freeze(path, raw / "original_producer")
        request, output = raw / "history_request.json", raw / "history_result.json"
        atomic_json(request, dict(number=8407, primary=frozen, authority=[]))
        receipt = child(
            "original_service_authority",
            [
                sys.executable,
                "-u",
                str(ROOT / CLI),
                "--history-request",
                str(request),
                "--history-output",
                str(output),
            ],
            raw / "history_logs",
            deadline=240,
        )
        qualified = read(reference(output)) if receipt["passed"] else {}
        if not qualified.get("qualified"):
            raise ValueError("original_service_authority")
        work = read(value["work_reference"])
        refs.extend([frozen, value["work_reference"], work["bundle"]["labels"]])
        for ref in refs:
            if sha256_file(Path(ref["path"])) != ref["sha256"]:
                raise ValueError("original_operand_hash")
        provenance = [
            dict(
                experiment_id=x.get("experiment_id"),
                MODEL_SPECS=x.get("MODEL_SPECS"),
                imported_counts=x.get("imported_counts"),
                original_producer=primary,
            )
            for x in value.get("historical_model_provenance", [])
        ]
        return work["bundle"], refs, [], provenance
    except (OSError, ValueError, KeyError, TypeError) as error:
        return (
            {},
            refs,
            [
                dict(
                    authority.legacy.failure(
                        path,
                        "original_service_authority",
                        True,
                        None if not path.is_file() else str(error),
                    ),
                    upstream_id="exp8407-python-transaction-cost",
                )
            ],
            [],
        )


def roster() -> list[Json]:
    """Warmups remain distinct from the twenty intended measured arm summaries."""
    return [
        dict(
            cell=cell,
            repeat=repeat,
            arm=arm,
            status="censored",
            complete_wall_s=None,
            missing_reason="required_authority_or_operand_absent",
            warmup=repeat == 0,
        )
        for cell in k.CELLS
        for repeat in range(6)
        for arm in (k.ARMS if repeat % 2 == 0 else tuple(reversed(k.ARMS)))
    ]


def freeze_workload(slots: list[Json]) -> Json:
    """Freeze exact causal events and journal response/state hashes for later consumers."""
    frozen: Json = dict(
        event_tape=k.cost_tape(8),
        inputs=5,
        cells=k.CELLS,
        original_producer_sha256="sha256:" + history.PINS[8407],
        expected_units=[],
    )
    for slot in slots:
        if slot["status"] == "completed":
            primitive = read(slot["primitive_reference"])
            frozen["expected_units"].append(
                dict(
                    cell=slot["cell"],
                    repeat=slot["repeat"],
                    arm=slot["arm"],
                    final_state_hash=slot["final_state_hash"],
                    response_hashes=slot["response_hashes"],
                    candidate_journal_work_units=[
                        dict(
                            event={key: row[key] for key in event},
                            request_id=row["request_id"],
                            expected_state_hash=row["after_state_hash"],
                            expected_response_hash=canonical_hash(
                                {
                                    key: item
                                    for key, item in row.items()
                                    if key not in ("start_ns", "end_ns", "transaction_id")
                                }
                            ),
                        )
                        for event, row in zip(frozen["event_tape"], primitive["rows"], strict=True)
                    ],
                )
            )
    frozen["missing_reason"] = (
        None if len(frozen["expected_units"]) == 24 else "incomplete_or_blocked_measurement"
    )
    return frozen


def measure(root: Path, raw: Path, private: Path) -> Json:
    """Check private storage and required authority before admitting any benchmark child."""
    began = time.monotonic_ns()
    progress("preconditions_before")
    raw.mkdir(parents=True, exist_ok=True, mode=0o700)
    probe = private / "disk_probe"
    probe.write_bytes(b"exp8449 disk-backed scratch")
    mounts = [line.split(" - ") for line in Path("/proc/self/mountinfo").read_text().splitlines()]
    mount = max(
        (m for m in mounts if private.resolve().is_relative_to(Path(m[0].split()[4]))),
        key=lambda m: len(m[0].split()[4]),
    )
    memory = int(Path("/proc/meminfo").read_text().split("MemAvailable:")[1].split()[0]) * 1024
    resources = dict(
        disk_backed=mount[1].split()[0] not in ("tmpfs", "ramfs"),
        private_scratch=probe.read_bytes() == b"exp8449 disk-backed scratch",
        private_permissions=private.stat().st_mode & 0o077 == 0,
        disk_free=shutil.disk_usage(private).free > 2_000_000_000,
        memory_available=memory > 256_000_000,
        available_memory_bytes=memory,
        task_cap_s=4800,
        measurement_cap_s=1800,
        validation_reserve_s=900,
        required_tool_hashes=[
            reference(ROOT / ".venv/bin" / name)
            for name in ("python", "pytest", "coverage", "ruff", "mypy")
            if (ROOT / ".venv/bin" / name).is_file()
        ],
        filesystem_identity=dict(
            mountinfo=" - ".join(mount),
            filesystem_type=mount[1].split()[0],
            device_id=private.stat().st_dev,
            scratch_path=str(private),
        ),
    )
    gates = [
        dict(authority.legacy.failure(private, name, True, resources[name]), upstream_id=TASK)
        for name in (
            "disk_backed",
            "private_scratch",
            "private_permissions",
            "disk_free",
            "memory_available",
        )
        if not resources[name]
    ]
    for name in ("python", "pytest", "coverage", "ruff", "mypy"):
        if not os.access(ROOT / ".venv/bin" / name, os.X_OK):
            gates.append(
                dict(
                    authority.legacy.failure(
                        ROOT / ".venv/bin" / name, "executable_available", True, None
                    ),
                    upstream_id=TASK,
                )
            )
    active = authority.authority(root, raw / "authority")
    gates.extend(active["gate_check_summary"])
    tasks = [task for task in active["tasks"] if task["id"] == TASK]
    if (
        len(tasks) != 1
        or tasks[0].get("MODEL_SPECS") != []
        or tasks[0].get("deliverable") != "results/" + NAME + ".json"
    ):
        gates.append(
            dict(
                authority.legacy.failure(
                    root / authority.ACTIVE, "exact_current_task", TASK, tasks
                ),
                upstream_id=TASK,
            )
        )
    bundle, refs, historical_gates, provenance = historical(root, raw, private)
    gates.extend(historical_gates)
    refs.extend(
        dict(path=snapshot["snapshot_path"], sha256=snapshot["sha256"])
        for snapshot in active["authority_snapshots"].values()
        if snapshot.get("exists")
    )
    slots = roster()
    work: Json = dict(
        raw=str(raw),
        root=str(root),
        started_monotonic_ns=began,
        gate_check_summary=gates,
        bundle=bundle,
        refs=refs,
        authority=active,
        historical_model_provenance=provenance,
        preconditions_checked=resources,
        code_config_hashes=[
            reference(ROOT / path)
            for path in [
                *OWNED,
                authority.PROTOCOL,
                "ops/exclusion_manifest.yaml",
                "python/carnot/reporting/primary_publication.py",
                "scripts/adversarial_verify.py",
                "scripts/verdict_row_consistency_lint.py",
            ]
        ],
        operands=[],
        receipts=[],
    )
    progress("preconditions_after", 1, 0)
    deadline = time.monotonic() + 1800
    if not gates:
        for index, slot in enumerate(slots):
            progress("benchmark_before_repeat", index, len(slots) - index)
            if time.monotonic() >= deadline:
                slot["missing_reason"] = "measurement_deadline"
                continue
            directory = raw / "repeats" / str(index)
            request, output = raw / f"repeat_request_{index}.json", directory / "primitive.json"
            atomic_json(
                request,
                dict(
                    bundle=bundle,
                    cell=slot["cell"],
                    repeat=slot["repeat"],
                    arm=slot["arm"],
                    directory=str(directory),
                    output=str(output),
                ),
            )
            receipt = child(
                f"repeat_{index}",
                [sys.executable, "-u", str(ROOT / CLI), "--repeat-request", str(request)],
                raw / "repeat_logs",
                deadline=min(600, deadline - time.monotonic()),
            )
            work["receipts"].append(receipt)
            try:
                if not receipt["passed"]:
                    raise ValueError(
                        "repeat_deadline" if receipt["timed_out"] else "repeat_child_failure"
                    )
                ref = reference(output)
                primitive = read(ref)
                slot.update(
                    k.reduce_repeat(bundle, primitive), primitive_reference=ref, missing_reason=None
                )
                work["operands"].append(ref)
            except (OSError, ValueError, KeyError, TypeError) as error:
                slot.update(
                    status="censored" if receipt["timed_out"] else "failed",
                    missing_reason=str(error),
                )
            progress("benchmark_after_repeat", index + 1, len(slots) - index - 1)
    frozen = freeze_workload(slots)
    atomic_json(raw / "frozen_workload.json", frozen)
    work.update(
        rows=[slot for slot in slots if not slot["warmup"]],
        warmups=[slot for slot in slots if slot["warmup"]],
        workload_reference=reference(raw / "frozen_workload.json"),
        duration_s=(time.monotonic_ns() - began) / 1e9,
        phase_spans=[
            dict(phase="preconditions_and_measurement", start_ns=began, end_ns=time.monotonic_ns())
        ],
    )
    atomic_json(raw / "measurement.json", work)
    return work


def build(work: Json, receipts: list[Json], output: Path) -> Json:
    """Readiness requires exact paired semantics and clocks, independent of speed gain."""
    rows = work["rows"]
    valid = [row for row in rows if row["status"] == "completed"]
    pairs = []
    for cell in k.CELLS:
        for repeat in range(1, 6):
            arms = [row for row in valid if row["cell"] == cell and row["repeat"] == repeat]
            pair: Json = dict(
                cell=cell,
                repeat=repeat,
                status="censored",
                instrumentation_overhead_s=None,
                instrumentation_overhead_ratio=None,
                missing_reason="incomplete_pair",
            )
            if len(arms) == 2:
                baseline = next(row for row in arms if row["arm"] == "baseline")
                instrumented = next(row for row in arms if row["arm"] == "instrumented")
                same = all(
                    baseline[key] == instrumented[key]
                    for key in ("semantic_hash", "response_hashes", "final_state_hash")
                )
                pair.update(
                    status="completed" if same else "failed",
                    missing_reason=None if same else "paired_semantic_drift",
                    order=[row["arm"] for row in arms],
                    baseline_complete_wall_s=baseline["complete_wall_s"],
                    instrumented_complete_wall_s=instrumented["complete_wall_s"],
                    instrumentation_overhead_s=instrumented["complete_wall_s"]
                    - baseline["complete_wall_s"],
                    instrumentation_overhead_ratio=instrumented["complete_wall_s"]
                    / baseline["complete_wall_s"]
                    - 1,
                )
            pairs.append(pair)
    coverage = work.get("owned_coverage", {})
    required = REQUIRED <= {row.get("name") for row in receipts if row["passed"]}
    passed = bool(receipts) and all(row["passed"] for row in receipts)
    covered = (
        coverage.get("percent_covered") == 100
        and coverage.get("missing_lines") == 0
        and coverage.get("num_statements", 0) > 0
    )
    complete = (
        len(valid) == 20
        and all(row["status"] == "completed" for row in pairs)
        and all(row["status"] == "completed" for row in work["warmups"])
    )
    verdict = (
        "disqualified"
        if not passed or any(row["status"] == "failed" for row in [*rows, *pairs])
        else (
            "blocked"
            if work["gate_check_summary"] or not complete
            else "null"
            if required and covered
            else "disqualified"
        )
    )
    stages = [
        dict(stage, cell=row["cell"], repeat=row["repeat"], arm=row["arm"])
        for row in valid
        for stage in row["stage_rows"]
    ]
    value: Json = dict(
        experiment_id=8449,
        task_id=TASK,
        milestone="2026.10.727",
        run_date="20261012",
        invocation_id=Path(work["raw"]).name,
        honest_verdict="complete_" + verdict + "_persistence_localization",
        verdict_class=verdict,
        gate_check_summary=deepcopy(work["gate_check_summary"]),
        inference_substrate="aggregation_from_upstream_artifacts",
        numerical_inference_substrate="verifier_ensemble_against_cached_candidates",
        inference_substrate_class="no_model_load",
        MODEL_SPECS=[],
        model_invocation_counts=dict(ZERO_INVOCATION_COUNTS, current_llm_invocations=0),
        historical_model_provenance=work["historical_model_provenance"],
        rows=rows,
        intended_count=20,
        completed_count=len(valid),
        failed_count=sum(row["status"] == "failed" for row in rows),
        censored_count=sum(row["status"] == "censored" for row in rows),
        excluded_count=0,
        independent_count=0,
        sample_size_budget=dict(
            cells=2,
            measured_repeats_per_cell=5,
            warmup_pairs_per_cell=1,
            arms=list(k.ARMS),
            repeats_add_independent_questions=0,
        ),
        verifier_is_oracle=False,
        exposure_scope="retrospective_exposed_cached_service_workload",
        independent_generalization_score=0,
        generalized_learning_benefit_score=0,
        required_checks_passed=passed and (verdict == "blocked" or required and covered),
        flagged_adversarial=not passed or verdict == "disqualified",
        acceptance_gates=dict(
            authentic_inputs=not work["gate_check_summary"],
            complete_pairs=complete,
            owned_validation=passed,
            scoped_validation=required,
            owned_coverage=covered,
        ),
        validation_receipts=receipts,
        terminal_validation_sidecar_path=str(Path(work["raw"]) / "terminal_validation.json"),
        adversarial_findings=work.get("adversarial_findings", []),
        preconditions_checked=work["preconditions_checked"],
        duration_s=work["duration_s"],
        phase_spans=work["phase_spans"],
        random_seed=7278449,
        source_artifact_hashes=work["refs"],
        code_config_hashes=work["code_config_hashes"],
        raw_shard_hashes=[
            *work["operands"],
            work["workload_reference"],
            *work.get("coverage_shards", []),
        ],
        cited_upstream_artifacts=[
            dict(
                ref,
                imported_fields="original complete service authority and cached five-input operands",
            )
            for ref in work["refs"]
        ],
        persistence_profile_ready_score=int(verdict == "null"),
        stage_rows=stages,
        paired_repeat_rows=pairs,
        instrumentation_overhead=dict(
            pairs=pairs,
            subtracted=False,
            interpretation="paired observed wall difference includes scheduling variation; baseline retains shipped service hooks",
        ),
        workload_reference=work["workload_reference"],
        filesystem_identity=work["preconditions_checked"]["filesystem_identity"],
        complete_wall_s=sum(row["complete_wall_s"] for row in valid) if valid else None,
        work_reference=reference(Path(work["raw"]) / "measurement.json"),
        owned_coverage=coverage,
        warmups=work["warmups"],
        methodology_note="Complete disk-backed service clocks preserve original delayed feedback. Nested exclusive spans isolate serialization and storage. Instrumentation overhead stays visible. No arithmetic acceleration or learning benefit is inferred.",
    )
    value["field_principles"] = {
        key: "Bind exact invocation or sealed evidence; missing differs from zero. Qualification grants no scientific benefit."
        for key in value
    }
    value["field_principles"]["instrumentation_overhead"] = (
        "Retain paired observed cost; never silently subtract instrumentation."
    )
    value["reproducibility_checksum"] = checksum(value)
    return value


def replay(path: Path) -> bool:
    """Cold readers reduce disk evidence again; repaired summary hashes cannot qualify it."""
    try:
        value = json.loads(path.read_bytes())
        if value["reproducibility_checksum"] != checksum(value):
            return False
        work = read(value["work_reference"])
        for ref in [
            *value["source_artifact_hashes"],
            *value["code_config_hashes"],
            *value["raw_shard_hashes"],
        ]:
            if sha256_file(Path(ref["path"])) != ref["sha256"]:
                return False
        if value["workload_reference"] != work["workload_reference"]:
            return False
        frozen = read(work["workload_reference"])
        if (
            frozen["event_tape"] != k.cost_tape(8)
            or frozen["cells"] != k.CELLS
            or frozen["inputs"] != 5
        ):
            return False
        if [
            (row["cell"], row["repeat"], row["arm"]) for row in [*work["warmups"], *work["rows"]]
        ] != [
            (row["cell"], row["repeat"], row["arm"])
            for row in [
                *[slot for slot in roster() if slot["warmup"]],
                *[slot for slot in roster() if not slot["warmup"]],
            ]
        ]:
            return False
        for row in [*work["rows"], *work["warmups"]]:
            if row["status"] == "completed":
                actual = k.reduce_repeat(work["bundle"], read(row["primitive_reference"]))
                if any(row[key] != item for key, item in actual.items()):
                    return False
        intended = roster()
        by_id = {
            (row["cell"]["batch"], row["repeat"], row["arm"]): row
            for row in [*work["rows"], *work["warmups"]]
        }
        ordered = [by_id[(row["cell"]["batch"], row["repeat"], row["arm"])] for row in intended]
        if frozen != freeze_workload(ordered):
            return False
        return bool(value == build(work, value["validation_receipts"], path))
    except (OSError, ValueError, KeyError, TypeError, IndexError, StopIteration):
        return False
