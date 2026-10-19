"""REQ-REPORT-8461: keep sealed historical operands separate from constructed labels."""

from __future__ import annotations

import json
from pathlib import Path
import shutil
import sys
import time
from typing import Any

from carnot.reporting import block_checkpoint_qualification_8433 as inherited
from carnot.reporting import v727_historical_inputs as history
from carnot.reporting import v728_contract_methods as methods
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash
from carnot.reporting.human_label_custody_8389 import gate
from carnot.reporting.v709_execution import child
from carnot.verify import objective_checkpoint_8461 as adapter
from carnot.verify import objective_derivatives_8461 as derivatives

__all__ = [
    "methods",
    "ROOT",
    "NAME",
    "TASK",
    "CLI",
    "reference",
    "read",
    "progress",
    "authenticate",
    "recovery",
    "measure",
]

Json = dict[str, Any]
ROOT = methods.ROOT
NAME = "experiment_8461_v728_objective_checkpoint_qualification"
TASK = "exp8461-objective-checkpoint-qualification"
CLI = f"scripts/experiments/{NAME}.py"
reference, read = methods.reference, methods.read


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """A supervisor sees actual phase completion while children keep separate durable logs."""
    print(f"[exp8461] phase={phase} completed={completed} pending={pending}", flush=True)


def authenticate(root: Path, raw: Path) -> Json:
    """Use current full authority for this task and original authority for historical producers."""
    failures, refs, receipts, summaries = [], [], [], []
    try:
        authority = methods.authority(root, raw / "current")
        failures.extend(authority["gate_check_summary"])
        task = next(t for t in authority["tasks"] if t["id"] == TASK)
        if task["deliverable"] != f"results/{NAME}.json" or not authority["activated"]:
            raise ValueError("task_authority")
        methods.protocol()
        refs.append(reference(ROOT / methods.PROTOCOL))
    except (OSError, ValueError, KeyError, StopIteration) as error:
        authority = dict(activated=False)
        failures.append(
            gate(
                str(root / methods.DESIGN),
                "current_full_authority",
                True,
                None if isinstance(error, FileNotFoundError) else str(error),
                upstream=TASK,
            )
        )
    head = None
    provenance = []
    try:
        original = history.custody(root, raw / "original")
        for number, name in [
            (8433, inherited.NAME),
            (8434, "experiment_8434_v726_capacity_online_learning"),
        ]:
            primary = dict(
                path=str(root / "results" / (name + ".json")),
                sha256="sha256:" + history.PINS[number],
            )
            value = read(primary)
            refs.append(primary)
            request, result = raw / f"request-{number}.json", raw / f"result-{number}.json"
            atomic_json(request, dict(number=number, primary=primary, authority=original))
            receipts.append(
                child(
                    f"original_producer_{number}",
                    [
                        sys.executable,
                        "-u",
                        str(ROOT / methods.CLI),
                        "--history-request",
                        str(request),
                        "--history-output",
                        str(result),
                    ],
                    raw / "processes",
                    deadline=120,
                )
            )
            summary = json.loads(result.read_bytes())
            summaries.append(summary)
            failures.extend(summary["gate_check_summary"])
            if not receipts[-1]["passed"] or not summary["qualified"]:
                raise ValueError("historical_producer_qualification")
            if number == 8434:
                work = read(value["primitive_reference"])
                ref = work["cells"][0]["configuration"]
                head = read(ref)["head"]
                refs.extend([value["primitive_reference"], ref])
                provenance = value["historical_model_provenance"]
        refs.extend(dict(path=r["snapshot_path"], sha256=r["sha256"]) for r in original)
    except (OSError, ValueError, KeyError, TypeError) as error:
        head = None
        failures.append(
            gate(
                str(root / "results/experiment_8434_v726_capacity_online_learning.json"),
                "original_head_and_checkpoint_qualification",
                True,
                None if isinstance(error, FileNotFoundError) else str(error),
                upstream="exp8434-capacity-online-learning",
            )
        )
    return dict(
        authority=authority,
        failures=failures,
        refs=refs,
        receipts=receipts,
        historical=summaries,
        head=head,
        historical_model_provenance=provenance,
    )


def recovery(
    raw: Path, head: Json, private: Path, *, deadline_ns: int
) -> tuple[list[Json], list[Json], list[Json]]:
    """Real exits on both sides of each commit must recover the same future predictions."""
    rows: list[Json] = []
    exits: list[Json] = []
    receipts: list[Json] = []
    for capacity in (8, 16):
        for policy in ("fifo", "uniform"):
            cfg = adapter.config(head, capacity, policy, shuffled=True)
            family = f"c{capacity}-{policy}"
            cfg_path = raw / family / "configuration.json"
            atomic_json(cfg_path, cfg)
            baseline = None
            cases = [
                "none",
                *[f"{t}:0:{b}" for t in ("seal", "release") for b in adapter.legacy.BARRIERS],
            ]
            for case in cases:
                if time.monotonic_ns() >= deadline_ns:
                    progress("measurement_deadline", len(rows), 36 - len(rows))
                    return rows, exits, receipts
                progress("recovery", len(rows), 36 - len(rows))
                directory = private / family / case.replace(":", "-")
                evidence_directory = raw / family / case.replace(":", "-")
                argv = [
                    sys.executable,
                    "-u",
                    str(ROOT / CLI),
                    "--worker",
                    str(directory),
                    "--configuration",
                    str(cfg_path),
                ]
                crashed_ok = True
                if case != "none":
                    receipt = child(
                        f"{family}-{case.replace(':', '-')}-exit",
                        argv + ["--crash", case],
                        raw / "processes",
                        expected=73,
                        deadline=60,
                    )
                    receipts.append(receipt)
                    snapshot = adapter.Store(directory, cfg).read()
                    expected = int(case.startswith("release")) + int(
                        case.split(":")[-1] in ("atomic_replace", "acknowledgment")
                    )
                    path = evidence_directory / "crash.json"
                    atomic_json(path, snapshot)
                    crashed_ok = (
                        receipt["passed"]
                        and snapshot["version"] == expected
                        and adapter.replay_state(snapshot, cfg)
                    )
                    exits.append(
                        dict(
                            unit_id=f"{family}:{case}",
                            actual_exit=receipt["exit_code"],
                            expected_version=expected,
                            checkpoint=reference(path),
                            configuration=reference(cfg_path),
                            passed=crashed_ok,
                        )
                    )
                receipt = child(
                    f"{family}-{case.replace(':', '-')}-resume",
                    argv,
                    raw / "processes",
                    deadline=60,
                )
                receipts.append(receipt)
                result = json.loads((directory / "final.json").read_bytes())
                final = evidence_directory / "final.json"
                atomic_json(final, result)
                if case == "none":
                    baseline = result
                passed = (
                    receipt["passed"]
                    and crashed_ok
                    and result == baseline
                    and adapter.replay_state(result, cfg)
                )
                rows.append(
                    dict(
                        unit_id=f"{family}:{case}",
                        family=family,
                        arm=policy,
                        capacity=capacity,
                        barrier=case,
                        passed=passed,
                        configuration=reference(cfg_path),
                        final_state=reference(final),
                        prediction_hash=canonical_hash(result["issued"]),
                        logical_state_hash=canonical_hash(result),
                        coefficient_change_count=sum(
                            int(a != b)
                            for a, b in zip(
                                adapter.sparse.parameters(head),
                                adapter.sparse.parameters(result["head"]),
                                strict=True,
                            )
                        ),
                    )
                )
                if not passed:
                    return rows, exits, receipts
    return rows, exits, receipts


def measure(root: Path, raw: Path, private: Path) -> Json:
    """Authenticate CPU resources and measure prerequisites without any public-label stream."""
    start = time.monotonic_ns()
    raw.mkdir(parents=True, exist_ok=True, mode=0o700)
    private.mkdir(parents=True, exist_ok=True, mode=0o700)
    progress("preconditions")
    authenticated = authenticate(root, raw / "authority")
    fs = inherited.filesystem_type(private)
    memory = (
        int(
            next(
                line.split()[1]
                for line in Path("/proc/meminfo").read_text().splitlines()
                if line.startswith("MemAvailable:")
            )
        )
        * 1024
    )
    tools = {
        n: shutil.which(str(ROOT / ".venv/bin" / n))
        for n in ("python", "pytest", "coverage", "ruff", "mypy")
    }
    preconditions = dict(
        private_disk_backed_scratch=fs not in {"tmpfs", "ramfs", "devtmpfs", "unknown"},
        filesystem=fs,
        private_path=str(private),
        available_memory_bytes=memory,
        free_disk_bytes=shutil.disk_usage(private).free,
        tools=tools,
        measurement_cap_s=3600,
        validation_reserve_s=900,
        task_cap_s=4800,
        child_poll_s=30,
        optional_cuda_required=False,
    )
    failures = authenticated["failures"]
    if (
        not preconditions["private_disk_backed_scratch"]
        or memory < 128 * 1024**2
        or preconditions["free_disk_bytes"] < 100 * 1024**2
        or not all(tools.values())
    ):
        failures.append(gate(str(private), "cpu_resources", True, preconditions, upstream=TASK))
    head = authenticated["head"]
    progress("derivatives")
    derivative_rows = derivatives.checks(head) if head is not None and not failures else []
    transition_rows = derivatives.transition_controls(head) if derivative_rows else []
    progress("derivatives_complete", len(derivative_rows), 0)
    rows: list[Json] = []
    exits: list[Json] = []
    receipts: list[Json] = []
    if (
        head is not None
        and derivative_rows
        and all(r["passed"] for r in derivative_rows + transition_rows)
    ):
        rows, exits, receipts = recovery(
            raw / "recovery", head, private / "recovery", deadline_ns=start + 3600 * 10**9
        )
    if (time.monotonic_ns() - start) / 1e9 > 3600:
        failures.append(
            gate(
                str(raw),
                "measurement_cap_s",
                3600,
                (time.monotonic_ns() - start) / 1e9,
                operator="<=",
                upstream=TASK,
            )
        )
    primitive = dict(
        derivative_rows=derivative_rows,
        restart_rows=rows,
        process_exit_rows=exits,
        head=head,
        synthetic_fallback=False,
        transition_rows=transition_rows,
    )
    atomic_json(raw / "primitives.json", primitive)
    work = dict(
        root=str(root),
        authority=authenticated["authority"],
        failures=failures,
        preconditions=preconditions,
        historical=authenticated["historical"],
        source_refs=authenticated["refs"],
        historical_model_provenance=authenticated["historical_model_provenance"],
        process_receipts=authenticated["receipts"] + receipts,
        primitive_reference=reference(raw / "primitives.json"),
        started_monotonic_ns=start,
        ended_monotonic_ns=time.monotonic_ns(),
    )
    atomic_json(raw / "measurement.json", work)
    return work
