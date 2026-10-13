"""REQ-REPORT-8493: authenticated stable append qualification with independent replay.

The unchanged snapshot oracle and separately pinned upstream tape determine
causal meaning. Primitive disk evidence, returned IDs and actual exits remain
independent operands; recalculating summary hashes cannot change them.
"""

from __future__ import annotations

from copy import deepcopy
import json
import os
from pathlib import Path
import shutil
import sys
from tempfile import TemporaryDirectory
import time
from typing import Any
from unittest.mock import patch

from carnot.reporting import journal_equivalence_8465 as legacy
from carnot.reporting import publication_cost_model_8492 as upstream
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash
from carnot.reporting.primary_publication import read_bound_sidecar, reader_receipt
from carnot.reporting.v709_execution import child
from carnot.verify import append_store_8493 as k

Json = dict[str, Any]
ROOT = k.ROOT
NAME, TASK = "experiment_8493_v730_append_store_qualification", "exp8493-append-store-qualification"
CLI, TEST = f"scripts/experiments/{NAME}.py", "tests/python/test_append_store_8493.py"
OWNED = [
    "python/carnot/verify/append_store_8493.py",
    "python/carnot/reporting/append_store_qualification_8493.py",
    "python/carnot/reporting/append_store_runner_8493.py",
    CLI,
]
MODEL_SPECS: list[Json] = []
UPSTREAM = "results/experiment_8492_v730_publication_cost_model.json"
UPSTREAM_PIN = "sha256:e4aac512456de843ad91b91f3ed92faedb93c42ebd94f8b7054ba62e346aacf0"
SLOTS = [("equivalence", "both", str(b)) for b in (1, 8)] + [
    ("crash", "append", b) for b in k.BARRIERS
]
REQUIRED = upstream.shared.REQUIRED - {"private_E2E018_consumers", "private_E2E021"}
reference, read, checksum = upstream.reference, upstream.read, upstream.checksum


class GateFailure(ValueError):
    """Keep the failed check's typed operands instead of confusing missing input with zero."""

    def __init__(self, check: str, path: Path, field: str, expected: Any, observed: Any):
        super().__init__(check)
        self.row = dict(
            check=check,
            upstream=upstream.TASK,
            path=str(path),
            sha256=upstream.sha256_file(path) if path.exists() else None,
            expected_sha256=UPSTREAM_PIN if path.name == Path(UPSTREAM).name else None,
            field=field,
            operator="==",
            expected=expected,
            observed=observed,
            missing=not path.exists(),
            passed=False,
        )


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Flush true phase counts even though this engineering task loads no model."""
    print(f"[exp8493] phase={phase} completed={completed} pending={pending}", flush=True)


def admission(root: Path, raw: Path) -> Json:
    """Bind this activated task and exact qualified protocol bytes before any store writes."""
    authority = upstream.methods.authority(root, raw / "authority")
    if (
        not authority["activated"]
        or upstream.methods.tasks_digest(authority["tasks"]) != upstream.methods.TASKS_PIN
    ):
        raise GateFailure(
            "activated_full_v730_authority",
            root / upstream.methods.ACTIVE,
            "activated_full_v730_authority",
            True,
            False,
        )
    task = next(t for t in authority["tasks"] if t["id"] == TASK)
    if task["deliverable"] != "results/" + NAME + ".json":
        raise GateFailure(
            "activated_task_deliverable",
            root / upstream.methods.ACTIVE,
            "deliverable",
            "results/" + NAME + ".json",
            task["deliverable"],
        )
    path = root / UPSTREAM
    value = read(dict(path=str(path), sha256=UPSTREAM_PIN))
    if (
        value.get("append_protocol_ready_score") != 1
        or value.get("required_checks_passed") is not True
        or value.get("flagged_adversarial") is not False
    ):
        fields = ("append_protocol_ready_score", "required_checks_passed", "flagged_adversarial")
        raise GateFailure(
            "qualified_append_protocol",
            path,
            " AND ".join(fields),
            dict(zip(fields, (1, True, False), strict=True)),
            {f: value.get(f) for f in fields},
        )
    if not reader_receipt(upstream.TASK, path.parent, field="append_protocol_ready_score")[
        "passed"
    ]:
        raise GateFailure("upstream_typed_consumer", path, "typed_consumer_passed", True, False)
    terminal = read(reference(Path(value["terminal_validation_sidecar_path"])))
    side = read_bound_sidecar(path, Path(terminal["publication"]["sidecar_path"]))
    if not side["report"]["passed"]:
        raise GateFailure("upstream_terminal_validation", path, "bound_sidecar_passed", True, False)
    if read(value["append_protocol_reference"]) != k.framing.protocol():
        raise GateFailure("sealed_append_protocol", path, "sealed_protocol_equal", True, False)
    receipt = child(
        "sealed_original_replay",
        [sys.executable, "-u", str(ROOT / upstream.CLI), "--cold-replay", str(path)],
        raw / "upstream_logs",
        deadline=240,
        scope="upstream",
    )
    if not receipt["passed"]:
        raise GateFailure("original_authority_replay", path, "original_replay_passed", True, False)
    tapes = [read(ref) for ref in value["transition_fixture"]["tape_references"]]
    refs = [
        reference(path),
        reference(Path(terminal["publication"]["sidecar_path"])),
        value["append_protocol_reference"],
        *value["transition_fixture"]["tape_references"],
    ]
    refs += [
        dict(path=r["snapshot_path"], sha256=r["sha256"])
        for r in authority["authority_snapshots"].values()
        if r.get("exists")
    ]
    refs += [
        dict(path=r.get("snapshot_path", r["path"]), sha256=r["sha256"])
        for r in value["source_artifact_hashes"]
    ]
    return dict(authority=authority, tapes=tapes, upstream=value, refs=refs, receipt=receipt)


def worker(frozen_path: Path, directory: Path, barrier: str) -> int:
    """Real process exits keep caller acknowledgments separate from complete unacknowledged frames."""
    frozen = json.loads(frozen_path.read_bytes())
    store = k.Store(directory / "store", frozen)
    store.initialize()
    acknowledgments: list[Json] = []
    for index, tx in enumerate(frozen["transactions"]):

        def hook(phase: str) -> None:
            if index == 31 and phase == barrier:
                atomic_json(
                    directory / "crash_marker.json", dict(barrier=barrier, sequence=index + 1)
                )
                atomic_json(directory / "operations.json", dict(rows=store.operations))
                os._exit(73)

        reply = store.apply(tx, hook)
        acknowledgments.append(dict(id=tx["id"], response=reply))
        atomic_json(directory / "acknowledged.json", dict(rows=acknowledgments))
        hook("after_acknowledgment")
        if index % 16 == 0:
            progress("worker", index + 1, len(frozen["transactions"]) - index - 1)
    return 0


def crash(tape: Json, directory: Path, barrier: str) -> Json:
    """Recover actual exited bytes and retry all IDs without applying feedback twice."""
    atomic_json(directory / "frozen.json", tape)
    receipt = child(
        barrier,
        [
            sys.executable,
            "-u",
            str(ROOT / CLI),
            "--worker",
            str(directory / "frozen.json"),
            "--store",
            str(directory),
            "--barrier",
            barrier,
        ],
        directory / "logs",
        expected=73,
        deadline=120,
    )
    shutil.copytree(directory / "store", directory / "crashed")
    acknowledgments = json.loads((directory / "acknowledged.json").read_bytes())["rows"]
    store = k.Store(directory / "store", tape)
    recovered = store.read([r["id"] for r in acknowledgments])
    replies = [store.apply(tx) for tx in tape["transactions"]]
    store.drain()
    expected = k.base.oracle(tape, len(tape["transactions"]))
    final = store.read()
    operations = json.loads((directory / "operations.json").read_bytes())["rows"] + store.operations
    atomic_json(directory / "operations-final.json", dict(rows=operations))
    return dict(
        kind="crash",
        arm="append",
        barrier=barrier,
        directory=str(directory),
        frozen_reference=reference(directory / "frozen.json"),
        receipt=receipt,
        actual_exit=receipt["actual_exit"],
        recovered=recovered,
        final=final,
        acknowledgments=acknowledgments,
        response_hash=canonical_hash(replies),
        recovered_state_hash=canonical_hash(recovered["state"]),
        namespace_operation_rows=operations,
        passed=receipt["passed"] and replies == expected["replies"] and final == expected,
    )


def equivalence(tape: Json, directory: Path) -> Json:
    """Compare both actual stores with the independent snapshot transition oracle."""
    atomic_json(directory / "frozen.json", tape)
    store = k.Store(directory / "append", tape)
    snapshot = k.base.Snapshot(directory / "snapshot", tape)
    store.initialize()
    snapshot.initialize()
    pin = store.read()
    inode = store.journal.stat().st_ino
    replies = []
    for i, tx in enumerate(tape["transactions"]):
        reply = store.apply(tx)
        if reply != snapshot.apply(tx):
            raise ValueError("snapshot_response_parity")
        replies.append(reply)
        if i % 16 == 0:
            progress("equivalence", i + 1, len(tape["transactions"]) - i - 1)
    store.drain()
    expected = k.base.oracle(tape, len(tape["transactions"]))
    final = store.read()
    snapshot_final = snapshot.read()
    atomic_json(directory / "operations-final.json", dict(rows=store.operations))
    return dict(
        kind="equivalence",
        arm="both",
        barrier=str(tape["cell"]["batch"]),
        directory=str(directory),
        frozen_reference=reference(directory / "frozen.json"),
        final=final,
        actual_exit=0,
        acknowledgments=[
            dict(id=tx["id"], response=reply)
            for tx, reply in zip(tape["transactions"], replies, strict=True)
        ],
        response_hash=canonical_hash(replies),
        recovered_state_hash=canonical_hash(expected["state"]),
        namespace_operation_rows=store.operations,
        passed=final == snapshot_final == expected
        and pin == k.base.oracle(tape, 0)
        and inode == store.journal.stat().st_ino,
    )


def reduce(primitive: Json) -> bool:
    """Reconstruct actual disk prefixes and acknowledgments from independently sealed inputs."""
    tape = read(primitive["frozen_reference"])
    expected = k.base.oracle(tape, len(tape["transactions"]))
    if canonical_hash(primitive["final"]) != canonical_hash(expected) or primitive[
        "response_hash"
    ] != canonical_hash(expected["replies"]):
        return False
    crash_row = primitive["kind"] == "crash"
    seq = (
        (31 if primitive["barrier"] == "before_append" else 32)
        if crash_row
        else len(tape["transactions"])
    )
    ack_seq = (32 if primitive["barrier"] == "after_acknowledgment" else 31) if crash_row else seq
    ledger = [
        dict(id=tx["id"], response=reply)
        for tx, reply in zip(
            tape["transactions"][:ack_seq], expected["replies"][:ack_seq], strict=True
        )
    ]
    if canonical_hash(primitive["acknowledgments"]) != canonical_hash(ledger):
        return False
    operations = json.loads((Path(primitive["directory"]) / "operations-final.json").read_bytes())
    if canonical_hash(primitive["namespace_operation_rows"]) != canonical_hash(operations["rows"]):
        return False
    with TemporaryDirectory(dir="/var/tmp", prefix="exp8493-replay-") as folder:
        source = Path(primitive["directory"]) / ("crashed" if crash_row else "append")
        target = Path(folder) / "store"
        shutil.copytree(source, target)
        recovered = k.Store(target, tape).read([r["id"] for r in ledger])
        if canonical_hash(recovered) != canonical_hash(k.base.oracle(tape, seq)):
            return False
        if crash_row and (
            primitive["actual_exit"] != 73
            or canonical_hash(primitive["recovered"]) != canonical_hash(recovered)
            or not primitive["receipt"]["passed"]
        ):
            return False
        if primitive["recovered_state_hash"] != canonical_hash(recovered["state"]):
            return False
    return bool(primitive["passed"])


def measure(root: Path, raw: Path) -> Json:
    """Measure CPU fixtures only after required authority; absence remains an explicit gate."""
    progress("preconditions")
    start = time.monotonic_ns()
    raw.mkdir(parents=True, exist_ok=True, mode=0o700)
    work: Json = dict(
        root=str(root),
        raw=str(raw),
        started_monotonic_ns=start,
        rows=[],
        primitives=[],
        refs=[],
        owned_failures={},
        gate_check_summary=[],
        historical_model_provenance=[],
        coverage_shards=[],
        phase_spans=[],
        adversarial_findings=[],
        authority={},
        receipts=[],
    )
    with TemporaryDirectory(dir="/var/tmp", prefix="exp8493-preflight-") as folder:
        scratch = Path(folder)
        probe = scratch / "readback"
        probe.write_bytes(b"private-readback")
        tools = {
            n: shutil.which(str(ROOT / ".venv/bin" / n))
            for n in ("python", "pytest", "coverage", "ruff", "mypy")
        }
        memory = int(Path("/proc/meminfo").read_text().split("MemAvailable:")[1].split()[0]) * 1024
        work["preconditions_checked"] = dict(
            tools=tools,
            memory_available_bytes=memory,
            disk_free_bytes=shutil.disk_usage(scratch).free,
            scratch_filesystem=upstream.filesystem(scratch),
            scratch_readback=probe.read_bytes() == b"private-readback",
            task_cap_s=4800,
            measurement_cap_s=3600,
            validation_reserve_s=900,
            optional_cuda_and_boards="not_required",
        )
        if (
            not all(tools.values())
            or memory < 256 * 1024 * 1024
            or not upstream.filesystem(scratch)["disk_backed"]
        ):
            work["owned_failures"]["preconditions"] = "tools_memory_disk"
    try:
        inputs = admission(root, raw)
        work.update(
            authority=inputs["authority"],
            refs=inputs["refs"],
            tapes=inputs["tapes"],
            historical_model_provenance=inputs["upstream"]["historical_model_provenance"],
            preserved_v728=inputs["upstream"]["preserved_v728"],
            receipts=[inputs["receipt"]],
        )
    except (OSError, ValueError, KeyError, TypeError, StopIteration) as error:
        path = root / UPSTREAM
        work["gate_check_summary"].append(
            error.row
            if isinstance(error, GateFailure)
            else dict(
                check="activated_authority_and_sealed_append_protocol:" + str(error),
                upstream=upstream.TASK,
                path=str(path),
                sha256=upstream.sha256_file(path) if path.exists() else None,
                expected_sha256=UPSTREAM_PIN,
                field="authenticated_input_readable",
                operator="==",
                expected=True,
                observed=None if not path.exists() else False,
                missing=not path.exists(),
                passed=False,
            )
        )
    for i, (kind, arm, barrier) in enumerate(SLOTS):
        unit = f"{kind}-{arm}-{barrier}"
        row: Json = dict(
            unit_id=unit,
            family="append8493_engineering_fixture",
            kind=kind,
            arm=arm,
            barrier=barrier,
            status="censored",
            passed=None,
            absolute_metric=None,
            missing_reason="required_authority_or_protocol_missing",
            effective_independent_groups=0,
        )
        if not work["gate_check_summary"] and not work["owned_failures"]:
            progress("fixture_before_" + unit, i, len(SLOTS) - i)
            try:
                if (time.monotonic_ns() - start) / 1e9 >= 3600:
                    raise TimeoutError("measurement_deadline")
                tape = work["tapes"][int(barrier == "8") if kind == "equivalence" else 0]
                directory = raw / unit
                primitive = (
                    equivalence(tape, directory)
                    if kind == "equivalence"
                    else crash(tape, directory, barrier)
                )
                primitive["store_references"] = [
                    reference(p) for p in sorted(directory.rglob("*")) if p.is_file()
                ]
                atomic_json(directory / "primitive.json", primitive)
                ref = reference(directory / "primitive.json")
                work["primitives"].append(ref)
                row.update(
                    status="completed" if primitive["passed"] else "failed",
                    passed=primitive["passed"],
                    absolute_metric=int(primitive["passed"]),
                    missing_reason=None,
                    primitive_reference=ref,
                    actual_exit=primitive["actual_exit"],
                    recovered_state_hash=primitive["recovered_state_hash"],
                )
            except (OSError, ValueError, KeyError, TypeError, TimeoutError) as error:
                work["owned_failures"][unit] = str(error)
                row.update(status="failed", passed=False, missing_reason=str(error))
            progress("fixture_after_" + unit, i + 1, len(SLOTS) - i - 1)
        elif work["owned_failures"]:
            row["missing_reason"] = "owned_precondition_or_measurement_failure"
        work["rows"].append(row)
    work["code_config_hashes"] = [
        reference(ROOT / p) for p in [*OWNED, TEST, *upstream.DEPENDENCIES]
    ]
    work["ended_monotonic_ns"] = time.monotonic_ns()
    work["phase_spans"] = [
        dict(
            phase="admission_and_qualification",
            duration_s=(work["ended_monotonic_ns"] - start) / 1e9,
        )
    ]
    return work


def build(work: Json, receipts: list[Json], raw: Path, output: Path) -> Json:
    """Use the shipped engineering schema and require all invocation gates for readiness."""
    execution = (
        json.loads((raw / "execution_manifest.json").read_bytes())
        if (raw / "execution_manifest.json").exists()
        else dict(commands=[])
    )
    required = REQUIRED | {p["name"] for p in execution["commands"]}
    with patch.multiple(legacy, SLOTS=SLOTS, OWNED=OWNED, TASK=TASK, REQUIRED=required):
        value = legacy.build(work, receipts, output)
    passed = bool(receipts) and all(r["passed"] for r in receipts)
    failures = (
        bool(work["owned_failures"])
        or any(not r["passed"] for r in receipts)
        or bool(work["adversarial_findings"])
    )
    verdict = (
        "disqualified"
        if failures
        else "blocked"
        if work["gate_check_summary"]
        else value["verdict_class"]
    )
    primitives = [read(ref) for ref in work["primitives"]]
    value.pop("journal_ready_score")
    value.update(
        experiment_id=8493,
        milestone="2026.10.730",
        run_date="20261013",
        honest_verdict="complete_" + verdict + "_append_store_qualification",
        verdict_class=verdict,
        append_store_ready_score=int(verdict == "circular_positive"),
        random_seed=7308493,
        required_checks_passed=passed
        and required <= {r["name"] for r in receipts if r["passed"]}
        and value["acceptance_gates"]["owned_coverage"],
        flagged_adversarial=bool(work["adversarial_findings"]),
        historical_validation_receipts=work["receipts"],
        namespace_operation_rows=[
            dict(
                unit_id=p["directory"],
                calls=p["namespace_operation_rows"],
                file_fsyncs=sum(
                    r["operation"] == "file_fsync" for r in p["namespace_operation_rows"]
                ),
                directory_fsyncs=sum(
                    r["operation"] == "directory_fsync" for r in p["namespace_operation_rows"]
                ),
                transaction_pointer_replacements=0,
            )
            for p in primitives
        ],
        sample_size_budget=dict(
            engineering_cells=2,
            crash_arms=1,
            boundaries_per_arm=len(k.BARRIERS),
            independent_science_questions=0,
        ),
        preserved_v728=work.get("preserved_v728"),
        authority_snapshots=work["authority"].get("authority_snapshots", {}),
        methodology_note="Stable-file framed append using independently sealed original five-input service tapes and unchanged snapshot transitions. Fourteen actual process exits and bitwise recovery qualify engineering fixtures only. No model calls, scientific benefit, speed measurement or physical power-loss certification.",
    )
    value["cited_upstream_artifacts"] = [
        dict(
            ref,
            imported_fields="Exp8492 protocol and sealed V728 producer tapes under original authority; preserved service null and cached provenance",
        )
        for ref in work["refs"]
    ]
    value["journal_limits"].update(
        reclamation="durable new checkpoint and prefix precede old checkpoint unlink; immutable reader copies retain all operands"
    )
    value["field_principles"] = {
        key: "Bind independently sealed invocation evidence; engineering recovery is not generalization or speed benefit."
        for key in value
    }
    value["field_principles"].update(
        append_store_ready_score="All original transition invariants, actual exit recovery, owned coverage and unchanged consumers must pass.",
        namespace_operation_rows="Record successful actual synchronization and namespace calls; no per-transaction pointer publication.",
        acknowledgment_ledger="Every returned stable ID survives recovery; complete unacknowledged frames replay idempotently.",
    )
    value["reproducibility_checksum"] = checksum(value)
    return dict(value)


def replay(path: Path) -> bool:
    """Fresh bounded readers reject repaired primitive, authority and aggregate hashes."""
    try:
        value = json.loads(path.read_bytes())
        if value["experiment_id"] != 8493 or value["reproducibility_checksum"] != checksum(value):
            return False
        work = read(value["work_reference"])
        if len(work["rows"]) != len(SLOTS):
            return False
        for ref in [
            *work["refs"],
            *work["code_config_hashes"],
            *work["primitives"],
            *work["coverage_shards"],
        ]:
            read_bytes = Path(ref["path"])
            if upstream.sha256_file(read_bytes) != ref["sha256"]:
                return False
        if work.get("tapes"):
            sealed = read(dict(path=str(Path(work["root"]) / UPSTREAM), sha256=UPSTREAM_PIN))
            if (
                upstream.methods.tasks_digest(work["authority"]["tasks"])
                != upstream.methods.TASKS_PIN
            ):
                return False
            if work["tapes"] != [
                read(ref) for ref in sealed["transition_fixture"]["tape_references"]
            ]:
                return False
            if work["preserved_v728"] != sealed["preserved_v728"]:
                return False
        for slot, row in zip(SLOTS, work["rows"], strict=True):
            if tuple(row[key] for key in ("kind", "arm", "barrier")) != slot:
                return False
            if "primitive_reference" not in row:
                continue
            primitive = read(row["primitive_reference"])
            if (
                tuple(primitive[key] for key in ("kind", "arm", "barrier")) != slot
                or row["passed"] != primitive["passed"]
            ):
                return False
            for ref in primitive["store_references"]:
                if upstream.sha256_file(Path(ref["path"])) != ref["sha256"]:
                    return False
            if read(primitive["frozen_reference"]) not in work["tapes"] or not reduce(primitive):
                return False
        return bool(value == build(work, value["validation_receipts"], Path(work["raw"]), path))
    except (OSError, ValueError, KeyError, TypeError, IndexError):
        return False
