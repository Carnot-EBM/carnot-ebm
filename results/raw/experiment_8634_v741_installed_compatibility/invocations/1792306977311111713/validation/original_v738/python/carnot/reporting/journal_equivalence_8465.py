"""REQ-REPORT-8465: qualify representation changes without a cost or learning claim.

The current full-object authority authorizes this task. Independent sealed
Exp8464 operands and original historical authority determine causal meaning.
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

from carnot.reporting import persistence_localization_8464 as upstream
from carnot.reporting.current_work_receipt import (
    atomic_json,
    canonical_hash,
    ZERO_INVOCATION_COUNTS,
)
from carnot.reporting.v709_execution import child
from carnot.verify import journal_equivalence_8465 as k
from carnot.verify import python_transaction_cost_8407 as source

Json = dict[str, Any]
ROOT = upstream.ROOT
NAME, TASK = "experiment_8465_v728_journal_equivalence", "exp8465-journal-equivalence"
CLI, TEST = f"scripts/experiments/{NAME}.py", "tests/python/test_journal_equivalence_8465.py"
OWNED = [
    "python/carnot/verify/journal_equivalence_8465.py",
    "python/carnot/reporting/journal_equivalence_8465.py",
    "python/carnot/reporting/journal_equivalence_runner_8465.py",
    CLI,
]
MODEL_SPECS: list[Json] = []
UPSTREAM = "results/experiment_8464_v728_persistence_localization.json"
UPSTREAM_PIN = "sha256:759bd38012061675801027e8d053b6df2ebe06970453bceede8c822648629f77"
reference, read, checksum, REQUIRED = (
    upstream.reference,
    upstream.read,
    upstream.checksum,
    upstream.REQUIRED,
)
SLOTS = [("equivalence", "both", str(b)) for b in (1, 8)] + [
    ("crash", arm, barrier) for arm in ("journal", "snapshot") for barrier in k.BARRIERS
]


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Report actual bounded work rather than leaving the supervisor to guess."""
    print(f"[exp8465] phase={phase} completed={completed} pending={pending}", flush=True)


def dependencies(root: Path, raw: Path, private: Path) -> Json:
    """Authenticate the tape producer and inherited service through their sealed roots."""
    value = read(dict(path=str(root / UPSTREAM), sha256=UPSTREAM_PIN))
    if value.get("persistence_profile_ready_score") != 1 or not value["required_checks_passed"]:
        raise ValueError("qualified_8464_tape")
    result = upstream.dependencies(root, raw, private)
    work = read(value["work_reference"])
    tape = read(value["workload_reference"])
    if work["bundle"] != result["bundle"] or tape["event_tape"] != source.cost_tape(8):
        raise ValueError("independently_sealed_tape")
    result.update(workload=tape)
    result["refs"] += [
        reference(root / UPSTREAM),
        value["work_reference"],
        value["workload_reference"],
    ]
    return dict(result)


def worker(frozen_path: Path, directory: Path, arm: str, barrier: str) -> int:
    """A real exit records the actual returned acknowledgments before losing this process."""
    frozen = json.loads(frozen_path.read_bytes())
    store = (k.Journal if arm == "journal" else k.Snapshot)(directory / "store", frozen)
    store.initialize()
    acknowledgments: list[Json] = []
    for index, tx in enumerate(frozen["transactions"]):

        def hook(phase: str) -> None:
            if index == 31 and phase == barrier and barrier != "after_acknowledgment":
                atomic_json(
                    directory / "crash_marker.json", dict(barrier=barrier, sequence=index + 1)
                )
                os._exit(73)

        reply = store.apply(tx, hook)
        acknowledgments.append(dict(id=tx["id"], response=reply))
        atomic_json(directory / "acknowledged.json", dict(rows=acknowledgments))
        if index == 31 and barrier == "after_acknowledgment":
            atomic_json(directory / "crash_marker.json", dict(barrier=barrier, sequence=index + 1))
            os._exit(73)
        if index % 16 == 0:
            progress("worker_transactions", index + 1, len(frozen["transactions"]) - index - 1)
    return 0


def evidence(frozen: Json, directory: Path, arm: str, barrier: str) -> Json:
    """Keep killed bytes, caller acknowledgments and resumed state as separate operands."""
    atomic_json(directory / "frozen.json", frozen)
    receipt = child(
        f"{arm}_{barrier}",
        [
            sys.executable,
            "-u",
            str(ROOT / CLI),
            "--worker",
            str(directory / "frozen.json"),
            "--store",
            str(directory),
            "--arm",
            arm,
            "--barrier",
            barrier,
        ],
        directory / "logs",
        expected=73,
        deadline=180,
    )
    shutil.copytree(directory / "store", directory / "crashed")
    store = (k.Journal if arm == "journal" else k.Snapshot)(directory / "store", frozen)
    recovered = store.read()
    acknowledged = json.loads((directory / "acknowledged.json").read_bytes())["rows"]
    expected = k.oracle(frozen, len(frozen["transactions"]))
    prefix_ok = recovered["ledger"][: len(acknowledged)] == [a["id"] for a in acknowledged]
    replies = [store.apply(tx) for tx in frozen["transactions"]]
    if isinstance(store, k.Journal):
        store.drain()
    return dict(
        kind="crash",
        arm=arm,
        barrier=barrier,
        frozen_reference=reference(directory / "frozen.json"),
        directory=str(directory),
        recovered=recovered,
        final=store.read(),
        acknowledgments=acknowledged,
        receipt=receipt,
        actual_exit=receipt["actual_exit"],
        passed=receipt["passed"]
        and prefix_ok
        and replies == expected["replies"]
        and store.read() == expected,
        response_hash=canonical_hash(replies),
        recovered_state_hash=canonical_hash(recovered["state"]),
    )


def equivalence(frozen: Json, directory: Path, sealed: Json) -> Json:
    """Compare actual disk stores and Exp8464's independent expected state and replies."""
    atomic_json(directory / "frozen.json", frozen)
    journal = k.Journal(directory / "journal", frozen)
    stores: list[k.Journal | k.Snapshot] = [journal, k.Snapshot(directory / "snapshot", frozen)]
    for store in stores:
        store.initialize()
    replies = []
    for index, tx in enumerate(frozen["transactions"]):
        pair = [store.apply(tx) for store in stores]
        if canonical_hash(pair[0]) != canonical_hash(pair[1]):
            raise ValueError("response_equivalence")
        replies.append(pair[0])
        if index % 16 == 0:
            progress("equivalence_transactions", index + 1, len(frozen["transactions"]) - index - 1)
    journal.drain()
    state_hash = canonical_hash(stores[0].read()["state"])
    hashes = [
        "sha256:"
        + __import__("hashlib").sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest()
        for rows in replies
    ]
    if state_hash != sealed["state_hash"] or canonical_hash(hashes) != sealed["response_hash"]:
        raise ValueError("independent_8464_expectation")
    return dict(
        kind="equivalence",
        arm="both",
        barrier=str(frozen["cell"]["batch"]),
        directory=str(directory),
        frozen_reference=reference(directory / "frozen.json"),
        final=stores[0].read(),
        passed=stores[0].read() == stores[1].read(),
        actual_exit=0,
        acknowledgments=[
            dict(id=t["id"], response=r)
            for t, r in zip(frozen["transactions"], replies, strict=True)
        ],
        response_hash=canonical_hash(replies),
        recovered_state_hash=state_hash,
    )


def measure(root: Path, raw: Path, private: Path) -> Json:
    """Admission keeps optional CUDA and boards separate from required CPU operands."""
    progress("preconditions_before")
    start = time.monotonic_ns()
    raw.mkdir(parents=True, exist_ok=True, mode=0o700)
    raw.chmod(0o700)
    private.mkdir(parents=True, exist_ok=True, mode=0o700)
    checked = upstream.authority.authority(root, raw / "authority")
    resources = dict(
        disk_backed=upstream.filesystem(raw)["disk_backed"]
        and upstream.filesystem(private)["disk_backed"],
        memory_available=int(Path("/proc/meminfo").read_text().split("MemAvailable:")[1].split()[0])
        > 256000,
        disk_free=shutil.disk_usage(private).free > 2_000_000_000,
        tools={
            name: os.access(ROOT / ".venv/bin" / name, os.X_OK)
            for name in ("python", "pytest", "coverage", "ruff", "mypy")
        },
        private_scratch=str(private),
        task_cap_s=4800,
        measurement_cap_s=3600,
        validation_reserve_s=900,
        optional_cuda_required=False,
        optional_board_required=False,
    )
    resources.update(
        available_memory_bytes=int(
            Path("/proc/meminfo").read_text().split("MemAvailable:")[1].split()[0]
        )
        * 1024,
        disk_free_bytes=shutil.disk_usage(private).free,
        raw_filesystem=upstream.filesystem(raw),
        private_filesystem=upstream.filesystem(private),
        raw_permissions=oct(raw.stat().st_mode & 0o777),
        private_permissions=oct(private.stat().st_mode & 0o777),
    )
    probe = private / "journal_disk_probe"
    probe.write_bytes(b"journal8465 actual readback")
    resources["scratch_readback"] = probe.read_bytes() == b"journal8465 actual readback"
    gates = []
    if not checked["activated"]:
        gates.append(
            upstream.gate(
                root / upstream.authority.ACTIVE,
                "activated_V728_full_contract",
                True,
                checked["activated"],
                TASK,
            )
        )
    task = [t for t in checked["tasks"] if t["id"] == TASK]
    if (
        len(task) != 1
        or task[0].get("MODEL_SPECS") != []
        or task[0].get("inference_substrate_class") != "no_model_load"
    ):
        gates.append(
            upstream.gate(root / upstream.authority.ACTIVE, "exact_task_contract", TASK, task, TASK)
        )
    for name in ("disk_backed", "memory_available", "disk_free", "scratch_readback"):
        if not resources[name]:
            gates.append(upstream.gate(private, name, True, resources[name], TASK))
    for name, ready in resources["tools"].items():
        if not ready:
            gates.append(
                upstream.gate(ROOT / ".venv/bin" / name, "tool_available", True, None, TASK)
            )
    work: Json = dict(
        root=str(root),
        raw=str(raw),
        refs=[],
        primitives=[],
        rows=[],
        bundle={},
        workload={},
        historical_model_provenance=[],
        gate_check_summary=gates,
        owned_failures={},
        authority=checked,
        preconditions_checked=resources,
        started_monotonic_ns=start,
        phase_spans=[],
    )
    try:
        work.update(dependencies(root, raw, private))
    except (OSError, ValueError, KeyError, TypeError, StopIteration) as error:
        gates.append(
            upstream.gate(
                root / UPSTREAM,
                "sealed_8464_service_tape",
                True,
                str(error) if (root / UPSTREAM).exists() else None,
                "exp8464-persistence-localization",
            )
        )
    progress("preconditions_after")
    for number, (kind, arm, barrier) in enumerate(SLOTS):
        unit = f"{kind}-{arm}-{barrier}"
        row: Json = dict(
            unit_id=unit,
            family="journal8465_engineering_fixture",
            kind=kind,
            arm=arm,
            barrier=barrier,
            status="censored",
            passed=None,
            absolute_metric=None,
            missing_reason="required_operands_missing",
        )
        if not gates:
            progress("qualification_before_" + unit, number, len(SLOTS) - number)
            try:
                if (time.monotonic_ns() - start) / 1e9 >= 3600:
                    raise TimeoutError("measurement_deadline")
                cell = dict(
                    workload="natural",
                    batch=int(barrier) if kind == "equivalence" else 1,
                    predictions_per_update=8,
                )
                frozen = k.freeze(work["bundle"], cell, 0)
                directory = raw / unit
                if kind == "equivalence":
                    sealed = next(
                        e
                        for e in work["workload"]["expected_state_response_hashes"]
                        if e["cell"] == cell and e["repeat"] == 0
                    )
                    primitive = equivalence(frozen, directory, sealed)
                else:
                    primitive = evidence(frozen, directory, arm, barrier)
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
            except (OSError, ValueError, KeyError, TypeError, TimeoutError, StopIteration) as error:
                work["owned_failures"][unit] = str(error)
                row.update(status="failed", missing_reason=str(error), passed=False)
            progress("qualification_after_" + unit, number + 1, len(SLOTS) - number - 1)
        work["rows"].append(row)
    work["code_config_hashes"] = [
        reference(ROOT / path) for path in [*OWNED, TEST, *upstream.shared.GUARDS]
    ]
    work["ended_monotonic_ns"] = time.monotonic_ns()
    work["phase_spans"] = [
        dict(phase="admission_and_qualification", start_ns=start, end_ns=work["ended_monotonic_ns"])
    ]
    atomic_json(raw / "measurement.json", work)
    return work


def coverage_ready(report: Json) -> bool:
    """Use the unchanged evidence-based coverage consumer with this task's owned files."""
    from unittest.mock import patch

    with patch.object(upstream, "OWNED", OWNED):
        return bool(upstream.coverage_ready(report))


def build(work: Json, receipts: list[Json], output: Path) -> Json:
    """Engineering oracle equivalence cannot establish scientific or speed benefit."""
    rows = work["rows"]
    complete = len(rows) == len(SLOTS) and all(r["passed"] is True for r in rows)
    passed = bool(receipts) and all(c["passed"] for c in receipts)
    required = REQUIRED <= {c.get("name") for c in receipts if c["passed"]}
    covered = coverage_ready(work.get("owned_coverage", {}))
    flags = work.get("adversarial_findings", [])
    verdict = (
        "disqualified"
        if not passed or flags or work["owned_failures"]
        else "blocked"
        if work["gate_check_summary"]
        else "circular_positive"
        if complete and required and covered
        else "disqualified"
    )
    primitives = [read(ref) for ref in work["primitives"]]
    value: Json = dict(
        experiment_id=8465,
        task_id=TASK,
        milestone="2026.10.728",
        run_date="20261012",
        honest_verdict="complete_" + verdict + "_journal_equivalence",
        verdict_class=verdict,
        gate_check_summary=work["gate_check_summary"],
        inference_substrate="verifier_ensemble_against_cached_candidates",
        reader_substrate="aggregation_from_upstream_artifacts",
        inference_substrate_class="no_model_load",
        MODEL_SPECS=[],
        model_invocation_counts=dict(ZERO_INVOCATION_COUNTS, current_llm_invocations=0),
        historical_model_provenance=work["historical_model_provenance"],
        rows=rows,
        intended_count=len(SLOTS),
        completed_count=sum(r["status"] == "completed" for r in rows),
        failed_count=sum(r["status"] == "failed" for r in rows),
        censored_count=sum(r["status"] == "censored" for r in rows),
        excluded_count=0,
        independent_count=0,
        sample_size_budget=dict(
            cells=2, crash_arms=2, boundaries_per_arm=8, independent_science_questions=0
        ),
        verifier_is_oracle=True,
        exposure_scope="retrospective_cached_five_input_engineering_fixture",
        independent_generalization_score=0,
        generalized_learning_benefit_score=0,
        required_checks_passed=passed and required,
        flagged_adversarial=bool(flags) or not passed,
        acceptance_gates=dict(
            complete_equivalence=complete,
            authentic_operands=not work["gate_check_summary"],
            required_validation=passed and required,
            owned_coverage=covered,
            no_unresolved_findings=not flags,
        ),
        validation_receipts=receipts,
        terminal_validation_sidecar_path=str(Path(work["raw"]) / "terminal_validation.json"),
        adversarial_findings=flags,
        preconditions_checked=work["preconditions_checked"],
        duration_s=(work["ended_monotonic_ns"] - work["started_monotonic_ns"]) / 1e9,
        phase_spans=work["phase_spans"],
        random_seed=7288465,
        source_artifact_hashes=work["refs"],
        code_config_hashes=work["code_config_hashes"],
        raw_shard_hashes=work["primitives"] + work.get("coverage_shards", []),
        cited_upstream_artifacts=[
            dict(
                ref,
                imported_fields="original authority, five-input head/slots/labels, Exp8464 exact tape and expected states/replies",
            )
            for ref in work["refs"]
        ],
        journal_ready_score=int(verdict == "circular_positive"),
        equivalence_rows=[r for r in rows if r["kind"] == "equivalence"],
        crash_rows=[r for r in rows if r["kind"] == "crash"],
        acknowledgment_ledger=[
            dict(unit_id=r["directory"], arm=r["arm"], rows=r["acknowledgments"])
            for r in primitives
        ],
        recovered_state_hashes=[
            dict(unit_id=r["unit_id"], sha256=r.get("recovered_state_hash")) for r in rows
        ],
        journal_limits=dict(
            MAX_EVENTS=256,
            MAX_BYTES=2 * 1024 * 1024,
            journal_ceiling_bytes=k.JOURNAL_BYTES,
            checkpoint_every_committed_transactions=32,
            final_drain_checkpoint=True,
            overflow="blocks",
            reclamation="none_all_segments_retained",
        ),
        durability_scope="actual file/directory fsync and private process-exit recovery; physical power-loss behavior is not certified",
        owned_coverage=work.get("owned_coverage", {}),
        work_reference=reference(Path(work["raw"]) / "measurement.json"),
        closed_exposed_utility=dict(H1=-0.00390625, H2=0, reopened=False),
        preserved_history=dict(
            V727_methods_ready_score=0,
            V727_science_executed=False,
            V726_admission_learning="null",
            V726_kernel_percent=0.093,
            V726_persistence_percent=96.44,
        ),
        speed_claim=False,
        methodology_note="Two independently sealed natural service cells, unchanged delayed transition oracle, bounded experimental journal and retained production snapshots. Sixteen real exits and exact cold replay qualify durability representation only. No model load, learning benefit or speed measurement.",
    )
    value["field_principles"] = {
        key: "Bind immutable invocation operands; engineering equivalence grants no scientific benefit."
        for key in value
    }
    value["field_principles"].update(
        journal_ready_score="All causal equivalence, crash, tamper, coverage and required validation gates must pass.",
        acknowledgment_ledger="Actual returned durable acknowledgment IDs and responses survive every tested exit.",
        durability_scope="Process exits and real fsync calls do not certify loss of physical power.",
        journal_limits="Explicit overflow rejects history growth; all checkpoints and segments remain available to pinned readers.",
    )
    value["reproducibility_checksum"] = checksum(value)
    return value


def replay(path: Path) -> bool:
    """A fresh reader reconstructs primitive meaning instead of trusting repaired hashes."""
    try:
        value = json.loads(path.read_bytes())
        if value["experiment_id"] != 8465 or value["reproducibility_checksum"] != checksum(value):
            return False
        work = read(value["work_reference"])
        if len(work["rows"]) != len(SLOTS):
            return False
        if (
            work["authority"]["activated"]
            and upstream.authority.tasks_digest(work["authority"]["tasks"])
            != upstream.authority.TASKS_PIN
        ):
            return False
        with TemporaryDirectory(dir="/var/tmp", prefix="exp8465-replay-") as folder:
            private = Path(folder)
            if work["bundle"]:
                deps = dependencies(Path(work["root"]), private, private)
                if canonical_hash(deps["bundle"]) != canonical_hash(
                    work["bundle"]
                ) or canonical_hash(deps["workload"]) != canonical_hash(work["workload"]):
                    return False
            for ref in [
                *value["source_artifact_hashes"],
                *value["code_config_hashes"],
                *value["raw_shard_hashes"],
            ]:
                if upstream.shared.sha256_file(Path(ref["path"])) != ref["sha256"]:
                    return False
            for index, row in enumerate(work["rows"]):
                if (row["kind"], row["arm"], row["barrier"]) != SLOTS[index]:
                    return False
                if "primitive_reference" not in row:
                    continue
                primitive = read(row["primitive_reference"])
                frozen = read(primitive["frozen_reference"])
                cell = dict(
                    workload="natural",
                    batch=int(row["barrier"]) if row["kind"] == "equivalence" else 1,
                    predictions_per_update=8,
                )
                if k.encode(frozen) != k.encode(k.freeze(work["bundle"], cell, 0)):
                    return False
                for ref in primitive["store_references"]:
                    if upstream.shared.sha256_file(Path(ref["path"])) != ref["sha256"]:
                        return False
                expected = k.oracle(frozen, len(frozen["transactions"]))
                if canonical_hash(primitive["final"]) != canonical_hash(expected) or primitive[
                    "response_hash"
                ] != canonical_hash(expected["replies"]):
                    return False
                arms = ["journal", "snapshot"] if row["kind"] == "equivalence" else [row["arm"]]
                for arm in arms:
                    original = Path(primitive["directory"]) / (
                        arm if row["kind"] == "equivalence" else "crashed"
                    )
                    copy = private / f"row{index}-{arm}"
                    shutil.copytree(original, copy)
                    store = (k.Journal if arm == "journal" else k.Snapshot)(copy, frozen)
                    observed = store.read()
                    if row["kind"] == "equivalence":
                        if canonical_hash(observed) != canonical_hash(expected):
                            return False
                    else:
                        n = 32 if row["barrier"] == "after_acknowledgment" else 31
                        acknowledged = [
                            dict(id=t["id"], response=r)
                            for t, r in zip(
                                frozen["transactions"][:n], expected["replies"][:n], strict=True
                            )
                        ]
                        if canonical_hash(primitive["acknowledgments"]) != canonical_hash(
                            acknowledged
                        ) or canonical_hash(observed) != canonical_hash(primitive["recovered"]):
                            return False
                        if (
                            observed["sequence"] not in (31, 32)
                            or observed["ledger"][:n] != expected["ledger"][:n]
                        ):
                            return False
                        if (
                            primitive["receipt"]["actual_exit"] != 73
                            or not primitive["receipt"]["passed"]
                        ):
                            return False
                        marker = json.loads(
                            (Path(primitive["directory"]) / "crash_marker.json").read_bytes()
                        )
                        if marker != dict(barrier=row["barrier"], sequence=32):
                            return False
                        if canonical_hash(observed["state"]) != primitive["recovered_state_hash"]:
                            return False
                if (
                    row["passed"] != primitive["passed"]
                    or row["recovered_state_hash"] != primitive["recovered_state_hash"]
                ):
                    return False
        return bool(k.encode(value) == k.encode(build(work, value["validation_receipts"], path)))
    except (OSError, ValueError, KeyError, TypeError, StopIteration, IndexError):
        return False


def reader(
    frozen_path: Path, directory: Path, arm: str, pin: Path, release: Path, timeout_s: float = 30
) -> int:
    """A real reader pins an old immutable value while another process publishes later work."""
    frozen = json.loads(frozen_path.read_bytes())
    store = (k.Journal if arm == "journal" else k.Snapshot)(directory, frozen)
    view = store.read()
    pointer = directory / ("checkpoint-current.json" if arm == "journal" else "current.json")
    pinned_file = directory / json.loads(pointer.read_bytes())["file"]
    digest = upstream.shared.sha256_file(pinned_file)
    atomic_json(pin, view)
    start = time.monotonic()
    while not release.exists():
        if time.monotonic() - start >= timeout_s:
            progress("reader_timeout", 0, 1)
            return 1
        progress("reader_waiting_for_writer", 0, 1)
        time.sleep(0.5)
    passed = pinned_file.exists() and upstream.shared.sha256_file(pinned_file) == digest
    progress("reader_released", int(passed), int(not passed))
    return int(not passed)
