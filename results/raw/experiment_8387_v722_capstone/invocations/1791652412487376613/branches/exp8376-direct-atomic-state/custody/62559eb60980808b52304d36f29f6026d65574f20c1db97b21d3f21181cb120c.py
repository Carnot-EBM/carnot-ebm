"""REQ-REPORT-8376: actual kills qualify direct transactions, never semantic benefit."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import json
import os
from pathlib import Path
import resource
import signal
import shutil
import subprocess
import sys
from threading import Event
import time
from typing import Any

import coverage
import yaml

from carnot.reporting import v722_contract_methods as authority
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash, sha256_file
from carnot.reporting.current_work_receipt import ZERO_INVOCATION_COUNTS
from carnot.reporting.v709_execution import child
from carnot.verify import direct_atomic_state_8376 as s

Json = dict[str, Any]
ROOT = s.ROOT
NAME = "experiment_8376_v722_direct_atomic_state"
TASK = "exp8376-direct-atomic-state"
CLI = f"scripts/experiments/{NAME}.py"
TEST = "tests/python/test_direct_atomic_state_8376.py"
OWNED = [
    "python/carnot/verify/direct_atomic_state_8376.py",
    "python/carnot/reporting/direct_atomic_state_8376.py",
    "python/carnot/reporting/direct_atomic_runner_8376.py",
    CLI,
]
MODEL_SPECS: list[Json] = []


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Flush completed counts at real boundaries so a parent can detect stalled work."""
    print(f"[exp8376] phase={phase} completed={completed} pending={pending}", flush=True)


def reference(path: Path) -> Json:
    """Bind exact operands so replay cannot absorb later source edits."""
    return dict(path=str(path.absolute()), sha256=sha256_file(path))


def worker(frozen_path: Path, path: Path, barrier: str, readers: bool) -> int:
    """Real killed children record intended and visible state before losing their process."""
    frozen = json.loads(frozen_path.read_bytes())
    store = s.Store(path, frozen)
    summaries: list[Json] = []
    acknowledged: list[Json] = []

    def reader(ready: Event, released: Event) -> Json:
        pinned = store.read()
        digest = canonical_hash(pinned)
        ready.set()
        released.wait(timeout=30)
        return dict(
            state_hash=digest,
            version=pinned["version"],
            mixed=int(pinned != s.fold(frozen, pinned["version"])),
        )

    with ThreadPoolExecutor(max_workers=2) as pool:
        for index, event in enumerate(frozen["events"]):
            before = store.read()
            ready = [Event(), Event()]
            released = Event()
            futures = [pool.submit(reader, flag, released) for flag in ready] if readers else []
            for flag in ready if readers else []:
                if not flag.wait(timeout=30):
                    raise TimeoutError("reader_deadline")

            def hook(phase: str) -> None:
                if phase == barrier and index == frozen.get("crash_event_index", 9):
                    intended, result = s.transition(before, event)
                    visible = store.read()
                    atomic_json(
                        path / "kill.json",
                        dict(
                            barrier=phase,
                            event_index=index,
                            event=event,
                            pre_state=before,
                            intended_state=intended,
                            visible_state=visible,
                            result=result,
                            acknowledged=acknowledged,
                            reader_rows=summaries,
                            directory_fsync=phase in s.BARRIERS[2:],
                        ),
                    )
                    current = coverage.Coverage.current()
                    progress("SIGKILL_" + phase, index, len(frozen["events"]) - index)
                    # The forked helper saves the caller's real executed lines before
                    # sending SIGKILL. It does not substitute a normal exit for death.
                    subprocess.run(
                        ["/usr/bin/kill", "-KILL", str(os.getpid())],
                        preexec_fn=current.save if current is not None else None,
                        timeout=5,
                        check=True,
                    )

            result = store.apply(event, hook)
            version_before_retry = store.read()["version"]
            retry = store.apply(event)
            if retry != result:
                raise ValueError("retry_result_drift")
            if store.read()["version"] != version_before_retry:
                raise ValueError("duplicate_update")
            acknowledged.append(dict(event=event, result=result))
            released.set()
            summaries.extend(future.result(timeout=30) for future in futures)
            if (index + 1) % 8 == 0:
                progress("worker_events", index + 1, len(frozen["events"]) - index - 1)
    store.cleanup()
    atomic_json(
        path / "worker.json",
        dict(
            state=store.read(),
            reader_rows=summaries,
            acknowledged=acknowledged,
            peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        ),
    )
    return 0


def measure_trace(frozen: Json, raw: Path) -> list[Json]:
    """Every intended seed/barrier/arm keeps its actual exits and exact state evidence."""
    raw.mkdir(parents=True, exist_ok=True)
    trace_path = raw / "trace.json"
    atomic_json(trace_path, frozen)
    rows = []
    arms = [("uninterrupted", "none")]
    arms += [(arm, barrier) for barrier in s.BARRIERS for arm in ("restart", "two_readers")]
    for number, (arm, barrier) in enumerate(arms):
        progress("crash_arm_before", number, len(arms) - number)
        path = raw / (arm + "-" + barrier)
        store = s.Store(path, frozen)
        store.initialize()
        prefix = [
            sys.executable,
            "-u",
            str(ROOT / CLI),
            "--worker",
            str(trace_path),
            "--state-directory",
            str(path),
        ]
        args = ["--two-readers"] if arm == "two_readers" else []
        killed = child(
            "execute",
            prefix + args + ["--barrier", barrier],
            path / "logs",
            deadline=60,
            expected=0 if barrier == "none" else -9,
        )
        visible = store.read()
        marker = json.loads((path / "kill.json").read_bytes()) if barrier != "none" else {}
        receipts = [killed]
        if barrier != "none":
            receipts.append(child("resume", prefix + args, path / "logs", deadline=60))
        final = store.read()
        details = json.loads((path / "worker.json").read_bytes())
        expected = s.fold(frozen)
        recovery = int(final != expected)
        mixed = sum(r["mixed"] for r in details["reader_rows"] + marker.get("reader_rows", []))
        mutated = sum(final["issued"].get(k) != v for k, v in visible["issued"].items())
        duplicates = abs(final["release_cursor"] - len(final["applied"]))
        acknowledged = details["acknowledged"] + marker.get("acknowledged", [])
        lost = sum(
            final["issued" if r["event"]["kind"] == "issue" else "applied"].get(r["event"]["id"])
            != r["result"]
            for r in acknowledged
        )
        crash_ok = not marker or (
            marker["visible_state"] == visible
            and visible == marker["intended_state" if barrier in s.BARRIERS[2:] else "pre_state"]
        )
        passed = (
            all(r["passed"] for r in receipts)
            and not (recovery or mixed or mutated or duplicates or lost)
            and crash_ok
            and details["peak_rss_bytes"] <= 1536 * 1024 * 1024
        )
        rows.append(
            dict(
                seed=frozen["seed"],
                arm=arm,
                barrier=barrier,
                status="completed",
                state_hash=canonical_hash(final),
                recovery_mismatch_count=recovery,
                mixed_version_read_count=mixed,
                duplicate_update_count=duplicates,
                lost_acknowledged_update_count=lost,
                issued_prediction_mutation_count=mutated,
                passed=passed,
                actual_exit=killed["actual_exit"],
                latency_s=sum(r["duration_s"] for r in receipts),
                peak_rss_bytes=details["peak_rss_bytes"],
                missing_reason=None,
                trace_reference=reference(trace_path),
                final_state_reference=reference(path / "worker.json"),
                kill_reference=reference(path / "kill.json") if marker else None,
                receipts=receipts,
                absolute_metric=int(passed),
                raw_numerator=int(passed),
                raw_denominator=1,
            )
        )
        progress("crash_arm_after", number + 1, len(arms) - number - 1)
    return rows


def measure(root: Path, raw: Path, private: Path) -> Json:
    """Input custody and external authority remain separate from measured recovery."""
    began = time.monotonic()
    progress("preconditions_before")
    raw.mkdir(parents=True, exist_ok=True, mode=0o700)
    checked = authority.authority(root, raw / "authority")
    gates = list(checked["gate_check_summary"])
    code_refs = [reference(ROOT / p) for p in [*OWNED, TEST]]
    refs = []
    inputs = [
        "AGENTS.md",
        "CODEX.md",
        "CLAUDE.md",
        "ops/e2e-test-plan.md",
        "ops/exclusion_manifest.yaml",
        "openspec/capabilities/research-reporting/spec.md",
        "openspec/capabilities/verification/spec.md",
        "scripts/experiment_template.py",
        "python/carnot/reporting/primary_publication.py",
        authority.DESIGN,
        "openspec/change-proposals/research-roadmap-v721-preserved-20261010.md",
        "python/carnot/verify/local_update_isolation_8306.py",
        "python/carnot/verify/continuous_local_learning_8348.py",
        "python/carnot/verify/spline_table_fidelity_8352.py",
        "python/carnot/verify/hard_exit_learning_qualification_8206.py",
        "python/carnot/reporting/current_work_receipt.py",
        "tests/python/test_hard_exit_learning_qualification_8206.py",
        "results/experiment_8363_atomic_table_state.json",
        "results/experiment_8374_v722_contract_methods.json",
        authority.ACTIVE,
        s.PROTOCOL,
        authority.METHODS,
    ]
    for index, name in enumerate(inputs):
        source = root / name
        if not source.is_file():
            gates.append(authority.failure(source, "input_available", True, None))
            continue
        snapshot = raw / "inputs" / (str(index) + ".bin")
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        snapshot.write_bytes(source.read_bytes())
        refs.append(
            dict(reference(snapshot), source_path=str(source), imported_fields="read-only input")
        )
    mounts = [line.split() for line in Path("/proc/self/mountinfo").read_text().splitlines()]
    mounted = max(
        (m for m in mounts if private.resolve().is_relative_to(Path(m[4]))), key=lambda m: len(m[4])
    )
    filesystem = mounted[mounted.index("-") + 1]
    available = (
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
        name: (ROOT / ".venv/bin" / name).is_file() for name in ("python", "pytest", "ruff", "mypy")
    }
    tools["system_kill"] = Path("/usr/bin/kill").is_file()
    preconditions = dict(
        scratch=str(private),
        filesystem=filesystem,
        disk_backed=filesystem not in {"tmpfs", "ramfs"},
        available_memory_bytes=available,
        tools=tools,
        task_cap_s=4800,
        free_disk_bytes=shutil.disk_usage(private).free,
        private_mode=oct(private.stat().st_mode & 0o777),
        writer_rss_cap_bytes=1536 * 1024 * 1024,
        state_size_cap_bytes=s.MAX_BYTES,
        event_cap=s.MAX_EVENTS,
        no_model_load=True,
    )
    resource_ok = preconditions["disk_backed"] and available >= 536870912 and all(tools.values())
    if not resource_ok:
        gates.append(authority.failure(private, "resource_preconditions", True, preconditions))
    protocol = root / s.PROTOCOL
    ready = protocol.is_file() and sha256_file(protocol) == authority.PROTOCOL_PIN
    if not ready:
        gates.append(
            authority.failure(
                protocol,
                "protocol_hash",
                authority.PROTOCOL_PIN,
                sha256_file(protocol) if protocol.is_file() else None,
            )
        )
    if (root / authority.ACTIVE).is_file():
        active = yaml.safe_load((root / authority.ACTIVE).read_bytes())
        task = [t for t in active["tasks"] if t["id"] == TASK]
        if len(task) != 1 or task[0]["MODEL_SPECS"] != []:
            gates.append(authority.failure(root / authority.ACTIVE, "exact_task", TASK, task))
        preconditions["task_sha256"] = canonical_hash(task)
    frozen_refs = []
    if ready:
        deployment = json.loads(protocol.read_bytes())
        methods = json.loads((root / authority.METHODS).read_bytes())
        if sha256_file(root / authority.METHODS) != authority.METHODS_PIN:
            ready = False
            gates.append(
                authority.failure(
                    root / authority.METHODS,
                    "methods_hash",
                    authority.METHODS_PIN,
                    sha256_file(root / authority.METHODS),
                )
            )
        for operand in [deployment["checkpoint"], *methods["reusable"].values()]:
            source = Path(operand["path"])
            if not source.is_file() or sha256_file(source) != operand["sha256"]:
                ready = False
                gates.append(
                    authority.failure(
                        source,
                        "qualified_input_hash",
                        operand["sha256"],
                        sha256_file(source) if source.is_file() else None,
                    )
                )
            else:
                imported = json.loads(source.read_bytes())
                field = operand.get("readiness_field")
                if field and imported.get(field) != 1:
                    ready = False
                    gates.append(authority.failure(source, field, 1, imported.get(field)))
                if source == Path(deployment["checkpoint"]["path"]):
                    selected = next(h for h in imported["heads"] if h["arm"] == "spline34")
                    if any(
                        selected[k] != deployment["head"][k]
                        for k in ("coefficients", "temperature")
                    ):
                        raise ValueError("frozen_head_drift")
                refs.append(
                    dict(
                        reference(source),
                        imported_fields=operand.get("readiness_field", "frozen head"),
                    )
                )
        for seed in (11, 22, 33):
            frozen_path = raw / f"frozen-trace-{seed}.json"
            atomic_json(frozen_path, s.trace(seed, deployment["head"]))
            frozen_refs.append(reference(frozen_path))
    progress("preconditions_after", len(refs), max(0, len(inputs) - len(refs)))
    rows = []
    for ref in frozen_refs if ready and resource_ok else []:
        frozen = json.loads(Path(ref["path"]).read_bytes())
        progress("benchmark_before_seed_" + str(frozen["seed"]))
        rows.extend(measure_trace(frozen, raw / str(frozen["seed"])))
        progress("benchmark_after_seed_" + str(frozen["seed"]), len(rows), 27 - len(rows))
    if not rows:
        for seed in (11, 22, 33):
            for arm, barrier in [
                ("uninterrupted", "none"),
                *[(a, b) for b in s.BARRIERS for a in ("restart", "two_readers")],
            ]:
                rows.append(
                    dict(
                        seed=seed,
                        arm=arm,
                        barrier=barrier,
                        status="unstarted",
                        passed=None,
                        state_hash=None,
                        actual_exit=None,
                        latency_s=None,
                        recovery_mismatch_count=None,
                        mixed_version_read_count=None,
                        duplicate_update_count=None,
                        issued_prediction_mutation_count=None,
                        lost_acknowledged_update_count=None,
                        receipts=[],
                        kill_reference=None,
                        missing_reason="authenticated_direct_input_or_resource_absent",
                        absolute_metric=None,
                        raw_numerator=None,
                        raw_denominator=1,
                    )
                )
    primitive = raw / "primitive_rows.json"
    atomic_json(primitive, dict(rows=rows, traces=frozen_refs))
    complete = sum(r["status"] == "completed" for r in rows)
    progress("measurement_complete", complete, 27 - complete)
    work = dict(
        rows=rows,
        traces=frozen_refs,
        gate_check_summary=gates,
        input_ready=ready,
        source_artifact_hashes=refs,
        primitive_reference=reference(primitive),
        code_config_hashes=code_refs,
        resource_preconditions_passed=resource_ok,
        preconditions_checked=preconditions,
        duration_s=time.monotonic() - began,
        phase_spans=[
            dict(phase="authentication_and_crash_measurement", duration_s=time.monotonic() - began)
        ],
        invocation_id=str(time.time_ns()),
        work_path=str(raw / "work.json"),
        terminal_validation_sidecar_path=str(raw / "terminal_validation.json"),
    )
    atomic_json(Path(work["work_path"]), work)
    return work


def build(work: Json, receipts: list[Json]) -> Json:
    """Only exact recovery with owned validation can grant engineering readiness."""
    rows = work["rows"]
    completed = [r for r in rows if r["status"] == "completed"]
    owned = [r for r in receipts if r.get("scope") != "global"]
    passed = (
        bool(owned)
        and all(r["passed"] for r in owned)
        and all(r["passed"] for r in completed)
        and work["resource_preconditions_passed"]
    )
    gates = [
        dict(
            g,
            upstream=g["upstream_id"],
            path=g["artifact_path"],
            sha256=g["artifact_hash"],
            field=g["artifact_field"],
            operator=g["op"],
            expected_value=g["expected"],
            observed_value=g["observed"],
        )
        for g in work["gate_check_summary"]
    ]
    kind = "disqualified" if not passed else "blocked" if gates else "circular_positive"
    ready = passed and not gates and len(completed) == 27 and work["input_ready"]
    counts = {
        key: sum(r[key] for r in completed)
        for key in (
            "recovery_mismatch_count",
            "mixed_version_read_count",
            "duplicate_update_count",
            "issued_prediction_mutation_count",
            "lost_acknowledged_update_count",
        )
    }
    value = dict(
        experiment_id=8376,
        task_id=TASK,
        milestone="2026.10.722",
        run_date="20261010",
        honest_verdict="complete_" + kind + "_direct_atomic_state",
        verdict_class=kind,
        gate_check_summary=gates,
        inference_substrate="verifier_ensemble_against_cached_candidates",
        inference_substrate_class="no_model_load",
        MODEL_SPECS=[],
        no_model_load=True,
        model_invocation_counts=dict(ZERO_INVOCATION_COUNTS),
        historical_model_provenance=work["source_artifact_hashes"],
        rows=rows,
        intended_count=27,
        completed_count=len(completed),
        failed_count=sum(not r["passed"] for r in completed),
        censored_count=27 - len(completed),
        excluded_count=0,
        independent_count=0,
        sample_size_budget=dict(
            seeds=[11, 22, 33],
            barriers=list(s.BARRIERS),
            arms=3,
            intended=27,
            constructed_only=True,
            repeated_timings_are_independent=False,
        ),
        verifier_is_oracle=True,
        exposure_scope="constructed process recovery; exposed cached head",
        independent_generalization_score=0,
        generalized_learning_benefit_score=0,
        required_checks_passed=passed,
        flagged_adversarial=not passed,
        acceptance_gates=dict(
            owned_validation=passed,
            exact_recovery=len(completed) == 27 and all(r["passed"] for r in completed),
            external_authority=not gates,
            direct_ready=bool(ready),
        ),
        validation_receipts=receipts,
        terminal_validation_sidecar_path=work["terminal_validation_sidecar_path"],
        adversarial_findings=work.get("adversarial_findings", []),
        preconditions_checked=work["preconditions_checked"],
        duration_s=work["duration_s"],
        phase_spans=work["phase_spans"],
        random_seed=[11, 22, 33],
        source_artifact_hashes=work["source_artifact_hashes"],
        code_config_hashes=work["code_config_hashes"],
        raw_shard_hashes=[work["primitive_reference"], *work["traces"]],
        cited_upstream_artifacts=work["source_artifact_hashes"],
        direct_state_ready_score=int(ready),
        state_version=s.VERSION,
        crash_rows=[r for r in rows if r["barrier"] != "none"],
        **counts,
        primitive_reference=work["primitive_reference"],
        invocation_id=work["invocation_id"],
        work_reference=reference(Path(work["work_path"])),
        repository_health=[r for r in receipts if r.get("scope") == "global"],
        table_ready_score=0,
        power_loss_certified=False,
        methodology_note="One writer, pinned immutable readers, direct SciPy evaluation and frozen sparse optimizer. Constructed seeds11/22/33; genuine SIGKILL at four durability barriers; exact replay from sealed events. Numeric parity and process recovery establish no semantic or generalization benefit. Closed H1/H2 and original protocol bytes remain unchanged. No model acquisition.",
    )
    value["field_principles"] = {
        key: "Bind invocation-local direct recovery evidence; execution is not semantic benefit."
        for key in [*value, "reproducibility_checksum", "field_principles"]
    }
    value["field_principles"].update(
        direct_state_ready_score="Require all owned checks, authenticated authority and27 exact recovered arms; never credit table proof.",
        state_version="Name the direct-only immutable schema shared by coefficients, issued decisions, pending IDs and feedback results.",
        crash_rows="Retain one actual SIGKILL and resumed child receipt for every intended seed, barrier and recovery arm.",
        recovery_mismatch_count="Compare complete final semantic states exactly with uninterrupted deterministic replay.",
        mixed_version_read_count="Recompute pinned reader versions while a writer publishes another version.",
        duplicate_update_count="Count release-cursor versus unique applied-feedback discrepancies; retry checks also require unchanged version.",
        lost_acknowledged_update_count="Compare every previously returned acknowledgment result with the recovered durable ledger.",
        issued_prediction_mutation_count="Compare all issued decisions visible before death with the recovered immutable records.",
        primitive_reference="Seal raw rows independently of the terminal aggregate and recompute them from traces during cold replay.",
        work_reference="Bind authentication, resources and frozen operand references used to rebuild the result.",
        table_ready_score="Preserve the old zero table gate; the direct state has no table or bound-certificate dependency.",
        power_loss_certified="Actual process kills do not test storage-device behavior during power loss.",
        repository_health="Keep the single broader suite receipt separate from owned service qualification.",
    )
    value["reproducibility_checksum"] = canonical_hash(value)
    return value


def replay(path: Path) -> bool:
    """Recompute causal states and aggregates, including after attacker-repaired hashes."""
    progress("cold_replay_before")
    try:
        value = json.loads(path.read_bytes())
        digest = value.pop("reproducibility_checksum")
        if digest != canonical_hash(value) or value["experiment_id"] != 8376:
            return False
        refs = [
            value["work_reference"],
            *value["raw_shard_hashes"],
            *value["code_config_hashes"],
            *value["source_artifact_hashes"],
        ]
        for ref in refs:
            if sha256_file(Path(ref["path"])) != ref["sha256"]:
                return False
        work = json.loads(Path(value["work_reference"]["path"]).read_bytes())
        rebuilt = build(work, value["validation_receipts"])
        rebuilt.pop("reproducibility_checksum")
        if rebuilt != value:
            return False
        primitive = json.loads(Path(value["primitive_reference"]["path"]).read_bytes())
        if primitive["rows"] != value["rows"] or primitive["traces"] != work["traces"]:
            return False
        expected_units = sorted(
            (seed, arm, barrier)
            for seed in (11, 22, 33)
            for arm, barrier in [
                ("uninterrupted", "none"),
                *[(a, b) for b in s.BARRIERS for a in ("restart", "two_readers")],
            ]
        )
        if sorted((r["seed"], r["arm"], r["barrier"]) for r in value["rows"]) != expected_units:
            return False
        for row in value["rows"]:
            if row["status"] == "unstarted":
                if work["input_ready"] and work["resource_preconditions_passed"]:
                    return False
                continue
            for ref in [
                row["trace_reference"],
                row["final_state_reference"],
                row["kill_reference"],
            ]:
                if ref and sha256_file(Path(ref["path"])) != ref["sha256"]:
                    return False
            frozen = json.loads(Path(row["trace_reference"]["path"]).read_bytes())
            observed = json.loads(Path(row["final_state_reference"]["path"]).read_bytes())
            protocol_ref = next(
                r
                for r in work["source_artifact_hashes"]
                if r.get("source_path", "").endswith(s.PROTOCOL)
            )
            if protocol_ref["sha256"] != authority.PROTOCOL_PIN:
                return False
            head = json.loads(Path(protocol_ref["path"]).read_bytes())["head"]
            if frozen != s.trace(row["seed"], head):
                return False
            expected = s.fold(frozen)
            if any(
                expected["issued" if r["event"]["kind"] == "issue" else "applied"].get(
                    r["event"]["id"]
                )
                != r["result"]
                for r in observed["acknowledged"]
            ):
                return False
            if any(
                r["mixed"] or r["state_hash"] != canonical_hash(s.fold(frozen, r["version"]))
                for r in observed["reader_rows"]
            ):
                return False
            if row["latency_s"] != sum(
                (r["ended_monotonic_ns"] - r["started_monotonic_ns"]) / 1e9 for r in row["receipts"]
            ):
                return False
            if (
                observed["state"] != expected
                or row["state_hash"] != canonical_hash(expected)
                or any(
                    row[k]
                    for k in (
                        "recovery_mismatch_count",
                        "mixed_version_read_count",
                        "duplicate_update_count",
                        "issued_prediction_mutation_count",
                        "lost_acknowledged_update_count",
                    )
                )
            ):
                return False
            if row["kill_reference"]:
                marker = json.loads(Path(row["kill_reference"]["path"]).read_bytes())
                index = frozen["crash_event_index"]
                post = index + int(row["barrier"] in s.BARRIERS[2:])
                if (
                    marker["event_index"] != index
                    or marker["event"] != frozen["events"][index]
                    or marker["result"] != expected["applied"][marker["event"]["id"]]
                    or any(
                        expected["issued" if r["event"]["kind"] == "issue" else "applied"].get(
                            r["event"]["id"]
                        )
                        != r["result"]
                        for r in marker["acknowledged"]
                    )
                    or any(
                        r["mixed"]
                        or r["state_hash"] != canonical_hash(s.fold(frozen, r["version"]))
                        for r in marker["reader_rows"]
                    )
                    or marker["pre_state"] != s.fold(frozen, index)
                    or marker["intended_state"] != s.fold(frozen, index + 1)
                    or marker["visible_state"] != s.fold(frozen, post)
                    or row["actual_exit"] != -9
                ):
                    return False
            for receipt in row["receipts"] + value["validation_receipts"]:
                for stream in ("stdout", "stderr"):
                    if (
                        stream + "_path" in receipt
                        and sha256_file(Path(receipt[stream + "_path"]))
                        != receipt[stream + "_sha256"]
                    ):
                        return False
        progress("cold_replay_after", len(value["rows"]), 0)
        return True
    except (OSError, ValueError, KeyError, TypeError, IndexError):
        return False
