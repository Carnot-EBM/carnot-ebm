"""REQ-VERIFY-8410: cached reading and changed device execution are separate facts."""

from __future__ import annotations

import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import time
from typing import Any

from carnot.gpu_lease_phase_journal import GpuLease, LeaseError
from carnot.reporting import historical_consumer_roots_8402 as h
from carnot.reporting.current_work_receipt import (
    ZERO_INVOCATION_COUNTS,
    atomic_json,
    canonical_hash,
    sha256_file,
)
from carnot.reporting.evidence_features_custody_7980 import reference
from carnot.reporting.v709_execution import child
from carnot.reporting.v710_contract_replay import snapshot
from carnot.verify import cuda_failure_cause_8396 as old
from carnot.verify import cuda_primitive_8290 as primitive
from carnot.verify import runtime_evidence_execution_8382 as base
from carnot.verify import typed_runtime_8368 as typed

Json = dict[str, Any]
ROOT = h.ROOT
NAME, TASK, MILESTONE = (
    "experiment_8410_v724_cuda_reader_qualification",
    "exp8410-cuda-reader-qualification",
    "2026.10.724",
)
CLI, TEST = f"scripts/experiments/{NAME}.py", "tests/python/test_cuda_reader_8410.py"
OWNED = [
    "python/carnot/verify/cuda_reader_8410.py",
    "python/carnot/verify/cuda_reader_runner_8410.py",
    CLI,
]
MODEL_SPECS: list[Json] = []
UPSTREAM = "results/experiment_8402_v724_historical_consumer_roots.json"
CACHED = "results/experiment_8396_v723_cuda_failure_cause.json"
UPSTREAM_PIN = "sha256:94d4baf2872bcf2bc58a7dd7a96f431c33ccad339c272927f1fc9673306b944d"
CACHED_PIN = h.PRIMARY_PINS[8396]
TASK_PIN = "sha256:a349f70280aef18adb041657e36a3df3dcb7b4eee93bb848cf6210a83ee4c34b"
checksum = old.checksum


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Flush work boundaries so owned children remain visible to the supervisor."""
    print(f"[exp8410] phase={phase} completed={completed} pending={pending}", flush=True)


def cached(primary: Path, raw: Path) -> Json:
    """Reduce original trace bytes without promoting the failed historical qualification."""
    source = snapshot(h.checked(primary, CACHED_PIN), raw, "primary")
    value = json.loads(typed.authenticate(source).read_bytes())
    work_ref = snapshot(typed.authenticate(value["primitive_reference"]), raw, "measurement")
    work = json.loads(typed.authenticate(work_ref).read_bytes())
    diagnostic = work["diagnostic"]
    receipt = diagnostic["rows"][0]["receipt"]
    stdout = snapshot(
        h.checked(Path(receipt["stdout_path"]), receipt["stdout_sha256"]), raw, "stdout"
    )
    observed = json.loads(typed.authenticate(stdout).read_text().splitlines()[-1])
    trace = snapshot(typed.authenticate(diagnostic["trace_manifest"]["ref"]), raw, "trace")
    text = typed.authenticate(trace).read_text()
    reduced = old.analyze(text, observed)
    returns = [
        s
        for s in text.splitlines()
        if "/dev/nvidia" in s and any(t in s for t in ("open(", "openat(", "ioctl("))
    ]
    if (
        observed != diagnostic["rows"][0]["primitive"]
        or returns != diagnostic["trace_manifest"]["device_returns"]
        or observed["api_returns"] != {"cuInit": 999}
        or reduced["correction"] is not None
        or diagnostic["intervention_rows"]
    ):
        raise ValueError("cached_primitive_drift")
    previous = dict(
        kernel_driver=observed["kernel_driver"],
        libraries=observed["libraries"],
        device_nodes=observed["device_nodes"],
        environment=observed["environment"],
        boot_identity=work["baseline"].get("boot_identity"),
        namespaces=work["baseline"].get("namespaces"),
        permitted_uuid=work["baseline"]["permitted_uuid"],
        driver_library=work["baseline"]["driver_library"],
    )
    return dict(
        previous=previous,
        diagnosis=dict(reduced, api_returns=observed["api_returns"]),
        trace_manifest=dict(diagnostic["trace_manifest"], ref=trace, device_returns=returns),
        historical_model_provenance=work["historical"].get("historical_model_provenance", {}),
        refs=[source, work_ref, stdout, trace],
    )


def inventory(previous: Json) -> Json:
    """Inspect files and access metadata; no CUDA library or device is opened."""
    driver = Path("/proc/driver/nvidia/version")
    nodes = []
    for path in sorted(Path("/dev").glob("nvidia*")):
        stat = path.stat()
        nodes.append(
            dict(
                path=str(path),
                mode=oct(stat.st_mode),
                uid=stat.st_uid,
                gid=stat.st_gid,
                read_write=os.access(path, os.R_OK | os.W_OK),
            )
        )
    return dict(
        kernel_driver=driver.read_text() if driver.is_file() else None,
        libraries=[
            dict(
                path=row["path"],
                sha256=sha256_file(Path(row["path"])) if Path(row["path"]).is_file() else None,
            )
            for row in previous["libraries"]
        ],
        device_nodes=nodes,
        environment={k: os.environ[k] for k in primitive.VARIABLES if k in os.environ},
        boot_identity=Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
        namespaces={p.name: os.readlink(p) for p in sorted(Path("/proc/self/ns").iterdir())},
        permitted_uuid=previous["permitted_uuid"],
        driver_library=previous["driver_library"],
    )


def compare(previous: Json, current: Json) -> list[Json]:
    """Only comparable execution operands can authorize a new probe; missing is unknown."""
    rows = []
    for field in (
        "kernel_driver",
        "libraries",
        "boot_identity",
        "namespaces",
        "device_nodes",
        "environment",
    ):
        before, after = previous.get(field), current.get(field)
        available = before is not None and after is not None
        if field == "libraries":
            available = available and all(r.get("sha256") for r in after or [])
        before_operand, after_operand = before, after
        if field == "environment" and available:
            # The child loads an absolute driver path; toolkit discovery hints
            # cannot change that execution. Keep the full observations below.
            before_operand = {
                k: v for k, v in before.items() if k not in {"CUDA_HOME", "CUDA_PATH"}
            }
            after_operand = {k: v for k, v in after.items() if k not in {"CUDA_HOME", "CUDA_PATH"}}
        rows.append(
            dict(
                field=field,
                previous=before,
                current=after,
                comparable=bool(available),
                changed=bool(available and before_operand != after_operand),
                missing_reason=None if available else "historical_or_current_operand_absent",
            )
        )
    return rows


def repair(path: Path, digest: str | None, current: Json, wall: int, frontier: int) -> Json:
    """An operator receipt needs independent hashes and a current execution operand."""
    document = json.loads(h.checked(path, str(digest)).read_bytes())
    if not (
        document["authority"] == "operator"
        and document["previous_primary_sha256"] == CACHED_PIN
        and frontier < document["executed_at_ns"] <= wall
        and document["action"] in {"driver_repair", "device_repair", "lease_rebind", "reboot"}
        and document["description"]
    ):
        raise ValueError("operator_repair_authority")
    supplied = json.loads(typed.authenticate(document["current_reference"]).read_bytes())
    if supplied != current:
        raise ValueError("operator_repair_inventory_drift")
    return document


def probe(binding: Json, raw: Path) -> Json:
    """An exclusive lease owns one bounded child and only that child's cleanup."""
    result: Json = dict(
        rows=[], checks=[], lease_receipt={}, cleanup_receipts=[], probe_count=0, ready=False
    )
    lease: Any = None
    try:
        lease = GpuLease.acquire(
            runtime_dir="/tmp/carnot-gpu-leases",
            task_id=TASK,
            device_uuid=binding["permitted_uuid"],
            expected_model="no_model_load:cuda_byte_copy",
            vram_before_mb=0,
            ttl_s=180,
        )
        result["lease_receipt"] = lease.owner_receipt()
        lease.transition("admitted")
        result["probe_count"] = 1
        receipt = child(
            "context_copy",
            [
                str(ROOT / ".venv/bin/python"),
                "-u",
                str(ROOT / CLI),
                "--cuda-probe",
                "driver",
                "--uuid",
                binding["permitted_uuid"],
                "--library",
                binding["driver_library"],
            ],
            raw,
            deadline=120,
            heartbeat=20,
            scope="external_cuda",
        )
        result["rows"].append(dict(receipt=receipt, primitive={}))
        lines = Path(receipt["stdout_path"]).read_text().splitlines()
        observed = json.loads(lines[-1]) if receipt["passed"] and lines else {}
        result["rows"][0]["primitive"] = observed
        result["ready"] = bool(
            receipt["passed"]
            and observed.get("allocation_bytes") == 32
            and all(
                observed.get(k) is True
                for k in ("context_copy_ready", "byte_copy_parity", "cleanup_passed")
            )
        )
        if not result["ready"]:
            result["checks"].append(h.failure(raw, "current_context_copy", True, observed))
    except (OSError, ValueError, KeyError, LeaseError) as error:
        result["checks"].append(h.failure(raw, "exclusive_owned_context_copy", True, str(error)))
    finally:
        if lease:
            # Driver-only work ends admission without entering the lease's model phases.
            lease.transition("terminal_blocked")
            result["cleanup_receipts"].append(lease.release())
    return result


def measure(
    root: Path,
    raw: Path,
    receipt: Path | None = None,
    receipt_hash: str | None = None,
    *,
    fixture: bool = False,
) -> Json:
    """Authenticate current authority and historical operands before admitting changed work."""
    progress("preconditions_and_authentication_before")
    raw.mkdir(parents=True, exist_ok=True, mode=0o700)
    observed = base.resources(raw)
    authority = h.authority(root, raw / "authority")
    failures = list(authority["gate_check_summary"])
    tasks = [t for t in authority["tasks"] if t["id"] == TASK]
    if len(tasks) != 1 or canonical_hash(tasks[0]) != TASK_PIN:
        failures.append(
            h.failure(
                root / h.ACTIVE,
                "exact_task_hash",
                TASK_PIN,
                canonical_hash(tasks[0]) if tasks else None,
            )
        )
    if (
        not observed["private_disk_backed_scratch"]
        or observed["available_memory_bytes"] < observed["minimum_memory_bytes"]
    ):
        failures.append(h.failure(raw, "private_disk_memory_budget", True, observed))
    work: Json = dict(
        root=str(root),
        authority=authority,
        failures=failures,
        resources=observed,
        refs=[],
        family={},
        cached={},
        current={},
        repair_receipt={},
        fixture=fixture,
        observed_wall_ns=time.time_ns(),
        probe=dict(
            rows=[], checks=[], lease_receipt={}, cleanup_receipts=[], probe_count=0, ready=False
        ),
    )
    for name, pin in dict(
        old.PINS, **{h.PROTOCOL: h.PROTOCOL_PIN, UPSTREAM: UPSTREAM_PIN, CACHED: CACHED_PIN}
    ).items():
        ref = snapshot(root / name, raw / "inputs", Path(name).stem)
        work["refs"].append(ref)
        if ref["sha256"] != pin:
            failures.append(h.failure(root / name, "immutable_input_hash", pin, ref["sha256"]))
    try:
        source = h.checked(root / UPSTREAM, UPSTREAM_PIN)
        upstream = json.loads(source.read_bytes())
        if upstream["historical_runtime_consumers_ready_score"] != 1:
            raise ValueError("historical_runtime_consumers_ready_score")
        work["family"] = h.authenticate_family(source, "runtime")
        work["cached"] = cached(root / CACHED, raw / "cached")
        work["refs"].extend(work["cached"]["refs"])
        progress("bounded_inventory_before")
        work["current"] = inventory(work["cached"]["previous"])
        path = raw / "current_inventory.json"
        atomic_json(path, work["current"])
        work["inventory_reference"] = reference(path)
        work["refs"].append(work["inventory_reference"])
        progress("bounded_inventory_after", 1, 0)
    except (OSError, ValueError, KeyError, TypeError) as error:
        failures.append(
            h.failure(root / UPSTREAM, "authenticated_runtime_family_and_cache", True, str(error))
        )
    if receipt is not None:
        try:
            work["repair_receipt"] = repair(
                receipt,
                receipt_hash,
                work["current"],
                work["observed_wall_ns"],
                json.loads((root / CACHED).read_bytes())["phase_spans"][0]["started_wall_ns"],
            )
            work["refs"].extend(
                [
                    snapshot(receipt, raw / "repair", "receipt"),
                    snapshot(
                        typed.authenticate(work["repair_receipt"]["current_reference"]),
                        raw / "repair",
                        "inventory",
                    ),
                ]
            )
        except (OSError, ValueError, KeyError, TypeError) as error:
            failures.append(h.failure(receipt, "explicit_operator_repair", True, str(error)))
    delta = compare(work["cached"].get("previous", {}), work["current"])
    if not failures and not fixture and any(row["changed"] for row in delta):
        progress("changed_context_copy_before")
        work["probe"] = probe(work["current"], raw / "context_copy")
        progress("changed_context_copy_after", work["probe"]["probe_count"], 0)
    progress("preconditions_and_authentication_after", 1, 0)
    atomic_json(raw / "measurement.json", work)
    return work


def reduction(work: Json, passed: bool) -> Json:
    """Reader correctness does not imply an environment delta, successful work or benefit."""
    reader = bool(
        passed
        and not work["failures"]
        and work["family"].get("ready")
        and work["cached"]
        and work["authority"].get("activated")
    )
    delta = compare(work["cached"].get("previous", {}), work["current"])
    changed = bool(reader and any(row["changed"] for row in delta))
    context = bool(
        changed
        and work["probe"]["ready"]
        and not work["probe"]["checks"]
        and work["probe"]["lease_receipt"]
        and work["probe"]["cleanup_receipts"]
        and all(r.get("released") is True for r in work["probe"]["cleanup_receipts"])
    )
    gates = list(work["failures"]) + work["probe"]["checks"]
    for field, value in [
        ("runtime_reader_ready_score", reader),
        ("runtime_changed_score", changed),
        ("cuda_context_ready_score", context),
    ]:
        if not value:
            gates.append(h.failure(Path(work["root"]) / CLI, field, 1, int(value)))
    if not changed:
        gates.append(
            h.failure(
                Path(work["root"]) / "ops/cuda-runtime-repair-receipt.json",
                "explicit_supplied_repair_receipt",
                "hash_bound_execution_delta",
                work["repair_receipt"] or None,
            )
        )
    rows = []
    for arm, completed, failed in [
        ("cached_reader", bool(work["cached"] and work["family"]), False),
        ("environment_inventory", bool(work["current"]), False),
        (
            "context_copy",
            bool(work["probe"]["probe_count"]),
            bool(work["probe"]["probe_count"] and not context),
        ),
    ]:
        rows.append(
            dict(
                unit_id=arm,
                arm=arm,
                status="failed" if failed else "completed" if completed else "censored",
                completed=completed and not failed,
                failed=failed,
                censored=not completed,
                excluded=False,
                absolute_metric=int(completed and not failed),
                missing_reason=None
                if completed
                else "authenticated_execution_delta_or_dependency_absent",
            )
        )
    verdict = "disqualified" if not passed else "null" if context else "blocked"
    diagnosis = work["cached"].get("diagnosis", old.analyze("", {}))
    return dict(
        honest_verdict="complete_" + verdict + "_cuda_reader_qualification",
        verdict_class=verdict,
        gate_check_summary=gates,
        rows=rows,
        intended_count=3,
        completed_count=sum(r["completed"] for r in rows),
        failed_count=sum(r["failed"] for r in rows),
        censored_count=sum(r["censored"] for r in rows),
        excluded_count=0,
        independent_count=0,
        runtime_reader_ready_score=int(reader),
        runtime_changed_score=int(changed),
        cuda_context_ready_score=int(context),
        failure_layer=diagnosis["failure_layer"],
        root_cause_status=diagnosis["root_cause_status"],
        environment_delta_rows=delta,
        current_device_probe_count=work["probe"]["probe_count"],
        acceptance_gates=dict(
            cached_reader=reader, actual_execution_delta=changed, owned_context_copy_cleanup=context
        ),
    )


def build(work: Json, raw: Path, receipts: list[Json]) -> Json:
    """Bind terminal claims to exact operands while keeping historical model activity separate."""
    passed = bool(receipts) and all(r["passed"] for r in receipts if r.get("scope") != "global")
    value = reduction(work, passed)
    value.update(
        experiment_id=8410,
        task_id=TASK,
        milestone=MILESTONE,
        run_date="20261011",
        inference_substrate="aggregation_from_upstream_artifacts",
        inference_substrate_class="no_model_load",
        MODEL_SPECS=[],
        no_model_load=True,
        model_invocation_counts=dict(ZERO_INVOCATION_COUNTS),
        historical_model_provenance=work["cached"].get("historical_model_provenance", {}),
        verifier_is_oracle=False,
        exposure_scope="exposed development runtime qualification; no scientific samples",
        independent_generalization_score=0,
        generalized_learning_benefit_score=0,
        required_checks_passed=passed,
        flagged_adversarial=not passed,
        validation_receipts=receipts,
        sample_size_budget=dict(intended_arms=3, independent_scientific_samples=0),
        terminal_validation_sidecar_path=str(raw / "terminal_validation.json"),
        preconditions_checked=True,
        resource_preconditions=work["resources"],
        duration_s=work.get("duration_s", 0.001),
        phase_spans=work.get("phase_spans", []),
        random_seed=7248410,
        source_artifact_hashes=work["refs"],
        primitive_reference=reference(raw / "measurement.json"),
        raw_shard_hashes=[
            reference(raw / "measurement.json"),
            *work.get("coverage_shard_references", []),
        ],
        code_config_hashes=[
            snapshot(ROOT / p, raw / "code", str(i))
            for i, p in enumerate(
                OWNED
                + [
                    TEST,
                    "python/carnot/reporting/primary_publication.py",
                    "scripts/adversarial_verify.py",
                    "scripts/verdict_row_consistency_lint.py",
                ]
            )
        ],
        trace_manifest=work["cached"].get("trace_manifest", {}),
        typed_runtime_binding=dict(
            previous=work["cached"].get("previous", {}),
            current=work["current"],
            current_reference=work.get("inventory_reference"),
            comparison_policy="comparable execution operands only; missing historical boot/namespace stays unknown",
        ),
        context_copy_receipt=work["probe"]["rows"],
        lease_receipt=work["probe"]["lease_receipt"],
        cleanup_receipts=work["probe"]["cleanup_receipts"],
        operator_remedy="Supply authenticated changed driver/library, device namespace or reboot evidence before one owned context/copy probe. Readiness after reboot cannot prove reboot caused recovery.",
        adversarial_findings=work.get("finding_audits", []),
        cited_upstream_artifacts=[
            dict(
                r,
                fields_imported=[
                    "full authority, sealed runtime family or cached CUDA primitive operands"
                ],
            )
            for r in work["refs"]
        ],
        execution_authority=work["authority"],
        historical_runtime_family=work["family"],
        owned_statement_coverage=work.get("owned_statement_coverage", {}),
        repository_health=dict(
            status="not_rerun_unknown", scope="owned checks only; global suite belongs to conductor"
        ),
        closed_scientific_results=dict(
            H1=-0.00390625, H2=0, status="closed_exposed_development_nulls"
        ),
        methodology_note="Authenticate Exp8402 administrative and independent runtime family seals; cold-reduce cached Exp8396 trace without repeating initialization. Read bounded metadata and admit one leased context/copy child only after actual execution change. Qualification grants no semantic benefit.",
    )
    value["field_principles"] = {
        k: "Bind sealed operands and independently qualified reader, execution change and context facts; missing evidence cannot become benefit."
        for k in value
    }
    value["field_principles"].update(
        field_principles="Explain each published field.",
        reproducibility_checksum="Bind all fields except this checksum.",
    )
    for names, principle in {
        "experiment_id task_id milestone run_date execution_authority": "Bind the actual date and exact full current task objects to independently pinned V724 authority.",
        "honest_verdict verdict_class gate_check_summary operator_remedy": "External absence is terminal blocked; owned check failure disqualifies. Name every missing operand separately from zero.",
        "inference_substrate inference_substrate_class MODEL_SPECS model_invocation_counts historical_model_provenance no_model_load": "Current LLM loads and generation calls are zero. Cached model identity belongs only to historical provenance.",
        "rows intended_count completed_count failed_count censored_count excluded_count independent_count sample_size_budget": "Retain reader, inventory and copy units including failure or missing reasons; these are zero scientific samples.",
        "runtime_reader_ready_score runtime_changed_score cuda_context_ready_score typed_runtime_binding environment_delta_rows current_device_probe_count": "Sealed reader qualification, comparable execution change and successful leased context/copy are independent facts. Missing old boot or namespace cannot authorize a probe.",
        "failure_layer root_cause_status trace_manifest context_copy_receipt lease_receipt cleanup_receipts historical_runtime_family": "Preserve cached cuInit=999, driver initialization and unproved cause; only exact owned child and cleanup evidence establish current readiness.",
        "required_checks_passed flagged_adversarial acceptance_gates validation_receipts terminal_validation_sidecar_path adversarial_findings owned_statement_coverage repository_health": "Exact bounded receipts and genuine child shards qualify owned code. Retain all validator findings and keep global repository health unknown.",
        "verifier_is_oracle exposure_scope independent_generalization_score generalized_learning_benefit_score closed_scientific_results": "Exposed engineering qualification grants no scientific benefit and cannot reopen the closed H1/H2 nulls.",
        "preconditions_checked resource_preconditions duration_s phase_spans random_seed reproducibility_checksum source_artifact_hashes code_config_hashes raw_shard_hashes primitive_reference cited_upstream_artifacts": "Bind resources, exact byte hashes, monotonic spans and imported operands to independent cold reduction.",
    }.items():
        value["field_principles"].update(dict.fromkeys(names.split(), principle))
    value["reproducibility_checksum"] = checksum(value)
    return value


def replay(path: Path) -> bool:
    """Cold reduction rejects changed summaries and sealed primitives even after their rehash."""
    try:
        value = json.loads(path.read_bytes())
        if value["reproducibility_checksum"] != checksum(value) or [
            value[k] for k in ("experiment_id", "task_id", "milestone", "run_date")
        ] != [8410, TASK, MILESTONE, "20261011"]:
            return False
        for ref in value["code_config_hashes"] + value["raw_shard_hashes"]:
            typed.authenticate(ref)
        work = json.loads(typed.authenticate(value["primitive_reference"]).read_bytes())
        for ref in work["refs"]:
            if ref.get("exists", True):
                typed.authenticate(ref)
        with TemporaryDirectory(prefix="exp8410-authority-", dir="/var/tmp") as directory:
            frozen = Path(directory)
            for key, name in [("design", h.DESIGN), ("active", h.ACTIVE), ("staged", h.STAGED)]:
                ref = work["authority"].get("authority_snapshots", {}).get(key, {})
                if ref.get("exists"):
                    destination = frozen / name
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    destination.write_bytes(
                        typed.authenticate(dict(ref, path=ref["source_path"])).read_bytes()
                    )
            rebuilt = h.authority(frozen, frozen / "assessment")
            if any(
                rebuilt.get(k) != work["authority"].get(k)
                for k in ("activated", "tasks", "canonical_tasks_sha256", "contract_rows")
            ):
                return False
        if work["authority"]["tasks"] and (
            h.tasks_digest(work["authority"]["tasks"]) != h.TASKS_PIN
            or canonical_hash(next(t for t in work["authority"]["tasks"] if t["id"] == TASK))
            != TASK_PIN
        ):
            return False
        for name, pin in dict(
            old.PINS, **{h.PROTOCOL: h.PROTOCOL_PIN, UPSTREAM: UPSTREAM_PIN, CACHED: CACHED_PIN}
        ).items():
            ref = next(r for r in work["refs"] if r["path"] == str(Path(work["root"]) / name))
            if ref.get("exists") and ref["sha256"] != pin:
                return False
        if work["cached"]:
            ref = next(r for r in work["refs"] if r["path"] == str(Path(work["root"]) / CACHED))
            recomputed = cached(
                typed.authenticate(ref), Path(work["cached"]["refs"][0]["snapshot_path"]).parent
            )
            if any(
                recomputed[k] != work["cached"][k]
                for k in ("previous", "diagnosis", "historical_model_provenance")
            ):
                return False
            upstream = next(
                r for r in work["refs"] if r["path"] == str(Path(work["root"]) / UPSTREAM)
            )
            if (
                h.authenticate_family(h.checked(Path(upstream["path"]), UPSTREAM_PIN), "runtime")
                != work["family"]
            ):
                return False
            if (
                json.loads(typed.authenticate(work["inventory_reference"]).read_bytes())
                != work["current"]
            ):
                return False
        for receipt in value["validation_receipts"] + [
            row["receipt"] for row in work["probe"]["rows"]
        ]:
            for stream in ("stdout", "stderr"):
                h.checked(Path(receipt[stream + "_path"]), receipt[stream + "_sha256"])
        for row in work["probe"]["rows"]:
            receipt = row["receipt"]
            lines = Path(receipt["stdout_path"]).read_text().splitlines()
            observed = json.loads(lines[-1]) if receipt["passed"] and lines else {}
            if observed != row["primitive"]:
                return False
            ready = bool(
                receipt["passed"]
                and observed.get("allocation_bytes") == 32
                and all(
                    observed.get(k) is True
                    for k in ("context_copy_ready", "byte_copy_parity", "cleanup_passed")
                )
            )
            if ready != work["probe"]["ready"]:
                return False
        passed = bool(value["validation_receipts"]) and all(
            r["passed"] for r in value["validation_receipts"] if r.get("scope") != "global"
        )
        expected = build(
            work, Path(value["primitive_reference"]["path"]).parent, value["validation_receipts"]
        )
        return (
            set(value) == set(expected)
            and all(
                value[k] == expected[k]
                for k in expected
                if k not in {"code_config_hashes", "reproducibility_checksum"}
            )
            and all(value[k] == v for k, v in reduction(work, passed).items())
        )
    except (OSError, ValueError, KeyError, TypeError, StopIteration, IndexError):
        return False
