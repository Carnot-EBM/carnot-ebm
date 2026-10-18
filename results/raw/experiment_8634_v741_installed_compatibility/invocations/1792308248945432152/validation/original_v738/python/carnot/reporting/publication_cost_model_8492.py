"""REQ-REPORT-8492: original authority qualifies costs; conditional bounds are not observations.

The producer reads immutable primitive clocks and separately sealed transition
inputs. It never reruns the old snapshot/journal service benchmark.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import shutil
import sys
from tempfile import TemporaryDirectory
import time
from typing import Any

from carnot.reporting import v730_contract_methods as methods
from carnot.reporting import transaction_attribution_8436 as shared
from carnot.reporting import v728_capstone_evidence as sealed
from carnot.reporting.current_work_receipt import (
    atomic_json,
    canonical_hash,
    ZERO_INVOCATION_COUNTS,
)
from carnot.reporting.primary_publication import read_bound_sidecar, reader_receipt
from carnot.reporting.v709_execution import child
from carnot.reporting.persistence_localization_8464 import filesystem
from carnot.verify import publication_cost_model_8492 as k
from carnot.verify import persistence_localization_8464 as profile
from carnot.verify import journal_service_cost_8466 as service
from carnot.verify import journal_equivalence_8465 as stores

Json = dict[str, Any]
ROOT, NAME, TASK = (
    methods.ROOT,
    "experiment_8492_v730_publication_cost_model",
    "exp8492-publication-cost-model",
)
CLI, TEST = f"scripts/experiments/{NAME}.py", "tests/python/test_publication_cost_model_8492.py"
OWNED = [
    "python/carnot/verify/publication_cost_model_8492.py",
    "python/carnot/reporting/publication_cost_model_8492.py",
    "python/carnot/reporting/publication_cost_runner_8492.py",
    CLI,
]
MODEL_SPECS: list[Json] = []
DEPENDENCIES = [
    "python/carnot/verify/direct_atomic_state_8376.py",
    "python/carnot/verify/python_transaction_cost_8407.py",
    "python/carnot/verify/persistence_localization_8464.py",
    "python/carnot/verify/journal_equivalence_8465.py",
    "python/carnot/verify/journal_service_cost_8466.py",
    "python/carnot/reporting/current_work_receipt.py",
    "python/carnot/reporting/primary_publication.py",
    "python/carnot/reporting/v730_contract_methods.py",
    "python/carnot/reporting/v729_historical_inputs.py",
    "scripts/experiment_template.py",
    "scripts/adversarial_verify.py",
    "scripts/verdict_row_consistency_lint.py",
]
METHODS_PIN = "sha256:552831473fa8df1fd272ae690082cbf23513e559fa9648e9ab0f588f603bb667"
reference, read, checksum, sha256_file = (
    shared.reference,
    shared.read,
    shared.checksum,
    shared.sha256_file,
)


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Flush actual counts so upstream replay and synthesis remain visible without a model."""
    print(f"[exp8492] phase={phase} completed={completed} pending={pending}", flush=True)


def reconstruct(refs: list[Json]) -> Json:
    """Recompute all original clocks and transitions using independently pinned producer bytes."""
    primaries: dict[int, Json] = {}
    for n, ref in zip((8464, 8465, 8466), refs, strict=True):
        if ref["sha256"] != "sha256:" + sealed.PINS[n]:
            raise ValueError("original_producer_pin")
        primaries[n] = read(ref)
    works = {n: read(v["work_reference"]) for n, v in primaries.items()}
    stages, namespaces, rows, ceilings, denominators = [], [], [], [], []
    by_slot: dict[tuple[int, int], Json] = {}
    for ref in works[8464]["primitives"]:
        primitive = read(ref)
        reduced = profile.reduce(works[8464]["bundle"], primitive)
        if primitive["instrumented"]:
            part = k.partition(primitive)
            if (
                sum(part["stage_ns"].values()) + part["outer_residual_ns"]
                != part["complete_wall_ns"]
            ):
                raise ValueError("complete_denominator")
            slot = (primitive["cell"]["batch"], primitive["repeat"])
            by_slot[slot] = part
            row = dict(
                cell=primitive["cell"],
                repeat=primitive["repeat"],
                **part,
                source_reference=ref,
                warmup=primitive["repeat"] == 0,
            )
            stages.append(row)
            rows.append(
                dict(
                    row,
                    unit_id=f"profile-b{slot[0]}-r{slot[1]}",
                    family="profile_cost",
                    arm="instrumented",
                    status="completed",
                    absolute_metric=reduced["complete_wall_s"],
                    effective_independent_groups=0,
                )
            )
    tape_refs, feedback = [], 0
    for i, ref in enumerate(works[8466]["primitives"]):
        progress("reconstruct_service", i, 24 - i)
        primitive = read(ref)
        tape = read(primitive["frozen_reference"])
        reduced = service.reduce(tape, primitive)
        batch = tape["cell"]["batch"]
        repeat = int(tape["transactions"][0]["id"].split("-rep")[1].split("-")[0])
        ns = k.namespace(primitive["arm"], len(tape["transactions"]))
        if ns["total_fsyncs"] != primitive["fsyncs"]:
            raise ValueError("syscall_count_binding")
        namespaces.append(dict(ns, cell=tape["cell"], repeat=repeat, source_reference=ref))
        rows.append(
            dict(
                reduced,
                unit_id=f"service-b{batch}-r{repeat}-{primitive['arm']}",
                cell=tape["cell"],
                repeat=repeat,
                family="complete_service_cost",
                status="completed",
                absolute_metric=reduced["wall_s"],
                warmup=repeat == 0,
                effective_independent_groups=0,
                source_reference=ref,
            )
        )
        wall = primitive["end_ns"] - primitive["start_ns"]
        request_ns = sum(q["end_ns"] - q["start_ns"] for q in primitive["requests"])
        denominators.append(
            dict(
                cell=tape["cell"],
                repeat=repeat,
                arm=primitive["arm"],
                complete_wall_ns=wall,
                cold_ns=primitive["cold_end_ns"] - primitive["start_ns"],
                startup_ns=primitive["initialized_ns"] - primitive["cold_end_ns"],
                requests_ns=request_ns,
                after_initialization_residual_ns=primitive["end_ns"]
                - primitive["initialized_ns"]
                - request_ns,
                restart_replay_ns=primitive["restart_end_ns"] - primitive["restart_start_ns"],
            )
        )
        if primitive["arm"] == "journal":
            ceilings.append(
                dict(
                    cell=tape["cell"],
                    repeat=repeat,
                    **k.ceiling(wall, by_slot[(batch, repeat)]["removable_transaction_ns"]),
                )
            )
        if repeat == 1 and primitive["arm"] == "journal":
            tape_refs.append(primitive["frozen_reference"])
            view = stores.oracle(tape, len(tape["transactions"]))
            independent = stores.oracle(tape, 0)
            data, previous = b"", k.GENESIS
            for seq, tx in enumerate(tape["transactions"], 1):
                replies = [
                    stores.source.step(tape["bundle"], tape["cell"], independent["state"], e, q)
                    for e, q in zip(tx["events"], tx["request_ids"], strict=True)
                ]
                independent["replies"].append(replies)
                independent["ledger"].append(tx["id"])
                independent["sequence"] = seq
                framed = k.frame(tape, seq, previous)
                previous = json.loads(framed[4:])["digest"]
                data += framed
            recovered = k.recover(data, tape)
            if (
                stores.encode(view) != stores.encode(independent)
                or recovered["committed_ids"] != view["ledger"]
            ):
                raise ValueError("independent_transition_fixture")
            feedback += sum(
                e["kind"] != "prediction" for tx in tape["transactions"] for e in tx["events"]
            )
    if len(stages) != 12 or len(namespaces) != 24 or len(tape_refs) != 2:
        raise ValueError("all_original_cost_rows")
    return dict(
        rows=rows,
        stage_rows=stages,
        namespace_operation_rows=namespaces,
        conditional_speed_ceiling=ceilings,
        complete_cost_denominators=denominators,
        transition_fixture=dict(passed=True, feedback_attempts=feedback, tape_references=tape_refs),
        preserved_v728=dict(
            pooled_speedup=primaries[8466]["pooled_speedup"],
            speedup_ci95=primaries[8466]["speedup_ci95"],
            journal_ready_score=primaries[8465]["journal_ready_score"],
        ),
    )


def measure(root: Path, raw: Path) -> Json:
    """Authenticate current authority before bounded original replay; optional devices are irrelevant."""
    progress("preconditions")
    start = time.monotonic_ns()
    raw.mkdir(parents=True, exist_ok=True)
    contract = methods.authority(root, raw / "authority")
    work: Json = dict(
        root=str(root),
        raw=str(raw),
        started_monotonic_ns=start,
        gate_check_summary=contract["gate_check_summary"],
        owned_failures=[],
        upstreams=[],
        qualifications=[],
        receipts=[],
        refs=[],
        evidence={},
        historical_model_provenance=[],
        adversarial_findings=[],
        coverage_shards=[],
        code_config_hashes=[reference(ROOT / p) for p in [*OWNED, TEST, *DEPENDENCIES]],
        phase_spans=[],
        contract=contract,
    )
    work["refs"] += [r for r in contract.get("authority_snapshots", {}).values() if r.get("exists")]
    with TemporaryDirectory(dir="/var/tmp", prefix="exp8492-preflight-") as folder:
        scratch = Path(folder)
        probe = scratch / "readback"
        probe.write_bytes(b"disk-backed-private-readback")
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
            name: shutil.which(str(ROOT / ".venv/bin" / name))
            for name in ["python", "pytest", "coverage", "ruff", "mypy"]
        }
        work["preconditions_checked"] = dict(
            tools=tools,
            memory_available_bytes=memory,
            scratch_filesystem=filesystem(scratch),
            scratch_readback=probe.read_bytes() == b"disk-backed-private-readback",
            disk_free_bytes=shutil.disk_usage(scratch).free,
            task_cap_s=4800,
            measurement_cap_s=3600,
            validation_reserve_s=900,
            current_llm_calls=0,
            optional_cuda_and_boards="not_required",
        )
        if (
            not all(tools.values())
            or not filesystem(scratch)["disk_backed"]
            or memory < 256 * 1024 * 1024
        ):
            work["owned_failures"].append("preconditions")
    path = root / "results" / (methods.NAME + ".json")
    required_path, required_upstream, required_hash = path, methods.TASK, METHODS_PIN
    required_field = "methods_ready_score AND service_inputs_ready_score"
    try:
        if (
            not contract["activated"]
            or contract["observed_active_tasks_sha256"] != methods.TASKS_PIN
        ):
            raise ValueError("activated_full_v730_authority")
        task = next(t for t in contract["tasks"] if t["id"] == TASK)
        if task["deliverable"] != "results/" + NAME + ".json":
            raise ValueError("activated_task_deliverable")
        upstream = read(dict(path=str(path), sha256=METHODS_PIN))
        for field in ["methods_ready_score", "service_inputs_ready_score"]:
            consumer = reader_receipt(methods.TASK, path.parent, field=field)
            if upstream.get(field) != 1 or not consumer["passed"]:
                raise ValueError(field)
        terminal = read(reference(Path(upstream["terminal_validation_sidecar_path"])))
        side = read_bound_sidecar(path, Path(terminal["publication"]["sidecar_path"]))
        if not side["report"]["passed"]:
            raise ValueError("methods_terminal_validation")
        work["refs"].extend(
            [reference(path), reference(Path(terminal["publication"]["sidecar_path"]))]
        )
        protocol = methods.protocol()
        authorities = []
        for original in protocol["historical_authorities"]["V728"]:
            ref = sealed.freeze(Path(original["path"]), raw / "V728-authority")
            ref["source_path"] = original["source_path"]
            if ref["sha256"] != original["sha256"]:
                raise ValueError("historical_authority_hash")
            authorities.append(ref)
        work["refs"] += authorities
        for n in (8464, 8465, 8466):
            required_upstream = f"exp{n}"
            required_hash = "sha256:" + sealed.PINS[n]
            required_path = (
                root
                / "results"
                / {
                    8464: "experiment_8464_v728_persistence_localization.json",
                    8465: "experiment_8465_v728_journal_equivalence.json",
                    8466: "experiment_8466_v728_journal_service_cost.json",
                }[n]
            )
            required_field = "original_producer_replay_and_original_cost_rows"
            progress("original_producer_before", len(work["upstreams"]), 3 - len(work["upstreams"]))
            primary = sealed.freeze(required_path, raw / "primaries")
            request, output = raw / f"request-{n}.json", raw / f"qualification-{n}.json"
            atomic_json(request, dict(number=n, primary=primary, authority=authorities))
            receipt = child(
                f"original_{n}",
                [
                    sys.executable,
                    "-u",
                    str(ROOT / CLI),
                    "--history-request",
                    str(request),
                    "--history-output",
                    str(output),
                ],
                raw / "history-logs",
                deadline=240,
                scope="upstream",
            )
            work["receipts"].append(receipt)
            if not receipt["passed"]:
                raise ValueError("bounded_original_producer_replay")
            qualified = json.loads(output.read_bytes())
            work["qualifications"].append(qualified)
            if not qualified["qualified"]:
                work["gate_check_summary"] += qualified["gate_check_summary"]
                raise ValueError("original_producer_qualification")
            work["upstreams"].append(dict(path=primary["snapshot_path"], sha256=primary["sha256"]))
            work["historical_model_provenance"] += qualified["historical_model_provenance"]
            progress("original_producer_after", len(work["upstreams"]), 3 - len(work["upstreams"]))
        progress("reconstruction_before")
        work["evidence"] = reconstruct(work["upstreams"])
        work["append_protocol"] = k.protocol()
        ordering = [
            "create",
            "file_sync",
            "directory_sync",
            "append",
            "file_sync",
            "ack",
            "checkpoint",
            "file_sync",
            "directory_sync",
            "reclaim",
            "directory_sync",
        ]
        work["ordering_fixture"] = dict(events=ordering, passed=k.ordering(ordering))
        atomic_json(raw / "append_protocol.json", work["append_protocol"])
        work["append_protocol_reference"] = reference(raw / "append_protocol.json")
        progress("reconstruction_after", 36, 0)
    except (OSError, ValueError, KeyError, TypeError, StopIteration) as error:
        if len(work["upstreams"]) == 3:
            work["owned_failures"].append("owned_cost_or_protocol_reconstruction:" + str(error))
        work["gate_check_summary"].append(
            dict(
                check="required_authority_and_original_inputs",
                upstream=required_upstream,
                path=str(required_path),
                sha256=required_hash,
                field=required_field,
                operator="==",
                expected=1,
                observed=None if not required_path.exists() else str(error),
                missing=not required_path.exists(),
                passed=False,
            )
        )
    end = time.monotonic_ns()
    work["ended_monotonic_ns"] = end
    work["phase_spans"] = [
        dict(
            phase="authenticated_reconstruction",
            start_ns=start,
            end_ns=end,
            duration_s=(end - start) / 1e9,
        )
    ]
    return work


def build(work: Json, receipts: list[Json], raw: Path, output: Path) -> Json:
    """Qualification earns fixture readiness; no counterfactual can earn observed speed benefit."""
    evidence = work["evidence"]
    coverage = read(reference(raw / "coverage.json")) if (raw / "coverage.json").exists() else {}
    covered = bool(coverage) and all(
        coverage["files"].get(p, {}).get("summary", {}).get("missing_lines", 1) == 0 for p in OWNED
    )
    required_names = {
        "owned_tests",
        "coverage_combine",
        "coverage_report",
        "coverage_json",
        "ruff_check",
        "ruff_format",
        "strict_mypy",
        "spec_coverage",
    }
    execution = (
        read(reference(raw / "execution_manifest.json"))
        if (raw / "execution_manifest.json").exists()
        else {"commands": []}
    )
    required_names |= {p["name"] for p in execution["commands"]}
    passed = bool(receipts) and all(r["passed"] for r in receipts)
    required = required_names <= {r["name"] for r in receipts if r["passed"]}
    qualified = (
        bool(evidence)
        and passed
        and required
        and covered
        and not work["adversarial_findings"]
        and work.get("ordering_fixture", {}).get("passed", False)
        and len(work["qualifications"]) == 3
        and all(q["qualified"] for q in work["qualifications"])
    )
    failures = work["owned_failures"] or [r["name"] for r in receipts if not r["passed"]]
    verdict = (
        "disqualified"
        if failures or work["adversarial_findings"]
        else (
            "blocked"
            if work["gate_check_summary"]
            else ("circular_positive" if qualified else "disqualified")
        )
    )
    ready = int(verdict == "circular_positive")
    rows = evidence.get(
        "rows",
        [
            dict(
                unit_id=f"diagnostic-{i}",
                family="publication_cost",
                status="censored",
                arm="unmeasured",
                absolute_metric=None,
                missing_reason="required_authority_or_original_input_missing",
                effective_independent_groups=0,
            )
            for i in range(36)
        ],
    )
    value: Json = dict(
        experiment_id=8492,
        task_id=TASK,
        milestone="2026.10.730",
        run_date="20261013",
        honest_verdict=f"complete_{verdict}_publication_cost_model",
        verdict_class=verdict,
        gate_check_summary=work["gate_check_summary"],
        inference_substrate="aggregation_from_upstream_artifacts",
        inference_substrate_class="no_model_load",
        MODEL_SPECS=[],
        model_invocation_counts=deepcopy(ZERO_INVOCATION_COUNTS),
        historical_model_provenance=work["historical_model_provenance"],
        current_llm_invocations=0,
        rows=rows,
        intended_count=36,
        completed_count=sum(r["status"] == "completed" for r in rows),
        failed_count=sum(r["status"] == "failed" for r in rows),
        censored_count=sum(r["status"] == "censored" for r in rows),
        excluded_count=0,
        independent_count=0,
        sample_size_budget=dict(
            engineering_cells=2,
            localization_repeats_per_cell=6,
            service_arms=2,
            service_repeats_per_cell=6,
            includes_warmups=True,
            independent_generalization_questions=0,
        ),
        verifier_is_oracle=True,
        exposure_scope="exposed engineering fixtures and original cached service tape",
        independent_generalization_score=0,
        generalized_learning_benefit_score=0,
        required_checks_passed=passed and required and covered,
        flagged_adversarial=bool(work["adversarial_findings"]),
        acceptance_gates=dict(
            original_producers=bool(evidence),
            independent_transition_fixture=bool(evidence),
            ordering_fixture=work.get("ordering_fixture", {}).get("passed", False),
            full_owned_coverage=covered,
            required_validation=passed and required,
            no_unresolved_findings=not work["adversarial_findings"],
        ),
        validation_receipts=receipts,
        historical_validation_receipts=work["receipts"],
        authority_snapshots=work["contract"].get("authority_snapshots", {}),
        terminal_validation_sidecar_path=str(raw / "terminal_validation.json"),
        adversarial_findings=work["adversarial_findings"],
        preconditions_checked=work["preconditions_checked"],
        duration_s=(work["ended_monotonic_ns"] - work["started_monotonic_ns"]) / 1e9,
        phase_spans=work["phase_spans"],
        random_seed=7308492,
        source_artifact_hashes=work["refs"] + work["upstreams"],
        code_config_hashes=work["code_config_hashes"],
        raw_shard_hashes=work["coverage_shards"] + [reference(raw / "measurement.json")],
        work_reference=reference(raw / "measurement.json"),
        owned_coverage=coverage.get("totals", {}),
        cited_upstream_artifacts=[
            dict(
                experiment_id=n,
                sha256=r["sha256"],
                path=r["path"],
                imported_fields="original authority, primitive clocks, fsync counts, independent tape and cached model provenance",
            )
            for n, r in zip((8464, 8465, 8466), work["upstreams"])
        ],
        append_protocol_ready_score=ready,
        append_protocol_reference=work.get("append_protocol_reference"),
        append_protocol=work.get("append_protocol"),
        stage_rows=evidence.get("stage_rows", []),
        namespace_operation_rows=evidence.get("namespace_operation_rows", []),
        conditional_speed_ceiling=evidence.get("conditional_speed_ceiling", []),
        complete_cost_denominators=evidence.get("complete_cost_denominators", []),
        transition_fixture=evidence.get("transition_fixture"),
        ordering_fixture=work.get("ordering_fixture"),
        fsync_obligations=dict(
            file="precreation and every complete append before acknowledgment",
            directory="all create/rename/unlink/checkpoint/rotation namespace changes",
            checkpoint="every32 committed transactions and final drain; prefix durable before reclamation",
            observed_journal="one append file sync plus certificate file and directory sync per transaction",
        ),
        preserved_v728=evidence.get("preserved_v728"),
        closed_exposed_utility=dict(H1=-0.00390625, H2=0),
        scientific_speed_benefit_score=0,
        measured_append_speedup=None,
        future_dependencies=[
            dict(experiment_id=n, status="future_dependency_not_observation") for n in (8493, 8494)
        ],
        methodology_note="Original sealed V728 primitive replay and disjoint complete-clock reconstruction; no new benchmark, no model. Conditional transferred pointer-cost bound requires unchanged retained work and zero new overhead. Reference recovery fixtures are not a durable append store or physical power-loss qualification.",
    )
    value["field_principles"] = {
        key: "Bind exact invocation operands; exposed engineering qualification does not establish generalization or measured append speedup."
        for key in value
    }
    value["field_principles"].update(
        conditional_speed_ceiling="Snapshot pointer marginal costs transferred to journal complete clocks only under an explicit stationarity assumption; never measured speedup.",
        complete_cost_denominators="Charge cold reconstruction, startup, request intervals and every residual; restart is separately reported outside original complete clock.",
        append_protocol_ready_score="All original rows and sealed authority, independent transition/ordering fixtures, full coverage and unchanged validators must qualify.",
    )
    value["reproducibility_checksum"] = checksum(value)
    return value


def replay(path: Path) -> bool:
    """Fresh reconstruction rejects repaired summary hashes and separately pinned input drift."""
    try:
        value = json.loads(path.read_bytes())
        if value["reproducibility_checksum"] != checksum(value):
            return False
        work = read(value["work_reference"])
        for ref in work["code_config_hashes"] + work["refs"]:
            if sha256_file(Path(ref.get("snapshot_path", ref["path"]))) != ref["sha256"]:
                return False
        if work["evidence"]:
            if methods.tasks_digest(work["contract"]["tasks"]) != methods.TASKS_PIN:
                return False
            if reconstruct(work["upstreams"]) != work["evidence"]:
                return False
            if (
                work["append_protocol"] != k.protocol()
                or read(work["append_protocol_reference"]) != k.protocol()
            ):
                return False
            k.ordering(work["ordering_fixture"]["events"])
        return bool(value == build(work, value["validation_receipts"], Path(work["raw"]), path))
    except (OSError, ValueError, KeyError, TypeError, IndexError):
        return False
