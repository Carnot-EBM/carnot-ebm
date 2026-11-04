"""REQ-REPORT-8783: exact identity must survive measurement and consumption."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import patch

from carnot.reporting import ordinary_durable_cost_8772 as historical
from carnot.reporting import lifecycle_archive_authority_8771 as archive
from carnot.reporting import v752_branch_methods as contract
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash
from carnot.reporting.roadmap_contract import parse_design
from carnot.reporting.v748_collection_contract import namespace

Json = dict[str, Any]
ROOT, ARMS = historical.ROOT, historical.ARMS
NAME, TASK = "experiment_8783_v752_durable_producer_identity", "exp8783-durable-producer-identity"
CLI = f"scripts/experiments/{NAME}.py"
OWNED = [
    "python/carnot/reporting/durable_producer_identity_8783.py",
    "python/carnot/reporting/durable_producer_runner_8783.py",
    CLI,
]
TESTS = ["tests/python/test_durable_producer_identity_8783.py"]
MODEL_SPECS: list[Json] = []
SEED = 7528783
reference, verify, read = historical.reference, historical.verify, historical.read
measurement = namespace(historical, "identity8783_measurement")
METHOD = dict(
    historical.METHOD,
    repeats=1,
    process_cells=3,
    measured_updates=96,
    child_launches=6,
    qualification_only=True,
    bootstrap_seed=SEED,
)
PRESERVED = ROOT / "openspec/change-proposals/research-roadmap-v751-preserved-20261103.md"


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Report real phase boundaries so a slow child remains visible."""
    if phase in {"sources_plan_frozen", "window_before", "window_after"}:
        pending = max(0, len(roster()) - completed)
    print(f"[exp8783] phase={phase} completed={completed} pending={pending}", flush=True)


def descriptor(invocation_id: str) -> Json:
    """Use the same exact contract fields at every trust boundary."""
    return dict(
        experiment_id=8783,
        task_id=TASK,
        milestone="2026.11.752",
        invocation_id=invocation_id,
        deliverable=f"results/{NAME}.json",
        inference_substrate_class="no_model_load",
        MODEL_SPECS=[],
        canonical_tasks_sha256=contract.PIN,
    )


def exact_identity(value: Json, producer: Json) -> None:
    """A matching number or a fresh hash cannot authenticate a different task."""
    if producer != descriptor(producer["invocation_id"]):
        raise ValueError("producer_descriptor")
    if any(
        value.get(k) != producer[k]
        for k in ["experiment_id", "task_id", "milestone", "invocation_id"]
    ):
        raise ValueError("producer_identity")


def roster() -> list[Json]:
    """One matched restart window qualifies identity without claiming a cost interval."""
    return [
        dict(unit_id=f"0:2032:{arm}", repeat=0, offset=2032, arm=arm, order=i, updates=32)
        for i, arm in enumerate(ARMS)
    ]


def archived_reference(ref: Json) -> Json:
    """Only identical preserved roadmap bytes may replace the moved historical path."""
    if (
        Path(ref["path"]) == ROOT / "openspec/change-proposals/research-roadmap-vNEXT.md"
        and ref["sha256"]
        == "sha256:da649a19cccc6296825a7b650bcab3dcbf5dc9f04e5e9b8e7d0f817bf8036c58"
    ):
        retained = dict(ref, path=str(PRESERVED))
        verify(retained)
        return retained
    return ref


def inputs() -> Json:
    """Retain all old controls while checking original source and loaded binary bytes."""
    with patch.object(measurement, "verify", lambda ref: verify(archived_reference(ref))):
        value: Json = measurement.inputs()
    if value["failures"]:
        return value
    try:
        value["refs"] = [archived_reference(ref) for ref in value["refs"]]
        text = (ROOT / contract.DESIGN).read_text()
        tasks = parse_design(text, milestone="2026.11.752")[1]
        task = next(t for t in tasks if t["id"] == TASK)
        if canonical_hash(tasks) != "sha256:" + contract.PIN or any(
            task[k] != descriptor("contract")[k]
            for k in ["milestone", "deliverable", "inference_substrate_class", "MODEL_SPECS"]
        ):
            raise ValueError("exact_current_task_contract")
        work = read(value["qualified"]["primitive_reference"])
        manifest = work["historical_fixture_manifest"]
        verify(manifest)
        authority = archive.verify_fixture(Path(manifest["path"]).parent, manifest["sha256"])
        controls = value["qualified"]
        if not all(
            r["passed"]
            for k in ["rows", "restart_rows", "threshold_control_rows"]
            for r in controls[k]
        ):
            raise ValueError("retained_lifecycle_controls")
        value.update(
            original_authority_rows=[dict(authority, manifest_reference=manifest)],
            retained_controls={k: controls[k] for k in ["restart_rows", "threshold_control_rows"]},
        )
        value["refs"] += [
            reference(ROOT / contract.DESIGN),
            manifest,
            *read(manifest)["references"],
        ]
    except (OSError, ValueError, KeyError, TypeError, StopIteration) as error:
        value["failures"].append(
            historical.failure(
                reference(ROOT / contract.DESIGN),
                "exact_contract_and_retained_controls",
                "authenticated",
                str(error),
                "rejected_producer",
            )
        )
    return value


def request(admitted: Json, offset: int, arm: str, updates: int = 32) -> Json:
    """Worker requests carry the descriptor sealed by their measurement parent."""
    return dict(
        historical.request(admitted, offset, arm, updates),
        producer_descriptor=admitted["producer_descriptor"],
    )


def projection(latencies: list[int], complete_ns: int) -> Json:
    """Scale the unchanged conservative admission probe to this qualification roster."""
    value: Json = historical.projection(latencies, complete_ns)
    value["projected_with_reserve_s"] = (
        (96 * value["maximum_latency_ns"] + 6 * value["measured_process_overhead_ns"]) / 1e9 * 1.25
    )
    value.update(
        durable_updates=96, process_launches=6, admitted=value["projected_with_reserve_s"] <= 3000
    )
    return value


def parity(work: Json) -> tuple[list[Json], list[Json]]:
    """Compare each acknowledged state and both restart segments to sealed authority."""
    expected = {r["update_index"]: r for r in work["expected_trace"]}
    reference_windows = work["window_rows"][0]["segments"]
    rows, restarts = [], []
    for window in work["window_rows"]:
        observed = window["update_rows"]
        for update in observed:
            old = expected.get(update["update_index"], {})
            passed = (
                update["passed"]
                and update.get("state_hash") == old.get("state_hash")
                and update.get("probability_after") == old.get("probability")
                and update.get("renormalization_writes") == old.get("dense_writes")
            )
            rows.append(
                dict(
                    unit_id=window["unit_id"],
                    update_index=update["update_index"],
                    passed=bool(passed),
                    state_hash=update.get("state_hash"),
                    expected_state_hash=old.get("state_hash"),
                )
            )
        segments = window["segments"]
        restarts.append(
            dict(
                arm=window["arm"],
                restart_after=2048,
                process_lifetimes=len(window["commands"]),
                passed=len(segments) == 2
                and len(observed) == 32
                and all(
                    s.get("producer_descriptor") == work["producer_descriptor"] for s in segments
                )
                and all(u["passed"] for u in rows if u["unit_id"] == window["unit_id"])
                and [u["update_index"] for u in observed] == list(range(2033, 2065))
                and all(s["durable_commits"] == 32 and s["fsync_calls"] >= 128 for s in segments)
                and [s["checkpoint"] for s in segments]
                == [s["checkpoint"] for s in reference_windows],
            )
        )
    return rows, restarts


def build(work: Json, receipts: list[Json], output: Path) -> Json:
    """Readiness needs complete owned execution; it says nothing about a speed gain."""
    producer, raw = work["producer_descriptor"], Path(work["raw"])
    exact_identity(producer, producer)
    rows = work["window_rows"]
    parity_rows, restart_rows = parity(work)
    owned = bool(receipts) and all(r["passed"] for r in receipts)
    coverage = any(r["name"] == "coverage_report" and r["passed"] for r in receipts)
    complete = sum(r["status"] == "completed" for r in rows)
    gates = dict(
        owned_validation=owned,
        owned_statement_coverage=coverage,
        authenticated_sources=not work["failures"],
        complete_windows=complete == 3,
        exact_parity=len(parity_rows) == 96 and all(r["passed"] for r in parity_rows),
        exact_restart=len(restart_rows) == 3 and all(r["passed"] for r in restart_rows),
        current_consumers=all(r["passed"] for r in work["consumer_admission_rows"])
        and len(work["consumer_admission_rows"]) == 3,
        resource_admission=work["resource_projection"].get("admitted", False),
    )
    ready = all(gates.values()) and not work["private_control"]
    bad = (
        not owned
        or (not coverage and not work["private_control"])
        or (
            not work["failures"]
            and not all(v for k, v in gates.items() if k != "owned_statement_coverage")
        )
    )
    verdict = (
        "disqualified"
        if bad
        else ("blocked" if work["failures"] else "circular_positive" if ready else "null")
    )
    value = dict(
        producer,
        run_date=work["run_date"],
        producer_descriptor=producer,
        honest_verdict="complete_" + verdict + "_durable_producer_identity",
        verdict_class=verdict,
        gate_check_summary=work["failures"],
        inference_substrate="deterministic_verifier",
        model_invocation_counts=dict(language_model_calls=0, model_loads=0, generation_calls=0),
        numerical_training_calls=0,
        numerical_training=dict(performed=False, arithmetic_replay_only=True),
        rows=rows,
        intended_count=3,
        completed_count=complete,
        failed_count=sum(r["status"] == "failed" for r in rows),
        censored_count=sum(r["status"] == "censored" for r in rows),
        excluded_count=0,
        independent_count=int(complete > 0),
        verifier_is_oracle=True,
        exposure_scope="constructed_ordinary_trajectory_identity_qualification_only",
        independent_generalization_score=0,
        durable_producer_ready_score=int(ready),
        producer_identity_rows=[
            dict(producer, phase=phase, passed=True)
            for phase in ["measurement", "worker_request", "cold_replay", "consumer_authentication"]
        ],
        original_authority_rows=work["original_authority_rows"],
        ordinary_starting_state_rows=work["starting_state_rows"],
        parity_rows=parity_rows,
        restart_rows=restart_rows,
        consumer_admission_rows=work["consumer_admission_rows"],
        retained_control_rows=work["retained_controls"],
        acceptance_gates=gates,
        required_checks_passed=not bad,
        flagged_adversarial=bad,
        validation_receipts=receipts,
        terminal_validation_sidecar_path=str(raw / "terminal_validation.json"),
        preconditions_checked=work["preconditions"],
        duration_s=work["duration_s"],
        started_at_utc=work["started_at_utc"],
        ended_at_utc=work["ended_at_utc"],
        phase_spans=work["phase_spans"],
        random_seed=SEED,
        source_artifact_hashes=work["source_refs"],
        code_config_hashes=work["code_refs"] + work["dependency_code_refs"],
        raw_shard_hashes=work["raw_refs"],
        cited_upstream_artifacts=work["source_refs"],
        repository_health_reference=work.get("repository_health_reference", {}),
        scoped_validation_scope=dict(
            owned=OWNED,
            tests=TESTS,
            named_e2es=["E2E-018", "E2E-021"],
            full_suite_pass_claimed=False,
        ),
        private_control=work["private_control"],
        primitive_reference=reference(raw / "measurement.json"),
        validation_reference=reference(raw / "validation.json"),
        publication_output=str(output),
        resource_projection=work["resource_projection"],
        durability_probe=work["durability_probe"],
        excluded_prefix_cost_rows=work["excluded_prefix_cost_rows"],
        admitted_current_hardware_cost_intervals=[],
        historical_cost_intervals_admitted=False,
        historical_alias_admitted=False,
        historical_identity_diagnostics=work["historical_identity_diagnostics"],
        physical_fpga_execution=False,
        long_run_claim_allowed=False,
        methodology_note="Three matched32-update ordinary durable windows cross a real process restart at2048. "
        "Complete cost includes setup, import, persistence, restart and shutdown. Arithmetic prefixes "
        "and the admission probe are separate. Authenticated synthetic threshold controls stay historical. "
        "This oracle parity qualification claims no throughput improvement, useful learning, natural "
        "generalization or physical acceleration. Exp8772's aliased costs remain excluded.",
    )
    value["reproducibility_checksum"] = canonical_hash(
        dict(
            producer=producer,
            method=METHOD,
            sources=work["source_refs"],
            code=work["code_refs"],
            roster=work["roster"],
        )
    )
    value["field_principles"] = {
        k: "Bind exact current identity and actual sealed operands; qualification "
        "does not establish scientific or hardware benefit."
        for k in value
    }
    return value


measurement.NAME, measurement.TASK, measurement.CLI = NAME, TASK, CLI
measurement.OWNED, measurement.TESTS = OWNED, TESTS
measurement.METHOD, measurement.SEED = METHOD, SEED
measurement.progress, measurement.roster = progress, roster
measurement.request, measurement.projection, measurement.build = request, projection, build
