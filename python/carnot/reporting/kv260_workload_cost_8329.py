"""REQ-REPORT-8329: preserve independent branches and unavailable observations.

Current producers are read only at their declared paths. Private test data can
exercise costs but cannot fill an absent production branch or grant board credit.
"""

from __future__ import annotations

import json
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory
import time
from typing import Any

from carnot.reporting import kv260_local_cost_boundary_8315 as base
from carnot.reporting import kv260_workload_primitives_8329 as p
from carnot.reporting import v718_contract_replay as authority_module
from carnot.reporting import v718_replay_history as history
from carnot.reporting import polarfire_packet_evaluator_8245 as packet_evaluator
from carnot.reporting.current_work_receipt import (
    ZERO_INVOCATION_COUNTS,
    atomic_json,
    canonical_hash,
)
from carnot.reporting.evidence_features_custody_7980 import checked
from carnot.reporting.primary_publication import read_bound_sidecar
from carnot.reporting.request_trace_inventory_8200 import operand

Json = dict[str, Any]
ROOT = base.ROOT
NAME = "experiment_8329_v718_kv260_workload_cost"
CLI = "scripts/experiments/" + NAME + ".py"
TEST = "tests/python/test_kv260_workload_cost_8329.py"
OWNED = [
    "python/carnot/reporting/kv260_workload_" + name + "_8329.py"
    for name in ["cost", "primitives", "runner"]
] + [CLI]
MODEL_SPECS: list[Json] = []
SOURCES = dict(
    constructed=("experiment_8319_v718_local_evidence_qualification", "local_kernel_ready_score"),
    capacity=("experiment_8323_v718_bounded_feedback_capacity", "capacity_ready_score"),
    natural=("experiment_8322_v718_continuous_local_learning", "trajectory_ready_score"),
)
SOURCE_TASKS = dict(
    constructed="exp8319-local-evidence-qualification",
    capacity="exp8323-bounded-feedback-capacity",
    natural="exp8322-continuous-local-learning",
)
reference, pin = base.reference, base.pin


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Flushed counts keep real child and workload progress visible to the operator."""
    print(f"[exp8329] phase={phase} completed={completed} pending={pending}", flush=True)


def polarfire(root: Path, raw: Path, work: Json) -> Json:
    """Retain CPU graduation only after terminal bytes and dispatch parity authenticate.

    Old planning files may change; only the authenticated terminal execution closure
    is imported. No current board probe or old FPGA acceleration claim is made.
    """
    path = root / "results/experiment_8259_v713_polarfire_dispatch_qualification.json"
    value = base.probe(path, raw, work, "polarfire_workload_validated")
    if value is None:
        return dict(polarfire_workload_validated=False, scope="unmet_terminal_authentication")
    try:
        terminal = json.loads(Path(value["terminal_validation_sidecar_path"]).read_bytes())
        report = read_bound_sidecar(path, Path(terminal["publication"]["sidecar_path"]))["report"]
        for receipt in report["receipts"] + value["validation_receipts"]:
            if receipt["passed"] is not True:
                raise ValueError("failed_polarfire_terminal_receipt")
            for stream in ["stdout", "stderr"]:
                pin(
                    checked(
                        dict(path=receipt[stream + "_path"], sha256=receipt[stream + "_sha256"])
                    ),
                    raw,
                    work["refs"],
                )
            if receipt["name"] == "adversarial":
                audit = json.loads(Path(receipt["stdout_path"]).read_bytes())
                if (
                    receipt["actual_exit"] != 0
                    or len(audit["reports"]) != 1
                    or audit["reports"][0]["loaded"] is not True
                    or audit["reports"][0]["flags"]
                    or audit["reports"][0]["flag_count"] != 0
                ):
                    raise ValueError("unadjudicated_polarfire_adversarial_findings")
        packet_path = checked(dict(path=value["packet_path"], sha256=value["packet_sha256"]))
        packet = json.loads(pin(packet_path, raw, work["refs"]).read_bytes())
        evaluator = ROOT / "python/carnot/reporting/polarfire_packet_evaluator_8245.py"
        checked(dict(path=str(evaluator), sha256=packet["evaluator_sha256"]))
        pin(evaluator, raw, work["refs"])
        board = json.loads(pin(checked(value["board_reference"]), raw, work["refs"]).read_bytes())
        host = json.loads(pin(checked(value["host_reference"]), raw, work["refs"]).read_bytes())
        wanted = packet_evaluator.evaluate(packet)
        if (
            not board["executed"]
            or not board["ready"]
            or board["output"] != wanted
            or host["output"] != wanted
        ):
            raise ValueError("polarfire_dispatch_parity")
        for receipt in board["receipts"]:
            for stream in ["stdout", "stderr"]:
                pinned = pin(
                    checked(
                        dict(path=receipt[stream + "_path"], sha256=receipt[stream + "_sha256"])
                    ),
                    raw,
                    work["refs"],
                )
                if (
                    receipt["name"] == "board_evaluate"
                    and stream == "stdout"
                    and json.loads(pinned.read_bytes()) != wanted
                ):
                    raise ValueError("polarfire_output_receipt")
        return dict(
            polarfire_workload_validated=True,
            scope="board_local_Linux_CPU_only",
            dispatch_sha256=value["packet_sha256"],
            output_sha256=wanted["output_sha256"],
            host_output_parity=True,
            fpga_fabric_acceleration=False,
        )
    except (OSError, ValueError, KeyError, TypeError) as error:
        work["checks"].append(
            operand("polarfire_terminal_authentication", path, "valid", str(error))
        )
        return dict(polarfire_workload_validated=False, scope="unmet_terminal_authentication")


def measure(root: Path, raw: Path) -> Json:
    """Authenticate preconditions before measuring every eligible branch independently."""
    start = time.monotonic_ns()
    work: Json = dict(
        checks=[],
        refs=[],
        timing_rows=[],
        branches={},
        authority={},
        root=str(root),
        phase_spans=[],
        historical_model_provenance=[],
    )
    with TemporaryDirectory(prefix="carnot8329-") as directory:
        scratch = Path(directory)
        (scratch / "probe").write_bytes(b"private")
        work["checks"].append(
            operand(
                "private_scratch", scratch, True, (scratch / "probe").read_bytes() == b"private"
            )
        )
        work["checks"].append(
            operand(
                "disk_free_at_least_1GiB", scratch, True, shutil.disk_usage(scratch).free >= 1024**3
            )
        )
        for tool in ["python", "pytest", "coverage", "ruff", "mypy"]:
            work["checks"].append(
                operand(
                    "tool_available",
                    ROOT / ".venv/bin" / tool,
                    True,
                    (ROOT / ".venv/bin" / tool).is_file(),
                )
            )
        progress("before_authority")
        try:
            work["authority"] = authority_module.authority(root, raw / "authority")
            work["checks"].append(
                operand(
                    "activated_V718_authority",
                    root / authority_module.ACTIVE,
                    True,
                    work["authority"]["activated"],
                )
            )
            for name in [
                authority_module.DESIGN,
                authority_module.ACTIVE,
                authority_module.PROTOCOL,
            ]:
                pin(root / name, raw, work["refs"])
            work["checks"].append(
                operand(
                    "frozen_science_sha256",
                    root / authority_module.PROTOCOL,
                    authority_module.base.PIN,
                    base.sha256_file(root / authority_module.PROTOCOL),
                )
            )
        except (OSError, ValueError, KeyError, TypeError) as error:
            work["checks"].append(
                operand(
                    "activated_V718_authority", root / authority_module.ACTIVE, True, str(error)
                )
            )
        progress("after_authority")
        can_measure = all(x["passed"] for x in work["checks"])
        for index, (branch, (name, field)) in enumerate(SOURCES.items()):
            progress("before_authenticate_" + branch, index, 3 - index)
            value = base.probe(root / "results" / (name + ".json"), raw, work, field)
            if value is not None:
                for key, wanted in [
                    ("task_id", SOURCE_TASKS[branch]),
                    ("milestone", "2026.10.718"),
                ]:
                    work["checks"].append(
                        operand(key, root / "results" / (name + ".json"), wanted, value.get(key))
                    )
                if (
                    value.get("task_id") != SOURCE_TASKS[branch]
                    or value.get("milestone") != "2026.10.718"
                ):
                    value = None
            work["branches"][branch] = dict(
                eligible=False, source_count=None, path=str(root / "results" / (name + ".json"))
            )
            progress("after_authenticate_" + branch, index + 1, 2 - index)
            if value is not None and can_measure:
                measuring = False
                try:
                    ref = value.get("workload_reference", value.get("protocol_reference"))
                    plan = json.loads(pin(checked(ref), raw, work["refs"]).read_bytes())
                    trajectories = plan["trajectories"]
                    if not trajectories or len({t["id"] for t in trajectories}) != len(
                        trajectories
                    ):
                        raise ValueError("workload_source_denominator")
                    for trajectory in trajectories:
                        p.expected(trajectory, "indexed")
                    work["branches"][branch].update(eligible=True, source_count=len(trajectories))
                    work["historical_model_provenance"].append(
                        dict(branch=branch, provenance=value.get("historical_model_provenance", []))
                    )
                    measuring = True
                    for arm in p.ARMS:
                        for repetition in range(-1, 5):
                            for n, trajectory in enumerate(trajectories):
                                progress(
                                    f"before_benchmark_{branch}_{arm}_{repetition}",
                                    n,
                                    len(trajectories) - n,
                                )
                                row = p.transaction(trajectory, arm, scratch / "checkpoint.json")
                                p.verify_row(row)
                                if repetition >= 0:
                                    work["timing_rows"].append(
                                        dict(
                                            row,
                                            branch=branch,
                                            source_id=trajectory["id"],
                                            repetition=repetition,
                                        )
                                    )
                                progress(
                                    f"after_benchmark_{branch}_{arm}_{repetition}",
                                    n + 1,
                                    len(trajectories) - n - 1,
                                )
                except (OSError, ValueError, KeyError, TypeError) as error:
                    work["branches"][branch]["eligible"] = False
                    work["branches"][branch]["owned_failure"] = measuring
                    progress("after_failed_benchmark_" + branch)
                    work["timing_rows"] = [x for x in work["timing_rows"] if x["branch"] != branch]
                    work["checks"].append(
                        operand(
                            "qualified_workload_primitives",
                            root / "results" / (name + ".json"),
                            "valid",
                            str(error),
                        )
                    )
    progress("before_hardware_authentication")
    historical = base.probe(
        root / "results/experiment_8315_v717_kv260_local_cost_boundary.json", raw, work, None
    )
    work["kv260_history"] = historical.get("kv260_obligation", {}) if historical else {}
    work["polarfire_graduation"] = polarfire(root, raw, work)
    work["checks"].append(
        operand(
            "compatible_update_kernel",
            root / "results/experiment_8315_v717_kv260_local_cost_boundary.json",
            True,
            False if historical else None,
        )
    )
    progress("after_hardware_authentication")
    ended = time.monotonic_ns()
    work.update(
        duration_s=(ended - start) / 1e9,
        phase_spans=[
            dict(
                phase="preconditions_and_available_costs",
                started_monotonic_ns=start,
                ended_monotonic_ns=ended,
            )
        ],
        code_config_hashes=[
            reference(ROOT / name)
            for name in [
                *OWNED,
                TEST,
                "python/carnot/verify/local_update_isolation_8306.py",
                "python/carnot/reporting/primary_publication.py",
                "scripts/adversarial_verify.py",
                "scripts/verdict_row_consistency_lint.py",
                "ops/exclusion_manifest.yaml",
            ]
        ],
    )
    atomic_json(raw / "measurement.json", work)
    return work


def build(work: Json, raw: Path, receipts: list[Json]) -> Json:
    """Keep all intended source arms while unrelated external blocks stay separate."""
    owned = (
        bool(receipts)
        and all(row["passed"] for row in receipts)
        and not any(b.get("owned_failure") for b in work["branches"].values())
    )
    rows = []
    for branch, detail in sorted(work["branches"].items()):
        identifiers = sorted({r["source_id"] for r in work["timing_rows"] if r["branch"] == branch})
        for identity in identifiers or [branch + ":unavailable"]:
            for arm in p.ARMS:
                clocks = [
                    r
                    for r in work["timing_rows"]
                    if (r["branch"], r["source_id"], r["arm"]) == (branch, identity, arm)
                ]
                complete = (
                    detail["eligible"]
                    and len(clocks) == 5
                    and {r["repetition"] for r in clocks} == set(range(5))
                )
                rows.append(
                    dict(
                        branch=branch,
                        source_id=identity,
                        condition=branch + ":" + arm,
                        arm=arm,
                        intended=1,
                        completed=complete,
                        failed=bool(detail.get("owned_failure")),
                        censored=not complete and not detail.get("owned_failure", False),
                        excluded=False,
                        independent=int(complete and branch == "natural"),
                        numerator=int(complete),
                        denominator=1,
                        eligible=detail["eligible"],
                        evidence_status="measured_CPU"
                        if complete
                        else "failed_owned_measurement"
                        if detail.get("owned_failure")
                        else "unavailable_external_operand",
                    )
                )
    completed = sum(int(row["completed"]) for row in rows)
    failures = [row for row in work["checks"] if not row["passed"]]
    kind = "disqualified" if not owned else "blocked" if failures else "circular_positive"
    hardware = p.boundary(work["timing_rows"])
    value: Json = dict(
        experiment_id=8329,
        task_id="exp8329-kv260-workload-cost",
        milestone="2026.10.718",
        run_date="20261009",
        honest_verdict="complete_"
        + kind
        + "_"
        + (
            "owned_checks"
            if not owned
            else failures[0]["upstream"]
            if failures
            else "qualified_CPU_costs"
        ),
        verdict_class=kind,
        gate_check_summary=work["checks"],
        inference_substrate="cached_candidate_scoring",
        inference_substrate_class="no_model_load",
        MODEL_SPECS=[],
        model_invocation_counts=dict(ZERO_INVOCATION_COUNTS),
        historical_model_provenance=work["historical_model_provenance"],
        rows=rows,
        intended_count=len(rows),
        completed_count=completed,
        failed_count=sum(int(row["failed"]) for row in rows),
        censored_count=sum(int(row["censored"]) for row in rows),
        excluded_count=0,
        independent_count=len({r["source_id"] for r in rows if r["independent"]}),
        sample_size_budget=dict(
            intended_branches=3,
            arms=2,
            repetitions_per_source_arm=5,
            warmups_per_source_arm=1,
            repetitions_are_independent=False,
            source_counts={k: v["source_count"] for k, v in work["branches"].items()},
        ),
        verifier_is_oracle=True,
        exposure_scope="exposed_cached_development",
        independent_generalization_score=0,
        generalized_learning_benefit_score=0,
        required_checks_passed=owned,
        flagged_adversarial=bool(work.get("adversarial_findings")),
        acceptance_gates=dict(
            owned_checks=owned,
            available_CPU_branches=owned and completed > 0,
            natural_costs=owned and any(r["completed"] and r["branch"] == "natural" for r in rows),
            board_execution=False,
        ),
        validation_receipts=receipts,
        terminal_validation_sidecar_path=str((raw / "terminal_validation.json").absolute()),
        adversarial_findings=work.get("adversarial_findings", []),
        finding_dispositions=work.get("finding_dispositions", []),
        finding_consumer_policy=history.POLICY,
        preconditions_checked=True,
        duration_s=work["duration_s"],
        phase_spans=work["phase_spans"],
        random_seed=7188329,
        source_artifact_hashes=work["refs"],
        code_config_hashes=work["code_config_hashes"],
        raw_shard_hashes=[reference(raw / "measurement.json")],
        measurement_reference=reference(raw / "measurement.json"),
        cited_upstream_artifacts=[
            dict(
                branch=branch,
                path=detail["path"],
                sha256=next(
                    (
                        r["sha256"]
                        for r in work["refs"]
                        if r["original_path"] == str(Path(detail["path"]).absolute())
                    ),
                    None,
                ),
                imported_fields=[
                    SOURCES[branch][1],
                    "workload_reference",
                    "terminal_validation_sidecar_path",
                ],
            )
            for branch, detail in sorted(work["branches"].items())
        ],
        cpu_cost_ready_score=int(owned and completed > 0),
        natural_cost_ready_score=int(
            owned and any(r["completed"] and r["branch"] == "natural" for r in rows)
        ),
        kv260_execution_ready_score=0,
        timing_rows=work["timing_rows"],
        **hardware,
        board_obligations=dict(
            kv260=dict(
                status="complete_blocked_compatible_update_kernel",
                current_execution=False,
                access=["ssh", "kria"],
                k_max=5,
                overlay="quadratic_Ising_only",
                historical=work["kv260_history"],
                next_requirement="Existing authenticated compatible kernel, identical outputs and transfer-inclusive bounded SSH transcript",
            ),
            polarfire=dict(
                status="authenticated_CPU_dispatch"
                if work["polarfire_graduation"]["polarfire_workload_validated"]
                else "complete_blocked_polarfire_terminal_authentication"
            ),
        ),
        polarfire_graduation=work["polarfire_graduation"],
        polarfire_workload_validated=work["polarfire_graduation"]["polarfire_workload_validated"],
        polarfire_terminal_evidence_hashes=[
            r for r in work["refs"] if "8259" in r["original_path"]
        ],
        current_device_execution_count=0,
        science_parameters=p.SCIENCE,
        invocation_argv=work.get("invocation_argv", []),
        execution_manifest_reference=work.get("execution_manifest_reference"),
        owned_coverage_reference=work.get("owned_coverage_reference"),
        nfr01_status="unproved_without_empirical_tenfold_whole_service_benefit",
        learning_100x_target="aspirational",
        methodology=dict(
            description="Authenticate exact current producer bytes before timing complete sparse/dense CPU transactions.",
            current_model_work=False,
            imported_model_provenance_separate=True,
            no_fixture_substitution=True,
        ),
        preserved_v717_outcomes=dict(
            producer_primaries=8,
            pre_gate_receipts=2,
            absent_primaries=4,
            H1_measured=False,
            H2_measured=False,
            exact_parity_finding="informational_retained",
            historical_authority_failure="V716_reader_read_V717",
            capstone_cold_replay="reduction_drift",
        ),
    )
    value["field_principles"] = {
        key: "Bind current work, exact evidence and separate unavailable external observations from measured outcomes."
        for key in value
    }
    value["field_principles"].update(
        experiment_id="Identify this producer invocation rather than its cited predecessors.",
        task_id="Match the exact activated task and declared deliverable.",
        milestone="Retain the activated V718 execution authority.",
        run_date="Use the invocation date requested by the operator.",
        honest_verdict="Terminal external blocks differ from failed owned execution.",
        verdict_class="Readiness classification does not require a scientific win.",
        gate_check_summary="Every operand records exact path/hash, field, expected and observed values.",
        inference_substrate="Available CPU work scores exposed cached candidates; hardware_smoke requires actual device execution.",
        inference_substrate_class="No model load belongs to current work.",
        MODEL_SPECS="An empty model declaration prevents accidental inherited model claims.",
        model_invocation_counts="Count current model activity only, with zero loads and generations.",
        historical_model_provenance="Imported provenance never increments current model counts.",
        rows="Preserve each source and arm, including missing and failed units.",
        intended_count="Retain all intended source-arm units without repetition inflation.",
        completed_count="Count complete five-repetition source-arm measurements.",
        failed_count="Count owned measurement failures separately from unavailable external operands.",
        censored_count="Keep missing external source arms visible in the denominator.",
        excluded_count="No absent source is silently removed.",
        independent_count="Count unique natural sources once across arms and repetitions.",
        sample_size_budget="Separate source, arm, repetition and warmup counts.",
        verifier_is_oracle="Constructed numerical checks are oracle controls.",
        exposure_scope="Cached development inputs are exposed and grant no independent generalization.",
        independent_generalization_score="Exposed inputs provide no independent generalization credit.",
        generalized_learning_benefit_score="Cost qualification does not establish learning benefit.",
        required_checks_passed="All owned receipts and measurements must pass before readiness.",
        flagged_adversarial="Retain findings even when they prevent readiness.",
        acceptance_gates="CPU, natural and device gates are distinct.",
        validation_receipts="Bind exact argv, exit codes, clocks and stream hashes to owned checks.",
        terminal_validation_sidecar_path="Point readers to byte-bound atomic publication and terminal receipts.",
        adversarial_findings="Keep every unchanged verifier finding.",
        preconditions_checked="Scratch, resources, tools, authority and source eligibility precede timings.",
        duration_s="Report actual measurement time with no runtime padding.",
        phase_spans="Keep original monotonic boundaries of measurement.",
        random_seed="Identify the invocation; imported workload seeds remain unchanged.",
        source_artifact_hashes="Copy exact immutable source bytes before importing conclusions.",
        code_config_hashes="Bind executed adapters and unchanged validators.",
        raw_shard_hashes="Authenticate primitive evidence used in cold replay.",
        cited_upstream_artifacts="Name exact declared producer paths, hashes and imported fields.",
        cpu_cost_ready_score="At least one complete qualified CPU branch survives unrelated external missingness.",
        natural_cost_ready_score="Natural readiness requires its own complete qualified source branch.",
        kv260_execution_ready_score="Unsupported fabric operations cannot grant device execution credit.",
        operation_rows="Keep disjoint measured CPU panels and explicit overlay incompatibility.",
        polarfire_terminal_evidence_hashes="Authenticate terminal dispatch streams and packet bytes without a redundant probe.",
        compatible_fraction="No compatible operation means the service fraction is unavailable, not measured zero.",
        amdahl_upper_bound="A service bound requires a measured compatible fraction with CPU and transfer costs retained.",
        nfr01_met="Tenfold service benefit needs empirical complete-service data.",
        timing_rows="Five clocks after one warmup; repetitions never add sources.",
        board_obligations="CPU costs and unsupported fabric operations have separate obligations.",
        polarfire_graduation="Byte-bound historical dispatch is board-local Linux CPU, with no new probe.",
        field_principles="Each added field states its evidentiary role.",
        reproducibility_checksum="Bind the full reduction excluding this checksum.",
    )
    value["reproducibility_checksum"] = canonical_hash(value)
    return value


def replay(path: Path) -> bool:
    """Verify evidence bytes and reconstruct semantics and every terminal report field."""
    try:
        value = json.loads(path.read_bytes())
        for ref in (
            value["source_artifact_hashes"]
            + value["code_config_hashes"]
            + value["raw_shard_hashes"]
        ):
            checked(ref)
        work = json.loads(checked(value["measurement_reference"]).read_bytes())
        for row in work["timing_rows"]:
            p.verify_row(row)
        for receipt in value["validation_receipts"]:
            for stream in ["stdout", "stderr"]:
                if receipt.get(stream + "_path"):
                    checked(
                        dict(path=receipt[stream + "_path"], sha256=receipt[stream + "_sha256"])
                    )
        return bool(
            build(
                work,
                Path(value["measurement_reference"]["path"]).parent,
                value["validation_receipts"],
            )
            == value
        )
    except (OSError, ValueError, KeyError, TypeError):
        return False
