"""REQ-REPORT-8303: separate completed accounting from measured scientific work.

The qualified readers retain byte custody. This adapter preserves failure
scope and checks the new CPU fixture independently of absent natural evidence.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import re
import tempfile
from typing import Any

from carnot.reporting import v715_capstone_evidence as qualified
from carnot.reporting import v713_capstone_evidence as prior
from carnot.reporting import dependency_admission_execution_8291 as dependency
from carnot.reporting.current_work_receipt import canonical_hash
from carnot.reporting.primary_publication import validate_primary
from carnot.reporting.roadmap_contract import parse_design
from carnot.reporting.v685_authority_lifecycle import assess_authorities
from carnot.reporting.v710_contract_replay import require_reference, snapshot

Json = dict[str, Any]
ROOT = prior.ROOT
NAME = "experiment_8303_v716_capstone"
CLI = f"scripts/experiments/{NAME}.py"
TEST = "tests/python/test_v716_capstone_8303.py"
MILESTONE = "2026.10.716"
DESIGN, ACTIVE, STAGED, PROTOCOL = (
    qualified.DESIGN,
    qualified.ACTIVE,
    qualified.STAGED,
    qualified.PROTOCOL,
)
MODEL_SPECS: list[Json] = []
OWNED = [
    "python/carnot/reporting/v716_capstone_evidence.py",
    "python/carnot/reporting/v716_capstone.py",
    CLI,
]
HISTORY = [
    "results/experiment_8289_v715_capstone.json",
    "results/experiment_8286_v715_arc_outcome_frontier.json",
    "results/experiment_8259_v713_polarfire_dispatch_qualification.json",
    "results/experiment_8277_v715_lease_backend_qualification.json",
    "openspec/change-proposals/research-roadmap-v715-preserved-20261008.md",
    "research-complete.yaml",
    "ops/conductor-log.md",
    "ops/exclusion_manifest.yaml",
    "ops/arc_solve_registry.yaml",
    "research-hardware-wishlist.md",
    "results/experiment_8261_v713_capstone.json",
    "results/experiment_8250_evidence_view_canary.json",
]
bind = qualified.bind


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Flush counts so long reductions remain visible to the task supervisor."""
    print(f"[exp8303] phase={phase} completed={completed} pending={pending}", flush=True)


def authority(refs: list[Json], raw: Path) -> Json:
    """Use the existing full-object reader with the current explicit task range."""
    paths = [Path(r.get("snapshot_path", raw / f"absent-{i}")) for i, r in enumerate(refs[:3])]
    text = paths[0].read_text()
    tasks = parse_design(text, milestone=MILESTONE)[1]
    digest = canonical_hash(tasks)[7:]
    printed = re.search(r"Canonical task digest: `([a-f0-9]{64})`", text)
    if printed and printed[1] != digest:
        raise ValueError("canonical_digest_drift")
    reader = raw / "reader.md"
    reader.parent.mkdir(parents=True, exist_ok=True)
    reader.write_text(text + "\nCanonical full-task SHA256: `" + digest + "`\n")
    return dict(
        assess_authorities(
            reader,
            paths[1],
            paths[2],
            raw / "assessment",
            milestone=MILESTONE,
            first_id=8290,
            count=14,
        )
    )


def measure(root: Path, raw: Path) -> Json:
    """Freeze declared paths first, then authenticated alternate gate receipts."""
    refs: list[Json] = [
        snapshot(root / p, raw / "custody", str(i))
        for i, p in enumerate([DESIGN, STAGED, ACTIVE, PROTOCOL])
    ]
    tasks = parse_design(Path(refs[0]["snapshot_path"]).read_text(), milestone=MILESTONE)[1]
    if len(tasks) != 14 or [t["id"].split("-")[0] for t in tasks] != [
        f"exp{i}" for i in range(8290, 8304)
    ]:
        raise ValueError("exact_fourteen_task_contract")
    inputs = []
    for index, task in enumerate(tasks[:-1]):
        progress("input_before", index, 13 - index)
        inputs.append(bind(prior.resolve(task, 8290 + index, root), raw, refs))
        progress("input_after", index + 1, 12 - index)
    history = {Path(p).stem: bind(root / p, raw, refs) for p in HISTORY[:4]}
    for p in HISTORY[4:]:
        refs.append(snapshot(root / p, raw / "custody", str(len(refs))))
    polar = history[Path(HISTORY[2]).stem]["reference"]
    if polar["exists"]:
        for key in ["board_reference", "host_reference", "input_reference"]:
            ref = prior.read(polar)[key]
            refs.append(
                dict(
                    snapshot(Path(ref["path"]), raw / "custody", str(len(refs))),
                    expected_sha256=ref["sha256"],
                )
            )
    historical = history[Path(HISTORY[0]).stem]["reference"]
    if historical["exists"]:
        value = prior.read(historical)
        activation = value["activation_snapshot"]
        refs.append(
            dict(
                snapshot(Path(activation["snapshot_path"]), raw / "custody", str(len(refs))),
                expected_sha256=activation["sha256"],
                evidence_role="historical_planning_activation",
            )
        )
        for receipt in value["validation_receipts"]:
            if receipt["name"] == "branch_8286_replay":
                for stream in ["stdout", "stderr"]:
                    refs.append(
                        dict(
                            snapshot(
                                Path(receipt[stream + "_path"]), raw / "custody", str(len(refs))
                            ),
                            expected_sha256=receipt[stream + "_sha256"],
                            evidence_role="authenticated_historical_failure_log",
                        )
                    )
    return dict(tasks=tasks, inputs=inputs, history=history, references=refs)


def outcome(task: Json, identity: int, item: Json) -> tuple[Json, list[Json], Json]:
    """An authenticated failed producer is evidence; an owned reader crash is not."""
    ref = item["reference"]
    try:
        row, failures, source = qualified.outcome(task, identity, item)
    except Exception as error:
        row = dict(
            experiment_id=identity,
            task_id=task["id"],
            unit_id=task["id"],
            arm="task_disposition",
            condition="terminal_accounting",
            status="completed",
            completed=True,
            missing=False,
            producer_executed=False,
            eligible=False,
            excluded=True,
            failed=True,
            censored=False,
            numerator=0,
            denominator=1,
            honest_verdict=None,
            verdict_class="disqualified",
            disposition="owned_reader_exception",
            evidence_type="reader_failure",
            path=ref["path"],
            sha256=ref["sha256"],
        )
        return (
            row,
            [
                prior.operand(
                    task["id"],
                    ref["path"],
                    ref["sha256"],
                    "reader_normal_return",
                    True,
                    repr(error),
                )
            ],
            {},
        )
    if row["missing"]:
        row["disposition"] = "corrupted_artifact" if ref["exists"] else "absent_producer"
        # Failed terminal checks remain authentic when their report binds these bytes.
        try:
            value, side = prior.read(ref), prior.read(item["sidecar"])
            validate_primary(value, Path(ref["path"]))
            if (
                value["task_id"] == task["id"]
                and value["verdict_class"] == "disqualified"
                and side["primary_sha256"] == ref["sha256"]
                and not item["sidecar"].get("binding_error")
            ):
                source = value
                row.update(
                    producer_executed=True,
                    missing=False,
                    failed=True,
                    numerator=1,
                    honest_verdict=value["honest_verdict"],
                    verdict_class="disqualified",
                    evidence_type="producer_primary",
                    source_counts={k: value[k] for k in prior.COUNTS},
                )
        except (OSError, ValueError, KeyError, TypeError):
            pass
    if row["producer_executed"] and row["verdict_class"] == "disqualified":
        row["disposition"] = "authenticated_upstream_failure"
    return row, failures, source


def reduce(work: Json, receipts: list[Json]) -> Json:
    """Rebuild scope and readiness from primitives without turning absence into zero."""
    ref: Json
    for ref in work["references"]:
        require_reference(ref)
    tasks = parse_design(
        Path(work["references"][0]["snapshot_path"]).read_text(), milestone=MILESTONE
    )[1]
    if tasks != work["tasks"] or len(work["inputs"]) != 13:
        raise ValueError("contract_primitive_drift")
    with tempfile.TemporaryDirectory(prefix="carnot8303-authority-") as private:
        checked = authority(work["references"], Path(private))
    rows, sources, failures = [], {}, list(checked["gate_check_summary"])
    for ref in work["references"]:
        if ref.get("evidence_role") and (
            not ref["exists"] or ref["sha256"] != ref["expected_sha256"]
        ):
            failures.append(
                prior.operand(
                    ref["evidence_role"],
                    ref["path"],
                    ref["sha256"],
                    "historical_evidence_sha256",
                    ref["expected_sha256"],
                    ref["sha256"],
                )
            )
    for identity, task, item in zip(range(8290, 8303), tasks[:-1], work["inputs"]):
        if item["reference"] not in work["references"]:
            raise ValueError("input_reference_drift")
        row, failed, source = outcome(task, identity, item)
        if row["disposition"] == "conductor_pre_gate":
            for gate in prior.read(item["reference"])["gates_evaluated"]:
                ref = next(
                    (r for r in work["references"] if r["path"] == gate["artifact_path"]), {}
                )
                if ref.get("sha256") != gate["artifact_sha256"]:
                    row.update(missing=True, disposition="unbound_pre_gate", numerator=0)
                    failed.append(
                        prior.operand(
                            gate["upstream"],
                            gate["artifact_path"],
                            ref.get("sha256"),
                            "conductor_receipt_bound_sha256",
                            gate["artifact_sha256"],
                            ref.get("sha256"),
                        )
                    )
        rows.append(row)
        sources[identity] = source
        failures.extend(failed)
    history: dict[int, Json] = {}
    for name, item in sorted(work["history"].items()):
        ref = item["reference"]
        task = (
            dict(id=prior.read(ref).get("task_id", "historical"))
            if ref["exists"]
            else dict(id="historical")
        )
        _, failed, source = outcome(task, int(name.split("_")[1]), item)
        history[int(name.split("_")[1])] = source
        failures.extend(failed)
    polar = history[8259]
    polar_item = work["history"][Path(HISTORY[2]).stem]
    terminal = (
        prior.read(polar_item["sidecar"]).get("report", {}).get("receipts", []) if polar else []
    )
    graduated = bool(
        polar
        and polar_item["reference"]["sha256"] == qualified.POLAR_PIN
        and polar.get("required_checks_passed") is True
        and polar.get("flagged_adversarial") is False
        and polar.get("polarfire_workload_validated") is True
        and polar.get("current_device_execution_count", 0) > 0
        and polar.get("host_parity") is True
        and polar.get("board_output_hashes") == polar.get("host_expected_hashes")
        and polar.get("board_output_hashes")
        and {r.get("name") for r in terminal if r.get("passed") and r.get("normal_exit")}
        >= {"cold_replay", "adversarial", "strict_rows"}
        and all(
            r["exists"] and r["sha256"] == r["expected_sha256"]
            for r in work["references"]
            if "expected_sha256" in r and r.get("evidence_role") is None
        )
    )
    if not graduated:
        ref = polar_item["reference"]
        failures.append(
            prior.operand(
                "exp8259",
                ref["path"],
                ref["sha256"],
                "authenticated_board_local_CPU_dispatch_and_hash_parity",
                True,
                False,
            )
        )
    owned = [r for r in receipts if r.get("scope") == "owned"]
    reader_failed = any(r["disposition"] == "owned_reader_exception" for r in rows)
    passed = (
        bool(owned)
        and all(r["passed"] and r.get("normal_exit", True) for r in owned)
        and not reader_failed
    )
    tools = all(r["passed"] for r in receipts if r.get("scope") == "preconditions")
    for r in receipts:
        if not r["passed"] and r.get("scope") in {"owned", "preconditions", "upstream"}:
            failures.append(
                prior.operand(
                    r["name"],
                    r["stdout_path"],
                    r["stdout_sha256"],
                    "normal_exit",
                    r["expected_exit"],
                    r["actual_exit"],
                )
            )
    ready = int(bool(passed and checked["activated"] and tools))
    kind = "blocked" if passed else "disqualified"
    verdict = "complete_" + kind + ("_upstream_evidence" if passed else "_owned_validation")
    rows.append(
        dict(
            experiment_id=8303,
            task_id=tasks[-1]["id"],
            unit_id=tasks[-1]["id"],
            arm="task_disposition",
            condition="terminal_accounting",
            status="completed",
            completed=True,
            missing=False,
            producer_executed=True,
            eligible=bool(ready),
            excluded=not ready,
            failed=not passed,
            censored=False,
            numerator=ready,
            denominator=1,
            metric="owned_execution_readiness",
            honest_verdict=verdict,
            verdict_class=kind,
            disposition="self_owned_completion",
            evidence_type="current_capstone",
        )
    )
    protocol = prior.read(work["references"][3])
    branches = [
        dict(
            status="blocked_unmeasured",
            alpha=0.025,
            statistics=None,
            intended_count=n,
            completed_count=None,
            audit_experiment_id=i,
            registered_protocol=protocol[k],
            shared_fallback="retain original slots; zero paired fallback only on measured qualified arms",
            selected_comparator=None,
            source_information_gain=None,
            energy_specific_gain=None,
            later_constraint_use=None,
            retention_gain=None,
            failed_operand="eligible_independent_audit_and_primitives",
        )
        for k, i, n in [("H1", 8297, 128), ("H2", 8299, 96)]
    ]
    branches[1]["retention_intended_count"] = 32
    h3 = sources[8291]
    replay_ok = any(
        r.get("name") == "branch_8291_replay" and r["passed"] and r["normal_exit"] for r in receipts
    )
    reconstruction: Json = {}
    if h3 and replay_ok:
        h3work = json.loads(
            (Path(h3["terminal_validation_sidecar_path"]).parent / "work.json").read_bytes()
        )
        reconstruction = dependency.reduce_work(h3work)
        if any(h3[k] != v for k, v in reconstruction.items()):
            raise ValueError("h3_primitive_reduction_drift")
    sound = int(
        bool(
            bool(reconstruction)
            and h3.get("required_checks_passed")
            and reconstruction.get("h3_sound")
            and not reconstruction.get("hard_constraint_violations", {}).get("count")
        )
    )
    efficiency = int(sound and reconstruction.get("efficiency_signal", False))
    prior_value = history[8289]
    historical_failed = [
        dict(
            r,
            disposition="authenticated_historical_failure",
            producer_verdict=history[8286].get("honest_verdict"),
        )
        for r in prior_value.get("validation_receipts", [])
        if r["name"] == "branch_8286_replay"
    ]
    boards = []
    for board, identity, source, condition in [
        (
            "PolarFire",
            8259,
            polar,
            "Authenticated board-local Linux CPU dispatch and hash parity; no FPGA fabric acceleration.",
        ),
        (
            "KV260",
            8301,
            sources[8301],
            "Useful implemented k<=5 workload via ssh kria, authenticated fabric transcript and transfer-inclusive clocks; no host storage prerequisite.",
        ),
        (
            "GateMate",
            8302,
            sources[8302],
            "Dated real cable/port/power/board change after Exp8288, GM1Ax IDCODE0x20000001, flash and n16 sample/hash smoke; no unchanged probe.",
        ),
    ]:
        obligation = source.get(board.lower() + "_obligation", {})
        historical = obligation.get("historical", obligation)
        boards.append(
            dict(
                board=board,
                obligation=obligation,
                experiment_id=identity,
                evidence=source,
                benefit_score=0,
                terminal_condition=condition,
                historical_transcript_hashes=dict(
                    path=historical.get("source_transcript"),
                    sha256=historical.get("source_transcript_sha256"),
                    board_reference=source.get("board_reference"),
                ),
            )
        )
    retirements = []
    for identity, task in zip(range(8290, 8304), tasks):
        condition = qualified.NEXT.get(
            identity - 14,
            "Independent event/crash reconstruction and complete paired costs; natural adoption needs separate preregistration.",
        )
        if identity == 8291:
            condition = "Separately preregister natural constraint extraction and distinct-source later benefit after fixture closure qualification."
        retirements.append(
            dict(
                task_id=task["id"],
                predecessors=task.get("prior_failures", []),
                decision="carry_anti_churn_obligation"
                if identity >= 8300
                else "await_eligible_evidence",
                informative_registered_null=False,
                learnable_control_passed=None,
                oracle_action_headroom=None,
                reopening_condition=condition,
            )
        )
    accounting = [
        dict(
            experiment_id=i,
            path=rows[i - 8290]["path"],
            sha256=rows[i - 8290]["sha256"],
            evidence_type=rows[i - 8290]["evidence_type"],
            counts=sources[i].get("model_invocation_counts"),
            spans=sources[i].get("load_spans", sources[i].get("service_phase_spans")),
            acquisition_cost_s=None,
            current_capture_status="unavailable",
            imported_canary_counted_once=i == 8292,
        )
        for i in [8290, 8292, 8293, 8294, 8296]
    ]
    return dict(
        honest_verdict=verdict,
        verdict_class=kind,
        rows=rows,
        task_dispositions=deepcopy(rows),
        intended_count=14,
        completed_count=14,
        failed_count=sum(r["failed"] for r in rows),
        censored_count=0,
        excluded_count=sum(r["excluded"] for r in rows),
        independent_count=0,
        actual_executed_task_count=sum(r["producer_executed"] for r in rows),
        pre_gate_count=sum(r["disposition"] == "conductor_pre_gate" for r in rows),
        missing_output_count=sum(r["missing"] for r in rows),
        required_checks_passed=passed,
        flagged_adversarial=False,
        capstone_execution_ready_score=ready,
        science_ready_score=0,
        H1=branches[0],
        H2=branches[1],
        h1_development_signal_score=0,
        h2_development_signal_score=0,
        H3=dict(
            status="disqualified_fixture_hard_constraint"
            if reconstruction.get("hard_constraint_violations", {}).get("count", 0)
            else "circular_positive_fixture"
            if efficiency
            else "informative_fixture_null"
            if sound
            else "unqualified_fixture",
            independent_reconstruction=reconstruction,
            natural_benefit=False,
            energy_specific_advantage=False,
        ),
        h3_fixture_soundness_score=sound,
        h3_fixture_efficiency_signal_score=efficiency,
        verifier_is_oracle=True,
        exposure_scope="exposed development",
        independent_generalization_score=0,
        generalized_learning_benefit_score=0,
        gate_check_summary=failures,
        task_contract=tasks,
        canonical_tasks_sha256=canonical_hash(tasks)[7:],
        full_task_authority_equal=checked["activated"],
        activation_snapshot=work["references"][2],
        archive_lag=dict(
            planning_archive_stopped_at="V714",
            v715_authorities="preserved design, planning active-roadmap hash, primaries and conductor log",
            historical_executed_count=6,
            historical_pre_gate_count=1,
            historical_missing_count=7,
            planning_activation_snapshot=prior_value.get("activation_snapshot"),
        ),
        historical_v715={
            k: prior_value.get(k)
            for k in [
                "honest_verdict",
                "verdict_class",
                "actual_executed_task_count",
                "pre_gate_count",
                "missing_output_count",
                "activation_snapshot",
                "rows",
            ]
        },
        historical_CUDA_block=history[8277],
        historical_failure_dispositions=historical_failed,
        branch_replay_receipts=[r for r in receipts if r.get("scope") == "upstream"],
        polarfire_graduation=dict(
            graduated=graduated,
            inherited=True,
            scope="board-local Linux CPU",
            new_scientific_benefit=False,
        ),
        polarfire_terminal_evidence_hashes=[
            r for r in work["references"] if "8259" in r["path"] or "expected_sha256" in r
        ],
        board_obligations=boards,
        arc_evidence={
            k: sources[8300].get(k)
            for k in [
                "honest_verdict",
                "verdict_class",
                "required_checks_passed",
                "new_outcome_count",
                "credited_new_levels",
                "current_game_execution_count",
                "shared_arms",
                "arm_overlap_games",
                "reopen_condition",
                "frontier_hashes",
                "validation_receipts",
            ]
        },
        live_call_accounting=accounting,
        current_capture_cost_scope=dict(
            status="unavailable",
            complete_acquisition_update_cost_s=None,
            imported_calls_are_new_execution=False,
            current_capstone_calls=0,
        ),
        evidence_improved=bool(sound),
        learning_improved=False,
        evidence_improvement_scope="Independent H3 deterministic CPU mechanics only; H1/H2 unmeasured; inherited PolarFire CPU graduation.",
        three_prd_gaps=[
            dict(gap=g, requirements=req, closed=False, moved=False, remaining=qualified.NEXT[i])
            for g, req, i in [
                ("useful_verified_decisions", ["FR-06", "FR-12"], 8283),
                ("later_learning_and_retention", ["FR-11"], 8285),
                ("request_scale_deployment", ["FR-05", "FR-08", "NFR-01"], 8287),
            ]
        ],
        retirements=retirements,
        acceptance_gates=dict(
            owned_validation=passed,
            full_task_authority=checked["activated"],
            terminal_accounting=True,
            independent_science=False,
        ),
    )
