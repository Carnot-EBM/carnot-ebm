"""REQ-VERIFY-8217: independently reduce source, learning and service evidence.

Scientific nulls remain evidence. Branch availability, statistical benefit and
independent deployment each answer a different question.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any

from carnot.reporting import v709_capstone_inputs as e
from carnot.reporting.current_work_receipt import atomic_json
from carnot.reporting.v685_authority_lifecycle import tasks_digest
from carnot.reporting.v708_capstone_science import operand
from carnot.verify import restricted_decision_audit_8210 as action
from carnot.verify import memory_benefit_audit_8212 as memory
from carnot.verify import prospective_service_8214 as service

Json = dict[str, Any]


def measure(data: Json, raw: Path) -> Json:
    """Reuse qualified primitives, sealing each branch before another is inspected."""
    raw.mkdir(parents=True, exist_ok=True)
    binder = e.q.base.Binder(raw / "custody", task=e.TASK)
    branches: Json = dict(
        H1={}, H2={}, service={}, board={}, arc={}, failures=[], references=binder.refs
    )
    tasks, values = data["tasks"], data["primaries"]
    for name, index in [("H1", 4), ("H2", 7), ("service", 9), ("board", 11), ("arc", 10)]:
        value = values.get(tasks[index]["id"], {})
        path = Path(data["dispositions"][index]["path"])
        e.progress("before_reduction_" + name, 0, 1)
        try:
            if not data["dispositions"][index]["eligible"]:
                raise ValueError("qualified_branch_input_unavailable")
            if name == "H1":
                if not value.get("measurement_reference"):
                    raise ValueError("sealed_source_rows_absent")
                work = action.measure(Path(data["root"]), raw / "H1")
                if work["owned_failure"] or not work["evidence"]:
                    branches["failures"].extend(c for c in work["checks"] if not c["passed"])
                    raise ValueError("sealed_source_authentication_failed")
                branches[name] = work["evidence"]
                for ref in [*work["refs"], *work["raw_shard_hashes"]]:
                    binder.bind(Path(ref["path"]), ref["sha256"])
            elif name == "H2":
                trajectory = values[tasks[6]["id"]]
                if not data["dispositions"][6]["eligible"] or not trajectory.get("trajectory_path"):
                    raise ValueError("causal_trajectory_absent")
                for ref in e.q.hash_references(trajectory["raw_shard_hashes"]):
                    binder.bind(Path(ref["path"]), ref["sha256"])
                reconstructed = memory.reconstruct(trajectory, raw / "H2")
                for key, observed in reconstructed.items():
                    if value[key] != observed:
                        raise ValueError("causal_primitive_drift:" + key)
                branches[name] = reconstructed
            elif name == "service":
                ref = next(
                    r
                    for r in e.q.hash_references(value["raw_shard_hashes"])
                    if r["path"] == value["result_path"]
                )
                work = binder.read(Path(ref["path"]), ref["sha256"])
                branches[name] = work["work"]
                if service.reduce(branches[name]) != {
                    k: value[k] for k in service.reduce(branches[name])
                }:
                    raise ValueError("service_primitive_drift")
            elif name == "board":
                ref = value["primitive_reference"]
                primitive = binder.read(Path(ref["path"]), ref["sha256"])
                ref = value["replay_input_reference"]
                inputs = binder.read(Path(ref["path"]), ref["sha256"])
                branches[name] = dict(inputs=inputs, primitive=primitive)
            else:
                primitive = Path(value["primitive_path"])
                ref = next(
                    r
                    for r in e.q.hash_references(value["raw_shard_hashes"])
                    if r["path"] == str(primitive)
                )
                branches[name] = binder.read(primitive, ref["sha256"])
        except (OSError, ValueError, KeyError, TypeError, StopIteration) as error:
            branches[name] = {}
            branches["failures"].append(
                operand(
                    str(path),
                    name + "_qualified_primitive_rows",
                    True,
                    str(error),
                    digest=data["dispositions"][index]["sha256"],
                )
            )
        e.progress("after_reduction_" + name, 1, 0)
    atomic_json(raw / "branch_primitives.json", branches)
    return branches


def reduce(data: Json, branches: Json) -> Json:
    """Keep fixed primary tests and all external obligations in complete accounting."""
    from carnot.reporting import hardware_workload_obligations_8216 as hardware
    from carnot.reporting import arc_authoritative_frontier_8215 as arc

    h1 = action.n.reduce(branches["H1"]) if branches["H1"] else {}
    h2 = memory.statistics(branches["H2"]["legacy_rows"]) if branches["H2"] else {}
    calibration = (
        memory.effects(branches["H2"]["legacy_rows"], "calibration_only", "frozen_qwen_offset")
        if h2
        else {}
    )
    structural = (
        memory.effects(branches["H2"]["legacy_rows"], "error_center", "fixed_public_center")
        if h2
        else {}
    )
    costs = service.reduce(branches["service"]) if branches["service"] else {}
    board = (
        hardware.reduce(
            branches["board"]["inputs"],
            branches["board"]["primitive"]["precision_rows"],
            branches["board"]["primitive"]["storage_probe_rows"],
        )
        if branches["board"]
        else {}
    )
    arc_rows = branches["arc"]
    if arc_rows:
        rebuilt = arc.reduction(arc_rows["rows"], arc_rows["prior_frontier"])
        if any(arc_rows[k] != v for k, v in rebuilt.items()):
            raise ValueError("arc_primitive_drift")
    rows = deepcopy(data["dispositions"])
    failures = data["failures"] + branches["failures"]
    deployment_path = rows[9]["path"]
    deployment_gate = operand(
        deployment_path,
        "cold_inclusive_independent_deployment_lower95_ratio",
        10,
        None,
        ">=",
        rows[9]["sha256"],
    )
    failures = [
        dict(
            g,
            upstream_id=next(
                (
                    t["id"]
                    for t in data["tasks"]
                    if Path(t["deliverable"]).name
                    == Path(g.get("path", g.get("artifact_path", ""))).name
                ),
                g.get("upstream_id", e.TASK),
            ),
        )
        for g in [*failures, deployment_gate]
    ]
    retirements = []
    for prior in data["prior_evidence"]:
        row = next((r for r in rows if r["task_id"] == prior["task_id"]), {})
        tested = row.get("eligible", False) and row.get("verdict_class") == "null"
        current = data["primaries"].get(prior["task_id"], {})
        scope = {
            k: current[k]
            for k in (
                "configuration_scope_sha256",
                "numerical_protocol_sha256",
                "protocol_sha256",
                "config",
                "claim_scope",
            )
            if k in current
        }
        same_scope = bool(
            scope and any(r.get("configuration_scope") == scope for r in prior["evidence"])
        )
        same = (
            tested
            and prior["prior_verdict_matches_artifact"]
            and row.get("honest_verdict") == prior["verdict"]
            and same_scope
        )
        retirements.append(
            dict(
                prior,
                same_verdict=bool(same),
                same_scope_configuration=same_scope,
                scientific_hypothesis_tested=bool(tested),
                retire_exact_configuration=bool(same and prior["retire_if_same_verdict"]),
                retire_method_family=False,
                decision="retire_matching_configuration"
                if same
                else "preserve_changed_or_untested_scope",
            )
        )
    verdict = "complete_blocked_external_science_obligations"
    rows.append(
        dict(
            experiment_id=8217,
            task_id=e.TASK,
            unit_id=e.TASK,
            source_cluster_id="owned_accounting",
            arm="task_disposition",
            condition="terminal_accounting",
            metric="eligible_branch",
            numerator=1,
            denominator=1,
            status="completed",
            completed=True,
            eligible=True,
            qualified=True,
            excluded=False,
            failed=False,
            censored=False,
            honest_verdict=verdict,
            verdict_class="blocked",
            disposition="self_owned_completion",
        )
    )
    gaps = {
        "source_decisions": dict(
            requirements=["FR-06", "FR-12"],
            closed=False,
            status="completed_null" if h1 else "blocked",
            evidence_path=rows[4]["path"],
            remaining="Independent source labels and generalization are required.",
        ),
        "later_learning_retention": dict(
            requirements=["FR-11"],
            closed=False,
            status="completed_null" if h2 else "blocked",
            evidence_path=rows[7]["path"],
            remaining="Structural action benefit and independent retention remain unproved.",
        ),
        "service_deployment": dict(
            requirements=["FR-05", "FR-08", "NFR-01"],
            closed=False,
            status="blocked",
            evidence_path=deployment_path,
            unmet_operands=[deployment_gate],
            remaining="Cold-inclusive independent deployment must meet the original10x threshold.",
        ),
    }
    return dict(
        honest_verdict=verdict,
        verdict_class="blocked",
        rows=rows,
        task_dispositions=rows,
        task_contract=data["tasks"],
        canonical_tasks_sha256=tasks_digest(data["tasks"]),
        authority=data["authority"],
        original_contract_failures=data["original_contract_failures"],
        H1=dict(
            status="completed_signal"
            if h1.get("h1_development_signal_score")
            else "completed_null"
            if h1
            else "blocked",
            alpha=0.025,
            intended_count=128,
            completed_count=h1.get("completed_count", 0),
            statistics=h1,
            policy_versus_baseline=h1.get("policy_gain"),
            energy_increment=h1.get("energy_increment_gain"),
            upstream_audit_disposition=rows[5]["disposition"],
            original_missing_mask=[
                not r["complete_pair"] for r in h1.get("rows", []) if r["arm"] == "energy"
            ],
        ),
        H2=dict(
            status="completed_signal"
            if h2.get("h2_passed") and h2.get("retention_passed")
            else "completed_null"
            if h2
            else "blocked",
            alpha=0.025,
            intended_count=192,
            completed_count=h2.get("completed_count", 0),
            statistics=h2,
            causal_checks=branches["H2"].get("causal_checks", {}),
            calibration_only_effect=calibration,
            structural_memory_effect=structural,
            retention_rows=h2.get("retention_rows", []),
        ),
        multiplicity=dict(
            family=["H1", "H2"], alpha_per_hypothesis=dict(H1=0.025, H2=0.025), alpha_transfer=False
        ),
        h1_development_signal_score=h1.get("h1_development_signal_score", 0),
        h2_development_signal_score=int(bool(h2.get("h2_passed") and h2.get("retention_passed"))),
        independent_generalization_score=0,
        generalized_learning_benefit_score=0,
        verifier_is_oracle=True,
        exposure_scope="exposed development; no independent generalization credit",
        science_ready_score=0,
        capstone_execution_ready_score=0,
        intended_count=13,
        completed_count=13,
        failed_count=sum(r["failed"] for r in rows),
        excluded_count=sum(r["excluded"] for r in rows),
        censored_count=0,
        independent_count=0,
        eligible_count=sum(r["eligible"] for r in rows),
        gate_check_summary=failures,
        retirement_decisions=retirements,
        gap_decisions=gaps,
        service_evidence_scope=dict(
            statistics=costs,
            primitive_request_count=len(branches["service"].get("requests", [])),
            whole_service_qualified=False,
            nfr01_met=False,
            nfr01_required_lower95_ratio=10,
            deployment_demand_observed=False,
            scope="current designed research requests; shared acquisition, not independent deployment",
        ),
        board_obligations=board.get("board_rows", []),
        access_obligations=board.get("access_obligations", []),
        arc_evidence=dict(
            primitive=arc_rows,
            scientific_rerun=False,
            new_solve_credit=0,
            current_game_execution_count=0,
            policy_changed=False,
        ),
        acceptance_gates=dict(
            independent_deployment=dict(
                passed=False, principle="Designed shared acquisition cannot close NFR-01."
            ),
            owned_validation=dict(passed=False, principle="Every frozen owned check must pass."),
        ),
    )
