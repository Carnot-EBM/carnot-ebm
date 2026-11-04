"""REQ-REPORT-8774: imported timing is useful only within its producer's scope."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from carnot.reporting import hardware_cost_inputs_8761 as previous
from carnot.reporting import v751_branch_contract as contract
from carnot.reporting.v748_collection_contract import namespace

Json = dict[str, Any]
i = namespace(previous.original, "operation_inputs_8774")
i.contract, i.MILESTONE, i.DESIGN = contract, contract.MILESTONE, contract.DESIGN
BOARD = "results/experiment_8761_v750_hardware_cost_boundaries.json"
BOARD_PIN = "sha256:f0f3e7dd142b277b467ed16d37181501f414687c5a1402bd487960473f80f76c"
BRANCHES = [
    (8766, "learning_cost_ready_score"),
    (8770, "residual_cost_ready_score"),
    (8772, "ordinary_update_cost_ready_score"),
]


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Flush counts so authentication and child validation remain observable."""
    print(f"[exp8774] phase={phase} completed={completed} pending={pending}", flush=True)


def cost_operands(value: Json) -> list[Json]:
    """Keep residual parent clocks separate; durable remainders include setup and storage."""
    if value["experiment_id"] == 8770:
        return [dict(row, parent=row["clock_domain"]) for row in value["component_cost_rows"]]
    if value["experiment_id"] == 8772:
        rows = []
        for window in value["per_window_rows"]:
            arithmetic = sum(u["arithmetic_ns"] for u in window["update_rows"])
            for operation, ns in [
                ("sparse_update_arithmetic", arithmetic),
                ("durable_window_remainder", window["complete_ns"] - arithmetic),
            ]:
                rows.append(
                    dict(
                        operation=operation,
                        duration_s=ns / 1e9,
                        parent_duration_s=window["complete_ns"] / 1e9,
                        process_id=window["unit_id"],
                        parent="ordinary_window",
                        clock_domain="host_perf_counter_ns",
                        includes_setup_and_restart=operation == "durable_window_remainder",
                        missing_leaf_partition="fsync, serialization, imports and restart not separated",
                    )
                )
        return rows
    return list(value.get("complete_cost_rows", []))


def producer_operands(value: Json, path: Path) -> list[Json]:
    """The original reducer authenticates costs; a benefit gate cannot veto a valid null."""
    number = value["experiment_id"]
    progress("producer_replay_before", 0, 1)
    if number == 8772:
        from carnot.reporting.ordinary_durable_runner_8772 import replay

        replay(path)
    elif number == 8770:
        from carnot.reporting.optimized_cost_attribution_runner_8770 import replay as residual

        if not residual(path, value["primitive_reference"]):
            raise ValueError("original_residual_replay")
    else:
        raise ValueError("unregistered_learning_producer")
    refs = [
        value["primitive_reference"],
        value["validation_reference"],
        i.reference(Path(value["terminal_validation_sidecar_path"])),
    ]
    for ref in refs:
        i.load_bytes(ref)
    progress("producer_replay_after", 1, 0)
    return refs


def admit(root: Path, task: Json, field: str) -> Json:
    """Exact declared files distinguish absence, zero and rejected construction."""
    ref = i.reference(root / task["deliverable"])
    check = i.gate(ref, field, 1, task["id"])
    branch: Json = dict(
        task_id=task["id"], field=field, qualified=False, checks=[check], refs=[ref], value={}
    )
    if not check["passed"]:
        if ref["sha256"] is not None:
            branch["observed_cost_rows"] = i.load(ref).get("component_cost_rows", [])
        return branch
    try:
        value = i.load(ref)
        for field_name, expected in [
            ("experiment_id", int(task["id"].split("-")[0][3:])),
            ("task_id", task["id"]),
            ("required_checks_passed", True),
            ("flagged_adversarial", False),
        ]:
            check = i.gate(ref, field_name, expected, task["id"])
            branch["checks"].append(check)
            if not check["passed"]:
                raise ValueError("producer_execution_qualification")
        if value["verdict_class"] not in {"positive", "circular_positive", "null"}:
            raise ValueError("producer_terminal_verdict")
        branch["refs"] += i.terminal(ref, value) + producer_operands(value, Path(ref["path"]))
        branch.update(qualified=True, value=dict(value, complete_cost_rows=cost_operands(value)))
        branch["checks"].append(dict(check, operand_state="qualified"))
    except (OSError, ValueError, KeyError, TypeError) as error:
        branch["checks"].append(
            dict(
                check,
                check="original_producer_authentication",
                field=check["field"] if not check["passed"] else "qualified_original_producer",
                expected=check["expected"] if not check["passed"] else True,
                observed=check["observed"] if not check["passed"] else False,
                passed=False,
                operand_state="rejected_producer",
                reason=str(error),
            )
        )
    return branch


def authenticate(root: Path) -> Json:
    """Current authority and each optional cost are independent of historical board access."""
    authority = contract.authority(root)
    refs, failures, prior = list(authority["references"]), [], {}
    if not authority["ready"]:
        failures.append(
            dict(
                upstream="V751",
                **i.reference(root / contract.DESIGN),
                check="full_object_authority",
                field="canonical_tasks_sha256",
                operator="==",
                expected=contract.PIN,
                observed=None,
                passed=False,
                operand_state="missing_or_changed_authority",
            )
        )
    ref = i.reference(root / BOARD)
    refs.append(ref)
    try:
        prior = i.load(dict(ref, sha256=BOARD_PIN))
        refs += i.terminal(ref, prior)
        if (
            prior["required_checks_passed"] is not True
            or prior["flagged_adversarial"] is not False
            or prior["hardware_mapping_ready_score"] != 1
        ):
            raise ValueError("unqualified_original_mapping")
    except (OSError, ValueError, KeyError, TypeError) as error:
        failures.append(
            dict(
                upstream="exp8761-hardware-cost-boundaries",
                **ref,
                check="pinned_original_mapping",
                field="original_sha256",
                operator="==",
                expected=BOARD_PIN,
                observed=ref["sha256"],
                passed=False,
                operand_state="missing_file" if ref["sha256"] is None else "rejected_producer",
                reason=str(error),
            )
        )
    tasks = {
        r["task_object"]["id"].split("-")[0]: r["task_object"]
        for r in authority["rows"]
        if r.get("task_object")
    }
    branches = []
    for number, field in BRANCHES:
        task = tasks.get(
            f"exp{number}",
            dict(
                id=f"exp{number}-unavailable",
                deliverable=f"results/unavailable_declared_exp{number}.json",
            ),
        )
        branches.append(admit(root, task, field))
        progress("independent_source_admission", len(branches), 3 - len(branches))
    fallback = previous.admit(
        root,
        dict(
            id="exp8757-feature-reuse-cost",
            deliverable="results/experiment_8757_v750_feature_reuse_cost.json",
        ),
        "installed_cost_ready_score",
    )
    refs += [r for b in [*branches, fallback] for r in b["refs"]]
    return dict(
        authority=authority,
        refs=refs,
        failures=failures,
        prior=prior,
        board_qualified=not failures,
        branches=branches,
        inference_fallback=fallback,
    )


i.authenticate = authenticate
