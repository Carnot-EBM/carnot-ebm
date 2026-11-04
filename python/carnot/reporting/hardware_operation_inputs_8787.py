"""REQ-REPORT-8787: a missing sibling must not erase a qualified cost producer."""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any

from carnot.reporting import hardware_cost_inputs_8761 as inference
from carnot.reporting import hardware_operation_inputs_8721 as original
from carnot.reporting import v752_branch_methods as contract
from carnot.reporting.v748_collection_contract import namespace

Json = dict[str, Any]
i = namespace(original, "operation_inputs_8787")
i.contract, i.MILESTONE, i.DESIGN = contract, contract.MILESTONE, contract.DESIGN
PAIRWISE = "quadratic_Ising_energy_k_max_5"
BOARD = "results/experiment_8774_v751_hardware_operation_decision.json"
BOARD_PIN = "sha256:14ae7cb2ea53d96866dc4c4c9cc0bf2ef965684879756d57cc8249200a32b68e"
BRANCHES = [
    ("exp8780-causal-template-learning", "causal_template_learning", "learning_cost_ready_score"),
    ("exp8784-durable-cost-components", "durable_cost_components", "durable_cost_rows_ready_score"),
    ("exp8785-memory-benefit-audit", "memory_benefit_audit", "learning_audit_ready_score"),
]


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Flush real phases so authentication and subprocess waits remain observable."""
    print(f"[exp8787] phase={phase} completed={completed} pending={pending}", flush=True)


def cost_operands(value: Json) -> list[Json]:
    """Rejected attribution leaves the whole durable window as an unpartitioned cost."""
    if value["experiment_id"] != 8784:
        return list(value.get("complete_cost_rows", []))
    # Keep all host, restart and storage costs in the denominator, even without leaves.
    return [
        dict(
            operation="durable_window_unpartitioned",
            duration_s=w["complete_ns"] / 1e9,
            parent_duration_s=w["complete_ns"] / 1e9,
            process_id=w["unit_id"],
            parent="complete_ordinary_window",
            clock_domain="parent_perf_counter_ns",
            denominator_updates=w["updates"],
            partition_admitted=False,
            includes_startup_restart_storage_shutdown=True,
        )
        for w in value["per_window_rows"]
    ]


def producer_operands(value: Json, path: Path) -> list[Json]:
    """Use the producer's own sealed replay; a summary scalar is not authentication."""
    modules = {8784: "durable_component_runner_8784"}
    number = value["experiment_id"]
    if number not in modules:
        raise ValueError("unregistered_original_producer:" + str(number))
    module = importlib.import_module("carnot.reporting." + modules[number])
    progress("producer_replay_before", 0, 1)
    module.replay(path)
    progress("producer_replay_after", 1, 0)
    return [
        value["primitive_reference"],
        value["validation_reference"],
        i.reference(Path(value["terminal_validation_sidecar_path"])),
    ]


def admit(root: Path, task: Json, field: str) -> Json:
    """Separate absence, zero and rejection, while accepting an authenticated null."""
    ref = i.reference(root / task["deliverable"])
    check = i.gate(ref, field, 1, task["id"])
    branch: Json = dict(
        task_id=task["id"], field=field, qualified=False, checks=[check], refs=[ref], value={}
    )
    if not check["passed"]:
        return branch
    try:
        value = i.load(ref)
        for key, expected in [
            ("experiment_id", int(task["id"].split("-")[0][3:])),
            ("task_id", task["id"]),
            ("milestone", contract.MILESTONE),
            ("run_date", "20261104"),
            ("required_checks_passed", True),
            ("flagged_adversarial", False),
        ]:
            check = i.gate(ref, key, expected, task["id"])
            branch["checks"].append(check)
            if not check["passed"]:
                raise ValueError("producer_identity_or_execution")
        if not value["invocation_id"] or value["verdict_class"] not in {
            "null",
            "positive",
            "circular_positive",
        }:
            raise ValueError("producer_terminal_verdict")
        branch["refs"] += producer_operands(value, Path(ref["path"])) + i.terminal(ref, value)
        branch.update(qualified=True, value=dict(value, complete_cost_rows=cost_operands(value)))
        branch["checks"].append(
            dict(
                check,
                operand_state="qualified_valid_null"
                if value["verdict_class"] == "null"
                else "qualified",
            )
        )
    except (OSError, ValueError, KeyError, TypeError, ImportError) as error:
        branch["checks"].append(
            dict(
                check,
                passed=False,
                check="original_producer_authentication",
                operand_state="rejected_producer",
                reason=str(error),
                field=check["field"] if not check["passed"] else "original_producer",
                expected=check["expected"] if not check["passed"] else True,
                observed=check["observed"] if not check["passed"] else False,
            )
        )
    return branch


def authenticate(root: Path) -> Json:
    """Read current branches unconditionally; historical board scope stays separately pinned."""
    authority = contract.authority(root)
    refs, failures, prior = list(authority["references"]), [], {}
    ref = i.reference(root / BOARD)
    refs.append(ref)
    try:
        prior = i.load(dict(ref, sha256=BOARD_PIN))
        refs += i.terminal(ref, prior)
        if (
            prior["required_checks_passed"] is not True
            or prior["hardware_mapping_ready_score"] != 1
        ):
            raise ValueError("unqualified_historical_mapping")
    except (OSError, ValueError, KeyError, TypeError) as error:
        failures.append(
            dict(
                upstream="exp8774-hardware-operation-decision",
                **ref,
                check="pinned_historical_mapping",
                field="sha256",
                operator="==",
                expected=BOARD_PIN,
                observed=ref["sha256"],
                passed=False,
                reason=str(error),
                operand_state="missing_file" if ref["sha256"] is None else "rejected_producer",
            )
        )
    board_qualified = not failures
    if not authority["ready"]:
        failures.append(
            dict(
                upstream="V752",
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
    tasks = {
        r["task_object"]["id"].split("-")[0]: r["task_object"]
        for r in authority["rows"]
        if r.get("task_object")
    }
    branches = []
    for task_id, suffix, field in BRANCHES:
        number = task_id.split("-")[0]
        task = tasks.get(
            number,
            dict(id=task_id, deliverable=f"results/experiment_{number[3:]}_v752_{suffix}.json"),
        )
        branches.append(admit(root, task, field))
        progress("independent_admission", len(branches), 3 - len(branches))
    fallback = inference.admit(
        root,
        dict(
            id="exp8757-feature-reuse-cost",
            deliverable="results/experiment_8757_v750_feature_reuse_cost.json",
        ),
        "installed_cost_ready_score",
    )
    refs += [ref for b in [*branches, fallback] for ref in b["refs"]]
    return dict(
        authority=authority,
        refs=refs,
        failures=failures,
        prior=prior,
        board_qualified=board_qualified,
        branches=[branches[0], fallback, branches[1]],
        qualification=branches[2],
        inference_fallback=fallback,
    )


i.authenticate = authenticate
