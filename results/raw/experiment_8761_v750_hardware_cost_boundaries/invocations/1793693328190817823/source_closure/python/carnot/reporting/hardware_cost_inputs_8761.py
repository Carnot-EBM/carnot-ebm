"""REQ-VERIFY-8761: each cost producer owns its admission and byte custody."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from carnot.reporting import hardware_operation_inputs_8721 as original
from carnot.reporting import hardware_cost_inputs_8747 as history
from carnot.reporting import v750_archive_bound as contract
from carnot.reporting.v748_collection_contract import namespace

Json = dict[str, Any]
i = namespace(original, "cost_inputs_8761")
i.contract, i.MILESTONE = contract, contract.MILESTONE
i.BRANCHES = [
    (8752, "learning_cost_ready_score"),
    (8757, "installed_cost_ready_score"),
    (8759, "ordinary_update_cost_ready_score"),
]
i.PINS, i.PRIOR = dict(history.i.PINS), history.i.PRIOR
i.CURRENT_PINS = {8757: "668218f9dffdaafa2a886789af61b26b9ab2f9743f56cf05d2092e610a144258"}
BASE_ADMIT = i.admit
HISTORICAL = "results/experiment_8747_v749_hardware_cost_boundary.json"


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Flush each phase so long authentication and validation stay observable."""
    print(f"[exp8761] phase={phase} completed={completed} pending={pending}", flush=True)


def producer_operands(value: Json, path: Path) -> list[Json]:
    """Recompute registered producer costs rather than trusting a readiness scalar."""
    refs = value["source_artifact_hashes"] + value["code_config_hashes"] + value["raw_shard_hashes"]
    refs += [
        value["primitive_reference"],
        i.reference(Path(value["terminal_validation_sidecar_path"])),
    ]
    refs = [i.versioned_reference(ref, refs) for ref in refs]
    final = i.load(refs[-1])
    if final["publication"]["primary_sha256"] != i.reference(path)["sha256"]:
        raise ValueError("producer_final_identity")
    if value["experiment_id"] != 8757:
        raise ValueError("unregistered_producer_replay_changed_prerequisite")
    from carnot.reporting.feature_reuse_cost_runner_8757 import replay

    progress("producer_replay_before", 0, 1)
    if not replay(path, value["primitive_reference"]):
        raise ValueError("original_installed_cost_construction")
    progress("producer_replay_after", 1, 0)
    return list(refs)


def admit(root: Path, task: Json, field: str) -> Json:
    """Keep missing files, missing fields, zero and rejected producers distinct."""
    branch: Json = BASE_ADMIT(root, task, field)
    if not branch["checks"][0]["passed"]:
        branch["checks"] = branch["checks"][:1]
    elif not branch["qualified"]:
        branch["checks"][-1]["operand_state"] = "rejected_producer"
    number = task["id"].split("-")[0][3:]
    skip = (
        root
        / "results"
        / f"experiment_{number}_{task['id'].split('-', 1)[1].replace('-', '_')}.json"
    )
    if branch["checks"][0]["operand_state"] == "missing_file" and skip.is_file():
        ref = i.reference(skip)
        branch["scheduling_disposition"] = dict(
            reference=ref, science_evidence=False, disposition=i.load(ref)
        )
        branch["refs"].append(ref)
    if branch["qualified"] and branch["value"]["experiment_id"] == 8757:
        work = i.load(branch["value"]["primitive_reference"])
        measured = work["measurement"]
        parents = {r["unit_id"]: r for r in measured["process_rows"]}
        branch["value"] = dict(
            branch["value"],
            complete_cost_rows=[
                dict(
                    row,
                    parent="complete_process",
                    clock_domain="parent_monotonic_ns",
                    parent_duration_s=parents[row["process_id"]]["duration_s"],
                    parent_reference=parents[row["process_id"]]["session_reference"],
                )
                for row in measured["complete_cost_rows"]
            ],
        )
    return branch


def authenticate(root: Path) -> Json:
    """Read three branches independently, preserving historical disqualification."""
    authority = contract.authority(root)
    refs, failures, prior = list(authority["references"]), list(authority["gate_check_summary"]), {}
    for name, digest in i.PINS.items():
        expected = dict(path=str(root / name), sha256="sha256:" + digest)
        observed = i.reference(root / name)
        refs.append(observed)
        try:
            value = i.load(expected)
            if name != i.previous.PROTOCOL:
                refs += i.terminal(expected, value)
                if (
                    value["required_checks_passed"] is not True
                    or value["flagged_adversarial"] is not False
                ):
                    raise ValueError("flagged_original_board_evidence")
            if name == i.PRIOR:
                prior = value
                refs.append(value["primitive_reference"])
                for ref in i.load(value["primitive_reference"])["source_refs"]:
                    if "/source_closure/" in ref["path"]:
                        i.load_bytes(ref)
                        refs.append(ref)
        except (OSError, ValueError, KeyError, TypeError) as error:
            failures.append(
                dict(
                    upstream=name,
                    **observed,
                    check="versioned_board_authority",
                    field="original_sha256",
                    operator="==",
                    expected=expected["sha256"],
                    observed=observed["sha256"],
                    passed=False,
                    operand_state="changed_bytes" if (root / name).is_file() else "missing_file",
                    reason=str(error),
                )
            )
    tasks = {
        r["task_object"]["id"].split("-")[0]: r["task_object"]
        for r in authority["rows"]
        if r.get("task_object")
    }
    branches = []
    for number, field in i.BRANCHES:
        task = tasks.get(
            f"exp{number}",
            dict(
                id=f"exp{number}-unavailable",
                deliverable=f"results/unavailable_declared_exp{number}.json",
            ),
        )
        branches.append(admit(root, task, field))
        progress("independent_source_admission", len(branches), 3 - len(branches))
    refs += [ref for branch in branches for ref in branch["refs"]]
    historical = i.reference(root / HISTORICAL)
    refs.append(historical)
    retained = dict(reference=historical, promoted=False, verdict_class=None)
    if historical["sha256"] is not None:
        value = i.load(historical)
        retained.update(
            verdict_class=value.get("verdict_class"),
            required_checks_passed=value.get("required_checks_passed"),
            flagged_adversarial=value.get("flagged_adversarial"),
        )
    return dict(
        authority=authority,
        refs=refs,
        failures=failures,
        prior=prior,
        board_qualified=not failures,
        branches=branches,
        historical_disqualified_producers=[retained],
    )


i.producer_operands, i.admit, i.authenticate = producer_operands, admit, authenticate
