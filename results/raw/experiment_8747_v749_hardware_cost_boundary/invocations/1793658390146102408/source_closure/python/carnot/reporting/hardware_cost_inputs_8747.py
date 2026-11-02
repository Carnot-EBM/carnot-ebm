"""REQ-REPORT-8747: independent producer gates preserve absent complete costs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from carnot.reporting import hardware_operation_inputs_8721 as original
from carnot.reporting import v748_collection_contract as namespaces
from carnot.reporting import v749_contract_methods as contract

Json = dict[str, Any]
i = namespaces.namespace(original, "hardware_cost_inputs_8747")
i.contract, i.MILESTONE = contract, contract.MILESTONE
i.BRANCHES = [
    (8740, "learning_cost_ready_score"),
    (8744, "installed_cost_ready_score"),
    (8745, "bounded_update_cost_ready_score"),
]
i.PRIOR = "results/experiment_8735_v748_hardware_cost_boundary.json"
i.PINS = dict(
    i.PINS, **{i.PRIOR: "d60f2112d6ba55c48675c39a5c38039d42f3d1877b4ae1ad79fd60be7bae7651"}
)
i.CURRENT_PINS = {}
BASE_ADMIT = i.admit


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Flush real evidence counts so storage reads do not look like a stalled task."""
    print(f"[exp8747] phase={phase} completed={completed} pending={pending}", flush=True)


def producer_operands(value: Json, path: Path) -> list[Json]:
    """A producer's own replay must authenticate the costs behind its ready scalar."""
    refs = value["source_artifact_hashes"] + value["code_config_hashes"] + value["raw_shard_hashes"]
    refs += [
        value["primitive_reference"],
        i.reference(Path(value["terminal_validation_sidecar_path"])),
    ]
    refs = [i.versioned_reference(ref, refs) for ref in refs]
    final = i.load(refs[-1])
    if (
        final.get("passed") is not True
        or final["publication"]["primary_sha256"] != i.reference(path)["sha256"]
    ):
        raise ValueError("producer_final_identity")
    progress("producer_replay_before", 0, 1)
    if value["experiment_id"] != 8745:
        raise ValueError("unregistered_model_free_producer_replay_changed_prerequisite")
    from carnot.reporting.bounded_durable_runner_8745 import replay

    replay(path)
    progress("producer_replay_after", 1, 0)
    return list(refs)


def admit(root: Path, task: Json, field: str) -> Json:
    """Report one exact readiness blocker; scheduling dispositions supply no science."""
    branch: Json = BASE_ADMIT(root, task, field)
    if not branch["checks"][0]["passed"]:
        branch["checks"] = branch["checks"][:1]
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
    return branch


def authenticate(root: Path) -> Json:
    """Pinned board evidence and current costs are authenticated on independent paths."""
    authority = contract.authority(root)
    refs, failures, prior = list(authority["references"]), [], {}
    if not authority["ready"]:
        failures.append(
            dict(
                upstream="V749",
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
                # Frozen source copies preserve historical methods when append-only docs grow.
                work = i.load(value["primitive_reference"])
                refs.append(value["primitive_reference"])
                for ref in work["source_refs"]:
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
    tasks = {t["id"].split("-")[0]: t for t in authority["tasks"]}
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
    refs += [ref for b in branches for ref in b["refs"]]
    return dict(
        authority=authority,
        refs=refs,
        failures=failures,
        prior=prior,
        board_qualified=not failures,
        branches=branches,
    )


i.producer_operands, i.admit, i.authenticate = producer_operands, admit, authenticate
