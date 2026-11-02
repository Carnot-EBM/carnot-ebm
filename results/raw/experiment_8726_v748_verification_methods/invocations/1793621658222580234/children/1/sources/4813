"""REQ-REPORT-8721: original producer evidence must precede numerical admission."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from carnot.reporting import hardware_operation_inputs_8707 as previous
from carnot.reporting import v747_contract_methods as contract
from carnot.reporting.current_work_receipt import sha256_file

Json = dict[str, Any]
ROOT, DESIGN, MILESTONE = previous.ROOT, contract.DESIGN, contract.MILESTONE
reference, load, gate, terminal = (
    previous.reference,
    previous.load,
    previous.gate,
    previous.terminal,
)
BRANCHES = [
    (8713, "learning_cost_ready_score"),
    (8717, "batch_cost_ready_score"),
    (8719, "native_update_ready_score"),
]
PRIOR = "results/experiment_8707_v746_hardware_operation_boundary.json"
PINS = dict(
    previous.PINS, **{PRIOR: "21802928248cb86fe4dcc2cdb4ea6ad8bf403d0884d3c4e049f13e8906f53cb6"}
)
CURRENT_PINS = {
    8717: "938d2cd6a28f8b0ae1ee1a551199a8ab7c3b29de91dca12f2d83aa8d1379fef6",
    8719: "9d27b1f8ca99b50d0452d7e1d204c18223a16b7eec36208d8bc5c44830c6f1ac",
}


def load_bytes(ref: Json) -> None:
    """An absent input is bound as absence so later appearance also invalidates replay."""
    if ref["sha256"] is None:
        if Path(ref["path"]).is_file():
            raise ValueError("changed_absence:" + ref["path"])
    elif sha256_file(Path(ref["path"])) != ref["sha256"]:
        raise ValueError("changed_bytes:" + ref["path"])


def producer_operands(value: Json, path: Path) -> list[Json]:
    """Rebuild original reports without rerunning benchmarks or historical controllers."""
    refs = value["source_artifact_hashes"] + value["code_config_hashes"] + value["raw_shard_hashes"]
    refs = [
        *refs,
        value["primitive_reference"],
        reference(Path(value["terminal_validation_sidecar_path"])),
    ]
    refs = [versioned_reference(ref, refs) for ref in refs]
    final = load(refs[-1])
    if (
        final.get("passed", True) is not True
        or final["publication"]["primary_sha256"] != reference(path)["sha256"]
    ):
        raise ValueError("producer_final_identity")
    if value["experiment_id"] == 8717:
        from carnot.reporting.installed_batch_cost_runner_8717 import replay

        if not replay(path, value["primitive_reference"]):
            raise ValueError("original_batch_construction")
    elif value["experiment_id"] == 8719:
        from carnot.reporting.compiled_sparse_study_8719 import build

        work, validation = load(value["primitive_reference"]), load(value["validation_reference"])
        anchor = path.parent / "raw" / path.stem
        for folder, field, operand in [
            ("seals", "measurement", value["primitive_reference"]),
            ("validation_seals", "validation", value["validation_reference"]),
        ]:
            seal = reference(anchor / folder / (value["invocation_id"] + ".json"))
            if load(seal)[field] != operand:
                raise ValueError("original_independent_seal")
            refs.append(seal)
        if (
            validation["receipts"] != value["validation_receipts"]
            or not all(r["passed"] for r in validation["receipts"])
            or work["private_control"]
            or build(work, validation["receipts"], path) != value
        ):
            raise ValueError("original_native_construction")
        refs.append(value["validation_reference"])
    else:
        raise ValueError("unregistered_learning_producer_identity")
    return refs


def versioned_reference(ref: Json, refs: list[Json]) -> Json:
    """Frozen documents bind historical methods; executable sources still require identity."""
    path = Path(ref["path"])
    if path.suffix == ".md" and reference(path)["sha256"] != ref["sha256"]:
        frozen = next(
            (r for r in refs if "/source_closure/" in r["path"] and r["sha256"] == ref["sha256"]),
            ref,
        )
        load_bytes(frozen)
        return dict(frozen)
    load_bytes(ref)
    return ref


def admit(root: Path, task: Json, field: str) -> Json:
    """Read exactly the declared deliverable; no sibling readiness bit can veto this one."""
    path = root / task["deliverable"]
    ref = reference(path)
    row = gate(ref, field, 1, task["id"])
    result: Json = dict(
        task_id=task["id"], field=field, qualified=False, checks=[row], refs=[ref], value={}
    )
    try:
        if not row["passed"]:
            raise ValueError(row["operand_state"])
        value = load(ref)
        number = value["experiment_id"]
        if number in CURRENT_PINS and ref["sha256"] != "sha256:" + CURRENT_PINS[number]:
            raise ValueError("versioned_original_primary")
        result["refs"] += terminal(ref, value)
        for key, expected in [
            ("task_id", task["id"]),
            ("required_checks_passed", True),
            ("flagged_adversarial", False),
        ]:
            check = gate(ref, key, expected, task["id"])
            result["checks"].append(check)
            if not check["passed"]:
                raise ValueError(key)
        gates = value["acceptance_gates"]
        passed = list(gates.values()) if isinstance(gates, dict) else [g["passed"] for g in gates]
        if (
            not passed
            or not all(v is True for v in passed)
            or value["verdict_class"] not in {"positive", "circular_positive", "null"}
            or any(f["severity"] != "info" for f in value["adversarial_findings"])
        ):
            raise ValueError("qualified_original_producer_checks")
        result["refs"] += producer_operands(value, path)
        result.update(qualified=True, value=value)
    except (OSError, ValueError, KeyError, TypeError) as error:
        result["checks"].append(
            dict(
                row,
                check="original_producer_authentication",
                passed=False,
                reason=str(error),
                field="qualified_original_producer",
                expected=True,
                observed=False,
            )
        )
    return result


def authenticate(root: Path) -> Json:
    """Pinned historical scope survives absent current learning or optional hardware."""
    authority = contract.authority(root)
    refs, failures, originals = (
        list(authority["references"]),
        list(authority["gate_check_summary"]),
        {},
    )
    for name, digest in PINS.items():
        ref = dict(path=str(root / name), sha256="sha256:" + digest)
        try:
            value = load(ref)
            refs.append(ref)
            if name != previous.PROTOCOL:
                refs += terminal(ref, value)
                if (
                    value["required_checks_passed"] is not True
                    or value["flagged_adversarial"] is not False
                ):
                    raise ValueError("flagged_original_producer")
            if name == previous.PROTOCOL:
                refs.append(value["head_checkpoint_reference"])
                load_bytes(refs[-1])
            if value.get("experiment_id") == 8440:
                for operand in [value["work_reference"], *value["source_artifact_hashes"]]:
                    load_bytes(operand)
                    refs.append(operand)
            originals[name] = value
        except (OSError, ValueError, KeyError, TypeError) as error:
            failures.append(
                dict(
                    upstream=name,
                    **reference(root / name),
                    check="versioned_board_authority",
                    field="original_sha256",
                    operator="==",
                    expected=ref["sha256"],
                    observed=reference(root / name)["sha256"],
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
    branches = [
        admit(
            root,
            tasks.get(
                f"exp{number}",
                dict(
                    id=f"exp{number}-unavailable",
                    deliverable=f"results/unavailable_declared_exp{number}.json",
                ),
            ),
            field,
        )
        for number, field in BRANCHES
    ]
    refs += [r for b in branches for r in b["refs"]]
    return dict(
        authority=authority,
        board_qualified=not failures,
        failures=failures,
        refs=refs,
        branches=branches,
        prior=originals.get(PRIOR, {}),
    )
