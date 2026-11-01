"""REQ-REPORT-8707: authenticate original operands without historical controllers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

from carnot.reporting.current_work_receipt import canonical_hash, sha256_file
from carnot.reporting.primary_publication import read_bound_sidecar, validate_primary
from carnot.reporting.roadmap_contract import compare_contract, parse_design

Json = dict[str, Any]
ROOT = Path(__file__).resolve().parents[3]
DESIGN = "openspec/change-proposals/research-roadmap-vNEXT.md"
MILESTONE = "2026.11.746"
PIN = "43132b7a0dad4165413d7bcce82bdc9aa5602d17fc1a5144a960edb9d01c3c97"
PROTOCOL = "openspec/change-proposals/v744-methods-protocol.json"
PINS = {
    PROTOCOL: "842e28d27098b7dadf41290c407e72cfa4019a2cbb09429c492355aa0f55cb67",
    "results/experiment_8440_v726_board_reader_qualification.json": "f1fe951405765fc39318a440f90c3baa2cb34d713f75945de638aa3b74a887bb",
    "results/experiment_8665_v743_hardware_operation_evidence.json": "436a721cdb3430fcc0ff79ee18bd747158356e9b320f93961f6605fad7a5548f",
    "results/experiment_8661_v743_installed_complete_cost.json": "6670350979ad4fdfcaac26996a1a8705d061d2f4d3cf9dc494621c69cafae4d8",
    "results/experiment_8659_v743_complete_source_comparison.json": "55b263c77701f25e828cdc2522e90286f23521c10b386eb45011ab1b7fc71e6f",
    "results/experiment_8679_v744_hardware_operation_boundary.json": "66c07c928fcec102fceff878087e1fbc6330ec2d1f8c34c6bde3712ed0d27f07",
    "results/experiment_8693_v745_hardware_operation_boundary.json": "1edc62c1f83d68185ba712fa181275ab711ba9b7e51a1ac684e8e6d4b1004006",
}
BRANCHES = [
    (8699, "learning_cost_ready_score"),
    (8703, "batch_cost_ready_score"),
    (8706, "lifecycle_rows_ready_score"),
    (8706, "numerical_lifecycle_ready_score"),
]


def reference(path: Path) -> Json:
    """Absence has no digest and must not become a numeric zero."""
    return dict(path=str(path.absolute()), sha256=sha256_file(path) if path.is_file() else None)


def load(ref: Json) -> Json:
    """Authenticate bytes before a parser can expose scores or numerical operands."""
    path = Path(ref["path"])
    if sha256_file(path) != ref["sha256"]:
        raise ValueError("changed_bytes:" + str(path))
    return dict(json.loads(path.read_bytes()))


def gate(ref: Json, field: str, expected: Any, upstream: str) -> Json:
    """Different failures require different evidence before the gate can reopen."""
    observed, state, passed = None, "missing_file", False
    try:
        value = load(ref)
        observed = value.get(field)
        state = (
            "missing_field"
            if field not in value
            else "observed_zero"
            if type(observed) is int and observed == 0
            else "observed"
        )
        passed = type(observed) is type(expected) and observed == expected
    except ValueError:
        state = "changed_bytes"
    except OSError:
        pass
    return dict(
        upstream=upstream,
        check="authenticated_field",
        **ref,
        field=field,
        operator="==",
        expected=expected,
        observed=observed,
        operand_state=state,
        passed=passed,
    )


def authority(root: Path) -> Json:
    """A fixed full-object digest rejects coordinated prompt and YAML substitutions."""
    text = (root / DESIGN).read_text()
    table, tasks = parse_design(text, milestone=MILESTONE)
    active = yaml.safe_load((root / "research-roadmap.yaml").read_bytes())
    comparison = compare_contract(text, active, active, milestone=MILESTONE, first_id=8695)
    envelope = json.loads(
        text.split("<!-- V746_TASK_CONTRACT_START -->", 1)[1]
        .split("```json", 1)[1]
        .split("```", 1)[0]
    )
    if not (
        comparison["passed"]
        and active["tasks"] == tasks
        and canonical_hash(tasks) == "sha256:" + PIN
        and envelope == dict(milestone=MILESTONE, tasks=tasks, canonical_tasks_sha256=PIN)
        and f"Canonical task SHA256: `{PIN}`" in text
        and "<!-- V746_TASK_CONTRACT_END -->" in text
    ):
        raise ValueError("v746_full_object_authority")
    return dict(
        ready=True,
        tasks=tasks,
        visible_table=table,
        canonical_tasks_sha256=PIN,
        comparison=comparison,
    )


def terminal(ref: Json, value: Json) -> list[Json]:
    """Original checked publication bytes bind a producer, including blocked board maps."""
    path = Path(value.get("publication_output", ref["path"]))
    validate_primary(value, path)
    side = path.parent / "raw" / path.stem / "validators" / (ref["sha256"][7:] + ".json")
    report = read_bound_sidecar(path, side)
    if (
        report["primary_path"] != str(path.absolute())
        or report["primary_sha256"] != ref["sha256"]
        or report["report"]["passed"] is not True
    ):
        raise ValueError("producer_terminal_acceptance")
    return [reference(path), reference(side)]


def producer_operands(value: Json, path: Path) -> list[Json]:
    """Rebuild the original lifecycle report; consumer coverage supplies no producer proof."""
    from carnot.reporting.spline_lifecycle_study_8706 import build

    raw = path.parent / "raw" / path.stem
    refs = [
        value["primitive_reference"],
        value["validation_reference"],
        *value["code_config_hashes"],
    ]
    work, validation = load(refs[0]), load(refs[1])
    for folder, field, operand in [
        ("seals", "measurement", refs[0]),
        ("validation_seals", "validation", refs[1]),
    ]:
        seal = reference(raw / folder / (value["invocation_id"] + ".json"))
        if load(seal)[field] != operand:
            raise ValueError("original_independent_seal")
        refs.append(seal)
    refs += work["upstream_refs"] + work["raw_refs"]
    refs += [r for r in work["source_refs"] if "/source_closure/" in r["path"]]
    for index, ref in enumerate(refs):
        load_bytes(ref)
        if (index + 1) % 100 == 0:
            print(
                f"[exp8707] phase=authenticate_lifecycle_operands completed={index + 1} pending={len(refs) - index - 1}",
                flush=True,
            )
    if build(work, validation["receipts"], Path(value["publication_output"])) != value:
        raise ValueError("original_producer_construction")
    return refs


def load_bytes(ref: Json) -> None:
    """Binary checkpoints and SQLite need byte authentication without JSON parsing."""
    if sha256_file(Path(ref["path"])) != ref["sha256"]:
        raise ValueError("changed_bytes:" + ref["path"])


def admit(root: Path, task: Json, field: str) -> Json:
    """One missing or flagged producer cannot veto its independently qualified sibling."""
    path = root / task["deliverable"]
    ref = reference(path)
    row = gate(ref, field, 1, task["id"])
    result: Json = dict(
        task_id=task["id"], field=field, qualified=False, checks=[row], refs=[], value={}
    )
    try:
        if not row["passed"]:
            raise ValueError(row["operand_state"])
        value = load(ref)
        refs = terminal(ref, value)
        for field_name, expected in [
            ("task_id", task["id"]),
            ("required_checks_passed", True),
            ("flagged_adversarial", False),
        ]:
            check = gate(ref, field_name, expected, task["id"])
            result["checks"].append(check)
            if not check["passed"]:
                raise ValueError(field_name)
        if not (
            value["verdict_class"] in {"positive", "circular_positive", "null"}
            and value["acceptance_gates"]
            and all(
                v is True
                or (
                    field == "lifecycle_rows_ready_score"
                    and k == "numerical_criteria"
                    and v is False
                )
                for k, v in value["acceptance_gates"].items()
            )
            and all(f["severity"] == "info" for f in value["adversarial_findings"])
        ):
            raise ValueError("qualified_original_producer_checks")
        refs += producer_operands(value, path)
        result.update(qualified=True, refs=[ref, *refs], value=value)
    except (OSError, ValueError, KeyError, TypeError) as error:
        result["checks"].append(
            dict(row, check="original_producer_authentication", passed=False, reason=str(error))
        )
        skip = (
            root
            / "results"
            / (
                f"experiment_{task['id'].split('-')[0][3:]}_"
                + task["id"].split("-", 1)[1].replace("-", "_")
                + ".json"
            )
        )
        if not path.exists() and skip.is_file():
            result["scheduling_disposition"] = dict(
                reference=reference(skip), science_evidence=False, disposition=load(reference(skip))
            )
            result["refs"].append(reference(skip))
    return result


def authenticate(root: Path) -> Json:
    """Original board authority, registered arithmetic and current operands stay separate."""
    refs, failures, originals = [], [], {}
    for name, digest in PINS.items():
        ref = dict(path=str(root / name), sha256="sha256:" + digest)
        try:
            value = load(ref)
            refs.append(ref)
            if name != PROTOCOL:
                refs += terminal(ref, value)
                if not value["required_checks_passed"] or value["flagged_adversarial"]:
                    raise ValueError("flagged_original_producer")
            if value.get("experiment_id") == 8440:
                original_refs = [value["work_reference"], *value["source_artifact_hashes"]]
                for operand in original_refs:
                    load_bytes(operand)
                refs += original_refs
            originals[name] = value
        except (OSError, ValueError, KeyError, TypeError) as error:
            failures.append(
                dict(
                    upstream=name,
                    check="original_authority",
                    **reference(root / name),
                    field="original_sha256",
                    operator="==",
                    expected="sha256:" + digest,
                    observed=reference(root / name)["sha256"],
                    operand_state="changed_bytes" if (root / name).is_file() else "missing_file",
                    reason=str(error),
                    passed=False,
                )
            )
    method = originals.get(PROTOCOL, {})
    try:
        checkpoint = method["head_checkpoint_reference"]
        heads = load(checkpoint)["heads"]["spline9"]
        update = method["objective_updates"]
        if not (
            len(heads) == 5
            and update["brier_derivative"] == "2*(p-y)*p*(1-p)/T"
            and update["learning_rate"] == 0.01
            and update["l2"] == 0.001
            and update["capacity"] == 8
            and update["seeds"] == [17, 29, 43, 71, 101]
        ):
            raise ValueError("registered_method_values")
        refs.append(checkpoint)
        head_ref = method["original_producers"]["8431"]
        head_producer = load(head_ref)
        refs += [head_ref, *terminal(head_ref, head_producer)]
        for field, expected in [
            ("head_ready_score", 1),
            ("required_checks_passed", True),
            ("flagged_adversarial", False),
        ]:
            check = gate(head_ref, field, expected, "exp8431-criterion-energy-fit")
            if not check["passed"]:
                failures.append(check)
    except (OSError, ValueError, KeyError, TypeError) as error:
        failures.append(
            dict(
                upstream="v744_original_methods",
                check="checkpoint_and_registered_method",
                **reference(root / PROTOCOL),
                field="checkpoint_and_method",
                operator="==",
                expected="original_five_heads_and_registered_Brier",
                observed=str(error),
                operand_state="missing_or_changed_method",
                passed=False,
            )
        )
    from carnot.reporting.hardware_boundary_costs_8679 import source_cells
    from carnot.reporting.hardware_operation_evidence_8665 import installed_cells

    historical_cost_rows = []
    for name, reducer in [
        ("results/experiment_8659_v743_complete_source_comparison.json", source_cells),
        ("results/experiment_8661_v743_installed_complete_cost.json", installed_cells),
    ]:
        if name in originals:
            try:
                historical_cost_rows += [
                    dict(row, historical_only=True, producer=name)
                    for row in reducer(originals[name], refs)
                ]
            except (OSError, ValueError, KeyError, TypeError) as error:
                failures.append(
                    dict(
                        upstream=name,
                        check="original_complete_cost_primitives",
                        **reference(root / name),
                        field="complete_cost_primitives",
                        operator="==",
                        expected="authenticated_original_endpoints",
                        observed=str(error),
                        operand_state="missing_or_changed_cost_primitive",
                        passed=False,
                    )
                )
    contract: Json = {}
    try:
        contract = authority(root)
    except (OSError, ValueError, KeyError, TypeError, IndexError) as error:
        failures.append(
            dict(
                upstream="V746",
                check="full_object_authority",
                **reference(root / DESIGN),
                field="canonical_tasks_sha256",
                operator="==",
                expected=PIN,
                observed=str(error),
                operand_state="missing_or_changed_authority",
                passed=False,
            )
        )
    tasks = {t["id"].split("-")[0]: t for t in contract.get("tasks", [])}
    branches = []
    for number, field in BRANCHES:
        fallback = dict(
            id=f"exp{number}-unavailable",
            deliverable=f"results/unavailable_declared_exp{number}.json",
        )
        branches.append(admit(root, tasks.get(f"exp{number}", fallback), field))
    refs += [r for b in branches for r in b["refs"]]
    prior = originals.get("results/experiment_8693_v745_hardware_operation_boundary.json", {})
    return dict(
        board_qualified=not failures,
        authority=contract,
        branches=branches,
        refs=refs,
        failures=failures,
        prior=prior,
        lifecycle_cost_rows=branches[2]["value"].get("complete_operation_cost_rows", []),
        historical_cost_rows=historical_cost_rows,
        historical_model_provenance=branches[2]["value"].get("historical_model_provenance", []),
    )
