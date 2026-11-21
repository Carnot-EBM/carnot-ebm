"""REQ-REPORT-9269-V789: split V782-V788 replay custody and V789 contract authentication."""

import ast
from copy import deepcopy
import json
from pathlib import Path
from typing import Any
from unittest.mock import patch

import yaml

from carnot.reporting import v788_contract_evidence as prior
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash
from carnot.reporting.historical_contract_inputs import authenticated, original_root
from carnot.reporting.logit_methods_9126 import ref as ref
from carnot.reporting.v749_contract_methods import encode

Json = dict[str, Any]
ROOT = prior.ROOT
NAME, TASK = "experiment_9269_v789_contract", "exp9269-contract"
CLI, TEST = f"scripts/experiments/{NAME}.py", "tests/python/test_v789_contract_9269.py"
CONFIG = "openspec/change-proposals/v789-contract-inputs.json"
FROZEN = json.loads((ROOT / CONFIG).read_bytes())
DESIGN, ACTIVE = "openspec/change-proposals/research-roadmap-vNEXT.md", "research-roadmap.yaml"
COLLISION_INPUTS = prior.COLLISION_INPUTS
OWNED = [
    "python/carnot/reporting/v789_contract_evidence.py",
    "python/carnot/reporting/v789_contract.py",
    "python/carnot/reporting/historical_contract_inputs.py",
    CLI,
]
AFFECTED: list[str] = []
MODEL_SPECS: list[Json] = []
np = prior.np
HISTORICAL_BASE: Path


def progress(phase: str) -> None:
    """Flush boundaries so authentication and supervised validation remain observable."""
    print(f"[exp9269] phase={phase}", flush=True)


def check(path: Path, field: str, expected: Any, observed: Any, name: str) -> Json:
    """Retain strict types so absence, zero and false remain distinct operands."""
    return dict(prior.check(path, field, expected, observed, name), upstream=TASK)


def authority(root: Path) -> Json:
    """Compare complete ordered objects and reject occupied IDs before any activation."""
    tasks, rows, checks = [], [], []
    try:
        text = (root / DESIGN).read_text()
        section = text.split("## Full machine-readable contract", 1)[1]
        machine = (
            yaml.safe_load(section.split("```yaml", 1)[1].split("```", 1)[0])
            if "```yaml" in section
            else {}
        ) or {}
        if not isinstance(machine, dict):
            raise ValueError("embedded_yaml_object")
        machine_tasks = machine.get("tasks") if isinstance(machine.get("tasks"), list) else []
        queue = yaml.safe_load((root / ACTIVE).read_bytes())
        tasks = queue["tasks"]
        table = [
            [c.strip().strip("`") for c in line.strip().strip("|").split("|")]
            for line in text.split("## Task contract", 1)[1]
            .split("## Full machine-readable", 1)[0]
            .splitlines()
            if line.startswith("| ") and line.split("|")[1].strip().isdigit()
        ]
        for name in [DESIGN, ACTIVE, *COLLISION_INPUTS]:
            checks.append(
                check(
                    root / name,
                    "sha256",
                    FROZEN["pins"][name],
                    ref(root / name)["sha256"] or "missing_file",
                    "contract_input",
                )
            )
        checks.append(
            check(
                root / ACTIVE,
                "milestone",
                ["2026.11.789"] * 2,
                [
                    queue.get("milestone", "missing_field"),
                    machine.get("milestone", "missing_field"),
                ],
                "milestone",
            )
        )
        checks.append(
            check(
                root / ACTIVE,
                "task_count",
                [14] * 3,
                list(map(len, [tasks, machine_tasks, table])),
                "task_count",
            )
        )
        for index in range(14):
            if index < len(tasks):
                task = tasks[index]
                if not isinstance(task, dict) or not isinstance(task.get("id"), str):
                    checks.append(
                        check(
                            root / ACTIVE,
                            f"tasks[{index}]",
                            "object_with_string_id",
                            task,
                            "task_object",
                        )
                    )
                    raise ValueError("malformed_task_object")
                original = machine_tasks[index] if index < len(machine_tasks) else {}
                expected = (
                    dict(original, prompt=encode(original["prompt"]))
                    if original
                    else "missing_field"
                )
                checks.append(
                    check(root / ACTIVE, f"tasks[{index}]", expected, task, "ordered_task_object")
                )
                matched = (
                    index < len(machine_tasks)
                    and index < len(table)
                    and task == expected
                    and table[index]
                    == [str(index + 1), str(task["phase"]), task["id"], task["deliverable"]]
                    and task["id"].startswith(f"exp{9269 + index}-")
                )
                rows.append(
                    dict(
                        unit_id=task["id"],
                        group_id=task["id"],
                        qid=None,
                        sibling=None,
                        arm="full_task_contract",
                        condition="exact_authority",
                        seed=FROZEN["seed"],
                        metric="ordered_full_object_match",
                        absolute_metric=int(matched),
                        numerator=int(matched),
                        denominator=1,
                        status="completed" if matched else "failed",
                        exclusion_reason=None,
                        censor_reason=None,
                    )
                )
            else:
                slot_id = f"exp{9269 + index}"
                rows.append(
                    dict(
                        unit_id=slot_id,
                        group_id=slot_id,
                        qid=None,
                        sibling=None,
                        arm="full_task_contract",
                        condition="exact_authority",
                        seed=FROZEN["seed"],
                        metric="ordered_full_object_match",
                        absolute_metric=None,
                        numerator=None,
                        denominator=0,
                        status="censored",
                        exclusion_reason=None,
                        censor_reason="missing_task_in_active_queue",
                    )
                )
                checks.append(
                    check(root / ACTIVE, f"tasks[{index}]", slot_id, "missing_field", "task_slot")
                )
        checks.append(
            check(
                root / ACTIVE,
                "all_complete_objects",
                True,
                len(rows) == 14 and all(r["numerator"] == 1 for r in rows),
                "full_contract",
            )
        )
        archive = yaml.safe_load((root / COLLISION_INPUTS[0]).read_bytes())
        historical = [t["id"] for m in archive["milestones"] for t in m.get("tasks", [])]
        manifest = (root / COLLISION_INPUTS[1]).read_text()
        current = yaml.safe_load((root / "research-roadmap.yaml").read_bytes())
        for task in tasks:
            number = task["id"].split("-", 1)[0].removeprefix("exp")
            collisions = [s for s in historical if s.split("-", 1)[0] == "exp" + number]
            collisions += [
                str(p.relative_to(root))
                for p in (root / "results").glob(f"experiment_{number}_*.json")
                if p != root / "results" / (NAME + ".json")
            ]
            collisions += (
                ["exclusion_manifest"]
                if any(s in manifest for s in [task["id"], "experiment_id: " + number + "\n"])
                else []
            )
            collisions += (
                ["current_queue"]
                if any(
                    t["id"].split("-", 1)[0] == "exp" + number and t != task
                    for t in current["tasks"]
                )
                else []
            )
            checks.append(
                check(
                    root / COLLISION_INPUTS[0],
                    task["id"] + ".collisions",
                    [],
                    collisions,
                    "id_availability",
                )
            )
    except (OSError, ValueError, KeyError, TypeError, IndexError, yaml.YAMLError) as error:
        observed = (
            "missing_file"
            if isinstance(error, FileNotFoundError)
            else "missing_field"
            if isinstance(error, KeyError)
            else str(error)
        )
        checks.append(
            check(
                Path(error.filename) if isinstance(error, FileNotFoundError) else root / ACTIVE,
                "full_task_objects",
                "14 exact ordered objects",
                observed,
                "authority",
            )
        )
    return dict(
        ready=bool(checks) and all(c["passed"] for c in checks),
        tasks=tasks,
        rows=rows,
        checks=checks,
    )


TABLES = ["validation_scope_manifest", "validation_obligations", "historical_input_roots"]


def compact_table(raw: Path, name: str, payload: Any) -> Json:
    """Keep complete tables authenticated without repeating their bytes in every primary."""
    path = raw / (name + ".json")
    atomic_json(path, payload)
    summary = dict(ref(path), count=len(payload), current_qualification_credit=0)
    if isinstance(payload, dict):
        summary["unit_counts"] = {k: len(v) for k, v in payload.items() if isinstance(v, list)}
        summary["union_count"] = payload.get("union_count", 0)
    return summary


def _inspect(root: Path) -> Json:
    """Replay original checks under isolated V788 authority; do not import its bulky arrays."""
    from carnot.reporting.historical_contract_inputs import load_compact_primary

    progress("authenticate_inputs_before")
    contract = authority(root)
    with patch.object(prior, "HISTORICAL_BASE", HISTORICAL_BASE, create=True):
        historical_root = (
            original_root(HISTORICAL_BASE / "v788", 788) if root.is_dir() else root / "absent"
        )
        inherited = prior.inspect(historical_root)
    inherited_checks = [c for c in inherited["checks"] if c not in inherited["contract"]["checks"]]
    checks = list(contract["checks"]) + inherited_checks
    values = {}
    references = []
    for name, pin in FROZEN["pins"].items():
        path = root / name
        row = check(path, "sha256", pin, ref(path)["sha256"] or "missing_file", "immutable_operand")
        checks.append(row)
        references.append(
            dict(
                path=str(path),
                sha256=pin,
                producer_path=str(path),
                imported_fields=["original_authority"],
            )
        )
        if row["passed"] and name.startswith("results/experiment_92"):
            values[Path(name).name.split("_")[1]] = load_compact_primary(path)
    outcomes = {}
    fields = {
        "9255": dict(contract_ready_score=1, replay_authority_ready_score=1),
        "9256": dict(corpus_scan_ready_score=0),
        "9258": dict(victris_harness_ready_score=1),
        "9268": dict(reconciliation_ready_score=1),
    }
    for number, operands in fields.items():
        value = values.get(number, {})
        outcomes[number] = dict(honest_verdict=value.get("honest_verdict", "missing_file"))
        for field, expected in operands.items():
            observed = value.get(field, "missing_field" if value else "missing_file")
            path = next(
                root / p for p in FROZEN["pins"] if p.startswith(f"results/experiment_{number}_")
            )
            checks.append(check(path, field, expected, observed, "v788_historical_outcome"))
            outcomes[number][field] = observed
        if value.get("work_reference"):
            authenticated(value["work_reference"], read=False)
            references.append(dict(value["work_reference"], imported_fields=["historical_work"]))
    outcomes["9257"] = dict(
        honest_verdict=values.get("9257", {}).get("honest_verdict", "missing_file")
    )
    victris = values.get("9258", {})
    for reference in victris.get("source_references", []):
        authenticated(reference, read=False)
        references.append(dict(reference, imported_fields=["victris_harness"]))
    inherited.update(contract=contract, checks=checks, preserved_v788=outcomes)
    inherited["source_references"] = references + inherited["source_references"]
    roots = list(inherited["historical_input_roots"])
    if historical_root.is_dir():
        roots.append(
            dict(
                root=str(historical_root),
                **json.loads((historical_root / "historical_inputs.json").read_bytes()),
            )
        )
    obligations = inherited["validation_obligations"]
    paths = [
        *OWNED,
        TEST,
        CONFIG,
        "tests/python/test_v788_contract_9255.py",
        "tests/python/test_v788_capstone_9268.py",
        "tests/python/test_victris_energy_bot_9258.py",
        "tests/python/test_v788_corpus_prefilter_9256.py",
        "tests/python/test_causal_validation_9181.py",
    ]
    obligations["affected_v789_paths"] = paths
    obligations["affected_v789_code_test_hashes"] = [ref(ROOT / p) for p in paths]
    obligations["affected_v789_node_assertions"] = [
        dict(
            node_id=p + "::" + node.name,
            sha256=canonical_hash(ast.dump(node, include_attributes=False)),
            producer_reference=ref(ROOT / p),
            current_qualification_credit=0,
        )
        for p in paths
        if p.startswith("tests/")
        for node in ast.parse((ROOT / p).read_bytes()).body
        if isinstance(node, ast.FunctionDef) and node.name.startswith("test_")
    ]
    obligations["required_by"] = ["exp9270-corpus-prefilter", "exp9271-forward-qualification"]
    raw = HISTORICAL_BASE.parent
    inherited["validation_obligations"] = compact_table(raw, "validation_obligations", obligations)
    summary = compact_table(raw, "historical_input_roots", roots)
    summary["suites"] = [
        dict(generation=r["generation"], suite=r["suite"], root=r["root"], count=len(r["rows"]))
        for r in roots
    ]
    inherited["historical_input_roots"] = summary
    non_contract = all(r["passed"] for r in checks if r not in contract["checks"])
    inherited["replay_ready"] = bool(non_contract and len(roots) == 9)
    for key in ["source_ready", "static_ready", "online_ready"]:
        inherited[key] = bool(inherited[key] and non_contract)
    inherited["protocol_references"]["victris_harness_reference"] = next(
        r for r in references if r["path"].endswith("experiment_9258_v788_victris_energy_bot.json")
    )
    progress("authenticate_inputs_after")
    return dict(inherited)


def inspect(root: Path) -> Json:
    """An unavailable external producer is blocked with its exact failed operand."""
    try:
        return _inspect(root)
    except (OSError, ValueError, KeyError) as error:
        result = _inspect(root / "absent")
        name = str(error).partition("historical_input_drift:")[2]
        path = Path(name) if name else root
        observed = (
            "missing_field"
            if isinstance(error, KeyError)
            else ref(path)["sha256"] or "missing_file"
        )
        result["checks"].append(
            check(
                path,
                str(error.args[0]),
                "authenticated_original_bytes",
                observed,
                "external_authority",
            )
        )
        return result
