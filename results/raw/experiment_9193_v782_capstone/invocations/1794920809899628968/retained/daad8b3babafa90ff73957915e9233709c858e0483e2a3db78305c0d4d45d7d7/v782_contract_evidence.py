"""REQ-REPORT-9180-V782: immutable prior authority remains separate from current readiness."""

import ast
from copy import deepcopy
import json
from pathlib import Path
from typing import Any
from unittest.mock import patch

import yaml

from carnot.reporting import v781_contract_evidence as prior
from carnot.reporting.v781_contract_evidence import np as np
from carnot.reporting.current_work_receipt import canonical_hash
from carnot.reporting.logit_methods_9126 import ref as ref
from carnot.reporting.v749_contract_methods import encode as encode

Json = dict[str, Any]
ROOT = prior.ROOT
NAME, TASK = "experiment_9180_v782_contract", "exp9180-contract"
CLI, TEST = f"scripts/experiments/{NAME}.py", "tests/python/test_v782_contract_9180.py"
CONFIG = "openspec/change-proposals/v782-contract-inputs.json"
FROZEN = json.loads((ROOT / CONFIG).read_bytes())
DESIGN, ACTIVE = prior.DESIGN, prior.ACTIVE
ARCHIVE = "openspec/change-proposals/research-roadmap-v781-preserved-20261117.md"
QUEUE = "openspec/change-proposals/research-roadmap-v781-queue-preserved-20261117.yaml"
OWNED = [
    "python/carnot/reporting/v782_contract_evidence.py",
    "python/carnot/reporting/v782_contract.py",
    CLI,
]
MODEL_SPECS: list[Json] = []


def progress(phase: str) -> None:
    """Flush phase boundaries so a missing input cannot look like a stalled model."""
    print(f"[exp9180] phase={phase}", flush=True)


def check(path: Path, field: str, expected: Any, observed: Any, name: str) -> Json:
    """Keep strict gate operands, including absence versus observed numeric zero."""
    row = dict(prior.check(path, field, expected, observed, name))
    if row["upstream"] == prior.TASK:
        row["upstream"] = TASK
    return row


def authority(root: Path) -> Json:
    """Compare complete ordered objects and rendered prompts without rewriting either authority."""
    tasks, rows, checks = [], [], []
    try:
        text = (root / DESIGN).read_text()
        machine = yaml.safe_load(
            text.split("## Full machine-readable contract", 1)[1]
            .split("```yaml", 1)[1]
            .split("```", 1)[0]
        )
        queue = yaml.safe_load((root / ACTIVE).read_bytes())
        tasks = queue["tasks"]
        table = [
            [c.strip().strip("`") for c in line.strip().strip("|").split("|")]
            for line in text.split("## Task contract", 1)[1]
            .split("## Full machine-readable", 1)[0]
            .splitlines()
            if line.startswith("| ") and line.split("|")[1].strip().isdigit()
        ]
        checks.extend(
            check(root / p, "sha256", FROZEN["pins"][p], ref(root / p)["sha256"], "contract_bytes")
            for p in [DESIGN, ACTIVE]
        )
        checks.append(
            check(
                root / ACTIVE,
                "milestone",
                ["2026.11.782"] * 2,
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
                list(map(len, [tasks, machine["tasks"], table])),
                "task_count",
            )
        )
        for index, task in enumerate(tasks):
            original = machine["tasks"][index]
            matched = (
                task in (original, dict(original, prompt=encode(original["prompt"])))
                and task["prompt"].format(project_root="/workspace", date="20261117")
                == original["prompt"]
                .replace("{project_root}", "/workspace")
                .replace("{date}", "20261117")
                .replace("{{", "{")
                .replace("}}", "}")
                and table[index]
                == [str(index + 1), task["id"], str(task["phase"]), task["deliverable"]]
                and task["id"].startswith(f"exp{9180 + index}-")
            )
            rows.append(
                dict(
                    unit_id=task["id"],
                    qid=None,
                    group_id=task["id"],
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
        checks.append(
            check(
                root / ACTIVE,
                "all_complete_objects",
                True,
                len(rows) == 14 and all(r["numerator"] for r in rows),
                "full_contract",
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
                root / ACTIVE,
                "full_task_objects",
                "14 exact ordered objects",
                observed,
                "authority",
            )
        )
    return dict(ready=all(c["passed"] for c in checks), tasks=tasks, rows=rows, checks=checks)


def inspect(root: Path) -> Json:
    """Authenticate every producer before importing its original protocol or failed outcome."""
    progress("authenticate_inputs_before")
    result: Json = dict(
        contract=authority(root),
        checks=[],
        source_references=[],
        source_ready=False,
        static_ready=False,
        online_ready=False,
        protocol_references={},
        source={},
        static_protocol={},
        online_protocol={},
        previous_outcome_rows=[],
        archive_lag={},
        retired_qualification={},
        method_receipt_replays={},
        execution_rosters=[],
        partition_union_equals_original=False,
        diagnostic_protocol={},
        retired_schedule={},
        validation_obligations={},
    )
    result["checks"].extend(result["contract"]["checks"])
    values: Json = {}
    for name, pin in FROZEN["pins"].items():
        path = root / name
        gate = check(path, "sha256", pin, ref(path)["sha256"] or "missing_file", "v782_operand")
        result["checks"].append(gate)
        result["source_references"].append(
            dict(
                path=str(path),
                producer_path=str(path),
                sha256=pin,
                imported_fields=["original_authority", "readiness", "rows"],
            )
        )
        if gate["passed"] and name.startswith("results/"):
            values[Path(name).name.split("_")[1]] = json.loads(path.read_bytes())
    result.update(
        entropy_reference_protocol=deepcopy(FROZEN["entropy_reference_protocol"]), preserved_v781={}
    )
    for number, fields in [
        (
            9166,
            dict(
                contract_ready_score=1,
                source_ready_score=1,
                static_methods_ready_score=1,
                online_methods_ready_score=1,
                required_checks_passed=True,
            ),
        ),
        (9167, dict(causal_fix_ready_score=0, required_checks_passed=False)),
        (9179, dict(reconciliation_ready_score=1, required_checks_passed=True)),
    ]:
        path = root / next(
            p for p in FROZEN["pins"] if p.startswith(f"results/experiment_{number}_")
        )
        for field, expected in fields.items():
            result["checks"].append(
                check(
                    path,
                    field,
                    expected,
                    values.get(str(number), {}).get(
                        field, "missing_field" if str(number) in values else "missing_file"
                    ),
                    "historical_gate",
                )
            )
    if not all(c["passed"] for c in result["checks"]):
        result.update(source_ready=False, static_ready=False, online_ready=False)
        return result
    source, diagnosis, capstone = [values[str(n)] for n in [9166, 9167, 9179]]
    for owner in [source, diagnosis, capstone]:
        for reference in [
            owner["work_reference"],
            *(r for r in owner.get("current_work_receipts", []) if "path" in r),
            *owner.get("protocol_references", {}).values(),
        ]:
            path = Path(reference["path"])
            result["checks"].append(
                check(
                    path,
                    "sha256",
                    reference["sha256"],
                    ref(path)["sha256"] or "missing_file",
                    "original_producer_binding",
                )
            )
            result["source_references"].append(
                dict(
                    reference,
                    producer_path=str(path),
                    imported_fields=["original_producer_snapshot"],
                )
            )
    if not all(c["passed"] for c in result["checks"]):
        result.update(source_ready=False, static_ready=False, online_ready=False)
        return result
    saved = json.loads(Path(capstone["work_reference"]["path"]).read_bytes())
    for reference in [*saved["snapshots"].values(), *saved["operands"].values()]:
        path = Path(reference["path"])
        result["checks"].append(
            check(
                path,
                "sha256",
                reference["sha256"],
                ref(path)["sha256"] or "missing_file",
                "scheduler_and_producer_snapshot",
            )
        )
        result["source_references"].append(
            dict(
                reference,
                producer_path=reference.get("original_path", str(path)),
                imported_fields=["scheduler_or_producer_evidence"],
            )
        )
    if not all(c["passed"] for c in result["checks"]):
        return result
    frozen = deepcopy(prior.FROZEN)
    frozen["pins"][ARCHIVE] = frozen["pins"].pop(DESIGN)
    frozen["pins"][QUEUE] = frozen["pins"].pop(ACTIVE)
    with patch.multiple(prior, DESIGN=ARCHIVE, ACTIVE=QUEUE, FROZEN=frozen):
        inherited = prior.inspect(root)
    result = dict(
        inherited,
        contract=result["contract"],
        checks=inherited["checks"] + result["checks"],
        source_references=inherited["source_references"] + result["source_references"],
        entropy_reference_protocol=result["entropy_reference_protocol"],
        preserved_v781={},
    )
    if not all(c["passed"] for c in result["checks"]):
        return result
    result["previous_outcome_rows"] = deepcopy(capstone["task_outcome_rows"])
    result["archive_lag"] = deepcopy(capstone["archive_lag"])
    result["preserved_v781"] = dict(
        design_reference=ref(root / ARCHIVE),
        queue_reference=ref(root / QUEUE),
        design_tasks=json.loads(
            (root / ARCHIVE)
            .read_text()
            .split("## Full machine-readable contract", 1)[1]
            .split("```json", 1)[1]
            .split("```", 1)[0]
        ),
        queue_tasks=yaml.safe_load((root / QUEUE).read_bytes())["tasks"],
    )
    obligations = result["validation_obligations"]
    assertion_sources = {
        r["assertion_reference"]["path"]: r["assertion_reference"]
        for r in obligations["node_bindings"]
    }
    functions = {}
    for name, reference in assertion_sources.items():
        path = Path(name)
        gate = check(
            path,
            "sha256",
            reference["sha256"],
            ref(path)["sha256"] or "missing_file",
            "original_assertions",
        )
        result["checks"].append(gate)
        if gate["passed"]:
            functions[name] = {
                n.name: canonical_hash(ast.dump(n, include_attributes=False))
                for n in ast.walk(ast.parse(path.read_bytes()))
                if isinstance(n, ast.FunctionDef)
            }
    for row in obligations["node_bindings"]:
        row["assertion_hash"] = functions.get(row["assertion_reference"]["path"], {}).get(
            row["unit_id"].split("::")[-1].split("[")[0], "missing_field"
        )
    obligations["affected_node_assertions"] = deepcopy(
        capstone["validation_obligations"]["affected_assertions"]
    )
    arc = json.loads(
        Path(
            saved["snapshots"]["results/experiment_9178_v781_arc_mechanic_routing.json"]["path"]
        ).read_bytes()
    )
    for reference in [
        r
        for owner in [source, diagnosis, capstone, arc]
        for r in owner["code_config_hashes"]
        if Path(r["path"]).name.startswith("test_")
    ]:
        path = Path(reference["path"])
        gate = check(
            path,
            "sha256",
            reference["sha256"],
            ref(path)["sha256"] or "missing_file",
            "affected_assertions",
        )
        result["checks"].append(gate)
        if gate["passed"]:
            obligations["affected_node_assertions"].extend(
                dict(
                    node_id="tests/python/" + path.name + "::" + node.name,
                    sha256=canonical_hash(ast.dump(node, include_attributes=False)),
                    producer_reference=reference,
                    current_qualification_credit=0,
                )
                for node in ast.parse(path.read_bytes()).body
                if isinstance(node, ast.FunctionDef) and node.name.startswith("test_")
            )
    for node in ast.parse((ROOT / TEST).read_bytes()).body:
        if isinstance(node, ast.FunctionDef) and node.name.startswith("test_"):
            obligations["affected_node_assertions"].append(
                dict(
                    node_id=TEST + "::" + node.name,
                    sha256=canonical_hash(ast.dump(node, include_attributes=False)),
                    producer_reference=ref(ROOT / TEST),
                    current_qualification_credit=0,
                )
            )
    obligations["affected_node_assertions"] = list(
        {r["node_id"]: r for r in obligations["affected_node_assertions"]}.values()
    )
    obligations["historical_v781_rows"] = deepcopy(diagnosis["rows"])
    obligations["remaining_v781_obligations"] = deepcopy(diagnosis["remaining_obligations"])
    arc = json.loads(
        Path(
            saved["snapshots"]["results/experiment_9178_v781_arc_mechanic_routing.json"]["path"]
        ).read_bytes()
    )
    log = next(r for r in arc["validation_receipts"] if r["name"] == "direct_consumers")
    path = Path(log["stdout_path"])
    result["checks"].append(
        check(
            path,
            "sha256",
            log["stdout_sha256"],
            ref(path)["sha256"] or "missing_file",
            "original_arc_failure_log",
        )
    )
    obligations["arc_regressions"] = [
        dict(
            node_id=line.split(" - ", 1)[0].removeprefix("FAILED "),
            status="failed",
            current_qualification_credit=0,
            log_reference=ref(path),
        )
        for line in (path.read_text().splitlines() if result["checks"][-1]["passed"] else [])
        if line.startswith("FAILED ")
    ]
    assertions = {r["node_id"]: r for r in obligations["affected_node_assertions"]}
    for row in obligations["arc_regressions"]:
        row["assertion_hash"] = assertions.get(row["node_id"], {}).get("sha256", "missing_field")
    obligations["affected_current_paths"] = [*OWNED, TEST, CONFIG]
    result["source_ready"] = bool(
        result["source_ready"] and all(c["passed"] for c in result["checks"])
    )
    result["static_ready"] = bool(result["static_ready"] and result["source_ready"])
    result["online_ready"] = bool(result["online_ready"] and result["source_ready"])
    progress("authenticate_inputs_after")
    return dict(result)
