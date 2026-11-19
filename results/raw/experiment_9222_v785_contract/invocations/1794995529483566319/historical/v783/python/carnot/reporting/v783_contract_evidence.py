"""REQ-REPORT-9194-V783: retained human authority cannot be replaced by checkout bytes."""

import ast
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
from typing import Any

import yaml

from carnot.reporting import v782_contract_evidence as prior
from carnot.reporting.current_work_receipt import canonical_hash
from carnot.reporting.logit_methods_9126 import ref as ref
from carnot.reporting.v749_contract_methods import encode

Json = dict[str, Any]
ROOT = prior.ROOT
NAME, TASK = "experiment_9194_v783_contract", "exp9194-contract"
CLI, TEST = f"scripts/experiments/{NAME}.py", "tests/python/test_v783_contract_9194.py"
CONFIG = "openspec/change-proposals/v783-contract-inputs.json"
FROZEN = json.loads((ROOT / CONFIG).read_bytes())
DESIGN = "openspec/change-proposals/research-roadmap-vNEXT.md"
ACTIVE = (
    "research-roadmap-next.yaml"
    if (ROOT / "research-roadmap-next.yaml").exists()
    else "research-roadmap.yaml"
)
ARCHIVE = "openspec/change-proposals/research-roadmap-v782-preserved-20261117.md"
QUEUE = "openspec/change-proposals/research-roadmap-v782-queue-preserved-20261117.yaml"
COLLISION_INPUTS = ["research-complete.yaml", "ops/exclusion_manifest.yaml"]
OWNED = [
    "python/carnot/reporting/v783_contract_evidence.py",
    "python/carnot/reporting/v783_contract.py",
    CLI,
]
MODEL_SPECS: list[Json] = []
np = prior.np


def progress(phase: str) -> None:
    """Flush boundaries so authentication and supervised validation remain observable."""
    print(f"[exp9194] phase={phase}", flush=True)


def check(path: Path, field: str, expected: Any, observed: Any, name: str) -> Json:
    """Use the established strict gate to distinguish missing evidence, false and zero."""
    return dict(prior.check(path, field, expected, observed, name), upstream=TASK)


def authority(root: Path) -> Json:
    """Compare complete ordered objects and reject occupied IDs before any activation."""
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
                ["2026.11.783"] * 2,
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
                and table[index]
                == [str(index + 1), str(task["phase"]), task["id"], task["deliverable"]]
                and task["id"].startswith(f"exp{9194 + index}-")
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
        checks.append(
            check(
                root / ACTIVE,
                "all_complete_objects",
                True,
                len(rows) == 14 and all(r["numerator"] for r in rows),
                "full_contract",
            )
        )
        if not all(c["passed"] for c in checks):
            raise ValueError("contract_mismatch")
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
                root / ACTIVE,
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


def inspect(root: Path) -> Json:
    """Reauthenticate original retained bytes; historical successes grant no current test credit."""
    progress("authenticate_inputs_before")
    contract = authority(root)
    checks, references, values = [], [], {}
    result: Json = dict(
        contract=contract,
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
        diagnostic_protocol={},
        retired_schedule={},
        validation_obligations={},
        entropy_reference_protocol=deepcopy(prior.FROZEN["entropy_reference_protocol"]),
        label_ambiguity_protocol=deepcopy(FROZEN["label_ambiguity_protocol"]),
        preserved_v781={},
    )
    for name, pin in FROZEN["pins"].items():
        path = root / name
        gate = check(
            path, "sha256", pin, ref(path)["sha256"] or "missing_file", "immutable_operand"
        )
        checks.append(gate)
        references.append(
            dict(
                path=str(path),
                producer_path=str(path),
                sha256=pin,
                imported_fields=["original_authority", "readiness", "rows"],
            )
        )
        if gate["passed"] and name.startswith("results/"):
            values[Path(name).name.split("_")[1]] = json.loads(path.read_bytes())
    result.update(checks=contract["checks"] + checks, source_references=references)
    tools = {
        name: importlib.util.find_spec(name) is not None
        for name in ["pytest", "coverage", "ruff", "mypy", "yaml", "numpy"]
    }
    result["checks"].append(
        check(root, "required_tools", True, all(tools.values()), "resource_tools")
    )
    if not all(
        c["passed"]
        for c in checks
        if str(c["path"]).startswith(str(root / "results"))
        or str(c["path"]) in {str(root / ARCHIVE), str(root / QUEUE)}
    ):
        return result

    def bound(reference: Json) -> Json:
        """Decode only authenticated original producers, never newly pinned replacements."""
        path = Path(reference["path"])
        gate = check(
            path,
            "sha256",
            reference["sha256"],
            ref(path)["sha256"] or "missing_file",
            "original_producer_binding",
        )
        result["checks"].append(gate)
        result["source_references"].append(
            dict(
                reference,
                producer_path=reference.get("producer_path", str(path)),
                imported_fields=["retained_original_bytes"],
            )
        )
        return dict(json.loads(path.read_bytes())) if gate["passed"] else {}

    original = bound(values["9180"]["work_reference"])
    capstone = bound(values["9193"]["work_reference"])
    if not original or not capstone:
        return result
    inherited = deepcopy(original["inspection"])
    retained = {
        **capstone["operands"],
        **{str(ROOT / k): v for k, v in capstone["snapshots"].items()},
    }
    bindings = [
        *inherited["source_references"],
        *values["9180"]["protocol_references"].values(),
        *capstone["operands"].values(),
        *capstone["snapshots"].values(),
    ]
    for index, reference in enumerate(bindings):
        selected = retained.get(reference["path"], reference)
        path = Path(selected["path"])
        result["checks"].append(
            check(
                path,
                "sha256",
                reference["sha256"],
                ref(path)["sha256"] or "missing_file",
                "retained_authority",
            )
        )
        result["source_references"].append(
            dict(
                path=str(path),
                producer_path=reference.get("producer_path", reference["path"]),
                sha256=reference["sha256"],
                imported_fields=["original_human_method_or_validation_evidence"],
            )
        )
        if index % 50 == 0:
            progress(f"retained_bindings_{index}_of_{len(bindings)}")
    inherited.update(
        contract=contract,
        checks=result["checks"],
        source_references=result["source_references"],
        previous_outcome_rows=deepcopy(values["9193"]["task_outcome_rows"]),
        label_ambiguity_protocol=result["label_ambiguity_protocol"],
        preserved_v782=dict(
            design_reference=ref(root / ARCHIVE),
            queue_reference=ref(root / QUEUE),
            correction=values["9181"]["causal_fix_ready_score"],
            correction_verdict=values["9181"]["verdict_class"],
            arc_verdict=values["9192"]["verdict_class"],
            retired_tasks=FROZEN["scheduling_repair_revision"]["retired_tasks"],
        ),
    )
    for number, field, expected in [
        (9180, "source_ready_score", 1),
        (9180, "static_methods_ready_score", 1),
        (9180, "online_methods_ready_score", 1),
        (9181, "causal_fix_ready_score", 0),
        (9193, "required_checks_passed", True),
    ]:
        name = next(p for p in FROZEN["pins"] if p.startswith(f"results/experiment_{number}_"))
        inherited["checks"].append(
            check(
                root / name,
                field,
                expected,
                values[str(number)].get(field, "missing_field"),
                "historical_gate",
            )
        )
    o = inherited["validation_obligations"]
    o["affected_node_assertions"].extend(
        dict(
            node_id=r["unit_id"],
            sha256=r["assertion_hash"],
            producer_reference=r["assertion_reference"],
            current_qualification_credit=0,
        )
        for r in o["node_bindings"]
        if "test_no_artifact_on_disk" in r["unit_id"]
    )
    o["historical_v782_validation_units"] = deepcopy(values["9193"]["validation_unit_rows"])
    o["historical_v782_remaining"] = deepcopy(values["9181"]["remaining_obligations"])
    paths = {TEST} | {
        "tests/python/" + Path(r["path"]).name
        for v in values.values()
        for r in v["code_config_hashes"]
        if Path(r["path"]).name.startswith("test_")
    }
    for name in sorted(paths):
        path = ROOT / name
        for node in ast.parse(path.read_bytes()).body:
            if isinstance(node, ast.FunctionDef) and node.name.startswith("test_"):
                o["affected_node_assertions"].append(
                    dict(
                        node_id=name + "::" + node.name,
                        sha256=canonical_hash(ast.dump(node, include_attributes=False)),
                        producer_reference=ref(path),
                        current_qualification_credit=0,
                    )
                )
    o["affected_node_assertions"] = list(
        {r["node_id"]: r for r in o["affected_node_assertions"]}.values()
    )
    o["affected_current_paths"] = [*OWNED, TEST, CONFIG]
    valid = all(
        c["passed"]
        for c in inherited["checks"]
        if c not in contract["checks"]
        and c["path"] not in {str(root / p) for p in [DESIGN, ACTIVE, *COLLISION_INPUTS]}
    )
    for key in ["source_ready", "static_ready", "online_ready"]:
        inherited[key] = bool(inherited[key] and valid)
    progress("authenticate_inputs_after")
    return dict(inherited)
