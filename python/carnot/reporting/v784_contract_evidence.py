"""REQ-REPORT-9208-V784: exact current authority remains distinct from historical replay."""

import ast
from copy import deepcopy
import json
from pathlib import Path
from typing import Any

import yaml

from carnot.reporting import v783_contract_evidence as prior
from carnot.reporting.logit_methods_9126 import ref as ref
from carnot.reporting.current_work_receipt import canonical_hash
from carnot.reporting.v749_contract_methods import encode
from carnot.reporting.historical_contract_inputs import original_root, authenticated

Json = dict[str, Any]
ROOT = prior.ROOT
NAME, TASK = "experiment_9208_v784_contract", "exp9208-contract"
CLI, TEST = f"scripts/experiments/{NAME}.py", "tests/python/test_v784_contract_9208.py"
CONFIG = "openspec/change-proposals/v784-contract-inputs.json"
FROZEN = json.loads((ROOT / CONFIG).read_bytes())
DESIGN, ACTIVE = "openspec/change-proposals/research-roadmap-vNEXT.md", "research-roadmap.yaml"
COLLISION_INPUTS = prior.COLLISION_INPUTS
OWNED = [
    "python/carnot/reporting/v784_contract_evidence.py",
    "python/carnot/reporting/v784_contract.py",
    "python/carnot/reporting/protocol_owners.py",
    "python/carnot/reporting/historical_contract_inputs.py",
    CLI,
]
MODEL_SPECS: list[Json] = []
np = prior.np
HISTORICAL_BASE: Path


def progress(phase: str) -> None:
    """Flush boundaries so authentication and supervised validation remain observable."""
    print(f"[exp9208] phase={phase}", flush=True)


def check(path: Path, field: str, expected: Any, observed: Any, name: str) -> Json:
    """Retain strict types so absence, zero and false remain distinct operands."""
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
                ["2026.11.784"] * 2,
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
                task == dict(original, prompt=encode(original["prompt"]))
                and table[index]
                == [str(index + 1), str(task["phase"]), task["id"], task["deliverable"]]
                and task["id"].startswith(f"exp{9208 + index}-")
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


def _inspect(root: Path) -> Json:
    """Reexecute original custody checks in private roots; old receipt success grants no credit."""
    progress("authenticate_inputs_before")
    contract = authority(root)
    inherited = (
        prior.inspect(root / "absent")
        if not root.is_dir()
        else prior.inspect(original_root(HISTORICAL_BASE / "v783", 783))
    )
    checks = list(contract["checks"])
    references = []
    values = {}
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
        if row["passed"] and name.startswith("results/"):
            values[Path(name).name.split("_")[1]] = json.loads(path.read_bytes())
    inherited.update(
        contract=contract,
        checks=checks + inherited["checks"],
        source_references=references + inherited["source_references"],
    )
    inherited["historical_input_roots"] = []
    inherited["preserved_v783"] = {}
    inherited["target_identifiability_rubric"] = FROZEN["target_identifiability_rubric"]
    if "9194" not in values or "9207" not in values:
        return dict(inherited)
    for generation in [782, 783]:
        path = original_root(HISTORICAL_BASE / f"v{generation}", generation)
        manifest = json.loads((path / "historical_inputs.json").read_bytes())
        inherited["historical_input_roots"].append(dict(root=str(path), **manifest))
    original = values["9194"]
    original_work = json.loads(authenticated(original["work_reference"]))
    failures = [
        r
        for r in original["validation_receipts"]
        if r["name"] in ["owned_tests", "direct_consumers"]
    ]
    for receipt in failures:
        for stream in ["stdout", "stderr"]:
            reference = dict(path=receipt[stream + "_path"], sha256=receipt[stream + "_sha256"])
            authenticated(reference)
            inherited["source_references"].append(
                dict(reference, imported_fields=["original_failed_validation"])
            )
    cold_reference = FROZEN["original_private_cold_replay_reference"]
    cold = json.loads(authenticated(cold_reference))
    inherited["source_references"].append(
        dict(cold_reference, imported_fields=["original_private_child_failure"])
    )
    for stream in ["stdout", "stderr"]:
        reference = dict(path=cold[stream + "_path"], sha256=cold[stream + "_sha256"])
        authenticated(reference)
        inherited["source_references"].append(
            dict(reference, imported_fields=["original_private_child_log"])
        )
    inherited["checks"].append(
        check(
            Path(cold_reference["path"]),
            "actual_exit",
            1,
            cold["actual_exit"],
            "original_private_failure",
        )
    )
    pre_ces = original_work["scope"]["pre_ces_binding_config_reference"]
    config = json.loads(authenticated(pre_ces))
    inherited["checks"].append(
        check(
            Path(pre_ces["path"]),
            "entropy_reference_protocol",
            "missing_field",
            config.get("entropy_reference_protocol", "missing_field"),
            "original_missing_protocol",
        )
    )
    inherited["preserved_v783"] = dict(
        readiness_scores=[
            original[k]
            for k in [
                "contract_ready_score",
                "source_ready_score",
                "static_methods_ready_score",
                "online_methods_ready_score",
            ]
        ],
        original_failed_receipts=failures,
        original_private_cold_replay=cold,
        pre_ces_configuration=pre_ces,
        original_verdict=original["verdict_class"],
        design_reference=ref(
            root / "openspec/change-proposals/research-roadmap-v783-preserved-20261117.md"
        ),
        queue_reference=ref(
            root / "openspec/change-proposals/research-roadmap-v783-queue-preserved-20261117.yaml"
        ),
        current_qualification_credit=0,
    )
    inherited["checks"].append(
        check(
            root / "results/experiment_9194_v783_contract.json",
            "readiness_scores",
            [0] * 4,
            inherited["preserved_v783"]["readiness_scores"],
            "preserved_zero_gates",
        )
    )
    path = original_root(HISTORICAL_BASE / "v783_capstone", 783, suite="capstone")
    inherited["historical_input_roots"].append(
        dict(root=str(path), **json.loads((path / "historical_inputs.json").read_bytes()))
    )
    obligations = inherited["validation_obligations"]
    obligations["historical_v783_validation_units"] = deepcopy(
        values["9207"]["validation_unit_rows"]
    )
    obligations["affected_current_paths"] = [*OWNED, TEST, CONFIG]
    obligations["affected_v783_obligations"] = deepcopy(values["9207"]["validation_obligations"])
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
    obligations["target_identifiability_rubric"] = FROZEN["target_identifiability_rubric"]
    progress("authenticate_inputs_after")
    return dict(inherited)


def inspect(root: Path) -> Json:
    """Missing external authority is a terminal blocked operand, never unfinished owned work."""
    try:
        return _inspect(root)
    except (OSError, ValueError, KeyError) as error:
        result = prior.inspect(root / "absent")
        contract = authority(root)
        name = str(error).partition("historical_input_drift:")[2]
        path = Path(name) if name else root
        observed = (
            "missing_file"
            if not path.exists()
            else "missing_field"
            if isinstance(error, KeyError)
            else str(error)
        )
        result.update(
            contract=contract,
            checks=contract["checks"]
            + [
                check(
                    path,
                    "original_authority",
                    "authenticated_original_bytes",
                    observed,
                    "historical_inputs",
                )
            ],
            historical_input_roots=[],
            preserved_v783={},
            target_identifiability_rubric=FROZEN["target_identifiability_rubric"],
        )
        return dict(result)
