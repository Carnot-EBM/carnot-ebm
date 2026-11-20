"""REQ-REPORT-9255-V788: split V782-V787 replay custody and V788 contract authentication."""

import ast
from copy import deepcopy
import json
from pathlib import Path
from typing import Any
from unittest.mock import patch

import yaml

from carnot.reporting import v787_contract_evidence as prior
from carnot.reporting.current_work_receipt import canonical_hash
from carnot.reporting.historical_contract_inputs import authenticated, original_root
from carnot.reporting.logit_methods_9126 import ref as ref
from carnot.reporting.v749_contract_methods import encode

Json = dict[str, Any]
ROOT = prior.ROOT
NAME, TASK = "experiment_9255_v788_contract", "exp9255-contract"
CLI, TEST = f"scripts/experiments/{NAME}.py", "tests/python/test_v788_contract_9255.py"
CONFIG = "openspec/change-proposals/v788-contract-inputs.json"
FROZEN = json.loads((ROOT / CONFIG).read_bytes())
DESIGN, ACTIVE = "openspec/change-proposals/research-roadmap-vNEXT.md", "research-roadmap.yaml"
COLLISION_INPUTS = prior.COLLISION_INPUTS
OWNED = [
    "python/carnot/reporting/v788_contract_evidence.py",
    "python/carnot/reporting/v788_contract.py",
    "python/carnot/reporting/historical_contract_inputs.py",
    CLI,
]
AFFECTED: list[str] = []
MODEL_SPECS: list[Json] = []
np = prior.np
HISTORICAL_BASE: Path


def progress(phase: str) -> None:
    """Flush boundaries so authentication and supervised validation remain observable."""
    print(f"[exp9255] phase={phase}", flush=True)


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
                ["2026.11.788"] * 2,
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
                    and task["id"].startswith(f"exp{9255 + index}-")
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
                slot_id = f"exp{9255 + index}"
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


def _inspect(root: Path) -> Json:
    """Reexecute original custody checks in private roots; old receipt success grants no credit."""
    progress("authenticate_inputs_before")
    contract = authority(root)
    v787_root = (
        root / "absent" if not root.is_dir() else original_root(HISTORICAL_BASE / "v787", 787)
    )
    with patch.object(prior, "HISTORICAL_BASE", HISTORICAL_BASE, create=True):
        inherited = prior.inspect(v787_root)
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
    prior_roots = list(inherited["historical_input_roots"])
    prior_non_contract = [
        c for c in inherited["checks"] if c not in inherited.get("contract", {}).get("checks", [])
    ]
    inherited.update(
        contract=contract,
        checks=checks + prior_non_contract,
        source_references=references + inherited["source_references"],
        historical_input_roots=[],
        preserved_v787={},
        selective_deferral_protocol=deepcopy(FROZEN["selective_deferral_protocol"]),
        conformal_risk_protocol=deepcopy(FROZEN["conformal_risk_protocol"]),
        replay_ready=False,
    )
    if not {"9211", "9222", "9223", "9225", "9240", "9241", "9242", "9244"}.issubset(values):
        return dict(inherited)
    inherited["historical_input_roots"] = [
        *prior_roots[:4],
        dict(
            root=str(v787_root), **json.loads((v787_root / "historical_inputs.json").read_bytes())
        ),
        *prior_roots[4:],
    ]
    original_9241 = values["9241"]
    authenticated(original_9241["work_reference"])
    receipt_map = {
        r["name"]: r
        for r in original_9241["validation_receipts"]
        if r["name"].startswith(("owned_tests", "direct_consumers"))
    }
    for receipt in receipt_map.values():
        for stream in ["stdout", "stderr"]:
            reference = dict(path=receipt[stream + "_path"], sha256=receipt[stream + "_sha256"])
            authenticated(reference)
            inherited["source_references"].append(
                dict(reference, imported_fields=["v787_validation_receipt"])
            )
    inherited["checks"].extend(
        [
            check(
                root / "results/experiment_9241_v787_contract.json",
                "readiness_and_verdict",
                [0, 0, 0, 0, 0, False, "complete_blocked_v787_contract"],
                [
                    original_9241["contract_ready_score"],
                    original_9241["replay_authority_ready_score"],
                    original_9241["source_ready_score"],
                    original_9241["static_methods_ready_score"],
                    original_9241["online_methods_ready_score"],
                    original_9241["required_checks_passed"],
                    original_9241["honest_verdict"],
                ],
                "v787_contract_historical_outcome",
            ),
            check(
                root / "results/experiment_9241_v787_contract.json",
                "consumer_shards_passed",
                [True] * 6,
                [
                    bool(r["passed"] and r["exit_code"] == 0)
                    for r in original_9241["consumer_shard_rows"]
                ],
                "v787_consumer_shards_pass",
            ),
            check(
                root / "results/experiment_9242_corpus_prefilter.json",
                "honest_verdict",
                "blocked_gate_check_failed",
                values["9242"]["honest_verdict"],
                "v787_prefilter_pre_gate_skip",
            ),
        ]
    )
    exp9244 = values["9244"]
    authenticated(exp9244["work_reference"])
    for key in [
        "conformal_risk_protocol_reference",
        "ca_ocrc_protocol_reference",
        "provenance_ledger_reference",
        "reducer_reference",
    ]:
        authenticated(exp9244[key])
        target_key = (
            "conformal_risk_reducer_reference"
            if key == "reducer_reference"
            else "qualified_conformal_risk_protocol_reference"
            if key == "conformal_risk_protocol_reference"
            else key
        )
        inherited["protocol_references"][target_key] = dict(
            exp9244[key], producer_path=exp9244[key]["path"]
        )
        if key == "provenance_ledger_reference":
            inherited["protocol_references"]["mycelium_provenance_ledger_reference"] = dict(
                exp9244[key], producer_path=exp9244[key]["path"]
            )
        inherited["source_references"].append(
            dict(exp9244[key], imported_fields=["qualified_exp9244_authority"])
        )
    inherited["checks"].append(
        check(
            root / "results/experiment_9244_v787_conformal_risk_protocol.json",
            "readiness_scores",
            [1, 1, 1, True, False, "complete_null_v787_conformal_risk_protocol"],
            [
                exp9244["conformal_risk_ready_score"],
                exp9244["ca_ocrc_ready_score"],
                exp9244["provenance_audit_ready_score"],
                exp9244["required_checks_passed"],
                exp9244["scientific_effect_measured"],
                exp9244["honest_verdict"],
            ],
            "qualified_exp9244",
        )
    )
    inherited["preserved_v787"] = dict(
        exp9241_readiness_scores=dict(
            contract_ready_score=original_9241["contract_ready_score"],
            replay_authority_ready_score=original_9241["replay_authority_ready_score"],
            source_ready_score=original_9241["source_ready_score"],
            static_methods_ready_score=original_9241["static_methods_ready_score"],
            online_methods_ready_score=original_9241["online_methods_ready_score"],
        ),
        exp9241_verdict=original_9241["honest_verdict"],
        exp9241_required_checks_passed=original_9241["required_checks_passed"],
        exp9241_consumer_shard_rows=deepcopy(original_9241["consumer_shard_rows"]),
        exp9241_owned_statement_coverage=dict(
            complete=original_9241["owned_statement_coverage"]["complete"],
            percent=original_9241["owned_statement_coverage"]["percent"],
            measured_file_count=len(original_9241["owned_statement_coverage"]["files"]),
            owned_changed_path_count=len(
                original_9241["validation_scope_manifest"]["owned_changed_paths"]
            ),
        ),
        exp9242_pre_gate_refusal=values["9242"]["honest_verdict"],
        exp9244_scores=dict(
            conformal_risk_ready_score=exp9244["conformal_risk_ready_score"],
            ca_ocrc_ready_score=exp9244["ca_ocrc_ready_score"],
            provenance_audit_ready_score=exp9244["provenance_audit_ready_score"],
            required_checks_passed=exp9244["required_checks_passed"],
            scientific_effect_measured=exp9244["scientific_effect_measured"],
        ),
        exp9244_verdict=exp9244["honest_verdict"],
        design_reference=ref(
            root / "openspec/change-proposals/research-roadmap-v787-preserved-20261120.md"
        ),
        queue_reference=ref(
            root / "openspec/change-proposals/research-roadmap-v787-queue-preserved-20261120.yaml"
        ),
        accounting=dict(
            executed=2,
            skipped=5,
            scheduler_artifact_skips=1,
            scheduler_log_skips=4,
            omitted_slots=7,
            scientific_success_inferred=False,
        ),
        current_qualification_credit=0,
    )
    obligations = inherited["validation_obligations"]
    obligations["affected_current_paths"] = [*OWNED, TEST, CONFIG, *AFFECTED]
    obligations["affected_v787_obligations"] = deepcopy(
        original_9241["validation_scope_manifest"]["validation_obligations"]
    )
    obligations["qualified_v787_conformal_obligations"] = deepcopy(
        exp9244["validation_scope_manifest"]
    )
    for test_rel in [TEST, "tests/python/test_conformal_risk_9244.py"]:
        for node in ast.parse((ROOT / test_rel).read_bytes()).body:
            if isinstance(node, ast.FunctionDef) and node.name.startswith("test_"):
                obligations["affected_node_assertions"].append(
                    dict(
                        node_id=test_rel + "::" + node.name,
                        sha256=canonical_hash(ast.dump(node, include_attributes=False)),
                        producer_reference=ref(ROOT / test_rel),
                        current_qualification_credit=0,
                    )
                )
    obligations["conformal_risk_protocol"] = deepcopy(FROZEN["conformal_risk_protocol"])
    non_contract = all(c["passed"] for c in inherited["checks"] if c not in contract["checks"])
    for key in ["source_ready", "static_ready", "online_ready"]:
        inherited[key] = bool(inherited[key] and non_contract)
    inherited["replay_ready"] = bool(non_contract and len(inherited["historical_input_roots"]) == 8)
    progress("authenticate_inputs_after")
    return dict(inherited)


def inspect(root: Path) -> Json:
    """Missing external authority is a terminal blocked operand, never unfinished owned work."""
    try:
        return _inspect(root)
    except (OSError, ValueError, KeyError) as error:
        with patch.object(prior, "HISTORICAL_BASE", HISTORICAL_BASE, create=True):
            result = prior.inspect(root / "absent")
        contract = authority(root)
        name = str(error).partition("historical_input_drift:")[2]
        path = Path(name) if name else root
        field = str(error.args[0]) if isinstance(error, KeyError) else "original_authority"
        observed = (
            "missing_field"
            if isinstance(error, KeyError)
            else ref(path)["sha256"] or "missing_file"
        )
        result.update(
            contract=contract,
            checks=contract["checks"]
            + [
                check(
                    path,
                    field,
                    "authenticated_original_bytes",
                    observed,
                    "historical_inputs",
                )
            ],
            historical_input_roots=[],
            preserved_v787={},
            selective_deferral_protocol=deepcopy(FROZEN["selective_deferral_protocol"]),
            conformal_risk_protocol=deepcopy(FROZEN["conformal_risk_protocol"]),
            replay_ready=False,
        )
        return dict(result)
