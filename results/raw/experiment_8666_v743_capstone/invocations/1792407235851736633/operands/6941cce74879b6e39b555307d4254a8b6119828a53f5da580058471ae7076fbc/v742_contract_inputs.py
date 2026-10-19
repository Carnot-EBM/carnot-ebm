"""REQ-REPORT-8640: research operands permit qualification after historical failure.

The original reader remains unchanged. This adapter checks current research
inputs without asking the execution repair to have succeeded in advance.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from carnot.reporting import v741_contract_methods as previous
from carnot.reporting import v742_contract_methods as e

Json = dict[str, Any]


def historical(p: Json) -> Json:
    """Keep the historical circular result while checking each available branch separately."""
    old = previous.historical(p)
    result: Json = dict(old)
    result["historical_gate_check_summary"] = old["gate_check_summary"]
    result["gate_check_summary"] = []
    result["families"] = []
    for branch, refs in p["operand_references"].items():
        failures = []
        values: Json = {}
        for ref in refs:
            if not e.checked(ref):
                path = Path(ref["path"])
                failures.append(
                    e.gate(
                        path,
                        "research_operand_sha256",
                        ref["sha256"],
                        e.sha256_file(path) if path.is_file() else None,
                        branch,
                    )
                )
        requirements = {
            "learning": {
                8431: "head_ready_score",
                8580: "current_validator_ready_score",
                8581: "online_adapter_ready_score",
            },
            "deployment": {8633: "installed_prototype_ready_score"},
            "arc": {8637: "real_horizon_rows_ready_score"},
        }.get(branch, {})
        for number, field in requirements.items():
            ref = p["current_producers"][str(number)]
            try:
                values[str(number)] = e.read(ref)
                observed = values[str(number)].get(field)
            except (OSError, ValueError, KeyError, TypeError):
                observed = None
            if observed != 1:
                failures.append(e.gate(Path(ref["path"]), field, 1, observed, f"exp{number}"))
        if branch == "learning":
            for key in [
                "head_checkpoint_reference",
                "learning_data_reference",
                "test_inventory_reference",
                "learning_protocol_reference",
            ]:
                ref = p[key]
                if not e.checked(ref):
                    path = Path(ref["path"])
                    failures.append(
                        e.gate(
                            path,
                            key,
                            ref["sha256"],
                            e.sha256_file(path) if path.is_file() else None,
                            branch,
                        )
                    )
            try:
                inventory = e.read(p["test_inventory_reference"])
                data = e.read(p["learning_data_reference"])["view"]["rows"]
                objective = e.read(p["learning_protocol_reference"])
                validator = e.read(values["8580"]["current_validator_reference"])
                valid = bool(
                    len(inventory["nodes"]) == len(set(inventory["nodes"])) == 158
                    and all(e.checked(r["source"]) for r in inventory["assertion_sources"].values())
                    and data
                    and all(len(r["x"]) == 9 and r["y"] in (0, 1) for r in data)
                    and p["objective_updates"] == objective["objective_updates"]
                    and p["objective_audit"] == objective["objective_audit"]
                    and validator["ready"]
                    and all(e.checked(r) for r in validator["code"])
                )
            except (OSError, ValueError, KeyError, TypeError):
                valid = False
            if not valid:
                failures.append(
                    e.gate(
                        Path(p["test_inventory_reference"]["path"]),
                        "original_data_objective_and_test_inventory",
                        True,
                        valid,
                        branch,
                    )
                )
        elif not old[branch + "_inputs_ready_score"]:
            failures += next(
                r["gate_check_summary"] for r in old["families"] if r["branch"] == branch
            )
        field = (
            "learning_operands_ready_score"
            if branch == "learning"
            else branch + "_inputs_ready_score"
        )
        result[field] = int(not failures)
        result["families"].append(
            dict(
                branch=branch, qualified=not failures, references=refs, gate_check_summary=failures
            )
        )
        result["gate_check_summary"] += failures
        result["references"] = result["references"] + refs
        e.progress(
            "research_operands_" + branch, len(result["families"]), 5 - len(result["families"])
        )
    original = e.read(p["current_producers"]["8627"])
    failed = e.read(p["current_producers"]["8616"])
    result["circular_gate_regression_rows"] = [
        dict(
            unit_id="historical_execution_failure_allows_operand_qualification",
            historical_learning_inputs_ready_score=original["learning_inputs_ready_score"],
            historical_execution_required_checks_passed=failed["required_checks_passed"],
            current_learning_operands_ready_score=result["learning_operands_ready_score"],
            execution_qualification_may_run=bool(result["learning_operands_ready_score"]),
            actual_training_requires=[
                "learning_operands_ready_score",
                "Exp8641.learning_execution_ready_score",
            ],
            passed=original["learning_inputs_ready_score"] == 0
            and failed["required_checks_passed"] is False,
        )
    ]
    return result
