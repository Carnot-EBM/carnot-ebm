"""REQ-REPORT-8667: qualified source operands remain separate from failed execution."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from carnot.reporting import v743_contract_methods as previous
from carnot.reporting import v744_contract_methods as e

Json = dict[str, Any]


def historical(p: Json) -> Json:
    """Authenticate each producer without treating historical readiness as future work."""
    result = dict(previous.historical(p))
    old_families = {row["branch"]: row for row in result["families"]}
    result["historical_gate_check_summary"] += result["gate_check_summary"]
    result["gate_check_summary"] = (
        old_families["learning"]["gate_check_summary"]
        + old_families["canary"]["gate_check_summary"]
    )
    result["preserved_v743_failures"] = {
        number: e.read(p["v743_producers"][number]) for number in ["8654", "8664"]
    }
    ref = p["learning_registration"]["count_validator_receipt"]
    path = Path(ref["path"])
    if not e.checked(ref):
        result["learning_operands_ready_score"] = 0
        result["gate_check_summary"].append(
            e.gate(
                path,
                "count_validator_receipt_sha256",
                ref["sha256"],
                e.sha256_file(path) if path.is_file() else None,
                "exp8580",
            )
        )
    else:
        observed = e.read(ref).get("current_validator_ready_score")
        if observed != 1:
            result["learning_operands_ready_score"] = 0
            result["gate_check_summary"].append(
                e.gate(path, "current_validator_ready_score", 1, observed, "exp8580")
            )
    for field, producers in p["branch_requirements"].items():
        branch = field.removesuffix("_inputs_ready_score")
        refs = (
            p["observation_ownership_protocol"]["actual_transfer_references"]
            if branch == "arc"
            else p["operand_references"][branch]
        )
        failures = [] if branch == "arc" else list(old_families[branch]["gate_check_summary"])
        for ref in refs:
            if not e.checked(ref):
                path = Path(ref["path"])
                row = e.gate(
                    path,
                    "scientific_operand_sha256",
                    ref["sha256"],
                    e.sha256_file(path) if path.is_file() else None,
                    branch,
                )
                row["operand_state"] = "changed_bytes" if path.is_file() else "missing_file"
                failures.append(row)
        for number, producer_field in producers.items():
            ref = p["v743_producers"][number]
            path = Path(ref["path"])
            if not e.checked(ref):
                observed = e.sha256_file(path) if path.is_file() else None
                failure = e.gate(path, "producer_sha256", ref["sha256"], observed, f"exp{number}")
                failure["operand_state"] = "changed_bytes" if path.is_file() else "missing_file"
                failures.append(failure)
                continue
            value = e.read(ref)
            for key, expected in [
                (producer_field, 1),
                ("required_checks_passed", True),
                ("flagged_adversarial", False),
            ]:
                observed = value.get(key)
                if observed != expected:
                    failures.append(e.gate(path, key, expected, observed, f"exp{number}"))
            result["references"].append(ref)
        # A source-qualified branch cannot repair a missing original scientific operand.
        result[field] = int(not failures)
        result["gate_check_summary"] += failures
        result["families"].append(
            dict(
                branch=field,
                qualified=bool(result[field]),
                references=[p["v743_producers"][n] for n in producers],
                gate_check_summary=failures,
            )
        )
    result["circular_gate_regression_rows"][0]["actual_training_requires"] = [
        "learning_operands_ready_score",
        "Exp8669.learning_execution_ready_score",
    ]
    return result
