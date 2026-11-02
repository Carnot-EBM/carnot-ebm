"""REQ-REPORT-8627: authenticate independent branch operands without historical promotion.

Registration reads immutable receipts and private custody. It performs no head
inference, and earlier failed execution remains a separate closed fact.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from carnot.reporting import v741_contract_methods as e

Json = dict[str, Any]


def historical(p: Json) -> Json:
    """Reduce branch readiness from pinned producers, actual bytes and unchanged gates."""
    fields = {
        "learning": {
            8431: "head_ready_score",
            8580: "current_validator_ready_score",
            8581: "online_adapter_ready_score",
            8616: "required_checks_passed",
        },
        "extraction": {
            8617: "triviaplus_coverage_inputs_ready_score",
            8620: "extraction_audit_ready_score",
        },
        "deployment": {
            8431: "head_ready_score",
            8621: "lean_measurement_ready_score",
            8622: "lean_cost_ready_score",
        },
        "arc": {8588: "arc_panel_rows_ready_score", 8624: "arc_observer_ready_score"},
        "canary": {8431: "head_ready_score", 8580: "current_validator_ready_score"},
    }
    result: Json = dict(
        families=[],
        references=[],
        gate_check_summary=[],
        historical_model_provenance=[],
        resident_inputs_ready_score=0,
    )
    for branch, requirements in fields.items():
        failures, values = [], {}
        refs = [p["current_producers"][str(n)] for n in requirements] + p["branch_references"][
            branch
        ]
        for ref in refs:
            if not e.checked(ref):
                failures.append(
                    e.gate(
                        Path(ref["path"]),
                        "original_operand_sha256",
                        ref["sha256"],
                        e.sha256_file(Path(ref["path"])) if Path(ref["path"]).is_file() else None,
                        branch,
                    )
                )
        for n, field in requirements.items():
            ref = p["current_producers"][str(n)]
            try:
                value = e.read(ref)
                values[n] = value
                observed = value.get(field)
            except (OSError, ValueError, KeyError, TypeError):
                observed = None
            if observed != 1:
                failures.append(e.gate(Path(ref["path"]), field, 1, observed, f"exp{n}"))
        if branch == "extraction" and len(values) == len(requirements):
            roster = values[8620]["extraction_rows"]
            identity = [
                (r["opaque_id"], r["question_hash"], r["source_hash"], r["answer_hash"])
                for r in roster
            ]
            counts = (len(roster), len({r["question_hash"] for r in roster}))
            predictor = p["source_roster_reference"]["predictor_reference"]
            try:
                private = [
                    json.loads(line) for line in Path(predictor["path"]).read_text().splitlines()
                ]
                complete = {r["opaque_id"] for r in roster} <= {r["opaque_id"] for r in private}
            except (OSError, ValueError, KeyError):
                complete = False
            if (
                counts != (190, 128)
                or e.canonical_hash(identity) != p["source_roster_reference"]["roster_sha256"]
                or not complete
            ):
                failures.append(
                    e.gate(
                        Path(predictor["path"]),
                        "complete190_by128_roster",
                        [190, 128, True],
                        [*counts, complete],
                        "exp8620",
                    )
                )
        if branch == "canary" and 8580 in values:
            ref = values[8580]["current_validator_reference"]
            try:
                validator = e.read(ref)
                custody = validator["ready"] and all(e.checked(r) for r in validator["code"])
            except (OSError, ValueError, KeyError, TypeError):
                custody = False
            if not custody:
                failures.append(
                    e.gate(Path(ref["path"]), "current_validator_custody", True, custody, "exp8580")
                )
        ready = int(not failures)
        result[branch + "_inputs_ready_score"] = ready
        result["families"].append(
            dict(branch=branch, qualified=bool(ready), references=refs, gate_check_summary=failures)
        )
        result["references"] += refs
        result["gate_check_summary"] += failures
        result["historical_model_provenance"] += [
            dict(experiment_id=n, provenance=v.get("historical_model_provenance", []))
            for n, v in values.items()
        ]
        e.progress(
            "branch_" + branch, len(result["families"]), len(fields) - len(result["families"])
        )
    return result
