"""REQ-REPORT-8709: preserve the shared report schema and original evidence boundaries."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import patch

from carnot.reporting import v746_contract_report as previous
from carnot.reporting import v747_contract_methods as m
from carnot.reporting.current_work_receipt import canonical_hash


def build(
    work: dict[str, Any], validation: dict[str, Any], raw: Path, output: Path
) -> dict[str, Any]:
    """Reuse schema rules while distinguishing retained execution from current binding checks."""
    with patch.object(previous, "a", m):
        value = previous.build(work, validation, raw, output)
    value.update(
        experiment_id=8709,
        random_seed=7478709,
        honest_verdict=value["honest_verdict"].replace(
            "v746_retained_contract_authority", "v747_contract_methods"
        ),
        task_contract_rows=value["rows"],
        retained_authority=work["retained_authority"],
        canonical_tasks_sha256=m.PIN,
        immutable_diagnostics=work["methods_manifest"]["immutable_diagnostics"],
        methodology_note="Bind exactly fourteen full V747 task objects and the visible table against an independent digest. Authenticate pinned Exp8695 sealed producer execution without rerunning historical controllers. Preserve original V744 objectives and39+25 node partition. Independent branches own their eligible operands. Retain current disjoint child SQLite, JUnit, commands and exits before combining copies. Constructed authority checks establish administrative readiness only; no learning benefit, batch speed or physical acceleration is measured.",
    )
    value["source_artifact_hashes"] += (
        work["retained_authority"]["references"] + work["contract"]["references"]
    )
    value["field_principles"].update(
        {
            k: "Bind current administrative claims to authenticated original evidence; preserve scientific limits."
            for k in value
        }
    )
    value.pop("reproducibility_checksum")
    value["reproducibility_checksum"] = canonical_hash(value)
    return value
