"""REQ-REPORT-8260: reuse qualified publication with the current task identity.

The existing supervisor already records child clocks, stream hashes, deadlines
and process cleanup. Binding its hooks avoids copying that reporting stack.
"""

from __future__ import annotations

from pathlib import Path
import json
from typing import Any
from unittest.mock import patch

from carnot.reporting import gatemate_physical_delta_8260 as h
from carnot.reporting import gatemate_ledger_execution_8246 as qualified
from carnot.reporting.evidence_features_custody_7980 import checked, reference
from carnot.reporting.experiment_7303_validation_scope import CommandSpec
from carnot.reporting.request_trace_inventory_8200 import operand
from scripts.experiment_template import normalize_artifact_for_template_write

Json = dict[str, Any]
TEST = "tests/python/test_gatemate_physical_delta_8260.py"
OWNED = [
    "python/carnot/reporting/gatemate_physical_delta_8260.py",
    "python/carnot/reporting/gatemate_delta_execution_8260.py",
]
checksum = qualified.checksum
_commands = qualified.commands


def commands(private: Path) -> list[CommandSpec]:
    """The same private coverage, E2E and consumer checks now bind the owned files."""
    with (
        patch.object(qualified, "h", h),
        patch.object(qualified, "OWNED", OWNED),
        patch.object(qualified, "TEST", TEST),
    ):
        return _commands(private)


def normalize(value: Json) -> Json:
    """Current identity and code provenance must describe this actual invocation."""
    result: Json = normalize_artifact_for_template_write(value)
    result.update(
        experiment_id=8260,
        task_id="exp8260-gatemate-physical-delta",
        milestone="2026.10.713",
        schema="carnot.gatemate_physical_delta.v713.v1",
        exposure_scope="exposed development: historical GateMate custody and operator documents",
        methodology="Authenticated Exp8246 frontier and original transcript; qualified dry-run receipt parsing and document hash comparison. No model load or device execution.",
    )
    result["code_config_hashes"] += [
        reference(h.ROOT / p)
        for p in [
            "python/carnot/reporting/gatemate_change_ledger_8246.py",
            "python/carnot/reporting/gatemate_ledger_execution_8246.py",
        ]
    ]
    failures = [r for r in result["precondition_receipts"] if not r["passed"]]
    for failed in failures:
        result["gate_check_summary"].append(
            operand(
                "required_tools.normal_exit", Path(failed["stdout_path"]), 0, failed["actual_exit"]
            )
        )
    if failures:
        result.update(
            honest_verdict="complete_blocked_required_tools",
            verdict_class="blocked",
            gatemate_obligation_ready_score=0,
            execution_ready_score=0,
        )
    return result


def replay(path: Path) -> Json:
    """Recompute primitives and identity; external tool failures remain honest blocks."""
    value = json.loads(path.read_bytes())
    if (value["experiment_id"], value["task_id"], value["schema"], value["config"]) != (
        8260,
        "exp8260-gatemate-physical-delta",
        "carnot.gatemate_physical_delta.v713.v1",
        h.CONFIG,
    ):
        raise ValueError("identity_or_configuration_drift")
    for ref in (
        value["source_artifact_hashes"] + value["code_config_hashes"] + value["raw_shard_hashes"]
    ):
        checked(ref)
    data = json.loads(checked(value["replay_input_reference"]).read_bytes())
    h.verify_primitives(data)
    reduction = h.reduce(data)
    failed = (
        value["verdict_class"] == "disqualified"
        or value["honest_verdict"] == "complete_blocked_required_tools"
    )
    if failed:
        reduction["future_probe_contract"]["eligible"] = False
    for key, expected in reduction.items():
        if failed and key in {"honest_verdict", "verdict_class", "gatemate_obligation_ready_score"}:
            continue
        if value[key] != expected:
            raise ValueError("reduction_drift:" + key)
    if value["reproducibility_checksum"] != checksum(value):
        raise ValueError("checksum_drift")
    return dict(passed=True, replay_passed=True)


def main(argv: list[str] | None = None) -> int:
    """Scoped adapters leave all historical producer and terminal checks unchanged."""
    with (
        patch.object(qualified, "h", h),
        patch.object(qualified, "OWNED", OWNED),
        patch.object(qualified, "TEST", TEST),
        patch.object(qualified, "normalize_artifact_for_template_write", normalize),
        patch.object(qualified, "replay", replay),
    ):
        return qualified.main(argv)
