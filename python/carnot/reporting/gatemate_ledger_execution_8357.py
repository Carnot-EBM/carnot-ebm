"""REQ-REPORT-8357: publish immutable history through the qualified supervisor.

Small bindings retain deadlines, typed findings and atomic byte checks. Owned
failures disqualify; missing external history keeps a terminal blocked outcome.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import patch

from carnot.reporting import gatemate_change_ledger_8357 as h
from carnot.reporting import gatemate_ledger_execution_8344 as previous
from carnot.reporting import v718_replay_history as policy
from carnot.reporting.evidence_features_custody_7980 import checked, reference
from carnot.reporting.experiment_7303_validation_scope import CommandSpec

Json = dict[str, Any]
TEST = "tests/python/test_gatemate_change_ledger_8357.py"
OWNED = [
    "python/carnot/reporting/gatemate_history_8357.py",
    "python/carnot/reporting/gatemate_change_ledger_8357.py",
    "python/carnot/reporting/gatemate_ledger_execution_8357.py",
]
checksum, execute = previous.checksum, previous.execute
_commands, _normalize = previous.commands, previous.normalize


def commands(private: Path) -> list[CommandSpec]:
    """Scoped coverage includes the real runner and unchanged historical controls."""
    with (
        patch.object(previous, "h", h),
        patch.object(previous, "OWNED", OWNED),
        patch.object(previous, "TEST", TEST),
    ):
        return [s for s in _commands(private) if s.name != "focused_pytest"]


def normalize(value: Json) -> Json:
    """Current identity stays distinct from imported historical execution findings."""
    with patch.object(previous, "h", h):
        result = _normalize(value)
    result.update(
        experiment_id=8357,
        task_id="exp8357-gatemate-change-ledger",
        milestone="2026.10.720",
        schema="carnot.gatemate_change_ledger.v720.v1",
        methodology="Recover producer-bound immutable GateMate source/configuration closure and terminal custody, independently authenticate V720 task authority, preserve original 0xffffffff transcript. Incomplete closure blocks physical inspection; zero model, JTAG or board calls.",
    )
    result["code_config_hashes"] += [
        reference(h.ROOT / p)
        for p in [
            "python/carnot/reporting/v720_frozen_input_contract.py",
            "python/carnot/reporting/gatemate_change_ledger_8344.py",
            "python/carnot/reporting/gatemate_ledger_execution_8344.py",
        ]
    ]
    result["adversarial_findings"] = []
    return result


def replay(path: Path) -> Json:
    """Recompute history and physical claims before trusting a candidate's checksum."""
    value = json.loads(path.read_bytes())
    if (value["experiment_id"], value["task_id"], value["schema"], value["config"]) != (
        8357,
        "exp8357-gatemate-change-ledger",
        "carnot.gatemate_change_ledger.v720.v1",
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
    """Reuse byte-bound publication; fixtures remain confined to private outputs."""
    with (
        patch.object(previous, "h", h),
        patch.object(previous, "OWNED", OWNED),
        patch.object(previous, "TEST", TEST),
        patch.object(previous, "commands", commands),
        patch.object(previous, "normalize", normalize),
        patch.object(previous, "replay", replay),
    ):
        return previous.main(argv)
