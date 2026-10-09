"""REQ-REPORT-8344: bind current identity to the qualified no-probe publisher.

Reusing the existing supervisor preserves bounded children, receipts, private
checks and atomic publication instead of introducing another validator stack.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import patch

from carnot.reporting import gatemate_change_ledger_8344 as h
from carnot.reporting import gatemate_ledger_execution_8330 as previous
from carnot.reporting import v718_replay_history as policy
from carnot.reporting.evidence_features_custody_7980 import checked, reference
from carnot.reporting.experiment_7303_validation_scope import CommandSpec

Json = dict[str, Any]
TEST = "tests/python/test_gatemate_change_ledger_8344.py"
OWNED = [
    "python/carnot/reporting/gatemate_change_ledger_8344.py",
    "python/carnot/reporting/gatemate_ledger_execution_8344.py",
]
checksum, execute = previous.checksum, previous.execute
_commands, _normalize = previous.commands, previous.normalize


def commands(private: Path) -> list[CommandSpec]:
    """The frozen scoped plan qualifies finding consumption before terminal use."""
    with (
        patch.object(previous, "h", h),
        patch.object(previous, "OWNED", OWNED),
        patch.object(previous, "TEST", TEST),
    ):
        return _commands(private)


def normalize(value: Json) -> Json:
    """Current work keeps immutable historical synthesis distinct from new evidence."""
    with patch.object(previous, "h", h):
        result = _normalize(value)
    result.update(
        experiment_id=8344,
        task_id="exp8344-gatemate-change-ledger",
        milestone="2026.10.719",
        schema="carnot.gatemate_change_ledger.v719.v1",
        methodology="Authenticated immutable Exp8330 primary and terminal sidecars, directly activated V719 authority and original 0xffffffff transcript. Qualified physical-change parsing after the Exp8330 end-clock frontier. Zero model, JTAG or board calls; missing observations are external obligations, not H1/H2 nulls.",
        historical_model_provenance=[],
    )
    result["code_config_hashes"] += [
        reference(h.ROOT / p)
        for p in [
            "python/carnot/reporting/gatemate_change_ledger_8330.py",
            "python/carnot/reporting/gatemate_ledger_execution_8330.py",
            "python/carnot/reporting/v719_contract_replay.py",
        ]
    ]
    for gate in result["gate_check_summary"]:
        gate["upstream"] = Path(gate["path"]).stem
    return result


def replay(path: Path) -> Json:
    """Cold reduction authenticates bytes and rejects rehashed physical claims."""
    value = json.loads(path.read_bytes())
    if (value["experiment_id"], value["task_id"], value["schema"], value["config"]) != (
        8344,
        "exp8344-gatemate-change-ledger",
        "carnot.gatemate_change_ledger.v719.v1",
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
    """Scoped adapters preserve all unchanged private and terminal enforcement."""
    with (
        patch.object(previous, "h", h),
        patch.object(previous, "OWNED", OWNED),
        patch.object(previous, "TEST", TEST),
        patch.object(previous, "commands", commands),
        patch.object(previous, "normalize", normalize),
        patch.object(previous, "replay", replay),
    ):
        return previous.main(argv)
