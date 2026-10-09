"""REQ-REPORT-8330: reuse bounded publication without repeating board execution.

The qualified runner owns child deadlines, stream hashes and atomic publication.
This small adapter binds the new identity and consumes unchanged verifier reports.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import patch

from carnot.reporting import gatemate_change_ledger_8330 as h
from carnot.reporting import gatemate_obligation_execution_8316 as previous
from carnot.reporting import v718_replay_history as policy
from carnot.reporting.evidence_features_custody_7980 import checked, reference
from carnot.reporting.experiment_7303_validation_scope import CommandSpec

Json = dict[str, Any]
TEST = "tests/python/test_gatemate_change_ledger_8330.py"
OWNED = [
    "python/carnot/reporting/gatemate_change_ledger_8330.py",
    "python/carnot/reporting/gatemate_ledger_execution_8330.py",
]
checksum = previous.checksum
_commands, _normalize = previous.commands, previous.normalize
_execute = previous._execute


def commands(private: Path) -> list[CommandSpec]:
    """Coverage includes real CLI children, history lifecycle and unchanged consumers."""
    with (
        patch.object(previous, "h", h),
        patch.object(previous, "OWNED", OWNED),
        patch.object(previous, "TEST", TEST),
    ):
        return _commands(private)


def execute(plan: list[CommandSpec], raw: Path) -> list[Json]:
    """Parse byte-bound findings; a raw nonzero exit alone never grants acceptance."""
    if any(s.name == "repository_health_once" for s in plan):
        h.progress("repository_health_owned_by_conductor")
        return []
    receipts = _execute(plan, raw)
    for spec, receipt in zip(plan, receipts, strict=True):
        if spec.name == "adversarial":
            candidate = Path(spec.argv[-1])
            try:
                report = dict(
                    json.loads(Path(receipt["stdout_path"]).read_bytes()),
                    candidate_sha256=reference(candidate)["sha256"],
                    verifier_sha256=policy.verifier_hash(),
                )
            except (OSError, ValueError, TypeError):
                report = {}
            found = policy.consume(report, candidate, receipt["actual_exit"], {})
            receipt.update(
                adversarial_policy=found, passed=found["passed"] and receipt["normal_exit"]
            )
    return receipts


def normalize(value: Json) -> Json:
    """Identity describes current work while historical provenance stays imported."""
    with patch.object(previous, "h", h):
        result = _normalize(value)
    result.update(
        experiment_id=8330,
        task_id="exp8330-gatemate-change-ledger",
        milestone="2026.10.718",
        schema="carnot.gatemate_change_ledger.v718.v1",
        adversarial_findings=[],
        finding_consumer_policy=policy.POLICY,
        methodology="Authenticated Exp8316, activated V718 authority and original 0xffffffff transcript; qualified dry-run physical receipt parsing after the frozen end-clock frontier. Zero model, board or JTAG calls.",
    )
    result["code_config_hashes"] += [
        reference(h.ROOT / p)
        for p in [
            "python/carnot/reporting/gatemate_obligation_8316.py",
            "python/carnot/reporting/gatemate_obligation_execution_8316.py",
            "python/carnot/reporting/v718_contract_replay.py",
            "python/carnot/reporting/v718_replay_history.py",
        ]
    ]
    for gate in result["gate_check_summary"]:
        gate["upstream"] = (
            "exp8330-gatemate-change-ledger"
            if gate["artifact_field"].startswith("required_tools")
            else "exp8316-gatemate-obligation"
            if "8316" in str(gate.get("artifact_path"))
            else "V718 operator/authority evidence"
        )
    return result


def replay(path: Path) -> Json:
    """Recompute every reduced field so rehashed claims cannot invent board success."""
    value = json.loads(path.read_bytes())
    if (value["experiment_id"], value["task_id"], value["schema"], value["config"]) != (
        8330,
        "exp8330-gatemate-change-ledger",
        "carnot.gatemate_change_ledger.v718.v1",
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
    """Reuse the qualified CLI; private fixtures cannot replace production evidence."""
    with (
        patch.object(previous, "h", h),
        patch.object(previous, "OWNED", OWNED),
        patch.object(previous, "TEST", TEST),
        patch.object(previous, "commands", commands),
        patch.object(previous, "execute", execute),
        patch.object(previous, "normalize", normalize),
        patch.object(previous, "replay", replay),
    ):
        return previous.main(argv)
