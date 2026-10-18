"""REQ-REPORT-8301: reuse qualified supervision and unchanged publication gates.

The adapter changes producer identity and the evidence reducer. The existing
runner still owns deadlines, child coverage, checks, replay and atomic writes.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import patch

from carnot.reporting import kv260_evidence_cost_boundary_8301 as h
from carnot.reporting import kv260_evidence_cost_execution_8287 as qualified
from carnot.reporting.evidence_features_custody_7980 import checked
from carnot.reporting.experiment_7303_validation_scope import CommandSpec

Json = dict[str, Any]
OWNED = [
    "python/carnot/reporting/kv260_evidence_cost_boundary_8301.py",
    "python/carnot/reporting/kv260_evidence_cost_execution_8301.py",
]
TEST = "tests/python/test_kv260_evidence_cost_boundary_8301.py"
READINESS = [
    "kv260_boundary_ready_score",
    "scoped_cpu_boundary_ready_score",
    "live_request_boundary_ready_score",
]
execute = qualified.execute
checksum = qualified.checksum
ORIGINAL_NORMALIZE = qualified.normalize_artifact_for_template_write
ORIGINAL_COMMANDS = qualified.commands
ORIGINAL_VALIDATORS = qualified.validators


def commands(private: Path) -> Any:
    """Freeze qualified coverage, strict types, E2E and consumer commands."""
    with (
        patch.object(qualified, "OWNED", OWNED),
        patch.object(qualified, "TEST", TEST),
        patch.object(qualified, "h", h),
    ):
        return ORIGINAL_COMMANDS(private)


def validators(path: Path) -> Any:
    """Retain the existing terminal auditors and use this real replay CLI."""
    with patch.object(qualified, "h", h):
        return ORIGINAL_VALIDATORS(path)


def normalize(value: Json) -> Json:
    """Correct invocation identity before the qualified publisher sees any bytes."""
    value.update(
        experiment_id=8301,
        task_id="exp8301-kv260-evidence-cost-boundary",
        milestone="2026.10.716",
        schema="carnot.kv260_evidence_cost_boundary.v716.v1",
    )
    if not value["required_checks_passed"] or value["verdict_class"] == "disqualified":
        for key in READINESS:
            value[key] = 0
    raw = Path(value["terminal_validation_sidecar_path"]).parent
    for name in [
        "docs/research-notes/v716-kv260-evidence-cost.md",
        "python/carnot/reporting/dependency_admission_execution_8291.py",
        "python/carnot/verify/dependency_scoped_admission_8291.py",
    ]:
        h.freeze(h.ROOT / name, raw, dict(references=value["code_config_hashes"]))
    value["field_principles"].update(
        scoped_cpu_boundary_ready_score="Qualifies current authenticated CPU fixture costs independently of live acquisition; no board timing or natural benefit.",
        live_request_boundary_ready_score="Requires complete same-invocation live request operands; absent spans are unavailable.",
        scoped_operation_rows="Preserve every CPU graph/arm/repeat and disjoint actual span; counters and typed checks have no isolated upstream clock.",
        cpu_condition_bounds="Each measured CPU condition has f=0 because no operation is an implemented quadratic k<=5 kernel.",
        cpu_graph_unit_count="Graph units are synthetic oracle fixtures; repeats and seeds add no independent natural sources.",
    )
    return dict(ORIGINAL_NORMALIZE(value))


def replay(path: Path) -> Json:
    """Rebuild from primitive bytes and reject rehashed summary tampering."""
    value = json.loads(path.read_bytes())
    if value["config"] != h.CONFIG:
        raise ValueError("configuration_drift")
    for ref in (
        value["source_artifact_hashes"] + value["code_config_hashes"] + value["raw_shard_hashes"]
    ):
        checked(ref)
    data = json.loads(checked(value["replay_input_reference"]).read_bytes())
    h.verify_cpu(data)
    measured = h.precision(data) if data.get("resources_available", True) else []
    if (
        measured
        != json.loads(checked(value["primitive_reference"]).read_bytes())["fixed_point_rows"]
    ):
        raise ValueError("primitive_drift")
    for key, expected in h.reduce(data, measured).items():
        if value["verdict_class"] == "disqualified" and key in [
            "honest_verdict",
            "verdict_class",
            *READINESS,
        ]:
            continue
        if value[key] != expected:
            raise ValueError("reduction_drift:" + key)
    if checksum(value) != value["reproducibility_checksum"]:
        raise ValueError("checksum_drift")
    return dict(passed=True, replay_passed=True)


def main(argv: list[str] | None = None) -> int:
    """Run unconditionally using the tested runner without any model load."""
    h.progress("start_no_model_load", 0, 1)
    with (
        patch.object(qualified, "h", h),
        patch.object(qualified, "OWNED", OWNED),
        patch.object(qualified, "TEST", TEST),
        patch.object(qualified, "execute", execute),
        patch.object(qualified, "commands", commands),
        patch.object(qualified, "validators", validators),
        patch.object(qualified, "replay", replay),
        patch.object(qualified, "normalize_artifact_for_template_write", normalize),
    ):
        return qualified.main(argv)
