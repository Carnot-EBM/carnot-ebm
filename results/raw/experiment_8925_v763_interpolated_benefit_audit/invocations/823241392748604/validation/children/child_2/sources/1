"""REQ-REPORT-8925: adapt checked publication without cloning the prior runner.

The original runner keeps validation, private E2E checks and atomic publication.
This module binds the new identity and independently cold-replays its evidence.
"""

import json
import shutil
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from carnot.reporting import guarded_benefit_runner_8911 as prior
from carnot.reporting import interpolated_benefit_audit_8925 as m
from carnot.reporting.current_work_receipt import (
    canonical_hash,
    sha256_file,
    validate_current_work_receipt,
)
from carnot.reporting.v746_retained_evidence import reference, load
from carnot.reporting.v748_collection_contract import namespace

from scripts.experiment_template import ExperimentTemplate

Json = dict[str, Any]
runtime = namespace(prior, "interpolated8925_publication")
_original_seal = runtime.seal
_original_validate, _original_terminal = runtime.validate, runtime.terminal


def validate(raw: Path, private: Path) -> Json:
    """Measure new owned statements and execute unchanged private E2E-018/021 controls."""
    return dict(_original_validate(raw, private))


def terminal(candidate: Path, raw: Path) -> list[Json]:
    """Run unchanged adversarial and row scanners plus the actual foreign-directory CLI."""
    return list(_original_terminal(candidate, raw))


def semantics(value: Json) -> tuple[bool, str]:
    """A complete null qualifies; external absence blocks and owned failures disqualify."""
    ready = bool(
        value["preconditions_checked"]["external_ready"]
        and value["preconditions_checked"]["resources"]["passed"]
        and runtime.qualification(value)
        and not value["owned_errors"]
        and value["completed_count"] == 18432
    )
    verdict = (
        "blocked"
        if not value["preconditions_checked"]["external_ready"]
        else "circular_positive"
        if ready and value["interpolated_benefit_score"]
        else "null"
        if ready
        else "disqualified"
    )
    return ready, verdict


def seal(value: Json, started: int, spans: list[Json]) -> None:
    """Bind current identity and provenance while the inherited runner records real clocks."""
    value["interpolated_benefit_score"] = value.pop(
        "guarded_benefit_score", value["interpolated_benefit_score"]
    )
    value.update(
        experiment_id=8925,
        milestone="2026.11.763",
        honest_verdict="complete_" + value["verdict_class"] + "_interpolated_benefit_audit",
    )
    value["cited_upstream_artifacts"] = [
        dict(
            experiment_id=8922,
            task_id=m.causal.TASK,
            sha256=m.PIN,
            fields_imported=["causal_rows_ready_score", "rows", "stream_refs"],
        ),
        dict(
            experiment_id=8919,
            task_id=m.causal.methods.TASK,
            sha256=m.causal.PIN,
            fields_imported=["methods_ready_score", "candidate_ready_score", "protocol"],
        ),
    ]
    reused = [
        "python/carnot/reporting/guarded_benefit_audit_8911.py",
        "python/carnot/reporting/guarded_benefit_runner_8911.py",
    ]
    for name in reused:
        ref = reference(m.ROOT / name)
        if ref not in value["code_config_hashes"]:
            value["code_config_hashes"].append(ref)
    health_path = Path("/tmp/carnot8925-health/receipt.json")
    if health_path.is_file() and "receipt" not in value["repository_health"]:
        health = json.loads(health_path.read_bytes())
        retained = []
        for path in [
            health_path,
            Path("/tmp/carnot8925-tests-first.log"),
            Path(health["stdout_path"]),
            Path(health["stderr_path"]),
        ]:
            target = Path(value["raw_directory"]) / path.name
            shutil.copyfile(path, target)
            retained.append(reference(target))
        value["raw_shard_hashes"] += retained
        value["repository_health"] = dict(
            command=".venv/bin/pytest tests/python -q",
            scope="Separate bounded repository health; never owned qualification",
            passed=health["passed"],
            exit_code=health["exit_code"],
            receipt=retained[0],
            logs=retained[2:],
        )
    _original_seal(value, started, spans)
    value["inference_substrate"] = value["current_work_receipt"]["inference_substrate"] = (
        "aggregation_from_upstream_artifacts"
    )
    value["field_principles"].pop("guarded_benefit_score", None)
    value["field_principles"].update(
        interpolated_benefit_score="All nine original simultaneous bounds, complete paired evidence and owned validity checks; independent natural-language benefit remains unestablished.",
        retention_bounds="Untouched Brier and typed-error increases at0/24/48 versus frozen.",
        operation_counts="Original Exp8922 CPU learning, guard and prediction work, distinct from zero current audit training.",
        information_controls="Original producer assertions plus evaluator-label joins; mutation rejection is exercised by owned tests.",
        promotion_rows="All original admission attempts including every rejected gradient; descriptive only.",
        continuation_decision="Retirement covers this fixed interpolated optimizer, guard and schedule only; continuous learning in general remains open.",
    )
    value["reproducibility_checksum"] = canonical_hash(
        {k: v for k, v in value.items() if k != "reproducibility_checksum"}
    )


def replay(path: Path) -> Json:
    """Cold-reduce pinned operands so a repaired checksum cannot authorize false claims."""
    value = json.loads(path.read_bytes())
    checksum = value.pop("reproducibility_checksum")
    m.require(canonical_hash(value) == checksum, "artifact_checksum")
    m.require(
        (value["experiment_id"], value["task_id"], value["milestone"], value["run_date"])
        == (8925, m.TASK, "2026.11.763", "20261108"),
        "producer_identity",
    )
    refs = value["source_artifact_hashes"] + value["code_config_hashes"] + value["raw_shard_hashes"]
    for index, ref in enumerate(refs):
        m.require(sha256_file(Path(ref["path"])) == ref["sha256"], "operand_hash")
        if index % 100 == 0:
            m.progress("replay_authenticate", index, len(refs) - index)
    m.require(
        value["producer_invocation"]["executed_sources"]
        == [reference(m.ROOT / n) for n in m.OWNED],
        "executed_source_drift",
    )
    m.require(
        value["source_artifact_hashes"] == value["preconditions_checked"]["references"],
        "source_inventory",
    )
    m.require(value["acceptance_gates"] == m.GATES, "acceptance_drift")
    receipt = value["current_work_receipt"]
    m.require(
        not validate_current_work_receipt(receipt)
        and value["numerical_training_calls"] == 0
        and value["duration_s"] == receipt["duration_s"]
        and value["phase_spans"] == receipt["phase_spans"]
        and value["model_invocation_counts"] == receipt["invocation_counts"]
        and value["inference_substrate_class"] == receipt["inference_substrate_class"]
        and value["inference_substrate"] == receipt["inference_substrate"],
        "compute_scope",
    )
    pre = value["preconditions_checked"]
    if pre["external_ready"] and pre["resources"]["passed"] and not value["owned_errors"]:
        source = Path(value["raw_directory"]) / (m.SOURCE + ".json")
        m.require(sha256_file(source) == m.PIN, "original_source_pin")
        expected = m.reduce(load(reference(source)))
    else:
        expected = m.empty(value["censor_reason"])
    ready, verdict = semantics(value)
    if not ready:
        expected["interpolated_benefit_score"] = 0
    for key, observed in expected.items():
        m.require(value[key] == observed, "reduction_drift:" + key)
    m.require(
        value["required_checks_passed"] == ready
        and type(value["audit_ready_score"]) is int
        and value["audit_ready_score"] == int(ready)
        and type(value["interpolated_benefit_score"]) is int
        and value["verdict_class"] == verdict
        and value["honest_verdict"] == "complete_" + verdict + "_interpolated_benefit_audit",
        "terminal_semantics",
    )
    m.require(verdict != "blocked" or value["gate_check_summary"], "blocked_diagnostic")
    m.require(
        value["verifier_is_oracle"] is True
        and value["independent_generalization_score"] == 0
        and value["MODEL_SPECS"] == [],
        "oracle_boundary",
    )
    return dict(passed=True, status="replay_passed", sha256=sha256_file(path))


def _legacy_score(value: Json) -> Json:
    """Confine the prior runner's internal score spelling to its private interface."""
    return dict(value, guarded_benefit_score=value["interpolated_benefit_score"])


def main(argv: list[str] | None = None) -> int:
    """Use the existing runner with private bindings and the current cold validator."""
    adapter = SimpleNamespace(**vars(m))
    adapter.reduce = lambda source: _legacy_score(m.reduce(source))
    adapter.empty = lambda reason: _legacy_score(m.empty(reason))
    runtime.ExperimentTemplate = lambda _id, _title, deliverable, **kwargs: ExperimentTemplate(
        8925, "Interpolated benefit audit", deliverable, **kwargs
    )
    runtime.m, runtime.methods = adapter, m.causal.methods
    runtime.seal, runtime.semantics = seal, semantics
    runtime.replay, runtime.validate, runtime.terminal = replay, validate, terminal
    return int(runtime.main(argv))
