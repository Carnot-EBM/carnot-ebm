"""REQ-REPORT-8582: bind fresh causal measurements to qualified original operands.

Only the small spline head learns. Qualification records execution; the exposed
historical labels cannot establish benefit on unseen questions or generators.
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import shutil
from typing import Any
from unittest.mock import patch

import numpy as np

from carnot.reporting import brier_online_learning_8517 as old
from carnot.reporting import causal_accounting_8581 as accounting
from carnot.reporting import causal_accounting_validation_8568 as validation
from carnot.reporting import v737_contract_methods as methods
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash
from carnot.reporting.primary_publication import read_bound_sidecar
from carnot.verify import objective_stream_8517 as stream

Json = dict[str, Any]
ROOT = old.ROOT
NAME, TASK = "experiment_8582_v737_causal_brier_stream", "exp8582-causal-brier-stream"
CLI = f"scripts/experiments/{NAME}.py"
TESTS = [
    "tests/python/test_causal_brier_stream_8582.py",
    "tests/python/test_objective_checkpoint_8461.py",
]
OWNED = [
    f"python/carnot/reporting/causal_brier_{p}_8582.py" for p in ("stream", "evidence", "runner")
] + [CLI]
MODEL_SPECS: list[Json] = []
ALIASES = dict(accounting.old.ALIAS_MAP, released_feedback_count="label_release_events")
_SOURCE = old.source
PINS = {
    methods.NAME: accounting.UPSTREAM[methods.NAME],
    accounting.NAME: "sha256:f60e202794f220440094c940ea5d5e84d3a70b7fc8ba840ecf09fb915bf1896c",
}
# The methods producer's pin was fixed by the qualified accounting authority.


def inspect(root: Path, raw: Path, private: Path) -> Json:
    """Authenticate original primary bytes and actual revision/publication consumers."""
    raw.mkdir(parents=True, exist_ok=True)
    private.mkdir(parents=True, exist_ok=True)
    refs, failures = [], []
    for name, field in [
        (methods.NAME, "methods_ready_score"),
        (accounting.NAME, "online_adapter_ready_score"),
    ]:
        path = root / "results" / (name + ".json")
        value: Json = {}
        try:
            ref = dict(path=str(path), sha256=PINS[name])
            value = old.read(ref)
            terminal = old.read(old.reference(Path(value["terminal_validation_sidecar_path"])))
            if not read_bound_sidecar(path, Path(terminal["publication"]["sidecar_path"]))[
                "report"
            ]["passed"]:
                raise ValueError("upstream_publication")
            if (
                value[field] != 1
                or not value["required_checks_passed"]
                or value["flagged_adversarial"]
            ):
                raise ValueError(field)
            if name == accounting.NAME:
                accounting.accept_revision(value["current_validator_reference"])
            refs.extend([ref, value["primitive_reference"]])
        except (OSError, ValueError, KeyError, TypeError) as error:
            failures.append(
                accounting.old.gate(
                    path,
                    field if str(error) == field else "upstream_authentication",
                    1 if str(error) == field else True,
                    value.get(field)
                    if str(error) == field
                    else None
                    if isinstance(error, FileNotFoundError)
                    else str(error),
                    upstream=name,
                )
            )
            failures[-1]["authentication_error"] = type(error).__name__ + ":" + str(error)
    resources = dict(
        old.report.methods.legacy.resources(private),
        free_disk_bytes=shutil.disk_usage(private).free,
        tools={
            n: shutil.which(str(ROOT / ".venv/bin" / n))
            for n in ("python", "pytest", "coverage", "ruff", "mypy")
        },
        measurement_cap_s=3600,
        validation_reserve_s=900,
        task_cap_s=4800,
        child_poll_s=30,
        model_loads=0,
        generations=0,
    )
    if (
        not resources["private_disk_backed_scratch"]
        or resources["available_memory_bytes"] < resources["minimum_memory_bytes"]
        or resources["free_disk_bytes"] < 104857600
        or not all(resources["tools"].values())
    ):
        failures.append(accounting.old.gate(private, "disk_memory_tools_deadline", True, resources))
    return dict(
        authority=dict(activated=not failures),
        failures=failures,
        preconditions=resources,
        source_refs=refs,
        external_receipts=[],
    )


def source() -> Json:
    """The original Exp8431 coefficients, scaler and temperature remain the starting state."""
    original = _SOURCE()
    registered = methods.protocol()["objective_updates"]
    if registered != old.methods.protocol()["objective_updates"]:
        raise ValueError("registered_stream_operands")
    fit = old.read(methods.protocol()["original_producers"]["8431"])
    heads = old.read(dict(path=fit["frozen_heads_path"], sha256=fit["frozen_heads_sha256"]))
    if original["heads"] != heads["heads"]["spline9"]:
        raise ValueError("original_head_scaler_temperature")
    return original


def measure(root: Path, raw: Path, private: Path) -> Json:
    """Execute all fixed arms through the existing sealed learner and evaluator children."""
    with (
        patch.multiple(
            old, NAME=NAME, TASK=TASK, CLI=CLI, OWNED=OWNED, inspect=inspect, source=source
        ),
        patch.object(stream, "CLI", CLI),
    ):
        work = dict(old.measure(root, raw, private))
    work.update(receipts=[], owned_ready=False, coverage_complete=False, union_passed=False)
    work["code_refs"] += [
        old.reference(Path(module.__file__))
        for module in (
            old,
            old.report,
            stream,
            stream.adapter,
            stream.shipped,
            accounting,
            accounting.old,
            methods,
        )
    ]
    reduced = old.reductions(work)
    atomic_json(raw / "coefficient_transitions.json", counters(work))
    atomic_json(raw / "released_feedback.json", reduced["release_rows"])
    atomic_json(raw / "retention_rows.json", reduced["retention_rows"])
    atomic_json(raw / "measurement.json", work)
    return work


def counters(work: Json) -> list[Json]:
    """Reconstruct committed assignments, keeping frozen releases separate from updates."""
    rows = []
    for cell in work["cells"]:
        if not cell.get("configuration"):
            continue
        cfg = old.read(cell["configuration"])
        for ref in cell["steps"]:
            step = old.read(ref)
            head = deepcopy(step["sealed_state"]["head"])
            for record, trace in zip(step["sealed_state"]["pending"], step["updates"], strict=True):
                before = deepcopy(head)
                writes = []
                if not cfg["frozen"]:
                    kernel = (
                        stream._LOG_UPDATE if cell["arm"] == "log_loss" else stream.adapter.update
                    )
                    head, info = kernel(head, np.asarray(record["x"]), trace["target"], 0.01)
                    writes = (
                        info["gradient_ids"] if "gradient_ids" in info else trace["gradient_ids"]
                    )
                row = accounting.old.shipped.coordinate_row(
                    before, head, writes, event=trace["feedback_id"], namespace=cell["identity"]
                )
                row.update(
                    applied=not cfg["frozen"],
                    admitted=True,
                    label_released=True,
                    source_reference=ref,
                    before_head_hash=canonical_hash(before),
                    after_head_hash=canonical_hash(head),
                )
                if row["after_head_hash"] != trace["after_head_hash"]:
                    raise ValueError("coefficient_transition")
                rows.append(row)
    return rows


def validated(work: Json) -> bool:
    """A claimed qualification must match actual coverage and exact terminal node evidence."""
    union = work.get("node_union", {})
    original = union.get("original_node_ids", [])
    terminal = union.get("terminal_node_outcomes", [])
    exact = (
        bool(original)
        and validation.exact_union(original, union.get("child_node_ids", []))
        and validation.exact_union(original, [[r["node_id"] for r in terminal]])
        and all(r["outcome"] == "passed" for r in terminal)
    )
    with patch.object(validation, "e", __import__(__name__, fromlist=["ready"])):
        return bool(exact and validation.ready(work))


def build(work: Json, output: Path) -> Json:
    """Readiness requires all owned checks, independent of favorable loss contrasts."""
    raw = Path(work["raw"])
    with patch.multiple(
        old,
        NAME=NAME,
        TASK=TASK,
        CLI=CLI,
        OWNED=OWNED,
        TEST=TESTS[0],
        REQUIRED_CHECKS=validation.REQUIRED,
    ):
        checked = dict(
            work,
            adversarial_findings=[
                f for f in work["adversarial_findings"] if f["severity"] == "critical"
            ],
        )
        value = dict(old.build(checked, work["receipts"], raw, output))
    events = counters(work)
    proof_events = [
        dict(
            row,
            **{
                side: {key: row[side][key] for key in ("parameters", "decay_scale")}
                for side in ("before", "after")
            },
        )
        for row in events
    ]
    metrics, proof = accounting.previous.proof(proof_events)
    transitions = old.reference(raw / "coefficient_transitions.json")
    proof["claims"] = {k: v for k, v in proof["claims"].items() if k in metrics}
    for alias, canonical in ALIASES.items():
        proof["claims"][alias] = deepcopy(proof["claims"][canonical])
    qualified = work["owned_ready"] and validated(work) and value["required_checks_passed"]
    ready = (
        qualified
        and value["objective_stream_ready_score"] == 1
        and work["independent_update"]["passed"]
        and value["pending_feedback_count"] == 0
    )
    verdict = (
        "disqualified"
        if not qualified
        else "blocked"
        if work["failures"]
        else "circular_positive"
        if ready
        else "disqualified"
    )
    value.update(
        experiment_id=8582,
        task_id=TASK,
        milestone="2026.10.737",
        run_date="20261016",
        honest_verdict=f"complete_{verdict}_v737_causal_brier_stream",
        verdict_class=verdict,
        inference_substrate="deterministic_cpu_sparse_spline_training_no_model_load",
        required_checks_passed=bool(qualified),
        flagged_adversarial=not bool(qualified),
        causal_learning_rows_ready_score=int(ready),
        learning_stream_ready_score=int(ready),
        objective_stream_ready_score=int(ready),
        metrics=metrics,
        **metrics,
        discrete_count_proof=proof,
        alias_map=ALIASES,
        adversarial_findings=work["adversarial_findings"],
        coefficient_change_count=metrics["changed_coordinate_count"],
        prediction_rows_reference=value["stream_rows_reference"],
        released_feedback_reference=old.reference(raw / "released_feedback.json"),
        coefficient_transition_rows=[
            dict(
                namespace=row["namespace"],
                event_id=row["event_id"],
                before_head_hash=row["before_head_hash"],
                after_head_hash=row["after_head_hash"],
                primitive=dict(transitions, json_pointer=f"/{index}"),
            )
            for index, row in enumerate(events)
        ],
        coefficient_transitions_reference=transitions,
        canonical_counter_rows=[
            dict(name=key, value=n, definition=accounting.old.DEFINITIONS[key])
            for key, n in metrics.items()
        ],
        retention_rows_reference=old.reference(raw / "retention_rows.json"),
        checkpoint_restart_rows=work["restart_rows"],
        complete_learning_cost=dict(
            cells=[
                dict(
                    identity=c["identity"],
                    inclusive_wall_s=next(
                        r["duration_s"]
                        for r in work["process_receipts"]
                        if r["name"] == c["identity"]
                    ),
                    measured_update_storage=old.read(c["costs"]),
                )
                for c in work["cells"]
                if c.get("costs")
            ],
            includes_prediction=True,
            includes_update=True,
            includes_storage=True,
            includes_final_feedback_drain=True,
            pending_feedback=value["pending_feedback_count"],
            scope="inclusive real worker wall time includes evaluator children and audit I/O; not an exclusive kernel sum",
        ),
        historical_quarantines=dict(
            exp8529="source_null_unchanged",
            exp8553="label_rejection_unchanged",
            exp8559="cold_cost_null_unchanged",
            exp8568="old_revision_rejection_unchanged",
            every_prior_quarantine="unchanged",
        ),
        future_dependencies=[dict(experiment_id=8583, status="future_dependency_not_observation")],
        conditional_evidence=True,
        methodology_note="FR-11 small-head continuous self-learning; original Exp8431 state. Uniform capacity8, rate0.01, L2.001. Calibeating and finite delayed capacity motivate controls, with no inherited theorem. Later sealed typed decisions use committed heads. Exposed-history execution only; benefit belongs to Exp8583.",
    )
    for descriptor in ("independent_generalization", "favorable_contrast_required"):
        value["acceptance_gates"].pop(descriptor)
    value["acceptance_gates"].update(
        count_proof_validation=not any(
            f["severity"] == "critical" for f in work["adversarial_findings"]
        )
    )
    value["acceptance_gates"].update(
        current_owned_validation=bool(qualified), causal_histories=bool(ready)
    )
    value["raw_shard_hashes"] += [
        old.reference(raw / name)
        for name in (
            "released_feedback.json",
            "retention_rows.json",
            "coefficient_transitions.json",
        )
    ]
    value["code_config_hashes"] += [old.reference(ROOT / p) for p in TESTS]
    value["field_principles"].update(
        {
            k: "Immutable current invocation evidence; readiness is execution, not independent learning benefit."
            for k in value
            if k not in value["field_principles"]
        }
    )
    value["field_principles"].update(
        metrics="Independent qualified integer reductions of assignment lists and before/after states.",
        complete_learning_cost="Inclusive measured process time covers prediction, all releases, durable writes and final drain; nested spans must not be added to it.",
        causal_learning_rows_ready_score="Complete validated histories qualify regardless of benefit; repeated seeds and siblings remain dependent.",
    )
    value["reproducibility_checksum"] = old.checksum(value)
    return value


def replay(path: Path) -> bool:
    """Use the independent primitive reconstruction for both CLI and publication."""
    from carnot.reporting.causal_brier_evidence_8582 import replay as reconstruct

    return reconstruct(path)
