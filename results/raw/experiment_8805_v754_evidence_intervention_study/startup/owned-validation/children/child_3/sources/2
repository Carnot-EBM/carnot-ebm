"""REQ-VERIFY-8805: retain input-only execution, actual coverage and terminal validation."""

import argparse
from datetime import UTC, datetime
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import time
from types import SimpleNamespace
from typing import Any

from carnot.reporting import evidence_intervention_study_8805 as m
from carnot.reporting import repaired_evidence_runner_8803 as previous
from carnot.reporting import evidence_evaluation_runner_8779 as shared
from carnot.reporting.current_work_receipt import (
    ZERO_INVOCATION_COUNTS,
    atomic_json,
    canonical_hash,
)
from carnot.reporting.primary_publication import publish_primary
from carnot.reporting.v709_execution import child
from carnot.reporting.v746_retained_evidence import reference, load
from carnot.reporting.v748_collection_contract import namespace

Json = dict[str, Any]


def now() -> str:
    """Real UTC phase clocks document execution without synthetic duration padding."""
    return datetime.now(UTC).isoformat()


def preflight(private: Path, upstream: Path | None = None) -> Json:
    """Reuse custody checks once, then project a measured unit plus the validation reserve."""
    start = time.monotonic()
    pre = previous.preflight(private)
    admitted = m.authenticate(upstream or m.UPSTREAM)
    pre["authentication_8803"] = admitted
    pre["checks"]["qualified_8803"] = admitted["passed"]
    pre["diagnostics"] += admitted["diagnostics"]
    setup = time.monotonic() - start
    unit = 0.0
    if admitted["passed"]:
        sample = m.cohort(admitted["producer"])["inputs"][:1]
        probe = time.monotonic()
        m.measure(sample)
        unit = time.monotonic() - probe
    projection = setup + unit * 3360 + 900
    pre["checks"]["projected_time"] = projection < 4800
    pre["passed"] = all(pre["checks"].values())
    pre["resource_projection"] = dict(
        setup_seconds_once=setup,
        measured_probe_seconds=unit,
        probe_units=1,
        projected_work_seconds=unit * 3360,
        reserved_validation_seconds=900,
        projected_total_seconds=projection,
    )
    return pre


def unqualified() -> Json:
    """Private execution cannot manufacture a passing production validation receipt."""
    return dict(
        receipts=[dict(name="private_unqualified", passed=False)],
        union=dict(passed=False),
        coverage=dict(complete=False),
        raw_references=[],
    )


def validate(raw: Path, private: Path) -> Json:
    """Reuse exact collection and SQLite combination, including unchanged historical tests and readers."""
    validator = namespace(shared, "intervention8805_validation")
    validator.e = SimpleNamespace(
        ROOT=m.ROOT, OWNED=m.OWNED, TESTS=m.TESTS + m.up.TESTS + m.up.old.TESTS
    )
    return dict(validator.validate(raw, private))


def build(work: Json, validation: Json) -> Json:
    """Reconstruct every scientific aggregate from sealed rows; readiness does not require benefit."""
    raw = Path(work["raw"])
    decisions = load(reference(raw / "decisions.json"))["rows"]
    labels = load(reference(raw / "labels.json"))["rows"]
    all_rows = m.score(labels, decisions)
    rows = [r for r in all_rows if r["scope"] == "scheduled" and r["arm"] in m.ARMS]
    controls = [r for r in all_rows if r["scope"] == "valid_alternative" and r["arm"] in m.ARMS]
    strong = [r for r in all_rows if r["arm"] == "unlimited_exact_reference"]
    summary = m.summarize(rows, controls)
    pre = work["preconditions_checked"]["passed"]
    checks = (
        bool(validation["receipts"])
        and all(r["passed"] for r in validation["receipts"])
        and validation["union"]["passed"]
        and validation["coverage"]["complete"]
    )
    gates = dict(
        preconditions=pre,
        qualified_source=work["preconditions_checked"]
        .get("authentication_8803", {})
        .get("passed", False)
        and bool(work["source_artifact_hashes"])
        and work["source_artifact_hashes"][0]["sha256"] == m.PIN,
        scoped_validation=bool(checks),
        complete_schedule=len(rows) == 8640
        and len({r["cluster_id"] for r in rows}) == 120
        and all(r["status"] == "completed" for r in rows),
        alternative_controls=len(controls) == 1440,
        input_only_execution=bool(work["measurement_receipts"])
        and all(r["passed"] for r in work["measurement_receipts"]),
        exact_ceiling=len(strong) == 3360 and all(r["metric"] for r in strong),
    )
    ready = all(gates.values())
    verdict = (
        "blocked"
        if not pre
        else "disqualified"
        if not ready
        else "circular_positive"
        if summary["method_benefit"]
        else "null"
    )
    value: Json = dict(
        experiment_id=8805,
        task_id=m.TASK,
        milestone="2026.11.754",
        run_date=work["run_date"],
        honest_verdict="complete_" + verdict + "_constructed_evidence_interventions",
        verdict_class=verdict,
        inference_substrate="deterministic_verifier",
        inference_substrate_class="no_model_load",
        MODEL_SPECS=[],
        model_invocation_counts=dict(ZERO_INVOCATION_COUNTS),
        numerical_training_calls=0,
        verifier_is_oracle=True,
        duration_s=work["duration_s"],
        random_seed=m.SEED,
        source_artifact_hashes=work["source_artifact_hashes"],
        code_config_hashes=work["code_config_hashes"],
        raw_shard_hashes=work["raw_shard_hashes"] + validation["raw_references"],
        preconditions_checked=work["preconditions_checked"],
        phase_spans=work["phase_spans"],
        rows=rows,
        intended_count=8640,
        completed_count=sum(r["status"] == "completed" for r in rows),
        censored_count=8640 - len(rows),
        failed_count=sum(r["status"] == "failed" for r in rows),
        excluded_count=0,
        independent_count=len({r["cluster_id"] for r in rows}),
        n_samples=len({r["cluster_id"] for r in rows}),
        gate_check_summary=work["preconditions_checked"].get("diagnostics", []),
        acceptance_gates=gates,
        required_checks_passed=bool(checks),
        owned_statement_coverage=validation["coverage"],
        validation_receipts=validation["receipts"],
        evidence_rows_ready_score=int(ready),
        exposure_scope="Sealed Exp8803 constructed intervention worlds; exposed V751-V753 methods. No natural extraction, hallucination or hidden-game transfer claim.",
        independent_generalization_score=0,
        constructed_evidence_only=True,
        valid_alternative_control_rows=controls,
        strong_solver_cost_rows=strong,
        strong_reference_ceiling=dict(
            denominator=len(strong),
            correct=sum(r["metric"] for r in strong),
            verifier_is_oracle=True,
            verdict_class="circular_positive",
            execution_grounded=True,
        ),
        primitive_budget_definition=dict(
            primitive="Python LINE events inside unchanged reporting solvers; external initialization and C builtin internals excluded",
            limit=m.up.BUDGET,
            equal_allowance_for=list(m.ARMS),
            unlimited_reference_separate=True,
            actual_consumption_reported=True,
            machine_instruction_equivalence=False,
        ),
        primitive_reference=reference(raw / "work.json"),
        validation_reference=reference(raw / "validation.json"),
        cited_upstream_artifacts=[
            dict(
                experiment_id=8803,
                fields_imported=[
                    "evidence_interventions_ready_score",
                    "cohort_references",
                    "primitive_budget_definition",
                ],
                **r,
            )
            for r in work["source_artifact_hashes"][:1]
        ],
        methodology_note="Missing evidence stays unknown. Four predeclared primary families use whole-world bootstrap draws. Exact reference gains are circular; no model load, generation or numerical training.",
        repository_health_reference=work.get("repository_health_reference"),
        global_reconciliation_debt=dict(
            scope="repository_health",
            reference=work.get("repository_health_reference"),
            note="One bounded full-suite run reported external failures and reached its 600-second limit. It is separate from owned acceptance. Reconciliation docs are delegated to the conductor.",
        ),
        **summary,
    )
    value["field_principles"] = {
        k: "Measured or reconstructed from exact sealed operands. Sibling conditions share one independent world."
        for k in value
    }
    value["field_principles"].update(
        field_principles="Every additional top-level field is annotated here.",
        reproducibility_checksum="Canonical hash of all other artifact fields.",
        evidence_rows_ready_score="Complete validated measurement readiness, independent of method benefit.",
        strong_solver_cost_rows="Unlimited exact checks have separate measured cost and never enter paid-method denominators.",
    )
    value["reproducibility_checksum"] = canonical_hash(value)
    return value


def seal(raw: Path, value: Json) -> None:
    """An independent immutable operand seal prevents repaired summary hashes from changing provenance."""
    atomic_json(
        raw / "independent_seal.json",
        {k: value[k] for k in ["primitive_reference", "validation_reference"]},
    )


def replay(path: Path) -> bool:
    """Cold execution checks exact source bytes, replays decisions and reconstructs all aggregates."""
    try:
        value = json.loads(path.read_bytes())
        raw = Path(value["primitive_reference"]["path"]).parent
        if load(reference(raw / "independent_seal.json")) != {
            k: value[k] for k in ["primitive_reference", "validation_reference"]
        }:
            return False
        work, validation = load(value["primitive_reference"]), load(value["validation_reference"])
        for ref in (
            work["code_config_hashes"]
            + work["source_artifact_hashes"]
            + work["raw_shard_hashes"]
            + validation["raw_references"]
        ):
            if reference(Path(ref["path"])) != ref:
                return False
        if work["preconditions_checked"]["passed"]:
            inputs = load(reference(raw / "inputs.json"))["rows"]
            decisions = load(reference(raw / "decisions.json"))["rows"]
            if work["source_artifact_hashes"]:
                admitted = m.authenticate(Path(work["source_artifact_hashes"][0]["path"]))
                if not admitted["passed"]:
                    return False
                expected = m.cohort(admitted["producer"])
                labels = load(reference(raw / "labels.json"))["rows"]
                if inputs != expected["inputs"] or labels != expected["labels"]:
                    return False
            if not m.audit(inputs, decisions):
                return False
        return bool(value == build(work, validation))
    except (OSError, ValueError, KeyError, TypeError):
        return False


def terminal(candidate: Path, raw: Path) -> Json:
    """Keep the shared scanners unchanged and replay outside the checkout without PYTHONPATH."""
    validator = namespace(previous, "intervention8805_terminal")
    validator.m = m
    return dict(validator.terminal(candidate, raw))


def main(argv: list[str] | None = None) -> int:
    """Run the sealed input-only study and publish only a validated terminal candidate."""
    m.progress("start_no_model_load")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", choices=["20261104"], default="20261104")
    parser.add_argument("--output", type=Path, default=m.ROOT / "results" / (m.NAME + ".json"))
    parser.add_argument("--upstream", type=Path, default=m.UPSTREAM)
    parser.add_argument("--private-control", action="store_true")
    parser.add_argument("--cold-replay", type=Path)
    parser.add_argument("--decide-input", type=Path)
    parser.add_argument("--decisions", type=Path)
    args = parser.parse_args(argv)
    if args.cold_replay:
        passed = replay(args.cold_replay)
        m.progress("replay_passed" if passed else "replay_rejected")
        return int(not passed)
    if args.decide_input:
        if args.decisions is None:
            parser.error("--decisions required")
        inputs = json.loads(
            args.decide_input.read_bytes(),
            object_pairs_hook=m.up.original.unique_object,
            parse_constant=m.up.original.reject_constant,
        )["rows"]
        atomic_json(args.decisions, dict(rows=m.measure(inputs)))
        return 0
    output = args.output.absolute()
    if args.private_control and output.resolve().is_relative_to(m.ROOT.resolve()):
        parser.error("private output must be outside the checkout")
    raw = output.parent / "raw" / m.NAME / "invocations" / str(time.time_ns())
    raw.mkdir(parents=True)
    start, clock = now(), time.monotonic()
    with TemporaryDirectory(prefix="exp8805-private-", dir="/var/tmp") as directory:
        private = Path(directory)
        m.progress("preconditions_before")
        try:
            pre = preflight(private, args.upstream)
        except (OSError, ValueError, ImportError, KeyError) as error:
            pre = dict(
                passed=False,
                diagnostics=[
                    dict(
                        check="missing_file" if isinstance(error, OSError) else "rejected_producer",
                        upstream=m.up.TASK,
                        artifact_field="preconditions",
                        expected=True,
                        observed=str(error),
                        path=str(args.upstream),
                        sha256=None,
                    )
                ],
            )
        m.progress("preconditions_after")
        data = (
            m.cohort(pre["authentication_8803"]["producer"])
            if pre["passed"]
            else dict(inputs=[], labels=[])
        )
        for name in ["inputs", "labels"]:
            atomic_json(raw / (name + ".json"), dict(rows=data[name]))
        atomic_json(raw / "protocol.json", m.protocol())
        sealed = now()
        measurements = []
        if pre["passed"]:
            measurements.append(
                child(
                    "input_only_study",
                    [
                        sys.executable,
                        str(m.ROOT / m.CLI),
                        "--decide-input",
                        str(raw / "inputs.json"),
                        "--decisions",
                        str(raw / "decisions.json"),
                    ],
                    raw / "measurement",
                    deadline=600,
                )
            )
        if not (raw / "decisions.json").is_file():
            atomic_json(raw / "decisions.json", dict(rows=[]))
        measured = now()
        m.progress("validation_before")
        validation = (
            unqualified() if args.private_control or not pre["passed"] else validate(raw, private)
        )
        m.progress("validation_after")
        atomic_json(raw / "validation.json", validation)
        health = m.ROOT / "results/raw" / m.NAME / "startup/full_python_suite.receipt.json"
        work = dict(
            raw=str(raw),
            run_date=args.date,
            preconditions_checked=pre,
            duration_s=time.monotonic() - clock,
            phase_spans=[
                dict(phase=p, start_utc=a, end_utc=b)
                for p, a, b in [
                    ("preconditions_and_seal", start, sealed),
                    ("measurement", sealed, measured),
                    ("validation", measured, now()),
                ]
            ],
            measurement_receipts=measurements,
            source_artifact_hashes=pre["authentication_8803"]["references"]
            if pre["passed"]
            else [],
            code_config_hashes=[reference(m.ROOT / p) for p in m.OWNED + m.TESTS],
            raw_shard_hashes=[
                reference(raw / (n + ".json"))
                for n in ["inputs", "labels", "decisions", "protocol"]
            ],
            repository_health_reference=reference(health) if health.is_file() else None,
        )
        atomic_json(raw / "work.json", work)
        value = build(work, validation)
        seal(raw, value)
        m.progress("terminal_publication_before", len(value["rows"]), 0)
        receipt = publish_primary(output, value, lambda candidate: terminal(candidate, raw))
        atomic_json(raw / "publication_receipt.json", receipt)
        m.progress("published_" + value["verdict_class"], len(value["rows"]), 0)
    return 0
