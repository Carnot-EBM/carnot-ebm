"""REQ-REPORT-8856: authenticate and publish the frozen linear stream experiment."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import tempfile
import time
from typing import Any
from unittest.mock import patch

from carnot.reporting import linear_ablation_runner_8854 as authority
from carnot.reporting import linear_causal_streams_8856 as m
from carnot.reporting.current_work_receipt import (
    atomic_json,
    build_current_work_receipt,
    canonical_hash,
    sha256_file,
)
from carnot.reporting.primary_publication import publish_primary
from scripts.experiment_template import normalize_artifact_for_template_write

Json = dict[str, Any]
ROOT = authority.ROOT
NAME = "experiment_8856_v758_linear_causal_streams"
CLI = "scripts/experiments/" + NAME + ".py"
OWNED = [
    "python/carnot/reporting/linear_causal_streams_8856.py",
    "python/carnot/reporting/linear_causal_runner_8856.py",
    CLI,
]
TESTS = ["tests/python/test_linear_causal_streams_8856.py"]
SOURCE = ROOT / "results/experiment_8854_v758_linear_ablation_methods.json"
SOURCE_PIN = "sha256:830795e2e0c73a0c819f0bff8ac9685a2d8dd142664d149d6e043fb7a295afe0"
reference = authority.reference
bound = authority.bound


def preflight(raw: Path, source: Path) -> tuple[Json, Json, list[Json]]:
    """Reject changed authorities before any fitting or future label is opened."""
    producer = bound(dict(path=str(source), sha256=SOURCE_PIN))
    if (
        producer["linear_methods_ready_score"] != 1
        or producer["required_checks_passed"] is not True
    ):
        raise PermissionError("linear_methods_ready_score:observed_zero")
    refs = (
        [reference(source)]
        + producer["source_artifact_hashes"]
        + producer["code_config_hashes"]
        + producer["raw_shard_hashes"]
    )
    m.a.prior.upstream.bindings(refs, "linear_method_authority")
    work = bound(producer["raw_shard_hashes"][0])
    m.a.require(work["protocol"] == m.a.protocol(), "protocol_drift")
    evidence, pre, original = authority.preflight(raw, m.a.prior.AUTHORITY)
    pre.update(
        actual_invocation_date="20261106",
        frozen_protocol_hash=canonical_hash(work["protocol"]),
        linear_methods_ready_score=1,
        deadline_s=4200,
        cpu_only=True,
    )
    return evidence, pre, [*refs, *original]


def initialize(evidence: Json) -> tuple[dict[int, Json], list[Json]]:
    """Reproduce the original five fits rather than importing trained coefficients."""
    fit = [r for r in evidence["development_rows"] if r["split"] == "fit"]
    valid = [r for r in evidence["development_rows"] if r["split"] == "validation"]
    heads, rows = {}, []
    for seed in m.a.c.d.SEEDS:
        m.progress("head_fit_before", len(heads), 5 - len(heads))
        actual = m.a.c.d.fit(fit, valid, seed, "linear", 0.1)
        original = next(
            r
            for r in evidence["fits"]
            if r["seed"] == seed and r["objective"] == "linear" and r["lr"] == 0.1
        )
        m.a.require(actual == original, "original_fit_drift")
        heads[seed] = dict(objective="linear", lr=0.1, weights=actual["parameters"])
        rows.append(
            dict(
                seed=seed,
                parameters=heads[seed],
                optimizer_steps=400,
                fitting_input_hash=canonical_hash(fit),
            )
        )
        m.progress("head_fit_after", len(heads), 5 - len(heads))
    return heads, rows


def execute(raw: Path, evidence: Json, work: Json, started: float) -> None:
    """Checkpoint whole streams and preserve the full roster if the deadline arrives."""
    began = time.monotonic()
    heads, work["initialization_rows"] = initialize(evidence)
    initialization_s = time.monotonic() - began
    m.progress("controls_before")
    work["controls"] = m.a.controls(raw / "controls")
    m.progress("controls_after", len(work["controls"]))
    pilots = []
    for plan in work["protocol"]["pilot_manifest"]:
        m.progress("pilot_before", len(pilots), 2 - len(pilots))
        began = time.monotonic()
        result = m.run_stream(raw / "pilots" / plan["stream_id"], plan, heads)
        path = raw / "pilots" / (plan["stream_id"] + ".json")
        atomic_json(path, result)
        work["pilot_refs"].append(reference(path))
        pilots.append(time.monotonic() - began)
        m.progress("pilot_after", len(pilots), 2 - len(pilots))
    work["timing_projection"] = m.a.projection(pilots, initialization_s)
    if not work["timing_projection"]["passed"]:
        raise TimeoutError("projected_execution_budget")
    for plan in work["protocol"]["cohort_manifest"]:
        if time.monotonic() - started > 2500:
            raise TimeoutError("registered_execution_deadline")
        m.progress("stream_before", len(work["stream_refs"]), 24 - len(work["stream_refs"]))
        result = m.run_stream(raw / "stores" / plan["stream_id"], plan, heads)
        path = raw / "streams" / (plan["stream_id"] + ".json")
        atomic_json(path, result)
        work["stream_refs"].append(reference(path))
        work["stream_results"].append(result)
        atomic_json(
            raw / "checkpoint.json",
            dict(
                stream_refs=work["stream_refs"],
                completed=len(work["stream_refs"]),
                pending=24 - len(work["stream_refs"]),
            ),
        )
        m.progress("stream_after", len(work["stream_refs"]), 24 - len(work["stream_refs"]))


def validate(raw: Path) -> Json:
    """Use existing bounded checks with actual owned unit and child coverage."""
    with patch.multiple(m.a.prior.validation, OWNED=OWNED, TESTS=TESTS, CLI=CLI):
        return m.a.prior.validation.validate(raw)


def terminal(candidate: Path, raw: Path) -> Json:
    """Allow the complete cold replay within the task deadline, keeping scanners unchanged."""
    py = str(ROOT / ".venv/bin/python")
    receipts = m.a.prior.validation.commands(
        [
            (
                "adversarial",
                [py, "scripts/adversarial_verify.py", str(candidate), "--json"],
                "owned",
                60,
            ),
            (
                "row_consistency",
                [py, "scripts/verdict_row_consistency_lint.py", "--strict", str(candidate)],
                "owned",
                60,
            ),
            (
                "cold_replay",
                [
                    "env",
                    "-u",
                    "PYTHONPATH",
                    "-C",
                    "/tmp",
                    py,
                    "-u",
                    str(ROOT / CLI),
                    "--replay",
                    str(candidate),
                ],
                "owned",
                1800,
            ),
        ],
        raw / "terminal",
    )
    return dict(passed=all(r["passed"] for r in receipts), receipts=receipts)


def reductions(work: Json, reason: str) -> Json:
    """Every primary headline reduces from retained rows or authenticated stream operands."""
    rows = m.result_rows(work, reason)
    streams = work["stream_results"]
    complete = sum(r["status"] == "complete" for r in rows)
    pilots = [bound(ref) for ref in work["pilot_refs"]]
    return dict(
        rows=rows,
        intended_count=27648,
        completed_count=complete,
        failed_count=0,
        censored_count=27648 - complete,
        excluded_count=0,
        independent_count=len(streams),
        per_stream_results=[
            dict(
                cluster_id=s["stream"]["stream_id"],
                integrity=m.integrity(s),
                descriptives=m.descriptives(s["rows"]),
            )
            for s in streams
        ],
        prediction_seals=[
            dict(
                unit_id=r["unit_id"],
                cluster_id=r["cluster_id"],
                arm=r["arm"],
                prediction_seal=r["prediction_seal"],
                wall_clock_ns=r["wall_clock_ns"],
            )
            for r in rows
            if r["split"] == "online" and r["status"] == "complete"
        ],
        **{
            k: [r for s in streams for r in s[k]]
            for k in (
                "feedback_rows",
                "rejected_feedback_rows",
                "template_reuse_rows",
                "movement_rows",
                "retention_rows",
                "restart_rows",
            )
        },
        delay_rows=m.descriptives(rows),
        numerical_training_calls=sum(r["optimizer_steps"] for r in work["initialization_rows"])
        + sum(r.get("numerical_training_calls", 0) for r in work["controls"])
        + sum(r["optimizer_steps"] for s in streams + pilots for r in s["feedback_rows"]),
    )


def replay(path: Path) -> Json:
    """Rebuild fits, sealed predictions and durable transitions in new private stores."""
    value = json.loads(path.read_bytes())
    for refs in (
        value["source_artifact_hashes"],
        value["code_config_hashes"],
        value["raw_shard_hashes"],
    ):
        m.a.prior.upstream.bindings(refs, "replay_operand")
    work = bound(value["raw_shard_hashes"][0])
    m.a.require(work["protocol"] == m.a.protocol(), "protocol_drift")
    m.a.require(value["acceptance_gates"] == m.a.GATES, "acceptance_drift")
    m.a.require(len(work["stream_refs"]) == len(work["stream_results"]), "stream_census")
    for key, expected in reductions(work, value["row_reason"]).items():
        m.a.require(value[key] == expected, "reduction_drift:" + key)
    for receipt in value["validation_receipts"]:
        m.a.require(
            sha256_file(ROOT / receipt["log_path"]) == receipt["log_sha256"], "validation_log"
        )
    if work["initialization_rows"]:
        with tempfile.TemporaryDirectory(prefix="carnot8856-replay-") as private:
            evidence, _, _ = preflight(Path(private) / "authority", SOURCE)
            heads, initial = initialize(evidence)
            m.a.require(initial == work["initialization_rows"], "initialization_drift")
            m.a.require(
                m.a.controls(Path(private) / "controls") == work["controls"], "control_drift"
            )
            for i, (ref, expected) in enumerate(zip(work["stream_refs"], work["stream_results"])):
                m.a.require(
                    bound(ref) == expected
                    and expected["stream"] == work["protocol"]["cohort_manifest"][i],
                    "stream_authority",
                )
                m.a.prior.upstream.bindings(expected["final_state_refs"], "durable_state")
                m.progress("replay_stream_before", i, len(work["stream_refs"]) - i)
                actual = m.run_stream(Path(private) / str(i), expected["stream"], heads)
                m.a.require(
                    [r["sha256"] for r in actual.pop("final_state_refs")]
                    == [r["sha256"] for r in expected["final_state_refs"]],
                    "durable_drift",
                )
                m.a.require(
                    m.semantic(actual)
                    == m.semantic({k: v for k, v in expected.items() if k != "final_state_refs"}),
                    "stream_drift",
                )
                m.progress("replay_stream_after", i + 1, len(work["stream_refs"]) - i - 1)
    ready = value["verdict_class"] == "circular_positive" and value["completed_count"] == 27648
    m.a.require(
        value["stream_rows_ready_score"] == int(ready) and value["required_checks_passed"] == ready,
        "readiness_drift",
    )
    if ready:
        coverage = value["owned_statement_coverage"]
        ref = coverage["combined_reference"]
        m.a.prior.upstream.bindings([ref], "coverage_operand")
        m.a.require(
            coverage == m.a.prior.validation.coverage_summary(Path(ref["path"]), OWNED)
            and coverage["complete"]
            and coverage["percent"] == 100
            and all(r["passed"] for r in value["validation_receipts"] if r["scope"] == "owned"),
            "qualification_drift",
        )
    return dict(
        replay_passed=True,
        completed_count=value["completed_count"],
        protocol_hash=canonical_hash(work["protocol"]),
    )


def main(argv: list[str] | None = None) -> int:
    """Separate private custody and replay choices from the fixed research protocol."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default="20261106")
    parser.add_argument(
        "--output", "--private-output", type=Path, default=ROOT / "results" / (NAME + ".json")
    )
    parser.add_argument("--raw-dir", type=Path)
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--replay", type=Path)
    args = parser.parse_args(argv)
    if args.replay:
        try:
            print(json.dumps(replay(args.replay)), flush=True)
            return 0
        except (OSError, ValueError, KeyError, RuntimeError) as error:
            print(f"replay_rejected:{error}", flush=True)
            return 1
    if args.date != "20261106":
        print("date_rejected: expected 20261106", flush=True)
        return 1
    started, started_ns = time.monotonic(), time.monotonic_ns()
    output = args.output.absolute()
    raw = (
        args.raw_dir or output.parent / "raw" / NAME / "invocations" / str(time.time_ns())
    ).absolute()
    raw.mkdir(parents=True, exist_ok=False)
    raw.chmod(0o700)
    work: Json = dict(
        protocol=m.a.protocol(),
        stream_results=[],
        stream_refs=[],
        pilot_refs=[],
        initialization_rows=[],
        controls=[],
        timing_projection={},
    )
    pre: Json = {}
    refs: list[Json] = []
    gates = []
    admitted = False
    verdict, reason = "circular_positive", "unfinished_execution"
    m.progress("preconditions_before")
    try:
        evidence, pre, refs = preflight(raw, args.source)
        admitted = True
        m.progress("preconditions_after", 1)
        execute(raw, evidence, work, started)
        if len(work["stream_results"]) != 24:
            verdict = "partial"
    except (OSError, ValueError, KeyError, RuntimeError, StopIteration) as error:
        external = not admitted and str(error) != "internal_resources"
        verdict = (
            "partial"
            if isinstance(error, TimeoutError)
            else "blocked"
            if external
            else "disqualified"
        )
        reason = (
            "missing_file"
            if isinstance(error, FileNotFoundError)
            else "missing_field"
            if isinstance(error, KeyError)
            else "observed_zero"
            if "observed_zero" in str(error)
            else "rejected_producer"
            if external
            else "owned_failure"
        )
        operand = Path(getattr(error, "filename", None) or args.source)
        gates.append(
            dict(
                check="linear_method_authority" if external else "owned_execution",
                upstream="exp8854-linear-ablation-methods",
                artifact_field="linear_methods_ready_score",
                expected=1,
                observed=0 if reason == "observed_zero" else str(error),
                path=str(operand),
                hash=sha256_file(operand) if operand.is_file() else None,
                status=reason,
            )
        )
    atomic_json(raw / "evidence.json", work)
    sources = raw / "sources"
    sources.mkdir()
    for name in OWNED + TESTS:
        shutil.copyfile(ROOT / name, sources / Path(name).name)
    m.progress("validation_before")
    checked = validate(raw)
    m.progress("validation_after", 1)
    if not checked["passed"]:
        verdict = "disqualified"
    ended_ns = time.monotonic_ns()
    spans = [
        dict(
            phase="preconditions_execution_validation",
            started_monotonic_ns=started_ns,
            ended_monotonic_ns=ended_ns,
            duration_s=(ended_ns - started_ns) / 1e9,
        )
    ]
    performed = bool(work["initialization_rows"])
    substrate = "deterministic_verifier" if performed else "blocked_no_run"
    receipt = build_current_work_receipt(
        run_id=raw.name,
        owner_pid=os.getpid(),
        events=[],
        inference_substrate=substrate,
        inference_substrate_class="no_model_load",
        execution_venue="host",
        inference_substrate_details=dict(backend="jax_cpu", generator_weights_updated=False),
        started_monotonic_ns=started_ns,
        ended_monotonic_ns=ended_ns,
        phase_spans=spans,
        small_ebm_training=dict(performed=performed),
    )
    value = dict(
        experiment_id=8856,
        task_id="exp8856-linear-causal-streams",
        milestone="2026.11.758",
        run_date=args.date,
        honest_verdict="complete_" + verdict + "_linear_causal_streams",
        verdict_class=verdict,
        row_reason=reason,
        inference_substrate=substrate,
        inference_substrate_class="no_model_load",
        execution_venue="host",
        MODEL_SPECS=[],
        model_invocation_counts=receipt["invocation_counts"],
        verifier_is_oracle=True,
        exposure_scope="constructed signed records; released exact labels only; retention labels never train",
        independent_generalization_score=0,
        duration_s=(ended_ns - started_ns) / 1e9,
        phase_spans=spans,
        preconditions_checked=pre,
        random_seed=8854000,
        reproducibility_checksum=canonical_hash(work["protocol"]),
        source_artifact_hashes=refs,
        code_config_hashes=[
            reference(ROOT / p)
            for p in dict.fromkeys(
                OWNED
                + TESTS
                + authority.OWNED
                + m.a.prior.validation.CODE
                + ["scripts/experiment_template.py"]
            )
        ],
        raw_shard_hashes=[reference(raw / "evidence.json")]
        + work["stream_refs"]
        + work["pilot_refs"]
        + [reference(p) for p in sorted(sources.iterdir())],
        gate_check_summary=gates,
        acceptance_gates=m.a.GATES,
        required_checks_passed=verdict == "circular_positive",
        validation_receipts=checked["receipts"],
        owned_statement_coverage=checked["coverage"],
        stream_rows_ready_score=int(verdict == "circular_positive"),
        current_work_receipt=receipt,
        repository_health_receipts=[
            r for r in checked["receipts"] if r["scope"] == "repository_health"
        ],
        arm_definitions=m.a.ARMS,
        protocol=work["protocol"],
        timing_projection=work["timing_projection"],
        methodology_note="Six paired linear arms; released verified labels; restart24; read-only retention0/24/48; future25-48. Immediate feedback is diagnostic. Exp8860 owns benefit acceptance. No milestone activation.",
        **reductions(work, reason),
    )
    value["field_principles"] = {
        k: "Bind actual measured work to immutable operands; completion and update counts do not establish future benefit."
        for k in value
    }
    value["field_principles"].update(
        stream_rows_ready_score="Complete integrity and owned validation release an independent audit; gain is not a gate.",
        delay_rows="Descriptives reduce from complete query rows by paired delay regime and time interval.",
        numerical_training_calls="Count individual SGD steps in initial fits, controls, pilots and registered streams; validation and cold replay are separate.",
    )
    candidate = raw / "candidate.json"
    atomic_json(candidate, value)
    m.progress("terminal_before")
    final = terminal(candidate, raw)
    m.progress("terminal_after", 1)
    value["validation_receipts"].extend(final["receipts"])
    finished_ns = time.monotonic_ns()
    spans.append(
        dict(
            phase="terminal_scanners_cold_replay",
            started_monotonic_ns=ended_ns,
            ended_monotonic_ns=finished_ns,
            duration_s=(finished_ns - ended_ns) / 1e9,
        )
    )
    value["duration_s"] = (finished_ns - started_ns) / 1e9
    receipt.update(
        ended_monotonic_ns=finished_ns, duration_s=value["duration_s"], phase_spans=spans
    )
    if not final["passed"] or value["duration_s"] > 4200:
        value.update(
            honest_verdict="complete_disqualified_terminal_validation",
            verdict_class="disqualified",
            required_checks_passed=False,
            stream_rows_ready_score=0,
        )
    value = normalize_artifact_for_template_write(value)
    if value["verdict_class"] == "partial":
        atomic_json(output, value)
    else:
        publish_primary(
            output, value, lambda path: dict(passed=json.loads(path.read_bytes()) == value)
        )
    m.progress("published", value["completed_count"], value["censored_count"])
    return 0
