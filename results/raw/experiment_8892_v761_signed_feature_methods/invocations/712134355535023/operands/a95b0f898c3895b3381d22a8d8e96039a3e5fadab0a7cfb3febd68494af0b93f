"""REQ-REPORT-8854: authenticate, qualify and publish frozen linear methods."""

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

from carnot.reporting import linear_ablation_methods_8854 as m
from carnot.reporting.current_work_receipt import (
    atomic_json,
    build_current_work_receipt,
    canonical_hash,
    sha256_file,
)
from carnot.reporting.primary_publication import publish_primary
from scripts.experiment_template import normalize_artifact_for_template_write

Json = dict[str, Any]
ROOT = m.prior.ROOT
NAME = "experiment_8854_v758_linear_ablation_methods"
CLI = "scripts/experiments/" + NAME + ".py"
OWNED = [
    "python/carnot/reporting/linear_ablation_methods_8854.py",
    "python/carnot/reporting/linear_ablation_runner_8854.py",
    CLI,
]
TESTS = ["tests/python/test_linear_ablation_methods_8854.py"]
AUDIT = ROOT / "results/experiment_8846_v757_future_benefit_audit.json"
AUDIT_PIN = "sha256:f0633d261a4cb99718f86e5cac07ef5abc36f6e737a0a67773ddef916214a8e0"
reference = m.prior.old.reference
bound = m.prior.old.bound


def preflight(raw: Path, source: Path) -> tuple[Json, Json, list[Json]]:
    """Check exact producer bytes and original methods before any owned training."""
    evidence, pre = m.prior.preflight(raw, source)
    audit = bound(dict(path=str(AUDIT), sha256=AUDIT_PIN))
    refs = [reference(source), reference(AUDIT)]
    refs += (
        audit["code_config_hashes"] + audit["raw_shard_hashes"] + audit["source_artifact_hashes"]
    )
    m.prior.upstream.bindings(refs, "original_authority")
    fresh = m.c.d.structure(m.payload(m.stream(0)["online"][0]))
    old = m.c.d.structure(m.prior.payload(m.prior.manifest()[0]["online"][0]))
    m.require(fresh != old and fresh not in pre["exposed_structure_hashes"], "graph_overlap")
    pre.update(
        configuration=m.protocol()["configuration"],
        fresh_query_structure=fresh,
        disjointness="seven query terms; exposed learning has at most six; unique anchored branch bits",
        authenticated_reference_count=len(refs),
        actual_invocation_date="20261105",
    )
    atomic_json(raw / "authority_snapshot.json", dict(preconditions=pre, references=refs))
    return evidence, pre, refs


def execute(raw: Path, evidence: Json, work: Json) -> None:
    """Refit original small heads and pilot two separate development streams only."""
    began = time.monotonic()
    fit = [r for r in evidence["development_rows"] if r["split"] == "fit"]
    valid = [r for r in evidence["development_rows"] if r["split"] == "validation"]
    heads = {}
    work.update(
        initialization_rows=[], pilot_refs=[], pilot_durations=[], numerical_training_calls=0
    )
    for seed in m.c.d.SEEDS:
        m.progress("head_fit_before", len(heads), 5 - len(heads))
        actual = m.c.d.fit(fit, valid, seed, "linear", 0.1)
        original = next(
            r
            for r in evidence["fits"]
            if r["seed"] == seed and r["objective"] == "linear" and r["lr"] == 0.1
        )
        m.require(actual == original, "original_fit_drift")
        heads[seed] = dict(objective="linear", lr=0.1, weights=actual["parameters"])
        work["initialization_rows"].append(
            dict(seed=seed, parameters=heads[seed], optimizer_steps=400)
        )
        work["numerical_training_calls"] += 1
        m.progress("head_fit_after", len(heads), 5 - len(heads))
    initialization_s = time.monotonic() - began
    m.progress("controls_before")
    work["causal_control_rows"] = m.controls(raw / "controls")
    work["numerical_training_calls"] += sum(
        r.get("numerical_training_calls", 0) for r in work["causal_control_rows"]
    )
    m.progress("controls_after", len(work["causal_control_rows"]))
    for plan in work["protocol"]["pilot_manifest"]:
        m.progress("pilot_before", len(work["pilot_refs"]), 2 - len(work["pilot_refs"]))
        began = time.monotonic()
        result = m.run_stream(raw / "pilots" / plan["stream_id"], plan, heads)
        checkpoint = raw / "pilots" / (plan["stream_id"] + ".json")
        atomic_json(checkpoint, result)
        work["pilot_refs"].append(reference(checkpoint))
        work["pilot_durations"].append(time.monotonic() - began)
        work["numerical_training_calls"] += len(result["gradient_rows"])
        m.progress("pilot_after", len(work["pilot_refs"]), 2 - len(work["pilot_refs"]))
    work["timing_projection"] = m.projection(work["pilot_durations"], initialization_s)
    m.require(work["timing_projection"]["passed"], "execution_budget")


def validate(raw: Path) -> Json:
    """Reuse actual unit/child coverage and keep repository health separate."""
    with (
        patch.dict(os.environ, TMPDIR=tempfile.gettempdir()),
        patch.multiple(m.prior.validation, OWNED=OWNED, TESTS=TESTS, CLI=CLI),
    ):
        return m.prior.validation.validate(raw)


def terminal(candidate: Path, raw: Path) -> Json:
    """Keep the unchanged scanners and a genuine outside-checkout replay."""
    with patch.object(m.prior.validation, "CLI", CLI):
        return m.prior.validation.terminal(candidate, raw)


def replay(path: Path) -> Json:
    """Rebuild protocol, row census and pilot arithmetic, rejecting rehashed edits."""
    value = json.loads(path.read_bytes())
    for ref in (
        value["source_artifact_hashes"] + value["code_config_hashes"] + value["raw_shard_hashes"]
    ):
        m.prior.upstream.bindings([ref], "replay_operand")
    work = bound(value["raw_shard_hashes"][0])
    frozen = m.protocol()
    m.require(
        work["protocol"] == frozen
        and value["configuration_hashes"] == dict(protocol=canonical_hash(frozen)),
        "protocol_drift",
    )
    expected = m.census(frozen, work["row_reason"])
    pilots = [bound(ref) for ref in work.get("pilot_refs", [])]
    for key, actual in dict(
        rows=expected,
        intended_count=27648,
        completed_count=0,
        censored_count=27648,
        failed_count=0,
        excluded_count=0,
        independent_count=0,
        arm_definitions=m.ARMS,
        cohort_manifest=frozen["cohort_manifest"],
        expected_row_counts=frozen["expected_row_counts"],
        acceptance_gates=m.GATES,
        gradient_rows=[row for pilot in pilots for row in pilot["gradient_rows"]],
        paired_delay_schedules=[
            dict(cluster_id=s["stream_id"], delays=s["delays"]) for s in frozen["cohort_manifest"]
        ],
    ).items():
        m.require(value[key] == actual, "reduction_drift:" + key)
    for key in ("causal_control_rows", "timing_projection", "numerical_training_calls"):
        m.require(
            value[key]
            == work.get(
                key, [] if key == "causal_control_rows" else {} if key == "timing_projection" else 0
            ),
            "work_drift:" + key,
        )
    with tempfile.TemporaryDirectory(prefix="carnot8854-replay-") as private:
        if pilots:
            evidence, _, _ = preflight(Path(private) / "authority", m.prior.AUTHORITY)
            heads = {
                r["seed"]: dict(objective="linear", lr=0.1, weights=r["parameters"])
                for r in evidence["fits"]
                if r["objective"] == "linear" and r["lr"] == 0.1
            }
            for i, pilot in enumerate(pilots):
                m.require(
                    pilot["stream"] == frozen["pilot_manifest"][i]
                    and {int(k): v for k, v in pilot["heads"].items()} == heads,
                    "pilot_authority",
                )
                m.replay_pilot(pilot, Path(private) / str(i))
            m.require(
                work["causal_control_rows"] == m.controls(Path(private) / "controls"),
                "control_drift",
            )
    ready = int(value["verdict_class"] == "circular_positive")
    m.require(
        value["linear_methods_ready_score"] == ready
        and value["required_checks_passed"] == bool(ready),
        "readiness_drift",
    )
    if ready:
        coverage = value["owned_statement_coverage"]
        measured = coverage["combined_reference"]
        m.prior.upstream.bindings([measured], "coverage_operand")
        m.require(
            coverage == m.prior.validation.coverage_summary(Path(measured["path"]), OWNED)
            and {r["path"] for r in coverage["files"]} == set(OWNED),
            "coverage_drift",
        )
        m.require(
            len(pilots) == 2
            and work["timing_projection"]["passed"]
            and coverage["complete"]
            and coverage["percent"] == 100
            and all(not r["excluded"] and not r["missing"] for r in coverage["files"])
            and all(r["passed"] for r in value["validation_receipts"] if r["scope"] == "owned"),
            "qualification_drift",
        )
    for receipt in value["validation_receipts"]:
        m.require(
            sha256_file(ROOT / receipt["log_path"]) == receipt["log_sha256"], "validation_log"
        )
    return dict(
        replay_passed=True,
        pilot_count=len(work.get("pilot_refs", [])),
        protocol_hash=canonical_hash(frozen),
    )


def main(argv: list[str] | None = None) -> int:
    """Keep private custody options separate from frozen scientific inputs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default="20261105")
    parser.add_argument(
        "--output", "--private-output", type=Path, default=ROOT / "results" / (NAME + ".json")
    )
    parser.add_argument("--raw-dir", type=Path)
    parser.add_argument("--source", type=Path, default=m.prior.AUTHORITY)
    parser.add_argument("--replay", type=Path)
    args = parser.parse_args(argv)
    if args.replay:
        try:
            print(json.dumps(replay(args.replay)), flush=True)
            return 0
        except (OSError, ValueError, KeyError, RuntimeError) as error:
            print(f"replay_rejected:{error}", flush=True)
            return 1
    if args.date != "20261105":
        print("date_rejected: expected 20261105", flush=True)
        return 1
    started = time.monotonic_ns()
    output = args.output.absolute()
    raw = (
        args.raw_dir or output.parent / "raw" / NAME / "invocations" / str(time.time_ns())
    ).absolute()
    raw.mkdir(parents=True, exist_ok=True)
    raw.chmod(0o700)
    work: Json = dict(protocol=m.protocol(), row_reason="registered_for_later_execution")
    pre: Json = {}
    refs: list[Json] = []
    failures = []
    verdict = "circular_positive"
    admitted = False
    m.progress("preconditions_before")
    try:
        evidence, pre, refs = preflight(raw, args.source)
        admitted = True
        m.progress("preconditions_after", 1)
        execute(raw, evidence, work)
    except (OSError, KeyError, ValueError, RuntimeError, StopIteration) as error:
        external = not admitted and str(error) not in {"internal_resources", "graph_overlap"}
        verdict = "blocked" if external else "disqualified"
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
        work["row_reason"] = reason
        failures.append(
            dict(
                check="original_authority" if external else "owned_controls",
                upstream="exp8841-memory-execution",
                artifact_field="memory_execution_ready_score",
                expected=1,
                observed=None if reason != "observed_zero" else 0,
                path=str(args.source),
                hash=sha256_file(args.source) if args.source.is_file() else None,
                status=reason,
                reason=str(error),
            )
        )
    atomic_json(raw / "evidence.json", work)
    atomic_json(raw / "protocol.json", work["protocol"])
    sources = raw / "sources"
    sources.mkdir()
    for name in OWNED + TESTS:
        shutil.copyfile(ROOT / name, sources / Path(name).name)
    m.progress("validation_before", 0, 1)
    checked = validate(raw)
    m.progress("validation_after", 1)
    if not checked["passed"]:
        verdict = "disqualified"
    ended = time.monotonic_ns()
    phases = [
        dict(
            phase="qualification_and_validation",
            started_monotonic_ns=started,
            ended_monotonic_ns=ended,
            duration_s=(ended - started) / 1e9,
        )
    ]
    substrate = "deterministic_verifier" if admitted else "blocked_no_run"
    receipt = build_current_work_receipt(
        run_id=raw.name,
        owner_pid=os.getpid(),
        events=[],
        inference_substrate=substrate,
        inference_substrate_class="no_model_load",
        execution_venue="host",
        inference_substrate_details=dict(backend="JAX CPU", generator_weights_updated=False),
        started_monotonic_ns=started,
        ended_monotonic_ns=ended,
        phase_spans=phases,
        small_ebm_training=dict(performed=bool(work.get("numerical_training_calls"))),
    )
    frozen = work["protocol"]
    value = dict(
        experiment_id=8854,
        task_id="exp8854-linear-ablation-methods",
        milestone="2026.11.758",
        run_date=args.date,
        honest_verdict="complete_" + verdict + "_linear_methods",
        verdict_class=verdict,
        inference_substrate=substrate,
        inference_substrate_class="no_model_load",
        execution_venue="host",
        MODEL_SPECS=[],
        model_invocation_counts=receipt["invocation_counts"],
        numerical_training_calls=work.get("numerical_training_calls", 0),
        verifier_is_oracle=True,
        exposure_scope="constructed development pilots only; exact symbolic labels; future and retention labels do not train or select methods",
        independent_generalization_score=0,
        duration_s=(ended - started) / 1e9,
        phase_spans=phases,
        preconditions_checked=pre,
        random_seed=8854000,
        reproducibility_checksum=canonical_hash(frozen),
        source_artifact_hashes=refs,
        code_config_hashes=[reference(ROOT / p) for p in OWNED + TESTS],
        raw_shard_hashes=[reference(raw / "evidence.json"), reference(raw / "protocol.json")]
        + [reference(p) for p in sorted(sources.iterdir())],
        rows=m.census(frozen, work["row_reason"]),
        intended_count=27648,
        completed_count=0,
        failed_count=0,
        censored_count=27648,
        excluded_count=0,
        independent_count=0,
        gate_check_summary=failures,
        acceptance_gates=m.GATES,
        required_checks_passed=verdict == "circular_positive",
        validation_receipts=checked["receipts"],
        owned_statement_coverage=checked["coverage"],
        linear_methods_ready_score=int(verdict == "circular_positive"),
        arm_definitions=m.ARMS,
        configuration_hashes=dict(protocol=canonical_hash(frozen)),
        cohort_manifest=frozen["cohort_manifest"],
        causal_control_rows=work.get("causal_control_rows", []),
        gradient_rows=[
            row for ref in work.get("pilot_refs", []) for row in bound(ref)["gradient_rows"]
        ],
        paired_delay_schedules=[
            dict(cluster_id=s["stream_id"], delays=s["delays"]) for s in frozen["cohort_manifest"]
        ],
        expected_row_counts=frozen["expected_row_counts"],
        timing_projection=work.get("timing_projection", {}),
        current_work_receipt=receipt,
        repository_health_receipts=[
            r for r in checked["receipts"] if r["scope"] == "repository_health"
        ],
        methodology_note="Two development pilots qualify methods only. All27648 later research slots remain censored. No pilot benefit gate or milestone activation. Same released counts, linear loss and plain SGD as original control.",
    )
    value["field_principles"] = {
        k: "Bind measured method qualification to exact operands; completion does not imply benefit."
        for k in value
    }
    value["field_principles"].update(
        rows="All27648 preregistered future slots retain identity and denominator; development pilots are separate immutable operands.",
        timing_projection="Measured complete pilot cost projects24 streams plus cold replay with safety factor1.5 and600 seconds validation reserve.",
        linear_methods_ready_score="Owned controls, causal integrity, feasible timing and actual100% owned statement coverage release execution; pilot benefit is irrelevant.",
    )
    candidate = raw / "candidate.json"
    atomic_json(candidate, value)
    m.progress("terminal_before", 0, 1)
    final = terminal(candidate, raw)
    m.progress("terminal_after", 1)
    value["validation_receipts"].extend(final["receipts"])
    finished = time.monotonic_ns()
    phases.append(
        dict(
            phase="terminal_scanners_and_replay",
            started_monotonic_ns=ended,
            ended_monotonic_ns=finished,
            duration_s=(finished - ended) / 1e9,
        )
    )
    value["duration_s"] = (finished - started) / 1e9
    receipt.update(ended_monotonic_ns=finished, duration_s=value["duration_s"], phase_spans=phases)
    if not final["passed"] or value["duration_s"] > 4200:
        value.update(
            honest_verdict="complete_disqualified_terminal_validation",
            verdict_class="disqualified",
            required_checks_passed=False,
            linear_methods_ready_score=0,
        )
    value = normalize_artifact_for_template_write(value)
    publish_primary(output, value, lambda path: dict(passed=json.loads(path.read_bytes()) == value))
    m.progress("published", len(work.get("pilot_refs", [])), value["censored_count"])
    return 0
