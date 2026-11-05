"""REQ-REPORT-8844: authenticated execution and cold replay with private evidence."""

import argparse
import json
import os
from pathlib import Path
import shutil
import sys
from tempfile import TemporaryDirectory
import time
from typing import Any

import jax
import numpy as np
import scipy

from carnot.reporting import selective_availability_8844 as m
from carnot.reporting import selective_methods_runner_8829 as inherited
from carnot.reporting.current_work_receipt import (
    atomic_json,
    canonical_hash,
    sha256_file,
    build_current_work_receipt,
    validate_current_work_receipt,
)
from carnot.reporting.primary_publication import publish_primary
from carnot.reporting.selective_runner_8816 import stream, unqualified
from carnot.reporting.v746_retained_evidence import reference, load
from carnot.reporting.v748_collection_contract import namespace
from scripts.experiment_template import normalize_artifact_for_template_write

Json = dict[str, Any]
UPSTREAM = m.ROOT / "results/experiment_8842_v757_selective_population.json"
PIN = "sha256:c6ba96b888d6fc66853ea846f7be857c8238c396cc6aa5f9fa62dc78fad765c9"
PROTOCOL_PIN = "sha256:a9d1ea70774fa2417058ed9e66de2c76f0e8de383ed73b10415eac926797e871"


def preconditions(raw: Path, upstream: Path) -> Json:
    """Missing external authority blocks; local resource or owned validation failure disqualifies."""
    raw.mkdir(parents=True, exist_ok=True, mode=0o700)
    raw.chmod(0o700)
    atomic_json(raw / "writable.json", dict(probe=True))
    diagnostic: Json = dict(
        upstream="exp8842-selective-population",
        artifact_field="selective_protocol_ready_score",
        expected=1,
        observed=None,
        path=str(upstream),
        sha256=None,
    )
    refs, errors, protocol = [], [], None
    try:
        value = json.loads(upstream.read_bytes())
        diagnostic["sha256"] = sha256_file(upstream)
        diagnostic["observed"] = value["selective_protocol_ready_score"]
        if value["selective_protocol_ready_score"] != 1:
            raise ValueError("observed_zero")
        if value["task_id"] != "exp8842-selective-population" or diagnostic["sha256"] != PIN:
            raise ValueError("rejected_producer")
        refs = (
            [reference(upstream)]
            + value["source_artifact_hashes"]
            + value["code_config_hashes"]
            + value["raw_shard_hashes"]
        )
        for ref in refs:
            diagnostic.update(
                path=ref["path"], artifact_field="sha256", expected=ref["sha256"], sha256=None
            )
            diagnostic["observed"] = sha256_file(Path(ref["path"]))
            diagnostic["sha256"] = diagnostic["observed"]
            if diagnostic["observed"] != ref["sha256"]:
                raise ValueError("rejected_operand")
        protocol = next(
            ref for ref in value["raw_shard_hashes"] if Path(ref["path"]).name == "protocol.json"
        )
        if protocol["sha256"] != PROTOCOL_PIN:
            raise ValueError("rejected_protocol")
        atomic_json(raw / "upstream.json", value)
    except (OSError, ValueError, KeyError, TypeError, StopIteration) as error:
        errors.append(
            dict(
                diagnostic,
                check="missing_file"
                if isinstance(error, OSError)
                else "missing_field"
                if isinstance(error, (KeyError, StopIteration))
                else "observed_zero"
                if str(error) == "observed_zero"
                else "rejected_producer",
                error=str(error),
            )
        )
    resources: Json = dict(
        repository_virtualenv=(m.ROOT / ".venv/bin/python").is_file(),
        dependencies=dict(jax=jax.__version__, numpy=np.__version__, scipy=scipy.__version__),
        backend=jax.default_backend(),
        private_output_writable=True,
        private_output_mode=oct(raw.stat().st_mode & 0o777),
        free_disk_bytes=shutil.disk_usage(raw).free,
        measurement_deadline_s=900,
        validation_deadline_per_child_s=600,
    )
    resources["passed"] = (
        resources["repository_virtualenv"]
        and resources["backend"] == "cpu"
        and resources["free_disk_bytes"] >= 512 * 1024 * 1024
    )
    return dict(
        resources=resources,
        references=refs,
        diagnostics=errors,
        external_ready=not errors,
        frozen_protocol=protocol,
    )


def measure(raw: Path, pre: Json) -> Json:
    """Seal score and cutoff bytes before releasing independent evaluation and census labels."""
    started = time.monotonic()
    upstream = load(reference(raw / "upstream.json"))
    cohort, trace = load(upstream["population_manifest"]), load(upstream["draw_manifest"])
    atomic_json(raw / "population.json", cohort)
    atomic_json(raw / "draws.json", trace)
    m.p.audit_draws(trace, 4096)
    fitted = m.fit(cohort)
    m.audit_fit(cohort, fitted)
    atomic_json(raw / "fit.json", fitted)
    predictions = m.predict(cohort, fitted)
    atomic_json(raw / "scores.json", dict(rows=predictions))
    m.progress("all_seven_scores_sealed_before_calibration", len(predictions), 0)
    units = sorted(
        (u for u in cohort["inputs"] if u["split"] == "population"), key=lambda u: u["unit_id"]
    )
    indices = sorted({i for rep in trace["replicates"] for i in rep["calibration"]["indices"]})
    calibration_truth = m.p.labels([units[i] for i in indices])
    selection = m.select(predictions, calibration_truth, trace)
    atomic_json(raw / "selection.json", selection)
    m.progress("all_cutoffs_sealed_before_census", len(selection["selected_cutoff_rows"]), 0)
    truth = m.p.labels(units)
    atomic_json(raw / "labels.json", dict(labels=truth))
    result = m.evaluate(predictions, truth, trace, selection)
    atomic_json(raw / "reduction.json", result)
    if time.monotonic() - started > pre["resources"]["measurement_deadline_s"]:
        raise ValueError("measurement_deadline_exceeded")
    return result


def validate(raw: Path, private: Path) -> Json:
    """Reuse unchanged consumer tests and actual unit/child statement accounting."""
    checker = namespace(inherited, "selective8844_validation")
    checker.m = m
    return dict(checker.validate(raw, private))


def replay(path: Path) -> Json:
    """Reproduce reductions and predictions in a fresh process rather than trusting self-hashes."""
    value = json.loads(path.read_bytes())
    if value["experiment_id"] != 8844 or value["task_id"] != m.TASK:
        raise ValueError("producer_identity")
    checksum = value.pop("reproducibility_checksum")
    if canonical_hash(value) != checksum:
        raise ValueError("artifact_checksum")
    for ref in (
        value["source_artifact_hashes"] + value["code_config_hashes"] + value["raw_shard_hashes"]
    ):
        if sha256_file(Path(ref["path"])) != ref["sha256"]:
            raise ValueError("operand_hash")
    if validate_current_work_receipt(value["current_work_receipt"]):
        raise ValueError("current_receipt")
    if (
        value["completed_count"] != sum(row["status"] == "completed" for row in value["rows"])
        or value["failed_count"] != 630 - value["completed_count"]
    ):
        raise ValueError("count_drift")
    if value["selective_rows_ready_score"] and (
        value["verdict_class"] != "null"
        or not value["required_checks_passed"]
        or value["completed_count"] != 630
        or value["frozen_protocol"]["sha256"] != PROTOCOL_PIN
        or not any(ref["sha256"] == PIN for ref in value["source_artifact_hashes"])
    ):
        raise ValueError("unsafe_readiness")
    if value["rows"]:
        raw = Path(value["measurement_manifest"]["path"]).parent
        cohort, fitted, trace = [
            load(reference(raw / name)) for name in ["population.json", "fit.json", "draws.json"]
        ]
        upstream = load(reference(raw / "upstream.json"))
        if canonical_hash(cohort) != canonical_hash(
            load(upstream["population_manifest"])
        ) or canonical_hash(trace) != canonical_hash(load(upstream["draw_manifest"])):
            raise ValueError("frozen_population_or_draw_drift")
        m.audit_fit(cohort, fitted)
        predictions = m.predict(cohort, fitted)
        if canonical_hash(predictions) != canonical_hash(
            load(reference(raw / "scores.json"))["rows"]
        ):
            raise ValueError("score_drift")
        units = sorted(
            (u for u in cohort["inputs"] if u["split"] == "population"), key=lambda u: u["unit_id"]
        )
        indices = sorted({i for rep in trace["replicates"] for i in rep["calibration"]["indices"]})
        selection = m.select(predictions, m.p.labels([units[i] for i in indices]), trace)
        calculated = m.evaluate(predictions, m.p.labels(units), trace, selection)
        for key, observed in calculated.items():
            if canonical_hash(observed) != canonical_hash(value[key]):
                raise ValueError("reduction_drift:" + key)
    return dict(passed=True, status="replay_passed", sha256=sha256_file(path))


def main(argv: list[str] | None = None) -> int:
    """Publish one checked terminal result without changing milestone activation or history."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", choices=["20261105"], default="20261105")
    parser.add_argument("--output", type=Path, default=m.ROOT / "results" / (m.NAME + ".json"))
    parser.add_argument("--private-output", type=Path)
    parser.add_argument("--upstream", type=Path, default=UPSTREAM)
    parser.add_argument("--cold-replay", type=Path)
    args = parser.parse_args(argv)
    m.progress("start_no_model_load")
    if args.cold_replay:
        try:
            print(json.dumps(replay(args.cold_replay)), flush=True)
            return 0
        except (OSError, ValueError, KeyError, TypeError) as error:
            print(f"replay_rejected: {error}", flush=True)
            return 1
    output = (args.private_output or args.output).absolute()
    if args.private_output and output.resolve().is_relative_to(m.ROOT):
        parser.error("private outputs must remain outside checkout")
    raw = output.parent / "raw" / m.NAME / "invocations" / str(time.time_ns())
    started = time.monotonic_ns()
    pre = preconditions(raw, args.upstream)
    atomic_json(
        raw / "invocation.json",
        dict(
            executable=sys.executable,
            cli=str(m.ROOT / m.CLI),
            arguments=argv if argv is not None else sys.argv[1:],
            cwd=os.getcwd(),
            pid=os.getpid(),
            effective_backend=jax.default_backend(),
            started_monotonic_ns=started,
        ),
    )
    m.progress("after_preconditions")
    result: Json = dict(
        rows=[],
        per_draw_results=[],
        population_score_rows=[],
        selected_cutoff_rows=[],
        certificate_event_rows=[],
        firing_intervals=[],
        conditional_failure_intervals=[],
        served_risk_rows=[],
        paired_comparisons=[],
        selective_utility_score=0,
        learned_selective_benefit_score=0,
    )
    owned_errors = []
    if pre["external_ready"] and pre["resources"]["passed"]:
        try:
            result = measure(raw, pre)
        except (OSError, ValueError, KeyError, TypeError) as error:
            owned_errors.append(str(error))
    fitted = load(reference(raw / "fit.json")) if (raw / "fit.json").is_file() else {}
    owned_errors.extend(
        f"failed_fit_{failure['seed']}:{failure['error']}"
        for failure in fitted.get("failed_seeds", [])
    )
    measured = time.monotonic_ns()
    m.progress("before_owned_validation")
    with TemporaryDirectory(prefix="exp8844-validation-") as directory:
        validation = (
            unqualified() if args.private_output else validate(raw / "validation", Path(directory))
        )
    m.progress("after_owned_validation")
    valid = bool(
        validation["coverage"]["complete"]
        and validation["union"]["passed"]
        and all(r["passed"] for r in validation["receipts"])
    )
    verdict = (
        "disqualified"
        if owned_errors or not pre["resources"]["passed"]
        else "blocked"
        if not pre["external_ready"]
        else "null"
        if valid
        else "disqualified"
    )
    ended = time.monotonic_ns()
    phases = [
        dict(phase="preconditions_measurement", start_s=0.0, end_s=(measured - started) / 1e9),
        dict(
            phase="owned_validation",
            start_s=(measured - started) / 1e9,
            end_s=(ended - started) / 1e9,
        ),
    ]
    training_calls = fitted.get("numerical_training_calls", 0)
    substrate = "no_model_load" if training_calls else "blocked_no_run"
    receipt = build_current_work_receipt(
        run_id=str(started),
        owner_pid=os.getpid(),
        events=[],
        inference_substrate="deterministic_local",
        inference_substrate_class=substrate,
        inference_substrate_details=dict(
            scope="in-process CPU heads and exact finite-population arithmetic"
        ),
        execution_venue="host",
        started_monotonic_ns=started,
        ended_monotonic_ns=ended,
        phase_spans=phases,
        small_ebm_training=dict(
            performed=training_calls > 0, numerical_training_calls=training_calls
        ),
    )
    atomic_json(raw / "current_work_receipt.json", receipt)
    health = Path("/tmp/carnot-8844-repository-health/receipt.json")
    health_refs = []
    if health.is_file():
        health_value = load(reference(health))
        for suffix in ["stdout", "stderr"]:
            target = raw / ("repository_health." + suffix)
            shutil.copyfile(health_value[suffix + "_path"], target)
            health_value[suffix + "_path"] = str(target)
            health_refs.append(reference(target))
        atomic_json(raw / "repository_health.json", health_value)
        health_refs.insert(0, reference(raw / "repository_health.json"))
    value = dict(
        result,
        experiment_id=8844,
        task_id=m.TASK,
        milestone="2026.11.757",
        run_date=args.date,
        honest_verdict=f"complete_{verdict}_selective_availability",
        verdict_class=verdict,
        inference_substrate=receipt["inference_substrate"],
        inference_substrate_class=substrate,
        execution_venue="host",
        MODEL_SPECS=[],
        model_invocation_counts=receipt["invocation_counts"],
        numerical_training_calls=training_calls,
        producer_invocation=reference(raw / "invocation.json"),
        verifier_is_oracle=True,
        independent_generalization_score=0,
        exposure_scope="conditional on one fixed4096-world constructed population; no natural-text or hidden-game generalization",
        duration_s=receipt["duration_s"],
        random_seed=m.p.SEED,
        phase_spans=phases,
        current_work_receipt=receipt,
        preconditions_checked=pre,
        source_artifact_hashes=pre["references"],
        code_config_hashes=[reference(m.ROOT / p) for p in m.OWNED + m.TESTS]
        + [reference(m.ROOT / "scripts/experiment_template.py")],
        raw_shard_hashes=[reference(p) for p in raw.glob("*.json")]
        + validation["raw_references"]
        + health_refs,
        measurement_manifest=reference(raw / "reduction.json")
        if (raw / "reduction.json").is_file()
        else None,
        intended_count=630,
        completed_count=sum(row["status"] == "completed" for row in result["rows"]),
        failed_count=630 - sum(row["status"] == "completed" for row in result["rows"]),
        censored_count=0,
        excluded_count=0,
        independent_count=18000 if result["rows"] else 0,
        gate_check_summary=pre["diagnostics"],
        owned_errors=owned_errors,
        acceptance_gates=dict(
            risk_upper=0.05,
            coverage=0.30,
            useful_replicates=24,
            total_replicates=30,
            utility_scope="preregistered operational utility; not a theorem",
            learned_coverage_gain_lower=0.05,
            comparison_delta=0.01,
            comparisons=10,
            certificate_alpha=m.p.ALPHA,
        ),
        required_checks_passed=valid,
        validation_receipts=validation["receipts"],
        owned_statement_coverage=validation["coverage"],
        selective_rows_ready_score=int(verdict == "null" and len(result["rows"]) == 630),
        frozen_protocol=pre["frozen_protocol"],
        familywise_allocation=dict(
            alpha_per_bound=m.p.ALPHA, delta=0.01, policies=7, budgets=3, thresholds=6
        ),
        methodology="Fit unchanged five CPU heads on120 fitting worlds; seal seven score arrays; use30 independent shared replacement draw pairs; calibrate nested60/180/360 prefixes; seal all cutoffs before240 evaluation draws and4096-world census. Exact binomial availability uncertainty and conservative paired Hoeffding superiority bounds. No empirical universal-validity claim.",
        repository_health=health_refs[0] if health_refs else None,
        flagged_adversarial=False,
        field_principles={},
    )
    value["field_principles"] = {
        k: "Measured producer operand or recomputable reduction; byte hashes bind evidence but do not prove scientific validity."
        for k in value
    }
    value["field_principles"].update(
        conditional_failure_intervals="Denominator is issued certificates; zero issuance yields null conditional failure. Thirty replicates cannot resolve a1-percent failure rate.",
        independent_count="Count18000 RNG draws once; nested budgets, policies, repeats and model seeds add no draws.",
        selective_rows_ready_score="Evidence and owned validation qualify readiness separately from operational utility or learned benefit.",
        selective_utility_score="At least one policy meets census risk<=.05 and coverage>=.30 in24/30 replicates at360; each seed is reported separately.",
        learned_selective_benefit_score="All five seeds must beat both fixed baselines by a simultaneous99-percent lower coverage bound>=.05 at retained risk.",
        per_draw_results="Every evaluation draw retains its original index, including repeats; all refused rules retain scheduled denominators.",
        population_score_rows="All seven policies score the same4096 worlds before calibration labels; census labels open after all cutoff choices seal.",
    )
    value = normalize_artifact_for_template_write(value)
    value["reproducibility_checksum"] = canonical_hash(value)
    prospective = raw / "prospective.json"
    atomic_json(prospective, value)
    if not args.private_output:
        m.progress("before_terminal_checks")
        reports = [
            stream(
                name,
                [
                    sys.executable,
                    "-u",
                    str(m.ROOT / "scripts" / script),
                    *options,
                    str(prospective),
                ],
                raw / "checks",
                deadline=180,
            )
            for name, script, options in [
                ("adversarial", "adversarial_verify.py", ["--json"]),
                ("rows", "verdict_row_consistency_lint.py", ["--strict"]),
            ]
        ]
        reports.append(
            stream(
                "cold_replay",
                [
                    "env",
                    "-u",
                    "PYTHONPATH",
                    "--chdir=/tmp",
                    sys.executable,
                    "-u",
                    str(m.ROOT / m.CLI),
                    "--cold-replay",
                    str(prospective),
                ],
                raw / "checks",
                deadline=600,
            )
        )
        value["validation_receipts"] += reports
        value["raw_shard_hashes"] += [
            reference(Path(r[s + "_path"])) for r in reports for s in ["stdout", "stderr"]
        ]
        if not all(r["passed"] for r in reports):
            value.update(
                honest_verdict="complete_disqualified_selective_availability",
                verdict_class="disqualified",
                selective_rows_ready_score=0,
                required_checks_passed=False,
            )
        del value["reproducibility_checksum"]
        value["reproducibility_checksum"] = canonical_hash(value)
        m.progress("after_terminal_checks")
    publish_primary(output, value, replay)
    m.progress("complete_" + value["verdict_class"], value["completed_count"], 0)
    return 0
