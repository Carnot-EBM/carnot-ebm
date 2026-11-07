"""REQ-REPORT-8855: authenticate historical operands and publish exact diagnostics."""

import argparse
import json
import os
from pathlib import Path
import shutil
import sys
from tempfile import TemporaryDirectory
import time
from typing import Any

import numpy as np
import scipy

from carnot.reporting import selective_ceiling_8855 as m
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
UPSTREAM = m.ROOT / "results/experiment_8844_v757_selective_availability.json"
POPULATION = m.ROOT / "results/experiment_8842_v757_selective_population.json"
PINS = [
    "sha256:12af82304608cc396350c7910cd230f6117fe7e6968b5e78702819bf7402166a",
    "sha256:c6ba96b888d6fc66853ea846f7be857c8238c396cc6aa5f9fa62dc78fad765c9",
]
PROTOCOL_PIN = "sha256:a9d1ea70774fa2417058ed9e66de2c76f0e8de383ed73b10415eac926797e871"
SIZE = 4096
REPLICATES = 30
MEASUREMENT_DEADLINE_S = 300
HEALTH = Path("/tmp/carnot-8855-repository-health/receipt.json")


def preconditions(raw: Path, upstream: Path, population: Path) -> Json:
    """Authenticate original producer bytes before opening scores or census labels."""
    raw.mkdir(parents=True, exist_ok=True, mode=0o700)
    raw.chmod(0o700)
    atomic_json(raw / "writable.json", dict(probe=True))
    refs, diagnostics = [], []
    for i, (path, gate, task) in enumerate(
        [
            (upstream, "selective_rows_ready_score", "exp8844-selective-availability"),
            (population, "selective_protocol_ready_score", "exp8842-selective-population"),
        ]
    ):
        diagnostic: Json = dict(
            upstream=task,
            artifact_field=gate,
            expected=1,
            observed=None,
            path=str(path),
            sha256=None,
        )
        try:
            value = json.loads(path.read_bytes())
            diagnostic.update(sha256=sha256_file(path), observed=value[gate])
            if value[gate] != 1:
                raise ValueError("observed_zero")
            if value["task_id"] != task or diagnostic["sha256"] != PINS[i]:
                raise ValueError("rejected_producer")
            operands = [reference(path)] + [
                ref
                for key in ["source_artifact_hashes", "code_config_hashes", "raw_shard_hashes"]
                for ref in value[key]
            ]
            for ref in operands:
                diagnostic.update(
                    path=ref["path"],
                    artifact_field="sha256",
                    expected=ref["sha256"],
                    observed=None,
                    sha256=None,
                )
                digest = sha256_file(Path(ref["path"]))
                diagnostic.update(observed=digest, sha256=digest)
                if digest != ref["sha256"]:
                    raise ValueError("rejected_operand")
            refs.extend(operands)
            shutil.copyfile(path, raw / f"source_{i}.json")
        except (OSError, ValueError, KeyError, TypeError) as error:
            check = (
                "missing_file"
                if isinstance(error, OSError)
                else "missing_field"
                if isinstance(error, KeyError)
                else "observed_zero"
                if str(error) == "observed_zero"
                else "rejected_producer"
            )
            diagnostics.append(dict(diagnostic, check=check, error=str(error)))
    resources: Json = dict(
        dependencies=dict(numpy=np.__version__, scipy=scipy.__version__),
        tools_available=all(
            (m.ROOT / ".venv/bin" / p).is_file() for p in ["python", "pytest", "ruff", "mypy"]
        ),
        private_output_writable=True,
        private_output_mode=oct(raw.stat().st_mode & 0o777),
        free_disk_bytes=shutil.disk_usage(raw).free,
        task_budget_s=4200,
        measurement_deadline_s=MEASUREMENT_DEADLINE_S,
        validation_deadline_per_child_s=600,
    )
    resources["passed"] = (
        resources["tools_available"] and resources["free_disk_bytes"] >= 512 * 1024**2
    )
    return dict(
        references=list({r["path"]: r for r in refs}.values()),
        diagnostics=diagnostics,
        external_ready=not diagnostics,
        resources=resources,
    )


def measure(raw: Path, *, write: bool = True) -> Json:
    """Recompute historical selection and outcomes using existing raw scores alone."""
    started = time.monotonic()
    source, population = [load(reference(raw / f"source_{i}.json")) for i in range(2)]
    protocol = load(source["frozen_protocol"])
    if (
        source["frozen_protocol"]["sha256"] != PROTOCOL_PIN
        or protocol["alpha"] != m.old.p.ALPHA
        or protocol["thresholds"] != m.old.p.THRESHOLDS
        or protocol["seeds"] != m.old.p.SEEDS
        or protocol["population"] != SIZE
        or source["acceptance_gates"]["risk_upper"] != 0.05
        or source["acceptance_gates"]["coverage"] != 0.30
    ):
        raise ValueError("original_protocol_or_gates")
    score_ref = next(r for r in source["raw_shard_hashes"] if Path(r["path"]).name == "scores.json")
    scores = load(score_ref)["rows"]
    cohort = load(population["population_manifest"])
    trace = load(population["draw_manifest"])
    units = sorted(
        (u for u in cohort["inputs"] if u["split"] == "population"), key=lambda u: u["unit_id"]
    )
    if len(units) != SIZE or len(trace["replicates"]) != REPLICATES:
        raise ValueError("original_population_or_draw_count")
    label_ref = next(r for r in source["raw_shard_hashes"] if Path(r["path"]).name == "labels.json")
    truth = load(label_ref)["labels"]
    population_labels = load(population["labels_manifest"])["labels"]
    if truth != {u["unit_id"]: population_labels[u["unit_id"]] for u in units}:
        raise ValueError("historical_label_drift")
    m.old.p.audit_draws(trace, SIZE)
    m.old.arrays(scores, SIZE)
    if sorted(truth) != sorted({r["unit_id"] for r in scores}):
        raise ValueError("population_unit_drift")
    selection = m.old.select(scores, truth, trace)
    calculated = m.old.evaluate(scores, truth, trace, selection)
    for key, value in calculated.items():
        if canonical_hash(value) != canonical_hash(source[key]):
            raise ValueError("historical_reduction_drift:" + key)
    for name, value in [
        ("scores", scores),
        ("population", cohort),
        ("draws", trace),
        ("protocol", protocol),
        ("labels", truth),
    ]:
        path = raw / (name + ".json")
        if write:
            atomic_json(path, dict(value=value))
        elif canonical_hash(load(reference(path))["value"]) != canonical_hash(value):
            raise ValueError("retained_operand_drift:" + name)
    result = m.diagnose(calculated["population_score_rows"])
    result["historical_recomputation"] = dict(
        rows=calculated["rows"],
        certificate_firing_count=calculated["certificate_firing_count"],
        certificate_failure_count=calculated["certificate_failure_count"],
        selective_utility_score=calculated["selective_utility_score"],
        selection_sha256=calculated["selection_sha256"],
        per_draw_results_sha256=canonical_hash(calculated["per_draw_results"]),
        population_score_rows_sha256=canonical_hash(calculated["population_score_rows"]),
    )
    if write:
        atomic_json(raw / "reduction.json", result)
    if time.monotonic() - started > MEASUREMENT_DEADLINE_S:
        raise ValueError("measurement_deadline")
    return result


def validate(raw: Path, private: Path) -> Json:
    """Use unchanged private E2E readers and actual unit/child statement accounting."""
    checker = namespace(inherited, "selective8855_validation")
    checker.m = m
    return dict(checker.validate(raw, private))


def replay(path: Path) -> Json:
    """Recompute from pinned historical bytes, rejecting self-consistently rehashed claims."""
    value = json.loads(path.read_bytes())
    checksum = value.pop("reproducibility_checksum")
    if canonical_hash(value) != checksum:
        raise ValueError("artifact_checksum")
    if value["experiment_id"] != 8855 or value["task_id"] != m.TASK:
        raise ValueError("producer_identity")
    for ref in (
        value["source_artifact_hashes"] + value["code_config_hashes"] + value["raw_shard_hashes"]
    ):
        if sha256_file(Path(ref["path"])) != ref["sha256"]:
            raise ValueError("operand_hash")
    if validate_current_work_receipt(value["current_work_receipt"]):
        raise ValueError("current_receipt")
    completed = sum(r["status"] == "completed" for r in value["rows"])
    if (
        len(value["rows"]) != 42
        or value["completed_count"] != completed
        or value["failed_count"] != 42 - completed
    ):
        raise ValueError("count_drift")
    if value["ceiling_rows_ready_score"] and (
        not value["required_checks_passed"]
        or completed != 42
        or value["verdict_class"] not in {"null", "circular_positive"}
    ):
        raise ValueError("unsafe_readiness")
    manifest = value["measurement_manifest"]
    if manifest:
        raw = Path(manifest["path"]).parent
        if any(sha256_file(raw / f"source_{i}.json") != PINS[i] for i in range(2)):
            raise ValueError("historical_producer_pin")
        result = measure(raw, write=False)
        if canonical_hash(load(manifest)) != canonical_hash(result):
            raise ValueError("retained_reduction_drift")
        for key, observed in result.items():
            if canonical_hash(observed) != canonical_hash(value[key]):
                raise ValueError("reduction_drift:" + key)
    elif completed or value["split_study_feasible_score"]:
        raise ValueError("missing_measurement")
    return dict(passed=True, status="replay_passed", sha256=sha256_file(path))


def main(argv: list[str] | None = None) -> int:
    """Publish a terminal diagnostic without changing any milestone or historical primary."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", choices=["20261106"], default="20261106")
    parser.add_argument("--output", type=Path, default=m.ROOT / "results" / (m.NAME + ".json"))
    parser.add_argument("--private-output", type=Path)
    parser.add_argument("--upstream", type=Path, default=UPSTREAM)
    parser.add_argument("--population", type=Path, default=POPULATION)
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
    pre = preconditions(raw, args.upstream, args.population)
    atomic_json(
        raw / "invocation.json",
        dict(
            executable=sys.executable,
            arguments=argv if argv is not None else sys.argv[1:],
            cwd=os.getcwd(),
            pid=os.getpid(),
            started_monotonic_ns=started,
            cli=str(m.ROOT / m.CLI),
        ),
    )
    m.progress("after_preconditions_before_recomputation")
    reason = pre["diagnostics"][0]["check"] if pre["diagnostics"] else "owned_failure"
    result: Json = dict(
        rows=[
            dict(
                unit_id=f"{a}/{s}/{c}",
                cluster_id="frozen_population",
                arm=a,
                seed=s,
                threshold=c,
                denominator=SIZE,
                metric=None,
                status="failed",
                missing_reason=reason,
            )
            for a, s in m.old.POLICIES
            for c in m.old.p.THRESHOLDS
        ],
        population_cutoff_rows=[],
        exact_sample_requirements=[],
        availability_power_rows=[],
        failure_decomposition=[],
        split_study_feasible_score=0,
    )
    errors = []
    if pre["external_ready"] and pre["resources"]["passed"]:
        try:
            result = measure(raw)
        except (OSError, ValueError, KeyError, TypeError, StopIteration) as error:
            errors.append(str(error))
    measured = time.monotonic_ns()
    m.progress("before_owned_validation")
    with TemporaryDirectory(prefix="exp8855-validation-") as directory:
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
        if errors or not pre["resources"]["passed"]
        else "blocked"
        if not pre["external_ready"]
        else "disqualified"
        if not valid
        else "circular_positive"
        if result["split_study_feasible_score"]
        else "null"
    )
    ended = time.monotonic_ns()
    phases = [
        dict(
            phase="preconditions_and_exact_reduction", start_s=0.0, end_s=(measured - started) / 1e9
        ),
        dict(
            phase="owned_validation",
            start_s=(measured - started) / 1e9,
            end_s=(ended - started) / 1e9,
        ),
    ]
    substrate = "no_model_load" if result["population_cutoff_rows"] else "blocked_no_run"
    receipt = build_current_work_receipt(
        run_id=str(started),
        owner_pid=os.getpid(),
        events=[],
        inference_substrate="aggregation_from_upstream_artifacts",
        inference_substrate_class=substrate,
        inference_substrate_details=dict(
            scope="retained scores and exact CPU binomial sums; no fitting"
        ),
        execution_venue="host",
        started_monotonic_ns=started,
        ended_monotonic_ns=ended,
        phase_spans=phases,
    )
    atomic_json(raw / "current_work_receipt.json", receipt)
    health = HEALTH
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
    completed = sum(row["status"] == "completed" for row in result["rows"])
    value = dict(
        result,
        experiment_id=8855,
        task_id=m.TASK,
        milestone="2026.11.758",
        run_date=args.date,
        honest_verdict=f"complete_{verdict}_{reason if verdict == 'blocked' else 'selective_ceiling'}",
        verdict_class=verdict,
        inference_substrate=receipt["inference_substrate"],
        inference_substrate_class=substrate,
        execution_venue="host",
        MODEL_SPECS=[],
        model_invocation_counts=receipt["invocation_counts"],
        numerical_training_calls=0,
        verifier_is_oracle=True,
        exposure_scope="exposed fixed4096-world constructed census; no deployment rule or generalization result",
        independent_generalization_score=0,
        duration_s=receipt["duration_s"],
        phase_spans=phases,
        preconditions_checked=pre,
        random_seed=m.old.p.SEED,
        current_work_receipt=receipt,
        producer_invocation=reference(raw / "invocation.json"),
        source_artifact_hashes=pre["references"],
        code_config_hashes=[
            reference(m.ROOT / p)
            for p in m.OWNED
            + m.TESTS
            + [
                "scripts/experiment_template.py",
                "python/carnot/reporting/current_work_receipt.py",
                "python/carnot/reporting/primary_publication.py",
            ]
        ],
        raw_shard_hashes=[reference(p) for p in raw.glob("*.json")]
        + validation["raw_references"]
        + health_refs,
        intended_count=42,
        completed_count=completed,
        failed_count=42 - completed,
        censored_count=0,
        excluded_count=0,
        independent_count=0,
        gate_check_summary=pre["diagnostics"],
        owned_errors=errors,
        acceptance_gates=dict(
            risk_upper=0.05,
            coverage=0.30,
            projected_availability=0.8,
            total_labels=720,
            selection_labels=240,
            certification_labels=480,
            split_alpha=0.01 / (2 * 7),
            matched_simultaneous_alpha=0.01 / (2 * 7 * 6),
            historical_alpha=m.old.p.ALPHA,
        ),
        required_checks_passed=valid,
        validation_receipts=validation["receipts"],
        owned_statement_coverage=validation["coverage"],
        ceiling_rows_ready_score=int(verdict in {"null", "circular_positive"} and completed == 42),
        original_protocol_hash=PROTOCOL_PIN,
        measurement_manifest=reference(raw / "reduction.json")
        if result["population_cutoff_rows"] and (raw / "reduction.json").is_file()
        else None,
        repository_health=health_refs[0] if health_refs else None,
        methodology="Exact census of all seven existing policies at all six cutoffs. Binomial accepted-count mixture with conditional exact error tails. No fitted parameters or new draws. Split projection conditions on independently selecting each fixed rule; 240 selection labels are charged. No superiority or transfer claim.",
        methodology_note="Exact finite-population arithmetic can yield zero or unit probabilities. These are constructed diagnostics, not measured natural-text accuracy.",
        flagged_adversarial=False,
        field_principles={},
    )
    value["field_principles"] = {
        k: "Retained producer operand or recomputable diagnostic; hashes establish custody, not scientific validity."
        for k in value
    }
    value["field_principles"].update(
        independent_count="No new independent observations; repeated census rules and the five seeds do not add independent evidence.",
        population_cutoff_rows="All 42 exposed census rules, including empty accepted sets with null risk; no deployment selection.",
        exact_sample_requirements="Minimum accepted counts for 0 through 10 errors under each exact allocation; integer predecessor checks prevent rounding errors.",
        availability_power_rows="Exact P(N accepted)*P(E passes|N), conditional on independent fixed-rule selection; selection success is unmeasured.",
        failure_decomposition="Each arm retains discrimination and certificate support separately; selection uncertainty remains even with high conditional power.",
        split_study_feasible_score="At least one registered non-oracle fixed rule meets census risk and coverage plus .8 conditional split power at 720 paid labels; permits a new test only.",
        ceiling_rows_ready_score="Complete authenticated diagnostics and successful owned checks, independent of scientific benefit.",
        original_protocol_hash="Byte pin of the historical protocol; the matched two-protocol study uses a separately declared allocation.",
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
                deadline=300,
            )
        )
        value["validation_receipts"] += reports
        value["raw_shard_hashes"] += [
            reference(Path(r[s + "_path"])) for r in reports for s in ["stdout", "stderr"]
        ]
        if not all(r["passed"] for r in reports):
            value.update(
                honest_verdict="complete_disqualified_selective_ceiling",
                verdict_class="disqualified",
                ceiling_rows_ready_score=0,
                required_checks_passed=False,
            )
        del value["reproducibility_checksum"]
        value["reproducibility_checksum"] = canonical_hash(value)
        m.progress("after_terminal_checks")
    publish_primary(output, value, replay)
    m.progress("complete_" + value["verdict_class"], completed, 42 - completed)
    return 0
