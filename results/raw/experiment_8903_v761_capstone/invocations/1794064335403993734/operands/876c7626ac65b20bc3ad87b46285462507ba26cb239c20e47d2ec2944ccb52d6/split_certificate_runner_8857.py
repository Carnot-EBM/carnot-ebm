"""REQ-REPORT-8857: seal the independent protocol and publish only replayable evidence."""

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

from carnot.reporting import split_certificate_methods_8857 as m
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
UPSTREAM = m.ROOT / "results/experiment_8855_v758_selective_ceiling.json"
PIN = "sha256:e73a26ed67c69cd730375d6c39cb4aab8d278d0849abb424b77a493b7b56edbe"
SIZE, REPLICATES, MEASUREMENT_DEADLINE_S = 4096, 60, 3000
HEALTH = Path("/tmp/carnot-8857-repository-health/receipt.json")


def preconditions(raw: Path, upstream: Path) -> Json:
    """Authenticate feasibility and every original bound operand before spending new labels."""
    raw.mkdir(parents=True, exist_ok=True, mode=0o700)
    raw.chmod(0o700)
    atomic_json(raw / "writable.json", dict(probe=True))
    refs, diagnostics = [], []
    diagnostic: Json = dict(
        upstream="exp8855-selective-ceiling",
        artifact_field="split_study_feasible_score",
        expected=1,
        observed=None,
        path=str(upstream),
        sha256=None,
    )
    try:
        value = json.loads(upstream.read_bytes())
        diagnostic.update(
            observed=value["split_study_feasible_score"], sha256=sha256_file(upstream)
        )
        if value["split_study_feasible_score"] != 1:
            raise ValueError("observed_zero")
        if (
            value["task_id"] != "exp8855-selective-ceiling"
            or value["ceiling_rows_ready_score"] != 1
            or sha256_file(upstream) != PIN
        ):
            raise ValueError("rejected_producer")
        refs = [reference(upstream)] + [
            ref
            for k in ["source_artifact_hashes", "code_config_hashes", "raw_shard_hashes"]
            for ref in value[k]
        ]
        for ref in refs:
            diagnostic.update(
                path=ref["path"], artifact_field="sha256", expected=ref["sha256"], observed=None
            )
            diagnostic["observed"] = sha256_file(Path(ref["path"]))
            if diagnostic["observed"] != ref["sha256"]:
                raise ValueError("rejected_operand")
        shutil.copyfile(upstream, raw / "source.json")
    except (OSError, KeyError, ValueError, TypeError) as error:
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
        refs = []
    resources: Json = dict(
        dependencies=dict(numpy=np.__version__, scipy=scipy.__version__),
        private_output_writable=True,
        private_mode=oct(raw.stat().st_mode & 0o777),
        free_disk_bytes=shutil.disk_usage(raw).free,
        task_budget_s=4200,
        measurement_deadline_s=MEASUREMENT_DEADLINE_S,
        tools_available=all(
            (m.ROOT / ".venv/bin" / n).is_file() for n in ["python", "pytest", "ruff", "mypy"]
        ),
    )
    resources["passed"] = (
        resources["tools_available"] and resources["free_disk_bytes"] >= 512 * 1024**2
    )
    return dict(
        references=refs,
        diagnostics=diagnostics,
        external_ready=not diagnostics,
        resources=resources,
    )


def measure(raw: Path) -> Json:
    """Seal each stage before opening the next stage's labels, including the final census."""
    started = time.monotonic()
    if MEASUREMENT_DEADLINE_S <= 0:
        raise ValueError("measurement_deadline")
    access: list[Json] = []

    def seal(name: str, value: Json) -> None:
        atomic_json(raw / (name + ".json"), value)
        access.append(
            dict(stage=name, monotonic_ns=time.monotonic_ns(), **reference(raw / (name + ".json")))
        )
        atomic_json(raw / "access_order.json", dict(events=access))
        m.progress("sealed_" + name, len(access), 13 - len(access))
        if time.monotonic() - started > MEASUREMENT_DEADLINE_S:
            raise ValueError("measurement_deadline")

    source = load(reference(raw / "source.json"))
    historical = load(
        next(x for x in source["raw_shard_hashes"] if Path(x["path"]).name == "population.json")
    )["value"]
    exposed = set(historical["exposed_invariants"]) | {w["structure"] for w in historical["worlds"]}
    seal("protocol", m.protocol())
    cohort = m.freeze(exposed, population_count=SIZE)
    seal("population", cohort)
    trace = m.draws(SIZE, replicates=REPLICATES)
    seal("draws", trace)
    fitted = m.old.fit(cohort)
    seal("fit", fitted)
    m.old.audit_fit(cohort, fitted)
    scores = m.old.predict(cohort, fitted)
    seal("scores", dict(rows=scores))
    units = sorted(
        [u for u in cohort["inputs"] if u["split"] == "population"], key=lambda u: u["unit_id"]
    )
    label_cache: Json = {}

    def release(name: str, indices: set[int]) -> Json:
        """Reuse only already released labels; replacement draws still incur their full budget."""
        requested = [units[i] for i in sorted(indices)]
        label_cache.update(
            m.old.p.labels([u for u in requested if u["unit_id"] not in label_cache])
        )
        truth = {u["unit_id"]: label_cache[u["unit_id"]] for u in requested}
        seal(name, dict(labels=truth))
        return dict(truth)

    selection_truth = release(
        "selection_labels",
        {x["index"] for rep in trace["replicates"] for x in rep["calibration"][:240]},
    )
    selection = m.select(scores, selection_truth, trace)
    seal("selection", selection)
    certification_truth = release(
        "certification_labels",
        {x["index"] for rep in trace["replicates"] for x in rep["calibration"][240:]},
    )
    calibration_truth = dict(selection_truth, **certification_truth)
    certificates = m.certify(scores, calibration_truth, trace, selection)
    seal("certificates", certificates)
    evaluation_truth = release(
        "evaluation_labels", {x["index"] for rep in trace["replicates"] for x in rep["evaluation"]}
    )
    rows = m.evaluate(scores, evaluation_truth, trace, certificates)
    seal("evaluation", dict(rows=rows))
    truth = release("census_labels", set(range(SIZE)))
    rows = m.census(scores, truth, rows)
    result = dict(
        rows=rows, **m.summarize(rows), numerical_training_calls=fitted["numerical_training_calls"]
    )
    seal("census", result)
    return result


def reduction(raw: Path) -> Json:
    """Replay every decision from immutable operands; neither a new checksum nor a changed order grants credit."""

    def get(name: str) -> Json:
        """Open only the retained bytes named by this invocation."""
        return load(reference(raw / (name + ".json")))

    if canonical_hash(get("protocol")) != canonical_hash(m.protocol()):
        raise ValueError("protocol_drift")
    cohort, trace = get("population"), get("draws")
    structures = [w["structure"] for w in cohort["worlds"]]
    if (
        len(structures) != len(set(structures))
        or set(structures) & set(cohort["exposed_invariants"])
        or trace != m.draws(SIZE, replicates=REPLICATES)
    ):
        raise ValueError("population_or_draw_drift")
    access = get("access_order")["events"]
    expected = ["protocol"] + m.protocol()["access_order"]
    if [e["stage"] for e in access] != expected or any(
        a["monotonic_ns"] > b["monotonic_ns"] for a, b in zip(access, access[1:])
    ):
        raise ValueError("access_order_drift")
    for event in access:
        load(event)
    scores = get("scores")["rows"]
    selection = m.select(scores, get("selection_labels")["labels"], trace)
    certificates = m.certify(
        scores,
        dict(get("selection_labels")["labels"], **get("certification_labels")["labels"]),
        trace,
        selection,
    )
    evaluation = m.evaluate(scores, get("evaluation_labels")["labels"], trace, certificates)
    rows = m.census(scores, get("census_labels")["labels"], evaluation)
    for name, result in [
        ("selection", selection),
        ("certificates", certificates),
        ("evaluation", dict(rows=evaluation)),
    ]:
        if canonical_hash(get(name)) != canonical_hash(result):
            raise ValueError("stage_reduction_drift:" + name)
    return dict(
        rows=rows,
        **m.summarize(rows),
        numerical_training_calls=get("fit")["numerical_training_calls"],
    )


def validate(raw: Path, private: Path) -> Json:
    """Reuse unchanged E2E018/021 readers and real unit/child statement accounting."""
    checker = namespace(inherited, "split8857_validation")
    checker.m = m
    return dict(checker.validate(raw, private))


def replay(path: Path) -> Json:
    """Recompute semantic claims in addition to checking byte hashes and receipt identity."""
    value = json.loads(path.read_bytes())
    checksum = value.pop("reproducibility_checksum")
    if canonical_hash(value) != checksum:
        raise ValueError("artifact_checksum")
    if value["experiment_id"] != 8857 or value["task_id"] != m.TASK:
        raise ValueError("producer_identity")
    frozen = dict(
        run_date="20261106",
        milestone="2026.11.758",
        acceptance_gates=dict(risk_upper=0.05, coverage=0.30, benefit_lower_99_gt=0.05),
        **{
            key: m.protocol()[key]
            for key in [
                "familywise_allocation",
                "policy_menu",
                "cutoff_rule",
                "expected_row_counts",
            ]
        },
    )
    if any(
        canonical_hash(value[key]) != canonical_hash(expected) for key, expected in frozen.items()
    ):
        raise ValueError("frozen_contract_drift")
    for ref in (
        value["source_artifact_hashes"] + value["code_config_hashes"] + value["raw_shard_hashes"]
    ):
        if sha256_file(Path(ref["path"])) != ref["sha256"]:
            raise ValueError("operand_hash")
    if validate_current_work_receipt(value["current_work_receipt"]):
        raise ValueError("current_receipt")
    count = len(value["rows"])
    completed = sum(x["status"] == "completed" for x in value["rows"])
    ready = value["split_methods_ready_score"]
    if (
        count != REPLICATES * 14
        or value["completed_count"] != completed
        or value["failed_count"] + value["censored_count"] != count - completed
        or ready not in [0, 1]
        or (
            ready
            and (
                not value["required_checks_passed"]
                or completed != count
                or value["verdict_class"] not in {"null", "circular_positive"}
            )
        )
    ):
        raise ValueError("count_or_readiness_drift")
    if value["protocol_hash"] != canonical_hash(m.protocol()):
        raise ValueError("protocol_hash_drift")
    if completed:
        raw = Path(value["population_manifest"]["path"]).parent
        if sha256_file(raw / "source.json") != PIN:
            raise ValueError("producer_pin")
        result = reduction(raw)
        if canonical_hash(load(reference(raw / "census.json"))) != canonical_hash(result):
            raise ValueError("retained_reduction_drift")
        for key, observed in result.items():
            if canonical_hash(value[key]) != canonical_hash(observed):
                raise ValueError("reduction_drift:" + key)
    elif value["protocol_benefit_score"] or ready:
        raise ValueError("missing_measurement")
    return dict(passed=True, status="replay_passed", sha256=sha256_file(path))


def main(argv: list[str] | None = None) -> int:
    """Execute this research only; private controls cannot certify production readiness."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", choices=["20261106"], default="20261106")
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
            arguments=argv if argv is not None else sys.argv[1:],
            cwd=os.getcwd(),
            pid=os.getpid(),
            started_monotonic_ns=started,
        ),
    )
    m.progress("after_preconditions_before_measurement")
    reason = pre["diagnostics"][0]["check"] if pre["diagnostics"] else "owned_failure"
    result: Json = dict(
        rows=[
            dict(
                unit_id=f"{rep}/{name}/{a}/{s}",
                cluster_id=f"rep/{rep}",
                replicate=rep,
                protocol=name,
                arm=a,
                seed=s,
                metric=None,
                denominator=240,
                status="failed",
                missing_reason=reason,
            )
            for rep in range(REPLICATES)
            for name in m.ALPHAS
            for a, s in m.POLICIES
        ],
        protocol_benefit_score=0,
        primary_contrast=None,
        policy_summaries=[],
        numerical_training_calls=0,
    )
    errors = []
    if pre["external_ready"] and pre["resources"]["passed"]:
        try:
            result = measure(raw)
        except (OSError, ValueError, KeyError, TypeError, RuntimeError, StopIteration) as error:
            errors.append(str(error))
    measured = time.monotonic_ns()
    m.progress("before_owned_validation")
    with TemporaryDirectory(prefix="exp8857-validation-") as directory:
        validation = (
            unqualified() if args.private_output else validate(raw / "validation", Path(directory))
        )
    valid = bool(
        validation["coverage"]["complete"]
        and validation["union"]["passed"]
        and all(x["passed"] for x in validation["receipts"])
    )
    m.progress("after_owned_validation")
    verdict = (
        "disqualified"
        if errors or not pre["resources"]["passed"]
        else "blocked"
        if not pre["external_ready"]
        else "disqualified"
        if not valid
        else "circular_positive"
        if result["protocol_benefit_score"]
        else "null"
    )
    ended = time.monotonic_ns()
    phases = [
        dict(phase="preconditions_and_measurement", start_s=0.0, end_s=(measured - started) / 1e9),
        dict(
            phase="owned_validation",
            start_s=(measured - started) / 1e9,
            end_s=(ended - started) / 1e9,
        ),
    ]
    completed = sum(x["status"] == "completed" for x in result["rows"])
    substrate = "no_model_load" if completed else "blocked_no_run"
    receipt = build_current_work_receipt(
        run_id=str(started),
        owner_pid=os.getpid(),
        events=[],
        inference_substrate="cpu_numerical_training_and_exact_certificates",
        inference_substrate_class=substrate,
        inference_substrate_details=dict(scope="fresh small CPU heads; no pretrained model"),
        execution_venue="host",
        started_monotonic_ns=started,
        ended_monotonic_ns=ended,
        phase_spans=phases,
        small_ebm_training=dict(
            performed=bool(result["numerical_training_calls"]),
            numerical_training_calls=result["numerical_training_calls"],
        ),
    )
    atomic_json(raw / "current_work_receipt.json", receipt)
    health_refs = []
    if HEALTH.is_file():
        health = load(reference(HEALTH))
        for suffix in ["stdout", "stderr"]:
            target = raw / ("repository_health." + suffix)
            shutil.copyfile(health[suffix + "_path"], target)
            health[suffix + "_path"] = str(target)
            health_refs.append(reference(target))
        atomic_json(raw / "repository_health.json", health)
        health_refs.insert(0, reference(raw / "repository_health.json"))
    protocol = m.protocol()
    value: Json = dict(
        result,
        experiment_id=8857,
        task_id=m.TASK,
        milestone="2026.11.758",
        run_date=args.date,
        honest_verdict=f"complete_{verdict}_{reason if verdict == 'blocked' else 'split_certificate_methods'}",
        verdict_class=verdict,
        inference_substrate=receipt["inference_substrate"],
        inference_substrate_class=substrate,
        execution_venue="host",
        MODEL_SPECS=[],
        model_invocation_counts=receipt["invocation_counts"],
        verifier_is_oracle=True,
        independent_generalization_score=0,
        exposure_scope="new constructed population; exact oracle labels; no natural-language or learned-head superiority claim",
        duration_s=receipt["duration_s"],
        phase_spans=phases,
        preconditions_checked=pre,
        random_seed=m.SEED,
        current_work_receipt=receipt,
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
        repository_health=health_refs[0] if health_refs else None,
        intended_count=REPLICATES * 14,
        completed_count=completed,
        failed_count=REPLICATES * 14 - completed,
        censored_count=0,
        excluded_count=0,
        independent_count=REPLICATES if completed else 0,
        gate_check_summary=pre["diagnostics"],
        owned_errors=errors,
        acceptance_gates=dict(risk_upper=0.05, coverage=0.30, benefit_lower_99_gt=0.05),
        required_checks_passed=valid,
        validation_receipts=validation["receipts"],
        owned_statement_coverage=validation["coverage"],
        split_methods_ready_score=int(
            verdict in {"null", "circular_positive"} and completed == REPLICATES * 14
        ),
        protocol_hash=canonical_hash(protocol),
        population_manifest=reference(raw / "population.json") if completed else None,
        draw_manifest=reference(raw / "draws.json") if completed else None,
        policy_menu=protocol["policy_menu"],
        cutoff_rule=protocol["cutoff_rule"],
        familywise_allocation=protocol["familywise_allocation"],
        expected_row_counts=protocol["expected_row_counts"],
        methodology="Freeze a new finite population before independent replacement draws. Fit the unchanged seven policies. Compare matched720-label simultaneous certification with240 selection plus480 independent certification. Freeze selection before certification, evaluate independent draws, then open census. Paired replicate bootstrap averages all seven policies; refusal serves zero.",
        methodology_note="Constructed exact labels certify only this population. CPU numerical training is recorded without pretrained loads. No new neural architecture or milestone activation.",
        flagged_adversarial=False,
        field_principles={},
    )
    value["field_principles"] = {
        k: "Measured operand or frozen protocol claim, bound to immutable sources; custody alone is not scientific evidence."
        for k in value
    }
    value["field_principles"].update(
        primary_contrast="One paired replicate contrast averages all seven policies; bootstrap resamples replicates, not policies.",
        split_methods_ready_score="Independent stages and authenticated complete operands plus successful owned controls, regardless of benefit.",
        independent_count="Sixty independent replicate draw streams; model seeds and repeated population support add no independent replicates.",
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
            reference(Path(x[s + "_path"])) for x in reports for s in ["stdout", "stderr"]
        ]
        if not all(x["passed"] for x in reports):
            value.update(
                honest_verdict="complete_disqualified_split_certificate_methods",
                verdict_class="disqualified",
                split_methods_ready_score=0,
                required_checks_passed=False,
            )
        del value["reproducibility_checksum"]
        value["reproducibility_checksum"] = canonical_hash(value)
        m.progress("after_terminal_checks")
    publish_primary(output, value, replay)
    m.progress("published_" + value["verdict_class"], completed, REPLICATES * 14 - completed)
    return 0
