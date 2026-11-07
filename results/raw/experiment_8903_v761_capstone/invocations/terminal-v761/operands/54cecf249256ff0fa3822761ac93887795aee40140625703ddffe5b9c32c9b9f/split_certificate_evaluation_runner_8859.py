"""REQ-REPORT-8859: fresh fits, sealed label access and immutable evaluation operands."""

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

from carnot.reporting import split_certificate_evaluation_8859 as m
from carnot.reporting import split_certificate_runner_8857 as previous
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
UPSTREAM = m.ROOT / "results/experiment_8857_v758_split_certificate_methods.json"
PIN = "sha256:1f3adf1e780cdd2a7ecf194a5436504b146986820b071fd0f016aed922734dc6"
SIZE, REPLICATES, MEASUREMENT_DEADLINE_S = 4096, 60, 2400
HEALTH = Path("/tmp/carnot-8859-repository-health/receipt.json")
GATES = dict(risk_upper=0.05, coverage=0.30, benefit_lower_99_gt=0.05)


def preconditions(raw: Path, upstream: Path) -> Json:
    """Authenticate original authorities and exact method choices before any new label release."""
    raw.mkdir(parents=True, exist_ok=True, mode=0o700)
    raw.chmod(0o700)
    atomic_json(raw / "writable.json", dict(probe=True))
    refs, diagnostics = [], []
    diagnostic: Json = dict(
        upstream="exp8857-split-certificate-methods",
        artifact_field="split_methods_ready_score",
        expected=1,
        observed=None,
        path=str(upstream),
        sha256=None,
    )
    try:
        value = json.loads(upstream.read_bytes())
        diagnostic.update(observed=value["split_methods_ready_score"], sha256=sha256_file(upstream))
        if value["split_methods_ready_score"] != 1:
            raise ValueError("observed_zero")
        if value["task_id"] != "exp8857-split-certificate-methods" or sha256_file(upstream) != PIN:
            raise ValueError("rejected_producer")
        diagnostic.update(
            artifact_field="protocol_hash",
            expected=m.METHOD_PROTOCOL_PIN,
            observed=value["protocol_hash"],
        )
        if (
            value["protocol_hash"] != m.METHOD_PROTOCOL_PIN
            or canonical_hash(m.registered.protocol()) != m.METHOD_PROTOCOL_PIN
        ):
            raise ValueError("rejected_producer")
        refs = [reference(upstream)] + [
            ref
            for key in ["source_artifact_hashes", "code_config_hashes", "raw_shard_hashes"]
            for ref in value[key]
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
    """Seal fitting, scores, selection and certification before independent evaluation and census."""
    started = time.monotonic()
    if MEASUREMENT_DEADLINE_S <= 0:
        raise ValueError("measurement_deadline")
    access: list[Json] = []

    def seal(name: str, value: Json) -> None:
        """A deadline never removes previously completed units or extends on progress output."""
        atomic_json(raw / (name + ".json"), value)
        access.append(
            dict(stage=name, monotonic_ns=time.monotonic_ns(), **reference(raw / (name + ".json")))
        )
        atomic_json(raw / "access_order.json", dict(events=access))
        m.progress("sealed_" + name, len(access), 14 - len(access))
        if time.monotonic() - started > MEASUREMENT_DEADLINE_S:
            raise ValueError("measurement_deadline")

    historical = load(load(reference(raw / "source.json"))["population_manifest"])
    exposed = set(historical["exposed_invariants"]) | {w["structure"] for w in historical["worlds"]}
    seal("protocol", m.protocol())
    cohort = m.registered.freeze(exposed, population_count=SIZE)
    seal("population", cohort)
    trace = m.draws(SIZE, replicates=REPLICATES)
    seal("draws", trace)
    m.progress("before_five_fresh_heads_no_model_load")
    fitted = m.old.fit(cohort)
    seal("fit", fitted)
    m.old.audit_fit(cohort, fitted)
    m.progress(
        "after_five_fresh_heads_no_model_load",
        5 - len(fitted["failed_seeds"]),
        len(fitted["failed_seeds"]),
    )
    scores = m.old.predict(cohort, fitted)
    seal("scores", dict(rows=scores))
    units = sorted(
        [u for u in cohort["inputs"] if u["split"] == "population"], key=lambda u: u["unit_id"]
    )
    cache: Json = {}

    def release(name: str, indices: set[int]) -> Json:
        """Only authorized indices receive labels; repeated draws retain their paid count."""
        requested = [units[i] for i in sorted(indices)]
        cache.update(m.old.p.labels([u for u in requested if u["unit_id"] not in cache]))
        truth = {u["unit_id"]: cache[u["unit_id"]] for u in requested}
        seal(name, dict(labels=truth))
        return truth

    selection_truth = release(
        "selection_labels",
        {x["index"] for rep in trace["replicates"] for x in rep["calibration"][:240]},
    )
    selection = m.select(scores, selection_truth, trace)
    seal("selection", selection)
    cert_truth = release(
        "certification_labels",
        {x["index"] for rep in trace["replicates"] for x in rep["calibration"][240:]},
    )
    certificates = m.certify(scores, dict(selection_truth, **cert_truth), trace, selection)
    seal("certificates", certificates)
    truth = release(
        "evaluation_labels", {x["index"] for rep in trace["replicates"] for x in rep["evaluation"]}
    )
    rows = m.evaluate(scores, truth, trace, certificates)
    seal("evaluation", dict(rows=rows))
    atomic_json(raw / "per_draw.json", dict(rows=m.per_draw(scores, truth, trace, certificates)))
    truth = release("census_labels", set(range(SIZE)))
    rows = m.census(scores, truth, rows)
    result = dict(
        rows=rows,
        **m.summarize(rows),
        numerical_training_calls=fitted["numerical_training_calls"],
        selected_cutoff_rows=selection["choices"],
        per_draw_results=reference(raw / "per_draw.json"),
    )
    seal("census", result)
    return result


def reduction(raw: Path) -> Json:
    """Rebuild stage decisions and independent labels; rehashing false operands cannot pass."""
    cohort = load(reference(raw / "population.json"))
    scores = load(reference(raw / "scores.json"))["rows"]
    fitted = load(reference(raw / "fit.json"))
    m.old.audit_fit(cohort, fitted)
    if canonical_hash(scores) != canonical_hash(m.old.predict(cohort, fitted)):
        raise ValueError("score_drift")
    all_truth = m.old.p.labels([u for u in cohort["inputs"] if u["split"] == "population"])
    for name in ["selection_labels", "certification_labels", "evaluation_labels", "census_labels"]:
        labels = load(reference(raw / (name + ".json")))["labels"]
        if labels != {k: all_truth[k] for k in labels}:
            raise ValueError("label_drift")
    replay = namespace(previous, "evaluation8859_reduction")
    replay.m = m
    replay.SIZE = SIZE
    replay.REPLICATES = REPLICATES
    result: Json = replay.reduction(raw)
    cert = load(reference(raw / "certificates.json"))
    trace = load(reference(raw / "draws.json"))
    observed = load(reference(raw / "per_draw.json"))
    if canonical_hash(observed) != canonical_hash(
        dict(rows=m.per_draw(scores, all_truth, trace, cert))
    ):
        raise ValueError("draw_decision_drift")
    result.update(
        selected_cutoff_rows=load(reference(raw / "selection.json"))["choices"],
        per_draw_results=reference(raw / "per_draw.json"),
    )
    return result


def validate(raw: Path, private: Path) -> Json:
    """Existing private E2E018/021 checks collect real unit and child coverage with no exclusions."""
    checker = namespace(inherited, "evaluation8859_validation")
    checker.m = m
    return dict(checker.validate(raw, private))


def replay(path: Path) -> Json:
    """Reject changed identities, scientific reductions, fixed gates and unsafe readiness."""
    value = json.loads(path.read_bytes())
    checksum = value.pop("reproducibility_checksum")
    if canonical_hash(value) != checksum:
        raise ValueError("artifact_checksum")
    if value["experiment_id"] != 8859 or value["task_id"] != m.TASK:
        raise ValueError("producer_identity")
    if (
        value["run_date"] != "20261106"
        or value["milestone"] != "2026.11.758"
        or value["acceptance_gates"] != GATES
        or value["protocol_hash"] != canonical_hash(m.protocol())
        or canonical_hash(value["protocol"]) != value["protocol_hash"]
    ):
        raise ValueError("frozen_contract_drift")
    for ref in (
        value["source_artifact_hashes"] + value["code_config_hashes"] + value["raw_shard_hashes"]
    ):
        if sha256_file(Path(ref["path"])) != ref["sha256"]:
            raise ValueError("operand_hash")
    if validate_current_work_receipt(value["current_work_receipt"]):
        raise ValueError("current_receipt")
    completed = sum(x["status"] == "completed" for x in value["rows"])
    ready = value["selective_rows_ready_score"]
    if (
        len(value["rows"]) != REPLICATES * 14
        or value["completed_count"] != completed
        or value["failed_count"] + value["censored_count"] != len(value["rows"]) - completed
        or ready not in [0, 1]
        or (
            ready
            and (
                not value["required_checks_passed"]
                or completed != len(value["rows"])
                or value["verdict_class"] not in {"null", "circular_positive"}
            )
        )
    ):
        raise ValueError("count_or_readiness_drift")
    if value["population_manifest"] is not None:
        raw = Path(value["population_manifest"]["path"]).parent
        if sha256_file(raw / "source.json") != PIN:
            raise ValueError("producer_pin")
        result = reduction(raw)
        if canonical_hash(load(reference(raw / "census.json"))) != canonical_hash(result):
            raise ValueError("retained_reduction_drift")
        for key, observed in result.items():
            if canonical_hash(value[key]) != canonical_hash(observed):
                raise ValueError("reduction_drift:" + key)
    elif value["split_protocol_benefit_score"] or ready:
        raise ValueError("missing_measurement")
    return dict(passed=True, status="replay_passed", sha256=sha256_file(path))


def main(argv: list[str] | None = None) -> int:
    """Execute a fixed research protocol without activating a milestone or loading a pretrained model."""
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
        split_protocol_benefit_score=0,
        primary_contrast=None,
        numerical_training_calls=0,
        **{
            key: []
            for key in [
                "selected_cutoff_rows",
                "certificate_event_rows",
                "served_risk_rows",
                "paired_replicate_rows",
                "census_risk_rows",
                "firing_intervals",
                "conditional_failure_intervals",
                "certificate_failure_intervals",
                "policy_summaries",
            ]
        },
        per_draw_results=None,
    )
    errors = []
    measured_ok = False
    m.progress("after_preconditions_before_measurement")
    if pre["external_ready"] and pre["resources"]["passed"]:
        try:
            result = measure(raw)
            measured_ok = True
        except (OSError, ValueError, KeyError, TypeError, RuntimeError, StopIteration) as error:
            errors.append(str(error))
    measured = time.monotonic_ns()
    m.progress("before_owned_validation")
    with TemporaryDirectory(prefix="exp8859-validation-") as directory:
        validation = (
            unqualified() if args.private_output else validate(raw / "validation", Path(directory))
        )
    valid = bool(
        validation["coverage"]["complete"]
        and validation["union"]["passed"]
        and all(x["passed"] for x in validation["receipts"])
    )
    m.progress("after_owned_validation")
    completed = sum(x["status"] == "completed" for x in result["rows"])
    verdict = (
        "disqualified"
        if errors or not pre["resources"]["passed"]
        else "blocked"
        if not pre["external_ready"]
        else "disqualified"
        if not valid or completed != REPLICATES * 14
        else "circular_positive"
        if result["split_protocol_benefit_score"]
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
    substrate = "no_model_load" if measured_ok else "blocked_no_run"
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
    value: Json = dict(
        result,
        experiment_id=8859,
        task_id=m.TASK,
        milestone="2026.11.758",
        run_date=args.date,
        honest_verdict=f"complete_{verdict}_{reason if verdict == 'blocked' else 'split_certificate_evaluation'}",
        verdict_class=verdict,
        inference_substrate=receipt["inference_substrate"],
        inference_substrate_class=substrate,
        execution_venue="host",
        MODEL_SPECS=[],
        model_invocation_counts=receipt["invocation_counts"],
        verifier_is_oracle=True,
        exposure_scope="Fresh constructed population; exact oracle labels; no natural-language correctness or learned-head superiority claim.",
        independent_generalization_score=0,
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
                "python/carnot/reporting/split_certificate_methods_8857.py",
                "python/carnot/reporting/split_certificate_runner_8857.py",
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
        independent_count=REPLICATES if measured_ok else 0,
        gate_check_summary=pre["diagnostics"],
        owned_errors=errors,
        acceptance_gates=GATES,
        required_checks_passed=valid,
        validation_receipts=validation["receipts"],
        owned_statement_coverage=validation["coverage"],
        selective_rows_ready_score=int(
            verdict in {"null", "circular_positive"} and completed == REPLICATES * 14
        ),
        protocol_hash=canonical_hash(m.protocol()),
        original_method_protocol_hash=m.METHOD_PROTOCOL_PIN,
        protocol=m.protocol(),
        population_manifest=reference(raw / "population.json") if measured_ok else None,
        draw_manifest=reference(raw / "draws.json") if measured_ok else None,
        methodology="Fit five fresh CPU policy seeds on fitting worlds only. Seal seven policies and population scores before calibration labels. Compare 720 simultaneous labels against 240 selection plus 480 independent certification labels. Refusals serve zero. Open independent 240-draw evaluation labels after certificates, then census. Bootstrap paired replicate means across all policies with 10000 draws at 99 percent.",
        methodology_note="Exact labels describe only a constructed finite population. Certificate validity, empirical utility and method superiority are separate. Zero failures or perfect fixture controls are not a universal guarantee.",
        flagged_adversarial=False,
        field_principles={},
    )
    value["field_principles"] = {
        k: "Frozen choice or measured evidence bound to immutable operands; custody alone is not scientific evidence."
        for k in value
    }
    value["field_principles"].update(
        per_draw_results="Immutable JSON reference exposes every repeated evaluation draw, decision, error and refusal for both protocols and all seven policies.",
        selective_rows_ready_score="Complete scheduled measurement and current validation, independent of demonstrated utility.",
        split_protocol_benefit_score="Lower paired coverage gain exceeds .05 and every claimed useful rule has population risk at most .05.",
        conditional_failure_intervals="Exact 99-percent intervals divide observed population false certifications only by issued certificates; empty denominators remain null.",
        independent_count="Sixty independent draw streams; seven paired policies and repeated indices do not increase the replicate count.",
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
                honest_verdict="complete_disqualified_split_certificate_evaluation",
                verdict_class="disqualified",
                selective_rows_ready_score=0,
                required_checks_passed=False,
            )
        del value["reproducibility_checksum"]
        value["reproducibility_checksum"] = canonical_hash(value)
        m.progress("after_terminal_checks")
    publish_primary(output, value, replay)
    m.progress("published_" + value["verdict_class"], completed, REPLICATES * 14 - completed)
    return 0
