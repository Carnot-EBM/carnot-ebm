"""REQ-REPORT-8868: retain immutable evidence and publish only checked methods."""

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

from carnot.reporting import conditional_risk_methods_8868 as m
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
UPSTREAM = {
    "experiment_8857_v758_split_certificate_methods": (
        "sha256:1f3adf1e780cdd2a7ecf194a5436504b146986820b071fd0f016aed922734dc6",
        "split_methods_ready_score",
    ),
    "experiment_8859_v758_split_certificate_evaluation": (
        "sha256:83ac2362e1a44898a3a341d3941cd448a22238cff1bb525ba72e3eae307d7063",
        "selective_rows_ready_score",
    ),
}
DESIGN = "openspec/change-proposals/research-roadmap-v758-preserved-20261106.md"
DESIGN_PIN = "sha256:b947c7b2712e9b75aef6cf127a03d1243e9ce761d548ea5b39192b4cca94273a"
HEALTH = Path("/tmp/carnot-8868-repository-health/receipt.json")
MEASUREMENT_DEADLINE_S = 1800


def preconditions(raw: Path, sources: Path = m.ROOT / "results") -> Json:
    """Authenticate original assertions and all their operands before any new labels."""
    raw.mkdir(parents=True, exist_ok=True, mode=0o700)
    raw.chmod(0o700)
    atomic_json(raw / "writable.json", dict(private_storage=True))
    refs: dict[str, Json] = {}
    diagnostics = []
    for name, (pin, field) in UPSTREAM.items():
        path = sources / (name + ".json")
        diagnostic = dict(
            upstream=name,
            artifact_field=field,
            expected=1,
            observed=None,
            path=str(path),
            sha256=None,
            check="missing_file",
        )
        try:
            diagnostic.update(check="rejected_producer", sha256=sha256_file(path))
            value = json.loads(path.read_bytes())
            diagnostic.update(check="missing_field", observed=value.get(field))
            score = value[field]
            diagnostic["check"] = "observed_zero"
            if score != 1:
                raise ValueError("readiness_zero")
            diagnostic.update(
                check="rejected_producer",
                artifact_field="sha256",
                expected=pin,
                observed=sha256_file(path),
            )
            if (
                sha256_file(path) != pin
                or value["experiment_id"] != int(name.split("_")[1])
                or value["verdict_class"] != "null"
            ):
                raise ValueError("producer_identity")
            for ref in [reference(path)] + [
                x
                for k in ["source_artifact_hashes", "code_config_hashes", "raw_shard_hashes"]
                for x in value[k]
            ]:
                diagnostic.update(path=ref["path"], expected=ref["sha256"], artifact_field="sha256")
                diagnostic["observed"] = sha256_file(Path(ref["path"]))
                if diagnostic["observed"] != ref["sha256"]:
                    raise ValueError("original_operand_drift")
                refs[ref["path"]] = ref
            shutil.copyfile(path, raw / (name + ".json"))
        except (OSError, ValueError, KeyError, TypeError) as error:
            diagnostics.append(
                dict(
                    diagnostic,
                    check="missing_file"
                    if isinstance(error, OSError)
                    else "missing_field"
                    if isinstance(error, KeyError)
                    else diagnostic["check"],
                    error=str(error),
                )
            )
    design = m.ROOT / DESIGN
    if not design.is_file() or sha256_file(design) != DESIGN_PIN:
        diagnostics.append(
            dict(
                check="missing_file" if not design.is_file() else "rejected_producer",
                upstream="preserved_v758_design",
                artifact_field="sha256",
                expected=DESIGN_PIN,
                observed=sha256_file(design) if design.is_file() else None,
                path=str(design),
                sha256=sha256_file(design) if design.is_file() else None,
            )
        )
    else:
        refs[str(design)] = reference(design)
    resources: Json = dict(
        private_output_writable=True,
        private_mode=oct(raw.stat().st_mode & 0o777),
        free_disk_bytes=shutil.disk_usage(raw).free,
        dependencies=dict(numpy=np.__version__, scipy=scipy.__version__),
        tools_available=all(
            (m.ROOT / ".venv/bin" / n).is_file() for n in ["python", "pytest", "ruff", "mypy"]
        ),
        measurement_deadline_s=MEASUREMENT_DEADLINE_S,
        total_budget_s=4200,
    )
    resources["passed"] = (
        resources["tools_available"] and resources["free_disk_bytes"] >= 512 * 1024**2
    )
    return dict(
        references=list(refs.values()),
        diagnostics=diagnostics,
        external_ready=not diagnostics,
        resources=resources,
    )


def exposed(pre: Json) -> set[str]:
    """Exclude actual relevant historical structures, including earlier fitting worlds."""
    result: set[str] = set()
    for ref in pre["references"]:
        if Path(ref["path"]).name == "population.json":
            value = load(ref)
            value = value.get("value", value)
            result.update(value.get("exposed_invariants", []))
            for u in value.get("inputs", []):
                result.add(m.relevant(u["input"]))
    return result


def measure(raw: Path, pre: Json) -> Json:
    """Seal each phase before its next label release; refuse deadlines without hiding slots."""
    started = time.monotonic()
    events = []

    def seal(name: str, value: Json) -> None:
        atomic_json(raw / (name + ".json"), value)
        events.append(
            dict(stage=name, monotonic_ns=time.monotonic_ns(), **reference(raw / (name + ".json")))
        )
        atomic_json(raw / "access_order.json", dict(events=events))
        m.progress("sealed_" + name, len(events), len(m.protocol()["access_order"]) - len(events))
        if time.monotonic() - started > MEASUREMENT_DEADLINE_S:
            raise ValueError("measurement_deadline")

    seal("protocol", m.protocol())
    pilot_start = time.monotonic()
    dev = m.freeze(set(), count=2, fit_count=1)
    controls = m.controls()
    dev_structures = {s for w in dev["worlds"] for s in w["relevant_structures"]}
    pilot: Json = dict(
        controls=controls,
        fixture=dev,
        elapsed_s=time.monotonic() - pilot_start,
        projected_total_s=(time.monotonic() - pilot_start) * 512 + 1200,
        zero_error_minimum=m.FLOOR,
    )
    seal("pilot", pilot)
    if not all(c["passed"] for c in controls) or pilot["projected_total_s"] > 4200:
        raise ValueError("pilot_control_or_runtime")
    cohort = m.freeze(exposed(pre) | dev_structures)
    seal("population", cohort)
    trace = m.draws(1024)
    seal("draws", trace)
    m.progress("before_training_no_model_load", 0, 5)
    fitted = m.old.old.fit(cohort)
    seal("fit", fitted)
    m.old.old.audit_fit(cohort, fitted)
    m.progress("after_training_before_prediction", 5, 0)
    scores = m.old.old.predict(cohort, fitted)
    seal("scores", dict(rows=scores))
    units = {
        g: sorted(
            [u for u in cohort["inputs"] if u["split"] == "population" and u["group"] == g],
            key=lambda u: u["unit_id"],
        )
        for g in m.GROUPS
    }

    def release(stage: str) -> Json:
        requested = {
            units[g][i]["unit_id"]: units[g][i]
            for rep in trace["replicates"]
            for g in m.GROUPS
            for i in rep["groups"][g][stage]
        }
        truth = m.old.old.p.labels(list(requested.values()))
        seal(stage + "_labels", dict(labels=truth))
        return truth

    selection = m.select(cohort, scores, release("selection"), trace)
    seal("selection", selection)
    certificates = m.certify(cohort, scores, release("certification"), trace, selection)
    seal("certificates", certificates)
    truth = m.old.old.p.labels([u for g in units.values() for u in g])
    seal("census_labels", dict(labels=truth))
    rows = m.census(cohort, scores, truth, certificates)
    seal("census", dict(rows=rows))
    m.progress("before_corruption_benchmark", 0, 4096)
    corrupted = m.corrupt(cohort)
    corrupt_scores = m.old.old.predict(corrupted, fitted)
    corrupt_truth = m.old.old.p.labels(
        [u for u in corrupted["inputs"] if u["split"] == "population"]
    )
    stress = m.census(corrupted, corrupt_scores, corrupt_truth, certificates, corrupted=True)
    seal("corruption", dict(scores=corrupt_scores, labels=corrupt_truth, rows=stress))
    m.progress("after_corruption_benchmark", 4096, 0)
    return dict(
        rows=rows + stress,
        **m.summarize(rows + stress),
        numerical_training_calls=fitted["numerical_training_calls"],
        positive_negative_controls=controls,
    )


def reduction(raw: Path, pre: Json) -> Json:
    """Replay from original data and scores rather than trusting rehashed aggregate assertions."""

    def get(name: str) -> Json:
        return load(reference(raw / (name + ".json")))

    events = get("access_order")["events"]
    if [e["stage"] for e in events] != m.protocol()["access_order"] or any(
        a["monotonic_ns"] > b["monotonic_ns"] for a, b in zip(events, events[1:])
    ):
        raise ValueError("access_order_drift")
    for event in events:
        load(event)
    if get("protocol") != json.loads(json.dumps(m.protocol())):
        raise ValueError("protocol_drift")
    pilot = get("pilot")
    dev = m.freeze(set(), count=2, fit_count=1)
    if (
        pilot["controls"] != m.controls()
        or pilot["fixture"] != dev
        or pilot["projected_total_s"] > 4200
    ):
        raise ValueError("pilot_drift")
    reserved = exposed(pre) | {s for w in dev["worlds"] for s in w["relevant_structures"]}
    cohort = get("population")
    if cohort != m.freeze(reserved) or get("draws") != m.draws(1024):
        raise ValueError("population_or_draw_drift")
    trace, fitted = get("draws"), get("fit")
    m.old.old.audit_fit(cohort, fitted)
    scores = m.old.old.predict(cohort, fitted)
    if scores != get("scores")["rows"]:
        raise ValueError("prediction_drift")
    truth = m.old.old.p.labels([u for u in cohort["inputs"] if u["split"] == "population"])
    for stage in ["selection", "certification", "census"]:
        if any(truth[u] != label for u, label in get(stage + "_labels")["labels"].items()):
            raise ValueError("label_drift")
    selection = m.select(cohort, scores, get("selection_labels")["labels"], trace)
    certificates = m.certify(
        cohort, scores, get("certification_labels")["labels"], trace, selection
    )
    rows = m.census(cohort, scores, truth, certificates)
    corrupted = m.corrupt(cohort)
    stress_scores = m.old.old.predict(corrupted, fitted)
    stress_truth = m.old.old.p.labels(
        [u for u in corrupted["inputs"] if u["split"] == "population"]
    )
    stress = m.census(corrupted, stress_scores, stress_truth, certificates, corrupted=True)
    for name, result in [
        ("selection", selection),
        ("certificates", certificates),
        ("census", dict(rows=rows)),
        ("corruption", dict(scores=stress_scores, labels=stress_truth, rows=stress)),
    ]:
        if canonical_hash(get(name)) != canonical_hash(result):
            raise ValueError("stage_reduction_drift:" + name)
    return dict(
        rows=rows + stress,
        **m.summarize(rows + stress),
        numerical_training_calls=fitted["numerical_training_calls"],
        positive_negative_controls=m.controls(),
    )


def validate(raw: Path, private: Path) -> Json:
    """Reuse unchanged exact-node consumers and unit/child statement coverage without exclusions."""
    checker = namespace(inherited, "conditional8868_validation")
    checker.m = m
    return dict(checker.validate(raw, private))


def replay(path: Path) -> Json:
    """Check identity, bytes, receipts and independently reconstructed scientific reductions."""
    value = json.loads(path.read_bytes())
    checksum = value.pop("reproducibility_checksum")
    if canonical_hash(value) != checksum:
        raise ValueError("artifact_checksum")
    if value["experiment_id"] != 8868 or value["task_id"] != m.TASK:
        raise ValueError("producer_identity")
    for ref in (
        value["source_artifact_hashes"] + value["code_config_hashes"] + value["raw_shard_hashes"]
    ):
        if sha256_file(Path(ref["path"])) != ref["sha256"]:
            raise ValueError("operand_hash")
    if validate_current_work_receipt(value["current_work_receipt"]):
        raise ValueError("current_receipt")
    protocol = m.protocol()
    for field in [
        "alpha_allocation",
        "exact_sample_requirements",
        "risk_bound_construction",
        "mixture_weights",
        "expected_row_counts",
    ]:
        if canonical_hash(value[field]) != canonical_hash(protocol[field]):
            raise ValueError("frozen_contract_drift")
    if (
        value["group_definition_hash"] != canonical_hash(m.GROUPS)
        or value["run_date"] != "20261106"
        or value["milestone"] != "2026.11.759"
    ):
        raise ValueError("group_or_identity_drift")
    rows = value["rows"]
    completed = sum(r["status"] == "completed" for r in rows)
    if (
        len(rows) != 3360
        or value["completed_count"] != completed
        or value["failed_count"] + value["censored_count"] != 3360 - completed
    ):
        raise ValueError("count_drift")
    if completed:
        result = reduction(
            Path(value["population_manifest"]["path"]).parent, value["preconditions_checked"]
        )
        if any(canonical_hash(value[k]) != canonical_hash(v) for k, v in result.items()):
            raise ValueError("reduction_drift")
    ready = value["conditional_methods_ready_score"]
    if ready not in [0, 1] or (
        ready
        and (
            completed != 3360
            or not value["required_checks_passed"]
            or value["verdict_class"] != "null"
        )
    ):
        raise ValueError("readiness_drift")
    return dict(passed=True, status="replay_passed", sha256=sha256_file(path))


def scheduled(reason: str) -> Json:
    """Every scheduled replicate, arm and stress remains visible when execution cannot proceed."""
    rows = [
        dict(
            unit_id=f"{rep}/{protocol}/{arm}/{seed}/{scenario}",
            cluster_id=f"rep/{rep}",
            replicate=rep,
            protocol=protocol,
            arm=arm,
            seed=seed,
            scenario=scenario,
            metric=None,
            denominator=4096,
            status="censored",
            missing_reason=reason,
        )
        for rep in range(40)
        for protocol in ["pooled", "grouped"]
        for arm, seed in m.POLICIES
        for scenario in [*m.MIXTURES, "evidence_corruption"]
    ]
    return dict(
        rows=rows,
        paired_coverage_difference=None,
        paired_replicate_differences=[],
        independent_replicates=0,
        certificate_failure_count=0,
        scope="unexecuted scheduled units",
        numerical_training_calls=0,
        positive_negative_controls=[],
    )


def main(argv: list[str] | None = None) -> int:
    """Execute the frozen study; a terminal null releases methods without claiming benefit."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", choices=["20261106"], default="20261106")
    parser.add_argument("--output", type=Path, default=m.ROOT / "results" / (m.NAME + ".json"))
    parser.add_argument("--private-output", type=Path)
    parser.add_argument("--sources", type=Path, default=m.ROOT / "results")
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
    pre = preconditions(raw, args.sources)
    atomic_json(
        raw / "invocation.json",
        dict(
            executable=sys.executable,
            arguments=argv if argv is not None else sys.argv[1:],
            cwd=os.getcwd(),
            pid=os.getpid(),
            started_monotonic_ns=started,
            executed_sources=[reference(m.ROOT / p) for p in m.OWNED],
        ),
    )
    atomic_json(raw / "preconditions.json", pre)
    m.progress("after_preconditions_before_measurement")
    reason = pre["diagnostics"][0]["check"] if pre["diagnostics"] else "owned_failure"
    result = scheduled(reason)
    errors = []
    if pre["external_ready"] and pre["resources"]["passed"]:
        try:
            result = measure(raw, pre)
        except (OSError, ValueError, KeyError, TypeError, RuntimeError) as error:
            errors.append(str(error))
            result = scheduled(str(error))
            if "deadline" not in str(error):
                result["rows"] = [dict(row, status="failed") for row in result["rows"]]
            if (raw / "fit.json").is_file():
                result["numerical_training_calls"] = load(reference(raw / "fit.json"))[
                    "numerical_training_calls"
                ]
    measured = time.monotonic_ns()
    m.progress("before_owned_validation")
    with TemporaryDirectory(prefix="exp8868-validation-") as directory:
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
        else "null"
        if valid
        else "disqualified"
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
        inference_substrate="cpu_small_energy_heads_and_exact_conditional_certificates",
        inference_substrate_class=substrate,
        execution_venue="host",
        inference_substrate_details=dict(
            scope="fresh in-process numerical heads; no pretrained load"
        ),
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
    manifests = {
        k: reference(raw / (name + ".json")) if completed else None
        for k, name in [
            ("population_manifest", "population"),
            ("fit_manifest", "fit"),
            ("draw_manifest", "draws"),
        ]
    }
    value = dict(
        result,
        experiment_id=8868,
        task_id=m.TASK,
        milestone="2026.11.759",
        run_date=args.date,
        honest_verdict=f"complete_{verdict}_conditional_risk_methods",
        verdict_class=verdict,
        inference_substrate=receipt["inference_substrate"],
        inference_substrate_class=substrate,
        execution_venue="host",
        MODEL_SPECS=[],
        model_invocation_counts=receipt["invocation_counts"],
        verifier_is_oracle=True,
        exposure_scope="constructed signed relational worlds; exact labels; no natural-language or oracle-distinct generalization",
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
                DESIGN,
                "scripts/experiment_template.py",
                "python/carnot/reporting/current_work_receipt.py",
                "python/carnot/reporting/primary_publication.py",
            ]
        ],
        raw_shard_hashes=[reference(p) for p in raw.glob("*.json")]
        + validation["raw_references"]
        + health_refs,
        repository_health=health_refs[0] if health_refs else None,
        intended_count=3360,
        completed_count=completed,
        failed_count=sum(x["status"] == "failed" for x in result["rows"]),
        censored_count=sum(x["status"] == "censored" for x in result["rows"]),
        excluded_count=0,
        independent_count=40 if completed else 0,
        gate_check_summary=pre["diagnostics"],
        acceptance_gates=dict(
            conditional_risk_upper=0.05,
            valid_controls=True,
            complete_separation=True,
            bounded_runtime_s=4200,
            benefit_required=False,
        ),
        required_checks_passed=valid,
        validation_receipts=validation["receipts"],
        owned_statement_coverage=validation["coverage"],
        owned_errors=errors,
        conditional_methods_ready_score=int(verdict == "null" and completed == 3360),
        group_definition_hash=canonical_hash(m.GROUPS),
        **manifests,
        **{
            k: protocol[k]
            for k in [
                "alpha_allocation",
                "exact_sample_requirements",
                "risk_bound_construction",
                "mixture_weights",
                "expected_row_counts",
            ]
        },
        methodology="Fixed query-only groups; development-only existing energy objective; paired independent stratified replacement draws; selection then one exact certification; pooled acceptance-mass weighted bounds or conditional bounds for each served group; complete census after certificates; corruption outside guarantee.",
        methodology_note="Familywise .01 applies per replicate. Readiness is valid methods, not benefit. Oracle labels cannot establish independent generalization.",
        flagged_adversarial=False,
        field_principles={},
    )
    value["field_principles"] = {
        k: "Frozen protocol or measured operand, bound to original bytes; administrative custody is not scientific evidence."
        for k in value
    }
    value["field_principles"].update(
        conditional_methods_ready_score="Valid mathematics, separation, controls and bounded checked execution, regardless of benefit.",
        independent_count="Forty independent calibration replicates; seven policies, repeated draws and stress mixtures share observations.",
        risk_bound_construction="Within-group exact bounds combine using known input-only acceptance mass; stratified draws are never one binomial sample.",
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
                honest_verdict="complete_disqualified_conditional_risk_methods",
                verdict_class="disqualified",
                conditional_methods_ready_score=0,
                required_checks_passed=False,
            )
        del value["reproducibility_checksum"]
        value["reproducibility_checksum"] = canonical_hash(value)
        m.progress("after_terminal_checks")
    publish_primary(output, value, replay)
    m.progress("published_" + value["verdict_class"], completed, 3360 - completed)
    return 0
