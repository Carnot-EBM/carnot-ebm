"""REQ-REPORT-8911: publish only a checked, independently reproducible terminal audit."""

import argparse
import json
import os
from pathlib import Path
import shutil
import sys
from tempfile import TemporaryDirectory
import time
from typing import Any

from carnot.reporting import guarded_benefit_audit_8911 as m
from carnot.reporting import guarded_update_methods_8905 as methods
from carnot.reporting import topology_causal_runner_8884 as previous
from carnot.reporting.current_work_receipt import (
    atomic_json,
    canonical_hash,
    sha256_file,
    build_current_work_receipt,
    validate_current_work_receipt,
)
from carnot.reporting.primary_publication import publish_primary
from carnot.reporting.selective_runner_8816 import unqualified
from carnot.reporting.v746_retained_evidence import reference, load
from carnot.reporting.v748_collection_contract import namespace
from scripts.experiment_template import ExperimentTemplate, normalize_artifact_for_template_write

Json = dict[str, Any]


def validate(raw: Path, private: Path) -> Json:
    """Reuse measured unit/child coverage and unchanged private E2E-018/021 checks."""
    checker = namespace(previous, "guarded8911_validation")
    checker.m = m
    return dict(checker.validate(raw, private))


def terminal(candidate: Path, raw: Path) -> list[Json]:
    """Inspect prospective bytes through unchanged scanners and outside-checkout replay."""
    checker = namespace(previous, "guarded8911_terminal")
    checker.m = m
    return list(checker.terminal(candidate, raw))


def qualification(value: Json) -> bool:
    """Readiness belongs to complete owned validation, regardless of scientific benefit."""
    c = value["owned_statement_coverage"]
    return bool(
        c["complete"]
        and c["percent"] == 100
        and {r["path"] for r in c["files"]} == set(m.OWNED)
        and not any(r["missing"] or r["excluded"] for r in c["files"])
        and value["test_union"]["passed"]
        and all(r["passed"] for r in value["validation_receipts"])
    )


def semantics(value: Json) -> tuple[bool, str]:
    """External absence blocks; failed owned controls disqualify; a complete null is ready."""
    ready = bool(
        value["preconditions_checked"]["external_ready"]
        and value["preconditions_checked"]["resources"]["passed"]
        and qualification(value)
        and not value["owned_errors"]
        and value["completed_count"] == 18432
    )
    verdict = (
        "blocked"
        if not value["preconditions_checked"]["external_ready"]
        else "circular_positive"
        if ready and value["guarded_benefit_score"]
        else "null"
        if ready
        else "disqualified"
    )
    return ready, verdict


def replay(path: Path) -> Json:
    """Rebuild every headline from pinned source bytes so rehashing a false claim fails."""
    value = json.loads(path.read_bytes())
    checksum = value.pop("reproducibility_checksum")
    m.require(canonical_hash(value) == checksum, "artifact_checksum")
    m.require(
        (value["experiment_id"], value["task_id"], value["milestone"], value["run_date"])
        == (8911, m.TASK, "2026.11.762", "20261108"),
        "producer_identity",
    )
    refs = value["source_artifact_hashes"] + value["code_config_hashes"] + value["raw_shard_hashes"]
    for i, ref in enumerate(refs):
        m.require(sha256_file(Path(ref["path"])) == ref["sha256"], "operand_hash")
        if i % 100 == 0:
            m.progress("replay_authenticate", i, len(refs) - i)
    m.require(value["acceptance_gates"] == m.GATES, "acceptance_drift")
    receipt = value["current_work_receipt"]
    m.require(
        not validate_current_work_receipt(receipt)
        and value["numerical_training_calls"] == 0
        and value["duration_s"] == receipt["duration_s"]
        and value["phase_spans"] == receipt["phase_spans"]
        and value["model_invocation_counts"] == receipt["invocation_counts"]
        and value["inference_substrate_class"] == receipt["inference_substrate_class"],
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
        expected["guarded_benefit_score"] = 0
    for key, observed in expected.items():
        m.require(value[key] == observed, "reduction_drift:" + key)
    m.require(
        value["required_checks_passed"] == ready
        and type(value["audit_ready_score"]) is int
        and value["audit_ready_score"] == int(ready)
        and type(value["guarded_benefit_score"]) is int
        and value["verdict_class"] == verdict
        and value["honest_verdict"] == "complete_" + verdict + "_guarded_benefit_audit",
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


def seal(value: Json, started: int, spans: list[Json]) -> None:
    """Keep actual elapsed work and publication identity consistent after terminal checks."""
    receipt = build_current_work_receipt(
        run_id=str(started),
        owner_pid=os.getpid(),
        events=[],
        inference_substrate="read_only_sealed_guarded_benefit_audit",
        inference_substrate_class="no_model_load" if value["completed_count"] else "blocked_no_run",
        execution_venue="host",
        inference_substrate_details=dict(
            scope="Recorded predictions and truth only; no training, generation or oracle calls"
        ),
        started_monotonic_ns=started,
        ended_monotonic_ns=started + round(spans[-1]["end_s"] * 1e9),
        phase_spans=spans,
    )
    value.update(
        current_work_receipt=receipt,
        duration_s=receipt["duration_s"],
        phase_spans=spans,
        inference_substrate=receipt["inference_substrate"],
        inference_substrate_class=receipt["inference_substrate_class"],
        model_invocation_counts=receipt["invocation_counts"],
    )
    value["field_principles"] = {
        k: "Actual producer or preserved original operand; finite constructed-truth scope."
        for k in value
        if k != "reproducibility_checksum"
    }
    value["field_principles"].update(
        audit_ready_score="Complete independent audit and owned validation; null outcomes can qualify.",
        guarded_benefit_score="All nine simultaneous upper bounds and every validity control must pass.",
        paired_intervals="One-sided99-percent family; Bonferroni0.01/9,10000 paired stream bootstrap draws.",
        paired_stream_rows="Within-stream arm means precede all resampling; future window25–48.",
        promotion_summary="Deployment admission frequencies only; operational guards are not independent evaluation.",
        retention_summary="Untouched retention at0/24/48; never optimizer or admission data.",
        calibration_summary="Descriptive Brier, error, mean confidence and confidence-minus-accuracy gap; no added success test.",
        independent_count="24 paired RNG streams across recurring eight online/eight retention topology families; no unseen-topology claim.",
        continuation_decision="Retire the unchanged mechanism on the same failed scientific conjunction.",
        repository_health="Separate bounded full-suite diagnostic; it cannot grant owned qualification.",
        reproducibility_checksum="Canonical artifact hash plus independent reconstruction from original pinned producer bytes.",
    )
    value.pop("reproducibility_checksum", None)
    value["reproducibility_checksum"] = canonical_hash(value)


def main(argv: list[str] | None = None) -> int:
    """Complete the audit under a fixed budget without equating operational readiness with benefit."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", choices=["20261108"], default="20261108")
    parser.add_argument("--output", type=Path, default=m.ROOT / "results" / (m.NAME + ".json"))
    parser.add_argument("--private-output", type=Path)
    parser.add_argument("--sources", type=Path, default=m.ROOT / "results")
    parser.add_argument("--replay", "--cold-replay", type=Path)
    args = parser.parse_args(argv)
    if args.replay:
        try:
            print(json.dumps(replay(args.replay)), flush=True)
            return 0
        except (OSError, ValueError, KeyError, TypeError, StopIteration, IndexError) as error:
            print("replay_rejected: " + str(error), flush=True)
            return 1
    output = (args.private_output or args.output).absolute()
    if args.private_output and output.resolve().is_relative_to(m.ROOT):
        parser.error("private outputs must remain outside checkout")
    started = time.monotonic_ns()
    raw = output.parent / "raw" / m.NAME / "invocations" / str(started)
    raw.mkdir(parents=True, mode=0o700)
    template = ExperimentTemplate(
        8911, "Guarded benefit audit", "results/" + m.NAME + ".json", repo_root=m.ROOT, seed=m.SEED
    )
    invocation = dict(
        executable=sys.executable,
        arguments=argv if argv is not None else sys.argv[1:],
        cwd=os.getcwd(),
        pid=os.getpid(),
        executed_sources=[reference(m.ROOT / p) for p in m.OWNED],
    )
    atomic_json(raw / "invocation.json", invocation)
    pre = m.preflight(raw, args.sources)
    atomic_json(raw / "preconditions.json", pre)
    reason = (
        "external_input_unavailable" if not pre["external_ready"] else "owned_resources_unavailable"
    )
    result, errors = m.empty(reason), []
    if pre["external_ready"] and pre["resources"]["passed"]:
        try:
            m.progress("reduction_before", 0, 24)
            with template.phase("independent_reduction"):
                result = m.reduce(load(reference(raw / (m.SOURCE + ".json"))))
            reason = "complete"
            m.progress("reduction_after", 24, 0)
        except (OSError, ValueError, KeyError, RuntimeError, StopIteration, IndexError) as error:
            reason = str(error)
            errors.append(reason)
            result = m.empty(reason)
    measured = time.monotonic_ns()
    m.progress("validation_before", 0, 1)
    with TemporaryDirectory(prefix="carnot8911-validation-", dir="/tmp") as directory:
        try:
            validation = (
                unqualified()
                if args.private_output
                else validate(raw / "validation", Path(directory))
            )
        except (OSError, ValueError, KeyError, RuntimeError, StopIteration, IndexError) as error:
            validation = unqualified()
            errors.append(str(error))
    m.progress("validation_after", 1, 0)
    ended = time.monotonic_ns()
    spans = [
        dict(phase="preconditions_and_reduction", start_s=0.0, end_s=(measured - started) / 1e9),
        dict(phase="validation", start_s=(measured - started) / 1e9, end_s=(ended - started) / 1e9),
    ]
    for name in ("tests-first", "preflight-red", "repository-health"):
        diagnostic = Path("/tmp/carnot8911-" + name + ".log")
        if diagnostic.is_file():
            shutil.copyfile(diagnostic, raw / (name + ".log"))
    helpers = (
        m.causal.OWNED
        + methods.OWNED
        + [
            "python/carnot/reporting/label_benefit_audit_8886.py",
            "python/carnot/reporting/causal_learning_audit_8810.py",
            "python/carnot/reporting/topology_causal_runner_8884.py",
            "python/carnot/reporting/topology_memory_runner_8881.py",
            "python/carnot/reporting/evidence_evaluation_runner_8779.py",
            "python/carnot/reporting/selective_runner_8816.py",
            "scripts/experiment_template.py",
            "python/carnot/reporting/current_work_receipt.py",
            "python/carnot/reporting/primary_publication.py",
            "scripts/adversarial_verify.py",
            "scripts/verdict_row_consistency_lint.py",
        ]
    )
    value = dict(
        result,
        experiment_id=8911,
        task_id=m.TASK,
        milestone="2026.11.762",
        run_date=args.date,
        execution_venue="host",
        MODEL_SPECS=[],
        numerical_training_calls=0,
        verifier_is_oracle=True,
        exposure_scope="Sealed finite constructed truth; no unrecorded oracle state, pilot evaluation or unseen-topology claim.",
        independent_generalization_score=0,
        preconditions_checked=pre,
        random_seed=m.SEED,
        producer_invocation=invocation,
        source_artifact_hashes=pre["references"],
        code_config_hashes=[reference(m.ROOT / p) for p in m.OWNED + m.TESTS + helpers],
        raw_shard_hashes=[reference(p) for p in sorted(raw.rglob("*")) if p.is_file()]
        + validation["raw_references"],
        raw_directory=str(raw),
        gate_check_summary=pre["diagnostics"],
        acceptance_gates=m.GATES,
        validation_receipts=validation["receipts"],
        owned_statement_coverage=validation["coverage"],
        test_union=validation["union"],
        owned_errors=errors,
        censor_reason=reason,
        cited_upstream_artifacts=[
            dict(
                experiment_id=8909,
                task_id=m.causal.TASK,
                sha256=m.PIN,
                artifact_field="causal_rows_ready_score",
                expected=1,
            ),
            dict(
                experiment_id=8905,
                task_id=methods.TASK,
                sha256=m.causal.PIN,
                artifact_field="guarded_methods_ready_score_and_candidate_ready_score",
                expected=1,
            ),
        ],
        methodology="Read-only independent sealed prediction/label reduction. Average future25–48 or48 retention queries per arm inside24 paired streams;10000 fixed-seed paired bootstrap draws; nine one-sided Bonferroni0.01/9 upper bounds. No head refit, new labels or independent natural-language generalization.",
        repository_health=dict(
            command=".venv/bin/pytest tests/python -q",
            scope="Separate repository health; one bounded240-second invocation",
            passed=False,
            exit_code=124,
            timeout_s=240,
            log=str(raw / "repository-health.log"),
        ),
        flagged_adversarial=False,
        field_principles={},
    )
    ready, verdict = semantics(value)
    value.update(
        required_checks_passed=ready,
        audit_ready_score=int(ready),
        guarded_benefit_score=value["guarded_benefit_score"] if ready else 0,
        verdict_class=verdict,
        honest_verdict="complete_" + verdict + "_guarded_benefit_audit",
    )
    seal(value, started, spans)
    value = normalize_artifact_for_template_write(value)
    seal(value, started, spans)
    candidate = raw / "prospective.json"
    atomic_json(candidate, value)
    if not args.private_output:
        m.progress("terminal_before", 0, 3)
        reports = terminal(candidate, raw)
        value["validation_receipts"] += reports
        for report in reports:
            for key in ("stdout_path", "stderr_path"):
                if key in report:
                    value["raw_shard_hashes"].append(reference(Path(report[key])))
        if not all(r["passed"] for r in reports):
            value.update(
                honest_verdict="complete_disqualified_guarded_benefit_audit",
                verdict_class="disqualified",
                audit_ready_score=0,
                guarded_benefit_score=0,
                required_checks_passed=False,
            )
        terminal_end = time.monotonic_ns()
        spans.append(
            dict(
                phase="terminal_validation",
                start_s=(ended - started) / 1e9,
                end_s=(terminal_end - started) / 1e9,
            )
        )
        seal(value, started, spans)
        m.progress("terminal_after", 3, 0)
    publish_primary(output, value, replay)
    for path in raw.rglob("*"):
        if path.is_file():
            path.chmod(0o400)
    m.progress(
        "published_" + value["verdict_class"], result["completed_count"], result["censored_count"]
    )
    return 0
