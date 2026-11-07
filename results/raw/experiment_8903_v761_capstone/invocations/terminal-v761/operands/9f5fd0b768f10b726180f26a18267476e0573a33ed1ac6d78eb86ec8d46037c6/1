"""REQ-REPORT-8898: publish a checked independent reduction with immutable custody."""

import argparse
import json
import os
from pathlib import Path
import shutil
import sys
from tempfile import TemporaryDirectory
import time
from typing import Any

from carnot.reporting import signed_benefit_audit_8898 as m
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
    """Reuse exact-node unit/child coverage and original private E2E checks unchanged."""
    checker = namespace(previous, "signed8898_validation")
    checker.m = m
    return dict(checker.validate(raw, private))


def terminal(candidate: Path, raw: Path) -> list[Json]:
    """The original scanners and outside-checkout replay inspect prospective bytes."""
    checker = namespace(previous, "signed8898_terminal")
    checker.m = m
    return list(checker.terminal(candidate, raw))


def replay(path: Path) -> Json:
    """Rehashed headline changes cannot replace reconstruction from original pinned bytes."""
    value = json.loads(path.read_bytes())
    checksum = value.pop("reproducibility_checksum")
    m.require(canonical_hash(value) == checksum, "artifact_checksum")
    m.require(
        (value["experiment_id"], value["task_id"], value["milestone"], value["run_date"])
        == (8898, m.TASK, "2026.11.761", "20261107"),
        "producer_identity",
    )
    refs = value["source_artifact_hashes"] + value["code_config_hashes"] + value["raw_shard_hashes"]
    for i, ref in enumerate(refs):
        m.require(sha256_file(Path(ref["path"])) == ref["sha256"], "operand_hash")
        if i % 1000 == 0:
            m.progress("replay_authenticate", i, len(refs) - i)
    m.require(value["acceptance_gates"] == m.GATES, "acceptance_drift")
    m.require(not validate_current_work_receipt(value["current_work_receipt"]), "current_receipt")
    receipt = value["current_work_receipt"]
    m.require(
        value["numerical_training_calls"] == 0
        and value["duration_s"] == receipt["duration_s"]
        and value["phase_spans"] == receipt["phase_spans"]
        and value["model_invocation_counts"] == receipt["invocation_counts"]
        and value["inference_substrate_class"] == receipt["inference_substrate_class"],
        "compute_scope",
    )
    if (
        value["preconditions_checked"]["external_ready"]
        and value["preconditions_checked"]["resources"]["passed"]
        and not value["owned_errors"]
    ):
        source = Path(value["raw_directory"]) / (m.SOURCE + ".json")
        m.require(sha256_file(source) == m.PIN, "original_source_pin")
        expected = m.reduce(load(reference(source)))
    else:
        expected = m.empty(value["censor_reason"])
    for key, observed in expected.items():
        m.require(value[key] == observed, "reduction_drift:" + key)
    valid = bool(
        value["required_checks_passed"]
        and value["completed_count"] == 36864
        and not value["owned_errors"]
    )
    verdict = (
        "blocked"
        if not value["preconditions_checked"]["external_ready"]
        else "circular_positive"
        if valid and expected["label_benefit_score"]
        else "null"
        if valid
        else "disqualified"
    )
    m.require(
        value["verdict_class"] == verdict
        and value["honest_verdict"] == "complete_" + verdict + "_signed_benefit_audit"
        and value["audit_ready_score"] == int(valid),
        "terminal_semantics",
    )
    m.require(
        value["verifier_is_oracle"] is True
        and value["independent_generalization_score"] == 0
        and value["MODEL_SPECS"] == [],
        "oracle_boundary",
    )
    if valid:
        coverage = value["owned_statement_coverage"]
        m.require(
            coverage["complete"]
            and coverage["percent"] == 100
            and {r["path"] for r in coverage["files"]} == set(m.OWNED)
            and all(not r["missing"] and not r["excluded"] for r in coverage["files"])
            and all(r["passed"] for r in value["validation_receipts"]),
            "validation_readiness",
        )
    return dict(passed=True, status="replay_passed", sha256=sha256_file(path))


def main(argv: list[str] | None = None) -> int:
    """A terminal null completes research; administrative readiness cannot establish learning."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", choices=["20261107"], default="20261107")
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
        8898,
        "Label benefit audit",
        "results/" + m.NAME + ".json",
        repo_root=m.ROOT,
        seed=m.base.SEED,
    )
    invocation = dict(
        executable=sys.executable,
        arguments=argv if argv is not None else sys.argv[1:],
        cwd=os.getcwd(),
        pid=os.getpid(),
        executed_sources=[reference(m.ROOT / p) for p in m.OWNED],
    )
    atomic_json(raw / "invocation.json", invocation)
    m.progress("preconditions_before", 0, 1)
    pre = m.preflight(raw, args.sources)
    atomic_json(raw / "preconditions.json", pre)
    reason = (
        "external_input_unavailable" if not pre["external_ready"] else "owned_resources_unavailable"
    )
    result = m.empty(reason)
    errors = []
    if pre["external_ready"] and pre["resources"]["passed"]:
        try:
            with template.phase("independent_reduction"):
                result = m.reduce(load(reference(raw / (m.SOURCE + ".json"))))
            reason = "complete"
        except (OSError, ValueError, KeyError, RuntimeError, StopIteration, IndexError) as error:
            reason = str(error)
            errors.append(reason)
            result = m.empty(reason)
    measured = time.monotonic_ns()
    m.progress("validation_before", 0, 1)
    with TemporaryDirectory(prefix="carnot8898-validation-", dir="/tmp") as directory:
        try:
            validation = (
                unqualified()
                if args.private_output
                else validate(raw / "validation", Path(directory))
            )
        except (OSError, ValueError, KeyError, RuntimeError, StopIteration, IndexError) as error:
            validation = unqualified()
            validation["receipts"][0].update(name="owned_validation_failure", error=str(error))
    m.progress("validation_after", 1, 0)
    valid = bool(
        validation["coverage"]["complete"]
        and validation["union"]["passed"]
        and all(r["passed"] for r in validation["receipts"])
        and result["completed_count"] == 36864
        and not errors
    )
    verdict = (
        "blocked"
        if not pre["external_ready"]
        else "circular_positive"
        if valid and result["label_benefit_score"]
        else "null"
        if valid
        else "disqualified"
    )
    ended = time.monotonic_ns()
    spans = [
        dict(phase="preconditions_and_reduction", start_s=0.0, end_s=(measured - started) / 1e9),
        dict(phase="validation", start_s=(measured - started) / 1e9, end_s=(ended - started) / 1e9),
    ]
    receipt = build_current_work_receipt(
        run_id=str(started),
        owner_pid=os.getpid(),
        events=[],
        inference_substrate="read_only_sealed_oracle_audit",
        inference_substrate_class="no_model_load"
        if result["completed_count"]
        else "blocked_no_run",
        execution_venue="host",
        inference_substrate_details=dict(
            scope="read-only sealed evidence; no optimizer or model calls"
        ),
        started_monotonic_ns=started,
        ended_monotonic_ns=ended,
        phase_spans=spans,
    )
    for name in ("tests-first", "repository-health"):
        diagnostic = Path("/tmp/carnot8898-" + name + ".log")
        if diagnostic.is_file():
            shutil.copyfile(diagnostic, raw / (name + ".log"))
    helpers = [
        "scripts/experiment_template.py",
        "python/carnot/reporting/current_work_receipt.py",
        "python/carnot/reporting/primary_publication.py",
        "scripts/adversarial_verify.py",
        "scripts/verdict_row_consistency_lint.py",
    ]
    value = dict(
        result,
        experiment_id=8898,
        task_id=m.TASK,
        milestone="2026.11.761",
        run_date=args.date,
        honest_verdict="complete_" + verdict + "_signed_benefit_audit",
        verdict_class=verdict,
        inference_substrate=receipt["inference_substrate"],
        inference_substrate_class=receipt["inference_substrate_class"],
        execution_venue="host",
        MODEL_SPECS=[],
        model_invocation_counts=receipt["invocation_counts"],
        numerical_training_calls=0,
        verifier_is_oracle=True,
        exposure_scope="Constructed signed-record truth only; no natural-language generalization or generator training.",
        independent_generalization_score=0,
        duration_s=receipt["duration_s"],
        phase_spans=spans,
        preconditions_checked=pre,
        random_seed=m.base.SEED,
        producer_invocation=invocation,
        current_work_receipt=receipt,
        source_artifact_hashes=pre["references"],
        code_config_hashes=[reference(m.ROOT / p) for p in m.OWNED + m.TESTS + helpers + m.HELPERS],
        raw_shard_hashes=[reference(p) for p in sorted(raw.rglob("*")) if p.is_file()]
        + validation["raw_references"],
        raw_directory=str(raw),
        gate_check_summary=[dict(r, hash=r.get("sha256")) for r in pre["diagnostics"]],
        acceptance_gates=m.GATES,
        required_checks_passed=valid,
        validation_receipts=validation["receipts"],
        owned_statement_coverage=validation["coverage"],
        owned_errors=errors,
        censor_reason=reason,
        audit_ready_score=int(valid),
        cited_upstream_artifacts=[
            dict(
                experiment_id=8895,
                task_id="exp8895-signed-causal-streams",
                sha256=m.PIN,
                artifact_field="causal_rows_ready_score",
                expected=1,
            ),
            dict(
                experiment_id=8892,
                task_id=m.methods.TASK,
                sha256=m.METHOD_PIN,
                artifact_field="signed_methods_ready_score",
                expected=1,
            ),
        ],
        methodology="Reconstruct sealed probabilities and causal joins without training; average within24 paired streams before10000 fixed simultaneous99-percent bootstrap draws over27 frozen allocations including direct feature attribution. Query-template utility is separate from label information.",
        repository_health=dict(
            command=".venv/bin/pytest tests/python -q",
            scope="unrelated repository health; one bounded240-second run",
            passed=False,
            exit_code=124,
            timeout_s=240,
            log=str(raw / "repository-health.log"),
        ),
        flagged_adversarial=False,
        field_principles={},
    )
    value = normalize_artifact_for_template_write(value)
    value["field_principles"] = {
        k: "Actual producer or retained frozen operand; constructed oracle evidence only."
        for k in value
    }
    value["field_principles"].update(
        audit_ready_score="Complete independent reduction and owned validation, separate from benefit.",
        label_benefit_score="All eight frozen label-information and retention gates pass jointly.",
        feature_specific_benefit_score="Signed benefit conjunction plus a strictly negative direct signed-minus-old matched_true future Brier upper bound.",
        old_feature_label_benefit_score="The legacy-feature conjunction is reported separately and cannot qualify the signed mechanism.",
        joint_gate_rows="Every old and signed information/retention operand remains visible, followed by the separate direct attribution gate.",
        paired_stream_rows="Average future25-48 or one untouched retention checkpoint inside each paired stream before resampling; repeats never add independent units.",
        topology_support="Eight recurring online families are reported separately for both feature arms; these descriptive strata grant no additional benefit claim.",
        control_rows="Recomputed retained separable/constant-feature legacy-head sensitivity controls; these are diagnostics, not signed-feature generalization evidence.",
        simultaneous_intervals="All27 registered intervals plus direct attribution share27 frozen error allocations; one diagnostic slot is split conservatively.",
        independent_count="24 paired RNG streams; repeated seeds and eight graph templates do not add independent units.",
        continuation_decision="A failed repeated matched-label mechanism retires unchanged; any successor needs a newly frozen falsifiable test.",
        repository_health="Unrelated failures or timeout remain separate from owned coverage and required validation.",
        reproducibility_checksum="Canonical artifact hash checked alongside pinned original bytes and independent reconstruction.",
    )
    value["reproducibility_checksum"] = canonical_hash(value)
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
                honest_verdict="complete_disqualified_signed_benefit_audit",
                verdict_class="disqualified",
                audit_ready_score=0,
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
        receipt = build_current_work_receipt(
            run_id=str(started),
            owner_pid=os.getpid(),
            events=[],
            inference_substrate=value["inference_substrate"],
            inference_substrate_class=value["inference_substrate_class"],
            execution_venue="host",
            inference_substrate_details=dict(
                scope="read-only sealed evidence; no optimizer or model calls"
            ),
            started_monotonic_ns=started,
            ended_monotonic_ns=terminal_end,
            phase_spans=spans,
        )
        value.update(
            duration_s=receipt["duration_s"], phase_spans=spans, current_work_receipt=receipt
        )
        value.pop("reproducibility_checksum")
        value["reproducibility_checksum"] = canonical_hash(value)
        m.progress("terminal_after", 3, 0)
    publish_primary(output, value, replay)
    for path in raw.rglob("*"):
        if path.is_file():
            path.chmod(0o400)
    m.progress(
        "published_" + value["verdict_class"], result["completed_count"], result["censored_count"]
    )
    return 0
