"""REQ-REPORT-8954: publish a checked null or qualified constructed-truth result."""

import argparse
import json
import os
from pathlib import Path
import shutil
import sys
from tempfile import TemporaryDirectory
import time
from types import SimpleNamespace
from typing import Any

from carnot.reporting import gradient_benefit_audit_8954 as m
from carnot.reporting import topology_causal_runner_8884 as inherited
from carnot.reporting import evidence_evaluation_runner_8779 as shared
from carnot.reporting.current_work_receipt import (
    atomic_json,
    canonical_hash,
    sha256_file,
    build_current_work_receipt,
    validate_current_work_receipt,
)
from carnot.reporting.primary_publication import publish_primary
from carnot.reporting.selective_runner_8816 import unqualified, stream
from carnot.reporting.v746_retained_evidence import reference, load
from carnot.reporting.v748_collection_contract import namespace
from scripts.experiment_template import ExperimentTemplate, normalize_artifact_for_template_write

Json = dict[str, Any]
checker = namespace(inherited, "compatible8954_validation")
checker.m = m
validator = namespace(shared, "compatible8954_statement_custody")
validator.e = SimpleNamespace(ROOT=m.ROOT, OWNED=m.OWNED, TESTS=m.TESTS)
validator.original = SimpleNamespace(
    CONSUMERS=[
        "tests/python/test_causal_memory_8802.py",
        "tests/python/test_experiment_7891_v685_authority_lifecycle.py",
        "tests/python/test_restricted_decision_audit_8210.py",
    ]
)


def validation_child(name: str, argv: list[str], logs: Path, **kwargs: Any) -> Json:
    """Allow bounded cold collection and full retained-row replay in the owned child."""
    if name == "collect_once":
        kwargs["deadline"] = 180
    if name == "child_0":
        kwargs["deadline"] = 1200
    return dict(stream(name, argv, logs, **kwargs))


validator.child = validation_child


def validate(raw: Path, private: Path) -> Json:
    """Reuse actual unit/child statement custody and private E2E-018/021 checks."""
    result = dict(validator.validate(raw, private))
    result["receipts"].append(checker.e2e018(raw / "e2e018", private / "e2e018"))
    result["raw_references"] += [
        reference(p) for p in sorted((raw / "e2e018").rglob("*")) if p.is_file()
    ]
    return result


def terminal(candidate: Path, raw: Path) -> list[Json]:
    """Run unchanged scanners and actual cold replay outside the checkout."""
    return list(checker.terminal(candidate, raw))


def semantics(value: Json) -> tuple[bool, str]:
    """A complete null can qualify; owned failures and absent upstream evidence cannot."""
    coverage = value["owned_statement_coverage"]
    ready = bool(
        value["preconditions_checked"]["external_ready"]
        and value["preconditions_checked"]["resources"]["passed"]
        and not value["owned_errors"]
        and value["completed_count"] == m.COUNT
        and bool(value["information_controls"])
        and all(r["passed"] for r in value["information_controls"])
        and coverage["complete"]
        and coverage["percent"] == 100
        and {r["path"] for r in coverage["files"]} == set(m.OWNED)
        and not any(r["missing"] or r["excluded"] for r in coverage["files"])
        and value["test_union"]["passed"]
        and bool(value["validation_receipts"])
        and all(r["passed"] for r in value["validation_receipts"])
    )
    verdict = (
        "blocked"
        if not value["preconditions_checked"]["external_ready"]
        else "circular_positive"
        if ready and value["gradient_benefit_score"]
        else "null"
        if ready
        else "disqualified"
    )
    return ready, verdict


def seal(value: Json, started: int, spans: list[Json]) -> None:
    """Bind actual current work and explain each field before hashing the final bytes."""
    receipt = build_current_work_receipt(
        run_id=str(started),
        owner_pid=os.getpid(),
        events=[],
        inference_substrate="aggregation_from_upstream_artifacts",
        inference_substrate_class="no_model_load" if value["completed_count"] else "blocked_no_run",
        execution_venue="host",
        inference_substrate_details=dict(
            scope="Read-only row reduction; no current labels, predictions, training or model calls"
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
        k: "Actual invocation or authenticated finite constructed evidence; no natural-language claim."
        for k in value
    }
    value["field_principles"].update(
        gradient_audit_ready_score="Complete independent reduction and owned checks; a valid null can qualify.",
        gradient_benefit_score="All eleven sealed future/retention upper bounds and owned validity must pass.",
        paired_intervals="10000 whole-stream paired draws; one-sided alpha0.01/11 allocation.",
        paired_stream_rows="Within-stream paired arm means; future ordinals25–48, retention checkpoints0/24/48.",
        independent_count="24 independently seeded histories on recurring constructed topology families; no unseen-topology claim.",
        selection_diversity="Descriptive source-group reuse only; cannot substitute for future benefit.",
        promotion_summary="Operational admission only; training improvement is not benefit.",
        continuation_decision="Retire only this fixed gradient-cosine/buffer32/SGD0.1/two-gradient/delay8/guard/query mechanism.",
        repository_health="Separate full-suite health diagnostic; cannot grant owned readiness.",
        mechanism_diagnostics="Paired selection disagreement, actual applied gradient diversity and guard rejection; cannot rescue failed loss bounds.",
        reproducibility_checksum="Canonical artifact identity plus cold reconstruction from pinned producer bytes.",
    )
    value.pop("reproducibility_checksum", None)
    value["reproducibility_checksum"] = canonical_hash(value)


def replay(path: Path) -> Json:
    """Independently rebuild every effect so rehashing false aggregates cannot qualify."""
    value = json.loads(path.read_bytes())
    checksum = value.pop("reproducibility_checksum")
    m.require(canonical_hash(value) == checksum, "artifact_checksum")
    m.require(
        (value["experiment_id"], value["task_id"], value["milestone"], value["run_date"])
        == (8954, m.TASK, "2026.11.765", "20261109"),
        "producer_identity",
    )
    refs = value["source_artifact_hashes"] + value["code_config_hashes"] + value["raw_shard_hashes"]
    for index, ref in enumerate(refs):
        m.require(sha256_file(Path(ref["path"])) == ref["sha256"], "operand_hash")
        if index % 100 == 0:
            m.progress("replay_authenticate", index, len(refs) - index)
    m.require(
        value["producer_invocation"]["executed_sources"] == [reference(m.ROOT / n) for n in m.OWNED]
        and value["source_artifact_hashes"] == value["preconditions_checked"]["references"],
        "source_inventory",
    )
    m.require(value["acceptance_gates"] == m.GATES, "acceptance_drift")
    receipt = value["current_work_receipt"]
    m.require(
        not validate_current_work_receipt(receipt)
        and value["numerical_training_calls"] == 0
        and value["duration_s"] == receipt["duration_s"]
        and value["phase_spans"] == receipt["phase_spans"]
        and value["model_invocation_counts"] == receipt["invocation_counts"]
        and value["inference_substrate_class"] == receipt["inference_substrate_class"]
        and value["inference_substrate"]
        == receipt["inference_substrate"]
        == "aggregation_from_upstream_artifacts",
        "compute_scope",
    )
    pre = value["preconditions_checked"]
    if pre["external_ready"] and pre["resources"]["passed"] and value["completed_count"] == m.COUNT:
        source = Path(value["raw_directory"]) / (m.SOURCE + ".json")
        m.require(sha256_file(source) == m.PIN, "original_source_pin")
        expected = m.reduce(load(reference(source)))
    else:
        expected = m.empty(value["censor_reason"])
    ready, verdict = semantics(value)
    if not ready:
        expected["gradient_benefit_score"] = 0
    for key, observed in expected.items():
        m.require(value[key] == observed, "reduction_drift:" + key)
    m.require(
        value["required_checks_passed"] == ready
        and type(value["gradient_audit_ready_score"]) is int
        and value["gradient_audit_ready_score"] == int(ready)
        and type(value["gradient_benefit_score"]) is int
        and value["verdict_class"] == verdict
        and value["honest_verdict"] == "complete_" + verdict + "_gradient_benefit_audit",
        "terminal_semantics",
    )
    m.require(verdict != "blocked" or value["gate_check_summary"], "blocked_diagnostic")
    m.require(
        value["verifier_is_oracle"] is True
        and value["independent_generalization_score"] == 0
        and value["MODEL_SPECS"] == value["model_specs"] == [],
        "oracle_boundary",
    )
    return dict(passed=True, status="replay_passed", sha256=sha256_file(path))


def main(argv: list[str] | None = None) -> int:
    """Complete private validation before atomically publishing one terminal audit."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", choices=["20261109"], default="20261109")
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
        8954,
        "Gradient selection benefit audit",
        "results/" + m.NAME + ".json",
        repo_root=m.ROOT,
        seed=m.SEED,
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
    with TemporaryDirectory(prefix="carnot8954-validation-", dir="/tmp") as directory:
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
    context = []
    for path in [Path("/tmp/carnot8954-tests-first.log")] + sorted(
        Path("/tmp/carnot8954-health").glob("*")
    ):
        target = raw / "context" / path.name
        target.parent.mkdir(exist_ok=True)
        shutil.copyfile(path, target)
        context.append(reference(target))
    helpers = (
        m.causal.OWNED
        + m.causal.methods.OWNED
        + [
            "python/carnot/reporting/topology_causal_runner_8884.py",
            "python/carnot/reporting/topology_memory_runner_8881.py",
            "python/carnot/reporting/evidence_evaluation_runner_8779.py",
            "python/carnot/reporting/selective_runner_8816.py",
            "python/carnot/reporting/causal_learning_audit_8810.py",
            "scripts/experiment_template.py",
            "python/carnot/reporting/current_work_receipt.py",
            "python/carnot/reporting/primary_publication.py",
            "scripts/adversarial_verify.py",
            "scripts/verdict_row_consistency_lint.py",
        ]
    )
    health_path = Path("/tmp/carnot8954-health/receipt.json")
    health = (
        json.loads(health_path.read_bytes())
        if health_path.is_file()
        else dict(passed=False, status="not_completed")
    )
    value = dict(
        result,
        experiment_id=8954,
        task_id=m.TASK,
        milestone="2026.11.765",
        run_date=args.date,
        execution_venue="host",
        MODEL_SPECS=[],
        model_specs=[],
        numerical_training_calls=0,
        verifier_is_oracle=True,
        exposure_scope="Sealed finite constructed truth; no natural-language, verifier moat or unseen-topology claim.",
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
                producer=m.SOURCE,
                sha256=m.PIN,
                fields_imported=["gradient_rows_ready_score", "rows", "stream_refs"],
            ),
            dict(
                producer=m.causal.SOURCE,
                sha256=m.causal.PIN,
                fields_imported=["gradient_methods_ready_score", "protocol"],
            ),
        ],
        methodology_note="Read-only constructed-oracle evidence; no live-model self-improvement or independent generalization.",
        global_test_debt=health,
        reproduction_commands=[
            "cd "
            + str(m.ROOT)
            + " && PYTHONPATH=python .venv/bin/python -u "
            + m.CLI
            + " --date 20261109",
            "env -u PYTHONPATH "
            + sys.executable
            + " -u "
            + str(m.ROOT / m.CLI)
            + " --cold-replay "
            + str(output),
        ],
        methodology="Read-only independent reduction:24 paired streams, future25–48,48 retention queries per checkpoint,10000 fixed-seed whole-stream draws, eleven one-sided Bonferroni0.01/11 upper bounds. No refit or new oracle labels.",
        repository_health=dict(
            command=".venv/bin/pytest tests/python -q",
            receipt=health,
            scope="Separate repository health; no owned qualification",
            retained=context,
        ),
        flagged_adversarial=False,
        field_principles={},
    )
    ready, verdict = semantics(value)
    value.update(
        required_checks_passed=ready,
        gradient_audit_ready_score=int(ready),
        gradient_benefit_score=value["gradient_benefit_score"] if ready else 0,
        verdict_class=verdict,
        honest_verdict="complete_" + verdict + "_gradient_benefit_audit",
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
                honest_verdict="complete_disqualified_gradient_benefit_audit",
                verdict_class="disqualified",
                gradient_audit_ready_score=0,
                gradient_benefit_score=0,
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
    m.require((time.monotonic_ns() - started) / 1e9 < 4200, "total_work_budget")
    publish_primary(output, value, replay)
    m.progress(
        "published_" + value["verdict_class"], result["completed_count"], result["censored_count"]
    )
    return 0
