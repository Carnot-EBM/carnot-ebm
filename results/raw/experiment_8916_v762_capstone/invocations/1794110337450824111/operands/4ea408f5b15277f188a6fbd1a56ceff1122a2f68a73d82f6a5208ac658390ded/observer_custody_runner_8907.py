"""REQ-REPORT-8907: publish custody qualification separately from cost science."""

import argparse
import json
import os
from pathlib import Path
import shutil
import sys
from tempfile import TemporaryDirectory
import time
from typing import Any

from carnot.reporting import observer_custody_8907 as m
from carnot.reporting import buffered_clock_runner_8894 as historical
from carnot.reporting import ordered_evidence_runner_8831 as ordered
from carnot.reporting import topology_memory_runner_8881 as terminal_runner
from carnot.reporting.current_work_receipt import (
    atomic_json,
    canonical_hash,
    sha256_file,
    build_current_work_receipt,
    validate_current_work_receipt,
)
from carnot.reporting.primary_publication import publish_primary
from carnot.reporting.selective_methods_runner_8829 import qualified
from carnot.reporting.selective_runner_8816 import unqualified
from carnot.reporting.v746_retained_evidence import load, reference
from carnot.reporting.v748_collection_contract import namespace
from scripts.experiment_template import ExperimentTemplate, normalize_artifact_for_template_write

Json = dict[str, Any]


def validate(raw: Path, private: Path) -> Json:
    """Reuse exact-node unit/child coverage and unchanged private E2E018/021 checks."""
    runner = namespace(historical, "custody8907_validation")
    runner.m = m
    return dict(runner.validate(raw, private))


def build(
    work: Json, validation: Json, receipt: Json, work_ref: Json, validation_ref: Json
) -> Json:
    """Completion and custody readiness grant no claim about observer overhead."""
    result, rows = work["result"], work["rows"]
    valid = qualified(validation)
    gates = dict(
        original_retained_replay=bool(result),
        mutation_controls=bool(rows) and all(r["passed"] for r in rows),
        owned_validation=valid,
    )
    ready = int(all(gates.values()))
    verdict = "blocked" if work["external_failure"] else "null" if ready else "disqualified"
    value: Json = dict(
        experiment_id=8907,
        task_id=m.TASK,
        milestone="2026.11.762",
        run_date="20261107",
        honest_verdict=f"complete_{verdict}_observer_custody",
        verdict_class=verdict,
        inference_substrate=receipt["inference_substrate"],
        inference_substrate_class=receipt["inference_substrate_class"],
        execution_venue="host",
        MODEL_SPECS=[],
        model_invocation_counts=receipt["invocation_counts"],
        numerical_training_calls=0,
        verifier_is_oracle=True,
        exposure_scope="Retained constructed observer controls; future24 worlds per size remain unexecuted.",
        independent_generalization_score=0,
        independent_count=0,
        duration_s=receipt["duration_s"],
        phase_spans=receipt["phase_spans"],
        preconditions_checked=dict(
            external_ready=not work["external_failure"], diagnostics=work["diagnostics"]
        ),
        random_seed=m.SEED,
        producer_invocation=work["producer_invocation"],
        current_work_receipt=receipt,
        source_artifact_hashes=work["sources"],
        code_config_hashes=work["code"],
        raw_shard_hashes=[work_ref, validation_ref] + validation["raw_references"] + work["raw"],
        cited_upstream_artifacts=work["sources"],
        rows=rows,
        intended_count=6,
        completed_count=sum(r["status"] == "completed" for r in rows),
        failed_count=sum(r["status"] == "failed" for r in rows),
        censored_count=sum(r["status"] == "censored" for r in rows),
        excluded_count=0,
        gate_check_summary=work["diagnostics"],
        acceptance_gates=gates,
        required_checks_passed=valid,
        validation_receipts=validation["receipts"],
        owned_statement_coverage=validation["coverage"],
        retained_evidence_ready_score=ready,
        retained_operand_manifest=result.get("retained_operand_manifest", []),
        original_receipt_hashes=result.get("original_receipt_hashes", []),
        consumer_bridge_hashes=work["code"],
        mutation_rows=rows,
        cold_replay_receipt={k: v for k, v in result.items() if k != "retained_operand_manifest"},
        observer_equivalence_rows=result.get("observer_equivalence_rows", []),
        repetition_count=1,
        future_world_count=144,
        future_worlds_executed=0,
        scientific_benefit_claim=False,
        repository_health=work["repository_health"],
        work_reference=work_ref,
        validation_reference=validation_ref,
        methodology="Authenticate the sealed V761 execution bundle and original Exp8894 producer; resolve exact retained operands in an isolated namespace; preserve original clock, semantic and validation assertions. Compare exact implementation bytes, replay retained summaries, and run one installed development world with one repetition. Reject private operand mutations. No full cost benchmark, optimization, overhead improvement or generalization claim.",
        flagged_adversarial=False,
    )
    value["field_principles"] = {
        k: "Exact retained or current owned evidence; custody readiness does not imply scientific benefit."
        for k in value
    }
    value["field_principles"].update(
        retained_evidence_ready_score="Integer one only after original replay, exact source equivalence, mutation controls and current owned validation.",
        retained_operand_manifest="Map original paths and seals to immutable execution-bundle bytes; current document edits cannot replace them.",
        independent_count="Zero: deterministic custody attacks are not independent scientific samples.",
        cold_replay_receipt="Original replay and one fresh installed development control; future cost worlds stay unexecuted.",
    )
    value = normalize_artifact_for_template_write(value)
    value["reproducibility_checksum"] = canonical_hash(value)
    return value


def verify(path: Path) -> Json:
    """Recompute the original observer rather than trusting rehashed readiness fields."""
    value = load(reference(path))
    m.require(
        value["reproducibility_checksum"]
        == canonical_hash({k: v for k, v in value.items() if k != "reproducibility_checksum"}),
        "artifact_checksum",
    )
    work, validation = load(value["work_reference"]), load(value["validation_reference"])
    for ref in (
        value["source_artifact_hashes"] + value["code_config_hashes"] + value["raw_shard_hashes"]
    ):
        m.require(sha256_file(Path(ref["path"])) == ref["sha256"], "operand_drift")
    m.require(not validate_current_work_receipt(value["current_work_receipt"]), "current_receipt")
    m.require(
        value
        == build(
            work,
            validation,
            value["current_work_receipt"],
            value["work_reference"],
            value["validation_reference"],
        ),
        "reduction_drift",
    )
    if work["result"]:
        m.require(m.qualify() == work["result"], "original_replay_drift")
        m.require(
            len(work["rows"]) == 6
            and all(
                r["passed"] and r["metric"] == 1 and r["rejected"] == r["expected_rejected"]
                for r in work["rows"]
            ),
            "mutation_control_drift",
        )
    if value["required_checks_passed"]:
        checker = namespace(ordered.checker, "custody8907_evidence")
        checker.m = m
        m.require(checker.validation_evidence(validation), "validation_evidence_drift")
    return dict(passed=True, status="replay_passed", sha256=sha256_file(path))


def terminal(path: Path, raw: Path) -> Json:
    """Use unchanged scanners and the real outside-checkout CLI before publication."""
    runner = namespace(terminal_runner, "custody8907_terminal")
    runner.m = m
    receipts = runner.terminal(path, raw)
    return dict(passed=all(r["passed"] for r in receipts), receipts=receipts)


def main(argv: list[str] | None = None) -> int:
    """Keep all checkpoints private until replay and unchanged terminal checks pass."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", choices=["20261107"], default="20261107")
    parser.add_argument("--output", type=Path, default=m.ROOT / "results" / (m.NAME + ".json"))
    parser.add_argument("--private-output", type=Path)
    parser.add_argument("--cold-replay", "--replay", dest="replay", type=Path)
    args = parser.parse_args(argv)
    if args.replay:
        result = m.replay(args.replay)
        print(json.dumps(result), flush=True)
        return int(not result["passed"])
    output = (args.private_output or args.output).absolute()
    if args.private_output and output.resolve().is_relative_to(m.ROOT):
        parser.error("private outputs must remain outside checkout")
    started = time.monotonic_ns()
    raw = output.parent / "raw" / m.NAME / "invocations" / str(started)
    raw.mkdir(parents=True, mode=0o700)
    template = ExperimentTemplate(
        8907, "Observer retained custody", "results/" + m.NAME + ".json", seed=m.SEED
    )
    work: Json = dict(
        result={},
        rows=[],
        external_failure=False,
        diagnostics=[],
        sources=[],
        code=[reference(m.ROOT / p) for p in m.OWNED + m.TESTS],
        raw=[],
        repository_health=None,
        producer_invocation=dict(
            python=sys.executable,
            argv=argv if argv is not None else sys.argv[1:],
            cwd=os.getcwd(),
            pid=os.getpid(),
        ),
    )
    m.progress("before_original_gates")
    try:
        load(m.BUNDLE)
        work["sources"] = [m.BUNDLE]
        with template.phase("retained_observer_replay"):
            work["result"] = m.qualify(control_raw=raw / "fresh_control")
        with template.phase("custody_mutation_controls"):
            work["rows"] = m.controls(raw / "controls")
        m.require(all(r["passed"] for r in work["rows"]), "owned_mutation_controls")
    except (OSError, ValueError, KeyError, TypeError, StopIteration) as error:
        work["external_failure"] = not str(error).startswith("owned_")
        missing = isinstance(error, (OSError, KeyError, StopIteration))
        work["diagnostics"] = [
            dict(
                check="missing_file"
                if missing
                else "rejected_producer"
                if work["external_failure"]
                else "owned_control_failure",
                upstream="exp8894-buffered-clock-methods",
                artifact_field="retained_execution_operand",
                expected="original sealed bytes and unchanged assertions",
                observed=str(error),
                path=str(getattr(error, "filename", None) or m.BUNDLE["path"]),
                hash=m.BUNDLE["sha256"],
            )
        ]
        if isinstance(error, m.CustodyError):
            work["diagnostics"] = [error.diagnostic]
        work["rows"] = [
            dict(
                unit_id=kind,
                cluster_id="custody_controls",
                arm="mutation",
                seed=m.SEED,
                metric=0,
                denominator=1,
                status="censored" if work["external_failure"] else "failed",
                missing_reason=str(error),
                passed=False,
            )
            for kind in m.CASES
        ]
    measured = time.monotonic_ns()
    m.progress("after_original_gates_before_validation", len(work["rows"]), 0)
    with TemporaryDirectory(prefix="exp8907-validation-") as directory:
        validation = (
            unqualified() if args.private_output else validate(raw / "validation", Path(directory))
        )
    ended = time.monotonic_ns()
    m.require((ended - started) / 1e9 < 4200, "owned_validation_deadline")
    m.progress("after_owned_validation")
    for name in ["red-import.log", "tests-first.log", "full-suite.log", "exit"]:
        source = Path("/tmp/exp8907-health") / name
        if source.is_file():
            target = raw / "repository_health" / name
            target.parent.mkdir(exist_ok=True)
            shutil.copyfile(source, target)
    health = raw / "repository_health"
    work["repository_health"] = dict(
        command=[".venv/bin/pytest", "tests/python", "-q"],
        full_suite_runs=1,
        status=(health / "exit").read_text().strip() if (health / "exit").exists() else "pending",
        logs=[reference(p) for p in sorted(health.glob("*"))],
    )
    substrate = "blocked_no_run" if work["external_failure"] else "no_model_load"
    receipt = build_current_work_receipt(
        run_id=m.TASK + "/" + str(started),
        owner_pid=os.getpid(),
        events=[],
        inference_substrate=substrate,
        inference_substrate_class=substrate,
        execution_venue="host",
        inference_substrate_details=dict(
            compute="retained observer replay and installed development control",
            model_invoked=False,
        ),
        started_monotonic_ns=started,
        ended_monotonic_ns=ended,
        phase_spans=[
            dict(phase="custody_replay_and_controls", start_s=0, end_s=(measured - started) / 1e9),
            dict(
                phase="owned_validation",
                start_s=(measured - started) / 1e9,
                end_s=(ended - started) / 1e9,
            ),
        ],
    )
    work["raw"] = [
        reference(p)
        for p in sorted(raw.rglob("*"))
        if p.is_file() and "validation" not in p.relative_to(raw).parts
    ]
    atomic_json(raw / "work.json", work)
    atomic_json(raw / "validation.json", validation)
    value = build(
        work, validation, receipt, reference(raw / "work.json"), reference(raw / "validation.json")
    )
    m.progress("before_terminal_publication")
    publication = publish_primary(output, value, lambda candidate: terminal(candidate, raw))
    atomic_json(raw / "publication_receipt.json", publication)
    m.progress(
        "complete_" + value["verdict_class"], value["completed_count"], value["censored_count"]
    )
    return 0
