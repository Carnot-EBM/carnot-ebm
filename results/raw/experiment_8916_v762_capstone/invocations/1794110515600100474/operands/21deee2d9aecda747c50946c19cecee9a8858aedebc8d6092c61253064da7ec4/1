"""REQ-REPORT-8914: publish durable cost readiness independently of scientific benefit."""

import argparse
from collections.abc import Callable
import json
import os
from pathlib import Path
import shutil
import sys
from tempfile import TemporaryDirectory
import time
from typing import Any

from carnot.reporting import guarded_durable_cost_8914 as m
from carnot.reporting import topology_causal_runner_8884 as previous
from carnot.reporting import topology_memory_runner_8881 as checks
from carnot.reporting.current_work_receipt import (
    atomic_json,
    canonical_hash,
    sha256_file,
    build_current_work_receipt,
    validate_current_work_receipt,
)
from carnot.reporting.primary_publication import publish_primary
from carnot.reporting.selective_runner_8816 import stream, unqualified
from carnot.reporting.v746_retained_evidence import reference
from carnot.reporting.v748_collection_contract import namespace
from scripts.experiment_template import ExperimentTemplate

Json = dict[str, Any]


def validate(raw: Path, private: Path) -> Json:
    """Reuse exact-node unit/child coverage and original private E2E controls."""
    checker = namespace(checks, "durable8914_validation")
    checker.m = m
    checker.stream = lambda name, argv, logs, **kwargs: stream(
        name,
        argv,
        logs,
        **dict(kwargs, deadline=180 if name == "collect_once" else kwargs.get("deadline", 600)),
    )
    value = dict(checker.validate(raw, private))
    value["receipts"].append(previous.e2e018(raw / "e2e018", private / "e2e018"))
    value["raw_references"] += [reference(p) for p in (raw / "e2e018").rglob("*") if p.is_file()]
    return value


def terminal(candidate: Path, raw: Path) -> list[Json]:
    """Run unchanged artifact scanners and actual cold CLI replay outside the checkout."""
    checker = namespace(checks, "durable8914_terminal")
    checker.m = m
    return list(checker.terminal(candidate, raw))


def qualified(validation: Json) -> bool:
    """Only complete owned statements and actual successful checks grant readiness."""
    coverage = validation["coverage"]
    return bool(
        coverage["complete"]
        and coverage["percent"] == 100
        and {r["path"] for r in coverage["files"]} == set(m.OWNED)
        and not any(r["missing"] or r["excluded"] for r in coverage["files"])
        and validation["union"]["passed"]
        and all(r["passed"] for r in validation["receipts"])
    )


def seal(value: Json, started: int, validation: Json) -> None:
    """Separate complete service measurement, owned qualification and scientific benefit."""
    ready = bool(
        value["completed_count"] == 48
        and qualified(validation)
        and value["preconditions_checked"]["external_ready"]
        and value["preconditions_checked"]["resources"]["passed"]
    )
    verdict = (
        "blocked"
        if not value["preconditions_checked"]["external_ready"]
        else "null"
        if ready
        else "disqualified"
    )
    receipt = build_current_work_receipt(
        run_id=str(started),
        owner_pid=os.getpid(),
        events=[],
        inference_substrate="durable_signed_categorical_head_cpu",
        inference_substrate_class="no_model_load" if value["completed_count"] else "blocked_no_run",
        execution_venue="host",
        inference_substrate_details=dict(
            scope="Real numerical service and uninterrupted controls; no pretrained model loads or generation"
        ),
        started_monotonic_ns=started,
        ended_monotonic_ns=time.monotonic_ns(),
        phase_spans=value["phase_spans"],
        small_ebm_training=dict(
            performed=value["numerical_training_calls"] > 0,
            numerical_training_calls=value["numerical_training_calls"],
        ),
    )
    value.update(
        adaptation_cost_ready_score=int(ready),
        required_checks_passed=ready,
        verdict_class=verdict,
        honest_verdict="complete_" + verdict + "_guarded_durable_cost",
        current_work_receipt=receipt,
        duration_s=receipt["duration_s"],
        inference_substrate=receipt["inference_substrate"],
        inference_substrate_class=receipt["inference_substrate_class"],
        model_invocation_counts=receipt["invocation_counts"],
        owned_statement_coverage=validation["coverage"],
        validation_receipts=validation["receipts"],
        test_union=validation["union"],
    )
    value.pop("reproducibility_checksum", None)
    value["field_principles"] = {
        k: "Actual producer identity, original operand or finite constructed service evidence."
        for k in value
    }
    value["field_principles"].update(
        adaptation_cost_ready_score="Complete correct durable cost data and owned checks; does not require candidate readiness or benefit.",
        inclusive_timer_rows="Nested inclusive wall timers overlap; never sum into a component partition.",
        service_cost_rows="Full measured wall costs include startup, process restart and all rejected work; zero promotions remain charged.",
        feedback_cost_rows="Individual durable acknowledgment latency; first feedback in a block includes original block validation and oracle label acquisition.",
        durability_rows="Actual serialized state bytes and successful file/directory fsync counts, including pending predictions.",
        promotion_cost_rows="Original admission decisions; candidate work is charged whether accepted or rejected.",
        paired_total_service_intervals="Three simultaneous99-percent paired Student-t whole-service differences; twelve independent RNG streams, recurring topology families.",
        numerical_training_calls="Actual service and uninterrupted-control updates; validation test processes have separate execution receipts.",
        repository_health="One bounded full-suite invocation; unrelated health never grants owned qualification.",
        reproducibility_checksum="Canonical artifact identity plus retained-byte and semantic reconstruction checks.",
    )
    value["reproducibility_checksum"] = canonical_hash(value)


def replay(path: Path) -> Json:
    """Rebuild every headline from immutable rows and reject rehashed aggregate drift."""
    value = json.loads(path.read_bytes())
    checksum = value.pop("reproducibility_checksum")
    m.require(canonical_hash(value) == checksum, "artifact_checksum")
    m.require(
        (value["experiment_id"], value["task_id"], value["milestone"], value["run_date"])
        == (8914, m.TASK, "2026.11.762", "20261108"),
        "producer_identity",
    )
    for i, ref in enumerate(
        value["source_artifact_hashes"] + value["code_config_hashes"] + value["raw_shard_hashes"]
    ):
        m.require(sha256_file(Path(ref["path"])) == ref["sha256"], "operand_hash")
        if i % 100 == 0:
            m.progress("replay_operands", i, 0)
    raw = Path(value["raw_directory"])
    expected = m.recover(raw, value["censor_reason"])
    for key, result in expected.items():
        if key != "adaptation_cost_ready_score":
            m.require(value[key] == result, "reduction_drift:" + key)
    m.require(
        not validate_current_work_receipt(value["current_work_receipt"])
        and value["duration_s"] == value["current_work_receipt"]["duration_s"]
        and value["model_invocation_counts"] == value["current_work_receipt"]["invocation_counts"],
        "compute_receipt",
    )
    validation = dict(
        coverage=value["owned_statement_coverage"],
        union=value["test_union"],
        receipts=value["validation_receipts"],
    )
    ready = bool(
        expected["adaptation_cost_ready_score"]
        and qualified(validation)
        and value["preconditions_checked"]["external_ready"]
        and value["preconditions_checked"]["resources"]["passed"]
    )
    verdict = (
        "blocked"
        if not value["preconditions_checked"]["external_ready"]
        else "null"
        if ready
        else "disqualified"
    )
    m.require(
        type(value["adaptation_cost_ready_score"]) is int
        and value["adaptation_cost_ready_score"] == int(ready)
        and value["required_checks_passed"] == ready
        and value["verdict_class"] == verdict
        and value["honest_verdict"] == "complete_" + verdict + "_guarded_durable_cost",
        "terminal_semantics",
    )
    m.require(
        value["verifier_is_oracle"] is True
        and value["independent_generalization_score"] == 0
        and value["speed_quality_recommendation_score"] == 0
        and (verdict != "blocked" or value["gate_check_summary"]),
        "scope_boundary",
    )
    for i, row in enumerate(value["rows"]):
        if row["status"] != "complete":
            continue
        first, last = row["segments"]
        m.require(
            first["state"] == last["initial_state"] and first["pid"] != last["pid"],
            "restart_boundary",
        )
        strip: Callable[[list[Json]], list[Json]] = lambda rows: [
            {k: v for k, v in r.items() if k != "duration_ns"} for r in rows
        ]
        control = json.loads(
            (raw / row["cluster_id"] / row["arm"] / "control/segment-48.json").read_bytes()
        )
        m.require(
            strip(first["predictions"] + last["predictions"]) == strip(control["predictions"])
            and last["state"] == control["state"],
            "cold_control_drift",
        )
        with TemporaryDirectory(prefix="carnot8914-reconstruct-", dir="/tmp") as directory:
            request = Path(directory) / "request.json"
            atomic_json(
                request,
                dict(
                    row,
                    head=value["preconditions_checked"]["heads"][str(row["seed"])],
                    anchors=value["preconditions_checked"]["anchors"],
                ),
            )
            rebuilt = m.segment(request, 48)
            m.require(
                rebuilt["state"] == control["state"]
                and strip(rebuilt["predictions"]) == strip(control["predictions"]),
                "numerical_reconstruction_drift",
            )
        m.progress("cold_service_checked", i + 1, 47 - i)
    return dict(passed=True, status="replay_passed", sha256=sha256_file(path))


def main(argv: list[str] | None = None) -> int:
    """Keep checkpoints private and publish only a terminal validated artifact."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", choices=["20261108"], default="20261108")
    parser.add_argument("--output", type=Path, default=m.ROOT / "results" / (m.NAME + ".json"))
    parser.add_argument("--private-output", type=Path)
    parser.add_argument("--sources", type=Path, default=m.ROOT / "results")
    parser.add_argument("--replay", "--cold-replay", type=Path)
    parser.add_argument("--segment", type=Path)
    parser.add_argument("--stop", type=int, choices=[24, 48], default=48)
    args = parser.parse_args(argv)
    if args.segment:
        m.segment(args.segment, args.stop)
        return 0
    if args.replay:
        try:
            print(json.dumps(replay(args.replay)), flush=True)
            return 0
        except (OSError, ValueError, KeyError, TypeError) as error:
            print("replay_rejected: " + str(error), flush=True)
            return 1
    output = (args.private_output or args.output).absolute()
    if args.private_output and output.resolve().is_relative_to(m.ROOT):
        parser.error("private outputs must remain outside checkout")
    started = time.monotonic_ns()
    raw = output.parent / "raw" / m.NAME / "invocations" / str(started)
    raw.mkdir(parents=True, mode=0o700)
    template = ExperimentTemplate(
        8914,
        "Guarded durable service cost",
        "results/" + m.NAME + ".json",
        repo_root=m.ROOT,
        seed=m.SEED,
    )
    template.check_exclusion_manifest()
    invocation = dict(
        executable=sys.executable,
        arguments=argv if argv is not None else sys.argv[1:],
        cwd=os.getcwd(),
        pid=os.getpid(),
        executed_sources=[reference(m.ROOT / n) for n in m.OWNED],
    )
    atomic_json(raw / "invocation.json", invocation)
    pre = m.preconditions(raw, args.sources)
    atomic_json(raw / "preconditions.json", pre)
    with template.phase("complete_service_measurement"):
        result = m.measure(raw, pre)
    measured = time.monotonic_ns()
    m.progress("validation_before", 0, 1)
    with TemporaryDirectory(prefix="carnot8914-validation-", dir="/tmp") as directory:
        validation = (
            unqualified() if args.private_output else validate(raw / "validation", Path(directory))
        )
    m.progress("validation_after", 1, 0)
    reason = "external_blocked" if not pre["external_ready"] else "unfinished_owned_work"
    health = Path("/tmp/carnot8914-repository-health.log")
    for name in ("tests-first", "tests-red", "repository-health"):
        path = Path("/tmp/carnot8914-" + name + ".log")
        if path.is_file():
            shutil.copyfile(path, raw / (name + ".log"))
    benefit_path = args.sources / "experiment_8911_v762_guarded_benefit_audit.json"
    benefit_hash = sha256_file(benefit_path) if benefit_path.is_file() else None
    benefit = json.loads(benefit_path.read_bytes()) if benefit_hash == m.BENEFIT_PIN else {}
    if benefit_hash == m.BENEFIT_PIN:
        pre["references"].append(reference(benefit_path))
    helpers = [
        "scripts/experiment_template.py",
        "python/carnot/reporting/current_work_receipt.py",
        "python/carnot/reporting/primary_publication.py",
        "python/carnot/reporting/selective_runner_8816.py",
        "python/carnot/reporting/topology_memory_runner_8881.py",
        "python/carnot/reporting/topology_causal_runner_8884.py",
        "scripts/adversarial_verify.py",
        "scripts/verdict_row_consistency_lint.py",
    ]
    value = dict(
        result,
        experiment_id=8914,
        task_id=m.TASK,
        milestone="2026.11.762",
        run_date=args.date,
        execution_venue="host",
        MODEL_SPECS=[],
        verifier_is_oracle=True,
        exposure_scope="Finite constructed typed truth and recurring topologies; no independent natural-language generalization.",
        independent_generalization_score=0,
        preconditions_checked=pre,
        random_seed=m.SEED,
        producer_invocation=invocation,
        source_artifact_hashes=pre["references"],
        code_config_hashes=[
            reference(m.ROOT / p) for p in m.OWNED + m.TESTS + m.methods.OWNED + helpers
        ],
        raw_shard_hashes=[reference(p) for p in sorted(raw.rglob("*")) if p.is_file()]
        + validation["raw_references"],
        raw_directory=str(raw),
        gate_check_summary=pre["diagnostics"],
        acceptance_gates=dict(
            method="Exp8905 guarded_methods_ready_score=1 only",
            cost="48 complete cells,48 individual feedback commits each, cold restart at24, matched work and100-percent owned statement coverage with zero exclusions",
            speed_quality="Separately require authentic Exp8911 guarded_benefit_score=1 and matched total-service intervals",
        ),
        cited_upstream_artifacts=[
            dict(
                path=str(args.sources / (m.SOURCE + ".json")),
                sha256=m.PIN,
                artifact_field="guarded_methods_ready_score",
                expected=1,
            )
        ],
        benefit_gate=dict(
            check="missing_file"
            if benefit_hash is None
            else "rejected_producer"
            if benefit_hash != m.BENEFIT_PIN
            else "observed_zero"
            if benefit.get("guarded_benefit_score") == 0
            else "passed",
            path=str(benefit_path),
            hash=benefit_hash,
            expected=1,
            observed=benefit.get("guarded_benefit_score"),
            authenticated=benefit_hash == m.BENEFIT_PIN,
        ),
        speed_quality_recommendation_score=0,
        censor_reason=reason,
        repository_health=dict(
            command=".venv/bin/pytest tests/python -q",
            scope="Separate unrelated repository health; one bounded300-second invocation",
            log=str(raw / "repository-health.log"),
            observed=health.read_text()[-4000:] if health.is_file() else "not available",
        ),
        flagged_adversarial=False,
        phase_spans=[
            dict(
                phase="preconditions_and_measurement", start_s=0.0, end_s=(measured - started) / 1e9
            ),
            dict(
                phase="owned_validation",
                start_s=(measured - started) / 1e9,
                end_s=(time.monotonic_ns() - started) / 1e9,
            ),
        ],
    )
    seal(value, started, validation)
    candidate = raw / "prospective.json"
    atomic_json(candidate, value)
    if not args.private_output:
        m.progress("terminal_before", 0, 3)
        reports = terminal(candidate, raw)
        validation["receipts"] += reports
        value["raw_shard_hashes"] += [
            reference(Path(r[k + "_path"])) for r in reports for k in ("stdout", "stderr")
        ]
        value["phase_spans"].append(
            dict(
                phase="terminal_validation",
                start_s=value["duration_s"],
                end_s=(time.monotonic_ns() - started) / 1e9,
            )
        )
        seal(value, started, validation)
        m.progress("terminal_after", 3, 0)
    m.require(time.monotonic_ns() - started < 4800 * 10**9, "hard_cap")
    publish_primary(output, value, replay)
    m.progress(
        "published_" + value["verdict_class"], value["completed_count"], value["censored_count"]
    )
    return 0
