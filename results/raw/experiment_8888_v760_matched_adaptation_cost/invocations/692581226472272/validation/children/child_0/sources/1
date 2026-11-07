"""REQ-REPORT-8888: authenticate, measure, validate and expose one cost-only primary."""

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
from unittest.mock import patch

from carnot.reporting import matched_adaptation_cost_8888 as m
from carnot.reporting import evidence_evaluation_runner_8779 as checks
from carnot.reporting.causal_learning_audit_8810 import require
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
from scripts.experiment_template import ExperimentTemplate, normalize_artifact_for_template_write

Json = dict[str, Any]
SOURCES = [
    (
        "experiment_8881_v760_topology_memory_methods",
        "sha256:4d43ee0ea01ae703efb1087b5a2f2a253c529f46d3e2c9d7f87ff2b028d3c1e3",
        "topology_methods_ready_score",
        "exp8881-topology-memory-methods",
    ),
    (
        "experiment_8862_v758_linear_durable_cost",
        "sha256:6fd5d69c40f4fc3884ef5c3cce6ee0de5f98778b969b9850bfd9df89c03d2931",
        "whole_cost_ready_score",
        "exp8862-linear-durable-cost",
    ),
]
AUTH = "openspec/change-proposals/research-methods-v760-20261106.json"
AUTH_PIN = "sha256:c99296bd441c10200bad04e77643649bd65f62fb2199029302b24473af6da3d7"


def preconditions(raw: Path, sources: Path) -> Json:
    """Check original producers, every retained operand, authority and private CPU resources."""
    raw.mkdir(parents=True, mode=0o700, exist_ok=True)
    raw.chmod(0o700)
    atomic_json(raw / "writable.json", dict(private=True))
    failures, refs, heads = [], [], {}
    fit_reference = None
    operands: Json = {}
    for name, pin, field, task in SOURCES:
        path = sources / (name + ".json")
        digest = sha256_file(path) if path.is_file() else None
        malformed = False
        try:
            value = json.loads(path.read_bytes()) if digest else {}
            if not isinstance(value, dict):
                value, malformed = {}, True
        except ValueError:
            value, malformed = {}, True
        check = (
            "missing_file"
            if digest is None
            else "rejected_producer"
            if malformed
            else "missing_field"
            if field not in value
            else "rejected_producer"
            if digest != pin or value.get("task_id") != task
            else "observed_zero"
            if value[field] != 1
            else None
        )
        if check:
            failures.append(
                dict(
                    check=check,
                    upstream=task,
                    artifact_field=field,
                    expected=1,
                    observed=value.get(field),
                    path=str(path),
                    hash=digest,
                )
            )
            continue
        target = raw / "inputs" / path.name
        target.parent.mkdir(exist_ok=True)
        shutil.copyfile(path, target)
        refs += [reference(path), reference(target)]
        for key in ("source_artifact_hashes", "code_config_hashes", "raw_shard_hashes"):
            for ref in value[key]:
                operands[ref["path"] + ref["sha256"]] = ref
        if field == "topology_methods_ready_score":
            fit_reference = value["fit_manifest"]
    authority = m.ROOT / AUTH
    authority_hash = sha256_file(authority) if authority.is_file() else None
    if authority_hash:
        refs.append(reference(authority))
    if authority_hash != AUTH_PIN:
        failures.append(
            dict(
                check="missing_file" if authority_hash is None else "rejected_producer",
                upstream="v760_authority",
                artifact_field="execution_authority",
                expected=AUTH_PIN,
                observed=authority_hash,
                path=str(authority),
                hash=authority_hash,
            )
        )
    for index, ref in enumerate(operands.values()):
        path = Path(ref["path"])
        digest = sha256_file(path) if path.is_file() else None
        if digest != ref["sha256"]:
            failures.append(
                dict(
                    check="missing_file" if digest is None else "rejected_producer",
                    upstream="original_operand",
                    artifact_field="retained_raw_or_config",
                    expected=ref["sha256"],
                    observed=digest,
                    path=str(path),
                    hash=digest,
                )
            )
        if index % 2000 == 0:
            m.progress("authenticate_operands", index, len(operands) - index)
    atomic_json(raw / "operand_manifest.json", dict(references=list(operands.values())))
    refs.append(reference(raw / "operand_manifest.json"))
    if not failures and fit_reference:
        heads = json.loads(Path(fit_reference["path"]).read_bytes())["heads"]
    resources = dict(
        private_scratch=oct(raw.stat().st_mode & 0o777) == "0o700",
        free_disk_bytes=shutil.disk_usage(raw).free,
        minimum_free_bytes=128 * 1024**2,
        tools=all(
            (m.ROOT / ".venv/bin" / n).is_file() for n in ("python", "pytest", "ruff", "mypy")
        ),
        cpu_backend=os.environ.get("JAX_PLATFORMS") == "cpu",
        runtime_budget_s=4200,
        hard_cap_s=4800,
        validation_reserve_s=1200,
    )
    resources["passed"] = bool(
        resources["private_scratch"]
        and resources["tools"]
        and resources["cpu_backend"]
        and resources["free_disk_bytes"] >= resources["minimum_free_bytes"]
    )
    return dict(
        external_ready=not failures,
        diagnostics=failures,
        references=refs,
        heads=heads,
        resources=resources,
    )


def measure(raw: Path, pre: Json) -> Json:
    """Keep every scheduled cell, using one complete paired stream as a runtime pilot."""
    began = time.monotonic_ns()
    schedule = m.roster()
    atomic_json(raw / "schedule.json", dict(rows=schedule))
    rows, errors = [], []
    reason = "external_precondition" if not pre["external_ready"] else "owned_resource_precondition"
    admitted = pre["external_ready"] and pre["resources"]["passed"]
    projected = None
    for index, spec in enumerate(schedule):
        if admitted:
            m.progress("unit_before", index, 48 - index)
            row = m.execute(raw / f"unit-{index:02d}", spec, pre["heads"][str(spec["seed"])])
            m.progress("unit_after", index + 1, 47 - index)
            if not m.parity(row):
                admitted, reason = False, "owned_lifecycle_control"
                errors.append(reason)
            if index == 3:
                projected = (time.monotonic_ns() - began) / 1e9 * 12 + 1200
                admitted = admitted and projected < 4200
                if projected >= 4200:
                    reason = "owned_runtime_projection"
                    errors.append(reason)
        else:
            row = dict(
                spec, status="censored", missing_reason=reason, metric=None, complete_ns=None
            )
        rows.append(row)
        if (time.monotonic_ns() - began) / 1e9 > 3000:
            admitted, reason = False, "owned_measurement_deadline"
            errors.append(reason)
    return dict(rows=rows, owned_errors=errors, projected_runtime_s=projected)


def validate(raw: Path, private: Path) -> Json:
    """Reuse exact-node unit/child coverage and unchanged private E2E-018/021 suites."""
    checker = namespace(checks, "cost8888_validation")
    checker.e = SimpleNamespace(ROOT=m.ROOT, OWNED=m.OWNED, TESTS=m.TESTS)
    checker.original = SimpleNamespace(
        CONSUMERS=[
            "tests/python/test_causal_memory_8802.py",
            "tests/python/test_current_work_receipt.py",
            "tests/python/test_primary_publication_7928.py",
            "tests/python/test_experiment_7891_v685_authority_lifecycle.py",
            "tests/python/test_restricted_decision_audit_8210.py",
        ]
    )
    checker.child = stream
    with patch.dict(os.environ, TMPDIR=str(private)):
        return dict(checker.validate(raw, private))


def build(work: Json, validation: Json) -> Json:
    """Separate complete operation readiness from scientific benefit and repository health."""
    pre, rows = work["preconditions"], work["measurement"]["rows"]
    completed = [r for r in rows if r["status"] == "complete"]
    valid = bool(
        validation["coverage"]["complete"]
        and validation["union"]["passed"]
        and all(r["passed"] for r in validation["receipts"])
    )
    ready = (
        pre["external_ready"]
        and pre["resources"]["passed"]
        and len(completed) == 48
        and all(m.parity(r) for r in completed)
        and valid
        and not work["measurement"]["owned_errors"]
    )
    verdict = "blocked" if not pre["external_ready"] else "null" if ready else "disqualified"
    receipt = build_current_work_receipt(
        run_id=str(work["started"]),
        owner_pid=work["invocation"]["pid"],
        events=[],
        inference_substrate="cpu_linear_normalized_energy_head",
        inference_substrate_class="no_model_load" if completed else "blocked_no_run",
        execution_venue="host",
        inference_substrate_details=dict(backend="JAX CPU", pretrained_loads=0),
        started_monotonic_ns=work["started"],
        ended_monotonic_ns=work["ended"],
        phase_spans=work["spans"],
        small_ebm_training=dict(
            performed=bool(completed),
            numerical_training_calls=2
            * sum(
                sum(r["optimizer_steps"] for s in row["segments"] for r in s["releases"])
                for row in completed
            ),
        ),
    )
    costs = m.reduce_costs(rows, work["setup_ns"])
    value = dict(
        experiment_id=8888,
        task_id=m.TASK,
        milestone="2026.11.760",
        run_date="20261107",
        honest_verdict=f"complete_{verdict}_matched_adaptation_cost",
        verdict_class=verdict,
        inference_substrate=receipt["inference_substrate"],
        inference_substrate_class=receipt["inference_substrate_class"],
        execution_venue="host",
        MODEL_SPECS=[],
        model_invocation_counts=receipt["invocation_counts"],
        numerical_training_calls=receipt["small_ebm_training"]["numerical_training_calls"],
        verifier_is_oracle=True,
        exposure_scope="constructed signed records; separate cost RNG streams; CPU only",
        independent_generalization_score=0,
        duration_s=receipt["duration_s"],
        phase_spans=work["spans"],
        preconditions_checked={k: v for k, v in pre.items() if k != "heads"},
        random_seed=m.SEED,
        producer_invocation=work["invocation"],
        current_work_receipt=receipt,
        source_artifact_hashes=pre["references"],
        code_config_hashes=work["code_refs"],
        raw_shard_hashes=work["raw_refs"] + validation["raw_references"],
        cited_upstream_artifacts=[
            dict(
                experiment_id=int(n.split("_")[1]),
                task_id=task,
                sha256=pin,
                artifact_field=field,
                expected=1,
            )
            for n, pin, field, task in SOURCES
        ],
        rows=rows,
        intended_count=48,
        completed_count=len(completed),
        failed_count=sum(r["status"] == "failed" for r in rows),
        censored_count=sum(r["status"] == "censored" for r in rows),
        excluded_count=0,
        independent_count=len(
            {
                r["cluster_id"]
                for r in completed
                if sum(
                    x["status"] == "complete" and x["cluster_id"] == r["cluster_id"] for x in rows
                )
                == 4
            }
        ),
        gate_check_summary=pre["diagnostics"],
        required_checks_passed=valid,
        acceptance_gates=dict(
            complete_cells=48,
            paired_streams=12,
            restart_at_query=24,
            confidence=0.99,
            owned_statement_coverage=100,
            quality_required_for_speed_claim=True,
            benefit_required_for_readiness=False,
        ),
        validation_receipts=validation["receipts"],
        owned_statement_coverage=validation["coverage"],
        adaptation_cost_ready_score=int(ready),
        per_stream_results=costs,
        label_call_counts=[
            dict(unit_id=r["unit_id"], segments=[s["label_counts"] for s in r["segments"]])
            for r in completed
        ],
        template_occupancy=[
            dict(unit_id=r["unit_id"], segments=[s["occupancy"] for s in r["segments"]])
            for r in completed
        ],
        phase_costs=[
            dict(unit_id=r["unit_id"], segments=[s["phase_costs"] for s in r["segments"]])
            for r in completed
        ],
        complete_parent_costs=[
            dict(unit_id=r["unit_id"], complete_ns=r["complete_ns"], denominator=48)
            for r in completed
        ],
        restart_receipts=[dict(unit_id=r["unit_id"], **r["restart"]) for r in completed],
        durable_commit_rows=[
            dict(unit_id=r["unit_id"], segments=[s["audit"] for s in r["segments"]])
            for r in completed
        ],
        tail_drain_rows=[
            dict(unit_id=r["unit_id"], **r["segments"][-1]["tail_drain"]) for r in completed
        ],
        quality_evidence_refs=[],
        operation_counts=[dict(unit_id=r["unit_id"], **r["operation_counts"]) for r in completed],
        cpu_memory_counters=[dict(unit_id=r["unit_id"], **r["cpu_counters"]) for r in completed],
        batch_paths=[],
        fpga_sampling_measured=False,
        owned_errors=work["measurement"]["owned_errors"],
        projected_runtime_s=work["measurement"]["projected_runtime_s"],
        primitive_reference=work["primitive_reference"],
        validation_reference=work["validation_reference"],
        repository_health=work.get("repository_health"),
        methodology="Twelve separate paired RNG streams; all four unchanged Exp8881 mechanisms; randomized arm order; genuine empty-to-eight topology occupancy; complete parent restart costs; final feedback drain; six Bonferroni paired99-percent t intervals.",
        methodology_note="No quality evidence admitted; cost only. Exact oracle source-label checks remain paid. Inclusive phase timers cannot be subtracted to infer service speed. No FPGA sampling or batch-path claim.",
        flagged_adversarial=False,
        field_principles={},
    )
    value = normalize_artifact_for_template_write(value)
    value["field_principles"] = {
        k: "Measured producer metadata or authenticated operand; CPU constructed-query cost scope only."
        for k in value
    }
    value["field_principles"].update(
        adaptation_cost_ready_score="Complete paid work and exact restart parity; never scientific benefit.",
        numerical_training_calls="Current measured and uninterrupted control updates are explicit; no pretrained model load.",
        phase_costs="Inclusive nested phases; source_check includes joins, gradients, admission and atomic ack serialization.",
        quality_evidence_refs="Empty by design: scientific evaluation did not select implementation or justify a speed claim.",
        reproducibility_checksum="Canonical content hash plus cold reconstruction of controls and summaries.",
    )
    value["reproducibility_checksum"] = canonical_hash(value)
    return value


def replay(path: Path, *, semantic: bool = True) -> Json:
    """Reject rehashed aggregate changes and reconstruct actual uninterrupted decisions."""
    value = json.loads(path.read_bytes())
    checksum = value.pop("reproducibility_checksum")
    require(canonical_hash(value) == checksum, "artifact_checksum")
    for ref in (
        value["source_artifact_hashes"] + value["code_config_hashes"] + value["raw_shard_hashes"]
    ):
        require(sha256_file(Path(ref["path"])) == ref["sha256"], "operand_hash")
    require(not validate_current_work_receipt(value["current_work_receipt"]), "compute_receipt")
    work = json.loads(Path(value["primitive_reference"]["path"]).read_bytes())
    validation = json.loads(Path(value["validation_reference"]["path"]).read_bytes())
    for key in ("primitive_reference", "validation_reference"):
        require(sha256_file(Path(value[key]["path"])) == value[key]["sha256"], "envelope_hash")
    work["primitive_reference"] = value["primitive_reference"]
    require(
        canonical_hash(build(work, validation))
        == canonical_hash(dict(value, reproducibility_checksum=checksum)),
        "reduction_drift",
    )
    require(
        [dict((k, r[k]) for k in m.roster()[i]) for i, r in enumerate(value["rows"])] == m.roster(),
        "schedule_drift",
    )
    manifest = next(
        r
        for r in value["source_artifact_hashes"]
        if Path(r["path"]).name == "operand_manifest.json"
    )
    for ref in json.loads(Path(manifest["path"]).read_bytes())["references"]:
        require(sha256_file(Path(ref["path"])) == ref["sha256"], "original_operand_drift")
    with TemporaryDirectory(prefix="carnot8888-replay-") as directory:
        for index, row in enumerate(value["rows"]):
            if row["status"] != "complete":
                continue
            require(m.parity(row), "lifecycle_drift")
            if semantic:
                m.progress("semantic_replay_before", index, 48 - index)
                request = Path(directory) / str(index) / "request.json"
                head = work["preconditions"]["heads"][str(row["seed"])]
                atomic_json(request, dict(row, head=head))
                control = m.segment(request, 0, None)
                require(
                    control["state"] == row["segments"][-1]["state"]
                    and control["checkpoint24_state"] == row["segments"][0]["state"]
                    and control["releases"] == [r for s in row["segments"] for r in s["releases"]]
                    and control["occupancy"] == [r for s in row["segments"] for r in s["occupancy"]]
                    and control["predictions"]
                    == [p for s in row["segments"] for p in s["predictions"]],
                    "semantic_drift",
                )
                m.progress("semantic_replay_after", index + 1, 47 - index)
    return dict(passed=True, replay_passed=True, units=len(value["rows"]))


def terminal(candidate: Path, raw: Path) -> list[Json]:
    """Retain unchanged scanner sources, output and a real foreign-directory cold replay."""
    commands = [
        (
            "adversarial",
            [sys.executable, "-u", str(m.ROOT / "scripts/adversarial_verify.py"), str(candidate)],
        ),
        (
            "strict_rows",
            [
                sys.executable,
                "-u",
                str(m.ROOT / "scripts/verdict_row_consistency_lint.py"),
                "--strict",
                str(candidate),
            ],
        ),
        (
            "cold_replay",
            [
                "env",
                "-u",
                "PYTHONPATH",
                "--chdir=/tmp",
                sys.executable,
                "-u",
                str(m.ROOT / m.CLI),
                "--replay",
                str(candidate),
            ],
        ),
    ]
    return [stream(name, argv, raw / "terminal", deadline=600) for name, argv in commands]


def main(argv: list[str] | None = None) -> int:
    """Keep working evidence private until checked terminal bytes can be published."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", choices=["20261107"], default="20261107")
    parser.add_argument("--output", type=Path, default=m.ROOT / "results" / (m.NAME + ".json"))
    parser.add_argument("--private-output", type=Path)
    parser.add_argument("--sources", type=Path, default=m.ROOT / "results")
    parser.add_argument("--replay", "--cold-replay", type=Path)
    parser.add_argument("--worker", type=Path)
    parser.add_argument("--start", type=int, choices=[0, 24], default=0)
    args = parser.parse_args(argv)
    if args.worker:
        m.segment(args.worker, args.start, 24 if args.start == 0 else None, emit=True)
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
        8888,
        "Matched adaptation complete cost",
        "results/" + m.NAME + ".json",
        repo_root=m.ROOT,
        seed=m.SEED,
    )
    template.check_exclusion_manifest()
    code = (
        m.OWNED
        + m.TESTS
        + [
            "scripts/experiment_template.py",
            "python/carnot/reporting/current_work_receipt.py",
            "python/carnot/reporting/primary_publication.py",
            "scripts/adversarial_verify.py",
            "scripts/verdict_row_consistency_lint.py",
            "python/carnot/reporting/topology_memory_methods_8881.py",
            "python/carnot/memory/categorical_memory_8830.py",
            "python/carnot/memory/causal_memory_8802.py",
            "python/carnot/memory/transactional_constraint_memory.py",
        ]
    )
    invocation = dict(
        executable=sys.executable,
        arguments=argv if argv is not None else sys.argv[1:],
        cwd=os.getcwd(),
        pid=os.getpid(),
        executed_sources=[reference(m.ROOT / p) for p in code],
    )
    atomic_json(raw / "invocation.json", invocation)
    m.progress("preconditions_before", 0, 1)
    pre = preconditions(raw, args.sources)
    atomic_json(raw / "preconditions.json", pre)
    setup_end = time.monotonic_ns()
    m.progress("preconditions_after", 1, 0)
    with template.phase("measurement"):
        measured = measure(raw, pre)
    measured_end = time.monotonic_ns()
    m.progress("validation_before", 0, 1)
    with TemporaryDirectory(prefix="carnot8888-validation-") as directory:
        validation = (
            unqualified() if args.private_output else validate(raw / "validation", Path(directory))
        )
    atomic_json(raw / "validation.json", validation)
    ended = time.monotonic_ns()
    work: Json = dict(
        started=started,
        ended=ended,
        setup_ns=setup_end - started,
        invocation=invocation,
        preconditions=pre,
        measurement=measured,
        code_refs=invocation["executed_sources"],
        spans=[
            dict(phase="setup", start_s=0.0, end_s=(setup_end - started) / 1e9),
            dict(
                phase="measurement",
                start_s=(setup_end - started) / 1e9,
                end_s=(measured_end - started) / 1e9,
            ),
            dict(
                phase="validation",
                start_s=(measured_end - started) / 1e9,
                end_s=(ended - started) / 1e9,
            ),
        ],
        primitive_reference=reference(raw / "invocation.json"),
        validation_reference=reference(raw / "validation.json"),
        raw_refs=[],
    )
    health = Path("/tmp/carnot-8888-repository-health.log")
    if health.is_file() and not args.private_output:
        shutil.copyfile(health, raw / "repository_health.log")
        work["repository_health"] = dict(
            command=".venv/bin/pytest tests/python -q",
            scope="unrelated repository health; bounded240s",
            passed=False,
            exit_code=124,
            duration_limit_s=240,
            reason="reported failures before the bounded whole-suite deadline",
            **reference(raw / "repository_health.log"),
        )
    atomic_json(raw / "work.json", work)
    work["primitive_reference"] = reference(raw / "work.json")
    # The primitive's self-reference is supplied by its immutable envelope at replay.
    work["raw_refs"] = [reference(p) for p in sorted(raw.rglob("*")) if p.is_file()]
    atomic_json(raw / "envelope.json", work)
    work["primitive_reference"] = reference(raw / "envelope.json")
    value = build(work, validation)
    candidate = raw / "prospective.json"
    atomic_json(candidate, value)
    if not args.private_output:
        m.progress("terminal_validation_before", 0, 3)
        reports = terminal(candidate, raw)
        validation["receipts"] += reports
        validation["raw_references"] += [
            reference(Path(r[k + "_path"])) for r in reports for k in ("stdout", "stderr")
        ]
        atomic_json(raw / "terminal_validation.json", validation)
        work["validation_reference"] = reference(raw / "terminal_validation.json")
        work["ended"] = time.monotonic_ns()
        work["spans"].append(
            dict(
                phase="terminal_validation",
                start_s=(ended - started) / 1e9,
                end_s=(work["ended"] - started) / 1e9,
            )
        )
        atomic_json(raw / "terminal_work.json", work)
        work["primitive_reference"] = reference(raw / "terminal_work.json")
        value = build(work, validation)
        m.progress("terminal_validation_after", 3, 0)
    publish_primary(output, value, lambda p: replay(p, semantic=False))
    for path in raw.rglob("*"):
        if path.is_file():
            path.chmod(0o400)
    m.progress(
        "published_" + value["verdict_class"], value["completed_count"], value["censored_count"]
    )
    return 0
