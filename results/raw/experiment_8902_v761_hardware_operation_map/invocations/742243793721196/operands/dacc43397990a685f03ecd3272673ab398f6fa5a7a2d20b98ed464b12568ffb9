"""REQ-REPORT-8901: publish only authenticated complete cost measurements.

Cost readiness does not require a quality gain. Missing external evidence blocks
execution, while a failed control owned by this producer disqualifies the run.
"""

import argparse
import json
import os
from pathlib import Path
import shutil
import sys
from tempfile import TemporaryDirectory
import time
from typing import Any

from carnot.reporting import signed_adaptation_cost_8901 as m
from carnot.reporting import matched_adaptation_runner_8888 as inherited
from carnot.reporting import v761_branch_contract as authority
from carnot.reporting import topology_memory_methods_8881 as old
from carnot.reporting.causal_learning_audit_8810 import require
from carnot.reporting.current_work_receipt import (
    atomic_json,
    canonical_hash,
    sha256_file,
    validate_current_work_receipt,
)
from carnot.reporting.primary_publication import publish_primary
from carnot.reporting.selective_runner_8816 import unqualified
from carnot.reporting.v746_retained_evidence import reference
from carnot.reporting.v748_collection_contract import namespace
from scripts.experiment_template import ExperimentTemplate

Json = dict[str, Any]
SOURCES = [
    (
        "experiment_8892_v761_signed_feature_methods",
        "sha256:027641956a2d80c1466048a1f3eb150f968fd7cf274b835774834653e00d8461",
        "signed_methods_ready_score",
        "exp8892-signed-feature-methods",
    ),
    (
        "experiment_8888_v760_matched_adaptation_cost",
        "sha256:9f5517cbd96ed2456a5b6ee222bf74ae1bacf9be49d06c7b1a29465a3a7867a1",
        "adaptation_cost_ready_score",
        "exp8888-matched-adaptation-cost",
    ),
]


def runner() -> Any:
    """Keep historical runners unchanged while reusing their validation and publication helpers."""
    runtime = namespace(inherited, "cost8901_runner")
    runtime.m, runtime.SOURCES = m, SOURCES
    return runtime


def preconditions(raw: Path, sources: Path) -> Json:
    """Authenticate original producer assertions and every raw/config operand before timing."""
    raw.mkdir(parents=True, mode=0o700, exist_ok=True)
    raw.chmod(0o700)
    atomic_json(raw / "scratch_probe.json", dict(private=True))
    failures, refs, operands, heads = [], [], {}, {}
    for name, pin, field, task in SOURCES:
        source = sources / (name + ".json")
        digest = sha256_file(source) if source.is_file() else None
        check = "missing_file" if digest is None else "rejected_producer"
        observed = None
        try:
            value = json.loads(source.read_bytes())
            require(isinstance(value, dict), "producer_object")
            check = "missing_field"
            observed = value[field]
            check = "observed_zero" if observed == 0 else "rejected_producer"
            require(observed == 1, "upstream_assertion")
            check = "rejected_producer"
            require(
                digest == pin
                and value["task_id"] == task
                and value["experiment_id"] == int(name.split("_")[1])
                and value["required_checks_passed"] is True,
                "producer_identity",
            )
            target = raw / "inputs" / source.name
            target.parent.mkdir(exist_ok=True)
            shutil.copyfile(source, target)
            refs.extend([reference(source), reference(target)])
            for key in ("source_artifact_hashes", "code_config_hashes", "raw_shard_hashes"):
                for ref in value[key]:
                    operands[ref["path"], ref["sha256"]] = ref
            if field == "signed_methods_ready_score":
                fit = next(
                    r for r in value["raw_shard_hashes"] if Path(r["path"]).name == "fit.json"
                )
                heads = json.loads(Path(fit["path"]).read_bytes())["heads"]
        except (OSError, ValueError, KeyError, TypeError, StopIteration) as error:
            failures.append(
                dict(
                    check="missing_file"
                    if isinstance(error, OSError)
                    else "missing_field"
                    if isinstance(error, (KeyError, StopIteration))
                    else check,
                    upstream=task,
                    artifact_field=field,
                    expected=1,
                    observed=observed,
                    path=str(source),
                    hash=digest,
                    error=str(error),
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
    contract = authority.authority(m.ROOT)
    if not contract["ready"]:
        failures.append(
            dict(
                check="rejected_producer",
                upstream="V761",
                artifact_field="execution_authority",
                expected=True,
                observed=False,
                path=str(m.ROOT / authority.DESIGN),
                hash=None,
            )
        )
    for name in (authority.DESIGN, authority.ACTIVE, authority.STAGED, authority.METHODS):
        path = m.ROOT / name
        if path.is_file():
            target = raw / "authority" / path.name
            target.parent.mkdir(exist_ok=True)
            shutil.copyfile(path, target)
            refs.append(dict(reference(target), original_path=str(path)))
    resources = dict(
        private_scratch=oct(raw.stat().st_mode & 0o777) == "0o700",
        free_disk_bytes=shutil.disk_usage(raw).free,
        minimum_free_bytes=512 * 1024**2,
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
    """Pilot one whole paired stream, preserving every failure and unrun scheduled cell."""
    began = time.monotonic_ns()
    schedule = m.roster()
    atomic_json(raw / "schedule.json", dict(rows=schedule))
    rows, errors = [], []
    reason = "external_precondition" if not pre["external_ready"] else "owned_resource_precondition"
    admitted = pre["external_ready"] and pre["resources"]["passed"]
    projected = None
    for index, spec in enumerate(schedule):
        if admitted:
            m.progress("unit_before", index, 96 - index)
            head = pre["heads"][spec["feature_arm"]][str(spec["seed"])]
            row = m.execute(raw / f"unit-{index:02d}", spec, head)
            m.progress("unit_after", index + 1, 95 - index)
            if not m.parity(row):
                admitted, reason = False, "owned_lifecycle_control"
                errors.append(reason)
            if index == 7:
                projected = (time.monotonic_ns() - began) / 1e9 * 12 + 1200
                if projected >= 4200:
                    admitted, reason = False, "owned_runtime_projection"
                    errors.append(reason)
        else:
            row = dict(
                spec, status="censored", missing_reason=reason, metric=None, complete_ns=None
            )
        rows.append(row)
        if admitted and (time.monotonic_ns() - began) / 1e9 > 3000:
            admitted, reason = False, "owned_measurement_deadline"
            errors.append(reason)
    return dict(rows=rows, owned_errors=errors, projected_runtime_s=projected)


def build(work: Json, validation: Json) -> Json:
    """Reuse the complete-cost schema while separating administrative readiness from benefit."""
    value = dict(runner().build(work, validation))
    rows, pre = work["measurement"]["rows"], work["preconditions"]
    completed = [row for row in rows if row["status"] == "complete"]
    ready = bool(
        pre["external_ready"]
        and pre["resources"]["passed"]
        and value["required_checks_passed"]
        and len(completed) == 96
        and all(m.parity(row) for row in completed)
        and not work["measurement"]["owned_errors"]
    )
    verdict = "blocked" if not pre["external_ready"] else "null" if ready else "disqualified"
    value.update(
        experiment_id=8901,
        task_id=m.TASK,
        milestone="2026.11.761",
        honest_verdict=f"complete_{verdict}_signed_adaptation_cost",
        verdict_class=verdict,
        intended_count=96,
        independent_count=sum(
            all(
                any(
                    row["cluster_id"] == cluster
                    and row["feature_arm"] == feature
                    and row["arm"] == arm
                    for row in completed
                )
                for feature in ("old", "signed")
                for arm in old.ARMS
            )
            for cluster in {row["cluster_id"] for row in rows}
        ),
        adaptation_cost_ready_score=int(ready),
        quality_evidence_refs=work.get("quality_evidence_refs", []),
        durability_scope=dict(
            local_file_fsync_measured=True,
            directory_fsync_measured=True,
            power_loss_durability="unknown",
            network_cost="unknown",
            physical_device_cost="unknown",
            sampler_calls=0,
        ),
        methodology="Twelve fresh 48-query cost streams crossed with old/signed features and four causal arms. Randomized cell order. Whole-parent time includes launch, features, typed prediction, exact labels, SGD, syntax admission, serialization, file/directory fsync, query-24 restart and final pending-label drain. Each feedback receives an individual atomic state-plus-ack commit. Four simultaneous paired 99-percent intervals per cost endpoint.",
        methodology_note="Old deployment omits both extra feature calculations; the frozen masked nine-coordinate head remains unchanged. Phase timers are inclusive. Complete costs qualify without label benefit. No imported component fractions, device acceleration, natural generalization or sampler claim.",
    )
    value["acceptance_gates"].update(
        complete_cells=96, causal_arms=4, feature_variants=2, individual_feedback_commits=48
    )
    calls = sum(row["operation_counts"]["applied_updates"] for row in completed)
    cold = any(
        receipt["name"] == "cold_replay" and receipt["passed"] for receipt in validation["receipts"]
    )
    value["numerical_training_calls"] = calls * (3 if cold else 2)
    value["current_work_receipt"]["small_ebm_training"].update(
        numerical_training_calls=value["numerical_training_calls"],
        scope="measured children plus uninterrupted controls and successful terminal semantic replay",
    )
    value["per_stream_results"]["quality_evidence_refs"] = value["quality_evidence_refs"]
    value["field_principles"].update(
        {
            key: "Actual producer operands and complete durable CPU costs; readiness does not imply benefit."
            for key in value
            if key not in value["field_principles"]
        }
    )
    value["field_principles"].update(
        operation_counts="Per-cell measured comparisons, kernel update calls and individual atomic feedback commits are separate.",
        durability_scope="Local host storage barriers are measured; power-loss, remote network and physical device behavior remain unknown.",
        quality_evidence_refs="Optional byte-bound evidence; no speed or efficiency-benefit headline is authorized.",
        per_stream_results="Paired old/new complete stream costs; four-arm 99-percent interval family for each endpoint; n=12.",
    )
    value.pop("reproducibility_checksum")
    value["reproducibility_checksum"] = canonical_hash(value)
    return value


def replay(path: Path, *, semantic: bool = True) -> Json:
    """Rebuild summaries and actual states so a rehashed aggregate cannot pass validation."""
    value = json.loads(path.read_bytes())
    checksum = value.pop("reproducibility_checksum")
    require(canonical_hash(value) == checksum, "artifact_checksum")
    for ref in (
        value["source_artifact_hashes"] + value["code_config_hashes"] + value["raw_shard_hashes"]
    ):
        require(sha256_file(Path(ref["path"])) == ref["sha256"], "operand_hash")
    require(
        not validate_current_work_receipt(value["current_work_receipt"]),
        "compute_receipt",
    )
    for key in ("primitive_reference", "validation_reference"):
        require(sha256_file(Path(value[key]["path"])) == value[key]["sha256"], "envelope_hash")
    work = json.loads(Path(value["primitive_reference"]["path"]).read_bytes())
    validation = json.loads(Path(value["validation_reference"]["path"]).read_bytes())
    work["primitive_reference"] = value["primitive_reference"]
    require(
        build(work, validation) == dict(value, reproducibility_checksum=checksum), "reduction_drift"
    )
    require(
        [{k: row[k] for k in spec} for row, spec in zip(value["rows"], m.roster(), strict=True)]
        == m.roster(),
        "schedule_drift",
    )
    manifest = next(
        ref
        for ref in value["source_artifact_hashes"]
        if Path(ref["path"]).name == "operand_manifest.json"
    )
    for ref in json.loads(Path(manifest["path"]).read_bytes())["references"]:
        require(sha256_file(Path(ref["path"])) == ref["sha256"], "original_operand_drift")
    with TemporaryDirectory(prefix="carnot8901-replay-") as directory:
        for index, row in enumerate(value["rows"]):
            if row["status"] != "complete":
                continue
            require(m.parity(row), "lifecycle_drift")
            if semantic:
                m.progress("semantic_replay_before", index, 96 - index)
                request = Path(directory) / str(index) / "request.json"
                head = work["preconditions"]["heads"][row["feature_arm"]][str(row["seed"])]
                atomic_json(request, dict(row, head=head))
                control = m.segment(request, 0, None)
                require(
                    control["state"] == row["segments"][-1]["state"]
                    and control["checkpoint24_state"] == row["segments"][0]["state"]
                    and all(
                        control[key] == [r for s in row["segments"] for r in s[key]]
                        for key in ("predictions", "releases", "occupancy")
                    ),
                    "semantic_drift",
                )
                m.progress("semantic_replay_after", index + 1, 95 - index)
    return dict(passed=True, replay_passed=True, units=len(value["rows"]))


def main(argv: list[str] | None = None) -> int:
    """Keep checkpoints private until scanners and a real foreign-directory replay finish."""
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
        except (OSError, ValueError, KeyError, TypeError, StopIteration) as error:
            print("replay_rejected: " + str(error), flush=True)
            return 1
    output = (args.private_output or args.output).absolute()
    if args.private_output and output.resolve().is_relative_to(m.ROOT):
        parser.error("private outputs must remain outside checkout")
    started = time.monotonic_ns()
    raw = output.parent / "raw" / m.NAME / "invocations" / str(started)
    raw.mkdir(parents=True, mode=0o700)
    template = ExperimentTemplate(
        8901,
        "Signed-feature complete durable adaptation cost",
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
            "python/carnot/reporting/matched_adaptation_cost_8888.py",
            "python/carnot/reporting/matched_adaptation_runner_8888.py",
            "python/carnot/reporting/memory_size_cost_8848.py",
            "python/carnot/reporting/v748_collection_contract.py",
            "python/carnot/reporting/topology_memory_methods_8881.py",
            "python/carnot/memory/signed_memory_8892.py",
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
    with TemporaryDirectory(prefix="carnot8901-validation-") as directory:
        validation = (
            unqualified()
            if args.private_output
            else runner().validate(raw / "validation", Path(directory))
        )
    atomic_json(raw / "validation.json", validation)
    ended = time.monotonic_ns()
    m.progress("validation_after", 1, 0)
    work: Json = dict(
        started=started,
        ended=ended,
        setup_ns=setup_end - started,
        invocation=invocation,
        preconditions=pre,
        measurement=measured,
        code_refs=invocation["executed_sources"],
        raw_refs=[],
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
    )
    health = Path("/tmp/carnot-8901-repository-health.log")
    if health.is_file() and not args.private_output:
        shutil.copyfile(health, raw / "repository_health.log")
        work["repository_health"] = dict(
            command=".venv/bin/pytest tests/python -q",
            scope="whole-repository health, separate from owned validation",
            passed=False,
            exit_code=124,
            duration_limit_s=600,
            reason="bounded whole-suite run reported failures before its 600-second deadline",
            **reference(raw / "repository_health.log"),
        )
    first = Path("/tmp/carnot-8901-tests-first.log")
    if first.is_file():
        shutil.copyfile(first, raw / "tests-first.log")
    atomic_json(raw / "work.json", work)
    work["raw_refs"] = [reference(p) for p in sorted(raw.rglob("*")) if p.is_file()]
    atomic_json(raw / "envelope.json", work)
    work["primitive_reference"] = reference(raw / "envelope.json")
    candidate = raw / "prospective.json"
    atomic_json(candidate, build(work, validation))
    if not args.private_output:
        m.progress("terminal_validation_before", 0, 3)
        reports = runner().terminal(candidate, raw)
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
        m.progress("terminal_validation_after", 3, 0)
    value = build(work, validation)
    publish_primary(output, value, lambda p: replay(p, semantic=False))
    for path in raw.rglob("*"):
        if path.is_file():
            path.chmod(0o400)
    m.progress(
        "published_" + value["verdict_class"], value["completed_count"], value["censored_count"]
    )
    return 0
