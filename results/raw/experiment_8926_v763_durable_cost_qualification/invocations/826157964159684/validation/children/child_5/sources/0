"""REQ-REPORT-8926: qualify historical durable costs with typed comparison rows.

Original bytes remain immutable. Current work authenticates and reduces them;
it does not repeat the benchmark or measure the interpolated optimizer.
"""

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

from carnot.reporting import guarded_durable_cost_8914 as original
from carnot.reporting import guarded_durable_runner_8914 as prior
from carnot.reporting import topology_memory_runner_8881 as checks
from carnot.reporting import topology_causal_runner_8884 as previous
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
ROOT, SEED = original.ROOT, 7638926
NAME, TASK = "experiment_8926_v763_durable_cost_qualification", "exp8926-durable-cost-qualification"
SOURCE, PIN = (
    original.NAME,
    "sha256:e4bc58031745b08524ed21e5c3d7ddf84329d7d88aaf0be2ad9ceac14034d479",
)
CLI = "scripts/experiments/" + NAME + ".py"
OWNED = ["python/carnot/reporting/durable_cost_qualification_8926.py", CLI]
TESTS = ["tests/python/test_durable_cost_qualification_8926.py"]
ARCHIVE = (
    ROOT / "results/raw/experiment_8916_v762_capstone/invocations/1794110515600100474/operands"
)
GATES = dict(
    cells=48,
    feedback_per_cell=48,
    restart_at=24,
    owned_statement_percent=100,
    exclusions=0,
    historical_guarded_only=True,
    speed_quality_recommendation_score=0,
)
require = original.require


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Expose phase and census counts while the fixed deadline continues."""
    print(f"[exp8926] phase={phase} completed={completed} pending={pending}", flush=True)


def preconditions(raw: Path, sources: Path) -> Json:
    """Retain authenticated original bytes, including archived original source code."""
    path = sources / (SOURCE + ".json")
    diagnostic = dict(
        check="missing_file",
        upstream=original.TASK,
        artifact_field="producer_bytes",
        expected=PIN,
        observed=None,
        path=str(path),
        hash=None,
    )
    manifest: list[Json] = []
    pre: Json = dict(
        external_ready=False,
        diagnostics=[],
        references=[],
        original_receipt_manifest=manifest,
        resources=dict(passed=True, total_budget_s=4200, hard_cap_s=4800),
    )
    progress("authentication_before")
    try:
        diagnostic["hash"] = diagnostic["observed"] = sha256_file(path)
        diagnostic["check"] = "rejected_producer"
        require(diagnostic["hash"] == PIN, "original_primary_bytes")
        value = json.loads(path.read_bytes())
        checksum = value.pop("reproducibility_checksum")
        require(canonical_hash(value) == checksum, "original_checksum")
        require(
            (value["experiment_id"], value["task_id"], value["completed_count"])
            == (8914, original.TASK, 48),
            "original_identity_and_census",
        )
        strict = [r for r in value["validation_receipts"] if r["name"] == "strict_rows"]
        require(
            len(strict) == 1
            and strict[0]["actual_exit"] == 1
            and all(r["passed"] for r in value["validation_receipts"] if r["name"] != "strict_rows")
            and prior.qualified(
                dict(
                    coverage=value["owned_statement_coverage"],
                    union=value["test_union"],
                    receipts=[],
                )
            ),
            "original_owned_controls",
        )
        refs = (
            [reference(path)]
            + value["source_artifact_hashes"]
            + value["code_config_hashes"]
            + value["raw_shard_hashes"]
        )
        unique = {r["path"]: r for r in refs}
        old_raw = Path(value["raw_directory"])
        for i, ref in enumerate(unique.values()):
            source = Path(ref["path"])
            diagnostic.update(
                path=str(source),
                artifact_field="original_operand",
                expected=ref["sha256"],
                observed=None,
                hash=None,
                check="missing_file",
            )
            if not source.is_file() or sha256_file(source) != ref["sha256"]:
                source = ARCHIVE / ref["sha256"].split(":")[1] / source.name
            diagnostic["hash"] = diagnostic["observed"] = sha256_file(source)
            diagnostic["check"] = "rejected_producer"
            require(diagnostic["observed"] == ref["sha256"], "original_operand_bytes")
            target = (
                raw / "original_raw" / Path(ref["path"]).relative_to(old_raw)
                if Path(ref["path"]).is_relative_to(old_raw)
                else raw / "operands" / ref["sha256"].split(":")[1] / source.name
            )
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists():
                require(sha256_file(target) == ref["sha256"], "immutable_original_bytes")
            else:
                shutil.copyfile(source, target)
            manifest.append(dict(original=ref, retained=reference(target)))
            if i % 100 == 0:
                progress("authentication", i, len(unique) - i)
        pre.update(
            external_ready=True,
            references=[r["retained"] for r in manifest],
            original_primary=manifest[0]["retained"],
            retained_raw_directory=str(raw / "original_raw"),
        )
    except (OSError, ValueError, KeyError, TypeError) as error:
        pre["diagnostics"].append(
            dict(
                diagnostic,
                check="missing_file"
                if isinstance(error, OSError)
                else "missing_field"
                if isinstance(error, KeyError)
                else "rejected_producer",
                error=str(error),
            )
        )
    progress("authentication_after", len(manifest), 0)
    return pre


def reduce_retained(raw: Path, pre: Json) -> Json:
    """Reuse original durability assertions and preserve every measured operand."""
    value = json.loads(Path(pre["original_primary"]["path"]).read_bytes())
    retained_raw = Path(pre["retained_raw_directory"])
    result = original.recover(retained_raw, value["censor_reason"])
    for key, expected in result.items():
        if key != "adaptation_cost_ready_score":
            require(expected == value[key], "original_reduction_drift:" + key)
    process = []
    for i, row in enumerate(result["rows"]):
        first, last = row["segments"]
        control = json.loads(
            (retained_raw / row["cluster_id"] / row["arm"] / "control/segment-48.json").read_bytes()
        )
        strip: Callable[[list[Json]], list[Json]] = lambda rows: [
            {k: v for k, v in r.items() if k != "duration_ns"} for r in rows
        ]
        require(
            first["state"] == last["initial_state"]
            and first["pid"] != last["pid"]
            and row["restart"]["feedback_cursor"] == 24
            and last["state"] == control["state"]
            and strip(first["predictions"] + last["predictions"]) == strip(control["predictions"]),
            "original_restart_control",
        )
        payload = dict(
            unit_id=row["unit_id"], commands=row["commands"], control_receipt=row["control_receipt"]
        )
        path = raw / "process_receipts" / (canonical_hash(payload).split(":")[1] + ".json")
        if path.exists():
            require(json.loads(path.read_bytes()) == payload, "immutable_process_receipt")
        else:
            atomic_json(path, payload)
        process.append(
            dict(unit_id=row["unit_id"], **reference(path), scope="historical_process_validation")
        )
        progress("retained_cell", i + 1, 47 - i)
    comparisons = [
        dict(
            r,
            control_duration_ns=s["control_duration_ns"],
            control_numerical_training_calls=s["control_numerical_training_calls"],
        )
        for r, s in zip(result["service_cost_rows"], result["rows"], strict=True)
    ]
    result.update(
        rows=comparisons,
        comparison_rows=comparisons,
        process_receipt_references=process,
        historical_numerical_training_calls=result["numerical_training_calls"],
        numerical_training_calls=0,
        reduction_identity=dict(
            original_primary=PIN,
            original_code_config_hashes=value["code_config_hashes"],
            original_reduction_hash=canonical_hash(
                {k: value[k] for k in result if k in value and k != "adaptation_cost_ready_score"}
            ),
        ),
        cost_scope=dict(
            timings="historical_exp8914_guarded_service",
            fresh_timings=False,
            optimizer="original_guarded_sgd",
            denominator="complete_ns: whole service wall time including cold startup and restart",
            inclusive_timers_additive=False,
            interpolated_optimizer_cost_measured=False,
        ),
        benefit_gate=value["benefit_gate"],
    )
    return result


def check_comparisons(value: Json, expected: Json) -> None:
    """Reject row, arm, timer or receipt-link drift against the original reduction."""
    for key in ("rows", "comparison_rows", "process_receipt_references"):
        require(value[key] == expected[key], "comparison_boundary:" + key)
    require(
        len(value["rows"]) == len(value["process_receipt_references"]) == 48, "cell_receipt_census"
    )
    for row, ref in zip(value["rows"], value["process_receipt_references"], strict=True):
        require(
            row["unit_id"] == ref["unit_id"] and sha256_file(Path(ref["path"])) == ref["sha256"],
            "process_receipt_link",
        )


def validation_child(name: str, argv: list[str], logs: Path, **kwargs: Any) -> Json:
    """Run original durability tests against authentic original code in a private mount view."""
    binding: Json = {}
    if name.startswith("child_") and any(
        "test_guarded_durable_cost_8914.py::" in arg for arg in argv
    ):
        digest = "379c78d6c15c75ca6e4410bf1b08a8c65b9276f27c562cd23cd6b8faf1dd6033"
        source = ARCHIVE / digest / "guarded_memory_8905.py"
        require(sha256_file(source) == "sha256:" + digest, "historical_validation_source")
        target = ROOT / "python/carnot/memory/guarded_memory_8905.py"
        binding = dict(original_source=reference(source), mapped_path=str(target), read_only=True)
        argv = [
            "bwrap",
            "--bind",
            "/",
            "/",
            "--dev-bind",
            "/dev",
            "/dev",
            "--ro-bind",
            str(source),
            str(target),
            *argv,
        ]
    progress("validation_child_before:" + name, 0, 1)
    receipt = stream(name, argv, logs, **kwargs)
    receipt["historical_source_view"] = binding
    progress("validation_child_after:" + name, 1, 0)
    return receipt


def validate(raw: Path, private: Path) -> Json:
    """Measure owned unit/child coverage and run unchanged durability and private CLI controls."""
    from types import SimpleNamespace

    checker = namespace(checks, "durable8926_validation")
    checker.m = SimpleNamespace(
        ROOT=ROOT,
        OWNED=OWNED,
        TESTS=TESTS + original.TESTS + ["tests/python/test_verdict_row_consistency_lint.py"],
    )
    checker.stream = validation_child
    result = dict(checker.validate(raw, private))
    result["receipts"].append(previous.e2e018(raw / "e2e018", private / "e2e018"))
    result["raw_references"] += [reference(p) for p in (raw / "e2e018").rglob("*") if p.is_file()]
    return result


def terminal(candidate: Path, raw: Path) -> list[Json]:
    """Run unchanged artifact scanners and foreign-directory replay of the actual CLI."""
    checker = namespace(prior, "durable8926_terminal")
    checker.m = sys.modules[__name__]
    return list(checker.terminal(candidate, raw))


def seal(value: Json, started: int, validation: Json) -> None:
    """Record current wall time and readiness independently from historical scientific benefit."""
    checker = namespace(prior, "durable8926_qualification")
    checker.m = sys.modules[__name__]
    ready = (
        value["completed_count"] == 48
        and checker.qualified(validation)
        and value["preconditions_checked"]["external_ready"]
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
        inference_substrate="aggregation_from_upstream_artifacts",
        inference_substrate_class="no_model_load",
        execution_venue="host",
        inference_substrate_details=dict(
            scope="Historical cost authentication and reduction; no new benchmark"
        ),
        started_monotonic_ns=started,
        ended_monotonic_ns=time.monotonic_ns(),
        phase_spans=value["phase_spans"],
        small_ebm_training=dict(performed=False, numerical_training_calls=0),
    )
    value.update(
        adaptation_cost_ready_score=int(ready),
        required_checks_passed=bool(ready),
        verdict_class=verdict,
        honest_verdict="complete_" + verdict + "_durable_cost_qualification",
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
        k: "Invocation identity, authenticated original bytes or historical measured operands; no new scientific benefit."
        for k in value
    }
    value["field_principles"].update(
        duration_s="Complete current qualification wall time, including owned checks; never a fresh benchmark timing.",
        comparison_rows="Original arms and complete wall costs, with scientific uninterrupted control metrics; execution status receipts are separate.",
        process_receipt_references="Content-addressed exact original process commands and receipts, bound one per cell.",
        inclusive_timer_rows="Nested inclusive timers overlap and cannot be summed as a partition.",
        adaptation_cost_ready_score="Qualified historical original guarded-service costs only; no interpolated optimizer timing.",
        numerical_training_calls="Zero current training; historical work and validation child work have separate receipts.",
        reproducibility_checksum="Canonical current artifact bytes plus cold original-operand reconstruction.",
        speed_quality_recommendation_score="Always zero: authenticated Exp8911 benefit is zero and no new optimizer cost was measured.",
    )
    value["reproducibility_checksum"] = canonical_hash(value)


def replay(path: Path) -> Json:
    """Cold-reduce original evidence and reject rehashed row, provenance or readiness changes."""
    value = json.loads(path.read_bytes())
    checksum = value.pop("reproducibility_checksum")
    require(canonical_hash(value) == checksum, "artifact_checksum")
    require(
        (value["experiment_id"], value["task_id"], value["milestone"], value["run_date"])
        == (8926, TASK, "2026.11.763", "20261108"),
        "producer_identity",
    )
    for i, ref in enumerate(
        value["source_artifact_hashes"] + value["code_config_hashes"] + value["raw_shard_hashes"]
    ):
        require(sha256_file(Path(ref["path"])) == ref["sha256"], "operand_hash")
        if i % 100 == 0:
            progress("replay_authentication", i, 0)
    pre = value["preconditions_checked"]
    if pre["external_ready"]:
        require(sha256_file(Path(pre["original_primary"]["path"])) == PIN, "original_primary_pin")
        for entry in value["original_receipt_manifest"]:
            require(
                entry["original"]["sha256"] == entry["retained"]["sha256"],
                "original_manifest_drift",
            )
        expected = reduce_retained(Path(value["raw_directory"]), pre)
        check_comparisons(value, expected)
        for key, result in expected.items():
            if key != "adaptation_cost_ready_score":
                require(value[key] == result, "reduction_drift:" + key)
    else:
        require(
            value["gate_check_summary"] and value["completed_count"] == 0, "blocked_gate_summary"
        )
    require(
        value["acceptance_gates"] == GATES
        and value["speed_quality_recommendation_score"] == 0
        and value["verifier_is_oracle"] is True
        and value["independent_generalization_score"] == 0
        and value["numerical_training_calls"] == 0
        and value["MODEL_SPECS"] == [],
        "scope_boundary",
    )
    receipt = value["current_work_receipt"]
    require(
        not validate_current_work_receipt(receipt)
        and value["duration_s"] == receipt["duration_s"]
        and value["model_invocation_counts"] == receipt["invocation_counts"],
        "current_compute_receipt",
    )
    checker = namespace(prior, "durable8926_replay_qualification")
    checker.m = sys.modules[__name__]
    ready = bool(
        pre["external_ready"]
        and value["completed_count"] == 48
        and checker.qualified(
            dict(
                coverage=value["owned_statement_coverage"],
                union=value["test_union"],
                receipts=value["validation_receipts"],
            )
        )
    )
    verdict = "blocked" if not pre["external_ready"] else "null" if ready else "disqualified"
    require(
        type(value["adaptation_cost_ready_score"]) is int
        and value["adaptation_cost_ready_score"] == int(ready)
        and value["required_checks_passed"] == ready
        and value["verdict_class"] == verdict
        and value["honest_verdict"] == "complete_" + verdict + "_durable_cost_qualification",
        "terminal_semantics",
    )
    return dict(passed=True, status="replay_passed", sha256=sha256_file(path))


def main(argv: list[str] | None = None) -> int:
    """Qualify retained work and publish only a terminal artifact checked by actual replay."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", choices=["20261108"], default="20261108")
    parser.add_argument("--output", type=Path, default=ROOT / "results" / (NAME + ".json"))
    parser.add_argument("--private-output", type=Path)
    parser.add_argument("--sources", type=Path, default=ROOT / "results")
    parser.add_argument("--replay", "--cold-replay", type=Path)
    args = parser.parse_args(argv)
    if args.replay:
        try:
            print(json.dumps(replay(args.replay)), flush=True)
            return 0
        except (OSError, ValueError, KeyError, TypeError) as error:
            print("replay_rejected: " + str(error), flush=True)
            return 1
    output = (args.private_output or args.output).absolute()
    if args.private_output and output.resolve().is_relative_to(ROOT):
        parser.error("private outputs must remain outside checkout")
    started = time.monotonic_ns()
    raw = output.parent / "raw" / NAME / "invocations" / str(started)
    raw.mkdir(parents=True, mode=0o700)
    template = ExperimentTemplate(
        8926,
        "Retained durable cost qualification",
        "results/" + NAME + ".json",
        repo_root=ROOT,
        seed=SEED,
    )
    template.check_exclusion_manifest()
    atomic_json(
        raw / "invocation.json",
        dict(
            executable=sys.executable,
            arguments=argv if argv is not None else sys.argv[1:],
            cwd=os.getcwd(),
            pid=os.getpid(),
            executed_sources=[reference(ROOT / p) for p in OWNED],
        ),
    )
    with template.phase("authenticate_and_reduce"):
        pre = preconditions(raw, args.sources)
        result = (
            reduce_retained(raw, pre)
            if pre["external_ready"]
            else dict(
                rows=[],
                comparison_rows=[],
                process_receipt_references=[],
                intended_count=48,
                completed_count=0,
                failed_count=0,
                censored_count=48,
                excluded_count=0,
                independent_count=0,
                numerical_training_calls=0,
                reduction_identity=None,
                cost_scope=dict(timings="blocked_no_run", fresh_timings=False),
            )
        )
    atomic_json(raw / "preconditions.json", pre)
    measured = time.monotonic_ns()
    progress("owned_validation_before", 0, 1)
    with TemporaryDirectory(prefix="carnot8926-validation-", dir="/tmp") as directory:
        validation = (
            unqualified() if args.private_output else validate(raw / "validation", Path(directory))
        )
    progress("owned_validation_after", 1, 0)
    health_path = Path("/tmp/carnot8926-health/receipt.json")
    health: Json = dict(passed=False, scope="unrelated_repository_health", observed="unavailable")
    if health_path.is_file():
        health = dict(json.loads(health_path.read_bytes()), scope="unrelated_repository_health")
        for channel in ("stdout", "stderr"):
            target = raw / "repository_health" / (channel + ".log")
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(health[channel + "_path"], target)
            health[channel + "_path"] = str(target)
        atomic_json(raw / "repository_health" / "receipt.json", health)
    for name in ("tests-first", "row-lint-callers"):
        source = Path("/tmp/carnot8926-" + name + (".log" if name == "tests-first" else ".json"))
        if source.is_file():
            shutil.copyfile(source, raw / source.name)
    value = dict(
        result,
        experiment_id=8926,
        task_id=TASK,
        milestone="2026.11.763",
        run_date=args.date,
        execution_venue="host",
        MODEL_SPECS=[],
        verifier_is_oracle=True,
        exposure_scope="Finite constructed typed truth; retained guarded service costs only.",
        independent_generalization_score=0,
        preconditions_checked=pre,
        random_seed=SEED,
        source_artifact_hashes=pre["references"],
        original_receipt_manifest=pre["original_receipt_manifest"],
        code_config_hashes=[
            reference(ROOT / p)
            for p in OWNED
            + TESTS
            + original.OWNED
            + [
                "scripts/experiment_template.py",
                "python/carnot/reporting/current_work_receipt.py",
                "python/carnot/reporting/primary_publication.py",
                "scripts/adversarial_verify.py",
                "scripts/verdict_row_consistency_lint.py",
            ]
        ],
        raw_shard_hashes=[reference(p) for p in sorted(raw.rglob("*")) if p.is_file()]
        + validation["raw_references"],
        raw_directory=str(raw),
        cited_upstream_artifacts=[dict(experiment_id=8914, sha256=PIN, task_id=original.TASK)],
        gate_check_summary=pre["diagnostics"],
        acceptance_gates=GATES,
        speed_quality_recommendation_score=0,
        strict_row_checks=dict(
            command="verdict_row_consistency_lint.py --strict",
            process_receipts_outside_comparisons=True,
        ),
        repository_health=health,
        flagged_adversarial=False,
        phase_spans=[
            dict(phase="authenticate_and_reduce", start_s=0.0, end_s=(measured - started) / 1e9),
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
        progress("terminal_validation_before", 0, 3)
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
        progress("terminal_validation_after", 3, 0)
    require(time.monotonic_ns() - started < 4800 * 10**9, "hard_cap")
    publish_primary(output, value, replay)
    progress(
        "published_" + value["verdict_class"], value["completed_count"], value["censored_count"]
    )
    return 0
