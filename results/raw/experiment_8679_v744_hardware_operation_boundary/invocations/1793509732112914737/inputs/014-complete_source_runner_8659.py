"""REQ-VERIFY-8659: bound measurements, seal primitives, then publish checked bytes."""

from __future__ import annotations

import argparse
from contextlib import redirect_stderr, redirect_stdout
import multiprocessing
import traceback
from datetime import datetime
import json
import os
from pathlib import Path
import shutil
import sys
from tempfile import mkdtemp
import time
from typing import Any

from carnot.reporting import complete_source_8659 as c
from carnot.reporting.current_work_receipt import ZERO_INVOCATION_COUNTS
from carnot.reporting.primary_publication import publish_primary
from carnot.reporting.streaming_execution_runner_8656 import child_suite, CONSUMERS
from carnot.reporting.streaming_custody_8642 import combine
from carnot.reporting.v709_execution import child
from carnot.reporting.v738_contract_runner import date_value, exact_union

Json = dict[str, Any]


def worker_job(
    source: Path, arm: str, chunk: int, primitive: Path, output: Path, logs: Path
) -> None:
    """Open only the public descriptor; inherited imports do not supply labels to extraction."""
    logs.mkdir(parents=True, exist_ok=True)
    with (logs / "worker.stdout").open("w") as out, (logs / "worker.stderr").open("w") as err:
        with redirect_stdout(out), redirect_stderr(err):
            try:
                c.progress("before_extraction")
                c.atomic_json(output, c.response(source, arm, chunk, primitive))
                c.progress("after_extraction", 1, 0)
            except BaseException:
                traceback.print_exc()
                raise


def fork_response(
    name: str,
    source: Path,
    arm: str,
    chunk: int,
    primitive: Path,
    output: Path,
    logs: Path,
    *,
    deadline: float = 40,
) -> Json:
    """A new PID owns each measurement; preloaded imports avoid charging startup1520 times."""
    logs.mkdir(parents=True, exist_ok=True)
    stdout, stderr = logs / "worker.stdout", logs / "worker.stderr"
    stdout.write_text("")
    stderr.write_text("")
    start, wall = time.monotonic_ns(), time.time_ns()
    c.progress("before_fork_" + name, 0, 1)
    process = multiprocessing.get_context("fork").Process(
        target=worker_job, args=(source, arm, chunk, primitive, output, logs)
    )
    process.start()
    timed_out = False
    while process.is_alive():
        remaining = deadline - (time.monotonic_ns() - start) / 1e9
        if remaining <= 0:
            timed_out = True
            process.kill()
            break
        process.join(timeout=min(30, remaining))
        if process.is_alive():
            c.progress("waiting_fork_" + name, 0, 1)
    process.join()
    end = time.monotonic_ns()
    receipt = dict(
        name=name,
        scope="owned",
        spawn_method="fork_after_imports",
        pid=process.pid,
        parent_pid=os.getpid(),
        parent_command=Path("/proc/self/cmdline").read_bytes().replace(b"\0", b" ").decode(),
        operation=dict(
            callable="complete_source_runner_8659.worker_job",
            source=str(source),
            arm=arm,
            chunk=chunk,
            primitive=str(primitive),
            output=str(output),
        ),
        exit_code=process.exitcode,
        expected_exit=0,
        passed=process.exitcode == 0 and not timed_out,
        timed_out=timed_out,
        started_monotonic_ns=start,
        ended_monotonic_ns=end,
        started_wall_ns=wall,
        duration_s=(end - start) / 1e9,
        deadline_s=deadline,
        stdout_path=str(stdout),
        stderr_path=str(stderr),
        stdout_sha256=c.sha256_file(stdout),
        stderr_sha256=c.sha256_file(stderr),
    )
    c.atomic_json(logs / "receipt.json", receipt)
    c.progress("after_fork_" + name, 1, 0)
    return receipt


def measure(root: Path, raw: Path, date: str, *, measurement_seconds: float = 3600) -> Json:
    """Extract without labels in bounded children, preserving every roster slot on timeout."""
    c.progress("preconditions")
    start = time.monotonic_ns()
    raw.mkdir(parents=True, exist_ok=True)
    scratch = Path(mkdtemp(prefix="exp8659-", dir="/var/tmp"))
    scratch.chmod(0o700)
    probe = scratch / "probe"
    probe.write_bytes(b"writable disk scratch")
    probe.unlink()
    mount = max(
        (
            v.split()
            for v in Path("/proc/mounts").read_text().splitlines()
            if scratch.is_relative_to(Path(v.split()[1]))
        ),
        key=lambda v: len(v[1]),
    )
    preconditions = dict(
        memory=Path("/proc/meminfo").read_text(),
        scratch=str(scratch),
        scratch_free_bytes=shutil.disk_usage(scratch).free,
        scratch_filesystem=mount[2],
        writable_disk_backed_scratch=mount[2] not in {"tmpfs", "ramfs"},
        tools=[c.reference(c.ROOT / ".venv/bin" / n) for n in ("python", "pytest", "ruff", "mypy")],
        optional_hardware_required=False,
        python=sys.executable,
    )
    try:
        data = c.inputs(root)
    except (OSError, ValueError, KeyError, TypeError) as error:
        data = dict(
            public=[],
            questions={},
            baseline=[],
            gates=[
                dict(
                    upstream="original_release",
                    check="input_authentication",
                    path=str(root),
                    sha256=None,
                    field="authenticated_inputs",
                    operator="==",
                    expected=True,
                    observed=str(error),
                    passed=False,
                    missing_reason=type(error).__name__,
                )
            ],
        )
    responses, receipts, refs = [], [], []
    for index, row in enumerate(data["public"]):
        c.progress("response", index, len(data["public"]) - index)
        source = scratch / f"{index}.json"
        c.atomic_json(source, row)
        result: Json = dict(
            opaque_id=row["opaque_id"],
            original=None,
            streaming=None,
            invariance=[],
            costs=[],
            references=[],
        )
        schedule = [
            (repeat, arm, 32)
            for repeat in range(3)
            for arm in (
                ("original_capped", "streaming_complete")
                if repeat != 1
                else ("streaming_complete", "original_capped")
            )
        ]
        schedule += [(3, "streaming_complete", 1), (3, "streaming_complete", 128)]
        for repeat, arm, chunk in schedule:
            remaining = measurement_seconds - (time.monotonic_ns() - start) / 1e9
            if remaining < 35:
                break
            label = f"r{index}-repeat{repeat}-{arm}-chunk{chunk}"
            output, primitive = raw / (label + ".json"), raw / (label + ".jsonl")
            receipt = fork_response(
                label,
                source,
                arm,
                chunk,
                primitive,
                output,
                raw / "measurement_checks" / label,
                deadline=min(40, remaining),
            )
            receipts.append(receipt)
            if not receipt["passed"]:
                continue
            observed = json.loads(output.read_bytes())
            if observed["window_reference"]:
                observed["window_reference"] = c.freeze(
                    Path(observed["window_reference"]["path"]), raw
                )
                c.atomic_json(output, observed)
            ref = c.freeze(output, raw)
            refs.append(ref)
            result["references"].append(dict(repeat=repeat, arm=arm, chunk=chunk, reference=ref))
            if repeat < 3:
                result["costs"].append(
                    dict(
                        observed["cost"],
                        repeat=repeat,
                        order_in_pair=(0 if arm == schedule[repeat * 2][1] else 1),
                        receipt=receipt,
                    )
                )
                if repeat == 0:
                    result["original" if arm == "original_capped" else "streaming"] = observed
            else:
                result["invariance"].append(observed)
        responses.append(result)
        c.progress("response_complete", index + 1, len(data["public"]) - index - 1)
    report = c.reduce(dict(data, responses=responses))
    seal = raw / "feature_seal.json"
    c.atomic_json(
        seal,
        dict(
            features=c.canonical_hash(report),
            sealed_monotonic_ns=time.monotonic_ns(),
            response_refs=refs,
        ),
    )
    c.progress("labels_after_feature_seal", len(responses), 0)
    label_metrics = c.label_report(data, report)
    end = time.monotonic_ns()
    work = dict(
        root=str(root),
        raw=str(raw),
        run_date=date,
        report=dict(report, **label_metrics),
        response_records=[
            dict(opaque_id=v["opaque_id"], references=v["references"]) for v in responses
        ],
        source_refs=[
            dict(origin=str(root / p), reference=c.freeze(root / p, raw))
            for p in c.PINS
            if (root / p).is_file()
        ],
        code_refs=[c.reference(c.ROOT / p) for p in [*c.OWNED, c.TEST]],
        gate_check_summary=data["gates"],
        measurement_receipts=receipts,
        preconditions_checked=preconditions,
        duration_s=(end - start) / 1e9,
        phase_spans=[
            dict(
                phase="measurement",
                started_monotonic_ns=start,
                ended_monotonic_ns=end,
                duration_s=(end - start) / 1e9,
            )
        ],
        feature_seal_reference=c.freeze(seal, raw),
        raw_refs=refs,
        roster_reference=data.get("method", {}).get("source_roster_reference"),
        source_protocol_reference=data.get("method", {}).get("streaming_extraction_protocol"),
        target_model=dict(
            data.get("method", {}).get("target_model", {}), evaluated_this_invocation=False
        ),
        historical_model_provenance=data.get("release", {}).get("historical_model_provenance", []),
    )
    c.atomic_json(raw / "measurement.json", work)
    c.progress("measurement_complete", len(responses), 0)
    return work


def build(work: Json, validation: Json, output: Path) -> Json:
    """Engineering readiness requires every control; exposed labels grant no scientific credit."""
    owned = bool(validation["receipts"]) and all(
        v["passed"] for v in validation["receipts"] + work["measurement_receipts"]
    )
    owned = owned and validation["coverage"]["passed"]
    report = work["report"]
    per = report["per_response_results"]
    acceptance = dict(
        owned_checks=owned,
        qualified_inputs=bool(per) and all(v["passed"] for v in work["gate_check_summary"]),
        common_domain_parity=report["common_parity"],
        chunk_invariance=report["chunk_parity"],
        complete_window_custody=report["complete_window_custody"],
        resource_bounds=report["resource_bounds"],
        recovered_response=report["recovered_count"] > 0,
        paired_costs=len(report["complete_extraction_cost_rows"]) == 190 * 6,
        complete_roster=len(per) == 190,
    )
    ready = int(all(acceptance.values()))
    verdict = (
        "disqualified"
        if not owned
        else "positive"
        if ready
        else "blocked"
        if not acceptance["qualified_inputs"]
        else "null"
    )
    rows = []
    for index in range(190):
        item = per[index] if index < len(per) else {}
        for arm in ("original_capped", "streaming_complete"):
            observed = item.get(
                "original_features" if arm == "original_capped" else "streaming_features"
            )
            rows.append(
                dict(
                    slot=index,
                    opaque_id=item.get("opaque_id"),
                    question_hash=item.get("question_hash"),
                    arm=arm,
                    status="completed" if observed is not None else "censored",
                    metric=int(observed is not None and observed["values"] is not None),
                    missing_reason=observed["abstention"]
                    if observed is not None
                    else "upstream_blocked"
                    if not per
                    else "measurement_deadline",
                )
            )
    gates = list(work["gate_check_summary"])
    gates += [
        dict(
            upstream=TASK,
            check=k,
            path=str(Path(work["raw"]) / "measurement.json"),
            sha256=c.sha256_file(Path(work["raw"]) / "measurement.json"),
            field=k,
            operator="==",
            expected=True,
            observed=v,
            passed=v,
            missing_reason="observed_zero",
        )
        for k, v in acceptance.items()
        if not v
    ]
    raw = Path(work["raw"])
    value = dict(
        report,
        experiment_id=8659,
        task_id=c.TASK,
        milestone="2026.10.743",
        run_date=work["run_date"],
        honest_verdict=f"complete_{verdict}_v743_complete_source_comparison",
        verdict_class=verdict,
        source_comparison_ready_score=ready,
        coverage_benefit_score=int(ready and report["recovered_count"] > 0),
        branch_gate_values=dict(
            source_comparison_ready_score=ready,
            coverage_benefit_score=int(ready and report["recovered_count"] > 0),
        ),
        gate_check_summary=gates,
        inference_substrate="no_model_load",
        inference_substrate_class="no_model_load",
        MODEL_SPECS=[],
        model_invocation_counts=dict(ZERO_INVOCATION_COUNTS),
        current_llm_invocations=0,
        numerical_head_invocations=0,
        target_model=work["target_model"],
        historical_model_provenance=work["historical_model_provenance"],
        rows=rows,
        intended_count=380,
        completed_count=sum(v["status"] == "completed" for v in rows),
        failed_count=0,
        censored_count=sum(v["status"] == "censored" for v in rows),
        excluded_count=0,
        independent_count=128,
        sample_size_budget=dict(
            responses=190,
            question_clusters=128,
            arms=2,
            paired_repeats=3,
            siblings_are_repeated_measures=True,
        ),
        verifier_is_oracle=False,
        exposure_scope="fixed released exposed Trivia+ roster; historical overlap unknown",
        independent_generalization_score=0,
        generalized_learning_benefit_score=0,
        required_checks_passed=owned,
        flagged_adversarial=bool(
            validation.get("adversarial_findings", {}).get("flagged_count", 0)
        ),
        acceptance_gates=acceptance,
        validation_receipts=validation["receipts"],
        validation_evidence=validation,
        terminal_validation_sidecar_path=str(raw / "terminal_validation.json"),
        adversarial_findings=validation.get("adversarial_findings", {}),
        preconditions_checked=work["preconditions_checked"],
        duration_s=work["duration_s"],
        phase_spans=work["phase_spans"],
        random_seed=c.SEED,
        source_artifact_hashes=work["source_refs"],
        code_config_hashes=work["code_refs"],
        raw_shard_hashes=work["raw_refs"],
        cited_upstream_artifacts=work["source_refs"],
        roster_reference=work["roster_reference"],
        source_protocol_reference=work["source_protocol_reference"],
        primitive_reference=c.freeze(raw / "measurement.json", raw),
        validation_reference=c.freeze(raw / "validation.json", raw),
        publication_output=str(output.absolute()),
        feature_seal_reference=work["feature_seal_reference"],
        methodology_note="Engineering source coverage only; equal-support fixed lexical scores are descriptive, never semantic entailment, calibrated factuality, detector benefit or generalization. No detector fit, p2 imputation or nine-input learner execution.",
    )
    value["field_principles"] = {
        k: "Immutable measured operand or explicit status; exposed responses do not establish independent generalization."
        for k in value
    }
    value["field_principles"].update(
        source_comparison_ready_score="Numeric1 only when exact parity, custody, resource bounds, recovered support and all owned checks pass.",
        coverage_benefit_score="Absolute engineering reach improvement only; never changed-denominator detector benefit.",
        rows="190 response units times two arms; siblings and cost repeats do not enlarge128-question N.",
        complete_extraction_cost_rows="Three interleaved paired repeats in original roster order; fresh child peak-memory baselines exclude imports.",
    )
    value["reproducibility_checksum"] = c.canonical_hash(value)
    return value


TASK = c.TASK


def replay(path: Path) -> bool:
    """Cold reconstruction rejects missing operands and rehashed numerical or summary edits."""
    try:
        value = json.loads(path.read_bytes())
        work = c.read(value["primitive_reference"])
        for ref in work["code_refs"]:
            if c.reference(Path(ref["path"])) != ref:
                return False
        for source in work["source_refs"]:
            if (
                source["reference"]["sha256"]
                != c.PINS[str(Path(source["origin"]).relative_to(work["root"]))]
            ):
                return False
            c.read(source["reference"])
        data = c.inputs(Path(work["root"]))
        responses = []
        scratch = Path(mkdtemp(prefix="exp8659-replay-", dir="/var/tmp"))
        for index, (row, record) in enumerate(
            zip(data["public"], work["response_records"], strict=True)
        ):
            result: Json = dict(
                opaque_id=row["opaque_id"], original=None, streaming=None, invariance=[], costs=[]
            )
            for item in record["references"]:
                observed = c.read(item["reference"])
                fresh = c.response(
                    row,
                    item["arm"],
                    item["chunk"],
                    scratch / f"{index}-{item['repeat']}-{item['chunk']}",
                )
                for field in (
                    "features",
                    "complete",
                    "total_window_count",
                    "complete_window_digest",
                    "reserved_working_bytes",
                ):
                    if fresh.get(field) != observed.get(field):
                        return False
                if observed["complete"] and observed["window_reference"]:
                    c.audit_windows(row, observed["window_reference"])
                repeat, arm = item["repeat"], item["arm"]
                if repeat < 3:
                    name = f"r{index}-repeat{repeat}-{arm}-chunk32"
                    receipt = next(v for v in work["measurement_receipts"] if v["name"] == name)
                    for stream in ("stdout", "stderr"):
                        if (
                            c.sha256_file(Path(receipt[stream + "_path"]))
                            != receipt[stream + "_sha256"]
                        ):
                            return False
                    result["costs"].append(
                        dict(
                            observed["cost"],
                            repeat=repeat,
                            order_in_pair=int(
                                arm != ("streaming_complete" if repeat == 1 else "original_capped")
                            ),
                            receipt=receipt,
                        )
                    )
                    if repeat == 0:
                        result["original" if arm == "original_capped" else "streaming"] = observed
                else:
                    result["invariance"].append(observed)
            responses.append(result)
            c.progress("cold_replay_response", index + 1, len(data["public"]) - index - 1)
        reduced = c.reduce(dict(data, responses=responses))
        if any(reduced[k] != work["report"][k] for k in reduced):
            return False
        seal = c.read(work["feature_seal_reference"])
        if seal["features"] != c.canonical_hash(reduced):
            return False
        if c.label_report(data, reduced) != {
            k: work["report"][k]
            for k in (
                "common_support_metrics",
                "recovered_support_metrics",
                "annotation_sensitivity",
            )
        }:
            return False
        return (
            build(work, c.read(value["validation_reference"]), Path(value["publication_output"]))
            == value
        )
    except (OSError, ValueError, KeyError, TypeError, StopIteration):
        return False


def run_checks(raw: Path) -> Json:
    """Keep producer and consumer coverage separate and combine only copies of raw SQLite."""
    c.progress("validation_start")
    scratch = Path(mkdtemp(prefix="exp8659-checks-", dir="/var/tmp"))
    receipts, plan, manifests = [], [], []
    for suite, role, sources, label in [(c.TEST, "current_adapter", c.OWNED, "owned8659")] + [
        (suite, "historical_consumer", sources, f"consumer{i}")
        for i, (suite, sources) in enumerate(CONSUMERS.items())
    ]:
        rs, ms, ps = child_suite(suite, role, sources, [], raw, scratch, label)
        receipts += rs
        plan += ps
        manifests += ms
    receipts.append(exact_union(plan))
    shards = [s for m in manifests if m["role"]["role"] == "current_adapter" for s in m["shards"]]
    coverage = combine(shards, [c.ROOT / p for p in c.OWNED], scratch / "combined")
    second = combine(shards, [c.ROOT / p for p in c.OWNED], scratch / "combined_again")
    coverage["passed"] = coverage["passed"] and second == coverage
    commands = [
        ("ruff_check", [str(c.ROOT / ".venv/bin/ruff"), "check", *c.OWNED, c.TEST]),
        ("ruff_format", [str(c.ROOT / ".venv/bin/ruff"), "format", "--check", *c.OWNED, c.TEST]),
        ("mypy", [str(c.ROOT / ".venv/bin/mypy"), "--strict", "--follow-imports=silent", *c.OWNED]),
        ("spec_coverage", [sys.executable, "scripts/check_spec_coverage.py", "--files", c.TEST]),
    ]
    for name, argv in commands:
        receipts.append(child(name, argv, raw / "checks", deadline=120))
    value = dict(
        receipts=receipts, coverage=coverage, child_shard_manifest=manifests, exact_node_union=plan
    )
    c.atomic_json(raw / "validation.json", value)
    c.progress("validation_complete", len(receipts), 0)
    return value


def validate_candidate(path: Path, raw: Path) -> Json:
    """Run unchanged consumers on the exact candidate and retain their real severities."""
    logs = raw / "terminal_checks" / c.sha256_file(path).split(":")[1]
    commands = [
        ("cold_replay", [sys.executable, "-u", str(c.ROOT / c.CLI), "--cold-replay", str(path)]),
        ("adversarial", [sys.executable, "scripts/adversarial_verify.py", "--json", str(path)]),
        (
            "row_lint",
            [sys.executable, "scripts/verdict_row_consistency_lint.py", "--strict", str(path)],
        ),
    ]
    receipts = [child(name, argv, logs, deadline=300) for name, argv in commands]
    findings = json.loads(Path(receipts[1]["stdout_path"]).read_text())
    report = dict(
        passed=all(v["passed"] for v in receipts), receipts=receipts, adversarial_findings=findings
    )
    c.atomic_json(
        raw / "terminal_validation.json", dict(candidate_reference=c.reference(path), **report)
    )
    return report


def main(argv: list[str] | None = None) -> int:
    """Expose private worker/replay modes while the default path publishes only checked bytes."""
    c.progress("start_no_model_load_calls_0")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", type=date_value, default=datetime.now().strftime("%Y%m%d"))
    parser.add_argument("--root", type=Path, default=c.ROOT)
    parser.add_argument("--output", type=Path, default=c.ROOT / "results" / (c.NAME + ".json"))
    parser.add_argument("--cold-replay", type=Path)
    parser.add_argument("--response", type=Path)
    parser.add_argument("--arm", choices=("original_capped", "streaming_complete"))
    parser.add_argument("--chunk", type=int, default=32)
    parser.add_argument("--primitive", type=Path)
    parser.add_argument("--response-output", type=Path)
    args = parser.parse_args(argv)
    if args.response:
        if not all((args.arm, args.primitive, args.response_output)):
            parser.error("--arm, --primitive and --response-output are required with --response")
        c.progress("before_extraction")
        c.atomic_json(
            args.response_output,
            c.response(args.response, args.arm, args.chunk, args.primitive),
        )
        c.progress("after_extraction", 1, 0)
        return 0
    if args.cold_replay:
        passed = replay(args.cold_replay)
        c.progress("replay_passed" if passed else "replay_rejected")
        return int(not passed)
    started = time.monotonic()
    output = args.output.absolute()
    raw = output.parent / "raw" / output.stem / "invocations" / str(time.time_ns())
    work = measure(args.root, raw, args.date)
    validation = run_checks(raw)
    value = build(work, validation, output)
    preflight = raw / "private_candidate.json"
    c.atomic_json(preflight, value)
    checked = validate_candidate(preflight, raw)
    validation["receipts"] += checked["receipts"]
    validation["adversarial_findings"] = checked["adversarial_findings"]
    c.atomic_json(raw / "validation.json", validation)
    value = build(work, validation, output)
    receipt = publish_primary(output, value, lambda p: validate_candidate(p, raw))
    c.atomic_json(
        raw / "publication_receipt.json", dict(receipt, total_duration_s=time.monotonic() - started)
    )
    c.progress("published", 1, 0)
    return 0
