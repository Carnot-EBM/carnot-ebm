"""REQ-REPORT-8947: authenticate inputs and publish checked replay methods only."""

import argparse
import json
import os
from pathlib import Path
import shutil
import sys
from tempfile import TemporaryDirectory
import time
from typing import Any

from carnot.reporting import gradient_replay_methods_8947 as m
from carnot.reporting import topology_memory_runner_8881 as inherited
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
PINS = {
    "experiment_8944_v765_branch_contract": "sha256:80229f95dc2507f5cbd9a88b1bf8be69c0365d44841603b39c6dfee5a7906129",
    "experiment_8933_v764_compatible_replay_methods": "sha256:317b615ee1bc56f48b56161577ca77a716c40214c18ce15badba4aa2cf210169",
    "experiment_8941_v764_compatible_replay_benefit": "sha256:99a9be53396e5fd3867ab0981296d6be37e8863686d1b278188c11ceb0b9b489",
}
HEALTH = Path("/tmp/carnot8947-global/receipt.json")


def preconditions(raw: Path, sources: Path) -> Json:
    """Authenticate retained bytes; only the current branch contract is a readiness gate."""
    refs, diagnostics = [], []
    operands = {}
    for name, pin in PINS.items():
        path = sources / (name + ".json")
        d: Json = dict(
            check="missing_file",
            upstream=name,
            artifact_field="producer_bytes",
            expected=pin,
            observed=None,
            path=str(path),
            hash=None,
        )
        m.progress("authenticate_before_" + name)
        try:
            digest = sha256_file(path)
            d.update(check="rejected_producer", hash=digest, observed=digest)
            value = json.loads(path.read_bytes())
            require(digest == pin, "producer_bytes")
            if name == "experiment_8944_v765_branch_contract":
                d.update(
                    check="missing_field",
                    artifact_field="contract_ready_score",
                    expected=1,
                    observed=None,
                )
                actual = value["contract_ready_score"]
                d.update(
                    check="observed_zero" if actual == 0 else "rejected_producer", observed=actual
                )
                require(type(actual) is int and actual == 1, "branch_contract")
            refs.append(reference(path))
            for key in ("source_artifact_hashes", "code_config_hashes", "raw_shard_hashes"):
                for ref in value[key]:
                    operands[ref["path"], ref["sha256"]] = ref
        except (OSError, ValueError, KeyError, TypeError) as error:
            if isinstance(error, KeyError) and error.args[0] != d["artifact_field"]:
                d.update(
                    artifact_field=error.args[0],
                    expected="present retained input field",
                    observed=None,
                )
            diagnostics.append(
                dict(
                    d,
                    check="missing_file"
                    if isinstance(error, OSError)
                    else "missing_field"
                    if isinstance(error, KeyError)
                    else d["check"],
                    error=str(error),
                )
            )
        m.progress("authenticate_after_" + name, 1, 0)
    for i, ((label, expected), ref) in enumerate(operands.items()):
        path = Path(label)
        observed = sha256_file(path) if path.is_file() else None
        if observed != expected:
            diagnostics.append(
                dict(
                    check="missing_file" if observed is None else "rejected_producer",
                    upstream="retained_operand",
                    artifact_field="sha256",
                    expected=expected,
                    observed=observed,
                    path=str(path),
                    hash=observed,
                )
            )
        else:
            refs.append(reference(path))
        if i % 200 == 0:
            m.progress("authenticate_operands", i, len(operands) - i)
    atomic_json(raw / "original_operands.json", dict(rows=refs))
    return dict(
        external_ready=not diagnostics,
        references=refs,
        diagnostics=diagnostics,
        resources=dict(
            passed=shutil.disk_usage(raw).free > 512 * 1024**2,
            total_budget_s=4200,
            hard_cap_s=4800,
            pretrained_loads=0,
        ),
    )


def adapter() -> Any:
    """Reuse original unit/child coverage, private E2E checks and unchanged scanners."""
    runtime = namespace(inherited, "compatible8947_checks")
    runtime.m = m
    runtime.stream = lambda name, argv, logs, **kwargs: stream(
        name,
        argv,
        logs,
        **dict(
            kwargs,
            deadline=180
            if name == "collect_once"
            else 1200
            if name == "child_0"
            else kwargs.get("deadline", 600),
        ),
    )
    return runtime


def replay(path: Path) -> Json:
    """Rebuild development transitions so rehashing a false aggregate cannot admit it."""
    value = json.loads(path.read_bytes())
    checksum = value.pop("reproducibility_checksum")
    require(canonical_hash(value) == checksum, "artifact_checksum")
    require(
        (value["experiment_id"], value["task_id"], value["milestone"], value["run_date"])
        == (8947, m.TASK, "2026.11.765", "20261109"),
        "producer_identity",
    )
    p = m.protocol()
    require(
        value["protocol"] == p
        and value["protocol_hash"] == canonical_hash(p)
        and value["stream_manifest"]
        == value["frozen_evaluation_manifest"]
        == p["frozen_evaluation_manifest"]
        and value["optimizer_equations"] == p["optimizer_equations"],
        "protocol_drift",
    )
    for ref in (
        value["source_artifact_hashes"] + value["code_config_hashes"] + value["raw_shard_hashes"]
    ):
        require(sha256_file(Path(ref["path"])) == ref["sha256"], "operand_hash")
    require(
        value["producer_invocation"]["executed_sources"]
        == [reference(m.ROOT / n) for n in m.OWNED],
        "executed_sources",
    )
    require(
        not validate_current_work_receipt(value["current_work_receipt"])
        and value["duration_s"] == value["current_work_receipt"]["duration_s"]
        and value["model_invocation_counts"] == value["current_work_receipt"]["invocation_counts"]
        and value["verifier_is_oracle"] is True
        and value["independent_generalization_score"] == 0,
        "compute_scope",
    )
    raw = Path(value["raw_directory"])
    measured = json.loads((raw / "measurement.json").read_bytes())
    for key, expected in measured.items():
        require(value[key] == expected, "reduction_drift:" + key)
    if value["completed_count"] == 18432 and not value["owned_errors"]:
        with TemporaryDirectory(prefix="carnot8947-replay-") as directory:
            rebuilt = m.measure(Path(directory), json.loads((raw / "heads.json").read_bytes()))
            require(rebuilt == measured, "semantic_drift")
    else:
        require(measured == m.recover(raw, value["missing_reason"]), "unfinished_drift")
    complete = sum(r["status"] == "complete" for r in value["rows"])
    failed = sum(r["status"] == "failed" for r in value["rows"])
    require(
        value["intended_count"] == len(value["rows"]) == 18432
        and value["completed_count"] == complete
        and value["censored_count"] == 18432 - complete - failed
        and value["failed_count"] == failed
        and value["excluded_count"] == value["independent_count"] == 0,
        "count_drift",
    )
    coverage = value["owned_statement_coverage"]
    valid = bool(
        coverage["complete"]
        and coverage["percent"] == 100
        and not any(r["missing"] or r["excluded"] for r in coverage["files"])
        and {r["path"] for r in coverage["files"]} == set(m.OWNED)
        and value["test_union"]["passed"]
        and all(r["passed"] for r in value["validation_receipts"])
    )
    require(value["required_checks_passed"] == valid, "validation_readiness")
    ready = int(
        valid
        and value["preconditions_checked"]["external_ready"]
        and value["preconditions_checked"]["resources"]["passed"]
        and complete == 18432
        and not value["owned_errors"]
        and all(r["passed"] for r in value["control_rows"])
    )
    require(
        value["gradient_methods_ready_score"] == int(ready and m.exercised(value)),
        "readiness_drift",
    )
    verdict = (
        "blocked"
        if not value["preconditions_checked"]["external_ready"]
        else "null"
        if ready
        else "disqualified"
    )
    require(value["verdict_class"] == verdict, "verdict_drift")
    return dict(passed=True, status="replay_passed", sha256=sha256_file(path))


def main(argv: list[str] | None = None) -> int:
    """Publish after owned checks; missing external evidence is a terminal blocked result."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", choices=["20261109"], default="20261109")
    parser.add_argument("--sources", type=Path, default=m.ROOT / "results")
    parser.add_argument("--output", type=Path, default=m.ROOT / "results" / (m.NAME + ".json"))
    parser.add_argument("--private-output", type=Path)
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
    if args.private_output and output.resolve().is_relative_to(m.ROOT):
        parser.error("private outputs must remain outside checkout")
    started = time.monotonic_ns()
    raw = output.parent / "raw" / m.NAME / "invocations" / str(started)
    raw.mkdir(parents=True, mode=0o700)
    invocation = dict(
        executable=sys.executable,
        arguments=argv if argv is not None else sys.argv[1:],
        cwd=os.getcwd(),
        pid=os.getpid(),
        executed_sources=[reference(m.ROOT / n) for n in m.OWNED],
    )
    atomic_json(raw / "invocation.json", invocation)
    template = ExperimentTemplate(
        8947,
        "Released-label gradient replay methods",
        "results/" + m.NAME + ".json",
        repo_root=m.ROOT,
        seed=m.SEED,
    )
    m.progress("preconditions_before", 0, 1)
    pre = preconditions(raw, args.sources)
    atomic_json(raw / "preconditions.json", pre)
    m.progress("preconditions_after", 1, 0)
    reason = "external_or_owned_precondition"
    result, errors = m.census(reason), []
    if pre["external_ready"] and pre["resources"]["passed"]:
        try:
            fit = next(r for r in pre["references"] if Path(r["path"]).name == "heads.json")
            heads = json.loads(Path(fit["path"]).read_bytes())
            with template.phase("development_pilots"):
                result = m.measure(raw, heads)
        except (OSError, ValueError, RuntimeError, KeyError) as error:
            reason = str(error)
            errors.append(reason)
            result = m.recover(raw, reason)
    atomic_json(raw / "measurement.json", result)
    measured = time.monotonic_ns()
    m.progress("validation_before", 0, 1)
    with TemporaryDirectory(prefix="carnot8947-validation-") as directory:
        validation = (
            unqualified()
            if args.private_output
            else adapter().validate(raw / "validation", Path(directory))
        )
    valid = bool(
        validation["coverage"]["complete"]
        and validation["coverage"]["percent"] == 100
        and {r["path"] for r in validation["coverage"]["files"]} == set(m.OWNED)
        and validation["union"]["passed"]
        and all(r["passed"] for r in validation["receipts"])
    )
    m.progress("validation_after", 1, 0)
    ended = time.monotonic_ns()
    complete = sum(r["status"] == "complete" for r in result["rows"])
    failed = sum(r["status"] == "failed" for r in result["rows"])
    ready = int(
        pre["external_ready"]
        and pre["resources"]["passed"]
        and valid
        and not errors
        and complete == 18432
        and all(r["passed"] for r in result["control_rows"])
    )
    verdict = "blocked" if not pre["external_ready"] else "null" if ready else "disqualified"
    p = m.protocol()
    health: Json = dict(
        command=".venv/bin/pytest tests/python -q",
        scope="one bounded repository health attempt; owned acceptance remains separate",
        receipt=dict(passed=False, status="not_finished"),
    )
    if HEALTH.is_file():
        health["receipt"] = json.loads(HEALTH.read_bytes())
        for suffix in ("stdout", "stderr"):
            target = raw / ("global_suite." + suffix)
            shutil.copyfile(health["receipt"][suffix + "_path"], target)
            health["receipt"][suffix + "_path"] = str(target)
    atomic_json(raw / "global_health.json", health)
    value = dict(
        result,
        experiment_id=8947,
        task_id=m.TASK,
        milestone="2026.11.765",
        run_date=args.date,
        honest_verdict=f"complete_{verdict}_gradient_replay_methods",
        verdict_class=verdict,
        inference_substrate="verifier_ensemble_against_cached_candidates",
        inference_substrate_class="no_model_load"
        if result["numerical_training_calls"]
        else "blocked_no_run",
        execution_venue="host",
        MODEL_SPECS=[],
        model_specs=[],
        verifier_is_oracle=True,
        exposure_scope="constructed development only; evaluation outcomes unopened",
        independent_generalization_score=0,
        preconditions_checked=pre,
        random_seed=m.SEED,
        producer_invocation=invocation,
        source_artifact_hashes=pre["references"],
        code_config_hashes=[
            reference(m.ROOT / n)
            for n in m.OWNED
            + m.TESTS
            + [
                "python/carnot/memory/signed_memory_8892.py",
                "python/carnot/memory/compatible_replay_8933.py",
                "python/carnot/reporting/compatible_replay_methods_8933.py",
                "python/carnot/memory/guarded_memory_8905.py",
                "python/carnot/reporting/guarded_update_methods_8905.py",
                "python/carnot/experiment_5857_clean_transfer_selective_replay.py",
                "scripts/experiment_template.py",
            ]
        ],
        raw_shard_hashes=[
            reference(path)
            for pattern in ("*.json", "*.stdout", "*.stderr")
            for path in sorted(raw.rglob(pattern))
        ]
        + validation["raw_references"],
        raw_directory=str(raw),
        intended_count=18432,
        completed_count=complete,
        failed_count=failed,
        censored_count=18432 - complete - failed,
        excluded_count=0,
        independent_count=0,
        gate_check_summary=pre["diagnostics"],
        acceptance_gates=p,
        required_checks_passed=valid,
        test_union=validation["union"],
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
        validation_receipts=validation["receipts"],
        owned_statement_coverage=validation["coverage"],
        owned_errors=errors,
        gradient_methods_ready_score=int(ready and m.exercised(result)),
        protocol=p,
        protocol_hash=canonical_hash(p),
        optimizer_equations=p["optimizer_equations"],
        stream_manifest=p["frozen_evaluation_manifest"],
        frozen_evaluation_manifest=p["frozen_evaluation_manifest"],
        hardware_path=dict(
            current="CPU signed-head gradients, all prior-buffer gradient scans and atomic JSON",
            batchable="bounded per-row gradient scans and tiny-head SGD can batch on CPU/GPU",
            hardware_speed="unmeasured",
            pretrained_loads=0,
        ),
        missing_reason=reason if complete < 18432 else None,
        cited_upstream_artifacts=[
            dict(producer=n, sha256=h, scope="authenticated retained input")
            for n, h in PINS.items()
        ],
        methodology="Eight development streams, six paired arms,48 queries and48 untouched retention queries at0/24/48;24 fresh evaluation recipes sealed.",
        methodology_note="Oracle constructed streams do not establish natural-language generalization, prospective benefit or a verifier moat.",
        flagged_adversarial=False,
        field_principles={},
    )
    spans = [
        dict(phase="preconditions_and_pilots", start_s=0.0, end_s=(measured - started) / 1e9),
        dict(phase="validation", start_s=(measured - started) / 1e9, end_s=(ended - started) / 1e9),
    ]

    def seal(final: int) -> None:
        receipt = build_current_work_receipt(
            run_id=str(started),
            owner_pid=os.getpid(),
            events=[],
            inference_substrate=value["inference_substrate"],
            inference_substrate_class=value["inference_substrate_class"],
            execution_venue="host",
            inference_substrate_details=dict(
                backend="JAX CPU signed normalized energy head", pretrained_loads=0
            ),
            started_monotonic_ns=started,
            ended_monotonic_ns=final,
            phase_spans=spans,
            small_ebm_training=dict(
                performed=result["numerical_training_calls"] > 0,
                numerical_training_calls=result["numerical_training_calls"],
            ),
        )
        value.update(
            current_work_receipt=receipt,
            model_invocation_counts=receipt["invocation_counts"],
            duration_s=receipt["duration_s"],
            phase_spans=spans,
        )
        value.pop("reproducibility_checksum", None)
        for key in list(value) + ["reproducibility_checksum"]:
            value["field_principles"].setdefault(
                key,
                "Current invocation or authenticated development evidence; evaluation remains unopened.",
            )
        value["reproducibility_checksum"] = canonical_hash(value)

    value = normalize_artifact_for_template_write(value)
    value["field_principles"].update(
        gradient_methods_ready_score="Owned causal controls, sealed protocol and measured owned validation; independent of benefit.",
        numerical_training_calls="All fresh development small-head evaluated gradients, including discarded candidates; no upstream or generator compute.",
        independent_count="Zero fresh evaluation streams opened; eight development RNG trials share constructed graph families.",
        replay_selection_rows="Label-dependent W1 choices from prior released TRAIN buffer, including empty replay and stable ties.",
        gradient_ledger="Both gradients for every trainable TRAIN release, even repeats; frozen arm has zero.",
        buffer_state_hashes="Final durable FIFO and RNG bytes for each stream and arm.",
        stream_manifest="24 independent world/history seeds,48 online and48 retention recipes each; labels unopened.",
    )
    seal(ended)
    candidate = raw / "prospective.json"
    atomic_json(candidate, value)
    if not args.private_output:
        m.progress("terminal_before", 0, 3)
        reports = adapter().terminal(candidate, raw)
        value["validation_receipts"] += reports
        value["raw_shard_hashes"] += [
            reference(Path(r[k + "_path"])) for r in reports for k in ("stdout", "stderr")
        ]
        if not all(r["passed"] for r in reports):
            value.update(
                honest_verdict="complete_disqualified_gradient_replay_methods",
                verdict_class="disqualified",
                gradient_methods_ready_score=0,
                required_checks_passed=False,
            )
        final = time.monotonic_ns()
        spans.append(
            dict(phase="terminal", start_s=(ended - started) / 1e9, end_s=(final - started) / 1e9)
        )
        seal(final)
        m.progress("terminal_after", 3, 0)
    require((time.monotonic_ns() - started) / 1e9 < 4200, "total_work_budget")
    publish_primary(output, value, replay)
    m.progress("published_" + value["verdict_class"], complete, 18432 - complete)
    return 0
