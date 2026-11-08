"""REQ-REPORT-8933: authenticate inputs and publish checked replay methods only."""

import argparse
import json
import os
from pathlib import Path
import shutil
import sys
from tempfile import TemporaryDirectory
import time
from typing import Any

from carnot.reporting import compatible_replay_methods_8933 as m
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
    "experiment_8925_v763_interpolated_benefit_audit": (
        "sha256:88f6d845ac54d3ff1b553e135b4d8bf416cf714bbbbd1e73bcc0acb1635c7269",
        "audit_ready_score",
        "exp8925-interpolated-benefit-audit",
    ),
    "experiment_8919_v763_interpolated_update_methods": (
        "sha256:ddc6d3cc5aecffabe0e99079833361e98f5085925ea6f30f3115b6109f7ab2ec",
        "methods_ready_score",
        "exp8919-interpolated-update-methods",
    ),
}
PRECEDENT = "experiment_5857_clean_transfer_selective_replay.json"
PRECEDENT_PIN = "sha256:5768b1141664f1c7574666773540b369beef751b6e1a856725fcc9384f22cfb8"


def preconditions(raw: Path, sources: Path) -> Json:
    """Check producer identities and retained original bytes before any gradients execute."""
    refs, diagnostics = [], []
    operands = {}
    for name, (pin, field, task) in PINS.items():
        path = sources / (name + ".json")
        d = dict(
            check="missing_file",
            upstream=name,
            artifact_field=field,
            expected=1,
            observed=None,
            path=str(path),
            hash=None,
        )
        m.progress("authenticate_before_" + name)
        try:
            data = json.loads(path.read_bytes())
            d.update(check="missing_field", hash=sha256_file(path))
            actual = data[field]
            d.update(check="observed_zero" if actual == 0 else "rejected_producer", observed=actual)
            require(type(actual) is int and actual == 1, "upstream_gate")
            d.update(
                check="rejected_producer",
                artifact_field="producer_bytes",
                expected=pin,
                observed=d["hash"],
            )
            require(
                d["hash"] == pin
                and data["task_id"] == task
                and data["experiment_id"] == int(name.split("_")[1])
                and data["milestone"] == "2026.11.763"
                and data["required_checks_passed"] is True,
                "producer_identity",
            )
            refs.append(reference(path))
            for ref in (
                data["source_artifact_hashes"]
                + data["code_config_hashes"]
                + data["raw_shard_hashes"]
            ):
                operands[ref["path"], ref["sha256"]] = ref
        except (OSError, ValueError, KeyError, TypeError) as error:
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
    for i, ((original, expected), ref) in enumerate(operands.items()):
        path = Path(original)
        if (
            path == m.ROOT / "python/carnot/memory/guarded_memory_8905.py"
            and expected
            == "sha256:379c78d6c15c75ca6e4410bf1b08a8c65b9276f27c562cd23cd6b8faf1dd6033"
        ):
            path = (
                m.ROOT
                / "results/raw/experiment_8919_v763_interpolated_update_methods/original_sources"
                / expected.split(":")[1]
            )
        observed = sha256_file(path) if path.is_file() else None
        if observed != expected:
            diagnostics.append(
                dict(
                    check="missing_file" if observed is None else "rejected_producer",
                    upstream="original_operand",
                    artifact_field="sha256",
                    expected=expected,
                    observed=observed,
                    path=str(path),
                    hash=observed,
                )
            )
        else:
            refs.append(dict(reference(path), original_path=original))
        if i % 200 == 0:
            m.progress("authenticate_operands", i, len(operands) - i)
    path = sources / PRECEDENT
    observed = sha256_file(path) if path.is_file() else None
    if observed != PRECEDENT_PIN:
        diagnostics.append(
            dict(
                check="missing_file" if observed is None else "rejected_producer",
                upstream="Exp5857",
                artifact_field="producer_bytes",
                expected=PRECEDENT_PIN,
                observed=observed,
                path=str(path),
                hash=observed,
            )
        )
    else:
        refs.append(reference(path))
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
    runtime = namespace(inherited, "compatible8933_checks")
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
        == (8933, m.TASK, "2026.11.764", "20261108"),
        "producer_identity",
    )
    p = m.protocol()
    require(
        value["protocol"] == p
        and value["protocol_hash"] == canonical_hash(p)
        and value["stream_manifest"] == p["frozen_evaluation_manifest"]
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
    if value["completed_count"] == 15360 and not value["owned_errors"]:
        with TemporaryDirectory(prefix="carnot8933-replay-") as directory:
            rebuilt = m.measure(Path(directory), json.loads((raw / "heads.json").read_bytes()))
            require(rebuilt == measured, "semantic_drift")
    else:
        require(measured == m.recover(raw, value["missing_reason"]), "unfinished_drift")
    complete = sum(r["status"] == "complete" for r in value["rows"])
    failed = sum(r["status"] == "failed" for r in value["rows"])
    require(
        value["intended_count"] == len(value["rows"]) == 15360
        and value["completed_count"] == complete
        and value["censored_count"] == 15360 - complete - failed
        and value["failed_count"] == failed
        and value["excluded_count"] == value["independent_count"] == 0,
        "count_drift",
    )
    coverage = value["owned_statement_coverage"]
    valid = bool(
        coverage["complete"]
        and coverage["percent"] == 100
        and not any(r["missing"] or r["excluded"] for r in coverage["files"])
        and all(r["passed"] for r in value["validation_receipts"])
    )
    require(value["required_checks_passed"] == valid, "validation_readiness")
    ready = int(
        valid
        and value["preconditions_checked"]["external_ready"]
        and value["preconditions_checked"]["resources"]["passed"]
        and complete == 15360
        and not value["owned_errors"]
        and all(r["passed"] for r in value["control_rows"])
    )
    require(value["replay_methods_ready_score"] == ready, "readiness_drift")
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
    parser.add_argument("--date", choices=["20261108"], default="20261108")
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
        8933,
        "Compatible feedback replay methods",
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
            fit = next(r for r in pre["references"] if Path(r["path"]).name == "fit.json")
            heads = json.loads(Path(fit["path"]).read_bytes())["heads"]["signed"]
            with template.phase("development_pilots"):
                result = m.measure(raw, heads)
        except (OSError, ValueError, RuntimeError, KeyError) as error:
            reason = str(error)
            errors.append(reason)
            result = m.recover(raw, reason)
    atomic_json(raw / "measurement.json", result)
    measured = time.monotonic_ns()
    m.progress("validation_before", 0, 1)
    with TemporaryDirectory(prefix="carnot8933-validation-") as directory:
        validation = (
            unqualified()
            if args.private_output
            else adapter().validate(raw / "validation", Path(directory))
        )
    valid = bool(
        validation["coverage"]["complete"]
        and validation["coverage"]["percent"] == 100
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
        and complete == 15360
        and all(r["passed"] for r in result["control_rows"])
    )
    verdict = "blocked" if not pre["external_ready"] else "null" if ready else "disqualified"
    p = m.protocol()
    value = dict(
        result,
        experiment_id=8933,
        task_id=m.TASK,
        milestone="2026.11.764",
        run_date=args.date,
        honest_verdict=f"complete_{verdict}_compatible_replay_methods",
        verdict_class=verdict,
        inference_substrate="verifier_ensemble_against_cached_candidates",
        inference_substrate_class="no_model_load"
        if result["numerical_training_calls"]
        else "blocked_no_run",
        execution_venue="host",
        MODEL_SPECS=[],
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
                "python/carnot/memory/guarded_memory_8905.py",
                "python/carnot/reporting/guarded_update_methods_8905.py",
                "python/carnot/experiment_5857_clean_transfer_selective_replay.py",
                "scripts/experiment_template.py",
            ]
        ],
        raw_shard_hashes=[reference(path) for path in sorted(raw.rglob("*.json"))]
        + validation["raw_references"],
        raw_directory=str(raw),
        intended_count=15360,
        completed_count=complete,
        failed_count=failed,
        censored_count=15360 - complete - failed,
        excluded_count=0,
        independent_count=0,
        gate_check_summary=pre["diagnostics"],
        acceptance_gates=p,
        required_checks_passed=valid,
        validation_receipts=validation["receipts"],
        owned_statement_coverage=validation["coverage"],
        owned_errors=errors,
        replay_methods_ready_score=ready,
        protocol=p,
        protocol_hash=canonical_hash(p),
        optimizer_equations=p["optimizer_equations"],
        stream_manifest=p["frozen_evaluation_manifest"],
        hardware_path=dict(
            current="CPU signed-head gradients, at most32 distance scans and atomic JSON",
            batchable="bounded matrix-distance scans and tiny-head SGD can batch on CPU/GPU",
            hardware_speed="unmeasured",
            pretrained_loads=0,
        ),
        missing_reason=reason if complete < 15360 else None,
        cited_upstream_artifacts=[
            dict(producer=n, sha256=h, artifact_field=f, expected=1)
            for n, (h, f, _) in PINS.items()
        ],
        methodology="Eight development streams, five paired arms,48 queries and48 untouched retention queries at0/24/48;24 fresh evaluation recipes sealed.",
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
        replay_methods_ready_score="Owned causal controls, sealed protocol and measured owned validation; independent of benefit.",
        numerical_training_calls="Fresh development small-head per-example gradients only; no upstream or generator compute.",
        independent_count="Zero fresh evaluation streams opened; eight development RNG trials share constructed graph families.",
        replay_selection_rows="Label-blind choices from prior released TRAIN buffer, including empty replay and stable ties.",
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
                honest_verdict="complete_disqualified_compatible_replay_methods",
                verdict_class="disqualified",
                replay_methods_ready_score=0,
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
    m.progress("published_" + value["verdict_class"], complete, 15360 - complete)
    return 0
