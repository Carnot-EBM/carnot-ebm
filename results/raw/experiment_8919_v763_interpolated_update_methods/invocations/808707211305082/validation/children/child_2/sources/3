"""REQ-REPORT-8919: authenticate, measure, validate and publish bounded development work."""

import argparse
import json
import os
from pathlib import Path
import shutil
import sys
from tempfile import TemporaryDirectory
import time
from typing import Any

from carnot.reporting import interpolated_update_methods_8919 as m
from carnot.reporting import topology_memory_runner_8881 as inherited
from carnot.reporting import guarded_update_runner_8905 as previous
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
    "experiment_8911_v762_guarded_benefit_audit": (
        "sha256:28f12afb613a933f778248172532fa84c71f474b631b3c9aaa9fd4d296048d74",
        "audit_ready_score",
        "exp8911-guarded-benefit-audit",
    ),
    "experiment_8905_v762_guarded_update_methods": (
        "sha256:acbd0cd68c70e51811c36897180a2dc302341f9fa8fa2f381c3c6e642ab16779",
        "guarded_methods_ready_score",
        "exp8905-guarded-update-methods",
    ),
}


def preconditions(raw: Path, sources: Path) -> Json:
    """Keep original producer assertions, including original bytes of the extended store."""
    refs, diagnostics = [], []
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
        m.progress("authenticate_before_" + name, 0, 1)
        try:
            d["check"] = "rejected_producer"
            value = json.loads(path.read_bytes())
            d.update(check="missing_field", hash=sha256_file(path))
            actual = value[field]
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
                and value["task_id"] == task
                and value["experiment_id"] == int(name.split("_")[1])
                and value["milestone"] == "2026.11.762"
                and value["required_checks_passed"] is True,
                "producer_identity",
            )
            refs.append(reference(path))
            operands = (
                value["source_artifact_hashes"]
                + value["code_config_hashes"]
                + value["raw_shard_hashes"]
            )
            for index, ref in enumerate(operands):
                original = Path(ref["path"])
                operand = original
                if original == m.ROOT / "python/carnot/memory/guarded_memory_8905.py":
                    operand = (
                        m.ROOT
                        / "results/raw"
                        / m.NAME
                        / "original_sources"
                        / ref["sha256"].split(":")[1]
                    )
                d.update(
                    artifact_field="original_operand", path=str(operand), expected=ref["sha256"]
                )
                d["observed"] = d["hash"] = sha256_file(operand)
                require(d["observed"] == d["expected"], "original_operand_drift")
                refs.append(dict(reference(operand), original_path=str(original)))
                if index % 100 == 0:
                    m.progress("authenticate_operands", index, len(operands) - index)
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
    """Use the existing validation and terminal supervision without cloning their runner."""
    runtime = namespace(inherited, "interpolated8919_checks")
    runtime.m = m
    runtime.stream = lambda name, argv, logs, **kwargs: stream(
        name,
        argv,
        logs,
        **dict(kwargs, deadline=180 if name == "collect_once" else kwargs.get("deadline", 600)),
    )
    return runtime


def census(reason: str) -> Json:
    """Censor all scheduled units when external evidence is absent."""
    runtime = namespace(previous, "interpolated8919_census")
    runtime.m = type(
        "Methods",
        (),
        dict(protocol=staticmethod(m.protocol), g=type("Arms", (), dict(ARMS=m.ARMS))),
    )
    return dict(runtime.census(reason), gradient_call_ledger=[])


def recover(raw: Path, reason: str) -> Json:
    """Retain completed seals and charge durable gradients even after an owned interruption."""
    result = census(reason)
    completed: dict[tuple[str, str], Json] = {}
    for path in sorted(raw.glob("v763-development-*.json")):
        value = json.loads(path.read_bytes())
        completed.update({(r["unit_id"], r["arm"]): r for r in value["rows"]})
        result["control_rows"].extend(value["control_rows"])
    for path in sorted((raw / "states").glob("*/rows.json")):
        completed.update(
            {(r["unit_id"], r["arm"]): r for r in json.loads(path.read_bytes())["rows"]}
        )
    failed = {
        p.stem.removeprefix("attempt-")
        for p in raw.glob("attempt-*.json")
        if json.loads(p.read_bytes())["status"] == "failed"
    }
    result["rows"] = [
        completed.get(
            (r["unit_id"], r["arm"]), dict(r, status="failed") if r["cluster_id"] in failed else r
        )
        for r in result["rows"]
    ]
    for arm in m.ARMS:
        for path in (raw / "states").glob("*/" + arm + ".json"):
            state = json.loads(path.read_bytes())
            result["numerical_training_calls"] += state["training_calls"]
            result["promotion_rows"].extend(
                dict(
                    r,
                    arm=arm,
                    cluster_id=path.parent.name,
                    seed=next(
                        p["seed"]
                        for p in m.protocol()["pilot_manifest"]
                        if p["stream_id"] == path.parent.name
                    ),
                )
                for r in state["promotions"]
            )
    return result


def replay(path: Path) -> Json:
    """Recompute semantic evidence so changing and rehashing an aggregate still fails."""
    value = json.loads(path.read_bytes())
    checksum = value.pop("reproducibility_checksum")
    require(canonical_hash(value) == checksum, "artifact_checksum")
    require(
        (value["experiment_id"], value["task_id"], value["milestone"], value["run_date"])
        == (8919, m.TASK, "2026.11.763", "20261108"),
        "producer_identity",
    )
    p = m.protocol()
    require(
        value["protocol"] == p
        and value["protocol_hash"] == canonical_hash(p)
        and value["evaluation_manifest"] == p["frozen_evaluation_manifest"]
        and value["optimizer_constants"] == p["optimizer_constants"]
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
        if key != "candidate_ready_score":
            require(value[key] == expected, "reduction_drift:" + key)
    if value["preconditions_checked"]["external_ready"]:
        require(
            value["source_artifact_hashes"] == value["preconditions_checked"]["references"],
            "source_inventory",
        )
        for name, (pin, field, task) in PINS.items():
            ref = next(r for r in value["source_artifact_hashes"] if Path(r["path"]).stem == name)
            source = json.loads(Path(ref["path"]).read_bytes())
            require(
                ref["sha256"] == pin
                and type(source[field]) is int
                and source[field] == 1
                and source["task_id"] == task,
                "upstream_pin",
            )
    if value["completed_count"] == 6144 and not value["owned_errors"]:
        with TemporaryDirectory(prefix="carnot8919-replay-") as directory:
            rebuilt = m.measure(Path(directory), json.loads((raw / "heads.json").read_bytes()))
            require(rebuilt == measured, "semantic_drift")
    else:
        require(measured == recover(raw, value["missing_reason"]), "unfinished_drift")
    complete = sum(r["status"] == "complete" for r in value["rows"])
    failed = sum(r["status"] == "failed" for r in value["rows"])
    require(
        value["intended_count"] == len(value["rows"]) == 6144
        and value["completed_count"] == complete
        and value["censored_count"] == 6144 - complete - failed
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
        and complete == 6144
        and not value["owned_errors"]
        and all(r["passed"] for r in value["control_rows"])
    )
    require(
        value["methods_ready_score"] == ready
        and value["candidate_ready_score"] == (measured["candidate_ready_score"] if ready else 0),
        "readiness_drift",
    )
    require(
        value["verdict_class"]
        == (
            "blocked"
            if not value["preconditions_checked"]["external_ready"]
            else "null"
            if ready
            else "disqualified"
        ),
        "verdict_drift",
    )
    return dict(passed=True, status="replay_passed", sha256=sha256_file(path))


def main(argv: list[str] | None = None) -> int:
    """Publish the terminal primary only after owned checks and unchanged scanners finish."""
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
        8919,
        "Interpolated candidate update methods",
        "results/" + m.NAME + ".json",
        repo_root=m.ROOT,
        seed=m.SEED,
    )
    m.progress("preconditions_before", 0, 1)
    pre = preconditions(raw, args.sources)
    atomic_json(raw / "preconditions.json", pre)
    m.progress("preconditions_after", 1, 0)
    reason = "external_or_owned_precondition"
    result, errors = census(reason), []
    if pre["external_ready"] and pre["resources"]["passed"]:
        try:
            fit = next(r for r in pre["references"] if Path(r["path"]).name == "fit.json")
            heads = json.loads(Path(fit["path"]).read_bytes())["heads"]["signed"]
            atomic_json(raw / "heads.json", heads)
            with template.phase("development_pilots"):
                result = m.measure(raw, heads)
        except (OSError, ValueError, RuntimeError, KeyError) as error:
            reason = str(error)
            errors.append(reason)
            result = recover(raw, reason)
    atomic_json(raw / "measurement.json", result)
    measured = time.monotonic_ns()
    m.progress("validation_before", 0, 1)
    with TemporaryDirectory(prefix="carnot8919-validation-") as directory:
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
        and complete == 6144
        and all(r["passed"] for r in result["control_rows"])
    )
    verdict = "blocked" if not pre["external_ready"] else "null" if ready else "disqualified"
    p = m.protocol()
    value = dict(
        result,
        experiment_id=8919,
        task_id=m.TASK,
        milestone="2026.11.763",
        run_date=args.date,
        honest_verdict=f"complete_{verdict}_interpolated_update_methods",
        verdict_class=verdict,
        inference_substrate="cpu_signed_linear_normalized_energy_head",
        inference_substrate_class="no_model_load"
        if result["numerical_training_calls"]
        else "blocked_no_run",
        execution_venue="host",
        MODEL_SPECS=[],
        verifier_is_oracle=True,
        exposure_scope="constructed development only; evaluation labels unopened",
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
                "python/carnot/reporting/guarded_update_methods_8905.py",
                "python/carnot/reporting/guarded_benefit_audit_8911.py",
                "scripts/experiment_template.py",
            ]
        ],
        raw_shard_hashes=[reference(path) for path in sorted(raw.rglob("*.json"))]
        + validation["raw_references"],
        raw_directory=str(raw),
        intended_count=6144,
        completed_count=complete,
        failed_count=failed,
        censored_count=6144 - complete - failed,
        excluded_count=0,
        independent_count=0,
        gate_check_summary=pre["diagnostics"],
        acceptance_gates=p,
        required_checks_passed=valid,
        validation_receipts=validation["receipts"],
        owned_statement_coverage=validation["coverage"],
        owned_errors=errors,
        methods_ready_score=ready,
        candidate_ready_score=result["candidate_ready_score"] if ready else 0,
        protocol=p,
        protocol_hash=canonical_hash(p),
        optimizer_equations=p["optimizer_equations"],
        optimizer_constants=p["optimizer_constants"],
        evaluation_manifest=p["frozen_evaluation_manifest"],
        state_schema=dict(
            schema=m.opt.SCHEMA,
            sequences=["fast", "slow"],
            counter="update_count",
            transaction="atomic replace with pending releases",
        ),
        hardware_path=dict(
            current="CPU vector arithmetic and durable JSON",
            pretrained_loads=0,
            future="GPU/NPU kernel only if measured costs justify it",
        ),
        missing_reason=reason if complete < 6144 else None,
        cited_upstream_artifacts=[
            dict(producer=n, sha256=h, artifact_field=f, expected=1)
            for n, (h, f, _) in PINS.items()
        ],
        methodology="Eight paired48-query development streams; four arms; unchanged guard and anchors.",
        methodology_note="Constructed oracle labels establish no independent natural-language benefit.",
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
            inference_substrate_details=dict(backend="JAX CPU", pretrained_loads=0),
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
                "Current invocation receipt or frozen development evidence; evaluation remains unopened.",
            )
        value["reproducibility_checksum"] = canonical_hash(value)

    value = normalize_artifact_for_template_write(value)
    value["field_principles"] = {
        k: "Byte-bound current development evidence; no evaluation or independent language claim."
        for k in value
    }
    value["field_principles"].update(
        methods_ready_score="Owned optimizer, causal, persistence and validation correctness; independent of benefit.",
        candidate_ready_score="Finite changed candidate, future development Brier mean<=-0.01 and retention increases<=0.01.",
        numerical_training_calls="Per-example gradients executed by four small-head arms; no generator training.",
        independent_count="Zero evaluation streams opened; eight paired development streams establish feasibility only.",
        evaluation_manifest="Twenty-four unopened seeds, four matched arms and a nine-bound family frozen before evaluation.",
        gradient_call_ledger="Per-stream and per-arm calls, guard costs and original promotion operands.",
        source_artifact_hashes="Original producer assertions and original operands; extended store's earlier bytes are retained.",
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
                honest_verdict="complete_disqualified_interpolated_update_methods",
                verdict_class="disqualified",
                methods_ready_score=0,
                candidate_ready_score=0,
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
    m.progress("published_" + value["verdict_class"], complete, 6144 - complete)
    return 0
