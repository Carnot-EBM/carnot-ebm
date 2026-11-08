"""REQ-REPORT-8905: publish only terminal, replayable development evidence."""

import argparse
import json
import os
from pathlib import Path
import shutil
import sys
from tempfile import TemporaryDirectory
import time
from typing import Any

from carnot.reporting import guarded_update_methods_8905 as m
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
UPSTREAM = {
    "experiment_8892_v761_signed_feature_methods": (
        "sha256:027641956a2d80c1466048a1f3eb150f968fd7cf274b835774834653e00d8461",
        "signed_methods_ready_score",
        1,
    ),
    "experiment_8898_v761_signed_benefit_audit": (
        "sha256:f4d42bc57859ef21919f379219b96dde0877e55cff9ec4840e42e2e59b146441",
        "label_benefit_score",
        0,
    ),
}


def preconditions(raw: Path, sources: Path) -> Json:
    """Authenticate original assertions and deduplicated byte custody before training."""
    refs: list[Json] = []
    diagnostics: list[Json] = []
    operands: dict[str, str] = {}
    for name, (pin, field, expected) in UPSTREAM.items():
        path = sources / (name + ".json")
        diagnostic = dict(
            check="missing_file",
            upstream=name,
            artifact_field=field,
            expected=expected,
            observed=None,
            path=str(path),
            hash=None,
        )
        m.progress("authenticate_before_" + name)
        try:
            data = path.read_bytes()
            diagnostic["check"] = "rejected_producer"
            value = json.loads(data)
            diagnostic.update(check="missing_field", hash=sha256_file(path))
            actual = value[field]
            diagnostic.update(
                check="observed_zero" if actual == 0 else "rejected_producer", observed=actual
            )
            require(actual == expected, "upstream_gate")
            diagnostic.update(
                check="rejected_producer",
                artifact_field="producer_bytes",
                expected=pin,
                observed=sha256_file(path),
            )
            require(
                sha256_file(path) == pin
                and value["experiment_id"] == int(name.split("_")[1])
                and value["task_id"]
                == f"exp{value['experiment_id']}-"
                + (
                    "signed-feature-methods"
                    if value["experiment_id"] == 8892
                    else "signed-benefit-audit"
                )
                and value["required_checks_passed"] is True,
                "producer_identity",
            )
            refs.append(reference(path))
            for key in ("source_artifact_hashes", "code_config_hashes", "raw_shard_hashes"):
                for ref in value[key]:
                    prior = operands.setdefault(ref["path"], ref["sha256"])
                    require(prior == ref["sha256"], "conflicting_operand_hash")
            if name.startswith("experiment_8892_"):
                fit_path = Path(value["raw_directory"]) / "fit.json"
                refs.append(reference(fit_path))
        except (OSError, ValueError, KeyError, TypeError) as error:
            diagnostics.append(
                dict(
                    diagnostic,
                    check="missing_file"
                    if isinstance(error, OSError)
                    else "missing_field"
                    if isinstance(error, KeyError)
                    else diagnostic["check"],
                    error=str(error),
                )
            )
        m.progress("authenticate_after_" + name, 1, 0)
    for index, (path_label, digest) in enumerate(operands.items()):
        path = Path(path_label)
        observed = sha256_file(path) if path.is_file() else None
        if observed != digest:
            diagnostics.append(
                dict(
                    check="missing_file" if observed is None else "rejected_producer",
                    upstream="original_operand_closure",
                    artifact_field="sha256",
                    expected=digest,
                    observed=observed,
                    path=str(path),
                    hash=observed,
                )
            )
        if index % 500 == 0:
            m.progress("operand_authentication", index, len(operands) - index)
    atomic_json(
        raw / "original_operands.json",
        dict(rows=[dict(path=p, sha256=h) for p, h in operands.items()]),
    )
    refs.append(reference(raw / "original_operands.json"))
    resources = dict(
        passed=all(
            (m.ROOT / ".venv/bin" / n).is_file() for n in ("python", "pytest", "ruff", "mypy")
        )
        and shutil.disk_usage(raw).free > 512 * 1024**2,
        total_budget_s=4200,
        hard_cap_s=4800,
        pretrained_loads=0,
    )
    return dict(
        external_ready=not diagnostics,
        references=refs,
        diagnostics=diagnostics,
        resources=resources,
    )


def validate(raw: Path, private: Path) -> Json:
    """Retain real unit/child coverage and the original private E2E-018/021 checks."""
    runner = namespace(inherited, "guarded8905_validation")
    runner.m = m
    # The original record guard resolves 1.64 million tracked paths before collection.
    runner.stream = lambda name, argv, logs, **kwargs: stream(
        name,
        argv,
        logs,
        **dict(kwargs, deadline=180 if name == "collect_once" else kwargs.get("deadline", 600)),
    )
    return dict(runner.validate(raw, private))


def terminal(candidate: Path, raw: Path) -> list[Json]:
    """Run unchanged scanners and actual outside-checkout replay with no PYTHONPATH."""
    runner = namespace(inherited, "guarded8905_terminal")
    runner.m = m
    return list(runner.terminal(candidate, raw))


def census(reason: str) -> Json:
    """Keep all scheduled development units when no external evidence permits execution."""
    rows = [
        dict(
            unit_id=u["unit_id"] + (f"/checkpoint{c}" if c is not None else ""),
            cluster_id=p["stream_id"],
            seed=p["seed"],
            arm=a,
            split=role,
            checkpoint=c,
            ordinal=i + 1,
            denominator=1,
            metric=None,
            error=None,
            status="censored",
            missing_reason=reason,
        )
        for p in m.protocol()["pilot_manifest"]
        for a in m.g.ARMS
        for role, checkpoints in (("online", [None]), ("retention", [0, 24, 48]))
        for c in checkpoints
        for i, u in enumerate(p[role])
    ]
    return dict(
        rows=rows,
        promotion_rows=[],
        control_rows=[],
        development_rows=[],
        numerical_training_calls=0,
        candidate_ready_score=0,
    )


def recover(raw: Path, reason: str) -> Json:
    """Preserve completed pilots and charge durable work from interrupted candidates."""
    result = census(reason)
    completed = {}
    for path in sorted(raw.glob("v762-development-*.json")):
        stream = json.loads(path.read_bytes())
        completed.update({(r["unit_id"], r["arm"]): r for r in stream["rows"]})
        result["control_rows"].extend(stream["control_rows"])
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
    for arm in m.g.ARMS:
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
    """Rebuild predictions and promotions so rehashed scientific tampering still fails."""
    value = json.loads(path.read_bytes())
    checksum = value.pop("reproducibility_checksum")
    require(canonical_hash(value) == checksum, "artifact_checksum")
    require(
        (value["experiment_id"], value["task_id"], value["milestone"], value["run_date"])
        == (8905, m.TASK, "2026.11.762", "20261107"),
        "producer_identity",
    )
    require(
        value["protocol"] == m.protocol() and value["anchor_manifest"] == m.anchors(),
        "protocol_drift",
    )
    for ref in (
        value["source_artifact_hashes"] + value["code_config_hashes"] + value["raw_shard_hashes"]
    ):
        require(sha256_file(Path(ref["path"])) == ref["sha256"], "operand_hash")
    p = m.protocol()
    expected_splits = [
        dict(group_id=u["unit_id"], split=m.g.split(u["unit_id"]))
        for plan in p["pilot_manifest"] + p["frozen_evaluation_manifest"]
        for u in plan["online"]
    ]
    require(
        value["split_manifest"] == expected_splits
        and value["expected_row_counts"] == p["expected_row_counts"]
        and value["frozen_evaluation_manifest"] == p["frozen_evaluation_manifest"],
        "manifest_drift",
    )
    require(
        value["producer_invocation"]["executed_sources"]
        == [reference(m.ROOT / p) for p in m.OWNED],
        "executed_sources",
    )
    require(
        value["source_artifact_hashes"] == value["preconditions_checked"]["references"],
        "source_inventory",
    )
    if value["preconditions_checked"]["external_ready"]:
        for name, (pin, field, expected) in UPSTREAM.items():
            producer_ref: Json = next(
                (
                    r
                    for r in value["source_artifact_hashes"]
                    if Path(r["path"]).stem == name and r["sha256"] == pin
                ),
                {},
            )
            require(bool(producer_ref), "upstream_pin")
            require(
                json.loads(Path(producer_ref["path"]).read_bytes())[field] == expected,
                "upstream_gate",
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
    for key in measured:
        if key != "candidate_ready_score":
            require(value[key] == measured[key], "reduction_drift:" + key)
    completed = sum(r["status"] == "complete" for r in value["rows"])
    failed = sum(r["status"] == "failed" for r in value["rows"])
    require(
        len(value["rows"]) == value["intended_count"] == 6144
        and value["completed_count"] == completed
        and value["censored_count"] == 6144 - completed - failed
        and value["failed_count"] == failed
        and value["excluded_count"] == value["independent_count"] == 0,
        "count_drift",
    )
    if completed == 6144:
        fit = next(r for r in value["source_artifact_hashes"] if Path(r["path"]).name == "fit.json")
        require(
            json.loads((raw / "heads.json").read_bytes())
            == json.loads(Path(fit["path"]).read_bytes())["heads"]["signed"],
            "initial_head_drift",
        )
        with TemporaryDirectory(prefix="carnot8905-replay-") as directory:
            rebuilt = m.measure(Path(directory), json.loads((raw / "heads.json").read_bytes()))
            require(rebuilt == measured, "semantic_drift")
    else:
        require(measured == recover(raw, value["missing_reason"]), "unfinished_drift")
    coverage = value["owned_statement_coverage"]
    valid = bool(
        coverage["complete"]
        and coverage["percent"] == 100
        and not any(r["excluded"] or r["missing"] for r in coverage["files"])
        and all(r["passed"] for r in value["validation_receipts"])
    )
    require(value["required_checks_passed"] == valid, "validation_readiness_drift")
    ready = int(
        valid
        and value["preconditions_checked"]["external_ready"]
        and value["preconditions_checked"]["resources"]["passed"]
        and completed == 6144
        and not value["owned_errors"]
        and all(r["passed"] for r in value["control_rows"])
    )
    require(value["guarded_methods_ready_score"] == ready, "readiness_drift")
    require(
        value["candidate_ready_score"]
        == (
            m.feasibility(value["rows"], value["promotion_rows"])["candidate_ready_score"]
            if ready
            else 0
        ),
        "candidate_readiness_drift",
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
    """Run bounded development work, validate privately, then atomically publish once."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", choices=["20261107"], default="20261107")
    parser.add_argument("--output", type=Path, default=m.ROOT / "results" / (m.NAME + ".json"))
    parser.add_argument("--private-output", type=Path)
    parser.add_argument("--sources", type=Path, default=m.ROOT / "results")
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
        8905,
        "Guarded signed-head update methods",
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
    with TemporaryDirectory(prefix="carnot8905-validation-") as directory:
        validation = (
            unqualified() if args.private_output else validate(raw / "validation", Path(directory))
        )
    valid = bool(
        validation["coverage"]["complete"]
        and validation["coverage"]["percent"] == 100
        and validation["union"]["passed"]
        and all(r["passed"] for r in validation["receipts"])
    )
    ended = time.monotonic_ns()
    m.progress("validation_after", 1, 0)
    spans = [
        dict(phase="preconditions_and_pilots", start_s=0.0, end_s=(measured - started) / 1e9),
        dict(phase="validation", start_s=(measured - started) / 1e9, end_s=(ended - started) / 1e9),
    ]
    completed = sum(r["status"] == "complete" for r in result["rows"])
    failed = sum(r["status"] == "failed" for r in result["rows"])
    ready = int(
        pre["external_ready"]
        and pre["resources"]["passed"]
        and valid
        and not errors
        and completed == 6144
        and all(r["passed"] for r in result["control_rows"])
    )
    verdict = "blocked" if not pre["external_ready"] else "null" if ready else "disqualified"
    p = m.protocol()
    value = dict(
        result,
        experiment_id=8905,
        task_id=m.TASK,
        milestone="2026.11.762",
        run_date=args.date,
        honest_verdict=f"complete_{verdict}_guarded_update_methods",
        verdict_class=verdict,
        inference_substrate="cpu_signed_linear_normalized_energy_head",
        inference_substrate_class="no_model_load"
        if result["numerical_training_calls"]
        else "blocked_no_run",
        execution_venue="host",
        MODEL_SPECS=[],
        verifier_is_oracle=True,
        exposure_scope="constructed development pilots only; fresh evaluation remains unopened",
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
                "python/carnot/reporting/topology_memory_methods_8881.py",
                "scripts/experiment_template.py",
                "python/carnot/reporting/current_work_receipt.py",
                "python/carnot/reporting/primary_publication.py",
            ]
        ],
        raw_shard_hashes=[reference(path) for path in sorted(raw.rglob("*.json"))]
        + validation["raw_references"],
        raw_directory=str(raw),
        intended_count=6144,
        completed_count=completed,
        failed_count=failed,
        censored_count=6144 - completed - failed,
        excluded_count=0,
        independent_count=0,
        gate_check_summary=pre["diagnostics"],
        acceptance_gates=p,
        required_checks_passed=valid,
        validation_receipts=validation["receipts"],
        owned_statement_coverage=validation["coverage"],
        owned_errors=errors,
        guarded_methods_ready_score=ready,
        candidate_ready_score=result["candidate_ready_score"] if ready else 0,
        protocol=p,
        split_manifest=[
            dict(group_id=u["unit_id"], split=m.g.split(u["unit_id"]))
            for plan in p["pilot_manifest"] + p["frozen_evaluation_manifest"]
            for u in plan["online"]
        ],
        anchor_manifest=m.anchors(),
        expected_row_counts=p["expected_row_counts"],
        frozen_evaluation_manifest=p["frozen_evaluation_manifest"],
        missing_reason=reason if completed < 6144 else None,
        cited_upstream_artifacts=[
            dict(producer=n, sha256=h, artifact_field=f, expected=e)
            for n, (h, f, e) in UPSTREAM.items()
        ],
        methodology="Eight paired48-query development streams; identical shadow fitting and guard work;16 fixed development anchors; untouched48-query retention at0/24/48.",
        methodology_note="Oracle supervision and repeatedly reused guards establish no independent generalization or risk certificate. A failing pilot stops evaluation without tuning.",
        flagged_adversarial=False,
        field_principles={},
    )
    health = Path("/tmp/carnot-8905-repository-health/repository_health_8905.receipt.json")
    if health.is_file():
        record = json.loads(health.read_bytes())
        for key in ("stdout_path", "stderr_path"):
            target = raw / ("repository_health." + key.split("_")[0])
            shutil.copyfile(record[key], target)
            value["raw_shard_hashes"].append(reference(target))
        value["repository_health"] = record

    def refresh_receipt(final: int) -> None:
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
            duration_s=receipt["duration_s"],
            phase_spans=spans,
            current_work_receipt=receipt,
            model_invocation_counts=receipt["invocation_counts"],
        )

    refresh_receipt(ended)
    value = normalize_artifact_for_template_write(value)
    value["field_principles"] = {
        k: "Actual producer metadata or byte-bound development evidence; no independent natural-language claim."
        for k in value
    }
    value["field_principles"].update(
        guarded_methods_ready_score="Causal controls and validated execution qualify methods independently of effect.",
        candidate_ready_score="At least one guarded_true promotion, future Brier mean<=-0.01 and every retention mean increase<=0.01.",
        protocol="Fixed before pilots; repeated operational guards provide no independent risk certificate.",
        numerical_training_calls="Actual per-example unchanged signed-head SGD steps across all four arms; no generator training.",
        frozen_evaluation_manifest="24 fresh stream recipes and18432 expected rows remain unopened.",
        independent_count="Zero fresh evaluation streams opened; eight pilot streams support development feasibility only.",
        reproducibility_checksum="Canonical artifact hash plus cold reconstruction of predictions, releases and promotions.",
    )
    value["reproducibility_checksum"] = canonical_hash(value)
    candidate = raw / "prospective.json"
    atomic_json(candidate, value)
    if not args.private_output:
        m.progress("terminal_before", 0, 3)
        reports = terminal(candidate, raw)
        value["validation_receipts"] += reports
        value["raw_shard_hashes"] += [
            reference(Path(r[k + "_path"])) for r in reports for k in ("stdout", "stderr")
        ]
        if not all(r["passed"] for r in reports):
            value.update(
                honest_verdict="complete_disqualified_guarded_update_methods",
                verdict_class="disqualified",
                guarded_methods_ready_score=0,
                candidate_ready_score=0,
                required_checks_passed=False,
            )
        final = time.monotonic_ns()
        spans.append(
            dict(phase="terminal", start_s=(ended - started) / 1e9, end_s=(final - started) / 1e9)
        )
        refresh_receipt(final)
        value.pop("reproducibility_checksum")
        value["reproducibility_checksum"] = canonical_hash(value)
        m.progress("terminal_after", 3, 0)
    require((time.monotonic_ns() - started) / 1e9 < 4200, "total_work_budget")
    publish_primary(output, value, replay)
    m.progress("published_" + value["verdict_class"], completed, 6144 - completed)
    return 0
