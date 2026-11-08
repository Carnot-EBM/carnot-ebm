"""REQ-REPORT-8920: authenticate, qualify and atomically publish local proof evidence."""

import argparse
import json
import os
from pathlib import Path
import shutil
import sys
from tempfile import TemporaryDirectory
import time
from typing import Any

from carnot.reporting import label_demand_methods_8920 as m
from carnot.reporting import selective_methods_runner_8829 as checks
from carnot.reporting import topology_memory_runner_8881 as terminal
from carnot.reporting.current_work_receipt import (
    atomic_json,
    canonical_hash,
    sha256_file,
    build_current_work_receipt,
    validate_current_work_receipt,
)
from carnot.reporting.finite_budget_runner_8906 import require
from carnot.reporting.primary_publication import publish_primary
from carnot.reporting.selective_runner_8816 import unqualified
from carnot.reporting.v746_retained_evidence import reference
from carnot.reporting.v748_collection_contract import namespace
from scripts.experiment_template import ExperimentTemplate

Json = dict[str, Any]
SOURCE = "experiment_8910_v762_finite_budget_evaluation"
PIN = "sha256:b0082cd2769f213053773133b71b845adce81473abd4fc9a6b7c35bf1794b971"
FIELD = "certificate_rows_ready_score"
HEALTH = Path("/tmp/carnot-8920-work/repository-health.json")


def preconditions(raw: Path, sources: Path) -> Json:
    """Authenticate original producer assertions and all retained operands before proof work."""
    raw.mkdir(parents=True, exist_ok=True, mode=0o700)
    path = sources / (SOURCE + ".json")
    diagnostic: Json = dict(
        check="missing_file",
        upstream=SOURCE,
        artifact_field=FIELD,
        expected=1,
        observed=None,
        path=str(path),
        hash=None,
    )
    refs, diagnostics = [], []
    try:
        diagnostic["hash"] = sha256_file(path)
        diagnostic["check"] = "rejected_producer"
        value = json.loads(path.read_bytes())
        require(isinstance(value, dict), "producer_object")
        diagnostic.update(check="missing_field", observed=value.get(FIELD))
        score = value[FIELD]
        diagnostic["check"] = "observed_zero"
        require(type(score) is int and score == 1, "readiness_zero")
        diagnostic.update(
            check="rejected_producer",
            artifact_field="sha256",
            expected=PIN,
            observed=diagnostic["hash"],
        )
        require(
            diagnostic["hash"] == PIN
            and value["experiment_id"] == 8910
            and value["task_id"] == "exp8910-finite-budget-evaluation"
            and value["verdict_class"] == "null"
            and value["required_checks_passed"] is True,
            "original_producer",
        )
        refs.append(reference(path))
        operands = value["code_config_hashes"] + value["raw_shard_hashes"]
        for i, ref in enumerate(operands):
            diagnostic.update(
                path=ref["path"], expected=ref["sha256"], artifact_field="retained_operand"
            )
            diagnostic["hash"] = diagnostic["observed"] = sha256_file(Path(ref["path"]))
            require(diagnostic["observed"] == ref["sha256"], "operand_drift")
            refs.append(ref)
            if i % 50 == 0:
                m.progress("authenticate_operands", i, len(operands) - i)
        shutil.copyfile(path, raw / "original_producer.json")
    except (OSError, ValueError, KeyError, TypeError) as error:
        diagnostics.append(
            dict(
                diagnostic,
                error=str(error),
                check="missing_file"
                if isinstance(error, OSError)
                else "missing_field"
                if isinstance(error, KeyError)
                else diagnostic["check"],
            )
        )
    return dict(
        external_ready=not diagnostics,
        diagnostics=diagnostics,
        references=refs,
        resources=dict(passed=shutil.disk_usage(raw).free > 128 * 1024 * 1024),
    )


def validate(raw: Path, private: Path) -> Json:
    """Reuse exact-node unit/child coverage, strict lint/types and private E2E018/021."""
    runner = namespace(checks, "label8920_validation")
    runner.m = m
    return dict(runner.validate(raw, private))


def replay(path: Path) -> Json:
    """Recompute proof operands so even rehashed semantic tampering cannot grant readiness."""
    value = json.loads(path.read_bytes())
    checksum = value.pop("reproducibility_checksum")
    require(canonical_hash(value) == checksum, "artifact_checksum")
    require(
        (value["experiment_id"], value["task_id"], value["milestone"], value["run_date"])
        == (8920, m.TASK, "2026.11.763", "20261108"),
        "producer_identity",
    )
    require(
        canonical_hash(value["protocol"]) == canonical_hash(m.protocol())
        and value["protocol_hash"] == canonical_hash(m.protocol()),
        "protocol_drift",
    )
    for ref in (
        value["source_artifact_hashes"] + value["code_config_hashes"] + value["raw_shard_hashes"]
    ):
        require(sha256_file(Path(ref["path"])) == ref["sha256"], "operand_hash")
    require(not validate_current_work_receipt(value["current_work_receipt"]), "current_receipt")
    require(
        value["verifier_is_oracle"] is True
        and value["independent_count"] == 0
        and value["independent_generalization_score"] == value["numerical_training_calls"] == 0
        and value["duration_s"] == value["current_work_receipt"]["duration_s"]
        and value["model_invocation_counts"] == value["current_work_receipt"]["invocation_counts"],
        "compute_scope_drift",
    )
    require(
        value["intended_count"] == len(value["rows"])
        and value["excluded_count"] == 0
        and sum(value[s + "_count"] for s in ["completed", "failed", "censored"])
        == len(value["rows"])
        and all(
            value[s + "_count"] == sum(x["status"] == s for x in value["rows"])
            for s in ["completed", "failed", "censored"]
        ),
        "count_drift",
    )
    require(value["verdict_class"] in ["null", "blocked", "disqualified"], "oracle_scope")
    if value["completed_count"]:
        pre = value["preconditions_checked"]
        require(
            pre["external_ready"]
            and pre["resources"]["passed"]
            and not pre["diagnostics"]
            and not value["gate_check_summary"],
            "producer_custody",
        )
        producer = next(
            (
                ref
                for ref in value["source_artifact_hashes"]
                if Path(ref["path"]).name == SOURCE + ".json" and ref["sha256"] == PIN
            ),
            None,
        )
        if producer is None:
            raise ValueError("producer_custody")
        original = json.loads(Path(producer["path"]).read_bytes())
        require(
            all(
                ref in value["source_artifact_hashes"]
                for ref in original["code_config_hashes"] + original["raw_shard_hashes"]
            ),
            "producer_custody",
        )
        result = m.qualify()
        require(
            all(canonical_hash(value[k]) == canonical_hash(v) for k, v in result.items()),
            "reduction_drift",
        )
        require(
            json.loads(Path(value["frozen_draw_manifest"]["path"]).read_bytes())
            == m.draw_manifest(),
            "manifest_drift",
        )
    cov = value["owned_statement_coverage"]
    valid = (
        cov["complete"]
        and cov["percent"] == 100
        and {x["path"] for x in cov["files"]} == set(m.OWNED)
        and all(not x["missing"] and not x["excluded"] for x in cov["files"])
        and all(x["passed"] for x in value["validation_receipts"])
    )
    require(
        type(value["label_demand_ready_score"]) is int
        and value["label_demand_ready_score"]
        == int(
            value["completed_count"] > 0
            and value["proof_obligations"]["passed"]
            and value["required_checks_passed"]
            and valid
            and value["verdict_class"] == "null"
        ),
        "readiness_drift",
    )
    require(not value["required_checks_passed"] or valid, "qualification_drift")
    require(value["verdict_class"] != "blocked" or value["gate_check_summary"], "gate_diagnostic")
    return dict(passed=True, status="replay_passed", sha256=sha256_file(path))


def main(argv: list[str] | None = None) -> int:
    """Publish terminal evidence only after measured owned checks and unchanged validators."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", choices=["20261108"], default="20261108")
    parser.add_argument("--output", type=Path, default=m.ROOT / "results" / (m.NAME + ".json"))
    parser.add_argument("--private-output", type=Path)
    parser.add_argument("--sources", type=Path, default=m.ROOT / "results")
    parser.add_argument("--cold-replay", "--replay", dest="replay", type=Path)
    args = parser.parse_args(argv)
    m.progress("start_no_model_load")
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
    template = ExperimentTemplate(
        8920,
        "Exact label demand methods",
        "results/" + m.NAME + ".json",
        repo_root=m.ROOT,
        seed=m.SEED,
    )
    m.progress("before_preconditions")
    pre = preconditions(raw, args.sources)
    atomic_json(raw / "preconditions.json", pre)
    atomic_json(raw / "protocol.json", m.protocol())
    atomic_json(
        raw / "invocation.json",
        dict(
            executable=sys.executable,
            pid=os.getpid(),
            arguments=argv if argv is not None else sys.argv[1:],
            cwd=os.getcwd(),
            executed_sources=[reference(m.ROOT / p) for p in m.OWNED],
        ),
    )
    result: Json = dict(
        rows=[
            dict(
                m.frozen.row("proof_not_run", False),
                seed=m.SEED,
                status="censored",
                missing_reason="external_precondition",
            )
        ],
        proof_rows=[],
        label_access_controls={},
        alpha_ledger=m.frozen.alpha_ledger(),
        proof_obligations=dict(proof=m.PROOF, passed=False),
        numerical_training_calls=0,
    )
    manifest, errors = None, []
    m.progress("after_preconditions_before_proof")
    if pre["external_ready"] and pre["resources"]["passed"]:
        try:
            with template.phase("label_demand_proof"):
                result = m.qualify()
                require(result["proof_obligations"]["passed"], "owned_proof_failure")
                atomic_json(raw / "proof.json", result)
                atomic_json(raw / "frozen_draw_manifest.json", m.draw_manifest())
                manifest = reference(raw / "frozen_draw_manifest.json")
        except (OSError, ValueError, KeyError, TypeError, RuntimeError) as error:
            errors.append(str(error))
            result["rows"] = [
                dict(
                    m.frozen.row("owned_failure", False),
                    seed=m.SEED,
                    status="failed",
                    missing_reason=str(error),
                )
            ]
            result["proof_obligations"]["passed"] = False
    measured = time.monotonic_ns()
    m.progress("after_proof_before_owned_validation")
    with TemporaryDirectory(prefix="exp8920-validation-") as directory:
        validation = (
            unqualified() if args.private_output else validate(raw / "validation", Path(directory))
        )
    valid = bool(
        validation["coverage"]["complete"]
        and validation["union"]["passed"]
        and all(x["passed"] for x in validation["receipts"])
    )
    m.progress("after_owned_validation")
    require((time.monotonic_ns() - started) / 1e9 < 4200, "validation_deadline")
    verdict = (
        "disqualified"
        if errors or not pre["resources"]["passed"]
        else "blocked"
        if not pre["external_ready"]
        else "null"
        if valid
        else "disqualified"
    )
    ended = time.monotonic_ns()
    phases = [
        dict(phase="authentication_and_proof", start_s=0, end_s=(measured - started) / 1e9),
        dict(
            phase="owned_validation",
            start_s=(measured - started) / 1e9,
            end_s=(ended - started) / 1e9,
        ),
    ]
    receipt = build_current_work_receipt(
        run_id=str(started),
        owner_pid=os.getpid(),
        events=[],
        inference_substrate="cpu_exact_label_demand_proof",
        inference_substrate_class="no_model_load" if result["proof_rows"] else "blocked_no_run",
        inference_substrate_details=dict(pretrained_loads=0, numerical_training_calls=0),
        execution_venue="host",
        started_monotonic_ns=started,
        ended_monotonic_ns=ended,
        phase_spans=phases,
    )
    atomic_json(raw / "current_work_receipt.json", receipt)
    health = json.loads(HEALTH.read_bytes()) if HEALTH.is_file() else None
    if health:
        require(
            sha256_file(Path(health["log"]["path"])) == health["log"]["sha256"], "health_log_drift"
        )
        shutil.copyfile(health["log"]["path"], raw / "repository_health.log")
        health["log"] = reference(raw / "repository_health.log")
    value = dict(
        result,
        experiment_id=8920,
        task_id=m.TASK,
        milestone="2026.11.763",
        run_date=args.date,
        honest_verdict=f"complete_{verdict}_label_demand_methods",
        verdict_class=verdict,
        inference_substrate=receipt["inference_substrate"],
        inference_substrate_class=receipt["inference_substrate_class"],
        execution_venue="host",
        MODEL_SPECS=[],
        model_invocation_counts=receipt["invocation_counts"],
        verifier_is_oracle=True,
        exposure_scope="constructed proof truth only; fresh labels unopened",
        independent_generalization_score=0,
        duration_s=receipt["duration_s"],
        total_wall_time_s=receipt["duration_s"],
        phase_spans=phases,
        preconditions_checked=pre,
        random_seed=m.SEED,
        current_work_receipt=receipt,
        source_artifact_hashes=pre["references"],
        code_config_hashes=[
            reference(m.ROOT / p)
            for p in m.OWNED
            + m.TESTS
            + [
                "scripts/experiment_template.py",
                "python/carnot/reporting/current_work_receipt.py",
                "python/carnot/reporting/primary_publication.py",
                "python/carnot/reporting/finite_budget_methods_8906.py",
                "python/carnot/reporting/shared_terminal_methods_8893.py",
            ]
        ],
        raw_shard_hashes=[reference(p) for p in raw.rglob("*.json")]
        + validation["raw_references"]
        + ([reference(raw / "repository_health.log")] if health else []),
        intended_count=len(result["rows"]),
        completed_count=sum(x["status"] == "completed" for x in result["rows"]),
        failed_count=sum(x["status"] == "failed" for x in result["rows"]),
        censored_count=sum(x["status"] == "censored" for x in result["rows"]),
        excluded_count=0,
        independent_count=0,
        gate_check_summary=pre["diagnostics"],
        owned_errors=errors,
        acceptance_gates=dict(
            owned_statement_coverage=100,
            exact_paths=True,
            zero_inclusion_violations=True,
            savings_required=False,
        ),
        required_checks_passed=valid,
        validation_receipts=validation["receipts"],
        owned_statement_coverage=validation["coverage"],
        label_demand_ready_score=int(bool(result["proof_rows"]) and valid and verdict == "null"),
        cited_upstream_artifacts=[dict(producer=SOURCE, sha256=PIN, readiness_field=FIELD)],
        frozen_draw_manifest=manifest,
        protocol=m.protocol(),
        protocol_hash=canonical_hash(m.protocol()),
        repository_health=health,
        target_model="seven frozen policies; no model load or training",
        methodology="Exact accepted-label induction and exhaustive finite masks; original V762 binomial tests.",
        methodology_note="Local readiness needs exactness, leakage controls and frozen protocol, never observed savings.",
        flagged_adversarial=False,
    )
    value["field_principles"] = {
        k: "Frozen or measured local proof operand; no empirical savings or natural generalization claim."
        for k in value
    }
    value["field_principles"].update(
        label_demand_ready_score="Exact proof, guarded label access, frozen protocol and owned checks; no savings gate.",
        frozen_draw_manifest="Forty unopened paired draws with independent seeds; repeated source draws remain paid.",
        proof_rows="Exhaustive masks and binary truth paths are deterministic controls; independent n is zero.",
        label_access_controls="Unpaid reads and premature census denied; evaluator census separate from paid repeats.",
        duration_s="Actual authentication, proof and owned-check wall time; terminal validator overhead retained in receipts.",
        total_wall_time_s="Actual elapsed time through terminal checks, before atomic publication replay.",
        cheap_policy_evaluations="Source-visible active-policy candidate checks; paid labels reported separately in retained controls.",
    )
    value["reproducibility_checksum"] = canonical_hash(value)
    candidate = raw / "prospective.json"
    atomic_json(candidate, value)
    if not args.private_output:
        m.progress("before_terminal_subprocesses")
        runner = namespace(terminal, "label8920_terminal")
        runner.m = m
        reports = runner.terminal(candidate, raw)
        value["validation_receipts"] += reports
        value["raw_shard_hashes"] += [
            reference(Path(x[s + "_path"])) for x in reports for s in ["stdout", "stderr"]
        ]
        if not all(x["passed"] for x in reports):
            value.update(
                honest_verdict="complete_disqualified_label_demand_methods",
                verdict_class="disqualified",
                label_demand_ready_score=0,
                required_checks_passed=False,
            )
        value["total_wall_time_s"] = (time.monotonic_ns() - started) / 1e9
        value.pop("reproducibility_checksum")
        value["reproducibility_checksum"] = canonical_hash(value)
        m.progress("after_terminal_subprocesses")
    require((time.monotonic_ns() - started) / 1e9 < 4800, "hard_deadline")
    publish_primary(output, value, replay)
    m.progress(
        "published_" + value["verdict_class"], value["completed_count"], value["censored_count"]
    )
    return 0
