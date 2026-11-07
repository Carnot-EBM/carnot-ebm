"""REQ-REPORT-8884: execute the frozen census and expose only checked terminal bytes."""

import argparse
import gzip
from collections import Counter
import json
import os
from pathlib import Path
import shutil
import sys
from tempfile import TemporaryDirectory
import time
from typing import Any

from carnot.reporting import topology_causal_streams_8884 as m
from carnot.reporting import topology_memory_runner_8881 as prior
from carnot.reporting import conditional_risk_runner_8868 as inherited
from carnot.reporting.current_work_receipt import (
    atomic_json,
    canonical_hash,
    sha256_file,
    build_current_work_receipt,
    validate_current_work_receipt,
)
from carnot.reporting.primary_publication import publish_primary
from carnot.reporting.selective_runner_8816 import unqualified, stream
from carnot.reporting.v746_retained_evidence import reference, load
from carnot.reporting.v748_collection_contract import namespace
from scripts.experiment_template import ExperimentTemplate, normalize_artifact_for_template_write

Json = dict[str, Any]
SOURCE = "experiment_8881_v760_topology_memory_methods"
PIN = "sha256:4d43ee0ea01ae703efb1087b5a2f2a253c529f46d3e2c9d7f87ff2b028d3c1e3"


def preconditions(raw: Path, sources: Path) -> Json:
    """Authenticate the original producer, operands and execution authority before fitting."""
    began = time.monotonic()
    checker = namespace(inherited, "causal8884_preconditions")
    checker.UPSTREAM = {SOURCE: (PIN, "topology_methods_ready_score")}
    checker.DESIGN = "openspec/change-proposals/research-methods-v760-20261106.json"
    checker.DESIGN_PIN = "sha256:c99296bd441c10200bad04e77643649bd65f62fb2199029302b24473af6da3d7"
    pre = dict(checker.preconditions(raw, sources))
    pre["original_validation_passed"] = False
    if pre["external_ready"]:
        source = sources / (SOURCE + ".json")
        value = json.loads(source.read_bytes())
        accepted = (
            value["task_id"] == m.base.TASK
            and value["required_checks_passed"] is True
            and value["owned_statement_coverage"]["complete"]
            and value["owned_statement_coverage"]["percent"] == 100
            and all(r["passed"] for r in value["control_rows"])
            and canonical_hash(value["protocol"]) == canonical_hash(m.base.protocol())
        )
        pre["original_validation_passed"] = bool(accepted)
        if not accepted:
            pre["external_ready"] = False
            pre["diagnostics"].append(
                dict(
                    check="rejected_producer",
                    upstream=SOURCE,
                    artifact_field="protocol_and_original_validation",
                    expected="frozen protocol and qualified controls",
                    observed=False,
                    path=str(source),
                    sha256=sha256_file(source),
                )
            )
    pre["duration_s"] = time.monotonic() - began
    return pre


def retained_stream(raw: Path, plan: Json, heads: Json, protocol: Json) -> Json:
    """Retain exact interrupted states while keeping mutable checkpoints in private scratch."""
    with TemporaryDirectory(prefix="carnot8884-state-", dir="/tmp") as directory:
        try:
            return m.run_stream(Path(directory), plan, heads, protocol)
        finally:
            shutil.copytree(directory, raw / "states" / plan["stream_id"])


def measurement(raw: Path, pre: Json) -> Json:
    """Qualify the frozen pilots without inspecting target benefit or changing the census."""
    began = time.monotonic()
    protocol = m.base.protocol()
    atomic_json(raw / "protocol.json", protocol)
    m.progress("fit_before", 0, 5)
    fit = m.base.fit(protocol)
    atomic_json(raw / "fit.json", fit)
    m.progress("fit_after", 5, 0)
    controls, calls = m.base.sensitivity()
    atomic_json(raw / "sensitivity.json", dict(rows=controls, numerical_training_calls=calls))
    checks = [dict(check=r["control"], passed=r["passed"]) for r in controls]
    pilot_start = time.monotonic()
    for plan in protocol["pilot_manifest"]:
        result = retained_stream(raw, plan, fit["heads"], protocol)
        atomic_json(raw / ("pilot_" + plan["stream_id"] + ".json"), result)
        checks += result["controls"]
    projected = (
        (time.monotonic() - pilot_start) * 24
        + (time.monotonic() - began)
        + pre.get("duration_s", 0)
        + 1200
    )
    atomic_json(raw / "controls.json", dict(rows=checks, projected_runtime_s=projected))
    m.require(all(c["passed"] for c in checks), "owned_pilot_control_failure")
    m.require(projected < 4200, "owned_runtime_projection")
    for i, plan in enumerate(protocol["cohort_manifest"]):
        m.require(time.monotonic() - began < 2400, "measurement_deadline")
        m.progress("cohort_before", i, 24 - i)
        result = retained_stream(raw, plan, fit["heads"], protocol)
        atomic_json(raw / ("stream_" + plan["stream_id"] + ".json"), result)
        m.progress("cohort_after", i + 1, 23 - i)
    return recover(raw, "unfinished")


def recover(raw: Path, reason: str) -> Json:
    """Keep every scheduled failure and completed unit without selecting retries or seeds."""
    streams = [load(reference(p)) for p in sorted(raw.glob("stream_*.json"))]
    result = m.reduce_streams(streams)
    completed = {(r["unit_id"], r["arm"]): r for r in result["rows"]}
    result["rows"] = [
        completed.get((r["unit_id"], r["arm"]), r) for r in m.base.census(m.base.protocol(), reason)
    ]
    result["stream_refs"] = [reference(p) for p in sorted(raw.glob("stream_*.json"))]
    checks = load(reference(raw / "controls.json")) if (raw / "controls.json").is_file() else {}
    result.update(
        control_rows=checks.get("rows", []),
        projected_runtime_s=checks.get("projected_runtime_s", 0),
    )
    result["numerical_training_calls"] = sum(
        load(reference(raw / name))["numerical_training_calls"]
        for name in ["fit.json", "sensitivity.json"]
        if (raw / name).is_file()
    )
    result["numerical_training_calls"] += sum(
        len(json.loads(p.read_bytes())["acks"]) for p in raw.glob("states/**/matched_*.json")
    )
    return result


def validate(raw: Path, private: Path) -> Json:
    """Reuse original exact-node tests, retained child coverage and private E2E controls."""
    checker = namespace(prior, "causal8884_validation")
    checker.m = m
    result = dict(checker.validate(raw, private))
    result["receipts"].append(e2e018(raw / "e2e018", private / "e2e018"))
    result["raw_references"] += [
        reference(p) for p in sorted((raw / "e2e018").rglob("*")) if p.is_file()
    ]
    return result


def e2e018(raw: Path, private: Path) -> Json:
    """Run all original authority checks on fixture copies without changing live roadmaps."""
    private.mkdir(parents=True)
    fixture = m.ROOT / "tests/fixtures/v685"
    (private / "design.md").write_bytes((fixture / "design.md").read_bytes())
    active = gzip.decompress((fixture / "active.yaml.gz").read_bytes())
    for name in ("active.yaml", "staged.yaml"):
        (private / name).write_bytes(active)
    report = stream(
        "original_e2e018_cli",
        [
            sys.executable,
            "-u",
            str(m.ROOT / "scripts/experiments/experiment_7891_v685_authority_lifecycle.py"),
            "--date",
            "20260929",
            "--design",
            str(private / "design.md"),
            "--active",
            str(private / "active.yaml"),
            "--staged",
            str(private / "staged.yaml"),
            "--output",
            str(private / "artifact.json"),
            "--raw",
            str(private / "rows.json"),
        ],
        raw,
        deadline=600,
    )
    artifact = (
        json.loads((private / "artifact.json").read_bytes())
        if (private / "artifact.json").is_file()
        else {}
    )
    validators = (
        json.loads((private / "terminal_validator_reports.json").read_bytes())
        if (private / "terminal_validator_reports.json").is_file()
        else {}
    )
    report["original_checks_passed"] = bool(
        artifact.get("contract_ready_score") == 1
        and artifact.get("validation_receipts")
        and all(r["passed"] for r in artifact["validation_receipts"])
        and len(artifact.get("mutation_rows", [])) == 12
        and all(r["passed"] for r in artifact["mutation_rows"])
        and validators.get("passed") is True
    )
    report["passed"] = report["passed"] and report["original_checks_passed"]
    if report["passed"]:
        cold = stream(
            "original_e2e018_cold",
            [
                "env",
                "-u",
                "PYTHONPATH",
                "--chdir=/tmp",
                sys.executable,
                "-u",
                str(m.ROOT / "scripts/experiments/experiment_7891_v685_authority_lifecycle.py"),
                "--cold-replay",
                str(private / "artifact.json"),
                "--raw",
                str(private / "rows.json"),
            ],
            raw,
            deadline=120,
        )
        report["cold_replay_receipt"] = cold
        report["passed"] = cold["passed"]
    shutil.copytree(private, raw / "retained_private_evidence")
    return report


def terminal(candidate: Path, raw: Path) -> list[Json]:
    """Use unchanged scanners and real outside-checkout replay with fixed child deadlines."""
    checker = namespace(prior, "causal8884_terminal")
    checker.m = m
    return list(checker.terminal(candidate, raw))


def replay(path: Path, *, cold: bool = True) -> Json:
    """Recompute original rows and causal histories; rehashing tampered claims is insufficient."""
    value = json.loads(path.read_bytes())
    checksum = value.pop("reproducibility_checksum")
    m.require(canonical_hash(value) == checksum, "artifact_checksum")
    m.require(
        (value["experiment_id"], value["task_id"], value["milestone"], value["run_date"])
        == (8884, m.TASK, "2026.11.760", "20261106"),
        "producer_identity",
    )
    m.require(
        canonical_hash(value["protocol"]) == canonical_hash(m.base.protocol()), "protocol_drift"
    )
    for index, ref in enumerate(
        value["source_artifact_hashes"] + value["code_config_hashes"] + value["raw_shard_hashes"]
    ):
        m.require(sha256_file(Path(ref["path"])) == ref["sha256"], "operand_hash")
        if index % 1000 == 0:
            m.progress("replay_authenticate", index, 0)
    m.require(not validate_current_work_receipt(value["current_work_receipt"]), "current_receipt")
    receipt = value["current_work_receipt"]
    m.require(
        value["duration_s"] == receipt["duration_s"]
        and value["phase_spans"] == receipt["phase_spans"]
        and value["model_invocation_counts"] == receipt["invocation_counts"]
        and value["inference_substrate_class"] == receipt["inference_substrate_class"]
        and value["numerical_training_calls"]
        == receipt["small_ebm_training"]["numerical_training_calls"],
        "compute_scope",
    )
    m.require(
        value["acceptance_gates"]
        == dict(
            complete_rows=18432,
            original_controls=True,
            owned_statement_coverage=100,
            independent_benefit_owner="exp8886",
        ),
        "acceptance_drift",
    )
    raw = Path(value["raw_directory"])
    rebuilt = recover(raw, value["censor_reason"])
    for key, expected in rebuilt.items():
        m.require(canonical_hash(value[key]) == canonical_hash(expected), "reduction_drift:" + key)
    counts = Counter(r["status"] for r in value["rows"])
    m.require(
        len(value["rows"]) == value["intended_count"] == 18432
        and value["completed_count"] == counts["complete"]
        and value["failed_count"] == counts["failed"]
        and value["censored_count"] == counts["censored"]
        and value["excluded_count"] == 0,
        "count_drift",
    )
    if counts["complete"] == 18432:
        m.require(sha256_file(raw / (SOURCE + ".json")) == PIN, "original_producer_drift")
        if cold:
            with TemporaryDirectory(prefix="carnot8884-replay-", dir="/tmp") as directory:
                protocol = m.base.protocol()
                fit = m.base.fit(protocol)
                m.require(
                    canonical_hash(fit) == canonical_hash(load(reference(raw / "fit.json"))),
                    "fit_drift",
                )
                controls, calls = m.base.sensitivity()
                m.require(
                    dict(rows=controls, numerical_training_calls=calls)
                    == load(reference(raw / "sensitivity.json")),
                    "control_drift",
                )
                for plan in protocol["pilot_manifest"] + protocol["cohort_manifest"]:
                    prefix = "pilot_" if plan["pilot"] else "stream_"
                    original = load(reference(raw / (prefix + plan["stream_id"] + ".json")))
                    actual = m.run_stream(
                        Path(directory) / plan["stream_id"], plan, fit["heads"], protocol
                    )
                    m.require(
                        canonical_hash(m.semantic(original)) == canonical_hash(m.semantic(actual)),
                        "causal_stream_drift",
                    )
    ready = int(
        counts["complete"] == 18432
        and value["required_checks_passed"]
        and value["verdict_class"] == "null"
    )
    m.require(
        value["causal_rows_ready_score"] == ready
        and value["independent_count"] == (24 if counts["complete"] == 18432 else 0),
        "readiness_drift",
    )
    m.require(
        value["verifier_is_oracle"] is True
        and value["independent_generalization_score"] == 0
        and value["verdict_class"] != "positive",
        "oracle_scope",
    )
    if ready:
        coverage = value["owned_statement_coverage"]
        m.require(
            coverage["complete"]
            and coverage["percent"] == 100
            and {r["path"] for r in coverage["files"]} == set(m.OWNED)
            and all(not r["missing"] and not r["excluded"] for r in coverage["files"])
            and all(r["passed"] for r in value["control_rows"] + value["validation_receipts"]),
            "qualification_drift",
        )
    return dict(passed=True, status="replay_passed", sha256=sha256_file(path))


def main(argv: list[str] | None = None) -> int:
    """Publish one terminal primary after measured validation; never activate a milestone."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", choices=["20261106"], default="20261106")
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
    template = ExperimentTemplate(
        8884,
        "Topology causal streams",
        "results/" + m.NAME + ".json",
        repo_root=m.ROOT,
        seed=m.base.SEED,
    )
    invocation = dict(
        executable=sys.executable,
        arguments=argv if argv is not None else sys.argv[1:],
        cwd=os.getcwd(),
        pid=os.getpid(),
        executed_sources=[reference(m.ROOT / p) for p in m.OWNED],
    )
    atomic_json(raw / "invocation.json", invocation)
    m.progress("preconditions_before", 0, 1)
    pre = preconditions(raw, args.sources)
    atomic_json(raw / "preconditions.json", pre)
    m.progress("preconditions_after", 1, 0)
    reason = (
        "external_input_unavailable" if not pre["external_ready"] else "owned_resources_unavailable"
    )
    result = recover(raw, reason)
    errors = []
    if pre["external_ready"] and pre["resources"]["passed"]:
        try:
            with template.phase("measurement"):
                result = measurement(raw, pre)
            reason = "unfinished"
        except (OSError, ValueError, KeyError, RuntimeError) as error:
            errors.append(str(error))
            reason = str(error)
            result = recover(raw, reason)
    measured = time.monotonic_ns()
    m.progress("validation_before", 0, 1)
    with TemporaryDirectory(prefix="carnot8884-validation-", dir="/tmp") as directory:
        try:
            validation = (
                unqualified()
                if args.private_output
                else validate(raw / "validation", Path(directory))
            )
        except (OSError, ValueError, KeyError, RuntimeError) as error:
            validation = unqualified()
            validation["receipts"][0].update(name="owned_validation_failure", error=str(error))
            errors.append(str(error))
    m.progress("validation_after", 1, 0)
    valid = bool(
        validation["coverage"]["complete"]
        and validation["union"]["passed"]
        and all(r["passed"] for r in validation["receipts"])
    )
    completed = sum(r["status"] == "complete" for r in result["rows"])
    verdict = (
        "blocked"
        if not pre["external_ready"]
        else "null"
        if valid and completed == 18432 and not errors
        else "disqualified"
    )
    ended = time.monotonic_ns()
    spans = [
        dict(phase="preconditions_and_measurement", start_s=0.0, end_s=(measured - started) / 1e9),
        dict(phase="validation", start_s=(measured - started) / 1e9, end_s=(ended - started) / 1e9),
    ]
    receipt = build_current_work_receipt(
        run_id=str(started),
        owner_pid=os.getpid(),
        events=[],
        inference_substrate="cpu_linear_normalized_energy_head",
        inference_substrate_class="no_model_load"
        if result["numerical_training_calls"]
        else "blocked_no_run",
        execution_venue="host",
        inference_substrate_details=dict(backend="JAX CPU", pretrained_loads=0),
        started_monotonic_ns=started,
        ended_monotonic_ns=ended,
        phase_spans=spans,
        small_ebm_training=dict(
            performed=result["numerical_training_calls"] > 0,
            numerical_training_calls=result["numerical_training_calls"],
        ),
    )
    helpers = m.base.OWNED + [
        "python/carnot/memory/categorical_memory_8830.py",
        "python/carnot/memory/causal_memory_8802.py",
        "scripts/experiment_template.py",
        "python/carnot/reporting/current_work_receipt.py",
        "python/carnot/reporting/primary_publication.py",
        "scripts/adversarial_verify.py",
        "scripts/verdict_row_consistency_lint.py",
    ]
    for name in ("carnot8884-tests-first.log", "carnot8884-validation-progress.log"):
        diagnostic = Path("/tmp") / name
        if diagnostic.is_file():
            (raw / "development_diagnostics").mkdir(exist_ok=True)
            shutil.copyfile(diagnostic, raw / "development_diagnostics" / name)
    value = dict(
        result,
        experiment_id=8884,
        task_id=m.TASK,
        milestone="2026.11.760",
        run_date=args.date,
        honest_verdict="complete_" + verdict + "_topology_causal_streams",
        verdict_class=verdict,
        inference_substrate=receipt["inference_substrate"],
        inference_substrate_class=receipt["inference_substrate_class"],
        execution_venue="host",
        MODEL_SPECS=[],
        model_invocation_counts=receipt["invocation_counts"],
        verifier_is_oracle=True,
        exposure_scope="Constructed signed-record oracle; no natural-language generalization; generator weights frozen.",
        independent_generalization_score=0,
        duration_s=receipt["duration_s"],
        phase_spans=spans,
        preconditions_checked=pre,
        random_seed=m.base.SEED,
        producer_invocation=invocation,
        current_work_receipt=receipt,
        source_artifact_hashes=pre["references"],
        code_config_hashes=[reference(m.ROOT / p) for p in m.OWNED + m.TESTS + helpers],
        raw_shard_hashes=[reference(p) for p in sorted(raw.rglob("*")) if p.is_file()]
        + validation["raw_references"],
        raw_directory=str(raw),
        intended_count=18432,
        completed_count=completed,
        failed_count=sum(r["status"] == "failed" for r in result["rows"]),
        censored_count=sum(r["status"] == "censored" for r in result["rows"]),
        excluded_count=0,
        independent_count=24 if completed == 18432 else 0,
        gate_check_summary=[dict(r, hash=r.get("sha256")) for r in pre["diagnostics"]],
        acceptance_gates=dict(
            complete_rows=18432,
            original_controls=True,
            owned_statement_coverage=100,
            independent_benefit_owner="exp8886",
        ),
        required_checks_passed=valid,
        validation_receipts=validation["receipts"],
        owned_statement_coverage=validation["coverage"],
        owned_errors=errors,
        censor_reason=reason,
        causal_rows_ready_score=int(verdict == "null"),
        cited_upstream_artifacts=[
            dict(
                experiment_id=8881,
                task_id=m.base.TASK,
                sha256=PIN,
                artifact_field="topology_methods_ready_score",
                expected=1,
            )
        ],
        protocol=m.base.protocol(),
        independent_benefit_score=0,
        methodology="Execute the exact frozen Exp8881 cohort with matched batch label marginals, admission schedules and optimizer work; read-only retention at0/24/48. Preliminary paired effects are descriptive; Exp8886 owns benefit testing.",
        methodology_note="Record cumulative release delay and maximum simultaneously pending feedback separately, following arXiv:2610.28058. Transaction costs and template insertion do not establish a convex regret bound.",
        cited_research=[
            dict(
                url="https://arxiv.org/abs/2610.28058",
                adopted="Separate cumulative delay from maximum pending feedback",
                inherited_benefit=False,
            )
        ],
        repository_health=None,
        flagged_adversarial=False,
        field_principles={},
    )
    health = Path("/tmp/carnot8884-repository-health.log")
    if health.is_file():
        shutil.copyfile(health, raw / "repository_health.log")
        value["repository_health"] = dict(
            scope="repository_health",
            command=".venv/bin/pytest tests/python -q",
            passed=False,
            exit_code=124,
            timeout_s=240,
            reason="One bounded full-suite health run reported failures and reached its deadline. Separate from owned checks.",
            **reference(raw / "repository_health.log"),
        )
        value["raw_shard_hashes"].append(reference(raw / "repository_health.log"))
    value = normalize_artifact_for_template_write(value)
    value["field_principles"] = {
        k: "Actual producer or retained frozen operand; constructed-query scope only."
        for k in value
    }
    value["field_principles"].update(
        causal_rows_ready_score="Complete causal evidence and original validation release independent audit; no scientific benefit claim.",
        independent_count="24 paired stream RNG trials; recurring templates and five nested seeds do not add independent topology families.",
        numerical_training_calls="Current development fits, sensitivity controls, two pilots and24 target streams only; cold replay and test children are validation work.",
        descriptive_effects="Frozen Exp8881 paired bootstrap estimates are descriptive; Exp8886 independently tests benefit.",
        delay_totals="Sum actual batch release ordinal minus prediction ordinal; configured lower-bound delays are separate.",
        maximum_pending="Maximum simultaneous unacknowledged predictions per arm, including the current sealed prediction before due releases.",
        transaction_costs="Actual state commit time includes file and directory fsync; journal overhead is outside this subtotal and inside total duration.",
        repository_health="One bounded whole-repository run is separate from required100-percent new unit/child statement coverage.",
        reproducibility_checksum="Canonical artifact hash; cold replay independently reconstructs every decision and transaction.",
    )
    value["reproducibility_checksum"] = canonical_hash(value)
    candidate = raw / "prospective.json"
    atomic_json(candidate, value)
    if not args.private_output:
        m.progress("terminal_validation_before", 0, 3)
        reports = terminal(candidate, raw)
        value["validation_receipts"] += reports
        for report in reports:
            for key in ("stdout_path", "stderr_path"):
                if key in report:
                    value["raw_shard_hashes"].append(reference(Path(report[key])))
        if not all(r["passed"] for r in reports):
            value.update(
                honest_verdict="complete_disqualified_topology_causal_streams",
                verdict_class="disqualified",
                causal_rows_ready_score=0,
                required_checks_passed=False,
            )
        terminal_end = time.monotonic_ns()
        spans.append(
            dict(
                phase="terminal_validation",
                start_s=(ended - started) / 1e9,
                end_s=(terminal_end - started) / 1e9,
            )
        )
        receipt = build_current_work_receipt(
            run_id=str(started),
            owner_pid=os.getpid(),
            events=[],
            inference_substrate=value["inference_substrate"],
            inference_substrate_class=value["inference_substrate_class"],
            execution_venue="host",
            inference_substrate_details=dict(backend="JAX CPU", pretrained_loads=0),
            started_monotonic_ns=started,
            ended_monotonic_ns=terminal_end,
            phase_spans=spans,
            small_ebm_training=dict(
                performed=result["numerical_training_calls"] > 0,
                numerical_training_calls=result["numerical_training_calls"],
            ),
        )
        value.update(
            duration_s=receipt["duration_s"], phase_spans=spans, current_work_receipt=receipt
        )
        value.pop("reproducibility_checksum")
        value["reproducibility_checksum"] = canonical_hash(value)
        m.progress("terminal_validation_after", 3, 0)
    publish_primary(output, value, lambda p: replay(p, cold=False))
    for path in raw.rglob("*"):
        if path.is_file():
            path.chmod(0o400)
    m.progress("published_" + value["verdict_class"], completed, 18432 - completed)
    return 0
