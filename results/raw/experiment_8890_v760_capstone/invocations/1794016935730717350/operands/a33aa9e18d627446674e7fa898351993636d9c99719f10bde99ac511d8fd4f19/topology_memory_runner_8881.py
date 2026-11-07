"""REQ-REPORT-8881: retain private causal evidence until terminal validation passes."""

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

from carnot.reporting import topology_memory_methods_8881 as m
from carnot.reporting import conditional_risk_runner_8868 as inherited
from carnot.reporting import diagnosis_qualification_8879 as qualified
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
from carnot.reporting.v746_retained_evidence import reference, load
from carnot.reporting.v748_collection_contract import namespace
from scripts.experiment_template import ExperimentTemplate, normalize_artifact_for_template_write

Json = dict[str, Any]
SOURCE = "experiment_8879_v760_diagnosis_qualification"
PIN = "sha256:b49a603dad46f45d4bfcbc08fd0caa1fc4102b952fa64a6a3c8815a65b81504e"
DEADLINE = 1800


def preconditions(raw: Path, sources: Path) -> Json:
    """Authenticate the repaired diagnosis and preserved execution authority before training."""
    began = time.monotonic()
    checker = namespace(inherited, "topology8881_preconditions")
    checker.UPSTREAM = {SOURCE: (PIN, "diagnosis_ready_score")}
    checker.DESIGN = "openspec/change-proposals/research-methods-v760-20261106.json"
    checker.DESIGN_PIN = "sha256:c99296bd441c10200bad04e77643649bd65f62fb2199029302b24473af6da3d7"
    pre = dict(checker.preconditions(raw, sources))
    pre["replayed_numerical_training_calls"] = 0
    if pre["external_ready"] and pre["resources"]["passed"]:
        source = sources / (SOURCE + ".json")
        m.progress("qualified_diagnosis_replay_before", 0, 1)
        try:
            qualified.replay(source)
            original = load(reference(source))
            state = load(original["raw_shard_hashes"][0])
            manifest = json.loads(Path(state["manifest"]["path"]).read_bytes())
            refs = pre["references"] + [state["manifest"]] + manifest
            pre["retained_original_operands"] = qualified.snapshot(refs, raw)
            pre["replayed_numerical_training_calls"] = original["numerical_training_calls"]
            pre["references"] += pre["retained_original_operands"]
        except (OSError, ValueError, KeyError, TypeError) as error:
            pre["external_ready"] = False
            pre["diagnostics"].append(
                dict(
                    check="missing_file"
                    if isinstance(error, OSError)
                    else "missing_field"
                    if isinstance(error, KeyError)
                    else "rejected_producer",
                    upstream=SOURCE,
                    artifact_field="retained_original_operands",
                    expected="cold replay and exact original bytes",
                    observed=str(error),
                    path=str(source),
                    sha256=sha256_file(source),
                )
            )
        m.progress("qualified_diagnosis_replay_after", 1, 0)
    pre["duration_s"] = time.monotonic() - began
    return pre


def validate(raw: Path, private: Path) -> Json:
    """Measure owned unit/child statements and run the original restart and E2E tests."""
    checker = namespace(checks, "topology8881_validation")
    checker.e = SimpleNamespace(ROOT=m.ROOT, OWNED=m.OWNED, TESTS=m.TESTS)
    checker.original = SimpleNamespace(
        CONSUMERS=[
            "tests/python/test_causal_memory_8802.py",
            "tests/python/test_experiment_7891_v685_authority_lifecycle.py",
            "tests/python/test_restricted_decision_audit_8210.py",
        ]
    )
    checker.child = stream
    with patch.dict(os.environ, TMPDIR=str(private)):
        return dict(checker.validate(raw, private))


def retained_stream(raw: Path, s: Json, heads: Json, p: Json) -> Json:
    """Keep atomic checkpoints private and retain even an interrupted stream's exact bytes."""
    with TemporaryDirectory(prefix="carnot8881-checkpoints-", dir="/tmp") as directory:
        try:
            return m.run_stream(Path(directory), s, heads, p)
        finally:
            shutil.copytree(
                directory, raw / ("pilots" if s["pilot"] else "states") / s["stream_id"]
            )


def measurement(raw: Path, pre: Json) -> Json:
    """Qualify development controls before opening any of the frozen24 paired streams."""
    began = time.monotonic()
    p = m.protocol()
    atomic_json(raw / "protocol.json", p)
    fitted = m.fit(p)
    atomic_json(raw / "fit.json", fitted)
    pilot_start = time.monotonic()
    sensitivity, calls = m.sensitivity()
    atomic_json(raw / "sensitivity.json", dict(rows=sensitivity, numerical_training_calls=calls))
    controls = [dict(check=r["control"], passed=r["passed"]) for r in sensitivity]
    for s in p["pilot_manifest"]:
        result = retained_stream(raw, s, fitted["heads"], p)
        atomic_json(raw / ("pilot_" + s["stream_id"] + ".json"), result)
        controls.extend(result["controls"])
    projected = (
        (time.monotonic() - pilot_start) * 24
        + (time.monotonic() - began)
        + pre.get("duration_s", 0)
        + 1200
    )
    atomic_json(
        raw / "pilot_runtime.json",
        dict(
            projected_runtime_s=projected,
            pilot_duration_s=time.monotonic() - pilot_start,
            validation_allowance_s=1200,
        ),
    )
    atomic_json(raw / "controls.json", dict(rows=controls))
    require(all(r["passed"] for r in controls), "owned_pilot_control_failure")
    require(projected < 4200, "owned_runtime_projection")
    for index, s in enumerate(p["cohort_manifest"]):
        require(time.monotonic() - began < DEADLINE, "measurement_deadline")
        m.progress("stream_before", index, 24 - index)
        result = retained_stream(raw, s, fitted["heads"], p)
        atomic_json(raw / ("stream_" + s["stream_id"] + ".json"), result)
        require(all(r["passed"] for r in result["controls"]), "owned_stream_control_failure")
        m.progress("stream_after", index + 1, 23 - index)
    return reduction(raw, pre)


def reduction(raw: Path, pre: Json) -> Json:
    """Reduce original per-stream rows and preserve their complete paired denominators."""
    p = m.protocol()
    results = [
        load(reference(raw / ("stream_" + s["stream_id"] + ".json"))) for s in p["cohort_manifest"]
    ]
    rows = [r for result in results for r in result["rows"]]
    controls = load(reference(raw / "controls.json"))["rows"]
    return dict(
        rows=rows,
        control_rows=controls,
        stream_refs=[
            reference(raw / ("stream_" + s["stream_id"] + ".json")) for s in p["cohort_manifest"]
        ],
        numerical_training_calls=1200
        + 480
        + 192
        + sum(r["numerical_training_calls"] for r in results)
        + pre["replayed_numerical_training_calls"],
        projected_runtime_s=load(reference(raw / "pilot_runtime.json"))["projected_runtime_s"],
        summary=m.summarize(rows),
    )


def recover(raw: Path, reason: str, pre: Json) -> Json:
    """Retain completed stream files and censor every scheduled unfinished slot."""
    p = m.protocol()
    rows = {r["unit_id"] + "/" + r["arm"]: r for r in m.census(p, reason)}
    refs = []
    for path in sorted(raw.glob("stream_*.json")):
        result = load(reference(path))
        refs.append(reference(path))
        rows.update({r["unit_id"] + "/" + r["arm"]: r for r in result["rows"]})
    controls = (
        load(reference(raw / "controls.json"))["rows"] if (raw / "controls.json").is_file() else []
    )
    calls = pre.get("replayed_numerical_training_calls", 0)
    for name in ["fit.json", "sensitivity.json"]:
        if (raw / name).is_file():
            calls += load(reference(raw / name))["numerical_training_calls"]
    calls += sum(
        len(json.loads(path.read_bytes())["acks"]) for path in raw.glob("**/matched_*.json")
    )
    return dict(
        rows=list(rows.values()),
        stream_refs=refs,
        control_rows=controls,
        numerical_training_calls=calls,
        projected_runtime_s=0,
        summary={},
    )


def terminal(candidate: Path, raw: Path) -> list[Json]:
    """Run unchanged scanners and real foreign-directory cold replay on prospective bytes."""
    commands = [
        (
            "adversarial",
            [
                sys.executable,
                "-u",
                str(m.ROOT / "scripts/adversarial_verify.py"),
                "--json",
                str(candidate),
            ],
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


def replay(path: Path) -> Json:
    """Reconstruct predictions and every transition so rehashed false evidence cannot pass."""
    value = json.loads(path.read_bytes())
    checksum = value.pop("reproducibility_checksum")
    require(canonical_hash(value) == checksum, "artifact_checksum")
    require(
        (value["experiment_id"], value["task_id"], value["milestone"], value["run_date"])
        == (8881, m.TASK, "2026.11.760", "20261106"),
        "producer_identity",
    )
    require(canonical_hash(value["protocol"]) == canonical_hash(m.protocol()), "protocol_drift")
    for key in [
        "arm_definitions",
        "topology_manifest",
        "cohort_manifest",
        "expected_row_counts",
        "frozen_gates",
    ]:
        require(
            canonical_hash(value[key]) == canonical_hash(m.protocol()[key]), "frozen_contract_drift"
        )
    for ref in (
        value["source_artifact_hashes"] + value["code_config_hashes"] + value["raw_shard_hashes"]
    ):
        require(sha256_file(Path(ref["path"])) == ref["sha256"], "operand_hash")
    require(not validate_current_work_receipt(value["current_work_receipt"]), "current_receipt")
    require(
        value["duration_s"] == value["current_work_receipt"]["duration_s"]
        and value["model_invocation_counts"] == value["current_work_receipt"]["invocation_counts"]
        and value["verifier_is_oracle"] is True
        and value["independent_generalization_score"] == 0,
        "compute_scope_drift",
    )
    require(
        value["acceptance_gates"]
        == dict(
            m.GATES,
            methods_controls_required=True,
            owned_statement_coverage=100,
            benefit_required_for_method_readiness=False,
        ),
        "acceptance_drift",
    )
    counts = {
        s: sum(r["status"] == s for r in value["rows"]) for s in ["complete", "failed", "censored"]
    }
    require(
        len(value["rows"]) == 18432
        and sum(counts.values()) == 18432
        and value["completed_count"] == counts["complete"]
        and value["failed_count"] == counts["failed"]
        and value["censored_count"] == counts["censored"]
        and value["excluded_count"] == 0,
        "count_drift",
    )
    raw = Path(value["raw_directory"])
    if counts["complete"] == 18432:
        require(sha256_file(raw / (SOURCE + ".json")) == PIN, "original_producer_drift")
        rebuilt = reduction(raw, value["preconditions_checked"])
        for key, expected in rebuilt.items():
            require(
                canonical_hash(value[key]) == canonical_hash(expected), "reduction_drift:" + key
            )
        with TemporaryDirectory(prefix="carnot8881-replay-") as directory:
            scratch = Path(directory)
            p = m.protocol()
            fitted = m.fit(p)
            require(
                canonical_hash(fitted) == canonical_hash(load(reference(raw / "fit.json"))),
                "fit_drift",
            )
            controls, calls = m.sensitivity()
            require(
                canonical_hash(dict(rows=controls, numerical_training_calls=calls))
                == canonical_hash(load(reference(raw / "sensitivity.json"))),
                "sensitivity_drift",
            )
            for s in p["pilot_manifest"] + p["cohort_manifest"]:
                prefix = "pilot_" if s["pilot"] else "stream_"
                original = load(reference(raw / (prefix + s["stream_id"] + ".json")))
                reproduced = m.run_stream(scratch / s["stream_id"], s, fitted["heads"], p)
                require(
                    canonical_hash(original) == canonical_hash(reproduced), "causal_stream_drift"
                )
    else:
        expected = recover(raw, value["rows"][0]["missing_reason"], value["preconditions_checked"])
        require(
            canonical_hash(value["rows"]) == canonical_hash(expected["rows"]),
            "censored_census_drift",
        )
    ready = int(
        counts["complete"] == 18432
        and value["required_checks_passed"]
        and value["verdict_class"] in {"null", "circular_positive"}
    )
    require(value["topology_methods_ready_score"] == ready, "readiness_drift")
    require(
        value["independent_count"] == (24 if counts["complete"] == 18432 else 0),
        "independence_drift",
    )
    if ready:
        require(all(r["passed"] for r in value["control_rows"]), "owned_control_failure")
        coverage = value["owned_statement_coverage"]
        require(
            coverage["complete"]
            and coverage["percent"] == 100
            and {r["path"] for r in coverage["files"]} == set(m.OWNED)
            and not any(r["missing"] or r["excluded"] for r in coverage["files"])
            and all(r["passed"] for r in value["validation_receipts"]),
            "qualification_drift",
        )
        benefit = value["summary"]["true_label_benefit"]
        require(
            value["verdict_class"] == ("circular_positive" if benefit else "null"),
            "science_verdict_drift",
        )
    return dict(passed=True, status="replay_passed", sha256=sha256_file(path))


def main(argv: list[str] | None = None) -> int:
    """Execute the finite protocol and expose only checked terminal primary bytes."""
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
        8881,
        "Topology memory methods",
        "results/" + m.NAME + ".json",
        repo_root=m.ROOT,
        seed=m.SEED,
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
    errors = []
    result = recover(raw, "external_input_unavailable", pre)
    if pre["external_ready"] and pre["resources"]["passed"]:
        try:
            with template.phase("measurement"):
                result = measurement(raw, pre)
        except (OSError, ValueError, KeyError, RuntimeError) as error:
            errors.append(str(error))
            result = recover(raw, str(error), pre)
    measured = time.monotonic_ns()
    m.progress("validation_before", 0, 1)
    with TemporaryDirectory(prefix="carnot8881-validation-") as directory:
        validation = (
            unqualified() if args.private_output else validate(raw / "validation", Path(directory))
        )
    valid = bool(
        validation["coverage"]["complete"]
        and validation["union"]["passed"]
        and all(r["passed"] for r in validation["receipts"])
    )
    completed = sum(r["status"] == "complete" for r in result["rows"])
    verdict = (
        "blocked"
        if not pre["external_ready"]
        else "disqualified"
        if errors or not pre["resources"]["passed"] or not valid or completed != 18432
        else "circular_positive"
        if result["summary"]["true_label_benefit"]
        else "null"
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
    p = m.protocol()
    helpers = [
        "scripts/experiment_template.py",
        "python/carnot/reporting/current_work_receipt.py",
        "python/carnot/reporting/primary_publication.py",
        "scripts/adversarial_verify.py",
        "scripts/verdict_row_consistency_lint.py",
        "python/carnot/memory/categorical_memory_8830.py",
        "python/carnot/memory/causal_memory_8802.py",
        "python/carnot/reporting/learning_signal_diagnosis_8815.py",
        "python/carnot/reporting/selective_methods_8829.py",
        "python/carnot/reporting/feedback_information_8867.py",
    ]
    value = dict(
        result,
        experiment_id=8881,
        task_id=m.TASK,
        milestone="2026.11.760",
        run_date=args.date,
        honest_verdict=f"complete_{verdict}_topology_memory_methods",
        verdict_class=verdict,
        inference_substrate=receipt["inference_substrate"],
        inference_substrate_class=receipt["inference_substrate_class"],
        execution_venue="host",
        MODEL_SPECS=[],
        model_invocation_counts=receipt["invocation_counts"],
        verifier_is_oracle=True,
        exposure_scope="constructed signed records; development-only fits; untouched retention; no natural-language generalization",
        independent_generalization_score=0,
        duration_s=receipt["duration_s"],
        phase_spans=spans,
        preconditions_checked=pre,
        random_seed=m.SEED,
        producer_invocation=invocation,
        current_work_receipt=receipt,
        source_artifact_hashes=pre["references"],
        code_config_hashes=[reference(m.ROOT / n) for n in m.OWNED + m.TESTS + helpers],
        raw_shard_hashes=[reference(path) for path in sorted(raw.rglob("*.json"))]
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
            m.GATES,
            methods_controls_required=True,
            owned_statement_coverage=100,
            benefit_required_for_method_readiness=False,
        ),
        required_checks_passed=valid,
        validation_receipts=validation["receipts"],
        owned_statement_coverage=validation["coverage"],
        owned_errors=errors,
        topology_methods_ready_score=int(verdict in {"null", "circular_positive"}),
        cited_upstream_artifacts=[
            dict(
                experiment_id=8879,
                task_id="exp8879-diagnosis-qualification",
                sha256=PIN,
                artifact_field="diagnosis_ready_score",
                expected=1,
            )
        ],
        protocol=p,
        **{
            key: p[key]
            for key in [
                "arm_definitions",
                "topology_manifest",
                "cohort_manifest",
                "expected_row_counts",
                "frozen_gates",
            ]
        },
        fit_manifest=reference(raw / "fit.json") if (raw / "fit.json").is_file() else None,
        release_manifest=dict(policy=p["release_policy"], stream_refs=result["stream_refs"]),
        methodology="Frozen linear normalized head, six existing counts plus bias, SGD .1. Paired graph streams share complete batch releases and verified admissions; exact batch permutations preserve targets and fixed points. Syntax-only admission is separate from label learning.",
        methodology_note="Oracle controls and simultaneous paired-stream bootstrap intervals establish constructed-query evidence only. No generator training, feature search, held-out selection or milestone activation.",
        flagged_adversarial=False,
        repository_health=None,
        field_principles={},
    )
    health = Path("/tmp/carnot-8881-repository-health.log")
    if health.is_file():
        shutil.copyfile(health, raw / "repository_health.log")
        value["repository_health"] = dict(
            scope="repository_health",
            command=".venv/bin/pytest tests/python -q",
            passed=False,
            exit_code=124,
            duration_limit_s=240,
            reason="full suite reported failures and reached bounded deadline; separate from owned checks",
            **reference(raw / "repository_health.log"),
        )
        value["raw_shard_hashes"].append(reference(raw / "repository_health.log"))
    value["field_principles"] = {
        key: "Actual producer metadata or retained frozen operand; constructed-query scope only."
        for key in value
    }
    value["field_principles"].update(
        rows="Per-unit Brier and argmax-label error reduce all headlines; missing scheduled units retain denominators.",
        independent_count="24 paired stream RNG trials; recurring templates and five nested head seeds do not create additional independent topology families.",
        numerical_training_calls="Current research SGD updates include authenticated diagnosis replay, development fits, controls, pilots and target streams; validation children have separate execution receipts.",
        topology_methods_ready_score="Administrative causal-method readiness; does not assert a true-label benefit.",
        summary="Nine Bonferroni paired-stream bootstrap intervals with10000 common draws; query-only utility cannot qualify label learning.",
        projected_runtime_s="Measured development pilot projection includes a1200-second validation allowance; never a padded observed duration.",
        source_artifact_hashes="Original producer hashes and copied original operands; prior failures and assertions remain unchanged.",
        verifier_is_oracle="Exact constructed truth is circular evidence; successful benefit requires circular_positive.",
        repository_health="One bounded whole-suite run, distinct from100-percent owned unit/child coverage.",
    )
    value = normalize_artifact_for_template_write(value)
    value["field_principles"]["reproducibility_checksum"] = (
        "Canonical artifact hash; replay also reconstructs semantics from original operands."
    )
    value["reproducibility_checksum"] = canonical_hash(value)
    candidate = raw / "prospective.json"
    atomic_json(candidate, value)
    if not args.private_output:
        m.progress("terminal_validation_before", 0, 3)
        reports = terminal(candidate, raw)
        value["validation_receipts"] += reports
        value["raw_shard_hashes"] += [
            reference(Path(r[k + "_path"])) for r in reports for k in ["stdout", "stderr"]
        ]
        if not all(r["passed"] for r in reports):
            value.update(
                honest_verdict="complete_disqualified_topology_memory_methods",
                verdict_class="disqualified",
                topology_methods_ready_score=0,
                required_checks_passed=False,
            )
        value.pop("reproducibility_checksum")
        ended_terminal = time.monotonic_ns()
        spans.append(
            dict(
                phase="terminal_validation",
                start_s=(ended - started) / 1e9,
                end_s=(ended_terminal - started) / 1e9,
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
            ended_monotonic_ns=ended_terminal,
            phase_spans=spans,
            small_ebm_training=dict(
                performed=result["numerical_training_calls"] > 0,
                numerical_training_calls=result["numerical_training_calls"],
            ),
        )
        value.update(
            duration_s=receipt["duration_s"], phase_spans=spans, current_work_receipt=receipt
        )
        value["reproducibility_checksum"] = canonical_hash(value)
        m.progress("terminal_validation_after", 3, 0)
    publish_primary(output, value, replay)
    for path in raw.rglob("*"):
        if path.is_file():
            path.chmod(0o400)
    m.progress("published_" + value["verdict_class"], completed, 18432 - completed)
    return 0
