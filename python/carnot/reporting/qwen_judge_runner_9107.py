"""REQ-REPORT-9107-V776: publish only recomputed private evidence and owned checks."""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import tempfile
import time
from typing import Any

from carnot.reporting import qwen_judge_9107 as d
from carnot.reporting.current_work_receipt import atomic_json, build_current_work_receipt
from carnot.reporting.natural_label_runner_8975 import read_ref
from carnot.reporting.primary_publication import publish_primary
from carnot.reporting.spline_basis_methods_8986 import gate
from carnot.reporting.v709_execution import child, execute

Json = dict[str, Any]
DIRECT = [
    "tests/python/test_primary_publication_7928.py",
    "tests/python/test_current_work_receipt.py",
    "tests/python/test_inference_sota_models.py",
    "tests/python/test_embedding_runtime_9098.py",
]


def plan(raw: Path) -> Json:
    """Seal owned and direct consumer commands before any model measurement."""
    config = raw / "coverage.ini"
    config.write_text(
        "[run]\nparallel = true\npatch = subprocess\ninclude =\n"
        + "".join("    " + str(d.ROOT / p) + "\n" for p in d.OWNED)
        + "[report]\nexclude_lines =\nomit =\n"
    )
    py = str(d.ROOT / ".venv/bin/python")
    tests = [str(d.ROOT / ".venv/bin/pytest"), "-n", "0", "-o", "addopts=", "--no-cov", "-q"]
    covered = [
        "env",
        "COVERAGE_RCFILE=" + str(config),
        "COVERAGE_FILE=" + str(raw / ".coverage"),
        py,
        "-m",
        "coverage",
        "run",
        "--rcfile=" + str(config),
        "-m",
        "pytest",
        "-n",
        "0",
        "-o",
        "addopts=",
        "--no-cov",
        "-q",
        d.TEST,
    ]
    commands = [
        ("owned_tests", covered),
        ("direct_consumers", tests + DIRECT),
        ("e2e018", tests + ["tests/python/test_experiment_7891_v685_authority_lifecycle.py"]),
        ("e2e021", tests + ["tests/python/test_restricted_decision_audit_8210.py"]),
        ("ruff_check", [str(d.ROOT / ".venv/bin/ruff"), "check", *d.OWNED, d.TEST]),
        ("ruff_format", [str(d.ROOT / ".venv/bin/ruff"), "format", "--check", *d.OWNED, d.TEST]),
        (
            "strict_mypy",
            [str(d.ROOT / ".venv/bin/mypy"), "--strict", "--follow-imports=silent", *d.OWNED],
        ),
        ("spec_coverage", [py, "scripts/check_spec_coverage.py", d.TEST]),
    ]
    return dict(
        commands=[
            dict(name=n, argv=a, expected=0, deadline=240, scope="owned") for n, a in commands
        ],
        code_config_hashes=[d.ref(d.ROOT / p) for p in [*d.OWNED, d.TEST]],
        frozen_ns=time.monotonic_ns(),
        coverage_configuration=d.ref(config),
        excluded_statements=0,
        e2e020_applicability="not applicable: no delayed learning or recovery",
    )


def measure(root: Path, raw: Path, date: str, scope: Json) -> Json:
    """Authenticate before a supervised native child and seal scores before target reduction."""
    start = time.monotonic_ns()
    d.progress("authentication_before")
    inputs = d.authenticate(root)
    atomic_json(raw / "protocol.json", d.protocol())
    atomic_json(raw / "validation_scope.json", scope)
    atomic_json(raw / "inputs.json", inputs)
    d.progress("authentication_after")
    runtime: Json = dict(
        rows=[],
        checks=[],
        model_loads=0,
        blocker=inputs["blocker"],
        events=[],
        runtime_reference={},
    )
    if not inputs["blocker"]:
        native = raw / "native"
        native.mkdir(mode=0o700)
        atomic_json(native / "job.json", dict(rows=inputs["rows"], views=inputs["views"]))
        call = child(
            "qwen_scoring",
            [
                "env",
                "-u",
                "PYTHONPATH",
                "-C",
                str(raw),
                "CUDA_VISIBLE_DEVICES=0",
                str(d.ROOT / ".venv/bin/python"),
                "-u",
                str(d.ROOT / d.CLI),
                "--worker",
                str(native),
            ],
            raw / "logs",
            deadline=3300,
            heartbeat=30,
        )
        if (native / "runtime.json").exists() and call["passed"]:
            runtime = json.loads((native / "runtime.json").read_bytes())
        else:
            events = (
                json.loads((native / "events.json").read_bytes())["events"]
                if (native / "events.json").exists()
                else []
            )
            if events:
                events.append(dict(events[-1], state="cancelled", monotonic_ns=time.monotonic_ns()))
            runtime.update(
                blocker="native_worker",
                events=events,
                model_loads=int(bool(events)),
                checks=[
                    gate(
                        Path(call["stderr_path"]),
                        "normal_exit",
                        0,
                        call["exit_code"],
                        "native_worker",
                        d.TASK,
                    )
                ],
            )
        runtime["process_receipt"] = call
    if "receipt" not in runtime:
        runtime["receipt"] = build_current_work_receipt(
            run_id="native",
            owner_pid=runtime["events"][0]["owner_pid"] if runtime["events"] else os.getpid(),
            events=runtime["events"],
            inference_substrate="frozen_gguf_forward_scoring",
            inference_substrate_details=dict(generated_tokens=0),
            inference_substrate_class="model_load_no_generation"
            if runtime["model_loads"]
            else "blocked_no_run",
            execution_venue="host",
            started_monotonic_ns=start,
            ended_monotonic_ns=time.monotonic_ns(),
        )
    atomic_json(raw / "sealed_predictions.json", dict(rows=runtime["rows"]))
    atomic_json(
        raw / "request_clocks.json",
        dict(
            rows=[
                {
                    k: r[k]
                    for k in [
                        "unit_id",
                        "outer_duration_s",
                        "started_monotonic_ns",
                        "ended_monotonic_ns",
                    ]
                }
                for r in runtime["rows"]
            ]
        ),
    )
    work = dict(
        inputs=inputs,
        runtime=runtime,
        scope=scope,
        run_date=date,
        duration_s=(time.monotonic_ns() - start) / 1e9,
        judge_protocol_reference=d.ref(raw / "protocol.json"),
        judge_prediction_reference=d.ref(raw / "sealed_predictions.json"),
        request_clock_reference=d.ref(raw / "request_clocks.json"),
    )
    atomic_json(raw / "work.json", work)
    d.progress("measurement_complete", len(runtime["rows"]), 450 - len(runtime["rows"]))
    return d.ref(raw / "work.json")


def build(reference: Json, validation: list[Json], coverage: Json) -> Json:
    """Rebuild every scientific aggregate; administrative readiness never claims benefit."""
    work = read_ref(reference)
    runtime, inputs = work["runtime"], work["inputs"]
    sealed = read_ref(work["judge_prediction_reference"])["rows"]
    if sealed != runtime["rows"]:
        raise ValueError("prediction_drift")
    for row in sealed:
        for value in row["likelihoods"].values():
            d.verify_likelihood(value)
        yes, no = (row["likelihoods"][w]["log_probability"] for w in ["Yes", "No"])
        if row["yes_logprob"] != yes or row["no_logprob"] != no:
            raise ValueError("raw_score_drift")
    rows = deepcopy(sealed)
    done = {r["unit_id"] for r in rows}
    blocker = runtime["blocker"]
    for row in inputs["rows"]:
        if row["unit_id"] not in done:
            rows.append(
                dict(
                    row,
                    original_qid=row["qid"],
                    arm="frozen_qwen_judge",
                    condition="exposed_development",
                    seed=7769107,
                    status="censored",
                    absolute_metric=None,
                    numerator=0,
                    denominator=1,
                    censor_reason=blocker,
                    exclusion_reason=None,
                )
            )
    checks = inputs["checks"] + runtime["checks"]
    names = [c["name"] for c in work["scope"]["commands"]]
    passed = (
        len(validation) == len(names)
        and [v["name"] for v in validation] == names
        and all(v["passed"] and v["normal_exit"] and not v["timed_out"] for v in validation)
        and coverage.get("statement_percent") == 100
        and coverage.get("excluded_statements") == 0
    )
    reduced: Json = dict(
        temperature=None, metrics=None, qid_rows=[], test_predictions=[], uncertainty=None
    )
    if not blocker:
        labels = {
            r["unit_id"]: r["annotator1"] for r in read_ref(inputs["label_reference"])["rows"]
        }
        reduced = d.reduce_scores(rows, labels)
    value = dict(
        reduced,
        experiment_id=9107,
        task_id=d.TASK,
        milestone="2026.11.776",
        schema="carnot.qwen_judge.v1",
        honest_verdict="complete_disqualified_owned_validation"
        if not passed
        else "complete_blocked_" + blocker
        if blocker
        else "complete_null_frozen_judge_baseline",
        verdict_class="disqualified" if not passed else "blocked" if blocker else "null",
        judge_ready_score=int(passed and not blocker),
        required_checks_passed=passed,
        flagged_adversarial=not passed,
        rows=rows,
        raw_likelihood_rows=sealed,
        intended_count=450,
        attempted_count=len(sealed),
        completed_count=len(sealed),
        failed_count=0,
        censored_count=450 - len(sealed),
        sample_size_budget=d.BUDGET,
        effective_independent_groups=len({r["group_id"] for r in sealed}),
        independent_generalization_score=0,
        verifier_is_oracle=False,
        claim_scope="Current frozen Qwen judgments of archived answers; no Qwen answer accuracy, search or ARC solve claim.",
        exposure_scope="LPHB exposed development; original human A1 truth, A2 missingness preserved",
        MODEL_SPECS=d.MODEL_SPECS,
        inference_substrate="frozen_gguf_forward_scoring",
        inference_substrate_class="model_load_no_generation"
        if runtime["model_loads"]
        else "blocked_no_run",
        requested_inference_substrate_class="model_load_no_generation",
        current_llm_invocations=runtime["model_loads"],
        model_invocation_counts=dict(
            model_loads=runtime["model_loads"],
            generations=0,
            forward_calls=2 * len(sealed),
            prompt_token_evaluations=sum(r["prompt_token_evaluations"] for r in sealed),
            continuation_token_evaluations=sum(r["continuation_token_evaluations"] for r in sealed),
        ),
        generated_tokens=0,
        duration_s=work["duration_s"],
        run_date=work["run_date"],
        random_seed=7769107,
        primitive_reference=reference,
        code_config_hashes=work["scope"]["code_config_hashes"],
        source_references=inputs["references"],
        current_work_receipts=[runtime["receipt"]],
        validation_receipts=validation,
        owned_statement_coverage=coverage,
        gate_check_summary=checks,
        acceptance_gates=dict(human_authority=1, raw_logits=1, owned_validation=passed),
        runtime_reference=runtime["runtime_reference"],
        offload_evidence={
            k: runtime.get(k)
            for k in [
                "gpu_lease",
                "memory_before",
                "memory_resident",
                "offload_layers",
                "embedded_chat_template",
            ]
        },
        judge_protocol_reference=work["judge_protocol_reference"],
        judge_prediction_reference=work["judge_prediction_reference"],
        request_clock_reference=work["request_clock_reference"],
        paired_head_contrasts=[],
        comparison_status="unavailable",
        comparison_reason="No qualified Exp9102 prediction artifact available; no benefit-dependent gate or retry.",
        validation_scope_manifest=work["scope"],
    )
    value["reproducibility_checksum"] = d.canonical_hash(
        dict(reference=reference, protocol=d.protocol(), rows=rows)
    )
    value["field_principles"] = {
        k: "Retain the exact "
        + k.replace("_", " ")
        + " so this operand can be independently checked."
        for k in value
    }
    value["field_principles"].update(
        temperature="CAL-only proper loss selection prevents TEST tuning.",
        raw_likelihood_rows="Full vocabulary logits make every literal continuation score recomputable.",
        sample_size_budget="Original question groups determine support; siblings do not multiply independent units.",
        comparison_status="The baseline survives unavailable small-head evidence without outcome-dependent retries.",
        current_work_receipts="Owned events separate this run from historical model provenance.",
        judge_ready_score="Operational completion establishes availability, not scientific benefit.",
        exposure_scope="Exposed development cannot support independent generalization.",
        request_clock_reference="Outer clocks include text preparation, both passes and serialization.",
    )
    return value


def statement_coverage(raw: Path) -> Json:
    """Count every owned executable statement with no pragma or configuration exclusions."""
    import coverage

    cov = coverage.Coverage(config_file=str(raw / "coverage.ini"), data_file=str(raw / ".coverage"))
    cov.combine(strict=True, keep=True)
    cov.save()
    files = []
    for owned in d.OWNED:
        filename, statements, excluded, missing, _ = cov.analysis2(str(d.ROOT / owned))
        files.append(
            dict(path=filename, statements=len(statements), excluded=excluded, missing=missing)
        )
    total = sum(f["statements"] for f in files)
    missing_count = sum(len(f["missing"]) for f in files)
    return dict(
        statement_percent=100 * (total - missing_count) / total,
        excluded_statements=sum(len(f["excluded"]) for f in files),
        files=files,
        coverage_data_reference=d.ref(raw / ".coverage"),
    )


def replay(path: Path) -> Json:
    """Cold reduction rejects altered totals even when the attacker updates a JSON hash."""
    value = json.loads(path.read_bytes())
    for reference in value["source_references"] + value["code_config_hashes"]:
        d.read_path(reference)
    for key in [
        "judge_protocol_reference",
        "judge_prediction_reference",
        "request_clock_reference",
    ]:
        d.read_path(value[key])
    rebuilt = build(
        value["primitive_reference"],
        value["validation_receipts"],
        value["owned_statement_coverage"],
    )
    if rebuilt != value:
        raise ValueError("reduction_drift")
    return dict(passed=True, replay_passed=True)


def validate(path: Path, logs: Path) -> Json:
    """Run the unchanged artifact checkers under the same bounded heartbeat supervisor."""
    replay(path)
    receipts = [
        child(
            name,
            [str(d.ROOT / ".venv/bin/python"), str(d.ROOT / script), *args, str(path)],
            logs,
            deadline=120,
            heartbeat=30,
        )
        for name, script, args in [
            ("adversarial", "scripts/adversarial_verify.py", ["--json"]),
            ("strict_rows", "scripts/verdict_row_consistency_lint.py", ["--strict"]),
        ]
    ]
    return dict(passed=all(r["passed"] for r in receipts), checks=receipts)


def main(argv: list[str] | None = None) -> int:
    """Keep native work isolated and leave historical artifacts and public defaults untouched."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default="20261115")
    parser.add_argument("--private", type=Path)
    parser.add_argument("--input-root", type=Path, default=d.ROOT)
    parser.add_argument("--output", type=Path, default=d.ROOT / "results" / (d.NAME + ".json"))
    parser.add_argument("--replay", type=Path)
    parser.add_argument("--worker", type=Path)
    args = parser.parse_args(argv)
    if args.worker:
        from carnot.inference.qwen_judge_9107 import worker

        worker(args.worker)
        return 0
    if args.replay:
        try:
            print(json.dumps(replay(args.replay)), flush=True)
            return 0
        except (OSError, ValueError, KeyError) as error:
            print(json.dumps(dict(passed=False, error=str(error))), flush=True)
            return 1
    if args.date != "20261115":
        parser.error("date must be 20261115")
    raw = args.private or Path(tempfile.mkdtemp(prefix="carnot-9107-"))
    raw.mkdir(parents=True, exist_ok=True, mode=0o700)
    d.progress("freeze_validation")
    scope = plan(raw)
    reference = measure(args.input_root, raw, args.date, scope)
    d.progress("validation_before")
    validation = execute(scope["commands"], raw / "validation")
    coverage = statement_coverage(raw)
    value = build(reference, validation, coverage)
    candidate = raw / "candidate.json"
    atomic_json(candidate, value)
    report = validate(candidate, raw / "artifact_checks")
    atomic_json(raw / "artifact_checks.json", report)
    if not report["passed"]:
        value.update(
            honest_verdict="complete_disqualified_artifact_checks",
            verdict_class="disqualified",
            judge_ready_score=0,
            required_checks_passed=False,
            flagged_adversarial=True,
        )
        atomic_json(args.output, value)
        return 1
    if value["required_checks_passed"]:
        publish_primary(args.output, value, lambda path: validate(path, raw / "publication_checks"))
    else:
        atomic_json(args.output, value)
    d.progress("terminal_" + value["verdict_class"])
    return 0
