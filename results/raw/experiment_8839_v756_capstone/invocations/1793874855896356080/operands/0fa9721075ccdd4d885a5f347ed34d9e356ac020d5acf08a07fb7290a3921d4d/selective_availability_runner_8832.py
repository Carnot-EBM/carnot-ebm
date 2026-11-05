"""REQ-VERIFY-8832: bind operands, execute finite work and publish checked bytes."""

import argparse
import json
import os
from pathlib import Path
import shutil
import sys
from tempfile import TemporaryDirectory
import time
from typing import Any

import jax
import numpy
import scipy

from carnot.reporting import selective_availability_8832 as m
from carnot.reporting import selective_methods_runner_8829 as inherited
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

Json = dict[str, Any]


def preconditions(raw: Path, upstream: Path) -> Json:
    """Missing external evidence blocks measurement; local resource failure disqualifies it."""
    raw.mkdir(parents=True, exist_ok=True, mode=0o700)
    raw.chmod(0o700)
    atomic_json(raw / "writable.json", dict(probe=True))
    refs = []
    diagnostics = []
    diagnostic: Json = dict(
        upstream="exp8829-selective-methods",
        artifact_field="selective_protocol_ready_score",
        expected=1,
        observed=None,
        path=str(upstream),
        sha256=None,
    )
    try:
        ref = reference(upstream)
        refs.append(ref)
        value = load(ref)
        diagnostic["sha256"] = ref["sha256"]
        diagnostic["observed"] = value["selective_protocol_ready_score"]
        if value["selective_protocol_ready_score"] != 1:
            diagnostics.append(
                dict(
                    diagnostic,
                    check="observed_zero"
                    if value["selective_protocol_ready_score"] == 0
                    else "rejected_producer",
                )
            )
        elif value["task_id"] != "exp8829-selective-methods" or value["verdict_class"] in [
            "blocked",
            "disqualified",
        ]:
            diagnostics.append(dict(diagnostic, check="rejected_producer"))
        else:
            for name in ["protocol.json", "cohort.json"]:
                operand = next(
                    (r for r in value["raw_shard_hashes"] if Path(r["path"]).name == name), None
                )
                if operand is None:
                    raise KeyError(name)
                diagnostic.update(
                    path=operand["path"],
                    artifact_field="sha256",
                    expected=operand["sha256"],
                    observed=None,
                )
                load(operand)
                refs.append(operand)
            for name in [
                "experiment_8819_v755_consensus_selective_energy.json",
                "research-roadmap-v755-preserved-20261104.md",
            ]:
                operand = next(
                    (r for r in value["source_artifact_hashes"] if Path(r["path"]).name == name),
                    None,
                )
                if operand is None:
                    raise KeyError(name)
                diagnostic.update(
                    path=operand["path"],
                    artifact_field="sha256",
                    expected=operand["sha256"],
                    observed=sha256_file(Path(operand["path"])),
                )
                if diagnostic["observed"] != operand["sha256"]:
                    raise ValueError("authority_hash")
                refs.append(operand)
    except (OSError, KeyError, ValueError, TypeError) as error:
        path = Path(diagnostic["path"])
        diagnostics.append(
            dict(
                diagnostic,
                check="missing_file"
                if isinstance(error, OSError)
                else "missing_field"
                if isinstance(error, KeyError)
                else "rejected_producer",
                observed=str(error),
                sha256=sha256_file(path) if path.is_file() else None,
            )
        )
    resources: Json = dict(
        repository_virtualenv=(m.ROOT / ".venv/bin/python").is_file(),
        cpu_numerical_dependencies=dict(
            numpy=numpy.__version__,
            scipy=scipy.__version__,
            jax=jax.__version__,
            backend=jax.default_backend(),
        ),
        private_output_writable=(raw / "writable.json").is_file(),
        private_output_mode=oct(raw.stat().st_mode & 0o777),
        free_disk_bytes=shutil.disk_usage(raw).free,
    )
    resources["passed"] = (
        resources["repository_virtualenv"]
        and resources["private_output_writable"]
        and resources["cpu_numerical_dependencies"]["backend"] == "cpu"
        and resources["free_disk_bytes"] >= 512 * 1024 * 1024
    )
    return dict(
        resources=resources,
        references=refs,
        diagnostics=diagnostics,
        external_ready=not diagnostics,
    )


def load_cohort(refs: list[Json]) -> Json:
    """Load only the cohort whose exact bytes passed the external authority checks."""
    return dict(load(next(r for r in refs if Path(r["path"]).name == "cohort.json")))


def validate(raw: Path, private: Path) -> Json:
    """Reuse exact-node collection and real unit/subprocess statement coverage."""
    checker = namespace(inherited, "selective8832_validation")
    checker.m = m
    return dict(checker.validate(raw, private))


def mechanical(path: Path, raw: Path) -> Json:
    """Both repository verifiers examine the prospective bytes before publication."""
    receipts = [
        stream(
            name,
            [sys.executable, "-u", str(m.ROOT / "scripts" / script), *options, str(path)],
            raw / "checks",
            deadline=120,
        )
        for name, script, options in [
            ("adversarial", "adversarial_verify.py", ["--json"]),
            ("row_consistency", "verdict_row_consistency_lint.py", ["--strict"]),
        ]
    ]
    receipts.append(
        stream(
            "cold_replay",
            [
                "env",
                "-u",
                "PYTHONPATH",
                "--chdir=/tmp",
                sys.executable,
                "-u",
                str(m.ROOT / m.CLI),
                "--cold-replay",
                str(path),
            ],
            raw / "checks",
            deadline=180,
        )
    )
    return dict(passed=all(r["passed"] for r in receipts), receipts=receipts)


def replay(path: Path) -> Json:
    """Recompute reductions from bound operands; reject changed identity and aggregates."""
    artifact = json.loads(path.read_text())
    if artifact["experiment_id"] != 8832 or artifact["task_id"] != m.TASK:
        raise ValueError("producer_identity")
    for ref in (
        artifact["raw_shard_hashes"]
        + artifact["code_config_hashes"]
        + artifact["source_artifact_hashes"]
    ):
        if sha256_file(Path(ref["path"])) != ref["sha256"]:
            raise ValueError("operand_hash")
    if artifact["completed_count"] != len(artifact["rows"]):
        raise ValueError("count_drift")
    if artifact["measurement"] is not None:
        retained = load(artifact["measurement"])
        cohort = load(retained["cohort"])
        rows = load(retained["scored"])["rows"]
        m.audit(rows, cohort)
        calculated = m.reduce(rows, draws=retained["draws"])
        for key in calculated:
            if canonical_hash(calculated[key]) != canonical_hash(artifact[key]):
                raise ValueError("reduction_drift:" + key)
    if validate_current_work_receipt(artifact["current_work_receipt"]):
        raise ValueError("current_receipt")
    return dict(passed=True, status="replay_passed", sha256=sha256_file(path))


def main(argv: list[str] | None = None) -> int:
    """Private outputs remain unqualified; production readiness requires owned checks."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", choices=["20261105"], default="20261105")
    parser.add_argument("--output", type=Path, default=m.ROOT / "results" / (m.NAME + ".json"))
    parser.add_argument("--upstream", type=Path, default=m.UPSTREAM)
    parser.add_argument("--private-output", action="store_true")
    parser.add_argument("--cold-replay", "--replay", type=Path)
    args = parser.parse_args(argv)
    if args.cold_replay:
        try:
            print(json.dumps(replay(args.cold_replay)), flush=True)
            return 0
        except (ValueError, OSError, KeyError) as error:
            print(json.dumps(dict(passed=False, error=str(error))), flush=True)
            return 1
    started = time.monotonic_ns()
    raw = args.output.absolute().parent / "raw" / m.NAME / "invocations" / str(started)
    m.legacy.old.progress("8832_preconditions")
    checked = preconditions(raw, args.upstream)
    phases = [
        dict(
            phase="preconditions",
            started_monotonic_ns=started,
            ended_monotonic_ns=time.monotonic_ns(),
        )
    ]
    refs = []
    result: Json = dict(
        rows=[],
        per_candidate_rows=[],
        certificate_rows=[],
        support_floor_rows=[],
        failure_decomposition=[],
        risk_numerators_denominators=[],
        coverage_rows=[],
        selected_threshold_rows=[],
        iid_assumptions=dict(certificate_issued=False),
        selective_benefit_score=0,
    )
    measurement = None
    training_calls = 0
    owned_errors: list[str] = []
    if checked["external_ready"] and checked["resources"]["passed"]:
        try:
            work_start = time.monotonic_ns()
            cohort = load_cohort(checked["references"])
            if not args.private_output and {
                split: sum(w["split"] == split for w in cohort["worlds"])
                for split in ["fit", "calibration", "evaluation"]
            } != dict(fit=120, calibration=360, evaluation=240):
                raise ValueError("cohort_roster")
            inputs = m.prepare(cohort)
            protocol = dict(
                calibration_budgets=m.BUDGETS,
                cutoffs=m.legacy.MENU,
                seeds=m.SEEDS,
                steps=m.STEPS,
                alpha_per_bound=m.ALPHA,
                zero_error_floor=m.FLOOR,
                expected_world_counts=dict(fit=120, calibration=360, evaluation=240),
                candidate_selection="minimum input hash variant, hash-fixed candidate index; label-free candidate lists",
                iid_world_sampling=False,
            )
            atomic_json(raw / "protocol.json", protocol)
            atomic_json(raw / "cohort.json", cohort)
            fitted = m.fit(inputs, cohort["labels"], steps=m.STEPS)
            training_calls = 3 * len(m.SEEDS)
            atomic_json(raw / "fitted.json", fitted)
            m.legacy.old.progress("8832_before_prediction_benchmark")
            predictions, checks = m.legacy.predict(
                [u for u in inputs if u["split"] != "fit"], fitted["parameters"], []
            )
            atomic_json(
                raw / "predictions.json",
                dict(
                    rows=predictions,
                    checks=checks,
                    seal=canonical_hash(dict(rows=predictions, checks=checks)),
                ),
            )
            m.legacy.old.progress("8832_after_prediction_benchmark", len(predictions), 0)
            calibration = m.score([r for r in predictions if r["split"] == "calibration"], cohort)
            frozen = [
                m.select(calibration, budget, arm, seed)
                for budget in m.BUDGETS
                for arm, seed in sorted({(r["arm"], r["seed"]) for r in calibration})
            ]
            atomic_json(raw / "frozen_thresholds.json", dict(rows=frozen))
            m.legacy.old.progress("8832_thresholds_frozen_before_evaluation")
            scored = m.score(predictions, cohort)
            # The exact reference deliberately reads encoded truth and is circular.
            for unit in inputs:
                if unit["split"] == "fit":
                    continue
                truth = m.legacy.old.upstream.old.strong(unit["input"])
                if truth != cohort["labels"][unit["unit_id"]]:
                    raise ValueError("exact_label_disagreement")
                for index, candidate in enumerate(unit["candidates"]):
                    target = m.legacy.old.target(candidate, truth)
                    base = dict(
                        unit_id=unit["unit_id"],
                        cluster_id=unit["world_id"],
                        candidate_index=index,
                        candidate=candidate,
                        arm="exact_oracle",
                        seed=0,
                        split=unit["split"],
                        condition=unit["condition"],
                        agreement=unit["agreement"],
                        probabilities=[float(i == target) for i in range(3)],
                        action=m.legacy.ACTIONS[target],
                    )
                    scored.extend(m.score([base], cohort))
            atomic_json(raw / "scored.json", dict(rows=scored))
            result = m.reduce(scored)
            result["per_candidate_rows"] = reference(raw / "scored.json")
            atomic_json(
                raw / "measurement.json",
                dict(
                    cohort=reference(raw / "cohort.json"),
                    scored=reference(raw / "scored.json"),
                    draws=2000,
                ),
            )
            measurement = reference(raw / "measurement.json")
            phases.append(
                dict(
                    phase="training_prediction_selection_evaluation",
                    started_monotonic_ns=work_start,
                    ended_monotonic_ns=time.monotonic_ns(),
                )
            )
            refs = [reference(p) for p in raw.glob("*.json")]
        except (ValueError, KeyError, RuntimeError) as error:
            owned_errors.append(str(error))
    m.legacy.old.progress("8832_before_owned_validation")
    validation_start = time.monotonic_ns()
    with TemporaryDirectory(prefix="exp8832-validation-") as scratch:
        validation = (
            unqualified() if args.private_output else validate(raw / "validation", Path(scratch))
        )
    phases.append(
        dict(
            phase="owned_validation",
            started_monotonic_ns=validation_start,
            ended_monotonic_ns=time.monotonic_ns(),
        )
    )
    m.legacy.old.progress("8832_after_owned_validation")
    ready = bool(
        not owned_errors
        and checked["external_ready"]
        and checked["resources"]["passed"]
        and validation["coverage"]["complete"]
        and validation["union"]["passed"]
        and all(r["passed"] for r in validation["receipts"])
    )
    verdict = "blocked" if not checked["external_ready"] else "null" if ready else "disqualified"
    receipt = build_current_work_receipt(
        run_id=str(started),
        owner_pid=os.getpid(),
        events=[],
        inference_substrate="cpu_energy_training",
        inference_substrate_details=dict(numerical_training_calls=training_calls),
        inference_substrate_class="no_model_load",
        execution_venue="host",
        started_monotonic_ns=started,
        ended_monotonic_ns=time.monotonic_ns(),
        phase_spans=phases,
        small_ebm_training=dict(performed=training_calls > 0, training_calls=training_calls),
    )
    atomic_json(raw / "current_work_receipt.json", receipt)
    value = dict(
        result,
        experiment_id=8832,
        task_id=m.TASK,
        milestone="2026.11.756",
        run_date=args.date,
        honest_verdict="complete_" + verdict + "_selective_availability",
        verdict_class=verdict,
        inference_substrate=receipt["inference_substrate"],
        inference_substrate_class=receipt["inference_substrate_class"],
        MODEL_SPECS=[],
        model_invocation_counts=receipt["invocation_counts"],
        numerical_training_calls=training_calls,
        verifier_is_oracle=False,
        exposure_scope="constructed relational evidence with previously sealed labels; no natural-language certification",
        independent_generalization_score=0,
        duration_s=receipt["duration_s"],
        random_seed=7558819,
        reproducibility_checksum=canonical_hash(
            dict(seed=7558819, protocol=m.BUDGETS, sources=checked["references"])
        ),
        source_artifact_hashes=checked["references"],
        code_config_hashes=[reference(m.ROOT / p) for p in m.OWNED + m.TESTS]
        + [
            reference(m.ROOT / "python/carnot/reporting" / name)
            for name in ["consensus_selective_energy_8819.py", "consensus_energy_8806.py"]
        ],
        raw_shard_hashes=refs
        + validation["raw_references"]
        + [reference(raw / "current_work_receipt.json")],
        phase_spans=phases,
        preconditions_checked=checked,
        intended_count=len(result["rows"]),
        completed_count=len(result["rows"]),
        failed_count=0,
        censored_count=0,
        excluded_count=0,
        independent_count=len({r["cluster_id"] for r in result["rows"]}),
        gate_check_summary=checked["diagnostics"],
        acceptance_gates=dict(
            rows_ready=dict(
                expected=1,
                observed=int(ready),
                principle="owned validation and complete operands admit diagnostic rows",
            ),
            risk=dict(upper=0.05, principle="served errors require accepted independent support"),
            coverage=dict(
                lower=0.30, principle="abstaining on every request cannot establish utility"
            ),
            useful_gain=dict(
                lower=0.05,
                familywise_delta=0.01,
                comparisons=30,
                bound_allocation=120,
                principle="all five seeds and three budgets compare both frozen fixed baselines",
            ),
            iid=dict(
                observed=False, principle="inherited topology sampling forbids issued certificates"
            ),
        ),
        required_checks_passed=all(r["passed"] for r in validation["receipts"])
        and validation["coverage"]["complete"]
        and validation["union"]["passed"],
        validation_receipts=validation["receipts"],
        owned_statement_coverage=validation["coverage"],
        selective_rows_ready_score=int(ready),
        owned_errors=owned_errors,
        measurement=measurement,
        current_work_receipt=receipt,
        methodology="Fixed independent-support budget diagnostic; exact Clopper-Pearson bounds with simultaneous cutoff correction; old whole-world bootstrap descriptive; held-out labels scored after frozen selection.",
        exact_reference=dict(
            verdict_class="circular_positive",
            principle="encoded exact truth cannot establish learned generalization",
        ),
        field_principles={},
    )
    value["field_principles"] = {
        key: "Retained producer operand or recomputable diagnostic; hashes bind bytes and do not prove scientific validity."
        for key in value
    }
    prospective = raw / "prospective.json"
    atomic_json(prospective, value)
    if not args.private_output:
        report = mechanical(prospective, raw)
        value["validation_receipts"] += report["receipts"]
        if not report["passed"]:
            value.update(
                honest_verdict="complete_disqualified_selective_availability",
                verdict_class="disqualified",
                selective_rows_ready_score=0,
                required_checks_passed=False,
            )
    atomic_json(prospective, value)
    replay(prospective)
    publish_primary(args.output, value, lambda path: replay(path))
    m.legacy.old.progress("8832_published", len(result["rows"]), 0)
    return 0
