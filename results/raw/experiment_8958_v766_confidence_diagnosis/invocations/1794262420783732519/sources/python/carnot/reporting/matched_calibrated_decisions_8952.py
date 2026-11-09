"""REQ-REPORT-8952: confidence-only selection on frozen matched graph responses.

Labels enter fitting only in the training and calibration roles. The test truth
is read after model parameters and predictions have been saved independently.
"""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import re
from typing import Any

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from carnot.reporting import calibration_methods_8937 as calibration
from carnot.reporting import matched_corpus_admission_8951 as admission
from carnot.reporting import matched_corpus_a_8949 as shard
from carnot.reporting.current_work_receipt import (
    ZERO_INVOCATION_COUNTS,
    atomic_json,
    canonical_hash,
)

Json = dict[str, Any]
b, ROOT = calibration.b, admission.ROOT
NAME, TASK, EXPERIMENT_ID = (
    "experiment_8952_v765_matched_calibrated_decisions",
    "exp8952-matched-calibrated-decisions",
    8952,
)
CLI = f"scripts/experiments/{NAME}.py"
OWNED = [
    "python/carnot/reporting/matched_calibrated_decisions_8952.py",
    "python/carnot/reporting/matched_decisions_runner_8952.py",
    CLI,
]
TESTS = ["tests/python/test_matched_calibrated_decisions_8952.py"]
INPUTS = [
    (
        "results/experiment_8946_v765_matched_source_methods.json",
        "sha256:7cd1769932adf184a01e75351d83a997b2f7d15797095b581ae54b6886777b19",
        None,
        "exp8946-matched-source-methods",
    ),
    (
        f"results/{admission.NAME}.json",
        "sha256:21238c55f69ecf8f4bf8be563d96143e314a6cfaeba6f4c5dfa3fb15580e0fc7",
        "corpus_ready_score",
        admission.TASK,
    ),
]


def authenticate(root: Path) -> Json:
    """Exact producer pins and retained operands precede all live small-head work."""
    failures, references = [], []
    for name, pin, field, task in INPUTS:
        path = root / name
        ref = calibration.reference(path)
        references.append(ref)
        try:
            admission.matched.audit.require(ref["sha256"], pin, path, "sha256", task)
            value = json.loads(path.read_text())
            expected = dict(
                task_id=task,
                required_checks_passed=True,
                calibration_protocol_hash=canonical_hash(calibration.manifest()),
            )
            if field:
                expected[field] = 1
            for key, operand in expected.items():
                admission.matched.audit.require(value.get(key), operand, path, key, task)
            for operand in (
                value["source_artifact_hashes"]
                + value["code_config_hashes"]
                + value["raw_shard_hashes"]
            ):
                actual = calibration.reference(Path(operand["path"]))
                admission.matched.audit.require(
                    actual["sha256"], operand["sha256"], Path(operand["path"]), "sha256", task
                )
                references.append(operand)
        except admission.matched.audit.Rejected as error:
            failures.append(dict(error.receipt, check=error.receipt["failure_kind"]))
        except (OSError, ValueError, KeyError, TypeError) as error:
            failures.append(
                dict(
                    upstream=task,
                    **ref,
                    hash=ref["sha256"],
                    check="rejected_producer",
                    artifact_field="producer_json",
                    expected="complete original object",
                    observed=str(error),
                )
            )
    authority = admission.authenticate(root)
    return dict(
        failures=failures + authority["failures"],
        references=references + authority["references"],
        authority_mode=authority["authority_mode"],
    )


def empty_result() -> Json:
    """Missing upstream data censors every scheduled unit and arm explicitly."""
    return dict(
        rows=[
            dict(
                unit_id=f"unopened-test-{i:03d}",
                cluster_id=f"unopened-test-{i:03d}",
                arm=arm,
                condition="matched_k2",
                metric="brier",
                value=None,
                numerator=None,
                denominator=1,
                status="censored",
                censor_reason="external_evidence",
                missing_reason="external_evidence",
            )
            for i in range(80)
            for arm in b.ARMS
        ],
        intended_count=80,
        completed_count=0,
        independent_count=0,
        failed_count=0,
        censored_count=80,
        excluded_count=0,
        numerical_training_calls=0,
        test_prediction_rows=[],
        paired_intervals={},
        coverage_rows=[],
        model_state_hashes={},
        oracle_access_controls={},
        descriptive_rows=[],
    )


def evaluate(corpus: list[Json], raw: Path, *, replay: bool = False) -> Json:
    """Use frozen train/calibration roles, seal predictions, then open test truth once."""
    b.progress("8952_before_roster", 0, 320)
    if (
        Counter(r["role"] for r in corpus) != dict(train=160, calibration=80, test=80)
        or len({r["cluster_id"] for r in corpus}) != 320
        or len({r["unit_id"] for r in corpus}) != 320
        or any(r["status"] != "completed" for r in corpus)
    ):
        raise ValueError("admitted_roster")
    splits = {
        role: [r for r in corpus if r["role"] == role] for role in ("train", "calibration", "test")
    }
    x = {role: [r["features"] for r in rows] for role, rows in splits.items()}
    syntax = {
        role: [
            dict(
                token_count=r["token_count"],
                format_valid=re.fullmatch(r"[A-Z]{2}[.!]?", r["answer"] or "") is not None,
            )
            for r in rows
        ]
        for role, rows in splits.items()
    }
    for role in splits:
        b.features(x[role])
        b.nuisance(syntax[role])
    b.progress("8952_before_frozen_training", 0, 600)
    state = b.fit(
        x["train"],
        [r["value"] for r in splits["train"]],
        syntax["train"],
        x["calibration"],
        [r["value"] for r in splits["calibration"]],
    )
    hashes = {
        arm: canonical_hash(
            dict(
                state=state[arm],
                scaling=state.get("scaling"),
                nuisance_scaling=state.get("nuisance_scaling"),
                temperature=state["temperature"],
            )
        )
        for arm in b.ARMS
    }
    hashes["complete"] = canonical_hash(state)
    seal(raw / "model_state.json", state, replay)
    b.progress("8952_after_state_freeze", 600, 0)
    probability = b.score(state, x["test"], syntax["test"])
    seal(raw / "test_probabilities.json", probability, replay)
    b.progress("8952_before_test_truth_open", 0, 80)
    truth = [r["value"] for r in splits["test"]]
    ids, clusters = ([r[k] for r in splits["test"]] for k in ("unit_id", "cluster_id"))
    rows = b.comparison_rows(probability, truth, ids, clusters)
    for row in rows:
        row.update(
            condition="matched_k2",
            censor_reason=None,
            fixture_only=False,
            correct=1 - row["error"],
            p_correct=1 - row["p_error"],
        )
    b.progress("8952_after_test_truth_open", 80, 0)
    intervals = b.summarize(rows)
    mutated = [
        dict(token_count=16 if r["token_count"] != 16 else 1, format_valid=not r["format_valid"])
        for r in syntax["test"]
    ]
    changed = b.score(state, x["test"], mutated)
    controls = [
        dict(
            control="syntax_length_isolation",
            passed=all(changed[a] == probability[a] for a in ("energy", "logistic", "prevalence")),
        )
    ]
    coverage, descriptive = [], []
    for arm in b.ARMS:
        arm_rows = [r for r in rows if r["arm"] == arm]
        accepted = [r for r in arm_rows if r["selected_50"]]
        errors = sum(r["error"] for r in accepted)
        coverage.append(
            dict(
                arm=arm,
                condition="diagnostic_50",
                metric="accepted_error",
                value=errors / 40,
                numerator=errors,
                denominator=40,
                selected_count=40,
                total_count=80,
                coverage=0.5,
            )
        )
        for metric, number in dict(
            brier=float(np.mean([r["value"] for r in arm_rows])),
            auroc=float(roc_auc_score(truth, probability[arm])),
            auprc=float(average_precision_score(truth, probability[arm])),
        ).items():
            descriptive.append(dict(arm=arm, metric=metric, value=number, scope="descriptive"))
    return dict(
        rows=rows,
        test_prediction_rows=rows,
        intended_count=80,
        completed_count=80,
        independent_count=80,
        failed_count=0,
        censored_count=0,
        excluded_count=0,
        numerical_training_calls=state["numerical_training_calls"],
        paired_intervals=intervals,
        coverage_rows=coverage,
        model_state_hashes=hashes,
        descriptive_rows=descriptive,
        oracle_access_controls=dict(
            control_rows=controls,
            test_label_open_count=1,
            state_and_predictions_sealed_before_truth=True,
        ),
        test_class_counts=dict(correct=truth.count(0), incorrect=truth.count(1)),
    )


def seal(path: Path, value: Json, replay: bool) -> None:
    """Replay compares sealed operands instead of replacing original producer bytes."""
    if replay:
        if json.loads(path.read_text()) != value:
            raise ValueError("sealed_operand_drift")
    else:
        atomic_json(path, value)


def build(work: Json, validation: Json, fixture: bool) -> Json:
    """Measurement readiness and scientific benefit require different evidence."""
    result, failures = work["result"], work["evidence"]["failures"]
    owned = shard.qualified(validation, OWNED) and not validation.get("terminal_rejected", False)
    controls = work["development"].get("control_rows", []) + result["oracle_access_controls"].get(
        "control_rows", []
    )
    control_pass = bool(controls) and all(r["passed"] for r in controls)
    ready = int(
        owned and control_pass and not failures and result["completed_count"] == 80 and not fixture
    )
    contrasts = result["paired_intervals"].get("contrasts", {})
    benefit = int(ready and len(contrasts) == 3 and all(r["passed"] for r in contrasts.values()))
    cls = (
        "disqualified"
        if not fixture and (not owned or (controls and not control_pass))
        else "blocked"
        if failures
        else "positive"
        if benefit
        else "null"
    )
    raw = Path(work["raw"])
    value: Json = dict(work["template"])
    value.update(
        result,
        experiment_id=EXPERIMENT_ID,
        task_id=TASK,
        milestone="2026.11.765",
        run_date="20261109",
        random_seed=b.SEED,
        honest_verdict=f"complete_{cls}_matched_population_final_answer_calibration",
        verdict_class=cls,
        status="error" if cls == "disqualified" else "blocked" if cls == "blocked" else "success",
        execution_venue="host",
        inference_substrate="verifier_ensemble_against_cached_candidates",
        inference_substrate_class="no_model_load",
        MODEL_SPECS=[],
        model_specs=[],
        model_invoked=False,
        model_invocation_counts=dict(ZERO_INVOCATION_COUNTS),
        numerical_training_calls=result["numerical_training_calls"]
        + work["development"].get("numerical_training_calls", 0),
        duration_s=work["ended"] - work["started"],
        started_at=work["started_at"],
        finished_at=work["finished_at"],
        phase_spans=work["phase_spans"],
        preconditions_checked=work["evidence"],
        verifier_is_oracle=False,
        exposure_scope="unseen graphs in restricted matched-cardinality Qwen response population; final answers only",
        independent_generalization_score=benefit,
        detector_rows_ready_score=ready,
        detector_benefit_score=benefit,
        gate_check_summary=failures,
        source_artifact_hashes=work["input_references"],
        code_config_hashes=work["sources"],
        raw_shard_hashes=[
            calibration.reference(raw / name)
            for name in (
                "work.json",
                "validation.json",
                "development.json",
                *(
                    ["model_state.json", "test_probabilities.json"]
                    if result["completed_count"]
                    else []
                ),
            )
        ],
        feature_allowlist=list(b.FEATURES),
        calibration_protocol_hash=canonical_hash(calibration.manifest()),
        frozen_calibration_protocol=calibration.manifest(),
        upstream_compute_references=[
            dict(r, scope="opaque_historical_model_receipts")
            for r in work["input_references"]
            if Path(r["path"]).name
            in {
                "experiment_8949_v765_matched_corpus_a.json",
                "experiment_8950_v765_matched_corpus_b.json",
            }
        ],
        acceptance_gates=dict(
            authenticated_inputs=not failures,
            complete_measurement=result["completed_count"] == 80,
            isolation_and_development=control_pass,
            owned_checks=owned,
            **{k: v["passed"] for k, v in result["paired_intervals"].get("contrasts", {}).items()},
        ),
        required_checks_passed=owned and (control_pass or bool(failures)),
        owned_statement_coverage=validation["coverage"],
        validation_receipts=validation["receipts"],
        global_test_debt=work["global_test_debt"],
        fixture_only=fixture,
        flagged_adversarial=bool(validation.get("terminal_rejected", False)),
        development_control_rows=work["development"].get("control_rows", []),
        development_numerical_training_calls=work["development"].get("numerical_training_calls", 0),
        live_numerical_training_calls=result["numerical_training_calls"],
        oracle_access_controls=dict(
            result["oracle_access_controls"],
            development_isolation=work["development"].get("feature_access_controls", {}),
        ),
        methodology_note="Frozen 160/80/80 graph split, unchanged two-feature Gibbs and logistic, 200 full-batch steps per head, calibration-only temperature. States and test probabilities sealed before truth. Four arms, 10000 paired graph bootstrap draws, Bonferroni family alpha0.05 across three one-sided bounds. Exactly50-percent diagnostic coverage uses predicted correctness and frozen label-blind ties, separately from typed thresholds. AUROC/AUPRC and class support are descriptive. Inconclusive bounds are a valid null; no further collection or test tuning. Scope excludes natural-language hallucinations and first-error localization.",
        reproduction_commands=[
            f"PYTHONPATH=python .venv/bin/python -u {CLI} --date 20261109",
            f".venv/bin/python -u {CLI} --cold-replay <primary.json>",
        ],
    )
    value["field_principles"] = {
        key: "Separate independent graph measurement, owned qualification and simultaneous scientific benefit."
        for key in [*value, "reproducibility_checksum"]
    }
    value["reproducibility_checksum"] = calibration.checksum(value)
    return value
