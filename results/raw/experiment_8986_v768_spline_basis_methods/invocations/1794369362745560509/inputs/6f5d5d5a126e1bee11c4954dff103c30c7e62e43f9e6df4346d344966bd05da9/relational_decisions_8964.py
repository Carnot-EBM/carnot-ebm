"""REQ-REPORT-8964: measure frozen graph scores without claiming oracle-free success.

Only calibration truth chooses temperatures. The scoring API receives neither
labels nor file paths, so exact graph truth cannot become an inference feature.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from carnot.memory.online_calibration_8960 import logit
from carnot.reporting import relational_methods_8959 as methods
from carnot.reporting import corpus_admission_8963 as admission
from carnot.reporting import matched_calibrated_decisions_8952 as prior
from carnot.reporting.current_work_receipt import ZERO_INVOCATION_COUNTS, canonical_hash
from carnot.reporting.corpus_views_8963 import release_final_test
from carnot.reporting.depth_opportunity_8945 import Rejected, require

Json = dict[str, Any]
ROOT, b, g, calibration = methods.ROOT, methods.b, methods.g, methods.calibration
NAME, TASK = "experiment_8964_v766_relational_decisions", "exp8964-relational-decisions"
EXPERIMENT_ID, SEED = 8964, methods.SEED
CLI = f"scripts/experiments/{NAME}.py"
OWNED = [
    "python/carnot/reporting/relational_decisions_8964.py",
    "python/carnot/reporting/relational_decisions_runner_8964.py",
    CLI,
]
TESTS = ["tests/python/test_relational_decisions_8964.py"]
INPUTS = [
    (*admission.INPUTS[0],),
    (
        f"results/{admission.NAME}.json",
        "sha256:586a8f5bff118cd9416f0f03cf9606bc35db0eb4068d946bbf17105329547d24",
        "corpus_ready_score",
        admission.TASK,
    ),
]
THRESHOLDS = dict(accept_max=0.1, reject_min=0.9, otherwise="escalate")


def authenticate(root: Path) -> Json:
    """Check exact producers and operands without rerunning retired scheduler gates."""
    result: Json = dict(failures=[], references=[], producers=[])
    for name, pin, field, task in INPUTS:
        path = root / name
        ref = calibration.reference(path)
        result["references"].append(ref)
        try:
            require(ref["sha256"], pin, path, "sha256", task)
            value = json.loads(path.read_text())
            if not isinstance(value, dict):
                raise ValueError("producer_object")
            for key, expected in ((field, 1), ("task_id", task), ("required_checks_passed", True)):
                require(value.get(key), expected, path, key, task)
            admission.shard.verify_references(value)
            result["references"].extend(
                value["source_artifact_hashes"]
                + value["code_config_hashes"]
                + value["raw_shard_hashes"]
            )
            result["producers"].append(value)
        except Rejected as error:
            result["failures"].append(dict(error.receipt, upstream=task))
        except (OSError, ValueError, KeyError, TypeError) as error:
            result["failures"].append(
                dict(
                    upstream=task,
                    path=str(path),
                    hash=ref["sha256"],
                    check="producer_structure",
                    artifact_field="producer_json",
                    expected="original complete object",
                    observed=str(error),
                    failure_kind="rejected_producer",
                )
            )
    return result


def score(service: Json, rows: list[Json]) -> Json:
    """Return calibrated probabilities and finite ensemble logits using allowed inputs only."""
    components, probabilities = {}, {}
    for arm, _, _ in methods.TRAINED_ARMS:
        values = [
            g.predict(s, rows, service["temperatures"][arm]).tolist()
            for s in service["states"][arm]
        ]
        components[arm] = values
        probabilities[arm] = np.mean(values, axis=0).tolist()
    probabilities["prevalence"] = [service["prevalence_probability"]] * len(rows)
    return dict(
        probabilities=probabilities,
        seed_probabilities=components,
        relational_logits=[logit(p) for p in probabilities["relational_energy"]],
    )


def summarize(probabilities: Json, truth: list[int], ids: list[str]) -> Json:
    """Paired draws keep the same graphs in all arms and rerank half within each draw."""
    y = np.asarray(truth)
    p = np.asarray([probabilities[a] for a in methods.ARMS]).T
    n = len(ids)
    ties = [hashlib.sha256(("8937/coverage/" + u).encode()).hexdigest() for u in ids]
    ranks = np.argsort(np.argsort(ties))

    def contrasts(indexes: Any) -> Any:
        sampled, labels = p[indexes], y[indexes]
        losses = ((sampled - labels[:, :, None]) ** 2).mean(axis=1)
        errors = []
        for arm in (0, 3, 1):
            order = np.lexsort(
                (np.broadcast_to(np.arange(n), indexes.shape), ranks[indexes], sampled[:, :, arm]),
                axis=1,
            )[:, : n // 2]
            errors.append(np.take_along_axis(labels, order, axis=1).mean(axis=1))
        return np.column_stack(
            [losses[:, k] - losses[:, 0] for k in (4, 3, 1)]
            + [errors[0] - errors[k] for k in (1, 2)]
        )

    rng = np.random.default_rng(SEED)
    draws = []
    b.progress("8964_before_paired_graph_bootstrap", 0, 10000)
    for completed in range(1000, 10001, 1000):
        draws.append(contrasts(rng.integers(0, n, (1000, n))))
        b.progress("8964_paired_graph_bootstrap", completed, 10000 - completed)
    samples = np.concatenate(draws)
    point = contrasts(np.arange(n)[None, :])[0]
    names = [
        "brier_vs_prevalence",
        "brier_vs_confidence_logistic",
        "brier_vs_relational_logistic",
        "accepted_error_vs_confidence_logistic",
        "accepted_error_vs_relational_logistic",
    ]
    result = {}
    for k, name in enumerate(names):
        bound = float(np.quantile(samples[:, k], 0.01 if k < 3 else 0.99, method="linear"))
        result[name] = dict(
            estimate=float(point[k]),
            one_sided_bound=bound,
            direction="lower" if k < 3 else "upper",
            passed=bool(bound > 0 if k < 3 else bound <= 0),
        )
    b.progress("8964_after_paired_graph_bootstrap", 10000, 0)
    return dict(
        draws=10000,
        bootstrap_seed=SEED,
        family_alpha=0.05,
        alpha_per_contrast=0.01,
        independent_clusters=n,
        multiplicity="Bonferroni five one-sided percentile bounds",
        contrasts=result,
        benefit_passed=all(r["passed"] for r in result.values()),
    )


def reduce(probabilities: Json, truth: list[int], ids: list[str]) -> Json:
    """Every descriptive headline and typed decision retains its graph-level operands."""
    y = b.labels(truth, len(ids))
    if len(ids) != 128 or len(set(ids)) != 128 or set(probabilities) != set(methods.ARMS):
        raise ValueError("test_roster")
    rows, typed, coverage, descriptive = [], [], [], []
    for arm in methods.ARMS:
        p: Any = np.asarray(probabilities[arm], dtype=float)
        selected = b.half_selection(p, ids)
        clipped = np.clip(p, 1e-15, 1 - 1e-15)
        losses = -(y * np.log(clipped) + (1 - y) * np.log1p(-clipped))
        actions = b.decisions(p.tolist())
        for i, unit in enumerate(ids):
            common = dict(
                unit_id=unit,
                cluster_id=unit.removesuffix("-k2"),
                arm=arm,
                condition="fresh_matched_graph_final_answers",
                status="completed",
                censor_reason=None,
                denominator=1,
            )
            loss = float((p[i] - y[i]) ** 2)
            rows.append(
                dict(
                    common,
                    metric="brier",
                    value=loss,
                    numerator=loss,
                    p_error=float(p[i]),
                    error=int(y[i]),
                    selected_50=selected[i],
                    decision=actions[i],
                    log_loss=float(losses[i]),
                )
            )
            typed.append(
                dict(
                    common,
                    metric="typed_decision",
                    value=actions[i],
                    numerator=int(actions[i] == "accept"),
                    error=int(y[i]),
                    accepted_error=int(actions[i] == "accept" and y[i] == 1),
                )
            )
        for condition, keep in (
            ("diagnostic_50", selected),
            ("typed_accept", [a == "accept" for a in actions]),
        ):
            count = sum(keep)
            errors = int(sum(y[np.asarray(keep)]))
            coverage.append(
                dict(
                    arm=arm,
                    condition=condition,
                    metric="accepted_error",
                    value=errors / count if count else None,
                    numerator=errors,
                    denominator=count,
                    selected_count=count,
                    total_count=128,
                    coverage=count / 128,
                )
            )
        for metric, value in dict(
            brier=float(np.mean((p - y) ** 2)),
            log_loss=float(losses.mean()),
            auroc=float(roc_auc_score(y, p)),
            auprc=float(average_precision_score(y, p)),
        ).items():
            descriptive.append(dict(arm=arm, metric=metric, value=value, scope="descriptive"))
    return dict(
        rows=rows,
        typed_decision_rows=typed,
        coverage_rows=coverage,
        descriptive_rows=descriptive,
        paired_intervals=summarize(probabilities, truth, ids),
    )


def empty_result() -> Json:
    """Missing evidence censors every registered test unit in all five arms."""
    return dict(
        rows=[
            dict(
                unit_id=f"unopened-test-{i:03d}",
                cluster_id=f"unopened-test-{i:03d}",
                arm=arm,
                condition="fresh_matched_graph_final_answers",
                metric="brier",
                value=None,
                numerator=None,
                denominator=1,
                status="censored",
                censor_reason="external_evidence",
            )
            for i in range(128)
            for arm in methods.ARMS
        ],
        intended_count=128,
        completed_count=0,
        independent_count=0,
        failed_count=0,
        censored_count=128,
        excluded_count=0,
        numerical_training_calls=0,
        test_prediction_rows=[],
        typed_decision_rows=[],
        paired_intervals={},
        coverage_rows=[],
        descriptive_rows=[],
        model_state_hashes={},
        oracle_access_controls={},
        calibrated_state_reference=None,
    )


def read_reference(ref: Json) -> Json:
    """Hash before decoding so even specially referenced views stay under custody."""
    path = Path(ref["path"])
    require(calibration.reference(path)["sha256"], ref["sha256"], path, "sha256", TASK)
    return dict(json.loads(path.read_text()))


def evaluate(evidence: Json, raw: Path, *, replay: bool = False) -> Json:
    """Freeze all adaptive choices and complete predictions before opening test truth."""
    method, corpus = evidence["producers"]
    if method["frozen_protocol"] != methods.protocol():
        raise ValueError("frozen_protocol_drift")
    b.progress("8964_before_frozen_small_head_state_read", 0, 12)
    trained = read_reference(method["trained_state_reference"])
    state_hashes = [
        dict(arm=arm, seed=s["seed"], sha256=canonical_hash(s))
        for arm, _, _ in methods.TRAINED_ARMS
        for s in trained["states"][arm]
    ]
    if state_hashes != method["trained_state_hashes"] or any(
        [s["seed"] for s in trained["states"][a]] != [71, 72, 73]
        for a, _, _ in methods.TRAINED_ARMS
    ):
        raise ValueError("frozen_seed_components")
    views = {
        role: read_reference(corpus["inference_views"][role]) for role in ("calibration", "test")
    }
    for role, view in views.items():
        roster = [r["unit_id"] for r in corpus["corpus_manifest"] if r["role"] == role]
        if (
            view["unit_ids"] != roster
            or len(roster) != 128
            or len(set(roster)) != 128
            or len(view["rows"]) != 128
        ):
            raise ValueError("inference_view_roster")
        g.pack(view["rows"])
    if set(views["calibration"]["unit_ids"]) & set(views["test"]["unit_ids"]):
        raise ValueError("role_overlap")
    cal = read_reference(corpus["broker_manifest"]["calibration"]["label_reference"])["rows"]
    if [r["unit_id"] for r in cal] != views["calibration"]["unit_ids"]:
        raise ValueError("calibration_label_roster")
    truth = [r["label"] for r in cal]
    b.labels(truth, 128)
    b.progress("8964_after_frozen_small_head_state_read", 12, 0)
    temperatures = {}
    b.progress("8964_before_calibration_only_temperature", 0, 5)
    for arm, _, _ in methods.TRAINED_ARMS:
        temperatures[arm] = g.select_temperature(
            trained["states"][arm], views["calibration"]["rows"], truth
        )
        b.progress("8964_calibration_temperature", len(temperatures), 5 - len(temperatures))
    prevalence_logit = logit(trained["prevalence"])
    temperatures["prevalence"] = min(
        b.GRID, key=lambda t: float(np.mean((b._sigmoid(prevalence_logit / t) - truth) ** 2))
    )
    service = dict(
        states=trained["states"],
        temperatures=temperatures,
        prevalence_probability=float(b._sigmoid(prevalence_logit / temperatures["prevalence"])),
        thresholds=THRESHOLDS,
        protocol=method["frozen_protocol"],
        historical_state_reference=method["trained_state_reference"],
        calibration_ids=views["calibration"]["unit_ids"],
        calibration_label_hash=canonical_hash(truth),
    )
    prior.seal(raw / "calibrated_states.json", service, replay)
    prior.seal(raw / "fixed_thresholds.json", THRESHOLDS, replay)
    b.progress("8964_after_calibration_only_temperature", 5, 0)
    b.progress("8964_before_frozen_test_prediction", 0, 128)
    scored = score(service, views["test"]["rows"])
    predictions = [
        dict(
            unit_id=unit,
            p_error=scored["probabilities"]["relational_energy"][i],
            probabilities={a: scored["probabilities"][a][i] for a in methods.ARMS},
            seed_probabilities={
                a: [s[i] for s in values] for a, values in scored["seed_probabilities"].items()
            },
            decisions={a: b.decisions([scored["probabilities"][a][i]])[0] for a in methods.ARMS},
            relational_logit=scored["relational_logits"][i],
        )
        for i, unit in enumerate(views["test"]["unit_ids"])
    ]
    seal = raw / "test_predictions.json"
    prior.seal(
        seal,
        dict(
            rows=predictions,
            prediction_hash=canonical_hash(predictions),
            service_hash=canonical_hash(service),
            thresholds=THRESHOLDS,
        ),
        replay,
    )
    b.progress("8964_after_frozen_test_prediction", 128, 0)
    label_ref = corpus["broker_manifest"]["test"]["label_reference"]
    require(
        calibration.reference(Path(label_ref["path"]))["sha256"],
        label_ref["sha256"],
        Path(label_ref["path"]),
        "sha256",
        admission.TASK,
    )
    b.progress("8964_before_final_test_truth_release", 0, 128)
    test_truth = release_final_test(Path(label_ref["path"]), seal)
    result = reduce(scored["probabilities"], test_truth, views["test"]["unit_ids"])
    b.progress("8964_after_final_test_truth_release", 128, 0)
    return dict(
        result,
        test_prediction_rows=predictions,
        intended_count=128,
        completed_count=128,
        independent_count=128,
        failed_count=0,
        censored_count=0,
        excluded_count=0,
        numerical_training_calls=0,
        model_state_hashes=state_hashes,
        calibrated_state_reference=calibration.reference(raw / "calibrated_states.json"),
        inference_test_reference=corpus["inference_views"]["test"],
        test_class_counts=dict(correct=test_truth.count(0), incorrect=test_truth.count(1)),
        oracle_access_controls=dict(
            test_label_open_count=1,
            state_and_predictions_sealed_before_truth=True,
            no_inference_traversal=True,
            no_inference_label_access=True,
            allowed_input_keys=sorted(g.KEYS),
            frozen_state_unchanged=canonical_hash(service)
            == canonical_hash(
                read_reference(calibration.reference(raw / "calibrated_states.json"))
            ),
        ),
    )


def build(work: Json, validation: Json, fixture: bool) -> Json:
    """Readiness qualifies predictions; all simultaneous bounds separately qualify benefit."""
    result, failures = work["result"], work["evidence"]["failures"]
    owned = admission.shard.qualified(validation, OWNED) and not validation.get(
        "terminal_rejected", False
    )
    controls = result["oracle_access_controls"]
    isolated = bool(controls) and all(
        controls[k]
        for k in (
            "state_and_predictions_sealed_before_truth",
            "no_inference_traversal",
            "no_inference_label_access",
            "frozen_state_unchanged",
        )
    )
    ready = int(
        owned and isolated and not failures and result["completed_count"] == 128 and not fixture
    )
    benefit = int(
        ready
        and result["paired_intervals"].get("benefit_passed", False)
        and len(result["paired_intervals"].get("contrasts", {})) == 5
        and all(r["passed"] for r in result["paired_intervals"]["contrasts"].values())
    )
    cls = (
        "disqualified"
        if not fixture
        and (not owned or work.get("owned_failure") or (not failures and not isolated))
        else "blocked"
        if failures
        else "positive"
        if benefit
        else "null"
    )
    verdict = f"complete_{cls}_fresh_matched_graph_final_answers"
    previous = work.get("prior_terminal", {})
    if (
        cls in ("blocked", "disqualified")
        and previous.get("verdict_class") == cls
        and previous.get("gate_check_summary") == failures
    ):
        verdict = previous["honest_verdict"]
    value: Json = dict(
        result,
        experiment_id=EXPERIMENT_ID,
        task_id=TASK,
        milestone="2026.11.766",
        run_date="20261110",
        random_seed=SEED,
        honest_verdict=verdict,
        verdict_class=cls,
        status="error" if cls == "disqualified" else "blocked" if failures else "success",
        inference_substrate="verifier_ensemble_against_cached_candidates",
        inference_substrate_class="no_model_load",
        execution_venue="host",
        MODEL_SPECS=[],
        model_specs=[],
        model_invoked=False,
        model_invocation_counts=dict(ZERO_INVOCATION_COUNTS),
        numerical_training_calls=0,
        duration_s=work["ended"] - work["started"],
        phase_spans=work["phase_spans"],
        preconditions_checked=dict(failures=failures, historical_tasks_are_retained_inputs=True),
        verifier_is_oracle=not isolated,
        exposure_scope="fresh_matched_graph_final_answers",
        independent_generalization_score=0,
        detector_rows_ready_score=ready,
        detector_benefit_score=benefit,
        gate_check_summary=failures,
        source_artifact_hashes=work["evidence"]["references"],
        code_config_hashes=work["sources"],
        raw_shard_hashes=[
            calibration.reference(Path(work["raw"]) / "work.json"),
            *work["raw_references"],
        ],
        required_checks_passed=owned,
        owned_statement_coverage=validation["coverage"],
        validation_receipts=validation["receipts"],
        global_test_debt=work["global_test_debt"],
        frozen_logit_service=dict(
            qualified=bool(ready),
            callable="carnot.reporting.relational_decisions_8964.score",
            arm="relational_energy",
            state_reference=result["calibrated_state_reference"],
            seeds=[71, 72, 73],
            output="relational_logits",
            representation_frozen=True,
            logit_definition="logit(clamp(mean calibrated seed probabilities,1e-6,1-1e-6))",
            deployment_promotion=False,
        ),
        fixture_only=fixture,
        flagged_adversarial=False,
        methodology_note="Five frozen arms; historical weights unchanged. Temperatures use128 calibration graphs only; all states, thresholds and128 predictions precede test labels. Five Bonferroni bounds use10000 paired graph draws; exact half is reranked per draw. Typed thresholds give no risk certificate. Exact oracle labels are evaluation-only. Within-population success cannot close the broader no-cheap-oracle north star.",
    )
    gates = dict(
        authenticated_inputs=not failures,
        complete_predictions=result["completed_count"] == 128,
        oracle_isolation=isolated,
        owned_checks=owned,
    )
    gates.update(
        {k: r["passed"] for k, r in result["paired_intervals"].get("contrasts", {}).items()}
    )
    value["acceptance_gates"] = {
        k: dict(
            passed=v,
            principle="Readiness requires custody and owned checks; scientific benefit requires all five registered comparisons.",
        )
        for k, v in gates.items()
    }
    value["field_principles"] = {
        k: "Bind scheduled identity, authentic frozen operands, measured clocks and independent graph rows to a terminal scoped result."
        for k in [*value, "field_principles", "reproducibility_checksum"]
    }
    value["field_principles"].update(
        detector_rows_ready_score="Complete authenticated predictions and owned validation qualify a separate calibration study even for a null.",
        detector_benefit_score="All original confidence-control conditions and both new relational-control bounds must pass.",
        independent_generalization_score="Cheap exact graph truth leaves the broader oracle-free north star open.",
        model_invocation_counts="Current generator calls are zero; retained Qwen generations and historical training are separate inputs.",
        gate_check_summary="Exact upstream fields and values distinguish absence, observed zero and rejected evidence.",
    )
    for fields, principle in (
        (
            "experiment_id task_id milestone run_date random_seed reproducibility_checksum",
            "Bind the result to the scheduled task, input bytes and reproducible randomness.",
        ),
        (
            "honest_verdict verdict_class",
            "An honest terminal scope avoids wasted retries; oracle inference or failed acceptance gates forbid positive.",
        ),
        (
            "inference_substrate inference_substrate_class execution_venue MODEL_SPECS model_specs model_invocation_counts numerical_training_calls",
            "Use the actual legal substrate and venue; distinguish retained outputs, small-head training and actual generator calls.",
        ),
        (
            "duration_s phase_spans preconditions_checked",
            "Actual clocks and resource receipts prevent fabricated execution and duration padding.",
        ),
        (
            "verifier_is_oracle exposure_scope independent_generalization_score",
            "Separate exact evaluation labels from learned inference; graph success cannot establish the no-cheap-oracle north star.",
        ),
        (
            "intended_count completed_count failed_count censored_count excluded_count independent_count rows",
            "Retain scheduled units and typed per-graph operands; repeated seeds never enlarge the independent denominator.",
        ),
        (
            "source_artifact_hashes code_config_hashes raw_shard_hashes field_principles",
            "Retain content custody and explain each field so later readers can verify its purpose.",
        ),
        (
            "acceptance_gates required_checks_passed owned_statement_coverage validation_receipts global_test_debt",
            "Owned failures disqualify; measured coverage has zero exclusions; global debt cannot waive owned failures.",
        ),
        (
            "test_prediction_rows typed_decision_rows paired_intervals model_state_hashes frozen_logit_service oracle_access_controls",
            "Expose per-graph evidence and the exact frozen inference path; qualified nulls support a separate calibration study, never deployment promotion.",
        ),
    ):
        value["field_principles"].update({field: principle for field in fields.split()})
    value["reproducibility_checksum"] = calibration.checksum(value)
    return value
