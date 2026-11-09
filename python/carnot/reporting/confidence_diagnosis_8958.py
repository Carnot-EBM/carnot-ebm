"""REQ-REPORT-8958: retain the confidence null and expose development custody.

Graph inputs can document an absent representation. They cannot establish why
confidence failed or predict whether a different representation will help.
"""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import re
from typing import Any

import numpy as np
from sklearn.metrics import roc_auc_score

from carnot.reporting import calibration_methods_8937 as calibration
from carnot.reporting import matched_calibrated_decisions_8952 as original
from carnot.reporting import matched_corpus_admission_8951 as admission
from carnot.reporting.current_work_receipt import ZERO_INVOCATION_COUNTS, canonical_hash

Json = dict[str, Any]
ROOT, b = admission.ROOT, calibration.b
NAME, TASK, SEED = (
    "experiment_8958_v766_confidence_diagnosis",
    "exp8958-confidence-diagnosis",
    766058,
)
EXPERIMENT_ID = 8958
CLI = f"scripts/experiments/{NAME}.py"
OWNED = [
    "python/carnot/reporting/confidence_diagnosis_8958.py",
    "python/carnot/reporting/confidence_diagnosis_runner_8958.py",
    CLI,
]
TESTS = ["tests/python/test_confidence_diagnosis_8958.py"]
INPUTS = [
    (
        f"results/{admission.NAME}.json",
        "sha256:21238c55f69ecf8f4bf8be563d96143e314a6cfaeba6f4c5dfa3fb15580e0fc7",
        admission.TASK,
    ),
    (
        f"results/{original.NAME}.json",
        "sha256:e1a4067c3369c6720b08cf1a1820e36edefd7c8c7621da0b47539ab5f784652c",
        original.TASK,
    ),
]


def authenticate(root: Path) -> Json:
    """Pin original producers and rebuild response features and labels before fitting."""
    failures, references, producers = [], [], []
    for name, pin, task in INPUTS:
        path = root / name
        ref = calibration.reference(path)
        references.append(ref)
        try:
            if ref["sha256"] != pin:
                raise ValueError("missing_file" if ref["sha256"] is None else "producer_hash")
            value = json.loads(path.read_text())
            for operand in (
                value["source_artifact_hashes"]
                + value["code_config_hashes"]
                + value["raw_shard_hashes"]
            ):
                if calibration.reference(Path(operand["path"]))["sha256"] != operand["sha256"]:
                    raise ValueError("retained_bytes_drift")
                references.append(operand)
            producers.append(value)
        except (OSError, ValueError, KeyError, TypeError) as error:
            failures.append(
                dict(
                    upstream=task,
                    **ref,
                    hash=ref["sha256"],
                    check=str(error),
                    artifact_field="sha256" if ref["sha256"] != pin else "retained_operands",
                    expected=pin if ref["sha256"] != pin else "authenticated original bytes",
                    observed=ref["sha256"] if ref["sha256"] != pin else str(error),
                )
            )
    result = dict(failures=failures, references=references, corpus=[], units=[], original={})
    if not failures:
        try:
            units, responses, old = admission.join(root)
            reduced = admission.reduce(units, responses, old)
            if (
                reduced["rows"] != producers[0]["rows"]
                or reduced["corpus_manifest"] != producers[0]["corpus_manifest"]
            ):
                raise ValueError("original_response_label_feature_drift")
            helper = Path(b.__file__)
            historical = [
                r
                for r in producers[1]["code_config_hashes"]
                if r.get("source_path", "").endswith("/" + helper.name)
            ]
            if (
                len(historical) != 1
                or calibration.reference(helper)["sha256"] != historical[0]["sha256"]
            ):
                raise ValueError("inference_helper_drift")
            result.update(corpus=reduced["rows"], units=units, original=producers[1])
        except (OSError, ValueError, KeyError, TypeError) as error:
            failures.append(
                dict(
                    upstream=admission.TASK,
                    path=str(root / INPUTS[0][0]),
                    hash=INPUTS[0][1],
                    check="rejected_producer",
                    artifact_field="original_response_label_feature_rows",
                    expected="320 original independently reconstructed rows and inference helper",
                    observed=str(error),
                )
            )
    return result


def views(units: list[Json], corpus: list[Json]) -> tuple[list[Json], list[Json]]:
    """Hash order uses no outcomes; absent candidates keep a node-free representation."""
    if (
        len(units) != 320
        or len(corpus) != 320
        or len({u["graph_hash"] for u in units}) != 320
        or {u["unit_id"] for u in units} != {r["unit_id"] for r in corpus}
    ):
        raise ValueError("development_roster")
    by_id = {r["unit_id"]: r for r in corpus}
    developer, labels = [], []
    for i, unit in enumerate(sorted(units, key=lambda u: u["graph_hash"])):
        row = by_id[unit["unit_id"]]
        answer = row["answer"]
        candidate = (
            answer.rstrip(".!")
            if isinstance(answer, str) and re.fullmatch(r"[A-Z]{2}[.!]?", answer)
            else None
        )
        node = candidate if candidate in unit["nodes"] else None
        developer.append(
            dict(
                unit_id=row["unit_id"],
                cluster_id=row["cluster_id"],
                graph_hash=unit["graph_hash"],
                original_role=row["role"],
                development_role="fit" if i < 256 else "guard",
                nodes=unit["nodes"],
                edges=unit["edges"],
                query_node=unit["start"],
                proposed_node=node,
                missing_node=int(node is None),
                raw_candidate=answer,
                features=row["features"],
                exposure_scope="exposed_development",
            )
        )
        labels.append(
            dict(
                unit_id=row["unit_id"],
                graph_hash=unit["graph_hash"],
                error=row["value"],
                parser_category=row["parser_category"],
                original_role=row["role"],
            )
        )
    b.features([r["features"] for r in developer])
    return developer, labels


def metric_rows(rows: list[Json]) -> list[Json]:
    """Metrics reduce typed predictions, retaining label-blind half coverage."""
    groups: dict[tuple[str, str], list[Json]] = {}
    for row in rows:
        groups.setdefault((row["arm"], row["condition"]), []).append(row)
    result = []
    for (arm, condition), group in groups.items():
        y = [r["error"] for r in group]
        p = [r["p_error"] for r in group]
        selected = b.half_selection(p, [r["unit_id"] for r in group])
        accepted_errors = sum(v for v, keep in zip(y, selected) if keep)
        for metric, value, numerator, denominator in [
            (
                "brier",
                float(np.mean([(a - c) ** 2 for a, c in zip(p, y)])),
                sum((a - c) ** 2 for a, c in zip(p, y)),
                len(y),
            ),
            ("auroc", float(roc_auc_score(y, p)) if len(set(y)) == 2 else None, None, len(y)),
            ("accepted_error_50", accepted_errors / (len(y) // 2), accepted_errors, len(y) // 2),
        ]:
            result.append(
                dict(
                    arm=arm,
                    condition=condition,
                    metric=metric,
                    value=value,
                    numerator=numerator,
                    denominator=denominator,
                    status="completed" if value is not None else "undefined_single_class",
                    censor_reason=None,
                )
            )
    return result


def original_reduction(evidence: Json) -> Json:
    """Recompute the original frozen split without training or changing its verdict."""
    value = evidence["original"]
    truth = {r["unit_id"]: r for r in evidence["corpus"] if r["role"] == "test"}
    rows = value["rows"]
    if len(rows) != 320 or any(
        r["unit_id"] not in truth or r["error"] != truth[r["unit_id"]]["value"] for r in rows
    ):
        raise ValueError("original_test_truth")
    for arm in b.ARMS:
        group = [r for r in rows if r["arm"] == arm]
        if (
            len(group) != 80
            or len({r["unit_id"] for r in group}) != 80
            or [r["selected_50"] for r in group]
            != b.half_selection([r["p_error"] for r in group], [r["unit_id"] for r in group])
        ):
            raise ValueError("original_selection")
    intervals = b.summarize(rows)
    metrics = metric_rows(rows)
    if (
        intervals != value["paired_intervals"]
        or any(
            r["value"]
            != next(
                x["value"]
                for x in value["descriptive_rows"]
                if x["arm"] == r["arm"] and x["metric"] == r["metric"]
            )
            for r in metrics
            if r["metric"] != "accepted_error_50"
        )
        or any(
            r["value"] != next(x["value"] for x in value["coverage_rows"] if x["arm"] == r["arm"])
            for r in metrics
            if r["metric"] == "accepted_error_50"
        )
    ):
        raise ValueError("original_reduction")
    return dict(
        original_rows=rows,
        original_metric_rows=metrics,
        original_paired_intervals=intervals,
        original_bound_rows=[dict(contrast=k, **v) for k, v in intervals["contrasts"].items()],
        original_honest_verdict=value["honest_verdict"],
    )


def development_fit(view: list[Json], labels: list[Json]) -> Json:
    """Resample only fit graphs; guard labels measure an exposed development comparison."""
    truth = {r["unit_id"]: r["error"] for r in labels}
    fit, guard = ([r for r in view if r["development_role"] == role] for role in ("fit", "guard"))
    output, states = [], []
    for number, seed in enumerate((71, 72, 73)):
        b.progress("8958_before_development_resample", number, 3 - number)
        rng = np.random.default_rng(seed)
        sample = [fit[i] for i in rng.integers(0, len(fit), len(fit))]
        x = b.features([r["features"] for r in sample])
        scaling = b._scaling(x)
        x = b._standardize(x, scaling)
        gx = b._standardize(b.features([r["features"] for r in guard]), scaling)
        y = b.labels([truth[r["unit_id"]] for r in sample], len(sample))
        py = rng.permutation(y)
        probabilities = dict(prevalence=[float(y.mean())] * len(guard))
        for permuted, target in ((False, y), (True, py)):
            for energy in (True, False):
                arm = ("permuted_" if permuted else "") + ("energy" if energy else "logistic")
                b.progress("8958_before_" + arm, number, 3 - number)
                params, _ = b._train(x, target, energy)
                logits = (
                    b._energy(tuple(b.jnp.asarray(v) for v in params), gx)
                    if energy
                    else gx @ params[:2] + params[2]
                )
                probabilities[arm] = b._sigmoid(logits).tolist()
                states.append(
                    dict(
                        seed=seed,
                        arm=arm,
                        params=params,
                        scaling=scaling,
                        sample_ids=[r["unit_id"] for r in sample],
                        training_labels_hash=canonical_hash(target.tolist()),
                        training_errors=int(target.sum()),
                        training_count=len(sample),
                    )
                )
                b.progress("8958_after_" + arm, number + 1, 2 - number)
        for arm, probability in probabilities.items():
            selected = b.half_selection(probability, [r["unit_id"] for r in guard])
            for row, p, keep in zip(guard, probability, selected, strict=True):
                error = truth[row["unit_id"]]
                loss = float((p - error) ** 2)
                output.append(
                    dict(
                        unit_id=row["unit_id"],
                        cluster_id=row["cluster_id"],
                        arm=arm,
                        condition=f"development_resample_{seed}",
                        seed=seed,
                        metric="brier",
                        value=loss,
                        numerator=loss,
                        denominator=1,
                        error=error,
                        p_error=p,
                        selected_50=keep,
                        status="completed",
                        censor_reason=None,
                        exposure_scope="exposed_development",
                    )
                )
        b.progress("8958_after_development_resample", number + 1, 2 - number)
    return dict(
        diagnostic_fit_rows=output,
        diagnostic_metric_rows=metric_rows(output),
        diagnostic_states=states,
        numerical_fit_count=12,
        numerical_training_calls=12 * b.STEPS,
    )


def diagnose(evidence: Json) -> Json:
    """Readiness is a custody result, independent of representation benefit."""
    b.progress("8958_before_original_reduction", 0, 320)
    result = original_reduction(evidence)
    b.progress("8958_after_original_reduction", 320, 0)
    view, labels = views(evidence["units"], evidence["corpus"])
    result.update(development_fit(view, labels))
    class_balance = {}
    for role in ("all", "fit", "guard", "train", "calibration", "test"):
        ids = {
            r["unit_id"]
            for r in view
            if role == "all" or r["development_role"] == role or r["original_role"] == role
        }
        errors = sum(r["error"] for r in labels if r["unit_id"] in ids)
        class_balance[role] = dict(
            count=len(ids),
            errors=errors,
            correct=len(ids) - errors,
            error_prevalence=errors / len(ids),
        )
    overlap = []
    for feature in b.FEATURES:
        classes = [
            [r["features"][feature] for r in evidence["corpus"] if r["value"] == label]
            for label in (0, 1)
        ]
        low, high = max(min(v) for v in classes), min(max(v) for v in classes)
        overlap.append(
            dict(
                feature=feature,
                correct_range=[min(classes[0]), max(classes[0])],
                error_range=[min(classes[1]), max(classes[1])],
                class_means=[float(np.mean(v)) for v in classes],
                overlap_range=[low, high],
                within_shared_range=[sum(low <= x <= high for x in v) for v in classes],
                class_counts=[len(v) for v in classes],
                scope="descriptive_marginal_overlap_not_causal",
            )
        )
    result.update(
        rows=evidence["corpus"],
        confidence_rows=evidence["corpus"],
        developer_rows=view,
        label_rows=labels,
        development_manifest=dict(
            algorithm="ascending unique graph_hash; first256 fit, remaining64 guard; no outcome operand",
            fit_count=256,
            guard_count=64,
            seed=SEED,
            units=[
                {k: r[k] for k in ("unit_id", "graph_hash", "original_role", "development_role")}
                for r in view
            ],
            exposure_scope="all320_graphs_exposed_development",
            confirmation_allowed=False,
        ),
        class_balance=class_balance,
        feature_overlap_rows=overlap,
        feature_access_rows=[
            dict(
                feature=k,
                available_at_inference=k in b.FEATURES,
                boundary="graph_calibration_benchmark.features exact key set; score passes standardized confidence only to Gibbs/logistic",
                helper=calibration.reference(Path(b.__file__)),
            )
            for k in (
                *b.FEATURES,
                "source_graph",
                "query_node",
                "proposed_node",
                "truth_label",
                "true_distance",
                "answer_set",
            )
        ],
        out_of_graph_count=sum(r["parser_category"] == "out_of_graph" for r in labels),
        intended_count=320,
        completed_count=320,
        independent_count=320,
        failed_count=0,
        censored_count=0,
        excluded_count=0,
    )
    return result


def empty_result() -> Json:
    """External absence leaves all scheduled graphs visible and censored."""
    rows = [
        dict(
            unit_id=f"unopened-{i:03d}",
            cluster_id=f"unopened-{i:03d}",
            arm="confidence_diagnosis",
            condition="exposed_development",
            metric="final_answer_error",
            value=None,
            numerator=None,
            denominator=1,
            status="censored",
            censor_reason="external_evidence",
        )
        for i in range(320)
    ]
    return dict(
        rows=rows,
        confidence_rows=[],
        original_bound_rows=[],
        feature_access_rows=[],
        diagnostic_fit_rows=[],
        developer_rows=[],
        label_rows=[],
        development_manifest={},
        intended_count=320,
        completed_count=0,
        independent_count=0,
        failed_count=0,
        censored_count=320,
        excluded_count=0,
        numerical_fit_count=0,
        numerical_training_calls=0,
    )


def build(work: Json, validation: Json, fixture: bool) -> Json:
    """Qualification closes this confirmation scope even when mechanism evidence is null."""
    owned = admission.shard_a.qualified(validation, OWNED)
    failures, result = work["evidence"]["failures"], work["result"]
    ready = int(owned and not failures and result["completed_count"] == 320 and not fixture)
    cls = "disqualified" if not owned and not fixture else "blocked" if failures else "null"
    value: Json = dict(result)
    value.pop("developer_rows")
    value.pop("label_rows")
    value.update(
        experiment_id=EXPERIMENT_ID,
        task_id=TASK,
        milestone="2026.11.766",
        run_date="20261109",
        random_seed=SEED,
        honest_verdict=f"complete_{cls}_development_only_confidence_diagnosis",
        verdict_class=cls,
        status="error" if cls == "disqualified" else "blocked" if cls == "blocked" else "success",
        inference_substrate="verifier_ensemble_against_cached_candidates"
        if result["numerical_training_calls"]
        else "aggregation_from_upstream_artifacts",
        inference_substrate_class="no_model_load",
        execution_venue="host",
        MODEL_SPECS=[],
        model_specs=[],
        model_invoked=False,
        model_invocation_counts=dict(ZERO_INVOCATION_COUNTS),
        duration_s=work["duration_s"],
        phase_spans=work["phase_spans"],
        preconditions_checked=dict(
            failures=failures,
            original_rows_reconstructed=not failures,
            historical_producers_are_retained_inputs=True,
        ),
        verifier_is_oracle=False,
        exposure_scope="all320_V765_graphs_exposed_development_including_original_test",
        independent_generalization_score=0,
        diagnosis_ready_score=ready,
        unchanged_confidence_confirmation_scope="closed; fresh independent graphs required for any confirmation",
        mechanism_evidence="Descriptive development comparisons cannot establish a causal failure or show graph context will help.",
        gate_check_summary=failures,
        source_artifact_hashes=work["input_references"],
        code_config_hashes=work["sources"],
        raw_shard_hashes=work["raw_references"],
        acceptance_gates={
            "authenticated_originals": dict(
                passed=not failures, principle="Custody precedes analysis."
            ),
            "complete_diagnosis": dict(
                passed=result["completed_count"] == 320,
                principle="Every scheduled graph remains represented.",
            ),
            "owned_checks": dict(
                passed=owned,
                principle="Readiness requires measured unit and child coverage and private controls.",
            ),
        },
        required_checks_passed=owned,
        owned_statement_coverage=validation["coverage"],
        validation_receipts=validation["receipts"],
        global_test_debt=work["global_test_debt"],
        fixture_only=fixture,
        flagged_adversarial=False,
        developer_view=work["developer_view"],
        protected_label_view=work["protected_label_view"],
        diagnostic_protocol=dict(
            seeds=[71, 72, 73],
            resample="256 fit graphs sampled with replacement; same64 development guard",
            heads="existing four-hidden-unit Gibbs and logistic; identical fixed initialization,200 full-batch steps,rate0.01; no tuning",
            permutation="independently permute each resample's training labels; preserve class balance; never permute guard truth",
            independent_graph_count=320,
            independent_guard_count=64,
            repeated_seeds_increase_independent_count=False,
        ),
        reproduction_commands=[
            f"PYTHONPATH=python .venv/bin/python -u {CLI} --date 20261109",
            f".venv/bin/python -u {CLI} --cold-replay <primary.json>",
        ],
    )
    if "original_paired_intervals" in value:
        value["methodology_note"] = (
            "Original80 test graphs and all three failed simultaneous bounds are unchanged. All320 graphs are now exposed development data; guard is a development role, never new confirmation. Audit of absent graph context is a representation boundary, not a causal explanation."
        )
    value["field_principles"] = {
        key: "Bind authentic development evidence, exact graph denominators, current work and honest null scope."
        for key in [*value, "field_principles", "reproducibility_checksum"]
    }
    value["field_principles"].update(
        diagnosis_ready_score="Correct reduction and custody determine readiness, regardless of mechanism benefit.",
        honest_verdict="Terminal development disposition prevents unchanged retries and preserves the original confidence null.",
        numerical_training_calls="Count2400 optimizer updates in12 small-head fits separately from zero LLM calls.",
        development_manifest="Outcome-independent256/64 roles contain only exposed development graphs.",
        protected_label_view="Separate mode0600 truth bytes from graph-and-candidate developer inputs.",
    )
    value["reproducibility_checksum"] = calibration.checksum(value)
    return value
