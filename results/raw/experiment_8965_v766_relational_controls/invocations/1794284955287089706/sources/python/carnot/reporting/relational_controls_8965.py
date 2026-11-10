"""REQ-REPORT-8965: challenge frozen scores without adapting to held-out failures."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

from carnot.reporting import relational_decisions_8964 as prior
from carnot.reporting.current_work_receipt import (
    ZERO_INVOCATION_COUNTS,
    atomic_json,
    canonical_hash,
)
from carnot.reporting.depth_opportunity_8945 import Rejected, require
from carnot.reporting.experiment_7303_validation_scope import CommandSpec, run_commands

Json = dict[str, Any]
ROOT, calibration, b = prior.ROOT, prior.calibration, prior.b
NAME, TASK, SEED = "experiment_8965_v766_relational_controls", "exp8965-relational-controls", 766065
CLI = f"scripts/experiments/{NAME}.py"
OWNED = [
    "python/carnot/reporting/relational_controls_8965.py",
    "python/carnot/reporting/relational_controls_runner_8965.py",
    "python/carnot/autoresearch/relational_controls_worker.py",
    CLI,
]
TESTS = ["tests/python/test_relational_controls_8965.py"]
INPUT = "results/experiment_8964_v766_relational_decisions.json"
PIN = "sha256:f70d9becc3486f2dec4f8c0c9400cb6ebb49c1bd1fcd0ae67e36b56f74872408"
ABLATIONS = (
    "adjacency_removed",
    "source_marker_removed",
    "answer_marker_removed",
    "confidence_fit_means",
)
ref = calibration.reference


def authenticate(root: Path) -> Json:
    """Authenticate original bytes and retained operands without scheduler dependencies."""
    path = root / INPUT
    result: Json = dict(failures=[], references=[ref(path)])
    try:
        require(ref(path)["sha256"], PIN, path, "sha256", prior.TASK)
        value = json.loads(path.read_text())
        for key, expected in (
            ("task_id", prior.TASK),
            ("detector_rows_ready_score", 1),
            ("required_checks_passed", True),
        ):
            require(value.get(key), expected, path, key, prior.TASK)
        references = (
            value["source_artifact_hashes"]
            + value["code_config_hashes"]
            + value["raw_shard_hashes"]
        )
        unique = {r["path"]: r for r in references}
        for i, operand in enumerate(unique.values()):
            require(
                ref(Path(operand["path"]))["sha256"],
                operand["sha256"],
                Path(operand["path"]),
                "sha256",
                prior.TASK,
            )
            if i % 500 == 0:
                b.progress("8965_authenticate_operands", i, len(unique) - i)
        result["references"].extend(unique.values())
        result["producer"] = value
        result["service"] = prior.read_reference(value["calibrated_state_reference"])
        corpus_ref = next(
            r
            for r in references
            if r["path"].endswith("experiment_8963_v766_corpus_admission.json")
        )
        result["corpus"] = prior.read_reference(corpus_ref)
        fit = prior.read_reference(result["corpus"]["inference_views"]["fit"])["rows"]
        result["service"]["fit_means"] = {
            k: float(np.mean([r["features"][k] for r in fit])) for k in b.FEATURES
        }
    except Rejected as error:
        result["failures"].append(error.receipt)
    except (OSError, ValueError, KeyError, TypeError, AttributeError, StopIteration) as error:
        result["failures"].append(
            dict(
                upstream=prior.TASK,
                path=str(path),
                hash=ref(path)["sha256"],
                check="producer_structure",
                artifact_field="producer_json",
                expected="original complete object",
                observed=str(error),
                failure_kind="rejected_producer",
            )
        )
    return result


def panel(rows: list[Json], service: Json) -> Json:
    """Freeze independent bijections and orderings using no label or oracle operand."""
    rng = np.random.default_rng(SEED)
    conditions, records = {"baseline": deepcopy(rows)}, []
    for repeat in range(4):
        name, transformed = f"permutation_{repeat}", []
        for i, row in enumerate(rows):
            nodes = row["nodes"]
            bijection = dict(
                zip(nodes, [f"n{k:02d}" for k in rng.permutation(len(nodes))], strict=True)
            )
            edge_order = rng.permutation(len(row["edges"])).tolist()
            transformed.append(
                dict(
                    nodes=[bijection[nodes[k]] for k in rng.permutation(len(nodes))],
                    edges=[[bijection[n] for n in row["edges"][k]] for k in edge_order],
                    query_node=bijection[row["query_node"]],
                    proposed_node=None
                    if row["proposed_node"] is None
                    else bijection[row["proposed_node"]],
                    features=deepcopy(row["features"]),
                )
            )
            records.append(
                dict(condition=name, graph_index=i, bijection=bijection, edge_order=edge_order)
            )
        conditions[name] = transformed
    for name in ABLATIONS:
        conditions[name] = deepcopy(rows)
    means = service["fit_means"]
    for row in conditions["confidence_fit_means"]:
        row["features"] = dict(means)
    return dict(conditions=conditions, permutations=records)


def child(request: Json, raw: Path) -> Json:
    """Forward bounded scorer output; file paths never enter its scoring envelope."""
    request = dict(
        request,
        denial_paths=request.get(
            "denial_paths", [str(raw / "test_labels.json"), str(raw / "oracle_distances.json")]
        ),
        service={
            k: request["service"][k] for k in ("states", "temperatures", "prevalence_probability")
        },
    )
    atomic_json(raw / "request.json", request)
    receipts = run_commands(
        ROOT,
        [
            CommandSpec(
                "isolated_score",
                (
                    sys.executable,
                    "-u",
                    "-m",
                    "carnot.autoresearch.relational_controls_worker",
                    str(raw / "request.json"),
                    str(raw / "response.json"),
                ),
                "owned_scoring",
                180,
            )
        ],
        log_dir=raw / "logs",
        extra_env={"JAX_PLATFORMS": "cpu"},
        heartbeat_s=15,
    )
    if not receipts[0]["passed"]:
        raise ValueError("scoring_child:" + receipts[0]["output_tail"])
    return dict(json.loads((raw / "response.json").read_text()))


def empty_result() -> Json:
    """External absence retains every scheduled graph and condition as censored."""
    return dict(
        intended_count=128,
        completed_count=0,
        failed_count=0,
        censored_count=128,
        excluded_count=0,
        independent_count=0,
        permutation_rows=[],
        ablation_rows=[],
        oracle_mutation_rows=[],
        maximum_probability_drift=None,
        paired_intervals=[],
        custody_controls={},
        detector_benefit_score=0,
        rows=[
            dict(
                unit_id=f"unopened-{i:03d}",
                cluster_id=f"unopened-{i:03d}",
                arm=arm,
                condition=condition,
                metric="brier_difference",
                value=None,
                numerator=None,
                denominator=1,
                status="censored",
                censor_reason="external_evidence",
            )
            for i in range(128)
            for arm in prior.methods.ARMS
            for condition in ABLATIONS
        ],
    )


def evaluate(evidence: Json, raw: Path, *, replay: bool = False) -> Json:
    """Freeze the perturbation panel, score without truth, then reduce paired diagnostics."""
    producer, service, corpus = evidence["producer"], evidence["service"], evidence["corpus"]
    view = prior.read_reference(producer["inference_test_reference"])
    ids, rows = view["unit_ids"], view["rows"]
    if len(ids) != 128 or len(set(ids)) != 128 or len(rows) != 128:
        raise ValueError("test_roster")
    frozen = panel(rows, service)
    prior.prior.seal(raw / "panel.json", frozen, replay)
    frozen_hash = canonical_hash(service)
    b.progress("8965_before_private_oracle_materialization", 0, 128)
    truth = prior.read_reference(corpus["broker_manifest"]["test"]["label_reference"])
    if [r["unit_id"] for r in truth["rows"]] != ids:
        raise ValueError("truth_roster")
    y = b.labels([r["label"] for r in truth["rows"]], 128)
    distances = dict(
        rows=[
            dict(
                unit_id=unit,
                distances=prior.admission.all_pairs(r["nodes"], r["edges"])[r["query_node"]],
            )
            for unit, r in zip(ids, rows, strict=True)
        ]
    )
    b.progress("8965_after_private_oracle_materialization", 128, 0)
    private = raw / "private_oracles"
    label_path, distance_path = private / "test_labels.json", private / "oracle_distances.json"
    for path, value in ((label_path, truth), (distance_path, distances)):
        prior.prior.seal(path, value, replay)
    request = dict(
        service=service,
        conditions=frozen["conditions"],
        denial_paths=[str(label_path), str(distance_path)],
    )
    scored = child(request, raw / "scoring")
    predictions = scored["predictions"]
    baseline = predictions["baseline"]
    permutation_rows, ablation_rows, intervals = [], [], []
    max_drift = 0.0
    for condition, arms in predictions.items():
        for arm, values in arms.items():
            p = np.asarray(values["mean"])
            original = np.asarray(baseline[arm]["mean"])
            if p.shape != (128,) or not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
                raise ValueError("probability_range")
            drift = np.max(np.abs(p - original))
            if condition.startswith("permutation"):
                seeds = np.asarray(values["seeds"])
                if seeds.size:
                    drift = max(drift, np.max(np.abs(seeds - np.asarray(baseline[arm]["seeds"]))))
                max_drift = max(max_drift, float(drift))
            deltas = (p - y) ** 2 - (original - y) ** 2
            for i, unit in enumerate(ids):
                common = dict(
                    unit_id=unit,
                    cluster_id=unit.removesuffix("-k2"),
                    arm=arm,
                    condition=condition,
                    denominator=1,
                    status="completed",
                    censor_reason=None,
                    p_error=float(p[i]),
                    baseline_probability=float(original[i]),
                )
                if condition.startswith("permutation"):
                    permutation_rows.append(
                        dict(
                            common,
                            metric="absolute_probability_drift",
                            value=float(abs(p[i] - original[i])),
                            numerator=float(abs(p[i] - original[i])),
                        )
                    )
                if condition in ABLATIONS:
                    ablation_rows.append(
                        dict(
                            common,
                            metric="brier_difference",
                            value=float(deltas[i]),
                            numerator=float(deltas[i]),
                            error=int(y[i]),
                            brier=float((p[i] - y[i]) ** 2),
                        )
                    )
            if condition in ABLATIONS:
                rng = np.random.default_rng(SEED)
                draws = np.concatenate(
                    [deltas[rng.integers(0, 128, (1000, 128))].mean(axis=1) for _ in range(10)]
                )
                intervals.append(
                    dict(
                        condition=condition,
                        arm=arm,
                        estimate=float(deltas.mean()),
                        ci95=np.quantile(draws, [0.025, 0.975]).tolist(),
                        draws=10000,
                        independent_clusters=128,
                        seed=SEED,
                        interpretation="distribution_shift_mechanism_diagnostic",
                    )
                )
    retained = {
        a: [r["probabilities"][a] for r in producer["test_prediction_rows"]]
        for a in prior.methods.ARMS
    }
    baseline_drift = max(
        float(np.max(np.abs(np.asarray(baseline[a]["mean"]) - retained[a]))) for a in retained
    )
    mutation_rows = []
    for name, path, before, after in (
        (
            "test_labels",
            label_path,
            truth,
            dict(rows=[dict(r, label=1 - r["label"]) for r in truth["rows"]]),
        ),
        (
            "oracle_distances",
            distance_path,
            distances,
            dict(
                rows=[
                    dict(r, distances={n: -99 for n in r["distances"]}) for r in distances["rows"]
                ]
            ),
        ),
    ):
        mutated_path = private / ("mutated_" + path.name)
        prior.prior.seal(mutated_path, after, replay)
        changed_request = dict(
            service=service,
            conditions={"baseline": rows},
            denial_paths=[
                str(mutated_path),
                str(distance_path if name == "test_labels" else label_path),
            ],
        )
        changed = child(changed_request, raw / name)
        unchanged = changed["predictions"]["baseline"] == baseline
        mutation_rows.append(
            dict(
                condition=name,
                metric="prediction_unchanged",
                value=int(unchanged),
                numerator=int(unchanged),
                denominator=1,
                arm="all_frozen_arms",
                cluster_id="128_graph_panel",
                status="completed",
                censor_reason=None,
                passed=unchanged and all(changed["denial_probes"].values()),
                original_hash=canonical_hash(before),
                mutated_hash=canonical_hash(after),
                original_reference=ref(path),
                mutated_reference=ref(mutated_path),
            )
        )
    controls = dict(
        file_and_traversal_denied=len(scored["denial_probes"]) == 4
        and all(scored["denial_probes"].values()),
        labeling_helper_absent=scored["labeling_helper_absent"],
        weights_unchanged=canonical_hash(service) == frozen_hash,
        baseline_reproduced=baseline_drift <= 1e-6,
        truth_mutations_unchanged=all(r["passed"] for r in mutation_rows),
    )
    result = dict(
        intended_count=128,
        completed_count=128,
        failed_count=0,
        censored_count=0,
        excluded_count=0,
        independent_count=128,
        rows=ablation_rows,
        ablation_rows=ablation_rows,
        permutation_rows=permutation_rows,
        oracle_mutation_rows=mutation_rows,
        maximum_probability_drift=max_drift,
        paired_intervals=intervals,
        custody_controls=controls,
        baseline_reproduction_drift=baseline_drift,
        detector_benefit_score=producer["detector_benefit_score"],
        original_statistical_benefit=producer["paired_intervals"],
        denial_probes=scored["denial_probes"],
        frozen_panel_reference=ref(raw / "panel.json"),
        fit_confidence_means=service["fit_means"],
        model_state_hashes=producer["model_state_hashes"],
        oracle_distance_origin="Private evaluation-only all-pairs distances from authenticated raw graphs; no retained distance file exists.",
        allowed_raw_graph_inputs=sorted(prior.g.KEYS),
        forbidden_derived_truth=["label", "distances", "expected", "answer_set", "parser_category"],
    )
    prior.prior.seal(raw / "measurements.json", result, replay)
    return result


def build(work: Json, validation: Json, fixture: bool) -> Json:
    """Integrity qualifies a score for later use without changing its scientific benefit."""
    result = work["result"]
    failures = deepcopy(work["evidence"]["failures"])
    owned = prior.admission.shard.qualified(validation, OWNED)
    integrity = (
        bool(result["custody_controls"])
        and all(result["custody_controls"].values())
        and result["maximum_probability_drift"] <= 1e-6
    )
    ready = int(owned and integrity and not failures and not fixture)
    cls = (
        "disqualified"
        if not fixture and (not owned or work["owned_failure"] or not failures and not integrity)
        else "blocked"
        if failures
        else "null"
    )
    if cls == "disqualified":
        failures.append(
            dict(
                upstream=TASK,
                path=work["raw"],
                hash=None,
                check="owned_controls",
                artifact_field="invariance_custody_and_validation",
                expected=True,
                observed=dict(
                    integrity=integrity, owned=owned, owned_failure=work["owned_failure"]
                ),
            )
        )
    verdict = f"complete_{cls}_relational_integrity_controls"
    previous = work.get("prior_terminal", {})
    if (
        cls in ("blocked", "disqualified")
        and previous.get("verdict_class") == cls
        and previous.get("gate_check_summary") == failures
    ):
        verdict = previous["honest_verdict"]
    value: Json = dict(
        result,
        experiment_id=8965,
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
        preconditions_checked=dict(
            historical_tasks_are_retained_inputs=True, failures=work["evidence"]["failures"]
        ),
        verifier_is_oracle=not integrity,
        exposure_scope="frozen_graph_structural_controls",
        independent_generalization_score=0,
        relational_controls_ready_score=ready,
        offending_service_disqualified=cls == "disqualified",
        required_checks_passed=owned,
        gate_check_summary=failures,
        source_artifact_hashes=work["evidence"]["references"],
        code_config_hashes=work["sources"],
        raw_shard_hashes=[ref(Path(work["raw"]) / "work.json"), *work["raw_references"]],
        owned_statement_coverage=validation["coverage"],
        validation_receipts=validation["receipts"],
        global_test_debt=work["global_test_debt"],
        fixture_only=fixture,
        flagged_adversarial=False,
        methodology_note="Four fixed bijections and edge orderings per graph;128 independent clusters. Frozen weights and temperatures, no retraining. Paired Brier intervals are distribution-shift mechanism diagnostics; failure to worsen is not a leakage or value verdict. Exp8964 benefit is unchanged. Any future model repair requires a new study.",
    )
    value["acceptance_gates"] = {
        k: dict(
            passed=v,
            principle="Integrity and owned checks qualify future use; ablation sensitivity is descriptive.",
        )
        for k, v in dict(
            authenticated_inputs=not work["evidence"]["failures"],
            invariance=integrity,
            custody=bool(result["custody_controls"]) and all(result["custody_controls"].values()),
            owned_checks=owned,
        ).items()
    }
    value["field_principles"] = {
        k: "Bind retained input bytes, typed graph rows, actual execution and scoped terminal claims."
        for k in [*value, "field_principles", "reproducibility_checksum"]
    }
    for fields, principle in (
        (
            "experiment_id task_id milestone run_date random_seed reproducibility_checksum",
            "Bind the result to the scheduled task, input bytes and reproducible randomness.",
        ),
        (
            "honest_verdict verdict_class",
            "Terminal dispositions prevent unchanged retries; failed integrity forbids positive.",
        ),
        (
            "inference_substrate inference_substrate_class execution_venue MODEL_SPECS model_specs model_invocation_counts numerical_training_calls",
            "Use actual host execution; retained generators, frozen small heads and new model calls are separate.",
        ),
        (
            "duration_s phase_spans preconditions_checked",
            "Actual clocks and resource receipts prevent fabricated execution and duration padding.",
        ),
        (
            "verifier_is_oracle exposure_scope independent_generalization_score",
            "Evaluation uses exact truth; graph controls cannot prove the no-cheap-oracle north star.",
        ),
        (
            "intended_count completed_count failed_count censored_count excluded_count independent_count rows permutation_rows ablation_rows oracle_mutation_rows maximum_probability_drift",
            "Keep all scheduled graphs and typed measurements; perturbations never enlarge independent n=128.",
        ),
        (
            "gate_check_summary",
            "Name upstream, hash, exact failed field, expected and observed values for every block.",
        ),
        (
            "source_artifact_hashes code_config_hashes raw_shard_hashes field_principles",
            "Retain content custody and explain why each evidence field exists.",
        ),
        (
            "acceptance_gates required_checks_passed owned_statement_coverage validation_receipts global_test_debt relational_controls_ready_score",
            "Invariance, custody and zero-exclusion100-percent owned checks are mandatory; global debt cannot waive them.",
        ),
        (
            "detector_benefit_score original_statistical_benefit paired_intervals",
            "Preserve Exp8964 benefit unchanged; distribution-shift ablations are mechanism diagnostics.",
        ),
    ):
        value["field_principles"].update({k: principle for k in fields.split()})
    value["reproducibility_checksum"] = calibration.checksum(value)
    return value
