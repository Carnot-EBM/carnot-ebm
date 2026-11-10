"""REQ-REPORT-8973: freeze common features without converting development into confirmation."""

from copy import deepcopy
import json
from pathlib import Path
import random
import time
from typing import Any

from carnot.reporting import encoder_head_diagnosis_8972 as prior
from carnot.reporting.current_work_receipt import (
    ZERO_INVOCATION_COUNTS,
    atomic_json,
    canonical_hash,
)

Json = dict[str, Any]
ROOT = prior.ROOT
NAME, TASK, SEED = (
    "experiment_8973_v767_shared_feature_methods",
    "exp8973-shared-feature-methods",
    767073,
)
EXPERIMENT_ID = 8973
CLI = f"scripts/experiments/{NAME}.py"
OWNED = [
    "python/carnot/reporting/shared_feature_methods_8973.py",
    "python/carnot/reporting/shared_feature_runner_8973.py",
    CLI,
]
TESTS = ["tests/python/test_shared_feature_methods_8973.py"]
INPUT = "results/experiment_8972_v767_encoder_head_diagnosis.json"
PIN = "sha256:aca5e51be78c64c911d49e6ce3cf3e036393ff59f1a300af29486f91733c56f4"
ARMS = [
    "shared_energy",
    "shared_logistic",
    "relational_logistic",
    "confidence_logistic",
    "prevalence",
]
reference, checksum, retain, seal, resources = (
    prior.reference,
    prior.checksum,
    prior.retain,
    prior.seal,
    prior.resources,
)


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Flush phase counts so the parent can distinguish compute from a stall."""
    print(f"[exp8973] phase={phase} completed={completed} pending={pending}", flush=True)


def protocol() -> Json:
    """Freeze feasibility and future comparisons before fresh responses exist."""
    return dict(
        architecture=dict(
            rounds=3,
            width=8,
            activation="tanh",
            aggregation="neighbor_mean",
            pooling="node_mean",
            gibbs_hidden=8,
        ),
        seeds=[71, 72, 73],
        steps=200,
        learning_rate=0.01,
        loss="population_binary_log_loss",
        seed_reduction="mean probabilities within arm; seeds are not examples",
        fit_count=256,
        guard_count=64,
        temperature_grid=[0.5, 1.0, 2.0, 4.0],
        temperature_selection="fresh calibration only; ensemble Brier minimum; ascending ties",
        family_alpha=0.05,
        bound_alpha=0.01,
        bootstrap_draws=10000,
        bootstrap_unit="paired graph; resample all arms together; rerank exact half within draw",
        multiplicity="Bonferroni five one-sided percentile bounds",
        detector_bounds=[
            dict(metric="brier_improvement", control=a, criterion="lower>0")
            for a in ("prevalence", "confidence_logistic", "shared_logistic")
        ]
        + [
            dict(metric="accepted_error_delta_50", control=a, criterion="upper<=0")
            for a in ("confidence_logistic", "shared_logistic")
        ],
        decisions=dict(accept="p_error<=0.1", reject="p_error>=0.9", escalate="otherwise"),
        fresh_count=1024,
        per_shard=dict(calibration=64, test=64, online=384),
        streams_per_shard=12,
        queries_per_stream=32,
        proposals_per_graph=2000,
        construction_seconds=600,
        response_selection=False,
        source_population="V765 n uniform50..60; independent undirected edges p=3/(n-1); uniform source; nonempty equal k2/k3 cardinality",
        training_equations=dict(
            energy="E(correct)=0; E(error)=-z; z=Gibbs scalar; p_error=sigmoid(z/T)",
            logistic="p_error=sigmoid(w dot shared27+b)",
            objective="population log loss;200 full-batch steps;learning rate0.01;original initialization",
        ),
        arms=ARMS,
        bootstrap_seed=SEED,
        binary_energies=dict(correct="0", error="-z", p_error="sigmoid(z)"),
        development_admission=dict(
            auroc_min=0.65,
            prevalence_brier_gain_min=0.005,
            shared_logistic_brier_slack=0.01,
            temperature=1.0,
        ),
        encoder_selection="Exp8972 protocol-fixed logistic encoders; all seeds71/72/73",
        feature_count=27,
        encoder_updates=0,
        scaling_updates=0,
        classifier_scope="Discriminatively trained nonlinear classifier; energy naming gives no special probabilistic guarantee.",
    )


def authenticate(root: Path, raw: Path) -> Json:
    """Bind historical bytes and typed gates directly; never invoke retired scheduler tasks."""
    result: Json = dict(failures=[], references=[], producer={}, work={})
    path = root / INPUT
    observed = reference(path)["sha256"]
    if observed != PIN:
        result["failures"].append(
            prior.failure(
                path,
                "exp8972-encoder-head-diagnosis",
                "sha256",
                PIN,
                "missing_file" if observed is None else "changed_bytes",
                "producer_bytes",
            )
        )
        return result
    try:
        value = json.loads(path.read_text())
        for field, expected in (("diagnosis_ready_score", 1), ("required_checks_passed", True)):
            actual = value.get(field)
            if type(actual) is not type(expected) or actual != expected:
                result["failures"].append(
                    prior.failure(
                        path,
                        "exp8972-encoder-head-diagnosis",
                        field,
                        expected,
                        actual,
                        "missing_field" if field not in value else "observed_value",
                    )
                )
                return result
        result["references"].append(
            dict(retain(reference(path), raw), experiment_id=8972, fields_imported=sorted(value))
        )
        for ref in (
            value["source_artifact_hashes"]
            + value["code_config_hashes"]
            + value["raw_shard_hashes"]
        ):
            actual_ref = reference(Path(ref["path"]))
            if actual_ref["sha256"] != ref["sha256"]:
                result["failures"].append(
                    prior.failure(
                        Path(ref["path"]),
                        "exp8972-encoder-head-diagnosis",
                        "sha256",
                        ref["sha256"],
                        "missing_file" if actual_ref["sha256"] is None else "changed_bytes",
                        "retained_bytes",
                    )
                )
                return result
            result["references"].append(retain(ref, raw))
        saved = retain(value["work_reference"], raw)
        result.update(producer=value, work=json.loads(Path(saved["path"]).read_text()))
        history_ref = next(
            r
            for r in result["work"]["evidence"]["producers"]["8959"]["raw_shard_hashes"]
            if Path(r["path"]).name == "work.json"
        )
        result["references"].append(retain(history_ref, raw))
    except (OSError, ValueError, KeyError, TypeError, AttributeError, StopIteration) as error:
        result["failures"].append(
            prior.failure(
                path,
                "exp8972-encoder-head-diagnosis",
                "retained_primitives",
                "authenticated complete original bytes",
                str(error),
                "producer_structure",
            )
        )
    return result


def admission(metrics: Json) -> list[Json]:
    """These operational margins close collection; they are not significance tests."""
    return [
        dict(
            metric=name,
            value=float(value),
            operator=operator,
            threshold=threshold,
            passed=bool(value >= threshold if operator == ">=" else value <= threshold),
            temperature=1.0,
            independent_count=64,
            scope="exposed_development_only",
        )
        for name, value, operator, threshold in (
            ("shared_energy_auroc", metrics["auroc"], ">=", 0.65),
            (
                "brier_improvement_over_fit_prevalence",
                metrics["prevalence_brier"] - metrics["energy_brier"],
                ">=",
                0.005,
            ),
            (
                "brier_excess_over_shared_logistic",
                metrics["energy_brier"] - metrics["logistic_brier"],
                "<=",
                0.01,
            ),
        )
    ]


def history(evidence: Json) -> Json:
    """Include every retained construction and rejected proposal, not only collected responses."""
    from carnot.reporting import relational_methods_8959 as old

    graphs: Json = {}
    seeds: set[int] = set()

    def collect(value: Any) -> None:
        if isinstance(value, dict):
            if {"nodes", "edges", "start"} <= value.keys():
                graphs[old.source.graph_hash(value)] = {
                    k: value[k] for k in ("nodes", "edges", "start")
                }
            if "graph_hash" in value and type(value.get("seed")) is int:
                seeds.add(value["seed"])
            for item in value.values():
                collect(item)
        elif isinstance(value, list):
            for item in value:
                collect(item)

    collect(evidence["work"]["evidence"]["producers"])
    historical_ref = next(
        r
        for r in evidence["work"]["evidence"]["producers"]["8959"]["raw_shard_hashes"]
        if Path(r["path"]).name == "work.json"
    )
    historical = load(evidence, historical_ref)["evidence"]
    collect(historical["old_graphs"])
    seeds.update(historical["old_seeds"])
    for ref in evidence["references"]:
        if Path(ref["path"]).suffix == ".json":
            collect(json.loads(Path(ref["path"]).read_text()))
    return dict(old_graphs=list(graphs.values()), old_seeds=sorted(seeds))


def construct(evidence: Json, raw: Path, replay: bool = False) -> Json:
    """Reuse bounded matched-cardinality construction in a separate V767 seed namespace."""
    from unittest.mock import patch
    from carnot.reporting import relational_methods_8959 as old

    historical = history(evidence)
    with patch.object(old, "SEED", SEED):
        value = old.graph_protocol(historical)
    for graph in value["graph_manifest"]:
        graph["graph_id"] = graph["graph_id"].replace("v766", "v767")
        graph["questions"][0]["unit_id"] = graph["questions"][0]["unit_id"].replace("v766", "v767")
    path = raw / "graphs.json"
    if replay:
        original = json.loads(path.read_text())
        value["construction_duration_s"] = original["construction_duration_s"]
    seal(path, value, replay)
    return dict(value)


def load(evidence: Json, ref: Json) -> Json:
    """Read our retained copy so replay does not inherit mutable historical authority."""
    saved = next(r for r in evidence["references"] if r.get("source_path") == ref["path"])
    if reference(Path(saved["path"]))["sha256"] != ref["sha256"]:
        raise ValueError("retained_operand_drift")
    return dict(json.loads(Path(saved["path"]).read_text()))


def evaluate(evidence: Json, raw: Path, replay: bool = False) -> Json:
    """Recompute guard scores, invariant controls and a future population from saved operands."""
    import jax.numpy as jnp
    import numpy as np
    from sklearn.metrics import roc_auc_score
    from carnot.autoresearch import encoder_head_8972 as n
    from carnot.autoresearch import relational_graph_encoder as g

    progress("before_development_benchmark", 0, 64)
    original = evidence["work"]["evidence"]
    views, old_states = original["views"], original["states"]
    fit, guard = views["fit"], views["guard"]
    producer = original["producers"]["8959"]
    if (
        len(fit["rows"]) != 256
        or len(guard["rows"]) != 64
        or fit["unit_ids"] != producer["development_fit_ids"]
        or guard["unit_ids"] != producer["development_guard_ids"]
    ):
        raise ValueError("development_roles")
    y = n.b.labels(guard["labels"], 64)
    candidate = load(evidence, evidence["producer"]["candidate_reference"])
    states: Json = {arm: [] for arm in ARMS[:-1]}
    controls: list[Json] = []
    encoders: list[Json] = []
    negatives: list[Json] = []
    permuted_labels = np.random.default_rng(SEED).permutation(fit["labels"]).tolist()
    for seed in (71, 72, 73):
        progress(f"before_seed{seed}_controls", seed - 71, 3 - (seed - 71))
        fixed = next(s for s in candidate["encoder_states"] if s["seed"] == seed)
        encoder_hash = canonical_hash(fixed["encoder"])
        encoders.append(
            dict(seed=seed, sha256=encoder_hash, scaling_hash=canonical_hash(fixed["scaling"]))
        )
        for arm, name in (
            ("shared_energy", "logistic_encoder_gibbs_head"),
            ("shared_logistic", "logistic_encoder_linear_head"),
        ):
            ref = next(
                r
                for r in evidence["producer"]["head_state_references"]
                if r["seed"] == seed and r["arm"] == name
            )
            state = load(evidence, ref)
            states[arm].append(state)
            invariant = (
                state["params"]["encoder"] == fixed["encoder"]
                and state["scaling"] == fixed["scaling"]
            )
            x = n.features(state, guard["rows"])
            feature_ref = next(
                r
                for r in evidence["producer"]["feature_hashes"]
                if r["encoder"] == "logistic" and r["seed"] == seed and r["role"] == "guard"
            )
            saved_features = np.asarray(load(evidence, feature_ref)["features"])
            feature_delta = float(np.max(np.abs(x - saved_features)))
            shuffled = []
            for row in guard["rows"]:
                names = {name: f"anonymous-{i}" for i, name in enumerate(row["nodes"])}
                shuffled.append(
                    dict(
                        row,
                        nodes=[names[v] for v in row["nodes"]][::-1],
                        edges=[[names[v], names[u]] for u, v in row["edges"]][::-1],
                        query_node=names[row["query_node"]],
                        proposed_node=names[row["proposed_node"]]
                        if row["proposed_node"] is not None
                        else None,
                    )
                )
            delta = float(
                np.max(np.abs(g.predict(state, guard["rows"]) - g.predict(state, shuffled)))
            )
            params = dict(
                encoder={k: jnp.asarray(v) for k, v in state["params"]["encoder"].items()},
                head=tuple(jnp.asarray(v) for v in state["params"]["head"]),
            )
            gradient = n.gradient_row(
                params,
                n.operands(state, fit["rows"]),
                jnp.asarray(fit["labels"]),
                state["energy"],
                200,
            )
            finite = all(
                np.isfinite(gradient[k])
                for k in (
                    "encoder_gradient_norm",
                    "head_gradient_norm",
                    "fit_loss",
                    "prediction_variance",
                )
            )
            denied = False
            try:
                g.pack([dict(guard["rows"][0], label=1)])
            except ValueError:
                denied = True
            for metric, value, passed in (
                ("encoder_and_scaling_immutable", encoder_hash, invariant),
                (
                    "identical_27_features",
                    feature_delta,
                    x.shape == (64, 27) and feature_delta <= 1e-12,
                ),
                ("node_edge_permutation", delta, delta <= 1e-6),
                ("finite_gradients", gradient, finite),
                ("forbidden_label_denial", denied, denied),
            ):
                controls.append(
                    dict(
                        seed=seed,
                        arm=arm,
                        metric=metric,
                        value=value,
                        passed=bool(passed),
                        independent_count=0,
                    )
                )
        progress(
            f"before_label_permutation_training_seed{seed}", len(negatives), 3 - len(negatives)
        )
        negative, gradients = n.fit_head(
            states["shared_energy"][-1], fit["rows"], permuted_labels, True
        )
        p = g.predict(negative, guard["rows"])
        difference = float(
            np.max(np.abs(p - g.predict(states["shared_energy"][-1], guard["rows"])))
        )
        negative_row = dict(
            seed=seed,
            labels_hash=canonical_hash(permuted_labels),
            guard_auroc=float(roc_auc_score(y, p)),
            guard_brier=float(np.mean((p - y) ** 2)),
            maximum_probability_change=difference,
            gradient_rows=gradients,
            scope="exposed_label_permutation_negative_control_no_chance_guarantee",
        )
        controls.append(
            dict(
                seed=seed,
                arm="shared_energy",
                metric="label_permutation_negative_control",
                value=negative_row,
                passed=bool(
                    difference > 1e-6
                    and np.isfinite(p).all()
                    and sorted(permuted_labels) == sorted(fit["labels"])
                ),
                independent_count=0,
            )
        )
        negatives.append(negative)
        progress(f"after_label_permutation_training_seed{seed}", len(negatives), 3 - len(negatives))
        for arm in ("relational_logistic", "confidence_logistic"):
            states[arm].append(old_states[arm][seed - 71])
    prevalence = float(np.mean(fit["labels"]))
    probabilities = {
        arm: g.ensemble(group, guard["rows"], 1.0).tolist() for arm, group in states.items()
    }
    probabilities["prevalence"] = [prevalence] * 64
    losses = {arm: float(np.mean((np.asarray(p) - y) ** 2)) for arm, p in probabilities.items()}
    metrics = dict(
        auroc=float(roc_auc_score(y, probabilities["shared_energy"])),
        energy_brier=losses["shared_energy"],
        logistic_brier=losses["shared_logistic"],
        prevalence_brier=losses["prevalence"],
        arm_brier=losses,
    )
    rows = [
        dict(
            unit_id=unit,
            cluster_id=unit,
            arm=arm,
            metric="brier",
            value=float((p[i] - y[i]) ** 2),
            numerator=float((p[i] - y[i]) ** 2),
            denominator=1,
            p_error=p[i],
            error=int(y[i]),
            condition="exposed_development_guard",
            status="completed",
            censor_reason=None,
        )
        for arm, p in probabilities.items()
        for i, unit in enumerate(guard["unit_ids"])
    ]
    seal(
        raw / "states.json",
        dict(states=states, prevalence=prevalence, negative_control_states=negatives),
        replay,
    )
    seal(raw / "guard_predictions.json", dict(rows=rows), replay)
    progress("after_development_benchmark", 64, 0)
    progress("before_graph_construction", 0, 1024)
    graphs = construct(evidence, raw, replay)
    progress(
        "after_graph_construction",
        len(graphs["graph_manifest"]),
        1024 - len(graphs["graph_manifest"]),
    )
    result = dict(
        development_guard_rows=rows,
        development_metrics=metrics,
        development_admission_rows=admission(metrics),
        invariant_control_rows=controls,
        encoder_hashes=encoders,
        trained_state_reference=reference(raw / "states.json"),
        numerical_training_calls=3,
        numerical_full_batch_updates=600,
        frozen_head_import_count=6,
        graph_manifest=graphs["graph_manifest"],
        construction=graphs,
        role_manifest=[
            dict(
                unit_id=r["questions"][0]["unit_id"],
                graph_hash=r["graph_hash"],
                **{k: r[k] for k in ("shard", "role", "stream_id", "stream_position")},
            )
            for r in graphs["graph_manifest"]
        ],
    )
    seal(raw / "reduction.json", result, replay)
    return result


def build(work: Json, validation: Json, fixture: bool) -> Json:
    """Separate reproducible mechanics, operational admission and unmeasured scientific benefit."""
    result, evidence = work["result"], work["evidence"]
    owned = False
    if validation["receipts"]:
        from carnot.reporting.matched_corpus_a_8949 import qualified

        owned = qualified(validation, OWNED)
    failures = evidence["failures"]
    mechanics = bool(result.get("invariant_control_rows")) and all(
        r["passed"] for r in result.get("invariant_control_rows", [])
    )
    gate = bool(result.get("development_admission_rows")) and all(
        r["passed"] for r in result.get("development_admission_rows", [])
    )
    complete = result.get("construction", {}).get("complete", False)
    failed_owned = (
        not fixture and not owned and (bool(validation["receipts"]) or not failures)
    ) or (not failures and not mechanics)
    cls = "disqualified" if failed_owned else "blocked" if failures else "null"
    ready = int(owned and not fixture and not failures and mechanics and gate and complete)
    count = len(result.get("graph_manifest", []))
    rows = list(result.get("development_guard_rows", []))
    for index in range(1024):
        for arm in ARMS:
            rows.append(
                dict(
                    unit_id=f"v767-graph-{index:04d}-k2",
                    cluster_id=f"v767-graph-{index:04d}",
                    arm=arm,
                    condition="frozen_future_graph",
                    metric="graph_role_frozen",
                    value=1 if index < count else None,
                    numerator=1 if index < count else None,
                    denominator=1,
                    status="completed" if index < count else "censored",
                    censor_reason=None
                    if index < count
                    else "external_precondition"
                    if failures
                    else result["construction"]["terminal_reason"],
                )
            )
    if failures:
        rows += [
            dict(
                unit_id=f"unavailable-guard-{i:02d}",
                cluster_id=f"unavailable-guard-{i:02d}",
                arm=arm,
                condition="exposed_development_guard",
                metric="brier",
                value=None,
                numerator=None,
                denominator=1,
                status="censored",
                censor_reason="external_precondition",
            )
            for i in range(64)
            for arm in ARMS
        ]
    completed = count + (64 if result else 0)
    reason = (
        "retained_input"
        if failures
        else "owned_checks"
        if failed_owned
        else "development_admission_failed"
        if not gate
        else "frozen"
    )
    value: Json = dict(
        experiment_id=8973,
        task_id=TASK,
        milestone="2026.11.767",
        run_date="20261110",
        random_seed=SEED,
        honest_verdict=f"complete_{cls}_{reason}",
        verdict_class=cls,
        status="error" if cls == "disqualified" else "blocked" if failures else "success",
        inference_substrate="verifier_ensemble_against_cached_candidates",
        inference_substrate_class="no_model_load",
        MODEL_SPECS=[],
        model_specs=[],
        model_invoked=False,
        model_invocation_counts=dict(ZERO_INVOCATION_COUNTS),
        execution_venue="host",
        numerical_training_calls=result.get("numerical_training_calls", 0),
        numerical_full_batch_updates=result.get("numerical_full_batch_updates", 0),
        rows=rows,
        intended_count=1088,
        completed_count=completed,
        failed_count=0,
        censored_count=1088 - completed,
        excluded_count=0,
        independent_count=completed,
        verifier_is_oracle=False,
        exposure_scope="64 exposed guard graphs;1024 frozen constructions without responses; cheap BFS truth cannot establish oracle-distinct benefit",
        independent_generalization_score=0,
        shared_methods_ready_score=ready,
        fresh_collection_permitted=bool(ready),
        development_admission_rows=result.get("development_admission_rows", []),
        invariant_control_rows=result.get("invariant_control_rows", []),
        frozen_protocol=protocol(),
        trained_state_reference=result.get("trained_state_reference"),
        encoder_hashes=result.get("encoder_hashes", []),
        graph_manifest=result.get("graph_manifest", []),
        role_manifest=result.get("role_manifest", []),
        feature_schema=dict(
            feature_count=27,
            readout=[
                "source_embedding8",
                "answer_embedding8",
                "mean_embedding8",
                "mean_token_nll",
                "first_answer_top2_margin",
                "missing_answer",
            ],
            allowed_keys=["nodes", "edges", "query_node", "proposed_node", "features"],
            forbidden=[
                "label",
                "shortest_distance",
                "BFS",
                "answer_set_membership",
                "graph_id",
                "split_id",
            ],
        ),
        source_manifest=dict(
            protocol="V765 matched-cardinality k2",
            response_based_selection=False,
            historical_graphs=result.get("construction", {}).get("history_graph_count", 0),
            historical_seeds=result.get("construction", {}).get(
                "history_construction_seed_count", 0
            ),
            future_dependencies=[
                dict(
                    task_id="exp8974-feature-online-methods",
                    status="future_dependency_not_required_input",
                )
            ],
        ),
        gate_check_summary=failures,
        preconditions_checked=work["preconditions"],
        duration_s=work["duration_s"],
        phase_spans=work["phase_spans"],
        source_artifact_hashes=evidence["references"],
        code_config_hashes=work["sources"],
        raw_shard_hashes=work["raw_references"],
        cited_upstream_artifacts=[
            dict(
                experiment_id=8972,
                task_id="exp8972-encoder-head-diagnosis",
                sha256=PIN,
                fields_imported=sorted(evidence["producer"]),
                provenance_scope="historical_only",
            )
        ],
        required_checks_passed=owned,
        owned_statement_coverage=validation["coverage"],
        validation_receipts=validation["receipts"],
        global_test_debt=work["global_test_debt"],
        flagged_adversarial=False,
        fixture_only=fixture,
        acceptance_gates={
            name: dict(passed=passed, principle=principle)
            for name, passed, principle in (
                (
                    "authenticated_inputs",
                    not failures,
                    "Authenticate original byte pins before compute.",
                ),
                (
                    "mechanics",
                    mechanics,
                    "Feature and custody controls qualify independently of signal.",
                ),
                (
                    "development_admission",
                    gate,
                    "Fixed operational thresholds close collection after any failure.",
                ),
                (
                    "complete_future_population",
                    complete,
                    "Construction exhaustion cannot relax the source or roles.",
                ),
                (
                    "owned_checks",
                    owned,
                    "Owned failures disqualify; unrelated global debt stays separate.",
                ),
                ("scientific_benefit", False, "No fresh responses or confirmation are measured."),
            )
        },
        methodology_note="Shared-feature Gibbs is a discriminatively trained nonlinear classifier. The energy name supplies no special probabilistic guarantee. Seeds average probabilities; seeds are not independent questions. Fixed guard gates are operational feasibility thresholds, not significance or confirmation.",
    )
    value = result | value
    value["field_principles"] = {
        key: "Bind task identity, authentic operands, independent graph units and the measured terminal disposition."
        for key in [*value, "field_principles", "reproducibility_checksum"]
    }
    value["reproducibility_checksum"] = checksum(value)
    return value
