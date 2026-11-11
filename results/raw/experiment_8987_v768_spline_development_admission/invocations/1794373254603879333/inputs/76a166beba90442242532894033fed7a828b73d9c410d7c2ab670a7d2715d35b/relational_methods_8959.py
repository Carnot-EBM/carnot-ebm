"""REQ-REPORT-8959: freeze methods without claiming a real-data detector gain."""

from __future__ import annotations

import json
from pathlib import Path
import random
import time
from typing import Any

import networkx as nx
import numpy as np

from carnot.autoresearch import relational_graph_encoder as g
from carnot.reporting import confidence_diagnosis_8958 as diagnosis
from carnot.reporting import matched_source_methods_8946 as matched
from carnot.reporting.current_work_receipt import ZERO_INVOCATION_COUNTS, canonical_hash

Json = dict[str, Any]
ROOT, b, calibration, source = diagnosis.ROOT, diagnosis.b, diagnosis.calibration, matched.source
NAME, TASK, SEED = "experiment_8959_v766_relational_methods", "exp8959-relational-methods", 766059
EXPERIMENT_ID = 8959
CLI = f"scripts/experiments/{NAME}.py"
OWNED = [
    "python/carnot/autoresearch/relational_graph_encoder.py",
    "python/carnot/reporting/relational_methods_8959.py",
    "python/carnot/reporting/relational_methods_runner_8959.py",
    CLI,
]
TESTS = ["tests/python/test_relational_methods_8959.py"]
INPUTS = [
    (
        f"results/{matched.NAME}.json",
        "sha256:7cd1769932adf184a01e75351d83a997b2f7d15797095b581ae54b6886777b19",
        "matched_source_ready_score",
    ),
    (
        f"results/{diagnosis.NAME}.json",
        "sha256:ff279de4b7b8f2a8eae95d978288da6f588bf6c0d5635e83bf062a72b605f6b0",
        "diagnosis_ready_score",
    ),
]
TRAINED_ARMS = (
    ("relational_energy", True, True),
    ("relational_logistic", True, False),
    ("confidence_energy", False, True),
    ("confidence_logistic", False, False),
)
ARMS = [a for a, _, _ in TRAINED_ARMS] + ["prevalence"]


def protocol() -> Json:
    """All adaptive choices are recorded before any fresh response or error label."""
    return dict(
        architecture=dict(
            rounds=3,
            width=8,
            activation="tanh",
            shared_round_weights=True,
            aggregation="neighbor_mean",
            pooling="node_mean",
            gibbs_hidden=8,
        ),
        seeds=[71, 72, 73],
        steps=200,
        learning_rate=0.01,
        loss="population_binary_log_loss",
        seed_reduction="mean probabilities within arm; seeds are not examples",
        arms=ARMS,
        fit_count=256,
        guard_count=64,
        temperature_grid=list(b.GRID),
        temperature_selection="one per arm; minimize ensemble Brier on fresh calibration only; ascending ties",
        family_alpha=0.05,
        bound_alpha=0.01,
        bootstrap_draws=10000,
        bootstrap_seed=SEED,
        bootstrap_unit="paired graph; resample all arms together; rerank exact half within draw",
        multiplicity="Bonferroni five one-sided percentile bounds; simultaneous family alpha0.05",
        detector_bounds=[
            dict(metric="brier_improvement", control=a, criterion="lower>0")
            for a in ("prevalence", "confidence_logistic", "relational_logistic")
        ]
        + [
            dict(metric="accepted_error_delta_50", control=a, criterion="upper<=0")
            for a in ("confidence_logistic", "relational_logistic")
        ],
        decisions=dict(accept="p_error<=0.1", reject="p_error>=0.9", escalate="otherwise"),
        coverage_scope="ranking diagnostic; no risk certificate",
        fresh_count=1024,
        per_shard=dict(calibration=64, test=64, online=384),
        streams_per_shard=12,
        queries_per_stream=32,
        proposals_per_graph=2000,
        construction_seconds=600,
        response_selection=False,
        source_population="V765 n uniform50..60; independent undirected edges p=3/(n-1); uniform source; nonempty equal k2/k3 cardinality",
        generator_settings=dict(matched.source.SETTINGS),
        scope="anonymous graph adaptation; not KG-Guard reproduction or energy superiority",
        training_equations=dict(
            node_input="[source_flag,answer_flag,1,degree/(n-1)]",
            initial="h0=tanh(X W_in+b_in)",
            round="h(t+1)=tanh(h(t) W_self+mean_neighbors(h(t)) W_neighbor+b); t=0,1,2",
            readout="concat(h_source,h_answer,mean_nodes(h),standardized_confidence[2],missing_answer)",
            energy="GibbsModel hidden8 SiLU; p_error=sigmoid(E/T)",
            logistic="p_error=sigmoid(w dot readout+b)",
            objective="mean(softplus(logit)-error*logit); theta <- theta-0.01*grad;200 full batches",
        ),
    )


def authenticate(root: Path) -> Json:
    """Historical tasks are byte-bound inputs; retired scheduler gates are never run."""
    result: Json = dict(
        failures=[], references=[], developer=[], labels=[], old_graphs=[], old_seeds=[]
    )
    graphs, seeds = {}, set()

    def collect(value: Any) -> None:
        if isinstance(value, dict):
            if {"nodes", "edges", "start"} <= set(value):
                graphs[source.graph_hash(value)] = {
                    k: value[k] for k in ("nodes", "edges", "start")
                }
            if "graph_hash" in value and type(value.get("seed")) is int:
                seeds.add(value["seed"])
            for item in value.values():
                collect(item)
        elif isinstance(value, list):
            for item in value:
                collect(item)

    for name, pin, field in INPUTS:
        path = root / name
        ref = calibration.reference(path)
        failure_ref = ref
        result["references"].append(ref)
        expected: Any = pin
        observed, check = ref["sha256"], "sha256"
        try:
            if observed != expected:
                raise ValueError("missing_file" if observed is None else "producer_hash")
            value = json.loads(path.read_text())
            for key in ("required_checks_passed", field):
                expected, observed, check = 1, value.get(key), key
                if observed != expected:
                    raise ValueError("missing_field" if observed is None else "observed_zero")
            refs = (
                value["source_artifact_hashes"]
                + value["code_config_hashes"]
                + value["raw_shard_hashes"]
            )
            if field == "diagnosis_ready_score":
                refs += [value["developer_view"], value["protected_label_view"]]
                result["developer"] = json.loads(Path(value["developer_view"]["path"]).read_text())[
                    "rows"
                ]
                result["labels"] = json.loads(
                    Path(value["protected_label_view"]["path"]).read_text()
                )["rows"]
                result["development_manifest"] = value["development_manifest"]
            collect(value)
            for operand in refs:
                target = Path(operand["path"])
                failure_ref = calibration.reference(target)
                expected, observed, check = (
                    operand["sha256"],
                    calibration.reference(target)["sha256"],
                    "retained_bytes",
                )
                if observed != expected:
                    raise ValueError("retained_bytes_drift")
                result["references"].append(operand)
                if target.suffix == ".json":
                    collect(json.loads(target.read_text()))
        except (OSError, ValueError, KeyError, TypeError) as error:
            result["failures"].append(
                dict(
                    upstream=path.stem,
                    **failure_ref,
                    hash=failure_ref["sha256"],
                    check=str(error),
                    artifact_field=check,
                    expected=expected,
                    observed=observed,
                )
            )
    result.update(old_graphs=list(graphs.values()), old_seeds=sorted(seeds))
    return result


def detector_view(rows: list[Json]) -> list[Json]:
    """Metadata remains in custody manifests and never reaches differentiable scoring."""
    return [{k: r[k] for k in g.KEYS} for r in rows]


def train(evidence: Json) -> Json:
    """Freeze all twelve trained states using only the historical development-fit role."""
    fit = [r for r in evidence["developer"] if r["development_role"] == "fit"]
    guard = [r for r in evidence["developer"] if r["development_role"] == "guard"]
    if len(fit) != 256 or len(guard) != 64 or len({r["graph_hash"] for r in fit + guard}) != 320:
        raise ValueError("development_roles")
    labels = {r["unit_id"]: r["error"] for r in evidence["labels"]}
    x, y = detector_view(fit), [labels[r["unit_id"]] for r in fit]
    states = {
        arm: [g.fit(x, y, seed, relational, energy) for seed in (71, 72, 73)]
        for arm, relational, energy in TRAINED_ARMS
    }
    probabilities = {arm: g.ensemble(group, detector_view(guard)) for arm, group in states.items()}
    probabilities["prevalence"] = [float(np.mean(y))] * len(guard)
    guard_rows = []
    for arm, values in probabilities.items():
        selected = b.half_selection(values, [r["unit_id"] for r in guard])
        for row, probability, keep in zip(guard, values, selected, strict=True):
            error = labels[row["unit_id"]]
            loss = float((probability - error) ** 2)
            guard_rows.append(
                dict(
                    unit_id=row["unit_id"],
                    cluster_id=row["graph_hash"],
                    arm=arm,
                    condition="exposed_development_guard",
                    metric="brier",
                    value=loss,
                    numerator=loss,
                    denominator=1,
                    error=error,
                    p_error=float(probability),
                    selected_50=keep,
                    status="completed",
                    censor_reason=None,
                )
            )
    return dict(
        states=states,
        prevalence=float(np.mean(y)),
        fit_ids=[r["unit_id"] for r in fit],
        guard_ids=[r["unit_id"] for r in guard],
        training_labels_hash=canonical_hash(y),
        training_count=256,
        independent_count=320,
        numerical_fit_count=12,
        numerical_training_calls=2400,
        guard_rows=guard_rows,
    )


def fixture_controls() -> Json:
    """A controlled adjacency signal must train; this says nothing about Qwen errors."""
    rows: list[Json] = [
        dict(
            nodes=["AA", "BB", "CC", "DD"],
            edges=[["AA", "BB"], ["BB", "CC"], ["CC", "DD"]],
            query_node="AA",
            proposed_node="BB" if i % 2 else "DD",
            features=dict(mean_token_nll=1.0, first_answer_top2_margin=1.0),
        )
        for i in range(32)
    ]
    truth = [i % 2 for i in range(32)]
    permuted = np.random.default_rng(71).permutation(truth).tolist()
    controls = []
    names = dict(zip(rows[0]["nodes"], ["WW", "XX", "YY", "ZZ"]))
    renamed = [
        dict(
            r,
            nodes=[names[n] for n in r["nodes"]][::-1],
            edges=[[names[c], names[a]] for a, c in r["edges"]][::-1],
            query_node=names[r["query_node"]],
            proposed_node=names[r["proposed_node"]],
        )
        for r in rows
    ]
    for arm, _, energy in TRAINED_ARMS[:2]:
        state = g.fit(rows, truth, 71, True, energy)
        other = g.fit(rows, permuted, 71, True, energy)
        p, q = g.predict(state, rows), g.predict(other, rows)
        for metric, value, passed in [
            ("nonzero_gradient", state["gradient_norm"], state["gradient_norm"] > 0),
            (
                "nonzero_encoder_gradient",
                state["encoder_gradient_norm"],
                state["encoder_gradient_norm"] > 0,
            ),
            (
                "informative_brier_improvement",
                float(0.25 - np.mean((p - truth) ** 2)),
                float(np.mean((p - truth) ** 2)) < 0.25,
            ),
            (
                "label_permutation_sensitivity",
                float(np.max(np.abs(p - q))),
                float(np.max(np.abs(p - q))) > 1e-6,
            ),
            (
                "node_edge_invariance_max_delta",
                float(np.max(np.abs(p - g.predict(state, renamed)))),
                float(np.max(np.abs(p - g.predict(state, renamed)))) <= 1e-6,
            ),
        ]:
            controls.append(
                dict(
                    arm=arm,
                    metric=metric,
                    value=value,
                    passed=bool(passed),
                    condition="private_informative_fixture",
                    independent_count=0,
                )
            )
    refused = False
    try:
        g.pack([dict(rows[0], label=1)])
    except ValueError:
        refused = True
    controls.append(
        dict(
            arm="all",
            metric="forbidden_feature_refusal",
            value=int(refused),
            passed=refused,
            condition="private_informative_fixture",
            independent_count=0,
        )
    )
    return dict(
        rows=controls,
        passed=all(r["passed"] for r in controls),
        numerical_fit_count=4,
        numerical_training_calls=800,
    )


def nx_graph(graph: Json) -> Any:
    """Unmarked topology makes exclusion stricter than source-marked isomorphism."""
    result = nx.Graph()
    result.add_nodes_from(graph["nodes"])
    result.add_edges_from(graph["edges"])
    return result


def topology_key(graph: Any) -> Any:
    """Node and degree counts cheaply narrow exact isomorphism candidates."""
    return (
        graph.number_of_nodes(),
        graph.number_of_edges(),
        tuple(sorted(dict(graph.degree()).values())),
    )


def role(index: int) -> Json:
    """A frozen seed permutes roles without labels, candidates or generator responses."""
    shard = "A" if index < 512 else "B"
    order = list(range(512))
    random.Random(f"{SEED}/{shard}/roles").shuffle(order)
    position = order.index(index % 512)
    kind = "calibration" if position < 64 else "test" if position < 128 else "online"
    online = position - 128
    return dict(
        shard=shard,
        role=kind,
        stream_id=f"{shard}-stream-{online // 32:02d}" if kind == "online" else None,
        stream_position=online % 32 if kind == "online" else None,
    )


def fresh_graph(graph: Json, seed: int, index: int) -> Json:
    """Use the qualified prompt text and seeded node/edge presentation permutation."""
    nodes, edges = list(graph["nodes"]), list(graph["edges"])
    rng = random.Random(f"{SEED}/{seed}/prompt")
    rng.shuffle(nodes)
    rng.shuffle(edges)
    prompt = (
        "Undirected simple graph. Nodes: "
        + ", ".join(nodes)
        + ".\nEdges: "
        + ", ".join(a + "-" + c for a, c in edges)
        + ".\n"
        + f"Name a node exactly 2 hops from {graph['start']} (shortest distance). Reply with only its two-letter label."
    )
    graph_id = f"v766-graph-{index:04d}"
    return dict(
        graph,
        seed=seed,
        graph_id=graph_id,
        graph_hash=source.graph_hash(graph),
        **role(index),
        questions=[
            dict(unit_id=graph_id + "-k2", k=2, prompt=prompt, prompt_hash=canonical_hash(prompt))
        ],
    )


def graph_protocol(evidence: Json) -> Json:
    """Exhaustion changes feasibility, never population, roles or scheduled denominator."""
    started, heartbeat = time.monotonic(), time.monotonic()
    buckets: dict[Any, list[Any]] = {}
    seeds = set(evidence["old_seeds"])
    history = list(evidence["old_graphs"])
    seen = {source.graph_hash(graph) for graph in history}
    graphs, rejections, reason = [], [], None
    for i, seed in enumerate(sorted(seeds)):
        history.append(source.candidate(seed))
        if time.monotonic() - started >= 600:
            reason = "construction_time_exhausted"
            break
        if time.monotonic() - heartbeat >= 30:
            b.progress("8959_history_index", i + 1, len(seeds) - i - 1)
            heartbeat = time.monotonic()
    if reason is None:
        for graph in history:
            topology = nx_graph(graph)
            buckets.setdefault(topology_key(topology), []).append(topology)
        for index in range(1024):
            for attempt in range(2000):
                if time.monotonic() - started >= 600:
                    reason = "construction_time_exhausted"
                    break
                seed = 2**52 + SEED * 10000000 + index * 2000 + attempt
                graph = source.candidate(seed)
                digest, topology = source.graph_hash(graph), nx_graph(graph)
                key = topology_key(topology)
                answers = [
                    source.distance_nodes(graph["nodes"], graph["edges"], graph["start"], k)
                    for k in (2, 3)
                ]
                rejected = (
                    "historical_or_proposed_identity"
                    if seed in seeds or digest in seen
                    else "unmatched_cardinality"
                    if not answers[0] or len(answers[0]) != len(answers[1])
                    else "isomorphic_relabeling"
                    if any(nx.is_isomorphic(topology, other) for other in buckets.get(key, []))
                    else None
                )
                seeds.add(seed)
                seen.add(digest)
                buckets.setdefault(key, []).append(topology)
                if rejected is None:
                    graphs.append(fresh_graph(graph, seed, index))
                    break
                rejections.append(
                    dict(
                        seed=seed, graph_hash=digest, reason=rejected, index=index, attempt=attempt
                    )
                )
                if time.monotonic() - heartbeat >= 30:
                    b.progress("8959_graph_proposals", len(graphs), 1024 - len(graphs))
                    heartbeat = time.monotonic()
            else:
                reason = "proposal_budget_exhausted"
            if reason:
                break
            if (index + 1) % 64 == 0:
                b.progress("8959_fresh_graph_freeze", index + 1, 1023 - index)
    return dict(
        graph_manifest=graphs,
        proposal_rejection_rows=rejections,
        complete=len(graphs) == 1024,
        terminal_reason=reason,
        history_graph_count=len(evidence["old_graphs"]),
        history_construction_seed_count=len(evidence["old_seeds"]),
        construction_duration_s=time.monotonic() - started,
    )


def verify_graphs(result: Json, evidence: Json) -> bool:
    """Replay rebuilds every proposed construction; altered graphs cannot be rehashed in."""
    seeds = set(evidence["old_seeds"])
    identities = {source.graph_hash(graph) for graph in evidence["old_graphs"]}
    for i, graph in enumerate(result["graph_manifest"]):
        seed = graph["seed"]
        original = source.candidate(seed)
        answers = [
            source.distance_nodes(original["nodes"], original["edges"], original["start"], k)
            for k in (2, 3)
        ]
        if (
            seed in seeds
            or source.graph_hash(original) in identities
            or graph != fresh_graph(original, seed, i)
            or not answers[0]
            or len(answers[0]) != len(answers[1])
        ):
            return False
        seeds.add(seed)
        identities.add(source.graph_hash(original))
    return bool(result["complete"] == (len(result["graph_manifest"]) == 1024))


def build(work: Json, validation: Json, fixture: bool) -> Json:
    """A qualified freeze permits collection but supplies no detector benefit claim."""
    owned = diagnosis.admission.shard_a.qualified(validation, OWNED)
    failures, result = work["evidence"]["failures"], work["graphs"]
    controls = work["controls"]
    failed_owned = (not owned and not fixture) or not controls["passed"] and not failures
    cls = "disqualified" if failed_owned else "blocked" if failures else "null"
    ready = int(
        owned and not fixture and not failures and controls["passed"] and result["complete"]
    )
    count = len(result["graph_manifest"])
    rows = [
        dict(
            unit_id=f"v766-graph-{i:04d}-k2",
            cluster_id=f"v766-graph-{i:04d}",
            arm="shared_five_arm_roster",
            condition="frozen_fresh_roles",
            metric="graph_role_frozen",
            value=1 if i < count else None,
            numerator=1 if i < count else None,
            denominator=1,
            status="completed" if i < count else "censored",
            censor_reason=None
            if i < count
            else "external_evidence"
            if failures
            else result["terminal_reason"],
        )
        for i in range(1024)
    ]
    trained = work["trained"]
    value: Json = dict(
        experiment_id=8959,
        task_id=TASK,
        milestone="2026.11.766",
        run_date="20261109",
        random_seed=SEED,
        honest_verdict=f"complete_{cls}_relational_methods_"
        + (
            "frozen"
            if ready or fixture
            else "feasibility_null"
            if not failures and not failed_owned
            else "unavailable"
        ),
        verdict_class=cls,
        status="error" if cls == "disqualified" else "blocked" if failures else "success",
        inference_substrate="verifier_ensemble_against_cached_candidates"
        if trained["numerical_training_calls"]
        else "aggregation_from_upstream_artifacts",
        inference_substrate_class="no_model_load",
        execution_venue="host",
        MODEL_SPECS=[],
        model_specs=[],
        model_invoked=False,
        model_invocation_counts=dict(ZERO_INVOCATION_COUNTS),
        numerical_fit_count=trained["numerical_fit_count"] + controls["numerical_fit_count"],
        numerical_training_calls=trained["numerical_training_calls"]
        + controls["numerical_training_calls"],
        numerical_training_breakdown=dict(
            development=trained["numerical_training_calls"],
            private_controls=controls["numerical_training_calls"],
            validation_replays="additional fits belong to private validation receipts, not the main fit count",
        ),
        duration_s=work["duration_s"],
        phase_spans=work["phase_spans"],
        preconditions_checked=dict(
            failures=failures,
            historical_tasks_are_retained_inputs=True,
            weights_saved_before_fresh_construction=work["freeze_order"],
        ),
        verifier_is_oracle=False,
        exposure_scope="320 exposed historical development graphs;1024 fresh roles;no fresh responses or error labels",
        independent_generalization_score=0,
        relational_methods_ready_score=ready,
        intended_count=1024,
        completed_count=count,
        failed_count=0,
        censored_count=1024 - count,
        excluded_count=0,
        independent_count=count,
        rows=rows,
        gate_check_summary=failures,
        source_artifact_hashes=work["evidence"]["references"],
        code_config_hashes=work["sources"],
        raw_shard_hashes=work["raw_references"],
        required_checks_passed=owned,
        owned_statement_coverage=validation["coverage"],
        validation_receipts=validation["receipts"],
        global_test_debt=work["global_test_debt"],
        fixture_only=fixture,
        flagged_adversarial=False,
        acceptance_gates=dict(
            authenticated_inputs=dict(passed=not failures, principle="Custody precedes compute."),
            functioning_private_controls=dict(
                passed=controls["passed"],
                principle="Methods need a trainable informative fixture, not real-data gain.",
            ),
            complete_fresh_roster=dict(
                passed=result["complete"],
                principle="Exhaustion is a feasibility null, never population relaxation.",
            ),
            owned_checks=dict(
                passed=owned,
                principle="Failed owned checks disqualify; global debt stays separate.",
            ),
        ),
        source_manifest=dict(
            inputs=[list(row) for row in INPUTS],
            historical_graph_count=result["history_graph_count"],
            historical_proposed_seed_count=result["history_construction_seed_count"],
            original_producers_and_quarantines_preserved=True,
        ),
        role_manifest=[
            dict(
                unit_id=r["questions"][0]["unit_id"],
                cluster_id=r["graph_id"],
                graph_hash=r["graph_hash"],
                **{k: r[k] for k in ("shard", "role", "stream_id", "stream_position")},
            )
            for r in result["graph_manifest"]
        ],
        trained_state_hashes=[
            dict(arm=arm, seed=state["seed"], sha256=canonical_hash(state))
            for arm, _, _ in TRAINED_ARMS
            for state in trained["states"].get(arm, [])
        ],
        trained_state_reference=work["trained_reference"],
        frozen_protocol_reference=work["protocol_reference"],
        feature_schema=dict(
            allowed_keys=sorted(g.KEYS),
            node_features=["source_flag", "proposed_answer_flag", "constant1", "normalized_degree"],
            readout=[
                "source_embedding8",
                "answer_embedding8",
                "mean_embedding8",
                *b.FEATURES,
                "missing_answer",
            ],
            forbidden=[
                "shortest_distance",
                "BFS",
                "answer_set_membership",
                "graph_id",
                "split_id",
                "label",
            ],
            node_names="indexing only; never embedded or passed to the head",
        ),
        training_equations=protocol()["training_equations"],
        frozen_protocol=protocol(),
        fixture_control_rows=controls["rows"],
        graph_manifest=result["graph_manifest"],
        proposal_rejection_rows=result["proposal_rejection_rows"],
        terminal_reason=result["terminal_reason"],
        development_manifest=work["evidence"].get("development_manifest", {}),
        development_fit_ids=trained.get("fit_ids", []),
        development_guard_ids=trained.get("guard_ids", []),
        development_guard_rows=trained.get("guard_rows", []),
        methodology_note="This freezes an anonymous-graph adaptation, not KG-Guard reproduction. No fresh error labels, detector benefit or oracle-distinct generalization are measured. Labels for later evaluation use an exact graph oracle; weights and protocol precede collection.",
        reproduction_commands=[
            f"PYTHONPATH=python .venv/bin/python -u {CLI} --date 20261109",
            f".venv/bin/python -u {CLI} --cold-replay <primary.json>",
        ],
    )
    value["field_principles"] = {
        key: "Bind scheduled identity, authentic operands, independent graph units and scoped terminal disposition."
        for key in [*value, "field_principles", "reproducibility_checksum"]
    }
    value["field_principles"].update(
        relational_methods_ready_score="Methods readiness permits collection without asserting detector benefit.",
        trained_state_hashes="Freeze independent arm weights before fresh response labels exist.",
        numerical_training_calls="Count small-head optimizer updates separately from zero generator calls.",
        role_manifest="Seeds and perturbations never enlarge the independent graph denominator.",
        frozen_protocol="Freeze all five simultaneous bounds before fresh evidence.",
    )
    value["reproducibility_checksum"] = calibration.checksum(value)
    return value
