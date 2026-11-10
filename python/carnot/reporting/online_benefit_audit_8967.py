"""REQ-REPORT-8967: reduce retained predictions without fitting or relabeling."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from carnot.reporting import online_response_streams_8966 as upstream
from carnot.reporting.causal_learning_audit_8810 import require as check
from carnot.reporting.current_work_receipt import canonical_hash
from carnot.reporting.current_work_receipt import build_current_work_receipt
from carnot.reporting.depth_opportunity_8945 import Rejected, require

Json = dict[str, Any]
__all__ = ["check"]
ROOT = upstream.ROOT
NAME, TASK = "experiment_8967_v766_online_benefit_audit", "exp8967-online-benefit-audit"
METHOD_TASK, SOURCE_TASK = "exp8960-online-calibration-methods", upstream.TASK
SEED, COUNT = 766060, 39936
ARMS = ("frozen", "bias_only", "adaptive", "permuted")
CLI = f"scripts/experiments/{NAME}.py"
OWNED = [
    "python/carnot/reporting/online_benefit_audit_8967.py",
    "python/carnot/reporting/online_benefit_runner_8967.py",
    CLI,
]
TESTS = ["tests/python/test_online_benefit_audit_8967.py"]
INPUTS = [
    (
        "results/experiment_8960_v766_online_calibration_methods.json",
        "sha256:1af36d37457cd5b7cda9be1b6cc80bc89c96f29dcf6427c4fbb2946728b3b09b",
        "online_methods_ready_score",
        METHOD_TASK,
    ),
    (
        f"results/{upstream.NAME}.json",
        "sha256:906a2143c36d902d19615ddaef00f5fc4b3b1c0c883e77f54e91c0821ac92cbd",
        "online_rows_ready_score",
        SOURCE_TASK,
    ),
]
ref = upstream.ref


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Flushed counts make each phase visible to the task supervisor."""
    print(f"[exp8967] phase={phase} completed={completed} pending={pending}", flush=True)


def authenticate(root: Path) -> Json:
    """Byte pins and typed gates prevent changed or absent inputs from becoming science."""
    result: Json = dict(failures=[], references=[], producers={})
    seen: set[str] = set()
    for name, pin, field, task in INPUTS:
        path = root / name
        result["references"].append(ref(path))
        try:
            require(ref(path)["sha256"], pin, path, "sha256", task)
            value = json.loads(path.read_bytes())
            for key, expected in ((field, 1), ("task_id", task), ("required_checks_passed", True)):
                require(value.get(key), expected, path, key, task)
            operands = (
                value["source_artifact_hashes"]
                + value["code_config_hashes"]
                + value["raw_shard_hashes"]
            )
            for i, operand in enumerate(operands):
                if operand["path"] not in seen:
                    target = Path(operand["path"])
                    require(ref(target)["sha256"], operand["sha256"], target, "sha256", task)
                    result["references"].append(ref(target))
                    seen.add(operand["path"])
                if i % 250 == 0:
                    progress("authenticate_operands", i, len(operands) - i)
            result["producers"][task] = value
        except Rejected as error:
            result["failures"].append(dict(error.receipt, check=error.receipt["failure_kind"]))
        except (OSError, ValueError, KeyError, TypeError) as error:
            result["failures"].append(
                dict(
                    upstream=task,
                    path=str(path),
                    hash=ref(path)["sha256"],
                    check="rejected_producer",
                    artifact_field="producer_json",
                    expected="original complete object",
                    observed=str(error),
                )
            )
    return result


def reconstruct(producers: Json) -> list[Json]:
    """Rejoin durable predictions with existing labels; never call the adapter or oracle."""
    source, methods = producers[SOURCE_TASK], producers[METHOD_TASK]
    check(
        source["independent_count"] == source["completed_count"] == len(source["streams"]) == 24
        and len(source["rows"]) == COUNT,
        "complete_roster",
    )
    check(
        [dict(stream_id=s["stream_id"], unit_ids=s["unit_ids"]) for s in source["streams"]]
        == methods["streams"],
        "stream_identities",
    )
    check(
        len(source["control_rows"]) == len(source["restart_rows"]) == 24
        and all(r["passed"] for r in source["control_rows"] + source["restart_rows"]),
        "upstream_controls",
    )
    corpus_ref = next(
        r
        for r in source["source_artifact_hashes"]
        if Path(r["path"]).name == "experiment_8963_v766_corpus_admission.json"
    )
    corpus = json.loads(Path(corpus_ref["path"]).read_bytes())
    manifest = {r["unit_id"]: r for r in corpus["corpus_manifest"]}
    truth_ref = corpus["broker_manifest"]["test"]["label_reference"]
    truth = {
        r["unit_id"]: r["label"] for r in json.loads(Path(truth_ref["path"]).read_bytes())["rows"]
    }
    raw = Path(source["raw_shard_hashes"][0]["path"]).parent
    online: list[Json] = []
    retention: list[Json] = []
    for index, stream in enumerate(source["streams"]):
        sid = stream["stream_id"]
        private = raw / "streams" / sid
        state = json.loads((private / "state.json").read_bytes())
        request = json.loads((private / "request.json").read_bytes())
        check(
            canonical_hash(state) == stream["final_state_hash"]
            and state["request_hash"] == canonical_hash(request),
            "durable_state",
        )
        check(
            state["params"] == stream["final_params"]
            and state["params"]["frozen"] == [1.0, 0.0]
            and stream["pending"] == state["pending"] == [29, 30, 31, 32]
            and state["released_through"] == 28,
            "arm_identity",
        )
        check(
            request["unit_ids"] == stream["unit_ids"]
            and request["retention_ids"] == source["retention_ids"]
            and len(set(request["retention_ids"])) == 128,
            "query_schedule",
        )
        check(
            {(r["query"], r["arm"], r["unit_id"]) for r in state["rows"]}
            == {(q, a, u) for q, u in enumerate(stream["unit_ids"], 1) for a in ARMS}
            and len(state["rows"]) == 128,
            "online_schedule",
        )
        check(
            {(r["query"], r["arm"], r["unit_id"]) for r in state["retention_rows"]}
            == {(q, a, u) for q in (0, 16, 32) for a in ARMS for u in request["retention_ids"]}
            and len(state["retention_rows"]) == 1536,
            "retention_schedule",
        )
        labels = json.loads(Path(request["label_path"]).read_bytes())["labels"]
        for feedback in state["feedback_rows"]:
            q = feedback["query"]
            check(
                feedback["labels"] == labels[q - 8 : q - 4]
                and feedback["permuted_labels"] == [feedback["labels"][i] for i in (1, 0, 3, 2)],
                "feedback_identity",
            )
        for r in state["rows"]:
            unit = manifest[r["unit_id"]]
            online.append(
                dict(
                    upstream.metric_row(r, labels[r["query"] - 1], sid, "online"),
                    graph_hash=unit["graph_hash"],
                    first_response_reference=unit["first_response_reference"],
                )
            )
        retention.extend(
            upstream.metric_row(r, truth[r["unit_id"]], sid, "retention")
            for r in state["retention_rows"]
        )
        progress("reconstructed_histories", index + 1, 23 - index)
    check(
        online == source["prediction_rows"]
        and retention == source["retention_rows"]
        and online + retention == source["rows"],
        "query_rows",
    )
    return online + retention


def bootstrap(grid: Any, seed: int) -> tuple[Any, Any]:
    """Share one graph sample across histories; conditional bounds hold that panel fixed."""
    rng = np.random.default_rng(seed)
    sampled, conditional = np.empty(10000), np.empty(10000)
    for i in range(10000):
        streams = rng.integers(24, size=24)
        graphs = rng.integers(128, size=128)
        subset = grid[streams]
        conditional[i] = subset.mean()
        sampled[i] = subset[:, graphs].mean()
    return sampled, conditional


def accepts(upper: float, bound: Json) -> bool:
    """Equality is allowed only for the frozen and retention margins."""
    return bool(upper < bound["upper"] if bound["strict"] else upper <= bound["upper"])


def benefit(passed: list[bool], diagnostics: Json) -> int:
    """Parameter motion cannot rescue failed benefit, headroom or attribution controls."""
    return int(
        len(passed) == 9
        and all(passed)
        and diagnostics["feedback_positions_differ"] > 0
        and diagnostics["frozen_future_brier_headroom"] > 0
    )


def summarize(rows: list[Json], ledger: Json) -> Json:
    """Reduce paired effects with the frozen nine-bound, two-axis bootstrap."""
    streams = sorted({r["stream_id"] for r in rows})
    groups: dict[tuple[Any, ...], list[Json]] = {}
    for r in rows:
        if r["condition"] == "retention" or r["query"] >= 17:
            key = (
                r["stream_id"],
                r["condition"],
                r["query"] if r["condition"] == "retention" else None,
                r["arm"],
            )
            groups.setdefault(key, []).append(r)
    paired, panel, intervals = [], [], []
    for index, bound in enumerate(ledger["bounds"]):
        condition, checkpoint = bound["split"], bound.get("checkpoint")
        metric = "brier" if bound["metric"] == "brier" else "classification_error"
        arm_values = {}
        for arm in ("adaptive", bound["against"]):
            arm_values[arm] = np.asarray(
                [
                    [
                        r["metrics"][metric]["value"]
                        for r in sorted(
                            groups[sid, condition, checkpoint, arm],
                            key=lambda r: r["unit_id"] if condition == "retention" else r["query"],
                        )
                    ]
                    for sid in streams
                ]
            )
        differences = arm_values["adaptive"] - arm_values[bound["against"]]
        for i, sid in enumerate(streams):
            value = float(differences[i].mean())
            paired.append(
                dict(
                    unit_id=f"{sid}/bound-{index}",
                    cluster_id=sid,
                    arm="adaptive",
                    condition=condition,
                    checkpoint=checkpoint,
                    comparison="adaptive_minus_" + bound["against"],
                    metric=metric,
                    paired_means={a: float(v[i].mean()) for a, v in arm_values.items()},
                    value=value,
                    numerator=value,
                    denominator=1,
                    query_denominator=differences.shape[1],
                    status="completed",
                    censor_reason=None,
                )
            )
        grid = (
            differences
            if condition == "retention"
            else np.repeat(differences.mean(axis=1)[:, None], 128, axis=1)
        )
        sampled, conditional = bootstrap(grid, ledger["seed"])
        upper = float(np.quantile(sampled, 1 - ledger["alpha_per_bound"], method="linear"))
        fixed_upper = float(
            np.quantile(conditional, 1 - ledger["alpha_per_bound"], method="linear")
        )
        intervals.append(
            dict(
                bound,
                bound_id=index,
                comparison="adaptive_minus_" + bound["against"],
                estimate=float(differences.mean()),
                upper=upper,
                conditional_on_panel_upper=fixed_upper,
                operator="<" if bound["strict"] else "<=",
                threshold=bound["upper"],
                passed=accepts(upper, bound),
                independent_count=24,
                status="completed",
                principle="All nine prespecified bounds must pass; retention shares one graph draw across histories.",
            )
        )
        if condition == "retention":
            graph_ids = sorted(
                r["unit_id"] for r in groups[streams[0], condition, checkpoint, "adaptive"]
            )
            for j, unit in enumerate(graph_ids):
                numerator = float(differences[:, j].sum())
                panel.append(
                    dict(
                        unit_id=f"{unit}/bound-{index}",
                        cluster_id=unit,
                        graph_id=unit,
                        arm="adaptive",
                        condition=condition,
                        checkpoint=checkpoint,
                        metric=metric,
                        value=numerator / 24,
                        numerator=numerator,
                        denominator=24,
                        status="completed",
                        censor_reason=None,
                    )
                )
        progress("bootstrap_bounds", index + 1, 8 - index)
    return dict(
        paired_stream_rows=paired,
        shared_panel_rows=panel,
        acceptance_gate_rows=intervals,
        multiplicity_ledger=ledger,
    )


def diagnostics(source: Json, rows: list[Json]) -> Json:
    """Describe actual control discrimination and headroom separately from success."""
    feedback = source["feedback_rows"]
    future = [r for r in rows if r["condition"] == "online" and r["query"] >= 17]
    frozen = {(r["stream_id"], r["query"]): r for r in future if r["arm"] == "frozen"}
    promotions = source["promotion_rows"]
    changed = sum(
        a != b for r in feedback for a, b in zip(r["labels"], r["permuted_labels"], strict=True)
    )
    total = sum(len(r["labels"]) for r in feedback)
    return dict(
        feedback_positions=total,
        feedback_positions_differ=changed,
        feedback_difference_fraction=changed / total,
        feedback_blocks=len(feedback),
        feedback_blocks_differ=sum(r["labels"] != r["permuted_labels"] for r in feedback),
        frozen_future_brier_headroom=float(np.mean([r["value"] for r in frozen.values()])),
        frozen_future_classification_error=float(
            np.mean([r["metrics"]["classification_error"]["value"] for r in frozen.values()])
        ),
        arm_rows=[
            dict(
                arm=arm,
                proposals=sum(r["arm"] == arm for r in promotions),
                accepted=sum(r["arm"] == arm and r["accepted"] for r in promotions),
                parameter_changes=sum(
                    r["arm"] == arm and r["before"] != r["after"] for r in promotions
                ),
                final_parameter_changes=sum(
                    s["final_params"][arm] != [1.0, 0.0] for s in source["streams"]
                ),
                future_probability_changes=sum(
                    r["arm"] == arm
                    and r["p_error"] != frozen[r["stream_id"], r["query"]]["p_error"]
                    for r in future
                ),
                future_decision_changes=sum(
                    r["arm"] == arm
                    and (r["p_error"] >= 0.5)
                    != (frozen[r["stream_id"], r["query"]]["p_error"] >= 0.5)
                    for r in future
                ),
            )
            for arm in ARMS
        ],
        interpretation="Descriptive mechanism activity cannot replace future benefit or retention bounds.",
    )


def reduce(evidence: Json) -> Json:
    """All comparative headlines come from authenticated typed query rows."""
    rows = reconstruct(evidence["producers"])
    ledger = evidence["producers"][METHOD_TASK]["multiplicity_ledger"]
    result = summarize(rows, ledger)
    mechanism = diagnostics(evidence["producers"][SOURCE_TASK], rows)
    result.update(
        rows=rows,
        mechanism_diagnostics=mechanism,
        intended_count=COUNT,
        completed_count=COUNT,
        failed_count=0,
        censored_count=0,
        excluded_count=0,
        independent_count=24,
        online_benefit_score=benefit(
            [r["passed"] for r in result["acceptance_gate_rows"]], mechanism
        ),
    )
    return result


def empty_result() -> Json:
    """Retain every scheduled query slot when observations cannot be authenticated."""
    rows = [
        dict(
            unit_id=f"scheduled/{sid}/{condition}/{query}/{i}",
            cluster_id=sid,
            arm=arm,
            condition=condition,
            query=query,
            metric="brier",
            value=None,
            numerator=None,
            denominator=1,
            status="censored",
            censor_reason="unavailable_authenticated_reduction",
        )
        for sid in [f"{batch}-stream-{i:02d}" for batch in ("A", "B") for i in range(12)]
        for condition, queries, n in (("online", range(1, 33), 1), ("retention", (0, 16, 32), 128))
        for query in queries
        for i in range(n)
        for arm in ARMS
    ]
    return dict(
        rows=rows,
        paired_stream_rows=[],
        shared_panel_rows=[],
        acceptance_gate_rows=[],
        multiplicity_ledger={},
        mechanism_diagnostics={},
        intended_count=COUNT,
        completed_count=0,
        failed_count=0,
        censored_count=COUNT,
        excluded_count=0,
        independent_count=0,
        online_benefit_score=0,
    )


def disposition(work: Json, validation: Json, fixture: bool) -> tuple[bool, str]:
    """External failures block; owned failures disqualify; a valid scientific null qualifies."""
    coverage = validation["coverage"]
    qualified = bool(
        not fixture
        and validation["receipts"]
        and all(r["passed"] for r in validation["receipts"])
        and coverage.get("complete")
        and coverage.get("percent") == 100
        and {r["path"] for r in coverage.get("files", [])} == set(OWNED)
        and all(
            r["statements"] and not r["missing"] and not r["excluded"]
            for r in coverage.get("files", [])
        )
    )
    external = [r for r in work["evidence"]["failures"] if r["check"] != "owned_reduction"]
    ready = (
        qualified
        and not work["owned_failure"]
        and not external
        and work["result"]["completed_count"] == COUNT
    )
    return (
        ready,
        "blocked"
        if external
        else "disqualified"
        if not ready
        else "circular_positive"
        if work["result"]["online_benefit_score"]
        else "null",
    )


def build(work: Json, validation: Json, fixture: bool) -> Json:
    """Keep audit completion separate from useful learning and restrict any retirement."""
    ready, cls = disposition(work, validation, fixture)
    failures = list(work["evidence"]["failures"])
    if cls == "disqualified":
        failures.append(
            dict(
                upstream=TASK,
                path=str(ROOT / CLI),
                hash=ref(ROOT / CLI)["sha256"],
                check="owned_controls",
                artifact_field="owned_qualification",
                expected=True,
                observed=False,
            )
        )
    verdict = f"complete_{cls}_real_response_online_benefit_audit"
    previous = work["prior_terminal"]
    if (
        cls in ("blocked", "disqualified")
        and previous.get("verdict_class") == cls
        and previous.get("gate_check_summary") == failures
    ):
        verdict = previous["honest_verdict"]
    receipt = build_current_work_receipt(
        run_id=str(work["started"]),
        owner_pid=work["pid"],
        events=[],
        inference_substrate="aggregation_from_upstream_artifacts",
        inference_substrate_class="no_model_load",
        execution_venue="host",
        inference_substrate_details=dict(
            scope="Read-only retained prediction and label reduction; no new inference, fitting or labeling."
        ),
        started_monotonic_ns=work["started"],
        ended_monotonic_ns=work["ended"],
        phase_spans=work["phase_spans"],
    )
    value: Json = dict(
        work["result"],
        experiment_id=8967,
        task_id=TASK,
        milestone="2026.11.766",
        run_date="20261110",
        random_seed=SEED,
        honest_verdict=verdict,
        verdict_class=cls,
        inference_substrate=receipt["inference_substrate"],
        inference_substrate_class="no_model_load",
        execution_venue="host",
        MODEL_SPECS=[],
        model_specs=[],
        model_invocation_counts=receipt["invocation_counts"],
        model_invoked=False,
        numerical_training_calls=0,
        retained_training_scope=dict(
            scheduled_adapter_steps=504,
            dependent_control_steps=1512,
            counted_as_current_training=False,
        ),
        current_work_receipt=receipt,
        duration_s=receipt["duration_s"],
        phase_spans=work["phase_spans"],
        preconditions_checked=dict(
            failures=work["evidence"]["failures"], historical_tasks_are_retained_inputs=True
        ),
        verifier_is_oracle=True,
        exposure_scope="Retained real Qwen graph errors with cheap BFS truth; shared final-test panel previously opened by Exp8964; no oracle-distinct or unseen-domain claim.",
        independent_generalization_score=0,
        source_artifact_hashes=work["evidence"]["references"],
        code_config_hashes=work["sources"],
        raw_shard_hashes=[
            ref(Path(work["raw"]) / "work.json"),
            ref(Path(work["raw"]) / "validation.json"),
        ]
        + work["raw_references"],
        gate_check_summary=failures,
        acceptance_gates={
            name: dict(passed=passed, principle=principle)
            for name, passed, principle in (
                (
                    "authenticated_inputs",
                    not work["evidence"]["failures"],
                    "Original retained evidence must authenticate before reduction.",
                ),
                (
                    "complete_reduction",
                    work["result"]["completed_count"] == COUNT,
                    "All39936 rows and24 paired histories remain visible.",
                ),
                (
                    "owned_checks",
                    ready,
                    "Zero-exclusion owned unit/child coverage and required checks must pass.",
                ),
            )
        },
        online_audit_ready_score=int(ready),
        online_benefit_score=work["result"]["online_benefit_score"] if ready else 0,
        required_checks_passed=ready,
        owned_statement_coverage=validation["coverage"],
        validation_receipts=validation["receipts"],
        global_test_debt=work["global_test_debt"],
        fixture_only=fixture,
        continuation_decision=dict(
            action="retire_exact_protocol"
            if ready and cls == "null"
            else "eligible_only_within_scope"
            if ready
            else "blocked_by_evidence"
            if cls == "blocked"
            else "repair_owned_controls",
            scope="V766 fixed two-scalar adapter, real-response data, four-query delayed blocks, historical64-graph guard and shared128-graph panel",
            self_learning_in_general_retired=False,
            prior_v765_eleven_bound_null_unchanged=True,
            post_test_extra_data=False,
            weakened_margins=False,
        ),
        methodology_note="All nine one-sided Bonferroni bounds;10000 paired stream draws with independent shared graph-ID samples. Conditional-on-panel bounds are descriptive. Repeated seeds and restarts add no independent histories. No generator calls, adapter refit or label recomputation.",
        flagged_adversarial=False,
    )
    value["field_principles"] = {
        key: "Bind this invocation to authenticated retained observations and explicitly scoped evidence."
        for key in value
    }
    for keys, principle in [
        (
            (
                "experiment_id",
                "task_id",
                "milestone",
                "run_date",
                "random_seed",
                "reproducibility_checksum",
            ),
            "Bind the result to the scheduled task, input bytes and reproducible randomness.",
        ),
        (
            ("honest_verdict", "verdict_class"),
            "An honest terminal outcome preserves exact rerun retirement and cannot overstate claim strength.",
        ),
        (
            ("inference_substrate", "inference_substrate_class", "execution_venue"),
            "Declare actual aggregation_from_upstream_artifacts/no_model_load and host venue.",
        ),
        (
            ("MODEL_SPECS", "model_specs", "model_invocation_counts", "numerical_training_calls"),
            "Distinguish retained training and outputs from zero current generator and training calls.",
        ),
        (
            ("duration_s", "phase_spans", "preconditions_checked"),
            "Actual clocks and resource receipts prevent fabricated execution or padding.",
        ),
        (
            ("verifier_is_oracle", "exposure_scope", "independent_generalization_score"),
            "Cheap BFS truth and graph-domain evidence do not establish the oracle-distinct north star.",
        ),
        (
            (
                "intended_count",
                "completed_count",
                "failed_count",
                "censored_count",
                "excluded_count",
                "independent_count",
                "rows",
            ),
            "Keep every scheduled unit and typed measurement; repeated queries never enlarge independent streams.",
        ),
        (
            ("gate_check_summary",),
            "Name upstream, path/hash, check, exact field and expected/observed values for each failure.",
        ),
        (
            (
                "source_artifact_hashes",
                "code_config_hashes",
                "raw_shard_hashes",
                "field_principles",
            ),
            "Retain byte custody and state why evidence fields exist.",
        ),
        (
            (
                "acceptance_gates",
                "required_checks_passed",
                "owned_statement_coverage",
                "validation_receipts",
                "global_test_debt",
                "online_audit_ready_score",
            ),
            "Audit readiness requires complete reproducible reduction and owned checks; global debt is separate.",
        ),
        (
            (
                "online_benefit_score",
                "paired_stream_rows",
                "shared_panel_rows",
                "multiplicity_ledger",
                "acceptance_gate_rows",
                "continuation_decision",
            ),
            "Useful learning requires all nine bounds, discriminating feedback and nonzero frozen headroom jointly; null retires only this protocol.",
        ),
    ]:
        value["field_principles"].update(dict.fromkeys(keys, principle))
    value["reproducibility_checksum"] = canonical_hash(value)
    return value
