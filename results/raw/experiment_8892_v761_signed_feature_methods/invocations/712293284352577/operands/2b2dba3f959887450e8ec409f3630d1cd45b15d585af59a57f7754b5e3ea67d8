"""REQ-REPORT-8867: diagnose label information without changing historical learning.

Signed synthetic records have exact oracle labels. Repeated queries and renamed
entities do not create independent evidence about new query topologies.
"""

from __future__ import annotations

from collections import Counter
from copy import deepcopy
import json
from pathlib import Path
from typing import Any

import numpy as np

from carnot.reporting import linear_benefit_audit_8860 as old
from carnot.reporting.causal_learning_audit_8810 import load, reference, require
from carnot.reporting.current_work_receipt import canonical_hash

Json = dict[str, Any]
a = old.a
ROOT = old.ROOT
NAME = "experiment_8867_v759_feedback_information_diagnosis"
CLI = "scripts/experiments/" + NAME + ".py"
SOURCE = ROOT / "results/experiment_8860_v758_linear_benefit_audit.json"
PIN = "sha256:dbc0e247ad0aae63b5e69d87e6a56035bf95b417fbb118cc54cf071ef3d41434"
OWNED = [
    "python/carnot/reporting/feedback_information_8867.py",
    "python/carnot/reporting/feedback_information_runner_8867.py",
    CLI,
]
TESTS = ["tests/python/test_feedback_information_8867.py"]
GATES = dict(
    historical_streams=24,
    development_streams=6,
    complete_batches=True,
    positive_control_minimum_brier_gap=0.1,
    zero_signal_maximum_absolute_gap=0.03,
)
ARMS = ("matched_true", "matched_permuted", "frozen_query_only", "frozen_verified")


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Expose completed work while keeping the original finite deadlines visible."""
    print(f"[exp8867] phase={phase} completed={completed} pending={pending}", flush=True)


def query_template(query: list[list[str]]) -> str:
    """Accept syntax alone so admission cannot retain labels or witness answers."""
    return str(a.c.m.normalize(dict(query=query, facts=[])))


def permute(labels: list[str], batch: int) -> list[str]:
    """Permute one complete released batch; sampling with replacement is forbidden."""
    return [labels[int(i)] for i in np.random.default_rng(8867000 + batch).permutation(len(labels))]


def verify_release(value: Json, sealed: Json, label: str, ordinal: int) -> str:
    """Reject changed, early or false feedback before either learned control updates."""
    require(
        ordinal > sealed["ordinal"] and canonical_hash(value) == sealed["evidence_hash"],
        "release_boundary",
    )
    require(label in a.c.m.LABELS and label == a.c.m.decide(value)["decision"], "verified_feedback")
    return label


def preflight(raw: Path, source: Path) -> tuple[Json, Json, list[Json]]:
    """Pin the original audit and its execution authority, never the mutable roadmap."""
    work, pre, refs = old.preflight(raw, old.SOURCE)
    audit = load(dict(path=str(source), sha256=PIN))
    require(
        audit["required_checks_passed"] is True and audit["learning_audit_ready_score"] == 1,
        "audit_readiness:observed_zero",
    )
    audit_refs = (
        audit["source_artifact_hashes"] + audit["code_config_hashes"] + audit["raw_shard_hashes"]
    )
    a.prior.upstream.bindings(audit_refs, "original_audit_authority")
    require(
        audit["raw_shard_hashes"][0]["sha256"]
        == load(dict(path=str(old.SOURCE), sha256=old.PIN))["raw_shard_hashes"][0]["sha256"],
        "original_raw_authority",
    )
    pre.update(
        original_audit=reference(source),
        historical_refitting=False,
        development_indices=list(range(26, 32)),
        release_batch_size=6,
        preserved_protocol_hash=canonical_hash(work["protocol"]),
        no_model_load=True,
    )
    return (
        work,
        pre,
        list(
            {(r["path"], r["sha256"]): r for r in refs + audit_refs + [reference(source)]}.values()
        ),
    )


def historical(work: Json) -> Json:
    """Reconstruct saved states without gradients, then describe exposure and labels."""
    cardinality, labels, exposure, rows = [], [], [], []
    with a.prior.upstream.primitives():
        for index, stream in enumerate(work["stream_results"]):
            reconstructed = old.reconstruct_stream(stream, work["initialization_rows"])
            rows.extend(reconstructed["rows"])
            plan = stream["stream"]
            templates = [query_template(a.payload(w)["query"]) for w in plan["online"]]
            for arm in a.ARMS:
                feedback = [r for r in stream["feedback_rows"] if r["arm"] == arm]
                uses = [r for r in stream["template_reuse_rows"] if r["arm"] == arm]
                admissions = [r for r in feedback if r["template_added"]]
                cardinality.append(
                    dict(
                        cluster_id=plan["stream_id"],
                        arm=arm,
                        seed=plan["seed"],
                        distinct_normalized_query_templates=len(set(templates)),
                        query_templates=sorted(set(templates)),
                        marker_identities=len({w["structure_id"] for w in plan["online"]}),
                        marker_branches_are_unused=True,
                        observations=48,
                        repeated_queries=48 - len(set(templates)),
                        admissions=[
                            dict(
                                template=r["template"],
                                event=r["event_id"],
                                released_at=r["feedback_ordinal"],
                                prediction_at=r["prediction_ordinal"],
                            )
                            for r in admissions
                        ],
                        recurring_uses=len(uses),
                        first_use_delays=[
                            dict(
                                template=r["template"],
                                admission_to_first_use=min(
                                    u["ordinal"] for u in uses if u["template"] == r["template"]
                                )
                                - r["feedback_ordinal"],
                                first_query_to_release=r["feedback_ordinal"] - 1,
                            )
                            for r in admissions
                        ],
                    )
                )
                for event in feedback:
                    labels.append(
                        dict(
                            cluster_id=plan["stream_id"],
                            arm=arm,
                            seed=plan["seed"],
                            event_id=event["event_id"],
                            prediction_ordinal=event["prediction_ordinal"],
                            release_ordinal=event["feedback_ordinal"],
                            verified_label=event["verified_label"],
                            assigned_label=event["feedback_label"],
                            label_matches=event["feedback_label"] == event["verified_label"],
                            selected_prefix_index=(event["feedback_ordinal"] * 7)
                            % len(event["released_prefix"])
                            if "shuffled" in arm
                            else len(event["released_prefix"]) - 1,
                            prefix_length=len(event["released_prefix"]),
                            template_added=event["template_added"],
                            optimizer_steps=event["optimizer_steps"],
                        )
                    )
            for other in ("linear_additions_only", "linear_shuffled_combined"):
                left = [
                    r
                    for r in stream["rows"]
                    if r["arm"] == "linear_combined" and r["split"] == "online"
                ]
                right = [r for r in stream["rows"] if r["arm"] == other and r["split"] == "online"]
                assigned = {
                    arm: Counter(
                        r["feedback_label"] for r in stream["feedback_rows"] if r["arm"] == arm
                    )
                    for arm in ("linear_combined", other)
                }
                exposure.append(
                    dict(
                        cluster_id=plan["stream_id"],
                        comparison="linear_combined_minus_" + other,
                        denominator=48,
                        exposure_differences=sum(
                            x["sealed"]["template_uses"] != y["sealed"]["template_uses"]
                            for x, y in zip(left, right, strict=True)
                        ),
                        feature_differences=sum(
                            x["sealed"]["counts"] != y["sealed"]["counts"]
                            for x, y in zip(left, right, strict=True)
                        ),
                        brier_difference=float(
                            np.mean(
                                [
                                    x["metric"] - y["metric"]
                                    for x, y in zip(left, right, strict=True)
                                ]
                            )
                        ),
                        released_training_label_counts=assigned,
                        attribution="descriptive diagnosis, not causal proof",
                    )
                )
            progress("historical_streams", index + 1, 23 - index)
    return dict(
        rows=rows,
        template_cardinality_rows=cardinality,
        label_assignment_rows=labels,
        exposure_difference_rows=exposure,
    )


def sensitivity() -> tuple[list[Json], int]:
    """Measure a detectable injected signal and a feature-independent negative control."""
    result = []
    for signal in (True, False):
        heads = {arm: a.c.head(8867, "linear") for arm in ("true", "permuted")}
        trajectories = []
        labels = list(a.c.m.LABELS) * 2
        counts = [
            [
                5,
                5 * int(signal and i == 0),
                5 * int(signal and i == 1),
                0,
                0,
                5 * int(signal and i == 2),
            ]
            for i in range(3)
        ]
        for batch in range(20):
            assigned = permute(labels, batch)
            for i, truth in enumerate(labels):
                x = counts[a.c.m.LABELS.index(truth)]
                for arm, target in (("true", truth), ("permuted", assigned[i])):
                    heads[arm], gradient = a.c.update(heads[arm], x, target)
                    trajectories.append(
                        dict(
                            batch=batch,
                            ordinal=i,
                            arm=arm,
                            target=target,
                            gradient=gradient,
                            checkpoint=canonical_hash(heads[arm]),
                        )
                    )
        metrics = {
            arm: float(
                np.mean(
                    [
                        a.score(a.c.probabilities(head, x), label, 0.5)["metric"]
                        for x, label in zip(counts, a.c.m.LABELS, strict=True)
                    ]
                )
            )
            for arm, head in heads.items()
        }
        gap = metrics["permuted"] - metrics["true"]
        passed = (
            gap >= GATES["positive_control_minimum_brier_gap"]
            if signal
            else abs(gap) <= GATES["zero_signal_maximum_absolute_gap"]
        )
        result.append(
            dict(
                control="separable_signal" if signal else "zero_signal",
                passed=passed,
                true_brier=metrics["true"],
                permuted_brier=metrics["permuted"],
                gap=gap,
                denominator=3,
                numerical_training_calls=240,
                heads=heads,
                trajectories=trajectories,
                interpretation="injected diagnostic sensitivity; not a new headline",
            )
        )
        progress("sensitivity", len(result), 2 - len(result))
    return result, 480


def development(work: Json) -> Json:
    """Run six fixed new streams with sealed predictions and release-batch matching."""
    rows, batches, matched = [], [], []
    with a.prior.upstream.primitives():
        for index in range(26, 32):
            plan = a.stream(index)
            initial = next(
                r["parameters"] for r in work["initialization_rows"] if r["seed"] == plan["seed"]
            )
            states = {
                arm: dict(parameters=deepcopy(initial), templates=[], pending={}) for arm in ARMS
            }
            worlds = [a.payload(w) for w in plan["online"]]
            local_rows: dict[tuple[str, int], Json] = {}
            for ordinal in range(1, 57):
                if ordinal <= 48:
                    value = worlds[ordinal - 1]
                    for arm, state in states.items():
                        counts = a.c.m.observe(value, state["templates"])["counts"]
                        p = a.c.probabilities(state["parameters"], counts)
                        sealed = dict(
                            ordinal=ordinal,
                            evidence_hash=canonical_hash(value),
                            counts=counts,
                            prediction=p,
                            state_hash=canonical_hash(state),
                            template_active=query_template(value["query"]) in state["templates"],
                        )
                        state["pending"][str(ordinal)] = sealed
                        local_rows[(arm, ordinal)] = dict(
                            unit_id=plan["online"][ordinal - 1]["unit_id"],
                            cluster_id=plan["stream_id"],
                            seed=plan["seed"],
                            arm=arm,
                            ordinal=ordinal,
                            denominator=1,
                            status="complete",
                            missing_reason=None,
                            sealed=sealed,
                        )
                        if arm == "frozen_query_only":
                            template = query_template(value["query"])
                            if template not in state["templates"]:
                                state["templates"].append(template)
                if ordinal >= 14 and (ordinal - 8) % 6 == 0:
                    end = ordinal - 8
                    events = list(range(end - 5, end + 1))
                    labels = [
                        verify_release(
                            worlds[i - 1],
                            states["matched_true"]["pending"][str(i)],
                            str(a.c.m.decide(worlds[i - 1])["decision"]),
                            ordinal,
                        )
                        for i in events
                    ]
                    assigned = permute(labels, index * 100 + end)
                    batches.append(
                        dict(
                            cluster_id=plan["stream_id"],
                            release_ordinal=ordinal,
                            events=events,
                            verified_labels=labels,
                            permuted_labels=assigned,
                            predictions_sealed_before_release=True,
                            label_multiset_preserved=Counter(labels) == Counter(assigned),
                        )
                    )
                    for i, truth, target in zip(events, labels, assigned, strict=True):
                        value = worlds[i - 1]
                        template = query_template(value["query"])
                        for arm, state in states.items():
                            sealed = state["pending"].pop(str(i))
                            row = local_rows[(arm, i)]
                            row.update(
                                truth=truth,
                                metric=a.score(sealed["prediction"], truth, 0.5)["metric"],
                                release_ordinal=ordinal,
                                optimizer_steps=0,
                                training_label=None,
                            )
                            if arm != "frozen_query_only":
                                verify_release(value, sealed, truth, ordinal)
                                if template not in state["templates"]:
                                    state["templates"].append(template)
                                    state.setdefault("provenance", {})[template] = dict(
                                        verified_label=truth,
                                        witnesses=a.c.binding_provenance(value),
                                        release_ordinal=ordinal,
                                    )
                            if arm.startswith("matched_"):
                                label = target if arm == "matched_permuted" else truth
                                state["parameters"], gradient = a.c.update(
                                    state["parameters"], sealed["counts"], label
                                )
                                row.update(
                                    training_label=label, optimizer_steps=1, gradient=gradient
                                )
                            row["checkpoint_hash"] = canonical_hash(state)
            require(all(not s["pending"] for s in states.values()), "development_tail")
            current = list(local_rows.values())
            rows.extend(current)
            pairs = [
                (local_rows[("matched_true", i)], local_rows[("matched_permuted", i)])
                for i in range(1, 49)
            ]
            frozen = [
                (local_rows[("frozen_query_only", i)], local_rows[("frozen_verified", i)])
                for i in range(1, 49)
            ]
            matched.append(
                dict(
                    cluster_id=plan["stream_id"],
                    seed=plan["seed"],
                    denominator=48,
                    label_multiset_preserved=all(
                        b["label_multiset_preserved"]
                        for b in batches
                        if b["cluster_id"] == plan["stream_id"]
                    ),
                    schedules_equal=all(
                        x["release_ordinal"] == y["release_ordinal"]
                        and x["sealed"]["template_active"] == y["sealed"]["template_active"]
                        and x["sealed"]["counts"] == y["sealed"]["counts"]
                        for x, y in pairs
                    ),
                    optimizer_steps=[
                        sum(x[side]["optimizer_steps"] for x in pairs) for side in (0, 1)
                    ],
                    true_minus_permuted_brier=float(
                        np.mean([x["metric"] - y["metric"] for x, y in pairs])
                    ),
                    query_only_minus_verified_brier=float(
                        np.mean([x["metric"] - y["metric"] for x, y in frozen])
                    ),
                    frozen_weights_unchanged=all(
                        states[arm]["parameters"] == initial for arm in ARMS[2:]
                    ),
                    frozen_exposure_differences=sum(
                        x["sealed"]["template_active"] != y["sealed"]["template_active"]
                        for x, y in frozen
                    ),
                    final_states=states,
                )
            )
            progress("development_streams", index - 25, 31 - index)
    controls, calls = sensitivity()
    return dict(
        rows=rows,
        batches=batches,
        matched_control_rows=matched,
        control_sensitivity=controls,
        numerical_training_calls=576 + calls,
    )


def reduce(evidence: Json) -> Json:
    """Readiness releases the next method comparison even when every benefit is null."""
    history, dev = evidence["historical"], evidence["development"]
    rows = history["rows"] + dev["rows"]
    ready = (
        len(rows) == 28800
        and len(history["template_cardinality_rows"]) == 144
        and len(dev["matched_control_rows"]) == 6
        and len(dev["batches"]) == 48
        and all(
            r["label_multiset_preserved"]
            and r["schedules_equal"]
            and r["optimizer_steps"] == [48, 48]
            and r["frozen_weights_unchanged"]
            for r in dev["matched_control_rows"]
        )
        and all(r["passed"] for r in dev["control_sensitivity"])
    )
    return dict(
        history,
        rows=rows,
        matched_control_rows=dev["matched_control_rows"],
        control_sensitivity=dev["control_sensitivity"],
        numerical_training_calls=dev["numerical_training_calls"],
        intended_count=28800,
        completed_count=len(rows),
        failed_count=0,
        censored_count=28800 - len(rows),
        excluded_count=0,
        independent_count=24,
        diagnosis_ready_score=int(ready),
        effective_information=dict(
            historical_stream_clusters=24,
            development_stream_clusters=6,
            distinct_query_templates=len(
                {t for r in history["template_cardinality_rows"] for t in r["query_templates"]}
            ),
            repeated_observations_are_independent=False,
            historical_shuffled_label_mismatches=sum(
                not r["label_matches"]
                for r in history["label_assignment_rows"]
                if r["arm"] == "linear_shuffled_combined"
            ),
            historical_shuffled_exposure_differences=sum(
                r["exposure_differences"]
                for r in history["exposure_difference_rows"]
                if r["comparison"] == "linear_combined_minus_linear_shuffled_combined"
            ),
            historical_shuffled_multiset_changes=sum(
                len(
                    {
                        canonical_hash(counts)
                        for counts in r["released_training_label_counts"].values()
                    }
                )
                > 1
                for r in history["exposure_difference_rows"]
                if r["comparison"] == "linear_combined_minus_linear_shuffled_combined"
            ),
            scope="stream randomization supports paired stream descriptives; repeated topology is not independent structural information",
        ),
        forward_method=dict(
            experiment_id=8869,
            arms=["frozen_verified", "matched_true", "matched_permuted", "frozen_query_only"],
            release_policy="complete batches after prediction seals; preserve multiset, events, optimizer steps and verified admission schedule",
            unsupported_mechanism_reason="One recurring normalized query topology and unused marker diversity do not establish structure-generalizing label information.",
            interpretation="release-matched ablations only; no new optimizer or learning-method selection",
        ),
    )


def empty(reason: str) -> Json:
    """Retain scheduled units when external absence prevents diagnosis."""
    rows = a.census(a.protocol(), reason)
    rows += [
        dict(
            unit_id=w["unit_id"],
            cluster_id=plan["stream_id"],
            seed=plan["seed"],
            arm=arm,
            denominator=1,
            metric=None,
            status="censored",
            missing_reason=reason,
        )
        for plan in [a.stream(i) for i in range(26, 32)]
        for arm in ARMS
        for w in plan["online"]
    ]
    return dict(
        rows=rows,
        intended_count=28800,
        completed_count=0,
        failed_count=0,
        censored_count=28800,
        excluded_count=0,
        independent_count=0,
        diagnosis_ready_score=0,
        numerical_training_calls=0,
        template_cardinality_rows=[],
        label_assignment_rows=[],
        exposure_difference_rows=[],
        matched_control_rows=[],
        control_sensitivity=[],
        effective_information={},
        forward_method=dict(experiment_id=8869, unsupported_mechanism_reason=reason),
    )


def replay(path: Path) -> Json:
    """Recompute original history and controls so repaired summary hashes cannot grant trust."""
    value = json.loads(path.read_bytes())
    a.prior.upstream.bindings(
        value["source_artifact_hashes"] + value["code_config_hashes"] + value["raw_shard_hashes"],
        "replay_operand",
    )
    require(value["acceptance_gates"] == GATES, "acceptance_drift")
    evidence = load(value["raw_shard_hashes"][0])
    if evidence:
        original = load(dict(path=str(SOURCE), sha256=PIN))
        require(original["required_checks_passed"] is True, "original_readiness")
        work = load(original["raw_shard_hashes"][0])
        require(evidence["historical"] == historical(work), "evidence_drift:historical")
        require(evidence["development"] == development(work), "evidence_drift:development")
        expected = reduce(evidence)
    else:
        expected = empty(value["rows"][0]["missing_reason"])
    if value["verdict_class"] in {"blocked", "disqualified", "partial"}:
        expected["diagnosis_ready_score"] = 0
    for key, observed in expected.items():
        require(value[key] == observed, "reduction_drift:" + key)
    for receipt in value["validation_receipts"]:
        if "log_path" in receipt:
            a.prior.upstream.bindings(
                [dict(path=str(ROOT / receipt["log_path"]), sha256=receipt["log_sha256"])],
                "validation_log",
            )
    return dict(passed=True, replay_passed=True)
