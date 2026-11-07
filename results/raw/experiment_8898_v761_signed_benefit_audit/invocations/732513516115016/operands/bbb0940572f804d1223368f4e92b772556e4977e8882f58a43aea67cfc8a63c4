"""REQ-REPORT-8886: independently audit saved oracle evidence without training."""

from collections import Counter, defaultdict
from copy import deepcopy
import json
from pathlib import Path
import time
from typing import Any

import numpy as np

from carnot.memory import categorical_memory_8830 as c
from carnot.memory import causal_memory_8802 as primitive
from carnot.reporting import evidence_set_methods_8768 as symbolic
from carnot.reporting import feedback_information_8867 as diagnosis
from carnot.reporting import topology_memory_methods_8881 as base
from carnot.reporting import topology_causal_streams_8884 as causal
from carnot.reporting import conditional_risk_runner_8868 as inherited
from carnot.reporting.causal_learning_audit_8810 import require, score
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash, sha256_file
from carnot.reporting.v746_retained_evidence import reference, load
from carnot.reporting.v748_collection_contract import namespace

__all__ = ["base", "require"]

Json = dict[str, Any]
ROOT = base.ROOT
NAME, TASK = "experiment_8886_v760_label_benefit_audit", "exp8886-label-benefit-audit"
SOURCE = "experiment_8884_v760_topology_causal_streams"
PIN = "sha256:9ea71573193ea86b4fb5f430685f8bbadf2862d16050e0b9d010c624b3d1ba4d"
METHOD = "experiment_8881_v760_topology_memory_methods"
METHOD_PIN = "sha256:4d43ee0ea01ae703efb1087b5a2f2a253c529f46d3e2c9d7f87ff2b028d3c1e3"
CLI = "scripts/experiments/" + NAME + ".py"
OWNED = [
    "python/carnot/reporting/label_benefit_audit_8886.py",
    "python/carnot/reporting/label_benefit_runner_8886.py",
    CLI,
]
TESTS = ["tests/python/test_label_benefit_audit_8886.py"]


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Expose finite work counts without extending any deadline."""
    print(f"[exp8886] phase={phase} completed={completed} pending={pending}", flush=True)


def preflight(raw: Path, sources: Path) -> Json:
    """Original producer pins bind the authority and every retained operand before scoring."""
    checker = namespace(inherited, "label8886_preconditions")
    checker.UPSTREAM = {
        SOURCE: (PIN, "causal_rows_ready_score"),
        METHOD: (METHOD_PIN, "topology_methods_ready_score"),
    }
    checker.DESIGN = "openspec/change-proposals/research-methods-v760-20261106.json"
    checker.DESIGN_PIN = "sha256:c99296bd441c10200bad04e77643649bd65f62fb2199029302b24473af6da3d7"
    progress("authenticate_before", 0, 2)
    pre = dict(checker.preconditions(raw, sources))
    if pre["external_ready"]:
        for name in (SOURCE, METHOD):
            value = load(reference(raw / (name + ".json")))
            accepted = bool(
                value["required_checks_passed"]
                and value["owned_statement_coverage"]["complete"]
                and value["owned_statement_coverage"]["percent"] == 100
                and all(r["passed"] for r in value["control_rows"])
                and value["protocol"] == base.protocol()
            )
            if not accepted:
                pre["diagnostics"].append(
                    dict(
                        check="rejected_producer",
                        upstream=name,
                        artifact_field="original_validation_and_protocol",
                        expected=True,
                        observed=False,
                        **reference(sources / (name + ".json")),
                    )
                )
        pre["external_ready"] = not pre["diagnostics"]
    if pre["external_ready"]:
        original = load(reference(raw / (SOURCE + ".json")))
        operands = original["stream_refs"] + [
            reference(Path(original["raw_directory"]) / "fit.json")
        ]
        retained = raw / "retained_operands"
        retained.mkdir()
        pre["retained_original_operands"] = []
        for i, ref in enumerate(operands):
            path = retained / Path(ref["path"]).name
            path.write_bytes(Path(ref["path"]).read_bytes())
            require(sha256_file(path) == ref["sha256"], "snapshot_bytes")
            pre["retained_original_operands"].append(
                dict(reference(path), original_path=ref["path"])
            )
            if i % 6 == 0:
                progress("retained_operands", i + 1, len(operands) - i - 1)
    if pre["external_ready"] and pre["resources"]["passed"]:
        progress("pilot_before", 0, 1)
        pilot_start = time.monotonic()
        reconstruct(load(original["stream_refs"][0]), base.protocol()["cohort_manifest"][0])
        pilot_s = time.monotonic() - pilot_start
        projected = pilot_s * 24 + 1200
        pre["pilot"] = dict(
            duration_s=pilot_s, projected_total_s=projected, validation_reserve_s=1200, limit_s=4200
        )
        pre["resources"]["passed"] = projected < 4200
        progress("pilot_after", 1, 0)
    pre["resources"].update(
        numerical_training_calls=0, model_loads=0, hard_cap_s=4800, projected_total_s=1200
    )
    progress("authenticate_after", 2, 0)
    return pre


def reconstruct(stream: Json, plan: Json) -> list[Json]:
    """Join serving states to seals and releases; recorded parameters are never refitted."""
    require(stream["stream"] == plan, "stream_identity")
    schedule = {
        max(i + plan["delays"][i - 1] for i in range(end - 5, end + 1)): list(
            range(end - 5, end + 1)
        )
        for end in range(6, 49, 6)
    }
    require(
        [(b["release_ordinal"], b["events"]) for b in stream["batches"]] == list(schedule.items()),
        "release_schedule",
    )
    batches = {b["release_ordinal"]: b for b in stream["batches"]}
    for key, expected_rows in causal.evidence(stream).items():
        observed_rows = stream[key]
        if key == "tail_drain_rows":
            expected_rows = sorted(expected_rows, key=lambda r: r["arm"])
            observed_rows = sorted(observed_rows, key=lambda r: r["arm"])
        require(observed_rows == expected_rows, "evidence_join:" + key)
    require(
        len(stream["rows"]) == 768 and all(r["passed"] for r in stream["controls"]), "stream_census"
    )
    for b in batches.values():
        labels = [
            symbolic.decide(stream["worlds"]["online"][i - 1])["decision"] for i in b["events"]
        ]
        require(
            b["true_labels"] == labels
            and b["permuted_labels"]
            == diagnosis.permute(labels, base.SEED + plan["rng_seed"] + b["release_ordinal"]),
            "batch_multiset",
        )
    states: dict[str, Json] = {}
    previous: dict[str, Json] = {}
    initial: dict[str, Json] = {}
    for tx in stream["transaction_rows"]:
        arm, state = tx["arm"], tx["state"]
        require(
            set(state) == {"acks", "parameters", "pending", "released", "templates"}, "state_scope"
        )
        prior = previous.get(arm, state)
        require(
            tx["before"] == canonical_hash(prior) and tx["after"] == canonical_hash(state),
            "transaction_chain",
        )
        expected = deepcopy(prior)
        if arm not in initial:
            require(
                not state["acks"]
                and not state["pending"]
                and not state["released"]
                and not state["templates"],
                "initial_label_leak",
            )
            initial[arm] = state
        added = set(state["pending"]) - set(prior["pending"])
        acked = state["acks"][len(prior["acks"]) :]
        if added:
            require(len(added) == 1 and not acked, "prediction_boundary")
            event = next(iter(added))
            sealed = state["pending"][event]
            require(
                sealed["state_hash"] == tx["before"] and int(event[1:]) == sealed["ordinal"],
                "predecessor_join",
            )
            expected["pending"][event] = sealed
        elif acked:
            transition = next(
                t for t in stream["transitions"] if t["after"] == tx["after"] and t["arm"] == arm
            )
            batch = batches[transition["ordinal"]]
            require(acked == [f"e{i}" for i in batch["events"]], "release_ack_join")
            for j, i in enumerate(batch["events"]):
                require(i <= transition["ordinal"] - plan["delays"][i - 1], "future_label_access")
                expected["pending"].pop(f"e{i}")
                if arm != "frozen_query_only":
                    release = transition["release_rows"][j]
                    target = batch[
                        "permuted_labels" if arm == "matched_permuted" else "true_labels"
                    ][j]
                    require(
                        release["target"] == target
                        and release["truth"] == batch["true_labels"][j]
                        and release["event"] == i,
                        "release_label_join",
                    )
                    key = base.topology(stream["worlds"]["online"][i - 1]["query"])
                    if key not in expected["templates"]:
                        expected["templates"].append(key)
            expected["acks"].extend(acked)
            if arm != "frozen_query_only":
                expected["released"].extend(batch["true_labels"])
            if arm.startswith("matched_"):
                require(state["parameters"] == transition["parameters"], "checkpoint_parameters")
                expected["parameters"] = state["parameters"]
        elif state["templates"] != prior["templates"]:
            require(arm == "frozen_query_only" and bool(prior["pending"]), "unmatched_admission")
            i = max(int(e[1:]) for e in prior["pending"])
            key = base.topology(stream["worlds"]["online"][i - 1]["query"])
            require(key not in expected["templates"], "duplicate_admission")
            expected["templates"].append(key)
        require(state == expected, "causal_state_or_retention_leak")
        previous[arm] = state
        states[tx["after"]] = state
    rows = []
    for row in stream["rows"]:
        arm, split, i = row["arm"], row["split"], row["ordinal"]
        sealed = row["sealed"]
        state = states[sealed["state_hash"]]
        value = stream["worlds"][split][i - 1]
        scheduled = plan[split][i - 1]
        require(
            value
            == base.world(
                base.graphs()[split][scheduled["template"]],
                scheduled["unit_id"] + "/",
                scheduled["error_type"],
                scheduled["world_seed"],
            ),
            "world_identity",
        )
        observation = base.observe(value, state["templates"])
        prediction = c.probabilities(state["parameters"], observation["counts"])
        require(
            prediction == sealed["prediction"]
            and observation["counts"] == sealed["counts"]
            and sealed["evidence_hash"] == canonical_hash(value),
            "saved_probability",
        )
        if split == "online":
            require(
                sealed["confidence"] == max(prediction)
                and sealed["typed_decision"] == primitive.LABELS[prediction.index(max(prediction))]
                and sealed["query_digest"] == canonical_hash(value["query"]),
                "typed_prediction_seal",
            )
            require(
                any(
                    t["arm"] == arm and t["state"]["pending"].get(f"e{i}") == sealed
                    for t in stream["transaction_rows"]
                ),
                "durable_prediction",
            )
        truth = symbolic.decide(value)["decision"]
        measured = dict(
            score(prediction, truth, 0.5),
            error=int(primitive.LABELS[prediction.index(max(prediction))] != truth),
        )
        require(
            truth == row["truth"] and all(row[k] == v for k, v in measured.items()), "scored_row"
        )
        if split == "retention":
            checkpoint = row["checkpoint"]
            expected_hash = (
                canonical_hash(initial[arm])
                if checkpoint == 0
                else next(
                    r["state_hash"]
                    for r in stream["restarts"]
                    if r["arm"] == arm
                    and r["ordinal"] == (24 if checkpoint == 24 else max(schedule))
                )
            )
            require(sealed["state_hash"] == expected_hash, "retention_checkpoint")
        rows.append(dict(row, error_type=scheduled["error_type"]))
    require(
        {(r["unit_id"], r["arm"]) for r in rows}
        == {(r["unit_id"], r["arm"]) for r in base.census(dict(cohort_manifest=[plan]), "")},
        "row_identity",
    )
    return rows


def gates(intervals: list[Json]) -> Json:
    """All frozen information and retention bounds must hold together."""
    rows = [
        dict(
            split=r["split"],
            checkpoint=r["checkpoint"],
            endpoint=r["endpoint"],
            comparison=r["comparison"],
            upper=r["upper"],
            threshold=0
            if r["comparison"].endswith("permuted")
            else -0.01
            if r["split"] == "online"
            else 0.01,
            passed=bool(
                r["upper"] < 0
                if r["comparison"].endswith("permuted")
                else r["upper"] <= -0.01
                if r["split"] == "online"
                else r["upper"] <= 0.01
            ),
        )
        for r in intervals
        if not r["comparison"].endswith("query_only")
    ]
    return dict(
        joint_gate_rows=rows,
        label_benefit_score=int(len(rows) == 8 and all(r["passed"] for r in rows)),
    )


def summarize(rows: list[Json]) -> Json:
    """Average inside streams before resampling the fixed nine-endpoint family."""
    groups: dict[tuple[Any, ...], list[Json]] = defaultdict(list)
    for row in rows:
        if row["split"] == "retention" or row["ordinal"] >= 25:
            groups[(row["cluster_id"], row["split"], row["checkpoint"], row["arm"])].append(row)
    streams = sorted({r["cluster_id"] for r in rows})
    rng = np.random.default_rng(base.SEED)
    draws = rng.integers(len(streams), size=(10000, len(streams)))
    specs: list[tuple[str, int | None, str, str]] = [
        ("online", None, "metric", a) for a in (base.ARMS[0], base.ARMS[2], base.ARMS[3])
    ]
    specs += [
        ("retention", step, metric, base.ARMS[0])
        for step in (0, 24, 48)
        for metric in ("metric", "error")
    ]
    paired, intervals = [], []
    for split, checkpoint, metric, other in specs:
        deltas = []
        for cluster in streams:
            means = {
                arm: float(np.mean([r[metric] for r in groups[cluster, split, checkpoint, arm]]))
                for arm in ("matched_true", other)
            }
            delta = means["matched_true"] - means[other]
            deltas.append(delta)
            paired.append(
                dict(
                    cluster_id=cluster,
                    seed=groups[cluster, split, checkpoint, other][0]["seed"],
                    split=split,
                    checkpoint=checkpoint,
                    endpoint="brier" if metric == "metric" else "error",
                    comparison="matched_true_minus_" + other,
                    metric=delta,
                    denominator=len(groups[cluster, split, checkpoint, other]),
                    arm="matched_true",
                    status="complete",
                    missing_reason=None,
                )
            )
        samples = np.mean(np.asarray(deltas)[draws], axis=1)
        tail = 0.01 / (2 * base.GATES["family_intervals"])
        intervals.append(
            dict(
                split=split,
                checkpoint=checkpoint,
                endpoint="brier" if metric == "metric" else "error",
                comparison="matched_true_minus_" + other,
                mean=float(np.mean(deltas)),
                lower=float(np.quantile(samples, tail)),
                upper=float(np.quantile(samples, 1 - tail)),
                paired_stream_means=deltas,
                independent_n=len(streams),
            )
        )
    progress("bootstrap_after", 10000, 0)
    return dict(paired_stream_rows=paired, simultaneous_intervals=intervals, **gates(intervals))


def reduce(source: Json) -> Json:
    """Recompute each original shard and preserve the complete scheduled denominator."""
    require(
        source["protocol"] == base.protocol() and len(source["stream_refs"]) == 24, "frozen_roster"
    )
    began = time.monotonic()
    rows, checks, uses = [], [], []
    fit = load(reference(Path(source["raw_directory"]) / "fit.json"))
    for i, (ref, plan) in enumerate(
        zip(source["stream_refs"], base.protocol()["cohort_manifest"], strict=True)
    ):
        require(time.monotonic() - began < 1800, "reduction_deadline")
        progress("stream_before", i, 24 - i)
        stream = load(ref)
        rebuilt = reconstruct(stream, plan)
        require(
            all(
                tx["state"]["parameters"] == fit["heads"][str(plan["seed"])]
                for tx in stream["transaction_rows"][:4]
            ),
            "initial_checkpoint_identity",
        )
        rows.extend(rebuilt)
        checks.append(
            dict(
                cluster_id=plan["stream_id"],
                check="sealed_states_releases_admissions_retention",
                passed=True,
                rows=len(rebuilt),
            )
        )
        counts = {
            a: sum(
                bool(r["sealed"].get("template_uses"))
                for r in rebuilt
                if r["arm"] == a and r["split"] == "online" and r["ordinal"] >= 25
            )
            for a in base.ARMS
        }
        uses.append(
            dict(
                cluster_id=plan["stream_id"],
                future_denominator=24,
                uses_by_arm=counts,
                true_minus_query_only=counts["matched_true"] - counts["frozen_query_only"],
                interpretation="syntax admission and template use are separate from learned label information",
            )
        )
        progress("stream_after", i + 1, 23 - i)
    indexed = {(r["unit_id"], r["arm"]): r for r in rows}
    require(
        len(indexed) == len(rows) == len(source["rows"])
        and all(
            {k: v for k, v in indexed[r["unit_id"], r["arm"]].items() if k != "error_type"} == r
            for r in source["rows"]
        ),
        "original_row_join",
    )
    rows = [indexed[r["unit_id"], r["arm"]] for r in source["rows"]]
    summary = summarize(rows)
    topologies = {s: {base.topology(q) for q in qs} for s, qs in base.graphs().items()}
    benefit = summary["label_benefit_score"]
    return dict(
        rows=rows,
        **summary,
        leakage_checks=checks,
        template_use_changes=uses,
        topology_support=dict(
            online_topologies=len(topologies["online"]),
            retention_topologies=len(topologies["retention"]),
            disjoint=not bool(topologies["online"] & topologies["retention"]),
            independent_streams=24,
            independent_topology_families=8,
            interpretation="24 paired RNG streams; five seeds and recurring graphs do not increase n",
        ),
        diagnosis=dict(
            frozen_future_brier_headroom=float(
                np.mean(
                    [
                        r["metric"]
                        for r in rows
                        if r["arm"] == "frozen_verified"
                        and r["split"] == "online"
                        and r["ordinal"] >= 25
                    ]
                )
            ),
            identical_true_permuted=all(
                r["metric"] == 0
                for r in summary["paired_stream_rows"]
                if r["comparison"].endswith("permuted")
            ),
            query_only_is_label_learning=False,
        ),
        continuation_decision=dict(
            decision="retain_constructed_label_evidence_only"
            if benefit
            else "retire_unchanged_matched_label_mechanism",
            falsifiable_condition="Any successor must freeze changed features on development data, then pass the same matched-label and untouched-retention gates on new streams; do not rerun this unchanged mechanism.",
        ),
        intended_count=18432,
        completed_count=len(rows),
        failed_count=0,
        censored_count=0,
        excluded_count=0,
        independent_count=24,
    )


def empty(reason: str) -> Json:
    """External absence retains every planned slot and cannot grant a science score."""
    return dict(
        rows=base.census(base.protocol(), reason),
        paired_stream_rows=[],
        simultaneous_intervals=[],
        leakage_checks=[],
        template_use_changes=[],
        topology_support={},
        joint_gate_rows=[],
        continuation_decision=dict(
            decision="await_original_external_evidence",
            falsifiable_condition="Authenticate every original operand before any reduction.",
        ),
        diagnosis={},
        label_benefit_score=0,
        intended_count=18432,
        completed_count=0,
        failed_count=0,
        censored_count=18432,
        excluded_count=0,
        independent_count=0,
    )
