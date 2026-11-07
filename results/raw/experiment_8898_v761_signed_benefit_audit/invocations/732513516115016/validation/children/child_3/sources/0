"""REQ-REPORT-8898: audit signed features from sealed evidence without training.

Stream means preserve the registered experimental unit. Constructed truth can
qualify this finite mechanism, but cannot establish natural generalization.
"""

from collections import defaultdict
from copy import deepcopy
from pathlib import Path
import time
from typing import Any

import numpy as np

from carnot.memory import signed_memory_8892 as signed
from carnot.memory import categorical_memory_8830 as categorical
from carnot.memory import causal_memory_8802 as primitive
from carnot.reporting import label_benefit_audit_8886 as inherited
from carnot.reporting import signed_causal_streams_8895 as causal
from carnot.reporting import signed_feature_methods_8892 as methods
from carnot.reporting import signed_feature_runner_8892 as inputs
from carnot.reporting.current_work_receipt import canonical_hash
from carnot.reporting.causal_learning_audit_8810 import score
from carnot.reporting.v746_retained_evidence import reference, load
from carnot.reporting.v748_collection_contract import namespace

__all__ = ["methods"]

Json = dict[str, Any]
ROOT = methods.ROOT
NAME, TASK = "experiment_8898_v761_signed_benefit_audit", "exp8898-signed-benefit-audit"
SOURCE, METHOD = causal.NAME, methods.NAME
PIN = "sha256:c12240bc5ad66355eb7d581649e2c737b6b1ae2b7c1700d687500653eac0b7d0"
METHOD_PIN = causal.PIN
CLI = "scripts/experiments/" + NAME + ".py"
OWNED = [
    "python/carnot/reporting/signed_benefit_audit_8898.py",
    "python/carnot/reporting/signed_benefit_runner_8898.py",
    CLI,
]
TESTS = ["tests/python/test_signed_benefit_audit_8898.py"]
HELPERS = inherited.OWNED[:2] + causal.OWNED[:2] + methods.OWNED[:3]
require = inherited.require
base = signed.engine()
base.SEED, base.protocol = methods.SEED, methods.protocol
GATES = dict(
    base.GATES,
    family_intervals=27,
    attribution="signed_minus_old_matched_true_future_brier_upper<0",
    diagnostic_slot_split=2,
)


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Flush finite work counts so supervision can detect a stalled child."""
    print(f"[exp8898] phase={phase} completed={completed} pending={pending}", flush=True)


def preflight(raw: Path, sources: Path) -> Json:
    """Reuse byte-pinned custody and authority checks before interpreting any rows."""
    checker = namespace(inputs, "signed8898_inputs")
    checker.m = __import__(__name__, fromlist=["*"])
    checker.UPSTREAM = {
        SOURCE: (PIN, "causal_rows_ready_score", 1),
        METHOD: (METHOD_PIN, "signed_methods_ready_score", 1),
    }
    progress("authenticate_before", 0, 2)
    pre = dict(checker.preconditions(raw, sources))
    if pre["external_ready"]:
        for name in (SOURCE, METHOD):
            value = load(reference(sources / (name + ".json")))
            accepted = (
                value["frozen_protocol"] == methods.protocol()
                and value["owned_statement_coverage"]["complete"]
                and value["owned_statement_coverage"]["percent"] == 100
                and bool(value["control_rows"])
                and all(r["passed"] for r in value["control_rows"])
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
        for name in (SOURCE, METHOD):
            (raw / (name + ".json")).write_bytes((sources / (name + ".json")).read_bytes())
    if pre["external_ready"] and pre["resources"]["passed"]:
        original = load(reference(raw / (SOURCE + ".json")))
        progress("pilot_before", 0, 1)
        began = time.monotonic()
        shard = load(original["stream_refs"][0])
        for arm in shard["arms"]:
            reconstruct(arm, methods.protocol()["cohort_manifest"][0])
        projected = (time.monotonic() - began) * 24 + 1200
        pre["pilot"] = dict(projected_total_s=projected, limit_s=4200)
        pre["resources"]["passed"] = projected < 4200
        progress("pilot_after", 1, 0)
    progress("authenticate_after", 2, 0)
    return pre


def reconstruct(stream: Json, plan: Json) -> list[Json]:
    """Keep the prior independent join, substituting only the frozen signed observation."""
    checker = namespace(inherited, "signed8898_reconstruct")
    checker.base, checker.c = base, signed
    rows = list(checker.reconstruct(stream, plan))
    require(
        len(stream["gradient_rows"]) == 96
        and all(r["optimizer_steps"] == 1 for r in stream["gradient_rows"]),
        "optimizer_work",
    )
    require(len({(r["unit_id"], r["arm"]) for r in rows}) == len(rows), "duplicate_unit")
    return rows


def gates(intervals: list[Json]) -> Json:
    """Learning and feature attribution have distinct conjunctions and strict bounds."""
    operands: list[Json] = []
    scores: Json = {}
    for feature in ("old", "signed"):
        selected = [r for r in intervals if r["feature_arm"] == feature]
        result = inherited.gates(selected)
        operands.extend(dict(r, feature_arm=feature) for r in result["joint_gate_rows"])
        scores[feature] = result["label_benefit_score"]
    contrast = next(r for r in intervals if r["feature_arm"] == "direct_signed_minus_old")
    attributed = bool(contrast["upper"] < 0)
    operands.append(
        dict(
            feature_arm="direct_signed_minus_old",
            split="online",
            checkpoint=None,
            endpoint="brier",
            comparison=contrast["comparison"],
            upper=contrast["upper"],
            threshold=0,
            passed=attributed,
        )
    )
    return dict(
        joint_gate_rows=operands,
        label_benefit_score=scores["signed"],
        old_feature_label_benefit_score=scores["old"],
        feature_specific_benefit_score=int(scores["signed"] and attributed),
    )


def summarize(rows: list[Json]) -> Json:
    """Resample paired stream means, retaining the frozen family and a stricter extra slot."""
    summaries = {}
    for feature in ("old", "signed"):
        checker = namespace(inherited, "signed8898_summary_" + feature)
        checker.base = namespace(base, "signed8898_summary_base")
        checker.base.GATES = GATES
        checker.base.SEED = methods.SEED
        summaries[feature] = checker.summarize(
            [dict(r, arm=r["arm"].split(":")[1]) for r in rows if r["feature_arm"] == feature]
        )
    intervals: list[Json] = []
    paired: list[Json] = []
    for feature, summary in summaries.items():
        intervals.extend(dict(r, feature_arm=feature) for r in summary["simultaneous_intervals"])
        paired.extend(
            dict(r, feature_arm=feature, unit_id=f"{feature}/{i}")
            for i, r in enumerate(summary["paired_stream_rows"])
        )
    draws = np.random.default_rng(methods.SEED).integers(24, size=(10000, 24))
    old_intervals = summaries["old"]["simultaneous_intervals"]
    signed_intervals = summaries["signed"]["simultaneous_intervals"]
    contrasts = [
        dict(
            r,
            feature_arm="signed_minus_old",
            paired_stream_means=(
                np.asarray(r["paired_stream_means"]) - np.asarray(o["paired_stream_means"])
            ).tolist(),
        )
        for r, o in zip(signed_intervals, old_intervals, strict=True)
    ]
    clusters = sorted({r["cluster_id"] for r in rows})
    future: dict[tuple[str, str], list[float]] = defaultdict(list)
    for r in rows:
        if r["split"] == "online" and r["ordinal"] >= 25 and r["arm"].endswith(":matched_true"):
            future[r["cluster_id"], r["feature_arm"]].append(r["metric"])
    direct = [float(np.mean(future[c, "signed"]) - np.mean(future[c, "old"])) for c in clusters]
    contrasts.append(
        dict(
            split="online",
            checkpoint=None,
            endpoint="brier",
            comparison="signed_minus_old_matched_true",
            feature_arm="direct_signed_minus_old",
            paired_stream_means=direct,
            independent_n=24,
        )
    )
    for r in contrasts:
        split_slot = r["feature_arm"] == "direct_signed_minus_old" or (
            r["split"] == "online" and r["comparison"].endswith("query_only")
        )
        tail = methods.protocol()["multiplicity"]["per_tail"] / (2 if split_slot else 1)
        deltas = np.asarray(r["paired_stream_means"])
        samples = np.mean(deltas[draws], axis=1)
        r.update(
            mean=float(np.mean(deltas)),
            lower=float(np.quantile(samples, tail)),
            upper=float(np.quantile(samples, 1 - tail)),
            per_tail=tail,
        )
        intervals.append(r)
        paired.extend(
            dict(
                cluster_id=c,
                unit_id=f"{r['feature_arm']}/{len(paired) + i}",
                seed=next(x["seed"] for x in rows if x["cluster_id"] == c),
                split=r["split"],
                checkpoint=r["checkpoint"],
                endpoint=r["endpoint"],
                comparison=r["comparison"],
                feature_arm=r["feature_arm"],
                metric=float(delta),
                denominator=24 if r["split"] == "online" else 48,
                arm="matched_true",
                status="complete",
                missing_reason=None,
            )
            for i, (c, delta) in enumerate(zip(clusters, deltas, strict=True))
        )
    return dict(simultaneous_intervals=intervals, paired_stream_rows=paired, **gates(intervals))


def controls(source: Json) -> list[Json]:
    """Recompute constant-feature sensitivity losses from retained heads without SGD."""
    value = load(reference(Path(source["raw_directory"]) / "sensitivity.json"))
    checks = []
    for r in value["rows"]:
        signal = r["control"] == "separable_signal"
        metrics = {}
        for arm, head in r["heads"].items():
            metrics[arm] = float(
                np.mean(
                    [
                        score(
                            categorical.probabilities(
                                head,
                                [
                                    5,
                                    5 * int(signal and i == 0),
                                    5 * int(signal and i == 1),
                                    0,
                                    0,
                                    5 * int(signal and i == 2),
                                ],
                            ),
                            label,
                            0.5,
                        )["metric"]
                        for i, label in enumerate(primitive.LABELS)
                    ]
                )
            )
        gap = metrics["permuted"] - metrics["true"]
        passed = gap >= 0.1 if signal else abs(gap) <= 0.03
        require(
            metrics["true"] == r["true_brier"]
            and metrics["permuted"] == r["permuted_brier"]
            and gap == r["gap"]
            and passed == r["passed"]
            and passed,
            "signal_control",
        )
        checks.append(
            dict(
                check=r["control"],
                passed=passed,
                recomputed_gap=gap,
                feature_scope="legacy diagnostic heads; signed evidence audited separately",
            )
        )
    require({r["check"] for r in checks} == {"separable_signal", "zero_signal"}, "control_census")
    return checks


def reduce(source: Json) -> Json:
    """Independently reduce every scheduled feature arm and retain adverse strata."""
    p = methods.protocol()
    require(source["frozen_protocol"] == p and len(source["stream_refs"]) == 24, "frozen_roster")
    fit = load(source["preconditions_checked"]["fitted_heads"])
    rows: list[Json] = []
    checks = controls(source)
    began = time.monotonic()
    for i, (ref, plan) in enumerate(zip(source["stream_refs"], p["cohort_manifest"], strict=True)):
        require(time.monotonic() - began < 1800, "reduction_deadline")
        progress("stream_before", i, 24 - i)
        stream = load(ref)
        causal.verify(stream)
        require(causal.assemble(stream["arms"]) == stream, "assembled_operand_drift")
        for arm in stream["arms"]:
            feature = arm["feature_arm"]
            rebuilt = reconstruct(arm, plan)
            require(
                all(
                    tx["state"]["parameters"] == fit["heads"][feature][str(plan["seed"])]
                    for tx in arm["transaction_rows"][:4]
                ),
                "initial_checkpoint_identity",
            )
            rows.extend(
                dict(
                    r,
                    arm=feature + ":" + r["arm"],
                    feature_arm=feature,
                    topology_family=plan[r["split"]][r["ordinal"] - 1]["template"],
                )
                for r in rebuilt
            )
        checks.append(
            dict(
                cluster_id=plan["stream_id"], check="independent_causal_reconstruction", passed=True
            )
        )
        progress("stream_after", i + 1, 23 - i)
    strip = {"error_type", "topology_family"}
    indexed = {(r["unit_id"], r["arm"]): r for r in rows}
    require(
        len(indexed) == len(rows) == len(source["rows"]) == 36864
        and all(
            {k: v for k, v in indexed[r["unit_id"], r["arm"]].items() if k not in strip} == r
            for r in source["rows"]
        ),
        "original_row_join",
    )
    rows = [indexed[r["unit_id"], r["arm"]] for r in source["rows"]]
    summary = summarize(rows)
    strata = []
    for feature in ("old", "signed"):
        for topology in range(8):
            means = {
                arm: float(
                    np.mean(
                        [
                            r["metric"]
                            for r in rows
                            if r["feature_arm"] == feature
                            and r["arm"] == feature + ":" + arm
                            and r["split"] == "online"
                            and r["ordinal"] >= 25
                            and r["topology_family"] == topology
                        ]
                    )
                )
                for arm in base.ARMS
            }
            strata.append(
                dict(
                    feature_arm=feature,
                    topology_family=topology,
                    future_brier=means,
                    independent_streams=24,
                    status="complete",
                    interpretation="descriptive stratum; no extra benefit claim",
                )
            )
    failed = next(
        (r for r in summary["joint_gate_rows"] if r["feature_arm"] == "signed" and not r["passed"]),
        None,
    )
    return dict(
        rows=rows,
        **summary,
        leakage_checks=checks,
        control_rows=checks[:2],
        topology_support=dict(online_topologies=8, independent_streams=24, strata=strata),
        continuation_decision=dict(
            decision="retire_signed_local_v1_matched_label_mechanism"
            if failed
            else "retain_constructed_label_evidence_only",
            failing_gate=failed,
            continuation="Freeze a development-only update-admission rule that improves future Brier over the frozen signed head by at least0.01; qualify it on development data before any new streams."
            if failed
            else "Seek independent non-oracle evidence.",
            untested=["corpus size", "source fidelity", "alternative architectures"],
            unchanged_streams_permitted=False,
        ),
        intended_count=36864,
        completed_count=len(rows),
        failed_count=0,
        censored_count=0,
        excluded_count=0,
        independent_count=24,
    )


def empty(reason: str) -> Json:
    """Keep every scheduled slot when original evidence cannot support a reduction."""
    return dict(
        rows=causal.census(reason),
        paired_stream_rows=[],
        simultaneous_intervals=[],
        leakage_checks=[],
        control_rows=[],
        topology_support={},
        joint_gate_rows=[],
        continuation_decision=dict(
            decision="await_original_external_evidence", missing_reason=reason
        ),
        label_benefit_score=0,
        old_feature_label_benefit_score=0,
        feature_specific_benefit_score=0,
        intended_count=36864,
        completed_count=0,
        failed_count=0,
        censored_count=36864,
        excluded_count=0,
        independent_count=0,
    )
