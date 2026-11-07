"""REQ-REPORT-8901: compare complete durable deployment on fresh paired streams.

The old head keeps its frozen schema but does not compute the two extra
coordinates. Each feedback acknowledgment pays its own atomic storage barriers.
"""

from collections import Counter
from copy import deepcopy
from pathlib import Path
import random
from typing import Any

import numpy as np
from scipy.stats import t

from carnot.memory import signed_memory_8892 as signed
from carnot.reporting import matched_adaptation_cost_8888 as base
from carnot.reporting import signed_feature_methods_8892 as features
from carnot.reporting import topology_memory_methods_8881 as old
from carnot.reporting import evidence_set_methods_8768 as symbolic
from carnot.reporting.causal_learning_audit_8810 import require
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash, sha256_file
from carnot.reporting.v746_retained_evidence import reference
from carnot.reporting.v748_collection_contract import namespace

Json = dict[str, Any]
ROOT, SEED = old.ROOT, 7618901
NAME, TASK = "experiment_8901_v761_signed_adaptation_cost", "exp8901-signed-adaptation-cost"
CLI = "scripts/experiments/" + NAME + ".py"
OWNED = [
    "python/carnot/reporting/signed_adaptation_cost_8901.py",
    "python/carnot/reporting/signed_adaptation_runner_8901.py",
    CLI,
]
TESTS = ["tests/python/test_signed_adaptation_cost_8901.py"]


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Keep live counts visible without extending the fixed deadline."""
    print(f"[exp8901] phase={phase} completed={completed} pending={pending}", flush=True)


def roster() -> list[Json]:
    """Freeze independent cost RNGs before any quality or timing outcome is opened."""
    planner = namespace(old, "cost8901_plan")
    planner.SEED = SEED
    rng, rows = random.Random(SEED), []
    for index in range(12):
        plan = planner.plan(48 + index if index < 6 else 60 + index - 6)
        plan["stream_id"] = f"cost8901-{index:02d}"
        for row in plan["online"]:
            row["unit_id"] = "cost8901/" + row["unit_id"]
        cells = [(feature, arm) for feature in ("old", "signed") for arm in old.ARMS]
        rng.shuffle(cells)
        for order, (feature, arm) in enumerate(cells):
            rows.append(
                dict(
                    unit_id=f"{plan['stream_id']}/{feature}/{arm}",
                    cluster_id=plan["stream_id"],
                    seed=plan["seed"],
                    arm=arm,
                    control_arm=arm,
                    feature_arm=feature,
                    plan=plan,
                    order=order,
                    denominator=48,
                )
            )
    return rows


def legacy_observe(payload: Json, templates: list[str]) -> Json:
    """Compute the frozen shared features without refutation fractions or pair overlaps."""
    symbolic.parse(payload)
    local = []
    signs: dict[tuple[str, ...], set[bool]] = {}
    for *triple, sign in payload["facts"]:
        signs.setdefault(tuple(triple), set()).add(sign)
    for atom in payload["query"]:
        positive, negative = [], 0
        for fact in payload["facts"]:
            binding: dict[str, str] = {}
            matched = True
            for term, actual in zip(atom, fact[:3], strict=True):
                matched &= (
                    binding.setdefault(term, actual) == actual
                    if term.startswith("?")
                    else term == actual
                )
            if matched:
                if fact[3]:
                    positive.append(binding)
                else:
                    negative += 1
        local.append((atom, positive, negative))
    local.sort(key=lambda item: (len(item[1]), item[2], tuple(t.startswith("?") for t in item[0])))
    variables = [v for atom in payload["query"] for v in (atom[0], atom[2]) if v.startswith("?")]
    bindings: list[dict[str, str]] = [{}]
    constrained = old.topology(payload["query"]) in templates
    joins = 0
    for _, candidates, _ in local:
        joined = []
        for prior in bindings if constrained else [{}]:
            for candidate in candidates:
                joins += 1
                if all(prior.get(k, v) == v for k, v in candidate.items()):
                    joined.append(dict(prior, **candidate))
        bindings = joined
    conflicts = sum(len(s) > 1 for s in signs.values()) * len(local)
    return dict(
        counts=[
            len(local),
            sum(bool(p) and not conflicts for _, p, _ in local),
            sum(not p for _, p, _ in local),
            sum(n > 1 for n in Counter(variables).values()),
            conflicts,
            len(bindings),
            0.0,
            0.0,
        ],
        primitive_checks=len(local),
        work=dict(
            fact_comparisons=len(local) * len(payload["facts"]),
            sign_record_reads=len(payload["facts"]),
            binding_comparisons=joins,
            candidate_pair_comparisons=0,
            atom_pairs=0,
        ),
    )


def release(
    store: Any, values: list[Json], events: list[int], ordinal: int, arm: str, targets: list[str]
) -> list[Json]:
    """Validate the full permutation, then durably acknowledge each individual feedback."""
    require(
        store.active is None and arm in old.ARMS[:3] and len(values) == len(events) == len(targets),
        "release_boundary",
    )
    labels = []
    for value, event in zip(values, events, strict=True):
        sealed = store.state["pending"].get(f"e{event}")
        require(
            sealed is not None
            and ordinal > sealed["ordinal"]
            and canonical_hash(value) == sealed["evidence_hash"],
            "release_boundary",
        )
        labels.append(str(symbolic.decide(value)["decision"]))
    require(
        Counter(labels) == Counter(targets) and (arm == "matched_permuted" or labels == targets),
        "release_multiset",
    )
    rows = []
    for value, event, label, target in zip(values, events, labels, targets, strict=True):
        state = deepcopy(store.state)
        sealed = state["pending"].pop(f"e{event}")
        gradient = dict(optimizer_steps=0, gradient_norm=0.0, parameter_delta=0.0)
        if arm.startswith("matched_"):
            state["parameters"], gradient = signed.update(
                state["parameters"], sealed["counts"], target
            )
        key = old.topology(value["query"])
        if key not in state["templates"] and len(state["templates"]) < 32:
            state["templates"].append(key)
        state["acks"].append(f"e{event}")
        state["released"].append(label)
        store.commit(state)
        rows.append(
            dict(
                event=event,
                release_ordinal=ordinal,
                truth=label,
                target=target,
                target_unchanged=label == target,
                **gradient,
            )
        )
    return rows


def runtime() -> Any:
    """Reuse complete parent timing and child supervision without changing older consumers."""
    engine = namespace(base, "cost8901")
    engine.segment_original = engine.segment
    engine.CLI, engine.progress, engine.segment = CLI, progress, segment
    return engine


def segment(request: Path, start: int, stop: int | None, *, emit: bool = False) -> Json:
    """Run the existing delayed-label lifecycle with real feature work and individual acks."""
    import json

    spec = json.loads(request.read_bytes())
    methods = signed.engine()
    methods.observe = legacy_observe if spec["feature_arm"] == "old" else signed.observe
    methods.release = release
    inherited_ack = methods.Store.acknowledge

    def acknowledge(store: Any, events: list[int]) -> None:
        for event in events:
            inherited_ack(store, [event])

    methods.Store.acknowledge = acknowledge
    engine = runtime()
    engine.methods = methods
    return dict(engine.__dict__["segment_original"](request, start, stop, emit=emit))


def execute(raw: Path, spec: Json, head: Json) -> Json:
    """Keep inclusive parent costs separate from actual gradient and atomic-update counts."""
    row = dict(runtime().execute(raw, spec, head))
    predictions = [p for s in row["segments"] for p in s["predictions"]]
    counts = row["operation_counts"]
    counts.update(
        {
            key: sum(p["work"][key] for p in predictions)
            for key in (
                "fact_comparisons",
                "sign_record_reads",
                "binding_comparisons",
                "candidate_pair_comparisons",
                "atom_pairs",
            )
        }
    )
    counts.update(
        update_attempts=counts["applied_updates"],
        atomic_feedback_commits=len(
            [
                a
                for s in row["segments"]
                for a in s["audit"]["commits"]
                if a["phase"] in {"source_check", "acknowledgement"}
            ]
        ),
        kernel_parameter_coordinates=27,
        kernel_gradient_calls=counts["applied_updates"],
        extra_feature_queries=48 if spec["feature_arm"] == "signed" else 0,
    )
    atomic_json(raw / "unit.json", row)
    return row


def parity(row: Json) -> bool:
    """Require full label, durability and restart work before a cell contributes to readiness."""
    if row["status"] != "complete":
        return False
    first, last = row["segments"]
    query_only = row["arm"] == "frozen_query_only"
    audits = [s["audit"] for s in row["segments"]]
    commits = [c for a in audits for c in a["commits"]]
    releases = [r for s in row["segments"] for r in s["releases"]]
    return bool(
        row["restart"]["passed"]
        and row["restart"]["exact_state"]
        and first["state"] == last["initial_state"]
        and not first["initial_state"]["templates"]
        and len(first["predictions"] + last["predictions"]) == 48
        and len(set(last["state"]["acks"])) == 48
        and len(last["state"]["templates"]) == len(set(last["state"]["templates"])) == 8
        and not last["state"]["pending"]
        and len(last["state"]["released"]) == len(releases) == (0 if query_only else 48)
        and all(
            sum(s["label_counts"][k] for s in row["segments"]) == (0 if query_only else 48)
            for k in ("acquisition", "source_check")
        )
        and sum(r["optimizer_steps"] for r in releases)
        == (48 if row["arm"].startswith("matched_") else 0)
        and all(s["maximum_pending"] <= 32 for s in row["segments"])
        and sum(a["file_fsync"] for a in audits)
        == sum(a["directory_fsync"] for a in audits)
        == len(commits)
        == (148 if query_only else 100)
        and sum(a["serialized_bytes"] for a in audits) == sum(c["bytes"] for c in commits)
        and row["operation_counts"]["atomic_feedback_commits"] == 48
        and last["tail_drain"]["pending_after"] == 0
        and last["tail_drain"]["release_events"] > 0
        and row["complete_ns"] == row["parent_ended_ns"] - row["parent_started_ns"] > 0
        and all(
            row["parent_started_ns"] <= c["started_ns"] < c["ended_ns"] <= row["parent_ended_ns"]
            and c["exit_code"] == 0
            for c in row["commands"]
        )
    )


def reduce_costs(rows: list[Json], setup_ns: int) -> Json:
    """Compare whole streams using four simultaneous paired 99-percent intervals."""
    lookup = {
        (r["cluster_id"], r["feature_arm"], r["arm"]): r for r in rows if r["status"] == "complete"
    }
    intervals = []
    for arm in old.ARMS:
        pairs = [
            (lookup[c, "old", arm]["complete_ns"], lookup[c, "signed", arm]["complete_ns"])
            for c in sorted({r["cluster_id"] for r in rows})
            if (c, "old", arm) in lookup and (c, "signed", arm) in lookup
        ]
        differences = [b - a for a, b in pairs]
        ratios = [b / a for a, b in pairs]
        n = len(pairs)
        mean = float(np.mean(differences)) if n else None
        margin = (
            float(t.ppf(1 - 0.01 / 8, n - 1) * np.std(differences, ddof=1) / n**0.5)
            if n > 1
            else None
        )
        ratio_mean = float(np.mean(ratios)) if n else None
        ratio_margin = (
            float(t.ppf(1 - 0.01 / 8, n - 1) * np.std(ratios, ddof=1) / n**0.5) if n > 1 else None
        )
        intervals.append(
            dict(
                arm=arm,
                control_feature="old",
                treatment_feature="signed",
                differences_ns=differences,
                complete_cost_ratios=ratios,
                paired_streams=n,
                missing_pairs=12 - n,
                mean_difference_ns=mean,
                lower_ns=mean - margin if mean is not None and margin is not None else None,
                upper_ns=mean + margin if mean is not None and margin is not None else None,
                mean_ratio=ratio_mean,
                lower_ratio=ratio_mean - ratio_margin
                if ratio_mean is not None and ratio_margin is not None
                else None,
                upper_ratio=ratio_mean + ratio_margin
                if ratio_mean is not None and ratio_margin is not None
                else None,
                familywise_confidence=0.99,
                family_size=4,
                method="Bonferroni paired Student-t on independent whole-stream differences and ratios; each endpoint has a four-arm family; Gaussian stream statistics assumed",
            )
        )
    return dict(
        intervals=intervals,
        setup_ns=setup_ns,
        speed_claim=False,
        quality_evidence_refs=[],
        scope="complete host CPU costs; no efficiency-benefit headline",
    )
