"""REQ-REPORT-8881: match released labels while changing actual query graphs.

The existing small linear energy head sees six counts and a bias. Constructed
signed records provide exact oracle truth, not natural-language generalization.
"""

from collections import Counter
from copy import deepcopy
from functools import lru_cache
from itertools import combinations, product
from pathlib import Path
import time
from typing import Any

import numpy as np

from carnot.memory import categorical_memory_8830 as c
from carnot.memory import causal_memory_8802 as primitive
from carnot.reporting import feedback_information_8867 as diagnosis
from carnot.reporting import evidence_set_methods_8768 as symbolic
from carnot.reporting import learning_signal_diagnosis_8815 as learning
from carnot.reporting.causal_learning_audit_8810 import require, score
from carnot.reporting.current_work_receipt import canonical_hash
from carnot.reporting.selective_methods_8829 import topology as graph_key

Json = dict[str, Any]
ROOT = Path(__file__).resolve().parents[3]
NAME, TASK = "experiment_8881_v760_topology_memory_methods", "exp8881-topology-memory-methods"
CLI = "scripts/experiments/" + NAME + ".py"
OWNED = [
    "python/carnot/reporting/topology_memory_methods_8881.py",
    "python/carnot/reporting/topology_memory_runner_8881.py",
    CLI,
]
TESTS = ["tests/python/test_topology_memory_methods_8881.py"]
SEED, SEEDS = 7608881, (11, 23, 37, 47, 59)
ARMS = ("frozen_verified", "matched_true", "matched_permuted", "frozen_query_only")
SPLITS = ("development", "online", "retention")
ERRORS = ("support", "refute", "missing", "binding_mismatch", "conflict", "alternative")
TRUTH = dict(
    zip(ERRORS, ("support", "refute", "insufficient", "insufficient", "insufficient", "support"))
)
GATES: Json = dict(
    future_window=[25, 48],
    retention_checkpoints=[0, 24, 48],
    simultaneous_confidence=0.99,
    bootstrap_draws=10000,
    true_minus_permuted_upper=0,
    true_minus_frozen_upper=-0.01,
    retention_brier_upper=0.01,
    retention_error_upper=0.01,
    resampling_unit="paired_stream",
    family_intervals=9,
)


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Expose actual work so heartbeats cannot conceal a finite deadline."""
    print(f"[exp8881] phase={phase} completed={completed} pending={pending}", flush=True)


def topology(query: list[list[str]]) -> str:
    """Ignore names and order; disconnected marker terms cannot create novelty."""
    require(bool(query) and all(len(q) == 3 for q in query), "query_shape")
    names = sorted({q[i] for q in query for i in (0, 2)})
    edges = [[names.index(q[0]), names.index(q[2])] for q in query]
    reached = {0}
    for _ in names:
        reached |= {v for a, b in edges if a in reached or b in reached for v in (a, b)}
    require(len(reached) == len(names), "connected_query_required")
    return graph_key(edges)


@lru_cache(maxsize=1)
def graphs() -> Json:
    """Enumerate a finite roster before outcomes, with disjoint canonical graph sets."""
    edges = [(a, b) for a in range(4) for b in range(4) if a != b]
    seen, buckets = set(), {}
    for size in range(3, 8):
        bucket = []
        for chosen in combinations(edges, size):
            if (
                len({n for e in chosen for n in e}) != 4
                or sum(0 in e for e in chosen) < 2
                or not any(0 not in e for e in chosen)
            ):
                continue
            q = [
                ["?x" if a == 0 else f"c{a}", f"r{i}", "?x" if b == 0 else f"c{b}"]
                for i, (a, b) in enumerate(chosen)
            ]
            try:
                key = topology(q)
            except ValueError:
                continue
            if key not in seen:
                seen.add(key)
                bucket.append(q)
            if len(bucket) == 6:
                break
        buckets[size] = bucket
    roster = [bucket[i] for i in range(6) for bucket in buckets.values() if i < len(bucket)][:24]
    require(len(roster) == 24, "topology_roster_shortfall")
    return {s: roster[j * 8 : (j + 1) * 8] for j, s in enumerate(SPLITS)}


def plan(index: int, *, pilot: bool = False) -> Json:
    """Freeze balanced labels and existing error strata without choosing on outcomes."""
    rng = np.random.default_rng(SEED + index)
    errors = ["support"] * 8 + ["alternative"] * 8 + ["refute"] * 16
    errors += ["missing"] * 6 + ["binding_mismatch"] * 5 + ["conflict"] * 5
    return dict(
        stream_id=f"v760-stream-{index:02d}",
        seed=SEEDS[index % 5],
        rng_seed=SEED + index,
        pilot=pilot,
        delays=[8] * 48 if index % 24 < 12 else [4, 8, 12] * 16,
        **{
            s: [
                dict(
                    unit_id=f"v760/{index}/{s}/{i}",
                    error_type=str(e),
                    template=i % 8,
                    world_seed=int(rng.integers(2**31)),
                )
                for i, e in enumerate(rng.permutation(errors))
            ]
            for s in ("online", "retention")
        },
    )


def protocol() -> Json:
    """Seal the complete finite census and every estimand before fitting."""
    return dict(
        arm_definitions={
            a: dict(
                training=a.startswith("matched_"),
                admission="syntax_after_prediction"
                if a.endswith("query_only")
                else "verified_release",
            )
            for a in ARMS
        },
        topology_manifest=deepcopy(graphs()),
        cohort_manifest=[plan(i) for i in range(24)],
        pilot_manifest=[plan(24, pilot=True), plan(37, pilot=True)],
        configuration=dict(
            objective="linear",
            optimizer="SGD",
            lr=0.1,
            features=[*primitive.COUNTS, "bias"],
            threshold=0.5,
            pending_cap=32,
            template_cap=32,
            fit_steps=240,
            seeds=list(SEEDS),
        ),
        expected_row_counts=dict(online=4608, retention=13824, total=18432),
        frozen_gates=GATES,
        release_policy="six events; latest configured due time; prediction before release; exact batch permutation including fixed points",
        recurring_templates="eight online templates recur six times per stream and across all24 streams; streams are paired RNG trials, not24 independent topology families",
        retention_policy="untouched read-only queries; checkpoint48 after tail drain",
        finite_roster=dict(
            streams=24,
            development_streams=2,
            fit_heads=5,
            online_per_stream=48,
            retention_per_stream=48,
            arms=4,
            separable_controls=1,
            zero_signal_controls=1,
        ),
        no_heldout_selection=True,
    )


def world(query: list[list[str]], prefix: str, error: str, seed: int) -> Json:
    """Use relevant alternative bindings, never disconnected markers, for fresh worlds."""
    rng = np.random.default_rng(seed)
    q = [
        [s if s.startswith("?") else prefix + s, r, o if o.startswith("?") else prefix + o]
        for s, r, o in query
    ]
    facts = [
        [prefix + "a" if s == "?x" else s, r, prefix + "a" if o == "?x" else o, True]
        for s, r, o in q
    ]
    variable = [i for i, t in enumerate(q) if "?x" in t]
    ground = next(i for i, t in enumerate(q) if "?x" not in t)
    for i in variable:
        s, r, o = q[i]
        for j in range(int(rng.integers(1, 4))):
            facts.append(
                [
                    prefix + f"d{i}-{j}" if s == "?x" else s,
                    r,
                    prefix + f"d{i}-{j}" if o == "?x" else o,
                    True,
                ]
            )
    if error == "refute":
        facts[ground][3] = False
    elif error == "missing":
        facts.pop(ground)
    elif error == "binding_mismatch":
        i = variable[0]
        facts[i][0 if q[i][0] == "?x" else 2] = prefix + "mismatch"
    elif error == "conflict":
        facts.append([*facts[ground][:3], False])
    elif error == "alternative":
        facts += [
            [prefix + "other" if s == "?x" else s, r, prefix + "other" if o == "?x" else o, True]
            for s, r, o in q
        ]
    return dict(query=q, facts=facts)


def observe(value: Json, templates: list[str]) -> Json:
    """Generalize the original five checks without exposing a whole-query label."""
    symbolic.parse(value)
    checks = [symbolic.decide(dict(facts=value["facts"], query=[q])) for q in value["query"]]
    bindings: list[Json] = [{}]
    constrained = topology(value["query"]) in templates
    for s, r, o in value["query"]:
        candidates = []
        for binding, (fs, fr, fo, sign) in product(
            bindings if constrained else [{}], value["facts"]
        ):
            candidate = dict(binding)
            if (
                sign
                and r == fr
                and all(
                    candidate.setdefault(term, actual) == actual
                    if term.startswith("?")
                    else term == actual
                    for term, actual in [(s, fs), (o, fo)]
                )
            ):
                candidates.append(candidate)
        bindings = candidates
    satisfied = sum(check["decision"] == "support" for check in checks)
    conflicts = sum(len(check["conflicts"]) for check in checks)
    variables = [t for q in value["query"] for t in (q[0], q[2]) if t.startswith("?")]
    counts = [
        len(checks),
        satisfied,
        sum(bool(check["missing_premises"]) for check in checks),
        len({v for v in variables if variables.count(v) > 1}),
        conflicts,
        len(bindings),
    ]
    decision = (
        "insufficient"
        if conflicts
        else "support"
        if satisfied == len(checks) and bindings
        else "refute"
        if any(c["decision"] == "refute" for c in checks)
        else "insufficient"
    )
    return dict(counts=counts, decision=decision, primitive_checks=len(checks))


class Store(c.Store):
    """Reuse the original atomic store and keep truth outside prediction and admission."""

    state: Json

    def predict(self, event: str, payload: Json, threshold: float) -> Json:
        """Seal before feedback; every serving read uses an immutable predecessor."""
        if self.active is not None:
            raise RuntimeError("active query")
        require(event not in self.state["pending"] and event not in self.state["acks"], "duplicate")
        require(len(self.state["pending"]) < 32, "pending capacity")
        snapshot = deepcopy(self.state)
        observation = observe(payload, snapshot["templates"])
        sealed = dict(
            observation,
            ordinal=int(event[1:]),
            evidence_hash=canonical_hash(payload),
            prediction=c.probabilities(snapshot["parameters"], observation["counts"]),
            state_hash=canonical_hash(snapshot),
            template_active=topology(payload["query"]) in snapshot["templates"],
        )
        state = deepcopy(snapshot)
        state["pending"][event] = sealed
        self.commit(state)
        self.active = snapshot
        return deepcopy(sealed)

    def admit(self, query: list[list[str]]) -> None:
        """The syntax-only interface cannot receive facts, labels or witnesses."""
        require(self.active is None, "active admission")
        state = deepcopy(self.state)
        key = topology(query)
        if key not in state["templates"] and len(state["templates"]) < 32:
            state["templates"].append(key)
        self.commit(state)

    def acknowledge(self, events: list[int]) -> None:
        """Query-only state drains pending events without receiving any labels."""
        state = deepcopy(self.state)
        for i in events:
            state["pending"].pop(f"e{i}")
            state["acks"].append(f"e{i}")
        self.commit(state)


def release(
    store: Store, values: list[Json], events: list[int], ordinal: int, arm: str, targets: list[str]
) -> list[Json]:
    """Validate every original label before an atomic complete-batch state-plus-ack commit."""
    require(
        store.active is None and arm in ARMS[:3] and len(values) == len(events) == len(targets),
        "release_boundary",
    )
    labels = []
    for value, i in zip(values, events, strict=True):
        sealed = store.state["pending"].get(f"e{i}")
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
    state, rows = deepcopy(store.state), []
    for value, i, label, target in zip(values, events, labels, targets, strict=True):
        sealed = state["pending"].pop(f"e{i}")
        gradient = dict(optimizer_steps=0, gradient_norm=0.0, parameter_delta=0.0)
        if arm.startswith("matched_"):
            state["parameters"], gradient = c.update(state["parameters"], sealed["counts"], target)
        key = topology(value["query"])
        if key not in state["templates"] and len(state["templates"]) < 32:
            state["templates"].append(key)
        state["acks"].append(f"e{i}")
        state["released"].append(label)
        rows.append(
            dict(
                event=i,
                release_ordinal=ordinal,
                truth=label,
                target=target,
                target_unchanged=label == target,
                **gradient,
            )
        )
    store.commit(state)
    return rows


def fit(p: Json) -> Json:
    """Fit fixed steps on development worlds only; validation cannot select a head."""
    rows = []
    for j, q in enumerate(p["topology_manifest"]["development"]):
        for error in ERRORS:
            value = world(q, f"fit/{j}/{error}/", error, SEED + j)
            rows.append(
                dict(
                    input=value,
                    truth=str(symbolic.decide(value)["decision"]),
                    counts=observe(value, [topology(q)])["counts"],
                )
            )
    heads: Json = {}
    fits: list[Json] = []
    for seed in SEEDS:
        progress("fit_before", len(fits), 5 - len(fits))
        fitted = learning.fit(rows, rows, seed, "linear", 0.1, steps=240)
        heads[str(seed)] = dict(objective="linear", lr=0.1, weights=fitted["parameters"])
        fits.append(fitted)
        progress("fit_after", len(fits), 5 - len(fits))
    return dict(heads=heads, fits=fits, worlds=rows, numerical_training_calls=1200)


def census(p: Json, reason: str) -> list[Json]:
    """Preserve every scheduled denominator if an external gate or owned control fails."""
    return [
        dict(
            unit_id=r["unit_id"] + ("" if step is None else f"/checkpoint{step}"),
            cluster_id=s["stream_id"],
            seed=s["seed"],
            arm=arm,
            split=split,
            ordinal=i + 1,
            checkpoint=step,
            metric=None,
            error=None,
            denominator=1,
            status="censored",
            missing_reason=reason,
        )
        for s in p["cohort_manifest"]
        for arm in ARMS
        for split, steps in (("online", [None]), ("retention", [0, 24, 48]))
        for step in steps
        for i, r in enumerate(s[split])
    ]


def run_stream(raw: Path, s: Json, heads: Json, p: Json) -> Json:
    """Pair the same worlds and releases, sealing predictions before any batch label opens."""
    stores = {a: Store(raw / (a + ".json"), deepcopy(heads[str(s["seed"])])) for a in ARMS}
    worlds = {
        split: [
            world(
                p["topology_manifest"]["development" if s["pilot"] else split][r["template"]],
                r["unit_id"] + "/",
                r["error_type"],
                r["world_seed"],
            )
            for r in s[split]
        ]
        for split in ("online", "retention")
    }
    rows, batches, transitions, restarts = [], [], [], []
    predictions: dict[tuple[str, int], Json] = {}
    schedule = {
        max(i + s["delays"][i - 1] for i in range(end - 5, end + 1)): list(range(end - 5, end + 1))
        for end in range(6, 49, 6)
    }

    def scored(
        arm: str, split: str, i: int, sealed: Json, truth: str, checkpoint: int | None
    ) -> Json:
        return dict(
            unit_id=s[split][i - 1]["unit_id"]
            + ("" if checkpoint is None else f"/checkpoint{checkpoint}"),
            cluster_id=s["stream_id"],
            seed=s["seed"],
            arm=arm,
            split=split,
            ordinal=i,
            checkpoint=checkpoint,
            truth=truth,
            sealed=sealed,
            **score(sealed["prediction"], truth, 0.5),
            error=int(primitive.LABELS[int(np.argmax(sealed["prediction"]))] != truth),
            denominator=1,
            status="complete",
            missing_reason=None,
        )

    def retention(checkpoint: int) -> None:
        for arm, store in stores.items():
            before = canonical_hash(store.state)
            for i, value in enumerate(worlds["retention"], 1):
                counts = observe(value, store.state["templates"])["counts"]
                sealed = dict(
                    counts=counts,
                    prediction=c.probabilities(store.state["parameters"], counts),
                    evidence_hash=canonical_hash(value),
                    state_hash=before,
                )
                truth = str(symbolic.decide(value)["decision"])
                rows.append(scored(arm, "retention", i, sealed, truth, checkpoint))
            require(before == canonical_hash(store.state), "retention_mutation")

    retention(0)
    for ordinal in range(1, max(schedule) + 1):
        if ordinal <= 48:
            value = worlds["online"][ordinal - 1]
            for arm, store in stores.items():
                predictions[arm, ordinal] = store.predict(f"e{ordinal}", value, 0.5)
                store.close()
                if arm == "frozen_query_only":
                    store.admit(value["query"])
        if ordinal in schedule:
            events = schedule[ordinal]
            values = [worlds["online"][i - 1] for i in events]
            labels = [str(symbolic.decide(v)["decision"]) for v in values]
            shuffled = diagnosis.permute(labels, SEED + s["rng_seed"] + ordinal)
            batches.append(
                dict(
                    events=events,
                    release_ordinal=ordinal,
                    true_labels=labels,
                    permuted_labels=shuffled,
                    unchanged_labels=sum(a == b for a, b in zip(labels, shuffled, strict=True)),
                    multiset_preserved=Counter(labels) == Counter(shuffled),
                )
            )
            for arm, store in stores.items():
                before = canonical_hash(store.state)
                releases = []
                if arm == "frozen_query_only":
                    store.acknowledge(events)
                else:
                    releases = release(
                        store,
                        values,
                        events,
                        ordinal,
                        arm,
                        shuffled if arm == "matched_permuted" else labels,
                    )
                transitions.append(
                    dict(
                        arm=arm,
                        ordinal=ordinal,
                        before=before,
                        after=canonical_hash(store.state),
                        release_rows=releases,
                        templates=list(store.state["templates"]),
                        parameters=deepcopy(store.state["parameters"]),
                    )
                )
                for i, truth in zip(events, labels, strict=True):
                    row = scored(arm, "online", i, predictions[arm, i], truth, None)
                    row["release_ordinal"] = ordinal
                    rows.append(row)
        if ordinal == 24 or ordinal == max(schedule):
            for arm, store in list(stores.items()):
                reopened = Store(store.path, {})
                restarts.append(
                    dict(
                        arm=arm,
                        ordinal=ordinal,
                        passed=reopened.state == store.state,
                        state_hash=canonical_hash(store.state),
                    )
                )
                stores[arm] = reopened
            retention(24 if ordinal == 24 else 48)
            progress("stream_" + s["stream_id"], min(ordinal, 48), max(0, 48 - ordinal))
    controls = [
        dict(
            check="tail_drained",
            passed=all(
                not x.state["pending"] and len(x.state["acks"]) == 48 for x in stores.values()
            ),
        ),
        dict(check="exact_restart", passed=all(r["passed"] for r in restarts)),
        dict(
            check="matched_counts_and_admissions",
            passed=all(
                predictions["matched_true", i]["counts"]
                == predictions["matched_permuted", i]["counts"]
                == predictions["frozen_verified", i]["counts"]
                for i in range(1, 49)
            ),
        ),
        dict(check="multisets", passed=all(b["multiset_preserved"] for b in batches)),
    ]
    return dict(
        stream=s,
        rows=rows,
        batches=batches,
        transitions=transitions,
        restarts=restarts,
        controls=controls,
        worlds=worlds,
        final_states={a: v.state for a, v in stores.items()},
        numerical_training_calls=96,
    )


def sensitivity() -> tuple[list[Json], int]:
    """Reuse original positive and zero-signal controls without selecting on future streams."""
    return diagnosis.sensitivity()


def summarize(rows: list[Json]) -> Json:
    """Bootstrap paired stream means with one frozen simultaneous family, never seed counts."""
    groups = sorted({r["cluster_id"] for r in rows})
    rng = np.random.default_rng(SEED)
    draws = rng.integers(len(groups), size=(10000, len(groups)))
    intervals: list[Json] = []
    conditions: list[tuple[str, list[int | None], list[str], tuple[str, ...]]] = [
        ("online", [None], ["metric"], ARMS[0:1] + ARMS[2:]),
        ("retention", [0, 24, 48], ["metric", "error"], ARMS[:1]),
    ]
    for split, checkpoints, metrics, others in conditions:
        for checkpoint, metric, other in product(checkpoints, metrics, others):
            paired = []
            for group in groups:
                selected = [
                    r
                    for r in rows
                    if r["cluster_id"] == group
                    and r["split"] == split
                    and r["checkpoint"] == checkpoint
                    and (split != "online" or r["ordinal"] >= 25)
                ]
                means = {
                    a: float(np.mean([r[metric] for r in selected if r["arm"] == a]))
                    for a in ("matched_true", other)
                }
                paired.append(means["matched_true"] - means[other])
            sampled = np.mean(np.asarray(paired)[draws], axis=1)
            tail = 0.01 / (2 * GATES["family_intervals"])
            intervals.append(
                dict(
                    split=split,
                    checkpoint=checkpoint,
                    endpoint="brier" if metric == "metric" else "error",
                    comparison="matched_true_minus_" + other,
                    mean=float(np.mean(paired)),
                    lower=float(np.quantile(sampled, tail)),
                    upper=float(np.quantile(sampled, 1 - tail)),
                    paired_stream_means=paired,
                    independent_n=len(groups),
                )
            )
    benefit = all(
        r["upper"] < 0
        if r["comparison"].endswith("permuted")
        else r["upper"] <= -0.01
        if r["split"] == "online"
        else r["upper"] <= 0.01
        for r in intervals
        if not r["comparison"].endswith("query_only")
    )
    return dict(
        intervals=intervals,
        true_label_benefit=benefit,
        query_only_interpretation="syntax admission utility is not label learning",
    )
