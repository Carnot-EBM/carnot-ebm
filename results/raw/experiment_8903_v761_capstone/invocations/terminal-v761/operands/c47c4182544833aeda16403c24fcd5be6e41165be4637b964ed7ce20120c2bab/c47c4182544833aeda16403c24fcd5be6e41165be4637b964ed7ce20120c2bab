"""REQ-REPORT-8854: isolate numerical calibration, template admission and delay.

These constructed worlds have exact symbolic labels. Method qualification does
not establish benefit on the frozen future streams or on natural language.
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any
from unittest.mock import patch

import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree
import numpy as np

from carnot.memory import categorical_memory_8830 as c
from carnot.reporting import causal_streams_8843 as prior
from carnot.reporting.causal_learning_audit_8810 import require, score
from carnot.reporting.current_work_receipt import canonical_hash

Json = dict[str, Any]
ARMS = {
    "linear_frozen": dict(calibration=False, additions=False, diagnostic=False),
    "linear_calibration_only": dict(calibration=True, additions=False, diagnostic=False),
    "linear_additions_only": dict(calibration=False, additions=True, diagnostic=False),
    "linear_combined": dict(calibration=True, additions=True, diagnostic=False),
    "linear_shuffled_combined": dict(calibration=True, additions=True, diagnostic=False),
    "linear_immediate_combined": dict(calibration=True, additions=True, diagnostic=True),
}
GATES = dict(
    future_gain=0.01,
    addition_gain=0.005,
    shuffled_gain_lower_bound=0,
    retention_upper_increase=0.01,
    error_upper_increase=0.01,
    simultaneous_confidence=0.99,
    resampling_unit="paired_stream_cluster",
    bootstrap_draws=10000,
)


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Flush actual boundaries so durable loops cannot resemble a stalled run."""
    print(f"[exp8854] phase={phase} completed={completed} pending={pending}", flush=True)


def stream(index: int) -> Json:
    """Assign balanced error strata and seeds before opening any outcomes."""
    rng = np.random.default_rng(8854000 + index)
    recipes = []
    for split in ("online", "retention"):
        errors = rng.permutation(list(prior.ERRORS) * 8)
        for j, error in enumerate(errors):
            marker = index * 96 + len(recipes)
            recipes.append(
                dict(
                    unit_id=f"v758-stream-{index:02d}/{split}-{j:02d}",
                    marker=marker,
                    structure_id=canonical_hash(["seven-query-terms-twelve-branches", marker]),
                    error_type=str(error),
                )
            )
    return dict(
        stream_id=f"v758-stream-{index:02d}",
        seed=c.d.SEEDS[index % 5],
        rng_seed=8854000 + index,
        online=recipes[:48],
        retention=recipes[48:],
        delays=[8] * 48 if index < 12 or index == 24 else [4, 8, 12] * 16,
    )


def payload(recipe: Json) -> Json:
    """Seven query terms and twelve anchored branches survive entity renaming."""
    value = prior.payload(recipe)
    value["query"][3][0] = "?y"
    prefix = recipe["unit_id"] + ":"
    nodes = [prefix + "root"] + [prefix + f"b11-{k}" for k in range(12)]
    value["facts"] += [[a, "marker", b, True] for a, b in zip(nodes, nodes[1:])]
    if recipe["marker"] & (1 << 11):
        value["facts"].append([nodes[-1], "marker", nodes[-1], True])
    return value


def protocol() -> Json:
    """Freeze the complete later execution census without measuring its outcomes."""
    return dict(
        arm_definitions=ARMS,
        cohort_manifest=[stream(i) for i in range(24)],
        pilot_manifest=[stream(24), stream(25)],
        future_window=[25, 48],
        retention_checkpoints=[0, 24, 48],
        error_strata=list(prior.ERRORS),
        configuration=dict(
            objective="linear",
            optimizer="SGD",
            lr=0.1,
            features=list(c.m.COUNTS),
            candidate_bits=3,
            pending_cap=32,
            template_cap=32,
            threshold=0.5,
        ),
        expected_row_counts=dict(online=6912, retention=20736, total=27648),
        acceptance_gates=GATES,
        movement_accounting="sum gradient norms and L2 parameter steps per [1,24] and [25,48]; template admissions counted separately",
        retention_policy="read-only evaluation; labels never enter updates or selection",
        shuffle_policy="prefix[(feedback_ordinal*7)%len(prefix)] including current verified release",
        checkpoint_policy="24 after due feedback;48 after draining all feedback",
        generator="payload: five predicates, seven terms, twelve distinct branch lengths with marker self-loops",
    )


def census(frozen: Json, reason: str) -> list[Json]:
    """Keep all scheduled slots visible while a later producer owns execution."""
    return [
        dict(
            unit_id=w["unit_id"] + (f":step{step}" if step is not None else ""),
            cluster_id=s["stream_id"],
            seed=s["seed"],
            arm=arm,
            split=split,
            checkpoint=step,
            ordinal=i + 1,
            metric=None,
            denominator=1,
            status="censored",
            missing_reason=reason,
        )
        for s in frozen["cohort_manifest"]
        for arm in ARMS
        for split, steps in (("online", [None]), ("retention", [0, 24, 48]))
        for step in steps
        for i, w in enumerate(s[split])
    ]


def release(
    store: c.Store,
    event: str,
    value: Json,
    ordinal: int,
    arm: str,
    delay: int,
    *,
    label: str | None,
) -> Json:
    """Commit selected learning components and acknowledgement in one durable write."""
    if arm not in ARMS or delay not in (0, 4, 8, 12) or (delay == 0) != ARMS[arm]["diagnostic"]:
        raise ValueError("arm_or_delay")
    pending = store.state["pending"].get(event)
    if (
        store.active is not None
        or pending is None
        or label not in c.m.LABELS
        or ordinal != pending["ordinal"] + delay
        or canonical_hash(value) != pending["evidence_hash"]
    ):
        return dict(status="invalid", optimizer_steps=0)
    verified = c.m.decide(value)
    if label != verified["decision"]:
        return dict(status="invalid", optimizer_steps=0)
    before = deepcopy(store.state)
    state = deepcopy(before)
    prefix = state["released"] + [label]
    training_label = prefix[(ordinal * 7) % len(prefix)] if "shuffled" in arm else label
    gradient: Json = dict(optimizer_steps=0, gradient_norm=0.0, parameter_delta=0.0)
    if ARMS[arm]["calibration"]:
        state["parameters"], gradient = c.update(
            state["parameters"], pending["counts"], training_label
        )
    template = c.m.normalize(value)
    add = (
        ARMS[arm]["additions"]
        and training_label == label
        and template not in state["templates"]
        and len(state["templates"]) < 32
    )
    if add:
        state["templates"].append(template)
        state.setdefault("template_provenance", {})[template] = dict(
            event_id=event,
            evidence_hash=canonical_hash(value),
            query=value["query"],
            verifier=verified,
            binding_witnesses=c.binding_provenance(value),
        )
    state["pending"].pop(event)
    state["acks"].append(event)
    state["released"].append(label)
    store.commit(state)
    return dict(
        gradient,
        status="complete",
        event_id=event,
        prediction_hash=canonical_hash(pending),
        prediction_ordinal=pending["ordinal"],
        feedback_ordinal=ordinal,
        verified_label=label,
        feedback_label=training_label,
        released_prefix=prefix,
        parameters_before=before["parameters"],
        parameters_after=state["parameters"],
        weights_changed=before["parameters"] != state["parameters"],
        templates_changed=before["templates"] != state["templates"],
        template_added=add,
        checkpoint_hash=canonical_hash(state),
    )


def run_stream(raw: Path, plan: Json, heads: dict[int, Json]) -> Json:
    """Reuse the authenticated cache only for pure verifier calls during one stream."""
    with prior.upstream.primitives():
        return _run_stream(raw, plan, heads)


def _run_stream(raw: Path, plan: Json, heads: dict[int, Json]) -> Json:
    """Reset arms independently, seal before release, and keep retention read-only."""
    work: Json = dict(
        stream=plan,
        heads=heads,
        rows=[],
        causal_control_rows=[],
        gradient_rows=[],
        restart_rows=[],
        movement_intervals=[],
    )
    worlds = [payload(w) for w in plan["online"]]
    retained = [payload(w) for w in plan["retention"]]
    for arm in ARMS:
        store = c.Store(raw / (arm + ".json"), deepcopy(heads[plan["seed"]]))
        identity = dict(cluster_id=plan["stream_id"], seed=plan["seed"], arm=arm)
        online: dict[str, Json] = {}
        moves: list[Json] = []

        def retention(checkpoint: int) -> None:
            before = canonical_hash(store.state)
            for recipe, value in zip(plan["retention"], retained):
                counts = c.m.observe(value, store.state["templates"])["counts"]
                p = c.probabilities(store.state["parameters"], counts)
                truth = c.m.decide(value)["decision"]
                measured = score(p, truth, 0.5)
                work["rows"].append(
                    dict(
                        identity,
                        unit_id=recipe["unit_id"] + f":step{checkpoint}",
                        split="retention",
                        checkpoint=checkpoint,
                        error_type=recipe["error_type"],
                        prediction=p,
                        truth=truth,
                        **measured,
                        denominator=1,
                        status="complete",
                        missing_reason=None,
                    )
                )
            require(before == canonical_hash(store.state), "retention_mutation")

        retention(0)
        delays = [0] * 48 if ARMS[arm]["diagnostic"] else plan["delays"]
        for ordinal in range(1, 61):
            if ordinal <= 48:
                event = f"q{ordinal}"
                sealed = store.predict(event, worlds[ordinal - 1], 0.5)
                store.close()
                online[event] = dict(
                    identity,
                    unit_id=plan["online"][ordinal - 1]["unit_id"],
                    split="online",
                    checkpoint=None,
                    ordinal=ordinal,
                    error_type=plan["online"][ordinal - 1]["error_type"],
                    future_window=ordinal >= 25,
                    sealed=sealed,
                    prediction_seal=canonical_hash(sealed),
                    prediction=sealed["prediction"],
                    denominator=1,
                    status="sealed",
                    missing_reason="feedback_pending",
                )
            for at, delay in enumerate(delays, 1):
                if at + delay != ordinal:
                    continue
                event = f"q{at}"
                truth = str(c.m.decide(worlds[at - 1])["decision"])
                row = release(store, event, worlds[at - 1], ordinal, arm, delay, label=truth)
                require(row["status"] == "complete", "release_integrity")
                work["causal_control_rows"].append(dict(identity, **row))
                moves.append(row)
                if row["optimizer_steps"]:
                    work["gradient_rows"].append(dict(identity, **row))
                measured = score(online[event]["prediction"], truth, 0.5)
                online[event].update(measured, truth=truth, status="complete", missing_reason=None)
            if ordinal == 24:
                before = canonical_hash(store.state)
                store = c.Store(store.path, {})
                unchanged = before == canonical_hash(store.state)
                require(unchanged, "restart_drift")
                work["restart_rows"].append(dict(identity, unchanged=unchanged))
                retention(24)
        require(not store.state["pending"] and len(store.state["acks"]) == 48, "pending_drain")
        retention(48)
        work["rows"].extend(online.values())
        for low, high in ((1, 24), (25, 48)):
            selected = [r for r in moves if low <= r["prediction_ordinal"] <= high]
            work["movement_intervals"].append(
                dict(
                    identity,
                    prediction_window=[low, high],
                    optimizer_steps=sum(r["optimizer_steps"] for r in selected),
                    gradient_norm_sum=sum(r["gradient_norm"] for r in selected),
                    parameter_path_length=sum(r["parameter_delta"] for r in selected),
                    template_admissions=sum(r["template_added"] for r in selected),
                )
            )
        progress(plan["stream_id"] + "/" + arm, len(work["rows"]), 1152 - len(work["rows"]))
    return work


def controls(raw: Path) -> list[Json]:
    """Measure finite differences and exact equivalence to the original linear store."""
    params = c.head(11, "linear")
    flat, restore = ravel_pytree([jnp.asarray(x) for x in params["weights"]])
    x = jnp.asarray([c.m.features([5, 4, 1, 2, 0, 1], i) for i in range(3)])
    fn = lambda v: c.d.loss(restore(v), x, jnp.asarray(1), "linear")
    analytic = np.asarray(jax.grad(fn)(flat))
    finite = np.asarray(
        [
            (float(fn(flat.at[i].add(0.01))) - float(fn(flat.at[i].add(-0.01)))) / 0.02
            for i in range(len(flat))
        ]
    )
    error = float(np.max(np.abs(analytic - finite)))
    rows = [dict(check="finite_difference", max_error=error, tolerance=1e-4, passed=error < 1e-4)]
    original = c.Store(raw / "original.json", deepcopy(params))
    matched = c.Store(raw / "matched.json", deepcopy(params))
    shuffled = c.Store(raw / "shuffled.json", deepcopy(params))
    for i in range(1, 5):
        value = c.m.world(f"control-{i}", c.m.LABELS[i % 3])
        label = str(c.m.decide(value)["decision"])
        for store in (original, matched, shuffled):
            store.predict(f"q{i}", value, 0.5)
            store.close()
        original.release(f"q{i}", value, i + 8, label=label)
        release(matched, f"q{i}", value, i + 8, "linear_combined", 8, label=label)
        row = release(shuffled, f"q{i}", value, i + 8, "linear_shuffled_combined", 8, label=label)
        rows.append(
            dict(
                check="original_combined",
                event_id=f"q{i}",
                passed=original.path.read_bytes() == matched.path.read_bytes(),
                numerical_training_calls=3,
            )
        )
        rows.append(
            dict(
                check="prefix_shuffle",
                event_id=f"q{i}",
                passed=row["feedback_label"] in row["released_prefix"],
                feedback_label=row["feedback_label"],
                released_prefix=row["released_prefix"],
            )
        )
    for arm, settings in ARMS.items():
        store = c.Store(raw / (arm + ".json"), deepcopy(params))
        value = c.m.world("lifecycle", "support")
        delay = 0 if settings["diagnostic"] else 8
        store.predict("q1", value, 0.5)
        before = store.path.read_bytes()
        active = release(store, "q1", value, 1 + delay, arm, delay, label="support")
        store.close()
        rejected = []
        for event, evidence, ordinal, label in (
            ("absent", value, 1 + delay, "support"),
            ("q1", value, 0, "support"),
            ("q1", value, 99, "support"),
            ("q1", value, 1 + delay, "refute"),
            ("q1", c.m.world("changed", "support"), 1 + delay, "support"),
        ):
            result = release(store, event, evidence, ordinal, arm, delay, label=label)
            rejected.append(result["status"] == "invalid" and store.path.read_bytes() == before)
        state = deepcopy(store.state)
        failed = False
        with patch.object(c, "_atomic_write", side_effect=OSError("owned_write_failure")):
            try:
                release(store, "q1", value, 1 + delay, arm, delay, label="support")
            except OSError:
                failed = store.state == state and store.path.read_bytes() == before
        result = release(store, "q1", value, 1 + delay, arm, delay, label="support")
        restarted = c.Store(store.path, {})
        duplicate = release(restarted, "q1", value, 1 + delay, arm, delay, label="support")
        rows.append(
            dict(
                check="causal_lifecycle",
                arm=arm,
                rejected_count=len(rejected),
                atomic_write_failure=failed,
                restart_unchanged=restarted.state == store.state,
                weights_byte_identity=canonical_hash(state["parameters"])
                == canonical_hash(store.state["parameters"]),
                templates_byte_identity=canonical_hash(state["templates"])
                == canonical_hash(store.state["templates"]),
                numerical_training_calls=2 * int(settings["calibration"]),
                passed=active["status"] == "invalid"
                and all(rejected)
                and failed
                and duplicate["status"] == "invalid"
                and restarted.state == store.state
                and result["weights_changed"] == settings["calibration"]
                and result["templates_changed"] == settings["additions"],
            )
        )
    require(all(r["passed"] for r in rows), "owned_controls")
    return rows


def projection(pilots: list[float], initialization_s: float) -> Json:
    """Budget full execution and cold replay without selecting on pilot benefit."""
    return prior.projection(pilots, initialization_s)


def replay_pilot(work: Json, raw: Path) -> bool:
    """Recompute each sealed row and transition instead of trusting rehashed claims."""
    for row in work["rows"]:
        require(
            all(row[k] == v for k, v in score(row["prediction"], row["truth"], 0.5).items()),
            "pilot_drift",
        )
    heads = {int(k): v for k, v in work["heads"].items()}
    actual = run_stream(raw, work["stream"], heads)
    expected = dict(work, heads=heads)
    require(actual == expected, "pilot_drift")
    return True
