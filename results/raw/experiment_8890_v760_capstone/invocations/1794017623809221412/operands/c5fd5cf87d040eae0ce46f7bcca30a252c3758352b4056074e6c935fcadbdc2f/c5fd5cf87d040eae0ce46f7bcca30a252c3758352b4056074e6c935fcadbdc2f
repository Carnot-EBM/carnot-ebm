"""REQ-REPORT-8804: measure later loss without teaching a query its own answer.

The existing Gibbs head and transaction store remain the learning mechanism.
These streams add immutable prediction receipts, disjoint graphs and complete
delayed acknowledgements, so the later audit can separate execution from benefit.
"""

from __future__ import annotations

from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import random
import time
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from carnot.memory import causal_memory_8802 as m
from carnot.memory.transactional_constraint_memory import _atomic_write, canonical_json_bytes
from carnot.reporting.current_work_receipt import canonical_hash, sha256_file

Json = dict[str, Any]


@jax.jit
def probability_array(parameters: list[jax.Array], x: jax.Array) -> jax.Array:
    """Compile the existing Gibbs forward pass once instead of dispatching each small operation."""
    return jax.nn.softmax(-jax.vmap(lambda row: m.energy(parameters, row))(x))


def probabilities(params: Json, counts: list[int]) -> list[float]:
    """Keep the frozen candidate features and float32 Gibbs calculation unchanged."""
    x = jnp.asarray([m.features(counts, i) for i in range(3)])
    return list(map(float, np.asarray(probability_array(m.arrays(params), x))))


def update(params: Json, counts: list[int], label: str) -> tuple[Json, Json]:
    """Use the frozen training step; compute diagnostic norms together on the host."""
    old = m.arrays(params)
    new, loss, grads = m.train_step(
        old,
        jnp.asarray([m.features(counts, i) for i in range(3)]),
        jnp.asarray(m.LABELS.index(label)),
    )
    result = {k: np.asarray(v).tolist() for k, v in zip(("w1", "b1", "w_out", "b_out"), new)}
    return result, dict(
        loss=float(loss),
        gradient_norm=float(sum(np.sum(np.asarray(g) ** 2) for g in grads) ** 0.5),
        parameter_delta=float(
            sum(np.sum((np.asarray(a) - np.asarray(b)) ** 2) for a, b in zip(old, new)) ** 0.5
        ),
    )


class StreamStore(m.Store):
    """Reuse shared durability and frozen state fields while batching numerical dispatch."""

    def predict(self, event: str, payload: Json, threshold: float) -> Json:
        """Seal the same fixed snapshot; optimized serving must preserve all causal guards."""
        if self.active is not None:
            raise RuntimeError("active query")
        if event in self.state["pending"] or event in self.state["acks"]:
            raise ValueError("duplicate prediction")
        if len(self.state["pending"]) >= 32:
            raise ValueError("pending capacity")
        self.active = deepcopy(self.state)
        observation = m.observe(payload, self.active["templates"])
        p = probabilities(self.active["parameters"], observation["counts"])
        sealed = dict(
            observation,
            event_id=event,
            ordinal=int(event[1:]),
            prediction=p,
            action=m.action(p, threshold),
            state_hash=canonical_hash(self.active),
            evidence_hash=canonical_hash(payload),
        )
        self.state["pending"][event] = sealed
        self.save()
        return deepcopy(sealed)

    def release(self, event: str, payload: Json, ordinal: int, arm: str) -> Json:
        """Admit templates only from released verified evidence, using the frozen arm rules."""
        if self.active is not None:
            raise RuntimeError("active query")
        if event in self.state["acks"]:
            raise ValueError("duplicate acknowledgement")
        pending = self.state["pending"][event]
        if ordinal < pending["ordinal"] + 8:
            raise ValueError("early feedback")
        if canonical_hash(payload) != pending["evidence_hash"]:
            raise ValueError("evidence mismatch")
        label = str(m.decide(payload)["decision"])
        prefix = self.state["released"] + [label]
        training_label = (
            prefix[(ordinal * 7) % len(prefix)] if arm == "shuffled_feedback" else label
        )
        new, gradient = update(self.state["parameters"], pending["counts"], training_label)
        coefficient = arm in {"calibration_only", "combined", "shuffled_feedback"}
        template = m.normalize(payload)
        add = (
            arm in {"additions_only", "combined", "shuffled_feedback"}
            and training_label == label
            and template not in self.state["templates"]
            and len(self.state["templates"]) < 32
        )
        state = deepcopy(self.state)
        if add:
            state["templates"].append(template)
        if coefficient:
            state["parameters"] = new
        state["pending"].pop(event)
        state["acks"].append(event)
        state["released"].append(label)
        self.state = state
        self.save()
        return dict(
            gradient,
            event_id=event,
            arm=arm,
            feedback_label=training_label,
            independently_verified_label=label,
            released_prefix=prefix,
            feedback_ordinal=ordinal,
            prediction_ordinal=pending["ordinal"],
            template_added=add,
            template=template,
            coefficient_committed=coefficient,
            checkpoint_hash=canonical_hash(state),
            feedback_verifier_calls=1,
            optimizer_steps=1,
        )


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Flush real work counts so the parent can distinguish progress from a stall."""
    print(f"[exp8804] phase={phase} completed={completed} pending={pending}", flush=True)


def reference(path: Path) -> Json:
    """Bind raw operands to their exact bytes rather than a mutable filename."""
    return dict(path=str(path.absolute()), sha256=sha256_file(path))


def durable(path: Path, value: Json) -> Json:
    """Include file and directory fsync in the measured receipt boundary."""
    _atomic_write(path, canonical_json_bytes(value))
    return reference(path)


def manifest() -> Json:
    """Preserve the prototype roster while honoring the explicit current task seed."""
    value = m.freeze_manifest()
    value.update(seeds=[11, 23, 47], arms=list(m.ARMS), online_executed=True)
    return value


def payload(recipe: Json, label: str) -> Json:
    """A disconnected path makes graph classes disjoint without adding answer features."""
    prefix = recipe["unit_id"] + ":"
    value = m.world(prefix, label)
    value["facts"] += [
        [f"{prefix}p{i}", "padding", f"{prefix}p{i + 1}", True]
        for i in range(recipe["disconnected_path_edges"])
    ]
    return value


def census(frozen: Json, reason: str) -> tuple[list[Json], list[Json]]:
    """Allocate every scheduled slot before work; absence must never look like zero loss."""
    online, retained = [], []
    for stream in frozen["streams"]:
        for seed in frozen["seeds"]:
            for arm in frozen["arms"]:
                for split, recipes, steps, target in [
                    ("online", stream["worlds"], [None], online),
                    ("retention", stream["retention"], frozen["retention_checkpoints"], retained),
                ]:
                    for step in steps:
                        for ordinal, recipe in enumerate(recipes):
                            target.append(
                                dict(
                                    unit_id=recipe["unit_id"]
                                    + (f":step{step}" if step is not None else ""),
                                    world_id=recipe["unit_id"],
                                    cluster_id=stream["stream_id"],
                                    arm=arm,
                                    seed=seed,
                                    split=split,
                                    checkpoint=step,
                                    ordinal=ordinal,
                                    metric=None,
                                    denominator=1,
                                    status="censored",
                                    missing_reason=reason,
                                )
                            )
    return online, retained


def brier(prediction: list[float], label: str) -> float:
    """Use the proper unnormalized three-class Brier score on sealed probabilities."""
    return float(
        sum((p - float(i == m.LABELS.index(label))) ** 2 for i, p in enumerate(prediction))
    )


def heads(upstream: Json, seeds: list[int]) -> tuple[list[Json], int]:
    """Reuse exposed development heads; fit seed 47 on the same forty development examples."""
    available = {h["seed"]: h for h in upstream["calibrated_head_parameters"]}
    selected, calls = [], 0
    for seed in seeds:
        if seed in available:
            selected.append(deepcopy(available[seed]))
            continue
        progress("development_fit_before", 0, 40)
        params = m.initial_parameters(seed)
        for i in range(40):
            sample = m.world(f"fit-{i}", m.LABELS[i % 3])
            params, _ = m.update(
                params, m.observe(sample, [m.normalize(sample)])["counts"], m.LABELS[i % 3]
            )
            calls += 1
        threshold = m.tune(params, [m.world(f"tune-{i}", m.LABELS[i % 3]) for i in range(20)])
        selected.append(dict(seed=seed, parameters=params, threshold=threshold))
        progress("development_fit_after", 40, 0)
    return selected, calls


def inputs(stream: Json) -> Json:
    """Freeze public graphs once per reset stream; only the generator uses construction labels."""
    rng = random.Random(8804 + int(stream["stream_id"].split("-")[-1]))
    result: Json = {}
    for split, key in [("online", "worlds"), ("retention", "retention")]:
        labels = [m.LABELS[i % 3] for i in range(len(stream[key]))]
        rng.shuffle(labels)
        result[split] = [payload(recipe, label) for recipe, label in zip(stream[key], labels)]
    return result


def run_arm(
    stream: Json,
    seed: int,
    arm: str,
    head: Json,
    frozen: Json,
    public: Json,
    raw: Path,
    work: Json,
    slots: Json,
) -> None:
    """Seal each read before truth, then release predecessors and drain all pending work."""
    directory = raw / stream["stream_id"] / str(seed) / arm
    store = StreamStore(directory / "state.json", deepcopy(head["parameters"]))
    initial = dict(templates=[], parameters=head["parameters"], pending={}, acks=[], released=[])
    work["reset_rows"].append(
        dict(
            cluster_id=stream["stream_id"],
            seed=seed,
            arm=arm,
            initial_state_equal=store.state == initial,
            state_hash=canonical_hash(store.state),
        )
    )
    starts: dict[int, int] = {}
    prediction_costs: dict[int, float] = {}

    def retain(step: int) -> None:
        before = canonical_hash(store.state)
        for i, sample in enumerate(public["retention"]):
            started = time.monotonic_ns()
            observation = m.observe(sample, store.state["templates"])
            p = probabilities(store.state["parameters"], observation["counts"])
            receipt = durable(
                directory / f"retention-{step}-{i}.json",
                dict(
                    payload=sample,
                    parameters=store.state["parameters"],
                    observation=observation,
                    prediction=p,
                    state_hash=before,
                    templates=list(store.state["templates"]),
                ),
            )
            sealed = time.monotonic_ns()
            truth = str(m.decide(sample)["decision"])
            row = slots[f"{stream['retention'][i]['unit_id']}:step{step}"]
            row.update(
                prediction=p,
                action=m.action(p, head["threshold"]),
                truth=truth,
                metric=brier(p, truth),
                paid_checks=5,
                grading_checks=1,
                active_template_ids=[canonical_hash(t) for t in store.state["templates"]],
                learning_state_unchanged=before == canonical_hash(store.state),
                sealed_ns=sealed,
                truth_recorded_ns=time.monotonic_ns(),
                duration_s=(time.monotonic_ns() - started) / 1e9,
                status="completed",
                missing_reason=None,
                seal_ref=receipt,
            )

    def release(ordinal: int, tail: bool) -> None:
        i = ordinal - 8
        started = time.monotonic_ns()
        update = store.release(f"e{i}", public["online"][i], ordinal, arm)
        work["numerical_training_calls"] += 1
        expected = deepcopy(store.state)
        restarted = m.Store(store.path, {})
        equal = restarted.state == expected
        ack = time.monotonic_ns()
        meta = dict(
            unit_id=stream["worlds"][i]["unit_id"],
            cluster_id=stream["stream_id"],
            seed=seed,
            arm=arm,
            denominator=1,
            status="completed",
            missing_reason=None,
        )
        work["release_order_rows"].append(dict(meta | update, tail=tail))
        if update["template_added"]:
            work["template_addition_rows"].append(
                dict(
                    meta | update,
                    template_id=canonical_hash(update["template"]),
                    novel_template=True,
                )
            )
        work["complete_learning_cost_rows"].append(
            dict(
                meta,
                predict_started_ns=starts[i],
                release_started_ns=started,
                ack_ns=ack,
                duration_s=(ack - starts[i]) / 1e9,
                prediction_service_s=prediction_costs[i],
                release_service_s=(ack - started) / 1e9,
                restart_equal=equal,
                tail=tail,
                paid_checks=5,
                feedback_verifier_calls=1,
                optimizer_steps=1,
                checkpoint_hash=canonical_hash(restarted.state),
                metric=(ack - starts[i]) / 1e9,
            )
        )

    retain(0)
    for i, sample in enumerate(public["online"]):
        starts[i] = time.monotonic_ns()
        pre = store.read()
        sealed = store.predict(f"e{i}", sample, head["threshold"])
        receipt = durable(
            directory / f"prediction-{i}.json",
            dict(payload=sample, pre_state=pre, pre_state_hash=canonical_hash(pre), sealed=sealed),
        )
        sealed_ns = time.monotonic_ns()
        truth = str(m.decide(sample)["decision"])
        truth_ns = time.monotonic_ns()
        store.close()
        baseline = next(
            (
                r
                for r in work["per_query_rows"]
                if r["world_id"] == stream["worlds"][i]["unit_id"]
                and r["seed"] == seed
                and r["arm"] == "read_only"
            ),
            {},
        )
        row = slots[stream["worlds"][i]["unit_id"]]
        row.update(
            sealed,
            prediction=sealed["prediction"],
            truth=truth,
            metric=brier(sealed["prediction"], truth),
            paid_checks=5,
            grading_checks=1,
            active_template_ids=[canonical_hash(t) for t in pre["templates"]],
            pre_update_state_hash=canonical_hash(pre),
            sealed_ns=sealed_ns,
            truth_recorded_ns=truth_ns,
            status="completed",
            missing_reason=None,
            changed_later_predicates=baseline.get("counts", sealed["counts"]) != sealed["counts"],
            seal_ref=receipt,
        )
        work["prediction_seal_refs"].append(receipt)
        prediction_costs[i] = (time.monotonic_ns() - starts[i]) / 1e9
        if i >= 8:
            release(i, False)
        if i + 1 in frozen["retention_checkpoints"]:
            retain(i + 1)
    tail_start = time.monotonic_ns()
    for ordinal in range(len(public["online"]), len(public["online"]) + 8):
        release(ordinal, True)
    elapsed = (time.monotonic_ns() - tail_start) / 1e9
    work["tail_drain_rows"].append(
        dict(
            cluster_id=stream["stream_id"],
            seed=seed,
            arm=arm,
            duration_s=elapsed,
            drained_count=8,
            pending_count=len(store.state["pending"]),
        )
    )
    work["tail_drain_cost_s"] += elapsed
    work["final_state_refs"].append(
        dict(reference(store.path), cluster_id=stream["stream_id"], seed=seed, arm=arm)
    )


def empty_work(frozen: Json, reason: str) -> Json:
    """A blocked or failed run keeps its full census and no invented measurements."""
    online, retained = census(frozen, reason)
    return dict(
        per_query_rows=online,
        retention_rows=retained,
        template_addition_rows=[],
        prediction_seal_refs=[],
        release_order_rows=[],
        complete_learning_cost_rows=[],
        tail_drain_rows=[],
        tail_drain_cost_s=0.0,
        reset_rows=[],
        input_refs=[],
        numerical_training_calls=0,
        failures=[],
        heads=[],
        final_state_refs=[],
        development_training_calls=0,
    )


def execute_serial(frozen: Json, selected: list[Json], raw: Path) -> Json:
    """Pair all arms on the same inputs and preserve failed slots in the denominator."""
    work = empty_work(frozen, "pending")
    work["heads"] = deepcopy(selected)
    heads_by_seed = {h["seed"]: h for h in selected}
    completed = 0
    intended = len(frozen["streams"]) * len(frozen["seeds"]) * len(frozen["arms"])
    for stream in frozen["streams"]:
        public = inputs(stream)
        work["input_refs"].append(durable(raw / "inputs" / (stream["stream_id"] + ".json"), public))
        for seed in frozen["seeds"]:
            for arm in frozen["arms"]:
                slots = {
                    r["unit_id"]: r
                    for key in ["per_query_rows", "retention_rows"]
                    for r in work[key]
                    if r["cluster_id"] == stream["stream_id"]
                    and r["seed"] == seed
                    and r["arm"] == arm
                }
                try:
                    run_arm(
                        stream, seed, arm, heads_by_seed[seed], frozen, public, raw, work, slots
                    )
                except (OSError, ValueError, RuntimeError) as exc:
                    work["failures"].append(
                        dict(cluster_id=stream["stream_id"], seed=seed, arm=arm, error=str(exc))
                    )
                    for row in slots.values():
                        if row["status"] != "completed":
                            row.update(status="failed", missing_reason=str(exc))
                completed += 1
                progress("stream_arm_complete", completed, intended - completed)
    return work


def execute(frozen: Json, selected: list[Json], raw: Path) -> Json:
    """Overlap independent durable streams, preserving each stream's causal sequence."""
    if len(frozen["streams"]) == 1:
        return execute_serial(frozen, selected, raw)
    with ThreadPoolExecutor(max_workers=min(30, len(frozen["streams"]))) as pool:
        futures = [
            pool.submit(execute_serial, dict(frozen, streams=[stream]), selected, raw)
            for stream in frozen["streams"]
        ]
        parts = []
        for future in futures:
            parts.append(future.result())
            progress("reset_stream_complete", len(parts), len(futures) - len(parts))
    work = empty_work(frozen, "pending")
    for field in [
        "per_query_rows",
        "retention_rows",
        "template_addition_rows",
        "prediction_seal_refs",
        "release_order_rows",
        "complete_learning_cost_rows",
        "tail_drain_rows",
        "reset_rows",
        "input_refs",
        "failures",
        "final_state_refs",
    ]:
        work[field] = [row for part in parts for row in part[field]]
    work["heads"] = deepcopy(selected)
    work["numerical_training_calls"] = sum(part["numerical_training_calls"] for part in parts)
    work["tail_drain_cost_s"] = sum(row["duration_s"] for row in work["tail_drain_rows"])
    return work


def replay_release(state: Json, released: Json, sample: Json, arm: str, cost: Json) -> bool:
    """Reconstruct an admitted update from its actual released predecessor and frozen arm rule."""
    i = released["prediction_ordinal"]
    pending = state["pending"].pop(f"e{i}")
    truth = str(m.decide(sample)["decision"])
    prefix = state["released"] + [truth]
    label = prefix[((i + 8) * 7) % len(prefix)] if arm == "shuffled_feedback" else truth
    template = m.normalize(sample)
    add = (
        arm in {"additions_only", "combined", "shuffled_feedback"}
        and label == truth
        and template not in state["templates"]
        and len(state["templates"]) < 32
    )
    coefficient = arm in {"calibration_only", "combined", "shuffled_feedback"}
    new, gradient = update(state["parameters"], pending["counts"], label)
    if coefficient:
        state["parameters"] = new
    if add:
        state["templates"].append(template)
    state["acks"].append(f"e{i}")
    state["released"].append(truth)
    return (
        released["feedback_ordinal"] == i + 8
        and released["feedback_label"] == label
        and released["independently_verified_label"] == truth
        and released["released_prefix"] == prefix
        and released["template_added"] == add
        and released["template"] == template
        and released["coefficient_committed"] == coefficient
        and all(released[k] == gradient[k] for k in gradient)
        and released["checkpoint_hash"] == cost["checkpoint_hash"] == canonical_hash(state)
    )


def check_work(work: Json, frozen: Json) -> Json:
    """Recompute measured loss and serving decisions from immutable pre-feedback operands.

    This verifies execution receipts; it does not estimate a learning benefit.
    State chains are checked against released updates, never evaluation truth
    copied into a predictor. Raw byte hashes also bind every durable seal.
    """
    try:
        online, retained = census(frozen, "pending")
        rows = work["per_query_rows"] + work["retention_rows"]
        training_calls = 0
        checks = dict(
            census=len(work["per_query_rows"]) == len(online)
            and len(work["retention_rows"]) == len(retained),
            completed=all(r["status"] == "completed" for r in rows),
            reset=bool(work["reset_rows"])
            and all(r["initial_state_equal"] for r in work["reset_rows"]),
            retention=all(r["learning_state_unchanged"] for r in work["retention_rows"]),
            release=len(work["release_order_rows"]) == len(online)
            and all(
                r["feedback_ordinal"] >= r["prediction_ordinal"] + 8
                and r["feedback_label"] in r["released_prefix"]
                for r in work["release_order_rows"]
            ),
            cost=len(work["complete_learning_cost_rows"]) == len(online)
            and all(
                r["restart_equal"]
                and r["ack_ns"] >= r["predict_started_ns"]
                and r["duration_s"] == (r["ack_ns"] - r["predict_started_ns"]) / 1e9
                for r in work["complete_learning_cost_rows"]
            ),
            tail=bool(work["tail_drain_rows"])
            and all(
                r["drained_count"] == 8 and r["pending_count"] == 0 for r in work["tail_drain_rows"]
            )
            and work["tail_drain_cost_s"] == sum(r["duration_s"] for r in work["tail_drain_rows"]),
            seals=True,
            predictions=True,
            state_chain=True,
            inputs=True,
        )
        expected_inputs = {stream["stream_id"]: inputs(stream) for stream in frozen["streams"]}
        for ref, public in zip(work["input_refs"], expected_inputs.values()):
            checks["inputs"] &= sha256_file(Path(ref["path"])) == ref["sha256"]
            checks["inputs"] &= json.loads(Path(ref["path"]).read_bytes()) == public
        checkpoints: Json = {}
        retention_states: Json = {}
        initial = {h["seed"]: h for h in work["heads"]}
        releases = {
            (r["cluster_id"], r["seed"], r["arm"], r["prediction_ordinal"]): r
            for r in work["release_order_rows"]
        }
        costs = {
            (r["cluster_id"], r["seed"], r["arm"], r["unit_id"]): r
            for r in work["complete_learning_cost_rows"]
        }
        finals = {(r["cluster_id"], r["seed"], r["arm"]): r for r in work["final_state_refs"]}
        for index, row in enumerate(rows):
            ref = row["seal_ref"]
            checks["seals"] &= sha256_file(Path(ref["path"])) == ref["sha256"]
            seal = json.loads(Path(ref["path"]).read_bytes())
            public = expected_inputs[row["cluster_id"]][row["split"]][row["ordinal"]]
            checks["inputs"] &= seal["payload"] == public
            if row["split"] == "online":
                state = seal["pre_state"]
                p = probabilities(state["parameters"], row["counts"])
                observed = m.observe(public, state["templates"])
                checks["predictions"] &= observed["counts"] == row["counts"]
                checks["seals"] &= (
                    seal["pre_state_hash"] == canonical_hash(state) == row["state_hash"]
                )
                checks["seals"] &= seal["sealed"]["prediction"] == row["prediction"]
                checks["seals"] &= all(row[k] == v for k, v in seal["sealed"].items())
                key = (row["cluster_id"], row["seed"], row["arm"])
                expected = checkpoints.setdefault(
                    key,
                    dict(
                        templates=[],
                        parameters=initial[row["seed"]]["parameters"],
                        pending={},
                        acks=[],
                        released=[],
                    ),
                )
                checks["state_chain"] &= state == expected
                if row["ordinal"] == 0:
                    retention_states[(*key, 0)] = deepcopy(expected)
                expected["pending"][f"e{row['ordinal']}"] = seal["sealed"]
                if row["ordinal"] >= 8:
                    i = row["ordinal"] - 8
                    released = releases[(*key, i)]
                    checks["state_chain"] &= replay_release(
                        expected,
                        released,
                        expected_inputs[row["cluster_id"]]["online"][i],
                        row["arm"],
                        costs[(*key, released["unit_id"])],
                    )
                    training_calls += 1
                if row["ordinal"] + 1 in frozen["retention_checkpoints"]:
                    retention_states[(*key, row["ordinal"] + 1)] = deepcopy(expected)
                if row["ordinal"] == 31:
                    for i in range(24, 32):
                        released = releases[(*key, i)]
                        checks["state_chain"] &= replay_release(
                            expected,
                            released,
                            expected_inputs[row["cluster_id"]]["online"][i],
                            row["arm"],
                            costs[(*key, released["unit_id"])],
                        )
                        training_calls += 1
                    ref = finals[key]
                    checks["state_chain"] &= sha256_file(Path(ref["path"])) == ref["sha256"]
                    checks["state_chain"] &= json.loads(Path(ref["path"]).read_bytes()) == expected
            else:
                p = probabilities(seal["parameters"], seal["observation"]["counts"])
                checks["predictions"] &= m.observe(public, seal["templates"]) == seal["observation"]
                expected = retention_states[
                    (row["cluster_id"], row["seed"], row["arm"], row["checkpoint"])
                ]
                checks["state_chain"] &= (
                    seal["parameters"] == expected["parameters"]
                    and seal["templates"] == expected["templates"]
                    and seal["state_hash"] == canonical_hash(expected)
                )
            truth = str(m.decide(public)["decision"])
            checks["predictions"] &= (
                p == row["prediction"]
                and truth == row["truth"]
                and brier(p, truth) == row["metric"]
                and row["paid_checks"] == 5
                and row["action"] == m.action(p, initial[row["seed"]]["threshold"])
                and row["sealed_ns"] <= row["truth_recorded_ns"]
            )
            if (index + 1) % 1000 == 0:
                progress("replay_rows", index + 1, len(rows) - index - 1)
        checks["release"] &= (
            work["numerical_training_calls"] == len(releases) + work["development_training_calls"]
        )
        checks["seals"] &= work["prediction_seal_refs"] == [
            r["seal_ref"] for r in work["per_query_rows"]
        ]
        checks["census"] &= [
            (r["unit_id"], r["cluster_id"], r["arm"], r["seed"], r["checkpoint"]) for r in rows
        ] == [
            (r["unit_id"], r["cluster_id"], r["arm"], r["seed"], r["checkpoint"])
            for r in online + retained
        ]
        return dict(
            passed=all(checks.values()), checks=checks, numerical_training_calls=training_calls
        )
    except (OSError, ValueError, KeyError, TypeError) as exc:
        return dict(passed=False, reason="invalid_work_operand", detail=str(exc))
