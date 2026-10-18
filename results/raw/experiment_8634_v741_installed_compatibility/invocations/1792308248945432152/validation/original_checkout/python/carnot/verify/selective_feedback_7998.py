"""REQ-SELF-7998: learn only from selected labels after durable predictions.

A shared scalar decay gives exact L2 updates while data writes remain local.
The label callback is the only learner input that can reveal a target.
"""

from __future__ import annotations

import copy
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from carnot.reporting.current_work_receipt import atomic_json, canonical_hash
from carnot.verify import sparse_energy_7996 as sparse
from carnot.verify.typed_development_7997 import action

Json = dict[str, Any]
ARMS = ("targeted_ipw", "targeted_unweighted", "uniform_ipw", "full_feedback", "frozen_no_write")
CONFIG = dict(
    primary_seed=17,
    algorithm_seeds=list(range(101, 121)),
    delay=20,
    learning_rate=0.01,
    l2=0.001,
    temperatures=[1.0, 0.5, 2.0],
    costs=dict(accept="5*y", reject="1-y", escalate=0.25),
    gradient_weighting="data_gradient_only_unweighted_L2",
    terminal_flush=False,
    stream_slots=256,
    random_seed=69398,
)


def acquire(head: Json, sources: list[Json], seed: int) -> list[Json]:
    """Freeze expected budgets using public predictions, never target availability.

    A source-keyed counter draw stays stable if an unrelated slot is missing.
    Sharing it across arms pairs schedules without equating realized counts.
    """
    probabilities = []
    for source in sources:
        if set(source) != {"family_id", "source_cluster_id", "q", "features", "status"}:
            raise ValueError("public_fields")
        usable = (
            source["status"] == "completed"
            and source["q"] is not None
            and source["features"] is not None
        )
        p = float(sparse.predict(head, sparse.inputs([source]))[0]) if usable else None
        probabilities.append(p)
    targeted = [0.0 if p is None else (0.5 if 0.05 <= p <= 0.75 else 0.125) for p in probabilities]
    eligible = sum(p is not None for p in probabilities)
    uniform = sum(targeted) / eligible if eligible else 0.0
    rows = []
    for slot, (source, p, pi) in enumerate(zip(sources, probabilities, targeted, strict=True)):
        key = canonical_hash(dict(source=source["source_cluster_id"], seed=seed))
        draw = int(key.split(":")[1][:13], 16) / 16**13
        for arm in ARMS:
            probability = (
                pi
                if arm.startswith("targeted")
                else (uniform if arm == "uniform_ipw" else (1.0 if arm == "full_feedback" else 0.0))
            )
            probability = probability if p is not None else 0.0
            rows.append(
                dict(
                    source,
                    arm=arm,
                    seed=seed,
                    slot=slot,
                    pi=probability,
                    draw=draw,
                    initial_probability=p,
                    selected=draw < probability,
                    numerator=probability,
                    denominator=1,
                    eligibility=p is not None,
                    failure_status=source["status"] == "failed",
                    censor_status=source["status"] == "censored",
                )
            )
    return rows


def apply_feedback(state: Json, receipt: Json, source: Json, dense: bool = False) -> Json | None:
    """Apply a due receipt once, including exact global decay on logical weights.

    Multiplying the data gradient by 1/pi corrects randomized observation.
    L2 is kept separate so label sampling does not change the registered penalty.
    """
    if receipt["receipt_id"] in state["seen_label_ids"]:
        return None
    if (
        receipt["due_slot"] != state["next_slot"]
        or receipt["origin_slot"] != receipt["due_slot"] - 20
    ):
        raise ValueError("feedback_not_due")
    state["seen_label_ids"].append(receipt["receipt_id"])
    if receipt["y"] is None:
        return None
    head, weight = state["head"], receipt["weight"]
    x = sparse.inputs([source])[0]
    ids, gradient = sparse.sparse_gradient(head, x, receipt["y"])
    if dense:
        derivative = sparse.dense_gradient(head, x, receipt["y"])
        theta = sparse.parameters(head)
        head.update(
            parameters=(
                theta - 0.01 * (weight * (derivative - 0.002 * theta) + 0.002 * theta)
            ).tolist(),
            decay_scale=1.0,
        )
    else:
        new_scale = head["decay_scale"] * (1 - 0.002 * 0.01)
        for i, g in zip(ids, gradient, strict=True):
            head["parameters"][int(i)] -= 0.01 * weight * float(g) / new_scale
        head["decay_scale"] = new_scale
    return dict(
        receipt,
        coefficient_touches=len(ids),
        global_decay_writes=1,
        logical_decay_coefficients=109,
        numerator=len(ids),
        denominator=109,
        eligibility=True,
        failure_status=False,
        censor_status=False,
        device="cpu",
    )


def stream(
    head: Json,
    sources: list[Json],
    acquisition: list[Json],
    label: Callable[[str], int | None],
    directory: Path,
    stop_at: int = 256,
    dense: bool = False,
) -> Json:
    """Persist each issued prediction before requesting its selected past label.

    Each complete slot is an atomic receipt containing the next state. A crash
    after issuance reissues the same deterministic prediction, then commits once.
    Terminal pending labels remain pending, with no artificial future decisions.
    """
    directory.mkdir(parents=True, exist_ok=True)
    arm, seed = acquisition[0]["arm"], acquisition[0]["seed"]
    commits = sorted(directory.glob("committed-*.json"))
    if commits:
        previous = json.loads(commits[-1].read_text())
        if previous["checksum"] != canonical_hash(previous["state"]):
            raise ValueError("checkpoint_checksum")
    state = (
        json.loads(commits[-1].read_text())["state"]
        if commits
        else dict(
            head=copy.deepcopy(head),
            next_slot=0,
            seen_label_ids=[],
            rng_state=dict(algorithm="sha256_source_seed_counter", seed=seed, next_slot=0),
            pending_feedback=[
                dict(family_id=r["family_id"], origin_slot=r["slot"], due_slot=r["slot"] + 20)
                for r in acquisition
                if r["selected"]
            ],
        )
    )
    for t in range(state["next_slot"], min(stop_at, len(sources))):
        source, selected = sources[t], acquisition[t]
        p = (
            float(sparse.predict(state["head"], sparse.inputs([source]))[0])
            if selected["eligibility"]
            else None
        )
        prediction = dict(
            family_id=source["family_id"],
            source_cluster_id=source["source_cluster_id"],
            slot=t,
            arm=arm,
            seed=seed,
            probability=p,
            action=action(p),
            numerator=p,
            denominator=int(p is not None),
            eligibility=p is not None,
            failure_status=selected["failure_status"],
            censor_status=selected["censor_status"],
            head_checksum=canonical_hash(state["head"]),
        )
        issued = directory / f"issued-{t:04d}.json"
        atomic_json(issued, dict(prediction=prediction, state_checksum=canonical_hash(state)))
        reveals, updates = [], []
        origin = t - 20
        if origin >= 0 and acquisition[origin]["selected"]:
            old = acquisition[origin]
            receipt = dict(
                family_id=old["family_id"],
                source_cluster_id=old["source_cluster_id"],
                origin_slot=origin,
                due_slot=t,
                arm=arm,
                seed=seed,
                receipt_id=f"{arm}:{seed}:{origin}",
                pi=old["pi"],
                weight=1 / old["pi"] if arm.endswith("ipw") else 1.0,
                y=label(old["family_id"]),
                numerator=1,
                denominator=1,
                eligibility=True,
                failure_status=False,
                censor_status=False,
            )
            reveals.append(receipt)
            updated = apply_feedback(state, receipt, sources[origin], dense)
            if updated is not None:
                updates.append(updated)
            state["pending_feedback"] = [
                r for r in state["pending_feedback"] if r["origin_slot"] != origin
            ]
        state["next_slot"] = t + 1
        state["rng_state"]["next_slot"] = t + 1
        atomic_json(
            directory / f"committed-{t:04d}.json",
            dict(
                state=state,
                checksum=canonical_hash(state),
                prediction=prediction,
                reveal_rows=reveals,
                update_rows=updates,
            ),
        )
        if t % 32 == 0:
            print(f"[exp7998] arm={arm} seed={seed} slots={t + 1}/{len(sources)}", flush=True)
    predictions, reveals, updates, checkpoints = [], [], [], []
    durable = 0
    for path in sorted(directory.glob("committed-*.json")):
        record = json.loads(path.read_text())
        if record["checksum"] != canonical_hash(record["state"]):
            raise ValueError("checkpoint_checksum")
        predictions.append(record["prediction"])
        reveals.extend(record["reveal_rows"])
        updates.extend(record["update_rows"])
        written = (
            path.stat().st_size
            + (directory / path.name.replace("committed-", "issued-")).stat().st_size
        )
        durable += written
        checkpoints.append(
            dict(
                arm=arm,
                seed=seed,
                slot=record["prediction"]["slot"],
                checksum=record["checksum"],
                durable_write_bytes=written,
                numerator=written,
                denominator=1,
                eligibility=True,
                failure_status=False,
                censor_status=False,
            )
        )
    return dict(
        issued_predictions=predictions,
        reveal_rows=reveals,
        update_rows=updates,
        checkpoint_rows=checkpoints,
        final_state=state,
        pending_feedback=state["pending_feedback"],
        durable_write_bytes=durable,
        parity_parameters=sparse.parameters(state["head"]).tolist(),
    )


def controls() -> Json:
    """Scripted controls test future consequences without natural-data claims.

    Uniform .5 observations have Brier headroom when every past target is one.
    Certain correct accepts have zero typed cost, so they cannot demonstrate a
    cost benefit even if an optimizer makes a tiny numerical change.
    """
    import tempfile
    from carnot.reporting.typed_validation_7997 import fixture_data

    outcomes: Json = {}
    for name, q, y in [("known_benefit", 0.5, 1), ("no_headroom", 0.001, 0)]:
        data = fixture_data()
        head = data["heads"]["spline"][0]
        sources = [dict(r, q=q) for r in data["public"]["stream"]]
        acquisitions = [r for r in acquire(head, sources, 101) if r["arm"] == "full_feedback"]
        with tempfile.TemporaryDirectory(prefix="carnot-7998-control-") as work:
            result = stream(head, sources, acquisitions, lambda _: y, Path(work))
        first, last = (
            result["issued_predictions"][0]["probability"],
            result["issued_predictions"][-1]["probability"],
        )
        outcomes[name] = dict(
            future_prediction_changed=abs(last - first) > 1e-6,
            initial_brier=(first - y) ** 2,
            final_brier=(last - y) ** 2,
            genuine_headroom=name == "known_benefit",
            benefit=name == "known_benefit" and (last - y) ** 2 < (first - y) ** 2,
            verifier_is_oracle=True,
            verdict_class="circular_positive",
            independent_natural_evidence=False,
        )
    outcomes["passed"] = (
        outcomes["known_benefit"]["future_prediction_changed"]
        and outcomes["known_benefit"]["benefit"]
        and not outcomes["no_headroom"]["benefit"]
    )
    return outcomes
