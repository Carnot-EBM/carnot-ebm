"""REQ-REPORT-8947: exercise gradient replay on development, then seal evaluation.

Shared orchestration preserves the prediction, feedback and retention schedule.
Neither descriptive development deltas nor promotions establish future benefit.
"""

from copy import deepcopy
import json
from pathlib import Path
from typing import Any

from carnot.memory import gradient_replay_8947 as replay
from carnot.memory import signed_memory_8892 as signed
from carnot.reporting import compatible_replay_methods_8933 as previous
from carnot.reporting import guarded_update_methods_8905 as base
from carnot.reporting import topology_memory_methods_8881 as old
from carnot.reporting import evidence_set_methods_8768 as symbolic
from carnot.reporting.current_work_receipt import canonical_hash
from carnot.reporting.current_work_receipt import atomic_json
from carnot.reporting.causal_learning_audit_8810 import require
from carnot.reporting.v748_collection_contract import namespace

__all__ = ["signed", "old"]
Json = dict[str, Any]
ROOT, SEED = base.ROOT, 765000
NAME, TASK = "experiment_8947_v765_gradient_replay_methods", "exp8947-gradient-replay-methods"
CLI = "scripts/experiments/" + NAME + ".py"
ARMS, COUNT = replay.ARMS, 18432
OWNED = [
    "python/carnot/memory/gradient_replay_8947.py",
    "python/carnot/reporting/gradient_replay_methods_8947.py",
    "python/carnot/reporting/gradient_replay_runner_8947.py",
    CLI,
]
TESTS = ["tests/python/test_gradient_replay_methods_8947.py"]
anchors = base.anchors
reuse = namespace(previous, "gradient8947_methods")
planning = namespace(previous, "gradient8947_planning")
planning.SEED = SEED


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Expose phase boundaries and actual counts without padding elapsed time."""
    print(f"[exp8947] phase={phase} completed={completed} pending={pending}", flush=True)


def protocol() -> Json:
    """Freeze fresh RNG histories and eleven bounds before any development outcome."""
    p: Json = planning.protocol()
    for role in ("pilot_manifest", "frozen_evaluation_manifest"):
        for plan in p[role]:
            plan["stream_id"] = plan["stream_id"].replace("v764", "v765-gradient")
            for kind in ("online", "retention"):
                for i, unit in enumerate(plan[kind]):
                    unit["unit_id"] = f"{plan['stream_id']}/{kind}/{i}"
    bounds = p["evaluation_plan"]["bounds"]
    for bound in bounds:
        if bound["against"] == "compatible_permuted":
            bound["against"] = "gradient_permuted"
    bounds.insert(
        3,
        dict(
            split="online",
            endpoint="brier",
            ordinals=[25, 48],
            against="signature_replay",
            upper=0,
            strict=True,
        ),
    )
    p["evaluation_plan"].update(
        candidate_arm="gradient_replay", family_intervals=11, alpha_per_bound=0.01 / 11, seed=765200
    )
    p.update(
        schema=replay.SCHEMA,
        arms=list(ARMS),
        expected_row_counts=dict(development=COUNT, evaluation=55296),
        replay_rule="At W1 after current TRAIN SGD, maximal cosine of current and prior TRAIN gradients; zero norms score zero; stable unit ID ties; empty repeats current",
        scan_accounting="Every trainable arm evaluates current plus all prior buffer rows at its own W1; one initial current evaluation; apply current and selected scan gradient only",
        optimizer_equations=[
            "p=softmax(W*x)",
            "W1=W-0.1*gradient(-log(p[y_current_released]))",
            "g_current=gradient(-log(softmax(W1*x_current)[y_current_released]))",
            "g_i=gradient(-log(softmax(W1*x_i)[y_i_released])) for each prior TRAIN buffer row",
            "j=argmax_i cosine(g_current,g_i); zero norm=0; stable IDs; empty=current",
            "W2=W1-0.1*g_selected",
        ],
        selector_precedent="Local falsifiable adaptation of arXiv:2508.01908; tiny signed head only, not generator training",
        stop_query=48,
        pending_at_stop=8,
    )
    return p


def trajectory(raw: Path, plan: Json, params: Json, *, restart: bool) -> Json:
    """Stop at query48 with eight pending labels; retention cannot mutate a store."""
    anchor = anchors()
    stores = {
        a: replay.Store(raw / (a + ".json"), params, a, anchor, seed=plan["seed"]) for a in ARMS
    }
    rows: list[Json] = []
    values = [
        old.world(
            old.graphs()["development"][u["template"]],
            u["unit_id"] + "/",
            u["error_type"],
            u["world_seed"],
        )
        for u in plan["online"]
    ]
    controls = []

    def retention(checkpoint: int) -> None:
        before = {a: canonical_hash(s.state) for a, s in stores.items()}
        for i, unit in enumerate(plan["retention"]):
            value = old.world(
                old.graphs()["development"][unit["template"]],
                unit["unit_id"] + "/",
                unit["error_type"],
                unit["world_seed"],
            )
            for arm, store in stores.items():
                observation = signed.observe(value, store.state["templates"])
                prediction = signed.probabilities(store.state["committed"], observation["counts"])
                rows.append(
                    base.scored(
                        plan,
                        arm,
                        unit,
                        dict(ordinal=i + 1, prediction=prediction),
                        str(symbolic.decide(value)["decision"]),
                        "retention",
                        checkpoint,
                    )
                )
        require(
            before == {a: canonical_hash(s.state) for a, s in stores.items()}, "retention_mutation"
        )
        controls.append(dict(check="retention_read_only", checkpoint=checkpoint, passed=True))
        atomic_json(raw / "rows.json", dict(rows=rows))

    retention(0)
    released = 0
    for tick, (unit, value) in enumerate(zip(plan["online"], values, strict=True), 1):
        for arm, store in stores.items():
            sealed = store.predict(unit["unit_id"], value, tick, tick + 8)
            rows.append(
                base.scored(
                    plan,
                    arm,
                    unit,
                    sealed,
                    str(symbolic.decide(value)["decision"]),
                    "online",
                    None,
                )
            )
        if tick >= released + 16:
            batch = [
                (plan["online"][i]["unit_id"], values[i]) for i in range(released, released + 8)
            ]
            for store in stores.values():
                store.release(batch, tick)
            released += 8
        if tick in (24, 48):
            retention(tick)
        if restart and tick == 24:
            for arm, store in list(stores.items()):
                resumed = replay.Store(store.path, {}, arm, anchor, seed=plan["seed"])
                require(resumed.state == store.state, "restart_drift")
                stores[arm] = resumed
            controls.append(dict(check="durable_restart_at24", passed=True))
        atomic_json(raw / "rows.json", dict(rows=rows))
        if tick % 8 == 0:
            progress("development_" + plan["stream_id"], tick, 48 - tick)
    states = {a: s.state for a, s in stores.items()}
    controls += [
        dict(
            check="stop_at48",
            passed=all(
                s["prediction_cursor"] == 48 and s["cursor"] == 40 and len(s["pending"]) == 8
                for s in states.values()
            ),
        ),
        dict(
            check="matched_guard_work",
            passed=len({s["guard_scores"] for s in states.values()}) == 1,
        ),
        dict(
            check="true_template_admission",
            passed=all(s["templates"] == states[ARMS[0]]["templates"] for s in states.values()),
        ),
    ]
    return dict(
        rows=rows,
        control_rows=controls,
        states=states,
        promotion_rows=[
            dict(r, arm=a, cluster_id=plan["stream_id"], seed=plan["seed"])
            for a, s in states.items()
            for r in s["promotions"]
        ],
        numerical_training_calls=sum(s["training_calls"] for s in states.values()),
    )


def run_stream(raw: Path, plan: Json, params: Json, *, restart: bool = True) -> Json:
    """Reuse the existing delayed stream and guard, removing obsolete selector parity."""
    result = trajectory(raw, plan, params, restart=restart)
    states = result["states"]
    trainable = [states[a] for a in ARMS[1:]]
    result["control_rows"] = [
        r for r in result["control_rows"] if r["check"] != "matched_training_work"
    ]
    result["control_rows"] += [
        dict(check="equal_scans", passed=len({s["evaluated_gradients"] for s in trainable}) == 1),
        dict(
            check="two_applied_gradients",
            passed=len({s["training_calls"] for s in trainable}) == 1
            and all(
                s["training_calls"] == 2 * sum(r["split"] == "train" for r in s["released"])
                for s in trainable
            )
            and states[ARMS[0]]["training_calls"] == 0,
        ),
        dict(check="unchanged_frozen_head", passed=states[ARMS[0]]["committed"] == params),
        dict(
            check="guard_truth_unchanged",
            passed=all(
                r["label"] == r["true_label"]
                for s in states.values()
                for r in s["released"]
                if r["split"] == "guard"
            ),
        ),
        dict(
            check="equal_true_feedback",
            passed=all(
                [(r["group_id"], r["true_label"]) for r in s["released"]]
                == [(r["group_id"], r["true_label"]) for r in states[ARMS[0]]["released"]]
                for s in states.values()
            ),
        ),
        dict(
            check="no_retention_reads",
            passed=all(
                not any("/retention/" in group for group in s["splits"]) for s in states.values()
            ),
        ),
    ]
    result["rows"] = [
        dict(r, metric=metric, value=value, numerator=value)
        for r in result["rows"]
        for metric, value in (("brier", r["metric"]), ("error", r["error"]))
    ]
    return result


def development_summary(rows: list[Json]) -> list[Json]:
    """Keep effect estimates descriptive so null effects cannot close exercised methods."""
    summaries = []
    for bound in protocol()["evaluation_plan"]["bounds"]:
        deltas = []
        for cluster in sorted({r["cluster_id"] for r in rows}):
            means = {}
            for arm in ("gradient_replay", bound["against"]):
                values = [
                    r["value"]
                    for r in rows
                    if r["cluster_id"] == cluster
                    and r["arm"] == arm
                    and r["metric"] == bound["endpoint"]
                    and r["split"] == bound["split"]
                    and r["checkpoint"] == bound.get("checkpoint")
                    and (bound["split"] != "online" or r["ordinal"] >= 25)
                ]
                means[arm] = sum(values) / len(values)
            deltas.append(means["gradient_replay"] - means[bound["against"]])
        summaries.append(
            dict(
                bound,
                paired_stream_deltas=deltas,
                mean=sum(deltas) / len(deltas),
                scope="development descriptive only; no benefit or readiness gate",
            )
        )
    return summaries


recover_previous = reuse.recover
census_previous = reuse.census


def census(reason: str) -> Json:
    """Blocked artifacts still name every scheduled cell and every required ledger."""
    return dict(
        census_previous(reason),
        candidate_gradient_rows=[],
        released_label_access_rows=[],
        applied_gradient_count=0,
        selection_disagreement=dict(
            numerator=0, denominator=0, value=None, within_gradient_count=0, nonzero_gradients=0
        ),
        cosine_distributions={arm: [] for arm in ARMS[1:]},
    )


def recover(raw: Path, reason: str) -> Json:
    """Retain partial scheduled rows and charge every persisted candidate evaluation."""
    result: Json = recover_previous(raw, reason)
    candidates, access = [], []
    evaluated = 0
    for plan in protocol()["pilot_manifest"]:
        for arm in ARMS:
            path = raw / "states" / plan["stream_id"] / (arm + ".json")
            if path.is_file():
                state = json.loads(path.read_bytes())
                evaluated += state["evaluated_gradients"]
                candidates += [
                    dict(r, arm=arm, cluster_id=plan["stream_id"])
                    for r in state["candidate_gradient_rows"]
                ]
                access += [
                    dict(
                        arm=arm,
                        cluster_id=plan["stream_id"],
                        **{
                            k: r[k]
                            for k in (
                                "group_id",
                                "split",
                                "label",
                                "true_label",
                                "ordinal",
                                "due",
                                "release_tick",
                            )
                        },
                    )
                    for r in state["released"]
                ]
    choices = {
        (r["cluster_id"], r["released_id"], r["arm"]): r for r in result["replay_selection_rows"]
    }
    pairs = [
        (r, choices[(r["cluster_id"], r["released_id"], "signature_replay")])
        for r in result["replay_selection_rows"]
        if r["arm"] == "gradient_replay"
    ]
    count = sum(a["selected_id"] != b["selected_id"] for a, b in pairs)
    result.update(
        candidate_gradient_rows=candidates,
        released_label_access_rows=access,
        applied_gradient_count=result["numerical_training_calls"],
        numerical_training_calls=evaluated,
        selection_disagreement=dict(
            numerator=count,
            denominator=len(pairs),
            value=count / len(pairs) if pairs else None,
            within_gradient_count=sum(a["gradient_id"] != a["signature_id"] for a, _ in pairs),
            nonzero_gradients=sum(
                r["gradient_norm"] > 0 for r in candidates if r["arm"] == "gradient_replay"
            ),
        ),
        cosine_distributions={
            arm: [r["cosine"] for r in candidates if r["arm"] == arm and r["role"] == "prior_W1"]
            for arm in ARMS[1:]
        },
    )
    return result


def exercised(result: Json) -> bool:
    """A mechanism must make a distinct informed choice before methods can qualify."""
    d = result["selection_disagreement"]
    return bool(
        d["numerator"] > 0
        and d["nonzero_gradients"] > 0
        and all(r["passed"] for r in result["control_rows"])
    )


reuse.ARMS, reuse.replay = ARMS, replay
reuse.protocol, reuse.progress = protocol, progress
reuse.run_stream, reuse.recover = run_stream, recover
reuse.census = census
reuse.development_summary = development_summary
measure = reuse.measure
