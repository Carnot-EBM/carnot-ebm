"""REQ-REPORT-8953: retain future causal measurements without granting benefit.

The authenticated methods own replay and guards. This loop stops at the final
query, so terminal pending feedback cannot train a head used for retention.
"""

from copy import deepcopy
import json
from pathlib import Path
import shutil
import time
from unittest.mock import patch
from typing import Any

import numpy as np

from carnot.memory import gradient_replay_8947 as gradient
from carnot.memory.gradient_replay_8947 import Store
from carnot.memory.compatible_replay_8933 import selector_controls
from carnot.reporting import gradient_replay_methods_8947 as methods
from carnot.reporting import guarded_update_methods_8905 as scoring
from carnot.reporting import guarded_causal_streams_8909 as causal
from carnot.reporting import evidence_set_methods_8768 as symbolic
from carnot.reporting.causal_learning_audit_8810 import require
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash, sha256_file
from carnot.reporting.v746_retained_evidence import reference

Json = dict[str, Any]
__all__ = ["require", "methods"]
ROOT, SEED = methods.ROOT, 765200
NAME, TASK = "experiment_8953_v765_gradient_causal_streams", "exp8953-gradient-causal-streams"
CLI = "scripts/experiments/" + NAME + ".py"
OWNED = [
    "python/carnot/reporting/gradient_causal_streams_8953.py",
    "python/carnot/reporting/gradient_causal_stream_runner_8953.py",
    CLI,
]
TESTS = ["tests/python/test_gradient_causal_streams_8953.py"]
SOURCE = methods.NAME
PIN = "sha256:65fffc8b8675bc066f25584d1fccf93257f9497b8d4e3ff2cad6a17a8f49a2ff"
protocol = methods.protocol
FIELDS = [
    "prediction_rows",
    "feedback_rows",
    "replay_rows",
    "retention_rows",
    "promotion_rows",
    "gradient_ledger",
    "state_hashes",
    "template_rows",
    "restart_rows",
    "pending_feedback_rows",
    "control_rows",
    "label_call_rows",
    "private_control_rows",
    "candidate_gradient_rows",
]


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Flush actual counts so a supervisor can distinguish slow work from a stall."""
    print(f"[exp8953] phase={phase} completed={completed} pending={pending}", flush=True)


def preconditions(raw: Path, sources: Path) -> Json:
    """Authenticate producer gates and retained bytes before opening fresh labels."""
    raw.mkdir(parents=True, exist_ok=True, mode=0o700)
    path = sources / (SOURCE + ".json")
    d: Json = dict(
        check="missing_file",
        upstream=SOURCE,
        artifact_field="gradient_methods_ready_score",
        expected=1,
        observed=None,
        path=str(path),
        hash=None,
    )
    refs: list[Json] = []
    diagnostics: list[Json] = []
    heads: Json = {}
    try:
        d["hash"] = sha256_file(path)
        d["check"] = "rejected_producer"
        value = json.loads(path.read_bytes())
        d.update(check="missing_field")
        actual = value["gradient_methods_ready_score"]
        d.update(check="observed_zero" if actual == 0 else "rejected_producer", observed=actual)
        require(type(actual) is int and actual == 1, "upstream_gate")
        d.update(artifact_field="producer_bytes", expected=PIN, observed=d["hash"])
        require(
            d["hash"] == PIN
            and value["experiment_id"] == 8947
            and value["task_id"] == methods.TASK
            and value["milestone"] == "2026.11.765"
            and value["required_checks_passed"] is True,
            "producer_identity",
        )
        require(
            value["protocol"] == protocol()
            and value["protocol_hash"] == canonical_hash(protocol())
            and value["owned_statement_coverage"]["complete"]
            and value["owned_statement_coverage"]["percent"] == 100
            and bool(value["control_rows"])
            and all(r["passed"] for r in value["control_rows"]),
            "producer_contract",
        )
        refs.append(reference(path))
        operands = (
            value["source_artifact_hashes"]
            + value["code_config_hashes"]
            + value["raw_shard_hashes"]
        )
        for i, ref in enumerate(operands):
            d.update(artifact_field="original_operand", path=ref["path"], expected=ref["sha256"])
            d["observed"] = d["hash"] = sha256_file(Path(ref["path"]))
            require(d["observed"] == d["expected"], "original_operand_drift")
            refs.append(ref)
            if i % 100 == 0:
                progress("authenticate_operands", i, len(operands) - i)
        causal.disjoint(protocol(), methods.anchors())
        head_path = Path(value["raw_directory"]) / "heads.json"
        require(reference(head_path) in refs, "head_custody")
        heads = json.loads(head_path.read_bytes())
        atomic_json(raw / "original_producer.json", value)
    except (OSError, ValueError, KeyError, TypeError) as error:
        diagnostics.append(
            dict(
                d,
                error=str(error),
                check="missing_file"
                if isinstance(error, OSError)
                else "missing_field"
                if isinstance(error, KeyError)
                else d["check"],
            )
        )
    return dict(
        external_ready=not diagnostics,
        diagnostics=diagnostics,
        references=refs,
        heads=heads,
        resources=dict(
            passed=shutil.disk_usage(raw).free > 512 * 1024**2,
            total_budget_s=4200,
            hard_cap_s=4800,
            pretrained_loads=0,
        ),
    )


def census(reason: str) -> Json:
    """Retain each planned comparison even when a prerequisite prevents execution."""
    rows = [
        dict(
            unit_id=u["unit_id"] + (f"/checkpoint{c}" if c is not None else ""),
            cluster_id=p["stream_id"],
            seed=p["seed"],
            arm=a,
            split=role,
            checkpoint=c,
            ordinal=i + 1,
            metric="brier",
            value=None,
            numerator=None,
            denominator=1,
            error=None,
            status="censored",
            missing_reason=reason,
            censor_reason=reason,
        )
        for p in protocol()["frozen_evaluation_manifest"]
        for a in methods.ARMS
        for role, c in [("online", None)] + [("retention", c) for c in (0, 24, 48)]
        for i, u in enumerate(p[role])
    ]
    return dict(
        rows=rows,
        **{key: [] for key in FIELDS},
        numerical_training_calls=0,
        applied_gradient_count=0,
        stream_refs=[],
        attempt_rows=[],
    )


def trajectory(raw: Path, plan: Json, params: Json, *, restart: bool) -> Json:
    """Seal before release and keep retention outside the store's training interface."""
    result: Json = dict(rows=[], **{key: [] for key in FIELDS})
    anchor = methods.anchors()

    def tagged(row: Json, arm: str) -> Json:
        return dict(row, arm=arm, cluster_id=plan["stream_id"], seed=plan["seed"])

    class RecordedStore(Store):
        def commit(self, state: Json) -> None:
            super().commit(state)
            result["state_hashes"].append(
                tagged(
                    dict(
                        sequence=len(result["state_hashes"]) + 1,
                        state_hash=canonical_hash(state),
                        prediction_cursor=state["prediction_cursor"],
                        release_cursor=state["cursor"],
                        pending_count=len(state["pending"]),
                    ),
                    state["arm"],
                )
            )

        def train_release(self, state: Json, row: Json) -> None:
            x = np.asarray(methods.signed.vector(state["candidate"], state["visible_counts"]))
            matches = [
                dict(
                    source_id=r["group_id"],
                    split=r["split"],
                    distance=float(
                        np.sum(
                            (np.asarray(methods.signed.vector(state["candidate"], r["counts"])) - x)
                            ** 2
                        )
                    ),
                )
                for r in state["buffer"]
            ]
            scanned: list[Json] = []
            scan = gradient.scan

            def recorded_scan(params: Json, current: Json, buffer: list[Json]) -> list[Json]:
                candidates = scan(params, current, buffer)
                scanned.extend(candidates)
                return candidates

            with patch.object(gradient, "scan", recorded_scan):
                super().train_release(state, row)
            if state["arm"] != "frozen_verified":
                for saved, candidate in zip(
                    state["candidate_gradient_rows"][-len(scanned) :], scanned, strict=True
                ):
                    saved["gradient"] = candidate["gradient"]
                    saved["proposed_weights"] = candidate["proposed_weights"]
                result["replay_rows"].append(
                    tagged(
                        dict(
                            state["selection_rows"][-1],
                            buffer_matches=matches,
                            query_cursor=state["prediction_cursor"],
                        ),
                        state["arm"],
                    )
                )
            else:
                result["replay_rows"].append(
                    tagged(
                        dict(
                            released_id=row["group_id"],
                            selected_id=None,
                            scan_count=0,
                            applied_gradients=0,
                            status="no_op",
                        ),
                        state["arm"],
                    )
                )

    stores = {
        a: RecordedStore(raw / (a + ".json"), params, a, anchor, seed=plan["seed"])
        for a in methods.ARMS
    }
    values = [
        methods.old.world(
            methods.old.graphs()["online"][u["template"]],
            u["unit_id"] + "/",
            u["error_type"],
            u["world_seed"],
        )
        for u in plan["online"]
    ]

    def scored(
        arm: str, unit: Json, sealed: Json, truth: str, role: str, checkpoint: int | None
    ) -> Json:
        row = scoring.scored(plan, arm, unit, sealed, truth, role, checkpoint)
        return dict(
            row,
            metric="brier",
            value=row["metric"],
            numerator=row["metric"],
            sealed=sealed,
            censor_reason=None,
        )

    def retain(checkpoint: int) -> None:
        before = {a: sha256_file(s.path) for a, s in stores.items()}
        for i, unit in enumerate(plan["retention"]):
            value = methods.old.world(
                methods.old.graphs()["retention"][unit["template"]],
                unit["unit_id"] + "/",
                unit["error_type"],
                unit["world_seed"],
            )
            for arm, store in stores.items():
                observed = methods.signed.observe(value, store.state["templates"])
                sealed = dict(
                    ordinal=i + 1,
                    counts=observed["counts"],
                    prediction=methods.signed.probabilities(
                        store.state["committed"], observed["counts"]
                    ),
                    state_hash=canonical_hash(store.state),
                )
                row = scored(
                    arm,
                    unit,
                    sealed,
                    str(symbolic.decide(value)["decision"]),
                    "retention",
                    checkpoint,
                )
                result["rows"].append(row)
                result["retention_rows"].append(row)
        require(before == {a: sha256_file(s.path) for a, s in stores.items()}, "retention_mutation")
        result["control_rows"].append(
            dict(check="retention_read_only", checkpoint=checkpoint, passed=True)
        )

    retain(0)
    released = 0
    for tick in range(1, 49):
        unit = plan["online"][tick - 1]
        for arm, store in stores.items():
            sealed = store.predict(unit["unit_id"], values[tick - 1], tick, tick + 8)
            result["prediction_rows"].append(
                tagged(dict(sealed, sealed_sequence=len(result["state_hashes"])), arm)
            )
        # Evaluator truth stays outside the store until the ordered due release.
        truth = str(symbolic.decide(values[tick - 1])["decision"])
        result["label_call_rows"].append(
            dict(
                group_id=unit["unit_id"],
                role="private_evaluator",
                tick=tick,
                after_sequence=len(result["state_hashes"]),
            )
        )
        for arm in stores:
            sealed = result["prediction_rows"][-len(stores) + list(stores).index(arm)]
            result["rows"].append(scored(arm, unit, sealed, truth, "online", None))
        if released < 40 and tick >= released + 16:
            batch = [
                (plan["online"][i]["unit_id"], values[i]) for i in range(released, released + 8)
            ]
            for arm, store in stores.items():
                before = list(store.state["templates"])
                sequence = len(result["state_hashes"])
                store.release(batch, tick)
                result["feedback_rows"] += [
                    tagged(dict(r, release_tick=tick, after_sequence=sequence), arm)
                    for r in store.state["released"][-8:]
                ]
                result["template_rows"].append(
                    tagged(
                        dict(
                            release_tick=tick,
                            source_ids=[g for g, _ in batch],
                            before=before,
                            after=deepcopy(store.state["templates"]),
                        ),
                        arm,
                    )
                )
            released += 8
        if tick in (24, 48):
            retain(tick)
        if tick == 24:
            checkpoint_states = {a: deepcopy(s.state) for a, s in stores.items()}
            atomic_json(raw / "query24_states.json", checkpoint_states)
            result["query24_state_hashes"] = {
                a: canonical_hash(s) for a, s in checkpoint_states.items()
            }
        if restart and tick == 24:
            for arm, store in list(stores.items()):
                restart_state = deepcopy(store.state)
                resumed = RecordedStore(store.path, {}, arm, anchor, seed=plan["seed"])
                require(resumed.state == restart_state, "restart_drift")
                result["restart_rows"].append(
                    tagged(
                        dict(
                            query=24,
                            before=canonical_hash(restart_state),
                            after=canonical_hash(resumed.state),
                            pending=deepcopy(restart_state["pending"]),
                            complete_state_equal=True,
                        ),
                        arm,
                    )
                )
                stores[arm] = resumed
        atomic_json(raw / "progress.json", result)
        if tick % 8 == 0:
            progress("stream_" + plan["stream_id"], tick, 48 - tick)
    states = {a: s.state for a, s in stores.items()}
    result.update(states=states, plan=plan)
    for arm, state in states.items():
        for target, source in (
            ("gradient_ledger", "gradient_ledger"),
            ("promotion_rows", "promotions"),
        ):
            result[target] += [tagged(r, arm) for r in state[source]]
        result["pending_feedback_rows"] += [tagged(r, arm) for r in state["pending"].values()]
        result["candidate_gradient_rows"] += [
            tagged(r, arm) for r in state["candidate_gradient_rows"]
        ]
    result["numerical_training_calls"] = sum(s["evaluated_gradients"] for s in states.values())
    result["applied_gradient_count"] = sum(s["training_calls"] for s in states.values())
    return result


def run_stream(raw: Path, plan: Json, params: Json) -> Json:
    """Compare exact restarts and private future-label/retention mutations on every arm."""
    progress("trajectory_before_" + plan["stream_id"], 0, 3)
    result = trajectory(raw, plan, params, restart=True)
    progress("uninterrupted_before_" + plan["stream_id"], 1, 2)
    control = trajectory(raw / "private_control", plan, params, restart=False)
    tail_hashes = {r["evidence_hash"] for r in result["pending_feedback_rows"]}
    decide = symbolic.decide

    def altered(value: Json) -> Json:
        answer: Json = decide(value)
        if canonical_hash(value) in tail_hashes:
            return dict(answer, decision="refute" if answer["decision"] == "support" else "support")
        return answer

    changed_plan = deepcopy(plan)
    for unit in changed_plan["retention"]:
        unit["world_seed"] ^= 7658953
    progress("causal_mutations_before_" + plan["stream_id"], 2, 1)
    with patch.object(symbolic, "decide", altered):
        mutated = trajectory(raw / "private_mutations", changed_plan, params, restart=False)
    for arm in methods.ARMS:
        equal = result["states"][arm] == control["states"][arm]
        checkpoint_equal = (
            result["query24_state_hashes"][arm] == control["query24_state_hashes"][arm]
        )

        def seals(r: Json) -> list[Json]:
            return [
                {k: v for k, v in p.items() if k != "sealed_sequence"}
                for p in r["prediction_rows"]
                if p["arm"] == arm
            ]

        predictions_equal = seals(result) == seals(control)
        restart_row = next(r for r in result["restart_rows"] if r["arm"] == arm)
        restart_row.update(
            subsequent_predictions_equal=predictions_equal,
            uninterrupted_complete_state_equal=equal,
            uninterrupted_query24_state_equal=checkpoint_equal,
            uninterrupted_state_hash=canonical_hash(control["states"][arm]),
        )
        selections_equal = (
            result["states"][arm]["selection_rows"] == mutated["states"][arm]["selection_rows"]
        )
        result["control_rows"] += [
            dict(
                check="exact_restart_" + arm,
                passed=equal and checkpoint_equal and predictions_equal,
            ),
            dict(
                check="future_label_isolation",
                arm=arm,
                passed=seals(result) == seals(mutated) and selections_equal,
            ),
            dict(
                check="retention_isolation",
                arm=arm,
                passed=result["states"][arm] == mutated["states"][arm],
            ),
        ]
        for scope, private in (
            ("uninterrupted", control),
            ("future_labels_and_retention", mutated),
        ):
            state = private["states"][arm]
            result["private_control_rows"].append(
                dict(
                    arm=arm,
                    cluster_id=plan["stream_id"],
                    scope=scope,
                    state_hash=canonical_hash(state),
                    training_calls=state["training_calls"],
                    evaluated_gradients=state["evaluated_gradients"],
                )
            )
    result["control_rows"] += selector_controls(params)
    result["control_rows"].append(
        dict(
            check="frozen_head_unchanged",
            passed=result["states"]["frozen_verified"]["committed"] == params,
        )
    )
    result["numerical_training_calls"] += (
        control["numerical_training_calls"] + mutated["numerical_training_calls"]
    )
    result["applied_gradient_count"] += (
        control["applied_gradient_count"] + mutated["applied_gradient_count"]
    )
    result["stream_checksum"] = canonical_hash(result)
    verify(result, plan)
    progress("trajectory_after_" + plan["stream_id"], 3, 0)
    return result


def verify(result: Json, plan: Json) -> None:
    """Reject score, schedule, replay custody and owned control failures."""
    require(
        result["stream_checksum"]
        == canonical_hash({k: v for k, v in result.items() if k != "stream_checksum"}),
        "stream_checksum",
    )
    require(
        result["plan"] == plan
        and len(result["rows"]) == 1152
        and len(result["prediction_rows"]) == 288
        and len(result["feedback_rows"]) == 240
        and len(result["retention_rows"]) == 864
        and len(result["pending_feedback_rows"]) == 48,
        "stream_inventory",
    )
    for row in result["rows"]:
        expected = scoring.scored(
            plan,
            row["arm"],
            dict(unit_id=row["unit_id"].split("/checkpoint")[0]),
            row["sealed"],
            row["truth"],
            row["split"],
            row["checkpoint"],
        )
        require(
            row["value"] == row["numerator"] == expected["metric"]
            and row["denominator"] == 1
            and row["error"] == expected["error"],
            "row_score",
        )
    predictions = {(p["arm"], p["group_id"]): p for p in result["prediction_rows"]}
    for release in result["feedback_rows"]:
        seal = predictions[release["arm"], release["group_id"]]
        require(
            release["release_tick"] >= seal["due"]
            and seal["sealed_sequence"] <= release["after_sequence"]
            and all(release[k] == seal[k] for k in seal if k in release),
            "label_before_seal",
        )
    states = result["states"]
    for arm, state in states.items():
        require(
            state["cursor"] == 40
            and state["prediction_cursor"] == 48
            and len(state["pending"]) == 8
            and state["buffer"] == [r for r in state["released"] if r["split"] == "train"][-32:]
            and state["templates"] == states[methods.ARMS[0]]["templates"]
            and all(
                r["label"] == r["true_label"] for r in state["released"] if r["split"] == "guard"
            ),
            "release_or_buffer_boundary",
        )
        expected_calls = (
            0
            if arm == "frozen_verified"
            else 2 * sum(r["split"] == "train" for r in state["released"])
        )
        require(
            state["training_calls"] == len(state["gradient_ledger"]) == expected_calls,
            "two_gradient_budget",
        )
    require(
        len({s["training_calls"] for a, s in states.items() if a != "frozen_verified"}) == 1
        and len({s["guard_scores"] for s in states.values()}) == 1
        and sum(s["training_calls"] for s in states.values()) == len(result["gradient_ledger"])
        and result["applied_gradient_count"]
        == len(result["gradient_ledger"])
        + sum(r["training_calls"] for r in result["private_control_rows"]),
        "matched_work",
    )
    require(
        len({s["evaluated_gradients"] for a, s in states.items() if a != "frozen_verified"}) == 1
        and result["numerical_training_calls"]
        == sum(s["evaluated_gradients"] for s in states.values())
        + sum(r["evaluated_gradients"] for r in result["private_control_rows"])
        and len(result["candidate_gradient_rows"])
        == sum(len(s["candidate_gradient_rows"]) for s in states.values()),
        "matched_scan_work",
    )
    require(all(r["passed"] for r in result["control_rows"]), "owned_control_failure")


def recover(raw: Path, reason: str) -> Json:
    """Preserve finished comparisons and actual charged work after an owned interruption."""
    result = census(reason)
    completed: dict[tuple[str, str], Json] = {}
    for plan in protocol()["frozen_evaluation_manifest"]:
        path = raw / (plan["stream_id"] + ".json")
        attempt = raw / ("attempt-" + plan["stream_id"] + ".json")
        if attempt.exists():
            result["attempt_rows"].append(
                dict(json.loads(attempt.read_bytes()), cluster_id=plan["stream_id"])
            )
        if path.exists():
            stream = json.loads(path.read_bytes())
            result["stream_refs"].append(reference(path))
            for key in FIELDS + ["numerical_training_calls", "applied_gradient_count"]:
                result[key] += stream[key]
            completed.update({(r["unit_id"], r["arm"]): r for r in stream["rows"]})
        elif attempt.exists():
            directory = raw / "states" / plan["stream_id"]
            partial = directory / "progress.json"
            if partial.exists():
                saved = json.loads(partial.read_bytes())
                completed.update({(r["unit_id"], r["arm"]): r for r in saved["rows"]})
                for key in FIELDS:
                    result[key] += saved[key]
            for state_path in directory.glob("**/*.json"):
                if state_path.stem in methods.ARMS:
                    state = json.loads(state_path.read_bytes())
                    calls = state["training_calls"]
                    result["numerical_training_calls"] += state["evaluated_gradients"]
                    result["applied_gradient_count"] += calls
                    if state_path.parent == directory:
                        for target, source in (
                            ("gradient_ledger", "gradient_ledger"),
                            ("promotion_rows", "promotions"),
                            ("candidate_gradient_rows", "candidate_gradient_rows"),
                        ):
                            result[target] += [
                                dict(
                                    r,
                                    arm=state["arm"],
                                    cluster_id=plan["stream_id"],
                                    seed=plan["seed"],
                                )
                                for r in state[source]
                            ]
                        result["pending_feedback_rows"] += [
                            dict(
                                r, arm=state["arm"], cluster_id=plan["stream_id"], seed=plan["seed"]
                            )
                            for r in state["pending"].values()
                        ]
                    else:
                        result["private_control_rows"].append(
                            dict(
                                arm=state["arm"],
                                cluster_id=plan["stream_id"],
                                state_hash=canonical_hash(state),
                                training_calls=calls,
                                evaluated_gradients=state["evaluated_gradients"],
                            )
                        )
    failed = {r["cluster_id"] for r in result["attempt_rows"] if r["status"] == "failed"}
    result["rows"] = [
        completed.get(
            (r["unit_id"], r["arm"]), dict(r, status="failed") if r["cluster_id"] in failed else r
        )
        for r in result["rows"]
    ]
    return result


def measure(raw: Path, heads: Json) -> Json:
    """Execute all24 sealed streams without selecting seeds or changing the protocol."""
    began = time.monotonic()
    atomic_json(raw / "heads.json", heads)
    for i, plan in enumerate(protocol()["frozen_evaluation_manifest"]):
        require(time.monotonic() - began < 2400, "measurement_deadline")
        progress("evaluation_before", i, 24 - i)
        attempt = raw / ("attempt-" + plan["stream_id"] + ".json")
        atomic_json(attempt, dict(status="attempted", seed=plan["seed"]))
        try:
            result = run_stream(
                raw / "states" / plan["stream_id"], plan, heads[str(plan["initial_head_seed"])]
            )
            atomic_json(raw / (plan["stream_id"] + ".json"), result)
        except (OSError, ValueError, RuntimeError, KeyError) as error:
            atomic_json(attempt, dict(status="failed", seed=plan["seed"], error=str(error)))
            raise
        atomic_json(attempt, dict(status="complete", seed=plan["seed"]))
        progress("evaluation_after", i + 1, 23 - i)
    return recover(raw, "unstarted")
