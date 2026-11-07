"""REQ-REPORT-8909: retain the complete frozen guarded-update evaluation.

The exact verifier supplies delayed constructed labels. All arms pay the same
work; only deployment and the registered permutation distinguish them.
"""

from copy import deepcopy
import json
from pathlib import Path
import random
import shutil
import time
from types import SimpleNamespace
from typing import Any

from carnot.reporting import guarded_update_methods_8905 as methods
from carnot.reporting import evidence_set_methods_8768 as symbolic
from carnot.reporting.causal_learning_audit_8810 import require, score
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash, sha256_file
from carnot.reporting.v746_retained_evidence import reference
from carnot.reporting.v748_collection_contract import namespace

Json = dict[str, Any]
__all__ = ["require"]
g, signed, old = methods.g, methods.signed, methods.old
ROOT, SEED = methods.ROOT, methods.SEED
NAME, TASK = "experiment_8909_v762_guarded_causal_streams", "exp8909-guarded-causal-streams"
CLI = "scripts/experiments/" + NAME + ".py"
OWNED = [
    "python/carnot/reporting/guarded_causal_streams_8909.py",
    "python/carnot/reporting/guarded_causal_runner_8909.py",
    CLI,
]
TESTS = ["tests/python/test_guarded_causal_streams_8909.py"]
SOURCE = methods.NAME
PIN = "sha256:acbd0cd68c70e51811c36897180a2dc302341f9fa8fa2f381c3c6e642ab16779"
FIELDS = [
    "prediction_rows",
    "promotion_rows",
    "retention_rows",
    "feedback_rows",
    "arm_work_ledger",
    "state_hashes",
    "label_call_rows",
    "control_rows",
]


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Flush real counts while the fixed deadline continues to run."""
    print(f"[exp8909] phase={phase} completed={completed} pending={pending}", flush=True)


def protocol() -> Json:
    """Reuse the producer's already frozen future recipes without generating worlds."""
    return dict(methods.protocol())


def disjoint(p: Json, anchors: list[Json]) -> None:
    """Reject reused groups or world seeds before opening any evaluation label."""
    pilots = [u for s in p["pilot_manifest"] for role in ("online", "retention") for u in s[role]]
    future = [
        u
        for s in p["frozen_evaluation_manifest"]
        for role in ("online", "retention")
        for u in s[role]
    ]
    groups = {u["unit_id"] for u in pilots} | {a["group_id"] for a in anchors}
    seeds = {u["world_seed"] for u in pilots} | {SEED + i for i in range(16)}
    require(
        not groups & {u["unit_id"] for u in future}
        and not seeds & {u["world_seed"] for u in future}
        and len({u["unit_id"] for u in future}) == len(future)
        and len({u["world_seed"] for u in future}) == len(future),
        "development_evaluation_overlap",
    )


def preconditions(raw: Path, sources: Path) -> Json:
    """Authenticate both integer gates and original operand bytes before expensive work."""
    raw.mkdir(parents=True, exist_ok=True, mode=0o700)
    path = sources / (SOURCE + ".json")
    d: Json = dict(
        check="missing_file",
        upstream=SOURCE,
        artifact_field="guarded_methods_ready_score",
        expected=1,
        observed=None,
        path=str(path),
        hash=None,
    )
    refs: list[Json] = []
    checks: list[Json] = []
    diagnostics: list[Json] = []
    heads: Json = {}
    try:
        d["hash"] = sha256_file(path)
        d["check"] = "rejected_producer"
        value = json.loads(path.read_bytes())
        require(isinstance(value, dict), "producer_object")
        for field in ("guarded_methods_ready_score", "candidate_ready_score"):
            d.update(check="missing_field", artifact_field=field, expected=1, observed=None)
            observed = value[field]
            d.update(
                check="observed_zero" if observed == 0 else "rejected_producer", observed=observed
            )
            require(type(observed) is int and observed == 1, "upstream_gate")
            checks.append(dict(d, check="passed"))
        d.update(
            check="rejected_producer",
            artifact_field="producer_bytes",
            expected=PIN,
            observed=d["hash"],
        )
        require(
            d["hash"] == PIN
            and value["experiment_id"] == 8905
            and value["task_id"] == methods.TASK
            and value["milestone"] == "2026.11.762"
            and value["required_checks_passed"] is True,
            "producer_identity",
        )
        require(
            value["protocol"] == protocol()
            and value["frozen_evaluation_manifest"] == protocol()["frozen_evaluation_manifest"]
            and value["owned_statement_coverage"]["complete"]
            and value["owned_statement_coverage"]["percent"] == 100
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
            if i % 50 == 0:
                progress("authenticate_operands", i, len(operands) - i)
        disjoint(value["protocol"], value["anchor_manifest"])
        require(value["anchor_manifest"] == methods.anchors(), "anchor_drift")
        head_path = Path(value["raw_directory"]) / "heads.json"
        require(reference(head_path) in refs, "initial_head_custody")
        heads = json.loads(head_path.read_bytes())
        shutil.copyfile(path, raw / "original_producer.json")
        checks.append(dict(check="original_bytes_and_disjoint_manifest", passed=True))
    except (OSError, ValueError, KeyError, TypeError) as error:
        diagnostics.append(
            dict(
                d,
                check="missing_file"
                if isinstance(error, OSError)
                else "missing_field"
                if isinstance(error, KeyError)
                else d["check"],
                error=str(error),
            )
        )
    return dict(
        external_ready=not diagnostics,
        diagnostics=diagnostics,
        checks=checks,
        references=refs,
        heads=heads,
        protocol_hash=canonical_hash(protocol()),
        resources=dict(
            passed=all(
                (ROOT / ".venv/bin" / n).is_file() for n in ("python", "pytest", "ruff", "mypy")
            )
            and shutil.disk_usage(raw).free > 512 * 1024**2,
            pilot_total_limit_s=4200,
            hard_cap_s=4800,
            pretrained_loads=0,
        ),
    )


def census(reason: str) -> Json:
    """Keep every scheduled outcome even when external evidence prevents execution."""
    rows = [
        dict(
            unit_id=u["unit_id"] + (f"/checkpoint{c}" if c is not None else ""),
            cluster_id=p["stream_id"],
            seed=p["seed"],
            arm=a,
            split=role,
            checkpoint=c,
            ordinal=i + 1,
            denominator=1,
            metric=None,
            error=None,
            status="censored",
            missing_reason=reason,
        )
        for p in protocol()["frozen_evaluation_manifest"]
        for a in g.ARMS
        for role, checkpoints in (("online", [None]), ("retention", [0, 24, 48]))
        for c in checkpoints
        for i, u in enumerate(p[role])
    ]
    return dict(
        rows=rows,
        **{k: [] for k in FIELDS},
        numerical_training_calls=0,
        stream_refs=[],
        attempt_rows=[],
    )


def run_stream(raw: Path, plan: Json, params: Json, *, restart: bool = True) -> Json:
    """Seal predictions before due labels and keep retention outside every store interface."""
    runtime = namespace(g, "guarded8909_store")
    transactions: list[Json] = []
    calls: list[Json] = []
    rows: list[Json] = []
    predictions: list[Json] = []
    feedback: list[Json] = []
    active: Json = {}
    anchor = methods.anchors()

    def decide(value: Json) -> Json:
        calls.append(
            dict(
                arm=active["arm"],
                tick=active["tick"],
                evidence_hash=canonical_hash(value),
                after_sequence=len(transactions),
            )
        )
        return dict(symbolic.decide(value))

    runtime.symbolic = SimpleNamespace(decide=decide)
    original = runtime.Store

    def commit(self: Any, state: Json) -> None:
        original.commit(self, state)
        transactions.append(
            dict(
                cluster_id=plan["stream_id"],
                seed=plan["seed"],
                arm=state["arm"],
                sequence=len(transactions) + 1,
                state_hash=canonical_hash(state),
                feedback_cursor=state["cursor"],
                prediction_cursor=state["prediction_cursor"],
                pending_count=len(state["pending"]),
                parameter_hash=canonical_hash(state["committed"]),
                candidate_hash=canonical_hash(state["candidate"]),
                template_hash=canonical_hash(state["templates"]),
                committed_version=state["committed_version"],
                candidate_version=state["candidate_version"],
            )
        )
        atomic_json(raw / "transactions.json", dict(rows=transactions))

    store_type = type("RecordedStore", (original,), {"commit": commit})
    stores = {a: store_type(raw / (a + ".json"), params, a, anchor) for a in g.ARMS}
    controls: list[Json] = []
    values = [
        old.world(
            old.graphs()["online"][u["template"]],
            u["unit_id"] + "/",
            u["error_type"],
            u["world_seed"],
        )
        for u in plan["online"]
    ]
    due = [i + 1 + delay for i, delay in enumerate(plan["delays"])]

    def retain(checkpoint: int) -> None:
        before = {a: sha256_file(s.path) for a, s in stores.items()}
        for i, unit in enumerate(plan["retention"]):
            value = old.world(
                old.graphs()["retention"][unit["template"]],
                unit["unit_id"] + "/",
                unit["error_type"],
                unit["world_seed"],
            )
            for arm, store in stores.items():
                observed = signed.observe(value, store.state["templates"])
                sealed = dict(
                    ordinal=i + 1,
                    prediction=signed.probabilities(store.state["committed"], observed["counts"]),
                    counts=observed["counts"],
                    state_hash=canonical_hash(store.state),
                    parameter_hash=canonical_hash(store.state["committed"]),
                    template_hash=canonical_hash(store.state["templates"]),
                )
                row = methods.scored(
                    plan,
                    arm,
                    unit,
                    sealed,
                    str(symbolic.decide(value)["decision"]),
                    "retention",
                    checkpoint,
                )
                rows.append(dict(row, sealed=sealed))
        require(
            before == {a: sha256_file(s.path) for a, s in stores.items()},
            "retention_state_mutation",
        )
        controls.append(dict(check="retention_read_only", checkpoint=checkpoint, passed=True))
        atomic_json(
            raw / "progress.json",
            dict(
                rows=rows,
                prediction_rows=predictions,
                feedback_rows=feedback,
                label_call_rows=calls,
            ),
        )

    retain(0)
    released = 0
    for tick in range(1, max(due) + 1):
        if tick <= 48:
            unit = plan["online"][tick - 1]
            for arm, store in stores.items():
                sealed = store.predict(unit["unit_id"], values[tick - 1], tick, due[tick - 1])
                predictions.append(
                    dict(
                        sealed,
                        arm=arm,
                        cluster_id=plan["stream_id"],
                        seed=plan["seed"],
                        sealed_sequence=len(transactions),
                    )
                )
            atomic_json(
                raw / "progress.json",
                dict(
                    rows=rows,
                    prediction_rows=predictions,
                    feedback_rows=feedback,
                    label_call_rows=calls,
                ),
            )
        while released < 48 and tick >= max(due[released : released + 8]):
            batch = [
                (plan["online"][i]["unit_id"], values[i]) for i in range(released, released + 8)
            ]
            for arm, store in stores.items():
                active.update(arm=arm, tick=tick)
                store.release(batch, tick)
                for record, unit in zip(
                    store.state["released"][-8:],
                    plan["online"][released : released + 8],
                    strict=True,
                ):
                    feedback.append(
                        dict(
                            record,
                            arm=arm,
                            cluster_id=plan["stream_id"],
                            seed=plan["seed"],
                            release_tick=tick,
                        )
                    )
                    rows.append(
                        methods.scored(
                            plan, arm, unit, record, record["true_label"], "online", None
                        )
                    )
            released += 8
            atomic_json(
                raw / "progress.json",
                dict(
                    rows=rows,
                    prediction_rows=predictions,
                    feedback_rows=feedback,
                    label_call_rows=calls,
                ),
            )
        if tick == 24:
            retain(24)
        if restart:
            for arm, store in list(stores.items()):
                resumed = store_type(store.path, {}, arm, anchor)
                require(resumed.state == store.state, "restart_drift")
                stores[arm] = resumed
        if tick % 16 == 0:
            progress("stream_" + plan["stream_id"], tick, max(due) - tick)
    retain(48)
    states = {a: s.state for a, s in stores.items()}
    result = dict(
        rows=rows,
        states=states,
        plan=plan,
        prediction_rows=predictions,
        feedback_rows=feedback,
        label_call_rows=calls,
        state_hashes=transactions,
        retention_rows=[r for r in rows if r["split"] == "retention"],
        promotion_rows=[
            dict(r, arm=a, cluster_id=plan["stream_id"], seed=plan["seed"])
            for a, s in states.items()
            for r in s["promotions"]
        ],
        arm_work_ledger=[
            dict(
                arm=a,
                cluster_id=plan["stream_id"],
                seed=plan["seed"],
                training_calls=s["training_calls"],
                guard_scores=s["guard_scores"],
                feedback_cursor=s["cursor"],
                prediction_cursor=s["prediction_cursor"],
            )
            for a, s in states.items()
        ],
        numerical_training_calls=sum(s["training_calls"] for s in states.values()),
        control_rows=controls + [dict(check="restart_equivalence", passed=True)],
    )
    result["stream_checksum"] = canonical_hash(result)
    verify(result, plan)
    return result


def verify(result: Json, plan: Json) -> None:
    """Recompute row scores and enforce schedule, work, permutation and deployment invariants."""
    require(
        canonical_hash({k: v for k, v in result.items() if k != "stream_checksum"})
        == result["stream_checksum"],
        "stream_checksum",
    )
    require(result["plan"] == plan and len(result["rows"]) == 768, "stream_inventory")
    for row in result["rows"]:
        sealed = row.get("sealed") or next(
            r
            for r in result["prediction_rows"]
            if r["arm"] == row["arm"] and r["ordinal"] == row["ordinal"]
        )
        require(
            all(row[k] == v for k, v in score(sealed["prediction"], row["truth"], 0.5).items()),
            "row_score",
        )
    require(
        len(result["prediction_rows"]) == len(result["feedback_rows"]) == 192
        and len(result["retention_rows"]) == 576,
        "causal_inventory",
    )
    for prediction in result["prediction_rows"]:
        release = next(
            r
            for r in result["feedback_rows"]
            if (r["arm"], r["group_id"]) == (prediction["arm"], prediction["group_id"])
        )
        call = next(
            r
            for r in result["label_call_rows"]
            if (r["arm"], r["evidence_hash"]) == (prediction["arm"], prediction["evidence_hash"])
        )
        require(
            release["release_tick"] >= prediction["due"]
            and prediction["sealed_sequence"] <= call["after_sequence"]
            and all(
                prediction[k] == release[k]
                for k in ("prediction", "counts", "parameter_hash", "template_hash", "split")
            ),
            "label_before_seal",
        )
    states = result["states"]
    require(
        len({s["training_calls"] for s in states.values()})
        == len({s["guard_scores"] for s in states.values()})
        == 1,
        "matched_work",
    )
    for arm, state in states.items():
        require(
            state["cursor"] == 48
            and not state["pending"]
            and state["templates"] == states[g.ARMS[0]]["templates"]
            and state["splits"] == states[g.ARMS[0]]["splits"],
            "schedule_alignment",
        )
        for i, block in enumerate(state["blocks"]):
            targets = (
                random.Random(SEED + 8 * i).sample(block["labels"], 8)
                if arm == "guarded_permuted"
                else block["labels"]
            )
            require(
                block["targets"] == targets and sorted(block["labels"]) == sorted(targets),
                "whole_group_permutation",
            )
        for candidate in state["promotions"]:
            require(
                not set(candidate["training_groups"]) & set(candidate["guard_groups"])
                and all(g.split(v) == "train" for v in candidate["training_groups"])
                and all(g.split(v) == "guard" for v in candidate["guard_groups"])
                and (
                    candidate["after"] == candidate["candidate"]
                    if candidate["promoted"]
                    else candidate["after"] == candidate["before"]
                ),
                "atomic_promotion",
            )
    require(all(r["passed"] for r in result["control_rows"]), "owned_control_failure")


def recover(raw: Path, reason: str) -> Json:
    """Retain completed streams, failed attempts and durable pending work without replacement."""
    result = census(reason)
    completed: dict[tuple[str, str], Json] = {}
    finished = set()
    for path in sorted(raw.glob("v762-evaluation-*.json")):
        stream = json.loads(path.read_bytes())
        finished.add(stream["plan"]["stream_id"])
        completed.update({(r["unit_id"], r["arm"]): r for r in stream["rows"]})
        for key in FIELDS:
            result[key].extend(stream[key])
        result["numerical_training_calls"] += stream["numerical_training_calls"]
        result["stream_refs"].append(reference(path))
    result["pending_feedback_rows"] = []
    for directory in sorted((raw / "states").glob("*")):
        if directory.name in finished:
            continue
        plan = next(
            p for p in protocol()["frozen_evaluation_manifest"] if p["stream_id"] == directory.name
        )
        progress_path = directory / "progress.json"
        if progress_path.is_file():
            partial = json.loads(progress_path.read_bytes())
            completed.update({(r["unit_id"], r["arm"]): r for r in partial["rows"]})
            for key in ("prediction_rows", "feedback_rows", "label_call_rows"):
                result[key].extend(partial[key])
            result["retention_rows"].extend(r for r in partial["rows"] if r["split"] == "retention")
        for arm in g.ARMS:
            path = directory / (arm + ".json")
            if path.is_file():
                state = json.loads(path.read_bytes())
                result["numerical_training_calls"] += state["training_calls"]
                result["promotion_rows"].extend(
                    dict(r, arm=arm, cluster_id=directory.name, seed=plan["seed"])
                    for r in state["promotions"]
                )
                result["pending_feedback_rows"].extend(
                    dict(r, arm=arm, cluster_id=directory.name, seed=plan["seed"])
                    for r in state["pending"].values()
                )
                for record in state["released"]:
                    unit = plan["online"][record["ordinal"] - 1]
                    completed.setdefault(
                        (unit["unit_id"], arm),
                        methods.scored(
                            plan, arm, unit, record, record["true_label"], "online", None
                        ),
                    )
                    if not any(
                        r["arm"] == arm and r["group_id"] == record["group_id"]
                        for r in result["feedback_rows"]
                    ):
                        block = next(
                            b for b in state["blocks"] if record["group_id"] in b["groups"]
                        )
                        result["feedback_rows"].append(
                            dict(
                                record,
                                arm=arm,
                                cluster_id=directory.name,
                                seed=plan["seed"],
                                release_tick=block["tick"],
                            )
                        )
        journal = directory / "transactions.json"
        if journal.is_file():
            result["state_hashes"].extend(json.loads(journal.read_bytes())["rows"])
    result["attempt_rows"] = [
        dict(json.loads(p.read_bytes()), cluster_id=p.stem.removeprefix("attempt-"))
        for p in sorted(raw.glob("attempt-*.json"))
    ]
    failed = {r["cluster_id"] for r in result["attempt_rows"] if r["status"] == "failed"}
    result["rows"] = [
        completed.get(
            (r["unit_id"], r["arm"]), dict(r, status="failed") if r["cluster_id"] in failed else r
        )
        for r in result["rows"]
    ]
    qualification = raw / "qualification.json"
    result["qualification_training_calls"] = 0
    if qualification.is_file():
        value = json.loads(qualification.read_bytes())
        result["qualification_training_calls"] = value["numerical_training_calls"]
        result["numerical_training_calls"] += value["numerical_training_calls"]
        result["control_rows"].extend(value["control_rows"])
    return result


def measure(raw: Path, pre: Json) -> Json:
    """Check a development replay pair, then execute every frozen evaluation stream once."""
    raw.mkdir(parents=True, exist_ok=True, mode=0o700)
    p, began = protocol(), time.monotonic()
    disjoint(p, methods.anchors())
    atomic_json(raw / "protocol.json", p)
    pilot = p["pilot_manifest"][0]
    params = pre["heads"][str(pilot["seed"])]
    progress("qualification_pair_before", 0, 2)
    a = run_stream(raw / "qualification" / "restarted", pilot, params)
    b = run_stream(raw / "qualification" / "uninterrupted", pilot, params, restart=False)
    projected = (time.monotonic() - began) * 14 + 1200
    atomic_json(
        raw / "qualification.json",
        dict(
            numerical_training_calls=a["numerical_training_calls"] + b["numerical_training_calls"],
            control_rows=[
                dict(check="development_restart_pair", passed=a == b),
                dict(check="runtime_projection", passed=projected < 4200),
            ],
            projected_runtime_s=projected,
            restarted=a,
            uninterrupted=b,
        ),
    )
    progress("qualification_pair_after", 2, 0)
    require(a == b and projected < 4200, "owned_qualification_failure")
    for index, plan in enumerate(p["frozen_evaluation_manifest"]):
        require(time.monotonic() - began < 2400, "measurement_deadline")
        progress("evaluation_before", index, len(p["frozen_evaluation_manifest"]) - index)
        attempt = raw / ("attempt-" + plan["stream_id"] + ".json")
        atomic_json(attempt, dict(status="attempted", seed=plan["seed"]))
        try:
            result = run_stream(
                raw / "states" / plan["stream_id"], plan, pre["heads"][str(plan["seed"])]
            )
            atomic_json(raw / (plan["stream_id"] + ".json"), result)
        except (OSError, ValueError, RuntimeError, KeyError) as error:
            atomic_json(attempt, dict(status="failed", seed=plan["seed"], error=str(error)))
            raise
        atomic_json(attempt, dict(status="complete", seed=plan["seed"]))
        progress("evaluation_after", index + 1, len(p["frozen_evaluation_manifest"]) - index - 1)
    return recover(raw, "unstarted")
