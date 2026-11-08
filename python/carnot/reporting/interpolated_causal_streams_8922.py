"""REQ-REPORT-8922: measure fixed future streams without inferring benefit.

The original causal loop owns predictions and releases. This adapter selects
Exp8919's optimizer and records its durable sequences without changing guards.
"""

from copy import deepcopy
import json
from pathlib import Path
import random
import shutil
import time
from typing import Any

from carnot.reporting import evidence_set_methods_8768 as symbolic
from carnot.reporting import guarded_causal_streams_8909 as base
from carnot.reporting import interpolated_update_methods_8919 as methods
from carnot.reporting.causal_learning_audit_8810 import require, score
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash, sha256_file
from carnot.reporting.v746_retained_evidence import reference
from carnot.reporting.v748_collection_contract import namespace

__all__ = ["require", "methods"]
Json = dict[str, Any]
ROOT, SEED = methods.ROOT, 763200
NAME, TASK = (
    "experiment_8922_v763_interpolated_causal_streams",
    "exp8922-interpolated-causal-streams",
)
CLI = "scripts/experiments/" + NAME + ".py"
OWNED = [
    "python/carnot/reporting/interpolated_causal_streams_8922.py",
    "python/carnot/reporting/interpolated_causal_runner_8922.py",
    CLI,
]
TESTS = ["tests/python/test_interpolated_causal_streams_8922.py"]
SOURCE = methods.NAME
PIN = "sha256:ddc6d3cc5aecffabe0e99079833361e98f5085925ea6f30f3115b6109f7ab2ec"
FIELDS = base.FIELDS + [
    "release_rows",
    "gradient_call_ledger",
    "optimizer_sequences",
    "restart_rows",
    "scoring_rows",
]
protocol = methods.protocol


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Flush real counts so a supervisor can distinguish slow work from a stall."""
    print(f"[exp8922] phase={phase} completed={completed} pending={pending}", flush=True)


def preconditions(raw: Path, sources: Path) -> Json:
    """Authenticate both independent gates and the bytes used by the original producer."""
    raw.mkdir(parents=True, exist_ok=True, mode=0o700)
    path = sources / (SOURCE + ".json")
    diagnostic: Json = dict(
        check="missing_file",
        upstream=SOURCE,
        artifact_field="methods_ready_score",
        expected=1,
        observed=None,
        path=str(path),
        hash=None,
    )
    refs: list[Json] = []
    checks: list[Json] = []
    errors: list[Json] = []
    heads: Json = {}
    try:
        diagnostic["hash"] = sha256_file(path)
        diagnostic["check"] = "rejected_producer"
        value = json.loads(path.read_bytes())
        require(isinstance(value, dict), "producer_object")
        for field in ("methods_ready_score", "candidate_ready_score"):
            diagnostic.update(check="missing_field", artifact_field=field, observed=None)
            actual = value[field]
            diagnostic.update(
                check="observed_zero" if actual == 0 else "rejected_producer", observed=actual
            )
            require(type(actual) is int and actual == 1, "upstream_gate")
            checks.append(dict(diagnostic, check="passed"))
        diagnostic.update(
            check="rejected_producer",
            artifact_field="producer_bytes",
            expected=PIN,
            observed=diagnostic["hash"],
        )
        require(
            diagnostic["hash"] == PIN
            and value["experiment_id"] == 8919
            and value["task_id"] == methods.TASK
            and value["milestone"] == "2026.11.763"
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
        for index, ref in enumerate(operands):
            diagnostic.update(
                artifact_field="original_operand", path=ref["path"], expected=ref["sha256"]
            )
            diagnostic["observed"] = diagnostic["hash"] = sha256_file(Path(ref["path"]))
            require(diagnostic["observed"] == diagnostic["expected"], "original_operand_drift")
            refs.append(ref)
            if index % 100 == 0:
                progress("authenticate_operands", index, len(operands) - index)
        base.disjoint(protocol(), methods.anchors())
        fit = next(r for r in refs if Path(r["path"]).name == "fit.json")
        initial = json.loads(Path(fit["path"]).read_bytes())["heads"]["signed"]
        heads = {
            str(p["seed"]): initial[str(p["initial_head_seed"])]
            for p in protocol()["pilot_manifest"] + protocol()["frozen_evaluation_manifest"]
        }
        atomic_json(raw / "original_producer.json", value)
        atomic_json(raw / "heads.json", heads)
    except (OSError, ValueError, KeyError, TypeError, StopIteration) as error:
        errors.append(
            dict(
                diagnostic,
                error=str(error),
                check="missing_file"
                if isinstance(error, OSError)
                else "missing_field"
                if isinstance(error, (KeyError, StopIteration))
                else diagnostic["check"],
            )
        )
    return dict(
        external_ready=not errors,
        diagnostics=errors,
        checks=checks,
        references=refs,
        heads=heads,
        resources=dict(
            passed=shutil.disk_usage(raw).free > 512 * 1024**2,
            total_budget_s=4200,
            hard_cap_s=4800,
            pretrained_loads=0,
        ),
    )


def runtime() -> Any:
    """Keep the historical loop's globals private rather than changing its producer."""
    runner = namespace(base, "interpolated8922_loop")
    runner.protocol, runner.progress = protocol, progress
    runner.g = namespace(methods.g, "interpolated8922_store")
    runner.g.ARMS = methods.ARMS
    runner.FIELDS = FIELDS
    return runner


def census(reason: str) -> Json:
    """Keep every planned comparison even when no future label may be opened."""
    return dict(runtime().census(reason))


def verify(result: Json, plan: Json) -> None:
    """Check actual training-only permutation before reusing all original causal assertions."""
    require(
        result["stream_checksum"]
        == canonical_hash({k: v for k, v in result.items() if k != "stream_checksum"}),
        "stream_checksum",
    )
    for arm, state in result["states"].items():
        for index, block in enumerate(state["blocks"]):
            expected = list(block["labels"])
            training = [
                i for i, group in enumerate(block["groups"]) if methods.g.split(group) == "train"
            ]
            if arm == "interpolated_permuted":
                shuffled = random.Random(methods.g.SEED + index * 8).sample(
                    [expected[i] for i in training], len(training)
                )
                for i, target in zip(training, shuffled, strict=True):
                    expected[i] = target
            require(block["targets"] == expected, "training_only_permutation")
    for row in result["rows"]:
        sealed = row.get("sealed") or next(
            r
            for r in result["prediction_rows"]
            if r["arm"] == row["arm"] and r["ordinal"] == row["ordinal"]
        )
        require(row["prediction"] == sealed["prediction"], "prediction_drift")
    # The legacy validator expects true targets for these new arm names; actual targets were checked above.
    projected = deepcopy(result)
    for state in projected["states"].values():
        for block in state["blocks"]:
            block["targets"] = list(block["labels"])
    projected["stream_checksum"] = canonical_hash(
        {k: v for k, v in projected.items() if k != "stream_checksum"}
    )
    runtime().verify(projected, plan)
    require(result["release_rows"] == result["feedback_rows"], "release_inventory")
    require(
        sum(r["calls"] for r in result["gradient_call_ledger"])
        == result["numerical_training_calls"],
        "gradient_inventory",
    )


def run_stream(raw: Path, plan: Json, params: Json, *, restart: bool = True) -> Json:
    """Record fast/slow transactions while the original loop seals all causal predictions."""
    sequences: list[Json] = []
    restarts: list[Json] = []
    scoring: list[Json] = []

    runner = runtime()
    store_base = runner.g.Store

    def initialize(self: Any, path: Path, head: Json, arm: str, anchors: list[Json]) -> None:
        store_base.__init__(
            self,
            path,
            head,
            arm,
            anchors,
            optimizer=methods.opt.SCHEMA if arm.startswith("interpolated_") else "sgd",
        )
        if not head and self.state["prediction_cursor"] == 24:
            restarts.append(
                dict(
                    arm=arm,
                    query=24,
                    state_hash=canonical_hash(self.state),
                    pending=deepcopy(self.state["pending"]),
                    optimizer=deepcopy(self.state.get("optimizer")),
                )
            )

    def commit(self: Any, state: Json) -> None:
        store_base.commit(self, state)
        sequences.append(
            dict(
                arm=state["arm"],
                prediction_cursor=state["prediction_cursor"],
                release_cursor=state["cursor"],
                candidate_hash=canonical_hash(state["candidate"]),
                committed_hash=canonical_hash(state["committed"]),
                optimizer=deepcopy(state.get("optimizer")),
            )
        )
        atomic_json(raw / "optimizer_sequences.json", dict(rows=sequences))

    def release(self: Any, batch: list[tuple[str, Json]], tick: int) -> None:
        for group, value in batch:
            sealed = self.state["pending"][group]
            require(tick >= sealed["due"], "early_evaluator_label")
            truth = str(symbolic.decide(value)["decision"])
            scoring.append(
                dict(
                    arm=self.state["arm"],
                    cluster_id=plan["stream_id"],
                    group_id=group,
                    prediction=sealed["prediction"],
                    truth=truth,
                    scored_before_release=True,
                    release_tick=tick,
                    state_hash=canonical_hash(self.state),
                    **score(sealed["prediction"], truth, 0.5),
                )
            )
        atomic_json(raw / "scoring.json", dict(rows=scoring))
        store_base.release(self, batch, tick)

    runner.g.Store = type(
        "InterpolatedStore",
        (store_base,),
        {"__init__": initialize, "commit": commit, "release": release},
    )
    runner.namespace = lambda source, name: source
    runner.verify = lambda result, plan: None
    result: Json = runner.run_stream(raw, plan, params, restart=restart)
    result.update(
        release_rows=result["feedback_rows"],
        optimizer_sequences=sequences,
        restart_rows=restarts,
        scoring_rows=scoring,
        gradient_call_ledger=[
            dict(
                cluster_id=plan["stream_id"],
                arm=a,
                calls=s["training_calls"],
                guard_scores=s["guard_scores"],
                update_count=s.get("optimizer", {}).get("update_count", 0),
            )
            for a, s in result["states"].items()
        ],
    )
    result["stream_checksum"] = canonical_hash(
        {k: v for k, v in result.items() if k != "stream_checksum"}
    )
    verify(result, plan)
    return result


def recover(raw: Path, reason: str) -> Json:
    """Preserve partial seals, charged gradients and all unfinished cells after interruption."""
    result = census(reason)
    completed: dict[tuple[str, str], Json] = {}
    finished = set()
    for path in sorted(raw.glob("v763-evaluation-*.json")):
        saved = json.loads(path.read_bytes())
        finished.add(saved["plan"]["stream_id"])
        completed.update({(r["unit_id"], r["arm"]): r for r in saved["rows"]})
        for key in FIELDS:
            result[key].extend(saved[key])
        result["numerical_training_calls"] += saved["numerical_training_calls"]
        result["stream_refs"].append(reference(path))
    result["pending_feedback_rows"] = []
    for directory in sorted((raw / "states").glob("*")):
        if directory.name in finished:
            continue
        partial = directory / "progress.json"
        if partial.is_file():
            value = json.loads(partial.read_bytes())
            completed.update({(r["unit_id"], r["arm"]): r for r in value["rows"]})
            for key in ("prediction_rows", "feedback_rows", "label_call_rows"):
                result[key].extend(value[key])
            result["retention_rows"].extend(r for r in value["rows"] if r["split"] == "retention")
        for arm in methods.ARMS:
            path = directory / (arm + ".json")
            if path.is_file():
                state = json.loads(path.read_bytes())
                result["numerical_training_calls"] += state["training_calls"]
                result["promotion_rows"].extend(
                    dict(r, arm=arm, cluster_id=directory.name) for r in state["promotions"]
                )
                result["pending_feedback_rows"].extend(
                    dict(r, arm=arm, cluster_id=directory.name) for r in state["pending"].values()
                )
                result["gradient_call_ledger"].append(
                    dict(
                        cluster_id=directory.name,
                        arm=arm,
                        calls=state["training_calls"],
                        guard_scores=state["guard_scores"],
                        update_count=state.get("optimizer", {}).get("update_count", 0),
                    )
                )
        for filename, key in (
            ("transactions.json", "state_hashes"),
            ("optimizer_sequences.json", "optimizer_sequences"),
            ("scoring.json", "scoring_rows"),
        ):
            path = directory / filename
            if path.is_file():
                result[key].extend(json.loads(path.read_bytes())["rows"])
    result["release_rows"] = list(result["feedback_rows"])
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
    return result


def measure(raw: Path, pre: Json) -> Json:
    """Execute precisely the sealed24 paired streams, retaining each attempt before work."""
    began = time.monotonic()
    for index, plan in enumerate(protocol()["frozen_evaluation_manifest"]):
        require(time.monotonic() - began < 2400, "measurement_deadline")
        progress("evaluation_before", index, 24 - index)
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
        progress("evaluation_after", index + 1, 23 - index)
    return recover(raw, "unstarted")
