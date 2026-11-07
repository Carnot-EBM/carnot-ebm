"""REQ-REPORT-8895: audit frozen signed heads on their registered fresh streams.

The exact constructed verifier supplies delayed supervision. These trajectories
release causal evidence; a later audit owns every feature-benefit decision.
"""

from copy import deepcopy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import time
from typing import Any

from carnot.memory import signed_memory_8892 as signed
from carnot.reporting import signed_feature_methods_8892 as methods
from carnot.reporting import topology_memory_methods_8881 as old
from carnot.reporting import signed_feature_runner_8892 as prior
from carnot.reporting import topology_causal_streams_8884 as causal
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash
from carnot.reporting.causal_learning_audit_8810 import require, score
from carnot.reporting.v746_retained_evidence import reference, load
from carnot.reporting.v748_collection_contract import namespace

__all__ = ["methods", "causal", "require"]

Json = dict[str, Any]
ROOT = methods.ROOT
NAME, TASK = "experiment_8895_v761_signed_causal_streams", "exp8895-signed-causal-streams"
CLI = "scripts/experiments/" + NAME + ".py"
OWNED = [
    "python/carnot/reporting/signed_causal_streams_8895.py",
    "python/carnot/reporting/signed_causal_runner_8895.py",
    CLI,
]
TESTS = ["tests/python/test_signed_causal_streams_8895.py"]
SEED = methods.SEED
SOURCE = methods.NAME
PIN = "sha256:027641956a2d80c1466048a1f3eb150f968fd7cf274b835774834653e00d8461"
FIELDS = [
    "prediction_rows",
    "release_rows",
    "gradient_rows",
    "admission_rows",
    "retention_rows",
    "state_hashes",
    "restart_receipts",
    "tail_drain_rows",
    "delay_totals",
]


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Flushed counts expose real work without extending any execution deadline."""
    print(f"[exp8895] phase={phase} completed={completed} pending={pending}", flush=True)


def upstream() -> Any:
    """Reuse original custody checks with isolated producer bindings."""
    checker = namespace(prior, "signed8895_inputs")
    checker.m = __import__(__name__, fromlist=["*"])
    checker.UPSTREAM = {SOURCE: (PIN, "signed_methods_ready_score", 1)}
    return checker


def preconditions(raw: Path, sources: Path) -> Json:
    """Authenticate original bytes, exact schema and frozen parameters before opening labels."""
    checker = upstream()
    pre = dict(checker.preconditions(raw, sources))
    if pre["external_ready"]:
        original = json.loads((sources / (SOURCE + ".json")).read_bytes())
        accepted = (
            original["frozen_protocol"] == methods.protocol()
            and original["feature_schema"] == methods.protocol()["feature_schema"]
            and original["owned_statement_coverage"]["complete"]
            and original["owned_statement_coverage"]["percent"] == 100
            and all(r["passed"] for r in original["control_rows"])
        )
        if not accepted:
            pre["external_ready"] = False
            pre["diagnostics"].append(
                dict(
                    check="rejected_producer",
                    upstream=SOURCE,
                    artifact_field="frozen_protocol_schema_controls",
                    expected=True,
                    observed=False,
                    path=str(sources / (SOURCE + ".json")),
                    hash=PIN,
                )
            )
        pre["fitted_heads"] = next(
            r
            for r in pre["references"]
            if r["original_path"] == original["raw_directory"] + "/fit.json"
        )
    return pre


def census(reason: str) -> list[Json]:
    """A failed run still owns every scheduled unit in every original arm."""
    return [
        dict(r, arm=f + ":" + r["arm"], feature_arm=f)
        for f in ("old", "signed")
        for r in old.census(methods.protocol(), reason)
    ]


def run_stream(raw: Path, plan: Json, heads: Json, protocol: Json, *, restart: bool = True) -> Json:
    """Reuse atomic transactions and due-time releases with signed observations and SGD."""
    arms = []
    for feature in ("old", "signed"):
        runtime = namespace(causal, "signed8895_" + feature)
        runtime.base = signed.engine()
        runtime.base.SEED, runtime.base.progress = SEED, progress
        runtime.c, runtime.progress = signed, progress
        original_store = runtime.Store
        cache: dict[Path, Any] = {}

        def commit(self: Any, state: Json) -> None:
            """Persist full parameter and template identities before any label can open."""
            for sealed in state["pending"].values():
                sealed.setdefault("parameter_hash", canonical_hash(state["parameters"]))
                sealed.setdefault("template_hash", canonical_hash(state["templates"]))
            original_store.commit(self, state)

        sealed_store = type("SealedStore", (original_store,), {"commit": commit})

        def store(path: Path, params: Json, transactions: list[Json]) -> Any:
            if restart or params or path not in cache:
                cache[path] = sealed_store(path, params, transactions)
            return cache[path]

        runtime.Store = store
        progress("stream_before_" + plan["stream_id"] + ":" + feature, 0, 48)
        result = dict(runtime.run_stream(raw / feature, plan, heads[feature], protocol))
        result["feature_arm"] = feature
        arms.append(result)
        atomic_json(raw / "partial.json", dict(arms=arms))
        progress("stream_after_" + plan["stream_id"] + ":" + feature, 48, 0)
    result = assemble(arms)
    verify(result)
    return result


def assemble(arms: list[Json]) -> Json:
    """Use the same reduction for complete streams and retained unfinished feature arms."""
    result: Json = dict(
        arms=arms,
        rows=[
            dict(r, arm=a["feature_arm"] + ":" + r["arm"], feature_arm=a["feature_arm"])
            for a in arms
            for r in a["rows"]
        ],
    )
    for field in FIELDS:
        source = "restarts" if field == "restart_receipts" else field
        result[field] = [
            dict(
                r,
                arm=a["feature_arm"] + ":" + r["arm"],
                cluster_id=a["stream"]["stream_id"],
                seed=a["stream"]["seed"],
            )
            for a in arms
            for r in a[source]
        ]
    for row in result["prediction_rows"]:
        row.update(
            parameter_hash=row["sealed"]["parameter_hash"],
            template_hash=row["sealed"]["template_hash"],
            accept_reject_escalate_probabilities=dict(
                zip(("accept", "reject", "escalate"), row["sealed"]["prediction"], strict=True)
            ),
        )
    result["numerical_training_calls"] = len(result["gradient_rows"])
    return result


def verify(result: Json) -> None:
    """Validate durable causal rows and recompute losses rather than trusting summaries."""
    for arm in result["arms"]:
        # JSON sorts map keys; the inherited row reducer uses the registered arm order.
        ordered = dict(arm, final_states={a: arm["final_states"][a] for a in old.ARMS})
        causal.verify_stream(ordered)
        for row in arm["rows"]:
            expected_score = score(row["sealed"]["prediction"], row["truth"], 0.5)
            require(all(row[k] == v for k, v in expected_score.items()), "scored_row_drift")
        for row in arm["prediction_rows"]:
            predecessor = next(
                t["state"]
                for t in arm["transaction_rows"]
                if t["arm"] == row["arm"] and t["after"] == row["sealed"]["state_hash"]
            )
            seal_sequence = next(
                t["sequence"]
                for t in arm["transaction_rows"]
                if t["arm"] == row["arm"]
                and t["state"]["pending"].get(f"e{row['ordinal']}") == row["sealed"]
            )
            first_label = min(
                r["after_transaction_count"]
                for r in arm["label_call_rows"]
                if r["split"] == "online" and r["evidence_hash"] == row["sealed"]["evidence_hash"]
            )
            require(seal_sequence < first_label, "label_before_prediction_seal")
            require(
                row["sealed"]["parameter_hash"] == canonical_hash(predecessor["parameters"])
                and row["sealed"]["template_hash"] == canonical_hash(predecessor["templates"]),
                "prelabel_parameter_template_seal",
            )
    expected = [
        dict(r, arm=a["feature_arm"] + ":" + r["arm"], feature_arm=a["feature_arm"])
        for a in result["arms"]
        for r in a["rows"]
    ]
    require(result["rows"] == expected, "row_inventory_drift")
    require(
        all(
            r["release_ordinal"] >= r["event"] + r["configured_delay"]
            for r in result["release_rows"]
        ),
        "early_label",
    )
    for feature in ("old", "signed"):
        arm = next(a for a in result["arms"] if a["feature_arm"] == feature)
        require(
            all(
                arm["final_states"][a]["templates"]
                == arm["final_states"]["matched_true"]["templates"]
                for a in ("matched_permuted", "frozen_verified")
            ),
            "matched_admissions",
        )
    require(
        all(
            result["arms"][0]["rows"][i]["sealed"]["counts"]
            == result["arms"][1]["rows"][i]["sealed"]["counts"]
            for i in range(768)
        ),
        "matched_observation_work",
    )


def restart_equal(a: Json, b: Json) -> bool:
    """An uninterrupted private pair must match predictions, transitions and acknowledgments."""
    return all(
        canonical_hash(causal.semantic(x[k])) == canonical_hash(causal.semantic(y[k]))
        for x, y in zip(a["arms"], b["arms"], strict=True)
        for k in ("rows", "transitions", "batches", "restarts", "final_states")
    )


def reduce(results: list[Json]) -> Json:
    """Head seeds and repeated topologies never increase the paired-stream denominator."""
    rows = [r for result in results for r in result["rows"]]
    return dict(
        rows=rows,
        **{k: [r for s in results for r in s[k]] for k in FIELDS},
        numerical_training_calls=sum(s["numerical_training_calls"] for s in results),
    )


def recover(raw: Path, reason: str) -> Json:
    """Preserve completed streams, failed attempts and every unstarted scheduled slot."""
    results = [load(reference(p)) for p in sorted(raw.glob("stream_*.json"))]
    finished = {a["stream"]["stream_id"] for r in results for a in r["arms"]}
    partial = [
        reference(p)
        for p in sorted(raw.glob("states/**/partial.json"))
        if p.parent.name not in finished
    ]
    results += [assemble(load(r)["arms"]) for r in partial]
    result = reduce(results)
    completed = {(r["unit_id"], r["arm"]): r for r in result["rows"]}
    attempts = [load(reference(p)) for p in sorted(raw.glob("attempt_*.json"))]
    failed = {a["cluster_id"] for a in attempts if a["status"] == "failed"}
    result["rows"] = [
        completed.get(
            (r["unit_id"], r["arm"]), dict(r, status="failed") if r["cluster_id"] in failed else r
        )
        for r in census(reason)
    ]
    result["partial_refs"] = partial
    result["attempt_rows"] = attempts
    result["stream_refs"] = [reference(p) for p in sorted(raw.glob("stream_*.json"))]
    controls = load(reference(raw / "controls.json")) if (raw / "controls.json").is_file() else {}
    result["control_rows"] = controls.get("rows", [])
    result["projected_runtime_s"] = controls.get("projected_runtime_s", 0)
    result["numerical_training_calls"] = sum(
        len(load(reference(p))["acks"])
        for parent in ("states", "pilots")
        for p in (raw / parent).rglob("matched_*.json")
    ) + (
        load(reference(raw / "sensitivity.json"))["numerical_training_calls"]
        if (raw / "sensitivity.json").is_file()
        else 0
    )
    return result


def retained_stream(raw: Path, plan: Json, heads: Json, p: Json, *, restart: bool = True) -> Json:
    """Only immutable copies leave private scratch, including failed atomic states."""
    import shutil

    with TemporaryDirectory(prefix="carnot8895-state-", dir="/tmp") as directory:
        try:
            return run_stream(Path(directory), plan, heads, p, restart=restart)
        finally:
            shutil.copytree(directory, raw)


def measure(raw: Path, pre: Json) -> Json:
    """Qualify fixed private controls, then run every registered stream exactly once."""
    began = time.monotonic()
    p = methods.protocol()
    atomic_json(raw / "protocol.json", p)
    fitted = load(pre["fitted_heads"])
    atomic_json(raw / "frozen_fit.json", fitted)
    progress("negative_controls_before", 0, 1)
    controls, calls = old.sensitivity()
    checks = [dict(check=r["control"], passed=r["passed"]) for r in controls]
    atomic_json(raw / "sensitivity.json", dict(rows=controls, numerical_training_calls=calls))
    progress("negative_controls_after", 1, 0)
    pilot_start = time.monotonic()
    for plan in p["pilot_manifest"]:
        progress("paired_pilot_before", 0, 2)
        restarted = retained_stream(
            raw / "pilots" / plan["stream_id"] / "restarted", plan, fitted["heads"], p
        )
        uninterrupted = retained_stream(
            raw / "pilots" / plan["stream_id"] / "uninterrupted",
            plan,
            fitted["heads"],
            p,
            restart=False,
        )
        equal = restart_equal(restarted, uninterrupted)
        atomic_json(
            raw / ("pilot_" + plan["stream_id"] + ".json"),
            dict(restarted=restarted, uninterrupted=uninterrupted, passed=equal),
        )
        checks.append(dict(check="paired_restart_" + plan["stream_id"], passed=equal))
        calls += restarted["numerical_training_calls"] + uninterrupted["numerical_training_calls"]
        progress("paired_pilot_after", 2, 0)
    projected = (time.monotonic() - pilot_start) * 6 + time.monotonic() - began + 1200
    atomic_json(
        raw / "controls.json",
        dict(rows=checks, projected_runtime_s=projected, numerical_training_calls=calls),
    )
    require(all(c["passed"] for c in checks), "owned_pilot_control_failure")
    require(projected < 4200, "owned_runtime_projection")
    for index, plan in enumerate(p["cohort_manifest"]):
        require(time.monotonic() - began < 2400, "measurement_deadline")
        progress("cohort_before", index, 24 - index)
        attempt = dict(cluster_id=plan["stream_id"], status="attempted", seed=plan["seed"])
        atomic_json(raw / ("attempt_" + plan["stream_id"] + ".json"), attempt)
        try:
            result = retained_stream(raw / "states" / plan["stream_id"], plan, fitted["heads"], p)
            atomic_json(raw / ("stream_" + plan["stream_id"] + ".json"), result)
            attempt["status"] = "complete"
        except (OSError, ValueError, RuntimeError, KeyError):
            attempt["status"] = "failed"
            raise
        finally:
            atomic_json(raw / ("attempt_" + plan["stream_id"] + ".json"), attempt)
        progress("cohort_after", index + 1, 23 - index)
    return recover(raw, "unstarted")
