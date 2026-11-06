"""REQ-REPORT-8843: collect causal evidence; independent benefit belongs to the audit.

Only small CPU heads train. Constructed signed records expose exact truth and
cannot establish natural-language or hidden-game generalization.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import time
from typing import Any
from unittest.mock import patch

import jax
import jax.numpy as jnp
import numpy as np

from carnot.memory import categorical_memory_8830 as c
from carnot.reporting import categorical_memory_runner_8830 as old
from carnot.reporting import learning_signal_runner_8815 as validation
from carnot.reporting import signal_replay_qualification_8828 as upstream
from carnot.reporting.causal_learning_audit_8810 import score
from carnot.reporting.current_work_receipt import (
    atomic_json,
    build_current_work_receipt,
    canonical_hash,
    sha256_file,
)
from carnot.reporting.primary_publication import publish_primary
from scripts.experiment_template import normalize_artifact_for_template_write

Json = dict[str, Any]
ROOT = old.ROOT
NAME = "experiment_8843_v757_causal_streams"
CLI = "scripts/experiments/" + NAME + ".py"
OWNED = ["python/carnot/reporting/causal_streams_8843.py", CLI]
TESTS = ["tests/python/test_causal_streams_8843.py"]
AUTHORITY = ROOT / "results/experiment_8841_v757_memory_execution.json"
AUTHORITY_PIN = "sha256:551f25fdf6bf7fe44570589fb140760342f463f637c941b6dd60f218136cabf1"
ARMS = (
    "categorical_frozen",
    "categorical_calibration_only",
    "categorical_combined",
    "categorical_shuffled_combined",
    "linear_frozen",
    "linear_combined",
)
ERRORS = ("support", "refute", "missing", "binding_mismatch", "conflict", "alternative")
SEED = 8843000


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Flush real boundaries so imports, fits and durable loops stay observable."""
    print(f"[exp8843] phase={phase} completed={completed} pending={pending}", flush=True)


def manifest(start: int = 0, count: int = 24) -> list[Json]:
    """Assign model seeds before labels; separate RNGs define independent streams."""
    streams = []
    for i in range(start, start + count):
        rng = np.random.default_rng(SEED + i)
        recipes = []
        for j in range(60):
            marker = i * 60 + j
            recipes.append(
                dict(
                    unit_id=f"stream-{i:02d}/world-{j:02d}",
                    marker=marker,
                    structure_id=canonical_hash(["six-term-query-eleven-branch-marker", marker]),
                    error_type=ERRORS[int(rng.integers(len(ERRORS)))],
                )
            )
        streams.append(
            dict(
                stream_id=f"stream-{i:02d}",
                seed=c.d.SEEDS[i % 5],
                rng_seed=SEED + i,
                online=recipes[:48],
                retention=recipes[48:],
            )
        )
    return streams


def payload(recipe: Json) -> Json:
    """Six query terms exclude old five-term corpora; branch bits distinguish worlds.

    Eleven distinct branch lengths anchor each bit under entity renaming. The
    disconnected marker never matches a query relation or carries a label.
    """
    prefix = recipe["unit_id"] + ":"
    query = [["?x", f"r{i}", prefix + f"c{i}"] for i in range(4)]
    query.append([prefix + "c3", "r4", prefix + "c4"])
    facts = [[prefix + "a" if s == "?x" else s, r, o, True] for s, r, o in query]
    error = recipe["error_type"]
    if error == "refute":
        facts[-1][3] = False
    elif error == "missing":
        facts.pop(0)
    elif error == "binding_mismatch":
        facts[0][0] = prefix + "different"
    elif error == "conflict":
        facts.append([*facts[0][:3], False])
    elif error == "alternative":
        facts += [[prefix + "other", r, o, True] for _, r, o in query[:4]]
    for branch in range(11):
        nodes = [prefix + "root"] + [prefix + f"b{branch}-{k}" for k in range(branch + 1)]
        facts += [[a, "marker", b, True] for a, b in zip(nodes, nodes[1:])]
        if recipe["marker"] & (1 << branch):
            facts.append([nodes[-1], "marker", nodes[-1], True])
    return dict(facts=facts, query=query)


def census(streams: list[Json], reason: str) -> list[Json]:
    """Retain every planned slot when a stream is blocked or unfinished."""
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
        for s in streams
        for arm in ARMS
        for split, steps in (("online", [None]), ("retention", [0, 24, 48]))
        for step in steps
        for i, w in enumerate(s[split])
    ]


def release(store: c.Store, event: str, value: Json, ordinal: int, arm: str) -> tuple[Json, Json]:
    """Verify released evidence before one atomic arm-specific update and acknowledgement."""
    pending = store.state["pending"].get(event)
    if (
        store.active is not None
        or pending is None
        or ordinal != pending["ordinal"] + 8
        or canonical_hash(value) != pending["evidence_hash"]
    ):
        raise ValueError("feedback_boundary")
    verified = str(c.m.decide(value)["decision"])
    prefix = store.state["released"] + [verified]
    label = prefix[(ordinal * 7) % len(prefix)] if "shuffled" in arm else verified
    state = deepcopy(store.state)
    gradient: Json = {}
    before = deepcopy(state["parameters"])
    if not arm.endswith("frozen"):
        weights = [jnp.asarray(x, dtype=jnp.float32) for x in before["weights"]]
        x = jnp.asarray([c.m.features(pending["counts"], i) for i in range(3)])
        new, loss, grads = c.step(
            weights, x, jnp.asarray(c.m.LABELS.index(label)), 0.1, before["objective"]
        )
        state["parameters"] = dict(before, weights=[np.asarray(x).tolist() for x in new])
        gradient = dict(
            gradients=[np.asarray(x).tolist() for x in grads],
            loss=float(loss),
            gradient_norm=float(sum(np.sum(np.asarray(g) ** 2) for g in grads) ** 0.5),
            parameter_delta=float(
                sum(np.sum((np.asarray(a) - np.asarray(b)) ** 2) for a, b in zip(weights, new))
                ** 0.5
            ),
            parameters_before=before,
            parameters_after=deepcopy(state["parameters"]),
        )
    template = c.m.normalize(value)
    add = (
        "combined" in arm
        and label == verified
        and template not in state["templates"]
        and len(state["templates"]) < 32
    )
    if add:
        state["templates"].append(template)
        state.setdefault("template_provenance", {})[template] = dict(
            event_id=event,
            evidence_hash=canonical_hash(value),
            verified_label=verified,
            binding_witnesses=c.binding_provenance(value),
        )
    state["pending"].pop(event)
    state["acks"].append(event)
    state["released"].append(verified)
    store.commit(state)
    return dict(
        event_id=event,
        prediction_ordinal=pending["ordinal"],
        feedback_ordinal=ordinal,
        feedback_label=label,
        verified_label=verified,
        released_prefix=prefix,
        prediction_hash=canonical_hash(pending),
        template_added=add,
        template=template,
        acknowledgement=event,
        checkpoint_hash=canonical_hash(state),
    ), gradient


COLLECTIONS = (
    "per_event_results",
    "retention_rows",
    "state_transition_rows",
    "label_release_rows",
    "template_use_rows",
    "gradient_rows",
    "restart_rows",
    "final_state_refs",
)


def run_stream(raw: Path, stream: Json, heads: dict[tuple[int, str], Json]) -> Json:
    """Reset every arm, restart at24, and drain all48 labels before final retention."""
    work: Json = {k: [] for k in COLLECTIONS}
    worlds = [payload(w) for w in stream["online"]]
    retained = [payload(w) for w in stream["retention"]]
    for arm in ARMS:
        objective = arm.split("_")[0]
        params = deepcopy(heads[(stream["seed"], objective)])
        store = c.Store(raw / (arm + ".json"), params)
        identity = dict(cluster_id=stream["stream_id"], seed=stream["seed"], arm=arm)
        online = []

        def retention(step: int) -> None:
            before = canonical_hash(store.state)
            for i, value in enumerate(retained):
                counts = c.m.observe(value, store.state["templates"])["counts"]
                p = c.probabilities(store.state["parameters"], counts)
                truth = str(c.m.decide(value)["decision"])
                measured = score(p, truth, 0.5)
                work["retention_rows"].append(
                    dict(
                        identity,
                        unit_id=stream["retention"][i]["unit_id"] + f":step{step}",
                        ordinal=i + 1,
                        checkpoint=step,
                        split="retention",
                        denominator=1,
                        status="complete",
                        missing_reason=None,
                        prediction=p,
                        truth=truth,
                        state_hash=before,
                        counts=counts,
                        **measured,
                        learning_state_unchanged=before == canonical_hash(store.state),
                    )
                )

        retention(0)
        for ordinal in range(1, 57):
            if ordinal <= 48:
                event = f"q{ordinal}"
                before = canonical_hash(store.state)
                sealed = store.predict(event, worlds[ordinal - 1], 0.5)
                store.close()
                row = dict(
                    identity,
                    unit_id=stream["online"][ordinal - 1]["unit_id"],
                    ordinal=ordinal,
                    checkpoint=None,
                    split="online",
                    denominator=1,
                    status="sealed",
                    missing_reason="feedback_pending",
                    metric=None,
                    sealed=sealed,
                    prediction=sealed["prediction"],
                    prediction_seal=canonical_hash(sealed),
                    error_type=stream["online"][ordinal - 1]["error_type"],
                    future_window=ordinal >= 25,
                )
                online.append(row)
                work["template_use_rows"].append(
                    dict(
                        identity,
                        event_id=event,
                        template_uses=sealed["template_uses"],
                        prediction_seal=canonical_hash(sealed),
                    )
                )
                work["state_transition_rows"].append(
                    dict(
                        identity,
                        operation="predict",
                        ordinal=ordinal,
                        before=before,
                        after=canonical_hash(store.state),
                        event_id=event,
                    )
                )
            if ordinal >= 9:
                event = f"q{ordinal - 8}"
                before = canonical_hash(store.state)
                released, gradient = release(store, event, worlds[ordinal - 9], ordinal, arm)
                released.update(identity)
                work["label_release_rows"].append(released)
                if gradient:
                    work["gradient_rows"].append(dict(identity, event_id=event, **gradient))
                row = online[ordinal - 9]
                row.update(
                    score(row["prediction"], released["verified_label"], 0.5),
                    truth=released["verified_label"],
                    status="complete",
                    missing_reason=None,
                )
                work["state_transition_rows"].append(
                    dict(
                        identity,
                        operation="release",
                        ordinal=ordinal,
                        before=before,
                        after=canonical_hash(store.state),
                        event_id=event,
                    )
                )
            if ordinal == 24:
                before = canonical_hash(store.state)
                store = c.Store(store.path, {})
                work["restart_rows"].append(
                    dict(
                        identity,
                        checkpoint=24,
                        before=before,
                        after=canonical_hash(store.state),
                        unchanged=before == canonical_hash(store.state),
                    )
                )
                retention(24)
        retention(48)
        work["per_event_results"].extend(online)
        work["final_state_refs"].append(dict(identity, **old.reference(store.path)))
        progress(
            stream["stream_id"] + "/" + arm,
            len(work["per_event_results"]),
            288 - len(work["per_event_results"]),
        )
    return work


def integrity(work: Json, streams: list[Json]) -> Json:
    """Recompute losses, seals, paired census and causal release identities without gain gates."""
    rows = work["per_event_results"] + work["retention_rows"]
    key = lambda r: (r["cluster_id"], r["arm"], r["unit_id"])
    expected = census(streams, "")
    c.d.require(
        len(rows) == len(expected) and {key(r) for r in rows} == {key(r) for r in expected},
        "row_census",
    )
    for row in rows:
        measured = score(row["prediction"], row["truth"], 0.5)
        c.d.require(
            all(row[k] == v for k, v in measured.items())
            and row["denominator"] == 1
            and row["status"] == "complete"
            and row["missing_reason"] is None,
            "row_integrity",
        )
        if row["split"] == "online":
            c.d.require(row["prediction_seal"] == canonical_hash(row["sealed"]), "prediction_seal")
        else:
            c.d.require(row["learning_state_unchanged"], "retention_mutation")
    for row in work["label_release_rows"]:
        c.d.require(
            row["feedback_ordinal"] == row["prediction_ordinal"] + 8
            and row["feedback_label"] in row["released_prefix"],
            "release_integrity",
        )
    c.d.require(
        len(work["label_release_rows"]) == len(streams) * 288
        and len(work["gradient_rows"]) == len(streams) * 192
        and all(r["unchanged"] for r in work["restart_rows"]),
        "transition_census",
    )
    return dict(
        passed=True, online=len(work["per_event_results"]), retention=len(work["retention_rows"])
    )


def projection(pilots: list[float], initialization_s: float) -> Json:
    """Budget both the registered execution and a full cold replay, reserving validation time."""
    projected = 2 * initialization_s + max(pilots) * 24 * 2 * 1.5 + 600
    return dict(
        pilot_seconds=pilots,
        initialization_s=initialization_s,
        registered_streams=24,
        cold_replay_streams=24,
        safety_factor=1.5,
        validation_reserve_s=600,
        projected_s=projected,
        budget_s=4200,
        passed=projected <= 4200,
    )


def preflight(raw: Path, authority_path: Path = AUTHORITY) -> tuple[Json, Json]:
    """Authenticate original inputs, method code and execution authority before fitting."""
    raw.mkdir(parents=True, exist_ok=True)
    raw.chmod(0o700)
    probe = raw / ".write_probe"
    probe.write_text("private")
    probe.unlink()
    if (
        shutil.disk_usage(raw).free < 2**30
        or Path(sys.prefix).resolve() != (ROOT / ".venv").resolve()
    ):
        raise RuntimeError("internal_resources")
    authority = old.bound(dict(path=str(authority_path), sha256=AUTHORITY_PIN))
    if (
        authority.get("memory_execution_ready_score") != 1
        or authority.get("required_checks_passed") is not True
    ):
        raise PermissionError("memory_execution_ready_score:observed_zero")
    for ref in (
        authority["source_artifact_hashes"]
        + authority["code_config_hashes"]
        + authority["raw_shard_hashes"]
    ):
        upstream.bindings([ref], "authority_hash")
    for ref in upstream.references(old.bound(authority["raw_shard_hashes"][0])):
        upstream.bindings([ref], "raw_authority_hash")
    _, evidence, pre = old.preflight(raw)
    snapshot = old.bound(evidence["snapshot"])
    upstream.bindings(snapshot["inputs"] + snapshot["code"], "original_method_authority")
    excluded = set(
        evidence["split_manifest"]["fit"]
        + evidence["split_manifest"]["validation"]
        + evidence["split_manifest"]["excluded"]
    )
    fresh_structure = c.d.structure(payload(manifest()[0]["online"][0]))
    c.d.require(fresh_structure not in excluded, "exposed_query_structure")
    pre.update(
        authority=old.reference(authority_path),
        original_method_inputs=snapshot["inputs"],
        exposed_structure_hashes=sorted(excluded),
        fresh_query_structure=fresh_structure,
        disjointness="six distinct query terms; exposed fit/evaluation queries have at most five",
        time_budget_s=4200,
        model_loads=0,
        generator_weights_updated=False,
    )
    atomic_json(raw / "preconditions.json", pre)
    return evidence, pre


def initialize(evidence: Json) -> tuple[dict[tuple[int, str], Json], list[Json]]:
    """Recompute the original fits, including every400-step curve, in this process."""
    fit = [r for r in evidence["development_rows"] if r["split"] == "fit"]
    valid = [r for r in evidence["development_rows"] if r["split"] == "validation"]
    heads, rows = {}, []
    for seed in c.d.SEEDS:
        for objective in ("categorical", "linear"):
            progress(f"fit_before_{seed}_{objective}", len(rows), 10 - len(rows))
            actual = c.d.fit(fit, valid, seed, objective, 0.1)
            original = next(
                r
                for r in evidence["fits"]
                if r["seed"] == seed and r["objective"] == objective and r["lr"] == 0.1
            )
            c.d.require(actual == original, "original_fit_drift")
            params = dict(objective=objective, lr=0.1, weights=actual["parameters"])
            heads[(seed, objective)] = params
            rows.append(
                dict(
                    seed=seed,
                    objective=objective,
                    updates=400,
                    parameters=params,
                    parameter_hash=canonical_hash(params),
                    fitting_input_hash=canonical_hash(fit),
                    recipe="original learning_signal_diagnosis_8815.fit(fit,valid,seed,objective,0.1,400)",
                )
            )
            progress("fit_after", len(rows), 10 - len(rows))
    return heads, rows


def execute(raw: Path, evidence: Json, work: Json, started: float) -> None:
    """Checkpoint each complete stream; never drop registered rows to meet the budget."""
    began = time.monotonic()
    heads, work["initialization_rows"] = initialize(evidence)
    initialization_s = time.monotonic() - began
    pilots = []
    for stream in manifest(24, 2):
        progress("timing_pilot_before", len(pilots), 2 - len(pilots))
        began = time.monotonic()
        pilot = run_stream(raw / "pilots" / stream["stream_id"], stream, heads)
        integrity(pilot, [stream])
        pilots.append(time.monotonic() - began)
        atomic_json(raw / "pilots" / (stream["stream_id"] + ".json"), pilot)
        progress("timing_pilot_after", len(pilots), 2 - len(pilots))
    work["resource_projection"] = projection(pilots, initialization_s)
    c.d.require(work["resource_projection"]["passed"], "projected_budget")
    for i, stream in enumerate(work["stream_manifest"]):
        if time.monotonic() - started > 3300:
            raise TimeoutError("registered_execution_deadline")
        progress("stream_before", i, 24 - i)
        result = run_stream(raw / "stores" / stream["stream_id"], stream, heads)
        integrity(result, [stream])
        checkpoint = raw / "streams" / (stream["stream_id"] + ".json")
        atomic_json(checkpoint, result)
        work["stream_checkpoint_refs"].append(old.reference(checkpoint))
        for key in COLLECTIONS:
            work[key].extend(result[key])
        atomic_json(
            raw / "execution_checkpoint.json",
            dict(
                completed_streams=i + 1,
                pending_streams=23 - i,
                stream_checkpoint_refs=work["stream_checkpoint_refs"],
            ),
        )
        progress("stream_after", i + 1, 23 - i)
    work["integrity"] = integrity(work, work["stream_manifest"])


def validate(raw: Path) -> Json:
    """Reuse bounded owned coverage, private E2E and a separate full-suite health receipt."""
    with patch.multiple(validation, OWNED=OWNED, TESTS=TESTS, CLI=CLI):
        return validation.validate(raw)


def replay(path: Path) -> Json:
    """Authenticate bytes and rerun causal transitions, gradients and losses in a fresh store."""
    value = json.loads(path.read_bytes())
    for ref in (
        value["code_config_hashes"] + value["source_artifact_hashes"] + value["raw_shard_hashes"]
    ):
        upstream.bindings([ref], "replay_operand_hash")
    work = old.bound(value["raw_shard_hashes"][0])
    c.d.require(work["stream_manifest"] == manifest(), "manifest_drift")
    for ref in work["stream_checkpoint_refs"] + work["final_state_refs"]:
        upstream.bindings([ref], "stream_operand_hash")
    c.d.require(value["rows"] == result_rows(work, value["verdict_class"]), "reduction_drift")
    for key in (
        "stream_manifest",
        "per_event_results",
        "retention_rows",
        "initialization_rows",
        "state_transition_rows",
        "label_release_rows",
        "template_use_rows",
        "gradient_rows",
        "resource_projection",
    ):
        c.d.require(value[key] == work[key], "evidence_drift:" + key)
    completed = sum(r["status"] == "complete" for r in value["rows"])
    c.d.require(
        value["completed_count"] == completed
        and value["intended_count"] == 12096
        and value["censored_count"] == 12096 - completed,
        "count_drift",
    )
    for receipt in value["validation_receipts"]:
        if "log_path" in receipt:
            log = Path(receipt["log_path"])
            log = log if log.is_absolute() else ROOT / log
            upstream.bindings([dict(path=str(log), sha256=receipt["log_sha256"])], "validation_log")
    if work["per_event_results"]:
        with tempfile.TemporaryDirectory(prefix="carnot8843-replay-") as temporary:
            evidence, _ = preflight(Path(temporary))
            heads, initial = initialize(evidence)
            c.d.require(initial == work["initialization_rows"], "initialization_drift")
            rebuilt: Json = {k: [] for k in COLLECTIONS}
            completed_streams = manifest()[: len(work["stream_checkpoint_refs"])]
            for i, stream in enumerate(completed_streams):
                actual = run_stream(Path(temporary) / stream["stream_id"], stream, heads)
                expected = old.bound(work["stream_checkpoint_refs"][i])
                for key in COLLECTIONS:
                    if key == "final_state_refs":
                        c.d.require(
                            [r["sha256"] for r in actual[key]]
                            == [r["sha256"] for r in expected[key]],
                            "final_state_drift",
                        )
                    else:
                        c.d.require(actual[key] == expected[key], "stream_drift:" + key)
                    rebuilt[key].extend(actual[key])
                progress("replay_stream", i + 1, 23 - i)
            for key in COLLECTIONS[:-1]:
                c.d.require(rebuilt[key] == work[key], "aggregate_drift:" + key)
            integrity(rebuilt, completed_streams)
    c.d.require(
        value["stream_rows_ready_score"]
        == int(
            value["verdict_class"] == "circular_positive"
            and completed == 12096
            and value["required_checks_passed"]
        ),
        "readiness_drift",
    )
    return dict(passed=True, replay_passed=True, completed_count=completed)


def terminal(candidate: Path, raw: Path) -> Json:
    """Use unchanged scanners and a bounded outside-checkout replay with streamed output."""
    with patch.object(validation, "CLI", CLI):
        return validation.terminal(candidate, raw)


def result_rows(work: Json, reason: str) -> list[Json]:
    """Censored slots remain visible and retain their original denominator."""
    key = lambda r: (r["cluster_id"], r["arm"], r["unit_id"])
    completed = {key(r): r for r in work["per_event_results"] + work["retention_rows"]}
    return [completed.get(key(r), r) for r in census(work["stream_manifest"], reason)]


def main(argv: list[str] | None = None) -> int:
    """Publish execution evidence after owned validation; never activate the milestone."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default="20261105")
    parser.add_argument(
        "--output", "--private-output", type=Path, default=ROOT / "results" / (NAME + ".json")
    )
    parser.add_argument("--raw-dir", type=Path)
    parser.add_argument("--authority", type=Path, default=AUTHORITY)
    parser.add_argument("--replay", type=Path)
    args = parser.parse_args(argv)
    if args.replay:
        try:
            print(json.dumps(replay(args.replay)), flush=True)
            return 0
        except (OSError, ValueError, KeyError, RuntimeError) as error:
            print(f"replay_rejected:{error}", flush=True)
            return 1
    if args.date != "20261105":
        print("date_rejected: expected 20261105", flush=True)
        return 1
    started = time.monotonic()
    started_ns = time.monotonic_ns()
    raw = (
        args.raw_dir or args.output.parent / "raw" / NAME / "invocations" / str(time.time_ns())
    ).absolute()
    work: Json = {k: [] for k in COLLECTIONS}
    work.update(
        stream_manifest=manifest(),
        initialization_rows=[],
        resource_projection={},
        stream_checkpoint_refs=[],
    )
    pre: Json = {}
    gates = []
    verdict = "circular_positive"
    admitted = False
    progress("preconditions_before")
    try:
        evidence, pre = preflight(raw, args.authority)
        admitted = True
        atomic_json(raw / "frozen_manifest.json", dict(streams=work["stream_manifest"], arms=ARMS))
        progress("preconditions_after", 1)
        execute(raw, evidence, work, started)
    except (OSError, ValueError, KeyError, RuntimeError) as error:
        external = not admitted and isinstance(
            error, (FileNotFoundError, PermissionError, KeyError)
        )
        verdict = (
            "partial"
            if isinstance(error, TimeoutError)
            else "blocked"
            if external
            else "disqualified"
        )
        path = Path(getattr(error, "filename", None) or args.authority)
        gates.append(
            dict(
                check="upstream_authority" if external else "owned_execution",
                upstream="exp8841-memory-execution",
                artifact_field="memory_execution_ready_score",
                expected=1,
                observed=str(error),
                path=str(path),
                hash=sha256_file(path) if path.is_file() else None,
                status="missing_file"
                if isinstance(error, FileNotFoundError)
                else "missing_field"
                if isinstance(error, KeyError)
                else "observed_zero"
                if "observed_zero" in str(error)
                else "rejected_producer"
                if external
                else "internal_failure",
            )
        )
    atomic_json(raw / "evidence.json", work)
    progress("validation_before")
    checked = validate(raw)
    progress("validation_after", 1)
    if not checked["passed"]:
        verdict = "disqualified"
    ended_ns = time.monotonic_ns()
    spans = [
        dict(
            phase="preconditions_execution_validation",
            started_monotonic_ns=started_ns,
            ended_monotonic_ns=ended_ns,
            duration_s=(ended_ns - started_ns) / 1e9,
        )
    ]
    performed = bool(work["initialization_rows"])
    receipt = build_current_work_receipt(
        run_id=raw.name,
        owner_pid=os.getpid(),
        events=[],
        inference_substrate="deterministic_verifier" if performed else "blocked_no_run",
        inference_substrate_class="no_model_load",
        execution_venue="host",
        inference_substrate_details=dict(backend="jax_cpu", generator_weights_updated=False),
        started_monotonic_ns=started_ns,
        ended_monotonic_ns=ended_ns,
        phase_spans=spans,
        small_ebm_training=dict(performed=performed),
    )
    completed = len(work["per_event_results"]) + len(work["retention_rows"])
    passed = verdict == "circular_positive" and completed == 12096
    value = dict(
        work,
        experiment_id=8843,
        task_id="exp8843-causal-streams",
        milestone="2026.11.757",
        run_date=args.date,
        honest_verdict="complete_" + verdict + "_causal_streams",
        verdict_class=verdict,
        inference_substrate="deterministic_verifier" if performed else "blocked_no_run",
        inference_substrate_class="no_model_load",
        execution_venue="host",
        MODEL_SPECS=[],
        model_invocation_counts=receipt["invocation_counts"],
        numerical_training_calls=sum(r["updates"] for r in work["initialization_rows"])
        + len(work["gradient_rows"])
        + 384 * int(bool(work["resource_projection"])),
        numerical_training_scope="Current original fits, two timing streams, registered updates; validation/replay work is separate.",
        verifier_is_oracle=True,
        exposure_scope="constructed signed records; original exposed development fits; fresh synthetic streams",
        independent_generalization_score=0,
        duration_s=(ended_ns - started_ns) / 1e9,
        random_seed=SEED,
        phase_spans=spans,
        preconditions_checked=pre,
        source_artifact_hashes=[
            old.reference(p) for p in (args.authority, old.UPSTREAM) if p.is_file()
        ],
        code_config_hashes=[
            old.reference(ROOT / p)
            for p in dict.fromkeys(
                OWNED
                + TESTS
                + old.OWNED
                + validation.CODE
                + upstream.OWNED
                + [
                    "python/carnot/reporting/primary_publication.py",
                    "scripts/experiment_template.py",
                ]
            )
        ],
        raw_shard_hashes=[old.reference(raw / "evidence.json")],
        rows=result_rows(work, verdict),
        intended_count=12096,
        completed_count=completed,
        failed_count=0,
        censored_count=12096 - completed,
        excluded_count=0,
        independent_count=len(work["stream_checkpoint_refs"]),
        gate_check_summary=gates,
        acceptance_gates=dict(
            integrity=dict(
                threshold=12096,
                principle="Every frozen row and causal transition; gain is not a gate.",
            ),
            time=dict(
                threshold=4200,
                principle="Measured execution, validation and replay must fit the deadline.",
            ),
        ),
        required_checks_passed=passed,
        validation_receipts=checked["receipts"],
        owned_statement_coverage=checked["coverage"],
        stream_rows_ready_score=int(passed),
        current_work_receipt=receipt,
        repository_health_receipts=[
            r for r in checked["receipts"] if r["scope"] == "repository_health"
        ],
        reproducibility_checksum=canonical_hash([manifest(), ARMS, pre.get("configuration")]),
        methodology_note="Six matched arms; delay8; SGD0.1; restart24; untouched retention0/24/48; primary future window25-48. Exp8846 alone audits benefit.",
    )
    value["field_principles"] = {
        k: "Bind this producer's actual execution to authenticated operands; completion is not benefit."
        for k in value
    }
    value["field_principles"].update(
        stream_manifest="Independent stream RNGs and complete world recipes; cyclic model seeds are paired measurements.",
        gradient_rows="Actual gradients and before/after coefficients establish numerical updates, not future gains.",
        label_release_rows="Only verified delay-eligible labels enter the released prefix and acknowledgements.",
        retention_rows="Untouched labels score fixed snapshots and never train or choose updates.",
        resource_projection="Two disjoint timing pilots budget all24 streams, full cold replay and validation.",
        stream_rows_ready_score="Integrity readiness can equal one even when adaptation harms future decisions.",
    )
    candidate = raw / "candidate.json"
    atomic_json(candidate, value)
    final = terminal(candidate, raw)
    value["validation_receipts"].extend(final["receipts"])
    finished_ns = time.monotonic_ns()
    spans.append(
        dict(
            phase="terminal_scanners_cold_replay",
            started_monotonic_ns=ended_ns,
            ended_monotonic_ns=finished_ns,
            duration_s=(finished_ns - ended_ns) / 1e9,
        )
    )
    value["duration_s"] = (finished_ns - started_ns) / 1e9
    receipt.update(
        ended_monotonic_ns=finished_ns, duration_s=value["duration_s"], phase_spans=spans
    )
    if not final["passed"] or value["duration_s"] > 4200:
        value.update(
            honest_verdict="complete_disqualified_terminal_validation",
            verdict_class="disqualified",
            stream_rows_ready_score=0,
            required_checks_passed=False,
        )
    atomic_json(raw / "terminal_validation.json", final)
    value["terminal_validation"] = old.reference(raw / "terminal_validation.json")
    value["field_principles"]["terminal_validation"] = (
        "Bind actual unchanged scanners and outside-checkout replay logs."
    )
    value = normalize_artifact_for_template_write(value)
    if value["verdict_class"] == "partial":
        # The shared publisher accepts only finished science classes; retain retryable partial evidence.
        atomic_json(args.output, value)
    else:
        publish_primary(
            args.output, value, lambda path: dict(passed=json.loads(path.read_bytes()) == value)
        )
    progress("published", completed, 12096 - completed)
    return 0
