"""REQ-REPORT-8119: batch crossings retain every necessary service cost.

Supplied numerical heads isolate the binding boundary. Captured public requests
and historical acquisition times remain separate from current zero-model work.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import asdict
import json
import os
from pathlib import Path
import random
import shutil
import tempfile
import threading
import time
from typing import Any

import numpy as np

from carnot import experiment_8106_v701_radial_service_cost as host
from carnot.reporting.current_work_receipt import (
    ZERO_INVOCATION_COUNTS,
    atomic_json,
    canonical_hash,
    sha256_file,
)
from carnot.reporting.experiment_7303_validation_scope import (
    CommandSpec,
    build_scoped_commands,
    run_commands,
)
from carnot.reporting.primary_publication import publish_primary, read_bound_sidecar
from carnot.verify import fresh_acquisition_cost_8118 as acquisition
from carnot.verify import native_radial_8105 as k

Json = dict[str, Any]
ROOT = host.ROOT
prior = host.prior
NAME = "experiment_8119_v702_batched_service_cost"
CLI = f"scripts/experiments/{NAME}.py"
TEST = "tests/python/test_batched_service_cost_8119.py"
OWNED = [f"python/carnot/{NAME}.py", CLI]
NATIVE = "results/experiment_8105_v701_native_radial_kernel.json"
ACQUISITION = "results/experiment_8118_v702_fresh_acquisition_cost.json"
LEARNING = "results/experiment_8116_v702_independent_online_memory.json"
ARMS = ("python", "rust")
MODES = ("original", "exact-reuse", "changed-content", "eviction", "restart")
CONFIG: Json = dict(
    seed=7028119, batches=[1, 8, 32, 128], centers=[16, 28], strata=3, repetitions=30, warmups=5
)


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Real counters distinguish completed work from a silent child."""
    print(f"[exp8119] {phase} completed={completed} pending={pending}", flush=True)


def check(
    checks: list[Json], name: str, path: Path, field: str, expected: Any, observed: Any, scope: str
) -> None:
    """Name the actual failed operand so an external block is terminal."""
    checks.append(
        dict(
            check=name,
            upstream=path.stem,
            path=str(path),
            hash=sha256_file(path) if path.is_file() else None,
            artifact_field=field,
            op="==",
            expected=expected,
            observed=observed,
            passed=observed == expected,
            scope=scope,
        )
    )


def inputs(root: Path, raw: Path) -> Json:
    """Authenticate terminal evidence before admitting timing or native bytes."""
    data: Json = dict(
        public=[],
        acquisition=[],
        learning=[],
        library={},
        checks=[],
        refs=[],
        loads=[],
        config=acquisition.config(),
        extractor=host.cache_service.extractor_identity(),
    )
    checks, refs = data["checks"], data["refs"]

    def snapshot(path: Path) -> None:
        destination = raw / "inputs" / (sha256_file(path)[7:] + "-" + path.name)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, destination)
        refs.append(
            dict(path=str(destination), sha256=sha256_file(destination), original_path=str(path))
        )

    def authenticated(label: str, readiness: str, scope: str) -> Json:
        path = root / label
        try:
            value: Json = json.loads(path.read_text())
            snapshot(path)
            terminal = Path(value["terminal_validation_sidecar_path"])
            publication = json.loads(terminal.read_text())["publication"]
            side = Path(publication["sidecar_path"])
            bound = read_bound_sidecar(path, side)
            observations = [
                ("required_checks_passed", True, value.get("required_checks_passed")),
                ("flagged_adversarial", False, value.get("flagged_adversarial")),
                (readiness, 1, value.get(readiness)),
                ("terminal_passed", True, bound["report"]["passed"]),
                ("terminal_primary_hash", sha256_file(path), publication["primary_sha256"]),
            ]
            for field, expected, observed in observations:
                check(checks, scope + "_" + field, path, field, expected, observed, scope)
            snapshot(terminal)
            snapshot(side)
            qualified = all(a == b for _, a, b in observations) and value.get(
                "verdict_class"
            ) not in ("blocked", "disqualified")
            check(checks, scope + "_qualified", path, "qualified", True, qualified, scope)
            return value if qualified else {}
        except (OSError, ValueError, KeyError) as exc:
            check(
                checks, scope + "_qualified", path, "authenticated_terminal", True, str(exc), scope
            )
            return {}

    native = authenticated(NATIVE, "native_kernel_ready_score", "host")
    if native:
        path = Path(native["native_library_path"])
        observed = sha256_file(path) if path.is_file() else None
        check(
            checks,
            "native_binary_hash",
            path,
            "sha256",
            native["native_library_sha256"],
            observed,
            "host",
        )
        if observed == native["native_library_sha256"]:
            data["library"] = dict(path=str(path), sha256=observed)
    captured = authenticated(ACQUISITION, "acquisition_cost_ready_score", "acquisition")
    if captured:
        try:
            ref = captured["acquisition_manifest"]
            path = Path(ref["path"])
            if sha256_file(path) != ref["sha256"]:
                raise ValueError("acquisition_manifest_hash")
            manifest = json.loads(path.read_text())
            rows = manifest["current_acquisition_rows"]
            if (
                rows != captured["current_acquisition_rows"]
                or manifest["capture_budget"] != data["config"]
                or manifest["acquisition_cost_ready_score"] != 1
            ):
                raise ValueError("acquisition_manifest_rows_or_config")
            ledger = {
                r["call_id"]: r for r in captured["call_ledger"] if r["status"] == "completed"
            }
            for row in rows:
                body = json.loads(row["request"]["messages"][1]["content"])
                event = ledger[row["family_id"]]
                if (
                    row["judgment_key"] != acquisition.judgment_key(row, row["key_identity"])
                    or row["key_identity"] != manifest["runtime_identity"]
                    or canonical_hash(row["request"]) != event["request_sha256"]
                    or canonical_hash(row["raw_response"]) != event["response_sha256"]
                    or body["complete_source"].encode().hex() != row["source_bytes"]
                    or body["original_answer"].encode().hex() != row["answer_bytes"]
                    or row["probability"]
                    != json.loads(row["raw_response"]["choices"][0]["message"]["content"])[
                        "unsupported_probability"
                    ]
                ):
                    raise ValueError("acquisition_exact_key_or_call_hash")
            data.update(acquisition=rows, loads=manifest["cold_load_rows"])
            snapshot(path)
        except (OSError, ValueError, KeyError) as exc:
            check(
                checks,
                "acquisition_join",
                root / ACQUISITION,
                "exact_keys_and_hashes",
                True,
                str(exc),
                "acquisition",
            )
    learned = authenticated(LEARNING, "learning_trajectory_ready_score", "learning")
    if learned:
        data["learning"] = learned["update_rows"]
    # Public requests can retain host utility even when acquisition is unavailable.
    rows = data["acquisition"] or [
        dict(
            family_id=f"fixture-{i}-{arm}",
            source_cluster_id=f"fixture-{i}",
            source_bytes=(f"Water contains atoms {i}. " * (i + 1)).encode().hex(),
            answer_bytes=b"Water contains atoms.".hex(),
            arm=arm,
            probability=None,
            key_identity={},
        )
        for i in range(24)
        for arm in ("full_source", "source_order_permuted")
    ]
    ordered = sorted(
        {
            r["source_cluster_id"]: len(bytes.fromhex(r["source_bytes"]))
            + len(bytes.fromhex(r["answer_bytes"]))
            for r in rows
        }.items(),
        key=lambda x: (x[1], x[0]),
    )
    strata = {key: min(2, i * 3 // len(ordered)) for i, (key, _) in enumerate(ordered)}
    data["public"] = [dict(r, stratum=strata[r["source_cluster_id"]]) for r in rows]
    data["panel_source_ids"] = [
        [key for key, _ in ordered if strata[key] == stratum][
            len([key for key, _ in ordered if strata[key] == stratum]) // 2
        ]
        for stratum in range(3)
    ]
    for label in [
        *OWNED,
        TEST,
        "tests/python/test_primary_publication_7928.py",
        "CLAUDE.md",
        "CODEX.md",
        "ops/e2e-test-plan.md",
        "ops/exclusion_manifest.yaml",
        "openspec/change-proposals/research-roadmap-vNEXT.md",
        "scripts/experiment_template.py",
    ]:
        path = ROOT / label
        check(checks, "input_path", path, "exists", True, path.is_file(), "host")
        if path.is_file():
            refs.append(dict(path=str(path), sha256=sha256_file(path)))
    for tool in ("python", "pytest", "coverage", "ruff", "mypy"):
        path = ROOT / ".venv/bin" / tool
        check(checks, "executable_path", path, "executable", True, os.access(path, os.X_OK), "host")
    atomic_json(raw / "input_observations.json", dict(checks=checks, refs=refs))
    return data


def judgment_key(row: Json, config: Json) -> str:
    """Recompute the upstream key from exact public bytes and runtime identity."""
    return canonical_hash(
        dict(
            source_bytes=row["source_bytes"],
            response_bytes=row["answer_bytes"],
            model_runtime_template=row["key_identity"],
            parser=config["parser_sha256"],
            intervention=row["arm"],
            numerical_config=config,
        )
    )


def schedule(config: Json) -> list[Json]:
    """Repeated timings change precision, never the number of independent sources."""
    rng = random.Random(config["seed"])
    slots = [
        dict(
            batch=b,
            centers=c,
            stratum=s,
            mode=m,
            repetition=r,
            unit_id=f"b{b}-c{c}-s{s}-{m}-r{r}",
            order=rng.sample(list(ARMS), 2),
        )
        for b in config["batches"]
        for c in config["centers"]
        for s in range(config["strata"])
        for m in MODES
        for r in range(config["repetitions"])
    ]
    rng.shuffle(slots)
    return slots


def prepare(state: Json) -> tuple[Any, ...]:
    """Both kernel arms prepare fixed model parameters before crossing timings."""
    g = state["geometry"]
    return (
        np.asarray(g["mean"]),
        np.asarray(g["std"]),
        np.asarray([c["x"] for c in state["centers"]]),
        np.asarray(state["coefficients"]),
        g["sigma"],
    )


def kernel(prepared: tuple[Any, ...], values: Any, model: Any, arm: str) -> Any:
    """Equivalent vector predictions include input/output conversion in each arm."""
    if arm == "rust":
        return np.asarray(model.predict(np.asarray(values).tolist()))
    mean, std, centers, theta, sigma = prepared
    x = (np.asarray(values, dtype=np.float64) - mean) / std
    phi = np.exp(-0.5 * np.sum(((x[:, None, :] - centers[None, :, :]) / sigma) ** 2, axis=2))
    return k.expit(theta[0] + phi @ theta[1:])


def transaction(data: Json, native: Any, slot: Json, arm: str, raw: Path) -> Json:
    """Time the complete batch through committed state and returned actions."""
    raw.mkdir(parents=True, exist_ok=True)
    sources = [
        r
        for r in data["public"]
        if r["stratum"] == slot["stratum"]
        and r["arm"] == "full_source"
        and r["source_cluster_id"] in data["panel_source_ids"]
    ]
    selected = [sources[(slot["repetition"] + i) % len(sources)] for i in range(slot["batch"])]
    originals = selected
    if slot["mode"] == "changed-content":
        by = {
            r["source_cluster_id"]: r for r in data["public"] if r["arm"] == "source_order_permuted"
        }
        selected = [by[r["source_cluster_id"]] for r in selected]
    state = prior.fixture(9, slot["centers"], CONFIG["seed"] + slot["stratum"])
    path = raw / "state.json"
    began = time.perf_counter_ns()
    cache = host.cache_service.FeatureCache(
        raw / "features.sqlite", capacity=len(sources), identity=data["extractor"]
    )
    if slot["mode"] != "original":
        for row in originals:
            cache.get({key: row[key] for key in ("family_id", "source_bytes", "answer_bytes")})
    if slot["mode"] == "eviction":
        for i in range(len(sources)):
            cache.get(
                dict(
                    family_id=f"decoy-{i}",
                    source_bytes=b"Water has atoms.".hex(),
                    answer_bytes=b"Water has atoms.".hex(),
                )
            )
    if slot["mode"] == "restart":
        atomic_json(path, dict(state=state, sha256=canonical_hash(state)))
    cache.close()
    setup_ns = time.perf_counter_ns() - began
    started = step = time.perf_counter_ns()
    costs: Json = {}
    cache = host.cache_service.FeatureCache(
        raw / "features.sqlite", capacity=len(sources), identity=data["extractor"]
    )
    if slot["mode"] == "restart":
        state = host.read_state(path)
    costs["lifecycle_ns"] = time.perf_counter_ns() - step
    step = time.perf_counter_ns()
    rendered = [
        json.dumps(
            dict(
                complete_source=bytes.fromhex(r["source_bytes"]).decode(),
                original_answer=bytes.fromhex(r["answer_bytes"]).decode(),
            ),
            sort_keys=True,
        )
        for r in selected
    ]
    costs["rendering_ns"] = time.perf_counter_ns() - step
    step = time.perf_counter_ns()
    vectors = [
        cache.get({key: row[key] for key in ("family_id", "source_bytes", "answer_bytes")})[
            "values"
        ]
        for row in selected
    ]
    events = list(cache.events)
    costs["features_lookup_invalidation_ns"] = time.perf_counter_ns() - step
    step = time.perf_counter_ns()
    known = {r["judgment_key"]: r for r in data["acquisition"]}
    keys = [judgment_key(r, data["config"]) for r in selected]
    judgments = [known.get(key) for key in keys]
    probabilities = np.array([0.5 if row is None else row["probability"] for row in judgments])
    probabilities = np.clip(probabilities, 1e-6, 1 - 1e-6)
    values = np.column_stack((np.log(probabilities / (1 - probabilities)), vectors))
    costs["judgment_lookup_feature_construction_ns"] = time.perf_counter_ns() - step
    step = time.perf_counter_ns()
    model = native.RustRadial8105(json.dumps(state)) if arm == "rust" else None
    prepared = prepare(state)
    costs["model_parameters_ns"] = time.perf_counter_ns() - step
    step = time.perf_counter_ns()
    p = kernel(prepared, values, model, arm)
    costs["kernel_crossing_ns"] = time.perf_counter_ns() - step
    step = time.perf_counter_ns()
    canonical = kernel(prepared, values, None, "python")
    flags = (np.minimum(abs(p - 0.1), abs(p - 0.5)) <= 1e-8) | (
        np.minimum(abs(canonical - 0.1), abs(canonical - 0.5)) <= 1e-8
    )
    p[flags] = canonical[flags]
    actions = [
        k.action(float(v)) if judgment is not None else "abstain"
        for v, judgment in zip(p, judgments, strict=True)
    ]
    costs["canonical_decision_ns"] = time.perf_counter_ns() - step
    step = time.perf_counter_ns()
    state.update(
        version=state["version"] + 1,
        commit_hash=canonical_hash(
            dict(keys=keys, actions=actions, previous=canonical_hash(state))
        ),
    )
    durable = (
        json.loads(native.RustRadial8105(json.dumps(state)).state_json())
        if arm == "rust"
        else state
    )
    costs["metadata_update_serialization_ns"] = time.perf_counter_ns() - step
    step = time.perf_counter_ns()
    atomic_json(path, dict(state=durable, sha256=canonical_hash(durable)))
    directory = os.open(raw, os.O_RDONLY)
    os.fsync(directory)
    os.close(directory)
    cache.close()
    costs["durable_write_ns"] = time.perf_counter_ns() - step
    step = time.perf_counter_ns()
    result = dict(
        arm=arm,
        probabilities=p.tolist(),
        actions=actions,
        fallback_flags=flags.tolist(),
        values=values.tolist(),
        durable_state=durable,
        state_path=str(path),
        rendered_hash=canonical_hash(rendered),
        judgment_keys=keys,
        source_cluster_ids=[r["source_cluster_id"] for r in selected],
        cache_events=events,
        setup_ns=setup_ns,
        components=costs,
        acquisition_available=[r is not None for r in judgments],
    )
    costs["return_ns"] = time.perf_counter_ns() - step
    elapsed = time.perf_counter_ns() - started
    result.update(
        transaction_ns=elapsed,
        numerator=elapsed,
        denominator=1,
        accounting_residual_ns=elapsed - sum(costs.values()),
    )
    return result


def parity(arms: list[Json]) -> bool:
    """Actions and complete durable state must agree, not just floating scores."""
    a, b = arms
    return bool(
        all(
            a[key] == b[key]
            for key in ("actions", "values", "durable_state", "rendered_hash", "judgment_keys")
        )
        and np.allclose(a["probabilities"], b["probabilities"], atol=1e-10, rtol=0)
    )


def measure(data: Json, native: Any, raw: Path, config: Json = CONFIG) -> Json:
    """Five untimed warmups precede thirty randomized pairs in each cell."""
    plan = schedule(config)
    evidence: Json = dict(batch_rows=[], paired_service_rows=[], warmup_rows=[], kernel_operands={})
    atomic_json(
        raw / "frozen_workload.json", dict(config=config, schedule=plan, public=data["public"])
    )
    rng = random.Random(config["seed"])
    for batch in config["batches"]:
        for centers in config["centers"]:
            for stratum in range(config["strata"]):
                cell = f"b{batch}-c{centers}-s{stratum}"
                progress("kernel_benchmark_before_" + cell)
                state = prior.fixture(9, centers, config["seed"] + stratum)
                values = (
                    np.random.default_rng(config["seed"] + stratum).normal(size=(batch, 9)).tolist()
                )
                prepared = prepare(state)
                model = native.RustRadial8105(json.dumps(state))
                evidence["kernel_operands"][cell] = dict(state=state, values=values)
                for repetition in range(-config["warmups"], config["repetitions"]):
                    arms = []
                    for arm in rng.sample(list(ARMS), 2):
                        start = time.perf_counter_ns()
                        probabilities = kernel(prepared, values, model, arm)
                        elapsed = time.perf_counter_ns() - start
                        arms.append(
                            dict(arm=arm, duration_ns=elapsed, probabilities=probabilities.tolist())
                        )
                    pair = dict(
                        unit_id=f"{cell}-kernel-r{repetition}",
                        batch=batch,
                        centers=centers,
                        stratum=stratum,
                        cell=cell,
                        arms=arms,
                        error=float(
                            np.max(
                                abs(np.array(arms[0]["probabilities"]) - arms[1]["probabilities"])
                            )
                        ),
                        repetition=repetition,
                    )
                    evidence["batch_rows" if repetition >= 0 else "warmup_rows"].append(pair)
                progress("kernel_benchmark_after_" + cell, config["repetitions"], 0)
    warmed: set[tuple[Any, ...]] = set()
    progress("service_benchmark_before", 0, len(plan))
    with (raw / "service_pairs.jsonl").open("w") as stream:
        for index, slot in enumerate(plan):
            progress("batch_before", index, len(plan) - index)
            cell_key = tuple(slot[key] for key in ("batch", "centers", "stratum", "mode"))
            if cell_key not in warmed:
                for repetition in range(config["warmups"]):
                    arms = [
                        transaction(data, native, slot, arm, raw / "warmup" / slot["unit_id"] / arm)
                        for arm in slot["order"]
                    ]
                    evidence["warmup_rows"].append(
                        dict(
                            unit_id=slot["unit_id"],
                            repetition=repetition,
                            condition=slot["mode"],
                            durations_ns=[r["transaction_ns"] for r in arms],
                            excluded_from_measurement=True,
                        )
                    )
                warmed.add(cell_key)
            arms = [
                transaction(data, native, slot, arm, raw / "transactions" / slot["unit_id"] / arm)
                for arm in slot["order"]
            ]
            pair = dict(
                slot,
                arms=arms,
                parity=parity(arms),
                error=float(
                    np.max(abs(np.array(arms[0]["probabilities"]) - arms[1]["probabilities"]))
                ),
                ratio=next(r["transaction_ns"] for r in arms if r["arm"] == "python")
                / next(r["transaction_ns"] for r in arms if r["arm"] == "rust"),
            )
            evidence["paired_service_rows"].append(pair)
            stream.write(json.dumps(pair) + "\n")
            stream.flush()
            progress("batch_after", index + 1, len(plan) - index - 1)
    atomic_json(raw / "primitive_rows.json", evidence)
    progress("service_benchmark_after", len(plan), 0)
    return evidence


def reduce_rows(evidence: Json, config: Json = CONFIG) -> Json:
    """All ratios and readiness reduce from individual paired observations."""
    pairs = evidence["paired_service_rows"]
    expected = {r["unit_id"] for r in schedule(config)}
    failures = 0
    summaries = []
    for pair in pairs:
        failures += int(
            not parity(pair["arms"])
            or pair["error"] > 1e-10
            or any(
                r["transaction_ns"] <= 0
                or r["numerator"] != r["transaction_ns"]
                or r["accounting_residual_ns"]
                != r["transaction_ns"] - sum(r["components"].values())
                or r["accounting_residual_ns"] < 0
                for r in pair["arms"]
            )
        )
    for kind, rows in (("kernel", evidence["batch_rows"]), ("host", pairs)):
        for batch in config["batches"]:
            for centers in config["centers"]:
                for stratum in range(config["strata"]):
                    for mode in ["kernel"] if kind == "kernel" else MODES:
                        selected = [
                            r
                            for r in rows
                            if r["batch"] == batch
                            and r["centers"] == centers
                            and r["stratum"] == stratum
                            and r.get("mode", "kernel") == mode
                        ]
                        if not selected:
                            continue
                        metric = "duration_ns" if kind == "kernel" else "transaction_ns"
                        py, rust = [
                            [
                                next(a[metric] for a in r["arms"] if a["arm"] == arm)
                                for r in selected
                            ]
                            for arm in ARMS
                        ]
                        ratios = [a / max(1, b) for a, b in zip(py, rust, strict=True)]
                        summaries.append(
                            dict(
                                kind=kind,
                                batch=batch,
                                centers=centers,
                                stratum=stratum,
                                mode=mode,
                                paired_count=len(selected),
                                python_median_ns=float(np.median(py)),
                                rust_median_ns=float(np.median(rust)),
                                python_p95_ns=float(np.percentile(py, 95)),
                                rust_p95_ns=float(np.percentile(rust, 95)),
                                median_ratio=float(np.median(ratios)),
                                p95_ratio=float(np.percentile(ratios, 95)),
                                p95_latency_ratio=float(
                                    np.percentile(py, 95) / np.percentile(rust, 95)
                                ),
                                max_error=max(r["error"] for r in selected),
                            )
                        )
    kernel_expected = (
        len(config["batches"]) * len(config["centers"]) * config["strata"] * config["repetitions"]
    )
    kernel_ok = (
        len(evidence["batch_rows"]) == kernel_expected
        and len({r["unit_id"] for r in evidence["batch_rows"]}) == kernel_expected
        and all(
            r["error"] <= 1e-10 and all(a["duration_ns"] > 0 for a in r["arms"])
            for r in evidence["batch_rows"]
        )
    )
    python = [next(a for a in p["arms"] if a["arm"] == "python") for p in pairs]
    total = sum(r["transaction_ns"] + r["setup_ns"] for r in python)
    fraction = sum(r["components"]["kernel_crossing_ns"] for r in python) / max(1, total)
    return dict(
        passed=len(pairs) == len(expected)
        and {r["unit_id"] for r in pairs} == expected
        and failures == 0
        and kernel_ok,
        completed_count=len(pairs),
        failed_count=failures,
        summaries=summaries,
        arithmetic_fraction=fraction,
        amdahl_infinite_bound=1 / (1 - fraction),
        amdahl_100x_kernel_bound=1 / (1 - fraction + fraction / 100),
        host_workload_throughput_ratio=sum(r["transaction_ns"] + r["setup_ns"] for r in python)
        / max(
            1,
            sum(
                a["transaction_ns"] + a["setup_ns"]
                for p in pairs
                for a in p["arms"]
                if a["arm"] == "rust"
            ),
        ),
    )


def accounting(data: Json, evidence: Json) -> list[Json]:
    """Charge exact acquired keys, observed batch reuse and fresh-every-call costs."""
    known = {r["judgment_key"]: r for r in data["acquisition"]}
    load_s = sum((r["ended_monotonic_ns"] - r["started_monotonic_ns"]) / 1e9 for r in data["loads"])
    result = []
    for pair in evidence["paired_service_rows"]:
        for row in pair["arms"]:
            keys = row["judgment_keys"]
            unique = set(keys)
            available = all(key in known for key in keys)
            cost = lambda key: (
                known[key]["component_costs"]["request_wall_s"]
                + (
                    known[key]["component_costs"]["parsing_ns"]
                    + known[key]["component_costs"]["cache_write_ns"]
                )
                / 1e9
            )
            acquired = sum(cost(key) for key in unique) if available else None
            no_reuse = sum(cost(key) for key in keys) if available else None
            host_s = (row["setup_ns"] + row["transaction_ns"]) / 1e9
            result.append(
                dict(
                    unit_id=pair["unit_id"],
                    arm=row["arm"],
                    condition=pair["mode"],
                    batch=pair["batch"],
                    judgment_keys=sorted(unique),
                    acquisition_upstream=ACQUISITION,
                    matched=available,
                    cold_load_s=load_s if available else None,
                    unique_key_count=len(unique),
                    observed_reuse_count=len(keys) - len(unique),
                    acquisition_s=acquired,
                    host_s=host_s,
                    composed_total_s=load_s + acquired + host_s if available else None,
                    no_reuse_total_s=load_s + no_reuse + host_s if available else None,
                    amortized_per_request_s=(load_s + acquired + host_s) / len(keys)
                    if available
                    else None,
                    learning_update_s=None if not data["learning"] else "unjoined_update_keys",
                    directly_measured_complete_service=False,
                    exclusion_reason=None if available else "exact_acquisition_key_unavailable",
                )
            )
    return result


def worker(root: Path, raw: Path, config: Json = CONFIG) -> Json:
    """Only authenticated binding bytes can enter the normally exiting worker."""
    began = time.monotonic()
    progress("preconditions_before")
    data = inputs(root, raw)
    phases = [dict(phase="preconditions", duration_s=time.monotonic() - began)]
    progress("preconditions_after")
    evidence: Json = dict(batch_rows=[], paired_service_rows=[], warmup_rows=[], kernel_operands={})
    loaded: Json = {}
    if data["library"] and all(c["passed"] for c in data["checks"] if c["scope"] == "host"):
        start = time.monotonic()
        progress("binding_load_before")
        native, loaded = host.load_binding(data)
        loaded["actual_loaded_sha256"] = sha256_file(Path(native.__file__))
        loaded["path_matches_actual"] = (
            Path(native.__file__).resolve() == Path(loaded["path"]).resolve()
        )
        if loaded["actual_loaded_sha256"] != loaded["sha256"] or not loaded["path_matches_actual"]:
            raise ValueError("actual_loaded_binding_identity")
        progress("binding_load_after")
        phases.append(dict(phase="binding_load", duration_s=time.monotonic() - start))
        start = time.monotonic()
        evidence = measure(data, native, raw, config)
        phases.append(dict(phase="benchmarks", duration_s=time.monotonic() - start))
    hashes = {p: sha256_file(ROOT / p) for p in [*OWNED, TEST]}
    for module in (
        host,
        prior,
        k,
        host.cache_service,
        host.cache_service.features,
        host.cache_service.alignment,
        acquisition,
    ):
        hashes[str(Path(module.__file__).relative_to(ROOT))] = sha256_file(Path(module.__file__))
    hashes["config"] = canonical_hash(config)
    work = dict(
        data=data,
        evidence=evidence,
        config=config,
        loaded_binding_receipt=loaded,
        duration_s=time.monotonic() - began,
        code_config_hashes=hashes,
        phase_spans=phases,
    )
    atomic_json(raw / "work.json", work)
    return work


def build(work: Json, receipts: list[Json], raw: Path, reduced: Json) -> Json:
    """A missing external stage preserves qualified host measurements."""
    data, evidence, config = work["data"], work["evidence"], work["config"]
    owned = bool(receipts) and all(
        r["passed"] for r in receipts if r.get("scope") != "repository_health"
    )
    host_block = any(not c["passed"] for c in data["checks"] if c["scope"] == "host")
    ready = int(owned and not host_block and reduced["passed"])
    joins = accounting(data, evidence)
    complete = int(
        ready
        and bool(data["learning"])
        and bool(joins)
        and all(r["matched"] and isinstance(r["learning_update_s"], (float, int)) for r in joins)
    )
    gates = deepcopy(data["checks"])
    check(
        gates,
        "complete_learning_update_join",
        ROOT / LEARNING,
        "qualified_key_matched_update_rows",
        True,
        bool(complete),
        "learning",
    )
    verdict = (
        "disqualified"
        if not owned or (not host_block and not reduced["passed"])
        else "blocked"
        if not complete
        else "circular_positive"
    )
    reason = next((c["check"] for c in gates if not c["passed"]), "fixture_batch_panel")
    if verdict == "disqualified":
        reason = "owned_validation"
    pairs = evidence["paired_service_rows"]
    rows = [
        dict(
            unit_id=p["unit_id"],
            source_cluster_id=f"exposed-stratum-{p['stratum']}",
            arm="paired_python_rust",
            condition=p["mode"],
            metric="host_batch_latency_ratio",
            numerator=next(a["transaction_ns"] for a in p["arms"] if a["arm"] == "python"),
            denominator=next(a["transaction_ns"] for a in p["arms"] if a["arm"] == "rust"),
            status="completed",
            exclusion_reason=None,
        )
        for p in pairs
    ]
    kernel_rows = evidence["batch_rows"]
    py_kernel = sum(
        a["duration_ns"] for p in kernel_rows for a in p["arms"] if a["arm"] == "python"
    )
    rust_kernel = sum(
        a["duration_ns"] for p in kernel_rows for a in p["arms"] if a["arm"] == "rust"
    )
    value: Json = dict(
        experiment_id=8119,
        task_id="exp8119-batched-service-cost",
        schema="carnot.v702.batched_service_cost.v1",
        run_date="20261004",
        honest_verdict=f"complete_{verdict}_{reason}",
        verdict_class=verdict,
        verifier_is_oracle=True,
        claim_scope="fixture-only numerical head; exposed public host service costs and historical stage composition",
        exposure_scope="exposed_development",
        independent_generalization_score=0,
        generalized_learning_benefit_score=0,
        required_checks_passed=owned,
        flagged_adversarial=False,
        validation_receipts=receipts,
        terminal_validation_sidecar_path=str(raw / "terminal_validation.json"),
        gate_check_summary=gates,
        inference_substrate="verifier_ensemble_against_cached_candidates"
        if pairs
        else "aggregation_from_upstream_artifacts",
        inference_substrate_class="no_model_load",
        MODEL_SPECS=[],
        model_invocation_counts=deepcopy(ZERO_INVOCATION_COUNTS),
        call_ledger=[],
        trained_head_specs=[
            dict(kind="supplied_numerical_fixture", centers=c, trained_currently=False)
            for c in config["centers"]
        ],
        rows=rows,
        intended_count=len(schedule(config)),
        eligible_count=len(pairs),
        independent_count=0,
        completed_count=len(pairs),
        excluded_count=len(schedule(config)) - len(pairs),
        censored_count=0,
        failed_count=reduced["failed_count"],
        sample_size_budget=dict(
            config,
            unique_exposed_source_clusters=len({r["source_cluster_id"] for r in data["public"]}),
            measured_exposed_source_clusters=len(data["panel_source_ids"]),
            independent_scientific_sources=0,
        ),
        duration_s=work["duration_s"],
        random_seed=config["seed"],
        reproducibility_checksum=canonical_hash(work),
        source_artifact_hashes=data["refs"],
        raw_shard_hashes=[],
        code_config_hashes=work["code_config_hashes"],
        phase_spans=work["phase_spans"],
        host_service_ready_score=ready,
        complete_service_ready_score=complete,
        loaded_binding_receipt=work["loaded_binding_receipt"],
        batch_rows=kernel_rows,
        paired_service_rows=pairs,
        acquisition_join_rows=joins,
        historical_acquisition_rows=data["acquisition"],
        historical_cold_load_rows=data["loads"],
        component_cost_rows=[
            dict(
                unit_id=p["unit_id"],
                arm=a["arm"],
                **a["components"],
                residual_ns=a["accounting_residual_ns"],
                setup_ns=a["setup_ns"],
            )
            for p in pairs
            for a in p["arms"]
        ],
        restart_rows=[p for p in pairs if p["mode"] == "restart"],
        warmup_rows=evidence["warmup_rows"],
        kernel_operands=evidence["kernel_operands"],
        accounting_identity="host_batch_ns = sum(nonoverlapping components_ns) + residual_ns; cold composition = model_load + exact_unique_acquisition + setup + host; learning update unavailable",
        amortization_assumptions=dict(
            observed_reuse="duplicates within each batch",
            cold_load_charged_once_per_batch_scenario=True,
            no_reuse="fresh acquisition for every request, one cold load",
            estimates_are_stage_sums=True,
            metadata_commit_is_not_learning=True,
        ),
        kernel_speedup=py_kernel / rust_kernel if rust_kernel else None,
        whole_service_speedup=None,
        host_service_speedup=reduced["host_workload_throughput_ratio"],
        nfr01_met=False,
        arithmetic_fraction=reduced["arithmetic_fraction"],
        measured_amdahl_bounds=dict(
            infinite_arithmetic=reduced["amdahl_infinite_bound"],
            kernel_100x=reduced["amdahl_100x_kernel_bound"],
            scope="host workload including setup; absent learning and acquisition lower arithmetic fraction further",
        ),
        reduction=reduced,
        measurement_config=config,
        public_request_size_strata=[
            dict(
                stratum=s,
                source_cluster_id=key,
                request_bytes=len(bytes.fromhex(row["source_bytes"]))
                + len(bytes.fromhex(row["answer_bytes"])),
                selection="median-sized original in each public byte-size third",
            )
            for s, key in enumerate(data["panel_source_ids"])
            for row in data["public"]
            if row["source_cluster_id"] == key and row["arm"] == "full_source"
        ],
        acceptance_gates=dict(
            equal_correctness_atol=1e-10,
            owned_checks_required=True,
            complete_service_required=True,
            nfr01_throughput_threshold=10,
            hardware_throughput_target=100,
        ),
        methodology_note="Thirty paired repetitions after five warmups per batch/center/byte-size/cache cell. Seeded randomized vectorized NumPy versus qualified Rust predict. Actual rendering, lexical feature cache, decision and fsynced metadata return are charged. Heads are supplied fixtures; disqualified Exp8116 update rows cannot be charged. Exp8118 costs are exact historical stage sums with observed within-batch reuse, not current model work or directly measured complete service.",
    )
    value["field_principles"] = {
        key: f"{key} binds observed work and does not grant independent learning or full-service credit."
        for key in value
    }
    value["field_principles"].update(
        host_service_ready_score="Owned host checks survive missing external learning costs.",
        complete_service_ready_score="All necessary acquisition and actual learning update rows must qualify; unavailable costs are never zero.",
        kernel_speedup="A function ratio cannot satisfy NFR-01.",
        arithmetic_fraction="Measured host components bound any hardware gain by Amdahl's law.",
        independent_count="Exposed requests, synthetic heads, batch duplicates and repetitions create no independent scientific sources.",
    )
    return value


def replay(path: Path) -> bool:
    """Cold recomputation rejects changed primitives, checkpoints and reductions."""
    try:
        value = json.loads(path.read_text())
        for ref in [*value["raw_shard_hashes"], *value["source_artifact_hashes"]]:
            if sha256_file(Path(ref["path"])) != ref["sha256"]:
                return False
        for label, digest in value["code_config_hashes"].items():
            if label != "config" and sha256_file(ROOT / label) != digest:
                return False
        evidence = {
            key: value[key]
            for key in ("batch_rows", "paired_service_rows", "warmup_rows", "kernel_operands")
        }
        if reduce_rows(evidence, value["measurement_config"]) != value["reduction"]:
            return False
        if not evidence["paired_service_rows"]:
            return value["host_service_ready_score"] == 0
        primitive = next(
            Path(r["path"])
            for r in value["raw_shard_hashes"]
            if Path(r["path"]).name == "primitive_rows.json"
        )
        if json.loads(primitive.read_text()) != evidence:
            return False
        saved_work = next(
            Path(r["path"])
            for r in value["raw_shard_hashes"]
            if Path(r["path"]).name == "work.json"
        )
        data = json.loads(saved_work.read_text())["data"]
        if accounting(data, evidence) != value["acquisition_join_rows"]:
            return False
        native, loaded = host.load_binding(dict(library=value["loaded_binding_receipt"]))
        if sha256_file(Path(native.__file__)) != loaded["sha256"]:
            return False
        for pair in evidence["batch_rows"]:
            operands = evidence["kernel_operands"][pair["cell"]]
            expected = kernel(prepare(operands["state"]), operands["values"], None, "python")
            if any(
                not np.allclose(a["probabilities"], expected, atol=1e-10, rtol=0)
                for a in pair["arms"]
            ):
                return False
        for index, pair in enumerate(evidence["paired_service_rows"]):
            if index % 100 == 0:
                progress("cold_replay", index, len(evidence["paired_service_rows"]) - index)
            for row in pair["arms"]:
                state = host.read_state(Path(row["state_path"]))
                restored = native.RustRadial8105.restore(
                    native.RustRadial8105(json.dumps(state)).checkpoint()
                )
                if (
                    state != row["durable_state"]
                    or json.loads(restored.state_json()) != state
                    or not np.allclose(
                        restored.predict(row["values"]), row["probabilities"], atol=1e-10, rtol=0
                    )
                ):
                    return False
        for receipt in value["validation_receipts"]:
            if (
                receipt.get("log_path")
                and sha256_file(Path(receipt["log_path"])) != receipt["log_sha256"]
            ):
                return False
        return True
    except (OSError, ValueError, KeyError, TypeError, StopIteration):
        return False


def validation_plan(private: Path) -> list[CommandSpec]:
    """Freeze owned, consumer and key-custody checks before any measurement."""
    commands = build_scoped_commands(
        ROOT,
        [
            TEST,
            "tests/python/test_primary_publication_7928.py",
            "tests/python/test_native_radial_8105.py::test_req_verify_8105_seeded_numpy",
            "tests/python/test_native_radial_8105.py::test_req_verify_8105_boundaries",
            "tests/python/test_native_radial_8105.py::test_req_verify_8105_serialization",
            "tests/python/test_radial_service_cost_8106.py::test_schedule_and_keys",
            "tests/python/test_radial_service_cost_8106.py::test_transactions",
            "tests/python/test_radial_service_cost_8106.py::test_state_hash_and_fallback",
            "tests/python/test_radial_service_cost_8106.py::test_binding_hash_rejected",
            "tests/python/test_source_boundary_7852.py",
            "tests/python/test_experiment_7942_v689_sentence_labels.py",
        ],
        OWNED[:1],
        static_paths=OWNED[1:],
        basetemp=private,
        coverage_file=private / ".coverage",
    )
    return [
        CommandSpec(
            c.name,
            tuple(f"--include=*/{NAME}.py" if a.startswith("--include=") else a for a in c.argv)
            + ("--strict", "--follow-imports=silent")
            if c.name == "changed_module_mypy"
            else (*c.argv[:3], *sorted({p.split("::")[0] for p in c.argv[3:]}))
            if c.name == "scoped_spec_coverage"
            else (*c.argv[:2], *sorted({p.split("::")[0] for p in c.argv[2:]}))
            if c.name == "ruff_check"
            else (*c.argv[:3], *sorted({p.split("::")[0] for p in c.argv[3:]}))
            if c.name == "ruff_format"
            else tuple(
                f"--include=*/{NAME}.py" if a.startswith("--include=") else a for a in c.argv
            ),
            c.scope,
            300,
        )
        for c in commands
    ]


def wait_progress(stop: Any, name: str, completed: int, pending: int) -> None:
    """A waiting child remains one pending command until its actual exit."""
    while not stop.wait(30):
        progress("waiting_" + name, completed, pending)


def execute(commands: list[CommandSpec], raw: Path, private: Path) -> list[Json]:
    """Bound every child with thirty-second heartbeat and byte-bound transcripts."""
    receipts = []
    for index, command in enumerate(commands):
        progress("subprocess_before_" + command.name, index, len(commands) - index)
        stop = threading.Event()
        monitor = threading.Thread(
            target=wait_progress,
            args=(stop, command.name, index, len(commands) - index),
            daemon=True,
        )
        monitor.start()
        receipts += run_commands(
            ROOT,
            [command],
            log_dir=raw / "validation_logs" / command.name,
            heartbeat_s=30,
            extra_env={
                "CARNOT_8119_E2E_RECEIPTS": str(raw / "private_cli_receipts.json"),
                "JAX_PLATFORMS": "cpu",
            },
        )
        stop.set()
        monitor.join(timeout=1)
        progress("subprocess_after_" + command.name, index + 1, len(commands) - index - 1)
    for receipt in receipts:
        receipt["log_path"] = str((ROOT / receipt["log_path"]).resolve())
        receipt["normal_exit"] = not receipt["timed_out"] and receipt["exit_code"] >= 0
    return receipts


def terminal(path: Path) -> Json:
    """Only cold replay and both established artifact readers permit publication."""
    py = str(ROOT / ".venv/bin/python")
    commands = [
        CommandSpec(
            "cold_replay",
            (py, "-u", str(ROOT / CLI), "--cold-replay", str(path)),
            "private_cli",
            180,
        ),
        CommandSpec(
            "adversarial",
            (py, str(ROOT / "scripts/adversarial_verify.py"), "--json", str(path)),
            "terminal",
            120,
        ),
        CommandSpec(
            "strict_rows",
            (py, str(ROOT / "scripts/verdict_row_consistency_lint.py"), "--strict", str(path)),
            "terminal",
            120,
        ),
    ]
    with tempfile.TemporaryDirectory(prefix="carnot-8119-terminal-") as tmp:
        receipts = execute(commands, path.parent, Path(tmp))
    report = dict(passed=all(r["passed"] for r in receipts), receipts=receipts)
    atomic_json(path.parent / "owned_terminal_report.json", report)
    return report


def disqualified_report(path: Path, error: str) -> Json:
    """A failure record can publish only when every science readiness is zero."""
    value = json.loads(path.read_text())
    return dict(
        passed=value.get("verdict_class") == "disqualified"
        and value.get("host_service_ready_score") == 0
        and value.get("complete_service_ready_score") == 0
        and value.get("required_checks_passed") is False,
        failed_owned_terminal=error,
        receipts=[],
        scope="failure_record_only",
    )


def main(argv: list[str] | None = None) -> int:
    """Preserve previous work and publish only normally exited checked bytes."""
    os.environ["PYTHONUNBUFFERED"] = "1"
    began = time.monotonic()
    progress("start")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", choices=["20261004"], default="20261004")
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, default=ROOT / "results" / (NAME + ".json"))
    parser.add_argument("--fixture-output", type=Path)
    parser.add_argument("--fixture-small", action="store_true")
    parser.add_argument("--worker-output", type=Path)
    parser.add_argument("--cold-replay", type=Path)
    parser.add_argument("--retry-owned-validation", action="store_true")
    args = parser.parse_args(argv)
    if args.cold_replay:
        return 0 if replay(args.cold_replay) else 1
    config = (
        dict(CONFIG, batches=[1, 8], centers=[16], repetitions=2, warmups=1)
        if args.fixture_small
        else CONFIG
    )
    if args.fixture_small and not (args.fixture_output or args.worker_output):
        parser.error("--fixture-small requires a private fixture output")
    if args.worker_output:
        worker(args.root, args.worker_output.parent, config)
        return 0
    output = (args.fixture_output or args.output).absolute()
    raw = output.parent / "raw" / output.stem
    base_raw = raw
    if args.retry_owned_validation:
        previous = json.loads(output.read_text())
        if (
            previous["verdict_class"] != "disqualified"
            or previous["required_checks_passed"] is not False
        ):
            raise ValueError("retry_requires_owned_failure")
        raw = base_raw / "invocations" / str(time.time_ns())
        atomic_json(raw / "superseded_primary.json", previous)
    if (output.exists() and not args.retry_owned_validation) or (raw / "work.json").exists():
        progress("existing_evidence_preserved")
        return 1
    with tempfile.TemporaryDirectory(prefix="carnot-8119-") as tmp:
        private = Path(tmp)
        commands = validation_plan(private)
        child = CommandSpec(
            "measurement_normal_exit",
            (
                str(ROOT / ".venv/bin/python"),
                "-u",
                str(ROOT / CLI),
                "--root",
                str(args.root),
                "--worker-output",
                str(raw / "work.json"),
                *(("--fixture-small",) if args.fixture_small else ()),
            ),
            "private_cli",
            900,
        )
        health = CommandSpec(
            "repository_health_once",
            (str(ROOT / ".venv/bin/pytest"), "tests/python", "-q"),
            "repository_health",
            120,
        )
        atomic_json(
            raw / "validation_commands.json",
            dict(
                commands=[asdict(c) for c in commands],
                measurement=asdict(child),
                repository_health=asdict(health),
                config=config,
                code_hashes={p: sha256_file(ROOT / p) for p in [*OWNED, TEST]},
            ),
        )
        receipts = execute([child], raw, private)
        if not (raw / "work.json").is_file():
            progress("worker_failed_without_evidence")
            return 1
        work = json.loads((raw / "work.json").read_text())
        start = time.monotonic()
        if not args.fixture_output:
            receipts += execute(commands, raw, private)
            previous_health = base_raw / "repository_health_once.json"
            health_rows = (
                json.loads(previous_health.read_text())["receipts"]
                if args.retry_owned_validation and previous_health.is_file()
                else execute([health], raw / "global_health", private)
            )
            atomic_json(raw / "repository_health_once.json", dict(receipts=health_rows))
        if (raw / "private_cli_receipts.json").is_file():
            receipts += json.loads((raw / "private_cli_receipts.json").read_text())["receipts"]
        work["duration_s"] = time.monotonic() - began
        work["phase_spans"].append(
            dict(phase="owned_validation", duration_s=time.monotonic() - start)
        )
        atomic_json(raw / "work.json", work)
        progress("independent_reduction_before")
        reduced = reduce_rows(work["evidence"], config)
        atomic_json(raw / "independent_reduction.json", reduced)
        progress("independent_reduction_after")
        value = build(work, receipts, raw, reduced)
        if args.fixture_output:
            value["fixture_protocol_only"] = True
        value["raw_shard_hashes"] = [
            dict(path=str(p), sha256=sha256_file(p))
            for p in raw.glob("*.json")
            if p.name not in ("terminal_candidate.json", "terminal_validation.json")
        ]
        if (raw / "repository_health_once.json").is_file():
            value["repository_health"] = json.loads(
                (raw / "repository_health_once.json").read_text()
            )["receipts"]
        value["field_principles"].update(
            {
                key: "Binds actual execution; global health remains separate from owned qualification."
                for key in value
                if key not in value["field_principles"]
            }
        )
        progress("publication_before")
        try:
            publication = publish_primary(output, value, terminal)
        except ValueError as exc:
            candidate = raw / "terminal_candidate.json"
            if candidate.is_file():
                shutil.copyfile(candidate, raw / "failed_terminal_candidate.json")
            atomic_json(raw / "failed_terminal_report.json", dict(error=str(exc)))
            value.update(
                host_service_ready_score=0,
                complete_service_ready_score=0,
                required_checks_passed=False,
                verdict_class="disqualified",
                honest_verdict="complete_disqualified_terminal_validation",
            )
            publication = publish_primary(
                output,
                value,
                lambda p: disqualified_report(p, str(exc)),
            )
        atomic_json(
            raw / "terminal_validation.json",
            dict(
                publication=publication,
                measurement_exit_receipt=receipts[0],
                normal_exit=receipts[0]["exit_code"] == 0,
            ),
        )
        progress("publication_after", value["completed_count"], 0)
    return 0
