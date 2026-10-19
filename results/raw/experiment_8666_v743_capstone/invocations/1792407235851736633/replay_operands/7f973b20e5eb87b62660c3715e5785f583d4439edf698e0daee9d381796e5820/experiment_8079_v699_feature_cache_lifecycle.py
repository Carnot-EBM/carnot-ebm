"""REQ-REPORT-8079: measure durable cache lifetimes on qualified transactions."""

from __future__ import annotations

from contextlib import contextmanager
import copy
from dataclasses import replace
import json
import math
import os
from pathlib import Path
import sqlite3
import time
from typing import Any, Iterator

from carnot import experiment_8078_v699_feature_cache_core as core

Json = dict[str, Any]
legacy = core.legacy
ROOT = core.ROOT
NAME = "experiment_8079_v699_feature_cache_lifecycle"
TASK = "exp8079-feature-cache-lifecycle"
SCRIPT = f"scripts/experiments/{NAME}.py"
TEST = "tests/python/test_feature_cache_lifecycle_8079.py"
OWNED = [f"python/carnot/{NAME}.py", SCRIPT]
MODES = ("changed_10pct", "eviction", "restart")
CONFIG: Json = dict(core.CONFIG)
START = time.monotonic()
HEALTH_PATH = Path("/tmp/exp8079-repository-health.json")
FIRST_PATH = Path("/tmp/exp8079-tests-first.log")
ORIGINAL = {
    name: getattr(core, name)
    for name in ("base", "measure", "replay", "validation_plan", "run_commands")
}
load_inputs = core.load_inputs


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Expose real work counts while the bounded measurement is running."""
    print(
        f"[exp8079] phase={phase} elapsed_s={time.monotonic() - START:.3f} "
        f"completed={completed} pending={pending}",
        flush=True,
    )


def schedule(data: Json) -> list[Json]:
    """Keep every sealed slot so a slower mode cannot disappear from the result."""
    if data["service_matrix"] != core.sealed.service_matrix():
        raise ValueError("service_matrix")
    return [r for r in data["service_matrix"]["rows"] if r["partition"] == "lifecycle"]


def base(failures: list[Json]) -> Json:
    """A complete slow service has readiness without a general speed claim."""
    value: Json = ORIGINAL["base"](failures)
    value.update(
        experiment_id=8079,
        task_id=TASK,
        schema="carnot.v699.feature_cache_lifecycle.v1",
        cache_lifecycle_ready_score=0,
        process_restart_rows=[],
        identity_check_rows=[],
        complete_service_ratios=[],
    )
    value.pop("cache_core_ready_score")
    return value


def cache_snapshot(cache: Any) -> list[Any]:
    """Compare all persisted rows after restart, including unused entries."""
    return [
        list(row) for row in cache.db.execute("SELECT key,vector,digest FROM features ORDER BY key")
    ]


def restart_check(payload: Json) -> Json:
    """Reopen committed cache and state in an executable child with its own imports."""
    calls = 0
    if payload["cache"]:
        cache = legacy.c.FeatureCache(Path(payload["cache"]))
        try:
            snapshot = cache_snapshot(cache)
            if snapshot != payload["snapshot"] or any(
                digest != legacy.canonical_hash(dict(key=key, values=json.loads(vector)))
                for key, vector, digest in snapshot
            ):
                raise ValueError("restart_cache_drift")
        finally:
            cache.close()
    if payload["checkpoint"]:
        path = legacy.checked(payload["checkpoint"])
        state = json.loads(path.read_bytes())
        with sqlite3.connect(path.with_suffix(".sqlite")) as journal:
            if journal.execute("SELECT payload FROM events").fetchone()[0] != path.read_bytes():
                raise ValueError("restart_journal_drift")
        native = None
        if payload["arm"] == "native":
            native = legacy.old.prior.old.build.load_native_extension(
                legacy.checked(payload["library"])
            )
            calls = 1
        check = legacy.old.m.restart_decisions(state["head"], state["guard_design"], native)
        if not check["passed"] or state["head"]["parameters"] != state["acceptance"]["parameters"]:
            raise ValueError("restart_state_drift")
    return dict(
        passed=True,
        child_pid=os.getpid(),
        native_calls=calls,
        cache_entries=len(payload["snapshot"]),
        checkpoint=payload["checkpoint"],
    )


def identity_checks(raw: Path) -> list[Json]:
    """Mutations must miss, and corrupted public vectors must be recomputed."""
    source = dict(
        family_id="private-control",
        source_bytes=b"Water is wet.".hex(),
        answer_bytes=b"Water is wet.".hex(),
    )
    cache = legacy.c.FeatureCache(raw / "identity.sqlite")
    first = cache.get(source)
    rows = []
    for operand in ("source_bytes", "answer_bytes", "config", "schema"):
        changed = copy.deepcopy(source)
        identity = copy.deepcopy(cache.identity)
        if operand in changed:
            changed[operand] += "20"
        else:
            identity[operand] = {"mutation": True}
        other = legacy.c.FeatureCache(cache.path, identity=identity)
        key = other.key(changed)
        value = other.get(changed)
        rows.append(
            dict(
                operand=operand,
                passed=key != cache.key(source)
                and other.events[-1]["status"] == "miss"
                and value == first,
            )
        )
        other.close()
    cache.db.execute("UPDATE features SET digest='corrupt'")
    cache.db.commit()
    restored = cache.get(source)
    rows.append(
        dict(
            operand="corruption",
            passed=restored == first and cache.events[-1]["status"] == "corrupt_miss",
        )
    )
    cache.close()
    legacy.atomic_json(raw / "identity_checks.json", dict(rows=rows))
    return rows


def measure(data: Json, native: Any, raw: Path) -> Json:
    """Reuse atomic quartet scheduling while measuring each lifecycle boundary."""
    caches: dict[tuple[str, str, str, str], Any] = {}
    population: list[Json] = []
    restarts: list[Json] = []

    def transaction(
        current: Json,
        w: Json,
        binding: Any,
        choice: str,
        mode: str,
        unused: Any,
        directory: Path,
        unit: str,
    ) -> tuple[Json, list[Json]]:
        arm, cache_mode = choice.split("_")
        cached = cache_mode == "cached"
        rep = int(unit.split("/")[-2])
        indices = sorted(set(w["updates"] + [g[0] for g in w["guards"]]))
        changed = [i for position, i in enumerate(indices) if (position + rep) % 10 == 0]
        current = dict(current, sources=copy.deepcopy(current["sources"]))
        if mode == "changed_10pct":
            for i in changed:
                current["sources"][i]["source_bytes"] += "20" * (rep + 6)
        else:
            changed = []
        key = (mode, w["arm"], w["class"], arm)
        creation = 0
        cache = None
        if cached and key not in caches:
            began = time.perf_counter_ns()
            cache = legacy.c.FeatureCache(
                directory / "cache" / ("-".join(key) + ".sqlite"),
                capacity=1 if mode == "eviction" else CONFIG["capacity"],
            )
            creation = cache.create_ns
            if mode != "eviction":
                public_indices = sorted(
                    {
                        i
                        for case in data["cases"]
                        if (case["arm"], case["class"]) == (w["arm"], w["class"])
                        for i in case["updates"] + [g[0] for g in case["guards"]]
                    }
                )
                progress("cache_population_before", 0, len(public_indices))
                for position, i in enumerate(public_indices):
                    cache.get({k: data["sources"][i][k] for k in legacy.c.PUBLIC_KEYS})
                    progress("cache_population", position + 1, len(public_indices) - position - 1)
                progress("cache_population_after", len(public_indices))
            population.append(
                dict(
                    mode=mode,
                    condition=w["arm"],
                    transaction_class=w["class"],
                    arm=arm,
                    elapsed_ns=time.perf_counter_ns() - began,
                    creation_ns=creation,
                    boundary="initial_population",
                    events=list(cache.events),
                )
            )
            caches[key] = cache
        cache = caches.get(key)
        if cached:
            cache.events.clear()
        began = time.perf_counter_ns()
        row = legacy.old.m.transaction(
            current,
            w,
            binding,
            arm,
            directory / "transactions" / (unit.replace("/", "-") + ".json"),
            feature_provider=cache.get if cached else None,
        )
        events = list(cache.events) if cached else []
        memory = cache.memory() if cached else {}
        restart_ns = 0
        if mode == "restart":
            restart_started = time.perf_counter_ns()
            snapshot = cache_snapshot(cache) if cached else []
            if cached:
                cache.close()
            request = directory / "restart" / (unit.replace("/", "-") + ".input.json")
            output = request.with_suffix(".output.json")
            payload = dict(
                cache=str(cache.path) if cached else None,
                snapshot=snapshot,
                checkpoint=row["checkpoint"],
                library=legacy.reference(Path(binding.__file__)),
                arm=arm,
                output=str(output),
            )
            legacy.atomic_json(request, payload)
            progress("restart_subprocess_before", len(restarts), 1)
            receipt = legacy.run_commands(
                ROOT,
                [
                    legacy.CommandSpec(
                        "real_process_restart",
                        (
                            str(ROOT / ".venv/bin/python"),
                            "-u",
                            str(ROOT / SCRIPT),
                            "--restart-input",
                            str(request),
                        ),
                        "restart",
                        60,
                    )
                ],
                log_dir=request.parent / request.stem,
                heartbeat_s=10,
            )[0]
            progress("restart_subprocess_after", len(restarts) + 1)
            if not receipt["passed"]:
                raise ValueError("restart_child_failed")
            checked = json.loads(output.read_text())
            restarts.append(dict(checked, unit=unit, parent_pid=os.getpid(), receipt=receipt))
            row["native_calls"] += checked["native_calls"]
            if cached:
                cache = legacy.c.FeatureCache(cache.path)
                caches[key] = cache
            restart_ns = time.perf_counter_ns() - restart_started
        elapsed = time.perf_counter_ns() - began
        row["components"]["real_process_restart_ns"] = restart_ns
        row["components"]["feature_service_wrapper_ns"] = (
            elapsed - row["transaction_ns"] - restart_ns
        )
        for component in ("hashing", "loading", "extraction", "storage", "invalidation"):
            amount = sum(event[component + "_ns"] for event in events)
            row["components"]["cache_" + component + "_ns"] = amount
            row["components"]["feature_gather_ns"] -= amount
        row.update(
            transaction_ns=elapsed,
            memory=memory,
            changed_source_indices=changed,
            content_change_numerator=len(changed),
            content_change_denominator=len(indices),
            complete_public_input_hash=legacy.canonical_hash(
                [{k: current["sources"][i][k] for k in legacy.c.PUBLIC_KEYS} for i in indices]
            ),
        )
        return row, events

    with pipeline():
        saved = core.transaction
        core.transaction = transaction
        try:
            result: Json = ORIGINAL["measure"](data, native, raw)
        finally:
            core.transaction = saved
            for key, cache in caches.items():
                began = time.perf_counter_ns()
                cache.close()
                population.append(
                    dict(
                        mode=key[0],
                        condition=key[1],
                        transaction_class=key[2],
                        arm=key[3],
                        elapsed_ns=time.perf_counter_ns() - began,
                        creation_ns=0,
                        events=[],
                        boundary="service_shutdown",
                    )
                )
    result.update(
        population_rows=population,
        process_restart_rows=restarts,
        identity_check_rows=identity_checks(raw / "controls"),
    )
    result["parity_rows"].extend(
        dict(unit="identity/" + r["operand"], passed=r["passed"])
        for r in result["identity_check_rows"]
    )
    result["complete_service_ratios"] = service_reduce(
        result["complete_workload_ratios"], population
    )
    result["break_even_requests"] = [
        dict(
            mode=r["mode"],
            condition=r["condition"],
            transaction_class=r["transaction_class"],
            arm=r["arm"],
            requests=math.ceil(
                r["population_cleanup_ns"]
                / ((r["numerator"] - r["denominator"]) / r["paired_repetitions"])
            )
            if r["numerator"] > r["denominator"]
            else None,
            status="conditional_local_timing"
            if r["numerator"] > r["denominator"]
            else "no_break_even",
        )
        for r in result["complete_service_ratios"]
    ]
    legacy.atomic_json(raw / "observations.json", result)
    return result


def service_reduce(ratios: list[Json], population: list[Json]) -> list[Json]:
    """Include population and cleanup in paired full-service intervals."""
    result = copy.deepcopy(ratios)
    for cell in result:
        costs = sum(
            p["elapsed_ns"]
            for p in population
            if (p["mode"], p["condition"], p["transaction_class"], p["arm"])
            == (cell["mode"], cell["condition"], cell["transaction_class"], cell["arm"])
        )
        a = legacy.np.asarray([p["numerator"] for p in cell["paired_ratios"]], dtype=float)
        b = legacy.np.asarray([p["denominator"] for p in cell["paired_ratios"]], dtype=float)
        draws = legacy.np.random.default_rng(CONFIG["seed"]).integers(
            0, len(a), (CONFIG["draws"], len(a))
        )
        sampled = a[draws].sum(axis=1) / (b[draws].sum(axis=1) + costs)
        cell.update(
            population_cleanup_ns=costs,
            complete_service_denominator=cell["denominator"] + costs,
            complete_service_ratio=cell["numerator"] / (cell["denominator"] + costs),
            ratio=cell["numerator"] / (cell["denominator"] + costs),
            lower_95=float(legacy.np.quantile(sampled, 0.025)),
            upper_95=float(legacy.np.quantile(sampled, 0.975)),
            paired_ratios=[
                dict(
                    p,
                    denominator=p["denominator"] + costs / len(a),
                    ratio=p["numerator"] / (p["denominator"] + costs / len(a)),
                )
                for p in cell["paired_ratios"]
            ],
        )
    return result


@contextmanager
def pipeline() -> Iterator[None]:
    """Scope shared producer bindings to this process and restore all prior values."""
    bindings = dict(
        NAME=NAME,
        TASK=TASK,
        SCRIPT=SCRIPT,
        TEST=TEST,
        OWNED=OWNED,
        MODES=MODES,
        CONFIG=CONFIG,
        schedule=schedule,
        base=base,
        measure=measure,
        replay=replay,
        validation_plan=validation_plan,
        publish=publish,
        progress=progress,
        run_commands=run_commands,
        speed_gate=speed_gate,
    )
    saved = {name: getattr(core, name) for name in bindings}
    try:
        for name, value in bindings.items():
            setattr(core, name, value)
        with core.pipeline():
            yield
    finally:
        for name, value in saved.items():
            setattr(core, name, value)


def replay(path: Path) -> Json:
    """Independently reconstruct paired costs and check restart receipts."""
    with pipeline():
        result: Json = ORIGINAL["replay"](path)
    value = json.loads(path.read_text())
    if value["rows"]:
        raw = json.loads((Path(value["raw_directory"]) / "observations.json").read_text())
        for field in (
            "population_rows",
            "process_restart_rows",
            "identity_check_rows",
            "complete_service_ratios",
        ):
            if value[field] != raw[field]:
                raise ValueError("lifecycle_evidence_drift")
        for cell in value["complete_service_ratios"]:
            cost = sum(
                p["elapsed_ns"]
                for p in value["population_rows"]
                if (p["mode"], p["condition"], p["transaction_class"], p["arm"])
                == (cell["mode"], cell["condition"], cell["transaction_class"], cell["arm"])
            )
            if cell["complete_service_denominator"] != cell["denominator"] + cost:
                raise ValueError("population_reduction_drift")
        if value["complete_service_ratios"] != service_reduce(
            value["complete_workload_ratios"], value["population_rows"]
        ):
            raise ValueError("service_interval_drift")
        errors = lifecycle_errors(value)
        if errors:
            raise ValueError(errors[0])
    return result


def lifecycle_errors(value: Json) -> list[str]:
    """Missing controls and missing restarted units must fail qualification."""
    errors = []
    controls = value["identity_check_rows"]
    if (
        len(controls) != 5
        or {r["operand"] for r in controls}
        != {"source_bytes", "answer_bytes", "config", "schema", "corruption"}
        or not all(r["passed"] for r in controls)
    ):
        errors.append("identity_control_drift")
    expected = {
        r["unit"]: r for r in value["rows"] if r["mode"] == "restart" and r["status"] != "censored"
    }
    observed = value["process_restart_rows"]
    if len(observed) != len(expected) or {r["unit"] for r in observed} != set(expected):
        errors.append("restart_receipt_drift")
    else:
        for row in observed:
            transaction = expected[row["unit"]]
            if not (
                row["passed"]
                and row["parent_pid"] != row["child_pid"]
                and row["receipt"]["passed"]
                and row["receipt"]["exit_code"] == 0
                and row["checkpoint"] == transaction["checkpoint"]
                and (transaction["path_arm"] != "native" or row["native_calls"] > 0)
            ):
                errors.append("restart_receipt_drift")
    return errors


def validation_plan(scratch: Path) -> list[Any]:
    """Freeze the current owned commands before any timing starts."""
    with pipeline():
        result: list[Any] = ORIGINAL["validation_plan"](scratch)
    return result


def run_commands(root: Path, commands: Any, **kwargs: Any) -> list[Json]:
    """Bound measurement and save exact argv, exits, durations and log hashes."""
    commands = [replace(c, timeout_s=1860) if c.scope == "measurement" else c for c in commands]
    if commands and commands[0].scope == "measurement":
        raw = Path(commands[0].argv[commands[0].argv.index("--raw") + 1])
        methods = json.loads((raw / "methods.json").read_text())
        methods["measurement_command"]["timeout_s"] = 1860
        methods["dependency_hashes"].update(
            {
                p: legacy.sha256_file(ROOT / p)
                for p in (
                    "python/carnot/experiment_8066_v698_content_addressed_feature_service.py",
                    "python/carnot/experiment_8078_v699_feature_cache_core.py",
                    "python/carnot/verify/content_addressed_features_8066.py",
                    "python/carnot/experiment_8072_v699_sealed_methods.py",
                )
            }
        )
        legacy.atomic_json(raw / "methods.json", methods)
        progress("measurement_manifest_frozen")
    result: list[Json] = core.ORIGINAL["run_commands"](root, commands, **kwargs)
    return result


def speed_gate(ratios: list[Json]) -> bool:
    """Lifecycle timings do not establish the separate core speed acceptance gate."""
    return False


def publish(output: Path, value: Json, validator: Any) -> Json:
    """Complete paired cells and current owned checks alone can earn readiness."""
    ready = value["feature_service_ready_score"]
    complete = value["complete_pair_count"] == 270 and len(value["complete_workload_ratios"]) == 18
    complete = complete and all(
        r["paired_repetitions"] == 30 for r in value["complete_workload_ratios"]
    )
    complete = (
        complete and len(value["process_restart_rows"]) == 420 and not lifecycle_errors(value)
    )
    complete = (
        complete and value["required_checks_passed"] and value["flagged_adversarial"] is False
    )
    if ready and not complete:
        value.update(
            feature_service_ready_score=0,
            service_speedup_score=0,
            verdict_class="disqualified",
            honest_verdict="complete_disqualified_lifecycle_matrix",
        )
    value["cache_lifecycle_ready_score"] = value["feature_service_ready_score"]
    value["config"] = copy.deepcopy(CONFIG)
    health = HEALTH_PATH
    first = FIRST_PATH
    if value.get("raw_directory"):
        raw = Path(value["raw_directory"])
        raw.mkdir(parents=True, exist_ok=True)
        if health.is_file():
            receipt = json.loads(health.read_text())
            destination = raw / "repository_health_8079.log"
            destination.write_bytes(Path(receipt["log_path"]).read_bytes())
            value["current_repository_health"] = dict(
                receipt, log_path=str(destination), log_sha256=legacy.sha256_file(destination)
            )
            value["raw_shard_hashes"].append(legacy.reference(destination))
        if first.is_file():
            destination = raw / "tests_first_8079.log"
            destination.write_bytes(first.read_bytes())
            value["raw_shard_hashes"].append(legacy.reference(destination))
    for field in value:
        value["field_principles"].setdefault(
            field,
            "Bind this field to complete lifecycle evidence; prevent warm-path timing, repeated sources or unknown acquisition costs from becoming a general service claim.",
        )
    result: Json = core.ORIGINAL["publish_primary"](output, value, validator)
    return result


def main(argv: list[str] | None = None) -> int:
    """Run the shared terminal publication path or the private restart reader."""
    import sys

    arguments = list(sys.argv[1:] if argv is None else argv)
    if arguments and arguments[0] == "--restart-input":
        progress("restart_child_before")
        payload = json.loads(Path(arguments[1]).read_text())
        legacy.atomic_json(Path(payload["output"]), restart_check(payload))
        progress("restart_child_after", 1)
        return 0
    with pipeline():
        result: int = legacy.main(arguments)
    return result
