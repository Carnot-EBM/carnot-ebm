"""REQ-REPORT-8078: partition complete feature costs without losing comparisons."""

from __future__ import annotations

from contextlib import contextmanager
import copy
from dataclasses import replace
import json
import math
from pathlib import Path
import time
from typing import Any, Iterator

from carnot import experiment_8066_v698_content_addressed_feature_service as legacy
from carnot import experiment_8072_v699_sealed_methods as sealed
from carnot.reporting.primary_publication import read_bound_sidecar

Json = dict[str, Any]
ROOT = legacy.ROOT
NAME = "experiment_8078_v699_feature_cache_core"
TASK = "exp8078-feature-cache-core"
SCRIPT = f"scripts/experiments/{NAME}.py"
TEST = "tests/python/test_feature_cache_core_8078.py"
OWNED = [f"python/carnot/{NAME}.py", SCRIPT]
MODES = ("cold", "warm", "all_miss")
CONFIG: Json = dict(legacy.CONFIG, seed=6998072, budget_s=1800)
ORIGINAL = {
    name: getattr(legacy, name)
    for name in (
        "base",
        "load_inputs",
        "reduce_rows",
        "validation_plan",
        "replay",
        "publish_primary",
        "run_commands",
    )
}


def load_inputs(root: Path, raw: Path) -> tuple[Json, list[Json]]:
    """Authenticate the sealed order independently of the old censored timings."""
    data, failures = ORIGINAL["load_inputs"](root, raw)
    path = root / "results/experiment_8072_v699_sealed_methods.json"
    try:
        value = json.loads(path.read_text())
        side = Path(value["terminal_validation_sidecar_path"])
        terminal = json.loads(side.read_text())
        binding = terminal["publication"]
        report = read_bound_sidecar(path, Path(binding["sidecar_path"]))
        seal_ref = next(r for r in value["raw_shard_hashes"] if Path(r["path"]).name == "seal.json")
        frozen = json.loads(legacy.checked(seal_ref).read_text())
        for field, expected, observed in (
            ("service_protocol_ready_score", 1, value.get("service_protocol_ready_score")),
            ("required_checks_passed", True, value.get("required_checks_passed")),
            ("flagged_adversarial", False, value.get("flagged_adversarial")),
            ("report.passed", True, report["report"]["passed"]),
            ("measurement_exit", 0, terminal["measurement_exit_receipt"]["exit_code"]),
        ):
            if expected != observed:
                failures.append(legacy.failure(path, field, expected, observed))
        data["service_matrix"] = value["service_matrix"]
        if data["service_matrix"] != frozen["service_matrix"]:
            raise ValueError("sealed_service_matrix_drift")
        schedule(data)
        data["references"].append(seal_ref)
        data["references"].extend(
            legacy.reference(p) for p in (path, side, Path(binding["sidecar_path"]))
        )
        previous = root / "results/experiment_8066_v698_content_addressed_feature_service.json"
        previous_value = json.loads(previous.read_text())
        previous_side = Path(previous_value["terminal_validation_sidecar_path"])
        previous_binding = json.loads(previous_side.read_text())["publication"]
        read_bound_sidecar(previous, Path(previous_binding["sidecar_path"]))
        data["references"].extend(legacy.reference(p) for p in (previous, previous_side))
    except (OSError, ValueError, KeyError, StopIteration) as error:
        failures.append(
            legacy.failure(path, "service_matrix", "authenticated V699 frozen matrix", str(error))
        )
    legacy.atomic_json(raw / "inputs.json", data)
    return data, failures


def schedule(data: Json) -> list[Json]:
    """Reject altered statistical budgets instead of silently reducing repetitions."""
    matrix = data["service_matrix"]
    if matrix != sealed.service_matrix():
        raise ValueError("service_matrix")
    return [r for r in matrix["rows"] if r["partition"] == "core"]


def speed_gate(ratios: list[Json]) -> bool:
    """Every cold and miss interval must support the same claimed warm gain."""
    return (
        bool(ratios)
        and {r["mode"] for r in ratios} == set(MODES)
        and all(
            r["lower_95"] > 1.2 if r["mode"] == "warm" else r["lower_95"] >= 1 / 1.05
            for r in ratios
        )
    )


def base(failures: list[Json]) -> Json:
    """Keep mechanism readiness separate from speed and independent scientific credit."""
    value: Json = ORIGINAL["base"](failures)
    value.update(
        experiment_id=8078,
        task_id=TASK,
        milestone="2026.10.699",
        schema="carnot.v699.feature_cache_core.v1",
        cache_core_ready_score=0,
        complete_pair_count=0,
        planned_cell_counts=[],
        censored_cell_counts=[],
        native_call_receipts=[],
        population_rows=[],
        all_miss_identity_rows=[],
        trained_head_specs=[
            dict(
                parameter_count=110,
                origin="qualified historical Exp8053 small numerical head",
                trained_currently=False,
            )
        ],
    )
    value["sample_size_budget"].update(budget_s=1800)
    value["acquisition_cost_status"] = dict(
        model_acquisition="unmeasured",
        upstream_qwen_inference="unmeasured historical scalar",
        external_feedback="unmeasured historical labels",
    )
    return value


def quartet_parity(pair: list[Json], w: Json) -> list[Json]:
    """Compare actual persisted actions and state rather than assuming a cache hit works."""
    first = json.loads(legacy.checked(pair[0]["checkpoint"]).read_text())
    rows = []
    for row in pair:
        state = json.loads(legacy.checked(row["checkpoint"]).read_text())
        compared = legacy.old.m.comparison(first["acceptance"], state["acceptance"])
        exact = (
            state["replay_pool"] == w["released"]
            and state["guards"] == w["guards"]
            and row["restart_equal"]
            and row["restart_decisions_equal"]
        )
        rows.append(
            dict(
                unit=row["unit"],
                **{k: v for k, v in compared.items() if k != "passed"},
                features_bit_identical=True,
                discrete_state_equal=exact,
                passed=compared["passed"] and exact,
            )
        )
    return rows


def transaction(
    data: Json, w: Json, native: Any, choice: str, mode: str, cache: Any, raw: Path, unit: str
) -> tuple[Json, list[Json]]:
    """Measure cache lifetime at the transaction boundary, including cold cleanup."""
    arm, cache_mode = choice.split("_")
    cached = cache_mode == "cached"
    request_index = 0

    def miss_provider(public: Json) -> Json:
        # Request position is fixed by the original update and guard lists.
        # Complete bytes stay identical across all four transaction arms.
        nonlocal request_index
        public = dict(public, family_id=public["family_id"] + f"/request/{request_index}")
        request_index += 1
        result: Json = (cache.get if cached else legacy.c.features.extract)(public)
        return result

    began = time.perf_counter_ns()
    creation = 0
    if cached and mode in ("cold", "all_miss"):
        cache = legacy.c.FeatureCache(raw / "cache" / (unit.replace("/", "-") + ".sqlite"))
        creation = time.perf_counter_ns() - began
    if cached:
        cache.events.clear()
    row = legacy.old.m.transaction(
        data,
        w,
        native,
        arm,
        raw / "transactions" / (unit.replace("/", "-") + ".json"),
        feature_provider=miss_provider if mode == "all_miss" else cache.get if cached else None,
    )
    events = list(cache.events) if cached else []
    memory = cache.memory() if cached else {}
    started = time.perf_counter_ns()
    if cached and mode in ("cold", "all_miss"):
        cache.close()
        cache.path.unlink()
    cleanup = time.perf_counter_ns() - started
    elapsed = time.perf_counter_ns() - began
    components = row["components"]
    components["cache_creation_ns"] = creation
    components["cache_cleanup_ns"] = cleanup
    components["feature_service_wrapper_ns"] = elapsed - row["transaction_ns"] - creation - cleanup
    for component in ("hashing", "loading", "extraction", "storage", "invalidation"):
        amount = sum(event[component + "_ns"] for event in events)
        components["cache_" + component + "_ns"] = amount
        components["feature_gather_ns"] -= amount
    row.update(transaction_ns=elapsed, memory=memory)
    return row, events


def reduce_rows(rows: list[Json]) -> list[Json]:
    """Keep each paired ratio alongside conditional bootstrap intervals."""
    with pipeline():
        result: list[Json] = ORIGINAL["reduce_rows"](rows)
    for cell in result:
        parts = [
            r
            for r in rows
            if r["status"] == "completed"
            and (r["mode"], r["condition"], r["transaction_class"], r["path_arm"])
            == (cell["mode"], cell["condition"], cell["transaction_class"], cell["arm"])
        ]
        by_rep: dict[int, Json] = {}
        for row in parts:
            by_rep.setdefault(row["repetition"], {})[row["cached"]] = row
        cell["paired_ratios"] = [
            dict(
                repetition=rep,
                numerator=p[False]["transaction_ns"],
                denominator=p[True]["transaction_ns"],
                ratio=p[False]["transaction_ns"] / p[True]["transaction_ns"],
            )
            for rep, p in sorted(by_rep.items())
        ]
        cell["resamples"] = CONFIG["draws"]
    return result


def measure(data: Json, native: Any, raw: Path) -> Json:
    """Interleave sealed quartets and retain every slot when the clock expires."""
    plan = schedule(data)
    raw.mkdir(parents=True, exist_ok=True)
    rows: list[Json] = []
    parity: list[Json] = []
    hits: list[Json] = []
    population: list[Json] = []
    identities: list[Json] = []
    caches: dict[tuple[str, str, str, str], Any] = {}
    pools = {
        (r["condition"], r["transaction_class"]): [
            w
            for w in data["cases"]
            if (w["arm"], w["class"]) == (r["condition"], r["transaction_class"])
        ]
        for r in plan
    }
    workloads: dict[str, Json] = {}
    for row in plan:
        pool = pools[(row["condition"], row["transaction_class"])]
        workloads[row["unit"]] = pool[(row["repetition"] + 5) % len(pool)]
    # Fix complete-record miss identities before any measured transaction begins.
    for start in range(0, len(plan), 4):
        first = plan[start]
        if first["mode"] == "all_miss":
            w = workloads[first["unit"]]
            for i in sorted(set(w["updates"] + [g[0] for g in w["guards"]])):
                source = data["sources"][i]
                identities.append(
                    dict(
                        quartet=start // 4,
                        index=i,
                        family_id=f"{source['family_id']}/8078/{start // 4}/{i}",
                        source_bytes_preserved=True,
                        source_bytes_hash=legacy.canonical_hash(source["source_bytes"]),
                        answer_bytes_hash=legacy.canonical_hash(source["answer_bytes"]),
                    )
                )
    deadline = time.monotonic() + CONFIG["budget_s"]
    legacy.progress("benchmark_before", 0, len(plan))
    for start in range(0, len(plan), 4):
        quartet = plan[start : start + 4]
        first = quartet[0]
        w = workloads[first["unit"]]
        mode, condition, klass = first["mode"], first["condition"], first["transaction_class"]
        current = data
        if mode == "all_miss":
            current = dict(data, sources=copy.deepcopy(data["sources"]))
            for identity in identities:
                if identity["quartet"] == start // 4:
                    current["sources"][identity["index"]]["family_id"] = identity["family_id"]
        censored = time.monotonic() >= deadline
        pair = []
        legacy.progress("benchmark_quartet_before", len(rows), len(plan) - len(rows))
        for planned in quartet:
            choice = planned["arm"]
            arm, cache_mode = choice.split("_")
            common = dict(
                planned,
                source=w["identity"],
                source_workload_hash=legacy.canonical_hash(w),
                path_arm=arm,
                cached=cache_mode == "cached",
                seed=w["seed"],
                numerator=1,
                denominator=1,
                status="censored" if censored else "excluded" if planned["warmup"] else "completed",
                exclusion_reason="measurement_budget"
                if censored
                else "warmup"
                if planned["warmup"]
                else None,
            )
            if censored:
                pair.append(common)
                continue
            key = (mode, condition, klass, arm)
            if cache_mode == "cached" and mode == "warm" and key not in caches:
                began = time.perf_counter_ns()
                cache = legacy.c.FeatureCache(raw / "cache" / ("-".join(key) + ".sqlite"))
                if mode == "warm":
                    indices = sorted(
                        {
                            i
                            for case in pools[(condition, klass)]
                            for i in case["updates"] + [g[0] for g in case["guards"]]
                        }
                    )
                    legacy.progress("cache_population_before", 0, len(indices))
                    for position, i in enumerate(indices):
                        source = data["sources"][i]
                        cache.get({k: source[k] for k in legacy.c.PUBLIC_KEYS})
                        legacy.progress(
                            "cache_population", position + 1, len(indices) - position - 1
                        )
                    legacy.progress("cache_population_after", len(indices))
                population.append(
                    dict(
                        mode=mode,
                        condition=condition,
                        transaction_class=klass,
                        arm=arm,
                        elapsed_ns=time.perf_counter_ns() - began,
                        creation_ns=0,
                        events=list(cache.events),
                    )
                )
                caches[key] = cache
            legacy.progress(
                "benchmark_transaction_before",
                len(rows) + len(pair),
                len(plan) - len(rows) - len(pair),
            )
            measured, events = transaction(
                current, w, native, choice, mode, caches.get(key), raw, planned["unit"]
            )
            measured.update(common)
            pair.append(measured)
            hits.extend(
                dict(event, unit=planned["unit"], mode=mode, arm=choice) for event in events
            )
            legacy.progress(
                "benchmark_transaction_after",
                len(rows) + len(pair),
                len(plan) - len(rows) - len(pair),
            )
        checks = [] if censored else quartet_parity(pair, w)
        legacy.atomic_json(
            raw / "pairs" / f"{start // 4:04d}.json", dict(rows=pair, parity_rows=checks)
        )
        rows.extend(pair)
        parity.extend(checks)
        legacy.progress("benchmark_quartet_after", len(rows), len(plan) - len(rows))
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
    ratios = reduce_rows(rows)
    breaks = []
    for warm in (r for r in ratios if r["mode"] == "warm"):
        saving = (warm["numerator"] - warm["denominator"]) / warm["paired_repetitions"]
        population_ns = sum(
            p["elapsed_ns"]
            for p in population
            if (p["mode"], p["condition"], p["transaction_class"], p["arm"])
            == ("warm", warm["condition"], warm["transaction_class"], warm["arm"])
        )
        breaks.append(
            dict(
                condition=warm["condition"],
                transaction_class=warm["transaction_class"],
                arm=warm["arm"],
                population_ns=population_ns,
                warm_saving_ns=saving,
                requests=math.ceil(population_ns / saving) if saving > 0 else None,
                amortized_population_ns_per_measured_request=population_ns
                / warm["paired_repetitions"],
            )
        )
    cells = sorted({(r["mode"], r["condition"], r["transaction_class"]) for r in plan})
    counts = [
        dict(
            mode=m,
            condition=c,
            transaction_class=k,
            intended_count=sum(
                not r["warmup"] and (r["mode"], r["condition"], r["transaction_class"]) == (m, c, k)
                for r in plan
            ),
            censored_count=sum(
                r["status"] == "censored"
                and not r["warmup"]
                and (r["mode"], r["condition"], r["transaction_class"]) == (m, c, k)
                for r in rows
            ),
        )
        for m, c, k in cells
    ]
    result = dict(
        rows=rows,
        parity_rows=parity,
        cache_hit_miss_rows=hits,
        population_rows=population,
        memory_bytes=[dict(unit=r["unit"], **r["memory"]) for r in rows if r.get("memory")],
        all_miss_identity_rows=identities,
        complete_workload_ratios=ratios,
        break_even_requests=breaks,
        invalidation_rows=[r for r in hits if r["evicted"] or r["status"] == "corrupt_miss"],
        component_timing_rows=[
            dict(unit=r["unit"], timings=r["components"]) for r in rows if "components" in r
        ],
        intended_count=sum(not r["warmup"] for r in plan),
        eligible_count=sum(not r["warmup"] for r in plan),
        completed_count=sum(r["status"] == "completed" for r in rows),
        censored_count=sum(r["status"] == "censored" and not r["warmup"] for r in rows),
        excluded_count=sum(r["status"] == "excluded" for r in rows),
        failed_count=0,
        complete_pair_count=sum(r["status"] == "completed" for r in rows) // 4,
        planned_cell_counts=counts,
        censored_cell_counts=counts,
        native_call_receipts=[
            dict(unit=r["unit"], calls=r["native_calls"], checkpoint=r["checkpoint"])
            for r in rows
            if r.get("native_calls", 0)
        ],
        feature_service_ready_score=0,
        service_speedup_score=0,
    )
    legacy.atomic_json(raw / "observations.json", result)
    legacy.progress("benchmark_after", len(rows), 0)
    return result


@contextmanager
def pipeline() -> Iterator[None]:
    """Reuse the qualified lifecycle while restoring the historical producer after use.

    Each CLI owns one process. Scoped bindings let this partition share its
    validated publication lifecycle without changing any historical source bytes.
    """
    bindings = dict(
        NAME=NAME,
        TASK=TASK,
        SCRIPT=SCRIPT,
        TEST=TEST,
        OWNED=OWNED,
        MODES=MODES,
        CONFIG=CONFIG,
        base=base,
        load_inputs=load_inputs,
        measure=measure,
        reduce_rows=reduce_rows,
        speed_gate=speed_gate,
        replay=replay,
        validation_plan=validation_plan,
        publish_primary=publish,
        run_commands=run_commands,
        progress=progress,
    )
    saved = {name: getattr(legacy, name) for name in bindings}
    try:
        for name, value in bindings.items():
            setattr(legacy, name, value)
        yield
    finally:
        for name, value in saved.items():
            setattr(legacy, name, value)


START = time.monotonic()


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Report real completed work while bounded children run."""
    print(
        f"[exp8078] phase={phase} elapsed_s={time.monotonic() - START:.3f} completed={completed} pending={pending}",
        flush=True,
    )


def replay(path: Path) -> Json:
    """Reopen durable cache and state in the terminal reader's fresh process."""
    with pipeline():
        result: Json = ORIGINAL["replay"](path)
    value = json.loads(path.read_text())
    for cell in value["complete_workload_ratios"]:
        primitive = [
            r
            for r in value["rows"]
            if r["status"] == "completed"
            and (r["mode"], r["condition"], r["transaction_class"], r["path_arm"])
            == (cell["mode"], cell["condition"], cell["transaction_class"], cell["arm"])
        ]
        columns = {
            cached: sorted(
                [r for r in primitive if r["cached"] == cached], key=lambda r: r["repetition"]
            )
            for cached in (False, True)
        }
        a, b = (
            legacy.np.asarray([r["transaction_ns"] for r in columns[c]], dtype=float)
            for c in (False, True)
        )
        draws = legacy.np.random.default_rng(CONFIG["seed"]).integers(
            0, len(a), (CONFIG["draws"], len(a))
        )
        sampled = a[draws].sum(axis=1) / b[draws].sum(axis=1)
        if (
            cell["numerator"] != int(a.sum())
            or cell["denominator"] != int(b.sum())
            or cell["lower_95"] != float(legacy.np.quantile(sampled, 0.025))
            or cell["upper_95"] != float(legacy.np.quantile(sampled, 0.975))
        ):
            raise ValueError("independent_reduction_drift")
    for pair_path in sorted((Path(value["raw_directory"]) / "pairs").glob("*.json")):
        pair = json.loads(pair_path.read_text())
        if len(pair["rows"]) != 4 or any(r not in value["rows"] for r in pair["rows"]):
            raise ValueError("quartet_drift")
    for cache_path in sorted((Path(value["raw_directory"]) / "cache").glob("*.sqlite")):
        cache = legacy.c.FeatureCache(cache_path)
        try:
            for key, vector, digest in cache.db.execute("SELECT key,vector,digest FROM features"):
                if digest != legacy.canonical_hash(dict(key=key, values=json.loads(vector))):
                    raise ValueError("persisted_cache_drift")
        finally:
            cache.close()
    return result


def validation_plan(scratch: Path) -> list[Any]:
    """Freeze only owned checks; repository health stays a separate diagnostic."""
    with pipeline():
        checks: list[Any] = ORIGINAL["validation_plan"](scratch)
    checks.append(
        legacy.CommandSpec(
            "private_e2e015_016",
            (
                str(ROOT / ".venv/bin/python"),
                "-m",
                "pytest",
                "-n",
                "0",
                "-o",
                "addopts=",
                "--no-cov",
                "--basetemp=" + str(scratch / "e2e"),
                "tests/python/test_source_boundary_7852.py",
                "tests/python/test_experiment_7868_v683_intervention_protocol.py",
                "-q",
            ),
            "owned",
            180,
        )
    )
    return checks


def run_commands(root: Path, commands: Any, **kwargs: Any) -> list[Json]:
    """Give the measurement child its sealed budget plus a bounded exit margin."""
    commands = [replace(c, timeout_s=1860) if c.scope == "measurement" else c for c in commands]
    if commands and commands[0].scope == "measurement":
        command = commands[0]
        raw = Path(command.argv[command.argv.index("--raw") + 1])
        methods = json.loads((raw / "methods.json").read_text())
        methods["measurement_command"]["timeout_s"] = command.timeout_s
        methods["mutation"] = (
            "complete source/answer bytes preserved; unique all_miss family identities fixed before timing"
        )
        for name in (
            "python/carnot/experiment_8066_v698_content_addressed_feature_service.py",
            "python/carnot/verify/content_addressed_features_8066.py",
            "python/carnot/experiment_8072_v699_sealed_methods.py",
        ):
            methods["dependency_hashes"][name] = legacy.sha256_file(ROOT / name)
        legacy.atomic_json(raw / "methods.json", methods)
        progress("measurement_manifest_frozen")
    receipts: list[Json] = ORIGINAL["run_commands"](root, commands, **kwargs)
    return receipts


def publish(output: Path, value: Json, validator: Any) -> Json:
    """Export core readiness only for the complete authenticated paired matrix."""
    value["cache_core_ready_score"] = value["feature_service_ready_score"]
    if value["cache_core_ready_score"] and (
        value["complete_pair_count"] != 270
        or len(value["complete_workload_ratios"]) != 18
        or any(r["paired_repetitions"] != 30 for r in value["complete_workload_ratios"])
    ):
        value.update(
            cache_core_ready_score=0,
            feature_service_ready_score=0,
            service_speedup_score=0,
            verdict_class="disqualified",
            honest_verdict="complete_disqualified_core_matrix",
        )
    value["config"] = copy.deepcopy(CONFIG)
    health = Path("/tmp/exp8078-repository-health.log")
    first = Path("/tmp/exp8078-tests-first.log")
    if value.get("raw_directory"):
        raw = Path(value["raw_directory"])
        for source in (health, first):
            if source.is_file():
                destination = raw / source.name
                destination.write_bytes(source.read_bytes())
                value["raw_shard_hashes"].append(legacy.reference(destination))
        if health.is_file():
            value["current_repository_health"] = dict(
                argv=[str(ROOT / ".venv/bin/pytest"), "tests/python", "-q"],
                exit_code=124,
                timeout_s=300,
                scope="repository_health",
                passed=False,
                status="bounded run timed out; repository-wide failures observed",
                log=legacy.reference(raw / health.name),
            )
    for field in value:
        value["field_principles"].setdefault(
            field,
            "Bind this field to complete current transactions; prevent partial timing and repeated rows from becoming readiness or independent scientific credit.",
        )
    result: Json = ORIGINAL["publish_primary"](output, value, validator)
    return result


def main(argv: list[str] | None = None) -> int:
    """Run the shared bounded lifecycle with this partition's immutable contract."""
    with pipeline():
        result: int = legacy.main(argv)
    return result
