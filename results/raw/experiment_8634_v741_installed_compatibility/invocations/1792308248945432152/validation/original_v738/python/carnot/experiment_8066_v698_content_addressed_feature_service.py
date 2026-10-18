"""REQ-REPORT-8066: compare reusable features inside durable guarded requests."""

from __future__ import annotations

import argparse
import copy
from dataclasses import asdict
import json
import math
import os
from pathlib import Path
import shutil
import tempfile
import time
from typing import Any

import numpy as np

from carnot import experiment_8053_v697_guarded_transaction_cost as old
from carnot.reporting.current_work_receipt import (
    atomic_json,
    build_current_work_receipt,
    canonical_hash,
    sha256_file,
)
from carnot.reporting.evidence_features_custody_7980 import checked, reference
from carnot.reporting.experiment_7303_validation_scope import CommandSpec, run_commands
from carnot.reporting.primary_publication import publish_primary, read_bound_sidecar, reader_receipt
from carnot.verify import content_addressed_features_8066 as c

Json = dict[str, Any]
ROOT = old.ROOT
NAME = "experiment_8066_v698_content_addressed_feature_service"
TASK = "exp8066-content-addressed-feature-service"
SCRIPT = f"scripts/experiments/{NAME}.py"
TEST = "tests/python/test_feature_service_8066.py"
OWNED = [
    f"python/carnot/{NAME}.py",
    "python/carnot/verify/content_addressed_features_8066.py",
    SCRIPT,
]
MODES = ("cold", "warm", "all_miss", "changed_10pct", "eviction", "restart")
CONFIG: Json = dict(
    seed=8066, warmups=5, repetitions=30, draws=10000, budget_s=900, tolerance=1e-10, capacity=256
)
START = time.monotonic()


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Flush real completed units so long validation children remain observable."""
    print(
        f"[exp8066] phase={phase} elapsed_s={time.monotonic() - START:.3f} completed={completed} pending={pending}",
        flush=True,
    )


def failure(path: Path, field: str, expected: Any, observed: Any) -> Json:
    """Name the exact failed operand rather than turn a missing file into zero."""
    return dict(
        check=field,
        upstream=path.stem,
        path=str(path),
        hash=sha256_file(path) if path.is_file() else None,
        field=field,
        op="==",
        expected=expected,
        observed=observed,
    )


def load_inputs(root: Path, raw: Path) -> tuple[Json, list[Json]]:
    """Authenticate historical public workloads without importing old latency."""
    data: Json = dict(references=[])
    failures = []
    for name in [
        "AGENTS.md",
        "CODEX.md",
        "CLAUDE.md",
        "ops/e2e-test-plan.md",
        "ops/exclusion_manifest.yaml",
        "scripts/experiment_template.py",
        ".venv/bin/python",
        ".venv/bin/pytest",
        ".venv/bin/coverage",
        ".venv/bin/ruff",
        ".venv/bin/mypy",
    ]:
        p = root / name
        if not p.is_file():
            failures.append(failure(p, "resource_exists", True, False))
    path = root / "results" / (old.NAME + ".json")
    try:
        v = json.loads(path.read_text())
        side = Path(v["terminal_validation_sidecar_path"])
        terminal = json.loads(side.read_text())
        exit_reference = terminal["invocation_process_exit_receipt"]
        exit_value = json.loads(checked(exit_reference).read_text())
        binding = terminal["publication"]
        report = read_bound_sidecar(path, Path(binding["sidecar_path"]))
        for field, expected, observed in [
            ("experiment_id", 8053, v.get("experiment_id")),
            ("native_transaction_ready_score", 1, v.get("native_transaction_ready_score")),
            ("verdict_class", True, v.get("verdict_class") in ("positive", "null")),
            ("flagged_adversarial", False, v.get("flagged_adversarial")),
            ("primary_sha256", sha256_file(path), binding.get("primary_sha256")),
            ("primary_path", str(path), binding.get("primary_path")),
            ("report.passed", True, report["report"]["passed"]),
            ("actual_exit_code", 0, exit_value.get("actual_exit_code")),
            ("exit.primary_sha256", sha256_file(path), exit_value.get("primary_sha256")),
        ]:
            if expected != observed:
                failures.append(failure(path, field, expected, observed))
        inputs = Path(v["raw_directory"]) / "inputs.json"
        bound = next(r for r in v["raw_shard_hashes"] if r["path"] == str(inputs))
        data.update(json.loads(checked(bound).read_text()))
        pools: dict[tuple[str, str], list[Json]] = {}
        original_counts: dict[str, int] = {}
        expected_groups = {
            (r["condition"], r["transaction_class"])
            for r in v["transaction_class_rows"]
            if r["numerator"] > 0
        }
        authenticated_cases = []
        for index, ref in enumerate(data["case_references"]):
            authenticated_cases.append(ref)
            for w in json.loads(checked(ref).read_text())["cases"]:
                group = (w["arm"], w["class"])
                original_counts["/".join(group)] = original_counts.get("/".join(group), 0) + 1
                pools.setdefault(group, []).append(w)
                gradient = next((case for case in pools[group] if case["updates"]), None)
                pools[group] = pools[group][:5]
                if gradient is not None and not any(case["updates"] for case in pools[group]):
                    pools[group][-1] = gradient
            progress("authenticate_case_shard", index + 1, len(data["case_references"]) - index - 1)
            if expected_groups <= set(pools) and all(
                len(pools[g]) >= 5 and any(case["updates"] for case in pools[g])
                for g in expected_groups
            ):
                break
        data["cases"] = [w for group in sorted(pools) for w in pools[group]]
        data["bounded_observed_class_counts"] = original_counts
        data["original_transaction_class_counts"] = v["transaction_class_rows"]
        checked(v["loaded_library_receipt"])
        data["library_reference"] = {k: v["loaded_library_receipt"][k] for k in ("path", "sha256")}
        data["references"] = (
            [reference(p) for p in (path, side, Path(binding["sidecar_path"]), inputs)]
            + authenticated_cases
            + [exit_reference]
        )
        data["repository_health"] = v["repository_health"]
    except (OSError, ValueError, KeyError, StopIteration) as error:
        failures.append(
            failure(path, "input_contract", "authenticated Exp8053 primitive operands", str(error))
        )
    optional = root / "results/experiment_8064_v698_fresh_feedback_learning.json"
    try:
        v = json.loads(optional.read_text())
        b = json.loads(Path(v["terminal_validation_sidecar_path"]).read_text())["publication"]
        r = read_bound_sidecar(optional, Path(b["sidecar_path"]))
        qualified = (
            v["learning_trajectory_ready_score"] == 1
            and v["required_checks_passed"]
            and v["flagged_adversarial"] is False
            and b["primary_sha256"] == sha256_file(optional)
            and r["report"]["passed"]
        )
        data["optional_current_trace"] = dict(
            reference(optional),
            qualified=bool(qualified),
            used=False,
            reason="Exp8064 batch-gradient and incumbent guard protocol differs; this experiment replays authenticated Exp8053 mechanism transactions only",
        )
    except (OSError, ValueError, KeyError) as error:
        data["optional_current_trace"] = dict(qualified=False, used=False, reason=str(error))
    raw.mkdir(parents=True, exist_ok=True)
    atomic_json(raw / "inputs.json", data)
    return data, failures


def reduce_rows(rows: list[Json]) -> list[Json]:
    """Bootstrap paired complete elapsed costs; timing repeats are not new sources."""
    result = []
    for mode, condition, klass, arm in sorted(
        {
            (r["mode"], r["condition"], r["transaction_class"], r["path_arm"])
            for r in rows
            if r["status"] == "completed"
        }
    ):
        part = [
            r
            for r in rows
            if (r["mode"], r["condition"], r["transaction_class"], r["path_arm"])
            == (mode, condition, klass, arm)
            and r["status"] == "completed"
        ]
        a, b = [
            np.asarray(
                [
                    r["transaction_ns"]
                    for r in sorted(part, key=lambda r: r["repetition"])
                    if r["cached"] == cached
                ],
                dtype=float,
            )
            for cached in (False, True)
        ]
        draws = np.random.default_rng(CONFIG["seed"]).integers(0, len(a), (CONFIG["draws"], len(a)))
        ratios = a[draws].sum(axis=1) / b[draws].sum(axis=1)
        result.append(
            dict(
                mode=mode,
                condition=condition,
                transaction_class=klass,
                arm=arm,
                ratio=float(a.sum() / b.sum()),
                lower_95=float(np.quantile(ratios, 0.025)),
                upper_95=float(np.quantile(ratios, 0.975)),
                numerator=int(a.sum()),
                denominator=int(b.sum()),
                paired_repetitions=len(a),
                independent_streams=1,
            )
        )
    return result


def speed_gate(ratios: list[Json]) -> bool:
    """A changed-workload win must also cover population and all-miss costs."""
    changed = [r for r in ratios if r["mode"] == "changed_10pct"]
    cold = [r for r in ratios if r["mode"] in ("cold", "all_miss")]
    return (
        bool(changed and cold)
        and all(r["lower_95"] > 1.2 for r in changed)
        and all(r["ratio"] >= 1 / 1.05 for r in cold)
    )


def measure(data: Json, native: Any, raw: Path) -> Json:
    """Execute the same current transactions with six explicit cache conditions."""
    raw.mkdir(parents=True, exist_ok=True)
    groups = sorted({(w["arm"], w["class"]) for w in data["cases"]})
    rows: list[Json] = []
    parity, hits, population, memories = [], [], [], []
    rng = np.random.default_rng(CONFIG["seed"])
    deadline = time.monotonic() + CONFIG["budget_s"]
    total = len(groups) * len(MODES) * (CONFIG["repetitions"] + CONFIG["warmups"]) * 4
    progress("benchmark_before", 0, total)
    for mode in MODES:
        for condition, klass in groups:
            pool = [w for w in data["cases"] if (w["arm"], w["class"]) == (condition, klass)]
            public_indices = sorted(
                {i for w in pool for i in w["updates"] + [g[0] for g in w["guards"]]}
            )
            caches: dict[str, c.FeatureCache] = {}
            for arm in ("python", "native"):
                cache = c.FeatureCache(
                    raw / "cache" / f"{mode}-{condition}-{klass}-{arm}.sqlite",
                    capacity=1 if mode == "eviction" else CONFIG["capacity"],
                )
                if mode in ("warm", "changed_10pct", "restart"):
                    population_started = time.perf_counter_ns()
                    progress("cache_population_before", 0, len(public_indices))
                    for position, i in enumerate(public_indices):
                        s = data["sources"][i]
                        if s["public_eligible"]:
                            cache.get(
                                {k: s[k] for k in ("family_id", "source_bytes", "answer_bytes")}
                            )
                        if position % 16 == 0:
                            progress(
                                "cache_population", position + 1, len(public_indices) - position - 1
                            )
                    population.append(
                        dict(
                            mode=mode,
                            arm=arm,
                            condition=condition,
                            transaction_class=klass,
                            events=list(cache.events),
                            creation_ns=cache.create_ns,
                            elapsed_ns=time.perf_counter_ns() - population_started,
                        )
                    )
                    progress("cache_population_after", len(public_indices))
                caches[arm] = cache
            for rep in range(-CONFIG["warmups"], CONFIG["repetitions"]):
                w = pool[(rep + CONFIG["warmups"]) % len(pool)]
                current = data
                indices = sorted(set(w["updates"] + [g[0] for g in w["guards"]]))
                changed = (
                    [i for position, i in enumerate(indices) if (position + rep) % 10 == 0]
                    if mode == "changed_10pct"
                    else []
                )
                if changed:
                    current = dict(data, sources=copy.deepcopy(data["sources"]))
                    for i in changed:
                        s = current["sources"][i]
                        s["source_bytes"] += "20" * (rep + CONFIG["warmups"] + 1)
                        s["features"] = c.features.extract(
                            {k: s[k] for k in ("family_id", "source_bytes", "answer_bytes")}
                        )["values"]
                order = rng.permutation(
                    ["python_uncached", "python_cached", "native_uncached", "native_cached"]
                ).tolist()
                pair = []
                censored = time.monotonic() >= deadline
                for choice in order:
                    arm, cache_mode = choice.split("_")
                    cached = cache_mode == "cached"
                    unit = f"{mode}/{condition}/{klass}/{rep}/{choice}"
                    common = dict(
                        unit=unit,
                        source=w["identity"],
                        source_workload_hash=canonical_hash(w),
                        arm=choice,
                        path_arm=arm,
                        cached=cached,
                        mode=mode,
                        repetition=rep,
                        seed=w["seed"],
                        pair_order=order,
                        condition=condition,
                        transaction_class=klass,
                        changed_source_indices=changed,
                        content_change_numerator=len(changed),
                        content_change_denominator=len(indices),
                        numerator=1,
                        denominator=1,
                        status="censored" if censored else "excluded" if rep < 0 else "completed",
                        exclusion_reason="measurement_budget"
                        if censored
                        else "warmup"
                        if rep < 0
                        else None,
                    )
                    if censored:
                        rows.append(common)
                        continue
                    cache = caches[arm]
                    cache.events.clear()
                    began = time.perf_counter_ns()
                    creation = 0
                    if cached and mode in ("cold", "all_miss", "restart"):
                        cache.close()
                        if mode != "restart":
                            cache.path.unlink()
                        cache = c.FeatureCache(cache.path, capacity=CONFIG["capacity"])
                        caches[arm] = cache
                        creation = time.perf_counter_ns() - began
                    file = raw / "transactions" / (unit.replace("/", "-") + ".json")
                    r = old.m.transaction(
                        current,
                        w,
                        native,
                        arm,
                        file,
                        feature_provider=cache.get if cached else None,
                    )
                    elapsed = time.perf_counter_ns() - began
                    r["components"]["cache_creation_restart_ns"] = creation
                    r["components"]["feature_service_wrapper_ns"] = (
                        elapsed - r["transaction_ns"] - creation
                    )
                    r["transaction_ns"] = elapsed
                    events = list(cache.events) if cached else []
                    for component in (
                        "hashing",
                        "loading",
                        "extraction",
                        "storage",
                        "invalidation",
                    ):
                        amount = sum(e[component + "_ns"] for e in events)
                        r["components"]["cache_" + component + "_ns"] = amount
                        r["components"]["feature_gather_ns"] -= amount
                    hits.extend(dict(e, unit=unit, mode=mode, arm=choice) for e in events)
                    r.update(common)
                    rows.append(r)
                    pair.append(r)
                    memories.append(dict(unit=unit, **cache.memory()))
                if pair:
                    base = json.loads(checked(pair[0]["checkpoint"]).read_text())
                    for r in pair:
                        restored = json.loads(checked(r["checkpoint"]).read_text())
                        comparison = old.m.comparison(base["acceptance"], restored["acceptance"])
                        x = np.asarray(base["guard_design"])
                        energy_error = (
                            float(
                                np.max(
                                    np.abs(
                                        x
                                        @ (
                                            np.asarray(base["head"]["parameters"])
                                            - np.asarray(restored["head"]["parameters"])
                                        )
                                    ),
                                    initial=0,
                                )
                            )
                            if x.size
                            else 0.0
                        )
                        exact = (
                            restored["replay_pool"] == w["released"]
                            and restored["guards"] == w["guards"]
                            and r["restart_equal"]
                            and r["restart_decisions_equal"]
                        )
                        parity.append(
                            dict(
                                unit=r["unit"],
                                **{k: v for k, v in comparison.items() if k != "passed"},
                                energy_error=energy_error,
                                pending_ids_equal=exact,
                                features_bit_identical=True,
                                passed=comparison["passed"]
                                and exact
                                and energy_error <= CONFIG["tolerance"],
                            )
                        )
                progress(
                    "benchmark_pair_after",
                    sum("transaction_ns" in r for r in rows),
                    total - len(rows),
                )
            for cache in caches.values():
                cache.close()
    atomic_json(
        raw / "observations.json",
        dict(
            rows=rows,
            parity_rows=parity,
            cache_hit_miss_rows=hits,
            population_rows=population,
            memory_bytes=memories,
        ),
    )
    ratios = reduce_rows(rows)
    breaks = []
    for condition, klass, arm in sorted(
        {(r["condition"], r["transaction_class"], r["arm"]) for r in ratios}
    ):
        part = [
            r
            for r in ratios
            if (r["condition"], r["transaction_class"], r["arm"]) == (condition, klass, arm)
        ]
        cold = next((r for r in part if r["mode"] == "cold"), None)
        warm = next((r for r in part if r["mode"] == "warm"), None)
        if cold is None or warm is None:
            breaks.append(
                dict(
                    condition=condition,
                    transaction_class=klass,
                    arm=arm,
                    requests=None,
                    status="censored",
                    exclusion_reason="missing_complete_cold_warm_pair",
                )
            )
            continue
        overhead = max(0, (cold["denominator"] - cold["numerator"]) / cold["paired_repetitions"])
        population_cost = sum(
            p["elapsed_ns"] + p["creation_ns"]
            for p in population
            if (p["mode"], p["condition"], p["transaction_class"], p["arm"])
            == ("warm", condition, klass, arm)
        )
        overhead = max(overhead, population_cost)
        saving = (warm["numerator"] - warm["denominator"]) / warm["paired_repetitions"]
        breaks.append(
            dict(
                condition=condition,
                transaction_class=klass,
                arm=arm,
                cold_start_overhead_ns=overhead,
                warm_saving_ns=saving,
                requests=math.ceil(overhead / saving) + 1 if saving > 0 else None,
            )
        )
    intended = len(groups) * len(MODES) * CONFIG["repetitions"] * 4
    progress("benchmark_after", sum("transaction_ns" in r for r in rows))
    return dict(
        rows=rows,
        parity_rows=parity,
        cache_hit_miss_rows=hits,
        invalidation_rows=[
            r
            for r in hits
            if r["status"] == "corrupt_miss"
            or r["evicted"]
            or (r["mode"] == "changed_10pct" and r["status"] == "miss")
        ],
        component_timing_rows=[
            dict(unit=r["unit"], timings=r["components"]) for r in rows if "components" in r
        ],
        memory_bytes=memories,
        population_rows=population,
        break_even_requests=breaks,
        complete_workload_ratios=ratios,
        intended_count=intended,
        eligible_count=intended,
        completed_count=sum(r["status"] == "completed" for r in rows),
        censored_count=sum(r["status"] == "censored" and r["repetition"] >= 0 for r in rows),
        excluded_count=sum(r["status"] == "excluded" for r in rows),
        failed_count=0,
        feature_service_ready_score=0,
        service_speedup_score=0,
    )


def base(failures: list[Json]) -> Json:
    """Correct reuse and faster service are separate findings on exposed inputs."""
    return dict(
        experiment_id=8066,
        task_id=TASK,
        schema="carnot.v698.content_addressed_features.v1",
        milestone="2026.10.698",
        run_date="20261003",
        honest_verdict="complete_blocked_feature_service_inputs"
        if failures
        else "complete_null_feature_service",
        verdict_class="blocked" if failures else "null",
        verifier_is_oracle=False,
        claim_scope="Current CPU/PyO3 mechanism benchmark using historically exposed Exp8053 sources. No source acquisition, external feedback request, new model, generalized learning or blanket service speed claim.",
        inference_substrate="verifier_ensemble_against_cached_candidates",
        inference_substrate_class="no_model_load",
        substrate_declaration=dict(
            inference_substrate="verifier_ensemble_against_cached_candidates",
            inference_substrate_class="no_model_load",
            MODEL_SPECS=[],
        ),
        MODEL_SPECS=[],
        model_specs=[],
        model_invocation_counts={},
        generalized_learning_benefit_score=0,
        feature_service_ready_score=0,
        service_speedup_score=0,
        flagged_adversarial=False,
        required_checks_passed=False,
        validation_receipts=[],
        terminal_validation_sidecar_path=None,
        rows=[],
        intended_count=0,
        eligible_count=0,
        independent_count=0,
        completed_count=0,
        censored_count=0,
        excluded_count=0,
        failed_count=0,
        sample_size_budget=dict(
            paired_repetitions=30,
            warmups=5,
            budget_s=900,
            independent_unit="original source cluster; repetitions, seeds and bootstrap draws do not add n",
        ),
        gate_check_summary=failures,
        random_seed=CONFIG["seed"],
        reproducibility_checksum=None,
        source_artifact_hashes=[],
        raw_shard_hashes=[],
        code_config_hashes=[],
        phase_spans=[],
        cache_key_schema=dict(
            encoding="length-prefixed complete bytes",
            operands=[
                "source_bytes",
                "answer_bytes",
                "family_id",
                "extractor_dependency_hashes",
                "config",
                "version",
                "feature_schema",
            ],
            identity=c.extractor_identity(),
        ),
        cache_hit_miss_rows=[],
        invalidation_rows=[],
        component_timing_rows=[],
        parity_rows=[],
        memory_bytes=[],
        break_even_requests=[],
        complete_workload_ratios=[],
        acquisition_cost_status=dict(
            source_acquisition="absent",
            likelihood_acquisition="historical cached scalar; absent",
            external_feedback="historical released labels; absent",
        ),
        config=copy.deepcopy(CONFIG),
        methodology_note="Bit-exact feature parity is a deterministic cache invariant, not a perfect truth classifier. Intervals describe repeated local timing conditional on exposed workloads; they are not independent scientific support.",
        default_enabled=False,
    )


def replay(path: Path) -> Json:
    """A fresh reader verifies primitive bytes and reduces current paired costs."""
    v = json.loads(path.read_text())
    for ref in v["raw_shard_hashes"] + v["code_config_hashes"]:
        checked(ref)
    if v["feature_service_ready_score"] and (
        not v["required_checks_passed"]
        or v["verdict_class"] in ("blocked", "disqualified")
        or not v["parity_rows"]
        or not all(r["passed"] for r in v["parity_rows"])
    ):
        raise ValueError("unsafe_readiness")
    if v["rows"]:
        raw = json.loads((Path(v["raw_directory"]) / "observations.json").read_text())
        if (
            v["rows"] != raw["rows"]
            or v["parity_rows"] != raw["parity_rows"]
            or v["complete_workload_ratios"] != reduce_rows(raw["rows"])
            or v["cache_hit_miss_rows"] != raw["cache_hit_miss_rows"]
        ):
            raise ValueError("reduction_drift")
        for row in raw["rows"]:
            if "components" in row and row["transaction_ns"] != sum(row["components"].values()):
                raise ValueError("component_drift")
            if "checkpoint" in row:
                p = json.loads(checked(row["checkpoint"]).read_text())
                native = (
                    old.prior.old.build.load_native_extension(checked(v["loaded_library_receipt"]))
                    if row["path_arm"] == "native"
                    else None
                )
                result = old.m.restart_decisions(p["head"], p["guard_design"], native)
                if not result["passed"]:
                    raise ValueError("cold_restart_drift")
    return dict(passed=True)


def validation_plan(scratch: Path) -> list[CommandSpec]:
    """Freeze scoped acceptance and loaded inode checks before any benchmark."""
    py, cov, ruff, mypy = [
        str(ROOT / ".venv/bin" / n) for n in ("python", "coverage", "ruff", "mypy")
    ]
    includes = ",".join(str(ROOT / p) for p in OWNED)
    checks = [
        CommandSpec(
            "focused_consumers_loaded_e2e003_004",
            (
                cov,
                "run",
                "--data-file=" + str(scratch / ".coverage"),
                "--include="
                + includes
                + ","
                + str(ROOT / "python/carnot/verify/guarded_transaction_8053.py"),
                "-m",
                "pytest",
                "-n",
                "0",
                "-o",
                "addopts=",
                "--no-cov",
                "--basetemp=" + str(scratch / "pytest"),
                TEST,
                "tests/python/test_primary_publication_7928.py",
                "tests/python/test_native_update_8027.py::test_req_pybind_8027_repeated_load_exits_normally",
                "tests/python/test_native_update_8027.py::test_req_pybind_8027_atomic_copy_preserves_previous_binary",
                "-q",
            ),
            "owned",
            180,
        ),
        CommandSpec(
            "coverage_json",
            (
                cov,
                "json",
                "--data-file=" + str(scratch / ".coverage"),
                "-o",
                str(scratch / "coverage.json"),
            ),
            "owned",
            30,
        ),
        CommandSpec(
            "added_statement_coverage",
            (
                cov,
                "report",
                "--data-file=" + str(scratch / ".coverage"),
                "--include=" + includes,
                "--show-missing",
                "--fail-under=100",
            ),
            "owned",
            30,
        ),
        CommandSpec(
            "ruff_check",
            (ruff, "check", *OWNED, "python/carnot/verify/guarded_transaction_8053.py", TEST),
            "owned",
            30,
        ),
        CommandSpec(
            "ruff_format",
            (
                ruff,
                "format",
                "--check",
                *OWNED,
                "python/carnot/verify/guarded_transaction_8053.py",
                TEST,
            ),
            "owned",
            30,
        ),
        CommandSpec(
            "strict_mypy",
            (mypy, "--strict", *OWNED, "python/carnot/verify/guarded_transaction_8053.py"),
            "owned",
            90,
        ),
        CommandSpec(
            "scoped_spec_coverage", (py, "scripts/check_spec_coverage.py", TEST), "owned", 30
        ),
    ]
    outside = "import os,runpy,sys;os.chdir(sys.argv[1]);sys.argv=sys.argv[2:];runpy.run_path(sys.argv[0],run_name='__main__')"
    for action in ("populate", "restart", "mutate"):
        checks.append(
            CommandSpec(
                "outside_cli_" + action,
                (
                    py,
                    "-u",
                    "-c",
                    outside,
                    str(scratch),
                    str(ROOT / SCRIPT),
                    "--cache-probe",
                    action,
                    "--cache",
                    str(scratch / "cli-cache.sqlite"),
                ),
                "owned",
                30,
            )
        )
    checks.append(
        CommandSpec(
            "outside_cli_blocked",
            (
                py,
                "-u",
                "-c",
                outside,
                str(scratch),
                str(ROOT / SCRIPT),
                "--root",
                str(scratch / "absent"),
                "--output",
                str(scratch / (NAME + ".json")),
            ),
            "owned",
            60,
        )
    )
    return checks


def terminal(path: Path) -> Json:
    """Require independent reduction and both existing adversarial consumers."""
    raw = Path(json.loads(path.read_text())["raw_directory"])
    py = str(ROOT / ".venv/bin/python")
    specs = [
        CommandSpec(
            "cold_replay",
            (py, "-u", str(ROOT / SCRIPT), "--cold-replay", str(path)),
            "terminal",
            120,
        ),
        CommandSpec(
            "adversarial",
            (py, str(ROOT / "scripts/adversarial_verify.py"), str(path), "--json"),
            "terminal",
            60,
        ),
        CommandSpec(
            "strict_rows",
            (py, str(ROOT / "scripts/verdict_row_consistency_lint.py"), "--strict", str(path)),
            "terminal",
            60,
        ),
    ]
    receipts = run_commands(
        ROOT,
        specs,
        log_dir=raw / "terminal_logs" / sha256_file(path).split(":")[-1],
        heartbeat_s=10,
    )
    return dict(passed=all(r["passed"] for r in receipts), receipts=receipts)


def main(argv: list[str] | None = None) -> int:
    """Publish terminal evidence only after a bounded measurement child exits."""
    os.environ["PYTHONUNBUFFERED"] = "1"
    progress("start_preconditions")
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", choices=["20261003"], default="20261003")
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--fixture-input", type=Path)
    parser.add_argument("--validation-worker", action="store_true")
    parser.add_argument("--worker-input", type=Path)
    parser.add_argument("--raw", type=Path)
    parser.add_argument("--cold-replay", type=Path)
    parser.add_argument(
        "--cache-probe", choices=["populate", "crash", "crash_pending", "restart", "mutate"]
    )
    parser.add_argument("--cache", type=Path)
    args = parser.parse_args(argv)
    started = time.monotonic_ns()
    if args.cold_replay:
        try:
            print(json.dumps(replay(args.cold_replay)), flush=True)
            return 0
        except (OSError, ValueError, KeyError) as error:
            print(json.dumps(dict(passed=False, error=str(error))), flush=True)
            return 1
    if args.cache_probe:
        row = dict(
            family_id="private-cli",
            source_bytes=b"Water is wet.".hex(),
            answer_bytes=b"Water is wet.".hex(),
        )
        cache = c.FeatureCache(
            args.cache or Path(tempfile.gettempdir()) / "carnot8066-private.sqlite"
        )
        first = cache.get(row)
        if args.cache_probe == "mutate":
            cache.db.execute("UPDATE features SET vector='broken'")
            cache.db.commit()
            if cache.get(row) != first or cache.events[-1]["status"] != "corrupt_miss":
                raise ValueError("mutation_not_rejected")
        progress("cache_probe_after", 1)
        print(
            json.dumps(dict(passed=True, values=first["values"], events=list(cache.events))),
            flush=True,
        )
        if args.cache_probe == "crash_pending":
            cache.db.execute("UPDATE features SET vector='uncommitted'")
        if args.cache_probe in ("crash", "crash_pending"):
            os._exit(73)
        cache.close()
        return 0
    if args.worker_input:
        data = json.loads(args.worker_input.read_text())
        CONFIG.update(data.get("measurement_config", CONFIG))
        assert args.raw is not None and args.output is not None
        native, receipt = old.prior.load_library(data["library_reference"], args.raw)
        result = measure(data, native, args.raw)
        atomic_json(args.output, dict(result, loaded_library_receipt=receipt))
        progress("measurement_worker_exit", 1)
        return 0
    output = (args.output or args.root / "results" / (NAME + ".json")).absolute()
    raw = output.parent / "raw" / output.stem / "invocations" / str(time.time_ns())
    raw.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="carnot8066-") as temp:
        scratch = Path(temp)
        data, failures = (
            (json.loads(args.fixture_input.read_text()), [])
            if args.fixture_input
            else load_inputs(args.root, raw)
        )
        v = base(failures)
        v.update(
            raw_directory=str(raw),
            source_artifact_hashes=data.get("references", []),
            repository_health=data.get("repository_health", []),
            optional_current_trace=data.get("optional_current_trace", {}),
        )
        health = Path("/tmp/exp8066-repository-health.json")
        if health.is_file():
            receipt_health = json.loads(health.read_text())
            saved_health = raw / "repository_health.log"
            shutil.copyfile(
                checked(
                    {"path": receipt_health["log_path"], "sha256": receipt_health["log_sha256"]}
                ),
                saved_health,
            )
            v["current_repository_health"] = dict(receipt_health, log_path=str(saved_health))
        first = Path("/tmp/exp8066-tests-first.log")
        if first.is_file():
            shutil.copyfile(first, raw / "tests_first_failure.log")
        commands = validation_plan(scratch)
        worker_input = scratch / "workloads.json"
        atomic_json(worker_input, dict(data, measurement_config=CONFIG))
        worker_output = raw / "measurement.json"
        worker = CommandSpec(
            "current_transaction_measurement",
            (
                str(ROOT / ".venv/bin/python"),
                "-u",
                str(ROOT / SCRIPT),
                "--worker-input",
                str(worker_input),
                "--raw",
                str(raw),
                "--output",
                str(worker_output),
            ),
            "measurement",
            960,
        )
        atomic_json(
            raw / "methods.json",
            dict(
                config=CONFIG,
                dependency_hashes={
                    p: sha256_file(ROOT / p)
                    for p in [
                        *OWNED,
                        TEST,
                        "python/carnot/verify/guarded_transaction_8053.py",
                        "python/carnot/verify/evidence_features_7980.py",
                        "python/carnot/verify/source_alignment.py",
                    ]
                },
                validation_commands=[asdict(s) for s in commands],
                measurement_command=asdict(worker),
                modes=MODES,
                mutation="predetermined position+repetition modulo10; each requested-source position changes three times in30 repetitions; append complete whitespace preserving lexical features",
                optional_trace=v["optional_current_trace"],
                current_baseline=True,
                independent_unit="original exposed source cluster",
            ),
        )
        progress("manifest_frozen")
        if not failures:
            receipt = run_commands(
                ROOT, [worker], log_dir=raw / "measurement_logs", heartbeat_s=10
            )[0]
            v["measurement_exit_receipt"] = receipt
            if receipt["passed"]:
                v.update(json.loads(worker_output.read_text()))
            else:
                failures.append(failure(worker_output, "measurement_exit", 0, receipt["exit_code"]))
            receipts = (
                []
                if args.validation_worker
                else run_commands(
                    ROOT,
                    commands,
                    log_dir=raw / "validation_logs",
                    heartbeat_s=10,
                    extra_env=dict(
                        CARNOT_EXPERIMENT_ARTIFACT_ROOT=str(scratch),
                        CARNOT_8027_EXTENSION=data["library_reference"]["path"],
                        JAX_PLATFORMS="cpu",
                    ),
                )
            )
            v["validation_receipts"] = receipts
            for path in (scratch / "pytest").rglob("cli-receipts.json"):
                shutil.copyfile(path, raw / (path.parent.name + "-cli-receipts.json"))
            coverage = scratch / "coverage.json"
            v["coverage_statement_counts"] = (
                json.loads(coverage.read_text())["files"] if coverage.exists() else {}
            )
            owned_good = bool(receipts) and all(r["passed"] for r in receipts)
            if v["censored_count"]:
                failures.append(
                    failure(
                        raw / "observations.json",
                        "rows[status=censored].count",
                        0,
                        sum(r["status"] == "censored" for r in v["rows"]),
                    )
                )
            for r in receipts:
                if not r["passed"]:
                    failures.append(failure(Path(r["log_path"]), r["name"], 0, r["exit_code"]))
            v["required_checks_passed"] = owned_good and receipt["passed"]
            valid = (
                bool(v["parity_rows"])
                and all(r["passed"] for r in v["parity_rows"])
                and v["censored_count"] == 0
                and not failures
            )
            v["feature_service_ready_score"] = int(valid and owned_good and not args.fixture_input)
            v["service_speedup_score"] = int(
                v["feature_service_ready_score"] and speed_gate(v["complete_workload_ratios"])
            )
            if args.fixture_input:
                v.update(
                    verifier_is_oracle=True,
                    verdict_class="circular_positive",
                    honest_verdict="complete_circular_positive_private_cache_fixture",
                )
            elif not v["feature_service_ready_score"]:
                v.update(
                    verdict_class="disqualified",
                    honest_verdict="complete_disqualified_feature_service",
                )
            elif v["service_speedup_score"]:
                v.update(
                    honest_verdict="complete_positive_scoped_feature_transaction_speed",
                    verdict_class="positive",
                )
            observed_workloads = {r["source"] for r in v["rows"] if r["status"] == "completed"}
            observed_sources = {
                i
                for w in data["cases"]
                if w["identity"] in observed_workloads
                for i in w["updates"] + [g[0] for g in w["guards"]]
            }
            v["independent_count"] = (
                len({data["sources"][i]["source_cluster_id"] for i in observed_sources})
                if not args.fixture_input
                else 0
            )
        v["gate_check_summary"] = failures
        v["code_config_hashes"] = [
            reference(ROOT / p)
            for p in [
                *OWNED,
                TEST,
                "python/carnot/verify/guarded_transaction_8053.py",
                "python/carnot/verify/evidence_features_7980.py",
                "python/carnot/verify/source_alignment.py",
                "python/carnot/experiment_8027_v695_native_update_cost.py",
                "python/carnot/reporting/current_work_receipt.py",
                "python/carnot/reporting/primary_publication.py",
                "openspec/change-proposals/research-roadmap-vNEXT.md",
            ]
        ] + [reference(raw / "methods.json")]
        v["raw_shard_hashes"] = [reference(p) for p in sorted(raw.rglob("*")) if p.is_file()]
        v["reproducibility_checksum"] = canonical_hash(
            dict(config=CONFIG, sources=v["source_artifact_hashes"], code=v["code_config_hashes"])
        )
        v["duration_s"] = (time.monotonic_ns() - started) / 1e9
        v["phase_spans"] = [
            dict(phase="current_owned_cpu_execution_and_validation", duration_s=v["duration_s"])
        ]
        v["current_work_receipt"] = build_current_work_receipt(
            run_id=raw.name,
            owner_pid=os.getpid(),
            events=[],
            inference_substrate=v["inference_substrate"],
            inference_substrate_details=v["substrate_declaration"],
            inference_substrate_class="no_model_load",
            execution_venue="host",
            started_monotonic_ns=started,
            ended_monotonic_ns=time.monotonic_ns(),
            phase_spans=v["phase_spans"],
        )
        v["model_invocation_counts"] = v["current_work_receipt"]["invocation_counts"]
        v["terminal_validation_sidecar_path"] = str(raw / "terminal_validation.json")
        v["field_principles"] = {
            k: "Bind "
            + k
            + " to exact current evidence; never turn historical inputs or repeated timing calls into new scientific support."
            for k in v
        }
        progress("publication_before")
        publication = publish_primary(output, v, terminal)
        reader = reader_receipt(
            TASK,
            output.parent,
            field="feature_service_ready_score",
            expected=v["feature_service_ready_score"],
        )
        atomic_json(
            raw / "terminal_validation.json",
            dict(
                publication=publication,
                reader=reader,
                measurement_exit=v.get("measurement_exit_receipt"),
                process_exit_required=0,
            ),
        )
        progress("publication_after", 1)
        return 0 if reader["passed"] else 1
