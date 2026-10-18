"""REQ-REPORT-8132: measure host work without borrowing historical model calls.

Compact receipts refer to full primitive bytes once. This keeps terminal checks
bounded while preserving enough evidence to repeat every numerical reduction.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import time
from typing import Any

import numpy as np

from carnot import experiment_8119_v702_batched_service_cost as old
from carnot.reporting.current_work_receipt import (
    ZERO_INVOCATION_COUNTS,
    atomic_json,
    canonical_hash,
    sha256_file,
)
from carnot.reporting.experiment_7303_validation_scope import CommandSpec, build_scoped_commands
from carnot.reporting.primary_publication import publish_primary, read_bound_sidecar

Json = dict[str, Any]
ROOT = old.ROOT
NAME = "experiment_8132_v703_service_cost"
CLI = f"scripts/experiments/{NAME}.py"
TEST = "tests/python/test_service_cost_8132.py"
OWNED = [f"python/carnot/{NAME}.py", CLI]
LEARNING = "results/experiment_8130_v703_delayed_energy_memory.json"
ARMS = ("python_scalar", "python_batch", "native_scalar", "native_batch")
CONDITIONS = ("cold", "warm", "miss", "changed-content", "eviction", "restart")
CONFIG: Json = dict(
    seed=7038132,
    batches=[1, 4, 16, 64, 256],
    centers=28,
    repetitions=30,
    warmups=5,
    measurement_ceiling_s=3000,
    bootstraps=10000,
)


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Flushed counters report work actually completed, including child waits."""
    print(f"[exp8132] {phase} completed={completed} pending={pending}", flush=True)


def reference(path: Path) -> Json:
    """Exact byte references prevent large source or log copies in each row."""
    return dict(path=str(path.absolute()), sha256=sha256_file(path))


def inputs(root: Path, raw: Path) -> Json:
    """Reuse qualified capture custody while separating the optional update gate."""
    # The inherited reader also opens a large, failed learning primary. A read-only
    # input view admits only the two independently qualified host dependencies.
    with tempfile.TemporaryDirectory(prefix="carnot-8132-input-view-") as tmp:
        view = Path(tmp)
        for label in (old.NATIVE, old.ACQUISITION):
            original = root / label
            if original.is_file():
                target = view / label
                target.parent.mkdir(parents=True, exist_ok=True)
                target.symlink_to(original.absolute())
                sidecars = target.parent / "raw" / target.stem
                sidecars.parent.mkdir(parents=True, exist_ok=True)
                sidecars.symlink_to((original.parent / "raw" / original.stem).absolute())
        data = old.inputs(view, raw)
        for row in data["checks"]:
            if row["path"].startswith(str(view) + "/"):
                row["path"] = str(root / Path(row["path"]).relative_to(view))
        for ref in data["refs"]:
            if str(ref.get("original_path", "")).startswith(str(view) + "/"):
                ref["original_path"] = str(root / Path(ref["original_path"]).relative_to(view))
    data["checks"] = [r for r in data["checks"] if r["scope"] != "learning"]
    data["learning"] = []
    data["public"] = [
        r for r in data["public"] if r["source_cluster_id"] in data["panel_source_ids"]
    ]
    path = root / LEARNING
    observed: Any = "absent"
    if path.is_file():
        try:
            value = json.loads(path.read_text())
            publication = json.loads(Path(value["terminal_validation_sidecar_path"]).read_text())[
                "publication"
            ]
            bound = read_bound_sidecar(path, Path(publication["sidecar_path"]))
            observed = bool(
                value["required_checks_passed"]
                and not value["flagged_adversarial"]
                and value.get("learning_trajectory_ready_score") == 1
                and bound["report"]["passed"]
                and publication["primary_sha256"] == sha256_file(path)
            )
            if observed:
                data["learning"] = value["update_rows"]
                copied = raw / "inputs" / (sha256_file(path)[7:] + "-" + path.name)
                copied.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(path, copied)
                data["refs"].append(reference(copied))
        except (OSError, ValueError, KeyError) as exc:
            observed = str(exc)
    old.check(
        data["checks"],
        "optional_natural_update_join",
        path,
        "qualified_update_rows",
        True,
        observed,
        "learning",
    )
    for label in [*OWNED, TEST, "tests/python/test_primary_publication_7928.py"]:
        path = ROOT / label
        old.check(data["checks"], "input_path", path, "exists", True, path.is_file(), "host")
        data["refs"].append(reference(path))
    atomic_json(raw / "input_observations.json", dict(checks=data["checks"], refs=data["refs"]))
    return data


def schedule(config: Json) -> list[Json]:
    """Each timed repetition keeps the same source order across all four arms."""
    return [
        dict(
            unit_id=f"b{batch}-{condition}-r{rep}",
            batch=batch,
            condition=condition,
            repetition=rep,
            order=list(ARMS if rep % 2 == 0 else reversed(ARMS)),
        )
        for batch in config["batches"]
        for condition in CONDITIONS
        for rep in range(config["repetitions"])
    ]


def kernel(prepared: tuple[Any, ...], values: Any, model: Any, arm: str) -> Any:
    """Scalar and batch paths run the same equations in the same query order."""
    backend = "rust" if arm.startswith("native") else "python"
    if arm.endswith("scalar"):
        return np.asarray([float(old.kernel(prepared, [row], model, backend)[0]) for row in values])
    return old.kernel(prepared, values, model, backend)


def transaction(data: Json, native: Any, slot: Json, arm: str, raw: Path) -> Json:
    """Charge lifecycle setup and full committed service work to each transaction."""
    raw.mkdir(parents=True, exist_ok=True)
    originals = [r for r in data["public"] if r["arm"] == "full_source"]
    selected = [originals[(slot["repetition"] + i) % len(originals)] for i in range(slot["batch"])]
    base = selected
    if slot["condition"] == "changed-content":
        by = {
            r["source_cluster_id"]: r for r in data["public"] if r["arm"] == "source_order_permuted"
        }
        selected = [by[r["source_cluster_id"]] for r in base]
    state = old.prior.fixture(9, CONFIG["centers"], CONFIG["seed"])
    path = raw / "state.json"
    costs: Json = {}
    began = step = time.perf_counter_ns()
    cache = old.host.cache_service.FeatureCache(
        raw / "features.sqlite", capacity=len(originals), identity=data["extractor"]
    )
    request = lambda row: {key: row[key] for key in ("family_id", "source_bytes", "answer_bytes")}
    if slot["condition"] in ("warm", "changed-content", "eviction", "restart"):
        for row in base:
            cache.get(request(row))
    if slot["condition"] in ("miss", "eviction"):
        for i in range(len(originals)):
            cache.get(
                dict(
                    family_id=f"fixture-decoy-{i}",
                    source_bytes=b"Water contains atoms.".hex(),
                    answer_bytes=b"Atoms.".hex(),
                )
            )
    if slot["condition"] == "restart":
        atomic_json(path, dict(state=state, sha256=canonical_hash(state)))
        cache.close()
        cache = old.host.cache_service.FeatureCache(
            raw / "features.sqlite", capacity=len(originals), identity=data["extractor"]
        )
        state = old.host.read_state(path)
    cache.events.clear()
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
    vectors = [cache.get(request(row))["values"] for row in selected]
    costs["feature_lookup_handling_ns"] = time.perf_counter_ns() - step
    step = time.perf_counter_ns()
    known = {r["judgment_key"]: r for r in data["acquisition"]}
    keys = [old.judgment_key(r, data["config"]) for r in selected]
    p = np.clip(
        [known[key]["probability"] if key in known else 0.5 for key in keys], 1e-6, 1 - 1e-6
    )
    values = np.column_stack((np.log(p / (1 - p)), vectors))
    prepared = old.prepare(state)
    model = native.RustRadial8105(json.dumps(state)) if arm.startswith("native") else None
    costs["lookup_parameters_conversion_ns"] = time.perf_counter_ns() - step
    step = time.perf_counter_ns()
    probabilities = kernel(prepared, values, model, arm)
    costs["arithmetic_and_boundary_ns"] = time.perf_counter_ns() - step
    step = time.perf_counter_ns()
    canonical = old.kernel(prepared, values, None, "python")
    flags = np.minimum(abs(probabilities - 0.1), abs(probabilities - 0.5)) <= 1e-8
    probabilities[flags] = canonical[flags]
    actions = [
        old.k.action(float(p)) if key in known else "abstain"
        for p, key in zip(probabilities, keys, strict=True)
    ]
    state.update(
        version=state["version"] + 1,
        commit_hash=canonical_hash(
            dict(keys=keys, actions=actions, previous=canonical_hash(state))
        ),
    )
    durable = (
        json.loads(native.RustRadial8105(json.dumps(state)).state_json())
        if arm.startswith("native")
        else state
    )
    costs["decision_serialization_ns"] = time.perf_counter_ns() - step
    step = time.perf_counter_ns()
    atomic_json(path, dict(state=durable, sha256=canonical_hash(durable)))
    directory = os.open(raw, os.O_RDONLY)
    os.fsync(directory)
    os.close(directory)
    events = list(cache.events)
    cache.close()
    costs["durable_write_ns"] = time.perf_counter_ns() - step
    step = time.perf_counter_ns()
    result = dict(
        arm=arm,
        probabilities=probabilities.tolist(),
        actions=actions,
        values=values.tolist(),
        durable_state=durable,
        state_path=str(path),
        state_sha256=sha256_file(path),
        rendered_hash=canonical_hash(rendered),
        judgment_keys=keys,
        source_cluster_ids=[r["source_cluster_id"] for r in selected],
        cache_events=events,
        input_bytes=sum(
            len(bytes.fromhex(r["source_bytes"])) + len(bytes.fromhex(r["answer_bytes"]))
            for r in selected
        ),
    )
    costs["return_ns"] = time.perf_counter_ns() - step
    elapsed = time.perf_counter_ns() - began
    result.update(
        full_latency_ns=elapsed,
        arithmetic_ns=costs["arithmetic_and_boundary_ns"],
        components=costs,
        residual_ns=elapsed - sum(costs.values()),
    )
    return result


def parity(arms: list[Json]) -> bool:
    """Numerical closeness cannot conceal changed actions or durable metadata."""
    return bool(
        len(arms) == 4
        and all(
            all(
                row[key] == arms[0][key]
                for key in (
                    "actions",
                    "values",
                    "durable_state",
                    "rendered_hash",
                    "judgment_keys",
                    "source_cluster_ids",
                )
            )
            and np.allclose(row["probabilities"], arms[0]["probabilities"], atol=1e-10, rtol=0)
            for row in arms[1:]
        )
    )


def measure(data: Json, native: Any, raw: Path, config: Json = CONFIG) -> Json:
    """Warm each condition explicitly and retain every interrupted batch mask."""
    plan = schedule(config)
    evidence: Json = dict(pairs=[], warmups=[], timer_units="perf_counter_ns", config=config)
    began = time.monotonic()
    warmed: set[tuple[int, str]] = set()
    raw.mkdir(parents=True, exist_ok=True)
    with (raw / "batch_transcript.jsonl").open("w") as stream:
        for index, slot in enumerate(plan):
            progress("benchmark_before_" + slot["unit_id"], index, len(plan) - index)
            if time.monotonic() - began >= config["measurement_ceiling_s"]:
                pair = dict(
                    slot, arms=[], status="censored", exclusion_reason="measurement_ceiling_3000s"
                )
            else:
                cell = (slot["batch"], slot["condition"])
                if cell not in warmed:
                    for warmup in range(config["warmups"]):
                        for arm in slot["order"]:
                            row = transaction(
                                data,
                                native,
                                slot,
                                arm,
                                raw / "warmups" / slot["unit_id"] / f"{warmup}-{arm}",
                            )
                            evidence["warmups"].append(
                                dict(
                                    unit_id=slot["unit_id"],
                                    warmup=warmup,
                                    arm=arm,
                                    duration_ns=row["full_latency_ns"],
                                    excluded_from_measurement=True,
                                )
                            )
                    warmed.add(cell)
                arms = [
                    transaction(
                        data, native, slot, arm, raw / "transactions" / slot["unit_id"] / arm
                    )
                    for arm in slot["order"]
                ]
                pair = dict(slot, arms=arms, status="completed", exclusion_reason=None)
            evidence["pairs"].append(pair)
            stream.write(json.dumps(pair) + "\n")
            stream.flush()
            progress("benchmark_after_" + slot["unit_id"], index + 1, len(plan) - index - 1)
    atomic_json(raw / "primitive_rows.json", evidence)
    return evidence


def reduce_rows(evidence: Json, config: Json = CONFIG) -> Json:
    """Whole paired batches determine log-ratio intervals and Amdahl ceilings."""
    pairs = evidence["pairs"]
    complete = [p for p in pairs if p["status"] == "completed"]
    failures = sum(
        not parity(p["arms"])
        or any(
            a["full_latency_ns"] <= 0
            or a["arithmetic_ns"] <= 0
            or a["residual_ns"] != a["full_latency_ns"] - sum(a["components"].values())
            or a["residual_ns"] < 0
            for a in p["arms"]
        )
        for p in complete
    )
    intervals = []
    for batch in config["batches"]:
        for condition in CONDITIONS:
            selected = [p for p in complete if p["batch"] == batch and p["condition"] == condition]
            if not selected:
                continue
            by = [{a["arm"]: a for a in p["arms"]} for p in selected]
            for baseline, contender in (
                ("python_scalar", "python_batch"),
                ("python_scalar", "native_scalar"),
                ("python_batch", "native_batch"),
            ):
                left = np.array([a[baseline]["full_latency_ns"] for a in by], dtype=float)
                right = np.array([a[contender]["full_latency_ns"] for a in by], dtype=float)
                if np.any(left <= 0) or np.any(right <= 0):
                    continue
                logs = np.log(left / right)
                rng = np.random.default_rng(config["seed"])
                boot = logs[
                    rng.integers(0, len(logs), size=(config["bootstraps"], len(logs)))
                ].mean(axis=1)
                lower = float(np.exp(np.quantile(boot, 0.05)))
                arithmetic = sum(a[baseline]["arithmetic_ns"] for a in by)
                fraction = arithmetic / float(left.sum())
                intervals.append(
                    dict(
                        batch=batch,
                        condition=condition,
                        baseline=baseline,
                        contender=contender,
                        paired_count=len(selected),
                        geometric_speed_ratio=float(np.exp(logs.mean())),
                        one_sided_95_lower=lower,
                        speed_claim=lower > 1,
                        nfr01_met=lower > 10,
                        arithmetic_fraction=fraction,
                        amdahl_zero_arithmetic_ceiling=1 / (1 - fraction),
                        baseline_median_ns=float(np.median(left)),
                        contender_median_ns=float(np.median(right)),
                    )
                )
    intended = len(schedule(config))
    return dict(
        passed=bool(complete)
        and failures == 0
        and len(complete) == intended
        and {p["unit_id"] for p in pairs} == {p["unit_id"] for p in schedule(config)},
        paired_intervals=intervals,
        bootstrap_repetitions=config["bootstraps"],
        intended_count=intended,
        completed_count=len(complete),
        failed_count=int(failures),
        censored_count=sum(p["status"] == "censored" for p in pairs),
    )


def modeled_bounds(data: Json, evidence: Json) -> list[Json]:
    """Historical acquisition prices exact keys without claiming fresh service."""
    known = {r["judgment_key"]: r for r in data["acquisition"]}
    load_s = sum((r["ended_monotonic_ns"] - r["started_monotonic_ns"]) / 1e9 for r in data["loads"])
    rows = []
    for pair in evidence["pairs"]:
        for arm in pair["arms"]:
            keys = arm["judgment_keys"]
            matched = all(key in known for key in keys)
            costs = (
                [
                    known[key]["component_costs"]["request_wall_s"]
                    + (
                        known[key]["component_costs"]["parsing_ns"]
                        + known[key]["component_costs"]["cache_write_ns"]
                    )
                    / 1e9
                    for key in sorted(set(keys))
                ]
                if matched
                else []
            )
            acquisition_s = sum(costs) if matched else None
            rows.append(
                dict(
                    unit_id=pair["unit_id"],
                    arm=arm["arm"],
                    condition=pair["condition"],
                    matched=matched,
                    exact_keys=sorted(set(keys)),
                    historical_acquisition_s=acquisition_s,
                    historical_load_s=load_s if matched else None,
                    host_s=arm["full_latency_ns"] / 1e9,
                    modeled_total_s=load_s + acquisition_s + arm["full_latency_ns"] / 1e9
                    if matched
                    else None,
                    no_reuse_acquisition_s=sum(
                        known[key]["component_costs"]["request_wall_s"]
                        + (
                            known[key]["component_costs"]["parsing_ns"]
                            + known[key]["component_costs"]["cache_write_ns"]
                        )
                        / 1e9
                        for key in keys
                    )
                    if matched
                    else None,
                    directly_measured_complete_service=False,
                    exclusion_reason=None if matched else "exact_historical_key_absent",
                )
            )
    return rows


def build(data: Json, evidence: Json, raw: Path, receipts: list[Json], duration: float) -> Json:
    """Branch readiness separates validated host work from absent acquisition."""
    reduced = reduce_rows(evidence, evidence["config"])
    owned = bool(receipts) and all(
        r["passed"] for r in receipts if r.get("scope") != "repository_health"
    )
    host_block = next((c for c in data["checks"] if c["scope"] == "host" and not c["passed"]), None)
    ready = int(owned and host_block is None and reduced["passed"])
    verdict = (
        "disqualified"
        if not owned or (host_block is None and not reduced["passed"])
        else "blocked"
        if host_block
        else "circular_positive"
    )
    reason = (
        "owned_validation"
        if verdict == "disqualified"
        else host_block["check"]
        if host_block
        else "host_service_fixture"
    )
    rows = [
        dict(
            unit_id=p["unit_id"],
            source_cluster_id=canonical_hash(sorted(set(a["source_cluster_ids"]))),
            arm=a["arm"],
            condition=p["condition"],
            metric="host_transaction_latency_ns",
            numerator=a["full_latency_ns"],
            denominator=1,
            status=p["status"],
            exclusion_reason=p["exclusion_reason"],
        )
        for p in evidence["pairs"]
        for a in p["arms"]
    ]
    rows += [
        dict(
            unit_id=p["unit_id"],
            source_cluster_id=None,
            arm=arm,
            condition=p["condition"],
            metric="host_transaction_latency_ns",
            numerator=None,
            denominator=None,
            status=p["status"],
            exclusion_reason=p["exclusion_reason"],
        )
        for p in evidence["pairs"]
        if not p["arms"]
        for arm in ARMS
    ]
    code = {
        p: sha256_file(ROOT / p)
        for p in [
            *OWNED,
            TEST,
            old.__file__,
            old.host.__file__,
            old.k.__file__,
            old.host.cache_service.__file__,
            old.prior.__file__,
            old.host.cache_service.features.__file__,
            old.host.cache_service.alignment.__file__,
            old.acquisition.__file__,
        ]
    }
    code["config"] = canonical_hash(evidence["config"])
    value: Json = dict(
        experiment_id=8132,
        task_id="exp8132-service-cost",
        schema="carnot.v703.service_cost.v1",
        run_date="20261004",
        honest_verdict=f"complete_{verdict}_{reason}",
        verdict_class=verdict,
        verifier_is_oracle=True,
        claim_scope="exposed host service with supplied numerical heads and fixture lifecycle injections",
        exposure_scope="exposed_development",
        independent_generalization_score=0,
        generalized_learning_benefit_score=0,
        required_checks_passed=owned,
        flagged_adversarial=False,
        validation_receipts=receipts,
        terminal_validation_sidecar_path=str(raw / "terminal_validation.json"),
        gate_check_summary=data["checks"],
        inference_substrate="verifier_ensemble_against_cached_candidates"
        if evidence["pairs"]
        else "aggregation_from_upstream_artifacts",
        inference_substrate_class="no_model_load",
        MODEL_SPECS=[],
        model_invocation_counts=dict(ZERO_INVOCATION_COUNTS),
        call_ledger=[],
        trained_head_specs=[
            dict(kind="supplied_numerical_fixture", centers=28, trained_currently=False)
        ],
        rows=rows,
        intended_count=reduced["intended_count"],
        eligible_count=reduced["completed_count"],
        independent_count=0,
        completed_count=reduced["completed_count"],
        excluded_count=reduced["intended_count"] - reduced["completed_count"],
        censored_count=reduced["censored_count"],
        failed_count=reduced["failed_count"],
        sample_size_budget=dict(
            evidence["config"],
            independent_scientific_sources=0,
            exposed_source_clusters=len({r["source_cluster_id"] for r in data["public"]}),
            measured_source_ids=data["panel_source_ids"],
            source_selection="median-sized original in each public byte-size third; cyclic within-batch reuse",
        ),
        duration_s=duration,
        random_seed=CONFIG["seed"],
        reproducibility_checksum=canonical_hash(evidence),
        source_artifact_hashes=data["refs"],
        raw_shard_hashes=[],
        code_config_hashes=code,
        phase_spans=[],
        acceptance_gates=dict(
            parity_atol=1e-10,
            identical_decisions=True,
            paired_repetitions=30,
            excluded_warmups=5,
            speed_claim_lower_bound=1,
            nfr01_lower_bound=10,
            current_acquisition_measured=False,
        ),
        cited_upstream_artifacts=[
            dict(
                path=r["path"],
                sha256=r["sha256"],
                fields_imported="historical custody and modeled costs only",
            )
            for r in data["refs"]
            if Path(r["path"]).name.endswith(Path(old.ACQUISITION).name)
        ],
        host_service_ready_score=ready,
        natural_update_cost_ready_score=0,
        complete_service_ready_score=0,
        primitive_rows=reference(raw / "primitive_rows.json"),
        per_transaction_rows=reference(raw / "primitive_rows.json"),
        batch_crossover=reduced["paired_intervals"],
        paired_intervals=reduced["paired_intervals"],
        reduction=reduced,
        component_cost_rows=reference(raw / "primitive_rows.json"),
        modeled_acquisition_bounds=reference(raw / "modeled_acquisition_bounds.json"),
        measurement_config=evidence["config"],
        natural_update_join=dict(
            status="blocked",
            path=str(ROOT / LEARNING),
            exclusion_reason="qualified_natural_update_rows_absent",
            lifecycle_injections="fixtures",
        ),
        nfr01_met=False,
        whole_service_speedup=None,
        methodology_note="Supplied heads and lifecycle injections are fixtures. No independent learning claim. Arithmetic timings include binding crossings; removing them gives an optimistic zero-arithmetic ceiling. Counts use whole batches; rows contain four arms per batch.",
    )
    value["field_principles"] = {
        key: "Binds measured host work to exact bytes; historical acquisition and fixtures provide no fresh complete-service or generalization claim."
        for key in value
    }
    value["field_principles"].update(
        paired_intervals="10000 whole-batch log-ratio bootstraps; the one-sided 95% lower bound must exceed 1 for a host speed claim and 10 for the NFR-01 threshold.",
        complete_service_ready_score="Historical acquisition sums are modeled bounds; current end-to-end acquisition was not measured.",
        natural_update_cost_ready_score="An absent qualified natural update blocks only its optional cost join; injected cache lifecycle changes are fixtures.",
        arithmetic_ns="Native call time includes boundary crossing; the zero-arithmetic ceiling optimistically removes that entire measured component.",
    )
    return value


def execute(commands: list[CommandSpec], raw: Path, expected: int = 0) -> list[Json]:
    """Poll children in bounded waits and bind normal exits to complete logs."""
    receipts = []
    for index, command in enumerate(commands):
        progress("subprocess_before_" + command.name, index, len(commands) - index)
        log = raw / "validation_logs" / (command.name + ".log")
        log.parent.mkdir(parents=True, exist_ok=True)
        began = time.monotonic()
        timed_out = False
        with log.open("w") as stream:
            process = subprocess.Popen(
                command.argv,
                cwd=ROOT,
                stdout=stream,
                stderr=subprocess.STDOUT,
                env=dict(os.environ, PYTHONUNBUFFERED="1", JAX_PLATFORMS="cpu"),
            )
            while process.poll() is None:
                try:
                    process.wait(
                        timeout=min(30, max(0.01, command.timeout_s - (time.monotonic() - began)))
                    )
                except subprocess.TimeoutExpired:
                    progress("waiting_" + command.name, index, len(commands) - index)
                    if time.monotonic() - began >= command.timeout_s:
                        timed_out = True
                        process.kill()
                        process.wait(timeout=30)
        receipts.append(
            dict(
                name=command.name,
                command_argv=list(command.argv),
                scope=command.scope,
                expected_exit=expected,
                actual_exit=process.returncode,
                exit_code=process.returncode,
                normal_exit=not timed_out and process.returncode >= 0,
                timed_out=timed_out,
                passed=not timed_out and process.returncode == expected,
                duration_s=time.monotonic() - began,
                log_path=str(log),
                log_sha256=sha256_file(log),
            )
        )
        progress("subprocess_after_" + command.name, index + 1, len(commands) - index - 1)
    return receipts


def validation_plan(private: Path) -> list[CommandSpec]:
    """Freeze the owned files and actual consumer test names before timing."""
    commands = build_scoped_commands(
        ROOT,
        [
            TEST,
            "tests/python/test_primary_publication_7928.py",
            "tests/python/test_native_radial_8105.py",
        ],
        OWNED[:1],
        static_paths=OWNED[1:],
        basetemp=private,
        coverage_file=private / ".coverage",
    )
    return [
        CommandSpec(
            c.name,
            (*c.argv, "--strict", "--follow-imports=silent")
            if c.name == "changed_module_mypy"
            else tuple(
                f"--include=*/{NAME}.py" if a.startswith("--include=") else a for a in c.argv
            ),
            c.scope,
            300,
        )
        for c in commands
    ]


def validator_commands(path: Path, cold: bool = True) -> list[CommandSpec]:
    """Use unchanged readers with explicit bounded normal-exit receipts."""
    py = str(ROOT / ".venv/bin/python")
    commands = [
        CommandSpec(
            "adversarial",
            (py, str(ROOT / "scripts/adversarial_verify.py"), "--json", str(path)),
            "terminal",
            180,
        ),
        CommandSpec(
            "strict_rows",
            (py, str(ROOT / "scripts/verdict_row_consistency_lint.py"), "--strict", str(path)),
            "terminal",
            180,
        ),
    ]
    if cold:
        commands.insert(
            0,
            CommandSpec(
                "cold_replay",
                (py, "-u", str(ROOT / CLI), "--cold-replay", str(path)),
                "private_cli",
                180,
            ),
        )
    return commands


def terminal(path: Path) -> Json:
    """A passing terminal report requires normal exit, never a timeout."""
    receipts = execute(validator_commands(path), path.parent / "terminal")
    report = dict(passed=all(r["passed"] for r in receipts), receipts=receipts)
    atomic_json(path.parent / "owned_terminal_report.json", report)
    return report


def replay(path: Path) -> bool:
    """Cold reads authenticate primitive, state, source, code and log bytes."""
    try:
        value = json.loads(path.read_text())
        for ref in [*value["raw_shard_hashes"], *value["source_artifact_hashes"]]:
            if sha256_file(Path(ref["path"])) != ref["sha256"]:
                return False
        for label, digest in value["code_config_hashes"].items():
            if label != "config" and sha256_file(ROOT / label) != digest:
                return False
        evidence = json.loads(Path(value["primitive_rows"]["path"]).read_text())
        data = json.loads(
            (Path(value["primitive_rows"]["path"]).parent / "input_data.json").read_text()
        )
        reduced = reduce_rows(evidence, value["measurement_config"])
        rebuilt = build(
            data,
            evidence,
            Path(value["primitive_rows"]["path"]).parent,
            value["validation_receipts"],
            value["duration_s"],
        )
        for field in (
            "rows",
            "reduction",
            "paired_intervals",
            "batch_crossover",
            "host_service_ready_score",
            "reproducibility_checksum",
        ):
            if value[field] != rebuilt[field]:
                return False
        if (
            value["code_config_hashes"]["config"] != canonical_hash(evidence["config"])
            or reduced != value["reduction"]
        ):
            return False
        modeled = json.loads(Path(value["modeled_acquisition_bounds"]["path"]).read_text())
        if modeled != modeled_bounds(data, evidence):
            return False
        if evidence["pairs"] and any(p["arms"] for p in evidence["pairs"]):
            progress("cold_binding_load_before")
            native, loaded = old.host.load_binding(data)
            if sha256_file(Path(native.__file__)) != loaded["sha256"]:
                return False
            progress("cold_binding_load_after")
            for index, pair in enumerate(evidence["pairs"]):
                if index % 50 == 0:
                    progress("cold_replay", index, len(evidence["pairs"]) - index)
                for row in pair["arms"]:
                    if sha256_file(Path(row["state_path"])) != row["state_sha256"]:
                        return False
                    state = old.host.read_state(Path(row["state_path"]))
                    restored = native.RustRadial8105.restore(
                        native.RustRadial8105(json.dumps(state)).checkpoint()
                    )
                    expected = old.kernel(old.prepare(state), row["values"], None, "python")
                    if (
                        state != row["durable_state"]
                        or json.loads(restored.state_json()) != state
                        or not np.allclose(expected, row["probabilities"], atol=1e-10, rtol=0)
                        or not np.allclose(
                            restored.predict(row["values"]), expected, atol=1e-10, rtol=0
                        )
                    ):
                        return False
        return all(
            not r.get("log_path") or sha256_file(Path(r["log_path"])) == r["log_sha256"]
            for r in value["validation_receipts"]
        )
    except (OSError, ValueError, KeyError, TypeError):
        return False


def worker(raw: Path) -> int:
    """The normally exiting child owns all measured CPU and binding activity."""
    data = json.loads((raw / "input_data.json").read_text())
    config = json.loads((raw / "validation_commands.json").read_text())["config"]
    if data["library"] and all(c["passed"] for c in data["checks"] if c["scope"] == "host"):
        progress("binding_load_before")
        native, loaded = old.host.load_binding(data)
        if (
            Path(native.__file__).resolve() != Path(loaded["path"]).resolve()
            or sha256_file(Path(native.__file__)) != loaded["sha256"]
        ):
            raise ValueError("actual_loaded_binding_identity")
        atomic_json(raw / "loaded_binding_receipt.json", loaded)
        progress("binding_load_after")
        measure(data, native, raw, config)
    else:
        evidence = dict(
            pairs=[
                dict(p, arms=[], status="blocked", exclusion_reason="external_host_operand")
                for p in schedule(config)
            ],
            warmups=[],
            timer_units="perf_counter_ns",
            config=config,
        )
        atomic_json(raw / "primitive_rows.json", evidence)
    return 0


def qualify_receipt(data: Json, raw: Path) -> list[Json]:
    """Full production row shape is checked privately before clocks are observed."""
    with tempfile.TemporaryDirectory(prefix="carnot-8132-full-size-") as tmp:
        private = Path(tmp)
        evidence = dict(
            pairs=[
                dict(
                    p, arms=[], status="censored", exclusion_reason="premeasurement_schema_fixture"
                )
                for p in schedule(CONFIG)
            ],
            warmups=[],
            config=CONFIG,
            timer_units="perf_counter_ns",
        )
        atomic_json(private / "primitive_rows.json", evidence)
        atomic_json(private / "modeled_acquisition_bounds.json", [])
        value = build(data, evidence, private, [dict(passed=True)], 1)
        value.update(
            verdict_class="blocked",
            honest_verdict="complete_blocked_premeasurement_schema_fixture",
            receipt_qualification_only=True,
            claim_scope="full-size private schema fixture; no timed observations",
        )
        path = private / (NAME + ".json")
        atomic_json(path, value)
        receipts = execute(validator_commands(path, cold=False), raw / "receipt_qualification")
        atomic_json(
            raw / "receipt_qualification.json",
            dict(
                candidate_sha256=sha256_file(path), row_count=len(value["rows"]), receipts=receipts
            ),
        )
        return receipts


def main(argv: list[str] | None = None) -> int:
    """Freeze checks, qualify storage, measure, then publish only checked bytes."""
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
    args = parser.parse_args(argv)
    if args.cold_replay:
        return 0 if replay(args.cold_replay) else 1
    if args.worker_output:
        return worker(args.worker_output.parent)
    if args.fixture_small and not args.fixture_output:
        parser.error("--fixture-small requires a private fixture output")
    output = (args.fixture_output or args.output).absolute()
    raw = output.parent / "raw" / output.stem
    if output.exists() or (raw / "primitive_rows.json").exists():
        progress("existing_evidence_preserved")
        return 1
    config = dict(CONFIG, batches=[1], repetitions=2, warmups=1) if args.fixture_small else CONFIG
    with tempfile.TemporaryDirectory(prefix="carnot-8132-validation-") as tmp:
        private = Path(tmp)
        commands = validation_plan(private)
        measurement = CommandSpec(
            "measurement_normal_exit",
            (
                str(ROOT / ".venv/bin/python"),
                "-u",
                str(ROOT / CLI),
                "--worker-output",
                str(raw / "primitive_rows.json"),
            ),
            "measurement",
            3120,
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
                config=config,
                measurement=asdict(measurement),
                repository_health=asdict(health),
                terminal=[asdict(c) for c in validator_commands(raw / "terminal_candidate.json")],
            ),
        )
        progress("preconditions_before")
        data = inputs(args.root, raw)
        if data["library"]:
            os.environ["CARNOT_8105_EXTENSION"] = data["library"]["path"]
        atomic_json(raw / "input_data.json", data)
        phases = [dict(phase="preconditions", duration_s=time.monotonic() - began)]
        progress("preconditions_after")
        start = time.monotonic()
        receipts = qualify_receipt(data, raw)
        phases.append(
            dict(phase="full_size_receipt_qualification", duration_s=time.monotonic() - start)
        )
        if not all(r["passed"] for r in receipts):
            progress("receipt_qualification_failed")
            return 1
        start = time.monotonic()
        receipts += execute([measurement], raw)
        phases.append(dict(phase="measurement", duration_s=time.monotonic() - start))
        if not (raw / "primitive_rows.json").is_file():
            progress("measurement_failed_without_primitives")
            return 1
        evidence = json.loads((raw / "primitive_rows.json").read_text())
        start = time.monotonic()
        if not args.fixture_output:
            os.environ["CARNOT_8132_E2E_RECEIPTS"] = str(private / "private_cli_receipts.json")
            receipts += execute(commands, raw)
            private_receipts = private / "private_cli_receipts.json"
            if private_receipts.is_file():
                routes = json.loads(private_receipts.read_text())["receipts"]
                for route in routes:
                    original = Path(route["log_path"])
                    saved = raw / "private_cli_logs" / (route["log_sha256"][7:] + ".log")
                    saved.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(original, saved)
                    route["log_path"] = str(saved)
                atomic_json(raw / "private_cli_receipts.json", dict(receipts=routes))
                receipts += routes
            health_receipts = execute([health], raw / "global_health")
            atomic_json(raw / "repository_health_once.json", dict(receipts=health_receipts))
            os.environ.pop("CARNOT_8132_E2E_RECEIPTS", None)
        phases.append(dict(phase="owned_validation", duration_s=time.monotonic() - start))
        progress("independent_reduction_before")
        atomic_json(raw / "modeled_acquisition_bounds.json", modeled_bounds(data, evidence))
        value = build(data, evidence, raw, receipts, time.monotonic() - began)
        atomic_json(raw / "independent_reduction.json", value["reduction"])
        value["phase_spans"] = phases
        value["fixture_protocol_only"] = bool(args.fixture_output)
        value["raw_shard_hashes"] = [reference(p) for p in raw.glob("*.json*")]
        progress("independent_reduction_after")
        progress("publication_before")
        try:
            publication = publish_primary(output, value, terminal)
        except ValueError as exc:
            candidate = raw / "terminal_candidate.json"
            if candidate.is_file():
                shutil.copyfile(candidate, raw / "failed_terminal_candidate.json")
            atomic_json(raw / "failed_terminal_report.json", dict(error=str(exc)))
            value.update(
                verdict_class="disqualified",
                honest_verdict="complete_disqualified_owned_terminal_validation",
                host_service_ready_score=0,
                natural_update_cost_ready_score=0,
                complete_service_ready_score=0,
                required_checks_passed=False,
            )
            publication = publish_primary(
                output, value, lambda p: dict(passed=True, scope="disqualified_failure_record_only")
            )
        atomic_json(
            raw / "terminal_validation.json",
            dict(
                publication=publication,
                normal_exit=True,
                required_checks_passed=value["required_checks_passed"],
                flagged_adversarial=value["flagged_adversarial"],
            ),
        )
        progress("publication_after", value["completed_count"], 0)
    return 0
