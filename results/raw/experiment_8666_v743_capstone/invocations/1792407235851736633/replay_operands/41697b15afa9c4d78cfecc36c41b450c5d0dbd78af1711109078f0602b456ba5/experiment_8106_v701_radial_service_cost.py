"""REQ-REPORT-8106: complete host costs cannot imply free model acquisition.

Both arithmetic arms share public features and durable state. Changed content
without a matching captured judgment abstains and grants only CPU timing credit.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import asdict
import json
import math
import os
from pathlib import Path
import random
import shutil
import tempfile
import time
from typing import Any

import numpy as np

from carnot import experiment_8105_v701_native_radial_kernel as prior
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
from carnot.verify import content_addressed_features_8066 as cache_service
from carnot.verify import native_radial_8105 as k

Json = dict[str, Any]
ROOT = Path(__file__).resolve().parents[2]
NAME = "experiment_8106_v701_radial_service_cost"
CLI = f"scripts/experiments/{NAME}.py"
TEST = "tests/python/test_radial_service_cost_8106.py"
OWNED = [f"python/carnot/{NAME}.py", CLI]
MODES = ("cold", "warm", "miss", "content-change", "eviction", "restart")
CONFIG: Json = dict(
    seed=7018106, families=30, repetitions=3, centers=[16, 28], modes=list(MODES), max_pairs=1080
)


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Flush real boundaries so a supervisor can distinguish work from silence."""
    print(f"[exp8106] {phase} completed={completed} pending={pending}", flush=True)


def judgment_key(public: Json, identity: Json) -> str:
    """Exact content and model/template bytes prevent reuse after invalidation."""
    return canonical_hash(dict(public=public, identity=identity))


def schedule() -> list[Json]:
    """Alternate the seeded first arm without increasing the source denominator."""
    slots = [(f, c, m, r) for f in range(30) for c in (16, 28) for m in MODES for r in range(3)]
    random.Random(CONFIG["seed"]).shuffle(slots)
    return [
        dict(
            family=f,
            centers=c,
            mode=m,
            repetition=r,
            unit_id=f"f{f}-c{c}-{m}-r{r}",
            order=["python", "rust"] if i % 2 == 0 else ["rust", "python"],
        )
        for i, (f, c, m, r) in enumerate(slots)
    ]


def read_state(path: Path) -> Json:
    """A cold reader checks exact state identity before restoring arithmetic."""
    value = json.loads(path.read_text())
    if value["sha256"] != canonical_hash(value["state"]):
        raise ValueError("state_hash")
    state: Json = value["state"]
    return state


def score(
    state: Json, x: list[float], native: Any, arm: str, costs: Json | None = None
) -> tuple[list[float], list[str], list[bool]]:
    """Reuse the qualified threshold policy so float rounding cannot change actions."""
    began = time.perf_counter_ns()
    model = native.RustRadial8105(json.dumps(state)) if arm == "rust" else None
    _, canonical, _ = k.reference(state, [x], [0.0])
    probabilities = canonical.copy() if model is None else np.asarray(model.predict([x]))
    scoring = time.perf_counter_ns() - began
    began = time.perf_counter_ns()
    flags = (np.minimum(abs(probabilities - 0.1), abs(probabilities - 0.5)) <= 1e-8) | (
        np.minimum(abs(canonical - 0.1), abs(canonical - 0.5)) <= 1e-8
    )
    probabilities[flags] = canonical[flags]
    actions = [k.action(float(p)) for p in probabilities]
    if costs is not None:
        costs.update(scoring_ns=scoring, threshold_fallback_ns=time.perf_counter_ns() - began)
    return probabilities.tolist(), actions, flags.tolist()


def transaction(data: Json, native: Any, slot: Json, arm: str, raw: Path) -> Json:
    """Measure rendering through fsynced state, including every cache lifecycle boundary."""
    raw.mkdir(parents=True, exist_ok=True)
    original = data["public"][slot["family"]]
    public = dict(original)
    if slot["mode"] == "content-change":
        public["source_bytes"] += b" Additional changed evidence.".hex()
    state = prior.fixture(9, slot["centers"], CONFIG["seed"] + slot["family"])
    state_path = raw / "state.json"
    cache_path = raw / "features.sqlite"
    components: Json = {}
    start = time.perf_counter_ns()
    cache = cache_service.FeatureCache(cache_path, capacity=1)
    population = []
    if slot["mode"] in ("warm", "content-change", "eviction", "restart"):
        cache.get(original)
        population.extend(list(cache.events))
    if slot["mode"] in ("miss", "eviction"):
        cache.get(dict(original, family_id=original["family_id"] + "/decoy"))
        population.append(cache.events[-1])
    if slot["mode"] == "restart":
        atomic_json(state_path, dict(state=state, sha256=canonical_hash(state)))
    cache.close()
    warmup_ns = time.perf_counter_ns() - start
    if slot["mode"] == "cold":
        cache_path.unlink()
    started = time.perf_counter_ns()
    step = time.perf_counter_ns()
    cache = cache_service.FeatureCache(cache_path, capacity=1)
    if slot["mode"] == "restart":
        state = read_state(state_path)
    components["lifecycle_ns"] = time.perf_counter_ns() - step
    step = time.perf_counter_ns()
    rendered = json.dumps(
        dict(
            complete_source=bytes.fromhex(public["source_bytes"]).decode(),
            original_answer=bytes.fromhex(public["answer_bytes"]).decode(),
        ),
        sort_keys=True,
    )
    components["rendering_ns"] = time.perf_counter_ns() - step
    extracted = cache.get(public)["values"]
    event = cache.events[-1]
    for name in ("hashing", "loading", "extraction", "storage", "invalidation"):
        components[name + "_ns"] = event[name + "_ns"]
    step = time.perf_counter_ns()
    key = judgment_key(public, data["identity"])
    judgment = data["judgments"].get(key)
    components["judgment_lookup_ns"] = time.perf_counter_ns() - step
    # Missing judgments use a declared neutral fixture operand, then abstain.
    qwen_logit = (
        0.0
        if judgment is None
        else math.log(max(1e-6, min(1 - 1e-6, judgment)) / (1 - max(1e-6, min(1 - 1e-6, judgment))))
    )
    x = [qwen_logit, *extracted]
    step = time.perf_counter_ns()
    probabilities, actions, flags = score(state, x, native, arm, components)
    step = time.perf_counter_ns()
    if judgment is None:
        actions = ["abstain"]
    components["missing_judgment_fallback_ns"] = time.perf_counter_ns() - step
    step = time.perf_counter_ns()
    issued = canonical_hash(state)
    state.update(
        version=state["version"] + 1,
        commit_hash=canonical_hash(dict(previous=issued, key=key, actions=actions)),
    )
    serialized = (
        native.RustRadial8105(json.dumps(state)).state_json()
        if arm == "rust"
        else json.dumps(state)
    )
    durable = json.loads(serialized)
    payload = json.dumps(
        dict(state=durable, sha256=canonical_hash(durable)), sort_keys=True
    ).encode()
    components["update_serialization_ns"] = time.perf_counter_ns() - step
    step = time.perf_counter_ns()
    with state_path.open("wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    directory = os.open(raw, os.O_RDONLY)
    os.fsync(directory)
    os.close(directory)
    cache.close()
    components["durable_commit_ns"] = time.perf_counter_ns() - step
    elapsed = time.perf_counter_ns() - started
    return dict(
        source_id=original["family_id"],
        unit_id=slot["unit_id"],
        arm=arm,
        condition=slot["mode"],
        centers=slot["centers"],
        issued_state=issued,
        metric="complete_host_transaction_ns",
        numerator=elapsed,
        denominator=1,
        status="completed",
        exclusion_reason=None,
        transaction_ns=elapsed,
        components=components,
        cache_event=event,
        judgment_key=key,
        judgment_status="blocked_content_key" if judgment is None else "exact_cached",
        fixture_numeric_operand=judgment is None,
        probabilities=probabilities,
        actions=actions,
        fallback_flags=flags,
        values=x,
        durable_state=durable,
        state_path=str(state_path),
        rendered_hash=canonical_hash(rendered),
        warmup_ns=warmup_ns,
        population_events=population,
    )


def parity(arms: list[Json]) -> bool:
    """Compare complete discrete state and actions alongside bounded float error."""
    a, b = arms
    return bool(
        a["actions"] == b["actions"]
        and a["durable_state"] == b["durable_state"]
        and a["values"] == b["values"]
        and a["rendered_hash"] == b["rendered_hash"]
        and np.allclose(a["probabilities"], b["probabilities"], atol=1e-10, rtol=0)
    )


def measure(data: Json, native: Any, raw: Path) -> Json:
    """Persist every pair before advancing, retaining untimed population separately."""
    plan = schedule()
    atomic_json(raw / "frozen_workload.json", dict(config=CONFIG, schedule=plan, data=data))
    pairs, warmups = [], []
    progress("benchmark_before", 0, len(plan))
    for i, slot in enumerate(plan):
        progress("transaction_before", i, len(plan) - i)
        arms = [
            transaction(data, native, slot, arm, raw / "transactions" / slot["unit_id"] / arm)
            for arm in slot["order"]
        ]
        arms.sort(key=lambda row: row["arm"])
        pair = dict(slot, arms=arms, parity=parity(arms))
        pairs.append(pair)
        warmups.extend(
            dict(
                unit_id=slot["unit_id"],
                arm=row["arm"],
                duration_ns=row["warmup_ns"],
                events=row["population_events"],
                excluded_from_measurement=True,
            )
            for row in arms
        )
        atomic_json(raw / "pairs" / (slot["unit_id"] + ".json"), pair)
        progress("transaction_after", i + 1, len(plan) - i - 1)
    evidence = dict(paired_service_rows=pairs, warmup_rows=warmups)
    atomic_json(raw / "primitive_rows.json", evidence)
    progress("benchmark_after", len(pairs), 0)
    return evidence


def reduce_rows(evidence: Json) -> Json:
    """Recompute paired ratios from primitive times, never from an earlier headline."""
    pairs = evidence.get("paired_service_rows", [])
    valid = len(pairs) == 1080 and {p["unit_id"] for p in pairs} == {
        p["unit_id"] for p in schedule()
    }
    failures = 0
    summaries = []
    for pair in pairs:
        arms = pair["arms"]
        okay = parity(arms) and all(
            row["transaction_ns"] > 0
            and row["numerator"] == row["transaction_ns"]
            and row["transaction_ns"] >= sum(row["components"].values())
            for row in arms
        )
        failures += int(not okay)
    for mode in MODES:
        for centers in (16, 28):
            rows = [p for p in pairs if p["mode"] == mode and p["centers"] == centers]
            if not rows:
                continue
            py = [
                next(r["transaction_ns"] for r in p["arms"] if r["arm"] == "python") for p in rows
            ]
            rust = [
                next(r["transaction_ns"] for r in p["arms"] if r["arm"] == "rust") for p in rows
            ]
            ratios = [a / max(1, b) for a, b in zip(py, rust, strict=True)]
            summaries.append(
                dict(
                    mode=mode,
                    centers=centers,
                    paired_count=len(rows),
                    python_median_ns=float(np.median(py)),
                    rust_median_ns=float(np.median(rust)),
                    python_p95_ns=float(np.percentile(py, 95)),
                    rust_p95_ns=float(np.percentile(rust, 95)),
                    python_population_median_ns=float(
                        np.median(
                            [
                                next(r["warmup_ns"] for r in p["arms"] if r["arm"] == "python")
                                for p in rows
                            ]
                        )
                    ),
                    rust_population_median_ns=float(
                        np.median(
                            [
                                next(r["warmup_ns"] for r in p["arms"] if r["arm"] == "rust")
                                for p in rows
                            ]
                        )
                    ),
                    paired_ratio_median=float(np.median(ratios)),
                    paired_ratio_p95=float(np.percentile(ratios, 95)),
                    normal_exits=len(rows) * 2,
                    normal_exit_scope="transaction function returns; one shared measurement process exit receipt",
                    parity_mismatches=sum(not parity(p["arms"]) for p in rows),
                )
            )
    return dict(
        passed=valid and failures == 0,
        completed_count=len(pairs),
        failed_count=failures,
        summaries=summaries,
    )


def inputs(root: Path, raw: Path) -> tuple[Json, list[Json], list[Json]]:
    """Authenticate exact terminal bytes before admitting native or captured operands."""
    checks: list[Json] = []
    refs: list[Json] = []
    data: Json = dict(
        public=[], judgments={}, identity={}, acquisition=[], library={}, fixture=False
    )

    def observe(
        check: str, path: Path, field: str, expected: Any, observed: Any, scope: str
    ) -> None:
        checks.append(
            dict(
                check=check,
                upstream=path.stem,
                path=str(path),
                hash=sha256_file(path) if path.is_file() else None,
                field=field,
                op="==",
                expected=expected,
                observed=observed,
                scope=scope,
            )
        )

    def authenticated(path: Path, scope: str) -> Json:
        value: Json = json.loads(path.read_text())
        observe(
            "required_checks_passed",
            path,
            "required_checks_passed",
            True,
            value.get("required_checks_passed"),
            scope,
        )
        observe(
            "flagged_adversarial",
            path,
            "flagged_adversarial",
            False,
            value.get("flagged_adversarial"),
            scope,
        )
        terminal = Path(value["terminal_validation_sidecar_path"])
        side = json.loads(terminal.read_text())["publication"]["sidecar_path"]
        bound = read_bound_sidecar(path, Path(side))
        qualified = (
            value.get("required_checks_passed") is True
            and value.get("flagged_adversarial") is False
            and bound["report"]["passed"] is True
        )
        observe(
            "capture_qualified" if scope == "acquisition" else "native_qualified",
            path,
            "qualified",
            True,
            qualified,
            scope,
        )
        if not qualified:
            raise ValueError("unqualified_terminal")
        for source in (path, terminal, Path(side)):
            destination = raw / "inputs" / (source.stem + "-" + sha256_file(source)[7:19] + ".json")
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, destination)
            refs.append(dict(path=str(destination), sha256=sha256_file(destination)))
        return value

    def checked(ref: Json) -> Json:
        path = Path(ref["path"])
        if sha256_file(path) != ref["sha256"]:
            raise ValueError("input_hash")
        destination = raw / "inputs" / (path.stem + "-" + ref["sha256"][7:19] + ".json")
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, destination)
        refs.append(dict(path=str(destination), sha256=sha256_file(destination)))
        return dict(json.loads(path.read_text()))

    for label in [
        *prior.INPUTS[:11],
        "scripts/experiments/experiment_8078_v699_feature_cache_core.py",
        "scripts/experiments/experiment_8079_v699_feature_cache_lifecycle.py",
        "python/carnot/experiment_8078_v699_feature_cache_core.py",
        "python/carnot/experiment_8079_v699_feature_cache_lifecycle.py",
    ]:
        path = root / label
        observe("input_exists", path, "exists", True, path.is_file(), "service")
        if path.is_file():
            refs.append(dict(path=str(path), sha256=sha256_file(path)))
    for tool in ("python", "pytest", "coverage", "ruff", "mypy"):
        path = ROOT / ".venv/bin" / tool
        observe("tool_exists", path, "exists", True, path.is_file(), "service")
    path = root / "results/experiment_8105_v701_native_radial_kernel.json"
    try:
        value = authenticated(path, "service")
        library = Path(value["native_library_path"])
        observe(
            "loaded_binding_hash",
            library,
            "sha256",
            value["native_library_sha256"],
            sha256_file(library),
            "service",
        )
        observe(
            "native_readiness",
            path,
            "native_kernel_ready_score",
            1,
            value["native_kernel_ready_score"],
            "service",
        )
        data["library"] = dict(path=str(library), sha256=sha256_file(library))
    except (OSError, ValueError, KeyError) as exc:
        observe("native_input", path, "authenticated", True, str(exc), "service")
    for identity, filename in [(8099, "fit_source_capture"), (8102, "learning_stream_capture")]:
        path = root / f"results/experiment_{identity}_v701_{filename}.json"
        try:
            capture = authenticated(path, "acquisition")
            methods = authenticated(
                root / "results/experiment_8098_v701_development_methods.json", "acquisition"
            )
            public = checked(methods["role_manifests"]["stream"])["request_rows"]
            manifest = checked(capture["capture_manifest"])
            checked(
                next(
                    r
                    for r in capture["raw_shard_hashes"]
                    if Path(r["path"]).name == "primitive_rows.json"
                )
            )
            identity_key = dict(
                model=capture["gguf_sha256"],
                template=capture["chat_template_sha256"],
                config=manifest["config"],
            )
            ledger = {
                r["call_id"]: r
                for r in capture["call_ledger"]
                if r["status"] == "completed" and r["scope"] == "current"
            }
            eligible = {
                r["family_id"]: r for r in capture["rows"] if r.get("probability") is not None
            }
            load = next(r for r in ledger.values() if r["operation"] == "model_load")
            selected, judgments, acquisition = [], {}, [dict(load, upstream=str(path))]
            for p in public:
                row = eligible.get(p["family_id"])
                if row is None:
                    continue
                event = ledger[row["unit_id"]]
                request = json.loads(row["request"]["messages"][1]["content"])
                if (
                    row["public_hash"] != canonical_hash(p)
                    or request["complete_source"] != bytes.fromhex(p["source_bytes"]).decode()
                    or request["original_answer"] != bytes.fromhex(p["answer_bytes"]).decode()
                    or canonical_hash(row["request"]) != event["request_sha256"]
                    or canonical_hash(row["raw_response"]) != event["response_sha256"]
                    or json.loads(row["raw_response"]["choices"][0]["message"]["content"])[
                        "unsupported_probability"
                    ]
                    != row["probability"]
                ):
                    raise ValueError("capture_content_or_receipt_key")
                key = judgment_key(p, identity_key)
                selected.append(p)
                judgments[key] = row["probability"]
                acquisition.append(
                    dict(
                        event,
                        upstream=str(path),
                        content_model_template_key=key,
                        public_hash=row["public_hash"],
                    )
                )
                if len(selected) == 30:
                    break
            if len(selected) != 30:
                raise ValueError("thirty_matched_captures_required")
            data.update(
                public=selected, judgments=judgments, identity=identity_key, acquisition=acquisition
            )
        except (OSError, ValueError, KeyError, StopIteration) as exc:
            observe(
                "capture_input",
                path,
                "authenticated_current_receipts",
                True,
                str(exc),
                "acquisition",
            )
    if not data["public"] and data["library"]:
        data.update(
            public=[
                dict(
                    family_id=f"private-fixture-{i}",
                    source_bytes=f"Water has {i} atoms.".encode().hex(),
                    answer_bytes=f"Water has {i} atoms.".encode().hex(),
                )
                for i in range(30)
            ],
            fixture=True,
        )
    atomic_json(raw / "input_observations.json", dict(checks=checks, refs=refs))
    return data, checks, refs


def load_binding(data: Json) -> tuple[Any, Json]:
    """Load only the exact qualified binary; successful compilation is insufficient."""
    ref = data["library"]
    if sha256_file(Path(ref["path"])) != ref["sha256"]:
        raise ValueError("loaded_binding_hash")
    progress("before_binding_load")
    native = k.build.load_native_extension(Path(ref["path"]))
    progress("after_binding_load")
    return native, dict(ref, actual_loaded=True, module_file=native.__file__)


def acquisition_totals(data: Json, summaries: list[Json]) -> list[Json]:
    """Charge one load and one fresh judgment under explicit identical-content reuse."""
    events = data["acquisition"]
    if not events:
        return []
    loads = [
        (r["ended_monotonic_ns"] - r["started_monotonic_ns"]) / 1e9
        for r in events
        if r["operation"] == "model_load"
    ]
    calls = [
        (r["ended_monotonic_ns"] - r["started_monotonic_ns"]) / 1e9
        for r in events
        if r["operation"] == "generation"
    ]
    result = []
    for cell in summaries:
        for arm in ("python", "rust"):
            for reuse in (1, 10, 100):
                host = cell[arm + "_median_ns"] / 1e9
                population = cell.get(arm + "_population_median_ns", 0) / 1e9
                load, judgment = sum(loads), float(np.median(calls))
                matched = cell["mode"] != "content-change"
                result.append(
                    dict(
                        mode=cell["mode"],
                        centers=cell["centers"],
                        arm=arm,
                        reuse_count=reuse,
                        model_load_s=load,
                        fresh_judgment_s=judgment,
                        host_per_request_s=host,
                        initial_cache_population_s=population,
                        total_s=load + judgment + population + reuse * host if matched else None,
                        amortized_per_request_s=(load + judgment + population) / reuse + host
                        if matched
                        else None,
                        fresh_every_request_total_s=load + population + reuse * (judgment + host)
                        if matched
                        else None,
                        composed_estimate=True,
                        directly_measured_end_to_end=False,
                        status="conditional" if matched else "blocked",
                        exclusion_reason=None if matched else "content_key_has_no_fresh_judgment",
                    )
                )
    return result


def worker(root: Path, raw: Path) -> Json:
    """Freeze exact code/input operands before the normally exiting measurement child."""
    began = time.monotonic()
    data, checks, refs = inputs(root, raw)
    phases = [dict(phase="preconditions", duration_s=time.monotonic() - began)]
    hashes = {p: sha256_file(ROOT / p) for p in [*OWNED, TEST]}
    for module in (k, prior, cache_service, cache_service.features, cache_service.alignment):
        hashes[str(Path(module.__file__).relative_to(ROOT))] = sha256_file(Path(module.__file__))
    hashes["config"] = canonical_hash(CONFIG)
    evidence: Json = dict(paired_service_rows=[], warmup_rows=[])
    loaded: Json = {}
    if data["public"] and not any(
        r["scope"] == "service" and r["expected"] != r["observed"] for r in checks
    ):
        step = time.monotonic()
        native, loaded = load_binding(data)
        phases.append(dict(phase="binding_load", duration_s=time.monotonic() - step))
        step = time.monotonic()
        evidence = measure(data, native, raw)
        phases.append(dict(phase="matched_transactions", duration_s=time.monotonic() - step))
    work = dict(
        data=data,
        evidence=evidence,
        checks=checks,
        refs=refs,
        loaded_binding_receipt=loaded,
        duration_s=time.monotonic() - began,
        code_config_hashes=hashes,
        phase_spans=phases,
    )
    atomic_json(raw / "work.json", work)
    return work


def build(work: Json, receipts: list[Json], raw: Path, reduced: Json) -> Json:
    """Valid host work survives optional upstream failures without making them zero cost."""
    checks, data = work["checks"], work["data"]
    gates = [r for r in checks if r["expected"] != r["observed"]]
    service_block = [r for r in gates if r["scope"] == "service"]
    owned = bool(receipts) and all(
        r["passed"] for r in receipts if r.get("scope") != "repository_health"
    )
    ready = int(owned and reduced["passed"] and not service_block)
    # The changed-content route has no current matching Qwen capture by design.
    acquisition_block = dict(
        check="changed_content_current_capture",
        upstream="changed_content_current_capture",
        path=str(raw / "primitive_rows.json"),
        hash=None,
        field="exact_judgment_key",
        op="==",
        expected="qualified content/model/template match",
        observed="absent; abstain; no model call",
        scope="acquisition",
    )
    gates.append(acquisition_block)
    verdict = (
        "disqualified" if not owned or (not service_block and not reduced["passed"]) else "blocked"
    )
    reason = service_block[0]["upstream"] if service_block else acquisition_block["upstream"]
    evidence = work["evidence"]
    arms = [r for p in evidence["paired_service_rows"] for r in p["arms"]]
    value: Json = dict(
        experiment_id=8106,
        task_id="exp8106-radial-service-cost",
        run_date="20261004",
        schema="carnot.v701.radial_service_cost.v1",
        honest_verdict="complete_"
        + verdict
        + "_"
        + (reason if verdict == "blocked" else "owned_validation"),
        verdict_class=verdict,
        service_cost_ready_score=ready,
        acquisition_cost_ready_score=0,
        verifier_is_oracle=0,
        claim_scope=0,
        exposure_scope=0,
        generalized_learning_benefit_score=0,
        scope_note="Previously exposed development requests and supplied radial coefficients; no independent generalization claim.",
        methodology_note="Matched complete host transactions include source rendering, lexical lookup/extraction, radial scoring, canonical threshold fallback and fsynced state. Separate current V701 load/judgment receipts give conditional composed estimates only; no current model calls.",
        flagged_adversarial=False,
        required_checks_passed=owned,
        validation_receipts=receipts,
        terminal_validation_sidecar_path=str(raw / "terminal_validation.json"),
        preconditions_checked=checks,
        gate_check_summary=gates,
        inference_substrate="verifier_ensemble_against_cached_candidates",
        inference_substrate_class="no_model_load",
        MODEL_SPECS=[],
        model_invocation_counts=deepcopy(ZERO_INVOCATION_COUNTS),
        trained_head_specs=[
            dict(kind="supplied_radial_fixture", centers=c, trained_currently=False)
            for c in (16, 28)
        ],
        rows=[
            dict(
                source_id=p["arms"][0]["source_id"],
                unit_id=p["unit_id"],
                arm="paired_python_rust",
                condition=p["mode"],
                issued_state=p["arms"][0]["issued_state"],
                metric="paired_host_latency_ratio",
                numerator=next(r["transaction_ns"] for r in p["arms"] if r["arm"] == "python"),
                denominator=next(r["transaction_ns"] for r in p["arms"] if r["arm"] == "rust"),
                status="completed",
                exclusion_reason=None,
            )
            for p in evidence["paired_service_rows"]
        ],
        intended_count=1080,
        eligible_count=len(evidence["paired_service_rows"]),
        independent_count=0,
        completed_count=reduced["completed_count"],
        excluded_count=1080 - reduced["completed_count"],
        censored_count=0,
        failed_count=reduced["failed_count"],
        sample_size_budget=dict(
            CONFIG, unique_request_families=len(data["public"]), independent_natural_sources=0
        ),
        duration_s=work["duration_s"],
        random_seed=CONFIG["seed"],
        reproducibility_checksum=canonical_hash(work),
        source_artifact_hashes=work["refs"],
        raw_shard_hashes=[],
        code_config_hashes=work["code_config_hashes"],
        phase_spans=work["phase_spans"],
        loaded_binding_receipt=work["loaded_binding_receipt"],
        paired_service_rows=evidence["paired_service_rows"],
        warmup_rows=evidence.get("warmup_rows", []),
        component_cost_rows=[
            dict(unit_id=r["unit_id"], arm=r["arm"], **r["components"]) for r in arms
        ],
        acquisition_join_rows=data["acquisition"],
        acquisition_estimates=acquisition_totals(data, reduced["summaries"]),
        reuse_assumptions=dict(
            counts=[1, 10, 100],
            model_loads=1,
            fresh_judgments=1,
            identical_content_required=True,
            state_updates_measured=True,
            estimates_are_separate_stage_sums=True,
            warm_population_in_separate_rows=True,
        ),
        cache_key_schema=dict(
            lexical=cache_service.extractor_identity(),
            judgment="SHA256(exact public fields, GGUF, chat template, capture config)",
        ),
        nfr01_scope=dict(
            closed=False,
            required_deployed_matched_speedup=10,
            deployed_service_measured=False,
            warm_cache_does_not_establish_first_use_cost=True,
        ),
        reduction=reduced,
        acceptance_gates=dict(
            owned_validation=owned,
            complete_host_matrix=reduced["passed"],
            exact_changed_capture=False,
        ),
    )
    value["field_principles"] = {
        key: "Recorded "
        + key
        + " binds actual operands; it grants no independent learning or deployed-service credit."
        for key in value
    }
    value["field_principles"].update(
        service_cost_ready_score="Complete matched host work retains its own readiness despite missing acquisition.",
        acquisition_cost_ready_score="No matching changed-content capture; unavailable acquisition is not zero.",
        acquisition_estimates="Sums of separate stages are conditional estimates, not measured end-to-end latencies.",
        independent_count="Thirty exposed source families, repetitions and synthetic heads are not independent scientific evidence.",
    )
    return value


def replay(path: Path) -> bool:
    """Independently reopen primitive state and recheck actions in a cold process."""
    try:
        value = json.loads(path.read_text())
        evidence = dict(paired_service_rows=value["paired_service_rows"])
        if not evidence["paired_service_rows"]:
            return value["service_cost_ready_score"] == 0
        if not reduce_rows(evidence)["passed"]:
            return False
        native, _ = load_binding(dict(library=value["loaded_binding_receipt"]))
        for pair in evidence["paired_service_rows"]:
            for row in pair["arms"]:
                state = read_state(Path(row["state_path"]))
                if state != row["durable_state"]:
                    return False
                restored = native.RustRadial8105.restore(
                    native.RustRadial8105(json.dumps(state)).checkpoint()
                )
                if json.loads(restored.state_json()) != state:
                    return False
                _, expected, _ = k.reference(state, [row["values"]], [0.0])
                if not np.allclose(
                    expected, row["probabilities"], atol=1e-10, rtol=0
                ) or not np.allclose(
                    restored.predict([row["values"]]), expected, atol=1e-10, rtol=0
                ):
                    return False
        for ref in value["raw_shard_hashes"]:
            if sha256_file(Path(ref["path"])) != ref["sha256"]:
                return False
            if (
                Path(ref["path"]).name == "primitive_rows.json"
                and json.loads(Path(ref["path"]).read_text())["paired_service_rows"]
                != evidence["paired_service_rows"]
            ):
                return False
        return True
    except (OSError, ValueError, KeyError, TypeError):
        return False


def validation_plan(private: Path) -> list[CommandSpec]:
    """Seal explicit checks before timing and cover only code introduced here."""
    commands = build_scoped_commands(
        ROOT,
        [TEST, "tests/python/test_primary_publication_7928.py"],
        OWNED[:1],
        static_paths=OWNED[1:],
        basetemp=private,
        coverage_file=private / ".coverage",
    )
    commands = [
        CommandSpec(
            c.name,
            c.argv + ("--strict", "--follow-imports=silent")
            if c.name == "changed_module_mypy"
            else c.argv,
            c.scope,
            300,
        )
        for c in commands
    ]
    return commands


def execute(commands: list[CommandSpec], raw: Path, private: Path) -> list[Json]:
    """Use the existing heartbeat supervisor and retain exact argv, exit and log hash."""
    receipts = []
    for command in commands:
        cwd = private if command.scope == "private_cli" else ROOT
        receipts += run_commands(
            cwd,
            [command],
            log_dir=raw / "validation_logs" / command.name,
            heartbeat_s=30,
            extra_env={"CARNOT_8106_E2E_RECEIPTS": str(raw / "private_cli_receipts.json")}
            if command.name in {"focused_pytest", "changed_module_coverage"}
            else None,
        )
        receipts[-1]["execution_cwd"] = str(cwd)
    return receipts


def terminal(path: Path) -> Json:
    """Owned replay and established adversarial readers must agree before publication."""
    py = str(ROOT / ".venv/bin/python")
    commands = [
        CommandSpec(
            "cold_replay",
            (py, "-u", str(ROOT / CLI), "--cold-replay", str(path)),
            "private_cli",
            120,
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
    with tempfile.TemporaryDirectory(prefix="carnot-8106-terminal-") as tmp:
        receipts = execute(commands, path.parent, Path(tmp))
    return dict(passed=all(r["passed"] for r in receipts), receipts=receipts)


def main(argv: list[str] | None = None) -> int:
    """Require normal measurement exit and owned checks without retrying completed science."""
    began = time.monotonic()
    os.environ["PYTHONUNBUFFERED"] = "1"
    progress("start")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", choices=["20261004"], default="20261004")
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, default=ROOT / "results" / (NAME + ".json"))
    parser.add_argument("--fixture-output", type=Path)
    parser.add_argument("--worker-output", type=Path)
    parser.add_argument("--cold-replay", type=Path)
    parser.add_argument("--mutate", action="store_true")
    args = parser.parse_args(argv)
    if args.cold_replay:
        return 0 if replay(args.cold_replay) else 1
    if args.worker_output:
        worker(args.root, args.worker_output.parent)
        return 0
    output = (args.fixture_output or args.output).absolute()
    raw = output.parent / "raw" / output.stem
    if output.exists() or (raw / "work.json").exists():
        progress("existing_evidence_preserved")
        return 1
    with tempfile.TemporaryDirectory(prefix="carnot-8106-") as tmp:
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
            ),
            "private_cli",
            300,
        )
        atomic_json(
            raw / "validation_commands.json",
            dict(
                commands=[asdict(c) for c in commands],
                measurement=asdict(child),
                config=CONFIG,
                code_hashes={p: sha256_file(ROOT / p) for p in [*OWNED, TEST]},
            ),
        )
        receipts = execute([child], raw, private)
        if (raw / "work.json").is_file():
            work = json.loads((raw / "work.json").read_text())
        else:
            work = worker(private / "missing-worker", raw)
            receipts[0]["passed"] = False
        validation_began = time.monotonic()
        if not args.fixture_output:
            receipts += execute(commands, raw, private)
        if (raw / "private_cli_receipts.json").is_file():
            receipts += json.loads((raw / "private_cli_receipts.json").read_text())["receipts"]
        work["duration_s"] = time.monotonic() - began
        work["phase_spans"].append(
            dict(phase="owned_validation", duration_s=time.monotonic() - validation_began)
        )
        atomic_json(raw / "work.json", work)
        reduced = reduce_rows(work["evidence"])
        atomic_json(raw / "independent_reduction.json", reduced)
        value = build(work, receipts, raw, reduced)
        if (raw / "repository_health_once.json").is_file():
            value["repository_health"] = json.loads(
                (raw / "repository_health_once.json").read_text()
            )["receipts"]
            value["field_principles"]["repository_health"] = (
                "One global diagnostic is reported separately from owned service qualification."
            )
        if args.fixture_output:
            value.update(
                service_cost_ready_score=0, required_checks_passed=False, fixture_protocol_only=True
            )
        value["raw_shard_hashes"] = [
            dict(path=str(p), sha256=sha256_file(p))
            for p in raw.glob("*.json")
            if p.name not in ("terminal_candidate.json", "terminal_validation.json")
        ]
        if args.mutate:
            value["service_cost_ready_score"] = 1
            atomic_json(raw / "failed_mutation_candidate.json", value)
            progress("mutation_rejected")
            return 1
        publication = publish_primary(output, value, terminal)
        atomic_json(
            raw / "terminal_validation.json",
            dict(publication=publication, measurement_exit_receipt=receipts[0], normal_exit=True),
        )
    progress("terminal_published", value["completed_count"], 0)
    return 0
