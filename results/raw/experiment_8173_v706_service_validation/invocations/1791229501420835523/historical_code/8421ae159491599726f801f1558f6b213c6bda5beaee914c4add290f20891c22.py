"""REQ-VERIFY-8145: time natural cached heads without loading a generator.

The existing Rust design evaluates Gaussian centers. The shared host adds the
historical logit offset because the qualified binding has no offset parameter.
"""

from __future__ import annotations

from copy import deepcopy
import json
import os
from pathlib import Path
import time
from typing import Any

import numpy as np

from carnot import experiment_8132_v703_service_cost as prior
from carnot.reporting.current_work_receipt import (
    ZERO_INVOCATION_COUNTS,
    atomic_json,
    canonical_hash,
    sha256_file,
)
from carnot.reporting.primary_publication import read_bound_sidecar
from carnot.verify import learning_protocol_8138 as engine

Json = dict[str, Any]
ROOT = prior.ROOT
NAME = "experiment_8145_v704_natural_service_cost"
CLI = f"scripts/experiments/{NAME}.py"
TEST = "tests/python/test_natural_service_cost_8145.py"
OWNED = [
    f"python/carnot/{NAME}.py",
    "python/carnot/reporting/natural_service_execution_8145.py",
    CLI,
]
UPSTREAM = "results/experiment_8143_v704_delayed_energy_memory.json"
MODEL_SPECS: list[Json] = []
CONFIG: Json = dict(prior.CONFIG, seed=7048145, measurement_ceiling_s=2400)
reference = prior.reference


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Report real completed counts under this invocation's identity, without buffering."""
    print(f"[exp8145] {phase} completed={completed} pending={pending}", flush=True)


def gate(data: Json, path: Path, field: str, expected: Any, observed: Any) -> None:
    """Keep the exact failed operand so external absence is a terminal result."""
    data["checks"].append(
        dict(
            check=field,
            upstream=path.stem,
            path=str(path),
            hash=sha256_file(path) if path.is_file() else None,
            artifact_field=field,
            op="==",
            expected=expected,
            observed=observed,
            passed=expected == observed,
        )
    )


def inputs(root: Path, raw: Path) -> Json:
    """Authenticate public vectors, every seed's categories and a sealed natural head."""
    data: Json = dict(
        ready=False,
        checks=[],
        refs=[],
        public=[],
        updates=[],
        categories=dict(accepted=0, rejected=0),
        library={},
    )
    path = root / UPSTREAM
    progress("input_authentication_before")
    gate(data, path, "resource_exists", True, path.is_file())
    if path.is_file():
        try:
            value = json.loads(path.read_text())
            for field, expected in [
                ("learning_trajectory_ready_score", 1),
                ("required_checks_passed", True),
                ("flagged_adversarial", False),
                ("fixture_mode", False),
            ]:
                gate(data, path, field, expected, value.get(field))
            terminal_path = Path(value["terminal_validation_sidecar_path"])
            terminal = json.loads(terminal_path.read_text())["publication"]
            bound = read_bound_sidecar(path, Path(terminal["sidecar_path"]))
            gate(data, path, "terminal.report.passed", True, bound["report"]["passed"])
            refs = [
                reference(path),
                reference(terminal_path),
                reference(Path(terminal["sidecar_path"])),
                value["final_head_manifest"],
                value["input_manifests"]["stream_feature_manifest"],
                *[r["state"] for r in value["state_manifest"]],
            ]
            for ref in refs:
                gate(
                    data, Path(ref["path"]), "sha256", ref["sha256"], sha256_file(Path(ref["path"]))
                )
            data["refs"] = refs
            public = json.loads(Path(refs[4]["path"]).read_text())["rows"]
            heads = json.loads(Path(refs[3]["path"]).read_text())["rows"]
            state_path = Path(value["state_manifest"][0]["state"]["path"])
            state = json.loads(state_path.read_text())
            gate(
                data,
                state_path,
                "final_head.state_hash",
                heads[0]["state_hash"],
                canonical_hash(state),
            )
            data.update(
                public=public[64:],
                head=state["arms"]["error_center"],
                geometry=state["geometry"],
                heads_ref=refs[3],
                historical_model_provenance=value["cited_upstream_artifacts"],
                trained_head_specs=value["trained_head_specs"],
            )
            for event in value["admission_rows"]:
                if event["kind"] == "admit_once":
                    for step in event["steps"].values():
                        data["categories"]["accepted" if step else "rejected"] += 1
            for event in state["events"]:
                if event["kind"] != "admit_once":
                    continue
                commit = max(
                    r["slot"]
                    for r in state["events"]
                    if r["kind"] == "commit_candidate" and r["slot"] < event["slot"]
                )
                checkpoint = state_path.parent / f"checkpoint-{commit}-commit_candidate.json"
                opened = state_path.parent / "opened_labels.json"
                for source in (checkpoint, opened):
                    ref = next(r for r in value["raw_shard_hashes"] if r["path"] == str(source))
                    gate(data, source, "sha256", ref["sha256"], sha256_file(source))
                    data["refs"].append(ref)
                before = json.loads(checkpoint.read_text())
                labels = json.loads(opened.read_text())
                records = [
                    dict(
                        state["issued"][slot - 1],
                        y=labels[str(slot)]["y"],
                        label_slot=slot,
                        values=public[slot - 1]["values"],
                        source_cluster_id=public[slot - 1]["source_cluster_id"],
                    )
                    for slot in event["labels"]
                ]
                data["updates"].append(
                    dict(
                        event=event,
                        event_key=canonical_hash(dict(seed=state["seed"], event=event)),
                        before=before,
                        records=records,
                    )
                )
            native_path = root / prior.old.NATIVE
            native = json.loads(native_path.read_text())
            data["library"] = dict(
                path=native["native_library_path"], sha256=native["native_library_sha256"]
            )
            for ref in [reference(native_path), data["library"]]:
                gate(
                    data, Path(ref["path"]), "sha256", ref["sha256"], sha256_file(Path(ref["path"]))
                )
                data["refs"].append(ref)
            gate(
                data,
                native_path,
                "native_kernel_ready_score",
                1,
                native.get("native_kernel_ready_score"),
            )
            data["ready"] = all(r["passed"] for r in data["checks"])
        except (OSError, ValueError, KeyError, StopIteration) as exc:
            gate(data, path, "authenticated_natural_inputs", True, str(exc))
    atomic_json(raw / "input_data.json", data)
    progress("input_authentication_after", int(data["ready"]), 0)
    return data


def state_for(head: Json, geometry: Json) -> Json:
    """Preserve head provenance while adapting coefficients to the existing binding."""
    return dict(
        geometry=geometry,
        centers=head["centers"],
        coefficients=[head["intercept"], *head["weights"]],
        optimizer_step=head["optimizer_step"],
        version=1,
        head_hash=canonical_hash(head),
    )


def design(state: Json, values: Any, native: Any, arm: str) -> Any:
    """Identical frozen inputs cross either the real binding or NumPy equations."""
    if arm.startswith("native"):
        model = native.RustRadial8105(json.dumps(state))
        rows = values.tolist()
        return np.asarray(
            [model.design([r])[0] for r in rows] if arm.endswith("scalar") else model.design(rows)
        )
    mean, std, centers, _, sigma = prior.old.prepare(state)

    def batch(rows: Any) -> Any:
        z = (np.asarray(rows) - mean) / std
        return np.column_stack(
            (
                np.ones(len(z)),
                np.exp(-0.5 * np.sum(((z[:, None, :] - centers[None, :, :]) / sigma) ** 2, axis=2)),
            )
        )

    return np.asarray([batch([r])[0] for r in values]) if arm.endswith("scalar") else batch(values)


def score(head: Json, geometry: Json, values: Any, native: Any, arm: str) -> Any:
    """Add the frozen historical offset once; it is not a current generation call."""
    x = np.asarray(values, dtype=float)
    state = state_for(head, geometry)
    return engine.expit(x[:, 0] + design(state, x, native, arm) @ np.asarray(state["coefficients"]))


def finish(raw: Path, state: Json, native: Any, arm: str) -> Json:
    """A query or update commits a recoverable state before returning to its caller."""
    durable = (
        json.loads(native.RustRadial8105(json.dumps(state)).state_json())
        if arm.startswith("native")
        else state
    )
    path = raw / "state.json"
    atomic_json(path, dict(state=durable, sha256=canonical_hash(durable)))
    directory = os.open(raw, os.O_RDONLY)
    os.fsync(directory)
    os.close(directory)
    return dict(durable_state=durable, state_path=str(path), state_sha256=sha256_file(path))


def transaction(data: Json, native: Any, slot: Json, arm: str, raw: Path) -> Json:
    """Time ingress to durable return on an original natural cached-feature panel."""
    raw.mkdir(parents=True, exist_ok=True)
    panel = [r for r in data["public"] if r["values"] is not None]
    rows = [panel[(slot["repetition"] + i) % len(panel)] for i in range(slot["batch"])]
    state = state_for(data["head"], data["geometry"])
    costs: Json = {}
    began = step = time.perf_counter_ns()
    cache: Json = {}
    if slot["condition"] in ("warm", "changed-content", "eviction", "restart"):
        cache = {canonical_hash(r): json.dumps(r["values"]) for r in rows}
    if slot["condition"] == "changed-content":
        # Another original natural event replaces this request's cached content.
        cache = {
            canonical_hash(dict(r, values=panel[(i + 1) % len(panel)]["values"])): json.dumps(
                r["values"]
            )
            for i, r in enumerate(rows)
        }
    if slot["condition"] in ("miss", "eviction"):
        cache.clear()
        cache[canonical_hash(panel[-1])] = json.dumps(panel[-1]["values"])
    if slot["condition"] == "restart":
        atomic_json(raw / "cache.json", cache)
        durable = finish(raw, state, native, arm)
        cache = json.loads((raw / "cache.json").read_text())
        state = prior.old.host.read_state(Path(durable["state_path"]))
    costs["lifecycle_ns"] = time.perf_counter_ns() - step
    step = time.perf_counter_ns()
    keys, vectors, events = [], [], []
    for row in rows:
        key = canonical_hash(row)
        hit = key in cache
        encoded = cache.get(key, json.dumps(row["values"]))
        vectors.append(json.loads(encoded))
        cache[key] = encoded
        if slot["condition"] == "eviction" and len(cache) > 1:
            cache.pop(next(iter(cache)))
        keys.append(key)
        events.append(dict(key=key, status="hit" if hit else "miss"))
    costs["hash_lookup_feature_preparation_ns"] = time.perf_counter_ns() - step
    step = time.perf_counter_ns()
    probabilities = score(data["head"], data["geometry"], vectors, native, arm)
    costs["arithmetic_and_boundary_ns"] = time.perf_counter_ns() - step
    step = time.perf_counter_ns()
    actions = [engine.historical.radial.action(float(p)) for p in probabilities]
    state["commit_hash"] = canonical_hash(dict(keys=keys, actions=actions))
    durable = finish(raw, state, native, arm)
    costs["decision_serialization_fsync_ns"] = time.perf_counter_ns() - step
    elapsed = time.perf_counter_ns() - began
    return dict(
        arm=arm,
        probabilities=probabilities.tolist(),
        actions=actions,
        values=vectors,
        judgment_keys=keys,
        source_cluster_ids=[r["source_cluster_id"] for r in rows],
        rendered_hash=canonical_hash(rows),
        cache_events=events,
        full_latency_ns=elapsed,
        arithmetic_ns=costs["arithmetic_and_boundary_ns"],
        components=costs,
        residual_ns=elapsed - sum(costs.values()),
        **durable,
    )


def fit(head: Json, geometry: Json, pool: list[Json], native: Any, arm: str) -> Json:
    """Recompute the four original SGD steps and charge Gaussian native crossings."""
    result = deepcopy(head)
    x = np.asarray([r["values"] for r in pool])
    phi = design(state_for(head, geometry), x, native, arm)
    theta = np.array([head["intercept"], *head["weights"]])
    for _ in range(4):
        errors = engine.expit(x[:, 0] + phi @ theta) - [r["y"] for r in pool]
        gradient = phi.T @ errors / len(pool)
        gradient[1:] += 0.01 * theta[1:]
        gradient /= max(1.0, float(np.linalg.norm(gradient)))
        theta -= 0.05 * gradient
        theta[1:] = np.clip(theta[1:], -4, 4)
    result.update(
        intercept=float(theta[0]),
        weights=theta[1:].tolist(),
        optimizer_step=head["optimizer_step"] + 4,
    )
    return result


def update_transaction(data: Json, native: Any, update: Json, arm: str, raw: Path) -> Json:
    """Recompute an original candidate, admission and durable installed head."""
    raw.mkdir(parents=True, exist_ok=True)
    began = step = time.perf_counter_ns()
    before = deepcopy(update["before"])
    key = canonical_hash(update)
    costs: Json = dict(hash_lookup_feature_preparation_ns=time.perf_counter_ns() - step)
    step = time.perf_counter_ns()
    for candidate in before["candidates"].values():
        fitted = fit(candidate["base"], before["geometry"], before["pool"][-64:], native, arm)
        if not np.allclose(
            [fitted["intercept"], *fitted["weights"]],
            [candidate["head"]["intercept"], *candidate["head"]["weights"]],
            atol=1e-10,
            rtol=0,
        ):
            raise ValueError("candidate_fit_parity")
    costs["candidate_fit_ns"] = time.perf_counter_ns() - step
    step = time.perf_counter_ns()
    records = deepcopy(update["records"])
    values = [r["values"] for r in records]
    for name, candidate in before["candidates"].items():
        for scale in (1.0, 0.5, 0.25, 0.125):
            probabilities = score(
                engine.interpolate(candidate, scale), before["geometry"], values, native, arm
            )
            for record, probability in zip(records, probabilities, strict=True):
                record["grid"][name][str(scale)] = float(probability)
        candidate["labels"] = records
    engine.admit(before, update["event"]["slot"])
    steps = before["events"][-2]["steps"]
    if steps != update["event"]["steps"]:
        raise ValueError("admission_parity")
    probabilities = score(before["arms"]["error_center"], before["geometry"], values, native, arm)
    actions = [engine.historical.radial.action(float(p)) for p in probabilities]
    costs["arithmetic_and_boundary_ns"] = time.perf_counter_ns() - step
    step = time.perf_counter_ns()
    state = state_for(before["arms"]["error_center"], before["geometry"])
    state.update(event_key=update["event_key"], steps=steps, transaction_input_hash=key)
    durable = finish(raw, state, native, arm)
    costs["decision_serialization_fsync_ns"] = time.perf_counter_ns() - step
    elapsed = time.perf_counter_ns() - began
    return dict(
        arm=arm,
        probabilities=probabilities.tolist(),
        actions=actions,
        values=values,
        judgment_keys=[key],
        source_cluster_ids=[r["source_cluster_id"] for r in records],
        rendered_hash=canonical_hash(update["records"]),
        full_latency_ns=elapsed,
        arithmetic_ns=costs["arithmetic_and_boundary_ns"],
        components=costs,
        residual_ns=elapsed - sum(costs.values()),
        steps=steps,
        natural_disposition="accepted" if steps["error_center"] else "rejected",
        **durable,
    )


def recovery_fixture(data: Json, native: Any, raw: Path) -> Json:
    """Corrupt a private checkpoint to test detection without inventing natural failures."""
    began = time.perf_counter_ns()
    state = state_for(data["head"], data["geometry"])
    model = native.RustRadial8105(json.dumps(state))
    encoded = json.loads(model.checkpoint())
    encoded["sha256"] = "injected_corruption"
    detected = False
    try:
        native.RustRadial8105.restore(json.dumps(encoded))
    except ValueError:
        detected = True
    restored = native.RustRadial8105.restore(model.checkpoint())
    durable = finish(raw, json.loads(restored.state_json()), native, "native_batch")
    return dict(
        verdict_class="circular_positive",
        fixture=True,
        condition="corrupt_checkpoint_recovery",
        detected=detected,
        recovered=json.loads(restored.state_json()) == state,
        duration_ns=time.perf_counter_ns() - began,
        **durable,
    )


def measure(data: Json, native: Any, raw: Path, config: Json = CONFIG) -> Json:
    """Keep every planned pair, including deadline censoring and excluded warmups."""
    plan = prior.schedule(config)
    work: Json = dict(pairs=[], updates=[], warmups=[], recovery=[], config=config)
    began = time.monotonic()
    warmed: set[tuple[int, str]] = set()
    raw.mkdir(parents=True, exist_ok=True)
    atomic_json(raw / "input_data.json", data)
    with (raw / "transcript.jsonl").open("w") as stream:
        for index, slot in enumerate(plan):
            progress("benchmark_before_" + slot["unit_id"], index, len(plan) - index)
            if time.monotonic() - began >= config["measurement_ceiling_s"]:
                pair = dict(
                    slot, arms=[], status="censored", exclusion_reason="measurement_ceiling_2400s"
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
                                raw / "warmup" / slot["unit_id"] / f"{warmup}-{arm}",
                            )
                            work["warmups"].append(
                                dict(
                                    cell=cell,
                                    arm=arm,
                                    duration_ns=row["full_latency_ns"],
                                    excluded=True,
                                )
                            )
                    warmed.add(cell)
                arms = [
                    transaction(data, native, slot, arm, raw / "query" / slot["unit_id"] / arm)
                    for arm in slot["order"]
                ]
                pair = dict(slot, arms=arms, status="completed", exclusion_reason=None)
            work["pairs"].append(pair)
            stream.write(json.dumps(dict(kind="query", **pair)) + "\n")
            stream.flush()
            progress("benchmark_after_" + slot["unit_id"], index + 1, len(plan) - index - 1)
        for update in data["updates"]:
            for rep in range(config["warmups"] + config["repetitions"]):
                progress(
                    "update_benchmark_before", rep, config["warmups"] + config["repetitions"] - rep
                )
                slot = dict(
                    unit_id=f"{update['event_key']}-r{rep}",
                    batch=12,
                    condition="accepted_update"
                    if update["event"]["steps"]["error_center"]
                    else "rejected_update",
                    repetition=rep,
                    order=list(prior.ARMS if rep % 2 == 0 else reversed(prior.ARMS)),
                )
                expired = time.monotonic() - began >= config["measurement_ceiling_s"]
                arms = (
                    []
                    if expired
                    else [
                        update_transaction(
                            data, native, update, arm, raw / "update" / slot["unit_id"] / arm
                        )
                        for arm in slot["order"]
                    ]
                )
                pair = dict(
                    slot,
                    arms=arms,
                    status="censored" if expired else "completed",
                    exclusion_reason="measurement_ceiling_2400s" if expired else None,
                )
                if rep < config["warmups"]:
                    work["warmups"].append(
                        dict(unit_id=slot["unit_id"], excluded=True, status=pair["status"])
                    )
                else:
                    work["updates"].append(pair)
                stream.write(json.dumps(dict(kind="update", **pair)) + "\n")
                stream.flush()
                progress(
                    "update_benchmark_after",
                    rep + 1,
                    config["warmups"] + config["repetitions"] - rep - 1,
                )
    work["recovery"] = [recovery_fixture(data, native, raw / "recovery_fixture")]
    work["measurement_duration_s"] = time.monotonic() - began
    atomic_json(raw / "primitive_rows.json", work)
    return work


def reduce_rows(work: Json) -> Json:
    """Qualified paired batch reduction is reused; natural admissions have their own rows."""
    result = prior.reduce_rows(work, work["config"])
    for interval in result["paired_intervals"]:
        interval["nfr01_met"] = interval["one_sided_95_lower"] >= 10
    result["update_pairs_passed"] = all(
        p["status"] == "completed" and prior.parity(p["arms"]) for p in work["updates"]
    )
    result["recovery_passed"] = all(r["detected"] and r["recovered"] for r in work["recovery"])
    result["passed"] = (
        result["passed"] and result["update_pairs_passed"] and result["recovery_passed"]
    )
    return result


def checksum(value: Json) -> str:
    """Exclude only the checksum itself so changing a readiness flag invalidates custody."""
    return canonical_hash({k: v for k, v in value.items() if k != "reproducibility_checksum"})


def build(
    data: Json,
    work: Json,
    raw: Path,
    receipts: list[Json],
    date: str,
    duration: float,
    fixture: bool,
) -> Json:
    """Natural query readiness cannot fill a missing rejected-update denominator."""
    reduced = (
        reduce_rows(work)
        if data["ready"]
        else dict(
            passed=True,
            intended_count=0,
            completed_count=0,
            failed_count=0,
            censored_count=0,
            paired_intervals=[],
        )
    )
    checked = all(r["passed"] for r in receipts)
    ready = bool(data["ready"] and reduced["passed"] and checked and not fixture)
    verdict = "null" if data["ready"] else "blocked"
    honest = (
        "complete_null_natural_host_cost_measured_rejection_unavailable"
        if data["ready"]
        else "complete_blocked_" + next(r["check"] for r in data["checks"] if not r["passed"])
    )
    if fixture and data["ready"]:
        verdict, honest = (
            "circular_positive",
            "complete_circular_positive_private_natural_host_fixture",
        )
    if not checked or not reduced["passed"]:
        verdict, honest = "disqualified", "complete_disqualified_owned_validation"
    rows, components = [], []
    for pair in work["pairs"] + work["updates"]:
        for arm in pair["arms"] or [dict(arm="all", full_latency_ns=None, source_cluster_ids=[])]:
            rows.append(
                dict(
                    unit_id=pair["unit_id"],
                    source_cluster_id=canonical_hash(arm["source_cluster_ids"]),
                    arm=arm["arm"],
                    condition=pair["condition"],
                    metric="host_latency_ns",
                    numerator=arm["full_latency_ns"],
                    denominator=pair["batch"],
                    status=pair["status"],
                    exclusion_reason=pair["exclusion_reason"],
                )
            )
            components.extend(
                dict(
                    unit_id=pair["unit_id"],
                    arm=arm["arm"],
                    condition=pair["condition"],
                    component=k,
                    duration_ns=v,
                )
                for k, v in sorted(arm.get("components", {}).items())
            )
    missing = [
        dict(
            r,
            arm="all",
            condition="original_source_mask",
            metric="host_latency_ns",
            numerator=None,
            denominator=1,
        )
        for r in data["public"]
        if r["values"] is None
    ]
    paths = [
        *OWNED,
        TEST,
        "python/carnot/experiment_8132_v703_service_cost.py",
        "python/carnot/verify/learning_protocol_8138.py",
        "python/carnot/reporting/primary_publication.py",
        "crates/carnot-python/src/radial_8105.rs",
        "Cargo.lock",
    ]
    code = {p: sha256_file(ROOT / p) for p in paths}
    value: Json = dict(
        experiment_id=8145,
        task_id="exp8145-natural-service-cost",
        schema="carnot.natural_service_cost.v1",
        honest_verdict=honest,
        verdict_class=verdict,
        verifier_is_oracle=fixture,
        claim_scope="Cached natural seed101 error-center head/query and original accepted admission costs; no current acquisition, live lexical extraction or external generalization",
        exposure_scope="exposed_historical_development",
        independent_generalization_score=0,
        generalized_learning_benefit_score=0,
        required_checks_passed=checked and reduced["passed"],
        flagged_adversarial=False,
        validation_receipts=receipts,
        terminal_validation_sidecar_path=str(raw / "terminal_validation.json"),
        gate_check_summary=data["checks"],
        inference_substrate="verifier_ensemble_against_cached_candidates"
        if data["ready"]
        else "aggregation_from_upstream_artifacts",
        inference_substrate_class="no_model_load",
        MODEL_SPECS=MODEL_SPECS,
        model_invocation_counts=ZERO_INVOCATION_COUNTS,
        call_ledger=[],
        trained_head_specs=data.get("trained_head_specs", []),
        rows=rows + missing,
        intended_count=reduced["intended_count"],
        eligible_count=reduced["intended_count"],
        independent_count=len(
            {r["source_cluster_id"] for r in data["public"] if r["values"] is not None}
        ),
        completed_count=reduced["completed_count"],
        excluded_count=0,
        censored_count=reduced["censored_count"],
        failed_count=reduced["failed_count"],
        sample_size_budget=dict(
            original_slots=192,
            original_missing=len(missing),
            batches=work["config"]["batches"],
            repeats=work["config"]["repetitions"],
            warmups=work["config"]["warmups"],
            independent_sources_are_not_repeats=True,
        ),
        run_date=date,
        duration_s=duration,
        random_seed=CONFIG["seed"],
        source_artifact_hashes=data["refs"],
        raw_shard_hashes=[
            reference(p)
            for p in [*raw.glob("*.json"), *raw.glob("*.jsonl")]
            if p.name != "terminal_validation.json"
        ],
        code_config_hashes=code,
        measurement_config=work["config"],
        phase_spans=[dict(phase="measurement", duration_s=work.get("measurement_duration_s", 0))],
        acceptance_gates=dict(
            upstream="Exp8143.learning_trajectory_ready_score==1",
            parity="1e-10 and identical typed decisions",
            speed="10000 paired batch log bootstraps; lower95>1; NFR-01>=10",
            update="Both original accepted and rejected admissions required",
        ),
        cited_upstream_artifacts=data.get("historical_model_provenance", []),
        natural_service_ready_score=int(ready),
        natural_update_cost_ready_score=int(ready and all(data["categories"].values())),
        service_rows=[r for r in rows if "update" not in r["condition"]],
        update_cost_rows=[r for r in rows if "update" in r["condition"]],
        final_head_manifest=data.get("heads_ref", {}),
        acquisition_prompt_manifest=dict(
            status="unavailable_current_acquisition", historical_upstream=UPSTREAM
        ),
        paired_speed_intervals=reduced["paired_intervals"],
        component_cost_rows=components,
        rejected_update_cost=None if not data["categories"]["rejected"] else "see_update_cost_rows",
        natural_update_categories=data["categories"],
        recovery_fixture_rows=work["recovery"],
        primitive_rows=reference(raw / "primitive_rows.json"),
        reduction=reduced,
        fixture_mode=fixture,
        methodology_note="Frozen natural vectors; timing repeats do not create independent sources. Lifecycle and corruption injections are fixtures. Loaded Rust design plus host offset; candidate generation/training and durable admission are charged. Historical acquisition is not measured current full service.",
    )
    value["field_principles"] = {
        k: "Exact bytes and original event keys bind conditional cached host costs; missing natural categories stay unavailable and fixtures grant no generalization."
        for k in value
    }
    value["reproducibility_checksum"] = checksum(value)
    return value


def replay(path: Path) -> bool:
    """Rehash evidence and independently evaluate every returned typed decision."""
    try:
        value = json.loads(path.read_text())
        if value["reproducibility_checksum"] != checksum(value):
            return False
        for ref in value["source_artifact_hashes"] + value["raw_shard_hashes"]:
            if sha256_file(Path(ref["path"])) != ref["sha256"]:
                return False
        for label, digest in value["code_config_hashes"].items():
            if sha256_file(ROOT / label) != digest:
                return False
        raw = Path(value["primitive_rows"]["path"]).parent
        data = json.loads((raw / "input_data.json").read_text())
        work = json.loads((raw / "primitive_rows.json").read_text())
        rebuilt = build(
            data,
            work,
            raw,
            value["validation_receipts"],
            value["run_date"],
            value["duration_s"],
            value["fixture_mode"],
        )
        for field in (
            "rows",
            "service_rows",
            "update_cost_rows",
            "component_cost_rows",
            "reduction",
            "paired_speed_intervals",
            "natural_service_ready_score",
            "natural_update_cost_ready_score",
            "natural_update_categories",
            "honest_verdict",
            "verdict_class",
            "required_checks_passed",
        ):
            if rebuilt[field] != value[field]:
                return False
        checked: Json = {}
        for pair in work["pairs"] + work["updates"]:
            for arm in pair["arms"]:
                state_path = Path(arm["state_path"])
                if sha256_file(state_path) != arm["state_sha256"]:
                    return False
                state = prior.old.host.read_state(state_path)
                if state != arm["durable_state"]:
                    return False
                head = dict(
                    centers=state["centers"],
                    intercept=state["coefficients"][0],
                    weights=state["coefficients"][1:],
                )
                probabilities = []
                for vector in arm["values"]:
                    key = canonical_hash(dict(head=head, geometry=state["geometry"], vector=vector))
                    if key not in checked:
                        checked[key] = engine.scalar_probability(head, state["geometry"], vector)
                    probabilities.append(checked[key])
                if not np.allclose(probabilities, arm["probabilities"], atol=1e-10, rtol=0) or arm[
                    "actions"
                ] != [engine.historical.radial.action(p) for p in probabilities]:
                    return False
            progress("cold_pair_complete", pair["repetition"] + 1, 0)
        return True
    except (OSError, ValueError, KeyError, TypeError):
        return False
