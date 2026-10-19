"""REQ-REPORT-7989: measure source-aware service without fresh model activity.

Only source groups supply independent observations. Historical acquisition
receipts retain their own dates and cannot become current model calls.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import resource
import subprocess
import time
from typing import Any

import numpy as np
from scipy.stats import t  # type: ignore[import-untyped]

from carnot import experiment_7982_v692_multivariate_energy as fitted
from carnot.reporting import service_cost_7976 as previous
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash, sha256_file
from carnot.reporting.evidence_features_custody_7980 import checked, operand, reference
from carnot.verify import evidence_features_7980 as features
from carnot.verify import multivariate_energy_7982 as multi
from carnot.verify import qwen_energy_calibration_7972 as scalar

Json = dict[str, Any]
ROOT = previous.ROOT
PINS = {
    7982: (
        "experiment_7982_v692_multivariate_energy.json",
        "energy_fit_ready_score",
        "sha256:9daef912592c6c17e9b2db46c4059e7be99adcb3d76be897169e1e0afc238d54",
    ),
    7985: (
        "experiment_7985_delayed_acquisition.json",
        "learning_measurement_ready_score",
        "sha256:53a1a2b52991fa33f67576c3031be54fc3633b31e990ca02cca881413714f2d5",
    ),
    7981: (
        "experiment_7981_v692_qwen_stream_capture.json",
        "stream_capture_ready_score",
        "sha256:e4ff5097dc4d2db6f8d1d533943f8f7216f069da1c1fdd404a229c2c56166dda",
    ),
    7972: (previous.PINS[7972][0], previous.PINS[7972][2], previous.PINS[7972][1]),
}
PHASES = (
    "public_byte_read",
    "source_feature_extraction",
    "normalization",
    "scoring",
    "typed_decision",
    "serialization",
    "storage_fsync",
)


def authenticate(root: Path) -> Json:
    """Qualify alternatives separately; a null fit still supplies useful timing."""
    plan: Json = dict(
        branch_readiness={},
        branch_gate_check_summary={},
        requests=[],
        heads={},
        source_artifact_hashes=[],
        upstream={},
        acquisition_setup={},
    )
    values, gates = {}, {}
    for eid, (name, field, pin) in PINS.items():
        path = root / "results" / name
        ready, checks, value = previous.branch_gate(path, eid, field, pin)
        gates[eid], values[eid] = ready, value
        plan["upstream"][str(eid)] = value
        if path.is_file():
            plan["source_artifact_hashes"].append(
                dict(
                    reference(path),
                    producer_id=eid,
                    producer_invocation_date=value.get("run_date"),
                    producer_invocation_timestamp=value.get(
                        "invocation_timestamp", value.get("started_at")
                    ),
                    role="branch_authority",
                )
            )
        plan["branch_gate_check_summary"][str(eid)] = checks
    plan["branch_readiness"] = dict(
        multivariate=int(gates[7982]),
        durable_learning=int(gates[7985]),
        scalar_current=int(gates[7972] and gates[7981]),
    )
    plan["branch_gate_check_summary"] = dict(
        multivariate=plan["branch_gate_check_summary"]["7982"],
        durable_learning=plan["branch_gate_check_summary"]["7985"],
        scalar_current=plan["branch_gate_check_summary"]["7972"]
        + plan["branch_gate_check_summary"]["7981"],
    )
    if gates[7982]:
        try:
            value = values[7982]
            fitted.replay(value)
            failures, parents = fitted.authenticate(root)
            if failures:
                plan["branch_gate_check_summary"]["multivariate"].extend(failures)
                raise ValueError(json.dumps(failures, sort_keys=True))
            plan["source_artifact_hashes"] += parents["refs"]
            capture = parents["upstream"][7969]
            if capture["model_identity_receipt"]["authenticated"] is not True:
                raise ValueError("gpu_identity")
            bundle = json.loads(checked(value["heads_seal"]).read_text())
            control = json.loads(checked(value["frozen_scalar_controls"]).read_text())
            scalar_heads = json.loads(checked(control["heads"]).read_text())["heads"]["gibbs"]
            plan["heads"] = dict(
                heads={**bundle["heads"], "scalar": scalar_heads},
                normalization=bundle["feature_normalization"],
            )
            refs = list(value["checkpoints"].values()) + [
                value["frozen_scalar_controls"],
                control["heads"],
            ]
            plan["source_artifact_hashes"] += refs
            data = json.loads(checked(value["checkpoints"]["inputs"]).read_text())["data"][
                "policy_design"
            ]
            public_ref = parents["upstream"][7980]["public_role_manifests"]["policy_design"]
            public = {
                r["family_id"]: r
                for r in json.loads(checked(public_ref).read_text())["request_rows"]
            }
            costs = {r["family_id"]: r for r in capture["rows"]}
            for row in data[:64]:
                fid = row["family_id"]
                extracted = features.extract(public[fid])
                cost = costs[fid]
                if (
                    extracted["values"] != row["features"]
                    or cost["parsed"]["probability"] != row["q"]
                    or cost["role"] != "policy_design"
                ):
                    raise ValueError("public_join_drift")
                plan["requests"].append(
                    dict(
                        family_id=fid,
                        source_cluster_id=row["source_cluster_id"],
                        role="policy_design",
                        public=public[fid],
                        q=row["q"],
                        status=row["status"],
                        original_model_s=cost["duration_s"] if cost["started"] else None,
                        acquisition_producer=7969,
                        acquisition_reference=next(
                            r
                            for r in capture["raw_response_shards"]
                            if json.loads(checked(r).read_text())["family_id"] == fid
                        ),
                    )
                )
            completed = sum(r["status"] == "generated" for r in capture["rows"])
            setup = capture["model_identity_receipt"]["duration_s"]
            plan["acquisition_setup"] = dict(
                historical=dict(
                    producer_id=7969,
                    producer_invocation_date=capture["run_date"],
                    model_load_s=setup,
                    actual_completed_requests=completed,
                    amortized_per_completed_request_s=setup / completed,
                    download_s=None,
                ),
                current=dict(
                    model_load_s=None,
                    model_loads=0,
                    completed_model_requests=0,
                    amortized_per_completed_request_s=None,
                ),
            )
        except (OSError, ValueError, KeyError, TypeError) as error:
            plan["branch_readiness"]["multivariate"] = 0
            plan["requests"], plan["heads"] = [], {}
            plan["branch_gate_check_summary"]["multivariate"].append(
                operand(
                    7982,
                    root / "results" / PINS[7982][0],
                    "checkpoint_replay",
                    "passed",
                    str(error),
                )
            )
    return plan


def fixture() -> tuple[list[Json], Json]:
    """Small public controls test service plumbing without scientific claims."""
    inputs = [
        dict(
            family_id=f"f{i}",
            source_cluster_id=f"c{i}",
            role="fixture",
            q=q,
            status="fixture",
            original_model_s=None if i == 3 else 1.0,
            public=dict(
                family_id=f"f{i}",
                source_bytes=b"The cat is black.".hex(),
                answer_bytes=b"The cat is black.".hex(),
            ),
        )
        for i, q in enumerate((0.0, 0.5, 1.0, None))
    ]
    heads = dict(
        heads=dict(
            gibbs=[dict(arm="gibbs", parameters=[0.1] * 97, temperature=1.0)],
            scalar=previous.fixture()[1],
        ),
        normalization=dict(means=[0.0] * 9, scales=[1.0] * 9),
    )
    return inputs, heads


def request(path: Path, heads_path: Path, output: Path, arm: str, storage: str) -> Json:
    """Adjacent boundaries include real byte handling and durable response writes."""
    ticks = [time.perf_counter_ns()]
    raw, coefficients = path.read_bytes(), heads_path.read_bytes()
    value, bundle = json.loads(raw), json.loads(coefficients)
    ticks.append(time.perf_counter_ns())
    extracted = features.extract(value["public"]) if arm != "scalar" else None
    ticks.append(time.perf_counter_ns())
    available = value["q"] is not None and (arm == "scalar" or extracted["values"] is not None)
    x = np.empty((0, 9))
    if available and arm != "scalar":
        x = (
            multi.inputs([dict(q=value["q"], features=extracted["values"])])
            - np.asarray(bundle["normalization"]["means"])
        ) / np.asarray(bundle["normalization"]["scales"])
    ticks.append(time.perf_counter_ns())
    heads = bundle["heads"][arm]
    p = None
    if available:
        q = np.array([value["q"]], dtype=float)
        p = float(
            np.mean(
                [scalar.predict(h, q) if arm == "scalar" else multi.predict(h, x) for h in heads]
            )
        )
    ticks.append(time.perf_counter_ns())
    response = dict(
        family_id=value["family_id"],
        probability=p,
        action=scalar.decision(p),
        verified=False,
        available=available,
    )
    ticks.append(time.perf_counter_ns())
    encoded = json.dumps(response, sort_keys=True).encode()
    ticks.append(time.perf_counter_ns())
    if storage == "fsync":
        with output.open("wb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        directory = os.open(output.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    ticks.append(time.perf_counter_ns())
    return dict(
        family_id=value["family_id"],
        source_cluster_id=value["source_cluster_id"],
        role=value["role"],
        status=value["status"],
        arm=arm,
        storage=storage,
        response=response,
        ticks_ns=ticks,
        wall_ns=ticks[-1] - ticks[0],
        exclusive_phase_spans=dict(zip(PHASES, np.diff(ticks).tolist(), strict=True)),
        bytes_read=len(raw) + len(coefficients),
        bytes_written=len(encoded) if storage == "fsync" else 0,
        coefficient_touches=sum(len(h["parameters"]) for h in heads) if available else 0,
        rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        request_sha256=canonical_hash(value),
        original_model_s=value["original_model_s"],
    )


def reduce(rows: list[Json]) -> Json:
    """Source means, rather than repeated timings, supply the paired interval."""
    summaries, complete, comparisons = [], [], []
    arms = sorted({r["arm"] for r in rows})
    for arm in arms:
        for storage in ("fsync", "no_write"):
            selected = [r for r in rows if r["arm"] == arm and r["storage"] == storage]
            times = np.array([r["wall_ns"] / 1e9 for r in selected])
            summaries.append(
                dict(
                    arm=arm,
                    storage=storage,
                    p50_s=float(np.quantile(times, 0.5)),
                    p95_s=float(np.quantile(times, 0.95)),
                    throughput_requests_s=float(1 / times.mean()),
                    rss_peak_bytes=max(r["rss_bytes"] for r in selected),
                    coefficient_touches=sum(r["coefficient_touches"] for r in selected),
                    bytes_read=sum(r["bytes_read"] for r in selected),
                    bytes_written=sum(r["bytes_written"] for r in selected),
                    independent=len({r["source_cluster_id"] for r in selected}),
                    timing_observations=len(selected),
                )
            )
            if storage == "fsync":
                known = [
                    r["original_model_s"] + r["wall_ns"] / 1e9
                    for r in selected
                    if r["original_model_s"] is not None
                ]
                complete.append(
                    dict(
                        arm=arm,
                        p50_s=float(np.quantile(known, 0.5)) if known else None,
                        p95_s=float(np.quantile(known, 0.95)) if known else None,
                        throughput_requests_s=len(known) / sum(known) if known else None,
                        known_count=len(
                            {r["family_id"] for r in selected if r["original_model_s"] is not None}
                        ),
                        unknown_count=len(
                            {r["family_id"] for r in selected if r["original_model_s"] is None}
                        ),
                        boundary="matching authenticated acquisition plus CPU fsync service; setup separate",
                        fresh_inference_speedup_claim=False,
                    )
                )
    for arm in arms:
        for baseline, storage in (("scalar", "fsync"), (arm, "no_write")):
            by_source: Json = {}
            for row in rows:
                key = row["source_cluster_id"]
                if (row["arm"], row["storage"]) == (arm, "fsync"):
                    by_source.setdefault(key, [[], []])[0].append(row["wall_ns"] / 1e9)
                if (row["arm"], row["storage"]) == (baseline, storage):
                    by_source.setdefault(key, [[], []])[1].append(row["wall_ns"] / 1e9)
            delta = np.array([np.mean(a) - np.mean(b) for a, b in by_source.values()])
            margin = (
                float(t.ppf(0.975, len(delta) - 1) * np.std(delta, ddof=1) / np.sqrt(len(delta)))
                if len(delta) > 1
                else None
            )
            mean = float(delta.mean())
            comparisons.append(
                dict(
                    arm=arm,
                    baseline_arm=baseline,
                    baseline_storage=storage,
                    mean_overhead_s=mean,
                    paired_95_interval_s=[mean - margin, mean + margin]
                    if margin is not None
                    else None,
                    independent=len(delta),
                    method="Student t interval on paired source-group means",
                    claim="engineering_overhead_only",
                )
            )
    return dict(
        service_summary=summaries,
        complete_service_cost=complete,
        paired_cpu_comparisons=comparisons,
    )


def replay(value: Json) -> None:
    """Authenticate inputs and recompute decisions and accounting without timing."""
    for item in value.get("source_artifact_hashes", []) + value.get("code_config_hashes", []):
        checked(item)
    checkpoint = value.get("input_checkpoint")
    inputs = json.loads(checked(checkpoint).read_text()) if checkpoint else value["replay_inputs"]
    pairs: Json = {}
    decisions: Json = {}
    expected = {r["family_id"]: r for r in inputs["requests"]}
    for index, row in enumerate(value["rows"]):
        if index % 128 == 0:
            print(f"[exp7989] cold_replay_row={index}/{len(value['rows'])}", flush=True)
        item = expected[row["family_id"]]
        heads = inputs["heads"]
        arm = row["arm"]
        decision_key = f"{row['family_id']}:{arm}"
        f = (
            features.extract(item["public"])
            if arm != "scalar" and decision_key not in decisions
            else None
        )
        p = None
        if (
            decision_key not in decisions
            and item["q"] is not None
            and (arm == "scalar" or f["values"] is not None)
        ):
            q = np.array([item["q"]], dtype=float)
            x = (
                (
                    multi.inputs([dict(q=item["q"], features=f["values"])])
                    - np.asarray(heads["normalization"]["means"])
                )
                / np.asarray(heads["normalization"]["scales"])
                if arm != "scalar"
                else np.empty((0, 9))
            )
            p = float(
                np.mean(
                    [
                        scalar.predict(h, q) if arm == "scalar" else multi.predict(h, x)
                        for h in heads["heads"][arm]
                    ]
                )
            )
        expected_response = decisions.setdefault(
            decision_key,
            dict(
                family_id=item["family_id"],
                probability=p,
                action=scalar.decision(p),
                verified=False,
                available=p is not None,
            ),
        )
        if (
            row["request_sha256"] != canonical_hash(item)
            or row["response"] != expected_response
            or row["original_model_s"] != item["original_model_s"]
        ):
            raise ValueError("input_decision_drift")
        spans = dict(zip(PHASES, np.diff(row["ticks_ns"]).tolist(), strict=True))
        if (
            spans != row["exclusive_phase_spans"]
            or any(v < 0 for v in spans.values())
            or sum(spans.values()) != row["wall_ns"]
        ):
            raise ValueError("span_drift")
        key = f"{row['family_id']}:{arm}:{row['repetition']}"
        pairs.setdefault(key, []).append(row["storage"])
    if any(sorted(modes) != ["fsync", "no_write"] for modes in pairs.values()):
        raise ValueError("paired_rows_drift")
    if any(value[k] != v for k, v in reduce(value["rows"]).items()):
        raise ValueError("reduction_drift")


def measure(
    inputs: list[Json], heads: Json, scratch: Path, *, storage_path: Path | None = None
) -> Json:
    """Ten paired repeats estimate cost variability on a fixed public roster."""
    scratch.mkdir(parents=True, exist_ok=True)
    output = storage_path if storage_path is not None else scratch / "response.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    filesystem_type = subprocess.check_output(
        ["stat", "-f", "-c", "%T", str(output.parent)], text=True
    ).strip()
    for arm, selected in heads["heads"].items():
        atomic_json(
            scratch / f"heads-{arm}.json",
            dict(heads={arm: selected}, normalization=heads["normalization"]),
        )
    paths = []
    for index, item in enumerate(inputs):
        path = scratch / f"request-{index}.json"
        atomic_json(path, item)
        paths.append(path)
    arms = sorted(heads["heads"])
    combinations = [(arm, storage) for arm in arms for storage in ("fsync", "no_write")]
    rows = []
    for repetition in range(-1, 10):
        print(
            f"[exp7989] before_benchmark repetition={repetition} warmup=complete_request_once",
            flush=True,
        )
        for index, path in enumerate(paths):
            order = np.random.default_rng(69289 + max(repetition, 0) * 64 + index).permutation(
                len(combinations)
            )
            for position in order:
                arm, storage = combinations[position]
                row = request(
                    path, scratch / f"heads-{arm}.json", output, arm, storage
                )
                if repetition >= 0:
                    row.update(
                        repetition=repetition,
                        head_seeds=[h.get("seed") for h in heads["heads"][arm]],
                    )
                    rows.append(row)
            if index % 8 == 0:
                print(
                    f"[exp7989] benchmark_source={index + 1}/{len(paths)} completed_timings={len(rows)}",
                    flush=True,
                )
        print(
            f"[exp7989] after_benchmark repetition={repetition} completed_timings={len(rows)}",
            flush=True,
        )
    value = dict(
        rows=rows,
        **reduce(rows),
        exclusive_phase_spans=[r["exclusive_phase_spans"] for r in rows],
        replay_inputs=dict(requests=inputs, heads=heads),
        sample_size_budget=dict(
            unit="original_source_group",
            intended=len(inputs),
            eligible=sum(r["q"] is not None for r in inputs),
            started=len(inputs),
            completed=len(inputs),
            failed=0,
            censored=sum(r["status"] == "censored" for r in inputs),
            excluded=0,
            independent=len({r["source_cluster_id"] for r in inputs}),
            timing_repetitions=10,
            named_warmups=1,
            seeds_are_independent=False,
        ),
        durable_costs=dict(
            online_gradient=None,
            admission=None,
            checkpoint_fsync=None,
            restart=None,
            reason="Exp7985 unqualified; response persistence does not establish durable learning",
        ),
        hardware_compatible_operations=[],
        compatible_fraction=0.0,
        transfer_bytes=0,
        operation_inventory=[
            dict(
                operation=name,
                span=span,
                executed=executed,
                board_compatible=False,
                reason="No authenticated board capability matches this operation",
            )
            for name, span, executed in [
                ("quadratic_Ising", None, False),
                ("dense_two_label_tanh_head", "scoring", "gibbs" in arms),
                (
                    "dense_logistic_quadratic_mlp_heads",
                    "scoring",
                    any(a in arms for a in ("logistic", "quadratic", "mlp")),
                ),
                ("sparse_spline_basis_to_dense_head", "scoring", "spline" in arms),
                (
                    "public_byte_handling_and_lexical_features",
                    "public_byte_read/source_feature_extraction",
                    True,
                ),
                ("response_file_and_directory_fsync", "storage_fsync", True),
            ]
        ],
        environmental_observations=dict(
            clock="perf_counter_ns",
            rss_unit="Linux ru_maxrss KiB converted to bytes",
            filesystem=str(output.parent),
            filesystem_type=filesystem_type,
            filesystem_cache="warm after named full-request sweep",
            cpu_affinity=sorted(os.sched_getaffinity(0)),
            acquisition_is_historical=True,
        ),
        cached_incremental_cost=None,
    )
    replay(value)
    return value
