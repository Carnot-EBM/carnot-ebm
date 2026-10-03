"""REQ-REPORT-7976: measure complete private CPU decisions on frozen inputs.

Cached judgments retain their upstream acquisition times. Repeated timings
measure scheduling variation and do not create new independent source evidence.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
import resource
import time
from typing import Any

import numpy as np
from scipy.special import expit  # type: ignore[import-untyped]
import yaml

from carnot import experiment_7972_v691_qwen_energy_calibration as prior
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash, sha256_file
from carnot.verify import qwen_energy_calibration_7972 as calibration

Json = dict[str, Any]
ROOT = Path(__file__).resolve().parents[3]
PINS = {
    7970: (
        "experiment_7970_energy_fit.json",
        "sha256:966ca46c2c41ef7b5e02e5ab887fb04dad5d0ab27d0cb029323781aacf44cee8",
        "energy_fit_ready_score",
    ),
    7972: (
        "experiment_7972_v691_qwen_energy_calibration.json",
        "sha256:a47d19ca7af500014117ff083249469597cd7e6bc0ff57778098fa39cfa41c63",
        "qwen_calibration_ready_score",
    ),
}
PHASES = (
    "public_byte_read",
    "feature_projection",
    "head_evaluation",
    "energy_normalization",
    "typed_policy",
    "serialization",
    "storage_fsync",
)


def branch_gate(
    path: Path, eid: int, readiness: str, pin: str | None, *, retired: bool = False
) -> tuple[bool, list[Json], Json]:
    """Inspect each branch separately so an unavailable peer cannot qualify it."""
    digest = sha256_file(path) if path.is_file() else None
    value = json.loads(path.read_text()) if digest else {}
    checks = []
    for field, op, expected, observed in (
        ("exists", "==", True, bool(digest)),
        ("sha256", "==", pin, digest),
        ("experiment_id", "==", eid, value.get("experiment_id", value.get("experiment"))),
        (readiness, "==", 1, value.get(readiness)),
        (
            "verdict_class",
            "in",
            ["positive", "circular_positive", "null"],
            value.get("verdict_class"),
        ),
        ("flagged_adversarial", "==", False, value.get("flagged_adversarial")),
        ("retired", "==", False, retired),
    ):
        passed = observed in expected if op == "in" else observed == expected
        checks.append(
            dict(
                upstream_id=f"exp{eid}",
                artifact_path=str(path),
                artifact_hash=digest,
                field=field,
                op=op,
                expected=expected,
                observed=observed,
                passed=passed,
            )
        )
    return all(r["passed"] for r in checks), checks, value


def authenticate(root: Path) -> Json:
    """Bind current heads and historical acquisition to their own producer bytes."""
    manifest = root / "ops/exclusion_manifest.yaml"
    exclusions = yaml.safe_load(manifest.read_text()) if manifest.exists() else {}
    retired = {
        r.get("experiment_id")
        for k in ("retired", "retired_experiments")
        for r in exclusions.get(k, [])
    }
    plan: Json = dict(
        branch_readiness={},
        branch_gate_check_summary={},
        source_artifact_hashes=[],
        requests=[],
        heads={},
        original_model_cost_rows=[],
        upstream={},
    )
    for eid, branch in ((7970, "source_energy"), (7972, "qwen_calibration")):
        name, pin, field = PINS[eid]
        path = root / "results" / name
        ready, checks, value = branch_gate(path, eid, field, pin, retired=eid in retired)
        plan["branch_readiness"][branch] = int(ready)
        plan["branch_gate_check_summary"][branch] = checks
        plan["upstream"][branch] = value
        if path.exists():
            plan["source_artifact_hashes"].append(prior.reference(path))
    learner = root / "results/experiment_7973_v691_causal_acquisition.json"
    ready, checks, _ = branch_gate(
        learner,
        7973,
        "learning_measurement_ready_score",
        sha256_file(learner) if learner.exists() else None,
        retired=7973 in retired,
    )
    plan["branch_readiness"]["durable_learning"] = int(ready)
    plan["branch_gate_check_summary"]["durable_learning"] = checks
    if plan["branch_readiness"]["qwen_calibration"]:
        value = plan["upstream"]["qwen_calibration"]
        try:
            prior.replay(value)
        except (OSError, ValueError, KeyError, TypeError) as error:
            plan["branch_readiness"]["qwen_calibration"] = 0
            plan["branch_gate_check_summary"]["qwen_calibration"].append(
                dict(
                    upstream_id="exp7972",
                    artifact_path=str(root / "results" / PINS[7972][0]),
                    artifact_hash=PINS[7972][1],
                    field="authenticated_checkpoint_replay",
                    op="==",
                    expected="passed",
                    observed=str(error),
                    passed=False,
                )
            )
            return plan
        checkpoints = value["calibrator_checkpoints"]
        data = json.loads(prior.checked_reference(checkpoints["primitives"]).read_text())
        plan["heads"] = json.loads(prior.checked_reference(checkpoints["heads"]).read_text())[
            "heads"
        ]["gibbs"]
        plan["source_artifact_hashes"].extend(checkpoints.values())
        history = json.loads(
            (root / "results/experiment_7958_v690_qwen_response_risk.json").read_text()
        )
        capture = json.loads(
            (root / "results/experiment_7969_v691_qwen_calibration_capture.json").read_text()
        )
        for eid, upstream in ((7958, history), (7969, capture)):
            path = next((root / "results").glob(f"experiment_{eid}_*.json"))
            expected = next(r for r in value["source_artifact_hashes"] if r["path"] == str(path))
            prior.checked_reference(expected)
            plan["source_artifact_hashes"].append(expected)
            for r in upstream["rows"]:
                plan["original_model_cost_rows"].append(
                    dict(
                        producer_id=eid,
                        family_id=r["family_id"],
                        arm=r.get("arm"),
                        role=r.get("role", "evaluation"),
                        acquisition_s=r.get("duration_s"),
                        status=r.get("status"),
                        model_timings=r.get("raw_response", {}).get("timings", {}),
                    )
                )
        by_id = {r["family_id"]: r for r in history["rows"] if r["arm"] == "full_source"}
        for r in data["evaluation"][:64]:
            original = by_id[r["family_id"]]
            plan["requests"].append(
                dict(
                    family_id=r["family_id"],
                    source_cluster_id=r["source_cluster_id"],
                    q=r["q"] if r["y"] is not None else None,
                    status=r["status"],
                    public_request=original["request"],
                    original_model_s=original.get("duration_s"),
                )
            )
        plan["acquisition_setup"] = {
            str(eid): u["model_identity_receipt"].get("duration_s")
            for eid, u in ((7958, history), (7969, capture))
        }
    return plan


def bounds(f: float | None, transfer_fraction: float | None) -> Json:
    """A kernel estimate needs explicit compatibility and transfer operands."""
    if f is None or transfer_fraction is None:
        return dict(ideal_amdahl_bound=None, modeled_100x_bound=None)
    if not 0 <= f <= 1 or not math.isfinite(transfer_fraction) or transfer_fraction < 0:
        raise ValueError("fraction")
    return dict(
        ideal_amdahl_bound=None if f == 1 else 1 / (1 - f),
        modeled_100x_bound=1 / ((1 - f) + f / 100 + transfer_fraction),
    )


def fixture() -> tuple[list[Json], list[Json]]:
    """A small public fixture checks the service contract without model activity."""
    return (
        [
            dict(
                family_id=f"f{i}",
                source_cluster_id=f"c{i}",
                q=q,
                original_model_s=1.0,
                status="fixture",
                public_request={"text": "public"},
            )
            for i, q in enumerate((0.0, 0.5, 1.0, None))
        ],
        [dict(arm="gibbs", parameters=[0.1] * 33, temperature=1.0)],
    )


def request(path: Path, heads_path: Path, output: Path, mode: str) -> Json:
    """Time public reads through fsynced response bytes using adjacent boundaries."""
    ticks = [time.perf_counter_ns()]
    raw, head_bytes = path.read_bytes(), heads_path.read_bytes()
    ticks.append(time.perf_counter_ns())
    value, heads = json.loads(raw), json.loads(head_bytes)
    q = np.array([] if value["q"] is None else [value["q"]], dtype=float)
    ticks.append(time.perf_counter_ns())
    if mode == "reference":
        evaluated = [calibration.predict(h, q) for h in heads]
    else:
        evaluated = (
            [calibration.logits_jacobian("gibbs", np.array(h["parameters"]), q)[0] for h in heads]
            if len(q)
            else []
        )
    ticks.append(time.perf_counter_ns())
    if mode == "reference":
        p = float(np.mean(evaluated)) if len(q) else None
    else:
        p = (
            float(
                np.mean(
                    [expit(z / h["temperature"]) for z, h in zip(evaluated, heads, strict=True)]
                )
            )
            if len(q)
            else None
        )
    ticks.append(time.perf_counter_ns())
    response = dict(
        family_id=value["family_id"],
        probability=p,
        action=calibration.decision(p),
        verified=False,
        available=p is not None,
    )
    ticks.append(time.perf_counter_ns())
    encoded = json.dumps(response, sort_keys=True).encode()
    ticks.append(time.perf_counter_ns())
    with output.open("wb") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())
    ticks.append(time.perf_counter_ns())
    spans = dict(zip(PHASES, np.diff(ticks).tolist(), strict=True))
    return dict(
        family_id=value["family_id"],
        source_cluster_id=value["source_cluster_id"],
        mode=mode,
        response=response,
        ticks_ns=ticks,
        exclusive_phase_spans=spans,
        wall_ns=ticks[-1] - ticks[0],
        bytes_touched=len(raw) + len(head_bytes) + len(encoded),
        rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        request_sha256=canonical_hash(value),
        original_model_s=value["original_model_s"],
    )


def reduce_rows(rows: list[Json]) -> list[Json]:
    """Reduce timing repeats separately from independent source counts."""
    summaries = []
    for mode in sorted({r["mode"] for r in rows}):
        selected = [r for r in rows if r["mode"] == mode]
        ns = np.array([r["wall_ns"] for r in selected], dtype=float)
        summaries.append(
            dict(
                mode=mode,
                p50_s=float(np.quantile(ns, 0.5) / 1e9),
                p95_s=float(np.quantile(ns, 0.95) / 1e9),
                throughput_requests_s=float(1e9 / np.mean(ns)),
                rss_peak_bytes=max(r["rss_bytes"] for r in selected),
                bytes_touched=sum(r["bytes_touched"] for r in selected),
                timing_requests=len(selected),
                independent=len({r["source_cluster_id"] for r in selected}),
            )
        )
    return summaries


def replay(value: Json) -> list[Json]:
    """Cold-check measured boundaries, paired decisions and derived summaries."""
    pairs: Json = {}
    counts: Json = {}
    expected: Json = {}
    if value.get("input_checkpoint"):
        checkpoint = json.loads(prior.checked_reference(value["input_checkpoint"]).read_text())
        for item in checkpoint["requests"]:
            q = np.array([] if item["q"] is None else [item["q"]], dtype=float)
            p = (
                float(np.mean([calibration.predict(h, q) for h in checkpoint["heads"]]))
                if len(q)
                else None
            )
            expected[item["family_id"]] = dict(
                request_sha256=canonical_hash(item),
                original_model_s=item["original_model_s"],
                probability=p,
                action=calibration.decision(p),
            )
    for reference in value.get("source_artifact_hashes", []) + value.get("code_config_hashes", []):
        prior.checked_reference(reference)
    for row in value["service_rows"]:
        if expected:
            bound = expected[row["family_id"]]
            if any(row[k] != bound[k] for k in ("request_sha256", "original_model_s")) or any(
                row["response"][k] != bound[k] for k in ("probability", "action")
            ):
                raise ValueError("input_decision_drift")
        ticks = row["ticks_ns"]
        spans = dict(zip(PHASES, np.diff(ticks).tolist(), strict=True))
        if (
            spans != row["exclusive_phase_spans"]
            or any(v < 0 for v in spans.values())
            or abs(sum(spans.values()) - row["wall_ns"]) > 0.01 * row["wall_ns"]
        ):
            raise ValueError("span_drift")
        key = f"{row['family_id']}:{row['repetition']}"
        if key in pairs and (
            pairs[key]["request_sha256"] != row["request_sha256"]
            or pairs[key]["response"] != row["response"]
        ):
            raise ValueError("decision_drift")
        pairs[key] = row
        counts[key] = counts.get(key, 0) + 1
    if any(n != 2 for n in counts.values()):
        raise ValueError("paired_rows")
    reduced = reduce_rows(value["service_rows"])
    if reduced != value["rows"]:
        raise ValueError("reduction_drift")
    if (
        value["service_rows"]
        and complete_cost(value["service_rows"], value["sample_size_budget"]["intended"])
        != value["complete_service_cost"]
    ):
        raise ValueError("complete_cost_drift")
    return reduced


def complete_cost(rows: list[Json], intended: int) -> Json:
    """Upstream latency remains in complete service even for rejected decisions."""
    known = [
        r["original_model_s"] + r["wall_ns"] / 1e9
        for r in rows
        if r["mode"] == "exclusive" and r["original_model_s"] is not None
    ]
    unique = {r["family_id"] for r in rows if r["original_model_s"] is not None}
    return dict(
        boundary="authenticated upstream acquisition plus current CPU service; resident model setup separate",
        p50_s=float(np.quantile(known, 0.5)) if known else None,
        p95_s=float(np.quantile(known, 0.95)) if known else None,
        known_count=len(unique),
        unknown_count=intended - len(unique),
        throughput_requests_s=float(len(known) / sum(known)) if known else None,
        fresh_inference_speedup_claim=False,
    )


def measure(inputs: list[Json], heads: list[Json], scratch: Path) -> Json:
    """Measure ten paired repetitions after one named full-request warmup."""
    started = time.monotonic()
    scratch.mkdir(parents=True, exist_ok=True)
    # Request reads use the exact same list of coefficients for each paired call.
    (scratch / "heads.json").write_text(json.dumps(heads, sort_keys=True))
    paths = []
    for i, value in enumerate(inputs):
        path = scratch / f"request-{i}.json"
        atomic_json(path, value)
        paths.append(path)
    print("[exp7976] before_benchmark named_warmup=complete_request_once", flush=True)
    for path in paths:
        request(path, scratch / "heads.json", scratch / "response.json", "reference")
    print("[exp7976] after_benchmark named_warmup=complete_request_once", flush=True)
    rows = []
    for repetition in range(10):
        print(
            f"[exp7976] before_benchmark repetition={repetition} elapsed_s={time.monotonic() - started:.3f}",
            flush=True,
        )
        for path in paths:
            modes = ["reference", "exclusive"]
            if int(canonical_hash([path.name, repetition]).split(":")[1], 16) % 2:
                modes.reverse()
            for mode in modes:
                row = request(path, scratch / "heads.json", scratch / "response.json", mode)
                row["repetition"] = repetition
                rows.append(row)
        print(
            f"[exp7976] after_benchmark completed_units={len(rows)} elapsed_s={time.monotonic() - started:.3f}",
            flush=True,
        )
    summaries = reduce_rows(rows)
    value = dict(
        service_rows=rows,
        rows=summaries,
        exclusive_phase_spans=[r["exclusive_phase_spans"] for r in rows],
        cached_incremental_cost=next(r for r in summaries if r["mode"] == "exclusive"),
        complete_service_cost=complete_cost(rows, len(inputs)),
        durable_costs={k: None for k in ("lookup", "update", "commit_fsync", "restart")},
        hardware_compatible_operations=[],
        compatible_fraction=0.0,
        transfer_bytes=0,
        transfer_fraction=0.0,
        **bounds(0, 0),
        sample_size_budget=dict(
            unit="intended_evaluation_family",
            intended=len(inputs),
            eligible=sum(r["q"] is not None for r in inputs),
            started=len(inputs),
            completed=len(inputs),
            failed=0,
            censored=sum(r["status"] == "censored" for r in inputs),
            excluded=0,
            independent=len({r["source_cluster_id"] for r in inputs}),
            independent_unit="original_source_cluster",
            timing_repetitions=10,
            named_warmups=1,
        ),
    )
    replay(value)
    return value
