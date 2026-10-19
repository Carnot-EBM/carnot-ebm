"""REQ-REPORT-8659: measure source reach without turning overlap into truth.

Private release bytes remain outside the repository. Ordered offsets and exact
feature hashes let a fresh reader check what each response actually exposed.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import resource
import time
from typing import Any

import numpy as np

from carnot.reporting import triviaplus_extraction_8620 as old
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash, sha256_file
from carnot.reporting.streaming_adapter_8656 import gate as upstream_gate
from carnot.reporting.streaming_custody_8642 import freeze
from carnot.verify import streaming_evidence as stream
from carnot.verify.evidence_features_7980 import extract as capped

Json = dict[str, Any]
ROOT = old.ROOT
NAME = "experiment_8659_v743_complete_source_comparison"
TASK = "exp8659-complete-source-comparison"
CLI = f"scripts/experiments/{NAME}.py"
TEST = "tests/python/test_complete_source_8659.py"
OWNED = [
    "python/carnot/reporting/complete_source_8659.py",
    "python/carnot/reporting/complete_source_runner_8659.py",
    CLI,
]
SEED = 7418632
MODEL_SPECS: list[Json] = []
PINS = {
    "results/experiment_8653_v743_contract_methods.json": "sha256:c1176a28073046f66bd1eb8ba412ec025b4695e3bc14a212d829a5c5818e04f9",
    "results/experiment_8656_v743_streaming_execution_custody.json": "sha256:167d1f04e0803aeec8d4d3ce2275035a902177956a4aa27edc88918c445bbd81",
    "results/experiment_8620_v740_triviaplus_extraction.json": "sha256:f92ba0a209d4b877d7c243a4aafa0d1ad1151b6e4e0a0fb6bb676dc2c3916ac3",
    "results/experiment_8642_v742_streaming_execution_custody.json": "sha256:0c8f23991c6986a987d5de6df8d664ba544b2db2a486b2c9cd25ba0c4b0c9169",
    "results/experiment_8645_complete_source_comparison.json": "sha256:7124ca0a612b141dc20c40f0d15a2c667a4210402395619b3574abbe924f30bd",
    "results/experiment_8617_v740_triviaplus_release.json": "sha256:7c45c28d09837ede0b976100cbdd356f7bc778da717fc3876c1c9bd80ca138cd",
}
reference = old.reference
read = old.read


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Flush counts so a supervisor can distinguish active work from a stall."""
    print(f"[exp8659] phase={phase} completed={completed} pending={pending}", flush=True)


def gate(path: Path, field: str, expected_hash: str | None = None) -> Json:
    """Keep missing operands, missing fields and a measured zero distinct."""
    result = upstream_gate(path, field)
    if expected_hash is not None and result["sha256"] != expected_hash:
        result.update(
            check="upstream_identity",
            expected=expected_hash,
            observed=result["sha256"],
            field="sha256",
            passed=False,
            missing_reason="missing_file" if not path.is_file() else "hash_mismatch",
        )
    return result


def inputs(root: Path) -> Json:
    """Authenticate original release and roster bytes before any label is opened."""
    gates = [
        gate(root / p, f, PINS[p])
        for p, f in zip(
            list(PINS)[:2],
            ("extraction_inputs_ready_score", "streaming_extractor_ready_score"),
            strict=True,
        )
    ]
    if not all(g["passed"] for g in gates):
        return dict(gates=gates, public=[], questions={}, baseline=[])
    method = read(dict(path=str(root / next(iter(PINS))), sha256=next(iter(PINS.values()))))
    _, release, public, questions = old.content(root)
    baseline = read(dict(path=str(root / list(PINS)[2]), sha256=list(PINS.values())[2]))
    roster = method["source_roster_reference"]
    if (
        reference(Path(roster["predictor_reference"]["path"])) != roster["predictor_reference"]
        or [r["opaque_id"] for r in public] != [r["opaque_id"] for r in baseline["extraction_rows"]]
        or len(public) != 190
        or len(set(questions.values())) != 128
        or sum(r["values"] is not None for r in baseline["extraction_rows"]) != 135
    ):
        raise ValueError("fixed_roster")
    checkpoint = method["target_model"]["checkpoint"]
    read(checkpoint)
    return dict(
        gates=gates,
        public=public,
        questions=questions,
        baseline=baseline["extraction_rows"],
        method=method,
        release=release,
    )


def io_counters() -> dict[str, int]:
    """Logical and physical kernel I/O counters describe different costs."""
    return {
        k: int(v)
        for k, v in (line.split(":") for line in Path("/proc/self/io").read_text().splitlines())
    }


def response(row: Json | Path, arm: str, chunk: int, primitive: Path) -> Json:
    """Use a fresh child for each cost repeat so peak memory belongs to one response."""
    if arm not in {"original_capped", "streaming_complete"}:
        raise ValueError("arm")
    started, before = time.monotonic_ns(), io_counters()
    peak_before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    if isinstance(row, Path):
        row = json.loads(row.read_bytes())
    public = dict(
        family_id=row["opaque_id"],
        source_bytes=row["source_bytes"],
        answer_bytes=row["answer_bytes"],
    )
    if arm == "original_capped":
        features = capped(public)
        result = dict(
            features=features,
            complete=features["abstention"] is None,
            total_window_count=None,
            reserved_working_bytes=None,
        )
    else:
        result = stream.extract(public, primitive, chunk_size=chunk)
    after = io_counters()
    result["window_reference"] = reference(primitive) if primitive.is_file() else None
    result["cost"] = dict(
        duration_s=(time.monotonic_ns() - started) / 1e9,
        started_monotonic_ns=started,
        ended_monotonic_ns=time.monotonic_ns(),
        source_bytes=len(bytes.fromhex(row["source_bytes"])),
        answer_bytes=len(bytes.fromhex(row["answer_bytes"])),
        io_bytes={k: after[k] - before[k] for k in before},
        peak_extra_memory_bytes=max(
            0, resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024 - peak_before
        ),
        peak_memory_method="fresh process ru_maxrss minus post-import high-water baseline",
        window_count=result["features"]["window_count"],
        total_window_count=result["total_window_count"],
        arm=arm,
        chunk_size=chunk,
    )
    return result


def audit_windows(row: Json, ref: Json) -> int:
    """Rebuild offsets and byte-equality deduplication; repaired hashes cannot hide edits."""
    path = Path(ref["path"])
    if reference(path) != ref:
        raise ValueError("window_custody")
    source = bytes.fromhex(row["source_bytes"])
    units = list(stream.spans(bytes.fromhex(row["answer_bytes"])))
    seen: dict[bytes, int] = {}
    records = []
    for index, (start, end, window) in enumerate(stream.windows(source)):
        key = window.strip()
        fresh = key not in seen
        if fresh:
            seen[key] = len(seen)
        records.append(
            dict(
                index=index,
                start=start,
                end=end,
                window_hash=stream.window_hash(key),
                group_id=seen[key],
                first_occurrence=fresh,
                answer_unit_ids=list(range(len(units))),
            )
        )
    expected = b"".join(stream.record_bytes(v) for v in records)
    if path.read_bytes() != expected:
        raise ValueError("window_custody")
    return len(records)


def reduce(work: Json) -> Json:
    """Keep all intended rows and separate feature parity from increased support."""
    per, windows, costs = [], [], []
    common_parity, chunk_parity, custody, bounds = True, True, True, True
    baseline = {r["opaque_id"]: r for r in work["baseline"]}
    recovered: list[str] = []
    for row, result in zip(work["public"], work["responses"], strict=True):
        uid = row["opaque_id"]
        original = baseline[uid]
        new = result["streaming"]
        available = new is not None
        usable = available and new["features"]["values"] is not None
        common = original["values"] is not None
        parity = not common or (available and original["values"] == new["features"]["values"])
        if result["original"] is not None:
            parity = parity and result["original"]["features"]["values"] == original["values"]
            if common and available:
                parity = parity and result["original"]["features"] == new["features"]
        common_parity = common_parity and parity
        invariant = (
            available
            and len(result["invariance"]) == 2
            and all(
                p["features"] == new["features"]
                and p["complete_window_digest"] == new["complete_window_digest"]
                for p in result["invariance"]
            )
        )
        chunk_parity = chunk_parity and invariant
        if available:
            if new["complete"]:
                count = audit_windows(row, new["window_reference"])
                custody = custody and count == new["total_window_count"]
                windows.append(
                    dict(
                        opaque_id=uid,
                        window_count=count,
                        reference=new["window_reference"],
                        complete_window_digest=new["complete_window_digest"],
                    )
                )
            else:
                custody = False
            bounds = (
                bounds
                and new["cost"]["duration_s"] <= 30
                and new["cost"]["peak_extra_memory_bytes"] <= 268435456
            )
        costs += [dict(v, opaque_id=uid) for v in result["costs"]]
        bounds = bounds and all(
            v["duration_s"] <= 30 and v["peak_extra_memory_bytes"] <= 268435456
            for v in result["costs"]
        )
        if usable and not common:
            recovered.append(uid)
        per.append(
            dict(
                opaque_id=uid,
                question_hash=work["questions"][uid],
                original_usable=common,
                recovered=usable and not common,
                common_parity=parity,
                chunk_invariant=invariant,
                status="completed" if available else "censored",
                abstention=new["features"]["abstention"] if available else "measurement_deadline",
                streaming_features=new["features"] if available else None,
                original_features=result["original"]["features"] if result["original"] else None,
            )
        )
    clusters = sorted(set(work["questions"].values()))
    denominators = np.array([sum(r["question_hash"] == q for r in per) for q in clusters])
    numerators = np.array(
        [sum(r["question_hash"] == q and r["recovered"] for r in per) for q in clusters]
    )
    interval = None
    if clusters:
        weights = np.random.default_rng(SEED).multinomial(
            len(clusters), np.full(len(clusters), 1 / len(clusters)), size=10000
        )
        draws = (weights @ numerators) / (weights @ denominators)
        interval = np.quantile(draws, [0.025, 0.975]).tolist()
    return dict(
        per_response_results=per,
        source_window_rows=windows,
        complete_extraction_cost_rows=costs,
        common_parity=common_parity,
        chunk_parity=chunk_parity,
        complete_window_custody=custody,
        resource_bounds=bounds,
        recovered_count=len(recovered),
        recovered_ids=recovered,
        question_cluster_intervals=dict(
            absolute_recovered_responses=len(recovered),
            recovered_fraction=len(recovered) / len(per) if per else None,
            interval95=interval,
            question_clusters=len(clusters),
            bootstrap_draws=10000,
            random_seed=SEED,
            status="descriptive_exposed_question_cluster",
        ),
    )


def metrics(common: list[Json], recovered: list[Json]) -> Json:
    """Report fixed lexical scores separately; a changed denominator is not detector benefit."""

    def describe(rows: list[Json]) -> list[Json]:
        selected = [
            dict(
                row,
                fixed_scores=dict(
                    zip(
                        old.SCORES,
                        [
                            row["values"][5],
                            row["values"][3],
                            row["values"][6],
                            1 - row["values"][0],
                        ],
                        strict=True,
                    )
                ),
            )
            for row in rows
            if row["values"] is not None
        ]
        # The historical function owns the original four scores and support floor.
        # Pass the registered seed explicitly through an isolated value restoration.
        from unittest.mock import patch

        with patch.object(old, "SEED", SEED):
            return old.intervals(selected)

    return dict(
        common_support_metrics=describe(common),
        recovered_support_metrics=describe(recovered),
        annotation_sensitivity=dict(
            common_unanimous=describe([v for v in common if not v["annotation_disagreement"]]),
            common_disputed=describe([v for v in common if v["annotation_disagreement"]]),
            recovered_unanimous=describe(
                [v for v in recovered if not v["annotation_disagreement"]]
            ),
            recovered_disputed=describe([v for v in recovered if v["annotation_disagreement"]]),
            semantic_entailment_claim=False,
            missing_p2_imputed=False,
        ),
    )


def label_report(data: Json, report: Json) -> Json:
    """Open human votes only after the feature seal, and keep support groups explicit."""
    if not data["public"]:
        return metrics([], [])
    import pyarrow.parquet as pq

    original = pq.read_table(
        Path(data["release"]["private_cache_reference"]["path"]) / old.release.PARQUET
    ).to_pylist()
    features = [
        dict(opaque_id=v["opaque_id"], values=v["streaming_features"]["values"])
        for v in report["per_response_results"]
        if v["streaming_features"] is not None
    ]
    joined = old.join_labels(features, original, data["questions"])
    common_ids = {v["opaque_id"] for v in report["per_response_results"] if v["original_usable"]}
    return metrics(
        [v for v in joined if v["opaque_id"] in common_ids],
        [v for v in joined if v["opaque_id"] not in common_ids],
    )
