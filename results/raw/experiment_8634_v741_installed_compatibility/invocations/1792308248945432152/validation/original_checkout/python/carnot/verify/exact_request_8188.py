"""REQ-VERIFY-8188: durable evidence reuse under a fixed-seed service contract.

Evidence and decisions have different lifetimes. Changing a head recalculates
its decision; changing any acquisition operand requires another verifier call.
"""

from __future__ import annotations

from copy import deepcopy
import json
import math
import os
from pathlib import Path
import random
import struct
import tempfile
import time
from typing import Any
from unittest.mock import patch

import numpy as np

from carnot.verify import complete_request_8174 as prior
from carnot.reporting.experiment_7303_validation_scope import REQUIRED_CHECK_NAMES

Json = dict[str, Any]
ROOT = prior.ROOT
NAME = "experiment_8188_v707_exact_request_service"
CLI = f"scripts/experiments/{NAME}.py"
MODULE = "python/carnot/verify/exact_request_8188.py"
RUNNER = "python/carnot/reporting/exact_request_execution_8188.py"
TEST = "tests/python/test_exact_request_8188.py"
OWNED = [MODULE, RUNNER, CLI]
UPSTREAM = "results/experiment_8174_v706_complete_request_cost.json"
PIN = "sha256:15a13284335442ff04a491c6683d825e2b4d96de3950e6aa602a7c0afa1a252f"
CONFIG = dict(prior.CONFIG, seed=7078188, main_calls=72, startup_amortization_requests=96)
CONDITIONS = ("cold_miss", "exact_repeat", "forced_refresh", "changed_source")
PROTOCOL = dict(
    schema="carnot.exact_request.v1",
    semantics="memoized fixed-seed verifier text",
    fresh_stochastic_sample=False,
    head_in_acquisition_key=False,
    forced_refresh=True,
    repeat_fractions=[0, 0.25, 0.5, 0.9],
)
progress, atomic_json, reference = prior.progress, prior.atomic_json, prior.reference
canonical_hash, sha256_file = prior.canonical_hash, prior.sha256_file


def load_template(data: Json) -> str:
    """Read the actual embedded template so the key includes rendered prompt bytes."""
    model = Path(data["runtime_freeze"]["model_path"])
    metadata = prior.shared.prior.qualified.legacy.read_gguf_metadata(model)
    offset = metadata["field_provenance"]["metadata_keys"]["tokenizer.chat_template"][
        "value_offset"
    ]
    with model.open("rb") as stream:
        stream.seek(offset)
        length = struct.unpack("<Q", stream.read(8))[0]
        return stream.read(length).decode()


def render(data: Json, slot: Json) -> str:
    """Use the embedded formatter; private fixtures use explicitly synthetic text."""
    if not data.get("chat_template"):
        return json.dumps(slot["request"]["messages"], sort_keys=True)
    from llama_cpp.llama_chat_format import Jinja2ChatFormatter

    formatter = Jinja2ChatFormatter(data["chat_template"], eos_token="<|im_end|>", bos_token="")
    return str(formatter(messages=slot["request"]["messages"], enable_thinking=False).prompt)


def inputs(root: Path, raw: Path, *, fixture: bool = False) -> Json:
    """Use the hash-bound original panel, rather than selecting new convenient sources."""
    data = prior.inputs(root, raw, fixture=True)
    gate = prior.shared.host.old.gate
    path = root / UPSTREAM
    gate(data, path, "original_panel_exists", True, path.is_file())
    if path.is_file():
        try:
            value = json.loads(path.read_text())
            gate(data, path, "original_panel_sha256", PIN, sha256_file(path))
            gate(
                data,
                path,
                "complete_service_ready_score",
                1,
                value.get("complete_service_ready_score"),
            )
            snapshot = next(
                r for r in value["raw_shard_hashes"] if Path(r["path"]).name == "input_data.json"
            )
            frozen = Path(snapshot["path"])
            gate(data, frozen, "panel_snapshot_sha256", snapshot["sha256"], sha256_file(frozen))
            original = json.loads(frozen.read_text())
            gate(
                data,
                frozen,
                "original_source_ids",
                [s["source_cluster_id"] for s in original["slots"]],
                [s["source_cluster_id"] for s in data["slots"]],
            )
            data["refs"].extend([reference(path), snapshot])
            data["slots"] = deepcopy(original["slots"])
            if all(c["passed"] for c in data["checks"]) and not fixture:
                data["runtime_freeze"], counts = prior.shared.tokenizer_counts(data)
                data["chat_template"] = load_template(data)
                gate(data, frozen, "bounded_input_tokens", True, all(c <= 6000 for c in counts))
        except (OSError, ValueError, KeyError, StopIteration, ImportError, RuntimeError) as error:
            gate(
                data, path, "authenticated_service_inputs", True, f"{type(error).__name__}:{error}"
            )
    for slot in data["slots"]:
        slot["request"].update(seed=CONFIG["seed"], max_tokens=128, cache_prompt=False)
    data.update(ready=all(c["passed"] for c in data["checks"]), fixture=fixture)
    atomic_json(raw / "input_data.json", data)
    return data


def identity(data: Json, slot: Json) -> Json:
    """Bind acquisition operands, keeping mutable decision-head parameters separate."""
    request, runtime = slot["request"], data["runtime_freeze"]
    body = json.loads(request["messages"][1]["content"])
    return dict(
        source_bytes=body["complete_source"].encode().hex(),
        answer_bytes=body["original_answer"].encode().hex(),
        model_gguf=data["protocol"]["gguf_sha256"],
        model_revision=data["protocol"]["model_revision"],
        tokenizer=runtime.get("tokenizer_sha256", "private_fixture"),
        runtime=data["protocol"]["runtime_sha256"],
        rendered_prompt=render(data, slot),
        grammar=request.get("grammar"),
        seed=request["seed"],
        generation_parameters={
            k: v for k, v in request.items() if k not in {"messages", "grammar", "seed"}
        },
        request_schema=PROTOCOL["schema"],
    )


def key(value: Json) -> str:
    """Canonical encoding prevents field ordering from changing request identity."""
    return str(canonical_hash(value))


class Cache:
    """Persist raw responses; corrupt bytes remain quarantined for diagnosis."""

    def __init__(self, path: Path):
        self.path = path
        path.mkdir(parents=True, exist_ok=True)

    def read(self, value: Json) -> Json | None:
        """A missing or unauthenticated entry requires acquisition, never a decision."""
        path = self.path / (key(value).split(":")[1] + ".json")
        if not path.exists():
            return None
        try:
            envelope = json.loads(path.read_text())
            if envelope["identity"] != value or envelope["sha256"] != key(envelope["evidence"]):
                raise ValueError("cache_custody")
            return dict(envelope["evidence"])
        except (OSError, ValueError, KeyError, TypeError):
            path.rename(path.with_suffix(f".corrupt-{time.time_ns()}"))
            return None

    def write(self, value: Json, evidence: Json) -> None:
        """Atomic file and directory fsync make insertion survive a process restart."""
        atomic_json(
            self.path / (key(value).split(":")[1] + ".json"),
            dict(identity=value, evidence=evidence, sha256=key(evidence)),
        )
        directory = os.open(self.path, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)


def features(slot: Json, response: Json) -> Json:
    """Reparse cached evidence so a head change cannot reuse an old decision."""
    parsed = prior.shared.prior.stream.risk.transport.parse_response(response, slot["visible_ids"])
    body = json.loads(slot["request"]["messages"][1]["content"])
    lexical = prior.shared.prior.stream.lexical.extract(
        dict(
            family_id=slot["family_id"],
            source_bytes=body["complete_source"].encode().hex(),
            answer_bytes=body["original_answer"].encode().hex(),
        )
    )
    if not parsed["completed"] or lexical["values"] is None:
        raise ValueError("unusable_cached_evidence")
    p = min(1 - 1e-6, max(1e-6, parsed["probability"]))
    return dict(parsed=parsed, values=[math.log(p / (1 - p)), *lexical["values"]])


def request(
    data: Json,
    slot: Json,
    condition: str,
    runtime: Any,
    native: Any,
    cache: Cache,
    raw: Path,
    ledger: Any,
    call_id: str,
) -> Json:
    """Arrival includes hashing, evidence access, acquisition and durable response."""
    progress("8188_request_before_" + call_id, 0, 1)
    arrived = time.monotonic_ns()
    operand = identity(data, slot)
    cache_key = key(operand)
    hashed = time.monotonic_ns()
    response = None if condition == "forced_refresh" else cache.read(operand)
    read_end = time.monotonic_ns()
    path = raw / call_id
    if response is None:
        row = prior.acquire(slot, prior.ARMS[1], runtime, path, ledger, call_id)
        if "raw_response" in row:
            call = next(r for r in ledger.rows if r["call_id"] == call_id)
            call.update(
                input_tokens=row["input_tokens"],
                output_tokens=row["output_tokens"],
                finish_reason=row["raw_response"]["choices"][0].get("finish_reason"),
            )
            ledger.save()
    else:
        parse_start = time.monotonic_ns()
        computed = features(slot, response)
        parse_end = time.monotonic_ns()
        row = dict(
            unit_id=call_id,
            source_unit_id=slot["unit_id"],
            source_cluster_id=slot["source_cluster_id"],
            family_id=slot["family_id"],
            visible_ids=slot["visible_ids"],
            request=deepcopy(slot["request"]),
            arm=prior.ARMS[1],
            raw_response=response,
            generated_text=response["choices"][0]["message"]["content"],
            status="acquired",
            exclusion_reason=None,
            input_tokens=0,
            output_tokens=0,
            generation_start_ns=read_end,
            generation_end_ns=read_end,
            parsing_start_ns=parse_start,
            parsing_end_ns=parse_end,
            feature_start_ns=parse_end,
            feature_end_ns=parse_end,
            acquisition_end_ns=parse_end,
            **computed,
        )
        atomic_json(path / "prompt.json", row["request"])
        atomic_json(path / "response.json", response)
        row["evidence"] = [reference(p) for p in path.glob("*.json")]
    write_start = time.monotonic_ns()
    if row["status"] == "acquired" and response is None and condition != "forced_refresh":
        cache.write(operand, row["raw_response"])
    write_end = time.monotonic_ns()
    row.update(
        arrival_ns=arrived,
        cache_identity=operand,
        cache_key=cache_key,
        cache_hit=response is not None,
        condition=condition,
        metric="complete_latency_ns",
        numerator=None,
        denominator=1,
        head_sha256=key(data["head"]),
        key_hash_start_ns=arrived,
        key_hash_end_ns=hashed,
        cache_read_start_ns=hashed,
        cache_read_end_ns=read_end,
        cache_write_start_ns=write_start,
        cache_write_end_ns=write_end,
    )
    row["finish_reason"] = row.get("raw_response", {}).get("choices", [{}])[0].get("finish_reason")
    row["queue_start_ns"] = arrived
    row["queue_end_ns"] = arrived
    if row["status"] == "acquired":
        prior.commit_group(data, [row], native, path / "decision")
    row["arm"] = condition
    progress("8188_request_after_" + call_id, int(row["status"] == "completed"), 0)
    return row


def measure(
    data: Json,
    runtime: Any,
    native: Any,
    raw: Path,
    ledger: Any,
    *,
    started: float | None = None,
    **kwargs: Any,
) -> Json:
    """Randomize paired controls; cold insertion precedes the identical repeat."""
    began = time.monotonic() if started is None else started
    slots = deepcopy(data["slots"])
    rng = random.Random(CONFIG["seed"])
    rng.shuffle(slots)
    work: Json = dict(requests=[], warmups=[], config=CONFIG, protocol=PROTOCOL)
    atomic_json(
        raw / "service_protocol.json",
        dict(
            protocol=PROTOCOL,
            config=CONFIG,
            source_ids=[s["source_cluster_id"] for s in slots],
            frozen_before_requests=True,
        ),
    )
    cache = Cache(raw / "evidence_cache")
    for i, slot in enumerate(slots[:2]):
        if time.monotonic() - began < CONFIG["latest_launch_s"]:
            work["warmups"].append(
                prior.acquire(slot, "warmup", runtime, raw / f"warmup-{i}", ledger, f"warmup-{i}")
            )
    for index, slot in enumerate(slots):
        middle = ["exact_repeat", "forced_refresh"]
        rng.shuffle(middle)
        for condition in ["cold_miss", *middle, "changed_source"]:
            variant = deepcopy(slot)
            if condition == "changed_source":
                body = json.loads(variant["request"]["messages"][1]["content"])
                body["complete_source"] += "\n"
                variant["request"]["messages"][1]["content"] = json.dumps(body)
            call_id = f"source-{slot['slot']}-{condition}"
            if (
                time.monotonic() - began >= CONFIG["latest_launch_s"]
                or ledger.counts()["generation_calls_attempted"] >= 74
            ):
                row = dict(
                    unit_id=call_id,
                    source_cluster_id=slot["source_cluster_id"],
                    arm=condition,
                    condition=condition,
                    metric="complete_latency_ns",
                    numerator=None,
                    denominator=1,
                    status="censored",
                    exclusion_reason="launch_cutoff",
                )
            else:
                row = request(
                    data,
                    variant,
                    condition,
                    runtime,
                    native,
                    Cache(cache.path),
                    raw,
                    ledger,
                    call_id,
                )
            work["requests"].append(row)
            progress(
                "8188_completed_requests",
                len(work["requests"]),
                4 * len(slots) - len(work["requests"]),
            )
        if (index + 1) % 8 == 0:
            atomic_json(raw / "primitive_rows.json", work)
            progress("8188_checkpoint", index + 1, len(slots) - index - 1)
    atomic_json(raw / "primitive_rows.json", work)
    work["pairs"] = [
        dict(
            source_cluster_id=s["source_cluster_id"],
            arms=[r for r in work["requests"] if r["source_cluster_id"] == s["source_cluster_id"]],
        )
        for s in slots
    ]
    atomic_json(raw / "primitive_rows.json", work)
    calls = {r["call_id"]: r for r in ledger.rows}
    for row in work["requests"] + work["warmups"]:
        if "raw_response" in row and not row.get("cache_hit", False):
            call = calls[row["unit_id"]]
            call.update(
                input_tokens=row["input_tokens"],
                output_tokens=row["output_tokens"],
                finish_reason=row["raw_response"]["choices"][0].get("finish_reason"),
            )
    ledger.save()
    return work


def components(row: Json) -> Json:
    """Partition clocks; native arithmetic stays inside its measured crossing."""
    boundaries = dict(
        key_hash=("key_hash_start_ns", "key_hash_end_ns"),
        cache_read=("cache_read_start_ns", "cache_read_end_ns"),
        generation=("generation_start_ns", "generation_end_ns"),
        parse=("parsing_start_ns", "parsing_end_ns"),
        features=("feature_start_ns", "feature_end_ns"),
        cache_write=("cache_write_start_ns", "cache_write_end_ns"),
        native_crossing_including_arithmetic=(
            "boundary_crossing_start_ns",
            "boundary_crossing_end_ns",
        ),
        decision_serialization=("serialization_start_ns", "serialization_end_ns"),
        decision_write_ack=("commit_start_ns", "commit_end_ns"),
    )
    costs = {k + "_ns": row[b] - row[a] for k, (a, b) in boundaries.items()}
    complete = row["response_ns"] - row["arrival_ns"]
    return dict(
        unit_id=row["unit_id"],
        source_cluster_id=row["source_cluster_id"],
        condition=row["condition"],
        **costs,
        queue_ns=0,
        load_ns=0,
        complete_latency_ns=complete,
        other_host_and_ack_ns=complete - sum(costs.values()),
        arithmetic_ns=None,
        arithmetic_clock_scope="included in native crossing; unavailable separately",
        overlap_double_counting=False,
    )


def reduce(work: Json) -> Json:
    """Compare complete paired responses and label traffic mixtures as hypothetical."""
    rows = work.get("requests", [])
    completed = [r for r in rows if r["status"] == "completed"]
    groups: Json = {}
    for row in completed:
        groups.setdefault(row["source_cluster_id"], {})[row["condition"]] = row
    pairs = [g for g in groups.values() if set(g) == set(CONDITIONS)]
    differences = [
        dict(
            source_cluster_id=g["cold_miss"]["source_cluster_id"],
            byte_parity=g["cold_miss"]["generated_text"]
            == g["exact_repeat"]["generated_text"]
            == g["forced_refresh"]["generated_text"],
            decision_parity=g["cold_miss"]["action"]
            == g["exact_repeat"]["action"]
            == g["forced_refresh"]["action"],
            probability_parity=abs(
                g["cold_miss"]["probability"] - g["forced_refresh"]["probability"]
            )
            <= 1e-10,
        )
        for g in pairs
    ]
    equivalent = bool(pairs) and all(
        all(r[k] for k in ("byte_parity", "decision_parity", "probability_parity"))
        for r in differences
    )
    intervals: list[Json] = []
    if pairs:
        logs = np.log(
            [g["forced_refresh"]["numerator"] / g["exact_repeat"]["numerator"] for g in pairs]
        )
        draws = np.random.default_rng(CONFIG["seed"]).choice(logs, (10000, len(logs))).mean(axis=1)
        intervals.append(
            dict(
                estimate=float(np.exp(logs.mean())),
                lower95=float(np.exp(np.quantile(draws, 0.025))),
                upper95=float(np.exp(np.quantile(draws, 0.975))),
                sources=len(pairs),
                draws=10000,
                conditional_exact_repeat_only=True,
                descriptive_only=not equivalent,
            )
        )
    distributions = {
        c: dict(
            count=len(values),
            p50_ns=float(np.quantile(values, 0.5)),
            p95_ns=float(np.quantile(values, 0.95)),
            mean_ns=float(np.mean(values)),
        )
        for c in CONDITIONS
        if (values := [r["numerator"] for r in completed if r["condition"] == c])
    }
    scenarios = []
    if "cold_miss" in distributions and "exact_repeat" in distributions:
        cold, hit = [distributions[c]["mean_ns"] for c in ("cold_miss", "exact_repeat")]
        startup = work.get("startup_ns", 0)
        scenarios = [
            dict(
                hypothetical_repeat_fraction=f,
                naturally_observed=False,
                unique_priming_charged_once=True,
                startup_ns=startup,
                startup_amortization_requests=96,
                expected_cost_ns=(1 - f) * cold + f * hit + startup / 96,
                cold_cost_ns=cold,
                repeat_cost_ns=hit,
            )
            for f in PROTOCOL["repeat_fractions"]
        ]
    return dict(
        rows=[
            {
                k: r[k]
                for k in (
                    "unit_id",
                    "source_cluster_id",
                    "arm",
                    "condition",
                    "metric",
                    "numerator",
                    "denominator",
                    "status",
                    "exclusion_reason",
                )
            }
            for r in rows
        ],
        request_rows=rows,
        cache_identity_rows=[
            dict(
                unit_id=r["unit_id"],
                source_cluster_id=r["source_cluster_id"],
                cache_key=r["cache_key"],
                cache_hit=r["cache_hit"],
                identity=r["cache_identity"],
            )
            for r in completed
        ],
        invalidation_rows=[
            dict(
                unit_id=r["unit_id"],
                source_cluster_id=r["source_cluster_id"],
                operand="complete_source_bytes",
                miss=not r["cache_hit"],
                status=r["status"],
            )
            for r in rows
            if r["condition"] == "changed_source" and "cache_hit" in r
        ],
        restart_rows=[
            dict(
                unit_id=r["unit_id"],
                evidence_reopened=True,
                cache_hit=r["cache_hit"],
                process_restart=False,
                scope="new cache object; private child restart also validated",
            )
            for r in completed
            if r["condition"] == "exact_repeat"
        ],
        component_cost_rows=[components(r) for r in completed],
        request_distributions=distributions,
        paired_speed_intervals=intervals,
        amortized_cost_scenarios=scenarios,
        measured_repeat_frequency=None,
        parity_rows=differences,
        stochastic_divergence_count=sum(not r["byte_parity"] for r in differences),
        byte_parity_scope="verifier text bytes; transport IDs and timestamps are provenance",
        intended_count=96,
        eligible_count=len(rows),
        independent_count=len(pairs),
        completed_count=len(completed),
        excluded_count=max(0, 96 - len(rows)),
        censored_count=sum(r["status"] == "censored" for r in rows),
        failed_count=sum(r["status"] == "failed" for r in rows),
        equivalent_behavior_score=int(equivalent),
        cached_service_ready_score=int(len(pairs) == 24 and len(rows) == 96),
        nfr01_met=False,
    )


def validate_work(data: Json, work: Json) -> bool:
    """Recompute source identity, native decision parity and nonoverlapping clocks."""
    try:
        seen: set[str] = set()
        for row in work.get("requests", []):
            if row["unit_id"] in seen or row["condition"] not in CONDITIONS:
                return False
            seen.add(row["unit_id"])
            slot = deepcopy(
                next(s for s in data["slots"] if s["source_cluster_id"] == row["source_cluster_id"])
            )
            if row["condition"] == "changed_source":
                body = json.loads(slot["request"]["messages"][1]["content"])
                body["complete_source"] += "\n"
                slot["request"]["messages"][1]["content"] = json.dumps(body)
            if row["status"] != "completed":
                continue
            expected = identity(data, slot)
            if (
                row["request"] != slot["request"]
                or row["cache_identity"] != expected
                or row["cache_key"] != key(expected)
            ):
                return False
            computed = features(slot, row["raw_response"])
            probability = float(
                prior.shared.host.old.score(
                    data["head"], data["geometry"], [computed["values"]], None, "python_batch"
                )[0]
            )
            costs = components(row)
            clocks = [
                row[k]
                for k in (
                    "arrival_ns",
                    "key_hash_end_ns",
                    "cache_read_end_ns",
                    "generation_start_ns",
                    "generation_end_ns",
                    "parsing_start_ns",
                    "parsing_end_ns",
                    "feature_start_ns",
                    "feature_end_ns",
                    "cache_write_start_ns",
                    "cache_write_end_ns",
                    "batch_start_ns",
                    "boundary_crossing_start_ns",
                    "boundary_crossing_end_ns",
                    "serialization_start_ns",
                    "serialization_end_ns",
                    "commit_start_ns",
                    "commit_end_ns",
                    "response_ns",
                )
            ]
            if (
                clocks != sorted(clocks)
                or row["numerator"] != costs["complete_latency_ns"]
                or costs["other_host_and_ack_ns"] < 0
            ):
                return False
            if (
                computed["values"] != row["values"]
                or abs(probability - row["probability"]) > 1e-10
                or row["head_sha256"] != key(data["head"])
            ):
                return False
            if (
                row["action"] != prior.shared.host.old.engine.historical.radial.action(probability)
                or row["generated_text"] != row["raw_response"]["choices"][0]["message"]["content"]
            ):
                return False
            for ref in row["evidence"] + [row["store"]]:
                if sha256_file(Path(ref["path"])) != ref["sha256"]:
                    return False
            evidence = {Path(r["path"]).name: Path(r["path"]) for r in row["evidence"]}
            if (
                json.loads(evidence["prompt.json"].read_text()) != row["request"]
                or json.loads(evidence["response.json"].read_text()) != row["raw_response"]
            ):
                return False
            durable = prior.shared.host.Store(Path(row["store"]["path"]), key(data["head"])).state[
                "records"
            ][0]
            if (
                durable["request_id"] != row["unit_id"]
                or durable["action"] != row["action"]
                or durable["input_hash"] != key(row["request"])
            ):
                return False
        return True
    except (OSError, ValueError, KeyError, StopIteration, TypeError):
        return False


def live(data: Json, raw: Path, scratch: Path) -> Json:
    """Reuse the qualified worker while giving its lease and call schedule this identity."""
    if os.environ.get("CARNOT_FORCE_LIVE") != "1":
        prior.shared.host.old.gate(
            data, raw / "environment", "CARNOT_FORCE_LIVE", "1", os.environ.get("CARNOT_FORCE_LIVE")
        )
        return dict(work={}, ledger=[], checks=[])
    legacy = prior.shared.prior.qualified.legacy
    capture, load = legacy.live_capture, legacy.QwenRuntime.load

    def owned_capture(*args: Any, **kwargs: Any) -> Json:
        """Nested legacy helpers cannot rename the owner of this experiment's lease."""
        with patch.object(legacy, "TASK", "exp8188-exact-request-service"):
            return dict(capture(*args, **kwargs))

    def bounded_load(runtime: Any) -> Json:
        """A slow model hash is part of the bounded load, not unbounded setup."""
        return dict(legacy.bounded(lambda: load(runtime), 300))

    def pulse(phase: str, started: float, units: int = 0) -> None:
        """Read persisted counts during waits instead of inventing completed activity."""
        counts = prior.shared.Ledger(raw / "ledger.json").counts()
        progress(
            "8188_" + phase,
            counts["generation_calls_completed"] + counts["model_loads_completed"],
            counts["generation_calls_in_flight"] + counts["model_loads_in_flight"],
        )

    with (
        patch.object(prior.shared.prior, "capture", measure),
        patch.object(legacy, "live_capture", owned_capture),
        patch.object(legacy.QwenRuntime, "load", bounded_load),
        patch.object(legacy, "progress", pulse),
    ):
        result = dict(prior.shared.prior.live(data, raw, scratch))
    atomic_json(raw / "live_result.json", result)
    return result


def resume_evidence(raw: Path) -> tuple[Json, Json]:
    """Use sealed closed-lease evidence after an adapter failure, without new calls."""
    from carnot.gpu_lease_phase_journal import validate_journal_document
    from carnot.inference.qwen_sufficiency_7920 import offload_layers

    seal = json.loads((raw / "closure_error.json").read_text())
    if any(sha256_file(Path(r["path"])) != r["sha256"] for r in seal["references"]):
        raise ValueError("resume_evidence_hash")
    data = json.loads((raw / "input_data.json").read_text())
    work = json.loads((raw / "primitive_rows.json").read_text())
    ledger = json.loads((raw / "ledger.json").read_text())["rows"]
    lease = json.loads((raw / "gpu_lease_journal.json").read_text())
    layers = offload_layers((raw / "server.log").read_text())
    errors = validate_journal_document(
        lease, check_freshness=False, expected_model=data["runtime_freeze"]["model_path"]
    )
    if (
        errors
        or lease["task_id"] != "exp8188-exact-request-service"
        or not lease["released"]
        or lease["phase"] != "terminal_complete"
        or layers[0] != layers[1]
        or layers[0] == 0
        or any(r["owner_pid"] != lease["owner"]["pid"] for r in ledger)
        or not validate_work(data, work)
        or not validate_ledger(work, ledger)
        or len(work["requests"]) != 96
        or any(r["status"] != "completed" for r in work["requests"])
    ):
        raise ValueError("resume_owned_lease_or_requests")
    return data, dict(
        work=work,
        ledger=ledger,
        checks=[],
        gpu_lease_receipt=lease,
        observed_offload_layers=layers,
        cleanup=dict(leak_free=lease["unload_evidence"]["observed"]),
        recovery_references=[*seal["references"], reference(raw / "closure_error.json")],
        measurement_code_hashes=seal["measurement_code_hashes"],
        closure_recovery=dict(
            original_error=seal["error"],
            reused_actual_requests=96,
            new_model_calls=0,
            unavailable_worker_properties="HTTP props were not saved before wrapper failure",
        ),
    )


def validate_ledger(work: Json, ledger: list[Json]) -> bool:
    """Match real calls to source rows; a cache hit contributes no invocation."""
    try:
        calls = {r["call_id"]: r for r in ledger if r["operation"] == "generation"}
        loads = [r for r in ledger if r["operation"] == "model_load"]
        expected = {
            r["unit_id"]: r
            for r in work.get("requests", []) + work.get("warmups", [])
            if "generation_start_ns" in r and not r.get("cache_hit", False)
        }
        if (
            len(ledger) != len(calls) + len(loads)
            or len(loads) > 1
            or len(calls) > 74
            or set(calls) != set(expected)
        ):
            return False
        for name, row in expected.items():
            call = calls[name]
            if call["request_sha256"] != key(row["request"]) or call["status"] not in {
                "completed",
                "failed",
            }:
                return False
            if "raw_response" in row:
                if (
                    call["response_sha256"] != key(row["raw_response"])
                    or call.get("input_tokens") != row["input_tokens"]
                    or not row["generation_start_ns"]
                    <= call["started_monotonic_ns"]
                    <= row["generation_end_ns"]
                    <= call["ended_monotonic_ns"]
                ):
                    return False
        return True
    except (KeyError, TypeError):
        return False


def build(
    data: Json,
    result: Json,
    raw: Path,
    receipts: list[Json],
    date: str,
    duration: float,
    fixture: bool,
) -> Json:
    """Separate qualified cache behavior from current live and natural-traffic claims."""
    work = result.get("work") or dict(requests=[], warmups=[])
    ledger = prior.shared.Ledger(raw / "build-ledger.json")
    ledger.rows = result.get("ledger", [])
    work["startup_ns"] = sum(
        r["ended_monotonic_ns"] - r["started_monotonic_ns"]
        for r in ledger.rows
        if r["operation"] == "model_load" and r["ended_monotonic_ns"]
    ) + sum(r["acquisition_end_ns"] - r["arrival_ns"] for r in work.get("warmups", []))
    atomic_json(raw / "primitive_rows.json", work)
    atomic_json(raw / "input_data.json", data)
    counts = dict(prior.shared.prior.ZERO_INVOCATION_COUNTS) if fixture else ledger.counts()
    checked = (
        bool(receipts)
        and (fixture or set(REQUIRED_CHECK_NAMES) <= {r.get("name") for r in receipts})
        and all(
            r.get("passed") and r.get("normal_exit") and r.get("actual_exit", 0) == 0
            for r in receipts
        )
    )
    reduction = reduce(work)
    checks = data["checks"] + result.get("checks", [])
    blocked = next((r["check"] for r in checks if not r["passed"]), None)
    ready = bool(
        reduction["cached_service_ready_score"]
        and checked
        and not blocked
        and not fixture
        and counts["generation_calls_completed"] == 74
        and counts["model_loads_completed"] == 1
        and result.get("gpu_lease_receipt")
        and duration >= 10
    )
    verdict = (
        "circular_positive"
        if fixture
        else "positive"
        if ready and reduction["equivalent_behavior_score"]
        else "null"
    )
    if blocked:
        verdict = "blocked"
    if not checked:
        verdict = "disqualified"
    reduction["cached_service_ready_score"] = int(ready)
    reduction["equivalent_behavior_score"] = int(ready and reduction["equivalent_behavior_score"])
    value: Json = dict(
        reduction,
        experiment_id=8188,
        task_id="exp8188-exact-request-service",
        schema="carnot.exact_request_service.v1",
        honest_verdict="complete_blocked_" + str(blocked)
        if verdict == "blocked"
        else "complete_" + verdict + "_exact_request_service",
        verdict_class=verdict,
        verifier_is_oracle=fixture,
        fixture_mode=fixture,
        claim_scope="Conditional exact-byte fixed-seed reuse; no fresh stochastic sample, observed traffic, hardware, semantic or learning gain",
        exposure_scope="exposed_historical_development",
        independent_generalization_score=0,
        generalized_learning_benefit_score=0,
        required_checks_passed=checked,
        flagged_adversarial=False,
        validation_receipts=receipts,
        terminal_validation_sidecar_path=str(raw / "terminal_validation.json"),
        preconditions_checked=True,
        gate_check_summary=checks,
        inference_substrate="live_llm_inference"
        if counts["generation_calls_completed"]
        else "private_transport_fixture"
        if fixture
        else "no_model_execution",
        inference_substrate_class="model_bounded_generation"
        if counts["model_loads_attempted"]
        else "no_model_load",
        declared_inference_substrate_class="model_bounded_generation",
        MODEL_SPECS=prior.MODEL_SPECS,
        trained_head_specs=data.get("trained_head_specs", []),
        model_invocation_counts=counts,
        call_ledger=[] if fixture else ledger.rows,
        private_transport_ledger=ledger.rows if fixture else [],
        unstarted_request_ledger=[r for r in work["requests"] if r["status"] == "censored"],
        cited_upstream_artifacts=[
            dict(
                experiment_id=8174,
                fields_imported=["original_source_panel", "deployment_inputs"],
                sha256=PIN,
            )
        ],
        source_artifact_hashes=data["refs"],
        raw_shard_hashes=[
            reference(raw / "primitive_rows.json"),
            reference(raw / "input_data.json"),
        ]
        + [reference(p) for p in (raw / "evidence_cache").glob("*.json")]
        + [
            reference(raw / n)
            for n in ("live_result.json", "ledger.json", "service_protocol.json")
            if (raw / n).is_file()
        ],
        code_config_hashes={p: sha256_file(ROOT / p) for p in [*OWNED, TEST]},
        config_sha256=key(CONFIG),
        run_date=date,
        duration_s=duration,
        random_seed=CONFIG["seed"],
        sample_size_budget=CONFIG,
        phase_spans=[
            dict(phase=r.get("name"), duration_s=r.get("duration_s", 0)) for r in receipts
        ],
        acceptance_gates=dict(
            sources=24,
            main_generations=72,
            warmups=2,
            duration_floor_s=10,
            equivalence_required=True,
            nfr01_complete_boundary_lower95=10,
            natural_repeat_frequency_required=True,
        ),
        field_principles=dict(
            identity="All acquisition bytes bind; heads recompute decisions",
            costs="Disjoint intervals; cold priming once; arithmetic included in native crossing",
            generalization="Exposed development and fixtures establish no independent benefit",
            readiness="Normal owned validation and current observed CUDA execution required",
            traffic="Repeat fractions are hypothetical; no natural repeat rate measured",
        ),
        methodology="24 original sources, randomized paired order, 72 bounded generations plus two warmups; cold/hit/refresh/changed source; source bootstrap with10000 draws",
        service_protocol=PROTOCOL,
        runtime_freeze=data.get("runtime_freeze", {}),
        gpu_receipts={k: v for k, v in result.items() if k not in {"work", "ledger", "checks"}},
        startup_costs=dict(
            load_and_warmups_ns=work["startup_ns"],
            charged_once=True,
            request_latency_includes_load=False,
        ),
        generator_weight_updates=0,
        device_changes=0,
        recovery_fixture_scope="Owned private tests exercise process restart, corruption reacquisition and head update without live credit",
    )
    value["reproducibility_checksum"] = prior.checksum(value)
    return value


def replay(path: Path) -> bool:
    """Reopen primitives and receipts in a fresh process before trusting headlines."""
    try:
        value = json.loads(path.read_text())
        refs = value["source_artifact_hashes"] + value["raw_shard_hashes"]
        refs += [dict(path=str(ROOT / p), sha256=h) for p, h in value["code_config_hashes"].items()]
        refs += [
            dict(path=r["log_path"], sha256=r["log_sha256"])
            for r in value["validation_receipts"] + value.get("repository_health", [])
            if "log_path" in r
        ]
        if any(sha256_file(Path(r["path"])) != r["sha256"] for r in refs):
            return False
        paths = {Path(r["path"]).name: Path(r["path"]) for r in value["raw_shard_hashes"]}
        data, work = [
            json.loads(paths[n].read_text()) for n in ("input_data.json", "primitive_rows.json")
        ]
        with tempfile.TemporaryDirectory(prefix="8188-replay-") as private:
            expected = build(
                data,
                dict(
                    work=work,
                    ledger=value["private_transport_ledger"]
                    if value["fixture_mode"]
                    else value["call_ledger"],
                    checks=[c for c in value["gate_check_summary"] if c not in data["checks"]],
                    **value["gpu_receipts"],
                ),
                Path(private),
                value["validation_receipts"],
                value["run_date"],
                value["duration_s"],
                value["fixture_mode"],
            )
        fields = [
            *reduce(work),
            "honest_verdict",
            "verdict_class",
            "required_checks_passed",
            "model_invocation_counts",
        ]
        return bool(
            validate_work(data, work)
            and validate_ledger(
                work,
                value["private_transport_ledger"]
                if value["fixture_mode"]
                else value["call_ledger"],
            )
            and all(value[k] == expected[k] for k in fields)
            and value["config_sha256"] == key(CONFIG)
            and value["reproducibility_checksum"] == prior.checksum(value)
        )
    except (OSError, ValueError, KeyError, TypeError):
        return False
