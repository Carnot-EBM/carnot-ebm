"""REQ-VERIFY-8146 / REQ-REPORT-8146: charge fresh acquisition to durable service.

Public retention requests and sealed natural coefficients are the only science
inputs. This timing experiment neither reads labels nor changes generator weights.
"""

from __future__ import annotations

from dataclasses import asdict
from copy import deepcopy
import argparse
import json
import math
import os
from pathlib import Path
import struct
import tempfile
import time
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import numpy as np

from carnot import experiment_8145_v704_natural_service_cost as host
from carnot import experiment_8099_v701_fit_source_capture as qualified
from carnot.reporting.current_work_receipt import (
    ZERO_INVOCATION_COUNTS,
    atomic_json,
    canonical_hash,
    sha256_file,
)
from carnot.reporting.experiment_7303_validation_scope import CommandSpec, build_scoped_commands
from carnot.reporting.primary_publication import publish_primary, read_bound_sidecar
from carnot.verify import qwen_learning_stream_capture_8102 as stream
from carnot.verify.qwen_development_capture_7995 import Ledger

Json = dict[str, Any]
ROOT = qualified.ROOT
NAME = "experiment_8146_v704_live_service_cost"
CLI = f"scripts/experiments/{NAME}.py"
MODULE = f"python/carnot/{NAME}.py"
TEST = "tests/python/test_live_service_cost_8146.py"
ARMS = ["python_scalar", "native_scalar"]
MODEL_SPECS = [dict(name="Qwen3.8-27B", hf_id=stream.risk.MODEL, quantization="Q4_K_M")]
CONFIG = dict(
    seed=7048146,
    measured_calls=48,
    warmup_calls=8,
    call_limit=56,
    max_tokens=128,
    output_tokens=7168,
    load_timeout_s=300,
    call_timeout_s=120,
    latest_launch_s=2200,
    capture_closure_s=2400,
    bootstrap_draws=10000,
    decision_tolerance=1e-10,
)
UPSTREAM = "results/experiment_8145_v704_natural_service_cost.json"
HISTORY = "results/experiment_8102_v701_learning_stream_capture.json"
PINS = {
    UPSTREAM: "sha256:9c11bb5c24e672f4371ca58f535bc199392d9fb1dcdac0bf0001a6298afcf30b",
    HISTORY: "sha256:32ca6d4b55322e3b31c8cf69fbc95dbb5a7fa065be8b84c5e59ce07c899623e8",
}
reference = host.reference


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Report actual phase counts without letting buffering hide pending work."""
    print(f"[exp8146] {phase} completed={completed} pending={pending}", flush=True)


def inputs(root: Path, raw: Path) -> Json:
    """Read public requests and the sealed head, avoiding training and label files."""
    data: Json = dict(ready=False, checks=[], refs=[], slots=[], library={})
    progress("8146_inputs_before")
    values: Json = {}
    for name, pin in PINS.items():
        path = root / name
        host.gate(data, path, "resource_exists", True, path.is_file())
        if not path.is_file():
            continue
        try:
            host.gate(data, path, "sha256", pin, sha256_file(path))
            value = json.loads(path.read_text())
            for field, expected in [
                ("required_checks_passed", True),
                ("flagged_adversarial", False),
            ]:
                host.gate(data, path, field, expected, value.get(field))
            terminal = Path(value["terminal_validation_sidecar_path"])
            publication = json.loads(terminal.read_text())["publication"]
            sidecar = Path(publication["sidecar_path"])
            host.gate(
                data,
                path,
                "terminal.report.passed",
                True,
                read_bound_sidecar(path, sidecar)["report"]["passed"],
            )
            data["refs"].extend(reference(p) for p in [path, terminal, sidecar])
            values[name] = value
        except (OSError, ValueError, KeyError) as error:
            host.gate(data, path, "authenticated_terminal", True, str(error))
    if len(values) == 2:
        try:
            upstream, history = values[UPSTREAM], values[HISTORY]
            host.gate(
                data,
                root / UPSTREAM,
                "natural_service_ready_score",
                1,
                upstream.get("natural_service_ready_score"),
            )
            refs = [
                upstream["final_head_manifest"],
                history["capture_manifest"],
                reference(root / host.prior.old.NATIVE),
            ]
            native = json.loads(Path(refs[2]["path"]).read_text())
            refs.append(
                dict(path=native["native_library_path"], sha256=native["native_library_sha256"])
            )
            for ref in refs:
                path = Path(ref["path"])
                host.gate(data, path, "sha256", ref["sha256"], sha256_file(path))
            heads = json.loads(Path(refs[0]["path"]).read_text())["rows"][0]
            manifest = json.loads(Path(refs[1]["path"]).read_text())["rows"]
            slots = deepcopy([r for r in manifest if r["role"] == "retention"][:24])
            for row in slots:
                row["request"].update(max_tokens=128, cache_prompt=False)
            data.update(
                slots=slots,
                head=heads["heads"]["error_center"],
                geometry=heads["geometry"],
                library=refs[3],
                trained_head_specs=upstream["trained_head_specs"],
                protocol={
                    k: history[k] for k in ["gguf_sha256", "runtime_sha256", "chat_template_sha256"]
                },
            )
            data["protocol"]["model_revision"] = history["runtime_identity"]["model_revision"]
            data["refs"].extend(refs)
            host.gate(
                data,
                root / HISTORY,
                "retention_first24_original_slots",
                list(range(1, 25)),
                [r["slot"] for r in slots],
            )
            data["ready"] = all(r["passed"] for r in data["checks"])
        except (OSError, ValueError, KeyError) as error:
            host.gate(data, root / UPSTREAM, "authenticated_public_inputs", True, str(error))
    atomic_json(raw / "input_data.json", data)
    progress("8146_inputs_after", int(data["ready"]), 0)
    return data


class FixtureRuntime:
    """Private scripted traffic checks transport but never earns model credit."""

    def count(self, text: str) -> int:
        return 1

    def generate(self, payload: Json) -> Json:
        return dict(
            model=stream.risk.MODEL,
            choices=[
                dict(
                    finish_reason="stop",
                    message=dict(
                        content='{"unsupported_probability":0.2,"source_sentence_id":null}'
                    ),
                )
            ],
            usage=dict(prompt_tokens=1, completion_tokens=16),
        )


def matched(rows: list[Json]) -> bool:
    """A favorable timing cannot conceal changed generated or durable response bytes."""
    if len(rows) != 2 or any(r["status"] != "completed" for r in rows):
        return False
    a, b = rows
    return (
        all(
            a[k] == b[k]
            for k in [
                "request_bytes",
                "response_bytes",
                "generated_text",
                "initial_state_hash",
                "decision",
            ]
        )
        and abs(a["probability"] - b["probability"]) <= 1e-10
    )


def transaction(
    data: Json,
    slot: Json,
    arm: str,
    runtime: Any,
    native: Any,
    raw: Path,
    ledger: Ledger,
    call_id: str,
) -> Json:
    """Time request ingress through a committed response from the same empty cache."""
    raw.mkdir(parents=True, exist_ok=True)
    state = host.state_for(data["head"], data["geometry"])
    atomic_json(raw / "initial.json", dict(state=state, cache={}))
    row: Json = dict(
        unit_id=call_id,
        source_cluster_id=slot["source_cluster_id"],
        arm=arm,
        condition="fresh_cache_miss",
        metric="complete_latency_ns",
        numerator=None,
        denominator=1,
        status="failed",
        exclusion_reason=None,
        started=False,
        initial_state_hash=sha256_file(raw / "initial.json"),
        components={},
        request=slot["request"],
        raw_response={},
    )
    began = time.monotonic_ns()
    step = began
    started = False
    try:
        baseline = json.loads((raw / "initial.json").read_text())
        if baseline["cache"]:
            raise ValueError("owned_cache_not_empty")
        request_bytes = json.dumps(slot["request"], sort_keys=True, separators=(",", ":"))
        row["request_bytes"] = request_bytes
        tokens = runtime.count(json.dumps(slot["request"]["messages"]))
        row["input_tokens"] = tokens
        if tokens > 6000:
            row.update(status="excluded", exclusion_reason="context_over_budget")
            return row
        row["components"]["ingress_hash_lookup_ns"] = time.monotonic_ns() - step
        progress("8146_generation_before_" + call_id)
        ledger.start("generation", call_id, slot["request"])
        started = row["started"] = True
        step = time.monotonic_ns()
        response = runtime.generate(slot["request"])
        row["components"]["generation_ns"] = time.monotonic_ns() - step
        row["raw_response"] = response
        ledger.finish(call_id, "completed", response)
        progress("8146_generation_after_" + call_id, 1, 0)
        step = time.monotonic_ns()
        parsed = stream.risk.transport.parse_response(response, slot["visible_ids"])
        body = json.loads(slot["request"]["messages"][1]["content"])
        lexical = stream.lexical.extract(
            dict(
                family_id=slot["family_id"],
                source_bytes=body["complete_source"].encode().hex(),
                answer_bytes=body["original_answer"].encode().hex(),
            )
        )
        if not parsed["completed"] or lexical["values"] is None:
            raise ValueError("unusable_generation_or_lexical")
        p = min(1 - 1e-6, max(1e-6, parsed["probability"]))
        values = [math.log(p / (1 - p)), *lexical["values"]]
        row["components"]["parse_lexical_ns"] = time.monotonic_ns() - step
        step = time.monotonic_ns()
        probability = float(host.score(data["head"], data["geometry"], [values], native, arm)[0])
        row["components"]["arithmetic_and_boundary_ns"] = time.monotonic_ns() - step
        step = time.monotonic_ns()
        decision = host.engine.historical.radial.action(probability)
        text = response["choices"][0]["message"]["content"]
        service = dict(generated_text=text, probability=probability, decision=decision)
        # Quantized wire probabilities preserve the stricter numerical comparison separately.
        wire = dict(service, probability=round(probability, 10))
        response_bytes = json.dumps(wire, sort_keys=True, separators=(",", ":"))
        row["components"]["decision_serialization_ns"] = time.monotonic_ns() - step
        step = time.monotonic_ns()
        atomic_json(
            raw / "response.json",
            dict(
                response_bytes=response_bytes,
                state=baseline["state"],
                cache={canonical_hash(slot["request"]): wire},
            ),
        )
        fd = os.open(raw, os.O_RDONLY)
        os.fsync(fd)
        os.close(fd)
        row["components"]["durable_commit_ns"] = time.monotonic_ns() - step
        row.update(
            service,
            binary_energy_difference=-math.log(
                max(probability, 1e-300) / max(1 - probability, 1e-300)
            ),
            cache_events=[
                dict(
                    key=canonical_hash([slot["request"], data.get("protocol"), state]),
                    status="miss",
                )
            ],
            response_bytes=response_bytes,
            values=values,
            lexical=lexical,
            parsed=parsed,
            status="completed",
            response_commit=reference(raw / "response.json"),
            output_tokens=response.get("usage", {}).get("completion_tokens", 0),
        )
    except (OSError, RuntimeError, TimeoutError, ValueError, KeyError) as error:
        row["exclusion_reason"] = f"{type(error).__name__}:{error}"
        if started and ledger.rows[-1]["status"] == "running":
            ledger.finish(call_id, "failed", {})
        progress("8146_generation_failed_" + call_id, 1, 0)
    finally:
        row.update(started_monotonic_ns=began, ended_monotonic_ns=time.monotonic_ns())
        row["full_latency_ns"] = row["ended_monotonic_ns"] - began
        row["numerator"] = row["full_latency_ns"] if row["status"] == "completed" else None
        row["latency_score_ns"] = row["numerator"]
        atomic_json(raw / "transaction.json", row)
    return row


def capture(
    data: Json,
    runtime: Any,
    native: Any,
    raw: Path,
    ledger: Ledger,
    *,
    deadline_s: float = 2200,
    started: float | None = None,
) -> Json:
    """Every original source retains both arms; warmups are explicitly excluded."""
    start = time.monotonic() if started is None else started
    work: Json = dict(pairs=[], warmups=[], config=CONFIG)
    schedule = [(True, s) for s in data["slots"][:4]] + [(False, s) for s in data["slots"]]
    for index, (warmup, slot) in enumerate(schedule):
        arms = (
            ARMS if int(slot["source_cluster_id"].split(":")[-1][-1], 16) % 2 == 0 else ARMS[::-1]
        )
        pair = dict(
            unit_id=slot["unit_id"],
            source_cluster_id=slot["source_cluster_id"],
            slot=slot["slot"],
            arms=[],
        )
        for arm in arms:
            reason = slot["exclusion_reason"] if not slot["public_eligible"] else None
            censored = time.monotonic() - start >= deadline_s
            call_id = f"{'warmup' if warmup else 'measured'}-{slot['slot']}-{arm}"
            if reason or censored:
                row = dict(
                    unit_id=call_id,
                    source_cluster_id=slot["source_cluster_id"],
                    arm=arm,
                    condition="fresh_cache_miss",
                    metric="complete_latency_ns",
                    numerator=None,
                    denominator=1,
                    status="excluded" if reason else "censored",
                    exclusion_reason=reason or "new_call_deadline",
                    started=False,
                )
            else:
                row = transaction(data, slot, arm, runtime, native, raw / call_id, ledger, call_id)
            row["excluded_warmup"] = warmup
            pair["arms"].append(row)
            if warmup:
                work["warmups"].append(row)
        if not warmup:
            work["pairs"].append(pair)
        atomic_json(raw / "primitive_rows.json", work)
        progress("8146_benchmark_pair_after", index + 1, len(schedule) - index - 1)
    return work


def reduce_rows(work: Json) -> Json:
    """Resample whole source pairs and remove only the measured arithmetic cost."""
    pairs = work.get("pairs", [])
    rows = [r for p in pairs for r in p["arms"]]
    good = [p for p in pairs if matched(p["arms"])]
    interval, ceiling = None, None
    if good:
        ratios, totals, arithmetic = [], [], []
        for pair in good:
            ordered = {r["arm"]: r for r in pair["arms"]}
            ratios.append(
                math.log(ordered[ARMS[0]]["full_latency_ns"] / ordered[ARMS[1]]["full_latency_ns"])
            )
            totals.extend(r["full_latency_ns"] for r in pair["arms"])
            arithmetic.extend(r["components"]["arithmetic_and_boundary_ns"] for r in pair["arms"])
        samples = (
            np.random.default_rng(CONFIG["seed"]).choice(ratios, (10000, len(ratios))).mean(axis=1)
        )
        lower = float(np.exp(np.quantile(samples, 0.05)))
        interval = dict(
            estimate=float(np.exp(np.mean(ratios))),
            lower95=lower,
            resamples=10000,
            sources=len(good),
            descriptive_only=len(good) != 24,
            speed_supported=len(good) == 24 and lower > 1,
            nfr01_met=len(good) == 24 and lower >= 10,
        )
        ceiling = dict(
            arithmetic_fraction=sum(arithmetic) / sum(totals),
            maximum_speedup=sum(totals) / (sum(totals) - sum(arithmetic)),
            all_host_costs_retained=True,
        )
    return dict(
        intended_count=48,
        eligible_count=sum(r["status"] != "excluded" for r in rows),
        independent_count=len(good),
        completed_count=sum(r["status"] == "completed" for r in rows),
        excluded_count=sum(r["status"] == "excluded" for r in rows),
        censored_count=sum(r["status"] == "censored" for r in rows),
        failed_count=sum(r["status"] == "failed" for r in rows),
        matched_pair_count=len(good),
        failed_pair_count=sum(
            all(r["status"] == "completed" for r in p["arms"]) and not matched(p["arms"])
            for p in pairs
        ),
        complete_service_ready_score=int(len(pairs) == len(good) == 24),
        paired_speed_interval=interval,
        zero_arithmetic_ceiling=ceiling,
        generated_output_parity=bool(good) and len(good) == len(pairs),
    )


def live(data: Json, raw: Path, scratch: Path) -> Json:
    """Reuse the CUDA process lease while recording only this run's real calls."""
    began = time.monotonic()
    legacy = qualified.legacy
    ledger = Ledger(raw / "ledger.json")
    result: Json = dict(checks=[], ledger=[], work={})
    spec = legacy.cached_current_model() or {}
    model = Path(spec.get("model_path", scratch / "missing"))
    binary = Path.home() / ".cache/llama.cpp-master/build/bin/llama-server"
    progress("8146_runtime_authentication_before")
    for path, field, expected, observed in [
        (model, "hf_id", stream.risk.MODEL, spec.get("hf_id")),
        (model, "model_revision", data["protocol"]["model_revision"], model.parent.name),
        (
            binary,
            "runtime_sha256",
            data["protocol"]["runtime_sha256"],
            sha256_file(binary) if binary.is_file() else None,
        ),
    ]:
        result["checks"].append(
            dict(
                check=field,
                upstream="runtime_identity",
                path=str(path),
                hash=observed if path == binary else None,
                artifact_field=field,
                op="==",
                expected=expected,
                observed=observed,
                passed=expected == observed,
            )
        )
    if not all(c["passed"] for c in result["checks"]):
        return result
    metadata = legacy.read_gguf_metadata(model)
    position = metadata["field_provenance"]["metadata_keys"]["tokenizer.chat_template"]
    with model.open("rb") as stream_file:
        stream_file.seek(position["value_offset"])
        size = struct.unpack("<Q", stream_file.read(8))[0]
        template = stream_file.read(size).decode()
    result["checks"].append(
        dict(
            check="embedded_chat_template_sha256",
            upstream="embedded_template",
            path=str(model),
            hash=None,
            artifact_field="embedded_chat_template_sha256",
            op="==",
            expected=data["protocol"]["chat_template_sha256"],
            observed=canonical_hash(template),
            passed=data["protocol"]["chat_template_sha256"] == canonical_hash(template),
        )
    )
    if not all(c["passed"] for c in result["checks"]):
        return result
    progress("8146_runtime_authentication_after", len(result["checks"]), 0)
    runtime_class = legacy.QwenRuntime

    class Recorded(runtime_class):  # type: ignore[misc,valid-type]
        def load(self) -> Json:
            """A failed load is still a current invocation, never historical work."""
            ledger.start("model_load", "owned-model-load", dict(model=str(model)))
            try:
                identity: Json = super().load()
                if (
                    canonical_hash(identity["props"]["chat_template"])
                    != data["protocol"]["chat_template_sha256"]
                ):
                    raise ValueError("served_template_drift")
            except (OSError, RuntimeError, TimeoutError, ValueError):
                ledger.finish("owned-model-load", "failed", {})
                raise
            ledger.finish("owned-model-load", "completed", identity)
            return identity

    def measured(
        frozen: list[Json], runtime: Any, path: Path, identity: str, *, started: float
    ) -> list[Json]:
        """Binding loading is startup; each request includes its actual crossings."""
        native, receipt = host.prior.old.host.load_binding(data)
        atomic_json(raw / "loaded_binding_receipt.json", receipt)
        result["startup_total_s"] = time.monotonic() - began
        result["work"] = capture(data, runtime, native, raw, ledger, started=started)
        return [r for p in result["work"]["pairs"] for r in p["arms"]]

    adapter = SimpleNamespace(freeze=lambda _: data["slots"], capture=measured)
    plan = dict(
        public_role_manifests={},
        capture_identity=canonical_hash(data["slots"]),
        protocol=data["protocol"],
    )
    with (
        patch.object(legacy, "capture", adapter),
        patch.object(legacy, "load_public", lambda _: {}),
        patch.object(legacy, "QwenRuntime", Recorded),
        patch.object(legacy, "TASK", "exp8146-live-service-cost"),
    ):
        measured_result = legacy.live_capture(plan, raw, scratch)
    result.update({k: v for k, v in measured_result.items() if k not in {"rows", "checks"}})
    result["ledger"] = ledger.rows
    for check in measured_result["checks"]:
        result["checks"].append(
            dict(
                check,
                artifact_field=check["field"],
                check=check["upstream_id"] + "_" + check["field"],
                upstream=check["upstream_id"],
            )
        )
    return result


def checksum(value: Json) -> str:
    """Bind all reduced fields without making the checksum depend on itself."""
    return canonical_hash({k: v for k, v in value.items() if k != "reproducibility_checksum"})


def build(
    data: Json, result: Json, raw: Path, receipts: list[Json], duration: float, fixture: bool
) -> Json:
    """Only complete current natural pairs and normal owned checks earn readiness."""
    work = result.get("work") or dict(
        pairs=[
            dict(
                unit_id=f"retention-{i}",
                arms=[
                    dict(
                        unit_id=f"retention-{i}-{arm}",
                        source_cluster_id=f"missing-{i}",
                        arm=arm,
                        condition="fresh_cache_miss",
                        metric="complete_latency_ns",
                        numerator=None,
                        denominator=1,
                        status="excluded",
                        exclusion_reason="external_upstream",
                        started=False,
                    )
                    for arm in ARMS
                ],
            )
            for i in range(1, 25)
        ],
        warmups=[],
        config=CONFIG,
    )
    reduced = reduce_rows(work)
    checks = [*data["checks"], *result.get("checks", [])]
    failed = [c for c in checks if not c["passed"]]
    checked = all(r["passed"] and r["normal_exit"] for r in receipts)
    ledger = Ledger(raw / "unused-ledger.json")
    ledger.rows = [] if fixture else result.get("ledger", [])
    counts = ledger.counts()
    generated, loaded = (
        counts["generation_calls_attempted"] > 0,
        counts["model_loads_attempted"] > 0,
    )
    klass = (
        "disqualified"
        if not checked
        else "blocked"
        if failed
        else "circular_positive"
        if fixture
        else "null"
    )
    cuda = result.get("model_identity_receipt", {}).get("authenticated", False) and bool(
        result.get("gpu_lease_receipt")
    )
    ready = int(
        reduced["complete_service_ready_score"]
        and checked
        and not failed
        and not fixture
        and cuda
        and generated
        and result.get("measured_duration_s", 0) >= 10
    )
    atomic_json(raw / "primitive_rows.json", work)
    atomic_json(raw / "input_data.json", data)
    atomic_json(raw / "independent_reduction.json", reduced)
    atomic_json(
        raw / "current_execution.json",
        dict(ledger=ledger.rows, runtime=result.get("runtime_receipts", [])),
    )
    rows = [r for p in work["pairs"] for r in p["arms"]]
    load_cost = result.get("model_identity_receipt", {}).get("duration_s", 0)
    value: Json = dict(
        reduced,
        experiment_id=8146,
        task_id="exp8146-live-service-cost",
        schema="carnot.live_service_cost.v1",
        run_date="20261005",
        honest_verdict="complete_"
        + klass
        + "_"
        + (failed[0]["check"] if failed else "bounded_live_service_cost"),
        verdict_class=klass,
        verifier_is_oracle=fixture,
        claim_scope="bounded retention-source complete miss latency; no correctness claim",
        exposure_scope="previously exposed development retention sources",
        independent_generalization_score=0,
        generalized_learning_benefit_score=0,
        required_checks_passed=checked,
        flagged_adversarial=False,
        validation_receipts=receipts,
        terminal_validation_sidecar_path=str(raw / "terminal_validation.json"),
        gate_check_summary=checks,
        preconditions_checked=checks,
        inference_substrate="live_llm_inference"
        if generated or loaded
        else "aggregation_from_upstream_artifacts",
        inference_substrate_class="model_bounded_generation"
        if generated
        else "model_load_no_generation"
        if loaded
        else "no_model_load",
        MODEL_SPECS=MODEL_SPECS if generated or loaded else [],
        model_invocation_counts=counts,
        call_ledger=ledger.rows,
        trained_head_specs=data.get("trained_head_specs", []),
        rows=rows,
        sample_size_budget=dict(CONFIG, intended=48, independent_sources=24),
        duration_s=duration,
        random_seed=CONFIG["seed"],
        source_artifact_hashes=data["refs"],
        raw_shard_hashes=[
            reference(raw / p)
            for p in [
                "primitive_rows.json",
                "input_data.json",
                "current_execution.json",
                "independent_reduction.json",
            ]
        ],
        code_config_hashes={
            p: sha256_file(ROOT / p)
            for p in [
                MODULE,
                CLI,
                TEST,
                "python/carnot/experiment_8145_v704_natural_service_cost.py",
                "python/carnot/verify/qwen_learning_stream_capture_8102.py",
                "python/carnot/inference/qwen_sufficiency_7920.py",
                "python/carnot/verify/evidence_features_7980.py",
                "python/carnot/verify/learning_protocol_8138.py",
                "python/carnot/verify/native_radial_8105.py",
                "python/carnot/inference/llama_cpp_process.py",
            ]
        },
        phase_spans=[
            dict(
                phase=r["call_id"],
                start_ns=r["started_monotonic_ns"],
                end_ns=r["ended_monotonic_ns"],
            )
            for r in ledger.rows
            if r["operation"] == "model_load"
        ]
        + [
            dict(
                phase=r["unit_id"],
                start_ns=r.get("started_monotonic_ns"),
                end_ns=r.get("ended_monotonic_ns"),
            )
            for r in rows
            if r["started"]
        ],
        complete_service_ready_score=ready,
        request_pair_rows=work["pairs"],
        complete_latency_rows=rows,
        warmup_rows=work["warmups"],
        model_load_cost=dict(
            duration_s=load_cost, receipt=result.get("model_identity_receipt", {})
        ),
        startup_cost=dict(
            duration_s=result.get("startup_total_s"), includes_model_load_and_binding=True
        ),
        total_without_startup_s=sum(r.get("full_latency_ns", 0) for r in rows) / 1e9,
        total_with_startup_s=(
            result.get("startup_total_s", load_cost)
            + sum(r.get("full_latency_ns", 0) for r in rows) / 1e9
        )
        if loaded
        else None,
        runtime_identity=result,
        fixture_mode=fixture,
        gpu_memory_delta_mb=(
            int(result["resident_gpu_receipt"]["output_tail"].strip())
            - int(result.get("unloaded_gpu_receipt", {}).get("output_tail", "0").strip())
        )
        if result.get("resident_gpu_receipt")
        else None,
        acceptance_gates=dict(
            complete_pairs=24,
            parity_tolerance=1e-10,
            lower95_speed=1,
            lower95_nfr01=10,
            duration_floor_s=10,
        ),
        cited_upstream_artifacts=data["refs"],
        methodology_note="Fresh calls use original public bytes and sealed nine-feature natural coefficients. Missing sources remain in the fixed panel. Warmups are excluded. No labels or independent generalization are evaluated.",
    )
    value["field_principles"] = {
        k: "Exact current bytes and original source denominators bound every claim; imported history supplies no current invocation."
        for k in value
    }
    value["reproducibility_checksum"] = checksum(value)
    return value


def replay(path: Path) -> bool:
    """Reopen primitives and independently score each generated public response."""
    try:
        value = json.loads(path.read_text())
        if checksum(value) != value["reproducibility_checksum"]:
            return False
        for ref in [*value["source_artifact_hashes"], *value["raw_shard_hashes"]]:
            if sha256_file(Path(ref["path"])) != ref["sha256"]:
                return False
        if any(sha256_file(ROOT / p) != h for p, h in value["code_config_hashes"].items()):
            return False
        work, data, execution = [
            json.loads(Path(r["path"]).read_text()) for r in value["raw_shard_hashes"][:3]
        ]
        reduced = reduce_rows(work)
        if json.loads(Path(value["raw_shard_hashes"][3]["path"]).read_text()) != reduced:
            return False
        for field in reduced:
            if field != "complete_service_ready_score" and value[field] != reduced[field]:
                return False
        rows = [r for p in work["pairs"] for r in p["arms"]]
        if value["rows"] != rows or value["request_pair_rows"] != work["pairs"]:
            return False
        ledger = Ledger(Path(value["raw_shard_hashes"][2]["path"]).parent / "unused-replay.json")
        ledger.rows = execution["ledger"]
        if (
            value["call_ledger"] != ledger.rows
            or value["model_invocation_counts"] != ledger.counts()
        ):
            return False
        events = {r["call_id"]: r for r in ledger.rows}
        slots = {r["source_cluster_id"]: r for r in data["slots"]}
        for row in [*rows, *work["warmups"]]:
            if row["status"] != "completed":
                continue
            slot = slots[row["source_cluster_id"]]
            parsed = stream.risk.transport.parse_response(row["raw_response"], slot["visible_ids"])
            body = json.loads(slot["request"]["messages"][1]["content"])
            features = stream.lexical.extract(
                dict(
                    family_id=slot["family_id"],
                    source_bytes=body["complete_source"].encode().hex(),
                    answer_bytes=body["original_answer"].encode().hex(),
                )
            )
            p = min(1 - 1e-6, max(1e-6, parsed["probability"]))
            values = [math.log(p / (1 - p)), *features["values"]]
            probability = host.engine.scalar_probability(data["head"], data["geometry"], values)
            commit = row["response_commit"]
            if (
                row["request"] != slot["request"]
                or row["generated_text"] != row["raw_response"]["choices"][0]["message"]["content"]
                or row["values"] != values
                or abs(probability - row["probability"]) > 1e-10
                or row["decision"] != host.engine.historical.radial.action(probability)
                or sha256_file(Path(commit["path"])) != commit["sha256"]
                or json.loads(Path(commit["path"]).read_text())["response_bytes"]
                != row["response_bytes"]
                or row["full_latency_ns"] != row["ended_monotonic_ns"] - row["started_monotonic_ns"]
                or row["full_latency_ns"] < sum(row["components"].values())
            ):
                return False
            if not value["fixture_mode"]:
                event = events[row["unit_id"]]
                if event["request_sha256"] != canonical_hash(row["request"]) or event[
                    "response_sha256"
                ] != canonical_hash(row["raw_response"]):
                    return False
        expected_ready = int(
            reduced["complete_service_ready_score"]
            and value["required_checks_passed"]
            and not value["fixture_mode"]
            and all(c["passed"] for c in value["gate_check_summary"])
            and value["runtime_identity"]
            .get("model_identity_receipt", {})
            .get("authenticated", False)
            and bool(value["runtime_identity"].get("gpu_lease_receipt"))
            and ledger.counts()["generation_calls_attempted"] > 0
            and value["runtime_identity"].get("measured_duration_s", 0) >= 10
        )
        return value["complete_service_ready_score"] == expected_ready and all(
            not r.get("log_path") or sha256_file(Path(r["log_path"])) == r["log_sha256"]
            for r in value["validation_receipts"]
        )
    except (OSError, ValueError, KeyError, TypeError):
        return False


def validation_plan(private: Path) -> list[CommandSpec]:
    """Freeze explicit owned, consumer and private E2E checks before model work."""
    tests = [
        TEST,
        "tests/python/test_primary_publication_7928.py",
        "tests/python/test_source_boundary_7852.py",
        "tests/python/test_experiment_7942_v689_sentence_labels.py",
    ]
    commands = build_scoped_commands(
        ROOT,
        tests,
        [MODULE],
        static_paths=[CLI],
        basetemp=private,
        coverage_file=private / ".coverage",
    )
    return [
        CommandSpec(
            c.name,
            (*c.argv, "--strict", "--follow-imports=silent")
            if c.name == "changed_module_mypy"
            else c.argv,
            c.scope,
            300,
        )
        for c in commands
    ]


def validators(path: Path) -> list[CommandSpec]:
    """Unmodified auditors must finish normally and authenticate the candidate."""
    py = str(ROOT / ".venv/bin/python")
    return [
        CommandSpec(
            "cold_replay", (py, "-u", str(ROOT / CLI), "--cold-replay", str(path)), "terminal", 300
        ),
        CommandSpec(
            "adversarial",
            (py, str(ROOT / "scripts/adversarial_verify.py"), "--json", str(path)),
            "terminal",
            300,
        ),
        CommandSpec(
            "strict_rows",
            (py, str(ROOT / "scripts/verdict_row_consistency_lint.py"), "--strict", str(path)),
            "terminal",
            300,
        ),
    ]


def main(argv: list[str] | None = None) -> int:
    """Qualify owned checks, capture once and publish independently checked bytes."""
    os.environ["PYTHONUNBUFFERED"] = "1"
    began = time.monotonic()
    progress("8146_start")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", choices=["20261005"], default="20261005")
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, default=ROOT / "results" / (NAME + ".json"))
    parser.add_argument("--private-small", action="store_true")
    parser.add_argument("--cold-replay", type=Path)
    args = parser.parse_args(argv)
    if args.cold_replay:
        passed = replay(args.cold_replay)
        progress("8146_replay_passed" if passed else "8146_replay_failed")
        return 0 if passed else 1
    output = args.output.absolute()
    if args.private_small and output.is_relative_to(ROOT / "results"):
        parser.error("private fixtures require output outside results")
    raw = output.parent / "raw" / output.stem / "invocations" / str(time.time_ns())
    private = Path(tempfile.mkdtemp(prefix="carnot-8146-validation-"))
    plan = validation_plan(private)
    atomic_json(
        raw / "validation_commands.json",
        dict(commands=[asdict(c) for c in plan], config=CONFIG, frozen_before_measurement=True),
    )
    receipts = [] if args.private_small else host.prior.execute(plan, private)
    data = inputs(args.root, raw)
    result: Json = {}
    if data["ready"] and all(r["passed"] for r in receipts):
        if args.private_small:
            native = host.prior.old.host.load_binding(data)[0]
            result = dict(
                work=capture(
                    data, FixtureRuntime(), native, raw, Ledger(raw / "fixture-ledger.json")
                )
            )
        else:
            progress("8146_capture_before")
            result = live(data, raw, private)
            progress("8146_capture_after")
    value = build(data, result, raw, receipts, time.monotonic() - began, args.private_small)
    if not args.private_small:
        value["repository_health"] = host.prior.execute(
            [
                CommandSpec(
                    "repository_health_once",
                    (str(ROOT / ".venv/bin/pytest"), "tests/python", "-q"),
                    "repository_health_not_science_gate",
                    120,
                )
            ],
            private / "health",
        )
    value["reproducibility_checksum"] = checksum(value)
    candidate = private / (NAME + ".json")
    atomic_json(candidate, value)
    terminal_receipts = host.prior.execute(validators(candidate), private / "terminal")
    if not all(r["passed"] and r["normal_exit"] for r in terminal_receipts):
        atomic_json(raw / "failed_terminal_candidate.json", value)
        atomic_json(
            raw / "terminal_validation.json", dict(passed=False, receipts=terminal_receipts)
        )
        progress("8146_terminal_validation_failed")
        return 1
    value["validation_receipts"] += terminal_receipts
    value["duration_s"] = time.monotonic() - began
    value["reproducibility_checksum"] = checksum(value)

    def terminal(path: Path) -> Json:
        checks = host.prior.execute(validators(path), private / "publication")
        return dict(passed=all(r["passed"] and r["normal_exit"] for r in checks), receipts=checks)

    if output.exists():
        atomic_json(raw / "preserved_primary.json", json.loads(output.read_text()))
    publication = publish_primary(output, value, terminal)
    atomic_json(
        raw / "terminal_validation.json", dict(publication=publication, normal_process_exit=True)
    )
    progress("8146_complete", value["completed_count"], value["censored_count"])
    return 0
