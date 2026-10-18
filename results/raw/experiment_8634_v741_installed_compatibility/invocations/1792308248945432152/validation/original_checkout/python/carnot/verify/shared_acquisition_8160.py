"""REQ-VERIFY-8160: compare durable branches using one captured output per source.

Common acquisition is charged to each arm. This keeps arithmetic comparisons
paired without claiming that separate model calls returned identical bytes.
"""

from __future__ import annotations

from copy import deepcopy
import json
import math
import os
from pathlib import Path
import time
from typing import Any
from unittest.mock import patch

import numpy as np

from carnot import experiment_8146_v704_live_service_cost as prior
from carnot.verify import durable_batch_8159 as host
from carnot.reporting.primary_publication import read_bound_sidecar

Json = dict[str, Any]
ROOT = prior.ROOT
NAME = "experiment_8160_v705_shared_acquisition_cost"
CLI = f"scripts/experiments/{NAME}.py"
TEST = "tests/python/test_shared_acquisition_8160.py"
OWNED = [
    "python/carnot/verify/shared_acquisition_8160.py",
    "python/carnot/reporting/shared_acquisition_execution_8160.py",
    CLI,
]
MODEL_SPECS = ["unsloth/Qwen3.8-27B-GGUF"]
CONFIG = dict(
    seed=7058160,
    measured_calls=32,
    warmup_calls=8,
    call_limit=40,
    max_tokens=128,
    input_tokens=6000,
    load_timeout_s=300,
    call_timeout_s=120,
    latest_launch_s=3000,
    closure_s=3120,
    bootstrap_draws=10000,
    startup_amortization_requests=32,
)
atomic_json, canonical_hash, sha256_file = (
    prior.atomic_json,
    prior.canonical_hash,
    prior.sha256_file,
)
reference, checksum, Ledger = prior.reference, prior.checksum, prior.Ledger
progress = prior.progress


def inputs(root: Path, raw: Path) -> Json:
    """Authenticate qualified history without changing any predecessor artifact."""
    raw.mkdir(parents=True, exist_ok=True)
    data = prior.inputs(root, raw)
    path = root / "results/experiment_8159_v705_durable_batch_service.json"
    host.old.gate(data, path, "host_batch_resource_exists", True, path.is_file())
    if path.is_file():
        try:
            value = json.loads(path.read_text())
            for field, expected in [
                ("host_batch_ready_score", 1),
                ("required_checks_passed", True),
                ("flagged_adversarial", False),
            ]:
                host.old.gate(data, path, field, expected, value.get(field))
            terminal = Path(value["terminal_validation_sidecar_path"])
            sidecar = Path(json.loads(terminal.read_text())["publication"]["sidecar_path"])
            host.old.gate(
                data,
                path,
                "host_terminal_passed",
                True,
                read_bound_sidecar(path, sidecar)["report"]["passed"],
            )
            refs = [
                reference(path),
                value["primitive_rows"],
                reference(sidecar),
                reference(
                    root / "openspec/change-proposals/research-roadmap-v704-preserved-20261005.md"
                ),
            ]
            for ref in refs:
                host.old.gate(
                    data, Path(ref["path"]), "sha256", ref["sha256"], sha256_file(Path(ref["path"]))
                )
            data["refs"].extend(refs)
            historical = json.loads((root / prior.HISTORY).read_text())
            manifest = historical["capture_manifest"]
            data["slots"] = deepcopy(
                [
                    s
                    for s in json.loads(Path(manifest["path"]).read_text())["rows"]
                    if s["role"] == "retention"
                ]
            )
        except (OSError, ValueError, KeyError) as error:
            host.old.gate(data, path, "host_authentication", True, str(error))
    data["ready"] = all(r["passed"] for r in data["checks"])
    atomic_json(raw / "input_data.json", data)
    return data


def tokenizer_counts(data: Json) -> tuple[Json, list[int]]:
    """Use only embedded vocabulary and public prompts before GPU weights load."""
    from llama_cpp import Llama
    from llama_cpp.llama_chat_format import Jinja2ChatFormatter

    legacy = prior.qualified.legacy
    spec = legacy.cached_current_model() or {}
    model = Path(spec.get("model_path", "/missing-qwen"))
    binary = Path.home() / ".cache/llama.cpp-master/build/bin/llama-server"
    progress("8160_model_hash_before", 0, 1)
    digest = legacy.bounded(lambda: sha256_file(model), 120)
    if (
        spec.get("hf_id") != MODEL_SPECS[0]
        or digest != data["protocol"]["gguf_sha256"]
        or model.parent.name != data["protocol"]["model_revision"]
        or sha256_file(binary) != data["protocol"]["runtime_sha256"]
    ):
        raise ValueError("runtime_identity_drift")
    progress("8160_tokenizer_load_before", 0, 1)
    tokenizer = Llama(model_path=str(model), vocab_only=True, n_gpu_layers=0, verbose=False)
    try:
        template = tokenizer.metadata["tokenizer.chat_template"]
        if canonical_hash(template) != data["protocol"]["chat_template_sha256"]:
            raise ValueError("embedded_template_drift")
        formatter = Jinja2ChatFormatter(template, eos_token="<|im_end|>", bos_token="")
        counts = [
            len(
                tokenizer.tokenize(
                    formatter(
                        messages=s["request"]["messages"], enable_thinking=False
                    ).prompt.encode(),
                    add_bos=True,
                    special=True,
                )
            )
            for s in data["slots"]
        ]
        identity = dict(
            model_path=str(model),
            model_revision=model.parent.name,
            gguf_shards=[dict(path=str(model), sha256=digest)],
            tokenizer="embedded_GGUF",
            tokenizer_sha256=canonical_hash(tokenizer.metadata),
            chat_template_sha256=canonical_hash(template),
            runtime=reference(binary),
            offload_configuration=dict(n_gpu_layers=99, context=8192, parallel=1),
        )
    finally:
        tokenizer.close()
    progress("8160_tokenizer_load_after", 1, 0)
    return identity, counts


def seal(data: Json, raw: Path, fixture: bool = False) -> None:
    """Select before output access and retain empty positions instead of topping up."""
    if not data["ready"]:
        return
    progress("8160_seal_before", 0, 32)
    try:
        identity, counts = (
            (dict(tokenizer="private_fixture"), [1] * len(data["slots"]))
            if fixture
            else tokenizer_counts(data)
        )
        selected = []
        seen: set[str] = set()
        public_checks = []
        for slot, count in zip(data["slots"], counts, strict=True):
            eligible = (
                slot["public_eligible"] and count <= 6000 and slot["source_cluster_id"] not in seen
            )
            public_checks.append(
                dict(
                    slot=slot["slot"],
                    input_tokens=count,
                    eligible=eligible,
                    exclusion_reason=None
                    if eligible
                    else "context_over_budget"
                    if count > 6000
                    else slot["exclusion_reason"] or "duplicate_source",
                )
            )
            if eligible and len(selected) < 32:
                seen.add(slot["source_cluster_id"])
                selected.append(dict(slot, preflight_input_tokens=count))
        for index in range(len(selected), 32):
            selected.append(
                dict(
                    unit_id=f"missing-{index + 1}",
                    source_cluster_id=f"missing-{index + 1}",
                    slot=index + 1,
                    public_eligible=False,
                    exclusion_reason="missing_public_eligible_source",
                )
            )
        for s in selected:
            if s["public_eligible"]:
                s["request"].update(max_tokens=128, cache_prompt=False)
        data.update(slots=selected, runtime_freeze=identity, public_tokenizer_checks=public_checks)
    except (OSError, ImportError, RuntimeError, ValueError, KeyError) as error:
        host.old.gate(
            data,
            raw / "runtime_freeze.json",
            "public_tokenizer_preflight",
            True,
            f"{type(error).__name__}:{error}",
        )
        data["ready"] = False
    atomic_json(raw / "input_data.json", data)
    progress("8160_seal_after", len(data["slots"]), 0)


def acquire(slot: Json, runtime: Any, raw: Path, ledger: Any, call_id: str) -> Json:
    """Save the actual generated bytes once, including transport and parsing losses."""
    raw.mkdir(parents=True, exist_ok=True)
    row = dict(
        unit_id=slot["unit_id"],
        source_cluster_id=slot["source_cluster_id"],
        slot=slot["slot"],
        arm="shared_acquisition",
        condition="current_capture",
        metric="acquisition_ns",
        numerator=None,
        denominator=1,
        status="failed",
        exclusion_reason=None,
        request=slot["request"],
    )
    began = time.monotonic_ns()
    ledger.start("generation", call_id, slot["request"])
    progress("8160_generation_before_" + call_id, 0, 1)
    try:
        response = runtime.generate(slot["request"])
        ledger.finish(call_id, "completed", response)
        row.update(
            raw_response=response,
            input_tokens=response.get("usage", {}).get("prompt_tokens", 0),
            output_tokens=response.get("usage", {}).get("completion_tokens", 0),
        )
        ledger.rows[-1]["input_tokens"] = row["input_tokens"]
        ledger.save()
        text = response["choices"][0]["message"]["content"]
        path = raw / "captured_output.bin"
        path.write_bytes(text.encode())
        row.update(capture_path=str(path), capture_sha256=sha256_file(path))
        parsed = prior.stream.risk.transport.parse_response(response, slot["visible_ids"])
        body = json.loads(slot["request"]["messages"][1]["content"])
        lexical = prior.stream.lexical.extract(
            dict(
                family_id=slot["family_id"],
                source_bytes=body["complete_source"].encode().hex(),
                answer_bytes=body["original_answer"].encode().hex(),
            )
        )
        if row["input_tokens"] > 6000 or row["output_tokens"] > 128:
            raise ValueError("actual_token_budget")
        if not parsed["completed"] or lexical["values"] is None:
            raise ValueError("unusable_parse_or_features")
        p = min(1 - 1e-6, max(1e-6, parsed["probability"]))
        row.update(
            values=[math.log(p / (1 - p)), *lexical["values"]],
            parsed=parsed,
            status="completed",
            generated_text=text,
        )
    except (OSError, RuntimeError, TimeoutError, ValueError, KeyError) as error:
        row["exclusion_reason"] = f"{type(error).__name__}:{error}"
        if ledger.rows[-1]["status"] == "running":
            ledger.finish(call_id, "failed", {})
    row.update(started_monotonic_ns=began, ended_monotonic_ns=time.monotonic_ns())
    row["acquisition_ns"] = row["ended_monotonic_ns"] - began
    row["numerator"] = row["acquisition_ns"] if row["status"] == "completed" else None
    row["acquisition_score_ns"] = row["numerator"]
    atomic_json(raw / "acquisition.json", row)
    progress("8160_generation_after_" + call_id, 1, 0)
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
    """Time captured inputs through the qualified queue, transaction and commit code."""
    began = time.monotonic() if started is None else started
    work: Json = dict(captures=[], warmups=[], host_groups=[], references=[], pairs=[])
    eligible = [s for s in data["slots"] if s["public_eligible"]]
    warmups = [eligible[i % len(eligible)] for i in range(8)] if eligible else []
    schedule = [(True, s) for s in warmups] + [(False, s) for s in data["slots"]]
    for index, (warmup, slot) in enumerate(schedule):
        late = slot["public_eligible"] and time.monotonic() - began >= CONFIG["latest_launch_s"]
        if not slot["public_eligible"] or late:
            row = dict(
                unit_id=slot["unit_id"],
                source_cluster_id=slot["source_cluster_id"],
                slot=slot["slot"],
                status="censored" if late else "excluded",
                exclusion_reason="launch_cutoff" if late else slot["exclusion_reason"],
                acquisition_ns=0,
                numerator=None,
                acquisition_score_ns=None,
                denominator=1,
                arm="shared_acquisition",
                condition="current_capture",
                metric="acquisition_ns",
            )
        else:
            row = acquire(slot, runtime, raw / f"call-{index}", ledger, f"call-{index}")
            if not warmup and row["status"] == "completed" and not work["references"]:
                progress("8160_reference_benchmark_before", 0, 1)
                reference_data = dict(data, public=[row])
                service = host.transaction(
                    reference_data,
                    native,
                    dict(
                        unit_id="actual_reference",
                        batch=1,
                        repetition=0,
                        condition="cold",
                        arrival="all_at_once",
                    ),
                    host.ARMS[0],
                    raw / "reference",
                )
                work["references"].append(
                    dict(
                        unit_id=row["unit_id"],
                        source_cluster_id=row["source_cluster_id"],
                        arm=host.ARMS[0],
                        capture_sha256=row["capture_sha256"],
                        service=service,
                        measured_end_to_end_ns=time.monotonic_ns() - row["started_monotonic_ns"],
                    )
                )
                progress("8160_reference_benchmark_after", 1, 0)
        work["warmups" if warmup else "captures"].append(row)
        progress("8160_capture_count", index + 1, len(schedule) - index - 1)
    usable = [r for r in work["captures"] if r["status"] == "completed"]
    if usable and time.monotonic() - began < CONFIG["closure_s"]:
        branch_data = dict(data, public=usable)
        for condition in host.CONFIG["conditions"]:
            for arrival in host.CONFIG["arrivals"]:
                slot = dict(
                    unit_id=f"shared-{condition}-{arrival}",
                    batch=len(usable),
                    repetition=0,
                    condition=condition,
                    arrival=arrival,
                )
                arms = []
                order = host.ARMS if len(work["host_groups"]) % 2 == 0 else host.ARMS[::-1]
                for arm in order:
                    progress("8160_host_benchmark_before_" + arm, len(arms), 4 - len(arms))
                    host_start = time.monotonic_ns()
                    measured = host.transaction(
                        branch_data, native, slot, arm, raw / "stores" / slot["unit_id"] / arm
                    )
                    elapsed = time.monotonic_ns() - host_start
                    measured["components"]["caller_preparation_and_recovery_ns"] = (
                        elapsed - measured["duration_ns"]
                    )
                    measured["duration_ns"] = elapsed
                    arms.append(measured)
                    progress("8160_host_benchmark_after_" + arm, len(arms), 4 - len(arms))
                work["host_groups"].append(dict(slot, arms=arms))
    atomic_json(raw / "primitive_rows.json", work)
    return work


def live(data: Json, raw: Path, scratch: Path) -> Json:
    """Reuse the owned CUDA lease and worker while replacing only its capture schedule."""
    if os.environ.get("CARNOT_FORCE_LIVE") != "1":
        host.old.gate(
            data,
            raw / "live_environment",
            "CARNOT_FORCE_LIVE",
            "1",
            os.environ.get("CARNOT_FORCE_LIVE"),
        )
        data["ready"] = False
        return dict(work={}, ledger=[], checks=[])
    runtime_class = prior.qualified.legacy.QwenRuntime
    original_load = runtime_class.load

    def deadline_load(runtime: Any) -> Json:
        """Bound the whole load, including identity hashing, to the declared deadline."""
        return dict(prior.qualified.legacy.bounded(lambda: original_load(runtime), 300))

    def ledger_progress(phase: str, started: float, units: int = 0) -> None:
        """Report completed and in-flight calls while the owned child is waiting."""
        counts = Ledger(raw / "ledger.json").counts()
        progress(
            "8160_" + phase,
            counts["generation_calls_completed"] + counts["model_loads_completed"],
            counts["generation_calls_in_flight"] + counts["model_loads_in_flight"],
        )

    with (
        patch.object(prior, "capture", measure),
        patch.object(runtime_class, "load", deadline_load),
        patch.object(prior.qualified.legacy, "progress", ledger_progress),
    ):
        return dict(prior.live(data, raw, scratch))


def reduce_rows(work: Json) -> Json:
    """Reduce original sources and retain common costs in every paired denominator."""
    captures = work.get("captures", [])
    usable = {r["source_cluster_id"]: r for r in captures if r["status"] == "completed"}
    amortized = (
        work.get("startup_ns", 0) + sum(r.get("acquisition_ns", 0) for r in work.get("warmups", []))
    ) / 32
    composed, intervals = [], []
    arithmetic, total = 0.0, 0.0
    parity = all(host.parity(g["arms"]) for g in work.get("host_groups", []))
    for group in work.get("host_groups", []):
        branch_rows: Json = {}
        for arm in group["arms"]:
            n = len(arm["requests"])
            envelope = max(r["response_ns"] for r in arm["requests"]) - min(
                r["enqueue_ns"] for r in arm["requests"]
            )
            overhead = max(0, arm["duration_ns"] - envelope) / n
            branch_rows[arm["arm"]] = {}
            for source in captures:
                sid = source["source_cluster_id"]
                request = next((r for r in arm["requests"] if r["source_cluster_id"] == sid), None)
                cost = (
                    source.get("acquisition_ns", 0) + request["latency_ns"] + overhead + amortized
                    if request
                    else None
                )
                row = dict(
                    unit_id=source["unit_id"],
                    source_cluster_id=sid,
                    arm=arm["arm"],
                    condition=group["condition"],
                    arrival=group["arrival"],
                    metric="composed_request_cost_ns",
                    numerator=cost,
                    denominator=1,
                    status=source["status"],
                    exclusion_reason=source["exclusion_reason"],
                    acquisition_ns=source.get("acquisition_ns", 0),
                    host_ns=request["latency_ns"] + overhead if request else None,
                    composed_request_cost_ns=cost - amortized if request else None,
                    amortized_composed_request_cost_ns=cost,
                    startup_warmup_amortized_ns=amortized,
                    capture_sha256=source.get("capture_sha256"),
                    acquisition_physically_rerun=False,
                )
                composed.append(row)
                if request:
                    branch_rows[arm["arm"]][sid] = cost
                    total += cost
                    arithmetic += arm["components"]["arithmetic_ns"] / n
        baseline = branch_rows[host.ARMS[0]]
        for arm in host.ARMS[1:]:
            ratios = [math.log(baseline[s] / branch_rows[arm][s]) for s in usable]
            draws = (
                np.random.default_rng(CONFIG["seed"])
                .choice(ratios, (10000, len(ratios)))
                .mean(axis=1)
            )
            lower = float(np.exp(np.quantile(draws, 0.05)))
            intervals.append(
                dict(
                    arm=arm,
                    condition=group["condition"],
                    arrival=group["arrival"],
                    estimate=float(np.exp(np.mean(ratios))),
                    lower95=lower,
                    resamples=10000,
                    sources=len(usable),
                    speed_supported=len(usable) >= 24 and parity and lower > 1,
                    nfr01_met=False,
                    comparison_scope="component_composed_descriptive",
                )
            )
    return dict(
        rows=captures,
        composed_cost_rows=composed,
        paired_speed_intervals=intervals,
        zero_arithmetic_ceiling=dict(
            arithmetic_fraction=arithmetic / total,
            maximum_speedup=total / (total - arithmetic),
            all_host_costs_retained=True,
        )
        if total
        else None,
        intended_count=32,
        eligible_count=sum(r["status"] != "excluded" for r in captures),
        independent_count=len(usable),
        completed_count=sum(r["status"] == "completed" for r in captures),
        excluded_count=32 - len(captures) + sum(r["status"] == "excluded" for r in captures),
        censored_count=sum(r["status"] == "censored" for r in captures),
        failed_count=sum(r["status"] == "failed" for r in captures),
        acquisition_composition_ready_score=int(
            len(captures) == 32
            and len(usable) >= 24
            and len(work.get("host_groups", [])) == 6
            and parity
        ),
        generated_output_parity=parity,
    )


def build(
    data: Json,
    result: Json,
    raw: Path,
    receipts: list[Json],
    date: str,
    duration: float,
    fixture: bool,
) -> Json:
    """Publish completed losses honestly and keep imported calls outside current counts."""
    work = result.get("work") or dict(
        captures=[], warmups=[], host_groups=[], references=[], pairs=[]
    )
    work["startup_ns"] = int(result.get("startup_total_s", 0) * 1e9)
    atomic_json(raw / "primitive_rows.json", work)
    reduction = reduce_rows(work)
    ledger = Ledger(raw / "build-ledger.json")
    ledger.rows = result.get("ledger", [])
    counts = dict(prior.ZERO_INVOCATION_COUNTS) if fixture else ledger.counts()
    checks = data["checks"] + result.get("checks", [])
    checked = bool(receipts) and all(r.get("passed") and r.get("normal_exit") for r in receipts)
    blocked = next((r["check"] for r in checks if not r["passed"]), None)
    ready = bool(
        reduction["acquisition_composition_ready_score"]
        and checked
        and not blocked
        and not fixture
        and counts["model_loads_completed"] == 1
        and counts["generation_calls_completed"] >= 24
        and result.get("gpu_lease_receipt")
        and duration >= 10
    )
    verdict = "circular_positive" if fixture else "null"
    if blocked:
        verdict = "blocked"
    if not checked:
        verdict = "disqualified"
    honest = (
        "complete_blocked_" + blocked
        if verdict == "blocked"
        else "complete_" + verdict + "_shared_acquisition_cost"
    )
    value = dict(
        experiment_id=8160,
        task_id="exp8160-shared-acquisition-cost",
        schema="carnot.shared_acquisition_cost.v1",
        honest_verdict=honest,
        verdict_class=verdict,
        verifier_is_oracle=fixture,
        claim_scope="Exposed current acquisition plus matched frozen durable host branches; component composition only; no NFR-01 deployment or learning claim",
        exposure_scope="exposed_historical_development",
        independent_generalization_score=0,
        generalized_learning_benefit_score=0,
        required_checks_passed=checked,
        flagged_adversarial=False,
        validation_receipts=receipts,
        terminal_validation_sidecar_path=str(raw / "terminal_validation.json"),
        preconditions_checked=True,
        gate_check_summary=checks,
        inference_substrate="private_transport_fixture"
        if fixture
        else "live_qwen_bounded_generation"
        if counts["generation_calls_completed"]
        else "model_load_no_generation"
        if counts["model_loads_completed"]
        else "no_model_execution",
        inference_substrate_class="model_bounded_generation"
        if counts["model_loads_attempted"]
        else "no_model_load",
        declared_inference_substrate_class="model_bounded_generation",
        MODEL_SPECS=MODEL_SPECS,
        trained_head_specs=data.get("trained_head_specs", []),
        model_invocation_counts=counts,
        call_ledger=[] if fixture else ledger.rows,
        private_transport_ledger=ledger.rows if fixture else [],
        run_date=date,
        duration_s=duration,
        random_seed=CONFIG["seed"],
        sample_size_budget=CONFIG,
        source_artifact_hashes=data["refs"],
        raw_shard_hashes=[
            reference(raw / "primitive_rows.json"),
            reference(raw / "input_data.json"),
        ],
        code_config_hashes={
            p: sha256_file(ROOT / p)
            for p in [
                *OWNED,
                TEST,
                prior.MODULE,
                "python/carnot/verify/durable_batch_8159.py",
                "python/carnot/inference/qwen_sufficiency_7920.py",
            ]
        },
        phase_spans=[dict(phase="startup", duration_s=work["startup_ns"] / 1e9)],
        acceptance_gates=dict(
            distinct_usable_sources=24,
            intended_sources=32,
            speed="10000 paired source log-ratio draws; lower95>1",
            nfr01="Cannot be inferred from component composition",
        ),
        inference_mode="private_fixture" if fixture else "live_gpu",
        comparison_scope="shared_acquisition_component_composition",
        fixture_mode=fixture,
        measured_reference_rows=work["references"],
        capture_manifest=data.get("slots", []),
        startup_cost=dict(duration_ns=work["startup_ns"], amortized_requests=32),
        model_load_cost=[r for r in ledger.rows if r["operation"] == "model_load"],
        warmup_cost=dict(
            calls=len(work["warmups"]),
            duration_ns=sum(r.get("acquisition_ns", 0) for r in work["warmups"]),
            input_tokens=sum(r.get("input_tokens", 0) for r in work["warmups"]),
            output_tokens=sum(r.get("output_tokens", 0) for r in work["warmups"]),
        ),
        runtime_freeze=data.get("runtime_freeze", {}),
        gpu_receipts={
            k: v for k, v in result.items() if "gpu" in k or k == "model_identity_receipt"
        },
        source_pair_rows=work["captures"],
        primitive_rows=reference(raw / "primitive_rows.json"),
        input_data=reference(raw / "input_data.json"),
        methodology="One generation per source; exact captured bytes and public features reused across four qualified durable branches under cold/warm/restart and all-at-once/4ms arrivals. Queue, commit/fsync, startup and eight warmups charged. One actual reference separately timed. All32 denominators retained; no replacements.",
        field_principles=dict(
            verdict_class="Only unfinished owned work is retryable partial.",
            composed_cost_rows="Shared acquisition is charged to each arm, never physically rerun.",
            acquisition_composition_ready_score="24 distinct usable sources and normal owned checks; no speed gate.",
            independent_generalization_score="Exposed development is not independent generalization.",
            model_invocation_counts="Only current execution; historical calls are not imported.",
            zero_arithmetic_ceiling="Remove arithmetic only; retain all acquisition and host costs.",
        ),
        **reduction,
    )
    value["acquisition_composition_ready_score"] = int(ready)
    value["reproducibility_checksum"] = checksum(value)
    return value


def independent_costs(value: Json, work: Json) -> bool:
    """Recalculate charged costs from clocks using separate arithmetic from the producer."""
    captured = {r["unit_id"]: r for r in work["captures"]}
    common = (work["startup_ns"] + sum(r.get("acquisition_ns", 0) for r in work["warmups"])) / 32
    totals: Json = {}
    groups = {(g["condition"], g["arrival"]): g for g in work["host_groups"]}
    for row in value["composed_cost_rows"]:
        if row["numerator"] is None:
            continue
        source = captured[row["unit_id"]]
        arms = groups[(row["condition"], row["arrival"])]["arms"]
        arm = next(a for a in arms if a["arm"] == row["arm"])
        requests = arm["requests"]
        request = next(r for r in requests if r["source_cluster_id"] == row["source_cluster_id"])
        outside = max(
            0,
            arm["duration_ns"]
            - (max(r["response_ns"] for r in requests) - min(r["enqueue_ns"] for r in requests)),
        ) / len(requests)
        actual = (
            source["ended_monotonic_ns"]
            - source["started_monotonic_ns"]
            + request["response_ns"]
            - request["enqueue_ns"]
            + outside
            + common
        )
        if not math.isclose(row["numerator"], actual, rel_tol=1e-12):
            return False
        totals[(row["condition"], row["arrival"], row["arm"], row["source_cluster_id"])] = actual
    total = sum(totals.values())
    arithmetic = sum(a["components"]["arithmetic_ns"] for g in groups.values() for a in g["arms"])
    if total:
        ceiling = value["zero_arithmetic_ceiling"]
        if not (
            math.isclose(ceiling["maximum_speedup"], total / (total - arithmetic), rel_tol=1e-12)
            and math.isclose(ceiling["arithmetic_fraction"], arithmetic / total, rel_tol=1e-12)
        ):
            return False
    sources = [r["source_cluster_id"] for r in work["captures"] if r["status"] == "completed"]
    for interval in value["paired_speed_intervals"]:
        c, a, arm = interval["condition"], interval["arrival"], interval["arm"]
        logratios = np.log(
            [totals[(c, a, host.ARMS[0], s)] / totals[(c, a, arm, s)] for s in sources]
        )
        indices = np.random.default_rng(CONFIG["seed"]).integers(
            0, len(sources), size=(10000, len(sources))
        )
        lower = float(np.exp(np.percentile(logratios[indices].sum(axis=1) / len(sources), 5)))
        if not (
            math.isclose(lower, interval["lower95"], rel_tol=1e-12)
            and math.isclose(float(np.exp(logratios.mean())), interval["estimate"], rel_tol=1e-12)
        ):
            return False
    return True


def replay(path: Path) -> bool:
    """Reopen primitive bytes and recompute costs and decisions without running a model."""
    progress("8160_replay_before", 0, 1)
    try:
        value = json.loads(path.read_text())
        if checksum(value) != value["reproducibility_checksum"]:
            return False
        for ref in [
            *value["source_artifact_hashes"],
            *value["raw_shard_hashes"],
            value["primitive_rows"],
            value["input_data"],
        ]:
            if sha256_file(Path(ref["path"])) != ref["sha256"]:
                return False
        for name, digest in value["code_config_hashes"].items():
            if sha256_file(ROOT / name) != digest:
                return False
        work = json.loads(Path(value["primitive_rows"]["path"]).read_text())
        data = json.loads(Path(value["input_data"]["path"]).read_text())
        selected = {s["unit_id"]: s for s in data["slots"]}
        for row in [*work["captures"], *work["warmups"]]:
            if "capture_path" in row:
                output = Path(row["capture_path"])
                if (
                    sha256_file(output) != row["capture_sha256"]
                    or output.read_text() != row["raw_response"]["choices"][0]["message"]["content"]
                ):
                    return False
            if row["status"] != "completed":
                continue
            slot = selected[row["unit_id"]]
            if (
                slot["source_cluster_id"] != row["source_cluster_id"]
                or slot["request"] != row["request"]
            ):
                return False
            parsed = prior.stream.risk.transport.parse_response(
                row["raw_response"], slot["visible_ids"]
            )
            body = json.loads(slot["request"]["messages"][1]["content"])
            lexical = prior.stream.lexical.extract(
                dict(
                    family_id=slot["family_id"],
                    source_bytes=body["complete_source"].encode().hex(),
                    answer_bytes=body["original_answer"].encode().hex(),
                )
            )
            p = min(1 - 1e-6, max(1e-6, parsed["probability"]))
            if row["values"] != [math.log(p / (1 - p)), *lexical["values"]]:
                return False
            if row["acquisition_ns"] != row["ended_monotonic_ns"] - row["started_monotonic_ns"]:
                return False
        for group in work["host_groups"]:
            for arm in group["arms"]:
                if sha256_file(Path(arm["store_path"])) != arm["store_sha256"]:
                    return False
                host.Store(Path(arm["store_path"]), arm["head_hash"])
                expected = host.old.score(
                    data["head"],
                    data["geometry"],
                    [r["values"] for r in arm["requests"]],
                    None,
                    "python_batch",
                )
                clocks = {(r["execution_start_ns"], r["execution_end_ns"]) for r in arm["requests"]}
                if (
                    sum(b - a for a, b in clocks) != arm["components"]["arithmetic_ns"]
                    or sum(arm["components"].values()) != arm["duration_ns"]
                ):
                    return False
                for request, probability in zip(arm["requests"], expected, strict=True):
                    source = next(
                        r
                        for r in work["captures"]
                        if r["source_cluster_id"] == request["source_cluster_id"]
                    )
                    if (
                        request["input_hash"] != canonical_hash(source)
                        or abs(request["probability"] - float(probability)) > 1e-10
                        or request["action"]
                        != host.old.engine.historical.radial.action(float(probability))
                        or request["latency_ns"] != request["response_ns"] - request["enqueue_ns"]
                        or request["commit_end_ns"] > request["response_ns"]
                    ):
                        return False
        reduction = reduce_rows(work)
        if not independent_costs(value, work):
            return False
        if any(
            value[k] != v
            for k, v in reduction.items()
            if k != "acquisition_composition_ready_score"
        ):
            return False
        ledger = Ledger(path.parent / "unused-replay-ledger.json")
        ledger.rows = (
            value["private_transport_ledger"] if value["fixture_mode"] else value["call_ledger"]
        )
        counts = ledger.counts()
        attempted = len(
            [
                r
                for r in work["captures"] + work["warmups"]
                if "raw_response" in r or r["status"] == "failed"
            ]
        )
        if (
            (dict(prior.ZERO_INVOCATION_COUNTS) if value["fixture_mode"] else counts)
            != value["model_invocation_counts"]
            or counts["generation_calls_attempted"] != attempted
            or attempted > 40
        ):
            return False
        if value["acquisition_composition_ready_score"] and (
            value["fixture_mode"]
            or not reduction["acquisition_composition_ready_score"]
            or counts["model_loads_completed"] != 1
            or not value["required_checks_passed"]
        ):
            return False
        progress("8160_replay_after", 1, 0)
        return True
    except (OSError, ValueError, KeyError, TypeError, StopIteration):
        return False
