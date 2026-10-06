"""REQ-REPORT-8181: test bounded syntax before spending on larger capture.

Original source bytes and every failed response stay available for independent
reduction. Valid addresses and relations are predictions, never semantic gold.
"""

from __future__ import annotations

from copy import deepcopy
import json
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

from carnot.reporting import methods_stream_execution_8111 as execution
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash, sha256_file
from carnot.reporting.primary_publication import read_bound_sidecar
from carnot.verify import fit_evidence_capture_8153 as qualified
from carnot.verify import sentence_transport_8179 as transport
from carnot.verify import sentence_transport_methods_8179 as methods

Json = dict[str, Any]
ROOT = methods.ROOT
NAME = "experiment_8181_v707_sentence_transport_canary"
TASK = "exp8181-sentence-transport-canary"
MODULE = "python/carnot/verify/sentence_transport_canary_8181.py"
CLI = f"scripts/experiments/{NAME}.py"
TEST = "tests/python/test_sentence_transport_canary_8181.py"
OWNED = [MODULE, CLI]
RUN_DATE = "20261006"
MODEL_SPECS = ["unsloth/Qwen3.8-27B-GGUF"]
UPSTREAM = "results/experiment_8179_v707_sentence_transport_methods.json"
PIN = "sha256:9e64df8a53c2542a7af5c1ed3e1c139bfd463b6023933efb8d40996a73f65ea1"
CONFIG = dict(
    seed=70781,
    maximum_calls=96,
    output_tokens=256,
    input_tokens=6000,
    latest_launch_s=3000,
    closure_s=4800,
    checkpoint_sources=8,
    duration_floor_s=10,
)
reference = methods.reference


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Flush real counters so bounded child waits remain visible to operators."""
    print(f"[exp8181] phase={phase} completed={completed} pending={pending}", flush=True)


def gate(
    plan: Json, path: Path, check: str, expected: Any, observed: Any, *, op: str = "=="
) -> None:
    """Retain exact operands while hashing a frozen large model only once."""
    cache = plan.setdefault("path_hashes", {})
    key = str(path.absolute())
    if key not in cache:
        cache[key] = sha256_file(path) if path.is_file() else None
    plan["checks"].append(
        dict(
            check=check,
            upstream=UPSTREAM,
            path=key,
            hash=cache[key],
            artifact_field=check,
            op=op,
            expected=expected,
            observed=observed,
            passed=expected == observed if op == "==" else observed <= expected,
        )
    )


def select(rows: list[Json], canary: Json) -> list[Json]:
    """Use only predeclared public IDs; labels cannot alter arm order."""
    by_id = {r["unit_id"]: r for r in rows if r["role"] == "fit"}
    if (
        any(i not in by_id for i in canary["source_ids"])
        or [r["unit_id"] for r in canary["arm_order"]] != canary["source_ids"]
    ):
        raise ValueError("frozen_canary_ids")
    return [dict(deepcopy(by_id[r["unit_id"]]), arms=r["arms"]) for r in canary["arm_order"]]


def capture(
    slots: list[Json], runtime: Any, raw: Path, identity: Json, *, started: float | None = None
) -> list[Json]:
    """Seal bounded attempts without retrying truncation or selecting on labels."""
    began = time.monotonic() if started is None else started
    rows: list[Json] = []
    total = sum(len(s["requests"]) * len(s["arms"]) for s in slots)
    for number, slot in enumerate(slots):
        for arm in slot["arms"]:
            for request in slot["requests"]:
                row = dict(
                    unit_id=slot["unit_id"],
                    source_cluster_id=slot["source_cluster_id"],
                    arm=arm,
                    request=request,
                    identity=identity,
                    status="censored",
                    exclusion_reason="launch_deadline",
                    response=None,
                    started_monotonic_ns=time.monotonic_ns(),
                    rendered_prompt=None,
                )
                rows.append(row)
                path = raw / f"call-{len(rows):03d}.json"
                atomic_json(path, row)
                if time.monotonic() - began < CONFIG["latest_launch_s"]:
                    messages = [dict(role="user", content=request["payload"])]
                    payload = dict(
                        model=MODEL_SPECS[0],
                        messages=messages,
                        temperature=0,
                        seed=CONFIG["seed"],
                        max_tokens=256,
                        stream=False,
                        chat_template_kwargs=dict(enable_thinking=False),
                    )
                    if arm == "grammar":
                        payload["grammar"] = request["grammar"]
                    progress("before_generation", len(rows) - 1, total - len(rows) + 1)
                    try:
                        rendered = runtime.worker.post_json(
                            "/apply-template",
                            dict(
                                messages=messages,
                                add_generation_prompt=True,
                                chat_template_kwargs=dict(enable_thinking=False),
                            ),
                            10,
                        )["prompt"]
                        count = len(
                            runtime.worker.post_json(
                                "/tokenize", dict(content=rendered, add_special=True), 10
                            )["tokens"]
                        )
                        row.update(
                            rendered_prompt=rendered,
                            rendered_input_tokens=count,
                            payload=payload,
                            cache_key=canonical_hash(
                                dict(
                                    payload=payload,
                                    identity=identity,
                                    source_bytes=slot["source_bytes"],
                                    answer_bytes=slot["answer_bytes"],
                                )
                            ),
                        )
                        if count > 6000:
                            raise ValueError("rendered_input_token_limit")
                        row.update(
                            status="completed",
                            exclusion_reason=None,
                            response=runtime.generate(payload),
                        )
                    except (OSError, RuntimeError, ValueError, TimeoutError, KeyError) as error:
                        row.update(status="failed", exclusion_reason=str(error))
                    progress("after_generation", len(rows), total - len(rows))
                row["ended_monotonic_ns"] = time.monotonic_ns()
                atomic_json(path, row)
        if (number + 1) % 8 == 0:
            atomic_json(raw / f"checkpoint-{number + 1:02d}.json", dict(rows=rows))
            progress("source_checkpoint", number + 1, len(slots) - number - 1)
    return rows


def reduce(slots: list[Json], calls: list[Json]) -> Json:
    """Reparse every response and count paired original source clusters once."""
    rows, schemas, budgets = [], [], []
    for slot in slots:
        for arm in slot["arms"]:
            group = [c for c in calls if c["unit_id"] == slot["unit_id"] and c["arm"] == arm]
            accepted, loss, truncations = [], 0, 0
            for call in group:
                response = call.get("response") or {}
                choices = response.get("choices") or []
                choice = choices[0] if choices else {}
                finish = choice.get("finish_reason")
                parsed = transport.parse(
                    choice.get("message", {}).get("content") or "",
                    call["request"]["sentence_indices"],
                    len(slot["source_segments"]),
                )
                ok = (
                    call["status"] == "completed"
                    and finish == "stop"
                    and parsed["status"] == "completed"
                )
                loss += int(not ok)
                truncations += int(finish == "length")
                accepted.extend(parsed["rows"] if ok else [])
                schemas.append(
                    dict(
                        unit_id=slot["unit_id"],
                        arm=arm,
                        accepted=ok,
                        parsed_records=parsed["rows"],
                        finish_reason=finish,
                        human_target=None,
                        semantic_gold=False,
                    )
                )
                budgets.append(
                    dict(
                        unit_id=slot["unit_id"],
                        arm=arm,
                        **response.get("usage", {}),
                        finish_reason=finish,
                        rendered_input_tokens=call.get("rendered_input_tokens"),
                    )
                )
            complete = len(group) == len(slot["requests"]) and bool(group) and loss == 0
            rows.append(
                dict(
                    unit_id=slot["unit_id"],
                    source_cluster_id=slot["source_cluster_id"],
                    arm=arm,
                    condition="original",
                    metric="complete_source_transport",
                    numerator=int(complete),
                    denominator=1,
                    status="completed"
                    if complete
                    else "censored"
                    if group and all(c["status"] == "censored" for c in group)
                    else "failed",
                    exclusion_reason=None if complete else "incomplete_source_transport",
                    structural_loss=loss,
                    truncations=truncations,
                    predictions=accepted,
                    duration_s=sum(
                        (c["ended_monotonic_ns"] - c["started_monotonic_ns"]) / 1e9 for c in group
                    ),
                )
            )
    grammar = [r for r in rows if r["arm"] == "grammar"]
    effects = []
    for slot in slots:
        pair = {r["arm"]: r for r in rows if r["unit_id"] == slot["unit_id"]}
        valid = all(pair[a]["numerator"] for a in ["grammar", "unconstrained"])
        left, right = pair["grammar"]["predictions"], pair["unconstrained"]["predictions"]
        effects.append(
            dict(
                unit_id=slot["unit_id"],
                complete_pair=valid,
                relation_changes=sum(a["relation"] != b["relation"] for a, b in zip(left, right))
                if valid
                else None,
                probability_absolute_difference=sum(
                    abs(a["p_unsupported"] - b["p_unsupported"]) for a, b in zip(left, right)
                )
                / len(left)
                if valid
                else None,
                semantic_gold=False,
            )
        )
    return dict(
        rows=rows,
        response_schema_rows=schemas,
        token_budget_rows=budgets,
        per_source_results=rows,
        format_effect_rows=effects,
        grammar_complete_sources=sum(r["numerator"] for r in grammar),
        unconstrained_complete_sources=sum(
            r["numerator"] for r in rows if r["arm"] == "unconstrained"
        ),
        structural_loss_delta=sum(
            r["structural_loss"] * (1 if r["arm"] == "grammar" else -1) for r in rows
        ),
        intended_count=24,
        eligible_count=len(slots),
        independent_count=len({s["source_cluster_id"] for s in slots}),
        completed_count=sum(r["numerator"] for r in grammar),
        excluded_count=24 - len(slots),
        censored_count=sum(r["status"] == "censored" for r in grammar),
        failed_count=sum(r["status"] == "failed" for r in grammar),
    )


def inputs(root: Path, raw: Path) -> Json:
    """Authenticate immutable public primitives before touching model weights."""
    plan: Json = dict(checks=[], slots=[], refs=[], upstream=[], identity={}, grammar_receipt={})
    path = root / UPSTREAM
    observed = sha256_file(path) if path.is_file() else None
    gate(plan, path, "upstream_sha256", PIN, observed)
    if observed != PIN:
        return plan
    try:
        value = json.loads(path.read_text())
        for field, expected in [
            ("sentence_protocol_ready_score", 1),
            ("required_checks_passed", True),
            ("flagged_adversarial", False),
        ]:
            gate(plan, path, field, expected, value.get(field))
        terminal = json.loads(Path(value["terminal_validation_sidecar_path"]).read_text())
        gate(
            plan,
            path,
            "upstream_terminal_passed",
            True,
            read_bound_sidecar(path, Path(terminal["publication"]["sidecar_path"]))["report"][
                "passed"
            ],
        )
        protocol = root / methods.PROTOCOL
        gate(
            plan,
            protocol,
            "protocol_sha256",
            methods.PROTOCOL_PIN,
            sha256_file(protocol) if protocol.is_file() else None,
        )
        config = json.loads(protocol.read_text())
        refs = [
            reference(path),
            reference(protocol),
            value["measurement_reference"],
            *value["raw_shard_hashes"],
            value["source_manifest"]["fit"],
        ]
        for ref in refs:
            original = Path(ref["path"])
            actual = sha256_file(original) if original.is_file() else None
            gate(plan, original, "primitive_sha256", ref["sha256"], actual)
            if actual != ref["sha256"]:
                return plan
            target = raw / "inputs" / (actual[7:] + "-" + original.name)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(original.read_bytes())
            plan["refs"].append(reference(target))
        primitive = json.loads(Path(value["raw_shard_hashes"][0]["path"]).read_text())
        plan.update(
            slots=select(primitive["rows"], config["canary"]),
            config=config,
            tokenizer_receipt=value["tokenizer_receipt"],
            upstream=[
                dict(
                    experiment_id=8179,
                    sha256=observed,
                    fields_imported=["canary_plan", "raw_shard_hashes", "tokenizer_receipt"],
                )
            ],
        )
    except (OSError, ValueError, KeyError, TypeError) as error:
        gate(plan, path, "upstream_structure", "authenticated public primitives", str(error))
    return plan


def preflight(plan: Json, raw: Path) -> None:
    """Compile finite grammars on CPU before leasing CUDA; never substitute a model."""
    from llama_cpp import Llama, llama_cpp as binding

    spec = qualified.legacy.cached_current_model() or {}
    model = Path(spec.get("model_path", "/missing"))
    gate(
        plan,
        model,
        "cache_revision",
        Path(plan["tokenizer_receipt"]["path"]).parent.name,
        model.parent.name,
    )
    gate(plan, model, "model_id", MODEL_SPECS[0], spec.get("hf_id"))
    progress("before_model_hash")
    digest = sha256_file(model) if model.is_file() else None
    progress("after_model_hash")
    gate(plan, model, "tokenizer_gguf_sha256", plan["tokenizer_receipt"]["sha256"], digest)
    gate(plan, ROOT, "CARNOT_FORCE_LIVE", "1", os.environ.get("CARNOT_FORCE_LIVE"))
    if not all(c["passed"] for c in plan["checks"]):
        return
    progress("before_CPU_vocabulary_load")
    vocab = Llama(model_path=str(model), vocab_only=True, n_gpu_layers=0, verbose=False)
    progress("after_CPU_vocabulary_load")
    count = lambda text: len(vocab.tokenize(text.encode(), add_bos=False, special=False))
    compiled = 0
    try:
        for slot in plan["slots"]:
            request = transport.requests(slot, count)
            gate(
                plan,
                model,
                "frozen_request_identity",
                canonical_hash(slot["requests"]),
                canonical_hash(request["requests"]),
            )
            for call in slot["requests"]:
                sampler = binding.llama_sampler_init_grammar(
                    vocab._model.vocab, call["grammar"].encode(), b"root"
                )
                if not sampler:
                    raise ValueError("CPU_grammar_compile_failed")
                binding.llama_sampler_free(sampler)
                compiled += 1
                bound = transport.output_budget(
                    call["sentence_indices"], len(slot["source_segments"]), count
                )
                gate(
                    plan,
                    model,
                    "serialized_output_reachable",
                    256,
                    bound["maximum_encoded_output_tokens"],
                    op="<=",
                )
        template = vocab.metadata.get("tokenizer.chat_template", "")
        gate(plan, model, "supported_reasoning_interface", True, "enable_thinking" in template)
        binary = Path.home() / ".cache/llama.cpp-master/build/bin/llama-server"
        plan["protocol"] = dict(gguf_sha256=digest, model_revision=model.parent.name)
        plan["identity"] = dict(
            model_sha256=digest,
            tokenizer_sha256=digest,
            revision=model.parent.name,
            runtime_sha256=sha256_file(binary),
            chat_template_sha256=canonical_hash(template),
            reasoning_interface=dict(enable_thinking=False),
            fixture=False,
        )
        plan["refs"].append(reference(binary))
        plan["grammar_receipt"] = dict(
            status="compiled_CPU",
            compiled_requests=compiled,
            compiler=reference(Path(binding._lib._name)),
            maximum_output_tokens=256,
            maximum_calls=96,
            neural_weights_loaded=False,
        )
    except (OSError, ValueError, RuntimeError) as error:
        gate(plan, model, "runtime_grammar_support", True, str(error))
    finally:
        vocab.close()
        progress("after_CPU_grammar_preflight", compiled, 36 - compiled)


def live(plan: Json, raw: Path) -> Json:
    """Reuse the qualified CUDA lease, measured offload and owned child cleanup."""
    from tempfile import TemporaryDirectory

    runtime_class = qualified.legacy.QwenRuntime

    class Recorded(runtime_class):  # type: ignore[misc,valid-type]
        def load(self) -> Json:
            """Bind the actual served template and load timestamps before calls."""
            began = time.monotonic_ns()
            receipt: Json = super().load()
            receipt.update(started_monotonic_ns=began, ended_monotonic_ns=time.monotonic_ns())
            if (
                canonical_hash(receipt["props"]["chat_template"])
                != plan["identity"]["chat_template_sha256"]
            ):
                raise ValueError("served_chat_template_drift")
            return receipt

    def pulse(phase: str, started: float, units: int = 0) -> None:
        finished = sum(
            "ended_monotonic_ns" in json.loads(p.read_text())
            for p in (raw / "slots").glob("call-*.json")
        )
        total = sum(len(s["requests"]) * len(s["arms"]) for s in plan["slots"])
        progress(phase, finished, total - finished)

    adapter = SimpleNamespace(
        freeze=lambda _: plan["slots"],
        capture=lambda slots, runtime, path, identity, started: capture(
            slots, runtime, path, plan["identity"], started=plan["started"]
        ),
    )
    with TemporaryDirectory(prefix="carnot8181-model-") as directory:
        with (
            patch.object(qualified.legacy, "TASK", TASK),
            patch.object(qualified.legacy, "QwenRuntime", Recorded),
            patch.object(qualified.legacy, "progress", pulse),
            patch.object(qualified.legacy, "capture", adapter),
            patch.object(qualified.legacy, "load_public", lambda _: {}),
        ):
            result: Json = qualified.legacy.live_capture(
                dict(plan, public_role_manifests={}, capture_identity=plan["identity"]),
                raw,
                Path(directory),
            )
    for check in result["checks"]:
        check.update(
            check=check["upstream_id"] + "_" + check["field"],
            upstream=check["upstream_id"],
            artifact_field=check["field"],
        )
    return result


def fixture_work(slots: list[Json], calls: list[Json], raw: Path) -> Json:
    """Save private scripted evidence without attributing it to model execution."""
    raw.mkdir(parents=True, exist_ok=True)
    atomic_json(raw / "primitive_calls.json", dict(rows=calls))
    atomic_json(raw / "source_plan.json", dict(rows=slots))
    work = dict(
        slots=slots,
        calls=calls,
        checks=[],
        refs=[],
        upstream=[],
        identity=dict(fixture=True),
        grammar_receipt=dict(status="private_fixture"),
        live_result={},
        fixture=True,
        duration_s=0.01,
        phase_spans=[],
        code_config_hashes=[
            reference(ROOT / p)
            for p in [
                *OWNED,
                TEST,
                methods.NUMERIC,
                methods.MODULE,
                qualified.MODULE,
                "python/carnot/inference/qwen_sufficiency_7920.py",
            ]
        ],
        raw_shard_hashes=[reference(raw / p) for p in ["primitive_calls.json", "source_plan.json"]],
    )
    atomic_json(raw / "measurement.json", work)
    return work


def measure(
    root: Path,
    raw: Path,
    *,
    fixture: bool = False,
    stream_path: Path | None = None,
    mutation: str = "",
) -> Json:
    """Close blocked inputs terminally; only qualified inputs can dispatch calls."""
    began = time.monotonic()
    progress("before_input_authentication")
    plan = inputs(root, raw)
    result: Json = dict(rows=[], checks=[])
    if fixture and (root / "canary-fixture.json").is_file():
        sources = json.loads((root / "canary-fixture.json").read_text())["rows"]
        plan.update(slots=sources, checks=[], refs=[reference(root / "canary-fixture.json")])
        # A private runtime supplies responses only in explicitly requested tests.
        result["rows"] = json.loads((root / "canary-fixture.json").read_text())["calls"]
    if not fixture and all(c["passed"] for c in plan["checks"]):
        preflight(plan, raw)
        plan["started"] = began
        if all(c["passed"] for c in plan["checks"]):
            progress("before_live_capture")
            result = live(plan, raw)
            progress("after_live_capture", len(result["rows"]), 72 - len(result["rows"]))
    work = fixture_work(plan["slots"], result["rows"], raw)
    work.update(
        checks=plan["checks"] + result["checks"],
        refs=plan["refs"],
        upstream=plan["upstream"],
        identity=plan["identity"],
        grammar_receipt=plan["grammar_receipt"],
        fixture=fixture,
        live_result=result,
        duration_s=time.monotonic() - began,
        phase_spans=[
            dict(
                phase="authenticate_compile_capture", start_s=0, duration_s=time.monotonic() - began
            )
        ],
    )
    if mutation:
        gate(work, root, "private_response_tamper", "unchanged", mutation)
    atomic_json(raw / "measurement.json", work)
    progress("measurement_complete", len(result["rows"]), 0)
    return work


def build(work: Json, raw: Path, receipts: list[Json], *, fixture: bool = False) -> Json:
    """Readiness proves bounded transport only and requires normal owned exits."""
    reduced = reduce(work["slots"], work["calls"])
    failures = [c["check"] for c in work["checks"] if not c["passed"]]
    valid = bool(receipts) and all(r["passed"] for r in receipts)
    live_result = work["live_result"]
    ready = int(
        valid
        and not failures
        and not work["fixture"]
        and reduced["grammar_complete_sources"] >= 22
        and reduced["structural_loss_delta"] <= 0
        and live_result.get("model_loads_completed") == 1
        and work["duration_s"] >= CONFIG["duration_floor_s"]
    )
    verdict = (
        "disqualified" if not valid else "blocked" if failures else "positive" if ready else "null"
    )
    honest = (
        "complete_"
        + verdict
        + "_"
        + (failures[0] if verdict == "blocked" else "sentence_transport_canary")
    )
    qualified_config = dict(
        identity=work["identity"],
        config=CONFIG,
        grammar_receipt=work["grammar_receipt"],
        original_grammar_cache_keys=[
            c["cache_key"] for c in work["calls"] if c["arm"] == "grammar" and c.get("cache_key")
        ],
        provenance_task=TASK,
        acquisition_cost_charged_once=True,
    )
    value = dict(
        experiment_id=8181,
        task_id=TASK,
        run_date=RUN_DATE,
        honest_verdict=honest,
        verdict_class=verdict,
        verifier_is_oracle=work["fixture"],
        claim_scope="bounded evidence syntax and source addresses only; semantic predictions descriptive",
        exposure_scope="exposed_development_within_run_disjoint",
        independent_generalization_score=0,
        generalized_learning_benefit_score=0,
        required_checks_passed=valid,
        flagged_adversarial=False,
        validation_receipts=receipts,
        terminal_validation_sidecar_path=str(raw / "terminal_validation.json"),
        gate_check_summary=work["checks"],
        preconditions_checked=work["checks"],
        inference_substrate="aggregation_from_upstream_artifacts"
        if work["fixture"] or not live_result.get("model_loads_attempted")
        else "live_llm_inference",
        inference_substrate_class="no_model_load"
        if work["fixture"]
        else "model_bounded_generation",
        MODEL_SPECS=MODEL_SPECS,
        trained_head_specs=[],
        model_invocation_counts=dict(
            model_loads_attempted=live_result.get("model_loads_attempted", 0),
            model_loads_completed=live_result.get("model_loads_completed", 0),
            generate=0 if work["fixture"] else len(live_result.get("runtime_receipts", [])),
        ),
        call_ledger=work["calls"],
        cited_upstream_artifacts=work["upstream"],
        sample_size_budget=dict(sources=24, arms=2, calls=96, input_tokens=6000, output_tokens=256),
        duration_s=work["duration_s"],
        random_seed=CONFIG["seed"],
        source_artifact_hashes=work["refs"],
        raw_shard_hashes=work["raw_shard_hashes"],
        code_config_hashes=work["code_config_hashes"],
        phase_spans=work["phase_spans"],
        acceptance_gates=dict(
            grammar_sources=22,
            denominator=24,
            valid_indices=True,
            structural_loss_increase_maximum=0,
            budgets=CONFIG,
        ),
        transport_canary_ready_score=ready,
        qualified_transport_sha256=canonical_hash(qualified_config) if ready else None,
        qualified_transport_configuration=qualified_config if ready else None,
        grammar_receipt=work["grammar_receipt"],
        fixture_protocol_only=work["fixture"],
        model_receipt=live_result,
        measurement_reference=reference(raw / "measurement.json"),
        methodology_note="Frozen paired fit sources and unchanged input bytes, finite grammar, bounded Qwen calls and independent transcript parsing. No labels used to select syntax; complete transport does not certify a relation.",
        repository_health=work.get("global_health", {}),
        **reduced,
    )
    value["field_principles"] = {
        k: "Bind original source custody and measured transport; syntax cannot certify semantics or independent benefit."
        for k in value
    }
    value["reproducibility_checksum"] = canonical_hash(value)
    return value


def replay(path: Path) -> bool:
    """Reject response, source, code, log or headline drift without model calls."""
    try:
        value = json.loads(path.read_text())
        for ref in [
            value["measurement_reference"],
            *value["source_artifact_hashes"],
            *value["raw_shard_hashes"],
            *value["code_config_hashes"],
        ]:
            if sha256_file(Path(ref["path"])) != ref["sha256"]:
                return False
        for receipt in value["validation_receipts"]:
            if (
                receipt.get("log_path")
                and sha256_file(Path(receipt["log_path"])) != receipt["log_sha256"]
            ):
                return False
        work = json.loads(Path(value["measurement_reference"]["path"]).read_text())
        if (
            json.loads(Path(work["raw_shard_hashes"][0]["path"]).read_text())["rows"]
            != work["calls"]
        ):
            return False
        if (
            json.loads(Path(work["raw_shard_hashes"][1]["path"]).read_text())["rows"]
            != work["slots"]
        ):
            return False
        for slot in work["slots"]:
            counts = {r["payload"]: r["input_tokens"] for r in slot["requests"]}
            if transport.requests(slot, lambda text: counts[text])["requests"] != slot["requests"]:
                return False
        return bool(
            build(
                work,
                Path(value["terminal_validation_sidecar_path"]).parent,
                value["validation_receipts"],
            )
            == value
        )
    except (OSError, ValueError, KeyError, TypeError):
        return False


BASE_MANIFEST = execution.manifest
BASE_RUN_CHECK = execution.run_check


def supervise(
    root: Path, spec: Json, private: Path, durable: Path, *, heartbeat_s: float = 20
) -> Json:
    """Freeze the bounded live worker deadline before its first model call."""
    if spec["name"] == "measurement":
        spec["deadline_s"] = 3300
        path = durable.parent / "validation_commands.json"
        frozen = json.loads(path.read_text())
        frozen["measurement"] = spec
        atomic_json(path, frozen)
    return BASE_RUN_CHECK(root, spec, private, durable, heartbeat_s=heartbeat_s)


def manifest(private: Path, candidate: Path) -> Json:
    """Reuse frozen argv and immutable logs; static tools receive file paths."""
    with (
        patch.object(execution, "e", sys.modules[__name__]),
        patch.object(execution, "OWNED", OWNED),
    ):
        specs = BASE_MANIFEST(private, candidate)
    specs["commands"][0]["argv"].remove("-s")
    specs["commands"][1]["argv"] = [
        str(ROOT / ".venv/bin/pytest"),
        "-n0",
        "-o",
        "addopts=",
        "--no-cov",
        "-q",
        TEST,
        "tests/python/test_arc_tool_grammar_transport.py",
        "tests/python/test_source_boundary_7852.py",
        "tests/python/test_experiment_7942_v689_sentence_labels.py",
    ]
    specs["repository_health"]["deadline_s"] = 300
    specs["commands"][4]["argv"][-1] = str(candidate.parent / "coverage.json")
    return specs


def main(argv: list[str] | None = None) -> int:
    """Keep the CLI thin while the qualified runner validates publication."""
    os.environ["PYTHONUNBUFFERED"] = "1"
    with (
        patch.object(execution, "e", sys.modules[__name__]),
        patch.object(execution, "manifest", manifest),
        patch.object(execution, "run_check", supervise),
    ):
        return execution.main(argv)
