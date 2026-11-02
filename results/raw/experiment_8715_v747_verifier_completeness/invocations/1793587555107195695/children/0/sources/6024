"""REQ-VERIFY-8264: reuse owned CUDA acquisition with bounded focal requests.

The server and GPU lease come from the qualified lifecycle. This adapter records
current traffic and costs without using evaluator labels or changing weights.
"""

from __future__ import annotations

from contextlib import nullcontext
import json
import os
from pathlib import Path
import time
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

from carnot import experiment_7969_v691_qwen_calibration_capture as legacy
from carnot.inference.qwen_sufficiency_7920 import bounded
from carnot.inference.sota_models import cached_current_model
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash, sha256_file
from carnot.verify import focal_capture_8263 as capture
from carnot.verify import protocol_conformance_8263 as qualified
from carnot.verify import evidence_view_canary_8264 as domain

Json = dict[str, Any]
TASK = "exp8264-evidence-view-canary"
MODEL_SPECS = [
    dict(
        hf_id="unsloth/Qwen3.8-27B-GGUF",
        quantization="Q4_K_M",
        model_path_sha256=qualified.TOKENIZER_PIN,
    )
]
gate, reference, progress = qualified.gate, qualified.reference, qualified.progress


def public_inputs(root: Path, work: Json) -> Json:
    """Import only public source bytes, role identities and cached sentence predictions."""
    qualified.authenticate(root, work)
    current = root / "results/experiment_8263_v714_protocol_conformance.json"
    gate(work, current, "exists", True, current.is_file())
    if not current.is_file():
        return {}
    value = json.loads(current.read_bytes())
    for key, expected in [
        ("view_kernel_ready_score", 1),
        ("required_checks_passed", True),
        ("flagged_adversarial", False),
    ]:
        gate(work, current, key, expected, value.get(key))
    from carnot.reporting.primary_publication import read_bound_sidecar

    terminal = Path(value["terminal_validation_sidecar_path"])
    report = json.loads(terminal.read_bytes())
    sidecar = Path(report["publication"]["sidecar_path"])
    gate(
        work,
        terminal,
        "publication.primary_sha256",
        sha256_file(current),
        report["publication"]["primary_sha256"],
    )
    gate(
        work,
        sidecar,
        "report.passed",
        True,
        read_bound_sidecar(current, sidecar)["report"]["passed"],
    )
    work["refs"].extend(
        dict(reference(p), fields_imported=fields)
        for p, fields in [
            (
                current,
                [
                    "view_kernel_ready_score",
                    "required_checks_passed",
                    "terminal_validation_sidecar_path",
                ],
            ),
            (terminal, ["publication"]),
            (sidecar, ["report.passed"]),
        ]
    )
    contract = root / "openspec/change-proposals/v714-evidence-execution-contract.json"
    authority = json.loads(
        (root / "results/experiment_8262_v714_coverage_custody.json").read_bytes()
    )
    gate(
        work,
        contract,
        "execution_contract_sha256",
        authority["execution_contract_sha256"],
        sha256_file(contract),
    )
    protocol = json.loads((root / qualified.PROTOCOL).read_bytes())
    result = {}
    for prefix in ["fit", "evaluation"]:
        primary = Path(protocol[prefix + "_primary"])
        value = json.loads(primary.read_bytes())
        ref = value["measurement_reference"]
        path = Path(ref["path"])
        gate(work, path, "measurement_reference.sha256", ref["sha256"], sha256_file(path))
        work["refs"].append(dict(reference(path), fields_imported=["slots.public_bytes_and_ids"]))
        slots = json.loads(path.read_bytes())["slots"]
        result[prefix] = dict(
            slots=[
                {
                    k: row[k]
                    for k in [
                        "unit_id",
                        "source_cluster_id",
                        "source_bytes",
                        "answer_bytes",
                        "role",
                    ]
                }
                for row in slots
            ],
            cache=[
                {k: row[k] for k in ["unit_id", "sentence_index", "p_unsupported", "relation"]}
                for row in value["sentence_rows"]
            ],
        )
    result["roles"] = protocol["role_manifest"]
    return result


def prepare(inputs: Json, count: Any) -> Json:
    """Freeze the twelve canary sources and all future branch counts before outputs."""
    fit_ids = {r["unit_id"] for r in inputs["roles"]["fit"]}
    full = inputs["fit"]
    selected = domain.plan_slots(
        [r for r in full["slots"] if r["unit_id"] in fit_ids], full["cache"], count
    )[:12]
    rosters: Json = {}
    for role in ["fit", "tune", "reserved"]:
        data = full if role != "reserved" else inputs["evaluation"]
        slots = (
            [r for r in data["slots"] if (r["unit_id"] in fit_ids) == (role == "fit")]
            if role != "reserved"
            else data["slots"]
        )
        plans = {p["unit_id"]: p for p in domain.plan_slots(slots, data["cache"], count)}
        rosters[role] = []
        for slot in sorted(slots, key=lambda r: r["source_cluster_id"]):
            plan = plans.get(slot["unit_id"], {}).get("plan", {})
            for name in domain.VIEWS:
                request = plan.get("views", {}).get(name, {}).get("request", {})
                rosters[role].append(
                    dict(
                        unit_id=slot["unit_id"],
                        source_cluster_id=slot["source_cluster_id"],
                        condition=name,
                        input_tokens=request.get("input_tokens"),
                        output_tokens=request.get("max_tokens"),
                        grammar_measured_tokens=request.get("output_bound", {}).get(
                            "maximum_measured_tokens"
                        ),
                        prompt=request.get("prompt"),
                        unavailable_reason=None
                        if request
                        else plan.get("reason", "missing_focal_cached_probability"),
                    )
                )
    return dict(plans=selected, calls=[], rosters=rosters, timings=[], spans={}, capture={})


def acquire(work: Json, raw: Path, private: Path) -> Json:
    """Bind the qualified journal to one actual server without shared request state."""
    evidence = work["evidence"]
    clocks = evidence.setdefault("service_clocks", [])
    frozen = domain.schedule(evidence["plans"])
    spec = cached_current_model() or {}
    model = Path(spec.get("model_path", "/missing"))
    gate(
        work,
        model,
        "model_path_matches_embedded_tokenizer",
        str(qualified.TOKENIZER_PATH),
        str(model),
    )
    gate(work, model, "CARNOT_FORCE_LIVE", "1", os.environ.get("CARNOT_FORCE_LIVE"))
    binary = Path.home() / ".cache/llama.cpp-master/build/bin/llama-server"
    gate(work, binary, "owned_server_available", True, binary.is_file())
    if not all(c["passed"] for c in work["checks"]):
        return {}
    runtime_class = legacy.QwenRuntime

    class Recorded(runtime_class):  # type: ignore[misc,valid-type]
        def load(self) -> Json:
            """Record cold load boundaries and authenticate the served embedded template."""
            before = time.monotonic_ns()
            result: Json = super().load()
            evidence["spans"]["load"] = (time.monotonic_ns() - before) / 1e9
            clocks.append(
                dict(
                    phase="cold_load",
                    started_monotonic_ns=before,
                    ended_monotonic_ns=time.monotonic_ns(),
                )
            )
            if canonical_hash(result["props"]["chat_template"]) != canonical_hash(
                work["tokenizer"]["metadata"]["tokenizer.chat_template"]
            ):
                raise ValueError("embedded_chat_template_drift")
            return result

        def close(self) -> Json:
            """Measure shutdown around owner-scoped process cleanup."""
            before = time.monotonic_ns()
            result: Json = super().close()
            evidence["spans"]["shutdown"] = (time.monotonic_ns() - before) / 1e9
            clocks.append(
                dict(
                    phase="shutdown",
                    started_monotonic_ns=before,
                    ended_monotonic_ns=time.monotonic_ns(),
                )
            )
            return result

    def collect(
        slots: list[Json], runtime: Any, path: Path, identity: Json, **kwargs: Any
    ) -> list[Json]:
        before = time.monotonic_ns()
        for entries in [evidence["rosters"].values()]:
            for entries_for_role in entries:
                for entry in entries_for_role:
                    if entry["prompt"] is None:
                        continue
                    rendered = runtime.worker.post_json(
                        "/apply-template",
                        dict(
                            messages=[dict(role="user", content=entry.pop("prompt"))],
                            add_generation_prompt=True,
                            chat_template_kwargs=dict(enable_thinking=False),
                        ),
                        10,
                    )["prompt"]
                    entry["input_tokens"] = runtime.count(rendered)
        for slot in slots:
            request = slot["view"]["request"]
            request["rendered_prompt"] = runtime.worker.post_json(
                "/apply-template",
                dict(
                    messages=[dict(role="user", content=request["prompt"])],
                    add_generation_prompt=True,
                    chat_template_kwargs=dict(enable_thinking=False),
                ),
                10,
            )["prompt"]
            request["rendered_input_tokens"] = runtime.count(request["rendered_prompt"])
            if request["rendered_input_tokens"] + 64 > 8192:
                raise ValueError("rendered_context_budget")
        frozen[:] = domain.schedule(evidence["plans"])
        evidence["spans"]["setup"] = (time.monotonic_ns() - before) / 1e9
        clocks.append(
            dict(
                phase="render_and_token_count",
                started_monotonic_ns=before,
                ended_monotonic_ns=time.monotonic_ns(),
            )
        )

        def dispatch(wire: str) -> str:
            import threading
            from carnot.reporting.v709_execution import child

            value = json.loads(wire)
            request = value["request"]
            index = len(evidence["timings"])
            payload = dict(
                prompt=request["rendered_prompt"],
                grammar=request["grammar"],
                n_predict=64,
                temperature=0,
                seed=7138250,
                cache_prompt=False,
                stream=False,
                id_slot=0,
                request_id=value["resume_key"],
            )
            serialization = time.monotonic_ns()
            canonical_hash(payload)
            evidence["spans"]["serialization"] = max(
                evidence["spans"].get("serialization", 0),
                (time.monotonic_ns() - serialization) / 1e9,
            )
            clocks.append(
                dict(
                    phase="serialization",
                    started_monotonic_ns=serialization,
                    ended_monotonic_ns=time.monotonic_ns(),
                )
            )
            stop = threading.Event()
            samples: list[Json] = []

            def monitor() -> None:
                while not stop.wait(0.1):
                    receipt = child(
                        f"inflight_{index}_{len(samples)}",
                        [
                            "nvidia-smi",
                            "--query-gpu=index,uuid,memory.used,utilization.gpu",
                            "--format=csv,noheader,nounits",
                        ],
                        raw / "telemetry",
                        deadline=10,
                        heartbeat=5,
                    )
                    samples.append(receipt)

            thread = threading.Thread(target=monitor, daemon=True)
            thread.start()
            started = time.monotonic_ns()
            response = None
            try:
                response = bounded(
                    lambda: runtime.worker.post_json("/completion", payload, 120), 120
                )
                atomic_json(
                    raw / "responses" / f"{index:03}.json", dict(payload=payload, response=response)
                )
            finally:
                ended = time.monotonic_ns()
                stop.set()
                thread.join(timeout=12)
                receipt = dict(
                    request_id=value["resume_key"],
                    payload=payload,
                    response=response,
                    started_monotonic_ns=started,
                    ended_monotonic_ns=ended,
                    telemetry=samples,
                    response_bytes=len(json.dumps(response).encode()) if response else 0,
                )
                evidence["timings"].append(
                    dict(
                        (response or {}).get("timings", {}),
                        response_wall_seconds=(ended - started) / 1e9,
                    )
                )
                runtime.receipts.append(receipt)
                atomic_json(raw / "responses" / f"{index:03}.receipt.json", receipt)
            return json.dumps(
                dict(resume_key=value["resume_key"], role=value["role"], text=response["content"])
            )

        captured = capture.capture(frozen, path / "capture.jsonl", nullcontext(dispatch))
        evidence["capture"] = captured
        evidence["calls"] = domain.decorate(frozen, captured["rows"])
        return list(evidence["calls"])

    with (
        patch.object(legacy, "TASK", TASK),
        patch.object(legacy, "QwenRuntime", Recorded),
        patch.object(legacy, "capture", SimpleNamespace(freeze=lambda _: frozen, capture=collect)),
        patch.object(legacy, "load_public", lambda _: {}),
    ):
        result: Json = legacy.live_capture(
            dict(
                public_role_manifests={},
                capture_identity={},
                protocol=dict(
                    gguf_sha256=qualified.TOKENIZER_PIN, model_revision=model.parent.name
                ),
            ),
            raw,
            private,
        )
    for check in result["checks"]:
        work["checks"].append(
            dict(check, artifact_field=check["field"], upstream=check["upstream_id"])
        )
    return result
