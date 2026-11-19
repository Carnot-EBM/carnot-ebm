"""REQ-REPORT-9098-V776: own frozen GGUF forwards on GPU0 with no sampling API."""

from __future__ import annotations

import os
from pathlib import Path
import re
import time
from typing import Any

import numpy as np

from carnot.gpu_lease_phase_journal import GpuLease, LeaseError
from carnot.inference.bfs_pilot_runtime_8918 import gpu_state, preflight
from carnot.inference.gguf_metadata import build_gguf_admission_record
from carnot.reporting import embedding_runtime_9098 as d
from carnot.reporting.current_work_receipt import atomic_json, build_current_work_receipt

Json = dict[str, Any]


class FrozenRuntime:
    """Read native token states and raw logits; model weights never enter a trainer."""

    def __init__(self, model: Any, raw: Path | None = None) -> None:
        self.model, self.raw, self.context = model, raw, None
        self.width = model.n_embd()
        self.pad_id = int(model.metadata["tokenizer.ggml.padding_token_id"])
        self.embedding_calls = 0
        self.last_hidden_references: list[Json] = []

    def tokenize(self, text: str) -> list[int]:
        """No special token or foreign tokenizer may alter the fixed text boundary."""
        return list(self.model.tokenize(text.encode(), add_bos=False, special=False))

    def hidden(self, sequences: list[list[int]]) -> list[Any]:
        """Use a two-sequence context so recurrent state cannot leak between batch items."""
        from llama_cpp import _internals as internals
        from llama_cpp import llama_cpp as cpp

        if cpp.llama_model_n_embd_out(self.model._model.model) != self.width:
            raise ValueError("classifier_scores_are_not_hidden_states")
        if self.context is None:
            params = type(self.model.context_params).from_buffer_copy(self.model.context_params)
            params.embeddings, params.n_seq_max, params.kv_unified = True, 2, True
            params.pooling_type = cpp.LLAMA_POOLING_TYPE_NONE
            self.context = internals.LlamaContext(model=self.model._model, params=params)
        self.context.kv_cache_clear()
        batch = self.model._batch
        batch.reset()
        for index, tokens in enumerate(sequences):
            batch.add_sequence(tokens, index, True)
        self.context.decode(batch)
        self.embedding_calls += 1
        self.last_hidden_references = []
        rows, offset = [], 0
        for tokens in sequences:
            states = []
            for i in range(len(tokens)):
                pointer = self.context.get_embeddings_ith(offset + i)
                if not pointer:
                    raise ValueError("hidden_states_unavailable")
                states.append(list(pointer[: self.width]))
            array = np.asarray(states)
            rows.append(array)
            if self.raw is not None:
                path = self.raw / f"hidden-{self.embedding_calls:03d}-{len(rows)}.npy"
                np.save(path, array)
                self.last_hidden_references.append(d.ref(path))
            offset += len(tokens)
        return rows

    def likelihood(self, prompt: str, continuation: str) -> Json:
        """Score every continuation token against full raw vocabulary logits in fresh state."""
        prefix, tokens = self.tokenize(prompt), self.tokenize(continuation)
        if not prefix or not tokens:
            raise ValueError("empty_likelihood_tokens")
        self.model.reset()
        self.model._ctx.kv_cache_clear()
        self.model.eval(prefix)
        probabilities, raw_logits = [], []
        for token in tokens:
            logits = np.asarray(self.model.scores[self.model.n_tokens - 1], dtype=np.float64)
            if (
                not np.isfinite(logits).all()
                or logits.shape != (self.model.n_vocab(),)
                or token >= len(logits)
            ):
                raise ValueError("nonfinite_or_wrong_logit_width")
            raw_logits.append(logits.copy())
            probabilities.append(float(logits[token] - np.logaddexp.reduce(logits)))
            self.model.eval([token])
        reference = None
        if self.raw is not None:
            key = d.canonical_hash(dict(prompt=prompt, continuation=continuation)).split(":")[1]
            path = self.raw / (key + ".npy")
            np.save(path, np.asarray(raw_logits))
            reference = d.ref(path)
        return dict(
            tokens=tokens,
            token_logprobs=probabilities,
            log_probability=sum(probabilities),
            raw_logits_finite=True,
            raw_logits_unmasked=True,
            raw_logits_width=len(raw_logits[0]),
            raw_logits_reference=reference,
        )

    def close(self) -> None:
        """Release the extra context before freeing its shared immutable model."""
        if self.context is not None:
            self.context.close()


def worker(raw: Path) -> Json:
    """Persist exact native failure and zero generation without repair or fallback."""
    import llama_cpp

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    raw.mkdir(parents=True, exist_ok=True, mode=0o700)
    start, owner = time.monotonic_ns(), os.getpid()
    events: list[Json] = []
    result: Json = dict(
        rows=[],
        blocker=None,
        gate_check_summary=[],
        embedding_calls=0,
        generated_tokens=0,
        runtime_ok=False,
        runtime_reference={},
        tokenizer_metadata={},
        raw_hidden_width=None,
        source_references=[],
    )
    lease, model, adapter = None, None, None
    stage = "cached_runtime"
    try:
        d.progress("preflight_before")
        resources = preflight(raw)
        spec, gpu = resources["model"], resources["gpu"]
        admission = build_gguf_admission_record(
            spec["model_path"],
            repository_id=d.MODEL_SPECS[0]["hf_id"],
            trusted_sha256=spec["sha256"],
            expected_architectures={"qwen35"},
        )
        if not admission["admitted"] or admission["content_metadata"]["quantization"] != "Q4_K_M":
            raise ValueError("gguf_admission")
        libraries = sorted(Path(llama_cpp.__file__).parent.rglob("*.so"))
        if not libraries:
            raise ValueError("native_build_missing")
        result["runtime_reference"] = dict(
            model=spec,
            admission=admission,
            native_libraries=[d.ref(p) for p in libraries],
            python_binding=d.ref(Path(llama_cpp.__file__)),
            system_info=llama_cpp.llama_print_system_info().decode(),
        )
        result["source_references"] = [
            d.ref(Path(spec["model_path"])),
            *result["runtime_reference"]["native_libraries"],
        ]
        result["tokenizer_metadata"] = admission["content_metadata"]["tokenizer_metadata"]
        d.progress("preflight_after")
        stage = "gpu0_lease_and_memory"
        lease = GpuLease.acquire(
            runtime_dir=os.environ.get("CARNOT_GPU_LEASE_RUNTIME_DIR", "/tmp/carnot-gpu-leases"),
            task_id=d.TASK,
            device_uuid=gpu["uuid"],
            expected_model=spec["model_path"],
            vram_before_mb=gpu["used_mb"],
            ttl_s=3400,
        )
        result["gpu_lease"] = lease.owner_receipt()
        locked = gpu_state()
        result["memory_before"] = locked
        if locked["index"] != 0 or locked["uuid"] != gpu["uuid"] or locked["free_mb"] < 20000:
            raise ValueError("gpu0_memory_or_identity")
        lease.transition("admitted")
        lease.transition("loading")
        stage = "model_load"
        event = dict(
            call_id="frozen-qwen",
            operation="model_load",
            scope="current",
            transport="owned_runtime",
            run_id=raw.name,
            owner_pid=owner,
        )
        events.append(dict(event, state="attempted", monotonic_ns=time.monotonic_ns()))
        atomic_json(raw / "events.json", dict(events=events))
        d.progress("model_load_before")
        try:
            model = llama_cpp.Llama(
                model_path=spec["model_path"],
                n_ctx=2048,
                n_batch=1536,
                n_ubatch=1536,
                n_gpu_layers=-1,
                main_gpu=0,
                split_mode=llama_cpp.LLAMA_SPLIT_MODE_NONE,
                tensor_split=[1.0],
                logits_all=True,
                embedding=False,
                seed=7769098,
                verbose=True,
            )
        except (ValueError, RuntimeError, OSError):
            events.append(dict(event, state="failed", monotonic_ns=time.monotonic_ns()))
            raise
        events.append(dict(event, state="completed", monotonic_ns=time.monotonic_ns()))
        d.progress("model_load_after")
        stage = "actual_cuda_offload"
        resident = gpu_state()
        result["memory_resident"] = resident
        log = Path(os.readlink("/proc/self/fd/2")).read_text(errors="replace")
        offload = re.findall(r"offloaded (\d+)/(\d+) layers", log)
        result["offload_layers"] = offload
        if (
            not offload
            or int(offload[-1][0]) <= 0
            or offload[-1][0] != offload[-1][1]
            or resident["used_mb"] - locked["used_mb"] < 1000
        ):
            raise ValueError("missing_actual_cuda_offload")
        lease.transition("resident", vram_mb=resident["used_mb"])
        lease.transition("inferencing")
        stage = "hidden_width"
        adapter = FrozenRuntime(model, raw)
        result["raw_hidden_width"] = adapter.width
        if adapter.width != int(model.metadata["qwen35.embedding_length"]):
            raise ValueError("wrong_hidden_width")
        result["runtime_ok"] = True
        stage = "fixed_fixture_forwards"
        result["rows"] = d.exercise(adapter, deadline=time.monotonic() + 3000)
        result["embedding_calls"] = adapter.embedding_calls
    except (OSError, ValueError, RuntimeError, TimeoutError, KeyError, LeaseError) as error:
        result["blocker"] = stage
        observed = (
            "missing_field"
            if isinstance(error, KeyError)
            else "missing_file"
            if str(error) in {"cache_missing", "native_build_missing", "mandated_Q4_K_M_missing"}
            else str(error)
        )
        result["gate_check_summary"].append(
            d.gate(
                raw,
                stage,
                "cached frozen GPU0 Qwen with supported raw capability",
                observed,
                stage,
                d.TASK,
            )
        )
        d.progress("blocked_" + stage)
    finally:
        d.progress("model_cleanup_before")
        if adapter is not None:
            result["embedding_calls"] = adapter.embedding_calls
            adapter.close()
        if model is not None:
            model.close()
        if lease is not None:
            if lease.document["phase"] == "inferencing":
                lease.transition("unloading")
                lease.transition(
                    "validating", vram_mb=gpu_state()["used_mb"], exit_code=0, unload_observed=True
                )
            lease.transition("terminal_blocked" if result["blocker"] else "terminal_complete")
            result["lease_release"] = lease.release()
            atomic_json(raw / "gpu_lease.json", lease.document)
            lease.close()
        if not result["rows"]:
            result["rows"] = d.censored(result["blocker"] or "unavailable_runtime")
        result["model_loads"] = sum(e["state"] == "attempted" for e in events)
        result["current_work_receipt"] = build_current_work_receipt(
            run_id=raw.name,
            owner_pid=owner,
            events=events,
            inference_substrate="frozen_gguf_forward_scoring",
            inference_substrate_details=dict(
                generated_tokens=0, embedding_calls=result["embedding_calls"]
            ),
            inference_substrate_class="model_load_no_generation",
            execution_venue="host_gpu0",
            started_monotonic_ns=start,
            ended_monotonic_ns=time.monotonic_ns(),
        )
        result["duration_s"] = (time.monotonic_ns() - start) / 1e9
        atomic_json(raw / "runtime.json", result)
        d.progress("model_cleanup_after")
    return result
