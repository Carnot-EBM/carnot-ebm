"""REQ-REPORT-8022: own one cached Qwen load and only teacher-forced forwards.

The bounded parent preserves native failure logs. A failed capability ends this
invocation without generation, model substitution or repeated qualification.
"""

from __future__ import annotations

import json
from pathlib import Path
import time
from typing import Any

from carnot.gpu_lease_phase_journal import GpuLease, LeaseError
from carnot.inference import fixed_answer_likelihood_8022 as s
from carnot.inference.qwen_sufficiency_7920 import bounded
from carnot.inference.sota_models import cached_current_model
from carnot.reporting.current_work_receipt import atomic_json, sha256_file
from carnot.reporting.experiment_7303_validation_scope import CommandSpec, run_commands
from carnot.verify.qwen_development_capture_7995 import Ledger

Json = dict[str, Any]
ROOT = Path(__file__).resolve().parents[3]
MODEL = "unsloth/Qwen3.8-27B-GGUF"
TASK = "exp8022-likelihood-protocol"


class EmbeddedRuntime:
    """Render the GGUF template without calling a completion or sampling API."""

    def __init__(self, model: Any) -> None:
        from llama_cpp.llama_chat_format import Jinja2ChatFormatter

        self.model = model
        self.formatter = Jinja2ChatFormatter(
            template=model.metadata["tokenizer.chat_template"],
            eos_token=model.detokenize([model.token_eos()], special=True).decode(),
            bos_token=model.detokenize([model.token_bos()], special=True).decode(),
        )

    def __getattr__(self, name: str) -> Any:
        return getattr(self.model, name)

    def render(self, source: bytes, question: bytes) -> bytes:
        """A common question prevents the intervention from changing the task."""
        return self.formatter(
            messages=[
                dict(role="user", content=question.decode() + "\nSource:\n" + source.decode())
            ],
            enable_thinking=False,
        ).prompt.encode()

    def eval(self, tokens: list[int]) -> None:
        """An uncertain forward ends the capability instead of enabling a retry."""
        bounded(lambda: self.model.eval(tokens), 120)


def worker(public_path: Path, output: Path) -> None:
    """Persist every gate and cleanup, even when the native loader rejects Qwen."""
    import llama_cpp
    from llama_cpp import llama_cpp as native

    started = time.monotonic()
    raw = output.parent
    result: Json = dict(
        checks=[],
        panel={},
        qualification={},
        cleanup={},
        current_invocation_ledger=[],
        model_identity_receipt={},
        gguf_sha256=None,
        model_revision=None,
        gpu_lease_receipt={},
        offload_evidence={},
        duration_s=0,
    )
    ledger = Ledger(raw / "ledger.json")
    ledger.save()
    lease: Any = None
    model: Any = None
    try:
        spec = cached_current_model()
        if spec is None or spec["hf_id"] != MODEL:
            raise ValueError("mandated_cache_missing")
        path = Path(spec["model_path"])
        s.progress("before_model_hash", started)
        digest = str(bounded(lambda: sha256_file(path), 120))
        result.update(gguf_sha256=digest, model_revision=path.parent.name)
        s.progress("after_model_hash", started)
        capacity = run_commands(
            ROOT,
            [
                CommandSpec(
                    "capacity",
                    (
                        "nvidia-smi",
                        "--query-gpu=index,uuid,memory.used,memory.free",
                        "--format=csv,noheader,nounits",
                    ),
                    "runtime",
                    10,
                )
            ],
            log_dir=raw / "capacity",
            heartbeat_s=5,
        )[0]
        devices = [
            line.split(",")
            for line in capacity["output_tail"].splitlines()
            if len(line.split(",")) == 4
            and int(line.split(",")[2]) < 1000
            and int(line.split(",")[3]) >= 20000
        ]
        result["capacity_receipt"] = capacity
        if not native.llama_supports_gpu_offload() or not devices:
            raise ValueError("cuda_offload_or_memory_unavailable")
        device = devices[0]
        lease = GpuLease.acquire(
            runtime_dir="/tmp/carnot-gpu-leases",
            task_id=TASK,
            device_uuid=device[1].strip(),
            expected_model=str(path),
            vram_before_mb=int(device[2]),
            ttl_s=660,
        )
        result["gpu_lease_receipt"] = lease.owner_receipt()
        lease.transition("admitted")
        lease.transition("loading")
        ledger.start("model_load", "qwen-vocabulary", dict(model_path=str(path), vocab_only=True))
        s.progress("before_tokenizer_model_load", started)
        vocabulary = bounded(
            lambda: llama_cpp.Llama(model_path=str(path), vocab_only=True, verbose=False), 60
        )
        ledger.finish("qwen-vocabulary", "completed", {})
        s.progress("after_tokenizer_model_load", started)
        try:
            result["panel"] = s.freeze_panel(
                json.loads(public_path.read_text()), EmbeddedRuntime(vocabulary)
            )
            atomic_json(raw / "panel.json", result["panel"])
        finally:
            vocabulary.close()
        ledger.start("model_load", "qwen-load", dict(model_path=str(path), gguf_sha256=digest))
        s.progress("before_model_load", started)
        model = bounded(
            lambda: llama_cpp.Llama(
                model_path=str(path),
                n_ctx=6384,
                n_batch=256,
                n_gpu_layers=-1,
                main_gpu=int(device[0]),
                tensor_split=[1.0 if i == int(device[0]) else 0.0 for i in range(2)],
                logits_all=True,
                seed=69522,
                verbose=True,
            ),
            300,
        )
        ledger.finish("qwen-load", "completed", {})
        s.progress("after_model_load", started)
        resident = run_commands(
            ROOT,
            [
                CommandSpec(
                    "resident",
                    (
                        "nvidia-smi",
                        f"--id={device[0].strip()}",
                        "--query-gpu=memory.used",
                        "--format=csv,noheader,nounits",
                    ),
                    "runtime",
                    10,
                )
            ],
            log_dir=raw / "resident",
            heartbeat_s=5,
        )[0]
        memory = int(resident["output_tail"].strip())
        result["offload_evidence"] = dict(
            supported=True,
            resident_memory_mb=memory,
            resident_receipt=resident,
            n_gpu_layers=-1,
            native_library=str(native._lib._name),
        )
        if memory < 10000 or not hasattr(model, "scores") or not model._logits_all:
            raise ValueError("teacher_forced_logits_or_offload_unavailable")
        lease.transition("resident", vram_mb=memory)
        result["model_identity_receipt"] = dict(
            hf_id=MODEL,
            model_path=str(path),
            gguf_sha256=digest,
            model_revision=path.parent.name,
            tokenizer="embedded_GGUF",
            api="Llama.eval / scores[position-1]",
            runtime_version=llama_cpp.__version__,
            chat_template_sha256=s.canonical_hash(model.metadata["tokenizer.chat_template"]),
            bos_token_id=model.token_bos(),
            eos_token_id=model.token_eos(),
            vocabulary_size=model.n_vocab(),
        )
        runtime = EmbeddedRuntime(model)
        lease.transition("inferencing")
        s.progress("before_benchmark", started, 0, 8)
        result["qualification"] = s.qualify(runtime, raw / "forwards")
        s.progress("after_benchmark", started, 8)
    except (OSError, RuntimeError, TimeoutError, ValueError, KeyError, LeaseError) as error:
        s.progress("after_runtime_phase_failed", started)
        result["checks"].append(
            dict(
                artifact_field="teacher_forced_capability",
                expected=True,
                observed=f"{type(error).__name__}:{error}",
                passed=False,
                upstream_id="cached_qwen_runtime",
                path=str(output),
                hash=None,
            )
        )
        if ledger.rows and ledger.rows[-1]["status"] == "running":
            ledger.finish(ledger.rows[-1]["call_id"], "failed", {})
    finally:
        s.progress("before_model_unload", started)
        if model is not None:
            model.close()
        if lease is not None:
            if lease.document["phase"] in {"resident", "inferencing"}:
                lease.transition("unloading")
                unloaded = run_commands(
                    ROOT,
                    [
                        CommandSpec(
                            "unloaded",
                            (
                                "nvidia-smi",
                                f"--id={device[0].strip()}",
                                "--query-gpu=memory.used",
                                "--format=csv,noheader,nounits",
                            ),
                            "runtime",
                            10,
                        )
                    ],
                    log_dir=raw / "unloaded",
                    heartbeat_s=5,
                )[0]
                result["offload_evidence"]["unloaded_receipt"] = unloaded
                lease.transition(
                    "validating",
                    vram_mb=int(unloaded["output_tail"].strip()),
                    exit_code=0,
                    unload_observed=True,
                )
            lease.transition(
                "terminal_complete" if result["qualification"].get("passed") else "terminal_blocked"
            )
            result["gpu_lease_receipt"]["release"] = lease.release()
            atomic_json(raw / "gpu_lease.json", lease.document)
        result.update(
            cleanup=dict(model_closed=model is not None, lease_released=lease is not None),
            current_invocation_ledger=ledger.rows,
            model_invocation_counts=ledger.counts(),
            duration_s=time.monotonic() - started,
        )
        result["forward_attempts"] = [
            json.loads(p.read_text()) for p in sorted((raw / "forwards").glob("*.json"))
        ]
        atomic_json(output, result)
        s.progress("after_model_unload", started)
