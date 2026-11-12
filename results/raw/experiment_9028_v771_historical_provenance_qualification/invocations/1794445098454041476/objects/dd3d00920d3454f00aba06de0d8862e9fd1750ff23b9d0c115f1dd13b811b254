"""Own a bounded Qwen server so a shared model cannot supply task evidence.

REQ-REPORT-7920-V687. The embedded tokenizer and launch path bind each request
with the actual CUDA worker. Timeouts apply to owned work without time padding.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from queue import Empty, Queue
import re
import socket
import threading
import time
from typing import Any, Callable
from urllib.request import urlopen

from carnot.inference.llama_cpp_process import OwnedLlamaCppProcess
from carnot.reporting.current_work_receipt import canonical_hash, sha256_file


def bounded(call: Callable[[], Any], timeout_s: float, *, interval_s: float = 15) -> Any:
    """Keep visible heartbeats during blocking native work and propagate failures."""
    queue: Queue[Any] = Queue()

    def invoke() -> None:
        try:
            queue.put((True, call()))
        except BaseException as error:
            queue.put((False, error))

    threading.Thread(target=invoke, daemon=True).start()
    started = time.monotonic()
    while True:
        remaining = timeout_s - (time.monotonic() - started)
        if remaining <= 0:
            raise TimeoutError("owned_deadline")
        try:
            ok, result = queue.get(timeout=min(interval_s, remaining))
        except Empty:
            print(f"[exp7920] heartbeat elapsed_s={time.monotonic() - started:.3f}", flush=True)
            continue
        if not ok:
            raise result
        return result


def get_json(url: str) -> dict[str, Any]:
    """Read the private worker identity with a short connection deadline."""
    with urlopen(url, timeout=5) as response:
        return dict(json.loads(response.read()))


def offload_layers(log: str) -> list[int]:
    """Report observed offload instead of inferring CUDA use from launch flags."""
    matches = re.findall(r"offloaded (\d+)/(\d+) layers to GPU", log)
    return list(map(int, matches[-1])) if matches else [0, 0]


class QwenRuntime:
    """Keep all model-visible traffic on one private owned server."""

    def __init__(self, model: Path, scratch: Path, gpu: int) -> None:
        self.model = model
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            self.port = sock.getsockname()[1]
        self.log = scratch / "server.log"
        self.worker: Any = None
        self.scratch = scratch
        self.gpu = gpu
        self.receipts: list[dict[str, Any]] = []
        self.command = [
            str(Path.home() / ".cache/llama.cpp-master/build/bin/llama-server"),
            "-m",
            str(model),
            "--host",
            "127.0.0.1",
            "--port",
            str(self.port),
            "-c",
            "8192",
            "-ngl",
            "99",
            "--parallel",
            "1",
            "--alias",
            "unsloth/Qwen3.8-27B-GGUF",
            "--jinja",
            "-lv",
            "5",
        ]

    def load(self) -> dict[str, Any]:
        """Authenticate the actual path, template and complete CUDA offload."""
        print("[exp7920] before_model_load", flush=True)
        started = time.monotonic()
        self.worker = OwnedLlamaCppProcess(
            command=self.command,
            port=self.port,
            env={**os.environ, "CUDA_VISIBLE_DEVICES": str(self.gpu)},
            log_path=self.log,
            state_path=self.scratch / "owner.json",
        )
        owner = self.worker.launch()
        health = bounded(lambda: self.worker.wait_for_health(300), 300)
        if not health["ok"]:
            raise RuntimeError("model_load_failed")
        props = get_json(f"http://127.0.0.1:{self.port}/props")
        offload = offload_layers(self.log.read_text())
        authenticated = (
            props.get("model_path") == str(self.model)
            and bool(props.get("chat_template"))
            and offload[0] == offload[1]
            and offload[0] > 0
        )
        print("[exp7920] after_model_load", flush=True)
        if not authenticated:
            raise RuntimeError("served_model_identity_or_offload")
        return dict(
            authenticated=True,
            owner=owner,
            props=props,
            command=self.command,
            gguf_sha256=sha256_file(self.model),
            tokenizer="embedded_GGUF",
            quantization="Q4_K_M",
            offload_layers=offload,
            duration_s=time.monotonic() - started,
        )

    def count(self, text: str) -> int:
        """Count the rendered chat template, including role and special tokens."""
        if text.startswith("[{") or text.startswith("[ {"):
            messages = json.loads(text)
            text = self.worker.post_json(
                "/apply-template",
                dict(
                    messages=messages,
                    add_generation_prompt=True,
                    chat_template_kwargs=dict(enable_thinking=False),
                ),
                10,
            )["prompt"]
        tokens = self.worker.post_json("/tokenize", dict(content=text, add_special=True), 10)[
            "tokens"
        ]
        return len(tokens)

    def generate(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Seal a real response receipt with actual token and monotonic counts."""
        begin = time.monotonic_ns()
        response = bounded(lambda: self.worker.post_json("/v1/chat/completions", payload, 120), 120)
        self.receipts.append(
            dict(
                request_sha256=canonical_hash(payload),
                response_sha256=canonical_hash(response),
                started_monotonic_ns=begin,
                ended_monotonic_ns=time.monotonic_ns(),
                input_tokens=response.get("usage", {}).get("prompt_tokens"),
                output_tokens=response.get("usage", {}).get("completion_tokens"),
                server_identity=self.worker.receipt,
            )
        )
        return dict(response)

    def close(self) -> dict[str, Any]:
        """Stop only the recorded worker and confirm its private port is free."""
        return dict(self.worker.cleanup()) if self.worker else dict(leak_free=True)
