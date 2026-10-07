"""REQ-VERIFY-8227: own both slots and observe streamed tokens on one leased GPU.

Slot routing is explicit. Transport responses remain tied to their request socket,
and cleanup uses the established owner identity rather than process-name searches.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import time
from typing import Any
from urllib.request import Request, urlopen

from carnot.gpu_lease_phase_journal import GpuLease, LeaseError
from carnot.inference.gguf_metadata import read_gguf_metadata
from carnot.inference.qwen_sufficiency_7920 import QwenRuntime, bounded, get_json
from carnot.inference.sota_models import cached_current_model
from carnot.reporting.recorder_execution_8213 import execute, CommandSpec
from carnot.reporting.request_trace_inventory_8200 import operand
from carnot.verify import concurrency_canary_8227 as e

Json = dict[str, Any]
BINARY = Path.home() / ".cache/llama.cpp-master/build/bin/llama-server"


def preflight(identity: Json, raw: Path) -> Json:
    """Authenticate cached bytes and observed free memory without loading weights."""
    model = cached_current_model(preferred_quant="Q4_K_M")
    checks = [operand("cached_current_model", BINARY, True, model is not None)]
    value: Json = dict(checks=checks, model=model, receipts=[], gpu=None)
    if model is None:
        return value
    path = Path(model["model_path"])
    metadata = read_gguf_metadata(path)
    template_ref = metadata["field_provenance"]["metadata_keys"]["tokenizer.chat_template"]
    with path.open("rb") as stream:
        stream.seek(template_ref["value_offset"] + 8)
        template = stream.read(
            template_ref["value_end_offset"] - template_ref["value_offset"] - 8
        ).decode()
    checks.extend(
        [
            operand("model_sha256", path, identity["model_sha256"], e.sha256_file(path)),
            operand(
                "chat_template_sha256", path, identity["chat_template_sha256"], e.key(template)
            ),
            operand("quantization", path, "Q4_K_M", metadata["quantization"]),
            operand(
                "runtime_sha256",
                BINARY,
                identity["runtime_sha256"],
                e.sha256_file(BINARY) if BINARY.exists() else None,
            ),
        ]
    )
    commands = [
        CommandSpec(
            "gpu_memory",
            (
                "nvidia-smi",
                "--query-gpu=index,uuid,name,memory.free,memory.used",
                "--format=csv,noheader,nounits",
            ),
            "preconditions",
            15,
        ),
        CommandSpec("server_help", (str(BINARY), "--help"), "preconditions", 15),
    ]
    e.atomic_json(raw / "resource_commands.json", dict(argv=[list(c.argv) for c in commands]))
    receipts = execute(commands, raw / "resources")
    checks.extend(operand(r["name"], Path(r["stdout_path"]), 0, r["exit_code"]) for r in receipts)
    free = []
    for line in Path(receipts[0]["stdout_path"]).read_text().splitlines():
        fields = [p.strip() for p in line.split(",")]
        if len(fields) == 5 and "RTX 3090" in fields[2]:
            free.append(
                dict(
                    index=int(fields[0]),
                    uuid=fields[1],
                    name=fields[2],
                    free_mb=int(fields[3]),
                    used_mb=int(fields[4]),
                )
            )
    required_mb = (path.stat().st_size + 2**20 - 1) // 2**20 + 4096
    selected = next((g for g in free if g["free_mb"] >= required_mb and g["used_mb"] < 128), None)
    checks.append(
        operand(
            "free_rtx3090_two_slot_memory",
            Path(receipts[0]["stdout_path"]),
            True,
            selected is not None,
        )
    )
    help_text = Path(receipts[1]["stdout_path"]).read_text()
    checks.append(
        operand(
            "two_slot_cache_off_flags",
            BINARY,
            True,
            all(flag in help_text for flag in ["--parallel", "--cache-ram", "--no-cache-prompt"]),
        )
    )
    return dict(
        value,
        gpu=selected,
        receipts=receipts,
        required_mb=required_mb,
        metadata=metadata,
        template=template,
    )


def command(model: Path, raw: Path, gpu: int) -> list[str]:
    """Both arms use identical two-slot settings; only client concurrency changes."""
    runtime = QwenRuntime(model, raw, gpu)
    argv = runtime.command
    argv[argv.index("--parallel") + 1] = "2"
    return argv + [
        "--cache-ram",
        "0",
        "--cache-reuse",
        "0",
        "--no-cache-prompt",
        "--no-cache-idle-slots",
        "--slots",
    ]


def generate(runtime: QwenRuntime, payload: Json, slot: int) -> Json:
    """Read native streaming bytes so first-token time is observed, not inferred."""
    prompt = runtime.worker.post_json(
        "/apply-template",
        dict(
            messages=payload["messages"],
            add_generation_prompt=True,
            chat_template_kwargs=dict(enable_thinking=False),
        ),
        10,
    )["prompt"]
    native = dict(
        prompt=prompt,
        n_predict=128,
        temperature=0,
        seed=e.SEED,
        cache_prompt=False,
        stream=True,
        id_slot=slot,
        grammar=payload["grammar"],
    )
    req = Request(
        f"http://127.0.0.1:{runtime.port}/completion",
        data=json.dumps(native).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    events, wire, text, first = [], [], "", None
    connection_started = time.monotonic_ns()
    with urlopen(req, timeout=120) as response:  # noqa: S310
        for line in response:
            wire.append(line.hex())
            if not line.startswith(b"data: "):
                continue
            event = json.loads(line[6:])
            events.append(event)
            content = event.get("content", "")
            if content and first is None:
                first = time.monotonic_ns()
            text += content
    terminal = events[-1]
    return dict(
        id=e.key(dict(wire=wire, connection_started_ns=connection_started)),
        native_response_id=terminal.get("id"),
        id_slot=terminal["id_slot"],
        first_token_monotonic_ns=first,
        choices=[
            dict(
                message=dict(content=text),
                finish_reason="stop"
                if terminal.get("stop") and not terminal.get("stopped_limit")
                else "length",
            )
        ],
        usage=dict(
            prompt_tokens=terminal["tokens_evaluated"],
            completion_tokens=terminal["tokens_predicted"],
        ),
        wire_bytes_hex=wire,
        native_events=events,
        rendered_prompt_sha256=e.key(prompt),
    )


def live(protocol: Json, resources: Json, raw: Path) -> Json:
    """Bound all canary work and preserve lease, load, token and teardown evidence."""
    gpu, model = resources["gpu"], Path(resources["model"]["model_path"])
    value: Json = dict(
        rows=[],
        checks=[],
        loads=[],
        shutdowns=[],
        phase_spans=[],
        gpu_lease={},
        actual_launch_order=[],
        resource_receipts=[],
    )
    began = time.monotonic()
    try:
        lease = GpuLease.acquire(
            runtime_dir=os.environ.get("CARNOT_GPU_LEASE_RUNTIME_DIR", "/tmp/carnot-gpu-leases"),
            task_id=e.TASK,
            device_uuid=gpu["uuid"],
            expected_model=str(model),
            vram_before_mb=gpu["used_mb"],
            ttl_s=1200,
        )
    except LeaseError as error:
        value["checks"].append(operand("gpu_lease", model, "free owned lease", str(error)))
        return value
    value["gpu_lease"] = lease.owner_receipt()
    lease.transition("admitted")
    lease.transition("loading")
    try:
        for arm in ["serial", "concurrent"]:
            name = "canary_" + arm
            runtime = QwenRuntime(model, raw / name, gpu["index"])
            runtime.command = protocol["server_argv"][name]
            runtime.port = int(runtime.command[runtime.command.index("--port") + 1])
            runtime.scratch.mkdir(parents=True, exist_ok=True)
            value["actual_launch_order"].append(name)
            start = time.monotonic_ns()
            e.progress("8227_model_load_before_" + name, 0, 1)
            value["loads"].append(dict(workload=name, attempted=True, completed=False))
            try:
                loaded = runtime.load()
                value["loads"][-1].update(completed=True, receipt=loaded)
                props = loaded["props"]
                if (
                    props.get("total_slots") != 2
                    or props["default_generation_settings"]["n_ctx"] != 4096
                    or e.key(props["chat_template"]) != protocol["identity"]["chat_template_sha256"]
                ):
                    raise RuntimeError("served_two_slots_context_template")
                mem = execute(
                    [
                        CommandSpec(
                            name + "_resident",
                            (
                                "nvidia-smi",
                                f"--id={gpu['index']}",
                                "--query-gpu=memory.used",
                                "--format=csv,noheader,nounits",
                            ),
                            "canary_preconditions",
                            15,
                        )
                    ],
                    raw / name,
                )
                value["resource_receipts"].extend(mem)
                used = int(Path(mem[0]["stdout_path"]).read_text().strip())
                if used < model.stat().st_size // 2**20 or used >= gpu["free_mb"]:
                    raise RuntimeError("actual_two_slot_vram")
                if arm == "serial":
                    lease.transition("resident", vram_mb=used)
                    lease.transition("inferencing")
                e.progress("8227_model_load_after_" + name, 1, 0)
                value["phase_spans"].append(
                    dict(phase="model_load", start_ns=start, end_ns=time.monotonic_ns())
                )
                e.progress("8227_canary_before_" + name, 0, 4)
                slots = [r for r in protocol["canary"] if r["arm"] == arm]
                start = time.monotonic_ns()
                value["rows"].extend(
                    e.acquire(
                        slots,
                        lambda payload, slot: bounded(
                            lambda: generate(runtime, payload, slot), 120
                        ),
                        raw / name / "requests",
                        1 if arm == "serial" else 2,
                        cap_s=max(0, 900 - (time.monotonic() - began)),
                    )
                )
                value["phase_spans"].append(
                    dict(phase="generation", start_ns=start, end_ns=time.monotonic_ns())
                )
                e.progress("8227_canary_after_" + name, 4, 0)
            finally:
                start = time.monotonic_ns()
                e.progress("8227_shutdown_before_" + name, 0, 1)
                cleanup = runtime.close()
                value["shutdowns"].append(dict(workload=name, receipt=cleanup))
                value["phase_spans"].append(
                    dict(phase="shutdown", start_ns=start, end_ns=time.monotonic_ns())
                )
                value["checks"].append(
                    operand(name + "_cleanup", runtime.log, True, cleanup["leak_free"])
                )
                e.progress("8227_shutdown_after_" + name, 1, 0)
    except (RuntimeError, OSError, KeyError, ValueError, TimeoutError) as error:
        value["checks"].append(
            operand("owned_two_slot_server", model, "supported bounded runtime", str(error))
        )
    finally:
        if lease.document["phase"] == "inferencing":
            lease.transition("unloading")
            lease.transition(
                "validating",
                vram_mb=gpu["used_mb"],
                exit_code=0,
                unload_observed=all(r["receipt"]["leak_free"] for r in value["shutdowns"]),
            )
        lease.transition(
            "terminal_blocked"
            if any(not c["passed"] for c in value["checks"])
            else "terminal_complete"
        )
        value["gpu_lease_release"] = lease.release()
    value["measurement_duration_s"] = time.monotonic() - began
    e.atomic_json(raw / "live.json", value)
    return value
