"""REQ-REPORT-8934: own GPU zero and preserve each stateless response before labels."""

from __future__ import annotations

import json
import os
from pathlib import Path
import time
from typing import Any
from urllib.request import Request, urlopen

from carnot.gpu_lease_phase_journal import GpuLease, LeaseError
from carnot.inference import bfs_pilot_runtime_8918 as p
from carnot.inference.qwen_sufficiency_7920 import QwenRuntime, bounded
from carnot.reporting import graph_source_8931 as source
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash
from carnot.verify.qwen_development_capture_7995 import Ledger

Json = dict[str, Any]


def wire_response(port: int, payload: Json, timeout: float, path: Path) -> Json:
    """Keep exact response bytes so later parsing cannot hide transport evidence."""
    request = Request(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urlopen(request, timeout=timeout) as response:
        body = response.read()
    path.write_bytes(body)
    return dict(json.loads(body))


def run(units: list[Json], raw: Path) -> Json:
    """Use the pilot's authenticated CUDA path without retries, repair or CPU fallback."""
    os.environ.update(CARNOT_FORCE_LIVE="1", CUDA_VISIBLE_DEVICES="0")
    raw.mkdir(parents=True, exist_ok=True)
    ledger = Ledger(raw / "invocation_ledger.json")
    ledger.save()
    result: Json = dict(
        rows=[],
        checks=[],
        live=False,
        blocker=None,
        model_receipt={},
        resource_lease={},
        phase_spans=[],
        settings=dict(source.SETTINGS),
        owned_failure=False,
    )
    lease: Any = None
    server: Any = None
    stage, inferencing = "preflight", False
    try:
        source.progress("before_preflight", 0, len(units))
        began = time.monotonic()
        resources = bounded(lambda: p.preflight(raw), 240)
        result["preconditions_checked"] = resources
        result["phase_spans"].append(dict(phase=stage, duration_s=time.monotonic() - began))
        source.progress("after_preflight", 0, len(units))
        spec, gpu = resources["model"], resources["gpu"]
        stage = "gpu_lease"
        lease = GpuLease.acquire(
            runtime_dir=os.environ.get("CARNOT_GPU_LEASE_RUNTIME_DIR", "/tmp/carnot-gpu-leases"),
            task_id="exp8934-depth-control",
            device_uuid=gpu["uuid"],
            expected_model=spec["model_path"],
            vram_before_mb=gpu["used_mb"],
            ttl_s=2400,
        )
        result["resource_lease"] = lease.owner_receipt()
        locked = p.gpu_state()
        if (
            locked["uuid"] != gpu["uuid"]
            or locked["index"] != 0
            or locked["used_mb"] >= 1000
            or locked["free_mb"] < 20000
        ):
            raise RuntimeError("gpu_0_not_free")
        lease.transition("admitted")
        lease.transition("loading")
        stage = "model_load"
        server = QwenRuntime(Path(spec["model_path"]), raw, 0)
        ledger.start("model_load", "qwen-load", spec)
        source.progress("before_model_load", 0, len(units))
        began = time.monotonic()
        try:
            loaded = bounded(server.load, 240)
        except (RuntimeError, OSError, ValueError, TimeoutError) as error:
            ledger.finish("qwen-load", "failed", dict(error=str(error)))
            raise
        ledger.finish("qwen-load", "completed", loaded)
        result["phase_spans"].append(dict(phase=stage, duration_s=time.monotonic() - began))
        result["model_receipt"] = dict(loaded, preflight_gguf_sha256=spec["sha256"])
        source.progress("after_model_load", 0, len(units))
        if (
            not loaded["authenticated"]
            or loaded["gguf_sha256"] != spec["sha256"]
            or loaded["offload_layers"][0] <= 0
            or loaded["offload_layers"][0] != loaded["offload_layers"][1]
        ):
            raise RuntimeError("actual_offload_or_hash_rejected")
        result["settings"].update(
            chat_template_sha256=canonical_hash(loaded["props"]["chat_template"]),
            per_request_timeout_s=30,
            total_generation_budget_s=1800,
            retries=0,
            stateless=True,
        )
        lease.transition("resident", vram_mb=p.gpu_state()["used_mb"])
        lease.transition("inferencing")
        inferencing = True
        stage = "generation"
        cursor = 0
        signal_blocked = False
        transport_blocker: str | None = None

        def transport(payload: Json, timeout: float) -> Json:
            nonlocal cursor, signal_blocked, transport_blocker
            if signal_blocked or transport_blocker:
                raise RuntimeError(transport_blocker or "unsupported_logprobs_capability")
            unit = units[cursor]
            cursor += 1
            ledger.start("generation", unit["unit_id"], payload)
            try:
                response = wire_response(
                    server.port,
                    payload,
                    timeout,
                    raw / "collection" / unit["unit_id"] / "wire_response.json",
                )
                ledger.finish(unit["unit_id"], "completed", response)
            except (RuntimeError, OSError, ValueError, TimeoutError) as error:
                ledger.finish(unit["unit_id"], "failed", dict(error=str(error)))
                raise
            usage = response.get("usage", {})
            if (
                type(usage.get("completion_tokens")) is not int
                or type(usage.get("prompt_tokens")) is not int
            ):
                transport_blocker = "actual_token_usage_missing"
                raise ValueError(transport_blocker)
            if response.get("model") != source.SETTINGS["model"] or not response.get("id"):
                transport_blocker = "response_identity"
                result["owned_failure"] = True
                raise ValueError(transport_blocker)
            try:
                source.signals(response)
            except ValueError:
                signal_blocked = True
            lease.heartbeat()
            return response

        began = time.monotonic()
        result["rows"] = source.collect(units, transport, raw / "collection", spec)
        result["phase_spans"].append(dict(phase=stage, duration_s=time.monotonic() - began))
        for row in result["rows"]:
            path = raw / "collection" / row["unit_id"] / "response.json"
            if path.is_file():
                row["actual_token_usage"] = json.loads(path.read_text())["raw_response"].get(
                    "usage"
                )
        result["live"] = ledger.counts()["generation_calls_completed"] > 0
        if signal_blocked or transport_blocker:
            raise RuntimeError(transport_blocker or "unsupported_logprobs_capability")
    except (RuntimeError, OSError, ValueError, KeyError, TimeoutError, LeaseError) as error:
        result["blocker"] = str(error)
        result["checks"].append(
            dict(
                check=stage,
                upstream="exp8934-depth-control",
                artifact_field=stage,
                expected="authenticated GPU-0 Qwen with supported log probabilities",
                observed=str(error),
                path=str(raw),
                hash=None,
            )
        )
        source.progress(
            "runtime_blocked_" + stage, len(result["rows"]), len(units) - len(result["rows"])
        )
    finally:
        try:
            if server is not None:
                source.progress("before_model_shutdown")
                result["cleanup"] = server.close()
                result["owned_failure"] = (
                    result["owned_failure"] or not result["cleanup"]["leak_free"]
                )
                source.progress("after_model_shutdown")
        except (RuntimeError, OSError, ValueError) as error:
            result["owned_failure"] = True
            result["cleanup"] = dict(error=str(error), leak_free=False)
        finally:
            if lease is not None:
                try:
                    if inferencing:
                        lease.transition("unloading")
                        lease.transition(
                            "validating",
                            vram_mb=p.gpu_state()["used_mb"],
                            exit_code=0,
                            unload_observed=not result["owned_failure"],
                        )
                    lease.transition(
                        "terminal_blocked"
                        if result["blocker"] or result["owned_failure"]
                        else "terminal_complete"
                    )
                    result["resource_lease"]["release"] = lease.release()
                except LeaseError as error:
                    result["owned_failure"] = True
                    result["resource_lease"]["release"] = dict(released=False, error=str(error))
                finally:
                    lease.close()
        if not result["rows"]:
            result["rows"] = [
                source.censored(u, result["blocker"] or "runtime_unavailable") for u in units
            ]
        result.update(counts=ledger.counts(), ledger=ledger.rows)
        atomic_json(raw / "runtime.json", result)
    return result
