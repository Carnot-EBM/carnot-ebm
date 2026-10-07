"""REQ-VERIFY-8227: isolate independent requests while retaining durable identity.

Two fixed worker queues prevent simultaneous reuse of a slot. The existing
journal remains the authority for issues and terminal responses, including errors.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import threading
import time
from typing import Any, Callable
from urllib.request import Request, urlopen

from carnot.verify import request_recorder_8213 as recorder
from carnot.verify.recorder_fixtures_8213 import synthetic

Json = dict[str, Any]
ROOT = recorder.ROOT
NAME = "experiment_8227_v711_concurrency_canary"
TASK = "exp8227-concurrency-canary"
CLI = f"scripts/experiments/{NAME}.py"
TEST = "tests/python/test_concurrency_canary_8227.py"
MODULES = [
    "python/carnot/verify/concurrency_canary_8227.py",
    "python/carnot/reporting/concurrency_execution_8227.py",
    "python/carnot/inference/concurrency_runtime_8227.py",
]
SEED = 7118227
MODEL_SPECS = [
    dict(
        hf_id="unsloth/Qwen3.8-27B-GGUF",
        quantization="Q4_K_M",
        backend="llama.cpp CUDA",
        role="headline",
    )
]
CONFIG = dict(
    max_tokens=128,
    context_per_slot=4096,
    server_slots=2,
    temperature=0,
    seed=SEED,
    request_timeout_s=120,
    canary_cap_s=900,
    duration_floor_s=10,
    cache_prompt=False,
    cache_ram_mb=0,
    cache_reuse=0,
    retries=0,
)
atomic_json, key, sha256_file, progress = (
    recorder.atomic_json,
    recorder.key,
    recorder.sha256_file,
    recorder.progress,
)


def freeze(roster: list[Json], identity: Json) -> Json:
    """Select public fit identities before current outcomes can influence roles."""
    fit = sorted([r for r in roster if r["role"] == "fit"], key=lambda r: r["source_cluster_id"])
    if len(fit) < 28 or len({r["source_cluster_id"] for r in fit[:28]}) != 28:
        raise ValueError("source_count")
    canary = recorder.schedule(fit[:24], identity)["rows"][:4]
    benchmark = recorder.schedule(fit[4:28], identity)["rows"]
    order = ["s1_serial", "s1_concurrent", "s2_concurrent", "s2_serial"]

    def workload(sources: list[Json], name: str) -> list[Json]:
        rows = []
        for i, source in enumerate(sources):
            row = deepcopy(source)
            payload = row["envelope"]["payload"]
            payload.update(seed=SEED, cache_prompt=False)
            row["envelope"] = recorder.envelope(row["envelope"], payload, identity)
            row.update(
                request_id=f"{name}-{i:02}",
                arm=name.split("_")[-1],
                workload=name,
                slot=i % (2 if name.endswith("concurrent") else 1),
            )
            rows.append(row)
        return rows

    return dict(
        schema="carnot.v711.concurrent-acquisition.v1",
        config=CONFIG,
        identity=deepcopy(identity),
        source_order="ascending source_cluster_id SHA256",
        prompt_serialization="Exp8213 canonical UTF-8 messages plus embedded GGUF template",
        canary=workload(canary, "canary_serial") + workload(canary, "canary_concurrent"),
        benchmark=[r for name in order for r in workload(benchmark, name)],
        launch_order=order,
        cold_starts_per_arm=2,
        shutdowns_per_arm=2,
        failure_rule="retain error, length limit and deadline censoring; no retries or substitution",
        benchmark_executed_here=False,
        benefit_claim=False,
    )


def acquire(
    schedule: list[Json],
    generate: Callable[[Json, int], Json],
    raw: Path,
    concurrency: int,
    *,
    cap_s: float = 900,
) -> list[Json]:
    """Fsync each issue before queued work, then preserve every original obligation."""
    journal = recorder.Journal(raw / "events.jsonl")
    lock = threading.Lock()
    began = time.monotonic()
    rows = []
    response_ids: set[str] = set()
    for source in schedule:
        issue = journal.issue(source["request_id"], source["envelope"])
        rows.append(
            dict(
                deepcopy(source),
                status="pending",
                result={},
                clocks=dict(
                    issue=issue["issued_monotonic_ns"],
                    queue=time.monotonic_ns(),
                    start=None,
                    first_token=None,
                    end=None,
                    durability=None,
                ),
            )
        )

    def worker(slot: int) -> None:
        result: Json
        for index in range(slot, len(rows), concurrency):
            row = rows[index]
            progress("8227_generation_before_" + row["request_id"], index, len(rows) - index)
            row["slot"] = slot
            clocks = row["clocks"]
            if time.monotonic() - began >= cap_s:
                status, result = "censored", dict(reason="canary_deadline")
            else:
                clocks["start"] = time.monotonic_ns()
                response: Json = {}
                try:
                    result = generate(row["envelope"]["payload"], slot)
                    response = result
                    if (
                        result["id_slot"] != slot
                        or result["usage"]["completion_tokens"] > 128
                        or result["choices"][0]["finish_reason"] != "stop"
                    ):
                        raise ValueError("slot_identity_or_completion")
                    with lock:
                        if result["id"] in response_ids:
                            raise ValueError("response_identity_cross_talk")
                        response_ids.add(result["id"])
                    clocks["first_token"] = result["first_token_monotonic_ns"]
                    if not isinstance(clocks["first_token"], int) or not (
                        clocks["start"] <= clocks["first_token"] <= time.monotonic_ns()
                    ):
                        raise ValueError("first_token_clock")
                    status = "completed"
                except Exception as error:
                    status, result = (
                        "error",
                        dict(error=f"{type(error).__name__}:{error}", response=response),
                    )
            clocks["end"] = time.monotonic_ns()
            with lock:
                terminal = journal.finish(row["request_id"], status, result)
                clocks["durability"] = time.monotonic_ns()
            row.update(status=status, result=result, terminal_sequence=terminal["sequence"])
            atomic_json(raw / (row["request_id"] + ".json"), row)
            progress("8227_generation_after_" + row["request_id"], index + 1, len(rows) - index - 1)

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [pool.submit(worker, slot) for slot in range(concurrency)]
        for future in futures:
            future.result()
    return rows


def qualify(raw: Path) -> Json:
    """Real private sockets test queue isolation without assigning model credit."""
    observed: list[Json] = []
    barrier = threading.Barrier(2)

    class Peer(BaseHTTPRequestHandler):
        def do_POST(self) -> None:  # noqa: N802
            """Observe durable issues at the peer before allowing interleaved replies."""
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            label, slot = body["label"], body["slot"]
            opened = recorder.Journal(raw / "events.jsonl")
            observed.append(dict(label=label, pending=list(opened.pending)))
            if label < 2:
                barrier.wait(timeout=5)
            if label == 2:
                self.send_error(503, "queued failure")
                return
            value = dict(
                id=f"fixture-{label}",
                id_slot=slot,
                fixture=True,
                first_token_monotonic_ns=time.monotonic_ns(),
                choices=[dict(message=dict(content=str(label)), finish_reason="stop")],
                usage=dict(prompt_tokens=1, completion_tokens=1),
            )
            body_bytes = json.dumps(value).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(body_bytes)))
            self.end_headers()
            self.wfile.write(body_bytes)

        def log_message(self, format: str, *args: Any) -> None:
            """Explicit phase lines keep evidence logs free from incidental HTTP text."""

    server = ThreadingHTTPServer(("127.0.0.1", 0), Peer)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    schedule = [
        dict(
            request_id=f"fixture-{i}",
            envelope=synthetic(i),
            source_cluster_id=f"fixture-{i}",
            arm="scripted",
        )
        for i in range(4)
    ]

    def generate(payload: Json, slot: int) -> Json:
        """The fixture uses the same bounded HTTP and journal boundary as real calls."""
        label = int(payload["messages"][0]["content"].split("-")[-1])
        req = Request(
            f"http://127.0.0.1:{server.server_port}/completion",
            data=json.dumps(dict(label=label, slot=slot)).encode(),
            method="POST",
        )
        with urlopen(req, timeout=5) as response:  # noqa: S310
            return dict(json.loads(response.read()))

    try:
        rows = acquire(schedule, generate, raw, 2)
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
    journal = recorder.Journal(raw / "events.jsonl")
    journal.issue("restart", synthetic(8))
    restarted = recorder.Journal(journal.path)
    restart_ok = list(restarted.pending) == ["restart"]
    restarted.finish("restart", "censored", dict(reason="explicit restart censor"))
    tail = raw / "partial.jsonl"
    tail.write_bytes(journal.path.read_bytes() + b'{"partial":')
    try:
        recorder.Journal(tail)
        partial_ok = False
    except ValueError:
        partial_ok = True
    checks = dict(
        interleaving=all(len(r["pending"]) == 4 for r in observed[:2]),
        identity_isolation=all(
            r["result"].get("id") == r["request_id"] for r in rows if r["status"] == "completed"
        ),
        queued_failure=rows[2]["status"] == "error" and rows[3]["status"] == "completed",
        restart=restart_ok and not recorder.Journal(journal.path).pending,
        partial_tail=partial_ok,
    )
    value = dict(
        rows=rows,
        checks=checks,
        passed=all(checks.values()),
        observed=observed,
        verdict_class="circular_positive",
        model_calls=0,
    )
    atomic_json(raw / "qualification.json", value)
    return value
