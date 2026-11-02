"""SCENARIO-VERIFY-8213-JOIN: synthetic HTTP bytes exercise real service code.

Fixture probabilities certify joins and durability only. They carry no model
measurement or evidence about natural deployment request frequency.
"""

from __future__ import annotations

from copy import deepcopy
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import threading
import time
from typing import Any

from carnot.verify import request_recorder_8213 as e
from carnot.verify import durable_batch_8159 as host

Json = dict[str, Any]


def synthetic(index: int) -> Json:
    """Explicit synthetic identity prevents a fixture from looking like Qwen."""
    slot = dict(
        source_cluster_id=f"fixture-{index}",
        source_bytes=b"synthetic source".hex(),
        answer_bytes=b"synthetic answer".hex(),
        condition="original",
    )
    payload = dict(
        model="scripted_http_fixture",
        messages=[dict(role="user", content=f"fixture-{index}")],
        max_tokens=128,
        seed=e.SEED,
        temperature=0,
        stream=False,
    )
    identity = dict(
        model_sha256=e.key("fixture model"),
        chat_template_sha256=e.key("fixture template"),
        runtime_sha256=e.key("scripted HTTP peer"),
    )
    return e.envelope(slot, payload, identity)


def qualify(data: Json, raw: Path) -> Json:
    """Exercise actual sockets, restarts and the existing native service signatures."""
    raw.mkdir(parents=True, exist_ok=True)
    schedule_roundtrip_rows = []
    for row in data["schedule"]["rows"]:
        reopened = json.loads(json.dumps(row["envelope"]))
        e.validate_envelope(reopened)
        schedule_roundtrip_rows.append(
            dict(
                request_id=row["request_id"],
                semantic_key=e.key(reopened),
                passed=reopened == row["envelope"],
            )
        )
    journal = e.Journal(raw / "events.jsonl")
    observed: list[Json] = []

    class Peer(BaseHTTPRequestHandler):
        def do_POST(self) -> None:  # noqa: N802
            """Reopen fsynced issue bytes at the peer before returning a response."""
            payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            current = e.Journal(journal.path)
            observed.append(
                dict(
                    payload=payload,
                    pending=list(current.pending),
                    journal_sha256=e.sha256_file(journal.path),
                )
            )
            if payload["messages"][0]["content"] == "fixture-2":
                self.send_error(503, "scripted error")
            else:
                body = json.dumps(dict(text="0|E|0.10|[0]", fixture=True)).encode()
                self.send_response(200)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        def log_message(self, format: str, *args: Any) -> None:
            """The recorder owns explicit progress lines instead of HTTP noise."""

    server = ThreadingHTTPServer(("127.0.0.1", 0), Peer)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    runtime = e.HTTPRuntime(f"http://127.0.0.1:{server.server_port}/v1/chat/completions")
    startup_start = time.monotonic_ns()
    native, binding = host.old.prior.old.host.load_binding(data)
    startup_end = time.monotonic_ns()
    joins: list[Json] = []
    try:
        for i in range(3):
            e.progress("8213_fixture", i, 3 - i)
            request_id = f"fixture-{i}"
            encode_start = time.monotonic_ns()
            value = synthetic(i)
            encode_end = time.monotonic_ns()
            lookup_start = time.monotonic_ns()
            semantic = e.key(value)
            lookup_end = time.monotonic_ns()
            inference_start = time.monotonic_ns()
            terminal = e.invoke(journal, runtime, request_id, value)
            inference_end = time.monotonic_ns()
            if terminal["status"] == "completed":
                values = [float(i) / 10] * len(data["geometry"]["mean"])
                score_start = time.monotonic_ns()
                python = float(
                    host.old.score(
                        data["head"], data["geometry"], [values], native, "python_batch"
                    )[0]
                )
                rust = float(
                    host.old.score(
                        data["head"], data["geometry"], [values], native, "native_batch"
                    )[0]
                )
                score_end = time.monotonic_ns()
                commit_start = time.monotonic_ns()
                store_path = raw / (request_id + ".store.json")
                store = host.Store(store_path, e.key(data["head"]))
                records = store.commit(
                    [
                        dict(
                            request_id=request_id,
                            input_hash=semantic,
                            source_cluster_id=value["source_cluster_id"],
                            values=values,
                            probability=rust,
                            action="fixture_only",
                            head_hash=e.key(data["head"]),
                        )
                    ]
                )
                commit_end = time.monotonic_ns()
                readout_start = time.monotonic_ns()
                restored = host.Store(store_path, e.key(data["head"])).state["records"]
                readout_end = time.monotonic_ns()
                spans = dict(
                    encode=[encode_start, encode_end],
                    lookup=[lookup_start, lookup_end],
                    inference=[inference_start, inference_end],
                    score=[score_start, score_end],
                    commit_fsync=[commit_start, commit_end],
                    readout=[readout_start, readout_end],
                )
                joins.append(
                    dict(
                        request_id=request_id,
                        semantic_key=semantic,
                        python_probability=python,
                        rust_probability=rust,
                        values=values,
                        head_hash=e.key(data["head"]),
                        store=e.reference(store_path),
                        committed=records == restored,
                        spans=spans,
                        exclusive_duration_ns=sum(b - a for a, b in spans.values()),
                        nested_spans=dict(
                            issue_to_terminal=[
                                terminal["issued_monotonic_ns"],
                                terminal["observed_monotonic_ns"],
                            ]
                        ),
                    )
                )
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
    journal.issue("restart", synthetic(3))
    restart = e.Journal(journal.path)
    restart_pending = list(restart.pending)
    restart.finish("restart", "censored", dict(reason="explicit restart qualification"))
    restart.issue("non_generation", synthetic(4), operation="encode")
    restart.finish("non_generation", "completed", dict(fixture=True))
    cases = [
        dict(
            case="http_issue_before_generation",
            passed=all(f"fixture-{i}" in r["pending"] for i, r in enumerate(observed)),
        ),
        dict(case="restart_pending", passed=restart_pending == ["restart"]),
        dict(
            case="python_rust_durable_join",
            passed=all(
                abs(r["python_probability"] - r["rust_probability"]) < 1e-10 and r["committed"]
                for r in joins
            ),
        ),
    ]
    for case, mutation in [
        ("duplicate_id", None),
        ("partial_tail", b'{"partial":'),
        ("hash_drift", b"{}\n"),
    ]:
        probe = raw / (case + ".jsonl")
        probe.write_bytes(journal.path.read_bytes())
        if mutation is not None:
            probe.write_bytes(probe.read_bytes() + mutation)
        rejected = False
        try:
            candidate = e.Journal(probe)
            candidate.issue("fixture-0", synthetic(0))
        except (ValueError, KeyError):
            rejected = True
        cases.append(dict(case=case, passed=rejected, probe=e.reference(probe)))
    for case in ("seed", "chat_template_sha256"):
        changed = deepcopy(synthetic(0))
        if case == "seed":
            changed["payload"][case] += 1
        else:
            changed[case] = e.key("changed")
        cases.append(dict(case=case + "_invalidates", passed=e.key(changed) != e.key(synthetic(0))))
    terminal_rows = [
        r for r in restart.events if r["event"] == "terminal" and r["operation"] == "generation"
    ]
    result = dict(
        fixture_rows=cases,
        schedule_roundtrip_rows=schedule_roundtrip_rows,
        envelope_roundtrip_rows=terminal_rows,
        service_join_rows=joins,
        journal=e.reference(journal.path),
        peer_observations=observed,
        pending_count=len(restart.pending),
        non_generation_count=1,
        service_configuration=dict(
            native_loaded=True,
            binding=binding,
            head=data["head"],
            geometry=data["geometry"],
            startup_span=[startup_start, startup_end],
            startup_charged_once=True,
            signatures=[
                "generate(payload)",
                "score(head, geometry, values, native, arm)",
                "RustRadial8105.design(rows)",
                "Store.commit(rows)",
            ],
        ),
    )
    result["work_path"] = str((raw / "work.json").absolute())
    e.seal(raw / "work.json", result)
    return result


def validate_work(work: Json) -> bool:
    """Recompute joins from journal and stored responses, excluding nested clocks."""
    journal = e.Journal(Path(work["journal"]["path"]))
    config = work["service_configuration"]
    native, _ = host.old.prior.old.host.load_binding(dict(library=config["binding"]))
    terminals = [
        r for r in journal.events if r["event"] == "terminal" and r["operation"] == "generation"
    ]
    if terminals != work["envelope_roundtrip_rows"] or journal.pending:
        return False
    if len(terminals) != 4 or work["non_generation_count"] != 1 or work["pending_count"] != 0:
        return False
    for row in work["service_join_rows"]:
        event = next(r for r in terminals if r["request_id"] == row["request_id"])
        if event["semantic_key"] != row["semantic_key"] or event["status"] != "completed":
            return False
        spans = [
            row["spans"][k]
            for k in ("encode", "lookup", "inference", "score", "commit_fsync", "readout")
        ]
        if any(a > b for a, b in spans) or any(b > c for (_, b), (c, _) in zip(spans, spans[1:])):
            return False
        if row["exclusive_duration_ns"] != sum(b - a for a, b in spans):
            return False
        for arm, field in [
            ("python_batch", "python_probability"),
            ("native_batch", "rust_probability"),
        ]:
            expected = float(
                host.old.score(config["head"], config["geometry"], [row["values"]], native, arm)[0]
            )
            if abs(expected - row[field]) > 1e-10:
                return False
        if e.sha256_file(Path(row["store"]["path"])) != row["store"]["sha256"]:
            return False
        store = host.Store(Path(row["store"]["path"]), row["head_hash"])
        record = store.state["records"][0]
        if (
            record["request_id"] != row["request_id"]
            or record["input_hash"] != event["semantic_key"]
        ):
            return False
    return len(work["service_join_rows"]) == 2 and all(r["passed"] for r in work["fixture_rows"])
