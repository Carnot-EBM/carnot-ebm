"""REQ-VERIFY-8213: durable issue evidence must precede the inference boundary.

The journal records observed event clocks. It never repairs missing historical
clocks or treats a synthetic HTTP response as a model measurement.
"""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime, UTC
import json
import os
from pathlib import Path
import time
from typing import Any, Protocol
from urllib.request import Request, urlopen

from carnot.reporting.current_work_receipt import atomic_json, canonical_hash, sha256_file
from carnot.verify.request_trace_8200 import progress, reference, seal
from carnot.verify import sentence_transport_8179 as transport

Json = dict[str, Any]
ROOT = Path(__file__).resolve().parents[3]
NAME = "experiment_8213_v709_prospective_request_recorder"
CLI = f"scripts/experiments/{NAME}.py"
SCHEMA = "carnot.prospective_request.v1"
SEED = 7098213
key = canonical_hash


def envelope(slot: Json, payload: Json, identity: Json) -> Json:
    """Bind wire prompt bytes and all decoding operands before a clock is read."""
    source, answer = (bytes.fromhex(slot[k]) for k in ("source_bytes", "answer_bytes"))
    return dict(
        request_schema_version=SCHEMA,
        source_cluster_id=slot["source_cluster_id"],
        source_bytes=source.hex(),
        answer_bytes=answer.hex(),
        source_sha256=key(source.hex()),
        answer_sha256=key(answer.hex()),
        condition=slot["condition"],
        condition_sha256=key(slot["condition"]),
        prompt_bytes=json.dumps(payload["messages"], ensure_ascii=False, sort_keys=True)
        .encode()
        .hex(),
        prompt_encoding="UTF-8 canonical wire messages; template bound separately",
        model=payload["model"],
        model_gguf_sha256=identity["model_sha256"],
        chat_template_sha256=identity["chat_template_sha256"],
        runtime_sha256=identity["runtime_sha256"],
        runtime_identity=deepcopy(identity),
        payload=deepcopy(payload),
        decoding_parameters={k: v for k, v in payload.items() if k != "messages"},
    )


def validate_envelope(value: Json) -> None:
    """A changed operand requires a new envelope instead of trusting old hashes."""
    expected = envelope(value, value["payload"], value["runtime_identity"])
    if value != expected or value["request_schema_version"] != SCHEMA:
        raise ValueError("envelope_hash")


class Journal:
    """A single invocation writer appends and fsyncs each hash-chained event.

    An incomplete tail remains untouched and prevents new acquisition. A restart
    keeps issued requests pending until an explicit completion or censor event.
    """

    def __init__(self, path: Path):
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        self.events: list[Json] = []
        self.pending: dict[str, Json] = {}
        self.issued: set[str] = set()
        raw = path.read_bytes() if path.exists() else b""
        if raw and not raw.endswith(b"\n"):
            raise ValueError("partial_tail")
        for line in raw.splitlines():
            row = json.loads(line)
            digest = row.pop("event_sha256")
            if key(row) != digest:
                raise ValueError("event_hash")
            row["event_sha256"] = digest
            self._accept(row)

    def _accept(self, row: Json) -> None:
        """Replay sequence and state transitions independently from the row hash."""
        previous = self.events[-1]["event_sha256"] if self.events else key(SCHEMA)
        if row["sequence"] != len(self.events) + 1 or row["previous_sha256"] != previous:
            raise ValueError("event_order")
        request_id = row["request_id"]
        if row["event"] == "issue":
            validate_envelope(row["envelope"])
            if (
                request_id in self.issued
                or row["semantic_key"] != key(row["envelope"])
                or row["issue_sequence"] != len(self.issued) + 1
                or row["status"] != "pending"
            ):
                raise ValueError("duplicate_or_semantic")
            self.issued.add(request_id)
            self.pending[request_id] = row
        else:
            if (
                request_id not in self.pending
                or row["semantic_key"] != self.pending[request_id]["semantic_key"]
            ):
                raise ValueError("terminal_without_issue")
            issue = self.pending[request_id]
            if (
                row["event"] != "terminal"
                or row["status"] not in {"completed", "error", "censored"}
                or row["result_sha256"] != key(row["result"])
                or row["observed_monotonic_ns"] < issue["issued_monotonic_ns"]
                or any(
                    row[k] != issue[k]
                    for k in (
                        "envelope",
                        "issue_sequence",
                        "issued_at",
                        "issued_monotonic_ns",
                        "operation",
                        "parent_id",
                        "retry_id",
                    )
                )
            ):
                raise ValueError("terminal_custody")
            del self.pending[request_id]
        self.events.append(row)

    def _append(self, row: Json) -> Json:
        """Fsync before returning ensures the caller cannot race ahead of issue."""
        row.pop("event_sha256", None)
        row.update(
            sequence=len(self.events) + 1,
            previous_sha256=self.events[-1]["event_sha256"] if self.events else key(SCHEMA),
            observed_at=datetime.now(UTC).isoformat(),
            observed_monotonic_ns=time.monotonic_ns(),
        )
        row["event_sha256"] = key(row)
        with self.path.open("ab") as stream:
            stream.write((json.dumps(row, sort_keys=True) + "\n").encode())
            stream.flush()
            os.fsync(stream.fileno())
        directory = os.open(self.path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
        self._accept(row)
        return row

    def issue(
        self,
        request_id: str,
        value: Json,
        *,
        parent_id: str | None = None,
        retry_id: str | None = None,
        operation: str = "generation",
    ) -> Json:
        """Record actual issue clocks and event identity before calling generate."""
        validate_envelope(value)
        if request_id in self.issued:
            raise ValueError("duplicate_id")
        return self._append(
            dict(
                event="issue",
                operation=operation,
                request_id=request_id,
                parent_id=parent_id,
                retry_id=retry_id,
                issue_sequence=len(self.issued) + 1,
                issued_at=datetime.now(UTC).isoformat(),
                issued_monotonic_ns=time.monotonic_ns(),
                status="pending",
                envelope=value,
                semantic_key=key(value),
            )
        )

    def finish(self, request_id: str, status: str, result: Json) -> Json:
        """A terminal event retains its issue envelope and exact response evidence."""
        if status not in {"completed", "error", "censored"}:
            raise ValueError("status")
        if request_id not in self.pending:
            raise ValueError("terminal_without_issue")
        issued = self.pending[request_id]
        return self._append(
            dict(
                issued,
                event="terminal",
                status=status,
                result=result,
                result_sha256=key(result),
                event_sha256=None,
            )
        )


class Generator(Protocol):
    """Use the same generate(payload) signature as the qualified service runtime."""

    def generate(self, payload: Json) -> Json: ...


class HTTPRuntime:
    """One bounded transport attempt supports both local fixtures and later calls."""

    def __init__(self, url: str):
        self.url = url

    def generate(self, payload: Json) -> Json:
        """Return exact response bytes after the durable issue event already exists."""
        request = Request(
            self.url,
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urlopen(request, timeout=5) as response:  # noqa: S310
            value: Json = json.loads(response.read())
        return value


def invoke(journal: Journal, runtime: Generator, request_id: str, value: Json) -> Json:
    """This adapter is the actual boundary: issue, one generate call, terminal."""
    journal.issue(request_id, value)
    progress("8213_generation_before_" + request_id, 0, 1)
    try:
        result = runtime.generate(value["payload"])
    except Exception as error:
        row = journal.finish(request_id, "error", dict(error=f"{type(error).__name__}:{error}"))
    else:
        row = journal.finish(request_id, "completed", result)
    progress("8213_generation_after_" + request_id, 1, 0)
    return row


def schedule(roster: list[Json], identity: Json) -> Json:
    """Fixed identity order and original first sentence forbid outcome selection."""
    fit = sorted([r for r in roster if r["role"] == "fit"], key=lambda r: r["source_cluster_id"])
    selected = fit[:24]
    if len(selected) != 24 or len({r["source_cluster_id"] for r in selected}) != 24:
        raise ValueError("source_count")
    rows = []
    for i, slot in enumerate(selected):
        body = json.loads(slot["requests"][0]["payload"])
        index = body["answer_sentence_indices"][0]
        body.update(answer_sentence_indices=[index], answer_offsets=[body["answer_offsets"][0]])
        payload = dict(
            model="unsloth/Qwen3.8-27B-GGUF",
            messages=[dict(role="user", content=json.dumps(body, ensure_ascii=False))],
            grammar=transport.grammar([index], len(body["source_segments"])),
            max_tokens=128,
            seed=SEED,
            temperature=0,
            stream=False,
            chat_template_kwargs=dict(enable_thinking=False),
        )
        rows.append(
            dict(
                source_cluster_id=slot["source_cluster_id"],
                request_id=f"prospective-{i + 1:02}",
                source_id=slot["source_id"],
                response_id=slot["response_id"],
                original_slot=slot["slot"],
                sentence_index=index,
                transport_attempts=1,
                envelope=envelope(slot, payload, identity),
                status="scheduled",
                metric="prospective_request_scheduled",
                numerator=1,
                denominator=1,
            )
        )
    return dict(
        request_schema_version=SCHEMA,
        selection="ascending source_cluster_id; first original sentence",
        workload_scope="designed_research_requests",
        deployment_demand_observed=False,
        rows=rows,
    )


def qualify(data: Json, raw: Path) -> Json:
    """Use real HTTP and native joins without loading a model or reusing outcomes."""
    from carnot.verify.recorder_fixtures_8213 import qualify as run

    return run(data, raw)


def replay(path: Path) -> bool:
    """Reopen primitive bytes and recompute recorder claims in a fresh process."""
    from carnot.reporting import recorder_execution_8213 as runner

    try:
        value = json.loads(path.read_text())
        if value["reproducibility_checksum"] != runner.checksum(value):
            return False
        for ref in (
            value["raw_shard_hashes"]
            + value["code_config_hashes"]
            + value["source_artifact_hashes"]
        ):
            if sha256_file(Path(ref.get("frozen_path", ref["path"]))) != ref["sha256"]:
                return False
        data = json.loads(Path(value["input_path"]).read_text())
        work = json.loads(Path(value["work_path"]).read_text()) if value.get("work_path") else {}
        if data["ready"]:
            frozen = json.loads(Path(value["schedule_path"]).read_text())
            if frozen != schedule(data["roster"], data["identity"]):
                return False
            from carnot.verify.recorder_fixtures_8213 import validate_work

            if work and not validate_work(work):
                return False
        expected = runner.build(
            data, work, Path(value["raw_path"]), value["validation_receipts"], value["duration_s"]
        )
        return all(
            value[k] == expected[k]
            for k in [
                "rows",
                "completed_count",
                "failed_count",
                "censored_count",
                "excluded_count",
                "independent_count",
                "request_recorder_ready_score",
                "verdict_class",
                "fixture_rows",
                "envelope_roundtrip_rows",
                "deployment_demand_observed",
            ]
        )
    except (OSError, ValueError, KeyError, TypeError):
        return False
