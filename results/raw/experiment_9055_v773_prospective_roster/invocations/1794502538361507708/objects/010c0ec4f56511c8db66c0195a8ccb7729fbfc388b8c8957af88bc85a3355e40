"""REQ-REPORT-9042-V772: durable first requests and purpose-limited truth access."""

import json
import os
from pathlib import Path
import signal
import time
from typing import Any

from carnot.memory import online_spline_8988 as online
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash
from carnot.reporting.invocation_provenance_9000 import reference

Json = dict[str, Any]
TRANSPORT = {"response_sha256", "request_id", "token_count", "duration_s", "status"}
POLICY = dict(
    test="final_evaluation only after both byte-bound Exp9048/9049 prediction seals",
    calibration="calibration only; private admission may return readiness, never TEST values",
    online="Exp8988 release at queries8,12,...,32; labels29..32 remain pending",
    early_artifacts="identity and scalar transport fields only; no TEST responses or outcomes",
)
SEAL_FIELDS = [
    "experiment_id",
    "manifest_sha256",
    "label_sha256",
    "prediction_reference",
    "required_checks_passed",
]


class Checkpoint:
    """Persist intent before transport so a killed process cannot request a second answer."""

    def __init__(self, manifest: Json, pin: str, raw: Path):
        ids = [g["unit_id"] for g in manifest["graph_manifest"]]
        if canonical_hash(manifest) != pin or len(ids) != len(set(ids)):
            raise ValueError("manifest_binding")
        for ref in manifest.get("state_references", []):
            if reference(Path(ref["path"]))["sha256"] != ref["sha256"]:
                raise ValueError("frozen_state_bytes")
        self.pin, self.raw, self.manifest = pin, raw, manifest
        self.path = raw / "checkpoint.json"
        self.access: list[Json] = []
        self.state: Json = dict(
            manifest_sha256=pin, status=dict.fromkeys(ids, "pending"), starts=[], transport={}
        )
        if self.path.is_file():
            self.state = json.loads(self.path.read_bytes())
            if self.state["manifest_sha256"] != pin or set(self.state["status"]) != set(ids):
                raise ValueError("restart_binding")
            for uid, status in self.state["status"].items():
                if status == "started":
                    self.state["status"][uid] = "censored"
        atomic_json(self.path, self.state)

    def start(self, uid: str) -> None:
        """A started or lost first response remains spent across every restart."""
        if self.state["status"].get(uid) != "pending":
            raise ValueError("duplicate_or_unknown_first_response")
        self.state["status"][uid] = "started"
        self.state["starts"].append(uid)
        atomic_json(self.path, self.state)

    def finish(self, uid: str, transport: Json) -> None:
        """A strict scalar allowlist prevents truth leaking through nested diagnostic logs."""
        if set(transport) - TRANSPORT or any(
            isinstance(v, (dict, list)) for v in transport.values()
        ):
            raise ValueError("transport_schema")
        if self.state["status"].get(uid) != "started":
            raise ValueError("response_without_unique_start")
        self.state["status"][uid] = "completed"
        self.state["transport"][uid] = transport
        atomic_json(self.path, self.state)

    def labels(
        self,
        path: Path,
        expected: str,
        role: str,
        purpose: str,
        *,
        seals: list[Json] | None = None,
        query: int = 0,
        online_seal: Path | None = None,
    ) -> list[int]:
        """Check authority before opening truth and log every read or denied purpose."""
        authorized = role == "calibration" and purpose == "calibration"
        if not self.manifest.get("fixture_only") and path.name != f"{role}_truth.json":
            authorized = False
            role = "unbound_sidecar"
        if role == "test" and purpose == "final_evaluation" and seals and len(seals) == 2:
            seen = set()
            for ref in seals:
                if reference(Path(ref["path"]))["sha256"] != ref["sha256"]:
                    raise ValueError("seal_bytes")
                seal = json.loads(Path(ref["path"]).read_bytes())
                pred = seal["prediction_reference"]
                if (
                    seal["manifest_sha256"] != self.pin
                    or seal["label_sha256"] != expected
                    or seal["required_checks_passed"] is not True
                    or reference(Path(pred["path"]))["sha256"] != pred["sha256"]
                ):
                    raise ValueError("prediction_seal_binding")
                seen.add(seal["experiment_id"])
            authorized = seen == {9048, 9049}
        if role == "online" and purpose == "scheduled_feedback":
            authorized = query in range(8, 33, 4) and online_seal is not None
        event = dict(
            role=role,
            purpose=purpose,
            authorized=authorized,
            path=str(path),
            monotonic_ns=time.monotonic_ns(),
        )
        self.access.append(event)
        with (self.raw / "label_access.jsonl").open("a") as stream:
            stream.write(json.dumps(event, sort_keys=True) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        if not authorized:
            raise PermissionError("label_access_denied")
        if reference(path)["sha256"] != expected:
            raise ValueError("label_bytes")
        if role == "online":
            return online.release(query, Path(str(online_seal)), path)
        values = json.loads(path.read_bytes())["labels"]
        if any(type(y) is not int or y not in (0, 1) for y in values):
            raise ValueError("label_schema")
        return list(values)


def worker(request: Path, raw: Path, interrupt: bool = False) -> None:
    """Private transport mechanics use synthetic hashes and never invoke a model."""
    value = json.loads(request.read_bytes())
    manifest = value["manifest"]
    if manifest.get("fixture_only") is not True:
        raise ValueError("private_worker_only")
    state = Checkpoint(manifest, value["manifest_sha256"], raw)
    for uid, status in list(state.state["status"].items()):
        if status != "pending":
            continue
        state.start(uid)
        print(
            f"[exp9042] phase=private_transport_started completed={len(state.state['transport'])} pending={len(state.state['status']) - len(state.state['transport'])}",
            flush=True,
        )
        if uid == value["lost_unit"]:
            if interrupt:
                os.kill(os.getpid(), signal.SIGKILL)
            state.state["status"][uid] = "censored"
            atomic_json(state.path, state.state)
        else:
            state.finish(uid, dict(response_sha256=canonical_hash(uid), token_count=1))
