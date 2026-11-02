"""Public structural learner for the isolated V644 scheduling fixture.

This module knows the public scheduling language and Boolean query protocol.
It has no rule generator or evaluator implementation. Keeping those concerns
out of this import closure makes the process receipt useful audit evidence.

Spec refs: REQ-CL-7330, SCENARIO-CL-7330-BOUNDARY, and
SCENARIO-CL-7330-COMPOUND.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import itertools
import json
import os
from pathlib import Path
import socket
import sys
import tempfile
import time
from typing import Any, Callable, Mapping, Sequence


JsonDict = dict[str, Any]
QUERY_BUDGET = 24
STATE_CAP_BYTES = 69_632
STATE_SCHEMA = "carnot.v644.public-structural-memory.v1"


class PublicLearningError(RuntimeError):
    """Reject malformed public data or unsupported state without guessing."""


def canonical_bytes(value: Any) -> bytes:
    """Use one stable encoding for public state, plans, and evidence seals."""

    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    """Mark SHA-256 values so they cannot be mistaken for plain identifiers."""

    return "sha256:" + hashlib.sha256(value).hexdigest()


def sha256_json(value: Any) -> str:
    """Hash a JSON value after canonical serialization."""

    return sha256_bytes(canonical_bytes(value))


def sha256_file(path: Path) -> str:
    """Authenticate exact file bytes instead of trusting a declared checksum."""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def make_plan(request_id: str, assignments: Mapping[str, int]) -> JsonDict:
    """Keep request identity and sorted assignments in exact proposal bytes."""

    return {
        "request_id": str(request_id),
        "assignments": {name: int(value) for name, value in sorted(assignments.items())},
    }


def validate_public_request(request: Mapping[str, Any]) -> None:
    """Reject public requests whose shape could make proposal checks ambiguous."""

    required = {
        "request_id",
        "version_token",
        "activities",
        "allowed_starts",
        "durations",
        "weights",
        "horizon",
    }
    if not required <= set(request):
        raise PublicLearningError("request_fields")
    activities = request["activities"]
    if not isinstance(activities, list) or not 1 <= len(activities) <= 6:
        raise PublicLearningError("activity_count")
    if len(set(activities)) != len(activities) or any(
        not isinstance(name, str) for name in activities
    ):
        raise PublicLearningError("activity_identity")
    names = set(activities)
    for field in ("allowed_starts", "durations", "weights"):
        value = request[field]
        if not isinstance(value, Mapping) or set(value) != names:
            raise PublicLearningError("activity_set")
    horizon = request["horizon"]
    if not isinstance(horizon, int) or horizon < 1:
        raise PublicLearningError("horizon")
    for name in activities:
        starts = request["allowed_starts"][name]
        duration = request["durations"][name]
        weight = request["weights"][name]
        if (
            not isinstance(starts, list)
            or not starts
            or len(starts) > 4
            or any(not isinstance(value, int) or value < 0 for value in starts)
            or len(set(starts)) != len(starts)
        ):
            raise PublicLearningError("allowed_starts")
        if not isinstance(duration, int) or duration < 1:
            raise PublicLearningError("duration")
        if not isinstance(weight, int) or weight < 1:
            raise PublicLearningError("weight")
        if any(start + duration > horizon for start in starts):
            raise PublicLearningError("window")


def _interval_gap(
    request: Mapping[str, Any], assignments: Mapping[str, int], left: str, right: str
) -> int:
    """Return empty slots between events, with negative values for overlap."""

    left_start = int(assignments[left])
    right_start = int(assignments[right])
    left_end = left_start + int(request["durations"][left])
    right_end = right_start + int(request["durations"][right])
    if left_end <= right_start:
        return right_start - left_end
    if right_end <= left_start:
        return left_start - right_end
    return -min(left_end, right_end) + max(left_start, right_start)


def _atom_allows_plan(
    atom: Mapping[str, Any], request: Mapping[str, Any], plan: Mapping[str, Any]
) -> bool:
    """Apply a learned public atom without consulting evaluator-only rules."""

    assignments = plan["assignments"]
    left, right = atom["pair"]
    if left not in assignments or right not in assignments:
        return True
    return _interval_gap(request, assignments, left, right) >= int(atom["minimum_gap"])


def make_pair_atom(
    version_token: str,
    left: str,
    right: str,
    minimum_gap: int,
    receipt: Mapping[str, Any],
) -> JsonDict:
    """Build a hash-bound pair atom from one charged Boolean witness."""

    body: JsonDict = {
        "kind": "pair_gap",
        "version_token": str(version_token),
        "pair": sorted((str(left), str(right))),
        "minimum_gap": int(minimum_gap),
        "query_receipts": [deepcopy(dict(receipt))],
    }
    body["atom_id"] = sha256_json(body)
    return body


class PublicConstraintLearner:
    """Learn conservative pair gaps from public data and Boolean witnesses."""

    def __init__(self, version_token: str, *, memory_cap_bytes: int = STATE_CAP_BYTES) -> None:
        self.memory_cap_bytes = int(memory_cap_bytes)
        self._state: JsonDict = {
            "schema": STATE_SCHEMA,
            "active_version_token": str(version_token),
            "atoms": [],
            "uncertain_compounds": [],
        }

    def state_bytes(self) -> bytes:
        """Serialize all durable learner state under one exact byte budget."""

        return canonical_bytes(self._state)

    def state_hash(self) -> str:
        """Bind a proposal row to the exact public learner state."""

        return sha256_bytes(self.state_bytes())

    @classmethod
    def from_state_bytes(cls, data: bytes) -> PublicConstraintLearner:
        """Restore canonical public state and reject changed or unsupported bytes."""

        try:
            state = json.loads(data)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise PublicLearningError("state_json") from error
        if not isinstance(state, dict) or state.get("schema") != STATE_SCHEMA:
            raise PublicLearningError("state_schema")
        learner = cls(str(state.get("active_version_token", "")))
        learner._state = state
        if learner.state_bytes() != data:
            raise PublicLearningError("state_not_canonical")
        if len(data) > learner.memory_cap_bytes:
            raise PublicLearningError("persistent_state_cap")
        return learner

    def activate_version(self, version_token: str) -> None:
        """Select only atoms authorized by the currently observed opaque token."""

        self._state["active_version_token"] = str(version_token)

    def active_atoms(self) -> list[JsonDict]:
        """Return copied atoms for the current token so callers cannot mutate state."""

        token = self._state["active_version_token"]
        return deepcopy([row for row in self._state["atoms"] if row["version_token"] == token])

    def admit_atom(self, atom: Mapping[str, Any], *, current_query_index: int) -> JsonDict:
        """Admit one hash-valid, timely atom for the active opaque version."""

        candidate = deepcopy(dict(atom))
        if candidate.get("version_token") != self._state["active_version_token"]:
            raise PublicLearningError("version_mismatch")
        receipts = candidate.get("query_receipts")
        if not isinstance(receipts, list) or len(receipts) != 1:
            raise PublicLearningError("query_receipt")
        if receipts[0].get("accepted") is not False:
            raise PublicLearningError("non_rejection_witness")
        if int(receipts[0].get("sequence", current_query_index + 1)) > current_query_index:
            raise PublicLearningError("early_evidence")
        unhashed = {key: value for key, value in candidate.items() if key != "atom_id"}
        if candidate.get("atom_id") != sha256_json(unhashed):
            raise PublicLearningError("atom_hash")
        existing = next(
            (row for row in self._state["atoms"] if row["atom_id"] == candidate["atom_id"]), None
        )
        if existing is not None:
            return deepcopy(existing)
        previous = deepcopy(self._state)
        self._state["atoms"].append(candidate)
        self._state["atoms"].sort(key=lambda row: row["atom_id"])
        if len(self.state_bytes()) > self.memory_cap_bytes:
            self._state = previous
            raise PublicLearningError("persistent_state_cap")
        return deepcopy(candidate)

    def propose(self, request: Mapping[str, Any]) -> JsonDict:
        """Enumerate the bounded public domain using only learned public atoms."""

        validate_public_request(request)
        activities = list(request["activities"])
        atoms = self.active_atoms()
        for values in itertools.product(*(request["allowed_starts"][name] for name in activities)):
            candidate = make_plan(
                str(request["request_id"]), dict(zip(activities, values, strict=True))
            )
            if all(_atom_allows_plan(atom, request, candidate) for atom in atoms):
                return candidate
        raise PublicLearningError("no_public_candidate")

    def localize_rejection(
        self,
        request: Mapping[str, Any],
        plan: Mapping[str, Any],
        query: Callable[[Mapping[str, Any], str], bool],
    ) -> list[JsonDict]:
        """Admit only pair gaps isolated by rejected two-activity projections."""

        assignments = plan["assignments"]
        uncertainty: JsonDict = {
            "version_token": self._state["active_version_token"],
            "plan_hash": sha256_json(plan),
            "resolved_atom_ids": [],
        }
        admitted: list[JsonDict] = []
        for left, right in itertools.combinations(sorted(assignments), 2):
            pair_plan = make_plan(
                str(request["request_id"]),
                {left: int(assignments[left]), right: int(assignments[right])},
            )
            accepted = query(pair_plan, "localization_pair")
            if not accepted:
                receipt = {
                    "query_id": sha256_json(pair_plan),
                    "accepted": False,
                    "sequence": len(admitted) + 1,
                }
                atom = make_pair_atom(
                    str(self._state["active_version_token"]),
                    left,
                    right,
                    max(0, _interval_gap(request, assignments, left, right) + 1),
                    receipt,
                )
                admitted.append(self.admit_atom(atom, current_query_index=len(admitted) + 1))
        uncertainty["resolved_atom_ids"] = [row["atom_id"] for row in admitted]
        previous = deepcopy(self._state)
        self._state["uncertain_compounds"] = [
            *self._state["uncertain_compounds"][-15:],
            uncertainty,
        ]
        if len(self.state_bytes()) > self.memory_cap_bytes:
            self._state = previous
            raise PublicLearningError("persistent_state_cap")
        return admitted


class _SocketOracle:
    """Send exact plans to the evaluator and accept only Boolean responses."""

    def __init__(self, endpoint: Path, request_id: str) -> None:
        self.request_id = request_id
        self.call_count = 0
        self.total_call_count = 0
        self.response_keys: set[str] = set()
        self._socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self._socket.connect(str(endpoint))
        self._reader = self._socket.makefile("r", encoding="utf-8")
        self._writer = self._socket.makefile("w", encoding="utf-8")

    def set_request(self, request_id: str) -> None:
        self.request_id = request_id
        self.call_count = 0

    def query(self, plan: Mapping[str, Any], reason: str) -> bool:
        if self.call_count >= QUERY_BUDGET:
            raise PublicLearningError("query_budget")
        self.call_count += 1
        self.total_call_count += 1
        query_id = f"learner-{os.getpid()}-{self.total_call_count:05d}"
        payload = {
            "op": "check",
            "query_id": query_id,
            "request_id": self.request_id,
            "reason": reason,
            "plan": deepcopy(dict(plan)),
        }
        self._writer.write(canonical_bytes(payload).decode("utf-8") + "\n")
        self._writer.flush()
        response = json.loads(self._reader.readline())
        self.response_keys.update(response)
        if set(response) != {"query_id", "accepted"} or response.get("query_id") != query_id:
            raise PublicLearningError("response_shape")
        if not isinstance(response["accepted"], bool):
            raise PublicLearningError("response_boolean")
        return response["accepted"]

    def close(self) -> None:
        self._writer.close()
        self._reader.close()
        self._socket.close()


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(canonical_bytes(value) + b"\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():  # pragma: no cover - only an interrupted replace leaves this path.
            temporary.unlink()


def run_worker(public_manifest_path: Path, endpoint: Path, output_path: Path) -> JsonDict:
    """Run development requests while auditing the learner's visible boundary."""

    opened: list[JsonDict] = []

    def audit(  # pragma: no cover - CPython disables tracing inside audit hooks.
        event: str, args: tuple[Any, ...]
    ) -> None:
        if event == "open" and args and isinstance(args[0], (str, bytes, os.PathLike)):
            opened.append({"event": event, "path": os.fsdecode(args[0])})

    sys.addaudithook(audit)
    manifest = json.loads(public_manifest_path.read_text(encoding="utf-8"))
    oracle = _SocketOracle(endpoint, "")
    rows: list[JsonDict] = []
    for stream_index, stream in enumerate(manifest["development_streams"]):
        learner: PublicConstraintLearner | None = None
        for request_index, request in enumerate(stream["requests"]):
            validate_public_request(request)
            if learner is None:
                learner = PublicConstraintLearner(str(request["version_token"]))
            learner.activate_version(str(request["version_token"]))
            oracle.set_request(str(request["request_id"]))
            before_calls = oracle.call_count
            plan = learner.propose(request)
            accepted = oracle.query(plan, "main")
            learned = [] if accepted else learner.localize_rejection(request, plan, oracle.query)
            if not accepted and learned:
                plan = learner.propose(request)
                accepted = oracle.query(plan, "main_retry")
            final_accepted = accepted and oracle.query(plan, "final")
            calls = oracle.call_count - before_calls
            rows.append(
                {
                    "stream_id": stream["stream_id"],
                    "cohort": stream["cohort"],
                    "request_id": request["request_id"],
                    "request_index": request_index,
                    "warmup": request_index < 4,
                    "arm": "public_structural_learner",
                    "query_attempts": calls,
                    "executor_calls": calls,
                    "query_cost": calls,
                    "abstentions": int(not final_accepted),
                    "failures": int(not final_accepted),
                    "censored": False,
                    "accepted": bool(final_accepted),
                    "returned": bool(final_accepted),
                    "new_atom_count": len(learned),
                    "state_bytes": len(learner.state_bytes()),
                    "state_hash": learner.state_hash(),
                }
            )
        print(
            f"[exp7330-learner] stream={stream_index + 1}/4 rows={len(rows)}",
            flush=True,
        )

    challenge = manifest["compound_conflict_challenge"]
    challenge_request = challenge["request"]
    challenge_learner = PublicConstraintLearner(str(challenge_request["version_token"]))
    oracle.set_request(str(challenge_request["request_id"]))
    full_plan = challenge["candidate_plan"]
    full_rejected = not oracle.query(full_plan, "compound_full")
    atoms = challenge_learner.localize_rejection(challenge_request, full_plan, oracle.query)
    pair_receipts = (
        len(challenge_request["activities"]) * (len(challenge_request["activities"]) - 1) // 2
    )
    oracle.close()

    import_closure = sorted(
        {
            str(Path(module.__file__).resolve())
            for module in sys.modules.values()
            if getattr(module, "__file__", None)
        }
    )
    opened_paths = sorted({str(row["path"]) for row in opened})
    forbidden_markers = (
        "experiment_7330_v644_private_executor",
        "experiment_7323_v643_addition_prototype",
        "experiment_7325_v643_addition_audit",
        "evaluator_private_manifest",
    )
    forbidden = sorted(
        value
        for value in {*opened_paths, *import_closure}
        if any(marker in value for marker in forbidden_markers)
    )
    evidence: JsonDict = {
        "schema": "carnot.experiment_7330.public-learner-evidence.v1",
        "learner_pid": os.getpid(),
        "allowed_input_paths": [str(public_manifest_path.resolve()), str(endpoint.resolve())],
        "allowed_output_paths": [str(output_path.resolve())],
        "public_manifest_sha256": sha256_file(public_manifest_path),
        "import_closure": import_closure,
        "open_file_receipts": opened_paths,
        "forbidden_accesses": forbidden,
        "response_keys": sorted(oracle.response_keys),
        "rows": rows,
        "compound_conflict_challenge": {
            "full_rejected": full_rejected,
            "all_pair_projections_accepted": full_rejected and not atoms,
            "pair_projection_count": pair_receipts,
            "learned_atom_count": len(atoms),
            "outside_acquisition_language": True,
        },
    }
    _atomic_json(output_path, evidence)
    return evidence


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:  # pragma: no cover
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--public-manifest", type=Path, required=True)
    parser.add_argument("--endpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - subprocess entrypoint.
    args = _parse_args(argv)
    started = time.monotonic()
    print("[exp7330-learner] phase=worker event=start", flush=True)
    evidence = run_worker(args.public_manifest, args.endpoint, args.output)
    print(
        f"[exp7330-learner] phase=worker event=end rows={len(evidence['rows'])} "
        f"elapsed_s={time.monotonic() - started:.3f}",
        flush=True,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
