"""ARC evaluation provenance shared by producers and headline consumers.

Spec: REQ-ARC-7010, REQ-ARC-7030, REQ-ARC-WMTE-6790, REQ-ARC-WMTE-6710.

The older generator summary remains available for diagnostics. New evaluation
rows use the strict record below: absence stays absence and never becomes a
guess made from a filename, port, or historical artifact.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import os
import re
import subprocess
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import urlparse

ARC_EVAL_PROVENANCE_SCHEMA_VERSION = "carnot.arc_eval_provenance.v1"
ARC_EVAL_PROVENANCE_SCHEMA_VERSION_V2 = "carnot.arc_eval_provenance.v2"
LIVE_LLM_INFERENCE_SUBSTRATE = "local_gguf_cuda"
NO_LLM_INFERENCE_SUBSTRATE = "offline_arcade_live_agent_runtime_self_discovery_no_llm"
NOT_APPLICABLE = "not_applicable"
SOLVE_PROVENANCE_VALUES = frozenset(
    {"live_agent_self_discovery", "development_proxy", "outer_loop_re"}
)
ARC_EVAL_PROVENANCE_REQUIRED_KEYS = (
    "schema_version",
    "inference_substrate",
    "gpu_uuid",
    "gpu_model",
    "cuda_device",
    "model_repository",
    "model_filename",
    "model_hash",
    "n_ctx",
    "server_binary",
    "server_binary_hash",
    "server_command_hash",
    "endpoint",
    "port",
    "lease_id",
    "lease_hash",
    "lease_issued_at",
    "lease_expires_at",
    "lease_checked_at",
    "request_count",
    "completion_count",
    "error_count",
    "policy_hash",
    "factory_hash",
    "git_commit",
    "solve_provenance",
    "provenance_hash",
)
ARC_MODEL_IDENTITY_KEYS = (
    "requested_model_path",
    "requested_model_filename",
    "requested_hf_id",
    "requested_revision",
    "observed_server_model_path",
    "resolved_model_path",
    "model_file_hash",
)
ARC_EVAL_PROVENANCE_V2_REQUIRED_KEYS = (
    *ARC_EVAL_PROVENANCE_REQUIRED_KEYS[:-1],
    *ARC_MODEL_IDENTITY_KEYS,
    "provenance_hash",
)
# These aliases deliberately share one object. A producer-only or consumer-only
# key list would let the contract drift while each side still passed its tests.
PRODUCER_REQUIRED_KEYS = ARC_EVAL_PROVENANCE_REQUIRED_KEYS
CONSUMER_REQUIRED_KEYS = ARC_EVAL_PROVENANCE_REQUIRED_KEYS

_HASH_RE = re.compile(r"sha256:[0-9a-f]{64}\Z")
_COMMIT_RE = re.compile(r"[0-9a-f]{40}\Z")
_GPU_UUID_RE = re.compile(r"GPU-[A-Za-z0-9-]{8,}\Z")
_MODEL_REPOSITORY_RE = re.compile(r"[^/\s]+/[^/\s]+\Z")
_CONTENT_HASH_RE = re.compile(r"[0-9a-f]{64}\Z")
_LIVE_NA_FIELDS = (
    "gpu_uuid",
    "gpu_model",
    "cuda_device",
    "model_repository",
    "model_filename",
    "model_hash",
    "n_ctx",
    "server_binary",
    "server_binary_hash",
    "server_command_hash",
    "endpoint",
    "port",
    "lease_id",
    "lease_hash",
    "lease_issued_at",
    "lease_expires_at",
    "lease_checked_at",
)


@dataclass(frozen=True, slots=True)
class ArcEvalProvenanceInput:
    """Typed observations from one evaluation row before its stable hash."""

    inference_substrate: str
    gpu_uuid: str
    gpu_model: str
    cuda_device: int | str
    model_repository: str
    model_filename: str
    model_hash: str
    n_ctx: int | str
    server_binary: str
    server_binary_hash: str
    server_command_hash: str
    endpoint: str
    port: int | str
    lease_id: str
    lease_hash: str
    lease_issued_at: str
    lease_expires_at: str
    lease_checked_at: str
    request_count: int
    completion_count: int
    error_count: int
    policy_hash: str
    factory_hash: str
    git_commit: str
    solve_provenance: str
    schema_version: str = ARC_EVAL_PROVENANCE_SCHEMA_VERSION
    requested_model_path: str | None = None
    requested_model_filename: str | None = None
    requested_hf_id: str | None = None
    requested_revision: str | None = None
    observed_server_model_path: str | None = None
    resolved_model_path: str | None = None
    model_file_hash: str | None = None


@dataclass(frozen=True, slots=True)
class ArcEvalProvenanceValidation:
    """Consumer decision; invalid provenance is never headline eligible."""

    valid: bool
    headline_eligible: bool
    errors: tuple[str, ...]


def canonical_arc_eval_provenance_bytes(value: Mapping[str, Any]) -> bytes:
    """Stable JSON bytes used for the record digest in every process."""

    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()


def compute_arc_eval_provenance_hash(record: Mapping[str, Any]) -> str:
    """Digest the complete record except its own digest field."""

    body = {key: value for key, value in record.items() if key != "provenance_hash"}
    return "sha256:" + hashlib.sha256(canonical_arc_eval_provenance_bytes(body)).hexdigest()


def _absolute_path(path: Any) -> Path | None:
    """Return an absolute path without resolving a receipt-bearing symlink."""

    if not isinstance(path, str) or not path.strip():
        return None
    value = Path(path)
    return value if value.is_absolute() else None


def huggingface_snapshot_revision(model_path: Any, hf_id: Any) -> str | None:
    """Read the revision only from an exact Hugging Face snapshot path.

    The path layout supplies independent hub and revision evidence. A filename
    cannot supply either fact, so this helper never falls back to a basename.
    """

    requested = _absolute_path(model_path)
    if requested is None or not isinstance(hf_id, str) or not _MODEL_REPOSITORY_RE.fullmatch(hf_id):
        return None
    revision_dir = requested.parent
    snapshots_dir = revision_dir.parent
    model_root = snapshots_dir.parent
    expected_root = "models--" + hf_id.replace("/", "--")
    if snapshots_dir.name != "snapshots" or model_root.name != expected_root:
        return None
    return revision_dir.name or None


def build_arc_model_identity_receipt(
    *,
    selected_model_spec: Mapping[str, Any],
    observed_server_model_path: Any,
) -> dict[str, str]:
    """Join one requested snapshot GGUF to the server's canonical blob.

    REQ-ARC-7030 requires filesystem and content evidence. Display aliases and
    file sizes do not identify model bytes, so neither can satisfy this bridge.
    """

    if not isinstance(selected_model_spec, Mapping):
        raise TypeError("selected_model_spec must be a mapping")
    requested = _absolute_path(selected_model_spec.get("model_path"))
    observed = _absolute_path(observed_server_model_path)
    filename = selected_model_spec.get("model_filename")
    hf_id = selected_model_spec.get("hf_id")
    revision = selected_model_spec.get("revision")
    expected_hash = selected_model_spec.get("model_file_hash")
    errors: list[str] = []

    if requested is None:
        errors.append("requested_model_path must be absolute")
    if (
        not isinstance(filename, str)
        or Path(filename).name != filename
        or not filename.lower().endswith(".gguf")
    ):
        errors.append("requested_model_filename must be one GGUF filename")
    if requested is not None and requested.name != filename:
        errors.append("requested_model_filename contradicts requested_model_path")
    if not isinstance(hf_id, str) or not _MODEL_REPOSITORY_RE.fullmatch(hf_id):
        errors.append("requested_hf_id must be one owner/repository")
    if not isinstance(revision, str) or not revision.strip() or Path(revision).name != revision:
        errors.append("requested_revision must be one snapshot revision")
    if not isinstance(expected_hash, str) or not _HASH_RE.fullmatch(expected_hash):
        errors.append("model_file_hash must be a labeled SHA-256")
    if observed is None:
        errors.append("observed_server_model_path must be absolute")
    if errors:
        raise ValueError("invalid ARC model identity: " + "; ".join(errors))

    assert requested is not None and observed is not None
    assert isinstance(filename, str) and isinstance(hf_id, str) and isinstance(revision, str)
    assert isinstance(expected_hash, str)
    path_revision = huggingface_snapshot_revision(str(requested), hf_id)
    if path_revision != revision:
        errors.append("requested hub ID or revision contradicts snapshot path")
    if requested.parent.name != revision:
        errors.append("requested path is outside the selected revision")
    if requested.is_symlink() is not True:
        errors.append("requested_model_path must be a snapshot symlink")
    if not requested.exists() or not requested.is_file():
        errors.append("requested_model_path is missing, broken, or not a file")
    if not observed.exists() or not observed.is_file():
        errors.append("observed_server_model_path is missing or not a file")
    if errors:
        raise ValueError("invalid ARC model identity: " + "; ".join(errors))

    try:
        resolved = requested.resolve(strict=True)
        observed_canonical = observed.resolve(strict=True)
    except OSError as exc:
        raise ValueError(f"invalid ARC model identity: path resolution failed: {exc}") from exc
    if observed != observed_canonical:
        errors.append("observed_server_model_path must be the canonical path, not an alias")

    model_root = requested.parent.parent.parent
    blobs_dir = model_root / "blobs"
    if resolved.parent != blobs_dir or observed_canonical.parent != blobs_dir:
        errors.append("observed blob is not reachable from the selected snapshot")
    requested_hash = _sha256_file(resolved)
    observed_hash = requested_hash if resolved == observed_canonical else _sha256_file(observed_canonical)
    digest = requested_hash.removeprefix("sha256:") if isinstance(requested_hash, str) else ""
    if not _CONTENT_HASH_RE.fullmatch(resolved.name) or resolved.name != digest:
        errors.append("resolved blob basename does not equal its content hash")
    if resolved != observed_canonical and requested_hash != observed_hash:
        errors.append("requested and observed canonical files have different content hashes")
    if requested_hash != expected_hash or observed_hash != expected_hash:
        errors.append("model_file_hash does not equal the requested and observed bytes")
    if errors:
        raise ValueError("invalid ARC model identity: " + "; ".join(errors))

    return {
        "requested_model_path": str(requested),
        "requested_model_filename": filename,
        "requested_hf_id": hf_id,
        "requested_revision": revision,
        "observed_server_model_path": str(observed),
        "resolved_model_path": str(resolved),
        "model_file_hash": expected_hash,
    }


def _integer(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _nonempty(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _timestamp(value: Any) -> datetime | None:
    if not isinstance(value, str):
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed if parsed.tzinfo is not None else None


def validate_arc_eval_provenance(record: Any) -> ArcEvalProvenanceValidation:
    """Validate one exact versioned record without aliases or inferred defaults."""

    if not isinstance(record, dict):
        return ArcEvalProvenanceValidation(False, False, ("record must be an object",))
    errors: list[str] = []
    schema_version = record.get("schema_version")
    if schema_version == ARC_EVAL_PROVENANCE_SCHEMA_VERSION:
        required_keys = ARC_EVAL_PROVENANCE_REQUIRED_KEYS
    elif schema_version == ARC_EVAL_PROVENANCE_SCHEMA_VERSION_V2:
        required_keys = ARC_EVAL_PROVENANCE_V2_REQUIRED_KEYS
    else:
        required_keys = ARC_EVAL_PROVENANCE_V2_REQUIRED_KEYS
        errors.append("schema_version is not a supported provenance contract")
    expected = set(required_keys)
    observed = set(record)
    for key in sorted(expected - observed):
        errors.append(f"missing required field: {key}")
    for key in sorted(observed - expected):
        errors.append(f"unknown or aliased field: {key}")
    for key in required_keys:
        if key in record and record[key] is None:
            errors.append(f"null required field: {key}")

    if record.get("solve_provenance") not in SOLVE_PROVENANCE_VALUES:
        errors.append("solve_provenance is not an allowed enum value")
    for key in ("policy_hash", "factory_hash"):
        value = record.get(key)
        if not isinstance(value, str) or not _HASH_RE.fullmatch(value):
            errors.append(f"{key} must be a labeled SHA-256")
    commit = record.get("git_commit")
    if not isinstance(commit, str) or not _COMMIT_RE.fullmatch(commit):
        errors.append("git_commit must be a full lowercase commit hash")

    counters: dict[str, int] = {}
    for key in ("request_count", "completion_count", "error_count"):
        value = record.get(key)
        if not _integer(value) or value < 0:
            errors.append(f"{key} must be a non-negative integer")
        else:
            counters[key] = value
    if len(counters) == 3 and counters["request_count"] != (
        counters["completion_count"] + counters["error_count"]
    ):
        errors.append("counter contradiction: requests must equal completions plus errors")

    substrate = record.get("inference_substrate")
    if substrate == NO_LLM_INFERENCE_SUBSTRATE:
        for key in _LIVE_NA_FIELDS:
            if record.get(key) != NOT_APPLICABLE:
                errors.append(f"no-LLM field {key} must be not_applicable")
        if schema_version == ARC_EVAL_PROVENANCE_SCHEMA_VERSION_V2:
            for key in ARC_MODEL_IDENTITY_KEYS:
                if record.get(key) != NOT_APPLICABLE:
                    errors.append(f"no-LLM identity field {key} must be not_applicable")
        if counters and any(counters.values()):
            errors.append("no-LLM counters must all be zero")
    elif substrate == LIVE_LLM_INFERENCE_SUBSTRATE:
        _validate_live_fields(record, errors)
        if schema_version == ARC_EVAL_PROVENANCE_SCHEMA_VERSION_V2:
            identity_spec = {
                "model_path": record.get("requested_model_path"),
                "model_filename": record.get("requested_model_filename"),
                "hf_id": record.get("requested_hf_id"),
                "revision": record.get("requested_revision"),
                "model_file_hash": record.get("model_file_hash"),
            }
            try:
                identity = build_arc_model_identity_receipt(
                    selected_model_spec=identity_spec,
                    observed_server_model_path=record.get("observed_server_model_path"),
                )
            except (TypeError, ValueError) as exc:
                errors.append(str(exc))
            else:
                for key, value in identity.items():
                    if record.get(key) != value:
                        errors.append(f"{key} contradicts the shared model identity receipt")
                if record.get("model_repository") != identity["requested_hf_id"]:
                    errors.append("model_repository contradicts requested_hf_id")
                if record.get("model_filename") != identity["requested_model_filename"]:
                    errors.append("model_filename contradicts requested_model_filename")
                if record.get("model_hash") != identity["model_file_hash"]:
                    errors.append("model_hash contradicts model_file_hash")
    else:
        errors.append("inference_substrate is not a supported explicit substrate")

    claimed_hash = record.get("provenance_hash")
    if not isinstance(claimed_hash, str) or not _HASH_RE.fullmatch(claimed_hash):
        errors.append("provenance_hash must be a labeled SHA-256")
    elif claimed_hash != compute_arc_eval_provenance_hash(record):
        errors.append("provenance_hash does not match the canonical record")
    unique_errors = tuple(dict.fromkeys(errors))
    return ArcEvalProvenanceValidation(not unique_errors, not unique_errors, unique_errors)


def _validate_live_fields(record: Mapping[str, Any], errors: list[str]) -> None:
    """Apply CUDA, server, and live authority invariants."""

    gpu_uuid = record.get("gpu_uuid")
    if not isinstance(gpu_uuid, str) or not _GPU_UUID_RE.fullmatch(gpu_uuid):
        errors.append("gpu_uuid must be a driver UUID")
    if not _nonempty(record.get("gpu_model")) or record.get("gpu_model") == NOT_APPLICABLE:
        errors.append("gpu_model must identify the CUDA device")
    cuda_device = record.get("cuda_device")
    if not _integer(cuda_device) or cuda_device < 0:
        errors.append("cuda_device must be a non-negative integer")
    repository = record.get("model_repository")
    if not isinstance(repository, str) or not _MODEL_REPOSITORY_RE.fullmatch(repository):
        errors.append("model_repository must be an explicit owner/repository")
    filename = record.get("model_filename")
    if (
        not isinstance(filename, str)
        or Path(filename).name != filename
        or not filename.lower().endswith(".gguf")
    ):
        errors.append("model_filename must be one GGUF filename")
    for key in ("model_hash", "server_binary_hash", "server_command_hash", "lease_hash"):
        value = record.get(key)
        if not isinstance(value, str) or not _HASH_RE.fullmatch(value):
            errors.append(f"{key} must be a labeled SHA-256")
    n_ctx = record.get("n_ctx")
    if not _integer(n_ctx) or n_ctx <= 0:
        errors.append("n_ctx must be a positive integer")
    binary = record.get("server_binary")
    if not isinstance(binary, str) or not binary or not Path(binary).is_absolute():
        errors.append("server_binary must be an absolute observed path")
    if not _nonempty(record.get("lease_id")) or record.get("lease_id") == NOT_APPLICABLE:
        errors.append("lease_id must identify the live authority")

    endpoint = record.get("endpoint")
    port = record.get("port")
    try:
        parsed = urlparse(endpoint if isinstance(endpoint, str) else "")
        endpoint_port = parsed.port
    except ValueError:
        endpoint_port = None
        parsed = urlparse("")
    if parsed.scheme not in {"http", "https"} or not parsed.hostname or endpoint_port is None:
        errors.append("endpoint must be an HTTP URL with an explicit port")
    if not _integer(port) or not 1 <= port <= 65535:
        errors.append("port must be an integer from 1 through 65535")
    elif endpoint_port != port:
        errors.append("endpoint port contradicts port; possible port reuse")

    issued = _timestamp(record.get("lease_issued_at"))
    expires = _timestamp(record.get("lease_expires_at"))
    checked = _timestamp(record.get("lease_checked_at"))
    if None in (issued, expires, checked):
        errors.append("lease timestamps must be timezone-aware ISO-8601")
    elif not issued <= checked < expires:
        errors.append("lease is stale or its time bounds contradict")


def build_arc_eval_provenance(source: ArcEvalProvenanceInput) -> dict[str, Any]:
    """Build, hash, and validate one record before a producer persists it."""

    if not isinstance(source, ArcEvalProvenanceInput):
        raise TypeError("source must be ArcEvalProvenanceInput")
    values = asdict(source)
    schema_version = values.pop("schema_version")
    identity = {key: values.pop(key) for key in ARC_MODEL_IDENTITY_KEYS}
    record = {"schema_version": schema_version, **values}
    if schema_version == ARC_EVAL_PROVENANCE_SCHEMA_VERSION_V2:
        record.update(identity)
    elif any(value is not None for value in identity.values()):
        raise ValueError("legacy provenance input cannot carry current model identity fields")
    record["provenance_hash"] = compute_arc_eval_provenance_hash(record)
    decision = validate_arc_eval_provenance(record)
    if not decision.valid:
        raise ValueError("invalid ARC evaluation provenance: " + "; ".join(decision.errors))
    return record


def validate_arc_evaluation_row(row: Any) -> ArcEvalProvenanceValidation:
    """Consumer gate requiring row and nested solve provenance to agree."""

    if not isinstance(row, dict):
        return ArcEvalProvenanceValidation(False, False, ("evaluation row must be an object",))
    decision = validate_arc_eval_provenance(row.get("arc_eval_provenance"))
    errors = list(decision.errors)
    row_solve = row.get("solve_provenance")
    if row_solve not in SOLVE_PROVENANCE_VALUES:
        errors.append("evaluation row solve_provenance is absent or invalid")
    provenance = row.get("arc_eval_provenance")
    if isinstance(provenance, dict) and provenance.get("solve_provenance") != row_solve:
        errors.append("evaluation row solve_provenance contradicts provenance record")
    unique_errors = tuple(dict.fromkeys(errors))
    return ArcEvalProvenanceValidation(not unique_errors, not unique_errors, unique_errors)


def _sha256_file(path: Any) -> str | None:
    try:
        digest = hashlib.sha256()
        with Path(path).open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return "sha256:" + digest.hexdigest()
    except (OSError, TypeError):
        return None


def _policy_path(policy: Any) -> str | None:
    try:
        return inspect.getsourcefile(type(policy))
    except (OSError, TypeError):
        return None


def evaluation_counters(policy: Any) -> dict[str, int]:
    """Monotone request, completed-response, and failed-request counters."""

    prop = getattr(policy, "proposer", None)
    requests = int(getattr(prop, "n_completion_calls", 0) or 0)
    completions = int(getattr(prop, "n_completion_ok", 0) or 0)
    return {
        "requests": requests,
        "completions": completions,
        "errors": max(0, requests - completions),
    }


def _counter_delta(before: Mapping[str, int], after: Mapping[str, int], key: str) -> int:
    return int(after.get(key, 0)) - int(before.get(key, 0))


def _explicit_lease(lease: Mapping[str, Any] | None) -> Mapping[str, Any]:
    if lease is not None:
        return lease
    raw = os.environ.get("CARNOT_ARC_EVAL_LEASE_JSON", "")
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        value = None
    return value if isinstance(value, dict) else {}


def _git_commit(root: Path) -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, check=False
    )
    return completed.stdout.strip()


def build_arc_eval_provenance_for_policy(
    policy: Any,
    *,
    counters_before: Mapping[str, int],
    counters_after: Mapping[str, int],
    envelope: Mapping[str, Any],
    solve_provenance: str,
    factory_path: Path,
    repo_root: Path,
    lease: Mapping[str, Any] | None = None,
    lease_checked_at: str | None = None,
    model_identity_receipt: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Translate explicit producer observations into the shared strict record."""

    requests = _counter_delta(counters_before, counters_after, "requests")
    completions = _counter_delta(counters_before, counters_after, "completions")
    errors = _counter_delta(counters_before, counters_after, "errors")
    common = {
        "request_count": requests,
        "completion_count": completions,
        "error_count": errors,
        "policy_hash": _sha256_file(_policy_path(policy)),
        "factory_hash": _sha256_file(factory_path),
        "git_commit": _git_commit(repo_root),
        "solve_provenance": solve_provenance,
    }
    if requests == 0:
        na = NOT_APPLICABLE
        return build_arc_eval_provenance(
            ArcEvalProvenanceInput(
                inference_substrate=NO_LLM_INFERENCE_SUBSTRATE,
                gpu_uuid=na,
                gpu_model=na,
                cuda_device=na,
                model_repository=na,
                model_filename=na,
                model_hash=na,
                n_ctx=na,
                server_binary=na,
                server_binary_hash=na,
                server_command_hash=na,
                endpoint=na,
                port=na,
                lease_id=na,
                lease_hash=na,
                lease_issued_at=na,
                lease_expires_at=na,
                lease_checked_at=na,
                **common,
            )
        )

    prop = getattr(policy, "proposer", None)
    gpu_rows = envelope.get("gpus_held")
    gpu_rows = gpu_rows if isinstance(gpu_rows, list) else []
    requested_device = envelope.get("generator_cuda_gpu_requested")
    selected = next(
        (row for row in gpu_rows if str(row.get("index")) == str(requested_device)),
        gpu_rows[0] if len(gpu_rows) == 1 else {},
    )
    command = getattr(prop, "last_launch_argv", None)
    command = list(command) if isinstance(command, (list, tuple)) else None
    model_path = getattr(prop, "model_path", None)
    model_filename = getattr(prop, "model_filename", None)
    identity: dict[str, Any] | None = None
    if model_identity_receipt is not None:
        candidate = dict(model_identity_receipt)
        identity = build_arc_model_identity_receipt(
            selected_model_spec={
                "model_path": candidate.get("requested_model_path"),
                "model_filename": candidate.get("requested_model_filename"),
                "hf_id": candidate.get("requested_hf_id"),
                "revision": candidate.get("requested_revision"),
                "model_file_hash": candidate.get("model_file_hash"),
            },
            observed_server_model_path=candidate.get("observed_server_model_path"),
        )
        for key, value in identity.items():
            if candidate.get(key) != value:
                raise ValueError(f"model_identity_receipt contradicts {key}")
    else:
        requested_path = getattr(prop, "requested_model_path", None)
        requested_filename = getattr(prop, "requested_model_filename", None)
        requested_revision = getattr(prop, "model_revision", None)
        observed_path = getattr(prop, "observed_server_model_path", None)
        if not observed_path:
            observed_reader = getattr(prop, "observed_model_path", None)
            try:
                observed_path = observed_reader() if callable(observed_reader) else observed_reader
            except Exception:  # noqa: BLE001 - missing observation must fail during validation
                observed_path = None
        if requested_path and requested_revision:
            requested_filename = requested_filename or Path(str(requested_path)).name
            identity = build_arc_model_identity_receipt(
                selected_model_spec={
                    "model_path": requested_path,
                    "model_filename": requested_filename,
                    "hf_id": getattr(prop, "model_repository", None),
                    "revision": requested_revision,
                    "model_file_hash": _sha256_file(requested_path),
                },
                observed_server_model_path=observed_path,
            )
    if identity is not None:
        model_path = identity["requested_model_path"]
        model_filename = identity["requested_model_filename"]
        model_repository = identity["requested_hf_id"]
        model_hash = identity["model_file_hash"]
        schema_version = ARC_EVAL_PROVENANCE_SCHEMA_VERSION_V2
    else:
        if model_path and Path(model_path).name != model_filename:
            model_filename = None
        model_repository = getattr(prop, "model_repository", None)
        model_hash = _sha256_file(model_path)
        schema_version = ARC_EVAL_PROVENANCE_SCHEMA_VERSION
    server_binary = getattr(prop, "generator_server_path", None)
    if not server_binary and command:
        server_binary = command[0]
    endpoint = getattr(prop, "_url", None)
    try:
        endpoint = endpoint() if callable(endpoint) else endpoint
    except Exception:  # noqa: BLE001 - unreadable state must reject, not acquire a guessed value
        endpoint = None
    authority = _explicit_lease(lease)
    source = ArcEvalProvenanceInput(
        inference_substrate=LIVE_LLM_INFERENCE_SUBSTRATE,
        gpu_uuid=selected.get("gpu_uuid"),
        gpu_model=selected.get("gpu_model"),
        cuda_device=selected.get("index"),
        model_repository=model_repository,
        model_filename=model_filename,
        model_hash=model_hash,
        n_ctx=getattr(prop, "observed_server_n_ctx", None) or getattr(prop, "n_ctx", None),
        server_binary=server_binary,
        server_binary_hash=_sha256_file(server_binary),
        server_command_hash=(
            "sha256:" + hashlib.sha256(canonical_arc_eval_provenance_bytes(command)).hexdigest()
            if command
            else None
        ),
        endpoint=endpoint,
        port=getattr(prop, "port", None),
        lease_id=authority.get("lease_id"),
        lease_hash=authority.get("lease_hash"),
        lease_issued_at=authority.get("lease_issued_at"),
        lease_expires_at=authority.get("lease_expires_at"),
        lease_checked_at=lease_checked_at or authority.get("lease_checked_at"),
        schema_version=schema_version,
        **(identity or {}),
        **common,
    )
    return build_arc_eval_provenance(source)


def generator_provenance(policy: Any) -> dict[str, object]:
    """What model, if any, this policy can actually reach right now."""

    out: dict[str, object] = {"resolved": False}
    prop = getattr(policy, "proposer", None)
    if prop is None:
        out["note"] = "policy exposes no proposer (explorer-tier policy has none)"
        return out
    out["resolved"] = True
    for name, attr in (
        ("server_url", "_url"),
        ("server_binary", "generator_server_path"),
        ("server_command", "last_launch_argv"),
        ("model_repository", "model_repository"),
        ("model_filename", "model_filename"),
        ("model_path", "model_path"),
        ("observed_server_model_path", "observed_server_model_path"),
        ("reuse_model_check", "reuse_model_check"),
        ("n_ctx", "n_ctx"),
        ("observed_server_n_ctx", "observed_server_n_ctx"),
        ("ffn_cpu_layers", "ffn_cpu_layers"),
    ):
        try:
            value = getattr(prop, attr, None)
            out[name] = value() if callable(value) else value
        except Exception as exc:  # noqa: BLE001
            out[name] = f"unreadable: {type(exc).__name__}: {exc}"
    return out


def completion_counters(policy: Any) -> dict:
    """Snapshot legacy proposer channel counters (REQ-ARC-WMTE-6710)."""

    prop = getattr(policy, "proposer", None)
    totals = getattr(prop, "channel_totals", None)
    return dict(totals) if isinstance(totals, dict) else {}
