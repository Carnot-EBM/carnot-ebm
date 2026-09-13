"""ARC evaluation provenance shared by producers and headline consumers.

Spec: REQ-ARC-7010, REQ-ARC-7030, REQ-ARC-7031, REQ-ARC-WMTE-6790,
REQ-ARC-WMTE-6710, REQ-REPORT-7111.

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
import stat
import subprocess
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import urlparse

ARC_EVAL_PROVENANCE_SCHEMA_VERSION = "carnot.arc_eval_provenance.v1"
ARC_EVAL_PROVENANCE_SCHEMA_VERSION_V2 = "carnot.arc_eval_provenance.v2"
ARC_EVAL_PROVENANCE_SCHEMA_VERSION_V3 = "carnot.arc_eval_provenance.v3"
ARC_MODEL_IDENTITY_SCHEMA_VERSION = "carnot.arc_model_identity.v3"
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
TYPED_ARC_MODEL_IDENTITY_KEYS = (
    "identity_schema_version",
    "requested_model_path",
    "requested_model_filename",
    "requested_hf_id",
    "requested_revision",
    "launch_model_argument",
    "observed_server_model_path",
    "observed_server_resolved_path",
    "resolved_model_path",
    "model_file_hash",
    "path_form",
    "raw_server_props",
    "raw_report_observations",
    "identity_obligation_rows",
    "source_provenance",
)
IDENTITY_OBLIGATIONS = (
    "absolute_raw_report",
    "path_resolution",
    "selected_snapshot_relation",
    "launch_report_agreement",
    "content_hash",
    "hub",
    "revision",
    "requested_filename",
    "unique_file_identity",
    "source_provenance",
)
IDENTITY_EVIDENCE_STATUSES = frozenset({"supported", "contradicted", "unknown"})
ARC_EVAL_PROVENANCE_V2_REQUIRED_KEYS = (
    *ARC_EVAL_PROVENANCE_REQUIRED_KEYS[:-1],
    *ARC_MODEL_IDENTITY_KEYS,
    "provenance_hash",
)
ARC_EVAL_PROVENANCE_V3_REQUIRED_KEYS = (
    *ARC_EVAL_PROVENANCE_REQUIRED_KEYS[:-1],
    *TYPED_ARC_MODEL_IDENTITY_KEYS,
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
    identity_schema_version: str | None = None
    launch_model_argument: str | None = None
    observed_server_resolved_path: str | None = None
    path_form: str | None = None
    raw_server_props: Mapping[str, Any] | None = None
    raw_report_observations: list[dict[str, Any]] | None = None
    identity_obligation_rows: list[dict[str, Any]] | None = None
    source_provenance: Mapping[str, Any] | None = None


@dataclass(frozen=True, slots=True)
class ArcEvalProvenanceValidation:
    """Separate evidence validity from the stricter level-headline decision."""

    valid: bool
    headline_eligible: bool
    errors: tuple[str, ...]
    headline_ineligibility: tuple[str, ...] = ()


def canonical_arc_eval_provenance_bytes(value: Mapping[str, Any]) -> bytes:
    """Stable JSON bytes used for the record digest in every process."""

    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()


def compute_arc_frame_sequence_hash(frame_sequence: Any) -> str:
    """Bind a runtime receipt to the exact public-frame sequence on its row."""

    return (
        "sha256:"
        + hashlib.sha256(
            json.dumps(
                frame_sequence, sort_keys=True, separators=(",", ":"), ensure_ascii=True
            ).encode()
        ).hexdigest()
    )


def build_arc_level_claim_receipts(
    *,
    game: str,
    started_at: Any,
    finished_at: Any,
    actions: int,
    level_up_actions: list[int],
    frame_sequence: list[Any],
    induction_attempts: list[Any],
    level_induction_events: list[Any],
) -> dict[str, dict[str, Any]]:
    """Build the two same-row receipts required for a future level headline.

    The attempt receipt identifies the bounded action attempt. The runtime-RE
    receipt binds the public frames and runtime induction observations. Neither
    receipt asserts that a level was solved; the consumer checks them against
    the independently serialized row before deciding headline eligibility.
    """

    return {
        "attempt_receipt": {
            "game": game,
            "started_at": started_at,
            "finished_at": finished_at,
            "actions": actions,
            "level_up_actions": list(level_up_actions),
        },
        "runtime_re_receipt": {
            "game": game,
            "frame_count": len(frame_sequence),
            "frame_sequence_hash": compute_arc_frame_sequence_hash(frame_sequence),
            "induction_attempts": list(induction_attempts),
            "level_induction_events": list(level_induction_events),
        },
    }


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

    REQ-ARC-7030 and REQ-ARC-7031 require filesystem and content evidence.
    Display aliases, file sizes, and ambiguous hard links cannot prove identity.
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

    requested_is_symlink = requested.is_symlink()
    if requested_is_symlink:
        model_root = requested.parent.parent.parent
        blobs_dir = model_root / "blobs"
        if resolved.parent != blobs_dir or observed_canonical.parent != blobs_dir:
            errors.append("observed blob is not reachable from the selected snapshot")
    elif resolved != requested or observed_canonical != requested:
        errors.append("direct snapshot GGUF must equal the server canonical path")

    try:
        if resolved.stat().st_nlink != 1 or observed_canonical.stat().st_nlink != 1:
            errors.append("ambiguous hard link is not accepted")
    except OSError as exc:
        raise ValueError(f"invalid ARC model identity: path metadata failed: {exc}") from exc

    requested_hash = _sha256_file(resolved)
    observed_hash = (
        requested_hash if resolved == observed_canonical else _sha256_file(observed_canonical)
    )
    digest = requested_hash.removeprefix("sha256:") if isinstance(requested_hash, str) else ""
    if requested_is_symlink and (
        not _CONTENT_HASH_RE.fullmatch(resolved.name) or resolved.name != digest
    ):
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


def identity_source_payload_sha256(value: Mapping[str, Any]) -> str:
    """Hash the terminal source projection used by Exp7051.

    The two excluded fields either contain the digest or a receipt that repeats
    it. Removing only those fields keeps every evidence-bearing value covered.
    """

    projection = {
        key: item
        for key, item in value.items()
        if key not in {"reproducibility_checksum", "checksum_recomputation_rows"}
    }
    return "sha256:" + hashlib.sha256(canonical_arc_eval_provenance_bytes(projection)).hexdigest()


def _path_fingerprint(path: Any) -> dict[str, Any] | None:
    """Capture link and target identity so a later symlink swap is visible."""

    candidate = _absolute_path(os.fspath(path) if isinstance(path, os.PathLike) else path)
    if candidate is None:
        return None
    try:
        link_stat = candidate.lstat()
        resolved = candidate.resolve(strict=True)
        target_stat = resolved.stat()
        link_target = os.readlink(candidate) if stat.S_ISLNK(link_stat.st_mode) else None
    except OSError:
        return None
    return {
        "path": str(candidate),
        "link_device": link_stat.st_dev,
        "link_inode": link_stat.st_ino,
        "link_mode": link_stat.st_mode,
        "link_target": link_target,
        "resolved_path": str(resolved),
        "target_device": target_stat.st_dev,
        "target_inode": target_stat.st_ino,
        "target_mode": target_stat.st_mode,
        "target_nlink": target_stat.st_nlink,
        "target_size": target_stat.st_size,
    }


def process_start_tick(pid: Any) -> int | None:
    """Read the kernel start tick that distinguishes a live process from PID reuse."""

    if not isinstance(pid, int) or isinstance(pid, bool) or pid <= 0:
        return None
    try:
        fields = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8").split()
        tick = int(fields[21])
    except (IndexError, OSError, ValueError):
        return None
    return tick if tick > 0 else None


def capture_arc_model_identity_source_provenance(
    *,
    raw_server_props: Mapping[str, Any],
    requested_model_path: Any,
    source_kind: str,
    source_artifact_path: str | Path | None = None,
    launch_model_argument: Any = None,
    server_pid: Any = None,
    server_pid_start_tick: Any = None,
) -> dict[str, Any]:
    """Seal the raw report and requested path before identity validation.

    A live caller gets a report digest and a filesystem fingerprint. A file-
    backed audit also gets the source file hash and its declared terminal
    checksum. The builder later recomputes these values instead of trusting
    this capture.
    """

    if not isinstance(raw_server_props, Mapping):
        raise TypeError("raw_server_props must be a mapping")
    if not isinstance(source_kind, str) or not source_kind.strip():
        raise ValueError("source_kind must be non-empty")
    source: dict[str, Any] = {
        "source_kind": source_kind,
        "raw_report_sha256": "sha256:"
        + hashlib.sha256(canonical_arc_eval_provenance_bytes(raw_server_props)).hexdigest(),
        "requested_path_fingerprint": _path_fingerprint(requested_model_path),
        "source_artifact_path": None,
        "source_artifact_hash": None,
        "source_reproducibility_checksum": None,
        "launch_model_argument": (
            str(launch_model_argument)
            if isinstance(launch_model_argument, str) and launch_model_argument
            else None
        ),
        "runtime_process": None,
    }
    if server_pid is not None or server_pid_start_tick is not None:
        observed_start_tick = process_start_tick(server_pid)
        source["runtime_process"] = {
            "pid": server_pid,
            "launch_start_tick": server_pid_start_tick,
            "observed_start_tick": observed_start_tick,
            "supported": bool(
                isinstance(server_pid, int)
                and not isinstance(server_pid, bool)
                and server_pid > 0
                and isinstance(server_pid_start_tick, int)
                and not isinstance(server_pid_start_tick, bool)
                and server_pid_start_tick > 0
                and observed_start_tick == server_pid_start_tick
            ),
        }
    if source_artifact_path is None:
        return source
    artifact_path = Path(source_artifact_path)
    source["source_artifact_path"] = str(artifact_path.absolute())
    source["source_artifact_hash"] = _sha256_file(artifact_path)
    try:
        artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        artifact = None
    if isinstance(artifact, dict):
        checksum = artifact.get("reproducibility_checksum")
        source["source_reproducibility_checksum"] = checksum if isinstance(checksum, str) else None
    return source


def _raw_report_kind(value: Any, *, present: bool) -> str:
    if not present:
        return "missing"
    if not isinstance(value, str):
        return "non_string"
    if not value.strip():
        return "blank"
    return "absolute_path" if Path(value).is_absolute() else "relative_path"


def _resolve_regular(path: Any) -> Path | None:
    candidate = _absolute_path(path)
    if candidate is None:
        return None
    try:
        resolved = candidate.resolve(strict=True)
        return resolved if resolved.is_file() else None
    except OSError:
        return None


def _stable_regular_file_digest(path: Path | None) -> tuple[str | None, os.stat_result | None]:
    """Hash one regular file descriptor and confirm its directory entry stayed put."""

    if path is None:
        return None, None
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError:
        return None, None
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            return None, before
        digest = hashlib.sha256()
        while chunk := os.read(descriptor, 1024 * 1024):
            digest.update(chunk)
        after = os.fstat(descriptor)
        current = path.stat()
    except OSError:
        return None, None
    finally:
        os.close(descriptor)
    stable = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
    ) == (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    ) and (after.st_dev, after.st_ino) == (current.st_dev, current.st_ino)
    return ("sha256:" + digest.hexdigest() if stable else None), after


def _obligation(
    obligation: str, status: str, evidence_source: str, observed: Any
) -> dict[str, Any]:
    return {
        "obligation": obligation,
        "status": status,
        "evidence_source": evidence_source,
        "observed_value": observed,
        "terminal": True,
    }


def _status(condition: bool | None) -> str:
    if condition is None:
        return "unknown"
    return "supported" if condition else "contradicted"


def build_typed_arc_model_identity_receipt(
    *,
    selected_model_spec: Mapping[str, Any],
    launch_model_argument: Any,
    raw_server_props: Mapping[str, Any],
    source_provenance: Mapping[str, Any],
) -> dict[str, Any]:
    """Build typed raw observations and independently supported obligations.

    REQ-ARC-7052 keeps raw `/props` values separate from path resolutions.
    Invalid evidence still produces typed rows so an audit can name whether
    evidence was contradicted or unavailable. The validator accepts a receipt
    only when it can rebuild the same rows and all obligations are supported.
    """

    if not isinstance(selected_model_spec, Mapping):
        raise TypeError("selected_model_spec must be a mapping")
    if not isinstance(raw_server_props, Mapping):
        raise TypeError("raw_server_props must be a mapping")
    if not isinstance(source_provenance, Mapping):
        raise TypeError("source_provenance must be a mapping")
    props = dict(raw_server_props)
    source = dict(source_provenance)
    requested_raw = selected_model_spec.get("model_path")
    filename = selected_model_spec.get("model_filename")
    hf_id = selected_model_spec.get("hf_id")
    revision = selected_model_spec.get("revision")
    expected_hash = selected_model_spec.get("model_file_hash")
    observed_raw = props.get("model_path")
    requested = _absolute_path(requested_raw)
    observed = _absolute_path(observed_raw)
    requested_resolved = _resolve_regular(requested_raw)
    observed_resolved = _resolve_regular(observed_raw)

    raw_rows: list[dict[str, Any]] = []
    absolute_resolutions: dict[str, Path] = {}
    for field in ("model_path", "model", "model_alias"):
        present = field in props
        raw = props.get(field)
        kind = _raw_report_kind(raw, present=present)
        resolved = _resolve_regular(raw) if kind == "absolute_path" else None
        if resolved is not None:
            absolute_resolutions[field] = resolved
        if kind in {"missing", "blank", "non_string", "relative_path"}:
            row_status = "unknown"
        else:
            row_status = "supported" if resolved is not None else "contradicted"
        raw_rows.append(
            {
                "field": field,
                "present": present,
                "raw_value": raw,
                "raw_kind": kind,
                "resolved_path": str(resolved) if resolved is not None else None,
                "status": row_status,
                "evidence_source": f"/props.{field}",
                "terminal": True,
            }
        )

    distinct_resolutions = {str(path) for path in absolute_resolutions.values()}
    report_conflict = len(distinct_resolutions) > 1
    if report_conflict:
        for row in raw_rows:
            if row["field"] in absolute_resolutions:
                row["status"] = "contradicted"

    requested_digest, requested_stat = _stable_regular_file_digest(requested_resolved)
    observed_digest, observed_stat = _stable_regular_file_digest(observed_resolved)
    requested_fingerprint = _path_fingerprint(requested_raw)
    requested_is_symlink = bool(requested and requested.is_symlink())
    layout_revision = huggingface_snapshot_revision(requested_raw, hf_id)
    model_root = requested.parent.parent.parent if requested is not None else None
    blobs_dir = model_root / "blobs" if model_root is not None else None

    if requested_resolved is None or observed_resolved is None:
        path_form = "unknown"
    elif report_conflict:
        path_form = "conflicting"
    elif requested_is_symlink and observed == requested:
        path_form = "snapshot_alias"
    elif requested_is_symlink and observed == observed_resolved == requested_resolved:
        path_form = "canonical_blob"
    elif not requested_is_symlink and observed == requested == requested_resolved:
        path_form = "direct_file"
    else:
        path_form = "unknown"

    absolute_raw_condition: bool | None
    if "model_path" not in props or observed_raw in (None, ""):
        absolute_raw_condition = None
    else:
        absolute_raw_condition = observed is not None and observed_resolved is not None
    path_condition = (
        None
        if requested is None or observed is None
        else requested_resolved is not None and observed_resolved is not None
    )
    hub_condition = (
        None if requested is None or not isinstance(hf_id, str) else layout_revision is not None
    )
    revision_condition = (
        None
        if requested is None or not isinstance(revision, str)
        else layout_revision == revision and requested.parent.name == revision
    )
    if path_form in {"snapshot_alias", "canonical_blob"}:
        snapshot_condition: bool | None = (
            requested_resolved is not None
            and observed_resolved == requested_resolved
            and blobs_dir is not None
            and requested_resolved.parent == blobs_dir
        )
    elif path_form == "direct_file":
        snapshot_condition = requested_resolved == observed_resolved == requested
    else:
        snapshot_condition = False if requested is not None and observed is not None else None

    launch = _absolute_path(launch_model_argument)
    launch_resolved = _resolve_regular(launch_model_argument)
    launch_condition = (
        None
        if launch_model_argument in (None, "")
        else launch is not None
        and launch_resolved is not None
        and launch_resolved == requested_resolved == observed_resolved
        and not report_conflict
    )
    digest_name = (
        requested_digest.removeprefix("sha256:") if isinstance(requested_digest, str) else None
    )
    blob_name_condition = (
        requested_resolved is not None
        and requested_resolved.parent == blobs_dir
        and requested_resolved.name == digest_name
    )
    content_condition = (
        None
        if requested_digest is None or observed_digest is None or not isinstance(expected_hash, str)
        else requested_digest == observed_digest == expected_hash
        and (not requested_is_symlink or blob_name_condition)
    )
    alias_values = [props.get(field) for field in ("model", "model_alias") if field in props]
    alias_names_agree = all(
        value in (None, "") or (isinstance(value, str) and Path(value).name == filename)
        for value in alias_values
    )
    filename_condition = (
        None
        if requested is None or not isinstance(filename, str)
        else Path(filename).name == filename
        and filename.lower().endswith(".gguf")
        and requested.name == filename
        and alias_names_agree
    )
    captured_fingerprint = source.get("requested_path_fingerprint")
    captured_target_matches = bool(
        isinstance(captured_fingerprint, Mapping)
        and requested_resolved is not None
        and captured_fingerprint.get("resolved_path") == str(requested_resolved)
        and requested_stat is not None
        and captured_fingerprint.get("target_device") == requested_stat.st_dev
        and captured_fingerprint.get("target_inode") == requested_stat.st_ino
        and captured_fingerprint.get("target_nlink") == requested_stat.st_nlink
        and captured_fingerprint.get("target_size") == requested_stat.st_size
    )
    unique_condition = (
        None
        if requested_stat is None or observed_stat is None
        else captured_target_matches
        and requested_stat.st_nlink == observed_stat.st_nlink
        and (requested_stat.st_dev, requested_stat.st_ino)
        == (observed_stat.st_dev, observed_stat.st_ino)
    )

    expected_raw_hash = (
        "sha256:" + hashlib.sha256(canonical_arc_eval_provenance_bytes(props)).hexdigest()
    )
    source_checks: list[bool] = [
        isinstance(source.get("source_kind"), str) and bool(source.get("source_kind")),
        source.get("raw_report_sha256") == expected_raw_hash,
        source.get("requested_path_fingerprint") == requested_fingerprint,
    ]
    captured_launch = source.get("launch_model_argument")
    if captured_launch is not None:
        source_checks.append(captured_launch == launch_model_argument)
    runtime_process = source.get("runtime_process")
    if source.get("source_kind") == "live_server_props":
        source_checks.append(isinstance(runtime_process, Mapping))
    if runtime_process is not None:
        source_checks.extend(
            [
                isinstance(runtime_process, Mapping),
                isinstance(runtime_process, Mapping) and runtime_process.get("supported") is True,
                isinstance(runtime_process, Mapping)
                and runtime_process.get("observed_start_tick")
                == runtime_process.get("launch_start_tick"),
            ]
        )
    artifact_path_raw = source.get("source_artifact_path")
    if artifact_path_raw is not None:
        artifact_path = _absolute_path(artifact_path_raw)
        artifact: Any = None
        try:
            artifact = (
                json.loads(artifact_path.read_text(encoding="utf-8")) if artifact_path else None
            )
        except (OSError, json.JSONDecodeError):
            artifact = None
        source_checks.extend(
            [
                artifact is not None,
                _sha256_file(artifact_path) == source.get("source_artifact_hash"),
                isinstance(artifact, dict) and artifact.get("raw_server_props") == props,
                isinstance(artifact, dict)
                and artifact.get("reproducibility_checksum")
                == source.get("source_reproducibility_checksum")
                == identity_source_payload_sha256(artifact),
            ]
        )
    source_condition = all(source_checks) if source_checks else None

    obligations = [
        _obligation(
            "absolute_raw_report",
            _status(
                absolute_raw_condition and not report_conflict
                if absolute_raw_condition is not None
                else None
            ),
            "/props.model_path and absolute /props candidates",
            {"model_path": observed_raw, "conflicting_absolute_fields": report_conflict},
        ),
        _obligation(
            "path_resolution",
            _status(path_condition),
            "requested path and /props.model_path strict resolution",
            {
                "requested": str(requested_resolved) if requested_resolved else None,
                "observed": str(observed_resolved) if observed_resolved else None,
            },
        ),
        _obligation(
            "selected_snapshot_relation",
            _status(snapshot_condition),
            "selected snapshot path, resolved target, and Hugging Face blob directory",
            path_form,
        ),
        _obligation(
            "launch_report_agreement",
            _status(launch_condition),
            "launch -m argument and /props.model_path",
            {"launch": launch_model_argument, "report": observed_raw},
        ),
        _obligation(
            "content_hash",
            _status(content_condition),
            "stable requested and observed file descriptors plus selected SHA-256",
            {
                "requested_hash": requested_digest,
                "observed_hash": observed_digest,
                "expected_hash": expected_hash,
            },
        ),
        _obligation("hub", _status(hub_condition), "selected snapshot directory", hf_id),
        _obligation(
            "revision", _status(revision_condition), "selected snapshot directory", revision
        ),
        _obligation(
            "requested_filename",
            _status(filename_condition),
            "selected file name and raw report aliases",
            filename,
        ),
        _obligation(
            "unique_file_identity",
            _status(unique_condition),
            "stable file descriptor device, inode, and link count",
            {
                "requested_device": requested_stat.st_dev if requested_stat else None,
                "requested_inode": requested_stat.st_ino if requested_stat else None,
                "requested_nlink": requested_stat.st_nlink if requested_stat else None,
                "observed_device": observed_stat.st_dev if observed_stat else None,
                "observed_inode": observed_stat.st_ino if observed_stat else None,
                "observed_nlink": observed_stat.st_nlink if observed_stat else None,
                "captured_target": captured_fingerprint,
            },
        ),
        _obligation(
            "source_provenance",
            _status(source_condition),
            "sealed raw report, requested path fingerprint, and optional source artifact",
            source,
        ),
    ]
    return {
        "identity_schema_version": ARC_MODEL_IDENTITY_SCHEMA_VERSION,
        "requested_model_path": requested_raw,
        "requested_model_filename": filename,
        "requested_hf_id": hf_id,
        "requested_revision": revision,
        "launch_model_argument": launch_model_argument,
        "observed_server_model_path": observed_raw,
        "observed_server_resolved_path": (
            str(observed_resolved) if observed_resolved is not None else None
        ),
        "resolved_model_path": str(requested_resolved) if requested_resolved is not None else None,
        "model_file_hash": expected_hash,
        "path_form": path_form,
        "raw_server_props": props,
        "raw_report_observations": raw_rows,
        "identity_obligation_rows": obligations,
        "source_provenance": source,
    }


def validate_typed_arc_model_identity_receipt(
    receipt: Any,
) -> ArcEvalProvenanceValidation:
    """Rebuild a typed receipt and require every named obligation to hold."""

    if not isinstance(receipt, dict):
        return ArcEvalProvenanceValidation(False, False, ("typed receipt must be an object",))
    errors: list[str] = []
    expected_keys = set(TYPED_ARC_MODEL_IDENTITY_KEYS)
    if set(receipt) != expected_keys:
        errors.append("typed receipt fields do not match the current schema")
    if receipt.get("identity_schema_version") != ARC_MODEL_IDENTITY_SCHEMA_VERSION:
        errors.append("identity_schema_version is not current")
    rows = receipt.get("identity_obligation_rows")
    if not isinstance(rows, list) or [
        row.get("obligation") for row in rows if isinstance(row, dict)
    ] != list(IDENTITY_OBLIGATIONS):
        errors.append("identity obligations are missing, duplicated, or out of order")
    elif any(
        row.get("status") not in IDENTITY_EVIDENCE_STATUSES
        or not isinstance(row.get("evidence_source"), str)
        or not row.get("evidence_source")
        for row in rows
    ):
        errors.append("identity obligation row is malformed")
    elif any(row.get("status") != "supported" for row in rows):
        errors.append("every identity obligation must be supported")
    raw_props = receipt.get("raw_server_props")
    source = receipt.get("source_provenance")
    if isinstance(raw_props, Mapping) and isinstance(source, Mapping):
        try:
            rebuilt = build_typed_arc_model_identity_receipt(
                selected_model_spec={
                    "model_path": receipt.get("requested_model_path"),
                    "model_filename": receipt.get("requested_model_filename"),
                    "hf_id": receipt.get("requested_hf_id"),
                    "revision": receipt.get("requested_revision"),
                    "model_file_hash": receipt.get("model_file_hash"),
                },
                launch_model_argument=receipt.get("launch_model_argument"),
                raw_server_props=raw_props,
                source_provenance=source,
            )
        except (OSError, TypeError, ValueError) as exc:
            errors.append(f"typed receipt rebuild failed: {exc}")
        else:
            if canonical_arc_eval_provenance_bytes(rebuilt) != canonical_arc_eval_provenance_bytes(
                receipt
            ):
                errors.append("typed receipt does not match independently rebuilt evidence")
    else:
        errors.append("raw report or source provenance is missing")
    unique = tuple(dict.fromkeys(errors))
    return ArcEvalProvenanceValidation(not unique, not unique, unique)


def read_complete_legacy_arc_eval_provenance(record: Any) -> dict[str, Any]:
    """Read a complete v1 or v2 row without inventing current identity facts."""

    if not isinstance(record, dict) or record.get("schema_version") not in {
        ARC_EVAL_PROVENANCE_SCHEMA_VERSION,
        ARC_EVAL_PROVENANCE_SCHEMA_VERSION_V2,
    }:
        raise ValueError("legacy schema must be version one or version two")
    decision = validate_arc_eval_provenance(record)
    if not decision.valid:
        raise ValueError("complete legacy provenance required: " + "; ".join(decision.errors))
    return json.loads(json.dumps(record, sort_keys=True))


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
    elif schema_version == ARC_EVAL_PROVENANCE_SCHEMA_VERSION_V3:
        required_keys = ARC_EVAL_PROVENANCE_V3_REQUIRED_KEYS
    else:
        required_keys = ARC_EVAL_PROVENANCE_V3_REQUIRED_KEYS
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
        elif schema_version == ARC_EVAL_PROVENANCE_SCHEMA_VERSION_V3:
            errors.append("typed model identity is not valid for a no-LLM row")
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
        elif schema_version == ARC_EVAL_PROVENANCE_SCHEMA_VERSION_V3:
            typed_identity = {key: record.get(key) for key in TYPED_ARC_MODEL_IDENTITY_KEYS}
            identity_decision = validate_typed_arc_model_identity_receipt(typed_identity)
            errors.extend(identity_decision.errors)
            if record.get("model_repository") != record.get("requested_hf_id"):
                errors.append("model_repository contradicts requested_hf_id")
            if record.get("model_filename") != record.get("requested_model_filename"):
                errors.append("model_filename contradicts requested_model_filename")
            if record.get("model_hash") != record.get("model_file_hash"):
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
    identity = {key: values.pop(key) for key in TYPED_ARC_MODEL_IDENTITY_KEYS}
    record = {"schema_version": schema_version, **values}
    if schema_version == ARC_EVAL_PROVENANCE_SCHEMA_VERSION_V2:
        record.update({key: identity[key] for key in ARC_MODEL_IDENTITY_KEYS})
        typed_only = set(TYPED_ARC_MODEL_IDENTITY_KEYS) - set(ARC_MODEL_IDENTITY_KEYS)
        if any(identity[key] is not None for key in typed_only):
            raise ValueError("version two provenance cannot carry typed identity evidence")
    elif schema_version == ARC_EVAL_PROVENANCE_SCHEMA_VERSION_V3:
        record.update(identity)
    elif any(value is not None for value in identity.values()):
        raise ValueError("legacy provenance input cannot carry current model identity fields")
    record["provenance_hash"] = compute_arc_eval_provenance_hash(record)
    decision = validate_arc_eval_provenance(record)
    if not decision.valid:
        raise ValueError("invalid ARC evaluation provenance: " + "; ".join(decision.errors))
    return record


def validate_arc_evaluation_row(row: Any) -> ArcEvalProvenanceValidation:
    """Validate forward provenance and independently gate positive level credit.

    Historical rows remain readable by callers even when this decision is
    invalid. New writers must reject that state. A valid development-proxy or
    outer-loop row remains evidence, but a positive level count is eligible for
    a live headline only when both same-row receipts reconcile.
    """

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
    if unique_errors:
        return ArcEvalProvenanceValidation(False, False, unique_errors)

    # REQ-ARC-7010 predates level-bearing rows and its direct record fixtures do
    # not carry `levels`. Preserve that validation API while enforcing the
    # stricter forward rule whenever a writer emits an explicit level count.
    if "levels" not in row:
        return ArcEvalProvenanceValidation(True, True, ())

    levels = row.get("levels")
    headline_errors: list[str] = []
    if not _integer(levels) or levels <= 0:
        headline_errors.append("evaluation row has no positive level claim")
    elif row_solve != "live_agent_self_discovery":
        headline_errors.append(
            "positive level claim requires solve_provenance=live_agent_self_discovery"
        )
    else:
        headline_errors.extend(_arc_level_claim_receipt_errors(row, levels))
    unique_headline_errors = tuple(dict.fromkeys(headline_errors))
    return ArcEvalProvenanceValidation(True, not unique_headline_errors, (), unique_headline_errors)


def _arc_level_claim_receipt_errors(row: Mapping[str, Any], levels: int) -> list[str]:
    """Reconcile a live claim's attempt and runtime-RE receipts with its row."""

    errors: list[str] = []
    game = row.get("game")
    actions = row.get("actions")
    attempt = row.get("attempt_receipt")
    if not isinstance(attempt, dict):
        errors.append("positive live level claim requires an attempt_receipt object")
    else:
        if not _nonempty(game) or attempt.get("game") != game:
            errors.append("attempt_receipt game must match the evaluation row")
        if (
            attempt.get("started_at") != row.get("started_at")
            or _timestamp(attempt.get("started_at")) is None
        ):
            errors.append("attempt_receipt started_at must match the timezone-aware row timestamp")
        if (
            attempt.get("finished_at") != row.get("finished_at")
            or _timestamp(attempt.get("finished_at")) is None
        ):
            errors.append("attempt_receipt finished_at must match the timezone-aware row timestamp")
        if not _integer(actions) or actions <= 0 or attempt.get("actions") != actions:
            errors.append("attempt_receipt actions must match the positive row action count")
        level_ups = attempt.get("level_up_actions")
        if (
            not isinstance(level_ups, list)
            or len(level_ups) < levels
            or any(not _integer(value) or value <= 0 for value in level_ups)
            or any(left > right for left, right in zip(level_ups, level_ups[1:]))
            or (_integer(actions) and any(value > actions for value in level_ups))
        ):
            errors.append("attempt_receipt level_up_actions must cover the claimed levels")

    runtime = row.get("runtime_re_receipt")
    frames = row.get("frame_sequence")
    if not isinstance(runtime, dict):
        errors.append("positive live level claim requires a runtime_re_receipt object")
    else:
        if not _nonempty(game) or runtime.get("game") != game:
            errors.append("runtime_re_receipt game must match the evaluation row")
        if not isinstance(frames, list) or not frames:
            errors.append("runtime_re_receipt requires a non-empty public frame_sequence")
        else:
            if runtime.get("frame_count") != len(frames):
                errors.append("runtime_re_receipt frame_count must match frame_sequence")
            try:
                expected_hash = compute_arc_frame_sequence_hash(frames)
            except (TypeError, ValueError):
                expected_hash = None
            if runtime.get("frame_sequence_hash") != expected_hash:
                errors.append("runtime_re_receipt frame_sequence_hash must bind frame_sequence")
        if not isinstance(runtime.get("induction_attempts"), list):
            errors.append("runtime_re_receipt induction_attempts must be a list")
        if not isinstance(runtime.get("level_induction_events"), list):
            errors.append("runtime_re_receipt level_induction_events must be a list")
    return errors


def prepare_arc_evaluation_row_for_write(row: Any) -> dict[str, Any]:
    """Validate one new row and stamp the non-authoritative consumer decision."""

    decision = validate_arc_evaluation_row(row)
    if not decision.valid:
        raise ValueError("invalid ARC evaluation row: " + "; ".join(decision.errors))
    prepared = dict(row)
    prepared["arc_provenance_valid"] = True
    prepared["arc_headline_eligible"] = decision.headline_eligible
    prepared["arc_headline_ineligibility"] = list(decision.headline_ineligibility)
    return prepared


def serialize_arc_evaluation_payload(payload: Any) -> str:
    """Serialize only forward rows that pass REQ-REPORT-7111 validation.

    The live evaluator imports this exact boundary for both partial and final
    files. Keeping it beside the validator lets deterministic canaries execute
    the production serialization path without importing or running the ARC
    game environment.
    """

    if not isinstance(payload, dict):
        raise ValueError("ARC evaluation payload must be an object")
    rows = payload.get("per_game")
    if not isinstance(rows, list):
        raise ValueError("ARC evaluation payload per_game must be a list")
    prepared = dict(payload)
    prepared["per_game"] = [prepare_arc_evaluation_row_for_write(row) for row in rows]
    return json.dumps(prepared, indent=2)


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
        if candidate.get("identity_schema_version") == ARC_MODEL_IDENTITY_SCHEMA_VERSION:
            typed_decision = validate_typed_arc_model_identity_receipt(candidate)
            if not typed_decision.valid:
                raise ValueError(
                    "model_identity_receipt is invalid: " + "; ".join(typed_decision.errors)
                )
            identity = candidate
        else:
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
            props_reader = getattr(prop, "server_props", None)
            if not callable(props_reader):
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
            else:
                try:
                    raw_props = props_reader()
                except Exception:  # noqa: BLE001 - missing raw evidence must reject
                    raw_props = {}
                raw_props = raw_props if isinstance(raw_props, Mapping) else {}
                source_provenance = capture_arc_model_identity_source_provenance(
                    raw_server_props=raw_props,
                    requested_model_path=requested_path,
                    source_kind="live_server_props",
                    launch_model_argument=(
                        command[command.index("-m") + 1]
                        if command and "-m" in command and command.index("-m") + 1 < len(command)
                        else None
                    ),
                    server_pid=getattr(getattr(prop, "_proc", None), "pid", None),
                    server_pid_start_tick=getattr(prop, "server_pid_start_tick", None),
                )
                launch_argument = requested_path
                if command and "-m" in command:
                    try:
                        launch_argument = command[command.index("-m") + 1]
                    except IndexError:
                        launch_argument = None
                identity = build_typed_arc_model_identity_receipt(
                    selected_model_spec={
                        "model_path": requested_path,
                        "model_filename": requested_filename,
                        "hf_id": getattr(prop, "model_repository", None),
                        "revision": requested_revision,
                        "model_file_hash": _sha256_file(requested_path),
                    },
                    launch_model_argument=launch_argument,
                    raw_server_props=raw_props,
                    source_provenance=source_provenance,
                )
                typed_decision = validate_typed_arc_model_identity_receipt(identity)
                if not typed_decision.valid:
                    raise ValueError(
                        "live model identity is invalid: " + "; ".join(typed_decision.errors)
                    )
    if identity is not None:
        model_path = identity["requested_model_path"]
        model_filename = identity["requested_model_filename"]
        model_repository = identity["requested_hf_id"]
        model_hash = identity["model_file_hash"]
        schema_version = (
            ARC_EVAL_PROVENANCE_SCHEMA_VERSION_V3
            if identity.get("identity_schema_version") == ARC_MODEL_IDENTITY_SCHEMA_VERSION
            else ARC_EVAL_PROVENANCE_SCHEMA_VERSION_V2
        )
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
