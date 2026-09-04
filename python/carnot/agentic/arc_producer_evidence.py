"""Prospective evidence for engines emitted by the live ARC producer.

REQ-ARC-WMTE-6993 requires evidence to exist before an engine can become a
scientific record. This module stages the exact prompt and ordered transitions
before synthesis. It publishes one directory by rename and appends the manifest
row last. The final row is the eligibility marker, so a stopped write cannot
look complete.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
from typing import Any
from uuid import uuid4


EVIDENCE_ENVELOPE_SCHEMA = "carnot.arc.live_engine_evidence.v1"
EVIDENCE_MANIFEST_SCHEMA = "carnot.arc.live_engine_manifest.v1"
LIVE_TRANSITION_SOURCE_KIND = "live_agent_attempts"

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SOURCE_PATHS = {
    "scorer": REPO_ROOT / "scripts" / "arc_e3_induced_model_quality.py",
    "live_policy": REPO_ROOT / "python" / "carnot" / "agentic" / "arc_competition_agent.py",
    "agent_factory": REPO_ROOT / "python" / "carnot" / "agentic" / "arc_competition_agent.py",
}

REQUIRED_ENVELOPE_FIELDS = (
    "schema",
    "run_id",
    "game",
    "created_at",
    "published_at",
    "raw_prompt_path",
    "raw_prompt_sha256",
    "transition_jsonl_path",
    "transition_sha256",
    "transition_count",
    "transition_source_kind",
    "engine_path",
    "engine_sha256",
    "environment_receipt_path",
    "environment_receipt_sha256",
    "scorer_path",
    "scorer_sha256",
    "live_policy_path",
    "live_policy_sha256",
    "agent_factory_path",
    "agent_factory_sha256",
    "manifest_row_path",
    "manifest_row_sha256",
    "envelope_sha256",
)

_SAFE_COMPONENT = re.compile(r"^[A-Za-z0-9_.-]+$")
_HASH_FIELDS = {
    "raw_prompt_path": "raw_prompt_sha256",
    "transition_jsonl_path": "transition_sha256",
    "engine_path": "engine_sha256",
    "environment_receipt_path": "environment_receipt_sha256",
    "scorer_path": "scorer_sha256",
    "live_policy_path": "live_policy_sha256",
    "agent_factory_path": "agent_factory_sha256",
    "manifest_row_path": "manifest_row_sha256",
}


class EvidencePublishInterrupted(RuntimeError):
    """A deterministic failpoint stopped publication before the manifest marker."""


def canonical_json_bytes(value: Any) -> bytes:
    """Return the stable bytes used by every JSON hash in this contract."""

    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    """Return a labeled SHA-256 digest so algorithms cannot be confused."""

    return "sha256:" + hashlib.sha256(value).hexdigest()


def compute_envelope_hash(envelope: Mapping[str, Any]) -> str:
    """Hash the envelope without its two later self-reference fields.

    The manifest hash depends on this value. Excluding the manifest hash from
    this projection removes the otherwise unavoidable hash cycle.
    """

    projected = dict(envelope)
    projected.pop("envelope_sha256", None)
    projected.pop("manifest_row_sha256", None)
    return sha256_bytes(canonical_json_bytes(projected))


def compute_manifest_row_hash(row: Mapping[str, Any]) -> str:
    """Hash the canonical row body without its own digest field."""

    projected = dict(row)
    projected.pop("manifest_row_sha256", None)
    return sha256_bytes(canonical_json_bytes(projected))


def _json_value(value: Any) -> Any:
    """Convert array and scalar fixtures without changing their values or order."""

    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if hasattr(value, "tolist"):
        return _json_value(value.tolist())
    if hasattr(value, "item"):
        return _json_value(value.item())
    return value


def _transition_row(value: Any, index: int) -> dict[str, Any]:
    """Copy one transition into the producer-owned ordered JSONL schema."""

    def get(name: str, default: Any = None) -> Any:
        if isinstance(value, Mapping):
            return value.get(name, default)
        return getattr(value, name, default)

    row = {
        "index": index,
        "grid": _json_value(get("grid")),
        "action": int(get("action", -1)),
        "data": _json_value(get("data")),
        "next_grid": _json_value(get("next_grid")),
        "level_before": int(get("level_before", -1)),
        "level_after": int(get("level_after", -1)),
    }
    row["transition_id"] = hashlib.sha256(canonical_json_bytes(row)).hexdigest()[:16]
    return row


def _write_fsynced(path: Path, value: bytes) -> None:
    """Write one staged file and flush it before synthesis or publication continues."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        handle.write(value)
        handle.flush()
        os.fsync(handle.fileno())


def _fsync_directory(path: Path) -> None:
    """Flush directory metadata where the platform supports directory handles."""

    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _safe_component(value: str, label: str) -> str:
    """Keep game and run identifiers as one path component."""

    if not value or not _SAFE_COMPONENT.fullmatch(value):
        raise ValueError(f"unsafe {label}: {value!r}")
    return value


def _timestamp() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%S_%f")


@dataclass
class PublishedEngineEvidence:
    """Paths and records returned after the manifest marker is durable."""

    row: dict[str, Any]
    envelope: dict[str, Any]
    prompt_path: Path
    transition_path: Path
    engine_path: Path
    canonical_engine_path: Path
    envelope_path: Path
    manifest_row_path: Path
    manifest_path: Path


@dataclass
class EngineEvidenceTransaction:
    """A staged producer transaction whose source bytes already exist."""

    store_root: Path
    game: str
    run_id: str
    created_at: str
    transition_source_kind: str | None
    transition_count: int
    staging_dir: Path

    @classmethod
    def begin(
        cls,
        *,
        store_root: Path,
        game: str,
        raw_prompt: bytes,
        transitions: Sequence[Any],
        transition_source_kind: str | None,
        environment_receipt: Mapping[str, Any],
        run_id: str | None = None,
        timestamp: str | None = None,
        source_paths: Mapping[str, Path] | None = None,
    ) -> EngineEvidenceTransaction:
        """Persist all producer inputs before any engine synthesis begins."""

        root = Path(store_root)
        safe_game = _safe_component(str(game), "game")
        created = timestamp or _timestamp()
        safe_run = _safe_component(run_id or f"arc-{created}-{uuid4().hex[:12]}", "run_id")
        staging = root / safe_game / "attempts" / ".evidence-staging" / safe_run
        final = root / safe_game / "attempts" / "evidence" / safe_run
        if staging.exists() or final.exists():
            raise FileExistsError(f"evidence run already exists: {safe_run}")
        staging.mkdir(parents=True)

        ordered = [_transition_row(value, index) for index, value in enumerate(transitions)]
        transition_bytes = b"".join(canonical_json_bytes(row) + b"\n" for row in ordered)
        _write_fsynced(staging / "prompt.raw", bytes(raw_prompt))
        _write_fsynced(staging / "transitions.jsonl", transition_bytes)
        _write_fsynced(
            staging / "environment_receipt.json",
            canonical_json_bytes(_json_value(environment_receipt)),
        )

        selected_sources = dict(DEFAULT_SOURCE_PATHS)
        if source_paths is not None:
            selected_sources.update({name: Path(path) for name, path in source_paths.items()})
        for name in ("scorer", "live_policy", "agent_factory"):
            _write_fsynced(staging / f"{name}.py", selected_sources[name].read_bytes())
        _fsync_directory(staging)
        return cls(
            store_root=root,
            game=safe_game,
            run_id=safe_run,
            created_at=created,
            transition_source_kind=transition_source_kind,
            transition_count=len(ordered),
            staging_dir=staging,
        )

    def _relative(self, name: str) -> str:
        return str(Path(self.game) / "attempts" / "evidence" / self.run_id / name)

    def publish(
        self,
        engine_bytes: bytes,
        *,
        writer: str = "live_engine_producer",
        model: str = "",
        note: str = "",
        timestamp: str | None = None,
        failpoint: str | None = None,
    ) -> PublishedEngineEvidence:
        """Publish the engine bundle and append its manifest marker last."""

        published_at = timestamp or self.created_at
        engine_temp = self.staging_dir / "engine.py.tmp"
        _write_fsynced(engine_temp, bytes(engine_bytes))
        if failpoint == "after_engine_temp":
            raise EvidencePublishInterrupted(failpoint)
        os.replace(engine_temp, self.staging_dir / "engine.py")

        path_fields = {
            "raw_prompt_path": self._relative("prompt.raw"),
            "transition_jsonl_path": self._relative("transitions.jsonl"),
            "engine_path": self._relative("engine.py"),
            "environment_receipt_path": self._relative("environment_receipt.json"),
            "scorer_path": self._relative("scorer.py"),
            "live_policy_path": self._relative("live_policy.py"),
            "agent_factory_path": self._relative("agent_factory.py"),
            "manifest_row_path": self._relative("manifest_row.json"),
        }
        envelope: dict[str, Any] = {
            "schema": EVIDENCE_ENVELOPE_SCHEMA,
            "run_id": self.run_id,
            "game": self.game,
            "created_at": self.created_at,
            "published_at": published_at,
            "transition_count": self.transition_count,
            "transition_source_kind": self.transition_source_kind,
            **path_fields,
        }
        for path_field, hash_field in _HASH_FIELDS.items():
            if path_field == "manifest_row_path":
                continue
            source = self.staging_dir / Path(path_fields[path_field]).name
            envelope[hash_field] = sha256_bytes(source.read_bytes())
        envelope["envelope_sha256"] = compute_envelope_hash(envelope)

        row_body: dict[str, Any] = {
            "schema": EVIDENCE_MANIFEST_SCHEMA,
            "ts": published_at,
            "run_id": self.run_id,
            "game": self.game,
            "complete": True,
            "policy": "e3",
            "writer": str(writer),
            "model": str(model),
            "note": str(note)[:200],
            "file": path_fields["engine_path"],
            "sha256_16": envelope["engine_sha256"].removeprefix("sha256:")[:16],
            "transition_source_kind": self.transition_source_kind,
            "manifest_row_path": path_fields["manifest_row_path"],
            "envelope": {
                "path": self._relative("envelope.json"),
                "sha256": envelope["envelope_sha256"],
            },
            "engine": {
                "path": path_fields["engine_path"],
                "sha256": envelope["engine_sha256"],
            },
            "prompt": {
                "path": path_fields["raw_prompt_path"],
                "sha256": envelope["raw_prompt_sha256"],
            },
            "transitions": {
                "path": path_fields["transition_jsonl_path"],
                "sha256": envelope["transition_sha256"],
            },
            "environment": {
                "path": path_fields["environment_receipt_path"],
                "sha256": envelope["environment_receipt_sha256"],
            },
            "scorer": {
                "path": path_fields["scorer_path"],
                "sha256": envelope["scorer_sha256"],
            },
            "live_policy": {
                "path": path_fields["live_policy_path"],
                "sha256": envelope["live_policy_sha256"],
            },
            "agent_factory": {
                "path": path_fields["agent_factory_path"],
                "sha256": envelope["agent_factory_sha256"],
            },
            "per_game": [{"game": self.game, "engine_sha256": envelope["engine_sha256"]}],
        }
        manifest_row_sha256 = compute_manifest_row_hash(row_body)
        envelope["manifest_row_sha256"] = manifest_row_sha256
        row = dict(row_body)
        row["manifest_row_sha256"] = manifest_row_sha256

        _write_fsynced(self.staging_dir / "manifest_row.json", canonical_json_bytes(row_body))
        _write_fsynced(self.staging_dir / "envelope.json", canonical_json_bytes(envelope))
        _fsync_directory(self.staging_dir)

        final_dir = self.store_root / self.game / "attempts" / "evidence" / self.run_id
        final_dir.parent.mkdir(parents=True, exist_ok=True)
        os.replace(self.staging_dir, final_dir)
        _fsync_directory(final_dir.parent)
        if failpoint == "after_bundle_publish":
            raise EvidencePublishInterrupted(failpoint)

        canonical_engine = self.store_root / self.game / "world_model.py"
        canonical_temp = self.store_root / self.game / f".world_model.{self.run_id}.tmp"
        _write_fsynced(canonical_temp, bytes(engine_bytes))
        os.replace(canonical_temp, canonical_engine)
        _fsync_directory(canonical_engine.parent)
        if failpoint == "after_engine_publish":
            raise EvidencePublishInterrupted(failpoint)
        if failpoint == "before_manifest":
            raise EvidencePublishInterrupted(failpoint)

        manifest = self.store_root / self.game / "attempts" / "manifest.jsonl"
        manifest.parent.mkdir(parents=True, exist_ok=True)
        line = canonical_json_bytes(row) + b"\n"
        descriptor = os.open(manifest, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX)
            if os.write(descriptor, line) != len(line):
                raise OSError("short manifest append")
            os.fsync(descriptor)
        finally:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
            os.close(descriptor)

        return PublishedEngineEvidence(
            row=row,
            envelope=envelope,
            prompt_path=final_dir / "prompt.raw",
            transition_path=final_dir / "transitions.jsonl",
            engine_path=final_dir / "engine.py",
            canonical_engine_path=canonical_engine,
            envelope_path=final_dir / "envelope.json",
            manifest_row_path=final_dir / "manifest_row.json",
            manifest_path=manifest,
        )


def produce_engine_evidence(
    *,
    store_root: Path,
    game: str,
    raw_prompt: bytes,
    transitions: Sequence[Any],
    transition_source_kind: str | None,
    environment_receipt: Mapping[str, Any],
    synthesize_engine: Callable[[], bytes | str],
    run_id: str | None = None,
    timestamp: str | None = None,
    source_paths: Mapping[str, Path] | None = None,
    writer: str = "live_engine_producer",
    model: str = "",
    note: str = "",
    failpoint: str | None = None,
) -> PublishedEngineEvidence:
    """Stage sources, invoke synthesis, then publish one complete record."""

    transaction = EngineEvidenceTransaction.begin(
        store_root=store_root,
        game=game,
        raw_prompt=raw_prompt,
        transitions=transitions,
        transition_source_kind=transition_source_kind,
        environment_receipt=environment_receipt,
        run_id=run_id,
        timestamp=timestamp,
        source_paths=source_paths,
    )
    engine = synthesize_engine()
    engine_bytes = engine.encode("utf-8") if isinstance(engine, str) else bytes(engine)
    return transaction.publish(
        engine_bytes,
        writer=writer,
        model=model,
        note=note,
        timestamp=timestamp,
        failpoint=failpoint,
    )


def _check(check: str, expected: Any, observed: Any, passed: bool | None = None) -> dict[str, Any]:
    return {
        "check": check,
        "expected_value": expected,
        "observed_value": observed,
        "passed": expected == observed if passed is None else bool(passed),
    }


def _safe_path(root: Path, value: Any) -> Path | None:
    if not isinstance(value, str) or not value or Path(value).is_absolute():
        return None
    candidate = (root / value).resolve()
    try:
        candidate.relative_to(root.resolve())
    except ValueError:
        return None
    return candidate


def _read_envelope(
    root: Path, row: Mapping[str, Any]
) -> tuple[dict[str, Any] | None, bytes | None]:
    record = row.get("envelope")
    path = _safe_path(root, record.get("path")) if isinstance(record, Mapping) else None
    if path is None:
        return None, None
    try:
        raw = path.read_bytes()
        value = json.loads(raw)
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None, None
    return (value if isinstance(value, dict) else None), raw


def _validate_manifest_row(root: Path, row: Mapping[str, Any]) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    row_hash = compute_manifest_row_hash(row)
    checks.append(_check("manifest_row_hash", row.get("manifest_row_sha256"), row_hash))
    envelope, envelope_raw = _read_envelope(root, row)
    checks.append(_check("envelope_readable", True, envelope is not None))
    if envelope is not None and envelope_raw is not None:
        checks.append(
            _check(
                "envelope_canonical_bytes",
                sha256_bytes(canonical_json_bytes(envelope)),
                sha256_bytes(envelope_raw),
            )
        )
        checks.append(_check("envelope_schema", EVIDENCE_ENVELOPE_SCHEMA, envelope.get("schema")))
        missing = [field for field in REQUIRED_ENVELOPE_FIELDS if field not in envelope]
        checks.append(_check("envelope_required_fields", [], missing))
        envelope_hash = compute_envelope_hash(envelope)
        checks.append(_check("envelope_hash", envelope.get("envelope_sha256"), envelope_hash))
        record = row.get("envelope")
        row_envelope_hash = record.get("sha256") if isinstance(record, Mapping) else None
        checks.append(_check("row_envelope_hash", envelope_hash, row_envelope_hash))
        checks.append(_check("run_id_binding", row.get("run_id"), envelope.get("run_id")))
        checks.append(_check("game_binding", row.get("game"), envelope.get("game")))
        checks.append(
            _check(
                "manifest_envelope_binding",
                row.get("manifest_row_sha256"),
                envelope.get("manifest_row_sha256"),
            )
        )
        checks.append(
            _check(
                "transition_source_kind",
                LIVE_TRANSITION_SOURCE_KIND,
                envelope.get("transition_source_kind"),
            )
        )
        for path_field, hash_field in _HASH_FIELDS.items():
            source = _safe_path(root, envelope.get(path_field))
            checks.append(_check(f"{path_field}_safe", True, source is not None))
            try:
                observed_hash = sha256_bytes(source.read_bytes()) if source is not None else None
            except OSError:
                observed_hash = None
            checks.append(_check(hash_field, envelope.get(hash_field), observed_hash))

        transition_path = _safe_path(root, envelope.get("transition_jsonl_path"))
        transition_rows: list[Any] = []
        if transition_path is not None:
            try:
                transition_rows = [
                    json.loads(line) for line in transition_path.read_bytes().splitlines()
                ]
            except (OSError, UnicodeError, json.JSONDecodeError):
                transition_rows = []
        checks.append(
            _check("transition_count", envelope.get("transition_count"), len(transition_rows))
        )
        indices = [value.get("index") for value in transition_rows if isinstance(value, Mapping)]
        checks.append(_check("transition_order", list(range(len(transition_rows))), indices))
    failed = [
        {
            "failed_check": value["check"],
            "expected_value": value["expected_value"],
            "observed_value": value["observed_value"],
        }
        for value in checks
        if value["passed"] is not True
    ]
    return {
        "classification": "evidence_envelope",
        "run_id": row.get("run_id"),
        "eligible": not failed,
        "checks": checks,
        "rejection_reasons": failed,
        "row": dict(row),
    }


def read_evidence_manifest(store_root: Path, game: str) -> list[dict[str, Any]]:
    """Read new and legacy attempt rows, then fail repeated run IDs closed."""

    root = Path(store_root)
    manifest = root / str(game) / "attempts" / "manifest.jsonl"
    try:
        lines = manifest.read_bytes().splitlines()
    except OSError:
        return []
    output: list[dict[str, Any]] = []
    for index, line in enumerate(lines):
        try:
            value = json.loads(line)
        except (UnicodeError, json.JSONDecodeError):
            value = None
        if not isinstance(value, Mapping):
            output.append(
                {
                    "classification": "malformed",
                    "run_id": None,
                    "eligible": False,
                    "checks": [],
                    "rejection_reasons": [
                        {
                            "failed_check": "manifest_row_json",
                            "expected_value": "json_object",
                            "observed_value": f"line_{index}",
                        }
                    ],
                    "row": None,
                }
            )
        elif value.get("schema") != EVIDENCE_MANIFEST_SCHEMA:
            output.append(
                {
                    "classification": "legacy",
                    "run_id": value.get("run_id"),
                    "eligible": False,
                    "checks": [],
                    "rejection_reasons": [
                        {
                            "failed_check": "evidence_envelope_schema",
                            "expected_value": EVIDENCE_MANIFEST_SCHEMA,
                            "observed_value": value.get("schema"),
                        }
                    ],
                    "row": dict(value),
                }
            )
        else:
            output.append(_validate_manifest_row(root, value))

    counts = Counter(row.get("run_id") for row in output if row.get("run_id"))
    for validation in output:
        run_id = validation.get("run_id")
        if run_id and counts[run_id] > 1:
            reason = {
                "failed_check": "unique_run_id",
                "expected_value": 1,
                "observed_value": counts[run_id],
            }
            validation["checks"].append({"check": "unique_run_id", **reason, "passed": False})
            validation["rejection_reasons"].append(reason)
            validation["eligible"] = False
    return output
