"""Experiment artifact output paths and atomic writes.

The production default remains the repository's tracked ``results/`` directory.
Tests may set ``CARNOT_EXPERIMENT_ARTIFACT_ROOT`` to redirect writers, but only
to a real temporary directory. That narrow override keeps experiment tests from
rewriting historical evidence while leaving normal experiment runs unchanged.
"""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from carnot.paths import repo_root

ARTIFACT_ROOT_ENV = "CARNOT_EXPERIMENT_ARTIFACT_ROOT"


class ArtifactPathError(ValueError):
    """Raised when an artifact root or artifact path could escape isolation."""


def _canonical_repo_root() -> Path:
    return repo_root(start=__file__).resolve()


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def production_artifact_root(*, root: Path | str | None = None) -> Path:
    """Return the historical production artifact root."""

    base = Path(root).expanduser().resolve() if root is not None else _canonical_repo_root()
    return (base / "results").resolve()


def validate_artifact_output_root(raw: str | Path) -> Path:
    """Validate an explicit test artifact root and return its canonical path."""

    if raw is None or not str(raw).strip():
        raise ArtifactPathError(f"{ARTIFACT_ROOT_ENV} must name a non-empty temporary directory")

    candidate = Path(raw).expanduser()
    try:
        resolved = candidate.resolve()
    except (OSError, RuntimeError, ValueError) as exc:
        raise ArtifactPathError(f"{ARTIFACT_ROOT_ENV} could not be resolved: {raw!r}") from exc

    if not resolved.exists() or not resolved.is_dir():
        raise ArtifactPathError(f"{ARTIFACT_ROOT_ENV} must exist and be a directory: {resolved}")

    repo = _canonical_repo_root()
    prod = production_artifact_root()
    temp_root = Path(tempfile.gettempdir()).resolve()

    if resolved in (repo, prod) or _is_relative_to(resolved, repo):
        raise ArtifactPathError(
            f"{ARTIFACT_ROOT_ENV} may not point inside the repository: {resolved}"
        )
    if resolved == temp_root:
        raise ArtifactPathError(f"{ARTIFACT_ROOT_ENV} may not be the broad temp root: {resolved}")
    if not _is_relative_to(resolved, temp_root):
        raise ArtifactPathError(f"{ARTIFACT_ROOT_ENV} must resolve inside {temp_root}: {resolved}")
    return resolved


def artifact_output_root(
    *,
    root: Path | str | None = None,
    env: Mapping[str, str] | None = None,
    allow_override: bool = True,
) -> Path:
    """Return the root where experiment artifacts should be written."""

    source = os.environ if env is None else env
    override = source.get(ARTIFACT_ROOT_ENV) if allow_override else None
    if override is not None:
        return validate_artifact_output_root(override)
    return production_artifact_root(root=root)


def _relative_artifact_path(
    raw_path: str | Path,
    *,
    base: Path,
    production_root: Path,
    allow_external_absolute: bool,
    override_active: bool,
) -> Path:
    raw_text = str(raw_path)
    if not raw_text.strip():
        raise ArtifactPathError("artifact path must be non-empty")

    path = Path(raw_path).expanduser()
    if path.is_absolute():
        resolved = path.resolve(strict=False)
        if _is_relative_to(resolved, base):
            rel = resolved.relative_to(base)
        elif _is_relative_to(resolved, production_root):
            rel = resolved.relative_to(production_root)
        elif allow_external_absolute and not override_active:
            return resolved
        else:
            raise ArtifactPathError(f"absolute artifact path is outside allowed roots: {resolved}")
    else:
        parts = path.parts
        if any(part in ("", ".", "..") for part in parts):
            raise ArtifactPathError(f"artifact path may not contain traversal: {raw_path!r}")
        if parts and parts[0] == "results":
            parts = parts[1:]
        if not parts:
            raise ArtifactPathError(
                f"artifact path must name a file below the artifact root: {raw_path!r}"
            )
        rel = Path(*parts)

    return rel


def resolve_experiment_artifact_path(
    path: str | Path,
    *,
    root: Path | str | None = None,
    ensure_parent: bool = False,
    env: Mapping[str, str] | None = None,
    allow_override: bool = True,
    allow_external_absolute: bool = False,
) -> Path:
    """Resolve a legacy or root-relative artifact path to its write target."""

    source = os.environ if env is None else env
    override_active = allow_override and source.get(ARTIFACT_ROOT_ENV) is not None
    base = artifact_output_root(root=root, env=source, allow_override=allow_override)
    prod = production_artifact_root(root=root)
    rel_or_abs = _relative_artifact_path(
        path,
        base=base,
        production_root=prod,
        allow_external_absolute=allow_external_absolute,
        override_active=override_active,
    )
    target = rel_or_abs if rel_or_abs.is_absolute() else base / rel_or_abs
    resolved = target.resolve(strict=False)
    if not rel_or_abs.is_absolute() and not _is_relative_to(resolved, base):
        raise ArtifactPathError(f"artifact path resolves outside artifact root: {path!r}")
    if ensure_parent:
        resolved.parent.mkdir(parents=True, exist_ok=True)
    return resolved


def is_legacy_results_path(path: object) -> bool:
    """Return True for relative legacy paths spelled as ``results/...``."""

    if isinstance(path, bytes):
        path = path.decode("utf-8", "surrogateescape")
    try:
        raw = os.fspath(path)
    except TypeError:
        return False
    if not isinstance(raw, str) or not raw:
        return False
    candidate = Path(raw).expanduser()
    return not candidate.is_absolute() and bool(candidate.parts) and candidate.parts[0] == "results"


def resolve_legacy_results_write_path(
    path: str | bytes | Path,
    *,
    root: Path | str | None = None,
    ensure_parent: bool = False,
    env: Mapping[str, str] | None = None,
    allow_override: bool = True,
) -> Path:
    """Resolve result-writer paths while preserving non-result path behavior.

    This helper is narrower than ``resolve_experiment_artifact_path``. Legacy
    result destinations are routed through the artifact resolver, but ordinary
    temporary or caller-owned paths keep their historical location.
    """

    raw = path.decode("utf-8", "surrogateescape") if isinstance(path, bytes) else os.fspath(path)
    candidate = Path(raw).expanduser()
    prod = production_artifact_root(root=root)
    if is_legacy_results_path(candidate):
        return resolve_experiment_artifact_path(
            candidate,
            root=root,
            ensure_parent=ensure_parent,
            env=env,
            allow_override=allow_override,
            allow_external_absolute=False,
        )
    if candidate.is_absolute():
        resolved = candidate.resolve(strict=False)
        if _is_relative_to(resolved, prod):
            return resolve_experiment_artifact_path(
                resolved,
                root=root,
                ensure_parent=ensure_parent,
                env=env,
                allow_override=allow_override,
                allow_external_absolute=False,
            )
        if ensure_parent:
            resolved.parent.mkdir(parents=True, exist_ok=True)
        return resolved
    if ensure_parent and candidate.parent != Path("."):
        candidate.parent.mkdir(parents=True, exist_ok=True)
    return candidate


def atomic_write_bytes(
    path: str | Path,
    data: bytes,
    *,
    root: Path | str | None = None,
    env: Mapping[str, str] | None = None,
    allow_override: bool = True,
) -> Path:
    """Atomically replace an experiment artifact with ``data``."""

    target = resolve_experiment_artifact_path(
        path,
        root=root,
        ensure_parent=True,
        env=env,
        allow_override=allow_override,
        allow_external_absolute=True,
    )
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{target.name}.",
        suffix=".tmp",
        dir=target.parent,
    )
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp_path, target)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()
    return target


def atomic_write_text(
    path: str | Path,
    text: str,
    *,
    root: Path | str | None = None,
    env: Mapping[str, str] | None = None,
    allow_override: bool = True,
) -> Path:
    """Atomically replace a text artifact and return the resolved target."""

    return atomic_write_bytes(
        path, text.encode("utf-8"), root=root, env=env, allow_override=allow_override
    )


def atomic_write_json(
    path: str | Path,
    payload: Mapping[str, Any],
    *,
    root: Path | str | None = None,
    env: Mapping[str, str] | None = None,
    allow_override: bool = True,
    indent: int = 2,
    sort_keys: bool = False,
) -> Path:
    """Atomically write a JSON artifact and return the resolved target."""

    text = json.dumps(payload, indent=indent, sort_keys=sort_keys) + "\n"
    return atomic_write_text(path, text, root=root, env=env, allow_override=allow_override)
