"""Commit-resolved provenance receipts for experiment artifacts.

A receipt records the sha256 of a file the experiment declared it read. The old
rule hashed the working-tree copy, so a later experiment appending to the same
shared spec turned every earlier receipt red. Spec-first discipline guarantees
that churn, so the receipt was red for a reason that was never a provenance
break, and a guard that is always red is read by nobody.

The rule here is different: hash the file as it stood at the commit where the
artifact landed. That is stable against unrelated later edits and still fails
when the declared dependency set, the commit, or the recorded bytes disagree.

Pinning has a cost, so it is not used where the cost is too high. A receipt over
a file under `results/` reads the working tree, because pinning that one would
hide a later gate stamp or a dropped corrigendum, and no other check sees those.
Pinning an evidence file has to be asked for by name.

What this still catches, and what it no longer catches, is written out in
REQ-REPORT-6610 (openspec/capabilities/research-reporting/spec.md).
"""

from __future__ import annotations

import hashlib
import re
import subprocess
from pathlib import Path

__all__ = [
    "ReceiptResolutionError",
    "artifact_commit",
    "receipt_bytes",
    "receipt_exists",
    "receipt_sha256",
    "repository_root",
]

_COMMIT_RE = re.compile(r"\A[0-9a-f]{40}\Z")

# Evidence lives here. Pinning a receipt over one of these to a past commit hides
# exactly the rewrites this project keeps having: a fabrication-gate stamp added
# later, or a hand-written corrigendum dropped. Neither changes any field a
# downstream module reads, so nothing else notices. Measured 2026-08-25; see
# REQ-REPORT-6610-EVIDENCE-LIVE.
_EVIDENCE_PREFIXES = ("results/",)

# Caches keyed by absolute repository path. Each receipt would otherwise fork a
# git process per file, and a milestone builds hundreds of receipts.
_COMMIT_CACHE: dict[tuple[str, str], str | None] = {}
_BLOB_CACHE: dict[tuple[str, str, str], bytes | None] = {}


class ReceiptResolutionError(RuntimeError):
    """A receipt could not be resolved against the artifact's own commit.

    Raised rather than falling back to the working tree. A receipt that
    silently answers from a different source than it claims is the failure this
    module exists to remove.
    """


def repository_root(path: Path | str) -> Path | None:
    """Return the git checkout containing `path`, or None when there is none.

    A worktree stores `.git` as a file, so both shapes count.
    """

    current = Path(path).resolve()
    if current.is_file() or not current.exists():
        current = current.parent
    for candidate in (current, *current.parents):
        if (candidate / ".git").exists():
            return candidate
    return None


def _run_git(root: Path, *args: str) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(
        ["git", "-C", str(root), *args],
        capture_output=True,
        check=False,
    )


def artifact_commit(root: Path | str, artifact_relative_path: Path | str) -> str | None:
    """Return the commit that added the artifact, or None if it never landed.

    The add commit is used rather than the newest commit touching the file
    because the fabrication gate re-stamps landed artifacts. Anchoring on the
    add commit keeps the receipt pinned to the state the experiment ran against.
    """

    root_path = Path(root).resolve()
    rel = Path(artifact_relative_path).as_posix()
    key = (str(root_path), rel)
    if key in _COMMIT_CACHE:
        return _COMMIT_CACHE[key]

    added = _run_git(root_path, "log", "-1", "--format=%H", "--diff-filter=A", "--", rel)
    commit = added.stdout.decode("utf-8", "replace").strip() if added.returncode == 0 else ""
    if not commit:
        # A renamed or copied artifact has no recorded add; fall back to the
        # newest commit that touched it so such artifacts still resolve.
        touched = _run_git(root_path, "log", "-1", "--format=%H", "--", rel)
        commit = (
            touched.stdout.decode("utf-8", "replace").strip() if touched.returncode == 0 else ""
        )
    if commit and not _COMMIT_RE.match(commit):
        raise ReceiptResolutionError(f"git returned a malformed commit id for {rel}: {commit!r}")
    _COMMIT_CACHE[key] = commit or None
    return _COMMIT_CACHE[key]


def _blob_at(root: Path, commit: str, rel: str) -> bytes | None:
    key = (str(root), commit, rel)
    if key not in _BLOB_CACHE:
        proc = _run_git(root, "cat-file", "blob", f"{commit}:{rel}")
        _BLOB_CACHE[key] = proc.stdout if proc.returncode == 0 else None
    return _BLOB_CACHE[key]


def _resolve(
    path: Path | str,
    artifact_relative_path: Path | str,
    root: Path | str | None,
    commit: str | None,
    allow_evidence_pin: bool = False,
) -> tuple[Path, Path | None, str | None, str | None]:
    """Return (absolute path, checkout, repo-relative path, commit).

    A None commit means authoring mode: the artifact has not landed yet, so the
    working tree is the only state that exists.
    """

    absolute = Path(path).resolve()
    root_path = Path(root).resolve() if root is not None else repository_root(absolute)
    if root_path is None:
        return absolute, None, None, None
    try:
        rel = absolute.relative_to(root_path).as_posix()
    except ValueError:
        # A fixture under tmp_path is not part of the checkout, so it has no
        # commit history to resolve against.
        return absolute, None, None, None

    if commit is None:
        commit = artifact_commit(root_path, artifact_relative_path)
        if commit is None:
            return absolute, root_path, rel, None
    if not _COMMIT_RE.match(commit):
        raise ReceiptResolutionError(f"malformed artifact commit id: {commit!r}")
    if _run_git(root_path, "cat-file", "-e", f"{commit}^{{commit}}").returncode != 0:
        raise ReceiptResolutionError(f"artifact commit {commit} is not in this repository")
    if not allow_evidence_pin and rel.startswith(_EVIDENCE_PREFIXES):
        # Answer from the working tree, so a later gate stamp or a dropped
        # corrigendum still moves the receipt. Raising here instead would force
        # every real caller to pass allow_evidence_pin=True, which is the one
        # outcome this rule exists to prevent.
        return absolute, None, None, None
    return absolute, root_path, rel, commit


def receipt_bytes(
    path: Path | str,
    *,
    artifact_relative_path: Path | str,
    root: Path | str | None = None,
    commit: str | None = None,
    allow_evidence_pin: bool = False,
) -> bytes:
    """Return the bytes of `path` as committed when the artifact landed."""

    absolute, root_path, rel, resolved = _resolve(
        path, artifact_relative_path, root, commit, allow_evidence_pin
    )
    if root_path is None or rel is None or resolved is None:
        return absolute.read_bytes()
    data = _blob_at(root_path, resolved, rel)
    if data is None:
        raise ReceiptResolutionError(f"{rel} is not a file at the artifact commit {resolved}")
    return data


def receipt_exists(
    path: Path | str,
    *,
    artifact_relative_path: Path | str,
    root: Path | str | None = None,
    commit: str | None = None,
    allow_evidence_pin: bool = False,
) -> bool:
    """Report whether `path` existed when the artifact landed.

    Callers guard optional dependencies with this instead of `Path.exists`, so
    a file deleted from the working tree later does not silently turn a real
    receipt into a null.
    """

    absolute, root_path, rel, resolved = _resolve(
        path, artifact_relative_path, root, commit, allow_evidence_pin
    )
    if root_path is None or rel is None or resolved is None:
        return absolute.exists()
    return _blob_at(root_path, resolved, rel) is not None


def receipt_sha256(
    path: Path | str,
    *,
    artifact_relative_path: Path | str,
    root: Path | str | None = None,
    commit: str | None = None,
    prefix: str = "sha256:",
    allow_evidence_pin: bool = False,
) -> str:
    """Return the prefixed sha256 receipt for one declared dependency."""

    data = receipt_bytes(
        path,
        artifact_relative_path=artifact_relative_path,
        root=root,
        commit=commit,
        allow_evidence_pin=allow_evidence_pin,
    )
    return prefix + hashlib.sha256(data).hexdigest()
