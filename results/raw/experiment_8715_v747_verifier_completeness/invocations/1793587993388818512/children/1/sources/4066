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
    "SOURCE_AUTHORING",
    "SOURCE_COMMIT",
    "SOURCE_LIVE_EVIDENCE",
    "SOURCE_OUTSIDE_CHECKOUT",
    "ReceiptResolutionError",
    "artifact_commit",
    "receipt_bytes",
    "receipt_exists",
    "receipt_sha256",
    "receipt_source",
    "repository_root",
]

# Where a receipt was actually read from. Three of these mean the working tree,
# and they are different things: one is deliberate policy, one is an experiment
# authoring its first artifact, one is a fixture with no history. A git failure
# is none of them -- it raises.
SOURCE_COMMIT = "commit"
SOURCE_LIVE_EVIDENCE = "live_evidence"
SOURCE_AUTHORING = "authoring_never_committed"
SOURCE_OUTSIDE_CHECKOUT = "outside_checkout"

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


def _git_log_commit(root: Path, rel: str, *extra: str) -> str:
    """Return one commit id for `rel`, or "" when git reports no such commit.

    Fail-closed on a git error. Answering "" for a broken git would send every
    receipt to the working tree while still calling itself a commit receipt --
    a guard that is trusted and silent, which is worse than no guard.
    """

    proc = _run_git(root, "log", "-1", "--format=%H", *extra, "--", rel)
    if proc.returncode != 0:
        detail = proc.stderr.decode("utf-8", "replace").strip()
        raise ReceiptResolutionError(f"git log failed for {rel} in {root}: {detail}")
    return proc.stdout.decode("utf-8", "replace").strip()


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

    if _run_git(root_path, "rev-parse", "--verify", "HEAD").returncode != 0:
        # A checkout with no commits at all. There is no history to resolve
        # against, so this is authoring, not a broken repository.
        _COMMIT_CACHE[key] = None
        return None

    commit = _git_log_commit(root_path, rel, "--diff-filter=A")
    if not commit:
        # A renamed or copied artifact has no recorded add; fall back to the
        # newest commit that touched it so such artifacts still resolve.
        commit = _git_log_commit(root_path, rel)
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
) -> tuple[Path, Path | None, str | None, str | None, str]:
    """Return (absolute path, checkout, repo-relative path, commit, source).

    Four results send the receipt to the working tree rather than a commit, and
    they are NOT the same thing. `source` names which one, so the caller and the
    tests can tell a deliberate live read from an unresolved one. Nothing here
    falls back after a git failure; `artifact_commit` raises on that.
    """

    absolute = Path(path).resolve()
    root_path = Path(root).resolve() if root is not None else repository_root(absolute)
    if root_path is None:
        # No checkout anywhere above this path. Fail-open by necessity: there is
        # no history that could answer, so the disk is the only source.
        return absolute, None, None, None, SOURCE_OUTSIDE_CHECKOUT
    try:
        rel = absolute.relative_to(root_path).as_posix()
    except ValueError:
        # A fixture built under tmp_path. Same reason as above.
        return absolute, None, None, None, SOURCE_OUTSIDE_CHECKOUT

    if commit is None:
        commit = artifact_commit(root_path, artifact_relative_path)
        if commit is None:
            # The artifact has never landed. Fail-open ON PURPOSE: this is the
            # first write of a new experiment, and refusing it would make an
            # experiment unable to author its own receipt. It is safe because a
            # git failure raises instead of arriving here.
            return absolute, root_path, rel, None, SOURCE_AUTHORING
    if not _COMMIT_RE.match(commit):
        raise ReceiptResolutionError(f"malformed artifact commit id: {commit!r}")
    if _run_git(root_path, "cat-file", "-e", f"{commit}^{{commit}}").returncode != 0:
        raise ReceiptResolutionError(f"artifact commit {commit} is not in this repository")
    if not allow_evidence_pin and rel.startswith(_EVIDENCE_PREFIXES):
        # DELIBERATELY LIVE, not an unresolved receipt. The commit resolved
        # fine; pinning it is what we refuse, so a later gate stamp or a dropped
        # corrigendum still moves the receipt. Raising instead would force every
        # real caller to pass allow_evidence_pin=True, which is the one outcome
        # this rule exists to prevent.
        return absolute, root_path, rel, None, SOURCE_LIVE_EVIDENCE
    return absolute, root_path, rel, commit, SOURCE_COMMIT


def receipt_source(
    path: Path | str,
    *,
    artifact_relative_path: Path | str,
    root: Path | str | None = None,
    commit: str | None = None,
    allow_evidence_pin: bool = False,
) -> str:
    """Name where a receipt for `path` would be read from.

    Exposed so the four working-tree cases are asserted by tests rather than
    only described in a comment.
    """

    return _resolve(path, artifact_relative_path, root, commit, allow_evidence_pin)[4]


def receipt_bytes(
    path: Path | str,
    *,
    artifact_relative_path: Path | str,
    root: Path | str | None = None,
    commit: str | None = None,
    allow_evidence_pin: bool = False,
) -> bytes:
    """Return the bytes of `path` as committed when the artifact landed."""

    absolute, root_path, rel, resolved, _source = _resolve(
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

    absolute, root_path, rel, resolved, _source = _resolve(
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
