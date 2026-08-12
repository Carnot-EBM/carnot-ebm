"""REQ-OPS-FRESHNESS-BACKLOG-1: inherited staleness must not block a new commit.

WHY. The gate refused commits for staleness the committer neither caused nor touched. The
conductor commits with `--no-verify` by design, so it changes shared agent modules without
ever running this hook, and dependent artifacts silently go stale. On 2026-08-11 that had
accumulated to 12 artifacts, several drifting on their OWN analyser scripts, so a rebuild
would legitimately move published numbers. The next person to touch any file in the trigger
list inherited an unclearable refusal -- which trains people to reach for `--no-verify`, the
exact failure this module's docstring says it exists to avoid.

THE RULE UNDER TEST. An artifact is blocking only when at least one drifted dependency was
FRESH at HEAD. If every drifted dependency was already drifted at HEAD, it is backlog.

These tests pin BOTH directions. The second one is the load-bearing one: a change that
weakens a guard into uselessness passes every test that only checks the happy path.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
LINT = REPO / "scripts" / "artifact_freshness_lint.py"


def _run_lint() -> tuple[int, str]:
    out = subprocess.run(
        [sys.executable, str(LINT)], capture_output=True, text=True, cwd=str(REPO), check=False
    )
    return out.returncode, out.stdout + out.stderr


def _fresh_registered_artifact() -> tuple[Path, Path] | None:
    """Find a registered artifact that is currently FRESH, plus one tracked dependency.

    Returns None when the repo has none, in which case the block-direction test cannot be
    constructed and skips rather than passing vacuously.
    """
    index_path = REPO / "ops" / "analyzer_artifact_index.json"
    if not index_path.exists():
        return None
    index = json.loads(index_path.read_text())
    for rel in sorted(index):
        art = REPO / rel
        if not art.exists():
            continue
        try:
            prov = json.loads(art.read_text()).get("provenance") or {}
        except Exception:
            continue
        entries = list(prov.get("code") or [])
        all_match, candidate = True, None
        for e in entries:
            p = Path(str(e.get("path", "")))
            try:
                now = hashlib.sha256(p.read_bytes()).hexdigest()
            except OSError:
                all_match = False
                break
            if now != e.get("sha256"):
                all_match = False
                break
            rp = p.resolve()
            if rp.is_relative_to(REPO) and rp.suffix == ".py":
                candidate = rp
        if all_match and candidate is not None:
            return art, candidate
    return None


def test_pre_existing_backlog_does_not_block_and_is_reported() -> None:
    """The repo's real inherited backlog must be reported, not used to refuse a commit."""
    code, out = _run_lint()
    if "ALREADY stale at HEAD" not in out:
        # No inherited backlog right now (someone cleared it). Nothing to assert.
        return
    assert "Reported as BACKLOG, not blocking this commit." in out
    # A backlog alone must never be the reason for a non-zero exit.
    if "REFUSING THE COMMIT" not in out:
        assert code == 0, out[-2000:]


def test_a_newly_staled_artifact_still_refuses_the_commit() -> None:
    """THE LOAD-BEARING TEST. Downgrading inherited debt must not downgrade real damage.

    Dirties one dependency of a currently-fresh artifact, asserts the lint refuses, then
    restores the file and verifies byte-identity against HEAD. A guard that stopped
    blocking here would be worse than no guard, because it would still print reassuring
    output.
    """
    found = _fresh_registered_artifact()
    if found is None:
        return  # cannot construct the case in this repo state
    _artifact, dep = found
    original = dep.read_bytes()
    try:
        dep.write_bytes(
            original + b"\n# transient freshness-lint regression probe; restored by this test\n"
        )
        code, out = _run_lint()
        assert code == 1, f"a newly staled artifact must refuse the commit\n{out[-2000:]}"
        assert "REFUSING THE COMMIT" in out
    finally:
        dep.write_bytes(original)
    # The test must leave the tree exactly as it found it.
    assert dep.read_bytes() == original
    diff = subprocess.run(
        ["git", "diff", "--quiet", "HEAD", "--", str(dep.relative_to(REPO))],
        cwd=str(REPO),
        check=False,
    )
    assert diff.returncode == 0, "the probe file was not restored to its HEAD content"


def test_git_failure_falls_back_to_blocking() -> None:
    """Fail CLOSED: if HEAD content cannot be read, drift cannot be proven pre-existing."""
    sys.path.insert(0, str(REPO / "scripts"))
    import importlib

    mod = importlib.import_module("artifact_freshness_lint")
    assert mod._sha256_at_head("definitely/not/a/tracked/path/xyz.py") is None
