#!/usr/bin/env python3
"""Record WHICH gate produced a fabrication-gate determination, and WHEN.

WHY THIS EXISTS (2026-08-25 incident)
-------------------------------------
``flagged_adversarial`` is the fabrication gate's verdict on an artifact. Every
consumer keys off it. ``scripts/conductor_gates.py`` goes further and BLOCKS a
downstream task when its upstream carries the flag ("UPSTREAM IS QUARANTINED").

Until this module, the stamp carried no record of which gate version made the
call. So a determination made months ago, under rules that have since changed,
was indistinguishable from one made by the current gate. A reader could not tell
the two apart at all.

That is not hypothetical. The conductor runs as ONE long-lived ``--loop``
process. Its fabrication-gate pass imported ``adversarial_verify`` once, so it
judged artifacts with the module copy cached at first use. exp6593 was stamped
CRITICAL ``DURATION_TOO_SHORT`` by a 14-hour-stale gate, under a rule its own
commit had already fixed. Commit 82d8219adf fixed the reload, so NEW stamps are
current -- but the history was never re-judged, and nothing on disk says which
of the 565 stamped artifacts were judged by which gate.

The dangerous direction is the one that is easy to miss. A stale stamp that
should CLEAR merely quarantines honest work. A stale stamp that should have
become MORE severe means a fabrication check added after the stamp never ran
against that artifact at all.

WHAT THE VERSION IS
-------------------
A content hash of ``adversarial_verify.py`` would change on every comment edit,
so every artifact would read "stale" forever and the signal would be worthless
-- the check-that-cries-wolf failure CLAUDE.md warns about. So the version is a
SEMANTIC fingerprint: the source is parsed, docstrings are stripped, and the
AST is re-unparsed before hashing. Comment-only and docstring-only edits do not
change it. A change to executable logic does. See
``test_stamp_provenance_stale_gate_6601.py`` for that property under test.

This is deliberately conservative in the safe direction. A refactor that moves
code without changing behaviour still bumps the version, so some artifacts read
"stale" when their determination would not actually change. Re-checking a stamp
that did not need it is cheap; trusting one that silently went stale is what
this module exists to stop.

WHAT THIS MODULE DOES NOT DO
----------------------------
It never re-judges an artifact and never writes ``results/``. It answers one
question -- "was this determination made by the gate now on disk?" -- in O(1)
per artifact, with no gate run. Deciding what to do about a stale determination
is an operator call, because correcting one means writing evidence.

Cross-references:
- CLAUDE.md "Adversarial Artifact Verification + Sample-Size Rigor" -> the gate
- CLAUDE.md "QA-Layer Authenticity Discipline" -> why a guard must not read a
  field with a bare ``.get`` (fields may be principle-wrapped)
- ``scripts/determination_preservation_lint.py`` -> its ``MARKER_PATTERNS``
  already match ``^flagged_adversarial``, so the provenance field this module
  writes inherits drop-protection at commit time
- REQ-VERIFY-6601 in ``openspec/capabilities/verification/spec.md``
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Derived from __file__, never an absolute literal: a fresh clone must fingerprint
# ITS OWN gate, not the operator's checkout (CLAUDE.md rule 4 on write targets).
GATE_SOURCE_PATH = Path(__file__).resolve().parent / "adversarial_verify.py"

DETERMINATION_FIELD = "flagged_adversarial"
PROVENANCE_FIELD = "flagged_adversarial_provenance"

#: Named so a future change of hashing scheme is detectable rather than silent.
FINGERPRINT_ALGO = "ast_normalized_docstring_stripped_sha256_v1"

STATUS_UNSTAMPED = "unstamped"
STATUS_CURRENT = "current"
STATUS_STALE = "stale"
STATUS_UNVERSIONED = "unversioned"

#: Statuses where the determination cannot be trusted as a current judgement.
UNTRUSTWORTHY_STATUSES = (STATUS_STALE, STATUS_UNVERSIONED)

_DOCSTRING_OWNERS = (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)


def semantic_fingerprint(source: str) -> str:
    """Hash the EXECUTABLE content of `source`, ignoring comments and docstrings.

    Comments vanish because `ast.parse` never keeps them. Docstrings are stripped
    explicitly below -- they are prose, and prose edits must not read as a gate
    change or the staleness signal becomes noise.
    """
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if not isinstance(node, _DOCSTRING_OWNERS):
            continue
        body = node.body
        if (
            body
            and isinstance(body[0], ast.Expr)
            and isinstance(body[0].value, ast.Constant)
            and isinstance(body[0].value.value, str)
        ):
            # A body must never become empty, so substitute `pass`.
            node.body = body[1:] or [ast.Pass()]
    normalized = ast.unparse(ast.fix_missing_locations(tree))
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


# Keyed on (mtime_ns, size) rather than the path alone. An edit to the gate must
# invalidate this -- caching by path is the exact bug that caused the incident.
_VERSION_CACHE: dict[tuple[str, int, int], str] = {}


def current_gate_version(gate_path: Path | None = None) -> str:
    """Semantic fingerprint of the gate source now ON DISK."""
    path = Path(gate_path) if gate_path is not None else GATE_SOURCE_PATH
    stat = path.stat()
    key = (str(path), stat.st_mtime_ns, stat.st_size)
    cached = _VERSION_CACHE.get(key)
    if cached is None:
        cached = semantic_fingerprint(path.read_text(encoding="utf-8"))
        _VERSION_CACHE[key] = cached
    return cached


def make_provenance(stamper: str, *, gate_path: Path | None = None) -> dict[str, Any]:
    """Build the provenance block to write beside a fresh determination.

    `stamper` names the code path that made the call, so a later reader can tell
    a completion-gate stamp from a retroactive backfill.
    """
    return {
        "gate_version": current_gate_version(gate_path),
        "gate_version_algo": FINGERPRINT_ALGO,
        "stamped_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "stamper": stamper,
    }


def _unwrap(value: Any) -> Any:
    """Read through a `{"principle": ..., "value": ...}` wrapper.

    Required by the QA-layer discipline: any artifact field may be written in
    that shape, and a bare `.get` on one silently reads a dict as its value.
    """
    if isinstance(value, dict) and "value" in value and "principle" in value:
        return value["value"]
    return value


def stamp_status(artifact: dict[str, Any], *, gate_path: Path | None = None) -> str:
    """Classify an artifact's determination against the gate now on disk.

    Returns one of `unstamped` / `current` / `stale` / `unversioned`. Runs no
    gate check, so this is O(1) per artifact over the whole corpus.
    """
    if not isinstance(artifact, dict):
        return STATUS_UNSTAMPED
    if not _unwrap(artifact.get(DETERMINATION_FIELD)):
        return STATUS_UNSTAMPED
    provenance = _unwrap(artifact.get(PROVENANCE_FIELD))
    if not isinstance(provenance, dict):
        return STATUS_UNVERSIONED
    recorded = _unwrap(provenance.get("gate_version"))
    if not isinstance(recorded, str) or not recorded:
        return STATUS_UNVERSIONED
    return STATUS_CURRENT if recorded == current_gate_version(gate_path) else STATUS_STALE


def describe_stamp_status(artifact: dict[str, Any], *, gate_path: Path | None = None) -> str:
    """One sentence a reader-side gate can append when it relies on a stamp.

    Empty string when the stamp is current or absent, so a caller can append
    unconditionally and stay silent in the ordinary case.
    """
    status = stamp_status(artifact, gate_path=gate_path)
    if status not in UNTRUSTWORTHY_STATUSES:
        return ""
    if status == STATUS_UNVERSIONED:
        return (
            " STAMP PROVENANCE MISSING: this determination records no gate version, so it "
            "cannot be told apart from one made under rules that have since changed. Treat "
            "the quarantine as unverified at this gate version, not as a fresh judgement."
        )
    provenance = _unwrap(artifact.get(PROVENANCE_FIELD)) or {}
    when = _unwrap(provenance.get("stamped_at")) if isinstance(provenance, dict) else None
    return (
        " STAMP IS STALE: this determination was made by an OLDER gate version"
        + (f" at {when}" if when else "")
        + ". The checks have changed since. It may no longer hold, and any check added "
        "after it never ran against this artifact."
    )


def scan(paths: list[Path], *, gate_path: Path | None = None) -> dict[str, list[str]]:
    """Group artifact filenames by stamp status. Read-only."""
    buckets: dict[str, list[str]] = {
        STATUS_UNSTAMPED: [],
        STATUS_CURRENT: [],
        STATUS_STALE: [],
        STATUS_UNVERSIONED: [],
    }
    for path in paths:
        try:
            data = json.loads(Path(path).read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(data, dict):
            continue
        buckets[stamp_status(data, gate_path=gate_path)].append(Path(path).name)
    return buckets


def main(argv: list[str]) -> int:  # pragma: no cover - CLI wrapper
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "results",
    )
    parser.add_argument("--json", action="store_true", help="Emit the full grouping as JSON.")
    args = parser.parse_args(argv)

    buckets = scan(sorted(Path(args.results_dir).glob("*.json")))
    if args.json:
        print(json.dumps(buckets, indent=2))
        return 0
    print(f"gate_version = {current_gate_version()[:16]} ({FINGERPRINT_ALGO})")
    for status in (STATUS_CURRENT, STATUS_STALE, STATUS_UNVERSIONED, STATUS_UNSTAMPED):
        print(f"  {status:12s} {len(buckets[status])}")
    untrusted = sum(len(buckets[s]) for s in UNTRUSTWORTHY_STATUSES)
    if untrusted:
        print(
            f"\n{untrusted} determination(s) cannot be trusted as a current judgement. "
            "Correcting them writes results/, which is evidence -- that is an operator call."
        )
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main(sys.argv[1:]))
