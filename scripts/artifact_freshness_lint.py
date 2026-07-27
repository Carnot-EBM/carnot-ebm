#!/usr/bin/env python3
"""Refuse a commit that leaves an ANALYSER-PRODUCED artifact stale w.r.t. the code that built it.

THE INCIDENT (2026-07-26). `results/outer_loop_scored_path_lever_ab_llm_on_20260726.json` was
committed at 08:53. Its analyser, `scripts/analyze_scored_path_lever_ab.py`, was then edited and
committed at 10:38 with NO rebuild, and the artifact was not refreshed until 12:34 -- a ~1h56m
window in which the file on disk was not the output of the code on disk. That particular window
happened to change no number (verified by rebuilding and deep-diffing: only `run_date` moved), but
NOBODY COULD HAVE KNOWN THAT without doing the rebuild-and-diff, which is exactly the work a reader
does not do before quoting a figure. A stale artifact's numbers are cited with precisely the same
confidence as a fresh one's, because nothing at read time distinguishes them.

WHY THE HOOK MUST FIRE ON THE ANALYSER SIDE, NOT JUST THE ARTIFACT SIDE. This is the whole design
point. An artifact-only hook -- "when a results/*.json changes, check it" -- would have waved the
real incident straight through, because in the real incident the ARTIFACT was not touched; the
analyser was. So `files:` in .pre-commit-config.yaml matches the analyser, the harness, and the
shared row-schema module as well as the artifacts, and `pass_filenames: false` means the whole
registered set is re-checked whichever side moved.

WHY AN INDEX RATHER THAN A SCAN. `results/` is 6.1 GB across 6300+ JSON files. A cold grep over it
inside a pre-commit hook costs minutes to answer a question about a handful of artifacts. Analysers
that can produce a stale artifact register their output in `ops/analyzer_artifact_index.json`; this
lint reads that one small file. The trade-off is stated plainly: an artifact whose analyser never
registers it is INVISIBLE to this lint. That is a coverage limit, not a pass -- when adding a new
analyser, call its `register_analyzed_artifact` equivalent, or this guard does not cover it.

THE HOLE IN THIS GUARD'S OWN TRIGGER, found by adversarial review 2026-07-26 and fixed the same
day. The `files:` regex above was HAND-MAINTAINED and listed only TWO of the FIVE code dependencies
the registered artifacts actually declare. Tested one at a time against the shipped regex:
`scripts/analyze_*.py` matched and `scripts/arc_scored_path_lever_harness.py` matched, but
`scripts/arc_scored_path_early_stop_sweep.py`, `scripts/arc_leaderboard_eval.py` and
`python/carnot/agentic/arc_competition_agent.py` did NOT. So editing the agent module -- which the
very session that shipped this guard did -- and committing would leave the artifact stale and never
invoke the hook at all. The guard was reachable-around through 3 of its own 5 paths, which is the
exact incident class it was written for, one indirection out.

The fix is to stop hand-maintaining the list. `registered_dependency_paths()` reads the union of
every registered artifact's `provenance.code` + `provenance.rows_sources` entries, and
`hook_files_pattern()` renders that union as the regex. `--check-hook-coverage` (run as part of the
default invocation, so it fires on every commit the hook sees) parses the hook's own `files:` out of
.pre-commit-config.yaml and REFUSES if any registered code dependency falls outside it, printing the
regenerated pattern to paste. A newly-registered dependency can therefore no longer fall silently
outside the trigger: it fails loudly, with the fix in the failure message.

Exit codes: 0 = all registered artifacts fresh (or none registered) AND the hook trigger covers
every registered code dependency; 1 = at least one artifact is STALE, or the trigger has a coverage
gap. An artifact that records no `provenance` block, or whose row-source files have been cleaned up,
is reported but does NOT fail the commit -- those are unknown-freshness, and blocking on "I cannot
check" would train people to pass --no-verify, which is worse than the gap it closes.

Usage:  python3 scripts/artifact_freshness_lint.py [--index PATH]
        python3 scripts/artifact_freshness_lint.py --emit-hook-pattern   # regenerate files: regex
        python3 scripts/artifact_freshness_lint.py --check-hook-coverage # trigger-gap check only
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
DEFAULT_INDEX = REPO / "ops" / "analyzer_artifact_index.json"
PRECOMMIT_CONFIG = REPO / ".pre-commit-config.yaml"
HOOK_ID = "artifact-freshness-lint"

# Trigger paths that are NOT derivable from any artifact's provenance and must therefore be stated:
# the index itself (registering a new artifact must invoke the hook), and the artifact tree (a
# hand-edited artifact must be re-checked even though no code moved).
# `scripts/analyze_.*\.py` is kept as a static pattern too, even though the concrete analysers are
# now derived: a BRAND-NEW analyser edited before it has ever registered an artifact would otherwise
# be outside the trigger, and the whole point of the coverage check is to surface that on the first
# commit rather than after the first stale citation.
STATIC_TRIGGER_PATTERNS = (
    r"scripts/analyze_.*\.py",
    r"ops/analyzer_artifact_index\.json",
    r"results/.*\.json",
)


def _sha256(path: Path) -> str | None:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return None


def _rows_source_entries(prov: dict) -> list[dict[str, Any]]:
    """Every {path, sha256} entry under `rows_sources`, for BOTH shapes the corpus uses.

    FIELD-SHAPE ROBUSTNESS (2026-07-27). `rows_sources` is a dict-of-named-groups in most artifacts
    and a FLAT LIST in others. Two call sites here did `.values()` unconditionally, so the list shape
    raised AttributeError and took the entire lint down -- blocking every commit while reporting
    nothing about staleness, which is strictly worse than a miss. Same bug class as the QA-Layer
    Authenticity Discipline's origin incident: a checker assuming one shape of a field the project's
    own conventions allow to vary.
    """
    rs = prov.get("rows_sources")
    groups: list[Any] = []
    if isinstance(rs, dict):
        groups = list(rs.values())
    elif isinstance(rs, list):
        groups = [rs]
    out: list[dict[str, Any]] = []
    for group in groups:
        if not isinstance(group, list):
            continue
        out.extend(e for e in group if isinstance(e, dict))
    return out


def check_artifact(artifact: Path) -> tuple[str, list[str], str | None]:
    """Return (status, detail_lines, rebuild_command).

    status is one of `fresh`, `stale`, `no_provenance`, `unverifiable`, `unreadable`.
    """
    try:
        d = json.loads(artifact.read_text())
    except Exception as exc:
        return ("unreadable", [f"{type(exc).__name__}: {exc}"], None)
    prov = d.get("provenance")
    if not isinstance(prov, dict) or not prov:
        return ("no_provenance", [], None)
    recorded: list[dict[str, Any]] = list(prov.get("code") or [])
    recorded.extend(_rows_source_entries(prov))
    if not recorded:
        return ("no_provenance", [], None)
    drift, unreadable = [], []
    for entry in recorded:
        p = Path(str(entry.get("path", "")))
        now = _sha256(p)
        if now is None:
            unreadable.append(str(p))
        elif now != entry.get("sha256"):
            drift.append(str(p))
    cmd = prov.get("rebuild_command")
    if isinstance(cmd, str):
        cmd = cmd.replace("<this file>", str(artifact))
    if drift:
        return ("stale", [f"drifted: {p}" for p in drift], cmd)
    if unreadable:
        return ("unverifiable", [f"unreadable input: {p}" for p in unreadable], cmd)
    return ("fresh", [f"{len(recorded)} dependencies verified"], cmd)


def _repo_relative(raw: str) -> str | None:
    """Provenance records ABSOLUTE paths. pre-commit matches `files:` against REPO-RELATIVE paths.

    That mismatch is why a naive "does the provenance path match the regex" check reports nothing
    useful, and it is worth stating: a row source under /tmp (a scratchpad input) is not a tracked
    file at all and can never be a commit trigger, so it is dropped rather than reported as a gap.
    """
    if not raw:
        return None
    p = Path(raw)
    try:
        return str(p.resolve().relative_to(REPO)) if p.is_absolute() else str(p)
    except ValueError:
        return None


def registered_dependency_paths(index_path: Path = DEFAULT_INDEX) -> dict[str, list[str]]:
    """The union of every registered artifact's declared dependencies, repo-relative.

    Returns {"code": [...], "rows": [...]} -- split because they carry different obligations. A CODE
    dependency MUST be a commit trigger (editing it invalidates the artifact and nothing else would
    notice). A ROW source is data: it is fingerprinted and checked, and the ones that live in
    results/ are already covered by the static artifact-tree pattern.
    """
    out: dict[str, list[str]] = {"code": [], "rows": []}
    try:
        index = json.loads(index_path.read_text())
    except Exception:
        return out
    if not isinstance(index, dict):
        return out
    for rel in sorted(index):
        try:
            d = json.loads((REPO / rel).read_text())
        except Exception:
            continue
        prov = d.get("provenance")
        if not isinstance(prov, dict):
            continue
        for entry in prov.get("code") or []:
            r = _repo_relative(str(entry.get("path", "")))
            if r and r not in out["code"]:
                out["code"].append(r)
        # FIELD-SHAPE ROBUSTNESS (2026-07-27). `rows_sources` is a dict-of-named-groups in most
        # artifacts but a FLAT LIST of {path, sha256} in others -- both shapes are in the corpus, and
        # `.values()` on the list shape raised AttributeError and took the whole lint down with it.
        # A crash in the freshness layer is worse than a miss: it blocks every commit while telling
        # the author nothing about staleness. Both shapes are now walked, and an unexpected shape is
        # skipped rather than fatal. (This is the same field-shape-assumption bug class the
        # QA-Layer Authenticity Discipline was written for -- the checker assumed one shape of a
        # field the project's own conventions allow to vary.)
        for entry in _rows_source_entries(prov):
            r = _repo_relative(str(entry.get("path", "")))
            if r and r not in out["rows"]:
                out["rows"].append(r)
        # The analyser named in the index is a code dependency even if a (buggy) artifact forgot to
        # fingerprint it -- belt and braces, because the index entry is the thing that always exists.
        an = _repo_relative(str((index.get(rel) or {}).get("analyzer") or ""))
        if an and an not in out["code"]:
            out["code"].append(an)
    out["code"].sort()
    out["rows"].sort()
    return out


def hook_files_pattern(index_path: Path = DEFAULT_INDEX) -> str:
    """Render the `files:` regex for the pre-commit hook from the registered dependency union."""
    code = registered_dependency_paths(index_path)["code"]
    alts = [re.escape(p) for p in code] + list(STATIC_TRIGGER_PATTERNS)
    return "^(" + "|".join(alts) + ")$"


def hook_files_regex_from_config(config: Path = PRECOMMIT_CONFIG) -> str | None:
    """Pull THIS hook's own `files:` out of .pre-commit-config.yaml.

    Deliberately a line scan rather than a YAML parse: pre-commit's config is read by pre-commit, and
    importing a YAML library into a hook that must run before the environment is guaranteed is a
    dependency this check does not need.
    """
    try:
        lines = config.read_text().splitlines()
    except OSError:
        return None
    seen_hook = False
    for line in lines:
        s = line.strip()
        if s.startswith("- id:"):
            seen_hook = s.split("- id:", 1)[1].strip() == HOOK_ID
        elif seen_hook and s.startswith("files:"):
            v = s.split("files:", 1)[1].strip()
            if len(v) >= 2 and v[0] in "'\"" and v[-1] == v[0]:
                v = v[1:-1]
            return v
    return None


def check_hook_coverage(index_path: Path = DEFAULT_INDEX) -> tuple[bool, list[str], str]:
    """Does the hook's own trigger fire when a registered CODE dependency is edited?

    Returns (ok, uncovered_paths, regenerated_pattern). This is the check that would have caught the
    2026-07-26 review finding: 3 of 5 registered code dependencies fell outside the shipped regex.
    """
    deps = registered_dependency_paths(index_path)["code"]
    generated = hook_files_pattern(index_path)
    configured = hook_files_regex_from_config()
    if configured is None:
        # Cannot locate the hook: report rather than block. A missing hook is a separate problem and
        # failing here would just make this lint unusable outside a repo checkout.
        return (True, [], generated)
    # `re.search`, not `re.match`: that is what pre-commit's own include filter uses
    # (`filter_by_include_exclude` -> `re.search(include, filename)`). Identical for the anchored
    # pattern we generate, but the check must mirror the real matcher, not a close relative of it.
    rx = re.compile(configured)
    uncovered = [p for p in deps if not rx.search(p)]
    return (not uncovered, uncovered, generated)


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", default=str(DEFAULT_INDEX))
    ap.add_argument(
        "--emit-hook-pattern",
        action="store_true",
        help="print the files: regex generated from the registered dependency union and exit",
    )
    ap.add_argument(
        "--check-hook-coverage",
        action="store_true",
        help="only check that the hook trigger covers every registered code dependency",
    )
    a = ap.parse_args(argv)
    index_path = Path(a.index)

    if a.emit_hook_pattern:
        print(hook_files_pattern(index_path))
        return 0

    # The coverage check is a statement about THIS REPO's real registered set versus THIS REPO's real
    # hook config, so it runs only for the default index. A test or ad-hoc invocation pointing at
    # some other index is asking a different question, and answering it against the repo's hook
    # regex would produce a meaningless failure.
    ok, uncovered, generated = (
        check_hook_coverage(index_path)
        if index_path.resolve() == DEFAULT_INDEX.resolve()
        else (True, [], "")
    )
    if not ok:
        print("artifact-freshness-lint: REFUSING THE COMMIT -- THE TRIGGER HAS A COVERAGE GAP.")
        print(
            "  These registered CODE dependencies do NOT match this hook's own `files:` regex, so "
            "editing one and committing would leave its artifact stale WITHOUT running this check:"
        )
        for p in uncovered:
            print(f"    {p}")
        print("\n  Replace the hook's files: in .pre-commit-config.yaml with:")
        print(f"    files: '{generated}'")
        print(
            "  (regenerate any time with: "
            "python3 scripts/artifact_freshness_lint.py --emit-hook-pattern)"
        )
        return 1
    if a.check_hook_coverage:
        print("artifact-freshness-lint: hook trigger covers every registered code dependency.")
        return 0
    if not index_path.exists():
        print(f"artifact-freshness-lint: no index at {index_path} -- nothing registered, passing.")
        return 0
    try:
        index = json.loads(index_path.read_text())
    except Exception as exc:
        print(f"artifact-freshness-lint: index unreadable ({exc}); passing rather than blocking.")
        return 0
    if not isinstance(index, dict) or not index:
        print("artifact-freshness-lint: index empty -- nothing registered, passing.")
        return 0

    stale: list[tuple[str, list[str], str | None]] = []
    for rel in sorted(index):
        artifact = REPO / rel
        if not artifact.exists():
            print(f"  [gone ] {rel} (registered but absent; it will be pruned on the next build)")
            continue
        status, detail, cmd = check_artifact(artifact)
        tag = {"fresh": "fresh", "stale": "STALE", "no_provenance": "unknwn"}.get(
            status, status[:6]
        )
        print(f"  [{tag:6}] {rel}" + (f" -- {detail[0]}" if detail else ""))
        for extra in detail[1:]:
            print(f"           {extra}")
        if status == "stale":
            stale.append((rel, detail, cmd))

    if stale:
        print()
        print("artifact-freshness-lint: REFUSING THE COMMIT.")
        print(
            "  One or more analyser-produced artifacts were NOT built from the code now on disk. "
            "Their numbers are of unknown provenance until rebuilt."
        )
        for rel, detail, cmd in stale:
            print(f"\n  {rel}")
            for line in detail:
                print(f"    {line}")
            if cmd:
                print(f"    rebuild with:\n      {cmd}")
        print(
            "\n  Then DIFF the rebuild against the committed version and report exactly which "
            "numbers moved -- a rebuild that silently changes a published figure is a correction "
            "owed, not a formality."
        )
        return 1
    print("artifact-freshness-lint: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
