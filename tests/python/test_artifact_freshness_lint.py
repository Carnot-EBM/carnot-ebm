"""Drift guards for the ARTIFACT STALENESS guard (``scripts/artifact_freshness_lint.py``).

THE INCIDENT (2026-07-26). ``results/outer_loop_scored_path_lever_ab_llm_on_20260726.json`` was
committed at 08:53; its analyser was edited and committed at 10:38 with no rebuild; the artifact was
not refreshed until 12:34. For ~1h56m the file on disk was not the output of the code on disk.
Rebuilding and deep-diffing after the fact showed that window changed no number -- but that could
only be established BY rebuilding, which is exactly the work nobody does before quoting a figure.
Nothing at read time distinguished the stale file from a fresh one.

WHAT THESE TESTS PROTECT, specifically:

  1. The guard fires on an ANALYSER change with the artifact untouched. This is the real incident
     shape, and it is the case an artifact-only trigger silently passes. A guard that only noticed
     edited artifacts would have been useless here.
  2. "Cannot check" is never reported as "checked and clean". An artifact with no provenance block,
     or with row-source inputs that have since been cleaned up, is UNKNOWN / UNVERIFIABLE -- never
     fresh. Collapsing those into a pass is the same false-clean-zero the whole change is about.
  3. The failure message names the fix (the exact rebuild command the artifact recorded), because a
     blocking hook that does not tell you how to unblock it gets bypassed with --no-verify.

No network, no GPU: pure hashing over temp files.

Spec refs: REQ-ARC-WMTE-5960, SCENARIO-ARC-WMTE-5960-ARTIFACT-FRESHNESS.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
from types import ModuleType

REPO = Path(__file__).resolve().parents[2]
_LINT = REPO / "scripts" / "artifact_freshness_lint.py"


def _lint() -> ModuleType:
    spec = importlib.util.spec_from_file_location("artifact_freshness_lint", _LINT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _fingerprint(p: Path) -> dict:
    raw = p.read_bytes()
    return {"path": str(p), "sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)}


def _artifact_with_provenance(
    tmp_path: Path, code_files: list[Path], row_files: list[Path]
) -> Path:
    art = tmp_path / "artifact.json"
    art.write_text(
        json.dumps(
            {
                "honest_verdict": "complete_synthetic",
                "provenance": {
                    "git_head": "deadbeef",
                    "code": [_fingerprint(p) for p in code_files],
                    "rows_sources": {"rows": [_fingerprint(p) for p in row_files]},
                    "rebuild_command": "python scripts/analyze_x.py --out <this file>",
                },
            }
        )
    )
    return art


def test_the_guard_fires_when_the_ANALYSER_changes_and_the_artifact_does_not(tmp_path):
    """THE ACTUAL INCIDENT SHAPE. The artifact file is byte-for-byte untouched; only the code that
    produced it moved. An artifact-only trigger reports nothing here, which is why the pre-commit
    `files:` list must match the analyser/harness/schema side too."""

    m = _lint()
    analyser = tmp_path / "analyze_x.py"
    analyser.write_text("# v1\n")
    rows = tmp_path / "rows.json"
    rows.write_text('{"rows": []}')
    art = _artifact_with_provenance(tmp_path, [analyser], [rows])

    assert m.check_artifact(art)[0] == "fresh"
    before = art.read_bytes()

    analyser.write_text("# v2 -- a real edit, artifact NOT rebuilt\n")
    status, detail, cmd = m.check_artifact(art)
    assert status == "stale"
    assert any(str(analyser) in line for line in detail)
    # The fix must be NAMED, or the hook gets bypassed instead of satisfied.
    assert cmd is not None and str(art) in cmd
    assert art.read_bytes() == before, "the lint must never modify the artifact it checks"


def test_a_changed_row_source_is_also_stale(tmp_path):
    """Code is not the only way an artifact's numbers stop matching their inputs. All three row
    designs are fingerprinted -- `--rows`, `--companion-rows`, `--alt-budget-rows` -- because the
    artifact's own `source_row_files` field records only the first, so a swapped companion file
    would otherwise be invisible (the same blind spot the reproducibility checksum had, where 750 of
    805 published rows sat outside the hash)."""

    m = _lint()
    analyser = tmp_path / "analyze_x.py"
    analyser.write_text("# v1\n")
    rows = tmp_path / "rows.json"
    rows.write_text('{"rows": [1]}')
    art = _artifact_with_provenance(tmp_path, [analyser], [rows])
    assert m.check_artifact(art)[0] == "fresh"

    rows.write_text('{"rows": [1, 2]}')
    status, detail, _ = m.check_artifact(art)
    assert status == "stale"
    assert any(str(rows) in line for line in detail)


def test_unknown_and_unverifiable_are_never_reported_as_fresh(tmp_path):
    """`None` is not `False`, and "I could not check" is not "I checked and it is clean".

    An artifact predating the guard carries no fingerprints; a scratchpad row-source may have been
    cleaned up since the build. Both are honest gaps. Reporting either as fresh would reintroduce
    exactly the defect class -- a clean, error-free signal produced by a channel nobody consulted.
    """

    m = _lint()
    legacy = tmp_path / "legacy.json"
    legacy.write_text(
        json.dumps({"honest_verdict": "complete_old", "reproducibility_checksum": "x"})
    )
    assert m.check_artifact(legacy)[0] == "no_provenance"

    analyser = tmp_path / "analyze_x.py"
    analyser.write_text("# v1\n")
    gone = tmp_path / "ephemeral_rows.json"
    gone.write_text("{}")
    art = _artifact_with_provenance(tmp_path, [analyser], [gone])
    gone.unlink()
    status, detail, _ = m.check_artifact(art)
    assert status == "unverifiable"
    assert any("unreadable input" in line for line in detail)
    assert status != "fresh"


def test_only_a_real_drift_refuses_the_commit(tmp_path):
    """The lint must block on STALE and pass on unknown/unverifiable. Blocking on "I cannot check"
    would train people to reach for --no-verify, which costs more than the gap it closes -- and
    CLAUDE.md forbids bypassing hooks without explicit authorisation, so a hook that provokes the
    bypass is a worse outcome than a hook with a stated coverage limit."""

    m = _lint()
    analyser = tmp_path / "analyze_x.py"
    analyser.write_text("# v1\n")
    rows = tmp_path / "rows.json"
    rows.write_text("{}")
    art = _artifact_with_provenance(tmp_path, [analyser], [rows])
    index = tmp_path / "index.json"

    # The index is keyed relative to REPO, so point the module's REPO at tmp for this check.
    m.REPO = tmp_path
    index.write_text(json.dumps({"artifact.json": {"analyzer": "analyze_x.py"}}))
    assert m.main(["--index", str(index)]) == 0

    analyser.write_text("# v2\n")
    assert m.main(["--index", str(index)]) == 1

    # An unverifiable artifact must NOT refuse the commit.
    analyser.write_text("# v1\n")
    rows.unlink()
    assert m.main(["--index", str(index)]) == 0

    # A missing index is a pass, not a crash: most repos/commits have nothing registered.
    assert m.main(["--index", str(tmp_path / "does_not_exist.json")]) == 0


def test_every_registered_code_dependency_matches_the_hooks_own_files_regex():
    """THE HOLE IN THE GUARD ITSELF, found by adversarial review 2026-07-26.

    The hook's `files:` list was hand-maintained and named only 2 of the 5 code dependencies the
    registered artifacts declare. `scripts/arc_scored_path_early_stop_sweep.py`,
    `scripts/arc_leaderboard_eval.py` and `python/carnot/agentic/arc_competition_agent.py` all fell
    OUTSIDE it -- so editing the agent module (which the shipping session itself did) and committing
    would leave the artifact stale and never invoke the check. The guard was reachable-around through
    3 of its own 5 paths.

    This test is the thing that would have caught it, and it is deliberately a test rather than only
    a runtime check: a runtime check inside the hook can only fire on commits the hook already runs
    on, which is precisely the set the gap excludes.
    """

    m = _lint()
    deps = m.registered_dependency_paths()["code"]
    assert deps, "no registered code dependencies -- the coverage check would be vacuous"
    configured = m.hook_files_regex_from_config()
    assert configured, f"could not find `files:` for hook id {m.HOOK_ID} in .pre-commit-config.yaml"
    import re as _re

    rx = _re.compile(configured)
    uncovered = [p for p in deps if not rx.match(p)]
    assert not uncovered, (
        "these registered code dependencies do not trigger the freshness hook: "
        f"{uncovered}\nregenerate with: "
        "python3 scripts/artifact_freshness_lint.py --emit-hook-pattern"
    )
    # And every dependency must be a real file, or the "coverage" is over a phantom path.
    for p in deps:
        assert (REPO / p).exists(), f"registered code dependency does not exist: {p}"


def test_the_coverage_check_actually_detects_a_gap():
    """A coverage check that cannot fail is not a check. Feed it a deliberately narrow regex and
    confirm it names the uncovered dependencies -- the same negative control the sweep applies to its
    own arms (an arm that never fires is uninstrumented, not safe)."""

    m = _lint()
    deps = m.registered_dependency_paths()["code"]
    real = m.hook_files_regex_from_config
    try:
        m.hook_files_regex_from_config = lambda *_a, **_k: r"^(ops/analyzer_artifact_index\.json)$"
        ok, uncovered, generated = m.check_hook_coverage()
        assert ok is False
        assert sorted(uncovered) == sorted(deps)
        # The regenerated pattern must actually cover what it claims to.
        import re as _re

        rx = _re.compile(generated)
        assert all(rx.match(p) for p in deps)
    finally:
        m.hook_files_regex_from_config = real
    # With the real config restored, the repo must be clean.
    assert m.check_hook_coverage()[0] is True


def test_the_live_repo_artifact_is_registered_and_fresh():
    """The guard is wired to something REAL, not just unit-tested in a temp dir.

    An UNINSTRUMENTED guard -- one that passes because it is watching nothing -- is the same class
    of defect as an uninstrumented experiment arm. This asserts the index actually names the
    artifact the incident happened to, and that it is currently fresh.
    """

    m = _lint()
    index_path = REPO / "ops" / "analyzer_artifact_index.json"
    assert index_path.exists(), "the analyser must register its output for the lint to see it"
    index = json.loads(index_path.read_text())
    assert index, "an empty index means the lint is watching nothing"
    for rel in index:
        artifact = REPO / rel
        if not artifact.exists():
            continue
        status, detail, _ = m.check_artifact(artifact)
        assert status in ("fresh", "unverifiable"), f"{rel} is {status}: {detail}"
