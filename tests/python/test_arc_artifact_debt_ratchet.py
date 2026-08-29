"""REQ-HARNESS-5935: an artifact committed after the debt baseline must pass its own lint.

WHY (2026-08-28, measured). 444 of 573 ARC candidate artifacts fail
scripts/arc_artifact_lint.py, and the guard was green the whole time: it is a
pre-commit hook, result artifacts are written once and never re-staged, and the
conductor commits with hooks skipped -- so the lint had never inspected the
population it governs. This test is the venue correction: pytest IS the gate the
conductor runs, so enforcement lives here. The standing 444 are frozen behind a
baseline commit and await per-class repair (ops/known-issues.md 2026-08-28); this
test only stops the debt GROWING.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
BASELINE = REPO / "ops" / "arc_artifact_lint_debt_baseline.json"

sys.path.insert(0, str(REPO / "scripts"))

from arc_artifact_lint import lint_paths  # noqa: E402


def _git(*args: str) -> str | None:
    """git stdout, or None when git could not answer. None must fail closed."""
    try:
        done = subprocess.run(
            ["git", "-C", str(REPO), *args], capture_output=True, text=True, check=False
        )
    except OSError:
        return None
    return done.stdout if done.returncode == 0 else None


def artifacts_added_since(baseline_commit: str) -> list[Path] | None:
    """Result artifacts first added after the baseline commit, or None if unknowable.

    SCENARIO-HARNESS-5935-GIT-UNAVAILABLE-FAILS-CLOSED: an empty answer is only
    trustworthy when git actually answered. A checkout where the baseline commit is
    unknown (shallow clone, corrupted history) returns None, and the caller fails
    with a message instead of passing on an unmeasured population.
    """
    if _git("cat-file", "-e", f"{baseline_commit}^{{commit}}") is None:
        return None
    out = _git(
        "log",
        "--diff-filter=A",
        "--name-only",
        "--format=",
        f"{baseline_commit}..HEAD",
        "--",
        "results",
    )
    if out is None:
        return None
    added = sorted(
        {
            line.strip()
            for line in out.splitlines()
            if line.strip().endswith(".json") and "/experiment_" in f"/{line.strip()}"
        }
    )
    # A path added then deleted again is git history, not a population member.
    return [REPO / rel for rel in added if (REPO / rel).exists()]


def test_req_harness_5935_baseline_is_well_formed() -> None:
    payload = json.loads(BASELINE.read_text(encoding="utf-8"))
    assert len(payload.get("baseline_commit", "")) == 40, "full sha required"
    for key in ("what_this_is", "how_to_shrink", "known_failing_count"):
        assert key in payload, f"honesty metadata dropped: {key}"


def test_req_harness_5935_no_new_artifact_fails_its_own_lint() -> None:
    """SCENARIO-HARNESS-5935-NEW-ARTIFACT-MUST-PASS, enforced on the live repo."""
    payload = json.loads(BASELINE.read_text(encoding="utf-8"))
    candidates = artifacts_added_since(payload["baseline_commit"])
    assert candidates is not None, (
        "git could not answer which artifacts were added after the baseline commit "
        f"{payload['baseline_commit'][:12]}; an unmeasured population must never read "
        "as clean (SCENARIO-HARNESS-5935-GIT-UNAVAILABLE-FAILS-CLOSED)"
    )
    issues = [i for i in lint_paths(candidates) if i.severity == "error"]
    assert not issues, (
        "artifact(s) committed after the debt baseline fail arc_artifact_lint; fix the "
        "WRITER and regenerate -- never move the baseline forward past them:\n"
        + "\n".join(f"  {i.path}: {i.kind}: {i.detail[:120]}" for i in issues)
    )


def test_req_harness_5935_a_failing_new_artifact_would_be_named(tmp_path: Path) -> None:
    """The rule bites: a synthetic post-baseline artifact with an illegal substrate
    draws an error-severity issue from the same lint call the live test uses.
    Deleting the enforcement (or the lint's candidate match) turns this RED."""
    bad = tmp_path / "experiment_999999_arc_solve.json"
    bad.write_text(
        json.dumps(
            {
                "experiment": 999999,
                "schema": "carnot.arc.solve.test",
                "inference_substrate": "definitely_not_a_legal_substrate",
                "honest_verdict": "complete: synthetic",
                "duration_s": 1.0,
                "offline_reproduced": True,
                "reproduced_levels": 1,
            }
        ),
        encoding="utf-8",
    )
    issues = [i for i in lint_paths([bad]) if i.severity == "error"]
    assert any(i.kind == "INVALID_INFERENCE_SUBSTRATE" for i in issues), (
        "the lint no longer flags an illegal substrate on a new artifact; the ratchet is decorative"
    )


def test_req_harness_5935_unknown_baseline_commit_fails_closed() -> None:
    """SCENARIO-HARNESS-5935-GIT-UNAVAILABLE-FAILS-CLOSED, pinned.

    A baseline commit git does not know (shallow clone, corrupted file) must yield
    None -- never an empty candidate list, which would read as a clean population.
    Deleting the cat-file existence guard turns this RED.
    """
    assert artifacts_added_since("0" * 40) is None
