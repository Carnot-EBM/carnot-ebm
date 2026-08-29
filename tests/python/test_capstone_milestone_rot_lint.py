"""REQ-HARNESS-5945: a capstone must not RAISE when the live roadmap moves past its milestone.

INCIDENT 2026-08-29. A capstone freezes `MILESTONE = "2026.08.580"`. Two of them then read the
LIVE research-roadmap.yaml and raised unless it still carried that milestone, so they were green
only during their own milestone. One built its artifact at MODULE scope, so the raise landed at
COLLECTION and pytest abandoned the entire run: 57,917 tests interrupted by one file, and every
conductor task shelling out to `pytest tests/python` failed with it.

TWO FALSE-POSITIVE CLASSES WERE FOUND WHILE WRITING THIS, and they are why the check is shaped
the way it is. Both would have shipped a lint that cries wolf, which CLAUDE.md rightly calls
worse than the gap it closes:

  1. Matching `payload[...] != MILESTONE` by variable name flagged six CORRECT capstones whose
     validators also call the artifact under validation `payload`. Comparing the ARTIFACT's
     milestone to the constant is the right 23-instance pattern; the name cannot tell them apart.
  2. Matching any comparison flagged eight more that merely RECORD
     `"milestone_matches": roadmap.get("milestone") == MILESTONE` as a field. Reporting the
     mismatch is honest; refusing on it is the rot.

So the rule is narrow on purpose: ONE function must both read the live roadmap AND raise on the
mismatch. Anything less specific misfires.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
_SCRIPT = REPO / "scripts" / "capstone_milestone_rot_lint.py"


def _lint():
    spec = importlib.util.spec_from_file_location("capstone_milestone_rot_lint", _SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "experiment_9999_capstone_vtest.py"
    path.write_text(
        'MILESTONE = "2026.08.580"\nROADMAP_RELATIVE_PATH = Path("research-roadmap.yaml")\n' + body
    )
    return path


ROT = """
def load_roadmap(repo_root):
    payload = yaml.safe_load((repo_root / ROADMAP_RELATIVE_PATH).read_text())
    if payload.get("milestone") != MILESTONE:
        raise ValueError("expected roadmap milestone")
    return payload
"""

RECORDS_ONLY = """
def _active_tasks(root):
    roadmap = yaml.safe_load((root / ROADMAP_RELATIVE_PATH).read_text())
    return {"milestone_matches": roadmap.get("milestone") == MILESTONE}
"""

VALIDATES_ARTIFACT = """
def validate_artifact(payload):
    if payload["milestone"] != MILESTONE:
        raise ValueError("milestone mismatch")
"""

# NO git-recovery fixture: the exemption it would have covered was deleted after a mutation
# proof showed it decorative. A helper that recovers from git raises at function level rather
# than inside an `if ... != MILESTONE`, so the raise-path rule never fires on it anyway.


ASSERTS = """
def load_roadmap(repo_root):
    payload = yaml.safe_load((repo_root / ROADMAP_RELATIVE_PATH).read_text())
    assert payload.get("milestone") == MILESTONE, "wrong milestone"
    return payload
"""


def test_the_rot_is_caught(tmp_path) -> None:
    """SCENARIO-HARNESS-5945-ROT: reads the live roadmap AND raises on a mismatch."""
    assert _lint().violations([_write(tmp_path, ROT)])


def test_recording_the_mismatch_is_not_the_rot(tmp_path) -> None:
    """False-positive class 2: eight live capstones report it as a field. Honest, not rot."""
    assert not _lint().violations([_write(tmp_path, RECORDS_ONLY)])


def test_validating_the_artifacts_own_milestone_is_the_correct_pattern(tmp_path) -> None:
    """False-positive class 1: the 23-instance pattern, which also names its variable payload."""
    assert not _lint().violations([_write(tmp_path, VALIDATES_ARTIFACT)])


def test_a_module_with_no_milestone_constant_is_ignored(tmp_path) -> None:
    path = tmp_path / "experiment_9999_capstone_vtest.py"
    path.write_text('ROADMAP_RELATIVE_PATH = Path("research-roadmap.yaml")\n' + ROT)
    assert not _lint().violations([path])


def test_the_live_repository_is_clean() -> None:
    """The two real offenders were fixed; this pins that they stay fixed."""
    modules = sorted((REPO / "python" / "carnot").glob("experiment_*capstone*.py"))
    assert modules, "no capstone modules found -- the glob has drifted"
    assert _lint().violations(modules) == []


def test_an_assert_rots_exactly_like_a_raise(tmp_path) -> None:
    """A capstone can refuse by assert instead of raise; both go stale identically.

    Added because a mutation proof found the assert branch DECORATIVE -- deleting it left the
    suite green. No live capstone uses this shape today, but the branch is reachable and the
    failure mode is the same, so it gets a test rather than deletion. (The git-recovery
    exemption, by contrast, was structurally unreachable and was deleted.)
    """
    assert _lint().violations([_write(tmp_path, ASSERTS)])
