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

WIDENED 2026-09-05 (ledger row `capstone_milestone_rot_lint.py`, SILENT_NON_FIRING). The
sibling-raise shape -- a guard `if ... == MILESTONE: return payload`, then a `raise` beside the
`if` -- rots identically and exited 0. Catching it also catches both blessed git-recovery
helpers, so the exemption deleted on 2026-08-29 as decorative returns, keyed on mechanism.
Scenarios: SCENARIO-HARNESS-5945-SIBLING-RAISE, SCENARIO-HARNESS-5945-GIT-RECOVERY,
SCENARIO-HARNESS-5945-UNREADABLE, SCENARIO-HARNESS-5945-LIVE-REPOSITORY.
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

# The sibling-raise shape the 2026-09-04 QA-layer audit named. It is byte-for-byte the shape of
# the two blessed recovery helpers, minus the recovery.
SIBLING_RAISE = """
def _roadmap_payload_for_milestone(repo_root):
    payload = yaml.safe_load((repo_root / ROADMAP_RELATIVE_PATH).read_text())
    if isinstance(payload, dict) and payload.get("milestone") == MILESTONE:
        return payload
    raise ValueError("expected roadmap milestone")
"""

# Recovery through git, "git" spelled inline in the call (the V580 helper's form).
GIT_RECOVERY_INLINE = """
def _roadmap_payload_for_milestone(repo_root):
    payload = yaml.safe_load((repo_root / ROADMAP_RELATIVE_PATH).read_text())
    if isinstance(payload, dict) and payload.get("milestone") == MILESTONE:
        return payload
    log = subprocess.run(["git", "-C", str(repo_root), "log", "--format=%H"], capture_output=True)
    for commit in log.stdout.split():
        archived = yaml.safe_load(subprocess.run(["git", "show", commit], capture_output=True).stdout)
        if archived.get("milestone") == MILESTONE:
            return archived
    raise ValueError("expected roadmap milestone")
"""

# Recovery through git with the prefix bound to a name and splatted (the V598 helper's form).
# The first draft of the exemption looked only inside Call nodes and MISSED this.
GIT_RECOVERY_PREFIX = """
def _milestone_inputs(repo_root):
    manifest = yaml.safe_load((repo_root / ROADMAP_RELATIVE_PATH).read_text())
    if not isinstance(manifest, dict) or manifest.get("milestone") == MILESTONE:
        return manifest
    git = ["git", "-C", str(repo_root)]
    log = subprocess.run([*git, "log", "--format=%H"], capture_output=True)
    for commit in log.stdout.split():
        archived = yaml.safe_load(subprocess.run([*git, "show", commit], capture_output=True).stdout)
        if isinstance(archived, dict) and archived.get("milestone") == MILESTONE:
            return archived
    raise ValueError("expected roadmap milestone")
"""

# Recovery through a replay helper that pins inputs to the closing commit (the V576 form).
REPLAY_RECOVERY = """
def _roadmap_payload_for_milestone(repo_root):
    path = repo_root / ROADMAP_RELATIVE_PATH
    payload = yaml.safe_load(_replay_bytes(repo_root, path).decode("utf-8"))
    if isinstance(payload, dict) and payload.get("milestone") == MILESTONE:
        return payload
    raise ValueError("expected roadmap milestone")
"""


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


def test_the_sibling_raise_shape_is_caught(tmp_path) -> None:
    """SCENARIO-HARNESS-5945-SIBLING-RAISE: the raise beside the guard `if` rots identically."""
    found = _lint().violations([_write(tmp_path, SIBLING_RAISE)])
    assert len(found) == 1
    assert "_roadmap_payload_for_milestone()" in found[0][1]


def test_git_recovery_is_exempt_in_both_spellings(tmp_path) -> None:
    """SCENARIO-HARNESS-5945-GIT-RECOVERY: a helper that falls back to history cannot rot."""
    assert not _lint().violations([_write(tmp_path, GIT_RECOVERY_INLINE)])
    assert not _lint().violations([_write(tmp_path, GIT_RECOVERY_PREFIX)])


def test_replay_recovery_is_exempt(tmp_path) -> None:
    """SCENARIO-HARNESS-5945-GIT-RECOVERY: `_replay_bytes` pins inputs to the closing commit."""
    assert not _lint().violations([_write(tmp_path, REPLAY_RECOVERY)])


def test_the_exemption_is_keyed_on_mechanism_not_on_the_helper_name(tmp_path) -> None:
    """A function NAMED like the blessed helper, with no recovery inside, is still the rot."""
    renamed = SIBLING_RAISE.replace("_roadmap_payload_for_milestone", "load_roadmap")
    assert _lint().violations([_write(tmp_path, renamed)])
    assert _lint().violations([_write(tmp_path, SIBLING_RAISE)])


def test_an_unreadable_module_is_a_violation_not_a_pass(tmp_path) -> None:
    """SCENARIO-HARNESS-5945-UNREADABLE: unreadable is not clean. Fail closed."""
    broken = tmp_path / "experiment_9998_capstone_vbroken.py"
    broken.write_text('MILESTONE = "2026.08.580"\ndef broken( :\n')
    found = _lint().violations([broken])
    assert len(found) == 1 and "SyntaxError" in found[0][1]
    missing = tmp_path / "experiment_9997_capstone_vmissing.py"
    assert _lint().violations([missing])


RECOVERY_CAPSTONES = (
    "experiment_6615_v576_independent_capstone.py",
    "experiment_6659_v580_capstone.py",
    "experiment_6847_v598_independent_capstone.py",
)


def test_the_three_recovery_capstones_stay_clean_and_the_exemption_is_exercised() -> None:
    """SCENARIO-HARNESS-5945-LIVE-REPOSITORY: the widened rule would catch all three blessed
    helpers (they are the sibling-raise shape) unless the exemption fires. Assert both halves:
    no violation, AND each module holds a roadmap-reading function the exemption recognises."""
    import ast

    lint = _lint()
    paths = [REPO / "python" / "carnot" / name for name in RECOVERY_CAPSTONES]
    assert all(p.exists() for p in paths)
    assert lint.violations(paths) == []
    for path in paths:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        aliases = lint._roadmap_alias_names(tree)
        exempted = [
            node.name
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef)
            and lint._reads_roadmap(node, aliases)
            and lint._recovers_from_history(node)
        ]
        assert exempted, path.name


def test_an_assert_rots_exactly_like_a_raise(tmp_path) -> None:
    """A capstone can refuse by assert instead of raise; both go stale identically.

    Added because a mutation proof found the assert branch DECORATIVE -- deleting it left the
    suite green. No live capstone uses this shape today, but the branch is reachable and the
    failure mode is the same, so it gets a test rather than deletion. (The git-recovery
    exemption, by contrast, was structurally unreachable and was deleted.)
    """
    assert _lint().violations([_write(tmp_path, ASSERTS)])
