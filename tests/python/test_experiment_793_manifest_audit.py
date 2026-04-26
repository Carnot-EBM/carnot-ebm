"""Tests for Experiment 793 — Manifest Full-Scope Audit.

Traces:
  REQ-INFRA-058 — manifest check MUST be at ALL dequeue sites
  REQ-INFRA-059 — excluded tasks MUST be logged at WARNING level before skip
  SCENARIO-INFRA-067 — Exp 527 excluded at dequeue, WARNING emitted, task skipped
  SCENARIO-INFRA-068 — Exp 793 not in manifest, task runs normally
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from experiment_793_manifest_audit import (  # noqa: E402
    DEQUEUE_PATTERNS,
    build_patch_sites,
    scan_conductor_text,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def guarded_conductor_text() -> str:
    """Synthetic conductor snippet with manifest check adjacent to dequeue."""
    return """\
def pick_next_task(completed_log):
    for task in RESEARCH_TASKS:
        excluded, reason = _task_is_excluded(task)  # manifest check
        if excluded:
            continue
        return task
    return None
"""


@pytest.fixture()
def unguarded_conductor_text() -> str:
    """Synthetic conductor snippet with manifest check far from dequeue.

    The for-loop is the dequeue site; the manifest check appears 20 lines later,
    outside the 5-line proximity window.  This models the real bug where Exp 527
    slipped through because the guard was logically present but not adjacent.
    """
    lines = [
        "def pick_next_task(completed_log):",
        "    for task in RESEARCH_TASKS:",
        "        title = task['title']",
        "        if title in completed:",
        "            continue",
        "        if _deliverable_exists(task):",
        "            continue",
        "        # Many lines of other checks...",
        "        x = 1",
        "        y = 2",
        "        z = 3",
        "        a = 4",
        "        b = 5",
        "        c = 6",
        "        d = 7",
        "        e = 8",
        "        # exclusion manifest (far from dequeue)",
        "        excluded, reason = _task_is_excluded(task)",
        "        if excluded:",
        "            continue",
        "        return task",
        "    return None",
    ]
    return "\n".join(lines)


@pytest.fixture()
def mixed_conductor_text() -> str:
    """Synthetic conductor with both guarded and unguarded dequeue patterns.

    The secondary path deliberately has no exclusion check — this is the bug
    pattern that allowed Exp 527 to bypass retirement for 7+ milestones.
    The comment is intentionally worded to avoid the word 'manifest' so the
    proximity scanner does not false-classify this site as guarded.
    """
    return """\
# Primary dispatch path (guarded)
def pick_next_task(completed_log):
    for task in RESEARCH_TASKS:
        excluded, reason = _task_is_excluded(task)  # manifest guard adjacent
        if excluded:
            continue
        return task

# Secondary path (no exclusion check — dequeue site is unguarded)
def pick_next_from_priority(queue):
    task = queue.pop()
    return task
"""


@pytest.fixture()
def no_dequeue_text() -> str:
    """Synthetic conductor with no dequeue patterns at all."""
    return """\
def run_cmd(cmd):
    result = subprocess.run(cmd)
    return result.returncode, result.stdout

def git_status():
    _, stdout, _ = run_cmd(['git', 'diff', '--stat'])
    return stdout.strip()
"""


# ---------------------------------------------------------------------------
# Tests for scan_conductor_text
# ---------------------------------------------------------------------------


class TestScanConductorText:
    """Tests for the core pattern-scanning function.  REQ-INFRA-058."""

    def test_finds_for_task_in_pattern(self, guarded_conductor_text: str) -> None:
        """Scanner finds 'for task in RESEARCH_TASKS' as a dequeue site.

        Traces: REQ-INFRA-058 — the for-loop over RESEARCH_TASKS is the
        primary dequeue site in the conductor.
        """
        sites = scan_conductor_text(guarded_conductor_text)
        assert len(sites) >= 1
        patterns_found = [s["pattern_matched"] for s in sites]
        assert any("for" in p and "task" in p for p in patterns_found)

    def test_finds_pop_pattern(self, mixed_conductor_text: str) -> None:
        """Scanner finds '.pop()' as a dequeue pattern.

        Traces: REQ-INFRA-058 — queue.pop() is a dequeue site pattern.
        The pattern_matched field stores the raw regex string (e.g. r'\.pop\(\)')
        so we match by checking for 'pop' in the pattern string.
        """
        sites = scan_conductor_text(mixed_conductor_text)
        pop_sites = [s for s in sites if "pop" in s["pattern_matched"]]
        assert len(pop_sites) >= 1

    def test_no_dequeue_returns_empty(self, no_dequeue_text: str) -> None:
        """Scanner returns empty list when no dequeue patterns are present.

        Traces: REQ-INFRA-058 — scanner must be silent on clean code paths.
        """
        sites = scan_conductor_text(no_dequeue_text)
        assert sites == []

    def test_comment_only_lines_skipped(self) -> None:
        """Scanner ignores lines that consist entirely of a comment.

        Traces: REQ-INFRA-058 — a comment mentioning 'for task in' is not a
        dequeue site; only executable statements count.
        """
        text = "# for task in RESEARCH_TASKS: iterate here\ndef foo(): pass\n"
        sites = scan_conductor_text(text)
        assert sites == []

    def test_returns_correct_line_numbers(self, guarded_conductor_text: str) -> None:
        """Line numbers in results are 1-based and match the source.

        Traces: REQ-INFRA-058 — patch sites need accurate line numbers so a
        human can apply the recommended patch without ambiguity.
        """
        sites = scan_conductor_text(guarded_conductor_text)
        assert all(s["line_number"] >= 1 for s in sites)

    def test_code_snippet_is_stripped(self, guarded_conductor_text: str) -> None:
        """code_snippet is stripped of leading whitespace.

        Traces: REQ-INFRA-058 — clean snippets are easier to match in patch tools.
        """
        sites = scan_conductor_text(guarded_conductor_text)
        for s in sites:
            assert s["code_snippet"] == s["code_snippet"].lstrip()

    def test_no_duplicate_line_entries(self, mixed_conductor_text: str) -> None:
        """Each source line appears at most once in the results.

        Traces: REQ-INFRA-058 — if a line matches multiple patterns the scanner
        must not create duplicate patch entries for the same line.
        """
        sites = scan_conductor_text(mixed_conductor_text)
        line_numbers = [s["line_number"] for s in sites]
        assert len(line_numbers) == len(set(line_numbers))


# ---------------------------------------------------------------------------
# Tests for guard classification
# ---------------------------------------------------------------------------


class TestGuardClassification:
    """Tests for the guarded/unguarded classification logic.

    REQ-INFRA-058 — every dequeue site must be independently checked.
    SCENARIO-INFRA-067, SCENARIO-INFRA-068.
    """

    def test_adjacent_manifest_classifies_as_guarded(self, guarded_conductor_text: str) -> None:
        """When 'manifest' appears within 5 lines, site is classified guarded.

        Traces: REQ-INFRA-058, SCENARIO-INFRA-068 — the for-loop in
        pick_next_task with an adjacent _task_is_excluded() call is guarded.
        """
        sites = scan_conductor_text(guarded_conductor_text)
        for_task_sites = [s for s in sites if "for" in s["pattern_matched"]]
        assert len(for_task_sites) >= 1
        assert all(s["is_manifest_checked"] for s in for_task_sites)
        assert all(not s["patch_required"] for s in for_task_sites)

    def test_distant_manifest_classifies_as_unguarded(self, unguarded_conductor_text: str) -> None:
        """When 'manifest' is >5 lines away, site is classified unguarded.

        Traces: REQ-INFRA-058, SCENARIO-INFRA-067 — the real bug: manifest
        check exists but is too far from the for-loop to be considered adjacent.
        This is how Exp 527 slipped through 7+ milestone cycles.
        """
        sites = scan_conductor_text(unguarded_conductor_text)
        for_task_sites = [s for s in sites if "for" in s["pattern_matched"]]
        assert len(for_task_sites) >= 1
        # At least one for-task site should be unguarded (manifest is far away)
        unguarded = [s for s in for_task_sites if not s["is_manifest_checked"]]
        assert len(unguarded) >= 1

    def test_pop_without_manifest_is_unguarded(self, mixed_conductor_text: str) -> None:
        """A bare .pop() without adjacent manifest check is flagged as unguarded.

        Traces: REQ-INFRA-058 — secondary dequeue paths (not going through
        pick_next_task) must also be independently guarded.  The pattern_matched
        field stores the raw regex string (r'\.pop\(\)'), so we match by checking
        for 'pop' in the pattern string.
        """
        sites = scan_conductor_text(mixed_conductor_text)
        pop_sites = [s for s in sites if "pop" in s["pattern_matched"]]
        assert len(pop_sites) >= 1
        assert any(not s["is_manifest_checked"] for s in pop_sites)

    def test_patch_required_matches_is_manifest_checked(self, mixed_conductor_text: str) -> None:
        """patch_required is always the logical inverse of is_manifest_checked.

        Traces: REQ-INFRA-058 — consistency invariant: a guarded site needs no
        patch; an unguarded site requires a patch.
        """
        sites = scan_conductor_text(mixed_conductor_text)
        for s in sites:
            assert s["patch_required"] == (not s["is_manifest_checked"])

    def test_recommended_patch_code_present_for_unguarded(
        self, unguarded_conductor_text: str
    ) -> None:
        """Unguarded sites include a non-empty recommended_patch_code string.

        Traces: REQ-INFRA-058, REQ-INFRA-059 — the patch code must reference
        _task_is_excluded() and logger.warning() so a human has a complete fix.
        """
        sites = scan_conductor_text(unguarded_conductor_text)
        unguarded = [s for s in sites if s["patch_required"]]
        for s in unguarded:
            code = s["recommended_patch_code"]
            assert "_task_is_excluded" in code
            assert "logger.warning" in code

    def test_recommended_patch_code_noop_for_guarded(self, guarded_conductor_text: str) -> None:
        """Guarded sites have recommended_patch_code indicating no change needed.

        Traces: REQ-INFRA-058 — humans must be able to distinguish sites that
        already satisfy the requirement from sites that need patching.
        """
        sites = scan_conductor_text(guarded_conductor_text)
        guarded = [s for s in sites if not s["patch_required"]]
        for s in guarded:
            assert "No patch required" in s["recommended_patch_code"]


# ---------------------------------------------------------------------------
# Tests for artifact field completeness
# ---------------------------------------------------------------------------


class TestArtifactFields:
    """Verify that build_patch_sites produces the required schema fields.

    Traces: REQ-INFRA-058 — the deliverable JSON must have all required fields
    so downstream consumers (humans, future audit scripts) can parse it.
    """

    REQUIRED_PATCH_SITE_FIELDS = {
        "line_number",
        "code_snippet",
        "pattern_matched",
        "is_manifest_checked",
        "patch_required",
        "recommended_patch_code",
    }

    def test_build_patch_sites_returns_required_fields(self, mixed_conductor_text: str) -> None:
        """build_patch_sites returns dicts with all required schema fields.

        Traces: REQ-INFRA-058 — artifact schema completeness.
        """
        raw = scan_conductor_text(mixed_conductor_text)
        sites = build_patch_sites(raw)
        for site in sites:
            missing = self.REQUIRED_PATCH_SITE_FIELDS - set(site.keys())
            assert missing == set(), f"Missing fields in patch site: {missing}"

    def test_build_patch_sites_drops_context_lines(self, mixed_conductor_text: str) -> None:
        """build_patch_sites strips internal context_lines from the output.

        Traces: REQ-INFRA-058 — the deliverable artifact should be lean;
        raw context lines are needed during analysis but excluded from the JSON.
        """
        raw = scan_conductor_text(mixed_conductor_text)
        sites = build_patch_sites(raw)
        for site in sites:
            assert "context_lines" not in site

    def test_line_number_is_int(self, mixed_conductor_text: str) -> None:
        """line_number field is an integer, not a string.

        Traces: REQ-INFRA-058 — typed fields prevent JSON consumers from having
        to guess whether to cast the value.
        """
        raw = scan_conductor_text(mixed_conductor_text)
        sites = build_patch_sites(raw)
        for site in sites:
            assert isinstance(site["line_number"], int)

    def test_is_manifest_checked_is_bool(self, mixed_conductor_text: str) -> None:
        """is_manifest_checked is a Python bool, not a truthy int or string.

        Traces: REQ-INFRA-058 — typed fields prevent ambiguous truthiness tests
        in consumers that use strict bool checks (e.g. site["is_manifest_checked"] is True).
        """
        raw = scan_conductor_text(mixed_conductor_text)
        sites = build_patch_sites(raw)
        for site in sites:
            assert isinstance(site["is_manifest_checked"], bool)

    def test_patch_required_is_bool(self, mixed_conductor_text: str) -> None:
        """patch_required is a Python bool.

        Traces: REQ-INFRA-058 — typed fields.
        """
        raw = scan_conductor_text(mixed_conductor_text)
        sites = build_patch_sites(raw)
        for site in sites:
            assert isinstance(site["patch_required"], bool)

    def test_artifact_is_json_serializable(self, mixed_conductor_text: str) -> None:
        """build_patch_sites output can be serialized to JSON without error.

        Traces: REQ-INFRA-058 — the deliverable must be a valid JSON file that
        downstream tools can load without custom serializers.
        """
        raw = scan_conductor_text(mixed_conductor_text)
        sites = build_patch_sites(raw)
        serialized = json.dumps(sites)
        reloaded = json.loads(serialized)
        assert len(reloaded) == len(sites)

    def test_dequeue_patterns_list_is_nonempty(self) -> None:
        """DEQUEUE_PATTERNS contains at least one compiled regex.

        Traces: REQ-INFRA-058 — scanner must cover at least the documented
        dequeue patterns (for/pop/popleft/next/choice/queue.get/task_queue).
        """
        assert len(DEQUEUE_PATTERNS) >= 7
        # Verify each entry is a compiled pattern (has a .search method)
        for pattern in DEQUEUE_PATTERNS:
            assert hasattr(pattern, "search")
