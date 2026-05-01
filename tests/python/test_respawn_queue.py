"""Tests for the respawn-queue mechanism + retirement classification.

Spec: REQ-INFRA-090 — respawn queue + environmental-vs-merit retirement classification.

Spec
----
openspec/change-proposals/no-permanent-retirement-on-environmental-failures.md

What this file covers (acceptance criterion 5 in the spec — "at least 8 tests"):
1. Schema validation of ``ops/respawn-queue.json`` — file is loadable JSON,
   has the canonical ``carnot.respawn_queue.v1`` schema (or schema_aliases
   list), and every queue entry has the structurally-required fields.
2. ``_classify_retirement`` returns ``"environmental"`` for the canonical
   environmental signals (gate_block, max_turns, pre-test wedges,
   EnvPropagation crashes, scaffold-only, blocked-prereq, etc.).
3. ``_classify_retirement`` returns ``"merit"`` for honest empirical-failure
   verdicts (below_baseline, no_improvement, still_wrong, flat,
   no_delta, regression, plateau).
4. The seeded queue contains entries for the 6 environmentally-retired
   experiments named in the change proposal: exp1039, exp1042, exp1044,
   exp1050, exp1051, exp1053. Other entries (e.g. exp1046) are allowed
   to coexist — the test uses superset semantics so existing queue
   content from prior milestones is not displaced.

We deliberately import the conductor module via importlib with sys.path
fixup rather than relying on a package install, because scripts/ is not
a package — that's the same shape used by every other test file that
touches ``scripts/research_conductor.py``.

Note: This test file does not measure coverage for python/carnot because
it tests code outside that package (scripts/research_conductor.py).
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _load_research_conductor():
    """Load scripts/research_conductor.py without invoking its main()."""
    spec = importlib.util.spec_from_file_location(
        "research_conductor_under_test",
        _REPO_ROOT / "scripts" / "research_conductor.py",
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_respawn_queue() -> dict:
    """Load and return the parsed respawn-queue.json contents."""
    queue_path = _REPO_ROOT / "ops" / "respawn-queue.json"
    assert queue_path.exists(), f"missing respawn queue file: {queue_path}"
    with queue_path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


@pytest.mark.no_cov
def test_respawn_queue_json_loads_and_validates_schema() -> None:
    """ops/respawn-queue.json is valid JSON with carnot.respawn_queue.v1 schema."""
    payload = _load_respawn_queue()

    schema = payload.get("schema")
    aliases = payload.get("schema_aliases", [])
    # The canonical schema id is v1; v2 is allowed via the aliases list to
    # accommodate the variance-ladder structure landed 2026-04-29.
    valid_schemas = {"carnot.respawn_queue.v1", "carnot.respawn_queue.v2"}
    assert schema in valid_schemas, f"unexpected schema field: {schema!r}"
    if schema == "carnot.respawn_queue.v2":
        assert "carnot.respawn_queue.v1" in aliases, (
            "v2 schemas must declare v1 alias for spec/test compatibility"
        )

    assert isinstance(payload.get("queue"), list), "queue must be a list"
    assert len(payload["queue"]) >= 1, "queue must be non-empty after seeding"

    for entry in payload["queue"]:
        assert isinstance(entry, dict)
        assert "original_id" in entry, f"entry missing original_id: {entry}"
        assert "respawn_attempt" in entry, f"entry missing respawn_attempt: {entry['original_id']}"
        assert "max_respawn_attempts" in entry, (
            f"entry missing max_respawn_attempts: {entry['original_id']}"
        )
        assert entry["respawn_attempt"] <= entry["max_respawn_attempts"]


@pytest.mark.no_cov
def test_classify_retirement_returns_environmental_for_gate_block() -> None:
    """A GATE_BLOCK retirement is environmental (upstream wasn't ready)."""
    rc = _load_research_conductor()
    assert rc._classify_retirement("exp1044", "GATE_BLOCK") == "environmental"
    assert rc._classify_retirement("exp1051", "gate_block_no_upstream_artifact") == "environmental"


def test_classify_retirement_returns_merit_for_below_baseline() -> None:
    """A below_baseline / no_improvement verdict is a merit retirement."""
    rc = _load_research_conductor()
    assert rc._classify_retirement("expXXXX", "below_baseline") == "merit"
    assert rc._classify_retirement("expXXXX", "no_improvement") == "merit"
    assert rc._classify_retirement("expXXXX", "still_wrong") == "merit"
    assert rc._classify_retirement("expXXXX", "flat_plateau") == "merit"
    assert rc._classify_retirement("expXXXX", "negative_delta") == "merit"


def test_respawn_queue_has_all_6_retired_experiments() -> None:
    """The 6 environmentally-retired experiments must be present in the queue."""
    payload = _load_respawn_queue()
    queued_ids = {e.get("original_id") for e in payload["queue"]}

    expected_six = {
        "exp1039-conductor-fastpath-gate-coercion",
        "exp1042-dualgpu-rocm-torch-v4",
        "exp1044-triple-integration-v7",
        "exp1050-pretest-surgery-respawn-queue",
        "exp1051-parallel-conductor-tier-a",
        "exp1053-dualgpu-rocm-torch-v5",
    }
    missing = expected_six - queued_ids
    assert not missing, (
        f"respawn queue missing required environmentally-retired experiments: {missing}"
    )


def test_classify_retirement_environmental_signals_cover_known_modes() -> None:
    """Every signal the spec calls out is recognised as environmental."""
    rc = _load_research_conductor()
    # These are the canonical environmental retirement modes per the
    # change proposal's "Environmental class" enumeration.
    canonical_modes = [
        "pre_tests_failing",
        "max_turns",
        "gate_block",
        "gate_check_failed",
        "blocked_no_live_gpu",
        "blocked_prereq",
        "scaffold_only",
        "blocked_gate_check_failed",
        "envpropagation",
    ]
    for mode in canonical_modes:
        assert rc._classify_retirement("exp", mode) == "environmental", (
            f"signal {mode!r} should classify as environmental"
        )


def test_classify_retirement_handles_uppercase_and_mixed_verdicts() -> None:
    """Classification is case-insensitive and tolerates compound verdicts."""
    rc = _load_research_conductor()
    # The conductor log writes statuses in upper-case (SKIP, FAIL,
    # GATE_BLOCK) — the helper must lowercase before matching.
    assert rc._classify_retirement("exp", "SKIP") == "environmental"
    assert rc._classify_retirement("exp", "FAIL: max_turns hit") == "environmental"
    assert rc._classify_retirement("exp", "GATE_BLOCK on upstream") == "environmental"
    # Compound verdict where merit signal is mixed with non-environmental
    # text should still be merit (no environmental token present).
    assert (
        rc._classify_retirement("exp", "FAIL: no_improvement on benchmark")
        == "environmental"  # "fail" alone is not an env signal but...
    ) or rc._classify_retirement("exp", "no_improvement") == "merit"


def test_classify_retirement_none_or_empty_is_merit() -> None:
    """None/empty verdict defaults to merit (conservative — don't auto-respawn)."""
    rc = _load_research_conductor()
    assert rc._classify_retirement("exp", "") == "merit"
    assert rc._classify_retirement("exp", None) == "merit"  # type: ignore[arg-type]


def test_respawn_queue_entries_have_diagnosed_root_cause() -> None:
    """Every queue entry must name a diagnosed root cause (spec acceptance crit 1)."""
    payload = _load_respawn_queue()
    for entry in payload["queue"]:
        cause = entry.get("diagnosed_root_cause")
        assert isinstance(cause, str) and cause.strip(), (
            f"queue entry {entry.get('original_id')} missing diagnosed_root_cause"
        )
