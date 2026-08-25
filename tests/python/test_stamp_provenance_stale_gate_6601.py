"""REQ-VERIFY-6601: a determination made by an older gate must be identifiable as stale.

INCIDENT (2026-08-25). The conductor runs as one long-lived ``--loop`` process. Its
fabrication-gate pass imported ``adversarial_verify`` once and judged artifacts with the
module copy cached at first use. exp6593 was stamped CRITICAL ``DURATION_TOO_SHORT`` by a
14-hour-stale gate under a rule its own commit had already fixed.

Commit 82d8219adf fixed the reload, so new stamps are current. It did not make an OLD stamp
recognisable. 565 artifacts carried the determination and none recorded which gate made it,
so a stale verdict and a fresh one were identical on disk.

These tests pin the property that closes that: a determination carries its gate version, and
a reader can tell current from stale without running the gate.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))

import stamp_provenance  # noqa: E402


# The gate whose logic the version tracks. A threshold, not prose.
_GATE_SRC = '''
"""Original docstring."""

THRESHOLD = 60


def check(duration):
    """Return True when the duration is below the floor."""
    # A comment that carries no semantics.
    return duration < THRESHOLD
'''

_PROSE_EDIT = '''
"""COMPLETELY rewritten module docstring, several sentences longer than before."""

THRESHOLD = 60


def check(duration):
    """A totally different docstring explaining the same behaviour at length."""

    # An entirely different comment.

    return duration < THRESHOLD
'''

_LOGIC_EDIT = '''
"""Original docstring."""

THRESHOLD = 2


def check(duration):
    """Return True when the duration is below the floor."""
    # A comment that carries no semantics.
    return duration < THRESHOLD
'''


def _write_gate(tmp_path: Path, source: str) -> Path:
    path = tmp_path / "adversarial_verify.py"
    path.write_text(source, encoding="utf-8")
    return path


def _stamped(gate: Path) -> dict:
    return {
        "flagged_adversarial": True,
        stamp_provenance.PROVENANCE_FIELD: stamp_provenance.make_provenance("test", gate_path=gate),
    }


# SCENARIO-VERIFY-6601-PROSE-EDIT-IS-NOT-A-VERSION-CHANGE
def test_comment_and_docstring_edits_do_not_change_the_version() -> None:
    """A version that moved on every comment would mark everything stale forever."""
    assert stamp_provenance.semantic_fingerprint(
        _GATE_SRC
    ) == stamp_provenance.semantic_fingerprint(_PROSE_EDIT)


def test_a_stamp_survives_a_prose_only_edit_as_current(tmp_path: Path) -> None:
    gate = _write_gate(tmp_path, _GATE_SRC)
    artifact = _stamped(gate)
    assert stamp_provenance.stamp_status(artifact, gate_path=gate) == "current"

    _write_gate(tmp_path, _PROSE_EDIT)
    assert stamp_provenance.stamp_status(artifact, gate_path=gate) == "current"
    assert stamp_provenance.describe_stamp_status(artifact, gate_path=gate) == ""


# SCENARIO-VERIFY-6601-LOGIC-EDIT-IS-A-VERSION-CHANGE
def test_a_threshold_change_changes_the_version() -> None:
    assert stamp_provenance.semantic_fingerprint(
        _GATE_SRC
    ) != stamp_provenance.semantic_fingerprint(_LOGIC_EDIT)


def test_the_incident_shape_a_stamp_made_before_a_logic_fix_reads_stale(
    tmp_path: Path,
) -> None:
    """exp6593 exactly: stamped under a 60s floor, then the floor rule changed."""
    gate = _write_gate(tmp_path, _GATE_SRC)
    artifact = _stamped(gate)
    assert stamp_provenance.stamp_status(artifact, gate_path=gate) == "current"

    _write_gate(tmp_path, _LOGIC_EDIT)
    assert stamp_provenance.stamp_status(artifact, gate_path=gate) == "stale"

    note = stamp_provenance.describe_stamp_status(artifact, gate_path=gate)
    assert "STALE" in note
    assert "OLDER gate version" in note


# SCENARIO-VERIFY-6601-CACHE-INVALIDATES-ON-EDIT
def test_the_version_cache_invalidates_when_the_gate_source_changes(
    tmp_path: Path,
) -> None:
    """Caching by path alone is the defect that caused the incident."""
    gate = _write_gate(tmp_path, _GATE_SRC)
    before = stamp_provenance.current_gate_version(gate)

    _write_gate(tmp_path, _LOGIC_EDIT)
    after = stamp_provenance.current_gate_version(gate)

    assert before != after, "a second read returned the cached version after an edit"


# SCENARIO-VERIFY-6601-LEGACY-STAMP-IS-IDENTIFIABLE
def test_a_legacy_stamp_with_no_provenance_reads_unversioned(tmp_path: Path) -> None:
    """All 565 artifacts stamped before this shipped are in this state."""
    gate = _write_gate(tmp_path, _GATE_SRC)
    legacy = {"flagged_adversarial": True, "corrigendum_pending": [{"kind": "TAUTOLOGY"}]}
    assert stamp_provenance.stamp_status(legacy, gate_path=gate) == "unversioned"
    assert "PROVENANCE MISSING" in stamp_provenance.describe_stamp_status(legacy, gate_path=gate)


@pytest.mark.parametrize(
    "provenance",
    [None, {}, {"stamped_at": "2026-01-01T00:00:00Z"}, {"gate_version": ""}, "not-a-dict"],
)
def test_a_malformed_provenance_block_reads_unversioned_not_current(
    tmp_path: Path, provenance: object
) -> None:
    """Fail toward re-checking. A broken block must never read as a fresh judgement."""
    gate = _write_gate(tmp_path, _GATE_SRC)
    artifact = {"flagged_adversarial": True, stamp_provenance.PROVENANCE_FIELD: provenance}
    assert stamp_provenance.stamp_status(artifact, gate_path=gate) == "unversioned"


def test_an_unstamped_artifact_is_not_reported_as_stale(tmp_path: Path) -> None:
    """The note is appended unconditionally by callers, so it must stay quiet here."""
    gate = _write_gate(tmp_path, _GATE_SRC)
    for artifact in ({}, {"flagged_adversarial": False}, {"flagged_adversarial": None}):
        assert stamp_provenance.stamp_status(artifact, gate_path=gate) == "unstamped"
        assert stamp_provenance.describe_stamp_status(artifact, gate_path=gate) == ""


# SCENARIO-VERIFY-6601-PRINCIPLE-WRAPPED-STAMP-IS-READ
def test_a_principle_wrapped_determination_is_still_read_as_stamped(
    tmp_path: Path,
) -> None:
    """QA-Layer rule: any field may be `{"principle": ..., "value": ...}`."""
    gate = _write_gate(tmp_path, _GATE_SRC)
    artifact = {"flagged_adversarial": {"principle": "the gate's verdict", "value": True}}
    assert stamp_provenance.stamp_status(artifact, gate_path=gate) == "unversioned"


def test_a_principle_wrapped_provenance_block_is_read_through(tmp_path: Path) -> None:
    gate = _write_gate(tmp_path, _GATE_SRC)
    artifact = {
        "flagged_adversarial": True,
        stamp_provenance.PROVENANCE_FIELD: {
            "principle": "records which gate judged this",
            "value": stamp_provenance.make_provenance("test", gate_path=gate),
        },
    }
    assert stamp_provenance.stamp_status(artifact, gate_path=gate) == "current"


# SCENARIO-VERIFY-6601-READER-SAYS-SO
def test_the_downstream_gate_names_the_staleness_and_still_blocks() -> None:
    """conductor_gates must not present an undatable stamp as a fresh judgement."""
    import conductor_gates

    legacy = {"flagged_adversarial": True, "corrigendum_pending": [{"kind": "TAUTOLOGY"}]}
    reason = conductor_gates._diagnose_missing_field(legacy, "score", "BASE")

    assert "UPSTREAM IS QUARANTINED" in reason, "the quarantine must still be reported"
    assert "PROVENANCE MISSING" in reason, "the staleness must be named"


def test_a_current_stamp_adds_no_noise_to_the_gate_message() -> None:
    import conductor_gates

    fresh = {
        "flagged_adversarial": True,
        stamp_provenance.PROVENANCE_FIELD: stamp_provenance.make_provenance("test"),
    }
    reason = conductor_gates._diagnose_missing_field(fresh, "score", "BASE")
    assert "UPSTREAM IS QUARANTINED" in reason
    assert "STALE" not in reason and "PROVENANCE MISSING" not in reason


# The gate writers must actually record provenance, or the field is decorative.
def test_the_backfill_writer_records_provenance(tmp_path: Path, monkeypatch) -> None:
    """Without this the stamp is written but undatable -- the whole defect, unchanged."""
    import adversarial_verify

    artifact_path = tmp_path / "experiment_9999_fake.json"
    artifact_path.write_text(json.dumps({"experiment": 9999, "duration_s": 0.5}))

    monkeypatch.setattr(
        adversarial_verify,
        "verify_artifact",
        lambda p, **kw: {"flags": [{"kind": "GATE_PASSED_WITHOUT_DATA", "severity": "critical"}]},
    )
    adversarial_verify.backfill_stamps([artifact_path], apply=True)

    written = json.loads(artifact_path.read_text())
    assert written["flagged_adversarial"] is True
    block = written[stamp_provenance.PROVENANCE_FIELD]
    assert block["gate_version"] == stamp_provenance.current_gate_version()
    assert block["stamper"] == "adversarial_verify.backfill_stamps"
    assert block["stamped_at"].endswith("Z")
    assert stamp_provenance.stamp_status(written) == "current"


def test_backfill_cannot_revise_an_existing_determination(tmp_path: Path, monkeypatch) -> None:
    """Pins the fact that --backfill --apply can only ADD, so it cannot correct the 565."""
    import adversarial_verify

    artifact_path = tmp_path / "experiment_9998_already.json"
    artifact_path.write_text(
        json.dumps({"experiment": 9998, "flagged_adversarial": True, "duration_s": 0.5})
    )

    monkeypatch.setattr(adversarial_verify, "verify_artifact", lambda p, **kw: {"flags": []})
    records = adversarial_verify.backfill_stamps([artifact_path], apply=True)

    assert records == [], "backfill reported work on an already-stamped artifact"
    assert json.loads(artifact_path.read_text())["flagged_adversarial"] is True


def test_scan_groups_the_corpus_without_running_the_gate(tmp_path: Path) -> None:
    gate = _write_gate(tmp_path, _GATE_SRC)
    (tmp_path / "a.json").write_text(json.dumps({"flagged_adversarial": True}))
    (tmp_path / "b.json").write_text(json.dumps(_stamped(gate)))
    (tmp_path / "c.json").write_text(json.dumps({"experiment": 1}))
    (tmp_path / "d.json").write_text("{ not json")

    buckets = stamp_provenance.scan(sorted(tmp_path.glob("*.json")), gate_path=gate)

    assert buckets["unversioned"] == ["a.json"]
    assert buckets["current"] == ["b.json"]
    assert buckets["unstamped"] == ["c.json"]
    assert "d.json" not in sum(buckets.values(), [])
