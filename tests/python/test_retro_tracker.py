"""Tests for python/carnot/pipeline/retro_tracker.py — RetroItemTracker.

Coverage targets (100% required)
---------------------------------
- RetroItemTracker.__init__: stores items as open
- close(): marks item closed with rationale and closed_by_exp
- open_items(): returns only unclosed items
- all_closed(): True when all items are closed
- to_dict() / from_dict(): round-trip serialization

Spec: REQ-INFRA-015 (RETRO-012 close tracking),
      REQ-INFRA-016 (RETRO-014 close tracking)
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from python.carnot.pipeline.retro_tracker import RetroItemTracker


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


class TestRetroItemTrackerInit:
    def test_creates_with_items(self) -> None:
        """All items start as open."""
        tracker = RetroItemTracker(
            [("RETRO-012", "CARNOT_FORCE_LIVE never set"),
             ("RETRO-013", "Exp 356 never implemented"),
             ("RETRO-014", "Missing result JSONs")]
        )
        assert len(tracker.open_items()) == 3

    def test_empty_items(self) -> None:
        """Empty list of items is valid."""
        tracker = RetroItemTracker([])
        assert tracker.open_items() == []
        assert tracker.all_closed() is True

    def test_open_items_contain_retro_id(self) -> None:
        """open_items() dicts contain retro_id and description."""
        tracker = RetroItemTracker([("RETRO-012", "desc")])
        items = tracker.open_items()
        assert items[0]["retro_id"] == "RETRO-012"
        assert items[0]["description"] == "desc"


# ---------------------------------------------------------------------------
# close()
# ---------------------------------------------------------------------------


class TestRetroItemTrackerClose:
    def _tracker(self) -> RetroItemTracker:
        return RetroItemTracker(
            [("RETRO-012", "CARNOT_FORCE_LIVE never set"),
             ("RETRO-013", "Exp 356 never implemented"),
             ("RETRO-014", "Missing result JSONs")]
        )

    def test_close_reduces_open_count(self) -> None:
        """Closing one item reduces open_items() by 1."""
        tracker = self._tracker()
        tracker.close("RETRO-012", closed_by_exp=365, rationale="env script created")
        assert len(tracker.open_items()) == 2

    def test_close_records_rationale(self) -> None:
        """Closed item retains rationale in to_dict() output."""
        tracker = self._tracker()
        tracker.close("RETRO-013", closed_by_exp=365, rationale="Addressed by Exp 366")
        d = tracker.to_dict()
        closed = [i for i in d["items"] if i["retro_id"] == "RETRO-013"][0]
        assert closed["rationale"] == "Addressed by Exp 366"

    def test_close_records_closed_by_exp(self) -> None:
        """Closed item records which experiment closed it."""
        tracker = self._tracker()
        tracker.close("RETRO-014", closed_by_exp=365, rationale="enforcer created")
        d = tracker.to_dict()
        closed = [i for i in d["items"] if i["retro_id"] == "RETRO-014"][0]
        assert closed["closed_by_exp"] == 365

    def test_close_nonexistent_raises(self) -> None:
        """Closing an unknown retro_id raises KeyError."""
        tracker = self._tracker()
        with pytest.raises(KeyError):
            tracker.close("RETRO-999", closed_by_exp=365, rationale="wrong")

    def test_close_all(self) -> None:
        """Closing all items yields open_items() == []."""
        tracker = self._tracker()
        tracker.close("RETRO-012", 365, "fix 1")
        tracker.close("RETRO-013", 365, "fix 2")
        tracker.close("RETRO-014", 365, "fix 3")
        assert tracker.open_items() == []

    def test_close_sets_closed_flag(self) -> None:
        """Closed item has closed=True in to_dict()."""
        tracker = self._tracker()
        tracker.close("RETRO-012", 365, "done")
        d = tracker.to_dict()
        item = [i for i in d["items"] if i["retro_id"] == "RETRO-012"][0]
        assert item["closed"] is True

    def test_open_item_has_closed_false(self) -> None:
        """Unclosed item has closed=False in to_dict()."""
        tracker = self._tracker()
        d = tracker.to_dict()
        item = [i for i in d["items"] if i["retro_id"] == "RETRO-012"][0]
        assert item["closed"] is False


# ---------------------------------------------------------------------------
# open_items()
# ---------------------------------------------------------------------------


class TestOpenItems:
    def test_returns_only_open(self) -> None:
        """open_items() excludes closed items."""
        tracker = RetroItemTracker(
            [("RETRO-012", "A"), ("RETRO-013", "B")]
        )
        tracker.close("RETRO-012", 365, "done")
        open_ids = [i["retro_id"] for i in tracker.open_items()]
        assert "RETRO-012" not in open_ids
        assert "RETRO-013" in open_ids


# ---------------------------------------------------------------------------
# all_closed()
# ---------------------------------------------------------------------------


class TestAllClosed:
    def test_false_when_items_open(self) -> None:
        tracker = RetroItemTracker([("RETRO-012", "x")])
        assert tracker.all_closed() is False

    def test_true_when_all_closed(self) -> None:
        tracker = RetroItemTracker([("RETRO-012", "x")])
        tracker.close("RETRO-012", 365, "done")
        assert tracker.all_closed() is True

    def test_true_with_empty_tracker(self) -> None:
        tracker = RetroItemTracker([])
        assert tracker.all_closed() is True


# ---------------------------------------------------------------------------
# to_dict() / from_dict()
# ---------------------------------------------------------------------------


class TestSerialisation:
    def test_roundtrip_all_open(self) -> None:
        """to_dict / from_dict preserves open items."""
        tracker = RetroItemTracker(
            [("RETRO-012", "A"), ("RETRO-013", "B")]
        )
        d = tracker.to_dict()
        restored = RetroItemTracker.from_dict(d)
        assert len(restored.open_items()) == 2

    def test_roundtrip_partial_closed(self) -> None:
        """Closed state survives round-trip."""
        tracker = RetroItemTracker(
            [("RETRO-012", "A"), ("RETRO-013", "B")]
        )
        tracker.close("RETRO-012", 365, "done")
        d = tracker.to_dict()
        restored = RetroItemTracker.from_dict(d)
        assert len(restored.open_items()) == 1
        assert restored.open_items()[0]["retro_id"] == "RETRO-013"

    def test_roundtrip_all_closed(self) -> None:
        """all_closed() is True after round-trip when all were closed."""
        tracker = RetroItemTracker([("RETRO-012", "A")])
        tracker.close("RETRO-012", 365, "done")
        restored = RetroItemTracker.from_dict(tracker.to_dict())
        assert restored.all_closed() is True

    def test_to_dict_contains_items_key(self) -> None:
        """to_dict() output has an 'items' key."""
        tracker = RetroItemTracker([("RETRO-012", "A")])
        assert "items" in tracker.to_dict()

    def test_from_dict_closed_by_exp_preserved(self) -> None:
        """closed_by_exp field survives round-trip."""
        tracker = RetroItemTracker([("RETRO-014", "C")])
        tracker.close("RETRO-014", closed_by_exp=365, rationale="test")
        restored = RetroItemTracker.from_dict(tracker.to_dict())
        d = restored.to_dict()
        item = [i for i in d["items"] if i["retro_id"] == "RETRO-014"][0]
        assert item["closed_by_exp"] == 365
