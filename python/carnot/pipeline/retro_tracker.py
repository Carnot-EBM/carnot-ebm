"""Retrospective item tracking — close RETRO-012, RETRO-013, RETRO-014.

**Why this module exists:**
    The research conductor identifies retrospective action items (RETRO-NNN) at
    milestone boundaries but has no persistent record of which items are open
    vs. closed across milestones.  Items identified in one milestone reappear in
    the next because there is no authoritative close record.

    ``RetroItemTracker`` provides a lightweight in-process tracker that:
    - Accepts a list of ``(retro_id, description)`` tuples at construction.
    - Allows items to be closed with a rationale and the experiment that closed them.
    - Reports open vs. closed state, and serialises to/from a plain dict so the
      tracker state can be embedded in an experiment result JSON.

    This tracker is *not* a database — it lives for the lifetime of a single
    experiment script run.  The authoritative closure record is the experiment
    result JSON written by ``scripts/experiment_365_retro_close.py``.

Spec: REQ-INFRA-015, REQ-INFRA-016 (used to document RETRO-012/013/014 closure)
"""

from __future__ import annotations

from typing import Any


class RetroItemTracker:
    """Track open/closed state of retrospective action items.

    Parameters
    ----------
    items : list[tuple[str, str]]
        ``(retro_id, description)`` pairs.  All items start as open.

    Example
    -------
    >>> tracker = RetroItemTracker([
    ...     ("RETRO-012", "CARNOT_FORCE_LIVE never set by conductor"),
    ...     ("RETRO-013", "Exp 356 LLMExtractor never implemented"),
    ...     ("RETRO-014", "Missing result JSONs for module-primary experiments"),
    ... ])
    >>> tracker.close("RETRO-012", closed_by_exp=365, rationale="env script created")
    >>> tracker.all_closed()
    False
    >>> tracker.open_items()
    [{'retro_id': 'RETRO-013', ...}, {'retro_id': 'RETRO-014', ...}]
    """

    def __init__(self, items: list[tuple[str, str]]) -> None:
        # Internal state: list of item dicts, each containing:
        #   retro_id, description, closed (bool), closed_by_exp (int|None), rationale (str|None)
        self._items: list[dict[str, Any]] = [
            {
                "retro_id": retro_id,
                "description": description,
                "closed": False,
                "closed_by_exp": None,
                "rationale": None,
            }
            for retro_id, description in items
        ]

    # ------------------------------------------------------------------
    # close()
    # ------------------------------------------------------------------

    def close(self, retro_id: str, closed_by_exp: int, rationale: str) -> None:
        """Mark a retrospective item as closed.

        Parameters
        ----------
        retro_id : str
            The identifier of the item to close (e.g. ``"RETRO-012"``).
        closed_by_exp : int
            Experiment number that produced the fix (e.g. ``365``).
        rationale : str
            Human-readable explanation of how the item was resolved.

        Raises
        ------
        KeyError
            If ``retro_id`` is not found in this tracker.
        """
        for item in self._items:
            if item["retro_id"] == retro_id:
                item["closed"] = True
                item["closed_by_exp"] = closed_by_exp
                item["rationale"] = rationale
                return
        raise KeyError(f"retro_id {retro_id!r} not found in tracker")

    # ------------------------------------------------------------------
    # open_items()
    # ------------------------------------------------------------------

    def open_items(self) -> list[dict[str, Any]]:
        """Return a list of dicts for items that are not yet closed.

        Each dict contains ``retro_id`` and ``description`` (and any other
        fields present in the internal representation).

        Returns
        -------
        list[dict[str, Any]]
            Items where ``closed`` is ``False``, in insertion order.
        """
        return [item for item in self._items if not item["closed"]]

    # ------------------------------------------------------------------
    # all_closed()
    # ------------------------------------------------------------------

    def all_closed(self) -> bool:
        """Return ``True`` when every item has been closed.

        An empty tracker is considered fully closed (vacuously true).
        """
        return all(item["closed"] for item in self._items)

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Serialise tracker state to a plain dict for JSON embedding.

        The output can be round-tripped via ``from_dict``.

        Returns
        -------
        dict[str, Any]
            ``{"items": [...]}`` where each element mirrors the internal
            item dict (retro_id, description, closed, closed_by_exp, rationale).
        """
        return {"items": [dict(item) for item in self._items]}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "RetroItemTracker":
        """Restore a tracker from a dict produced by ``to_dict()``.

        Parameters
        ----------
        d : dict[str, Any]
            Must contain an ``"items"`` key with a list of item dicts.

        Returns
        -------
        RetroItemTracker
            A tracker with the same items and closed/open state as the source.
        """
        # Build with empty list; then restore state directly to avoid
        # re-deriving the full state through the public close() API.
        tracker = cls([])
        tracker._items = [dict(item) for item in d["items"]]
        return tracker
