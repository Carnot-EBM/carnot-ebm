"""REQ-REPORT-8624: retain visible horizons without assigning causal credit.

The observer owns separate records. It never writes supervisor receipts, draws
random numbers, or chooses actions. Repeated deliveries use action identities.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from carnot.agentic.arc_e3_outcome_transport import normalize_action
from carnot.reporting.current_work_receipt import canonical_hash

Json = dict[str, Any]
HORIZON = 120


def visible(frame: Any) -> Json | None:
    """Keep missing levels explicit; identical pixels alone do not mean duplicates."""
    if frame is None:
        return None
    if isinstance(frame, dict):
        return deepcopy(frame)
    return dict(
        level=getattr(frame, "levels_completed", None),
        state=str(getattr(frame, "state", "")),
        full_reset=getattr(frame, "full_reset", False),
    )


class ArcOutcomeObserver:
    """Count distinct action returns independently for each applied redirect."""

    def __init__(self) -> None:
        self.action_index = 0
        self.issues: list[Json] = []
        self._rows: list[Json] = []
        self._queued: list[Json] = []
        self._actions: dict[int, Json] = {}
        self._seen: dict[int, str] = {}

    def redirect(
        self,
        event_id: str,
        arm: str,
        level: int | None,
        *,
        applied: bool,
        legacy: Json | None = None,
    ) -> None:
        """Selection is separate from the wrapper's later action application."""
        if any(r["redirect_id"] == event_id for r in self._rows + self._queued):
            raise ValueError("duplicate_redirect")
        self._queued.append(
            dict(
                redirect_id=event_id,
                arm=arm,
                level_before=level,
                selected=True,
                applied=applied,
                legacy_receipt=deepcopy(legacy),
            )
        )

    def applied(self, action: Any) -> None:
        """Bind queued selections to the exact action returned by choose_action."""
        self.action_index += 1
        normalized = normalize_action(action)
        if normalized["kind"] == "RESET":
            self.finish("reset")
        self._actions[self.action_index] = normalized
        for queued in self._queued:
            row = dict(
                queued,
                application_index=self.action_index,
                application_action=deepcopy(normalized),
                horizon=HORIZON,
                observed_horizon=0,
                horizon_status="pending",
                end_reason=None,
                sequence=[],
                overlapping_redirect_ids=[],
                level_progress=None,
                causal_attribution=None,
            )
            if not row["applied"]:
                self._close(row, "censored", "redirect_not_applied")
            elif normalized["kind"] == "RESET":
                self._close(row, "censored", "reset")
            for previous in self._rows:
                if previous["horizon_status"] == "pending" and row["horizon_status"] == "pending":
                    previous["overlapping_redirect_ids"].append(row["redirect_id"])
                    row["overlapping_redirect_ids"].append(previous["redirect_id"])
            self._rows.append(row)
        self._queued.clear()

    @staticmethod
    def _close(row: Json, status: str, reason: str) -> None:
        row.update(horizon_status=status, end_reason=reason)

    def observe(self, action_index: int, observation: Any) -> None:
        """A gap or conflicting duplicate refuses support instead of inventing it."""
        if action_index == 0:
            return
        if action_index not in self._actions:
            raise ValueError("action_index")
        frame = visible(observation)
        digest = canonical_hash(frame)
        if action_index in self._seen:
            reason = (
                "duplicate_observation"
                if self._seen[action_index] == digest
                else "conflicting_duplicate"
            )
            self.issues.append(dict(action_index=action_index, reason=reason))
            if reason == "conflicting_duplicate":
                self.finish(reason)
            return
        self._seen[action_index] = digest
        for row in self._rows:
            if row["horizon_status"] != "pending":
                continue
            expected = row["application_index"] + row["observed_horizon"]
            row["sequence"].append(
                dict(
                    action_index=action_index,
                    action=deepcopy(self._actions[action_index]),
                    observation=frame,
                    observation_sha256=digest,
                )
            )
            level = None if frame is None else frame.get("level", frame.get("levels_completed"))
            if action_index != expected or frame is None:
                self._close(row, "censored", "missing_observation")
            elif type(level) is not int or type(row["level_before"]) is not int:
                self._close(row, "censored", "missing_level")
            elif frame.get("full_reset") or level < row["level_before"]:
                self._close(row, "censored", "reset")
            else:
                row["observed_horizon"] += 1
                row["level_progress"] = level - row["level_before"]
                if level > row["level_before"]:
                    self._close(row, "level_progress_observed", "level_progress")
                elif str(frame.get("state", "")) in {
                    "WIN",
                    "GAME_OVER",
                    "GameState.WIN",
                    "GameState.GAME_OVER",
                }:
                    self._close(row, "censored", "terminal")
                elif row["observed_horizon"] == HORIZON:
                    self._close(row, "no_progress_within_horizon", "horizon_completed")

    def finish(self, reason: str = "cutoff") -> None:
        """Ending collection cannot turn an unfinished window into a negative."""
        for row in self._rows:
            if row["horizon_status"] == "pending":
                self._close(row, "censored", reason)

    def rows(self) -> list[Json]:
        """Give readers copies so they cannot mutate the live evidence."""
        return deepcopy(self._rows)


def notify(policy: Any, phase: str, value: Any) -> None:
    """Receipt errors remain visible while the original action path continues."""
    observer = getattr(policy, "_arc_outcome_observer", None)
    if observer is None:
        return
    try:
        if phase == "observation":
            observer.observe(observer.action_index, value)
        elif phase == "application":
            observer.applied(value)
        elif phase == "redirect":
            observer.redirect(**value)
        elif phase == "finish":
            observer.finish(value)
        else:
            raise ValueError("observer_phase")
    except Exception as exc:
        observer.issues.append(dict(reason="observer_error", error=type(exc).__name__))
        observer.finish("observer_error")
