"""ModelAdaptiveThresholdGate: per-(model, constraint_type) precision gate.

**Researcher summary:**
    Exp 706 identified that Gemma4-E4B-it's VR failure mode was
    "threshold_too_high" rather than "extraction_fp", but the gate architecture
    is designed to handle both cases.  When a constraint type fires too many
    false positives on a specific model, this gate suppresses that
    (model_id, constraint_type) pair so the pipeline stops harming correct
    responses with noisy extractors.

**Detailed explanation for engineers:**
    This is a Tier 1 self-learning component per research-program.md:
    "Upweight constraints that catch real errors, downweight noisy ones."

    The extension here is per-MODEL tracking instead of global per-constraint-type:
    the same extractor might be precise on model A but noisy on model B
    (different token distributions, different chain-of-thought patterns).

    The gate maintains a state dict keyed by (model_id, constraint_type) pairs.
    For each pair it counts:
      - tp_count: times extractor fired AND violation was real (TP)
      - fp_count: times extractor fired AND response was already correct (FP)

    precision = tp_count / (tp_count + fp_count)

    If precision < 0.5 AND there is at least one observation, the constraint
    type is suppressed for that model.  Zero observations → default allow
    (we have no evidence to suppress).

    State persists to a JSON file so learning accumulates across sessions.
    Writes are atomic (write-then-rename) to avoid corrupt state on crash.

Spec: REQ-VERIFY-146, REQ-VERIFY-147, SCENARIO-VERIFY-146, SCENARIO-VERIFY-147
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any


class ModelAdaptiveThresholdGate:
    """Gate that suppresses constraint types with precision < 0.5 per model.

    The gate is intentionally stateful: update() is called after each
    verify-repair cycle to record whether the extractor firing was a TP or FP,
    and is_suppressed() is consulted at the START of each cycle to decide
    whether to skip a constraint type.

    Precision falls below 0.5 when there are more false positives than true
    positives — meaning the extractor is hurting more than helping for that
    (model_id, constraint_type) pair.  The gate then silences it.
    """

    # Default location for persisted state.  The results/ directory is
    # already gitignored for large artefacts so this is a safe landing spot.
    DEFAULT_STATE_FILE = Path("results/adaptive_gate_state.json")

    def __init__(self, state_file: Path | None = None) -> None:
        # {model_id: {constraint_type: {"tp": int, "fp": int}}}
        self.state: dict[str, dict[str, dict[str, int]]] = {}
        self.state_file: Path = state_file if state_file is not None else self.DEFAULT_STATE_FILE

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(self, model_id: str, constraint_type: str, was_tp: bool) -> None:
        """Record one observation for (model_id, constraint_type) and persist.

        Call this after each verify-repair cycle where the extractor fired.
        ``was_tp=True`` when the extractor correctly caught a real violation;
        ``was_tp=False`` when the extractor fired on a response that was
        already correct (a false positive that may trigger a bad repair).
        """
        if model_id not in self.state:
            self.state[model_id] = {}
        if constraint_type not in self.state[model_id]:
            self.state[model_id][constraint_type] = {"tp": 0, "fp": 0}
        if was_tp:
            self.state[model_id][constraint_type]["tp"] += 1
        else:
            self.state[model_id][constraint_type]["fp"] += 1
        self.save()

    def is_suppressed(self, model_id: str, constraint_type: str) -> bool:
        """Return True if this constraint type should be skipped for this model.

        Suppression requires BOTH conditions to hold:
          1. We have at least one observation (tp + fp > 0).
          2. Precision is below 0.5 (more FPs than TPs).

        Zero observations → False (default allow — no evidence to suppress).
        """
        p = self._raw_precision(model_id, constraint_type)
        if p is None:
            return False  # no data → allow
        return p < 0.5

    def precision(self, model_id: str, constraint_type: str) -> float:
        """Return precision for (model_id, constraint_type), or 0.5 if no data.

        0.5 is the neutral/uninformative prior — exactly at the suppression
        threshold, so no suppression occurs when there are no observations.
        """
        p = self._raw_precision(model_id, constraint_type)
        return 0.5 if p is None else p

    def save(self) -> None:
        """Atomically write gate state to state_file.

        We write to a temp file in the same directory then rename so a crash
        mid-write cannot leave a corrupt JSON file.
        """
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        # Write to a sibling temp file then atomically rename.
        fd, tmp_path = tempfile.mkstemp(
            dir=self.state_file.parent, suffix=".tmp"
        )
        try:
            with os.fdopen(fd, "w") as fh:
                json.dump(self.state, fh, indent=2)
            os.replace(tmp_path, self.state_file)
        except Exception:
            # Clean up the temp file if anything went wrong.
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise

    def load(self) -> None:
        """Load gate state from state_file if it exists.

        If the file does not exist, state stays empty (fresh start).
        If the file is malformed, raises json.JSONDecodeError so callers
        can decide whether to treat that as a fatal error or reset state.
        """
        if not self.state_file.exists():
            return
        with self.state_file.open() as fh:
            loaded: Any = json.load(fh)
        # Validate top-level structure is a dict.
        if not isinstance(loaded, dict):
            raise ValueError(f"Expected dict in {self.state_file}, got {type(loaded)}")
        self.state = loaded

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _raw_precision(self, model_id: str, constraint_type: str) -> float | None:
        """Return tp/(tp+fp) or None if no observations exist."""
        entry = self.state.get(model_id, {}).get(constraint_type)
        if entry is None:
            return None
        tp = entry.get("tp", 0)
        fp = entry.get("fp", 0)
        total = tp + fp
        if total == 0:
            return None
        return tp / total
