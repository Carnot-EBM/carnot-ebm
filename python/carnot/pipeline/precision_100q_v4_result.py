"""Precision100qV4Result and CoTPairCollector — Exp 476 result types.

**Why a new V4 result class (Exp 476, RETRO-033 third attempt):**
    Exp 464 (V3) defined `Precision100qResult` with `confidence_interval_95` but the
    field name in the task spec is `ci_95_wilson`.  This V4 version renames the
    property for consistency with the task spec and the test contract, and renames
    `n_questions` → `n` to match the Precision100qV4Result constructor signature.

    `CoTPairCollector` replaces the raw `cot_pairs: list[dict]` accumulation pattern
    from Exp 464.  It adds an atomic flush step (write to `.tmp` then rename) that
    mirrors `_write_json()` but is encapsulated for testability.

**CoTPairCollector.flush() atomicity guarantee:**
    The collector writes to `<output_path>.tmp` first, then calls `Path.replace()`
    which is atomic on POSIX.  If the process is killed between write and rename,
    the `.tmp` file is left behind (safe to delete) and the primary path is absent
    (guarded by DeliverableGuard on the main artifact).

Spec: REQ-BENCH-025, REQ-BENCH-027,
      SCENARIO-BENCH-044, SCENARIO-BENCH-046
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

# z-score for 95% two-sided confidence interval (1.96)
_Z95 = 1.959963984540054


# ---------------------------------------------------------------------------
# Precision100qV4Result
# ---------------------------------------------------------------------------


@dataclass
class Precision100qV4Result:
    """Per-model result from the 100-question live precision benchmark (Exp 476).

    Parameters
    ----------
    model_id : str
        Human-readable model name, e.g. 'Gemma4-E4B-it'.
    pre_accuracy : float
        Fraction correct with NO pipeline applied (baseline). Range [0.0, 1.0].
    post_accuracy : float
        Fraction correct WITH pipeline applied. Range [0.0, 1.0].
    n : int
        Number of questions in the benchmark run. Must be > 0.
    extractor_used : str
        Comma-separated names of extractors that produced violations.
        'none' when no violations were detected.
    inference_mode : str
        'live_gpu' when running on real hardware; 'synthetic' for fallback questions.

    Computed properties
    -------------------
    signed_improvement : float
        post_accuracy - pre_accuracy (unclamped, may be negative).
    ci_95_wilson : tuple[float, float]
        Wilson score 95% CI on post_accuracy.  Always within [0.0, 1.0].
    is_positive : bool
        True iff signed_improvement > 0 (strict).

    Spec: REQ-BENCH-025, SCENARIO-BENCH-044
    """

    model_id: str
    pre_accuracy: float
    post_accuracy: float
    n: int
    extractor_used: str
    inference_mode: str

    @property
    def signed_improvement(self) -> float:
        """post_accuracy minus pre_accuracy — never clamp, negative means regression."""
        return self.post_accuracy - self.pre_accuracy

    @property
    def ci_95_wilson(self) -> tuple[float, float]:
        """Wilson score 95% confidence interval on post_accuracy.

        Formula (Agresti & Coull, 1998):
            center = (p̂ + z²/(2n)) / (1 + z²/n)
            margin = z * sqrt(p̂*(1-p̂)/n + z²/(4n²)) / (1 + z²/n)

        Returns (lower, upper) clamped to [0.0, 1.0].  At n=100 the interval
        width is typically < 0.10, making directional claims statistically credible.

        Spec: REQ-BENCH-025, SCENARIO-BENCH-044
        """
        p = self.post_accuracy
        n = max(self.n, 1)
        z = _Z95
        z2 = z * z

        center = (p + z2 / (2 * n)) / (1 + z2 / n)
        margin = (z * math.sqrt(p * (1 - p) / n + z2 / (4 * n * n))) / (1 + z2 / n)

        lower = max(0.0, center - margin)
        upper = min(1.0, center + margin)
        return (lower, upper)

    @property
    def is_positive(self) -> bool:
        """True iff signed_improvement > 0 (strict greater-than; zero is not positive)."""
        return self.signed_improvement > 0

    def to_dict(self) -> dict:
        """Return a JSON-serializable dict including all computed properties."""
        lo, hi = self.ci_95_wilson
        return {
            "model_id": self.model_id,
            "pre_accuracy": self.pre_accuracy,
            "post_accuracy": self.post_accuracy,
            "n": self.n,
            "extractor_used": self.extractor_used,
            "inference_mode": self.inference_mode,
            "signed_improvement": self.signed_improvement,
            "ci_95_wilson": [lo, hi],
            "is_positive": self.is_positive,
        }


# ---------------------------------------------------------------------------
# CoTPairCollector
# ---------------------------------------------------------------------------


@dataclass
class CoTPairCollector:
    """Collect Chain-of-Thought pairs during a benchmark and flush atomically.

    Each pair records one (question, cot_text, correct) triple from a benchmark
    run.  The collector accumulates pairs in memory and writes them all at once
    via `flush()`, using a `.tmp` → rename pattern for POSIX atomicity.

    Parameters
    ----------
    output_path : str
        Destination JSON path.  The parent directory must exist before flush().

    Usage
    -----
    ::

        collector = CoTPairCollector('results/exp476_cot_pairs.json')
        for q, cot, ok in pairs:
            collector.add(model_id, q, cot, ok)
        n_written = collector.flush()

    Spec: REQ-BENCH-027, SCENARIO-BENCH-046
    """

    output_path: str
    _pairs: list[dict] = field(default_factory=list, init=False, repr=False)

    def add(self, model: str, question: str, cot: str, correct: bool) -> None:
        """Append one CoT pair to the in-memory buffer.

        Parameters
        ----------
        model : str
            Model name that generated this CoT (e.g. 'Gemma4-E4B-it').
        question : str
            The benchmark question text.
        cot : str
            The model's chain-of-thought response.
        correct : bool
            True if the response contained the correct final answer.
        """
        self._pairs.append({
            "model": model,
            "question": question,
            "cot_text": cot,
            "correct": correct,
        })

    def flush(self) -> int:
        """Write all accumulated pairs to disk atomically and return the count.

        Writes to `<output_path>.tmp` first, then renames to `output_path`.
        This is POSIX-atomic: the destination is either the old file or the
        new file — never a partially-written state.

        Returns
        -------
        int
            Number of pairs written.  0 if the buffer was empty.
        """
        out = Path(self.output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = Path(str(out) + ".tmp")
        tmp.write_text(json.dumps(self._pairs, indent=2))
        tmp.replace(out)
        return len(self._pairs)
