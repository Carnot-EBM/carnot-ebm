"""MetaJuLS-style online adapter for StructuredEquationForcer's system prompt.

**Why this module exists (Exp 676, Tier 2 Constraint Memory):**

    StructuredEquationForcer (Exp 653) forces arithmetic into COMPUTE: format
    via a STATIC system prompt addendum (FORCER_SYSTEM_ADDENDUM).  The addendum
    works well on average but recall varies by problem domain: percentage problems
    and multi-step chain-of-thought diverge more than simple arithmetic.

    MetaJuLS (arXiv 2601.00095) proposes meta-RL constraint propagation policies
    that adapt per-task without full retraining.  We apply that idea here:
    after each live inference session, recall observations are logged per domain,
    and when a domain's mean recall drops below 0.30 the adapter upgrades its
    addendum to a stronger CRITICAL: variant.

    This is Tier 2 continuous self-learning (Constraint Memory) applied to the
    generation-layer forcer rather than the post-hoc extractor.

**Adaptation policy:**

    If mean recall for a domain < 0.30:
        Domain emphasis → "CRITICAL: You MUST write every arithmetic step as
        COMPUTE: <operand> <op> <operand> = <result>.  This is non-optional."
    Else:
        Domain emphasis stays at the base addendum.

**Thread safety:** not thread-safe.  Wrap with an external lock when used
concurrently across multiple inference threads.

Spec: REQ-LEARN-085, REQ-LEARN-086,
      SCENARIO-LEARN-133, SCENARIO-LEARN-134, SCENARIO-LEARN-135
"""

from __future__ import annotations

import json
import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# ForcingFeedback — one observation from a live inference session
# ---------------------------------------------------------------------------


@dataclass
class ForcingFeedback:
    """Observation from a single question processed under the forcing prompt.

    Fields:
        question              — The input question text (for audit traceability).
        compute_lines_found   — Number of COMPUTE: lines extracted from the response.
        total_arithmetic_ops  — Number of raw arithmetic expressions detected
                                (denominator for recall).
        recall                — compute_lines_found / max(total_arithmetic_ops, 1).
                                1.0 means every arithmetic op was in COMPUTE: format.
        domain                — Problem domain label, e.g. 'arithmetic', 'percentage',
                                'multi_step'.  Used to track per-domain recall trends.
    """

    question: str
    compute_lines_found: int
    total_arithmetic_ops: int
    recall: float
    domain: str


# ---------------------------------------------------------------------------
# MetaJuLSForcingAdapter — online forcing strategy updater
# ---------------------------------------------------------------------------

# The critical-emphasis template is injected when a domain's mean recall
# falls below the LOW_RECALL_THRESHOLD.  The all-caps CRITICAL and non-optional
# phrasing increase instruction salience for RLHF-tuned models.
_CRITICAL_EMPHASIS_TEMPLATE: str = (
    "CRITICAL: You MUST write every arithmetic step as "
    "COMPUTE: <left_operand> <operator> <right_operand> = <result>.  "
    "This is non-optional.  Do not skip this format for {domain} problems."
)

# Below this mean recall, a domain triggers the stronger addendum.
LOW_RECALL_THRESHOLD: float = 0.30


class MetaJuLSForcingAdapter:
    """Online meta-RL adapter that updates StructuredEquationForcer's forcing prompt.

    **How adaptation works:**

        1. Call update(feedback) after each live inference session.
        2. The adapter accumulates recall observations keyed by domain.
        3. When a domain's mean recall drops below LOW_RECALL_THRESHOLD (0.30),
           the adapter stores a CRITICAL: emphasis string for that domain.
        4. get_adapted_addendum(question, domain) returns the base addendum
           concatenated with any domain-specific CRITICAL: emphasis.

    **Why 0.30 as the threshold:**
        Exp 668 VR #18 measured 36% baseline accuracy.  Domains that fall below
        30% recall are producing fewer COMPUTE: lines than the system-level
        baseline, signalling that the base addendum is insufficient for that
        domain type.

    **Persistence:**
        save_state() / load_state() serialise the accumulated recall history
        and emphasis strings to JSON so adaptation survives process restarts.

    Args:
        base_addendum   : The static FORCER_SYSTEM_ADDENDUM from structured_equation_forcer.py.
        learning_rate   : Reserved for future gradient-based extensions.  Not used in
                          the current rule-based update — kept for API forward compatibility.

    Spec: REQ-LEARN-085, REQ-LEARN-086
    """

    def __init__(
        self,
        base_addendum: str,
        learning_rate: float = 0.1,
    ) -> None:
        self._base_addendum: str = base_addendum
        self._learning_rate: float = learning_rate
        # domain → list of recall floats (grows monotonically, never trimmed)
        self._domain_recalls: dict[str, list[float]] = {}
        # domain → CRITICAL: emphasis string (set when mean recall < threshold)
        self._domain_emphasis: dict[str, str] = {}

    # ------------------------------------------------------------------
    # update
    # ------------------------------------------------------------------

    def update(self, feedback: ForcingFeedback) -> None:
        """Update forcing strategy from one observation.  Low recall → stronger addendum.

        Records the recall observation for this domain.  If the running mean falls
        below LOW_RECALL_THRESHOLD, installs a CRITICAL: emphasis for this domain
        so future calls to get_adapted_addendum return a stronger prompt.

        Args:
            feedback: ForcingFeedback instance from a single forced-equation run.
        """
        domain = feedback.domain
        if domain not in self._domain_recalls:
            self._domain_recalls[domain] = []

        self._domain_recalls[domain].append(feedback.recall)

        mean_recall = statistics.mean(self._domain_recalls[domain])
        if mean_recall < LOW_RECALL_THRESHOLD:
            # Domain recall is consistently below the threshold — escalate to CRITICAL.
            self._domain_emphasis[domain] = _CRITICAL_EMPHASIS_TEMPLATE.format(
                domain=domain
            )
        else:
            # Recall is acceptable — remove any previously installed emphasis so we
            # do not permanently over-force domains that have recovered.
            self._domain_emphasis.pop(domain, None)

    # ------------------------------------------------------------------
    # get_adapted_addendum
    # ------------------------------------------------------------------

    def get_adapted_addendum(self, question: str, domain: Optional[str] = None) -> str:
        """Return FORCER_SYSTEM_ADDENDUM adapted for this question's domain.

        If no domain-specific emphasis has been installed, returns the base addendum
        unchanged.  When a domain has triggered the CRITICAL: upgrade, appends the
        emphasis block so the model receives both the general and domain-specific
        instructions.

        Args:
            question : Question text (unused in rule-based policy; included for
                       API forward compatibility with future content-based routing).
            domain   : Optional domain label.  When None, returns only the base
                       addendum (no domain context for lookup).

        Returns:
            System prompt addendum string, potentially augmented with domain emphasis.
        """
        if domain is None or domain not in self._domain_emphasis:
            return self._base_addendum

        return self._base_addendum + "\n" + self._domain_emphasis[domain]

    # ------------------------------------------------------------------
    # save_state / load_state
    # ------------------------------------------------------------------

    def save_state(self, path: str) -> None:
        """Persist domain recall history and emphasis strings to a JSON file.

        The saved state is a plain dict so it can be inspected and edited without
        special tooling.  Float precision is preserved by Python's json encoder.

        Args:
            path: File system path for the JSON output (will be created or overwritten).
        """
        state = {
            "base_addendum": self._base_addendum,
            "learning_rate": self._learning_rate,
            "domain_recalls": self._domain_recalls,
            "domain_emphasis": self._domain_emphasis,
        }
        Path(path).write_text(json.dumps(state, indent=2))

    @classmethod
    def load_state(cls, path: str) -> "MetaJuLSForcingAdapter":
        """Restore a MetaJuLSForcingAdapter from a previously saved state file.

        All accumulated recall history and installed emphasis strings are restored
        exactly, so adaptation continues from where it left off without re-processing
        historical observations.

        Args:
            path: File system path of the JSON state file written by save_state().

        Returns:
            MetaJuLSForcingAdapter with restored internal state.
        """
        state = json.loads(Path(path).read_text())
        adapter = cls(
            base_addendum=state["base_addendum"],
            learning_rate=state["learning_rate"],
        )
        adapter._domain_recalls = state["domain_recalls"]
        adapter._domain_emphasis = state["domain_emphasis"]
        return adapter

    # ------------------------------------------------------------------
    # properties for inspection
    # ------------------------------------------------------------------

    @property
    def domain_recalls(self) -> dict[str, list[float]]:
        """Read-only view of accumulated per-domain recall observations."""
        return {domain: list(recalls) for domain, recalls in self._domain_recalls.items()}

    @property
    def domain_emphasis(self) -> dict[str, str]:
        """Read-only view of currently installed per-domain emphasis strings."""
        return dict(self._domain_emphasis)
