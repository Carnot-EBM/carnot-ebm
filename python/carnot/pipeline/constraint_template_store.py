"""AutoRefine-style constraint template distillation and retrieval.

**Researcher summary (AutoRefine, arXiv 2601.22758):**
    AutoRefine converts agent interaction trajectories into reusable abstract
    strategic principles via offline self-distillation.  For Carnot, this maps
    to: verify-repair trajectories (violation_type, context, was_repaired) are
    collected in a ConstraintTemplateStore.  When enough observations of the
    same violation family accumulate (min_observations=3), the store
    ``distill()`` call promotes that pattern to a named ConstraintTemplate.
    At inference time, ``retrieve(query_context)`` fetches the top-K templates
    whose context keywords best overlap with the current query, allowing the
    pipeline to apply domain-specific constraints only when the context matches.

**Key benefit over simple threshold-based addition:**
    Simple addition fires on violation TYPE alone.  Template retrieval fires on
    violation TYPE + CONTEXT SIMILARITY.  This reduces false positives from
    applying a "carry-arithmetic" constraint to a code-generation query.

**Data flow:**
    1. Pipeline calls ``add_violation(violation_type, context)`` after each repair.
    2. Offline step calls ``distill(min_observations=3)`` → list of ConstraintTemplate.
    3. At inference time, ``retrieve(query_context, top_k=3)`` → ranked templates.
    4. Store is persisted across sessions via ``save(path)`` / ``load(path)``.

Spec: REQ-LEARN-058, REQ-LEARN-059,
SCENARIO-LEARN-090, SCENARIO-LEARN-091, SCENARIO-LEARN-092
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# ConstraintTemplate dataclass
# ---------------------------------------------------------------------------


@dataclass
class ConstraintTemplate:
    """One distilled constraint guard derived from recurring violation patterns.

    A ConstraintTemplate is lightweight and CPU-safe — no matrix operations,
    no model calls.  It encodes WHAT violation to guard against (violation_type),
    WHEN to fire it (context_keywords), and HOW to describe the constraint to
    downstream pipeline components (constraint_text).

    Fields
    ------
    name
        Human-readable identifier, e.g. ``"carry_arithmetic_guard"``.  Derived
        from violation_type by the store's distillation logic.
    violation_type
        The primary error class that caused this template to be created
        (e.g. ``"carry"``, ``"semantic"``).
    context_keywords
        List of lowercase keywords extracted from the context strings of all
        observed violations that fed this template.  Used by ``retrieve()`` to
        score keyword overlap against a query context.
    constraint_text
        Human-readable constraint description that can be injected into a
        prompt or passed to a verifier as a guard clause.
    n_violations_observed
        Total number of violation observations that fed this template.  Used
        to rank templates and to report how mature a pattern is.
    """

    name: str
    violation_type: str
    context_keywords: list[str]
    constraint_text: str
    n_violations_observed: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "violation_type": self.violation_type,
            "context_keywords": self.context_keywords,
            "constraint_text": self.constraint_text,
            "n_violations_observed": self.n_violations_observed,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ConstraintTemplate:
        return cls(
            name=str(payload.get("name") or ""),
            violation_type=str(payload.get("violation_type") or ""),
            context_keywords=list(payload.get("context_keywords") or []),
            constraint_text=str(payload.get("constraint_text") or ""),
            n_violations_observed=int(payload.get("n_violations_observed") or 0),
        )


# ---------------------------------------------------------------------------
# Internal violation accumulator
# ---------------------------------------------------------------------------


@dataclass
class _ViolationAccumulator:
    """Private accumulator for a single violation type.

    Stores the running count and the union of context keyword tokens seen
    across all observations.  NOT part of the public API — ConstraintTemplateStore
    uses this internally and exposes only ConstraintTemplate objects externally.

    Fields
    ------
    violation_type
        The primary error class (e.g. ``"carry"``).
    count
        Total number of add_violation() calls for this type.
    keyword_counts
        Token → frequency map built from all context strings observed.
        Used to select the top keywords for the distilled template.
    """

    violation_type: str
    count: int = 0
    keyword_counts: dict[str, int] = field(default_factory=dict)

    def observe(self, context: str) -> None:
        """Record one observation with the given context string.

        Tokenises context by whitespace (lowercased) and increments keyword
        counts.  This is intentionally simple — no stemming, no stop-word
        removal — to keep the CPU cost negligible.
        """
        self.count += 1
        for token in context.lower().split():
            self.keyword_counts[token] = self.keyword_counts.get(token, 0) + 1

    def top_keywords(self, n: int = 10) -> list[str]:
        """Return the n most frequently observed keywords, sorted by frequency desc."""
        return [
            kw
            for kw, _ in sorted(
                self.keyword_counts.items(), key=lambda kv: (-kv[1], kv[0])
            )[:n]
        ]


# ---------------------------------------------------------------------------
# ConstraintTemplateStore
# ---------------------------------------------------------------------------

# Template names follow a simple pattern: "<violation_type>_guard"
# WHY a fixed naming scheme: predictable names make it easy for pipeline code
# to detect whether a template has already been applied to a given query.
_NAME_SUFFIX = "_guard"

# Maximum context keywords to embed in each distilled template.
# Keeps the keyword list short enough for fast overlap scoring.
_MAX_KEYWORDS = 10

# Schema version for serialised store files.  Bump when the JSON structure
# changes in a backwards-incompatible way.
_STORE_SCHEMA_VERSION = 1


class ConstraintTemplateStore:
    """Accumulate violation observations and distil them into retrievable templates.

    **Usage pattern (offline self-distillation):**

    1. Ingest violation trajectories::

          store = ConstraintTemplateStore()
          for vtype, ctx in trajectory_pairs:
              store.add_violation(vtype, ctx)

    2. Distil patterns that have matured::

          templates = store.distill(min_observations=3)

    3. Retrieve at inference time::

          relevant = store.retrieve("arithmetic carry sum", top_k=3)

    4. Persist across sessions::

          store.save("results/constraint_templates_546.json")
          store2 = ConstraintTemplateStore.load("results/constraint_templates_546.json")

    Thread safety
    -------------
    Not thread-safe.  Each experiment should use a single-threaded store and
    serialise access if used from multiple threads.

    Spec: REQ-LEARN-058, REQ-LEARN-059,
    SCENARIO-LEARN-090, SCENARIO-LEARN-091, SCENARIO-LEARN-092
    """

    def __init__(self) -> None:
        # violation_type → _ViolationAccumulator
        self._accumulators: dict[str, _ViolationAccumulator] = {}
        # Cached distilled templates (invalidated on add_violation())
        self._distilled: list[ConstraintTemplate] | None = None

    # ------------------------------------------------------------------
    # add_violation
    # ------------------------------------------------------------------

    def add_violation(self, violation_type: str, context: str) -> None:
        """Record one violation observation.

        Accumulates the count for violation_type and extracts keywords from
        context.  Calling this invalidates any previously cached distill()
        result so that the next distill() reflects all observations.

        Parameters
        ----------
        violation_type
            Primary error class (e.g. ``"carry"``, ``"semantic"``).  Should
            be a single token or colon-separated label; only the portion before
            the first colon is used as the key.
        context
            Free-text description of the query context in which the violation
            was observed (e.g. the question text, domain label, or benchmark
            slice).  Tokenised by whitespace for keyword extraction.

        Spec: REQ-LEARN-058, SCENARIO-LEARN-090
        """
        # Normalise: take the prefix before any colon separator so that
        # "carry:overflow" and "carry" both accumulate under "carry".
        vtype = violation_type.split(":", 1)[0].strip()
        if not vtype:
            return

        if vtype not in self._accumulators:
            self._accumulators[vtype] = _ViolationAccumulator(violation_type=vtype)
        self._accumulators[vtype].observe(context)

        # Invalidate the distilled cache so the next distill() re-runs.
        self._distilled = None

    # ------------------------------------------------------------------
    # distill
    # ------------------------------------------------------------------

    def distill(self, *, min_observations: int = 3) -> list[ConstraintTemplate]:
        """Produce ConstraintTemplate objects for all mature violation patterns.

        A pattern is "mature" when its observation count >= min_observations.
        The distilled templates are cached until the next add_violation() call.

        Parameters
        ----------
        min_observations
            Minimum number of add_violation() calls required before a violation
            type is promoted to a template.  Default 3 — matches the AutoRefine
            paper's minimum-support threshold for reliable generalisation.

        Returns
        -------
        list[ConstraintTemplate]
            One template per mature violation type, sorted by violation_type for
            deterministic output.

        Spec: REQ-LEARN-058, SCENARIO-LEARN-090, SCENARIO-LEARN-091
        """
        if self._distilled is not None:
            # Return the cached result — no new observations since last distill.
            return list(self._distilled)

        templates: list[ConstraintTemplate] = []
        for vtype, acc in sorted(self._accumulators.items()):
            if acc.count < min_observations:
                continue

            keywords = acc.top_keywords(_MAX_KEYWORDS)
            name = f"{vtype}{_NAME_SUFFIX}"
            constraint_text = (
                f"Guard against recurring '{vtype}' violations. "
                f"Observed {acc.count} times. "
                f"Key context signals: {', '.join(keywords[:5]) if keywords else 'none'}."
            )
            templates.append(
                ConstraintTemplate(
                    name=name,
                    violation_type=vtype,
                    context_keywords=keywords,
                    constraint_text=constraint_text,
                    n_violations_observed=acc.count,
                )
            )

        self._distilled = templates
        return list(templates)

    # ------------------------------------------------------------------
    # retrieve
    # ------------------------------------------------------------------

    def retrieve(self, query_context: str, *, top_k: int = 3) -> list[ConstraintTemplate]:
        """Retrieve the top-K distilled templates most relevant to query_context.

        Relevance is scored by keyword overlap: count of template keywords that
        appear as substrings in the lowercased query_context.  Templates with
        equal scores are sorted by violation_type for determinism.

        If fewer templates exist than top_k, all templates are returned.  If
        distill() has not been called, it is called automatically with
        min_observations=1 so that any observed pattern is retrievable.

        Parameters
        ----------
        query_context
            Free-text description of the current inference context (e.g. the
            question text or domain label).
        top_k
            Maximum number of templates to return.

        Returns
        -------
        list[ConstraintTemplate]
            Up to top_k templates, ranked by keyword overlap (descending).

        Spec: REQ-LEARN-059, SCENARIO-LEARN-092
        """
        # Auto-distill with min_observations=1 if cache is empty.
        # WHY: retrieve() must be callable at any time without a prior explicit
        # distill() call.  Using min_observations=1 ensures any pattern that
        # has been observed at all is retrievable (conservative).
        if self._distilled is None:
            self.distill(min_observations=1)

        templates = self._distilled or []
        if not templates:
            return []

        lowered_query = query_context.lower()

        def _overlap_score(tmpl: ConstraintTemplate) -> int:
            return sum(1 for kw in tmpl.context_keywords if kw in lowered_query)

        ranked = sorted(
            templates,
            key=lambda t: (-_overlap_score(t), t.violation_type),
        )
        return ranked[:top_k]

    # ------------------------------------------------------------------
    # save / load
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        """Persist the store's accumulators and distilled templates to a JSON file.

        The file includes:
        - ``schema_version``: integer for future compatibility checks.
        - ``accumulators``: raw observation data (count + keyword_counts) so
          the store can be rehydrated and add_violation() can continue.
        - ``distilled_templates``: the last cached distillation result (or
          empty list if distill() was never called).

        Parameters
        ----------
        path
            Destination file path.  Parent directories must exist.

        Spec: REQ-LEARN-058
        """
        payload: dict[str, Any] = {
            "schema_version": _STORE_SCHEMA_VERSION,
            "accumulators": {
                vtype: {
                    "violation_type": acc.violation_type,
                    "count": acc.count,
                    "keyword_counts": acc.keyword_counts,
                }
                for vtype, acc in sorted(self._accumulators.items())
            },
            "distilled_templates": [
                t.to_dict() for t in (self._distilled or [])
            ],
        }
        Path(path).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

    @classmethod
    def load(cls, path: str | Path) -> ConstraintTemplateStore:
        """Load a store from a previously saved JSON file.

        Rehydrates both the raw accumulators (so add_violation() can resume)
        and the cached distilled templates (so retrieve() works immediately
        without re-running distill()).

        Parameters
        ----------
        path
            Path to the JSON file written by save().

        Returns
        -------
        ConstraintTemplateStore
            Fully rehydrated store instance.

        Raises
        ------
        ValueError
            If the file's schema_version is not supported.

        Spec: REQ-LEARN-058
        """
        raw = json.loads(Path(path).read_text())
        version = raw.get("schema_version")
        if version is not None and int(version) != _STORE_SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported ConstraintTemplateStore schema_version "
                f"(expected {_STORE_SCHEMA_VERSION}, got {version})"
            )

        store = cls()

        for vtype, acc_data in raw.get("accumulators", {}).items():
            acc = _ViolationAccumulator(
                violation_type=str(acc_data.get("violation_type") or vtype),
                count=int(acc_data.get("count") or 0),
                keyword_counts=dict(acc_data.get("keyword_counts") or {}),
            )
            store._accumulators[vtype] = acc

        raw_distilled = raw.get("distilled_templates")
        if raw_distilled is not None:
            store._distilled = [
                ConstraintTemplate.from_dict(item)
                for item in raw_distilled
                if isinstance(item, dict)
            ]

        return store

    # ------------------------------------------------------------------
    # Inspection
    # ------------------------------------------------------------------

    def violation_counts(self) -> dict[str, int]:
        """Return a snapshot of current observation counts per violation type.

        Returns a copy to protect internal state from mutation.
        """
        return {vtype: acc.count for vtype, acc in self._accumulators.items()}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    "ConstraintTemplate",
    "ConstraintTemplateStore",
]
