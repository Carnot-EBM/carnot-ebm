"""Constraint Memory: persistent cross-session pattern learning (Tier 2).

**Researcher summary:**
    Learns which error patterns repeat across sessions for each domain and
    auto-generates new constraints when a pattern is seen 3+ times. This is
    Tier 2 of Continuous Self-Learning from research-program.md -- where
    Tier 1 adjusts weights of existing constraints, Tier 2 discovers and
    persists entirely new constraints learned from accumulated verifications.

**Detailed explanation for engineers:**
    The core idea: every time the VerifyRepairPipeline catches a constraint
    violation, we don't just fix the current response -- we remember the
    error TYPE and the constraint that caught it, keyed by domain. Over
    many verifications, patterns emerge: "arithmetic domain always has
    carry errors" → auto-generate a carry-check constraint so the next
    arithmetic verification gets it for free.

    The learning lifecycle:
    1. ``record_pattern(domain, error_type, constraint_desc)`` -- called
       after a violation is detected. Increments the (domain, error_type)
       counter and appends the catching constraint description.
    2. ``suggest_constraints(text, domain)`` -- called before extraction.
       Returns ConstraintResult objects for any learned patterns that have
       fired 3+ times for this domain. The pipeline merges these with the
       normal extractor output.
    3. ``save(path)`` / ``load(path)`` -- JSON round-trip for cross-session
       persistence. The memory file is small (pure counters + strings), so
       loading is fast even with thousands of patterns.

    Pattern-to-constraint promotion:
    - A pattern is "mature" when its frequency >= PATTERN_THRESHOLD (default 3).
    - For mature patterns, suggest_constraints() generates a lightweight
      ConstraintResult (no energy_term) with type "learned" and a
      human-readable description reminding the verifier what to check.
    - These suggestions are soft hints: they add to the constraint list but
      don't veto the verification if no violation is found.

    Hardware path (from research-program.md Tier 2):
    - CPU + system memory for storage, FPGA for fast pattern matching
      against constraint template library. The current pure-Python
      implementation runs in CPU RAM; the pattern-match loop is O(patterns)
      per domain and is suitable for FPGA acceleration when the library
      grows large.

Spec: REQ-LEARN-003, SCENARIO-LEARN-003 (Tier 2 Constraint Memory)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from carnot.pipeline.extract import ConstraintResult

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Minimum number of times a (domain, error_type) pattern must appear before
# it is promoted to a suggested constraint.  The value 3 means the system
# waits for evidence before generalising -- a single fluke does not become a
# rule.  Tune upward for less aggressive learning; downward for faster ramp.
PATTERN_THRESHOLD: int = 3


# ---------------------------------------------------------------------------
# Internal data holders
# ---------------------------------------------------------------------------


@dataclass
class _PatternRecord:
    """Counters and examples for one (domain, error_type) pattern.

    **Detailed explanation for engineers:**
        ``frequency`` is the total times this error type was seen in this
        domain. ``constraint_examples`` is a list of human-readable
        descriptions of the constraints that caught it -- used to generate
        the suggested constraint text. We cap examples at MAX_EXAMPLES so the
        file does not grow unboundedly.

        ``auto_generated`` is True once this pattern has been promoted to a
        suggested constraint (frequency >= PATTERN_THRESHOLD). We track this
        separately from just checking frequency so that future code can apply
        promotion logic to a subset of patterns without re-checking the
        threshold everywhere.
    """

    frequency: int = 0
    constraint_examples: list[str] = field(default_factory=list)
    auto_generated: bool = False

    # Maximum constraint description examples to store per pattern.
    MAX_EXAMPLES: int = 10


# ---------------------------------------------------------------------------
# ConstraintMemory
# ---------------------------------------------------------------------------


class ConstraintMemory:
    """Persistent cross-session memory for learned constraint patterns (Tier 2).

    **Researcher summary:**
        Remembers which error types repeat per domain. When a pattern appears
        3+ times, auto-generates a soft constraint suggestion for future
        verifications. Persists to JSON for cross-session accumulation.

    **Detailed explanation for engineers:**
        Storage structure (in memory):
        ``_patterns[domain][error_type] -> _PatternRecord``

        Typical pipeline integration:
        1. Call ``suggest_constraints(text, domain)`` BEFORE extraction to
           prepend learned constraints.
        2. Call ``record_pattern(domain, error_type, constraint_desc)``
           AFTER verification when violations are detected.
        3. Call ``memory.save(path)`` at session end to persist.

        The memory is keyed by (domain, error_type) where:
        - domain: the extraction domain ("arithmetic", "code", "logic", "nl")
        - error_type: the constraint_type of the violation (e.g. "arithmetic",
          "initialization", "implication").

        Example flow for arithmetic:
            memory.record_pattern("arithmetic", "arithmetic", "47 + 28 = 75 (correct: 75)")
            # ... 4 more times ...
            constraints = memory.suggest_constraints("12 + 9 = 21", "arithmetic")
            # returns [ConstraintResult(constraint_type="learned", ...)]

    Spec: REQ-LEARN-003, SCENARIO-LEARN-003
    """

    def __init__(self) -> None:
        # Nested dict: domain -> error_type -> _PatternRecord.
        # Both levels grow lazily on first access.
        self._patterns: dict[str, dict[str, _PatternRecord]] = {}

    # -------------------------------------------------------------------
    # Recording
    # -------------------------------------------------------------------

    def record_pattern(
        self,
        domain: str,
        error_type: str,
        constraint_that_caught_it: str,
    ) -> None:
        """Record that an error of ``error_type`` was caught in ``domain``.

        **Detailed explanation for engineers:**
            Increments the frequency counter for (domain, error_type) and
            appends the catching constraint description as an example.
            Once frequency reaches PATTERN_THRESHOLD, marks the record as
            auto_generated=True so ``suggest_constraints`` will include it.

            This method is idempotent to re-recording the same constraint
            description: if the description is already in constraint_examples
            we still increment frequency (the same constraint type fired
            again) but we do not duplicate the example text.

        Args:
            domain: The verification domain, e.g. "arithmetic", "code".
            error_type: The constraint_type of the violation, e.g.
                "arithmetic", "initialization".
            constraint_that_caught_it: Human-readable description of the
                specific constraint that detected the error (e.g.
                "47 + 28 = 75 (correct: 75)").
        """
        if domain not in self._patterns:
            self._patterns[domain] = {}

        domain_patterns = self._patterns[domain]
        if error_type not in domain_patterns:
            domain_patterns[error_type] = _PatternRecord()

        record = domain_patterns[error_type]
        record.frequency += 1

        # Store the example description (capped to avoid unbounded growth).
        if (
            constraint_that_caught_it not in record.constraint_examples
            and len(record.constraint_examples) < record.MAX_EXAMPLES
        ):
            record.constraint_examples.append(constraint_that_caught_it)

        # Promote to auto-generated when threshold is met.
        if record.frequency >= PATTERN_THRESHOLD:
            record.auto_generated = True

    # -------------------------------------------------------------------
    # Suggestion
    # -------------------------------------------------------------------

    def suggest_constraints(
        self, text: str, domain: str
    ) -> list[ConstraintResult]:
        """Return learned constraint suggestions for ``domain`` based on memory.

        **Detailed explanation for engineers:**
            For each mature pattern (frequency >= PATTERN_THRESHOLD) stored
            under ``domain``, generates a ConstraintResult with:
              - ``constraint_type`` = "learned"
              - ``description`` = a human-readable hint, e.g.
                "Learned: arithmetic errors were seen 5x in 'arithmetic' domain"
              - ``metadata`` = {
                    "domain": domain,
                    "error_type": error_type,
                    "frequency": N,
                    "examples": [list of constraint descriptions],
                    "auto_generated": True,
                }

            The returned constraints have no ``energy_term`` -- they are soft
            hints that the pipeline merges with extractor output. The
            VerifyRepairPipeline will check their metadata["satisfied"] field,
            which is intentionally absent here so they are treated as
            informational rather than violations (unless the pipeline is
            extended to evaluate them further).

            Returns an empty list if no mature patterns exist for the domain.

        Args:
            text: The response text being verified (provided for future
                keyword-matching extensions; currently unused).
            domain: The domain to look up learned patterns for.

        Returns:
            List of ConstraintResult objects for mature patterns. May be empty.
        """
        # Suppress the "unused parameter" concern: text is an intentional
        # forward-compatible hook for future keyword matching against patterns.
        _ = text

        if domain not in self._patterns:
            return []

        suggestions: list[ConstraintResult] = []
        for error_type, record in self._patterns[domain].items():
            if not record.auto_generated:
                # Pattern not yet mature -- skip.
                continue

            suggestions.append(
                ConstraintResult(
                    constraint_type="learned",
                    description=(
                        f"Learned: '{error_type}' errors seen {record.frequency}x"
                        f" in '{domain}' domain -- apply extra scrutiny"
                    ),
                    metadata={
                        "domain": domain,
                        "error_type": error_type,
                        "frequency": record.frequency,
                        "examples": list(record.constraint_examples),
                        "auto_generated": True,
                    },
                )
            )

        return suggestions

    # -------------------------------------------------------------------
    # Persistence
    # -------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Persist learned patterns to a JSON file.

        **Detailed explanation for engineers:**
            Serialises the full pattern table as a versioned JSON document.
            The file is small (one entry per (domain, error_type) pair with
            frequency + examples list) and human-readable for debugging.

            Version 1 format:
            {
              "version": 1,
              "patterns": {
                "<domain>": {
                  "<error_type>": {
                    "frequency": N,
                    "constraint_examples": [...],
                    "auto_generated": true/false
                  }
                }
              }
            }

        Args:
            path: Filesystem path to write. Overwrites if exists.

        Raises:
            OSError: If the path is not writable.
        """
        payload: dict[str, Any] = {
            "version": 1,
            "patterns": {
                domain: {
                    error_type: {
                        "frequency": record.frequency,
                        "constraint_examples": record.constraint_examples,
                        "auto_generated": record.auto_generated,
                    }
                    for error_type, record in domain_patterns.items()
                }
                for domain, domain_patterns in self._patterns.items()
            },
        }
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)

    @classmethod
    def load(cls, path: str) -> "ConstraintMemory":
        """Restore a ConstraintMemory from a JSON file written by save().

        **Detailed explanation for engineers:**
            Reads the file, validates the version field, and reconstructs
            the full pattern table. Returns a fresh ConstraintMemory instance
            populated with the stored patterns.

        Args:
            path: Path to a JSON file previously written by save().

        Returns:
            New ConstraintMemory with patterns loaded from the file.

        Raises:
            OSError: If the file cannot be read.
            ValueError: If the file format is invalid or version is unsupported.
        """
        with open(path, "r", encoding="utf-8") as fh:
            payload = json.load(fh)

        if not isinstance(payload, dict) or payload.get("version") != 1:
            raise ValueError(
                f"Unsupported memory file format (expected version=1): {path}"
            )

        memory = cls()
        for domain, domain_patterns in payload.get("patterns", {}).items():
            memory._patterns[domain] = {}
            for error_type, raw in domain_patterns.items():
                memory._patterns[domain][error_type] = _PatternRecord(
                    frequency=int(raw.get("frequency", 0)),
                    constraint_examples=list(raw.get("constraint_examples", [])),
                    auto_generated=bool(raw.get("auto_generated", False)),
                )
        return memory

    # -------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------

    def summary(self) -> dict[str, Any]:
        """Return a per-domain summary of pattern counts and top patterns.

        **Detailed explanation for engineers:**
            Returns a dict keyed by domain. Each value has:
              - "total_patterns": total distinct (error_type) entries for
                the domain (both immature and mature).
              - "mature_patterns": count of patterns that have been promoted
                (frequency >= PATTERN_THRESHOLD).
              - "top_patterns": list of up to 5 (error_type, frequency) pairs
                sorted by frequency descending -- the most commonly seen error
                types in this domain.

            Useful for logging, operator dashboards, and feeding into research
            analysis scripts to understand what the system has learned.

        Returns:
            Dict of {domain: {summary_key: value, ...}}.
        """
        result: dict[str, Any] = {}
        for domain, domain_patterns in self._patterns.items():
            # Sort by frequency descending to find top patterns.
            sorted_patterns = sorted(
                domain_patterns.items(),
                key=lambda kv: kv[1].frequency,
                reverse=True,
            )
            top_patterns = [
                {"error_type": et, "frequency": rec.frequency}
                for et, rec in sorted_patterns[:5]
            ]
            result[domain] = {
                "total_patterns": len(domain_patterns),
                "mature_patterns": sum(
                    1 for rec in domain_patterns.values() if rec.auto_generated
                ),
                "top_patterns": top_patterns,
            }
        return result

    def __repr__(self) -> str:
        n_domains = len(self._patterns)
        n_patterns = sum(len(dp) for dp in self._patterns.values())
        return f"ConstraintMemory(domains={n_domains}, patterns={n_patterns})"
