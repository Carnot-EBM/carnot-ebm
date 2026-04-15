"""CoTCircuitVerifier: circuit-based reasoning verification (arXiv 2510.09312).

**Researcher summary:**
    CRV extracts the computational dependency graph from a chain-of-thought
    response — which intermediate values depend on which others — and then
    checks structural consistency.  A "broken circuit" means a downstream
    step uses a value that does not match the upstream step's actual output.

    This is complementary to Z3 (which checks arithmetic consistency):
    - Z3:  "Is 47 + 28 = 76 a valid arithmetic statement?" → No
    - CRV: "Does step 5's answer follow from step 3 and step 4?"  → structural check

    CRV catches reasoning chain errors that regex and Z3 miss: cases where each
    step is individually valid but the chain is inconsistent (wrong variable
    substituted, result from wrong step used, etc.).

**Detailed explanation for engineers:**
    The pipeline works in three stages:

    1. **Boundary detection** (`extract_cot_steps`):
       Splits the raw response text into discrete reasoning steps using regex
       patterns that match common step markers: "Step N:", numbered lines
       ("1."), and discourse markers ("First,", "Then,", "Next,", "Finally,").
       Each step is wrapped in a `CoTStep` dataclass.

    2. **Circuit construction** (`build_circuit`):
       For each step, scan for back-references ("from step N", "(N)") to
       identify which earlier steps this step depends on.  Extract the last
       numeric result in each step as `output_value`.  Build a `CoTCircuit`
       recording any cycles (impossible forward references) and broken links.

    3. **Broken-link detection** (`find_broken_links`):
       A broken link is: step i declares it uses the output of step j, but
       step j's actual `output_value` is not None and differs from the value
       that step i appears to be using (within `tolerance`).

    The `CoTCircuitVerifier` class implements the `ConstraintExtractor` protocol
    so it can be registered with `AutoExtractor` and used in the
    `VerifyRepairPipeline` just like `ArithmeticExtractor` or `NL2Z3Extractor`.

    CI safety:
    - No LLM calls; purely regex/string-based.  Always runs in CI.
    - Tolerance parameter prevents floating-point false positives (default 0.01).

Spec: REQ-EXTRACT-015, REQ-EXTRACT-016,
      SCENARIO-EXTRACT-031, SCENARIO-EXTRACT-032, SCENARIO-EXTRACT-033,
      SCENARIO-EXTRACT-034, SCENARIO-EXTRACT-035
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from carnot.pipeline.extract import ConstraintResult

# ---------------------------------------------------------------------------
# Regex patterns for step boundary detection
# ---------------------------------------------------------------------------

# Matches common step-opening patterns at the START of a line (case-insensitive):
#   "Step 1:", "step 2.", "1.", "2)", "First,", "Then:", "Next,", "Finally,"
_STEP_BOUNDARY_RE = re.compile(
    r"(?im)"
    r"(?:^step\s*\d+[\s.:)\-]+)"  # "Step 1:", "step 2."
    r"|(?:^\d+[.)]\s+)"            # "1. ", "2) "
    r"|(?:^(?:first|then|next|finally)[,:\s]+)",  # discourse markers
)

# Matches a back-reference to a previous step inside step text:
#   "from step 3", "step 2's", "step 2,", "the result from step 1", "(step 2)"
# The word-boundary character class ['\s,.\-)] covers common punctuation after step numbers.
_BACKREF_RE = re.compile(
    r"(?i)(?:from\s+step\s+(\d+)|step\s+(\d+)['\s,.\-)]|"
    r"result\s+from\s+step\s+(\d+)|\(step\s*(\d+)\))",
)

# Matches a float or int value in text (last one wins as output_value)
_NUMERIC_RE = re.compile(r"-?\d+(?:\.\d+)?")

# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass
class CoTStep:
    """One discrete reasoning step extracted from a chain-of-thought response.

    **Detailed explanation for engineers:**
        `step_id` is 0-based (first step = 0).  `input_refs` lists the
        step_ids (0-based) that this step explicitly references.  `output_value`
        is the last float found in the step text — it is the step's "result"
        that later steps should use.  When no number appears in the text,
        `output_value` is None and the step is excluded from link checking.

    Attributes:
        step_id:       0-based index of this step in the CoT sequence.
        text:          Raw text content of the step (after stripping the marker).
        input_refs:    Sorted list of step_ids this step references (0-based).
        output_value:  Last numeric result in the step text, or None.
        is_final_answer: True for the last step in the sequence.

    Spec: REQ-EXTRACT-015
    """

    step_id: int
    text: str
    input_refs: list[int] = field(default_factory=list)
    output_value: float | None = None
    is_final_answer: bool = False


@dataclass
class CoTCircuit:
    """Computational dependency graph extracted from a chain-of-thought response.

    **Detailed explanation for engineers:**
        `broken_links` is a list of 4-tuples:
          (downstream_step_id, upstream_step_id, expected_value_str, actual_value_str)
        where:
        - downstream_step_id: the step that reads the upstream value
        - upstream_step_id:   the step whose output is being referenced
        - expected_value_str: the value the downstream step *appears* to use
                              (last numeric value in the downstream step's text,
                              as a string for display)
        - actual_value_str:   the upstream step's actual output_value (as a string)

        `has_cycle` is True if any step's `input_refs` contain a step_id >=
        the step's own step_id (forward reference — impossible in valid CoT).

    Attributes:
        steps:        All extracted CoTStep objects in sequence order.
        has_cycle:    True if any step references a later step.
        broken_links: List of (downstream_id, upstream_id, expected, actual) tuples.

    Spec: REQ-EXTRACT-016
    """

    steps: list[CoTStep]
    has_cycle: bool
    broken_links: list[tuple[int, int, str, str]]


# ---------------------------------------------------------------------------
# Step extraction
# ---------------------------------------------------------------------------


def extract_cot_steps(response: str) -> list[CoTStep]:
    """Split a chain-of-thought response into discrete reasoning steps.

    **Detailed explanation for engineers:**
        Uses `_STEP_BOUNDARY_RE` to find step-opening markers.  The text between
        two consecutive markers (or from the last marker to end-of-string) becomes
        one step.  Text before the first marker is treated as a preamble step.

        For each step:
        - `input_refs` are populated by `_BACKREF_RE` matches in the step text.
          References to step numbers in the response are converted to 0-based IDs
          (the user writes "Step 1" but we store step_id=0).
        - `output_value` is the last float/int in the step text (via `_NUMERIC_RE`).
        - `is_final_answer` is True only for the very last step.

        Edge cases:
        - If no step markers are found, the entire response is treated as one step.
        - Empty response → empty list.

    Args:
        response: Raw chain-of-thought response text.

    Returns:
        List of CoTStep objects in order; empty list for empty/whitespace input.

    Spec: REQ-EXTRACT-015, SCENARIO-EXTRACT-031, SCENARIO-EXTRACT-032
    """
    if not response or not response.strip():
        return []

    # Find all marker positions.
    marker_spans: list[int] = [m.start() for m in _STEP_BOUNDARY_RE.finditer(response)]

    if not marker_spans:
        # No markers found — treat the whole response as a single step.
        step_text = response.strip()
        nums = _NUMERIC_RE.findall(step_text)
        return [
            CoTStep(
                step_id=0,
                text=step_text,
                input_refs=[],
                output_value=float(nums[-1]) if nums else None,
                is_final_answer=True,
            )
        ]

    # Build text segments: one per marker interval.
    segments: list[str] = []
    for i, start in enumerate(marker_spans):
        end = marker_spans[i + 1] if i + 1 < len(marker_spans) else len(response)
        segments.append(response[start:end].strip())

    steps: list[CoTStep] = []
    for idx, seg in enumerate(segments):
        # Remove the leading marker from the segment text to get clean content.
        # The marker is the first match within the segment.
        marker_match = _STEP_BOUNDARY_RE.match(seg)
        text = seg[marker_match.end():].strip() if marker_match else seg

        # Extract back-references (convert 1-based step numbers to 0-based IDs).
        refs: list[int] = []
        for m in _BACKREF_RE.finditer(seg):
            # Groups are alternatives; find the first non-None group.
            raw_num = next(g for g in m.groups() if g is not None)
            ref_id = int(raw_num) - 1  # convert 1-based to 0-based
            if 0 <= ref_id < idx:  # only valid back-references
                refs.append(ref_id)
        refs = sorted(set(refs))

        # Extract output value (last number in the step *content*, not the marker).
        # We search `text` (marker stripped) to avoid capturing step numbers in
        # "Step 1:" from bleeding into output_value.
        nums = _NUMERIC_RE.findall(text)
        output_value = float(nums[-1]) if nums else None

        steps.append(
            CoTStep(
                step_id=idx,
                text=text,
                input_refs=refs,
                output_value=output_value,
                is_final_answer=False,  # will be set below
            )
        )

    if steps:
        steps[-1].is_final_answer = True

    return steps


# ---------------------------------------------------------------------------
# Broken-link detection
# ---------------------------------------------------------------------------


def find_broken_links(
    steps: list[CoTStep],
    tolerance: float = 0.01,
) -> list[tuple[int, int, str, str]]:
    """Find broken links in a CoT dependency graph.

    **Detailed explanation for engineers:**
        A broken link is: step `i` declares `input_refs=[j, ...]` and step `j`
        has a non-None `output_value`, but the value that step `i` *appears* to
        use (the last numeric value in step `i`'s text near the reference) does
        not match step `j`'s output within `tolerance`.

        Heuristic for "expected value":
        We look at the last numeric value in step `i`'s text as the expected
        value the step is using.  This is the same field as `output_value` for
        the downstream step.  When the downstream step's own output_value is
        not None, we compare it against the upstream's output_value.

        This heuristic keeps the implementation simple and avoids false positives
        from steps that reference a prior result but produce a very different
        numeric output themselves (e.g. "From step 1 (value=10), multiply by 5
        to get 50" — the output_value here is 50, not 10).  We therefore only
        flag a broken link when the *downstream* step's `output_value` (its own
        final result) is close in magnitude to the *upstream* step's claimed
        output but differs by more than `tolerance`.  A large ratio (e.g. 50 vs
        10) is not flagged because the downstream step is clearly doing further
        computation.

        Concretely:
        - For each (i, j) pair where j ∈ steps[i].input_refs:
          - If steps[j].output_value is None → skip (no value to compare)
          - If steps[i].output_value is None → skip (no downstream result)
          - Compute relative difference = |downstream - upstream| / max(|upstream|, 1e-9)
          - If relative difference > tolerance AND downstream ≠ upstream (abs) AND
            abs(downstream / upstream) is between 0.5 and 2.0 (same order of magnitude,
            so probably the same value modulo a typo or rounding error):
            → broken link: expected=downstream's output (what step i uses),
                           actual=step j's output (the upstream ground truth)

        The order-of-magnitude filter prevents flagging "step 2 multiplied step 1
        (=10) by 5 to get 50" as a broken link because 50/10 = 5.0 is outside
        the [0.5, 2.0] range.

    Args:
        steps:     Ordered list of CoTStep objects.
        tolerance: Relative-difference tolerance (default 0.01 = 1%).

    Returns:
        List of (downstream_step_id, upstream_step_id, expected_str, actual_str).

    Spec: REQ-EXTRACT-016, SCENARIO-EXTRACT-033
    """
    broken: list[tuple[int, int, str, str]] = []

    for step in steps:
        for ref_id in step.input_refs:
            if ref_id >= len(steps):
                continue  # guard against out-of-range refs
            upstream = steps[ref_id]

            # Skip if either side has no numeric value to compare.
            if upstream.output_value is None or step.output_value is None:
                continue

            up_val = upstream.output_value
            down_val = step.output_value

            # Relative difference between the two values.
            denom = max(abs(up_val), 1e-9)
            rel_diff = abs(down_val - up_val) / denom

            if rel_diff <= tolerance:
                continue  # values match; no broken link

            # Only flag when the values are in the same order of magnitude.
            # (Ratio between 0.5 and 2.0 means they're "close but not equal".)
            ratio = abs(down_val / up_val) if abs(up_val) > 1e-9 else float("inf")
            if 0.5 <= ratio <= 2.0:
                broken.append(
                    (
                        step.step_id,
                        upstream.step_id,
                        str(down_val),
                        str(up_val),
                    )
                )

    return broken


# ---------------------------------------------------------------------------
# Circuit builder
# ---------------------------------------------------------------------------


def build_circuit(steps: list[CoTStep], tolerance: float = 0.01) -> CoTCircuit:
    """Assemble a CoTCircuit from a list of CoTStep objects.

    **Detailed explanation for engineers:**
        Checks for cycles (step references a later step — impossible in valid
        chain-of-thought) and delegates to `find_broken_links` for value
        consistency checking.

    Args:
        steps:     Ordered CoTStep list (from `extract_cot_steps`).
        tolerance: Tolerance for broken-link value comparison (default 0.01).

    Returns:
        CoTCircuit with `has_cycle` and `broken_links` populated.

    Spec: REQ-EXTRACT-016, SCENARIO-EXTRACT-034
    """
    # Cycle detection: any step that references a step_id >= its own step_id.
    has_cycle = any(
        ref >= step.step_id
        for step in steps
        for ref in step.input_refs
    )

    broken_links = find_broken_links(steps, tolerance=tolerance)
    return CoTCircuit(steps=steps, has_cycle=has_cycle, broken_links=broken_links)


# ---------------------------------------------------------------------------
# CoTCircuitVerifier
# ---------------------------------------------------------------------------


class CoTCircuitVerifier:
    """Verify chain-of-thought responses for structural (circuit) consistency.

    **Detailed explanation for engineers:**
        Implements the `ConstraintExtractor` protocol so it can be used
        anywhere an extractor is accepted (AutoExtractor, VerifyRepairPipeline).

        Unlike `NL2Z3Extractor`, this class makes NO LLM calls — it is purely
        regex/string-based and always runs in CI.

        The tolerance parameter controls how close two numeric values must be
        (as a relative difference) to be considered "the same".  The default
        of 0.01 (1%) is appropriate for chain-of-thought arithmetic where small
        rounding differences may appear.

    Attributes:
        tolerance: Relative tolerance for broken-link value comparison.
        last_circuit: The CoTCircuit from the most recent `verify()` call.

    Spec: REQ-EXTRACT-015, REQ-EXTRACT-016,
          SCENARIO-EXTRACT-031, SCENARIO-EXTRACT-032, SCENARIO-EXTRACT-033,
          SCENARIO-EXTRACT-034, SCENARIO-EXTRACT-035
    """

    def __init__(self, tolerance: float = 0.01) -> None:
        self.tolerance = tolerance
        self.last_circuit: CoTCircuit | None = None

    @property
    def supported_domains(self) -> list[str]:
        """Domains this extractor handles: chain-of-thought reasoning traces."""
        return ["reasoning"]

    def verify(self, response: str) -> CoTCircuit:
        """Extract the computational circuit and check for structural inconsistency.

        **Detailed explanation for engineers:**
            Combines `extract_cot_steps` and `build_circuit` into a single call.
            Stores the result in `self.last_circuit` for inspection after the call.

        Args:
            response: The chain-of-thought response to verify.

        Returns:
            CoTCircuit with steps, has_cycle, and broken_links populated.

        Spec: REQ-EXTRACT-016, SCENARIO-EXTRACT-034
        """
        steps = extract_cot_steps(response)
        circuit = build_circuit(steps, tolerance=self.tolerance)
        self.last_circuit = circuit
        return circuit

    def extract(
        self,
        question: str,
        response: str,
        domain: str | None = None,
    ) -> list[ConstraintResult]:
        """Implement the ConstraintExtractor protocol; return one violation per broken link.

        **Detailed explanation for engineers:**
            Calls `verify(response)` and maps each broken link to a
            `ConstraintResult(constraint_type="circuit_broken_link", ...)`.
            The `description` field names both the downstream and upstream step
            IDs so downstream pipeline stages can surface the exact location.

            Domain filter: if `domain` is set and is not "reasoning", returns [].

            A consistent response (no broken links) returns an empty list; this
            does NOT mean the response is correct — only that its computational
            dependency graph is structurally sound.

        Args:
            question: The original question (unused; kept for protocol compatibility).
            response: The chain-of-thought response to verify.
            domain:   Optional domain hint; non-"reasoning" domains are skipped.

        Returns:
            List of ConstraintResult (empty when no broken links found).

        Spec: REQ-EXTRACT-015, REQ-EXTRACT-016, SCENARIO-EXTRACT-035
        """
        if domain is not None and domain not in self.supported_domains:
            return []

        circuit = self.verify(response)

        results: list[ConstraintResult] = []
        for downstream_id, upstream_id, expected, actual in circuit.broken_links:
            results.append(
                ConstraintResult(
                    constraint_type="circuit_broken_link",
                    description=(
                        f"Broken circuit link: step {downstream_id + 1} uses value "
                        f"{expected} but step {upstream_id + 1} produced {actual}. "
                        f"The downstream step's result does not follow from the upstream "
                        f"step's output — this indicates a wrong value was substituted "
                        f"in the reasoning chain."
                    ),
                    metadata={
                        "downstream_step_id": downstream_id,
                        "upstream_step_id": upstream_id,
                        "expected_value": expected,
                        "actual_value": actual,
                        "tolerance": self.tolerance,
                    },
                )
            )

        return results
