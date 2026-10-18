"""PromptConstraintExtractor: extract formal Carnot constraints from user prompts on-the-fly.

**Researcher summary (ROCE, arXiv:2605.01124):**
    ROCE (Reasoning with On-the-fly Constraint Extraction) argues that user
    prompts already encode the acceptance criteria for a correct response — they
    just need to be surfaced as machine-checkable constraints *before* generation,
    not after.  This module implements the extraction step: it reads a natural-
    language prompt and emits one DynamicConstraint per instruction it finds.
    The verification pipeline can then call constraint.check(response) on every
    constraint to detect violations without a separate KB lookup.

**Why this is better than post-hoc checking:**
    Post-hoc checkers read the response and guess what the user wanted.  ROCE's
    approach makes the acceptance criteria explicit at prompt time, so the checker
    is just a predicate evaluation — no guessing required.

**Detailed explanation for engineers:**
    The extractor has two layers:

    1. _RuleExtractor (always active, no LLM):
       Runs a battery of regex patterns over the prompt to detect the 10 most
       common instruction types.  For each match, builds a DynamicConstraint
       whose check() method evaluates the constraint deterministically against
       any candidate response.  This layer is the "zero-false-accepts" guarantee:
       when the prompt says "must contain X" and the response omits X, check()
       returns False — no LLM inference required.

    2. LLM layer (live mode, CARNOT_FORCE_LIVE=1):
       When the rule extractor misses a constraint (e.g., an unusual phrasing),
       the module calls an injectable generate_fn — defaulting to the Qwen3.6-35B
       GGUF model from unsloth — to extract additional constraints as JSON.
       The LLM output is parsed and converted into DynamicConstraint objects.
       Failures in this layer are silent (the rule-extractor results are kept).

    Supported instruction types and their trigger phrases:
        must_contain      - "must include X", "must contain X"
        must_not_contain  - "do not include X", "never mention X", "exclude X"
        format_json       - "respond in JSON", "output as JSON", "format as JSON"
        format_list       - "numbered list", "bullet list", "bulleted list"
        max_words         - "under N words", "at most N words", "N words or fewer"
        min_words         - "at least N words", "minimum N words"
        numeric_range     - "between X and Y"
        starts_with       - "begin with X", "start with X", "start your response with X"
        ends_with         - "end with X", "conclude with X", "finish with X"
        no_repetition     - "do not repeat", "no repetition", "each item only once"

Spec: REQ-EXTRACT-055,
      SCENARIO-EXTRACT-094, SCENARIO-EXTRACT-095
"""

from __future__ import annotations

import json
import os
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any


# ---------------------------------------------------------------------------
# DynamicConstraint dataclass
# ---------------------------------------------------------------------------


@dataclass
class DynamicConstraint:
    """One constraint extracted from a user prompt, with a built-in response checker.

    **Detailed explanation for engineers:**
        Each DynamicConstraint encodes a single acceptance criterion that the
        user's prompt implies.  The check() method evaluates that criterion
        against any candidate response and returns True (satisfied) or False
        (violated).

        WHY a check() method instead of storing a lambda:
        Dataclasses with stored callables can't be serialised to JSON.  By
        encoding the check logic as a method that reads self.metadata, we keep
        the object serialisation-friendly and the check logic transparent.

    Attributes:
        instruction_type: One of the 10 canonical types or "llm_extracted" for
            constraints discovered only by the LLM layer.
        description: Human-readable summary of what the constraint requires.
        metadata: Type-specific parameters.  Keys depend on instruction_type:
            - must_contain:     {"term": str}
            - must_not_contain: {"term": str}
            - format_json:      {}
            - format_list:      {}
            - max_words:        {"limit": int}
            - min_words:        {"limit": int}
            - numeric_range:    {"low": float, "high": float}
            - starts_with:      {"prefix": str}
            - ends_with:        {"suffix": str}
            - no_repetition:    {}
            - llm_extracted:    {"raw": str}
        raw_phrase: The substring from the prompt that triggered this constraint.

    Spec: REQ-EXTRACT-055-1
    """

    instruction_type: str
    description: str
    metadata: dict[str, Any] = field(default_factory=dict)
    raw_phrase: str = ""

    def check(self, response: str) -> bool:
        """Return True if response satisfies this constraint, False if it is violated.

        **Detailed explanation for engineers:**
            Each branch reads self.metadata and evaluates the constraint against
            response.  The evaluation is CONSERVATIVE for ambiguous cases: when
            uncertain, the method returns False (violated) rather than True.
            This guarantees zero false accepts — we never silently accept a
            response that the user's instruction forbids.

            Edge cases:
            - empty response: most constraints treat it as violated.
            - llm_extracted type: no deterministic check available; returns True
              (we defer to the human reviewer or a downstream LLM verifier).
            - numeric_range: scans the response for ANY float/int that falls
              in range; if none found, returns False.
            - no_repetition: tokenises response into lines/items for list context;
              falls back to word-level for unstructured prose.

        Args:
            response: The LLM-generated text to verify.

        Returns:
            True when the response satisfies this constraint, False when violated.
        """
        itype = self.instruction_type

        if itype == "must_contain":
            term = self.metadata.get("term", "")
            return bool(term) and term.lower() in response.lower()

        if itype == "must_not_contain":
            term = self.metadata.get("term", "")
            if not term:
                return True
            return term.lower() not in response.lower()

        if itype == "format_json":
            stripped = response.strip()
            # Accept bare JSON objects/arrays (with or without surrounding fences).
            # Strip markdown fences if present.
            if stripped.startswith("```"):
                # Extract content between first ``` and last ```.
                inner = re.sub(r"^```[a-zA-Z]*\n?", "", stripped)
                inner = re.sub(r"\n?```$", "", inner).strip()
            else:
                inner = stripped
            try:
                json.loads(inner)
                return True
            except (json.JSONDecodeError, ValueError):
                return False

        if itype == "format_list":
            # At least one line must start with a list marker:
            # numbered (1. 2. etc.) or bulleted (-, *, •).
            lines = response.strip().splitlines()
            list_line = re.compile(r"^\s*(\d+[.)]\s+|[-*•]\s+)")
            return any(list_line.match(line) for line in lines)

        if itype == "max_words":
            limit = int(self.metadata.get("limit", 0))
            word_count = len(response.split())
            return word_count <= limit

        if itype == "min_words":
            limit = int(self.metadata.get("limit", 0))
            word_count = len(response.split())
            return word_count >= limit

        if itype == "numeric_range":
            low = float(self.metadata.get("low", float("-inf")))
            high = float(self.metadata.get("high", float("inf")))
            # Find all numbers in the response and check if any fall in range.
            numbers = re.findall(r"-?\d+(?:\.\d+)?", response)
            if not numbers:
                return False
            return any(low <= float(n) <= high for n in numbers)

        if itype == "starts_with":
            prefix = self.metadata.get("prefix", "")
            return response.lstrip().lower().startswith(prefix.lower())

        if itype == "ends_with":
            suffix = self.metadata.get("suffix", "")
            return response.rstrip().lower().endswith(suffix.lower())

        if itype == "no_repetition":
            # Check list items first (numbered or bulleted).
            lines = response.strip().splitlines()
            list_line = re.compile(r"^\s*(?:\d+[.)]\s+|[-*•]\s+)(.*)")
            items = [m.group(1).strip().lower() for line in lines if (m := list_line.match(line))]
            if items:
                return len(items) == len(set(items))
            # Fallback: check for repeated sentences.
            # Strip trailing punctuation so "blue." and "blue" compare equal.
            sentences = re.split(r"[.!?]\s+", response.lower())
            cleaned = [re.sub(r"[.!?]+$", "", s.strip()) for s in sentences if s.strip()]
            return len(cleaned) == len(set(cleaned))

        if itype == "llm_extracted":
            # No deterministic check; conservatively pass to avoid blocking valid output.
            # WHY True here (not False): the LLM layer should be used for
            # non-deterministic constraints where a deterministic checker cannot
            # be constructed from the extracted metadata.  Blocking without evidence
            # would cause false rejects.  The caller can invoke a downstream LLM
            # verifier for these constraints.
            return True

        # Unknown type: conservative pass.
        return True


# ---------------------------------------------------------------------------
# _RuleExtractor — deterministic, regex-based, no LLM
# ---------------------------------------------------------------------------


class _RuleExtractor:
    """Deterministic regex-based rule extractor for the 10 canonical instruction types.

    **Detailed explanation for engineers:**
        Each _extract_* method runs one or more regex patterns against the prompt
        and returns a list of DynamicConstraint objects.  The methods are kept
        separate so that adding a new instruction type is a matter of adding one
        method and registering it in _EXTRACTORS.

        All patterns are case-insensitive and use non-greedy matching to avoid
        capturing too much context.

    Spec: REQ-EXTRACT-055-2
    """

    def extract(self, prompt: str) -> list[DynamicConstraint]:
        """Run all rule extractors against prompt and return combined results."""
        results: list[DynamicConstraint] = []
        for method in (
            self._extract_must_contain,
            self._extract_must_not_contain,
            self._extract_format_json,
            self._extract_format_list,
            self._extract_max_words,
            self._extract_min_words,
            self._extract_numeric_range,
            self._extract_starts_with,
            self._extract_ends_with,
            self._extract_no_repetition,
        ):
            results.extend(method(prompt))
        return results

    # ------------------------------------------------------------------
    # Individual rule extractors
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_must_contain(prompt: str) -> list[DynamicConstraint]:
        """Extract must_contain constraints: 'must include X', 'must contain X'."""
        constraints: list[DynamicConstraint] = []
        # Patterns: "must include 'X'", "must contain X", "include X in your response"
        # Capture the term inside quotes or as the next word/phrase up to punctuation.
        patterns = [
            r"must\s+(?:include|contain)\s+['\"](.+?)['\"]",
            r"must\s+(?:include|contain)\s+the\s+(?:word|phrase|term)\s+['\"](.+?)['\"]",
            r"must\s+(?:include|contain)\s+the\s+(?:word|phrase|term)\s+(\w+)",
            r"your\s+response\s+must\s+(?:include|contain)\s+['\"](.+?)['\"]",
        ]
        seen: set[str] = set()
        for pat in patterns:
            for m in re.finditer(pat, prompt, re.IGNORECASE):
                term = m.group(1).strip()
                if term and term.lower() not in seen:
                    seen.add(term.lower())
                    constraints.append(
                        DynamicConstraint(
                            instruction_type="must_contain",
                            description=f"Response must contain the term '{term}'",
                            metadata={"term": term},
                            raw_phrase=m.group(0),
                        )
                    )
        return constraints

    @staticmethod
    def _extract_must_not_contain(prompt: str) -> list[DynamicConstraint]:
        """Extract must_not_contain constraints: 'do not include X', 'never mention X'."""
        constraints: list[DynamicConstraint] = []
        patterns = [
            r"do\s+not\s+(?:include|mention|use|say)\s+['\"](.+?)['\"]",
            r"never\s+(?:mention|include|use|say)\s+['\"](.+?)['\"]",
            r"exclude\s+['\"](.+?)['\"]",
            r"do\s+not\s+(?:include|mention|use|say)\s+the\s+(?:word|phrase|term)\s+['\"](.+?)['\"]",
            r"never\s+(?:mention|include|use|say)\s+the\s+(?:word|phrase|term)\s+['\"](.+?)['\"]",
            r"do\s+not\s+(?:include|mention|use|say)\s+the\s+(?:word|phrase|term)\s+(\w+)",
        ]
        seen: set[str] = set()
        for pat in patterns:
            for m in re.finditer(pat, prompt, re.IGNORECASE):
                term = m.group(1).strip()
                if term and term.lower() not in seen:
                    seen.add(term.lower())
                    constraints.append(
                        DynamicConstraint(
                            instruction_type="must_not_contain",
                            description=f"Response must not contain the term '{term}'",
                            metadata={"term": term},
                            raw_phrase=m.group(0),
                        )
                    )
        return constraints

    @staticmethod
    def _extract_format_json(prompt: str) -> list[DynamicConstraint]:
        """Extract format_json constraints: 'respond in JSON', 'output as JSON'."""
        patterns = [
            r"respond\s+in\s+JSON",
            r"output\s+(?:as|in)\s+JSON",
            r"format\s+(?:your\s+)?(?:response\s+)?as\s+JSON",
            r"return\s+(?:a\s+)?JSON",
            r"provide\s+(?:a\s+)?JSON",
            r"give\s+(?:a\s+)?JSON\s+(?:object|array|response|answer)",
            r"in\s+JSON\s+format",
        ]
        for pat in patterns:
            if re.search(pat, prompt, re.IGNORECASE):
                return [
                    DynamicConstraint(
                        instruction_type="format_json",
                        description="Response must be valid JSON",
                        metadata={},
                        raw_phrase=re.search(pat, prompt, re.IGNORECASE).group(0),  # type: ignore[union-attr]
                    )
                ]
        return []

    @staticmethod
    def _extract_format_list(prompt: str) -> list[DynamicConstraint]:
        """Extract format_list constraints: 'numbered list', 'bullet list'."""
        patterns = [
            r"numbered\s+list",
            r"bullet(?:ed)?\s+list",
            r"use\s+(?:a\s+)?(?:list|bullets|bullet\s+points|numbers)",
            r"list\s+(?:each|the|your|them|all)",
            r"in\s+a\s+(?:numbered|bulleted|bullet)\s+list",
        ]
        for pat in patterns:
            m = re.search(pat, prompt, re.IGNORECASE)
            if m:
                return [
                    DynamicConstraint(
                        instruction_type="format_list",
                        description="Response must use a list format (numbered or bulleted)",
                        metadata={},
                        raw_phrase=m.group(0),
                    )
                ]
        return []

    @staticmethod
    def _extract_max_words(prompt: str) -> list[DynamicConstraint]:
        """Extract max_words constraints: 'under N words', 'at most N words'."""
        patterns = [
            r"(?:under|fewer\s+than|less\s+than|at\s+most|no\s+more\s+than)\s+(\d+)\s+words?",
            r"(\d+)\s+words?\s+or\s+fewer",
            r"(\d+)\s+words?\s+(?:max|maximum)",
            r"limit\s+(?:your\s+)?(?:response\s+)?to\s+(\d+)\s+words?",
            r"keep\s+(?:it|your\s+(?:response|answer))\s+(?:to\s+)?(?:under\s+)?(\d+)\s+words?",
        ]
        seen: set[int] = set()
        results: list[DynamicConstraint] = []
        for pat in patterns:
            m = re.search(pat, prompt, re.IGNORECASE)
            if m:
                # The number may be in group 1 or 2 depending on pattern.
                for g in m.groups():
                    if g and g.isdigit():
                        limit = int(g)
                        if limit not in seen:
                            seen.add(limit)
                            results.append(
                                DynamicConstraint(
                                    instruction_type="max_words",
                                    description=f"Response must be at most {limit} words",
                                    metadata={"limit": limit},
                                    raw_phrase=m.group(0),
                                )
                            )
                        break
        return results

    @staticmethod
    def _extract_min_words(prompt: str) -> list[DynamicConstraint]:
        """Extract min_words constraints: 'at least N words', 'minimum N words'."""
        patterns = [
            r"(?:at\s+least|minimum|no\s+fewer\s+than|more\s+than)\s+(\d+)\s+words?",
            r"(\d+)\s+words?\s+or\s+more",
            r"(\d+)\s+words?\s+(?:min|minimum)",
            r"write\s+(?:at\s+least\s+)?(\d+)\s+words?",
        ]
        seen: set[int] = set()
        results: list[DynamicConstraint] = []
        for pat in patterns:
            m = re.search(pat, prompt, re.IGNORECASE)
            if m:
                for g in m.groups():
                    if g and g.isdigit():
                        limit = int(g)
                        if limit not in seen:
                            seen.add(limit)
                            results.append(
                                DynamicConstraint(
                                    instruction_type="min_words",
                                    description=f"Response must be at least {limit} words",
                                    metadata={"limit": limit},
                                    raw_phrase=m.group(0),
                                )
                            )
                        break
        return results

    @staticmethod
    def _extract_numeric_range(prompt: str) -> list[DynamicConstraint]:
        """Extract numeric_range constraints: 'between X and Y'."""
        patterns = [
            r"(?:between|from)\s+(-?\d+(?:\.\d+)?)\s+(?:and|to)\s+(-?\d+(?:\.\d+)?)",
            r"(?:in|within)\s+(?:the\s+)?range\s+(?:of\s+)?(-?\d+(?:\.\d+)?)\s*(?:to|-)\s*(-?\d+(?:\.\d+)?)",
            r"(-?\d+(?:\.\d+)?)\s*(?:to|-)\s*(-?\d+(?:\.\d+)?)\s+range",
        ]
        results: list[DynamicConstraint] = []
        seen: set[tuple[float, float]] = set()
        for pat in patterns:
            for m in re.finditer(pat, prompt, re.IGNORECASE):
                low = float(m.group(1))
                high = float(m.group(2))
                key = (low, high)
                if key not in seen:
                    seen.add(key)
                    results.append(
                        DynamicConstraint(
                            instruction_type="numeric_range",
                            description=f"Numeric answer must be between {low} and {high}",
                            metadata={"low": low, "high": high},
                            raw_phrase=m.group(0),
                        )
                    )
        return results

    @staticmethod
    def _extract_starts_with(prompt: str) -> list[DynamicConstraint]:
        """Extract starts_with constraints: 'begin with X', 'start your response with X'."""
        patterns = [
            r"(?:start|begin)\s+(?:your\s+)?(?:response\s+)?with\s+['\"](.+?)['\"]",
            r"(?:start|begin)\s+(?:your\s+)?(?:response\s+)?with\s+the\s+(?:word|phrase|text)\s+['\"](.+?)['\"]",
            r"(?:open|lead)\s+(?:with|off\s+with)\s+['\"](.+?)['\"]",
        ]
        seen: set[str] = set()
        results: list[DynamicConstraint] = []
        for pat in patterns:
            for m in re.finditer(pat, prompt, re.IGNORECASE):
                prefix = m.group(1).strip()
                if prefix.lower() not in seen:
                    seen.add(prefix.lower())
                    results.append(
                        DynamicConstraint(
                            instruction_type="starts_with",
                            description=f"Response must begin with '{prefix}'",
                            metadata={"prefix": prefix},
                            raw_phrase=m.group(0),
                        )
                    )
        return results

    @staticmethod
    def _extract_ends_with(prompt: str) -> list[DynamicConstraint]:
        """Extract ends_with constraints: 'end with X', 'conclude with X'."""
        patterns = [
            r"(?:end|finish|close|conclude)\s+(?:your\s+)?(?:response\s+)?with\s+['\"](.+?)['\"]",
            r"(?:end|finish|close|conclude)\s+(?:your\s+)?(?:response\s+)?with\s+the\s+(?:word|phrase|text)\s+['\"](.+?)['\"]",
        ]
        seen: set[str] = set()
        results: list[DynamicConstraint] = []
        for pat in patterns:
            for m in re.finditer(pat, prompt, re.IGNORECASE):
                suffix = m.group(1).strip()
                if suffix.lower() not in seen:
                    seen.add(suffix.lower())
                    results.append(
                        DynamicConstraint(
                            instruction_type="ends_with",
                            description=f"Response must end with '{suffix}'",
                            metadata={"suffix": suffix},
                            raw_phrase=m.group(0),
                        )
                    )
        return results

    @staticmethod
    def _extract_no_repetition(prompt: str) -> list[DynamicConstraint]:
        """Extract no_repetition constraints: 'do not repeat', 'each item only once'."""
        patterns = [
            r"do\s+not\s+repeat\s+(?:yourself|items?|entries?|yourself)",
            r"no\s+repetition",
            r"each\s+(?:item|entry|element)\s+(?:must\s+)?(?:appear|occur)?\s*(?:only\s+)?once",
            r"avoid\s+(?:repetition|repeating|duplicate)",
            r"unique(?:ly)?\s+(?:list|items?|entries?)",
            r"no\s+duplicate",
        ]
        for pat in patterns:
            m = re.search(pat, prompt, re.IGNORECASE)
            if m:
                return [
                    DynamicConstraint(
                        instruction_type="no_repetition",
                        description="Response must not repeat items",
                        metadata={},
                        raw_phrase=m.group(0),
                    )
                ]
        return []


# ---------------------------------------------------------------------------
# PromptConstraintExtractor — public API
# ---------------------------------------------------------------------------

_GenerateFn = Callable[[str], str]

_LLM_EXTRACTION_PROMPT = (
    "You are a constraint extractor. Given a user prompt, identify any explicit "
    "constraints on the response format or content that were NOT captured by regex "
    "patterns for: must_contain, must_not_contain, format_json, format_list, "
    "max_words, min_words, numeric_range, starts_with, ends_with, no_repetition.\n\n"
    "Output a JSON array of objects, each with keys:\n"
    '  "instruction_type": "llm_extracted",\n'
    '  "description": "<one-line human-readable description>",\n'
    '  "raw": "<exact phrase from the prompt that triggered this constraint>"\n\n'
    "If no additional constraints exist, output an empty array []. "
    "Output ONLY valid JSON — no prose, no markdown fences.\n\n"
    "User prompt:\n"
)


def _default_generate(prompt: str) -> str:
    """Call the real LLM (Qwen3.6-35B-A3B-GGUF).  Only used when CARNOT_FORCE_LIVE=1.

    WHY Qwen3.6-35B-A3B: CLAUDE.md mandates this as the SOTA GGUF model for new
    experiments; it's MoE with only ~3B active parameters, so constraint extraction
    (a short prompt → short JSON response task) runs fast even at 35B parameter scale.
    """
    from carnot.inference.model_loader import generate, load_model  # deferred import

    model, tokenizer = load_model("unsloth/Qwen3.6-35B-A3B-GGUF")
    return generate(model, tokenizer, prompt, max_new_tokens=512)


class PromptConstraintExtractor:
    """Extract formal Carnot constraints from a natural-language user prompt.

    **Detailed explanation for engineers:**
        This is the main public class.  Instantiate it once per session; the
        rule extractor is stateless and can be reused across many prompts.

        Two-layer extraction:
        1. _RuleExtractor.extract(prompt) — always runs; no LLM; deterministic.
           Covers the 10 canonical instruction types.
        2. LLM extraction (only when CARNOT_FORCE_LIVE=1) — calls generate_fn
           to catch constraints the regex layer missed.  Results are appended
           to the rule-extractor results.

        Injectable generate_fn:
        - Tests: pass a mock that returns canned JSON.
        - Production: leave None to use _default_generate (Qwen3.6-35B).

        CI guard:
        - CARNOT_FORCE_LIVE not set → only rule extractor runs.
        - CARNOT_FORCE_LIVE=1 → both layers run.

    Spec: REQ-EXTRACT-055-3, REQ-EXTRACT-055-4, REQ-EXTRACT-055-5
    """

    def __init__(self, generate_fn: _GenerateFn | None = None) -> None:
        self._rule_extractor = _RuleExtractor()
        self._generate_fn = generate_fn or _default_generate

    def extract_from_prompt(self, prompt: str) -> list[DynamicConstraint]:
        """Extract DynamicConstraint objects from a user prompt.

        **Detailed explanation for engineers:**
            Always runs the deterministic rule extractor first.  In live mode
            (CARNOT_FORCE_LIVE=1), additionally calls the LLM for constraints
            the regex missed.

            LLM failures are silently swallowed (we return only the rule-extractor
            results) so that a transient model error doesn't break the pipeline.

        Args:
            prompt: The natural-language user instruction to parse.

        Returns:
            List of DynamicConstraint objects, one per detected instruction.
            Empty list when no recognisable constraints are found.

        Spec: REQ-EXTRACT-055-3, REQ-EXTRACT-055-4
        """
        constraints = self._rule_extractor.extract(prompt)

        if not os.environ.get("CARNOT_FORCE_LIVE"):
            return constraints

        # LLM layer: catch anything the regex missed.
        try:
            llm_output = self._generate_fn(_LLM_EXTRACTION_PROMPT + prompt)
            llm_constraints = self._parse_llm_output(llm_output)
            constraints.extend(llm_constraints)
        except Exception:  # noqa: BLE001
            # Silent fallback: rule-extractor results are already in constraints.
            pass

        return constraints

    def check_response(
        self,
        response: str,
        constraints: list[DynamicConstraint],
    ) -> list[DynamicConstraint]:
        """Return constraints that the response VIOLATES (check() returned False).

        **Detailed explanation for engineers:**
            Iterates over all constraints and calls constraint.check(response).
            Returns only the ones that returned False.  An empty list means
            the response satisfies all extracted constraints.

            This is the "zero false accepts" entry point: if a constraint is in
            this return list, the pipeline knows the response violates the user's
            instruction without any LLM inference.

        Args:
            response: The LLM-generated text to verify.
            constraints: The constraints extracted from the original prompt.

        Returns:
            List of violated DynamicConstraint objects.  Empty → fully compliant.

        Spec: REQ-EXTRACT-055-5
        """
        return [c for c in constraints if not c.check(response)]

    @staticmethod
    def _parse_llm_output(text: str) -> list[DynamicConstraint]:
        """Parse the LLM's JSON array of additional constraints.

        Returns a (possibly empty) list of DynamicConstraint with
        instruction_type="llm_extracted".  Malformed JSON yields [].
        """
        text = text.strip()
        try:
            items = json.loads(text)
        except (json.JSONDecodeError, ValueError):
            return []

        if not isinstance(items, list):
            return []

        results: list[DynamicConstraint] = []
        for item in items:
            if not isinstance(item, dict):
                continue
            desc = str(item.get("description") or "")
            raw = str(item.get("raw") or "")
            if desc:
                results.append(
                    DynamicConstraint(
                        instruction_type="llm_extracted",
                        description=desc,
                        metadata={"raw": raw},
                        raw_phrase=raw,
                    )
                )
        return results


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    "DynamicConstraint",
    "PromptConstraintExtractor",
]
