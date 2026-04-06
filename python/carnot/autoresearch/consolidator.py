"""Hierarchical lesson consolidation via tree-reduction merge.

**Researcher summary:**
    Implements the Trace2Skill hierarchical consolidation algorithm.
    Takes a pool of raw lessons from trajectory analysts and merges
    them into a conflict-free, deduplicated set. Uses LLM-powered
    inductive reasoning to identify prevalent patterns across lessons,
    merge duplicates, and resolve contradictions.

**Detailed explanation for engineers:**
    The consolidation algorithm follows the Trace2Skill paper
    (arxiv 2603.25158):

    1. **Group**: Partition lessons into batches of ``batch_size`` (default 32)
    2. **Merge**: For each batch, use an LLM to:
       - Deduplicate equivalent lessons (merge into one, boost confidence)
       - Resolve contradictions (keep the better-supported lesson)
       - Extract cross-cutting meta-patterns (create new higher-level lessons)
    3. **Reduce**: If more than one batch remains, repeat (tree reduction)
       - Total levels: L = ceil(log_batch_size(N))
    4. **Filter**: Remove lessons below ``min_confidence`` threshold

    This produces a smaller, higher-quality lesson set where:
    - Recurring patterns have higher confidence (validated by multiple experiments)
    - One-off observations are filtered out (likely idiosyncratic)
    - Contradictions are resolved (e.g., "use large step size" vs "use small step size")

    The consolidation uses a lower LLM temperature (0.3) than hypothesis
    generation (0.7) because this is analytical work, not creative.

Spec: REQ-AUTO-013, SCENARIO-AUTO-010
"""

from __future__ import annotations

import contextlib
import json
import logging
import math
import os
import re
from dataclasses import dataclass
from typing import Any

from carnot.autoresearch.trajectory_analyst import Lesson

logger = logging.getLogger(__name__)


@dataclass
class ConsolidatorConfig:
    """Configuration for lesson consolidation.

    **Researcher summary:**
        Controls batch size, confidence threshold, and LLM settings
        for the hierarchical merge.

    **Detailed explanation for engineers:**
        - ``batch_size``: How many lessons per merge batch. The Trace2Skill
          paper uses 32. Larger batches = fewer LLM calls but longer prompts.
        - ``min_confidence``: Minimum confidence to survive consolidation.
          Lessons below this after merging are discarded. Default 0.3.
        - ``temperature``: LLM sampling temperature. Low (0.3) because
          consolidation is analytical, not creative.
        - ``api_base``, ``model``, ``api_key``: LLM endpoint settings.
          Follows the same pattern as ``AnalystConfig`` and ``GeneratorConfig``.

    Spec: REQ-AUTO-013
    """

    api_base: str = os.environ.get("CARNOT_API_BASE", "http://localhost:8080/v1")
    model: str = "sonnet"
    api_key: str = "not-needed"
    batch_size: int = 32
    min_confidence: float = 0.3
    temperature: float = 0.3


CONSOLIDATION_PROMPT = """\
You are an expert at synthesizing research findings. You have been given a \
batch of lessons learned from Energy-Based Model (EBM) optimization experiments.

Your job is to consolidate these lessons into a cleaner, more useful set by:

1. **Deduplicating**: If two lessons say essentially the same thing, merge them \
into one with increased confidence (average + 0.1, capped at 1.0).

2. **Resolving contradictions**: If two lessons contradict each other (e.g., \
"use large step sizes" vs "use small step sizes"), keep the one with higher \
confidence or synthesize both into a more nuanced lesson (e.g., "step size \
should match landscape curvature").

3. **Extracting meta-patterns**: If multiple lessons point to a common \
underlying principle, create a new higher-level lesson that captures it.

4. **Preserving unique insights**: Don't discard lessons just because they're \
unusual — only merge or remove if there's a clear reason.

## Input lessons (JSON array)

{lessons_json}

## Output format

Return a JSON array of consolidated lessons. Each lesson has:
```json
[
  {{
    "title": "...",
    "description": "...",
    "confidence": 0.0-1.0,
    "applicable_benchmarks": ["..."],
    "model_tier": "all|ising|gibbs|boltzmann",
    "lesson_type": "error_pattern|success_pattern"
  }}
]
```

Return ONLY the JSON array, no other text.
"""


def _merge_batch_with_llm(
    config: ConsolidatorConfig,
    batch: list[Lesson],
) -> list[Lesson]:
    """Merge a single batch of lessons using an LLM.

    Returns the consolidated lessons. If the LLM call fails, returns
    the batch unchanged (graceful degradation).

    Spec: REQ-AUTO-013
    """
    try:
        from openai import OpenAI
    except ImportError:
        logger.warning("openai not installed, returning batch unchanged")
        return list(batch)

    lessons_json = json.dumps([lesson.to_dict() for lesson in batch], indent=2)
    prompt = CONSOLIDATION_PROMPT.format(lessons_json=lessons_json)

    try:
        client = OpenAI(base_url=config.api_base, api_key=config.api_key)
        response = client.chat.completions.create(
            model=config.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=config.temperature,
        )
        raw = response.choices[0].message.content or ""
    except Exception:
        logger.exception("Consolidation LLM call failed, returning batch unchanged")
        return list(batch)

    return _parse_consolidated_lessons(raw, batch)


def _parse_consolidated_lessons(
    raw: str,
    fallback: list[Lesson],
) -> list[Lesson]:
    """Parse consolidated lessons from LLM response.

    Tries multiple parsing strategies. Falls back to returning
    the original batch unchanged if parsing fails.

    Spec: REQ-AUTO-013
    """
    text = raw.strip()

    # Try direct JSON parse
    parsed: list[dict[str, Any]] | None = None
    with contextlib.suppress(json.JSONDecodeError, ValueError):
        parsed = json.loads(text)

    # Try extracting from code block
    if parsed is None:
        match = re.search(r"```(?:json)?\s*\n(.*?)```", text, re.DOTALL)
        if match:
            with contextlib.suppress(json.JSONDecodeError, ValueError):
                parsed = json.loads(match.group(1).strip())

    # Try finding [ ... ] in the text
    if parsed is None:
        match = re.search(r"\[.*\]", text, re.DOTALL)
        if match:
            with contextlib.suppress(json.JSONDecodeError, ValueError):
                parsed = json.loads(match.group(0))

    if parsed is None or not isinstance(parsed, list):
        logger.warning("Could not parse consolidated lessons, returning originals")
        return list(fallback)

    # Convert dicts to Lesson objects
    results: list[Lesson] = []
    for item in parsed:
        if not isinstance(item, dict):
            continue
        results.append(
            Lesson(
                title=item.get("title", "Consolidated lesson"),
                description=item.get("description", ""),
                examples=item.get("examples", []),
                confidence=float(item.get("confidence", 0.5)),
                applicable_benchmarks=item.get("applicable_benchmarks", ["all"]),
                model_tier=item.get("model_tier", "all"),
                lesson_type=item.get("lesson_type", "success_pattern"),
                source_experiment_id=item.get("source_experiment_id", "consolidated"),
            )
        )

    return results if results else list(fallback)


def consolidate_lessons(
    config: ConsolidatorConfig,
    lessons: list[Lesson],
) -> list[Lesson]:
    """Consolidate a pool of lessons via hierarchical tree-reduction.

    **Researcher summary:**
        Takes raw lessons from trajectory analysts and produces a
        smaller, higher-quality set by deduplicating, resolving
        contradictions, and extracting meta-patterns.

    **Detailed explanation for engineers:**
        The algorithm:

        1. If ``len(lessons) <= batch_size``, do a single merge pass
        2. Otherwise, split into batches of ``batch_size``, merge each
        3. Collect all merged batches and repeat (tree reduction)
        4. Continue until a single batch remains
        5. Filter out lessons below ``min_confidence``

        Total LLM calls: O(N / batch_size * L) where L = ceil(log_batch(N))

        If ``lessons`` is empty, returns empty. If a single lesson,
        returns it as-is (no LLM call needed, but still applies the
        confidence filter).

    Args:
        config: Consolidation settings (batch size, confidence threshold).
        lessons: Raw lessons from trajectory analysts.

    Returns:
        Consolidated, deduplicated, conflict-free lesson set.

    Spec: REQ-AUTO-013, SCENARIO-AUTO-010
    """
    if not lessons:
        return []

    # Single lesson — just apply confidence filter
    if len(lessons) == 1:
        return lessons if lessons[0].confidence >= config.min_confidence else []

    current = list(lessons)

    # Tree reduction: keep merging until we have one batch
    max_levels = max(1, math.ceil(math.log(len(lessons)) / math.log(max(config.batch_size, 2))))

    for level in range(max_levels):
        if len(current) <= config.batch_size:
            # Final merge
            current = _merge_batch_with_llm(config, current)
            break

        # Split into batches
        batches = [
            current[i : i + config.batch_size] for i in range(0, len(current), config.batch_size)
        ]

        # Merge each batch
        merged: list[Lesson] = []
        for batch in batches:
            merged.extend(_merge_batch_with_llm(config, batch))

        current = merged

        logger.info(
            "Consolidation level %d: %d → %d lessons",
            level,
            len(lessons) if level == 0 else len(current),
            len(merged),
        )
    else:
        # If we exhausted levels without breaking, do a final merge
        if len(current) > config.batch_size:
            current = _merge_batch_with_llm(config, current[: config.batch_size])

    # Apply confidence filter
    result = [lesson for lesson in current if lesson.confidence >= config.min_confidence]

    logger.info(
        "Consolidation complete: %d → %d lessons (after confidence filter)",
        len(lessons),
        len(result),
    )

    return result
