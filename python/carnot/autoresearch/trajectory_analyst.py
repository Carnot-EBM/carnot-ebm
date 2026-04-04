"""Trajectory analyst: parallel sub-agents that extract lessons from experiments.

**Researcher summary:**
    Implements the Trace2Skill pattern (arxiv 2603.25158) for Carnot's
    autoresearch loop. Error analysts diagnose WHY a hypothesis failed
    (not just "energy regressed" but "gradient explosion due to landscape
    curvature"). Success analysts extract generalizable optimization
    patterns from accepted hypotheses.

**Detailed explanation for engineers:**
    The current autoresearch loop feeds shallow failure feedback to the
    hypothesis generator: a list of (description, reason) strings. This
    module replaces that with deep trajectory analysis.

    After each hypothesis evaluation, the orchestrator dispatches an
    analyst sub-agent:

    - **Error analyst**: Receives the full experiment trajectory (code,
      metrics, errors, evaluation verdict). Uses an LLM to reason about
      the root cause. Produces a structured ``Lesson`` with a diagnosis,
      confidence score, and applicable benchmarks. Example: instead of
      "energy regression on rosenbrock", it produces "gradient explosion
      due to steep 100*(x[i+1]-x[i]^2)^2 curvature exceeding step size
      0.1 — need gradient clipping or smaller step sizes for ill-conditioned
      landscapes."

    - **Success analyst**: Receives an accepted experiment's trajectory.
      Extracts the generalizable pattern — not "step_size=0.005 worked"
      but "smaller step sizes prevent divergence on steep landscapes."

    Analysts run in parallel via ``ThreadPoolExecutor`` because the
    existing codebase is synchronous and each analyst call is an
    independent LLM request.

    The ``Lesson`` dataclass is the core data type that flows through
    the entire Trace2Skill pipeline: trajectory_analyst -> consolidator
    -> skill_directory -> hypothesis_generator prompt.

Spec: REQ-AUTO-011, SCENARIO-AUTO-008, SCENARIO-AUTO-009
"""

from __future__ import annotations

import json
import logging
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from carnot.autoresearch.baselines import BaselineRecord
    from carnot.autoresearch.experiment_log import ExperimentEntry

logger = logging.getLogger(__name__)


@dataclass
class Lesson:
    """A structured insight extracted from an experiment trajectory.

    **Researcher summary:**
        The atomic unit of knowledge in the Trace2Skill pipeline. Each
        lesson captures one generalizable insight — a failure pattern to
        avoid or a success pattern to replicate.

    **Detailed explanation for engineers:**
        Lessons flow through the pipeline:

        1. Created by ``analyze_error`` or ``analyze_success``
        2. Accumulated in the orchestrator's pending list
        3. Consolidated by ``consolidate_lessons`` (dedup, merge, filter)
        4. Stored in ``SkillDirectory.lessons``
        5. Serialized into the hypothesis generator's prompt

        Fields:
        - ``title``: Short name for the lesson (e.g., "Gradient clipping
          needed for steep landscapes")
        - ``description``: Full explanation of the insight, written to be
          useful as prompt context for future hypothesis generation
        - ``examples``: Concrete experiment IDs or code snippets that
          demonstrate this lesson
        - ``confidence``: 0.0-1.0, how well-supported this lesson is.
          Higher when multiple experiments confirm it. The consolidator
          boosts confidence on deduplication.
        - ``applicable_benchmarks``: Which benchmarks this applies to
          (e.g., ["rosenbrock"]) or ["all"] for universal lessons
        - ``model_tier``: Which model tier this was learned on ("ising",
          "gibbs", "boltzmann", or "all"). Used for cross-tier transfer.
        - ``lesson_type``: "error_pattern" or "success_pattern"
        - ``source_experiment_id``: The experiment that produced this lesson

    Spec: REQ-AUTO-011
    """

    title: str
    description: str
    examples: list[str] = field(default_factory=list)
    confidence: float = 0.5
    applicable_benchmarks: list[str] = field(default_factory=lambda: ["all"])
    model_tier: str = "all"
    lesson_type: str = "error_pattern"
    source_experiment_id: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Convert to a plain dict for JSON serialization.

        Spec: REQ-AUTO-011
        """
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Lesson:
        """Create a Lesson from a plain dict (e.g., loaded from JSON).

        Spec: REQ-AUTO-011
        """
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class AnalystConfig:
    """Configuration for trajectory analyst LLM calls.

    **Researcher summary:**
        Controls the LLM endpoint and behavior for error/success analysis.
        Follows the same pattern as ``GeneratorConfig``.

    **Detailed explanation for engineers:**
        - ``api_base``: URL of the OpenAI-compatible API endpoint.
        - ``model``: Model identifier (e.g., "sonnet", "haiku").
          Consider using a cheaper/faster model than the hypothesis
          generator since analysis is less creative.
        - ``temperature``: Lower than generation (0.3 vs 0.7) because
          analysis is diagnostic, not creative.
        - ``max_workers``: Thread pool size for parallel analysis.
        - ``api_key``: API key. Default "not-needed" for local bridges.

    Spec: REQ-AUTO-011
    """

    api_base: str = "http://localhost:8080/v1"
    model: str = "sonnet"
    api_key: str = "not-needed"
    temperature: float = 0.3
    max_workers: int = 4


# System prompts for the two analyst types.
# These are long because they frame the LLM's diagnostic task precisely.

ERROR_ANALYST_PROMPT = """\
You are an expert Energy-Based Model (EBM) diagnostician analyzing a FAILED \
autoresearch hypothesis. Your job is to determine the ROOT CAUSE of the \
failure — not just that it failed, but WHY it failed and what generalizable \
lesson can be drawn.

## Your analysis process

1. Read the hypothesis code carefully
2. Examine the sandbox metrics and error messages
3. Compare against the baseline metrics
4. Identify the specific mechanism of failure (e.g., gradient explosion, \
   step size too large for landscape curvature, wrong sampler for topology)
5. Generalize: what property of the energy landscape or algorithm caused this?

## Output format

Return a JSON object with these fields:
```json
{
  "title": "Short name for the failure pattern (max 80 chars)",
  "description": "Detailed explanation of why this failed and how to avoid it. \
Write this as advice for a future hypothesis generator.",
  "applicable_benchmarks": ["benchmark_names_where_this_applies"],
  "confidence": 0.7
}
```

Return ONLY the JSON object, no other text.
"""

SUCCESS_ANALYST_PROMPT = """\
You are an expert Energy-Based Model (EBM) analyst examining a SUCCESSFUL \
autoresearch hypothesis. Your job is to extract the GENERALIZABLE PATTERN — \
not "step_size=0.005 worked" but "smaller step sizes prevent divergence on \
steep landscapes."

## Your analysis process

1. Read the hypothesis code and the improvement metrics
2. Identify what specific change produced the improvement
3. Abstract from the specific parameters to the general principle
4. Consider which benchmarks and model tiers this principle applies to

## Output format

Return a JSON object with these fields:
```json
{
  "title": "Short name for the success pattern (max 80 chars)",
  "description": "Detailed explanation of the generalizable optimization \
pattern. Write this as a reusable strategy for future hypotheses.",
  "applicable_benchmarks": ["benchmark_names_where_this_applies"],
  "confidence": 0.8
}
```

Return ONLY the JSON object, no other text.
"""


def _build_analyst_context(
    entry: ExperimentEntry,
    baselines: BaselineRecord,
) -> str:
    """Build the context message for an analyst LLM call.

    Includes the hypothesis code, sandbox results, evaluation verdict,
    and current baseline metrics for comparison.

    Spec: REQ-AUTO-011
    """
    parts: list[str] = []

    parts.append("## Hypothesis Code\n```python")
    parts.append(entry.hypothesis_code)
    parts.append("```\n")

    parts.append(f"## Description: {entry.hypothesis_description}\n")
    parts.append(f"## Sandbox Success: {entry.sandbox_success}")
    parts.append(f"## Timed Out: {entry.sandbox_timed_out}\n")

    if entry.sandbox_error:
        parts.append(f"## Sandbox Error\n{entry.sandbox_error}\n")

    if entry.sandbox_metrics:
        parts.append("## Sandbox Metrics")
        for name, metrics in entry.sandbox_metrics.items():
            if isinstance(metrics, dict):
                parts.append(f"- **{name}**: {metrics}")
        parts.append("")

    parts.append(f"## Evaluation Verdict: {entry.eval_verdict}")
    parts.append(f"## Evaluation Reason: {entry.eval_reason}")

    if entry.eval_improvements:
        parts.append(f"## Improvements: {', '.join(entry.eval_improvements)}")
    if entry.eval_regressions:
        parts.append(f"## Regressions: {', '.join(entry.eval_regressions)}")

    parts.append("\n## Current Baselines")
    if baselines.benchmarks:
        for name, bm in sorted(baselines.benchmarks.items()):
            parts.append(
                f"- **{name}**: energy={bm.final_energy:.6f}, time={bm.wall_clock_seconds:.2f}s"
            )

    return "\n".join(parts)


def _parse_lesson_json(response: str) -> dict[str, Any] | None:
    """Extract a JSON object from an LLM response.

    Tries direct JSON parse first, then falls back to extracting
    from a code block. Returns None if parsing fails.

    Spec: REQ-AUTO-011
    """
    # Try direct parse
    text = response.strip()
    try:
        return json.loads(text)  # type: ignore[no-any-return]
    except (json.JSONDecodeError, ValueError):
        pass

    # Try extracting from code block
    match = re.search(r"```(?:json)?\s*\n(.*?)```", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1).strip())  # type: ignore[no-any-return]
        except (json.JSONDecodeError, ValueError):
            pass

    # Try finding { ... } in the text
    match = re.search(r"\{[^{}]*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))  # type: ignore[no-any-return]
        except (json.JSONDecodeError, ValueError):
            pass

    return None


def analyze_error(
    config: AnalystConfig,
    entry: ExperimentEntry,
    baselines: BaselineRecord,
) -> Lesson | None:
    """Dispatch an error analyst to diagnose a failed hypothesis.

    **Researcher summary:**
        Uses an LLM to deeply diagnose WHY a hypothesis failed, producing
        a structured Lesson with root cause analysis.

    **Detailed explanation for engineers:**
        Sends the full experiment trajectory (hypothesis code, sandbox
        output, evaluation verdict) to an LLM with the error analyst
        system prompt. The LLM returns a JSON lesson which is parsed
        into a ``Lesson`` object.

        Returns ``None`` if the LLM call fails or the response can't
        be parsed. The caller (orchestrator) handles this gracefully
        by skipping the lesson.

    Args:
        config: LLM API settings for the analyst.
        entry: The failed experiment's full record.
        baselines: Current baseline metrics for comparison.

    Returns:
        A Lesson with lesson_type="error_pattern", or None on failure.

    Spec: REQ-AUTO-011, SCENARIO-AUTO-008
    """
    try:
        from openai import OpenAI
    except ImportError:
        logger.warning("openai package not installed, skipping error analysis")
        return None

    context = _build_analyst_context(entry, baselines)

    try:
        client = OpenAI(base_url=config.api_base, api_key=config.api_key)
        response = client.chat.completions.create(
            model=config.model,
            messages=[
                {"role": "system", "content": ERROR_ANALYST_PROMPT},
                {"role": "user", "content": context},
            ],
            temperature=config.temperature,
        )
        raw = response.choices[0].message.content or ""
    except Exception:
        logger.exception("Error analyst LLM call failed")
        return None

    parsed = _parse_lesson_json(raw)
    if parsed is None:
        logger.warning("Could not parse error analyst response: %s...", raw[:200])
        return None

    return Lesson(
        title=parsed.get("title", "Unknown error pattern"),
        description=parsed.get("description", raw),
        examples=[entry.id],
        confidence=float(parsed.get("confidence", 0.5)),
        applicable_benchmarks=parsed.get("applicable_benchmarks", ["all"]),
        model_tier="all",
        lesson_type="error_pattern",
        source_experiment_id=entry.id,
    )


def analyze_success(
    config: AnalystConfig,
    entry: ExperimentEntry,
    baselines: BaselineRecord,
) -> Lesson | None:
    """Dispatch a success analyst to extract a generalizable pattern.

    **Researcher summary:**
        Uses an LLM to extract the generalizable optimization pattern
        from a successful hypothesis, producing a structured Lesson.

    **Detailed explanation for engineers:**
        Same structure as ``analyze_error`` but with a different system
        prompt that focuses on extracting reusable strategies rather than
        diagnosing failures.

    Args:
        config: LLM API settings for the analyst.
        entry: The accepted experiment's full record.
        baselines: Current baseline metrics for comparison.

    Returns:
        A Lesson with lesson_type="success_pattern", or None on failure.

    Spec: REQ-AUTO-011, SCENARIO-AUTO-009
    """
    try:
        from openai import OpenAI
    except ImportError:
        logger.warning("openai package not installed, skipping success analysis")
        return None

    context = _build_analyst_context(entry, baselines)

    try:
        client = OpenAI(base_url=config.api_base, api_key=config.api_key)
        response = client.chat.completions.create(
            model=config.model,
            messages=[
                {"role": "system", "content": SUCCESS_ANALYST_PROMPT},
                {"role": "user", "content": context},
            ],
            temperature=config.temperature,
        )
        raw = response.choices[0].message.content or ""
    except Exception:
        logger.exception("Success analyst LLM call failed")
        return None

    parsed = _parse_lesson_json(raw)
    if parsed is None:
        logger.warning("Could not parse success analyst response: %s...", raw[:200])
        return None

    return Lesson(
        title=parsed.get("title", "Unknown success pattern"),
        description=parsed.get("description", raw),
        examples=[entry.id],
        confidence=float(parsed.get("confidence", 0.5)),
        applicable_benchmarks=parsed.get("applicable_benchmarks", ["all"]),
        model_tier="all",
        lesson_type="success_pattern",
        source_experiment_id=entry.id,
    )


def analyze_batch(
    config: AnalystConfig,
    entries: list[ExperimentEntry],
    baselines: BaselineRecord,
) -> list[Lesson]:
    """Analyze multiple experiment trajectories in parallel.

    **Researcher summary:**
        Dispatches error/success analysts concurrently for a batch of
        experiments. Returns all successfully extracted lessons.

    **Detailed explanation for engineers:**
        Uses ``ThreadPoolExecutor`` to run analyst LLM calls in parallel.
        Each entry is routed to either ``analyze_error`` (if rejected) or
        ``analyze_success`` (if accepted) based on its outcome.

        Entries with outcome "pending_review" are analyzed as errors
        (they didn't fully pass). Entries with no outcome are skipped.

        Returns only non-None results — failed analyses are logged
        but don't block the batch.

    Args:
        config: LLM API settings for the analysts.
        entries: Experiment entries to analyze.
        baselines: Current baseline metrics for comparison.

    Returns:
        List of extracted Lessons (may be shorter than input if some
        analyses failed).

    Spec: REQ-AUTO-011
    """
    if not entries:
        return []

    lessons: list[Lesson] = []

    with ThreadPoolExecutor(max_workers=config.max_workers) as executor:
        futures = {}
        for entry in entries:
            if entry.outcome == "accepted":
                fut = executor.submit(analyze_success, config, entry, baselines)
            elif entry.outcome in ("rejected", "pending_review"):
                fut = executor.submit(analyze_error, config, entry, baselines)
            else:
                continue
            futures[fut] = entry.id

        for future in as_completed(futures):
            exp_id = futures[future]
            try:
                lesson = future.result()
                if lesson is not None:
                    lessons.append(lesson)
                    logger.info("Extracted lesson from %s: %s", exp_id, lesson.title)
                else:
                    logger.info("No lesson extracted from %s", exp_id)
            except Exception:
                logger.exception("Analyst failed for %s", exp_id)

    return lessons
