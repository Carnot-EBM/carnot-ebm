"""Skill directory: persistent, evolving optimization playbook.

**Researcher summary:**
    Implements the Trace2Skill "skill directory" concept — a structured
    knowledge base that accumulates lessons from autoresearch experiments.
    Replaces the shallow ``recent_failures`` list with a rich, evolving
    optimization guide that the hypothesis generator uses as context.

**Detailed explanation for engineers:**
    The skill directory is the knowledge layer between trajectory analysis
    and hypothesis generation. It stores:

    1. **lessons.json**: All accumulated ``Lesson`` objects with confidence
       scores, benchmark tags, and model tier info. This is the structured
       data store.

    2. **SKILL.md**: A natural-language "optimization playbook" that
       summarizes the lessons into a coherent guide. Periodically rewritten
       by an LLM from the current lesson set. This is what gets injected
       into the hypothesis generator's prompt.

    3. **scripts/**: Proven sampler configurations extracted from successful
       hypotheses. These are reusable code snippets.

    4. **references/**: Benchmark-specific edge cases. Low-frequency lessons
       that are too niche for the main playbook but valuable for specific
       benchmarks (e.g., "Rosenbrock needs gradient clipping").

    The ``to_prompt_context()`` method serializes the skill directory into
    a prompt section. It supports cross-tier transfer: when generating for
    the Gibbs tier, it includes relevant Ising lessons.

    Persistence is file-based: JSON for lessons, plain text for SKILL.md,
    individual files for scripts and references. This matches the existing
    codebase pattern (e.g., ``ExperimentLog.save/load``).

Spec: REQ-AUTO-012, REQ-AUTO-014
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from carnot.autoresearch.trajectory_analyst import Lesson

logger = logging.getLogger(__name__)


@dataclass
class SkillDirectoryConfig:
    """Configuration for the skill directory.

    **Researcher summary:**
        Controls where the skill directory is stored and size limits.

    **Detailed explanation for engineers:**
        - ``path``: Root directory for persistence. Created on first save.
        - ``max_lessons``: Cap on total lessons stored. Oldest lessons
          are evicted when this limit is exceeded. Default 200.
        - ``max_prompt_chars``: Maximum character count for the
          ``to_prompt_context()`` output. Prevents prompt bloat.
          Default 4000 (~1000 tokens).
        - ``max_scripts``: Maximum number of stored code snippets.

    Spec: REQ-AUTO-012
    """

    path: Path = field(default_factory=lambda: Path("skill_directory"))
    max_lessons: int = 200
    max_prompt_chars: int = 4000
    max_scripts: int = 50


@dataclass
class SkillPatch:
    """Record of changes made to the skill directory in one evolution step.

    **Researcher summary:**
        Audit trail for skill directory updates — what lessons were added,
        what was removed, and what the playbook looks like now.

    **Detailed explanation for engineers:**
        Returned by ``SkillDirectory.evolve()`` so the caller (orchestrator)
        can log the changes. Not persisted separately — the skill directory
        state after applying the patch is what gets saved.

    Spec: REQ-AUTO-012
    """

    added_lessons: list[Lesson] = field(default_factory=list)
    removed_lesson_titles: list[str] = field(default_factory=list)
    updated_playbook: str = ""
    patch_timestamp: str = ""


class SkillDirectory:
    """Persistent, evolving optimization playbook for autoresearch.

    **Researcher summary:**
        The skill directory is the brain of the Trace2Skill pipeline.
        It accumulates lessons, organizes them, and provides structured
        context to the hypothesis generator. Think of it as a growing
        textbook that the LLM reads before proposing each hypothesis.

    **Detailed explanation for engineers:**
        Core operations:

        - ``evolve(new_lessons)``: Integrate new lessons into the directory.
          Appends to ``lessons``, caps at ``max_lessons`` (oldest-first
          eviction), extracts code snippets to ``scripts``, adds benchmark-
          specific notes to ``references``. Optionally rewrites the
          playbook via LLM.

        - ``to_prompt_context(model_tier)``: Serialize the directory into
          a prompt section for the hypothesis generator. Includes the
          playbook, recent high-confidence lessons, and cross-tier
          lessons if a specific tier is requested.

        - ``save()`` / ``load()``: File-based persistence.

        Thread safety: Not thread-safe. The orchestrator calls these
        sequentially between hypothesis evaluations.

    For example::

        sd = SkillDirectory(SkillDirectoryConfig(path=Path("./skills")))
        sd.evolve([lesson1, lesson2])
        context = sd.to_prompt_context(model_tier="gibbs")
        sd.save()

    Spec: REQ-AUTO-012, REQ-AUTO-014
    """

    def __init__(self, config: SkillDirectoryConfig | None = None) -> None:
        self.config = config or SkillDirectoryConfig()
        self.lessons: list[Lesson] = []
        self.playbook: str = ""
        self.scripts: dict[str, str] = {}
        self.references: dict[str, list[str]] = {}

    def evolve(self, new_lessons: list[Lesson]) -> SkillPatch:
        """Integrate new lessons into the skill directory.

        **Researcher summary:**
            Adds lessons, enforces size cap, updates references and scripts.
            Returns a SkillPatch describing what changed.

        **Detailed explanation for engineers:**
            Steps:
            1. Append new lessons to ``self.lessons``
            2. If over ``max_lessons``, evict the oldest
            3. For each new lesson, add benchmark-specific notes to
               ``references`` and code examples to ``scripts``
            4. Rebuild the playbook from current lessons (simple template,
               no LLM needed — the LLM-rewritten version is handled by
               ``evolve_playbook_with_llm`` which the orchestrator can
               call separately)
            5. Return a SkillPatch audit record

        Args:
            new_lessons: Lessons to integrate.

        Returns:
            SkillPatch describing the changes made.

        Spec: REQ-AUTO-012
        """
        patch = SkillPatch(
            added_lessons=list(new_lessons),
            patch_timestamp=datetime.now(UTC).isoformat(),
        )

        # Append new lessons
        self.lessons.extend(new_lessons)

        # Enforce size cap — evict oldest first
        removed: list[str] = []
        while len(self.lessons) > self.config.max_lessons:
            evicted = self.lessons.pop(0)
            removed.append(evicted.title)
        patch.removed_lesson_titles = removed

        # Update references (benchmark-specific edge cases)
        for lesson in new_lessons:
            for bench in lesson.applicable_benchmarks:
                if bench == "all":
                    continue
                if bench not in self.references:
                    self.references[bench] = []
                note = f"[{lesson.lesson_type}] {lesson.title}: {lesson.description}"
                if note not in self.references[bench]:
                    self.references[bench].append(note)

        # Extract code snippets from examples
        for lesson in new_lessons:
            for example in lesson.examples:
                # Only store examples that look like code (contain 'def ' or 'import ')
                if "def " in example or "import " in example:
                    script_name = lesson.title.lower().replace(" ", "_")[:40]
                    if len(self.scripts) < self.config.max_scripts:
                        self.scripts[script_name] = example

        # Rebuild playbook from lessons (simple template)
        self.playbook = self._build_playbook()
        patch.updated_playbook = self.playbook

        return patch

    def _build_playbook(self) -> str:
        """Build the SKILL.md playbook from current lessons.

        Groups lessons by type (error vs success) and sorts by
        confidence. This is a deterministic rebuild — no LLM needed.

        Spec: REQ-AUTO-012
        """
        if not self.lessons:
            return "# Optimization Playbook\n\nNo lessons learned yet.\n"

        parts: list[str] = ["# Optimization Playbook\n"]

        # Success patterns (what works)
        successes = sorted(
            [ls for ls in self.lessons if ls.lesson_type == "success_pattern"],
            key=lambda ls: ls.confidence,
            reverse=True,
        )
        if successes:
            parts.append("## What Works\n")
            for lesson in successes:
                benches = ", ".join(lesson.applicable_benchmarks)
                parts.append(
                    f"- **{lesson.title}** (confidence: {lesson.confidence:.1f}, "
                    f"benchmarks: {benches}): {lesson.description}"
                )
            parts.append("")

        # Error patterns (what to avoid)
        errors = sorted(
            [ls for ls in self.lessons if ls.lesson_type == "error_pattern"],
            key=lambda ls: ls.confidence,
            reverse=True,
        )
        if errors:
            parts.append("## What to Avoid\n")
            for lesson in errors:
                benches = ", ".join(lesson.applicable_benchmarks)
                parts.append(
                    f"- **{lesson.title}** (confidence: {lesson.confidence:.1f}, "
                    f"benchmarks: {benches}): {lesson.description}"
                )
            parts.append("")

        return "\n".join(parts)

    def to_prompt_context(self, model_tier: str | None = None) -> str:
        """Serialize skill directory into prompt context for the generator.

        **Researcher summary:**
            Produces a text block that replaces the old ``recent_failures``
            list. Includes the playbook, recent lessons, and cross-tier
            knowledge.

        **Detailed explanation for engineers:**
            The output is injected as ``extra_context`` into the hypothesis
            generator's prompt. It includes:

            1. The SKILL.md playbook (truncated to fit)
            2. Recent high-confidence lessons (up to char limit)
            3. If ``model_tier`` is specified: lessons from other tiers
               that might transfer (REQ-AUTO-014)

            The total output is capped at ``max_prompt_chars`` to prevent
            prompt bloat.

        Args:
            model_tier: Target model tier for cross-tier transfer. If None,
                includes all lessons. If "gibbs", includes "ising" and "all"
                lessons too.

        Returns:
            A text block suitable for prompt injection.

        Spec: REQ-AUTO-012, REQ-AUTO-014, SCENARIO-AUTO-011
        """
        if not self.lessons:
            return ""

        parts: list[str] = []

        # Include the playbook
        if self.playbook:
            parts.append(self.playbook)

        # Cross-tier transfer: include lessons from other tiers
        if model_tier:
            cross_tier = [
                ls
                for ls in self.lessons
                if ls.model_tier != model_tier and ls.model_tier != "all" and ls.confidence >= 0.5
            ]
            if cross_tier:
                parts.append("\n## Cross-Tier Lessons (from other model tiers)\n")
                for lesson in sorted(cross_tier, key=lambda ls: ls.confidence, reverse=True):
                    parts.append(
                        f"- [{lesson.model_tier}] **{lesson.title}**: {lesson.description}"
                    )
                parts.append("")

        # Benchmark-specific references
        if self.references:
            parts.append("\n## Benchmark-Specific Notes\n")
            for bench, notes in sorted(self.references.items()):
                parts.append(f"### {bench}")
                for note in notes[-3:]:  # Last 3 notes per benchmark
                    parts.append(f"- {note}")
            parts.append("")

        result = "\n".join(parts)

        # Truncate to max chars
        if len(result) > self.config.max_prompt_chars:
            result = result[: self.config.max_prompt_chars - 20] + "\n\n[truncated]"

        return result

    def save(self) -> None:
        """Persist the skill directory to disk.

        **Detailed explanation for engineers:**
            Creates the directory structure if it doesn't exist:
            ```
            skill_directory/
                SKILL.md
                lessons.json
                scripts/
                    *.py
                references/
                    *.md
            ```

        Spec: REQ-AUTO-012
        """
        root = self.config.path
        root.mkdir(parents=True, exist_ok=True)

        # Save playbook
        (root / "SKILL.md").write_text(self.playbook)

        # Save lessons as JSON
        lessons_data = [ls.to_dict() for ls in self.lessons]
        (root / "lessons.json").write_text(json.dumps(lessons_data, indent=2))

        # Save scripts
        scripts_dir = root / "scripts"
        scripts_dir.mkdir(exist_ok=True)
        for name, code in self.scripts.items():
            safe_name = "".join(c if c.isalnum() or c == "_" else "_" for c in name)
            (scripts_dir / f"{safe_name}.py").write_text(code)

        # Save references
        refs_dir = root / "references"
        refs_dir.mkdir(exist_ok=True)
        for bench, notes in self.references.items():
            safe_name = "".join(c if c.isalnum() or c == "_" else "_" for c in bench)
            (refs_dir / f"{safe_name}.md").write_text("\n".join(notes))

        logger.info("Skill directory saved to %s", root)

    @classmethod
    def load(cls, config: SkillDirectoryConfig) -> SkillDirectory:
        """Load a skill directory from disk.

        **Detailed explanation for engineers:**
            Reads the directory structure created by ``save()``. If the
            directory doesn't exist, returns an empty SkillDirectory.
            Missing sub-components (e.g., no scripts/) are handled
            gracefully — the directory starts with empty dicts.

        Spec: REQ-AUTO-012
        """
        sd = cls(config)
        root = config.path

        if not root.exists():
            return sd

        # Load playbook
        playbook_path = root / "SKILL.md"
        if playbook_path.exists():
            sd.playbook = playbook_path.read_text()

        # Load lessons
        lessons_path = root / "lessons.json"
        if lessons_path.exists():
            try:
                data = json.loads(lessons_path.read_text())
                sd.lessons = [Lesson.from_dict(d) for d in data]
            except (json.JSONDecodeError, TypeError):
                logger.warning("Could not parse lessons.json, starting fresh")

        # Load scripts
        scripts_dir = root / "scripts"
        if scripts_dir.exists():
            for script_file in scripts_dir.glob("*.py"):
                sd.scripts[script_file.stem] = script_file.read_text()

        # Load references
        refs_dir = root / "references"
        if refs_dir.exists():
            for ref_file in refs_dir.glob("*.md"):
                sd.references[ref_file.stem] = ref_file.read_text().splitlines()

        return sd

    def __len__(self) -> int:
        return len(self.lessons)

    def to_dict(self) -> dict[str, Any]:
        """Serialize the skill directory state to a dict for inspection.

        Spec: REQ-AUTO-012
        """
        return {
            "lesson_count": len(self.lessons),
            "script_count": len(self.scripts),
            "reference_benchmarks": list(self.references.keys()),
            "playbook_length": len(self.playbook),
        }
