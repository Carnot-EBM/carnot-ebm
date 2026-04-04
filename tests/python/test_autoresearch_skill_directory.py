"""Tests for skill directory: persistence, prompt serialization, cross-tier transfer.

Spec coverage: REQ-AUTO-012, REQ-AUTO-014, SCENARIO-AUTO-011
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

from carnot.autoresearch.skill_directory import (
    SkillDirectory,
    SkillDirectoryConfig,
)
from carnot.autoresearch.trajectory_analyst import Lesson

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_lesson(
    title: str = "test lesson",
    lesson_type: str = "success_pattern",
    benchmarks: list[str] | None = None,
    model_tier: str = "all",
    confidence: float = 0.7,
) -> Lesson:
    return Lesson(
        title=title,
        description=f"Description for {title}",
        examples=["exp-001"],
        confidence=confidence,
        applicable_benchmarks=benchmarks or ["all"],
        model_tier=model_tier,
        lesson_type=lesson_type,
        source_experiment_id="exp-001",
    )


# ---------------------------------------------------------------------------
# Tests: SkillDirectory basics
# ---------------------------------------------------------------------------


class TestSkillDirectoryBasics:
    """Tests for basic skill directory operations."""

    def test_empty_directory(self) -> None:
        """REQ-AUTO-012: empty directory has no lessons."""
        sd = SkillDirectory()
        assert len(sd) == 0
        assert sd.playbook == ""
        assert sd.scripts == {}
        assert sd.references == {}

    def test_evolve_adds_lessons(self) -> None:
        """REQ-AUTO-012: evolve appends lessons."""
        sd = SkillDirectory()
        lesson = _make_lesson()
        patch = sd.evolve([lesson])
        assert len(sd) == 1
        assert sd.lessons[0].title == "test lesson"
        assert len(patch.added_lessons) == 1

    def test_evolve_updates_playbook(self) -> None:
        """REQ-AUTO-012: evolve rebuilds the playbook."""
        sd = SkillDirectory()
        sd.evolve([_make_lesson(lesson_type="success_pattern")])
        assert "What Works" in sd.playbook
        assert "test lesson" in sd.playbook

    def test_evolve_error_pattern_in_playbook(self) -> None:
        """REQ-AUTO-012: error patterns appear in 'What to Avoid' section."""
        sd = SkillDirectory()
        sd.evolve([_make_lesson(lesson_type="error_pattern")])
        assert "What to Avoid" in sd.playbook

    def test_evolve_updates_references(self) -> None:
        """REQ-AUTO-012: benchmark-specific lessons go to references."""
        sd = SkillDirectory()
        sd.evolve([_make_lesson(benchmarks=["rosenbrock"])])
        assert "rosenbrock" in sd.references
        assert len(sd.references["rosenbrock"]) == 1

    def test_evolve_skips_all_benchmarks_in_references(self) -> None:
        """REQ-AUTO-012: 'all' benchmarks don't create reference entries."""
        sd = SkillDirectory()
        sd.evolve([_make_lesson(benchmarks=["all"])])
        assert sd.references == {}

    def test_max_lessons_cap(self) -> None:
        """REQ-AUTO-012: lessons are evicted when cap is exceeded."""
        config = SkillDirectoryConfig(max_lessons=3)
        sd = SkillDirectory(config)
        lessons = [_make_lesson(title=f"lesson-{i}") for i in range(5)]
        patch = sd.evolve(lessons)
        assert len(sd) == 3
        assert sd.lessons[0].title == "lesson-2"  # oldest evicted
        assert len(patch.removed_lesson_titles) == 2

    def test_evolve_extracts_code_snippets(self) -> None:
        """REQ-AUTO-012: code examples are stored as scripts."""
        sd = SkillDirectory()
        lesson = _make_lesson()
        lesson.examples = ["def run(d): return {'energy': -5.0}"]
        sd.evolve([lesson])
        assert len(sd.scripts) == 1

    def test_evolve_max_scripts_cap(self) -> None:
        """REQ-AUTO-012: scripts are capped at max_scripts."""
        config = SkillDirectoryConfig(max_scripts=2)
        sd = SkillDirectory(config)
        lessons = []
        for i in range(5):
            lesson = _make_lesson(title=f"script-{i}")
            lesson.examples = [f"def run_{i}(d): pass"]
            lessons.append(lesson)
        sd.evolve(lessons)
        assert len(sd.scripts) <= 2

    def test_empty_playbook_text(self) -> None:
        """REQ-AUTO-012: empty directory produces 'no lessons' playbook."""
        sd = SkillDirectory()
        sd.playbook = sd._build_playbook()
        assert "No lessons learned yet" in sd.playbook

    def test_to_dict(self) -> None:
        """REQ-AUTO-012: to_dict provides summary stats."""
        sd = SkillDirectory()
        sd.evolve([_make_lesson()])
        d = sd.to_dict()
        assert d["lesson_count"] == 1
        assert d["script_count"] == 0


# ---------------------------------------------------------------------------
# Tests: to_prompt_context
# ---------------------------------------------------------------------------


class TestPromptContext:
    """Tests for prompt serialization."""

    def test_empty_returns_empty(self) -> None:
        """REQ-AUTO-012: no lessons means empty context."""
        sd = SkillDirectory()
        assert sd.to_prompt_context() == ""

    def test_includes_playbook(self) -> None:
        """REQ-AUTO-012: context includes playbook content."""
        sd = SkillDirectory()
        sd.evolve([_make_lesson()])
        ctx = sd.to_prompt_context()
        assert "Optimization Playbook" in ctx

    def test_includes_references(self) -> None:
        """REQ-AUTO-012: context includes benchmark-specific notes."""
        sd = SkillDirectory()
        sd.evolve([_make_lesson(benchmarks=["rosenbrock"])])
        ctx = sd.to_prompt_context()
        assert "rosenbrock" in ctx

    def test_truncation(self) -> None:
        """REQ-AUTO-012: output is capped at max_prompt_chars."""
        config = SkillDirectoryConfig(max_prompt_chars=100)
        sd = SkillDirectory(config)
        lessons = [_make_lesson(title=f"long lesson {i}") for i in range(20)]
        sd.evolve(lessons)
        ctx = sd.to_prompt_context()
        assert len(ctx) <= 100
        assert "[truncated]" in ctx

    def test_cross_tier_transfer(self) -> None:
        """REQ-AUTO-014, SCENARIO-AUTO-011: includes lessons from other tiers."""
        sd = SkillDirectory()
        sd.evolve(
            [
                _make_lesson(title="Ising HMC insight", model_tier="ising", confidence=0.8),
            ]
        )
        ctx = sd.to_prompt_context(model_tier="gibbs")
        assert "Ising HMC insight" in ctx
        assert "Cross-Tier" in ctx

    def test_cross_tier_skips_low_confidence(self) -> None:
        """REQ-AUTO-014: low-confidence cross-tier lessons are excluded from cross-tier section."""
        sd = SkillDirectory()
        sd.evolve(
            [
                _make_lesson(title="Weak insight", model_tier="ising", confidence=0.3),
            ]
        )
        ctx = sd.to_prompt_context(model_tier="gibbs")
        # Lesson appears in the playbook (it's still a lesson) but NOT in cross-tier section
        assert "Cross-Tier" not in ctx

    def test_cross_tier_skips_same_tier(self) -> None:
        """REQ-AUTO-014: same-tier lessons aren't in cross-tier section."""
        sd = SkillDirectory()
        sd.evolve(
            [
                _make_lesson(title="Ising lesson", model_tier="ising"),
            ]
        )
        ctx = sd.to_prompt_context(model_tier="ising")
        assert "Cross-Tier" not in ctx

    def test_no_cross_tier_when_no_model_tier(self) -> None:
        """REQ-AUTO-014: no cross-tier section when model_tier is None."""
        sd = SkillDirectory()
        sd.evolve([_make_lesson(model_tier="ising")])
        ctx = sd.to_prompt_context(model_tier=None)
        # Should still include the lesson in the playbook, just not in cross-tier
        assert "test lesson" in ctx


# ---------------------------------------------------------------------------
# Tests: save/load
# ---------------------------------------------------------------------------


class TestPersistence:
    """Tests for disk persistence."""

    def test_save_creates_directory(self, tmp_path: Path) -> None:
        """REQ-AUTO-012: save creates the directory structure."""
        config = SkillDirectoryConfig(path=tmp_path / "skills")
        sd = SkillDirectory(config)
        sd.evolve([_make_lesson(benchmarks=["rosenbrock"])])
        sd.save()

        assert (tmp_path / "skills" / "SKILL.md").exists()
        assert (tmp_path / "skills" / "lessons.json").exists()
        assert (tmp_path / "skills" / "scripts").is_dir()
        assert (tmp_path / "skills" / "references").is_dir()

    def test_save_load_roundtrip(self, tmp_path: Path) -> None:
        """REQ-AUTO-012: save -> load preserves state."""
        config = SkillDirectoryConfig(path=tmp_path / "skills")
        sd = SkillDirectory(config)
        sd.evolve(
            [
                _make_lesson(title="lesson 1", benchmarks=["double_well"]),
                _make_lesson(title="lesson 2", lesson_type="error_pattern"),
            ]
        )
        sd.save()

        loaded = SkillDirectory.load(config)
        assert len(loaded) == 2
        assert loaded.lessons[0].title == "lesson 1"
        assert loaded.lessons[1].title == "lesson 2"
        assert "Optimization Playbook" in loaded.playbook
        assert "double_well" in loaded.references

    def test_load_nonexistent_returns_empty(self, tmp_path: Path) -> None:
        """REQ-AUTO-012: loading nonexistent path gives empty directory."""
        config = SkillDirectoryConfig(path=tmp_path / "nonexistent")
        sd = SkillDirectory.load(config)
        assert len(sd) == 0

    def test_load_corrupt_json_returns_empty_lessons(self, tmp_path: Path) -> None:
        """REQ-AUTO-012: corrupt JSON degrades gracefully."""
        config = SkillDirectoryConfig(path=tmp_path / "skills")
        root = config.path
        root.mkdir(parents=True)
        (root / "lessons.json").write_text("not valid json{{{")
        (root / "SKILL.md").write_text("# Playbook")

        sd = SkillDirectory.load(config)
        assert len(sd) == 0
        assert sd.playbook == "# Playbook"

    def test_save_scripts_and_references(self, tmp_path: Path) -> None:
        """REQ-AUTO-012: scripts and references persist correctly."""
        config = SkillDirectoryConfig(path=tmp_path / "skills")
        sd = SkillDirectory(config)

        lesson = _make_lesson(benchmarks=["ackley"])
        lesson.examples = ["import jax\ndef sampler(): pass"]
        sd.evolve([lesson])
        sd.save()

        loaded = SkillDirectory.load(config)
        assert len(loaded.scripts) == 1
        assert "ackley" in loaded.references
