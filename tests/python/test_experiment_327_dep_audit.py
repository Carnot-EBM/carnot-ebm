"""Tests for the pre-experiment dependency audit tool.

Spec coverage:
  REQ-INFRA-005  — Pre-experiment dependency audit (SCENARIO-INFRA-007, SCENARIO-INFRA-008)
"""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest
import yaml

# ---------------------------------------------------------------------------
# Imports under test — imported lazily to avoid ImportError when the
# module doesn't yet exist (test-first discipline).
# ---------------------------------------------------------------------------

from scripts.experiment_dependency_audit import (
    DependencyAudit,
    build_blocked_artifact,
    check_dependencies,
    extract_required_files,
    load_experiment_prompt,
)


# ---------------------------------------------------------------------------
# DependencyAudit dataclass
# ---------------------------------------------------------------------------


class TestDependencyAudit:
    """REQ-INFRA-005: DependencyAudit holds audit results."""

    def test_fields_accessible(self, tmp_path: Path) -> None:
        """DependencyAudit stores experiment_id, required_files, missing_files, all_present."""
        audit = DependencyAudit(
            experiment_id="exp327",
            required_files=["a.py", "b.json"],
            missing_files=["b.json"],
            all_present=False,
        )
        assert audit.experiment_id == "exp327"
        assert audit.required_files == ["a.py", "b.json"]
        assert audit.missing_files == ["b.json"]
        assert audit.all_present is False

    def test_all_present_true_when_no_missing(self) -> None:
        """all_present is True when missing_files is empty."""
        audit = DependencyAudit(
            experiment_id="exp327",
            required_files=["a.py"],
            missing_files=[],
            all_present=True,
        )
        assert audit.all_present is True
        assert audit.missing_files == []


# ---------------------------------------------------------------------------
# extract_required_files
# ---------------------------------------------------------------------------


class TestExtractRequiredFiles:
    """REQ-INFRA-005: extract_required_files parses the prompt correctly."""

    def test_extracts_two_files(self, tmp_path: Path) -> None:
        """Extracts both file paths from EXISTING CODE TO READ FIRST section."""
        # Create the files so the paths exist (extract doesn't check existence)
        prompt = textwrap.dedent("""\
            CONTEXT: Some context here.

            EXISTING CODE TO READ FIRST:
            - scripts/experiment_template.py — ExperimentTemplate (understand the structure)
            - ops/status.md — current operational status

            TASK: Do something.
            1. Step one.
        """)
        root = str(tmp_path)
        files = extract_required_files(prompt, root)
        assert len(files) == 2
        assert files[0].endswith("scripts/experiment_template.py")
        assert files[1].endswith("ops/status.md")

    def test_returns_empty_when_no_section(self, tmp_path: Path) -> None:
        """Returns [] for prompts with no EXISTING CODE TO READ FIRST section (SCENARIO-INFRA-007)."""
        prompt = "CONTEXT: No files needed.\n\nTASK: Just do it.\n"
        files = extract_required_files(prompt, str(tmp_path))
        assert files == []

    def test_strips_bullet_prefix(self, tmp_path: Path) -> None:
        """Strips the '- ' bullet prefix from each path."""
        prompt = textwrap.dedent("""\
            EXISTING CODE TO READ FIRST:
            - scripts/foo.py — some explanation

            TASK: Go.
        """)
        files = extract_required_files(prompt, str(tmp_path))
        assert len(files) == 1
        assert not files[0].startswith("- ")

    def test_strips_em_dash_comment(self, tmp_path: Path) -> None:
        """Strips explanatory comments after ' — ' (em dash)."""
        prompt = textwrap.dedent("""\
            EXISTING CODE TO READ FIRST:
            - results/foo.json — important results from Exp 123

            TASK: Process.
        """)
        files = extract_required_files(prompt, str(tmp_path))
        assert len(files) == 1
        assert "important" not in files[0]
        assert files[0].endswith("results/foo.json")

    def test_strips_hash_comment(self, tmp_path: Path) -> None:
        """Strips comments after ' # '."""
        prompt = textwrap.dedent("""\
            EXISTING CODE TO READ FIRST:
            - scripts/bar.py # used for training

            TASK: Go.
        """)
        files = extract_required_files(prompt, str(tmp_path))
        assert len(files) == 1
        assert "used for training" not in files[0]
        assert files[0].endswith("scripts/bar.py")

    def test_relative_paths_resolved_to_absolute(self, tmp_path: Path) -> None:
        """Relative paths are resolved relative to project_root."""
        prompt = textwrap.dedent("""\
            EXISTING CODE TO READ FIRST:
            - ops/status.md — status

            TASK: Do.
        """)
        files = extract_required_files(prompt, str(tmp_path))
        assert len(files) == 1
        assert os.path.isabs(files[0])
        assert files[0] == str(tmp_path / "ops" / "status.md")

    def test_absolute_paths_kept_as_is(self, tmp_path: Path) -> None:
        """Absolute paths are preserved without joining to project_root."""
        abs_path = str(tmp_path / "absolute" / "file.py")
        prompt = f"EXISTING CODE TO READ FIRST:\n- {abs_path} — absolute\n\nTASK: Go.\n"
        files = extract_required_files(prompt, str(tmp_path))
        assert len(files) == 1
        assert files[0] == abs_path

    def test_project_root_placeholder_substitution(self, tmp_path: Path) -> None:
        """Substitutes /home/ianblenke/github.com/ianblenke/carnot with actual project_root."""
        placeholder = "/home/ianblenke/github.com/ianblenke/carnot"
        prompt = textwrap.dedent(f"""\
            EXISTING CODE TO READ FIRST:
            - {placeholder}/results/foo.json — some file

            TASK: Go.
        """)
        files = extract_required_files(prompt, str(tmp_path))
        assert len(files) == 1
        # The placeholder should be replaced with tmp_path
        assert placeholder not in files[0]
        assert files[0] == str(tmp_path / "results" / "foo.json")

    def test_braced_project_root_placeholder(self, tmp_path: Path) -> None:
        """Handles {project_root} template placeholder substitution."""
        prompt = textwrap.dedent("""\
            EXISTING CODE TO READ FIRST:
            - {project_root}/results/exp.json — some file

            TASK: Go.
        """)
        files = extract_required_files(prompt, str(tmp_path))
        assert len(files) == 1
        assert "{project_root}" not in files[0]
        assert files[0] == str(tmp_path / "results" / "exp.json")

    def test_stops_at_blank_line_then_task(self, tmp_path: Path) -> None:
        """Stops collecting files when it hits a blank line followed by TASK:."""
        prompt = textwrap.dedent("""\
            EXISTING CODE TO READ FIRST:
            - scripts/one.py — first

            TASK: Do something after the blank line.
            - scripts/two.py — this should NOT be parsed as a file
        """)
        files = extract_required_files(prompt, str(tmp_path))
        assert len(files) == 1
        assert files[0].endswith("scripts/one.py")

    def test_stops_at_non_indented_line_after_blank(self, tmp_path: Path) -> None:
        """Stops at the first non-indented, non-bullet line after a blank line."""
        prompt = textwrap.dedent("""\
            EXISTING CODE TO READ FIRST:
            - scripts/a.py — a

            Some other section not starting with TASK:
            - scripts/b.py — should not be included
        """)
        files = extract_required_files(prompt, str(tmp_path))
        # Only one file: the one before the blank line
        assert len(files) == 1
        assert files[0].endswith("scripts/a.py")

    def test_multiple_files_no_comments(self, tmp_path: Path) -> None:
        """Handles multiple file entries without any comment suffixes."""
        prompt = textwrap.dedent("""\
            EXISTING CODE TO READ FIRST:
            - scripts/a.py
            - scripts/b.py
            - scripts/c.py

            TASK: Go.
        """)
        files = extract_required_files(prompt, str(tmp_path))
        assert len(files) == 3
        assert files[0].endswith("scripts/a.py")
        assert files[1].endswith("scripts/b.py")
        assert files[2].endswith("scripts/c.py")

    def test_skips_non_bullet_lines_in_section(self, tmp_path: Path) -> None:
        """Non-bullet lines inside the section are ignored (only '- ' lines parsed)."""
        prompt = textwrap.dedent("""\
            EXISTING CODE TO READ FIRST:
            - scripts/a.py — a
            This is not a bullet line, should be skipped.
            - scripts/b.py — b

            TASK: Go.
        """)
        files = extract_required_files(prompt, str(tmp_path))
        # Both bullet lines should be parsed; the plain text line is skipped
        assert len(files) == 2
        assert files[0].endswith("scripts/a.py")
        assert files[1].endswith("scripts/b.py")


# ---------------------------------------------------------------------------
# check_dependencies
# ---------------------------------------------------------------------------


class TestCheckDependencies:
    """REQ-INFRA-005: check_dependencies checks each required file for existence."""

    def test_all_present(self, tmp_path: Path) -> None:
        """SCENARIO-INFRA-007: all_present=True when every required file exists."""
        # Create the files
        (tmp_path / "scripts").mkdir()
        (tmp_path / "scripts" / "foo.py").write_text("# foo")
        (tmp_path / "ops").mkdir()
        (tmp_path / "ops" / "status.md").write_text("# status")

        prompt = textwrap.dedent("""\
            EXISTING CODE TO READ FIRST:
            - scripts/foo.py — something
            - ops/status.md — status

            TASK: Go.
        """)
        audit = check_dependencies(prompt, str(tmp_path))
        assert isinstance(audit, DependencyAudit)
        assert audit.all_present is True
        assert audit.missing_files == []
        assert len(audit.required_files) == 2

    def test_missing_file_detected(self, tmp_path: Path) -> None:
        """SCENARIO-INFRA-008: missing file appears in missing_files, all_present=False."""
        (tmp_path / "scripts").mkdir()
        (tmp_path / "scripts" / "real.py").write_text("# real")

        prompt = textwrap.dedent("""\
            EXISTING CODE TO READ FIRST:
            - scripts/real.py — exists
            - results/experiment_999_results.json — does not exist

            TASK: Go.
        """)
        audit = check_dependencies(prompt, str(tmp_path))
        assert audit.all_present is False
        assert len(audit.missing_files) == 1
        assert audit.missing_files[0].endswith("results/experiment_999_results.json")

    def test_empty_required_files_is_all_present(self, tmp_path: Path) -> None:
        """When no EXISTING CODE section, required_files=[] and all_present=True."""
        prompt = "TASK: Nothing needed.\n"
        audit = check_dependencies(prompt, str(tmp_path))
        assert audit.required_files == []
        assert audit.missing_files == []
        assert audit.all_present is True

    def test_experiment_id_defaults_to_unknown(self, tmp_path: Path) -> None:
        """experiment_id defaults to 'unknown' when not provided."""
        prompt = "TASK: Nothing.\n"
        audit = check_dependencies(prompt, str(tmp_path))
        assert audit.experiment_id == "unknown"

    def test_experiment_id_passed_through(self, tmp_path: Path) -> None:
        """experiment_id is passed through when provided."""
        prompt = "TASK: Nothing.\n"
        audit = check_dependencies(prompt, str(tmp_path), experiment_id="exp327")
        assert audit.experiment_id == "exp327"

    def test_multiple_missing_files(self, tmp_path: Path) -> None:
        """All missing files appear in missing_files list."""
        prompt = textwrap.dedent("""\
            EXISTING CODE TO READ FIRST:
            - results/experiment_100.json — missing
            - results/experiment_200.json — also missing

            TASK: Go.
        """)
        audit = check_dependencies(prompt, str(tmp_path))
        assert audit.all_present is False
        assert len(audit.missing_files) == 2


# ---------------------------------------------------------------------------
# build_blocked_artifact
# ---------------------------------------------------------------------------


class TestBuildBlockedArtifact:
    """REQ-INFRA-005: build_blocked_artifact returns correct structure."""

    def test_includes_missing_files(self) -> None:
        """SCENARIO-INFRA-008: blocked artifact includes missing_files list."""
        audit = DependencyAudit(
            experiment_id="exp327",
            required_files=["scripts/a.py", "results/b.json"],
            missing_files=["results/b.json"],
            all_present=False,
        )
        artifact = build_blocked_artifact(audit)
        assert "missing_files" in artifact
        assert artifact["missing_files"] == ["results/b.json"]

    def test_includes_next_action(self) -> None:
        """SCENARIO-INFRA-008: blocked artifact includes next_action field."""
        audit = DependencyAudit(
            experiment_id="exp327",
            required_files=["results/b.json"],
            missing_files=["results/b.json"],
            all_present=False,
        )
        artifact = build_blocked_artifact(audit)
        assert "next_action" in artifact
        assert isinstance(artifact["next_action"], str)
        assert len(artifact["next_action"]) > 0

    def test_includes_status_blocked(self) -> None:
        """Blocked artifact status field is 'blocked'."""
        audit = DependencyAudit(
            experiment_id="exp327",
            required_files=["results/b.json"],
            missing_files=["results/b.json"],
            all_present=False,
        )
        artifact = build_blocked_artifact(audit)
        assert artifact.get("status") == "blocked"

    def test_includes_experiment_id(self) -> None:
        """Blocked artifact includes the experiment_id."""
        audit = DependencyAudit(
            experiment_id="exp327",
            required_files=["results/b.json"],
            missing_files=["results/b.json"],
            all_present=False,
        )
        artifact = build_blocked_artifact(audit)
        assert artifact.get("experiment_id") == "exp327"

    def test_includes_required_files(self) -> None:
        """Blocked artifact includes all required_files for traceability."""
        audit = DependencyAudit(
            experiment_id="exp327",
            required_files=["scripts/a.py", "results/b.json"],
            missing_files=["results/b.json"],
            all_present=False,
        )
        artifact = build_blocked_artifact(audit)
        assert "required_files" in artifact
        assert artifact["required_files"] == ["scripts/a.py", "results/b.json"]


# ---------------------------------------------------------------------------
# load_experiment_prompt
# ---------------------------------------------------------------------------


class TestLoadExperimentPrompt:
    """REQ-INFRA-005: load_experiment_prompt reads a prompt from a roadmap YAML."""

    def test_loads_prompt_by_exp_id(self, tmp_path: Path) -> None:
        """Finds the correct task by matching exp_id in the task id field."""
        roadmap = {
            "tasks": [
                {
                    "id": "exp327-dependency-audit",
                    "title": "Exp 327: Dependency audit",
                    "prompt": "TASK: Do the audit.\n",
                },
                {
                    "id": "exp328-other",
                    "title": "Exp 328: Other",
                    "prompt": "TASK: Do something else.\n",
                },
            ]
        }
        yaml_path = tmp_path / "roadmap.yaml"
        yaml_path.write_text(yaml.dump(roadmap))

        prompt = load_experiment_prompt(str(yaml_path), "327")
        assert "Do the audit" in prompt

    def test_loads_second_task(self, tmp_path: Path) -> None:
        """Correctly finds the second task when exp_id matches second entry."""
        roadmap = {
            "tasks": [
                {"id": "exp100-first", "prompt": "TASK: First.\n"},
                {"id": "exp328-second", "prompt": "TASK: Second.\n"},
            ]
        }
        yaml_path = tmp_path / "roadmap.yaml"
        yaml_path.write_text(yaml.dump(roadmap))

        prompt = load_experiment_prompt(str(yaml_path), "328")
        assert "Second" in prompt

    def test_raises_when_not_found(self, tmp_path: Path) -> None:
        """Raises ValueError when no task matches the exp_id."""
        roadmap = {"tasks": [{"id": "exp100-only", "prompt": "TASK: Only.\n"}]}
        yaml_path = tmp_path / "roadmap.yaml"
        yaml_path.write_text(yaml.dump(roadmap))

        with pytest.raises(ValueError, match="999"):
            load_experiment_prompt(str(yaml_path), "999")

    def test_reads_from_milestones_key(self, tmp_path: Path) -> None:
        """Also handles YAML with a top-level 'milestones' key wrapping tasks."""
        roadmap = {
            "milestones": [
                {
                    "tasks": [
                        {"id": "exp327-audit", "prompt": "TASK: Audit.\n"},
                    ]
                }
            ]
        }
        yaml_path = tmp_path / "roadmap.yaml"
        yaml_path.write_text(yaml.dump(roadmap))

        prompt = load_experiment_prompt(str(yaml_path), "327")
        assert "Audit" in prompt


# ---------------------------------------------------------------------------
# CLI integration
# ---------------------------------------------------------------------------


class TestCLI:
    """REQ-INFRA-005: CLI exits 0 on all_present, 1 on missing files."""

    _script = str(
        Path(__file__).resolve().parents[2] / "scripts" / "experiment_dependency_audit.py"
    )

    def test_exit_0_when_all_present(self, tmp_path: Path) -> None:
        """CLI exits 0 when every required file exists."""
        # Create a real file
        (tmp_path / "scripts").mkdir()
        (tmp_path / "scripts" / "ok.py").write_text("# ok")

        prompt = textwrap.dedent("""\
            EXISTING CODE TO READ FIRST:
            - scripts/ok.py — exists

            TASK: Go.
        """)
        prompt_file = tmp_path / "prompt.txt"
        prompt_file.write_text(prompt)

        result = subprocess.run(
            [
                sys.executable,
                self._script,
                "--exp-id", "327",
                "--prompt-file", str(prompt_file),
                "--project-root", str(tmp_path),
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "All" in result.stdout

    def test_exit_1_when_missing(self, tmp_path: Path) -> None:
        """CLI exits 1 when any required file is missing."""
        prompt = textwrap.dedent("""\
            EXISTING CODE TO READ FIRST:
            - results/experiment_999.json — missing

            TASK: Go.
        """)
        prompt_file = tmp_path / "prompt.txt"
        prompt_file.write_text(prompt)

        result = subprocess.run(
            [
                sys.executable,
                self._script,
                "--exp-id", "327",
                "--prompt-file", str(prompt_file),
                "--project-root", str(tmp_path),
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 1
        assert "experiment_999.json" in result.stdout

    def test_exit_0_with_no_section(self, tmp_path: Path) -> None:
        """CLI exits 0 when the prompt has no EXISTING CODE TO READ FIRST section."""
        prompt_file = tmp_path / "prompt.txt"
        prompt_file.write_text("TASK: Nothing needed.\n")

        result = subprocess.run(
            [
                sys.executable,
                self._script,
                "--exp-id", "327",
                "--prompt-file", str(prompt_file),
                "--project-root", str(tmp_path),
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "All" in result.stdout

    def test_yaml_path_loading(self, tmp_path: Path) -> None:
        """CLI --yaml-path + --exp-id loads the prompt from YAML and exits correctly."""
        (tmp_path / "scripts").mkdir()
        (tmp_path / "scripts" / "real.py").write_text("# real")

        roadmap = {
            "tasks": [
                {
                    "id": "exp327-test",
                    "prompt": (
                        "EXISTING CODE TO READ FIRST:\n"
                        "- scripts/real.py — exists\n\n"
                        "TASK: Go.\n"
                    ),
                }
            ]
        }
        yaml_path = tmp_path / "roadmap.yaml"
        yaml_path.write_text(yaml.dump(roadmap))

        result = subprocess.run(
            [
                sys.executable,
                self._script,
                "--exp-id", "327",
                "--yaml-path", str(yaml_path),
                "--project-root", str(tmp_path),
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stdout + result.stderr
