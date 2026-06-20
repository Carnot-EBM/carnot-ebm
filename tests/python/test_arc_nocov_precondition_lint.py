"""Tests for ARC roadmap precondition no-cov linting.

Spec refs: REQ-REPORT-4482, SCENARIO-REPORT-4482-ROADMAP-LINT,
SCENARIO-REPORT-4482-ACTIVATION-GUARD.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import yaml

from scripts import arc_nocov_precondition_lint as lint
from scripts import research_conductor as conductor


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "research-reporting" / "spec.md"
PRE_COMMIT_PATH = REPO / ".pre-commit-config.yaml"


def _write_roadmap(path: Path, tasks: list[dict[str, object]]) -> Path:
    path.write_text(
        yaml.safe_dump(
            {"milestone": "2026.06.999", "tasks": tasks},
            sort_keys=False,
            width=120,
        ),
        encoding="utf-8",
    )
    return path


def _task(
    task_id: str,
    *,
    track: str = "arc-north-star",
    prompt: str,
) -> dict[str, object]:
    return {
        "id": task_id,
        "title": task_id,
        "track": track,
        "deliverable": f"results/{task_id}.json",
        "prompt": prompt,
    }


def test_req_report_4482_spec_declares_roadmap_guard_contract() -> None:
    """REQ-REPORT-4482: OpenSpec names the roadmap lint and activation guard."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-REPORT-4482" in spec
    assert "SCENARIO-REPORT-4482-ROADMAP-LINT" in spec
    assert "SCENARIO-REPORT-4482-ACTIVATION-GUARD" in spec
    assert "scripts/arc_nocov_precondition_lint.py" in spec
    assert "activation_guard_wired" in spec


def test_scenario_report_4482_flags_arc_precondition_pytest_without_no_cov(
    tmp_path: Path,
) -> None:
    """SCENARIO-REPORT-4482-ROADMAP-LINT: ARC pytest preconditions require --no-cov."""

    roadmap = _write_roadmap(
        tmp_path / "roadmap.yaml",
        [
            _task(
                "exp9999-arc-bad",
                prompt=(
                    "CONCRETE STEPS:\n"
                    "  0. PRECONDITIONS (BEFORE any other step):\n"
                    '     a. `.venv/bin/pytest -k "arc_solver_kit" -q`\n'
                    "  1. Implement the solve.\n"
                ),
            )
        ],
    )

    issues = lint.lint_roadmap(roadmap)

    assert [issue.kind for issue in issues] == ["PYTEST_PRECONDITION_MISSING_NO_COV"]
    assert issues[0].task_id == "exp9999-arc-bad"
    assert issues[0].track == "arc-north-star"
    assert "pytest" in issues[0].command
    assert "--no-cov" in issues[0].detail


def test_scenario_report_4482_accepts_no_cov_and_ignores_non_arc_or_non_preconditions(
    tmp_path: Path,
) -> None:
    """SCENARIO-REPORT-4482-ROADMAP-LINT: only ARC PRECONDITIONS pytest smoke commands gate."""

    roadmap = _write_roadmap(
        tmp_path / "roadmap.yaml",
        [
            _task(
                "exp9998-arc-good",
                prompt=(
                    "CONCRETE STEPS:\n"
                    "  0. PRECONDITIONS (BEFORE any other step):\n"
                    '     a. `.venv/bin/pytest -k "arc_solver_kit" -q --no-cov`\n'
                ),
            ),
            _task(
                "exp9997-infra-not-arc",
                track="infra",
                prompt=(
                    "CONCRETE STEPS:\n"
                    "  0. PRECONDITIONS (BEFORE any other step):\n"
                    '     a. `.venv/bin/pytest -k "arc_solver_kit" -q`\n'
                ),
            ),
            _task(
                "exp9996-arc-not-precondition",
                prompt=(
                    "IMPLEMENTATION NOTES:\n"
                    '  Developers may run `.venv/bin/pytest -k "arc_solver_kit" -q` manually.\n'
                ),
            ),
        ],
    )

    assert lint.lint_roadmap(roadmap) == []


def test_req_report_4482_cli_emits_json_report(tmp_path: Path, capsys) -> None:
    """REQ-REPORT-4482: CLI returns machine-readable failures for activation tooling."""

    roadmap = _write_roadmap(
        tmp_path / "roadmap.yaml",
        [
            _task(
                "exp9995-arc-bad",
                prompt=(
                    "0. PRECONDITIONS:\n"
                    "   - python -m pytest tests/python/test_arc_solver_kit.py -q\n"
                ),
            )
        ],
    )

    exit_code = lint.main(["--json", str(roadmap)])
    report = json.loads(capsys.readouterr().out)

    assert exit_code == 1
    assert report["ok"] is False
    assert report["issue_count"] == 1
    assert report["issues"][0]["kind"] == "PYTEST_PRECONDITION_MISSING_NO_COV"


def test_req_report_4482_parser_defensive_edges() -> None:
    """REQ-REPORT-4482: malformed task and command fragments do not crash the lint."""

    roadmap = yaml.safe_dump({"milestone": "2026.06.999", "tasks": ["not-a-task"]})

    assert lint.lint_roadmap_text(Path("roadmap.yaml"), roadmap) == []
    assert lint._command_tokens('pytest "unterminated') == ["pytest", '"unterminated']
    assert lint._is_pytest_command([]) is False


def test_req_report_4482_precommit_hook_runs_on_roadmaps() -> None:
    """REQ-REPORT-4482: pre-commit runs the roadmap lint on active and next roadmaps."""

    config = PRE_COMMIT_PATH.read_text(encoding="utf-8")

    assert "- id: arc-nocov-precondition-lint" in config
    hook_block = config.split("- id: arc-nocov-precondition-lint", maxsplit=1)[1].split(
        "\n      - id:",
        maxsplit=1,
    )[0]
    files_match = re.search(r"files: '([^']+)'", hook_block)
    assert "scripts/arc_nocov_precondition_lint.py" in hook_block
    assert files_match is not None
    files_re = re.compile(files_match.group(1))
    assert files_re.search("research-roadmap.yaml")
    assert files_re.search("research-roadmap-next.yaml")
    assert not files_re.search("ops/arc_solve_registry.yaml")


def test_scenario_report_4482_activation_guard_blocks_bad_next_roadmap(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """SCENARIO-REPORT-4482-ACTIVATION-GUARD: bad next roadmap blocks before copy."""

    roadmap = _write_roadmap(
        tmp_path / "research-roadmap-next.yaml",
        [
            _task(
                "exp9994-arc-bad",
                prompt=('0. PRECONDITIONS:\n   - `.venv/bin/pytest -k "first_contact" -q`\n'),
            )
        ],
    )
    logged: list[tuple[str, str, str]] = []
    monkeypatch.setattr(conductor, "log_step", lambda *args: logged.append(args))

    assert conductor._arc_nocov_precondition_activation_guard(roadmap, "2026.06.999") is False
    assert logged
    assert logged[0][1] == "BLOCK"
    assert "exp9994-arc-bad" in logged[0][2]


def test_scenario_report_4482_activation_guard_blocks_unreadable_next_roadmap(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """SCENARIO-REPORT-4482-ACTIVATION-GUARD: unreadable next roadmap fails closed."""

    logged: list[tuple[str, str, str]] = []
    monkeypatch.setattr(conductor, "log_step", lambda *args: logged.append(args))

    assert (
        conductor._arc_nocov_precondition_activation_guard(
            tmp_path / "missing-roadmap.yaml",
            "2026.06.999",
        )
        is False
    )
    assert logged
    assert logged[0][1] == "BLOCK"
    assert "failed while reading" in logged[0][2]


def test_scenario_report_4482_activation_guard_accepts_clean_next_roadmap(
    tmp_path: Path,
) -> None:
    """SCENARIO-REPORT-4482-ACTIVATION-GUARD: clean next roadmap may activate."""

    roadmap = _write_roadmap(
        tmp_path / "research-roadmap-next.yaml",
        [
            _task(
                "exp9993-arc-good",
                prompt=(
                    '0. PRECONDITIONS:\n   - `.venv/bin/pytest -k "first_contact" -q --no-cov`\n'
                ),
            )
        ],
    )

    assert conductor._arc_nocov_precondition_activation_guard(roadmap, "2026.06.999") is True


def test_scenario_report_4482_activation_wrapper_blocks_before_copy(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """SCENARIO-REPORT-4482-ACTIVATION-GUARD: wrapped activation stops before copy."""

    roadmap = _write_roadmap(
        tmp_path / "research-roadmap-next.yaml",
        [
            _task(
                "exp9992-arc-bad",
                prompt=('0. PRECONDITIONS:\n   - `.venv/bin/pytest -k "first_contact" -q`\n'),
            )
        ],
    )
    called = False
    logged: list[tuple[str, str, str]] = []

    def original_activate(*, push: bool = True) -> bool:
        nonlocal called
        called = True
        return push

    monkeypatch.setattr(conductor, "NEXT_ROADMAP_FILE", roadmap)
    monkeypatch.setattr(conductor, "_activate_next_roadmap", original_activate)
    monkeypatch.setattr(conductor, "log_step", lambda *args: logged.append(args))

    lint.install_research_conductor_activation_guard(conductor)

    assert conductor._activate_next_roadmap(push=True) is False
    assert called is False
    assert logged
    assert "exp9992-arc-bad" in logged[0][2]

    clean_roadmap = _write_roadmap(
        tmp_path / "research-roadmap-next-clean.yaml",
        [
            _task(
                "exp9991-arc-good",
                prompt=(
                    '0. PRECONDITIONS:\n   - `.venv/bin/pytest -k "first_contact" -q --no-cov`\n'
                ),
            )
        ],
    )
    called = False
    monkeypatch.setattr(conductor, "NEXT_ROADMAP_FILE", clean_roadmap)
    monkeypatch.setattr(conductor, "_activate_next_roadmap", original_activate)

    lint.install_research_conductor_activation_guard(conductor)

    assert conductor._activate_next_roadmap(push=True) is True
    assert called is True

    list_roadmap = tmp_path / "research-roadmap-next-list.yaml"
    list_roadmap.write_text("- not-a-mapping-root\n", encoding="utf-8")
    called = False
    monkeypatch.setattr(conductor, "NEXT_ROADMAP_FILE", list_roadmap)
    monkeypatch.setattr(conductor, "_activate_next_roadmap", original_activate)

    lint.install_research_conductor_activation_guard(conductor)

    assert conductor._activate_next_roadmap(push=True) is True
    assert called is True
