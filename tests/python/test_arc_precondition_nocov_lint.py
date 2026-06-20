"""Tests for ARC focused pytest precondition no-cov linting.

Spec refs: REQ-REPORT-4475, SCENARIO-REPORT-4475-NOCOV-LINT.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import yaml

from scripts import arc_precondition_nocov_lint as lint


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "research-reporting" / "spec.md"
PRE_COMMIT_PATH = REPO / ".pre-commit-config.yaml"


def _write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def test_req_report_4475_spec_declares_nocov_smoke_contract() -> None:
    """REQ-REPORT-4475: OpenSpec names the helper, lint, and required fields."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-REPORT-4475" in spec
    assert "SCENARIO-REPORT-4475-SMOKE" in spec
    assert "SCENARIO-REPORT-4475-NOCOV-LINT" in spec
    assert "SCENARIO-REPORT-4475-SC25-COUNT" in spec
    for field in (
        "smoke_helper_shipped",
        "nocov_lint_shipped",
        "catches_cov_gated_precondition",
        "count_integrity_extended",
    ):
        assert field in spec


def test_scenario_report_4475_flags_pytest_k_precondition_without_no_cov() -> None:
    """SCENARIO-REPORT-4475-NOCOV-LINT: exp4455-class focused pytest is rejected."""

    source = '''
import subprocess

def precondition_probe(root):
    pytest_cmd = [
        str(root / ".venv" / "bin" / "pytest"),
        "-k",
        "config_rule or arc_solver_kit",
        "-q",
    ]
    return subprocess.run(pytest_cmd)
'''

    issues = lint.lint_source(Path("python/carnot/experiment_9999_arc_solve.py"), source)

    assert [issue.kind for issue in issues] == ["PYTEST_PRECONDITION_MISSING_NO_COV"]
    assert "-k" in issues[0].detail
    assert "--no-cov" in issues[0].detail


def test_scenario_report_4475_accepts_helper_and_literal_no_cov_commands() -> None:
    """SCENARIO-REPORT-4475-NOCOV-LINT: helper use and explicit no-cov smoke pass."""

    source = '''
import subprocess
from carnot.agentic.arc_precondition_smoke import arc_precondition_smoke

BASELINE_COMMAND_TEXT = '.venv/bin/pytest -k "arc_solver_kit or first_contact" -q --no-cov'

def precondition_probe(root):
    green, summary = arc_precondition_smoke("arc_solver_kit or first_contact", root=root)
    pytest_cmd = [
        str(root / ".venv" / "bin" / "pytest"),
        "-k",
        "arc_solver_kit or first_contact",
        "-q",
        "--no-cov",
    ]
    subprocess.run(pytest_cmd)
    return green, summary
'''

    issues = lint.lint_source(Path("python/carnot/experiment_9998_first_contact.py"), source)

    assert issues == []


def test_req_report_4475_allows_explicit_full_suite_coverage_allowlist() -> None:
    """REQ-REPORT-4475: full-suite coverage commands are separate from smoke gates."""

    source = '''
import subprocess

def coverage_gate(root):
    cmd = [
        str(root / ".venv" / "bin" / "pytest"),
        "tests/python",
        "--cov=python/carnot",
        "--cov-fail-under=100",
    ]  # arc-precondition-nocov: allow-full-suite-coverage
    return subprocess.run(cmd)
'''

    issues = lint.lint_source(Path("python/carnot/experiment_9997_arc_coverage.py"), source)

    assert issues == []


def test_req_report_4475_discovers_registry_referenced_solvers(tmp_path: Path) -> None:
    """REQ-REPORT-4475: lint discovery includes solver paths named by the registry."""

    solver = _write(
        tmp_path / "python" / "carnot" / "experiment_9996_custom_solver.py",
        "def main():\n    return 0\n",
    )
    _write(
        tmp_path / "python" / "carnot" / "experiment_9995_arc_solver.py",
        "def main():\n    return 0\n",
    )
    registry = {
        "games": [
            {
                "game": "zz99",
                "levels_reproduced": 1,
                "solver": "python/carnot/experiment_9996_custom_solver.py --game zz99",
            }
        ]
    }
    registry_path = tmp_path / "ops" / "arc_solve_registry.yaml"
    registry_path.parent.mkdir(parents=True)
    registry_path.write_text(yaml.safe_dump(registry), encoding="utf-8")

    discovered = lint.discover_candidate_scripts(root=tmp_path, registry_path=registry_path)

    assert solver in discovered
    assert tmp_path / "python" / "carnot" / "experiment_9995_arc_solver.py" in discovered


def test_req_report_4475_precommit_hook_runs_nocov_lint() -> None:
    """REQ-REPORT-4475: pre-commit runs the no-cov lint on ARC solver script edits."""

    config = PRE_COMMIT_PATH.read_text(encoding="utf-8")

    assert "- id: arc-precondition-nocov-lint" in config
    hook_block = config.split("- id: arc-precondition-nocov-lint", maxsplit=1)[1].split(
        "\n      - id:",
        maxsplit=1,
    )[0]
    files_match = re.search(r"files: '([^']+)'", hook_block)
    assert "scripts/arc_precondition_nocov_lint.py" in hook_block
    assert files_match is not None
    files_re = re.compile(files_match.group(1))
    assert files_re.search("python/carnot/experiment_4471_first_contact_rotated_new_game.py")
    assert files_re.search("python/carnot/experiment_4467_solve_dc22_cegis_nocov.py")
    assert files_re.search("ops/arc_solve_registry.yaml")
    assert not files_re.search("python/carnot/experiment_4134_archive_v382_activate_v383.py")


def test_req_report_4475_cli_reports_json_issues(tmp_path: Path, capsys) -> None:
    """REQ-REPORT-4475: CLI emits machine-readable no-cov lint failures."""

    script = _write(
        tmp_path / "python" / "carnot" / "experiment_9994_arc_solve.py",
        '''
import subprocess
def precondition_probe(root):
    return subprocess.run([".venv/bin/pytest", "-k", "arc_solver_kit", "-q"])
''',
    )

    exit_code = lint.main(["--json", str(script)])
    report = json.loads(capsys.readouterr().out)

    assert exit_code == 1
    assert report["ok"] is False
    assert report["issue_count"] == 1
    assert report["issues"][0]["kind"] == "PYTEST_PRECONDITION_MISSING_NO_COV"
