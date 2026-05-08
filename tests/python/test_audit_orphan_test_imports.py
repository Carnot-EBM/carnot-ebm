"""Tests for the planner orphan-test import guard.

Spec: REQ-HARNESS-014, SCENARIO-HARNESS-009
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = REPO_ROOT / "scripts"


def _load_module(name: str, path: Path):
    """Load a script module from disk without requiring scripts/ to be a package."""
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


guard_mod = _load_module("audit_orphan_test_imports", SCRIPTS_DIR / "audit_orphan_test_imports.py")


def _write_yaml(path: Path, payload: dict) -> Path:
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return path


def _project_with_carnot_package(tmp_path: Path) -> Path:
    package_dir = tmp_path / "python" / "carnot" / "reporting"
    package_dir.mkdir(parents=True)
    (tmp_path / "python" / "carnot" / "__init__.py").write_text("", encoding="utf-8")
    (package_dir / "__init__.py").write_text("", encoding="utf-8")
    (tmp_path / "scripts").mkdir()
    (tmp_path / "tests" / "python").mkdir(parents=True)
    return tmp_path


def _roadmap(path: Path, prompt: str = "No implementation deliverables declared.") -> Path:
    return _write_yaml(
        path,
        {
            "milestone": "2026.04.118",
            "milestone_title": "Synthetic orphan import guard",
            "milestone_doc": "openspec/change-proposals/synthetic.md",
            "tasks": [
                {
                    "id": "exp1534-planner-orphan-test-discipline-guard",
                    "milestone": "2026.04.118",
                    "deliverable": "results/experiment_1534_planner_orphan_test_guard.json",
                    "title": "Planner Orphan-Test Discipline Guard",
                    "prompt": prompt,
                }
            ],
        },
    )


def _test_file(project_root: Path, body: str, name: str = "test_generated.py") -> Path:
    path = project_root / "tests" / "python" / name
    path.write_text(body, encoding="utf-8")
    return path


def test_scenario_harness_009_reports_117_orphan_import(tmp_path: Path) -> None:
    """SCENARIO-HARNESS-009: the .117 missing reporting module shape fails."""
    project_root = _project_with_carnot_package(tmp_path)
    roadmap_path = _roadmap(project_root / "research-roadmap.yaml")
    test_path = _test_file(
        project_root,
        "from carnot.reporting.milestone_117_activation_manifest import build_manifest\n",
        name="test_milestone_117_activation_manifest.py",
    )

    artifact = guard_mod.audit_generated_tests(
        project_root=project_root,
        roadmap_paths=[roadmap_path],
        test_paths=[test_path],
    ).to_artifact()

    assert artifact["roadmaps_audited"] == 1
    assert artifact["tests_audited"] == 1
    assert artifact["import_targets_checked"] == 1
    assert artifact["orphan_imports_detected"] == 1
    assert artifact["declared_deliverable_imports_allowed"] == 0
    assert artifact["orphan_test_guard_ready"] is False
    assert "carnot.reporting.milestone_117_activation_manifest" in artifact["failure_details"][0]


def test_req_harness_014_allows_declared_future_deliverable(tmp_path: Path) -> None:
    """REQ-HARNESS-014: a roadmap-declared Python deliverable may be imported."""
    project_root = _project_with_carnot_package(tmp_path)
    roadmap_path = _roadmap(
        project_root / "research-roadmap.yaml",
        prompt=(
            "Implementation deliverables:\n"
            "- python/carnot/reporting/milestone_117_activation_manifest.py\n"
        ),
    )
    test_path = _test_file(
        project_root,
        "from carnot.reporting.milestone_117_activation_manifest import build_manifest\n",
    )

    artifact = guard_mod.audit_generated_tests(
        project_root=project_root,
        roadmap_paths=[roadmap_path],
        test_paths=[test_path],
    ).to_artifact()

    assert artifact["orphan_test_guard_ready"] is True
    assert artifact["orphan_imports_detected"] == 0
    assert artifact["declared_deliverable_imports_allowed"] == 1
    assert artifact["failure_details"] == []


def test_req_harness_014_existing_local_modules_pass(tmp_path: Path) -> None:
    """REQ-HARNESS-014: existing local module imports are counted but not flagged."""
    project_root = _project_with_carnot_package(tmp_path)
    (project_root / "python" / "carnot" / "reporting" / "existing_manifest.py").write_text(
        "def build_manifest():\n    return {}\n",
        encoding="utf-8",
    )
    (
        project_root
        / "python"
        / "carnot"
        / "reporting"
        / f"native_manifest{guard_mod.EXTENSION_SUFFIXES[0]}"
    ).write_text("", encoding="utf-8")
    (
        project_root
        / "python"
        / "carnot"
        / "reporting"
        / "foreign_native_manifest.cpython-999-x86_64-linux-gnu.so"
    ).write_text("", encoding="utf-8")
    roadmap_path = _roadmap(project_root / "research-roadmap.yaml")
    test_path = _test_file(
        project_root,
        "import carnot.reporting.existing_manifest\n"
        "import carnot.reporting.native_manifest\n"
        "import carnot.reporting.foreign_native_manifest\n"
        "from carnot.reporting.existing_manifest import build_manifest\n"
        "from carnot.reporting import existing_manifest\n",
    )

    artifact = guard_mod.audit_generated_tests(
        project_root=project_root,
        roadmap_paths=[roadmap_path],
        test_paths=[test_path],
    ).to_artifact()

    assert artifact["orphan_test_guard_ready"] is True
    assert artifact["import_targets_checked"] == 5
    assert artifact["orphan_imports_detected"] == 0
    assert artifact["honest_verdict"] == "passed_orphan_test_import_guard"
    assert guard_mod._configured_ignored_test_names(project_root) == set()
    assert guard_mod._module_from_python_path("not-a-python-path") is None
    assert guard_mod._display_path(project_root, Path("/outside/project/test.py")) == (
        "/outside/project/test.py"
    )


def test_req_harness_014_default_discovery_skips_configured_ignores(tmp_path: Path) -> None:
    """REQ-HARNESS-014: default discovery follows configured pytest ignores."""
    project_root = _project_with_carnot_package(tmp_path)
    roadmap_path = _roadmap(project_root / "research-roadmap.yaml")
    _test_file(
        project_root,
        "from carnot.reporting.present import marker\n",
        name="test_present.py",
    )
    _test_file(
        project_root,
        "from carnot.reporting.missing_ignored import marker\n",
        name="test_ignored.py",
    )
    (project_root / "python" / "carnot" / "reporting" / "present.py").write_text(
        "marker = True\n",
        encoding="utf-8",
    )
    (project_root / "pyproject.toml").write_text(
        '[tool.pytest.ini_options]\naddopts = ["--ignore=tests/python/test_ignored.py"]\n',
        encoding="utf-8",
    )

    artifact = guard_mod.audit_generated_tests(
        project_root=project_root,
        roadmap_paths=[roadmap_path],
    ).to_artifact()

    assert artifact["tests_audited"] == 1
    assert artifact["import_targets_checked"] == 1
    assert artifact["orphan_test_guard_ready"] is True


def test_req_harness_014_cli_reports_json_and_status(tmp_path: Path, capsys) -> None:
    """REQ-HARNESS-014: CLI emits JSON and exits nonzero when orphans exist."""
    project_root = _project_with_carnot_package(tmp_path)
    roadmap_path = _roadmap(project_root / "research-roadmap.yaml")
    test_path = _test_file(
        project_root,
        "from carnot.reporting.milestone_117_activation_manifest import build_manifest\n",
    )

    exit_code = guard_mod.main(
        [
            "--project-root",
            str(project_root),
            "--roadmap",
            str(roadmap_path),
            "--test-file",
            str(test_path),
        ]
    )
    printed = json.loads(capsys.readouterr().out)

    assert exit_code == 1
    assert printed["orphan_imports_detected"] == 1
    assert printed["orphan_test_guard_ready"] is False
