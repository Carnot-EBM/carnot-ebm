#!/usr/bin/env python3
"""Audit generated pytest files for local imports that cannot exist yet.

Spec: REQ-HARNESS-014, SCENARIO-HARNESS-009
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import sys
import tomllib
from dataclasses import dataclass, field
from importlib.machinery import EXTENSION_SUFFIXES
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_ROADMAP = PROJECT_ROOT / "research-roadmap.yaml"
DEFAULT_TEST_DIR = PROJECT_ROOT / "tests" / "python"
_PY_PATH_RE = re.compile(r"(?:(?:python/)?carnot|scripts)/[A-Za-z0-9_./-]+\.py")


@dataclass(frozen=True)
class LocalImport:
    """One local import target found by parsing a test file with AST."""

    test_path: Path
    module: str
    line: int


@dataclass
class OrphanImportAuditResult:
    """Structured output for the planner/test orphan-import guard."""

    roadmaps_audited: int
    tests_audited: int
    import_targets_checked: int
    orphan_imports_detected: int = 0
    declared_deliverable_imports_allowed: int = 0
    failure_details: list[str] = field(default_factory=list)

    @property
    def orphan_test_guard_ready(self) -> bool:
        return self.orphan_imports_detected == 0

    @property
    def honest_verdict(self) -> str:
        if self.orphan_test_guard_ready:
            return "passed_orphan_test_import_guard"
        return "orphan_test_imports_detected"

    def to_artifact(self) -> dict[str, Any]:
        return {
            "roadmaps_audited": self.roadmaps_audited,
            "tests_audited": self.tests_audited,
            "import_targets_checked": self.import_targets_checked,
            "orphan_imports_detected": self.orphan_imports_detected,
            "declared_deliverable_imports_allowed": self.declared_deliverable_imports_allowed,
            "orphan_test_guard_ready": self.orphan_test_guard_ready,
            "failure_details": list(self.failure_details),
            "honest_verdict": self.honest_verdict,
        }


def _discover_local_roots(project_root: Path) -> set[str]:
    roots = {
        path.name
        for path in (project_root / "python").iterdir()
        if path.is_dir() and (path / "__init__.py").exists()
    }
    if (project_root / "scripts").is_dir():
        roots.add("scripts")
    return roots


def _module_from_python_path(path_text: str) -> str | None:
    path_text = path_text.strip().lstrip("./")
    if path_text.startswith("python/"):
        path_text = path_text.removeprefix("python/")
    if not path_text.endswith(".py"):
        return None
    return path_text[:-3].replace("/", ".")


def _module_exists(project_root: Path, module: str) -> bool:
    relative = Path(*module.split("."))
    for base in (project_root / "python", project_root):
        for suffix in (".py", *EXTENSION_SUFFIXES):
            if (base / relative).with_suffix(suffix).exists():
                return True
        for compiled_path in (base / relative.parent).glob(f"{relative.name}*.so"):
            if compiled_path.is_file():
                return True
        if (base / relative).is_dir():
            return True
    return False


def _iter_string_values(value: Any):
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from _iter_string_values(item)
    elif isinstance(value, list):
        for item in value:
            yield from _iter_string_values(item)


def _declared_modules_from_roadmaps(roadmap_paths: list[Path]) -> set[str]:
    declared: set[str] = set()
    for roadmap_path in roadmap_paths:
        data = yaml.safe_load(roadmap_path.read_text(encoding="utf-8")) or {}
        for text in _iter_string_values(data):
            for path_text in _PY_PATH_RE.findall(text):
                module = _module_from_python_path(path_text)
                if module is not None:
                    declared.add(module)
    return declared


def _configured_ignored_test_names(project_root: Path) -> set[str]:
    pyproject = project_root / "pyproject.toml"
    if not pyproject.exists():
        return set()
    data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
    addopts = (
        data.get("tool", {})
        .get("pytest", {})
        .get("ini_options", {})
        .get("addopts", [])
    )
    ignored: set[str] = set()
    for option in addopts:
        if isinstance(option, str) and option.startswith("--ignore="):
            ignored.add(Path(option.split("=", 1)[1]).name)
    return ignored


def _default_test_paths(project_root: Path) -> list[Path]:
    ignored_names = _configured_ignored_test_names(project_root)
    return [
        path
        for path in sorted((project_root / "tests" / "python").rglob("test_*.py"))
        if "quarantine" not in path.parts and path.name not in ignored_names
    ]


def _extract_local_imports(test_path: Path, local_roots: set[str]) -> list[LocalImport]:
    tree = ast.parse(test_path.read_text(encoding="utf-8"), filename=str(test_path))
    imports: list[LocalImport] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend(
                LocalImport(test_path=test_path, module=alias.name, line=node.lineno)
                for alias in node.names
                if alias.name.split(".", 1)[0] in local_roots
            )
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            if node.module.split(".", 1)[0] in local_roots:
                imports.append(LocalImport(test_path=test_path, module=node.module, line=node.lineno))
    return imports


def _display_path(project_root: Path, path: Path) -> str:
    try:
        return str(path.relative_to(project_root))
    except ValueError:
        return str(path)


def audit_generated_tests(
    *,
    project_root: Path = PROJECT_ROOT,
    roadmap_paths: list[Path] | None = None,
    test_paths: list[Path] | None = None,
) -> OrphanImportAuditResult:
    project_root = project_root.resolve()
    local_roots = _discover_local_roots(project_root)
    roadmaps = [path.resolve() for path in (roadmap_paths or [DEFAULT_ROADMAP])]
    tests = [path.resolve() for path in (test_paths or _default_test_paths(project_root))]
    declared_modules = _declared_modules_from_roadmaps(roadmaps)
    local_imports = [
        local_import
        for test_path in tests
        for local_import in _extract_local_imports(test_path, local_roots)
    ]
    result = OrphanImportAuditResult(
        roadmaps_audited=len(roadmaps),
        tests_audited=len(tests),
        import_targets_checked=len(local_imports),
    )
    for local_import in local_imports:
        if _module_exists(project_root, local_import.module):
            continue
        if local_import.module in declared_modules:
            result.declared_deliverable_imports_allowed += 1
            continue
        result.orphan_imports_detected += 1
        result.failure_details.append(
            f"{_display_path(project_root, local_import.test_path)}:{local_import.line}: "
            f"orphan local import {local_import.module}"
        )
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Audit generated pytest local imports.")
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--roadmap", type=Path, action="append", dest="roadmaps")
    parser.add_argument("--test-file", type=Path, action="append", dest="test_files")
    args = parser.parse_args(argv)
    result = audit_generated_tests(
        project_root=args.project_root,
        roadmap_paths=args.roadmaps or [args.project_root / "research-roadmap.yaml"],
        test_paths=args.test_files,
    )
    print(json.dumps(result.to_artifact(), indent=2, sort_keys=True))
    return 0 if result.orphan_test_guard_ready else 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
