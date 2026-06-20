#!/usr/bin/env python3
"""Lint ARC focused pytest preconditions for missing ``--no-cov``.

Spec refs: REQ-REPORT-4475, SCENARIO-REPORT-4475-NOCOV-LINT.
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import shlex
import sys
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
REGISTRY_RELATIVE_PATH = Path("ops/arc_solve_registry.yaml")
ALLOW_FULL_SUITE_COVERAGE_MARKER = "arc-precondition-nocov: allow-full-suite-coverage"
ARC_EXPERIMENT_GLOBS = (
    "experiment_*_arc*.py",
    "experiment_*solve*.py",
    "experiment_*first_contact*.py",
)
REGISTRY_SCRIPT_RE = re.compile(r"((?:python/carnot|scripts)/[A-Za-z0-9_./-]+\.py)")
PRECONDITION_CONTEXT_WORDS = ("baseline", "focused", "nocov", "precondition", "smoke")


CommandValue = list[str] | str


@dataclass(frozen=True)
class ArcPreconditionNoCovIssue:
    """One no-cov lint failure for one source location."""

    path: Path
    line: int
    kind: str
    detail: str
    severity: str = "error"

    def to_dict(self) -> dict[str, str | int]:
        return {
            "path": str(self.path),
            "line": self.line,
            "kind": self.kind,
            "severity": self.severity,
            "detail": self.detail,
        }


def discover_candidate_scripts(
    *,
    root: Path | str = REPO_ROOT,
    registry_path: Path | str | None = None,
) -> list[Path]:
    """REQ-REPORT-4475: discover ARC experiment and registry-referenced solvers."""

    repo = Path(root)
    paths: set[Path] = set()
    experiment_root = repo / "python" / "carnot"
    for pattern in ARC_EXPERIMENT_GLOBS:
        paths.update(
            path
            for path in experiment_root.glob(pattern)
            if path.is_file() and _path_marks_arc_experiment(path)
        )
    scripts_root = repo / "scripts"
    paths.update(path for path in scripts_root.glob("arc*.py") if path.is_file())

    registry = Path(registry_path) if registry_path is not None else repo / REGISTRY_RELATIVE_PATH
    try:
        registry_text = registry.read_text(encoding="utf-8")
    except OSError:
        registry_text = ""
    for match in REGISTRY_SCRIPT_RE.finditer(registry_text):
        candidate = repo / match.group(1)
        if candidate.is_file():
            paths.add(candidate)
    return sorted(paths)


def lint_paths(
    paths: Iterable[Path | str],
) -> list[ArcPreconditionNoCovIssue]:
    """Lint explicit paths for focused pytest preconditions without ``--no-cov``."""

    issues: list[ArcPreconditionNoCovIssue] = []
    for raw_path in paths:
        path = Path(raw_path)
        if path.name == REGISTRY_RELATIVE_PATH.name:
            issues.extend(
                lint_paths(
                    discover_candidate_scripts(
                        root=_infer_repo_root(path),
                        registry_path=path,
                    )
                )
            )
            continue
        if path.suffix != ".py":
            continue
        try:
            source = path.read_text(encoding="utf-8")
        except OSError:
            continue
        issues.extend(lint_source(path, source))
    return issues


def lint_source(path: Path | str, source: str) -> list[ArcPreconditionNoCovIssue]:
    """Lint one Python source string."""

    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        return [
            ArcPreconditionNoCovIssue(
                path=Path(path),
                line=exc.lineno or 1,
                kind="PYTHON_PARSE_ERROR",
                detail=f"could not parse Python source: {exc.msg}",
            )
        ]
    visitor = _PytestCommandVisitor(Path(path), source)
    visitor.visit(tree)
    return visitor.issues


def lint_default_repo(
    *,
    root: Path | str = REPO_ROOT,
    registry_path: Path | str | None = None,
) -> list[ArcPreconditionNoCovIssue]:
    """Lint the default ARC script set in the repository."""

    return lint_paths(discover_candidate_scripts(root=root, registry_path=registry_path))


def main(argv: list[str] | None = None) -> int:
    """Run the no-cov lint CLI."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="*", help="Optional script paths to lint.")
    parser.add_argument("--root", default=str(REPO_ROOT), help="Repository root for discovery.")
    parser.add_argument(
        "--registry-path",
        default=None,
        help="ARC solve registry path for registry-referenced solver discovery.",
    )
    parser.add_argument("--json", action="store_true", help="Emit a JSON report.")
    args = parser.parse_args(argv)

    if args.paths:
        issues = lint_paths(args.paths)
    else:
        issues = lint_default_repo(root=args.root, registry_path=args.registry_path)

    report = {"ok": not issues, "issue_count": len(issues), "issues": [i.to_dict() for i in issues]}
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:  # pragma: no cover - JSON mode is the conductor-facing path.
        for issue in issues:
            print(f"{issue.path}:{issue.line}: {issue.kind}: {issue.detail}")
    return 1 if issues else 0


class _PytestCommandVisitor(ast.NodeVisitor):
    def __init__(self, path: Path, source: str) -> None:
        self.path = path
        self.source = source
        self.lines = source.splitlines()
        self.env: dict[str, CommandValue] = {}
        self.bad_command_names: set[str] = set()
        self.issues: list[ArcPreconditionNoCovIssue] = []
        self._issue_keys: set[tuple[int, str]] = set()

    def visit_Assign(self, node: ast.Assign) -> Any:
        value = self._eval_command_value(node.value)
        target_names = [name for target in node.targets for name in self._target_names(target)]
        if value is not None:
            for name in target_names:
                self.env[name] = value
            context = [*target_names, self._line_text(node.lineno)]
            issue = self._issue_for_command(node.lineno, value, context)
            if issue is not None:
                self._add_issue(issue)
                self.bad_command_names.update(target_names)
        self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> Any:
        value = self._eval_command_value(node.value) if node.value is not None else None
        target_names = self._target_names(node.target)
        if value is not None:
            for name in target_names:
                self.env[name] = value
            context = [*target_names, self._line_text(node.lineno)]
            issue = self._issue_for_command(node.lineno, value, context)
            if issue is not None:
                self._add_issue(issue)
                self.bad_command_names.update(target_names)
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> Any:
        command = self._call_command(node)
        if command is not None:
            first_arg = node.args[0] if node.args else None
            if not (isinstance(first_arg, ast.Name) and first_arg.id in self.bad_command_names):
                context = [self._function_name(node.func), self._line_text(node.lineno)]
                issue = self._issue_for_command(node.lineno, command, context)
                if issue is not None:
                    self._add_issue(issue)
        self.generic_visit(node)

    def _call_command(self, node: ast.Call) -> CommandValue | None:
        if not node.args:
            return None
        func_name = self._function_name(node.func)
        if not (func_name == "run" or func_name.endswith(".run")):
            return None
        return self._eval_command_value(node.args[0])

    def _eval_command_value(self, node: ast.AST | None) -> CommandValue | None:
        if node is None:
            return None
        if isinstance(node, ast.Name):
            return self.env.get(node.id)
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            return node.value
        if isinstance(node, (ast.List, ast.Tuple)):
            return [self._atom_text(element) for element in node.elts]
        return None

    def _atom_text(self, node: ast.AST) -> str:
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            return node.value
        segment = ast.get_source_segment(self.source, node)
        return segment or ""

    def _issue_for_command(
        self,
        line: int,
        command: CommandValue,
        context: Sequence[str],
    ) -> ArcPreconditionNoCovIssue | None:
        tokens = _command_tokens(command)
        if not _is_pytest_command(tokens):
            return None
        if _has_no_cov(tokens):
            return None
        if _is_allowed_full_suite_coverage(tokens, self._line_text(line)):
            return None
        if not _is_focused_precondition_command(tokens, context):
            return None
        detail = (
            "focused ARC pytest precondition command uses pytest -k or a test subset "
            "without --no-cov; smoke gates must not inherit global coverage fail-under."
        )
        return ArcPreconditionNoCovIssue(
            path=self.path,
            line=line,
            kind="PYTEST_PRECONDITION_MISSING_NO_COV",
            detail=detail,
        )

    def _add_issue(self, issue: ArcPreconditionNoCovIssue) -> None:
        key = (issue.line, issue.kind)
        if key not in self._issue_keys:
            self._issue_keys.add(key)
            self.issues.append(issue)

    def _line_text(self, line: int) -> str:
        if 1 <= line <= len(self.lines):
            return self.lines[line - 1]
        return ""

    @staticmethod
    def _target_names(node: ast.AST) -> list[str]:
        if isinstance(node, ast.Name):
            return [node.id]
        if isinstance(node, (ast.Tuple, ast.List)):
            names: list[str] = []
            for element in node.elts:
                names.extend(_PytestCommandVisitor._target_names(element))
            return names
        return []

    @staticmethod
    def _function_name(node: ast.AST) -> str:
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Attribute):
            parent = _PytestCommandVisitor._function_name(node.value)
            return f"{parent}.{node.attr}" if parent else node.attr
        return ""


def _command_tokens(command: CommandValue) -> list[str]:
    if isinstance(command, str):
        try:
            return shlex.split(command)
        except ValueError:
            return command.split()
    return [str(token) for token in command]


def _path_marks_arc_experiment(path: Path) -> bool:
    name = path.name
    return "solve" in name or "first_contact" in name or "_arc_" in name or name.endswith("_arc.py")


def _infer_repo_root(path: Path) -> Path:
    resolved = path.resolve()
    for parent in (resolved.parent, *resolved.parents):
        if (parent / "python" / "carnot").is_dir() and (parent / "ops").is_dir():
            return parent
    if resolved.parent.name == "ops":
        return resolved.parent.parent
    return REPO_ROOT


def _is_pytest_command(tokens: Sequence[str]) -> bool:
    if not tokens:
        return False
    first = tokens[0]
    if Path(first).name == "pytest" or "pytest" in first:
        return True
    return len(tokens) >= 3 and Path(tokens[0]).name.startswith("python") and tokens[1:3] == [
        "-m",
        "pytest",
    ]


def _has_no_cov(tokens: Sequence[str]) -> bool:
    return "--no-cov" in tokens


def _has_k_selector(tokens: Sequence[str]) -> bool:
    return "-k" in tokens or any(token.startswith("-k=") for token in tokens)


def _has_cov_gate(tokens: Sequence[str]) -> bool:
    return any(
        token == "--cov"
        or token.startswith("--cov=")
        or token == "--cov-fail-under"
        or token.startswith("--cov-fail-under=")
        for token in tokens
    )


def _is_test_subset(tokens: Sequence[str]) -> bool:
    for token in tokens:
        normalized = token.rstrip("/")
        if normalized.startswith("tests/") and normalized != "tests/python":
            return True
        if normalized.startswith("tests\\") and normalized != "tests\\python":
            return True
    return False


def _is_allowed_full_suite_coverage(tokens: Sequence[str], line_text: str) -> bool:
    if _has_k_selector(tokens) or not _has_cov_gate(tokens):
        return False
    if ALLOW_FULL_SUITE_COVERAGE_MARKER in line_text:
        return True
    return "tests/python" in tokens and not _is_test_subset(tokens)


def _is_focused_precondition_command(tokens: Sequence[str], context: Sequence[str]) -> bool:
    if _has_k_selector(tokens) or _is_test_subset(tokens):
        return True
    context_text = " ".join(context).lower()
    return any(word in context_text for word in PRECONDITION_CONTEXT_WORDS)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main(sys.argv[1:]))
