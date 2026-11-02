#!/usr/bin/env python3
"""Lint ARC solve/scoring artifacts for substrate and verdict discipline.

Spec refs: REQ-VERIFY-4437, SCENARIO-VERIFY-4437.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from carnot.agentic.arc_solve_artifact_discipline import (
    ArtifactDisciplineIssue,
    LIVE_LLM_SUBSTRATE,
    validate_arc_solve_artifact,
)


PATH_MARKERS = ("config_rule", "solve", "scoring", "world_model")
METADATA_KEYS = ("experiment", "schema", "artifact_kind", "task_kind", "result_path", "tags")
ARC_SOLVE_EVIDENCE_KEYS = (
    "grounded_win_condition",
    "object_centric_digest",
    "offline_reproduced",
    "qwen_generation",
    "reproduced_levels",
    "reproduction_result",
    "solver",
)


@dataclass(frozen=True)
class ArcArtifactLintIssue:
    """One lint failure for one artifact path."""

    path: Path
    kind: str
    detail: str
    severity: str = "error"

    def to_dict(self) -> dict[str, str]:
        return {
            "path": str(self.path),
            "kind": self.kind,
            "severity": self.severity,
            "detail": self.detail,
        }


def discover_candidate_artifacts(results_dir: Path | str = "results") -> list[Path]:
    """Return result artifacts whose path or metadata marks ARC solve/scoring work."""

    root = Path(results_dir)
    if not root.exists():
        return []
    candidates: list[Path] = []
    for path in sorted(root.rglob("experiment_*.json")):
        payload = _read_json_mapping(path)
        if _path_marks_candidate(path) or _metadata_marks_candidate(payload):
            candidates.append(path)
    return candidates


def lint_results_dir(
    results_dir: Path | str = "results",
    *,
    allow_live_artifacts: Iterable[Path | str] = (),
) -> list[ArcArtifactLintIssue]:
    """Lint every discovered candidate artifact below a results directory."""

    return lint_paths(
        discover_candidate_artifacts(results_dir),
        allow_live_artifacts=allow_live_artifacts,
    )


def lint_paths(
    paths: Iterable[Path | str],
    *,
    allow_live_artifacts: Iterable[Path | str] = (),
) -> list[ArcArtifactLintIssue]:
    """Lint explicit artifact paths."""

    allow_live = {_normalize_path(path) for path in allow_live_artifacts}
    issues: list[ArcArtifactLintIssue] = []
    for raw_path in paths:
        path = Path(raw_path)
        payload = _read_json_mapping(path)
        if not (_path_marks_candidate(path) or _metadata_marks_candidate(payload)):
            continue
        path_allows_live = _normalize_path(path) in allow_live
        issues.extend(lint_artifact(path, payload, allow_live=path_allows_live))
    return issues


def lint_artifact(
    path: Path | str,
    payload: Mapping[str, Any],
    *,
    allow_live: bool = False,
) -> list[ArcArtifactLintIssue]:
    """Lint one already-loaded artifact mapping."""

    artifact_path = Path(path)
    issues: list[ArcArtifactLintIssue] = []
    for issue in validate_arc_solve_artifact(payload, allow_live=allow_live):
        issues.append(_lint_issue(artifact_path, issue, payload))
    return issues


def main(argv: list[str] | None = None) -> int:
    """Run the lint CLI."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="*", help="Optional explicit artifact paths.")
    parser.add_argument("--results-dir", default="results", help="Results directory to scan.")
    parser.add_argument(
        "--allow-live",
        action="append",
        default=[],
        help="Artifact path allowed to declare live_llm_inference.",
    )
    parser.add_argument(
        "--allow-live-file",
        action="append",
        default=[],
        help="File containing live_llm_inference artifact paths, one per line.",
    )
    parser.add_argument("--json", action="store_true", help="Emit a JSON report.")
    args = parser.parse_args(argv)

    allow_live = [*args.allow_live, *_read_allow_live_files(args.allow_live_file)]
    if args.paths:
        issues = lint_paths(args.paths, allow_live_artifacts=allow_live)
    else:
        issues = lint_results_dir(args.results_dir, allow_live_artifacts=allow_live)

    report = {"ok": not issues, "issue_count": len(issues), "issues": [i.to_dict() for i in issues]}
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:  # pragma: no cover - JSON mode is the conductor-facing path.
        for issue in issues:
            print(f"{issue.path}: {issue.kind}: {issue.detail}")
    return 1 if issues else 0


def _lint_issue(
    path: Path,
    issue: ArtifactDisciplineIssue,
    payload: Mapping[str, Any],
) -> ArcArtifactLintIssue:
    kind = issue.kind
    detail = issue.detail
    if kind == "NON_TERMINAL_HONEST_VERDICT" and _has_partial_verdict(payload):
        kind = "NON_TERMINAL_PARTIAL_VERDICT"
        detail = "partial verdicts are non-terminal for ARC solve/scoring artifacts."
    if (
        kind == "LIVE_LLM_NOT_ALLOWLISTED"
        and payload.get("inference_substrate") == LIVE_LLM_SUBSTRATE
    ):
        detail = "live_llm_inference requires an explicit --allow-live artifact path."
    return ArcArtifactLintIssue(path=path, kind=kind, detail=detail)


def _path_marks_candidate(path: Path) -> bool:
    text = path.as_posix().lower()
    tokens = _tokens(text)
    return any(marker in text for marker in PATH_MARKERS) or _has_arc_token(tokens)


def _metadata_marks_candidate(payload: Mapping[str, Any]) -> bool:
    for key in METADATA_KEYS:
        value = payload.get(key)
        if _value_marks_candidate(value):
            return True
    return _looks_like_arc_solve_payload(payload)


def _looks_like_arc_solve_payload(payload: Mapping[str, Any]) -> bool:
    target_game = payload.get("target_game")
    return isinstance(target_game, str) and any(key in payload for key in ARC_SOLVE_EVIDENCE_KEYS)


def _value_marks_candidate(value: Any) -> bool:
    if isinstance(value, str):
        text = value.lower()
        return any(marker in text for marker in PATH_MARKERS) or _has_arc_token(_tokens(text))
    if isinstance(value, Iterable) and not isinstance(value, (bytes, bytearray, Mapping, str)):
        return any(_value_marks_candidate(item) for item in value)
    return False


def _has_partial_verdict(payload: Mapping[str, Any]) -> bool:
    for key in ("honest_verdict", "verdict"):
        value = payload.get(key)
        if isinstance(value, str) and value.lower().startswith(("partial:", "partial_")):
            return True
    return False


def _has_arc_token(tokens: Iterable[str]) -> bool:
    return any(
        token == "arc" or (token.startswith("arc") and not token.startswith("archive"))
        for token in tokens
    )


def _tokens(text: str) -> list[str]:
    return [token for token in re.split(r"[^a-z0-9]+|_", text.lower()) if token]


def _read_json_mapping(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):  # pragma: no cover - corrupt files fail elsewhere.
        return {}
    return payload if isinstance(payload, dict) else {}


def _read_allow_live_files(paths: Iterable[str]) -> list[str]:
    allow_live: list[str] = []
    for raw_path in paths:
        path = Path(raw_path)
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except OSError:
            continue
        for line in lines:
            stripped = line.strip()
            if stripped and not stripped.startswith("#"):
                allow_live.append(stripped)
    return allow_live


def _normalize_path(path: Path | str) -> str:
    return str(Path(path).expanduser().resolve())


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main(sys.argv[1:]))
