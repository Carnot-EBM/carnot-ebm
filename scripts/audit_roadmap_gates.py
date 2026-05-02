#!/usr/bin/env python3
"""Audit roadmap gates before the conductor sees a new milestone.

The conductor already blocks doomed reruns and bad gates at dispatch
time. This script runs the cheap structural checks earlier, while the
roadmap is still planner output and can be fixed without losing an
experiment slot.

Spec: REQ-INFRA-075, SCENARIO-INFRA-084, SCENARIO-INFRA-085,
      SCENARIO-INFRA-086
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_COMPLETE_PATH = PROJECT_ROOT / "research-complete.yaml"
DEFAULT_ACTIVE_ROADMAP = PROJECT_ROOT / "research-roadmap.yaml"

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_STOPWORDS = {
    "a",
    "an",
    "and",
    "by",
    "for",
    "from",
    "in",
    "of",
    "on",
    "or",
    "the",
    "to",
    "via",
    "with",
    *{f"v{i}" for i in range(1, 10)},
}


@dataclass
class AuditResult:
    """Structured result matching the Exp 1140 required artifact fields."""

    n_tasks_audited: int
    n_gate_upstream_checks: int = 0
    n_gate_upstream_failures: int = 0
    n_prior_failures_checks: int = 0
    n_prior_failures_missing: int = 0
    n_model_agent_coherence_failures: int = 0
    n_gate_field_cross_ref_failures: int = 0
    failure_details: list[str] = field(default_factory=list)
    audit_script_written: bool = True

    @property
    def roadmap_gate_audit_passed(self) -> bool:
        return (
            self.n_gate_upstream_failures
            + self.n_prior_failures_missing
            + self.n_model_agent_coherence_failures
            + self.n_gate_field_cross_ref_failures
        ) == 0

    @property
    def honest_verdict(self) -> str:
        if self.n_model_agent_coherence_failures:
            return "model_agent_incoherence_found"
        if self.n_gate_upstream_failures or self.n_gate_field_cross_ref_failures:
            return "gate_field_gaps_found"
        if self.n_prior_failures_missing:
            return "prior_failures_gaps_found"
        return "all_checks_pass"

    def to_artifact(self) -> dict[str, Any]:
        return {
            "n_tasks_audited": self.n_tasks_audited,
            "n_gate_upstream_checks": self.n_gate_upstream_checks,
            "n_gate_upstream_failures": self.n_gate_upstream_failures,
            "n_prior_failures_checks": self.n_prior_failures_checks,
            "n_prior_failures_missing": self.n_prior_failures_missing,
            "n_model_agent_coherence_failures": self.n_model_agent_coherence_failures,
            "n_gate_field_cross_ref_failures": self.n_gate_field_cross_ref_failures,
            "roadmap_gate_audit_passed": self.roadmap_gate_audit_passed,
            "failure_details": list(self.failure_details),
            "audit_script_written": self.audit_script_written,
            "honest_verdict": self.honest_verdict,
        }


def scope_keywords(title: str) -> set[str]:
    """Return case-insensitive substantive title tokens for scope matching."""
    return {
        token
        for token in _TOKEN_RE.findall(title.lower())
        if len(token) > 1 and token not in _STOPWORDS
    }


def select_roadmap_path(
    requested_path: Path, active_path: Path = DEFAULT_ACTIVE_ROADMAP
) -> tuple[Path, str]:
    """Resolve the audit target, falling back from missing next-roadmap to active roadmap."""
    if requested_path.exists():
        return requested_path, "requested roadmap path exists"
    if requested_path.name == "research-roadmap-next.yaml" and active_path.exists():
        return active_path, "requested next roadmap missing; audited active research-roadmap.yaml"
    return requested_path, "requested roadmap path is missing"


def _load_yaml_mapping(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"YAML file not found: {path}")
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Top-level YAML value must be a mapping: {path}")
    return data


def _tasks_from_roadmap(data: dict[str, Any]) -> list[dict[str, Any]]:
    tasks = data.get("tasks", []) or []
    return [task for task in tasks if isinstance(task, dict)]


def _completed_task_titles(complete_path: Path) -> list[tuple[str, str, set[str]]]:
    if not complete_path.exists():
        return []
    complete = _load_yaml_mapping(complete_path)
    completed: list[tuple[str, str, set[str]]] = []
    for milestone in complete.get("milestones", []) or []:
        if not isinstance(milestone, dict):
            continue
        for task in milestone.get("tasks", []) or []:
            if not isinstance(task, dict):
                continue
            task_id = str(task.get("id") or "")
            title = str(task.get("title") or "")
            keywords = scope_keywords(title)
            if task_id and title and keywords:
                completed.append((task_id, title, keywords))
    return completed


def _required_artifact_fields_block(prompt: str) -> str:
    lines = prompt.splitlines()
    for index, line in enumerate(lines):
        if "REQUIRED ARTIFACT FIELDS:" not in line.upper():
            continue
        block = [line]
        for following in lines[index + 1 :]:
            stripped = following.strip()
            if not stripped:
                break
            if stripped.endswith(":") and not stripped.startswith(("-", "*")):
                break
            block.append(following)
        return "\n".join(block)
    return ""


def _prior_matches_for_task(
    title: str,
    completed_titles: list[tuple[str, str, set[str]]],
) -> list[str]:
    task_keywords = scope_keywords(title)
    matches = []
    for prior_id, _prior_title, prior_keywords in completed_titles:
        if len(task_keywords & prior_keywords) >= 2:
            matches.append(prior_id)
    return matches


def _has_prior_failures(task: dict[str, Any]) -> bool:
    prior_failures = task.get("prior_failures")
    return isinstance(prior_failures, list) and bool(prior_failures)


def audit_roadmap(roadmap_path: Path, complete_path: Path = DEFAULT_COMPLETE_PATH) -> AuditResult:
    """Run all Exp 1140 roadmap audits and return a structured result."""
    roadmap = _load_yaml_mapping(roadmap_path)
    tasks = _tasks_from_roadmap(roadmap)
    tasks_by_id = {str(task.get("id")): task for task in tasks if task.get("id")}
    completed_titles = _completed_task_titles(complete_path)
    result = AuditResult(n_tasks_audited=len(tasks), n_prior_failures_checks=len(tasks))

    for task in tasks:
        task_id = str(task.get("id") or "<missing-id>")
        agent_type = str(task.get("agent_type") or "").strip().lower()
        model = str(task.get("model") or "").strip()
        if agent_type == "codex" and model != "gpt-5.5":
            result.n_model_agent_coherence_failures += 1
            result.failure_details.append(
                f"MODEL_AGENT_COHERENCE {task_id}: agent_type=codex requires model=gpt-5.5, got {model or '<missing>'}"
            )
        if agent_type == "gemini":
            result.n_model_agent_coherence_failures += 1
            result.failure_details.append(
                f"MODEL_AGENT_COHERENCE {task_id}: agent_type=gemini is not allowed in this roadmap"
            )

        prior_matches = _prior_matches_for_task(str(task.get("title") or ""), completed_titles)
        if prior_matches and not _has_prior_failures(task):
            result.n_prior_failures_missing += 1
            result.failure_details.append(
                f"PRIOR_FAILURES_COVERAGE {task_id}: title matches prior tasks {prior_matches[:5]} but prior_failures is missing or empty"
            )

        gates = task.get("gated_on") or []
        if not isinstance(gates, list):
            continue
        for gate in gates:
            if not isinstance(gate, dict):
                continue
            result.n_gate_upstream_checks += 1
            upstream_id = str(gate.get("upstream") or "")
            artifact_field = str(gate.get("artifact_field") or "")
            upstream_task = tasks_by_id.get(upstream_id)
            if upstream_task is None:
                result.n_gate_upstream_failures += 1
                result.failure_details.append(
                    f"GATE_UPSTREAM_EXISTS {task_id}: gated_on upstream {upstream_id or '<missing>'} is not in roadmap"
                )
                continue
            required_block = _required_artifact_fields_block(str(upstream_task.get("prompt") or ""))
            if artifact_field and artifact_field not in required_block:
                result.n_gate_field_cross_ref_failures += 1
                result.failure_details.append(
                    f"GATE_FIELD_CROSS_REF {task_id}: gate field {artifact_field!r} is absent from upstream {upstream_id} REQUIRED ARTIFACT FIELDS"
                )
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Audit roadmap gate/prior-failure metadata.")
    parser.add_argument("roadmap", type=Path, help="Roadmap YAML path to audit")
    parser.add_argument(
        "--complete",
        type=Path,
        default=DEFAULT_COMPLETE_PATH,
        help="Completed research YAML path",
    )
    args = parser.parse_args(argv)
    roadmap_path, note = select_roadmap_path(args.roadmap)
    try:
        result = audit_roadmap(roadmap_path, complete_path=args.complete)
    except (FileNotFoundError, ValueError, yaml.YAMLError) as exc:
        print(json.dumps({"error": str(exc), "roadmap_path_note": note}, indent=2))
        return 2
    artifact = result.to_artifact()
    artifact["roadmap_path_note"] = note
    artifact["roadmap_path_used"] = str(roadmap_path)
    print(json.dumps(artifact, indent=2, sort_keys=True))
    return 0 if result.roadmap_gate_audit_passed else 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
