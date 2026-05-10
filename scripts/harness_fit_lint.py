#!/usr/bin/env python3
"""Harness-fit pre-emit linter for research-roadmap.yaml.

Detects gate definitions whose expected value is unlikely to be emitted
by the agent assigned to the upstream task. This catches the cascade
pattern that drove ~21-40% retire rate across .123-.131:

  Planner emits:  gate: {artifact_field: status, op: "==", value: "success"}
  Gemini agent:   {"status": "complete", ...}
  Conductor:      exact-match fails 3x -> retire downstream task

Per Bustamante "Model-Harness-Fit" (May 2026), each model has trained
artifact-emission conventions. Codex prefers `complete`/`complete_<x>`,
Gemini also tends to `complete`, Claude varies between
`success`/`complete`. An exact-match `==` gate that hardcodes one of
these against an upstream task whose agent prefers a different one is a
predictable failure.

Usage:

    python scripts/harness_fit_lint.py [path/to/research-roadmap.yaml]

Exits non-zero on detected harness-fit risks. Designed for use in the
conductor's pre-emit check (planner output validation) and as a manual
audit tool.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import yaml

# Per-agent artifact-emission conventions. Sources:
# - Bustamante (May 2026) on per-model trained instincts
# - Empirical observation across .123-.131 (gemini emits `complete`)
# - CLAUDE.md "Verdict Terminal-Prefix Discipline" (terminal markers)
AGENT_VERDICT_CONVENTIONS: dict[str, set[str]] = {
    "codex": {"complete", "success", "passed", "shipped"},
    "gemini": {"complete", "success", "passed", "shipped"},
    "claude": {"complete", "success", "passed", "shipped"},
}

# Status field values commonly emitted (the field used by the conductor's
# default exact-match gate). When the planner gates on `status == X`,
# X must be in this set OR be matched by a regex/substring op (not ==).
COMMON_STATUS_VALUES: set[str] = {"success", "complete", "passed", "ok"}


class HarnessFitRisk:
    """A single detected risk in the roadmap YAML."""

    def __init__(
        self,
        downstream_id: str,
        upstream_id: str,
        gate_field: str,
        gate_op: str,
        gate_value: Any,
        agent_type: str,
        reason: str,
    ) -> None:
        self.downstream_id = downstream_id
        self.upstream_id = upstream_id
        self.gate_field = gate_field
        self.gate_op = gate_op
        self.gate_value = gate_value
        self.agent_type = agent_type
        self.reason = reason

    def __str__(self) -> str:  # pragma: no cover
        return (
            f"  {self.downstream_id} -> gates on {self.upstream_id}.{self.gate_field} "
            f"{self.gate_op} {self.gate_value!r} (upstream agent: {self.agent_type})\n"
            f"    {self.reason}"
        )


def _agent_for_task(task: dict[str, Any], default: str) -> str:
    """Resolve the effective agent_type for a task.

    Honors `agent_type:` field; falls back to the milestone-wide default.
    """
    agent = task.get("agent_type")
    if isinstance(agent, str) and agent:
        return agent
    return default


def lint(roadmap_path: Path, default_agent: str = "gemini") -> list[HarnessFitRisk]:
    """Scan the roadmap for harness-fit risks.

    Returns a list of risks. Empty list means no risks detected.
    """
    with open(roadmap_path) as f:
        roadmap = yaml.safe_load(f) or {}

    tasks = roadmap.get("tasks", []) or []
    by_id: dict[str, dict[str, Any]] = {t["id"]: t for t in tasks if "id" in t}

    risks: list[HarnessFitRisk] = []

    for task in tasks:
        task_id = task.get("id", "<unknown>")
        gates = task.get("gated_on") or []
        if not isinstance(gates, list):
            continue

        for gate in gates:
            if not isinstance(gate, dict):
                continue
            upstream_id = gate.get("upstream", "<unknown>")
            field = gate.get("artifact_field", "")
            op = gate.get("op", "==")
            expected = gate.get("value")

            upstream_task = by_id.get(upstream_id)
            if upstream_task is None:
                # Gate references a task not in this roadmap; can't lint.
                continue
            upstream_agent = _agent_for_task(upstream_task, default_agent)

            # Risk class A: exact-match `==` on `status` with a value
            # not in the agent's conventional emission set. The agent
            # might emit a different terminal marker (e.g. plans
            # for "success" but agent emits "complete").
            if op == "==" and field == "status":
                if expected not in COMMON_STATUS_VALUES:
                    continue
                allowed = AGENT_VERDICT_CONVENTIONS.get(upstream_agent, set())
                # If expected is in allowed AND it's "success" or
                # similar, the agent might emit a sibling. Hard-coding
                # exactly one is the risk.
                if len(COMMON_STATUS_VALUES & allowed) > 1:
                    risks.append(
                        HarnessFitRisk(
                            downstream_id=task_id,
                            upstream_id=upstream_id,
                            gate_field=field,
                            gate_op=op,
                            gate_value=expected,
                            agent_type=upstream_agent,
                            reason=(
                                f"Exact-match `status == {expected!r}` is fragile: "
                                f"agent {upstream_agent!r} may emit any of "
                                f"{sorted(COMMON_STATUS_VALUES & allowed)}. "
                                f"Use op: 'in' with a value list, or op: 'truthy'."
                            ),
                        )
                    )

            # Risk class B: exact-match `==` on a custom field with
            # value `True`/`true` against an agent. Agents often emit
            # `"true"` (string) vs `True` (bool); YAML parsers coerce
            # but JSON artifacts may not. Flag it.
            if op == "==" and isinstance(expected, bool) and expected is True:
                risks.append(
                    HarnessFitRisk(
                        downstream_id=task_id,
                        upstream_id=upstream_id,
                        gate_field=field,
                        gate_op=op,
                        gate_value=expected,
                        agent_type=upstream_agent,
                        reason=(
                            f"Exact-match `{field} == True` is fragile: agents "
                            f"may emit string 'true', bool True, or omit the "
                            f"field entirely. Use op: 'truthy' to accept any."
                        ),
                    )
                )

            # Risk class C: gate on a custom flag field where the agent
            # is responsible for setting it explicitly (no fallback).
            # If the gate field is not `status` and uses op `==`, the
            # planner must specify it in the upstream task's prompt.
            # Without an explicit instruction, the agent may not emit it.
            if op == "==" and field != "status" and not field.startswith("status."):
                upstream_prompt = (upstream_task.get("prompt") or "").lower()
                # Heuristic: if the field name doesn't appear in the
                # upstream prompt, the agent has no instruction to set it.
                if field.lower() not in upstream_prompt:
                    risks.append(
                        HarnessFitRisk(
                            downstream_id=task_id,
                            upstream_id=upstream_id,
                            gate_field=field,
                            gate_op=op,
                            gate_value=expected,
                            agent_type=upstream_agent,
                            reason=(
                                f"Gate field {field!r} is not mentioned in "
                                f"the upstream task's prompt. Agent has no "
                                f"instruction to emit it; gate will fail. "
                                f"Either add the field to the upstream "
                                f"prompt's REQUIRED ARTIFACT FIELDS list, "
                                f"or change the gate to op: 'truthy' on a "
                                f"field the agent reliably emits."
                            ),
                        )
                    )

    return risks


def main(argv: list[str]) -> int:  # pragma: no cover
    if len(argv) > 1:
        roadmap_path = Path(argv[1])
    else:
        roadmap_path = Path(__file__).resolve().parent.parent / "research-roadmap.yaml"

    if not roadmap_path.exists():
        print(f"ERROR: roadmap not found: {roadmap_path}", file=sys.stderr)
        return 2

    risks = lint(roadmap_path)
    if not risks:
        print(f"OK: no harness-fit risks in {roadmap_path}")
        return 0

    print(f"FAIL: {len(risks)} harness-fit risk(s) in {roadmap_path}\n")
    for risk in risks:
        print(str(risk))
        print()
    return 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main(sys.argv))
