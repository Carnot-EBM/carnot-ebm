"""Pydantic v2 models for research-roadmap.yaml structural validation.

Why this exists: the conductor bare-indexes task["title"] / task["id"] / task["prompt"]
in three places. A planner that omits any required field causes a KeyError that the
outer try/except swallows, and the conductor spins forever on the same crash. These
models catch that at parse time — required fields are validated; extra fields are
passed through (extra="allow") so planner-invented fields are never rejected.

See: openspec/change-proposals/roadmap-schema-validation.md
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, field_validator, model_validator


GateOp = Literal["==", "!=", ">", "<", ">=", "<=", "exists", "in"]


class GateSpec(BaseModel):
    """A single deliverable-gating condition on a task.

    The conductor's gate-checker reads ``upstream_artifact[artifact_field]``
    and compares against ``value`` with ``op``. Constraining ``op`` to a
    known set catches planner-invented operators (``==>``, ``contains``,
    typos like ``>==``) before the conductor silently fails-open at gate
    time.
    """

    model_config = ConfigDict(extra="allow")

    upstream: str
    artifact_field: str
    op: GateOp
    value: Any


class PriorFailure(BaseModel):
    """Records a prior failed attempt at the same experimental scope.

    Required under CLAUDE.md 'Failed-Experiment Rerun Discipline': a task
    that re-attempts previously-failed scope MUST declare WHY it will succeed
    this time. Without this field the conductor issues a DOOMED_RERUN_BLOCK.
    """

    model_config = ConfigDict(extra="allow")

    experiment_id: str
    verdict: str
    addressed_by: str
    retire_if_same_verdict: bool = False


class ResearchTask(BaseModel):
    """Single research task in a milestone roadmap.

    Required fields map directly to the three bare-lookup crash sites in
    research_conductor.py (load_research_tasks / pick_next_task /
    _archive_current_milestone). Optional fields are common conductor
    extensions; extra fields from the planner are preserved without error.
    """

    model_config = ConfigDict(extra="allow")

    id: str
    milestone: str
    deliverable: str
    title: str
    prompt: str

    # Optional fields with sensible defaults so old roadmaps stay valid
    priority: Literal["critical", "high", "medium"] = "medium"
    requires_gpu: bool = False
    max_turns: int = 50
    estimated_wall_time_min: int = 30
    gated_on: list[GateSpec] = []
    prior_failures: list[PriorFailure] = []

    # Differential agent routing (2026-04-29). Override the default agent model
    # for complex tasks that consistently exhaust Sonnet's max-turns budget
    # (schema validation, hardware integration, ROCm fixes, KV260 work, manifest
    # retirement, multi-step preflight). Pre-emptive Opus routing avoids the
    # wasted Sonnet attempt + the bootstrap-and-bail failure mode where Sonnet
    # writes a defensive `status: running` artifact then exits without updating
    # it — observed in the .80 wedge where exp1028 cascade-blocked exp1030.
    #
    # The field is None by default → falls through to AGENT_MODEL (Sonnet) and
    # the C+E (Sonnet → Opus) escalation pattern handles max-turns failures.
    # Setting `model: opus` skips the Sonnet attempt entirely.
    #
    # Heuristics for when planners should set `model: opus` directly:
    #   - hardware integration (FPGA, ROCm probes, KV260 work)
    #   - schema/preflight infrastructure work
    #   - multi-step coordination (manifest retirement + pretest fix in one task)
    #   - any task whose prompt instructs `CRITICAL: write artifact FIRST`
    #     (these are the bootstrap-and-bail risk class)
    #
    # See:
    #   - scripts/research_conductor.py:2659 (task_model = task.get("model"))
    #   - openspec/change-proposals/differential-agent-routing.md
    model: Literal["sonnet", "opus"] | None = None
    escalate_on_max_turns: bool = True

    @field_validator("deliverable")
    @classmethod
    def deliverable_must_be_json_under_results(cls, v: str) -> str:
        if not v.startswith("results/") or not v.endswith(".json"):
            raise ValueError(
                f"deliverable must start with 'results/' and end with '.json', got: {v!r}"
            )
        return v


class Roadmap(BaseModel):
    """Top-level structure of research-roadmap.yaml.

    Cross-task validator ensures every task's milestone field equals the
    roadmap milestone — catches copy-paste drift that silently routes tasks
    to wrong milestones.
    """

    model_config = ConfigDict(extra="allow")

    milestone: str
    milestone_title: str
    milestone_doc: str
    tasks: list[ResearchTask]

    @model_validator(mode="after")
    def tasks_milestone_must_match_roadmap(self) -> "Roadmap":
        mismatches = [t.id for t in self.tasks if t.milestone != self.milestone]
        if mismatches:
            raise ValueError(
                f"Tasks have mismatched milestone field (expected {self.milestone!r}): "
                + ", ".join(mismatches)
            )
        return self

    @model_validator(mode="after")
    def tasks_must_be_nonempty(self) -> "Roadmap":
        if not self.tasks:
            raise ValueError("Roadmap must contain at least one task")
        return self
