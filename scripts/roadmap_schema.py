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
    #
    # NOTE: when agent_type is "codex" or "gemini", `model` is interpreted as
    # the model identifier within that agent's namespace (e.g. "gpt-5.5",
    # "gemini-3.1-pro-preview"). The Literal here intentionally includes only
    # Claude-specific identifiers; for other agent types, model_must_match_agent_type
    # validator below relaxes to accept any str (since we can't enumerate every
    # vendor's catalogue).
    model: Literal["sonnet", "opus"] | str | None = None  # type: ignore[valid-type]
    escalate_on_max_turns: bool = True

    # Multi-agent routing (2026-04-29 evening). Per-task agent backend
    # selection orthogonal to `model` above. The conductor supports four
    # backends — claude (default), codex, gemini, opencode — selected at
    # process startup via the AGENT_TYPE env var. Setting agent_type on a
    # task overrides AGENT_TYPE for that task only, falling back to the
    # process default when None.
    #
    # Routing heuristics for the planner:
    #   - claude (default): synthesis-heavy tasks, retros, planning, hardware
    #     integration, position paper drafting, multi-file coordination
    #   - codex (gpt-5.5): formulaic code generation — WOPR cartridges,
    #     verifier implementations, test scaffolding, PyO3 bindings,
    #     sampler implementations, dataset pipelines
    #   - gemini (Ultra): long-context analysis (1M tokens) — failure-ledger
    #     pattern detection across milestone history, architecture coherence
    #     audits across the full Phase-3 → Phase-7 chain, multi-paper
    #     literature synthesis, multimodal verification (FPGA bitstream /
    #     oscilloscope traces in future)
    #   - opencode: experimental; not currently used in production
    #
    # CAVEAT: Gemini Deep Think is NOT exposed via the standard API as of
    # 2026-04-29 — only via consumer Gemini app (Google AI Ultra subscription)
    # or the early-access program. agent_type=gemini routes to standard
    # Gemini API thinking mode, which is roughly comparable to Sonnet's
    # extended thinking but NOT the deeper Deep Think mode used for the
    # six-round Phase-3 → Phase-7 architectural derivation chain.
    #
    # See:
    #   - openspec/change-proposals/multi-agent-routing.md
    #   - scripts/research_conductor.py (per-task agent_type override path)
    agent_type: Literal["claude", "codex", "gemini", "opencode"] | None = None

    @field_validator("deliverable")
    @classmethod
    def deliverable_must_be_json_under_results(cls, v: str) -> str:
        if not v.startswith("results/") or not v.endswith(".json"):
            raise ValueError(
                f"deliverable must start with 'results/' and end with '.json', got: {v!r}"
            )
        return v

    @model_validator(mode="after")
    def model_must_match_agent_type(self) -> "ResearchTask":
        """Cross-field validator: model identifier must be valid for the agent_type.

        For agent_type=claude (or None, falling through to default Claude), the
        model field is restricted to {sonnet, opus} — typos like 'sonet' or
        'haiku' (retired) or 'gpt-4' (wrong vendor) are caught at parse time.

        For other agent types (codex/gemini/opencode), the model field accepts
        any non-empty string because the schema can't enumerate every vendor's
        model catalogue. Operator/planner is responsible for picking valid
        identifiers per agent.
        """
        if self.model is None:
            return self
        effective_agent_type = self.agent_type or "claude"
        if effective_agent_type == "claude":
            if self.model not in ("sonnet", "opus"):
                raise ValueError(
                    f"For agent_type=claude (or default), model must be 'sonnet' "
                    f"or 'opus'; got {self.model!r}. If you intended a different "
                    f"agent backend, set agent_type=codex/gemini/opencode."
                )
        else:
            if not isinstance(self.model, str) or not self.model.strip():
                raise ValueError(
                    f"For agent_type={effective_agent_type!r}, model must be a "
                    f"non-empty string identifying the vendor's model "
                    f"(e.g. 'gpt-5.5' for codex, 'gemini-3.1-pro-preview' for "
                    f"gemini); got {self.model!r}."
                )
        return self


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
