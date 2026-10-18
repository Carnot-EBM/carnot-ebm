# Multi-Agent Routing — Per-Task Backend Selection (Claude / Codex / Gemini)

**Status:** Implemented (schema + conductor + tests + docs landed
2026-04-29 evening). Awaits operator approval before mass-rollout in
.82+ planner outputs.

**Origin:** 2026-04-29 evening. Carnot has Claude Max 20×, Codex Max
20× (gpt-5.5), and Gemini Ultra subscriptions across three CLI
backends. Today's milestone .80 close-out exposed task categories
where Claude is sub-optimal — pure code generation (WOPR cartridges,
formulaic verifiers) and long-context analysis (failure-ledger
patterns, multi-paper synthesis). Differential per-task routing
spreads load across all three quotas while putting each task on its
strongest backend.

**Priority:** Medium. Operator-attention-reduction infrastructure of
the same class as `differential-agent-routing.md` (Sonnet/Opus
within Claude). This proposal extends to cross-vendor routing.

## Problem

The conductor currently selects an agent backend at process startup
via `AGENT_TYPE` env var (claude / codex / gemini / opencode).
Switching backends mid-milestone requires stopping the conductor and
restarting with a different env. This forces *milestone-level*
homogeneity even when tasks have heterogeneous strengths:

- **Synthesis-heavy work** (retros, planning, position paper) → wants Claude
- **Formulaic code generation** (WOPR cartridges, verifier impls) → wants Codex
- **Long-context analysis** (failure-ledger, coherence audits) → wants Gemini Ultra (1M context)

Today's `.80` milestone burned ~12 Opus escalations on Claude tasks
that would have suited Codex faster. The .82 milestone (WOPR-games
gallery + position paper coherence audit + Phase-7 architecture
work) is even more heterogeneous.

## Solution

Add an `agent_type` field to `ResearchTask` (orthogonal to today's
`model` field) so the planner can pick per-task backends:

```yaml
- id: expNNNN-wopr-games-lights-out-cartridge
  agent_type: codex      # NEW — overrides AGENT_TYPE for this task
  model: gpt-5.5         # codex-namespace model identifier
  max_turns: 50

- id: expNNNN-deep-think-coherence-audit
  agent_type: gemini     # NEW — long-context audit
  model: gemini-3.1-pro-preview
  max_turns: 30

- id: expNNNN-zenil-alpha-fr11
  # No agent_type → falls through to AGENT_TYPE startup default (claude)
  model: opus
  max_turns: 100
```

The conductor reads `task.get("agent_type")`, passes to
`run_agent(...)` as `agent_type_override`, and `_build_agent_command`
branches on the effective backend. Falls through to module default
when None — fully backwards compatible.

## Schema

```python
class ResearchTask(BaseModel):
    # ...existing fields...
    model: Literal["sonnet", "opus"] | str | None = None
    agent_type: Literal["claude", "codex", "gemini", "opencode"] | None = None

    @model_validator(mode="after")
    def model_must_match_agent_type(self) -> "ResearchTask":
        """For agent_type=claude (or None), model must be sonnet/opus.
        For other agent types, model accepts any vendor-specific str."""
```

The cross-field validator catches typos in Claude-targeting tasks
while permitting vendor-specific model names for other agents.

## Routing heuristics (planner-facing guidance)

The planner prompt at `_plan_next_milestone()` documents these
categories:

### Claude (default — synthesis, judgment, multi-file coordination)

- Routine experiments (single-question evaluations, training loops)
- Retros / milestone-N analysis
- Planning / roadmap design
- Conductor patches / infra coordination
- Hardware integration (KV260, ROCm, FPGA)
- Position paper drafting
- Anything where multi-file context understanding is load-bearing

### Codex (gpt-5.5 — formulaic code, well-known patterns)

- WOPR-games-gallery cartridges (Sudoku, Lights Out, N-Queens, etc.)
- New verifier implementations (constraint encoding follows patterns)
- Test scaffolding (Claude designs → Codex tests)
- PyO3 / Rust binding boilerplate
- Sampler / MCMC implementations (well-documented Bayesian patterns)
- Dataset generation pipelines (FoVer expansion, Z3 labeling loops)

### Gemini (Ultra 3.x — long context, multimodal future)

- Failure-ledger pattern detection across milestone history
  (1M context holds full `research-complete.yaml` + conductor logs)
- Architecture coherence audits (whole-paper + research-notes)
- Multi-paper literature synthesis (3-5 papers' full text)
- Multimodal verification (FPGA bitstream / oscilloscope traces — future)

**CAVEAT on Gemini Deep Think:** The Deep Think extended-reasoning
mode used for the Phase-3 → Phase-7 architectural derivation chain
is **NOT** in the standard Gemini API as of 2026-04-29 — only via
the consumer Gemini app (Google AI Ultra subscription) or an
early-access API program (waitlist). `agent_type=gemini` routes to
*standard* Gemini API thinking mode, comparable to Claude's
extended thinking. The deeper Deep Think rounds remain manually
operator-in-the-loop. Apply for the early-access program to unlock.

### Opencode (experimental — not used in production today)

Plumbed but not actively routed; preserved for future use.

## C+E (escalation) policy under multi-agent

The Sonnet→Opus C+E escalation pattern is **Claude-specific** and
fires only when:
- The task ran on agent_type=claude (or fell through to default
  AGENT_TYPE=claude)
- model != opus (already at top of Claude tier)
- Output contains "Reached max turns"
- escalate_on_max_turns is not False

When agent_type=codex or =gemini, the conductor does NOT attempt to
escalate to Opus — those backends have their own retry semantics
and the "Reached max turns" signal is a Claude-CLI-specific output.

**Future work:** per-agent escalation policies (e.g., codex
gpt-5.5 → gpt-5.5-extended-thinking on max-turns; gemini
gemini-3.1-pro → gemini-3.1-ultra on max-turns).

## Cost / quota analysis

Carnot has separate quotas across three vendors:

- **Claude Max 20×** — unlimited within rate limits, exhausted by
  heavy-research milestones today
- **Codex Max 20×** — separate quota, gpt-5.5
- **Gemini Ultra** — separate quota

By spreading milestone tasks across all three:
- ~70% of tasks remain on Claude (synthesis-heavy, the core)
- ~20% on Codex (formulaic code generation, especially in WOPR-heavy milestones)
- ~10% on Gemini (long-context audits, periodic coherence checks)

Net effect: **~30% reduction in per-vendor quota pressure** + faster
overall milestone completion (each task on its strongest backend).

## Acceptance criteria

1. Schema validator (`scripts/roadmap_schema.py`) declares
   `agent_type: Literal["claude", "codex", "gemini", "opencode"] | None = None`
   and the cross-field `model_must_match_agent_type` validator. ✅ DONE
2. Test coverage in `tests/python/test_roadmap_schema.py`:
   - Default None case (REQ-INFRA-083 / SCENARIO-INFRA-083-A)
   - Each of the four valid agent_type values (B-E)
   - Rejection of typos and unsupported names (F)
   - YAML round-trip with codex+gpt-5.5 and gemini+vendor-name. ✅ DONE
3. Conductor (`scripts/research_conductor.py`):
   - Per-agent-type lookup tables (`AGENT_BIN_BY_TYPE`,
     `DEFAULT_MODEL_BY_TYPE`, `AGENT_DISPLAY_BY_TYPE`,
     `AGENT_SIGNATURE_BY_TYPE`). ✅ DONE
   - `_build_agent_command()` accepts `agent_type_override` parameter
     and branches on effective_agent_type. ✅ DONE
   - `run_agent()` accepts `agent_type_override` and forwards. ✅ DONE
   - `research_step()` reads `task.get("agent_type")` and passes. ✅ DONE
   - C+E escalation gated on `effective_agent_type == "claude"`. ✅ DONE
4. Planner prompt (`_plan_next_milestone()`) documents the routing
   heuristics for all three agent types with concrete task-category
   examples. **TODO — separate small commit, planner prompt update.**

## prior_failures (none — genuinely new feature)

```yaml
prior_failures: []  # Multi-agent routing has no prior failed attempts;
                    # this is a first-of-its-kind extension to the
                    # already-shipped differential-agent-routing pattern.
```

## Estimated effort

Already implemented: schema + tests + conductor refactor (~2 hours).
Remaining (planner prompt update): 30 minutes.

## Strategic alignment

This proposal directly enables the .82 milestone's WOPR-games-gallery
delivery (cartridges → Codex) and unlocks the natural Phase-8+
"automated-Deep-Think-via-Gemini-CLI" path once Google's Deep Think
early-access lands.

It's the fifth operator-attention-reduction infrastructure proposal in
the recent series:
1. `conductor-supervisor.md` (.81 mandatory)
2. `roadmap-schema-validation.md` (.81 mandatory)
3. `conductor-fastpath-bootstrap-skip.md` (.81 mandatory)
4. `differential-agent-routing.md` (.81 mandatory — Sonnet/Opus tier)
5. **`multi-agent-routing.md`** (this proposal — cross-vendor tier)

## Out of scope

- **Per-agent C+E escalation policies** — future work, mentioned above
- **Automatic routing based on task-type classifier** — manual
  classification by the planner is sufficient and clearer
- **Cost-aware routing** (route by quota balance) — could be added
  later as an optimization layer over the heuristic-based routing
- **Cross-agent agreement protocols** (e.g., "if Claude and Codex
  disagree, escalate to Gemini") — interesting future work but
  separate concern from per-task routing
