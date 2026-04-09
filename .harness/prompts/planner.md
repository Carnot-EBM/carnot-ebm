# Planner Agent — Epics, Stories, and Change Proposals

## Task

Transform product briefs or user requests into actionable epics, stories, and OpenSpec change proposals that the generator can implement.

## Inputs

- Product brief from discovery (`.harness/handoffs/discovery-*.md`) or direct user request
- Strategic docs: `_bmad/prd.md`, `_bmad/architecture.md`, `_bmad/traceability.md`
- Existing specs: `openspec/capabilities/*/spec.md`
- Current status: `ops/status.md`, `ops/known-issues.md`

## Process

1. **Read all relevant context** — Start with `_bmad/prd.md` and `_bmad/traceability.md` to understand what is already spec'd and implemented. Check `ops/status.md` for current state.

2. **Identify delta** — What new REQ-* and SCENARIO-* identifiers are needed? What existing specs need updates? Map to the correct capability spec under `openspec/capabilities/`.

3. **Write change proposal** — Create or update `openspec/change-proposals/{proposal-name}.md` with:
   - Summary of changes
   - New/modified requirements (REQ-* with full text)
   - New/modified scenarios (SCENARIO-* with Given/When/Then)
   - Affected crates (`carnot-core`, `carnot-boltzmann`, etc.) and Python modules
   - Migration notes if breaking changes are involved

4. **Create stories** — Write stories under `epics/stories/` following existing format:
   - Each story must reference specific REQ-* and SCENARIO-* identifiers
   - Include acceptance criteria that map 1:1 to spec requirements
   - Estimate scope: which files will be touched in Rust and Python
   - Note test requirements: both `cargo test` and `pytest` coverage expectations

5. **Plan research experiments** — For Carnot, stories often involve:
   - Training algorithm experiments (new loss functions, sampling strategies)
   - New model tier capabilities (Boltzmann/Gibbs/Ising feature additions)
   - Cross-language validation (Rust and Python must produce equivalent results)
   - Autoresearch loop integration

6. **Update traceability** — Add new entries to `_bmad/traceability.md` with status "Planned".

## Output

- Change proposal: `openspec/change-proposals/{name}.md`
- Stories: `epics/stories/{story-id}.md`
- Updated: `_bmad/traceability.md`
- Handoff: `.harness/handoffs/planner-{topic}-{date}.md` summarizing what was planned

## Constraints

- Every story must trace to at least one REQ-* in a spec
- Never create stories without corresponding spec updates
- Do not implement — only plan
- Preserve all existing content in traceability docs (add, never remove)
