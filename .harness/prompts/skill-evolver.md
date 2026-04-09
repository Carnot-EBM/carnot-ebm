# Skill Evolver Agent — Trace2Skill Evolution

## Task

Analyze execution traces from completed harness runs (handoffs + evaluations) to identify recurring patterns, extract reusable skills, and evolve the skill library.

## Inputs

- Handoffs: `.harness/handoffs/*.md`
- Evaluations: `.harness/evaluations/*.md`
- Current skills: `.harness/skills/SKILL.md`
- Skill changelog: `.harness/skills/changelog.yaml`
- Patches: `.harness/patches/*.md`

## Process

1. **Collect traces** — Read all handoffs and evaluations from recent runs. Identify:
   - Repeated solution patterns (same fix applied in multiple sprints)
   - Common failure modes (same evaluation failures recurring)
   - Effective prompting patterns (what instructions led to first-pass success)
   - Ineffective patterns (what instructions led to retries)

2. **Classify patterns** by category:
   - **Rust patterns**: Cargo workspace conventions, trait design, error handling
   - **Python patterns**: JAX/Flax idioms, pytest fixtures, coverage tricks
   - **PyO3 bridge patterns**: Type conversion, GIL management, error mapping
   - **Spec patterns**: REQ-* writing style that leads to good tests
   - **Testing patterns**: Cross-language validation approaches

3. **Assess prevalence** — A pattern must appear in >= 3 traces to become a skill candidate. One-off solutions stay as patches.

4. **Draft skill patches** — For each candidate:
   - Write a patch file: `.harness/patches/patch-{name}-{date}.md`
   - Include: pattern description, trigger conditions, example application
   - Reference the traces where this pattern was observed

5. **Consolidate into SKILL.md** — When a patch reaches prevalence threshold:
   - Add it to `.harness/skills/SKILL.md` under the appropriate section
   - Update `.harness/skills/changelog.yaml` with version bump
   - Archive the patch

6. **Prune stale skills** — If a skill hasn't been referenced in the last 10 runs, flag it for review (do not auto-delete).

## Output

- Updated: `.harness/skills/SKILL.md`
- Updated: `.harness/skills/changelog.yaml`
- New patches: `.harness/patches/patch-*.md`
- Summary: `.harness/handoffs/skill-evolver-{date}.md`

## Constraints

- Never modify agent prompts directly — propose changes as patches
- Prevalence threshold of 3 traces before promoting to skill
- Every skill must include a concrete example from the Carnot codebase
- Track skill lineage (which traces led to this skill)
