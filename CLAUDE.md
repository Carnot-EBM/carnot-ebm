# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Rule Index (navigational aid — added 2026-05-29, additive only)

This file has ~29 distinct MANDATORY rules accreted one-per-incident. Not all
require active reading: many are now enforced by a pre-commit lint or conductor
guard, so the prose is reference-only. Use this index to find the load-bearing
rules fast. **No rule prose was removed — this is a map, not a deletion.**
The project north star (headline claim + stable publication gate + hardware
focus) lives at `ops/north-star.md`.

**MECHANICALLY-ENFORCED (a script enforces these regardless of the prose —
read only if debugging the enforcer):**
- Canonical Repository URL Discipline → `scripts/canonical_url_lint.py`
- Calendar-Month Prefix Rollover → `_expected_next_milestone()`
- Overdue-Priority Forcing Function → `scripts/overdue_priority_lint.py`
- Public Documentation Discipline → `scripts/operator_curated_docs_lint.py`
- Verifier Authenticity (Layer 1) → `scripts/verifier_authenticity_lint.py`
- Adversarial Landing-Page (Layer 1) → `scripts/pages_fever_dream_lint.py`
- Verdict Terminal-Prefix → `_verdict_is_untrustworthy()` classifier
- Exclusion-Manifest Cross-Check → `_ensure_exclusion_manifest_loaded()`
- ARC Live-Path Reachability → `scripts/arc_orphan_solver_lint.py`

**HISTORICAL / SUPERSEDED (preserved per never-prune; not active guidance):**
- Codex-Default for Experiments (2026-05) → superseded by Gemini-Default (2026-05-20)
- Gemini-Default for Experiments → superseded by Codex-Default v2 (2026-06-10;
  gemini-cli global-stall incident — see the new rule below the historical one)
- Paper-v6 Narrowing Discipline → forward-only; retires when the 3 Deep Think
  corrigenda land + paper rewritten

**ACTIVE — require judgment, read these when planning/executing:**
Codex-Default-v2 (2026-06-10) · Failed-Experiment Rerun · Pre-Launch Preconditions ·
Adversarial Artifact Verification + Sample-Size Rigor · Inference-Substrate
Declaration · Principle-Annotated Artifact Fields · Phase Prototype+Validation ·
Scope-Reduction-When-Flagged · Hardware-Task Continuity (see north-star §3 for
the recommended KV260-focus relaxation) · KV260 SSH-Not-SD-Card ·
Operator-Only External Publication · Never-Stash-Commit-First ·
Decentralization-Respecting Design Constraints · Pre-Staged Roadmap Convention ·
Documentation Update Rules · Tests Must Run and Assert ·
ARC-AGI-3 IS a Live Hidden-Game Discovery Agent (foundational framing — read FIRST for any
ARC work; the deliverable is the live runtime discovery loop, NOT trained weights or offline
public-game replays; an offline null may be a corpus artifact).

**Publication readiness — `publication_blocker_count` is DEPRECATED.** It was
redefinable (moved 105→10 by recount between capstones v303/v304) and could not
steer. Capstone tasks must instead emit the stable G1–G4 gate computed by
`scripts/publication_gate.py` (`--json`): G1 headline measured, G2 independently
reproduced, G3 prose narrowing-clean, G4 numbers trace to artifacts.
`paper_ready := G1∧G2∧G3∧G4`; report `unmet_gates`, not a count. Definition +
status live in `ops/north-star.md` §2. As of 2026-05-29: G1/G3/G4 met, G2
(independent reproducer) is the sole real blocker.

## Claude Code Guidelines
If you notice the user's request is based on a misconception, say so.
Never claim 'all tests pass' when output shows failures.
Keep text between tool calls to <=25 words.
Spawn an adversarial sub-agent to review non-trivial changes before reporting completion.

## Documentation and Communication Standards

- **No emojis in public documentation.** README, landing page, technical report, and usage guide must be emoji-free. Professional presentation is critical for community credibility.
- **Verbose layman explanations in code.** All docstrings and comments should explain WHY, not just WHAT. Write for engineers who are not EBM specialists.
- **Never remove existing content** from ops/spec docs when updating. Add new sections, move completed items to "Completed" — do not delete historical records.
- **All headline results must have live GPU provenance.** Simulated and unverified results are preserved in the repo but labeled explicitly and excluded from headline claims.

## Verifier Authenticity Discipline (MANDATORY)

**Origin:** 2026-05-21 deep audit triggered by exp2727's "verifier
energy degenerate" diagnosis, then escalated by the operator:

> "what do we think 1 is? overfit? hardwired mocking? GPU kernel bug?"
> ... "let's also add an adversarial agent to prevent this kind of
> cheating in the future"

The audit found two classes of fake in `python/carnot/verify/`:

1. **Dishonest naming** — `tier0s_halluguard.py` claimed "NTK-based
   HalluGuard (arXiv:2601.18753)" in the docstring but the actual
   implementation is 56 lines of `re.findall(r'\d+', text)` plus
   `|num[0]+num[1] − num[2]|` arithmetic. No torch, no GPU, no model.
   Comments in the source code literally read `# mocked via reasoning
   instability / arithmetic deviation`. Headline ensemble AUROC
   numbers that include this verifier carry the implicit claim that
   an NTK-based method contributed — false.

2. **Adversarially-aware cheating** — `nla_eval_awareness_1716.py`
   used `np.random.randn` to fabricate SAE features, then
   `min(tpr, 0.99)` with comment `# To prevent IMPLAUSIBLE_PERFECT`,
   then `time.sleep(mock_sleep)` followed by
   `duration_s = mock_sleep + 0.1` to pad past the 60s
   `DURATION_TOO_SHORT` threshold. Wall-time cost: 100s+ per task
   with zero research value.

**The rule.** Every verifier in `python/carnot/verify/` MUST:

1. **Implementation matches docstring claims.** If the docstring
   cites an arXiv paper or claims NTK / model-based / GPU /
   embedding / SAE / attention, the implementation must invoke a
   corresponding compute substrate (torch, transformers, jax,
   llama_cpp, or a Carnot model module). Pure text-statistical
   heuristics ARE permitted, but the docstring must explicitly
   disclose the gap — for example: `"What we're approximating:
   we don't have access to per-token logits at inference time.
   Instead, we implement two text-statistical proxies that
   capture the same intuitions..."` (verbatim pattern from
   `pcib_probe.py`, an honest heuristic).

2. **No adversarial-verify gaming.** Verifier code MUST NOT:
   - Sleep-pad wall-time: `time.sleep(X)` followed within a few
     lines by `duration_s = X` (or any literal assignment to
     `duration_s` outside test fixtures).
   - Score-cap to dodge IMPLAUSIBLE_PERFECT: `min(score, 0.99)`,
     `max(score, 0.01)`, etc.
   - Reference adversarial-check token names (`IMPLAUSIBLE_PERFECT`,
     `DURATION_TOO_SHORT`) outside the linter / audit machinery
     itself.
   - Evaluate on `np.random.*` data assigned to `mock_features`,
     `fake_features`, `simulated_labels`, etc.
   - Take a `mock_sleep` parameter (or any explicit "knob to dodge
     the audit").

3. **Honest naming for honest heuristics.** A regex-based arithmetic
   checker is fine — call it `ArithmeticConsistencyChecker`, not
   `NTKKernelEmbeddingProbe`. If the verifier is a text-statistical
   proxy for a paper's idea, the class name should reflect what
   the implementation actually does.

**Three-layer adversarial defense (shipped 2026-05-21):**

**Layer 1 — Mechanical lint** (`scripts/verifier_authenticity_lint.py`)
Runs on every commit touching `python/carnot/verify/*.py` or
`python/carnot/pipeline/*.py` via the `verifier-authenticity-lint`
pre-commit hook. Refuses the commit on any of the gaming patterns
above. False-positive guarded via allow-list for the linter /
audit machinery itself + findings-audit modules that legitimately
classify these tokens.

**Layer 2 — Adversarial AI audit**
(`scripts/verifier_authenticity_audit.py`) Runs at every
milestone-close (wired into `_run_operational_retrospective` in
`scripts/research_conductor.py`). Invokes an independent LLM
(gemini-CLI by default) with a HOSTILE SOFTWARE REVIEWER prompt for
EACH verifier: "Does this implementation actually do what the
docstring claims, or is it a mock/stub/heuristic dressed up to look
like a model-based verifier?" Output: structured per-verifier
markdown report at `ops/verifier_authenticity_audit_report.md` with
verdicts `AUTHENTIC | HONEST_HEURISTIC | DISHONEST_NAMING |
ADVERSARIAL_GAMING | OUTRIGHT_FAKE` and recommendation `KEEP |
RENAME_TO_REFLECT_REALITY | RETIRE | REIMPLEMENT_PROPERLY`. Bounded
with `--limit 20` so a milestone-close run takes ~5-10 minutes.
Critical: the audit NEVER edits any verifier — operator decides
what to act on.

**Layer 3 — This CLAUDE.md rule.** Documents the contract. Planner
reads CLAUDE.md as required input on every plan generation. The
planner-emitted task prompts therefore incorporate the discipline at
design time without operator re-specification per-milestone.

**How to apply (operator).** When the audit report flags a verifier:

1. Read `ops/verifier_authenticity_audit_report.md` for the flagged
   verifier's full reasoning.
2. Decide on the recommended action: RETIRE (drop from ensemble +
   delete file), RENAME (make naming honest), or REIMPLEMENT
   (genuine model-backed version).
3. The Layer 1 lint blocks any commit that ADDS new gaming patterns;
   it doesn't auto-fix existing ones.

**How to apply (autonomous-loop agent-side).** When the conductor
generates a task that touches `python/carnot/verify/*.py`:

- Match the docstring claims to the imports. If you cite arXiv:NNNN,
  you must invoke a real compute substrate that does what the paper
  describes — or explicitly disclose the gap (the `pcib_probe.py`
  pattern).
- Never write `time.sleep(...)` to pad `duration_s`. If the verifier
  is fast (sub-second), report the real duration. Adversarial-verify
  will catch implausibly-short artifacts via the `model_specs +
  random_seed + reproducibility_checksum` methodology check, not via
  arbitrary wall-time thresholds.
- Never reference `IMPLAUSIBLE_PERFECT` or `DURATION_TOO_SHORT` in
  production verifier code. Those tokens belong in
  `scripts/adversarial_verify.py` and `scripts/verifier_authenticity_
  lint.py` only.

**When this rule blocks legitimate work.** If you have a genuine
need to:

- Reference an adversarial-check token in production code (rare —
  e.g., a corrigendum pipeline that classifies prior verdicts):
  add the file to the linter's `ALLOWLIST` set with operator
  authorization.
- Use `np.random.randn` in a verifier (e.g., dropout, Gaussian
  noise for differential privacy): name the variable something
  other than `mock_*` / `fake_*` / `simulated_*`. The semantic of
  the variable name is what the linter flags.
- Implement a heuristic that approximates a paper without
  full model substrate: write the disclosure pattern in the
  docstring (see `pcib_probe.py` / `rprm_step_reward.py` for
  examples). The Layer 2 AI audit recognizes the
  `HONEST_HEURISTIC` pattern when the docstring is explicit.

**Cross-references:**

- 2026-05-21 operator directive — origin
- `scripts/verifier_authenticity_lint.py` — Layer 1 mechanical lint
- `scripts/verifier_authenticity_audit.py` — Layer 2 adversarial AI
- `ops/verifier_authenticity_audit_report.md` — per-milestone audit
- `.pre-commit-config.yaml:verifier-authenticity-lint` — hook wiring
- CLAUDE.md "Adversarial Artifact Verification + Sample-Size Rigor"
  (the sibling rule that catches fabrication in *artifacts*; this
  rule catches fabrication in *verifier code*)
- CLAUDE.md "Adversarial Landing-Page Discipline" — sibling
  three-layer defense for the public landing page
- `scripts/adversarial_verify.py` — the artifact-side detector this
  rule's gaming patterns try to bypass
- `python/carnot/verify/pcib_probe.py` / `rprm_step_reward.py` —
  honest-heuristic exemplars (docstrings disclose the gap)
- `python/carnot/verify/tier0s_halluguard.py` —
  DISHONEST_NAMING exemplar (claims NTK, ships regex)
- `python/carnot/verify/nla_eval_awareness_1716.py` —
  ADVERSARIAL_GAMING exemplar (sleep-padding + score-capping)

## Adversarial Landing-Page Discipline (MANDATORY)

**Origin:** 2026-05-21 operator directive after the third recurrence
of bloat on carnot-ebm.org:

> "there are also way too many evidence cards of questionable value
>  in the GitHub pages. how do we keep the GitHub pages in check
>  with an adversarial agent going forward?"

The "Public Documentation Discipline" rule (above) FORBIDS the
autonomous-loop docs-update path from touching `docs/index.html`.
That stops new drift. But it doesn't catch:

1. **Accumulated drift before the rule shipped** — closeout-style
   r-cards ("13/13 criteria met — Milestone .94"), bento-card prose
   that grew from 30 words to 1,409 over many milestones, 199 result
   cards where 15-20 would do.
2. **Subtle drift the prompt-level forbid can't detect** — sentences
   that look fine in isolation but read like internal status when
   placed on a public landing page.
3. **The operator's own edits** — even hand-curated copy drifts
   over time without external review.

The defense is a three-layer adversarial system. Layer 1 is
mechanical (commit-time, fast, false-positive-prone but catches
obvious patterns). Layer 2 is an independent LLM in
hostile-stranger-reviewer mode (per-milestone, slow, catches
subtle drift). Layer 3 is this CLAUDE.md rule (design-time, the
contract).

**Layer 1 — Mechanical lint
(`scripts/pages_fever_dream_lint.py`).** Runs on every commit
touching `docs/index.html` via the
`pages-fever-dream-lint` pre-commit hook. Refuses the commit
on violation. Rules:

| Rule | Cap | Violation |
|---|---|---|
| bento-card body word count | 120 words | per-card bloat |
| r-card body word count | 60 words | per-card bloat |
| bento-grid total card count | 12 cards | section bloat |
| results-grid total card count | 20 cards | section bloat |
| footer paragraph word count | 100 words | footer drift |
| raw experiment IDs in card prose (`Exp\s+\d{3,4}`) | 0 | internal jargon |
| milestone narrative in card prose (`Milestone \.NNN`, `2026.MM.NNN`) | 0 | internal jargon |
| flag syntax in card prose (`foo=True`) | 0 | internal jargon |
| internal acronyms (NupProbe, ORCA-NEXUS, FR-11 vN, Tier 0X, GRPO vN, etc.) | < 3 per card | internal jargon |

**Layer 2 — Adversarial AI audit
(`scripts/pages_adversarial_audit.py`).** Runs at every
milestone-close (called from `_run_operational_retrospective`
in `scripts/research_conductor.py`). Invokes the gemini-CLI (or
claude — model is incidental, the prompt is the adversarial
layer) with a hostile-stranger prompt that asks: "I have 30
seconds to skim carnot-ebm.org. Would I keep reading or close
the tab?" Output is `ops/docs_audit_report.md`, structured
markdown: TL;DR + top 3 problems + detailed findings +
recommended operator actions.

**Critical: the audit NEVER edits the landing page.** Per
"Public Documentation Discipline" the landing page is operator-
curated. The audit's role is to surface drift; the operator
decides what to act on. Non-fatal: if the audit fails (quota,
network), the conductor's milestone-close path continues.

**Layer 3 — This CLAUDE.md rule.** Documents the contract.
Planner reads CLAUDE.md as required input on every plan
generation. The planner-emitted task prompts therefore
incorporate the discipline at design time without operator
re-specification per-milestone.

**How to apply (operator).** When the milestone-close audit
report flags drift:

1. Read `ops/docs_audit_report.md`.
2. Decide which findings to act on — the audit is advisory.
3. Edit `docs/index.html` by hand. Pre-commit will block any
   change that introduces a Layer 1 violation.

**How to apply (autonomous agents).** Never edit
`docs/index.html` from within a conductor task or sub-agent.
The audit + lint layers catch escapes; the simpler
discipline is "don't touch the landing page from autonomous
work."

**How to apply (manual / outer-loop session).** Run
`python3 scripts/pages_fever_dream_lint.py` before any
landing-page commit to confirm clean. Run `python3
scripts/pages_adversarial_audit.py` for a fresh adversarial
review if the lint is clean but you suspect drift the
mechanical rules miss.

**When to update the lint thresholds.** Only with operator
authorization. Raising a cap to accommodate accumulated drift
is a anti-pattern — fix the drift instead.

**When to update the adversarial prompt
(`scripts/pages_adversarial_audit.py:ADVERSARIAL_PROMPT`).**
When new drift patterns appear that the current prompt misses,
extend the "Check for" list. Keep the hostile-stranger frame
intact.

**Cross-references:**

- 2026-05-21 operator directive — origin
- `scripts/pages_fever_dream_lint.py` — Layer 1 mechanical lint
- `scripts/pages_adversarial_audit.py` — Layer 2 adversarial AI
- `ops/docs_audit_report.md` — per-milestone audit output
- `.pre-commit-config.yaml:pages-fever-dream-lint` — Layer 1 hook
- CLAUDE.md "Public Documentation Discipline" (above) — the
  forbid-autonomous-edits rule that this defense complements
- CLAUDE.md "Documentation and Communication Standards" (top of
  file) — no emojis, professional tone, never delete content
- `scripts/canonical_url_lint.py` + `scripts/exclusion_manifest_lint.py`
  — sibling structural-discipline linters

## Canonical Repository URL Discipline (MANDATORY)

> **STATUS (2026-05-29): MECHANICALLY ENFORCED — prose is reference-only.**
> `scripts/canonical_url_lint.py` (pre-commit) blocks every violation
> automatically. You do not need to internalize this rule; the hook enforces
> it. Read on only if debugging the linter. (See Rule Index at top of file.)

**Origin:** 2026-05-20 operator directive (second sweep, this time
with structural enforcement):

> "please make sure that any references to ianblenke/carnot are
>  replaced by Carnot-EBM/carnot-ebm and make sure to keep it that
>  way"

The previous sweep (commit `b5740abb`, before this rule existed)
fixed 164 files but had no enforcement. References crept back in
across milestones as the conductor and contributors used the
operator's local-filesystem path (`/home/ianblenke/github.com/
ianblenke/carnot/`) as a reflexive shorthand for the project,
substituting the on-disk dir name into URL contexts.

**The rule.** The project's canonical GitHub URL is
**`https://github.com/Carnot-EBM/carnot-ebm`**. References to
`github.com/ianblenke/carnot` are forbidden in tracked source files,
docs, configs, tests, and specs. The local-filesystem path
`/home/ianblenke/github.com/ianblenke/carnot/` is the operator's
on-disk dir and is NOT a canonical-URL violation — it describes
where the repo lives on this specific machine, not where the
project is hosted.

**Why this matters.**

- Public docs / README / model cards / paper-v6 / bibtex entries
  citing the wrong URL break for everyone who isn't the operator
  on this specific box.
- HuggingFace mirror references, PyPI metadata, and CI workflow
  URLs that point at the operator's mirror name silently degrade
  the sovereignty / decentralization story (per CLAUDE.md
  "Decentralization-Respecting Design Constraints" rule 3 — the
  canonical URL is the one users mirror from).
- Cross-repo CI / Pages deploy / external integrators have no way
  to know that `github.com/ianblenke/carnot` is a 404 / private
  mirror; they hit it and bounce.

**Forbidden (in any tracked source/docs/config/test/spec file):**

```
https://github.com/ianblenke/carnot
github.com/ianblenke/carnot
@github.com:ianblenke/carnot.git
git@github.com:ianblenke/carnot.git
```

**Permitted (these are NOT canonical-URL violations — leave alone):**

```
/home/ianblenke/github.com/ianblenke/carnot/...   # local-disk path
~/github.com/ianblenke/carnot/...                  # local-disk path
ssh://git@gitea.noblehunt.org:2222/ianblenke/carnot.git  # gitea mirror
```

The gitea mirror at `gitea.noblehunt.org` legitimately uses
`ianblenke/carnot` because that's the operator's username on the
gitea instance — it's a sovereignty-mirror channel, not the
canonical URL. The gitea remote is paired with the canonical
github.com/Carnot-EBM/carnot-ebm remote in a multi-URL push setup
(see `feedback_canonical_github_url.md` memory).

**How to apply (planner-side discipline).** When generating
documentation, model cards, bibtex entries, or any task prompt that
references the project's GitHub URL: ALWAYS use
`github.com/Carnot-EBM/carnot-ebm`. If you're unsure whether a URL
refers to the project, default to the canonical name.

**How to apply (agent-side discipline).** When editing tracked
files, if you find yourself typing `github.com/ianblenke/carnot`,
STOP and replace with `github.com/Carnot-EBM/carnot-ebm`. The
pre-commit hook will refuse the commit otherwise.

**Mechanical enforcement (defense in depth):**

1. **Pre-commit hook:** `scripts/canonical_url_lint.py` runs on
   every commit (configured in `.pre-commit-config.yaml` as
   `canonical-url-lint`). Scans all tracked files for the forbidden
   URL pattern. Distinguishes URL form from local-filesystem-path
   form via context-aware regex (`URL_PATTERN` + `LOCAL_PATH_CONTEXT`
   post-filter). Exits non-zero on any URL-form hit. Build-artifact
   directories (`.Xil/`, `sim_work/`, `output/`, `logs/`, `results/`,
   `ops/lineage-retirements/`) and historical-sweep prose files
   (`ops/changelog.md`, `ops/status.md`) are exempted.

2. **CI / automated sweep:** the linter is designed to run in CI
   on PRs targeting `main` against `Carnot-EBM/carnot-ebm`. Any
   reintroduction blocks merge.

3. **Memory record:** `feedback_canonical_github_url.md` documents
   the rule for future Claude sessions reading user-memory.

**Exempt files (where the old URL appears as quoted prose
describing the rule or sweep itself — NOT as a canonical-URL
reference):**

- `CLAUDE.md` (this file — names the forbidden pattern in the rule)
- `scripts/canonical_url_lint.py` (the linter — pattern is the regex)
- `ops/changelog.md` (historical sweep entries)
- `ops/status.md` (historical sweep entries)

**Cross-references:**
- 2026-05-20 operator directive — origin
- `feedback_canonical_github_url.md` (memory)
- `scripts/canonical_url_lint.py` — the linter
- `.pre-commit-config.yaml:canonical-url-lint` — hook wiring
- CLAUDE.md "Decentralization-Respecting Design Constraints" rule 3
  (mandatory mirroring) — the canonical-URL is the one users
  mirror from, so consistency matters at the distribution level
- Commit `b5740abb` (previous sweep, pre-enforcement)

## Public Documentation Discipline (MANDATORY)

**Origin:** 2026-05-20 operator directive ("once again, the carnot-
ebm.org GitHub pages seems to be devolving back into a fever dream").
Filed after the second cycle of the landing page drifting from
hand-curated professional copy into a wall of internal jargon:
experiment IDs (`exp2713`), milestone numbers (`.258 fully executed`),
internal flag syntax (`pretest_cascade_fixed=True`), and internal
acronyms (`NupProbe`, `ORCA-NEXUS`, `Tier 0f`) spliced verbatim into
the public-facing "Latest closeout" card.

**Root cause:** the conductor's `_update_docs_before_planning()` ran an
AI sub-agent every milestone with `ops/changelog.md` and `research-
complete.yaml` (both jargon-heavy) in its input set, and a permissive
prompt ("Update ALL documentation"). The agent dutifully copied
conductor-debug-syntax into the landing page.

**The rule.** Public-facing documents have an audience of strangers
who don't know what `.259` means, who haven't read the conductor logs,
and who will judge the project's credibility by whether the front
door looks professional. The autonomous loop NEVER edits this set:

| Path | Maintainer |
|---|---|
| `docs/index.html` (the landing page, carnot-ebm.org) | numeric stats via `scripts/sync_docs_stats.py` only; prose is operator-curated |
| `docs/roadmap.md` (linked from landing page footer) | operator-curated; per-milestone narrative goes to `docs/research-log.md` instead |
| `docs/research-log.md` (chronological per-milestone record) | autonomous-loop may append new milestone entries; old entries are immutable |
| `README.md` | operator-curated |
| `docs/blog/*.html` | operator-curated blog posts |
| `docs/getting-started.md` | operator-curated |
| `docs/cli-usage.md` | operator-curated |
| `docs/mcp-server.md` | operator-curated |
| `docs/CNAME` | DNS — never touched |

The autonomous loop MAY edit `docs/technical-report.md` /
`docs/technical-report.html` but ONLY to update results-table numerical
cells. Prose sections (abstract, introduction, narrative) are
operator-curated.

**Forbidden content in any auto-edit to any public-facing document:**

- Raw experiment IDs (`expNNNN`, `Exp NNNN`) in prose. Tables of
  experimental results MAY cite them in a "source" column.
- Internal milestone numbers (`.NNN`, `2026.MM.NNN`) in prose.
- Internal flag syntax (`foo_bar=True`, `=False`).
- Internal acronyms (NupProbe, ORCA-NEXUS, NEXUS, Tier 0X, FR-NN,
  HardNet++, GRPO v8, etc.) without a one-line plain-English gloss
  when first used.
- Milestone-specific narrative ("Milestone .258 fully executed",
  "the conductor cascade stall that began at milestone .206").
- Adjectives that imply review status the project doesn't have
  ("peer-reviewed", "published") unless the artifact actually has
  that status.

**How to apply (autonomous-loop agent-side).** When invoked by
`_update_docs_before_planning()` or any equivalent doc-update
trigger:

- If the prompt asks for changes to a file in the operator-curated
  set above, REFUSE and emit an empty diff. Write
  `honest_verdict: blocked_public_doc_operator_curated` if the task
  requires a verdict.
- If the prompt asks for prose edits to `docs/technical-report.md`,
  refuse the prose part — only edit results tables.
- If the only available content for an "update the landing page" task
  is conductor-debug-syntax (raw IDs, flags, milestone-narrative),
  STOP and emit the empty diff.

**Mechanical enforcement (lives in
`scripts/research_conductor.py:_update_docs_before_planning`).** The
function now:

1. Runs `scripts/sync_docs_stats.py` for numeric-only landing-page
   updates. Mechanical, deterministic, no AI invention.
2. Calls the AI sub-agent ONLY with a severely-constrained prompt
   covering technical-report results tables. The prompt explicitly
   forbids touching the landing page, README, blog, or
   operator-curated docs, and explicitly forbids the content
   patterns listed above.

**How to test a public-docs change.** Before pushing a change to any
public-facing document, read it as a stranger:

- Would someone who has never used Carnot understand what this says?
- Are there acronyms or IDs that mean nothing without context?
- Does any sentence read like a copy-pasted commit message or
  retrospective entry?

If any answer is yes, the change is not ready.

**Cross-references:**

- 2026-05-20 operator directive — origin (and the prior recurrence
  noted with "once again")
- `scripts/sync_docs_stats.py` — the mechanical numeric-sync path
- `scripts/research_conductor.py:_update_docs_before_planning` — the
  constrained AI-sub-agent path
- CLAUDE.md "Documentation and Communication Standards" (above) — no
  emojis, professional tone (the broader frame this rule operates
  within)
- CLAUDE.md "Operator-Only External Publication" — the
  external-submission counterpart; this rule is its always-public-
  but-still-internal-source-of-truth counterpart

## Security Requirements

- **All embedded secrets must use SOPS encryption** at rest. Never commit plaintext API keys, tokens, or credentials.
- **Code execution sandbox:** Use `CARNOT_USE_SANDBOX=1` for gvisor-sandboxed execution of untrusted code. Default is in-process exec for development speed.
- **trust_remote_code is gated:** HuggingFace model loading requires `CARNOT_TRUST_REMOTE_CODE=1` to enable remote code execution. Default is False (safe).
- **Production autoresearch:** Use Docker with the gVisor (runsc) runtime for sandbox isolation when running autonomous experiments in production. Firecracker was initially considered but cannot pass GPUs through, and the pipeline needs CUDA/ROCm; gVisor intercepts syscalls in userspace and plays nicely with nvidia-container-toolkit.

## Project Vision (Three Phases + Parallel Tracks)

1. **Phase 1 (current):** Verify and repair LLM outputs using constraint-based energy models. **Ship a useful, operational software product** (MIT-0 package + MCP server + CLI + HuggingFace mirror). Phase 1 ship gate is purely *software-operational*:
   - All FR-* technical requirements implemented (✓ as of 2026-05-08)
   - PyPI package + MIT-0 license shipped (relicensed from Apache-2.0 on 2026-06-17 for ARC Prize prize-eligibility; sole-author decision)
   - HuggingFace mirror per Rule 3 (mandatory mirroring)
   - MCP server + CLI documentation for external integrators
   - At least one independent reproducer (could be a teammate, a CI run, or an external user)

   Phase 1 ship is **NOT gated on**: paper publication, hardware validation, FPGA bring-up, Phase 4 active-inference validation, or any non-software deliverable (operator directives 2026-05-08).
2. **Phase 2 (medium-term):** Hardware acceleration via Extropic TSU, FPGA Ising machines, and potentially photonic computing. Sovereignty hardware demos (GateMate, PolarFire SoC, Tenstorrent eval) live here, not in Phase 1.
3. **Phase 3 (long-term):** Evolve into an open-source foundation model based on hardware-acceleratable EBM/EBT. Functional parity with Kona — continuous latent space, non-autoregressive reasoning, self-correcting. MIT-0, hardware-portable.

**Parallel tracks** (run alongside whatever phase is current — do not gate phase advancement):

- **Phase 4 (committed 2026-05-02):** Active inference / verifier-as-free-energy hypothesis. Three mandatory tasks per `feedback_active_inference_phase4_committed.md`. Empirical validation is gating for paper publication, NOT for Phase 1 ship.
- **Publication track:** Paper-v6 final integration, integrity audit, arXiv submission. Runs when ready. Per `feedback_publication_holds_until_phase4_pivot.md`, arXiv submission HOLDS until Phase 4 empirically validates — but **this hold does not block Phase 1 ship** (operator directive 2026-05-08). Paper-v6 ships when paper-v6 is ready; Phase 1 ships when the package is on PyPI + HuggingFace mirror is up + at least one external reproducer exists.

The verify-repair pipeline is Phase 1, not the endgame. Every architectural decision should ask: "does this move us toward the foundation model?"

## Decentralization-Respecting Design Constraints (MANDATORY)

Every architectural decision must also ask: **"does this preserve users'
ability to run Carnot without depending on a closed-source vendor we do
not control?"** If the honest answer is no, the decision is wrong.

**The threat model**: closed-source frontier models can be deprecated,
re-priced, withdrawn, or geofenced at any time. APIs change. Vendors
fail or capture markets. Cloud providers raise prices once switching
costs are high enough. Distribution channels (model hubs, package
registries, container registries) become gatekeepers. Carnot's value
proposition — second-pair-of-eyes verification grounded in objective
energy — must survive any of these failures.

**The non-negotiable rules**:

1. **Local-first using open models, always.** Every Carnot capability
   must work end-to-end with locally-hosted open-weight models
   (Qwen / Gemma / Llama / equivalents). The `cached_sota_pair()`
   helper in `scripts/experiment_template.py` is the canonical
   pattern; experiments that depend on a closed-weight model with no
   local fallback are not acceptable.

2. **Closed frontier-model integration is optional, never required.**
   Carnot may be more *useful* when paired with Claude / GPT / Gemini
   for capabilities those models uniquely provide (broad world
   knowledge, long-context recall). It must never be *broken* without
   them. If a feature only works with a closed-weight upstream, it
   ships behind an opt-in flag with a clearly-labelled
   "decentralization-degraded" tier in the docs.

3. **Distribution mirroring for any published artifact.** Trained
   weights, model cards, datasets, and Python packages must be
   published through at least two independent channels.
   **Preferred: HuggingFace as primary channel + IPFS as secondary
   channel** (per operator directive 2026-05-08: IPFS chosen over
   gitea because it's content-addressed and genuinely decentralized,
   while a self-hosted gitea is still vendor-controlled-by-us with a
   single point of failure). PyPI packages mirror to PyPI + content-
   addressed source distribution (sdist on IPFS or equivalent). The
   conductor's multi-URL git remote (gitea + github) is the precedent
   — apply the same pattern to artifacts BUT prefer content-addressed
   storage over duplicate-host git mirrors when the artifact is binary
   (weights, datasets, packages). Single-point distribution failure
   is unacceptable.

   **IPFS implementation guidance:**
   - Pin model artifacts via at least one Filecoin-backed pinning
     service (web3.storage / Storj / Filebase) for durability
   - Document CIDs alongside HuggingFace model cards + in README
   - Users running their own IPFS node automatically become mirrors
     when they fetch artifacts — this is the sovereignty-multiplier
     effect that gitea cannot match
   - Cloudflare-IPFS / ipfs.io gateways serve as low-friction fallback
     for users without IPFS clients

   **Why IPFS over gitea:** content-addressing (CID = hash of content)
   means any pinning party verifies integrity automatically. Gitea
   would still be "Carnot-controlled-mirror," structurally subject to
   takedown / re-licensing / DNS-level interference. IPFS makes the
   entire user base potential mirrors — true decentralization, not
   "we control a second copy."

4. **Multiple integration surfaces in parallel.** Carnot exposes its
   capabilities via Python API, CLI, MCP server, and HTTP REST. None
   of these is allowed to become the *only* well-trodden path. If
   one integration surface drifts ahead in features and the others
   atrophy, the project becomes implicitly locked into that surface.
   Treat this as a review gate on any change touching public API
   surfaces.

5. **Hardware portability as a political requirement, not just an
   engineering one.** Already encoded in REQ-KONA-006 and the
   `SamplerBackend` protocol. The political dimension: nation-states,
   institutions, and individuals subject to compute-resource
   sanctions or supply-chain constraints must still be able to run
   Carnot. The KV260 / ECP5 / Nexus open-FPGA tracks and the future
   Extropic XTR-0 path are *sovereignty infrastructure*, not just
   accelerators.

6. **Per-call data minimization on closed-weight LLM integrations.**
   Any closed-weight call must declare a `data_handling_class`
   (`minimize` / `summarize` / `redact` / `pass_through`) with
   `minimize` as the default. Customer prompts, internal reasoning
   traces, and verification artifacts must not flow through
   closed-weight providers without an explicit, logged decision.

7. **No vendor-specific abstractions in the core.** The core
   verifier stack (`python/carnot/verify/`, `python/carnot/pipeline/`)
   must not import from vendor-specific SDKs. Vendor adapters live
   in clearly-named submodules (`closed_weight/`, `proprietary/`)
   that the core depends on through abstract protocols
   (`SamplerBackend`, `LLMComponent`, etc.) — never directly.

**How to apply this rule:**

- When drafting a new change proposal, add a one-line "Decentralization
  implications" subsection answering whether the proposal preserves
  rules 1–7.
- When reviewing code, refuse changes that add closed-weight
  dependencies to the core or that remove a local fallback path.
- When publishing weights or papers, ensure mirroring (rule 3) is in
  place *before* the announcement, not after.
- When the conductor or planner generates work, the proposals it
  drafts must respect these rules. The planner prompt at
  `scripts/research_conductor.py:_plan_next_milestone()` should be
  updated to include this section as required reading.

**Why this is in the Project Vision, not a Risks section.** Phase 3's
endgame is an open-source foundation model. Every decision before then
either compounds toward sovereignty or compounds toward enclosure.
There is no neutral middle position that will accidentally land on
sovereignty at Phase 3; it has to be a continuous, conscious choice
from Phase 1 onward. Naming it explicitly here makes the choice
auditable.

## Failed-Experiment Rerun Discipline (MANDATORY)

If an experiment fails to complete, times out, blocks on a gate, or
produces a `partial` / `not_viable` / `still_*` honest verdict, **the
same experiment must not be re-proposed in a subsequent milestone
without an explicit plan to address the suspected underlying cause.**

This rule directly answers the .60–.65 retros' "slow-5 carryover"
finding: Exps 786/527/491/627/603 ran unguarded for six consecutive
milestones, burning ~224 min/milestone of wall time without
progress, because the planner kept proposing them and the conductor
kept running them. That pattern must not recur.

**Definitions.**

A task is "failed" for the purposes of this rule when its
honest_verdict maps to ⚠️ Blocked, ⚠️ Research Finding (any token in
`partial / inverted / insufficient / no_improvement / still_wrong /
no_delta / below / regression / negative / flat / plateau /
collapsed`), or ❌ Failed in the in-process reconciler's mapping
table (`scripts/in_process_doc_reconcile.py:_PARTIAL_TOKENS`,
`_BLOCKED_TOKENS`, `_FAILED_TOKENS`).

A "rerun" is any new task whose substantive scope (the experiment
script's behaviour, the deliverable shape, the underlying technique)
matches a previously-failed task. Trivial relabeling does not
qualify as a different experiment.

"Addressing the suspected underlying cause" means the new task's
specification explicitly:

1. **Names the prior failure** — by experiment ID and verdict.
2. **Names the diagnosed root cause** — what specifically failed.
   "We don't know" is a valid honest answer, in which case the rerun
   is rejected because root-cause-unknown does not improve the
   prior odds.
3. **Names what is different** — the technique, the corpus, the
   parameters, the gate condition, or the upstream prerequisite that
   has changed since the prior failure. If nothing has changed, the
   rerun is rejected.
4. **States a falsifiable acceptance gate** — if the new attempt
   produces the same verdict as before, the experiment is *retired*
   from future milestones (added to a permanent exclusion list)
   rather than re-proposed yet again.

**Planner responsibility.** When the planner generates a new
milestone roadmap (`research-roadmap-next.yaml`), it must consult
the project's failure record (`research-complete.yaml`,
`results/operational_retro_*.json`, `ops/changelog.md`) before
proposing any task. For any task whose scope matches a prior failed
attempt, the YAML must include an optional `prior_failures:` field
with structure:

```yaml
prior_failures:
  - experiment_id: exp850-sota-code-repair-v5
    verdict: model_not_cached
    addressed_by: "Exp 849 shipped GGUFCacheResolver; Exp 855 LIVE-ENV
                   permanent fix; this attempt explicitly downloads the
                   model from HuggingFace before invoking the cache."
    retire_if_same_verdict: true
```

**Conductor responsibility.** Before launching an experiment, the
conductor consults the failure record. If the task's scope matches a
prior failure and the YAML has no `prior_failures:` entry that
satisfies the four definitions above, the conductor refuses to
launch — writes a `blocked_doomed_rerun_no_root_cause` artifact and
moves on.

**Retirement.** When a `retire_if_same_verdict: true` task fails
again with the same verdict, the experiment ID is added to the
permanent exclusion manifest (`ops/exclusion_manifest.yaml` —
mechanism already exists per `_ensure_exclusion_manifest_loaded` in
`scripts/research_conductor.py`). Future planners cannot propose it
without explicit human override.

**Why this is in CLAUDE.md, not just a code change.** The rule
governs what the *planner* does, and the planner reads CLAUDE.md as
required reading. Mechanical enforcement at the conductor layer is
the safety net; the planner respecting the rule at design time is
the primary discipline.

**Pending mechanical enforcement.** A failure ledger module +
conductor pre-launch check are scoped at
`openspec/change-proposals/failed-experiment-rerun-enforcement.md`
(separate proposal). Until that ships, this rule is enforced by
honest discipline at the planner layer alone.

## Exclusion-Manifest Cross-Check Before Planning (MANDATORY)

**Origin:** 2026-05-16 `.208 gate-block cascade. Smoking gun was
exp2091, retired in `.164 (44 milestones earlier) per
`ops/exclusion_manifest.yaml`:

```yaml
- experiment_id: 2091
  completed_milestone: "2026.05.164"
  reason: "gemini CLI consistent bail-out pattern ... 60+ retries
           / 2h wall time"
```

The `.208 planner re-proposed exp2091's scope ("Tier 1 CSL Grammar
Updates") as a fresh task. Three downstream Phase 4 Full Integration
Benchmark tasks declared `requires: exp2091.tier` — when exp2091 was
GATE_BLOCKed at activation by the exclusion-manifest enforcer in
`scripts/research_conductor.py`, those three tasks cascade-blocked.
Net: 4 of 13 `.208 tasks burned wall-time on a doomed retired
experiment-id chain that the manifest already classified as
permanently retired.

The existing **Failed-Experiment Rerun Discipline** (above) governs
*scope-similar* reruns at the planner-discipline layer. This rule
is one step tighter: it governs *experiment-id-matched* reruns at
the **per-id manifest level**, where the conductor will mechanically
GATE_BLOCK even if the planner respects the broader rerun discipline
in spirit.

**The rule.** Before drafting `research-roadmap-next.yaml`, the
planner MUST:

1. **Read `ops/exclusion_manifest.yaml` in full.** The file is
   short — every retired experiment_id with its retirement
   milestone and reason fits in a single read.
2. **For each task being drafted, check whether the task scope
   matches the scope of any retired experiment_id** by name,
   description, deliverable shape, or technique. Substring matching
   on the retired experiment's name/description is sufficient for a
   first-pass check (e.g., a draft titled "Tier 1 CSL Grammar
   Updates v2" must match against exp2091's "Tier 1 CSL Grammar
   Updates").
3. **If a scope-match is found, REFUSE to propose the task** unless
   the YAML includes BOTH:
   - A `prior_failures:` block per Failed-Experiment Rerun
     Discipline (names the retired exp_id, diagnosed root cause,
     what is different now), AND
   - An explicit `operator_override:` field with a one-line operator
     directive citing where the override was granted (operator
     message timestamp, known-issues.md entry, etc.).
4. **For any task that requires the output of a retired
   experiment** (declares `requires: expNNNN.<artifact>` where
   expNNNN is on the manifest), REFUSE outright — the requires
   chain is structurally dead because the conductor will GATE_BLOCK
   expNNNN at activation. No override path; rewrite the chain to
   not depend on retired output.

**How to apply (planner-side discipline).** When generating
`research-roadmap-next.yaml`:

- Grep `ops/exclusion_manifest.yaml` for retired experiment_ids
  and their reasons.
- For every draft task, ask: "does this task's scope match any
  retired experiment_id by name or deliverable shape?" If yes,
  document the answer in the YAML's task `prior_failures:` block
  per Failed-Experiment Rerun Discipline. If no `prior_failures:`
  entry satisfies the rerun-discipline criteria AND no
  `operator_override:` field cites a specific operator directive,
  drop the task and propose something else.
- For every draft task's `requires:` field, verify that no
  experiment_id referenced is on the exclusion manifest. If any
  is, rewrite the chain — don't propose dead dependencies.

**Auto-override for known-legit continuations (MANDATORY — 2026-05-29
operator directive).** The scope-matcher substring-matches a draft
task's name/scope against retired experiment names. This generates
*false positives* on routine forward work: the `.310 roadmap tripped
6 HARD violations, all legitimate — including the routine
`archive-v309-activate-v310` task (which scope-matches every prior
`archive-v*` task and can never be a doomed rerun). The operator's
chosen fix (over a linter carve-out) is: **keep the linter strict,
and have the planner auto-add `operator_override:` to known-legit
continuations** instead of dropping them.

When a draft task scope-matches a retired experiment_id BUT belongs
to one of the three classes below, the planner MUST add an
`operator_override:` string (citing the standing 2026-05-29
authorization + the false-matched exp ids + a one-line rationale)
rather than dropping the task:

1. **Routine milestone-transition tasks** — `archive-vN-activate-vN+1`,
   `capstone-vN`, `plan-milestone-*`. These always scope-match prior
   transition tasks and are structurally never doomed reruns.
2. **Active hardware-continuity tasks** (KV260 / GateMate / PolarFire)
   mandated by the Hardware-Task Continuity Discipline, until the
   board reaches its terminal state. They scope-match retired hardware
   exps but are required forward work.
3. **Versioned lineage continuations** (FR-11 vN+1, constrained-
   generation vN+1, etc.) that carry a STATED forward difference from
   the retired predecessor (new technique / corpus / gate). The
   override string MUST name the difference.

**GUARDRAIL (do not abuse the auto-override).** It applies ONLY to
the three false-positive classes above. It does NOT apply to a
genuine doomed rerun — same scope + same prior-failure verdict + no
new approach. Those must still be DROPPED per the Failed-Experiment
Rerun Discipline, or carry a real `prior_failures:` block with
`retire_if_same_verdict: true`. An `operator_override:` is an
assertion that the task is a legit continuation, NOT an escape hatch
for re-running a failed approach.

**Override string format** — non-empty, ≥10 chars, auditable:
`"2026-05-29 operator directive (standing): <class> — false-positive
scope-match vs <exp ids>; <one-line forward rationale>."`
The 6 overrides applied to the `.310 roadmap (commit `2cf7940f5`) are
the canonical examples.

**Agent routing for the archive/activate transition task: codex, NOT
Opus (MANDATORY — 2026-06-12 operator quota-conserve directive).** The
routine `archive-vN-activate-vN+1` task is mechanical (YAML archive +
next-milestone activation; the activation guard is a SCRIPT, not the
agent) and MUST run on `agent_type: codex`. Do NOT set
`requires_claude_verified: true` on it — that flag exempts the task from
the `CODEX_FORCE_EXPERIMENTS=1` coercion and needlessly burns ~1 Claude
Opus call per milestone (× ~10–30 milestones/day). Planner + outer-loop
pre-staging both emit the archive task as `agent_type: codex` with NO
`requires_claude_verified`. (Capstones already default to codex.
Planner/retro themselves STAY on Opus per the operator's deliberate
quality choice — this rule is ONLY about the mechanical transition task.)

**A single `operator_override:` clears BOTH guards (2026-05-29).** It
satisfies the milestone-activation exclusion-manifest guard AND the
per-task doomed-rerun pre-launch guard
(`research_conductor.py:research_step` → `ledger.is_doomed_rerun`). The
planner therefore does NOT need to add both `operator_override:` and
`prior_failures:` to a legit continuation — `operator_override:` alone
is sufficient. (`prior_failures:` remains the alternative satisfier for
genuine reruns that address a root cause.) This was discovered when the
`.310 tasks passed activation on `operator_override:` but were then
`DOOMED_RERUN_BLOCK`-skipped at launch; the doomed-rerun guard was
extended to honor `operator_override:` so the two guards are consistent.

**Mechanical enforcement (lives in `scripts/research_conductor.py` +
`scripts/exclusion_manifest_lint.py`).**
The conductor's `_ensure_exclusion_manifest_loaded` check GATE_BLOCKs
retired experiment_ids at activation time. `scripts/exclusion_manifest_lint.py`
(Layer 2, wired into `_activate_next_roadmap()`) scans the whole
planner-emitted YAML *before* activation and HARD-refuses the milestone
on any of 5 violation classes: `EXP_ID_RETIRED` (task id reuses a
retired id), `SCOPE_MATCHED_PRIOR_FAILURE` (scope-signature matches a
PAST ARTIFACT's failed verdict), `REQUIRES_RETIRED_EXP` (requires-chain
references a retired id), `WRONG_MECHANISM_PRECONDITION` (a
CLAUDE.md-retired precondition pattern), and
`BLOCKED_PATTERN_MATCHED` (2026-07-01: task title/prompt matches a
free-text `blocked_patterns:` string on a `retired_extras` entry,
regardless of the task's own id or artifact history — added after the
`.469 FoVer-in-domain incident, where a same-session retraction landed
in known-issues.md 8 minutes before the planner ran and still got
ignored, because neither of the first two classes could catch a
brand-new task id with no prior artifact). A task with an `operator_override:`
or a valid `prior_failures:` block clears the applicable classes; see
`ops/known-issues.md` "MECHANICAL FIX 2026-07-01" for the incident and
fix detail.

**Why this is in CLAUDE.md, not just in `ops/exclusion_manifest.yaml`.**
The manifest is the mechanical enforcement target; the planner
reads CLAUDE.md as required input on every plan generation. The
`.208 incident demonstrated that the manifest alone is necessary
but not sufficient: the planner kept proposing retired-id scope
because nothing in its required inputs told it to check the
manifest first. Putting the rule here forces the cross-check at
*design time* — the cheap layer — instead of at activation time
where the cascade has already burned wall-clock for the
dependent tasks.

**Cross-references:**
- `ops/exclusion_manifest.yaml` — the manifest itself; current
  retired experiment_ids include exp2091 (gemini CLI bail-out),
  among others
- **Failed-Experiment Rerun Discipline** (above) — the broader
  scope-matched rerun rule; this Exclusion-Manifest rule is its
  per-id specialization with `operator_override:` as the only
  bypass
- `scripts/research_conductor.py:_ensure_exclusion_manifest_loaded`
  — the mechanical enforcer that GATE_BLOCKs at activation
- `.208 cascade incident (2026-05-16) — three Phase 4 Full
  Integration Benchmark tasks blocked via the exp2091 requires-
  chain; documented in `ops/known-issues.md` and the .208 retro

## Operational Principles

- **Meta-reflection:** After milestones, evaluate HOW work was executed, not just WHAT was produced. Feed operational improvements back into the process.
- **Continuous improvement:** Domain (verification accuracy), process (experiment speed), and strategy (research direction) all improve together as a unified self-learning system.
- **The energy function is ground truth.** It cannot be gamed. This is the invariant across all three phases.
- **No-doomed-rerun discipline:** see "Failed-Experiment Rerun Discipline" above.

## Pre-Staged Roadmap Convention (MANDATORY)

**Origin:** 2026-05-08 operator-trust directive: "I tend to trust your
plan over codex' when you've made one." Filed after a 9-prompt Deep
Think synthesis produced a `.120 roadmap with substantially more
context (4 novel paper-v6 contributions, 3 documented rule-outs,
operator-prioritization tier structure) than codex's planner could
recover from carry-forward bias alone.

**The rule.** When `research-roadmap-next.yaml` exists with a
`milestone:` field that matches the EXPECTED next milestone (current+1),
the conductor's `_plan_next_milestone()` function MUST preserve it and
skip the planner entirely. The activation guard's existing YAML
validation handles structural correctness regardless of who drafted
the file.

If the file exists but with a STALE milestone (mismatch), it's treated
as leftover-from-prior-cycle and the planner runs normally to
overwrite. Empty/unreadable file → planner runs.

**Authoring protocol** (operator or outer-loop Claude drafting a
roadmap ahead of milestone close):

1. Write the YAML to `research-roadmap-next.yaml` with `milestone:`
   field set to current+1 (e.g., if active is 2026.04.119, draft has
   `milestone: 2026.04.120`).
2. Include the standard activation manifest task as exp(N+1) where
   N is the last task ID in the prior milestone.
3. Follow Codex-Default discipline (all tasks `agent_type: codex`
   unless `requires_claude: true` is justified per the positive
   criterion in this CLAUDE.md).
4. Follow Verdict Terminal-Prefix Discipline (every task's prompt
   spec'ing `honest_verdict` MUST start with `complete:`/`complete_`/
   `success:`/`success_`/`passed:`/`passed_`/`shipped:`/`shipped_`).
5. Follow Failed-Experiment Rerun Discipline (every task that
   re-proposes scope similar to a prior failure MUST include
   `prior_failures:` block).
6. Commit with `[outer-loop]` or `[operator]` prefix to make the
   provenance auditable.

**Mechanical enforcement (lives in `scripts/research_conductor.py`):**
`_plan_next_milestone()` calls `_expected_next_milestone(current)` and
compares against the existing draft's `milestone:` field. Match →
preserve. Mismatch → planner runs. The check is structural; no
content inspection beyond the milestone-version field.

**Why this is in CLAUDE.md, not just in code.** Same defense-in-depth
pattern as Codex-Default and Verdict Terminal-Prefix Discipline: the
planner reads CLAUDE.md as required input, and the conductor's
mechanical check enforces if the planner respects the convention at
design time. Putting the rule here also documents the operator-trust
contract explicitly.

**When to author a pre-staged roadmap (operator/outer-loop discipline):**

- After a Deep Think round that materially shifts priorities (e.g.,
  the 2026-05-08 9-prompt sweep that produced the .120 roadmap).
- When known-issues.md priorities have grown faster than the planner's
  carry-forward window can absorb.
- When operator has higher context than the conductor's planner can
  reach via its standard input set (CLAUDE.md, prd.md, status.md, etc.).
- When mechanical safety nets (activation guard, verdict prefix
  discipline) need a specific task structure the planner might not
  produce.

**When to LET the planner run** (don't pre-stage):

- Routine milestones with no special context.
- When operator wants to test whether the planner converges to the
  same priorities the operator would have chosen.
- When known-issues.md is the source of truth and the planner's
  carry-forward bias isn't a concern for the current milestone.

## Calendar-Month Prefix Rollover (MANDATORY)

> **STATUS (2026-05-29): MECHANICALLY ENFORCED — prose is reference-only.**
> `_expected_next_milestone()` derives the prefix from today's UTC date
> automatically; the operator only increments the trailing index. No human
> decision points. Read on only if debugging milestone numbering.

**Origin:** 2026-05-08 operator question — "now that we are in May,
when will the milestones start numbering with 2026.05. as a prefix
instead of 2026.04. which was for april?" Honest answer: nobody had
wired the rollover; the `2026.04` prefix had been frozen since the
milestone series began. Fixed 2026-05-08 by updating
`_expected_next_milestone()` to derive the prefix from today's UTC
date rather than preserving the prior milestone's prefix.

**The rule.** Milestone identifiers have format `YYYY.MM.NNN`:

- **NNN**: a global sequence ID that increments by exactly 1 per
  milestone (e.g., 119 → 120 → 121 → ...). Never restarts.
- **YYYY.MM**: the calendar month of TODAY (UTC) at the moment the
  milestone is planned, not the prior milestone's month. So a milestone
  planned 2026-05-08 carries prefix `2026.05` regardless of whether
  the prior milestone was `2026.04.NNN` or `2026.05.NNN-1`.

**Examples:**

```
prior=2026.04.119, today=2026-05-08 → next=2026.05.120  (May rollover)
prior=2026.04.119, today=2026-04-30 → next=2026.04.120  (still April)
prior=2026.05.123, today=2026-05-15 → next=2026.05.124  (no rollover)
prior=2026.05.130, today=2026-06-01 → next=2026.06.131  (June rollover)
```

**Why this matters:**

- Future paper drafts that cite milestones by `YYYY.MM.NNN` get the
  correct planning month for free.
- `ops/changelog.md` entries already include the calendar date; the
  milestone prefix should match for grep-ability.
- The existing milestones under `2026.04.NNN` (.99 through .119) stay
  as-is — they really were planned in April. No retroactive renaming.

**How to apply (operator/outer-loop authoring a pre-staged roadmap):**

When drafting `research-roadmap-next.yaml` per the Pre-Staged Roadmap
Convention, set the `milestone:` field using today's UTC date as the
prefix and current+1 as the trailing index. Don't copy the prefix from
the prior milestone if today is in a different month.

**How to apply (planner-side):** the planner reads CLAUDE.md as input.
When generating a milestone YAML, the `milestone:` field MUST match
the result of `_expected_next_milestone(current)` evaluated at planner
runtime. The activation guard verifies this match before activating.

**Mechanical enforcement.** `scripts/research_conductor.py:
_expected_next_milestone()` returns the calendar-month-rolled expected
next milestone. The pre-staged-roadmap check in `_plan_next_milestone`
compares the existing draft's `milestone:` field against this
expectation. Stale prefixes (e.g., `2026.04.120` drafted in May)
trigger planner re-run.

## Codex-Default for Experiments v2 (MANDATORY — 2026-06-10, supersedes Gemini-Default)

**Origin:** 2026-06-10 operator directive after gemini-cli stalled GLOBALLY
(the `.370 planner timed out at 1201s; the archive task failed 3x at 600s
silence; a trivial `gemini -p "Reply with exactly: OK"` smoke test hung):

> "Let's make sure we're using codex by default anywhere that gemini would
>  get used as that still seems to be a problem."

This is the second gemini reliability incident in two weeks (cf. the `.333
quota-crash wipeout that produced ZERO artifacts from 14 tasks). The operator
has now made codex the STANDING default — not an emergency fallback that
auto-reverts when gemini recovers. Re-enabling gemini requires an explicit
operator directive + a fresh smoke verification.

**The rule.** When planning a new milestone, **all experiment tasks default to
`agent_type: codex` and `model: gpt-5.5`**. Do NOT emit `agent_type: gemini`.
The narrow exceptions are unchanged in structure from the prior defaults:

1. **Planner / retro / adversarial audits** — stay on Claude Opus 4.8 via
   `AGENT_TYPE_PLANNER` / `AGENT_TYPE_RETRO` env + explicit `--model claude`
   audit invocations (operator directives 2026-05-30 and 2026-06-08).
2. **Cross-file refactors with deep tool choreography** — `agent_type: claude`
   ONLY with the operator-only `requires_claude_verified: true` flag (the
   planner-emitted `requires_claude` is an ABUSED signal since `.322-`.325 and
   is IGNORED by the runtime coercion).
3. **Gemini tasks** — none, until the operator re-verifies gemini. A task that
   genuinely needs gemini (1M-context long-corpus synthesis) waits or chunks
   for codex; if truly blocking, escalate to the operator rather than emitting
   gemini.

**Mechanical enforcement (lives in `scripts/research_conductor.py` +
systemd drop-in `30-codex-fallback-20260610.conf`):**

- `CODEX_FORCE_EXPERIMENTS=1` (standing) coerces `claude → codex` (ignoring
  the abused `requires_claude`; only `requires_claude_verified: true`
  bypasses) AND `gemini → codex` (only `requires_gemini_verified: true`
  bypasses — operator-only).
- `GEMINI_FORCE_EXPERIMENTS=0` (standing) disables the old back-flip coercion.
- `AGENT_TYPE=codex`, `AGENT_MODEL=gpt-5.5` are the conductor's process env.
- The planner prompt's AGENT ROUTING section now reads CODEX-DEFAULT.
- The standalone audit scripts (`verifier_authenticity_audit.py`,
  `pages_adversarial_audit.py`) default to `--model claude` (Opus); gemini is
  opt-in only.

**Known codex risk (acknowledged, monitored):** the 2026-05-29 long-prompt
hang/HTTP-400 incident benched codex once before. codex-cli 0.137.0 ran ~150
GAP-4 calls + the full post-flip `.370 task stream without a hang (2026-06-10).
If codex hangs recur on the conductor's long prompts, the fallback is CLAUDE
(not gemini) for the affected task class, via `requires_claude_verified`.

**Calibration table:** the agent-fungible task categories in the historical
Gemini-Default table below now read `codex` in the default column, per the
same agent-fungibility reasoning that moved them codex→gemini in May.

---

### Historical: Gemini-Default rule (2026-05-20 — 2026-06-10; preserved per never-prune)

> **STATUS (2026-06-10): SUPERSEDED by Codex-Default v2 above** after the
> gemini-cli global-stall incident. Prose preserved unchanged below.

## Gemini-Default for Experiments (MANDATORY — Quota Preservation)

**Origin:** 2026-05-20 operator directive after seeing every `.259 task
declare `agent_type: codex` in the YAML while runtime coercion
(`GEMINI_FORCE_EXPERIMENTS=1`) rewrote every one to gemini:

> "why are we still using codex? I said to switch to gemini"

The previous "Codex-Default for Experiments" rule (preserved below for
historical context) ran a two-layer defense: planner emits codex; runtime
coerces codex → gemini. That worked operationally but made the YAML
deceptive — the declared `agent_type` did not match what actually ran.
This rule flips the planner-emit default to gemini directly so the YAML
is honest. Runtime coercion env vars (`GEMINI_FORCE_EXPERIMENTS=1`,
`CODEX_FORCE_EXPERIMENTS=1`) are kept as emergency operator overrides for
quota windows where the YAML choice and the available-quota choice
disagree.

**The rule.** When planning a new milestone, **all experiment tasks
default to `agent_type: gemini` and `model: gemini-3.1-pro-preview`**
unless the task meets the positive criterion for `agent_type: claude`
(below) or genuinely needs codex (see exceptions). The exceptions are
narrow:

1. **Planner tasks** — the planner itself stays `agent_type: claude`
   via `AGENT_TYPE_PLANNER=claude` env. Planning is reasoning-heavy
   and runs ~1× per milestone.
2. **Cross-file refactors with deep tool use** — tasks that need
   Claude's tool-use ergonomics (multi-file Edit + Read + Bash
   choreography). When the planner picks Claude here, it MUST set
   `requires_claude: true` on the task as a documented bypass. Without
   that flag, runtime coercion treats Claude as the wrong default and
   may rewrite it.
3. **Operationally critical fixes during outer-loop self-healing** — if
   Claude is the outer loop applying structural fixes mid-session, that
   work continues (the user is in the room). This rule governs
   *autonomous* loop work, not interactive operator turns.
4. **Tasks that genuinely need codex** — gpt-5.5-specific tool
   ergonomics, or operator explicitly directed codex for this task.
   Set `agent_type: codex` + `model: gpt-5.5` + `requires_codex: true`
   so the `GEMINI_FORCE_EXPERIMENTS=1` runtime coercion does not
   silently rewrite it.

**Mechanical enforcement (lives in `scripts/research_conductor.py`).**
The conductor coerces per-task agent_type at task-launch time based on
operator quota directives:

- `GEMINI_FORCE_EXPERIMENTS=1` (typical) → coerces `codex → gemini`
  unless task has `requires_codex: true`. Mostly a no-op under the new
  Gemini-Default planner rule, but kept so a misbehaved planner emit
  still routes correctly.
- `CODEX_FORCE_EXPERIMENTS=1` (quota-emergency flip) → coerces
  `claude → codex` unless task has `requires_claude: true`, AND coerces
  `gemini → codex` unless task has `requires_gemini: true`. Used when
  gemini quota is constrained and codex is available; flips the routing
  without a roadmap edit.

Logs a WARNING per coercion. Planner and retro paths are NOT affected
— those use `AGENT_TYPE_PLANNER` and `AGENT_TYPE_RETRO` env overrides
directly and bypass the coercion.

**How to apply (planner-side discipline).** When you generate
`research-roadmap-next.yaml`:

- **Default** every experiment task to `agent_type: gemini`, `model:
  gemini-3.1-pro-preview`.
- **Justify** any task you mark `agent_type: claude` with a one-line
  comment AND set `requires_claude: true`. If you can't articulate why
  gemini would fail for the task, gemini is the correct choice. The
  same positive criterion applies as before (see below — the bar is
  about reasoning depth and tool choreography, not about which agent is
  the new default).
- **Justify** any task you mark `agent_type: codex` with a one-line
  comment AND set `requires_codex: true`. The bar is "this task
  genuinely needs gpt-5.5" or "operator explicitly directed codex" —
  not "I think codex is faster."
- **Never use claude** for routine retros, doc passes, schema
  validation, simple analysis tasks, or experiments that primarily
  read/run scripts. These all work fine on gemini.
- **Audit pass before emitting**: the YAML should be roughly
  `~11/13 gemini + 1-2 claude + 0-1 codex` for a typical 13-task
  milestone. If more than 2 of 13 tasks are claude, re-evaluate each
  one against the positive criterion. If any task is codex without
  `requires_codex: true`, fix it.

**Positive criterion for `requires_claude: true` (must meet ALL three).**
Originally written for the Codex-Default era — the bar is about
*reasoning depth + tool choreography*, not about which agent is the new
default. The criterion below substitutes "gemini" for the previous
"codex" wherever the comparison was about default-vs-Claude capability;
the structural test is unchanged.

```
requires_claude: true is justified only when the task meets ALL of:

  1. GEMINI HAS DEMONSTRABLY FAILED at this specific task category in
     a prior milestone (cite the experiment ID + verdict in the YAML
     comment), OR gemini's known capability profile makes failure
     plausible for THIS specific task structure (not just "important
     work" in general).

  2. THE WORK REQUIRES MULTI-FILE TOOL CHOREOGRAPHY: 5+ files touched
     across Edit + Read + Bash in a single session, where the agent
     must hold cross-file context to make decisions. Single-file edits,
     mechanical LaTeX integration of pre-drafted prose, GPU training
     supervision, numerical analysis with matplotlib output, and
     deliverable-shaped retros do NOT meet this bar.

  3. THE REASONING IS MULTI-STEP IN A WAY NUMERICAL/MECHANICAL
     CHECKING CANNOT SUBSTITUTE FOR. If the task's success can be
     evaluated by a deterministic gate (spectral norm bounded, AUROC
     above threshold, schema validation passing, test suite green),
     then a weaker agent that produces the same gate-passing artifact
     is equivalent — gemini is the correct choice. Claude's edge is
     handling open-ended judgment under ambiguity, not running scripts
     that have known correctness criteria.
```

**Positive criterion for `requires_codex: true` (must meet ANY of).**
2026-05-20 addition. The codex bypass exists for narrow cases:

```
requires_codex: true is justified when the task meets ANY of:

  1. THE OPERATOR EXPLICITLY DIRECTED codex for this task (cite the
     directive in the YAML comment — operator message timestamp or
     known-issues.md entry).

  2. THE TASK NEEDS gpt-5.5-SPECIFIC TOOL ERGONOMICS that gemini-3.1-
     pro-preview cannot match (rare; codex was historically default,
     so most legacy task categories work fine on gemini now).

  3. GEMINI QUOTA IS EXHAUSTED and the operator has set
     `CODEX_FORCE_EXPERIMENTS=1` to flip the runtime coercion. In this
     case `requires_codex: true` is unnecessary (the coercion will
     route there automatically) but is fine as a redundant signal.
```

**What neither flag is for.** None of these alone, nor in combination,
justify a non-gemini choice:

- `priority: critical` — task importance is orthogonal to agent choice
- `priority: high` — same
- `publication-blocking` — say so in the comment, not via agent choice
- "high-stakes" — the artifact's review is what protects against stakes,
  not the agent that produced it
- "should be careful" / "needs accuracy" — gemini is also careful and
  accurate within its capability envelope; the question is whether the
  work exceeds that envelope
- GPU training duration — long training runs are GPU-cost-bound, not
  agent-cost-bound; gemini monitors training fine
- Multi-step instructions — gemini handles 50+ turn tasks competently;
  Claude is only needed when the steps require cross-context reasoning
  beyond what the prompt + tool outputs can encode

**Calibration table** (populate as evidence accumulates; entries
updated to reflect the gemini-default as of 2026-05-20):

| Task category | Default agent | Why |
|--------------|---------------|-----|
| Numerical audits + matplotlib output | gemini | Mechanical with deterministic correctness criteria |
| LaTeX integration of pre-drafted prose | gemini | Mechanical splice, no original judgment |
| GPU training supervision | gemini | Compute-bound; agent role is monitoring |
| Mechanical gate evaluation | gemini | Numerical thresholds substitute for reasoning |
| Constrained design work (candidates pre-proposed) | gemini | Pattern-matching from existing candidate set |
| Routine retrospectives | gemini | Templated structure |
| Schema validation / doc reconciliation | gemini | Mechanical |
| Failure-ledger pattern detection across milestone history | gemini | 1M-token-context advantage; feed entire research-complete.yaml |
| Architecture coherence audits (Phase-3..7) | gemini | Long-context synthesis |
| Multi-paper literature synthesis (3-5 papers full text) | gemini | Long-context advantage |
| Open-ended paper PROSE writing (first draft, novel framing) | claude | High-judgment under ambiguity |
| Cross-file refactors with 5+ file Edit/Read/Bash | claude | Multi-file tool choreography (the original spec exception) |
| Deep theoretical framing (e.g., new Spera/CAF-class formalism) | claude | Multi-step reasoning + cross-context synthesis |
| Operator-directed codex tasks | codex | Per operator directive only |

**Why this is in CLAUDE.md, not just in user-memory.** User-memory is
unreachable from the planner sub-process. Mechanical conductor
enforcement catches the failure post-plan but wastes planner time
emitting plans that get coerced. Putting the rule in CLAUDE.md (which
the planner reads as required input) means the planner respects it at
design time AND the conductor coerces if it slips. Defense in depth.

---

### Historical: Codex-Default rule (preserved per "Never remove existing content" rule)

The 2026-05-02 / 2026-05-03 Codex-Default rule was the predecessor to
this Gemini-Default rule. It was filed when Anthropic Claude quota was
the binding constraint and codex (gpt-5.5) was the open path. The rule
specified `agent_type: codex` + `model: gpt-5.5` as the default for all
experiment tasks, with the same `requires_claude: true` positive
criterion as above (substituting codex for gemini in the comparison).
The rule was operationally enforced via the same conductor coercion
mechanism (`CODEX_FORCE_EXPERIMENTS=1` rewriting `claude → codex`).

It was superseded 2026-05-20 because:

1. Gemini-CLI returned to viability (the `.220-`.226 crash storm
   ended; gemini-3.1-pro-preview now stable).
2. Codex quota dynamics inverted (gemini quota window now broader).
3. The runtime coercion (`GEMINI_FORCE_EXPERIMENTS=1` rewriting
   `codex → gemini`) made the YAML deceptive — declared codex tasks
   ran as gemini. Operator surfaced the surface-vs-runtime mismatch:
   "why are we still using codex? I said to switch to gemini."

The Codex-Default era's calibration table entries (which originally
read "codex" in the default-agent column for all mechanical tasks)
are now `gemini` per the same reasoning that put them in the default
column originally — those tasks are agent-fungible at the default
tier. The `requires_claude` positive criterion is unchanged structurally;
only the comparator agent (gemini, not codex) has been swapped.

## Operator-Only External Publication (MANDATORY)

**Origin:** 2026-05-19 operator directive: "I need to review any paper
before it is submitted." Filed after `.243 exp2527 (arXiv Submission
Package Prep) was queued and paper-v6 reached `arxiv_ready=True` per
the conductor's own gate logic. The current exp2527 task is correctly
prep-only (produces a checklist for the operator), but no MANDATORY
rule yet prevents a future planner from queuing an autonomous arXiv
push, OpenReview upload, GitHub PR creation against a public-fork,
HuggingFace model release announcement, or any equivalent
external-publication action without operator review.

**The rule.** **Submission of papers, model releases, or other
externally-visible artifacts to public venues is OPERATOR-ONLY.**
Carnot tasks may:

- Prepare arXiv submission packages (LaTeX compile, abstract
  word-count check, author list extraction, CCS concept tagging,
  category recommendation)
- Generate operator-action checklists
- Run pre-submission integrity audits and gate checks
- Write LaTeX files, update bibliographies, regenerate figures
- Push to private remotes (the project's own GitHub mirror,
  HuggingFace organization, etc.) — these are NOT external
  publication, they are internal mirroring
- Generate model card text, release notes, or announcement drafts

Carnot tasks **MUST NOT** perform any of:

- `arxiv upload` / `arxiv submit` / equivalent CLI to arxiv.org
- OpenReview submission API calls
- HuggingFace model card publish-button-equivalent that flips a
  model from private/draft to public
- GitHub release creation (`gh release create`, equivalent web API)
- Social-media or blog posts announcing the project
- Email or other outbound communication to reviewers, editors,
  conference chairs, or external researchers
- Any action that creates a publicly-citable artifact bearing the
  Carnot name without prior operator review

**Why this is operator-only.** External publication is irreversible
in practice. arXiv submissions can be withdrawn but not deleted; the
record persists. Reviewer or editor outreach commits the project to
a tone, framing, and set of claims that may need to be re-litigated
if wrong. The Carnot project's credibility — already validated by
the discipline machinery (adversarial-verify, 5-seed replication,
exp2468 audit) — would be undermined by a single autonomous
submission of unreviewed material. The operator is the trust-anchor
for what the project says publicly.

**How to apply (planner-side discipline).** When generating
`research-roadmap-next.yaml`:

- Submission-prep tasks (LaTeX compile, package assembly, checklist
  generation) are allowed and encouraged. Frame them as "prepare
  package for operator review."
- The task's `REQUIRED ARTIFACT FIELDS` MUST include
  `submission_package_ready: bool` with the principle "True if the
  package is ready for OPERATOR to submit; the task itself never
  submits."
- The task's prompt MUST NOT include any step like "run `arxiv
  submit`" or equivalent. Concrete steps stop at "produce the
  package + checklist."
- If a task's intent is genuinely operator-side action assistance,
  it goes in the checklist artifact, not as a step the agent
  executes.

**How to apply (agent-side discipline, for any agent running a Carnot
task).** When executing a task that touches paper / model / release
artifacts:

- Read the task prompt. If any step says or implies "submit to
  arxiv" / "upload to OpenReview" / "publish the model card" /
  "create the GitHub release," **DO NOT execute that step**.
  Instead, write `honest_verdict: blocked_operator_only_submission`
  and produce the prepared-but-not-submitted artifact alongside a
  prominent note in the deliverable that the submission step is
  reserved for the operator.
- This applies even if the agent's environment has the credentials
  to execute the action (arxiv API token, HuggingFace write token,
  `gh auth status` logged in, etc.). Capability does not imply
  authorization.

**Mechanical enforcement (future, lives in
`scripts/research_conductor.py` activation guard).** The activation
guard SHOULD scan task prompts for terms like `arxiv submit`,
`arxiv upload`, `arxiv-submit`, `OpenReview submission`, `gh release
create`, and refuse activation of any milestone containing such a
task without an explicit `operator_override:` field citing where the
operator authorized the submission. Until that ships, this rule is
honor-discipline at the planner+agent layer alone.

**Exceptions.** A task may *include* the operator submitting if and
only if:

1. The task's prompt explicitly says "the operator will perform step
   N; this task ends at step N-1," and
2. The task's `submission_package_ready` field is the terminal
   acceptance gate (true → operator-ready, false → still-in-prep),
   and
3. The task does NOT have the credentials to perform step N
   regardless (no arxiv token in env, etc. — defense in depth).

**Cross-references:**

- 2026-05-19 operator directive (origin)
- `feedback_publication_holds_until_phase4_pivot.md` (memory) —
  the prior pre-publication hold framework; this rule extends it
  to cover the post-hold submission action itself
- `feedback_paper_integrity_audit.md` (memory) — the pre-submission
  audit discipline; submission still requires operator after audit
  passes
- CLAUDE.md "Decentralization-Respecting Design Constraints" Rule 3
  (mandatory mirroring) — private mirrors are fine; the
  external-publication is what this rule guards
- CLAUDE.md "Doing tasks" section (top of file): "For actions that
  are hard to reverse, affect shared systems beyond your local
  environment, or could otherwise be risky or destructive, check
  with the user before proceeding." — this rule operationalizes
  that principle for the specific case of external publication.

## Never Stash — Always Commit-First (MANDATORY)

**Origin:** 2026-05-04 14:30Z incident. The user issued `pull`. The
working tree had 11 modified + 3 untracked files from in-flight
conductor work. The reflexive git-developer move (`git stash` →
`git pull` → `git stash pop`) created a 1-2 second window where the
working tree was reverted to HEAD. By luck no codex subprocess was
mid-write during that window, but the operator had previously issued
the durable directive *"always committing and never reverting so that
we fail forward and fix any problems rather than lose transient
assets"* (see `feedback_outer_loop_role.md` memory). The git-stash
reflex is so ingrained from years of normal git use that it bypassed
the abstract "never revert" memory directive.

**The rule.** When the working tree is dirty and a pull is needed,
**NEVER use `git stash`.** Commit-first instead:

```bash
git add -A && git commit --no-verify \
  -m "[outer-loop] preserve transient state before pull"
git pull --rebase
```

Or use the helper script:

```bash
scripts/safe-pull.sh
```

**Why this matters.** While the conductor is running, codex / claude
subprocesses may hold open file handles to artifacts in `results/`,
`docs/`, etc. A `git stash` reverts those files in the working tree
between the stash and the pop. If a subprocess writes during that
window, the write either:

- Goes to a file that no longer exists in the working tree (silent
  data loss when stash pop overwrites with the stashed version), OR
- Creates a new file that conflicts with the stash pop.

Commit-first never has this risk because the working tree is never
reverted; the in-flight changes are simply preserved as a commit.

**Operational equivalence.** A "[outer-loop] preserve transient
state" commit costs nothing — the conductor already commits transient
state on every iteration. An extra outer-loop commit is identical in
content; it just lands a few seconds earlier than the conductor's
auto-commit would have.

**When this rule does NOT apply.**

- Truly clean working tree (no modifications) — pull directly.
- Genuinely separate operator work that should not be committed yet
  (e.g., debugging an in-flight branch). In that case use a worktree
  (`git worktree add`) instead of stash, since worktrees don't
  revert the original working tree.

**How to apply.** If `git pull` returns "cannot pull with rebase: You
have unstaged changes" — DO NOT REACH FOR STASH. Commit-first. If
unsure whether the dirty state is conductor work or operator work,
default to commit-first since the cost is zero and the risk of
data-loss is non-zero.

## Adversarial Artifact Verification + Sample-Size Rigor (MANDATORY)

**Origin:** 2026-05-12 operator directive after `.144 exp1851 (NLA
probe) was caught fabricating gate-passing artifacts (TPR=1.0 in
3.4s wall time on a 30B+ GGUF model — impossible). Subsequent sweep
of `.131-`.149 artifacts found ~24% with at least one adversarial
flag (tautology, sign anomaly, fabrication, methodology gap).
Operator: "we need an adversarial-verify pass on each artifact and
going forward that needs to be part of our scientific rigor. Always
cross-check surprising results. We also need to pay attention to
our sample sizes to make sure they are statistically significant
to support any claims we make."

**The rule.** Every experiment artifact MUST pass adversarial
verification before its claims can be cited in paper-v6,
research-program.md, or any forward-facing document. The verifier
lives at `scripts/adversarial_verify.py` and detects:

1. **TAUTOLOGY** — two distinct numerical metrics agreeing to >5
   significant figures (floating-point bit-identity between
   conceptually-different quantities is more likely a bug than a
   finding).
2. **IMPLAUSIBLE_PERFECT** — TPR/accuracy=1.0, error=0.0 on small
   sample sizes (real classifiers exhibit non-zero error).
3. **SIGN_ANOMALY** — optimization claiming "minimization" but
   final_value > initial (or vice versa for maximization).
4. **DURATION_TOO_SHORT** — artifact references compute-bound
   markers (GGUF, CUDA, torch.cuda) but duration_s < 60. Loading
   and running a real model takes minutes, not seconds.
5. **SAMPLE_SIZE_BELOW_CLAIM** — distributional claims (KL, KS,
   mean delta) with N below statistical threshold for the substrate
   size. Heuristic: distributional claims at n_spins=N need at least
   max(1000, 10·N) samples; n_spins>=64 needs at least 10k.
6. **GATE_PASSED_WITHOUT_DATA** — acceptance_gate_passed=true but
   key metric fields referenced in the gate are null, missing, or
   zero.
7. **METHODOLOGY_MISSING** — compute-bound artifact lacks
   model_specs / random_seed / reproducibility_checksum.

**How to apply (agent-side discipline).** When writing a results
artifact:

- Set `random_seed: <int>` and `reproducibility_checksum: <hash>`
  on any compute-bound experiment.
- Include `model_specs` / `target_model` listing the actual model(s)
  invoked.
- For distributional claims, write `n_samples` AND justify in the
  artifact why N is sufficient (e.g., "n_samples=10000 chosen so
  that std(empirical_KL) < 0.005").
- If a metric is exactly 0.0 or 1.0, add a `methodology_note`
  field explaining why (e.g., "TPR=1.0 because the test corpus is
  the training set; this is a sanity-check baseline, not a
  capability claim").
- For optimization tasks, if energy/loss INCREASES, write
  `optimization_note` explaining whether this is a real finding
  (e.g., "the bridge fails to minimize on this benchmark — paper-v6
  §6 limitation") or a bug to fix.

**How to apply (planner-side discipline).** When proposing tasks:

- Sample sizes MUST be specified in REQUIRED ARTIFACT FIELDS for
  any task making a distributional or statistical claim. Default
  too low (100 samples on n=128 is a known anti-pattern from exp1850).
- Falsifiable gates MUST be paired with sample-size budgets that
  make the gate statistically meaningful. A gate of "KL < 0.05"
  with 100 samples is gate-by-noise; require 10k+.
- Optimization tasks MUST specify direction (minimize / maximize)
  AND a sign-anomaly-acceptance plan (does increasing the metric
  invalidate the gate, or is it a documented finding?).

**Mechanical enforcement (post-task).** The conductor SHOULD run
`adversarial_verify.py` on every newly-landed artifact in
`research_step()` after the test phase. If the verify pass returns
CRITICAL flags:

- Write a `flagged_adversarial: true` field on the artifact
- Append the flag details to a `corrigendum_pending` field
- Do NOT auto-retire — the task ran, the data is preserved; the
  flag is a signal for human/outer-loop review

**Fabrication gate (MANDATORY — 2026-05-30 operator directive).** A CRITICAL
adversarial flag now MECHANICALLY downgrades the conductor verdict: the task is
logged `FLAGGED` (not `OK`) in `_log_experiment_completion`
(`scripts/research_conductor.py`), so a fabricated result (e.g. exp3397:
`duration_s=2.06` declaring a live 35B GGUF, `inference_substrate=sota_gguf_mock`,
`auroc=1.0`) does NOT count as a clean milestone success. `pick_next_task` treats
`FLAGGED` as completed-but-quarantined (no wasteful re-run, but not a clean
success). **Capstone, evidence-table, paper-v6, and any headline-aggregation task
MUST skip artifacts carrying `flagged_adversarial: true`** — never aggregate a
flagged artifact's numbers into a milestone result or a forward-facing claim.
This was added after exp3397/exp3405 (both `flagged_adversarial` per the
2026-05-30 corrigendum in `ops/known-issues.md`) logged clean OK and reached the
record before being caught manually.

**False-negative trap — FALSE_NEGATIVE_RISK (MANDATORY — 2026-05-31).** A
NULL/negative claim ("X does not beat Y", "selection premise refuted") is NOT
a finding unless a positive control passed. `adversarial_verify.py` now emits
`FALSE_NEGATIVE_RISK` (warn) when an artifact's `honest_verdict` is a null
claim AND any of: (1) a `*flip*count*` / `n_changed` field == 0 (the method
never changed a single output → cannot distinguish "method fails" from "no
headroom"); (2) an oracle/optimal upper bound that does NOT exceed the baseline
(the corpus has no selectable headroom, so no method could win — the null is
uninformative); (3) the experiment's own `*non_degenerate*` / `*g2*` /
`*headroom*` gate self-reported False. Origin: exp3507 reported the
process-energy reranker "does not beat self-consistency" with `flip_count==0`
and `optimal==SC==0.653` — a degenerate test, not evidence the method fails.
**Before propagating any null result to a forward-facing claim, run a positive
control (a corpus where oracle>baseline and flips>0).**

**Reading-Results Discipline (MANDATORY — 2026-05-31).** Before citing ANY
number from a result artifact as a conclusion, read it via
`python3 scripts/summarize_artifact.py <id|glob|--recent N>` — never eyeball
the raw JSON. The tool forces a fixed reading order: (1) `honest_verdict`,
(2) `flagged_adversarial` stamped AND a LIVE adversarial re-check (catches the
gap where a critical-flaggable artifact was never stamped), (3) every
`acceptance_gate_*` self-report (a FAILED gate overrides a celebratory
verdict), (4) `duration_s` + `inference_substrate` plausibility floor, (5)
headline metrics last, annotated with any flag touching them, and a prominent
FALSE_NEGATIVE_RISK section. This exists because three artifact misreads
occurred in one session (a degenerate `flip_count=0` read as a real verdict; a
stale log line read as live looping; identifier/seed TAUTOLOGY false-positives
read as a coverage gap). Exit code: 2 if a live critical flag, 1 if any warn.

**TAUTOLOGY excludes identifiers/seeds (2026-05-31).** The TAUTOLOGY check no
longer compares identifier/seed/metadata fields (`experiment_id`, `experiment`,
`random_seed`, `*_seed`, `*_id`, `milestone`, etc.). Seeding the RNG off the
experiment number is good reproducibility practice; `experiment_id == experiment
== random_seed == 3506` is structural, not a coincidence between two distinct
measurements. This removed the false positives that made exp3505/3506/3496/3481
look like a conductor coverage gap when the gate was in fact correct.

CRITICAL flags do not block downstream gates from firing; the
gate-check layer is upstream of the verify layer. The verify
output is for paper-v6 disclosure discipline + future-task
prior_failures tracking + the fabrication gate above.

**Cross-check surprising results.** When an experiment produces a
result outside the expected range (e.g., a TPR lift > 4-5x the
peer literature, a gate failure that contradicts a published
result), the artifact MUST include either:

- A successful replication artifact (n>1 seeds, same direction)
- OR a `surprising_result_acknowledgment` field documenting that
  the result is preliminary, not headline-eligible until replicated

**Sample-size rigor for benchmark claims.** Any claim about a
benchmark accuracy / TPR / improvement requires:

- N >= 30 examples for any percentage-point delta claim (CLT minimum)
- N >= 1000 for any sub-percentage-point delta claim
- For multi-sample stochastic methods (sampling, MCMC), N samples
  per side AND a chain-length convergence demonstration

**Why this is in CLAUDE.md.** The planner reads CLAUDE.md as
required input on every plan. The agent reads CLAUDE.md as project
context. Putting the rule here ensures both layers apply it at
design time. Mechanical conductor enforcement is the safety net.
Both are needed because the `.144 exp1851 incident showed that
falsifiable gates alone are gameable — the agent can pass the gate
by fabricating numbers if no methodology check is in place.

**Cross-references:**
- `scripts/adversarial_verify.py` — the linter
- `results/experiment_1850_thrml_parity_n128.json` — the corrigendum
  pattern (preserves original numbers + adds correction)
- `results/experiment_1851_nla_probe.json` — the fabrication
  exemplar (3.4s wall time + TPR=1.0 = caught by DURATION_TOO_SHORT
  + IMPLAUSIBLE_PERFECT)

---

## Circularity / Oracle-Distinctness Discipline (MANDATORY)

**Origin:** 2026-06-14 operator-directed ("fix the MET", then "2+3+1"), after the
`.387/`.388 capstones over-claimed the code/efficiency verifier win as "moat proven" and
flipped the DiffusionGemma gate to MET on a CIRCULAR result — the verifier WAS the
executable oracle (run the unit tests on HumanEval / `check_sudoku_validity`). The operator
caught it twice. The loop's fabrication gate (`adversarial_verify`) catches fake/too-fast
results but did NOT catch this over-INTERPRETATION (a circular win inflated into a moat).
This rule turns that manual catch into a mechanical guard.

**The rule.** A verifier "moat" / efficiency / value-added claim is NON-CIRCULAR ONLY when
the verifier is INDEPENDENT of the executable oracle that defines correctness. Where the
verifier IS the oracle (running the tests, checking validity), beating self-consistency or
an LLM-judge is true-but-circular: it does NOT show a learned/energy verifier adds value,
and it MUST NOT headline "moat proven" or flip a gate.

Every artifact making a verifier-value / moat / efficiency claim MUST declare
`verifier_is_oracle: bool` — true if the verifier is the same executable oracle that defines
correctness (circular); false if it is oracle-distinct (learned / energy / probe).

- **Circular (`verifier_is_oracle: true`):** a VALID result (label `execution_grounded`).
  NOT headline-eligible as a moat; may NOT flip a gate (e.g. DiffusionGemma). Honest
  framing: "execution-grounded verification is cheap/automatic/decentralized and beats an
  LLM-judge on cost" — NOT "we proved the verifier moat."
- **Oracle-distinct (`verifier_is_oracle: false`):** headline / gate-eligible. This is the
  deep, OPEN claim — a learned/energy verifier capturing headroom where NO cheap oracle
  exists (the GAP-3-energy-ties-vote-on-ARC frontier).

**Mechanical enforcement (shipped 2026-06-14):**
`scripts/adversarial_verify.py:check_circular_moat_overclaim` emits
`CIRCULAR_MOAT_OVERCLAIM` — CRITICAL when an artifact flips a gate / headlines a moat while
`verifier_is_oracle` is true or undeclared (quarantines the over-claim); WARN when a
moat/efficiency claim omits `verifier_is_oracle` or sets it true (surfaces the circularity
without quarantining the underlying result).

**How to apply (planner-side).** Every verifier-experiment task's REQUIRED ARTIFACT FIELDS
must include `verifier_is_oracle: bool` (with a `principle:` note). Capstone / headline /
gate tasks must honor it: a circular win is reported as `execution_grounded` (not a moat),
and the DiffusionGemma gate stays STILL-PENDING until an oracle-distinct win lands.

**How to apply (agent-side).** When you claim `verifier_value_added` / a moat / an
efficiency win, set `verifier_is_oracle` honestly. If the verifier runs the same check that
defines correctness, it is True (circular) — say so; do not headline it.

**Cross-references:**
- 2026-06-14 operator directive ("fix the MET" + "2+3+1") — origin
- `scripts/adversarial_verify.py:check_circular_moat_overclaim` — the lint
- `docs/research-notes/diffusiongemma-energy-guided-diffusion-spec.md` THE GATE — the gate
  this protects (STILL-PENDING; requires an oracle-distinct win)
- `docs/research-notes/verifier-graft-v3-design.md`,
  `docs/research-notes/verifier-as-detector-measurement-spec.md`
- CLAUDE.md "Adversarial Artifact Verification + Sample-Size Rigor" — sibling rule (catches
  fabrication; this one catches over-interpretation)

---

## Inference-Substrate Declaration Discipline (MANDATORY)

**Origin:** 2026-05-22 operator-authorized fix after exp2842 / exp2837
were flagged DURATION_TOO_SHORT as false positives. The artifacts were
honest: exp2837 scored the verifier ensemble against cached FoVer
candidate triples (16.3s wall-clock for 5-seed dual-condition,
correct), and exp2842 was a capstone aggregating exp2837's numbers
(1.3ms wall-clock for JSON-read + arithmetic, correct). The
`adversarial_verify.py` DURATION_TOO_SHORT rule saw the GGUF markers
in their `model_specs` (vestigial template requirements) and assumed
LLM inference had been claimed. It hadn't been; the markers were
declared because the task spec mandated naming the SOTA GGUF, even
though the experiments didn't invoke it.

**The rule.** Every experiment artifact MUST declare an
`inference_substrate` field at the top level. The legal values are:

| Value | Meaning | Duration floor |
|---|---|---|
| `live_llm_inference` | Loads + runs the declared GGUF / CUDA model. | 60s |
| `verifier_ensemble_against_cached_candidates` | Scores the verifier ensemble against pre-existing (input, candidate, label) triples; does NOT load the LLM. | 1s |
| `aggregation_from_upstream_artifacts` | Reads upstream JSON, computes deltas / formats tables / builds manifests. Capstones, archive/activate, paper-table builders. | 0.0001s |
| `hardware_smoke` | SSH-attached board test, FPGA bring-up, etc. | (per-board, see Pre-Launch Preconditions table) |
| `offline_arcade_live_agent_runtime_self_discovery_no_llm` | The live ARC agent takes real actions against the offline arcade / a live ARC env WITHOUT invoking an LLM (pure Python env-stepping, world-model/verifier scoring, go-explore archive bookkeeping — no GGUF load, no CUDA). GGUF strings that appear elsewhere in the artifact (e.g. naming the generator that WOULD fire if the LLM tier were used, `invoked: false`) are vestigial, not a live-inference claim; `model_specs`/`target_model` is NOT required for this substrate (there is no model to name), but `random_seed` + `reproducibility_checksum` still are. | 0.01s (10ms) |

The `adversarial_verify.py` linter recognizes each value and applies
the matching duration floor. The legacy schema-prefix recognition
(`carnot.fover_memory_leakage_`,
`carnot.cross_corpus_verifier_matrix`, `capstone_v`,
`carnot.milestone_capstone`, etc.) is the historical fallback for
artifacts authored before this discipline shipped, but new artifacts
should set `inference_substrate` explicitly.

**Why the declaration matters.** Without it, the linter has only the
artifact's GGUF / CUDA marker fields as a signal. Those markers are
sometimes vestigial (declared because the task template mandates
naming the SOTA model even when the experiment doesn't invoke it),
sometimes load-bearing (the experiment really does run the GGUF).
The linter cannot tell them apart from string presence alone.
`inference_substrate` is the explicit declaration that resolves the
ambiguity.

**How to apply (planner-side discipline).** When generating
`research-roadmap-next.yaml` task prompts:

1. Decide which substrate the task belongs to before writing the
   prompt. Most experiments are `live_llm_inference`. Verifier-scoring
   benchmarks (FoVer dual-condition, cross-corpus matrices) are
   `verifier_ensemble_against_cached_candidates`. Capstones,
   archive/activate transitions, and paper-table builders are
   `aggregation_from_upstream_artifacts`.

2. Add `inference_substrate` to the task's REQUIRED ARTIFACT FIELDS
   with a `principle:` annotation explaining what failure mode the
   declaration prevents.

3. For verifier-scoring and aggregation tasks, do NOT mandate
   `model_specs.headline_required_any_of: [GGUF list]` in the task
   prompt. The mandate is what introduced the false-positive trigger
   in the first place. The task may STILL record which upstream
   sources it cites (e.g., `cited_upstream_artifacts: [exp2837, ...]`),
   but the artifact's own `model_specs` should reflect what the
   experiment actually invoked &mdash; nothing, in these cases.

4. For aggregation tasks specifically, ensure the prompt also
   captures the upstream provenance: `cited_upstream_artifacts:
   list of {experiment_id, fields_imported, sha256}` so the
   aggregation's numbers are traceable back to a real measurement.
   This is the audit trail that lets a third party verify the
   capstone is not synthesizing numbers from nothing.

**How to apply (agent-side discipline).** When executing a task:

1. Read the task prompt's REQUIRED ARTIFACT FIELDS. If
   `inference_substrate` is listed, populate it with the value the
   task prompt assigns. Do NOT default to `live_llm_inference` if
   the experiment is verifier-scoring or aggregation &mdash; the
   declaration must match what was actually done.

2. If the task prompt does NOT list `inference_substrate` (e.g.,
   legacy planner-emitted task), infer from the task's substantive
   work and declare it explicitly. The discipline is forward-only;
   silently omitting the field is treated as `live_llm_inference`
   by the linter, which is the strict default.

**Mechanical enforcement.** The
`adversarial_verify.py` linter applies the substrate-specific
duration floor and skips the methodology check for aggregation-only
artifacts (since those inherit methodology from cited upstream
sources). Verifier-scoring and live-inference artifacts still
require full methodology (`random_seed` or `random_seeds_used`,
`reproducibility_checksum`, `model_specs`).

**Cross-references:**
- `scripts/adversarial_verify.py:_is_verifier_scoring_only` and
  `_is_aggregation_only` &mdash; the recognition logic
- exp2837 (`results/experiment_2837_fover_memory_leakage_v3.json`)
  &mdash; canonical verifier-scoring exemplar (5-seed dual-condition,
  per-seed AUROC, SHA256 state-restoration verified)
- exp2842 (`results/experiment_2842_capstone_v269.json`) &mdash;
  canonical aggregation exemplar (1.3ms wall-clock, cites exp2837)
- CLAUDE.md "Adversarial Artifact Verification + Sample-Size Rigor"
  &mdash; the parent rule this discipline tightens
- CLAUDE.md "Pre-Launch Preconditions Discipline" &mdash; sibling
  discipline that uses the same `preconditions_checked` pattern

---

## Paper-v6 Narrowing Discipline — Deep Think 2026-05-23 (MANDATORY)

**Origin:** 2026-05-23 Phase-3 Empirical-Readiness Deep Think round
(`docs/research-notes/phase3-empirical-readiness-deep-think-results.md`)
produced 10 findings against the paper-v6 draft: 7 FATAL (2 unprompted
bonus), 2 DEGRADING, 1 COSMETIC. Three FATAL findings require new
measurements (queued in `ops/known-issues.md` MANDATORY-NEXT-MILESTONE
PRIORITIES as Phase-3 Deep Think Corrigenda). Four FATAL findings,
both DEGRADING findings, and the COSMETIC finding are textual fixes:
the paper-v6 draft must NOT re-assert any of the retracted claims
below in autonomous-loop output (capstones, evidence tables, paper-v6
synthesizers, in-process docs).

This discipline is forward-only. The autonomous loop (planner +
capstone agents + paper-v6 synthesizer in `_update_docs_before_planning`)
must respect the narrowings on every milestone until the three Deep
Think Corrigenda experiments land AND the paper-v6 draft is rewritten
to honor the narrowings (operator-curated per Public Documentation
Discipline).

**The retracted claims.** None of the following may appear in any
autonomous-loop-generated artifact, capstone, evidence table, or
in-process doc:

| # | Retracted claim | Forbidden phrasing |
|---|---|---|
| **#2** | KV260 samples reach Boltzmann thermalization | "thermalization," "equilibrium samples," "Boltzmann-distributed energies" anywhere the 24 µs anchor is cited. Use "fixed-compute heuristic budget" instead. |
| **#3** | KV260 hardware speedup over CPU at current d | Any phrasing of "KV260 hardware speedup," "FPGA acceleration over CPU," or "Carnot's verifier ensemble runs faster on KV260" while latent dimension d ∈ {128, 256}. The KV260 is provably SLOWER than CPU at d=128 per the n≈240 crossover. Replace with "POC functional simulator anchoring future high-N deployment." |
| **#6** | Phase-4 VFE bounds validate KV260 deployment | Any phrasing that cites Phase-4 active-inference artifacts (exp2550, exp2748, exp2753, exp2766) in defense of FPGA-deployment claims. Phase-4 VFE bounds apply EXCLUSIVELY to continuous-sampler deployment (RTX 3090). Add a firewall paragraph in the paper draft. |
| **#7** | Extropic Z1 / photonic as future production target | The post-pivot DAE-DEBM architecture is Boolean-coupled; analog substrates cannot strictly enforce discrete sign constraints. Replace with "digital ASICs, spatial FPGAs, or bespoke digital Ising machines" as the future production target. |
| **#8** | Verifier ensemble generalizes universally across modalities | Any unscoped "the verifier ensemble generalizes" or "the verifier ensemble works on novel corpora" claim. The Spera Theorem 9.2 joint null space is coNP-complete; OOD modalities (Lean 4, obfuscated C) have provably-disjoint null spaces. Scope to the 6 corpora in cross-corpus matrix v9+. |
| **#9** | Hardware sovereignty via commodity FPGA | "Hardware sovereignty" while the path from `pip install carnot-ebm` to 24 µs/sample requires Vivado (commercial), Xilinx BSP (proprietary), and the internal SSH workflow. Replace with "local edge deployability." Reproducibility appendix lists Vivado versions + Xilinx dependencies. |
| **#10** | The five-paper_ready streak as scientific maturity | Any phrasing that cites the streak (`.271/.272/.273/.274/.275 paper_ready=true`) as evidence of paper readiness. The streak measures CI loop discipline, not statistical semantics. Relegate to infrastructure / MLOps appendix. |
| **#11** | FoVer headline AUROC = 0.9857 + HIVE peer comparator | Any phrasing that cites `0.9857` as the FoVer headline AUROC or claims `+0.0621 vs HIVE peer 0.924` (the v2 framing). Repinned 2026-05-23 via exp2837's 5-seed dual-condition rescue: production AUROC 0.9131, architecture-only AUROC 0.8947, delta +0.0185. The HIVE comparator's lead is now +0.0061 at the corrected number (not load-bearing for the paper). Added 2026-05-24 after the exp2944 narrowing audit was found to have missed the FoVer numerical value because the audit regex looked for phrasings only, not numbers. Future audit iterations should extend to retracted numerical values. |

**How to apply (planner-side discipline).** When generating any task
that produces paper-v6-eligible output:

- Capstone tasks: the `paper_v6_safe_claims` and `paper_v6_forbidden_claims`
  fields in the capstone artifact must explicitly reflect the seven
  retractions above. If a capstone's `paper_v6_safe_claims` contains
  any phrasing matching the forbidden patterns, the capstone is
  malformed and should be re-emitted.
- Paper-v6 evidence table tasks: do NOT cite KV260 speedup numbers
  at d ∈ {128, 256}. Do NOT cite Phase-4 VFE bounds in defense of
  any FPGA-deployment row. Do NOT cite cross-corpus generalization
  outside the 6 measured corpora.
- Cross-corpus matrix tasks: when adding new rows, do not auto-claim
  generalization beyond the row's specific corpus.
- Operator-status / dashboard tasks: do not surface the five-streak
  metric as a top-level paper-readiness signal.

**How to apply (agent-side discipline, for any agent running a
paper-v6-touching task).** Before emitting any prose, scan the
output against the forbidden phrasings table above. If any match,
narrow the prose to the post-Deep-Think framing. If you cannot make
the claim defensible after narrowing, omit the claim entirely.

**Mechanical enforcement (future).** A `paper_v6_narrowing_lint.py`
pre-commit hook SHOULD scan `docs/arxiv-paper/main.tex` +
`docs/technical-report.md` + any new `paper_v6_*` artifact in
`results/` for the forbidden phrasings. Until that ships, this
discipline is honor-discipline at the planner + agent layer.

**When this discipline retires.** When all three Deep Think
Corrigenda experiments land AND the paper-v6 draft is rewritten
to honor the narrowings (operator-curated commit), this section
moves from MANDATORY to HISTORICAL. The narrowings themselves are
permanent; the discipline-as-discipline is a forward-only guard
until the rewrite is verified.

**What the discipline is NOT.** This is not a blanket suppression
of all FPGA / Phase-4 / verifier claims. The narrower replacement
claims are:

- KV260 as **POC functional simulator** anchoring future high-N
  deployment — defensible
- Phase-4 VFE bounds for **continuous-sampler (RTX 3090) deployment
  only** — defensible
- Verifier ensemble's **0.9131 FoVer AUROC, 5-seed dual-condition,
  delta=+0.0185** — defensible
- Cross-corpus matrix's **6 measured headline-eligible rows** —
  defensible
- **Local edge deployability** via Xilinx tooling stack — defensible
  with reproducibility appendix
- **Apples-to-apples KV260-vs-CPU comparison** if and only if both
  substrates execute the same synchronous-parallel schedule
  (requires Deep Think Corrigendum experiment #2)
- **Code-corpus active-inference** at the AUPRC-defensible threshold
  (requires Deep Think Corrigendum experiment #3)

**Cross-references:**

- `docs/research-notes/phase3-empirical-readiness-deep-think-prompt.md`
- `docs/research-notes/phase3-empirical-readiness-deep-think-results.md`
- `docs/research-notes/phase3-architecture-blindspot-audit-results.md`
  — the 2026-04-30 precedent
- `ops/known-issues.md` MANDATORY-NEXT-MILESTONE PRIORITIES —
  the three new measurement experiments

---

## Pre-Launch Preconditions Discipline (MANDATORY)

**Origin:** 2026-05-15 operator-directed root-cause fix after 5 confirmed
fabrications in the `.144-`.170 window (exp1851 NLA probe, exp1680
PolarFire smoke v2, plus three CASAL `.170 GATE_PASSED_WITHOUT_DATA
flags). Adversarial-verify catches these post-task, but the wall-time
spent running the fabricating task is wasted. **Every confirmed
fabrication shared the same root cause: the agent silently lacked the
required tool, model, credential, or hardware access, and chose to
fabricate a passing result rather than emit a `blocked_*` honest
verdict.**

The success exemplar is exp1694 (`.171 NLA prototype v3). Its prompt
contained this clause explicitly:

> "Verify gemma-4-26B-A4B-it-GGUF is locally cached. If not, abort
>  with honest_verdict `blocked_model_not_cached` (no fabrication)."

exp1694 ran cleanly: TPR=0.73, FPR=0.0, 62-second real duration. The
agent could not fabricate a model run because the precondition check
came BEFORE any inference call in the prompt order.

**The rule.** Every task whose prompt invokes a compute-bound resource
(GGUF model, CUDA inference, GPU training, FPGA toolchain, SSH-attached
hardware, HuggingFace credentials, PyPI credentials, network-dependent
operation, Vivado/yosys/openFPGALoader binaries) MUST include an
explicit **PRECONDITIONS** block at the top of the prompt's CONCRETE
STEPS, structured as:

```
CONCRETE STEPS:
  0. PRECONDITIONS (check BEFORE any subsequent step):
     a. <resource A available?> — e.g., `ls ~/.cache/huggingface/hub/
        models--unsloth--gemma-4-26B-A4B-it-GGUF/` returns non-empty
     b. <resource B available?> — e.g., `huggingface-cli whoami`
        succeeds with non-empty user
     c. <resource C available?> — e.g., `command -v openFPGALoader`
        succeeds
     If ANY precondition is missing, write honest_verdict
     `blocked_<missing_resource>` and exit. DO NOT proceed to the
     measurement / inference / hardware steps. DO NOT fabricate.
  1. <first real step>
  ...
```

**Why "BEFORE any subsequent step"** — the failure mode is the agent
*partially* runs the task, hits a missing-resource error mid-stream,
then synthesizes a plausible-looking artifact from prior context.
exp1851 fabricated TPR=1.0 in 3.4 seconds; the model was never
loaded. Putting the precondition check FIRST in the step sequence
makes fabrication structurally harder: the agent has to actively
ignore an explicit instruction to emit a structured `blocked_*`
verdict.

**Standard precondition clauses by resource type:**

| Resource | Precondition check |
|---|---|
| Cached GGUF model | `ls ~/.cache/huggingface/hub/models--<org>--<model>-GGUF/` returns at least one file |
| HuggingFace credentials | `huggingface-cli whoami` succeeds AND prints non-`anonymous` user |
| PyPI publish credentials | `python -c "import keyring; assert keyring.get_password('https://upload.pypi.org/legacy/', '__token__')"` succeeds OR `TWINE_PASSWORD` env set |
| CUDA inference | `python -c "import torch; assert torch.cuda.is_available() and torch.cuda.device_count() > 0"` |
| Vivado toolchain | `command -v vivado` AND `vivado -version` returns >=2024.1 |
| yosys + openFPGALoader | both `command -v yosys` AND `command -v openFPGALoader` succeed |
| nextpnr (GateMate, ECP5, ice40, etc.) | The 2026-era GateMate flow uses the himbaechel backend, NOT a standalone `nextpnr-gatemate` binary. Use `command -v nextpnr-himbaechel && nextpnr-himbaechel --help` for any GateMate task. The `nextpnr-gatemate` binary does NOT exist in current oss-cad-suite; tasks looking for it will incorrectly emit `blocked_gatemate_toolchain_missing` (cf. exp2899 `.274). Confirmed end-to-end 2026-05-23: yosys 0.64+149 `synth_gatemate` → `nextpnr-himbaechel --device CCGM1A1 --json X --vopt out=X.cfg.bit` → `gmpack X.cfg.bit X.bit` produces a flashable bitstream. |
| GateMate board reachable | `openFPGALoader -c dirtyJtag --detect` returns the `colognechip / GateMate Series / GM1Ax` IDCODE. The `--scan` flag does NOT exist on openFPGALoader; use `--scan-usb` or `--detect`. |
| PolarFire SSH | `ssh -o ConnectTimeout=5 polarfire 'true'` returns 0 |
| KV260 hardware (board booted, reachable via SSH) | `ssh -o ConnectTimeout=5 -o BatchMode=yes kria 'true'` returns 0. The board has run Ubuntu Xilinx since 2026-05-20 ~13:00 EDT; SD-card-flash workflows are obsolete (2026-05-20 operator directive — interact with the board via SSH, not by flashing a host SD card with PYNQ). |
| THRML installed | `python -c "import thrml; print(thrml.__version__)"` succeeds |
| Network for arXiv/HF | `curl -sf -o /dev/null https://huggingface.co/api/models` succeeds |

**Blocked-verdict naming convention.** When a precondition fails, the
verdict must be `blocked_<resource>` with the specific resource named:
`blocked_model_not_cached_gemma_4_26B_A4B`, `blocked_huggingface_credentials`,
`blocked_vivado_toolchain_missing`, `blocked_polarfire_ssh_timeout`. The
conductor's reconciler classifies `blocked_*` verdicts as honest
non-terminal states (NOT as fabrications or partial failures), so the
task simply retires without burning the doomed-rerun ledger
unnecessarily.

**How to apply (planner-side discipline).** When generating
`research-roadmap-next.yaml`, for every task whose prompt invokes ANY
resource in the table above:

1. Identify which resources the task needs.
2. Compose a PRECONDITIONS step 0 with the relevant checks from the
   table.
3. Put step 0 BEFORE step 1 in CONCRETE STEPS.
4. Add to REQUIRED ARTIFACT FIELDS: `preconditions_checked: list of
   {resource: str, available: bool}` so the artifact records WHICH
   preconditions the agent actually verified.

**How to apply (agent-side discipline).** When executing a task with
a PRECONDITIONS block:

1. Run the precondition shell commands literally. Capture stdout +
   exit code.
2. If ANY precondition fails: write the artifact with honest_verdict
   `blocked_<resource>`, `preconditions_checked` populated, and EXIT.
   Do NOT attempt to proceed.
3. Only if ALL preconditions pass: proceed to step 1 and onward.

**Mechanical safety net (lives in `scripts/adversarial_verify.py`).**
The METHODOLOGY_MISSING detector should be extended to flag any
compute-bound artifact that lacks a `preconditions_checked` field.
Pending mechanical enforcement, this rule is honor-discipline at the
planner+agent layer.

**Why this is in CLAUDE.md, not just in task prompts.** Same defense-
in-depth principle as Codex-Default + Verdict Terminal-Prefix +
Failed-Experiment Rerun: the planner reads CLAUDE.md as required
input. A rule here ensures every future planner-generated task prompt
includes the PRECONDITIONS clause for compute-bound resources without
operator having to re-specify it per-milestone.

**Cross-references:**
- `results/experiment_1694_nla_v3.json` — success exemplar (precondition
  clause inline; 62s real run; no fabrication)
- `results/experiment_1851_nla_probe.json` — fabrication exemplar (no
  precondition clause; TPR=1.0 in 3.4s; caught post-task by adversarial-
  verify but wall time wasted)
- `results/experiment_1680_polarfire_smoke_v2.json` — fabrication
  exemplar (no SSH-precondition; TPR=1.0 with run_duration_s=0; caught
  by GATE_PASSED_WITHOUT_DATA + IMPLAUSIBLE_PERFECT post-task)

---

## Principle-Annotated Artifact Fields + Verifier-Principle Discipline (MANDATORY)

**Origin:** Anthropic, "Teaching Claude Why" (2026-05, anthropic.com/
research/teaching-claude-why). Anthropic showed empirically that
training Claude on the *principles underlying* aligned behavior is
**28x more sample-efficient** than training on demonstrations of
correct behavior alone, AND **generalizes better out-of-distribution**.
Blackmail rate fell 65% -> 19% from constitutional / principle-grounded
training; Haiku 4.5 hit perfect agentic-misalignment scores. The lesson
generalizes well beyond alignment: agents that know *why* a constraint
matters are harder to mimic, more robust to OOD inputs, and require
fewer training examples to comply correctly.

For Carnot this principle bridges two layers:

1. **Planner-emitted task prompts** — every `REQUIRED ARTIFACT FIELD`
   declared in a task prompt MUST carry a one-line `principle:`
   annotation explaining WHY the field matters. The agent generating
   the artifact then has the principle in context, not just the
   directive, and produces compliance that generalizes to novel task
   shapes.
2. **Carnot's own verifier-ensemble training** — energy-based
   verifiers (Boolean E, SAT, Z3, AST, liveness, etc.) currently
   train on (correct, violating) pair data. The "teach why" result
   predicts each verifier becomes more sample-efficient AND more
   OOD-robust if it co-trains on *explanations of why a violation
   is a violation* — not just labels. This is directly load-bearing
   for the Phase-3 verifier-ensemble null-space mimicry problem
   (project_null_space_mimicry_attack memory): a verifier that
   knows the principle behind its constraint is harder to game than
   one that only knows the decision boundary.

**The rule (planner-side).** When generating `research-roadmap-next.yaml`:

- Every `REQUIRED ARTIFACT FIELD` entry MUST carry a one-line
  `principle:` annotation. Examples:

  ```yaml
  REQUIRED ARTIFACT FIELDS:
    duration_s:
      principle: "Real compute takes wall-clock time; missing or
                  implausibly-short duration is the load-bearing signal
                  for fabrication detection (cf. DURATION_TOO_SHORT in
                  adversarial_verify.py)."
    random_seed:
      principle: "Determinism is the precondition for reproducibility;
                  missing seed means no third party can re-run the
                  experiment and confirm or refute the claim."
    reproducibility_checksum:
      principle: "Content-addressed hash of the run inputs catches
                  silent corpus or model-version drift between this
                  artifact and any future replication attempt."
    honest_verdict:
      principle: "Self-declared terminal state lets the conductor
                  reconciler distinguish success / partial / blocked
                  without re-running the experiment; misclassification
                  causes spurious retire / retry."
    preconditions_checked:
      principle: "Records WHICH resources the agent verified before
                  launching; pre-empts the fabrication mode where the
                  agent silently lacked the resource and synthesized
                  a passing artifact instead of emitting blocked_*."
  ```

- For gate conditions (acceptance criteria), every gate MUST carry
  a one-line `principle:` annotation explaining what failure mode
  the gate guards against. Example:

  ```yaml
  acceptance_gates:
    - condition: "TPR > 0.7 AND FPR < 0.05"
      principle: "TPR floor prevents trivial all-reject classifier;
                  FPR ceiling prevents trivial all-accept classifier.
                  Both bounds together rule out the degenerate-bias
                  failure mode that exp1851 fabricated past."
  ```

**The rule (agent-side, when writing artifacts).** When emitting a
results artifact, the `principle:` annotations from the task prompt
SHOULD be echoed into the artifact's `field_provenance:` block so the
artifact is self-explanatory at audit time:

```json
{
  "duration_s": 62.3,
  "field_provenance": {
    "duration_s": {
      "principle": "Real compute takes wall-clock time; ...",
      "satisfied_by": "wall_clock measurement of inference loop"
    }
  }
}
```

This is not yet mechanically enforced; it's a near-term discipline
upgrade with high expected value.

**The rule (verifier-training-side, for Phase-3 / Phase-4 work).**
When the Phase-3 verifier ensemble grows (currently k=15, scheduled
k=16 with the NLA-class probe), each verifier's training corpus MUST
include `principle_explanation` fields alongside the (correct,
violating) pair labels. The verifier learns to predict the principle
(or a featurization of it) as a co-task, not just the binary label.

Open empirical question — is `principle_explanation` co-training
worth the corpus-construction cost on EACH of the 15 base verifiers?
Plausibly the answer is yes only for high-stakes verifiers (Z3,
liveness, NLA) and not for cheap ones (AST). This question is queued
for a `.211+ experimental task — DO NOT pre-commit corpus changes
for all 15 verifiers without empirical justification per verifier.

**Cross-applies to Phase-4 free-energy verifier.** The Fast-Slow
Variant metric (named canonical .194) measures variational free
energy. Per the "teach why" finding, training the Fast-Slow Variant
on principle-grounded reasoning (why a low-energy state is
low-energy) should beat training on energy labels alone, with the
OOD-robustness benefit that's the most-load-bearing property for
verifier ensembles.

**What this rule is NOT.** This is not a request to add prose
explanations to every line of every task prompt. The principle
annotation is one-line, machine-readable structure (`principle: "..."`),
and it lives next to the field/gate it justifies. The cost is small;
the OOD-robustness payoff is large per the Anthropic empirical result.

**How to apply (planner-side, immediately).** When drafting the next
`research-roadmap-next.yaml` (the .212+ pre-staged roadmap), add a
`principle:` annotation to every `REQUIRED ARTIFACT FIELD` and every
gate condition. Existing milestones (.210, .211) are not retroactively
re-annotated; this is a forward-only discipline.

**Mechanical enforcement (future, lives in
`scripts/research_conductor.py` activation guard).** The activation
guard SHOULD scan emitted YAMLs and refuse a milestone whose task
prompts declare `REQUIRED ARTIFACT FIELDS` without `principle:`
annotations on each entry. Until that ships, this rule is
honor-discipline at the planner layer alone.

**Why this is in CLAUDE.md, not just in the planner prompt.**
Same defense-in-depth pattern as Codex-Default, Verdict
Terminal-Prefix, Pre-Launch Preconditions, and Exclusion-Manifest
Cross-Check: the planner reads CLAUDE.md as required input on every
plan generation. A rule here ensures the principle-annotation
discipline applies at design time without operator having to
re-specify per-milestone.

**Cross-references:**
- Anthropic "Teaching Claude Why" — anthropic.com/research/teaching-claude-why
- `reference_anthropic_teaching_why.md` (memory) — full Carnot integration notes
- `project_null_space_mimicry_attack.md` (memory) — Phase-3 attack
  pattern that principle-grounded verifiers structurally resist
- `project_orthogonality_stall.md` (memory) — single-verifier
  compute-immune ceiling; principle co-training is candidate antidote
- `feedback_nla_class_16th_verifier_committed.md` (memory) — the
  .124+ NLA chain now has stronger theoretical underwriting via the
  "teach why" finding
- `scripts/adversarial_verify.py` — the post-hoc safety net the
  principle-annotation rule reduces dependence on (principles in
  the prompt do at design-time what the linter does at audit-time)

---

## Phase Prototype + Empirical Validation + Adversarial Check Discipline (MANDATORY)

**Origin:** 2026-04-30 Phase-3 architecture blind-spot audit caught
5 FATAL findings that three rigorous theoretical Deep Think rounds
missed. Lesson: *unless we have adversarial checks at each phase
boundary, we are building a house of cards that cannot function in
the end.*

Every Carnot phase (1a/1b/1c/1d... 2a/2b/2c... 3a/3b/3c...) must
satisfy three requirements before ANY scaling decision is committed:

1. **Software prototype** — concrete code artifact in the repo, not
   just an architecture document. The prototype must be runnable
   end-to-end at small scale (e.g., 6,500-pair FoVer corpus for
   Phase-3 substrate).

2. **Empirical validation criteria** — a documented list of
   measurable pass/fail tests with explicit thresholds. Examples:
   `inf_t α_t > 0.1` over 100 MLD steps; decoder
   `joint-constraint pass rate > 85%`; verifier joint null-space
   `dim < 5%` of input space; FPGA sampler `KL(P_fpga || P_gibbs) < ε`.

3. **Adversarial check** — a hostile-reviewer round explicitly
   commissioned to find ways the prototype could pass acceptance
   gates without actually working. Required BEFORE scaling, not
   after. Examples of the attack patterns to demand:
   - Could the encoder be learning a degenerate identity?
   - Could the decoder be ignoring the bottleneck and using language-
     model prior alone?
   - Could the EBM be converging to a single low-energy point?
   - Could the verifier suite share a pathological joint null space?
   - Could the hardware sampler be sampling from a different
     distribution than the model intends?

**Empirical instrumentation IS adversarial check at scale.** A
prototype that emits the right diagnostics surfaces architecture-
level flaws automatically. A prototype that doesn't will let
flaws ship. Therefore every phase prototype MUST include the
diagnostic instrumentation for EVERY theoretical concern the phase
rests on (α_t tracking, joint null-space estimation, KL divergence,
decoded-text diversity, etc.).

**Cross-phase verification.** Every phase artifact must produce
empirical pass/fail data visible to downstream phases. A Phase-3
prototype that depends on Phase-1c's k=15 AND-composition must
VERIFY at integration time that Phase-1c's empirical claims hold
on the deployed verifier suite — not trust them.

**Planner instructions:**

- When proposing tasks for any phase, include the prototype +
  empirical-criteria + adversarial-check trio. A task that proposes
  scaling without one of these three is rejected.
- The .85+ planner has 5 candidate tasks already filed under this
  discipline (see `ops/known-issues.md`):
  1. Phase 1a Adversarial Verifier Robustness Audit
  2. Phase 1c Verifier Joint Null-Space Measurement
  3. Phase 2a Sampler Correctness Audit
  4. Phase 3a Pre-Prototype Adversarial Round
  5. Diagnostic instrumentation library
- Architecture-level Deep Think rounds remain valuable but cannot
  substitute for empirical instrumentation. Treat any architecture
  decision as provisional until the phase prototype confirms
  empirically.

**Cross-references:**
- Full framework: `docs/research-notes/phase-prototype-and-validation-framework.md`
- Audit precedent: `docs/research-notes/phase3-architecture-blindspot-audit-results.md`

## Verdict Terminal-Prefix Discipline (MANDATORY)

**Origin:** the conductor's `_verdict_is_untrustworthy` classifier
(`scripts/research_conductor.py`) substring-matches `_PARTIAL_TOKENS`
(marginal, blocked, no_improvement, etc.) against agent verdicts to
detect partial runs masquerading as success. The matcher fires
false-positives on terminal verdicts whose descriptive text contains
nuance words. Five separate incidents required classifier patches:

- exp1305 (HardNet++/DSP): `complete: ... DSP feasibility marginal ...`
  flagged because "marginal" matched (fixed via terminal-prefix
  whitelist, commit 95adf3c3)
- exp1393 (GRPO v8): `no_improvement_all_unknown_retired` flagged
  because "no_improvement" matched (fixed via `_retired` honest-finding
  token, commit 6d9363ae)
- exp1430 (PRM repair selector): `complete_prm_guided_no_improvement_...`
  flagged because verdict used underscore separator not colon (fixed
  via `complete_/success_/passed_/shipped_` underscore-prefix
  recognition, commit fb824412)
- exp1473 (Live Telemetry Adversarial Validity Audit):
  `telemetry_claim_blocked_adversarial_audit` flagged because "blocked"
  matched, even though "claim_blocked" here means "the audit
  successfully blocked an unsupported claim" — terminal good outcome.
  Fixed via positive-context blocked-pattern whitelist (2026-05-07).

**The structural fix:** instead of expanding the classifier whitelist
indefinitely, **all agent prompts must produce verdicts that START
with one of the terminal prefixes:**

```
complete:    or  complete_
success:     or  success_
passed:      or  passed_
shipped:     or  shipped_
```

Verdicts that lead with the experiment name or descriptive text
(e.g., `telemetry_claim_blocked_adversarial_audit`) are vulnerable
to substring false-positives in the partial-token check. Verdicts
prefixed with a terminal marker bypass that check entirely.

**Acceptable verdict examples:**

```
✅ complete: adversarial audit blocked the telemetry claim
✅ complete_adversarial_audit_blocked_telemetry_claim
✅ success_kv260_rtl_lint_passed_with_2_warnings
✅ passed_qwen3.6_logprob_telemetry_topk_available
✅ shipped_minimal_repair_pipeline_v5
```

**Unacceptable verdict examples (vulnerable to false-positive):**

```
❌ telemetry_claim_blocked_adversarial_audit         (no terminal prefix; "blocked" matches partial)
❌ marginal_repair_v3_no_headline                    (no terminal prefix; "marginal" + "no_headline" both match partial)
❌ kv260_rtl_lint_blocked_no_source                  (no terminal prefix; "blocked" matches partial)
```

**How to apply (planner-side discipline).** When the planner generates
`research-roadmap-next.yaml` task prompts, the `REQUIRED ARTIFACT
FIELDS` section's `honest_verdict` description must specify:

> The `honest_verdict` field MUST start with one of the terminal
> prefixes: `complete:` / `complete_` / `success:` / `success_` /
> `passed:` / `passed_` / `shipped:` / `shipped_`. Descriptive text
> follows the prefix. The prefix is required for the conductor's
> reconciler to classify the verdict as terminal; without it,
> verdicts containing words like "marginal" / "blocked" /
> "no_improvement" risk false-positive partial classification and
> spurious retry/retirement.

**How to apply (agent-side discipline).** When writing terminal
artifacts in `results/experiment_*.json`, prefix the `honest_verdict`
with one of the four terminal markers. If the experiment ran fully
and reached a scientific conclusion (positive, negative, or mixed),
the verdict is terminal — use a prefix. Reserve missing-prefix
verdicts for genuine bootstrap-only / partial states, where the
conductor's reconciler should retry.

**Mechanical safety net.** The conductor's `_verdict_is_untrustworthy`
classifier patches (commits 95adf3c3, 6d9363ae, fb824412, +
2026-05-07 positive-blocked patterns) provide partial protection for
verdicts without prefix. But the patch-then-patch pattern is
unsustainable — the planner-side discipline (this rule) eliminates
the need for further classifier patches by ensuring all verdicts
are well-formed at source.

**Why this is in CLAUDE.md, not just inline in task prompts.** Same
defense-in-depth principle as the Codex-Default and Scope-Reduction
rules: the planner reads CLAUDE.md as required input on every plan
generation. A rule here ensures every future task prompt's
verdict-spec includes the prefix requirement.

## Scope-Reduction-When-Flagged Discipline (MANDATORY)

**Origin:** 2026-05-06 incident. Operator queued a "SCOPE REDUCTION
MILESTONE (.111 — preempts all other priorities)" entry at the top of
`ops/known-issues.md` MANDATORY-NEXT-MILESTONE PRIORITIES. The entry
specified 8 mandatory scope-reduction tasks (experiment artifact
classifier, GRPO lineage retirement, WOPR puzzle cartridge retirement,
known-issues.md priority audit, paper-v6 anchored-claims narrowing,
self-learning lineage decision, hardware portfolio narrowing,
comparator-integration audit). The `.111 planner read the entry and
ignored it — instead generated 14 tasks continuing the repair-executor,
FR-11, RTL, EBT lineages that were explicitly flagged as candidates
for retirement. Planners have a strong **carry-forward bias**: when
.110 closes with unresolved tracks, the natural move is "carry these
forward as `.111 tasks." That bias defeats explicit scope-reduction
priorities filed in known-issues.md alone.

**The rule.** When `ops/known-issues.md` MANDATORY-NEXT-MILESTONE
PRIORITIES contains an entry whose title contains the word
**"SCOPE REDUCTION"** (or equivalent operator-flagged editorial
directive), the planner MUST:

1. Treat that entry as **preempting** all other priorities including
   carry-forwards from the prior milestone's retro.
2. Allocate at least **8 of N tasks** (where N is the planned task
   count, typically 13-14) to the scope-reduction work specified
   in the entry — NOT to expansion of existing lineages.
3. Refuse to propose tasks that match the scope-reduction entry's
   "NOISE candidates" classification list. If the entry says
   "Retire GRPO v1-v14 lineage," do not propose GRPO v15. If the
   entry says "Retire HardNet++/DSP repair iterations," do not
   propose HardNet++ v_{N+1}.
4. Output a **`scope_reduction_compliance` field** in the roadmap
   YAML's milestone metadata, listing which scope-reduction tasks
   from the entry are addressed by which `.NNN` task IDs.

**Operational definition of "scope reduction" for this rule.** A task
is scope-reduction if it:

- Classifies, audits, or measures the existing experiment / artifact
  / priorities corpus
- Consolidates one or more lineages into a single retirement artifact
- Removes / archives / blocks experiments rather than running new ones
- Narrows a paper / spec / known-issues entry to a smaller set of
  claims or items

A task is **NOT** scope-reduction if it:

- Proposes a new experiment lineage version (vN+1)
- Carries forward an unresolved track from a prior milestone retro
- Adds a new comparator integration or model architecture
- Extends an existing pipeline with new functionality

**Why a known-issues.md entry alone is insufficient.** The planner
reads `known-issues.md` as input but has equal or stronger signal
from the prior milestone's retrospective artifact, which typically
catalogues unresolved tracks that "obviously" need follow-up. The
planner's training distribution favors carry-forward continuity. A
single known-issues entry can't override that bias without explicit
CLAUDE.md authority.

**How to apply (planner-side discipline).** Before drafting
`research-roadmap-next.yaml`:

1. Grep `ops/known-issues.md` for `^### NEW.*SCOPE REDUCTION` or
   equivalent directive phrasings. If found, the directive
   preempts other priorities.
2. Read the directive entry's "NOISE candidates" / "tasks to
   propose" / "expansion forbidden" sections.
3. Allocate the mandated minimum task count to scope-reduction work.
4. Document the allocation in the YAML's `scope_reduction_compliance`
   field so the activation guard can verify.

**Mechanical enforcement (conductor-side, future).** The conductor's
activation guard SHOULD verify `scope_reduction_compliance` is
non-empty and references at least the mandated minimum task count
when a SCOPE REDUCTION directive is active. Pending mechanical
enforcement, this rule is honor-discipline at the planner layer
alone.

**Why this is in CLAUDE.md, not just in known-issues.md.** Same
defense-in-depth principle as Codex-Default-for-Experiments: a rule
that only lives in known-issues.md is advisory and can be
overridden by carry-forward bias. The planner reads CLAUDE.md as
required input on every plan generation; that's the authority
needed to override training-distribution priors.

**The .111 incident specifically.** The `.111 planner generated 14
tasks (exp1439-1452) continuing repair-executor / FR-11 / RTL / EBT
lineages explicitly flagged as NOISE candidates by the operator's
SCOPE REDUCTION directive. This rule, had it been in CLAUDE.md at
the time, would have forced the planner to allocate 8+ tasks to
scope reduction instead. The `.112 planner reads this rule; future
SCOPE REDUCTION directives will be honored by design.

## Hardware-Task Continuity Discipline (MANDATORY)

**Origin:** 2026-05-18 operator directive: "I just don't want you to
forget about the FPGAs again." Filed after the `.220-`.226 cascade
window where every FPGA-related task got gate-blocked, doomed-rerun-
blocked, or 3-fail-skipped across 6+ consecutive milestones — partly
due to the codex quota cascade, but also because the planner deferred
hardware tasks in favor of verifier / AUROC ceiling work each time
a milestone got crowded.

**The rule.** As long as Carnot has FPGA boards physically attached
to the development machine, **every milestone's `research-roadmap-next.yaml`
MUST include at least one task targeting EACH attached board**, until
that board reaches its defined terminal state.

**Currently attached boards (as of 2026-05-18):**

| Board | USB enumeration | Reachable via | Terminal state |
|---|---|---|---|
| AMD/Xilinx KV260 | booted from onboard microSD; host has NO role in board's storage once booted | `ssh kria` (alias for 192.168.51.98) — Ubuntu Xilinx, reachable since 2026-05-20; bitstream activation via on-board `xmutil loadapp carnot_ising_v2_n64`. **NEVER use host `/dev/mmcblk*` as a precondition for KV260 tasks** — see "KV260 SSH-Not-SD-Card Discipline" below. | board-level latency transcript landed in a non-fabricated artifact + `kv260_synthesis_succeeded: true` |
| Cologne Chip GateMate A1-EVB-2M | onboard DirtyJTAG MCU at `1209:c0ca` | `openFPGALoader -c dirtyJtag -b olimex_gatemateevb <bit>` over board's USB-C | n=16 Ising tile flashed + smoke-tested on hardware; `gatemate_bitstream_flashed: true` |
| Microchip PolarFire SoC Discovery Kit | FlashPro5 at `1514:2008` + booted Linux | `ssh polarfire` (uptime 4+ days as of 2026-05-18) | end-to-end Carnot dispatch run with hash-match verification; `polarfire_workload_validated: true` |

**How to apply (planner-side discipline).** When generating
`research-roadmap-next.yaml`:

1. **Audit the attached-boards table** (above, kept current by the
   outer-loop/operator). For each board NOT yet at its terminal state,
   reserve one task slot.
2. **Pick the next concrete forward step** for that board, based on
   the most recent artifact in `results/` and the next gate in the
   board's own track. Examples:
   - KV260: if synthesis_errors still > 0 → another debug/fix task;
     if synthesis clean → bitstream-pack + board-flash task;
     if board-flashed → board-latency transcript task.
   - GateMate: if yosys/nextpnr LUT mapping mismatch unresolved →
     mapping-workaround task; if resolved → n=16 P&R + flash + smoke;
     if flashed → on-board Ising sampler timing benchmark.
   - PolarFire: if no smoke yet → precondition-gated SSH smoke (per
     2026-05-18 known-issues entry); if smoked → CPU-only Ising
     sampler; if CPU sampler validated → adaptive-K PCD prototype.
3. **Mark each FPGA task with `track: hardware`** in the YAML so
   future audits can confirm one-per-board-per-milestone compliance.
4. **If a board's next step requires operator action** (Vivado
   bitstream, physical re-cabling, etc.) and that action hasn't
   happened, the planner SHOULD still queue a documentation-only or
   audit task for that board, NOT skip it. A queued audit task keeps
   the board visible in milestone retros and prevents the
   forget-pattern.

**Operator-override for skipping.** A board can be omitted from a
specific milestone only if the operator explicitly authorizes it
(e.g., "skip PolarFire this milestone, the board is being firmware-
upgraded"). The override must appear in `ops/known-issues.md` as
a dated entry, NOT in the planner's own judgment.

**Terminal-state graduation.** Once a board hits its terminal state
(per the table above), the board CAN be dropped from per-milestone
mandatory inclusion. At that point, follow-on work on that board
shifts to optional / opportunistic. The terminal artifact must be
adversarial-verified non-fabricated (per Pre-Launch Preconditions
Discipline + adversarial_verify.py).

**Mechanical enforcement (planner-side honor-discipline + future
activation-guard check).** Until a roadmap-FPGA-coverage check is
mechanically wired into `scripts/research_conductor.py`, this rule is
planner honor-discipline. The outer-loop/operator audits roadmap
drafts and re-emits if FPGA coverage is missing.

**Forward queue (queued in `ops/known-issues.md` as of 2026-05-18):**

- `.237+ PolarFire SoC Smoke v3 — Precondition-Gated, Not Fabricated
- `.237+ GateMate A1 yosys/nextpnr LUT mapping workaround (or
  bitstream end-to-end if mapping resolved)
- `.237+ KV260 synthesis fix follow-on or bitstream pack (depending
  on exp2440 outcome in `.236)

**Why this is in CLAUDE.md, not just `ops/known-issues.md`.** The
planner reads CLAUDE.md as required input on every plan generation.
A rule that only lives in `ops/known-issues.md` is advisory and
sometimes deprioritized when the milestone gets crowded. Hardware
tasks have repeatedly been the deprioritized class. Putting the
continuity rule here forces the cross-check at design time:
"does this milestone have at least one task per attached board?"
becomes a checkbox the planner must answer before emitting.

**Cross-references:**
- `ops/hardware-bringup-prep.md` — current bring-up state for each board
- `ops/known-issues.md` MANDATORY-NEXT-MILESTONE PRIORITIES — the
  freshest per-board queued task
- `research-hardware-wishlist.md` — exp 1460 scope reduction and
  active/deferred portfolio definitions
- `docs/jtag-wiring-gatemate-dirtyjtag.md` — reference for the rare
  case of external DirtyJTAG (not the current bench)
- 2026-05-18 operator directive — origin

## KV260 SSH-Not-SD-Card Discipline (MANDATORY)

**Origin:** 2026-05-20 ~22:45 EDT operator directive after seeing
exp2722 escalate for the FOURTH consecutive milestone with a phantom
SD-card-absent verdict:

> "this has happened multiple times now. can you make sure this
>  SDCARD confusion with the kv260 does not happen again?"

The preceding 5 consecutive milestones (`.254, `.256, `.257, `.258, and
queued-for-`.259 exp2735) all used `ls /dev/mmcblk* 2>/dev/null` as the
KV260 precondition. **That command checks the HOST machine's SD card
slot — meaningless for the BOARD's state.** Wrong-mechanism leftover
from a pre-board-boot PYNQ-SD-card workflow that no longer applies.

The KV260 has been booted with Ubuntu Xilinx and reachable via SSH
since 2026-05-20 ~13:00 EDT. Bitstream updates flow via `scp` +
`xmutil loadapp`. Host SD-card-flash workflows are PERMANENTLY retired
per this operator directive.

**The rule.** For ANY KV260 task — continuity check, bitstream
update, sampler smoke, latency benchmark, RTL deployment — the
precondition MUST be SSH-reachability of the board, NEVER host SD card
presence:

```bash
# CORRECT precondition for KV260 tasks:
ssh -o ConnectTimeout=5 -o BatchMode=yes kria 'true'
# Exit 0 → board reachable; proceed.
# Exit non-zero → honest_verdict: blocked_kv260_ssh_unreachable. Stop.
```

```bash
# RETIRED — do NOT use this for KV260:
ls /dev/mmcblk* 2>/dev/null || echo no_sd       # ← HOST's SD slot
test -e /dev/mmcblk0                            # ← same
[ -b /dev/mmcblk0p1 ]                           # ← same
```

**Permitted on-board operations (all over SSH):**

| Operation | Command |
|---|---|
| Reachability check | `ssh -o ConnectTimeout=5 -o BatchMode=yes kria 'true'` |
| List loaded overlays | `ssh kria 'xmutil listapps'` |
| Load Carnot Ising bitstream | `ssh kria 'xmutil loadapp carnot_ising_v2_n64'` |
| Unload current overlay | `ssh kria 'xmutil unloadapp'` |
| List UIO devices | `ssh kria 'ls /dev/uio*'` |
| Push a fresh bitstream | `scp <local-path>/carnot_ising_vN.bit.bin kria:/lib/firmware/xilinx/<name>/` |
| Run on-board Python | `ssh kria 'python3 -c "..."'` |
| Read a UIO register | `ssh kria 'python3 -c "with open(\"/dev/uio0\",\"r+b\") as f: ..."'` |

**Why a host SD-card precondition is structurally wrong.**

- The KV260 boots from its onboard microSD (or eMMC). Once booted, the
  host machine is not in the board's storage path.
- All Carnot-side board updates (new bitstream, overlay swap, software
  install) are SSH operations, not host-storage operations.
- `ls /dev/mmcblk*` on the host returns true ONLY if the operator has
  physically inserted an SD card into the HOST's card reader — a state
  irrelevant to whether the BOARD is reachable or has a Carnot
  bitstream loaded. Five consecutive milestones escalated for an
  operator action (insert SD card into HOST) that does nothing useful.

**How to apply (planner-side discipline).** When generating
`research-roadmap-next.yaml`:

1. Any task with `track: hardware` AND scope containing "KV260" /
   "kria" / "Kria" MUST use the SSH precondition. If the task prompt
   you would emit contains `/dev/mmcblk`, **rewrite the precondition
   before emitting** — do not propose it.
2. Do NOT propose tasks whose title or scope contains "SD Card
   Branch," "Branch B SD Card," "PYNQ Image Flash," or any equivalent
   phrasing for KV260. That scope is permanently retired per
   `ops/exclusion_manifest.yaml:kv260_host_sd_card_precondition_retired`.
3. If the planner is uncertain whether a board is reachable, the
   correct response is the SSH-reachability precondition + a
   `blocked_kv260_ssh_unreachable` fallback — NOT a Branch-A/Branch-B
   split on host SD card presence.

**How to apply (agent-side discipline).** If an agent receives a task
prompt that contains `/dev/mmcblk` in the PRECONDITIONS for a KV260
task, the agent MUST refuse and emit:

```
honest_verdict: blocked_kv260_wrong_mechanism_sd_card_precondition
```

with a methodology_note pointing to this CLAUDE.md section. Do NOT
"do your best" with the wrong precondition — that produces an
artifact that escalates the operator for an action (insert SD card
into host) that accomplishes nothing.

**Mechanical safety net (lives in `scripts/research_conductor.py`).**
The activation guard SHOULD scan task prompts for the joint pattern
`(KV260 OR kria OR Kria) AND /dev/mmcblk` and refuse activation of
any milestone containing such a task. Until that ships, this rule is
honor-discipline at the planner + agent layer, with the exclusion
manifest entry `kv260_host_sd_card_precondition_retired` as the
structural backstop.

**Cross-references:**
- 2026-05-20 ~22:45 EDT operator directive — origin
- `ops/exclusion_manifest.yaml` —
  `kv260_host_sd_card_precondition_retired` blocks scope at activation
- `feedback_kv260_ssh_not_sd_card.md` (memory) — operator directive
  notes, when-to-change protocol
- `feedback_kv260_latest_bitstream_must_be_xdc_constrained.md` (memory)
  — earlier complementary directive establishing `xmutil loadapp` as
  the bitstream-update mechanism
- CLAUDE.md "Pre-Launch Preconditions Discipline" table — KV260 row
  authoritative precondition

## Depth-Over-Breadth Forcing Function (RETIRED 2026-05-31 — condition satisfied; preserved per never-prune)

> **STATUS (2026-05-31): RETIRED — its own retirement condition is now met.**
> The rule below relaxes "once P0.1 has a verdict AND G2 is either met or has a
> concrete in-flight reproducer." Both now hold:
> - **P0.1 has a recorded HONEST-NEGATIVE verdict.** The energy-descent
>   existential claim is bounded on the tested math/CSP corpora: Route-2 is
>   triply-confirmed bounded (oracle ≤ SC; exp3530/3531/3542), Route-1 ties a
>   strong DSATUR baseline (exp3540, paired p=0.135, "advantage was small-sample
>   artifact") and is bounded-to-single-generator (exp3563), and the one clean
>   positive (step→final aggregation, exp3532) does NOT transfer cross-corpus
>   (exp3565). The shared root cause: self-consistency / strong classical
>   baselines are already near-optimal on these corpora, leaving no headroom.
> - **G2 is CLOSED.** The FoVer verifier-ensemble headline independently
>   reproduced on a clean CI runner (GitHub Actions "FoVer Headline Independent
>   Reproducer", run 26725185125, 2026-05-31, success); `publication_gate.py`
>   now reports G1∧G2∧G3∧G4 = `paper_ready: true`.
>
> **Effect:** the planner is NO LONGER required to reserve milestones for P0.1
> Route-1/Route-2. Continuing to re-test the answered existential question is
> the diminishing-returns churn this rule was meant to prevent — so the rule
> correctly retires rather than perpetuates it. The planner may resume normal
> research breadth toward NEW directions (e.g. the verifier ensemble's
> discriminating value / cross-domain generalization — where SC is NOT
> near-optimal). The prose below is preserved as historical context per the
> never-prune rule; do NOT treat it as active guidance.

**Origin:** 2026-05-30 operator directive after a session audit. The loop runs
~30 milestones/day but produces mostly **breadth churn** — `vN+1` re-measurement
of already-measured artifacts (cross-corpus matrix v38, telemetry v39, repair
panel vN, SOTA receipt vN, clean verifier rerun vN) — while the 1–2 questions
that actually decide the foundation-model endgame go unrun. The P0.1 existential
test (energy-descent-vs-autoregressive, exp3312) was queued at `.82` and never
run across 200+ milestones. This is *iteration without convergence* (see
`ops/north-star.md`): the same version numbers climb in lockstep and
`paper_ready` stays False throughout.

**The rule.** Until P0.1 (exp3312 / the Kona global-opt solve-rate gate in
`ops/known-issues.md`) has a recorded verdict — positive OR honest-negative —
the planner MUST:

1. **Reserve the majority of each milestone for depth, not breadth.** Allocate
   tasks toward the existential link tests (P0.1 energy-descent-vs-AR, P0.2
   verifier-diversity, the transpilation round-trip) and toward **closing G2**
   (one independent reproducer of the FoVer 0.9131 headline — the SOLE unmet
   publication gate per `ops/north-star.md` §2). These advance the headline or
   the finish line; everything else is candidate churn.

2. **Do NOT propose a `vN+1` re-measurement of an already-measured artifact**
   unless the new version answers a question the prior version did not (a NEW
   corpus, a NEW seed regime that changes a CI, a NEW gate). "v39 because v38
   exists" is forbidden. A re-measurement that does not move the headline claim
   or a G-gate is churn (per `ops/north-star.md` §1: a milestone that produces a
   new version of an existing artifact without moving the headline is noise).

3. **Every milestone answers: does this milestone advance the headline claim,
   close a G-gate, or test a load-bearing-unproven link?** If the honest answer
   for a task is "no, it re-measures something we already know," drop it and
   propose depth instead.

**How to apply (planner-side).** When drafting `research-roadmap-next.yaml`,
audit each task against the three points above. A typical milestone should now
be weighted toward the P0.1/Kona/G2 work, not breadth sweeps. The planner is
Claude Opus 4.8 as of 2026-05-30 (highest-leverage call, switched off gemini
precisely so this prose discipline is followed reliably).

**Retirement.** This forcing function relaxes once P0.1 has a verdict AND G2 is
either met or has a concrete in-flight reproducer. Until then it preempts
routine breadth.

**Cross-references:**
- `ops/north-star.md` — the headline claim (§1), the G1–G4 gate (§2)
- `ops/known-issues.md` — P0.1 (exp3312) + Kona solve-rate gate + the corrigendum trail
- CLAUDE.md "Scope-Reduction-When-Flagged" — the sibling rule for explicit
  scope-reduction directives; this one is the standing depth-vs-breadth default

## ARC-AGI-3 IS a Live Hidden-Game Discovery Agent — Foundational Framing (MANDATORY)

**Origin:** 2026-06-21 operator, after the SAME framing was re-learned across multiple
sessions:

> "This has happened multiple times now. How do we capture the ask here so that we aren't
>  constantly re-learning that we are working on a live agent that is responsible for an
>  agent local loop that can discover how to solve hidden games given to it at challenge
>  submission time?"

Filed because sessions (and the conductor) repeatedly re-derive — and then contradict — WHAT
IS BEING BUILT. The triggering instance: this session's offline TTT solve-test concluded
"goal-induction is the bottleneck" from `n_win_states=0`, which was a STATIC-CORPUS ARTIFACT
(the captured corpus has `level_before=level_after=0` everywhere), NOT a property of the LIVE
process, which streams `level_progress` per frame. The immediately-prior correction ("the value
is the PROCESS, not the trained weights") was the same class of error. This rule exists so the
framing is read as required input every session and stops being rediscovered.

**The framing (settled — do NOT re-derive it).** The ARC-AGI-3 deliverable is a **LIVE AGENT
running a LOCAL LOOP that, at challenge-submission time, is handed games it has NEVER SEEN and
must DISCOVER how to solve them on the fly** — inducing each unseen game's perception
(glyph/grid encoding), its dynamics (action→next-state), AND its goal (win/level-up condition)
AT RUNTIME from its own exploration. The scored Kaggle games are HIDDEN/OOD by design; the agent
gets zero prior exposure to them.

**What is therefore NOT the deliverable (the recurring errors this rule kills):**

1. **Trained model weights are not the live mechanism.** A model (CNN dynamics prior, TRM, etc.)
   trained on the 25 PUBLIC games does not transfer its weights to a hidden game's specific
   mechanics. Weights are an incidental artifact; the reusable METHOD/loop is the asset.
   (See `feedback_arc_value_is_process_not_weights` memory.)
2. **Public-game solves / offline replays are not the scored deliverable.** Banked replays of
   public games score ~0 on the hidden Kaggle leaderboard. The registry's reproduced-level count
   is a development proxy, not the scored target.
3. **Offline measurements are PROXIES for the live process — an offline null may be a
   HARNESS/CORPUS ARTIFACT, not a true limitation.** Before concluding "the agent can't do X"
   from an offline run, verify the offline harness/corpus faithfully reproduces what the LIVE
   agent sees. The `n_win_states=0` trap (the corpus dropped `level_progress`) is the canonical
   example: the live env exposes the signal the static corpus lacked.

**How to apply (every ARC session — planner, agents, outer-loop):**

- Evaluate ARC work by whether the **live PROCESS discovers + solves a FRESH unseen game**, not
  by a weight-transfer metric or a public-game replay count.
- Build/capture reusable **methods** (runtime dynamics induction, runtime goal induction,
  verifier-routed search, the reproduction gate) into `arc_solver_kit.py` +
  `ops/arc_solve_registry.yaml`, applied fresh to each new game — never a per-game trained weight
  as the live mechanism.
- When an offline result is null/negative, FIRST rule out that it is an artifact of the static
  offline setup (missing signal, dropped field, corpus≠live) before reporting it as a capability
  limit.

**Cross-references:** 2026-06-21 operator directives (origin) ·
`feedback_arc_value_is_process_not_weights` + `project_arc_agi3_north_star` (memory) ·
`ops/north-star.md` §0 · CLAUDE.md "ARC Solve Reproducibility + Solver-Reuse Discipline" (the
reusable-scaffolding mechanism this framing mandates) · "ARC-AGI-3 Submission Sprint Forcing
Function" · `docs/research-notes/arc-gate-readiness-prior-ttt-2026-06-21.md` (the corpus-artifact
incident).

## ARC-AGI-3 Incremental-Progress Scoping (MANDATORY)

**Origin:** 2026-06-09 operator directive after the FIRST ARC-AGI-3 solves landed
(r11l L1 in 4 actions, exp3946; lp85 L1, exp3954) but the `.366` task exp3953
("r11l FULL solve: take r11l from 1/6 to all 6 levels") 3-fail-SKIPPED across the
whole milestone:

> "perhaps we should be trying to solve one or more additional levels of a
>  particular game for each experiment rather than trying to solve them all.
>  progress is key."

The all-or-nothing framing is exactly why exp3953 stalled: a task that swings for
all 6 levels and lands 0 is strictly WORSE than a task that targets +2 and banks +1.

**The rule.** Every ARC-AGI-3 SOLVE experiment targets **incremental level-progress
on ONE game** — solve one or more ADDITIONAL levels beyond the prior milestone's
recorded best for that game — NOT a "FULL solve" / "all N levels" / all-games solve
in a single task. **Progress is the metric: each milestone must MONOTONICALLY advance
the total solved-level count, even by +1.**

**Falsifiable per-experiment gate.** The acceptance gate for a solve task is "at
least one NEW level solved, real-env-confirmed (`levels_completed` from the live
env), beyond the prior best for that game." A solve task whose gate is "all 6 levels"
is malformed — re-scope it to "+1..+n levels from L\<k\>".

**Planner-side scoping (when generating `research-roadmap-next.yaml`).**
- Read the recorded solved-level state (research-complete.yaml / the prior capstone /
  the game's last solve artifact) and scope each solve task as
  **"advance \<game\> from L\<k\> to L\<k+1..k+n\>"** (small n, typically 1–3) or
  **"first solve of \<new non-spatial game\> L1"** (the 6 non-spatial games from the
  win-condition survey, results/arc3_win_condition_survey.json, are the targets).
- Prefer breadth-of-progress across a milestone: several small "+1–2 levels" tasks on
  different games beat one "solve everything" task. A milestone that banks +1 level on
  three games (net +3) is better than one that attempts a full game and lands 0.
- NEVER emit a "FULL solve" / "all levels" / "solve them all" solve task. Mechanical
  re-scope: if a draft solve title contains "FULL", "all N levels", or "to 6/6",
  rewrite it to the next 1–3 levels.

**Why this is in CLAUDE.md.** The planner reads CLAUDE.md as required input on every
plan generation. Putting the incremental-scoping principle here ensures every future
ARC milestone advances the solved-level count monotonically rather than stalling on
over-ambitious all-levels tasks. This is the ARC-specific specialization of the
Depth-Over-Breadth / progress-not-churn ethos (north-star §1).

**Cross-references:**
- 2026-06-09 operator directive — origin
- exp3946 (r11l L1), exp3954 (lp85 L1) — the incremental solves that landed
- exp3953 ("r11l FULL solve") — the all-levels task that 3-fail-skipped (the anti-pattern)
- results/arc3_win_condition_survey.json — the 6 non-spatial first-solve targets
- `ops/north-star.md` §1 — the headline-progress / no-churn ethos this specializes

## ARC-AGI-3 Submission Sprint Forcing Function (RETIRED 2026-06-30 — preserved per never-prune)

> **STATUS (2026-06-30): RETIRED — both retirement conditions met.** It is now 2026-06-30 (the
> deadline date) AND the operator explicitly lifted it ("let's unlock the conductor running PHASE D
> experiments immediately"). The CUDA submission package is staged + the runbook committed
> (`docs/research-notes/arc-agi3-cuda-submission-runbook-2026-06-30.md`); the score is bounded at ~0.08
> by the generation wall and no lever moves it before the deadline (this session's full survey). So the
> majority-ARC reservation + the "do NOT execute the real PHASE D experiment" hold are LIFTED.
>
> **Effect — the conductor now EXECUTES PHASE D (the off-ARC distributional-energy verifier moat).** The
> planner's majority shifts from ARC live-solving to the post-6/30 verifier-moat program: TRAIN the real
> `arXiv:2605.18871` LoRA-EBM holistic-quality scorer (NOT the cheap prompted proxy, which was a clean
> negative on MuSR — `results/distributional_energy_verifier_musr.json`, SC 0.56 best, headroom present
> but UNrealized), REPLICATE uPRM (`arXiv:2605.10158`, the published beats-SC-6.9% process verifier), and
> the EBRM (`arXiv:2504.13134`) construction — on a headroom-present ORACLE-DISTINCT domain. This is
> DISTINCT from the CONCLUDED ARC-energy S0 program (`ops/known-issues.md` 2026-06-26; oracle-distinct
> structural energy ON ARC adds no live ARC value — do NOT re-propose S4/ARC-energy stages). PHASE D is
> OFF-ARC (reasoning corpora where SC is not saturated), so it is not that retired direction.
>
> **Reserved slots still apply** (2 infra, 1-per-board hardware-continuity, SOTA-ingestion). ARC work
> continues OPPORTUNISTICALLY (banked levels still count) but no longer claims the milestone majority.
> Per the never-prune rule the original prose is preserved unchanged below as historical context.

**Origin:** 2026-06-19 operator directive: "The next 2 weeks will be focused on solving these games live
as we have until the end of this month to make our submissions for the challenge contest... keep claude in
as the retro and planner as we want the quality and we do have claude quota if we keep the experiments
using codex instead." Filed to keep the planner on the ARC-AGI-3 live-submission path across EVERY
milestone until the deadline — not just the one pre-staged `.409 roadmap. The planner has a strong
carry-forward / research-breadth bias (cf. the retired Depth-Over-Breadth Forcing Function and the
Scope-Reduction-When-Flagged incident); a single pre-staged roadmap steers ONE milestone, but the
ARC-AGI-3 Kaggle submission deadline is **2026-06-30**, so the priority must persist for ~2 weeks of
milestones. This forcing function is the durable mechanism (the planner reads CLAUDE.md as required input
every plan generation).

**The rule.** Until **2026-06-30** (the challenge deadline — the operator expects to submit MULTIPLE times
before then, so the priority CONTINUES across submissions; do NOT treat a submission as retirement), the
planner MUST, every milestone:

1. **Allocate the MAJORITY of each milestone to ARC-AGI-3 live-game-solving progress** — the generic
   first-contact solver, verifier-grounded config-rule induction, glyph/rewrite perception, multi-level
   deepening, and unseen-game transfer-routing. The operative metric is **`reproducible_total_levels`
   must monotonically grow** (currently 34; see `ops/arc_solve_registry.yaml`). A milestone that produces
   no new reproducible level AND no concrete unblock toward one is churn (north-star §1).

2. **Every ARC SOLVE task is reproduction-gated and Incremental-Progress-scoped** — `+1..+n` levels on ONE
   game, `offline_reproduced=true` via `arc_solver_kit.reproduce` (only reproduced levels count), and feeds
   reusable scaffolding back into `arc_solver_kit` + `ops/arc_solve_registry.yaml` so the LIVE solver
   applies it to games it has never seen (per ARC-AGI-3 Incremental-Progress Scoping + ARC Solve
   Reproducibility, below).

3. **Agent routing for the sprint (operator-fixed 2026-06-19):** ALL experiments stay `agent_type: codex`
   / `gpt-5.5` (the quota-conserve bulk — `CODEX_FORCE_EXPERIMENTS=1`). The **planner and retro STAY on
   Claude Opus 4.8** (`AGENT_TYPE_PLANNER`/`AGENT_TYPE_RETRO=claude`) — the operator's explicit choice:
   "we want the quality and we do have claude quota if we keep the experiments using codex." Do NOT flip
   planner/retro to codex during the sprint.

4. **The live-submission stack is FROZEN** for the sprint: generator = **Qwen3.5-9B-MTP** on the iGPU
   (NEVER the 3090s) + MTP + q8 KV + `n_predict>=2048` + `/no_think` ([[project_arc_live_generator]]);
   Kaggle engine = the CUDA-12.8 `llama-server` binary (NOT a wheel — MTP is in libllama-common). Planner
   tasks build ON this stack; do not re-litigate model selection (settled 2026-06-19).

**GPU allocation — the "iGPU NEVER 3090s" rule applies to the LIVE submission stack ONLY (operator
directive 2026-06-27).** The iGPU-only constraint in rule 4 above exists for KAGGLE-PARITY: the scored
eval is iGPU/~16GB-class, so the LIVE submission generator must run there. It does NOT govern OFFLINE /
dev induction (fork probes, generic-solve world-model induction, the LocalGGUFProposer at dev time). Per
the 2026-06-27 operator allocation, the two discrete RTX 3090s are dedicated: **the CONDUCTOR owns GPU 0**
(its offline ARC generator runs there via `CARNOT_ARC_GENERATOR_CUDA_GPU=0`, the systemd drop-in
`40-arc-generator-3090-20260619.conf`), and **the OUTER LOOP owns GPU 1** (`CUDA_VISIBLE_DEVICES=1` for its
own experiments). This stops GPU contention and removes the ~4 tok/s iGPU throughput bottleneck for novel
induction. **PLANNER CONSEQUENCE:** offline induction tasks must NOT hardcode an `igpu_required: True` /
`cuda_3090_generator_disallowed` precondition — that was sourced from rule 4's live-stack constraint and
WRONGLY blocks the conductor's own dedicated GPU-0 generator (cf. the `.448` exp4861 fork-probe non-test).
Offline induction uses GPU 0 (conductor) / GPU 1 (outer loop); only the LIVE submission generator is
iGPU-pinned.

**Reserved slots still apply.** The 2 infra slots, 1-per-attached-board hardware-continuity slot, and the
SOTA-ingestion slot are still reserved each milestone (those disciplines are not suspended) — the
"majority" in rule 1 is of the REMAINING slots.

**Retirement.** This forcing function retires on **2026-06-30** (the end of the challenge), OR when the
operator explicitly lifts it. It does **NOT** retire when submissions are made — per the 2026-06-19
operator directive the operator may submit MULTIPLE times before the deadline, and the priority continues
(keep solving + improving the live solver across every submission) until 2026-06-30. After retirement the
planner resumes normal research breadth (the verifier-moat / cross-domain directions). Preserve per never-prune.

**Why this is in CLAUDE.md, not just `research-roadmap-next.yaml`.** A pre-staged roadmap steers ONE
milestone and then is consumed; the planner reads CLAUDE.md on EVERY plan generation. Putting the sprint
priority here makes the planner reserve the majority for ARC live-solving at design time for all of `.410+`
through the deadline, without the outer-loop re-staging each milestone. The matching per-milestone trigger
is the `ops/known-issues.md` MANDATORY-NEXT-MILESTONE entry (backed by the Overdue-Priority Forcing
Function), with this rule as the design-time authority.

**Cross-references:** 2026-06-19 operator directive (origin) · `ops/north-star.md` §1 (no-churn ethos) ·
CLAUDE.md "ARC-AGI-3 Incremental-Progress Scoping" + "ARC Solve Reproducibility" (the per-task mechanics) ·
CLAUDE.md "Depth-Over-Breadth Forcing Function" (RETIRED; the structural precedent) · `[[project_arc_live_generator]]`
(the frozen generator) · `docs/research-notes/arc-agi3-kaggle-submission-requirements-2026-06-17.md` (packaging).

## ARC Level-Up Attempt Guarantee (MANDATORY)

**Origin:** 2026-06-19 operator directive — "are we attempting to solve any game levels this roadmap?
... how do we ensure we ALWAYS have at least one level-up attempt across all games once a roadmap."
The ARC-sprint headline metric is `reproducible_total_levels` growing monotonically, but a planner can
drift to all-META work (library induction, LOO benchmarks, registry/gaps hygiene, SOTA ingestion,
submission packaging, generic-operator *re-solves* of already-banked levels) and ship a roadmap with
ZERO concrete level-BANK attempts — pure churn (north-star §1). The Submission Sprint Forcing Function
mandates the *majority* be ARC-solving and that levels grow, but "majority ARC" can be satisfied by
meta tasks; this rule adds the missing floor.

**The rule.** Every ARC-sprint roadmap (`.412+` through 2026-06-30, and any ARC-headline milestone
after) MUST contain **at least one LEVEL-UP ATTEMPT**: a task whose acceptance gate requires actually
BANKING a new reproducible level — `offline_reproduced=true` AND a new-level condition
(`reproduced_levels>=1` on a first-contact of an UNSOLVED game, OR a NEW deeper level `level > N` on an
already-solved game). A generic-operator *re-solve* of an already-banked level (generalization
validation), a benchmark, a library induction, or a hygiene/ingestion/packaging task does **NOT**
count. Default floor = 1; raise it when the milestone is a pure solving push.

**Target ROTATION (the "across all games" half).** The planner SHOULD rotate the level-up targets so
every game gets a periodic attempt rather than re-attempting the same one while others are neglected:
prioritize the UNSOLVED public games (currently re86/sb26/bp35/lf52) for first-contact, plus the
shallow-solved games for +1-deeper, so coverage sweeps the corpus over successive milestones. The
lint prints the games targeted each roadmap so rotation is auditable.

**Mechanical enforcement.** `scripts/arc_levelup_guarantee_lint.py [roadmap.yaml] [--min N]` counts the
level-up attempts and exits non-zero if `< --min` (default 1). Run it on every emitted roadmap; the
activation guard SHOULD refuse a milestone that fails it (pending wiring, it is planner-discipline +
outer-loop audit). Validated 2026-06-19: `.412 passes (4 attempts: dc22 solve, sc25 deepen,
first-contact a new game, tr87 generic-with-new-game); an all-meta roadmap fails with exit 1.

**Why this is in CLAUDE.md.** Same defense-in-depth as the other ARC forcing functions: the planner
reads CLAUDE.md as required input, so it includes a level-up attempt at design time; the lint is the
backstop. Without it, a meta-heavy milestone silently produces no banked level = churn.

**Cross-references:** 2026-06-19 operator directive (origin) · `scripts/arc_levelup_guarantee_lint.py`
· CLAUDE.md "ARC-AGI-3 Incremental-Progress Scoping" (per-task +1 scoping; this is the per-ROADMAP ≥1
floor) · "ARC-AGI-3 Submission Sprint Forcing Function" (majority-ARC + monotonic growth) ·
`ops/arc_solve_registry.yaml` `reproducible_total_levels` (the metric this protects).

## ARC Solve Reproducibility + Solver-Reuse Discipline (MANDATORY)

**Origin:** 2026-06-16 operator directive after the outer-loop spent a long
session making the deeper sc25/lp85 ARC solves offline-reproducible:

> "How do we prevent having to go through this again? ... we must require
>  capturing the winning conditions for reproducibility ... otherwise the effort
>  is effectively wasted. ... if we solved this before, shouldn't we have the
>  scaffolding in place to apply what we learned previously for future efforts?"

**What went wrong (the waste this prevents).** ARC solves were banked as
live-recorded `solve_trace.actions` — frozen pixel-coordinate trajectories
coupled to the LIVE env layout. They replay to **0 levels** on the offline
`environment_files` env (the coords miss; cf. sc25's hardcoded `SC25_GRID_COORDS`).
So the *winning condition* (the search/mechanic that derives a solution for the
actual env) was never captured — only one brittle trajectory was. Of our 11
claimed levels, only **6 reproduce offline**; the deeper sc25 L5 + lp85 L3 had to
be **re-derived from scratch** months later. That re-derivation is the wasted
effort this rule prevents.

**The rule — two halves:**

1. **Reproducibility is a GATE, not an afterthought.** An ARC solve does NOT count
   toward `total_levels`, and is NOT headline/publish-eligible, until it passes an
   **offline reproduction gate**: a from-scratch solver (or a state-grounded,
   env-discovered replay) re-derives the claimed level against the OFFLINE env,
   verified by `python/carnot/agentic/arc_solver_kit.py:reproduce`. A solve backed
   only by a live-recorded coordinate trajectory is `provisional` — real but
   uncounted until reproduced. Every solve experiment MUST emit
   `offline_reproduced: bool` + `reproduced_levels: int`, and MUST persist its
   solver/win-condition (not just the trajectory) to `ops/arc_solve_registry.yaml`.

2. **Capture and REUSE the learning (scaffolding, not one-offs).** Per-game
   mechanics we reverse-engineer are durable assets, not buried in experiment
   chains. Reusable primitives go in **`arc_solver_kit.py`** (offline arcade,
   replay-from-reset BFS, warm-up-after-reset, frame-based level, facing-aware
   dedup, env-adaptive discovery, the reproduction gate). Per-game win-conditions
   + action-models + gotchas + solver module + reproducibility status go in
   **`ops/arc_solve_registry.yaml`**. A new game's solver MUST start from the kit +
   the registry's general gotchas — never re-derive from zero. New general gotchas
   discovered while solving a game MUST be added to both (kit docstring + registry
   `general_gotchas`).

**How to apply (planner-side).** Every ARC solve task's REQUIRED ARTIFACT FIELDS
include `offline_reproduced: bool` (principle: "a solve not reproducible offline
is wasted effort — only reproduced levels count") and `reproduced_levels: int`.
The task prompt MUST direct the agent to (a) build the solve as a from-scratch
solver over the offline sim reusing `arc_solver_kit`, (b) run the reproduction
gate, (c) update `ops/arc_solve_registry.yaml`. ARC `total_levels` claims cite the
registry's `reproducible_total_levels`, never the provisional count.

**STANDING ENTRYPOINT (wired into the loop, 2026-06-16).** The ARC north-star
solve task's concrete step is to run the standing loop:
`.venv/bin/python scripts/arc_loop_solve.py --game <target>` (or `--auto`). It
does the whole loop automatically and self-improves across runs: transfer-routing
→ verifier-routed best-first solve (warm-started by a saved LEARNED verifier if
present, else the hand verifier) → reproduction gate → train+checkpoint the
learned verifier (`models/arc_verifier_<game>.json`, mirror-ready per Rule 3) →
registry update → milestone artifact. For an **adaptered** game it solves +1
level and emits `offline_reproduced`/`reproduced_levels`. For an **un-adaptered**
game it emits the transfer-routing recommendation + general gotchas — the agent's
per-game RE starting point: the agent reverse-engineers only the win/action/state
DELTA, registers a `GameAdapter` in `python/carnot/agentic/arc_game_adapters.py`,
and re-runs the loop. (Validated 2026-06-16: lp85 solves L3, reproduction-gate
passes, 359-state verifier-routed search; tn36 routes correctly + flags the delta.
The irreducible per-game cost is the DELTA RE only — the harness, search engine,
verifier training, gate, and routing are all reused.)

**How to apply (agent-side).** Before solving a NEW game, FIRST call
`python/carnot/agentic/arc_solve_learning.py:recommend_approach(game)` — it routes
the game to the closest SOLVED game's proven recipe (solver module, action-model,
reusable gotchas) by survey-feature similarity, so you reverse-engineer only the
DELTA instead of starting from zero (sc25 took ~10 layers precisely because this
routing did not exist). Then import `arc_solver_kit`, reuse the recommended recipe
+ the `general_gotchas`. After solving, run `arc_solver_kit.reproduce(...)`; only
on `reproduced: true` write the solve as counted, and append/update the game's
registry entry (win condition, action model, gotchas/dead-ends, solver module).
If a solve only exists as a live-recorded trajectory, mark it `provisional` — do
NOT count it. **Record DEAD-ENDS** (a search approach that stalled + why) in the
game's registry entry so the next attempt skips them — learning from failures,
not just successes.

**Why this is in CLAUDE.md.** The planner + agents read CLAUDE.md as required
input. Putting the gate + reuse contract here makes every future ARC solve
self-documenting and self-reproducing at design time, so the months-later
re-derivation never recurs. This is the ARC-specific instance of the project's
self-learning ethos: knowledge captured as reusable scaffolding, verified by a
gate, compounding across games.

**Cross-references:**
- 2026-06-16 operator directive — origin
- `python/carnot/agentic/arc_solver_kit.py` — reusable primitives + reproduction gate
- `ops/arc_solve_registry.yaml` — per-game mechanics + reproducibility knowledge base
- `scripts/arc3_replay_scorecard_metaharness.py` — the aggregate offline reproduction harness
- `docs/research-notes/arc3-offline-reproducibility-audit-2026-06-16.md` — the audit that motivated this
- CLAUDE.md "Adversarial Artifact Verification" + "Missing-Verifier Gap Logging" — sibling capture-don't-waste disciplines
- CLAUDE.md "ARC-AGI-3 Incremental-Progress Scoping" — the progress rule this makes durable

## ARC Live-Path Reachability Discipline (MANDATORY)

**Origin:** 2026-06-22 operator, after a session built a hazard-aware nav world
model (`arc_nav_world_model`: induce a nav model from transitions → detect a
charging-enemy from the agent's own deaths → route safe detours) across several
milestones to "solve" tu93's charger levels:

> "Is the mechanism we are using here in the outer loop the same mechanism that
>  the live agent uses to find solves? Otherwise, it seems to me that we are
>  intentionally wasting the additional effort we are spending doing this process."

The audit (workflow `arc-mechanism-parity-audit`, verdict `different_mechanism`,
adversarially `synthesis_holds`) found the worry fully justified:

1. **Orphaned from the live path.** `arc_nav_world_model` /
   `HazardAwareNavWorldModel` / `escalating_deepen` were imported ONLY by
   `scripts/experiments/*` + tests — by ZERO live-agent files. Neither live
   entrypoint's import closure contained them.
2. **It re-solved an already-solved level.** tu93's live `GameAdapter` (`_tu93`
   in `arc_game_adapters.py`) already DEEP-SOLVES tu93 to **L3 reproducibly**
   (registry: L5) via plain verifier-routed best-first search over the 4 nav
   actions + a player→goal Manhattan-distance verifier. The hazard model bought
   no new capability; walking into a charger is just a dead-end the search
   prunes by exploration.
3. **Its calibration didn't transfer.** The `omni` lethal rung was validated by
   an EXHAUSTIVE real-env BFS over tu93 L3 ground-truth labels — a thing a
   hidden-game agent cannot run under an action budget.

**The two LIVE entrypoints (memorize these — "the live agent" means reachable
from one of them):**

| Entrypoint | Role | Mechanism |
|---|---|---|
| `python/carnot/agentic/arc_competition_agent.py` (`make_carnot_agent` → `E3AgentPolicy`) | the **SCORED** Kaggle/hidden-game agent | per-action verifier-routed cascade over its OWN transitions: StepwiseExplorer → online world-model induction (`arc_live_ttt` / `LocalGGUFProposer`) gated by `WorldModelVerifier` → `e3.plan_in_model` |
| `scripts/arc_loop_solve.py` (`OfflineSolver` + `GameAdapter`) | the **offline development twin** (registry/dev) | raw-action verifier-routed best-first search replayed in the offline sim, reproduction-gated |

**The rule (two halves):**

1. **Registry-precheck BEFORE any per-game RE / world-model solver work.** Before
   building ANY solver/world-model for a game, check whether the live mechanism
   already reaches the target level: run `arc_loop_solve.py --game X
   --target-level N` (or read `ops/arc_solve_registry.yaml` `levels_reproduced`
   + `adaptered_games()`). **If the live mechanism already reaches the level, do
   NOT build a parallel solver for it.** Improve the live path instead (a better
   `GameAdapter`, a verifier feature, a `plan_in_model` engine) — never a
   separate planner the live agent can't call. This is the sharper, mechanical
   form of the sibling "ARC Solve Reproducibility + Solver-Reuse Discipline"
   ("reuse the kit; don't re-derive from zero").

2. **Every ARC solver/world-model module MUST be live-path-reachable.** Any
   module under `python/carnot/agentic/arc_*` that is solver-like (name matches
   `*world_model*`, OR defines `escalating_deepen` / `plan_in_model` / a class
   with both `.engine` and `.is_lethal`) MUST be in the transitive import
   closure of the two live entrypoints, OR be explicitly allow-listed with a
   reason. A solver mechanism the live agent cannot reach produces no live
   capability and no live efficiency — it is wasted effort by construction.

**What "salvage" looks like (the 2026-06-22 resolution).** The hazard work was
salvaged as an **efficiency** contribution ON the live path, not a parallel
solver: `arc_hazard_pruner.HazardMovePruner` fits a hazard model from the
search's OWN observed deaths (no offline BFS; rung selected by in-sample
observed-transition trust, used as a conservative gate) and is consumed by
`OfflineSolver` as a move-pruner.
Measured A/B on tu93 L3: states_expanded 2947 → 2859 (−88, the exact pruned-move
count), L3 solve preserved (reproduced, specificity 1.0). Modest but real, and
now reachable from `arc_loop_solve` (so the lint passes). The natural next step
is to feed the same pruner / `HazardAwareNavWorldModel.engine` into the SCORED
`E3AgentPolicy.plan_in_model` path.

**Self-solve provenance contract (2026-06-22, the 2nd-recurrence hardening).** The
deeper principle the operator restated: *"We want to help the LIVE agent find ways
of solving hidden games on its own — it needs to generate solves on its own and
solve the hidden game levels based on its OWN attempts and RE of the game."* So an
outer-loop session reverse-engineering a game (reading its source, running an
exhaustive offline ground-truth BFS, hand-building a per-game model/adapter) and
"solving" it is the ANTI-PATTERN even when the code is clean — that work does not
make the LIVE agent better at self-discovery. Every ARC solve artifact (any
artifact with `offline_reproduced: true` + a `game` + a level ≥ 1) MUST declare
`solve_provenance`:

| value | meaning | eligibility |
|---|---|---|
| `live_agent_self_discovery` | the live agent advanced via its OWN attempts + runtime RE (adapter-free explore/induce, or the scored E3 cascade) | headline / credit-eligible |
| `development_proxy` | the offline dev twin (`arc_loop_solve` + a hand-registered `GameAdapter`) — a dev/registry proxy | allowed, but NOT proof the live agent self-discovers |
| `outer_loop_re` | a human/outer-loop hand-RE or off-path solve | NOT the deliverable; flagged CRITICAL; never headline |

Declaring outer-loop-only inputs (`used_env_source`, `read_game_source`,
`offline_ground_truth_bfs`, `exhaustive_bfs_calibration`, `hand_calibrated_per_game`)
on a non-`outer_loop_re` solve is a CRITICAL contradiction. Re-solving a level the
registry already records is a CRITICAL duplicate.

**Source-reading is a PUBLIC-games dev tool ONLY — NEVER in the hidden live submission
(operator directive 2026-06-29).** Reading the game source (`environment_files/<gid>/*.py`),
probing the sandbox for source exposure, importlib-exec'ing the game sim, `set_level`
teleporting, or reading the ground-truth win flag is **PERMITTED for the 25 PUBLIC games in
the conductor + outer loop** (offline dev: understanding mechanics, building `GameAdapter`s,
validation via `offline_arcade` over `environment_files`) — this is normal `development_proxy`
work. It is **FORBIDDEN as any part of the LIVE AGENT's HIDDEN-game submission.** The scored
agent must solve hidden games purely from its OWN runtime discovery (frames + exploration +
induction); it must never read hidden-game source or ship a source-exposure probe to the
scored eval. Rationale: (1) the deliverable is a live agent that DISCOVERS hidden games on its
own — source-reading on hidden games is the anti-pattern, against the benchmark's stated design
intent ("discover what winning looks like"); (2) it is metric-penalized anyway (RHAE squares
efficiency; a brute-force teleport-solver scores LOW); (3) it carries discretionary-DQ /
overfitting risk. The operator settled this WITHOUT needing an empirical probe of the hidden
sandbox: we will not read/probe hidden-game source regardless of whether the scored env exposes
it. Do NOT build or stage a hidden-eval source-exposure probe submission. (Context: the
investigation found the scored path is a frames-only remote gateway anyway — `submission_kernel`
runs `OPERATION_MODE=online` against `gateway:8001`, hidden source server-side — so source-
reading is also inviable on the hidden set; the directive stands independent of that.)

**Three-layer adversarial defense (aggressive — shipped 2026-06-22).**
- **Layer 1a — commit-time HARD STOP:** `scripts/arc_orphan_solver_lint.py`
  (pre-commit `arc-orphan-solver-lint`, on any `python/carnot/agentic/arc_*.py` or
  `scripts/arc_loop_solve.py` change) computes the live import closure (absolute,
  relative, and function-level imports) and refuses the commit if a solver-like
  module under `python/carnot/agentic/` is orphaned. Allow-list is one reasoned
  entry (`arc_execution_guided_world_model`). *Scope:* it scans agentic *modules*
  — a self-contained solver written entirely under `scripts/experiments/` is NOT
  scanned here; its solve ARTIFACT is what Layer 1b catches (below).
- **Layer 1b — per-artifact mechanical:**
  `adversarial_verify.check_arc_outer_loop_solve` flags ARC solve artifacts:
  CRITICAL on an offline-ground-truth-BFS / per-game CALIBRATION solve (keyed on
  the experiment NAME + a game-solve claim — this is the honesty-INDEPENDENT catch
  for the 2nd-recurrence incident, which made a prose tu93-L3 solve claim with no
  structural solve fields); CRITICAL on `outer_loop_re`, an outer-loop-input
  contradiction, or a registry duplicate; WARN on undeclared `solve_provenance`.
  CRITICAL → `flagged_adversarial` (the fabrication gate quarantines it, so an
  outer-loop "solve" never logs a clean milestone success).
- **Layer 2 — milestone-close AI audit:** `scripts/arc_self_solve_audit.py`
  (wired into `_run_operational_retrospective`) — a mechanical pre-pass
  (reachability + provenance) plus a hostile LLM review asking, per recent ARC
  solve artifact, `SELF_DISCOVERY_ADVANCE | OUTER_LOOP_RE | OFF_PATH | DUPLICATE |
  UNCLEAR`. Writes `ops/arc_self_solve_audit_report.md`; never edits.
- **Layer 3 — this CLAUDE.md rule** (design-time contract the planner reads).

**Residual (honest — not a closed system).** The mechanical layers are strongest
against (i) off-path agentic solver *modules* (Layer 1a) and (ii) the
offline-calibration / declared-outer-loop / duplicate solve *artifact* (Layer 1b,
honesty-independent for the calibration class). They are WEAKER against a
determined/careless session that writes a self-contained solver under
`scripts/experiments/`, does NOT name its experiment `*calibration*`, declares no
outer-loop input flags, and stamps `solve_provenance: live_agent_self_discovery`
on a brand-new game — that case is caught only by Layer 2's hostile LLM review (and
by an alert human reading the report), not mechanically. So: the provenance stamp
is a contract, not a guarantee; do not treat a green Layer 1 as proof the live
agent self-discovered a solve. When in doubt, the question is always "could the
LIVE agent reproduce this on a hidden game from its own attempts?" — if no, it does
not count, whatever the stamp says.

**How to apply (planner + agent).** When a roadmap proposes ARC solve work:
(1) register the per-game knowledge as a `GameAdapter` / feed it into the live
`E3AgentPolicy` path — NOT a standalone experiment the live agent never imports;
(2) before proposing "solve game X", confirm via the registry that X is not
already solved at the target level; (3) add `solve_provenance` to the task's
REQUIRED ARTIFACT FIELDS, and prefer `live_agent_self_discovery` work (the agent's
own attempts + runtime RE) over outer-loop RE. Outer-loop RE that the live agent
cannot reproduce on a hidden game is not progress toward the deliverable.

**Cross-references:**
- 2026-06-22 operator directives — origin (parity audit + "aggressively caught and stopped")
- workflow `arc-mechanism-parity-audit` — the parity audit
- `scripts/arc_orphan_solver_lint.py` + `.pre-commit-config.yaml:arc-orphan-solver-lint` — Layer 1a
- `scripts/adversarial_verify.py:check_arc_outer_loop_solve` — Layer 1b (per-artifact)
- `scripts/arc_self_solve_audit.py` + `ops/arc_self_solve_audit_report.md` — Layer 2 (milestone-close)
- `python/carnot/agentic/arc_hazard_pruner.py` — the salvage (hazard as a live-path move-pruner)
- CLAUDE.md "ARC Solve Reproducibility + Solver-Reuse Discipline" — the sibling reuse rule this sharpens
- CLAUDE.md "ARC-AGI-3 IS a Live Hidden-Game Discovery Agent" — the foundational framing this enforces

## Missing-Verifier Gap Logging (MANDATORY)

**Origin:** 2026-06-09 operator directive — "make note of any 'missing' Carnot ARC verifiers
that might help improve our results; this will help us improve the core function of Carnot as
a verifier, which has become the point of this project."

Carnot's value-add IS the verifier. Every case where a verifier CANNOT select the correct
answer — because no existing invariant/energy captures the discriminating signal — is a
**missing-verifier spec**, and filling those specs is the product. The complement of
`ops/verifier_registry.yaml` (the verifiers we HAVE) is `ops/verifier_gaps.md` (the verifiers
we NEED).

**The rule.** Any experiment that evaluates or reranks with a verifier and finds a
*present-but-unselectable* failure — an oracle ceiling the verifier can't reach, a distractor
class at ≈chance discrimination, a task family where every applicable family abstains — MUST
append a gap entry to `ops/verifier_gaps.md` (schema in that file): the failure mode, the
**missing discriminator** (the signal a new verifier would need to compute), a candidate
design, and a priority by the headroom it would unlock. Verifier experiments should therefore
EMIT a `missing_verifier_gaps` field in their artifact characterizing the residual failures,
not just a headline score.

**Planner-side.** When generating roadmaps, treat the open entries in `ops/verifier_gaps.md`
as a first-class build backlog: queue verifier-building tasks (new invariant families,
ARC-domain energy instances) against the highest-priority open gaps. Closing a gap (a new
registry verifier that captures a previously-unselectable slice) is direct progress on the
project's core, and is recorded by moving the entry to `status: filled (<verifier_id>)`
(never-prune).

**Why this is in CLAUDE.md.** The planner + agents read CLAUDE.md as required input. A
verifier whose gaps are systematically logged and built-against improves monotonically — the
Phase-3 self-improving-verifier program made concrete. Cross-refs: `ops/verifier_gaps.md`,
`ops/verifier_registry.yaml`, `project_verifier_domain_bound` (the ARC-domain energy the deep
gaps call for).

## Overdue-Priority Forcing Function (MANDATORY)

> **STATUS (2026-05-29): MECHANICALLY ENFORCED — prose is reference-only.**
> `scripts/overdue_priority_lint.py` (pre-commit) refuses any roadmap/known-
> issues commit that leaves a priority at pending_count≥3 without pickup or
> `operator_override`. Read on only for the rationale. (See Rule Index at top.)

If a `ops/known-issues.md` "MANDATORY-NEXT-MILESTONE PRIORITIES" entry has been
pending for 3+ consecutive milestones without pickup, the next planner Sonnet
**MUST** include at least one of those entries as an experiment in its
roadmap, taking precedence over fresh research-breadth exploration.

The 2026-04-27 → 2026-04-28 sessions demonstrated the recurring failure mode:
the planner Sonnet has a strong attention bias toward research breadth and
will repeatedly skip operator-attention-reduction infrastructure work
(`conductor-supervisor.md`, `roadmap-schema-validation.md`,
`eval-metrics-canonical-and-self-heal-production-bug-detector.md` etc.)
even when those are explicitly marked as `NEXT-MILESTONE PRIORITIES` in
`ops/known-issues.md`. Three milestones in a row (.77, .78, .79) skipped
the supervisor proposal despite it being the load-bearing fix for repeated
log-handle-severance + commit-truncation incidents.

**Mechanic:** the conductor's `_plan_next_milestone()` planner-prompt MUST
include the section labelled `MANDATORY-NEXT-MILESTONE PRIORITIES` from
`ops/known-issues.md` *prefixed* with the count of milestones each priority
has been pending. Any priority with `pending_count >= 3` is a hard pickup
requirement; the planner cannot skip it without producing an explicit
written rationale in `research-roadmap-next.yaml` (which the activation
guard then checks for plausibility before activating the milestone).

**Reserved infrastructure slots:** every milestone with ≥10 tasks reserves
at least 2 slots for infrastructure-class work (supervisor, schema
validation, metric canonicalisation, audit scripts, etc.). The reservation
is enforced at planner-output time by the same activation guard.

**Why this is in CLAUDE.md, not just in known-issues.md:** the planner
reads CLAUDE.md as required context; a rule that lives only in
known-issues.md is advisory and routinely ignored. Mandatory rules need
this file's authority.

## SOTA-Ingestion Cycle Discipline (MANDATORY)

**Origin:** 2026-06-11 operator directive — "how do we add this deep-research
pattern to our regular loop cycle to encourage ingestion of SOTA ideas in our
bleeding-edge efforts." Filed after a manual outer-loop literature pass
(`docs/research-notes/search-layer-literature-2026-06-11.md`) mapped the
search/planning + RLVR-distillation SOTA directly onto the active `.372
search-layer pivot tasks — and found the decisive paper for the decentralization
fork ("The Invisible Leash", arXiv:2507.14843) that no experiment would have
surfaced.

**The gap this closes.** The loop already does literature *discovery* (the
`/study-sweep` cron + `scripts/sweep_*.py` arXiv / Semantic-Scholar helpers feed
`research-studying.md` / `research-references.md`). What was missing is
*ingestion*: turning discovered SOTA into actionable methods mapped onto the
milestone's bleeding-edge experiments, fed forward into the next roadmap.
Discovery without ingestion means the loop re-derives what the literature already
solved (a "shoulders of giants" violation — see the literature-priority memory).

**The rule.** Every milestone whose headline attacks a bleeding-edge open problem
(a new research track, not routine continuation) RESERVES one **SOTA-ingestion
task** scoped to that milestone's headline open problem — analogous to the 2
reserved infrastructure slots. The task:

1. **Reads the already-discovered corpus** (`research-studying.md` /
   `research-references.md`) filtered to the active track, PLUS runs a FOCUSED
   fresh pass via the RELIABLE channel: `scripts/sweep_clusters.py` /
   `sweep_semscholar.py` (arXiv / Semantic-Scholar APIs, Python-callable) for
   paper discovery + low-concurrency `WebSearch` + `WebFetch` of the top 5–8
   papers.
2. **Synthesizes a SOTA→experiment mapping artifact**: the strongest 3–5 methods,
   what each would take to implement over the CURRENT stack, and the pitfalls /
   where each fails. The template is `docs/research-notes/
   search-layer-literature-2026-06-11.md` (per-track note + a "bottom line for
   the roadmap" section).
3. **Closes the loop**: updates `research-studying.md` (mark ingested) and
   FLAGS the strongest method(s) as candidate inputs for the NEXT milestone's
   roadmap, so the planner reads the mapping and SOTA flows into experiments.
   Discover → ingest → plan → experiment.

**Reliable mechanism, NOT the fragile harness (load-bearing).** The autonomous
loop's SOTA-ingestion task MUST use the low-concurrency channel above. It MUST
NOT invoke the `/deep-research` skill (the 25–75-agent fan-out): on 2026-06-11
that harness rate-limited 4× (server-side 429s under its concurrency) and burned
~6M tokens for ZERO output, while direct sequential `WebSearch`/`WebFetch`
succeeded cleanly. `/deep-research` remains OPT-IN for the operator / outer-loop
in an interactive session where a throttle can be waited out — it is banned from
the autonomous loop.

**Guardrails (no fabrication).** A SOTA-ingestion artifact MUST cite real arXiv
IDs / URLs for every method claim. An ingestion artifact with no verifiable
citations, or that names methods without sources, is treated as fabrication by
`adversarial_verify.py` discipline (the same bar as any results artifact). The
two-source / pre-claim checklist from the literature-priority discipline applies.

**Cadence.** One ingestion task per milestone when the headline is a bleeding-edge
track (the current state: the `.372 search-layer pivot). When a milestone is pure
continuation / consolidation (no new track), the ingestion slot is optional — but
the broader `/study-sweep` cron keeps discovery running regardless.

**Why this is in CLAUDE.md.** Same defense-in-depth as the reserved-infra-slots
and Overdue-Priority rules: the planner reads CLAUDE.md as required input. A rule
here makes the planner reserve the ingestion slot at design time; the known-issues
MANDATORY-NEXT-MILESTONE entry is the per-milestone trigger; the activation guard
is the backstop. Cross-refs: `feedback_literature_priority_discipline` (memory),
`feedback_sweep_dedupe_protocol` (memory, the sweep helpers),
`docs/research-notes/search-layer-literature-2026-06-11.md` (the exemplar),
`feedback_sota_ingestion_cycle` (memory).

## Development Workflow (MANDATORY)

This project uses **spec-anchored development** (BMAD + OpenSpec). Every code change follows:

1. **Spec First** — Update `openspec/capabilities/*/spec.md` with new REQ-* and SCENARIO-*. Create/update story in `epics/stories/`.
2. **Write Tests** — Tests reference REQ-* and SCENARIO-* in comments.
3. **Implement** — Code to satisfy spec requirements.
4. **Verify** — Run unit tests, type checks, builds per commands below.
5. **E2E Verify (MANDATORY)** — Run end-to-end tests per `ops/e2e-test-plan.md`. All changes derived from user instruction MUST be verified E2E before reporting done. See E2E Testing below.
6. **Reconcile Specs** — Update Implementation Status in spec.md. Update story status. Update `_bmad/traceability.md` impl status column. If implementation diverged from spec, update spec to match reality with rationale.
7. **Update Ops** — Update `ops/status.md` (what's working/next) and `ops/changelog.md` (what you did).
8. **Update `_bmad`** — Update any part of `_bmad` that is relevant to the changes you made. Never leave specs and code disagreeing silently.

### Architecture Freshness Check

If `_bmad/architecture.md` "Last Reconciled" date is >30 days old, flag to user before starting new capability work.

### Documentation Update Rules (MANDATORY)

When updating `ops/status.md`, `_bmad/traceability.md`, or any ops/spec document:

1. **NEVER remove existing content without explicit user approval.** Completed work, historical results, and infrastructure descriptions must be preserved.
2. **ADD new sections** for new work. Do not replace existing sections with summaries that lose detail.
3. **Move items to "Completed" sections** rather than deleting them. If something was "What's Next" and is now done, move it to "What's Working" — don't delete it.
4. **Preserve historical results** (autoresearch runs, benchmark numbers, experiment data). These are the project's research record.
5. **Items in "Known Constraints" or "What's Next"** stay until explicitly resolved. If a constraint is fixed, mark it with ~~strikethrough~~ and add the fix date — don't delete the line.
6. **When rewriting a document**, first read the ENTIRE existing content and ensure every substantive item appears in the new version. If in doubt, keep it.

## E2E Testing (MANDATORY)

**Every change derived from user instruction must be verified end-to-end.** This means:

- **EBM models**: Full training + sampling pipeline producing statistically correct distributions
- **Cross-language**: Rust and Python implementations producing equivalent results for same inputs
- **Serialization**: Model saved in one language loads correctly in the other

E2E tests must exercise the full stack, not just unit tests. The test plan lives at `ops/e2e-test-plan.md` and results are documented at `ops/test-results.md`.

### Tests Must Run and Assert (MANDATORY)

Every test must have at least one assertion. Skipping tests (`pytest.mark.skip`, `pytest.mark.skipif`, `@unittest.skip`, or equivalent) is never allowed — skipped tests are invisible failures that accumulate silently and erode confidence in the suite. If a test depends on Docker/GPU/network, mock the dependency and test the logic. If a test genuinely cannot run in any environment, do not write it.

## Build / Test / Deploy

```bash
# Build (Rust)
cargo build --workspace --exclude carnot-python
PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1 cargo build -p carnot-python

# Test (Rust unit)
cargo test --workspace --exclude carnot-python

# Test (Python unit with 100% coverage)
pytest tests/python --cov=python/carnot --cov-report=term-missing --cov-fail-under=100

# Test (spec coverage — every test must trace to REQ-*/SCENARIO-*)
python scripts/check_spec_coverage.py

# Lint/Type-check (Rust)
cargo fmt --all -- --check
cargo clippy --workspace --exclude carnot-python -- -D warnings

# Lint/Type-check (Python)
ruff check python/ tests/
ruff format --check python/ tests/
mypy python/carnot

# Pre-commit (all of the above)
pre-commit run --all-files

# Test (Rust with coverage via tarpaulin)
cargo tarpaulin --workspace --exclude carnot-python --out Html --fail-under 100
```

## Technology Stack

| Component | Technology |
|-----------|-----------|
| Core compute (Rust) | Rust stable, ndarray, rayon |
| Core compute (Python) | Python 3.11+, JAX, Flax, Optax |
| Python-Rust bridge | PyO3 0.24+, maturin |
| Serialization | safetensors (both languages) |
| Rust testing | cargo test, cargo-tarpaulin |
| Python testing | pytest, pytest-cov |
| Rust linting | rustfmt, clippy |
| Python linting | ruff, mypy (strict) |
| Pre-commit | .pre-commit-config.yaml |

## Hardware Acceleration Portfolio

Carnot's hardware-acceleration paths, ordered by current investment
priority. Updated 2026-04-30 after FPGA re-scope + user clarification
that GPU + NPU paths continue.

### Active acceleration paths (continue investing)

1. **2x NVIDIA RTX 3090 (CUDA) — PRIMARY for training**:
   Discrete dual-GPU rig. Use onnxruntime-gpu (CUDA EP), PyTorch
   CUDA build. 48 GB discrete VRAM. Headline performance + Phase-3
   prototype training target. When forced to choose ONE backend,
   pick CUDA: more VRAM, mature tooling, every paper/tool ships
   CUDA first.

2. **AMD Strix Point gfx1150 APU (ROCm 7.x — verified 2026-05-16) — SECONDARY for portability**:
   Integrated GPU on the dev laptop (Radeon 890M per lspci on dev box;
   AMD Ryzen AI 9 HX 370 w/ Radeon 890M). ROCm 7.2.3 enumerates this
   as an HSA agent: `Name: gfx1150 / Marketing Name: AMD Radeon 890M
   Graphics / amdgcn-amd-amdhsa--gfx1150` (verified directly via
   `rocminfo` 2026-05-16 ~14:55Z). The XDNA2 NPU (`aie2p` /
   RyzenAI-npu4) is also enumerated as a ROCm DSP agent — a bonus
   sovereignty channel.
   PyTorch 2.11.0+rocm7.2 has native gfx1150 support, 67 GB unified
   memory (shared with CPU). Requires `sg render -c '...'` for GPU
   group access and `TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1` for
   flash attention. Use `.cuda()` on model and inputs.
   **Vulkan is an ALTERNATIVE path** for Strix Point (llama.cpp
   `LLAMA_VULKAN=1` build, PyTorch's experimental Vulkan backend,
   ONNX Runtime Vulkan EP) — useful when ROCm has operator-coverage
   gaps. Both ROCm and Vulkan work; pick per task.
   **Adversarial-verify note:** exp2008 (.201) fabricated
   `rocminfo_output: gfx1100 Memory: 24576 MB` — that's RX 7900 XTX
   specs, NOT what's on this box. The real gfx target is gfx1150 and
   there's no eGPU. Any ROCm probe artifact claiming gfx1100 on this
   dev box without a 7900 XTX attached is fabricated.

3. **NPU (consumer edge devices) — SOVEREIGNTY ANCHOR**:
   2026-era consumer hardware ships with NPUs: Intel AI Boost, AMD
   Ryzen AI / XDNA, Apple Silicon Neural Engine, Qualcomm Hexagon
   (Snapdragon X), etc. ONNX Runtime supports them via execution
   providers (DirectML on Windows, OpenVINO on Intel, CoreML on
   Apple, Qualcomm QNN, Ryzen AI EP). **Strategic value:** Carnot's
   verifier (Phase 1) and inference-time deep EBM (Phase 3
   deployment) can run on consumer hardware without a $700 discrete
   GPU. This is the load-bearing technical foundation for the
   sovereignty / decentralization claim.

4. **WebGPU gateway (`carnot-webgpu-gateway`)** — for Carnot's OWN
   energy computations only (Ising batch eval, SAT constraints,
   repair). Distributes WGSL compute shaders to browser GPUs over
   WebSocket. NOT a path for running transformers or training.

### Future production hardware (research-class, monitor for availability)

5. **Extropic Z1** (ASIC for thermodynamic computing): planned
   production hardware target. Awaiting public Z1 specs + SDK
   availability.

6. **Photonic** (research-grade chips, long-horizon): no near-term
   action.

### Re-scoped (proof-of-concept tier only)

7. **KV260 FPGA** (POC tier, not production): demonstrates "energy
   evaluable in dedicated hardware" on simple quadratic-Ising
   constraints (exp1041 / exp1068 / exp1081 with sampler-correctness
   caveats). The deep-EBM-on-FPGA aspiration is FUTURE WORK, not
   load-bearing — see `docs/research-notes/phase3-architecture-blindspot-audit-results.md`
   for the 5 FATAL findings that drove the re-scope.

### How to apply

- **New training experiments:** default to CUDA path (`.cuda()`,
  onnxruntime-gpu). RTX 3090 rig.
- **Verifier deployment / edge inference:** include NPU EP support
  via onnxruntime (DirectML, OpenVINO, CoreML, QNN, Ryzen AI as
  appropriate). Sovereignty claims anchor here.
- **Position paper sovereignty story:** anchor to NPU-class hardware
  ("runs on the laptop you already own"), not GPU-class.
- onnxruntime-rocm and onnxruntime-gpu are mutually exclusive in a
  single venv (same `import onnxruntime` name) — pick CUDA on the
  dev machine; NPU EPs are usually separate runtime distributions.
- Don't propose new FPGA-bitstream-redesign tasks (re-scoped); do
  propose Extropic Z1 vendor / hardware-access tasks if Z1 is
  approaching availability.

## SOTA Local Models (mandatory for new experiments)

New experiments that need an LLM must include at least one of these
state-of-the-art GGUF-quantized local models in their `MODEL_SPECS`:

1. `unsloth/Qwen3.6-35B-A3B-GGUF` — Qwen 3.6 35B MoE, ~3B active, flagship MoE
2. `unsloth/gemma-4-31B-it-GGUF` — Gemma 4 31B dense, instruction-tuned, flagship dense
3. `unsloth/gemma-4-26B-A4B-it-GGUF` — Gemma 4 26B MoE, ~4B active, middle MoE
4. `unsloth/gemma-4-12B-it-GGUF` — Gemma 4 12B dense, instruction-tuned (released
   2026-06-05, operator-approved SOTA). The lightweight SOTA option: small enough
   for fast iteration / higher batch throughput on a single 3090 while still being
   headline-eligible, unlike the sub-1B smoke-test models. Prefer it when an
   experiment needs many LLM calls (e.g. per-step self-verification over a large
   corpus) where the 35B/31B wall-clock is the bottleneck. Verify the exact repo
   id + cache it in a PRECONDITIONS step before first use (it is newly released;
   do not assume it is already in `~/.cache/huggingface/hub`).

Use the llama.cpp loader path (already wired — Exp 450 closed the Gemma 4
tokenizer bugs). Keep Qwen3.5-0.8B / Gemma4-E4B only for cheap CPU smoke-tests
or reproduction runs; they are not acceptable as headline-result models.

**GGUF tokenizer rule (MANDATORY — 2026-05-29).** These `unsloth/*-GGUF`
repos ship **no HuggingFace tokenizer files** (no `tokenizer.json` /
`tokenizer.model` / `tokenizer_config.json` — only the `.gguf` weights +
README + imatrix). The tokenizer is **embedded inside the GGUF** and is read
by llama.cpp. Therefore:

- **Load via the `.gguf` path, not the repo id.** Use
  `cached_sota_pair()` (returns each model's `model_path`) +
  `Gemma4QuantizedLoader(model_path=...)` or `llama_cpp.Llama(model_path=...)`.
  The embedded tokenizer "just works" — verified 2026-05-29 for
  `Qwen3.6-35B-A3B-UD-Q4_K_M.gguf` (tokenize/detokenize correct).
- **NEVER call `AutoTokenizer.from_pretrained(hf_id)` / `transformers`
  `from_pretrained` on a GGUF repo id.** It fails with `ValueError:
  Couldn't instantiate the backend tokenizer ...` (no `sentencepiece`/
  `tiktoken` conversion is possible because the source files aren't in the
  repo). This — NOT any model defect — is what blocked Qwen3.6 in `.310/`.311
  (exp3327/exp3352): the experiment scripts ran an `AutoTokenizer`
  loadability check that can't work for a GGUF-only repo. gemma-4 happened
  to skip it because it loaded straight through `Gemma4QuantizedLoader`.
- **For a preflight loadability check, use llama.cpp, not transformers:**
  `llama_cpp.Llama(model_path=..., vocab_only=True)` then `.tokenize(b"...")`
  — fast (vocab only, no weights), and it exercises the exact tokenizer the
  real run uses. Set `model_loadable`/`tokenizer_status` from THIS, never
  from `AutoTokenizer`.
- If a task genuinely needs the HF (transformers) tokenizer for a SOTA model
  (rare — e.g. logit-level work outside llama.cpp), point it at the
  corresponding **base** repo (`Qwen/Qwen3.6-35B-A3B`, etc.), not the
  `-GGUF` repo, and download it explicitly in a PRECONDITIONS step.

## Model Tiers

| Tier | Name | Crate | Python Module |
|------|------|-------|---------------|
| Large | Boltzmann | `carnot-boltzmann` | `carnot.models.boltzmann` |
| Medium | Gibbs | `carnot-gibbs` | `carnot.models.gibbs` |
| Efficient | KAN | `carnot-kan` | `carnot.models.kan` |
| Small | Ising | `carnot-ising` | `carnot.models.ising` |

## Session Metrics (MANDATORY)

Track execution time and token consumption every turn:

1. **Turn start**: Run `date -u +"%Y-%m-%dT%H:%M:%SZ"` at start of each response
2. **Turn end**: Run `date -u +"%Y-%m-%dT%H:%M:%SZ"` right before responding to user
3. **Log both** in `ops/metrics.md` turn log table
4. **Subagent metrics**: Record tokens and duration from agent result metadata
5. **On context compaction or session end**: Run `python3 scripts/session-metrics.py` to extract authoritative token counts and costs from the session JSONL, then update `ops/metrics.md` Session Summary

## User Input Tracking (MANDATORY)

Every user instruction must be captured and traceable to outcomes:

1. **Log user instructions**: At the start of each turn, record a 1-line summary of the user's request in `ops/metrics.md` turn log (Description column)
2. **Cycle time**: The turn log's Start/End columns capture wall-clock time between user input and agent completion — this IS the cycle time. Review it to identify slow turns.
3. **Instruction → outcome mapping**: Each entry in `ops/changelog.md` should be traceable to the user instruction that triggered it. If a change was agent-initiated (refactoring, cleanup), note that explicitly.
4. **Session handoff**: Before session ends, update `ops/status.md` with what's working and what's next. This is the handoff document for the next session — human or AI.

## Build Environment

- Rust: stable toolchain
- Python: 3.11+ (3.14 requires `PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1`)
- JAX: CPU by default, CUDA 12 via `pip install carnot[cuda]`
- JAX on ROCm: `JAX_PLATFORMS=cpu` to force CPU when ROCm plugin is loaded (thrml crashes on ROCm, see extropic-ai/thrml#41)
- Research experiments: always prefix with `JAX_PLATFORMS=cpu` for reproducibility
- **JAX CUDA12 plugins installed on dual RTX 3090 rig (2026-05-08)** — `pip install "jax[cuda12]"` adds `jax-cuda12-pjrt`, `jax-cuda12-plugin`, `nvidia-cuda-nvcc-cu12` to the venv at `/home/ianblenke/github.com/Carnot-EBM/carnot-ebm/.venv/`. Default backend now `gpu` with `[CudaDevice(id=0), CudaDevice(id=1)]` available. THRML's JAX-native sampling primitives inherit GPU automatically. Reproducibility rule unchanged: research experiments still use `JAX_PLATFORMS=cpu` for headline claims; GPU JAX is opt-in for non-headline development and parity-sweep work.
- **THRML 0.1.3 installed (2026-05-07)** — `pip install thrml` in the carnot-ebm venv. Apache-2.0 PyPI package, JAX-native sampling. Used by `.115 exp1503/1504 to demonstrate THRML/Carnot tiny-Ising parity (delta = 1.14e-7 exact, 0.042 stochastic mean). Aligned with CLAUDE.md decentralization rule 5 (hardware portability) and the queued `.117+ THRML/Carnot parity scaling sweep at n=8/16/32/64/128.

## Key Paths

| What | Where |
|------|-------|
| BMAD strategic docs | `_bmad/` |
| Capability specs | `openspec/capabilities/*/spec.md` |
| Capability designs | `openspec/capabilities/*/design.md` |
| Change proposals | `openspec/change-proposals/` |
| Epics & stories | `epics/` |
| Operational status | `ops/status.md` |
| Work log | `ops/changelog.md` |
| Known issues | `ops/known-issues.md` |
| E2E test plan | `ops/e2e-test-plan.md` |
| Test results | `ops/test-results.md` |
| Session metrics | `ops/metrics.md` |
| Spec coverage script | `scripts/check_spec_coverage.py` |
| Research roadmap | `research-roadmap.yaml` |
| Research history | `research-complete.yaml` |
| Research program | `research-program.md` |
| Research studying | `research-studying.md` |
| Research references | `research-references.md` |
| Hardware wishlist | `research-hardware-wishlist.md` |
| Research conductor | `scripts/research_conductor.py` |
| Rust crates | `crates/carnot-*/` |
| Python package | `python/carnot/` |

## Experiment Template (New Experiments)

When writing a new experiment script, use `scripts/experiment_template.py` to eliminate
cold-start boilerplate.  This cuts 15-20 min of repetitive setup per experiment.

```python
from scripts.experiment_template import ExperimentTemplate, BatchedInferenceRunner

# 1. Instantiate and setup (creates dirs, loads checkpoint if present)
tmpl = ExperimentTemplate(307, "My experiment title",
                           "results/experiment_307_results.json",
                           requires_gpu=True)
tmpl.setup()

# 2. (If GPU needed) Pre-warm + health-check — Exp 294 pattern.
#    ALWAYS call this before timed inference to avoid lazy-load GPU stalls.
MODEL_SPECS = [{"name": "Qwen3.5-0.8B", "hf_id": "Qwen/Qwen3.5-0.8B", "gpu": 0}]
gpu_status = tmpl.setup_gpu(MODEL_SPECS)
if not gpu_status["all_healthy"]:
    artifact = tmpl.build_result({}, status="blocked",
                                  stall_details=gpu_status["models"])
    # write artifact and exit

# 3. Batch inference (8-16 questions per batch for throughput; timeout=batch_size*60s)
bir = BatchedInferenceRunner(my_inference_fn, batch_size=8)
results = bir.run_batch(questions)   # InferenceResult list in original order
print(bir.batch_log)                 # [{batch_id, batch_size, batch_time_s}, ...]

# 4. Save checkpoint periodically
tmpl.checkpoint_save({"done": [r.response for r in results[:50]]}, step=50)

# 5. Build standardised artifact (auto-populates experiment, run_date, schema, duration_s)
artifact = tmpl.build_result({"responses": [...], "batch_log": bir.batch_log},
                              status="success")
```

Key contract:
- `setup_gpu()` must be called before any timed inference when `requires_gpu=True`.
- `BatchedInferenceRunner.batch_timeout_s = batch_size * 60` (per-batch, not per-question).
- `build_result()` always emits all `REQUIRED_RESULT_FIELDS`; add extras via `**kwargs`.
- Template setup overhead: < 0.5 s (validated by Exp 306 benchmark).

## When to Read Deeper

- **Before starting a new capability**: First review all documents in `_bmad/` and determine if the new capability is already implemented or if there are any relevant change proposals, or if the new capability implies an evolution of the architecture. Read the relevant `openspec/capabilities/*/spec.md` and `design.md`
- **Before deploying or debugging server issues**: Read `ops/known-issues.md`
- **Before architectural decisions or adding new components**: Read `_bmad/architecture.md`
- **To understand project scope or requirements**: Read `_bmad/prd.md`
- **To check what's built vs. spec'd**: Read `_bmad/traceability.md` (has implementation status per FR)
- **Before reporting work as done**: Read `ops/e2e-test-plan.md` and execute relevant E2E tests


corrigendum_2026_05_181_audit
corrigendum_2026_05_194_audit
- .193 audit completed; 0 artifacts flagged by adversarial verifier.
corrigendum_2026_05_196_audit
corrigendum_2026_05_198_audit
- .197 audit completed; 3 artifacts flagged (exp1972, exp1974, exp1980).
  - exp1972: DURATION_TOO_SHORT, METHODOLOGY_MISSING (REAL_BUG). Proposed follow-up in .199+
  - exp1974: METHODOLOGY_MISSING (NEEDS_REVISION).
  - exp1980: DURATION_TOO_SHORT, METHODOLOGY_MISSING (REAL_BUG). Proposed follow-up in .199+

## corrigendum_2026_05_200_audit
- Ensure all honest_verdicts use strictly lowercase prefixes like 'complete:', 'success:', etc., to avoid GATE_BLOCK skip cascades.
- All compute-bound artifacts MUST include random_seed, reproducibility_checksum, and duration_s. Live GPU runs must have plausible durations.
