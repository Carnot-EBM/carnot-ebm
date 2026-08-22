# Cumulative coherence: converting CLAUDE.md rules to checks — 2026-08-21

Read-only planning note. No code was changed. Every claim below was verified
against the live checkout on 2026-08-21 (UTC 2026-08-22 02:50). File and
function references are exact.

The question: which ACTIVE judgment-rules in CLAUDE.md can become mechanical
checks or tools? The test comes from harness-engineering's principle: lessons
from accepted work, corrections, and failures should become context,
boundaries, tools, examples, and checks — not prose a future agent must
remember. A rule you must read is weaker than a check that cannot be
forgotten.

## 1. Evidence verification (the briefing's four failures)

All four failures are real. Three were fixed today, before this note was
written. The numbers need small corrections.

| # | Briefing claim | Verified state |
|---|---|---|
| 1 | 51 phantom `OK (conductor)` rows | **57**, not 51. All 57 missing deliverables are OK-stamped (`docs/research-notes/conductor-self-improvement-2026-08-21.md`, CORRECTION 1). **Fixed today**: commit `87f209cdcf` derives `result` from evidence and refuses duplicate milestone appends. |
| 2 | QA audit silent since 2026-07-29, rotation still advancing | Confirmed. Report dated Jul 24 in git; rotation offset 78 with today's mtime. **Fixed today**: commit `c340cb16e2` adds `_run_audit_with_receipt` (`scripts/research_conductor.py:4979`) and a `--budget-seconds` deadline the audit knows about (`scripts/qa_layer_authenticity_audit.py:1321`, REQ-CONDUCTOR-RECEIPT-1). |
| 3 | Two stalls in 8 hours; human hand-writes `prior_failures:` | Confirmed, and larger: 5,125 refusal lines total; worst stall ~66 hours. **Fixed today**: commit `deff0b847d` wires bounded replan with the lint report as planner input, then park-and-escalate. |
| 4 | Frozen harness pins a retired generator; warning is prose | Confirmed and **still open**. `scripts/arc_scored_path_lever_harness.py:68` carries the banner "FROZEN PRE-2026-07-28 HARNESS — MEASURES THE **RETIRED** GENERATOR"; line 425 still pins `repo_substr="Qwen3.5-9B-MTP"`. Nothing refuses a run. Conversion 4 below closes it. |

Consequence for this plan: items 1–3 are done. The plan builds the regression
locks and the remaining conversions, not the fixes themselves.

## 2. Inventory of ACTIVE rules

Classification: **(a)** already mechanically enforced, **(b)** convertible to
a check or tool, **(c)** irreducible judgment — prose is correct.

Enforcement status was verified against the code, not against the prose. Two
index entries are stale (marked STALE-INDEX below): the prose says
"honor-discipline until the check ships" but the check shipped.

| Rule (CLAUDE.md section) | Class | Verified enforcement state | Disposition |
|---|---|---|---|
| Project Writing Style: STE100 | (c) | No lint, by design | Keep prose. See section 5. |
| Codex-Default for Experiments v2 | (a) | `CODEX_FORCE_EXPERIMENTS=1` coercion + planner prompt. Residual is an operator grant (`requires_claude_verified`), not agent judgment | Reclassify in the Rule Index (append-only). |
| Failed-Experiment Rerun Discipline | (a)+(c) | `exclusion_manifest_lint.py` SCOPE_MATCHED_PRIOR_FAILURE + `failure_ledger.is_doomed_rerun` + today's replan loop that feeds the violation report to the planner | Root-cause quality stays judgment. Merge index entry with Exclusion-Manifest (section 4). |
| Exclusion-Manifest Cross-Check | (a) | Same enforcer stack | Same merge. |
| Pre-Launch Preconditions | (b) | `preconditions_checked` is NOT read anywhere in `scripts/adversarial_verify.py` (verified by grep). The prose says the detector "should be extended" — never done | Conversion 5. Artifact-side WARN only; do NOT lint prompt structure (section 5). |
| Adversarial Artifact Verification + Sample-Size Rigor | (a)+(b)+(c) | Core enforced (linter + fabrication gate). Gap: nothing verifies a capstone excluded `flagged_adversarial` upstreams | Conversion 8 for the capstone gap. Sample-size design choice and "cross-check surprising results" stay judgment. |
| Inference-Substrate Declaration | (a) | Substrate floors enforced. Recent friction (exp6491, 54.5s vs 60s floor) is threshold calibration, not a missing check | No conversion. Note the calibration cost in section 5. |
| Principle-Annotated Artifact Fields | (b weak)+(c) | Activation guard does not scan for `principle:` (verified). The annotation's CONTENT is unlintable | Conversion 10, WARN-only presence check. Low rank. |
| Phase Prototype + Empirical Validation + Adversarial Check | (c) | Research methodology | Keep prose. |
| Scope-Reduction-When-Flagged | (b) | `scope_reduction_compliance` is not checked anywhere (verified) | Conversion 9. Trigger-gated, cheap, rare. |
| Hardware-Task Continuity | consolidate | North-star §3 already recommends the relaxation: KV260 to terminal, GateMate/PolarFire opportunistic | Operator decision, not a check. Section 4. |
| KV260 SSH-Not-SD-Card | (a) STALE-INDEX | ENFORCED: `exclusion_manifest_lint.py:108` blocks `/dev/mmcblk` in KV260 task prompts (WRONG_MECHANISM_PRECONDITION), plus the manifest entry. The rule's own prose still says "until that ships" | Reclassify to MECHANICALLY-ENFORCED in the index. |
| Operator-Only External Publication | (b) | No scan exists: `arxiv submit` / `gh release create` appear nowhere in `research_conductor.py` (verified) | Conversion 3. Two layers: harness deny-hook + activation-guard WARN. |
| Never Stash — Commit-First | (b) | `scripts/safe-pull.sh` exists (tool half done). Nothing blocks `git stash` | Conversion 3 (same deny-hook batch). |
| Decentralization-Respecting Design Constraints | (b sliver)+(c) | Rules 1–6 are architecture judgment. Rule 7 (no vendor SDK imports in core) is mechanically checkable | Conversion 7 for rule 7 only. |
| Pre-Staged Roadmap Convention | (a)+(c) | `_expected_next_milestone` match is mechanical. The authoring protocol is judgment | No action. |
| Documentation Update Rules (never-prune) | (a partial)+(c) | `determination_preservation_lint.py` + `operator_curated_docs_lint.py` cover the incident classes that actually occurred | Generic shrink-detection deferred (section 5). |
| Tests Must Run and Assert | (b) | No skip-marker lint exists (verified: no `mark.skip` reference in `scripts/` or `tests/python/conftest.py`) | Conversion 6. The rule is absolute, so FP risk is zero. |
| QA-Layer Authenticity | (a)+(c) | Layer 2 audit + today's receipts. "How to write a good check" is judgment | Keep. Section 4 consolidates its thesis. |
| Test-Run Record Integrity | (a partial)+(c) | Rules 2–5 enforced. Rule 1 (`git status` before `git add -A`) is partly covered by `test_suite_mutation_check.py --gate`; the attribution step is inherently judgment (over-reporting is by design) | Keep rule 1 as prose. |
| Verdict Terminal-Prefix Discipline | (a, treadmill) | Enforced by four substring token lists patched at least six times; `disqualified:` (an honest negative) drew a critical flag this week | Conversion 2 replaces the mechanism. Rule then retires to historical. |
| Circularity / Oracle-Distinctness | (a) | `check_circular_moat_overclaim` fires. Gap: an honest `verifier_is_oracle: true` artifact can still carry a `complete_positive` prefix that downstream readers over-read (exp6478) | Conversion 2's enum closes this (`circular_positive` class). |
| ARC-AGI-3 IS a Live Hidden-Game Discovery Agent | (c) | Foundational framing | Keep prose. |
| ARC Solve Reproducibility + Solver-Reuse | (a) | `arc_solver_kit.reproduce` gate + registry + `check_arc_outer_loop_solve` | No action. |
| ARC Live-Path Reachability | (a) | Orphan lint + artifact check + milestone audit | No action. |
| ARC Generalization-Testing Floor | (a soft) | WARN-only lint, deliberately | No action; promotion decision is the operator's per its own prose. |
| Missing-Verifier Gap Logging | (b weak) | `missing_verifier_gaps` is not read by `adversarial_verify.py` or the conductor (verified) | Deferred (section 5): the trigger ("is this a verifier experiment?") is fuzzy. |
| SOTA-Ingestion Cycle | (c) | Slot reservation is planner judgment ("is this milestone bleeding-edge?") | Keep prose. A warn-heuristic could mirror the generalization-floor lint later. |
| SOTA Local Models / GGUF tokenizer rule | (a mostly) | Failure mode documented; preconditions pattern covers it. A grep lint banning `AutoTokenizer.from_pretrained` on `-GGUF` ids is possible | Optional, low rank; fold into Conversion 7's banned-API config if free. |
| Paper-v6 Narrowing Discipline (indexed historical) | (a) STALE-INDEX | `scripts/paper_v6_narrowing_lint.py` exists and is a pre-commit hook (`paper-v6-narrowing-lint`). The rule's prose still says "until that ships" | Update index note. Ask the operator whether the three corrigenda landed (its retirement condition). |
| Session Metrics / User Input Tracking (workflow sections) | (b cheap)+(c) | No enforcement. A SessionEnd hook could run `scripts/session-metrics.py` | Low value. Offer to the operator; do not build unprompted. |

## 3. Ranked conversion plan

Ranked by expected value per unit of effort. "Run point" names where the
check fires. Every new check prefers a structural fact (file exists, field
declared, enum member) over a threshold — thresholds are the FP treadmill
this project already pays for (three substrate-taxonomy extensions to date).

### 1. Ledger-invariant lint (regression lock for truthful archival)

- **Asserts:** `research-complete.yaml` has one entry per milestone id, and
  every task row whose `result` is `OK` names a deliverable that exists on
  disk. Forward-only from a cutoff date, so uncorrected history does not
  block commits.
- **Run point:** pre-commit, on `research-complete.yaml`.
- **Home:** new `scripts/research_complete_ledger_lint.py` (~50 lines), wired
  in `.pre-commit-config.yaml`. The prior note already specified it.
- **Cost:** 2–4 hours. **FP risk:** near zero — both facts are structural.
- **Why first:** the archiver fix (`87f209cdcf`) is one revert away from
  regressing, and this file is the planner's failure record. The lint makes
  today's fix permanent. Add it to `GUARD_TARGETS` (or
  `ACKNOWLEDGED_NON_QA_LAYER` with a reason) per the QA-Layer rule.

### 2. Declared verdict class, cross-checked (retires the token-list treadmill)

- **Asserts:** every artifact declares `verdict_class` from a closed enum
  (`positive | circular_positive | null | blocked | disqualified | partial`).
  `adversarial_verify.py` cross-checks the class against structural fields
  it already reads: `verifier_is_oracle: true` forbids `positive`;
  `flip_count == 0` forbids an informative `null`; a failed acceptance gate
  forbids `positive`. Capstones consume only the enum.
- **Run point:** artifact write (planner-mandated field) + `adversarial_verify`
  + `_verdict_is_untrustworthy` (enum first, legacy token lists as fallback).
- **Home:** `scripts/adversarial_verify.py` (new check),
  `scripts/in_process_doc_reconcile.py` (enum mapping),
  `scripts/research_conductor.py:_verdict_is_untrustworthy`, planner prompt.
- **Cost:** 2–3 days. **FP risk:** low; the enum is closed and the
  cross-checks are structural, not textual. Abandon if >10% of artifacts
  need a class outside the enum.
- **Why second:** the terminal-prefix whitelist has been narrower than honest
  practice six times (`disqualified:` is the latest). This is the largest
  recurring maintenance cost among the ACTIVE rules, and it also closes the
  exp6478 gap where an honestly-declared circular win still reads as a
  research positive downstream. On ship, mark Verdict Terminal-Prefix
  historical (append-only banner).

### 3. Harness deny-hooks for forbidden commands (two prose rules become boundaries)

- **Asserts:** a Bash tool call matching `git stash`, `--no-verify`,
  `arxiv (submit|upload)`, `openreview`, `gh release create`, or
  `twine upload` is denied at the harness layer, with a message pointing at
  the rule and the sanctioned alternative (`scripts/safe-pull.sh`; "prepare
  the package for the operator").
- **Run point:** `.claude/settings.json` `PreToolUse` hook with a `Bash`
  matcher. The hook infrastructure already exists in this repo (a
  `Write|Edit` PreToolUse hook runs `scripts/validate-phase-gate.sh`).
- **Second layer:** an activation-guard WARN when a task prompt contains a
  submission verb in an imperative step. WARN, not HARD: the legitimate
  phrasing "do NOT run arxiv submit" would trip a hard block — the QA-Layer
  rule's negation-blindness class. `operator_override:` clears it.
- **Home:** `.claude/settings.json` + a small hook script; the prompt scan
  extends the pre-emit path in `_activate_next_roadmap` next to the
  exclusion-manifest call.
- **Cost:** 2–4 hours. **FP risk:** near zero for the deny-hook (these
  commands have no legitimate autonomous use per Never-Stash and
  Operator-Only External Publication); WARN-only for the prompt scan.
- **Converts:** "Never Stash — Always Commit-First" and the enforcement gap
  in "Operator-Only External Publication". Also encodes the standing
  no-`--no-verify` feedback.

### 4. Frozen-harness pin guard (briefing item 4, still open)

- **Asserts:** a harness that pins a generator refuses to run when its pin
  differs from the live pin, unless the caller passes an explicit override
  flag (`--allow-retired-pin`). The live pin moves to one importable
  constant; harnesses and the agent both read it.
- **Run point:** harness `main()`, before any model load.
- **Home:** new small module (for example
  `python/carnot/agentic/arc_live_pins.py`) consumed by
  `python/carnot/agentic/arc_competition_agent.py` and
  `scripts/arc_scored_path_lever_harness.py:425`.
- **Cost:** 2–4 hours plus a sweep for other pin duplicates. **FP risk:**
  zero — a deliberate archaeology run passes the flag.
- **Why:** the banner at line 68 was read only after a run was launched and
  discarded. This is the general pattern "prose warning in a file header →
  runtime refusal". The same derive-don't-duplicate defect family is already
  under repair elsewhere this session (stale token defaults vs the
  Qwen3.8-27B pin), so the constant should land once, in one place.

### 5. `preconditions_checked` presence check

- **Asserts:** a compute-bound artifact (`live_llm_inference`,
  `live_llm_embedding_extraction`, `hardware_smoke`) that lacks
  `preconditions_checked` draws a WARN (`PRECONDITIONS_UNDECLARED`).
- **Run point:** `scripts/adversarial_verify.py`, next to
  METHODOLOGY_MISSING — the extension the rule's own prose requested and
  never got.
- **Cost:** 1–2 hours. **FP risk:** low at WARN severity; do not make it
  CRITICAL until a backfill dry-run over the corpus shows the hit rate.

### 6. Skip-marker rejection in the test suite

- **Asserts:** collection fails if any test carries `pytest.mark.skip` /
  `skipif` / `unittest.skip`.
- **Run point:** `tests/python/conftest.py` `pytest_collection_modifyitems`
  (fires on every run, not only at commit).
- **Cost:** under 1 hour. **FP risk:** zero — "Tests Must Run and Assert"
  is absolute ("skipping tests is never allowed").

### 7. Vendor-SDK import ban for the core

- **Asserts:** `python/carnot/verify/` and `python/carnot/pipeline/` import
  no vendor SDK (`openai`, `anthropic`, `google.generativeai`, etc.) —
  Decentralization rule 7, verbatim.
- **Run point:** ruff `flake8-tidy-imports` banned-api config in
  `pyproject.toml`; no new script, rides the existing `ruff` hook.
- **Cost:** about 1 hour. **FP risk:** near zero; adapters live in named
  submodules by the rule's own design.

### 8. Capstone flagged-aggregation cross-check

- **Asserts:** an `aggregation_from_upstream_artifacts` artifact whose
  `cited_upstream_artifacts` includes an experiment stamped
  `flagged_adversarial: true` draws a CRITICAL flag.
- **Run point:** `scripts/adversarial_verify.py`.
- **Cost:** half a day (resolve cited ids to files, read stamps). **FP
  risk:** low when keyed on the structured citation field; artifacts without
  the field get a WARN for the missing provenance instead.
- **Converts:** the "capstone MUST skip flagged artifacts" clause of the
  fabrication gate, which today has no verifier.

### 9. Scope-reduction compliance check

- **Asserts:** when `ops/known-issues.md` has an active heading containing
  "SCOPE REDUCTION", the roadmap YAML must carry a non-empty
  `scope_reduction_compliance` mapping.
- **Run point:** `_activate_next_roadmap` pre-emit path.
- **Cost:** about 2 hours. **FP risk:** low — the trigger is explicit
  operator phrasing; fires rarely, but the one time it mattered (.111) cost
  a full milestone.

### 10. Principle-annotation presence WARN

- **Asserts:** a task prompt's `REQUIRED ARTIFACT FIELDS` block whose entries
  lack `principle:` draws an activation-guard WARN.
- **Run point:** `_activate_next_roadmap` pre-emit path.
- **Cost:** half a day. **FP risk:** moderate — the fields live inside
  free-text prompts, and prose parsing is the exact class-B trap. WARN-only,
  never HARD. Lowest rank for that reason; the annotation's value is its
  content, which no lint can judge.

### Deferred (deliberately not built now)

- **ops-docs shrink detector** (never-prune, generic form): a net-deletion
  WARN on `ops/*.md`. Moderate FP on legitimate reflows. Build only if a
  never-prune incident recurs outside the classes the existing two lints
  already cover.
- **`missing_verifier_gaps` presence check:** the trigger ("this artifact is
  a verifier evaluation") has no crisp structural marker today. Add the
  marker to the planner's REQUIRED ARTIFACT FIELDS first; lint second.
- **STE100 prose lint:** recommend against permanently (section 5).

## 4. Consolidation proposal

Never-prune applies: consolidation means append-only banners and index
updates, not deletion.

1. **Fix the Rule Index's own drift (cheapest, do first).** Two entries are
   misfiled: KV260 SSH-Not-SD-Card is enforced
   (`exclusion_manifest_lint.py:108`) and Paper-v6 Narrowing has its lint
   (`scripts/paper_v6_narrowing_lint.py`, hook `paper-v6-narrowing-lint`).
   Move both to the MECHANICALLY-ENFORCED list with a dated note. The index
   is itself a pattern list that drifted narrower than reality — the
   project's named class-B bug, inside the map meant to prevent it.
2. **Merge the rerun rules' index entries.** Failed-Experiment Rerun and
   Exclusion-Manifest Cross-Check now share one enforcer stack (the lint,
   the ledger, and today's replan loop). One index line pointing at the
   stack, two prose sections retained as history.
3. **Name the receipts invariant once.** Verifier Authenticity, QA-Layer,
   Test-Run Record Integrity, and Adversarial Landing-Page all restate one
   thesis: a trusted-and-silent guard is worse than no guard. Today's
   REQ-CONDUCTOR-RECEIPT-1 plus `_run_audit_with_receipt` is that thesis as
   an enforceable invariant. Write it as a spec requirement in
   `openspec/capabilities/research-harnesses/spec.md` and let the four prose
   sections cross-reference it, so the next audit caller inherits the
   contract instead of re-deriving it.
4. **Retire Verdict Terminal-Prefix when Conversion 2 ships.** The enum
   replaces the prefix convention. Append the historical banner; keep the
   prose.
5. **Hardware-Task Continuity: execute the already-recommended narrowing.**
   North-star §3 says KV260-to-terminal, others opportunistic. This needs an
   operator sign-off, not a check. If the rule survives, convert its board
   table to a small data file (board, reachability command, terminal-state
   flag) that a 20-line activation-guard check reads — the rule becomes data
   plus check instead of prose.
6. **Ask the operator two retirement questions:** did the three Deep Think
   corrigenda land (Paper-v6 Narrowing's retirement condition)? And should
   Session Metrics tracking move to a SessionEnd hook or retire?

## 5. Where converting would make things worse

The briefing asked for this seriously; here it is.

- **STE100 writing style.** A prose-style lint false-positives on quoted
  operator directives, exact error strings, and legitimate long technical
  sentences. Style noise at commit time trains agents to bypass the hook —
  the failure mode `determination_preservation_lint.py`'s docstring warns
  about. The rule's own text already reached this conclusion ("no mechanical
  lint... honor-discipline"). Agree with it.
- **Tightening the exclusion-manifest scope-matcher.** It already
  false-positives on routine forward work; that is why `operator_override:`
  exists and why six overrides were needed for one roadmap. Today's fix went
  the right direction — change the CALLER (replan with the report), not the
  matcher. Further pattern-tightening buys stalls, not safety.
- **Prompt-structure lints as HARD gates.** Checking that a prompt "contains
  a PRECONDITIONS block" or "has principle: annotations" means
  pattern-matching free prose — the pattern-narrower-than-concept trap, plus
  negation blindness. That is why Conversions 5 and 10 are artifact-side or
  WARN-only. The artifact is structured; the prompt is not.
- **Threshold checks accrue calibration debt.** The duration floors have
  needed three taxonomy extensions (exp2842, exp5178, exp6491). Each was
  correct, and each cost an incident first. New checks in this plan assert
  structural facts precisely to avoid inheriting this cost curve.
- **Design-time judgment: sample sizes, surprising results, phase
  adversarial rounds, the ARC framing.** These rules ask "is this the right
  experiment?" — a question with no structural witness. A lint would either
  be vacuous or would block legitimate exploratory work. The post-hoc
  detectors (SAMPLE_SIZE_BELOW_CLAIM, FALSE_NEGATIVE_RISK) are the correct
  mechanical halves; the design half stays prose.
- **A cost the briefing under-weights:** this repo already has 29 pre-commit
  hooks, and every new guard joins the QA-layer audit surface
  (`--check-targets` forces classification of each one). Checks are not
  free to carry either. That is why the plan prefers extending existing
  enforcers (`adversarial_verify.py` classes, the activation-guard pre-emit
  path, ruff config, conftest) over new standalone scripts — only
  Conversion 1 adds a file.

## 6. Corrections to the briefing framing

1. **The numbers:** 57 phantom OKs, not 51, and the deeper problem was the
   1,891 duplicate milestone entries (97% of the ledger) — see the prior
   note's CORRECTION 1.
2. **Three of the four failures were fixed today** (commits `87f209cdcf`,
   `deff0b847d`, `c340cb16e2`). The charge stands historically; the plan's
   job is regression locks (Conversion 1) and the remaining item
   (Conversion 4), not re-proposing the fixes.
3. **The prose-vs-check dichotomy misses the strongest form: derivation.**
   The project's best conversions did not add a check — they made the
   violation inexpressible. `_expected_next_milestone()` computes the
   milestone prefix, so nobody can get it wrong. Agent-routing coercion
   rewrites the field instead of rejecting it. Truthful archival derives
   `result` instead of validating a claim. Where a rule can become a
   derivation, prefer it over a check: a check still fires and needs a
   human; a derivation removes the failure mode. Conversion 2 (the enum) and
   Conversion 4 (one pin, imported) are derivation-shaped for this reason.
4. **The Rule Index's ACTIVE list overstates the prose burden.** At least
   two "ACTIVE" rules are already enforced, and several others are
   enforced-with-a-judgment-residual. The honest count of fully-prose
   ACTIVE rules is roughly ten, of which this plan converts or partially
   converts seven and defends three as correctly prose.
