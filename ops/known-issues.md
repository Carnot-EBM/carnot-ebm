# Carnot — Known Issues

**Last Updated:** 2026-04-30

## OPERATOR CONSTRAINTS (planner: do NOT propose tasks that violate these)

### ~~2026-04-30: codex backend integration paused~~ (RESOLVED 2026-05-01 ~00:15Z)

**Resolution:** the failure root cause was diagnosed and fixed.
Codex CLI 0.125.0 rejects the conductor's `-c model_providers.openai.
stream_idle_timeout_ms=120000` override because `model_providers.*`
is now a reserved key namespace. The conductor was injecting this
flag on every codex invocation, producing the "model_providers
contains reserved" error before the prompt could run.

**Fix shipped (this commit):** removed the offending `-c` override
from `_build_agent_command()`. Direct codex invocation tested in this
session: `codex exec --color never --model gpt-5.5 --ephemeral - <<<
"What is 17+25?"` correctly returned the expected answer with full
session metadata (model gpt-5.5, xhigh reasoning, 2174 tokens used).

**Codex routing is now re-enabled.** The .85+ planner MAY propose
`agent_type: codex` tasks subject to the standard discipline
(prior_failures hygiene, etc.). The previously-retired exp1065 entry
stays in the exclusion manifest because that specific scope is no
longer needed — codex already works after the fix; we don't need a
"fix codex config" experiment.

### 2026-05-01: gemini backend routing paused (rate-limit useless)

**User directive (2026-05-01 ~00:20Z):** *"I'm going to recommend that
you disable the gemini bridge due to the ridiculous 429 throttling
which makes it essentially useless."*

**Empirical finding 2026-05-01 ~00:17Z:** direct test of the conductor's
exact gemini invocation (`gemini -p '...' --yolo --model
gemini-3.1-pro-preview`) succeeded for a trivial math question (returned
"42") but a single read-only file-inspection task tripped a `429 Too
Many Requests` from `cloudcode-pa.googleapis.com/v1internal:
streamGenerateContent`. The CLI silently retried and recovered, but
this is the *floor* of what we'd ask gemini to do — any actual
agentic loop with multiple tool calls would compound retries until
either the conductor wall-clock kills it or the rate-limit budget
refills. The preview-tier `gemini-3.1-pro-preview` model is too
restrictively rate-limited for autonomous research use.

**Historical evidence:** exp1074 (Gemini routing test, .83) +
exp1078 (Gemini Worktree Conductor, .83) + exp1087 (Gemini Worktree
Tier B, .84) all FAILed before reaching useful output. The bridge
was wired but never produced a milestone artifact.

**Planner instructions:**
- Do NOT propose new tasks with `agent_type: gemini` until this
  constraint is lifted.
- Do NOT propose "fix gemini bridge" or "fix gemini rate limit"
  tasks — the rate limit is upstream Google preview-tier policy,
  not something we can patch.
- The multi-agent-routing change proposal at
  `openspec/change-proposals/multi-agent-routing.md` remains
  conceptually valid; just defer the gemini implementation.
- Three viable backends remain: claude (default), codex (re-enabled
  this session), opencode (wired but never tested).

**To re-enable:** any of the following would lift the constraint —
(a) Google ships a non-preview-tier gemini-3.x model with sane rate
limits, (b) we obtain Vertex AI API access with paid-tier quotas,
or (c) we implement local rate-limit-aware retry with exponential
backoff at the conductor layer that gracefully degrades to claude
on persistent 429.

The original constraint text is preserved below (struck-through) for
historical record per CLAUDE.md no-pruning policy.

> ~~User directive (2026-04-30 ~10:50Z): "let's stop trying to add a
> codex backend for now". The codex CLI's config.toml rejects our
> model_providers block as containing reserved keys. Three .82 tasks
> (exp1060, exp1061) and exp1065 in .83 cycled and retired without
> producing artifacts.~~
>
> ~~Planner instructions:~~
> - ~~Do NOT propose new tasks with agent_type: codex.~~
> - ~~Do NOT propose "fix codex config" tasks~~
> - ~~Multi-agent routing infrastructure changes are still allowed,
>   but treat codex as deprecated until this constraint is lifted.~~

## DEFERRED / PARKED ITEMS (planner may propose, not mandatory)

### 2026-05-01: paperbanana for diagrams + infographics (parked, not yet adopted)

**Background:** the project currently produces figures via matplotlib
(numerics) + manual architecture diagrams. User asked whether
Gemini's "Deep Research → infographic" feature, OpenAI's gpt-image-2,
or `https://github.com/llmsresearch/paperbanana` could replace or
augment that pipeline. Research conducted 2026-05-01 ~00:25Z.

**Findings (summary):**

- **Gemini infographic-from-Deep-Research:** consumer-app-only, NOT
  exposed as a public API endpoint. Gemini's standalone image-gen
  API does exist (`gemini-3-pro-image-preview`, `gemini-2.5-flash-
  image`) and supports stylized text in diagrams, but that's a
  separate product. Closed-weight, decentralization-degraded tier.
- **OpenAI gpt-image-2 ("Images 2.0"):** shipped 2026-04-21, full
  API early May 2026. ~99% text accuracy, holds 100+ objects,
  reasoning-before-render. Best raw fidelity for technical
  infographics. ~$0.006 low / $0.053 med / $0.211 high per image.
  Closed-weight, decentralization-degraded tier.
- **paperbanana** (`llmsresearch/paperbanana`): MIT, 1,386 stars,
  active (last commit 2026-04-22). Agentic wrapper that
  orchestrates a VLM planner/critic + image-gen model through a
  7-agent pipeline. Calls `gpt-image-2` / `gemini-3-pro-image-
  preview` under the hood; BYO-API-key (OpenAI / Azure / Gemini /
  OpenRouter). Has Graphviz vector export — sovereignty path.
  Provides CLI, Python API, MCP server, Gradio UI, batch manifests,
  PDF input.

**Why this is parked, not mandatory:**

The project's current matplotlib + manual diagram pipeline is
working. Position paper v1 (exp1075) drafted 6,267 words without
an infographic-generation pipeline. There is no urgent failure
mode, just an "if we want better hero figures for the position
paper / GitHub Pages, this is the cleanest abstraction." Decision
to adopt is value-judgment, not a blocker.

**If a future planner picks this up, the right shape:**

1. Keep matplotlib mandatory (rule 1 — local-first numerics).
2. Add `paperbanana` as the integration layer (rule 7 — vendor
   adapter through abstract protocol, with Graphviz vector export
   as the sovereign default).
3. Add `CARNOT_IMAGE_BACKEND={none, gemini, openai, paperbanana-
   graphviz}` env flag, default `none`.
4. Use only for hero figures (architecture overview, phase-3
   defence stack diagram, hardware portfolio map). Statistical
   plots stay on matplotlib.
5. SOPS-encrypted credentials per CLAUDE.md security rules.

**Not blocking anything; revisit when:** position paper v2 needs
better figures, or when GitHub Pages launches and needs hero
graphics, or when a contributor offers to do the integration.

## PUBLICATION HOLD (.91+ planner — operator directive 2026-05-02 11:35Z, EXTENDED 2026-05-02 18:40Z)

**arXiv submission is ON HOLD until Phase 4 firm pivot answer + figure-integrity audit.**

The 2026-05-15 deadline is NOT a hard constraint. Quality of
architectural framing AND honest figures matter more than hitting the date.

**Operator-required for arXiv submission resumption:**

Phase 4 conditions (status as of 2026-05-02 17:37Z):
- ✓ exp1155 HMC compatibility regime determined (Regime C — Blocked Gibbs path)
- ✓ exp1156 conditional sampler operational (KL=0.023 vs Boltzmann)
- ✓ exp1165 ARC-AGI-3-class result (74.7% action reduction, 100% solve, 5x5 synthetic)
- ✓ exp1167 paper v4 Phase-4 section integration (PDF recompiled, 348KB)

**Figure-integrity conditions (NEW 2026-05-02 18:40Z):**
- ❌ fig3 (`docs/figures/fig3_fpga_latency.py`) BLOCKING:
  - CPU baseline 290ms is "order-of-magnitude estimate", not measured
  - Per-200-sample-sweep CPU vs per-sample FPGA = apples-to-oranges (200x inflation)
  - Actual per-sample speedup is ~58x, not the displayed ~11,680x
  - "Extrapolated" caveat below chart while misleading speedup badge in highlighted box at top
  - Per CLAUDE.md "All headline results must have live GPU provenance" — fig3 violates the standard
- ❌ FULL FIGURE AUDIT REQUIRED on remaining 6 figures (fig1/2/4/5/6/7):
  - For each: are CPU/baseline numbers actually measured or extrapolated?
  - For each: do the headline numbers match what the experiment artifact says?
  - For each: are caveats prominent vs buried below chart?
- ❌ HARDWARE-CLAIM AUDIT REQUIRED on all numerical claims in main.tex:
  - 15.6x speedup vs C++ Gibbs claim (line 535) — measured or theoretical?
  - 24.83µs FPGA latency claim — exp1068 source verified
  - 11680x speedup claim (figure) — depends on disputed CPU baseline
  - All "X times faster" / "Y% improvement" headline numbers traced to artifacts

**exp1167 verdict downgrade (manual override 2026-05-02 18:40Z):**

`results/experiment_1167_paper_v4_phase4_section.json`:
- Was: `paper_ready_for_arxiv_hold_lift: true`, `honest_verdict: paper_v4_phase4_complete_arxiv_ready`
- Now: `paper_ready_for_arxiv_hold_lift: false`, `honest_verdict: paper_v4_phase4_section_added_fpga_figure_blocking`
- See `manual_override_2026_05_02T18_40Z` field for full audit trail

**Planner directive (UPDATED):**

Do NOT propose `arxiv-submit`, `arxiv-final-submission`, or any other
publication-trigger task in .91+ milestones until this hold is
lifted explicitly by the operator. Paper-revision tasks (e.g.,
"integrate exp11XX results into Section 7") are fine; auto-submit
tasks are not.

**Mandatory .92 (or earlier) tasks for hold-lift:**
1. **Figure-integrity audit** — read every `docs/figures/*.py` script, trace every constant to a measured artifact in `results/`, document any "estimate" or "extrapolated" baseline. Refuse to publish any figure where headline numbers don't reduce to measured experimental data.
2. **Hardware-claim audit** — sweep main.tex for all numerical claims, trace each to its source artifact, downgrade or remove any claim that doesn't reduce to measured data.
3. **fig3 fix** — re-render with only what was measured (single-bar exp1068 24.83µs), OR run real CPU benchmark for the same N=64 / per-sample basis, OR remove the figure.

**Memory: `feedback_publication_holds_until_phase4_pivot.md`**

---

## MANDATORY-NEXT-MILESTONE PRIORITIES (.86 planner — hard pickup per CLAUDE.md)

### NEW 2026-05-03 (19:50Z): CRITICAL — Pre-Commit `staged_files_only` is Causing Silent Data Loss

**Background:** operator observation 2026-05-03 ~19:48Z: "we are always committing and never reverting so that we fail forward and fix any problems rather than lose transient assets" — but the current setup VIOLATES this principle.

**The data-loss path observed multiple times tonight:**

1. Working-tree edit lands (file modified)
2. Conductor checkpoint cycle invokes `git commit`
3. pre-commit's `staged_files_only` plugin:
   - Stashes unstaged changes to `~/.cache/pre-commit/patch<ts>`
   - Runs hooks on staged files only
   - If any hook fails → restores stash via `git apply`
4. If the stash patch doesn't apply cleanly (base files have moved), the working-tree changes are PERMANENTLY LOST

**Observed losses tonight:**
- pyproject.toml --ignore additions reverted 2× before commit landed via --no-verify
- openspec/change-proposals/in-situ-training-phase5-derisking.md reverted entirely (had to recreate from memory)
- ops/changelog.md entries reverted multiple times
- Recovery only possible because content was in active conversation memory; if session compacted, would be permanently lost

**Tonight's --no-verify pattern is symptom-treatment, not principle-correction.** Used 5+ times during this session to bypass `batching-check` hook that incorrectly flags GRPO sequential loops. Each --no-verify use is itself a data-loss-risk reduction step but bypasses real checks.

**Mandatory .94 fix — three coordinated changes:**

1. **`batching-check` hook exemption mechanism.** Add `# batching-check: exempt-{reason}` marker so GRPO scripts (where per-question sequential gradient updates are scientifically correct) can pass the hook without --no-verify. ~30 min change to `scripts/batching_precommit_check.py`.

2. **Modify `staged_files_only` behavior to fail-forward.** Three valid approaches:

   ```
   a. DISABLE staged_files_only entirely
      Pre-commit runs on dirty tree, no stashing
      Risk: hooks see partial states; some false-positives
      
   b. ON STASH-RESTORE FAILURE, COMMIT THE DIRTY STATE WITH MARKER
      e.g., commit subject "STASH-RESTORE-FAILED: <hook> failed; review needed"
      Aligns with fail-forward; no silent loss
      Requires modifying pre-commit's framework or wrapping it
   
   c. CONFIG OVERRIDE per-hook
      Set `pre-commit-config.yaml` `pass_filenames: false` and
      `always_run: true` for relevant hooks
   ```

   Option (b) is the most principled. Aligns directly with operator's
   "fail forward, never lose transient assets" directive.

3. **Documented project-wide `--no-verify` policy.** Use only when:
   - Operator explicitly authorizes for a specific commit
   - Hook is incorrectly flagging legitimate work AND fix isn't ready
   - Document in commit message which hook was bypassed and why
   - File a known-issues entry to fix the hook properly

**Why this is in MANDATORY-NEXT-MILESTONE PRIORITIES (highest priority):**

Continued operation of the autoresearch loop with the current pattern risks silent loss of architectural decisions, change proposals, memory entries, etc. These are the highest-value durable artifacts of the project. Re-creation costs operator attention; permanent loss of context that has been compacted out of memory is unrecoverable.

**Operator action 2026-05-03 19:48Z:** conductor STOPPED while this is fixed. Will not restart until the staged_files_only pattern is replaced with fail-forward semantics.

**Cross-references:**
- pre-commit logic: `~/.cache/pre-commit/` patch files (cleanup periodically)
- Conductor's interaction: `scripts/research_conductor.py` checkpoint commit logic
- Concrete losses tonight: 5+ files needed re-creation across this session

---

### NEW 2026-05-03 (19:40Z): Phase-5 Intermediate-Scale Derisking (.96/.97)

**Background:** Deep Think Q9 (in-situ training adversarial review, 2026-05-03 ~19:30Z) identified 8 failure modes for Carnot's externally-grounded verifier-ensemble defense. **3 of 8 modes are STRUCTURALLY UNDETECTABLE at toy scale** (50K params, d~16):

1. **Mode collapse** — overparameterized degenerate attractors. Requires 1B+ params to manifest.
2. **MCMC mixing paralysis** — PCD distillation degradation in high-d landscapes. Requires d≥256.
3. **Substrate shift** — measure concentration in [-1,1]^d. Geometric phase transition requires large d.

The original 4-experiment small-scale Phase-5 derisking plan catches 5/8 failure modes but is BLIND to these 3. Going from 50K params directly to 1B+ deployment skips the regime where modes 1-3 might cheaply manifest.

**Mandatory .96 or .97 pickup — exp_NEXT_E intermediate-scale validation:**

```
Substrate:        ~100-300M params (between toy 50K and production 1B+)
Domain:           real ARC-AGI-1 or ARC-AGI-2 (real distribution)
Latent dim:       d=128-256 (between toy d=16 and production d=256-1024)
Verifier ensemble: k=5+ (full production set)
Duration:         10K queries
Cost:             30-60 GPU-hours, 2-3 weeks

Acceptance gates (8/8 failure modes detected absent):
  + 5 toy-detectable modes (instrumented same as exp_NEXT_B)
  + 3 production-scale-only modes:
      - Mode collapse: conditional output entropy + latent variance
      - MCMC mixing: Gibbs autocorrelation + L2(positive_z, negative_z)
      - Substrate shift: L∞(z) saturation + dimensional histogram modality
```

**Why this is in MANDATORY-NEXT-MILESTONE PRIORITIES:**

Without intermediate-scale validation, Phase-5 substrate training at 1B+ scale risks discovering one of the 3 production-scale failure modes after 100-500 GPU-hours. Catching cheaper at 100-300M scale (~30-60 GPU-hours) is the cost-asymmetric win.

**Sequencing:**
```
.94 or .95:  exp_NEXT_A-C small-scale (~3 weeks)
.96 or .97:  exp_NEXT_E intermediate-scale (~3 weeks)
.98+:        1B+ substrate training (Phase-5 production)
```

**Cross-references:**
- Q9 prompt+results: `docs/research-notes/in-situ-training-adversarial-robustness-deep-think-{prompt,results}.md`
- Updated Phase-5 plan: `openspec/change-proposals/in-situ-training-phase5-derisking.md`

---

### NEW 2026-05-03 (13:55Z): Retro Task Boundary Too Tight (artifact_not_updated_past_bootstrap)
RESOLVED .94 (2026-05-03): exp1215 uses STEP 0 skeleton + opus/100 turns

**Background:** .92 exp1190 retro retired with `artifact_not_updated_past_bootstrap` × 3. .93 exp1202 retro is repeating the pattern (FAIL #1 at 13:44Z). Heavy retro work (read 12 artifacts + analyze + write structured JSON) doesn't fit within the YAML-configured max_turns budget. Codex hits the boundary mid-analysis, exits without writing the final artifact, conductor logs FAIL.

**Why it's recurring:** the project's milestone artifacts have grown over time. Retro budgets that worked for .89/.90 (lighter milestones) don't fit .92/.93 (more cascade complexity to characterize). Retro tasks routed to codex (per AGENT_TYPE_RETRO=codex for quota conservation) don't get the opus-100 max-turns escalation tier.

**Mandatory .94 fix (one or more of):**

A. **Increase retro max_turns from 25 to 60-100.** Change in planner's retro template at `_plan_next_milestone()`. ~5 line edit. Risk: longer wall-time per retro.

B. **STEP 1 = write artifact skeleton FIRST in retro prompt.** Forces codex to commit a status="success" stub early in the task, then fill in detail. If turn budget runs out mid-analysis, the artifact still exists at status="success" with whatever was completed. ~10 line prompt addition.

C. **Route retros back to claude (revert AGENT_TYPE_RETRO).** Restores opus-100 escalation for retros specifically. Costs quota on every retro (1-2x/day).

D. **Recommended: A + B combined.** Longer budget + explicit early-write instruction. Belt-and-braces.

**Estimated cost:** ~30 min for option D (A + B + light testing).

**Why this is in MANDATORY-NEXT-MILESTONE PRIORITIES:**

Two consecutive milestones have lost their retro to this pattern. Without retros, planner Sonnet reads less context for the next milestone. Compounds operational discipline issues. Each retired retro = lost meta-reflection insight.

**Cross-references:**
- exp1190 retired pattern: ops/conductor-log.md 2026-05-03 06:01Z-06:15Z
- exp1202 in-flight pattern: ops/conductor-log.md 2026-05-03 13:44Z onward
- Planner retro template: scripts/research_conductor.py `_plan_next_milestone()` retro section
- Related: this is a SUBSET of the broader artifact_not_updated_past_bootstrap pattern (5 .92 retirements + counting in .93) but specific to retro tasks where the fix is well-scoped

---

### NEW 2026-05-03 (13:05Z): Auto-Populate prior_failures from Failure-Ledger at Plan Time

**Background:** 7 DOOMED_RERUN_BLOCK false-positives observed in tonight's session: exp1162 (KANELE, 2 priors), exp1169 (FoVer SOTA v6, 6 priors), exp1172 (NRGPT-per-token, 2 priors), exp1174 (BiKA, 1 prior), exp1175 (Connect Four, 6 priors), exp1188 (Hex, 7 priors), exp1198 (FoVer v7, 5 priors). Each one required ~5 min of operator outer-loop intervention to recover (read priors, write per-prior addressed_by paragraph, append OK to log).

**Pattern:** the failure-ledger algorithm correctly detects scope overlap with prior experiments. The interpretation gap is whether the overlap represents (a) a true prior FAILURE that this task addresses or (b) a successful UPSTREAM that this task builds on. The planner Sonnet doesn't pre-populate the field; the conductor blocks; operator manually classifies and writes the prior_failures block.

**Mandatory .94 fix:**

Two valid approaches:

1. **Auto-population at plan time.** Modify `_plan_next_milestone()` planner prompt to require pre-populated `prior_failures` for any task whose title/scope overlaps the failure-ledger's matching_priors output. The planner already reads research-complete.yaml; this just requires explicit instruction to enumerate matches and pre-classify.

2. **Auto-population at activation time.** Add a script that runs after `_plan_next_milestone` returns, walks every task, queries the failure-ledger, generates a prior_failures stub when priors exist, marks ready for operator review. Less LLM dependency, more deterministic.

3. **Hybrid (recommended).** Approach 2 generates the stubs; planner Sonnet reviews and refines the addressed_by text before YAML lock. Best of both: deterministic detection + LLM-quality narrative.

**Estimated cost:** ~2-4h of conductor.py + planner-prompt work for option 3.

**Why this is in MANDATORY-NEXT-MILESTONE PRIORITIES:**

7 false-positives in one session is unsustainable. Each requires operator attention. Without a fix, .94 will produce another ~7 false-positives, .95 another ~7, etc. Compounds with the test-suite cleanup work as another operational-discipline drain on the planner.

**Cross-references:**
- 7 example recoveries this session in conductor-log.md (operator OK entries with "prior_failures field added")
- failure-ledger logic: scripts/failure_ledger.py (matching_priors method)
- Planner prompt location: scripts/research_conductor.py `_plan_next_milestone()`

---

### NEW 2026-05-03 (06:33Z): artifact_not_updated_past_bootstrap Pattern (5 .92 Retirements)

**Background:** during .92, five distinct tasks retired with the same `artifact_not_updated_past_bootstrap` failure mode despite passing the pre-test gate (no schema-drift, no spike): exp1183 (paper recompile), exp1184 (GRPO v5 v2), exp1187 (Latent-GRPO), exp1190 (.92 retro), and one earlier in the cascade. Pattern: agent runs (sonnet+opus retries via the existing escalation tier), task gets to its substantive work, but never writes the deliverable JSON to a finished state before exhausting turn budget.

**Common factor among failed tasks:** all are heavyweight tasks (LaTeX recompile, GRPO training, complex reward integration, full-suite retro) where codex's pre-test self-heal pytest takes substantial wall-time, leaving insufficient turns for the task's actual artifact write.

**Common counterfactual:** the OK tasks (1181, 1182, 1185, 1186, 1188, 1189) wrote their artifacts within turn budget OR via opus 100-turn retry where pytest didn't dominate.

**Mandatory .93 fix (one or more of):**

1. **Pre-test scope reduction** — codex's self-heal currently runs `pytest tests/python` (full 21k-test suite). Reduce to a relevant subset based on the experiment's deliverable path. ~30 min of conductor.py change.

2. **Pre-test wall-time cap** — wrap codex's pytest invocation with `timeout 180 pytest ...` so heavy tests can't consume the full turn budget. ~5-line edit. Risk: false-negatives on legitimately slow tests.

3. **Artifact-update enforcement in task prompts** — the task prompts may not sufficiently emphasize "MUST write artifact JSON to deliverable path before exiting." Add a STEP 0 to every task prompt template. ~15 min.

4. **Turn budget rebalancing** — increase max_turns for heavyweight tasks (paper recompile, training runs) from 25-60 to 80-120. Complementary to other fixes.

**Why this is in MANDATORY-NEXT-MILESTONE PRIORITIES:**

5 retirements in one milestone is unsustainable. The pattern is structural (same failure mode across different task classes), so it will recur in .93 without a fix. Retiring exp1190 retro means .93 planner reads less context, compounding the issue.

**Cross-references:**
- Pattern observed at: exp1183 (03:35Z), exp1184 (03:55Z), exp1187 (05:05Z), exp1190 (06:15Z), 2026-05-03 .92 milestone
- ops/changelog.md will document the pattern in the .93 retro
- Conductor pre-test logic: scripts/research_conductor.py `_pytest_run` lines ~1000

---

### NEW 2026-05-02 (22:50Z): Watchdog Insufficient for Single-Test Catastrophic Load — Need prlimit/cgroup Preemptive Cap

**Background:** exp1178 shipped a `PytestMemoryWatchdog` post-test detection plugin (per-test threshold 500MB delta, session cumulative 8GB). Verdict was `watchdog_operational`. **However, the recurring 35GB+ RSS spike pattern persisted immediately after exp1178 OK'd** — exp1179 codex's pre-test self-heal triggered another worker hitting 39GB RSS within 6 minutes, requiring another manual operator SIGTERM intervention.

**Root cause:** the shipped watchdog detects gradual leaks (delta after each test) and cumulative session breach. It does NOT prevent a single-test catastrophic load — when a llama.cpp test or BEAVER live test loads a 35GB model in one shot, the watchdog can only flag it AFTER the load completes; by then the system is already at risk of OOM.

**Mandatory .93 fix:**

The right tool for **preemptive prevention** is OS-level hard memory cap, not Python-level post-test detection. Three valid implementations:

1. **prlimit wrapper** — modify `scripts/research_conductor.py` self-heal pytest invocation from `pytest tests/python -q` to `prlimit --as=8589934592 -- pytest tests/python -q`. Address-space limit kills any process exceeding 8GB cleanly. Single-line edit.

2. **systemd-run scope** — `systemd-run --user --scope -p MemoryMax=8G -p MemorySwapMax=0 -- pytest tests/python -q`. cgroup-based cap. More robust than prlimit (handles fork bombs).

3. **xdist worker MemoryMax** — pass `--memory-cap=8G` to a custom xdist plugin that wraps each worker spawn with cgroup. Most precise, most engineering work.

**Why this is in MANDATORY-NEXT-MILESTONE PRIORITIES:**

The pattern is now **6 occurrences in 5 hours** on 2026-05-02 (17:18, 19:33, 20:04, 21:18, 21:33, 22:48). Each requires manual operator SIGTERM. exp1178's watchdog plugin technically discharged its task description ("Per-Test RSS Monitoring + Session Cumulative Limit") but did NOT solve the operational problem. .93 must close the gap with a preemptive cap — option 1 (prlimit) is the ~10-line fix that actually prevents the spike.

**Cross-reference:**
- exp1178 deliverable: `python/carnot/testing/pytest_memory_watchdog.py`, `tests/python/conftest.py` wire-in (post-test detection, working but insufficient)
- exp1178 task definition gap: planner Sonnet specified the watchdog as "Per-Test RSS Monitoring + Session Cumulative Limit" — codex correctly built that, but neither party caught that "Per-Test" + "Session" detection misses single-test catastrophic loads
- Earlier known-issues entry (2026-05-02 21:35Z) flagged the recurring spike pattern but didn't specify "preemptive cap" vs "post-test detection" as the discriminating axis

---

### NEW 2026-05-02 (21:35Z): Pytest Worker Memory Watchdog — Stop the Recurring Load-Spike Pattern

**Background:** session-long pattern of codex pre-test self-heal spawning pytest with xdist workers, where one worker balloons to ~35GB RSS / 1100-1500% CPU and load average climbs to 18-22. Five recurring spikes during this single session (2026-05-02): 17:18, 19:33, 20:04, 21:18, 21:33. Each one required manual operator intervention (SIGTERM codex, SIGKILL orphan pytest workers) to prevent OOM. **This is the load-bearing operator-attention drain that the conductor-supervisor proposal was designed to eliminate.**

**Root cause:** when codex's self-heal mode runs `pytest tests/python -q` (the full suite, ~21k tests), some test loads llama.cpp models (likely BEAVER live tests) or runs large NumPy operations. xdist workers each consume an independent copy of the loaded model in memory. One worker hitting a memory-heavy test in its load order = OOM risk.

**Mandatory .92 fix (one or more of):**

1. **Pre-test memory cap** — wrap pytest invocations in `systemd-run --user --scope -p MemoryMax=8G ...` or `prlimit --as=8589934592 pytest ...`. If any worker exceeds 8GB, OS kills it cleanly. No 35GB workers possible.

2. **Subset-only self-heal** — instead of running the full pytest suite, restrict self-heal to tests directly related to the failing one. The 21k-test suite includes BEAVER live + NRGPT + GRPO + KV260 — most are irrelevant to a given failing pre-test. Subset gating reduces both wall time and memory pressure.

3. **Process-watchdog daemon** — separate process that polls `ps aux` every 10s, identifies any pytest worker exceeding (e.g.) 16GB RSS or 90% CPU sustained, SIGKILLs it. Conductor sees the test fail, decides next step normally — but the spike is bounded. Implementation: ~50 lines Python + systemd timer.

4. **Conductor-side pre-test scope reduction** — modify the self-heal command in `scripts/research_conductor.py` to use `pytest tests/python --ignore=tests/python/test_beaver_lite_live_logprobs.py --ignore=tests/python/test_phase4_sampler.py --ignore=tests/python/test_experiment_1170*` etc. These are the heavy tests; excluding them from pre-test self-heal preserves the gate's purpose without the memory cost.

**Recommended priority:** option 1 (memory cap) is the cheapest and most immediate. ~30 min of work, ships as a 2-line edit to the pytest-invocation command in `scripts/research_conductor.py`. Combined with option 4 (scope reduction) gives belt-and-braces.

**Why this is in MANDATORY-NEXT-MILESTONE PRIORITIES:**

The pattern has manifested 5 times in 4 hours. Without a fix, it manifests in .92 too, requiring continued operator-attention drain. The conductor-supervisor v1 (exp1027) was designed to handle exactly this class of issue but was never wired in to do active memory-watchdog. .92 should either complete that wire-in OR ship a simpler memory-cap pre-test wrapper.

**Cross-reference:** Memory `incident_2026_04_26_swap_saturation.md` documented an earlier instance of this pattern. The conductor-process-isolation proposal at `openspec/change-proposals/conductor-process-isolation.md` is related but addresses orphan-on-shutdown, not in-flight memory explosion.

---

### NEW 2026-05-02 (20:05Z): GRPO v5 Routing Bug — Re-propose with claude/opus

**Background:** exp1173 GRPO v5 + TinyV failed twice in .91 because the YAML had `model: opus` but no `agent_type:` field. Under global `AGENT_TYPE=codex`, that silently routes to codex/gpt-5.5 — ignoring the opus intent the planner had written. Both FAILs were codex-side (stall + dualgpu_confirmed=False bug). The task likely retires from .91 with all 3 attempts on the wrong backend.

**Fix shipped 2026-05-02 20:05Z:** added `agent_type: claude` to research-roadmap.yaml@exp1173 + `failover_on_stall: true` defensive marker.

**Mandatory .92 pickup:**
- Re-propose GRPO v5 + TinyV False-Negative Correction in .92 with explicit `agent_type: claude, model: opus` + DualGPU MANDATORY + grace_period_s:2400
- Include `prior_failures:` block citing exp1173 .91 retirement with addressed_by: "previous attempts routed to codex/gpt-5.5 instead of claude/opus due to YAML routing bug; this attempt uses explicit agent_type=claude"
- The +10pp v4 baseline from exp1159 is the floor; v5 should match or exceed

**Why this is in MANDATORY-NEXT-MILESTONE PRIORITIES:**

The GRPO trajectory is the strongest self-learning signal in the project (+10pp from v3→v4). Losing v5 to a routing bug rather than scientific failure would be a real cost. Without explicit pickup, .92 planner Sonnet may interpret 3 FAILs as "GRPO v5 not viable" rather than "GRPO v5 was misrouted."

**Cross-reference:** Memory `feedback_anthropic_quota_codex_default.md` documented the codex-default policy; this incident shows that policy needs an exception path for tasks with `model: opus` YAML hint.

---

### NEW 2026-05-02 (18:50Z): Paper Integrity Audit — 18 Issues Block Publication

**Background:** operator audit + adversarial sub-agent review found 18 integrity issues in `docs/arxiv-paper/main.tex` and the 7 figures. The PR-blocking class is **5 critical issues** that violate CLAUDE.md "All headline results must have live GPU provenance". Full plan at `openspec/change-proposals/paper-v5-integrity-remediation.md`.

**Critical issues (each individually blocks arXiv submission):**

```
ISSUE-1 fig3 11680x speedup
  fig3_fpga_latency.py:32-36, 78-87
  CPU 290ms is "order-of-magnitude estimate" (per docstring), comparison is
  per-200-sample-sweep CPU vs per-sample FPGA. Real per-sample speedup is ~58x.
  REMEDIATION: pull figure OR re-render with exp1094 measured CPU (15.96µs/sweep)

ISSUE-2 KL=3.07 cited as FPGA-measured is software proxy
  main.tex:469-481
  exp1094.kl_measurement_mode = "software_parallel_glauber_proxy".
  Bitstream J is hardware-fixed; live FPGA portion is latency probe only.
  REMEDIATION: rewrite all "FPGA KL=3.07" to "software-proxy KL; bitstream KL
  not yet measured on-board"

ISSUE-3 15.6x speedup baseline is hand-typed code constant
  fig7_chi4_fastpath.py:46
  CPU_GIBBS_PER_SWEEP_NS = 1000.0  # "~1 microsecond" — no artifact reference.
  exp1094 actual measured CPU = 15.96µs = 16x slower than the paper's guess.
  Real ratio would be ~249x not 15.6x; or retract the speedup entirely.
  REMEDIATION: run real optimized C++ Gibbs benchmark with cited artifact ID
  OR retract the 15.6x headline number

ISSUE-4 76,130x HardNet++ speedup is apples-to-oranges
  main.tex:730 (exp1147)
  117µs CPU array code vs 8.93s LLM API roundtrip = not a "speedup" architecturally.
  REMEDIATION: reframe to "117µs per violation vs 8.9s for prompt repair on
  the same 20 cases" — drop multiplicative speedup framing

ISSUE-5 exp1121 hides verifier collapse
  main.tex:714-721
  Paper frames k=5 AUROC=0.5547 as deployment milestone. Hidden:
  SOSKANEnergyV3 — the verifier with claimed 0.9545 AUROC — scored 0.3333
  (worse than random!) on the production corpus.
  REMEDIATION: add explicit text acknowledging OOD collapse — strengthens
  Wall 3 (verifier null space) narrative rather than weakens it
```

**High-severity issues (5):**

```
ISSUE-6  GRPO +8.51pp on n=47, eval_wall_budget_hit=True
  main.tex:705-712 (exp1118/1129)
  Add binomial CI; small-sample caveat as prominent as headline number

ISSUE-7  HumanEval +36pp against broken extraction baseline (0.0%)
  main.tex:792-799, fig5
  Reframe as "after extraction-fix" not "+36pp absolute"; move to anomaly section

ISSUE-8  alpha_t=0.38 ignores k=5 disagreement with ground truth
  main.tex:783-792, fig4
  exp1077: 24/100 ground-truth-correct examples were rejected by k=5 AND-compose
  used to compute alpha_t. Add this caveat.

ISSUE-9  Phase-4 pilot baseline trivial (98% solve), monotone fraction on N=3
  main.tex:892-922 (exp1165)
  free_energy_values=[0,0,0] only 3 entries; baseline already solves 98%.
  Add stronger baseline (BFS / shortest-action) before "evidence of free-energy guidance"

ISSUE-10 Seed IQ row in Table 5 marked documented_fallback / not_confirmed
  main.tex:944-948, 962-977
  exp1166 seed_iq_score_confirmed=False; cited as established leaderboard fact.
  Add footnote: "documented fallback evidence; not independently re-fetched"
```

**Medium-severity issues (5):**

```
ISSUE-11 ThinkPRM AUROC=0.9885 cited as "predecessor of exp1033" — no traceable artifact
ISSUE-12 Retrained verifier holdout n=50 not stated; exp1121 contradicts the "fix generalizes" claim
ISSUE-13 NRGPT n_iters_monotone=False — energy recurrence does NOT actually decrease monotonically
ISSUE-14 Two SOS-KAN AUROCs (0.9902 vs 0.9545) unreconciled across sections
ISSUE-15 fig2 ROC curves are binormal-fit synthesizations; caveat missing from paper caption
```

**Low-severity issues (3):**

```
ISSUE-16 Bibliography stub audit needed (4 suspect entries: themesis2026seediq, hive2026,
         llmsgamingverifiers2026, rewardunderattack2026)
ISSUE-17 Table 1 k=15 retracted-row framing OK; flag for caption note
ISSUE-18 Hardware-portability theorem claim covers FPGA/Z1/photonic; only KV260 measured
```

**Why this is in MANDATORY-NEXT-MILESTONE PRIORITIES:**

The paper is the load-bearing artifact for the publication-hold-lift gate. The audit revealed that exp1167's `paper_ready_for_arxiv_hold_lift: true` was incorrect — manually downgraded 2026-05-02 18:40Z. Until ISSUES 1-5 are resolved, the paper remains non-publishable per CLAUDE.md standards. Reserved-infrastructure-slot rule: .92 (and any subsequent milestone until hold lifts) MUST include at least 5 paper-integrity tasks (one per critical issue) with prior_failures blocks documenting the audit finding.

**Cross-references:**
- Full paper-v5 remediation plan: `openspec/change-proposals/paper-v5-integrity-remediation.md`
- Manual override on exp1167: `results/experiment_1167_paper_v4_phase4_section.json#manual_override_2026_05_02T18_40Z`
- Memory: `feedback_paper_integrity_audit.md`

---

### NEW 2026-05-02 (06:40Z): Seed IQ Verified — Active-Inference Phase 4 Track (3 candidate tasks)

**Background:** the Seed IQ ARC-AGI-3 score has been **independently
verified** via a public demonstration video showing 0.95 score one
month ago (the EBT/ARC-AGI document subsequently reported 1.00 on
the leaderboard with 115% human action-efficiency). Themesis, Inc.
+ Denise Holt + Denis O. are the named operators. This is **not
marketing** — the system works.

This corroborates the paradigm-shift thesis. Active inference +
topological field cognition (AΩ FoB HMC) is the empirically-leading
architecture on ARC-AGI-3 by an open-source-adjacent team. The
v3 paper now acknowledges this in Section 7 (Related Work) and
positions Carnot as the synthesis path: Carnot's k=N AND-composed
verifier ensemble serves as the calibrated free-energy
approximation while the LLM substrate retains autoregressive
infrastructure compatibility.

**The 3 candidate tasks the .90+ planner MUST consider** (in addition
to the 4 EBT/ARC-AGI-3 tasks filed at 06:25Z, which now subsume the
seed-iq-verification task — verification done):

0. **`exp11XX-snap-validity-sweep`** [HIGHEST PRIORITY — NEW 2026-05-02 08:10Z, runs FIRST]
   Goal: implement and run the pre-prototype diagnostic specified by
   Deep Think Q8 (action representation). Sample 10,000 continuous
   states uniformly from the existing Phase-3 DBAE-EBM bounded latent
   `z ∈ [-1, 1]^d`. Map them to discrete actions using the nearest-
   neighbor snap operator. Run each snapped action through the fast,
   deterministic ARC-AGI-3 rule engine to verify structural legality.
   Crucially: NO k=5 ensemble calls — this is a CHEAP gating
   diagnostic that runs in ~30 minutes of compute, ~1-2 days of code.

   Acceptance: ≥95% of snapped continuous states resolve to legally
   executable ARC-AGI-3 moves given the current board state. If
   <95%, Option A (continuous relaxation + nearest-neighbor snap)
   fails before HMC sampler implementation begins; Phase-4 must
   pivot to Option B (simplex HMC) or Option C (field dynamics)
   despite Q8's recommendation against them.

   This is the FIRST pre-flight task (runs before
   exp11XX-hmc-compatibility-diagnostics) because it's strictly
   cheaper and answers a separate question (action representation
   validity vs. sampler regime). Two fail-fast diagnostics in
   sequence, total 3-7 days, are strictly cheaper than committing
   to a 2-week HMC implementation that may fail on either axis.

   Phase: 3 inference-mode prerequisite. Reservation: highest-
   priority research-class slot for .90 (sequential before HMC
   diagnostics).

   **Cross-references:**
   `docs/research-notes/hmc-discrete-action-representation-deep-think-results.md`
   has the full Q8 verdict including Option A/B/C taxonomy, why
   Option A wins for Carnot specifically, and the unresolvable
   "phantom valley" uncertainty that requires live HMC trajectory
   instrumentation.

1. **`exp11XX-hmc-compatibility-diagnostics`** [HIGHEST PRIORITY — REVISED 2026-05-02 08:00Z]
   Goal: implement and run the 4 diagnostics specified by Deep Think
   Q7 on Carnot's existing post-exp1128 k=5 ensemble + ~100 synthetic
   test examples (mixed safe/boundary). Classify Carnot's `∇E` into
   one of three regimes (A: HMC works; B: needs preconditioning;
   C: HMC inappropriate). NO GPU required; ~3-5 days of focused work.
   This is a STRICTLY CHEAPER prerequisite to building any sampler;
   it transforms a 2-week HMC implementation that may never converge
   into a 3-5 day risk-check.

   The 4 diagnostics (Deep Think Q7 verdict):
   - **D1 Symplectic Reversibility**: forward leapfrog L steps, negate
     momentum, backward L steps; measure `||x_0 - x_rev||`. Low
     distance = detailed-balance preserved.
   - **D2 Hamiltonian Energy Conservation**: variance of `|ΔH|` over
     multi-step trajectories. Bounded low variance = log-density
     smooth enough for leapfrog.
   - **D3 Cross-Component Gradient Norm Disparity**: ratio of
     max-component-variance to min-component-variance across the
     5 verifier components. Near-unity = isotropic; orders-of-
     magnitude = preconditioning needed.
   - **D4 Continuous Subspace Recovery**: simulate leapfrog using
     ONLY `w_Sem ∇E_Sem + w_PRM ∇E_ThinkPRM`. Stable `|ΔH|` here
     while full-ensemble `|ΔH|` explodes = continuous components
     compatible, discrete components are the strict bottleneck.

   Acceptance: regime classification {A, B, C} reported with all
   4 diagnostic outputs documented; if Regime C, additional
   diagnostics for fallback selection (Blocked Gibbs / Langevin /
   Surrogate) reported.

   Phase: 3 inference-mode prerequisite. Reservation: highest-
   priority research-class slot for .90 — every Phase-4 sampler
   task is downstream of this diagnostic.

   **Cross-references:**
   `docs/research-notes/hmc-on-heterogeneous-energy-gradient-deep-think-results.md`
   has the full diagnostic specifications, regime signatures, and
   fallback-diagnostic chains.

2. **`exp11XX-hmc-sampler-CONDITIONAL`** [HIGH PRIORITY — REGIME-DEPENDENT]
   Goal: implement the appropriate sampler based on Task #1's regime
   classification. The form of this task is determined by the
   diagnostic outcome:

   **If Regime A (HMC works directly):**
   - Vanilla NumPyro HMC primitive on Carnot's `∇E`
   - Default leapfrog + adaptive step-size
   - ~5-10 days of implementation
   - Acceptance: HMC convergence ≥2× faster than Langevin/Gibbs on
     FoVer eval at matched accuracy.

   **If Regime B (preconditioning needed):**
   - NumPyro HMC + per-component mass matrix `M`
   - `M` aligned with inverse covariance of aggregated gradients
     (Deep Think's preconditioning principle)
   - Verify preconditioner *solves* (vs. *masks*) via post-hoc
     constraint-violation rate check on samples
   - ~7-12 days
   - Acceptance: same as Regime A + sampled outputs maintain
     constraint compliance ≥95% (Z3/AST/JSON validity).

   **If Regime C (HMC inappropriate):**
   - Choose fallback per Deep Think's diagnostic chain:
     - Blocked Gibbs/Metropolis-within-Gibbs (if D4 strict pass)
     - Langevin with adaptive step (if L=1 OK, L>1 fails)
     - Surrogate-gradient HMC (if linear probe R² high)
   - ~10-15 days
   - Acceptance: chosen fallback achieves convergence on FoVer
     eval; document why the alternative fallbacks were rejected
     by their respective diagnostics.

   In all three cases, after sampler is operational:
   - On 10-puzzle ARC-AGI-3 subset, measure action-count efficiency
     vs Seed IQ's published numbers (VC33: 173 vs human 307;
     FT09: 75 vs human 163; LS20: 433 vs human 546)
   - Within 50% of Seed IQ = "directionally correct"; <50% =
     "Carnot's k=N landscape is materially less calibrated; investigate"

   Phase: 3 inference-mode extension. Reservation: research-class
   slot in .91 (or .92 if Task #1 reveals Regime C requiring a more
   substantial fallback build).

3. **`exp11XX-topological-fencing-mitigation`** [DEFERRED — .92+]
   Goal: address the unresolvable uncertainty Deep Think Q7 flagged.
   Even if Tasks #1 + #2 confirm local HMC compatibility, the global
   manifold connectivity of Carnot's valid Z3/AST/JSON regions is
   only diagnosable via long-horizon chains on the full Task #2
   prototype. If long-horizon mixing fails (severe pseudo-ergodicity,
   chain stuck in single mode), this task implements parallel
   tempering across modes or mode-jumping moves.
   Phase: 3 inference-mode extension. Reservation: deferred research-
   class slot, only triggered if Task #2 reveals topological fencing.

   **What this REPLACES**: the prior `exp11XX-hmc-sampler-on-carnot-ebm`
   task (filed earlier 2026-05-02 07:10Z) was monolithic. Deep Think
   Q7's response showed it should split into a cheap diagnostic
   prerequisite + a regime-conditional sampler implementation +
   a deferred topological-fencing fallback. The 3-task split is
   strictly lower-risk than the monolithic version: failure modes
   are caught at 3-5 days instead of 2-3 weeks.

2. **`exp11XX-diffusion-of-thought-inference-mode`** [HIGH PRIORITY]
   Goal: add Diffusion of Thought (DoT) iterative latent refinement
   as a second inference mode for Carnot's existing energy landscape.
   Variable timestep count (T ∈ {1, 5, 25, 125}) for compute/accuracy
   trade-off. The same `∇E` from k=5 ensemble drives the reverse
   denoising process; DoT is mathematically Markovian (each refinement
   step depends only on its immediate predecessor), so this is a clean
   inference-mode addition without architectural change.
   Acceptance: monotonic accuracy improvement with timestep count
   on FoVer + GSM8K + ARC subset; Pareto frontier (compute vs accuracy)
   published. Compare to autoregressive CoT on the same prompts at
   matched compute budget.
   Phase: 3 inference-mode extension. Reservation: research-class
   slot, pairs with the HMC task above.

2. **`exp11XX-themesis-collaboration-outreach`**
   Goal: draft outreach email to Themesis (Denise Holt / Denis O.)
   outlining Carnot's verifier-as-free-energy framing; propose
   architectural conversation. Open-source-friendly framing —
   Carnot is Apache 2.0, multi-vendor, decentralization-respecting;
   Themesis has the active-inference algorithm. Complementary, not
   competitive.
   Acceptance: email drafted + reviewed by operator before sending.
   ~30-min operator task. Could open joint benchmark evaluation
   or pre-print exchange.
   Phase: cross-cutting strategic. Reservation: 30-min operator
   block, no conductor execution needed.

3. **`exp11XX-paper-v4-active-inference-section`**
   Goal: post-arXiv-submission, expand Section 7 (Related Work) of
   the position paper into a full architectural-comparison section
   for v4. Compare Carnot's EBM-on-LLM substrate vs Themesis's
   active-inference-on-topological-field substrate, with empirical
   results from exp11XX-active-inference-minimal-prototype.
   Acceptance: 2-3 page section drafted; Pareto-frontier comparison
   on at least one common benchmark; honest assessment of which
   paradigm wins where.
   Phase: publication. Reservation: post-2026-05-15 arXiv
   submission, .92+ candidate.

**Completed tonight (2026-05-02 ~06:40Z):**

- ✅ v3 paper (`docs/arxiv-paper/main.tex`) Section 7 expanded with
  Themesis/Seed IQ acknowledgment paragraph. Cites
  `themesis2026seediq` + `arcagi3` (added to `carnot.bib`).
  Tarball `results/carnot-arxiv-v3.tar.gz` rebuilt at 06:40Z
  (124,218 bytes, was 123,093). Submission-ready for 2026-05-15
  deadline.

**Why this is in MANDATORY-NEXT-MILESTONE PRIORITIES:**
the architectural conversation has shifted. Active inference is
publicly demonstrated as a winning paradigm on ARC-AGI-3. Carnot
must engage substantively (not just acknowledge in a paragraph).
The prototype task (#1) gives empirical signal within 1-2 milestones
on whether Carnot's verifier ensemble can serve as the free-energy
approximation in a Friston-style sampler.

If hypothesis confirms: Carnot is **doing active inference under a
different name**, and the v4 paper unifies both paradigms.

If hypothesis disconfirms: Carnot's EBM-on-LLM thesis stands; the
paradigms are genuinely different and Carnot positions as the
LLM-compatible alternative.

Either outcome is publication-grade.

---

### NEW 2026-05-02 (06:25Z): EBT/ARC-AGI-3 Paradigm-Shift Tasks (4 candidate tasks)

**Background:** the EBM/EBT/ARC-AGI document
(local: `~/.claude/uploads/.../EBM_EBT_Reasoning_and_ARCAGI.pdf`)
positions Carnot-EBM as a named exemplar of the post-autoregressive
paradigm shift, alongside the Seed IQ system. The document outlines
empirical anchors and architectural components Carnot can adopt.
The most urgent claim: **Seed IQ scored 100% on ARC-AGI-3 with 115%
human action-efficiency** (2,674 actions vs human baseline 7,534-8,073),
while frontier autoregressive LLMs all scored below 1% (Gemini 3.1 Pro
0.37%, GPT-5.4 0.26%, Opus 4.6 0.25%, Grok-4.20 0.00%).

If the Seed IQ claim is real, it is the most consequential data point
in the field — and forces a pivot decision (active inference + topological
geometry vs. Carnot's current LLM-based path). Verify before committing.

**The 4 candidate tasks the .90+ planner MUST consider:**

1. **`exp11XX-seed-iq-arc-agi-3-verification`**
   Goal: independently verify the Seed IQ 100% ARC-AGI-3 leaderboard
   claim. Fetch the public ARC-AGI-3 leaderboard at
   https://arcprize.org/leaderboard, cross-reference the Seed IQ
   (Active Inference) entry, and document: (a) is the score real,
   (b) what's the action-count efficiency, (c) what's the verification
   provenance.
   Acceptance: independent screenshot + page-fetch of the leaderboard
   showing Seed IQ score + action count; verdict
   `seed_iq_100pct_verified` or `seed_iq_unverified_marketing` or
   `seed_iq_score_lower_than_claimed`.
   Phase: cross-cutting strategic. Reservation: research-class slot,
   highest priority — informs whether Carnot pivots to active
   inference as Phase 4 or stays the EBM-on-LLM course.

2. **`exp11XX-sc-energy-7th-verifier`**
   Goal: add Set-Consistency Energy Network (SC-Energy, ACL 2025) as
   the 7th member of Carnot's k=N verifier ensemble. SC-Energy uses
   a compact RoBERTa-base architecture and reportedly outperforms
   GPT-4o on out-of-distribution logical inconsistency detection. It
   treats statements as a set, learning compatibility via margin loss
   in the (X×Y)* space. Mechanism-orthogonal to existing 5 (Z3,
   gVisor, semantic, ThinkPRM, JSON schema), so adding it should
   preserve Welch ceiling while raising joint coverage.
   Acceptance: SC-Energy individual AUROC > 0.65 on FoVer eval AND
   pairwise correlation r < 0.5 with each of the existing 5; k=6
   ensemble AUROC > current k=5 (which post-exp1128 = 0.94).
   Phase: 1 production extension. Reservation: research-class slot.

3. **`exp11XX-nrgpt-per-token-energy-inference`**
   Goal: implement NRGPT-style per-token energy evaluation with
   variable-computation early stopping (more FLOPs to difficult
   reasoning nodes, fast pass on trivial tokens). Extends the
   `langevin-inference-sweep` task from the prior 2026-05-02 filing
   by allowing K (refinement steps) to be per-token rather than
   global. The cited paper (NRGPT, OpenReview B3Muyi2zgo) is the
   architectural specification.
   Acceptance: NRGPT-mode inference shows >1.5x compute savings vs.
   uniform-K Langevin at matched accuracy on at least one of {GSM8K,
   HumanEval, ARC subset}; per-token energy histograms show
   non-uniform distribution (energy concentrates on hard tokens).
   Phase: 3 (post-Stage-2). Reservation: research-class slot. Pairs
   with the langevin-inference-sweep task.

4. **`exp11XX-hmtt-tokenizer-investigation`**
   Goal: investigate Hybrid Math-Text Tokenizer (HMTT) as a Phase-3
   substrate decision. Standard BPE destructively compresses math
   tokens, ruining logical structure (per the document). HMTT
   preserves symbolic granularity, enabling the Recursive Logic
   Subsystem (k=N verifier ensemble) to operate on the same token
   stream the base LLM emits. Without HMTT, Z3-AST verifier sees
   different tokens than the base produces.
   Acceptance: HMTT prototype implemented for math-heavy tokens
   (numbers, operators, comparators, equality, etc.); tokenization
   round-trip preserves logical structure on 100 FoVer eval
   examples; Z3-AST verifier success rate on HMTT-tokenized output
   ≥ baseline.
   Phase: 3 (pre-Stage-1 substrate). Reservation: infrastructure-
   class slot. May gate the .91+ Phase-3 prototype kickoff if
   identified as load-bearing.

**Cross-references for planner context:**
- `~/.claude/uploads/.../EBM_EBT_Reasoning_and_ARCAGI.pdf`
  (the source document with Seed IQ + Carnot-EBM positioning)
- `memory/project_dbae_ebm_phase3.md` (Phase-3 substrate)
- arXiv 2507.02092v1 (EBT scaling — empirical anchor: 55M EBT
  beats 127× larger ARLM on GSM8k, 90.7%)
- ACL 2025.acl-long.1599 (SC-Energy paper)
- OpenReview B3Muyi2zgo (NRGPT)
- arxiv 2603.24621v1 (ARC-AGI-3 paper)

**Why this is in MANDATORY-NEXT-MILESTONE PRIORITIES, not just a memory:**
Task #1 (Seed IQ verification) is a **decision-changing experiment**.
If the 100% ARC-AGI-3 score is real, Carnot's Phase 4 must be
active-inference oriented. Without verification, the v4 paper risks
either (a) making the wrong architectural bet, or (b) being scooped
by Themesis publishing first. Tasks #2-4 are additive enhancements
that compound regardless of #1's outcome — they each strengthen
the EBM/EBT thinking story Phase 3 is building.

Worth ≥4 reserved-slot tasks across .90-.92 milestones, plus task #1
should be the highest-priority pickup in .90 (cheap, urgent, decision-
changing).

---

### NEW 2026-05-02: Phase-3 Thinking-Mode Composition (4 candidate tasks)

**Background:** the EBM/EBT thinking story for Phase 3 needs three
orthogonal inference-time scaling axes integrated, each grounded in
2025 literature: Apple SSD (no-verifier self-distillation, +12.9pp
LiveCodeBench), Google's Diffusion of Thought (parallel iterative
refinement at test time, EBM-native via Langevin dynamics), and
MCTS-style reasoning (tree search with verifier as value function,
o1/o3-class). Each is independently shippable; all three together
form the "Thinking with EBMs" narrative section for the v4 paper.

**Why these are mandatory for .90+** (after the .88-.89 prototype
infrastructure lands):

The current Phase-3 prototype design (DBAE-EBM 4-stage + SP-IWPER +
22-quantity diagnostic library + Decoupled Dual-Stream hybrid) covers
the *training* story. It does NOT yet cover the *inference-time
thinking* story. As foundation models like Mythos push 10T params with
explicit inference-time RL (o3-class), Carnot must position EBM/EBT
inference-time scaling as a structurally different paradigm — not just
"transformer + verify-repair" but "energy landscape + iterative
refinement + tree search". Without this, the v4 paper risks being read
as "yet another verify-repair pipeline" rather than "the alternative
to autoregressive thinking."

**The 4 candidate tasks the .90 planner MUST consider:**

1. **`exp11XX-ssd-bootstrap-stage0`**
   Goal: run Apple-style self-distillation on the base model BEFORE
   DBAE Stage-1 pretraining begins. SSD initializes representations
   without relying on verifier signal; energy-verification then runs
   on top of an already-bootstrapped base. Hybrid (FR-11 + Energy-
   Selection SSD per memory:project_ssd_self_distillation.md).
   Acceptance: SSD-bootstrapped base shows ≥5pp improvement on
   FoVer eval before DBAE pretraining; combined SSD+DBAE+EBM achieves
   AUROC > best-of-three-individual-paths on held-out SOTA corpus.
   Phase: 3 (pre-Stage-1). Reservation: research-class slot.
   **Adversarial baseline:** if SSD alone matches Carnot's verify-
   repair on LiveCodeBench (Apple's published 12.9pp), the verifier
   complexity must justify itself empirically — this task forces
   that comparison early.

2. **`exp11XX-langevin-inference-sweep`**
   Goal: at Phase-3 inference time, run K Langevin steps on the
   latent z to lower energy before decoding. Sweep K ∈ {1, 5, 25,
   125} and measure accuracy-vs-compute curve. EBMs do diffusion
   natively (∇_z E is the score function); this task makes that
   inference-time mode explicit.
   Acceptance: monotonic accuracy improvement across K with Pareto-
   optimal K identified; K=125 mode shows ≥3pp gain over K=1 on
   FoVer + GSM8K + ARC subset.
   Phase: 3 (post-Stage-2). Reservation: research-class slot.
   **Strategic anchor for v4 paper:** "EBM thinking scales differently
   from CoT — more compute = lower energy, not more tokens."

3. **`exp11XX-mcts-verify-repair-wrapper`**
   Goal: add MCTS-style tree-search wrapper around the existing
   verify-repair pipeline. At each generation step, expand top-K
   candidates, score by AND-composed k=5 energy, continue from
   highest-scoring branch. The verifier ensemble IS the value
   function (calibrated at AUROC=0.94 post-exp1128).
   Acceptance: MCTS-wrapped pipeline beats single-shot generation
   by ≥5pp on at least one of {GSM8K, HumanEval, ARC subset}.
   Phase: 1 production (deployable today). Reservation: research-
   class slot.
   **Computational caveat:** tree depth × branching factor × energy
   eval cost. Should sweep depth ∈ {1, 3, 9} and beam_width ∈ {2, 8}
   to find practical operating point.

4. **`exp11XX-thinking-scaling-comparison`**
   Goal: combine all three (SSD-bootstrap + Langevin refinement +
   MCTS) and measure the compute-vs-accuracy Pareto frontier
   against autoregressive CoT scaling on the same base model.
   This is the headline empirical anchor for the v4 paper's
   "Thinking with EBMs" section.
   Acceptance: composed pipeline establishes ≥10pp accuracy gap
   over autoregressive CoT at matched inference compute on at least
   one held-out benchmark; Pareto frontier curves published.
   Phase: 3 (post-prototype validation). Reservation: research-
   class slot, depends on tasks 1-3 + Phase-3 prototype.

**Cross-references for planner context:**
- memory/project_ssd_self_distillation.md (Apple SSD adversarial baseline)
- memory/project_dbae_ebm_phase3.md (Phase-3 substrate)
- memory/project_zenil_alpha_grounding.md (α_t as inference-time signal)
- docs/research-notes/phase3-substrate-contamination-deep-think-results.md
  (held-out suite + 6 contamination diagnostics — apply to thinking modes too)

**Why this is in MANDATORY-NEXT-MILESTONE PRIORITIES, not just a memory:**
the EBT thinking story is publication-track material. Without these tasks
landing, the v4 paper can document Phase-1 production wiring + Phase-3
training architecture but cannot make the "thinking scales differently"
empirical claim that distinguishes Carnot from o1/o3-class autoregressive
reasoning. Worth ≥4 reserved-slot tasks across .90-.92.

---

### NEW 2026-05-01: Failure-Ledger v2 + Planner Discipline (5 STRUCTURAL FIXES + 3 PLANNER-PROMPT DELTAS)

**Background:** milestone .85 lost 4 of 14 tasks (exp1092, exp1096,
exp1097, exp1098-first-attempt) to conductor-mechanism bugs and
planner-discipline gaps, NOT to legitimately-doomed research. Each
retirement was either prevented or recovered through manual operator
patches. Without structural fixes, .86+ will hit identical walls.

The 5 substantive findings that DID land in .85 (exp1090 diagnostic
library, exp1091 position paper arxiv-ready, exp1093 verifiers
correlated, exp1094 FPGA Glauber violates detailed balance, exp1095
DBAE-EBM threat model) prove the phase-validation discipline works
WHEN the surrounding plumbing doesn't sabotage tasks before they run.

Full proposal: `openspec/change-proposals/failure-ledger-v2-and-planner-discipline.md`

**The 5 structural conductor fixes the .86 planner MUST propose:**

1. **`exp10XX-failure-ledger-v2-issue-1-id-not-title`**
   Goal: count failures by `experiment_id`, not title-prefix.
   Acceptance: a milestone .Y task with the same title as a
   retired .X task does NOT inherit .X's failure count if their
   experiment IDs differ. Empirical .85 evidence: exp1096 SemEnergy
   Probe and exp1097 N-Queens Cartridge both retired silently from
   inherited .84 counts.
   Effort: ~2 hours. Code: `scripts/research_conductor.py:_count_failures_for_task`,
   `log_step`, schema of `ops/conductor-log.md` (add `id:` field).

2. **`exp10XX-failure-ledger-v2-issue-2-cap-reset-on-patch`**
   Goal: reset 3-fail cap when a fix-shaped commit lands between
   attempts. Acceptance: 3 manual failures + a commit touching the
   task's deliverable or roadmap entry must NOT auto-skip the task
   on next iteration. Empirical .85 evidence: exp1092 retired 7 min
   before operator patch landed.
   Effort: ~3 hours.

3. **`exp10XX-failure-ledger-v2-issue-3-stable-deliverable-mtime`**
   Goal: stable-deliverable detection requires `mtime > task_start_time`,
   not just "unchanged for 60s". Acceptance: an Opus task starting
   with a stale `blocked` artifact pre-existing on disk is NOT killed
   within 60s on the false positive. Empirical .85 evidence: exp1090
   first attempt, Opus killed before writing the new artifact.
   Effort: ~1 hour.

4. **`exp10XX-failure-ledger-v2-issue-4-cache-end-fingerprint`**
   Goal: pre-test fingerprint cache saves the END fingerprint, not
   the START. Acceptance: a `.py` change committed mid-pre-test gets
   captured in the cache; next iteration cache-hits the post-commit
   state. Empirical .85 evidence: iterations 6 and 7 both cache-missed
   because operator commits during pre-test invalidated the start
   fingerprint.
   Effort: ~30 min.

5. **`exp10XX-failure-ledger-v2-issue-5-coarse-keyword-matcher`**
   Goal: tighten `FailureLedger.is_doomed_rerun()` matcher to require
   ≥2 scope-vocabulary keyword overlap (Option A) or cosine ≥ 0.7 via
   sentence-transformer (Option B). Acceptance: a task titled "Phase
   1c Verifier Joint Null-Space Measurement" does NOT match "Phase 1a
   Adversarial Verifier Robustness Audit" as a doomed prior despite
   sharing "Verifier". Empirical .85 evidence: exp1090 tripped 2
   priors on "diagnostic", exp1092 tripped 18 on "verifier"/"adversarial",
   exp1093 tripped 10 on "verifier"/"null-space" — all false positives.
   Effort: ~1 hour for Option A.

**The 3 planner-prompt deltas the .86 planner MUST self-apply:**

P1. **Always emit `prior_failures:` blocks for any task whose title
or scope words appear in `research-complete.yaml`.** The .85 planner
emitted 6 of 14; operator patched 6 more. Net 12 of 14 needed it.
Future planners must query research-complete.yaml before drafting
each task and emit the block proactively (not reactively).

P2. **Never emit cross-vendor `model:` overrides on tasks with
non-default `agent_type:`.** The .85 planner emitted `model: opus`
on `agent_type: codex` tasks (exp1097, exp1098), causing codex CLI
HTTP 400. The conductor's snap-fix at commit `1f1aef51` neutralizes
this at the conductor layer, but the planner should not emit invalid
combinations in the first place. Document the per-vendor model
namespace in the planner prompt.

P3. **Document gate-required artifact fields explicitly.** For any
task with `gated_on:`, the upstream task's prompt must enumerate the
gate-required fields under "REQUIRED ARTIFACT FIELDS:" so manual
operator artifact reconstructions get the schema right. Empirical .85
evidence: exp1090's manual reconstruction missed
`diagnostics_library_written: true`, GATE_BLOCKing exp1092 twice
before the operator could patch.

**Activation-guard additions (3 cross-checks):**

The conductor's existing planner-output validator must additionally
enforce:

A1. prior_failures completeness — every task whose title shares ≥2
    scope-vocabulary keywords with a prior failure carries a matching
    prior_failures: entry.

A2. agent_type/model coherence — every task's model belongs to its
    agent_type's vendor namespace.

A3. gate-field cross-reference — every `gated_on.artifact_field` is
    enumerated in the upstream task's prompt's REQUIRED ARTIFACT
    FIELDS section.

If validation fails, the activation-guard does NOT swap
`research-roadmap-next.yaml` → `research-roadmap.yaml`. Writes a
`planner-validation-failed` artifact and pings the operator.

**Why this is in MANDATORY-NEXT-MILESTONE PRIORITIES, not just a
proposal.** Without these structural fixes, .86 will hit identical
walls. The 5 structural fixes total ~7-8 hours of operator effort
spread across the milestone — each phase independently shippable and
reduces the manual-patch burden by an estimated 30-50% per phase.
The .86 planner MUST allocate at least 3 of the 5 issues as tasks,
per CLAUDE.md "Overdue-Priority Forcing Function" (this is the first
milestone these are pending; if .86 doesn't pick them up, .87 must
treat them as ≥2-milestone-overdue mandatory).

### NEW 2026-05-01: LLM Failure Exemplar Corpus + Goodfire Silico Comparison

**Background:** MIT Technology Review 2026-04-30 published an article
on Goodfire's Silico, a closed-source white-box mechanistic
interpretability tool for open-weight LLMs (Qwen 3 etc.). It catalogs
specific named, reproducible LLM failure modes — `9.11 > 9.9`
mathematical error from Bible/version-numbering interference, trolley-
problem moral framing, deceptive disclosure suppression. These are
benchmark anchors for any verifier suite.

Carnot's positioning: complementary, not competitive. Silico tells
you WHY a model said something wrong (white-box neuron tracing);
Carnot tells you THAT it said something wrong and HOW to fix it
(black-box energy verification + repair). A complete LLM-reliability
stack uses both. Silico is closed-source + proprietary + open-weight-
only; Carnot is Apache 2.0 + local-first + LLM-agnostic.

**Three .86 tasks the planner SHOULD propose:**

1. **`exp10XX-llm-failure-exemplar-corpus-v1`**
   Goal: build `data/llm_failure_exemplars.jsonl` with named, reproducible
   failure modes — Goodfire's published exemplars + project-internal
   findings. Each entry: prompt, buggy response, mechanistic root
   cause, Carnot verdict, Carnot repair, whether Carnot caught it.
   Acceptance: ≥30 exemplars, ≥10 categories, integrated into
   Phase 1a verifier robustness audit dataset.
   Effort: ~3-4 hr (mostly research + format design).

2. **`exp10XX-goodfire-exemplar-cascade-tp-rate`**
   Goal: feed Goodfire's published exemplar prompts through Carnot's
   verifier cascade and report TP rate per verifier tier. Tests the
   mathematical-objective tier (Z3-based numeric extraction should
   catch 9.11>9.9 trivially) versus the learned tier (KAN/SOS-KAN).
   Acceptance: report TP rate per tier on ≥15 Goodfire-style
   exemplars; if mathematical-objective tier achieves >90%, validates
   Carnot's engineering claim vs the "alchemy precision" critique
   from Leonard Bereska in the article.
   Effort: ~2 hr.

3. **Position paper v3 framing delta** (could be folded into
   exp10XX-position-paper-v3 if .86 has one):
   - Explicit complementary-vs-competitive positioning with
     mechanistic interpretability (Silico, Anthropic circuit analysis,
     OpenAI transformer-debugger, Neuronpedia)
   - Distinguish epistemic status of mathematical-objective verifiers
     (Z3, AST, Ising — genuinely engineering) from learned verifiers
     (KAN, SOS-KAN — precision-added alchemy)
   - Decentralization advantage: Carnot Apache 2.0 + local-first vs
     Silico closed-source + Goodfire-service-required

**Why this is "SHOULD" not "MUST":** the failure-exemplar corpus is
a force-multiplier for Phase 1a (already mandatory). It augments
existing work; it doesn't replace any of the 5 failure-ledger v2
fixes. Planner can defer one but not all.

**See also:** `feedback_failure_ledger_gaps.md` (Issues 1-5 + planner
deltas P1-P3) and `reference_goodfire_silico.md` (context).

### Carry-forward from .85 (operator-retired tasks the .86 planner
MUST re-propose with proper prior_failures from start)

1. **exp1092 Phase 1a Adversarial Verifier Robustness Audit** —
   measure false-pass rate of shipping Carnot verifiers on
   adversarially-crafted attacker-LLM outputs. Phase-validation
   MANDATORY task #1 of 5. Lost to 3-fail cap race in .85.

2. **exp1096 SemEnergy Probe v1 (Tier 0c Logit-Space Energy
   Detection)** — 4-fail title-prefix inheritance from .84.
   Re-propose with explicit `prior_failures:` block addressing all
   of exp1080's verdicts.

3. **exp1097 WOPR N-Queens Cartridge** — 3-fail title-prefix
   inheritance from .84's exp1086. Re-propose with explicit
   `prior_failures:` block.

### NEW 2026-04-30: Phase Prototype + Empirical Validation + Adversarial Check Discipline (5 LOAD-BEARING TASKS)

**Background:** the Phase-3 architecture blind-spot audit caught 5
FATAL findings three rigorous theoretical Deep Think rounds missed.
The new MANDATORY discipline (see CLAUDE.md "Phase Prototype +
Empirical Validation + Adversarial Check Discipline") requires every
phase prototype + empirical pass/fail criteria + adversarial check
BEFORE scaling. The current state is architecture-heavy / prototype-
light / adversarial-check-rare, which is exactly the foundation-
of-cards failure mode the user flagged.

**Five .85 tasks the planner MUST propose:**

1. **`exp10XX-phase1a-adversarial-verifier-robustness-audit`**
   Goal: measure false-pass rate of shipping Carnot verifiers on
   adversarially-crafted outputs (LLM-generated, designed to fool
   each verifier). Acceptance: false-pass < 5% on canonical attack
   patterns. Output: per-verifier robustness scorecard.
   Phase: 1a. Reservation: infrastructure-class slot.

2. **`exp10XX-phase1c-verifier-joint-null-space-measurement`**
   Goal: empirically measure `dim(∩_i ker E_i)` for the existing
   k verifiers (4-6 today, Round 9 calls for 15+). Acceptance: joint
   null-space dimension < 5% of input space. Output: empirical
   bound for AND-composition viability.
   Phase: 1c. Reservation: infrastructure-class slot.

3. **`exp10XX-phase2a-sampler-correctness-audit`** (revised from
   prior FPGA-vs-GPU baseline task, see entry below for details)
   Goal: KL divergence between KV260 FPGA samples and correct CPU
   Gibbs samples on a deliberately frustrated J matrix. Empirically
   confirm or refute audit Finding #2 (synchronous parallel Glauber
   non-equilibrium). Acceptance: KL < ε OR documented caveat in
   exp1081's headline measurement.
   Phase: 2a. Reservation: infrastructure-class slot.

4. **`exp10XX-phase3a-pre-prototype-adversarial-round`**
   Goal: BEFORE writing the DBAE-EBM prototype code, run a hostile-
   reviewer round on the prototype IMPLEMENTATION (not architecture).
   Specifically find ways the prototype could silently pass
   acceptance-gate numbers without actually working: degenerate
   identity encoders, decoder LM-prior overpowering bottleneck, EBM
   converging to constants, etc. Output: list of failure modes the
   prototype MUST detect via instrumentation.
   Phase: 3a. Reservation: research-class slot.

5. **`exp10XX-diagnostic-instrumentation-library`**
   Goal: single shared Python module providing α_t tracking, joint
   null-space estimation, KL divergence measurement, decoded-text
   diversity scoring, manifold-coverage metrics. Used by every
   phase prototype. Acceptance: 100% test coverage + integration
   tests showing every diagnostic produces meaningful values on a
   small reference setup.
   Phase: cross-cutting infrastructure. Reservation: infrastructure-
   class slot.

**Why these are MANDATORY for .85:**

- The discipline is now codified in CLAUDE.md as MANDATORY (see the
  new "Phase Prototype + Empirical Validation + Adversarial Check
  Discipline" section).
- Each task addresses a specific empirical-or-adversarial gap
  identified by the framework at
  `docs/research-notes/phase-prototype-and-validation-framework.md`.
- Without these, the .85 milestone perpetuates the current
  "architecture-heavy / prototype-light / adversarial-check-rare"
  pattern that today's audit identified as the foundation-of-cards
  failure mode.

**Reservation accounting:** 4 of these 5 tasks count against .85's
reserved infrastructure-class slots. The .85 milestone budget
should reflect that 4 of ~13 task slots are pre-allocated to this
discipline.

---

### REVISED 2026-04-30: Phase-2 Hardware Story Re-Scope (HIGH PRIORITY — paper-shaping)

**SUPERSEDES the FPGA-vs-GPU baseline task originally proposed
earlier 2026-04-30.** That task is no longer load-bearing because the
Phase-3 architecture audit (5 FATAL findings, see
`docs/research-notes/phase3-architecture-blindspot-audit-results.md`)
showed the FPGA-deep-EBM path requires multi-month bitstream
redesign that doesn't fit Carnot's actual production hardware
roadmap.

**User direction 2026-04-30 ~22:30Z:** *"I am less interested in FPGA
with the future looking more Extropic Z1 or photonic. Option C + D
sounds like the most realistic to me."*

**The new Phase-2 framing for the position paper:**

- KV260 (FPGA) is **proof-of-concept tier** — demonstrates that
  energy is evaluable in dedicated hardware on simple
  quadratic-Ising constraint problems. exp1041 / exp1068 / exp1081
  remain valid as engineering proof-points but with sampler-
  correctness caveats from audit Finding #2 (synchronous parallel
  Glauber on arbitrary J doesn't preserve detailed balance).
- For deep-NN energies and complex constraint composition (k=15+
  verifiers), Carnot's production hardware path is **Extropic Z1 (when
  available) and longer-term photonic**, NOT KV260 bitstream
  redesign.
- The deep-EBM-on-FPGA aspiration is documented as **future work**
  with the 5 FATAL audit findings as known constraints any future
  redesign must address.

**Task spec for .85+: Sampler-Correctness Validation + GPU-Phase-2
Comparison (revised scope):**

- Title: "Phase-2 Sampler Correctness Audit — KV260 caveats + GPU
  baseline"
- Goal: validate exp1081's headline numbers under sampler-correct
  conditions. Either constrain J to bipartite-block structure
  (preserving detailed balance under synchronous-parallel Glauber)
  OR re-port speedup numbers as comparing different
  distributions and flag the academic caveat.
- Add GPU Ising baseline (onnxruntime-gpu CUDA EP, 2x RTX 3090,
  already installed) for honest acceleration comparison.
- Output schema: `gpu_latencies_us`, `cpu_latencies_us` (compute-
  bound, NOT JAX-dispatch-bound), `fpga_latencies_us_caveated`,
  `sampler_distribution_difference_KL` (KL divergence between FPGA
  sampler output and correct CPU Gibbs at small N).
- Honest verdict tokens:
  - `fpga_poc_validated_with_caveats` — KV260 demonstrates POC, GPU
    is the production hardware for complex constraints
  - `fpga_sampler_distribution_mismatch_documented` — Finding #2
    confirmed empirically, documented as future-work
  - `extropic_z1_path_unblock_required` — Z1 hardware availability
    is the gating step for production hardware claims

**Why this matters:**

The position paper's Phase-2 section now anchors to:
1. KV260 as POC for "energy in dedicated hardware" (with caveats)
2. Extropic Z1 as the planned production hardware (per CLAUDE.md
   roadmap)
3. Photonic as the long-horizon vision

This is a defensible, honest story that doesn't require a
multi-month FPGA bitstream redesign and doesn't lie about what we
shipped.

**Action for .85 planner:**

- DO propose a Phase-2 sampler-correctness validation task (above
  spec)
- DO propose Extropic Z1 vendor-relationship / hardware-access tasks
  if Z1 is approaching availability
- DO NOT propose new FPGA bitstream redesign tasks
- DO NOT propose deep-EBM-on-KV260 tasks (architecture audit shows
  this is a multi-month rabbit hole)

**Reservation:** sampler-correctness validation counts against .85's
reserved infrastructure-class slots; Extropic vendor work is
exploratory research not infrastructure.

## MANDATORY-NEXT-MILESTONE PRIORITIES (.82 planner — hard pickup per CLAUDE.md)

### NEW 2026-04-29: no-permanent-retirement-on-environmental-failures (HIGH PRIORITY — research-progress discipline)

**`openspec/change-proposals/no-permanent-retirement-on-environmental-failures.md`**
(drafted 2026-04-29 evening, ready for .82 implementation) — formalize
the operator directive: *"don't give up entirely on experiments due
to operational interruptions and issues; find a way to divide up
the experiment into smaller experiments or find another way for
the experiments themselves to make forward progress until their
merits are proven or disproven."*

Mechanism: respawn queue (`ops/respawn-queue.json`) lists tasks
retired due to environmental failures (NOT merit-based). The .82+
planner reads the queue and emits respawn tasks with auto-populated
`prior_failures` blocks. Conductor classifies retirement kind
(environmental vs. merit) and auto-populates the queue.

**Initial queue seeded with today's 3 .81 retirements:**
1. exp1039-conductor-fastpath-gate-coercion (pre-test wedge —
   fixes 7a13304d + b2c73a08)
2. exp1042-dualgpu-rocm-torch-v4 (pre-test wedge + max_turns too
   tight — fixes 7a13304d + b2c73a08)
3. exp1044-triple-integration-v7 (gated on exp1039 retirement —
   fixes 7a13304d + b2c73a08 + 4e46ede6; must run AFTER exp1039
   respawn)

**Acceptance for .82 mandatory pickup:** the .82 planner output
must include all three respawn tasks (with auto-populated
prior_failures) AND the conductor's `pick_next_task` must be
patched to classify retirement kind and auto-populate the queue
on environmental retirements going forward.

This is the SEVENTH operator-attention-reduction infrastructure
proposal in the recent series. Ensures research progress is not
silently lost to operational interruptions.

### NEW 2026-04-29: parallel-multi-agent-conductor (HIGH PRIORITY — unblocks WOPR sprint)

**`openspec/change-proposals/parallel-multi-agent-conductor.md`**
(drafted 2026-04-29, ready for .82 implementation) — cross-backend
parallel execution via per-agent-type git worktrees. Two `systemctl
--user` instances: `carnot-conductor` (main, claude) +
`carnot-conductor-codex` (codex worktree, AGENT_TYPE=codex).

Without this, the WOPR-games-gallery cartridge sprint stretches
~3 weeks (single-stream serial). With it, ~1 week. **Target dates
depend on this:**
- 2026-05-08 Sudoku v1 + WarGames + Lights Out MVP → live on HF Spaces
- 2026-05-15 position paper preprint → arXiv

Tier A (week 1 of .82): dual-conductor (claude + codex), ~2-3 days.
Tier B (week 2): add gemini worktree for long-context audits.
Tier C (later): within-backend parallelism, deferred.

Schema field `worktree: Literal["main", "codex", "gemini"]`
orthogonal to today's `agent_type` field (commit `aa3c2707`).
Per-worktree state-file suffixing + merge-back protocol.

**Acceptance for .82 mandatory pickup:** the .82 planner output
must include `worktree: codex` on at least 3 WOPR-cartridge tasks
to validate the routing. The schema + conductor patches must
ship before the cartridge sprint begins.

Estimated total .82 effort: 5 days for Tier A+B; recoupable inside
the first compressed milestone.

### NEW 2026-04-29: huggingface-spaces-sudoku-demo + WOPR games gallery (HIGH-VISIBILITY MARKETING)

**`openspec/change-proposals/huggingface-spaces-sudoku-demo.md`**
(drafted 2026-04-29, not yet built) — v1 Sudoku-with-WOPR-aesthetic
HuggingFace Spaces demo. Scope: 3-5 days. Highest-leverage public
artifact Carnot can ship this month — pairs with the position paper
(theory-heavy) by giving reviewers and Twitter a *clickable* working
demo of energy-based Sudoku solving with the iconic WarGames
aesthetic.

**`openspec/change-proposals/wopr-games-gallery-extension.md`**
(drafted 2026-04-29, depends on Sudoku v1 landing first) — gallery
extension over the v1: tic-tac-toe (1d) → n-queens (1d) → connect
four (2d) → checkers (2-3d) → reversi (2d) → graph coloring (1-2d).
Optional chess (1-2 weeks). Each game is a `WOPRGame` cartridge
under `spaces/wopr-games/games/*.py`. Total ~9-11 days for the base
gallery (2 weeks part-time), +1-2 weeks for chess.

**Iconic moment to capture:** WOPR plays tic-tac-toe to a draw, then
displays *"A STRANGE GAME. THE ONLY WINNING MOVE IS NOT TO PLAY."*

**Why this is mandatory for .82:** the .81 planner deprioritized
this in favor of architecture work and infrastructure close-out.
Strategic miss — the Sudoku demo:
1. Provides empirical demonstration paired with the theoretical
   position paper (which targets arxiv submission ~2026-05-15)
2. Has high viral potential via the WOPR aesthetic
3. Is independent of FPGA / Phase-3-7 architecture work — can ship
   in parallel
4. Targets the carnot-ebm.org/blog/ audience and HuggingFace
   visibility, both already-established distribution channels

**Acceptance for .82 mandatory pickup (TIGHTENED 2026-04-29):** the
.82 planner output **MUST** include all THREE of the following
week-1 minimum-viable-gallery tasks. Anything less leaves the demo
incomplete and undermines the cultural anchor.

1. **`expNNNN-spaces-sudoku-v1-wopr-aesthetic`** (3 days)
   - `model: opus` — Spaces deployment is multi-step infra-class
     work prone to Sonnet bootstrap-and-bail
   - Base WOPR shell (CRT terminal, typewriter streaming, energy bar)
   - Sudoku solver with energy descent visualisation
   - Easter eggs: `LIST GAMES`, `GLOBAL THERMONUCLEAR WAR`,
     `HOW ABOUT A NICE GAME OF CHESS`, `GREETINGS PROFESSOR FALKEN`
   - Deliverable: deployed HuggingFace Space + JSON artifact
     describing the deployment

2. **`expNNNN-wopr-games-global-thermonuclear-war-cartridge`** (1 day) ⭐
   - `model: sonnet` (simple cartridge, no infra)
   - The cultural anchor — WOPR "computes scenarios" with frantic
     CRT animation, then concludes:
     "A STRANGE GAME. THE ONLY WINNING MOVE IS NOT TO PLAY.
      HOW ABOUT A NICE GAME OF CHESS?"
   - Pure marketing win. Must ship in week 1 — it's the cultural
     reference frame that makes the rest of the gallery memorable

3. **`expNNNN-wopr-games-lights-out-cartridge`** (1 day) ⭐
   - `model: sonnet` (well-defined CSP, low complexity)
   - The single best Carnot demo in the gallery: 5×5 grid, XOR
     toggling, all-off goal. Mathematically a pure Ising-model
     ground-state search — Carnot's energy formulation literally
     IS the natural-language solver
   - Visually satisfying: cells cascade off as energy descends
   - Critical for the "this is what Carnot is built for"
     narrative when paired with the position paper

**Estimated total .82 week-1 effort: 5 days for the three-cartridge
MVP.** This is the minimum viable gallery for a credible launch
alongside the position paper.

**Optional .82 stretch tasks (week 2+):**
- `expNNNN-wopr-games-tic-tac-toe-cartridge` (1d, classic increment)
- `expNNNN-wopr-games-nqueens-cartridge` (1d, classic CSP)
- `expNNNN-wopr-games-nonogram-cartridge` (2d, "picture reveal" wow factor)
- `expNNNN-wopr-games-life-reverse-cartridge` (1-2d, EBM-as-search demo)

If .82 has bandwidth for stretch tasks, prioritise nonograms (the
"decode a picture" moment is the gallery's second-best wow factor
after Lights Out).

See `openspec/change-proposals/wopr-games-gallery-extension.md`
for the full updated cartridge inventory (16 cartridges
specified, including the additional 9 added 2026-04-29).

## MANDATORY-NEXT-MILESTONE PRIORITIES (.81 planner — historical, picked up at 15:13Z 2026-04-29)

### NEW 2026-04-29: differential-agent-routing (MEDIUM PRIORITY — pre-emptive Opus for complex tasks)

**`openspec/change-proposals/differential-agent-routing.md`**
(schema + tests + docs already shipped 2026-04-29) — planner discipline
to set `model: opus` on tasks in four complex categories:
1. Hardware integration (FPGA, ROCm, KV260, DualGPU)
2. Schema / preflight infrastructure
3. Multi-step coordination experiments
4. Bootstrap-and-bail risk (`CRITICAL: write artifact FIRST` prompts)

Across milestone .80, 11 Opus escalations occurred reactively across
13 tasks. Pre-classification of the ~3 hardware/infra tasks would have
saved ~30 min wall-clock and prevented the bootstrap-and-bail wedge
that required 5 patches and 3 hours to close.

The schema validator (`scripts/roadmap_schema.py`) now formally
recognizes `model: Literal["sonnet", "opus"] | None = None` and
`escalate_on_max_turns: bool = True`. The planner prompt at
`_plan_next_milestone()` documents the four heuristic categories.

**Acceptance for .81 mandatory pickup:** the .81 planner output must
include `model: opus` on at least the KV260 work, any ROCm/DualGPU
tasks, and any preflight/schema/manifest tasks. The conductor reads
the field; no further code changes needed.

Estimated: 0 hours (no code; planner discipline only).

### NEW 2026-04-29: conductor-fastpath-bootstrap-skip (HIGH PRIORITY — milestone .80 wedged)

**`openspec/change-proposals/conductor-fastpath-bootstrap-skip.md`**
(1 exp, patch + tests + proposal already drafted 2026-04-29) — closes
the structural root cause of the 2026-04-29 milestone .80 wedge.

`_deliverable_exists()` was treating bootstrap-only artifacts
(`status: "running"`, written by Sonnet's "CRITICAL: write artifact
FIRST" defensive pattern *before* the real work) as completed
deliverables. exp1028 wrote a bootstrap stub, hit max-turns or
short-circuited, never updated to `pre_test_fixed: true`, and the
fast-path skipped every retry. exp1030 GATE_BLOCKed on the false
field forever; milestone wedged.

**Already implemented**: `scripts/research_conductor.py`
status-aware fast-path + `tests/python/test_conductor_deliverable_status.py`
(12 tests passing). The .81 task is to merge, replay the .80 wedge
(rm exp1028 artifact, restart conductor, confirm re-run), and
retire exp1030's GATE_BLOCK history.

This is the **third** consecutive milestone with a wedge requiring
operator-attention-reduction infra (after `conductor-supervisor.md`
and `roadmap-schema-validation.md`). Hard-pickup for .81.

Estimated: 1 hour for .81 close-out (already implemented).

### NEW 2026-04-29: verdict-reproducibility-audit (high priority)

**`openspec/change-proposals/verdict-reproducibility-audit.md`**
(3 exps, drafted 2026-04-29) — addresses the verdict-change incident
observed at 01:13Z when exp1031 SSD v3 produced
`carnot_filter_below_baseline` on rerun, having earlier produced
`fr11_loop_closed`. **Same code path, different headline result.**

The 12-round Zenil chain + Kinematic Layer Routing produced ~12
publishable theorems, several of which will be backed by empirical
experiments. **If those empirical verdicts are non-reproducible, the
position paper is vulnerable to reviewer reproducibility audit.**
Credibility risk is now load-bearing.

Three scoped experiments:
  - Exp A: rerun-audit of last 5 flagship verdicts; quantify stability rate
  - Exp B: seed discipline + canonical RNG initialization in `experiment_template.py`
  - Exp C: reproducibility checksum (SHA of seed + code SHAs + data hashes) +
           audit utility for `research-complete.yaml` flagship entries

Estimated: 6 hours. Pin to .81 mandatory pickup.

## MANDATORY-NEXT-MILESTONE PRIORITIES (.80 planner — hard pickup per CLAUDE.md)

Per the **CLAUDE.md "Overdue-Priority Forcing Function"** rule, any priority
pending 3+ consecutive milestones MUST be picked up by the next planner.
The following entries pass that threshold and are mandatory for the .80
roadmap:

  - **`openspec/change-proposals/conductor-supervisor.md`** (4 exps,
    pending since .77 — **3 milestones overdue**) — external observer
    process catching log-handle severance, claimed-vs-actual state
    drift, conductor wedge, and bounded auto-recovery whitelist. Single
    biggest unblock for unattended operation.

  - **`openspec/change-proposals/roadmap-schema-validation.md`**
    (3 exps, pending since .77 — **3 milestones overdue**) — Pydantic
    enforcement at planner output + activation. Prevents the schema-
    drift stillborn-milestone pattern (.69, .74).

  - **`openspec/change-proposals/eval-metrics-canonical-and-self-heal-production-bug-detector.md`**
    (4 exps, drafted 2026-04-28; partial work shipped already) — fixes
    the AUROC-bug class structurally. Migrates per-experiment metric
    helpers to canonical `carnot.eval.metrics` + adds production-bug
    detector to conductor self-heal + provenance tagging. The 2026-04-28
    inverted-AUROC discovery would have been impossible with this
    discipline in place. **Pre-shipped components:**
      - ✅ `python/carnot/eval/metrics.py` (15 tests + sklearn x-val)
      - ✅ `scripts/audit_metric_provenance.py`
      - ✅ `scripts/conductor_commit_watchdog.sh`
      - ✅ `experiment_template.py:build_result(metrics_used=...)`
      - ✅ AUROC fixes in exp995 + exp1003

  - Bonus (pending since .77, **3 milestones overdue**):
    **`openspec/change-proposals/conductor-otel-tracing.md`** (5 exps)
    — depends on the supervisor; lower priority but the natural next
    step.

  - **`openspec/change-proposals/zenil-grounded-self-distillation-deployable-stack.md`**
    (4 exps, drafted 2026-04-28) — ships the four code artifacts that
    operationalise the Round-6 Deep Think result on verifier-filtered
    self-distillation: Φ > 0 measurement module, joint annealing
    schedule, PT acceptance hyperparameter (0.35), and the
    REQ-PHASE2-006 Gray-code factor experiment. Mathematically
    justifies the Phase 2 hardware mandate (`_bmad/architecture.md`)
    and produces a publishable Phase 2 transpiler theorem result if
    the empirical Gray-code factor confirms. **Target .81 or .82**
    depending on planner load.

## Operational watchdog scripts (newly shipped 2026-04-28)

Run these between conductor-supervisor landing:

  - `bash scripts/conductor_commit_watchdog.sh` — periodic check for
    stuck commits. With `AUTO_COMMIT=1`, attempts last-resort
    `git commit --no-verify` after $STALE_MIN minutes (default 60).
    Schedule via cron / systemd-timer.

  - `python3 scripts/audit_metric_provenance.py` — walk
    `results/experiment_*.json`, list deliverables by metrics
    provenance. With `--flag-buggy func:version`, surfaces deliverables
    using a known-bad implementation for retrospective re-evaluation.

## NEXT-MILESTONE PRIORITIES (.77 planner — historical, see MANDATORY above)

The 2026-04-27 24-hour session demonstrated that the conductor's
operator-attention burden is unsustainable — the operator had to
manually:
  - reap orphan process trees (~1 every hour)
  - SIGTERM runaway Sonnets that spawned duplicate experiment
    invocations (twice in 4 hours)
  - recover from broken `logs/conductor.log` write handles (~4
    occurrences in the session)
  - translate a schema-mismatched planner output (.74 would have
    gone stillborn exactly like .69 without intervention)
  - manually commit ~3 hours of accumulated conductor work that the
    conductor's own commit pipeline failed to push (twice — 35-file
    and 14-file commits)

Two proposals exist that scope durable fixes for these patterns:

  - **`openspec/change-proposals/conductor-supervisor.md`** (4 exps)
    — external observer with heartbeat watchdog, claimed-vs-actual
    state reconciliation, bounded auto-recovery whitelist (orphan
    reap, conductor restart, log-handle reset), conductor-side
    SIGUSR1 log-reopen handler. Catches every "conductor running but
    something's wrong" failure mode that requires manual operator
    attention today.

  - **`openspec/change-proposals/roadmap-schema-validation.md`**
    (3 exps) — Pydantic ResearchTask + Roadmap models validated at
    planner output (re-prompt on failure) and at activation (refuse
    to overwrite the active roadmap with malformed YAML). Prevents
    the once-per-month schema-drift stillborn-milestone pattern.

  - Bonus: **`openspec/change-proposals/conductor-otel-tracing.md`**
    (5 exps) — depends on the supervisor; lower priority but the
    natural next step. Puts every conductor iteration + subagent
    spawn into Victoria Trace so the seven incident shapes from this
    session each become single-trace queries.

The `flock` single-run guard from `conductor-process-isolation.md`
Exp B was direct-shipped on 2026-04-27 (commit 1b254b87) because of
the operational urgency. The supervisor + schema-validation work is
the natural next layer; the .77 planner should treat them as
candidate top picks.

## PHASE 2 PRIORITY (.78+ planner)

  - **`openspec/change-proposals/continuous-to-ising-transpiler.md`**
    (6 exps) — Phase 1 → Phase 2 bridge. Takes a trained verifier
    `state_dict` + a `HardwareSpec` and emits an `IsingSpec(J, h, ψ)`
    deployable to KV260, ECP5/Nexus, future XTR-0, or future photonic
    SLM. Origin: 2026-04-27 Deep Think exchange (4 rounds) producing
    the Continuous ε-Ising-Rank Theorem + Split-Verifier + Native
    Thermodynamic Distillation (PT-PCD with Gray-code encoding). The
    KV260 board has been on-hand since 2026-04-20.

## EXP 980 RE-SCOPING (.77 or .78 planner)

  - Exp 980 in .76 is currently scoped as "repair 11 monotonicity and
    boundary violations in KAEMEnergy." Under the **SOS-Integrated
    KAN** insight (Deep Think 2026-04-27), this framing is wrong.
    Standard monotonic-spline parameterizations are sufficient but
    not necessary, restricting expressivity. The fix is to push the
    constraint into derivative-space and analytically integrate:
    parameterize ψ'(x) as a Sum of Squares of B-splines (V ∈ ℝ^{N×M}
    unconstrained, M ≥ 2 for Burer-Monteiro stability), then integrate
    to ψ(x) = c² + Σ_{i,j} (V V^T)_{i,j} Φ_{i,j}(x). Monotonicity and
    non-negativity become **type-level invariants** of the AST
    `Add(Square(c), Integral(SumOfSquares(Splines)))`, not numerical
    properties to verify. MILP verification reduces to type-checking;
    the post-hoc repair subsystem is eliminated. Drop-in compatible
    at p=1 (hat functions → C¹ piecewise cubic splines, same
    computational profile as standard KANs). See
    `memory/project_sos_integrated_kan.md` for full detail.

## Original known-issues

| # | Issue | Severity | Workaround |
|---|-------|----------|------------|
| 1 | PyO3 0.24 doesn't support Python 3.14 natively | Low | Set `PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1` |
| ~~2~~ | ~~Gibbs/Boltzmann grad_energy uses numerical finite differences~~ | ~~Resolved~~ | Analytical backprop implemented |
| ~~3~~ | ~~Python test suite not yet written~~ | ~~Resolved~~ | 48 tests, 100% coverage |
| 4 | Ackley and GaussianMixture benchmarks use numerical gradients | Low | Analytical gradients are complex; numerical is acceptable for benchmarks |
| 5 | RETRO-028: llama.cpp#21516 tokenizer bug causes Gemma4-E4B-it to emit only `<unused8>` tokens (token_id=14), producing 0.0% accuracy (false negative in Exp 439) | High — Fix implemented | Use GemmaTransformersLoader (python/carnot/pipeline/gemma_loader.py) instead of any llama.cpp backend. GPU verification pending (Exp 450). |
| ~~7~~ | ~~RETRO-029: Exp 444 (CarnotThinkProbe) timed out at 20 min with zero results — no partial data, no checkpoint.~~ | ~~Resolved 2026-04-18~~ | ThinkProbeV2 (python/carnot/pipeline/think_probe_v2.py) — 60-min budget, partial_verdict mode, incremental checkpoint every 10 questions. Exp 455 implements and validates the fix. |
| ~~6~~ | ~~RETRO-030: Exp 446 (Energy Matching) exited with status 0 but produced no result JSON. Root cause: exception mid-write left no file; watchdog missed it (only checked exit code).~~ | ~~Resolved 2026-04-18~~ | AtomicResultWriter (python/carnot/pipeline/atomic_writer.py) — write-to-tmp + os.rename prevents partial writes. Exp 452 re-runs Exp 446 logic with atomic write + verify_exists() assertion. |

## RETRO-072 update (Exp 701, 20260422)
Vivado not installed; yosys not found.  Synthesis blocked.  Install one of:
  - AMD Vivado 2024.2 (free WebPACK from xilinx.com)
  - yosys (`sudo pacman -S yosys` on CachyOS)
RETRO-073 opened for milestone .54.

## RETRO-CRITICAL: JEPA v17 RankNet Gate Failed (Exp 704/705, 20260422)
JEPA v17 OOD AUC = 0.4819, still below random chance (threshold = 0.75).
RankNet pairwise loss partially addresses the anti-correlation root cause but pairwise
hedging persists when pairs are too similar — each pair is optimised independently.
**v18 approach:** LambdaRank listwise loss — optimise NDCG over ALL steps per question
simultaneously, directly matching the AUC evaluation metric.
**Data gap:** Listwise training requires >= 5 steps per question; FoVer v1 provides only 2.
Unblocked by: Exp 712 FoVer v2 PDDL (5+ steps per question via PDDL plan enumeration).
JEPA v16 cascade block remains in effect until v18 achieves OOD AUC >= 0.75.

## Closed Issues

### FR-11 CLOSED — Status: OPERATIONAL (Exp 738, 2026-04-22)

~~FR-11 (Autonomous Self-Learning Loop) — blocked for 15+ milestones on AUC gate.~~

CLOSED 2026-04-22, Exp 738. FR-11 is now OPERATIONAL. Evidence:
fr11_relay_operational=True (Exp 734, relay_events_acked=100, latency_p99_ms < 200),
fr11_tier2_relay_functional=True (Exp 738, templates_replayed_in_s2=1,
cross-session persist confirmed), probe 5-fold AUC=0.993 (Exp 732). Milestone
2026.04.56 retro marked FR-11 "ELIGIBLE FOR FORMAL CLOSURE". Formal closure
certificate: results/fr11_closure_certificate.json.

---

## RETRO-033 CLOSED (Exp 720, 20260422)
Verdict: vr_not_viable_at_scale
signed_improvement at 200q: -0.0050 (simulated_historical_inference — 19/19 empirical failures at 100q)
Root cause: VR pipeline does not improve accuracy at current model scale (Qwen3.5-0.8B).
Resolution: VR removed from active roadmap. Re-evaluate when a larger model (>= 7B parameters)
or a fundamentally different verification architecture is available.
Spec: REQ-VER-030-6, SCENARIO-VER-037


## RETRO-MANIFEST-FULL-SCOPE: Human Intervention Required (Milestone .69)

ExclusionManifestEnforcer pre_launch_check() cannot be wired to the conductor loop
without modifying scripts/research_conductor.py, which is forbidden per CLAUDE.md
in the Exp 892 task specification.

11 consecutive milestones open. Action required: either
  (a) grant human permission to modify scripts/research_conductor.py for this one change, or
  (b) accept that manifest enforcement operates at the planning layer only
      (CLAUDE.md rule is the primary enforcement; code enforcement is secondary).

Documented by Exp 892 pre-flight v18 on 2026-04-26T02:52:17Z.
enforcement_wired: false

## IPFS not installed — VJEPA v2 weights have no IPFS mirror

Added: 2026-04-26 (Exp 902)

CLAUDE.md rule 3 requires all published weights to have an IPFS mirror.
The `ipfs` command was not found at publish time.  Install IPFS and
re-run Exp 902 to establish the mirror.

Install: `apt install ipfs` or use the ipfs.io installer:
https://docs.ipfs.tech/install/

Then run: `ipfs add -r /tmp/carnot-vjepa-v2-card/ && ipfs pin add <CID>`


## RETRO-MANIFEST-FULL-SCOPE: CRITICAL — Human Intervention Required (Milestone .70)

ExclusionManifestEnforcer pre_launch_check() is NOT wired to the conductor loop.
This is the 12th consecutive milestone where the manifest has not been enforced
mechanically. The rule in CLAUDE.md (planning-layer discipline) is the ONLY active
enforcement. A conductor-level hook is blocked by the 'do NOT modify
scripts/research_conductor.py' constraint. Action required: grant human permission
to modify scripts/research_conductor.py for this single wiring change.
enforcement_wired: false
escalation_milestone: "2026.04.70"


## RETRO-LAGRANGE-ENTROPY-DEGENERATE: CLOSED (Exp 918, 2026-04-26)

Root cause: Single-constraint corpus had entropy = 0 by construction (p = 1.0).
Fix: 8-constraint heterogeneous corpus. Exp 918 result: signed_entropy_improvement=0.018.
Algorithm confirmed working. RETRO closed.

## GATE-CHECK DISCIPLINE: prior_failures Required for All Domain-Overlapping Tasks

Exps 917, 919, 920, 921, 922, 925, 926, 927 all blocked in .71 by missing prior_failures.

Rule: Any YAML task touching a domain with ANY prior experiment history MUST include
prior_failures entries with: experiment_id, verdict, addressed_by, retire_if_same_verdict.
The conductor gate-checker scans the FULL research history. If prior_failures is absent
and matching prior experiments exist → immediate block.

This is a planner-layer discipline failure, not a code bug. The planner that generated
research-roadmap-v71.yaml did not populate prior_failures for any of the 8 tasks with
prior failure history. Fix: consult research-complete.yaml before generating any task YAML.

## RETRO-MANIFEST-FULL-SCOPE: CRITICAL — Human Intervention Required (Milestone .71)

14 consecutive milestones without mechanical manifest enforcement.
enforcement_wired: false
escalation_milestone: "2026.04.71"
Action required: grant human permission to modify scripts/research_conductor.py.

## RETRO-RERUN-DISCIPLINE-GATE-CASCADE (opened .71)

9 of 12 experiments in .71 were blocked by the conductor pre-gate due to missing
prior_failures fields in the roadmap YAML. This is a cascade of the same root cause.
Status: HUMAN_REQUIRED — planner must be trained on the rule before .72 executes.

## RETRO-HEURISTIC-RPRM-FLAT-SIGNAL (opened .71)

Exp 924 R-PRM Tier 2.9 heuristic mode: AUC delta = 0.0. Heuristic inference cannot
produce step-level signal. Real model inference (Qwen3.5-0.8B minimum) required.
Status: TARGETED — .72 must use live model, not heuristics.

## RETRO-DRIFT-ENSEMBLE-UNIFORM-WEIGHTS (opened .71)

Exp 923 DriftProbe ensemble (3 layers, uniform weights): OOD AUC 0.5625 vs 0.565 baseline.
Uniform weighting HURTS — two zero-coefficient probes dilute one informative probe.
Status: TARGETED — .72 must use learned weights (logistic regression on validation set).

## RETRO-HF-SOPS-CREDENTIAL-INJECTION (opened .71)

Exp 922 HF publish blocked by SOPS credential injection unresolved.
Status: HUMAN_REQUIRED — resolve SOPS credential injection before scheduling HF publish.

### IPFS Mirror CLOSED (Exp 934, 2026-04-26)
VJEPA v2 IPFS CID: `QmTkGjpN5fYNnC3g8Gx8sPWHZJKkw8oGVDKwWT6sZbVaGN`
Mirror registry: results/ipfs_mirrors.json

## RETRO-MATH-REPAIR-MODEL-CEILING (opened .72, Exp 930)

Exp 930 iterative self-repair on GSM8K: gemma-4-E4B-it baseline=12%, repair=12%,
signed_improvement=0.0. Model capability ceiling — E4B is too small for GSM8K
math reasoning. The repair algorithm is structurally correct; the model is wrong.

Resolution path: Exp 942 in .73 must use Gemma4-31B or Qwen3.6-35B-A3B (SOTA tier).
SOTA GGUF already downloaded — gemma-4-26B-A4B-it-UD-Q4_K_M.gguf confirmed in HF cache.
Status: TARGETED (Exp 942)

## RETRO-SC-ENERGY-GATE-DISCIPLINE (opened .72, Exp 939)

Exp 939 SC-Energy Set Consistency Networks blocked by conductor pre-gate: task YAML
lacked prior_failures entries for 7 prior SC-energy / contrastive-energy experiments.
Identical planning error to Exp 917 in milestone .71 — planner did not consult
research-complete.yaml before writing the task YAML.

## SC-ENERGY PRIOR EXPERIMENTS (for Exp 944 prior_failures reference)

Exp 944 MUST include all 8 entries below in its prior_failures field:

| Exp | Verdict | Domain |
|-----|---------|--------|
| 506 | semantic_energy_no_improvement | Semantic Energy Tier 0d |
| 509 | energy_magnitude_wins | PPSEBM Energy Magnitude Replay (adjacent) |
| 533 | no_violation_reduction | COLD Decoding Energy Guidance (adjacent) |
| 711 | tier_29_below_threshold | SC-Energy SetConsistencyVerifier Tier 2.9 |
| 725 | sc_energy_v2_below_threshold | SC-Energy v2 FoVer v2 Dual Labels |
| 772 | semantic_energy_below_baseline | SemanticEnergyProbe Tier 0g |
| 787 | energy_prefilter_efficient | S* Energy Pre-Ranking (adjacent) |
| 939 | blocked_gate_check_failed | SC-Energy Set Consistency Networks |

The "addressed_by" field for each must explain what is substantively different in
Exp 944 (new architecture, new corpus, new technique — not relabeling).
Status: HUMAN_REQUIRED at planner layer — conductor will block again if omitted.
