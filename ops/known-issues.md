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

## MANDATORY-NEXT-MILESTONE PRIORITIES (.86 planner — hard pickup per CLAUDE.md)

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
