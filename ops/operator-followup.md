# Operator Follow-Up — Things That Would Help the Project Going Forward

This document captures action items that require **operator judgment,
external action, or non-Carnot-internal decisions**. None of these can
be done autonomously by the conductor; all benefit from operator
attention when convenient.

Maintained by the outer-loop watcher. Items added with date stamps so
freshness is auditable. Items move to the `## Completed` section when
done, with completion date and outcome — do not delete (preserves the
historical record per CLAUDE.md "Never remove existing content from
ops/spec docs").

---

## Open — External Relationships

### 2026-05-20: UW SyFI lab outreach (VibeServe collaboration potential)

**Source:** `syfi.cs.washington.edu/blog/2026-05-12-introducing-vibeserve/`

**The fit.** VibeServe (UW SyFI, May 2026) and Carnot reached the same
agentic-research-loop architecture independently:

| Architecture element | VibeServe | Carnot |
|---|---|---|
| Outer loop | search policy + issue backlog + memory file + git checkpoint history | operator + research-program.md + research-references.md + memory + git history |
| Inner loop | 3 specialized agents in fresh contexts (Implementer / Accuracy Judge / Performance Evaluator) | planner / experiment agent / retro / adversarial-verify |
| Skills/Verifier library | extensible techniques added as new skills | Tier 0a-0u verifiers added as new modules |
| Substrate | Claude / Codex CLI | Claude / Codex CLI / Gemini CLI |
| Problem domain | LLM serving | LLM output verification |

**Why outreach matters strategically.** Carnot's paper-v6 currently has
no academic collaborator. UW SyFI is an active CS systems lab with a
verifier-friendly publication track record (FlashAttention authors are
adjacent). Mutual citation + potential co-authorship on a "agentic
research-loop generalization" position paper would expand both papers'
reach.

**Per CLAUDE.md "Operator-Only External Publication" rule (commit
`49cc90b7e` 2026-05-19): outbound email to external researchers is
operator-only.** The conductor cannot send the message.

**Suggested message draft** (for your editing, not for the conductor
to send):

```
Subject: Carnot — independent agentic-loop architecture for LLM-output verification

Hi VibeServe team,

Read your May 12 post on agentic synthesis of bespoke LLM-serving
runtimes. The architecture (outer search loop + inner specialized
agents in fresh contexts + extensible skills library + persistent
git-checkpoint memory) is structurally identical to what we've built
independently for Carnot — an agentic research system for energy-
based LLM-output verification.

Concrete points of resonance:
- Our 'verify_code', 'verify_with_properties', 'verify_and_repair'
  tools (carnot-ebm on PyPI) are structurally what your Case B
  "predicted outputs verified in K-token blocks" exposes — 5.95×
  vLLM speedup for you maps to a verifier-side serving API we should
  prototype.
- Your Case E jump-forward decoding for constrained JSON (2.6× on
  MacBook) translates directly to our CSL grammar / FST PATH C
  constraint-satisfaction work.
- Your Case C "6 accuracy-gate failures before iter 7 success" is
  the exact pattern our adversarial-verify discipline catches —
  we've caught fabricated experiment artifacts (exp1100), un-
  replicable headline AUROC (exp2473 0.9351 → 0.7964 under 5-seed
  replication), and methodology-fallback Phase 4 signals (exp2519
  retire_if_methodology_fallback) the same way.

Would love to talk about:
1. Cross-citation in our respective papers (Carnot paper-v6 in
   preparation; targeting arXiv submission once Phase 4 v3 lands)
2. Joint position-paper on "agentic loops for systems & research"
3. Whether VibeServe could synthesize a Carnot-optimized serving
   runtime (our verifier ensemble at AUROC 0.9750 verified, +5.14pp
   over HIVE peer 0.9236, currently CPU-only)

Carnot is at:
- https://github.com/Carnot-EBM/carnot-ebm
- PyPI: carnot-ebm
- HuggingFace: Carnot-EBM org

Happy to share the paper-v6 draft on request.

Best,
Ian Blenke
ian@blenke.com
```

**What to consider before sending**:
- Is paper-v6 in a state you're comfortable sharing? (Yes per `.243
  exp2536: `submission_package_ready=true`, LaTeX compiles, abstract
  205 words. arXiv submission still on hold per gate_3 / Phase 4 v3.)
- Do you want the conductor to refresh the `.245+ paper-v6-vibeserve-
  related-work cite (queued in `ops/known-issues.md`) BEFORE you
  reach out, so the citation is already in main.tex when they read
  the draft? (Probably yes.)

---

## Open — Hardware

### 2026-05-20: KV260 — bitstream stays in sync with RTL going forward

**Status as of 2026-05-20 13:00 EDT:** Active overlay is
`carnot_ising_v2_n64_image_1` (= carnot_ising_v4.bit.bin from Apr 27
build via v4_bd project with XDC constraints). UIO devices
`/dev/uio0..uio4` available. FPGA `operating`.

**Operator-side decision pending**: when the conductor lands the
`.245+ "KV260 Real-Board Bitstream Refresh" task (queued in
`ops/known-issues.md`), Vivado will produce a fresh bitstream from
current `hardware/kv260/carnot_ising_top.v`. The task will scp the
bitstream to `kria:/tmp` and run bootgen on-board. **If the new
bitstream has different AXI register addresses than v4**, any in-flight
KV260 experiments holding `/dev/uio0` references will break. Worth
operator-aware re-sync after the task lands.

**Future operator action**: if Carnot adds a larger-scale FPGA target
(KV260 → Alveo / Agilex) per Exp 1460 portfolio expansion, only you
can authorize the hardware purchase + bench-side setup.

### 2026-05-20: PolarFire — sustained-load thermal monitoring

**Status:** Carnot runs end-to-end on PolarFire RISC-V (exp2490 `.241).
Per `ops/hardware-bringup-prep.md` 2026-05-14 22:15Z thermal note:
the PolarFire SoC Discovery Kit is passively cooled. Sustained 100%
RISC-V utilization will thermal-throttle.

**Operator-side decision**: do you want active cooling (small USB
fan) on the PolarFire for long-running benchmark runs? Without it,
hardware experiments >5 min wall time will degrade. Cheap fix
($10-20 fan); your judgment whether the bench-side complexity is
worth the experiment fidelity.

### 2026-05-20: GateMate — terminal state met, optional follow-on tasks

GateMate has fully graduated from per-milestone mandatory inclusion
(both terminal-state halves met: bitstream flashed + on-board sampler
timing benchmark at 951 Hz sample rate, KL=1.92 from analytic
Boltzmann).

**Optional operator-side direction**: do you want to push KL toward
tighter analytic match (which would require RNG-source improvement or
larger n)? Or treat the n=16 result as sufficient for paper-v6's
sovereignty story and move on? My read: the latter — paper-v6 just
needs "Carnot runs on $60 open-toolchain hardware," not "Carnot
matches THRML in KL on $60 hardware." But your call.

---

## Open — Publication

### 2026-05-19: Paper-v6 arXiv submission — operator-only per CLAUDE.md

`feedback_publication_holds_until_phase4_pivot.md` (2026-05-02) +
CLAUDE.md "Operator-Only External Publication" rule (commit
`49cc90b7e` 2026-05-19) both apply. Current state:

- `arxiv_ready=False` per `.243 capstone (Phase 4 v3 blocked on real
  IsingVerifier step-energy implementation)
- exp2536 `.244 LaTeX compile success, 205-word abstract, package
  assembled
- AUROC 0.9750 adversarially-verified, +5.14pp over HIVE peer
- 3 FPGA boards: GateMate terminal, PolarFire terminal, KV260
  partial (bitstream loaded, board reachable, awaiting refreshed
  bitstream from `.245+ task)

**What you can do now**:
1. Review `docs/arxiv-paper/main.tex` for tone, claims, narrative
2. Decide whether to ship with Phase 4 v1 as "tested hypothesis,
   mixed empirical signal" framing, or wait for `.245+ Phase 4 v3
   real-IsingVerifier-step-energy task to land
3. If shipping: run `arxiv upload` / submit yourself per the rule

**What the conductor will keep doing**: prep tasks (LaTeX compile,
abstract checks, package assembly) — never the actual submit.

### 2026-05-20: Phase 4 hypothesis status — narrative decision

Cumulative empirical record (`.239-`.244):

| Test | Result | Methodology |
|---|---|---|
| exp2474 ODAR free-energy AUROC | 0.5584 partial | Real measurement |
| exp2480 Phase 4 Empirical Report | partially_validated | Aggregator |
| exp2486 ARM-EBM Bijection | r=0.108 refuted | Real |
| exp2487 Qwen PRC Divergence | refuted | Used mock_model — INVALID |
| exp2496 Phase 4 PRC v3 real GGUF | 3-fail-skipped | Infrastructure failure (gemini 429) |
| exp2497 Spilled Energy CPU | r=-0.02 refuted | Real |
| exp2508 Step-Level ARM-EBM v2 | r=-0.43, p<0.01 positive | Fallback proxy — METHODOLOGY CAVEAT |
| exp2519 ARM-EBM v3 NO FALLBACK | blocked_precondition | Discipline-correct refusal |

**Net read**: 4 refutations (3 valid + 1 mock-invalid) + 2 partials
+ 1 positive-via-fallback (caveated) + 1 honest block. Zero strong
validations under proper methodology.

**Three operator-narrative options**:
1. **Drop Phase 4 from paper-v6 main claims** — keep verifier-ensemble
   + AUROC + hardware sovereignty as the story. Phase 4 becomes a
   "tested-but-not-supported" footnote.
2. **Wait for `.245+ exp24XX-ising-verifier-real-step-energy** —
   implement the real step-level IsingVerifier (the missing piece),
   re-run Phase 4 v3 properly, accept whatever the result says.
3. **Reframe Phase 4 as "preliminary empirical exploration"** —
   honest negative-result framing. The hypothesis is tested under
   N operationalizations and not yet supported. Future work.

I'd lean (2) for intellectual honesty + (3) as fallback narrative.
(1) is the path-of-least-resistance ship-faster option.

---

## Completed

### 2026-05-20 ~13:00 EDT: KV260 reachable + Carnot bitstream activated

Operator brought KV260 online (`ssh kria` at 192.168.51.98). Watcher
verified state, activated `carnot_ising_v4` overlay via `xmutil
loadapp carnot_ising_v2_n64` (legacy name resolves to v4 via symlink).
Memory entry `feedback_kv260_latest_bitstream_must_be_xdc_constrained.md`
shipped. `ops/known-issues.md` queued the `.245+ XDC-refresh task.

Outcome: FPGA `operating`, `/dev/uio0..uio4` available, Carnot Ising
sampler accessible for future verifier-on-FPGA experiments.

### 2026-05-20 ~22:45 EDT: KV260 SD-card-flash workflow retired

Operator question: "why is an SD CARD flash required when you can ssh
to kv260.local and install anything needed?" Honest answer: it isn't.
exp2670/2710/2722 (and the originally-queued exp2735 for `.259) all
used a `ls /dev/mmcblk*` precondition on the HOST, which checks
whether the host machine has an SD card inserted — meaningless for the
board's state. Wrong-mechanism leftover from a pre-board-boot
PYNQ-flash workflow.

Outcome (this turn):
- exp2735 rewritten to use `ssh -o ConnectTimeout=5 kria 'true'` +
  on-board xmutil + uio0 register read (research-roadmap.yaml:1063+)
- CLAUDE.md Pre-Launch Preconditions table KV260 row updated to
  ssh-reachability
- `ops/exclusion_manifest.yaml` gained
  `kv260_host_sd_card_precondition_retired` entry blocking the scope
  pattern from future planners
- Memory `feedback_kv260_ssh_not_sd_card.md` shipped + MEMORY.md
  indexed
- Per-milestone KV260 continuity check now runs the actual on-board
  smoke (the real terminal-state progress signal) instead of escalating
  on a phantom host-SD-card-absent branch

### 2026-05-19: Phase 1 ship gate met + paper-v6 LaTeX package ready

- `carnot-ebm 0.1.0b1` on PyPI
- HuggingFace `Carnot-EBM/` mirror up
- MCP server docs (`docs/mcp-server.md`) shipped
- CLI usage docs (`docs/cli-usage.md`) shipped
- Independent reproducer present (CI workflow + ops/test-results.md)
- `.244 exp2536: `latex_compile_success=true`,
  `abstract_word_count=205 (<=250)`, `submission_package_ready=true`

Outcome: paper-v6 is technically arXiv-submittable. Operator review
+ submission pending.

---

## How this document is maintained

The outer-loop watcher (Claude in the active session) adds entries
when:
- The operator surfaces a relationship or external decision
- The conductor produces output that requires operator review or
  external action (board attach, account upgrade, etc.)
- A CLAUDE.md "Operator-Only" rule blocks an otherwise-autonomous task

Operator moves completed entries to the `## Completed` section with
outcome notes. Entries do NOT get deleted (audit trail).

The conductor's planner reads this document during the `.YYY planner
phase (alongside `ops/known-issues.md`) and uses it to:
- Avoid queuing tasks that need operator action without operator
  having seen the trigger
- Prioritize external-action-blocked items when the underlying
  blocker may be resolved (e.g., operator just plugged in the board)
