# Research Roadmap vNEXT: Milestone 2026.04.100

Planned: 2026-05-04
Status: Draft for conductor execution
Predecessor: 2026.04.99 Publication Closeout + SOTA Certificates + PRIME Self-Learning + Continuous Repair
Roadmap YAML: `research-roadmap-next.yaml`

## What Milestone .99 Proved

Milestone 2026.04.99 closed most of the work that had been stuck behind stale skeletons:

- Publication closeout is no longer blocked by critical text fixes. `exp1269` fixed all five critical paper issues, and `exp1270` compiled the arXiv v10 bundle. The remaining publication gap is external submission/receipt, not local compilation.
- PRIME-style verifier selection works as a measured reward-selection step. `exp1272` wrote a verifier weight vector with SemEnergyProbe as the strongest selected signal.
- Self-learning moved from theory to measured smoke/replay artifacts. `exp1273` reported `self_learning_delta_overall=0.83798`, but only as smoke-only because no SOTA GGUF model was available. `exp1274` reported certificate-memory replay improvement with `self_learning_delta_overall=0.357143`.
- Continuous repair is the strongest new positive. `exp1275` found an FSNet-style feasibility delta of `4.5832`, and `exp1276` found a SnareNet-style adaptive-repair delta of `0.21996` over FSNet.
- The EST gaming-defense measurement and both minimal WOPR cartridges shipped. `exp1278` reported vulnerability 0.00 with k=5 blocking true, `exp1279` shipped Kakuro, and `exp1280` shipped Masyu.
- The milestone retro scored 12/14 criteria met. The two misses are decisive: `exp1271` did not produce headline SOTA GGUF certificate extraction, and `exp1277` stayed gate-blocked because no certificate parse rate existed.

The next milestone therefore does not rerun the whole .99 plan. It makes the SOTA GGUF certificate path executable, converts smoke-only self-learning into headline-eligible evidence if the SOTA gate opens, and strengthens continuous repair with nonlinear/feasibility diagnostics.

## Current Research Signals Added Before Planning

The 2026-05-04 vNEXT literature/source scan added these entries to `research-references.md` before this plan was written:

- Codex planning-agent verification scan: DCCD draft-conditioned constrained decoding, NSVIF CSP-style instruction verification, Residual Drift / MUS-Repair, PCC certainty+consistency routing, and the current EBT / Extropic / Kona status check. These refine `.100` by making the certificate rerun draft-conditioned rather than grammar-only, making semantic routing solver/CSP-backed, adding residual-drift bookkeeping, and keeping the architecture local-first rather than pivoting to a closed vendor.
- ARS answer-agreement representation shaping (arXiv 2601.17467): use answer stability under trace-boundary perturbation as a hallucination/certificate trust diagnostic.
- TruncProof (OpenReview ICLR 2026): grammar-constrained generation with a hard maximum token budget, directly relevant to bounded certificate tails.
- SEM-CTRL (arXiv 2503.01804 / OpenReview TMLR 2026): Answer Set Grammars for semantic control, useful after syntactic certificate validity is reliable.
- DVI (arXiv 2510.05421 / OpenReview ICLR 2026): verifier accept/reject decisions as online self-speculation training signals.
- XGrammar-2 and llguidance: practical local structured-generation engines with llama.cpp/vLLM/SGLang integration, the most direct implementation path for local GGUF certificates.
- BEAVER + CRANE (arXiv 2512.05439 / 2502.09061): deterministic constraint-bound and reasoning-preserving constrained generation references; use them as design constraints, not as a full .100 implementation.
- Optimal KAN abstractions (arXiv 2602.06737): formal PWA/MILP verification path for KAN verifier components; queued as a future verifier-certification direction after SOTA certificates stabilize.
- Current EBT / ARM-EBM / Extropic / Kona status: no broad architecture pivot; the near-term task remains measurable local certificates, verifier learning, and continuous repair while keeping the sampler abstraction portable.
- Supplemental scan entries added during final planning: HoVer safe-prefix holistic verification; FLy/TriSpec loose/proxy speculative verification; and the April 2026 fully parallel probabilistic Ising machine with inertia. These are not new .100 critical-path tasks, but they shape the follow-up path: safe-prefix repair after certificate parsing succeeds, proxy verifier triage after Cactus is measurable, and sampler-parity hardware work in a later hardware-focused milestone.
- Final verification scan entries added before YAML emission: OnlineSpec / When Drafts Evolve, HalluHard, CaR Construct-and-Refine, and the Scientific Reports parallel-pbit performance/cost study. These refine existing `.100` tasks rather than adding more tasks: OnlineSpec adds an acceptance-delta metric to DVI replay, CaR adds a construct/refine iteration diagnostic to continuous repair, HalluHard becomes a `.101` multi-turn benchmark candidate, and the p-bit study is filed for the next hardware-focused sampler-parity pass.
- Fresh planning scan addendum: Structure Snowballing, InterWhen, ConstraintBench, and EBM-CoT. These refine the `.100` task definitions: grammar backends must report a snowballing-risk diagnostic, verifier learning must include an interleaved-during-reasoning comparison, semantic routing gets a tiny ConstraintBench-style micro-eval, and the energy bridge audit includes CoT-sequence energy rather than final-answer energy only.
- Active planning verification scan: ARC-Decode risk-bounded acceptance, SPEED-Bench replay/evaluation tagging, HiSpec / Speculative Speculative Decoding, and a Semantic Scholar citation check on EBT. These refine `.100` without changing the task count: Cactus v2 must report risk-bound proxy and evaluation-mode fields, HiSpec/SSD remain `.101` proxy-verifier acceleration candidates, and the EBT citation check reinforces the need for the energy bridge audit.

## Three Biggest Gaps

1. **Headline SOTA local extraction gap.** .99 failed at the first SOTA GGUF certificate task because the failure ledger blocked the rerun before any parse-rate measurement. This is now the main PRD gap: Carnot still lacks a headline-eligible verifier result on the mandated local SOTA GGUF models.

2. **Smoke-only self-learning gap.** .99 produced positive self-learning deltas, but the GRPO/VPRM result was marked `headline_result_allowed=false` because no SOTA model ran. FR-11 needs a loop where verifier feedback improves future decisions on real local model outputs, not only replay/simulation.

3. **Continuous repair scale/semantics gap.** FSNet and SnareNet were positive on continuous EBM smoke tests, but the repair operator is not yet stress-tested against nonlinear constraints, feasibility-channel diagnostics, or EBT/ARM lookahead-energy interpretation. Without that, the Phase-3/Kona bridge is still empirical but shallow.

Publication receipt is a fourth operational gap, but it is narrow and should not dominate another research milestone.

## Architecture Target

```text
Phase 0: local structured certificate substrate
  SOTA GGUF cache/provenance preflight
      |
      v
  Grammar backend bakeoff
  llama.cpp native grammar vs llguidance vs XGrammar
      |
      v
Phase 1: headline local certificate path
  SOTA GGUF reasoning traces
  ARS answer-stability audit
      |
      v
  DCCD + TruncProof/XGrammar bounded <CARNOT_CERT> tail
  with structure-snowballing diagnostic
      |
      +---------------------------+
      |                           |
      v                           v
  BEAVER-lite/NSVIF routing + MCS/MUS  Cactus constrained acceptance v2
  claim -> verifier route        draft accept if cert/energy pass

Phase 2: continuous self-learning
  Certificate/energy decisions
      |
      v
  InterWhen + DVI verifier feedback replay
  uses SOTA certs when available, otherwise exp1274/FoVer replay
      |
      v
  Skill graph promotion/demotion
      |
      v
  GRPO/VPRM v9 only if SOTA + replay gates pass

Phase 3: Phase-3 repair and publication close
  FSNet/SnareNet positives
      |
      +--> HardNet++ nonlinear projection comparison
      +--> DSP feasibility-channel diagnostic
      +--> EBT/ARM lookahead-energy bridge audit
      |
      v
  arXiv receipt/blocker artifact + milestone retro
```

## Phase 0: Certificate Infrastructure Preflight

Goal: prevent another `exp1271` DOOMED_RERUN_BLOCK by proving the SOTA cache, grammar backend, and artifact/provenance path before asking an agent to run expensive model inference.

- `exp1282-sota-gguf-cache-provenance-preflight`: inspect the HF/llama.cpp cache for mandated SOTA GGUFs, verify the `cached_sota_pair()` path, and write a structured readiness artifact.
- `exp1283-certificate-grammar-backend-bakeoff`: benchmark llama.cpp native grammar, llguidance, and XGrammar availability/overhead for the fixed Carnot certificate schema, including a structure-snowballing-risk diagnostic.

Success bar: either `cached_sota_ready=true` and `grammar_backend_selected` is set, or the blocker is named before downstream tasks burn turns.

## Phase 1: SOTA Certificates and Constrained Acceptance

Goal: produce the missing headline local SOTA certificate result and only then run downstream acceptance/repair experiments.

- `exp1284-ars-answer-stability-sota-audit`: use mandated SOTA GGUFs to measure answer stability under reasoning/certificate boundary perturbations.
- `exp1285-triggered-certificate-extraction-v2`: rerun triggered certificate extraction with complete prior-failure metadata, bounded grammar, and SOTA cache gate.
- `exp1286-beaver-nsvif-semantic-routing`: turn successful certificates into routed claims, BEAVER-lite prefix/risk bounds, NSVIF-style CSP checks, a tiny ConstraintBench micro-eval, Minimal Correction Subsets, and residual-drift/MUS diagnostics.
- `exp1287-cactus-constrained-acceptance-v2`: rerun Cactus only when certificate parse rate reaches the structured gate, and report ARC-Decode-style `risk_bound_proxy`, `low_risk_acceptance_rate`, plus `speedbench_eval_mode` so replay-only speed claims cannot become headline claims.

Success bar: `exp1285.certificate_parse_rate >= 0.8` and `headline_result_allowed=true`, or the milestone honestly records why the local SOTA path is blocked.

## Phase 2: Continuous Self-Learning With Verifier Feedback

Goal: satisfy the research program's continuous self-learning mandate with a loop that uses verifier decisions from real or replayed certificate data.

- `exp1288-interwhen-dvi-verifier-feedback-replay`: convert verifier accept/reject decisions into an online drafter/routing-policy update and measure acceptance/violation deltas. It compares post-hoc verifier replay with an InterWhen-style interleaved verifier pass and records an OnlineSpec-style accepted-span/acceptance-rate delta. This task runs even if the SOTA certificate gate blocks by falling back to the `.99` certificate-memory/FoVer replay corpus, and it marks headline eligibility honestly.
- `exp1289-grpo-v9-sota-headline-gated`: run a bounded PRIME/VPRM/GRPO v9 headline attempt only if SOTA certificates and DVI replay gates pass.
- `exp1290-skill-graph-promotion-demotion`: promote reusable certificate-memory patterns into skill-graph entries with replay evidence and demotion conditions.

Success bar: at least one artifact reports `self_learning_delta_overall`, `dvi_acceptance_delta`, or `skill_replay_delta`, and marks whether the result is headline-eligible.

## Phase 3: Continuous Repair, Phase-3 Bridge, Publication, Retro

Goal: deepen the .99 continuous repair positive and close publication bookkeeping.

- `exp1291-hardnetpp-nonlinear-repair-benchmark`: compare HardNet++-style damped local-linear projection against FSNet/SnareNet on nonlinear constraints, recording fixed-budget construct/refine iteration behavior inspired by CaR.
- `exp1292-dsp-feasibility-channel-diagnostic`: add a DSP-style local/global feasibility-channel diagnostic (`phi`/`Phi`) around the continuous repair loop and report whether feasibility-channel signals predict when additional refine steps help.
- `exp1293-ebt-arm-ebm-cot-energy-bridge-audit`: connect Boltzmann-GPT/NRGPT/Carnot energy traces to EBT, ARM-EBM, and EBM-CoT sequence-energy predictions using existing artifacts.
- `exp1294-arxiv-v10-submission-receipt-or-blocker`: record an actual arXiv receipt if present or the exact external blocker if submission still needs a human.
- `exp1295-milestone-retro-100`: evaluate all criteria mechanically and write carry-forwards.

Success bar: continuous repair produces measured nonlinear/feasibility diagnostics, and publication state is no longer ambiguous.

## Dependency Graph

```text
exp1282 ---> exp1284 ---> exp1285 ---> exp1286
    |            |            |           |
    |            |            v           |
    |            |         exp1287        |
    |            |                        |
    |            +------------------------+
    |
exp1283 --------------------^

exp1285 - - optional SOTA corpus - -> exp1288 ---> exp1289
                                      |
                                      v
                                   exp1290

exp1291 ----+
exp1292 ----+--> exp1295
exp1293 ----+
exp1294 ----+
```

Structured conductor gates:

- `exp1284` gates on `exp1282.cached_sota_ready == true`.
- `exp1285` gates on `exp1282.cached_sota_ready == true`, `exp1283.grammar_backend_available == true`, and `exp1284.answer_stability_score >= 0.6`.
- `exp1286` gates on `exp1285.certificate_parse_rate >= 0.8`.
- `exp1287` gates on `exp1285.certificate_parse_rate >= 0.8`.
- `exp1288` is deliberately ungated so the continuous self-learning requirement still runs; it must record whether it used SOTA certificates or `.99` replay/FoVer fallback data.
- `exp1289` gates on `exp1285.headline_result_allowed == true` and `exp1288.dvi_acceptance_delta > 0.0`.
- `exp1290` gates on `exp1288.memory_update_written == true`.

## Hardware Requirements

Minimum:

- CPU-only path for grammar backend discovery, semantic routing, DVI replay fallback, skill-graph work, continuous repair diagnostics, publication receipt/blocker, and retro.
- Existing Python/JAX environment and repo dependencies.

Required for headline LLM results:

- At least one cached mandated SOTA GGUF:
  - `unsloth/Qwen3.6-35B-A3B-GGUF`
  - `unsloth/gemma-4-31B-it-GGUF`
  - `unsloth/gemma-4-26B-A4B-it-GGUF`
- Prefer dual RTX 3090 CUDA path for SOTA GGUF inference through llama.cpp-backed loaders.

Not required in .100:

- KV260/Vivado FPGA synthesis. The continuous repair tasks are software diagnostics only.
- AMD XDNA NPU unblocking. NPU remains useful for future edge deployment but has repeated human-install blockers.
- Extropic TSU access. The Extropic/THRML path remains strategic context until public hardware/API access changes.

## Decentralization Implications

This milestone preserves local-first operation. Every LLM-bearing task is gated on local open-weight GGUF availability and must record exact `MODEL_SPECS`. Tiny legacy models may only produce smoke artifacts with `headline_result_allowed=false`. No task depends on closed-weight provider calls, no vendor SDK enters the core verifier stack, and all hardware-facing claims stay behind sampler/reasoner abstractions.

## Milestone Success Criteria

1. `exp1282` records SOTA GGUF cache/provenance readiness or a named blocker.
2. `exp1283` selects a local certificate grammar backend or reports no viable backend.
3. `exp1284` measures answer-stability plus certainty/consistency routing signals on SOTA GGUF outputs.
4. `exp1285` produces a headline-eligible SOTA certificate parse-rate measurement or blocks honestly, comparing raw trigger, grammar-only, and DCCD-style draft-conditioned paths when possible.
5. `exp1286` writes semantic routing, BEAVER-lite prefix/risk bounds, NSVIF-style CSP checks, a ConstraintBench micro-eval, MCS/MUS diagnostics, and residual-drift cases when certificate parse succeeds.
6. `exp1287` measures Cactus constrained acceptance when certificate parse succeeds, including risk-bound proxy, low-risk acceptance rate, and SPEED-Bench-style evaluation-mode tagging.
7. `exp1288` measures InterWhen-style interleaved verification and DVI-style verifier-feedback replay deltas.
8. `exp1289` reports a headline-eligible GRPO/VPRM v9 delta when gates pass.
9. `exp1290` emits skill-graph promotion/demotion entries with replay evidence.
10. `exp1291` compares HardNet++ nonlinear repair against FSNet/SnareNet.
11. `exp1292` measures DSP feasibility-channel diagnostics.
12. `exp1293` writes the EBT/ARM/EBM-CoT energy bridge audit.
13. `exp1294` records arXiv v10 submitted receipt or exact external blocker.
14. `exp1295` completes the .100 retrospective.

## Key Planning Decisions

- The milestone sequence increments from 2026.04.99 to 2026.04.100.
- All planned tasks use `agent_type: codex` and `model: gpt-5.5` per CLAUDE.md. No task justifies `requires_claude: true`.
- LLM-bearing tasks include explicit mandated SOTA GGUF `MODEL_SPECS` requirements and use `cached_sota_pair()` first.
- Carry-forwards from .99 include explicit `prior_failures` entries: `exp1271`, `exp1273`, and `exp1277`.
- Structured gates are populated wherever a title or task dependency depends on an artifact field.
- No task modifies `research-roadmap.yaml`.
- No task modifies `scripts/research_conductor.py`.
