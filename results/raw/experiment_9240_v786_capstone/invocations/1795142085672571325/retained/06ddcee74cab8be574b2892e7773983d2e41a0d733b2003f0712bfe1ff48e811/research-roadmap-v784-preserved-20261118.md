# Carnot Research Roadmap v784: Closed Replay Authority, Human Energy Decisions, and Continuous Label-Robust Learning

**Milestone:** 2026.11.784
**Created:** 2026-11-17 America/New_York
**Status:** Proposed; no experiment execution or queue activation.
**Predecessor:** 2026.11.783, scheduling complete.
**Execution contract:** Exactly **14 tasks, Exp9208 through Exp9221**, in the
order below, across **four phases**. The full embedded task objects must equal
`research-roadmap-next.yaml`, including prompts, gates and model metadata.

The [V783 design](research-roadmap-v783-preserved-20261117.md) and
[V783 activated queue](research-roadmap-v783-queue-preserved-20261117.yaml)
are preserved byte for byte. Their SHA256 values are respectively
`14f2bec94e215fe965080ff2388f5df0c1b221398b527a98d5a05cd9671dfa08` and
`1c0ff24ad77aa6c1f361117cf842aaa1e61cc068be548ccd3e5f99d30ba899df`.
The active queue, conductor, completion archive and exclusion manifest remain
unchanged. At planning read time the completion archive ends at V782; V783
primaries and conductor logs establish the newer outcomes.

## What V783 proved

| Experiment | Observed outcome | What it does not establish |
|---|---|---|
| Exp9194 | Fourteen complete objects matched; owned coverage eventually reached 100 percent | Required checks failed; contract/source/static/online readiness all remain zero |
| Exp9195 | Scheduler `blocked_gate_check_failed`, contract operand zero | No lossless corpus reader was executed |
| Exp9196 | Log-only upstream-retired skip | No complete forward qualification |
| Exp9197 | Scheduler `blocked_gate_check_failed`, source operand zero | No ambiguity reducer or label-method qualification |
| Exp9198–9206 | Nine log-only upstream-retired skips | No new Qwen features, static/online effect, CES or complete cost |
| Exp9207 | Reconciliation ready; `complete_blocked_declared_gate` | Administrative closure does not turn absent science into a null result |

**Two tasks executed and twelve skipped.** Authority is
`results/experiment_9194_v783_contract.json`,
`results/experiment_9207_v783_capstone.json`, their retained raw work/receipts,
`results/experiment_9195_corpus_prefilter.json`,
`results/experiment_9197_label_ambiguity.json` and `ops/conductor-log.md`.
The capstone's normalized accounting verdict is not an invented verdict from
an absent scientific primary. V782 remains four executed and ten skipped.

The new failure is observable. Exp9194's original owned suite had one failure:
its private publication child reached cold replay and refused with
`'entropy_reference_protocol'`. Inherited V782 replay uses a rebound evidence
module whose newer configuration does not own that protocol. Its direct-consumer
run had twelve failures: V782 tests compared old pins with the current design
and queue, then V782 capstone reduction received `exp9194-contract` outside its
own task list. The old assertions detected real input-isolation failures.

V784 tests explicit protocol ownership and private historical input roots.
Original assertions, hashes and quarantines remain. It then tests the previously
unexecuted lossless corpus correction and complete validation union. No prior
failure becomes passing through a new name, copied receipt or changed pin.

## Three biggest gaps between current state and PRD

1. **FR-12: an oracle-distinct factual verifier.** Authenticated human labels
   exist, but the matched energy/probability comparison is still unmeasured.
   Measure it with original controls, then state exactly which annotation target
   was evaluated. The exposed panel cannot establish independent generalization.
2. **FR-06/FR-11: useful continuous self-learning.** Durable mechanics are not
   evidence of lower future loss. Test the frozen six-arm small-head learner,
   delayed labels, calibration-only comparator, retention and actual recovery.
3. **FR-05/FR-08/NFR-01: reproducible, affordable execution.** Mutable historical
   inputs and incomplete cold replay prevent valid research. Qualify the concrete
   correction, complete all affected checks, and measure whole-service cost before
   selecting a native or hardware acceleration path.

The November 7 north star is the oracle-distinct verifier. ARC is opportunistic;
its disqualified compatibility route is deferred. This plan changes neither
public defaults nor generator weights. Architecture's global reconciliation date
is still October 11; this proposal makes a scoped planning addition only.

## Research recorded before experiment design

The V784 scan in [research-references.md](../../research-references.md) checks
all requested primary topics and secondary channels, with retrieval limits.

| Finding | Experiment use | Boundary |
|---|---|---|
| [Hallucination taxonomy v2, November 11](https://arxiv.org/abs/2610.38520v2) | Exp9211 target-identifiability audit; Exp9219 target-scope reporting | Original factual-error labels cannot automatically identify expressed uncertainty or prompt-recoverable errors |
| [Labeling Problem](https://arxiv.org/abs/2610.08026), [Automatic Labeling](https://arxiv.org/abs/2611.04697) | Exp9211 exact ambiguity reducer; Exp9219 sensitivity | Keep A1 primary, actual A2 and missing labels distinct |
| [ARM–EBM equivalence](https://arxiv.org/abs/2512.15605), [Spilled Energy](https://arxiv.org/abs/2602.18671) | Exp9212–9219 preserved energy/probability and gauge controls | Unnormalized logits are not factual truth |
| [Continual Calibration](https://arxiv.org/abs/2604.23987), [online spline locality](https://arxiv.org/abs/2602.02056) | Exp9218 causal small-head updates; Exp9220 whole adaptation costs | No imported guarantee or hardware speedup |
| [Entropy Distribution](https://arxiv.org/abs/2605.28264) | Exp9219 retained CES comparison | Secondary ranking never rescues primary gates |

EBT and the November Energy-Based Reasoners paper remain generator research
leads. Neural Ising Machines, nonlinear constraint projection and EBD do not
repair source or replay authority. Retired repair, generated-confidence transport,
importance-anchor and generic external-text reranker mechanisms remain closed.
The current study scores authenticated human responses with frozen internal logits;
it is not a reopened candidate-reranking experiment from retired Phase D.

OpenReview's direct forum required browser verification. Both Semantic Scholar
citation endpoints returned HTTP429 and browser fallback failed; no new citation
count is claimed. GitHub Trending returned a stale snapshot. Extropic Z1T is
an operation-mapping lead; Kona's product page supplies no newly verified local
architecture. No vendor performance number becomes a Carnot measurement.

## Architecture

```mermaid
flowchart TD
  O[Original human source and method authority]
  R[Protocol-owner replay and private historical inputs]
  C[Exact V784 contract and qualified source/method gates]
  P[Lossless full-corpus reader]
  V[Complete original and affected validation]
  Q[Frozen Qwen3.8 forward-only qualification]
  A[Shard A: 100 qids]
  B[Shard B: 100 qids]
  D[Shard C: 100 qids]
  J[Exact 900-answer admission]
  S[Five static energy arms]
  L[Six delayed-learning arms and recovery]
  U[Independent source authentication, target audit, ambiguity method]
  E[Primary, CES and sensitivity audits]
  K[Complete raw-text and durable-update costs]
  Z[Unconditional fourteen-outcome reconciliation]
  R --> C --> P --> V --> Q
  O --> C
  O --> U
  Q --> A --> J
  Q --> B --> J
  Q --> D --> J
  J --> S --> E
  J --> L --> E
  U -. sensitivity and scope only .-> E
  Q --> K
  S -. eligible service .-> K
  L -. eligible service .-> K
  E --> Z
  K --> Z
  U --> Z
  C --> Z
```

Labels stay outside model inputs. Static fitting uses TRAIN150; online fitting
uses its separate TRAIN100 initialization. CAL guards promotion; TEST retention
cannot select updates. All five current model tasks use forward-only execution
and generate zero tokens. Archived answers are not new Qwen generations.

## Phase 1 — Close replay and identify targets (Exp9208–9211)

Exp9208 fixes the two demonstrated replay/input-authority failures and binds the
fourteen tasks. All original affected assertions must pass. Exp9209 requires
both contract and replay readiness, then qualifies lossless corpus traversal.
Exp9210 executes the complete original 621 plus 53 added nodes and every affected
V780–V784 obligation. A subset grants no full-qualification credit.

Exp9211 has no administrative or model gate. It independently authenticates the
original human-source custody chain, measures target identifiability and proves
the existing exact ambiguity bounds. Failed source authentication still blocks
it. This branch can produce a useful label-method result when model validation
fails, without bypassing any scientific prerequisite.

## Phase 2 — Collect the unchanged human population (Exp9212–9216)

Exp9212 qualifies serial/chunk128 frozen Qwen forwards and budgets three complete
300-answer shards. Exp9213–9215 share one collector with separate checkpoints;
none depends on another shard. Exp9216 admits exactly the original 900 answers.
No changed cap, selected intersection or replacement response rescues an incomplete
panel. Each shard keeps original roles; shard membership is not a new data split.

## Phase 3 — Test decisions and continuous learning (Exp9217–9219)

Exp9217 fits five fixed binary-energy arms. Exp9218 runs six delayed-learning arms
and genuine hard-exit recovery. Exp9219 audits every eligible branch independently,
with the original two static, seven learning and three secondary CES bounds.
Target identifiability and ambiguity extrema describe scope and robustness;
neither changes fitting, primary acceptance or promotion.

## Phase 4 — Measure complete costs and reconcile (Exp9220–9221)

Exp9220 times each eligible service independently of measured benefit. Include
cold/warm forwards, rejected updates, guards, storage and recovery. Exp9221 always
accounts for all fourteen outcomes and exact failed operands. Scientific absence
is terminal blocked; only unfinished owned work uses partial. Retirement follows
authenticated repeated failed scope, not an unmeasured scientific hypothesis.

## Replay correction and falsifiable stop rule

Two observed failure paths define Exp9208's scope:

- **Protocol ownership:** inherited entropy replay must use its original,
  hash-bound owning configuration. The newer ambiguity configuration is a
  separate authority. Fresh-process replay must validate both exact sidecars.
  Adding a default key, copying the expected answer or ignoring missing fields
  cannot qualify. Missing, modified and rehashed invented protocols must fail.
- **Historical input isolation:** run historical suites against private roots
  containing their original design, activated queue, scheduler evidence and
  producer snapshots. Change fixture/input construction only when needed; retain
  every assertion and mutation. Never give a V782 reducer V783/V784 task objects.
  Current activation checks must still validate the actual current task list.

Use existing source-reference, immutable-snapshot and private-CLI mechanisms.
No new generic validation framework is required. Preserve failed originals and
record the corrected revision separately. Test old/new replay interleaving and
successor-queue contamination. Full affected suites, real publication/refusal,
tamper controls and cold replay must pass; coverage alone cannot grant readiness.
`replay_authority_ready_score` and `contract_ready_score` are separate exact
integer fields. Exp9209 requires both. Even after corpus qualification, Exp9210
must execute the complete original and affected validation union.

## New target-identifiability experiment

Exp9211 authenticates the original human source directly through existing
Exp9097 custody and original Exp9126/9132 role references. It does not depend
on Exp9208 readiness or an Exp9194 success that never occurred. All original
source hashes, labels, annotation instructions and role checks remain necessary.
This is a different dependency, not weaker source authentication.

Freeze a rubric before reading predictions. For every original response, record:

1. Whether original human factual-error evidence and its criterion are present.
2. Whether an original human annotation explicitly identifies expressed uncertainty.
3. Whether original same-generator, semantically equivalent prompt attempts provide
   the re-elicitation evidence required for the paper's SI/CF classification.

Each evidence status is present, absent or ambiguous, with exact source pointers
and counts by original role. Absent annotation is not a negative class. Different
model siblings are not repeated prompts to one generator. A single answer cannot
show knowledge is absent. Do not add a keyword detector, judge model, new generated
response or guessed taxonomy label. If no SI/CF target qualifies, report zero
identifiable rows and unavailable category metrics. That is a complete result.

The existing binary A1 target remains unchanged. Test missing evidence, conflicting
criteria and cross-generator substitution. `target_identifiability_ready_score`
means the audit is complete and checked, not that useful categories exist. Its
readiness is separate from `label_ambiguity_ready_score`. Exp9219 can evaluate
primary factual-error predictions without either secondary result; it must then
state the missing secondary evidence. Neither result selects a learned state.

## Inherited methods and acceptance thresholds

The following lossless-reader, annotation-sensitivity, scientific and cost
protocols retain V783's exact population, arms and bounds. Exp9126 and Exp9132
original immutable methods remain scientific authority. V784 changes prerequisite
replay/input custody and adds target identifiability, not a more favorable test.
## Lossless corpus correction and its stop rule

The target classifier recognizes five fixed ASCII substrate strings at the top
level or in the existing principle wrapper. A JSON string spelling any of those
values contains either the literal sentinel or a JSON Unicode escape. Therefore,
absence of every sentinel and of the Unicode escape marker is a necessary
negative condition. It is not a positive classifier.

Stream each file in bounded chunks. Retain overlap of at least the longest
needle minus one byte. A negative decision requires EOF. Any sentinel or Unicode
escape sends the entire file to the unchanged `read_text`/`json.loads` path and
classifier. False-positive candidates are harmless extra parsing. Invalid JSON,
invalid encoding, non-dicts, duplicate keys, wrapped values and annotated suffixes
retain original behavior. No directory, filename, age or size is excluded.
Original `checked > 0` and `offenders == []` assertions remain executable.

Freeze the classifier revision and sentinel inventory together. Add controls for
every sentinel and escape handling; deleting any fallback must fail a control.
Measure original and candidate decoding on identical predetermined size strata.
The planning census is not a content seal. Record a complete current path manifest,
bytes read and changed files. Do not call a concurrently changing corpus complete.
Do not claim full-corpus speedup from a censored baseline.

The new gate is `corpus_scan_ready_score`. Exp9181's old gate remains zero.
Readiness requires the full original target module, original V782 mutation and
structural controls, every affected consumer, immutable receipts and owned coverage.
Exp9210 still runs the entire validation union, including the corrected replay and historical input-root checks, with original assertions. No subset,
old passing receipt or cache of verdicts can stand in for execution. If full
validation cannot finish, the model branch stays blocked and Exp9211 still runs.

## New label-ambiguity protocol

Use the original 300 question groups and all three answer siblings. A1 remains
the primary binary factual-error target. Actual A2 annotations are observations;
`-1` means missing. Do not replace either annotator with model judgments.

Freeze two sets per original response before model work:

- **Observed-disagreement sensitivity:** agreements fix A1; observed A1/A2
  disagreements permit either binary value; missing A2 fixes A1 but is reported.
- **Missing-annotation envelope:** agreements fix A1; disagreements and missing
  A2 permit either binary value. This deliberately broad envelope does not assume
  that all unobserved annotations would disagree.

For fixed control and treatment error probabilities c and t, the Brier improvement
is `d(y) = (c-y)^2 - (t-y)^2 = c*c - t*t - 2*y*(c-t)`.
It is affine in binary y. Evaluate allowed endpoints to obtain exact minimum and
maximum contrast under each scenario. Use the original equal-qid weights and all
three siblings. An unchanged label belongs to the response, not to a measurement:
for repeated retention across streams/updates, first sum that response's weighted
coefficient, then choose its endpoint. Never assign a different label to the same
response at each appearance. Pairwise worst cases need not be jointly attainable
across several contrasts; do not claim a simultaneous worst-case assignment.

Exp9211 independently authenticates original source custody and tests the reducer using exhaustive small binary panels, duplicated
retention, permuted order, constant scores, missing annotations and boundary
probabilities. It publishes masks and code receipts, not natural model effects.
Exp9219 uses genuine frozen predictions to report both endpoint intervals beside
the original A1 estimate and sampling interval. Endpoint robustness is conditional
on the declared ambiguity set and is not a confidence interval or factuality proof.
No mask, endpoint, A2 label or sensitivity result selects training, temperature,
learning promotion or primary acceptance. Missing sensitivity authority blocks
only that secondary branch.

## Frozen scientific protocols


The following methods preserve the V778 and V779 scientific contract. Their implemented
authorities are Exp9126 and Exp9132. Their immutable receipts take precedence
over mutable checkout files. The new diagnostic section below adds no acceptance
gate and changes no arm, cohort, optimization rule or interval threshold.

### Whole-answer forward statistics

Use original TRAIN150/CAL50/TEST100 question groups, three siblings each.
Primary target is original A1 factuality. A2 sensitivity uses only actual observed
labels; unknown -1 stays missing. The task evaluates archived answers under Qwen,
not newly generated Qwen answers. No answer accuracy claim transfers to Qwen.

Preserve V777's exact protocol. Tokenize question/answer separately with the
embedded tokenizer, no BOS or chat template, and right caps256/768. The prefix is
literal `Question:\n`, retained question text and literal `\nAnswer:\n`.
Freeze detokenization and prefix bytes. Append retained answer token IDs explicitly.
Reset state for every answer. Capture in128-token chunks with the preceding-prefix
logit for the first answer token. Preserve adjacent pairs across chunk boundaries.

For full-vocabulary logits z_i and answer token a_i, compute in stable float64:

- L_i=logsumexp(z_i); NLL_i=L_i-z_i[a_i].
- p_i=softmax(z_i); H_i=-sum_v p_iv*log(p_iv).
- Marginal energy M_i=-L_i.
- Adjacent spill D_i=-z_i[a_i]+L_(i+1), within one answer, i=1..length-1.

The seven features are log1p(answer length), mean NLL, mean H, maximum H,
mean M, mean absolute D and the no-adjacent-pair indicator. Retain selected logits,
L_i, H_i, max probability, token IDs, positions and full-vector hashes per token.
One-token answers lack only spill; TRAIN-only median imputation is fixed. An
entirely missing training feature blocks fitting. Empty answers retain their slots
as escalations. Loss contrasts share predeclared nonempty support; coverage uses
all original TEST rows. No label-based filtering is permitted.

Store full vectors for the first eight answer positions of the global first20
lexical qids/all siblings. Preflight2GiB for bounded raw evidence. Stream statistics
instead of retaining all900 complete token-by-vocabulary arrays in RAM.

### Static energy decisions and susceptibility

The five original arms are prevalence, probability+length logistic (first four
features), raw-energy logistic (length, mean M, mean absolute D, missing-pair
indicator), combined logistic (all seven) and combined with whole-qid TRAIN donor
labels. Donors use a fixed nonidentity cyclic permutation of sorted TRAIN qids;
sibling positions stay fixed. Report the actual changed-label numerator/denominator.

These are binary Gibbs energies E(0)=0, E(1)=-theta·phi and
p(error)=sigmoid(theta·phi). Fit TRAIN-only population mean/std, with constant
scale1. Minimize mean binary logloss plus0.5*0.01 times squared nonintercept
coefficients. L-BFGS-B starts at zero: maxiter2000, gtol1e-7, ftol1e-12.
Require finite parameters/objective and gradient infinity norm<=1e-6.
No regularization or architecture search is scheduled.

CAL alone selects temperature from[0.5,1,2,4], ascending ties. Accept at
p(error)<=0.1; reject at>=0.9; otherwise escalate. Seal TEST predictions before
evaluator access. Report Brier, logloss, coverage, accepted error and exact-half-
coverage error. Rerank half-coverage choices inside each resampled draw.

Use10000 paired TEST-qid bootstrap draws, seed777121, preserving siblings.
`energy_increment_score=1` requires both combined-arm Brier improvement lower
bounds>0 against probability+length and donor, with one-sided tail0.05/2.
Empirical accepted error must not increase versus probability+length.
Undefined accepted risk cannot pass. These are conditional development intervals.

Add c_i=3 or c_i=3*(-1)^i to every vocabulary logit. Normalized probabilities,
NLL and entropy must remain unchanged. M changes by -c_i; D changes by
c_(i+1)-c_i. Report raw-score and frozen-head sensitivity under both gauges.
Do not select the favorable gauge. A gauge-sensitive score can be an empirical
representation signal; it is not evidence of a broken probability chain rule.
The original Exp6980 exact-span null and V776 embedding study remain unchanged.

### Continuous self-learning with a recalibration control

Preserve the original SHA256(`v776-online`+qid) ordering:100 TRAIN groups for
initialization and50 for ten disjoint streams of five. Original CAL50 is the
operational guard; TEST100 is research retention. Preserve the Exp9099 donor map
(seed776099, initialization labels only, original sibling positions).

Fit imputation and scaling only on the100 initialization groups. After standard
scaling, fit initialization min/max to[-1,1], constants0, and clip later inputs.
Use the seven logit features with fixed cubic knots
[-1,-1,-1,-1,-0.5,0,0.5,1,1,1,1]: intercept, seven linear columns and49 basis
columns. Use the existing spline helper. Fit initial heads with the static ridge
objective and convergence contract; temperature starts at1. No random projection
or embedding transformation is used in this separate seven-feature study.

Six arms are frozen spline, adaptive spline, bias-only spline, adaptive linear,
donor-label spline and temperature-only spline. All start from comparable
initial fits. For the original adaptive arms, retain the Exp9099 gradient:
`(sigmoid(theta·phi)-y)*phi + 0.01*(theta-theta_initial)`, with no intercept
anchor. Bias-only zeros other coordinates. Clip L2 norm to1. Use AdaGrad,
learning rate0.05, accumulator initially0 and epsilon1e-8. Apply three sibling
updates in original order, then test promotion once per released qid.

For temperature-only adaptation, keep spline coefficients fixed. Let s be its
initial logit, u=log(T), q=exp(-u), p=sigmoid(q*s). Initially u=0. Use gradient
`(y-p)*q*s + 0.01*u`, clipped to[-1,1], with the same AdaGrad settings.
Clip updated u to[log(0.5),log(4)]. Persist u and its accumulator. This comparator
uses the same feedback and guard access; it is not a multi-group MT-CC replication.

Promote only if complete CAL Brier increase<=0.005 and CAL classification-error
increase<=0 versus that arm's initial state. Otherwise restore parameters and
accumulators while acknowledging consumed feedback. Research retention does not
participate in this decision. Seal all18 sibling/arm predictions for g before
releasing labels for g-1. Drain the final group after all predictions. Reset
every stream to identical initial states. No static150 data may enter this fit.

Atomic snapshots include predictions, parameters, accumulators, temperature,
pending feedback and acknowledgments. Real hard exits at prediction fsync,
feedback receipt and state replacement must reproduce uninterrupted states and
predictions exactly. Preserve private actual-child E2E-020 coverage.

Evaluate future groups2–5:40 distinct qids/120 responses. Use10000 paired stream
bootstrap draws, seed778135, one-sided tail0.05/7. The seven simultaneous bounds
are five adaptive-spline future Brier improvements over each other arm (lower>0),
retained Brier increase (upper<=0.005), and retained classification-error increase
(upper<=0). All seven plus exact recovery are needed for `learning_signal_score=1`.
This adds a comparator and tightens multiplicity correction; it does not replace
the old six-bound embedding protocol. Report per-qid retention changes averaged
over streams. Repeated retention is100 questions, not1000. Ten streams have limited
power and the intervals condition on the initial fit, guard and retention panel.



## Descriptive diagnostics fixed before execution

Reuse Exp9141's authenticated evaluator masks and original agreement counts.
Unknown A2 labels stay missing. Agreement is not an accuracy ceiling. Static
slices distinguish mixed-correctness siblings from homogeneous qids. These
label-defined masks never enter feature extraction, fitting or decisions.

For online predictions, retain `abs(p_adaptive-p_initial)>=0.05` and crossings
of the fixed0.5 classification threshold. Membership uses predictions only.
Compare every arm on identical changed and unchanged units. Report Brier,
`mean(p-y)`, exact counts and unavailable empty slices. Use paired whole-stream
resampling for future and retention contrasts, conditional on the fixed panels.

The new TOOD-inspired diagnostic compares ranking, scale and calibration at
each released-group update. Use the same TEST100 qids and all siblings for every
arm. Record raw logits, RMS logit magnitude, tie-aware AUROC, Brier and class
counts before and after each update. Rejected updates retain their unchanged
state. One-class AUROC remains unavailable. Report every stream and update.
No threshold, temperature, group or model is selected from these results.
This diagnostic cannot rescue any failed primary bound. It tests whether the
observed behavior is consistent with scale drift or lost ranking information;
it cannot identify a causal mechanism from a correlation alone.


## Additional descriptive diagnostic: exact Brier moments

Use the frozen original supports and equal-qid weighting. Let p be predicted
error probability and y the human error label. Use population moments throughout:

`Brier = (mean(p)-mean(y))^2 + (sd(p)-sd(y))^2 + 2*(sd(p)*sd(y)-cov(p,y))`.

The three terms are squared mean bias, variance mismatch and covariance deficit.
Compute the last term directly from covariance when either variance is zero;
correlation is then unavailable. The identity must match direct Brier within
1e-12 on deterministic controls and measured rows. Report raw signed residuals.
Do not clamp roundoff before checking the identity. Empty supports remain unavailable.

Use all five static arms and all six online arms. Report static TEST, future
stream support and each retained panel separately. The original qid and stream
resampling units apply to diagnostic deltas. Neither labels nor these diagnostics
select features, update rules, temperatures or promotion thresholds. No new
significance test replaces the original two static and seven learning bounds.

## Secondary experiment: a frozen entropy reference

Exp9208 fixes this protocol before model work from its original owning authority. For all nonempty A1-correct
TRAIN150 answers, pool token entropies and define F0(x) as the fraction<=x.
Preserve the paper's token weighting and report tokens and distinct qids.
A missing correct reference blocks this branch. No CAL or TEST response defines
F0. For each TEST answer, compute `CES=sqrt(F0(mean(H))*F0(max(H)))`.
Larger CES, mean H and max H predict error. CES is not an error probability.

Compare CES AUROC against mean H, max H and the probability+length head's error
probability. Use10000 paired TEST-qid bootstrap draws, seed782190, preserving
siblings and half-credit ties. Require every contrast's one-sided lower bound
at tail0.05/3>0 for `entropy_reference_signal_score=1`. If a draw lacks either
class, the affected interval is unavailable; do not drop/redraw it to obtain a
favorable interval. Retain raw estimates and valid/invalid draw counts. Score
all original nonempty TEST answers; report empty exclusions and all900 original
slots separately. Missing fitted-head predictions block that contrast, not the
mean/max comparisons. Completion and effect gates are separate.

Record per-answer token count, lag1 correlation when variance/support permit,
ties, mean/max entropy and every score. An empty answer or one-class TEST set
cannot receive invented AUROC. This is a secondary ranking experiment: no new
probability threshold, learned feature, online arm or primary bound. Its outcome
can motivate a later study, not close independent factual verification.


## Complete cost and hardware requirements

Use first20 TEST qids/all60 siblings, three fresh processes, seed776108 and
probability parity atol1e-5/rtol1e-4 with exact typed decisions. Distinguish cold
load, warm uncached raw-text service and cached-feature head timing. Include
forward passes, full-vocabulary reduction, transfer, transforms and serialization.
Online costs include initialization, proposal, rejection, all guards, fsync,
acknowledgment, drain and recovery. Retain censored units and complete denominators.

| Resource | Requirement and claim boundary |
|---|---|
| CPU/RAM | Validation, small-head fitting, bootstrap and durable updates; measure peak memory and stream vocabulary statistics. |
| Private disk | At least2GiB for designated raw vectors plus measured traces/checkpoints; test actual free space. |
| GPU0 RTX3090 24GB | Verify CUDA offload under the existing lease. Five model tasks execute serially; do not disturb GPU1 ownership. |
| Mandated model | Frozen `unsloth/Qwen3.8-27B-GGUF`, Q4_K_M, existing `cached_current_model(gpu_index=0)` resolver; pin model, tokenizer and runtime hashes. |
| KV260 | Preserve qualified Ising operations and k_max<=5; no dense Qwen or spline-training kernel is inferred. |
| PolarFire | Preserve qualified Linux CPU dispatch; this does not prove fabric acceleration. |
| GateMate | Preserve unchanged physical/JTAG `0xffffffff` blocker; no identical probe without changed physical attestation. |
| NPU/TSU/larger FPGA wishlist | Require access, matching operations and a measured bottleneck before implementation or procurement. |

Continuous learning has a CPU path now and possible batched GPU/native or
fixed-point FPGA arithmetic later. Dense anchors, guard scans and disk commits
remain part of its cost. Amdahl's limit is `1/(1-f)` for an infinitely fast
replacement of fraction f: at least0.99 for100x,0.90 for10x; finite acceleration
needs strictly more coverage. Unknown fractions stay unknown. Sampling hardware
cannot by itself remove dense forwards or durable storage. No power claim without
measured energy integrals, and no purchase or board operation is scheduled.


## Execution, routing and stop rules

Every prompt has a numbered step requiring flushed phase boundaries, before/after
minute-scale calls, loop progress and supervised 60-second heartbeats. Every
silence gap stays below 600 seconds. Files over about 200 lines are written in
several tool calls with progress between them. The 4800-second cap reserves 3300
seconds for work, 1200 for final verification and 300 for publication/cleanup.
Measured budgets precede full runs; estimates never certify completion.

Exp9208–9210 use Claude Opus with 100 turns for schema/replay and validation
coordination. Formulaic runtime, collection and admission tasks use Codex
`gpt-6.1-sol`; scientific and judgment tasks use Sonnet. No Luna or Gemini tasks.
All five LLM tasks require `unsloth/Qwen3.8-27B-GGUF` in MODEL_SPECS and request
`model_load_no_generation`, with actual forward receipts and a 2-second floor.
No generation is scheduled. Any later bounded canary needs the 10-second class;
full generation needs the 60-second class. Never pad time. A pre-load refusal
is blocked_no_run; attempted loads keep their actual counts. CPU reductions
have empty MODEL_SPECS and zero current model invocations.

All comparative tasks require primitive rows, original independent-unit counts,
censor reasons and closed verdict classes. Failed external gates are blocked,
not partial. Every blocked artifact includes gate_check_summary with the exact
failed operand. Gates test readiness, not scientific effect. All scope continuations
carry four-field prior failures and named forward differences. Standing May 29
overrides are limited to those differences or routine capstone accounting.

## Dependency graph and exact gate contract

Conductor execution is the table order. Collection shards have independent gates;
the label-method task and capstone have no structured upstream dependency.
Missing label-method authority blocks only sensitivity, not primary auditing.

| Consumer | Required current upstream field(s), each == 1 |
|---|---|
| exp9208-contract | None; authenticate own inputs |
| exp9209-corpus-prefilter | exp9208-contract.contract_ready_score; exp9208-contract.replay_authority_ready_score |
| exp9210-forward-qualification | exp9209-corpus-prefilter.corpus_scan_ready_score |
| exp9211-label-ambiguity | None; authenticate own inputs |
| exp9212-logit-runtime | exp9210-forward-qualification.forward_validation_ready_score; exp9208-contract.static_methods_ready_score |
| exp9213-logits-a | exp9208-contract.source_ready_score; exp9212-logit-runtime.logit_runtime_ready_score; exp9212-logit-runtime.shard_feasible_score |
| exp9214-logits-b | exp9208-contract.source_ready_score; exp9212-logit-runtime.logit_runtime_ready_score; exp9212-logit-runtime.shard_feasible_score |
| exp9215-logits-c | exp9208-contract.source_ready_score; exp9212-logit-runtime.logit_runtime_ready_score; exp9212-logit-runtime.shard_feasible_score |
| exp9216-logit-admission | exp9213-logits-a.logits_a_ready_score; exp9214-logits-b.logits_b_ready_score; exp9215-logits-c.logits_c_ready_score |
| exp9217-static-energy | exp9216-logit-admission.admission_ready_score; exp9208-contract.static_methods_ready_score |
| exp9218-delayed-learning | exp9216-logit-admission.admission_ready_score; exp9208-contract.online_methods_ready_score |
| exp9219-decision-audit | exp9216-logit-admission.admission_ready_score |
| exp9220-complete-cost | exp9212-logit-runtime.logit_runtime_ready_score |
| exp9221-capstone | None; authenticate own inputs |

## Task contract

Exactly 14 tasks, Exp9208 through Exp9221, in this order. Each path is a distinct
canonical deliverable; raw receipts and method sidecars are subsidiary evidence.

| Order | Phase | Task ID | Deliverable |
|---|---|---|---|
| 1 | 1 | exp9208-contract | results/experiment_9208_v784_contract.json |
| 2 | 1 | exp9209-corpus-prefilter | results/experiment_9209_v784_corpus_prefilter.json |
| 3 | 1 | exp9210-forward-qualification | results/experiment_9210_v784_forward_qualification.json |
| 4 | 1 | exp9211-label-ambiguity | results/experiment_9211_v784_label_ambiguity.json |
| 5 | 2 | exp9212-logit-runtime | results/experiment_9212_v784_logit_runtime.json |
| 6 | 2 | exp9213-logits-a | results/experiment_9213_v784_logits_a.json |
| 7 | 2 | exp9214-logits-b | results/experiment_9214_v784_logits_b.json |
| 8 | 2 | exp9215-logits-c | results/experiment_9215_v784_logits_c.json |
| 9 | 2 | exp9216-logit-admission | results/experiment_9216_v784_logit_admission.json |
| 10 | 3 | exp9217-static-energy | results/experiment_9217_v784_static_energy.json |
| 11 | 3 | exp9218-delayed-learning | results/experiment_9218_v784_delayed_learning.json |
| 12 | 3 | exp9219-decision-audit | results/experiment_9219_v784_decision_audit.json |
| 13 | 4 | exp9220-complete-cost | results/experiment_9220_v784_complete_cost.json |
| 14 | 4 | exp9221-capstone | results/experiment_9221_v784_capstone.json |

## Planning verification

Validate complete object equality, exact task order, prompt rendering, unique
IDs, backward gate fields, four-field prior failures and operation-accurate
model metadata. Run roadmap schema, gate, exclusion, harness-fit, overdue-priority
and ARC-floor checks. Run relevant unit and private E2E-018/020/021 tests, scoped
lint/spec tracing and whitespace checks. Record repository-wide reconciliation
limitations separately. Planning does not execute the proposed experiments.


## Full machine-readable contract

```yaml
milestone: 2026.11.784
milestone_title: Closed Replay Authority, Human Energy Decisions, and Continuous Label-Robust Learning
milestone_doc: openspec/change-proposals/research-roadmap-vNEXT.md
tasks:
- id: exp9208-contract
  title: 'Exp9208: Qualify protocol-owner replay and bind fourteen current tasks'
  phase: 1
  track: infrastructure
  priority: high
  requires_gpu: false
  max_turns: 100
  estimated_wall_time_min: 75
  per_unit_rows: true
  milestone: 2026.11.784
  deliverable: results/experiment_9208_v784_contract.json
  inference_substrate_class: no_model_load
  MODEL_SPECS: []
  model: opus
  prior_failures:
  - experiment_id: exp9194-contract
    verdict: complete_disqualified_v783_contract
    addressed_by: Exp9194 failed cold replay on missing entropy_reference_protocol and mutable V782 authority.
      Exp9208 now requires explicit protocol ownership and immutable historical fixtures, with original assertions
      and real cold CLI checks; Exp9209 separately qualifies the unexecuted lossless corpus reader and Exp9210
      the full validation union. This task runs only after its new exact prerequisite gates pass.
    retire_if_same_verdict: true
  - experiment_id: exp7329-contract
    verdict: complete_disqualified_v644_contract_mismatched_or_invalid
    addressed_by: Retain the recorded prior failure. Exp9194 failed cold replay on missing entropy_reference_protocol
      and mutable V782 authority. Exp9208 now requires explicit protocol ownership and immutable historical
      fixtures, with original assertions and real cold CLI checks; Exp9209 separately qualifies the unexecuted
      lossless corpus reader and Exp9210 the full validation union. This task runs only after its new exact
      prerequisite gates pass.
    retire_if_same_verdict: true
  - experiment_id: exp7421-contract-ingestion
    verdict: complete_disqualified_stale_markdown_contract_with_method_ingestion
    addressed_by: Retain the recorded prior failure. Exp9194 failed cold replay on missing entropy_reference_protocol
      and mutable V782 authority. Exp9208 now requires explicit protocol ownership and immutable historical
      fixtures, with original assertions and real cold CLI checks; Exp9209 separately qualifies the unexecuted
      lossless corpus reader and Exp9210 the full validation union. This task runs only after its new exact
      prerequisite gates pass.
    retire_if_same_verdict: true
  prompt: |-
    CONTEXT:
    Work in {project_root} on {date}. V783 closed with two executed administrative tasks and twelve skips. Exp9194 full objects matched, but cold replay lacked entropy_reference_protocol and historical V782 consumers read mutable new authority. All four published readiness scores were zero. Preserve those failures. V784 qualifies protocol-owner replay and immutable historical inputs before the lossless corpus correction and complete validation. The original human panel remains exposed development; no independent generalization is claimed.
    EXISTING CODE TO READ FIRST:
    research-references.md (V784 planning scan); openspec/change-proposals/research-roadmap-v783-preserved-20261117.md; openspec/change-proposals/research-roadmap-v783-queue-preserved-20261117.yaml; CLAUDE.md; CODEX.md; ops/north-star.md; ops/e2e-test-plan.md; ops/exclusion_manifest.yaml; openspec/change-proposals/research-roadmap-vNEXT.md; openspec/capabilities/research-reporting/spec.md; scripts/experiment_template.py; python/carnot/reporting/primary_publication.py; python/carnot/reporting/current_work_receipt.py; python/carnot/reporting/v783_contract.py; python/carnot/reporting/v783_contract_evidence.py; python/carnot/reporting/v782_contract.py; python/carnot/reporting/v782_capstone.py; python/carnot/reporting/human_authority_9097.py; python/carnot/reporting/logit_methods_9126.py; python/carnot/reporting/online_methods_9132.py; results/experiment_9180_v782_contract.json; results/experiment_9181_v782_causal_validation.json; results/experiment_9193_v782_capstone.json; openspec/change-proposals/research-roadmap-v782-preserved-20261117.md; openspec/change-proposals/research-roadmap-v782-queue-preserved-20261117.yaml; results/experiment_9194_v783_contract.json; results/experiment_9207_v783_capstone.json; tests/python/test_v783_contract_9194.py; tests/python/test_v782_contract_9180.py; tests/python/test_v782_capstone_9193.py; openspec/change-proposals/v783-contract-inputs.json; research-roadmap-next.yaml
    TASK:
    Exp9208: Qualify protocol-owner replay and bind fourteen current tasks. Deliver results/experiment_9208_v784_contract.json. Implement scripts/experiments/experiment_9208_v784_contract.py using existing seams. MODEL_SPECS: []. Requested inference_substrate_class: no_model_load. No current LLM load or generation; current_llm_invocations=0, generated_tokens=0.
    CONCRETE STEPS:
    1. Emit a flushed progress line at each phase boundary and before and after every model load, generation, benchmark or subprocess that can take minutes. Print inside long loops and supervise blocking calls with flushed heartbeats every 60 seconds. Keep every silence gap below 600 seconds. Write any file over about 200 lines in several tool calls of at most 200 lines; send a progress message between calls. Never use one huge tool call.
    2. PRECONDITIONS: authenticate exact inputs, immutable producer hashes and named gate operands. Check CPU, memory, private free disk and required tools. Record missing_file, missing_field and observed zero separately in gate_check_summary. External absence is terminal blocked, never partial. Do not repin historical evidence to mutable checkout bytes.
    3. Before implementation use rg to inspect every caller. Confirm the REQ-REPORT-9208-V784 / SCENARIO-REPORT-9208-V784 in the research-reporting spec, extend the spec first, then add failing controls before code. Preserve original assertions, artifacts and quarantines. Freeze units, formulas, exclusions and the complete affected validation scope before outcomes.
    4. Authenticate the Exp9194 original owned_tests and direct_consumers logs plus its private-child cold_replay receipt. Reproduce missing entropy_reference_protocol and historical V782/new-queue failures in private roots. Trace every caller through v783_contract.binding, v782_contract.replay and historical contract/capstone readers. Preserve original function assertions and historical pins; no expected-failure, skip or weakened checker is permitted.
    5. Bind each inherited entropy and ambiguity protocol to its owning immutable configuration and hash, including in a fresh cold-replay process. The observed entropy KeyError comes from inherited replay using a newer FROZEN configuration without that field. Use existing explicit protocol/sidecar seams; never synthesize defaults, blindly merge configuration dictionaries or repin historical methods. Replay must reject missing, changed and freshly rehashed false protocol bytes.
    6. Provide private historical input roots for V782/V783 tests from original retained design, activated queue, scheduler log, producer snapshots and configurations. Adapt fixture/input construction only where necessary; preserve every original test assertion and mutation. Never run a historical reducer against the current task list. Keep production active-queue validation strict. Test successor-queue pollution and interleaved old/new replays in a fresh process. Do not mutate the real active queue or original results.
    7. Compare all fourteen complete ordered objects in the design and V784 queue, including exact rendering, model fields and gates. Accept only literal-brace encoding already performed by the conductor. Check ID availability across the archive, active queue, exclusions and canonical artifacts. Preserve V783 design and queue bytes and do not activate anything. Separate diagnostic matches from qualified readiness.
    8. Authenticate original Exp9097 human source and Exp9126/9132 method bytes via retained original references. Preserve 300 qids/900 siblings, TRAIN150/CAL50/TEST100 and lexical shards 0:100,100:200,200:300; keep online TRAIN100 initialization distinct. Publish current source, static and online readiness only after their actual custody and affected checks pass. Never require Exp9194 readiness to be one or relabel its zero gates.
    9. Freeze the original Brier, CES and label-ambiguity protocols. Add the target-identifiability rubric from the V784 design before prediction work. Inventory original 621 plus 53 added validation nodes and every affected V780–V784 obligation. No node, file category, mutation or quarantine is removed. Historical receipts identify obligations; only current executions qualify them.
    10. Require the entire original V783 owned suite, affected V782 contract/capstone suites and new regression controls to pass against isolated original authority. Private publish, refusal, tamper and cold replay must work from outside the checkout without PYTHONPATH. Missing protocol, changed owner pin and successor-queue contamination must still fail. Set replay_authority_ready_score=1 only on this complete corrected path; a matching task list or owned coverage alone is insufficient.
    11. Run owned unit tests and actual child-process coverage with 100 percent owned statements and zero exclusions. Preserve original assertions. Run scoped Ruff check/format, strict mypy and scripts/check_spec_coverage.py with explicit test paths. Run private real CLI success, refusal, tamper and cold replay outside the checkout without PYTHONPATH under E2E-018/021; learning/recovery changes also require E2E-020. Every affected failure disqualifies readiness; record unrelated repository health separately.
    12. Stop measurement by 3300 seconds using one monotonic deadline for descendants and cleanup. Reserve 1200 seconds for final verification and 300 for publication within the 4800-second cap. Preserve unfinished units and censor reasons. Recompute every claim from rows. Run scripts/adversarial_verify.py and scripts/verdict_row_consistency_lint.py --strict with heartbeats; publish through primary_publication only after actual checker receipts pass. Reconcile openspec/, _bmad/traceability.md, ops/status.md and ops/changelog.md. Do not activate a queue, change the exclusion manifest, publish externally or modify historical results.
    REQUIRED ARTIFACT FIELDS:
    - honest_verdict: complete_* for terminal outcomes; partial_* only for retryable owned work; principle: A terminal prefix must agree with its verdict class.
    - verdict_class: positive | circular_positive | null | blocked | disqualified | partial; principle: External absence, retirement or gate failure is blocked. Only unfinished owned work is partial.
    - gate_check_summary: check, upstream, path, sha256, artifact_field, op, expected, observed, passed; principle: Every blocked result names its failed operand; missing evidence differs from an observed zero.
    - preconditions_checked: resource, available and actual check receipt; principle: Unavailable resources cannot produce invented measurements.
    - rows: unit_id, qid/group, sibling, arm/condition, seed, absolute metrics, numerator, denominator, status, exclusion_reason, censor_reason; principle: All comparative headlines must be reproducible from their independent units.
    - sample_size_budget: intended, attempted, completed, failed, censored, effective_independent_groups; principle: Siblings and repeated retention do not multiply independent sample size.
    - verifier_is_oracle: boolean; principle: Reusing executable truth as the verifier requires circular_positive, not an independent verifier win.
    - claim_scope: exposure_scope, independent_generalization_score=0 and allowed claims; principle: Exposed development and fixtures cannot close the independent-verifier gaps.
    - MODEL_SPECS: actual requested list and invocation status; principle: An upstream model reference is not current model work.
    - inference_substrate_class: actual class and requested_inference_substrate_class; principle: Use blocked_no_run for refusal before loading; attempted loads remain attempted loads.
    - inference_substrate: operation-accurate work description; principle: Forward work, generation and cached CPU reduction have different evidence requirements.
    - model_invocation_counts: current_llm_invocations, loads attempted/completed/failed, forward calls, generated_tokens; principle: Counters must agree with actual operation-specific receipts.
    - duration_s: measured wall time; principle: No artificial sleeping or duration padding may satisfy a substrate floor.
    - random_seed: fixed before outcomes; principle: A frozen seed permits replay without favorable seed selection.
    - reproducibility_checksum: inputs and configuration digest; principle: Changed inputs cannot pass as the original experiment.
    - source_references: original producer paths, SHA256 and imported fields; principle: Historical evidence retains its original authority.
    - code_config_hashes: owned implementation, tests and configuration; principle: Validation qualifies the executed revision.
    - current_work_receipts: operation-specific records; principle: Historical receipts cannot certify current inference.
    - validation_receipts: commands, logs, exit codes and coverage references; principle: Self-reported passing checks do not establish readiness.
    - required_checks_passed: boolean; principle: Readiness requires every affected check.
    - acceptance_gates: fixed conditions, each with a principle annotation; principle: Outcome-dependent thresholds cannot manufacture success.
    - protocol_references: source, evaluator labels, static, online, diagnostic and three shard rosters; principle: Consumers share immutable inputs with exact names.
    - replay_authority_ready_score: integer 0/1; principle: Fresh-process protocol closure and all original affected assertions must qualify the repaired path.
    - contract_ready_score: integer 0/1; principle: Exact full-object agreement is administrative readiness, independent of scientific effects.
    - source_ready_score: integer 0/1; principle: Original authenticated roles and labels cannot be replaced with convenient current bytes.
    - static_methods_ready_score: integer 0/1; principle: Exp9126 five arms and both original bounds remain fixed.
    - online_methods_ready_score: integer 0/1; principle: Exp9132 six arms, recovery and seven bounds remain fixed.
    - historical_input_roots: source paths, original hashes and per-suite isolated roots; principle: Successor queues cannot silently change historical test authority.
    - protocol_owner_rows: protocol, owning configuration, original hash, sidecar and cold-replay outcome; principle: An inherited replay must read its own protocol authority.
    - full_task_objects: fourteen ordered complete objects; principle: A short aspirational task list is not an execution contract.
    - validation_scope_manifest: original and affected nodes with code/test hashes; principle: Corpus optimization grants no omitted-test credit.
    - label_ambiguity_protocol_reference: immutable path and SHA256; principle: Label sensitivity is fixed before predictions and never selects an update.
    Run command: cd {project_root} && JAX_PLATFORMS=cpu .venv/bin/python -u scripts/experiments/experiment_9208_v784_contract.py --date {date}
    Do NOT push. Do NOT modify scripts/research_conductor.py.
  agent_type: claude
  operator_override: '2026-05-29 operator directive (standing): versioned lineage continuation vs exp9194-contract;
    Exp9194 failed cold replay on missing entropy_reference_protocol and mutable V782 authority. Exp9208 now
    requires explicit protocol ownership and immutable historical fixtures, with original assertions and real
    cold CLI checks; Exp9209 separately qualifies the unexecuted lossless corpus reader and Exp9210 the full
    validation union. This task runs only after its new exact prerequisite gates pass.'
- id: exp9209-corpus-prefilter
  title: 'Exp9209: Qualify a lossless byte prefilter for the complete corpus walk'
  phase: 1
  track: infrastructure
  priority: high
  requires_gpu: false
  max_turns: 100
  estimated_wall_time_min: 75
  per_unit_rows: true
  milestone: 2026.11.784
  deliverable: results/experiment_9209_v784_corpus_prefilter.json
  inference_substrate_class: no_model_load
  MODEL_SPECS: []
  model: opus
  gated_on:
  - upstream: exp9208-contract
    artifact_field: contract_ready_score
    op: ==
    value: 1
  - upstream: exp9208-contract
    artifact_field: replay_authority_ready_score
    op: ==
    value: 1
  prior_failures:
  - experiment_id: exp9195-corpus-prefilter
    verdict: blocked_gate_check_failed
    addressed_by: Exp9194 failed cold replay on missing entropy_reference_protocol and mutable V782 authority.
      Exp9208 now requires explicit protocol ownership and immutable historical fixtures, with original assertions
      and real cold CLI checks; Exp9209 separately qualifies the unexecuted lossless corpus reader and Exp9210
      the full validation union. This task runs only after its new exact prerequisite gates pass.
    retire_if_same_verdict: true
  - experiment_id: exp9181-validation-loop-repair
    verdict: complete_disqualified_causal_validation
    addressed_by: Retain the recorded prior failure. Exp9194 failed cold replay on missing entropy_reference_protocol
      and mutable V782 authority. Exp9208 now requires explicit protocol ownership and immutable historical
      fixtures, with original assertions and real cold CLI checks; Exp9209 separately qualifies the unexecuted
      lossless corpus reader and Exp9210 the full validation union. This task runs only after its new exact
      prerequisite gates pass.
    retire_if_same_verdict: true
  - experiment_id: exp9167-causal-validation
    verdict: complete_disqualified_causal_validation
    addressed_by: Retain the recorded prior failure. Exp9194 failed cold replay on missing entropy_reference_protocol
      and mutable V782 authority. Exp9208 now requires explicit protocol ownership and immutable historical
      fixtures, with original assertions and real cold CLI checks; Exp9209 separately qualifies the unexecuted
      lossless corpus reader and Exp9210 the full validation union. This task runs only after its new exact
      prerequisite gates pass.
    retire_if_same_verdict: true
  - experiment_id: exp9153-measured-validation
    verdict: complete_blocked_validation_prerequisite
    addressed_by: Retain the recorded prior failure. Exp9194 failed cold replay on missing entropy_reference_protocol
      and mutable V782 authority. Exp9208 now requires explicit protocol ownership and immutable historical
      fixtures, with original assertions and real cold CLI checks; Exp9209 separately qualifies the unexecuted
      lossless corpus reader and Exp9210 the full validation union. This task runs only after its new exact
      prerequisite gates pass.
    retire_if_same_verdict: true
  - experiment_id: exp9125-receipt-qualification
    verdict: complete_disqualified_bounded_receipt_qualification
    addressed_by: Retain the recorded prior failure. Exp9194 failed cold replay on missing entropy_reference_protocol
      and mutable V782 authority. Exp9208 now requires explicit protocol ownership and immutable historical
      fixtures, with original assertions and real cold CLI checks; Exp9209 separately qualifies the unexecuted
      lossless corpus reader and Exp9210 the full validation union. This task runs only after its new exact
      prerequisite gates pass.
    retire_if_same_verdict: true
  - experiment_id: exp9111-forward-evidence
    verdict: complete_disqualified_owned_checks
    addressed_by: Retain the recorded prior failure. Exp9194 failed cold replay on missing entropy_reference_protocol
      and mutable V782 authority. Exp9208 now requires explicit protocol ownership and immutable historical
      fixtures, with original assertions and real cold CLI checks; Exp9209 separately qualifies the unexecuted
      lossless corpus reader and Exp9210 the full validation union. This task runs only after its new exact
      prerequisite gates pass.
    retire_if_same_verdict: true
  - experiment_id: exp9140-forward-qualification
    verdict: complete_blocked_infeasible_forward_schedule
    addressed_by: Retain the recorded prior failure. Exp9194 failed cold replay on missing entropy_reference_protocol
      and mutable V782 authority. Exp9208 now requires explicit protocol ownership and immutable historical
      fixtures, with original assertions and real cold CLI checks; Exp9209 separately qualifies the unexecuted
      lossless corpus reader and Exp9210 the full validation union. This task runs only after its new exact
      prerequisite gates pass.
    retire_if_same_verdict: true
  prompt: |-
    CONTEXT:
    Work in {project_root} on {date}. V783 closed with two executed administrative tasks and twelve skips. Exp9194 full objects matched, but cold replay lacked entropy_reference_protocol and historical V782 consumers read mutable new authority. All four published readiness scores were zero. Preserve those failures. V784 qualifies protocol-owner replay and immutable historical inputs before the lossless corpus correction and complete validation. The original human panel remains exposed development; no independent generalization is claimed.
    EXISTING CODE TO READ FIRST:
    research-references.md (V784 planning scan); openspec/change-proposals/research-roadmap-v783-preserved-20261117.md; openspec/change-proposals/research-roadmap-v783-queue-preserved-20261117.yaml; CLAUDE.md; CODEX.md; ops/north-star.md; ops/e2e-test-plan.md; ops/exclusion_manifest.yaml; openspec/change-proposals/research-roadmap-vNEXT.md; openspec/capabilities/research-reporting/spec.md; scripts/experiment_template.py; python/carnot/reporting/primary_publication.py; python/carnot/reporting/current_work_receipt.py; scripts/adversarial_verify.py (_is_arc_live_agent_no_llm, _inference_substrate_text); tests/python/test_adversarial_verify_arc_no_llm_floor_2026_07_30.py; python/carnot/reporting/causal_validation_9181.py; python/carnot/reporting/validation_diagnostics_9181.py; results/experiment_9181_v782_causal_validation.json; python/carnot/reporting/receipt_qualification_9125.py
    TASK:
    Exp9209: Qualify a lossless byte prefilter for the complete corpus walk. Deliver results/experiment_9209_v784_corpus_prefilter.json. Implement scripts/experiments/experiment_9209_v784_corpus_prefilter.py using existing seams. MODEL_SPECS: []. Requested inference_substrate_class: no_model_load. No current LLM load or generation; current_llm_invocations=0, generated_tokens=0.
    CONCRETE STEPS:
    1. Emit a flushed progress line at each phase boundary and before and after every model load, generation, benchmark or subprocess that can take minutes. Print inside long loops and supervise blocking calls with flushed heartbeats every 60 seconds. Keep every silence gap below 600 seconds. Write any file over about 200 lines in several tool calls of at most 200 lines; send a progress message between calls. Never use one huge tool call.
    2. PRECONDITIONS: authenticate exact inputs, immutable producer hashes and named gate operands. Check CPU, memory, private free disk and required tools. Record missing_file, missing_field and observed zero separately in gate_check_summary. External absence is terminal blocked, never partial. Do not repin historical evidence to mutable checkout bytes.
    3. Before implementation use rg to inspect every caller. Confirm the REQ-REPORT-9209-V784 / SCENARIO-REPORT-9209-V784 in the research-reporting spec, extend the spec first, then add failing controls before code. Preserve original assertions, artifacts and quarantines. Freeze units, formulas, exclusions and the complete affected validation scope before outcomes.
    4. Authenticate Exp9208 replay_authority_ready_score and contract_ready_score, both exactly 1. Exp9195 never executed the prefilter: its scheduler refusal recorded contract_ready_score=0. Current work must qualify the same still-unmeasured lossless corpus hypothesis after the distinct replay repair, without inventing a prior prefilter result.
    5. Authenticate Exp9181 direct_consumers stdout/stderr and node events. It exited -15 after1224.316 seconds with an outer_work_cutoff; the active node recursively decodes results/**/*.json. Treat the planning size census as advisory. Record a fresh complete path/size manifest and bounded read/decode/classifier timings on deterministic size strata before asserting the cost cause. Do not rerun the failed structural-cache experiment.
    6. Implement one conservative reader for this corpus-impact test: derive all five currently recognized ASCII sentinels from the unchanged classifier constants. Visit every original recursive path. Stream every file to EOF in bounded chunks with enough overlap to detect sentinels and the JSON Unicode escape marker across boundaries. If a sentinel or Unicode escape is present, use the original read_text/json.loads and classifier. Only absence of every sentinel and Unicode escape can skip decoding. Never use filename, directory, size, age, schema or verdict as an exclusion. Keep original malformed/non-dict behavior, checked>0 and offenders==[] assertions.
    7. Prove parity against the original decoder on escaped keys/values, wrapped declarations, suffix boundaries, duplicate keys, malformed UTF-8/JSON, non-dicts, Unicode-escaped sentinels, cross-chunk markers and nested decoys. Run mutation controls deleting each sentinel and the escape fallback. No pass/fail cache, production validator semantic change, monkeypatch of outcomes, test skip or removal of corpus files is allowed. Audit callers before sharing the reader with any other scan.
    8. Measure original versus prefilter on the same bounded, preselected corpus strata; retain every byte/path denominator and censoring. Then execute the complete original target test over the full manifest, plus all original V782 targeted controls and every consumer affected by the new reader. No full-corpus original-versus-new speed ratio from a censored original run. Verify stable candidate bytes and record concurrent file additions or changes; unstable input cannot qualify corpus completeness.
    9. Set corpus_scan_ready_score=1 only when complete path traversal, exact refusal/semantic parity, the full original target module and all affected checks pass within budget. Retain original V782 structural correction code and all unchanged mutation tests. This qualifies the changed corpus walker only; Exp9210 must still run the entire validation union. If the complete corpus does not finish, report the measured remaining work; no subset can pass.
    10. Run owned unit tests and actual child-process coverage with 100 percent owned statements and zero exclusions. Preserve original assertions. Run scoped Ruff check/format, strict mypy and scripts/check_spec_coverage.py with explicit test paths. Run private real CLI success, refusal, tamper and cold replay outside the checkout without PYTHONPATH under E2E-018/021; learning/recovery changes also require E2E-020. Every affected failure disqualifies readiness; record unrelated repository health separately.
    11. Stop measurement by 3300 seconds using one monotonic deadline for descendants and cleanup. Reserve 1200 seconds for final verification and 300 for publication within the 4800-second cap. Preserve unfinished units and censor reasons. Recompute every claim from rows. Run scripts/adversarial_verify.py and scripts/verdict_row_consistency_lint.py --strict with heartbeats; publish through primary_publication only after actual checker receipts pass. Reconcile openspec/, _bmad/traceability.md, ops/status.md and ops/changelog.md. Do not activate a queue, change the exclusion manifest, publish externally or modify historical results.
    REQUIRED ARTIFACT FIELDS:
    - honest_verdict: complete_* for terminal outcomes; partial_* only for retryable owned work; principle: A terminal prefix must agree with its verdict class.
    - verdict_class: positive | circular_positive | null | blocked | disqualified | partial; principle: External absence, retirement or gate failure is blocked. Only unfinished owned work is partial.
    - gate_check_summary: check, upstream, path, sha256, artifact_field, op, expected, observed, passed; principle: Every blocked result names its failed operand; missing evidence differs from an observed zero.
    - preconditions_checked: resource, available and actual check receipt; principle: Unavailable resources cannot produce invented measurements.
    - rows: unit_id, qid/group, sibling, arm/condition, seed, absolute metrics, numerator, denominator, status, exclusion_reason, censor_reason; principle: All comparative headlines must be reproducible from their independent units.
    - sample_size_budget: intended, attempted, completed, failed, censored, effective_independent_groups; principle: Siblings and repeated retention do not multiply independent sample size.
    - verifier_is_oracle: boolean; principle: Reusing executable truth as the verifier requires circular_positive, not an independent verifier win.
    - claim_scope: exposure_scope, independent_generalization_score=0 and allowed claims; principle: Exposed development and fixtures cannot close the independent-verifier gaps.
    - MODEL_SPECS: actual requested list and invocation status; principle: An upstream model reference is not current model work.
    - inference_substrate_class: actual class and requested_inference_substrate_class; principle: Use blocked_no_run for refusal before loading; attempted loads remain attempted loads.
    - inference_substrate: operation-accurate work description; principle: Forward work, generation and cached CPU reduction have different evidence requirements.
    - model_invocation_counts: current_llm_invocations, loads attempted/completed/failed, forward calls, generated_tokens; principle: Counters must agree with actual operation-specific receipts.
    - duration_s: measured wall time; principle: No artificial sleeping or duration padding may satisfy a substrate floor.
    - random_seed: fixed before outcomes; principle: A frozen seed permits replay without favorable seed selection.
    - reproducibility_checksum: inputs and configuration digest; principle: Changed inputs cannot pass as the original experiment.
    - source_references: original producer paths, SHA256 and imported fields; principle: Historical evidence retains its original authority.
    - code_config_hashes: owned implementation, tests and configuration; principle: Validation qualifies the executed revision.
    - current_work_receipts: operation-specific records; principle: Historical receipts cannot certify current inference.
    - validation_receipts: commands, logs, exit codes and coverage references; principle: Self-reported passing checks do not establish readiness.
    - required_checks_passed: boolean; principle: Readiness requires every affected check.
    - acceptance_gates: fixed conditions, each with a principle annotation; principle: Outcome-dependent thresholds cannot manufacture success.
    - protocol_references: source, evaluator labels, static, online, diagnostic and three shard rosters; principle: Consumers share immutable inputs with exact names.
    - corpus_scan_ready_score: integer 0/1; principle: Complete traversal, unchanged assertions and exact semantic parity are all necessary.
    - corpus_manifest_reference: ordered paths, sizes, population hashes and change observations; principle: Every original file remains in scope, including raw and nested results.
    - scan_rows: path, bytes_read, decode_reason, decode_status, original/candidate comparison, duration and censor_reason; principle: Parsing less must not mean checking fewer eligible artifacts.
    - candidate_counts: visited, decoded, safely_negative, malformed, changed and unfinished; principle: Every path has one accountable outcome.
    - parity_controls: each sentinel, escape fallback, chunk boundary and mutation; principle: A filter is useful only if false negatives are excluded.
    - cost_model_rows: same-input read/decode/classification clocks; principle: Size counts alone do not identify the slow operation.
    Run command: cd {project_root} && JAX_PLATFORMS=cpu .venv/bin/python -u scripts/experiments/experiment_9209_v784_corpus_prefilter.py --date {date}
    Do NOT push. Do NOT modify scripts/research_conductor.py.
  agent_type: claude
  operator_override: '2026-05-29 operator directive (standing): versioned lineage continuation vs exp9195-corpus-prefilter;
    Exp9194 failed cold replay on missing entropy_reference_protocol and mutable V782 authority. Exp9208 now
    requires explicit protocol ownership and immutable historical fixtures, with original assertions and real
    cold CLI checks; Exp9209 separately qualifies the unexecuted lossless corpus reader and Exp9210 the full
    validation union. This task runs only after its new exact prerequisite gates pass.'
- id: exp9210-forward-qualification
  title: 'Exp9210: Execute the complete causally corrected validation revision'
  phase: 1
  track: infrastructure
  priority: high
  requires_gpu: false
  max_turns: 100
  estimated_wall_time_min: 75
  per_unit_rows: true
  milestone: 2026.11.784
  deliverable: results/experiment_9210_v784_forward_qualification.json
  inference_substrate_class: no_model_load
  MODEL_SPECS: []
  model: opus
  gated_on:
  - upstream: exp9209-corpus-prefilter
    artifact_field: corpus_scan_ready_score
    op: ==
    value: 1
  prior_failures:
  - experiment_id: exp9196-forward-qualification
    verdict: complete_blocked_declared_gate
    addressed_by: V783 has a log-only scheduler skip and no scientific primary; the verdict here is Exp9207
      accounting. Exp9194 failed cold replay on missing entropy_reference_protocol and mutable V782 authority.
      Exp9208 now requires explicit protocol ownership and immutable historical fixtures, with original assertions
      and real cold CLI checks; Exp9209 separately qualifies the unexecuted lossless corpus reader and Exp9210
      the full validation union. This task runs only after its new exact prerequisite gates pass.
    retire_if_same_verdict: true
  - experiment_id: exp9182-forward-qualification
    verdict: blocked_gate_check_failed
    addressed_by: Retain the recorded prior failure. Exp9194 failed cold replay on missing entropy_reference_protocol
      and mutable V782 authority. Exp9208 now requires explicit protocol ownership and immutable historical
      fixtures, with original assertions and real cold CLI checks; Exp9209 separately qualifies the unexecuted
      lossless corpus reader and Exp9210 the full validation union. This task runs only after its new exact
      prerequisite gates pass.
    retire_if_same_verdict: true
  - experiment_id: exp9168-forward-qualification
    verdict: blocked_gate_check_failed
    addressed_by: Retain the recorded prior failure. Exp9194 failed cold replay on missing entropy_reference_protocol
      and mutable V782 authority. Exp9208 now requires explicit protocol ownership and immutable historical
      fixtures, with original assertions and real cold CLI checks; Exp9209 separately qualifies the unexecuted
      lossless corpus reader and Exp9210 the full validation union. This task runs only after its new exact
      prerequisite gates pass.
    retire_if_same_verdict: true
  - experiment_id: exp9154-forward-qualification
    verdict: blocked_gate_check_failed
    addressed_by: Retain the recorded prior failure. Exp9194 failed cold replay on missing entropy_reference_protocol
      and mutable V782 authority. Exp9208 now requires explicit protocol ownership and immutable historical
      fixtures, with original assertions and real cold CLI checks; Exp9209 separately qualifies the unexecuted
      lossless corpus reader and Exp9210 the full validation union. This task runs only after its new exact
      prerequisite gates pass.
    retire_if_same_verdict: true
  - experiment_id: exp9125-receipt-qualification
    verdict: complete_disqualified_bounded_receipt_qualification
    addressed_by: Retain the recorded prior failure. Exp9194 failed cold replay on missing entropy_reference_protocol
      and mutable V782 authority. Exp9208 now requires explicit protocol ownership and immutable historical
      fixtures, with original assertions and real cold CLI checks; Exp9209 separately qualifies the unexecuted
      lossless corpus reader and Exp9210 the full validation union. This task runs only after its new exact
      prerequisite gates pass.
    retire_if_same_verdict: true
  - experiment_id: exp9140-forward-qualification
    verdict: complete_blocked_infeasible_forward_schedule
    addressed_by: Retain the recorded prior failure. Exp9194 failed cold replay on missing entropy_reference_protocol
      and mutable V782 authority. Exp9208 now requires explicit protocol ownership and immutable historical
      fixtures, with original assertions and real cold CLI checks; Exp9209 separately qualifies the unexecuted
      lossless corpus reader and Exp9210 the full validation union. This task runs only after its new exact
      prerequisite gates pass.
    retire_if_same_verdict: true
  prompt: |-
    CONTEXT:
    Work in {project_root} on {date}. V783 closed with two executed administrative tasks and twelve skips. Exp9194 full objects matched, but cold replay lacked entropy_reference_protocol and historical V782 consumers read mutable new authority. All four published readiness scores were zero. Preserve those failures. V784 qualifies protocol-owner replay and immutable historical inputs before the lossless corpus correction and complete validation. The original human panel remains exposed development; no independent generalization is claimed.
    EXISTING CODE TO READ FIRST:
    research-references.md (V784 planning scan); openspec/change-proposals/research-roadmap-v783-preserved-20261117.md; openspec/change-proposals/research-roadmap-v783-queue-preserved-20261117.yaml; CLAUDE.md; CODEX.md; ops/north-star.md; ops/e2e-test-plan.md; ops/exclusion_manifest.yaml; openspec/change-proposals/research-roadmap-vNEXT.md; openspec/capabilities/research-reporting/spec.md; scripts/experiment_template.py; python/carnot/reporting/primary_publication.py; python/carnot/reporting/current_work_receipt.py; python/carnot/reporting/forward_qualification_9140.py; python/carnot/reporting/validation_schedule_9153.py; python/carnot/reporting/receipt_qualification_9125.py; tests/python/test_forward_qualification_9140.py; results/experiment_9181_v782_causal_validation.json
    TASK:
    Execute the complete causally corrected validation revision. Deliver results/experiment_9210_v784_forward_qualification.json. Implement scripts/experiments/experiment_9210_v784_forward_qualification.py using existing library seams. MODEL_SPECS: []. Requested inference_substrate_class: no_model_load.
    Do not load an LLM. Set current_llm_invocations=0 and generated_tokens=0. Small energy-head training is CPU work. Keep historical model provenance in source_references.
    CONCRETE STEPS:
    1. Emit a flushed progress line at each phase boundary. Emit before and after every model load, generation, benchmark and subprocess that can take minutes. Print inside long loops. Supervise blocking calls with flushed heartbeats every 60 seconds. Keep every silence gap below 600 seconds. Write any file over about 200 lines in several tool calls of at most 200 lines. Send a progress message between writing calls. Never use one huge tool call.
    2. PRECONDITIONS: Authenticate exact required paths, hashes and gate fields before dependent work. Check imports, CPU/memory limits and private free disk. Distinguish missing_file, missing_field and observed zero. Missing external inputs produce terminal blocked evidence with gate_check_summary. Never repin historical evidence to mutable checkout code.
    3. Before implementation inspect callers with rg. Confirm REQ-REPORT-9210-V784 and SCENARIO-REPORT-9210-V784. Extend specs first. Write failing controls before code. Freeze units, exclusions, formulas and validation scope before outcomes.
    4. Include the corrected protocol-owner replay and isolated historical input-root regressions from Exp9208 in the full obligation union. Execute the complete original 674 nodes plus all affected V780–V784 nodes exactly once for this revision. Past structural or corpus checks grant no current full-qualification credit.
    5. Authenticate Exp9209 corpus_scan_ready_score and exact source revision. Freeze the union of all 621 original nodes, 53 V779 additions, affected V780 tests and new obligations. Preserve assertion bodies and full command roles. Historical completions do not count as current completions.
    6. Execute the complete normal orchestration once with measured-cost accounting and the causal correction. Retain whole-module fixture lifetimes. Record setup/call/teardown, actual process costs, child exits and coverage after exit. Stop measurement at 3300 seconds. Unknown forecasts cannot establish completion.
    7. Count each obligation once in this invocation; consume authenticated same-invocation results in final verification instead of launching the suite twice. Include semantic/count controls, real CLI refusal/tamper/cold replay, lint, strict types and spec checks. Combine only current actual child coverage. Require 100 percent owned statements with zero exclusions.
    8. Emit forward_validation_ready_score=1 only for the full node/command union, exact revision, coverage and checker semantics unchanged by the optimization. Missing nodes, unresolved stalls or exhausted budget leave readiness zero. Save exact remaining work and causes. Do not lower watchdogs, force feasible=true, drop assertions or repin quarantined history.
    9. Authenticate Exp9209 immutable attempt logs, child coverage and algorithmic parity. Execute all affected V780, V781, V782 and V784 controls in addition to the original union. Do not treat a new task ID as satisfying the retired Exp9168 reopen condition. If the correction is unqualified, the structured gate must skip this task.
    10. The changed prerequisite is Exp9209 full-corpus reader qualification. Preserve original621+53 nodes and every affected V780-V784 obligation, including full recursive-corpus assertions. Run each obligation once under this current revision with original outcomes. Do not execute a historical cached pass, rename a zero prerequisite, or reclassify an unfinished full scan as unrelated health. Any incomplete obligation keeps forward_validation_ready_score=0.
    11. Preserve original V782 and V783 methods, assertions and failures. Exp9208 must qualify protocol-owner replay and historical input isolation; Exp9209 qualifies only the lossless corpus reader; Exp9210 must execute all original 674 and every affected obligation before model work. No retired upstream task is a dependency. Preserve the original 900 answers, scientific bounds and original protocol hashes.
    12. Run owned unit tests and real child-process coverage with 100 percent owned statements and zero exclusions. Preserve original assertions. Run scoped Ruff check/format, strict mypy and scripts/check_spec_coverage.py with explicit test paths. Execute private real CLI success, refusal, tamper and cold replay outside the checkout without PYTHONPATH, following E2E-018/021. Learning/recovery changes also require E2E-020. Separate unrelated repository health from affected failures. Affected failures disqualify readiness.
    13. Stop measurement by 3300 seconds. Use one monotonic deadline for all descendants, including cleanup; do not spend the reserve on another measurement attempt. Reserve 1200 seconds for final verification and 300 seconds for publication within the 4800-second cap. Count each obligation once; distinguish elapsed time from its watchdog ceiling. Save unfinished units and censor reasons. Recompute claims from rows. Run scripts/adversarial_verify.py and scripts/verdict_row_consistency_lint.py --strict with heartbeats. Publish through primary_publication only after actual checker receipts pass. Reconcile openspec/, _bmad/traceability.md, ops/status.md and ops/changelog.md. Preserve historical primaries, quarantines and exclusions. Do not activate a queue or publish externally.
    REQUIRED ARTIFACT FIELDS:
    - honest_verdict: complete_* for terminal outcomes; partial_* only for retryable owned work; principle: A terminal prefix must agree with its verdict class.
    - verdict_class: positive | circular_positive | null | blocked | disqualified | partial; principle: External absence, retirement or gate failure is blocked. Only unfinished owned work is partial.
    - gate_check_summary: check, upstream, path, sha256, artifact_field, op, expected, observed, passed; principle: Every blocked result names its failed operand; missing evidence differs from an observed zero.
    - preconditions_checked: resource, available and actual check receipt; principle: Unavailable resources cannot produce invented measurements.
    - rows: unit_id, qid/group, sibling, arm/condition, seed, absolute metrics, numerator, denominator, status, exclusion_reason, censor_reason; principle: All comparative headlines must be reproducible from their independent units.
    - sample_size_budget: intended, attempted, completed, failed, censored, effective_independent_groups; principle: Siblings and repeated retention do not multiply independent sample size.
    - verifier_is_oracle: boolean; principle: Reusing executable truth as the verifier requires circular_positive, not an independent verifier win.
    - claim_scope: exposure_scope, independent_generalization_score=0 and allowed claims; principle: Exposed development and fixtures cannot close the independent-verifier gaps.
    - MODEL_SPECS: actual requested list and invocation status; principle: An upstream model reference is not current model work.
    - inference_substrate_class: actual class and requested_inference_substrate_class; principle: Use blocked_no_run for refusal before loading; attempted loads remain attempted loads.
    - inference_substrate: operation-accurate work description; principle: Forward work, generation and cached CPU reduction have different evidence requirements.
    - model_invocation_counts: current_llm_invocations, loads attempted/completed/failed, forward calls, generated_tokens; principle: Counters must agree with actual operation-specific receipts.
    - duration_s: measured wall time; principle: No artificial sleeping or duration padding may satisfy a substrate floor.
    - random_seed: fixed before outcomes; principle: A frozen seed permits replay without favorable seed selection.
    - reproducibility_checksum: inputs and configuration digest; principle: Changed inputs cannot pass as the original experiment.
    - source_references: original producer paths, SHA256 and imported fields; principle: Historical evidence retains its original authority.
    - code_config_hashes: owned implementation, tests and configuration; principle: Validation qualifies the executed revision.
    - current_work_receipts: operation-specific records; principle: Historical receipts cannot certify current inference.
    - validation_receipts: commands, logs, exit codes and coverage references; principle: Self-reported passing checks do not establish readiness.
    - required_checks_passed: boolean; principle: Readiness requires every affected check.
    - acceptance_gates: fixed conditions, each with a principle annotation; principle: Outcome-dependent thresholds cannot manufacture success.
    - forward_validation_ready_score: integer 0/1; principle: Only complete normal-path execution authorizes current model evidence.
    - node_completion_rows: original/additional node IDs, phase exits and actual durations; principle: No duplicated or absent node can satisfy the full union.
    - qualified_revision_reference: exact code and configuration plus full validation receipts; principle: Downstream work uses only the revision actually tested.
    Run command: cd {project_root} && JAX_PLATFORMS=cpu .venv/bin/python -u scripts/experiments/experiment_9210_v784_forward_qualification.py --date {date}
    Do NOT push. Do NOT modify scripts/research_conductor.py.
  agent_type: claude
  operator_override: '2026-05-29 operator directive (standing): versioned lineage continuation vs exp9196-forward-qualification;
    Exp9194 failed cold replay on missing entropy_reference_protocol and mutable V782 authority. Exp9208 now
    requires explicit protocol ownership and immutable historical fixtures, with original assertions and real
    cold CLI checks; Exp9209 separately qualifies the unexecuted lossless corpus reader and Exp9210 the full
    validation union. This task runs only after its new exact prerequisite gates pass.'
- id: exp9211-label-ambiguity
  title: 'Exp9211: Qualify human target identifiability and exact label-ambiguity bounds'
  phase: 1
  track: research
  priority: high
  requires_gpu: false
  max_turns: 50
  estimated_wall_time_min: 45
  per_unit_rows: true
  milestone: 2026.11.784
  deliverable: results/experiment_9211_v784_label_ambiguity.json
  inference_substrate_class: no_model_load
  MODEL_SPECS: []
  prior_failures:
  - experiment_id: exp9197-label-ambiguity
    verdict: blocked_gate_check_failed
    addressed_by: Exp9197 was skipped because unrelated contract validation zeroed source_ready_score. Authenticate
      original Exp9097/9126/9132 source bytes directly with full custody checks, independent of contract or
      model readiness; add the v2 target-identifiability audit without inferred labels.
    retire_if_same_verdict: true
  - experiment_id: exp9190-decision-audit
    verdict: blocked_upstream_retired
    addressed_by: Retain the recorded prior failure. Exp9197 was skipped because unrelated contract validation
      zeroed source_ready_score. Authenticate original Exp9097/9126/9132 source bytes directly with full custody
      checks, independent of contract or model readiness; add the v2 target-identifiability audit without inferred
      labels.
    retire_if_same_verdict: true
  prompt: |-
    CONTEXT:
    Work in {project_root} on {date}. V783 closed with two executed administrative tasks and twelve skips. Exp9194 full objects matched, but cold replay lacked entropy_reference_protocol and historical V782 consumers read mutable new authority. All four published readiness scores were zero. Preserve those failures. V784 qualifies protocol-owner replay and immutable historical inputs before the lossless corpus correction and complete validation. The original human panel remains exposed development; no independent generalization is claimed.
    EXISTING CODE TO READ FIRST:
    research-references.md (V784 planning scan); openspec/change-proposals/research-roadmap-v783-preserved-20261117.md; openspec/change-proposals/research-roadmap-v783-queue-preserved-20261117.yaml; CLAUDE.md; CODEX.md; ops/north-star.md; ops/e2e-test-plan.md; ops/exclusion_manifest.yaml; openspec/change-proposals/research-roadmap-vNEXT.md; openspec/capabilities/research-reporting/spec.md; scripts/experiment_template.py; python/carnot/reporting/primary_publication.py; python/carnot/reporting/current_work_receipt.py; python/carnot/reporting/human_contrast_9141.py; python/carnot/reporting/human_authority_9097.py; python/carnot/reporting/logit_methods_9126.py; python/carnot/reporting/online_methods_9132.py; results/experiment_9141_v779_human_contrast_audit.json; results/experiment_9180_v782_contract.json
    TASK:
    Exp9211: Qualify human target identifiability and exact label-ambiguity bounds without model work. Deliver results/experiment_9211_v784_label_ambiguity.json. Implement scripts/experiments/experiment_9211_v784_label_ambiguity.py using existing seams. MODEL_SPECS: []. Requested inference_substrate_class: no_model_load. No current LLM load or generation; current_llm_invocations=0, generated_tokens=0.
    CONCRETE STEPS:
    1. Emit a flushed progress line at each phase boundary and before and after every model load, generation, benchmark or subprocess that can take minutes. Print inside long loops and supervise blocking calls with flushed heartbeats every 60 seconds. Keep every silence gap below 600 seconds. Write any file over about 200 lines in several tool calls of at most 200 lines; send a progress message between calls. Never use one huge tool call.
    2. PRECONDITIONS: authenticate exact inputs, immutable producer hashes and named gate operands. Check CPU, memory, private free disk and required tools. Record missing_file, missing_field and observed zero separately in gate_check_summary. External absence is terminal blocked, never partial. Do not repin historical evidence to mutable checkout bytes.
    3. Before implementation use rg to inspect every caller. Confirm the REQ-REPORT-9211-V784 / SCENARIO-REPORT-9211-V784 in the research-reporting spec, extend the spec first, then add failing controls before code. Preserve original assertions, artifacts and quarantines. Freeze units, formulas, exclusions and the complete affected validation scope before outcomes.
    4. Independently authenticate original Exp9097 A1/A2 source, immutable producer work and Exp9126/9132 role references through their existing retained custody chain. Do not read source_ready_score from Exp9208 or Exp9194 as authority. Validate every original hash and receipt; external source absence blocks this task. Preserve all 900 rows and A2=-1 as missing. Labels and annotator tags stay outside inference views.
    5. Audit target identifiability using the November 11 v2 taxonomy paper and the frozen rubric: original factual-error label; explicit human annotation of expressed uncertainty; same-generator equivalent-prompt re-elicitation. For each response/role record whether each target has authenticated source evidence. Missing fields are unavailable, not negative. Three different-generator siblings do not establish semantic illusion versus confabulation. Do not infer categories with keywords, train a judge, call an LLM or relabel A1. This is an evidence-availability measurement, not category accuracy.
    6. Freeze source_evidence_present / absent / ambiguous decisions using original field definitions and annotator instructions. Test that missing provenance, synthetic category labels and cross-generator sibling substitutions cannot qualify a target. Publish rubric, per-response evidence pointers and counts; zero identifiable SI/CF rows is a valid result. Keep target_identifiability_ready_score separate from label_ambiguity_ready_score.
    7. Build two immutable per-response admissible label sets. Observed-disagreement sensitivity: observed agreement fixes A1, observed disagreement permits0/1, missing A2 fixes A1 but is reported. Missing-annotation envelope: observed agreement fixes A1 and every disagreement or missing A2 permits0/1. These are conditional robustness scenarios, not ground-truth sets or assumptions about actual error rates.
    8. Implement the exact endpoints for d(y)=(p_control-y)^2-(p_treatment-y)^2=p_control^2-p_treatment^2-2*y*(p_control-p_treatment). Evaluate both admissible endpoints at each original unit and sum with frozen weights. One label variable is shared across repeated appearances of the same qid/sibling, especially retention across ten streams: first sum its weighted coefficients, then optimize endpoints. Never choose a different label per repeated observation or claim that pairwise bounds share one joint worst case.
    9. Use analytic, binary exhaustive small-panel and permutation/duplicate-retention controls to prove min/max equality, sign correctness, constant predictions, missing labels, exact weights and finite probabilities. Freeze the reducer and masks before any current model prediction. Current900-label masks are empirical source evidence; artificial probabilities are method fixtures only.
    10. Publish label_ambiguity_ready_score only after all controls and custody checks pass. Report exact eligible/missing counts by TRAIN/CAL/TEST and original online supports; no empirical verifier Brier or learning effect exists here. Exp9219 can consume the reducer and report bounds on genuine predictions independently of primary gates. No automatic source-based exclusion or direction selection.
    11. Run owned unit tests and actual child-process coverage with 100 percent owned statements and zero exclusions. Preserve original assertions. Run scoped Ruff check/format, strict mypy and scripts/check_spec_coverage.py with explicit test paths. Run private real CLI success, refusal, tamper and cold replay outside the checkout without PYTHONPATH under E2E-018/021; learning/recovery changes also require E2E-020. Every affected failure disqualifies readiness; record unrelated repository health separately.
    12. Stop measurement by 3300 seconds using one monotonic deadline for descendants and cleanup. Reserve 1200 seconds for final verification and 300 for publication within the 4800-second cap. Preserve unfinished units and censor reasons. Recompute every claim from rows. Run scripts/adversarial_verify.py and scripts/verdict_row_consistency_lint.py --strict with heartbeats; publish through primary_publication only after actual checker receipts pass. Reconcile openspec/, _bmad/traceability.md, ops/status.md and ops/changelog.md. Do not activate a queue, change the exclusion manifest, publish externally or modify historical results.
    REQUIRED ARTIFACT FIELDS:
    - honest_verdict: complete_* for terminal outcomes; partial_* only for retryable owned work; principle: A terminal prefix must agree with its verdict class.
    - verdict_class: positive | circular_positive | null | blocked | disqualified | partial; principle: External absence, retirement or gate failure is blocked. Only unfinished owned work is partial.
    - gate_check_summary: check, upstream, path, sha256, artifact_field, op, expected, observed, passed; principle: Every blocked result names its failed operand; missing evidence differs from an observed zero.
    - preconditions_checked: resource, available and actual check receipt; principle: Unavailable resources cannot produce invented measurements.
    - rows: unit_id, qid/group, sibling, arm/condition, seed, absolute metrics, numerator, denominator, status, exclusion_reason, censor_reason; principle: All comparative headlines must be reproducible from their independent units.
    - sample_size_budget: intended, attempted, completed, failed, censored, effective_independent_groups; principle: Siblings and repeated retention do not multiply independent sample size.
    - verifier_is_oracle: boolean; principle: Reusing executable truth as the verifier requires circular_positive, not an independent verifier win.
    - claim_scope: exposure_scope, independent_generalization_score=0 and allowed claims; principle: Exposed development and fixtures cannot close the independent-verifier gaps.
    - MODEL_SPECS: actual requested list and invocation status; principle: An upstream model reference is not current model work.
    - inference_substrate_class: actual class and requested_inference_substrate_class; principle: Use blocked_no_run for refusal before loading; attempted loads remain attempted loads.
    - inference_substrate: operation-accurate work description; principle: Forward work, generation and cached CPU reduction have different evidence requirements.
    - model_invocation_counts: current_llm_invocations, loads attempted/completed/failed, forward calls, generated_tokens; principle: Counters must agree with actual operation-specific receipts.
    - duration_s: measured wall time; principle: No artificial sleeping or duration padding may satisfy a substrate floor.
    - random_seed: fixed before outcomes; principle: A frozen seed permits replay without favorable seed selection.
    - reproducibility_checksum: inputs and configuration digest; principle: Changed inputs cannot pass as the original experiment.
    - source_references: original producer paths, SHA256 and imported fields; principle: Historical evidence retains its original authority.
    - code_config_hashes: owned implementation, tests and configuration; principle: Validation qualifies the executed revision.
    - current_work_receipts: operation-specific records; principle: Historical receipts cannot certify current inference.
    - validation_receipts: commands, logs, exit codes and coverage references; principle: Self-reported passing checks do not establish readiness.
    - required_checks_passed: boolean; principle: Readiness requires every affected check.
    - acceptance_gates: fixed conditions, each with a principle annotation; principle: Outcome-dependent thresholds cannot manufacture success.
    - protocol_references: source, evaluator labels, static, online, diagnostic and three shard rosters; principle: Consumers share immutable inputs with exact names.
    - target_identifiability_ready_score: integer 0/1; principle: A complete availability audit may honestly find no identifiable categories.
    - target_rubric_reference: immutable path, SHA256 and original criterion definitions; principle: Evidence categories must be frozen before model predictions.
    - target_identifiability_rows: qid, sibling, role, target, evidence path/hash, original criterion, status and reason; principle: Missing category authority cannot become a measured class.
    - label_ambiguity_ready_score: integer 0/1; principle: Custody and exact endpoint controls establish a method, not a useful verifier.
    - ambiguity_masks_reference: every qid/sibling, A1/A2 status and both admissible sets; principle: Missing annotations cannot be invented or silently dropped.
    - endpoint_control_rows: probabilities, admissible assignments, exact and reducer extrema; principle: Repeated measurements share one underlying label.
    - label_exposure_scope: exposed_development_only; principle: Reusing the human panel supplies no untouched confirmation.
    - scientific_effect_measured: false; principle: Fixture probabilities cannot establish a natural-verification benefit.
    Run command: cd {project_root} && JAX_PLATFORMS=cpu .venv/bin/python -u scripts/experiments/experiment_9211_v784_label_ambiguity.py --date {date}
    Do NOT push. Do NOT modify scripts/research_conductor.py.
  operator_override: '2026-05-29 operator directive (standing): versioned lineage continuation vs exp9197-label-ambiguity;
    Exp9197 was skipped because unrelated contract validation zeroed source_ready_score. Authenticate original
    Exp9097/9126/9132 source bytes directly with full custody checks, independent of contract or model readiness;
    add the v2 target-identifiability audit without inferred labels.'
  agent_type: claude
  model: sonnet
- id: exp9212-logit-runtime
  title: 'Exp9212: Qualify frozen Qwen forwards and all three shard budgets'
  phase: 2
  track: research
  priority: high
  requires_gpu: true
  max_turns: 50
  estimated_wall_time_min: 65
  per_unit_rows: true
  milestone: 2026.11.784
  deliverable: results/experiment_9212_v784_logit_runtime.json
  inference_substrate_class: model_load_no_generation
  MODEL_SPECS:
  - hf_id: unsloth/Qwen3.8-27B-GGUF
    quantization: Q4_K_M
  model: gpt-6.1-sol
  gated_on:
  - upstream: exp9210-forward-qualification
    artifact_field: forward_validation_ready_score
    op: ==
    value: 1
  - upstream: exp9208-contract
    artifact_field: static_methods_ready_score
    op: ==
    value: 1
  prior_failures:
  - experiment_id: exp9198-logit-runtime
    verdict: complete_blocked_declared_gate
    addressed_by: V783 has a log-only scheduler skip and no scientific primary; the verdict here is Exp9207
      accounting. Exp9194 failed cold replay on missing entropy_reference_protocol and mutable V782 authority.
      Exp9208 now requires explicit protocol ownership and immutable historical fixtures, with original assertions
      and real cold CLI checks; Exp9209 separately qualifies the unexecuted lossless corpus reader and Exp9210
      the full validation union. This task runs only after its new exact prerequisite gates pass.
    retire_if_same_verdict: true
  - experiment_id: exp9183-logit-runtime
    verdict: blocked_upstream_retired
    addressed_by: Retain the recorded prior failure. Exp9194 failed cold replay on missing entropy_reference_protocol
      and mutable V782 authority. Exp9208 now requires explicit protocol ownership and immutable historical
      fixtures, with original assertions and real cold CLI checks; Exp9209 separately qualifies the unexecuted
      lossless corpus reader and Exp9210 the full validation union. This task runs only after its new exact
      prerequisite gates pass.
    retire_if_same_verdict: true
  - experiment_id: exp9169-logit-runtime
    verdict: 'GATE_BLOCK: upstream retired; no canonical honest_verdict artifact'
    addressed_by: Retain the recorded prior failure. Exp9194 failed cold replay on missing entropy_reference_protocol
      and mutable V782 authority. Exp9208 now requires explicit protocol ownership and immutable historical
      fixtures, with original assertions and real cold CLI checks; Exp9209 separately qualifies the unexecuted
      lossless corpus reader and Exp9210 the full validation union. This task runs only after its new exact
      prerequisite gates pass.
    retire_if_same_verdict: true
  - experiment_id: exp9155-logit-runtime
    verdict: 'GATE_BLOCK: upstream retired; no canonical honest_verdict artifact'
    addressed_by: Retain the recorded prior failure. Exp9194 failed cold replay on missing entropy_reference_protocol
      and mutable V782 authority. Exp9208 now requires explicit protocol ownership and immutable historical
      fixtures, with original assertions and real cold CLI checks; Exp9209 separately qualifies the unexecuted
      lossless corpus reader and Exp9210 the full validation union. This task runs only after its new exact
      prerequisite gates pass.
    retire_if_same_verdict: true
  prompt: |-
    CONTEXT:
    Work in {project_root} on {date}. V783 closed with two executed administrative tasks and twelve skips. Exp9194 full objects matched, but cold replay lacked entropy_reference_protocol and historical V782 consumers read mutable new authority. All four published readiness scores were zero. Preserve those failures. V784 qualifies protocol-owner replay and immutable historical inputs before the lossless corpus correction and complete validation. The original human panel remains exposed development; no independent generalization is claimed.
    EXISTING CODE TO READ FIRST:
    research-references.md (V784 planning scan); openspec/change-proposals/research-roadmap-v783-preserved-20261117.md; openspec/change-proposals/research-roadmap-v783-queue-preserved-20261117.yaml; CLAUDE.md; CODEX.md; ops/north-star.md; ops/e2e-test-plan.md; ops/exclusion_manifest.yaml; openspec/change-proposals/research-roadmap-vNEXT.md; openspec/capabilities/research-reporting/spec.md; scripts/experiment_template.py; python/carnot/reporting/primary_publication.py; python/carnot/reporting/current_work_receipt.py; python/carnot/reporting/embedding_runtime_9098.py; python/carnot/reporting/logit_methods_9126.py; python/carnot/reporting/logit_methods_runner_9126.py; results/experiment_9098_v776_embedding_runtime.json; results/experiment_9182_v782_forward_qualification.json
    TASK:
    Qualify frozen Qwen forwards and all three shard budgets. Deliver results/experiment_9212_v784_logit_runtime.json. Implement scripts/experiments/experiment_9212_v784_logit_runtime.py using existing library seams. MODEL_SPECS: [{{"hf_id": "unsloth/Qwen3.8-27B-GGUF", "quantization": "Q4_K_M"}}]. Requested inference_substrate_class: model_load_no_generation.
    Use frozen Qwen teacher-forced forwards only; generated_tokens=0. model_load_no_generation has a 2-second floor. A small-token canary would require model_bounded_generation (10s); a real generative run requires model_full_generation (60s). Do not add either operation here.
    CONCRETE STEPS:
    1. Emit a flushed progress line at each phase boundary. Emit before and after every model load, generation, benchmark and subprocess that can take minutes. Print inside long loops. Supervise blocking calls with flushed heartbeats every 60 seconds. Keep every silence gap below 600 seconds. Write any file over about 200 lines in several tool calls of at most 200 lines. Send a progress message between writing calls. Never use one huge tool call.
    2. PRECONDITIONS: Authenticate exact required paths, hashes and gate fields before dependent work. Check imports, CPU/memory limits and private free disk. Distinguish missing_file, missing_field and observed zero. Missing external inputs produce terminal blocked evidence with gate_check_summary. Never repin historical evidence to mutable checkout code. Before model work, resolve cached_current_model(gpu_index=0), as documented by cached_sota_pair() in the experiment template. Require the mandated local GGUF, embedded tokenizer, GPU0 lease, free VRAM and actual CUDA offload support. Pin GGUF and llama.cpp build hashes. GPU1 remains with its owner. Cache or resource failure blocks; no download or small-model headline fallback.
    3. Before implementation inspect callers with rg. Confirm REQ-REPORT-9212-V784 and SCENARIO-REPORT-9212-V784. Extend specs first. Write failing controls before code. Freeze units, exclusions, formulas and validation scope before outcomes.
    4. Bind the Exp9126 frozen tokenization, prefix, caps256/768, full vocabulary and chunk128 protocol. Use only the supported serial forward API. Do not implement embeddings, batching masks or a different tokenizer. Freeze a six-text pilot selected by text-length quantiles from the label-blind roster before model access.
    5. Check logits for serial versus chunk128 parity with atol1e-5/rtol1e-4, identical token IDs and correct first-answer alignment. Compare sufficient statistics and raw vector hashes. Test one-token answers, context reset, adjacent chunk boundaries and nonfinite-output refusal. Classify pilot source exposure honestly; never retain pilot answers as new independent evidence.
    6. Measure actual load, forward, reduction, serialization, checkpoint and memory/disk costs. Project each of the three300-answer shards from its actual capped token lengths and worst observed per-token rate. Include load and checkpoint overhead. Require each full shard projection<=3300 seconds and available disk>=2GiB raw evidence plus measured storage needs. A watchdog ceiling is not a runtime observation. No row dropping or shorter cap rescues infeasibility.
    7. Validate operation-specific current receipts under Exp9210. Emit logit_runtime_ready_score and shard_feasible_score separately. Both require actual model work; all three shard projections must pass. Generated tokens remain0. Block before collection if any whole shard is infeasible; preserve the unchanged total cohort.
    8. Preserve original V782 and V783 methods, assertions and failures. Exp9208 must qualify protocol-owner replay and historical input isolation; Exp9209 qualifies only the lossless corpus reader; Exp9210 must execute all original 674 and every affected obligation before model work. No retired upstream task is a dependency. Preserve the original 900 answers, scientific bounds and original protocol hashes.
    9. Run owned unit tests and real child-process coverage with 100 percent owned statements and zero exclusions. Preserve original assertions. Run scoped Ruff check/format, strict mypy and scripts/check_spec_coverage.py with explicit test paths. Execute private real CLI success, refusal, tamper and cold replay outside the checkout without PYTHONPATH, following E2E-018/021. Learning/recovery changes also require E2E-020. Separate unrelated repository health from affected failures. Affected failures disqualify readiness.
    10. Stop measurement by 3300 seconds. Use one monotonic deadline for all descendants, including cleanup; do not spend the reserve on another measurement attempt. Reserve 1200 seconds for final verification and 300 seconds for publication within the 4800-second cap. Count each obligation once; distinguish elapsed time from its watchdog ceiling. Save unfinished units and censor reasons. Recompute claims from rows. Run scripts/adversarial_verify.py and scripts/verdict_row_consistency_lint.py --strict with heartbeats. Publish through primary_publication only after actual checker receipts pass. Reconcile openspec/, _bmad/traceability.md, ops/status.md and ops/changelog.md. Preserve historical primaries, quarantines and exclusions. Do not activate a queue or publish externally.
    REQUIRED ARTIFACT FIELDS:
    - honest_verdict: complete_* for terminal outcomes; partial_* only for retryable owned work; principle: A terminal prefix must agree with its verdict class.
    - verdict_class: positive | circular_positive | null | blocked | disqualified | partial; principle: External absence, retirement or gate failure is blocked. Only unfinished owned work is partial.
    - gate_check_summary: check, upstream, path, sha256, artifact_field, op, expected, observed, passed; principle: Every blocked result names its failed operand; missing evidence differs from an observed zero.
    - preconditions_checked: resource, available and actual check receipt; principle: Unavailable resources cannot produce invented measurements.
    - rows: unit_id, qid/group, sibling, arm/condition, seed, absolute metrics, numerator, denominator, status, exclusion_reason, censor_reason; principle: All comparative headlines must be reproducible from their independent units.
    - sample_size_budget: intended, attempted, completed, failed, censored, effective_independent_groups; principle: Siblings and repeated retention do not multiply independent sample size.
    - verifier_is_oracle: boolean; principle: Reusing executable truth as the verifier requires circular_positive, not an independent verifier win.
    - claim_scope: exposure_scope, independent_generalization_score=0 and allowed claims; principle: Exposed development and fixtures cannot close the independent-verifier gaps.
    - MODEL_SPECS: actual requested list and invocation status; principle: An upstream model reference is not current model work.
    - inference_substrate_class: actual class and requested_inference_substrate_class; principle: Use blocked_no_run for refusal before loading; attempted loads remain attempted loads.
    - inference_substrate: operation-accurate work description; principle: Forward work, generation and cached CPU reduction have different evidence requirements.
    - model_invocation_counts: current_llm_invocations, loads attempted/completed/failed, forward calls, generated_tokens; principle: Counters must agree with actual operation-specific receipts.
    - duration_s: measured wall time; principle: No artificial sleeping or duration padding may satisfy a substrate floor.
    - random_seed: fixed before outcomes; principle: A frozen seed permits replay without favorable seed selection.
    - reproducibility_checksum: inputs and configuration digest; principle: Changed inputs cannot pass as the original experiment.
    - source_references: original producer paths, SHA256 and imported fields; principle: Historical evidence retains its original authority.
    - code_config_hashes: owned implementation, tests and configuration; principle: Validation qualifies the executed revision.
    - current_work_receipts: operation-specific records; principle: Historical receipts cannot certify current inference.
    - validation_receipts: commands, logs, exit codes and coverage references; principle: Self-reported passing checks do not establish readiness.
    - required_checks_passed: boolean; principle: Readiness requires every affected check.
    - acceptance_gates: fixed conditions, each with a principle annotation; principle: Outcome-dependent thresholds cannot manufacture success.
    - logit_runtime_ready_score: integer 0/1; principle: Correct supported forwards and actual receipts precede collection.
    - shard_feasible_score: integer 0/1; principle: The complete planned populations must fit measured budgets.
    - runtime_reference: model/build/tokenizer hashes, prefix and role-blind pilot roster, full-vocabulary parity rows and actual counters; principle: A forward-only canary must be reproducible.
    - shard_budget_rows: load, tokens, worst observed rate, overhead, memory, disk and per-shard upper projection; principle: Feasibility includes the whole intended workload.
    Run command: cd {project_root} && JAX_PLATFORMS=cpu CARNOT_FORCE_LIVE=1 .venv/bin/python -u scripts/experiments/experiment_9212_v784_logit_runtime.py --date {date}
    Do NOT push. Do NOT modify scripts/research_conductor.py.
  agent_type: codex
  operator_override: '2026-05-29 operator directive (standing): versioned lineage continuation vs exp9198-logit-runtime;
    Exp9194 failed cold replay on missing entropy_reference_protocol and mutable V782 authority. Exp9208 now
    requires explicit protocol ownership and immutable historical fixtures, with original assertions and real
    cold CLI checks; Exp9209 separately qualifies the unexecuted lossless corpus reader and Exp9210 the full
    validation union. This task runs only after its new exact prerequisite gates pass.'
- id: exp9213-logits-a
  title: 'Exp9213: Collect complete frozen Qwen logit shard A'
  phase: 2
  track: research
  priority: high
  requires_gpu: true
  max_turns: 50
  estimated_wall_time_min: 65
  per_unit_rows: true
  milestone: 2026.11.784
  deliverable: results/experiment_9213_v784_logits_a.json
  inference_substrate_class: model_load_no_generation
  MODEL_SPECS:
  - hf_id: unsloth/Qwen3.8-27B-GGUF
    quantization: Q4_K_M
  gated_on:
  - upstream: exp9208-contract
    artifact_field: source_ready_score
    op: ==
    value: 1
  - upstream: exp9212-logit-runtime
    artifact_field: logit_runtime_ready_score
    op: ==
    value: 1
  - upstream: exp9212-logit-runtime
    artifact_field: shard_feasible_score
    op: ==
    value: 1
  prior_failures:
  - experiment_id: exp9199-logits-a
    verdict: complete_blocked_declared_gate
    addressed_by: V783 has a log-only scheduler skip and no scientific primary; the verdict here is Exp9207
      accounting. Exp9194 failed cold replay on missing entropy_reference_protocol and mutable V782 authority.
      Exp9208 now requires explicit protocol ownership and immutable historical fixtures, with original assertions
      and real cold CLI checks; Exp9209 separately qualifies the unexecuted lossless corpus reader and Exp9210
      the full validation union. This task runs only after its new exact prerequisite gates pass.
    retire_if_same_verdict: true
  - experiment_id: exp9184-logits-a
    verdict: blocked_upstream_retired
    addressed_by: Retain the recorded prior failure. Exp9194 failed cold replay on missing entropy_reference_protocol
      and mutable V782 authority. Exp9208 now requires explicit protocol ownership and immutable historical
      fixtures, with original assertions and real cold CLI checks; Exp9209 separately qualifies the unexecuted
      lossless corpus reader and Exp9210 the full validation union. This task runs only after its new exact
      prerequisite gates pass.
    retire_if_same_verdict: true
  - experiment_id: exp9170-logits-a
    verdict: 'GATE_BLOCK: upstream retired; no canonical honest_verdict artifact'
    addressed_by: Retain the recorded prior failure. Exp9194 failed cold replay on missing entropy_reference_protocol
      and mutable V782 authority. Exp9208 now requires explicit protocol ownership and immutable historical
      fixtures, with original assertions and real cold CLI checks; Exp9209 separately qualifies the unexecuted
      lossless corpus reader and Exp9210 the full validation union. This task runs only after its new exact
      prerequisite gates pass.
    retire_if_same_verdict: true
  - experiment_id: exp9156-logits-a
    verdict: 'GATE_BLOCK: upstream retired; no canonical honest_verdict artifact'
    addressed_by: Retain the recorded prior failure. Exp9194 failed cold replay on missing entropy_reference_protocol
      and mutable V782 authority. Exp9208 now requires explicit protocol ownership and immutable historical
      fixtures, with original assertions and real cold CLI checks; Exp9209 separately qualifies the unexecuted
      lossless corpus reader and Exp9210 the full validation union. This task runs only after its new exact
      prerequisite gates pass.
    retire_if_same_verdict: true
  prompt: |-
    CONTEXT:
    Work in {project_root} on {date}. V783 closed with two executed administrative tasks and twelve skips. Exp9194 full objects matched, but cold replay lacked entropy_reference_protocol and historical V782 consumers read mutable new authority. All four published readiness scores were zero. Preserve those failures. V784 qualifies protocol-owner replay and immutable historical inputs before the lossless corpus correction and complete validation. The original human panel remains exposed development; no independent generalization is claimed.
    EXISTING CODE TO READ FIRST:
    research-references.md (V784 planning scan); openspec/change-proposals/research-roadmap-v783-preserved-20261117.md; openspec/change-proposals/research-roadmap-v783-queue-preserved-20261117.yaml; CLAUDE.md; CODEX.md; ops/north-star.md; ops/e2e-test-plan.md; ops/exclusion_manifest.yaml; openspec/change-proposals/research-roadmap-vNEXT.md; openspec/capabilities/research-reporting/spec.md; scripts/experiment_template.py; python/carnot/reporting/primary_publication.py; python/carnot/reporting/current_work_receipt.py; python/carnot/reporting/logit_methods_9126.py; python/carnot/reporting/logit_methods_runner_9126.py; python/carnot/reporting/embedding_runtime_9098.py; results/experiment_9180_v782_contract.json; results/experiment_9183_v782_logit_runtime.json
    TASK:
    Collect complete frozen Qwen logit shard A. Deliver results/experiment_9213_v784_logits_a.json. Implement scripts/experiments/experiment_9213_v784_logits_a.py using existing library seams. MODEL_SPECS: [{{"hf_id": "unsloth/Qwen3.8-27B-GGUF", "quantization": "Q4_K_M"}}]. Requested inference_substrate_class: model_load_no_generation.
    Use frozen Qwen teacher-forced forwards only; generated_tokens=0. model_load_no_generation has a 2-second floor. A small-token canary would require model_bounded_generation (10s); a real generative run requires model_full_generation (60s). Do not add either operation here.
    CONCRETE STEPS:
    1. Emit a flushed progress line at each phase boundary. Emit before and after every model load, generation, benchmark and subprocess that can take minutes. Print inside long loops. Supervise blocking calls with flushed heartbeats every 60 seconds. Keep every silence gap below 600 seconds. Write any file over about 200 lines in several tool calls of at most 200 lines. Send a progress message between writing calls. Never use one huge tool call.
    2. PRECONDITIONS: Authenticate exact required paths, hashes and gate fields before dependent work. Check imports, CPU/memory limits and private free disk. Distinguish missing_file, missing_field and observed zero. Missing external inputs produce terminal blocked evidence with gate_check_summary. Never repin historical evidence to mutable checkout code. Before model work, resolve cached_current_model(gpu_index=0), as documented by cached_sota_pair() in the experiment template. Require the mandated local GGUF, embedded tokenizer, GPU0 lease, free VRAM and actual CUDA offload support. Pin GGUF and llama.cpp build hashes. GPU1 remains with its owner. Cache or resource failure blocks; no download or small-model headline fallback.
    3. Before implementation inspect callers with rg. Confirm REQ-REPORT-9213-V784 and SCENARIO-REPORT-9213-V784. Extend specs first. Write failing controls before code. Freeze units, exclusions, formulas and validation scope before outcomes.
    4. Authenticate shard A from Exp9208: lexical qid indices0:100,100 qids/all three siblings,300 answers. Preserve original roles and source identities. Use question and answer text only. Forbid labels, references, generator tags and annotation metadata as model features.
    5. Run the frozen Exp9126 forward/statistic protocol. Keep question cap256, answer cap768, prefix bytes, no BOS/chat template and chunk128. Stream float64 sufficient statistics. Retain designated raw full-vocabulary vectors for the original global first20 lexical qids/all siblings/first8 positions. Keep per-token IDs, offsets, selected logits, logsumexp, entropy and hashes.
    6. Checkpoint after each qid. Emit progress within every answer/chunk loop. Resume only matching hashes and unfinished units. Keep failed and empty answers in their intended slots with reasons. Do not rerun completed rows selectively or replace difficult answers. Preserve cross-chunk adjacent pairs and reset context between answers.
    7. Independently recompute sampled raw statistics. Require all300 intended records, valid nonempty forwards, exact model custody and all required fields. Empty source answers retain typed escalation slots under the frozen policy. Model failure prevents complete-shard readiness. Keep evaluator labels separate. Reuse the same collector for all three shards; do not duplicate implementation.
    8. Preserve original V782 and V783 methods, assertions and failures. Exp9208 must qualify protocol-owner replay and historical input isolation; Exp9209 qualifies only the lossless corpus reader; Exp9210 must execute all original 674 and every affected obligation before model work. No retired upstream task is a dependency. Preserve the original 900 answers, scientific bounds and original protocol hashes.
    9. Run owned unit tests and real child-process coverage with 100 percent owned statements and zero exclusions. Preserve original assertions. Run scoped Ruff check/format, strict mypy and scripts/check_spec_coverage.py with explicit test paths. Execute private real CLI success, refusal, tamper and cold replay outside the checkout without PYTHONPATH, following E2E-018/021. Learning/recovery changes also require E2E-020. Separate unrelated repository health from affected failures. Affected failures disqualify readiness.
    10. Stop measurement by 3300 seconds. Use one monotonic deadline for all descendants, including cleanup; do not spend the reserve on another measurement attempt. Reserve 1200 seconds for final verification and 300 seconds for publication within the 4800-second cap. Count each obligation once; distinguish elapsed time from its watchdog ceiling. Save unfinished units and censor reasons. Recompute claims from rows. Run scripts/adversarial_verify.py and scripts/verdict_row_consistency_lint.py --strict with heartbeats. Publish through primary_publication only after actual checker receipts pass. Reconcile openspec/, _bmad/traceability.md, ops/status.md and ops/changelog.md. Preserve historical primaries, quarantines and exclusions. Do not activate a queue or publish externally.
    REQUIRED ARTIFACT FIELDS:
    - honest_verdict: complete_* for terminal outcomes; partial_* only for retryable owned work; principle: A terminal prefix must agree with its verdict class.
    - verdict_class: positive | circular_positive | null | blocked | disqualified | partial; principle: External absence, retirement or gate failure is blocked. Only unfinished owned work is partial.
    - gate_check_summary: check, upstream, path, sha256, artifact_field, op, expected, observed, passed; principle: Every blocked result names its failed operand; missing evidence differs from an observed zero.
    - preconditions_checked: resource, available and actual check receipt; principle: Unavailable resources cannot produce invented measurements.
    - rows: unit_id, qid/group, sibling, arm/condition, seed, absolute metrics, numerator, denominator, status, exclusion_reason, censor_reason; principle: All comparative headlines must be reproducible from their independent units.
    - sample_size_budget: intended, attempted, completed, failed, censored, effective_independent_groups; principle: Siblings and repeated retention do not multiply independent sample size.
    - verifier_is_oracle: boolean; principle: Reusing executable truth as the verifier requires circular_positive, not an independent verifier win.
    - claim_scope: exposure_scope, independent_generalization_score=0 and allowed claims; principle: Exposed development and fixtures cannot close the independent-verifier gaps.
    - MODEL_SPECS: actual requested list and invocation status; principle: An upstream model reference is not current model work.
    - inference_substrate_class: actual class and requested_inference_substrate_class; principle: Use blocked_no_run for refusal before loading; attempted loads remain attempted loads.
    - inference_substrate: operation-accurate work description; principle: Forward work, generation and cached CPU reduction have different evidence requirements.
    - model_invocation_counts: current_llm_invocations, loads attempted/completed/failed, forward calls, generated_tokens; principle: Counters must agree with actual operation-specific receipts.
    - duration_s: measured wall time; principle: No artificial sleeping or duration padding may satisfy a substrate floor.
    - random_seed: fixed before outcomes; principle: A frozen seed permits replay without favorable seed selection.
    - reproducibility_checksum: inputs and configuration digest; principle: Changed inputs cannot pass as the original experiment.
    - source_references: original producer paths, SHA256 and imported fields; principle: Historical evidence retains its original authority.
    - code_config_hashes: owned implementation, tests and configuration; principle: Validation qualifies the executed revision.
    - current_work_receipts: operation-specific records; principle: Historical receipts cannot certify current inference.
    - validation_receipts: commands, logs, exit codes and coverage references; principle: Self-reported passing checks do not establish readiness.
    - required_checks_passed: boolean; principle: Readiness requires every affected check.
    - acceptance_gates: fixed conditions, each with a principle annotation; principle: Outcome-dependent thresholds cannot manufacture success.
    - logits_a_ready_score: integer 0/1; principle: An incomplete shard cannot authorize a smaller favorable corpus.
    - feature_shard_reference: token/statistic rows, model receipts, role and roster hashes; principle: Every feature traces to actual frozen Qwen work.
    - raw_vector_reference: selected full-vocabulary arrays with positions and hashes; principle: Aggregate energy statistics remain independently checkable.
    - checkpoint_reference: completed IDs and exact resume ledger; principle: Resume cannot silently duplicate or replace evidence.
    Run command: cd {project_root} && JAX_PLATFORMS=cpu CARNOT_FORCE_LIVE=1 .venv/bin/python -u scripts/experiments/experiment_9213_v784_logits_a.py --date {date}
    Do NOT push. Do NOT modify scripts/research_conductor.py.
  agent_type: codex
  model: gpt-6.1-sol
  operator_override: '2026-05-29 operator directive (standing): versioned lineage continuation vs exp9199-logits-a;
    Exp9194 failed cold replay on missing entropy_reference_protocol and mutable V782 authority. Exp9208 now
    requires explicit protocol ownership and immutable historical fixtures, with original assertions and real
    cold CLI checks; Exp9209 separately qualifies the unexecuted lossless corpus reader and Exp9210 the full
    validation union. This task runs only after its new exact prerequisite gates pass.'
- id: exp9214-logits-b
  title: 'Exp9214: Collect complete frozen Qwen logit shard B'
  phase: 2
  track: research
  priority: high
  requires_gpu: true
  max_turns: 50
  estimated_wall_time_min: 65
  per_unit_rows: true
  milestone: 2026.11.784
  deliverable: results/experiment_9214_v784_logits_b.json
  inference_substrate_class: model_load_no_generation
  MODEL_SPECS:
  - hf_id: unsloth/Qwen3.8-27B-GGUF
    quantization: Q4_K_M
  gated_on:
  - upstream: exp9208-contract
    artifact_field: source_ready_score
    op: ==
    value: 1
  - upstream: exp9212-logit-runtime
    artifact_field: logit_runtime_ready_score
    op: ==
    value: 1
  - upstream: exp9212-logit-runtime
    artifact_field: shard_feasible_score
    op: ==
    value: 1
  prior_failures:
  - experiment_id: exp9200-logits-b
    verdict: complete_blocked_declared_gate
    addressed_by: V783 has a log-only scheduler skip and no scientific primary; the verdict here is Exp9207
      accounting. Exp9194 failed cold replay on missing entropy_reference_protocol and mutable V782 authority.
      Exp9208 now requires explicit protocol ownership and immutable historical fixtures, with original assertions
      and real cold CLI checks; Exp9209 separately qualifies the unexecuted lossless corpus reader and Exp9210
      the full validation union. This task runs only after its new exact prerequisite gates pass.
    retire_if_same_verdict: true
  - experiment_id: exp9185-logits-b
    verdict: blocked_upstream_retired
    addressed_by: Retain the recorded prior failure. Exp9194 failed cold replay on missing entropy_reference_protocol
      and mutable V782 authority. Exp9208 now requires explicit protocol ownership and immutable historical
      fixtures, with original assertions and real cold CLI checks; Exp9209 separately qualifies the unexecuted
      lossless corpus reader and Exp9210 the full validation union. This task runs only after its new exact
      prerequisite gates pass.
    retire_if_same_verdict: true
  - experiment_id: exp9171-logits-b
    verdict: 'GATE_BLOCK: upstream retired; no canonical honest_verdict artifact'
    addressed_by: Retain the recorded prior failure. Exp9194 failed cold replay on missing entropy_reference_protocol
      and mutable V782 authority. Exp9208 now requires explicit protocol ownership and immutable historical
      fixtures, with original assertions and real cold CLI checks; Exp9209 separately qualifies the unexecuted
      lossless corpus reader and Exp9210 the full validation union. This task runs only after its new exact
      prerequisite gates pass.
    retire_if_same_verdict: true
  - experiment_id: exp9157-logits-b
    verdict: 'GATE_BLOCK: upstream retired; no canonical honest_verdict artifact'
    addressed_by: Retain the recorded prior failure. Exp9194 failed cold replay on missing entropy_reference_protocol
      and mutable V782 authority. Exp9208 now requires explicit protocol ownership and immutable historical
      fixtures, with original assertions and real cold CLI checks; Exp9209 separately qualifies the unexecuted
      lossless corpus reader and Exp9210 the full validation union. This task runs only after its new exact
      prerequisite gates pass.
    retire_if_same_verdict: true
  prompt: |-
    CONTEXT:
    Work in {project_root} on {date}. V783 closed with two executed administrative tasks and twelve skips. Exp9194 full objects matched, but cold replay lacked entropy_reference_protocol and historical V782 consumers read mutable new authority. All four published readiness scores were zero. Preserve those failures. V784 qualifies protocol-owner replay and immutable historical inputs before the lossless corpus correction and complete validation. The original human panel remains exposed development; no independent generalization is claimed.
    EXISTING CODE TO READ FIRST:
    research-references.md (V784 planning scan); openspec/change-proposals/research-roadmap-v783-preserved-20261117.md; openspec/change-proposals/research-roadmap-v783-queue-preserved-20261117.yaml; CLAUDE.md; CODEX.md; ops/north-star.md; ops/e2e-test-plan.md; ops/exclusion_manifest.yaml; openspec/change-proposals/research-roadmap-vNEXT.md; openspec/capabilities/research-reporting/spec.md; scripts/experiment_template.py; python/carnot/reporting/primary_publication.py; python/carnot/reporting/current_work_receipt.py; python/carnot/reporting/logit_methods_9126.py; python/carnot/reporting/logit_methods_runner_9126.py; python/carnot/reporting/embedding_runtime_9098.py; results/experiment_9180_v782_contract.json; results/experiment_9183_v782_logit_runtime.json
    TASK:
    Collect complete frozen Qwen logit shard B. Deliver results/experiment_9214_v784_logits_b.json. Implement scripts/experiments/experiment_9214_v784_logits_b.py using existing library seams. MODEL_SPECS: [{{"hf_id": "unsloth/Qwen3.8-27B-GGUF", "quantization": "Q4_K_M"}}]. Requested inference_substrate_class: model_load_no_generation.
    Use frozen Qwen teacher-forced forwards only; generated_tokens=0. model_load_no_generation has a 2-second floor. A small-token canary would require model_bounded_generation (10s); a real generative run requires model_full_generation (60s). Do not add either operation here.
    CONCRETE STEPS:
    1. Emit a flushed progress line at each phase boundary. Emit before and after every model load, generation, benchmark and subprocess that can take minutes. Print inside long loops. Supervise blocking calls with flushed heartbeats every 60 seconds. Keep every silence gap below 600 seconds. Write any file over about 200 lines in several tool calls of at most 200 lines. Send a progress message between writing calls. Never use one huge tool call.
    2. PRECONDITIONS: Authenticate exact required paths, hashes and gate fields before dependent work. Check imports, CPU/memory limits and private free disk. Distinguish missing_file, missing_field and observed zero. Missing external inputs produce terminal blocked evidence with gate_check_summary. Never repin historical evidence to mutable checkout code. Before model work, resolve cached_current_model(gpu_index=0), as documented by cached_sota_pair() in the experiment template. Require the mandated local GGUF, embedded tokenizer, GPU0 lease, free VRAM and actual CUDA offload support. Pin GGUF and llama.cpp build hashes. GPU1 remains with its owner. Cache or resource failure blocks; no download or small-model headline fallback.
    3. Before implementation inspect callers with rg. Confirm REQ-REPORT-9214-V784 and SCENARIO-REPORT-9214-V784. Extend specs first. Write failing controls before code. Freeze units, exclusions, formulas and validation scope before outcomes.
    4. Authenticate shard B from Exp9208: lexical qid indices100:200,100 qids/all three siblings,300 answers. Preserve original roles and source identities. Use question and answer text only. Forbid labels, references, generator tags and annotation metadata as model features.
    5. Run the frozen Exp9126 forward/statistic protocol. Keep question cap256, answer cap768, prefix bytes, no BOS/chat template and chunk128. Stream float64 sufficient statistics. Retain designated raw full-vocabulary vectors for the original global first20 lexical qids/all siblings/first8 positions. Keep per-token IDs, offsets, selected logits, logsumexp, entropy and hashes.
    6. Checkpoint after each qid. Emit progress within every answer/chunk loop. Resume only matching hashes and unfinished units. Keep failed and empty answers in their intended slots with reasons. Do not rerun completed rows selectively or replace difficult answers. Preserve cross-chunk adjacent pairs and reset context between answers.
    7. Independently recompute sampled raw statistics. Require all300 intended records, valid nonempty forwards, exact model custody and all required fields. Empty source answers retain typed escalation slots under the frozen policy. Model failure prevents complete-shard readiness. Keep evaluator labels separate. Reuse the same collector for all three shards; do not duplicate implementation.
    8. Preserve original V782 and V783 methods, assertions and failures. Exp9208 must qualify protocol-owner replay and historical input isolation; Exp9209 qualifies only the lossless corpus reader; Exp9210 must execute all original 674 and every affected obligation before model work. No retired upstream task is a dependency. Preserve the original 900 answers, scientific bounds and original protocol hashes.
    9. Run owned unit tests and real child-process coverage with 100 percent owned statements and zero exclusions. Preserve original assertions. Run scoped Ruff check/format, strict mypy and scripts/check_spec_coverage.py with explicit test paths. Execute private real CLI success, refusal, tamper and cold replay outside the checkout without PYTHONPATH, following E2E-018/021. Learning/recovery changes also require E2E-020. Separate unrelated repository health from affected failures. Affected failures disqualify readiness.
    10. Stop measurement by 3300 seconds. Use one monotonic deadline for all descendants, including cleanup; do not spend the reserve on another measurement attempt. Reserve 1200 seconds for final verification and 300 seconds for publication within the 4800-second cap. Count each obligation once; distinguish elapsed time from its watchdog ceiling. Save unfinished units and censor reasons. Recompute claims from rows. Run scripts/adversarial_verify.py and scripts/verdict_row_consistency_lint.py --strict with heartbeats. Publish through primary_publication only after actual checker receipts pass. Reconcile openspec/, _bmad/traceability.md, ops/status.md and ops/changelog.md. Preserve historical primaries, quarantines and exclusions. Do not activate a queue or publish externally.
    REQUIRED ARTIFACT FIELDS:
    - honest_verdict: complete_* for terminal outcomes; partial_* only for retryable owned work; principle: A terminal prefix must agree with its verdict class.
    - verdict_class: positive | circular_positive | null | blocked | disqualified | partial; principle: External absence, retirement or gate failure is blocked. Only unfinished owned work is partial.
    - gate_check_summary: check, upstream, path, sha256, artifact_field, op, expected, observed, passed; principle: Every blocked result names its failed operand; missing evidence differs from an observed zero.
    - preconditions_checked: resource, available and actual check receipt; principle: Unavailable resources cannot produce invented measurements.
    - rows: unit_id, qid/group, sibling, arm/condition, seed, absolute metrics, numerator, denominator, status, exclusion_reason, censor_reason; principle: All comparative headlines must be reproducible from their independent units.
    - sample_size_budget: intended, attempted, completed, failed, censored, effective_independent_groups; principle: Siblings and repeated retention do not multiply independent sample size.
    - verifier_is_oracle: boolean; principle: Reusing executable truth as the verifier requires circular_positive, not an independent verifier win.
    - claim_scope: exposure_scope, independent_generalization_score=0 and allowed claims; principle: Exposed development and fixtures cannot close the independent-verifier gaps.
    - MODEL_SPECS: actual requested list and invocation status; principle: An upstream model reference is not current model work.
    - inference_substrate_class: actual class and requested_inference_substrate_class; principle: Use blocked_no_run for refusal before loading; attempted loads remain attempted loads.
    - inference_substrate: operation-accurate work description; principle: Forward work, generation and cached CPU reduction have different evidence requirements.
    - model_invocation_counts: current_llm_invocations, loads attempted/completed/failed, forward calls, generated_tokens; principle: Counters must agree with actual operation-specific receipts.
    - duration_s: measured wall time; principle: No artificial sleeping or duration padding may satisfy a substrate floor.
    - random_seed: fixed before outcomes; principle: A frozen seed permits replay without favorable seed selection.
    - reproducibility_checksum: inputs and configuration digest; principle: Changed inputs cannot pass as the original experiment.
    - source_references: original producer paths, SHA256 and imported fields; principle: Historical evidence retains its original authority.
    - code_config_hashes: owned implementation, tests and configuration; principle: Validation qualifies the executed revision.
    - current_work_receipts: operation-specific records; principle: Historical receipts cannot certify current inference.
    - validation_receipts: commands, logs, exit codes and coverage references; principle: Self-reported passing checks do not establish readiness.
    - required_checks_passed: boolean; principle: Readiness requires every affected check.
    - acceptance_gates: fixed conditions, each with a principle annotation; principle: Outcome-dependent thresholds cannot manufacture success.
    - logits_b_ready_score: integer 0/1; principle: An incomplete shard cannot authorize a smaller favorable corpus.
    - feature_shard_reference: token/statistic rows, model receipts, role and roster hashes; principle: Every feature traces to actual frozen Qwen work.
    - raw_vector_reference: selected full-vocabulary arrays with positions and hashes; principle: Aggregate energy statistics remain independently checkable.
    - checkpoint_reference: completed IDs and exact resume ledger; principle: Resume cannot silently duplicate or replace evidence.
    Run command: cd {project_root} && JAX_PLATFORMS=cpu CARNOT_FORCE_LIVE=1 .venv/bin/python -u scripts/experiments/experiment_9214_v784_logits_b.py --date {date}
    Do NOT push. Do NOT modify scripts/research_conductor.py.
  agent_type: codex
  model: gpt-6.1-sol
  operator_override: '2026-05-29 operator directive (standing): versioned lineage continuation vs exp9200-logits-b;
    Exp9194 failed cold replay on missing entropy_reference_protocol and mutable V782 authority. Exp9208 now
    requires explicit protocol ownership and immutable historical fixtures, with original assertions and real
    cold CLI checks; Exp9209 separately qualifies the unexecuted lossless corpus reader and Exp9210 the full
    validation union. This task runs only after its new exact prerequisite gates pass.'
- id: exp9215-logits-c
  title: 'Exp9215: Collect complete frozen Qwen logit shard C'
  phase: 2
  track: research
  priority: high
  requires_gpu: true
  max_turns: 50
  estimated_wall_time_min: 65
  per_unit_rows: true
  milestone: 2026.11.784
  deliverable: results/experiment_9215_v784_logits_c.json
  inference_substrate_class: model_load_no_generation
  MODEL_SPECS:
  - hf_id: unsloth/Qwen3.8-27B-GGUF
    quantization: Q4_K_M
  gated_on:
  - upstream: exp9208-contract
    artifact_field: source_ready_score
    op: ==
    value: 1
  - upstream: exp9212-logit-runtime
    artifact_field: logit_runtime_ready_score
    op: ==
    value: 1
  - upstream: exp9212-logit-runtime
    artifact_field: shard_feasible_score
    op: ==
    value: 1
  prior_failures:
  - experiment_id: exp9201-logits-c
    verdict: complete_blocked_declared_gate
    addressed_by: V783 has a log-only scheduler skip and no scientific primary; the verdict here is Exp9207
      accounting. Exp9194 failed cold replay on missing entropy_reference_protocol and mutable V782 authority.
      Exp9208 now requires explicit protocol ownership and immutable historical fixtures, with original assertions
      and real cold CLI checks; Exp9209 separately qualifies the unexecuted lossless corpus reader and Exp9210
      the full validation union. This task runs only after its new exact prerequisite gates pass.
    retire_if_same_verdict: true
  - experiment_id: exp9186-logits-c
    verdict: blocked_upstream_retired
    addressed_by: Retain the recorded prior failure. Exp9194 failed cold replay on missing entropy_reference_protocol
      and mutable V782 authority. Exp9208 now requires explicit protocol ownership and immutable historical
      fixtures, with original assertions and real cold CLI checks; Exp9209 separately qualifies the unexecuted
      lossless corpus reader and Exp9210 the full validation union. This task runs only after its new exact
      prerequisite gates pass.
    retire_if_same_verdict: true
  - experiment_id: exp9172-logits-c
    verdict: 'GATE_BLOCK: upstream retired; no canonical honest_verdict artifact'
    addressed_by: Retain the recorded prior failure. Exp9194 failed cold replay on missing entropy_reference_protocol
      and mutable V782 authority. Exp9208 now requires explicit protocol ownership and immutable historical
      fixtures, with original assertions and real cold CLI checks; Exp9209 separately qualifies the unexecuted
      lossless corpus reader and Exp9210 the full validation union. This task runs only after its new exact
      prerequisite gates pass.
    retire_if_same_verdict: true
  - experiment_id: exp9158-logits-c
    verdict: 'GATE_BLOCK: upstream retired; no canonical honest_verdict artifact'
    addressed_by: Retain the recorded prior failure. Exp9194 failed cold replay on missing entropy_reference_protocol
      and mutable V782 authority. Exp9208 now requires explicit protocol ownership and immutable historical
      fixtures, with original assertions and real cold CLI checks; Exp9209 separately qualifies the unexecuted
      lossless corpus reader and Exp9210 the full validation union. This task runs only after its new exact
      prerequisite gates pass.
    retire_if_same_verdict: true
  prompt: |-
    CONTEXT:
    Work in {project_root} on {date}. V783 closed with two executed administrative tasks and twelve skips. Exp9194 full objects matched, but cold replay lacked entropy_reference_protocol and historical V782 consumers read mutable new authority. All four published readiness scores were zero. Preserve those failures. V784 qualifies protocol-owner replay and immutable historical inputs before the lossless corpus correction and complete validation. The original human panel remains exposed development; no independent generalization is claimed.
    EXISTING CODE TO READ FIRST:
    research-references.md (V784 planning scan); openspec/change-proposals/research-roadmap-v783-preserved-20261117.md; openspec/change-proposals/research-roadmap-v783-queue-preserved-20261117.yaml; CLAUDE.md; CODEX.md; ops/north-star.md; ops/e2e-test-plan.md; ops/exclusion_manifest.yaml; openspec/change-proposals/research-roadmap-vNEXT.md; openspec/capabilities/research-reporting/spec.md; scripts/experiment_template.py; python/carnot/reporting/primary_publication.py; python/carnot/reporting/current_work_receipt.py; python/carnot/reporting/logit_methods_9126.py; python/carnot/reporting/logit_methods_runner_9126.py; python/carnot/reporting/embedding_runtime_9098.py; results/experiment_9180_v782_contract.json; results/experiment_9183_v782_logit_runtime.json
    TASK:
    Collect complete frozen Qwen logit shard C. Deliver results/experiment_9215_v784_logits_c.json. Implement scripts/experiments/experiment_9215_v784_logits_c.py using existing library seams. MODEL_SPECS: [{{"hf_id": "unsloth/Qwen3.8-27B-GGUF", "quantization": "Q4_K_M"}}]. Requested inference_substrate_class: model_load_no_generation.
    Use frozen Qwen teacher-forced forwards only; generated_tokens=0. model_load_no_generation has a 2-second floor. A small-token canary would require model_bounded_generation (10s); a real generative run requires model_full_generation (60s). Do not add either operation here.
    CONCRETE STEPS:
    1. Emit a flushed progress line at each phase boundary. Emit before and after every model load, generation, benchmark and subprocess that can take minutes. Print inside long loops. Supervise blocking calls with flushed heartbeats every 60 seconds. Keep every silence gap below 600 seconds. Write any file over about 200 lines in several tool calls of at most 200 lines. Send a progress message between writing calls. Never use one huge tool call.
    2. PRECONDITIONS: Authenticate exact required paths, hashes and gate fields before dependent work. Check imports, CPU/memory limits and private free disk. Distinguish missing_file, missing_field and observed zero. Missing external inputs produce terminal blocked evidence with gate_check_summary. Never repin historical evidence to mutable checkout code. Before model work, resolve cached_current_model(gpu_index=0), as documented by cached_sota_pair() in the experiment template. Require the mandated local GGUF, embedded tokenizer, GPU0 lease, free VRAM and actual CUDA offload support. Pin GGUF and llama.cpp build hashes. GPU1 remains with its owner. Cache or resource failure blocks; no download or small-model headline fallback.
    3. Before implementation inspect callers with rg. Confirm REQ-REPORT-9215-V784 and SCENARIO-REPORT-9215-V784. Extend specs first. Write failing controls before code. Freeze units, exclusions, formulas and validation scope before outcomes.
    4. Authenticate shard C from Exp9208: lexical qid indices200:300,100 qids/all three siblings,300 answers. Preserve original roles and source identities. Use question and answer text only. Forbid labels, references, generator tags and annotation metadata as model features.
    5. Run the frozen Exp9126 forward/statistic protocol. Keep question cap256, answer cap768, prefix bytes, no BOS/chat template and chunk128. Stream float64 sufficient statistics. Retain designated raw full-vocabulary vectors for the original global first20 lexical qids/all siblings/first8 positions. Keep per-token IDs, offsets, selected logits, logsumexp, entropy and hashes.
    6. Checkpoint after each qid. Emit progress within every answer/chunk loop. Resume only matching hashes and unfinished units. Keep failed and empty answers in their intended slots with reasons. Do not rerun completed rows selectively or replace difficult answers. Preserve cross-chunk adjacent pairs and reset context between answers.
    7. Independently recompute sampled raw statistics. Require all300 intended records, valid nonempty forwards, exact model custody and all required fields. Empty source answers retain typed escalation slots under the frozen policy. Model failure prevents complete-shard readiness. Keep evaluator labels separate. Reuse the same collector for all three shards; do not duplicate implementation.
    8. Preserve original V782 and V783 methods, assertions and failures. Exp9208 must qualify protocol-owner replay and historical input isolation; Exp9209 qualifies only the lossless corpus reader; Exp9210 must execute all original 674 and every affected obligation before model work. No retired upstream task is a dependency. Preserve the original 900 answers, scientific bounds and original protocol hashes.
    9. Run owned unit tests and real child-process coverage with 100 percent owned statements and zero exclusions. Preserve original assertions. Run scoped Ruff check/format, strict mypy and scripts/check_spec_coverage.py with explicit test paths. Execute private real CLI success, refusal, tamper and cold replay outside the checkout without PYTHONPATH, following E2E-018/021. Learning/recovery changes also require E2E-020. Separate unrelated repository health from affected failures. Affected failures disqualify readiness.
    10. Stop measurement by 3300 seconds. Use one monotonic deadline for all descendants, including cleanup; do not spend the reserve on another measurement attempt. Reserve 1200 seconds for final verification and 300 seconds for publication within the 4800-second cap. Count each obligation once; distinguish elapsed time from its watchdog ceiling. Save unfinished units and censor reasons. Recompute claims from rows. Run scripts/adversarial_verify.py and scripts/verdict_row_consistency_lint.py --strict with heartbeats. Publish through primary_publication only after actual checker receipts pass. Reconcile openspec/, _bmad/traceability.md, ops/status.md and ops/changelog.md. Preserve historical primaries, quarantines and exclusions. Do not activate a queue or publish externally.
    REQUIRED ARTIFACT FIELDS:
    - honest_verdict: complete_* for terminal outcomes; partial_* only for retryable owned work; principle: A terminal prefix must agree with its verdict class.
    - verdict_class: positive | circular_positive | null | blocked | disqualified | partial; principle: External absence, retirement or gate failure is blocked. Only unfinished owned work is partial.
    - gate_check_summary: check, upstream, path, sha256, artifact_field, op, expected, observed, passed; principle: Every blocked result names its failed operand; missing evidence differs from an observed zero.
    - preconditions_checked: resource, available and actual check receipt; principle: Unavailable resources cannot produce invented measurements.
    - rows: unit_id, qid/group, sibling, arm/condition, seed, absolute metrics, numerator, denominator, status, exclusion_reason, censor_reason; principle: All comparative headlines must be reproducible from their independent units.
    - sample_size_budget: intended, attempted, completed, failed, censored, effective_independent_groups; principle: Siblings and repeated retention do not multiply independent sample size.
    - verifier_is_oracle: boolean; principle: Reusing executable truth as the verifier requires circular_positive, not an independent verifier win.
    - claim_scope: exposure_scope, independent_generalization_score=0 and allowed claims; principle: Exposed development and fixtures cannot close the independent-verifier gaps.
    - MODEL_SPECS: actual requested list and invocation status; principle: An upstream model reference is not current model work.
    - inference_substrate_class: actual class and requested_inference_substrate_class; principle: Use blocked_no_run for refusal before loading; attempted loads remain attempted loads.
    - inference_substrate: operation-accurate work description; principle: Forward work, generation and cached CPU reduction have different evidence requirements.
    - model_invocation_counts: current_llm_invocations, loads attempted/completed/failed, forward calls, generated_tokens; principle: Counters must agree with actual operation-specific receipts.
    - duration_s: measured wall time; principle: No artificial sleeping or duration padding may satisfy a substrate floor.
    - random_seed: fixed before outcomes; principle: A frozen seed permits replay without favorable seed selection.
    - reproducibility_checksum: inputs and configuration digest; principle: Changed inputs cannot pass as the original experiment.
    - source_references: original producer paths, SHA256 and imported fields; principle: Historical evidence retains its original authority.
    - code_config_hashes: owned implementation, tests and configuration; principle: Validation qualifies the executed revision.
    - current_work_receipts: operation-specific records; principle: Historical receipts cannot certify current inference.
    - validation_receipts: commands, logs, exit codes and coverage references; principle: Self-reported passing checks do not establish readiness.
    - required_checks_passed: boolean; principle: Readiness requires every affected check.
    - acceptance_gates: fixed conditions, each with a principle annotation; principle: Outcome-dependent thresholds cannot manufacture success.
    - logits_c_ready_score: integer 0/1; principle: An incomplete shard cannot authorize a smaller favorable corpus.
    - feature_shard_reference: token/statistic rows, model receipts, role and roster hashes; principle: Every feature traces to actual frozen Qwen work.
    - raw_vector_reference: selected full-vocabulary arrays with positions and hashes; principle: Aggregate energy statistics remain independently checkable.
    - checkpoint_reference: completed IDs and exact resume ledger; principle: Resume cannot silently duplicate or replace evidence.
    Run command: cd {project_root} && JAX_PLATFORMS=cpu CARNOT_FORCE_LIVE=1 .venv/bin/python -u scripts/experiments/experiment_9215_v784_logits_c.py --date {date}
    Do NOT push. Do NOT modify scripts/research_conductor.py.
  agent_type: codex
  model: gpt-6.1-sol
  operator_override: '2026-05-29 operator directive (standing): versioned lineage continuation vs exp9201-logits-c;
    Exp9194 failed cold replay on missing entropy_reference_protocol and mutable V782 authority. Exp9208 now
    requires explicit protocol ownership and immutable historical fixtures, with original assertions and real
    cold CLI checks; Exp9209 separately qualifies the unexecuted lossless corpus reader and Exp9210 the full
    validation union. This task runs only after its new exact prerequisite gates pass.'
- id: exp9216-logit-admission
  title: 'Exp9216: Admit the exact full cohort and retain separate fit views'
  phase: 2
  track: research
  priority: high
  requires_gpu: false
  max_turns: 30
  estimated_wall_time_min: 45
  per_unit_rows: true
  milestone: 2026.11.784
  deliverable: results/experiment_9216_v784_logit_admission.json
  inference_substrate_class: no_model_load
  MODEL_SPECS: []
  agent_type: codex
  model: gpt-6.1-sol
  gated_on:
  - upstream: exp9213-logits-a
    artifact_field: logits_a_ready_score
    op: ==
    value: 1
  - upstream: exp9214-logits-b
    artifact_field: logits_b_ready_score
    op: ==
    value: 1
  - upstream: exp9215-logits-c
    artifact_field: logits_c_ready_score
    op: ==
    value: 1
  prior_failures:
  - experiment_id: exp9202-logit-admission
    verdict: complete_blocked_declared_gate
    addressed_by: V783 has a log-only scheduler skip and no scientific primary; the verdict here is Exp9207
      accounting. Exp9194 failed cold replay on missing entropy_reference_protocol and mutable V782 authority.
      Exp9208 now requires explicit protocol ownership and immutable historical fixtures, with original assertions
      and real cold CLI checks; Exp9209 separately qualifies the unexecuted lossless corpus reader and Exp9210
      the full validation union. This task runs only after its new exact prerequisite gates pass.
    retire_if_same_verdict: true
  - experiment_id: exp9187-logit-admission
    verdict: blocked_upstream_retired
    addressed_by: Retain the recorded prior failure. Exp9194 failed cold replay on missing entropy_reference_protocol
      and mutable V782 authority. Exp9208 now requires explicit protocol ownership and immutable historical
      fixtures, with original assertions and real cold CLI checks; Exp9209 separately qualifies the unexecuted
      lossless corpus reader and Exp9210 the full validation union. This task runs only after its new exact
      prerequisite gates pass.
    retire_if_same_verdict: true
  - experiment_id: exp9173-logit-admission
    verdict: 'GATE_BLOCK: upstream retired; no canonical honest_verdict artifact'
    addressed_by: Retain the recorded prior failure. Exp9194 failed cold replay on missing entropy_reference_protocol
      and mutable V782 authority. Exp9208 now requires explicit protocol ownership and immutable historical
      fixtures, with original assertions and real cold CLI checks; Exp9209 separately qualifies the unexecuted
      lossless corpus reader and Exp9210 the full validation union. This task runs only after its new exact
      prerequisite gates pass.
    retire_if_same_verdict: true
  - experiment_id: exp9159-logit-admission
    verdict: 'GATE_BLOCK: upstream retired; no canonical honest_verdict artifact'
    addressed_by: Retain the recorded prior failure. Exp9194 failed cold replay on missing entropy_reference_protocol
      and mutable V782 authority. Exp9208 now requires explicit protocol ownership and immutable historical
      fixtures, with original assertions and real cold CLI checks; Exp9209 separately qualifies the unexecuted
      lossless corpus reader and Exp9210 the full validation union. This task runs only after its new exact
      prerequisite gates pass.
    retire_if_same_verdict: true
  prompt: |-
    CONTEXT:
    Work in {project_root} on {date}. V783 closed with two executed administrative tasks and twelve skips. Exp9194 full objects matched, but cold replay lacked entropy_reference_protocol and historical V782 consumers read mutable new authority. All four published readiness scores were zero. Preserve those failures. V784 qualifies protocol-owner replay and immutable historical inputs before the lossless corpus correction and complete validation. The original human panel remains exposed development; no independent generalization is claimed.
    EXISTING CODE TO READ FIRST:
    research-references.md (V784 planning scan); openspec/change-proposals/research-roadmap-v783-preserved-20261117.md; openspec/change-proposals/research-roadmap-v783-queue-preserved-20261117.yaml; CLAUDE.md; CODEX.md; ops/north-star.md; ops/e2e-test-plan.md; ops/exclusion_manifest.yaml; openspec/change-proposals/research-roadmap-vNEXT.md; openspec/capabilities/research-reporting/spec.md; scripts/experiment_template.py; python/carnot/reporting/primary_publication.py; python/carnot/reporting/current_work_receipt.py; python/carnot/reporting/logit_methods_9126.py; python/carnot/reporting/online_methods_9132.py; results/experiment_9180_v782_contract.json; results/experiment_9184_v782_logits_a.json; results/experiment_9185_v782_logits_b.json; results/experiment_9186_v782_logits_c.json
    TASK:
    Admit the exact full cohort and retain separate fit views. Deliver results/experiment_9216_v784_logit_admission.json. Implement scripts/experiments/experiment_9216_v784_logit_admission.py using existing library seams. MODEL_SPECS: []. Requested inference_substrate_class: no_model_load.
    Do not load an LLM. Set current_llm_invocations=0 and generated_tokens=0. Small energy-head training is CPU work. Keep historical model provenance in source_references.
    CONCRETE STEPS:
    1. Emit a flushed progress line at each phase boundary. Emit before and after every model load, generation, benchmark and subprocess that can take minutes. Print inside long loops. Supervise blocking calls with flushed heartbeats every 60 seconds. Keep every silence gap below 600 seconds. Write any file over about 200 lines in several tool calls of at most 200 lines. Send a progress message between writing calls. Never use one huge tool call.
    2. PRECONDITIONS: Authenticate exact required paths, hashes and gate fields before dependent work. Check imports, CPU/memory limits and private free disk. Distinguish missing_file, missing_field and observed zero. Missing external inputs produce terminal blocked evidence with gate_check_summary. Never repin historical evidence to mutable checkout code.
    3. Before implementation inspect callers with rg. Confirm REQ-REPORT-9216-V784 and SCENARIO-REPORT-9216-V784. Extend specs first. Write failing controls before code. Freeze units, exclusions, formulas and validation scope before outcomes.
    4. Join exactly900 answers/300 qids from three disjoint100-qid shards. Verify original TRAIN150/CAL50/TEST100 membership, sibling order and exact model/tokenizer/build hashes. Prove equality to the original two-shard population. Retain every failed/empty slot. Never select a favorable intersection.
    5. Cold-recompute all retained raw-vector statistics and token indexing. Check full-vocabulary dimensions, prefix/answer boundary, clipping/cap accounting and chunk adjacency. Reject model-input label fields and opaque aggregate-only evidence.
    6. Emit separate static and online feature views. Static preprocessing fits TRAIN150; online initialization fits only its prescribed100 TRAIN groups. Keep evaluator labels outside both scoring views. Record provenance of historical generation separately from current teacher forcing.
    7. Admission passes on a complete well-formed corpus even when every downstream scientific gain may be null. Empty answers follow the frozen escalation/missing-feature rules. Preserve all original denominators and expose exact common support for proper-loss comparisons.
    8. Retain token-entropy sequences in the admitted feature view for the frozen CES diagnostic. They are already required collection statistics; no extra LLM invocation or replacement source is permitted. Preserve distinct TRAIN label authority for reference construction and evaluator-only TEST labels.
    9. Preserve original V782 and V783 methods, assertions and failures. Exp9208 must qualify protocol-owner replay and historical input isolation; Exp9209 qualifies only the lossless corpus reader; Exp9210 must execute all original 674 and every affected obligation before model work. No retired upstream task is a dependency. Preserve the original 900 answers, scientific bounds and original protocol hashes.
    10. Run owned unit tests and real child-process coverage with 100 percent owned statements and zero exclusions. Preserve original assertions. Run scoped Ruff check/format, strict mypy and scripts/check_spec_coverage.py with explicit test paths. Execute private real CLI success, refusal, tamper and cold replay outside the checkout without PYTHONPATH, following E2E-018/021. Learning/recovery changes also require E2E-020. Separate unrelated repository health from affected failures. Affected failures disqualify readiness.
    11. Stop measurement by 3300 seconds. Use one monotonic deadline for all descendants, including cleanup; do not spend the reserve on another measurement attempt. Reserve 1200 seconds for final verification and 300 seconds for publication within the 4800-second cap. Count each obligation once; distinguish elapsed time from its watchdog ceiling. Save unfinished units and censor reasons. Recompute claims from rows. Run scripts/adversarial_verify.py and scripts/verdict_row_consistency_lint.py --strict with heartbeats. Publish through primary_publication only after actual checker receipts pass. Reconcile openspec/, _bmad/traceability.md, ops/status.md and ops/changelog.md. Preserve historical primaries, quarantines and exclusions. Do not activate a queue or publish externally.
    REQUIRED ARTIFACT FIELDS:
    - honest_verdict: complete_* for terminal outcomes; partial_* only for retryable owned work; principle: A terminal prefix must agree with its verdict class.
    - verdict_class: positive | circular_positive | null | blocked | disqualified | partial; principle: External absence, retirement or gate failure is blocked. Only unfinished owned work is partial.
    - gate_check_summary: check, upstream, path, sha256, artifact_field, op, expected, observed, passed; principle: Every blocked result names its failed operand; missing evidence differs from an observed zero.
    - preconditions_checked: resource, available and actual check receipt; principle: Unavailable resources cannot produce invented measurements.
    - rows: unit_id, qid/group, sibling, arm/condition, seed, absolute metrics, numerator, denominator, status, exclusion_reason, censor_reason; principle: All comparative headlines must be reproducible from their independent units.
    - sample_size_budget: intended, attempted, completed, failed, censored, effective_independent_groups; principle: Siblings and repeated retention do not multiply independent sample size.
    - verifier_is_oracle: boolean; principle: Reusing executable truth as the verifier requires circular_positive, not an independent verifier win.
    - claim_scope: exposure_scope, independent_generalization_score=0 and allowed claims; principle: Exposed development and fixtures cannot close the independent-verifier gaps.
    - MODEL_SPECS: actual requested list and invocation status; principle: An upstream model reference is not current model work.
    - inference_substrate_class: actual class and requested_inference_substrate_class; principle: Use blocked_no_run for refusal before loading; attempted loads remain attempted loads.
    - inference_substrate: operation-accurate work description; principle: Forward work, generation and cached CPU reduction have different evidence requirements.
    - model_invocation_counts: current_llm_invocations, loads attempted/completed/failed, forward calls, generated_tokens; principle: Counters must agree with actual operation-specific receipts.
    - duration_s: measured wall time; principle: No artificial sleeping or duration padding may satisfy a substrate floor.
    - random_seed: fixed before outcomes; principle: A frozen seed permits replay without favorable seed selection.
    - reproducibility_checksum: inputs and configuration digest; principle: Changed inputs cannot pass as the original experiment.
    - source_references: original producer paths, SHA256 and imported fields; principle: Historical evidence retains its original authority.
    - code_config_hashes: owned implementation, tests and configuration; principle: Validation qualifies the executed revision.
    - current_work_receipts: operation-specific records; principle: Historical receipts cannot certify current inference.
    - validation_receipts: commands, logs, exit codes and coverage references; principle: Self-reported passing checks do not establish readiness.
    - required_checks_passed: boolean; principle: Readiness requires every affected check.
    - acceptance_gates: fixed conditions, each with a principle annotation; principle: Outcome-dependent thresholds cannot manufacture success.
    - admission_ready_score: integer 0/1; principle: Complete custody is not scientific benefit.
    - static_features_reference: all role-bound feature records; principle: Static fitting cannot silently alter the population.
    - online_features_reference: initial100 and stream50 roles; principle: Static150 information cannot leak into online initialization.
    - evaluator_reference: private labels plus original authority; principle: Scoring cannot observe TEST truth.
    - raw_reduction_rows: per-token parity, joins, exclusions and exact support; principle: The admission decision can be recomputed.
    Run command: cd {project_root} && JAX_PLATFORMS=cpu .venv/bin/python -u scripts/experiments/experiment_9216_v784_logit_admission.py --date {date}
    Do NOT push. Do NOT modify scripts/research_conductor.py.
  operator_override: '2026-05-29 operator directive (standing): versioned lineage continuation vs exp9202-logit-admission;
    Exp9194 failed cold replay on missing entropy_reference_protocol and mutable V782 authority. Exp9208 now
    requires explicit protocol ownership and immutable historical fixtures, with original assertions and real
    cold CLI checks; Exp9209 separately qualifies the unexecuted lossless corpus reader and Exp9210 the full
    validation union. This task runs only after its new exact prerequisite gates pass.'
- id: exp9217-static-energy
  title: 'Exp9217: Train the five fixed human energy decision arms'
  phase: 3
  track: research
  priority: high
  requires_gpu: false
  max_turns: 50
  estimated_wall_time_min: 60
  per_unit_rows: true
  milestone: 2026.11.784
  deliverable: results/experiment_9217_v784_static_energy.json
  inference_substrate_class: no_model_load
  MODEL_SPECS: []
  gated_on:
  - upstream: exp9216-logit-admission
    artifact_field: admission_ready_score
    op: ==
    value: 1
  - upstream: exp9208-contract
    artifact_field: static_methods_ready_score
    op: ==
    value: 1
  prior_failures:
  - experiment_id: exp9203-static-energy
    verdict: complete_blocked_declared_gate
    addressed_by: V783 has a log-only scheduler skip and no scientific primary; the verdict here is Exp9207
      accounting. Exp9194 failed cold replay on missing entropy_reference_protocol and mutable V782 authority.
      Exp9208 now requires explicit protocol ownership and immutable historical fixtures, with original assertions
      and real cold CLI checks; Exp9209 separately qualifies the unexecuted lossless corpus reader and Exp9210
      the full validation union. This task runs only after its new exact prerequisite gates pass.
    retire_if_same_verdict: true
  - experiment_id: exp9188-static-energy
    verdict: blocked_upstream_retired
    addressed_by: Retain the recorded prior failure. Exp9194 failed cold replay on missing entropy_reference_protocol
      and mutable V782 authority. Exp9208 now requires explicit protocol ownership and immutable historical
      fixtures, with original assertions and real cold CLI checks; Exp9209 separately qualifies the unexecuted
      lossless corpus reader and Exp9210 the full validation union. This task runs only after its new exact
      prerequisite gates pass.
    retire_if_same_verdict: true
  - experiment_id: exp9174-static-energy
    verdict: 'GATE_BLOCK: upstream retired; no canonical honest_verdict artifact'
    addressed_by: Retain the recorded prior failure. Exp9194 failed cold replay on missing entropy_reference_protocol
      and mutable V782 authority. Exp9208 now requires explicit protocol ownership and immutable historical
      fixtures, with original assertions and real cold CLI checks; Exp9209 separately qualifies the unexecuted
      lossless corpus reader and Exp9210 the full validation union. This task runs only after its new exact
      prerequisite gates pass.
    retire_if_same_verdict: true
  - experiment_id: exp9160-static-energy
    verdict: 'GATE_BLOCK: upstream retired; no canonical honest_verdict artifact'
    addressed_by: Retain the recorded prior failure. Exp9194 failed cold replay on missing entropy_reference_protocol
      and mutable V782 authority. Exp9208 now requires explicit protocol ownership and immutable historical
      fixtures, with original assertions and real cold CLI checks; Exp9209 separately qualifies the unexecuted
      lossless corpus reader and Exp9210 the full validation union. This task runs only after its new exact
      prerequisite gates pass.
    retire_if_same_verdict: true
  prompt: |-
    CONTEXT:
    Work in {project_root} on {date}. V783 closed with two executed administrative tasks and twelve skips. Exp9194 full objects matched, but cold replay lacked entropy_reference_protocol and historical V782 consumers read mutable new authority. All four published readiness scores were zero. Preserve those failures. V784 qualifies protocol-owner replay and immutable historical inputs before the lossless corpus correction and complete validation. The original human panel remains exposed development; no independent generalization is claimed.
    EXISTING CODE TO READ FIRST:
    research-references.md (V784 planning scan); openspec/change-proposals/research-roadmap-v783-preserved-20261117.md; openspec/change-proposals/research-roadmap-v783-queue-preserved-20261117.yaml; CLAUDE.md; CODEX.md; ops/north-star.md; ops/e2e-test-plan.md; ops/exclusion_manifest.yaml; openspec/change-proposals/research-roadmap-vNEXT.md; openspec/capabilities/research-reporting/spec.md; scripts/experiment_template.py; python/carnot/reporting/primary_publication.py; python/carnot/reporting/current_work_receipt.py; python/carnot/reporting/logit_methods_9126.py; python/carnot/reporting/logit_methods_runner_9126.py; results/experiment_9126_v778_logit_methods.json; results/experiment_9187_v782_logit_admission.json
    TASK:
    Train the five fixed human energy decision arms. Deliver results/experiment_9217_v784_static_energy.json. Implement scripts/experiments/experiment_9217_v784_static_energy.py using existing library seams. MODEL_SPECS: []. Requested inference_substrate_class: no_model_load.
    Do not load an LLM. Set current_llm_invocations=0 and generated_tokens=0. Small energy-head training is CPU work. Keep historical model provenance in source_references.
    CONCRETE STEPS:
    1. Emit a flushed progress line at each phase boundary. Emit before and after every model load, generation, benchmark and subprocess that can take minutes. Print inside long loops. Supervise blocking calls with flushed heartbeats every 60 seconds. Keep every silence gap below 600 seconds. Write any file over about 200 lines in several tool calls of at most 200 lines. Send a progress message between writing calls. Never use one huge tool call.
    2. PRECONDITIONS: Authenticate exact required paths, hashes and gate fields before dependent work. Check imports, CPU/memory limits and private free disk. Distinguish missing_file, missing_field and observed zero. Missing external inputs produce terminal blocked evidence with gate_check_summary. Never repin historical evidence to mutable checkout code.
    3. Before implementation inspect callers with rg. Confirm REQ-REPORT-9217-V784 and SCENARIO-REPORT-9217-V784. Extend specs first. Write failing controls before code. Freeze units, exclusions, formulas and validation scope before outcomes.
    4. Use the frozen seven features and five arms: prevalence, probability+length logistic, raw-energy logistic, combined logistic, and combined with whole-qid TRAIN donor labels. Treat each classifier as a binary Gibbs energy. Keep original TRAIN150/CAL50/TEST100 roles, ridge0.01, zero start and original convergence checks.
    5. Fit imputation/scaling on TRAIN only. Fit each CAL temperature over[0.5,1,2,4], ascending ties. Emit accept at p(error)<=0.1, reject at>=0.9, otherwise escalate. Retain empty-answer escalation slots. Do not use generator identity or evaluator-only contrast masks as features.
    6. Seal all per-answer TEST probabilities, logits and typed decisions before evaluation. Include original A1 targets only in the evaluator view. Retain A2 as separately observed sensitivity. Record actual changed donor-label counts rather than assuming that a permutation changed every label.
    7. Emit static_predictions_ready_score from completeness, convergence and provenance only. Do not gate the later audit on improvement. Retain fitted parameters and transforms for complete-service replay.
    8. Preserve original V782 and V783 methods, assertions and failures. Exp9208 must qualify protocol-owner replay and historical input isolation; Exp9209 qualifies only the lossless corpus reader; Exp9210 must execute all original 674 and every affected obligation before model work. No retired upstream task is a dependency. Preserve the original 900 answers, scientific bounds and original protocol hashes.
    9. Run owned unit tests and real child-process coverage with 100 percent owned statements and zero exclusions. Preserve original assertions. Run scoped Ruff check/format, strict mypy and scripts/check_spec_coverage.py with explicit test paths. Execute private real CLI success, refusal, tamper and cold replay outside the checkout without PYTHONPATH, following E2E-018/021. Learning/recovery changes also require E2E-020. Separate unrelated repository health from affected failures. Affected failures disqualify readiness.
    10. Stop measurement by 3300 seconds. Use one monotonic deadline for all descendants, including cleanup; do not spend the reserve on another measurement attempt. Reserve 1200 seconds for final verification and 300 seconds for publication within the 4800-second cap. Count each obligation once; distinguish elapsed time from its watchdog ceiling. Save unfinished units and censor reasons. Recompute claims from rows. Run scripts/adversarial_verify.py and scripts/verdict_row_consistency_lint.py --strict with heartbeats. Publish through primary_publication only after actual checker receipts pass. Reconcile openspec/, _bmad/traceability.md, ops/status.md and ops/changelog.md. Preserve historical primaries, quarantines and exclusions. Do not activate a queue or publish externally.
    REQUIRED ARTIFACT FIELDS:
    - honest_verdict: complete_* for terminal outcomes; partial_* only for retryable owned work; principle: A terminal prefix must agree with its verdict class.
    - verdict_class: positive | circular_positive | null | blocked | disqualified | partial; principle: External absence, retirement or gate failure is blocked. Only unfinished owned work is partial.
    - gate_check_summary: check, upstream, path, sha256, artifact_field, op, expected, observed, passed; principle: Every blocked result names its failed operand; missing evidence differs from an observed zero.
    - preconditions_checked: resource, available and actual check receipt; principle: Unavailable resources cannot produce invented measurements.
    - rows: unit_id, qid/group, sibling, arm/condition, seed, absolute metrics, numerator, denominator, status, exclusion_reason, censor_reason; principle: All comparative headlines must be reproducible from their independent units.
    - sample_size_budget: intended, attempted, completed, failed, censored, effective_independent_groups; principle: Siblings and repeated retention do not multiply independent sample size.
    - verifier_is_oracle: boolean; principle: Reusing executable truth as the verifier requires circular_positive, not an independent verifier win.
    - claim_scope: exposure_scope, independent_generalization_score=0 and allowed claims; principle: Exposed development and fixtures cannot close the independent-verifier gaps.
    - MODEL_SPECS: actual requested list and invocation status; principle: An upstream model reference is not current model work.
    - inference_substrate_class: actual class and requested_inference_substrate_class; principle: Use blocked_no_run for refusal before loading; attempted loads remain attempted loads.
    - inference_substrate: operation-accurate work description; principle: Forward work, generation and cached CPU reduction have different evidence requirements.
    - model_invocation_counts: current_llm_invocations, loads attempted/completed/failed, forward calls, generated_tokens; principle: Counters must agree with actual operation-specific receipts.
    - duration_s: measured wall time; principle: No artificial sleeping or duration padding may satisfy a substrate floor.
    - random_seed: fixed before outcomes; principle: A frozen seed permits replay without favorable seed selection.
    - reproducibility_checksum: inputs and configuration digest; principle: Changed inputs cannot pass as the original experiment.
    - source_references: original producer paths, SHA256 and imported fields; principle: Historical evidence retains its original authority.
    - code_config_hashes: owned implementation, tests and configuration; principle: Validation qualifies the executed revision.
    - current_work_receipts: operation-specific records; principle: Historical receipts cannot certify current inference.
    - validation_receipts: commands, logs, exit codes and coverage references; principle: Self-reported passing checks do not establish readiness.
    - required_checks_passed: boolean; principle: Readiness requires every affected check.
    - acceptance_gates: fixed conditions, each with a principle annotation; principle: Outcome-dependent thresholds cannot manufacture success.
    - static_predictions_ready_score: integer 0/1; principle: Complete null results must reach the independent audit.
    - static_prediction_reference: five arms by qid/sibling with prediction seal; principle: Every risk and coverage comparison uses identical units.
    - fitted_heads_reference: parameters, transforms, CAL temperatures and convergence receipts; principle: Service replay must use the measured policy.
    - donor_change_rows: original and permuted training-label identities; principle: An unchanged donor is not a valid information control.
    Run command: cd {project_root} && JAX_PLATFORMS=cpu .venv/bin/python -u scripts/experiments/experiment_9217_v784_static_energy.py --date {date}
    Do NOT push. Do NOT modify scripts/research_conductor.py.
  model: sonnet
  operator_override: '2026-05-29 operator directive (standing): versioned lineage continuation vs exp9203-static-energy;
    Exp9194 failed cold replay on missing entropy_reference_protocol and mutable V782 authority. Exp9208 now
    requires explicit protocol ownership and immutable historical fixtures, with original assertions and real
    cold CLI checks; Exp9209 separately qualifies the unexecuted lossless corpus reader and Exp9210 the full
    validation union. This task runs only after its new exact prerequisite gates pass.'
  agent_type: claude
- id: exp9218-delayed-learning
  title: 'Exp9218: Measure continuous small-energy learning with exact recovery'
  phase: 3
  track: research
  priority: high
  requires_gpu: false
  max_turns: 50
  estimated_wall_time_min: 70
  per_unit_rows: true
  milestone: 2026.11.784
  deliverable: results/experiment_9218_v784_delayed_learning.json
  inference_substrate_class: no_model_load
  MODEL_SPECS: []
  gated_on:
  - upstream: exp9216-logit-admission
    artifact_field: admission_ready_score
    op: ==
    value: 1
  - upstream: exp9208-contract
    artifact_field: online_methods_ready_score
    op: ==
    value: 1
  prior_failures:
  - experiment_id: exp9204-delayed-learning
    verdict: complete_blocked_declared_gate
    addressed_by: V783 has a log-only scheduler skip and no scientific primary; the verdict here is Exp9207
      accounting. Exp9194 failed cold replay on missing entropy_reference_protocol and mutable V782 authority.
      Exp9208 now requires explicit protocol ownership and immutable historical fixtures, with original assertions
      and real cold CLI checks; Exp9209 separately qualifies the unexecuted lossless corpus reader and Exp9210
      the full validation union. This task runs only after its new exact prerequisite gates pass.
    retire_if_same_verdict: true
  - experiment_id: exp9189-delayed-learning
    verdict: blocked_upstream_retired
    addressed_by: Retain the recorded prior failure. Exp9194 failed cold replay on missing entropy_reference_protocol
      and mutable V782 authority. Exp9208 now requires explicit protocol ownership and immutable historical
      fixtures, with original assertions and real cold CLI checks; Exp9209 separately qualifies the unexecuted
      lossless corpus reader and Exp9210 the full validation union. This task runs only after its new exact
      prerequisite gates pass.
    retire_if_same_verdict: true
  - experiment_id: exp9175-delayed-learning
    verdict: 'GATE_BLOCK: upstream retired; no canonical honest_verdict artifact'
    addressed_by: Retain the recorded prior failure. Exp9194 failed cold replay on missing entropy_reference_protocol
      and mutable V782 authority. Exp9208 now requires explicit protocol ownership and immutable historical
      fixtures, with original assertions and real cold CLI checks; Exp9209 separately qualifies the unexecuted
      lossless corpus reader and Exp9210 the full validation union. This task runs only after its new exact
      prerequisite gates pass.
    retire_if_same_verdict: true
  - experiment_id: exp9161-delayed-learning
    verdict: 'GATE_BLOCK: upstream retired; no canonical honest_verdict artifact'
    addressed_by: Retain the recorded prior failure. Exp9194 failed cold replay on missing entropy_reference_protocol
      and mutable V782 authority. Exp9208 now requires explicit protocol ownership and immutable historical
      fixtures, with original assertions and real cold CLI checks; Exp9209 separately qualifies the unexecuted
      lossless corpus reader and Exp9210 the full validation union. This task runs only after its new exact
      prerequisite gates pass.
    retire_if_same_verdict: true
  prompt: |-
    CONTEXT:
    Work in {project_root} on {date}. V783 closed with two executed administrative tasks and twelve skips. Exp9194 full objects matched, but cold replay lacked entropy_reference_protocol and historical V782 consumers read mutable new authority. All four published readiness scores were zero. Preserve those failures. V784 qualifies protocol-owner replay and immutable historical inputs before the lossless corpus correction and complete validation. The original human panel remains exposed development; no independent generalization is claimed.
    EXISTING CODE TO READ FIRST:
    research-references.md (V784 planning scan); openspec/change-proposals/research-roadmap-v783-preserved-20261117.md; openspec/change-proposals/research-roadmap-v783-queue-preserved-20261117.yaml; CLAUDE.md; CODEX.md; ops/north-star.md; ops/e2e-test-plan.md; ops/exclusion_manifest.yaml; openspec/change-proposals/research-roadmap-vNEXT.md; openspec/capabilities/research-reporting/spec.md; scripts/experiment_template.py; python/carnot/reporting/primary_publication.py; python/carnot/reporting/current_work_receipt.py; python/carnot/reporting/online_methods_9132.py; python/carnot/reporting/online_methods_runner_9132.py; python/carnot/memory/human_online_9099.py; tests/python/test_hard_exit_learning_qualification_8206.py; results/experiment_9132_v778_online_methods.json; results/experiment_9187_v782_logit_admission.json
    TASK:
    Measure continuous small-energy learning with exact recovery. Deliver results/experiment_9218_v784_delayed_learning.json. Implement scripts/experiments/experiment_9218_v784_delayed_learning.py using existing library seams. MODEL_SPECS: []. Requested inference_substrate_class: no_model_load.
    Do not load an LLM. Set current_llm_invocations=0 and generated_tokens=0. Small energy-head training is CPU work. Keep historical model provenance in source_references.
    CONCRETE STEPS:
    1. Emit a flushed progress line at each phase boundary. Emit before and after every model load, generation, benchmark and subprocess that can take minutes. Print inside long loops. Supervise blocking calls with flushed heartbeats every 60 seconds. Keep every silence gap below 600 seconds. Write any file over about 200 lines in several tool calls of at most 200 lines. Send a progress message between writing calls. Never use one huge tool call.
    2. PRECONDITIONS: Authenticate exact required paths, hashes and gate fields before dependent work. Check imports, CPU/memory limits and private free disk. Distinguish missing_file, missing_field and observed zero. Missing external inputs produce terminal blocked evidence with gate_check_summary. Never repin historical evidence to mutable checkout code.
    3. Before implementation inspect callers with rg. Confirm REQ-REPORT-9218-V784 and SCENARIO-REPORT-9218-V784. Extend specs first. Write failing controls before code. Freeze units, exclusions, formulas and validation scope before outcomes.
    4. Reuse Exp9132 exactly: original100-group initialization, ten disjoint five-qid streams from the other50 TRAIN groups, CAL50 operational guard and TEST100 research retention. Fit every transform only on initialization. Keep the fixed57-column cubic basis and six original arms: frozen spline, adaptive spline, bias-only, adaptive linear, donor spline and temperature-only.
    5. Use original AdaGrad0.05, clipping, anchoring, donor map and temperature bounds. Seal all18 sibling/arm predictions for group g before releasing g-1 labels. Apply siblings in original order and guard once per released qid. Preserve consumed-feedback acknowledgments on rejected updates. Drain final feedback only after predictions finish.
    6. Use only CAL for promotion: Brier increase<=0.005 and classification-error increase<=0 against that arm's initialization. TEST retention never chooses an update. Reset each stream to the same initial state. No static150 fit or future feedback enters initialization.
    7. Persist predictions, parameters, accumulators, temperature, pending feedback and acknowledgments atomically. Run real hard-exit children at prediction fsync, feedback receipt and state replacement. Require exact uninterrupted/resumed predictions and complete states. Record all proposed, accepted and rejected updates and actual coefficient writes.
    8. Seal complete online and retention predictions, including raw pre-sigmoid logits before and after every update. Key each retained logit by stream, released qid, update, arm, evaluation qid, sibling and state hash. Retain initial logits and explicit pre/post state identities; never recover logits from saturated probabilities. Emit online_predictions_ready_score from valid causal execution and recovery, regardless of gain. Freeze probability-change and classification-flip diagnostic memberships without labels for Exp9219; these never affect updates or promotion.
    9. Preserve original V782 and V783 methods, assertions and failures. Exp9208 must qualify protocol-owner replay and historical input isolation; Exp9209 qualifies only the lossless corpus reader; Exp9210 must execute all original 674 and every affected obligation before model work. No retired upstream task is a dependency. Preserve the original 900 answers, scientific bounds and original protocol hashes.
    10. Run owned unit tests and real child-process coverage with 100 percent owned statements and zero exclusions. Preserve original assertions. Run scoped Ruff check/format, strict mypy and scripts/check_spec_coverage.py with explicit test paths. Execute private real CLI success, refusal, tamper and cold replay outside the checkout without PYTHONPATH, following E2E-018/021. Learning/recovery changes also require E2E-020. Separate unrelated repository health from affected failures. Affected failures disqualify readiness.
    11. Stop measurement by 3300 seconds. Use one monotonic deadline for all descendants, including cleanup; do not spend the reserve on another measurement attempt. Reserve 1200 seconds for final verification and 300 seconds for publication within the 4800-second cap. Count each obligation once; distinguish elapsed time from its watchdog ceiling. Save unfinished units and censor reasons. Recompute claims from rows. Run scripts/adversarial_verify.py and scripts/verdict_row_consistency_lint.py --strict with heartbeats. Publish through primary_publication only after actual checker receipts pass. Reconcile openspec/, _bmad/traceability.md, ops/status.md and ops/changelog.md. Preserve historical primaries, quarantines and exclusions. Do not activate a queue or publish externally.
    REQUIRED ARTIFACT FIELDS:
    - honest_verdict: complete_* for terminal outcomes; partial_* only for retryable owned work; principle: A terminal prefix must agree with its verdict class.
    - verdict_class: positive | circular_positive | null | blocked | disqualified | partial; principle: External absence, retirement or gate failure is blocked. Only unfinished owned work is partial.
    - gate_check_summary: check, upstream, path, sha256, artifact_field, op, expected, observed, passed; principle: Every blocked result names its failed operand; missing evidence differs from an observed zero.
    - preconditions_checked: resource, available and actual check receipt; principle: Unavailable resources cannot produce invented measurements.
    - rows: unit_id, qid/group, sibling, arm/condition, seed, absolute metrics, numerator, denominator, status, exclusion_reason, censor_reason; principle: All comparative headlines must be reproducible from their independent units.
    - sample_size_budget: intended, attempted, completed, failed, censored, effective_independent_groups; principle: Siblings and repeated retention do not multiply independent sample size.
    - verifier_is_oracle: boolean; principle: Reusing executable truth as the verifier requires circular_positive, not an independent verifier win.
    - claim_scope: exposure_scope, independent_generalization_score=0 and allowed claims; principle: Exposed development and fixtures cannot close the independent-verifier gaps.
    - MODEL_SPECS: actual requested list and invocation status; principle: An upstream model reference is not current model work.
    - inference_substrate_class: actual class and requested_inference_substrate_class; principle: Use blocked_no_run for refusal before loading; attempted loads remain attempted loads.
    - inference_substrate: operation-accurate work description; principle: Forward work, generation and cached CPU reduction have different evidence requirements.
    - model_invocation_counts: current_llm_invocations, loads attempted/completed/failed, forward calls, generated_tokens; principle: Counters must agree with actual operation-specific receipts.
    - duration_s: measured wall time; principle: No artificial sleeping or duration padding may satisfy a substrate floor.
    - random_seed: fixed before outcomes; principle: A frozen seed permits replay without favorable seed selection.
    - reproducibility_checksum: inputs and configuration digest; principle: Changed inputs cannot pass as the original experiment.
    - source_references: original producer paths, SHA256 and imported fields; principle: Historical evidence retains its original authority.
    - code_config_hashes: owned implementation, tests and configuration; principle: Validation qualifies the executed revision.
    - current_work_receipts: operation-specific records; principle: Historical receipts cannot certify current inference.
    - validation_receipts: commands, logs, exit codes and coverage references; principle: Self-reported passing checks do not establish readiness.
    - required_checks_passed: boolean; principle: Readiness requires every affected check.
    - acceptance_gates: fixed conditions, each with a principle annotation; principle: Outcome-dependent thresholds cannot manufacture success.
    - online_predictions_ready_score: integer 0/1; principle: Complete causal nulls must remain auditable.
    - online_prediction_reference: six arms, stream/qid/sibling order and timestamps; principle: Prediction must precede feedback.
    - retention_prediction_reference: initial and post-update TEST predictions; principle: Repeated evaluation does not create new independent questions.
    - raw_logit_reference: initial and per-update pre/post logits keyed by stream, released qid, update, arm, qid, sibling and state hash; principle: Scale diagnostics require original logits, not inversion of clipped probabilities.
    - durable_state_reference: all state, release, proposal, rejection and acknowledgment rows; principle: Learning must survive real process failure without data loss.
    - recovery_rows: actual exit codes, state hashes and prediction equality; principle: A simulated exception is not hard-exit recovery.
    Run command: cd {project_root} && JAX_PLATFORMS=cpu .venv/bin/python -u scripts/experiments/experiment_9218_v784_delayed_learning.py --date {date}
    Do NOT push. Do NOT modify scripts/research_conductor.py.
  model: sonnet
  operator_override: '2026-05-29 operator directive (standing): versioned lineage continuation vs exp9204-delayed-learning;
    Exp9194 failed cold replay on missing entropy_reference_protocol and mutable V782 authority. Exp9208 now
    requires explicit protocol ownership and immutable historical fixtures, with original assertions and real
    cold CLI checks; Exp9209 separately qualifies the unexecuted lossless corpus reader and Exp9210 the full
    validation union. This task runs only after its new exact prerequisite gates pass.'
  agent_type: claude
- id: exp9219-decision-audit
  title: 'Exp9219: Audit primary gains, entropy controls and human-label ambiguity'
  phase: 3
  track: research
  priority: high
  requires_gpu: false
  max_turns: 50
  estimated_wall_time_min: 65
  per_unit_rows: true
  milestone: 2026.11.784
  deliverable: results/experiment_9219_v784_decision_audit.json
  inference_substrate_class: no_model_load
  MODEL_SPECS: []
  gated_on:
  - upstream: exp9216-logit-admission
    artifact_field: admission_ready_score
    op: ==
    value: 1
  prior_failures:
  - experiment_id: exp9205-decision-audit
    verdict: complete_blocked_declared_gate
    addressed_by: V783 has a log-only scheduler skip and no scientific primary; the verdict here is Exp9207
      accounting. Exp9194 failed cold replay on missing entropy_reference_protocol and mutable V782 authority.
      Exp9208 now requires explicit protocol ownership and immutable historical fixtures, with original assertions
      and real cold CLI checks; Exp9209 separately qualifies the unexecuted lossless corpus reader and Exp9210
      the full validation union. This task runs only after its new exact prerequisite gates pass.
    retire_if_same_verdict: true
  - experiment_id: exp9190-decision-audit
    verdict: blocked_upstream_retired
    addressed_by: Retain the recorded prior failure. Exp9194 failed cold replay on missing entropy_reference_protocol
      and mutable V782 authority. Exp9208 now requires explicit protocol ownership and immutable historical
      fixtures, with original assertions and real cold CLI checks; Exp9209 separately qualifies the unexecuted
      lossless corpus reader and Exp9210 the full validation union. This task runs only after its new exact
      prerequisite gates pass.
    retire_if_same_verdict: true
  - experiment_id: exp9176-decision-audit
    verdict: 'GATE_BLOCK: upstream retired; no canonical honest_verdict artifact'
    addressed_by: Retain the recorded prior failure. Exp9194 failed cold replay on missing entropy_reference_protocol
      and mutable V782 authority. Exp9208 now requires explicit protocol ownership and immutable historical
      fixtures, with original assertions and real cold CLI checks; Exp9209 separately qualifies the unexecuted
      lossless corpus reader and Exp9210 the full validation union. This task runs only after its new exact
      prerequisite gates pass.
    retire_if_same_verdict: true
  - experiment_id: exp9162-energy-audit
    verdict: 'GATE_BLOCK: upstream retired; no canonical honest_verdict artifact'
    addressed_by: Retain the recorded prior failure. Exp9194 failed cold replay on missing entropy_reference_protocol
      and mutable V782 authority. Exp9208 now requires explicit protocol ownership and immutable historical
      fixtures, with original assertions and real cold CLI checks; Exp9209 separately qualifies the unexecuted
      lossless corpus reader and Exp9210 the full validation union. This task runs only after its new exact
      prerequisite gates pass.
    retire_if_same_verdict: true
  - experiment_id: exp9163-learning-audit
    verdict: 'GATE_BLOCK: upstream retired; no canonical honest_verdict artifact'
    addressed_by: Retain the recorded prior failure. Exp9194 failed cold replay on missing entropy_reference_protocol
      and mutable V782 authority. Exp9208 now requires explicit protocol ownership and immutable historical
      fixtures, with original assertions and real cold CLI checks; Exp9209 separately qualifies the unexecuted
      lossless corpus reader and Exp9210 the full validation union. This task runs only after its new exact
      prerequisite gates pass.
    retire_if_same_verdict: true
  - experiment_id: exp6980-spilled-energy-requalification
    verdict: complete_null_spilled_energy_requalification_retired
    addressed_by: Retain the recorded prior failure. Exp9194 failed cold replay on missing entropy_reference_protocol
      and mutable V782 authority. Exp9208 now requires explicit protocol ownership and immutable historical
      fixtures, with original assertions and real cold CLI checks; Exp9209 separately qualifies the unexecuted
      lossless corpus reader and Exp9210 the full validation union. This task runs only after its new exact
      prerequisite gates pass.
    retire_if_same_verdict: true
  prompt: |-
    CONTEXT:
    Work in {project_root} on {date}. V783 closed with two executed administrative tasks and twelve skips. Exp9194 full objects matched, but cold replay lacked entropy_reference_protocol and historical V782 consumers read mutable new authority. All four published readiness scores were zero. Preserve those failures. V784 qualifies protocol-owner replay and immutable historical inputs before the lossless corpus correction and complete validation. The original human panel remains exposed development; no independent generalization is claimed.
    EXISTING CODE TO READ FIRST:
    research-references.md (V784 planning scan); openspec/change-proposals/research-roadmap-v783-preserved-20261117.md; openspec/change-proposals/research-roadmap-v783-queue-preserved-20261117.yaml; CLAUDE.md; CODEX.md; ops/north-star.md; ops/e2e-test-plan.md; ops/exclusion_manifest.yaml; openspec/change-proposals/research-roadmap-vNEXT.md; openspec/capabilities/research-reporting/spec.md; scripts/experiment_template.py; python/carnot/reporting/primary_publication.py; python/carnot/reporting/current_work_receipt.py; python/carnot/reporting/logit_methods_9126.py; python/carnot/reporting/online_methods_9132.py; python/carnot/reporting/human_contrast_9141.py; python/carnot/verify/restricted_decision_audit_8210.py; results/experiment_9188_v782_static_energy.json; results/experiment_9189_v782_delayed_learning.json; results/experiment_9180_v782_contract.json
    TASK:
    Exp9219: Audit primary gains, entropy controls and human-label ambiguity. Deliver results/experiment_9219_v784_decision_audit.json. Implement scripts/experiments/experiment_9219_v784_decision_audit.py using existing library seams. MODEL_SPECS: []. Requested inference_substrate_class: no_model_load.
    Do not load an LLM. Set current_llm_invocations=0 and generated_tokens=0. Small energy-head training is CPU work. Keep historical model provenance in source_references.
    CONCRETE STEPS:
    1. Emit a flushed progress line at each phase boundary. Emit before and after every model load, generation, benchmark and subprocess that can take minutes. Print inside long loops. Supervise blocking calls with flushed heartbeats every 60 seconds. Keep every silence gap below 600 seconds. Write any file over about 200 lines in several tool calls of at most 200 lines. Send a progress message between writing calls. Never use one huge tool call.
    2. PRECONDITIONS: Authenticate exact required paths, hashes and gate fields before dependent work. Check imports, CPU/memory limits and private free disk. Distinguish missing_file, missing_field and observed zero. Missing external inputs produce terminal blocked evidence with gate_check_summary. Never repin historical evidence to mutable checkout code.
    3. Before implementation inspect callers with rg. Confirm REQ-REPORT-9219-V784 and SCENARIO-REPORT-9219-V784. Extend specs first. Write failing controls before code. Freeze units, exclusions, formulas and validation scope before outcomes.
    4. Check static_predictions_ready_score in Exp9217 and online_predictions_ready_score in Exp9218 independently. Audit either complete branch even when the other is absent or blocked. Record each failed operand and branch verdict. The task-level admission gate does not authorize missing predictions. No positive scientific effect is a launch prerequisite.
    5. Recompute the original two static Brier bounds using 10000 paired TEST-qid bootstrap draws, seed777121, tail0.05/2. Preserve all siblings and exact-half-coverage reselection inside each draw. Require both lower bounds>0 versus probability+length and donor, with no accepted-risk increase and defined accepted risk. Recompute original gauge controls and diagnostic masks.
    6. Recompute the original five future contrasts plus two retention bounds using 10000 paired whole-stream draws, seed778135, tail0.05/7. Require every future lower bound>0, retained Brier upper<=0.005, retained error upper<=0 and exact hard-exit recovery. Forty future qids and100 unique retention qids remain the effective supports; repeated evaluations are not extra data.
    7. For each eligible arm and fixed support, calculate mean Brier as (mean(p)-mean(y))^2 + (sd(p)-sd(y))^2 + 2*(sd(p)*sd(y)-cov(p,y)), using population moments and the original equal-qid weighting. Use the covariance expression for zero-variance cases; report correlation as unavailable then. Require agreement with direct mean squared error within1e-12 on fixed test vectors and measured rows. Do not clamp signed roundoff before testing the identity.
    8. Report those three terms for original static TEST, future groups and each unchanged retention panel. Preserve the V780 RMS-logit, tie-aware AUROC, Brier and class-count diagnostics. Show one-class and empty supports explicitly. Bootstrap diagnostic deltas at the original qid/stream unit. These intervals are descriptive; never label covariance deficit as causal information gain or irreducible truth uncertainty.
    9. Run independent recomputation from per-unit rows and tamper controls for signs, supports, sibling duplication and gauge formulas. Emit separate static_audit_complete_score, learning_audit_complete_score, energy_increment_score and learning_signal_score. Complete nulls are terminal. Exposed-panel positives remain conditional development evidence; independent_generalization_score stays0.
    10. Independently execute the design CES diagnostic from admitted token rows even if a fitted static/online branch is unavailable. Build F0 only from nonempty original A1-correct TRAIN responses, pooling their token entropies. Reference data, token weighting and all reference hashes seal before TEST reduction. Score sqrt(F0(mean H)*F0(max H)); larger means greater error suspicion. This is a rank score, not p(error); do not apply0.1/0.9 decision thresholds to it.
    11. Compare TEST qid-clustered AUROC against mean H, max H and the probability+length arm when its sealed predictions exist. Use10000 paired qid draws, seed782190, one-sided tail0.05/3, preserving siblings and ties. A one-class bootstrap draw makes that contrast interval unavailable; retain all draw counts and do not drop or redraw invalid draws. Report all planned contrasts; entropy_reference_signal_score=1 requires all three defined lower bounds>0. Missing comparator predictions block only that comparison and the aggregate signal gate; still retain the two available comparisons. Empty/one-class/reference-empty supports remain explicit, never zero-filled.
    12. Report per-answer lag1 entropy correlation when defined, answer length and tied-score fraction. No iid token theorem, self-generation claim or independent-generalization claim transfers from the paper. CES is a secondary registered ranking experiment; it cannot rescue either original static bound or any of the seven learning bounds. Do not tune CDF, clipping, thresholds or direction after seeing TEST.
    13. Preserve original V782 and V783 methods, assertions and failures. Exp9208 must qualify protocol-owner replay and historical input isolation; Exp9209 qualifies only the lossless corpus reader; Exp9210 must execute all original 674 and every affected obligation before model work. No retired upstream task is a dependency. Preserve the original 900 answers, scientific bounds and original protocol hashes.
    14. Read Exp9211 label_ambiguity_ready_score and its hash-bound reducer. For each eligible static/online contrast compute both ambiguity envelopes on identical original supports, with shared labels across repeated retention. Mark the sensitivity branch blocked if its authority is missing; still evaluate other eligible branches. Report robustness of the point contrast separately from sampling intervals. Never use ambiguous labels to refit, promote, tune or rescue failed primary bounds. Missing A2 remains visible.
    15. Require Exp9211 target_identifiability_ready_score=1 and its exact rubric hash before consuming target-identifiability rows, separately from its ambiguity reducer. Missing or failed authority blocks only this scope audit and any category claims, with an exact gate_check_summary operand. Report the evaluated target as original human factual-error annotation. Category metrics require authenticated original category labels and nonempty support; otherwise record unavailable with the exact missing evidence. Do not infer semantic illusion/confabulation from cross-model siblings or assign uncertainty using regex. This diagnostic cannot change the original primary target or bounds.
    16. Run owned unit tests and real child-process coverage with 100 percent owned statements and zero exclusions. Preserve original assertions. Run scoped Ruff check/format, strict mypy and scripts/check_spec_coverage.py with explicit test paths. Execute private real CLI success, refusal, tamper and cold replay outside the checkout without PYTHONPATH, following E2E-018/021. Learning/recovery changes also require E2E-020. Separate unrelated repository health from affected failures. Affected failures disqualify readiness.
    17. Stop measurement by 3300 seconds. Use one monotonic deadline for all descendants, including cleanup; do not spend the reserve on another measurement attempt. Reserve 1200 seconds for final verification and 300 seconds for publication within the 4800-second cap. Count each obligation once; distinguish elapsed time from its watchdog ceiling. Save unfinished units and censor reasons. Recompute claims from rows. Run scripts/adversarial_verify.py and scripts/verdict_row_consistency_lint.py --strict with heartbeats. Publish through primary_publication only after actual checker receipts pass. Reconcile openspec/, _bmad/traceability.md, ops/status.md and ops/changelog.md. Preserve historical primaries, quarantines and exclusions. Do not activate a queue or publish externally.
    REQUIRED ARTIFACT FIELDS:
    - honest_verdict: complete_* for terminal outcomes; partial_* only for retryable owned work; principle: A terminal prefix must agree with its verdict class.
    - verdict_class: positive | circular_positive | null | blocked | disqualified | partial; principle: External absence, retirement or gate failure is blocked. Only unfinished owned work is partial.
    - gate_check_summary: check, upstream, path, sha256, artifact_field, op, expected, observed, passed; principle: Every blocked result names its failed operand; missing evidence differs from an observed zero.
    - preconditions_checked: resource, available and actual check receipt; principle: Unavailable resources cannot produce invented measurements.
    - rows: unit_id, qid/group, sibling, arm/condition, seed, absolute metrics, numerator, denominator, status, exclusion_reason, censor_reason; principle: All comparative headlines must be reproducible from their independent units.
    - sample_size_budget: intended, attempted, completed, failed, censored, effective_independent_groups; principle: Siblings and repeated retention do not multiply independent sample size.
    - verifier_is_oracle: boolean; principle: Reusing executable truth as the verifier requires circular_positive, not an independent verifier win.
    - claim_scope: exposure_scope, independent_generalization_score=0 and allowed claims; principle: Exposed development and fixtures cannot close the independent-verifier gaps.
    - MODEL_SPECS: actual requested list and invocation status; principle: An upstream model reference is not current model work.
    - inference_substrate_class: actual class and requested_inference_substrate_class; principle: Use blocked_no_run for refusal before loading; attempted loads remain attempted loads.
    - inference_substrate: operation-accurate work description; principle: Forward work, generation and cached CPU reduction have different evidence requirements.
    - model_invocation_counts: current_llm_invocations, loads attempted/completed/failed, forward calls, generated_tokens; principle: Counters must agree with actual operation-specific receipts.
    - duration_s: measured wall time; principle: No artificial sleeping or duration padding may satisfy a substrate floor.
    - random_seed: fixed before outcomes; principle: A frozen seed permits replay without favorable seed selection.
    - reproducibility_checksum: inputs and configuration digest; principle: Changed inputs cannot pass as the original experiment.
    - source_references: original producer paths, SHA256 and imported fields; principle: Historical evidence retains its original authority.
    - code_config_hashes: owned implementation, tests and configuration; principle: Validation qualifies the executed revision.
    - current_work_receipts: operation-specific records; principle: Historical receipts cannot certify current inference.
    - validation_receipts: commands, logs, exit codes and coverage references; principle: Self-reported passing checks do not establish readiness.
    - required_checks_passed: boolean; principle: Readiness requires every affected check.
    - acceptance_gates: fixed conditions, each with a principle annotation; principle: Outcome-dependent thresholds cannot manufacture success.
    - static_audit_complete_score: integer 0/1; principle: A complete static null is still an auditable result.
    - learning_audit_complete_score: integer 0/1; principle: A blocked static branch cannot erase eligible learning evidence.
    - energy_increment_score: integer 0/1; principle: Both original static bounds and accepted-risk requirements must pass.
    - learning_signal_score: integer 0/1; principle: All seven original bounds plus exact recovery must pass.
    - branch_outcomes: static and learning verdict_class, gate_check_summary, support and bounds; principle: One blocked input cannot masquerade as a scientific null.
    - brier_components: mean_bias_squared, variance_mismatch, covariance_deficit, direct_brier, identity_residual; principle: Exact descriptive identities explain score changes without a new promotion threshold.
    - gauge_rows: original, constant and alternating shifts with all metrics; principle: Raw-energy sensitivity cannot be hidden by favorable gauge selection.
    - entropy_reference_complete_score: integer0/1; principle: Completeness requires all three declared comparisons, independently of a positive effect.
    - entropy_reference_signal_score: integer0/1; principle: All three registered qid-clustered ranking bounds must exceed zero; this never replaces primary scientific gates.
    - entropy_reference_rows: qid, sibling, token count, mean/max entropy, CES, comparator scores, exclusions and correlation; principle: Reference-distribution claims must be reproducible from individual answers.
    - entropy_reference_cdf: TRAIN IDs, A1 authority, token counts, sorted values or authenticated path and hash; principle: TEST labels cannot define a reference distribution.
    - label_ambiguity_rows: paired contrast, fixed support, observed-disagreement interval, missing-annotation interval and counts; principle: Conditional label robustness is separate from statistical confidence and primary benefit.
    Run command: cd {project_root} && JAX_PLATFORMS=cpu .venv/bin/python -u scripts/experiments/experiment_9219_v784_decision_audit.py --date {date}
    Do NOT push. Do NOT modify scripts/research_conductor.py.
  model: sonnet
  operator_override: '2026-05-29 operator directive (standing): versioned lineage continuation vs exp9205-decision-audit;
    Exp9194 failed cold replay on missing entropy_reference_protocol and mutable V782 authority. Exp9208 now
    requires explicit protocol ownership and immutable historical fixtures, with original assertions and real
    cold CLI checks; Exp9209 separately qualifies the unexecuted lossless corpus reader and Exp9210 the full
    validation union. This task runs only after its new exact prerequisite gates pass.'
  agent_type: claude
- id: exp9220-complete-cost
  title: 'Exp9220: Measure complete service costs and operation-matched hardware limits'
  phase: 4
  track: hardware
  priority: high
  requires_gpu: true
  max_turns: 50
  estimated_wall_time_min: 70
  per_unit_rows: true
  milestone: 2026.11.784
  deliverable: results/experiment_9220_v784_complete_cost.json
  inference_substrate_class: model_load_no_generation
  MODEL_SPECS:
  - hf_id: unsloth/Qwen3.8-27B-GGUF
    quantization: Q4_K_M
  model: sonnet
  gated_on:
  - upstream: exp9212-logit-runtime
    artifact_field: logit_runtime_ready_score
    op: ==
    value: 1
  prior_failures:
  - experiment_id: exp9206-complete-cost
    verdict: complete_blocked_declared_gate
    addressed_by: V783 has a log-only scheduler skip and no scientific primary; the verdict here is Exp9207
      accounting. Exp9194 failed cold replay on missing entropy_reference_protocol and mutable V782 authority.
      Exp9208 now requires explicit protocol ownership and immutable historical fixtures, with original assertions
      and real cold CLI checks; Exp9209 separately qualifies the unexecuted lossless corpus reader and Exp9210
      the full validation union. This task runs only after its new exact prerequisite gates pass.
    retire_if_same_verdict: true
  - experiment_id: exp9191-complete-cost
    verdict: blocked_upstream_retired
    addressed_by: Retain the recorded prior failure. Exp9194 failed cold replay on missing entropy_reference_protocol
      and mutable V782 authority. Exp9208 now requires explicit protocol ownership and immutable historical
      fixtures, with original assertions and real cold CLI checks; Exp9209 separately qualifies the unexecuted
      lossless corpus reader and Exp9210 the full validation union. This task runs only after its new exact
      prerequisite gates pass.
    retire_if_same_verdict: true
  - experiment_id: exp9177-complete-cost
    verdict: 'GATE_BLOCK: upstream retired; no canonical honest_verdict artifact'
    addressed_by: Retain the recorded prior failure. Exp9194 failed cold replay on missing entropy_reference_protocol
      and mutable V782 authority. Exp9208 now requires explicit protocol ownership and immutable historical
      fixtures, with original assertions and real cold CLI checks; Exp9209 separately qualifies the unexecuted
      lossless corpus reader and Exp9210 the full validation union. This task runs only after its new exact
      prerequisite gates pass.
    retire_if_same_verdict: true
  - experiment_id: exp9164-complete-cost
    verdict: 'GATE_BLOCK: upstream retired; no canonical honest_verdict artifact'
    addressed_by: Retain the recorded prior failure. Exp9194 failed cold replay on missing entropy_reference_protocol
      and mutable V782 authority. Exp9208 now requires explicit protocol ownership and immutable historical
      fixtures, with original assertions and real cold CLI checks; Exp9209 separately qualifies the unexecuted
      lossless corpus reader and Exp9210 the full validation union. This task runs only after its new exact
      prerequisite gates pass.
    retire_if_same_verdict: true
  prompt: |-
    CONTEXT:
    Work in {project_root} on {date}. V783 closed with two executed administrative tasks and twelve skips. Exp9194 full objects matched, but cold replay lacked entropy_reference_protocol and historical V782 consumers read mutable new authority. All four published readiness scores were zero. Preserve those failures. V784 qualifies protocol-owner replay and immutable historical inputs before the lossless corpus correction and complete validation. The original human panel remains exposed development; no independent generalization is claimed.
    EXISTING CODE TO READ FIRST:
    research-references.md (V784 planning scan); openspec/change-proposals/research-roadmap-v783-preserved-20261117.md; openspec/change-proposals/research-roadmap-v783-queue-preserved-20261117.yaml; CLAUDE.md; CODEX.md; ops/north-star.md; ops/e2e-test-plan.md; ops/exclusion_manifest.yaml; openspec/change-proposals/research-roadmap-vNEXT.md; openspec/capabilities/research-reporting/spec.md; scripts/experiment_template.py; python/carnot/reporting/primary_publication.py; python/carnot/reporting/current_work_receipt.py; python/carnot/reporting/complete_cost_9150.py; python/carnot/reporting/online_methods_9132.py; results/experiment_9183_v782_logit_runtime.json; results/experiment_9188_v782_static_energy.json; results/experiment_9189_v782_delayed_learning.json; research-hardware-wishlist.md; ops/known-issues.md (2026-09-09 hardware continuity correction)
    TASK:
    Measure complete service costs and operation-matched hardware limits. Deliver results/experiment_9220_v784_complete_cost.json. Implement scripts/experiments/experiment_9220_v784_complete_cost.py using existing library seams. MODEL_SPECS: [{{"hf_id": "unsloth/Qwen3.8-27B-GGUF", "quantization": "Q4_K_M"}}]. Requested inference_substrate_class: model_load_no_generation.
    Use frozen Qwen teacher-forced forwards only; generated_tokens=0. model_load_no_generation has a 2-second floor. A small-token canary would require model_bounded_generation (10s); a real generative run requires model_full_generation (60s). Do not add either operation here.
    CONCRETE STEPS:
    1. Emit a flushed progress line at each phase boundary. Emit before and after every model load, generation, benchmark and subprocess that can take minutes. Print inside long loops. Supervise blocking calls with flushed heartbeats every 60 seconds. Keep every silence gap below 600 seconds. Write any file over about 200 lines in several tool calls of at most 200 lines. Send a progress message between writing calls. Never use one huge tool call.
    2. PRECONDITIONS: Authenticate exact required paths, hashes and gate fields before dependent work. Check imports, CPU/memory limits and private free disk. Distinguish missing_file, missing_field and observed zero. Missing external inputs produce terminal blocked evidence with gate_check_summary. Never repin historical evidence to mutable checkout code. Before model work, resolve cached_current_model(gpu_index=0), as documented by cached_sota_pair() in the experiment template. Require the mandated local GGUF, embedded tokenizer, GPU0 lease, free VRAM and actual CUDA offload support. Pin GGUF and llama.cpp build hashes. GPU1 remains with its owner. Cache or resource failure blocks; no download or small-model headline fallback.
    3. Before implementation inspect callers with rg. Confirm REQ-REPORT-9220-V784 and SCENARIO-REPORT-9220-V784. Extend specs first. Write failing controls before code. Freeze units, exclusions, formulas and validation scope before outcomes.
    4. After the structured runtime gate passes, inventory exact source, qualification, admission, static and online readiness. Measure each eligible branch independently. A static service can run without learning. Scientific benefit is not a timing prerequisite. If no service is eligible, write terminal blocked evidence with exact gate_check_summary. No repeated pre-runtime inventory or invented timings.
    5. Use original first20 TEST qids/all60 siblings, three fresh processes and order seed776108. Match scientific probabilities within atol1e-5/rtol1e-4 and typed decisions exactly. Separate cold load, warm uncached raw-text verification and cached-feature head costs. Include tokenization, GPU forwards, transfers, statistics, transforms, scores and serialization.
    6. For available online service, time proposals, all guard scans, rejected updates, durable write/fsync, acknowledgment, final drain and recovery. Include the six arms and every intended unit. Preserve censored work and initialization costs. Reuse qualified heads and methods; do not redesign them during timing.
    7. Map measured operations to CPU, GPU and possible future native/FPGA kernels. Compute Amdahl ceilings only from actual nonoverlapping component times. Explain what fraction would need acceleration for the program's100x target and PRD10x target. Unknown fractions remain unknown. Power/energy stays unavailable unless measured integrals exist.
    8. Preserve KV260 supported Ising fabric and k_max<=5; spline updates require a different qualified kernel. Preserve PolarFire CPU-only evidence and GateMate physical/JTAG block. TSU/NPU/larger FPGA wishlist items need demonstrated operation fit and access. No repeated board probe or procurement is scheduled. Extropic projections are not local measurements.
    9. Keep the GateMate physical/JTAG continuation blocked under its unchanged physical prerequisite. The September9 operator record explicitly rejects another identical probe; no new attestation is invented. Preserve graduated KV260 and PolarFire scope. Map measured sparse-update, guard-scan, storage and Qwen components to compatible future kernels. Amdahl bounds require measured disjoint fractions; unknown fractions stay unknown. No board run, purchase, TSU access or power claim is authorized by a CPU/GPU timing.
    10. Preserve original V782 and V783 methods, assertions and failures. Exp9208 must qualify protocol-owner replay and historical input isolation; Exp9209 qualifies only the lossless corpus reader; Exp9210 must execute all original 674 and every affected obligation before model work. No retired upstream task is a dependency. Preserve the original 900 answers, scientific bounds and original protocol hashes.
    11. Run owned unit tests and real child-process coverage with 100 percent owned statements and zero exclusions. Preserve original assertions. Run scoped Ruff check/format, strict mypy and scripts/check_spec_coverage.py with explicit test paths. Execute private real CLI success, refusal, tamper and cold replay outside the checkout without PYTHONPATH, following E2E-018/021. Learning/recovery changes also require E2E-020. Separate unrelated repository health from affected failures. Affected failures disqualify readiness.
    12. Stop measurement by 3300 seconds. Use one monotonic deadline for all descendants, including cleanup; do not spend the reserve on another measurement attempt. Reserve 1200 seconds for final verification and 300 seconds for publication within the 4800-second cap. Count each obligation once; distinguish elapsed time from its watchdog ceiling. Save unfinished units and censor reasons. Recompute claims from rows. Run scripts/adversarial_verify.py and scripts/verdict_row_consistency_lint.py --strict with heartbeats. Publish through primary_publication only after actual checker receipts pass. Reconcile openspec/, _bmad/traceability.md, ops/status.md and ops/changelog.md. Preserve historical primaries, quarantines and exclusions. Do not activate a queue or publish externally.
    REQUIRED ARTIFACT FIELDS:
    - honest_verdict: complete_* for terminal outcomes; partial_* only for retryable owned work; principle: A terminal prefix must agree with its verdict class.
    - verdict_class: positive | circular_positive | null | blocked | disqualified | partial; principle: External absence, retirement or gate failure is blocked. Only unfinished owned work is partial.
    - gate_check_summary: check, upstream, path, sha256, artifact_field, op, expected, observed, passed; principle: Every blocked result names its failed operand; missing evidence differs from an observed zero.
    - preconditions_checked: resource, available and actual check receipt; principle: Unavailable resources cannot produce invented measurements.
    - rows: unit_id, qid/group, sibling, arm/condition, seed, absolute metrics, numerator, denominator, status, exclusion_reason, censor_reason; principle: All comparative headlines must be reproducible from their independent units.
    - sample_size_budget: intended, attempted, completed, failed, censored, effective_independent_groups; principle: Siblings and repeated retention do not multiply independent sample size.
    - verifier_is_oracle: boolean; principle: Reusing executable truth as the verifier requires circular_positive, not an independent verifier win.
    - claim_scope: exposure_scope, independent_generalization_score=0 and allowed claims; principle: Exposed development and fixtures cannot close the independent-verifier gaps.
    - MODEL_SPECS: actual requested list and invocation status; principle: An upstream model reference is not current model work.
    - inference_substrate_class: actual class and requested_inference_substrate_class; principle: Use blocked_no_run for refusal before loading; attempted loads remain attempted loads.
    - inference_substrate: operation-accurate work description; principle: Forward work, generation and cached CPU reduction have different evidence requirements.
    - model_invocation_counts: current_llm_invocations, loads attempted/completed/failed, forward calls, generated_tokens; principle: Counters must agree with actual operation-specific receipts.
    - duration_s: measured wall time; principle: No artificial sleeping or duration padding may satisfy a substrate floor.
    - random_seed: fixed before outcomes; principle: A frozen seed permits replay without favorable seed selection.
    - reproducibility_checksum: inputs and configuration digest; principle: Changed inputs cannot pass as the original experiment.
    - source_references: original producer paths, SHA256 and imported fields; principle: Historical evidence retains its original authority.
    - code_config_hashes: owned implementation, tests and configuration; principle: Validation qualifies the executed revision.
    - current_work_receipts: operation-specific records; principle: Historical receipts cannot certify current inference.
    - validation_receipts: commands, logs, exit codes and coverage references; principle: Self-reported passing checks do not establish readiness.
    - required_checks_passed: boolean; principle: Readiness requires every affected check.
    - acceptance_gates: fixed conditions, each with a principle annotation; principle: Outcome-dependent thresholds cannot manufacture success.
    - cost_inventory_ready_score: integer 0/1; principle: Complete availability accounting survives upstream failure.
    - service_cost_ready_score: integer 0/1; principle: Whole raw-text service requires actual matched outputs and complete clocks.
    - adaptation_cost_ready_score: integer 0/1; principle: Durable guard and feedback work belongs in learning cost.
    - cost_rows: unit, arm, cold/warm boundary, component clocks, complete clock and censor reason; principle: A head-only speedup cannot stand in for end-to-end benefit.
    - hardware_operation_rows: actual supported operation, measured fraction, Amdahl ceiling or unavailable reason; principle: Hardware recommendations require matching work.
    Run command: cd {project_root} && JAX_PLATFORMS=cpu CARNOT_FORCE_LIVE=1 .venv/bin/python -u scripts/experiments/experiment_9220_v784_complete_cost.py --date {date}
    Do NOT push. Do NOT modify scripts/research_conductor.py.
  agent_type: claude
  operator_override: '2026-05-29 operator directive (standing): versioned lineage continuation vs exp9206-complete-cost;
    Exp9194 failed cold replay on missing entropy_reference_protocol and mutable V782 authority. Exp9208 now
    requires explicit protocol ownership and immutable historical fixtures, with original assertions and real
    cold CLI checks; Exp9209 separately qualifies the unexecuted lossless corpus reader and Exp9210 the full
    validation union. This task runs only after its new exact prerequisite gates pass.'
- id: exp9221-capstone
  title: 'Exp9221: Reconcile fourteen outcomes and exact continuation boundaries'
  phase: 4
  track: research
  priority: high
  requires_gpu: false
  max_turns: 30
  estimated_wall_time_min: 45
  per_unit_rows: true
  milestone: 2026.11.784
  deliverable: results/experiment_9221_v784_capstone.json
  inference_substrate_class: no_model_load
  MODEL_SPECS: []
  prior_failures:
  - experiment_id: exp9207-capstone
    verdict: complete_blocked_declared_gate
    addressed_by: Unconditional reconciliation of fourteen NEW V784 outcomes, including replay qualification
      and independently authenticated label methods; preserve the V783 blocked science.
    retire_if_same_verdict: true
  - experiment_id: exp9193-capstone
    verdict: complete_blocked_declared_gate
    addressed_by: Retain the recorded prior failure. Unconditional reconciliation of fourteen NEW V784 outcomes,
      including replay qualification and independently authenticated label methods; preserve the V783 blocked
      science.
    retire_if_same_verdict: true
  - experiment_id: exp9179-capstone
    verdict: complete_blocked_declared_gate
    addressed_by: Retain the recorded prior failure. Unconditional reconciliation of fourteen NEW V784 outcomes,
      including replay qualification and independently authenticated label methods; preserve the V783 blocked
      science.
    retire_if_same_verdict: true
  - experiment_id: exp9165-capstone
    verdict: complete_blocked_declared_gate
    addressed_by: Retain the recorded prior failure. Unconditional reconciliation of fourteen NEW V784 outcomes,
      including replay qualification and independently authenticated label methods; preserve the V783 blocked
      science.
    retire_if_same_verdict: true
  prompt: |-
    CONTEXT:
    Work in {project_root} on {date}. V783 closed with two executed administrative tasks and twelve skips. Exp9194 full objects matched, but cold replay lacked entropy_reference_protocol and historical V782 consumers read mutable new authority. All four published readiness scores were zero. Preserve those failures. V784 qualifies protocol-owner replay and immutable historical inputs before the lossless corpus correction and complete validation. The original human panel remains exposed development; no independent generalization is claimed.
    EXISTING CODE TO READ FIRST:
    research-references.md (V784 planning scan); openspec/change-proposals/research-roadmap-v783-preserved-20261117.md; openspec/change-proposals/research-roadmap-v783-queue-preserved-20261117.yaml; CLAUDE.md; CODEX.md; ops/north-star.md; ops/e2e-test-plan.md; ops/exclusion_manifest.yaml; openspec/change-proposals/research-roadmap-vNEXT.md; openspec/capabilities/research-reporting/spec.md; scripts/experiment_template.py; python/carnot/reporting/primary_publication.py; python/carnot/reporting/current_work_receipt.py; python/carnot/reporting/v780_capstone.py; python/carnot/reporting/v780_contract.py; scripts/publication_gate.py; scripts/recurring_blocker_ledger.py; results/experiment_9165_v780_capstone.json; research-roadmap-next.yaml; python/carnot/reporting/v781_capstone.py; python/carnot/reporting/v781_capstone_evidence.py; python/carnot/reporting/v781_capstone_reduction.py; results/experiment_9179_v781_capstone.json
    TASK:
    Exp9221: Reconcile fourteen outcomes and exact continuation boundaries. Deliver results/experiment_9221_v784_capstone.json. Implement scripts/experiments/experiment_9221_v784_capstone.py using existing library seams. MODEL_SPECS: []. Requested inference_substrate_class: no_model_load.
    Do not load an LLM. Set current_llm_invocations=0 and generated_tokens=0. Small energy-head training is CPU work. Keep historical model provenance in source_references.
    CONCRETE STEPS:
    1. Emit a flushed progress line at each phase boundary. Emit before and after every model load, generation, benchmark and subprocess that can take minutes. Print inside long loops. Supervise blocking calls with flushed heartbeats every 60 seconds. Keep every silence gap below 600 seconds. Write any file over about 200 lines in several tool calls of at most 200 lines. Send a progress message between writing calls. Never use one huge tool call.
    2. PRECONDITIONS: Authenticate exact required paths, hashes and gate fields before dependent work. Check imports, CPU/memory limits and private free disk. Distinguish missing_file, missing_field and observed zero. Missing external inputs produce terminal blocked evidence with gate_check_summary. Never repin historical evidence to mutable checkout code.
    3. Before implementation inspect callers with rg. Confirm REQ-REPORT-9221-V784 and SCENARIO-REPORT-9221-V784. Extend specs first. Write failing controls before code. Freeze units, exclusions, formulas and validation scope before outcomes.
    4. Read both complete V784 contracts and all fourteen outcomes in order. Accept canonical primaries or explicitly identified scheduler skips; absent scientific artifacts stay absent. Resolve exact original source hashes. Preserve administrative accounting even when the scientific branches are blocked; no positive upstream gate applies to this task.
    5. Preserve V783 contract failure and scheduler evidence: Exp9194 had full-object agreement but failed private cold replay and historical consumers; all four gates stayed zero. Exp9195/9197 have scheduler refusal artifacts; ten other tasks have log-only skips. A capstone normalized complete_blocked_declared_gate is accounting, not a fabricated scientific primary verdict. Separate this from V782 four-executed/ten-skipped history.
    6. Reduce the two original static and seven original learning bounds from independently eligible rows. Keep CES completeness and three ranking bounds separate. Preserve original prediction/feedback roles, qid/stream denominators, recovery and source exposure. A positive secondary score does not promote the primary energy method.
    7. Account for lossless corpus-input parity with unchanged checker semantics, complete obligation execution, three source shards, cost boundaries and label-ambiguity completeness separately. Reconcile the two executed and twelve skipped V783 tasks without converting scheduling closure into findings. Keep missing field, missing file and observed zero distinct.
    8. Apply exact repeated-verdict retirement only to authenticated matching failed scope, with all four prior-failure fields. Keep unmeasured scientific hypotheses separate from retired execution routes. Never depend on a retired task ID. External absence is terminal blocked, never partial. An unconditional capstone can be administratively complete while its scientific verdict is blocked.
    9. Reconcile specs, traceability, architecture scope, status and changelog using actual outcomes. Retain historical FoVer publication gates and current_science_paper_ready=false for this exposed development panel. State exact reopen conditions and whether a later untouched human cohort is warranted. No queue activation, external publication or push.
    10. Preserve original V782 and V783 methods, assertions and failures. Exp9208 must qualify protocol-owner replay and historical input isolation; Exp9209 qualifies only the lossless corpus reader; Exp9210 must execute all original 674 and every affected obligation before model work. No retired upstream task is a dependency. Preserve the original 900 answers, scientific bounds and original protocol hashes.
    11. Account for the independent Exp9211 label method even if all model tasks skip. Preserve Exp9192 disqualified ARC compatibility as history; no ARC experiment is scheduled. Separate all14 ordered outcomes, new scan readiness, complete forward qualification, human science and two sensitivity branches. Keep exp9181/exp9182 retirement untouched; recommend exact retirement in the artifact for the conductor rather than editing the manifest.
    12. Report Exp9211 independent source/target audit even if Exp9208 and all model tasks fail. Recompute stable historical G1–G4 with scripts/publication_gate.py --json and keep that result separate from current_science_paper_ready=false. Preserve independently authenticated original protocols; missing current readiness never authorizes relabeling an old disqualification.
    13. Run owned unit tests and real child-process coverage with 100 percent owned statements and zero exclusions. Preserve original assertions. Run scoped Ruff check/format, strict mypy and scripts/check_spec_coverage.py with explicit test paths. Execute private real CLI success, refusal, tamper and cold replay outside the checkout without PYTHONPATH, following E2E-018/021. Learning/recovery changes also require E2E-020. Separate unrelated repository health from affected failures. Affected failures disqualify readiness.
    14. Stop measurement by 3300 seconds. Use one monotonic deadline for all descendants, including cleanup; do not spend the reserve on another measurement attempt. Reserve 1200 seconds for final verification and 300 seconds for publication within the 4800-second cap. Count each obligation once; distinguish elapsed time from its watchdog ceiling. Save unfinished units and censor reasons. Recompute claims from rows. Run scripts/adversarial_verify.py and scripts/verdict_row_consistency_lint.py --strict with heartbeats. Publish through primary_publication only after actual checker receipts pass. Reconcile openspec/, _bmad/traceability.md, ops/status.md and ops/changelog.md. Preserve historical primaries, quarantines and exclusions. Do not activate a queue or publish externally.
    REQUIRED ARTIFACT FIELDS:
    - honest_verdict: complete_* for terminal outcomes; partial_* only for retryable owned work; principle: A terminal prefix must agree with its verdict class.
    - verdict_class: positive | circular_positive | null | blocked | disqualified | partial; principle: External absence, retirement or gate failure is blocked. Only unfinished owned work is partial.
    - gate_check_summary: check, upstream, path, sha256, artifact_field, op, expected, observed, passed; principle: Every blocked result names its failed operand; missing evidence differs from an observed zero.
    - preconditions_checked: resource, available and actual check receipt; principle: Unavailable resources cannot produce invented measurements.
    - rows: unit_id, qid/group, sibling, arm/condition, seed, absolute metrics, numerator, denominator, status, exclusion_reason, censor_reason; principle: All comparative headlines must be reproducible from their independent units.
    - sample_size_budget: intended, attempted, completed, failed, censored, effective_independent_groups; principle: Siblings and repeated retention do not multiply independent sample size.
    - verifier_is_oracle: boolean; principle: Reusing executable truth as the verifier requires circular_positive, not an independent verifier win.
    - claim_scope: exposure_scope, independent_generalization_score=0 and allowed claims; principle: Exposed development and fixtures cannot close the independent-verifier gaps.
    - MODEL_SPECS: actual requested list and invocation status; principle: An upstream model reference is not current model work.
    - inference_substrate_class: actual class and requested_inference_substrate_class; principle: Use blocked_no_run for refusal before loading; attempted loads remain attempted loads.
    - inference_substrate: operation-accurate work description; principle: Forward work, generation and cached CPU reduction have different evidence requirements.
    - model_invocation_counts: current_llm_invocations, loads attempted/completed/failed, forward calls, generated_tokens; principle: Counters must agree with actual operation-specific receipts.
    - duration_s: measured wall time; principle: No artificial sleeping or duration padding may satisfy a substrate floor.
    - random_seed: fixed before outcomes; principle: A frozen seed permits replay without favorable seed selection.
    - reproducibility_checksum: inputs and configuration digest; principle: Changed inputs cannot pass as the original experiment.
    - source_references: original producer paths, SHA256 and imported fields; principle: Historical evidence retains its original authority.
    - code_config_hashes: owned implementation, tests and configuration; principle: Validation qualifies the executed revision.
    - current_work_receipts: operation-specific records; principle: Historical receipts cannot certify current inference.
    - validation_receipts: commands, logs, exit codes and coverage references; principle: Self-reported passing checks do not establish readiness.
    - required_checks_passed: boolean; principle: Readiness requires every affected check.
    - acceptance_gates: fixed conditions, each with a principle annotation; principle: Outcome-dependent thresholds cannot manufacture success.
    - reconciliation_ready_score: integer 0/1; principle: Administrative completeness stays separate from scientific success.
    - outcome_rows: fourteen ordered task IDs with actual paths, classes and gate operands; principle: No blocked branch disappears from the milestone record.
    - continuation_decisions: branch, disposition, evidence and exact reopen condition; principle: Renaming a failure cannot authorize a rerun.
    - publication_gates: historical G1-G4, unmet_gates and current_science_paper_ready=false; principle: Historical paper readiness cannot authorize a new exposed-panel claim.
    Run command: cd {project_root} && JAX_PLATFORMS=cpu .venv/bin/python -u scripts/experiments/experiment_9221_v784_capstone.py --date {date}
    Do NOT push. Do NOT modify scripts/research_conductor.py.
  model: sonnet
  operator_override: '2026-05-29 operator directive (standing): routine milestone transition vs exp9207-capstone;
    Unconditional reconciliation of fourteen NEW V784 outcomes, including replay qualification and independently
    authenticated label methods; preserve the V783 blocked science.'
  agent_type: claude
```
