# SOTA ingestion 2026-06-11: verifier-as-reward pivot

**Receipt fields**
- honest_verdict: `complete: sota_ingestion_verifier_as_reward_mapped`
- inference_substrate: `aggregation_from_upstream_artifacts`
- methods_mapped:
  - {arxiv_id: `2411.15124`, one_line: `Tulu 3 supplies the open SFT/DPO/RLVR recipe to adapt only after Carnot's verifier-certification gate is clean.`}
  - {arxiv_id: `2507.14843`, one_line: `Invisible Leash makes latent-vs-absent support the first gate before spending on RLVR or RFT.`}
  - {arxiv_id: `2505.14216`, one_line: `RL-vs-distillation separates pass@1 accuracy lift from genuine capability expansion and external-knowledge injection.`}
  - {arxiv_id: `2604.03128`, one_line: `Self-Distilled RLVR keeps verifier/environment reward as the update direction while using self-distillation for token credit magnitude.`}
  - {arxiv_id: `2203.14465`, one_line: `STaR gives the minimal generate-filter-finetune rationale loop for verifier-certified traces.`}
  - {arxiv_id: `2308.08998`, one_line: `ReST gives the offline generate-filter-improve cadence that fits Carnot's cached trace pools.`}
  - {arxiv_id: `2601.17223`, one_line: `VPRM turns deterministic rule verifiers into dense process rewards instead of opaque neural step judges.`}
  - {arxiv_id: `2605.10325`, one_line: `VPR extends dense verifiable process rewards to long-horizon agentic trajectories where sparse outcome rewards under-credit correct steps.`}
- strongest_for_next_roadmap:
  - `latent_vs_absent_precision_gate_before_rft`
  - `process_reward_weighted_sft_over_trace_certification`
  - `rest_star_offline_rft_iteration_with_label_ablation`
  - `self_distilled_rlvr_only_after_external_reward_direction`

**Fresh-pass provenance**

Read the local reward/distillation material in `research-studying.md`,
`research-references.md`, `docs/research-notes/verifier-as-self-improvement-reward-scoping.md`,
`docs/research-notes/step-level-process-reward-scoping.md`, and the current `.377`
RFT artifacts. The current ARC RFT corpus gate is not green: Exp 4077 reports
`certification_precision=0.6818` and `certification_recall=1.0000`, so the hard
demo-perfect trace label is below the 0.85 precision gate. Exp 4078 is therefore
blocked on missing corpora. Exp 4080 reproduces the Sudoku pipeline-sanity
direction, but that is not enough to rescue the ARC trace-level RFT gate.

Ran the required discovery helpers:

- `python scripts/sweep_clusters.py 0 --max-results 8`
- `python scripts/sweep_clusters.py 1 --max-results 8`
- `python scripts/sweep_semscholar.py "verifier certified RFT RLVR process reward distillation STaR ReST self training" --limit 8`
- `python scripts/sweep_semscholar.py "Tulu 3 RLVR verifiable rewards reinforcement fine tuning distillation" --limit 8`

Semantic Scholar returned no IDs for the broad verifier-certified RFT query and
eight adjacent IDs for the Tulu/RLVR query; none displaced the seed set after
primary-source verification. Low-concurrency WebSearch/WebFetch then verified
the eight mapped primary arXiv pages. The `/deep-research` loop was not invoked.

---

## Current RFT pipeline anchor

The `.377` pipeline is already the right experimental shell: it isolates the
verifier label by comparing RFT-correct against RFT-ablation from the same
generator pool, with gold-SFT as the oracle upper-bound arm. The binding problem
is label quality. A verifier-certified corpus whose label means "demo-perfect"
does not yet imply held-out gold correctness well enough for training. Therefore
the next roadmap should improve the reward signal before launching another train,
not simply rerun the blocked corpus builder.

---

## Verifier-certified RFT over the current RFT pipeline

**Method:** Tulu 3 (`arXiv:2411.15124`, https://arxiv.org/abs/2411.15124) is the
open post-training recipe to copy at the systems level: SFT, preference tuning,
and a final RLVR stage over verifiable tasks. The important lesson for Carnot is
not the exact model family; it is the complete, reproducible post-training stack
around verifiable answers.

**Implementation over current RFT pipeline:** Keep the three-arm `.377` design
but treat Tulu 3 as the trainer/protocol template after the corpus gate passes.
The immediate work is not a full RLVR run; it is to replace the poisoned
demo-perfect hard label with a high-precision certifier. Once precision clears
0.85 at usable recall, train the same base on RFT-correct, RFT-ablation, and
gold-SFT with identical LoRA settings, then evaluate held-out induction.

**Pitfalls / where it fails:** Tulu-style RLVR assumes the verifiable reward is
the task objective, or close enough to it. Carnot's current hard ARC label fails
that assumption at 0.6818 precision. Training on it would test label noise, not
verifier-as-reward.

---

## RLVR / Tulu 3 open post-training recipe

**Method:** Self-Distilled RLVR (`arXiv:2604.03128`, https://arxiv.org/abs/2604.03128)
is the better fine-grained variant to hold in reserve. It argues that privileged
self-distillation alone leaks information and destabilizes long training, while
RLVR should keep the reliable reward direction and self-distillation should shape
token-level update magnitudes.

**Implementation over current RFT pipeline:** Only add this after the external
verifier reward direction is clean. For Carnot, "direction" means the label
contrast RFT-correct > RFT-ablation on held-out tasks. The self-distillation part
would then supply token/step weights from the model's own traces, while the
verifier or execution outcome supplies the sign of the update.

**Pitfalls / where it fails:** If Carnot uses self-distillation to compensate for
a bad verifier label, it recreates the leakage/instability failure mode. The
reward direction must be external and measurable first.

---

## Invisible Leash latent-vs-absent diagnostic

**Method:** The Invisible Leash (`arXiv:2507.14843`, https://arxiv.org/abs/2507.14843)
warns that RLVR often sharpens what is already in the base model's support
instead of creating genuinely absent solutions. RL vs. Distillation
(`arXiv:2505.14216`, https://arxiv.org/abs/2505.14216) sharpens the fork:
RLVR can improve pass@1 while capability ceilings stay flat; distillation raises
capability only when it introduces new knowledge or stronger traces.

**Implementation over current RFT pipeline:** Add a support/headroom gate before
any `.378` train. For each base and corpus, measure cold pass@1, pass@k/oracle,
demo-perfect coverage, gold correctness of demo-perfect candidates, and whether
the desired transform appears anywhere in the same generated pool. If the correct
program is absent from the base's support, route to a stronger local base or
external verified traces; do not expect RFT/RLVR to invent it.

**Pitfalls / where it fails:** A positive RFT delta can be only an easy-case
pass@1 sharpening result. The roadmap must require both the verifier-label
ablation contrast and a support diagnostic so a small improvement is not
misread as capability expansion.

---

## Process-reward distillation

**Method:** Verifiable Process Reward Models (`arXiv:2601.17223`,
https://arxiv.org/abs/2601.17223) and Verifiable Process Rewards
(`arXiv:2605.10325`, https://arxiv.org/abs/2605.10325) are the closest match to
Carnot's strongest local evidence. They use deterministic rule or oracle checks
as dense intermediate rewards instead of relying only on sparse outcome rewards.
This fits the local scoping result: hard trace certification was weak, but dense
process-reward aggregation ranked correct traces above incorrect ones.

**Implementation over current RFT pipeline:** Convert the `.377` corpus builder
from a hard demo-perfect label into step/action-level reward records. For ARC,
that means reward features for demo fit, state validity, action legality,
execution consistency, exact state/action equivalence, hidden-test success when
available, and verifier abstention. The first training arm should be
process-reward-weighted SFT before GRPO/PPO.

**Pitfalls / where it fails:** Local validity does not guarantee final
correctness. A dense process reward can produce valid-looking but unproductive
program edits if progress-to-solution is not part of the reward. This is the
same process-vs-outcome gap that blocked hard trace RFT, softened rather than
eliminated.

---

## RFT / STaR / ReST self-training

**Method:** STaR (`arXiv:2203.14465`, https://arxiv.org/abs/2203.14465) gives
the minimal generate-filter-finetune loop over rationales that lead to correct
answers. ReST (`arXiv:2308.08998`, https://arxiv.org/abs/2308.08998) generalizes
that into an offline generate-filter-improve cadence with reusable batches.

**Implementation over current RFT pipeline:** Use STaR/ReST as the iteration
schedule, not as the label definition. A `.378` loop should generate a fixed
pool, score with the improved verifier/process-reward record, train
RFT-correct/RFT-ablation/gold-SFT arms, evaluate held-out tasks, then refresh
the pool only if the ablation contrast is positive. Reuse old batches as ReST
does so the conductor can resume rather than restart.

**Pitfalls / where it fails:** STaR filters on correctness; if Carnot's filter is
only demo-perfect and not gold-predictive, the loop distills confounded generator
behavior. ReST's offline batches can also go stale after policy drift, so each
iteration needs a held-out recalibration of certification precision.

---

## Adjacent citations retained but not first targets

- FoVer (`arXiv:2505.15960`, https://arxiv.org/abs/2505.15960) remains the
  formal-verification data-synthesis anchor for step labels, but the current
  pivot's blocker is not how to make step labels; it is how to turn them into
  outcome-useful training signal.
- ThinkPRM (`arXiv:2504.16828`, https://arxiv.org/abs/2504.16828) remains the
  generative PRM comparator. It should be a benchmark/control, not the first
  local implementation, because the next Carnot step needs a cheap deterministic
  reward signal.

---

## Bottom line for the .378 roadmap

1. **Strongest next gate:** `latent_vs_absent_precision_gate_before_rft`. Do not
   launch another ARC RFT train from the current demo-perfect label. First prove
   certification precision >= 0.85 and measure whether correct programs are
   latent in the base's same-pool support.
2. **Strongest method change:** `process_reward_weighted_sft_over_trace_certification`.
   Replace hard trace certification with dense process-reward weighting, because
   the local evidence says the soft aggregate carries outcome signal while hard
   trace labels poison.
3. **Training loop once the gate passes:** `rest_star_offline_rft_iteration_with_label_ablation`.
   Keep the three-arm deconfounded design and use ReST/STaR as the generate,
   filter, train, evaluate cadence.
4. **Hold RLSD for later:** `self_distilled_rlvr_only_after_external_reward_direction`.
   Self-distilled RLVR is attractive only after RFT-correct beats RFT-ablation;
   before that it risks stabilizing the wrong signal.
5. **Honest null to preserve:** if the label contrast stays flat, the conclusion
   is not "RFT failed"; it is "the current verifier label adds no training signal
   beyond codex-distillation." That null is decision-grade and should route `.378`
   toward stronger local base support or external verified traces.
