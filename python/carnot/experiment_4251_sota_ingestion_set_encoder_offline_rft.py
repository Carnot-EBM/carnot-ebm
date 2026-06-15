"""Exp 4251 SOTA ingestion for the set-encoder and offline RFT .394 plan.

Spec refs: REQ-REPORT-4251, SCENARIO-REPORT-4251.

This module writes a planning artifact, not a benchmark result. It closes the
`.393 planning sweep` into a concrete SOTA-to-experiment mapping after Exp
4245 produced a clean ARC Set-Encoder beats-vote win, while Exp 4246 and Exp
4248 left code robustness and offline reward-weighted training blocked.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import json
from pathlib import Path
from typing import Any


REQUIRED_ARTIFACT_FIELDS = frozenset(
    {
        "honest_verdict",
        "methods_mapped",
        "flagged_for_v394",
        "field_principles",
    }
)
REQUIRED_METHOD_FIELDS = frozenset(
    {
        "name",
        "arxiv_id_or_url",
        "url",
        "carnot_stack_mapping",
        "a3_arc_mapping",
        "a4_code_mapping",
        "b2_reward_mapping",
        "failure_mode",
        "experiment_mapping",
    }
)
TERMINAL_PREFIXES = ("complete:", "success:", "passed:", "shipped:", "blocked_")
DEFAULT_HONEST_VERDICT = "complete: sota_ingestion_set_encoder_offline_rft_mapped_v394"
DEFAULT_FLAGGED_FOR_V394 = "agglm_synthesize_corrected_grid_from_set_encoder_evidence_v394"

FIELD_PRINCIPLES = {
    "honest_verdict": "Terminal-prefixed. Records ingestion completed with verifiable citations.",
    "methods_mapped": (
        "Each method MUST carry a real arXiv ID/URL; an ingestion note without "
        "verifiable citations is treated as fabrication (adversarial_verify "
        "discipline)."
    ),
    "flagged_for_v394": (
        "Closes discover->ingest->plan: names the strongest method for the next "
        "planner, conditioned on the A3/A4/B2 outcomes."
    ),
}

VERIFIED_SOURCE_URLS = {
    "2505.15433": "https://arxiv.org/abs/2505.15433",
    "2509.06870": "https://arxiv.org/abs/2509.06870",
    "2605.26172": "https://arxiv.org/abs/2605.26172",
    "2510.14913": "https://arxiv.org/abs/2510.14913",
    "2504.11343": "https://arxiv.org/abs/2504.11343",
    "2502.11026": "https://arxiv.org/abs/2502.11026",
    "2506.10947": "https://arxiv.org/abs/2506.10947",
    "2512.15146": "https://arxiv.org/abs/2512.15146",
}
NOTE_REQUIRED_SOURCE_CITATIONS = frozenset(f"arXiv:{source}" for source in VERIFIED_SOURCE_URLS)

DEFAULT_METHODS_MAPPED = [
    {
        "name": "Set-LLM permutation-invariant set architecture",
        "arxiv_id_or_url": "2505.15433",
        "url": "https://arxiv.org/abs/2505.15433",
        "carnot_stack_mapping": (
            "Maps the proven ARC set-encoder direction onto a stronger "
            "permutation-invariant LLM architecture: set-aware masking and "
            "position encodings for unordered candidate pools."
        ),
        "a3_arc_mapping": (
            "Exp 4245 already shows the CPU-fast DeepSets set encoder beat vote "
            "by 0.4423076923 with CI95 excluding zero. Set-LLM is therefore a "
            "scale-up architecture, not a reason to re-litigate set awareness."
        ),
        "a4_code_mapping": (
            "Exp 4246 blocked before a distinct second code corpus, so Set-LLM "
            "does not yet prove cross-domain robustness. Keep code as a separate "
            "candidate-corpus discovery gate."
        ),
        "b2_reward_mapping": (
            "Set-LLM is orthogonal to the blocked offline reward-weighted SFT "
            "path. Do not use the ARC win to claim reward-training progress."
        ),
        "failure_mode": (
            "Set-LLM evidence is multiple-choice/set-text, not ARC grid synthesis. "
            "A direct .394 use still needs exact grid-match validation and a "
            "larger ARC pool."
        ),
        "experiment_mapping": (
            "Use it as the high-capacity selector baseline over a bigger ARC pool, "
            "paired with the current DeepSets model to separate architecture "
            "scale from data growth."
        ),
    },
    {
        "name": "AggLM review-reconcile-synthesize aggregation",
        "arxiv_id_or_url": "2509.06870",
        "url": "https://arxiv.org/abs/2509.06870",
        "carnot_stack_mapping": (
            "Maps to a generative reconciler that reviews the Set-Encoder-ranked "
            "candidate family, reconciles conflicting local evidence, and "
            "synthesizes a corrected grid rather than only selecting a cached one."
        ),
        "a3_arc_mapping": (
            "Because Exp 4245 proved set-aware selection can beat vote, the next "
            "ARC lever is not another selector-only rerun. AggLM scales the win "
            "by using the selected evidence to synthesize the corrected grid."
        ),
        "a4_code_mapping": (
            "The code replication block keeps AggLM's immediate scope on ARC. A "
            "code version should wait until Exp 4246 has a source-distinct hidden "
            "label candidate pool."
        ),
        "b2_reward_mapping": (
            "AggLM does not unblock LoRA training. If its generated grids become "
            "training data later, the B2 same-base random-label control still "
            "has to remain intact."
        ),
        "failure_mode": (
            "Generative reconciliation increases fabrication risk. It must be "
            "gated by exact ARC grid match, matched selector-only controls, and "
            "an explicit no-synthesis baseline."
        ),
        "experiment_mapping": (
            "Flag for .394: Set-Encoder evidence in, AggLM-style corrected grid "
            "out, exact grid validation, SCOPE per-region ablation, and vote plus "
            "selector-only controls on a bigger pool."
        ),
    },
    {
        "name": "ARBITER conservative evidence over vote prior",
        "arxiv_id_or_url": "2605.26172",
        "url": "https://arxiv.org/abs/2605.26172",
        "carnot_stack_mapping": (
            "Maps to wrong-majority recovery: treat vote as a prior, add bounded "
            "same-pool evidence, and only override when evidence clears the "
            "pre-registered margin."
        ),
        "a3_arc_mapping": (
            "Exp 4245's margin override beat vote by 0.4230769231, so ARBITER "
            "becomes the framing and diagnostic for which basins the set encoder "
            "recovers, not the primary architecture."
        ),
        "a4_code_mapping": (
            "No second code corpus means ARBITER-style basin recovery has not "
            "been replicated outside the ARC pool in this milestone."
        ),
        "b2_reward_mapping": (
            "ARBITER is a test-time aggregation method. It does not provide the "
            "offline reward-weighted training evidence B2 was gated to test."
        ),
        "failure_mode": (
            "Hidden-state ARBITER variants are gray-box. Carnot's .394 use should "
            "stay on sampled outputs, local grid evidence, and calibrated margins "
            "to preserve the oracle-distinct claim."
        ),
        "experiment_mapping": (
            "Report basin-level wrong-majority recovery for the AggLM arm: vote "
            "prior, Set-Encoder margin, ARBITER-style evidence sum, and exact "
            "recovery count."
        ),
    },
    {
        "name": "Budget-aware discriminative verification hybrid",
        "arxiv_id_or_url": "2510.14913",
        "url": "https://arxiv.org/abs/2510.14913",
        "carnot_stack_mapping": (
            "Maps to the hybrid policy already supported by Exp 4245: keep "
            "self-consistency as a cheap prior, then spend discriminative "
            "verification only where the learned margin can change the answer."
        ),
        "a3_arc_mapping": (
            "The A3 win and matched_control_delta=0.4807692308 justify treating "
            "the set encoder as value-added over budget alone. The .394 plan "
            "should now cost-normalize that win."
        ),
        "a4_code_mapping": (
            "The blocked A4 result means no robustness claim should be made for "
            "code until a distinct candidate pool exists."
        ),
        "b2_reward_mapping": (
            "The hybrid verifier is a selection-time method, not reward training. "
            "Keep it separate from B2's execution-oracle reward axis."
        ),
        "failure_mode": (
            "A hybrid can hide verifier weakness behind the vote prior. The .394 "
            "report must keep selector-only, vote-only, and matched-budget "
            "controls visible."
        ),
        "experiment_mapping": (
            "Add fixed-budget accounting to the AggLM run: vote, Set-Encoder, "
            "margin-triggered hybrid, and generative reconciler at matched "
            "candidate and token budgets."
        ),
    },
    {
        "name": "RAFT rejection-sampled reward-positive SFT",
        "arxiv_id_or_url": "2504.11343",
        "url": "https://arxiv.org/abs/2504.11343",
        "carnot_stack_mapping": (
            "Maps to the offline reward pivot: train on positively rewarded "
            "samples with a simple SFT-style objective before attempting a more "
            "complex online RL loop."
        ),
        "a3_arc_mapping": (
            "A3 does not depend on reward training. RAFT should not be used to "
            "reinterpret the ARC Set-Encoder win."
        ),
        "a4_code_mapping": (
            "A4's missing second code pool limits code robustness evidence, but "
            "the stable A/B/C corpora remain the right source for a bounded RAFT "
            "smoke once the harness actually trains."
        ),
        "b2_reward_mapping": (
            "B2 blocked because harness_smoke_passed=false, steps_run=0, and "
            "trainable_param_count=0. RAFT remains the correct offline form only "
            "after the harness proves real optimizer steps and loss movement."
        ),
        "failure_mode": (
            "Reward-positive SFT can learn dataset and generator artifacts. It "
            "must be paired with the same-base random-label ablation."
        ),
        "experiment_mapping": (
            "For .394, keep RAFT as an owed B-path gate: fix adapter attachment, "
            "prove at least 20 real steps, then compare verifier-certified Arm A "
            "against same-generator random Arm B."
        ),
    },
    {
        "name": "VAR offline reward-weighted alignment",
        "arxiv_id_or_url": "2502.11026",
        "url": "https://arxiv.org/abs/2502.11026",
        "carnot_stack_mapping": (
            "Maps to the exact loss-shape refinement for B2: an offline "
            "reward-driven re-weighted SFT objective rather than live LoRA RL."
        ),
        "a3_arc_mapping": (
            "VAR is not the strongest next ARC method because A3 already landed "
            "without training the base model. It belongs to the owed reward path."
        ),
        "a4_code_mapping": (
            "Code replication did not run on a second corpus, so VAR cannot be "
            "claimed as a robustness amplifier yet."
        ),
        "b2_reward_mapping": (
            "The blocked B2 result says the next refinement is mechanical first: "
            "make the non-Qwen offline harness attach LoRA to supported modules, "
            "then run reward-weighted SFT with A-vs-B controls."
        ),
        "failure_mode": (
            "A weighted SFT loss can look stable while learning spurious labels. "
            "The random-label control and gold Arm C must remain pass/fail gates."
        ),
        "experiment_mapping": (
            "Use VAR as the B2 loss implementation once training is real: fixed "
            "step budget, reward weights preserved, A minus B pass@1 CI95, and "
            "gold-control non-regression."
        ),
    },
    {
        "name": "Spurious Rewards same-base random-label control",
        "arxiv_id_or_url": "2506.10947",
        "url": "https://arxiv.org/abs/2506.10947",
        "carnot_stack_mapping": (
            "Maps to the de-confounder for any reward-weighted training claim: "
            "Arm A verifier labels must beat same-generator random labels on the "
            "same non-Qwen base."
        ),
        "a3_arc_mapping": (
            "This control is not needed to accept the A3 selection win, because "
            "A3 is test-time reranking with verifier_is_oracle=false, not model "
            "training from rewards."
        ),
        "a4_code_mapping": (
            "The missing second code corpus leaves robustness unresolved, but it "
            "does not weaken the requirement that reward-training controls use "
            "the same base and same generator."
        ),
        "b2_reward_mapping": (
            "B2 could not run, so no A-vs-B inference is available. The .394 "
            "reward task must preserve the same-base random-label control before "
            "claiming verifier reward signal."
        ),
        "failure_mode": (
            "Spurious reward effects are model-family dependent. A single Qwen "
            "success would be invalid here; the trainable base must stay non-Qwen."
        ),
        "experiment_mapping": (
            "Make the B-path acceptance gate explicit: verifier-certified Arm A "
            "must beat same-generator random Arm B with CI95 excluding zero, with "
            "Qwen training forbidden."
        ),
    },
    {
        "name": "SCOPE fine-grained per-region evidence",
        "arxiv_id_or_url": "2512.15146",
        "url": "https://arxiv.org/abs/2512.15146",
        "carnot_stack_mapping": (
            "Maps to per-region ARC evidence for candidate-grid disagreement: "
            "use local confidence and subgroup signals so a minority-correct "
            "grid has evidence beyond final-answer frequency."
        ),
        "a3_arc_mapping": (
            "Exp 4245's Set-Encoder win is the right substrate for SCOPE-style "
            "features. The .394 reconciler should explain which grid regions "
            "support the synthesized answer."
        ),
        "a4_code_mapping": (
            "SCOPE's ARC use does not resolve the missing code corpus. Code needs "
            "its own trace/evidence source before a comparable region signal "
            "exists."
        ),
        "b2_reward_mapping": (
            "SCOPE can shape future pseudo-labels, but the blocked B2 result means "
            "it must not be promoted into training until the offline harness runs."
        ),
        "failure_mode": (
            "Fine-grained pseudo-labels can amplify confirmation bias if they are "
            "not tied to exact ARC grid labels and local disagreement evidence."
        ),
        "experiment_mapping": (
            "Pair AggLM synthesis with a SCOPE per-region evidence ablation: "
            "synthesis with and without local evidence, exact grid validation, "
            "and wrong-majority recovery counts."
        ),
    },
]

NOTE_MARKDOWN = """# SOTA ingestion 2026-06-15: set-encoder and offline RFT map for .394

**Artifact fields**
- honest_verdict: `complete: sota_ingestion_set_encoder_offline_rft_mapped_v394`
  - principle: Terminal-prefixed. Records ingestion completed with verifiable citations.
- methods_mapped:
  - {name: `Set-LLM permutation-invariant set architecture`, arxiv_id_or_url: `2505.15433`, url: `https://arxiv.org/abs/2505.15433`}
  - {name: `AggLM review-reconcile-synthesize aggregation`, arxiv_id_or_url: `2509.06870`, url: `https://arxiv.org/abs/2509.06870`}
  - {name: `ARBITER conservative evidence over vote prior`, arxiv_id_or_url: `2605.26172`, url: `https://arxiv.org/abs/2605.26172`}
  - {name: `Budget-aware discriminative verification hybrid`, arxiv_id_or_url: `2510.14913`, url: `https://arxiv.org/abs/2510.14913`}
  - {name: `RAFT rejection-sampled reward-positive SFT`, arxiv_id_or_url: `2504.11343`, url: `https://arxiv.org/abs/2504.11343`}
  - {name: `VAR offline reward-weighted alignment`, arxiv_id_or_url: `2502.11026`, url: `https://arxiv.org/abs/2502.11026`}
  - {name: `Spurious Rewards same-base random-label control`, arxiv_id_or_url: `2506.10947`, url: `https://arxiv.org/abs/2506.10947`}
  - {name: `SCOPE fine-grained per-region evidence`, arxiv_id_or_url: `2512.15146`, url: `https://arxiv.org/abs/2512.15146`}
  - principle: Each method MUST carry a real arXiv ID/URL; an ingestion note without verifiable citations is treated as fabrication (adversarial_verify discipline).
- flagged_for_v394: `agglm_synthesize_corrected_grid_from_set_encoder_evidence_v394`
  - principle: Closes discover->ingest->plan: names the strongest method for the next planner, conditioned on the A3/A4/B2 outcomes.

## Fresh-pass provenance

Read `research-references.md` `.393 planning sweep` and `.392 planning sweep`,
`research-studying.md`, `results/experiment_4245_arc_set_encoder_beats_vote.json`,
`results/experiment_4246_code_oracle_distinct_replication.json`,
`results/experiment_4247_verifier_reward_offline_harness_retire_livelora.json`,
and `results/experiment_4248_verifier_as_reward_offline_3arm.json`.

Reliable-channel helper pass, not `/deep-research`:
- `python3 scripts/sweep_clusters.py 0 --max-results 8`
- `python3 scripts/sweep_clusters.py 1 --max-results 8`
- `python3 scripts/sweep_semscholar.py "Set-LLM permutation invariant LLM majority vote ARC verifier aggregation" --limit 8`
- `python3 scripts/sweep_semscholar.py "reward weighted SFT verifier reward offline RAFT VAR spurious rewards" --limit 8`

The cluster helper emitted the same two broadened arXiv API URLs used by the
reliable discovery channel. Semantic Scholar returned HTTP 429 for both
focused queries, so no Semantic-Scholar-only promotion is claimed. Low
concurrency WebSearch/WebFetch verified arXiv:2505.15433, arXiv:2509.06870,
arXiv:2605.26172, arXiv:2510.14913, arXiv:2504.11343, arXiv:2502.11026,
arXiv:2506.10947, and arXiv:2512.15146.

## Exp 4245 ARC A3, Exp 4246 code A4, and Exp 4248 offline B2 read

Exp 4245 produced the first clean ARC oracle-distinct win:
`headline_outcome=arc_oracle_distinct_set_encoder_beats_vote`,
`set_encoder_minus_vote_delta=0.4423076923`, CI95 `[0.3076923077, 0.5961538462]`,
`set_encoder_minus_vote_delta` excludes zero, `margin_override_minus_vote=0.4230769231`,
`matched_control_delta=0.4807692308`, `oracle_at_k=0.8269230769`,
`held_out_task_n=52`, and `oracle_distinct_beats_vote=true`. The read is
decision-grade for ARC selection: the grown-pool set encoder beat vote and beat
the matched no-verifier control.

Exp 4246 did not replicate or refute the code oracle-distinct win. It ended as
`blocked_code_second_corpus_missing` because no cached second code candidate
pool was both hidden-label viable and source-distinct from Exp 4233. Therefore
code remains a robustness read, not a negative result against the ARC A3 win.

Exp 4248 did not run the offline reward-weighted A-vs-B comparison. It ended as
`blocked_gate_check_failed` because Exp 4247 reported `harness_smoke_passed=false`,
`steps_run=0`, `trainable_param_count=0`, and no loss movement. The B2 pivot is
still owed: fix the harness first, then run same-base Arm A verifier labels
against Arm B random labels.

## SOTA -> experiment mapping

## Set-LLM: scale the proven set architecture

**Method/source:** Set-LLM, arXiv:2505.15433
(https://arxiv.org/abs/2505.15433), adapts pretrained LLMs for permutation
invariant mixed set-text inputs.

**Carnot stack mapping:** Use it as the high-capacity version of the Exp 4245
DeepSets-style selector after the CPU-fast set encoder proved the mechanism.

**A3 ARC mapping:** Exp 4245 already landed the set-aware selector win. Set-LLM
is a scale-up baseline over a bigger pool, not the strongest new .394 idea.

**A4 code mapping:** Exp 4246 was blocked by missing corpus evidence, so Set-LLM
does not make a cross-domain code robustness claim.

**B2 reward mapping:** Orthogonal to offline reward-weighted SFT.

**Failure mode:** Multiple-choice set-text evidence does not by itself solve
free-form ARC grid synthesis.

**Experiment mapping:** Compare DeepSets selector, Set-LLM-style selector, vote,
and matched control on a bigger ARC pool.

## AggLM: synthesize a corrected grid

**Method/source:** AggLM, arXiv:2509.06870
(https://arxiv.org/abs/2509.06870), trains an aggregator to review, reconcile,
and synthesize a final answer from candidate solutions.

**Carnot stack mapping:** Add a generative reconciler after the Set-Encoder. It
should read the ranked candidates and SCOPE per-region evidence, then synthesize
a corrected grid rather than only choose an existing candidate.

**A3 ARC mapping:** The A3 win makes this the strongest .394 method: selection
works, so the next step is synthesis for cases where correct evidence is split
across candidate families.

**A4 code mapping:** Keep the AggLM arm ARC-first until a source-distinct code
pool exists.

**B2 reward mapping:** Do not mix it with the blocked reward-training claim; any
future generated training data still needs same-base A-vs-B controls.

**Failure mode:** Synthesis can fabricate grids. Exact ARC validation, vote,
selector-only, and matched-budget controls are mandatory.

**Experiment mapping:** Run an AggLM-style ARC reconciler that synthesizes a
corrected grid from Set-Encoder evidence, with a bigger pool and SCOPE
per-region ablation.

## ARBITER: diagnose wrong-majority recovery

**Method/source:** ARBITER, arXiv:2605.26172
(https://arxiv.org/abs/2605.26172), frames majority-vote failures as reasoning
basins where the most stable basin can be wrong.

**Carnot stack mapping:** Record vote basin, Set-Encoder margin, and bounded
evidence-over-prior for each wrong-majority recovery.

**A3 ARC mapping:** Exp 4245's margin override also won, so ARBITER is the
right diagnostic language for which wrong-majority basins were recovered.

**A4 code mapping:** No code-basin replication can be claimed while A4 is
blocked.

**B2 reward mapping:** Not a reward-training result.

**Failure mode:** Hidden-state variants would expand the claim surface. Keep the
Carnot arm output-evidence-only unless explicitly gated.

**Experiment mapping:** Add basin-level recovery accounting to the AggLM and
Set-Encoder comparison.

## budget-aware discriminative verification: keep the vote prior

**Method/source:** Budget-aware discriminative verification, arXiv:2510.14913
(https://arxiv.org/abs/2510.14913), supports a practical hybrid of
self-consistency and discriminative verification.

**Carnot stack mapping:** Preserve vote as prior and add learned verification
only when the margin can change an answer.

**A3 ARC mapping:** Exp 4245 reports both selector lift and matched-control
lift, so .394 should add cost-normalized hybrid accounting rather than replacing
vote wholesale.

**A4 code mapping:** Robustness across code remains blocked.

**B2 reward mapping:** Selection-time efficiency is separate from reward SFT.

**Failure mode:** Hybrid scores can hide weak verifiers behind vote. Keep
selector-only and vote-only rows.

**Experiment mapping:** Match candidate and token budgets across vote,
Set-Encoder, margin hybrid, and AggLM synthesis.

## RAFT and VAR: keep offline reward weighting owed, not headline

**Method/source:** RAFT, arXiv:2504.11343
(https://arxiv.org/abs/2504.11343), and VAR, arXiv:2502.11026
(https://arxiv.org/abs/2502.11026), support offline reward-positive or
reward-weighted SFT instead of live online RL.

**Carnot stack mapping:** They remain the right shape for the offline B path:
bounded SFT over precomputed A/B/C corpora after the harness proves real
training.

**A3 ARC mapping:** They do not explain the ARC A3 selection win and should not
be flagged over AggLM for .394.

**A4 code mapping:** They reuse the stable code corpora for B2, but A4 still
needs a distinct second code candidate source for robustness.

**B2 reward mapping:** B2 blocked before training, so the next refinement is
harness repair: supported LoRA modules, at least 20 optimizer steps,
loss_final<loss_initial, and trainable_param_count>0.

**Failure mode:** Offline SFT can look stable while learning generator artifacts.

**Experiment mapping:** Once the harness passes, run VAR/RAFT-style Arm A
verifier-certified vs Arm B same-generator random labels and Arm C gold.

## Spurious Rewards: preserve the same-base random-label ablation

**Method/source:** Spurious Rewards, arXiv:2506.10947
(https://arxiv.org/abs/2506.10947), shows random rewards can recover much of an
RLVR gain on some models and are model-family dependent.

**Carnot stack mapping:** The A-vs-B reward task must use the same non-Qwen
base, same generator, and same step budget.

**A3 ARC mapping:** Not required for the test-time ARC A3 claim.

**A4 code mapping:** Does not solve the second-corpus gap.

**B2 reward mapping:** This is the non-negotiable B2 control once training
actually runs.

**Failure mode:** Without same-base random labels, a positive reward-weighted
result is uninterpretable.

**Experiment mapping:** Keep Qwen training forbidden and require Arm A minus Arm
B CI95 excluding zero.

## SCOPE: add per-region evidence

**Method/source:** SCOPE, arXiv:2512.15146
(https://arxiv.org/abs/2512.15146), replaces flat majority pseudo-labels with
fine-grained, subgroup-specific confidence signals.

**Carnot stack mapping:** Convert ARC candidate disagreement into local
evidence: which regions support the minority candidate and which regions the
vote basin fails.

**A3 ARC mapping:** A3 proved the global set encoder can select; SCOPE tells
.394 how to make the selected or synthesized grid explainable and higher
resolution.

**A4 code mapping:** Code has no comparable per-region evidence source in the
blocked A4 result.

**B2 reward mapping:** Do not train on SCOPE-style pseudo-labels until the B2
harness passes.

**Failure mode:** Fine-grained pseudo-labels can amplify confirmation bias if
they are not tied to exact grid labels.

**Experiment mapping:** Ablate AggLM synthesis with and without SCOPE per-region
evidence on the bigger ARC pool.

## Flagged for .394

`agglm_synthesize_corrected_grid_from_set_encoder_evidence_v394` is the single
strongest method for the next planner. The reason is conditional on the actual
A3/A4/B2 outcomes: Exp 4245 already proved the ARC set-encoder selector beats
vote, Exp 4246 says code robustness is unresolved rather than negative, and
Exp 4248 says reward-weighted SFT is still blocked at the harness gate. So .394
should scale the ARC win with AggLM-style generative reconciliation that
synthesizes a corrected grid from Set-Encoder plus SCOPE per-region evidence on
a bigger pool. Keep code replication as a robustness gate and treat
reward-weighted SFT as an owed gate after the harness proves real training.
"""

STUDYING_SECTION = """## 2026-06-15 Exp 4251 - .393 planning sweep SOTA ingestion ingested

**Status:** INGESTED into `docs/research-notes/sota-ingestion-set-encoder-offline-rft-v394-2026-06-15.md`.

**Filtered track:** ARC oracle-distinct set-encoder scale-up after Exp 4245
landed the clean A3 beats-vote win, with Exp 4246 code replication blocked on a
missing distinct candidate corpus and Exp 4248 offline reward-weighted SFT
blocked by the upstream harness smoke.

**Seed and fresh-pass candidates marked ingested:**
- Set-LLM, arXiv:2505.15433 - mapped as the high-capacity selector scale-up
  after the Exp 4245 DeepSets-style set encoder already beat vote.
- AggLM, arXiv:2509.06870 - mapped as the strongest .394 method: a generative
  reconciler that synthesizes a corrected grid from Set-Encoder evidence.
- ARBITER, arXiv:2605.26172 - mapped to wrong-majority basin diagnostics and
  conservative evidence-over-vote accounting.
- Budget-aware discriminative verification, arXiv:2510.14913 - mapped to
  cost-normalized vote-plus-verifier hybrid reporting.
- RAFT, arXiv:2504.11343, and VAR, arXiv:2502.11026 - mapped to the owed
  offline reward-weighted SFT path after the harness proves real training.
- Spurious Rewards, arXiv:2506.10947 - mapped to the required same-base
  random-label Arm B control for any reward-training claim.
- SCOPE, arXiv:2512.15146 - mapped to per-region ARC evidence for the .394
  AggLM synthesis ablation.

Exp 4245 status mapped honestly: `headline_outcome=arc_oracle_distinct_set_encoder_beats_vote`,
`set_encoder_minus_vote_delta=0.4423076923`, CI95 `[0.3076923077, 0.5961538462]`,
`margin_override_minus_vote=0.4230769231`, and `oracle_distinct_beats_vote=true`.
Exp 4246 status mapped honestly: `blocked_code_second_corpus_missing`; code
robustness is unresolved, not refuted. Exp 4248 status mapped honestly:
`blocked_gate_check_failed` because Exp 4247 reported `harness_smoke_passed=false`,
`steps_run=0`, and `trainable_param_count=0`.

flagged_for_v394:
`agglm_synthesize_corrected_grid_from_set_encoder_evidence_v394`.

Flagged for .394: `agglm_synthesize_corrected_grid_from_set_encoder_evidence_v394`.

**Bottom line for the .394 roadmap:** scale the proven ARC set-encoder win with
AggLM-style corrected-grid synthesis plus SCOPE per-region evidence on a bigger
pool. Keep code replication as a robustness gate and treat reward-weighted SFT as an owed gate after the harness proves real training.
"""

STUDYING_MARKER = "## 2026-06-15 Exp 4251 - .393 planning sweep SOTA ingestion ingested"


def build_artifact(
    *,
    methods_mapped: Sequence[Mapping[str, str]],
    flagged_for_v394: str,
    honest_verdict: str = DEFAULT_HONEST_VERDICT,
) -> dict[str, object]:
    """Build and validate the REQ-REPORT-4251 mapping artifact."""

    artifact: dict[str, object] = {
        "honest_verdict": honest_verdict,
        "methods_mapped": [dict(method) for method in methods_mapped],
        "flagged_for_v394": flagged_for_v394,
        "field_principles": dict(FIELD_PRINCIPLES),
    }
    validate_artifact(artifact)
    return artifact


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """Validate the JSON contract so uncited method rows fail closed."""

    missing = REQUIRED_ARTIFACT_FIELDS.difference(artifact)
    extra = set(artifact).difference(REQUIRED_ARTIFACT_FIELDS)
    if missing:
        raise ValueError(f"artifact missing required fields: {sorted(missing)}")
    if extra:
        raise ValueError(f"artifact has unexpected fields: {sorted(extra)}")

    verdict = artifact["honest_verdict"]
    if not isinstance(verdict, str) or not verdict.startswith(TERMINAL_PREFIXES):
        raise ValueError("honest_verdict must use a terminal prefix")

    if artifact["field_principles"] != FIELD_PRINCIPLES:
        raise ValueError("field_principles must match the required annotations")

    methods_mapped = artifact["methods_mapped"]
    if not isinstance(methods_mapped, list) or len(methods_mapped) < 3 or len(methods_mapped) > 8:
        raise ValueError("methods_mapped must contain three to eight methods")

    seen: set[str] = set()
    for method in methods_mapped:
        if not isinstance(method, dict) or set(method) != REQUIRED_METHOD_FIELDS:
            raise ValueError(
                "each method must contain exactly name, arxiv_id_or_url, url, "
                "carnot_stack_mapping, a3_arc_mapping, a4_code_mapping, "
                "b2_reward_mapping, failure_mode, and experiment_mapping"
            )
        source = method["arxiv_id_or_url"]
        if source not in VERIFIED_SOURCE_URLS:
            raise ValueError(f"method arxiv_id_or_url must be a verified source: {source}")
        if source in seen:
            raise ValueError(f"duplicate source: {source}")
        seen.add(source)
        expected_url = VERIFIED_SOURCE_URLS[source]
        if method["url"] != expected_url:
            raise ValueError(f"method url must be {expected_url!r}")
        for field in REQUIRED_METHOD_FIELDS - {"arxiv_id_or_url", "url"}:
            value = method[field]
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"method {field} must be a non-empty string")

    flagged = artifact["flagged_for_v394"]
    if not isinstance(flagged, str) or not flagged.strip():
        raise ValueError("flagged_for_v394 must be a non-empty string")
    if flagged != DEFAULT_FLAGGED_FOR_V394:
        raise ValueError("flagged_for_v394 must name the AggLM corrected-grid synthesis plan")


def validate_markdown_note(markdown: str) -> None:
    """Check that the paired note maps verified sources to A3/A4/B2 outcomes."""

    required_phrases = (
        "Fresh-pass provenance",
        "Exp 4245 ARC A3",
        "Exp 4246 code A4",
        "Exp 4248 offline B2",
        "SOTA -> experiment mapping",
        "headline_outcome=arc_oracle_distinct_set_encoder_beats_vote",
        "set_encoder_minus_vote_delta=0.4423076923",
        "margin_override_minus_vote=0.4230769231",
        "matched_control_delta=0.4807692308",
        "oracle_at_k=0.8269230769",
        "oracle_distinct_beats_vote=true",
        "blocked_code_second_corpus_missing",
        "blocked_gate_check_failed",
        "harness_smoke_passed=false",
        "steps_run=0",
        "trainable_param_count=0",
        "Set-LLM",
        "AggLM",
        "ARBITER",
        "budget-aware discriminative verification",
        "RAFT",
        "VAR",
        "Spurious Rewards",
        "SCOPE",
        "synthesizes a corrected grid",
        "SCOPE per-region evidence",
        "bigger pool",
        "same-base random-label",
        "Carnot stack mapping",
        "A3 ARC mapping",
        "A4 code mapping",
        "B2 reward mapping",
        "Failure mode",
        "Experiment mapping",
        "Flagged for .394",
    )
    missing_phrases = [phrase for phrase in required_phrases if phrase not in markdown]
    if missing_phrases:
        raise ValueError(f"markdown note missing required sections: {missing_phrases}")

    missing_sources = [
        source for source in NOTE_REQUIRED_SOURCE_CITATIONS if source not in markdown
    ]
    if missing_sources:
        raise ValueError(f"markdown note missing verified source citations: {missing_sources}")


def write_outputs(
    *,
    note_path: Path,
    artifact_path: Path,
    studying_path: Path,
) -> dict[str, object]:
    """Write the note, JSON artifact, and idempotent studying update."""

    artifact = build_artifact(
        methods_mapped=DEFAULT_METHODS_MAPPED,
        flagged_for_v394=DEFAULT_FLAGGED_FOR_V394,
    )
    validate_markdown_note(NOTE_MARKDOWN)

    note_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    note_path.write_text(NOTE_MARKDOWN + "\n", encoding="utf-8")
    artifact_path.write_text(
        json.dumps(artifact, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    studying_path.write_text(
        _with_studying_section(studying_path.read_text(encoding="utf-8")),
        encoding="utf-8",
    )
    return artifact


def _replace_or_insert_section(
    existing: str,
    *,
    marker: str,
    section: str,
) -> str:
    if marker not in existing:
        if existing.startswith("## "):
            return section + "\n" + existing
        if "\n## " not in existing:
            return existing.rstrip() + "\n\n" + section
        return existing.replace("\n## ", "\n" + section + "\n## ", 1)

    before, after_marker = existing.split(marker, 1)
    next_section = after_marker.find("\n## ")
    if next_section == -1:
        return before + section.rstrip() + "\n"
    return before + section + after_marker[next_section + 1 :]


def _with_studying_section(existing: str) -> str:
    return _replace_or_insert_section(
        existing,
        marker=STUDYING_MARKER,
        section=STUDYING_SECTION,
    )


def main() -> int:
    """Write the default Exp 4251 deliverables under the repository root."""

    repo_root = Path(__file__).resolve().parents[2]
    artifact = write_outputs(
        note_path=repo_root
        / "docs/research-notes/sota-ingestion-set-encoder-offline-rft-v394-2026-06-15.md",
        artifact_path=repo_root
        / "results/experiment_4251_sota_ingestion_set_encoder_offline_rft.json",
        studying_path=repo_root / "research-studying.md",
    )
    print(artifact["honest_verdict"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
