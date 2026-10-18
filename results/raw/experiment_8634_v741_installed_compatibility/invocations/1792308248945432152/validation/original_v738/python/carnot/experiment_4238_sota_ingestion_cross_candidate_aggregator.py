"""Exp 4238 SOTA ingestion for the cross-candidate aggregator .393 plan.

Spec refs: REQ-REPORT-4238, SCENARIO-REPORT-4238.

This module writes a planning artifact, not a benchmark result. It closes the
`.392 planning sweep` into a concrete SOTA-to-experiment mapping after Exp
4232 tied ARC vote at power and Exp 4233 showed the code verifier can beat vote.
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
        "flagged_for_v393",
        "field_principles",
    }
)
REQUIRED_METHOD_FIELDS = frozenset(
    {
        "name",
        "arxiv_id_or_url",
        "url",
        "carnot_stack_mapping",
        "a2_a3_mapping",
        "failure_mode",
        "experiment_mapping",
    }
)
TERMINAL_PREFIXES = ("complete:", "success:", "passed:", "shipped:", "blocked_")
DEFAULT_HONEST_VERDICT = "complete: sota_ingestion_cross_candidate_aggregator_mapped_v393"
DEFAULT_FLAGGED_FOR_V393 = "bigger_arc_pool_full_set_encoder_agglm_aggregator_v393"

FIELD_PRINCIPLES = {
    "honest_verdict": "Terminal-prefixed. Records ingestion completed with verifiable citations.",
    "methods_mapped": (
        "Each method MUST carry a real arXiv ID/URL; an ingestion note without "
        "verifiable citations is treated as fabrication (adversarial_verify "
        "discipline)."
    ),
    "flagged_for_v393": (
        "Closes discover->ingest->plan: names the strongest method for the next "
        "planner, conditioned on the A2/A3 outcomes."
    ),
}

VERIFIED_SOURCE_URLS = {
    "2404.06912": "https://arxiv.org/abs/2404.06912",
    "2509.19681": "https://arxiv.org/abs/2509.19681",
    "2606.04323": "https://arxiv.org/abs/2606.04323",
    "2512.15146": "https://arxiv.org/abs/2512.15146",
    "2602.03975": "https://arxiv.org/abs/2602.03975",
    "2603.03417": "https://arxiv.org/abs/2603.03417",
    "2509.06870": "https://arxiv.org/abs/2509.06870",
    "2602.09341": "https://arxiv.org/abs/2602.09341",
}
NOTE_REQUIRED_SOURCE_CITATIONS = frozenset(f"arXiv:{source}" for source in VERIFIED_SOURCE_URLS)

DEFAULT_METHODS_MAPPED = [
    {
        "name": "Set-Encoder full cross-candidate attention",
        "arxiv_id_or_url": "2404.06912",
        "url": "https://arxiv.org/abs/2404.06912",
        "carnot_stack_mapping": (
            "Maps the ARC selector from hand-built set summary features to a "
            "true permutation-invariant candidate-set encoder with inter-candidate "
            "attention over all grids and vote basins."
        ),
        "a2_a3_mapping": (
            "Exp 4232 tied vote with headroom after Exp 4231 used an augmented "
            "logistic aggregator. The next lever is a full set encoder, because "
            "the tied A3 read says explicit summary features were not enough."
        ),
        "failure_mode": (
            "Set-Encoder is an information-retrieval architecture, not an ARC grid "
            "solver. It scales cause (1), isolated scoring, but still needs a "
            "larger ARC pool and minority-correct evidence to avoid learning vote."
        ),
        "experiment_mapping": (
            "Build a larger ARC candidate pool, train a task-held-out Set-Encoder "
            "over full candidate sets, and ablate it against the Exp 4231 "
            "cross-candidate augmented logistic aggregator on the same A3 gate."
        ),
    },
    {
        "name": "Calibrated Reasoning explanatory verifier",
        "arxiv_id_or_url": "2509.19681",
        "url": "https://arxiv.org/abs/2509.19681",
        "carnot_stack_mapping": (
            "Maps to a calibrated verifier objective for sparse ARC positives: "
            "class-balanced or focal loss, train-fold calibration, and score "
            "diagnostics that detect collapse under severe imbalance."
        ),
        "a2_a3_mapping": (
            "Exp 4231 still had only 20 positive candidates after growth and "
            "reported no learnable AUROC gain. Exp 4233's balanced code pool won, "
            "so the ARC direction is more positives plus calibrated training."
        ),
        "failure_mode": (
            "Calibration can make scores honest without adding missing evidence. "
            "If the ARC pool remains sparse, a better loss will still be "
            "underpowered for wrong-majority selection."
        ),
        "experiment_mapping": (
            "Require positive-candidate growth before .393, then compare "
            "class-weighted, focal, and calibrated pairwise losses with the same "
            "held-out ARC split and score-collapse checks."
        ),
    },
    {
        "name": "Margin-triggered question re-arbitration",
        "arxiv_id_or_url": "2606.04323",
        "url": "https://arxiv.org/abs/2606.04323",
        "carnot_stack_mapping": (
            "Maps to the deployment gate: keep vote unless the learned aggregator "
            "has a pre-registered confidence margin over the vote candidate."
        ),
        "a2_a3_mapping": (
            "Exp 4232's margin override also tied vote at 0.0 lift, so the margin "
            "policy is a guardrail, not the main .393 research lever until the "
            "underlying set-aware score has nonzero separation."
        ),
        "failure_mode": (
            "The cited video QA re-arbitration is sensitive to triggered-subset "
            "composition. Used alone, it can preserve every vote decision or "
            "fire on brittle margins."
        ),
        "experiment_mapping": (
            "Keep the fixed margin-trigger policy in .393 as an evaluation arm, "
            "but only interpret it after the full set encoder creates measurable "
            "score margins on wrong-majority ARC tasks."
        ),
    },
    {
        "name": "SCOPE fine-grained reward signal",
        "arxiv_id_or_url": "2512.15146",
        "url": "https://arxiv.org/abs/2512.15146",
        "carnot_stack_mapping": (
            "Maps to ARC per-region evidence: local confidence and subgroup "
            "signals should explain why a minority grid is correct, instead of "
            "treating the final answer frequency as the reward."
        ),
        "a2_a3_mapping": (
            "Because Exp 4232 tied vote while oracle_minus_vote remained 0.1731, "
            "the next ARC run needs denser region-level supervision for the cases "
            "where a correct candidate exists but vote fails."
        ),
        "failure_mode": (
            "SCOPE is a test-time RL pseudo-labeling recipe; it does not by itself "
            "provide ARC grid-local labels or solve the small wrong-majority count."
        ),
        "experiment_mapping": (
            "Add localized ARC disagreement features and report whether they "
            "change full-set encoder decisions on wrong-majority tasks before any "
            "verifier-as-reward training is attempted."
        ),
    },
    {
        "name": "Adaptive verification allocation over categorical structure",
        "arxiv_id_or_url": "2602.03975",
        "url": "https://arxiv.org/abs/2602.03975",
        "carnot_stack_mapping": (
            "Maps to compute routing for ARC candidate families: spend expensive "
            "evidence checks on uncertain transformation families and avoid "
            "redundant verifier calls on duplicate grids."
        ),
        "a2_a3_mapping": (
            "Exp 4232's matched control was close to the aggregator, so .393 "
            "should separate architecture gain from compute allocation and route "
            "verification to informative candidate basins."
        ),
        "failure_mode": (
            "Allocation improves efficiency but cannot turn an uninformative "
            "score into a vote-beating selector. It should follow, not replace, "
            "the bigger-pool set-encoder experiment."
        ),
        "experiment_mapping": (
            "Use adaptive allocation as a secondary ablation: fixed all-candidate "
            "Set-Encoder versus uncertainty-routed evidence collection under the "
            "same candidate budget."
        ),
    },
    {
        "name": "MSV multi-sequence verifier",
        "arxiv_id_or_url": "2603.03417",
        "url": "https://arxiv.org/abs/2603.03417",
        "carnot_stack_mapping": (
            "Maps to verifier calibration across all candidate solutions rather "
            "than independent row scores, with interactions between competing "
            "answers exposed to the scorer."
        ),
        "a2_a3_mapping": (
            "MSV supports the Set-Encoder direction after the Exp 4232 tie: the "
            "augmented-feature aggregator approximated context, but the next "
            "model should jointly process candidate interactions directly."
        ),
        "failure_mode": (
            "Cross-sequence calibration can still learn frequency if the target "
            "pool is too sparse. It needs the Exp 4233 data-sparsity lesson: grow "
            "ARC before treating another tie as a thesis bound."
        ),
        "experiment_mapping": (
            "Report isolated scoring, explicit summary features, and full "
            "multi-sequence/set attention as three arms on the expanded ARC pool."
        ),
    },
    {
        "name": "AggLM review-reconcile-synthesize aggregation",
        "arxiv_id_or_url": "2509.06870",
        "url": "https://arxiv.org/abs/2509.06870",
        "carnot_stack_mapping": (
            "Maps to an aggregator that can synthesize a corrected grid from "
            "candidate evidence, not only select one cached candidate."
        ),
        "a2_a3_mapping": (
            "Exp 4232's selector tie leaves oracle headroom unused. AggLM is the "
            "best synthesis precedent for cases where the correct evidence is "
            "distributed across candidate families."
        ),
        "failure_mode": (
            "A generative reconciler expands the claim surface: it can hallucinate "
            "a grid unless exact ARC validation and matched selector controls stay "
            "in the gate."
        ),
        "experiment_mapping": (
            "Make synthesis an optional .393 arm after the full set encoder: "
            "selector-only Set-Encoder versus AggLM-style review/reconcile output, "
            "both scored by exact grid match."
        ),
    },
    {
        "name": "AgentAuditor localized branch evidence",
        "arxiv_id_or_url": "2602.09341",
        "url": "https://arxiv.org/abs/2602.09341",
        "carnot_stack_mapping": (
            "Maps to localized ARC evidence auditing: compare branches where "
            "candidate transformations diverge and reward evidence-based minority "
            "selection over popular errors."
        ),
        "a2_a3_mapping": (
            "AgentAuditor explains why the Exp 4232 tie is not just a threshold "
            "problem. The scorer needs local evidence for wrong-majority cases, "
            "not only global vote and set statistics."
        ),
        "failure_mode": (
            "Reasoning-tree evidence is richer than cached ARC rows. Without "
            "region-level candidate traces, the auditor can collapse into another "
            "LLM judge rather than a cheap verifier."
        ),
        "experiment_mapping": (
            "Use AgentAuditor as the localized-evidence comparator for .393: "
            "measure whether region disagreements explain Set-Encoder overrides "
            "and whether the cost beats an LLM-as-judge fallback."
        ),
    },
]

NOTE_MARKDOWN = """# SOTA ingestion 2026-06-15: cross-candidate aggregator map for .393

**Artifact fields**
- honest_verdict: `complete: sota_ingestion_cross_candidate_aggregator_mapped_v393`
  - principle: Terminal-prefixed. Records ingestion completed with verifiable citations.
- methods_mapped:
  - {name: `Set-Encoder full cross-candidate attention`, arxiv_id_or_url: `2404.06912`, url: `https://arxiv.org/abs/2404.06912`}
  - {name: `Calibrated Reasoning explanatory verifier`, arxiv_id_or_url: `2509.19681`, url: `https://arxiv.org/abs/2509.19681`}
  - {name: `Margin-triggered question re-arbitration`, arxiv_id_or_url: `2606.04323`, url: `https://arxiv.org/abs/2606.04323`}
  - {name: `SCOPE fine-grained reward signal`, arxiv_id_or_url: `2512.15146`, url: `https://arxiv.org/abs/2512.15146`}
  - {name: `Adaptive verification allocation over categorical structure`, arxiv_id_or_url: `2602.03975`, url: `https://arxiv.org/abs/2602.03975`}
  - {name: `MSV multi-sequence verifier`, arxiv_id_or_url: `2603.03417`, url: `https://arxiv.org/abs/2603.03417`}
  - {name: `AggLM review-reconcile-synthesize aggregation`, arxiv_id_or_url: `2509.06870`, url: `https://arxiv.org/abs/2509.06870`}
  - {name: `AgentAuditor localized branch evidence`, arxiv_id_or_url: `2602.09341`, url: `https://arxiv.org/abs/2602.09341`}
  - principle: Each method MUST carry a real arXiv ID/URL; an ingestion note without verifiable citations is treated as fabrication (adversarial_verify discipline).
- flagged_for_v393: `bigger_arc_pool_full_set_encoder_agglm_aggregator_v393`
  - principle: Closes discover->ingest->plan: names the strongest method for the next planner, conditioned on the A2/A3 outcomes.

## Fresh-pass provenance

Read `research-references.md` `.392 planning sweep`, `research-studying.md`,
`results/experiment_4231_oracle_distinct_arc_aggregator_build.json`,
`results/experiment_4232_oracle_distinct_arc_aggregator_beats_vote.json`, and
`results/experiment_4233_oracle_distinct_code_beats_vote.json`.

Reliable-channel helper pass, not `/deep-research`:
- `python3 scripts/sweep_clusters.py 0 --max-results 8`
- `python3 scripts/sweep_clusters.py 1 --max-results 8`
- `python3 scripts/sweep_semscholar.py "permutation invariant inter passage attention listwise reranking verifier candidate set" --limit 8`
- `python3 scripts/sweep_semscholar.py "calibrated reasoning explanatory verifier margin triggered re arbitration multi sequence verifier" --limit 8`

The cluster helper emitted the broadened verifier/process-reward and
energy/verifier arXiv API URLs. Semantic Scholar returned 0 arXiv IDs for the
first focused query and HTTP 429 for the second, so no S2-only promotion is
claimed. Low-concurrency WebSearch/WebFetch verified arXiv:2404.06912,
arXiv:2509.19681, arXiv:2606.04323, arXiv:2512.15146, arXiv:2602.03975,
arXiv:2603.03417, arXiv:2509.06870, and arXiv:2602.09341.

## Exp 4231 A2 build, Exp 4232 ARC A3, and Exp 4233 code read

Exp 4231 did build the strengthened ARC aggregator, but it stayed sparse:
`oracle_distinct_auroc=0.7865558646`, CI95 `[0.6319719028, 0.9258842843]`,
`positive_candidate_n=20`, `wrong_majority_n=9`,
`no_learnable_gain_reason=too_few_positives_after_growth`, and architecture
`cross_candidate_augmented_calibrated_logistic_aggregator`.

Exp 4232 then ran the held-out ARC A3 gate at `held_out_task_n=52` and tied
vote despite headroom: `aggregator_minus_vote_delta=0.0`, CI95 `[0.0, 0.0]`,
`margin_override_minus_vote=0.0`, `oracle_minus_vote=0.1730769231`,
`matched_control_delta=0.0384615385`, and
`oracle_distinct_beats_vote=false`. This is not an under-power n=14 repeat,
but it still leaves the strongest false-negative risk in the sparse-positive
wrong-majority ARC stratum.

Exp 4233 disambiguated the ARC null with code: `code_predictor_minus_vote_delta=0.03125`,
CI95 `[0.00625, 0.0625]`, `held_out_task_n=160`, `code_oracle_distinct_beats_vote=true`,
and `disambiguation_read=ARC_null_is_data_sparsity`. That read is load-bearing:
the next ARC step should grow the ARC pool and change architecture before
declaring the oracle-distinct selection thesis bounded.

## SOTA -> experiment mapping

## Set-Encoder: fix isolated scoring

**Method/source:** Set-Encoder, arXiv:2404.06912
(https://arxiv.org/abs/2404.06912), introduces permutation-invariant
inter-passage attention for listwise reranking.

**Carnot stack mapping:** Replace the Exp 4231 explicit set-statistics
aggregator with a full candidate-set encoder. The model should see all
candidates, vote basins, duplicate families, shape/palette families, and local
grid evidence in one attention pass.

**A2/A3 mapping:** Exp 4232 tied vote after the augmented-feature aggregator.
That makes a full Set-Encoder the strongest architecture lever: it directly
addresses the `.391` isolated scoring cause and tests whether learned
cross-candidate attention beats manual summary features.

**Failure mode:** Set-Encoder is not an ARC solver. If the ARC pool stays at
20 positive candidates and 9 wrong-majority tasks, it may simply learn frequency.

**Experiment mapping:** For .393, grow the ARC pool first, then compare isolated
scoring, Exp 4231 summary features, and a full Set-Encoder on identical
task-held-out splits.

## Calibrated Reasoning: fix class imbalance

**Method/source:** Calibrated Reasoning, arXiv:2509.19681
(https://arxiv.org/abs/2509.19681), trains an explanatory verifier with
calibrated confidence for candidate solutions.

**Carnot stack mapping:** Keep the class-balanced/calibrated loss, but make it
auditable: report score histograms, positive/negative calibration, and
wrong-majority margins rather than only AUROC.

**A2/A3 mapping:** Exp 4231 used a balanced logistic objective and isotonic
calibration, yet still had too few positives after growth. Exp 4233 won on a
larger, less sparse code pool, so the .393 calibration step needs more ARC data,
not just another loss variant.

**Failure mode:** Calibration cannot create missing positives or local grid
evidence. A better loss on the same sparse pool risks a cleaner-looking tie.

**Experiment mapping:** Pair the bigger ARC pool with class-weighted, focal,
and pairwise calibrated losses, then report whether any loss creates nonzero
wrong-majority margins for the Set-Encoder.

## Margin-triggered re-arbitration: fix override degeneracy

**Method/source:** Margin-triggered question re-arbitration, arXiv:2606.04323
(https://arxiv.org/abs/2606.04323), conditions re-arbitration on the
self-consistency vote margin.

**Carnot stack mapping:** Use the margin trigger as the final deployment guard:
keep vote unless the learned score margin over vote clears a pre-registered
threshold.

**A2/A3 mapping:** Exp 4232's margin override tied vote too, with
`margin_override_minus_vote=0.0`. Therefore margin-triggering should stay as an
evaluation arm, not become the .393 headline by itself.

**Failure mode:** A margin policy with no meaningful score separation either
never fires or fires on noise. The cited paper also reports sensitivity to the
triggered subset.

**Experiment mapping:** Retain the fixed margin-trigger policy after the
Set-Encoder produces margins; report selector@1, margin override, vote, and
matched control.

## SCOPE: add fine-grained reward and per-region evidence

**Method/source:** SCOPE, arXiv:2512.15146
(https://arxiv.org/abs/2512.15146), moves beyond majority voting with
step-wise confidence and subgroup-specific pseudo-label estimation.

**Carnot stack mapping:** Convert ARC grid disagreement into per-region evidence
features so the model has a reason to trust a minority answer.

**A2/A3 mapping:** Exp 4232 left `oracle_minus_vote=0.1730769231` on the table.
SCOPE maps to denser evidence for those tasks, not to a new majority-vote
threshold.

**Failure mode:** SCOPE is not an ARC label generator. Without exact grid labels
and region evidence, it could amplify pseudo-label bias.

**Experiment mapping:** Add region disagreement features before verifier-as-reward
training; ablate Set-Encoder with and without SCOPE-style local evidence.

## Adaptive verification allocation: route scarce checks

**Method/source:** Adaptive verification allocation, arXiv:2602.03975
(https://arxiv.org/abs/2602.03975), allocates costly verification over
structured intermediate states.

**Carnot stack mapping:** Route evidence gathering to uncertain ARC candidate
families, especially duplicate grids and high-uncertainty transformation
families.

**A2/A3 mapping:** Exp 4232's matched control remained close enough that .393
should separate score quality from compute routing.

**Failure mode:** Allocation is an efficiency lever, not the primary
vote-beating mechanism. It cannot rescue an uninformative score.

**Experiment mapping:** Make adaptive allocation a secondary arm after the full
Set-Encoder baseline exists.

## MSV: jointly process candidate solutions

**Method/source:** Multi-Sequence Verifier, arXiv:2603.03417
(https://arxiv.org/abs/2603.03417), jointly processes candidate solutions and
models their interactions.

**Carnot stack mapping:** Treat ARC candidates as a set and calibrate scores
with cross-candidate interactions, rather than independent candidate rows.

**A2/A3 mapping:** MSV corroborates the Set-Encoder diagnosis: the Exp 4231
summary-feature aggregator was only a proxy for direct cross-sequence modeling.

**Failure mode:** Cross-sequence calibration can still track frequency without
better positive support.

**Experiment mapping:** Report three arms: isolated scoring, explicit summary
features, and full multi-sequence/set attention on the expanded ARC pool.

## AggLM and AgentAuditor: synthesize or audit evidence

**Method/source:** AggLM, arXiv:2509.06870
(https://arxiv.org/abs/2509.06870), learns to review, reconcile, and synthesize
answers. AgentAuditor, arXiv:2602.09341
(https://arxiv.org/abs/2602.09341), audits localized reasoning-tree branch
evidence and targets majority-failure cases.

**Carnot stack mapping:** If selector-only Set-Encoder still ties, the next arm
should synthesize a corrected grid from candidate evidence or audit the
localized evidence behind the minority candidate.

**A2/A3 mapping:** Exp 4232 shows a selector tie, while Exp 4233 says the thesis
is not bounded in a higher-power domain. Therefore .393 should not stop at
selection-only reranking if a bigger ARC pool still leaves oracle headroom.

**Failure mode:** Synthesis increases fabrication risk, and AgentAuditor assumes
richer trace evidence than cached ARC rows may contain.

**Experiment mapping:** Add an AggLM-style synthesize corrected grid arm after
the full Set-Encoder baseline. Keep exact grid-match validation and compare
against an AgentAuditor localized-evidence audit and LLM-as-judge fallback.

## Flagged for .393

`bigger_arc_pool_full_set_encoder_agglm_aggregator_v393` is the strongest single
method for the next planner. The reason is conditional on the actual A2/A3
outcomes: ARC tied vote with headroom, but the higher-power code read beat vote
and explicitly reported `ARC_null_is_data_sparsity`. The .393 plan should grow
the ARC pool, run a full Set-Encoder against the augmented-feature aggregator,
and reserve AggLM-style synthesis for the case where selection still leaves
oracle headroom unused. Build a bigger ARC pool before declaring the
oracle-distinct selection thesis bounded.
"""

STUDYING_SECTION = """## 2026-06-15 Exp 4238 - .392 planning sweep SOTA ingestion ingested

**Status:** INGESTED into `docs/research-notes/sota-ingestion-cross-candidate-aggregator-v393-2026-06-15.md`.

**Filtered track:** strengthened oracle-distinct ARC aggregation after Exp 4231
built a sparse cross-candidate aggregator, Exp 4232 tied vote despite headroom,
and Exp 4233 beat vote on code with `disambiguation_read=ARC_null_is_data_sparsity`.

**Seed and fresh-pass candidates marked ingested:**
- Set-Encoder, arXiv:2404.06912 - mapped as the strongest .393 architecture
  lever: full cross-candidate attention instead of Exp 4231's augmented-feature
  logistic aggregator.
- Calibrated Reasoning, arXiv:2509.19681 - mapped to imbalance-aware calibrated
  losses, but only after ARC positive-candidate growth.
- Margin-triggered re-arbitration, arXiv:2606.04323 - kept as the deployment
  guard because Exp 4232's margin override also tied vote.
- SCOPE, arXiv:2512.15146 - mapped to per-region ARC evidence and dense
  confidence signals for wrong-majority cases.
- Adaptive verification allocation, arXiv:2602.03975 - mapped to compute
  routing after a stronger score exists.
- MSV, arXiv:2603.03417 - mapped to joint cross-sequence scoring as the
  direct model-class corroboration for Set-Encoder.
- AggLM, arXiv:2509.06870, and AgentAuditor, arXiv:2602.09341 - mapped to
  review/reconcile/synthesize and localized evidence audit arms if selection
  still leaves oracle headroom unused.

Exp 4231 status mapped honestly: `oracle_distinct_auroc=0.7865558646`,
`positive_candidate_n=20`, `wrong_majority_n=9`, and
`no_learnable_gain_reason=too_few_positives_after_growth`. Exp 4232 status
mapped honestly: `aggregator_minus_vote_delta=0.0`,
`oracle_minus_vote=0.1730769231`, and `oracle_distinct_beats_vote=false`.
Exp 4233 status mapped honestly: `code_predictor_minus_vote_delta=0.03125`,
CI95 `[0.00625, 0.0625]`, and
`disambiguation_read=ARC_null_is_data_sparsity`.

flagged_for_v393:
`bigger_arc_pool_full_set_encoder_agglm_aggregator_v393`.

Flagged for .393: `bigger_arc_pool_full_set_encoder_agglm_aggregator_v393`.

**Bottom line for the .393 roadmap:** grow ARC positives, run a full Set-Encoder against the augmented-feature aggregator, and build a bigger ARC pool before declaring the oracle-distinct selection thesis bounded.
"""

STUDYING_MARKER = "## 2026-06-15 Exp 4238 - .392 planning sweep SOTA ingestion ingested"


def build_artifact(
    *,
    methods_mapped: Sequence[Mapping[str, str]],
    flagged_for_v393: str,
    honest_verdict: str = DEFAULT_HONEST_VERDICT,
) -> dict[str, object]:
    """Build and validate the REQ-REPORT-4238 mapping artifact."""

    artifact: dict[str, object] = {
        "honest_verdict": honest_verdict,
        "methods_mapped": [dict(method) for method in methods_mapped],
        "flagged_for_v393": flagged_for_v393,
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
    if not isinstance(methods_mapped, list) or len(methods_mapped) < 5 or len(methods_mapped) > 8:
        raise ValueError("methods_mapped must contain five to eight methods")

    seen: set[str] = set()
    for method in methods_mapped:
        if not isinstance(method, dict) or set(method) != REQUIRED_METHOD_FIELDS:
            raise ValueError(
                "each method must contain exactly name, arxiv_id_or_url, url, "
                "carnot_stack_mapping, a2_a3_mapping, failure_mode, and "
                "experiment_mapping"
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

    flagged = artifact["flagged_for_v393"]
    if not isinstance(flagged, str) or not flagged.strip():
        raise ValueError("flagged_for_v393 must be a non-empty string")
    if flagged != DEFAULT_FLAGGED_FOR_V393:
        raise ValueError("flagged_for_v393 must name the bigger ARC Set-Encoder plan")


def validate_markdown_note(markdown: str) -> None:
    """Check that the paired note maps verified sources to required axes."""

    required_phrases = (
        "Fresh-pass provenance",
        "Exp 4231 A2 build",
        "Exp 4232 ARC A3",
        "Exp 4233 code read",
        "SOTA -> experiment mapping",
        "isolated scoring",
        "class imbalance",
        "margin-trigger",
        "under-power",
        "SCOPE",
        "adaptive allocation",
        "MSV",
        "AggLM",
        "AgentAuditor",
        "synthesize corrected grid",
        "bigger ARC pool",
        "full Set-Encoder",
        "Carnot stack mapping",
        "A2/A3 mapping",
        "Failure mode",
        "Experiment mapping",
        "Flagged for .393",
    )
    missing_phrases = [phrase for phrase in required_phrases if phrase not in markdown]
    if missing_phrases:
        raise ValueError(f"markdown note missing required sections: {missing_phrases}")

    missing_sources = [
        source for source in NOTE_REQUIRED_SOURCE_CITATIONS if source not in markdown
    ]
    if missing_sources:
        raise ValueError(f"markdown note missing verified source citations: {missing_sources}")
    if "oracle_distinct_beats_vote=false" not in markdown:
        raise ValueError("markdown note must preserve oracle_distinct_beats_vote=false")
    if "ARC_null_is_data_sparsity" not in markdown:
        raise ValueError("markdown note must preserve ARC_null_is_data_sparsity")


def write_outputs(
    *,
    note_path: Path,
    artifact_path: Path,
    studying_path: Path,
) -> dict[str, object]:
    """Write the note, JSON artifact, and idempotent studying update."""

    artifact = build_artifact(
        methods_mapped=DEFAULT_METHODS_MAPPED,
        flagged_for_v393=DEFAULT_FLAGGED_FOR_V393,
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
    """Write the default Exp 4238 deliverables under the repository root."""

    repo_root = Path(__file__).resolve().parents[2]
    artifact = write_outputs(
        note_path=repo_root
        / "docs/research-notes/sota-ingestion-cross-candidate-aggregator-v393-2026-06-15.md",
        artifact_path=repo_root
        / "results/experiment_4238_sota_ingestion_cross_candidate_aggregator.json",
        studying_path=repo_root / "research-studying.md",
    )
    print(artifact["honest_verdict"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
