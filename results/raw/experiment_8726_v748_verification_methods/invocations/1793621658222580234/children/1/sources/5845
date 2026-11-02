"""Schema helpers for the Exp 4094 precision-calibration SOTA ingestion.

Spec refs: REQ-REPORT-4094, SCENARIO-REPORT-4094.

The note is a planning bridge, not a new experiment result.  It gathers the
local precision-rescue artifacts and verified papers into one dated map so the
next roadmap can decide which verifier-noise fix to try before another RFT run.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import json
from pathlib import Path
from typing import Any


EXPECTED_INFERENCE_SUBSTRATE = "aggregation_from_upstream_artifacts"
REQUIRED_RECEIPT_FIELDS = frozenset(
    {
        "honest_verdict",
        "methods_mapped",
        "strongest_for_next_roadmap",
        "inference_substrate",
    }
)
TERMINAL_PREFIXES = ("complete:", "success:", "passed:", "shipped:", "blocked_")
DEFAULT_HONEST_VERDICT = "complete: sota_ingestion_precision_calibration_mapped"

VERIFIED_ARXIV_IDS = frozenset(
    {
        "2308.01825",
        "2402.06457",
        "2410.17621",
        "2411.02272",
        "2507.14843",
        "2510.00915",
        "2603.16140",
    }
)

DEFAULT_METHODS_MAPPED = [
    {
        "arxiv_id": "2411.02272",
        "one_line": (
            "BARC-style augmentation consistency ranks ARC programs before "
            "labels become reward data."
        ),
    },
    {
        "arxiv_id": "2603.16140",
        "one_line": (
            "Noisy RLVR says do not train through the 0.32 false-positive "
            "channel as if algorithms will absorb it."
        ),
    },
    {
        "arxiv_id": "2510.00915",
        "one_line": (
            "Imperfect-verifier correction gives explicit noise hooks for FP/FN-calibrated RLVR."
        ),
    },
    {
        "arxiv_id": "2402.06457",
        "one_line": (
            "V-STaR keeps rejected traces to train a verifier rather than "
            "discarding false-positive evidence."
        ),
    },
    {
        "arxiv_id": "2308.01825",
        "one_line": (
            "RFT scaling says rejection-sampled fine-tuning helps weak models "
            "only with clean, diverse positives."
        ),
    },
    {
        "arxiv_id": "2507.14843",
        "one_line": (
            "Invisible Leash makes latent support a gate before RFT/RLVR precision spend."
        ),
    },
    {
        "arxiv_id": "2410.17621",
        "one_line": (
            "Step-level code PRM rewards turn sparse execution outcomes into dense training signal."
        ),
    },
]

DEFAULT_STRONGEST_FOR_NEXT_ROADMAP = [
    "calibrated_forward_noise_correction_before_rlvr",
    "augmentation_consistency_filter_before_rft_corpus",
    "vstar_rejected_trace_verifier_training",
    "step_level_process_reward_weighted_sft",
    "latent_support_gate_before_rft_spend",
]

NOTE_MARKDOWN = """# SOTA ingestion 2026-06-12: precision calibration for verifier-as-reward

**Receipt fields**
- honest_verdict: `complete: sota_ingestion_precision_calibration_mapped`
- inference_substrate: `aggregation_from_upstream_artifacts`
- methods_mapped:
  - {arxiv_id: `2411.02272`, one_line: `BARC-style augmentation consistency ranks ARC programs before labels become reward data.`}
  - {arxiv_id: `2603.16140`, one_line: `Noisy RLVR says do not train through the 0.32 false-positive channel as if algorithms will absorb it.`}
  - {arxiv_id: `2510.00915`, one_line: `Imperfect-verifier correction gives explicit noise hooks for FP/FN-calibrated RLVR.`}
  - {arxiv_id: `2402.06457`, one_line: `V-STaR keeps rejected traces to train a verifier rather than discarding false-positive evidence.`}
  - {arxiv_id: `2308.01825`, one_line: `RFT scaling says rejection-sampled fine-tuning helps weak models only with clean, diverse positives.`}
  - {arxiv_id: `2507.14843`, one_line: `Invisible Leash makes latent support a gate before RFT/RLVR precision spend.`}
  - {arxiv_id: `2410.17621`, one_line: `Step-level code PRM rewards turn sparse execution outcomes into dense training signal.`}
- strongest_for_next_roadmap:
  - `calibrated_forward_noise_correction_before_rlvr`
  - `augmentation_consistency_filter_before_rft_corpus`
  - `vstar_rejected_trace_verifier_training`
  - `step_level_process_reward_weighted_sft`
  - `latent_support_gate_before_rft_spend`

**Fresh-pass provenance**

Read the local verifier-precision, reward, and distillation material in
`research-studying.md`, `research-references.md`, the `.377` verifier-as-reward
SOTA note, and the current `.378` artifacts. The load-bearing ARC value is still
the execution verifier's false-positive channel: Exp 4077 measured
certification precision 0.6818, which means about 0.32 of demo-perfect
certifications were not test-gold. Exp 4087 then rescued the offline operating
point to best certified precision 0.8824 at recall 0.7143, but Exp 4088 blocked
at LoRA smoke checkpoints and Exp 4089 therefore did not run the train. Exp 4093
shows the primitive is not universally broken: OFF-ARC cached demo-fit precision
was 0.9562 raw and 0.9605 after the mutation-consistency filter.

Ran the required discovery helpers:

- `python3 scripts/sweep_clusters.py 0 --max-results 8`
- `python3 scripts/sweep_clusters.py 1 --max-results 8`
- `python3 scripts/sweep_semscholar.py "verifier precision false positives RLVR noisy data process reward execution verifier" --limit 8`
- `python3 scripts/sweep_semscholar.py "BARC augmentation consistency ranking V-STaR rejected traces CodePRM step level process reward" --limit 8`

Semantic Scholar returned zero new arXiv IDs for both focused queries. The
mapped set therefore remains the operator-specified primary-paper set, verified
by low-concurrency WebSearch/WebFetch against the official arXiv pages plus the
ACL page for the CodePRM adjacent implementation. The `/deep-research` loop was
not invoked.

---

## Current precision-rescue + RFT anchor

The current pipeline has three useful facts and one hard blocker. First, the
plain ARC demo-perfect label was too noisy for reward use: 0.6818 precision is a
0.32 false-positive channel. Second, an offline precision-rescue filter can
clear the 0.85 floor on saved GAP-4 programs, with Exp 4087 reporting 0.8824
precision at 0.7143 recall. Third, OFF-ARC code candidates are much cleaner than
the ARC label, with Exp 4093 around 0.96 P(hidden-pass | visible-pass). The hard
blocker is that the RFT train still did not launch because the corpus/trainer
smoke path failed after the precision rescue.

So `.379` should not ask "does RFT work?" yet. It should ask which calibrated
certification rule is stable enough to generate reward data and whether the
RFT-correct arm beats an RFT-ablation arm after that rule is fixed.

---

## Augmentation-consistency ranking / BARC

**Method:** BARC is the system from Combining Induction and Transduction for
Abstract Reasoning (`arXiv:2411.02272`, https://arxiv.org/abs/2411.02272). The
paper trains ARC induction and transduction models on synthetic program
variations and shows the two modes solve complementary problem types. The
operator-relevant hook is its augmentation/test-time reranking pattern: score
candidate transforms by whether they stay consistent under task-preserving
augmentations before treating them as high-confidence solutions.

**Implementation over current precision-rescue + RFT pipeline:** Put an
augmentation-consistency filter between Exp 4087's rescued operating point and
Exp 4088's corpus writer. For every demo-perfect ARC candidate, generate a small
fixed panel of public-example augmentations or leave-one-example pseudo-tests,
execute the same program/transform, and require agreement before the candidate
enters RFT-correct. Hidden labels stay held out for measurement only.

**Pitfalls / where it fails:** Consistency is not truth. A candidate can be
wrong in the same way under every augmentation if the augmentation family is too
weak or if the program overfits the visible examples. This should be a precision
filter and ranker, not a replacement for hidden-test calibration.

---

## Noisy Data is Destructive to RLVR

**Method:** Noisy Data is Destructive to Reinforcement Learning with Verifiable
Rewards (`arXiv:2603.16140`, https://arxiv.org/abs/2603.16140) is the strongest
negative result for the current temptation. Its arXiv abstract says the authors
re-verified previously claimed noisy data and found that truly incorrect
annotations make RLVR worse; existing RLVR improvements did not compensate for
poor data quality.

**Implementation over current precision-rescue + RFT pipeline:** Treat this as
the no-train-through-noise rule. The RFT corpus builder should expose the
estimated false-positive rate, reject the corpus when precision is below the
floor, and include a clean-label ablation rather than relying on GRPO/RLVR to
absorb the 0.32 ARC false-positive channel.

**Pitfalls / where it fails:** A strict clean-only policy can starve `.379` if
precision is bought by throwing away too much recall. The right acceptance gate
is precision floor plus retained-support minimum, not precision alone.

---

## Imperfect-verifier noise correction

**Method:** Reinforcement Learning with Verifiable yet Noisy Rewards under
Imperfect Verifiers (`arXiv:2510.00915`, https://arxiv.org/abs/2510.00915)
formalizes verifier errors as asymmetric false-positive and false-negative
rates. It proposes lightweight backward and forward corrections for RLVR, with
the forward correction needing only an FN estimate and proving more stable under
heavier noise.

**Implementation over current precision-rescue + RFT pipeline:** Add an
explicit verifier-noise calibration object to the reward record: `fp_rate`,
`fn_rate`, confidence intervals, source split, and whether rates were measured
pre-policy or post-policy. For supervised RFT, use this first as sample weights
or abstention thresholds. For later RLVR, wire the forward correction into the
GRPO advantage/reward hook only after the held-out rate estimates are stable.

**Pitfalls / where it fails:** Noise correction is only as good as its rate
estimates. The rates can shift after the policy learns to target the verifier's
boundary, so `.379` needs recalibration after each train or must stop at offline
RFT until post-policy noise is measured.

---

## V-STaR keep rejected traces

**Method:** V-STaR: Training Verifiers for Self-Taught Reasoners
(`arXiv:2402.06457`, https://arxiv.org/abs/2402.06457) keeps both correct and
incorrect self-generated solutions, trains a verifier with DPO, and uses that
verifier to select among candidates. The relevant move is not merely more
self-training; it is preserving rejected traces as training signal for the
verifier.

**Implementation over current precision-rescue + RFT pipeline:** Stop treating
demo-perfect rejects and hidden-fail candidates as trash. Store candidate,
visible verdict, mutation-consistency verdict, hidden/test-gold label when
available, parser failure, and verifier reason. Train a small selection verifier
or calibration head on accepted-vs-rejected pairs, then compare its selected
RFT-correct corpus against the current hard rule.

**Pitfalls / where it fails:** Rejected traces mix true negatives, verifier false
negatives, parser misses, and candidates outside the current DSL. Without
reason-coded buckets, the learned verifier can inherit the current verifier's
blind spots and become a smoother false-positive generator.

---

## RFT scaling for weak models

**Method:** Scaling Relationship on Learning Mathematical Reasoning with Large
Language Models (`arXiv:2308.01825`, https://arxiv.org/abs/2308.01825) is the
rejection-sampling fine-tuning baseline. It finds RFT benefits from collecting
correct reasoning paths as augmented fine-tuning data, with more diverse
reasoning paths helping more and weaker models benefiting more than stronger
ones.

**Implementation over current precision-rescue + RFT pipeline:** Keep RFT as the
simple first train once labels are clean, especially for the weak local ARC
model. But the corpus must be deduplicated by transform family and difficulty
bucket, not just accepted by the verifier. The right `.379` training table is
clean positives, rejected-trace contrast data, and an RFT-ablation arm from the
same generator pool.

**Pitfalls / where it fails:** RFT amplifies the generator's support. If the
base model never generated the correct transform, RFT cannot learn it from
filtered self-samples. It also overfits easy accepted patterns unless diversity
and held-out induction buckets are explicit.

---

## Invisible Leash support gate

**Method:** The Invisible Leash (`arXiv:2507.14843`,
https://arxiv.org/abs/2507.14843) argues that RLVR often improves pass@1 by
sharpening existing support while narrowing exploration, rather than expanding a
model's reasoning boundary.

**Implementation over current precision-rescue + RFT pipeline:** Before `.379`
spends on RFT/RLVR, measure same-pool support: cold pass@1, pass@k/oracle,
number of distinct valid transform families, and how often a hidden-gold
candidate appears anywhere in the generated pool. If support is absent, route to
external verified traces or a stronger local base; if support is present but
mis-ranked, use precision rescue plus RFT.

**Pitfalls / where it fails:** A support gate can become a pessimistic blocker
if k is too small or the generator budget is underpowered. The gate should be
budgeted enough to distinguish absent capability from a bad ranker.

---

## Step-level process reward for code

**Method:** Process Supervision-Guided Policy Optimization for Code Generation
(`arXiv:2410.17621`, https://arxiv.org/abs/2410.17621) replaces sparse unit-test
reward with dense line-level process feedback for code generation. The adjacent
CodePRM paper (https://aclanthology.org/2025.findings-acl.428/) makes the same
implementation direction concrete by using execution feedback to score thought
steps and support generate-verify-refine.

**Implementation over current precision-rescue + RFT pipeline:** Convert the
binary demo-perfect label into a process-reward record: visible test pass,
mutation-probe pass, per-step state/action legality, exact state equivalence,
hidden-test label when available, and abstention reason. Start with
process-reward-weighted SFT before trying policy optimization.

**Pitfalls / where it fails:** Dense rewards can reward local progress that does
not improve final hidden correctness. The process reward needs an outcome-tied
calibration check, or the model may learn valid-looking but non-solving edits.

---

## Bottom line for the .379 roadmap

1. **First method to implement:** `calibrated_forward_noise_correction_before_rlvr`.
   Add explicit FP/FN rate estimates and sample weighting/abstention before any
   RLVR step. Use `arXiv:2510.00915` as the correction template and
   `arXiv:2603.16140` as the stop-rule: do not trust RLVR to absorb dirty
   verifier labels.
2. **Best precision-rescue upgrade:** `augmentation_consistency_filter_before_rft_corpus`.
   Add BARC-style augmentation consistency as a pre-corpus filter on top of Exp
   4087, then remeasure P(test-gold | certified) and retained recall.
3. **Best use of the current false positives:** `vstar_rejected_trace_verifier_training`.
   Preserve hidden-fail and verifier-rejected traces with reason codes; use them
   to train or calibrate the selector instead of discarding the evidence.
4. **Best training shape after the gate passes:** `step_level_process_reward_weighted_sft`.
   Replace a single hard demo-perfect label with process-reward-weighted SFT,
   then only promote to GRPO/RLVR if the RFT-correct arm beats RFT-ablation.
5. **Gate every train with support:** `latent_support_gate_before_rft_spend`.
   If the correct transform is absent from the same generated pool, RFT is the
   wrong tool; route to stronger local base support or external verified traces.
"""

STUDYING_SECTION = """## 2026-06-12 Exp 4094 - .378 precision-calibration SOTA ingestion ingested

**Status:** INGESTED into `docs/research-notes/sota-ingestion-precision-calibration-2026-06-12.md`.

**Filtered track:** the `.378` verifier-precision / verifier-as-reward headline:
the 0.32 false-positive channel behind ARC certification precision 0.6818, the
Exp 4087 precision rescue to 0.8824 at 0.7143 recall, the blocked Exp 4088/4089
RFT path, and the Exp 4093 OFF-ARC demo-fit precision replay.

**Seed and fresh-pass candidates marked ingested:**
- BARC / Combining Induction and Transduction, arXiv:2411.02272 - mapped as an
  augmentation-consistency filter before RFT corpus admission.
- Noisy Data is Destructive to RLVR, arXiv:2603.16140 - mapped as the stop-rule
  against training through the 0.32 false-positive channel.
- RLVR with imperfect verifiers, arXiv:2510.00915 - mapped as explicit FP/FN
  noise correction and calibration metadata for future RLVR hooks.
- V-STaR, arXiv:2402.06457 - mapped as rejected-trace retention and verifier
  training over accepted/rejected pairs.
- RFT scaling, arXiv:2308.01825 - retained as the simple fine-tuning baseline
  only after clean, diverse positives exist.
- Invisible Leash, arXiv:2507.14843 - retained as the same-pool latent-support
  gate before RFT/RLVR spend.
- Process Supervision-Guided Policy Optimization for Code Generation,
  arXiv:2410.17621, plus CodePRM ACL 2025 - mapped as step-level process reward
  records before sparse reward RL.

**Bottom line for the .379 roadmap:** prioritize
`calibrated_forward_noise_correction_before_rlvr`,
`augmentation_consistency_filter_before_rft_corpus`,
`vstar_rejected_trace_verifier_training`, and
`step_level_process_reward_weighted_sft`; keep
`latent_support_gate_before_rft_spend` as the launch gate so RFT is only used
when correct transforms are already present in the generated pool.
"""

STUDYING_MARKER = "## 2026-06-12 Exp 4094 - .378 precision-calibration SOTA ingestion ingested"


def build_receipt(
    *,
    methods_mapped: Sequence[Mapping[str, str]],
    strongest_for_next_roadmap: Sequence[str],
    honest_verdict: str = DEFAULT_HONEST_VERDICT,
) -> dict[str, object]:
    """Build and validate the REQ-REPORT-4094 receipt payload."""

    receipt: dict[str, object] = {
        "honest_verdict": honest_verdict,
        "methods_mapped": [dict(method) for method in methods_mapped],
        "strongest_for_next_roadmap": list(strongest_for_next_roadmap),
        "inference_substrate": EXPECTED_INFERENCE_SUBSTRATE,
    }
    validate_receipt(receipt)
    return receipt


def validate_receipt(receipt: Mapping[str, Any]) -> None:
    """Validate the exact receipt contract for SCENARIO-REPORT-4094."""

    missing = REQUIRED_RECEIPT_FIELDS.difference(receipt)
    extra = set(receipt).difference(REQUIRED_RECEIPT_FIELDS)
    if missing:
        raise ValueError(f"receipt missing required fields: {sorted(missing)}")
    if extra:
        raise ValueError(f"receipt has unexpected fields: {sorted(extra)}")

    verdict = receipt["honest_verdict"]
    if not isinstance(verdict, str) or not verdict.startswith(TERMINAL_PREFIXES):
        raise ValueError("honest_verdict must use a terminal prefix")

    methods_mapped = receipt["methods_mapped"]
    if not isinstance(methods_mapped, list) or not methods_mapped:
        raise ValueError("methods_mapped must be a non-empty list")
    if not 5 <= len(methods_mapped) <= 8:
        raise ValueError("methods_mapped must contain five to eight methods")

    seen: set[str] = set()
    for method in methods_mapped:
        if not isinstance(method, dict) or set(method) != {"arxiv_id", "one_line"}:
            raise ValueError("each method must contain exactly arxiv_id and one_line")
        arxiv_id = method["arxiv_id"]
        one_line = method["one_line"]
        if arxiv_id not in VERIFIED_ARXIV_IDS:
            raise ValueError(f"method arxiv_id must be a verified arxiv ID: {arxiv_id}")
        if arxiv_id in seen:
            raise ValueError(f"duplicate method arxiv_id: {arxiv_id}")
        seen.add(arxiv_id)
        if not isinstance(one_line, str) or not one_line.strip():
            raise ValueError("method one_line must be a non-empty string")

    strongest = receipt["strongest_for_next_roadmap"]
    if (
        not isinstance(strongest, list)
        or not strongest
        or not all(isinstance(item, str) and item for item in strongest)
    ):
        raise ValueError("strongest_for_next_roadmap must be a non-empty list of strings")

    if receipt["inference_substrate"] != EXPECTED_INFERENCE_SUBSTRATE:
        raise ValueError(f"inference_substrate must be {EXPECTED_INFERENCE_SUBSTRATE!r}")


def validate_markdown_note(markdown: str) -> None:
    """Check that the paired note has the required mapping and planning shape."""

    required_phrases = (
        "Current precision-rescue + RFT anchor",
        "Augmentation-consistency ranking / BARC",
        "Noisy Data is Destructive to RLVR",
        "Imperfect-verifier noise correction",
        "V-STaR keep rejected traces",
        "RFT scaling for weak models",
        "Invisible Leash support gate",
        "Step-level process reward for code",
        "Implementation over current precision-rescue + RFT pipeline",
        "Pitfalls / where it fails",
        "Bottom line for the .379 roadmap",
    )
    missing_phrases = [phrase for phrase in required_phrases if phrase not in markdown]
    if missing_phrases:
        raise ValueError(f"markdown note missing required sections: {missing_phrases}")

    missing_ids = [
        arxiv_id
        for arxiv_id in VERIFIED_ARXIV_IDS
        if f"arXiv:{arxiv_id}" not in markdown and f"arxiv.org/abs/{arxiv_id}" not in markdown
    ]
    if missing_ids:
        raise ValueError(f"markdown note missing verified arxiv citations: {missing_ids}")


def write_outputs(
    *,
    note_path: Path,
    receipt_path: Path,
    studying_path: Path,
) -> dict[str, object]:
    """Write the note, receipt, and idempotent studying-section update."""

    receipt = build_receipt(
        methods_mapped=DEFAULT_METHODS_MAPPED,
        strongest_for_next_roadmap=DEFAULT_STRONGEST_FOR_NEXT_ROADMAP,
    )
    validate_markdown_note(NOTE_MARKDOWN)

    note_path.parent.mkdir(parents=True, exist_ok=True)
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    note_path.write_text(NOTE_MARKDOWN + "\n", encoding="utf-8")
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    existing = studying_path.read_text(encoding="utf-8")
    studying_path.write_text(_with_studying_section(existing), encoding="utf-8")
    return receipt


def _with_studying_section(existing: str) -> str:
    if STUDYING_MARKER not in existing:
        if "\n## " not in existing:
            return existing.rstrip() + "\n\n" + STUDYING_SECTION
        return existing.replace("\n## ", "\n" + STUDYING_SECTION + "\n## ", 1)

    before, after_marker = existing.split(STUDYING_MARKER, 1)
    next_section = after_marker.find("\n## ")
    if next_section == -1:
        return before + STUDYING_SECTION.rstrip() + "\n"
    return before + STUDYING_SECTION + after_marker[next_section + 1 :]
