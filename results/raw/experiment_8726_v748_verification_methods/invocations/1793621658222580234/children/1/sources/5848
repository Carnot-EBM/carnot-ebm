"""Schema helpers for the Exp 4102 TRM self-training SOTA ingestion.

Spec refs: REQ-REPORT-4102, SCENARIO-REPORT-4102.

The artifact is a planning bridge, not a model result. It records which
self-training-with-verifier papers are credible enough to map onto the current
`nano-trm` plus Carnot-verifier stack, what each would require, and which single
method should be handed to the next roadmap without re-discovering the same
literature.
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
        "flagged_for_v380",
    }
)
REQUIRED_METHOD_FIELDS = frozenset(
    {
        "name",
        "arxiv_id",
        "url",
        "implementation_over_stack",
        "failure_mode",
    }
)
TERMINAL_PREFIXES = ("complete:", "success:", "passed:", "shipped:", "blocked_")
DEFAULT_HONEST_VERDICT = "complete: sota_ingestion_trm_self_training_mapped"
DEFAULT_FLAGGED_FOR_V380 = "vstar_rejected_trace_selector_for_trm_rft"

VERIFIED_ARXIV_IDS = frozenset(
    {
        "2203.14465",
        "2308.08998",
        "2402.06457",
        "2510.00915",
        "2511.02886",
        "2601.17223",
        "2605.10325",
        "2605.30290",
    }
)

NOTE_REQUIRED_ARXIV_IDS = frozenset(
    {
        "2203.14465",
        "2308.08998",
        "2402.06457",
        "2510.00915",
        "2511.02886",
        "2601.17223",
        "2605.10325",
        "2605.30290",
    }
)

DEFAULT_METHODS_MAPPED = [
    {
        "name": "V-STaR keep-rejected verifier training",
        "arxiv_id": "2402.06457",
        "url": "https://arxiv.org/abs/2402.06457",
        "implementation_over_stack": (
            "Label nano-trm candidate traces with Carnot verifier outcomes and train a "
            "contrastive selector on accepted and rejected traces."
        ),
        "failure_mode": (
            "False-positive verifier labels turn rejected-trace learning into reward "
            "hacking unless calibration gates hold first."
        ),
    },
    {
        "name": "STaR / ReST generate-filter-improve loop",
        "arxiv_id": "2203.14465",
        "url": "https://arxiv.org/abs/2203.14465",
        "implementation_over_stack": (
            "Iterate nano-trm sampling, verifier filtering, and full fine-tuning from "
            "the reusable cached ARC trace pool."
        ),
        "failure_mode": (
            "The loop only amplifies traces already in the model support and discards "
            "hard negative structure."
        ),
    },
    {
        "name": "TTA-TRM full fine-tuning with verifier admission",
        "arxiv_id": "2511.02886",
        "url": "https://arxiv.org/abs/2511.02886",
        "implementation_over_stack": (
            "Use public-task pretraining plus bounded full fine-tuning, with Carnot "
            "verifier precision gates controlling which task traces enter adaptation."
        ),
        "failure_mode": (
            "It can become task memorization or leakage if public/private splits and "
            "full-finetune budgets are not isolated."
        ),
    },
    {
        "name": "Imperfect-verifier forward correction",
        "arxiv_id": "2510.00915",
        "url": "https://arxiv.org/abs/2510.00915",
        "implementation_over_stack": (
            "Attach FP/FN calibration metadata to verifier-certified TRM rewards and "
            "weight updates instead of treating the verifier as noiseless."
        ),
        "failure_mode": (
            "Noise-rate estimates drift after the TRM policy changes, so stale "
            "correction can bias the next RFT round."
        ),
    },
    {
        "name": "Verifiable process rewards for recursive steps",
        "arxiv_id": "2605.10325",
        "url": "https://arxiv.org/abs/2605.10325",
        "implementation_over_stack": (
            "Score each recursive grid-edit step with deterministic Carnot verifier "
            "checks before outcome-level hidden-test selection."
        ),
        "failure_mode": (
            "Locally valid recursive edits can still fail the final ARC transformation "
            "unless dense rewards are outcome-calibrated."
        ),
    },
]

NOTE_MARKDOWN = """# SOTA ingestion 2026-06-12: TRM self-training with verifiers

**Artifact fields**
- honest_verdict: `complete: sota_ingestion_trm_self_training_mapped`
- methods_mapped:
  - {name: `V-STaR keep-rejected verifier training`, arxiv_id: `2402.06457`, url: `https://arxiv.org/abs/2402.06457`}
  - {name: `STaR / ReST generate-filter-improve loop`, arxiv_id: `2203.14465`, url: `https://arxiv.org/abs/2203.14465`}
  - {name: `TTA-TRM full fine-tuning with verifier admission`, arxiv_id: `2511.02886`, url: `https://arxiv.org/abs/2511.02886`}
  - {name: `Imperfect-verifier forward correction`, arxiv_id: `2510.00915`, url: `https://arxiv.org/abs/2510.00915`}
  - {name: `Verifiable process rewards for recursive steps`, arxiv_id: `2605.10325`, url: `https://arxiv.org/abs/2605.10325`}
- flagged_for_v380: `vstar_rejected_trace_selector_for_trm_rft`

**Fresh-pass provenance**

Read the local verifier-RFT and self-training track in `research-studying.md`
and `research-references.md`, including the `.377` verifier-as-reward ingestion,
the `.378` precision-calibration ingestion, and the TRM recursive-refiner entries
around `arXiv:2511.02886`. Ran the required helpers:

- `python3 scripts/sweep_clusters.py 0 --max-results 8`
- `python3 scripts/sweep_clusters.py 1 --max-results 8`
- `python3 scripts/sweep_semscholar.py "self training verifier recursive reasoner TRM RFT V-STaR ReST process reward" --limit 8`
- `python3 scripts/sweep_semscholar.py "Tiny Recursive Models test time adaptation verifier reward self training" --limit 8`
- `python3 scripts/sweep_semscholar.py "imperfect verifier noisy verifiable rewards RLVR process reward self training" --limit 8`

Semantic Scholar rate-limited two of the focused queries and returned
`arXiv:2603.02203` plus `arXiv:2602.05570` for the TRM-adaptation query; neither
displaced the operator-specified verifier-RFT anchors. Low-concurrency
WebSearch/WebFetch then verified the primary arXiv pages for `arXiv:2402.06457`,
`arXiv:2203.14465`, `arXiv:2308.08998`, `arXiv:2511.02886`,
`arXiv:2510.00915`, `arXiv:2601.17223`, `arXiv:2605.10325`, and the fresh
adjacent verifier-training paper `arXiv:2605.30290`. The `/deep-research` loop
was not invoked.

## Current .379 TRM verifier-RFT anchor

The `.379` headline is no longer generic "verifier-as-reward." It is
verifier-certified RFT of a recursive reasoner: a `nano-trm`/TRM-style model
generates candidate grid transformations or recursive edit traces, the Carnot
verifier stack certifies or rejects them, and training must improve the recursive
model rather than only rerank a fixed candidate pool.

That makes the load-bearing question narrower than prior SOTA ingestions. The
method must answer: which traces enter full fine-tuning, how rejected traces are
used instead of thrown away, how verifier noise is corrected, and whether dense
per-recursion feedback can be trusted without losing hidden-test correctness.
`arXiv:2605.30290` is important adjacent evidence because it frames verifier
quality as the bottleneck for both test-time refinement and training-time
self-improvement, but the first `.380` candidate should stay closer to the
existing accepted/rejected TRM trace pool.

## V-STaR keep-rejected verifier training

**Method:** V-STaR, `arXiv:2402.06457`
(https://arxiv.org/abs/2402.06457), trains a verifier from both correct and
incorrect self-generated solutions rather than discarding failures, then uses
that verifier to select among candidates.

**Implementation over nano-trm + Carnot-verifier stack:** Keep every sampled
TRM trace from the same ARC/Sudoku task pool: verifier-certified pass,
verifier-rejected, hidden-fail, parser-fail, and timeout. Convert pairs from
the same prompt into contrastive selector data: accepted trace should score
above rejected trace when the downstream hidden label confirms the verifier.
Use the selector first as a reranker and then as a corpus-admission gate for a
small full-fine-tune RFT arm. This is the cleanest way to turn Carnot's
rejected evidence into training signal without immediately changing the TRM
generator.

**Pitfalls / where it fails:** V-STaR assumes the accept/reject labels contain
real ranking information. If the Carnot verifier's false-positive channel is
not below the `.378` precision floor, DPO-style contrast training will teach the
selector to prefer verifier artifacts rather than hidden-correct transformations.
It also needs trace diversity; if nano-trm emits near-duplicate wrong traces,
the selector learns surface features rather than semantic repair.

## STaR / ReST generate-filter-improve loop

**Method:** STaR, `arXiv:2203.14465`
(https://arxiv.org/abs/2203.14465), iteratively generates rationales, keeps
those that yield correct answers, fine-tunes, and repeats. ReST,
`arXiv:2308.08998` (https://arxiv.org/abs/2308.08998), gives the offline
generate/filter/improve cadence with reusable batches and stronger filtering.

**Implementation over nano-trm + Carnot-verifier stack:** Treat a TRM recursive
trace as the rationale analogue. Run bounded candidate generation, filter with
the Carnot verifier plus hidden labels where available, train a full-fine-tune
TRM arm on unique certified traces, then regenerate from the updated TRM. Cache
all batches so the next improve step can use a stricter acceptance threshold
without paying for new sampling immediately.

**Pitfalls / where it fails:** STaR/ReST improve support that already exists.
If a correct ARC transform never appears in the candidate pool, verifier
filtering cannot invent it. The method also wastes rejected traces unless
combined with the V-STaR selector, and it can overfit public ARC task variants
if augmentation families are not held out.

## TTA-TRM full fine-tuning

**Method:** Test-time Adaptation of Tiny Recursive Models,
`arXiv:2511.02886` (https://arxiv.org/abs/2511.02886), reports that public-task
pretraining plus bounded full fine-tuning can adapt a 7M TRM, and explicitly
notes that full fine-tuning outperformed LoRA or task-embedding-only adaptation
for that setting.

**Implementation over nano-trm + Carnot-verifier stack:** Use the public
nano-trm training tasks as the pretraining/adaptation split. Keep the
competition-like budget explicit: number of optimizer steps, task count, and
wall-clock. Apply Carnot verifier gates before a trace can enter the
fine-tuning set, and keep a no-RFT full-fine-tune control so any gain is not
misattributed to the verifier when it came from task adaptation alone.

**Pitfalls / where it fails:** This is the substrate method, not a verifier
method by itself. It can "win" by memorizing public task structure or spending
more adaptation compute, and it can erase the planned verifier contribution if
the experiment does not isolate full fine-tune, verifier admission, and
reranking arms.

## Imperfect-verifier correction

**Method:** Reinforcement Learning with Verifiable yet Noisy Rewards under
Imperfect Verifiers, `arXiv:2510.00915`
(https://arxiv.org/abs/2510.00915), models verifier rewards as an asymmetric
false-positive/false-negative channel and adds backward or forward correction
hooks; the forward correction is the lighter-weight candidate because it mainly
needs a false-negative estimate.

**Implementation over nano-trm + Carnot-verifier stack:** Every verifier
certificate should carry `fp_rate`, `fn_rate`, calibration split, confidence
interval, and source verifier. Use those rates to downweight or abstain on
borderline TRM traces before RFT, and reserve a small appeal path where a
stronger checker re-examines rule-based negatives. This belongs before any
policy-gradient RLVR attempt and also informs weighted SFT.

**Pitfalls / where it fails:** The correction only helps if the noise rates
match the current generator distribution. Once full fine-tuning changes the TRM
trace distribution, stale FP/FN rates can become actively misleading. It also
does not solve absent support: cleanly correcting verifier noise cannot train a
trace the generator never produced.

## Verifiable process rewards

**Method:** VPRM, `arXiv:2601.17223`
(https://arxiv.org/abs/2601.17223), and VPR for agentic reasoning,
`arXiv:2605.10325` (https://arxiv.org/abs/2605.10325), replace sparse
outcome-only rewards with deterministic step or turn checks where the task
structure permits objective intermediate verification.

**Implementation over nano-trm + Carnot-verifier stack:** Add per-recursion
telemetry to TRM traces: current grid, proposed edit, latent halt decision,
visible-example consistency, exact-state equivalence, mutation consistency, and
final hidden outcome when available. Start with process-reward-weighted SFT or
reranking; only promote to RLVR if dense rewards predict final hidden
correctness on a held-out calibration split.

**Pitfalls / where it fails:** ARC intermediate states are often
underdetermined. A locally consistent edit can preserve all public examples and
still fail the intended transformation. Dense reward should therefore be a
credit-assignment aid, not a replacement for final hidden-test calibration.

## Flagged for the .380 roadmap

`vstar_rejected_trace_selector_for_trm_rft` is the strongest single `.380`
candidate. It uses evidence the `.379` TRM verifier-RFT run already produces,
turns both successful and failed traces into a selector training set, and stays
compatible with TTA-TRM full fine-tuning, imperfect-verifier correction, and
later process rewards. The first `.380` experiment should build this selector
over the saved nano-trm candidate pool and require a rerank win before allowing
the selector to gate a second full-fine-tune RFT corpus.
"""

STUDYING_SECTION = """## 2026-06-12 Exp 4102 - .379 TRM self-training SOTA ingestion ingested

**Status:** INGESTED into `docs/research-notes/sota-ingestion-trm-self-training-2026-06-12.md`.

**Filtered track:** verifier-certified RFT over a recursive `nano-trm`/TRM
substrate, with Carnot verifier labels selecting, correcting, or densifying the
training signal.

**Seed and fresh-pass candidates marked ingested:**
- V-STaR, arXiv:2402.06457 - mapped as accepted/rejected TRM trace selector
  training before any second RFT corpus gate.
- STaR, arXiv:2203.14465, and ReST, arXiv:2308.08998 - mapped as the cached
  generate-filter-improve cadence for recursive traces.
- TTA-TRM, arXiv:2511.02886 - mapped as the full-fine-tune substrate and a
  control against attributing adaptation-only gains to the verifier.
- RLVR with imperfect verifiers, arXiv:2510.00915 - mapped as FP/FN-calibrated
  weighting and abstention before verifier-certified RFT.
- VPRM/VPR, arXiv:2601.17223 and arXiv:2605.10325 - mapped as dense
  per-recursion step rewards only after outcome calibration.
- Self-Trained Verification, arXiv:2605.30290 - marked as fresh adjacent
  verifier-training evidence, but deferred behind the cheaper V-STaR trace
  selector because `.379` already emits accepted/rejected TRM traces.

Flagged for .380: `vstar_rejected_trace_selector_for_trm_rft`.

**Bottom line for the .380 roadmap:** build a V-STaR-style selector over the
saved nano-trm candidate pool, require a rerank win against the current Carnot
verifier ordering, and only then let the selector gate a second full-fine-tune
RFT corpus.
"""

STUDYING_MARKER = (
    "## 2026-06-12 Exp 4102 - .379 TRM self-training SOTA ingestion ingested"
)


def build_artifact(
    *,
    methods_mapped: Sequence[Mapping[str, str]],
    flagged_for_v380: str,
    honest_verdict: str = DEFAULT_HONEST_VERDICT,
) -> dict[str, object]:
    """Build and validate the REQ-REPORT-4102 mapping artifact."""

    artifact: dict[str, object] = {
        "honest_verdict": honest_verdict,
        "methods_mapped": [dict(method) for method in methods_mapped],
        "flagged_for_v380": flagged_for_v380,
    }
    validate_artifact(artifact)
    return artifact


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """Validate the exact JSON contract so uncited method rows fail closed."""

    missing = REQUIRED_ARTIFACT_FIELDS.difference(artifact)
    extra = set(artifact).difference(REQUIRED_ARTIFACT_FIELDS)
    if missing:
        raise ValueError(f"artifact missing required fields: {sorted(missing)}")
    if extra:
        raise ValueError(f"artifact has unexpected fields: {sorted(extra)}")

    verdict = artifact["honest_verdict"]
    if not isinstance(verdict, str) or not verdict.startswith(TERMINAL_PREFIXES):
        raise ValueError("honest_verdict must use a terminal prefix")

    methods_mapped = artifact["methods_mapped"]
    if not isinstance(methods_mapped, list) or not 3 <= len(methods_mapped) <= 5:
        raise ValueError("methods_mapped must contain three to five methods")

    seen: set[str] = set()
    for method in methods_mapped:
        if not isinstance(method, dict) or set(method) != REQUIRED_METHOD_FIELDS:
            raise ValueError(
                "each method must contain exactly name, arxiv_id, url, "
                "implementation_over_stack, and failure_mode"
            )
        arxiv_id = method["arxiv_id"]
        if arxiv_id not in VERIFIED_ARXIV_IDS:
            raise ValueError(f"method arxiv_id must be a verified arxiv ID: {arxiv_id}")
        if arxiv_id in seen:
            raise ValueError(f"duplicate method arxiv_id: {arxiv_id}")
        seen.add(arxiv_id)
        expected_url = f"https://arxiv.org/abs/{arxiv_id}"
        if method["url"] != expected_url:
            raise ValueError(f"method url must be {expected_url!r}")
        for field in REQUIRED_METHOD_FIELDS - {"arxiv_id", "url"}:
            value = method[field]
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"method {field} must be a non-empty string")

    flagged = artifact["flagged_for_v380"]
    if not isinstance(flagged, str) or not flagged.strip():
        raise ValueError("flagged_for_v380 must be a non-empty string")


def validate_markdown_note(markdown: str) -> None:
    """Check that the paired note maps papers to TRM work and closes planning."""

    required_phrases = (
        "Current .379 TRM verifier-RFT anchor",
        "V-STaR keep-rejected verifier training",
        "STaR / ReST generate-filter-improve loop",
        "TTA-TRM full fine-tuning",
        "Imperfect-verifier correction",
        "Verifiable process rewards",
        "Implementation over nano-trm + Carnot-verifier stack",
        "Pitfalls / where it fails",
        "Flagged for the .380 roadmap",
    )
    missing_phrases = [phrase for phrase in required_phrases if phrase not in markdown]
    if missing_phrases:
        raise ValueError(f"markdown note missing required sections: {missing_phrases}")

    missing_ids = [
        arxiv_id
        for arxiv_id in NOTE_REQUIRED_ARXIV_IDS
        if f"arXiv:{arxiv_id}" not in markdown
        and f"arxiv.org/abs/{arxiv_id}" not in markdown
    ]
    if missing_ids:
        raise ValueError(f"markdown note missing verified arxiv citations: {missing_ids}")


def write_outputs(
    *,
    note_path: Path,
    artifact_path: Path,
    studying_path: Path,
) -> dict[str, object]:
    """Write the note, JSON artifact, and idempotent studying-section update."""

    artifact = build_artifact(
        methods_mapped=DEFAULT_METHODS_MAPPED,
        flagged_for_v380=DEFAULT_FLAGGED_FOR_V380,
    )
    validate_markdown_note(NOTE_MARKDOWN)

    note_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    note_path.write_text(NOTE_MARKDOWN + "\n", encoding="utf-8")
    artifact_path.write_text(
        json.dumps(artifact, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    existing = studying_path.read_text(encoding="utf-8")
    studying_path.write_text(_with_studying_section(existing), encoding="utf-8")
    return artifact


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
