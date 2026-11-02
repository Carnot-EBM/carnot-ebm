"""Schema helpers for the Exp 4111 TRM verifier-training SOTA ingestion.

Spec refs: REQ-REPORT-4111, SCENARIO-REPORT-4111.

The artifact is a planning bridge, not a model result. It records which TRM,
self-training, and verifier-guided search papers are credible enough to map
onto the current `nano-trm` plus Carnot Sudoku-verifier stack after the `.380`
baseline-plus-verifier run produced an honest null.
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
        "flagged_for_v381",
        "field_principles",
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
DEFAULT_HONEST_VERDICT = "complete: sota_ingestion_trm_verifier_training_mapped"
DEFAULT_FLAGGED_FOR_V381 = "verifier_guided_adaptive_sudoku_search_before_training"

FIELD_PRINCIPLES = {
    "honest_verdict": "Terminal-prefixed. Records ingestion completed with verifiable citations.",
    "methods_mapped": (
        "Each method MUST carry a real arXiv ID/URL; an ingestion note without "
        "verifiable citations is treated as fabrication."
    ),
    "flagged_for_v381": (
        "Closes the discover->ingest->plan loop: names the strongest method "
        "for the next planner."
    ),
}

VERIFIED_ARXIV_IDS = frozenset(
    {
        "2203.14465",
        "2308.08998",
        "2402.06457",
        "2510.04871",
        "2511.02886",
        "2601.17223",
        "2602.01070",
        "2605.10325",
    }
)

NOTE_REQUIRED_ARXIV_IDS = frozenset(
    {
        "2203.14465",
        "2308.08998",
        "2402.06457",
        "2510.04871",
        "2511.02886",
        "2601.17223",
        "2602.01070",
        "2605.10325",
    }
)

DEFAULT_METHODS_MAPPED = [
    {
        "name": "TRM Sudoku baseline reproduction",
        "arxiv_id": "2510.04871",
        "url": "https://arxiv.org/abs/2510.04871",
        "implementation_over_stack": (
            "Reproduce the native nano-trm Sudoku Extreme baseline before claiming "
            "any Carnot verifier lift."
        ),
        "failure_mode": (
            "A partial checkpoint can validate the mechanism while still failing to "
            "match the published accuracy target."
        ),
    },
    {
        "name": "TTA-TRM full fine-tuning control",
        "arxiv_id": "2511.02886",
        "url": "https://arxiv.org/abs/2511.02886",
        "implementation_over_stack": (
            "Use bounded full fine-tuning as the adaptation control so verifier "
            "admission is separated from generic task adaptation."
        ),
        "failure_mode": (
            "Compute leakage or public-task memorization can look like verifier value "
            "unless full-finetune and no-verifier arms are isolated."
        ),
    },
    {
        "name": "V-STaR accepted/rejected trace selector",
        "arxiv_id": "2402.06457",
        "url": "https://arxiv.org/abs/2402.06457",
        "implementation_over_stack": (
            "Train a selector from verifier-valid and verifier-invalid Sudoku "
            "candidate traces sampled from the same TRM checkpoint."
        ),
        "failure_mode": (
            "If candidate pools are near-duplicates or already vote-saturated, the "
            "selector learns surface artifacts without pass@1 lift."
        ),
    },
    {
        "name": "STaR / ReST generate-filter-improve loop",
        "arxiv_id": "2203.14465",
        "url": "https://arxiv.org/abs/2203.14465",
        "implementation_over_stack": (
            "Generate Sudoku traces, filter with the executable Carnot verifier, "
            "fine-tune on unique positives, and repeat from cached batches."
        ),
        "failure_mode": (
            "Filtering cannot teach solutions the TRM never samples, and sparse "
            "positives can collapse the improve step."
        ),
    },
    {
        "name": "Verifier-guided adaptive Sudoku search",
        "arxiv_id": "2602.01070",
        "url": "https://arxiv.org/abs/2602.01070",
        "implementation_over_stack": (
            "Move the Sudoku verifier into candidate expansion so compute is spent "
            "on promising partial completions before post-hoc reranking or RFT."
        ),
        "failure_mode": (
            "Local row, column, and box satisfaction can still prefer near-valid "
            "dead ends unless final exact validity remains authoritative."
        ),
    },
]

NOTE_MARKDOWN = """# SOTA ingestion 2026-06-12: TRM baseline plus verifier-guided training

**Artifact fields**
- honest_verdict: `complete: sota_ingestion_trm_verifier_training_mapped`
  - principle: Terminal-prefixed. Records ingestion completed with verifiable citations.
- methods_mapped:
  - {name: `TRM Sudoku baseline reproduction`, arxiv_id: `2510.04871`, url: `https://arxiv.org/abs/2510.04871`}
  - {name: `TTA-TRM full fine-tuning control`, arxiv_id: `2511.02886`, url: `https://arxiv.org/abs/2511.02886`}
  - {name: `V-STaR accepted/rejected trace selector`, arxiv_id: `2402.06457`, url: `https://arxiv.org/abs/2402.06457`}
  - {name: `STaR / ReST generate-filter-improve loop`, arxiv_id: `2203.14465`, url: `https://arxiv.org/abs/2203.14465`}
  - {name: `Verifier-guided adaptive Sudoku search`, arxiv_id: `2602.01070`, url: `https://arxiv.org/abs/2602.01070`}
  - principle: Each method MUST carry a real arXiv ID/URL; an ingestion note without verifiable citations is treated as fabrication.
- flagged_for_v381: `verifier_guided_adaptive_sudoku_search_before_training`
  - principle: Closes the discover->ingest->plan loop: names the strongest method for the next planner.

**Fresh-pass provenance**

Read the TRM and verifier-guided-training track in `research-studying.md` and
`research-references.md`, including the Exp 4102 `.379` ingestion that flagged
V-STaR for `.380`, the `.351` recursive-refiner notes, and the `.380` Exp 4108
and Exp 4109 result artifacts. Ran the required helpers:

- `.venv/bin/python scripts/sweep_clusters.py 0 --max-results 8`
- `.venv/bin/python scripts/sweep_clusters.py 1 --max-results 8`
- `.venv/bin/python scripts/sweep_semscholar.py "TRM Tiny Recursive Models verifier Sudoku baseline test-time adaptation" --limit 8`
- `.venv/bin/python scripts/sweep_semscholar.py "verifier guided training V-STaR STaR ReST recursive reasoning verifier" --limit 8`
- `.venv/bin/python scripts/sweep_semscholar.py "Tiny Recursive Model TRM verifier guided self training process reward" --limit 8`

Semantic Scholar returned zero IDs for the V-STaR/STaR/ReST query and HTTP 429
for two TRM-focused queries. The arXiv cluster helpers emitted reliable
verifier/energy query URLs. Low-concurrency WebSearch/WebFetch then verified
the primary arXiv pages for `arXiv:2510.04871`, `arXiv:2511.02886`,
`arXiv:2402.06457`, `arXiv:2203.14465`, `arXiv:2308.08998`,
`arXiv:2602.01070`, `arXiv:2601.17223`, and `arXiv:2605.10325`. The
`/deep-research` loop was not invoked.

## Current .380 baseline-plus-verifier anchor

The `.380` headline should stay honest. Exp 4108 confirmed the native
nano-trm Sudoku Extreme trainer can produce and reload a checkpoint, but the
measured validation exact accuracy was 0.0232 and `matches_published_087=false`,
so it is a partial baseline rather than a reproduced published number. Exp 4109
then grafted the executable Sudoku verifier over that checkpoint's candidate
pools and found an honest null: verifier selection tied TRM vote with
`rerank_lift_vs_vote.delta=0.0`, and the bounded A-vs-cold comparison also
reported `delta=0.0`.

That changes the next SOTA question. Post-hoc verifier reranking is not enough
on the current checkpoint. The strongest follow-up should move the verifier
earlier in the loop, where it can shape candidate expansion and data admission
before any expensive full fine-tuning.

## TRM Sudoku baseline reproduction

**Method:** TRM, `arXiv:2510.04871`
(https://arxiv.org/abs/2510.04871), is the load-bearing substrate because it
reports a tiny recursive model that beats HRM-style baselines on Sudoku, maze,
and ARC-style puzzles with a 7M-parameter recursive network.

**Implementation over nano-trm + Carnot-verifier stack:** Treat reproduction
as a gate, not as a background detail. Re-run the native nano-trm Sudoku
Extreme baseline with a clean progress callback, stable dataset checksum, and
checkpoint reload proof. Only after the baseline approaches the published
target should the Carnot verifier graft be allowed to claim lift over vote.

**Pitfalls / where it fails:** Exp 4108 already showed the failure mode: a
checkpoint can exist and the trainer mechanism can be real while the accuracy
is far below the published target. A verifier experiment on that checkpoint can
still be useful as a mechanism probe, but it cannot support a reproduction
claim or a strong negative about verifier value on a faithful TRM.

## TTA-TRM full fine-tuning control

**Method:** Test-time Adaptation of Tiny Recursive Models,
`arXiv:2511.02886` (https://arxiv.org/abs/2511.02886), is the adaptation
control. It argues that bounded full fine-tuning can matter more than LoRA or
task-embedding updates for a tiny recursive model.

**Implementation over nano-trm + Carnot-verifier stack:** Keep three arms
separate: full fine-tuning without verifier labels, full fine-tuning admitted
by the executable Sudoku verifier, and post-hoc verifier reranking without
training. The comparison must report compute, optimizer steps, checkpoint
source, and data split so adaptation gain is not mislabeled as verifier gain.

**Pitfalls / where it fails:** TTA-TRM can win by spending adaptation compute or
memorizing public task structure. If the experiment does not isolate the
no-verifier full-fine-tune arm, every gain will be ambiguous.

## V-STaR accepted/rejected trace selector

**Method:** V-STaR, `arXiv:2402.06457`
(https://arxiv.org/abs/2402.06457), trains a verifier from both correct and
incorrect self-generated solutions rather than throwing away failures, then
uses that verifier to select among candidates.

**Implementation over nano-trm + Carnot-verifier stack:** Reuse the Exp 4109
candidate pools, but keep every sampled Sudoku completion: exact-valid,
near-valid, row/column/box-invalid, duplicate-vote, and timeout. Convert
within-puzzle pairs into selector data where the executable Sudoku verifier and
final exact-valid label agree. Use the selector first as a cheap reranker
against vote before letting it gate a second RFT corpus.

**Pitfalls / where it fails:** If the current TRM emits many near-duplicate
invalid completions, the selector learns shallow token regularities instead of
semantic validity. V-STaR is also downstream of verifier coverage; it cannot
invent correct completions absent from the sampled pool.

## STaR / ReST generate-filter-improve loop

**Method:** STaR, `arXiv:2203.14465`
(https://arxiv.org/abs/2203.14465), gives the minimal generate-filter-finetune
loop for self-generated reasoning traces. ReST, `arXiv:2308.08998`
(https://arxiv.org/abs/2308.08998), adds a reusable offline
generate/filter/improve cadence.

**Implementation over nano-trm + Carnot-verifier stack:** Treat Sudoku
candidate completions as rationale traces. Generate K candidates per puzzle,
filter exact-valid completions with the Carnot Sudoku verifier, fine-tune on
unique positives, regenerate from the updated checkpoint, and keep rejected
rows available for V-STaR-style selector training rather than discarding them.

**Pitfalls / where it fails:** STaR/ReST need support. If the TRM rarely samples
valid completions from the partial Exp 4108 checkpoint, filtering leaves too few
positives and the improve step becomes either unstable or a memorization pass.

## Verifier-guided adaptive Sudoku search

**Method:** Adaptive test-time compute allocation, `arXiv:2602.01070`
(https://arxiv.org/abs/2602.01070), is the search-side candidate: spend more
compute where verification says it can change the answer, not after a fixed
candidate pool has already been sampled. Verifiable process rewards,
`arXiv:2601.17223` (https://arxiv.org/abs/2601.17223) and `arXiv:2605.10325`
(https://arxiv.org/abs/2605.10325), give the adjacent dense-feedback pattern
when intermediate states are objectively checkable.

**Implementation over nano-trm + Carnot-verifier stack:** Move Sudoku row,
column, and box checks into candidate expansion. Instead of sampling K complete
boards and reranking, allocate extra recursive steps, resampling, or branch
budget to partial boards whose verifier state is recoverable and prune branches
that violate exact constraints irreparably. Keep final exact validity as the
only acceptance authority, and measure against the fixed-K vote and post-hoc
verifier rerank from Exp 4109.

**Pitfalls / where it fails:** Local validity is not final correctness. A board
can satisfy many local constraints and still be unrecoverable from the puzzle
givens. The verifier-guided arm must therefore report final exact accuracy,
oracle support, and prune-error rate, not just average verifier score.

## Flagged for the .381 roadmap

`verifier_guided_adaptive_sudoku_search_before_training` is the strongest
single `.381` candidate. Exp 4109 already tested post-hoc verifier reranking
and found no lift. Before spending on another full fine-tune or a V-STaR
selector, the next planner should test whether putting the executable Sudoku
verifier inside candidate expansion creates support that post-hoc reranking did
not have. If it does not beat fixed-K vote on pass@1 or oracle support, the
training routes should remain blocked.
"""

STUDYING_SECTION = """## 2026-06-12 Exp 4111 - .380 TRM verifier-training SOTA ingestion ingested

**Status:** INGESTED into `docs/research-notes/sota-ingestion-trm-verifier-training-2026-06-12.md`.

**Filtered track:** TRM baseline reproduction plus verifier-guided training and
search over the `nano-trm` Sudoku substrate after Exp 4108 produced an honest
partial baseline and Exp 4109 produced an honest post-hoc verifier null.

**Seed and fresh-pass candidates marked ingested:**
- TRM, arXiv:2510.04871 - mapped as the faithful Sudoku Extreme baseline
  reproduction gate before any verifier-lift claim.
- TTA-TRM, arXiv:2511.02886 - mapped as the full-fine-tuning adaptation control
  that must be isolated from verifier-admission effects.
- V-STaR, arXiv:2402.06457 - mapped as accepted/rejected Sudoku trace selector
  training once candidate diversity exists.
- STaR, arXiv:2203.14465, and ReST, arXiv:2308.08998 - mapped as the cached
  generate-filter-improve cadence, with rejected rows retained for selector data.
- Adaptive verifier-guided search, arXiv:2602.01070, with VPRM/VPR support from
  arXiv:2601.17223 and arXiv:2605.10325 - mapped as the next in-loop verifier
  use because Exp 4109 post-hoc reranking tied vote.

Flagged for .381: `verifier_guided_adaptive_sudoku_search_before_training`.

**Bottom line for the .381 roadmap:** move the executable Sudoku verifier into
candidate expansion before spending on another full fine-tune. Require pass@1
or oracle-support lift over fixed-K vote and Exp 4109 post-hoc verifier rerank;
otherwise keep V-STaR and RFT routes blocked.
"""

STUDYING_MARKER = (
    "## 2026-06-12 Exp 4111 - .380 TRM verifier-training SOTA ingestion ingested"
)


def build_artifact(
    *,
    methods_mapped: Sequence[Mapping[str, str]],
    flagged_for_v381: str,
    honest_verdict: str = DEFAULT_HONEST_VERDICT,
) -> dict[str, object]:
    """Build and validate the REQ-REPORT-4111 mapping artifact."""

    artifact: dict[str, object] = {
        "honest_verdict": honest_verdict,
        "methods_mapped": [dict(method) for method in methods_mapped],
        "flagged_for_v381": flagged_for_v381,
        "field_principles": dict(FIELD_PRINCIPLES),
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

    field_principles = artifact["field_principles"]
    if field_principles != FIELD_PRINCIPLES:
        raise ValueError("field_principles must match the required annotations")

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

    flagged = artifact["flagged_for_v381"]
    if not isinstance(flagged, str) or not flagged.strip():
        raise ValueError("flagged_for_v381 must be a non-empty string")


def validate_markdown_note(markdown: str) -> None:
    """Check that the paired note maps papers to TRM work and closes planning."""

    required_phrases = (
        "Current .380 baseline-plus-verifier anchor",
        "TRM Sudoku baseline reproduction",
        "TTA-TRM full fine-tuning control",
        "V-STaR accepted/rejected trace selector",
        "STaR / ReST generate-filter-improve loop",
        "Verifier-guided adaptive Sudoku search",
        "Implementation over nano-trm + Carnot-verifier stack",
        "Pitfalls / where it fails",
        "Flagged for the .381 roadmap",
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
        flagged_for_v381=DEFAULT_FLAGGED_FOR_V381,
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
