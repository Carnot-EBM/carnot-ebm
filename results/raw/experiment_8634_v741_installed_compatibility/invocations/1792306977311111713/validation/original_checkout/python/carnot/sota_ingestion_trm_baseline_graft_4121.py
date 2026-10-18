"""Schema helpers for the Exp 4121 TRM baseline-graft SOTA ingestion.

Spec refs: REQ-REPORT-4121, SCENARIO-REPORT-4121.

This module produces a research-planning artifact, not a model score. The
current TRM line has an important trap: a resumed checkpoint can be operational
without reproducing the published Sudoku baseline, and a verifier can be real
without adding value when it only reranks a weak fixed candidate pool. The JSON
and markdown validators keep that distinction explicit so the next planner gets
cite-backed implementation choices instead of an ungrounded headline.
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
        "flagged_for_v382",
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
DEFAULT_HONEST_VERDICT = "complete: sota_ingestion_trm_baseline_graft_mapped"
DEFAULT_FLAGGED_FOR_V382 = "verifier_guided_adaptive_candidate_expansion_over_resumed_trm"

FIELD_PRINCIPLES = {
    "honest_verdict": "Terminal-prefixed. Records ingestion completed with verifiable citations.",
    "methods_mapped": (
        "Each method MUST carry a real arXiv ID/URL; an ingestion note without "
        "verifiable citations is treated as fabrication."
    ),
    "flagged_for_v382": (
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
        "name": "TRM resumable Sudoku baseline gate",
        "arxiv_id": "2510.04871",
        "url": "https://arxiv.org/abs/2510.04871",
        "implementation_over_stack": (
            "Resume the nano-trm Sudoku Extreme checkpoint until the baseline "
            "reproduction is trustworthy before treating verifier lift as meaningful."
        ),
        "failure_mode": (
            "A checkpoint can reload and still remain far below the published TRM "
            "target, making any verifier-graft conclusion underpowered."
        ),
    },
    {
        "name": "TTA-TRM full-fine-tune control",
        "arxiv_id": "2511.02886",
        "url": "https://arxiv.org/abs/2511.02886",
        "implementation_over_stack": (
            "Keep a no-verifier full-fine-tune arm beside verifier-admitted training "
            "so adaptation compute is not confused with verifier value."
        ),
        "failure_mode": (
            "Full fine-tuning can win by public-task adaptation or leakage unless "
            "checkpoint source, split, and optimizer budget are isolated."
        ),
    },
    {
        "name": "Verifier-guided adaptive candidate expansion",
        "arxiv_id": "2602.01070",
        "url": "https://arxiv.org/abs/2602.01070",
        "implementation_over_stack": (
            "Move exact Sudoku checks into candidate expansion so resumed TRM compute "
            "is spent on recoverable partial boards before post-hoc reranking."
        ),
        "failure_mode": (
            "Local verifier scores can prefer near-valid dead ends, so final exact "
            "validity and prune-error rate must remain authoritative."
        ),
    },
    {
        "name": "V-STaR accepted/rejected Sudoku selector",
        "arxiv_id": "2402.06457",
        "url": "https://arxiv.org/abs/2402.06457",
        "implementation_over_stack": (
            "Train a selector from exact-valid and verifier-rejected Sudoku traces "
            "sampled from the same resumed checkpoint."
        ),
        "failure_mode": (
            "Near-duplicate invalid boards can teach shallow artifacts unless the "
            "pool has real within-puzzle diversity."
        ),
    },
    {
        "name": "ReST resumable generate-filter-improve curriculum",
        "arxiv_id": "2308.08998",
        "url": "https://arxiv.org/abs/2308.08998",
        "implementation_over_stack": (
            "Cache generated Sudoku batches, filter them with the Carnot verifier, "
            "resume improvement from unique positives, and retain rejects for selectors."
        ),
        "failure_mode": (
            "If the baseline checkpoint rarely samples valid completions, the cached "
            "curriculum collapses into memorization or too few positives."
        ),
    },
]

NOTE_MARKDOWN = """# SOTA ingestion 2026-06-13: TRM baseline graft with resumable verifier discipline

**Artifact fields**
- honest_verdict: `complete: sota_ingestion_trm_baseline_graft_mapped`
  - principle: Terminal-prefixed. Records ingestion completed with verifiable citations.
- methods_mapped:
  - {name: `TRM resumable Sudoku baseline gate`, arxiv_id: `2510.04871`, url: `https://arxiv.org/abs/2510.04871`}
  - {name: `TTA-TRM full-fine-tune control`, arxiv_id: `2511.02886`, url: `https://arxiv.org/abs/2511.02886`}
  - {name: `Verifier-guided adaptive candidate expansion`, arxiv_id: `2602.01070`, url: `https://arxiv.org/abs/2602.01070`}
  - {name: `V-STaR accepted/rejected Sudoku selector`, arxiv_id: `2402.06457`, url: `https://arxiv.org/abs/2402.06457`}
  - {name: `ReST resumable generate-filter-improve curriculum`, arxiv_id: `2308.08998`, url: `https://arxiv.org/abs/2308.08998`}
  - principle: Each method MUST carry a real arXiv ID/URL; an ingestion note without verifiable citations is treated as fabrication.
- flagged_for_v382: `verifier_guided_adaptive_candidate_expansion_over_resumed_trm`
  - principle: Closes the discover->ingest->plan loop: names the strongest method for the next planner.

**Fresh-pass provenance**

Read the local TRM, verifier-guided-training, and long-horizon-training track in
`research-studying.md` and `research-references.md`, including the Exp 4102
`.379` V-STaR flag, the Exp 4111 `.380` in-loop verifier-guided search flag,
the `.351` recursive-refiner notes, and the long-horizon VPR notes. Ran the
required helpers:

- `.venv/bin/python scripts/sweep_clusters.py 0 --max-results 8`
- `.venv/bin/python scripts/sweep_clusters.py 1 --max-results 8`
- `.venv/bin/python scripts/sweep_clusters.py 3 --max-results 8`
- `.venv/bin/python scripts/sweep_semscholar.py "TRM Tiny Recursive Models verifier Sudoku baseline test-time adaptation" --limit 8`
- `.venv/bin/python scripts/sweep_semscholar.py "verifier guided training V-STaR STaR ReST recursive reasoning verifier" --limit 8`
- `.venv/bin/python scripts/sweep_semscholar.py "long horizon training verifier process reward recursive reasoning TRM" --limit 8`

Semantic Scholar returned `arXiv:2603.15641` for the TRM baseline/adaptation
query and HTTP 429 for the two verifier-training queries; it did not displace
the operator-specified anchors. The arXiv cluster helpers emitted reliable
verifier, energy, and active-inference query URLs. Low-concurrency
WebSearch/WebFetch verified the primary arXiv pages for `arXiv:2510.04871`,
`arXiv:2511.02886`, `arXiv:2402.06457`, `arXiv:2203.14465`,
`arXiv:2308.08998`, `arXiv:2602.01070`, `arXiv:2601.17223`, and
`arXiv:2605.10325`. The `/deep-research` loop was not invoked.

## Current .381 resumable baseline-graft anchor

The `.381` headline should stay narrower than a general verifier-training
claim. Exp 4108 proved the native nano-trm Sudoku Extreme path can train,
checkpoint, and reload, but its measured validation exact accuracy was 0.0232
with `matches_published_087=false`. Exp 4109 then grafted the executable Sudoku
verifier onto candidate pools from that checkpoint and found no post-hoc lift
over vote. Exp 4111 therefore flagged moving verification into candidate
expansion before spending on another training loop.

For `.382`, the planner needs a method that preserves all three facts: the TRM
baseline must be resumable and reproduced before headline claims, the verifier
must act before a fixed weak pool is exhausted, and any training curriculum must
record rejected traces instead of throwing them away.

## TRM resumable Sudoku baseline gate

**Method:** TRM, `arXiv:2510.04871`
(https://arxiv.org/abs/2510.04871), is the substrate to reproduce. The paper
reports a tiny recursive model with a single small network and strong puzzle
generalization, including Sudoku Extreme and ARC-style tasks.

**Implementation over nano-trm + Carnot-verifier stack:** Treat the resumed
baseline as a gate. Continue from the saved nano-trm checkpoint only with a
stable dataset checksum, optimizer-state receipt, checkpoint reload proof, and
held-out Sudoku Extreme validation trace. The Carnot verifier graft should only
claim value after the TRM baseline approaches the published Sudoku target or is
explicitly labeled as a partial-baseline mechanism probe.

**Pitfalls / where it fails:** A resumable checkpoint is not a reproduced TRM.
The existing partial baseline can validate code paths while still producing a
candidate pool too weak for post-hoc verifier selection to matter.

## TTA-TRM full-fine-tune control

**Method:** Test-time Adaptation of Tiny Recursive Models,
`arXiv:2511.02886` (https://arxiv.org/abs/2511.02886), is the adaptation
control because it reports that full fine-tuning, not LoRA or task embeddings
alone, drives the tiny recursive model's competition-budget adaptation.

**Implementation over nano-trm + Carnot-verifier stack:** Keep three arms:
resumed baseline without extra training, no-verifier full fine-tuning, and
verifier-admitted full fine-tuning. Log optimizer steps, task splits, wall time,
checkpoint source, and verifier admission counts so adaptation gain is not
misreported as verifier gain.

**Pitfalls / where it fails:** Full fine-tuning can memorize public-task
structure or simply spend more compute. Without the no-verifier control, every
improvement is ambiguous.

## Verifier-guided adaptive candidate expansion

**Method:** Adaptive test-time compute allocation,
`arXiv:2602.01070` (https://arxiv.org/abs/2602.01070), is the strongest
follow-on because it uses verification during generation and expansion rather
than only for final reranking. Verifiable process reward work, `arXiv:2601.17223`
(https://arxiv.org/abs/2601.17223) and `arXiv:2605.10325`
(https://arxiv.org/abs/2605.10325), supplies the long-horizon dense-feedback
pattern when intermediate steps are objectively checkable.

**Implementation over nano-trm + Carnot-verifier stack:** Move row, column,
box, and given-cell checks into the recursive candidate expansion loop. Spend
extra samples or recursive steps on partial boards that remain recoverable,
prune irreparable branches early, and compare against fixed-K vote plus Exp
4109 post-hoc verifier reranking. Report pass@1, oracle support, final exact
validity, verifier-call count, and prune-error rate.

**Pitfalls / where it fails:** Sudoku has local constraints that can look good
while the board is globally unrecoverable. Final exact validity must remain the
acceptance authority, and the experiment must measure whether pruning removes
any candidate that could have become valid.

## V-STaR accepted/rejected Sudoku selector

**Method:** V-STaR, `arXiv:2402.06457`
(https://arxiv.org/abs/2402.06457), trains a verifier from both correct and
incorrect self-generated solutions and uses it to choose among candidates.

**Implementation over nano-trm + Carnot-verifier stack:** Keep all completions
from the resumed TRM: exact-valid, row/column/box invalid, duplicate vote,
timeout, and parse fail. Build within-puzzle preference pairs only where the
executable Sudoku verifier and final exact label agree. Use the selector first
as a reranker before allowing it to gate a second RFT corpus.

**Pitfalls / where it fails:** V-STaR needs diverse failures. If the resumed
checkpoint emits many near-duplicate wrong boards, the selector learns surface
regularities and still cannot create correct candidates absent from the pool.

## ReST resumable generate-filter-improve curriculum

**Method:** ReST, `arXiv:2308.08998`
(https://arxiv.org/abs/2308.08998), gives the reusable offline
generate-filter-improve cadence. STaR, `arXiv:2203.14465`
(https://arxiv.org/abs/2203.14465), supplies the older rationale
self-training loop that keeps only generated reasoning that reaches correct
answers.

**Implementation over nano-trm + Carnot-verifier stack:** Cache Sudoku
candidate batches, filter exact-valid completions with Carnot, train on unique
positives, and then resume generation from the updated checkpoint. Retain
rejected rows for the V-STaR selector rather than discarding them.

**Pitfalls / where it fails:** The loop only amplifies support already present
in the generator. If resumed TRM rarely samples valid boards, the curriculum
creates too few positives and can collapse into memorization.

## Flagged for the .382 roadmap

`verifier_guided_adaptive_candidate_expansion_over_resumed_trm` is the strongest
single `.382` candidate. It directly addresses the Exp 4109 null by moving the
Sudoku verifier before fixed-pool reranking, while preserving the Exp 4108
baseline-reproduction gate. The next planner should require pass@1 or oracle
support lift over fixed-K vote and post-hoc verifier rerank. If it fails,
selector/RFT work should stay blocked.
"""

STUDYING_SECTION = """## 2026-06-13 Exp 4121 - .381 TRM baseline-graft SOTA ingestion ingested

**Status:** INGESTED into `docs/research-notes/sota-ingestion-trm-baseline-graft-2026-06-13.md`.

**Filtered track:** resumable TRM Sudoku baseline reproduction plus Carnot
verifier graft, after Exp 4108 produced a checkpointed but partial baseline,
Exp 4109 found no post-hoc verifier lift over vote, and Exp 4111 flagged
in-loop verifier-guided search as the next candidate.

**Seed and fresh-pass candidates marked ingested:**
- TRM, arXiv:2510.04871 - mapped as the resumed Sudoku Extreme baseline gate
  before any verifier-lift claim.
- TTA-TRM, arXiv:2511.02886 - mapped as the full-fine-tuning adaptation control
  that must be isolated from verifier-admission effects.
- Adaptive verifier-guided candidate expansion, arXiv:2602.01070, with VPRM/VPR
  support from arXiv:2601.17223 and arXiv:2605.10325 - mapped as the strongest
  .382 follow-on because post-hoc verifier reranking already tied vote.
- V-STaR, arXiv:2402.06457 - mapped as accepted/rejected Sudoku trace selector
  training once candidate diversity and oracle support exist.
- ReST, arXiv:2308.08998, and STaR, arXiv:2203.14465 - mapped as the resumable
  generate-filter-improve curriculum, with rejected rows retained for selector
  data.

Flagged for .382: `verifier_guided_adaptive_candidate_expansion_over_resumed_trm`.

**Bottom line for the .382 roadmap:** put the executable Sudoku verifier inside
candidate expansion over the resumed TRM checkpoint before spending on selector
or RFT work. Require pass@1 or oracle-support lift over fixed-K vote and Exp
4109 post-hoc verifier rerank; otherwise selector/RFT work should stay blocked.
"""

STUDYING_MARKER = (
    "## 2026-06-13 Exp 4121 - .381 TRM baseline-graft SOTA ingestion ingested"
)


def build_artifact(
    *,
    methods_mapped: Sequence[Mapping[str, str]],
    flagged_for_v382: str,
    honest_verdict: str = DEFAULT_HONEST_VERDICT,
) -> dict[str, object]:
    """Build and validate the REQ-REPORT-4121 mapping artifact."""

    artifact: dict[str, object] = {
        "honest_verdict": honest_verdict,
        "methods_mapped": [dict(method) for method in methods_mapped],
        "flagged_for_v382": flagged_for_v382,
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

    flagged = artifact["flagged_for_v382"]
    if not isinstance(flagged, str) or not flagged.strip():
        raise ValueError("flagged_for_v382 must be a non-empty string")


def validate_markdown_note(markdown: str) -> None:
    """Check that the paired note maps papers to TRM work and closes planning."""

    required_phrases = (
        "Current .381 resumable baseline-graft anchor",
        "TRM resumable Sudoku baseline gate",
        "TTA-TRM full-fine-tune control",
        "Verifier-guided adaptive candidate expansion",
        "V-STaR accepted/rejected Sudoku selector",
        "ReST resumable generate-filter-improve curriculum",
        "Implementation over nano-trm + Carnot-verifier stack",
        "Pitfalls / where it fails",
        "Flagged for the .382 roadmap",
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
        flagged_for_v382=DEFAULT_FLAGGED_FOR_V382,
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
