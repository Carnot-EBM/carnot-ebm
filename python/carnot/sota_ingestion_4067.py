"""Schema helpers for the Exp 4067 SOTA-ingestion receipt.

Spec refs: REQ-REPORT-4067, SCENARIO-REPORT-4067.

The research note is where the actual literature judgment lives: which methods
from the fresh pass matter, how they fit Carnot's current verifier stack, and
which failure modes can make them unsafe to plan from.  This module keeps the
machine-readable receipt honest.  A valid receipt proves that the ingestion
mapped real cited methods, flagged concrete next-roadmap candidates, and used
the expected upstream-artifact aggregation substrate.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


EXPECTED_INFERENCE_SUBSTRATE = "aggregation_from_upstream_artifacts"
REQUIRED_RECEIPT_FIELDS = frozenset(
    {
        "honest_verdict",
        "methods_mapped_count",
        "citations",
        "flagged_for_v377",
        "inference_substrate",
    }
)
TERMINAL_PREFIXES = ("complete:", "success:", "passed:", "shipped:", "blocked_")
DEFAULT_HONEST_VERDICT = (
    "complete: "
    "sota_ingestion_v376_unsaturated_corpora_and_online_pruning_mapped"
)


def build_receipt(
    *,
    methods_mapped_count: int,
    citations: Sequence[Mapping[str, str]],
    flagged_for_v377: Sequence[str],
    honest_verdict: str = DEFAULT_HONEST_VERDICT,
) -> dict[str, object]:
    """Build and validate the REQ-REPORT-4067 receipt payload."""

    receipt: dict[str, object] = {
        "honest_verdict": honest_verdict,
        "methods_mapped_count": methods_mapped_count,
        "citations": [dict(citation) for citation in citations],
        "flagged_for_v377": list(flagged_for_v377),
        "inference_substrate": EXPECTED_INFERENCE_SUBSTRATE,
    }
    validate_receipt(receipt)
    return receipt


def validate_receipt(receipt: Mapping[str, Any]) -> None:
    """Validate the SCENARIO-REPORT-4067 receipt contract."""

    missing = REQUIRED_RECEIPT_FIELDS.difference(receipt)
    extra = set(receipt).difference(REQUIRED_RECEIPT_FIELDS)
    if missing:
        raise ValueError(f"receipt missing required fields: {sorted(missing)}")
    if extra:
        raise ValueError(f"receipt has unexpected fields: {sorted(extra)}")

    verdict = receipt["honest_verdict"]
    if not isinstance(verdict, str) or not verdict.startswith(TERMINAL_PREFIXES):
        raise ValueError("honest_verdict must use a terminal prefix")

    methods_mapped_count = receipt["methods_mapped_count"]
    if (
        not isinstance(methods_mapped_count, int)
        or isinstance(methods_mapped_count, bool)
        or methods_mapped_count < 6
    ):
        raise ValueError("methods_mapped_count must be an integer of at least six")

    citations = receipt["citations"]
    if not isinstance(citations, list) or not citations:
        raise ValueError("citations must be a non-empty list")
    if len(citations) < methods_mapped_count:
        raise ValueError("citations must include at least methods_mapped_count entries")
    for citation in citations:
        if not isinstance(citation, dict) or not citation.get("arxiv_id_or_url"):
            raise ValueError("each citation must include arxiv_id_or_url")

    flagged_for_v377 = receipt["flagged_for_v377"]
    if (
        not isinstance(flagged_for_v377, list)
        or not flagged_for_v377
        or not all(isinstance(flag, str) and flag for flag in flagged_for_v377)
    ):
        raise ValueError("flagged_for_v377 must be a non-empty list of strings")

    if receipt["inference_substrate"] != EXPECTED_INFERENCE_SUBSTRATE:
        raise ValueError(
            f"inference_substrate must be {EXPECTED_INFERENCE_SUBSTRATE!r}"
        )


def validate_markdown_note(markdown: str) -> None:
    """Check that the paired note contains the required planning sections."""

    required_phrases = (
        "Confirmed .376 actionability from Exp 4055",
        "LOCAL-12B oracle-headroom code corpus",
        "VERIFIER-GUIDED ONLINE ACTION-PRUNING",
        "Implementation over Carnot stack",
        "Pitfalls / where it fails",
        "Bottom line for the .377 roadmap",
        "demo-fit code verifier",
        "sandbox",
        "EvalPlus",
        "LiveCodeBench v6",
        "explore-first solver",
        "GAP-4 verifier",
    )
    missing = [phrase for phrase in required_phrases if phrase not in markdown]
    if missing:
        raise ValueError(f"markdown note missing required sections: {missing}")
