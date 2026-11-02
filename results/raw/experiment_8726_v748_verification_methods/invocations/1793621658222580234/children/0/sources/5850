"""Schema helpers for the Exp 4081 verifier-as-reward SOTA-ingestion receipt.

Spec refs: REQ-REPORT-4081, SCENARIO-REPORT-4081.

The research note contains the judgment call: which post-training papers matter
for the `.377` pivot, how each would change Carnot's current RFT pipeline, and
where the method can fail.  This module validates the compact receipt and the
minimum note structure so the next roadmap pass can trust that the ingestion was
based on real primary papers rather than invented method names.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
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
DEFAULT_HONEST_VERDICT = "complete: sota_ingestion_verifier_as_reward_mapped"

VERIFIED_ARXIV_IDS = frozenset(
    {
        "2203.14465",  # STaR
        "2308.08998",  # ReST
        "2411.15124",  # Tulu 3
        "2505.14216",  # RL vs distillation
        "2507.14843",  # Invisible Leash
        "2601.17223",  # Verifiable Process Reward Models
        "2604.03128",  # Self-Distilled RLVR
        "2605.10325",  # Verifiable Process Rewards
    }
)


def build_receipt(
    *,
    methods_mapped: Sequence[Mapping[str, str]],
    strongest_for_next_roadmap: Sequence[str],
    honest_verdict: str = DEFAULT_HONEST_VERDICT,
) -> dict[str, object]:
    """Build and validate the REQ-REPORT-4081 receipt payload."""

    receipt: dict[str, object] = {
        "honest_verdict": honest_verdict,
        "methods_mapped": [dict(method) for method in methods_mapped],
        "strongest_for_next_roadmap": list(strongest_for_next_roadmap),
        "inference_substrate": EXPECTED_INFERENCE_SUBSTRATE,
    }
    validate_receipt(receipt)
    return receipt


def validate_receipt(receipt: Mapping[str, Any]) -> None:
    """Validate the exact receipt contract for SCENARIO-REPORT-4081."""

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
        raise ValueError(
            "strongest_for_next_roadmap must be a non-empty list of strings"
        )

    if receipt["inference_substrate"] != EXPECTED_INFERENCE_SUBSTRATE:
        raise ValueError(
            f"inference_substrate must be {EXPECTED_INFERENCE_SUBSTRATE!r}"
        )


def validate_markdown_note(markdown: str) -> None:
    """Check that the paired note has the required mapping and planning shape."""

    required_phrases = (
        "Verifier-certified RFT over the current RFT pipeline",
        "RLVR / Tulu 3 open post-training recipe",
        "Invisible Leash latent-vs-absent diagnostic",
        "Process-reward distillation",
        "RFT / STaR / ReST self-training",
        "Implementation over current RFT pipeline",
        "Pitfalls / where it fails",
        "Bottom line for the .378 roadmap",
    )
    missing_phrases = [phrase for phrase in required_phrases if phrase not in markdown]
    if missing_phrases:
        raise ValueError(f"markdown note missing required sections: {missing_phrases}")

    missing_ids = [
        arxiv_id
        for arxiv_id in VERIFIED_ARXIV_IDS
        if f"arXiv:{arxiv_id}" not in markdown
        and f"arxiv.org/abs/{arxiv_id}" not in markdown
    ]
    if missing_ids:
        raise ValueError(f"markdown note missing verified arxiv citations: {missing_ids}")
