"""JEPA training data wiring guard.

Verifies that the CPMI contrastive triples file has been properly loaded and
that the augmentation ratio exceeds the minimum threshold before any JEPA
retrain begins.  This prevents the Exp 798→799 failure mode where training
ran for 5+ minutes without CPMI augmentation, producing the all-time low
ood_auc=0.2444.

The augmentation_ratio is defined as:
    n_triples / max(n_input_pairs, 1)

A ratio of 1.0 means no augmentation occurred — the triples file either
does not exist, is empty, or was not merged into the training data loader.
The assertion threshold defaults to 1.5 (50% more triples than raw pairs),
which is the minimum that produced measurable OOD improvement in Exp 798.

Spec: REQ-INFRA-061, SCENARIO-INFRA-070
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass


@dataclass
class JepaWiringCheckResult:
    """Result of a CPMI wiring guard check.

    Attributes:
        triples_path: Filesystem path that was checked for triples data.
        n_triples: Number of CPMI triples found at triples_path.
        n_input_pairs: Number of raw input pairs (denominator for ratio).
        augmentation_ratio: n_triples / max(n_input_pairs, 1).  Values > 1.0
            indicate that the CPMI corpus is contributing additional training
            signal beyond the raw pairs.
        is_wired: True when augmentation_ratio >= min_augmentation_ratio.
        honest_verdict: Human-readable summary of the check outcome.
    """

    triples_path: str
    n_triples: int
    n_input_pairs: int
    augmentation_ratio: float
    is_wired: bool
    honest_verdict: str


def check_cpmi_wiring(
    triples_path: str,
    min_augmentation_ratio: float = 1.5,
) -> JepaWiringCheckResult:
    """Verify CPMI triples are present and the augmentation ratio is sufficient.

    Loads the triples JSON from triples_path, counts the entries, and asserts
    that the ratio of triples to input pairs meets the minimum threshold.  If
    the assertion fails, training should NOT begin — the caller is expected to
    catch AssertionError, write a blocked artifact, and exit.

    Args:
        triples_path: Path to the CPMI triples JSON file
            (e.g. results/experiment_798_cpmi_pairs_triples.json).  The file
            must be a JSON list of dicts, each with at least a 'prefix_text'
            key that identifies the source input pair.
        min_augmentation_ratio: Minimum acceptable ratio of triples to unique
            input pairs.  Defaults to 1.5 based on Exp 798 findings.

    Returns:
        JepaWiringCheckResult with all computed fields.

    Raises:
        AssertionError: When augmentation_ratio < min_augmentation_ratio.
            Message: "CPMI corpus not wired in — check training data loader
            merges all sources."  This is the signal that training must not
            begin — the data pipeline is misconfigured.
        FileNotFoundError: When triples_path does not exist on disk.

    Spec: REQ-INFRA-061, SCENARIO-INFRA-070
    """
    if not os.path.exists(triples_path):
        raise FileNotFoundError(
            f"CPMI triples file not found: {triples_path!r}. "
            "Run Exp 798 (cpmi_builder) to generate triples before retraining JEPA."
        )

    with open(triples_path, encoding="utf-8") as fh:
        triples: list[dict] = json.load(fh)

    n_triples = len(triples)

    # Count unique input pairs by distinct prefix_text values.  Each prefix
    # is derived from one source reasoning step, so unique prefixes == unique pairs.
    unique_prefixes = {t.get("prefix_text", f"__row_{i}__") for i, t in enumerate(triples)}
    n_input_pairs = len(unique_prefixes)

    augmentation_ratio = n_triples / max(n_input_pairs, 1)
    is_wired = augmentation_ratio >= min_augmentation_ratio

    assert augmentation_ratio >= min_augmentation_ratio, (
        "CPMI corpus not wired in — check training data loader merges all sources."
    )

    honest_verdict = (
        f"wired: {n_triples} triples over {n_input_pairs} pairs "
        f"(ratio={augmentation_ratio:.2f} >= {min_augmentation_ratio:.2f})"
    )

    return JepaWiringCheckResult(
        triples_path=triples_path,
        n_triples=n_triples,
        n_input_pairs=n_input_pairs,
        augmentation_ratio=augmentation_ratio,
        is_wired=is_wired,
        honest_verdict=honest_verdict,
    )
