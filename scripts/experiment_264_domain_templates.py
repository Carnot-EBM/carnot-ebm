"""Experiment 264: Domain constraint template mining.

Mines regex-based constraint templates from violation/clean corpus rows.
Templates are filtered by corpus precision (>= min_precision) and minimum
match count (>= min_matches). Used to seed FormalClaimVerifier routing rules.

Spec: REQ-CONSTRAINT-264-A, REQ-CONSTRAINT-264-B, REQ-CONSTRAINT-264-C
"""

from __future__ import annotations

import re
from typing import Any

# ---------------------------------------------------------------------------
# Domain patterns: (domain, regex, route) triples
# REQ-CONSTRAINT-264-C: DOMAIN_PATTERNS is a fixed deterministic list
# ---------------------------------------------------------------------------

DOMAIN_PATTERNS: list[tuple[str, str, str]] = [
    # Arithmetic patterns
    ("arithmetic", r"\b\d+\s*\+\s*\d+", "arithmetic"),
    ("arithmetic", r"\b\d+\s*-\s*\d+", "arithmetic"),
    ("arithmetic", r"\b\d+\s*\*\s*\d+", "arithmetic"),
    ("arithmetic", r"\b\d+\s*/\s*\d+", "arithmetic"),
    ("arithmetic", r"\b\d+\s*=\s*\d+", "arithmetic"),
    # Cardinality patterns
    ("cardinality", r"\b(total|each|sum|count|number of)\b", "cardinality"),
    ("cardinality", r"\b\d+\s+(items?|elements?|members?)\b", "cardinality"),
    # Set membership patterns
    ("set_membership", r"\b(including|such as|for example|e\.g\.)\b", "set_membership"),
    ("set_membership", r"\b(is a|are a|belongs? to|member of)\b", "set_membership"),
]


# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------


def _pattern_stats(
    regex: str,
    rows: list[dict[str, Any]],
) -> tuple[float, float, int, int]:
    """Compute precision and recall for a regex pattern over a corpus.

    REQ-CONSTRAINT-264-B: returns (precision, recall, n_positive_matches, n_negative_matches).

    Precision = n_pos_matches / (n_pos_matches + n_neg_matches)  [among matched rows]
    Recall    = n_pos_matches / total_positive_rows

    Args:
        regex: Regular expression to test against partial_response field.
        rows: List of corpus rows with 'partial_response' and 'violation_label'.

    Returns:
        Tuple of (precision, recall, n_pos_matches, n_neg_matches).
        Returns (0.0, 0.0, 0, 0) when the pattern matches no rows.
    """
    pattern = re.compile(regex, re.IGNORECASE)
    n_pos_match = 0
    n_neg_match = 0
    n_total_pos = 0

    for row in rows:
        is_violation = bool(row.get("violation_label", False))
        text = str(row.get("partial_response", ""))
        matched = bool(pattern.search(text))

        if is_violation:
            n_total_pos += 1
        if matched:
            if is_violation:
                n_pos_match += 1
            else:
                n_neg_match += 1

    n_matched = n_pos_match + n_neg_match
    if n_matched == 0:
        return 0.0, 0.0, 0, 0

    precision = n_pos_match / n_matched
    recall = (n_pos_match / n_total_pos) if n_total_pos > 0 else 0.0
    return precision, recall, n_pos_match, n_neg_match


def _model_specificity(
    regex: str,
    rows: list[dict[str, Any]],
) -> str:
    """Determine whether a pattern is specific to one model or common to both.

    REQ-CONSTRAINT-264-C: returns 'both' or a model name string.

    Args:
        regex: Regular expression to test.
        rows: Corpus rows with 'model' and 'partial_response' fields.

    Returns:
        'both' if the pattern matches rows from multiple models,
        otherwise the name of the single model it matches.
    """
    pattern = re.compile(regex, re.IGNORECASE)
    matched_models: set[str] = set()

    for row in rows:
        text = str(row.get("partial_response", ""))
        if pattern.search(text):
            model = str(row.get("model", "unknown"))
            matched_models.add(model)

    if len(matched_models) > 1:
        return "both"
    if len(matched_models) == 1:
        return next(iter(matched_models))
    return "both"


def _make_template_record(
    *,
    domain: str,
    regex: str,
    route: str,
    precision: float,
    recall: float,
    model_spec: str,
    n_pos: int,
    n_neg: int,
) -> dict[str, Any]:
    """Build a template record dict with all required fields.

    REQ-CONSTRAINT-264-A: all required fields present with correct types.

    Returns:
        Dict with domain, token_pattern_regex, associated_claim_route,
        corpus_precision, corpus_recall, model_specificity, n_positive_cases,
        n_negative_cases, experiment, run_date.
    """
    return {
        "domain": domain,
        "token_pattern_regex": regex,
        "associated_claim_route": route,
        "corpus_precision": float(precision),
        "corpus_recall": float(recall),
        "model_specificity": model_spec,
        "n_positive_cases": int(n_pos),
        "n_negative_cases": int(n_neg),
        "experiment": 264,
        "run_date": "20260413",
    }


# ---------------------------------------------------------------------------
# Template mining
# ---------------------------------------------------------------------------


def mine_templates(
    rows: list[dict[str, Any]],
    patterns: list[tuple[str, str, str]],
    *,
    min_precision: float = 0.50,
    min_matches: int = 3,
) -> list[dict[str, Any]]:
    """Mine constraint templates from corpus rows.

    REQ-CONSTRAINT-264-B: templates with corpus_precision < min_precision excluded.
    REQ-CONSTRAINT-264-B: templates matching fewer than min_matches rows excluded.

    Args:
        rows: Corpus rows from experiment 262/263.
        patterns: List of (domain, regex, route) tuples to evaluate.
        min_precision: Minimum corpus_precision threshold (default 0.50).
        min_matches: Minimum total matches required (default 3).

    Returns:
        List of template record dicts passing all filters, in deterministic order.
    """
    templates = []
    for domain, regex, route in patterns:
        precision, recall, n_pos, n_neg = _pattern_stats(regex, rows)
        n_total = n_pos + n_neg

        if n_total < min_matches:
            continue
        if precision < min_precision:
            continue

        model_spec = _model_specificity(regex, rows)
        rec = _make_template_record(
            domain=domain,
            regex=regex,
            route=route,
            precision=precision,
            recall=recall,
            model_spec=model_spec,
            n_pos=n_pos,
            n_neg=n_neg,
        )
        templates.append(rec)

    return templates


# ---------------------------------------------------------------------------
# Top-level runner
# ---------------------------------------------------------------------------


def run_mining(
    rows: list[dict[str, Any]],
    *,
    min_precision: float = 0.50,
    min_matches: int = 3,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Run the full template mining pipeline.

    REQ-CONSTRAINT-264-A / REQ-CONSTRAINT-264-C: returns templates + summary.

    Args:
        rows: Corpus rows.
        min_precision: Minimum precision threshold.
        min_matches: Minimum total match count.

    Returns:
        Tuple of (templates list, summary dict).
        Summary has keys: template_counts_by_domain, precision_stats,
        model_stats, experiment.
    """
    templates = mine_templates(rows, DOMAIN_PATTERNS, min_precision=min_precision, min_matches=min_matches)

    # Build summary
    counts_by_domain: dict[str, int] = {}
    precisions: list[float] = []
    model_counts: dict[str, int] = {}

    for rec in templates:
        d = rec["domain"]
        counts_by_domain[d] = counts_by_domain.get(d, 0) + 1
        precisions.append(rec["corpus_precision"])
        ms = rec["model_specificity"]
        model_counts[ms] = model_counts.get(ms, 0) + 1

    summary: dict[str, Any] = {
        "template_counts_by_domain": counts_by_domain,
        "precision_stats": {
            "mean": sum(precisions) / len(precisions) if precisions else 0.0,
            "min": min(precisions) if precisions else 0.0,
            "max": max(precisions) if precisions else 0.0,
        },
        "model_stats": model_counts,
        "experiment": 264,
        "n_templates": len(templates),
        "min_precision_threshold": min_precision,
        "min_matches_threshold": min_matches,
    }

    return templates, summary


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import json
    import sys

    corpus_path = sys.argv[1] if len(sys.argv) > 1 else None
    if corpus_path is None:
        print("Usage: experiment_264_domain_templates.py <corpus.jsonl>", file=sys.stderr)
        sys.exit(1)

    corpus_rows: list[dict[str, Any]] = []
    with open(corpus_path) as f:
        for line in f:
            line = line.strip()
            if line:
                corpus_rows.append(json.loads(line))

    tmpl, summ = run_mining(corpus_rows)
    print(json.dumps({"templates": tmpl, "summary": summ}, indent=2))
