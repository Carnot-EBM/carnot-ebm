#!/usr/bin/env python3
"""Experiment 262: Generate predictive calibration corpus for Tier 3 gate.

Derives a structured calibration corpus from the Exp 252 predictive-
verification corpus by slicing each partial_response at three prefix
fractions (0.25, 0.50, 0.75), extracting token features at each prefix, and
labelling each slice with a binary violation flag.

**Why this corpus exists:**
    The PredictiveVerifier gate in Exp 256 routed 100% of cases to FAST_PATH
    because its default prior weights are uncalibrated (negative bias too
    strong for the GSM8K distribution). To calibrate the gate we need
    (feature_vector, violation_label) pairs at varying prefix lengths — the
    262 corpus provides exactly that.

**Row schema (per row in JSONL output):**
    case_id                — string identifier from source corpus
    prefix_fraction        — 0.25 | 0.50 | 0.75 (how much of response used)
    token_feature_vector   — list[float] of length 9 (PredictiveFeatures order)
    n_tokens_in_prefix     — int, number of whitespace-split tokens in prefix
    token_pattern_features — dict: digit_density, operator_count, equals_count,
                             sentence_count
    violation_label        — bool, True when verifier_outcome == "violated"
    n_violations_final     — int (0 or 1, from verifier_outcome)
    provenance_exp         — str, source experiment identifier
    run_date               — "20260413"
    experiment             — 262

Writes:
    data/research/predictive_calibration_corpus_262.jsonl
    results/experiment_262_summary.json

Run:
    python scripts/experiment_262_calibration_corpus.py
    python scripts/experiment_262_calibration_corpus.py --n-cases 50

Spec: REQ-PRED-262-A, REQ-PRED-262-B, REQ-PRED-262-C,
      REQ-PRED-262-D, REQ-PRED-262-E
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

RUN_DATE: str = "20260413"
EXPERIMENT: int = 262
PREFIX_FRACTIONS: tuple[float, ...] = (0.25, 0.50, 0.75)
CORPUS_OUTPUT_NAME: str = "predictive_calibration_corpus_262.jsonl"
SUMMARY_OUTPUT_NAME: str = "experiment_262_summary.json"

# Source corpus: Exp 252 predictive verification corpus
_SOURCE_CORPUS_NAME: str = "predictive_verification_corpus_252.jsonl"

# ---------------------------------------------------------------------------
# Helpers: feature extraction (mirrors PredictiveFeatures.to_array logic)
# ---------------------------------------------------------------------------

_NUMERIC_RE = re.compile(r"^[+-]?\d+(\.\d+)?$")
_OPERATOR_RE = re.compile(r"^[+\-*/=]$")


def _token_features(text: str, domain: str | None, prior_confidence: float) -> dict[str, Any]:
    """Extract feature dict from a text prefix.

    Returns a dict with all PredictiveFeatures fields plus token_pattern_features
    (digit_density, operator_count, equals_count, sentence_count).

    Spec: REQ-PRED-262-A
    """
    tokens = text.split()
    token_count = len(tokens)
    char_count = len(text)

    if token_count > 0:
        stripped = [t.strip(".,;:!?()'\"") for t in tokens]
        numeric_count = sum(1 for t in stripped if _NUMERIC_RE.match(t))
        operator_count = sum(1 for t in stripped if _OPERATOR_RE.match(t))
        numeric_density = float(numeric_count) / token_count
        operator_density = float(operator_count) / token_count
        # digit_density: fraction of tokens containing any digit
        digit_count = sum(1 for t in stripped if any(c.isdigit() for c in t))
        digit_density = digit_count / token_count
        # equals_count: absolute count of "=" tokens
        equals_count = sum(1 for t in stripped if t == "=")
    else:
        numeric_density = 0.0
        operator_density = 0.0
        digit_density = 0.0
        operator_count = 0
        equals_count = 0

    # JSON structure detection
    json_parseable = 0.0
    n_claims = 0
    has_final_answer = 0.0
    if text.strip():
        try:
            parsed = json.loads(text)
            json_parseable = 1.0
            if isinstance(parsed, dict):
                claims = parsed.get("claims")
                if isinstance(claims, list):
                    n_claims = len(claims)
                if "final_answer" in parsed:
                    has_final_answer = 1.0
        except (json.JSONDecodeError, ValueError):
            pass

    domain_code = 0.0 if (domain is None or domain == "reasoning") else 1.0
    prior = float(max(0.0, min(1.0, prior_confidence)))

    # The 9-element feature vector in PredictiveFeatures order
    feature_vector = [
        min(token_count / 100.0, 1.0),   # token_count scaled
        min(char_count / 500.0, 1.0),    # char_count scaled
        numeric_density,
        operator_density,
        json_parseable,
        min(n_claims / 10.0, 1.0),       # n_claims scaled
        has_final_answer,
        domain_code,
        prior,
    ]

    # Sentence count: approximate by splitting on ". " and ".\n"
    sentence_count = max(1, len(re.split(r"\.\s", text)))

    token_pattern_features = {
        "digit_density": round(digit_density, 6),
        "operator_count": operator_count,
        "equals_count": equals_count,
        "sentence_count": sentence_count,
    }

    return {
        "feature_vector": [round(v, 6) for v in feature_vector],
        "n_tokens": token_count,
        "token_pattern_features": token_pattern_features,
    }


def _truncate_at_fraction(text: str, fraction: float) -> str:
    """Truncate *text* to the first ``fraction`` of its whitespace tokens.

    If the text has 0 tokens, returns empty string.  Fraction is clamped to
    (0, 1]; a fraction of 1.0 returns all tokens joined by single spaces.

    Spec: REQ-PRED-262-D
    """
    tokens = text.split()
    if not tokens:
        return ""
    fraction = max(0.01, min(1.0, fraction))
    n = max(1, round(len(tokens) * fraction))
    return " ".join(tokens[:n])


# ---------------------------------------------------------------------------
# Corpus builder
# ---------------------------------------------------------------------------


def build_corpus(
    source_rows: list[dict[str, Any]],
    *,
    n_cases: int | None = None,
    prefix_fractions: tuple[float, ...] = PREFIX_FRACTIONS,
) -> list[dict[str, Any]]:
    """Build calibration corpus rows from source corpus rows.

    Each source row that is not "abstain" produces ``len(prefix_fractions)``
    output rows — one per prefix fraction.

    Args:
        source_rows: Rows from Exp 252 corpus.
        n_cases:     Optional limit on distinct case_ids to include.
        prefix_fractions: Prefix fractions to generate per case.

    Returns:
        List of calibration corpus rows.

    Spec: REQ-PRED-262-A, REQ-PRED-262-D
    """
    # Filter to non-abstain rows with a partial_response
    usable = [
        r for r in source_rows
        if str(r.get("verifier_outcome", "")).lower() not in ("abstain", "")
        and r.get("partial_response")
    ]

    # Deduplicate by case_id, keep first occurrence per case_id
    seen_ids: set[str] = set()
    unique: list[dict[str, Any]] = []
    for row in usable:
        cid = str(row.get("case_id") or row.get("corpus_id") or "")
        if cid not in seen_ids:
            seen_ids.add(cid)
            unique.append(row)
        if n_cases is not None and len(unique) >= n_cases:
            break

    output_rows: list[dict[str, Any]] = []
    for row in unique:
        cid = str(row.get("case_id") or row.get("corpus_id") or "")
        partial = str(row.get("partial_response") or "")
        domain = str(row.get("domain") or "reasoning") or "reasoning"
        confidence = float(row.get("confidence") or 0.5)
        outcome = str(row.get("verifier_outcome") or "").lower()
        violation_label = outcome == "violated"
        n_violations_final = 1 if violation_label else 0
        provenance_exp = str(row.get("source_experiment") or row.get("experiment") or "252")
        model = str(row.get("model") or "")

        for frac in prefix_fractions:
            prefix_text = _truncate_at_fraction(partial, frac)
            feats = _token_features(prefix_text, domain, confidence)
            output_rows.append({
                "case_id": cid,
                "prefix_fraction": frac,
                "token_feature_vector": feats["feature_vector"],
                "n_tokens_in_prefix": feats["n_tokens"],
                "token_pattern_features": feats["token_pattern_features"],
                "violation_label": violation_label,
                "n_violations_final": n_violations_final,
                "provenance_exp": provenance_exp,
                "domain": domain,
                "model": model,
                "confidence": confidence,
                "partial_response": partial,
                "run_date": RUN_DATE,
                "experiment": EXPERIMENT,
            })

    return output_rows


def build_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Build summary artifact from corpus rows.

    Spec: REQ-PRED-262-B, REQ-PRED-262-E
    """
    # Count distinct case_ids
    case_ids = {r["case_id"] for r in rows}
    n_cases = len(case_ids)
    total_rows = len(rows)
    n_positive = sum(1 for r in rows if r["violation_label"])
    violation_rate = n_positive / total_rows if total_rows else 0.0

    # Feature importance by prefix fraction: mean violation_label per fraction
    pfi: dict[str, float] = {}
    for frac in PREFIX_FRACTIONS:
        subset = [r for r in rows if r["prefix_fraction"] == frac]
        pfi[str(frac)] = sum(1 for r in subset if r["violation_label"]) / len(subset) if subset else 0.0

    # Token pattern stats: mean value for positive vs negative cases
    tps: dict[str, dict[str, float]] = {}
    pos_rows = [r for r in rows if r["violation_label"]]
    neg_rows = [r for r in rows if not r["violation_label"]]
    for feature in ("digit_density", "operator_count", "equals_count", "sentence_count"):
        mean_pos = (
            sum(r["token_pattern_features"][feature] for r in pos_rows) / len(pos_rows)
            if pos_rows else 0.0
        )
        mean_neg = (
            sum(r["token_pattern_features"][feature] for r in neg_rows) / len(neg_rows)
            if neg_rows else 0.0
        )
        tps[feature] = {
            "mean_positive": round(mean_pos, 6),
            "mean_negative": round(mean_neg, 6),
        }

    return {
        "experiment": EXPERIMENT,
        "run_date": RUN_DATE,
        "n_cases": n_cases,
        "total_rows": total_rows,
        "n_positive": n_positive,
        "n_negative": total_rows - n_positive,
        "violation_rate": round(violation_rate, 6),
        "prefix_fraction_feature_importance": pfi,
        "token_pattern_stats": tps,
    }


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------


def get_repo_root() -> Path:
    """Resolve the repository root from environment or file location."""
    override = os.environ.get("CARNOT_REPO_ROOT")
    if override:
        return Path(override).resolve()
    return Path(__file__).resolve().parents[1]


def _load_source_corpus(repo_root: Path) -> list[dict[str, Any]]:
    """Load the Exp 252 predictive verification corpus.

    Falls back to an empty list with a warning if the file is absent (mock mode).
    """
    src = repo_root / "data" / "research" / _SOURCE_CORPUS_NAME
    if not src.exists():
        print(f"[WARN] Source corpus not found: {src}. Using empty list.", file=sys.stderr)
        return []
    rows: list[dict[str, Any]] = []
    with src.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    """Generate the Exp 262 calibration corpus and summary.

    Can be called programmatically with a list of argument strings (useful
    for tests and subagent invocation).

    Args:
        argv: Command-line argument list.  If None, reads from sys.argv.
    """
    parser = argparse.ArgumentParser(
        description="Experiment 262: generate predictive calibration corpus"
    )
    parser.add_argument(
        "--n-cases",
        type=int,
        default=None,
        help="Limit number of distinct case_ids to include (default: all)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Override output directory for corpus JSONL (default: data/research/)",
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=None,
        help="Override repository root (default: auto-detect)",
    )
    args = parser.parse_args(argv)

    repo_root: Path = args.repo_root or get_repo_root()
    output_dir: Path = args.output_dir or (repo_root / "data" / "research")
    results_dir: Path = repo_root / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    # Load source corpus
    source_rows = _load_source_corpus(repo_root)
    if not source_rows:
        # Mock mode: generate synthetic rows for testing
        source_rows = _generate_mock_rows(args.n_cases or 10)

    # Build corpus
    corpus_rows = build_corpus(source_rows, n_cases=args.n_cases)

    # Write corpus JSONL
    corpus_path = output_dir / CORPUS_OUTPUT_NAME
    with corpus_path.open("w", encoding="utf-8") as fh:
        for row in corpus_rows:
            fh.write(json.dumps(row, sort_keys=True) + "\n")
    print(f"[262] Wrote {len(corpus_rows)} rows to {corpus_path}")

    # Write summary
    summary = build_summary(corpus_rows)
    summary_path = results_dir / SUMMARY_OUTPUT_NAME
    with summary_path.open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, sort_keys=True)
        fh.write("\n")
    print(f"[262] Wrote summary to {summary_path}")
    print(f"[262] n_cases={summary['n_cases']}, violation_rate={summary['violation_rate']:.4f}")


def _generate_mock_rows(n: int) -> list[dict[str, Any]]:
    """Generate synthetic source rows for testing without a real corpus file.

    Produces alternating violated/verified rows with simple GSM8K-style
    partial responses so the corpus builder can function in mock mode.

    Spec: SCENARIO-PRED-262-A
    """
    rows: list[dict[str, Any]] = []
    for i in range(n):
        violated = i % 2 == 0
        if violated:
            partial = (
                '{"final_answer": 42, "claims": ["Step 1: 3 * 14 = 42. '
                'Step 2: 14 + 14 = 42."]}'
            )
            outcome = "violated"
        else:
            partial = (
                '{"final_answer": 42, "claims": ["Step 1: 3 * 14 = 42. '
                'Verified: 3 + 3 + 3 + ... = 42."]}'
            )
            outcome = "verified"
        rows.append({
            "case_id": f"mock-{i:04d}",
            "partial_response": partial,
            "verifier_outcome": outcome,
            "domain": "reasoning",
            "confidence": 0.5 + 0.1 * (i % 5),
            "model": "mock-model",
            "source_experiment": 235,
            "experiment": 252,
        })
    return rows


if __name__ == "__main__":
    main()
