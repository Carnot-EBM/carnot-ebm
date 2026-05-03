#!/usr/bin/env python3
"""Audit numerical claims in docs/arxiv-paper/main.tex against result artifacts.

Spec traces: REQ-PUBLISH-009.
"""

from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_PAPER = _REPO_ROOT / "docs" / "arxiv-paper" / "main.tex"
_DEFAULT_RESULTS = _REPO_ROOT / "results"

CLAIM_RE = re.compile(r"(?P<value>\d+(?:\.\d+)?)\s*(?P<unit>×|x|%|pp|µs|ms|fold|AUROC|KL)(?!\d)")
EXP_RE = re.compile(r"exp(?P<id>\d{3,5})", re.IGNORECASE)
PAREN_RE = re.compile(r"\((?P<body>[^)]{0,240})\)")


@dataclass(frozen=True)
class Claim:
    """A normalized numeric paper claim.

    ``start`` and ``end`` are offsets into the normalized paper text, which is
    also what citation lookup uses.
    """

    raw_value: str
    value: float
    unit: str
    start: int
    end: int
    context: str


def normalize_tex(tex: str) -> str:
    """Return a search-friendly representation of the LaTeX paper text."""
    normalized = tex
    replacements = {
        r"\%": "%",
        r"\times": "×",
        r"\mu s": "µs",
        r"\mu{}s": "µs",
        r"{,}": "",
        r"\,": " ",
        "$": "",
        "{": "",
        "}": "",
    }
    for old, new in replacements.items():
        normalized = normalized.replace(old, new)
    normalized = re.sub(r"(?<=\d),(?=\d)", "", normalized)
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized


def extract_claims(tex: str) -> list[Claim]:
    """Extract normalized numerical claims from paper text."""
    normalized = normalize_tex(tex)
    claims: list[Claim] = []
    for match in CLAIM_RE.finditer(normalized):
        start, end = match.span()
        claims.append(
            Claim(
                raw_value=match.group("value"),
                value=float(match.group("value")),
                unit=match.group("unit"),
                start=start,
                end=end,
                context=normalized[max(0, start - 80) : min(len(normalized), end + 160)],
            )
        )
    return claims


def _candidate_exp_ids(normalized_tex: str, claim: Claim) -> list[str]:
    """Return experiment ids cited in the 200 characters after a claim."""
    window = normalized_tex[claim.end : claim.end + 200]
    ids: list[str] = []
    for paren in PAREN_RE.finditer(window):
        body = paren.group("body")
        if "refsec:exp" in body or "Section~\\refsec:exp" in body:
            continue
        for match in EXP_RE.finditer(body):
            exp_id = match.group("id")
            if exp_id not in ids:
                ids.append(exp_id)
            slash_tail = re.match(r"(?:/\d{3,5})+", body[match.end() :])
            if slash_tail:
                for tail_id in re.findall(r"\d{3,5}", slash_tail.group(0)):
                    if tail_id not in ids:
                        ids.append(tail_id)
    return ids


def _artifact_paths(results_dir: Path, exp_id: str) -> list[Path]:
    """Return local result JSON paths that could back ``exp_id``."""
    exact = results_dir / f"experiment_{exp_id}.json"
    paths = [exact] if exact.exists() else []
    paths.extend(sorted(results_dir.glob(f"experiment_{exp_id}_*.json")))
    return paths


def _walk_numbers(value: Any) -> list[float]:
    """Recursively collect numeric values from a decoded JSON value."""
    if isinstance(value, bool) or value is None:
        return []
    if isinstance(value, int | float):
        return [float(value)]
    if isinstance(value, dict):
        numbers: list[float] = []
        for child in value.values():
            numbers.extend(_walk_numbers(child))
        return numbers
    if isinstance(value, list):
        numbers = []
        for child in value:
            numbers.extend(_walk_numbers(child))
        return numbers
    return []


def _decimal_tolerance(raw_value: str) -> float:
    """Return tolerance implied by the printed precision in ``raw_value``."""
    if "." not in raw_value:
        return 0.5 + 1e-12
    decimals = len(raw_value.split(".", 1)[1])
    return 0.5 * 10 ** (-decimals) + 1e-12


def _claim_value_forms(claim: Claim) -> list[float]:
    """Return numeric forms that may correspond to artifact fields."""
    forms = [claim.value]
    if claim.unit == "%":
        forms.extend([claim.value / 100.0, 1.0 - claim.value / 100.0])
    elif claim.unit == "pp":
        forms.append(claim.value / 100.0)
    return forms


def _number_matches_claim(number: float, claim: Claim) -> bool:
    """Return True when ``number`` matches ``claim`` after rounding/unit handling."""
    tol = _decimal_tolerance(claim.raw_value)
    for form in _claim_value_forms(claim):
        if abs(number - form) <= tol:
            return True
    return False


def _artifact_matches_claim(path: Path, claim: Claim) -> bool:
    """Return whether any numeric field in ``path`` matches ``claim``."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return any(_number_matches_claim(number, claim) for number in _walk_numbers(payload))


def audit_paper_claims(
    paper_path: Path = _DEFAULT_PAPER, results_dir: Path = _DEFAULT_RESULTS
) -> dict:
    """Audit paper claims and return a JSON-serializable report."""
    tex = paper_path.read_text(encoding="utf-8")
    normalized = normalize_tex(tex)
    claims = extract_claims(tex)
    mismatches: list[dict[str, Any]] = []
    cited_count = 0
    verified_count = 0

    for claim in claims:
        exp_ids = _candidate_exp_ids(normalized, claim)
        if not exp_ids:
            continue
        cited_count += 1

        verified = False
        checked_paths: list[str] = []
        for exp_id in exp_ids:
            paths = _artifact_paths(results_dir, exp_id)
            checked_paths.extend(str(path.relative_to(results_dir.parent)) for path in paths)
            if any(_artifact_matches_claim(path, claim) for path in paths):
                verified = True
                break

        if verified:
            verified_count += 1
        else:
            mismatches.append(
                {
                    "value": claim.raw_value,
                    "unit": claim.unit,
                    "exp_id": exp_ids[0],
                    "candidate_exp_ids": exp_ids,
                    "artifact_paths": checked_paths,
                    "context": claim.context,
                }
            )

    total = len(claims)
    citation_ratio = cited_count / total if total else 1.0
    passes = not mismatches and citation_ratio >= 0.8
    return {
        "paper_path": str(paper_path),
        "n_claims_total": total,
        "n_claims_with_artifact_citation": cited_count,
        "n_claims_verified": verified_count,
        "n_mismatches": len(mismatches),
        "citation_ratio": citation_ratio,
        "mismatches": mismatches,
        "passes": passes,
    }


def main() -> None:
    """CLI entrypoint; exits 1 when the paper-claim audit fails."""
    paper_path = Path(sys.argv[1]) if len(sys.argv) >= 2 else _DEFAULT_PAPER
    results_dir = Path(sys.argv[2]) if len(sys.argv) >= 3 else _DEFAULT_RESULTS
    report = audit_paper_claims(paper_path, results_dir)
    print(json.dumps(report, indent=2))
    if not report["passes"]:
        raise SystemExit(1)


if __name__ == "__main__":  # pragma: no cover
    main()
