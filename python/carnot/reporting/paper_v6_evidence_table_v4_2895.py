"""Build the Exp 2895 paper-v6 evidence table and claim-boundary artifact.

Spec refs: REQ-REPORT-2895, SCENARIO-REPORT-2895.

This module is an aggregation step. It reads the already-audited matrix v7 and
the prior capstone claim boundary, turns those source fields into reviewer-
friendly statements, and records the result in JSON. It deliberately does not
edit the paper, the operator-curated landing page, or any external submission
surface because this artifact is only for operator review.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
RUN_DATE = "20260523"
OUTPUT_REL_PATH = Path("results/experiment_2895_paper_v6_evidence_table_v4.json")
MATRIX_V7_REL_PATH = Path("results/experiment_2894_cross_corpus_matrix_v7.json")
CAPSTONE_V272_REL_PATH = Path("results/experiment_2884_capstone_v272.json")

SOURCE_PATHS = (MATRIX_V7_REL_PATH, CAPSTONE_V272_REL_PATH)
EXPECTED_CORPORA = ("FoVer", "HaluEval/FEVER", "MBPP", "HumanEval", "TruthfulQA")

FIELD_PRINCIPLES = {
    "honest_verdict": "Terminal synthesis verdict; no paper edit or external submission is implied.",
    "paper_evidence_table_ready": (
        "True only when matrix v7 is present, built, and yields at least one"
        " headline claim plus bounded non-headline statements."
    ),
    "source_artifacts": "Only source JSON artifacts read from disk for this operator-review table.",
    "headline_claims": "Clean matrix-v7 headline rows that may support paper-v6 row-level claims.",
    "pilot_only_statements": "Pilot rows that may be described only as pilots, never benchmarks.",
    "taxonomy_only_statements": "Taxonomy rows with no generated-answer metric promotion.",
    "blocked_claims": "Rows or support cells held out because flags, blocks, or missing evidence remain.",
    "forbidden_claims": "Claims the paper must not make from the current evidence boundary.",
    "markdown_table": "Compact table for paper-v6 integration review; values mirror machine fields.",
    "arxiv_submission_performed": "Always false; publication is operator-only.",
    "landing_page_modified": "Always false; docs/index.html is operator-curated.",
    "duration_s": "Measured wall time for local JSON aggregation; never padded.",
}


def read_json(path: Path) -> dict[str, Any]:
    """Return a JSON object from disk, or ``{}`` when it cannot be trusted.

    Aggregation artifacts should fail closed. A missing, malformed, or non-object
    input cannot support a citation boundary, so callers receive an empty object
    and record the source problem explicitly in the result artifact.
    """

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def build_artifact(
    root: Path | str = REPO_ROOT,
    *,
    started_s: float | None = None,
    now_s: float | None = None,
) -> dict[str, Any]:
    """REQ-REPORT-2895: build the paper-v6 evidence table without side effects."""

    root_path = Path(root)
    started = time.perf_counter() if started_s is None else started_s
    matrix = read_json(root_path / MATRIX_V7_REL_PATH)
    capstone = read_json(root_path / CAPSTONE_V272_REL_PATH)
    rows = _matrix_rows(matrix)
    missing_rows = _missing_rows(matrix)

    headline_claims = [_headline_claim(row) for row in rows if row.get("headline_eligible")]
    pilot_only = [_pilot_statement(row) for row in rows if row.get("pilot_only")]
    taxonomy_only = [_taxonomy_statement(row) for row in rows if row.get("taxonomy_only")]
    blocked_claims = _blocked_claims(matrix)
    forbidden_claims = _forbidden_claims(matrix, capstone, rows, missing_rows, blocked_claims)
    ready = bool(matrix.get("cross_corpus_matrix_built")) and bool(headline_claims)
    end = time.perf_counter() if now_s is None else now_s

    return {
        "schema": "carnot.paper_v6_evidence_table.v4",
        "artifact": "experiment_2895_paper_v6_evidence_table_v4",
        "honest_verdict": _honest_verdict(
            ready, headline_claims, pilot_only, taxonomy_only, blocked_claims
        ),
        "paper_evidence_table_ready": ready,
        "source_artifacts": _existing_source_artifacts(root_path),
        "headline_claims": headline_claims,
        "pilot_only_statements": pilot_only,
        "taxonomy_only_statements": taxonomy_only,
        "blocked_claims": blocked_claims,
        "forbidden_claims": forbidden_claims,
        "markdown_table": _markdown_table(rows, missing_rows, blocked_claims),
        "arxiv_submission_performed": False,
        "landing_page_modified": False,
        "field_principles": dict(FIELD_PRINCIPLES),
        "run_date": RUN_DATE,
        "duration_s": round(max(0.0, end - started), 6),
    }


def write_artifact(
    root: Path | str = REPO_ROOT,
    *,
    output_path: Path | str = OUTPUT_REL_PATH,
    started_s: float | None = None,
    now_s: float | None = None,
) -> Path:
    """Build and persist the Exp 2895 JSON deliverable."""

    root_path = Path(root)
    out_path = Path(output_path)
    if not out_path.is_absolute():
        out_path = root_path / out_path
    artifact = build_artifact(root_path, started_s=started_s, now_s=now_s)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return out_path


def _matrix_rows(matrix: dict[str, Any]) -> list[dict[str, Any]]:
    rows = matrix.get("matrix_rows", [])
    return [dict(row) for row in rows if isinstance(row, dict)]


def _missing_rows(matrix: dict[str, Any]) -> dict[str, dict[str, Any]]:
    rows = matrix.get("missing_rows", {})
    return {str(corpus): dict(row) for corpus, row in rows.items() if isinstance(row, dict)}


def _existing_source_artifacts(root: Path) -> list[str]:
    return [str(path) for path in SOURCE_PATHS if (root / path).is_file()]


def _fmt(value: object, *, signed: bool = False) -> str:
    numeric = float(value)
    return f"{numeric:+.6f}" if signed else f"{numeric:.6f}"


def _headline_claim(row: dict[str, Any]) -> dict[str, str]:
    corpus = str(row["corpus"])
    primary = dict(row.get("primary_metric") or {})
    label = dict(row.get("label_evidence") or {})
    if "production_auroc" in primary:
        statement = (
            f"{corpus} is a headline-citable row from clean matrix-v7 evidence: "
            f"production AUROC {_fmt(primary['production_auroc'])}, "
            f"architecture-only AUROC {_fmt(primary['architecture_only_auroc'])}, "
            f"learning contribution {_fmt(primary['learning_contribution'], signed=True)} "
            f"over n={label.get('n_examples')} examples / {label.get('n_seeds')} seeds."
        )
        boundary = (
            "Headline citation is limited to this FoVer AUROC comparison and its stated n/seeds."
        )
    else:
        auroc = dict(primary.get("measured_auroc_by_dataset") or {})
        counts = dict(primary.get("n_examples_by_dataset") or {})
        vericot = dict(row.get("vericot_exact_support") or {})
        statement = (
            f"{corpus} is a headline-citable local calibration row: "
            f"HaluEval AUROC {_fmt(auroc['halueval'])} (n={counts.get('halueval')}); "
            f"FEVER AUROC {_fmt(auroc['fever'])} (n={counts.get('fever')})."
        )
        boundary = (
            "VeriCoT support "
            f"{vericot.get('supported_rows')}/{vericot.get('candidate_rows')}; "
            "cite as weak calibration evidence, not broad exact verification."
        )
    return {
        "corpus": corpus,
        "statement": statement,
        "source_artifact": str(row.get("source_artifact")),
        "boundary": boundary,
    }


def _pilot_statement(row: dict[str, Any]) -> dict[str, str]:
    corpus = str(row["corpus"])
    label = dict(row.get("label_evidence") or {})
    structural = dict(row.get("structural_dependency_verification") or {})
    statement = (
        f"{corpus} is pilot-only: manifest row {label.get('stable_id')} has "
        f"{label.get('n_tests')} deterministic tests and structural metadata "
        f"({structural.get('reference_passed')}/{structural.get('reference_rows')} references passed)."
    )
    boundary = (
        "Do not cite pass@k/AUROC, generated-code benchmark lift, or headline code performance."
    )
    return {
        "corpus": corpus,
        "statement": statement,
        "source_artifact": str(row.get("source_artifact")),
        "boundary": boundary,
    }


def _taxonomy_statement(row: dict[str, Any]) -> dict[str, str]:
    taxonomy = dict(row.get("truthfulqa_taxonomy") or {})
    statement = (
        "TruthfulQA has "
        f"{taxonomy.get('n_rows_materialized')}/{taxonomy.get('n_rows_available')} "
        "local taxonomy rows; generated-answer metrics are absent."
    )
    boundary = (
        "Taxonomy-only evidence; do not cite TruthfulQA accuracy, AUROC, or "
        "generated-answer performance."
    )
    return {
        "corpus": str(row["corpus"]),
        "statement": statement,
        "source_artifact": str(row.get("source_artifact")),
        "boundary": boundary,
    }


def _blocked_claims(matrix: dict[str, Any]) -> list[dict[str, Any]]:
    blocked = matrix.get("blocked_rows", {})
    return [
        {
            "corpus": str(corpus),
            "claim": f"{corpus} generated-code support is blocked and cannot be cited.",
            "source_artifact": str(details.get("source_artifact")),
            "reasons": [str(reason) for reason in details.get("reasons", [])],
        }
        for corpus in EXPECTED_CORPORA
        if isinstance((details := blocked.get(corpus)), dict)
    ]


def _forbidden_claims(
    matrix: dict[str, Any],
    capstone: dict[str, Any],
    rows: list[dict[str, Any]],
    missing_rows: dict[str, dict[str, Any]],
    blocked_claims: list[dict[str, Any]],
) -> list[str]:
    claims = [str(claim) for claim in capstone.get("paper_v6_forbidden_claims", [])]
    if not matrix.get("cross_corpus_matrix_built"):
        claims.append(
            "Do not cite headline claims from this artifact because matrix v7 is missing or not built."
        )
    claims.extend(
        f"Do not cite {row['corpus']} as a headline benchmark row; matrix v7 marks it {row['row_status']}."
        for row in rows
        if row.get("pilot_only") or row.get("taxonomy_only")
    )
    claims.extend(
        f"Do not cite {claim['corpus']} generated-code support; blocked by {', '.join(claim['reasons'])}."
        for claim in blocked_claims
    )
    claims.extend(
        f"Do not cite {corpus}; matrix v7 marks the row missing: {_missing_reason(details)}."
        for corpus, details in missing_rows.items()
    )
    return _dedupe(claims)


def _dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    unique: list[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            unique.append(value)
    return unique


def _honest_verdict(
    ready: bool,
    headline_claims: list[dict[str, str]],
    pilot_only: list[dict[str, str]],
    taxonomy_only: list[dict[str, str]],
    blocked_claims: list[dict[str, Any]],
) -> str:
    if ready:
        return (
            "complete: paper-v6 evidence table ready; "
            f"headline={len(headline_claims)}; pilot_only={len(pilot_only)}; "
            f"taxonomy_only={len(taxonomy_only)}; blocked={len(blocked_claims)}"
        )
    return "blocked: paper-v6 evidence table not ready; matrix v7 is missing or not built"


def _markdown_table(
    rows: list[dict[str, Any]],
    missing_rows: dict[str, dict[str, Any]],
    blocked_claims: list[dict[str, Any]],
) -> str:
    by_corpus = {str(row["corpus"]): row for row in rows}
    blocked_by_corpus = {claim["corpus"]: claim for claim in blocked_claims}
    lines = [
        "| Corpus | Evidence class | Paper-v6 use | Blocked/Missing boundary | Source |",
        "|---|---|---|---|---|",
    ]
    for corpus in EXPECTED_CORPORA:
        lines.append(_markdown_row(corpus, by_corpus.get(corpus), missing_rows, blocked_by_corpus))
    return "\n".join(lines)


def _markdown_row(
    corpus: str,
    row: dict[str, Any] | None,
    missing_rows: dict[str, dict[str, Any]],
    blocked_by_corpus: dict[str, dict[str, Any]],
) -> str:
    if row is None:
        missing = missing_rows.get(corpus, {})
        return f"| {corpus} | missing | Not citable | missing: {_missing_reason(missing)} | n/a |"
    if row.get("headline_eligible"):
        return f"| {corpus} | headline | Headline row | none | {row.get('source_artifact')} |"
    if row.get("pilot_only"):
        blocked = blocked_by_corpus.get(corpus)
        boundary = (
            "blocked: " + "; ".join(blocked["reasons"])
            if blocked
            else "pilot-only no headline metric"
        )
        return f"| {corpus} | pilot-only | Pilot statement only | {boundary} | {row.get('source_artifact')} |"
    taxonomy = dict(row.get("truthfulqa_taxonomy") or {})
    boundary = (
        "no generated-answer metric"
        if taxonomy.get("generated_answer_metrics_available") is False
        else "taxonomy-only"
    )
    return f"| {corpus} | taxonomy-only | Taxonomy statement only | {boundary} | {row.get('source_artifact')} |"


def _missing_reason(details: dict[str, Any]) -> str:
    primary = details.get("primary_metric")
    if isinstance(primary, dict) and primary.get("reason"):
        return str(primary["reason"])
    return str(details.get("reason") or details.get("row_status") or "row_absent")


if __name__ == "__main__":  # pragma: no cover
    print(write_artifact())
