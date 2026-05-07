"""Build the Exp 1462 paper-v6 anchored-claims narrowing artifact.

The publication hold is an editorial safety gate: paper-v6 should make only
claims that have both empirical artifacts and theory support. This module
keeps that gate mechanical. It writes the deliverable first, narrows the claim
set to four explicit rows, updates the paper source when one is present, and
records that no arXiv submission was triggered.

Spec refs: REQ-PUBLISH-021, SCENARIO-PUBLISH-023.
"""

from __future__ import annotations

import json
import os
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[3]
RUN_DATE = "20260507"
OUTPUT_FILENAME = "experiment_1462_paper_v6_anchored_claims_narrowing.json"
DEFAULT_OUT_PATH = REPO_ROOT / "results" / OUTPUT_FILENAME
CLAIM_MATRIX_REL = Path("docs/research-notes/paper_v6_anchored_claim_matrix.md")
DEFAULT_CLAIM_MATRIX_PATH = REPO_ROOT / CLAIM_MATRIX_REL
PROJECT_ROOT_FOR_METADATA = "/home/ianblenke/github.com/ianblenke/carnot"
SCHEMA = "paper_v6_anchored_claims_narrowing_v1"
EXPERIMENT = "1462_paper_v6_anchored_claims_narrowing"

ANCHOR_START = "% exp1462-anchored-claims-start"
ANCHOR_END = "% exp1462-anchored-claims-end"
FUTURE_START = "% exp1462-unsupported-future-work-start"
FUTURE_END = "% exp1462-unsupported-future-work-end"

DEFAULT_PAPER_CANDIDATES = (
    Path("docs/arxiv-paper/main.tex"),
    Path("paper/main.tex"),
    Path("paper/paper-v6.tex"),
    Path("docs/position-paper-draft-v3.md"),
)

REQUIRED_ARTIFACT_FIELDS = {
    "status",
    "paper_source_path",
    "anchored_claim_count",
    "anchored_claims",
    "unanchored_claims_moved",
    "claim_matrix_path",
    "paper_updated",
    "arxiv_submission_triggered",
    "honest_verdict",
}

REQUIRED_SOURCE_INPUTS = {
    Path("ops/experiment_signal_noise_summary.md"): "Experiment Signal / Noise Classification Summary",
    Path("ops/mandatory_priority_audit.md"): "Mandatory Priority Audit",
    Path("docs/research-notes/self_learning_lineage_decision.md"): "Self-Learning Lineage Decision",
    Path("docs/research-notes/hardware_portfolio_narrowing.md"): "Hardware Portfolio Narrowing",
    Path("docs/research-notes/comparator_cite_retire_audit.md"): "Comparator Cite/Retire Audit",
    Path("results/experiment_1454_experiment_artifact_signal_noise_classifier.json"): "exp1454",
    Path("results/experiment_1455_known_issues_mandatory_priority_audit.json"): "exp1455",
    Path("results/experiment_1459_self_learning_nonheadline_lineage_decision.json"): "exp1459",
    Path("results/experiment_1460_hardware_portfolio_narrowing.json"): "exp1460",
    Path("results/experiment_1461_comparator_integration_cite_retire_audit.json"): "exp1461",
}

_ANCHORED_CLAIMS: list[dict[str, Any]] = [
    {
        "claim_id": "CLAIM-1",
        "title": "Verifier composition is bounded by measured correlation",
        "claim": (
            "Carnot may claim a measured heterogeneous k=5 verifier stack and a "
            "homogeneous text-probe ceiling; it may not claim arbitrary k scaling."
        ),
        "empirical_artifact_paths": [
            "results/experiment_1093_phase1c_verifier_joint_null_space_measurement.json",
            "results/experiment_1224_phase5c_adversarial_probe.json",
            "results/experiment_1256_verifier_orthogonality_audit_v3.json",
        ],
        "theoretical_support": [
            "sqrt(det(Sigma)) joint-volume correction",
            "Welch/Rankin Simplex bound on verifier packing",
            "Spera non-composability boundary for shared verifier null spaces",
        ],
        "claim_boundary": (
            "Does not claim k=6, k=15, or exponential AND-composition gains "
            "beyond the measured evidence."
        ),
        "paper_section": "Anchored Claims / Theoretical Bounds",
    },
    {
        "claim_id": "CLAIM-2",
        "title": "Exact sampling requires sparse fast-path plus CPU fallback",
        "claim": (
            "Carnot's hardware story is a correctness-first chi<=4 sparse "
            "constraint fast-path with CPU fallback, not a universal FPGA speedup."
        ),
        "empirical_artifact_paths": [
            "results/experiment_1068_kv260_smoke_test_v9.json",
            "results/experiment_1094_phase2a_sampler_correctness_audit.json",
            "results/experiment_1451_discrete_sb_rtl_lint_sim_rerun.json",
            "results/experiment_1460_hardware_portfolio_narrowing.json",
        ],
        "theoretical_support": [
            "single-site Gibbs detailed balance",
            "chromatic-Glauber batching under graph-coloring constraints",
            "scope-reduction hardware portfolio gate",
        ],
        "claim_boundary": (
            "Does not claim same-basis CPU-vs-FPGA speedup, KV260 board "
            "execution for new bitstreams, Extropic execution, NPU acceleration, "
            "or photonic execution."
        ),
        "paper_section": "Anchored Claims / Hardware Acceleration",
    },
    {
        "claim_id": "CLAIM-3",
        "title": "Energy verifier calibration is distribution-bound",
        "claim": (
            "Carnot may claim in-distribution SOS-KAN calibration, a measured "
            "SOTA-output energy-ordering inversion, and a SOTA-inclusive retrain "
            "that fixes the observed inversion on its validation split."
        ),
        "empirical_artifact_paths": [
            "results/experiment_1072_sos_kan_v3_neural_gram.json",
            "results/experiment_1100_cascade_validation_sota_outputs.json",
            "results/experiment_1120_energy_verifier_retrain_sota.json",
            "results/experiment_1265_diffutruth_vs_carnot_baseline.json",
        ],
        "theoretical_support": [
            "Goodhart/reward-hacking interpretation of verifier-shaped optimization",
            "energy-based calibration under distribution shift",
            "OOD boundary between FoVer and optimized SOTA-output corpora",
        ],
        "claim_boundary": (
            "Does not claim universal verifier dominance, cross-corpus DiffuTruth "
            "dominance, or future SOTA-family generalization."
        ),
        "paper_section": "Anchored Claims / Empirical Realities",
    },
    {
        "claim_id": "CLAIM-4",
        "title": "Self-learning is a narrow verified-memory-growth claim",
        "claim": (
            "Carnot may claim one verified self-learning pivot: semantic-verified "
            "fresh memory growth with non-forgetting, as selected by Exp 1459."
        ),
        "empirical_artifact_paths": [
            "results/experiment_1374_continuous_self_learning_v3_verifier_selected_or_csp_fallback.json",
            "results/experiment_1433_fr11_self_learning_v6_dvi_v3_gated.json",
            "results/experiment_1447_fr11_v7_memory_policy_growth.json",
            "results/experiment_1459_self_learning_nonheadline_lineage_decision.json",
        ],
        "theoretical_support": [
            "Zenil exogenous-grounding condition for self-distillation",
            "primary semantic-verifier acceptance path",
            "non-forgetting gate for persisted memory promotion",
        ],
        "claim_boundary": (
            "Does not claim replay-only learning, adapter-only learning, broad "
            "autonomous self-improvement, or completed DVI training."
        ),
        "paper_section": "Anchored Claims / Self-Learning Scope",
    },
]

_UNANCHORED_MOVED: list[dict[str, str]] = [
    {
        "topic": "same-basis CPU-vs-FPGA speedup",
        "destination": "future_work",
        "reason": (
            "The KV260 latency point is measured, but same-N CPU-vs-FPGA timing "
            "has not been measured on the same per-sample basis."
        ),
        "supporting_input": "results/experiment_1460_hardware_portfolio_narrowing.json",
    },
    {
        "topic": "Extropic Z1/XTR-0, NPU, photonic, D-Wave, and large-FPGA execution",
        "destination": "future_work",
        "reason": (
            "The 20260507 hardware narrowing keeps these tracks deferred until "
            "authenticated local execution or concrete reopen gates exist."
        ),
        "supporting_input": "docs/research-notes/hardware_portfolio_narrowing.md",
    },
    {
        "topic": "k=6, k=15, and arbitrary verifier-composition scaling",
        "destination": "appendix",
        "reason": (
            "Measured k_eff and joint-null-space evidence support only the "
            "bounded k=5 framing plus theory-backed future cross-mechanism work."
        ),
        "supporting_input": "results/experiment_1256_verifier_orthogonality_audit_v3.json",
    },
    {
        "topic": "broad self-learning improves everything wording",
        "destination": "future_work",
        "reason": (
            "Exp 1459 permits only the Exp 1447 verified-memory-growth pivot; "
            "replay-only and adapter-only claims stay non-headline."
        ),
        "supporting_input": "results/experiment_1459_self_learning_nonheadline_lineage_decision.json",
    },
    {
        "topic": "LARQL, Skillify, GStack, and ontology-governance comparator territory",
        "destination": "future_work",
        "reason": (
            "Exp 1461 classifies these as watchlist or retired rather than active "
            "paper-v6 claims."
        ),
        "supporting_input": "results/experiment_1461_comparator_integration_cite_retire_audit.json",
    },
    {
        "topic": "full ARC-AGI-3 or Seed IQ parity",
        "destination": "future_work",
        "reason": (
            "The local Phase-4 result is a toy proxy bridge; no full ARC-AGI-3 "
            "leaderboard or Seed IQ reproduction claim is locally anchored."
        ),
        "supporting_input": "results/experiment_1165_phase4_active_inference_pilot_v1.json",
    },
]


def default_anchored_claims() -> list[dict[str, Any]]:
    """Return the conservative paper-v6 claim set selected by the scope gate."""

    return deepcopy(_ANCHORED_CLAIMS)


def default_unanchored_claims_moved() -> list[dict[str, str]]:
    """Return unsupported territory that stays visible as appendix/future work."""

    return deepcopy(_UNANCHORED_MOVED)


def write_in_progress_artifact(out_path: str | Path = DEFAULT_OUT_PATH) -> dict[str, Any]:
    """REQ-PUBLISH-021: write a schema-shaped startup marker before analysis."""

    artifact: dict[str, Any] = {
        "schema": SCHEMA,
        "experiment": EXPERIMENT,
        "run_date": RUN_DATE,
        "project_root": PROJECT_ROOT_FOR_METADATA,
        "spec": ["REQ-PUBLISH-021", "SCENARIO-PUBLISH-023"],
        "status": "in_progress",
        "paper_source_path": None,
        "anchored_claim_count": 0,
        "anchored_claims": [],
        "unanchored_claims_moved": [],
        "claim_matrix_path": None,
        "paper_updated": False,
        "arxiv_submission_triggered": False,
        "honest_verdict": "in_progress",
    }
    return _write_json(Path(out_path), artifact)


def locate_paper_source(
    root: str | Path = REPO_ROOT,
    paper_candidates: Sequence[str | Path] | None = None,
) -> Path | None:
    """Find the active paper source without inventing one when it is absent."""

    root_path = Path(root)
    candidates = DEFAULT_PAPER_CANDIDATES if paper_candidates is None else paper_candidates
    for candidate in candidates:
        candidate_path = Path(candidate)
        path = candidate_path if candidate_path.is_absolute() else root_path / candidate_path
        if path.is_file():
            return path
    return None


def inspect_source_inputs(root: str | Path = REPO_ROOT) -> dict[str, dict[str, Any]]:
    """Summarize the source artifacts that constrain the narrowed claims."""

    root_path = Path(root)
    status: dict[str, dict[str, Any]] = {}
    for rel_path in REQUIRED_SOURCE_INPUTS:
        path = root_path / rel_path
        row: dict[str, Any] = {"exists": path.exists()}
        if path.exists() and path.suffix == ".json":
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                row["parse_error"] = "json_decode_error"
            else:
                row["status"] = payload.get("status")
                row["honest_verdict"] = payload.get("honest_verdict")
        elif path.exists():
            row["bytes"] = path.stat().st_size
        status[rel_path.as_posix()] = row
    return status


def render_claim_matrix(
    anchored_claims: Sequence[Mapping[str, Any]],
    unanchored_claims_moved: Sequence[Mapping[str, str]],
    *,
    paper_source_path: str | None,
    source_input_status: Mapping[str, Mapping[str, Any]] | None = None,
) -> str:
    """Render the reviewer-facing matrix that ties every claim to evidence."""

    lines = [
        "# Paper v6 Anchored Claims Matrix",
        "",
        f"Run date: `{RUN_DATE}`",
        f"Paper source path: `{paper_source_path or 'not found'}`",
        "",
        "## Anchored Claims",
        "",
        "| Claim | Paper Section | Empirical Artifacts | Theoretical Support | Boundary |",
        "|---|---|---|---|---|",
    ]
    for claim in anchored_claims:
        lines.append(
            "| {claim_id}: {title} | {paper_section} | {artifacts} | {theory} | {boundary} |".format(
                claim_id=_md_cell(str(claim["claim_id"])),
                title=_md_cell(str(claim["title"])),
                paper_section=_md_cell(str(claim["paper_section"])),
                artifacts=_md_cell(", ".join(claim["empirical_artifact_paths"])),
                theory=_md_cell(", ".join(claim["theoretical_support"])),
                boundary=_md_cell(str(claim["claim_boundary"])),
            )
        )
    lines.extend(
        [
            "",
            "## Unsupported Territory Moved",
            "",
            "| Topic | Destination | Reason | Supporting Input |",
            "|---|---|---|---|",
        ]
    )
    for row in unanchored_claims_moved:
        lines.append(
            "| {topic} | {destination} | {reason} | {supporting_input} |".format(
                topic=_md_cell(row["topic"]),
                destination=_md_cell(row["destination"]),
                reason=_md_cell(row["reason"]),
                supporting_input=_md_cell(row["supporting_input"]),
            )
        )
    if source_input_status is not None:
        lines.extend(["", "## Source Inputs Reviewed", ""])
        for path, row in source_input_status.items():
            verdict = row.get("honest_verdict") or row.get("status") or row.get("bytes") or "present"
            exists = "yes" if row.get("exists") else "no"
            lines.append(f"- `{path}`: exists={exists}; summary={verdict}")
    return "\n".join(lines) + "\n"


def update_paper_text(
    paper_text: str,
    anchored_claims: Sequence[Mapping[str, Any]],
    unanchored_claims_moved: Sequence[Mapping[str, str]],
) -> tuple[str, bool]:
    """Insert or replace the paper's anchored-claims and future-work sections."""

    anchored_block = _render_latex_anchored_claims(anchored_claims)
    future_block = _render_latex_future_work(unanchored_claims_moved)
    with_anchored = _replace_or_insert_block(
        paper_text,
        start_marker=ANCHOR_START,
        end_marker=ANCHOR_END,
        block=anchored_block,
        fallback_marker="\\section{Related Work}",
    )
    with_future = _replace_or_insert_block(
        with_anchored,
        start_marker=FUTURE_START,
        end_marker=FUTURE_END,
        block=future_block,
        fallback_marker="\\end{document}",
        insert_after_marker="\\appendix",
    )
    return with_future, with_future != paper_text


def build_artifact(
    *,
    paper_source_path: str | None,
    anchored_claims: Sequence[Mapping[str, Any]],
    unanchored_claims_moved: Sequence[Mapping[str, str]],
    claim_matrix_path: str,
    paper_updated: bool,
    source_input_status: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Build the terminal Exp 1462 artifact after paper/matrix writes complete."""

    claims = [dict(claim) for claim in anchored_claims]
    moved = [dict(row) for row in unanchored_claims_moved]
    suffix = "paper_updated_true" if paper_updated else "no_paper_source_updated_false"
    artifact: dict[str, Any] = {
        "schema": SCHEMA,
        "experiment": EXPERIMENT,
        "run_date": RUN_DATE,
        "project_root": PROJECT_ROOT_FOR_METADATA,
        "spec": ["REQ-PUBLISH-021", "SCENARIO-PUBLISH-023"],
        "status": "complete",
        "paper_source_path": paper_source_path,
        "anchored_claim_count": len(claims),
        "anchored_claims": claims,
        "unanchored_claims_moved": moved,
        "claim_matrix_path": claim_matrix_path,
        "paper_updated": paper_updated,
        "arxiv_submission_triggered": False,
        "source_input_status": dict(source_input_status),
        "publish_or_submit_commands_run": [],
        "honest_verdict": f"paper_v6_narrowed_to_{len(claims)}_anchored_claims_{suffix}",
    }
    return artifact


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """Validate the required schema and the no-submission claim boundary."""

    missing = REQUIRED_ARTIFACT_FIELDS - set(artifact)
    if missing:
        raise ValueError(f"missing required fields: {sorted(missing)}")
    if artifact["status"] != "complete":
        raise ValueError("status must be complete")
    claims = artifact["anchored_claims"]
    if not isinstance(claims, list) or not 3 <= len(claims) <= 5:
        raise ValueError("anchored_claims must contain between 3 and 5 claims")
    if artifact["anchored_claim_count"] != len(claims):
        raise ValueError("anchored_claim_count must match anchored_claims")
    for claim in claims:
        _validate_claim(claim)
    moved = artifact["unanchored_claims_moved"]
    if not isinstance(moved, list) or not moved:
        raise ValueError("unanchored_claims_moved must be a non-empty list")
    for row in moved:
        if row.get("destination") not in {"appendix", "future_work"}:
            raise ValueError(f"unsupported destination for {row.get('topic')}")
        if not row.get("reason"):
            raise ValueError(f"missing reason for {row.get('topic')}")
    if not artifact["claim_matrix_path"]:
        raise ValueError("claim_matrix_path must be set")
    if artifact["paper_source_path"] is None and artifact["paper_updated"] is True:
        raise ValueError("paper_updated cannot be true when no paper source exists")
    if artifact["arxiv_submission_triggered"] is not False:
        raise ValueError("arxiv_submission_triggered must remain false")
    if "anchored_claims" not in str(artifact["honest_verdict"]):
        raise ValueError("honest_verdict must describe anchored-claims narrowing")


def run(
    *,
    root: str | Path = REPO_ROOT,
    out_path: str | Path = DEFAULT_OUT_PATH,
    claim_matrix_path: str | Path | None = None,
    paper_candidates: Sequence[str | Path] | None = None,
) -> dict[str, Any]:
    """Run Exp 1462 locally without any publication or submission command."""

    root_path = Path(root)
    output = Path(out_path)
    matrix_path = Path(claim_matrix_path) if claim_matrix_path else root_path / CLAIM_MATRIX_REL
    write_in_progress_artifact(output)

    source_input_status = inspect_source_inputs(root_path)
    claims = default_anchored_claims()
    moved = default_unanchored_claims_moved()
    paper_path = locate_paper_source(root_path, paper_candidates)
    paper_source_rel = _relative_path(paper_path, root_path) if paper_path else None

    matrix_path.parent.mkdir(parents=True, exist_ok=True)
    matrix_path.write_text(
        render_claim_matrix(
            claims,
            moved,
            paper_source_path=paper_source_rel,
            source_input_status=source_input_status,
        ),
        encoding="utf-8",
    )

    paper_updated = False
    if paper_path is not None:
        updated_text, paper_updated = update_paper_text(
            paper_path.read_text(encoding="utf-8"),
            claims,
            moved,
        )
        if paper_updated:
            paper_path.write_text(updated_text, encoding="utf-8")

    artifact = build_artifact(
        paper_source_path=paper_source_rel,
        anchored_claims=claims,
        unanchored_claims_moved=moved,
        claim_matrix_path=_relative_path(matrix_path, root_path),
        paper_updated=paper_updated,
        source_input_status=source_input_status,
    )
    validate_artifact(artifact)
    return _write_json(output, artifact)


def _validate_claim(claim: Mapping[str, Any]) -> None:
    required = {
        "claim_id",
        "title",
        "claim",
        "empirical_artifact_paths",
        "theoretical_support",
        "claim_boundary",
        "paper_section",
    }
    missing = required - set(claim)
    if missing:
        raise ValueError(f"claim {claim.get('claim_id')} missing fields: {sorted(missing)}")
    if not claim["empirical_artifact_paths"]:
        raise ValueError(f"claim {claim['claim_id']} lacks empirical_artifact_paths")
    if not claim["theoretical_support"]:
        raise ValueError(f"claim {claim['claim_id']} lacks theoretical_support")
    if not str(claim["claim_boundary"]).startswith("Does not claim"):
        raise ValueError(f"claim {claim['claim_id']} must state a negative boundary")


def _render_latex_anchored_claims(anchored_claims: Sequence[Mapping[str, Any]]) -> str:
    lines = [
        ANCHOR_START,
        "\\section{Anchored Claims}",
        "\\label{sec:anchored-claims}",
        "",
        (
            "This paper-v6 narrowing pass treats the following claims as the "
            "complete headline claim set. Each claim is retained only because it "
            "has checked-in empirical artifacts and a named theoretical support "
            "boundary; broader interpretations move to Appendix or future work."
        ),
        "",
        "\\begin{enumerate}",
    ]
    for claim in anchored_claims:
        artifacts = ", ".join(f"\\path{{{path}}}" for path in claim["empirical_artifact_paths"])
        theory = "; ".join(claim["theoretical_support"])
        lines.extend(
            [
                (
                    f"  \\item \\textbf{{{claim['claim_id']} --- "
                    f"{_latex_text(str(claim['title']))}.}} "
                    f"{_latex_text(str(claim['claim']))}"
                ),
                f"  Empirical artifacts: {artifacts}.",
                f"  Theoretical support: {_latex_text(theory)}.",
                f"  Boundary: {_latex_text(str(claim['claim_boundary']))}.",
            ]
        )
    lines.extend(["\\end{enumerate}", ANCHOR_END])
    return "\n".join(lines) + "\n\n"


def _render_latex_future_work(unanchored_claims_moved: Sequence[Mapping[str, str]]) -> str:
    lines = [
        FUTURE_START,
        "\\section{Unsupported Territory Moved to Appendix/Future Work}",
        "\\label{app:unsupported-future-work}",
        "",
        (
            "The following topics remain useful research context, but they are not "
            "paper-v6 headline claims because the local evidence chain is not yet "
            "strong enough."
        ),
        "",
        "\\begin{itemize}",
    ]
    for row in unanchored_claims_moved:
        destination = "appendix" if row["destination"] == "appendix" else "future work"
        lines.extend(
            [
                (
                    f"  \\item \\textbf{{{_latex_text(row['topic'])}}} moves to "
                    f"{destination}: {_latex_text(row['reason'])}"
                ),
                f"  Supporting input: \\path{{{row['supporting_input']}}}.",
            ]
        )
    lines.extend(["\\end{itemize}", FUTURE_END])
    return "\n".join(lines) + "\n\n"


def _replace_or_insert_block(
    text: str,
    *,
    start_marker: str,
    end_marker: str,
    block: str,
    fallback_marker: str,
    insert_after_marker: str | None = None,
) -> str:
    if start_marker in text and end_marker in text:
        start = text.index(start_marker)
        end = text.index(end_marker, start) + len(end_marker)
        if end < len(text) and text[end : end + 2] == "\n\n":
            end += 2
        elif end < len(text) and text[end : end + 1] == "\n":
            end += 1
        return text[:start] + block + text[end:]
    if insert_after_marker and insert_after_marker in text:
        insert_at = text.index(insert_after_marker) + len(insert_after_marker)
        if insert_at < len(text) and text[insert_at] == "\n":
            insert_at += 1
        return text[:insert_at] + "\n" + block + text[insert_at:]
    if fallback_marker in text:
        insert_at = text.index(fallback_marker)
        return text[:insert_at] + block + text[insert_at:]
    return text.rstrip() + "\n\n" + block


def _relative_path(path: Path, root: Path = REPO_ROOT) -> str:
    return os.path.relpath(path, root)


def _md_cell(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", " ")


def _latex_text(value: str) -> str:
    return (
        value.replace("\\", "\\textbackslash{}")
        .replace("&", "\\&")
        .replace("%", "\\%")
        .replace("_", "\\_")
    )


def _write_json(path: Path, payload: Mapping[str, Any]) -> dict[str, Any]:
    artifact = dict(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return artifact


def main() -> None:  # pragma: no cover
    """CLI entry point for manual artifact refreshes."""

    artifact = run()
    print(
        artifact["anchored_claim_count"],
        artifact["paper_updated"],
        artifact["arxiv_submission_triggered"],
    )


if __name__ == "__main__":  # pragma: no cover
    main()
