"""Build the Exp 1461 comparator cite/retire audit artifact.

Spec: REQ-REPORT-045, SCENARIO-REPORT-045.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
RUN_DATE = "20260507"
PROJECT_ROOT_FOR_METADATA = "/home/ianblenke/github.com/ianblenke/carnot"
SCHEMA = "comparator_cite_retire_audit_v1"
EXPERIMENT = "1461_comparator_integration_cite_retire_audit"

DEFAULT_OUT_PATH = (
    REPO_ROOT / "results" / "experiment_1461_comparator_integration_cite_retire_audit.json"
)
DEFAULT_TABLE_PATH = REPO_ROOT / "docs" / "research-notes" / "comparator_cite_retire_audit.md"
DEFAULT_REFERENCES_PATH = REPO_ROOT / "research-references.md"

STATUS_CLARIFICATION_SENTINEL = "<!-- exp1461-comparator-audit-status -->"

REQUIRED_ARTIFACT_FIELDS = {
    "status",
    "comparator_decision_count",
    "cite_count",
    "retire_count",
    "watchlist_count",
    "decision_table_path",
    "references_updated",
    "paper_related_work_implications",
    "honest_verdict",
}

REQUIRED_COMPARATORS = [
    "Abstract-CoT",
    "Meta-Harness",
    "Autodata",
    "LARQL",
    "Skillify",
    "GStack",
    "EBT/NRGPT",
    "ARM-as-EBM",
    "BEAVER",
    "ontology-constrained reasoning",
]

REFERENCE_STATUS_PHRASES = {
    "cite": "paper-v6 cite",
    "retire": "retired from active scope",
    "future_watchlist": "future watchlist",
}


@dataclass(frozen=True)
class ComparatorDecision:
    comparator: str
    decision: str
    rationale: str
    impacted_paper_section: str
    future_reopen_condition: str
    source_basis: str


DEFAULT_DECISIONS = [
    ComparatorDecision(
        comparator="Abstract-CoT",
        decision="cite",
        rationale=(
            "Closest discrete-latent reasoning peer; use it to contrast internal "
            "self-distillation with Carnot's externally grounded verifier ensemble."
        ),
        impacted_paper_section="Related Work / discrete-latent reasoning",
        future_reopen_condition="Not applicable.",
        source_basis="docs/research-notes/paper-v5-decentralization-section-draft.md",
    ),
    ComparatorDecision(
        comparator="Meta-Harness",
        decision="cite",
        rationale=(
            "Measured outer-loop harness optimization is a constructive self-improvement "
            "comparator for the Sakana-defense and conductor-harness discussion."
        ),
        impacted_paper_section="Related Work / self-improving harnesses",
        future_reopen_condition="Not applicable.",
        source_basis="research-references.md and paper-v5 decentralization draft",
    ),
    ComparatorDecision(
        comparator="Autodata",
        decision="cite",
        rationale=(
            "Boltzmann-sampled Challenger/Solver/Judge data generation sharpens the "
            "contrast between single-judge loops and Carnot's formal verifier ensemble."
        ),
        impacted_paper_section="Related Work / self-improving data generation",
        future_reopen_condition="Not applicable.",
        source_basis="research-references.md and paper-v5 decentralization draft",
    ),
    ComparatorDecision(
        comparator="LARQL",
        decision="future_watchlist",
        rationale=(
            "Sovereignty-aligned decoupled-attention infrastructure is promising for "
            "paper-v7 hardware deployment, but it has no local validation for paper-v6."
        ),
        impacted_paper_section="Future Work / sovereignty deployment",
        future_reopen_condition="Prototype only after hardware narrowing selects LARQL as active scope.",
        source_basis="ops/known-issues.md LARQL series",
    ),
    ComparatorDecision(
        comparator="Skillify",
        decision="future_watchlist",
        rationale=(
            "Useful operational pattern for trace2skill daily evals, but it is process "
            "hygiene rather than a paper-v6 model or verifier comparator."
        ),
        impacted_paper_section="Future Work / operational self-learning hygiene",
        future_reopen_condition="Reconsider after a daily-eval trace2skill audit produces local evidence.",
        source_basis="ops/known-issues.md trace2skill + Skillify series",
    ),
    ComparatorDecision(
        comparator="GStack",
        decision="retire",
        rationale=(
            "Only appears as a named comparator in the scope-reduction directive; no "
            "source, empirical hook, or paper-v6 thesis link is recorded locally."
        ),
        impacted_paper_section="None; remove from active comparator scope",
        future_reopen_condition=(
            "Reopen only if a concrete source, thesis link, and falsifiable Carnot "
            "acceptance gate are filed."
        ),
        source_basis="ops/known-issues.md comparator list only",
    ),
    ComparatorDecision(
        comparator="EBT/NRGPT",
        decision="cite",
        rationale=(
            "They define the energy-based reasoning novelty boundary; Carnot should cite "
            "them while keeping local EBT/NRGPT work smoke-only until decoded quality improves."
        ),
        impacted_paper_section="Related Work / energy-based reasoning baselines",
        future_reopen_condition="Not applicable.",
        source_basis="research-references.md and energy-based LLM alternatives notes",
    ),
    ComparatorDecision(
        comparator="ARM-as-EBM",
        decision="cite",
        rationale=(
            "The ARM/EBM bridge is the theory boundary for interpreting local AR logits "
            "as implicit lookahead energy beside Carnot's explicit verifier energy."
        ),
        impacted_paper_section="Related Work / AR-as-energy theory",
        future_reopen_condition="Not applicable.",
        source_basis="research-references.md ARM-EBM status checks",
    ),
    ComparatorDecision(
        comparator="BEAVER",
        decision="cite",
        rationale=(
            "Deterministic semantic-bound reporting directly strengthens Carnot's "
            "certificate and false-acceptance-bound story."
        ),
        impacted_paper_section="Related Work / deterministic verification bounds",
        future_reopen_condition="Not applicable.",
        source_basis="research-references.md and verifier-orthogonality notes",
    ),
    ComparatorDecision(
        comparator="ontology-constrained reasoning",
        decision="future_watchlist",
        rationale=(
            "Enterprise ontology constraints reinforce symbolic contracts, but the current "
            "paper-v6 claim set is verifier-grounded output checking rather than tool governance."
        ),
        impacted_paper_section="Future Work / agent-governance contracts",
        future_reopen_condition=(
            "Reconsider if Carnot adds ontology-backed tool discovery or governance "
            "thresholds as measured local functionality."
        ),
        source_basis="research-references.md Ontology-Constrained Agentic Reasoning",
    ),
]


def _write_json(path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return dict(payload)


def _relative_path(path: Path, root: Path = REPO_ROOT) -> str:
    return os.path.relpath(path, root)


def default_decisions() -> list[ComparatorDecision]:
    return list(DEFAULT_DECISIONS)


def write_in_progress_artifact(out_path: Path | str = DEFAULT_OUT_PATH) -> dict[str, Any]:
    artifact: dict[str, Any] = {field: None for field in REQUIRED_ARTIFACT_FIELDS}
    artifact.update(
        {
            "schema": SCHEMA,
            "experiment": EXPERIMENT,
            "run_date": RUN_DATE,
            "project_root": PROJECT_ROOT_FOR_METADATA,
            "status": "in_progress",
            "comparator_decision_count": 0,
            "cite_count": 0,
            "retire_count": 0,
            "watchlist_count": 0,
            "decision_table_path": _relative_path(DEFAULT_TABLE_PATH),
            "references_updated": False,
            "paper_related_work_implications": "pending",
            "honest_verdict": "in_progress",
        }
    )
    return _write_json(Path(out_path), artifact)


def _decision_count(decisions: list[ComparatorDecision], decision: str) -> int:
    return sum(row.decision == decision for row in decisions)


def build_artifact(
    decisions: list[ComparatorDecision],
    decision_table_path: str,
    references_updated: bool,
) -> dict[str, Any]:
    cite_count = _decision_count(decisions, "cite")
    retire_count = _decision_count(decisions, "retire")
    watchlist_count = _decision_count(decisions, "future_watchlist")
    artifact: dict[str, Any] = {field: None for field in REQUIRED_ARTIFACT_FIELDS}
    artifact.update(
        {
            "schema": SCHEMA,
            "experiment": EXPERIMENT,
            "run_date": RUN_DATE,
            "project_root": PROJECT_ROOT_FOR_METADATA,
            "status": "complete",
            "comparator_decision_count": len(decisions),
            "cite_count": cite_count,
            "retire_count": retire_count,
            "watchlist_count": watchlist_count,
            "decision_table_path": decision_table_path,
            "references_updated": references_updated,
            "paper_related_work_implications": (
                "paper-v6 cites only the six comparator rows that sharpen the "
                "Related Work novelty boundary; watchlist rows move to future "
                "work and the retired row leaves active scope."
            ),
            "honest_verdict": (
                f"comparator_scope_narrowed_{cite_count}_cite_"
                f"{retire_count}_retire_{watchlist_count}_watchlist"
            ),
            "decisions": [asdict(row) for row in decisions],
            "source_files_checked": [
                "ops/known-issues.md",
                "research-references.md",
                "docs/research-notes/paper-v5-decentralization-section-draft.md",
                "docs/research-notes/energy-based-llm-alternatives-deep-research-results.md",
                "docs/research-notes/nrgpt-non-monotonicity-interpretation-deep-think-results.md",
            ],
        }
    )
    return artifact


def _cell(value: str) -> str:
    return value.replace("|", "\\|")


def render_decision_table(decisions: list[ComparatorDecision]) -> str:
    lines = [
        "# Comparator Cite/Retire Audit",
        "",
        f"Run date: `{RUN_DATE}`",
        "",
        "| Comparator | Decision | Rationale | Impacted Paper Section | Future Reopen Condition |",
        "|---|---|---|---|---|",
    ]
    lines.extend(
        (
            f"| {_cell(row.comparator)} | {_cell(row.decision)} | {_cell(row.rationale)} | "
            f"{_cell(row.impacted_paper_section)} | {_cell(row.future_reopen_condition)} |"
        )
        for row in decisions
    )
    lines.extend(
        [
            "",
            "## Source Basis",
            "",
            (
                "The comparator list was consolidated from `ops/known-issues.md`, "
                "`research-references.md`, and the relevant `docs/research-notes/` "
                "paper integration notes."
            ),
        ]
    )
    return "\n".join(lines) + "\n"


def apply_references_status_clarification(
    references_text: str,
    decisions: list[ComparatorDecision],
) -> tuple[str, bool]:
    if STATUS_CLARIFICATION_SENTINEL in references_text:
        return references_text, False
    lines = [
        "",
        "### Exp 1461 Comparator Audit Status Clarification",
        "",
        STATUS_CLARIFICATION_SENTINEL,
        "",
        "Status clarification only. No unrelated references were added.",
    ]
    lines.extend(
        f"- {row.comparator}: {REFERENCE_STATUS_PHRASES[row.decision]} - {row.rationale}"
        for row in decisions
    )
    return references_text.rstrip() + "\n\n" + "\n".join(lines) + "\n", True


def run(
    root: Path | str = REPO_ROOT,
    out_path: Path | str = DEFAULT_OUT_PATH,
    decision_table_path: Path | str = DEFAULT_TABLE_PATH,
    references_path: Path | str = DEFAULT_REFERENCES_PATH,
    decisions: list[ComparatorDecision] | None = None,
) -> dict[str, Any]:
    root_path = Path(root)
    out = Path(out_path)
    table = Path(decision_table_path)
    references = Path(references_path)
    rows = default_decisions() if decisions is None else list(decisions)

    write_in_progress_artifact(out)
    table.parent.mkdir(parents=True, exist_ok=True)
    table.write_text(render_decision_table(rows), encoding="utf-8")

    updated_references, references_updated = apply_references_status_clarification(
        references.read_text(encoding="utf-8"),
        rows,
    )
    references.write_text(updated_references, encoding="utf-8")

    artifact = build_artifact(
        decisions=rows,
        decision_table_path=_relative_path(table, root_path),
        references_updated=references_updated,
    )
    return _write_json(out, artifact)
