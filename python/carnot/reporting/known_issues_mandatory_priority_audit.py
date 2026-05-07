"""Audit active mandatory known-issues priorities for Exp 1455.

Spec: REQ-REPORT-041, SCENARIO-REPORT-041.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
PROJECT_ROOT_FOR_METADATA = "/home/ianblenke/github.com/ianblenke/carnot"
RUN_DATE = "20260507"
EXPERIMENT = "1455_known_issues_mandatory_priority_audit"
SCHEMA = "known_issues_mandatory_priority_audit_v1"

DEFAULT_OUT_PATH = (
    REPO_ROOT / "results" / "experiment_1455_known_issues_mandatory_priority_audit.json"
)
DEFAULT_KNOWN_ISSUES_PATH = REPO_ROOT / "ops" / "known-issues.md"
DEFAULT_AUDIT_PATH = REPO_ROOT / "ops" / "mandatory_priority_audit.md"
DEFAULT_INDEX_PATH = REPO_ROOT / "ops" / "active-priorities.md"
DEFAULT_SCOPE_MANIFEST_PATH = REPO_ROOT / "ops" / "milestone_112_scope_reduction_manifest.md"
DEFAULT_SIGNAL_SUMMARY_PATH = REPO_ROOT / "ops" / "experiment_signal_noise_summary.md"
DEFAULT_ROADMAP_PROPOSAL_PATH = (
    REPO_ROOT / "openspec" / "change-proposals" / "research-roadmap-vNEXT.md"
)

REQUIRED_ARTIFACT_FIELDS = {
    "status",
    "initial_priority_count",
    "active_priority_count",
    "trim_fraction",
    "priority_audit_path",
    "known_issues_updated",
    "active_priorities_index_path",
    "retired_or_consolidated_priorities",
    "honest_verdict",
}

TERMINAL_OR_CONSOLIDATED_STATUSES = {"consolidate", "superseded", "retire"}


@dataclass(frozen=True)
class PriorityEntry:
    line_number: int
    marker: str
    title: str


@dataclass(frozen=True)
class PriorityDecision:
    status: str
    active_priority_id: str | None
    rationale: str


@dataclass(frozen=True)
class ActivePriority:
    priority_id: str
    title: str
    source_entries: list[str]
    next_action: str


DEFAULT_ACTIVE_PRIORITIES = [
    ActivePriority(
        priority_id="scope_reduction_execution",
        title="Scope-reduction execution",
        source_entries=["SCOPE REDUCTION MILESTONE"],
        next_action=(
            "Finish the .112 scope-reduction tasks and block new variant expansion until "
            "the signal/noise, priority, lineage, claim, hardware, and comparator audits land."
        ),
    ),
    ActivePriority(
        priority_id="repair_runtime_and_validation_context_gate",
        title="Repair runtime and validation-context gate",
        source_entries=["Repair-Loop Validation-Error-as-Context Fix"],
        next_action=(
            "Repair the local SOTA GGUF runtime first, then run only the validation-error-as-context "
            "salvage test before preserving or retiring the repair-executor lineage."
        ),
    ),
    ActivePriority(
        priority_id="paper_integrity_and_claim_narrowing",
        title="Paper integrity and claim narrowing",
        source_entries=["Paper Integrity Audit", "Paper-v6 Related Work Overhaul"],
        next_action=(
            "Keep publication hold active until critical figure, hardware-claim, related-work, "
            "and anchored-claim issues are reconciled to measured artifacts."
        ),
    ),
    ActivePriority(
        priority_id="verifier_orthogonality_and_phase_gates",
        title="Verifier orthogonality and phase gates",
        source_entries=[
            "Verifier Joint-Orthogonality Audit",
            "Phase Prototype + Empirical Validation + Adversarial Check Discipline",
        ],
        next_action=(
            "Measure verifier joint overlap before k-count claims or scale-up, and keep each "
            "phase behind prototype, empirical, and adversarial gates."
        ),
    ),
    ActivePriority(
        priority_id="planning_and_artifact_lifecycle_hygiene",
        title="Planning and artifact lifecycle hygiene",
        source_entries=[
            "Failure-Ledger v2 + Planner Discipline",
            "artifact_not_updated_past_bootstrap Pattern",
            "Auto-Populate prior_failures",
        ],
        next_action=(
            "Treat prior-failure coverage, STEP-0 artifacts, and terminal artifact finalization "
            "as one operational hygiene lane rather than separate mandatory priorities."
        ),
    ),
    ActivePriority(
        priority_id="test_memory_safety_guardrails",
        title="Test memory safety guardrails",
        source_entries=[
            "Watchdog Insufficient for Single-Test Catastrophic Load",
            "Pytest Worker Memory Watchdog",
        ],
        next_action=(
            "Keep the pytest RSS watchdog and RLIMIT_AS cap active; do not create new memory "
            "watchdog lineages unless fresh failures bypass both guards."
        ),
    ),
    ActivePriority(
        priority_id="hardware_portfolio_narrowing",
        title="Hardware portfolio narrowing",
        source_entries=["Phase-2 Hardware Story Re-Scope"],
        next_action=(
            "Carry the FPGA proof-of-concept caveats into the .112 hardware portfolio narrowing "
            "and cap active production hardware tracks."
        ),
    ),
]


DEFAULT_DECISION_RULES = [
    (
        "repair-loop validation-error-as-context",
        PriorityDecision(
            status="keep",
            active_priority_id="repair_runtime_and_validation_context_gate",
            rationale="Still valid as the single scoped repair-executor salvage gate after local SOTA runtime is fixed.",
        ),
    ),
    (
        "scope reduction milestone",
        PriorityDecision(
            status="keep",
            active_priority_id="scope_reduction_execution",
            rationale="Controlling .112 governance directive; keep active until the scope-reduction milestone closes.",
        ),
    ),
    (
        "trace2skill + skillify",
        PriorityDecision(
            status="parked",
            active_priority_id=None,
            rationale="Useful operational follow-up, but explicitly deferred until active scope reduction settles what stays.",
        ),
    ),
    (
        "larql decoupled-attention",
        PriorityDecision(
            status="parked",
            active_priority_id=None,
            rationale="Strategic substrate work, but parked behind hardware narrowing and comparator cite/retire decisions.",
        ),
    ),
    (
        "verifier joint-orthogonality",
        PriorityDecision(
            status="keep",
            active_priority_id="verifier_orthogonality_and_phase_gates",
            rationale="Still publication-blocking and scale-up-blocking for any k-verifier headline claim.",
        ),
    ),
    (
        "paper-v6 related work",
        PriorityDecision(
            status="consolidate",
            active_priority_id="paper_integrity_and_claim_narrowing",
            rationale="Fold into the paper integrity and anchored-claims narrowing lane instead of tracking separately.",
        ),
    ),
    (
        "nrgpt frozen-prefix",
        PriorityDecision(
            status="parked",
            active_priority_id=None,
            rationale="The entry labels itself optional; it should not consume the active mandatory priority budget.",
        ),
    ),
    (
        "pre-commit",
        PriorityDecision(
            status="superseded",
            active_priority_id="planning_and_artifact_lifecycle_hygiene",
            rationale="Superseded by Exp 1216 and the batching-check exemption/fail-forward hook changes.",
        ),
    ),
    (
        "phase-5 intermediate-scale",
        PriorityDecision(
            status="parked",
            active_priority_id=None,
            rationale="Valid future scale-up risk, but parked until the .112 scope and paper claim set are smaller.",
        ),
    ),
    (
        "retro task boundary",
        PriorityDecision(
            status="superseded",
            active_priority_id="planning_and_artifact_lifecycle_hygiene",
            rationale="Superseded by the Exp 1215 retro STEP-0 and max-turns pattern.",
        ),
    ),
    (
        "auto-populate prior_failures",
        PriorityDecision(
            status="superseded",
            active_priority_id="planning_and_artifact_lifecycle_hygiene",
            rationale="Superseded by the shipped conductor_priors_autofill workflow and its tests.",
        ),
    ),
    (
        "artifact_not_updated_past_bootstrap pattern",
        PriorityDecision(
            status="consolidate",
            active_priority_id="planning_and_artifact_lifecycle_hygiene",
            rationale="Keep the issue only as part of the broader artifact lifecycle hygiene lane.",
        ),
    ),
    (
        "watchdog insufficient",
        PriorityDecision(
            status="consolidate",
            active_priority_id="test_memory_safety_guardrails",
            rationale="Consolidate with the pytest memory watchdog and RLIMIT_AS guardrail status.",
        ),
    ),
    (
        "pytest worker memory watchdog",
        PriorityDecision(
            status="consolidate",
            active_priority_id="test_memory_safety_guardrails",
            rationale="Consolidate into one test memory safety guardrail instead of two mandatory entries.",
        ),
    ),
    (
        "grpo v5 routing bug",
        PriorityDecision(
            status="retire",
            active_priority_id=None,
            rationale="Retire as a standalone priority; GRPO/VPRM lineages are under .112 consolidation/retirement.",
        ),
    ),
    (
        "paper integrity audit",
        PriorityDecision(
            status="keep",
            active_priority_id="paper_integrity_and_claim_narrowing",
            rationale="Still active because publication remains blocked until the critical evidence issues close.",
        ),
    ),
    (
        "seed iq verified",
        PriorityDecision(
            status="parked",
            active_priority_id=None,
            rationale="Strategic context is preserved, but active-inference expansion is parked during scope reduction.",
        ),
    ),
    (
        "ebt/arc-agi-3",
        PriorityDecision(
            status="parked",
            active_priority_id=None,
            rationale="Paradigm-shift ideas remain research context, not current active mandatory work.",
        ),
    ),
    (
        "phase-3 thinking-mode",
        PriorityDecision(
            status="parked",
            active_priority_id=None,
            rationale="Inference-mode expansion is parked until scope reduction and current paper claims are narrowed.",
        ),
    ),
    (
        "failure-ledger v2",
        PriorityDecision(
            status="consolidate",
            active_priority_id="planning_and_artifact_lifecycle_hygiene",
            rationale="Partly shipped and now tracked as planning/artifact lifecycle hygiene rather than five separate fixes.",
        ),
    ),
    (
        "llm failure exemplar corpus",
        PriorityDecision(
            status="parked",
            active_priority_id=None,
            rationale="Useful benchmark curation, but not mandatory while scope-reduction tasks are closing.",
        ),
    ),
    (
        "carry-forward from .85",
        PriorityDecision(
            status="superseded",
            active_priority_id="planning_and_artifact_lifecycle_hygiene",
            rationale="Historical carry-forward batch; later milestones either executed or reclassified these tasks.",
        ),
    ),
    (
        "phase prototype + empirical validation",
        PriorityDecision(
            status="keep",
            active_priority_id="verifier_orthogonality_and_phase_gates",
            rationale="Still active as the project-wide gate against architecture-heavy, evidence-light expansion.",
        ),
    ),
    (
        "phase-2 hardware story re-scope",
        PriorityDecision(
            status="consolidate",
            active_priority_id="hardware_portfolio_narrowing",
            rationale="Consolidate into the .112 hardware portfolio narrowing decision.",
        ),
    ),
]


def _relative_path(path: Path | str, root: Path | str = REPO_ROOT) -> str:
    path = Path(path)
    root = Path(root)
    try:
        return str(path.relative_to(root))
    except ValueError:
        parts = path.parts
        for anchor in ("ops", "results", "openspec"):
            if anchor in parts:
                return str(Path(*parts[parts.index(anchor) :]))
        return path.name


def _write_json(path: Path | str, payload: dict[str, Any]) -> dict[str, Any]:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def write_in_progress_artifact(
    out_path: Path | str = DEFAULT_OUT_PATH,
    *,
    root: Path | str = REPO_ROOT,
    audit_path: Path | str = DEFAULT_AUDIT_PATH,
    index_path: Path | str = DEFAULT_INDEX_PATH,
) -> dict[str, Any]:
    """REQ-REPORT-041: seed the required artifact before auditing."""

    artifact = {
        "experiment": EXPERIMENT,
        "schema": SCHEMA,
        "spec": ["REQ-REPORT-041", "SCENARIO-REPORT-041"],
        "run_date": RUN_DATE,
        "project_root": PROJECT_ROOT_FOR_METADATA,
        "status": "in_progress",
        "initial_priority_count": 0,
        "active_priority_count": 0,
        "trim_fraction": 0.0,
        "priority_audit_path": _relative_path(audit_path, root),
        "known_issues_updated": False,
        "active_priorities_index_path": _relative_path(index_path, root),
        "retired_or_consolidated_priorities": [],
        "honest_verdict": "in_progress",
    }
    return _write_json(out_path, artifact)


def _normalize_title(value: str) -> str:
    normalized = (
        value.lower().replace("—", "-").replace("–", "-").replace("`", "").replace("  ", " ")
    )
    return " ".join(normalized.split())


def _split_marker_and_title(raw_heading: str) -> tuple[str, str]:
    if raw_heading.startswith("Carry-forward"):
        return "Carry-forward", raw_heading
    if ": " in raw_heading:
        marker, title = raw_heading.split(": ", 1)
        return marker.strip(), title.strip()
    return raw_heading, raw_heading


def _current_mandatory_block_lines(markdown: str) -> list[tuple[int, str]]:
    lines = markdown.splitlines()
    start_index = None
    for index, line in enumerate(lines):
        if line.startswith("## MANDATORY-NEXT-MILESTONE PRIORITIES"):
            start_index = index + 1
            break
    if start_index is None:
        return []
    end_index = len(lines)
    for index in range(start_index, len(lines)):
        if lines[index].startswith("## MANDATORY-NEXT-MILESTONE PRIORITIES"):
            end_index = index
            break
    return [(index + 1, lines[index]) for index in range(start_index, end_index)]


def extract_active_priority_entries(markdown: str) -> list[PriorityEntry]:
    """REQ-REPORT-041: parse one row per current active priority heading."""

    entries: list[PriorityEntry] = []
    for line_number, line in _current_mandatory_block_lines(markdown):
        if not line.startswith("### "):
            continue
        raw_heading = line.removeprefix("### ").strip()
        if not (
            raw_heading.startswith("NEW ")
            or raw_heading.startswith("REVISED ")
            or raw_heading.startswith("Carry-forward")
        ):
            continue
        marker, title = _split_marker_and_title(raw_heading)
        entries.append(PriorityEntry(line_number=line_number, marker=marker, title=title))
    return entries


def _normalize_decisions(decisions: dict[str, PriorityDecision]) -> dict[str, PriorityDecision]:
    return {_normalize_title(key): value for key, value in decisions.items()}


def _default_decision(entry: PriorityEntry) -> PriorityDecision:
    normalized = _normalize_title(entry.title)
    for phrase, decision in DEFAULT_DECISION_RULES:
        if phrase in normalized:
            return decision
    return PriorityDecision(
        status="parked",
        active_priority_id=None,
        rationale="Not mapped to a current .112 active priority; preserve history and park pending operator review.",
    )


def _decide_priority(
    entry: PriorityEntry,
    decisions: dict[str, PriorityDecision],
) -> PriorityDecision:
    return decisions.get(_normalize_title(entry.title), _default_decision(entry))


def _audit_rows(
    entries: list[PriorityEntry],
    decisions: dict[str, PriorityDecision],
) -> list[dict[str, str | int | None]]:
    rows: list[dict[str, str | int | None]] = []
    for number, entry in enumerate(entries, start=1):
        decision = _decide_priority(entry, decisions)
        rows.append(
            {
                "number": number,
                "line": entry.line_number,
                "marker": entry.marker,
                "priority": entry.title,
                "status": decision.status,
                "active_priority_id": decision.active_priority_id,
                "rationale": decision.rationale,
            }
        )
    return rows


def _markdown_table(rows: list[dict[str, str | int | None]]) -> list[str]:
    table = [
        "| # | source line | marker | priority | status | active index | rationale |",
        "|---|---:|---|---|---|---|---|",
    ]
    for row in rows:
        active_id = row["active_priority_id"] or "-"
        table.append(
            "| {number} | {line} | {marker} | {priority} | {status} | {active_id} | {rationale} |".format(
                active_id=active_id,
                **row,
            )
        )
    return table


def _write_audit(path: Path | str, rows: list[dict[str, str | int | None]]) -> None:
    lines = [
        "# Mandatory Priority Audit",
        "",
        f"Run date: `{RUN_DATE}`",
        "",
        "Statuses: `keep`, `consolidate`, `superseded`, `parked`, `retire`.",
        "",
        *_markdown_table(rows),
        "",
    ]
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")


def _write_index(path: Path | str, active_priorities: list[ActivePriority]) -> None:
    lines = [
        "# Active Priorities",
        "",
        f"Run date: `{RUN_DATE}`",
        "",
        f"Active priority count: `{len(active_priorities)}`",
        "",
    ]
    for number, priority in enumerate(active_priorities, start=1):
        lines.extend(
            [
                f"## {number}. {priority.title}",
                "",
                f"- Active index id: `{priority.priority_id}`",
                f"- Source entries: {', '.join(priority.source_entries)}",
                f"- Next action: {priority.next_action}",
                "",
            ]
        )
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")


def _known_issues_section(
    initial_count: int,
    active_count: int,
    trim_fraction: float,
    audit_path: str,
    index_path: str,
) -> str:
    return "\n".join(
        [
            "## CURRENT ACTIVE PRIORITIES (20260507 audit)",
            "",
            f"- Initial active mandatory priority entries audited: `{initial_count}`",
            f"- Current active priority index count: `{active_count}`",
            f"- Trim fraction: `{trim_fraction:.4f}`",
            f"- Audit table: `{audit_path}`",
            f"- Active index: `{index_path}`",
            "- Historical entries below are preserved for audit trail; superseded, parked, consolidated, and retired statuses live in the audit table.",
            "",
        ]
    )


def _update_known_issues(
    path: Path | str,
    section: str,
) -> bool:
    known_path = Path(path)
    text = known_path.read_text(encoding="utf-8")
    marker = "## CURRENT ACTIVE PRIORITIES (20260507 audit)"
    if marker in text:
        before, remainder = text.split(marker, 1)
        next_section = remainder.find("\n## ")
        if next_section == -1:
            updated = before.rstrip() + "\n\n" + section
        else:
            updated = before.rstrip() + "\n\n" + section + remainder[next_section + 1 :]
    else:
        insert_after = text.find("\n## ")
        if insert_after == -1:
            updated = text.rstrip() + "\n\n" + section
        else:
            updated = text[:insert_after].rstrip() + "\n\n" + section + text[insert_after:]
    changed = updated != text
    if changed:
        known_path.write_text(updated, encoding="utf-8")
    return changed


def _read_context_paths(paths: list[Path]) -> list[str]:
    observed: list[str] = []
    for path in paths:
        if path.exists():
            path.read_text(encoding="utf-8")
            observed.append(_relative_path(path))
    return observed


def _retired_or_consolidated(
    rows: list[dict[str, str | int | None]],
) -> list[dict[str, str | int | None]]:
    return [row for row in rows if str(row["status"]) in TERMINAL_OR_CONSOLIDATED_STATUSES]


def run(
    *,
    root: Path | str = REPO_ROOT,
    known_issues_path: Path | str = DEFAULT_KNOWN_ISSUES_PATH,
    audit_path: Path | str = DEFAULT_AUDIT_PATH,
    index_path: Path | str = DEFAULT_INDEX_PATH,
    out_path: Path | str = DEFAULT_OUT_PATH,
    scope_manifest_path: Path | str = DEFAULT_SCOPE_MANIFEST_PATH,
    signal_summary_path: Path | str = DEFAULT_SIGNAL_SUMMARY_PATH,
    roadmap_proposal_path: Path | str = DEFAULT_ROADMAP_PROPOSAL_PATH,
    decisions: dict[str, PriorityDecision] | None = None,
    active_priorities: list[ActivePriority] | None = None,
) -> dict[str, Any]:
    """Run the Exp 1455 audit and write markdown, known-issues, and JSON outputs."""

    root_path = Path(root)
    known_path = Path(known_issues_path)
    audit = Path(audit_path)
    index = Path(index_path)
    active = list(active_priorities or DEFAULT_ACTIVE_PRIORITIES)
    normalized_decisions = _normalize_decisions(decisions or {})

    write_in_progress_artifact(out_path, root=root_path, audit_path=audit, index_path=index)
    context_paths_read = _read_context_paths(
        [Path(scope_manifest_path), Path(signal_summary_path), Path(roadmap_proposal_path)]
    )
    known_text = known_path.read_text(encoding="utf-8")
    entries = extract_active_priority_entries(known_text)
    rows = _audit_rows(entries, normalized_decisions)
    initial_count = len(entries)
    active_count = len(active)
    trim_fraction = (
        round((initial_count - active_count) / initial_count, 4) if initial_count else 0.0
    )

    _write_audit(audit, rows)
    _write_index(index, active)
    audit_rel = _relative_path(audit, root_path)
    index_rel = _relative_path(index, root_path)
    known_issues_updated = _update_known_issues(
        known_path,
        _known_issues_section(initial_count, active_count, trim_fraction, audit_rel, index_rel),
    )
    terminal_rows = _retired_or_consolidated(rows)
    artifact: dict[str, Any] = {
        "experiment": EXPERIMENT,
        "schema": SCHEMA,
        "spec": ["REQ-REPORT-041", "SCENARIO-REPORT-041"],
        "run_date": RUN_DATE,
        "project_root": PROJECT_ROOT_FOR_METADATA,
        "status": "complete",
        "initial_priority_count": initial_count,
        "active_priority_count": active_count,
        "trim_fraction": trim_fraction,
        "priority_audit_path": audit_rel,
        "known_issues_updated": known_issues_updated,
        "active_priorities_index_path": index_rel,
        "retired_or_consolidated_priorities": terminal_rows,
        "parked_priorities": [row for row in rows if row["status"] == "parked"],
        "context_paths_read": context_paths_read,
        "honest_verdict": (
            f"complete_exp1455_known_issues_priorities_trimmed_from_{initial_count}_to_{active_count}"
        ),
    }
    return _write_json(out_path, artifact)


if __name__ == "__main__":  # pragma: no cover
    run()
