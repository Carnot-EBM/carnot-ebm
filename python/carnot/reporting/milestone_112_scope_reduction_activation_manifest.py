"""Build the Exp 1453 `.112` scope-reduction activation manifest.

Spec: REQ-REPORT-039, SCENARIO-REPORT-039.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
PROJECT_ROOT_FOR_METADATA = "/home/ianblenke/github.com/ianblenke/carnot"
RUN_DATE = "20260507"
PRIOR_MILESTONE = "2026.04.111"
TARGET_MILESTONE = "2026.04.112"
EXPERIMENT = "1453_112_scope_reduction_activation_manifest"
SCHEMA = "milestone_112_scope_reduction_activation_manifest_v1"

DEFAULT_OUT_PATH = (
    REPO_ROOT / "results" / "experiment_1453_112_scope_reduction_activation_manifest.json"
)
DEFAULT_MANIFEST_PATH = REPO_ROOT / "ops" / "milestone_112_scope_reduction_manifest.md"
RETRO_FILE = "experiment_1452_milestone_111_retro.json"
ROADMAP_PROPOSAL = "openspec/change-proposals/research-roadmap-vNEXT.md"

REQUIRED_SCOPE_REDUCTION_TASK_COUNT = 8
REQUIRED_ARTIFACT_FIELDS = {
    "status",
    "milestone",
    "prior_milestone",
    "scope_reduction_required",
    "required_scope_reduction_task_count",
    "planned_scope_reduction_task_count",
    "scope_reduction_manifest_path",
    "planned_scope_task_ids",
    "carryforward_from_111",
    "forbidden_exact_expansions",
    "honest_verdict",
}

MANDATORY_KNOWN_ISSUES_SCOPE_ITEMS = {
    "exp_NEXT_SCOPE_A": "Experiment artifact classifier",
    "exp_NEXT_SCOPE_B": "GRPO lineage consolidation + retirement",
    "exp_NEXT_SCOPE_C": "WOPR puzzle cartridge retirement",
    "exp_NEXT_SCOPE_D": "known-issues.md MANDATORY priority audit",
    "exp_NEXT_SCOPE_E": "Paper-v6 anchored-claims narrowing",
    "exp_NEXT_SCOPE_F": "Self-learning non-headline lineage decision",
    "exp_NEXT_SCOPE_G": "Hardware portfolio narrowing",
    "exp_NEXT_SCOPE_H": "Comparator-integration audit",
}

SCOPE_REQUIREMENTS = [
    {
        "requirement_id": "activation_manifest",
        "requirement": "Activation / compliance manifest",
        "task_id": "exp1453",
        "deliverable_path": (
            "results/experiment_1453_112_scope_reduction_activation_manifest.json; "
            "ops/milestone_112_scope_reduction_manifest.md"
        ),
        "acceptance_field": "exp1453.scope_reduction_manifest_complete",
        "retire_block_rule": (
            "Block .112 scope-compliance claims if any mandatory scope item lacks a "
            "mapped task, deliverable, acceptance field, or retire/block rule."
        ),
    },
    {
        "requirement_id": "artifact_classifier",
        "requirement": "Experiment artifact classifier",
        "task_id": "exp1454",
        "deliverable_path": "results/experiment_1454_scope_artifact_classifier.json",
        "acceptance_field": "exp1454.classification_table_written",
        "retire_block_rule": (
            "Classify SIGNAL / NOISE / AMBIGUOUS before adding more experiment variants."
        ),
    },
    {
        "requirement_id": "known_issues_priority_audit",
        "requirement": "known-issues.md MANDATORY priority audit",
        "task_id": "exp1455",
        "deliverable_path": "results/experiment_1455_known_issues_priority_audit.json",
        "acceptance_field": "exp1455.active_priority_count <= 10",
        "retire_block_rule": (
            "Block new mandatory-priority expansion until active priorities are trimmed "
            "by at least 40%."
        ),
    },
    {
        "requirement_id": "grpo_vprm_retirement",
        "requirement": "GRPO/VPRM lineage consolidation and retirement",
        "task_id": "exp1456",
        "deliverable_path": "results/experiment_1456_grpo_vprm_lineage_retirement.json",
        "acceptance_field": "exp1456.grpo_lineage_retired",
        "retire_block_rule": (
            "Block planner proposals for GRPO v15 unless a human override names a new "
            "root cause and falsifiable gate."
        ),
    },
    {
        "requirement_id": "wopr_puzzle_retirement",
        "requirement": "WOPR puzzle cartridge retirement",
        "task_id": "exp1457",
        "deliverable_path": "results/experiment_1457_wopr_puzzle_cartridge_retirement.json",
        "acceptance_field": "exp1457.wopr_puzzle_lineage_retired",
        "retire_block_rule": (
            "Block future puzzle cartridges that do not connect to the verify-repair "
            "thesis or Phase-3 substrate trajectory."
        ),
    },
    {
        "requirement_id": "hardnet_dsp_retirement",
        "requirement": "HardNet++/DSP repair stack consolidation",
        "task_id": "exp1458",
        "deliverable_path": "results/experiment_1458_hardnet_dsp_repair_stack_retirement.json",
        "acceptance_field": "exp1458.hardnet_dsp_lineage_retired",
        "retire_block_rule": (
            "Block new HardNet++/DSP variants; preserve the conservative-replay lesson "
            "in one consolidation artifact."
        ),
    },
    {
        "requirement_id": "self_learning_non_headline_decision",
        "requirement": "Self-learning `_improved_non_headline` lineage decision",
        "task_id": "exp1459",
        "deliverable_path": "results/experiment_1459_self_learning_non_headline_decision.json",
        "acceptance_field": (
            "exp1459.self_learning_headline_pivot_selected or "
            "exp1459.self_learning_lineage_retired"
        ),
        "retire_block_rule": (
            "Block another improved-non-headline variant unless it selects a headline "
            "pivot or retires the lineage."
        ),
    },
    {
        "requirement_id": "hardware_portfolio_narrowing",
        "requirement": "Hardware portfolio narrowing",
        "task_id": "exp1460",
        "deliverable_path": "results/experiment_1460_hardware_portfolio_narrowing.json",
        "acceptance_field": "exp1460.active_hardware_track_count <= 3",
        "retire_block_rule": (
            "Block broad new hardware branches until active hardware tracks are capped "
            "at three and out-of-scope tracks are documented."
        ),
    },
    {
        "requirement_id": "comparator_integration_audit",
        "requirement": "Comparator-integration audit",
        "task_id": "exp1461",
        "deliverable_path": "results/experiment_1461_comparator_integration_audit.json",
        "acceptance_field": "exp1461.comparator_decision_count >= 6",
        "retire_block_rule": (
            "Block broad new comparator branches until Abstract-CoT, Meta-Harness, "
            "Autodata, LARQL, Skillify, and GStack each receive cite/retire decisions."
        ),
    },
    {
        "requirement_id": "paper_v6_claim_narrowing",
        "requirement": "Paper-v6 anchored-claims narrowing",
        "task_id": "exp1462",
        "deliverable_path": "results/experiment_1462_paper_v6_anchored_claims_narrowing.json",
        "acceptance_field": "3 <= exp1462.anchored_claim_count <= 5",
        "retire_block_rule": (
            "Block paper claim expansion until each retained claim has artifact evidence "
            "and unsupported territory is moved to appendix or future work."
        ),
    },
]

FORBIDDEN_EXACT_EXPANSIONS = [
    {
        "forbidden_scope_id": "grpo_v15",
        "forbidden_scope": "new GRPO v15 or GRPO/VPRM variant expansion during .112",
        "blocked_until": "exp1456.grpo_lineage_retired=true or explicit human override",
        "retire_or_block_rule": "Block as exact noise-line expansion.",
    },
    {
        "forbidden_scope_id": "wopr_puzzle_cartridges",
        "forbidden_scope": "new WOPR puzzle cartridges during .112",
        "blocked_until": "exp1457.wopr_puzzle_lineage_retired=true",
        "retire_or_block_rule": "Block future cartridges unless they name a thesis link.",
    },
    {
        "forbidden_scope_id": "hardnet_dsp_variants",
        "forbidden_scope": "new HardNet++/DSP variants during .112",
        "blocked_until": "exp1458.hardnet_dsp_lineage_retired=true",
        "retire_or_block_rule": "Block variant expansion and consolidate the lesson first.",
    },
    {
        "forbidden_scope_id": "broad_comparator_hardware_branches",
        "forbidden_scope": "broad new comparator or hardware branches during .112",
        "blocked_until": "exp1460 and exp1461 narrow active hardware and comparator scope",
        "retire_or_block_rule": "Block branch expansion until cite/retire and active-track decisions land.",
    },
    {
        "forbidden_scope_id": "self_learning_non_headline_variants",
        "forbidden_scope": "new self-learning `_improved_non_headline` variants during .112",
        "blocked_until": "exp1459 chooses a headline pivot or retires the lineage",
        "retire_or_block_rule": "Block more non-headline suffix churn without a decision.",
    },
]


def _write_json(path: Path, payload: Mapping[str, Any]) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    artifact = dict(payload)
    path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return artifact


def write_in_progress_artifact(out_path: Path | str = DEFAULT_OUT_PATH) -> dict[str, Any]:
    """REQ-REPORT-039: record the bootstrap state before terminal scoring."""

    artifact: dict[str, Any] = {field: None for field in REQUIRED_ARTIFACT_FIELDS}
    artifact.update(
        {
            "experiment": EXPERIMENT,
            "schema": SCHEMA,
            "run_date": RUN_DATE,
            "project_root": PROJECT_ROOT_FOR_METADATA,
            "status": "in_progress",
            "milestone": TARGET_MILESTONE,
            "prior_milestone": PRIOR_MILESTONE,
            "scope_reduction_required": True,
            "required_scope_reduction_task_count": REQUIRED_SCOPE_REDUCTION_TASK_COUNT,
            "planned_scope_reduction_task_count": 0,
            "scope_reduction_manifest_path": _relative_path(DEFAULT_MANIFEST_PATH),
            "planned_scope_task_ids": [],
            "carryforward_from_111": [],
            "forbidden_exact_expansions": [],
            "honest_verdict": "in_progress_scope_reduction_activation_manifest_seeded",
        }
    )
    return _write_json(Path(out_path), artifact)


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _read_text(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8")


def _relative_path(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        parts = path.parts
        if "ops" in parts:
            return str(Path(*parts[parts.index("ops") :]))
        if "results" in parts:
            return str(Path(*parts[parts.index("results") :]))
        return path.name


def _scope_reduction_required(known_issues_text: str) -> bool:
    text = known_issues_text.lower()
    return "scope reduction" in text and "at least 8" in text


def _missing_known_issue_items(known_issues_text: str) -> list[str]:
    return [
        item_id
        for item_id in MANDATORY_KNOWN_ISSUES_SCOPE_ITEMS
        if item_id not in known_issues_text
    ]


def _normalize_prior_failures(track: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "experiment_id": str(prior.get("experiment_id", "missing_experiment_id")),
            "verdict": str(prior.get("verdict", "missing_prior_verdict")),
            "evidence_path": prior.get("evidence_path"),
        }
        for prior in track.get("prior_failures", [])
        if isinstance(prior, Mapping)
    ]


def _carryforward_from_111(retro: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "id": str(track.get("id", "missing_track_id")),
            "title": str(track.get("title", "missing title")),
            "next_rule": str(track.get("next_rule", "missing next_rule")),
            "prior_failures": _normalize_prior_failures(track),
            "retire_if_same_verdict": bool(track.get("retire_if_same_verdict")),
        }
        for track in retro.get("carry_forward_tracks", [])
        if isinstance(track, Mapping)
    ]


def _live_sota_runtime_rules(carryforward: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rules = []
    for track in carryforward:
        searchable = f"{track['id']} {track['title']} {track['next_rule']}".lower()
        if (
            "live_sota" in searchable
            or "live-sota" in searchable
            or "live-runtime" in searchable
            or "repair_v3" in searchable
            or "repair-v3" in searchable
            or "repair v3" in searchable
        ):
            rules.append(track)
    return rules


def _planned_scope_task_ids() -> list[str]:
    return [str(row["task_id"]) for row in SCOPE_REQUIREMENTS]


def _missing_mapped_task_ids(roadmap_text: str, roadmap_next_text: str) -> list[str]:
    combined = f"{roadmap_text}\n{roadmap_next_text}"
    return [task_id for task_id in _planned_scope_task_ids() if task_id not in combined]


def _source_inputs_read(
    *,
    retro: Mapping[str, Any],
    known_issues_text: str,
    roadmap_text: str,
    roadmap_next_text: str,
    exclusion_manifest_text: str,
) -> dict[str, dict[str, bool]]:
    return {
        f"results/{RETRO_FILE}": {"exists": bool(retro)},
        "ops/known-issues.md": {"exists": bool(known_issues_text)},
        ROADMAP_PROPOSAL: {"exists": bool(roadmap_text)},
        "research-roadmap-next.yaml": {"exists": bool(roadmap_next_text)},
        "ops/exclusion_manifest.yaml": {"exists": bool(exclusion_manifest_text)},
    }


def _md_cell(value: object) -> str:
    return str(value).replace("\n", " ").replace("|", "\\|")


def _render_scope_table(rows: list[dict[str, str]]) -> list[str]:
    lines = [
        "| requirement | mapped task id | deliverable path | acceptance field | retire/block rule |",
        "|---|---|---|---|---|",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    _md_cell(row["requirement"]),
                    _md_cell(row["task_id"]),
                    _md_cell(row["deliverable_path"]),
                    _md_cell(row["acceptance_field"]),
                    _md_cell(row["retire_block_rule"]),
                ]
            )
            + " |"
        )
    return lines


def _render_live_sota_rules(rules: list[dict[str, Any]]) -> list[str]:
    lines = ["", "## Live-SOTA Runtime Carry-Forward Rules", ""]
    if not rules:
        return lines + ["No live-SOTA runtime carry-forward rules were found."]
    for rule in rules:
        failures = ", ".join(
            f"{failure['experiment_id']}={failure['verdict']}"
            for failure in rule["prior_failures"]
        )
        lines.append(
            f"- {rule['id']}: {rule['next_rule']} Prior failures: {failures}. "
            f"retire_if_same_verdict={rule['retire_if_same_verdict']}."
        )
    return lines


def render_manifest(
    *,
    rows: list[dict[str, str]],
    live_sota_rules: list[dict[str, Any]],
    forbidden: list[dict[str, str]],
    blocked_reasons: list[str],
) -> str:
    """REQ-REPORT-039: render the operator-facing `.112` activation table."""

    lines = [
        "# Milestone .112 Scope-Reduction Manifest",
        "",
        f"Prior milestone: `{PRIOR_MILESTONE}`",
        f"Target milestone: `{TARGET_MILESTONE}`",
        f"Run date: `{RUN_DATE}`",
        "",
    ]
    if blocked_reasons:
        lines.append(f"Manifest is blocked: {'; '.join(blocked_reasons)}")
        lines.append("")
    lines.extend(_render_scope_table(rows))
    lines.extend(_render_live_sota_rules(live_sota_rules))
    lines.extend(["", "## Forbidden Exact Expansions", ""])
    for item in forbidden:
        lines.append(
            f"- {item['forbidden_scope_id']}: {item['forbidden_scope']}. "
            f"Blocked until: {item['blocked_until']}. Rule: {item['retire_or_block_rule']}"
        )
    lines.extend(
        [
            "",
            "## No-Change Confirmation",
            "",
            "- scripts/research_conductor.py: unchanged_by_exp1453_activation_workflow",
            "- research-roadmap.yaml: unchanged_by_exp1453_activation_workflow",
            "",
        ]
    )
    return "\n".join(lines)


def build_artifact(
    *,
    retro: Mapping[str, Any],
    known_issues_text: str,
    roadmap_text: str,
    roadmap_next_text: str,
    exclusion_manifest_text: str,
    manifest_path: str,
) -> tuple[dict[str, Any], str]:
    """REQ-REPORT-039: map mandatory `.112` scope-reduction work into tasks."""

    rows = [dict(row) for row in SCOPE_REQUIREMENTS]
    carryforward = _carryforward_from_111(retro)
    live_sota_rules = _live_sota_runtime_rules(carryforward)
    missing_known_items = _missing_known_issue_items(known_issues_text)
    missing_task_ids = _missing_mapped_task_ids(roadmap_text, roadmap_next_text)
    scope_required = _scope_reduction_required(known_issues_text)
    blocked_reasons: list[str] = []
    if not scope_required:
        blocked_reasons.append("known-issues scope-reduction directive not found")
    if missing_known_items:
        blocked_reasons.append("missing known-issues mandatory items: " + ", ".join(missing_known_items))
    if missing_task_ids:
        blocked_reasons.append("missing mapped task ids in roadmap: " + ", ".join(missing_task_ids))
    if not live_sota_rules:
        blocked_reasons.append("missing live-SOTA runtime carry-forward rules")

    complete = (
        not blocked_reasons
        and len(rows) >= REQUIRED_SCOPE_REDUCTION_TASK_COUNT
        and len(FORBIDDEN_EXACT_EXPANSIONS) >= 4
    )
    manifest = render_manifest(
        rows=rows,
        live_sota_rules=live_sota_rules,
        forbidden=FORBIDDEN_EXACT_EXPANSIONS,
        blocked_reasons=blocked_reasons,
    )
    artifact: dict[str, Any] = {
        "experiment": EXPERIMENT,
        "schema": SCHEMA,
        "run_date": RUN_DATE,
        "project_root": PROJECT_ROOT_FOR_METADATA,
        "status": "complete" if complete else "blocked",
        "milestone": TARGET_MILESTONE,
        "prior_milestone": str(retro.get("milestone", PRIOR_MILESTONE)),
        "prior_milestone_honest_verdict": retro.get("honest_verdict"),
        "scope_reduction_required": scope_required,
        "required_scope_reduction_task_count": REQUIRED_SCOPE_REDUCTION_TASK_COUNT,
        "planned_scope_reduction_task_count": len(rows),
        "scope_reduction_manifest_path": manifest_path,
        "scope_reduction_manifest_complete": complete,
        "planned_scope_task_ids": _planned_scope_task_ids(),
        "scope_manifest_rows": rows,
        "carryforward_from_111": carryforward,
        "live_sota_runtime_carryforward_rules": live_sota_rules,
        "forbidden_exact_expansions": list(FORBIDDEN_EXACT_EXPANSIONS),
        "missing_known_issues_scope_items": missing_known_items,
        "missing_mapped_task_ids_in_roadmap": missing_task_ids,
        "blocked_reasons": blocked_reasons,
        "source_inputs_read": _source_inputs_read(
            retro=retro,
            known_issues_text=known_issues_text,
            roadmap_text=roadmap_text,
            roadmap_next_text=roadmap_next_text,
            exclusion_manifest_text=exclusion_manifest_text,
        ),
        "roadmap_mapping_check": {
            "mapped_scope_task_count": len(rows),
            "missing_mapped_task_ids_in_roadmap": missing_task_ids,
        },
        "no_change_confirmations": {
            "scripts/research_conductor.py": "unchanged_by_exp1453_activation_workflow",
            "research-roadmap.yaml": "unchanged_by_exp1453_activation_workflow",
        },
        "honest_verdict": (
            "milestone_112_scope_reduction_activation_complete_10_scope_tasks_"
            "live_sota_carryforward_and_exact_noise_expansion_forbidden"
            if complete
            else "milestone_112_scope_reduction_activation_blocked_missing_directive_mapping_or_carryforward"
        ),
    }
    return artifact, manifest


def run(
    root: Path | str = REPO_ROOT,
    out_path: Path | str = DEFAULT_OUT_PATH,
    manifest_path: Path | str = DEFAULT_MANIFEST_PATH,
) -> dict[str, Any]:
    """REQ-REPORT-039: write bootstrap, markdown manifest, and terminal JSON."""

    root_path = Path(root)
    out = Path(out_path)
    manifest_out = Path(manifest_path)
    write_in_progress_artifact(out)
    retro = _read_json(root_path / "results" / RETRO_FILE) or {"milestone": PRIOR_MILESTONE}
    artifact, manifest = build_artifact(
        retro=retro,
        known_issues_text=_read_text(root_path / "ops" / "known-issues.md"),
        roadmap_text="\n".join(
            [
                _read_text(root_path / ROADMAP_PROPOSAL),
                _read_text(root_path / "research-roadmap.yaml"),
            ]
        ),
        roadmap_next_text=_read_text(root_path / "research-roadmap-next.yaml"),
        exclusion_manifest_text=_read_text(root_path / "ops" / "exclusion_manifest.yaml"),
        manifest_path=_relative_path(manifest_out),
    )
    manifest_out.parent.mkdir(parents=True, exist_ok=True)
    manifest_out.write_text(manifest, encoding="utf-8")
    return _write_json(out, artifact)


if __name__ == "__main__":  # pragma: no cover - thin CLI convenience
    run()
