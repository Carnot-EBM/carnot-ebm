"""Build the Exp 1467 `.113` activation manifest.

Spec: REQ-REPORT-048, SCENARIO-REPORT-048.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
PROJECT_ROOT_FOR_METADATA = "/home/ianblenke/github.com/ianblenke/carnot"
RUN_DATE = "20260507"
PREDECESSOR_MILESTONE = "2026.04.112"
TARGET_MILESTONE = "2026.04.113"
EXPERIMENT = "1467_112_completion_archive_113_activation"
SCHEMA = "milestone_113_activation_manifest_v1"

DEFAULT_OUT_PATH = (
    REPO_ROOT / "results" / "experiment_1467_112_completion_archive_113_activation.json"
)
DEFAULT_MANIFEST_PATH = REPO_ROOT / "ops" / "milestone_113_activation_manifest.md"
PREDECESSOR_RETRO_FILE = "experiment_1466_milestone_112_retro.json"

REQUIRED_ARTIFACT_FIELDS = {
    "status",
    "milestone",
    "predecessor_milestone",
    "criteria_met",
    "criteria_total",
    "research_complete_has_112_entry",
    "activation_manifest_complete",
    "retired_lineages_preserved",
    "allowed_113_tracks",
    "forbidden_reopen_tracks",
    "honest_verdict",
}

ALLOWED_113_TRACKS = [
    {
        "track": "live_sota_telemetry",
        "name": "Live SOTA Telemetry",
        "guardrail": "Use the repaired local GGUF runtime from Exp 1463; measure telemetry instead of reopening runtime repair.",
    },
    {
        "track": "beaver_lite_bounds",
        "name": "BEAVER-lite Bounds",
        "guardrail": "Run exactly the minimal deterministic-bound smoke selected by Exp 1465.",
    },
    {
        "track": "one_self_learning_pivot",
        "name": "One Self-Learning Pivot",
        "guardrail": "Only the Exp 1447-style verified-memory-growth pivot selected by Exp 1459 is allowed.",
    },
    {
        "track": "tskm_static_smokes",
        "name": "T-SKM/STATIC Smokes",
        "guardrail": "Keep these as bounded constraint-projection and CSR automaton smokes.",
    },
    {
        "track": "kv260_rtl_regression",
        "name": "KV260 RTL Regression",
        "guardrail": "Source-level RTL lint/simulation only; no board, latency, or deployment claim.",
    },
    {
        "track": "thrml_simulation",
        "name": "THRML Simulation",
        "guardrail": "Simulator parity only; no Extropic or TSU hardware execution claim.",
    },
]

FORBIDDEN_REOPEN_TRACKS = [
    {
        "track": "grpo_vprm",
        "name": "GRPO/VPRM",
        "source": "Exp 1456 retirement",
        "rule": "Do not reopen GRPO/VPRM variants unless operator-reopened with a new root cause and falsifiable gate.",
    },
    {
        "track": "wopr_puzzle_cartridges",
        "name": "WOPR Puzzle Cartridges",
        "source": "Exp 1457 retirement",
        "rule": "Do not add new game/gallery cartridges unless operator-reopened with a thesis or substrate link.",
    },
    {
        "track": "hardnet_dsp",
        "name": "HardNet++/DSP",
        "source": "Exp 1458 retirement",
        "rule": "Do not add HardNet++/DSP variants unless operator-reopened with non-replay evidence.",
    },
    {
        "track": "validation_error_repair",
        "name": "Validation-Error Repair",
        "source": "Exp 1464 retirement",
        "rule": "Do not revive validation-error-as-context repair unless operator-reopened and acceptance_delta_pp beats zero.",
    },
    {
        "track": "broad_vnn_comp_runners",
        "name": "Broad VNN-COMP Runners",
        "source": "Exp 1465 deferral",
        "rule": "Do not build broad VNNLIB/VNN-COMP runners before the BEAVER-lite smoke earns expansion.",
    },
    {
        "track": "hardware_execution_claims",
        "name": "Hardware Execution Claims",
        "source": "Exp 1460 portfolio narrowing",
        "rule": "Do not claim board, photonic, TSU, D-Wave, NPU, or large-FPGA execution unless operator-reopened with live evidence.",
    },
]

_EXP_LOG_NEEDLES = {
    "exp1453": ".112 Scope-Reduction Activation Manifest",
    "exp1454": "Experiment Artifact Signal/Noise Classifier",
    "exp1455": "Known-Issues Mandatory Priority Audit",
    "exp1456": "GRPO/VPRM Lineage Consolidation + Retirement",
    "exp1457": "WOPR Puzzle Cartridge Retirement",
    "exp1458": "HardNet++/DSP Repair Stack Consolidation",
    "exp1459": "Self-Learning Non-Headline Lineage Decision",
    "exp1460": "Hardware Portfolio Narrowing",
    "exp1461": "Comparator Integration Cite/Retire Audit",
    "exp1462": "Paper-v6 Anchored Claims Narrowing",
    "exp1463": "Local SOTA GGUF Runtime Repair",
    "exp1464": "Repair Validation-Error-as-Context A/B",
    "exp1465": "External Verifier Benchmark Fit Audit",
    "exp1466": "Milestone .112 Retrospective",
}

_REQUIRED_RETIREMENT_BLOCKS = {
    "grpo_vprm": {
        "name": "GRPO/VPRM",
        "retro_tokens": ("grpo", "vprm"),
        "exclusion_tokens": ("grpo", "vprm", "2026.04.112"),
    },
    "wopr_puzzle_cartridges": {
        "name": "WOPR puzzle cartridges",
        "retro_tokens": ("wopr", "puzzle"),
        "exclusion_tokens": ("wopr", "puzzle", "2026.04.112"),
    },
    "hardnet_dsp": {
        "name": "HardNet++/DSP",
        "retro_tokens": ("hardnet", "dsp"),
        "exclusion_tokens": ("hardnet", "dsp", "2026.04.112"),
    },
    "validation_error_repair": {
        "name": "Validation-error repair",
        "retro_tokens": ("validation-error", "repair"),
        "exclusion_tokens": (),
    },
}


def _write_json(path: Path, payload: Mapping[str, Any]) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    artifact = dict(payload)
    path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return artifact


def write_in_progress_artifact(out_path: Path | str = DEFAULT_OUT_PATH) -> dict[str, Any]:
    """REQ-REPORT-048: record that Exp 1467 started before reading evidence.

    The conductor treats a bootstrap artifact as proof that the task began.  The
    richer terminal artifact replaces it after the predecessor retro, archive
    row, and retirement blocks have been checked.
    """

    artifact: dict[str, Any] = {field: None for field in REQUIRED_ARTIFACT_FIELDS}
    artifact.update(
        {
            "experiment": EXPERIMENT,
            "schema": SCHEMA,
            "run_date": RUN_DATE,
            "project_root": PROJECT_ROOT_FOR_METADATA,
            "status": "in_progress",
            "milestone": TARGET_MILESTONE,
            "predecessor_milestone": PREDECESSOR_MILESTONE,
            "criteria_met": None,
            "criteria_total": None,
            "research_complete_has_112_entry": None,
            "activation_manifest_complete": False,
            "retired_lineages_preserved": False,
            "allowed_113_tracks": [],
            "forbidden_reopen_tracks": [],
            "honest_verdict": "in_progress_112_completion_archive_113_activation_seeded",
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


def _contains_all(text: str, tokens: tuple[str, ...]) -> bool:
    lowered = text.lower()
    return all(token.lower() in lowered for token in tokens)


def _research_complete_has_112_entry(research_complete_text: str) -> bool:
    return (
        "- id: 2026.04.112" in research_complete_text
        or "id: 2026.04.112" in research_complete_text
        or 'id: "2026.04.112"' in research_complete_text
        or "id: '2026.04.112'" in research_complete_text
    )


def _conductor_log_summary(conductor_log_text: str) -> dict[str, Any]:
    rows = conductor_log_text.splitlines()
    entries: dict[str, dict[str, Any]] = {}
    for exp_id, title in _EXP_LOG_NEEDLES.items():
        matches = [
            row
            for row in rows
            if exp_id in row or title in row or title[:38] in row
        ]
        ok = any("| OK |" in row for row in matches)
        entries[exp_id] = {
            "found": bool(matches),
            "ok": ok,
            "line": matches[-1] if matches else None,
        }
    missing = [exp_id for exp_id, entry in entries.items() if not entry["found"]]
    return {
        "entries": entries,
        "ok_count": sum(1 for entry in entries.values() if entry["ok"]),
        "expected_count": len(_EXP_LOG_NEEDLES),
        "missing_experiments": missing,
    }


def _retired_lineage_blocks(
    retro: Mapping[str, Any],
    exclusion_manifest_text: str,
) -> dict[str, dict[str, Any]]:
    retro_text = json.dumps(retro.get("retired_lineages", []), sort_keys=True)
    blocks: dict[str, dict[str, Any]] = {}
    for block_id, spec in _REQUIRED_RETIREMENT_BLOCKS.items():
        retro_present = _contains_all(retro_text, spec["retro_tokens"])
        exclusion_tokens = spec["exclusion_tokens"]
        exclusion_present = (
            True if not exclusion_tokens else _contains_all(exclusion_manifest_text, exclusion_tokens)
        )
        blocks[block_id] = {
            "name": spec["name"],
            "retro_present": retro_present,
            "exclusion_manifest_present": exclusion_present,
            "operator_reopen_required": True,
            "preserved": retro_present and exclusion_present,
        }
    return blocks


def _source_inputs_read(
    *,
    retro: Mapping[str, Any],
    conductor_log_text: str,
    research_complete_text: str,
    research_roadmap_text: str,
    exclusion_manifest_text: str,
    active_priorities_text: str,
) -> dict[str, dict[str, bool]]:
    return {
        f"results/{PREDECESSOR_RETRO_FILE}": {"exists": bool(retro)},
        "ops/conductor-log.md": {"exists": bool(conductor_log_text)},
        "research-complete.yaml": {"exists": bool(research_complete_text)},
        "research-roadmap.yaml": {"exists": bool(research_roadmap_text)},
        "ops/exclusion_manifest.yaml": {"exists": bool(exclusion_manifest_text)},
        "ops/active-priorities.md": {"exists": bool(active_priorities_text)},
    }


def _md_cell(value: object) -> str:
    return str(value).replace("\n", " ").replace("|", "\\|")


def _render_track_table(rows: list[dict[str, str]], *, forbidden: bool) -> list[str]:
    if forbidden:
        lines = ["| track | source | rule |", "|---|---|---|"]
        for row in rows:
            lines.append(
                f"| {_md_cell(row['name'])} | {_md_cell(row['source'])} | {_md_cell(row['rule'])} |"
            )
        return lines
    lines = ["| track | guardrail |", "|---|---|"]
    for row in rows:
        lines.append(f"| {_md_cell(row['name'])} | {_md_cell(row['guardrail'])} |")
    return lines


def render_manifest(
    *,
    artifact: Mapping[str, Any],
    blocked_reasons: list[str],
) -> str:
    """REQ-REPORT-048: render the operator-facing `.113` activation manifest.

    The markdown gives the terminal operator the short, readable version of the
    JSON decision: what can proceed in `.113`, what stays closed, and whether
    the predecessor archive row still needs reconciliation.
    """

    lines = [
        "# Milestone .113 Activation Manifest",
        "",
        f"Predecessor milestone: `{PREDECESSOR_MILESTONE}`",
        f"Target milestone: `{TARGET_MILESTONE}`",
        f"Run date: `{RUN_DATE}`",
        f".112 criteria: `{artifact['criteria_met']}` of `{artifact['criteria_total']}` met",
        "",
    ]
    if blocked_reasons:
        lines.extend(["Manifest blocked: " + "; ".join(blocked_reasons), ""])
    if not artifact["research_complete_has_112_entry"]:
        lines.extend(["Archive gap: `research-complete.yaml` lacks `2026.04.112`.", ""])

    lines.extend(["## Allowed .113 Tracks", ""])
    lines.extend(_render_track_table(list(ALLOWED_113_TRACKS), forbidden=False))
    lines.extend(["", "## Forbidden Reopen Tracks", ""])
    lines.extend(_render_track_table(list(FORBIDDEN_REOPEN_TRACKS), forbidden=True))
    lines.extend(["", "## Retired Lineage Preservation", ""])
    for block_id, block in artifact["retired_lineage_blocks"].items():
        lines.append(
            f"- {block_id}: preserved={block['preserved']}; "
            f"retro_present={block['retro_present']}; "
            f"exclusion_manifest_present={block['exclusion_manifest_present']}; "
            "operator-reopened required for future work."
        )
    lines.extend(
        [
            "",
            "## No-Change Confirmation",
            "",
            "- research-roadmap.yaml: unchanged_by_exp1467_activation_workflow",
            "- scripts/research_conductor.py: unchanged_by_exp1467_activation_workflow",
            "",
        ]
    )
    return "\n".join(lines)


def build_artifact(
    *,
    retro: Mapping[str, Any],
    conductor_log_text: str,
    research_complete_text: str,
    research_roadmap_text: str,
    exclusion_manifest_text: str,
    active_priorities_text: str,
    manifest_path: str,
) -> tuple[dict[str, Any], str]:
    """REQ-REPORT-048: summarize `.112` closure and activate only bounded `.113` work."""

    criteria_met = int(retro.get("criteria_met") or 0)
    criteria_total = int(retro.get("criteria_total") or 0)
    predecessor_complete = (
        str(retro.get("milestone")) == PREDECESSOR_MILESTONE
        and criteria_met == 14
        and criteria_total == 14
    )
    has_112_archive = _research_complete_has_112_entry(research_complete_text)
    retired_blocks = _retired_lineage_blocks(retro, exclusion_manifest_text)
    missing_retirement_blocks = [
        block_id for block_id, block in retired_blocks.items() if not block["preserved"]
    ]
    retired_preserved = not missing_retirement_blocks
    blocked_reasons: list[str] = []
    if missing_retirement_blocks:
        blocked_reasons.append(
            "missing retired-lineage blocks: " + ", ".join(missing_retirement_blocks)
        )
    if not predecessor_complete:
        blocked_reasons.append("predecessor retro criteria not complete")

    activation_manifest_complete = not blocked_reasons
    archive_gap = None
    if not has_112_archive:
        archive_gap = {
            "missing_milestone": PREDECESSOR_MILESTONE,
            "recommended_action": (
                "append .112 archive row to research-complete.yaml without modifying research-roadmap.yaml"
            ),
        }

    artifact: dict[str, Any] = {
        "experiment": EXPERIMENT,
        "schema": SCHEMA,
        "run_date": RUN_DATE,
        "project_root": PROJECT_ROOT_FOR_METADATA,
        "status": "complete" if activation_manifest_complete else "blocked",
        "milestone": TARGET_MILESTONE,
        "predecessor_milestone": PREDECESSOR_MILESTONE,
        "predecessor_honest_verdict": retro.get("honest_verdict"),
        "criteria_met": criteria_met,
        "criteria_total": criteria_total,
        "research_complete_has_112_entry": has_112_archive,
        "research_complete_archive_update_needed": not has_112_archive,
        "archive_gap": archive_gap,
        "activation_manifest_complete": activation_manifest_complete,
        "manifest_path": manifest_path,
        "retired_lineages_preserved": retired_preserved,
        "retired_lineage_blocks": retired_blocks,
        "allowed_113_tracks": list(ALLOWED_113_TRACKS),
        "forbidden_reopen_tracks": list(FORBIDDEN_REOPEN_TRACKS),
        "blocked_reasons": blocked_reasons,
        "conductor_log_exp1453_to_exp1466": _conductor_log_summary(conductor_log_text),
        "carry_forward_tracks_from_112": list(retro.get("carry_forward_tracks", [])),
        "source_inputs_read": _source_inputs_read(
            retro=retro,
            conductor_log_text=conductor_log_text,
            research_complete_text=research_complete_text,
            research_roadmap_text=research_roadmap_text,
            exclusion_manifest_text=exclusion_manifest_text,
            active_priorities_text=active_priorities_text,
        ),
        "research_roadmap_yaml_modified": False,
        "scripts_research_conductor_modified": False,
        "no_change_confirmations": {
            "research-roadmap.yaml": "unchanged_by_exp1467_activation_workflow",
            "scripts/research_conductor.py": "unchanged_by_exp1467_activation_workflow",
        },
        "honest_verdict": (
            "milestone_113_activation_complete_112_archived_retirements_preserved"
            if activation_manifest_complete and has_112_archive
            else "milestone_113_activation_complete_research_complete_112_archive_gap_recorded"
            if activation_manifest_complete
            else "milestone_113_activation_blocked_missing_predecessor_or_retirement_evidence"
        ),
    }
    manifest = render_manifest(artifact=artifact, blocked_reasons=blocked_reasons)
    return artifact, manifest


def run(
    root: Path | str = REPO_ROOT,
    out_path: Path | str = DEFAULT_OUT_PATH,
    manifest_path: Path | str = DEFAULT_MANIFEST_PATH,
) -> dict[str, Any]:
    """REQ-REPORT-048: write bootstrap, markdown manifest, and terminal JSON.

    The function deliberately writes only the Exp 1467 artifact and the
    operator activation markdown.  It does not mutate the active roadmap or the
    conductor because this task is a handoff gate, not a planner rewrite.
    """

    root_path = Path(root)
    out = Path(out_path)
    manifest_out = Path(manifest_path)
    write_in_progress_artifact(out)
    retro = _read_json(root_path / "results" / PREDECESSOR_RETRO_FILE) or {}
    artifact, manifest = build_artifact(
        retro=retro,
        conductor_log_text=_read_text(root_path / "ops" / "conductor-log.md"),
        research_complete_text=_read_text(root_path / "research-complete.yaml"),
        research_roadmap_text=_read_text(root_path / "research-roadmap.yaml"),
        exclusion_manifest_text=_read_text(root_path / "ops" / "exclusion_manifest.yaml"),
        active_priorities_text=_read_text(root_path / "ops" / "active-priorities.md"),
        manifest_path=_relative_path(manifest_out),
    )
    manifest_out.parent.mkdir(parents=True, exist_ok=True)
    manifest_out.write_text(manifest, encoding="utf-8")
    return _write_json(out, artifact)


if __name__ == "__main__":  # pragma: no cover - thin CLI convenience
    run()
