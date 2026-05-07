"""Build the Exp 1492 `.115` activation manifest.

Spec: REQ-REPORT-052, SCENARIO-REPORT-052.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
PROJECT_ROOT_FOR_METADATA = "/home/ianblenke/github.com/ianblenke/carnot"
RUN_DATE = "20260507"
PREDECESSOR_MILESTONE = "2026.04.114"
TARGET_MILESTONE = "2026.04.115"
EXPERIMENT = "1492_114_completion_archive_115_activation"
SCHEMA = "milestone_115_activation_manifest_v1"

DEFAULT_OUT_PATH = (
    REPO_ROOT / "results" / "experiment_1492_114_completion_archive_115_activation.json"
)
DEFAULT_MANIFEST_PATH = REPO_ROOT / "ops" / "milestone_115_activation_manifest.md"
PREDECESSOR_RETRO_FILE = "experiment_1491_milestone_114_retro.json"

REQUIRED_ARTIFACT_FIELDS = {
    "status",
    "milestone",
    "predecessor_milestone",
    "predecessor_criteria_met",
    "predecessor_criteria_total",
    "activation_manifest_complete",
    "retired_headline_signals",
    "allowed_115_tracks",
    "gated_115_tracks",
    "continuous_self_learning_required",
    "mandated_sota_models",
    "research_complete_has_114_entry",
    "honest_verdict",
}

MANDATED_SOTA_MODELS = [
    "unsloth/Qwen3.6-35B-A3B-GGUF",
    "unsloth/gemma-4-31B-it-GGUF",
    "unsloth/gemma-4-26B-A4B-it-GGUF",
]

RETIRED_HEADLINE_SIGNALS = [
    "Semantic Energy/logit telemetry headline claims",
    "V_1 pairwise self-verification headline claims",
]

GUARDRAIL_BLOCKS = [
    "Semantic Energy/logit telemetry headline claims",
    "V_1 pairwise headline claims",
    "decoded-quality claims from injected-failure localization",
    "THRML parity before import readiness",
    "KV260 board claims",
    "TSU hardware claims",
    "GRPO/VPRM reopenings",
    "WOPR puzzle cartridges",
    "legacy small-model headline results",
]

ALLOWED_115_TRACKS = [
    {
        "track": "trigger_token_certificate_export",
        "name": "Trigger-Token Certificate Export",
        "guardrail": "Switch to constrained certificates only after a trigger token.",
    },
    {
        "track": "prompt_to_validator_compilation",
        "name": "Prompt-to-Validator Compilation",
        "guardrail": (
            "Compile prompt constraints into executable validators with false-accept "
            "accounting."
        ),
    },
    {
        "track": "interwhen_style_monitoring",
        "name": "interwhen-Style Monitoring",
        "guardrail": (
            "Poll intermediate traces only after certificate and validator readiness "
            "gates pass."
        ),
    },
    {
        "track": "hover_safe_prefix_continuation",
        "name": "HoVer Safe-Prefix Continuation",
        "guardrail": "Continue from the last verified prefix without increasing false accepts.",
    },
    {
        "track": "fr11_trace2skill_daily_eval",
        "name": "FR-11 Trace2Skill Daily Eval",
        "guardrail": (
            "Mandatory continuous self-learning work must include rot and resolver "
            "checks."
        ),
    },
    {
        "track": "artifact_reachability",
        "name": "Artifact Reachability",
        "guardrail": "Learned skills must not point at missing, stale, or unreachable evidence.",
    },
    {
        "track": "verifier_orthogonality",
        "name": "Verifier Orthogonality",
        "guardrail": "Measure redundancy before promoting verifier ensembles.",
    },
    {
        "track": "graph_energy_adapters",
        "name": "Graph-Energy Adapters",
        "guardrail": "Treat graph risk as deterministic adapter evidence, not a Kona/TSU claim.",
    },
    {
        "track": "kan_hardware_accounting",
        "name": "KAN Hardware Accounting",
        "guardrail": "Run no-synthesis resource accounting only; no board or accelerator claim.",
    },
    {
        "track": "gated_thrml_import_parity",
        "name": "Gated THRML Import/Parity",
        "guardrail": "Classify import readiness before any simulator parity run.",
    },
]

GATED_115_TRACKS = [
    {
        "track": "interwhen_style_monitoring",
        "task_id": "exp1495",
        "gated_on": [
            "exp1493.trigger_certificate_ready == true",
            "exp1494.validator_compiler_ready == true",
        ],
    },
    {
        "track": "hover_safe_prefix_continuation",
        "task_id": "exp1496",
        "gated_on": ["exp1495.monitor_intervention_ready == true"],
    },
    {
        "track": "artifact_reachability",
        "task_id": "exp1498",
        "gated_on": ["exp1497.daily_eval_manifest_ready == true"],
    },
    {
        "track": "latent_vs_deterministic_discipline_gate",
        "task_id": "exp1500",
        "gated_on": ["exp1499.orthogonality_matrix_written == true"],
    },
    {
        "track": "thrml_carnot_simulator_parity",
        "task_id": "exp1504",
        "gated_on": ["exp1503.thrml_import_ready == true"],
    },
]

_EXP_LOG_NEEDLES = {
    "exp1479": ".113 Completion Archive + .114 Activation Manifest",
    "exp1480": "Live SOTA Telemetry v2",
    "exp1481": "Semantic Energy Feasibility Audit",
    "exp1482": "BEAVER-Lite Live Prefix Bound Calibration",
    "exp1483": "HalluGuard Risk-Bound Fit Audit",
    "exp1484": "FR-11 v9 Query-Time Memory Policy",
    "exp1485": "FR-11 Completeness Reduction Audit",
    "exp1486": "CCTU Executable Constraint Micro-Benchmark",
    "exp1487": "V_1 Pairwise Self-Verification",
    "exp1488": "THRML Installability and Import Preflight",
    "exp1489": "THRML/Carnot Simulator Parity v2",
    "exp1490": "Kona/EBT Partial-Trace Energy Localization",
    "exp1491": "Milestone .114 Retrospective",
}


def _write_json(path: Path, payload: Mapping[str, Any]) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    artifact = dict(payload)
    path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return artifact


def write_in_progress_artifact(out_path: Path | str = DEFAULT_OUT_PATH) -> dict[str, Any]:
    """REQ-REPORT-052: record that Exp 1492 started before evidence loading.

    The bootstrap file is intentionally not terminal: it lets the conductor see
    that the activation task began, then the real run replaces it after reading
    the `.114` retro, ops evidence, archive state, and guardrail context.
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
            "activation_manifest_complete": False,
            "retired_headline_signals": list(RETIRED_HEADLINE_SIGNALS),
            "allowed_115_tracks": [],
            "gated_115_tracks": [],
            "continuous_self_learning_required": True,
            "mandated_sota_models": list(MANDATED_SOTA_MODELS),
            "research_complete_has_114_entry": None,
            "honest_verdict": "passed_in_progress_114_completion_archive_115_activation_seeded",
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


def _research_complete_has_114_entry(research_complete_text: str) -> bool:
    return (
        "- id: 2026.04.114" in research_complete_text
        or "id: 2026.04.114" in research_complete_text
        or 'id: "2026.04.114"' in research_complete_text
        or "id: '2026.04.114'" in research_complete_text
    )


def _text_has_any(text: str, *needles: str) -> bool:
    lowered = text.lower()
    return any(needle.lower() in lowered for needle in needles)


def _combined_evidence_text(
    retro: Mapping[str, Any],
    ops_status_text: str,
    ops_changelog_text: str,
) -> str:
    return f"{json.dumps(retro, sort_keys=True)}\n{ops_status_text}\n{ops_changelog_text}".lower()


def _retired_headline_signals_preserved(
    retro: Mapping[str, Any],
    ops_status_text: str,
    ops_changelog_text: str,
) -> bool:
    evidence = _combined_evidence_text(retro, ops_status_text, ops_changelog_text)
    semantic_retired = _text_has_any(evidence, "semantic_energy", "semantic energy") and (
        "retired" in evidence or "claim_allowed" in evidence
    )
    v1_retired = _text_has_any(evidence, "v1_pairwise", "v_1", "pairwise") and (
        "do_not_promote" in evidence or "retired" in evidence or "underperformed" in evidence
    )
    return bool(semantic_retired and v1_retired)


def _structured_thrml_gate_skip_recorded(retro: Mapping[str, Any]) -> bool:
    skip_text = json.dumps(retro.get("honest_structured_gate_skips", []), sort_keys=True).lower()
    blocked_ids = {str(task_id).lower() for task_id in retro.get("blocked_task_ids", [])}
    return bool(
        int(retro.get("honest_structured_gate_skip_count") or 0) >= 1
        and ("exp1489" in blocked_ids or "exp1489" in skip_text)
        and "thrml" in skip_text
    )


def _guardrail_blocks_preserved(
    retro: Mapping[str, Any],
    ops_status_text: str,
    ops_changelog_text: str,
) -> bool:
    evidence = _combined_evidence_text(retro, ops_status_text, ops_changelog_text)
    return bool(
        _retired_headline_signals_preserved(retro, ops_status_text, ops_changelog_text)
        and _text_has_any(evidence, "no_decoded_quality_claim", "decoded-quality")
        and "thrml" in evidence
        and ("prior_scope_reduction_blocks" in evidence or "grpo" in evidence or "wopr" in evidence)
    )


def _continuous_self_learning_recorded(ops_status_text: str, ops_changelog_text: str) -> bool:
    evidence = f"{ops_status_text}\n{ops_changelog_text}".lower()
    return "continuous self-learning" in evidence and "exp1497" in evidence


def _mandated_models_recorded(ops_status_text: str, ops_changelog_text: str) -> bool:
    evidence = f"{ops_status_text}\n{ops_changelog_text}"
    return all(model in evidence for model in MANDATED_SOTA_MODELS)


def _protected_files_unchanged(retro: Mapping[str, Any]) -> bool:
    checks = retro.get("protected_file_checks", {})
    if not isinstance(checks, Mapping):
        checks = {}
    roadmap_check = str(checks.get("research-roadmap.yaml", "")).lower()
    conductor_check = str(checks.get("scripts/research_conductor.py", "")).lower()
    return bool(
        retro.get("research_roadmap_yaml_modified") is False
        and retro.get("scripts_research_conductor_modified") is False
        and roadmap_check == "unchanged"
        and conductor_check == "unchanged"
    )


def _conductor_log_summary(conductor_log_text: str) -> dict[str, Any]:
    rows = conductor_log_text.splitlines()
    entries: dict[str, dict[str, Any]] = {}
    for exp_id, title in _EXP_LOG_NEEDLES.items():
        matches = [row for row in rows if exp_id in row or title in row or title[:38] in row]
        entries[exp_id] = {
            "found": bool(matches),
            "ok": any("| OK |" in row for row in matches),
            "terminal": any(
                marker in row
                for row in matches
                for marker in ("| OK |", "| GATE_BLOCK |", "| FAIL |")
            ),
            "line": matches[-1] if matches else None,
        }
    return {
        "entries": entries,
        "ok_count": sum(1 for entry in entries.values() if entry["ok"]),
        "terminal_count": sum(1 for entry in entries.values() if entry["terminal"]),
        "expected_count": len(_EXP_LOG_NEEDLES),
        "missing_experiments": [
            exp_id for exp_id, entry in entries.items() if not entry["found"]
        ],
    }


def _source_inputs_read(
    *,
    retro: Mapping[str, Any],
    conductor_log_text: str,
    research_complete_text: str,
    ops_status_text: str,
    ops_changelog_text: str,
) -> dict[str, dict[str, bool]]:
    return {
        f"results/{PREDECESSOR_RETRO_FILE}": {"exists": bool(retro)},
        "ops/conductor-log.md": {"exists": bool(conductor_log_text)},
        "research-complete.yaml": {"exists": bool(research_complete_text)},
        "ops/status.md": {"exists": bool(ops_status_text)},
        "ops/changelog.md": {"exists": bool(ops_changelog_text)},
    }


def _md_cell(value: object) -> str:
    return str(value).replace("\n", " ").replace("|", "\\|")


def _render_allowed_track_table(rows: list[dict[str, str]]) -> list[str]:
    lines = ["| track | guardrail |", "|---|---|"]
    for row in rows:
        lines.append(f"| {_md_cell(row['name'])} | {_md_cell(row['guardrail'])} |")
    return lines


def _render_gated_track_table(rows: list[dict[str, Any]]) -> list[str]:
    lines = ["| track | task | gate |", "|---|---|---|"]
    for row in rows:
        gates = "; ".join(row["gated_on"])
        lines.append(
            f"| {_md_cell(row['track'])} | {_md_cell(row['task_id'])} | "
            f"{_md_cell(gates)} |"
        )
    return lines


def render_manifest(
    *,
    artifact: Mapping[str, Any],
    blocked_reasons: list[str],
) -> str:
    """REQ-REPORT-052: render the operator-facing `.115` activation manifest.

    The markdown mirrors the JSON in terms operators can audit quickly: what
    may run in `.115`, what remains gated, which signals are retired, and which
    protected files this activation workflow deliberately leaves alone.
    """

    lines = [
        "# Milestone .115 Activation Manifest",
        "",
        f"Predecessor milestone: `{PREDECESSOR_MILESTONE}`",
        f"Target milestone: `{TARGET_MILESTONE}`",
        f"Run date: `{RUN_DATE}`",
        (
            f".114 criteria: `{artifact['predecessor_criteria_met']}` of "
            f"`{artifact['predecessor_criteria_total']}` met"
        ),
        "",
    ]
    if blocked_reasons:
        lines.extend(["Manifest blocked: " + "; ".join(blocked_reasons), ""])
    if not artifact["research_complete_has_114_entry"]:
        lines.extend(["Archive gap: `research-complete.yaml` lacks `2026.04.114`.", ""])

    lines.extend(["## Allowed .115 Tracks", ""])
    lines.extend(_render_allowed_track_table(list(ALLOWED_115_TRACKS)))
    lines.extend(["", "## Gated .115 Tracks", ""])
    lines.extend(_render_gated_track_table(list(GATED_115_TRACKS)))
    lines.extend(["", "## Retired Headline Signals", ""])
    for signal in artifact["retired_headline_signals"]:
        lines.append(f"- {signal}")
    lines.extend(["", "## Carry-Forward Guardrail Blocks", ""])
    for block in artifact["guardrail_blocks"]:
        lines.append(f"- {block}")
    lines.extend(
        [
            "",
            "## Continuous Self-Learning",
            "",
            f"- continuous_self_learning_required: {artifact['continuous_self_learning_required']}",
            "- required task: exp1497 FR-11 trace2skill daily eval with rot and resolver checks",
            "",
            "## Mandated Local SOTA Models",
            "",
        ]
    )
    for model in artifact["mandated_sota_models"]:
        lines.append(f"- {model}")
    lines.extend(
        [
            "",
            "## No-Change Confirmation",
            "",
            "- research-roadmap.yaml: unchanged_by_exp1492_activation_workflow",
            "- scripts/research_conductor.py: unchanged_by_exp1492_activation_workflow",
            "",
        ]
    )
    return "\n".join(lines)


def build_artifact(
    *,
    retro: Mapping[str, Any],
    conductor_log_text: str,
    research_complete_text: str,
    ops_status_text: str,
    ops_changelog_text: str,
    manifest_path: str,
) -> tuple[dict[str, Any], str]:
    """REQ-REPORT-052: summarize `.114` closure and activate guarded `.115` work."""

    criteria_met = int(retro.get("criteria_met") or 0)
    criteria_total = int(retro.get("criteria_total") or 0)
    predecessor_complete = (
        str(retro.get("status")) == "complete"
        and str(retro.get("milestone")) == PREDECESSOR_MILESTONE
        and criteria_met == 12
        and criteria_total == 13
        and retro.get("success_threshold_met") is True
    )
    has_114_archive = _research_complete_has_114_entry(research_complete_text)
    retired_signals_preserved = _retired_headline_signals_preserved(
        retro,
        ops_status_text,
        ops_changelog_text,
    )
    structured_gate_skip = _structured_thrml_gate_skip_recorded(retro)
    guardrail_blocks = _guardrail_blocks_preserved(retro, ops_status_text, ops_changelog_text)
    continuous_self_learning = _continuous_self_learning_recorded(
        ops_status_text,
        ops_changelog_text,
    )
    mandated_models = _mandated_models_recorded(ops_status_text, ops_changelog_text)
    protected_files = _protected_files_unchanged(retro)

    blocked_reasons: list[str] = []
    if not predecessor_complete:
        blocked_reasons.append("predecessor retro criteria not complete")
    if not structured_gate_skip:
        blocked_reasons.append("structured THRML gate skip not recorded")
    if not retired_signals_preserved:
        blocked_reasons.append("retired headline signals not preserved")
    if not guardrail_blocks:
        blocked_reasons.append("guardrail blocks not preserved")
    if not continuous_self_learning:
        blocked_reasons.append("continuous self-learning requirement not recorded")
    if not mandated_models:
        blocked_reasons.append("mandated SOTA models not recorded")
    if not protected_files:
        blocked_reasons.append("protected files not confirmed unchanged")

    activation_manifest_complete = not blocked_reasons
    archive_gap = None
    if not has_114_archive:
        archive_gap = {
            "missing_milestone": PREDECESSOR_MILESTONE,
            "recommended_action": (
                "append .114 archive row to research-complete.yaml without modifying "
                "research-roadmap.yaml"
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
        "predecessor_criteria_met": criteria_met,
        "predecessor_criteria_total": criteria_total,
        "activation_manifest_complete": activation_manifest_complete,
        "retired_headline_signals": list(RETIRED_HEADLINE_SIGNALS),
        "retired_headline_signals_preserved": retired_signals_preserved,
        "allowed_115_tracks": list(ALLOWED_115_TRACKS),
        "gated_115_tracks": list(GATED_115_TRACKS),
        "guardrail_blocks": list(GUARDRAIL_BLOCKS),
        "guardrail_blocks_preserved": guardrail_blocks,
        "continuous_self_learning_required": True,
        "continuous_self_learning_requirement_recorded": continuous_self_learning,
        "mandated_sota_models": list(MANDATED_SOTA_MODELS),
        "mandated_sota_models_recorded": mandated_models,
        "research_complete_has_114_entry": has_114_archive,
        "research_complete_archive_update_needed": not has_114_archive,
        "archive_gap": archive_gap,
        "manifest_path": manifest_path,
        "blocked_reasons": blocked_reasons,
        "structured_thrml_gate_skip_recorded": structured_gate_skip,
        "honest_structured_gate_skip_count": retro.get("honest_structured_gate_skip_count"),
        "conductor_log_exp1479_to_exp1491": _conductor_log_summary(conductor_log_text),
        "predecessor_carry_forward_recommendations": list(
            retro.get("carry_forward_recommendations", [])
        ),
        "predecessor_retired_lineages": list(retro.get("retired_lineages", [])),
        "source_inputs_read": _source_inputs_read(
            retro=retro,
            conductor_log_text=conductor_log_text,
            research_complete_text=research_complete_text,
            ops_status_text=ops_status_text,
            ops_changelog_text=ops_changelog_text,
        ),
        "research_roadmap_yaml_modified": False,
        "scripts_research_conductor_modified": False,
        "protected_files_unchanged": protected_files,
        "no_change_confirmations": {
            "research-roadmap.yaml": "unchanged_by_exp1492_activation_workflow",
            "scripts/research_conductor.py": "unchanged_by_exp1492_activation_workflow",
        },
        "honest_verdict": (
            "complete: milestone_115_activation_complete_114_archived_guardrails_preserved"
            if activation_manifest_complete and has_114_archive
            else (
                "complete: "
                "milestone_115_activation_complete_research_complete_114_archive_gap_recorded"
            )
            if activation_manifest_complete
            else (
                "passed: "
                "milestone_115_activation_blocked_missing_predecessor_or_guardrail_evidence"
            )
        ),
    }
    manifest = render_manifest(artifact=artifact, blocked_reasons=blocked_reasons)
    return artifact, manifest


def run(
    root: Path | str = REPO_ROOT,
    out_path: Path | str = DEFAULT_OUT_PATH,
    manifest_path: Path | str = DEFAULT_MANIFEST_PATH,
) -> dict[str, Any]:
    """REQ-REPORT-052: write bootstrap, markdown manifest, and terminal JSON.

    The activation step writes only the Exp 1492 artifact and the operator
    manifest. It intentionally does not mutate `research-roadmap.yaml`,
    `research-roadmap-next.yaml`, `ops/status.md`, `ops/changelog.md`, or the
    conductor, because those files are reconciled by separate milestone steps.
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
        ops_status_text=_read_text(root_path / "ops" / "status.md"),
        ops_changelog_text=_read_text(root_path / "ops" / "changelog.md"),
        manifest_path=_relative_path(manifest_out),
    )
    manifest_out.parent.mkdir(parents=True, exist_ok=True)
    manifest_out.write_text(manifest, encoding="utf-8")
    return _write_json(out, artifact)


if __name__ == "__main__":  # pragma: no cover - thin CLI convenience
    run()
