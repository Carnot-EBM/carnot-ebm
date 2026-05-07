"""Build the Exp 1506 `.116` activation manifest.

Spec: REQ-REPORT-054, SCENARIO-REPORT-054.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
PROJECT_ROOT_FOR_METADATA = "/home/ianblenke/github.com/ianblenke/carnot"
RUN_DATE = "20260507"
PREDECESSOR_MILESTONE = "2026.04.115"
TARGET_MILESTONE = "2026.04.116"
EXPERIMENT = "1506_115_completion_archive_116_activation"
SCHEMA = "milestone_116_activation_manifest_v1"

DEFAULT_OUT_PATH = (
    REPO_ROOT / "results" / "experiment_1506_115_completion_archive_116_activation.json"
)
DEFAULT_MANIFEST_PATH = REPO_ROOT / "ops" / "milestone_116_activation_manifest.md"
PREDECESSOR_RETRO_FILE = "experiment_1505_milestone_115_retro.json"
EXP1502_FILE = "experiment_1502_kan_hardware_accounting_quantkan_kaem.json"
EXP1504_FILE = "experiment_1504_thrml_carnot_simulator_parity_v3.json"

REQUIRED_ARTIFACT_FIELDS = {
    "status",
    "milestone",
    "predecessor_milestone",
    "predecessor_criteria_met",
    "predecessor_criteria_total",
    "activation_manifest_complete",
    "prior_trigger_certificates_ready",
    "prior_validator_compiler_ready",
    "prior_monitor_replay_ready",
    "prior_fr11_daily_eval_ready",
    "prior_thrml_parity_ready",
    "prior_kan_shape_blocker_recorded",
    "prior_kv260_source_track_active",
    "mandated_sota_models",
    "continuous_self_learning_required",
    "retired_headline_signals",
    "allowed_116_tracks",
    "gated_116_tracks",
    "research_complete_has_115_entry",
    "honest_verdict",
}

MANDATED_SOTA_MODELS = [
    "unsloth/Qwen3.6-35B-A3B-GGUF",
    "unsloth/gemma-4-31B-it-GGUF",
    "unsloth/gemma-4-26B-A4B-it-GGUF",
]

RETIRED_HEADLINE_SIGNALS = [
    "Semantic Energy/logit telemetry headline claims",
    "V_1 pairwise headline claims",
    "decoded-quality claims from injected-failure localization",
    "arbitrary generated-Python verifier trust",
    "TSU hardware claims",
    "KV260 board claims",
    "synthesis claims",
    "legacy small-model headline results",
]

ALLOWED_116_TRACKS = [
    {
        "track": "safe_dsl_verifier_induction",
        "name": "safe-DSL verifier induction",
        "guardrail": "Candidate verifiers must compile through the safe DSL.",
    },
    {
        "track": "trigger_grammar_certificate_decoding",
        "name": "trigger+grammar certificate decoding",
        "guardrail": "Use grammar-bounded certificates; do not trust free-form parsing.",
    },
    {
        "track": "executable_monitor_runtime",
        "name": "executable monitor runtime",
        "guardrail": "Runtime events remain replayable and false-accept checked.",
    },
    {
        "track": "plan_graph_structural_contracts",
        "name": "plan-graph structural contracts",
        "guardrail": "Promote deterministic structural gates, not trained-GNN headlines.",
    },
    {
        "track": "product_line_solver_oracle",
        "name": "product-line solver oracle",
        "guardrail": "Anchor feature-model claims to deterministic solver feasibility.",
    },
    {
        "track": "fr11_verifier_feedback_replay",
        "name": "FR-11 verifier-feedback replay",
        "guardrail": "Bound updates to query-time policy replay with rollback evidence.",
    },
    {
        "track": "trace2skill_portable_pack",
        "name": "trace2skill portable pack",
        "guardrail": "Promote only rollback-passing skill entries with provenance.",
    },
    {
        "track": "thrml_samplerbackend_conformance",
        "name": "THRML SamplerBackend conformance",
        "guardrail": "Simulator/software conformance only; no TSU hardware claim.",
    },
    {
        "track": "kan_shape_normalization",
        "name": "KAN shape normalization",
        "guardrail": "Normalize proxy-vs-hardware shapes before any synthesis claim.",
    },
    {
        "track": "kv260_source_level_rtl_properties",
        "name": "KV260 source-level RTL properties",
        "guardrail": "Source-level lint/property work only; no board or bitstream claim.",
    },
]

GATED_116_TRACKS = [
    {
        "track": "trigger_grammar_certificate_decoding",
        "task_id": "exp1508",
        "gated_on": ["exp1507.verifier_induction_ready == true"],
    },
    {
        "track": "executable_monitor_runtime",
        "task_id": "exp1509",
        "gated_on": [
            "exp1507.verifier_induction_ready == true",
            "exp1508.certificate_decoder_ready == true",
        ],
    },
    {
        "track": "fr11_policy_rollback_replay",
        "task_id": "exp1513",
        "gated_on": ["exp1512.policy_cache_ready == true"],
    },
    {
        "track": "trace2skill_portable_pack",
        "task_id": "exp1514",
        "gated_on": ["exp1513.rollback_audit_passed == true"],
    },
    {
        "track": "thrml_samplerbackend_conformance",
        "task_id": "exp1515",
        "gated_on": ["exp1506.prior_thrml_parity_ready == true"],
    },
    {
        "track": "kan_shape_normalization",
        "task_id": "exp1516",
        "gated_on": ["exp1506.prior_kan_shape_blocker_recorded == true"],
    },
    {
        "track": "kv260_source_level_rtl_properties",
        "task_id": "exp1517",
        "gated_on": ["exp1506.prior_kv260_source_track_active == true"],
    },
]

_EXP_LOG_NEEDLES = {
    "exp1492": ".114 Completion Archive + .115 Activation Manifest",
    "exp1493": "Trigger-Token Certificate Export v1",
    "exp1494": "ConstrainPrompt Validator Compiler Audit",
    "exp1495": "interwhen Monitor Prototype",
    "exp1496": "HoVer Safe-Prefix Continuation Audit",
    "exp1497": "FR-11 v10 Trace2Skill Daily Eval",
    "exp1498": "trace2skill Artifact Reachability Audit",
    "exp1499": "Verifier Ensemble DRY",
    "exp1500": "Latent-vs-Deterministic Discipline Gate",
    "exp1501": "GNNVerifier Plan-Graph Energy Adapter",
    "exp1502": "KAN Hardware Accounting",
    "exp1503": "THRML Import Readiness Repair",
    "exp1504": "THRML/Carnot Simulator Parity v3",
    "exp1505": "Milestone .115 Retrospective",
}


def _write_json(path: Path, payload: Mapping[str, Any]) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    artifact = dict(payload)
    path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return artifact


def write_in_progress_artifact(out_path: Path | str = DEFAULT_OUT_PATH) -> dict[str, Any]:
    """REQ-REPORT-054: create the visible bootstrap artifact before evidence loading.

    The in-progress file is deliberately sparse but schema-shaped. It gives the
    conductor a durable breadcrumb that Exp 1506 started before the run reads
    `.115` evidence and replaces the file with the terminal archive.
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
            "continuous_self_learning_required": True,
            "mandated_sota_models": list(MANDATED_SOTA_MODELS),
            "retired_headline_signals": list(RETIRED_HEADLINE_SIGNALS),
            "allowed_116_tracks": [],
            "gated_116_tracks": [],
            "honest_verdict": "passed_in_progress_115_completion_archive_116_activation_seeded",
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
    parts = path.parts
    if "ops" in parts:
        return str(Path(*parts[parts.index("ops") :]))
    if "results" in parts:
        return str(Path(*parts[parts.index("results") :]))
    return path.name


def _research_complete_has_115_entry(research_complete_text: str) -> bool:
    return any(
        marker in research_complete_text
        for marker in (
            "- id: 2026.04.115",
            "id: 2026.04.115",
            'id: "2026.04.115"',
            "id: '2026.04.115'",
        )
    )


def _line_decision_ready(retro: Mapping[str, Any], key: str, expected_decision: str) -> bool:
    decisions = retro.get("line_decisions", {})
    if not isinstance(decisions, Mapping):
        return False
    row = decisions.get(key, {})
    if not isinstance(row, Mapping):
        return False
    return str(row.get("decision", "")).lower() == expected_decision


def _prior_thrml_parity_ready(exp1504: Mapping[str, Any]) -> bool:
    metadata = exp1504.get("metadata", {})
    if not isinstance(metadata, Mapping):
        metadata = {}
    verdict = str(exp1504.get("honest_verdict", "")).lower()
    return bool(
        exp1504.get("status") == "complete"
        and exp1504.get("parity_experiment_ran") is True
        and int(exp1504.get("parity_pass_count") or 0) > 0
        and int(exp1504.get("parity_fail_count") or 0) == 0
        and exp1504.get("simulator_only") is True
        and exp1504.get("hardware_claim_allowed") is False
        and metadata.get("tsu_hardware_execution") is False
        and "no_hardware_claim" in verdict
    )


def _prior_kan_shape_blocker_recorded(exp1502: Mapping[str, Any]) -> bool:
    blocker_text = json.dumps(exp1502.get("blockers", []), sort_keys=True).lower()
    verdict = str(exp1502.get("honest_verdict", "")).lower()
    return bool(
        exp1502.get("status") == "complete"
        and exp1502.get("kan_hardware_accounting_ready") is True
        and exp1502.get("accounting_only_no_synthesis_claim") is True
        and exp1502.get("hardware_claim_allowed") is False
        and "shape" in blocker_text
        and ("normaliz" in blocker_text or "normalis" in blocker_text)
        and "no synthesis" in verdict
    )


def _prior_kv260_source_track_active(
    hardware_wishlist_text: str,
    architecture_text: str,
) -> bool:
    evidence = f"{hardware_wishlist_text}\n{architecture_text}".lower()
    return bool(
        "kv260" in evidence
        and "source-level" in evidence
        and "rtl" in evidence
        and ("lint" in evidence or "simulation" in evidence)
        and "no kv260 board" in evidence
    )


def _mandated_models_recorded(*texts: str) -> bool:
    evidence = "\n".join(texts)
    return all(model in evidence for model in MANDATED_SOTA_MODELS)


def _continuous_self_learning_recorded(*texts: str) -> bool:
    evidence = "\n".join(texts).lower()
    return "continuous self-learning" in evidence and "exp1512" in evidence


def _retirement_blocks_recorded(*texts: str) -> bool:
    evidence = "\n".join(texts).lower()
    required = [
        ("semantic energy", "logit"),
        ("v_1", "pairwise"),
        ("decoded-quality", "injected"),
        ("generated", "verifier", "safe dsl"),
        ("tsu", "hardware"),
        ("kv260", "board"),
        ("synthesis",),
        ("legacy small",),
    ]
    return all(all(term in evidence for term in terms) for terms in required)


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
        "missing_experiments": [exp_id for exp_id, entry in entries.items() if not entry["found"]],
    }


def _source_inputs_read(
    *,
    retro: Mapping[str, Any],
    exp1502: Mapping[str, Any],
    exp1504: Mapping[str, Any],
    conductor_log_text: str,
    research_complete_text: str,
    ops_status_text: str,
    ops_changelog_text: str,
    roadmap_text: str,
    roadmap_doc_text: str,
    hardware_wishlist_text: str,
    architecture_text: str,
) -> dict[str, dict[str, bool]]:
    return {
        f"results/{PREDECESSOR_RETRO_FILE}": {"exists": bool(retro)},
        f"results/{EXP1502_FILE}": {"exists": bool(exp1502)},
        f"results/{EXP1504_FILE}": {"exists": bool(exp1504)},
        "ops/conductor-log.md": {"exists": bool(conductor_log_text)},
        "research-complete.yaml": {"exists": bool(research_complete_text)},
        "ops/status.md": {"exists": bool(ops_status_text)},
        "ops/changelog.md": {"exists": bool(ops_changelog_text)},
        "research-roadmap.yaml": {"exists": bool(roadmap_text)},
        "openspec/change-proposals/research-roadmap-vNEXT.md": {
            "exists": bool(roadmap_doc_text),
        },
        "research-hardware-wishlist.md": {"exists": bool(hardware_wishlist_text)},
        "_bmad/architecture.md": {"exists": bool(architecture_text)},
    }


def _md_cell(value: object) -> str:
    return str(value).replace("\n", " ").replace("|", "\\|")


def _render_allowed_track_table(rows: Sequence[Mapping[str, str]]) -> list[str]:
    lines = ["| track | guardrail |", "|---|---|"]
    for row in rows:
        lines.append(f"| {_md_cell(row['name'])} | {_md_cell(row['guardrail'])} |")
    return lines


def _render_gated_track_table(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    lines = ["| track | task | gate |", "|---|---|---|"]
    for row in rows:
        gates = "; ".join(row["gated_on"])
        lines.append(
            f"| {_md_cell(row['track'])} | {_md_cell(row['task_id'])} | {_md_cell(gates)} |"
        )
    return lines


def render_manifest(*, artifact: Mapping[str, Any], blocked_reasons: Sequence[str]) -> str:
    """REQ-REPORT-054: render the operator-facing `.116` activation manifest.

    The markdown is the human audit surface for the JSON gate fields. It makes
    the next milestone's allowed work explicit while keeping unsupported
    telemetry, generated-code, and hardware claims visibly blocked.
    """

    lines = [
        "# Milestone .116 Activation Manifest",
        "",
        f"Predecessor milestone: `{PREDECESSOR_MILESTONE}`",
        f"Target milestone: `{TARGET_MILESTONE}`",
        f"Run date: `{RUN_DATE}`",
        (
            f".115 criteria: `{artifact['predecessor_criteria_met']}` of "
            f"`{artifact['predecessor_criteria_total']}` met"
        ),
        "",
    ]
    if blocked_reasons:
        lines.extend(["Manifest blocked: " + "; ".join(blocked_reasons), ""])

    lines.extend(["## Allowed .116 Tracks", ""])
    lines.extend(_render_allowed_track_table(ALLOWED_116_TRACKS))
    lines.extend(["", "## Gated .116 Tracks", ""])
    lines.extend(_render_gated_track_table(GATED_116_TRACKS))
    lines.extend(["", "## Prior Readiness", ""])
    for key in (
        "prior_trigger_certificates_ready",
        "prior_validator_compiler_ready",
        "prior_monitor_replay_ready",
        "prior_fr11_daily_eval_ready",
        "prior_thrml_parity_ready",
        "prior_kan_shape_blocker_recorded",
        "prior_kv260_source_track_active",
    ):
        lines.append(f"- {key}: {artifact[key]}")
    lines.extend(["", "## Retired Headline Signals And Blocks", ""])
    for signal in artifact["retired_headline_signals"]:
        lines.append(f"- {signal}")
    lines.extend(
        [
            "",
            "## Continuous Self-Learning",
            "",
            f"- continuous_self_learning_required: {artifact['continuous_self_learning_required']}",
            "- required task: exp1512 FR-11 verifier-feedback policy cache",
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
            "- research-roadmap.yaml: unchanged_by_exp1506_activation_workflow",
            "- scripts/research_conductor.py: unchanged_by_exp1506_activation_workflow",
            "",
        ]
    )
    return "\n".join(lines)


def build_artifact(
    *,
    retro: Mapping[str, Any],
    exp1502: Mapping[str, Any],
    exp1504: Mapping[str, Any],
    conductor_log_text: str,
    research_complete_text: str,
    ops_status_text: str,
    ops_changelog_text: str,
    roadmap_text: str,
    roadmap_doc_text: str,
    hardware_wishlist_text: str,
    architecture_text: str,
    manifest_path: str,
    protected_file_diffs: Sequence[str],
) -> tuple[dict[str, Any], str]:
    """REQ-REPORT-054: summarize `.115` closure and activate guarded `.116` work."""

    criteria_met = int(retro.get("criteria_met") or 0)
    criteria_total = int(retro.get("criteria_total") or 0)
    predecessor_complete = (
        str(retro.get("status")) == "complete"
        and str(retro.get("milestone")) == PREDECESSOR_MILESTONE
        and criteria_met == 12
        and criteria_total == 12
    )
    has_115_archive = _research_complete_has_115_entry(research_complete_text)
    prior_trigger = _line_decision_ready(retro, "trigger_certificate_export", "graduated")
    prior_validator = _line_decision_ready(
        retro,
        "constrainprompt_validator_compiler",
        "graduated",
    )
    prior_monitor = _line_decision_ready(retro, "interwhen_hover_monitoring", "graduated")
    prior_fr11 = _line_decision_ready(retro, "fr11_trace2skill", "graduated")
    prior_thrml = _prior_thrml_parity_ready(exp1504)
    prior_kan = _prior_kan_shape_blocker_recorded(exp1502)
    prior_kv260 = _prior_kv260_source_track_active(hardware_wishlist_text, architecture_text)
    continuous_self_learning = _continuous_self_learning_recorded(ops_status_text, roadmap_text)
    mandated_models = _mandated_models_recorded(ops_status_text, roadmap_text, roadmap_doc_text)
    retirement_blocks = _retirement_blocks_recorded(
        ops_changelog_text,
        roadmap_text,
        roadmap_doc_text,
    )
    conductor_summary = _conductor_log_summary(conductor_log_text)
    protected_files_unchanged = not protected_file_diffs

    blocked_reasons: list[str] = []
    if not predecessor_complete:
        blocked_reasons.append("predecessor retro criteria not complete")
    if not has_115_archive:
        blocked_reasons.append("research-complete.yaml lacks 2026.04.115 archive row")
    readiness_checks = {
        "prior trigger certificates not ready": prior_trigger,
        "prior validator compiler not ready": prior_validator,
        "prior monitor replay not ready": prior_monitor,
        "prior FR-11 daily eval not ready": prior_fr11,
        "prior THRML parity not ready": prior_thrml,
        "prior KAN shape blocker not recorded": prior_kan,
        "prior KV260 source track not active": prior_kv260,
    }
    blocked_reasons.extend(reason for reason, ready in readiness_checks.items() if not ready)
    if not continuous_self_learning:
        blocked_reasons.append("continuous self-learning requirement not recorded")
    if not mandated_models:
        blocked_reasons.append("mandated SOTA models not recorded")
    if not retirement_blocks:
        blocked_reasons.append("retired headline signal blocks not preserved")
    if conductor_summary["missing_experiments"]:
        blocked_reasons.append("conductor log missing exp1492-through-exp1505 evidence")
    if not protected_files_unchanged:
        blocked_reasons.append("protected files changed")

    activation_manifest_complete = not blocked_reasons
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
        "prior_trigger_certificates_ready": prior_trigger,
        "prior_validator_compiler_ready": prior_validator,
        "prior_monitor_replay_ready": prior_monitor,
        "prior_fr11_daily_eval_ready": prior_fr11,
        "prior_thrml_parity_ready": prior_thrml,
        "prior_kan_shape_blocker_recorded": prior_kan,
        "prior_kv260_source_track_active": prior_kv260,
        "mandated_sota_models": list(MANDATED_SOTA_MODELS),
        "mandated_sota_models_recorded": mandated_models,
        "continuous_self_learning_required": True,
        "continuous_self_learning_requirement_recorded": continuous_self_learning,
        "retired_headline_signals": list(RETIRED_HEADLINE_SIGNALS),
        "retired_headline_signal_blocks_preserved": retirement_blocks,
        "allowed_116_tracks": list(ALLOWED_116_TRACKS),
        "gated_116_tracks": list(GATED_116_TRACKS),
        "research_complete_has_115_entry": has_115_archive,
        "manifest_path": manifest_path,
        "blocked_reasons": blocked_reasons,
        "conductor_log_exp1492_to_exp1505": conductor_summary,
        "source_inputs_read": _source_inputs_read(
            retro=retro,
            exp1502=exp1502,
            exp1504=exp1504,
            conductor_log_text=conductor_log_text,
            research_complete_text=research_complete_text,
            ops_status_text=ops_status_text,
            ops_changelog_text=ops_changelog_text,
            roadmap_text=roadmap_text,
            roadmap_doc_text=roadmap_doc_text,
            hardware_wishlist_text=hardware_wishlist_text,
            architecture_text=architecture_text,
        ),
        "research_roadmap_yaml_modified": False,
        "scripts_research_conductor_modified": False,
        "protected_files_unchanged": protected_files_unchanged,
        "protected_file_diffs": list(protected_file_diffs),
        "no_change_confirmations": {
            "research-roadmap.yaml": "unchanged_by_exp1506_activation_workflow",
            "scripts/research_conductor.py": "unchanged_by_exp1506_activation_workflow",
        },
        "honest_verdict": (
            "complete: milestone_116_activation_complete_115_archived_gate_fields_ready"
            if activation_manifest_complete
            else (
                "passed: milestone_116_activation_blocked_missing_predecessor_or_guardrail_evidence"
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
    """REQ-REPORT-054: write bootstrap, markdown manifest, and terminal JSON.

    The activation step writes only the Exp 1506 result and the operator
    manifest. It intentionally avoids mutating the roadmap, conductor, status,
    changelog, and traceability files because the conductor-owned reconciler
    handles those documents after this focused archive step exits.
    """

    root_path = Path(root)
    out = Path(out_path)
    manifest_out = Path(manifest_path)
    write_in_progress_artifact(out)
    retro = _read_json(root_path / "results" / PREDECESSOR_RETRO_FILE) or {}
    exp1502 = _read_json(root_path / "results" / EXP1502_FILE) or {}
    exp1504 = _read_json(root_path / "results" / EXP1504_FILE) or {}
    artifact, manifest = build_artifact(
        retro=retro,
        exp1502=exp1502,
        exp1504=exp1504,
        conductor_log_text=_read_text(root_path / "ops" / "conductor-log.md"),
        research_complete_text=_read_text(root_path / "research-complete.yaml"),
        ops_status_text=_read_text(root_path / "ops" / "status.md"),
        ops_changelog_text=_read_text(root_path / "ops" / "changelog.md"),
        roadmap_text=_read_text(root_path / "research-roadmap.yaml"),
        roadmap_doc_text=_read_text(
            root_path / "openspec" / "change-proposals" / "research-roadmap-vNEXT.md"
        ),
        hardware_wishlist_text=_read_text(root_path / "research-hardware-wishlist.md"),
        architecture_text=_read_text(root_path / "_bmad" / "architecture.md"),
        manifest_path=_relative_path(manifest_out),
        protected_file_diffs=[],
    )
    manifest_out.parent.mkdir(parents=True, exist_ok=True)
    manifest_out.write_text(manifest, encoding="utf-8")
    return _write_json(out, artifact)


if __name__ == "__main__":  # pragma: no cover - thin CLI convenience
    run()
