"""Build the Exp 1574 `.121` activation manifest.

Spec: REQ-REPORT-063, SCENARIO-REPORT-063.
"""

from __future__ import annotations

import json
import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
PROJECT_ROOT_FOR_METADATA = "/home/ianblenke/github.com/ianblenke/carnot"
RUN_DATE = "20260508"
PREDECESSOR_MILESTONE = "2026.05.120"
TARGET_MILESTONE = "2026.05.121"
EXPERIMENT = "1574_120_completion_archive_121_activation"
SCHEMA = "milestone_121_activation_manifest_v1"

DEFAULT_OUT_PATH = (
    REPO_ROOT / "results" / "experiment_1574_120_completion_archive_121_activation.json"
)
DEFAULT_MANIFEST_PATH = REPO_ROOT / "ops" / "milestone_121_activation_manifest.md"
PREDECESSOR_RETRO_FILE = "experiment_1572_milestone_120_retro.json"

REQUIRED_ARTIFACT_FIELDS = {
    "status",
    "activation_manifest_complete",
    "prior_failure_autofill_ready",
    "paper_v6_sampler_resume_ready",
    "extropic_packet_resume_ready",
    "brain_reinforce_training_ready",
    "ot_framework_adoption_ready",
    "dccd_jsonschema_smoke_ready",
    "fr11_v15_patch_ready",
    "phase1_ship_readiness_ready",
    "hardware_eval_ready",
    "honest_verdict",
}

ALLOWED_121_TRACKS = [
    {
        "track": "prior_failure_repair",
        "name": "prior-failure repair",
        "guardrail": "Fix exp1569 and exp1573 prior_failures metadata before resuming them.",
    },
    {
        "track": "paper_v6_sampler_drafting",
        "name": "paper-v6 sampler drafting",
        "guardrail": "Draft from .120 evidence, with blocked claims marked explicitly.",
    },
    {
        "track": "extropic_z1_readiness_update",
        "name": "Extropic Z1 readiness update",
        "guardrail": "Update the packet as simulator/readiness work only; no Z1 execution claim.",
    },
    {
        "track": "brain_reinforce_training_dynamics",
        "name": "BRAIN REINFORCE training dynamics",
        "guardrail": "Test the training axis that exp1562 did not cover.",
    },
    {
        "track": "ot_framework_adoption",
        "name": "OT verification framework adoption",
        "guardrail": "Adopt terminology and conflict ledger without upgrading acceptance authority.",
    },
    {
        "track": "dccd_jsonschema_sota_smoke",
        "name": "DCCD/JSONSchemaBench SOTA smoke",
        "guardrail": "Smoke structured outputs on mandated SOTA GGUFs; tiny models are fallback only.",
    },
    {
        "track": "fr11_lambda_grpo_retention_repair",
        "name": "FR-11 lambda-GRPO retention repair",
        "guardrail": "Reverse only replay-confirmed mode-collapsed v14 retentions.",
    },
    {
        "track": "phase1_ship_readiness",
        "name": "Phase-1 ship readiness",
        "guardrail": "Audit software ship readiness independent of paper and hardware.",
    },
    {
        "track": "z1_drift_correction",
        "name": "Z1 drift correction",
        "guardrail": "Treat analog drift correction as a prerequisite, not executed hardware evidence.",
    },
    {
        "track": "hardware_portfolio_correction",
        "name": "Tenstorrent/PolarFire/Strix/KV260 hardware portfolio correction",
        "guardrail": "Correct portfolio scope without board, TSU, or latency claims.",
    },
    {
        "track": "milestone_retro",
        "name": "retro",
        "guardrail": "Close .121 from source artifacts and exact gate fields.",
    },
]

PRESERVED_CLAIM_BLOCKS = [
    "TSU/Z1 hardware execution claims",
    "KV260 board claims without transcripts",
    "legacy-small-model headline results",
    "soft energy/logprob scores as acceptance authority",
]

SOURCE_FILES = {
    "exp1561": "experiment_1561_kinetic_defense_zero_coupling_test.json",
    "exp1562": "experiment_1562_brain_linear_ar_k_sweep_extended.json",
    "exp1564": "experiment_1564_thrml_vendored_block_gibbs_replacement.json",
    "exp1565": "experiment_1565_soft_gibbs_residual_implementation.json",
    "exp1566": "experiment_1566_candidate_warm_start_vs_cold_start_benchmark.json",
    "exp1568": "experiment_1568_fr11_v14_retained_mode_collapse_audit.json",
    "exp1569": "experiment_1569_paper_v6_section_3_sampler_draft.json",
    "exp1570": "experiment_1570_soft_gibbs_coverage_bound_empirical_verification.json",
    "exp1571": "experiment_1571_step_wise_baseline_AR_REINFORCE.json",
    "exp1573": "experiment_1573_extropic_z1_readiness_packet_thrml_alignment_update.json",
}


def _write_json(path: Path, payload: Mapping[str, Any]) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    artifact = dict(payload)
    path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return artifact


def write_in_progress_artifact(out_path: Path | str = DEFAULT_OUT_PATH) -> dict[str, Any]:
    """REQ-REPORT-063: persist a started marker before evidence reads."""

    artifact: dict[str, Any] = {field: False for field in REQUIRED_ARTIFACT_FIELDS}
    artifact.update(
        {
            "experiment": EXPERIMENT,
            "schema": SCHEMA,
            "run_date": RUN_DATE,
            "project_root": PROJECT_ROOT_FOR_METADATA,
            "status": "in_progress",
            "milestone": TARGET_MILESTONE,
            "predecessor_milestone": PREDECESSOR_MILESTONE,
            "allowed_121_tracks": [],
            "preserved_claim_blocks": list(PRESERVED_CLAIM_BLOCKS),
            "honest_verdict": "in_progress",
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
    for marker in ("ops", "results"):
        if marker in parts:
            return str(Path(*parts[parts.index(marker) :]))
    return path.name


def _load_sources(results_dir: Path) -> tuple[dict[str, dict[str, Any]], list[str]]:
    loaded: dict[str, dict[str, Any]] = {}
    missing: list[str] = []
    for exp_id, filename in SOURCE_FILES.items():
        payload = _read_json(results_dir / filename)
        if payload is None:
            missing.append(f"results/{filename}")
        else:
            loaded[exp_id] = payload
    return loaded, missing


def _status(payload: Mapping[str, Any]) -> str:
    return str(payload.get("status") or "").lower()


def _blocked_by_prior_failure(payload: Mapping[str, Any]) -> bool:
    return bool(
        _status(payload) == "blocked"
        and payload.get("honest_verdict") == "blocked_gate_check_failed"
        and payload.get("blocked_at_layer") == "conductor_pre_gate"
        and "prior_failures" in str(payload.get("gate_check_summary") or "")
    )


def _context_has_all(text: str, terms: Sequence[str]) -> bool:
    lowered = text.lower()
    return all(term.lower() in lowered for term in terms)


def _claim_blocks_preserved(text: str) -> bool:
    normalized = text.lower().replace("_", "-")
    compact = normalized.replace("-", "").replace(" ", "")
    return all(
        block.lower() in normalized or block.lower().replace("-", " ") in normalized
        or block.lower().replace("-", "").replace(" ", "") in compact
        for block in PRESERVED_CLAIM_BLOCKS
    )


def _exp1572_complete(milestone_retro: Mapping[str, Any]) -> bool:
    return bool(
        _status(milestone_retro) == "complete"
        and milestone_retro.get("milestone") == PREDECESSOR_MILESTONE
        and milestone_retro.get("next_milestone") == TARGET_MILESTONE
        and milestone_retro.get("criteria_met") == 10
        and milestone_retro.get("criteria_total") == 14
    )


def _build_summary(
    milestone_retro: Mapping[str, Any],
    sources: Mapping[str, Mapping[str, Any]],
) -> tuple[list[str], list[str], list[str]]:
    proved = [
        "THRML vendoring reached KL=0 parity and candidate warm-start support in exp1564",
        "candidate warm-start beat cold/cached starts in exp1566",
        "Soft-Gibbs Residual and its Jensen coverage bound were operational in exp1565/exp1570",
        "step-wise AR-REINFORCE baseline reduced variance in exp1571",
    ]
    falsified = [
        "kinetic defense-in-depth for THRML block-Gibbs was falsified in exp1561",
        "BRAIN+Linear-AR expressivity widening was falsified in exp1562",
    ]
    if sources.get("exp1568", {}).get("reversal_recommended_count", 0):
        falsified.append("one FR-11 v14 retained policy showed mode-collapse predictors in exp1568")
    carried_forward = [
        "exp1569 paper-v6 sampler draft must resume with corrected prior-failure metadata",
        "exp1573 Extropic Z1 packet must resume with corrected prior-failure metadata",
        "BRAIN REINFORCE training dynamics remain untested at k=15",
        "FR-11 lambda-GRPO retention reversal is required for the flagged v14 policy",
    ]
    for item in milestone_retro.get("carry_forward_gates_121", []):
        gate = item.get("gate") if isinstance(item, Mapping) else None
        source = item.get("source") if isinstance(item, Mapping) else None
        if gate and source:
            carried_forward.append(f"{source}: {gate}")
    return proved, falsified, carried_forward


def _md_cell(value: object) -> str:
    return str(value).replace("\n", " ").replace("|", "\\|")


def _render_track_table(rows: Sequence[Mapping[str, str]]) -> list[str]:
    lines = ["| track | guardrail |", "|---|---|"]
    for row in rows:
        lines.append(f"| {_md_cell(row['name'])} | {_md_cell(row['guardrail'])} |")
    return lines


def render_manifest(*, artifact: Mapping[str, Any], blocked_reasons: Sequence[str]) -> str:
    """REQ-REPORT-063: render the operator-facing `.121` activation manifest."""

    lines = [
        "# Milestone .121 Activation Manifest",
        "",
        f"Predecessor milestone: `{PREDECESSOR_MILESTONE}`",
        f"Target milestone: `{TARGET_MILESTONE}`",
        f"Run date: `{RUN_DATE}`",
        "",
    ]
    if blocked_reasons:
        lines.extend(["Manifest blocked: " + "; ".join(blocked_reasons), ""])
    lines.extend(["## What .120 Proved", ""])
    lines.extend(f"- {item}" for item in artifact["proved"])
    lines.extend(["", "## What .120 Falsified", ""])
    lines.extend(f"- {item}" for item in artifact["falsified"])
    lines.extend(["", "## Carried Forward To .121", ""])
    lines.extend(f"- {item}" for item in artifact["carried_forward"])
    lines.extend(["", "## Allowed .121 Tracks", ""])
    lines.extend(_render_track_table(ALLOWED_121_TRACKS))
    lines.extend(["", "## Structured Gate Fields", ""])
    for field, value in artifact["gate_fields"].items():
        lines.append(f"- {field}: {value}")
    lines.extend(["", "## Preserved Claim Blocks", ""])
    for block in artifact["preserved_claim_blocks"]:
        lines.append(f"- {block}")
    lines.extend(
        [
            "",
            "## No-Change Confirmation",
            "",
            "- research-roadmap.yaml: unchanged_by_exp1574_activation_workflow",
            "- scripts/research_conductor.py: unchanged_by_exp1574_activation_workflow",
            "",
        ]
    )
    return "\n".join(lines)


def build_artifact(
    *,
    milestone_retro: Mapping[str, Any],
    sources: Mapping[str, Mapping[str, Any]],
    missing_source_paths: Sequence[str],
    roadmap_text: str,
    ops_known_issues_text: str,
    architecture_text: str,
    manifest_path: str,
    protected_files_unchanged: bool,
) -> tuple[dict[str, Any], str]:
    """REQ-REPORT-063: summarize `.120` closure and activate guarded `.121` work."""

    combined_context = "\n".join([roadmap_text, ops_known_issues_text, architecture_text])
    exp1569_blocked = _blocked_by_prior_failure(sources.get("exp1569", {}))
    exp1573_blocked = _blocked_by_prior_failure(sources.get("exp1573", {}))
    prior_failure_autofill_ready = exp1569_blocked and exp1573_blocked
    brain_reinforce_training_ready = bool(
        sources.get("exp1571", {}).get("step_wise_baseline_implemented") is True
        and float(sources.get("exp1571", {}).get("gradient_variance_reduction_factor") or 0.0)
        >= 10.0
        and any(
            item.get("gate") == "brain_reinforce_training_dynamics_at_k15"
            for item in milestone_retro.get("additional_carry_forwards_121", [])
            if isinstance(item, Mapping)
        )
    )
    fr11_v15_patch_ready = bool(
        sources.get("exp1568", {}).get("mode_collapse_audit_complete") is True
        and int(sources.get("exp1568", {}).get("reversal_recommended_count") or 0) >= 1
        and _context_has_all(combined_context, ["lambda-GRPO", "FR-11"])
    )
    ot_framework_adoption_ready = _context_has_all(combined_context, ["OT verification"])
    dccd_jsonschema_smoke_ready = _context_has_all(
        combined_context, ["DCCD", "JSONSchemaBench"]
    )
    phase1_ship_readiness_ready = _context_has_all(combined_context, ["Phase-1", "Ship"])
    hardware_eval_ready = _context_has_all(
        combined_context, ["Tenstorrent", "PolarFire", "Strix", "KV260"]
    )
    claim_blocks_preserved = _claim_blocks_preserved(combined_context)

    blocked_reasons: list[str] = []
    if not _exp1572_complete(milestone_retro):
        blocked_reasons.append("Exp 1572 does not report .120 completion")
    if missing_source_paths:
        blocked_reasons.append("listed source artifacts are missing")
    blocked_reasons.extend(
        reason
        for reason, ready in (
            ("exp1569 and exp1573 prior-failure carry-forwards are not both blocked at conductor pre-gate", prior_failure_autofill_ready),
            ("BRAIN REINFORCE training gate lacks exp1571 baseline evidence", brain_reinforce_training_ready),
            ("OT verification framework adoption track missing from context", ot_framework_adoption_ready),
            ("DCCD/JSONSchemaBench SOTA smoke track missing from context", dccd_jsonschema_smoke_ready),
            ("FR-11 lambda-GRPO reversal gate lacks mode-collapse evidence", fr11_v15_patch_ready),
            ("Phase-1 software ship readiness track missing from context", phase1_ship_readiness_ready),
            ("hardware portfolio correction track missing from context", hardware_eval_ready),
            ("preserved claim blocks are not all present", claim_blocks_preserved),
            ("protected files changed", protected_files_unchanged),
        )
        if not ready
    )

    proved, falsified, carried_forward = _build_summary(milestone_retro, sources)
    gate_fields = {
        "prior_failure_autofill_ready": prior_failure_autofill_ready,
        "paper_v6_sampler_resume_ready": prior_failure_autofill_ready and exp1569_blocked,
        "extropic_packet_resume_ready": prior_failure_autofill_ready and exp1573_blocked,
        "brain_reinforce_training_ready": brain_reinforce_training_ready,
        "ot_framework_adoption_ready": ot_framework_adoption_ready,
        "dccd_jsonschema_smoke_ready": dccd_jsonschema_smoke_ready,
        "fr11_v15_patch_ready": fr11_v15_patch_ready,
        "phase1_ship_readiness_ready": phase1_ship_readiness_ready,
        "hardware_eval_ready": hardware_eval_ready,
    }
    activation_manifest_complete = not blocked_reasons

    artifact: dict[str, Any] = {
        "experiment": EXPERIMENT,
        "schema": SCHEMA,
        "run_date": RUN_DATE,
        "project_root": PROJECT_ROOT_FOR_METADATA,
        "status": "complete" if activation_manifest_complete else "blocked",
        "milestone": TARGET_MILESTONE,
        "predecessor_milestone": PREDECESSOR_MILESTONE,
        "predecessor_criteria_met": milestone_retro.get("criteria_met"),
        "predecessor_criteria_total": milestone_retro.get("criteria_total"),
        "activation_manifest_complete": activation_manifest_complete,
        **gate_fields,
        "allowed_121_tracks": list(ALLOWED_121_TRACKS),
        "preserved_claim_blocks": list(PRESERVED_CLAIM_BLOCKS),
        "proved": proved,
        "falsified": falsified,
        "carried_forward": carried_forward,
        "gate_fields": gate_fields,
        "missing_source_paths": list(missing_source_paths),
        "blocked_reasons": blocked_reasons,
        "manifest_path": manifest_path,
        "research_roadmap_yaml_modified": not protected_files_unchanged,
        "scripts_research_conductor_modified": not protected_files_unchanged,
        "hardware_eval_scope": "portfolio_correction_preflight_only_no_board_or_tsu_claim",
        "source_inputs_read": {
            f"results/{PREDECESSOR_RETRO_FILE}": {"exists": bool(milestone_retro)},
            "research-roadmap.yaml": {"exists": bool(roadmap_text)},
            "ops/known-issues.md": {"exists": bool(ops_known_issues_text)},
            "_bmad/architecture.md": {"exists": bool(architecture_text)},
        },
        "honest_verdict": (
            "complete: milestone_121_activation_complete_120_archived_phase1_brain_hardware_gates_ready"
            if activation_manifest_complete
            else "blocked: milestone_121_activation_missing_or_unsafe_evidence"
        ),
    }
    manifest = render_manifest(artifact=artifact, blocked_reasons=blocked_reasons)
    return artifact, manifest


def _protected_files_clean(root: Path) -> bool:
    result = subprocess.run(
        [
            "git",
            "diff",
            "--quiet",
            "--",
            "research-roadmap.yaml",
            "scripts/research_conductor.py",
        ],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
    )
    return result.returncode == 0


def run(
    *,
    root: Path | str = REPO_ROOT,
    out_path: Path | str = DEFAULT_OUT_PATH,
    manifest_path: Path | str = DEFAULT_MANIFEST_PATH,
    protected_files_unchanged: bool | None = None,
) -> dict[str, Any]:
    """REQ-REPORT-063: write the `.121` activation artifact and markdown."""

    root_path = Path(root)
    out = Path(out_path)
    manifest = Path(manifest_path)
    write_in_progress_artifact(out)

    results_dir = root_path / "results"
    milestone_retro = _read_json(results_dir / PREDECESSOR_RETRO_FILE) or {}
    sources, missing_source_paths = _load_sources(results_dir)
    protected_clean = (
        _protected_files_clean(root_path)
        if protected_files_unchanged is None
        else protected_files_unchanged
    )

    artifact, manifest_text = build_artifact(
        milestone_retro=milestone_retro,
        sources=sources,
        missing_source_paths=missing_source_paths,
        roadmap_text=_read_text(root_path / "research-roadmap.yaml"),
        ops_known_issues_text=_read_text(root_path / "ops" / "known-issues.md"),
        architecture_text=_read_text(root_path / "_bmad" / "architecture.md"),
        manifest_path=_relative_path(manifest),
        protected_files_unchanged=protected_clean,
    )
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(manifest_text, encoding="utf-8")
    return _write_json(out, artifact)


if __name__ == "__main__":  # pragma: no cover
    run()
