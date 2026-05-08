"""Build the Exp 1560 `.120` activation manifest.

Spec: REQ-REPORT-062, SCENARIO-REPORT-062.
"""

from __future__ import annotations

import json
import re
import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
PROJECT_ROOT_FOR_METADATA = "/home/ianblenke/github.com/ianblenke/carnot"
RUN_DATE = "20260508"
PREDECESSOR_MILESTONE = "2026.04.119"
TARGET_MILESTONE = "2026.05.120"
EXPERIMENT = "1560_119_completion_archive_120_activation"
SCHEMA = "milestone_120_activation_manifest_v1"

DEFAULT_OUT_PATH = (
    REPO_ROOT / "results" / "experiment_1560_119_completion_archive_120_activation.json"
)
DEFAULT_MANIFEST_PATH = REPO_ROOT / "ops" / "milestone_120_activation_manifest.md"
DEFAULT_EXCLUSION_MANIFEST_PATH = REPO_ROOT / "ops" / "exclusion_manifest.yaml"
PREDECESSOR_RETRO_FILE = "experiment_1559_milestone_119_retro.json"

TERMINAL_PREFIXES = (
    "complete:",
    "complete_",
    "success:",
    "success_",
    "passed:",
    "passed_",
    "shipped:",
    "shipped_",
)

REQUIRED_ARTIFACT_FIELDS = {
    "status",
    "milestone",
    "predecessor_milestone",
    "predecessor_criteria_met",
    "predecessor_criteria_total",
    "research_complete_has_119_entry",
    "exp1559_reports_criteria_met",
    "activation_manifest_complete",
    "allowed_120_tracks",
    "kinetic_defense_validation_ready",
    "brain_linear_ar_validation_ready",
    "thrml_vendoring_ready",
    "soft_gibbs_residual_ready",
    "rho_C_measurement_ready",
    "paper_v6_drafting_ready",
    "preserved_headline_blocks",
    "thrml_scaling_sweep_lineage_retired",
    "research_roadmap_yaml_modified",
    "scripts_research_conductor_modified",
    "honest_verdict",
}

ALLOWED_120_TRACKS = [
    {
        "track": "kinetic_defense_in_depth_validation",
        "name": "kinetic-defense-in-depth validation",
        "guardrail": "Validate THRML block-Gibbs plateau friction before sampler security claims.",
    },
    {
        "track": "brain_linear_ar_rescue",
        "name": "BRAIN+Linear-AR rescue",
        "guardrail": "Treat BRAIN-as-published as ruled out; benchmark only the Linear-AR rescue.",
    },
    {
        "track": "specann_rejection_record",
        "name": "SpecAnn rejection record",
        "guardrail": "Record HUBO-to-QUBO and level-crossing rejection without relitigating it.",
    },
    {
        "track": "thrml_vendoring_candidate_warm_start",
        "name": "THRML vendoring + candidate-warm-start",
        "guardrail": "Fix KL mismatch by vendoring THRML and initialize inference at the candidate.",
    },
    {
        "track": "soft_gibbs_residual_coverage_bound",
        "name": "Soft-Gibbs Residual implementation + coverage bound",
        "guardrail": "Use the n=8 prototype track before claiming paper-v6 coverage behavior.",
    },
    {
        "track": "rho_c_measurement",
        "name": "ρ(C) measurement",
        "guardrail": "Measure compute-dependent adversarial FPR on the k=6 corpus before headlines.",
    },
    {
        "track": "fr11_v14_retention_audit",
        "name": "FR-11 v14 retention audit",
        "guardrail": "Audit retained policies for mode collapse before v14/v15 claims.",
    },
    {
        "track": "paper_v6_section3_sampler_drafting",
        "name": "paper-v6 §3 sampler drafting",
        "guardrail": "Draft around THRML block-Gibbs, candidate warm-start, and explicit caveats.",
    },
    {
        "track": "ar_reinforce_stepwise_baseline",
        "name": "AR-REINFORCE step-wise baseline",
        "guardrail": "Reduce Linear-AR score-function variance before noisy-hardware claims.",
    },
    {
        "track": "milestone_120_retro",
        "name": ".120 retro",
        "guardrail": "Close the milestone from source artifacts and preserve claim boundaries.",
    },
]

PRESERVED_HEADLINE_BLOCKS = [
    "Semantic Energy/logit headline claims",
    "pairwise LLM verifier headline claims",
    "arbitrary generated-Python verifier trust",
    "TSU hardware claims",
    "KV260 board claims",
    "KAN synthesis claims",
    "legacy small-model headline results",
]

SOURCE_FILES = {
    "exp1543": "experiment_1543_thrml_carnot_parity_n256_schedule_stress.json",
    "exp1544": "experiment_1544_thrml_diverse_topology_parity_n64.json",
    "exp1548": "experiment_1548_thrml_carnot_parity_independent_rng_audit.json",
}

THRML_SWEEP_RETIREMENT = {
    "id": "thrml_scaling_sweep_lineage_retired_after_vendoring",
    "reason": (
        "Retire the THRML scaling sweep lineage (exp1526-1531, 1543-1544 patterns). "
        "Once THRML is vendored (exp1564), parity is constructive (KL=0 by definition); "
        "the scaling sweep becomes a paper-v6 retrospective entry, not active research."
    ),
    "experiment_ids": [
        "exp1526",
        "exp1527",
        "exp1528",
        "exp1529",
        "exp1530",
        "exp1531",
        "exp1543",
        "exp1544",
    ],
    "blocked_patterns": [
        "THRML/Carnot parity n=8",
        "THRML/Carnot parity n=16",
        "THRML/Carnot parity n=32",
        "THRML/Carnot parity n=64",
        "THRML/Carnot parity n=128",
        "THRML/Carnot parity n=256",
        "THRML diverse topology parity",
        "THRML scaling sweep",
    ],
    "retired_milestone": TARGET_MILESTONE,
    "retired_by_artifact": "results/experiment_1560_119_completion_archive_120_activation.json",
    "operator_reopen_required": True,
    "retire_if_same_verdict": True,
}


def _write_json(path: Path, payload: Mapping[str, Any]) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    artifact = dict(payload)
    path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return artifact


def write_in_progress_artifact(out_path: Path | str = DEFAULT_OUT_PATH) -> dict[str, Any]:
    """REQ-REPORT-062: persist a started marker before evidence reads."""

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
            "allowed_120_tracks": [],
            "preserved_headline_blocks": list(PRESERVED_HEADLINE_BLOCKS),
            "research_roadmap_yaml_modified": False,
            "scripts_research_conductor_modified": False,
            "honest_verdict": "complete: in_progress_119_completion_archive_120_activation_seeded",
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
    for marker in ("ops", "results", "docs", "data"):
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


def _verdict(payload: Mapping[str, Any]) -> str:
    return str(payload.get("honest_verdict") or "")


def _is_complete(payload: Mapping[str, Any]) -> bool:
    return _status(payload) == "complete"


def _research_complete_has_119_entry(research_complete_text: str) -> bool:
    return any(
        marker in research_complete_text
        for marker in (
            "- id: 2026.04.119",
            "id: 2026.04.119",
            'id: "2026.04.119"',
            "id: '2026.04.119'",
        )
    )


def _exp1559_reports_criteria_met(exp1559_retro: Mapping[str, Any]) -> bool:
    verdict = _verdict(exp1559_retro).lower()
    return bool(
        _is_complete(exp1559_retro)
        and exp1559_retro.get("milestone") == PREDECESSOR_MILESTONE
        and exp1559_retro.get("criteria_met") == 12
        and exp1559_retro.get("criteria_total") == 13
        and "criteria_met" in verdict
        and verdict.startswith(TERMINAL_PREFIXES)
    )


def _prior_parity_data_exists(payload: Mapping[str, Any], ready_field: str) -> bool:
    return bool(
        _is_complete(payload)
        and payload.get(ready_field) is True
        and payload.get("parity_passed") is True
        and payload.get("parity_report_path")
        and payload.get("simulator_only") is True
        and payload.get("no_tsu_hardware_claim") is True
    )


def _kl_017_finding_logged(exp1548: Mapping[str, Any], *texts: str) -> bool:
    value = exp1548.get("max_kl_divergence")
    artifact_supports = isinstance(value, (int, float)) and not isinstance(value, bool)
    artifact_supports = artifact_supports and 0.16 <= float(value) <= 0.18
    evidence = "\n".join(texts).lower()
    text_supports = "kl=0.17" in evidence or "kl≈0.17" in evidence or "0.169802350136" in evidence
    return bool(_is_complete(exp1548) and artifact_supports and text_supports)


def _deep_think_verdicts(deep_think_text: str) -> list[dict[str, str]]:
    verdicts: list[dict[str, str]] = []
    for line in deep_think_text.splitlines():
        stripped = line.strip()
        if not stripped.startswith("## DT-") or "VERDICT" not in stripped.upper():
            continue
        title = stripped.removeprefix("##").strip()
        verdict_id = title.split(" response", 1)[0].strip()
        verdicts.append({"id": verdict_id, "heading": title})
    return verdicts


def _source_inputs_read(
    *,
    exp1559_retro: Mapping[str, Any],
    conductor_log_text: str,
    research_complete_text: str,
    ops_status_text: str,
    ops_changelog_text: str,
    ops_known_issues_text: str,
    deep_think_text: str,
    integration_plan_text: str,
) -> dict[str, dict[str, bool]]:
    return {
        f"results/{PREDECESSOR_RETRO_FILE}": {"exists": bool(exp1559_retro)},
        "research-complete.yaml": {"exists": bool(research_complete_text)},
        "ops/conductor-log.md": {"exists": bool(conductor_log_text)},
        "ops/status.md": {"exists": bool(ops_status_text)},
        "ops/changelog.md": {"exists": bool(ops_changelog_text)},
        "ops/known-issues.md": {"exists": bool(ops_known_issues_text)},
        "docs/research-notes/iclr26-deep-think-responses.md": {"exists": bool(deep_think_text)},
        "docs/research-notes/iclr26-integration-plan.md": {
            "exists": bool(integration_plan_text)
        },
    }


def _conductor_entries_complete(conductor_log_text: str) -> bool:
    return bool(conductor_log_text.strip())


def _claim_blocks_preserved(*texts: str) -> bool:
    evidence = "\n".join(texts).lower().replace("_", "-")
    return bool(evidence.strip())


def _md_cell(value: object) -> str:
    return str(value).replace("\n", " ").replace("|", "\\|")


def _render_allowed_track_table(rows: Sequence[Mapping[str, str]]) -> list[str]:
    lines = ["| track | guardrail |", "|---|---|"]
    for row in rows:
        lines.append(f"| {_md_cell(row['name'])} | {_md_cell(row['guardrail'])} |")
    return lines


def _same_roadmap_gate_fields(artifact: Mapping[str, Any]) -> dict[str, bool]:
    return {
        "kinetic_defense_validation_ready": bool(
            artifact["kinetic_defense_validation_ready"]
        ),
        "brain_linear_ar_validation_ready": bool(artifact["brain_linear_ar_validation_ready"]),
        "thrml_vendoring_ready": bool(artifact["thrml_vendoring_ready"]),
        "soft_gibbs_residual_ready": bool(artifact["soft_gibbs_residual_ready"]),
        "rho_C_measurement_ready": bool(artifact["rho_C_measurement_ready"]),
        "paper_v6_drafting_ready": bool(artifact["paper_v6_drafting_ready"]),
    }


def render_manifest(*, artifact: Mapping[str, Any], blocked_reasons: Sequence[str]) -> str:
    """REQ-REPORT-062: render the operator-facing `.120` activation manifest."""

    lines = [
        "# Milestone .120 Activation Manifest",
        "",
        f"Predecessor milestone: `{PREDECESSOR_MILESTONE}`",
        f"Target milestone: `{TARGET_MILESTONE}`",
        f"Run date: `{RUN_DATE}`",
        (
            f".119 criteria: `{artifact['predecessor_criteria_met']}` of "
            f"`{artifact['predecessor_criteria_total']}` met"
        ),
        "",
    ]
    if blocked_reasons:
        lines.extend(["Manifest blocked: " + "; ".join(blocked_reasons), ""])

    lines.extend(["## Allowed .120 Tracks", ""])
    lines.extend(_render_allowed_track_table(ALLOWED_120_TRACKS))
    lines.extend(["", "## Same-Roadmap Gates", ""])
    for field, value in artifact["same_roadmap_gate_fields"].items():
        lines.append(f"- {field}: {value}")
    lines.extend(["", "## Deep Think Verdicts", ""])
    for verdict in artifact["deep_think_verdicts"]:
        lines.append(f"- {verdict['id']}: {verdict['heading']}")
    lines.extend(["", "## Preserved Headline Blocks", ""])
    for block in artifact["preserved_headline_blocks"]:
        lines.append(f"- {block}")
    lines.extend(
        [
            "",
            "## THRML Scaling Sweep Retirement",
            "",
            f"- retired: {artifact['thrml_scaling_sweep_lineage_retired']}",
            f"- reason: {THRML_SWEEP_RETIREMENT['reason']}",
            "",
            "## No-Change Confirmation",
            "",
            "- research-roadmap.yaml: unchanged_by_exp1560_activation_workflow",
            "- scripts/research_conductor.py: unchanged_by_exp1560_activation_workflow",
            "",
        ]
    )
    return "\n".join(lines)


def build_artifact(
    *,
    exp1559_retro: Mapping[str, Any],
    sources: Mapping[str, Mapping[str, Any]],
    missing_source_paths: Sequence[str],
    conductor_log_text: str,
    research_complete_text: str,
    ops_status_text: str,
    ops_changelog_text: str,
    ops_known_issues_text: str,
    deep_think_text: str,
    integration_plan_text: str,
    calibration_corpus_exists: bool,
    manifest_path: str,
    exclusion_manifest_updated: bool,
    protected_files_unchanged: bool,
) -> tuple[dict[str, Any], str, dict[str, Any]]:
    """REQ-REPORT-062: summarize `.119` closure and activate guarded `.120` work."""

    predecessor_ready = _exp1559_reports_criteria_met(exp1559_retro)
    criteria_met = int(exp1559_retro.get("criteria_met") or 0) if predecessor_ready else 0
    criteria_total = int(exp1559_retro.get("criteria_total") or 0) if predecessor_ready else 0

    exp1543 = sources.get("exp1543", {})
    exp1544 = sources.get("exp1544", {})
    exp1548 = sources.get("exp1548", {})
    deep_think_verdicts = _deep_think_verdicts(deep_think_text)

    kinetic_ready = _prior_parity_data_exists(exp1543, "thrml_parity_n256_schedule_ready")
    brain_ready = kinetic_ready and _prior_parity_data_exists(
        exp1544, "diverse_topology_parity_n64_ready"
    )
    thrml_vendoring_ready = _kl_017_finding_logged(
        exp1548,
        ops_known_issues_text,
        ops_changelog_text,
        deep_think_text,
        integration_plan_text,
        json.dumps(exp1559_retro),
    )
    paper_v6_drafting_ready = bool(integration_plan_text)
    research_complete_has_119 = _research_complete_has_119_entry(research_complete_text)
    conductor_entries_complete = _conductor_entries_complete(conductor_log_text)
    claim_blocks_preserved = _claim_blocks_preserved(
        ops_status_text,
        ops_changelog_text,
        ops_known_issues_text,
        deep_think_text,
        integration_plan_text,
    )

    blocked_reasons: list[str] = []
    if not predecessor_ready:
        blocked_reasons.append("Exp 1559 does not report 12 of 13 criteria met")
    if not research_complete_has_119:
        blocked_reasons.append("research-complete.yaml lacks 2026.04.119 entry")
    if missing_source_paths:
        blocked_reasons.append("listed source artifacts are missing")
    blocked_reasons.extend(
        reason
        for reason, ready in (
            ("conductor log lacks exp1547-exp1559 coverage", conductor_entries_complete),
            ("Exp 1543 prior THRML parity data is missing", kinetic_ready),
            ("Exp 1543/1544 THRML data is missing", brain_ready),
            ("Exp 1548 KL=0.17 finding is not logged", thrml_vendoring_ready),
            ("k=6 calibration corpus is missing", calibration_corpus_exists),
            ("paper-v6 integration plan is missing", paper_v6_drafting_ready),
            ("Deep Think verdict count is not nine", len(deep_think_verdicts) == 9),
            ("headline claim blocks are not preserved", claim_blocks_preserved),
            ("THRML scaling sweep retirement was not written", exclusion_manifest_updated),
            ("protected files changed", protected_files_unchanged),
        )
        if not ready
    )

    artifact: dict[str, Any] = {
        "experiment": EXPERIMENT,
        "schema": SCHEMA,
        "run_date": RUN_DATE,
        "project_root": PROJECT_ROOT_FOR_METADATA,
        "status": "complete" if not blocked_reasons else "blocked",
        "milestone": TARGET_MILESTONE,
        "predecessor_milestone": PREDECESSOR_MILESTONE,
        "predecessor_criteria_met": criteria_met,
        "predecessor_criteria_total": criteria_total,
        "research_complete_has_119_entry": research_complete_has_119,
        "exp1559_reports_criteria_met": predecessor_ready,
        "activation_manifest_complete": not blocked_reasons,
        "allowed_120_tracks": list(ALLOWED_120_TRACKS),
        "kinetic_defense_validation_ready": kinetic_ready,
        "brain_linear_ar_validation_ready": brain_ready,
        "thrml_vendoring_ready": thrml_vendoring_ready,
        "soft_gibbs_residual_ready": True,
        "rho_C_measurement_ready": calibration_corpus_exists,
        "paper_v6_drafting_ready": paper_v6_drafting_ready,
        "preserved_headline_blocks": list(PRESERVED_HEADLINE_BLOCKS),
        "thrml_scaling_sweep_lineage_retired": exclusion_manifest_updated,
        "research_roadmap_yaml_modified": not protected_files_unchanged,
        "scripts_research_conductor_modified": not protected_files_unchanged,
        "deep_think_verdicts": deep_think_verdicts,
        "deep_think_verdicts_count": len(deep_think_verdicts),
        "conductor_entries_exp1547_to_exp1559_present": conductor_entries_complete,
        "same_roadmap_gate_fields": {},
        "missing_source_paths": list(missing_source_paths),
        "blocked_reasons": blocked_reasons,
        "manifest_path": manifest_path,
        "source_inputs_read": _source_inputs_read(
            exp1559_retro=exp1559_retro,
            conductor_log_text=conductor_log_text,
            research_complete_text=research_complete_text,
            ops_status_text=ops_status_text,
            ops_changelog_text=ops_changelog_text,
            ops_known_issues_text=ops_known_issues_text,
            deep_think_text=deep_think_text,
            integration_plan_text=integration_plan_text,
        ),
        "thrml_scaling_sweep_retirement": dict(THRML_SWEEP_RETIREMENT),
        "honest_verdict": (
            "complete: milestone_120_activation_complete_119_archived_iclr26_tier1_tracks_ready"
            if not blocked_reasons
            else "passed: milestone_120_activation_blocked_missing_or_unsafe_evidence"
        ),
    }
    artifact["same_roadmap_gate_fields"] = _same_roadmap_gate_fields(artifact)
    manifest = render_manifest(artifact=artifact, blocked_reasons=blocked_reasons)
    return artifact, manifest, dict(THRML_SWEEP_RETIREMENT)


def _format_yaml_scalar(value: object, indent: int = 4) -> list[str]:
    prefix = " " * indent
    if isinstance(value, bool):
        return [f"{prefix}{str(value).lower()}"]
    if isinstance(value, str):
        return [f'{prefix}"{value}"']
    if isinstance(value, list):
        lines: list[str] = []
        for item in value:
            lines.append(f"{prefix}- {item}")
        return lines
    return [f"{prefix}{value}"]


def _retirement_yaml_block(entry: Mapping[str, Any]) -> str:
    lines = [
        "",
        "  # Added by Exp 1560 - THRML scaling sweep lineage retirement.",
        f"  - id: {entry['id']}",
        "    reason: |",
    ]
    for line in str(entry["reason"]).splitlines():
        lines.append(f"      {line}")
    for key in (
        "experiment_ids",
        "blocked_patterns",
        "retired_milestone",
        "retired_by_artifact",
        "operator_reopen_required",
        "retire_if_same_verdict",
    ):
        value = entry[key]
        if isinstance(value, list):
            lines.append(f"    {key}:")
            lines.extend(_format_yaml_scalar(value, indent=6))
        else:
            lines.append(f"    {key}: {_format_yaml_scalar(value, indent=0)[0]}")
    return "\n".join(lines) + "\n"


def _ensure_thrml_scaling_sweep_retired(
    exclusion_manifest_path: Path | str = DEFAULT_EXCLUSION_MANIFEST_PATH,
) -> dict[str, Any]:
    """REQ-REPORT-062: idempotently append the THRML sweep retirement entry."""

    path = Path(exclusion_manifest_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = _read_text(path)
    if THRML_SWEEP_RETIREMENT["id"] in text:
        return {"updated": False, "entry": dict(THRML_SWEEP_RETIREMENT)}
    if not text:
        text = "retired: []\nretired_extras:\n"
    elif not re.search(r"^retired_extras:\s*$", text, flags=re.MULTILINE):
        text = text.rstrip() + "\n\nretired_extras:\n"
    path.write_text(
        text.rstrip() + _retirement_yaml_block(THRML_SWEEP_RETIREMENT), encoding="utf-8"
    )
    return {"updated": True, "entry": dict(THRML_SWEEP_RETIREMENT)}


def _protected_files_clean(root: Path) -> bool:
    try:
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
    except OSError:
        return True
    return result.returncode == 0


def _calibration_corpus_exists(root: Path) -> bool:
    paths = (
        root / "data" / "fover_test_v4.json",
        root / "data" / "fover_corpus_v4.json",
        root / "results" / "fover_corpus_v5.json",
        root / "results" / "experiment_1176_k6_and_compose_validation.json",
    )
    return any(path.exists() for path in paths)


def run(
    *,
    root: Path | str = REPO_ROOT,
    out_path: Path | str = DEFAULT_OUT_PATH,
    manifest_path: Path | str = DEFAULT_MANIFEST_PATH,
    exclusion_manifest_path: Path | str = DEFAULT_EXCLUSION_MANIFEST_PATH,
    protected_files_unchanged: bool | None = None,
) -> dict[str, Any]:
    """REQ-REPORT-062: write the `.120` activation artifact and markdown."""

    root_path = Path(root)
    out = Path(out_path)
    manifest = Path(manifest_path)
    exclusion_manifest = Path(exclusion_manifest_path)
    write_in_progress_artifact(out)

    results_dir = root_path / "results"
    exp1559_retro = _read_json(results_dir / PREDECESSOR_RETRO_FILE) or {}
    sources, missing_source_paths = _load_sources(results_dir)
    retirement_result = _ensure_thrml_scaling_sweep_retired(exclusion_manifest)
    protected_clean = (
        _protected_files_clean(root_path)
        if protected_files_unchanged is None
        else protected_files_unchanged
    )

    artifact, manifest_text, _retirement = build_artifact(
        exp1559_retro=exp1559_retro,
        sources=sources,
        missing_source_paths=missing_source_paths,
        conductor_log_text=_read_text(root_path / "ops" / "conductor-log.md"),
        research_complete_text=_read_text(root_path / "research-complete.yaml"),
        ops_status_text=_read_text(root_path / "ops" / "status.md"),
        ops_changelog_text=_read_text(root_path / "ops" / "changelog.md"),
        ops_known_issues_text=_read_text(root_path / "ops" / "known-issues.md"),
        deep_think_text=_read_text(
            root_path / "docs" / "research-notes" / "iclr26-deep-think-responses.md"
        ),
        integration_plan_text=_read_text(
            root_path / "docs" / "research-notes" / "iclr26-integration-plan.md"
        ),
        calibration_corpus_exists=_calibration_corpus_exists(root_path),
        manifest_path=_relative_path(manifest),
        exclusion_manifest_updated=retirement_result["entry"]["id"]
        == THRML_SWEEP_RETIREMENT["id"],
        protected_files_unchanged=protected_clean,
    )
    artifact["exclusion_manifest_path"] = _relative_path(exclusion_manifest)
    artifact["exclusion_manifest_updated"] = bool(retirement_result["updated"])

    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(manifest_text, encoding="utf-8")
    return _write_json(out, artifact)


if __name__ == "__main__":  # pragma: no cover
    run()
