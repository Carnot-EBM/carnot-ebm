"""Exp 1577 Extropic Z1 readiness packet THRML alignment.

This module updates the Z1 readiness packet after THRML vendoring while keeping
the evidence boundary explicit: current evidence is simulator-only, and any Z1,
XTR, or TSU hardware claim remains blocked until authenticated device evidence
and detailed-balance drift correction exist.

Spec refs: REQ-REPORT-065, SCENARIO-REPORT-065.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
PROJECT_ROOT_FOR_METADATA = "/home/ianblenke/github.com/ianblenke/carnot"
RUN_DATE = "20260508"
MILESTONE = "2026.05.121"
EXPERIMENT = "1577_extropic_z1_readiness_packet_thrml_alignment_resumed"
SCHEMA = "extropic_z1_readiness_packet_thrml_alignment_v1"

DEFAULT_OUT_PATH = (
    REPO_ROOT
    / "results"
    / "experiment_1577_extropic_z1_readiness_packet_thrml_alignment_resumed.json"
)
DEFAULT_PACKET_PATH = (
    REPO_ROOT / "docs" / "research-notes" / "extropic-z1-readiness-packet-2026-05-121.md"
)

SOURCE_FILES = {
    "exp1545": "experiment_1545_extropic_z1_access_readiness_packet.json",
    "exp1564": "experiment_1564_thrml_vendored_block_gibbs_replacement.json",
    "exp1565": "experiment_1565_soft_gibbs_residual_implementation.json",
    "exp1566": "experiment_1566_candidate_warm_start_vs_cold_start_benchmark.json",
}

REQUIRED_ARTIFACT_FIELDS = {
    "status",
    "packet_path",
    "extropic_z1_packet_updated",
    "thrml_vendoring_reflected",
    "analog_drift_correction_required",
    "simulator_only_no_hardware_claim",
    "honest_verdict",
}

COMPLETE_VERDICT = (
    "complete: extropic_z1_readiness_packet_thrml_alignment_simulator_only_drift_gate_added"
)

FORBIDDEN_CLAIM_PHRASES = (
    "authenticated z1 access",
    "device latency measured",
    "hardware execution completed",
    "tsu hardware execution",
    "xtr hardware execution",
    "z1 hardware execution",
)


def _write_json(path: Path, payload: Mapping[str, Any]) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    artifact = dict(payload)
    path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return artifact


def write_in_progress_artifact(out_path: Path | str = DEFAULT_OUT_PATH) -> dict[str, Any]:
    """Write the bootstrap artifact required by REQ-REPORT-065."""

    artifact: dict[str, Any] = {
        "experiment": EXPERIMENT,
        "schema": SCHEMA,
        "run_date": RUN_DATE,
        "project_root": PROJECT_ROOT_FOR_METADATA,
        "status": "in_progress",
        "packet_path": None,
        "extropic_z1_packet_updated": False,
        "thrml_vendoring_reflected": False,
        "analog_drift_correction_required": False,
        "simulator_only_no_hardware_claim": True,
        "honest_verdict": "in_progress",
    }
    return _write_json(Path(out_path), artifact)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _relative_path(path: Path, repo_root: Path) -> str:
    return str(path.resolve().relative_to(repo_root.resolve()))


def _expect(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def validate_sources(
    *,
    exp1545: Mapping[str, Any],
    exp1564: Mapping[str, Any],
    exp1565: Mapping[str, Any],
    exp1566: Mapping[str, Any],
) -> bool:
    """Validate source artifacts for the Exp 1577 update."""

    _expect(exp1545.get("status") == "complete", "Exp1545 must be complete")
    _expect(
        exp1545.get("extropic_z1_readiness_packet_ready") is True,
        "Exp1545 readiness packet must be ready",
    )
    _expect(
        exp1545.get("no_hardware_execution_claim") is True,
        "Exp1545 must preserve no_hardware_execution_claim",
    )
    _expect(exp1564.get("status") == "complete", "Exp1564 must be complete")
    _expect(
        exp1564.get("thrml_vendoring_complete") is True
        and exp1564.get("thrml_version") == "0.1.3"
        and exp1564.get("simulator_only") is True
        and exp1564.get("no_tsu_hardware_claim") is True,
        "Exp1564 must report THRML 0.1.3 vendoring with simulator-only no-TSU status",
    )
    _expect(
        exp1564.get("candidate_warm_start_implemented") is True,
        "Exp1564 must include candidate warm-start implementation",
    )
    _expect(
        exp1565.get("status") == "complete"
        and exp1565.get("soft_gibbs_residual_implemented") is True
        and exp1565.get("soft_brs_decay_confirmed") is True,
        "Exp1565 must report Soft-Gibbs Residual readiness",
    )
    _expect(
        exp1566.get("status") == "complete"
        and exp1566.get("candidate_warm_start_validated") is True
        and exp1566.get("recommended_deployment_policy") == "candidate_warm_start",
        "Exp1566 must validate candidate warm-start deployment policy",
    )
    return True


def render_packet(
    *,
    exp1545: Mapping[str, Any],
    exp1564: Mapping[str, Any],
    exp1565: Mapping[str, Any],
    exp1566: Mapping[str, Any],
) -> str:
    """Render the updated readiness packet markdown."""

    return "\n".join(
        [
            "# Extropic Z1 readiness packet - .121 THRML alignment",
            "",
            "Spec refs: REQ-REPORT-065, SCENARIO-REPORT-065.",
            "",
            "## Status",
            "",
            "- Current status: simulator-only readiness update.",
            "- No Z1, XTR, or TSU hardware access is claimed.",
            "- No Extropic device latency, sample-quality, SDK, firmware, or transcript evidence is claimed.",
            f"- Prior packet: `{exp1545.get('readiness_packet_path')}`.",
            "",
            "## THRML vendoring alignment",
            "",
            f"- THRML {exp1564.get('thrml_version')} is vendored under {exp1564.get('thrml_license', 'Apache-2.0')}.",
            "- Carnot uses the vendored THRML block-Gibbs transition as the simulator reference.",
            f"- Vendoring complete: {exp1564.get('thrml_vendoring_complete')}.",
            f"- KL to THRML after vendoring: {exp1564.get('kl_to_thrml_after_vendoring')}.",
            "- This is a software/simulator alignment fact, not a hardware-execution result.",
            "",
            "## Candidate warm-start API requirement",
            "",
            "- The required policy is candidate warm-start for every verifier request.",
            "- Future THRML or SDK-backed Z1 evaluation must accept the current verifier payload as `{prompt, candidate}`.",
            "- The sampler initialization state must be `bits(candidate)`, not uniform cold-start and not cached state from another prompt.",
            f"- Exp 1566 deployment policy: {exp1566.get('recommended_deployment_policy')}.",
            f"- Cold-start accuracy drop at K=100: {exp1566.get('cold_start_accuracy_drop_percent_at_k100')} percent.",
            "",
            "## Soft-Gibbs Residual relevance",
            "",
            "- Soft-Gibbs Residual remains relevant because hard residual rejection can have an empty operational intersection.",
            f"- Soft residual implemented: {exp1565.get('soft_gibbs_residual_implemented')}.",
            f"- Soft BRS decay confirmed: {exp1565.get('soft_brs_decay_confirmed')}.",
            f"- Hard BRS acceptance rate on the contradictory fixture: {exp1565.get('hard_brs_acceptance_rate')}.",
            "- A future hardware packet should keep residual conditioning separate from any hardware sampling evidence.",
            "",
            "## pre-silicon correction prerequisites",
            "",
            "- Detailed-balance drift correction is required before any Z1 claim.",
            "- The explicit prerequisite name is detailed-balance drift correction.",
            "- The correction must account for analog beta drift across die, temperature, and voltage before Carnot compares Z1 samples to the THRML simulator reference.",
            "- The prerequisite is software correction plus validation on synthetic drift before any authenticated Z1 packet can move from readiness to hardware evidence.",
            "",
            "## Claim boundary",
            "",
            "- Simulator-only evidence can support API readiness, benchmark manifests, and transcript requirements.",
            "- It cannot support Z1, XTR, TSU, device-latency, device-throughput, or sample-quality hardware claims.",
            "- The next unblocker is an authenticated device transcript plus the detailed-balance drift-correction prerequisite above.",
            "",
        ]
    )


def validate_packet_text(packet_text: str) -> bool:
    """Validate packet sections and forbid hardware-claim drift."""

    for phrase in (
        "THRML 0.1.3",
        "candidate warm-start",
        "Soft-Gibbs Residual",
        "pre-silicon correction prerequisites",
        "detailed-balance drift correction",
        "simulator-only",
        "No Z1, XTR, or TSU hardware access is claimed",
    ):
        _expect(phrase in packet_text, f"packet missing required phrase: {phrase}")
    lowered = packet_text.lower()
    for phrase in FORBIDDEN_CLAIM_PHRASES:
        _expect(phrase not in lowered, f"packet contains forbidden hardware claim phrase: {phrase}")
    return True


def validate_artifact(artifact: Mapping[str, Any]) -> bool:
    """Validate the terminal JSON artifact required by REQ-REPORT-065."""

    missing = REQUIRED_ARTIFACT_FIELDS - set(artifact)
    _expect(not missing, f"missing required fields: {sorted(missing)}")
    _expect(artifact.get("status") == "complete", "status must be complete")
    _expect(
        artifact.get("extropic_z1_packet_updated") is True,
        "extropic_z1_packet_updated must be true",
    )
    _expect(
        artifact.get("thrml_vendoring_reflected") is True,
        "thrml_vendoring_reflected must be true",
    )
    _expect(
        artifact.get("analog_drift_correction_required") is True,
        "analog_drift_correction_required must be true",
    )
    _expect(
        artifact.get("simulator_only_no_hardware_claim") is True,
        "simulator_only_no_hardware_claim must be true",
    )
    _expect(artifact.get("honest_verdict") == COMPLETE_VERDICT, "honest_verdict is invalid")
    return True


def build_artifact(
    *,
    exp1545: Mapping[str, Any],
    exp1564: Mapping[str, Any],
    exp1565: Mapping[str, Any],
    exp1566: Mapping[str, Any],
    packet_path: str,
) -> tuple[dict[str, Any], str]:
    """Build and validate the Exp 1577 packet and terminal artifact."""

    validate_sources(exp1545=exp1545, exp1564=exp1564, exp1565=exp1565, exp1566=exp1566)
    packet_text = render_packet(exp1545=exp1545, exp1564=exp1564, exp1565=exp1565, exp1566=exp1566)
    validate_packet_text(packet_text)
    artifact: dict[str, Any] = {
        "experiment": EXPERIMENT,
        "schema": SCHEMA,
        "run_date": RUN_DATE,
        "project_root": PROJECT_ROOT_FOR_METADATA,
        "milestone": MILESTONE,
        "status": "complete",
        "packet_path": packet_path,
        "extropic_z1_packet_updated": True,
        "thrml_vendoring_reflected": True,
        "analog_drift_correction_required": True,
        "simulator_only_no_hardware_claim": True,
        "source_artifacts": [f"results/{filename}" for filename in SOURCE_FILES.values()],
        "pre_silicon_prerequisites": [
            "detailed-balance drift correction",
            "synthetic analog-drift validation",
            "authenticated Z1/XTR/TSU transcript before any hardware claim",
        ],
        "hardware_claims": {
            "z1_access_claimed": False,
            "xtr_access_claimed": False,
            "tsu_access_claimed": False,
            "device_latency_claimed": False,
            "device_sample_quality_claimed": False,
        },
        "honest_verdict": COMPLETE_VERDICT,
    }
    validate_artifact(artifact)
    return artifact, packet_text


def run(*, repo_root: Path | str = REPO_ROOT) -> dict[str, Any]:
    """Write the Exp 1577 markdown packet and terminal result JSON."""

    root = Path(repo_root)
    out_path = (
        root
        / "results"
        / "experiment_1577_extropic_z1_readiness_packet_thrml_alignment_resumed.json"
    )
    packet_path = root / "docs" / "research-notes" / "extropic-z1-readiness-packet-2026-05-121.md"
    write_in_progress_artifact(out_path)
    results_dir = root / "results"
    artifact, packet_text = build_artifact(
        exp1545=_read_json(results_dir / SOURCE_FILES["exp1545"]),
        exp1564=_read_json(results_dir / SOURCE_FILES["exp1564"]),
        exp1565=_read_json(results_dir / SOURCE_FILES["exp1565"]),
        exp1566=_read_json(results_dir / SOURCE_FILES["exp1566"]),
        packet_path=_relative_path(packet_path, root),
    )
    packet_path.parent.mkdir(parents=True, exist_ok=True)
    packet_path.write_text(packet_text, encoding="utf-8")
    return _write_json(out_path, artifact)


__all__ = [
    "COMPLETE_VERDICT",
    "DEFAULT_OUT_PATH",
    "DEFAULT_PACKET_PATH",
    "REPO_ROOT",
    "REQUIRED_ARTIFACT_FIELDS",
    "build_artifact",
    "render_packet",
    "run",
    "validate_artifact",
    "validate_packet_text",
    "validate_sources",
    "write_in_progress_artifact",
]


if __name__ == "__main__":  # pragma: no cover
    run()
