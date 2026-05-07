"""Build the Exp 1483 HalluGuard-style risk-bound fit audit.

Carnot has useful local evidence for hallucination-risk decomposition, but it
does not implement HalluGuard's formal NTK/certification assumptions.  This
module turns the available .113/.114 telemetry, FoVer-style labels,
BEAVER-lite bounds, and verifier outcomes into an honest fit audit: what can
be mapped into data-driven and reasoning-driven risk fields, and what must
remain explicitly unclaimed.

Spec: REQ-REPORT-051, SCENARIO-REPORT-051.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Mapping

REPO_ROOT = Path(__file__).resolve().parents[3]
RUN_DATE = "20260507"
PROJECT_ROOT_FOR_METADATA = "/home/ianblenke/github.com/ianblenke/carnot"
SCHEMA = "halluguard_risk_bound_fit_audit_v1"
EXPERIMENT = "1483_halluguard_risk_bound_fit_audit"
OUTPUT_FILENAME = "experiment_1483_halluguard_risk_bound_fit_audit.json"
AUDIT_NOTE_REL = "docs/research-notes/halluguard_carnot_risk_bound_fit.md"

DEFAULT_OUT_PATH = REPO_ROOT / "results" / OUTPUT_FILENAME
DEFAULT_AUDIT_NOTE_PATH = REPO_ROOT / AUDIT_NOTE_REL

REQUIRED_ARTIFACT_FIELDS = {
    "status",
    "source_artifacts",
    "risk_decomposition_complete",
    "data_driven_fields_available",
    "reasoning_driven_fields_available",
    "implemented_assumptions",
    "missing_assumptions",
    "claim_allowed",
    "audit_note_path",
    "honest_verdict",
}

DATA_DRIVEN_FIELDS_AVAILABLE = [
    "live_sota_model_inference_used",
    "topk_logprobs_available",
    "logits_available",
    "known_verifier_label",
    "balanced_label_counts",
    "telemetry_adversarial_validity_verdict",
    "missing_evidence_caveats",
]

REASONING_DRIVEN_FIELDS_AVAILABLE = [
    "bound_is_sound",
    "unsafe_mass_bounds",
    "empirical_violation_rates",
    "prefix_closed_constraints",
    "reasoning_step_validity_limitations",
]

JSON_SOURCE_FIELDS: dict[str, tuple[str, tuple[str, ...]]] = {
    "results/experiment_1468_live_sota_logprob_telemetry_preflight.json": (
        "milestone_113_live_telemetry_preflight",
        (
            "status",
            "honest_verdict",
            "live_sota_model_inference_used",
            "topk_logprobs_available",
            "logits_available",
            "telemetry_cases_completed",
        ),
    ),
    "results/experiment_1470_beaver_lite_deterministic_bound_smoke.json": (
        "milestone_113_beaver_lite_bound_smoke",
        (
            "status",
            "honest_verdict",
            "bound_is_sound",
            "unsafe_mass_bounds",
            "empirical_violation_rates",
            "mock_or_live_logprobs",
        ),
    ),
    "results/experiment_1473_live_telemetry_adversarial_validity_audit.json": (
        "milestone_113_telemetry_validity_audit",
        (
            "status",
            "honest_verdict",
            "claim_allowed",
            "telemetry_validity_verdict",
            "superficial_baseline_results",
        ),
    ),
    "results/experiment_1480_live_sota_balanced_telemetry_v2.json": (
        "milestone_114_balanced_live_telemetry",
        (
            "status",
            "honest_verdict",
            "live_sota_model_inference_used",
            "topk_logprobs_available",
            "logits_available",
            "balanced_label_counts",
            "telemetry_cases_completed",
            "telemetry_manifest_path",
        ),
    ),
    "results/experiment_1482_beaver_lite_live_prefix_bound_calibration.json": (
        "milestone_114_beaver_lite_prefix_calibration",
        (
            "status",
            "honest_verdict",
            "bound_is_sound",
            "constraints_evaluated",
            "unsafe_mass_bounds",
            "empirical_violation_rates",
            "prefix_closed_constraints",
            "mock_or_live_logprobs",
        ),
    ),
}

TEXT_SOURCE_FIELDS = {
    "research-references.md": "halluguard_reference_entry",
    "docs/research-notes/paper_v6_anchored_claim_matrix.md": "claim_boundary_matrix",
}

IMPLEMENTED_ASSUMPTIONS = [
    "Live local SOTA telemetry artifacts report top-k logprobs and logits availability.",
    "FoVer-style verifier labels are present in the balanced telemetry manifest.",
    "BEAVER-lite artifacts report sound prefix-bound checks over live logprob provenance.",
    "Adversarial telemetry validity audit blocks headline telemetry claims under superficial-confound checks.",
    "Paper-v6 claim boundaries already avoid broad universal verifier and reproduction claims.",
]

MISSING_ASSUMPTIONS = [
    "HalluGuard NTK feature construction is not implemented in Carnot.",
    "Formal HalluGuard DHRB/RHRB theorem assumptions are not checked locally.",
    "No calibrated data-driven hallucination risk bound is certified over the deployment distribution.",
    "No full reasoning-step certificate proves every chain-of-thought or latent reasoning step valid.",
    "Current BEAVER-lite bounds cover terminal/prefix-closed constraints, not arbitrary hallucination semantics.",
    "The telemetry signal is known to be vulnerable to superficial or mechanical confounds from Exp 1473.",
    "The .114 balanced telemetry run used one live model family, not a complete HalluGuard reproduction suite.",
]

ALLOWED_WORDING = [
    (
        "Carnot has a HalluGuard-style fit audit that separates evidence-availability "
        "risk from reasoning-step risk using existing telemetry, FoVer-style labels, "
        "BEAVER-lite bounds, and verifier outcomes."
    ),
    (
        "Full HalluGuard reproduction is not claimed: NTK/certification assumptions "
        "and complete DHRB/RHRB formal checks are unimplemented."
    ),
]


def write_in_progress_artifact(out_path: str | Path = DEFAULT_OUT_PATH) -> dict[str, Any]:
    """REQ-REPORT-051: write the durable startup artifact before analysis."""

    artifact: dict[str, Any] = {
        "schema": SCHEMA,
        "experiment": EXPERIMENT,
        "run_date": RUN_DATE,
        "project_root": PROJECT_ROOT_FOR_METADATA,
        "spec": ["REQ-REPORT-051", "SCENARIO-REPORT-051"],
        "status": "in_progress",
        "source_artifacts": [],
        "risk_decomposition_complete": False,
        "data_driven_fields_available": [],
        "reasoning_driven_fields_available": [],
        "implemented_assumptions": [],
        "missing_assumptions": [],
        "claim_allowed": False,
        "audit_note_path": "",
        "honest_verdict": "in_progress",
    }
    return _write_json(Path(out_path), artifact)


def load_source_inputs(root: str | Path = REPO_ROOT) -> dict[str, Any]:
    """Load only the artifact fields needed for the HalluGuard fit audit."""

    root_path = Path(root)
    json_payloads = {rel_path: _read_json(root_path / rel_path) for rel_path in JSON_SOURCE_FIELDS}
    text_payloads = {
        rel_path: (root_path / rel_path).read_text(encoding="utf-8")
        for rel_path in TEXT_SOURCE_FIELDS
    }
    manifest_rel = str(
        json_payloads["results/experiment_1480_live_sota_balanced_telemetry_v2.json"].get(
            "telemetry_manifest_path", "results/live_sota_balanced_telemetry_manifest_1480.jsonl"
        )
    )
    return {
        "json": json_payloads,
        "text": text_payloads,
        "manifest_path": manifest_rel,
        "manifest_summary": summarize_manifest(root_path / manifest_rel),
        "halluguard_reference": extract_halluguard_reference_summary(
            text_payloads["research-references.md"]
        ),
        "claim_matrix_summary": summarize_claim_matrix(
            text_payloads["docs/research-notes/paper_v6_anchored_claim_matrix.md"]
        ),
    }


def extract_halluguard_reference_summary(text: str) -> dict[str, Any]:
    """Extract the local HalluGuard planning entry without copying the paper."""

    lines = text.splitlines()
    start = next(
        (index for index, line in enumerate(lines) if line.startswith("### HalluGuard")),
        -1,
    )
    end = next(
        (
            index
            for index, line in enumerate(lines[start + 1 :], start + 1)
            if line.startswith("### ")
        ),
        len(lines),
    )
    entry = "\n".join(lines[start:end]) if start >= 0 else ""
    lower_entry = entry.lower()
    return {
        "entry_found": start >= 0,
        "risk_decomposition_reference_found": (
            "data-driven" in lower_entry and "reasoning-driven" in lower_entry
        ),
        "full_reproduction_warning_found": (
            "do not claim full" in lower_entry and "ntk" in lower_entry
        ),
        "entry_line_count": len(entry.splitlines()) if entry else 0,
    }


def summarize_claim_matrix(text: str) -> dict[str, Any]:
    """Summarize whether the paper claim matrix already uses negative boundaries."""

    lower_text = text.lower()
    return {
        "negative_boundaries_present": "does not claim" in lower_text,
        "full_reproduction_claim_absent": "full halluguard reproduction" not in lower_text,
    }


def summarize_manifest(path: str | Path) -> dict[str, Any]:
    """Summarize telemetry-row evidence used as data-driven risk input."""

    rows = [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    fields_seen = sorted({field for row in rows for field in row})
    return {
        "row_count": len(rows),
        "known_verifier_label_available": "known_verifier_label" in fields_seen,
        "known_verifier_label_values": sorted(
            {int(row["known_verifier_label"]) for row in rows if "known_verifier_label" in row}
        ),
        "token_logprobs_available_rows": sum(
            1 for row in rows if row.get("token_logprobs_available") is True
        ),
        "topk_available_rows": sum(
            1 for row in rows if row.get("topk_alternatives_available") is True
        ),
        "format_valid_rows": sum(1 for row in rows if row.get("format_valid") is True),
        "fields_seen": fields_seen,
    }


def build_artifact(inputs: Mapping[str, Any], *, audit_note_path: str) -> dict[str, Any]:
    """Build the terminal Exp 1483 fit-audit artifact from loaded sources."""

    json_sources: Mapping[str, Mapping[str, Any]] = inputs["json"]
    exp1468 = json_sources["results/experiment_1468_live_sota_logprob_telemetry_preflight.json"]
    exp1470 = json_sources["results/experiment_1470_beaver_lite_deterministic_bound_smoke.json"]
    exp1473 = json_sources["results/experiment_1473_live_telemetry_adversarial_validity_audit.json"]
    exp1480 = json_sources["results/experiment_1480_live_sota_balanced_telemetry_v2.json"]
    exp1482 = json_sources["results/experiment_1482_beaver_lite_live_prefix_bound_calibration.json"]
    manifest_summary: Mapping[str, Any] = inputs["manifest_summary"]

    data_proxy = {
        "local_definition": (
            "Carnot data-driven hallucination risk means risk from missing, stale, "
            "or confounded evidence about whether the generated answer is grounded."
        ),
        "live_sota_model_inference_used": bool(
            exp1480.get("live_sota_model_inference_used")
            and exp1468.get("live_sota_model_inference_used")
        ),
        "topk_logprobs_available": bool(
            exp1480.get("topk_logprobs_available") and exp1468.get("topk_logprobs_available")
        ),
        "logits_available": bool(
            exp1480.get("logits_available") and exp1468.get("logits_available")
        ),
        "known_verifier_label_available": bool(manifest_summary["known_verifier_label_available"]),
        "known_verifier_label_values": manifest_summary["known_verifier_label_values"],
        "balanced_label_counts": exp1480.get("balanced_label_counts", {}),
        "telemetry_cases_completed": int(exp1480.get("telemetry_cases_completed", 0)),
        "manifest_rows_audited": manifest_summary["row_count"],
        "telemetry_adversarial_validity_verdict": exp1473.get("telemetry_validity_verdict"),
        "missing_evidence_caveats": [
            "Exp 1473 blocks telemetry-only headline claims.",
            "Telemetry fields are availability/proxy evidence, not a formal HalluGuard DHRB certificate.",
        ],
    }

    unsafe_mass_bounds = [
        *[float(value) for value in exp1470.get("unsafe_mass_bounds", [])],
        *[float(value) for value in exp1482.get("unsafe_mass_bounds", [])],
    ]
    empirical_violation_rates = [
        *[float(value) for value in exp1470.get("empirical_violation_rates", [])],
        *[float(value) for value in exp1482.get("empirical_violation_rates", [])],
    ]
    prefix_constraints = list(exp1482.get("prefix_closed_constraints", []))
    reasoning_proxy = {
        "local_definition": (
            "Carnot reasoning-driven hallucination risk means risk that a reasoning "
            "step or bounded certificate path is invalid even when evidence is present."
        ),
        "bound_is_sound": bool(exp1470.get("bound_is_sound") and exp1482.get("bound_is_sound")),
        "mock_or_live_logprobs": sorted(
            {str(exp1470.get("mock_or_live_logprobs")), str(exp1482.get("mock_or_live_logprobs"))}
        ),
        "constraints_evaluated": int(exp1482.get("constraints_evaluated", len(prefix_constraints))),
        "prefix_closed_constraint_count": len(prefix_constraints),
        "max_unsafe_mass_bound": max(unsafe_mass_bounds) if unsafe_mass_bounds else None,
        "max_empirical_violation_rate": (
            max(empirical_violation_rates) if empirical_violation_rates else None
        ),
        "reasoning_step_validity_limitations": [
            "BEAVER-lite covers prefix/terminal constraints in the artifact, not every hidden or chain-of-thought step.",
            "No HalluGuard RHRB proof object is emitted by the current verifier stack.",
        ],
    }

    artifact: dict[str, Any] = {
        "schema": SCHEMA,
        "experiment": EXPERIMENT,
        "run_date": RUN_DATE,
        "project_root": PROJECT_ROOT_FOR_METADATA,
        "spec": ["REQ-REPORT-051", "SCENARIO-REPORT-051"],
        "status": "complete",
        "source_artifacts": _source_artifact_summaries(json_sources, inputs),
        "risk_decomposition_complete": True,
        "data_driven_fields_available": list(DATA_DRIVEN_FIELDS_AVAILABLE),
        "reasoning_driven_fields_available": list(REASONING_DRIVEN_FIELDS_AVAILABLE),
        "data_driven_hallucination_risk_proxy": data_proxy,
        "reasoning_driven_hallucination_risk_proxy": reasoning_proxy,
        "implemented_assumptions": list(IMPLEMENTED_ASSUMPTIONS),
        "missing_assumptions": list(MISSING_ASSUMPTIONS),
        "claim_allowed": False,
        "allowed_wording": list(ALLOWED_WORDING),
        "audit_note_path": audit_note_path,
        "honest_verdict": "halluguard_style_fit_audit_only_no_full_reproduction",
    }
    validate_artifact(artifact)
    return artifact


def render_audit_note(artifact: Mapping[str, Any]) -> str:
    """Render the markdown note with allowed wording and missing assumptions."""

    lines = [
        "# HalluGuard Carnot Risk-Bound Fit Audit",
        "",
        f"Run date: `{RUN_DATE}`",
        "",
        "## Verdict",
        "",
        "Full HalluGuard reproduction is not claimed.",
        f"`claim_allowed`: `{str(artifact['claim_allowed']).lower()}`",
        "",
        "## Data-Driven Evidence-Availability Risk",
        "",
        artifact["data_driven_hallucination_risk_proxy"]["local_definition"],
        "",
        "Available fields: "
        + ", ".join(f"`{field}`" for field in artifact["data_driven_fields_available"]),
        "",
        "## Reasoning-Step Risk",
        "",
        artifact["reasoning_driven_hallucination_risk_proxy"]["local_definition"],
        "",
        "Available fields: "
        + ", ".join(f"`{field}`" for field in artifact["reasoning_driven_fields_available"]),
        "",
        "## Implemented Assumptions",
        "",
    ]
    lines.extend(f"- {item}" for item in artifact["implemented_assumptions"])
    lines.extend(["", "## Missing Assumptions", ""])
    lines.extend(f"- {item}" for item in artifact["missing_assumptions"])
    lines.extend(["", "## Allowed Wording", ""])
    lines.extend(f"- {item}" for item in artifact["allowed_wording"])
    return "\n".join(lines) + "\n"


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """Validate the required Exp 1483 schema and blocked-claim boundary."""

    missing = REQUIRED_ARTIFACT_FIELDS - set(artifact)
    if missing:
        raise ValueError(f"missing required fields: {sorted(missing)}")
    if artifact["status"] != "complete":
        raise ValueError("status must be complete")
    if not artifact["risk_decomposition_complete"]:
        raise ValueError("risk_decomposition_complete must be true")
    if not artifact["data_driven_fields_available"]:
        raise ValueError("data_driven_fields_available must be non-empty")
    if not artifact["reasoning_driven_fields_available"]:
        raise ValueError("reasoning_driven_fields_available must be non-empty")
    if not artifact["missing_assumptions"]:
        raise ValueError("missing_assumptions must be non-empty")
    if artifact["claim_allowed"]:
        raise ValueError("claim_allowed must remain false while assumptions are missing")
    if not artifact["audit_note_path"]:
        raise ValueError("audit_note_path must be set")
    if artifact["honest_verdict"] != "halluguard_style_fit_audit_only_no_full_reproduction":
        raise ValueError("honest_verdict must block full HalluGuard reproduction wording")


def run(
    *,
    root: str | Path = REPO_ROOT,
    out_path: str | Path = DEFAULT_OUT_PATH,
    audit_note_path: str | Path = DEFAULT_AUDIT_NOTE_PATH,
) -> dict[str, Any]:
    """Write the Exp 1483 audit note and terminal artifact."""

    root_path = Path(root)
    output = Path(out_path)
    note_path = Path(audit_note_path)
    write_in_progress_artifact(output)
    artifact = build_artifact(
        load_source_inputs(root_path),
        audit_note_path=_relative_path(note_path, root_path),
    )
    note_path.parent.mkdir(parents=True, exist_ok=True)
    note_path.write_text(render_audit_note(artifact), encoding="utf-8")
    return _write_json(output, artifact)


def _source_artifact_summaries(
    json_sources: Mapping[str, Mapping[str, Any]],
    inputs: Mapping[str, Any],
) -> list[dict[str, Any]]:
    summaries = [
        {
            "path": rel_path,
            "role": JSON_SOURCE_FIELDS[rel_path][0],
            "status": payload.get("status", "unknown"),
            "honest_verdict": payload.get("honest_verdict"),
            "fields_used": [field for field in JSON_SOURCE_FIELDS[rel_path][1] if field in payload],
        }
        for rel_path, payload in json_sources.items()
    ]
    summaries.extend(
        [
            {
                "path": "research-references.md",
                "role": TEXT_SOURCE_FIELDS["research-references.md"],
                "status": "read",
                "honest_verdict": None,
                "fields_used": [
                    "entry_found",
                    "risk_decomposition_reference_found",
                    "full_reproduction_warning_found",
                ],
                "summary": inputs["halluguard_reference"],
            },
            {
                "path": "docs/research-notes/paper_v6_anchored_claim_matrix.md",
                "role": TEXT_SOURCE_FIELDS["docs/research-notes/paper_v6_anchored_claim_matrix.md"],
                "status": "read",
                "honest_verdict": None,
                "fields_used": [
                    "negative_boundaries_present",
                    "full_reproduction_claim_absent",
                ],
                "summary": inputs["claim_matrix_summary"],
            },
            {
                "path": inputs["manifest_path"],
                "role": "balanced_telemetry_jsonl_manifest",
                "status": "read",
                "honest_verdict": None,
                "fields_used": inputs["manifest_summary"]["fields_seen"],
                "summary": inputs["manifest_summary"],
            },
        ]
    )
    return summaries


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Mapping[str, Any]) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    artifact = dict(payload)
    path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return artifact


def _relative_path(path: Path, root: Path = REPO_ROOT) -> str:
    return os.path.relpath(path, root)


__all__ = [
    "AUDIT_NOTE_REL",
    "OUTPUT_FILENAME",
    "REPO_ROOT",
    "REQUIRED_ARTIFACT_FIELDS",
    "build_artifact",
    "extract_halluguard_reference_summary",
    "load_source_inputs",
    "render_audit_note",
    "run",
    "summarize_claim_matrix",
    "summarize_manifest",
    "validate_artifact",
    "write_in_progress_artifact",
]
