"""Exp 1349 EBT citation and Kona parity gap audit.

This module writes a conservative, document-only audit. It compares the current
EBT citation neighborhood and Kona-style public positioning against evidence
that already exists in the local repository. The point is not to decide whether
EBT, Kona, or Extropic are good ideas. The point is to prevent Carnot from
promising external parity before local artifacts prove it.

Spec refs: REQ-KONA-009, SCENARIO-KONA-009.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
import json
from pathlib import Path
from typing import Any

DEFAULT_RUN_DATE = "20260505"
DEFAULT_OUTPUT_PATH = Path("results/experiment_1349_ebt_citation_kona_parity_gap_audit.json")
DEFAULT_THRML_PATH = Path("results/experiment_1347_thrml_compatibility_parity_audit.json")
DEFAULT_PBIT_PATH = Path("results/experiment_1348_pbit_update_dynamics_dual_bram_packet_v2.json")
DEFAULT_TOKEN_HEALTH_PATH = Path(
    "results/experiment_1323_sota_gguf_token_health_prompt_runtime_diagnostic.json"
)
DEFAULT_CERTIFICATE_TAXONOMY_PATH = Path(
    "results/experiment_1324_certificate_failure_taxonomy_formalizer_reality_check.json"
)

ARTIFACT_NAME = "experiment_1349_ebt_citation_kona_parity_gap_audit"
SCHEMA = "ebt_citation_kona_parity_gap_audit_v1"
HONEST_VERDICT = "external_parity_gap_audit_complete_local_evidence_only_no_kona_or_external_dependency_claim"

REQUIRED_ARTIFACT_FIELDS = {
    "status",
    "ebt_citation_themes",
    "kona_public_claims_mapped",
    "carnot_local_evidence",
    "parity_gaps",
    "phase3_obligations",
    "publication_claim_changes_needed",
    "external_dependency_claim_allowed",
    "honest_verdict",
}

WriteObserver = Callable[[Path, dict[str, Any]], None]


def build_artifact(
    *,
    thrml_artifact: Mapping[str, Any],
    pbit_artifact: Mapping[str, Any],
    token_health_artifact: Mapping[str, Any],
    certificate_taxonomy_artifact: Mapping[str, Any],
    project_root: str | Path = ".",
    run_date: str = DEFAULT_RUN_DATE,
) -> dict[str, Any]:
    """Build the completed audit from local evidence already on disk.

    The audit treats public EBT/Kona/Extropic material as roadmap pressure, not
    as evidence that Carnot can make the same claim. Each local capability claim
    below is tied to an artifact or spec, and the external-dependency flag stays
    false because the current repo has no reproducible Kona or TSU execution.
    """
    artifact = _base_artifact(project_root=project_root, run_date=run_date, status="complete")
    artifact.update(
        {
            "ebt_citation_themes": _ebt_citation_themes(),
            "kona_public_claims_mapped": _kona_public_claims_mapped(
                thrml_artifact=thrml_artifact,
                pbit_artifact=pbit_artifact,
                token_health_artifact=token_health_artifact,
                certificate_taxonomy_artifact=certificate_taxonomy_artifact,
            ),
            "carnot_local_evidence": _carnot_local_evidence(
                thrml_artifact=thrml_artifact,
                pbit_artifact=pbit_artifact,
                token_health_artifact=token_health_artifact,
                certificate_taxonomy_artifact=certificate_taxonomy_artifact,
            ),
            "parity_gaps": _parity_gaps(thrml_artifact, pbit_artifact, certificate_taxonomy_artifact),
            "phase3_obligations": _phase3_obligations(),
            "publication_claim_changes_needed": _publication_claim_changes_needed(),
            "external_dependency_claim_allowed": False,
            "honest_verdict": HONEST_VERDICT,
        }
    )
    validate_artifact(artifact)
    return artifact


def run_experiment(
    *,
    project_root: str | Path = ".",
    run_date: str = DEFAULT_RUN_DATE,
    thrml_path: str | Path = DEFAULT_THRML_PATH,
    pbit_path: str | Path = DEFAULT_PBIT_PATH,
    token_health_path: str | Path = DEFAULT_TOKEN_HEALTH_PATH,
    certificate_taxonomy_path: str | Path = DEFAULT_CERTIFICATE_TAXONOMY_PATH,
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
    write_observer: WriteObserver | None = None,
) -> dict[str, Any]:
    """Write an in-progress marker, then write the completed audit artifact."""
    root = Path(project_root)
    output = _resolve(root, output_path)
    _write_json(
        output,
        _base_artifact(project_root=root, run_date=run_date, status="in_progress"),
        write_observer=write_observer,
    )
    artifact = build_artifact(
        project_root=root,
        run_date=run_date,
        thrml_artifact=_read_json(_resolve(root, thrml_path)),
        pbit_artifact=_read_json(_resolve(root, pbit_path)),
        token_health_artifact=_read_json(_resolve(root, token_health_path)),
        certificate_taxonomy_artifact=_read_json(_resolve(root, certificate_taxonomy_path)),
    )
    _write_json(output, artifact, write_observer=write_observer)
    return artifact


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """Validate the claim-boundary fields required by REQ-KONA-009."""
    missing = REQUIRED_ARTIFACT_FIELDS - set(artifact)
    _require(not missing, f"missing required fields: {sorted(missing)}")
    _require(artifact["status"] == "complete", "status must be complete")
    _require(
        artifact["external_dependency_claim_allowed"] is False,
        "external_dependency_claim_allowed must remain false without local external evidence",
    )
    _require(artifact["honest_verdict"] == HONEST_VERDICT, "honest_verdict is not allowed")
    _require(bool(artifact["ebt_citation_themes"]), "ebt_citation_themes must be non-empty")
    _require(
        bool(artifact["kona_public_claims_mapped"]),
        "kona_public_claims_mapped must be non-empty",
    )
    _require(bool(artifact["parity_gaps"]), "parity_gaps must be non-empty")
    _require(bool(artifact["phase3_obligations"]), "phase3_obligations must be non-empty")


def _base_artifact(*, project_root: str | Path, run_date: str, status: str) -> dict[str, Any]:
    resolved_root = Path(project_root).resolve()
    return {
        "artifact": ARTIFACT_NAME,
        "schema": SCHEMA,
        "status": status,
        "ebt_citation_themes": [],
        "kona_public_claims_mapped": [],
        "carnot_local_evidence": {},
        "parity_gaps": [],
        "phase3_obligations": [],
        "publication_claim_changes_needed": [],
        "external_dependency_claim_allowed": False,
        "honest_verdict": "in_progress" if status == "in_progress" else HONEST_VERDICT,
        "artifact_metadata": {
            "project_root": str(resolved_root),
            "run_date": run_date,
            "source_documents": [
                "research-references.md",
                "_bmad/prd.md",
                "_bmad/architecture.md",
                "openspec/change-proposals/research-roadmap-vNEXT.md",
            ],
            "source_artifacts": [
                str(DEFAULT_TOKEN_HEALTH_PATH),
                str(DEFAULT_CERTIFICATE_TAXONOMY_PATH),
                str(DEFAULT_THRML_PATH),
                str(DEFAULT_PBIT_PATH),
            ],
        },
    }


def _ebt_citation_themes() -> list[dict[str, Any]]:
    return [
        {
            "theme": "ebt_reasoning",
            "source_refs": ["arXiv:2507.02092", "Semantic Scholar EBT citation watch"],
            "material_effect_on_carnot": (
                "Treat prediction as energy minimization over candidate answers, but require "
                "local verifier-tail or continuous-latent evidence before claiming EBT reasoning."
            ),
            "phase3_obligation": "implement REQ-KONA-001/002 continuous refinement evidence",
            "claim_boundary": "related-work support only; not local EBT parity",
        },
        {
            "theme": "nrgpt_energy_recurrence",
            "source_refs": ["arXiv:2512.16762"],
            "material_effect_on_carnot": (
                "NRGPT supports an energy-recurrence bridge from GPT-like behavior to EBM "
                "scoring, but Carnot must compare local energy traces before using it as evidence."
            ),
            "phase3_obligation": "compare NRGPT/Boltzmann-GPT energy ordering against verifier truth",
            "claim_boundary": "architecture reference; no local NRGPT parity claim",
        },
        {
            "theme": "ebt_policy_dynamic_compute",
            "source_refs": ["arXiv:2510.27545"],
            "material_effect_on_carnot": (
                "EBT-Policy and adaptive Langevin-style methods raise the bar for dynamic "
                "compute allocation and stop policies in Carnot's verifier-tail roadmap."
            ),
            "phase3_obligation": "measure dynamic compute allocation and failure-type recovery policies",
            "claim_boundary": "policy inspiration; no embodied-control or EBT-Policy performance claim",
        },
        {
            "theme": "intrinsic_optimizer_variants",
            "source_refs": ["arXiv:2511.00907", "ARM-to-EBM bridge"],
            "material_effect_on_carnot": (
                "Optimizer-view transformer work keeps the ARM-to-EBM bridge plausible, but "
                "Carnot still needs local optimizer or energy-descent evidence."
            ),
            "phase3_obligation": "test optimizer-like energy descent against local certificate traces",
            "claim_boundary": "theory support only; no optimizer-parity result",
        },
        {
            "theme": "metacognitive_code_generation",
            "source_refs": ["MetaGenAI 2025 OpenReview EBT code assessment"],
            "material_effect_on_carnot": (
                "Code-generation-specific EBT metacognition means Carnot must show code "
                "metacognition or verifier-tail value locally before PRD language implies it."
            ),
            "phase3_obligation": "run a native or sidecar code metacognition audit with local evidence",
            "claim_boundary": "must not claim metacognitive code-generation ability yet",
        },
    ]


def _kona_public_claims_mapped(
    *,
    thrml_artifact: Mapping[str, Any],
    pbit_artifact: Mapping[str, Any],
    token_health_artifact: Mapping[str, Any],
    certificate_taxonomy_artifact: Mapping[str, Any],
) -> list[dict[str, Any]]:
    return [
        {
            "public_claim": "non_autoregressive_ebm_reasoning_layer",
            "local_evidence": "REQ-KONA-001..008 specify the target, but no native model proves it.",
            "parity_gap": "No local continuous-latent, full-answer, non-autoregressive refinement artifact.",
            "claim_allowed_for_carnot": False,
        },
        {
            "public_claim": "valid_safe_permissible_state_scoring",
            "local_evidence": (
                "Phase-1 verifier/certificate diagnostics exist; current parse rate is "
                f"{_metric(certificate_taxonomy_artifact, 'source_metrics', 'exp1312_certificate_parse_rate')} "
                "and token health recovered multi-token local GGUF output."
            ),
            "parity_gap": "Semantic certificate gate and false-acceptance accounting remain unsolved.",
            "claim_allowed_for_carnot": bool(
                token_health_artifact.get("min_tokens_recovered")
                and certificate_taxonomy_artifact.get("status") == "complete"
            ),
        },
        {
            "public_claim": "under_llm_stack_not_chatbot",
            "local_evidence": "Architecture documents Carnot as verifier-side energy layer around LLM outputs.",
            "parity_gap": "Still a sidecar verify-repair pipeline, not an internalized foundation model.",
            "claim_allowed_for_carnot": True,
        },
        {
            "public_claim": "hardware_portable_energy_execution",
            "local_evidence": (
                "Exp 1347 THRML import available="
                f"{bool(thrml_artifact.get('thrml_import_available'))}; "
                "Exp 1348 board_executed="
                f"{bool(_metric(pbit_artifact, 'metadata', 'board_executed'))}."
            ),
            "parity_gap": "THRML parity or real accelerator execution is not locally reproduced.",
            "claim_allowed_for_carnot": False,
        },
        {
            "public_claim": "open_reproducible_local_certificates",
            "local_evidence": (
                "Local JSON artifacts exist for token health, certificate failure taxonomy, "
                "THRML accounting, and p-bit accounting."
            ),
            "parity_gap": "Certificates are diagnostic and incomplete; parse/truthfulness gates remain below target.",
            "claim_allowed_for_carnot": True,
        },
    ]


def _carnot_local_evidence(
    *,
    thrml_artifact: Mapping[str, Any],
    pbit_artifact: Mapping[str, Any],
    token_health_artifact: Mapping[str, Any],
    certificate_taxonomy_artifact: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "prd_and_architecture": {
            "current_role": "open EBM verifier framework with Phase-3 foundation-model aspiration",
            "claim_boundary": "PRD vision is not evidence of shipped Kona-style parity",
        },
        "phase3_kona_spec": {
            "spec_requirements_present": ["REQ-KONA-001", "REQ-KONA-006", "REQ-KONA-009"],
            "implementation_status": "spec and primitives only for parity; external audit implemented here",
        },
        "exp1323_local_sota_token_health": {
            "min_tokens_recovered": token_health_artifact.get("min_tokens_recovered"),
            "empty_or_one_token_rate": token_health_artifact.get("empty_or_one_token_rate"),
            "models_used": token_health_artifact.get("models_used", []),
            "honest_verdict": token_health_artifact.get("honest_verdict"),
        },
        "exp1324_certificate_failure_taxonomy": {
            "certificate_parse_rate": _metric(
                certificate_taxonomy_artifact,
                "source_metrics",
                "exp1312_certificate_parse_rate",
            ),
            "certificate_truthfulness_rate": _metric(
                certificate_taxonomy_artifact,
                "source_metrics",
                "exp1312_certificate_truthfulness_rate",
            ),
            "minimum_parseable_attempts_to_recover": certificate_taxonomy_artifact.get(
                "minimum_parseable_attempts_to_recover"
            ),
            "honest_verdict": certificate_taxonomy_artifact.get("honest_verdict"),
        },
        "exp1347_thrml_compatibility": {
            "thrml_import_available": thrml_artifact.get("thrml_import_available"),
            "energy_parity_max_abs_error": thrml_artifact.get("energy_parity_max_abs_error"),
            "hardware_claim_allowed": thrml_artifact.get("hardware_claim_allowed"),
            "honest_verdict": thrml_artifact.get("honest_verdict"),
        },
        "exp1348_pbit_dual_bram_packet": {
            "hardware_claim_allowed": pbit_artifact.get("hardware_claim_allowed"),
            "kv260_claim_allowed": pbit_artifact.get("kv260_claim_allowed"),
            "board_executed": _metric(pbit_artifact, "metadata", "board_executed"),
            "best_cpu_kl_to_gibbs": _best_cpu_kl(pbit_artifact),
            "honest_verdict": pbit_artifact.get("honest_verdict"),
        },
    }


def _parity_gaps(
    thrml_artifact: Mapping[str, Any],
    pbit_artifact: Mapping[str, Any],
    certificate_taxonomy_artifact: Mapping[str, Any],
) -> list[dict[str, str]]:
    return [
        {
            "gap": "native_kona_style_reasoning",
            "why_it_matters": "Kona parity requires continuous latent, non-autoregressive refinement.",
            "local_status": "Phase-3 spec exists; no local artifact proves native full-answer refinement.",
        },
        {
            "gap": "semantic certificate gate",
            "why_it_matters": "State-scoring claims need parsed and truthful certificates, not parseable JSON alone.",
            "local_status": (
                "Exp 1324 reports parse rate "
                f"{_metric(certificate_taxonomy_artifact, 'source_metrics', 'exp1312_certificate_parse_rate')} "
                "with semantic invalidity still present."
            ),
        },
        {
            "gap": "metacognitive_code_generation",
            "why_it_matters": "The EBT citation neighborhood includes code metacognition claims.",
            "local_status": "No native EBT code-generation metacognition artifact is present.",
        },
        {
            "gap": "THRML_or_TSU_execution",
            "why_it_matters": "Extropic-adjacent language requires THRML parity or real TSU evidence.",
            "local_status": (
                "THRML import available="
                f"{bool(thrml_artifact.get('thrml_import_available'))}; "
                "hardware claim allowed="
                f"{bool(thrml_artifact.get('hardware_claim_allowed'))}."
            ),
        },
        {
            "gap": "KV260_or_pbit_hardware_execution",
            "why_it_matters": "Hardware-portable energy execution cannot rest on CPU-only packets.",
            "local_status": (
                "Exp 1348 hardware claim allowed="
                f"{bool(pbit_artifact.get('hardware_claim_allowed'))}; board execution="
                f"{bool(_metric(pbit_artifact, 'metadata', 'board_executed'))}."
            ),
        },
    ]


def _phase3_obligations() -> list[dict[str, str]]:
    return [
        {
            "obligation": "Implement Stage-1/2 continuous latent refinement before using Kona parity language.",
            "spec_refs": "REQ-KONA-001, REQ-KONA-002, REQ-KONA-004",
        },
        {
            "obligation": "Recover certificate parse/truthfulness and semantic validator gates before claiming reliable state scoring.",
            "spec_refs": "REQ-KONA-005, SCENARIO-KONA-009",
        },
        {
            "obligation": "Compare NRGPT/Boltzmann-GPT or ARM-to-EBM energy traces against local verifier truth.",
            "spec_refs": "REQ-KONA-009",
        },
        {
            "obligation": "Run code-metacognition evidence locally before borrowing EBT code-generation language.",
            "spec_refs": "REQ-KONA-009",
        },
        {
            "obligation": "Rerun THRML parity or actual accelerator execution before hardware dependency claims.",
            "spec_refs": "REQ-KONA-006, REQ-KONA-009",
        },
    ]


def _publication_claim_changes_needed() -> list[str]:
    return [
        "Replace broad 'Kona parity' wording with 'Phase-3 target; current evidence is verifier-side only.'",
        "Describe EBT, NRGPT, EBT-Policy, and optimizer variants as related-work pressure, not proof of Carnot capability.",
        "For Extropic/THRML/Z1/XTR-0, say Carnot has simulation/accounting work only unless a local run proves parity.",
        "Limit PRD-facing claims to open, local verifier evidence and diagnostic certificate artifacts.",
        "State that external_dependency_claim_allowed=false for current publication language.",
    ]


def _metric(artifact: Mapping[str, Any], *keys: str) -> Any:
    value: Any = artifact
    for key in keys:
        value = value.get(key) if isinstance(value, Mapping) else None
    return value


def _best_cpu_kl(pbit_artifact: Mapping[str, Any]) -> float | None:
    values = [
        row.get("cpu_kl_to_gibbs")
        for row in pbit_artifact.get("reuse_factor_grid", [])
        if isinstance(row, Mapping) and row.get("cpu_kl_to_gibbs") is not None
    ]
    return min(values) if values else None


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(
    path: Path,
    payload: dict[str, Any],
    *,
    write_observer: WriteObserver | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if write_observer is not None:
        write_observer(path, payload)


def _resolve(project_root: Path, path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else project_root / candidate


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)
