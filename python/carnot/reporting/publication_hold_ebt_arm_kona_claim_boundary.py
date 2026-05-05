"""Build the Exp 1362 publication-hold claim-boundary artifact.

Spec refs: REQ-KONA-009, SCENARIO-KONA-009.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
PROJECT_ROOT_FOR_METADATA = "/home/ianblenke/github.com/ianblenke/carnot"
DEFAULT_RUN_DATE = "20260505"
EXPERIMENT = "1362_publication_hold_ebt_arm_kona_claim_boundary"
SCHEMA = "publication_hold_ebt_arm_kona_claim_boundary_v1"
HONEST_VERDICT = (
    "publication_hold_active_local_evidence_does_not_support_ebt_arm_kona_or_hardware_claims"
)
DEFAULT_OUT_PATH = (
    Path("results") / "experiment_1362_publication_hold_ebt_arm_kona_claim_boundary.json"
)

REQUESTED_EXP1349_PATH = Path(
    "results/experiment_1349_external_parity_gap_audit_kona_extropic_ebt_arm.json"
)

SOURCE_PATHS = {
    "exp1324": Path(
        "results/experiment_1324_certificate_failure_taxonomy_formalizer_reality_check.json"
    ),
    "exp1344": Path(
        "results/experiment_1344_continuous_self_learning_failure_type_memory_policy.json"
    ),
    "exp1349": Path("results/experiment_1349_ebt_citation_kona_parity_gap_audit.json"),
    "exp1353": Path("results/experiment_1353_triggered_certificate_v7_truncproof_sota.json"),
    "exp1354": Path("results/experiment_1354_logicskills_certificate_skill_split.json"),
    "exp1355": Path("results/experiment_1355_logitext_nsvif_partial_smt_validator.json"),
    "exp1313": Path(
        "results/experiment_1313_constrainprompt_nsvif_semantic_validator_mus_repair.json"
    ),
    "exp1316": Path("results/experiment_1316_dvi_certificate_tail_online_update.json"),
    "exp1358": Path(
        "results/experiment_1358_continuous_self_learning_verifier_selected_memory.json"
    ),
    "exp1361": Path("results/experiment_1361_pdit_certificate_state_hardware_mapping.json"),
}

REQUIRED_ARTIFACT_FIELDS = {
    "status",
    "certificate_evidence_summary",
    "self_learning_evidence_summary",
    "hardware_evidence_summary",
    "ebt_arm_claim_boundary",
    "kona_parity_gaps",
    "publication_hold_state",
    "claim_changes_needed",
    "external_dependency_claim_allowed",
    "honest_verdict",
}

WriteObserver = Callable[[Path, dict[str, Any]], None]


def write_in_progress_artifact(
    out_path: Path | str = DEFAULT_OUT_PATH,
    *,
    project_root: Path | str = REPO_ROOT,
    run_date: str = DEFAULT_RUN_DATE,
    write_observer: WriteObserver | None = None,
) -> dict[str, Any]:
    """REQ-KONA-009: persist a bootstrap artifact before reading claim evidence."""

    artifact = _base_artifact(project_root=project_root, run_date=run_date, status="in_progress")
    _write_json(Path(out_path), artifact, write_observer=write_observer)
    return artifact


def build_artifact(
    sources: Mapping[str, Mapping[str, Any]],
    *,
    project_root: Path | str = REPO_ROOT,
    run_date: str = DEFAULT_RUN_DATE,
    input_resolution: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """REQ-KONA-009: summarize only local evidence before changing publication claims."""

    artifact = _base_artifact(project_root=project_root, run_date=run_date, status="complete")
    artifact.update(
        {
            "input_resolution": dict(input_resolution or {}),
            "certificate_evidence_summary": _certificate_evidence_summary(sources),
            "self_learning_evidence_summary": _self_learning_evidence_summary(sources),
            "hardware_evidence_summary": _hardware_evidence_summary(sources),
            "ebt_arm_claim_boundary": _ebt_arm_claim_boundary(sources),
            "kona_parity_gaps": _kona_parity_gaps(sources),
            "publication_hold_state": "active",
            "publication_hold_rationale": _publication_hold_rationale(sources),
            "claim_changes_needed": _claim_changes_needed(),
            "external_dependency_claim_allowed": False,
            "honest_verdict": HONEST_VERDICT,
        }
    )
    validate_artifact(artifact)
    return artifact


def run(
    project_root: Path | str = REPO_ROOT,
    out_path: Path | str = DEFAULT_OUT_PATH,
    *,
    run_date: str = DEFAULT_RUN_DATE,
    write_observer: WriteObserver | None = None,
) -> dict[str, Any]:
    """SCENARIO-KONA-009: write an auditable hold artifact from repository JSON only."""

    root = Path(project_root)
    output = _resolve(root, out_path)
    write_in_progress_artifact(
        output,
        project_root=root,
        run_date=run_date,
        write_observer=write_observer,
    )
    sources, input_resolution = _load_sources(root)
    artifact = build_artifact(
        sources,
        project_root=root,
        run_date=run_date,
        input_resolution=input_resolution,
    )
    _write_json(output, artifact, write_observer=write_observer)
    return artifact


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """REQ-KONA-009: reject artifacts that lift the hold or borrow external proof."""

    missing = REQUIRED_ARTIFACT_FIELDS - set(artifact)
    _require(not missing, f"missing required fields: {sorted(missing)}")
    _require(artifact["status"] == "complete", "status must be complete")
    _require(artifact["publication_hold_state"] == "active", "publication hold must remain active")
    _require(
        artifact["external_dependency_claim_allowed"] is False,
        "external_dependency_claim_allowed must remain false without local parity evidence",
    )
    _require(artifact["honest_verdict"] == HONEST_VERDICT, "honest_verdict is not allowed")
    _require(
        artifact["certificate_evidence_summary"].get("certificate_claim_allowed") is False,
        "certificate claim must stay disallowed when parse/truth gates fail",
    )
    _require(
        artifact["self_learning_evidence_summary"].get("headline_self_learning_claim_allowed")
        is False,
        "self-learning headline claim must stay disallowed without fresh verifier samples",
    )
    _require(
        artifact["hardware_evidence_summary"].get("hardware_execution_claim_allowed") is False,
        "hardware execution claim must stay disallowed without hardware execution",
    )
    _require(bool(artifact["kona_parity_gaps"]), "kona_parity_gaps must be non-empty")
    _require(bool(artifact["claim_changes_needed"]), "claim_changes_needed must be non-empty")


def _base_artifact(*, project_root: Path | str, run_date: str, status: str) -> dict[str, Any]:
    root = Path(project_root).resolve()
    return {
        "experiment": EXPERIMENT,
        "schema": SCHEMA,
        "artifact_metadata": {
            "project_root": PROJECT_ROOT_FOR_METADATA,
            "actual_project_root": str(root),
            "run_date": run_date,
            "source_artifacts": [str(path) for path in SOURCE_PATHS.values()],
            "requested_exp1349_path": str(REQUESTED_EXP1349_PATH),
            "source_documents": [
                "CODEX.md",
                "CLAUDE.md",
                "_bmad/prd.md",
                "_bmad/architecture.md",
                "research-references.md",
                "research-hardware-wishlist.md",
            ],
            "spec_refs": ["REQ-KONA-009", "SCENARIO-KONA-009"],
        },
        "run_date": run_date,
        "project_root": PROJECT_ROOT_FOR_METADATA,
        "status": status,
        "certificate_evidence_summary": {},
        "self_learning_evidence_summary": {},
        "hardware_evidence_summary": {},
        "ebt_arm_claim_boundary": {},
        "kona_parity_gaps": [],
        "publication_hold_state": "active" if status == "complete" else "unknown",
        "claim_changes_needed": [],
        "external_dependency_claim_allowed": False,
        "honest_verdict": "in_progress" if status == "in_progress" else HONEST_VERDICT,
    }


def _certificate_evidence_summary(sources: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    exp1324 = _source(sources, "exp1324")
    exp1353 = _source(sources, "exp1353")
    exp1354 = _source(sources, "exp1354")
    exp1355 = _source(sources, "exp1355")
    exp1313 = _source(sources, "exp1313")
    parse_rate = _metric(exp1353, "certificate_parse_rate")
    truthfulness_rate = _metric(exp1353, "certificate_truthfulness_rate")
    return {
        "certificate_claim_allowed": False,
        "exp1353_status": exp1353.get("status"),
        "exp1353_honest_verdict": exp1353.get("honest_verdict"),
        "certificate_case_count": exp1353.get("certificate_case_count"),
        "certificate_parse_rate": parse_rate,
        "certificate_truthfulness_rate": truthfulness_rate,
        "trigger_token_hit_rate": exp1353.get("trigger_token_hit_rate"),
        "unknown_preservation_rate": exp1353.get("unknown_preservation_rate"),
        "sota_generation_provenance_available": exp1353.get("headline_result_allowed"),
        "certificate_success_gate_satisfied": _number_at_least(parse_rate, 0.75)
        and _number_at_least(truthfulness_rate, 0.75),
        "dominant_skill_gap": exp1354.get("dominant_skill_gap"),
        "symbolization_pass_rate": exp1354.get("symbolization_pass_rate"),
        "skill_split_claim_allowed": exp1354.get("skill_split_claim_allowed"),
        "prior_exp1324_parse_rate": _metric(
            exp1324,
            "source_metrics",
            "exp1312_certificate_parse_rate",
        ),
        "prior_exp1324_truthfulness_rate": _metric(
            exp1324,
            "source_metrics",
            "exp1312_certificate_truthfulness_rate",
        ),
        "semantic_validator_state": {
            "exp1355_status": exp1355.get("status"),
            "exp1355_gate_check_summary": exp1355.get("gate_check_summary"),
            "exp1313_status": exp1313.get("status"),
            "exp1313_gate_check_summary": exp1313.get("gate_check_summary"),
        },
        "claim_boundary": (
            "Local SOTA generation ran, but every Exp 1353 certificate row missed the "
            "structural tag; parse, truthfulness, trigger-token, and UNKNOWN-preservation "
            "rates are 0.0. Exp 1354 supports a symbolization failure diagnosis, not a "
            "successful certificate claim. Semantic validator work remains gate-blocked."
        ),
    }


def _self_learning_evidence_summary(sources: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    exp1344 = _source(sources, "exp1344")
    exp1358 = _source(sources, "exp1358")
    exp1316 = _source(sources, "exp1316")
    return {
        "headline_self_learning_claim_allowed": False,
        "exp1344_honest_verdict": exp1344.get("honest_verdict"),
        "exp1344_self_learning_delta_overall": exp1344.get("self_learning_delta_overall"),
        "exp1344_dvi_ready": exp1344.get("dvi_ready"),
        "exp1344_headline_result_allowed": exp1344.get("headline_result_allowed"),
        "exp1358_honest_verdict": exp1358.get("honest_verdict"),
        "fresh_verified_sample_count": exp1358.get("fresh_verified_sample_count"),
        "update_is_replay_only": exp1358.get("update_is_replay_only"),
        "nonforgetting_certificate_rate": exp1358.get("nonforgetting_certificate_rate"),
        "memory_regression_count": exp1358.get("memory_regression_count"),
        "accepted_violation_delta": exp1358.get("accepted_violation_delta"),
        "dvi_status": {
            "exp1316_status": exp1316.get("status"),
            "exp1316_gate_check_summary": exp1316.get("gate_check_summary"),
            "exp1358_dvi_ready": exp1358.get("dvi_ready"),
        },
        "claim_boundary": (
            "The local memory policy has replay-positive, nonforgetting evidence and "
            "demotes or quarantines unsafe memories, but Exp 1358 has zero fresh "
            "verifier-selected samples and explicitly marks the update replay-only and "
            "non-headline. DVI readiness is not an executed DVI update."
        ),
    }


def _hardware_evidence_summary(sources: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    exp1361 = _source(sources, "exp1361")
    metadata = _metric(exp1361, "metadata") or {}
    return {
        "hardware_execution_claim_allowed": False,
        "hardware_claim_allowed": exp1361.get("hardware_claim_allowed"),
        "kv260_claim_allowed": exp1361.get("kv260_claim_allowed"),
        "hardware_executed": metadata.get("hardware_executed"),
        "external_hardware_executed": metadata.get("external_hardware_executed"),
        "synthesis_performed": metadata.get("synthesis_performed"),
        "board_executed": metadata.get("board_executed"),
        "energy_equivalence_error": exp1361.get("energy_equivalence_error"),
        "energy_equivalence_proxy": exp1361.get("energy_equivalence_proxy"),
        "state_expansion_ratio": exp1361.get("state_expansion_ratio"),
        "next_hardware_requirements": exp1361.get("next_hardware_requirements", []),
        "honest_verdict": exp1361.get("honest_verdict"),
        "claim_boundary": (
            "Exp 1361 is a CPU-only p-dit/p-int certificate-state mapping with an "
            "exact proxy energy-table check. It does not include Vivado synthesis, "
            "KV260 board execution, TSU execution, analog measurements, or hardware energy data."
        ),
    }


def _ebt_arm_claim_boundary(sources: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    exp1349 = _source(sources, "exp1349")
    return {
        "architectural_inspiration_allowed": True,
        "local_ebt_arm_equivalence_proven": False,
        "external_dependency_claim_allowed": False,
        "source_exp1349_honest_verdict": exp1349.get("honest_verdict"),
        "allowed_language": [
            "EBT and ARM-EBM are related-work and roadmap pressure for verifier-trained energy reasoning.",
            "Carnot currently has local verifier-side diagnostics, replay-only memory evidence, and CPU-only hardware-mapping evidence.",
            "Phase 3 can be described as an aspiration toward continuous-latent, energy-minimized reasoning when the statement is explicitly gated on future local proof.",
        ],
        "disallowed_language": [
            "Carnot has achieved EBT or ARM-EBM equivalence.",
            "Carnot has Kona parity or native non-autoregressive full-answer EBM reasoning.",
            "Carnot has demonstrated Extropic, THRML, TSU, KV260, or other hardware execution for these certificate-state claims.",
            "External EBT, ARM-EBM, Kona, or Extropic progress lifts Carnot's publication hold.",
        ],
        "local_proof_still_required": [
            "recover parseable and truthful certificate generation under the current local SOTA path",
            "run semantic validation after the parse gate is satisfied",
            "execute DVI or self-learning updates on fresh verifier-selected samples",
            "measure local continuous-latent energy descent before using EBT/ARM equivalence wording",
            "run synthesis or board/TSU hardware before claiming hardware execution",
        ],
    }


def _kona_parity_gaps(sources: Mapping[str, Mapping[str, Any]]) -> list[dict[str, Any]]:
    exp1349 = _source(sources, "exp1349")
    exp1353 = _source(sources, "exp1353")
    exp1355 = _source(sources, "exp1355")
    exp1358 = _source(sources, "exp1358")
    exp1361 = _source(sources, "exp1361")
    inherited = exp1349.get("parity_gaps", [])
    return [
        {
            "gap": "native_kona_style_reasoning",
            "local_status": "No local continuous-latent, full-answer, non-autoregressive refinement artifact is present.",
            "claim_allowed": False,
        },
        {
            "gap": "certificate_success_gate",
            "local_status": (
                "Exp 1353 certificate_parse_rate="
                f"{exp1353.get('certificate_parse_rate')} and certificate_truthfulness_rate="
                f"{exp1353.get('certificate_truthfulness_rate')}."
            ),
            "claim_allowed": False,
        },
        {
            "gap": "semantic_validator_execution",
            "local_status": exp1355.get("gate_check_summary")
            or "semantic validator evidence absent",
            "claim_allowed": False,
        },
        {
            "gap": "fresh_self_learning_and_dvi_execution",
            "local_status": (
                "Exp 1358 fresh_verified_sample_count="
                f"{exp1358.get('fresh_verified_sample_count')} and update_is_replay_only="
                f"{exp1358.get('update_is_replay_only')}."
            ),
            "claim_allowed": False,
        },
        {
            "gap": "hardware_execution",
            "local_status": (
                "Exp 1361 hardware_claim_allowed="
                f"{exp1361.get('hardware_claim_allowed')}; kv260_claim_allowed="
                f"{exp1361.get('kv260_claim_allowed')}; honest_verdict="
                f"{exp1361.get('honest_verdict')}."
            ),
            "claim_allowed": False,
        },
        {
            "gap": "inherited_exp1349_parity_gaps",
            "local_status": inherited,
            "claim_allowed": False,
        },
    ]


def _publication_hold_rationale(sources: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    cert = _certificate_evidence_summary(sources)
    self_learning = _self_learning_evidence_summary(sources)
    hardware = _hardware_evidence_summary(sources)
    blockers = [
        "certificate_parse_truthfulness_and_unknown_preservation_gates_not_satisfied",
        "semantic_validator_and_dvi_paths_gate_blocked_or_replay_only",
        "fresh_verifier_selected_self_learning_samples_absent",
        "hardware_synthesis_board_tsu_or_thrml_execution_absent",
        "external_ebt_arm_kona_extropic_parity_not_locally_demonstrated",
    ]
    return {
        "blockers_satisfied": False,
        "active_blockers": blockers,
        "certificate_gate_satisfied": cert["certificate_success_gate_satisfied"],
        "fresh_self_learning_available": self_learning["fresh_verified_sample_count"]
        not in {None, 0},
        "hardware_execution_available": hardware["hardware_executed"] is True,
        "external_parity_demonstrated": False,
    }


def _claim_changes_needed() -> list[str]:
    return [
        "Do not state or imply external parity with EBT, ARM-EBM, Kona, Extropic, THRML, or hardware roadmaps.",
        "Describe EBT and ARM-EBM as architectural inspiration and related-work pressure, not as local proof of Carnot capability.",
        "Replace any Kona-parity wording with a local-evidence boundary: verifier-side diagnostics only, no native continuous-latent full-answer reasoning.",
        "State that Exp 1353 ran local SOTA generation but produced 0.0 certificate parse, truthfulness, trigger-token, and UNKNOWN-preservation rates.",
        "State that semantic validation and DVI remain gated or replay-only until certificate gates and fresh verifier-selected samples pass.",
        "State that self-learning evidence is non-headline replay evidence, not fresh autonomous improvement.",
        "State that Exp 1361 is CPU-only p-dit/p-int mapping evidence and does not prove KV260, FPGA, THRML, TSU, analog, or hardware energy execution.",
        "Keep publication hold active until the local blockers are satisfied by reproducible repository artifacts.",
        "Keep external_dependency_claim_allowed=false in publication-facing language.",
    ]


def _load_sources(root: Path) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    sources: dict[str, dict[str, Any]] = {}
    missing: list[str] = []
    for key, rel_path in SOURCE_PATHS.items():
        path = _resolve(root, rel_path)
        if path.exists():
            sources[key] = _read_json(path)
        else:
            sources[key] = {}
            missing.append(str(rel_path))
    input_resolution = {
        "missing_source_artifacts": missing,
        "exp1349": {
            "requested": str(REQUESTED_EXP1349_PATH),
            "requested_available": _resolve(root, REQUESTED_EXP1349_PATH).exists(),
            "used": str(SOURCE_PATHS["exp1349"]),
            "used_available": _resolve(root, SOURCE_PATHS["exp1349"]).exists(),
        },
    }
    return sources, input_resolution


def _source(sources: Mapping[str, Mapping[str, Any]], name: str) -> Mapping[str, Any]:
    return sources.get(name, {})


def _metric(artifact: Mapping[str, Any], *keys: str) -> Any:
    value: Any = artifact
    for key in keys:
        value = value.get(key) if isinstance(value, Mapping) else None
    return value


def _number_at_least(value: object, threshold: float) -> bool:
    return isinstance(value, int | float) and float(value) >= threshold


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


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
