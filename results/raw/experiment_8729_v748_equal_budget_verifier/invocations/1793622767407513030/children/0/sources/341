"""Exp 3073 EBT/ARM-EBM adapter feasibility audit.

Spec refs: REQ-VERIFY-3073, SCENARIO-VERIFY-3073.

This audit is deliberately static: it reads the repository shape, names the
local surfaces that could host future EBT or ARM-EBM work, and records the
gates that must be met before anyone can claim an adapter exists.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import json
from pathlib import Path
import time
from typing import Any


JsonDict = dict[str, Any]
ClockFn = Callable[[], float]

ARTIFACT = "experiment_3073_ebt_arm_ebm_adapter_feasibility_audit_v1"
ARTIFACT_FILENAME = f"{ARTIFACT}.json"
SCHEMA = "carnot.ebt_arm_ebm_adapter_feasibility_audit.v1"
RUN_DATE = "20260525"
REPO_ROOT = Path(__file__).resolve().parents[3]
OUTPUT_REL_PATH = Path("results") / ARTIFACT_FILENAME
SUCCESS_PREFIXES = (
    "complete:",
    "complete_",
    "success:",
    "success_",
    "passed:",
    "passed_",
    "shipped:",
    "shipped_",
)
REQUIRED_ARTIFACT_FIELDS = (
    "ebt_arm_adapter_feasibility_ready",
    "ebt_arm_adapter_feasible",
    "adapter_surface",
    "required_prerequisites",
    "blockers",
    "recommended_next_experiment",
    "source_refs",
    "inference_substrate",
    "honest_verdict",
)
PREREQUISITE_GATES = (
    "data_shape",
    "energy_objective",
    "verifier_interface",
    "sampling_path",
    "evaluation_metric",
    "rollback_claim_boundaries",
)

SURFACE_DEFINITIONS: tuple[JsonDict, ...] = (
    {
        "surface": "core_energy_protocol",
        "path": "python/carnot/core/energy.py",
        "spec_refs": ["REQ-CORE-002", "SCENARIO-CORE-003", "REQ-VERIFY-3073"],
        "current_role": "Shared scalar energy, batch energy, and grad_energy contract.",
        "future_adapter_use": "Host a composite EBT/ARM energy behind the same EnergyFunction protocol.",
        "classification": "near_term_adapter_opportunity",
        "claim_boundary": "Protocol exists; no adapter implementation is claimed.",
    },
    {
        "surface": "verify_repair_pipeline",
        "path": "python/carnot/pipeline/verify_repair.py",
        "spec_refs": ["REQ-VERIFY-001", "REQ-VERIFY-003", "REQ-VERIFY-3073"],
        "current_role": "User-facing extraction, verification, certificate, and repair loop.",
        "future_adapter_use": "Expose EBT/ARM energy as an additive verifier signal or candidate reranker.",
        "classification": "near_term_adapter_opportunity",
        "claim_boundary": "Pipeline hook exists; no EBT/ARM hook is wired here by this audit.",
    },
    {
        "surface": "arm_logprob_energy_bridge",
        "path": "python/carnot/inference/arm_ebm_bridge.py",
        "spec_refs": ["REQ-VERIFY-3073"],
        "current_role": "Converts logits or logprobs into token rewards and sequence energy.",
        "future_adapter_use": "Provide an ARM-EBM confidence-energy feature for candidate scoring.",
        "classification": "near_term_adapter_opportunity",
        "claim_boundary": "Logprob energy is confidence telemetry, not correctness proof.",
    },
    {
        "surface": "ebt_reasoning_bridge",
        "path": "python/carnot/models/ebt_reasoning_bridge.py",
        "spec_refs": ["REQ-NRGPT-002", "REQ-VERIFY-3073"],
        "current_role": "Adapts continuous candidate embeddings to an EBT energy call.",
        "future_adapter_use": "Map candidate latent states into the common energy interface.",
        "classification": "near_term_adapter_opportunity",
        "claim_boundary": "Bridge shape exists; trained local EBT reasoning quality is not claimed.",
    },
    {
        "surface": "ebt_gradient_refinement_loop",
        "path": "python/carnot/models/ebt_gradient_refinement.py",
        "spec_refs": ["REQ-EBT-1742-1", "REQ-VERIFY-3073"],
        "current_role": "Minimal gradient descent loop over a continuous energy state.",
        "future_adapter_use": "Prototype bounded latent refinement after offline scoring gates pass.",
        "classification": "near_term_adapter_opportunity",
        "claim_boundary": "Loop exists; it is not wired to verifier candidates by this audit.",
    },
    {
        "surface": "reasoning_energy_embedding_path",
        "path": "python/carnot/inference/reasoning_energy.py",
        "spec_refs": ["REQ-VERIFY-3073"],
        "current_role": "Embeds reasoning text and scores/refines it with Gibbs energy.",
        "future_adapter_use": "Supply a deterministic fixture embedding path for first offline adapter tests.",
        "classification": "near_term_adapter_opportunity",
        "claim_boundary": "Template reasoning energy is not a general EBT/Kona implementation.",
    },
)


@dataclass(frozen=True)
class ExperimentConfig:
    """Runtime knobs for the static Exp 3073 audit."""

    repo_root: Path = REPO_ROOT
    output_path: Path | None = None
    run_date: str = RUN_DATE
    tests_run: Sequence[str] = ()
    clock: ClockFn = time.monotonic

    def artifact_path(self) -> Path:
        return self.output_path or self.repo_root / OUTPUT_REL_PATH


def build_artifact(config: ExperimentConfig | None = None, *, duration_s: float) -> JsonDict:
    """Build the terminal feasibility audit from local repository evidence."""

    active = config or ExperimentConfig()
    surfaces = _surface_records(active.repo_root)
    prerequisites = _required_prerequisites()
    paper_context = _paper_context_only_references()
    feasible = len(surfaces) >= 4 and all(row["local_evidence_present"] for row in surfaces)
    artifact: JsonDict = {
        "artifact": ARTIFACT,
        "schema": SCHEMA,
        "run_date": active.run_date,
        "spec_refs": ["REQ-VERIFY-3073", "SCENARIO-VERIFY-3073"],
        "ebt_arm_adapter_feasibility_ready": feasible,
        "ebt_arm_adapter_feasible": feasible,
        "bounded_theory_context_only": not feasible,
        "adapter_implementation_claimed": False,
        "adapter_surface": surfaces,
        "near_term_adapter_opportunities": _near_term_adapter_opportunities(surfaces),
        "paper_context_only_references": paper_context,
        "required_prerequisites": prerequisites,
        "blockers": _blockers(),
        "recommended_next_experiment": (
            "Build an offline fixture adapter over 6-12 exact solver rows: normalize "
            "candidate text, verifier certificates, optional token logprobs, and optional "
            "latent embeddings; compare constraint-only ranking against a composite "
            "constraint+ARM energy score before attempting EBT latent refinement."
        ),
        "source_refs": _source_refs(),
        "inference_substrate": _inference_substrate(),
        "tests_or_checks_run": list(active.tests_run),
        "duration_s": round(float(duration_s), 6),
        "methodology_note": (
            "Static local-code and literature audit only; no live model inference, no "
            "adapter wiring, no model weight loading, and no generation or verifier "
            "scoring was performed."
        ),
        "honest_verdict": (
            "complete: ebt_arm_adapter_future_path_feasible_with_prerequisites_"
            "no_implementation_claim"
            if feasible
            else "complete: bounded_theory_context_only_no_local_adapter_path"
        ),
    }
    validate_artifact(artifact)
    return artifact


def run_experiment(config: ExperimentConfig | None = None, *, write: bool = True) -> JsonDict:
    """Run the static audit and optionally persist the terminal JSON artifact."""

    active = config or ExperimentConfig()
    started = active.clock()
    duration_s = active.clock() - started
    artifact = build_artifact(active, duration_s=duration_s)
    if write:
        _write_json(active.artifact_path(), artifact)
    return artifact


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """Reject artifacts that overclaim implementation, inference, or feasibility."""

    missing = sorted(set(REQUIRED_ARTIFACT_FIELDS) - set(artifact))
    _require(not missing, f"missing required fields: {missing}")
    _require(
        artifact.get("adapter_implementation_claimed") is False,
        "adapter_implementation_claimed must remain false",
    )
    substrate = artifact.get("inference_substrate")
    _require(isinstance(substrate, Mapping), "inference_substrate must be an object")
    _require(substrate.get("live_model_inference") is False, "live_model_inference must be false")
    _require(substrate.get("model_weights_loaded") is False, "model_weights_loaded must be false")
    prerequisites = artifact.get("required_prerequisites")
    _require(isinstance(prerequisites, list) and prerequisites, "required_prerequisites missing")
    gates = {str(row.get("gate")) for row in prerequisites if isinstance(row, Mapping)}
    _require(gates == set(PREREQUISITE_GATES), "required_prerequisites must cover all gates")
    surfaces = artifact.get("adapter_surface")
    _require(isinstance(surfaces, list) and len(surfaces) >= 4, "adapter_surface too small")
    _require(
        bool(artifact.get("near_term_adapter_opportunities")),
        "near_term_adapter_opportunities must be non-empty",
    )
    _require(
        bool(artifact.get("paper_context_only_references")),
        "paper_context_only_references must be non-empty",
    )
    feasible = artifact.get("ebt_arm_adapter_feasible") is True
    bounded = artifact.get("bounded_theory_context_only") is True
    _require(feasible != bounded, "bounded_theory_context_only must be inverse of feasible")
    verdict = str(artifact.get("honest_verdict", ""))
    _require(verdict.startswith(SUCCESS_PREFIXES), "honest_verdict must start terminal success")


def _surface_records(repo_root: Path) -> list[JsonDict]:
    records: list[JsonDict] = []
    for definition in SURFACE_DEFINITIONS:
        path = repo_root / str(definition["path"])
        record = dict(definition)
        record["local_evidence_present"] = path.is_file()
        records.append(record)
    return records


def _near_term_adapter_opportunities(surfaces: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    return [
        {
            "surface": str(surface["surface"]),
            "first_test": "fixture candidate rows with exact labels and no live generation",
            "implementation_boundary": str(surface["claim_boundary"]),
        }
        for surface in surfaces
        if surface.get("classification") == "near_term_adapter_opportunity"
    ]


def _paper_context_only_references() -> list[JsonDict]:
    return [
        {
            "id": "arXiv:2507.02092",
            "topic": "Energy-Based Transformers",
            "claim_boundary": "context_only_not_local_implementation",
        },
        {
            "id": "arXiv:2512.15605",
            "topic": "Autoregressive language models as EBMs",
            "claim_boundary": "context_only_not_local_implementation",
        },
        {
            "id": "arXiv:2605.11011",
            "topic": "LoopUS latent recurrence",
            "claim_boundary": "context_only_not_local_implementation",
        },
        {
            "id": "Logical Intelligence Kona public material",
            "topic": "Kona-style public EBM reasoning positioning",
            "claim_boundary": "context_only_not_local_implementation",
        },
    ]


def _required_prerequisites() -> list[JsonDict]:
    return [
        {
            "gate": "data_shape",
            "requirement": (
                "A checked-in AdapterExample schema with prompt, candidate_text, verifier "
                "certificate, exact label, optional token_logprobs, optional latent vector, "
                "model identity, and source artifact provenance."
            ),
        },
        {
            "gate": "energy_objective",
            "requirement": (
                "A predeclared composite energy with calibrated weights for deterministic "
                "constraint energy, ARM sequence energy, and optional EBT latent compatibility."
            ),
        },
        {
            "gate": "verifier_interface",
            "requirement": (
                "A sidecar verifier callable that returns scalar energy, confidence, "
                "per-term decomposition, violations, and abstention/rollback metadata."
            ),
        },
        {
            "gate": "sampling_path",
            "requirement": (
                "Offline replay and candidate reranking first; only then bounded Langevin "
                "latent refinement with fixed seeds, step budget, and no hidden model mutation."
            ),
        },
        {
            "gate": "evaluation_metric",
            "requirement": (
                "Held-out exact-solver or execution labels with AUROC/ranking lift, "
                "abstention precision, rejection recall, repair yield, and false-accept rate."
            ),
        },
        {
            "gate": "rollback_claim_boundaries",
            "requirement": (
                "Feature flag, no model-weight mutation, explicit fallback to existing "
                "verifiers, and no implementation claim until local code and tests cover it."
            ),
        },
    ]


def _blockers() -> list[str]:
    return [
        "No checked-in EBT/ARM adapter currently wires these surfaces into VerifyRepairPipeline.",
        "No canonical adapter data shape currently binds token logprobs, latent states, and certificates.",
        "ARM logprob energy is a confidence signal and cannot be treated as correctness evidence alone.",
        "Current EBT bridge/refinement code does not provide a calibrated verifier objective for real traces.",
        "This audit did not run live model inference, generation, gradient refinement, or verifier scoring.",
    ]


def _source_refs() -> list[JsonDict]:
    refs = [
        {
            "id": "CODEX.md",
            "kind": "local_workflow",
            "path": "CODEX.md",
            "claim_boundary": "repo_instruction",
        },
        {
            "id": "_bmad/architecture.md",
            "kind": "local_architecture",
            "path": "_bmad/architecture.md",
            "claim_boundary": "local_architecture_context",
        },
        {
            "id": "research-references.md",
            "kind": "local_literature_sweep",
            "path": "research-references.md",
            "claim_boundary": "literature_context",
        },
    ]
    refs.extend(
        {
            "id": str(surface["surface"]),
            "kind": "local_code",
            "path": str(surface["path"]),
            "claim_boundary": str(surface["claim_boundary"]),
        }
        for surface in SURFACE_DEFINITIONS
    )
    refs.extend(_paper_context_only_references())
    return refs


def _inference_substrate() -> JsonDict:
    return {
        "kind": "static_local_code_and_literature_audit",
        "live_model_inference": False,
        "live_llm_inference": False,
        "model_weights_loaded": False,
        "gpu_required": False,
        "generation_performed": False,
        "verifier_scoring_performed": False,
    }


def _write_json(path: Path, artifact: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)
