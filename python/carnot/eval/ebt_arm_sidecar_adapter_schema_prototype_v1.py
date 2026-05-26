"""Exp 3091 EBT/ARM sidecar adapter schema prototype.

This experiment writes a terminal artifact for a sidecar boundary only. It
does not train an EBT/ARM model, call a live LLM, load model weights, wire the
scorer into generation, claim a benchmark speedup, or claim hardware
acceleration.

Spec refs: REQ-VERIFY-3091, SCENARIO-VERIFY-3091.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import json
from pathlib import Path
import time
from typing import Any

from carnot.inference.ebt_arm_sidecar_adapter import (
    REQUIRED_SIDECAR_FIELDS,
    SCHEMA_REL_PATH,
    SidecarReplayScorer,
    example_sidecar_records,
    load_sidecar_schema,
)


JsonDict = dict[str, Any]

ARTIFACT = "experiment_3091_ebt_arm_sidecar_adapter_schema_prototype_v1"
ARTIFACT_FILENAME = f"{ARTIFACT}.json"
SCHEMA = "carnot.ebt_arm_sidecar_adapter_schema_prototype.v1"
RUN_DATE = "20260526"
REPO_ROOT = Path(__file__).resolve().parents[3]
OUTPUT_REL_PATH = Path("results") / ARTIFACT_FILENAME
REPLAY_SCORER_REL_PATH = Path("python/carnot/inference/ebt_arm_sidecar_adapter.py")
TEST_REL_PATH = Path(
    "tests/python/test_experiment_3091_ebt_arm_sidecar_adapter_schema_prototype.py"
)
PRIOR_AUDIT_REL_PATH = Path("results/experiment_3073_ebt_arm_ebm_adapter_feasibility_audit_v1.json")
SPEC_REFS = ["REQ-VERIFY-3091", "SCENARIO-VERIFY-3091"]
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
    "adapter_schema_ready",
    "sidecar_replay_scorer_ready",
    "schema_path",
    "replay_scorer_path",
    "tests_added_or_reused",
    "implementation_claim_boundary",
    "no_weight_update_claim",
    "source_artifacts",
    "inference_substrate",
    "honest_verdict",
)
IMPLEMENTATION_CLAIM_BOUNDARY = (
    "Sidecar schema plus deterministic cached replay scorer only; no EBT/ARM training, "
    "no model weight update, no live model inference integration, no benchmark speedup, "
    "and no hardware acceleration claim."
)
INFERENCE_SUBSTRATE: JsonDict = {
    "kind": "deterministic_cached_sidecar_replay",
    "sidecar_only": True,
    "live_model_inference": False,
    "live_llm_inference": False,
    "model_weights_loaded": False,
    "generation_performed": False,
    "gpu_required": False,
}


@dataclass(frozen=True)
class ExperimentConfig:
    """Runtime knobs for writing the Exp 3091 terminal artifact."""

    repo_root: Path = REPO_ROOT
    output_path: Path | None = None
    tests_run: Sequence[str] = ()
    clock: Callable[[], float] = time.monotonic

    def artifact_path(self) -> Path:
        return self.output_path or self.repo_root / OUTPUT_REL_PATH


def build_artifact(config: ExperimentConfig | None = None, *, duration_s: float) -> JsonDict:
    """Build the sidecar-only terminal artifact from local schema and fixtures."""

    active = config or ExperimentConfig()
    schema = load_sidecar_schema(active.repo_root)
    records = example_sidecar_records()
    scorer = SidecarReplayScorer(schema=schema)
    first_scores = [scorer.score(record) for record in records]
    second_scores = [scorer.score(record) for record in records]
    replay_is_deterministic = first_scores == second_scores
    schema_ready = _schema_ready(active.repo_root, schema)
    scorer_ready = replay_is_deterministic and all(
        score.inference_substrate["live_model_inference"] is False
        and score.inference_substrate["model_weights_loaded"] is False
        for score in first_scores
    )

    artifact: JsonDict = {
        "artifact": ARTIFACT,
        "schema": SCHEMA,
        "run_date": RUN_DATE,
        "spec_refs": list(SPEC_REFS),
        "adapter_schema_ready": schema_ready,
        "sidecar_replay_scorer_ready": scorer_ready,
        "schema_path": str(SCHEMA_REL_PATH),
        "replay_scorer_path": str(REPLAY_SCORER_REL_PATH),
        "tests_added_or_reused": list(active.tests_run),
        "implementation_claim_boundary": IMPLEMENTATION_CLAIM_BOUNDARY,
        "no_weight_update_claim": True,
        "source_artifacts": _source_artifacts(),
        "inference_substrate": dict(INFERENCE_SUBSTRATE),
        "validation_manifest": {
            "fixture_count": len(records),
            "deterministic_replay": replay_is_deterministic,
            "score_records": [score.to_json() for score in first_scores],
        },
        "non_claims": [
            "no EBT/ARM training",
            "no model weight update",
            "no live model inference integration",
            "no benchmark speedup",
            "no hardware acceleration",
        ],
        "duration_s": round(float(duration_s), 6),
        "honest_verdict": (
            "complete: ebt_arm_sidecar_schema_and_replay_scorer_prototype_only_no_live_inference"
        ),
    }
    validate_artifact(artifact)
    return artifact


def run_experiment(config: ExperimentConfig | None = None, *, write: bool = True) -> JsonDict:
    """Run the deterministic prototype and optionally persist the JSON artifact."""

    active = config or ExperimentConfig()
    started = active.clock()
    duration_s = active.clock() - started
    artifact = build_artifact(active, duration_s=duration_s)
    if write:
        _write_json(active.artifact_path(), artifact)
    return artifact


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """Reject artifacts that overclaim integration, training, or live inference."""

    missing = sorted(set(REQUIRED_ARTIFACT_FIELDS) - set(artifact))
    _require(not missing, f"missing required fields: {missing}")
    _require(artifact.get("adapter_schema_ready") is True, "adapter_schema_ready must be true")
    _require(
        artifact.get("sidecar_replay_scorer_ready") is True,
        "sidecar_replay_scorer_ready must be true",
    )
    _require(
        artifact.get("no_weight_update_claim") is True,
        "no_weight_update_claim must be true",
    )
    boundary = str(artifact.get("implementation_claim_boundary", ""))
    for phrase in (
        "no EBT/ARM training",
        "no model weight update",
        "no benchmark speedup",
        "no hardware acceleration",
    ):
        _require(phrase in boundary, f"implementation_claim_boundary missing {phrase}")
    substrate = artifact.get("inference_substrate")
    _require(isinstance(substrate, Mapping), "inference_substrate must be an object")
    _require(substrate.get("live_model_inference") is False, "live_model_inference must be false")
    _require(substrate.get("live_llm_inference") is False, "live_llm_inference must be false")
    _require(substrate.get("model_weights_loaded") is False, "model_weights_loaded must be false")
    _require(substrate.get("generation_performed") is False, "generation_performed must be false")
    _require(bool(artifact.get("schema_path")), "schema_path must be populated")
    _require(bool(artifact.get("replay_scorer_path")), "replay_scorer_path must be populated")
    sources = artifact.get("source_artifacts")
    _require(isinstance(sources, list) and sources, "source_artifacts must be non-empty")
    _require(
        any(
            isinstance(source, Mapping) and source.get("path") == str(PRIOR_AUDIT_REL_PATH)
            for source in sources
        ),
        "source_artifacts must include the Exp 3073 audit",
    )
    verdict = str(artifact.get("honest_verdict", ""))
    _require(verdict.startswith(SUCCESS_PREFIXES), "honest_verdict must start terminal success")


def _schema_ready(repo_root: Path, schema: Mapping[str, Any]) -> bool:
    schema_path = repo_root / SCHEMA_REL_PATH
    return schema_path.is_file() and REQUIRED_SIDECAR_FIELDS <= set(schema.get("required", []))


def _source_artifacts() -> list[JsonDict]:
    return [
        {
            "id": "exp3073_feasibility_audit",
            "kind": "prior_feasibility_audit",
            "path": str(PRIOR_AUDIT_REL_PATH),
            "claim_boundary": "source_surface_only_no_adapter_implementation_claim",
        },
        {
            "id": "verification_spec",
            "kind": "openspec",
            "path": "openspec/capabilities/verification/spec.md",
            "claim_boundary": "spec_anchor",
        },
        {
            "id": "sidecar_json_schema",
            "kind": "schema",
            "path": str(SCHEMA_REL_PATH),
            "claim_boundary": "data_shape_only",
        },
        {
            "id": "sidecar_replay_scorer",
            "kind": "local_code",
            "path": str(REPLAY_SCORER_REL_PATH),
            "claim_boundary": "deterministic_cached_replay_only",
        },
        {
            "id": "sidecar_tests",
            "kind": "test",
            "path": str(TEST_REL_PATH),
            "claim_boundary": "focused_schema_and_replay_validation",
        },
    ]


def _write_json(path: Path, artifact: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)
