"""Tests for Exp 3091 EBT/ARM sidecar adapter schema prototype.

Spec refs: REQ-VERIFY-3091, SCENARIO-VERIFY-3091.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from carnot.eval import ebt_arm_sidecar_adapter_schema_prototype_v1 as exp
from carnot.inference.ebt_arm_sidecar_adapter import (
    REQUIRED_SIDECAR_FIELDS,
    SidecarReplayScorer,
    example_sidecar_records,
    load_sidecar_schema,
    validate_sidecar_record,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO_ROOT / "openspec" / "capabilities" / "verification" / "spec.md"


class FakeClock:
    def __init__(self) -> None:
        self.value = 100.0

    def __call__(self) -> float:
        self.value += 0.25
        return self.value


def _config(tmp_path: Path) -> exp.ExperimentConfig:
    return exp.ExperimentConfig(
        repo_root=REPO_ROOT,
        output_path=tmp_path / exp.ARTIFACT_FILENAME,
        tests_run=("pytest focused",),
        clock=FakeClock(),
    )


def test_req_verify_3091_spec_anchor_exists() -> None:
    """REQ-VERIFY-3091: the sidecar prototype is anchored in OpenSpec."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-VERIFY-3091" in spec
    assert "SCENARIO-VERIFY-3091" in spec
    assert exp.ARTIFACT_FILENAME in spec
    assert "adapter_schema_ready" in spec
    assert "sidecar_replay_scorer_ready" in spec


def test_req_verify_3091_schema_covers_canonical_sidecar_fields() -> None:
    """REQ-VERIFY-3091: schema rows expose candidate, constraints, and labels."""

    schema = load_sidecar_schema(REPO_ROOT)

    assert REQUIRED_SIDECAR_FIELDS <= set(schema["required"])
    properties = schema["properties"]
    assert {"candidate", "constraints", "energy_terms"} <= set(properties)
    assert {"verifier_feedback", "confidence", "exact_label_reference"} <= set(properties)
    assert {
        "candidate_id",
        "prompt_id",
        "candidate_text",
        "candidate_label",
        "model_id",
        "token_logprobs",
    } <= set(properties["candidate"]["properties"])

    for record in example_sidecar_records():
        validate_sidecar_record(record, schema)


def test_scenario_verify_3091_replay_scorer_is_deterministic_and_cached_only() -> None:
    """SCENARIO-VERIFY-3091: cached rows replay without live model weights."""

    scorer = SidecarReplayScorer()
    records = example_sidecar_records()

    first_pass = [scorer.score(record) for record in records]
    second_pass = [scorer.score(record) for record in records]

    assert first_pass == second_pass
    assert first_pass[0].total_energy < first_pass[1].total_energy
    assert first_pass[0].candidate_id == "sidecar-fixture-correct"
    assert first_pass[1].abstain is True
    assert first_pass[0].inference_substrate["live_model_inference"] is False
    assert first_pass[0].inference_substrate["model_weights_loaded"] is False
    assert first_pass[0].inference_substrate["live_llm_inference"] is False
    assert {term["name"] for term in first_pass[1].energy_terms} == {
        "constraint_violation_energy",
        "arm_sequence_energy",
        "verifier_feedback_energy",
        "confidence_energy",
        "abstention_energy",
        "exact_label_mismatch_energy",
    }
    assert first_pass[0].total_energy == pytest.approx(0.08)
    assert first_pass[1].total_energy == pytest.approx(30.9)


def test_req_verify_3091_validation_rejects_invalid_sidecar_records() -> None:
    """REQ-VERIFY-3091: malformed cached rows are rejected before scoring."""

    schema = load_sidecar_schema(REPO_ROOT)
    valid_record = example_sidecar_records()[0]

    with pytest.raises(ValueError, match="missing required field"):
        validate_sidecar_record({}, schema)
    with pytest.raises(ValueError, match="additional field"):
        validate_sidecar_record(valid_record | {"live_model_inference": True}, schema)
    with pytest.raises(ValueError, match="token_logprobs"):
        invalid = dict(valid_record)
        invalid["candidate"] = dict(valid_record["candidate"], token_logprobs=["not-float"])
        validate_sidecar_record(invalid, schema)


def test_req_verify_3091_validation_rejects_invalid_field_families() -> None:
    """REQ-VERIFY-3091: validation failures cover each sidecar field family."""

    schema = load_sidecar_schema(REPO_ROOT)
    valid_record = example_sidecar_records()[0]

    def changed(mutator_name: str) -> dict[str, object]:
        record = deepcopy(valid_record)
        if mutator_name == "candidate_object":
            record["candidate"] = "not-object"
        elif mutator_name == "constraints_array":
            record["constraints"] = "not-array"
        elif mutator_name == "constraints_empty":
            record["constraints"] = []
        elif mutator_name == "candidate_string":
            record["candidate"]["candidate_text"] = 7
        elif mutator_name == "candidate_non_empty":
            record["candidate"]["candidate_id"] = ""
        elif mutator_name == "constraint_boolean":
            record["constraints"][0]["satisfied"] = "yes"
        elif mutator_name == "constraint_nonnegative":
            record["constraints"][0]["weight"] = -1.0
        elif mutator_name == "verifier_status":
            record["verifier_feedback"][0]["status"] = "maybe"
        elif mutator_name == "confidence_range":
            record["confidence"]["confidence"] = 1.5
        elif mutator_name == "checksum":
            record["exact_label_reference"]["checksum"] = "not-a-sha"
        return record

    cases = (
        ("candidate_object", "object"),
        ("constraints_array", "array"),
        ("constraints_empty", "must not be empty"),
        ("candidate_string", "string"),
        ("candidate_non_empty", "must not be empty"),
        ("constraint_boolean", "boolean"),
        ("constraint_nonnegative", "non-negative"),
        ("verifier_status", "one of"),
        ("confidence_range", "between 0 and 1"),
        ("checksum", "sha256"),
    )

    for mutator_name, match in cases:
        with pytest.raises(ValueError, match=match):
            validate_sidecar_record(changed(mutator_name), schema)


def test_req_verify_3091_artifact_builder_records_claim_boundaries(tmp_path: Path) -> None:
    """REQ-VERIFY-3091: terminal artifact exposes paths and no-training limits."""

    artifact = exp.build_artifact(_config(tmp_path), duration_s=0.125)

    exp.validate_artifact(artifact)
    assert set(exp.REQUIRED_ARTIFACT_FIELDS) <= set(artifact)
    assert artifact["adapter_schema_ready"] is True
    assert artifact["sidecar_replay_scorer_ready"] is True
    assert (REPO_ROOT / artifact["schema_path"]).is_file()
    assert (REPO_ROOT / artifact["replay_scorer_path"]).is_file()
    assert artifact["tests_added_or_reused"] == ["pytest focused"]
    assert artifact["no_weight_update_claim"] is True
    assert "no EBT/ARM training" in artifact["implementation_claim_boundary"]
    assert "no benchmark speedup" in artifact["implementation_claim_boundary"]
    assert artifact["inference_substrate"]["live_model_inference"] is False
    assert artifact["inference_substrate"]["generation_performed"] is False
    assert artifact["honest_verdict"].startswith("complete:")
    assert any(
        source["path"] == "results/experiment_3073_ebt_arm_ebm_adapter_feasibility_audit_v1.json"
        for source in artifact["source_artifacts"]
    )


def test_req_verify_3091_writer_persists_terminal_json(tmp_path: Path) -> None:
    """REQ-VERIFY-3091: run_experiment writes the requested deliverable."""

    config = _config(tmp_path)
    artifact = exp.run_experiment(config, write=True)
    saved = json.loads((tmp_path / exp.ARTIFACT_FILENAME).read_text(encoding="utf-8"))

    assert saved == artifact
    assert artifact["duration_s"] == pytest.approx(0.25)
    assert artifact["schema"] == exp.SCHEMA
    exp.validate_artifact(artifact)


def test_req_verify_3091_artifact_validation_rejects_overclaims(tmp_path: Path) -> None:
    """REQ-VERIFY-3091: validation blocks live inference and training overclaims."""

    artifact = exp.build_artifact(_config(tmp_path), duration_s=0.125)
    exp.validate_artifact(artifact)

    with pytest.raises(ValueError, match="missing required fields"):
        exp.validate_artifact({"honest_verdict": "complete: incomplete"})
    with pytest.raises(ValueError, match="adapter_schema_ready"):
        exp.validate_artifact(artifact | {"adapter_schema_ready": False})
    with pytest.raises(ValueError, match="sidecar_replay_scorer_ready"):
        exp.validate_artifact(artifact | {"sidecar_replay_scorer_ready": False})
    with pytest.raises(ValueError, match="no_weight_update_claim"):
        exp.validate_artifact(artifact | {"no_weight_update_claim": False})
    with pytest.raises(ValueError, match="live_model_inference"):
        exp.validate_artifact(
            artifact
            | {
                "inference_substrate": artifact["inference_substrate"]
                | {"live_model_inference": True}
            }
        )
    with pytest.raises(ValueError, match="honest_verdict"):
        exp.validate_artifact(artifact | {"honest_verdict": "ready"})
