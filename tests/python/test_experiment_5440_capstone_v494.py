"""Tests for Exp5440 .494 terminal capstone truth table.

Spec refs: REQ-CAPSTONE-5440, SCENARIO-CAPSTONE-5440,
SCENARIO-CAPSTONE-5440-MISSING-INPUT,
SCENARIO-CAPSTONE-5440-FIELD-PRINCIPLES.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from shutil import copyfile
import subprocess

import pytest

from carnot import experiment_5440_capstone_v494 as exp


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec/capabilities/capstone/spec.md"
RESULT_PATH = REPO / exp.RESULT_RELATIVE_PATH


def _copy_inputs(root: Path) -> None:
    for relative in exp.EXPECTED_INPUT_PATHS:
        source = REPO / relative
        if not source.exists() or source.is_dir():
            continue
        destination = root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        copyfile(source, destination)


def _lane_names(rows: list[dict[str, object]]) -> list[str]:
    return [str(row["lane"]) for row in rows]


def test_req_capstone_5440_spec_declares_capstone_contract() -> None:
    """REQ-CAPSTONE-5440: OpenSpec anchors the .494 capstone schema."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    section = spec[spec.index("REQ-CAPSTONE-5440") :]
    normalized = " ".join(section.split())

    for marker in (
        "REQ-CAPSTONE-5440",
        "SCENARIO-CAPSTONE-5440",
        "SCENARIO-CAPSTONE-5440-MISSING-INPUT",
        "SCENARIO-CAPSTONE-5440-FIELD-PRINCIPLES",
        str(exp.RESULT_RELATIVE_PATH),
        "Exp5430 structured corrigendum plus",
        "Exp5431 structured taxonomy replication as headline-ready",
        "Exp5437 ARC live reinduction as",
        "`hardware_speedup_claim` = \"no unsupported acceleration",
        "`future_token_signal_allowed` = \"token/internal lane closure",
    ):
        assert marker in section

    for field, principle in exp.FIELD_PRINCIPLES.items():
        assert f"`{field}`" in section
        assert " ".join(principle.split()) in normalized


def test_scenario_capstone_5440_builds_terminal_truth_table_from_actual_artifacts() -> None:
    """SCENARIO-CAPSTONE-5440: actual .494 artifacts produce honest lane buckets."""

    artifact = exp.build_artifact(
        root=REPO,
        tests_run=[{"command": "unit capstone 5440", "outcome": "passed"}],
    )

    exp.validate_artifact(artifact)
    assert artifact["milestone"] == exp.MILESTONE
    assert artifact["upstream_artifacts_missing"] == []
    assert artifact["upstream_artifacts_read"] == list(exp.EXPECTED_INPUT_PATHS)
    assert artifact["inference_substrate"] == exp.INFERENCE_SUBSTRATE
    assert artifact["honest_verdict"].startswith("complete:")

    assert _lane_names(artifact["headline_ready_lanes"]) == [
        "structured_corrigendum",
        "structured_taxonomy_replication",
        "ontology_softlogic_memory",
        "verified_workflow_memory_csl",
        "csl_memory_transfer_stress",
    ]
    assert _lane_names(artifact["bounded_lanes"]) == [
        "active_constraint_diversity_lns",
        "pbit_polarfire_timing_variance",
        "kan_ontology_certificates",
    ]
    assert _lane_names(artifact["honest_null_lanes"]) == [
        "arc_live_reinduction_levelup",
        "hardware_speedup_claim",
    ]
    assert _lane_names(artifact["blocked_lanes"]) == ["token_internal_feature_lane_closed"]

    rows = {row["lane"]: row for row in artifact["truth_table"]}
    assert rows["structured_corrigendum"]["classification"] == "headline_ready"
    assert rows["structured_taxonomy_replication"]["classification"] == "headline_ready"
    assert rows["ontology_softlogic_memory"]["classification"] == "headline_ready"
    assert rows["active_constraint_diversity_lns"]["classification"] == "bounded"
    assert rows["pbit_polarfire_timing_variance"]["classification"] == "bounded"
    assert rows["verified_workflow_memory_csl"]["classification"] == "headline_ready"
    assert rows["csl_memory_transfer_stress"]["classification"] == "headline_ready"
    assert rows["arc_live_reinduction_levelup"]["classification"] == "honest_null"
    assert rows["kan_ontology_certificates"]["classification"] == "bounded"
    assert rows["token_internal_feature_lane_closed"]["classification"] == "blocked"
    assert rows["hardware_speedup_claim"]["classification"] == "honest_null"

    assert artifact["arc_new_level_banked"] is False
    assert artifact["hardware_speedup_claim"] is False
    assert artifact["future_token_signal_allowed"] is False
    assert artifact["local_sota_gguf_receipts_valid"] is True
    assert artifact["research_roadmap_yaml_unchanged"] is True
    assert artifact["conductor_unchanged"] is True

    assert [row["target"] for row in artifact["next_recommendations"]] == [
        "structured_verification",
        "continuous_self_learning",
        "arc_live_levelup",
        "pbit_hardware_timing",
        "token_internal_backend",
    ]
    assert "ARC no-bank" in artifact["honest_verdict"]
    assert "no hardware speedup" in artifact["honest_verdict"]
    assert "token/internal lane closed" in artifact["honest_verdict"]


def test_scenario_capstone_5440_missing_inputs_fail_closed(tmp_path: Path) -> None:
    """SCENARIO-CAPSTONE-5440-MISSING-INPUT: absent inputs become missing lanes."""

    _copy_inputs(tmp_path)
    (tmp_path / exp.EXP5436).unlink()
    (tmp_path / exp.EXP5433).write_text("{", encoding="utf-8")

    artifact = exp.build_artifact(root=tmp_path)

    exp.validate_artifact(artifact)
    assert artifact["honest_verdict"].startswith("blocked:")
    assert artifact["upstream_artifacts_missing"] == [exp.EXP5433, exp.EXP5436]
    assert _lane_names(artifact["headline_ready_lanes"]) == [
        "structured_corrigendum",
        "structured_taxonomy_replication",
        "ontology_softlogic_memory",
        "verified_workflow_memory_csl",
    ]
    assert _lane_names(artifact["bounded_lanes"]) == [
        "pbit_polarfire_timing_variance",
        "kan_ontology_certificates",
    ]
    assert any(
        row["lane"] == "active_constraint_diversity_lns" and row["classification"] == "missing"
        for row in artifact["missing_lanes"]
    )
    assert any(
        row["lane"] == "csl_memory_transfer_stress" and row["classification"] == "missing"
        for row in artifact["missing_lanes"]
    )


def test_req_capstone_5440_run_writes_stable_json(tmp_path: Path) -> None:
    """REQ-CAPSTONE-5440: run() writes a deterministic capstone deliverable."""

    tests_run = [
        {
            "command": ".venv/bin/pytest tests/python/test_experiment_5440_capstone_v494.py -q",
            "outcome": "passed",
        },
        {
            "command": (
                ".venv/bin/coverage run "
                "--include=python/carnot/experiment_5440_capstone_v494.py "
                "-m pytest tests/python/test_experiment_5440_capstone_v494.py "
                "-q --no-cov -n 0"
            ),
            "outcome": "passed",
        },
        {
            "command": (
                ".venv/bin/coverage report "
                "--include=python/carnot/experiment_5440_capstone_v494.py "
                "--fail-under=100"
            ),
            "outcome": "passed",
        },
        {"command": ".venv/bin/pytest tests/python -q", "outcome": "passed"},
    ]
    result_path = tmp_path / exp.RESULT_RELATIVE_PATH

    artifact = exp.run(root=REPO, result_path=result_path, tests_run=tests_run)

    assert json.loads(result_path.read_text(encoding="utf-8")) == artifact
    assert artifact == exp.build_artifact(root=REPO, tests_run=tests_run)
    assert artifact["result_path"] == str(exp.RESULT_RELATIVE_PATH)
    assert artifact["field_principles"] == exp.FIELD_PRINCIPLES
    assert artifact["spec_refs"] == list(exp.SPEC_REFS)
    assert artifact["reproducibility_checksum"].startswith("sha256:")
    exp.validate_artifact(artifact)


def test_req_capstone_5440_committed_result_matches_replay() -> None:
    """REQ-CAPSTONE-5440: checked-in result is stable under deterministic replay."""

    result = json.loads(RESULT_PATH.read_text(encoding="utf-8"))
    replay = exp.build_artifact(root=REPO, tests_run=result["tests_run"])

    assert result == replay


def test_req_capstone_5440_validation_rejects_overclaims() -> None:
    """REQ-CAPSTONE-5440: validation rejects schema drift and overclaims."""

    artifact = exp.build_artifact(root=REPO)

    missing_field = deepcopy(artifact)
    missing_field.pop("upstream_artifacts_read")
    with pytest.raises(ValueError, match="missing required fields"):
        exp.validate_artifact(missing_field)

    bad_milestone = deepcopy(artifact)
    bad_milestone["milestone"] = "2026.07.493"
    with pytest.raises(ValueError, match="milestone"):
        exp.validate_artifact(bad_milestone)

    bad_field_principles = deepcopy(artifact)
    bad_field_principles["field_principles"] = {}
    with pytest.raises(ValueError, match="field_principles"):
        exp.validate_artifact(bad_field_principles)

    bad_bool = deepcopy(artifact)
    bad_bool["arc_new_level_banked"] = "false"
    with pytest.raises(ValueError, match="arc_new_level_banked must be boolean"):
        exp.validate_artifact(bad_bool)

    bad_arc = deepcopy(artifact)
    bad_arc["arc_new_level_banked"] = True
    with pytest.raises(ValueError, match="arc_new_level_banked"):
        exp.validate_artifact(bad_arc)

    bad_speedup = deepcopy(artifact)
    bad_speedup["hardware_speedup_claim"] = True
    with pytest.raises(ValueError, match="hardware_speedup_claim"):
        exp.validate_artifact(bad_speedup)

    bad_token = deepcopy(artifact)
    bad_token["future_token_signal_allowed"] = True
    with pytest.raises(ValueError, match="future_token_signal_allowed"):
        exp.validate_artifact(bad_token)

    bad_receipts = deepcopy(artifact)
    bad_receipts["local_sota_gguf_receipts_valid"] = False
    with pytest.raises(ValueError, match="local_sota_gguf_receipts_valid"):
        exp.validate_artifact(bad_receipts)

    bad_headline = deepcopy(artifact)
    bad_headline["headline_ready_lanes"].append(bad_headline["blocked_lanes"][0])
    with pytest.raises(ValueError, match="lane bucket"):
        exp.validate_artifact(bad_headline)

    bad_substrate = deepcopy(artifact)
    bad_substrate["inference_substrate"] = "live_llm_inference"
    with pytest.raises(ValueError, match="inference_substrate"):
        exp.validate_artifact(bad_substrate)

    bad_roadmap = deepcopy(artifact)
    bad_roadmap["research_roadmap_yaml_unchanged"] = False
    with pytest.raises(ValueError, match="research_roadmap_yaml_unchanged"):
        exp.validate_artifact(bad_roadmap)

    bad_conductor = deepcopy(artifact)
    bad_conductor["conductor_unchanged"] = False
    with pytest.raises(ValueError, match="conductor_unchanged"):
        exp.validate_artifact(bad_conductor)

    bad_lanes = deepcopy(artifact)
    bad_lanes["truth_table"] = list(reversed(bad_lanes["truth_table"]))
    with pytest.raises(ValueError, match="truth_table lane order"):
        exp.validate_artifact(bad_lanes)

    bad_row_classification = deepcopy(artifact)
    bad_row_classification["truth_table"][0]["classification"] = "surprise"
    with pytest.raises(ValueError, match="truth_table classification"):
        exp.validate_artifact(bad_row_classification)

    bad_row_evidence = deepcopy(artifact)
    bad_row_evidence["truth_table"][0]["evidence"] = []
    with pytest.raises(ValueError, match="truth_table row evidence"):
        exp.validate_artifact(bad_row_evidence)

    bad_headline_drift = deepcopy(artifact)
    bad_headline_drift["truth_table"][6]["classification"] = "bounded"
    bad_headline_drift.update(exp.bucket_lanes(bad_headline_drift["truth_table"]))
    with pytest.raises(ValueError, match="headline_ready_lanes"):
        exp.validate_artifact(bad_headline_drift)

    bad_missing_verdict = deepcopy(artifact)
    bad_missing_verdict["upstream_artifacts_missing"] = [exp.EXP5430]
    bad_missing_verdict["honest_verdict"] = "complete: not blocked"
    with pytest.raises(ValueError, match="blocked: when inputs are missing"):
        exp.validate_artifact(bad_missing_verdict)

    bad_complete_verdict = deepcopy(artifact)
    bad_complete_verdict["honest_verdict"] = "blocked: not complete"
    with pytest.raises(ValueError, match="complete: when all inputs are readable"):
        exp.validate_artifact(bad_complete_verdict)

    bad_recommendations = deepcopy(artifact)
    bad_recommendations["next_recommendations"] = []
    with pytest.raises(ValueError, match="next_recommendations"):
        exp.validate_artifact(bad_recommendations)

    bad_checksum = deepcopy(artifact)
    bad_checksum["reproducibility_checksum"] = "sha256:bad"
    with pytest.raises(ValueError, match="reproducibility_checksum"):
        exp.validate_artifact(bad_checksum)

    bad_verdict = deepcopy(artifact)
    bad_verdict["honest_verdict"] = "done"
    with pytest.raises(ValueError, match="honest_verdict"):
        exp.validate_artifact(bad_verdict)


def test_req_capstone_5440_helper_branches_and_cli(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-CAPSTONE-5440: helper branches stay covered and fail closed."""

    _copy_inputs(tmp_path)
    payloads, read, missing, errors = exp.read_inputs(tmp_path)
    assert len(payloads) == len(exp.RESULT_ARTIFACT_PATHS)
    assert read == list(exp.EXPECTED_INPUT_PATHS)
    assert missing == []
    assert errors == []

    original_read_text = Path.read_text

    def _raise_for_status(path: Path, *args: object, **kwargs: object) -> str:
        if str(path).endswith("ops/status.md"):
            raise OSError("permission denied")
        return original_read_text(path, *args, **kwargs)

    monkeypatch.setattr(exp.Path, "read_text", _raise_for_status)
    _, _, missing, errors = exp.read_inputs(tmp_path)
    assert "ops/status.md" in missing
    assert any(
        error["path"] == "ops/status.md" and str(error["classification"]).startswith("read_error:")
        for error in errors
    )
    monkeypatch.setattr(exp.Path, "read_text", original_read_text)

    scalar = tmp_path / exp.EXP5434
    scalar.write_text("[]", encoding="utf-8")
    _, read, missing, errors = exp.read_inputs(tmp_path)
    assert exp.EXP5434 not in read
    assert exp.EXP5434 in missing
    assert {"path": exp.EXP5434, "classification": "not_json_object"} in errors

    assert exp.unwrap({"principle": "p", "value": False}) is False
    assert exp.unwrap("plain") == "plain"
    assert exp.json_ready((Path("a"), Path("b"))) == ["a", "b"]
    assert exp.lane_bucket_name("headline_ready") == "headline_ready_lanes"
    assert exp.lane_bucket_name("bounded") == "bounded_lanes"
    assert exp.lane_bucket_name("honest_null") == "honest_null_lanes"
    assert exp.lane_bucket_name("blocked") == "blocked_lanes"
    assert exp.lane_bucket_name("missing") == "missing_lanes"
    assert exp.lane_bucket_name("other") == "blocked_lanes"

    assert exp.flag_reasons({}) == []
    assert exp.flag_reasons({"flagged_adversarial": True, "corrigendum_pending": [{"kind": "x"}]}) == [
        "flagged_adversarial",
        "corrigendum_pending",
    ]
    flagged_row = exp.classify_lane(
        exp.LANE_SPECS[0],
        {exp.EXP5430: {"flagged_adversarial": True, "corrigendum_pending": [{"kind": "x"}]}},
        [],
    )
    assert flagged_row["classification"] == "blocked"
    assert flagged_row["evidence"]["flag_reasons"] == ["flagged_adversarial", "corrigendum_pending"]

    assert exp.local_sota_gguf_receipts_valid({}) is False
    assert exp.local_sota_gguf_receipts_valid({exp.EXP5430: {"gpu_offload_verified": True}}) is False
    assert exp.local_sota_gguf_receipts_valid(
        {
            exp.EXP5430: {
                "gpu_offload_verified": True,
                "gpu_offload_receipt": {"offload_evidence": True},
                "model_specs": [],
            },
        }
    ) is False
    bad_spec_payload = {
        "gpu_offload_verified": True,
        "gpu_offload_receipt": {"offload_evidence": True},
        "model_specs": ["not a mapping"],
    }
    assert exp.local_sota_gguf_receipts_valid({exp.EXP5430: bad_spec_payload}) is False
    bad_status_payload = {
        "gpu_offload_verified": True,
        "gpu_offload_receipt": {"offload_evidence": True},
        "model_specs": [{"status": "missing", "hf_id": "model-GGUF"}],
    }
    assert exp.local_sota_gguf_receipts_valid({exp.EXP5430: bad_status_payload}) is False
    bad_marker_payload = {
        "gpu_offload_verified": True,
        "gpu_offload_receipt": {"offload_evidence": True},
        "model_specs": [{"status": "local_gguf_resolved", "hf_id": "plain-model", "model_path": "/tmp/model.bin"}],
    }
    assert exp.local_sota_gguf_receipts_valid({exp.EXP5430: bad_marker_payload}) is False

    assert exp.recursive_key_true({"outer": [{"future_token_signal_allowed": True}]}, "future_token_signal_allowed")
    assert exp.lane_classification("unknown_lane", {}, {}) == "blocked"
    assert exp.classify_lane(
        exp.LANE_SPECS[0],
        {},
        [exp.EXP5430],
    )["classification"] == "missing"

    def _raise_oserror(*_args: object, **_kwargs: object) -> subprocess.CompletedProcess[str]:
        raise OSError("git unavailable")

    monkeypatch.setattr(exp.subprocess, "run", _raise_oserror)
    assert exp.git_path_unchanged(REPO, "research-roadmap.yaml") is True

    cli_result = tmp_path / "cli" / exp.RESULT_RELATIVE_PATH
    assert exp.main(["--root", str(REPO), "--result-path", str(cli_result)]) == 0
    assert json.loads(cli_result.read_text(encoding="utf-8"))["milestone"] == exp.MILESTONE
