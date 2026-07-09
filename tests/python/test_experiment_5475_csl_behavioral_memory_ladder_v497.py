"""Tests for Exp5475 CSL behavioral memory replay audit.

Spec refs: REQ-LEARN-5475,
SCENARIO-LEARN-5475-SUPPORT-REMOVAL,
SCENARIO-LEARN-5475-IRRELEVANT-MEMORY,
SCENARIO-LEARN-5475-LADDER,
SCENARIO-LEARN-5475-ARTIFACT.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from carnot import experiment_5475_csl_behavioral_memory_ladder_v497 as mod


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec/capabilities/self-learning/spec.md"
RESULT_PATH = REPO / mod.RESULT_RELATIVE_PATH
TEST_COMMAND = (
    ".venv/bin/pytest "
    "tests/python/test_experiment_5475_csl_behavioral_memory_ladder_v497.py "
    "-q --no-cov -n 0"
)
COVERAGE_COMMAND = (
    ".venv/bin/coverage run "
    "--include=python/carnot/experiment_5475_csl_behavioral_memory_ladder_v497.py "
    "-m pytest tests/python/test_experiment_5475_csl_behavioral_memory_ladder_v497.py "
    "-q --no-cov -n 0 && .venv/bin/coverage report "
    "--include=python/carnot/experiment_5475_csl_behavioral_memory_ladder_v497.py "
    "--fail-under=100"
)


def _complete_artifact() -> dict[str, object]:
    return mod.build_artifact(root=REPO, tests_run=[TEST_COMMAND, COVERAGE_COMMAND])


def test_req_learn_5475_spec_declares_behavioral_memory_audit() -> None:
    """REQ-LEARN-5475: OpenSpec anchors the no-LLM memory audit contract."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    section = spec[spec.index("## REQ-LEARN-5475") :]
    normalized = " ".join(section.split())

    for marker in (
        "REQ-LEARN-5475",
        "SCENARIO-LEARN-5475-SUPPORT-REMOVAL",
        "SCENARIO-LEARN-5475-IRRELEVANT-MEMORY",
        "SCENARIO-LEARN-5475-LADDER",
        "SCENARIO-LEARN-5475-ARTIFACT",
        str(mod.RESULT_RELATIVE_PATH),
        mod.INFERENCE_SUBSTRATE,
        "support removal",
        "paraphrase robustness",
        "locality",
        "conflict handling",
        "downstream action use",
        "stale memory rejection",
    ):
        assert marker in section

    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert f"`{field}`" in section
    assert " ".join(mod.LADDER_AXES) in normalized


def test_req_learn_5475_replay_fixtures_cover_axes_and_shared_variants() -> None:
    """REQ-LEARN-5475-1/2/3: rows cover every axis with auditable variants."""

    fixtures = mod.build_replay_fixtures()
    evaluation = mod.evaluate_replay(fixtures)
    row_ids_by_variant = evaluation["row_ids_by_variant"]

    assert len(fixtures) >= len(mod.LADDER_AXES)
    assert {row["axis"] for row in fixtures} == set(mod.LADDER_AXES)
    assert set(row_ids_by_variant) == set(mod.VARIANT_NAMES)
    assert len({tuple(ids) for ids in row_ids_by_variant.values()}) == 1
    assert evaluation["replay_fixture_count"] == len(fixtures)

    for result in evaluation["row_results"]:
        assert result["variant"] in mod.VARIANT_NAMES
        assert result["memory_retrieval_ids"] == [
            record["memory_id"] for record in result["retrieved_memory_records"]
        ]
        assert isinstance(result["provenance"], list)
        assert isinstance(result["decision_path"], list)
        assert isinstance(result["accepted_memory_records"], list)
        assert isinstance(result["rejected_memory_records"], list)
        assert isinstance(result["rollback_pointers"], list)
        assert result["exact_validator_results"]["authority"] == mod.EXACT_VALIDATOR_AUTHORITY
        assert result["final_authority_bypassed"] is False
        assert result["row_checksum"] == mod.row_checksum(result)


def test_scenario_learn_5475_removed_support_cannot_pass_memory_use_claim() -> None:
    """SCENARIO-LEARN-5475-SUPPORT-REMOVAL: removed support fails closed."""

    fixture = mod.fixture_by_axis(mod.build_replay_fixtures(), "support_removal")
    removed = deepcopy(fixture["memory_records"][0])
    removed["answer"] = fixture["expected_answer"]
    claim = mod.validate_memory_use_claim(
        fixture,
        accepted_memory_records=[removed],
        selected_answer=fixture["expected_answer"],
        memory_use_claimed=True,
    )
    result = mod.evaluate_variant(fixture, mod.GOVERNED_VARIANT)

    assert removed["support_status"] == "removed"
    assert claim["answer_accepted"] is True
    assert claim["memory_use_claim_valid"] is False
    assert "support_removed_memory_accepted" in claim["claim_failure_reasons"]
    assert result["accepted_memory_records"] == []
    assert {record["memory_id"] for record in result["rejected_memory_records"]} >= {
        removed["memory_id"]
    }
    assert mod.row_axis_pass(fixture, result) is True


def test_scenario_learn_5475_irrelevant_memory_cannot_pass_claim() -> None:
    """SCENARIO-LEARN-5475-IRRELEVANT-MEMORY: wrong-locality memory is rejected."""

    fixture = mod.fixture_by_axis(mod.build_replay_fixtures(), "locality")
    irrelevant = deepcopy(fixture["memory_records"][0])
    irrelevant["answer"] = fixture["expected_answer"]
    claim = mod.validate_memory_use_claim(
        fixture,
        accepted_memory_records=[irrelevant],
        selected_answer=fixture["expected_answer"],
        memory_use_claimed=True,
    )
    result = mod.evaluate_variant(fixture, mod.KAN_VARIANT)

    assert irrelevant["locality_key"] != fixture["locality_key"]
    assert claim["answer_accepted"] is True
    assert claim["memory_use_claim_valid"] is False
    assert "irrelevant_memory_accepted" in claim["claim_failure_reasons"]
    assert {record["memory_id"] for record in result["rejected_memory_records"]} >= {
        irrelevant["memory_id"]
    }
    assert result["exact_validator_results"]["memory_use_claim_valid"] is True
    assert mod.row_axis_pass(fixture, result) is True


def test_scenario_learn_5475_ladder_metrics_and_baselines_are_exact() -> None:
    """SCENARIO-LEARN-5475-LADDER: six behavioral axes produce pass rates."""

    artifact = _complete_artifact()

    mod.validate_artifact(artifact)
    assert artifact["replay_fixture_count"] == len(mod.LADDER_AXES)
    assert artifact["support_removal_pass_rate"] == pytest.approx(1.0)
    assert artifact["paraphrase_robustness_rate"] == pytest.approx(1.0)
    assert artifact["locality_pass_rate"] == pytest.approx(1.0)
    assert artifact["conflict_handling_pass_rate"] == pytest.approx(1.0)
    assert artifact["downstream_action_use_rate"] == pytest.approx(1.0)
    assert artifact["stale_memory_rejection_rate"] == pytest.approx(1.0)
    assert artifact["no_memory_baseline_score"] == pytest.approx(0.166667)
    assert artifact["naive_icl_baseline_score"] == pytest.approx(0.333333)
    assert artifact["governed_memory_score"] == pytest.approx(1.0)
    assert artifact["kan_surrogate_policy_score"] == pytest.approx(1.0)
    assert artifact["csl_behavioral_memory_ready"] is True
    assert artifact["model_weight_mutation"] is False
    assert artifact["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert artifact["random_seed"] == mod.RANDOM_SEED
    assert artifact["honest_verdict"].startswith("complete:")


def test_scenario_learn_5475_artifact_write_and_repository_replay_match(
    tmp_path: Path,
) -> None:
    """SCENARIO-LEARN-5475-ARTIFACT: run() writes stable deliverable JSON."""

    destination = tmp_path / mod.RESULT_RELATIVE_PATH
    artifact = mod.run(
        root=REPO,
        result_path=destination,
        tests_run=[TEST_COMMAND, COVERAGE_COMMAND],
        write=True,
    )
    dry_run = mod.run(
        root=REPO,
        result_path=tmp_path / "dry-run.json",
        tests_run=[TEST_COMMAND, COVERAGE_COMMAND],
        write=False,
    )
    repo_result = json.loads(RESULT_PATH.read_text(encoding="utf-8"))
    repo_replay = mod.build_artifact(root=REPO, tests_run=repo_result["tests_run"])

    assert json.loads(destination.read_text(encoding="utf-8")) == artifact
    assert dry_run == artifact
    assert not (tmp_path / "dry-run.json").exists()
    assert repo_result == repo_replay
    assert repo_result["csl_behavioral_memory_ready"] is True
    assert repo_result["model_weight_mutation"] is False


def test_req_learn_5475_validation_rejects_schema_and_evidence_drift() -> None:
    """REQ-LEARN-5475-5/6: validator fails closed on readiness drift."""

    artifact = _complete_artifact()
    fixture = mod.build_replay_fixtures()[0]
    empty_claim = mod.validate_memory_use_claim(
        fixture,
        accepted_memory_records=[],
        selected_answer=fixture["expected_answer"],
        memory_use_claimed=True,
    )
    rejected_axis = deepcopy(artifact["row_results"][0])
    rejected_axis["exact_validator_results"]["accepted"] = False
    unknown_axis_fixture = deepcopy(fixture)
    unknown_axis_fixture["axis"] = "unknown"

    assert empty_claim["memory_use_claim_valid"] is False
    assert "no_accepted_memory_evidence" in empty_claim["claim_failure_reasons"]
    assert mod.row_axis_pass(fixture, rejected_axis) is False
    assert mod.row_axis_pass(unknown_axis_fixture, artifact["row_results"][2]) is False
    with pytest.raises(ValueError, match="unknown fixture axis"):
        mod.fixture_by_axis(mod.build_replay_fixtures(), "missing-axis")

    scalar_cases = [
        ("support_removal_pass_rate", 0.0, "support_removal_pass_rate"),
        ("paraphrase_robustness_rate", 0.0, "paraphrase_robustness_rate"),
        ("locality_pass_rate", 0.0, "locality_pass_rate"),
        ("conflict_handling_pass_rate", 0.0, "conflict_handling_pass_rate"),
        ("downstream_action_use_rate", 0.0, "downstream_action_use_rate"),
        ("stale_memory_rejection_rate", 0.0, "stale_memory_rejection_rate"),
        ("no_memory_baseline_score", 1.0, "baseline ordering"),
        ("naive_icl_baseline_score", 1.0, "baseline ordering"),
        ("governed_memory_score", 0.0, "governed_memory_score"),
        ("csl_behavioral_memory_ready", False, "csl_behavioral_memory_ready"),
        ("model_weight_mutation", True, "model_weight_mutation"),
        ("inference_substrate", "live_llm_inference", "inference_substrate"),
        ("random_seed", 1, "random_seed"),
        ("honest_verdict", "done", "honest_verdict"),
        ("research_conductor_modified", True, "scripts/research_conductor.py"),
    ]
    for field, value, expected in scalar_cases:
        bad = deepcopy(artifact)
        bad[field] = value
        with pytest.raises(ValueError, match=expected):
            mod.validate_artifact(bad)

    missing = deepcopy(artifact)
    missing.pop("replay_fixture_count")
    with pytest.raises(ValueError, match="missing required fields"):
        mod.validate_artifact(missing)

    bad_rows = deepcopy(artifact)
    bad_rows["row_results"][0]["final_authority_bypassed"] = True
    with pytest.raises(ValueError, match="final authority"):
        mod.validate_artifact(bad_rows)

    bad_checksum = deepcopy(artifact)
    bad_checksum["row_results"][0]["row_checksum"] = "sha256:bad"
    with pytest.raises(ValueError, match="row checksum"):
        mod.validate_artifact(bad_checksum)

    bad_variant_ids = deepcopy(artifact)
    bad_variant_ids["row_ids_by_variant"][mod.NO_MEMORY_VARIANT] = ["only-one-row"]
    with pytest.raises(ValueError, match="identical row IDs"):
        mod.validate_artifact(bad_variant_ids)

    missing_variant_ids = deepcopy(artifact)
    missing_variant_ids["row_ids_by_variant"].pop(mod.NO_MEMORY_VARIANT)
    with pytest.raises(ValueError, match="identical row IDs"):
        mod.validate_artifact(missing_variant_ids)

    empty_rows = deepcopy(artifact)
    empty_rows["row_results"] = []
    with pytest.raises(ValueError, match="row_results"):
        mod.validate_artifact(empty_rows)

    missing_evidence = deepcopy(artifact)
    missing_evidence["row_results"][0].pop("provenance")
    missing_evidence["row_results"][0]["row_checksum"] = mod.row_checksum(
        missing_evidence["row_results"][0]
    )
    with pytest.raises(ValueError, match="row evidence fields"):
        mod.validate_artifact(missing_evidence)

    bad_authority = deepcopy(artifact)
    bad_authority["row_results"][0]["exact_validator_results"]["authority"] = "model_self_verdict"
    bad_authority["row_results"][0]["row_checksum"] = mod.row_checksum(
        bad_authority["row_results"][0]
    )
    with pytest.raises(ValueError, match="exact validator"):
        mod.validate_artifact(bad_authority)

    bad_repro = deepcopy(artifact)
    bad_repro["reproducibility_checksum"] = "sha256:bad"
    with pytest.raises(ValueError, match="reproducibility_checksum"):
        mod.validate_artifact(bad_repro)
