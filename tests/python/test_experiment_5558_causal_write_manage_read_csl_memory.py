"""Tests for Exp5558 causal write-manage-read CSL memory.

Spec refs: REQ-LEARN-5558,
SCENARIO-LEARN-5558-WRITE,
SCENARIO-LEARN-5558-MANAGE,
SCENARIO-LEARN-5558-READ,
SCENARIO-LEARN-5558-FORBIDDEN,
SCENARIO-LEARN-5558-ARTIFACT.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from carnot import experiment_5558_causal_write_manage_read_csl_memory as exp


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec/capabilities/self-learning/spec.md"
RESULT_PATH = REPO / exp.RESULT_RELATIVE_PATH
TEST_COMMAND = (
    ".venv/bin/pytest "
    "tests/python/test_experiment_5558_causal_write_manage_read_csl_memory.py "
    "-q --no-cov -n 0"
)
COVERAGE_COMMAND = (
    ".venv/bin/coverage run "
    "--include=python/carnot/experiment_5558_causal_write_manage_read_csl_memory.py "
    "-m pytest tests/python/test_experiment_5558_causal_write_manage_read_csl_memory.py "
    "-q --no-cov -n 0 && .venv/bin/coverage report "
    "--include=python/carnot/experiment_5558_causal_write_manage_read_csl_memory.py "
    "--fail-under=100"
)
FULL_TEST_COMMAND = ".venv/bin/pytest tests/python -q"
TESTS_ADDED_OR_REUSED = [TEST_COMMAND, COVERAGE_COMMAND, FULL_TEST_COMMAND]


def _artifact() -> dict[str, object]:
    return exp.build_artifact(root=REPO, tests_added_or_reused=TESTS_ADDED_OR_REUSED)


def test_req_learn_5558_spec_declares_write_manage_read_contract() -> None:
    """REQ-LEARN-5558: OpenSpec anchors the causal memory fixture."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    section = spec[spec.index("## REQ-LEARN-5558") :]

    for marker in (
        "REQ-LEARN-5558",
        "SCENARIO-LEARN-5558-WRITE",
        "SCENARIO-LEARN-5558-MANAGE",
        "SCENARIO-LEARN-5558-READ",
        "SCENARIO-LEARN-5558-FORBIDDEN",
        "SCENARIO-LEARN-5558-ARTIFACT",
        str(exp.RESULT_RELATIVE_PATH),
        str(exp.UPSTREAM_FIVE_ARM_CORRIGENDUM),
        exp.INFERENCE_SUBSTRATE,
        "write-manage-read taxonomy",
        "forbidden-direction reuse rate",
    ):
        assert marker in section
    for field in exp.REQUIRED_ARTIFACT_FIELDS:
        assert f"`{field}`" in section
        assert exp.FIELD_PRINCIPLES[field]


def test_scenario_learn_5558_write_filters_unverified_noise() -> None:
    """SCENARIO-LEARN-5558-WRITE: only verified evidence is written."""

    fixture = exp.build_fixture()
    write = exp.write_memory(fixture["events"])

    assert write["write_filter_precision"] == pytest.approx(1.0)
    assert {entry["event_id"] for entry in write["accepted_entries"]} == {
        "evt-cache-success",
        "evt-timeout-stale",
        "evt-timeout-fresh",
        "evt-api-forbidden",
        "evt-secret-transfer",
        "evt-secret-correction",
        "evt-auth-policy",
    }
    assert write["rejected_candidates"] == [
        {
            "event_id": "evt-pagination-noise",
            "context_key": "pagination:index",
            "rejected_reason": "unverified",
        }
    ]
    assert all(entry["verified"] for entry in write["accepted_entries"])


def test_scenario_learn_5558_manage_forgets_stale_and_contradicted_rows() -> None:
    """SCENARIO-LEARN-5558-MANAGE: stale rows are removed before aligned reads."""

    fixture = exp.build_fixture()
    write = exp.write_memory(fixture["events"])
    managed = exp.manage_memory(write["accepted_entries"], fixture["decisions"])
    active_ids = {entry["memory_id"] for entry in managed["active_entries"]}
    forgotten = {entry["memory_id"]: entry["forget_reason"] for entry in managed["forgotten_entries"]}

    assert managed["manage_forget_precision"] == pytest.approx(1.0)
    assert forgotten == {
        "mem-timeout-stale": "stale",
        "mem-secret-transfer": "contradicted",
    }
    assert "mem-timeout-fresh" in active_ids
    assert "mem-secret-correction" in active_ids
    assert "mem-timeout-stale" not in active_ids
    assert "mem-secret-transfer" not in active_ids


def test_scenario_learn_5558_read_changes_actions_and_deflects_conflicts() -> None:
    """SCENARIO-LEARN-5558-READ: causal memory changes later behavior."""

    fixture = exp.build_fixture()
    evaluation = exp.evaluate_fixture(fixture)
    aligned_rows = evaluation["arm_results"][exp.ALIGNED_CAUSAL_MEMORY_ARM]
    no_memory_rows = evaluation["arm_results"][exp.NO_MEMORY_ARM]
    always_full_rows = evaluation["arm_results"][exp.ALWAYS_FULL_MEMORY_ARM]
    aligned_by_id = {row["decision_id"]: row for row in aligned_rows}
    no_memory_by_id = {row["decision_id"]: row for row in no_memory_rows}
    always_full_by_id = {row["decision_id"]: row for row in always_full_rows}

    assert evaluation["scores"][exp.ALIGNED_CAUSAL_MEMORY_ARM] == pytest.approx(1.0)
    assert evaluation["scores"][exp.NO_MEMORY_ARM] == pytest.approx(0.1666666667)
    assert evaluation["scores"][exp.SHUFFLED_MEMORY_ARM] == pytest.approx(0.0)
    assert evaluation["scores"][exp.ALWAYS_FULL_MEMORY_ARM] == pytest.approx(0.6666666667)
    assert evaluation["action_selection_changed_count"] == 5
    assert evaluation["causal_support_link_rate"] == pytest.approx(1.0)
    assert evaluation["read_retrieval_precision"] == pytest.approx(1.0)
    assert evaluation["contradiction_deflection_rate"] == pytest.approx(1.0)

    assert aligned_by_id["dec-timeout"]["selected_action"] == "pin-timeout-window"
    assert always_full_by_id["dec-timeout"]["selected_action"] == "reuse-old-timeout-window"
    assert aligned_by_id["dec-secret"]["selected_action"] == "reject-secret-rotation-transfer"
    assert always_full_by_id["dec-secret"]["selected_action"] == "transfer-secret-rotation"
    assert aligned_by_id["dec-pagination"]["read_memory_id"] is None
    assert aligned_by_id["dec-pagination"]["selected_action"] == "choose-zero-index-pagination"
    assert no_memory_by_id["dec-cache"]["selected_action"] == "baseline-cache-reset"
    assert aligned_by_id["dec-cache"]["selected_action"] == "resume-cache-replay"


def test_scenario_learn_5558_forbidden_direction_reuse_avoids_failed_choice() -> None:
    """SCENARIO-LEARN-5558-FORBIDDEN: failed directions become avoidance memory."""

    evaluation = exp.evaluate_fixture(exp.build_fixture())
    api_row = {
        row["decision_id"]: row
        for row in evaluation["arm_results"][exp.ALIGNED_CAUSAL_MEMORY_ARM]
    }["dec-api"]

    assert api_row["baseline_action"] == "retry-nonidempotent-call"
    assert api_row["forbidden_direction"] == "retry-nonidempotent-call"
    assert api_row["selected_action"] == "retry-idempotent-call"
    assert api_row["forbidden_direction_avoided"] is True
    assert evaluation["forbidden_direction_reuse_rate"] == pytest.approx(1.0)


def test_scenario_learn_5558_artifact_fields_and_stable_write(tmp_path: Path) -> None:
    """SCENARIO-LEARN-5558-ARTIFACT: receipt exposes causal gates."""

    destination = tmp_path / exp.RESULT_RELATIVE_PATH.name
    artifact = exp.run(
        root=REPO,
        result_path=destination,
        tests_added_or_reused=TESTS_ADDED_OR_REUSED,
        write=True,
    )
    no_write = exp.run(
        root=REPO,
        result_path=exp.RESULT_RELATIVE_PATH,
        tests_added_or_reused=TESTS_ADDED_OR_REUSED,
        write=False,
    )

    assert json.loads(destination.read_text(encoding="utf-8")) == artifact
    assert no_write["reproducibility_checksum"] == artifact["reproducibility_checksum"]
    assert exp.validate_artifact(artifact) is True
    assert exp._resolve_path(REPO, exp.RESULT_RELATIVE_PATH) == REPO / exp.RESULT_RELATIVE_PATH
    assert exp._resolve_path(REPO, destination) == destination

    for field in exp.REQUIRED_ARTIFACT_FIELDS:
        assert field in artifact
        assert artifact["field_principles"][field]

    assert artifact["continuous_self_learning_target"] is True
    assert artifact["upstream_five_arm_corrigendum"] == str(exp.UPSTREAM_FIVE_ARM_CORRIGENDUM)
    assert artifact["upstream_five_arm_status"]["csl_five_arm_clean"] is True
    assert artifact["llm_invoked"] is False
    assert artifact["no_model_specs_required"] is True
    assert artifact["write_filter_precision"] == pytest.approx(1.0)
    assert artifact["manage_forget_precision"] == pytest.approx(1.0)
    assert artifact["read_retrieval_precision"] == pytest.approx(1.0)
    assert artifact["causal_support_link_rate"] == pytest.approx(1.0)
    assert artifact["forbidden_direction_reuse_rate"] == pytest.approx(1.0)
    assert artifact["contradiction_deflection_rate"] == pytest.approx(1.0)
    assert artifact["action_impact_delta_vs_no_memory"] == pytest.approx(0.8333333333)
    assert artifact["quality_delta_vs_shuffled_memory"] == pytest.approx(1.0)
    assert artifact["quality_delta_vs_always_full"] == pytest.approx(0.3333333333)
    assert artifact["unsafe_false_accepts"] == 0
    assert artifact["no_weight_mutation"] is True
    assert artifact["csl_memory_ready"] is True
    assert artifact["csl_claim_allowed"] is True
    assert artifact["posthoc_only_claim_rejected"] is True
    assert artifact["inference_substrate"] == exp.INFERENCE_SUBSTRATE
    assert artifact["honest_verdict"].startswith("complete:")
    assert artifact["tests_added_or_reused"] == TESTS_ADDED_OR_REUSED


def test_scenario_learn_5558_claim_gate_rejects_score_only_or_unsafe_mutations() -> None:
    """REQ-LEARN-5558-6: claim gate rejects non-causal or unsafe artifacts."""

    artifact = _artifact()
    assert exp.validate_artifact(artifact) is True

    score_only = deepcopy(artifact)
    score_only["action_selection_changed_count"] = 0
    score_only["csl_claim_allowed"] = False
    score_only["csl_memory_ready"] = False
    score_only["posthoc_only_claim_rejected"] = True
    score_only["honest_verdict"] = exp.honest_verdict(score_only)
    score_only["reproducibility_checksum"] = exp.reproducibility_checksum(score_only)
    assert exp.validate_artifact(score_only) is True
    assert score_only["honest_verdict"].startswith("blocked:")

    invalid_claim = deepcopy(score_only)
    invalid_claim["csl_claim_allowed"] = True
    invalid_claim["csl_memory_ready"] = True
    invalid_claim["honest_verdict"] = "complete: invalid"
    invalid_claim["reproducibility_checksum"] = exp.reproducibility_checksum(invalid_claim)
    with pytest.raises(ValueError, match="csl_claim_allowed"):
        exp.validate_artifact(invalid_claim)

    bad_cases = [
        ("continuous_self_learning_target", False, "continuous_self_learning_target"),
        ("upstream_five_arm_corrigendum", "results/wrong.json", "upstream_five_arm_corrigendum"),
        ("llm_invoked", True, "llm_invoked"),
        ("no_model_specs_required", False, "no_model_specs_required"),
        ("write_filter_precision", 0.5, "write_filter_precision"),
        ("manage_forget_precision", 0.5, "manage_forget_precision"),
        ("read_retrieval_precision", 0.5, "read_retrieval_precision"),
        ("causal_support_link_rate", 0.5, "causal_support_link_rate"),
        ("forbidden_direction_reuse_rate", 0.5, "forbidden_direction_reuse_rate"),
        ("contradiction_deflection_rate", 0.5, "contradiction_deflection_rate"),
        ("action_impact_delta_vs_no_memory", 0.0, "action_impact_delta_vs_no_memory"),
        ("quality_delta_vs_shuffled_memory", 0.0, "quality_delta_vs_shuffled_memory"),
        ("quality_delta_vs_always_full", 0.0, "quality_delta_vs_always_full"),
        ("unsafe_false_accepts", 1, "unsafe_false_accepts"),
        ("no_weight_mutation", False, "no_weight_mutation"),
        ("spec_files_updated_or_confirmed", [], "spec_files_updated_or_confirmed"),
        ("tests_added_or_reused", [], "tests_added_or_reused"),
        ("field_principles", {}, "field_principles"),
        ("inference_substrate", "aggregation_from_upstream_artifacts", "inference_substrate"),
        ("honest_verdict", "ready", "honest_verdict"),
    ]
    for field, value, expected in bad_cases:
        bad = deepcopy(artifact)
        bad[field] = value
        bad["reproducibility_checksum"] = exp.reproducibility_checksum(bad)
        with pytest.raises(ValueError, match=expected):
            exp.validate_artifact(bad)

    missing = deepcopy(artifact)
    missing.pop("honest_verdict")
    missing["reproducibility_checksum"] = exp.reproducibility_checksum(missing)
    with pytest.raises(ValueError, match="missing required fields"):
        exp.validate_artifact(missing)

    bad_checksum = deepcopy(artifact)
    bad_checksum["reproducibility_checksum"] = "sha256:bad"
    with pytest.raises(ValueError, match="reproducibility_checksum"):
        exp.validate_artifact(bad_checksum)

    bad_principle_type = deepcopy(artifact)
    bad_principle_type["field_principles"] = "not-a-mapping"
    bad_principle_type["reproducibility_checksum"] = exp.reproducibility_checksum(
        bad_principle_type
    )
    with pytest.raises(ValueError, match="field_principles"):
        exp.validate_artifact(bad_principle_type)

    assert exp.upstream_five_arm_status(tmp_path := Path("/tmp/nonexistent-carnot-5558"))[
        "loadable"
    ] is False
    assert exp.score_rows([]) == 0.0
    assert exp.precision(0, 0) == 0.0
    assert exp.arm_scores_from_artifact(None) == {}


def test_req_learn_5558_repository_artifact_is_valid() -> None:
    """REQ-LEARN-5558-6: committed JSON remains a valid causal memory receipt."""

    artifact = json.loads(RESULT_PATH.read_text(encoding="utf-8"))

    assert exp.validate_artifact(artifact) is True
    assert set(exp.REQUIRED_ARTIFACT_FIELDS) <= set(artifact)
    assert artifact["inference_substrate"] == exp.INFERENCE_SUBSTRATE
    assert artifact["csl_memory_ready"] is True
    assert artifact["csl_claim_allowed"] is True
