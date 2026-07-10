"""Tests for Exp5557 CSL five-arm tautology corrigendum v2.

Spec refs: REQ-LEARN-5557,
SCENARIO-LEARN-5557-BASELINES,
SCENARIO-LEARN-5557-CONTROLS,
SCENARIO-LEARN-5557-ARTIFACT.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from carnot import experiment_5557_csl_five_arm_tautology_corrigendum_v2 as mod


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec/capabilities/self-learning/spec.md"
RESULT_PATH = REPO / mod.RESULT_RELATIVE_PATH
TEST_COMMAND = (
    ".venv/bin/pytest "
    "tests/python/test_experiment_5557_csl_five_arm_tautology_corrigendum_v2.py "
    "-q --no-cov -n 0"
)
COVERAGE_COMMAND = (
    ".venv/bin/coverage run "
    "--include=python/carnot/experiment_5557_csl_five_arm_tautology_corrigendum_v2.py "
    "-m pytest tests/python/test_experiment_5557_csl_five_arm_tautology_corrigendum_v2.py "
    "-q --no-cov -n 0 && .venv/bin/coverage report "
    "--include=python/carnot/experiment_5557_csl_five_arm_tautology_corrigendum_v2.py "
    "--fail-under=100"
)
FULL_TEST_COMMAND = ".venv/bin/pytest tests/python -q"
TESTS_ADDED_OR_REUSED = [TEST_COMMAND, COVERAGE_COMMAND, FULL_TEST_COMMAND]


def _artifact() -> dict[str, object]:
    return mod.build_artifact(root=REPO, tests_added_or_reused=TESTS_ADDED_OR_REUSED)


def test_req_learn_5557_spec_declares_corrigendum_contract() -> None:
    """REQ-LEARN-5557: OpenSpec anchors the five-arm corrigendum."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    section = spec[spec.index("## REQ-LEARN-5557") :]
    normalized = " ".join(section.split())

    for marker in (
        "REQ-LEARN-5557",
        "SCENARIO-LEARN-5557-BASELINES",
        "SCENARIO-LEARN-5557-CONTROLS",
        "SCENARIO-LEARN-5557-ARTIFACT",
        str(mod.RESULT_RELATIVE_PATH),
        str(mod.UPSTREAM_CSL_RESIDUE_CORRIGENDUM),
        str(mod.UPSTREAM_FLAGGED_ABLATION),
        mod.INFERENCE_SUBSTRATE,
    ):
        assert marker in section
    for marker in (
        "best-constant and per-query-random are equal within tolerance",
        "aligned-memory does not beat shuffled-memory",
    ):
        assert marker in normalized
    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert f"`{field}`" in section
        assert mod.FIELD_PRINCIPLES[field]


def test_scenario_learn_5557_baselines_separate_on_nondegenerate_fixture() -> None:
    """SCENARIO-LEARN-5557-BASELINES: random control cannot equal constant."""

    fixture = mod.build_fixture()
    evaluation = mod.evaluate_controls(fixture)
    scores = evaluation["scores"]
    arm_results = evaluation["arm_results"]

    assert mod.fixture_is_non_degenerate(fixture) is True
    assert evaluation["same_heldout_query_set"] is True
    assert scores["best_constant_score"] == pytest.approx(0.0833333333)
    assert scores["per_query_random_score"] == pytest.approx(0.0)
    assert scores["no_memory_score"] == pytest.approx(0.1666666667)
    assert scores["shuffled_memory_score"] == pytest.approx(0.25)
    assert scores["aligned_memory_score"] == pytest.approx(0.8333333333)
    assert evaluation["aligned_delta_over_shuffled"] == pytest.approx(0.5833333333)
    assert evaluation["duplicated_metric_pairs"] == []
    assert evaluation["tautology_resolved"] is True

    constant_actions = {
        row["selected_action"] for row in arm_results[mod.BEST_CONSTANT_ARM]
    }
    random_actions = {
        row["selected_action"] for row in arm_results[mod.PER_QUERY_RANDOM_ARM]
    }
    assert len(constant_actions) == 1
    assert len(random_actions) > 1
    assert random_actions != constant_actions

    collapsed_scores = dict(scores)
    collapsed_scores["per_query_random_score"] = collapsed_scores["best_constant_score"]
    pairs = mod.duplicated_metric_pairs(collapsed_scores, mod.EQUALITY_TOLERANCE)
    assert ("best_constant_score", "per_query_random_score") in {
        (pair["left"], pair["right"]) for pair in pairs
    }
    assert mod.tautology_resolved(pairs) is False


def test_scenario_learn_5557_artifact_fields_and_stable_write(tmp_path: Path) -> None:
    """SCENARIO-LEARN-5557-ARTIFACT: receipt records the clean fix."""

    destination = tmp_path / mod.RESULT_RELATIVE_PATH.name
    artifact = mod.run(
        root=REPO,
        result_path=destination,
        tests_added_or_reused=TESTS_ADDED_OR_REUSED,
        write=True,
    )
    no_write = mod.run(
        root=REPO,
        result_path=mod.RESULT_RELATIVE_PATH,
        tests_added_or_reused=TESTS_ADDED_OR_REUSED,
        write=False,
    )

    assert json.loads(destination.read_text(encoding="utf-8")) == artifact
    assert no_write["reproducibility_checksum"] == artifact["reproducibility_checksum"]
    assert mod.validate_artifact(artifact) is True
    assert mod._resolve_path(REPO, mod.RESULT_RELATIVE_PATH) == REPO / mod.RESULT_RELATIVE_PATH
    assert mod._resolve_path(REPO, destination) == destination

    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert field in artifact
        assert artifact["field_principles"][field]

    assert artifact["upstream_csl_residue_corrigendum"] == str(
        mod.UPSTREAM_CSL_RESIDUE_CORRIGENDUM
    )
    assert artifact["upstream_flagged_ablation"] == str(mod.UPSTREAM_FLAGGED_ABLATION)
    assert artifact["upstream_residue_status"]["csl_residue_tautology_resolved"] is True
    assert artifact["upstream_flagged_ablation_status"]["flagged_adversarial"] is True
    assert artifact["upstream_flagged_ablation_status"]["tautology_pair_observed"] is True
    assert artifact["llm_invoked"] is False
    assert artifact["no_model_specs_required"] is True
    assert artifact["duplicated_metric_pairs"] == []
    assert artifact["tautology_resolved"] is True
    assert artifact["csl_five_arm_clean"] is True
    assert artifact["adversarial_clean"] is True
    assert artifact["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert artifact["honest_verdict"].startswith("complete:")
    assert artifact["tests_added_or_reused"] == TESTS_ADDED_OR_REUSED


def test_scenario_learn_5557_gate_rejects_duplicate_or_flat_controls(
    tmp_path: Path,
) -> None:
    """SCENARIO-LEARN-5557-CONTROLS: clean gate fails closed on regressions."""

    artifact = _artifact()

    duplicated_random = deepcopy(artifact)
    duplicated_random["arm_results"][mod.PER_QUERY_RANDOM_ARM] = deepcopy(
        duplicated_random["arm_results"][mod.BEST_CONSTANT_ARM]
    )
    duplicated_random["per_query_random_score"] = duplicated_random[
        "best_constant_score"
    ]
    duplicated_random["duplicated_metric_pairs"] = []
    duplicated_random["reproducibility_checksum"] = mod.reproducibility_checksum(
        duplicated_random
    )
    with pytest.raises(ValueError, match="duplicated_metric_pairs"):
        mod.validate_artifact(duplicated_random)

    blocked = deepcopy(artifact)
    blocked["arm_results"][mod.ALIGNED_MEMORY_ARM] = deepcopy(
        blocked["arm_results"][mod.SHUFFLED_MEMORY_ARM]
    )
    for row in blocked["arm_results"][mod.ALIGNED_MEMORY_ARM]:
        row["arm"] = mod.ALIGNED_MEMORY_ARM
    blocked["aligned_memory_score"] = blocked["shuffled_memory_score"]
    blocked["aligned_delta_over_shuffled"] = 0.0
    blocked["duplicated_metric_pairs"] = mod.duplicated_metric_pairs(
        mod.headline_scores_from_artifact(blocked),
        mod.EQUALITY_TOLERANCE,
    )
    blocked["tautology_resolved"] = False
    blocked["csl_five_arm_clean"] = False
    blocked["adversarial_clean"] = False
    blocked["honest_verdict"] = mod.honest_verdict(blocked)
    blocked["reproducibility_checksum"] = mod.reproducibility_checksum(blocked)

    assert mod.validate_artifact(blocked) is True
    assert blocked["honest_verdict"].startswith("blocked:")

    invalid_clean = deepcopy(blocked)
    invalid_clean["csl_five_arm_clean"] = True
    invalid_clean["adversarial_clean"] = True
    invalid_clean["honest_verdict"] = "complete: invalid"
    invalid_clean["reproducibility_checksum"] = mod.reproducibility_checksum(
        invalid_clean
    )
    with pytest.raises(ValueError, match="csl_five_arm_clean"):
        mod.validate_artifact(invalid_clean)

    assert mod.upstream_residue_status(tmp_path)["loadable"] is False
    assert mod.upstream_flagged_ablation_status(tmp_path)["loadable"] is False
    assert mod.arm_scores_from_artifact(None) == {}

    mutations: list[tuple[dict[str, object], str, bool]] = []
    missing = deepcopy(artifact)
    missing.pop("honest_verdict")
    mutations.append((missing, "missing required fields", True))

    bad_residue = deepcopy(artifact)
    bad_residue["upstream_csl_residue_corrigendum"] = "results/wrong.json"
    mutations.append((bad_residue, "upstream_csl_residue_corrigendum", True))

    bad_flagged = deepcopy(artifact)
    bad_flagged["upstream_flagged_ablation"] = "results/wrong.json"
    mutations.append((bad_flagged, "upstream_flagged_ablation", True))

    llm_called = deepcopy(artifact)
    llm_called["llm_invoked"] = True
    mutations.append((llm_called, "llm_invoked", True))

    bad_model_specs = deepcopy(artifact)
    bad_model_specs["no_model_specs_required"] = False
    mutations.append((bad_model_specs, "no_model_specs_required", True))

    no_tests = deepcopy(artifact)
    no_tests["tests_added_or_reused"] = []
    mutations.append((no_tests, "tests_added_or_reused", True))

    bad_same_queries = deepcopy(artifact)
    bad_same_queries["same_heldout_query_set"] = False
    mutations.append((bad_same_queries, "same_heldout_query_set", True))

    bad_score = deepcopy(artifact)
    bad_score["best_constant_score"] = 0.0
    mutations.append((bad_score, "best_constant_score", True))

    bad_delta = deepcopy(artifact)
    bad_delta["aligned_delta_over_shuffled"] = 0.0
    mutations.append((bad_delta, "aligned_delta_over_shuffled", True))

    bad_tolerance = deepcopy(artifact)
    bad_tolerance["equality_tolerance"] = 0.5
    mutations.append((bad_tolerance, "equality_tolerance", True))

    bad_hashes = deepcopy(artifact)
    bad_hashes["query_hashes"] = []
    mutations.append((bad_hashes, "query_hashes", True))

    bad_principles = deepcopy(artifact)
    bad_principles["field_principles"] = {}
    mutations.append((bad_principles, "field_principles", True))

    bad_substrate = deepcopy(artifact)
    bad_substrate["inference_substrate"] = "aggregation_from_upstream_artifacts"
    mutations.append((bad_substrate, "inference_substrate", True))

    bad_checksum = deepcopy(artifact)
    bad_checksum["reproducibility_checksum"] = "sha256:bad"
    mutations.append((bad_checksum, "reproducibility_checksum", False))

    for payload, expected, refresh_checksum in mutations:
        if refresh_checksum:
            payload["reproducibility_checksum"] = mod.reproducibility_checksum(payload)
        with pytest.raises(ValueError, match=expected):
            mod.validate_artifact(payload)


def test_req_learn_5557_repository_artifact_is_valid() -> None:
    """REQ-LEARN-5557-6: committed JSON remains a valid clean receipt."""

    artifact = json.loads(RESULT_PATH.read_text(encoding="utf-8"))

    assert mod.validate_artifact(artifact) is True
    assert set(mod.REQUIRED_ARTIFACT_FIELDS) <= set(artifact)
    assert artifact["inference_substrate"] == "deterministic_csl_ablation_no_llm"
    assert artifact["csl_five_arm_clean"] is True
    assert artifact["adversarial_clean"] is True
