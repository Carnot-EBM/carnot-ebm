"""Tests for Exp5638 FR-11 gate schema corrigendum.

Spec refs: REQ-LEARN-5638,
SCENARIO-LEARN-5638-ORIGINAL-SHAPE,
SCENARIO-LEARN-5638-FAIL-CLOSED,
SCENARIO-LEARN-5638-DETERMINISTIC.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from carnot import experiment_5638_fr11_gate_schema_corrigendum as mod


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec/capabilities/self-learning/spec.md"
SOURCE_PATH = REPO / mod.SOURCE_ARTIFACT_RELATIVE_PATH
TEST_COMMAND = (
    ".venv/bin/pytest "
    "tests/python/test_experiment_5638_fr11_gate_schema_corrigendum.py "
    "-q --no-cov -n 0"
)
COVERAGE_COMMAND = (
    ".venv/bin/coverage run "
    "--include=python/carnot/experiment_5638_fr11_gate_schema_corrigendum.py "
    "-m pytest tests/python/test_experiment_5638_fr11_gate_schema_corrigendum.py "
    "-q --no-cov -n 0 && .venv/bin/coverage report "
    "--include=python/carnot/experiment_5638_fr11_gate_schema_corrigendum.py "
    "--fail-under=100"
)
FULL_TEST_COMMAND = ".venv/bin/pytest tests/python -q"
SPEC_COMMAND = ".venv/bin/python scripts/check_spec_coverage.py"
ADVERSARIAL_COMMAND = (
    ".venv/bin/python scripts/adversarial_verify.py "
    "results/experiment_5638_fr11_gate_schema_corrigendum.json"
)
TESTS_ADDED_OR_REUSED = [
    TEST_COMMAND,
    COVERAGE_COMMAND,
    FULL_TEST_COMMAND,
    SPEC_COMMAND,
    ADVERSARIAL_COMMAND,
]


def _write_source(path: Path, unsafe_false_accept_count: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "continuous_self_learning_ready": True,
        "honest_verdict": "complete: source_fixture",
        "unsafe_false_accept_count": unsafe_false_accept_count,
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def test_req_learn_5638_spec_declares_hash_bound_scalar_contract() -> None:
    """REQ-LEARN-5638: OpenSpec anchors immutable scalar normalization."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    section = spec[spec.index("## REQ-LEARN-5638") :]

    for marker in (
        "REQ-LEARN-5638",
        "SCENARIO-LEARN-5638-ORIGINAL-SHAPE",
        "SCENARIO-LEARN-5638-FAIL-CLOSED",
        "SCENARIO-LEARN-5638-DETERMINISTIC",
        str(mod.RESULT_RELATIVE_PATH),
        str(mod.SOURCE_ARTIFACT_RELATIVE_PATH),
        mod.NORMALIZATION_JSON_PATH,
        mod.INFERENCE_SUBSTRATE,
        "scripts/conductor_gates.py",
        "scripts/research_conductor.py",
    ):
        assert marker in section
    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert f"`{field}`" in section
        assert mod.FIELD_PRINCIPLES[field]


def test_scenario_learn_5638_original_exp5628_shape_normalizes_to_scalar_zero() -> None:
    """SCENARIO-LEARN-5638-ORIGINAL-SHAPE: structured Exp5628 evidence is preserved."""

    source_bytes_before = SOURCE_PATH.read_bytes()
    source = json.loads(source_bytes_before.decode("utf-8"))

    artifact = mod.build_artifact(root=REPO, tests_added_or_reused=TESTS_ADDED_OR_REUSED)

    assert SOURCE_PATH.read_bytes() == source_bytes_before
    assert artifact["source_artifact_path"] == mod.SOURCE_ARTIFACT_RELATIVE_PATH.as_posix()
    assert artifact["source_artifact_sha256"] == mod.EXPECTED_SOURCE_ARTIFACT_SHA256
    assert artifact["source_honest_verdict"] == source["honest_verdict"]
    assert artifact["raw_unsafe_false_accept_count"] == source["unsafe_false_accept_count"]
    assert artifact["normalization_json_path"] == mod.NORMALIZATION_JSON_PATH
    assert artifact["unsafe_false_accept_count_total"] == 0
    assert type(artifact["unsafe_false_accept_count_total"]) is int
    assert artifact["by_arm_sum"] == 0
    assert artifact["by_arm_reconciliation_pass"] is True
    assert artifact["source_continuous_self_learning_ready"] is True
    assert artifact["scientific_recompute_performed"] is False
    assert artifact["source_artifact_modified"] is False
    assert artifact["gate_contract_ready_score"] == 1.0
    assert artifact["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert artifact["honest_verdict"].startswith("complete:")
    assert "independent FR-11 validation" not in artifact["honest_verdict"]
    assert artifact["reproducibility_checksum"] == mod.reproducibility_checksum(artifact)
    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert field in artifact
        assert artifact["field_principles"][field] == mod.FIELD_PRINCIPLES[field]
    assert mod.validate_artifact(artifact) is True


def test_scenario_learn_5638_json_source_must_be_unambiguous_object() -> None:
    """SCENARIO-LEARN-5638-FAIL-CLOSED: invalid JSON cannot be normalized."""

    with pytest.raises(ValueError, match="invalid JSON source artifact"):
        mod.load_json_object_from_bytes(b"{")
    with pytest.raises(ValueError, match="source artifact must be a JSON object"):
        mod.load_json_object_from_bytes(b"[]")


@pytest.mark.parametrize(
    ("unsafe_false_accept_count", "match"),
    [
        ([], "unsafe_false_accept_count must be an object"),
        ({"by_arm": {"a": 0}}, "unsafe_false_accept_count.total"),
        ({"by_arm": {"a": 0}, "total": False}, "non-boolean integer"),
        ({"by_arm": {"a": 0}, "total": 0.0}, "non-boolean integer"),
        ({"by_arm": {"a": 0}, "total": -1}, "non-negative"),
        ({"by_arm": {"a": 0}, "total": "0"}, "non-boolean integer"),
        ({"total": 0}, "unsafe_false_accept_count.by_arm"),
        ({"by_arm": [], "total": 0}, "unsafe_false_accept_count.by_arm"),
        ({"by_arm": {"a": True}, "total": 0}, "non-boolean integer"),
        ({"by_arm": {"a": 0.5}, "total": 0}, "non-boolean integer"),
        ({"by_arm": {"a": -1}, "total": 0}, "non-negative"),
        ({"by_arm": {"a": "0"}, "total": 0}, "non-boolean integer"),
        ({"by_arm": {"a": 1}, "total": 0}, "by_arm_sum"),
    ],
)
def test_scenario_learn_5638_malformed_shapes_are_rejected(
    tmp_path: Path,
    unsafe_false_accept_count: object,
    match: str,
) -> None:
    """SCENARIO-LEARN-5638-FAIL-CLOSED: malformed totals never emit a contract."""

    source_path = tmp_path / "source.json"
    _write_source(source_path, unsafe_false_accept_count)

    with pytest.raises(ValueError, match=match):
        mod.build_artifact(
            source_path=source_path,
            tests_added_or_reused=TESTS_ADDED_OR_REUSED,
        )


def test_scenario_learn_5638_duplicate_total_is_ambiguous_and_rejected(tmp_path: Path) -> None:
    """SCENARIO-LEARN-5638-FAIL-CLOSED: duplicate JSON keys are ambiguous totals."""

    source_path = tmp_path / "source.json"
    source_path.write_text(
        "{"
        '"continuous_self_learning_ready": true,'
        '"honest_verdict": "complete: source_fixture",'
        '"unsafe_false_accept_count": {"by_arm": {"a": 0}, "total": 1, "total": 0}'
        "}\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="duplicate JSON key: total"):
        mod.build_artifact(
            source_path=source_path,
            tests_added_or_reused=TESTS_ADDED_OR_REUSED,
        )


def test_scenario_learn_5638_nonzero_consistent_values_block_without_rewriting(
    tmp_path: Path,
) -> None:
    """REQ-LEARN-5638: nonzero but reconciled evidence is preserved and blocked."""

    source_path = tmp_path / "source.json"
    _write_source(source_path, {"by_arm": {"arm_a": 1, "arm_b": 2}, "total": 3})
    source_bytes_before = source_path.read_bytes()

    artifact = mod.build_artifact(
        source_path=source_path,
        tests_added_or_reused=TESTS_ADDED_OR_REUSED,
    )

    assert source_path.read_bytes() == source_bytes_before
    assert artifact["raw_unsafe_false_accept_count"] == {
        "by_arm": {"arm_a": 1, "arm_b": 2},
        "total": 3,
    }
    assert artifact["unsafe_false_accept_count_total"] == 3
    assert artifact["by_arm_sum"] == 3
    assert artifact["by_arm_reconciliation_pass"] is True
    assert artifact["gate_contract_ready_score"] == 0.0
    assert artifact["honest_verdict"].startswith("blocked:")
    assert mod.validate_artifact(artifact) is True


def test_scenario_learn_5638_source_hash_drift_blocks_contract(tmp_path: Path) -> None:
    """SCENARIO-LEARN-5638-FAIL-CLOSED: hash drift cannot receive readiness credit."""

    source_path = tmp_path / "source.json"
    _write_source(source_path, {"by_arm": {"arm_a": 0}, "total": 0})

    artifact = mod.build_artifact(
        source_path=source_path,
        tests_added_or_reused=TESTS_ADDED_OR_REUSED,
    )

    assert artifact["source_artifact_sha256"] == mod.sha256_file(source_path)
    assert artifact["source_hash_exact"] is False
    assert artifact["unsafe_false_accept_count_total"] == 0
    assert artifact["gate_contract_ready_score"] == 0.0
    assert artifact["honest_verdict"].startswith("blocked:")
    assert artifact["scientific_recompute_performed"] is False
    assert artifact["source_artifact_modified"] is False
    assert mod.validate_artifact(artifact) is True

    bad_verdict = deepcopy(artifact)
    bad_verdict["honest_verdict"] = "complete: wrong_for_blocked_contract"
    bad_verdict["reproducibility_checksum"] = mod.reproducibility_checksum(bad_verdict)
    with pytest.raises(ValueError, match="honest_verdict"):
        mod.validate_artifact(bad_verdict)


def test_scenario_learn_5638_run_is_deterministic_and_validation_rejects_drift(
    tmp_path: Path,
) -> None:
    """SCENARIO-LEARN-5638-DETERMINISTIC: repeated writes are byte stable."""

    result_path = tmp_path / mod.RESULT_RELATIVE_PATH.name

    first = mod.run(
        root=REPO,
        result_path=result_path,
        tests_added_or_reused=TESTS_ADDED_OR_REUSED,
        write=True,
    )
    first_bytes = result_path.read_bytes()
    second = mod.run(
        root=REPO,
        result_path=result_path,
        tests_added_or_reused=TESTS_ADDED_OR_REUSED,
        write=True,
    )
    second_bytes = result_path.read_bytes()

    assert first == second
    assert first_bytes == second_bytes
    assert json.loads(second_bytes.decode("utf-8")) == second
    assert second["reproducibility_checksum"] == first["reproducibility_checksum"]

    drift_cases = [
        ("source_artifact_sha256", "sha256:bad", "source_artifact_sha256"),
        ("unsafe_false_accept_count_total", True, "unsafe_false_accept_count_total"),
        ("by_arm_sum", 1, "by_arm_sum"),
        ("by_arm_reconciliation_pass", False, "by_arm_reconciliation_pass"),
        ("scientific_recompute_performed", True, "scientific_recompute_performed"),
        ("source_artifact_modified", True, "source_artifact_modified"),
        ("gate_contract_ready_score", 0.5, "gate_contract_ready_score"),
        ("inference_substrate", "llm_inference", "inference_substrate"),
        ("honest_verdict", "ready", "honest_verdict"),
    ]
    for field, value, match in drift_cases:
        bad = deepcopy(second)
        bad[field] = value
        bad["reproducibility_checksum"] = mod.reproducibility_checksum(bad)
        with pytest.raises(ValueError, match=match):
            mod.validate_artifact(bad)

    missing = deepcopy(second)
    missing.pop("source_artifact_path")
    missing["reproducibility_checksum"] = mod.reproducibility_checksum(missing)
    with pytest.raises(ValueError, match="missing required fields"):
        mod.validate_artifact(missing)

    missing_principle = deepcopy(second)
    missing_principle["field_principles"].pop("source_artifact_path")
    missing_principle["reproducibility_checksum"] = mod.reproducibility_checksum(missing_principle)
    with pytest.raises(ValueError, match="field_principles"):
        mod.validate_artifact(missing_principle)

    nonmapping_principle = deepcopy(second)
    nonmapping_principle["field_principles"] = []
    nonmapping_principle["reproducibility_checksum"] = mod.reproducibility_checksum(
        nonmapping_principle
    )
    with pytest.raises(ValueError, match="field_principles"):
        mod.validate_artifact(nonmapping_principle)

    bad_raw = deepcopy(second)
    bad_raw["raw_unsafe_false_accept_count"] = []
    bad_raw["reproducibility_checksum"] = mod.reproducibility_checksum(bad_raw)
    with pytest.raises(ValueError, match="raw_unsafe_false_accept_count"):
        mod.validate_artifact(bad_raw)

    no_tests = deepcopy(second)
    no_tests["tests_added_or_reused"] = []
    no_tests["reproducibility_checksum"] = mod.reproducibility_checksum(no_tests)
    with pytest.raises(ValueError, match="tests_added_or_reused"):
        mod.validate_artifact(no_tests)

    bad_checksum = deepcopy(second)
    bad_checksum["reproducibility_checksum"] = "sha256:bad"
    with pytest.raises(ValueError, match="reproducibility_checksum"):
        mod.validate_artifact(bad_checksum)
