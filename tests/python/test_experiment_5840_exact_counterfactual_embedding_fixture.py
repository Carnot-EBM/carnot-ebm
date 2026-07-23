"""Tests for Exp5840 exact counterfactual embedding fixture.

Spec refs: REQ-VERIFY-5840, SCENARIO-VERIFY-5840-PAIRS,
SCENARIO-VERIFY-5840-LEAKAGE, SCENARIO-VERIFY-5840-FAIL-CLOSED.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from copy import deepcopy
import json
from pathlib import Path
from typing import Any

import pytest

from carnot import experiment_5840_exact_counterfactual_embedding_fixture as mod


REPO = Path(__file__).resolve().parents[2]
VERIFY_SPEC = REPO / "openspec/capabilities/verification/spec.md"
TEST_COMMAND = (
    ".venv/bin/pytest "
    "tests/python/test_experiment_5840_exact_counterfactual_embedding_fixture.py "
    "-q --no-cov -n 0"
)
COVERAGE_COMMAND = (
    ".venv/bin/coverage run --rcfile=/dev/null "
    "--include=python/carnot/experiment_5840_exact_counterfactual_embedding_fixture.py "
    "-m pytest tests/python/test_experiment_5840_exact_counterfactual_embedding_fixture.py "
    "-q --no-cov -n 0 && .venv/bin/coverage report --rcfile=/dev/null "
    "--include=python/carnot/experiment_5840_exact_counterfactual_embedding_fixture.py "
    "--fail-under=100"
)
FULL_TEST_COMMAND = ".venv/bin/pytest tests/python -q"
SPEC_COMMAND = ".venv/bin/python scripts/check_spec_coverage.py"
ADVERSARIAL_COMMAND = (
    ".venv/bin/python scripts/adversarial_verify.py "
    "results/experiment_5840_exact_counterfactual_embedding_fixture.json"
)
ROOT_CLUTTER_COMMAND = ".venv/bin/python scripts/root_clutter_sweep.py"
PROTECTED_FILE_COMMAND = (
    '.venv/bin/python -c "from pathlib import Path; '
    "assert Path('scripts/research_conductor.py').exists()\""
)
TEST_COMMANDS = [
    TEST_COMMAND,
    COVERAGE_COMMAND,
    FULL_TEST_COMMAND,
    SPEC_COMMAND,
    ADVERSARIAL_COMMAND,
    ROOT_CLUTTER_COMMAND,
    PROTECTED_FILE_COMMAND,
]
TEST_EXIT_CODES = {command: 0 for command in TEST_COMMANDS}


def _preconditions(tmp_path: Path) -> dict[str, Any]:
    return mod.collect_preconditions(
        result_path=tmp_path / mod.RESULT_RELATIVE_PATH.name,
        row_file_path=tmp_path / mod.ROW_FILE_RELATIVE_PATH.name,
        memory_probe=lambda: {"available_mb": 32768, "required_mb": 1024, "ok": True},
        disk_probe=lambda root: {"available_mb": 32768, "required_mb": 1024, "ok": True},
    )


@pytest.fixture(scope="module")
def exp5840_fixture(tmp_path_factory: pytest.TempPathFactory) -> tuple[dict[str, Any], list[dict[str, Any]], Path]:
    """REQ-VERIFY-5840: build the deterministic fixture once for the test module."""

    base = tmp_path_factory.mktemp("exp5840")
    conductor = REPO / "scripts/research_conductor.py"
    before_hash = mod.sha256_file(conductor)
    artifact = mod.run(
        result_path=base / mod.RESULT_RELATIVE_PATH.name,
        row_file_path=base / mod.ROW_FILE_RELATIVE_PATH.name,
        preconditions_checked=_preconditions(base),
        test_commands=TEST_COMMANDS,
        test_exit_codes=TEST_EXIT_CODES,
        write=True,
    )
    rows = mod.read_row_file(base / mod.ROW_FILE_RELATIVE_PATH.name)
    assert mod.sha256_file(conductor) == before_hash
    return artifact, rows, base


def _condition_inputs(rows: list[dict[str, Any]]) -> list[str]:
    return [
        str(condition["model_input"])
        for row in rows
        for condition in row["conditions"]
    ]


def _condition_ids(rows: list[dict[str, Any]]) -> list[str]:
    return [
        str(condition["condition_id"])
        for row in rows
        for condition in row["conditions"]
    ]


def test_req_verify_5840_spec_declares_fixture_contract() -> None:
    """REQ-VERIFY-5840: OpenSpec names fields, principles, and scenarios."""

    text = VERIFY_SPEC.read_text(encoding="utf-8")
    section = text[text.index("### REQ-VERIFY-5840") :]
    normalized = " ".join(section.split())

    for marker in (
        "REQ-VERIFY-5840",
        "SCENARIO-VERIFY-5840-PAIRS",
        "SCENARIO-VERIFY-5840-LEAKAGE",
        "SCENARIO-VERIFY-5840-FAIL-CLOSED",
        mod.RESULT_RELATIVE_PATH.as_posix(),
        mod.ROW_FILE_RELATIVE_PATH.as_posix(),
        mod.INFERENCE_SUBSTRATE,
        "`counterfactual_fixture_ready_score`",
    ):
        assert marker in section
    for field, principle in mod.REQUIRED_FIELD_PRINCIPLES.items():
        assert f"`{field}`" in section
        assert " ".join(principle.split()) in normalized


def test_req_verify_5840_terminal_artifact_is_hash_bound_and_replayable(
    exp5840_fixture: tuple[dict[str, Any], list[dict[str, Any]], Path],
) -> None:
    """REQ-VERIFY-5840: terminal JSON/JSONL commitments replay exactly."""

    artifact, rows, base = exp5840_fixture
    rerun = mod.run(
        result_path=base / "rerun.json",
        row_file_path=base / "rerun.rows.jsonl",
        preconditions_checked=_preconditions(base / "rerun"),
        test_commands=TEST_COMMANDS,
        test_exit_codes=TEST_EXIT_CODES,
        write=True,
    )

    assert mod.validate_artifact(artifact) is True
    assert mod.verify_row_file(rows, artifact) is True
    assert json.loads((base / mod.RESULT_RELATIVE_PATH.name).read_text()) == artifact
    assert artifact["status"] == "complete"
    assert artifact["honest_verdict"].startswith("ready:")
    assert artifact["counterfactual_fixture_ready_score"] == pytest.approx(1.0)
    assert isinstance(artifact["counterfactual_fixture_ready_score"], float)
    assert artifact["duration_s"] >= 0.0
    assert artifact["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert artifact["verifier_is_oracle"] is True
    assert artifact["preconditions_checked"]["preconditions_ready"] is True
    assert artifact["preconditions_checked"]["structured_gate_replay"]["ok"] is True
    assert artifact["row_file_receipt"]["sha256"] == mod.sha256_file(
        base / mod.ROW_FILE_RELATIVE_PATH.name
    )
    assert artifact["row_file_receipt"]["sha256"] == rerun["row_file_receipt"]["sha256"]
    assert artifact["reproducibility_checksum"] == rerun["reproducibility_checksum"]
    assert set(mod.REQUIRED_ARTIFACT_FIELDS) <= set(artifact)
    assert set(artifact["test_exit_codes"]) == set(artifact["test_commands"])
    assert all(code == 0 for code in artifact["test_exit_codes"].values())

    for field, principle in mod.REQUIRED_FIELD_PRINCIPLES.items():
        assert artifact["field_provenance"][field]["principle"] == principle


def test_scenario_verify_5840_pairs_are_exact_causal_and_balanced(
    exp5840_fixture: tuple[dict[str, Any], list[dict[str, Any]], Path],
) -> None:
    """SCENARIO-VERIFY-5840-PAIRS: every family/axis cell has exact causal pairs."""

    artifact, rows, _base = exp5840_fixture
    counts = artifact["family_axis_cell_counts"]

    assert len(rows) == artifact["row_file_receipt"]["row_count"]
    assert artifact["exact_label_and_minimality_receipts"]["all_exact_labels_passed"] is True
    assert artifact["exact_label_and_minimality_receipts"]["all_minimal_violations_passed"] is True
    assert artifact["constraint_ablation_receipts"]["all_ablation_checks_passed"] is True
    assert artifact["proof_preserving_surface_receipts"]["all_surface_checks_passed"] is True

    for family in mod.PRIMARY_FAMILIES:
        for axis in mod.CAUSAL_AXES:
            key = f"{family}|{axis}"
            assert counts["family_axis_counts"][key] >= mod.MIN_PAIRS_PER_FAMILY_AXIS
            assert counts["family_axis_minimums"][key] >= mod.MIN_PAIRS_PER_FAMILY_AXIS
            for change in mod.CHANGE_ORDER:
                assert counts["family_axis_change_counts"][f"{family}|{axis}|{change}"] > 0
            for hardness in mod.HARDNESS_BINS:
                assert counts["family_axis_hardness_counts"][f"{family}|{axis}|{hardness}"] > 0
            for surface in mod.PROOF_PRESERVING_SURFACES:
                assert counts["family_axis_surface_counts"][f"{family}|{axis}|{surface}"] > 0

    axis_counts = Counter(row["axis"] for row in rows)
    assert set(axis_counts) == set(mod.CAUSAL_AXES)
    assert min(axis_counts.values()) >= len(mod.PRIMARY_FAMILIES) * mod.MIN_PAIRS_PER_FAMILY_AXIS

    for row in rows:
        left, right = row["conditions"]
        assert row["row_hash"] == mod.row_hash(row)
        assert row["bootstrap_unit_id"].startswith("sha256:")
        assert row["pair_group_id"].startswith("exp5840-group-")
        assert left["token_count"] == right["token_count"] == mod.TOKEN_BUDGET
        assert left["model_input_hash"] != right["model_input_hash"]
        assert left["model_input"] != right["model_input"]
        if row["axis"] == "candidate_correctness":
            assert left["context_hash"] == right["context_hash"]
            assert left["candidate_hash"] != right["candidate_hash"]
            assert left["exact_label"] is True
            assert right["exact_label"] is False
        else:
            assert row["axis"] == "constraint_ablation"
            assert left["candidate_hash"] == right["candidate_hash"]
            assert left["context_hash"] != right["context_hash"]
            assert left["exact_label"] is False
            assert right["exact_label"] is True
        receipt = row["exact_receipt"]
        assert receipt["validators_agree"] is True
        assert receipt["minimal_edit_distance"] == 1
        assert receipt["minimal_violation_proof"]["one_minimal_violation"] is True
        assert row["ablation_receipt"]["candidate_fixed"] is (row["axis"] == "constraint_ablation")


def test_scenario_verify_5840_feature_inputs_are_masked_token_matched_and_split_safe(
    exp5840_fixture: tuple[dict[str, Any], list[dict[str, Any]], Path],
) -> None:
    """SCENARIO-VERIFY-5840-LEAKAGE: model inputs are masked and split-disjoint."""

    artifact, rows, _base = exp5840_fixture
    inputs = _condition_inputs(rows)
    condition_ids = _condition_ids(rows)
    joined_inputs = "\n".join(inputs).lower()

    assert len(condition_ids) == len(set(condition_ids))
    assert artifact["target_leakage_checks"]["all_checks_passed"] is True
    assert artifact["target_leakage_checks"]["identity_leakage_count"] == 0
    assert artifact["target_leakage_checks"]["answer_leakage_count"] == 0
    assert artifact["target_leakage_checks"]["duplicate_model_input_count"] == 0
    assert artifact["target_leakage_checks"]["near_duplicate_pair_count"] == 0
    assert artifact["target_leakage_checks"]["split_overlap_count"] == 0
    assert artifact["token_budget_parity"]["all_pairs_matched"] is True
    assert artifact["token_budget_parity"]["unique_token_counts"] == [mod.TOKEN_BUDGET]
    assert artifact["split_definition_and_hashes"]["label_blind"] is True
    assert artifact["split_definition_and_hashes"]["split_overlap_count"] == 0

    forbidden = {
        "finite_domain_csp",
        "finite-domain-csp",
        "weighted_maxsat",
        "weighted-maxsat",
        "hard_soft_packing",
        "hard-soft-packing",
        "finite_state_planning",
        "finite-state-planning",
        "oracle",
        "label",
        "correct",
        "incorrect",
        "accepted",
        "rejected",
        "answer",
        "exp5826",
        "exp5840",
        "source_row",
        "row_id",
    }
    assert not forbidden.intersection(set(joined_inputs.replace(":", " ").split()))

    split_units: dict[str, set[str]] = defaultdict(set)
    split_conditions: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        split_units[row["split"]].add(row["pair_id"])
        for condition in row["conditions"]:
            split_conditions[row["split"]].add(condition["condition_id"])

    for left_name, left_values in split_units.items():
        for right_name, right_values in split_units.items():
            if left_name != right_name:
                assert left_values.isdisjoint(right_values)
                assert split_conditions[left_name].isdisjoint(split_conditions[right_name])


def test_scenario_verify_5840_fail_closed_for_missing_inputs_and_tampering(
    tmp_path: Path,
    exp5840_fixture: tuple[dict[str, Any], list[dict[str, Any]], Path],
) -> None:
    """SCENARIO-VERIFY-5840-FAIL-CLOSED: bad gates cannot report readiness."""

    blocked = mod.run(
        root=tmp_path,
        result_path=tmp_path / mod.RESULT_RELATIVE_PATH.name,
        row_file_path=tmp_path / mod.ROW_FILE_RELATIVE_PATH.name,
        test_commands=TEST_COMMANDS,
        test_exit_codes=TEST_EXIT_CODES,
        write=True,
    )
    assert blocked["status"] == "blocked"
    assert blocked["honest_verdict"].startswith("blocked:")
    assert blocked["counterfactual_fixture_ready_score"] == 0.0
    assert "missing_upstream_artifact" in blocked["preconditions_checked"]["blocked_reasons"]
    assert blocked["row_file_receipt"]["row_count"] == 0

    artifact, rows, _base = exp5840_fixture
    tampered = deepcopy(artifact)
    tampered["test_exit_codes"][TEST_COMMAND] = 1
    assert mod.counterfactual_fixture_ready_score(tampered) == 0.0
    with pytest.raises(ValueError, match="counterfactual_fixture_ready_score"):
        mod.validate_artifact(tampered)

    tampered_row = deepcopy(rows[0])
    tampered_row["conditions"][0]["model_input"] += " label"
    assert mod.target_leakage_checks([tampered_row])["answer_leakage_count"] > 0
    assert mod.rows_to_jsonl([]) == ""
    assert mod.read_row_file(tmp_path / "missing.rows.jsonl") == []
