"""Tests for Exp6103 Phase D exact difficulty ladder fixture.

Spec refs: REQ-VERIFY-6103, SCENARIO-VERIFY-6103-GENERATION,
SCENARIO-VERIFY-6103-TRANSFORMS, SCENARIO-VERIFY-6103-REPLAY,
SCENARIO-VERIFY-6103-POLICY.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any

import pytest

from carnot import experiment_6103_phase_d_difficulty_ladder_fixture as mod


REPO = Path(__file__).resolve().parents[2]
SPEC = REPO / "openspec/capabilities/verifiable-reasoning/spec.md"
TEST_COMMAND = (
    ".venv/bin/pytest "
    "tests/python/test_experiment_6103_phase_d_difficulty_ladder_fixture.py "
    "-q --no-cov -n 0"
)
COVERAGE_COMMAND = (
    ".venv/bin/coverage run --rcfile=/dev/null "
    "--include=python/carnot/experiment_6103_phase_d_difficulty_ladder_fixture.py "
    "-m pytest tests/python/test_experiment_6103_phase_d_difficulty_ladder_fixture.py "
    "-q --no-cov -n 0 && .venv/bin/coverage report --rcfile=/dev/null "
    "--include=python/carnot/experiment_6103_phase_d_difficulty_ladder_fixture.py "
    "--fail-under=100"
)
FULL_TEST_COMMAND = ".venv/bin/pytest tests/python -q"
SPEC_COMMAND = ".venv/bin/python scripts/check_spec_coverage.py"
ROOT_CLUTTER_COMMAND = ".venv/bin/python scripts/root_clutter_sweep.py"
ADVERSARIAL_COMMAND = (
    ".venv/bin/python scripts/adversarial_verify.py "
    "results/experiment_6103_phase_d_difficulty_ladder_fixture.json"
)
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


def _paths(tmp_path: Path) -> tuple[Path, Path, Path]:
    return (
        tmp_path / mod.RESULT_RELATIVE_PATH.name,
        tmp_path / mod.ROW_FILE_RELATIVE_PATH.name,
        tmp_path / mod.SPLIT_MANIFEST_RELATIVE_PATH.name,
    )


def _preconditions(tmp_path: Path) -> dict[str, Any]:
    result_path, row_path, split_path = _paths(tmp_path)
    return mod.collect_preconditions(
        result_path=result_path,
        row_file_path=row_path,
        split_manifest_path=split_path,
        memory_probe=lambda: {"available_mb": 65536, "required_mb": 1024, "ok": True},
        disk_probe=lambda root: {"available_mb": 65536, "required_mb": 1024, "ok": True},
        z3_probe=lambda: {"available": True, "version": "4.16.0-fixture", "ok": True},
    )


@pytest.fixture(scope="module")
def exp6103_fixture(
    tmp_path_factory: pytest.TempPathFactory,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any], Path]:
    """REQ-VERIFY-6103: build the deterministic no-LLM ladder once."""

    base = tmp_path_factory.mktemp("exp6103")
    result_path, row_path, split_path = _paths(base)
    artifact = mod.run(
        result_path=result_path,
        row_file_path=row_path,
        split_manifest_path=split_path,
        preconditions_checked=_preconditions(base),
        test_commands=TEST_COMMANDS,
        test_exit_codes=TEST_EXIT_CODES,
        duration_s=6.103,
        write=True,
    )
    rows = mod.read_row_file(row_path)
    split_manifest = json.loads(split_path.read_text(encoding="utf-8"))
    return artifact, rows, split_manifest, base


def test_req_verify_6103_spec_declares_ladder_contract() -> None:
    """REQ-VERIFY-6103: OpenSpec names required fields and principles."""

    text = SPEC.read_text(encoding="utf-8")
    section = text[text.index("### REQ-VERIFY-6103") :]
    normalized = " ".join(section.split())

    for marker in (
        "REQ-VERIFY-6103",
        "SCENARIO-VERIFY-6103-GENERATION",
        "SCENARIO-VERIFY-6103-TRANSFORMS",
        "SCENARIO-VERIFY-6103-REPLAY",
        "SCENARIO-VERIFY-6103-POLICY",
        mod.RESULT_RELATIVE_PATH.as_posix(),
        mod.ROW_FILE_RELATIVE_PATH.as_posix(),
        mod.SPLIT_MANIFEST_RELATIVE_PATH.as_posix(),
        "600 calibration",
        "360 held-test",
        "`phase_d_ladder_fixture_ready_score`",
    ):
        assert marker in section
    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert f"`{field}`" in section
    for field, principle in mod.REQUIRED_FIELD_PRINCIPLES.items():
        assert f"`{field}`" in section
        assert " ".join(principle.split()) in normalized


def test_scenario_verify_6103_generation_counts_chance_and_source_policy(
    exp6103_fixture: tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any], Path],
) -> None:
    """SCENARIO-VERIFY-6103-GENERATION: exact low-chance splits are sealed."""

    artifact, rows, split_manifest, base = exp6103_fixture
    result_path, row_path, split_path = _paths(base)
    rerun = mod.run(
        result_path=base / "rerun.json",
        row_file_path=base / "rerun.rows.jsonl",
        split_manifest_path=base / "rerun.splits.json",
        preconditions_checked=_preconditions(base / "rerun"),
        test_commands=TEST_COMMANDS,
        test_exit_codes=TEST_EXIT_CODES,
        duration_s=999.0,
        write=True,
    )

    assert mod.validate_artifact(artifact) is True
    assert mod.verify_row_file(rows, artifact) is True
    assert mod.verify_split_manifest(split_manifest, rows, artifact) is True
    assert json.loads(result_path.read_text(encoding="utf-8")) == artifact
    assert artifact["status"] == "complete"
    assert artifact["honest_verdict"].startswith("complete_ready:")
    assert artifact["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert artifact["duration_s"] == pytest.approx(6.103)
    assert artifact["verifier_is_oracle"] is True
    assert artifact["phase_d_ladder_fixture_ready_score"] == pytest.approx(1.0)
    assert artifact["calibration_and_held_test_counts"]["calibration_question_count"] == 600
    assert artifact["calibration_and_held_test_counts"]["held_test_question_count"] == 360
    assert artifact["calibration_and_held_test_counts"]["independent_question_group_count"] == 960
    assert artifact["calibration_and_held_test_counts"]["family_counts_by_split"] == {
        "calibration": {family: 200 for family in mod.FAMILIES},
        "held_test": {family: 120 for family in mod.FAMILIES},
    }
    assert len(rows) == 960
    assert {row["family"] for row in rows} == set(mod.FAMILIES)
    assert {row["split"] for row in rows} == {"calibration", "held_test"}
    assert all(row["chance_floor"] <= 0.25 for row in rows)
    assert artifact["answer_space_and_enumerated_chance_floors"]["max_chance_floor"] <= 0.25
    assert artifact["answer_space_and_enumerated_chance_floors"]["chance_floor_ambiguity_count"] == 0
    assert artifact["preconditions_checked"]["fixture_row_source_policy"][
        "candidate_generation_artifact_imported"
    ] is False
    assert artifact["preconditions_checked"]["fixture_row_source_policy"][
        "model_response_artifact_imported"
    ] is False
    assert artifact["preconditions_checked"]["fixture_row_source_policy"][
        "imported_fixture_row_count"
    ] == 0
    assert set(mod.REQUIRED_ARTIFACT_FIELDS) <= set(artifact)
    assert artifact["row_paths_hashes_and_prefix_chain"]["row_file_sha256"] == mod.sha256_file(
        row_path
    )
    assert artifact["row_paths_hashes_and_prefix_chain"][
        "split_manifest_sha256"
    ] == mod.sha256_file(split_path)
    assert artifact["row_paths_hashes_and_prefix_chain"]["row_count"] == len(rows)
    assert artifact["row_paths_hashes_and_prefix_chain"]["terminal_prefix_hash"].startswith(
        "sha256:"
    )
    assert artifact["reproducibility_checksum"] == rerun["reproducibility_checksum"]
    assert artifact["row_paths_hashes_and_prefix_chain"][
        "terminal_prefix_hash"
    ] == rerun["row_paths_hashes_and_prefix_chain"]["terminal_prefix_hash"]


def test_scenario_verify_6103_transforms_shortcuts_and_method_validity(
    exp6103_fixture: tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any], Path],
) -> None:
    """SCENARIO-VERIFY-6103-TRANSFORMS: inverses and method labels are auditable."""

    artifact, rows, _split_manifest, _base = exp6103_fixture
    transform_manifest = artifact["proof_preserving_relabel_paraphrase_and_inverse_receipts"]
    shortcut_manifest = artifact["shortcut_salience_and_method_validity_manifest"]

    assert transform_manifest["all_transform_inverses_valid"] is True
    assert transform_manifest["proof_preserving_transform_count"] == len(rows) * 4
    assert transform_manifest["boundary_condition_row_count"] > 0
    assert shortcut_manifest["final_answer_correctness_separate_from_method_validity"] is True
    assert shortcut_manifest["right_answer_wrong_method_count"] > 0
    assert shortcut_manifest["invalid_shortcut_method_count"] == len(rows) * 3

    saw_shortcut_right_answer = False
    for row in rows[:40]:
        assert row["schema"] == mod.ROW_SCHEMA
        assert row["row_hash"] == mod.row_hash(row)
        assert row["model_facing_prompt_hash"] == mod.sha256_text(row["model_facing_prompt"])
        lowered_prompt = row["model_facing_prompt"].lower()
        for forbidden in ("exact_label", "validator", "certificate", "z3 trace", "correct answer"):
            assert forbidden not in lowered_prompt
        assert set(row["transform_receipts"]) == set(mod.TRANSFORM_KINDS)
        assert mod.validate_transform_receipts(row) is True
        assert row["shortcut_salience"]["distractor_label"] in mod.LABELS
        exact = row["exact_label"]
        methods = {item["method_id"]: item for item in row["method_validity_labels"]}
        assert methods["exact_derivation"]["answer_label"] == exact
        assert methods["exact_derivation"]["method_valid"] is True
        assert methods["exact_derivation"]["final_answer_correct"] is True
        for method_id in ("salient_shortcut", "answer_order_prior", "solver_conflict_proxy"):
            assert methods[method_id]["method_valid"] is False
            assert methods[method_id]["final_answer_correct"] is (
                methods[method_id]["answer_label"] == exact
            )
        saw_shortcut_right_answer = (
            saw_shortcut_right_answer or methods["salient_shortcut"]["final_answer_correct"]
        )
    assert saw_shortcut_right_answer


def test_scenario_verify_6103_replay_splits_order_and_tamper_fail_closed(
    exp6103_fixture: tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any], Path],
) -> None:
    """SCENARIO-VERIFY-6103-REPLAY: exact authorities and manifests fail closed."""

    artifact, rows, split_manifest, _base = exp6103_fixture
    counts = artifact["duplicate_leakage_unreachable_and_order_dependence_counts"]
    parity = artifact["python_z3_parity"]

    assert parity["python_z3_disagreement_count"] == 0
    assert parity["method_validity_disagreement_count"] == 0
    assert counts == {
        "duplicate_semantic_group_count": 0,
        "split_leakage_count": 0,
        "unreachable_truth_count": 0,
        "prompt_hidden_answer_leakage_count": 0,
        "answer_order_dependence_count": 0,
        "chance_floor_ambiguity_count": 0,
        "row_hash_tamper_count": 0,
    }
    for row in rows[:30]:
        python_receipt = mod.python_validate_row(row)
        z3_receipt = mod.z3_validate_row(row)
        assert python_receipt["exact_label"] == z3_receipt["exact_label"] == row["exact_label"]
        assert python_receipt["method_validity_labels"] == z3_receipt["method_validity_labels"]
        assert mod.answer_order_dependence_receipt(row)["order_dependent"] is False

    tampered = deepcopy(rows[0])
    tampered["exact_label"] = next(label for label in mod.LABELS if label != rows[0]["exact_label"])
    with pytest.raises(mod.ManifestReplayError):
        mod.verify_row_file([tampered, *rows[1:]], artifact)

    duplicated = [*rows, deepcopy(rows[0])]
    with pytest.raises(mod.ManifestReplayError):
        mod.verify_row_file(duplicated, artifact)

    duplicated_group = deepcopy(rows[1])
    duplicated_group["semantic_group_id"] = rows[0]["semantic_group_id"]
    duplicated_group["row_hash"] = mod.row_hash(duplicated_group)
    with pytest.raises(mod.ManifestReplayError):
        mod.verify_row_file([rows[0], duplicated_group, *rows[2:]], artifact)

    hash_mismatch_artifact = deepcopy(artifact)
    hash_mismatch_artifact["row_paths_hashes_and_prefix_chain"]["row_hashes"][
        rows[0]["row_id"]
    ] = "sha256:" + "0" * 64
    with pytest.raises(mod.ManifestReplayError):
        mod.verify_row_file(rows, hash_mismatch_artifact)

    leaked_split = deepcopy(split_manifest)
    leaked_group = split_manifest["splits"]["calibration"]["semantic_group_ids"][0]
    leaked_split["splits"]["held_test"]["semantic_group_ids"].append(leaked_group)
    with pytest.raises(mod.ManifestReplayError):
        mod.verify_split_manifest(leaked_split, rows, artifact)


def test_scenario_verify_6103_policy_and_field_provenance(
    exp6103_fixture: tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any], Path],
) -> None:
    """SCENARIO-VERIFY-6103-POLICY: held labels cannot steer Exp6104 calibration."""

    artifact, _rows, _split_manifest, _base = exp6103_fixture
    policy = artifact["calibration_policy_and_test_secrecy"]

    assert policy["exp6104_allowed_calibration_actions"] == [
        "select_difficulty_strata",
        "select_temperature",
        "select_fixed_decoding_parameters",
    ]
    assert policy["held_test_labels_may_be_inspected"] is False
    assert policy["held_rows_may_change_after_sealing"] is False
    assert policy["target_test_band_measured_later"] == [0.4, 0.7]
    assert policy["target_band_promised_by_fixture"] is False
    assert artifact["missing_verifier_gaps"] == []
    assert artifact["protected_files_unchanged"]["all_unchanged"] is True
    partial = deepcopy(artifact)
    partial["phase_d_ladder_fixture_ready_score"] = 0.0
    assert mod.honest_verdict(partial).startswith("complete_partial:")
    assert set(artifact["test_exit_codes"]) == set(artifact["test_commands"])
    assert all(code == 0 for code in artifact["test_exit_codes"].values())
    for field, principle in mod.REQUIRED_FIELD_PRINCIPLES.items():
        assert artifact["field_provenance"][field]["principle"] == principle
