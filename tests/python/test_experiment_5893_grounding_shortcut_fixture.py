"""Tests for Exp5893 grounding shortcut fixture.

Spec refs: REQ-VERIFY-5893, SCENARIO-VERIFY-5893-SCHEMA,
SCENARIO-VERIFY-5893-SHORTCUTS, SCENARIO-VERIFY-5893-CONTROLS,
SCENARIO-VERIFY-5893-REPLAY-AND-LEAKAGE.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any

import pytest

from carnot import experiment_5893_grounding_shortcut_fixture as mod


REPO = Path(__file__).resolve().parents[2]
VERIFY_SPEC = REPO / "openspec/capabilities/verification/spec.md"
FOCUSED_TEST_COMMAND = (
    ".venv/bin/pytest "
    "tests/python/test_experiment_5893_grounding_shortcut_fixture.py "
    "-q --no-cov -n 0"
)
COVERAGE_COMMAND = (
    ".venv/bin/coverage run --rcfile=/dev/null "
    "--include=python/carnot/experiment_5893_grounding_shortcut_fixture.py "
    "-m pytest tests/python/test_experiment_5893_grounding_shortcut_fixture.py "
    "-q --no-cov -n 0 && .venv/bin/coverage report --rcfile=/dev/null "
    "--include=python/carnot/experiment_5893_grounding_shortcut_fixture.py "
    "--fail-under=100"
)
FULL_TEST_COMMAND = ".venv/bin/pytest tests/python -q"
SPEC_COMMAND = ".venv/bin/python scripts/check_spec_coverage.py"
ADVERSARIAL_COMMAND = (
    ".venv/bin/python scripts/adversarial_verify.py --json "
    "results/experiment_5893_grounding_shortcut_fixture.json"
)
ROOT_CLUTTER_COMMAND = ".venv/bin/python scripts/root_clutter_sweep.py"
PROTECTED_FILE_COMMAND = (
    '.venv/bin/python -c "from pathlib import Path; '
    "assert Path('scripts/research_conductor.py').exists()\""
)
TEST_COMMANDS = [
    FOCUSED_TEST_COMMAND,
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
        disk_probe=lambda root: {"available_mb": 32768, "required_mb": 512, "ok": True},
    )


@pytest.fixture(scope="module")
def exp5893_artifact(
    tmp_path_factory: pytest.TempPathFactory,
) -> tuple[dict[str, Any], list[dict[str, Any]], Path]:
    """REQ-VERIFY-5893: build the exact fixture once for row-level assertions."""

    base = tmp_path_factory.mktemp("exp5893")
    before_hashes = {
        path.as_posix(): mod.sha256_file(REPO / path)
        for path in mod.PROTECTED_RELATIVE_PATHS
    }
    artifact = mod.run(
        result_path=base / mod.RESULT_RELATIVE_PATH.name,
        row_file_path=base / mod.ROW_FILE_RELATIVE_PATH.name,
        preconditions_checked=_preconditions(base),
        test_commands=TEST_COMMANDS,
        test_exit_codes=TEST_EXIT_CODES,
        duration_s=5.0,
        write=True,
    )
    rows = mod.read_rows(base / mod.ROW_FILE_RELATIVE_PATH.name)
    assert before_hashes == {
        path.as_posix(): mod.sha256_file(REPO / path)
        for path in mod.PROTECTED_RELATIVE_PATHS
    }
    return artifact, rows, base


def test_req_verify_5893_spec_declares_grounding_shortcut_contract() -> None:
    """REQ-VERIFY-5893: OpenSpec anchors the Exp5893 fields and principles."""

    text = VERIFY_SPEC.read_text(encoding="utf-8")
    section = text[text.index("### REQ-VERIFY-5893") :]
    normalized = " ".join(section.split())

    for marker in (
        "REQ-VERIFY-5893",
        "SCENARIO-VERIFY-5893-SCHEMA",
        "SCENARIO-VERIFY-5893-SHORTCUTS",
        "SCENARIO-VERIFY-5893-CONTROLS",
        "SCENARIO-VERIFY-5893-REPLAY-AND-LEAKAGE",
        mod.RESULT_RELATIVE_PATH.as_posix(),
        mod.ROW_FILE_RELATIVE_PATH.as_posix(),
        mod.INFERENCE_SUBSTRATE,
        "`grounding_shortcut_fixture_ready_score`",
        "`shortcut_type_definitions`",
    ):
        assert marker in section
    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert f"`{field}`" in section
    for principle in mod.REQUIRED_FIELD_PRINCIPLES.values():
        assert " ".join(principle.split()) in normalized


def test_req_verify_5893_terminal_artifact_is_ready_and_hash_bound(
    exp5893_artifact: tuple[dict[str, Any], list[dict[str, Any]], Path],
) -> None:
    """REQ-VERIFY-5893: terminal JSON/JSONL are complete and replay stable."""

    artifact, rows, base = exp5893_artifact
    rerun = mod.run(
        result_path=base / "rerun.json",
        row_file_path=base / "rerun.rows.jsonl",
        preconditions_checked=_preconditions(base / "rerun"),
        test_commands=TEST_COMMANDS,
        test_exit_codes=TEST_EXIT_CODES,
        duration_s=99.0,
        write=True,
    )

    assert mod.validate_artifact(artifact) is True
    assert set(mod.REQUIRED_ARTIFACT_FIELDS) <= set(artifact)
    assert artifact["status"] == "ready"
    assert artifact["honest_verdict"].startswith("ready:")
    assert artifact["grounding_shortcut_fixture_ready_score"] == 1.0
    assert artifact["duration_s"] == pytest.approx(5.0)
    assert artifact["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert artifact["verifier_is_oracle"] is True
    assert artifact["upstream_gate_receipt"]["exp5892_non_retired_admission"] is True
    assert artifact["row_file_receipt"]["row_count"] == len(rows)
    assert artifact["deterministic_replay_receipt"]["content_match"] is True
    assert artifact["protected_files_unchanged"]["all_unchanged"] is True
    assert artifact["reproducibility_checksum"] == rerun["reproducibility_checksum"]
    assert json.loads((base / mod.RESULT_RELATIVE_PATH.name).read_text()) == artifact

    for field, principle in mod.REQUIRED_FIELD_PRINCIPLES.items():
        assert artifact["field_provenance"][field]["principle"] == principle


def test_scenario_verify_5893_rows_separate_schema_oracles_and_witnesses(
    exp5893_artifact: tuple[dict[str, Any], list[dict[str, Any]], Path],
) -> None:
    """SCENARIO-VERIFY-5893-SCHEMA: rows expose distinct exact concepts."""

    _artifact, rows, _base = exp5893_artifact
    families = {row["family"] for row in rows}
    regimes = {row["grounding_regime"] for row in rows}

    assert len(families) >= 3
    assert {
        "canonical_one_to_one",
        "one_to_one_negative_control",
        "constraint_satisfaction_many_to_one",
        "constraint_satisfaction_soft_mass_swap",
        "cognition_biased_permutation",
        "soft_distributed_control",
        "shuffled_control",
        "label_permutation_control",
        "frequency_balanced_control",
        "surface_matched_control",
        "no_information_control",
    } <= regimes

    for row in rows:
        for field in (
            "concepts",
            "logical_atoms",
            "grounding_matrix",
            "intended_semantics",
            "encoded_constraint",
            "exact_semantic_label",
            "exact_constraint_label",
            "exact_outcome",
            "certificate",
            "witness",
            "provenance",
            "relabel_group",
            "family_group",
            "split_group",
            "chronology_batch",
            "row_hash",
        ):
            assert field in row
        assert row["grounding_matrix"]["rows"] == [concept["concept_id"] for concept in row["concepts"]]
        assert row["grounding_matrix"]["columns"] == [
            atom["atom_id"] for atom in row["logical_atoms"]
        ]
        assert row["witness"]["semantic_oracle"]["label"] is row["exact_semantic_label"]
        assert row["witness"]["constraint_oracle"]["label"] is row["exact_constraint_label"]
        assert row["exact_outcome"]["semantic_label"] is row["exact_semantic_label"]
        assert row["exact_outcome"]["constraint_label"] is row["exact_constraint_label"]


def test_scenario_verify_5893_shortcut_modes_have_canonical_counterparts(
    exp5893_artifact: tuple[dict[str, Any], list[dict[str, Any]], Path],
) -> None:
    """SCENARIO-VERIFY-5893-SHORTCUTS: both shortcut modes retain headroom."""

    artifact, rows, _base = exp5893_artifact
    by_id = {row["row_id"]: row for row in rows}
    shortcut_rows = [
        row
        for row in rows
        if row["shortcut_type"] in {"constraint_satisfaction_shortcut", "cognition_shortcut"}
    ]
    counts = artifact["label_witness_and_headroom_balance"]["shortcut_headroom_counts"]

    assert counts["constraint_satisfaction_shortcut"] > 0
    assert counts["cognition_shortcut"] > 0
    assert shortcut_rows
    for row in shortcut_rows:
        counterpart = by_id[row["canonical_counterpart_row_id"]]
        assert counterpart["grounding_regime"] == "canonical_one_to_one"
        assert counterpart["exact_semantic_label"] is True
        assert counterpart["exact_constraint_label"] is True
        assert counterpart["split_group"] == row["split_group"]
        assert counterpart["semantic_problem_id"] == row["semantic_problem_id"]
        assert row["exact_semantic_label"] is False
        assert row["exact_constraint_label"] is True
        assert row["witness"]["semantic_constraint_disagreement"] is True
        assert row["certificate"]["validated"] is True


def test_scenario_verify_5893_controls_are_balanced_matched_and_leakage_safe(
    exp5893_artifact: tuple[dict[str, Any], list[dict[str, Any]], Path],
) -> None:
    """SCENARIO-VERIFY-5893-CONTROLS: controls and groups replay cleanly."""

    artifact, rows, _base = exp5893_artifact
    controls = artifact["one_to_one_soft_distributed_and_shuffled_controls"]
    bias = artifact["bias_and_frequency_controls"]
    leakage = artifact["split_and_group_leakage_receipts"]

    assert controls["all_required_controls_present"] is True
    assert controls["no_information_controls"]["count"] > 0
    assert controls["no_information_controls"]["answer_bearing_grounding"] is False
    assert controls["surface_matched_controls"]["all_within_tolerance"] is True
    assert controls["soft_distributed_controls"]["exact_constraint_labels_replayed"] is True
    assert bias["biased_frequency_rows_present"] is True
    assert bias["exact_label_balance"]["constraint_true"] == bias["exact_label_balance"]["constraint_false"]
    assert bias["exact_label_balance"]["semantic_true"] == bias["exact_label_balance"]["semantic_false"]
    assert leakage["cross_split_semantic_duplicate_count"] == 0
    assert leakage["all_group_leakage_checks_passed"] is True

    split_by_group: dict[str, set[str]] = {}
    for row in rows:
        split_by_group.setdefault(row["split_group"], set()).add(row["split"])
    assert all(len(splits) == 1 for splits in split_by_group.values())


def test_scenario_verify_5893_defensive_validation_branches(
    tmp_path: Path,
    exp5893_artifact: tuple[dict[str, Any], list[dict[str, Any]], Path],
) -> None:
    """SCENARIO-VERIFY-5893-REPLAY-AND-LEAKAGE: bad evidence fails closed."""

    artifact, _rows, _base = exp5893_artifact

    bad_preconditions = mod.collect_preconditions(
        root=tmp_path / "missing-root",
        result_path=tmp_path / "missing-root" / mod.RESULT_RELATIVE_PATH.name,
        row_file_path=tmp_path / "missing-root" / mod.ROW_FILE_RELATIVE_PATH.name,
        memory_probe=lambda: {"available_mb": 1, "required_mb": 1024, "ok": False},
        disk_probe=lambda root: {"available_mb": 32768, "required_mb": 512, "ok": True},
    )
    blocked = mod.run(
        root=tmp_path / "missing-root",
        result_path=tmp_path / "blocked.json",
        row_file_path=tmp_path / "blocked.rows.jsonl",
        preconditions_checked=bad_preconditions,
        test_commands=TEST_COMMANDS,
        test_exit_codes=TEST_EXIT_CODES,
        duration_s=1.0,
        write=True,
    )
    assert blocked["status"] == "blocked"
    assert blocked["honest_verdict"].startswith("blocked:")
    assert blocked["grounding_shortcut_fixture_ready_score"] == 0.0
    assert mod.validate_artifact(blocked) is True

    missing_witness = deepcopy(artifact)
    missing_witness["label_witness_and_headroom_balance"]["missing_witness_count"] = 1
    missing_witness["grounding_shortcut_fixture_ready_score"] = (
        mod.grounding_shortcut_fixture_ready_score(missing_witness)
    )
    missing_witness["status"] = mod.status(missing_witness)
    missing_witness["honest_verdict"] = mod.honest_verdict(missing_witness)
    missing_witness["reproducibility_checksum"] = mod.reproducibility_checksum(
        missing_witness
    )
    assert missing_witness["status"] == "blocked"
    assert mod.validate_artifact(missing_witness) is True

    null_artifact = deepcopy(artifact)
    null_artifact["label_witness_and_headroom_balance"]["shortcut_headroom_counts"][
        "cognition_shortcut"
    ] = 0
    null_artifact["grounding_shortcut_fixture_ready_score"] = (
        mod.grounding_shortcut_fixture_ready_score(null_artifact)
    )
    null_artifact["status"] = mod.status(null_artifact)
    null_artifact["honest_verdict"] = mod.honest_verdict(null_artifact)
    null_artifact["reproducibility_checksum"] = mod.reproducibility_checksum(null_artifact)
    assert null_artifact["status"] == "complete_null"
    assert null_artifact["honest_verdict"].startswith("complete_null:")
    assert mod.validate_artifact(null_artifact) is True

    missing = deepcopy(artifact)
    missing.pop("row_file_receipt")
    with pytest.raises(ValueError, match="missing_fields"):
        mod.validate_artifact(missing)

    checksum_bad = deepcopy(artifact)
    checksum_bad["honest_verdict"] = "ready: edited"
    with pytest.raises(ValueError, match="reproducibility_checksum"):
        mod.validate_artifact(checksum_bad)

    score_bad = deepcopy(artifact)
    score_bad["protected_files_unchanged"]["all_unchanged"] = False
    score_bad["reproducibility_checksum"] = mod.reproducibility_checksum(score_bad)
    with pytest.raises(ValueError, match="grounding_shortcut_fixture_ready_score"):
        mod.validate_artifact(score_bad)

    status_bad = deepcopy(artifact)
    status_bad["status"] = "blocked"
    status_bad["reproducibility_checksum"] = mod.reproducibility_checksum(status_bad)
    with pytest.raises(ValueError, match="status"):
        mod.validate_artifact(status_bad)

    verdict_bad = deepcopy(artifact)
    verdict_bad["honest_verdict"] = "ready: wrong_but_rehashed"
    verdict_bad["reproducibility_checksum"] = mod.reproducibility_checksum(verdict_bad)
    with pytest.raises(ValueError, match="honest_verdict"):
        mod.validate_artifact(verdict_bad)
