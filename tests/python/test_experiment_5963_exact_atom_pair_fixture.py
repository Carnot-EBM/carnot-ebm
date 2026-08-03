"""Tests for Exp5963 sealed exact atom-pair fixture.

Spec refs: REQ-VERIFY-5963, SCENARIO-VERIFY-5963-ENUMERATION,
SCENARIO-VERIFY-5963-NEGATIVES, SCENARIO-VERIFY-5963-SPLITS-AND-TRANSFORMS,
SCENARIO-VERIFY-5963-REPLAY.
"""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
from typing import Any

import pytest

from carnot import experiment_5963_exact_atom_pair_fixture as mod


REPO = Path(__file__).resolve().parents[2]
VERIFY_SPEC = REPO / "openspec/capabilities/verifiable-reasoning/spec.md"
FOCUSED_TEST_COMMAND = (
    ".venv/bin/pytest tests/python/test_experiment_5963_exact_atom_pair_fixture.py "
    "-q --no-cov -n 0"
)
COVERAGE_COMMAND = (
    ".venv/bin/coverage run --rcfile=/dev/null "
    "--include=python/carnot/experiment_5963_exact_atom_pair_fixture.py "
    "-m pytest tests/python/test_experiment_5963_exact_atom_pair_fixture.py "
    "-q --no-cov -n 0 && .venv/bin/coverage report --rcfile=/dev/null "
    "--include=python/carnot/experiment_5963_exact_atom_pair_fixture.py --fail-under=100"
)
FULL_TEST_COMMAND = ".venv/bin/pytest tests/python -q"
SPEC_COMMAND = (
    ".venv/bin/python scripts/check_spec_coverage.py "
    "tests/python/test_experiment_5963_exact_atom_pair_fixture.py"
)
ADVERSARIAL_COMMAND = (
    ".venv/bin/python scripts/adversarial_verify.py "
    "results/experiment_5963_exact_atom_pair_fixture.json"
)
ROOT_CLUTTER_COMMAND = ".venv/bin/python scripts/root_clutter_sweep.py"
PROTECTED_FILE_COMMAND = (
    "git status --short -- scripts/research_conductor.py "
    "ops/changelog.md ops/status.md _bmad/traceability.md"
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


@pytest.fixture(scope="module")
def exp5963_fixture(
    tmp_path_factory: pytest.TempPathFactory,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], Path]:
    """REQ-VERIFY-5963: materialize the deterministic pair fixture once."""

    base = tmp_path_factory.mktemp("exp5963")
    conductor = REPO / "scripts/research_conductor.py"
    before_hash = mod.sha256_file(conductor)
    artifact = mod.write_artifact(
        output_path=base / mod.RESULT_RELATIVE_PATH.name,
        context_rows_path=base / mod.CONTEXT_ROW_RELATIVE_PATH.name,
        pair_rows_path=base / mod.PAIR_ROW_RELATIVE_PATH.name,
        duration_s=7.0,
        test_exit_codes=TEST_EXIT_CODES,
    )
    context_rows = mod.read_jsonl(base / mod.CONTEXT_ROW_RELATIVE_PATH.name)
    pair_rows = mod.read_jsonl(base / mod.PAIR_ROW_RELATIVE_PATH.name)
    assert mod.sha256_file(conductor) == before_hash
    return artifact, context_rows, pair_rows, base


def test_req_verify_5963_spec_declares_pair_fixture_contract() -> None:
    """REQ-VERIFY-5963: OpenSpec anchors required fields and principles."""

    text = VERIFY_SPEC.read_text(encoding="utf-8")
    section = text[text.index("### REQ-VERIFY-5963") :]
    normalized = " ".join(section.split())

    for marker in (
        "REQ-VERIFY-5963",
        "SCENARIO-VERIFY-5963-ENUMERATION",
        "SCENARIO-VERIFY-5963-NEGATIVES",
        "SCENARIO-VERIFY-5963-SPLITS-AND-TRANSFORMS",
        "SCENARIO-VERIFY-5963-REPLAY",
        mod.RESULT_RELATIVE_PATH.as_posix(),
        mod.CONTEXT_ROW_RELATIVE_PATH.as_posix(),
        mod.PAIR_ROW_RELATIVE_PATH.as_posix(),
        "`pair_fixture_ready_score`",
        "aggregation_from_upstream_artifacts",
    ):
        assert marker in section
    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert f"`{field}`" in section
    for principle in mod.FIELD_PRINCIPLES.values():
        assert " ".join(principle.split()) in normalized


def test_scenario_verify_5963_enumeration_is_public_label_blind_and_sealed() -> None:
    """SCENARIO-VERIFY-5963-ENUMERATION: candidate creation is label-blind."""

    schema = mod.versioned_pair_atom_schema()
    base_cases = mod.build_base_context_cases(min_base_cases=300)
    contexts = mod.build_context_rows(base_cases, schema)
    pairs = mod.build_pair_rows(contexts, schema)
    contract = mod.atom_schema_and_enumeration_contract(schema, contexts, pairs)
    separation = mod.model_visible_vs_hidden_label_separation(contexts, pairs)
    leakage = mod.unreachable_truth_and_leakage_counts(contexts, pairs)

    assert len(base_cases) >= 300
    assert schema["schema_hash"].startswith("sha256:")
    assert contract["candidate_source"] == "public_schema_visible_symbols_bounded_depth"
    assert contract["hidden_reference_used_for_candidate_creation"] is False
    assert contract["bounded_composition_depth"] == mod.BOUNDED_COMPOSITION_DEPTH
    assert contract["candidate_order_uses_hidden_labels"] is False
    assert separation["candidate_generation_before_split_sealing"] is True
    assert separation["label_opened_after_candidate_and_split_seal"] is True
    assert separation["hidden_labels_in_model_visible_text_count"] == 0
    assert leakage["unreachable_true_atom_count"] == 0
    assert leakage["hidden_answer_leakage_count"] == 0
    assert all(row["model_visible_text_hidden_marker_count"] == 0 for row in contexts)
    assert all("target_constraint_ir" not in mod.canonical_json(row) for row in pairs)


def test_scenario_verify_5963_negative_classes_and_five_seed_splits_are_balanced(
    exp5963_fixture: tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], Path],
) -> None:
    """SCENARIO-VERIFY-5963-NEGATIVES: balanced hard negatives cover all classes."""

    artifact, context_rows, pair_rows, _base = exp5963_fixture
    counts = artifact["base_case_pair_and_class_counts"]
    negative_manifest = artifact["negative_type_manifest"]
    splits = artifact["semantic_group_splits"]

    assert counts["base_context_case_count"] >= 300
    assert counts["compatible_pair_count"] == counts["incompatible_pair_count"]
    assert counts["pair_count"] == len(pair_rows)
    assert counts["base_context_case_count"] == len(context_rows)
    assert set(mod.REQUIRED_NEGATIVE_TYPES) <= set(counts["negative_type_counts"])
    assert all(counts["negative_type_counts"][name] > 0 for name in mod.REQUIRED_NEGATIVE_TYPES)
    assert negative_manifest["all_required_negative_types_present"] is True
    assert negative_manifest["near_semantic_negative_count"] == counts["incompatible_pair_count"]

    for seed_name, manifest in splits["five_seed_group_splits"].items():
        assert seed_name.startswith("seed_")
        assert manifest["all_groups_disjoint"] is True
        assert manifest["label_balance_by_split"]["train"]["compatible"] == manifest[
            "label_balance_by_split"
        ]["train"]["incompatible"]
        assert manifest["label_balance_by_split"]["calibration"]["compatible"] == manifest[
            "label_balance_by_split"
        ]["calibration"]["incompatible"]
        assert manifest["label_balance_by_split"]["test"]["compatible"] == manifest[
            "label_balance_by_split"
        ]["test"]["incompatible"]
        assert manifest["sibling_cross_split_leakage_count"] == 0

    assert splits["family_held_split"]["held_family_count"] >= 1
    assert splits["proof_preserving_relabel_held_split"]["held_relabel_group_count"] >= 1


def test_scenario_verify_5963_transform_receipts_controls_and_label_parity(
    exp5963_fixture: tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], Path],
) -> None:
    """SCENARIO-VERIFY-5963-SPLITS-AND-TRANSFORMS: transforms and labels replay."""

    artifact, _context_rows, pair_rows, _base = exp5963_fixture
    transforms = artifact["relabel_paraphrase_claim_flip_and_inverse_receipts"]
    controls = artifact["shortcut_control_manifest"]
    parity = artifact["python_z3_label_parity"]
    strata = artifact["hardness_density_width_and_family_strata"]

    assert transforms["paraphrase_label_invariance"] is True
    assert transforms["entity_permutation_label_invariance"] is True
    assert transforms["proof_preserving_relabel_label_invariance"] is True
    assert transforms["all_inverse_receipts_valid"] is True
    assert transforms["claim_flip_exact_inversions"] > 0
    assert transforms["claim_flip_non_invertible_count"] == 0
    assert controls["model_features_present"] is False
    assert set(mod.REQUIRED_SHORTCUT_CONTROLS) <= set(controls["controls"])
    assert all(control["exp5964_5965_gate"] for control in controls["controls"].values())
    assert parity["all_python_z3_agree"] is True
    assert parity["pair_count"] == len(pair_rows)
    assert parity["candidate_order_permutation_invariant"] is True
    assert strata["all_hardness_rows_replayed"] is True
    assert strata["density_width_surface_controls_declared"] is True

    label_counts = Counter(row["label"] for row in pair_rows)
    assert label_counts == {"compatible": len(pair_rows) // 2, "incompatible": len(pair_rows) // 2}


def test_scenario_verify_5963_artifact_rows_replay_tamper_and_validate(
    exp5963_fixture: tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], Path],
) -> None:
    """SCENARIO-VERIFY-5963-REPLAY: artifacts are hash-bound and tamper-safe."""

    artifact, _context_rows, _pair_rows, base = exp5963_fixture
    context_path = base / mod.CONTEXT_ROW_RELATIVE_PATH.name
    pair_path = base / mod.PAIR_ROW_RELATIVE_PATH.name
    loaded = json.loads((base / mod.RESULT_RELATIVE_PATH.name).read_text(encoding="utf-8"))
    rerun = mod.write_artifact(
        output_path=base / "rerun.json",
        context_rows_path=base / "rerun.contexts.jsonl",
        pair_rows_path=base / "rerun.pairs.jsonl",
        duration_s=99.0,
        test_exit_codes=TEST_EXIT_CODES,
    )

    assert loaded == artifact
    assert set(mod.REQUIRED_ARTIFACT_FIELDS) <= set(artifact)
    assert mod.validate_artifact(artifact) is True
    assert artifact["status"] == "complete_ready"
    assert artifact["honest_verdict"].startswith("complete_ready:")
    assert artifact["pair_fixture_ready_score"] == 1.0
    assert artifact["duration_s"] == pytest.approx(7.0)
    assert artifact["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert artifact["verifier_is_oracle"] is True
    assert artifact["protected_files_unchanged"]["unchanged"] is True
    assert artifact["reproducibility_checksum"] == rerun["reproducibility_checksum"]
    assert mod.replay_context_rows(context_path)["ok"] is True
    assert mod.replay_pair_rows(pair_path)["ok"] is True
    assert artifact["replay_and_tamper_matrix"]["tamper_control"]["tamper_rejected"] is True

    tampered_lines = pair_path.read_text(encoding="utf-8").splitlines()
    first_row = json.loads(tampered_lines[0])
    first_row["label"] = "incompatible"
    tampered_lines[0] = json.dumps(first_row, sort_keys=True)
    tampered_path = base / "tampered.pairs.jsonl"
    tampered_path.write_text("\n".join(tampered_lines) + "\n", encoding="utf-8")
    assert mod.replay_pair_rows(tampered_path)["ok"] is False


def test_req_verify_5963_validation_fails_closed_on_ready_gate_breaks(
    exp5963_fixture: tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], Path],
) -> None:
    """REQ-VERIFY-5963: ready artifacts reject leakage, bad parity, and drift."""

    artifact, _context_rows, _pair_rows, base = exp5963_fixture

    for key, value, message in [
        ("inference_substrate", "live_llm_inference", "inference_substrate"),
        ("verifier_is_oracle", False, "verifier_is_oracle"),
        ("pair_fixture_ready_score", 0.5, "pair_fixture_ready_score"),
        ("honest_verdict", "complete_partial: wrong", "complete_ready"),
    ]:
        broken = json.loads(json.dumps(artifact))
        broken[key] = value
        with pytest.raises(ValueError, match=message):
            mod.validate_artifact(broken)

    missing = dict(artifact)
    del missing["shortcut_control_manifest"]
    with pytest.raises(ValueError, match="missing required fields"):
        mod.validate_artifact(missing)

    leaky = json.loads(json.dumps(artifact))
    leaky["unreachable_truth_and_leakage_counts"]["hidden_answer_leakage_count"] = 1
    with pytest.raises(ValueError, match="hidden-answer leakage"):
        mod.validate_artifact(leaky)

    parity_bad = json.loads(json.dumps(artifact))
    parity_bad["python_z3_label_parity"]["all_python_z3_agree"] = False
    with pytest.raises(ValueError, match="Python/Z3 parity"):
        mod.validate_artifact(parity_bad)

    refreshed = mod.refresh_artifact_test_exit_codes(
        artifact_path=base / mod.RESULT_RELATIVE_PATH.name,
        test_exit_codes={"focused": 0, "coverage": 0},
    )
    assert refreshed["test_exit_codes"] == {"focused": 0, "coverage": 0}
