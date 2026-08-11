"""Tests for Exp6313 exact code safety pair fixture.

Spec refs: REQ-CODE-6313, SCENARIO-CODE-6313-SIDECARS,
SCENARIO-CODE-6313-SPLITS, SCENARIO-CODE-6313-CONTROLS.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from carnot import experiment_6313_exact_code_safety_pair_fixture as mod


REPO = Path(__file__).resolve().parents[2]
CODE_SPEC = REPO / "openspec/capabilities/code-verification/spec.md"
TEST_COMMAND = (
    ".venv/bin/pytest "
    "tests/python/test_experiment_6313_exact_code_safety_pair_fixture.py "
    "-q --no-cov -n 0"
)
COVERAGE_COMMAND = (
    ".venv/bin/coverage run --rcfile=/dev/null "
    "--include=python/carnot/experiment_6313_exact_code_safety_pair_fixture.py "
    "-m pytest tests/python/test_experiment_6313_exact_code_safety_pair_fixture.py "
    "-q --no-cov -n 0 && .venv/bin/coverage report --rcfile=/dev/null "
    "--include=python/carnot/experiment_6313_exact_code_safety_pair_fixture.py "
    "--fail-under=100"
)
FULL_TEST_COMMAND = ".venv/bin/pytest tests/python -q"
SPEC_COMMAND = (
    ".venv/bin/python scripts/check_spec_coverage.py "
    "tests/python/test_experiment_6313_exact_code_safety_pair_fixture.py"
)
RUN_COMMAND = (
    ".venv/bin/python -m carnot.experiment_6313_exact_code_safety_pair_fixture --date 20260811"
)
ADVERSARIAL_COMMAND = (
    ".venv/bin/python scripts/adversarial_verify.py "
    "results/experiment_6313_exact_code_safety_pair_fixture.json"
)
TEST_COMMANDS = [
    TEST_COMMAND,
    COVERAGE_COMMAND,
    FULL_TEST_COMMAND,
    SPEC_COMMAND,
    RUN_COMMAND,
    ADVERSARIAL_COMMAND,
]
TEST_EXIT_CODES = {command: 0 for command in TEST_COMMANDS}


@pytest.fixture(scope="module")
def exp6313_fixture(
    tmp_path_factory: pytest.TempPathFactory,
) -> tuple[dict[str, object], list[dict[str, object]], list[dict[str, object]], Path]:
    """REQ-CODE-6313: build the local exact fixture once for this module."""

    base = tmp_path_factory.mktemp("exp6313")
    artifact = mod.run(
        result_path=base / mod.RESULT_RELATIVE_PATH.name,
        corpus_path=base / mod.CORPUS_RELATIVE_PATH.name,
        sidecar_path=base / mod.SIDECAR_RELATIVE_PATH.name,
        control_manifest_path=base / mod.CONTROL_MANIFEST_RELATIVE_PATH.name,
        split_manifest_path=base / mod.SPLIT_MANIFEST_RELATIVE_PATH.name,
        protected_hashes_before=mod.protected_file_hashes(),
        test_commands=TEST_COMMANDS,
        test_exit_codes=TEST_EXIT_CODES,
        write=True,
    )
    corpus = mod.read_jsonl(base / mod.CORPUS_RELATIVE_PATH.name)
    sidecars = mod.read_jsonl(base / mod.SIDECAR_RELATIVE_PATH.name)
    return artifact, corpus, sidecars, base


def test_req_code_6313_spec_declares_required_fields_and_scenarios() -> None:
    """REQ-CODE-6313: OpenSpec anchors the artifact schema and scenarios."""

    text = CODE_SPEC.read_text(encoding="utf-8")
    section = text[text.index("### REQ-CODE-6313") :]
    normalized = " ".join(section.split())

    for marker in (
        "SCENARIO-CODE-6313-SIDECARS",
        "SCENARIO-CODE-6313-SPLITS",
        "SCENARIO-CODE-6313-CONTROLS",
        mod.RESULT_RELATIVE_PATH.as_posix(),
        "`verifier_is_oracle` shall be true",
    ):
        assert marker in section

    for field, principle in mod.REQUIRED_FIELD_PRINCIPLES.items():
        assert f"`{field}`" in section
        assert " ".join(principle.split()) in normalized


def test_req_code_6313_artifact_is_hash_bound_replayable_and_licensed(
    exp6313_fixture: tuple[
        dict[str, object], list[dict[str, object]], list[dict[str, object]], Path
    ],
) -> None:
    """REQ-CODE-6313: JSON, corpus, sidecar, controls, and splits replay exactly."""

    artifact, corpus, sidecars, base = exp6313_fixture
    rerun = mod.run(
        result_path=base / "rerun.json",
        corpus_path=base / "rerun.corpus.jsonl",
        sidecar_path=base / "rerun.sidecars.jsonl",
        control_manifest_path=base / "rerun.controls.json",
        split_manifest_path=base / "rerun.splits.json",
        protected_hashes_before=mod.protected_file_hashes(),
        test_commands=TEST_COMMANDS,
        test_exit_codes=TEST_EXIT_CODES,
        write=True,
    )

    assert mod.validate_artifact(artifact) is True
    assert json.loads((base / mod.RESULT_RELATIVE_PATH.name).read_text()) == artifact
    assert artifact["status"] == "complete_ready"
    assert artifact["honest_verdict"].startswith("ready:")
    assert artifact["exact_code_safety_fixture_ready_score"] == 1.0
    assert artifact["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert artifact["verifier_is_oracle"] is True
    assert set(mod.REQUIRED_ARTIFACT_FIELDS) <= set(artifact)
    assert set(artifact["test_exit_codes"]) == set(artifact["test_commands"])
    assert all(code == 0 for code in artifact["test_exit_codes"].values())
    assert artifact["source_and_license_receipts"]["external_corpus_count"] == 0
    assert artifact["source_and_license_receipts"]["license_id"] == "MIT-0"
    assert artifact["corpus_path_and_hash"]["sha256"] == mod.sha256_file(
        base / mod.CORPUS_RELATIVE_PATH.name
    )
    assert artifact["sidecar_path_and_hash"]["sha256"] == mod.sha256_file(
        base / mod.SIDECAR_RELATIVE_PATH.name
    )
    assert artifact["control_manifest_path_and_hash"]["sha256"] == mod.sha256_file(
        base / mod.CONTROL_MANIFEST_RELATIVE_PATH.name
    )
    assert artifact["split_manifest_path_and_hash"]["sha256"] == mod.sha256_file(
        base / mod.SPLIT_MANIFEST_RELATIVE_PATH.name
    )
    assert artifact["reproducibility_checksum"] == rerun["reproducibility_checksum"]
    assert artifact["corpus_path_and_hash"]["sha256"] == rerun["corpus_path_and_hash"]["sha256"]
    assert artifact["sidecar_path_and_hash"]["sha256"] == rerun["sidecar_path_and_hash"]["sha256"]
    assert len(corpus) == len(sidecars) == mod.EXPECTED_PAIR_COUNT

    for field, principle in mod.REQUIRED_FIELD_PRINCIPLES.items():
        assert artifact["field_principles"][field] == principle
        assert artifact["field_provenance"][field]["principle"] == principle


def test_scenario_code_6313_sidecars_label_only_declared_safety_properties(
    exp6313_fixture: tuple[
        dict[str, object], list[dict[str, object]], list[dict[str, object]], Path
    ],
) -> None:
    """SCENARIO-CODE-6313-SIDECARS: exact sidecars prove each pair label."""

    artifact, corpus, sidecars, _base = exp6313_fixture
    forbidden_surface_tokens = {"vulnerable", "fixed", "unsafe", "safe", "label", "template"}

    assert artifact["compile_results"]["all_passed"] is True
    assert artifact["executable_property_results"]["all_passed"] is True
    assert artifact["ast_and_constraint_results"]["all_passed"] is True
    assert artifact["targeted_mutation_results"]["all_passed"] is True
    assert artifact["vulnerable_fixed_label_receipts"]["all_labels_proven"] is True
    assert artifact["length_and_token_proxy_balance"]["all_pairs_balanced"] is True

    seen_pairs = set()
    for row, sidecar in zip(corpus, sidecars, strict=True):
        assert row["pair_id"] == sidecar["pair_id"]
        assert row["pair_hash"] == mod.row_hash(row)
        assert sidecar["sidecar_hash"] == mod.sidecar_hash(sidecar)
        assert row["pair_id"] not in seen_pairs
        seen_pairs.add(row["pair_id"])
        assert row["vulnerable"]["char_length"] == row["fixed"]["char_length"]
        assert row["vulnerable"]["token_proxy_count"] == row["fixed"]["token_proxy_count"]
        assert sidecar["compile"]["vulnerable"]["ok"] is True
        assert sidecar["compile"]["fixed"]["ok"] is True
        assert sidecar["executable_property"]["vulnerable_label"] == "vulnerable"
        assert sidecar["executable_property"]["fixed_label"] == "fixed"
        assert sidecar["ast_or_constraint"]["vulnerable_label"] == "vulnerable"
        assert sidecar["ast_or_constraint"]["fixed_label"] == "fixed"
        assert sidecar["targeted_mutation"]["mutation_detected"] is True
        assert sidecar["label_receipt"]["validators_agree"] is True
        lower_surface = (row["vulnerable"]["code"] + "\n" + row["fixed"]["code"]).lower()
        assert forbidden_surface_tokens.isdisjoint(lower_surface.replace("_", " ").split())


def test_scenario_code_6313_splits_and_controls_are_leakage_clean(
    exp6313_fixture: tuple[
        dict[str, object], list[dict[str, object]], list[dict[str, object]], Path
    ],
) -> None:
    """SCENARIO-CODE-6313-SPLITS/CONTROLS: groups do not leak and controls are sealed."""

    artifact, corpus, sidecars, base = exp6313_fixture
    split_manifest = json.loads((base / mod.SPLIT_MANIFEST_RELATIVE_PATH.name).read_text())
    control_manifest = json.loads((base / mod.CONTROL_MANIFEST_RELATIVE_PATH.name).read_text())

    assert artifact["duplicate_and_overlap_checks"]["all_checks_passed"] is True
    assert artifact["duplicate_and_overlap_checks"]["split_leakage_count"] == 0
    assert artifact["duplicate_and_overlap_checks"]["normalized_text_overlap_count"] == 0
    assert artifact["duplicate_and_overlap_checks"]["template_overlap_count"] == 0
    assert artifact["duplicate_and_overlap_checks"]["source_overlap_count"] == 0
    assert artifact["duplicate_and_overlap_checks"]["mutation_overlap_count"] == 0
    assert artifact["held_weakness_source_template_and_perturbation_groups"]
    assert artifact["positive_and_negative_control_results"]["all_controls_passed"] is True
    assert control_manifest["held_labels_exposed_to_surface_selection"] is False

    for control_name in (
        "aa_duplicates",
        "semantically_irrelevant_edits",
        "label_permutations",
        "pair_swaps",
        "evaluator_swaps",
    ):
        assert control_manifest[control_name]

    split_by_pair = {
        pair_id: split
        for split, pair_ids in split_manifest["pair_ids_by_split"].items()
        for pair_id in pair_ids
    }
    assert set(split_by_pair) == {row["pair_id"] for row in corpus}
    assert {row["split"] for row in corpus} == set(mod.SPLIT_ORDER)
    assert artifact["minimum_power_projection"]["claim"] == "not_powered_for_universal_coverage"
    assert all(item["label_receipt"]["validators_agree"] for item in sidecars)


def test_scenario_code_6313_fail_closed_for_tampering_and_inconsistent_rows(
    exp6313_fixture: tuple[
        dict[str, object], list[dict[str, object]], list[dict[str, object]], Path
    ],
) -> None:
    """SCENARIO-CODE-6313-SIDECARS: tampered rows cannot stay ready."""

    artifact, corpus, sidecars, _base = exp6313_fixture

    bad_exit = deepcopy(artifact)
    bad_exit["test_exit_codes"][TEST_COMMAND] = 1
    bad_exit["exact_code_safety_fixture_ready_score"] = mod.ready_score(bad_exit)
    bad_exit["honest_verdict"] = mod.honest_verdict(bad_exit)
    bad_exit["reproducibility_checksum"] = mod.reproducibility_checksum(bad_exit)
    assert bad_exit["exact_code_safety_fixture_ready_score"] == 0.0
    with pytest.raises(ValueError, match="exact_code_safety_fixture_ready_score"):
        mod.validate_artifact(bad_exit)

    bad_balance = deepcopy(corpus[0])
    bad_balance["fixed"]["code"] += "\n"
    assert mod.length_and_token_proxy_balance([bad_balance])["all_pairs_balanced"] is False

    bad_sidecar = deepcopy(sidecars[0])
    bad_sidecar["label_receipt"]["validators_agree"] = False
    assert mod.vulnerable_fixed_label_receipts([bad_sidecar])["all_labels_proven"] is False

    invalid = deepcopy(corpus[0])
    invalid["vulnerable"]["code"] = "def broken(:\n    pass\n"
    excluded = mod.evaluate_rows([invalid])["invalid_or_excluded_rows"]
    assert excluded[0]["reason"] == "compile_sidecar_failed"


def test_req_code_6313_readiness_and_validation_reject_each_failed_gate(
    exp6313_fixture: tuple[
        dict[str, object], list[dict[str, object]], list[dict[str, object]], Path
    ],
) -> None:
    """REQ-CODE-6313: each readiness gate has a visible blocked reason."""

    artifact, _corpus, _sidecars, _base = exp6313_fixture
    gate_cases = [
        ("compile_sidecars_failed", ("compile_results", "all_passed"), False),
        ("executable_sidecars_failed", ("executable_property_results", "all_passed"), False),
        ("ast_or_constraint_sidecars_failed", ("ast_and_constraint_results", "all_passed"), False),
        ("targeted_mutations_failed", ("targeted_mutation_results", "all_passed"), False),
        ("labels_not_proven", ("vulnerable_fixed_label_receipts", "all_labels_proven"), False),
        (
            "length_or_token_imbalance",
            ("length_and_token_proxy_balance", "all_pairs_balanced"),
            False,
        ),
        (
            "split_leakage_or_duplicates",
            ("duplicate_and_overlap_checks", "all_checks_passed"),
            False,
        ),
        (
            "controls_failed",
            ("positive_and_negative_control_results", "all_controls_passed"),
            False,
        ),
        ("protected_files_changed", ("protected_files_unchanged", "unchanged"), False),
        ("preconditions_failed", ("preconditions_checked", "preconditions_ready"), False),
        ("verifier_not_oracle", ("verifier_is_oracle",), False),
    ]

    for reason, path, value in gate_cases:
        bad = deepcopy(artifact)
        if len(path) == 1:
            bad[path[0]] = value
        else:
            bad[path[0]][path[1]] = value
        assert reason in mod.blocked_reasons(bad)

    invalid = deepcopy(artifact)
    invalid["invalid_or_excluded_rows"] = [{"pair_id": "x", "reason": "bad"}]
    assert "invalid_rows_present" in mod.blocked_reasons(invalid)

    missing = deepcopy(artifact)
    missing.pop("status")
    assert "missing_required_fields" in mod.blocked_reasons(missing)
    with pytest.raises(ValueError, match="missing required fields"):
        mod.validate_artifact(missing)

    incomplete_principles = deepcopy(artifact)
    incomplete_principles["field_principles"].pop("status")
    assert "field_principles_incomplete" in mod.blocked_reasons(incomplete_principles)
    with pytest.raises(ValueError, match="field_principles incomplete"):
        mod.validate_artifact(incomplete_principles)

    score_mismatch = deepcopy(artifact)
    score_mismatch["test_exit_codes"][TEST_COMMAND] = 1
    with pytest.raises(ValueError, match="exact_code_safety_fixture_ready_score mismatch"):
        mod.validate_artifact(score_mismatch)

    checksum_mismatch = deepcopy(artifact)
    checksum_mismatch["minimum_power_projection"]["claim"] = "drift"
    with pytest.raises(ValueError, match="reproducibility_checksum mismatch"):
        mod.validate_artifact(checksum_mismatch)

    verdict_mismatch = deepcopy(artifact)
    verdict_mismatch["honest_verdict"] = "ready: wrong"
    with pytest.raises(ValueError, match="honest_verdict mismatch"):
        mod.validate_artifact(verdict_mismatch)

    verifier_false = deepcopy(artifact)
    verifier_false["verifier_is_oracle"] = False
    with pytest.raises(ValueError, match="verifier_is_oracle must be true"):
        mod.validate_artifact(verifier_false)

    assert mod._assignment_constant(mod.ast.parse("x = object()"), "missing") is None
