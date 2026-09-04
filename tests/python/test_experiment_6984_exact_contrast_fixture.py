"""Tests for the source-grouped exact contrast fixture.

Spec refs: REQ-VERIFY-6984 and SCENARIO-VERIFY-6984-*.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from carnot import experiment_6984_exact_contrast_fixture as exp


REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def complete_artifact(tmp_path_factory: pytest.TempPathFactory) -> dict[str, object]:
    """Build the expensive dual-authority fixture once for read-only assertions."""

    root = tmp_path_factory.mktemp("exp6984-complete")
    return exp.build_artifact(REPO_ROOT, root / "raw", duration_s=1.25)


def test_required_fields_have_principles_and_bare_score_contract() -> None:
    """REQ-VERIFY-6984: Every required field has one stated reason."""

    assert set(exp.REQUIRED_ARTIFACT_FIELDS) == set(exp.FIELD_PRINCIPLES)
    assert exp.INFERENCE_SUBSTRATE == "deterministic_z3_contrast_fixture_no_llm"
    assert exp.EXPECTED_PAIR_COUNT == 36


def test_preconditions_require_both_schemas_exact_engines_and_row_storage(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-6984-REPLAY: Preflight checks every frozen dependency."""

    checks = exp.collect_preconditions(REPO_ROOT, tmp_path / "rows")
    assert all(row["passed"] for row in checks)
    missing = exp.collect_preconditions(tmp_path / "missing", tmp_path / "other-rows")
    assert any(row["check"] == "v609_mapping_schema" and not row["passed"] for row in missing)
    assert exp.gate_summary(missing)


def test_unwritable_row_storage_fails_closed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """REQ-VERIFY-6984: A failed exclusive-write probe blocks construction."""

    def fail_probe(*args: object, **kwargs: object) -> tuple[int, str]:
        del args, kwargs
        raise OSError("read only")

    monkeypatch.setattr(exp.tempfile, "mkstemp", fail_probe)
    assert exp._row_storage_writable(tmp_path / "rows") is False


def test_manifest_freezes_36_source_disjoint_groups_without_candidate_labels() -> None:
    """SCENARIO-VERIFY-6984-LEAKAGE: Groups are frozen before label opening."""

    source_pairs = exp.frozen_source_pairs()
    manifest = exp.build_source_manifest(source_pairs)
    assert len(manifest) == 36
    assert exp.sha256_json(manifest) == exp.sha256_json(exp.build_source_manifest(source_pairs))
    counts = {split: sum(row["split"] == split for row in manifest) for split in exp.SPLITS}
    assert counts == {"train": 18, "calibration": 6, "held_out": 12}
    assert len({row["source_group_id"] for row in manifest}) == 36
    assert all("label" not in key and "fault" not in key for row in manifest for key in row)
    assert all(row["manifest_frozen_before_candidate_outcomes"] for row in manifest)


@pytest.mark.parametrize("fault_family", exp.FAULT_FAMILIES)
def test_each_frozen_fault_is_one_changed_semantic_operation(fault_family: str) -> None:
    """SCENARIO-VERIFY-6984-FAULT: A negative contains one declared fault."""

    base = exp.frozen_source_pairs()["0-0-0"]
    negative, mutation = exp.apply_fault(base, fault_family)
    assert mutation["fault_family"] == fault_family
    assert mutation["fault_count"] == 1
    assert mutation["changed"] is True
    assert mutation["before_hash"] != mutation["after_hash"]
    assert negative["mapping"]["claimed_relation"] == "equivalent"


def test_fault_mutator_rejects_unknown_and_ineffective_coefficient_swaps() -> None:
    """SCENARIO-VERIFY-6984-FAULT: A no-op or unknown fault cannot be labeled negative."""

    base = exp.frozen_source_pairs()["0-0-0"]
    with pytest.raises(ValueError, match="unknown_fault_family"):
        exp.apply_fault(base, "unknown")
    equal_coefficients = deepcopy(base)
    terms = equal_coefficients["target"]["objective"]["expression"]["terms"]
    for name in terms:
        terms[name] = "1"
    with pytest.raises(ValueError, match="distinct_coefficients"):
        exp.apply_fault(equal_coefficients, "coefficient_swap")


def test_alpha_renaming_is_injective_label_blind_and_semantics_preserving() -> None:
    """SCENARIO-VERIFY-6984-RENAMING: Renaming preserves the exact relation."""

    base = exp.frozen_source_pairs()["0-0-0"]
    renamed, receipt = exp.alpha_rename_pair(base, "blind-group")
    before = exp.certify_pair(base, "before")
    after = exp.certify_pair(renamed, "after")
    assert receipt["injective"] is True
    assert receipt["label_blind"] is True
    assert before["agreement"]["certified_relation"] == "equivalent"
    assert after["agreement"]["certified_relation"] == "equivalent"
    assert receipt["original_identifiers_present"] is False


def test_serialization_blinds_source_mutation_label_and_authority_fields() -> None:
    """SCENARIO-VERIFY-6984-BLINDING: Feature rows omit oracle provenance."""

    base = exp.frozen_source_pairs()["0-0-0"]
    renamed, _ = exp.alpha_rename_pair(base, "blind-group")
    feature, receipt = exp.blinded_feature_row(
        renamed,
        candidate_id="candidate-0",
        contrast_group_id="contrast-0",
        split="train",
        family="bounded_integer_linear",
        pair_position=0,
    )
    flattened = json.dumps(feature, sort_keys=True)
    assert receipt["forbidden_fields_present"] == []
    assert receipt["canonical_serialization"] is True
    assert set(feature) == set(exp.FEATURE_ROW_FIELDS)
    assert "source_pair_id" not in flattened
    assert "fault_family" not in flattened
    assert "exact_label" not in flattened
    assert "authorities_agree" not in flattened
    assert exp._nested_keys([]) == set()


def test_certifier_rejects_a_mapping_that_fails_the_frozen_v611_parser(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-VERIFY-6984: The V611 syntax boundary is mandatory before certification."""

    monkeypatch.setattr(
        exp.bank_exp,
        "parse_syntax",
        lambda raw: {"constraintir_shape_valid": False, "syntax_reason": "injected"},
    )
    with pytest.raises(ValueError, match="v611_mapping_schema_rejected:injected"):
        exp.certify_pair(exp.frozen_source_pairs()["0-0-0"], "candidate")


def test_solver_disagreement_and_nondecision_reject_the_pair() -> None:
    """SCENARIO-VERIFY-6984-SOLVERS: Disagreement cannot enter the fixture."""

    base = exp.frozen_source_pairs()["0-0-0"]
    certified = exp.certify_pair(base, "candidate")
    disagreement = deepcopy(certified["z3"])
    disagreement["objective_order_preserved"] = not disagreement["objective_order_preserved"]
    agreement = exp.build_authority_agreement("candidate", certified["enumeration"], disagreement)
    assert agreement["objective_order_agreement"] is False
    assert agreement["all_required_agreement"] is False
    assert exp.pair_is_accepted([agreement], [0], ["equivalent"]) is False
    unknown = deepcopy(certified["z3"])
    unknown["status"] = "unknown"
    assert (
        exp.build_authority_agreement("candidate", certified["enumeration"], unknown)[
            "all_required_agreement"
        ]
        is False
    )


def test_complete_artifact_is_balanced_source_disjoint_and_hash_replayable(
    complete_artifact: dict[str, object],
) -> None:
    """REQ-VERIFY-6984: The complete controlled fixture meets every exact gate."""

    artifact = complete_artifact
    assert exp.validate_artifact(artifact, repo_root=REPO_ROOT) == []
    assert artifact["expected_pair_count"] == artifact["observed_pair_count"] == 36
    assert len(artifact["rows"]) == len(artifact["per_candidate_rows"]) == 72
    assert len(artifact["per_pair_results"]) == 36
    assert len(artifact["mutation_attempt_rows"]) == 36
    assert len(artifact["z3_authority_rows"]) == 72
    assert len(artifact["enumeration_authority_rows"]) == 72
    assert all(row["all_required_agreement"] for row in artifact["authority_agreement_rows"])
    assert {
        row["split"]: (row["positive_count"], row["negative_count"])
        for row in artifact["label_balance_rows"]
    } == {
        "train": (18, 18),
        "calibration": (6, 6),
        "held_out": (12, 12),
    }
    assert all(row["overlap_count"] == 0 for row in artifact["source_overlap_rows"])
    held_faults = {
        row["fault_family"]
        for row in artifact["mutation_attempt_rows"]
        if row["split"] == "held_out"
    }
    held_families = {
        row["formulation_family"]
        for row in artifact["per_pair_results"]
        if row["split"] == "held_out"
    }
    assert held_faults == set(exp.FAULT_FAMILIES)
    assert held_families == set(exp.FORMULATION_FAMILIES)
    assert artifact["contrast_fixture_complete_score"] == 1
    assert artifact["label_balance_ready_score"] == 1
    assert type(artifact["contrast_fixture_complete_score"]) is int
    assert type(artifact["label_balance_ready_score"]) is int
    assert artifact["controlled_fixture_only"] is True
    assert artifact["live_extraction_claimed"] is False
    assert artifact["verifier_is_oracle"] is True
    assert artifact["verdict_class"] == "circular_positive"
    assert artifact["honest_verdict"].startswith("complete_")


@pytest.mark.parametrize(
    "mutation, expected_error",
    [
        (lambda value: value.__setitem__("contrast_fixture_complete_score", True), "bare_score"),
        (lambda value: value.__setitem__("label_balance_ready_score", {"value": 1}), "bare_score"),
        (lambda value: value["per_pair_results"].pop(), "pair_count"),
        (
            lambda value: value["source_overlap_rows"][0].__setitem__("overlap_count", 1),
            "source_overlap",
        ),
        (
            lambda value: value["per_candidate_rows"][0].__setitem__("exact_label", "equivalent"),
            "feature_blinding",
        ),
        (lambda value: value.__setitem__("controlled_fixture_only", False), "claim_boundary"),
        (lambda value: value.__setitem__("reproducibility_checksum", "sha256:forged"), "checksum"),
    ],
)
def test_validator_rejects_forged_readiness_and_evidence(
    complete_artifact: dict[str, object],
    mutation: object,
    expected_error: str,
) -> None:
    """SCENARIO-VERIFY-6984-BARE: Reported readiness must replay from rows."""

    artifact = deepcopy(complete_artifact)
    mutation(artifact)  # type: ignore[operator]
    errors = exp.validate_artifact(artifact, repo_root=REPO_ROOT)
    assert any(expected_error in error for error in errors)


def test_blocked_artifact_is_complete_and_names_failed_gate(tmp_path: Path) -> None:
    """REQ-VERIFY-6984: Missing frozen inputs produce the full blocked schema."""

    checks = exp.collect_preconditions(tmp_path / "missing", tmp_path / "rows")
    artifact = exp.build_blocked_artifact(
        repo_root=tmp_path / "missing",
        preconditions=checks,
        duration_s=0.1,
    )
    assert set(exp.REQUIRED_ARTIFACT_FIELDS) <= set(artifact)
    assert artifact["contrast_fixture_complete_score"] == 0
    assert artifact["label_balance_ready_score"] == 0
    assert artifact["verdict_class"] == "blocked"
    assert artifact["honest_verdict"].startswith("blocked_exact_contrast_fixture")
    assert artifact["gate_check_summary"]
    assert all(
        {"check", "expected_value", "observed_value"} <= set(row)
        for row in artifact["gate_check_summary"]
    )
    assert exp.validate_artifact(artifact, repo_root=tmp_path / "missing") == []
    rebuilt = exp.build_artifact(tmp_path / "missing", tmp_path / "other-rows", duration_s=0.2)
    assert rebuilt["verdict_class"] == "blocked"


def test_validator_exercises_every_fail_closed_schema_branch(
    complete_artifact: dict[str, object], tmp_path: Path
) -> None:
    """SCENARIO-VERIFY-6984-BARE: Each independent artifact contract can fail."""

    mutations = [
        (lambda value: value.pop("rows"), "missing_required_fields"),
        (lambda value: value.__setitem__("field_principles", {}), "field_principles"),
        (lambda value: value.__setitem__("verifier_is_oracle", False), "oracle_declaration"),
        (lambda value: value.__setitem__("verdict_class", "unknown"), "verdict_class"),
        (lambda value: value.__setitem__("inference_substrate", "other"), "inference_substrate"),
        (lambda value: value.__setitem__("expected_pair_count", 0), "expected_pair_count"),
        (lambda value: value["rows"].pop(), "candidate_count"),
        (
            lambda value: value["source_manifest_rows"][0].__setitem__("split", "other"),
            "source_manifest_hash",
        ),
        (lambda value: value.__setitem__("split_hashes", {}), "split_hashes"),
    ]
    for mutate, expected in mutations:
        artifact = deepcopy(complete_artifact)
        mutate(artifact)
        assert any(
            expected in error for error in exp.validate_artifact(artifact, repo_root=REPO_ROOT)
        )

    checks = exp.collect_preconditions(tmp_path / "missing", tmp_path / "rows")
    blocked = exp.build_blocked_artifact(
        repo_root=tmp_path / "missing", preconditions=checks, duration_s=0.1
    )
    blocked_mutations = [
        (lambda value: value.__setitem__("contrast_fixture_complete_score", 1), "blocked_score"),
        (lambda value: value.__setitem__("gate_check_summary", []), "blocked_gate_summary"),
        (lambda value: value.__setitem__("honest_verdict", "blocked_other"), "blocked_verdict"),
    ]
    for mutate, expected in blocked_mutations:
        artifact = deepcopy(blocked)
        mutate(artifact)
        assert any(
            expected in error for error in exp.validate_artifact(artifact, repo_root=tmp_path)
        )


def test_immutable_writer_replays_identical_bytes_and_rejects_drift(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-6984-REPLAY: Immutable row paths reject changed bytes."""

    path = tmp_path / "row.json"
    first = exp.write_immutable_json(path, {"value": 1})
    second = exp.write_immutable_json(path, {"value": 1})
    assert first == second == exp.sha256_path(path)
    with pytest.raises(exp.ImmutableRowError, match="immutable_row_mismatch"):
        exp.write_immutable_json(path, {"value": 2})


def test_run_writes_one_valid_stable_artifact(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-6984-REPLAY: The command surface writes validated JSON."""

    result_path = tmp_path / "experiment_6984.json"
    artifact = exp.run(
        repo_root=REPO_ROOT,
        result_path=result_path,
        row_root=tmp_path / "raw",
        run_date="20260904",
    )
    assert result_path.is_file()
    assert json.loads(result_path.read_text()) == artifact
    assert exp.validate_artifact(artifact, repo_root=REPO_ROOT) == []


def test_run_rejects_wrong_date_and_internal_validation_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """REQ-VERIFY-6984: The command refuses an unfrozen date or invalid aggregate."""

    with pytest.raises(ValueError, match="run_date_mismatch"):
        exp.run(
            repo_root=REPO_ROOT,
            result_path=tmp_path / "wrong-date.json",
            row_root=tmp_path / "wrong-date-rows",
            run_date="20260905",
        )
    monkeypatch.setattr(exp, "build_artifact", lambda *args, **kwargs: {"duration_s": 0.0})
    monkeypatch.setattr(exp, "payload_checksum", lambda artifact: "sha256:test")
    monkeypatch.setattr(exp, "validate_artifact", lambda *args, **kwargs: ["injected"])
    with pytest.raises(ValueError, match="artifact_validation_failed:injected"):
        exp.run(
            repo_root=REPO_ROOT,
            result_path=tmp_path / "invalid.json",
            row_root=tmp_path / "invalid-rows",
            run_date="20260904",
        )
