"""Tests for Exp6268 frozen exact sampler fixture suite.

Spec refs: REQ-SAMPLER-6268, SCENARIO-SAMPLER-6268-EXACT-SUITE,
SCENARIO-SAMPLER-6268-CONTROLS-FAIL-CLOSED,
SCENARIO-SAMPLER-6268-NO-PERFORMANCE-CLAIM.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from carnot import experiment_6268_multimodal_sampler_fixture_suite as mod


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec/capabilities/samplers/spec.md"


def _passing_exit_codes() -> dict[str, int]:
    return {command: 0 for command in mod.DEFAULT_TEST_COMMANDS}


def test_req_sampler_6268_spec_declares_fields_and_scenarios() -> None:
    """REQ-SAMPLER-6268: OpenSpec anchors the fixture-suite contract."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    section = spec[spec.index("### REQ-SAMPLER-6268") :]
    normalized = " ".join(section.split())

    for marker in (
        "REQ-SAMPLER-6268-PRECONDITIONS",
        "REQ-SAMPLER-6268-FAMILIES",
        "REQ-SAMPLER-6268-EXACT",
        "REQ-SAMPLER-6268-CONTROLS",
        "REQ-SAMPLER-6268-NEGATIVE-TESTS",
        "REQ-SAMPLER-6268-READY-GATE",
        "SCENARIO-SAMPLER-6268-EXACT-SUITE",
        "SCENARIO-SAMPLER-6268-CONTROLS-FAIL-CLOSED",
        "SCENARIO-SAMPLER-6268-NO-PERFORMANCE-CLAIM",
        mod.RESULT_RELATIVE_PATH.as_posix(),
        mod.MODULE_RELATIVE_PATH.as_posix(),
        mod.TEST_RELATIVE_PATH.as_posix(),
        mod.INFERENCE_SUBSTRATE,
    ):
        assert marker in section

    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert f"`{field}`" in section
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert f"`{field}`" in section
        assert " ".join(principle.split()) in normalized


def test_scenario_sampler_6268_artifact_schema_and_family_gates(tmp_path: Path) -> None:
    """SCENARIO-SAMPLER-6268-EXACT-SUITE: all required families are present."""

    output = tmp_path / mod.RESULT_RELATIVE_PATH.name
    manifest = tmp_path / mod.FIXTURE_MANIFEST_RELATIVE_PATH.name
    artifact = mod.write_artifact(
        output_path=output,
        manifest_path=manifest,
        root=REPO,
        run_date="20260810",
        duration_s=0.0,
        test_exit_codes=_passing_exit_codes(),
    )
    loaded = json.loads(output.read_text(encoding="utf-8"))

    assert loaded == artifact
    assert set(mod.REQUIRED_ARTIFACT_FIELDS) <= set(artifact)
    assert mod.validate_artifact(artifact) is True
    assert artifact["status"] == "complete_ready"
    assert artifact["honest_verdict"].startswith("complete_ready:")
    assert artifact["sampler_fixture_suite_ready_score"] == 1.0
    assert artifact["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert artifact["verifier_is_oracle"]["value"] is True
    assert type(artifact["duplicate_fixture_count"]) is int
    assert artifact["duplicate_fixture_count"] == 0
    assert type(artifact["source_mutation_count"]) is int
    assert artifact["source_mutation_count"] == 0

    counts = artifact["fixture_family_counts"]
    assert counts["original_six_state_positive_control"] == 1
    assert counts["ising_multimodal"] >= 2
    assert counts["potts"] >= 2
    assert counts["typed_factor"] >= 2
    assert counts["all_preregistered_families_present"] is True
    assert artifact["fixture_manifest_path_and_hash"]["path"] == manifest.as_posix()
    assert artifact["fixture_manifest_path_and_hash"]["sha256"] == mod.sha256_file(manifest)


def test_scenario_sampler_6268_exact_probabilities_basins_and_controls() -> None:
    """REQ-SAMPLER-6268-EXACT: enumeration owns probabilities and controls."""

    artifact = mod.build_artifact(
        root=REPO,
        run_date="20260810",
        duration_s=0.0,
        test_exit_codes=_passing_exit_codes(),
    )

    assert len(artifact["exact_enumeration_receipts"]) >= 8
    assert artifact["original_six_state_positive_control"]["passed"] is True
    assert artifact["original_six_state_positive_control"]["reproduces_exp6237_fixture"] is True
    assert artifact["unimodal_control"]["valid_unimodal_control"] is True
    assert artifact["unimodal_control"]["multimodal_claim_allowed"] is False
    assert artifact["inactive_treatment_control"]["valid_inactive_control"] is True
    assert artifact["inactive_treatment_control"]["null_sampler_verdict_allowed"] is False
    assert artifact["unsupported_shape_control"]["valid_unsupported_shape_control"] is True

    hashes = artifact["normalized_target_probability_hashes"]
    barriers = artifact["basin_labels_and_barrier_metadata"]
    normalization_errors = artifact["exact_probability_normalization_error_by_fixture"]
    support = artifact["mode_jump_support_by_fixture"]

    assert support["exp6237_original_six_state"]["mode_jump_rust_supported"] is True
    assert any(not row["mode_jump_rust_supported"] for row in support.values())
    assert max(normalization_errors.values()) <= mod.EXACT_TOLERANCE

    for receipt in artifact["exact_enumeration_receipts"]:
        name = receipt["fixture_name"]
        assert mod.validate_fixture_receipt(receipt) is True
        assert hashes[name] == mod.normalized_target_probability_hash(receipt)
        assert name in barriers
        assert barriers[name]["basin_count"] >= 1
        assert "barrier_pairs" in barriers[name]
        assert sum(row["probability"] for row in receipt["support"]) == pytest.approx(1.0)
        assert artifact["state_space_sizes"][name] >= receipt["support_count"]


def test_req_sampler_6268_label_permutation_duplicates_and_energy_sign_errors() -> None:
    """REQ-SAMPLER-6268-NEGATIVE-TESTS: subtle fixture mutations fail closed."""

    artifact = mod.build_artifact(
        root=REPO,
        run_date="20260810",
        duration_s=0.0,
        test_exit_codes=_passing_exit_codes(),
    )
    receipt = deepcopy(artifact["exact_enumeration_receipts"][1])

    permuted = deepcopy(receipt)
    permuted["support"] = list(reversed(permuted["support"]))
    assert mod.normalized_target_probability_hash(
        permuted
    ) == mod.normalized_target_probability_hash(receipt)

    duplicated = artifact["exact_enumeration_receipts"] + [
        deepcopy(artifact["exact_enumeration_receipts"][0])
    ]
    assert mod.duplicate_fixture_count(duplicated) == 1

    bad_energy = deepcopy(receipt)
    bad_energy["support"][0]["energy"] = -float(bad_energy["support"][0]["energy"])
    with pytest.raises(ValueError, match="energy_probability_consistency"):
        mod.validate_fixture_receipt(bad_energy)

    bad_normalization = deepcopy(receipt)
    bad_normalization["support"][0]["probability"] += 0.1
    with pytest.raises(ValueError, match="normalization_error"):
        mod.validate_fixture_receipt(bad_normalization)

    bad_receipt_hash = deepcopy(receipt)
    bad_receipt_hash["target_probability_hash"] = "sha256:bad"
    with pytest.raises(ValueError, match="normalized_target_probability_hashes"):
        mod.validate_fixture_receipt(bad_receipt_hash)

    bad_hash = deepcopy(artifact)
    bad_hash["normalized_target_probability_hashes"][receipt["fixture_name"]] = "sha256:bad"
    bad_hash["sampler_fixture_suite_ready_score"] = mod.sampler_fixture_suite_ready_score(bad_hash)
    bad_hash["status"] = mod.status(bad_hash)
    bad_hash["honest_verdict"] = mod.honest_verdict(bad_hash)
    bad_hash["reproducibility_checksum"] = mod.reproducibility_checksum(bad_hash)
    with pytest.raises(ValueError, match="normalized_target_probability_hashes"):
        mod.validate_artifact(bad_hash)


def test_req_sampler_6268_potts_cardinality_and_typed_factor_arity_controls() -> None:
    """REQ-SAMPLER-6268-NEGATIVE-TESTS: Potts and typed-factor shapes are checked."""

    fixtures = {fixture["fixture_name"]: fixture for fixture in mod.build_fixture_suite(REPO)}
    potts = deepcopy(fixtures["potts_chain3_q3"])
    potts["definition"]["q_states"] = 2
    with pytest.raises(ValueError, match="Potts cardinality"):
        mod.enumerate_fixture(potts)

    typed = deepcopy(fixtures["typed_access_control_exp6152"])
    typed["definition"]["expected_max_kernel_arity"] = 99
    receipt = mod.enumerate_fixture(typed)
    with pytest.raises(ValueError, match="typed_factor_arity"):
        mod.validate_fixture_receipt(receipt)

    original_potts = mod.enumerate_fixture(fixtures["potts_chain3_q3"])
    support = mod.mode_jump_support_by_fixture([original_potts])
    assert support["potts_chain3_q3"]["mode_jump_rust_supported"] is False


def test_req_sampler_6268_schema_mutations_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """SCENARIO-SAMPLER-6268-CONTROLS-FAIL-CLOSED: readiness is mechanical."""

    artifact = mod.build_artifact(
        root=REPO,
        run_date="20260810",
        duration_s=0.0,
        test_exit_codes=_passing_exit_codes(),
    )
    mutations = [
        ("duplicate_fixture_count", lambda data: data.__setitem__("duplicate_fixture_count", True)),
        ("duplicate_fixture_count", lambda data: data.__setitem__("duplicate_fixture_count", 1)),
        ("source_mutation_count", lambda data: data.__setitem__("source_mutation_count", 1)),
        (
            "unimodal_control",
            lambda data: data["unimodal_control"].__setitem__("valid_unimodal_control", False),
        ),
        (
            "inactive_treatment_control",
            lambda data: data["inactive_treatment_control"].__setitem__(
                "null_sampler_verdict_allowed", True
            ),
        ),
        (
            "unsupported_shape_control",
            lambda data: data["unsupported_shape_control"].__setitem__(
                "valid_unsupported_shape_control", False
            ),
        ),
        (
            "exact_probability_normalization_error_by_fixture",
            lambda data: data["exact_probability_normalization_error_by_fixture"].__setitem__(
                "exp6237_original_six_state", 1.0
            ),
        ),
        (
            "inference_substrate",
            lambda data: data.__setitem__("inference_substrate", "cuda_timing"),
        ),
        ("field_principles", lambda data: data["field_principles"].__setitem__("status", "wrong")),
        (
            "field_provenance:status",
            lambda data: data["field_provenance"]["status"].__setitem__("principle", "wrong"),
        ),
    ]

    for expected, mutate in mutations:
        bad = deepcopy(artifact)
        mutate(bad)
        bad["sampler_fixture_suite_ready_score"] = mod.sampler_fixture_suite_ready_score(bad)
        bad["status"] = mod.status(bad)
        bad["honest_verdict"] = mod.honest_verdict(bad)
        bad["reproducibility_checksum"] = mod.reproducibility_checksum(bad)
        with pytest.raises(ValueError, match=expected):
            mod.validate_artifact(bad)

    missing = deepcopy(artifact)
    del missing["status"]
    with pytest.raises(ValueError, match="missing required"):
        mod.validate_artifact(missing)

    checksum = deepcopy(artifact)
    checksum["reproducibility_checksum"] = "sha256:bad"
    with pytest.raises(ValueError, match="reproducibility_checksum"):
        mod.validate_artifact(checksum)

    family_bad = deepcopy(artifact)
    family_bad["fixture_family_counts"]["all_preregistered_families_present"] = False
    protected_bad = deepcopy(artifact)
    protected_bad["protected_files_unchanged"]["unchanged"] = False
    preconditions_bad = deepcopy(artifact)
    preconditions_bad["preconditions_checked"]["preconditions_ready"] = False
    command_bad = deepcopy(artifact)
    command_bad["test_exit_codes"][mod.DEFAULT_TEST_COMMANDS[0]] = 7

    assert "fixture_families" in mod.blocked_reasons(family_bad)
    assert "protected_files" in mod.blocked_reasons(protected_bad)
    assert "preconditions" in mod.blocked_reasons(preconditions_bad)
    assert "test_commands" in mod.blocked_reasons(command_bad)

    receipts = tmp_path / "receipts.json"
    receipts.write_text(json.dumps(_passing_exit_codes()), encoding="utf-8")
    monkeypatch.setenv("CARNOT_6268_COMMAND_RECEIPTS", str(receipts))
    assert mod._external_test_exit_codes() == _passing_exit_codes()  # noqa: SLF001

    missing_receipts = tmp_path / "missing.json"
    monkeypatch.delenv("CARNOT_6268_COMMAND_RECEIPTS", raising=False)
    monkeypatch.setattr(mod, "DEFAULT_RECEIPT_PATH", missing_receipts)
    assert mod._external_test_exit_codes() == {}  # noqa: SLF001

    bad_receipts = tmp_path / "bad.json"
    bad_receipts.write_text("[]", encoding="utf-8")
    monkeypatch.setenv("CARNOT_6268_COMMAND_RECEIPTS", str(bad_receipts))
    with pytest.raises(ValueError, match="command receipt payload"):
        mod._external_test_exit_codes()  # noqa: SLF001

    output = tmp_path / "artifact.json"
    manifest = tmp_path / "manifest.json"
    monkeypatch.setenv("CARNOT_6268_COMMAND_RECEIPTS", str(receipts))
    monkeypatch.setattr(mod, "DEFAULT_RECEIPT_PATH", missing_receipts)
    assert (
        mod.main(
            [
                "--date",
                "20260810",
                "--output",
                str(output),
                "--manifest-output",
                str(manifest),
            ]
        )
        == 0
    )
    assert output.exists()
    assert manifest.exists()
    assert "complete_ready" in capsys.readouterr().out
