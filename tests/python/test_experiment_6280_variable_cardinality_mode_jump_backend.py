"""Tests for Exp6280 variable-cardinality mode-jump backend ABI.

Spec refs: REQ-SAMPLER-6280,
SCENARIO-SAMPLER-6280-METADATA-ROUNDTRIP,
SCENARIO-SAMPLER-6280-PROPOSAL-PARITY,
SCENARIO-SAMPLER-6280-NO-AB-VALUE-CLAIM.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from carnot import experiment_6280_variable_cardinality_mode_jump_backend as mod


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec/capabilities/samplers/spec.md"


def _passing_exit_codes() -> dict[str, int]:
    return {command: 0 for command in mod.DEFAULT_TEST_COMMANDS}


@pytest.fixture(scope="module")
def artifact() -> dict[str, object]:
    return mod.build_artifact(
        root=REPO,
        run_date="20260810",
        duration_s=0.0,
        test_exit_codes=_passing_exit_codes(),
    )


def test_req_sampler_6280_spec_declares_required_fields_and_principles() -> None:
    """REQ-SAMPLER-6280: OpenSpec anchors each required artifact field."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    section = spec[spec.index("### REQ-SAMPLER-6280") :]
    normalized = " ".join(section.split())

    for marker in (
        "REQ-SAMPLER-6280-SPEC-FIRST",
        "REQ-SAMPLER-6280-METADATA",
        "REQ-SAMPLER-6280-COMPATIBILITY",
        "REQ-SAMPLER-6280-PARITY",
        "REQ-SAMPLER-6280-ACTIVATION",
        "REQ-SAMPLER-6280-CONTROLS",
        mod.RESULT_RELATIVE_PATH.as_posix(),
        mod.MODULE_RELATIVE_PATH.as_posix(),
        mod.INFERENCE_SUBSTRATE,
    ):
        assert marker in section

    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert f"`{field}`" in section
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert f"`{field}`" in section
        assert " ".join(principle.split()) in normalized


def test_req_sampler_6280_artifact_schema_and_ready_gates(
    artifact: dict[str, object],
    tmp_path: Path,
) -> None:
    """REQ-SAMPLER-6280-ARTIFACT: terminal JSON validates without A/B claims."""

    output = tmp_path / mod.RESULT_RELATIVE_PATH.name
    written = mod.write_artifact(
        output_path=output,
        root=REPO,
        run_date="20260810",
        duration_s=0.0,
        test_exit_codes=_passing_exit_codes(),
    )
    loaded = json.loads(output.read_text(encoding="utf-8"))

    assert loaded == written
    assert written == artifact
    assert set(mod.REQUIRED_ARTIFACT_FIELDS) <= set(written)
    assert mod.validate_artifact(written) is True
    assert written["status"] == "complete_ready"
    assert written["honest_verdict"].startswith("complete_ready:")
    assert written["variable_cardinality_backend_ready_score"] == 1.0
    assert written["source_mutation_count"] == 0
    assert type(written["source_mutation_count"]) is int
    assert written["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert written["verifier_is_oracle"]["value"] is True
    assert written["exp6269_failure_path_hash_and_root_cause"]["scientific_ab_rerun"] is False


def test_scenario_sampler_6280_all_exp6268_families_have_parity_and_activation(
    artifact: dict[str, object],
) -> None:
    """SCENARIO-SAMPLER-6280-PROPOSAL-PARITY: all fixture families are supported."""

    supported = artifact["supported_fixture_families_and_shapes"]
    roundtrips = artifact["rust_python_encode_decode_roundtrip_by_fixture"]["fixtures"]
    proposals = artifact["rust_python_proposal_parity_by_fixture"]["fixtures"]
    replay = artifact["deterministic_seed_replay_by_fixture"]["fixtures"]
    treatment = artifact["treatment_attempt_accept_and_fire_counts_by_fixture"]["fixtures"]

    assert supported["all_preregistered_families_supported"] is True
    assert supported["fixture_count"] == 8
    assert supported["families"]["ising_multimodal"]["fixture_count"] == 2
    assert supported["families"]["potts"]["fixture_count"] == 2
    assert supported["families"]["typed_factor"]["fixture_count"] == 2
    assert supported["families"]["original_six_state_positive_control"]["fixture_count"] == 1
    assert supported["families"]["unimodal_control"]["fixture_count"] == 1

    for fixture in supported["fixtures"]:
        name = fixture["fixture_name"]
        assert roundtrips[name]["passed"] is True
        assert proposals[name]["passed"] is True
        assert replay[name]["passed"] is True
        assert treatment[name]["treatment_attempt_count"] > 0
        assert treatment[name]["treatment_accept_count"] > 0
        assert treatment[name]["treatment_fire_count"] > 0

    assert artifact["original_six_state_regression_receipt"]["passed"] is True
    assert artifact["malformed_cardinality_controls"]["all_fail_closed"] is True
    assert artifact["malformed_shape_controls"]["all_fail_closed"] is True
    assert artifact["out_of_domain_proposal_controls"]["all_fail_closed"] is True
    assert artifact["label_permutation_controls"]["all_fail_closed"] is True
    assert artifact["unsupported_shapes_and_fail_closed_behavior"]["fallback_output_substituted"] is False


def test_req_sampler_6280_validation_rejects_gate_and_control_mutations(
    artifact: dict[str, object],
) -> None:
    """REQ-SAMPLER-6280-CONTROLS: schema and gate mutations fail closed."""

    mutations = [
        ("missing required", lambda data: data.pop("status")),
        ("field_principles", lambda data: data.__setitem__("field_principles", {})),
        (
            "field_provenance:status",
            lambda data: data["field_provenance"]["status"].__setitem__("principle", "bad"),
        ),
        ("source_mutation_count", lambda data: data.__setitem__("source_mutation_count", 1)),
        (
            "supported_fixture_families_and_shapes",
            lambda data: data["supported_fixture_families_and_shapes"].__setitem__(
                "all_preregistered_families_supported",
                False,
            ),
        ),
        (
            "original_six_state_regression_receipt",
            lambda data: data["original_six_state_regression_receipt"].__setitem__(
                "passed",
                False,
            ),
        ),
        (
            "malformed_cardinality_controls",
            lambda data: data["malformed_cardinality_controls"].__setitem__(
                "all_fail_closed",
                False,
            ),
        ),
        (
            "deterministic_seed_replay_by_fixture",
            lambda data: data["deterministic_seed_replay_by_fixture"]["fixtures"][
                "potts_chain3_q3"
            ].__setitem__("passed", False),
        ),
        (
            "rust_python_proposal_parity_by_fixture",
            lambda data: data["rust_python_proposal_parity_by_fixture"].__setitem__(
                "all_passed",
                False,
            ),
        ),
        (
            "treatment_attempt_accept_and_fire_counts_by_fixture",
            lambda data: data[
                "treatment_attempt_accept_and_fire_counts_by_fixture"
            ].__setitem__("activation_proven_before_readiness", False),
        ),
        (
            "inference_substrate",
            lambda data: data.__setitem__("inference_substrate", "scientific_ab"),
        ),
        (
            "verifier_is_oracle",
            lambda data: data["verifier_is_oracle"].__setitem__("value", False),
        ),
        (
            "variable_cardinality_backend_ready_score",
            lambda data: data.__setitem__("variable_cardinality_backend_ready_score", 0.0),
        ),
        ("status", lambda data: data.__setitem__("status", "blocked")),
        ("honest_verdict", lambda data: data.__setitem__("honest_verdict", "blocked")),
        (
            "reproducibility_checksum",
            lambda data: data.__setitem__("reproducibility_checksum", "sha256:bad"),
        ),
    ]

    for expected, mutate in mutations:
        bad = deepcopy(artifact)
        mutate(bad)
        with pytest.raises(ValueError, match=expected):
            mod.validate_artifact(bad)


def test_req_sampler_6280_cli_writes_tmp_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """SCENARIO-SAMPLER-6280-NO-AB-VALUE-CLAIM: CLI writes the ABI artifact."""

    receipts = tmp_path / "receipts.json"
    receipts.write_text(json.dumps(_passing_exit_codes()), encoding="utf-8")
    output = tmp_path / "artifact.json"
    monkeypatch.setenv("CARNOT_6280_COMMAND_RECEIPTS", str(receipts))

    assert mod.main(["--date", "20260810", "--output", str(output)]) == 0
    written = json.loads(output.read_text(encoding="utf-8"))
    assert written["status"] == "complete_ready"
    assert written["exp6269_failure_path_hash_and_root_cause"]["scientific_ab_rerun"] is False
    assert "complete_ready" in capsys.readouterr().out


def test_req_sampler_6280_auxiliary_negative_branches(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-SAMPLER-6280-CONTROLS: helper negative branches stay explicit."""

    assert mod._run_text(["/definitely/missing/command"], REPO)["available"] is False

    non_object = tmp_path / "list.json"
    non_object.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="JSON object"):
        mod._read_json(non_object)

    assert mod._control_result("accepted", lambda: None)["fail_closed"] is False
    assert mod.honest_verdict({"status": "blocked"}) == (
        "blocked: variable-cardinality mode-jump ABI readiness gates failed"
    )

    monkeypatch.delenv("CARNOT_6280_COMMAND_RECEIPTS", raising=False)
    missing_default = tmp_path / "missing_receipts.json"
    monkeypatch.setattr(mod, "DEFAULT_RECEIPT_PATH", missing_default)
    assert mod._external_test_exit_codes() == {}

    default_receipts = tmp_path / "default_receipts.json"
    default_receipts.write_text(json.dumps({"cmd": 0}), encoding="utf-8")
    monkeypatch.setattr(mod, "DEFAULT_RECEIPT_PATH", default_receipts)
    assert mod._external_test_exit_codes() == {"cmd": 0}

    monkeypatch.setenv("CARNOT_6280_COMMAND_RECEIPTS", str(non_object))
    with pytest.raises(ValueError, match="command receipt payload"):
        mod._external_test_exit_codes()

    import carnot._rust as rust_module

    class MismatchingMetadata:
        def __init__(self, *_args: object) -> None:
            pass

        def encode_label(self, _label: str) -> int:
            return 99

        def decode_index(self, _index: int) -> str:
            return "wrong"

        def state_value(self, _label: str) -> list[int]:
            return [1]

    monkeypatch.setattr(rust_module, "RustModeJumpStateMetadata", MismatchingMetadata)
    metadata = {
        "schema": mod.TYPED_STATE_METADATA_SCHEMA_VERSION,
        "shape": [1],
        "cardinalities": [2],
        "encoding": "categorical_label_rank1",
        "state_labels": ["a"],
        "state_values": [[0]],
        "proposal_domain": "explicit_support_table",
        "state_space_size": 2,
    }
    mismatched = mod.rust_python_encode_decode_roundtrip_by_fixture(
        [
            {
                "fixture": "fake",
                "labels": ["a"],
                "metadata": metadata,
            }
        ]
    )
    assert mismatched["all_passed"] is False
    assert mismatched["fixtures"]["fake"]["mismatches"] == [
        "encode:a",
        "decode:0",
        "value:a",
    ]
