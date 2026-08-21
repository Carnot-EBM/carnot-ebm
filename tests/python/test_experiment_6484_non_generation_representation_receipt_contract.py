"""Tests for Exp6484 non-generation representation receipts.

Spec refs: REQ-INFRA-6484, SCENARIO-INFRA-6484-COMMITMENT,
SCENARIO-INFRA-6484-PERSISTENCE, SCENARIO-INFRA-6484-NO-GENERATION,
SCENARIO-INFRA-6484-FAMILY-SEPARATION, SCENARIO-INFRA-6484-ATTACKS,
SCENARIO-INFRA-6484-ARTIFACT.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from carnot import experiment_6484_non_generation_representation_receipt_contract as mod
import scripts.adversarial_verify as av


REPO = Path(__file__).resolve().parents[2]
SPEC = REPO / mod.SPEC_RELATIVE_PATH


def _contract() -> dict:
    return mod.build_contract_rows(root=REPO)


def _validate(contract: dict) -> dict:
    return mod.validate_contract_rows(
        contract["rows"],
        fixture_manifest=contract["fixture_manifest"],
        transform_manifest=contract["transform_manifest"],
    )


def _with_checksum(artifact: dict) -> dict:
    artifact["reproducibility_checksum"] = mod.payload_checksum(artifact)
    return artifact


def test_req_infra_6484_spec_declares_contract_fields_and_scenarios() -> None:
    """REQ-INFRA-6484: OpenSpec owns the representation receipt contract."""

    text = SPEC.read_text(encoding="utf-8")
    section = text[text.index("REQ-INFRA-6484") : text.index("REQ-INFRA-6351")]
    for marker in (
        "SCENARIO-INFRA-6484-COMMITMENT",
        "SCENARIO-INFRA-6484-PERSISTENCE",
        "SCENARIO-INFRA-6484-NO-GENERATION",
        "SCENARIO-INFRA-6484-FAMILY-SEPARATION",
        "SCENARIO-INFRA-6484-ATTACKS",
        "SCENARIO-INFRA-6484-ARTIFACT",
        mod.MODULE_RELATIVE_PATH.as_posix(),
        mod.RESULT_RELATIVE_PATH.as_posix(),
        mod.INFERENCE_SUBSTRATE,
    ):
        assert marker in section
    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert f"`{field}`" in section
        assert mod.FIELD_PRINCIPLES[field] in section


def test_scenario_infra_6484_commitment_persistence_and_family_separation() -> None:
    """SCENARIO-INFRA-6484-COMMITMENT/PERSISTENCE/FAMILY-SEPARATION."""

    contract = _contract()
    report = _validate(contract)
    rows = contract["rows"]
    raw_rows = [row for row in rows if row["row_type"] == "raw_vector_persistence"]
    phase_rows = [row for row in rows if row["row_type"] == "phase"]
    transform_rows = [row for row in rows if row["row_type"] == "transform_binding"]
    family_rows = [row for row in rows if row["row_type"] == "family_separation"]

    assert report["accepted"] is True
    assert report["reasons"] == []
    assert report["row_type_counts"]["candidate_commitment"] == len(mod.FIXTURE_SPECS)
    assert report["row_type_counts"]["raw_vector_persistence"] == (
        len(mod.FIXTURE_SPECS) * len(mod.FAMILY_SPECS)
    )
    assert report["row_type_counts"]["phase"] == (
        len(mod.FIXTURE_SPECS) * len(mod.FAMILY_SPECS) * len(mod.PHASES)
    )
    assert {row["native_dimension"] for row in raw_rows} == {2048, 2816, 5376}
    assert {tuple(row["native_dimensions_seen"]) for row in family_rows} == {
        (2048,),
        (2816,),
        (5376,),
    }
    assert all(row["write_count"] == 1 for row in raw_rows)
    assert all(row["label_read_monotonic_ns"] > row["raw_persist_end_ns"] for row in raw_rows)
    assert all(row["monotonic_start_ns"] < row["monotonic_end_ns"] for row in phase_rows)
    assert all(
        row["transform_manifest_hash"] == contract["transform_manifest"]["manifest_hash"]
        for row in transform_rows
    )
    assert all(not row["pooled_with_families"] for row in family_rows)


def test_scenario_infra_6484_no_generation_and_attack_matrix() -> None:
    """SCENARIO-INFRA-6484-NO-GENERATION/ATTACKS: attacks fail closed."""

    contract = _contract()
    matrix = mod.mutation_attack_matrix(
        contract["rows"],
        fixture_manifest=contract["fixture_manifest"],
        transform_manifest=contract["transform_manifest"],
    )
    by_id = {row["attack_id"]: row for row in matrix["rows"]}

    assert set(by_id) == set(mod.ATTACK_IDS)
    assert matrix["all_critical_fail_closed"] is True
    assert matrix["false_accept_count"] == 0
    expected_reason = {
        "generation_api_call": "generation_api_called",
        "post_load_candidate_edit": "post_load_candidate_edit",
        "duplicate_vector_write": "duplicate_vector_write",
        "label_read_before_persistence": "label_read_before_raw_persistence",
        "pooled_family_vectors": "family_pooling_detected",
        "dimension_identity": "dimension_identity_shortcut",
        "norm_only_signal": "norm_only_shortcut",
        "length_only_signal": "length_only_shortcut",
        "pair_permutation": "pair_permutation_detected",
        "claim_flip": "claim_flip_detected",
    }
    for attack_id, reason in expected_reason.items():
        assert by_id[attack_id]["fail_closed"] is True
        assert reason in by_id[attack_id]["reasons"]

    with pytest.raises(ValueError, match="unknown attack_id"):
        mod.mutate_rows_for_attack("unknown", contract["rows"])


def test_req_infra_6484_validator_defensive_edges(tmp_path: Path) -> None:
    """REQ-INFRA-6484: malformed receipt edges fail closed."""

    contract = _contract()

    bad = deepcopy(contract["rows"])
    bad[0]["candidate_text"] = "tamper without row hash refresh"
    report = mod.validate_contract_rows(
        bad,
        fixture_manifest=contract["fixture_manifest"],
        transform_manifest=contract["transform_manifest"],
    )
    assert "row_hash_mismatch" in report["reasons"]

    bad = [row for row in deepcopy(contract["rows"]) if row["row_type"] != "candidate_commitment"]
    report = mod.validate_contract_rows(
        bad,
        fixture_manifest=contract["fixture_manifest"],
        transform_manifest=contract["transform_manifest"],
    )
    assert "candidate_commitment_count" in report["reasons"]

    bad = deepcopy(contract["rows"])
    candidate = next(row for row in bad if row["row_type"] == "candidate_commitment")
    candidate["fixture_id"] = "unknown-fixture"
    mod._refresh_row(candidate)
    report = mod.validate_contract_rows(
        bad,
        fixture_manifest=contract["fixture_manifest"],
        transform_manifest=contract["transform_manifest"],
    )
    assert "unknown_fixture" in report["reasons"]

    bad = deepcopy(contract["rows"])
    candidate = next(row for row in bad if row["row_type"] == "candidate_commitment")
    candidate["pre_model_commitment_ns"] = candidate["model_access_start_ns"]
    candidate["prompt_hash"] = "sha256:" + "1" * 64
    candidate["pair_position"] = "candidate-z"
    candidate["claim_commitment_hash"] = "sha256:" + "2" * 64
    mod._refresh_row(candidate)
    report = mod.validate_contract_rows(
        bad,
        fixture_manifest=contract["fixture_manifest"],
        transform_manifest=contract["transform_manifest"],
    )
    assert {
        "candidate_not_committed_before_model_access",
        "prompt_hash_mismatch",
        "pair_permutation_detected",
        "claim_flip_detected",
    } <= set(report["reasons"])

    bad = deepcopy(contract["rows"])
    phase = next(row for row in bad if row["row_type"] == "phase")
    phase["monotonic_end_ns"] = phase["monotonic_start_ns"]
    mod._refresh_row(phase)
    report = mod.validate_contract_rows(
        bad,
        fixture_manifest=contract["fixture_manifest"],
        transform_manifest=contract["transform_manifest"],
    )
    assert "phase_interval_invalid" in report["reasons"]

    bad = deepcopy(contract["rows"])
    phase = next(row for row in bad if row["row_type"] == "phase")
    bad.remove(phase)
    report = mod.validate_contract_rows(
        bad,
        fixture_manifest=contract["fixture_manifest"],
        transform_manifest=contract["transform_manifest"],
    )
    assert "phase_order_or_count_mismatch" in report["reasons"]

    bad = [row for row in deepcopy(contract["rows"]) if row["row_type"] != "raw_vector_persistence"]
    report = mod.validate_contract_rows(
        bad,
        fixture_manifest=contract["fixture_manifest"],
        transform_manifest=contract["transform_manifest"],
    )
    assert {"raw_vector_row_count", "raw_vector_cell_count"} <= set(report["reasons"])

    bad = deepcopy(contract["rows"])
    raw = next(row for row in bad if row["row_type"] == "raw_vector_persistence")
    raw["fixture_id"] = "unknown-fixture"
    mod._refresh_row(raw)
    report = mod.validate_contract_rows(
        bad,
        fixture_manifest=contract["fixture_manifest"],
        transform_manifest=contract["transform_manifest"],
    )
    assert "unknown_raw_fixture_or_family" in report["reasons"]

    bad = deepcopy(contract["rows"])
    raw = next(row for row in bad if row["row_type"] == "raw_vector_persistence")
    raw["raw_vector"] = raw["raw_vector"][:-1]
    raw["native_dimension"] = 999
    raw["vector_hash"] = "sha256:" + "3" * 64
    raw["model_hash"] = "sha256:" + "4" * 64
    raw["candidate_hash"] = "sha256:" + "5" * 64
    raw["claim_commitment_hash"] = "sha256:" + "6" * 64
    mod._refresh_row(raw)
    report = mod.validate_contract_rows(
        bad,
        fixture_manifest=contract["fixture_manifest"],
        transform_manifest=contract["transform_manifest"],
    )
    assert {
        "native_dimension_mismatch",
        "vector_hash_mismatch",
        "model_hash_mismatch",
        "candidate_hash_mismatch",
        "claim_flip_detected",
    } <= set(report["reasons"])

    bad = deepcopy(contract["rows"])
    duplicate = deepcopy(next(row for row in bad if row["row_type"] == "raw_vector_persistence"))
    duplicate["durable_record_id"] = "duplicate"
    mod._refresh_row(duplicate)
    report = mod.validate_contract_rows(
        [*bad, duplicate],
        fixture_manifest=contract["fixture_manifest"],
        transform_manifest=contract["transform_manifest"],
    )
    assert "duplicate_vector_write" in report["reasons"]

    bad = deepcopy(contract["rows"])
    family = next(row for row in bad if row["row_type"] == "family_separation")
    family["family"] = "unknown-family"
    mod._refresh_row(family)
    report = mod.validate_contract_rows(
        bad,
        fixture_manifest=contract["fixture_manifest"],
        transform_manifest=contract["transform_manifest"],
    )
    assert "unknown_family_separation_row" in report["reasons"]

    bad = deepcopy(contract["rows"])
    family = next(row for row in bad if row["row_type"] == "family_separation")
    family["raw_vector_hashes"] = []
    family["native_dimensions_seen"] = [1, 2]
    family["native_dimension_preserved"] = False
    mod._refresh_row(family)
    report = mod.validate_contract_rows(
        bad,
        fixture_manifest=contract["fixture_manifest"],
        transform_manifest=contract["transform_manifest"],
    )
    assert {"family_hash_set_mismatch", "family_pooling_detected"} <= set(
        report["reasons"]
    )

    bad = deepcopy(contract["rows"])
    transform = next(row for row in bad if row["row_type"] == "transform_binding")
    transform["raw_vector_hash"] = "sha256:" + "7" * 64
    mod._refresh_row(transform)
    report = mod.validate_contract_rows(
        bad,
        fixture_manifest=contract["fixture_manifest"],
        transform_manifest=contract["transform_manifest"],
    )
    assert "transform_raw_binding_missing" in report["reasons"]

    bad = deepcopy(contract["rows"])
    transform = next(row for row in bad if row["row_type"] == "transform_binding")
    transform["transform_manifest_hash"] = "sha256:" + "8" * 64
    transform["derived_feature_hash"] = "sha256:" + "9" * 64
    transform["feature_primitives"]["uses_label"] = True
    transform["pooled_raw_vector_hashes"] = []
    mod._refresh_row(transform)
    report = mod.validate_contract_rows(
        bad,
        fixture_manifest=contract["fixture_manifest"],
        transform_manifest=contract["transform_manifest"],
    )
    assert {
        "transform_manifest_mismatch",
        "derived_feature_hash_mismatch",
        "label_leakage_shortcut",
        "family_pooling_detected",
    } <= set(report["reasons"])

    written = mod.build_artifact(
        root=REPO,
        result_path=tmp_path / "build-write.json",
        write=True,
        duration_s=1.0,
        tests_run=[],
    )
    assert json.loads((tmp_path / "build-write.json").read_text(encoding="utf-8")) == written


def test_scenario_infra_6484_artifact_recomputes_and_validates(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SCENARIO-INFRA-6484-ARTIFACT: terminal artifact is row-recomputed."""

    artifact = mod.build_artifact(
        root=REPO,
        result_path=tmp_path / mod.RESULT_RELATIVE_PATH.name,
        write=False,
        duration_s=1.0,
        tests_run=[{"command": "focused", "exit_code": 0}],
    )

    assert set(artifact) == set(mod.REQUIRED_ARTIFACT_FIELDS)
    assert mod.validate_artifact(artifact) == []
    assert artifact["status"] == "complete_non_generation_representation_receipt_contract"
    assert artifact["non_generation_surface_contract_ready_score"] == 1.0
    assert artifact["aggregate_row_recomputation"] == mod.recompute_aggregates_from_rows(
        artifact["per_unit_rows"],
        fixture_manifest=artifact["fixture_manifest"],
        transform_manifest=artifact["transform_manifest"],
    )
    assert artifact["protected_files_unchanged"]["protected_files_unchanged"] is True
    assert artifact["protected_files_unchanged"]["files"]["scripts/research_conductor.py"][
        "unchanged"
    ] is True
    assert artifact["protected_files_unchanged"]["files"]["research-roadmap.yaml"][
        "unchanged"
    ] is True
    assert artifact["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert artifact["verifier_is_oracle"] is True
    assert set(artifact["field_principles"]) == set(mod.REQUIRED_ARTIFACT_FIELDS)
    assert set(artifact["field_provenance"]) == set(mod.REQUIRED_ARTIFACT_FIELDS)
    assert artifact["reproducibility_checksum"] == mod.payload_checksum(artifact)
    assert artifact["honest_verdict"].startswith("complete:")

    bad = _with_checksum({**artifact, "non_generation_surface_contract_ready_score": 0.0})
    assert "non_generation_surface_contract_ready_score mismatch" in mod.validate_artifact(bad)

    bad = deepcopy(artifact)
    bad["per_unit_rows"] = bad["per_unit_rows"][:-1]
    bad = _with_checksum(bad)
    assert "aggregate_row_recomputation mismatch" in mod.validate_artifact(bad)

    bad = _with_checksum({**artifact, "inference_substrate": "live_llm_inference"})
    assert "inference_substrate mismatch" in mod.validate_artifact(bad)

    bad = _with_checksum({**artifact, "verifier_is_oracle": False})
    assert "verifier_is_oracle must be true" in mod.validate_artifact(bad)

    bad = deepcopy(artifact)
    bad["protected_files_unchanged"]["protected_files_unchanged"] = False
    bad = _with_checksum(bad)
    assert "protected_files_unchanged must be true" in mod.validate_artifact(bad)

    bad = _with_checksum({**artifact, "field_provenance": {}})
    assert "field_provenance must cover exactly required fields" in mod.validate_artifact(bad)

    bad = _with_checksum({**artifact, "field_principles": {}})
    assert "missing field_principles entry: status" in mod.validate_artifact(bad)

    bad = _with_checksum({**artifact, "honest_verdict": "done"})
    assert "honest_verdict lacks required terminal prefix" in mod.validate_artifact(bad)

    bad = {**artifact, "reproducibility_checksum": "sha256:bad"}
    assert "reproducibility_checksum mismatch" in mod.validate_artifact(bad)

    bad = deepcopy(artifact)
    del bad["status"]
    assert "missing required field: status" in mod.validate_artifact(bad)

    with monkeypatch.context() as mp:
        mp.setattr(
            mod,
            "_protected_unchanged",
            lambda root, before: {"protected_files_unchanged": False, "files": {}},
        )
        blocked = mod.build_artifact(
            root=REPO,
            result_path=tmp_path / "blocked.json",
            write=False,
            duration_s=1.0,
            tests_run=[],
        )
    assert blocked["status"] == "blocked_non_generation_representation_receipt_contract"
    assert blocked["non_generation_surface_contract_ready_score"] == 0.0
    assert "protected_files_unchanged" in blocked["gate_check_summary"]["failed_gates"]
    assert blocked["honest_verdict"].startswith("complete_blocked:")


def test_req_infra_6484_run_write_and_cli_validate(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """REQ-INFRA-6484: run writes the deliverable and validates it."""

    result = tmp_path / mod.RESULT_RELATIVE_PATH.name
    artifact = mod.run(
        date="20260821",
        result_path=result,
        write=True,
        tests_run=[{"command": "focused", "exit_code": 0}],
    )

    assert json.loads(result.read_text(encoding="utf-8")) == artifact
    assert artifact["non_generation_surface_contract_ready_score"] == 1.0

    result_cli = tmp_path / "cli.json"
    assert mod.main(["--date", "20260821", "--result-path", str(result_cli)]) == 0
    written = json.loads(result_cli.read_text(encoding="utf-8"))
    assert written["status"] == "complete_non_generation_representation_receipt_contract"

    assert mod.main(["--validate", "--result-path", str(result_cli)]) == 0
    out = json.loads(capsys.readouterr().out.splitlines()[-1])
    assert out["ok"] is True

    missing = tmp_path / "missing.json"
    assert mod.main(["--validate", "--result-path", str(missing)]) == 1
    out = json.loads(capsys.readouterr().out.splitlines()[-1])
    assert out == {"errors": ["artifact missing"], "ok": False}


def test_req_infra_6484_substrate_is_recognized_no_llm(tmp_path: Path) -> None:
    """REQ-INFRA-6484: artifact verification recognizes the no-LLM substrate."""

    payload = {
        "experiment": "exp6484_substrate_fixture",
        "honest_verdict": "complete: deterministic representation contract fixture",
        "inference_substrate": mod.INFERENCE_SUBSTRATE,
        "duration_s": 0.01,
        "random_seed": mod.RANDOM_SEED,
        "reproducibility_checksum": "sha256:" + "6" * 64,
        "quoted_context": (
            f"{mod.FAMILY_SPECS[0]['model_id']} and CUDA are fixture strings; "
            "no model was loaded."
        ),
    }
    path = tmp_path / "payload.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    classification = av._classify_inference_substrate(payload)
    floor = av.duration_floor_for_artifact(payload)
    report = av.verify_artifact(path)

    assert classification["kind"] == "no_llm"
    assert classification["matched_value"] == mod.INFERENCE_SUBSTRATE
    assert floor == {
        "substrate": mod.INFERENCE_SUBSTRATE,
        "min_duration_s": av.NO_LLM_DECLARED_MIN_DURATION_S,
        "reason": "no_llm_declared",
    }
    assert report["flags"] == []
