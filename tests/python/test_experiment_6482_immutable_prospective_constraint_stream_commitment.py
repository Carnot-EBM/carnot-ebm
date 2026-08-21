"""Tests for Exp6482 immutable prospective constraint stream commitment.

Spec refs: REQ-VERIFY-6482, SCENARIO-VERIFY-6482-COMMITMENT,
SCENARIO-VERIFY-6482-BACKEND-PARITY, SCENARIO-VERIFY-6482-RAW-OUTPUT-GATE,
SCENARIO-VERIFY-6482-HELD-ISOLATION, SCENARIO-VERIFY-6482-ATTACKS,
SCENARIO-VERIFY-6482-ROWS.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from carnot import (
    experiment_6482_immutable_prospective_constraint_stream_commitment as mod,
)


REPO = Path(__file__).resolve().parents[2]
SPEC = REPO / mod.SPEC_RELATIVE_PATH


def _passing_tests() -> dict[str, int]:
    return {command: 0 for command in mod.DEFAULT_TEST_COMMANDS}


def _with_checksum(artifact: dict[str, object]) -> dict[str, object]:
    artifact["reproducibility_checksum"] = mod.payload_checksum(artifact)
    return artifact


def test_req_verify_6482_spec_declares_contract_fields_and_scenarios() -> None:
    """REQ-VERIFY-6482: OpenSpec owns the prospective stream contract."""

    text = SPEC.read_text(encoding="utf-8")
    section = text[text.index("REQ-VERIFY-6482") : text.index("REQ-VERIFY-6478")]
    for marker in (
        "SCENARIO-VERIFY-6482-COMMITMENT",
        "SCENARIO-VERIFY-6482-BACKEND-PARITY",
        "SCENARIO-VERIFY-6482-RAW-OUTPUT-GATE",
        "SCENARIO-VERIFY-6482-HELD-ISOLATION",
        "SCENARIO-VERIFY-6482-ATTACKS",
        "SCENARIO-VERIFY-6482-ROWS",
        mod.MODULE_RELATIVE_PATH.as_posix(),
        mod.RESULT_RELATIVE_PATH.as_posix(),
        mod.MANIFEST_DIR_RELATIVE_PATH.as_posix(),
        mod.INFERENCE_SUBSTRATE,
    ):
        assert marker in section
    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert f"`{field}`" in section
        assert mod.FIELD_PRINCIPLES[field] in section


def test_scenario_verify_6482_commitment_units_are_balanced_and_sealed() -> None:
    """SCENARIO-VERIFY-6482-COMMITMENT: units, splits, labels, and policies seal."""

    units = mod.predeclared_units()
    rows = [unit.to_manifest_row() for unit in units]
    assert len(units) == mod.UNIT_COUNT
    assert len({unit.unit_id for unit in units}) == mod.UNIT_COUNT
    assert {unit.family_id for unit in units} == set(mod.FAMILY_IDS)
    assert sum(1 for unit in units if unit.split == "held") >= 24

    split_counts = mod.family_split_counts(rows)
    for family_id in mod.FAMILY_IDS:
        assert split_counts[family_id] == {"calibration": 2, "development": 6, "held": 8}

    for row in rows:
        assert row["prompt_hash"].startswith("sha256:")
        assert row["record_hash"].startswith("sha256:")
        assert row["candidate_policy_ids"] == list(mod.CANDIDATE_POLICY_DEFINITIONS)
        assert row["protected_constraint_ids"]
        assert row["candidate_headroom"]["can_differentiate"] is True
        assert row["seed"] >= mod.RANDOM_SEED
        assert "exp6463" not in json.dumps(row).lower()

    manifest = mod.build_prospective_stream_manifest(units, root=REPO)
    assert manifest["manifest_hash"].startswith("sha256:")
    assert manifest["unit_count"] == mod.UNIT_COUNT
    assert manifest["held_unit_count"] == 24
    assert manifest["candidate_policy_definitions"] == mod.CANDIDATE_POLICY_DEFINITIONS
    assert manifest["family_split_counts"] == split_counts


def test_scenario_verify_6482_backend_parity_and_commitment_receipts(
    tmp_path: Path,
) -> None:
    """SCENARIO-VERIFY-6482-BACKEND-PARITY: Z3 and exhaustive labels match."""

    artifact = mod.build_artifact(
        root=REPO,
        run_date="20260821",
        duration_s=0.25,
        tests_run=_passing_tests(),
        manifest_dir=tmp_path / "manifest",
        future_raw_output_dir=tmp_path / "raw_outputs",
    )
    assert set(mod.REQUIRED_ARTIFACT_FIELDS) <= set(artifact)
    assert artifact["status"] == "complete"
    assert artifact["prospective_contract_ready_score"] == 1.0
    assert artifact["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert artifact["verifier_is_oracle"] is True
    assert artifact["gate_check_summary"]["all_gates_passed"] is True
    assert artifact["preconditions_checked"]["exp6477_ready_score"] == 1.0
    assert artifact["preconditions_checked"]["exp6477_artifact_sha256"].startswith("sha256:")
    assert artifact["preconditions_checked"]["exp6463_lineage_retired"] is True

    manifest = artifact["prospective_stream_manifest"]
    assert manifest["unit_count"] == 48
    assert manifest["held_unit_count"] == 24
    assert manifest["manifest_path"].endswith("prospective_stream_manifest.json")
    assert Path(manifest["manifest_path"]).is_file()
    assert Path(manifest["commitment_event_path"]).is_file()
    assert Path(manifest["unit_rows_path"]).is_file()

    assert len(artifact["label_commitment_receipts"]) == 48
    assert len(artifact["membership_commitment_receipts"]) == 48
    assert len(artifact["backend_parity_rows"]) == 96
    assert len(artifact["protected_clause_manifest"]["rows"]) == 48
    assert artifact["raw_output_empty_state_receipt"]["empty_state_pass"] is True
    assert artifact["held_isolation_receipt"]["held_leakage_count"] == 0
    assert artifact["headroom_manifest"]["all_units_have_headroom"] is True
    assert artifact["aggregate_row_recomputation"] == mod.recompute_aggregates_from_rows(
        artifact["per_unit_rows"]
    )
    assert set(artifact["field_provenance"]) == set(mod.REQUIRED_ARTIFACT_FIELDS)
    assert artifact["reproducibility_checksum"] == mod.payload_checksum(artifact)
    assert mod.validate_artifact(artifact) == []


def test_scenario_verify_6482_raw_output_gate_blocks_fake_prior_output(
    tmp_path: Path,
) -> None:
    """SCENARIO-VERIFY-6482-RAW-OUTPUT-GATE: fake raw output fails closed."""

    raw_dir = tmp_path / "exp6483" / "raw_outputs"
    assert mod.raw_output_empty_state_receipt(raw_dir)["empty_state_pass"] is True
    raw_dir.mkdir(parents=True)
    (raw_dir / "fake-earlier.raw.json").write_text('{"candidate":"posthoc"}\n', encoding="utf-8")
    receipt = mod.raw_output_empty_state_receipt(raw_dir)
    assert receipt["empty_state_pass"] is False
    assert receipt["file_count"] == 1

    artifact = mod.build_artifact(
        root=REPO,
        run_date="20260821",
        duration_s=0.25,
        tests_run=_passing_tests(),
        manifest_dir=tmp_path / "manifest",
        future_raw_output_dir=raw_dir,
    )
    assert artifact["status"] == "blocked_prospective_contract"
    assert artifact["prospective_contract_ready_score"] == 0.0
    assert "raw_outputs_absent_or_empty" in artifact["gate_check_summary"]["failed_gates"]
    assert "raw_output_empty_state_receipt failed" in mod.validate_artifact(
        _with_checksum(deepcopy(artifact))
    )


def test_scenario_verify_6482_attack_matrix_and_held_isolation() -> None:
    """SCENARIO-VERIFY-6482-HELD-ISOLATION and ATTACKS: attacks fail closed."""

    bundle = mod.build_commitment_bundle(root=REPO, future_raw_output_dir=Path("/tmp/absent-6482"))
    isolation = mod.held_isolation_receipt(bundle["units"])
    assert isolation["held_leakage_count"] == 0
    assert isolation["development_selector_input_hash"].startswith("sha256:")
    assert isolation["held_secret_hashes"]

    attack_matrix = mod.build_attack_matrix(bundle)
    assert {row["attack_id"] for row in attack_matrix["rows"]} == set(mod.ATTACK_IDS)
    assert attack_matrix["all_attacks_failed_closed"] is True
    assert attack_matrix["false_accept_count"] == 0

    by_id = {row["attack_id"]: row for row in attack_matrix["rows"]}
    assert by_id["posthoc_label_edit"]["detected"] is True
    assert by_id["split_move"]["detected"] is True
    assert by_id["duplicate_unit"]["detected"] is True
    assert by_id["family_imbalance"]["detected"] is True
    assert by_id["objective_sign_change"]["detected"] is True
    assert by_id["unsupported_operation"]["detected"] is True
    assert by_id["held_prompt_leakage"]["detected"] is True
    assert by_id["fake_earlier_raw_output"]["detected"] is True
    assert by_id["exp6463_hash_reuse"]["detected"] is True


def test_scenario_verify_6482_rows_validate_and_fail_closed(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-6482-ROWS: validation derives readiness from rows."""

    artifact = mod.build_artifact(
        root=REPO,
        run_date="20260821",
        duration_s=0.25,
        tests_run=_passing_tests(),
        manifest_dir=tmp_path / "manifest",
        future_raw_output_dir=tmp_path / "raw_outputs",
    )
    assert mod.validate_artifact(artifact) == []

    bad = deepcopy(artifact)
    bad["prospective_contract_ready_score"] = 0.0
    assert "prospective_contract_ready_score mismatch" in mod.validate_artifact(
        _with_checksum(bad)
    )

    bad = deepcopy(artifact)
    bad["per_unit_rows"] = bad["per_unit_rows"][:-1]
    assert "aggregate_row_recomputation mismatch" in mod.validate_artifact(
        _with_checksum(bad)
    )

    incomplete_backend_rows = [
        row
        for row in artifact["per_unit_rows"]
        if not (
            row.get("row_type") == "backend_parity"
            and row.get("unit_id") == "exp6482-boolean-guard-00"
            and row.get("backend") == "exhaustive"
        )
    ]
    incomplete = mod.recompute_aggregates_from_rows(incomplete_backend_rows)
    assert incomplete["backend_pairs_complete"] is False
    assert incomplete["backend_parity_mismatch_count"] == 1
    assert incomplete["prospective_contract_ready_score_from_rows"] == 0.0

    bad = deepcopy(artifact)
    bad["inference_substrate"] = "live_llm_inference"
    assert "inference_substrate mismatch" in mod.validate_artifact(_with_checksum(bad))

    bad = deepcopy(artifact)
    bad["verifier_is_oracle"] = False
    assert "verifier_is_oracle must be true within declared finite-domain record" in (
        mod.validate_artifact(_with_checksum(bad))
    )

    bad = deepcopy(artifact)
    bad["field_principles"] = {}
    assert "missing field_principles entry: status" in mod.validate_artifact(
        _with_checksum(bad)
    )

    bad = deepcopy(artifact)
    bad["field_provenance"] = {}
    assert "field_provenance must cover exactly required fields" in mod.validate_artifact(
        _with_checksum(bad)
    )

    bad = deepcopy(artifact)
    bad["attack_matrix"]["all_attacks_failed_closed"] = False
    assert "attack matrix must fail closed" in mod.validate_artifact(_with_checksum(bad))

    bad = deepcopy(artifact)
    bad["protected_files_unchanged"]["unchanged"] = False
    assert "protected files changed" in mod.validate_artifact(_with_checksum(bad))

    bad = deepcopy(artifact)
    bad["honest_verdict"] = "done"
    assert "honest_verdict lacks required terminal prefix" in mod.validate_artifact(
        _with_checksum(bad)
    )

    bad = deepcopy(artifact)
    del bad["status"]
    assert "missing required field: status" in mod.validate_artifact(bad)

    bad = deepcopy(artifact)
    bad["reproducibility_checksum"] = "sha256:bad"
    assert "reproducibility_checksum mismatch" in mod.validate_artifact(bad)

    path = tmp_path / "artifact.json"
    mod.write_artifact(artifact, path)
    assert json.loads(path.read_text(encoding="utf-8")) == artifact


def test_req_verify_6482_run_write_and_cli_validate(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """REQ-VERIFY-6482: CLI writes and validates the terminal artifact."""

    result = tmp_path / "experiment_6482.json"
    manifest = tmp_path / "manifest"
    raw_dir = tmp_path / "raw_outputs"
    artifact = mod.run(
        date="20260821",
        result_path=result,
        manifest_dir=manifest,
        future_raw_output_dir=raw_dir,
        test_exit_codes=_passing_tests(),
    )
    assert json.loads(result.read_text(encoding="utf-8")) == artifact
    assert artifact["status"] == "complete"

    assert mod.main(
        [
            "--date",
            "20260821",
            "--result-path",
            str(result),
            "--manifest-dir",
            str(manifest),
            "--future-raw-output-dir",
            str(raw_dir),
        ]
    ) == 0
    written = json.loads(result.read_text(encoding="utf-8"))
    assert written["prospective_contract_ready_score"] == 1.0

    assert mod.main(["--validate", "--result-path", str(result)]) == 0
    validate_out = json.loads(capsys.readouterr().out.splitlines()[-1])
    assert validate_out["ok"] is True

    missing = tmp_path / "missing.json"
    assert mod.main(["--validate", "--result-path", str(missing)]) == 1
    missing_out = json.loads(capsys.readouterr().out.splitlines()[-1])
    assert missing_out["ok"] is False
    assert missing_out["errors"] == ["artifact missing"]
