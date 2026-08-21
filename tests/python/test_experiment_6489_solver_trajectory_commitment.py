"""Tests for Exp6489 solver trajectory commitment.

Spec refs: REQ-VERIFY-6489, SCENARIO-VERIFY-6489-TRAJECTORY-COMMITMENT,
SCENARIO-VERIFY-6489-LABEL-AUTHORITY, SCENARIO-VERIFY-6489-SPLITS,
SCENARIO-VERIFY-6489-LEAKAGE, SCENARIO-VERIFY-6489-ROWS.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any

import pytest

import scripts.adversarial_verify as adversarial_verify
from carnot import experiment_6489_solver_trajectory_commitment as mod


REPO = Path(__file__).resolve().parents[2]
SPEC = REPO / mod.SPEC_RELATIVE_PATH
TESTS_RUN = [{"command": command, "exit_code": 0} for command in mod.DEFAULT_TEST_COMMANDS]


def _artifact(tmp_path: Path) -> dict[str, Any]:
    return mod.build_artifact(
        root=REPO,
        result_path=tmp_path / mod.RESULT_RELATIVE_PATH.name,
        write=True,
        duration_s=1.0,
        tests_run=TESTS_RUN,
    )


def _with_checksum(artifact: dict[str, Any]) -> dict[str, Any]:
    artifact["reproducibility_checksum"] = mod.reproducibility_checksum(artifact)
    return artifact


def test_req_verify_6489_spec_declares_trajectory_contract() -> None:
    """REQ-VERIFY-6489: OpenSpec owns the trajectory commitment."""

    text = SPEC.read_text(encoding="utf-8")
    section = text[text.index("REQ-VERIFY-6489") : text.index("REQ-VERIFY-6486")]
    normalized = " ".join(section.split())

    for marker in (
        "SCENARIO-VERIFY-6489-TRAJECTORY-COMMITMENT",
        "SCENARIO-VERIFY-6489-LABEL-AUTHORITY",
        "SCENARIO-VERIFY-6489-SPLITS",
        "SCENARIO-VERIFY-6489-LEAKAGE",
        "SCENARIO-VERIFY-6489-ROWS",
        mod.MODULE_RELATIVE_PATH.as_posix(),
        mod.RESULT_RELATIVE_PATH.as_posix(),
        mod.INFERENCE_SUBSTRATE,
    ):
        assert marker in section
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert f"`{field}`" in section
        assert " ".join(principle.split()) in normalized


def test_scenario_verify_6489_records_chronological_exact_trajectories(
    tmp_path: Path,
) -> None:
    """SCENARIO-VERIFY-6489-TRAJECTORY-COMMITMENT: events bind to outcomes."""

    artifact = _artifact(tmp_path)
    written = json.loads((tmp_path / mod.RESULT_RELATIVE_PATH.name).read_text())
    raw_rows = artifact["raw_trajectory_rows"]
    outcome_rows = artifact["final_exact_outcome_rows"]
    labels = artifact["persistence_label_rows"]

    assert written == artifact
    assert mod.validate_artifact(artifact) == []
    assert artifact["status"] == "complete_trajectory_commitment"
    assert artifact["honest_verdict"].startswith("complete_trajectory_commitment:")
    assert artifact["trajectory_contract_ready_score"] == 1.0
    assert artifact["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert artifact["verifier_is_oracle"] is True

    assert artifact["upstream_gate_receipt"]["field"] == "v560_lineage_lock_ready_score"
    assert artifact["upstream_gate_receipt"]["expected"] == 1.0
    assert artifact["upstream_gate_receipt"]["observed"] == 1.0
    assert artifact["gate_check_summary"]["observed_field"] == "v560_lineage_lock_ready_score"
    assert artifact["gate_check_summary"]["observed_value"] == 1.0
    assert artifact["source_stream_receipt"]["path"].endswith(
        "experiment_6482_immutable_prospective_constraint_stream_commitment.json"
    )
    assert artifact["source_stream_receipt"]["sha256"].startswith("sha256:")

    assert len(raw_rows) == mod.UNIT_COUNT * len(mod.BACKENDS) * len(mod.CHECKPOINTS)
    assert len(outcome_rows) == mod.UNIT_COUNT * len(mod.BACKENDS)
    assert len(labels) == len(raw_rows)
    assert len({row["raw_row_hash"] for row in raw_rows}) == len(raw_rows)
    assert all(row["event_index"] == index for index, row in enumerate(raw_rows))
    assert all(row["event_time_s"] >= 0.0 for row in raw_rows)
    assert all(row["branch_depth"] >= 1 for row in raw_rows)
    assert all(row["final_exact_outcome_hash"].startswith("sha256:") for row in raw_rows)
    assert all(row["exact_bounds"]["candidate_count_under_partial"] >= 1 for row in raw_rows)
    assert all(row["constraint_residuals"]["constraint_count"] >= 1 for row in raw_rows)
    assert {row["backend"] for row in raw_rows} == set(mod.BACKENDS)
    assert {row["family_id"] for row in raw_rows} == set(mod.FAMILY_IDS)
    assert all(row["release_authority"] is True for row in outcome_rows)
    assert all(row["verifier_is_oracle"] is True for row in outcome_rows)


def test_scenario_verify_6489_persistence_labels_replay_from_final_outcomes(
    tmp_path: Path,
) -> None:
    """SCENARIO-VERIFY-6489-LABEL-AUTHORITY: labels replay from final rows."""

    artifact = _artifact(tmp_path)
    outcomes = mod.outcomes_by_hash(artifact["final_exact_outcome_rows"])
    raw_by_hash = {row["raw_row_hash"]: row for row in artifact["raw_trajectory_rows"]}

    for label in artifact["persistence_label_rows"]:
        replayed = mod.persistence_label_for_raw_row(
            raw_by_hash[label["raw_row_hash"]],
            outcomes[label["final_exact_outcome_hash"]],
        )
        assert replayed == label
        assert label["label_source"] == "final_exact_solver_outcome"
        assert label["llm_label_used"] is False
        assert label["model_seen_before_commitment"] is False

    mutated = deepcopy(artifact)
    mutated["final_exact_outcome_rows"][0]["final_assignment"] = {"bad": 1}
    _with_checksum(mutated)
    assert "persistence_label_rows mismatch" in mod.validate_artifact(mutated)


def test_scenario_verify_6489_splits_and_features_are_identity_free(
    tmp_path: Path,
) -> None:
    """SCENARIO-VERIFY-6489-SPLITS: held split predates feature extraction."""

    artifact = _artifact(tmp_path)
    split = artifact["split_commitment"]
    contract = artifact["identity_free_feature_contract"]

    assert split["label_inspected_before_split"] is False
    assert split["commitment_event_index"] < split["feature_extraction_event_index"]
    assert split["held_predates_feature_extraction"] is True
    assert split["family_split_counts"] == {
        family_id: {"train": 6, "development": 2, "held": 8}
        for family_id in mod.FAMILY_IDS
    }
    assert all("persistence" not in json.dumps(row) for row in split["rows"])
    assert contract["no_label_fields_allowed"] is True
    assert contract["feature_extraction_after_split"] is True
    assert set(contract["forbidden_feature_fields"]) >= {
        "unit_id",
        "backend",
        "family_id",
        "serialization_length",
        "checkpoint_index",
        "row_order",
    }
    assert set(contract["allowed_feature_fields"]).isdisjoint(
        set(contract["forbidden_feature_fields"])
    )


def test_scenario_verify_6489_leakage_attacks_fail_closed_and_rows_recompute(
    tmp_path: Path,
) -> None:
    """SCENARIO-VERIFY-6489-LEAKAGE/ROWS: attacks and rows drive readiness."""

    artifact = _artifact(tmp_path)
    attacks = {row["attack_id"]: row for row in artifact["leakage_attack_matrix"]["rows"]}
    aggregate = artifact["aggregate_row_recomputation"]

    assert set(attacks) == set(mod.LEAKAGE_ATTACK_IDS)
    assert all(row["fail_closed"] is True for row in attacks.values())
    assert all(row["allowed_as_feature"] is False for row in attacks.values())
    assert artifact["leakage_attack_matrix"]["false_accept_count"] == 0
    assert aggregate == mod.recompute_aggregates_from_rows(artifact["per_unit_rows"])
    assert aggregate["trajectory_contract_ready_score_from_rows"] == 1.0
    assert aggregate["label_reproducibility_failure_count"] == 0
    assert aggregate["split_predates_feature_extraction"] is True
    assert aggregate["duplicate_trajectory_key_count"] == 0
    assert artifact["protected_files_unchanged"]["active_roadmap_and_conductor_unchanged"] is True

    bad = deepcopy(artifact)
    bad["raw_trajectory_rows"] = bad["raw_trajectory_rows"][:-1]
    bad["per_unit_rows"] = [
        row for row in bad["per_unit_rows"] if row.get("raw_row_hash") != artifact["raw_trajectory_rows"][-1]["raw_row_hash"]
    ]
    _with_checksum(bad)
    assert "aggregate_row_recomputation mismatch" in mod.validate_artifact(bad)


def test_scenario_verify_6489_blocked_gate_and_validation_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SCENARIO-VERIFY-6489-ROWS: malformed artifacts fail validation."""

    assert mod.canonical_json({"b": 2, "a": 1}) == '{"a":1,"b":2}'

    missing_gate = tmp_path / "missing-exp6488.json"
    blocked = mod.build_artifact(
        root=REPO,
        result_path=tmp_path / "blocked.json",
        exp6488_path=missing_gate,
        write=False,
        duration_s=1.0,
        tests_run=TESTS_RUN,
    )

    assert blocked["status"] == "blocked_trajectory_commitment"
    assert blocked["trajectory_contract_ready_score"] == 0.0
    assert blocked["honest_verdict"].startswith("blocked_trajectory_commitment:")
    assert blocked["upstream_gate_receipt"]["observed"] is None
    assert "upstream_gate_passed" in blocked["gate_check_summary"]["failed_gates"]
    assert mod.validate_artifact(blocked) == []

    clean = _artifact(tmp_path / "clean")
    missing_field = deepcopy(clean)
    del missing_field["status"]
    assert mod.validate_artifact(missing_field) == ["missing required fields: status"]

    bad_checksum = deepcopy(clean)
    bad_checksum["reproducibility_checksum"] = "sha256:bad"
    assert "reproducibility_checksum mismatch" in mod.validate_artifact(bad_checksum)

    bad_contract = deepcopy(clean)
    bad_contract["identity_free_feature_contract"]["forbidden_feature_fields"] = []
    _with_checksum(bad_contract)
    assert "identity_free_feature_contract allows forbidden leakage fields" in mod.validate_artifact(
        bad_contract
    )

    bad_principles = deepcopy(clean)
    bad_principles["field_principles"] = {}
    _with_checksum(bad_principles)
    assert "field_principles must cover exactly required fields" in mod.validate_artifact(
        bad_principles
    )

    bad_provenance = deepcopy(clean)
    bad_provenance["field_provenance"] = {}
    _with_checksum(bad_provenance)
    assert "field_provenance must cover exactly required fields" in mod.validate_artifact(
        bad_provenance
    )

    bad_substrate = deepcopy(clean)
    bad_substrate["inference_substrate"] = "live_llm"
    _with_checksum(bad_substrate)
    assert "inference_substrate mismatch" in mod.validate_artifact(bad_substrate)

    bad_oracle = deepcopy(clean)
    bad_oracle["verifier_is_oracle"] = False
    _with_checksum(bad_oracle)
    assert "verifier_is_oracle must be true for final exact outcomes" in mod.validate_artifact(
        bad_oracle
    )

    bad_score = deepcopy(clean)
    bad_score["trajectory_contract_ready_score"] = 0.0
    _with_checksum(bad_score)
    assert "trajectory_contract_ready_score mismatch" in mod.validate_artifact(bad_score)

    bad_overlap = deepcopy(clean)
    bad_overlap["identity_free_feature_contract"]["allowed_feature_fields"] = ["unit_id"]
    _with_checksum(bad_overlap)
    assert "identity_free_feature_contract overlaps allowed and forbidden fields" in (
        mod.validate_artifact(bad_overlap)
    )

    bad_attack = deepcopy(clean)
    bad_attack["leakage_attack_matrix"]["false_accept_count"] = 1
    _with_checksum(bad_attack)
    assert "leakage_attack_matrix must fail closed" in mod.validate_artifact(bad_attack)

    bad_protected = deepcopy(clean)
    bad_protected["protected_files_unchanged"][
        "active_roadmap_and_conductor_unchanged"
    ] = False
    _with_checksum(bad_protected)
    assert "protected files changed" in mod.validate_artifact(bad_protected)

    bad_verdict = deepcopy(clean)
    bad_verdict["honest_verdict"] = "done"
    _with_checksum(bad_verdict)
    assert "honest_verdict lacks required terminal prefix" in mod.validate_artifact(bad_verdict)

    monkeypatch.setattr(
        mod,
        "_git_output",
        lambda root, args: " M scripts/research_conductor.py",
    )
    protected = mod._protected_files_unchanged(REPO)
    assert protected["changed_paths"] == ["scripts/research_conductor.py"]


def test_req_verify_6489_cli_validate_and_substrate_recognition(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """REQ-VERIFY-6489: CLI writes, validates, and the substrate has a floor."""

    result = tmp_path / "experiment_6489.json"
    artifact = mod.run(
        date="20260821",
        result_path=result,
        root=REPO,
        tests_run=TESTS_RUN,
    )
    assert json.loads(result.read_text(encoding="utf-8")) == artifact
    assert artifact["trajectory_contract_ready_score"] == 1.0

    assert mod.main(["--date", "20260821", "--result-path", str(result)]) == 0
    assert mod.main(["--validate", "--result-path", str(result)]) == 0
    validate_out = json.loads(capsys.readouterr().out.splitlines()[-1])
    assert validate_out == {"errors": [], "ok": True}

    missing = tmp_path / "missing.json"
    assert mod.main(["--validate", "--result-path", str(missing)]) == 1
    missing_out = json.loads(capsys.readouterr().out.splitlines()[-1])
    assert missing_out == {"errors": ["artifact missing"], "ok": False}

    floor = adversarial_verify.duration_floor_for_artifact(artifact)
    report = adversarial_verify.verify_artifact(result)
    assert adversarial_verify._classify_inference_substrate(artifact)["kind"] == "no_llm"
    assert floor == {
        "substrate": mod.INFERENCE_SUBSTRATE,
        "min_duration_s": adversarial_verify.NO_LLM_DECLARED_MIN_DURATION_S,
        "reason": "no_llm_declared",
    }
    assert report["flags"] == []
