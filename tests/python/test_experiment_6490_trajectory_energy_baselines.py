"""Tests for Exp6490 trajectory energy baselines.

Spec refs: REQ-VERIFY-6490,
SCENARIO-VERIFY-6490-HELD-TRAJECTORY-DISCRIMINATION,
SCENARIO-VERIFY-6490-FAMILY-SEPARATED-REPORTING,
SCENARIO-VERIFY-6490-SHORTCUT-REJECTION,
SCENARIO-VERIFY-6490-BRANCH-RETIREMENT, SCENARIO-VERIFY-6490-ROWS.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any

import scripts.adversarial_verify as adversarial_verify

from carnot import experiment_6490_trajectory_energy_baselines as mod


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


def test_req_verify_6490_spec_declares_baseline_contract() -> None:
    """REQ-VERIFY-6490: OpenSpec owns the trajectory baseline contract."""

    text = SPEC.read_text(encoding="utf-8")
    section = text[text.index("REQ-VERIFY-6490") : text.index("REQ-VERIFY-6486")]
    normalized = " ".join(section.split())

    for marker in (
        "SCENARIO-VERIFY-6490-HELD-TRAJECTORY-DISCRIMINATION",
        "SCENARIO-VERIFY-6490-FAMILY-SEPARATED-REPORTING",
        "SCENARIO-VERIFY-6490-SHORTCUT-REJECTION",
        "SCENARIO-VERIFY-6490-BRANCH-RETIREMENT",
        "SCENARIO-VERIFY-6490-ROWS",
        mod.MODULE_RELATIVE_PATH.as_posix(),
        mod.RESULT_RELATIVE_PATH.as_posix(),
        mod.INFERENCE_SUBSTRATE,
    ):
        assert marker in section
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert f"`{field}`" in section
        assert " ".join(principle.split()) in normalized


def test_scenario_verify_6490_held_rows_and_manifest_are_frozen(
    tmp_path: Path,
) -> None:
    """SCENARIO-VERIFY-6490-HELD-TRAJECTORY-DISCRIMINATION: rows cover heads."""

    artifact = _artifact(tmp_path)
    written = json.loads((tmp_path / mod.RESULT_RELATIVE_PATH.name).read_text())

    assert written == artifact
    assert set(mod.REQUIRED_ARTIFACT_FIELDS) <= set(artifact)
    assert mod.validate_artifact(artifact) == []
    assert artifact["status"] == "disqualified"
    assert artifact["honest_verdict"].startswith("disqualified:")
    assert artifact["trajectory_signal_ready_score"] == 0.0
    assert artifact["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert artifact["verifier_is_oracle"] is True
    classification = adversarial_verify._classify_inference_substrate(artifact)
    assert classification["kind"] == "no_llm"
    floor = adversarial_verify.duration_floor_for_artifact(artifact)
    assert floor == {
        "substrate": mod.INFERENCE_SUBSTRATE,
        "min_duration_s": adversarial_verify.NO_LLM_DECLARED_MIN_DURATION_S,
        "reason": "no_llm_declared",
    }

    gate = artifact["upstream_gate_receipt"]
    assert gate["field"] == "trajectory_contract_ready_score"
    assert gate["expected"] == 1.0
    assert gate["observed"] == 1.0
    assert gate["gate_passed"] is True
    assert gate["sha256"].startswith("sha256:")

    manifest = artifact["frozen_analysis_manifest"]
    assert manifest["held_rows_opened_once"] is True
    assert manifest["held_threshold_tuning_used"] is False
    assert set(manifest["feature_groups"]) == {
        "solver_state_observables",
        "exact_constraint_residuals",
        "exact_bounds",
    }
    assert manifest["preprocessing"]["fitted_on_splits"] == ["development"]
    assert manifest["threshold_policy"]["selected_on"] == "development"
    assert manifest["llm_used"] is False

    config_ids = {row["head_id"] for row in artifact["model_configuration_rows"]}
    assert set(mod.HEAD_IDS) <= config_ids
    assert set(mod.CONTROL_IDS) <= config_ids
    assert all(row["model_is_oracle"] is False for row in artifact["model_configuration_rows"])

    rows = artifact["rows"]
    assert rows == artifact["per_unit_rows"]
    assert {row["head_id"] for row in rows} == set(mod.HEAD_IDS + mod.CONTROL_IDS)
    assert {row["split"] for row in rows} == {"held"}
    assert all(row["verifier_is_oracle"] is True for row in rows)
    assert all(row["model_is_oracle"] is False for row in rows)
    assert all(row["source_raw_row_hash"].startswith("sha256:") for row in rows)
    assert all(row["loss"] >= 0.0 for row in rows)
    assert len({row["unit_id"] for row in rows}) == 24


def test_scenario_verify_6490_cells_attacks_and_retirement_recompute(
    tmp_path: Path,
) -> None:
    """SCENARIO-VERIFY-6490-FAMILY/SHORTCUT/RETIREMENT/ROWS: gates close."""

    artifact = _artifact(tmp_path)
    aggregate = artifact["aggregate_row_recomputation"]
    attacks = artifact["shortcut_attack_matrix"]
    retirement = artifact["branch_retirement_receipt"]

    assert aggregate == mod.recompute_aggregate_row(artifact)
    assert aggregate["trajectory_signal_ready_score_from_rows"] == 0.0
    assert aggregate["held_row_count"] == len(artifact["per_unit_rows"])
    assert aggregate["best_learned_head_id"] in {"linear", "mlp", "kan"}
    assert aggregate["best_learned_beats_analytical"] is True
    assert aggregate["all_shortcuts_rejected"] is False
    assert aggregate["headline_recomputed"] is True

    family_cells = artifact["family_cell_results"]
    assert {row["family_id"] for row in family_cells["rows"]} == set(mod.FAMILY_IDS)
    assert family_cells["disqualifying_family_cell_count"] > 0
    assert family_cells["no_failing_family_pooled_away"] is True

    assert {row["attack_id"] for row in attacks["rows"]} == set(mod.ATTACK_IDS)
    assert attacks["all_shortcuts_rejected"] is False
    assert attacks["surviving_shortcut_count"] > 0
    assert any(row["survived"] for row in attacks["rows"])

    assert artifact["harmful_flip_rows"]
    assert all(row["learned_head_id"] in {"linear", "mlp", "kan"} for row in artifact["harmful_flip_rows"])

    assert retirement["retired"] is True
    assert retirement["reason"] in {"shortcut_verdict_repeated", "complete_null_no_learned_gain"}
    assert "exp5853-paired-embedding-integrity-audit" in retirement["prior_failure_verdicts"]
    assert "exp6487-representation-integrity-audit" in retirement["prior_failure_verdicts"]
    assert artifact["protected_files_unchanged"]["active_roadmap_and_conductor_unchanged"] is True


def test_scenario_verify_6490_blocked_gate_fails_closed(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-6490-ROWS: missing Exp6489 blocks held scoring."""

    blocked = mod.build_artifact(
        root=REPO,
        result_path=tmp_path / "blocked.json",
        exp6489_path=tmp_path / "missing-exp6489.json",
        write=True,
        duration_s=1.0,
        tests_run=TESTS_RUN,
    )

    assert blocked["status"] == "blocked_upstream_gate"
    assert blocked["honest_verdict"].startswith("blocked_upstream_gate:")
    assert blocked["trajectory_signal_ready_score"] == 0.0
    assert blocked["rows"] == []
    assert blocked["per_unit_rows"] == []
    assert blocked["upstream_gate_receipt"]["observed"] is None
    assert "upstream_gate_passed" in blocked["gate_check_summary"]["failed_gates"]
    assert mod.validate_artifact(blocked) == []


def test_scenario_verify_6490_validation_fail_closed(tmp_path: Path, monkeypatch) -> None:
    """SCENARIO-VERIFY-6490-ROWS: malformed artifacts fail validation."""

    artifact = _artifact(tmp_path)

    assert mod._round(None) is None
    assert mod._round(float("nan")) is None
    assert mod._std([]) == 1.0
    assert mod._balanced_accuracy([1, 1], [1, 0]) is None
    assert mod._auroc([0, 0], [0.1, 0.2]) is None
    assert mod._best_threshold([], []) == (0.5, 0.0)

    positive_aggregate = deepcopy(artifact["aggregate_row_recomputation"])
    positive_aggregate["trajectory_signal_ready_score_from_rows"] = 1.0
    assert mod._branch_retirement(REPO, positive_aggregate, {"surviving_shortcut_count": 0})[
        "retired"
    ] is False
    assert mod._status_and_verdict(
        positive_aggregate,
        {"surviving_shortcut_count": 0},
        {"gate_passed": True},
    )[0] == "complete_positive"
    null_aggregate = deepcopy(artifact["aggregate_row_recomputation"])
    null_aggregate["trajectory_signal_ready_score_from_rows"] = 0.0
    assert mod._status_and_verdict(
        null_aggregate,
        {"surviving_shortcut_count": 0},
        {"gate_passed": True},
    )[0] == "complete_null"

    monkeypatch.setattr(
        mod,
        "_git_output",
        lambda root, args: " M research-roadmap.yaml\n M scripts/research_conductor.py",
    )
    protected = mod._protected_files_unchanged(REPO)
    assert protected["changed_paths"] == [
        "research-roadmap.yaml",
        "scripts/research_conductor.py",
    ]

    missing = deepcopy(artifact)
    del missing["status"]
    assert mod.validate_artifact(missing) == ["missing required fields: status"]

    bad = deepcopy(artifact)
    bad["reproducibility_checksum"] = "sha256:bad"
    assert "reproducibility_checksum mismatch" in mod.validate_artifact(bad)

    bad = deepcopy(artifact)
    bad["trajectory_signal_ready_score"] = 1.0
    _with_checksum(bad)
    assert "trajectory_signal_ready_score mismatch" in mod.validate_artifact(bad)

    bad = deepcopy(artifact)
    bad["aggregate_row_recomputation"]["held_row_count"] = -1
    _with_checksum(bad)
    assert "aggregate_row_recomputation mismatch" in mod.validate_artifact(bad)

    bad = deepcopy(artifact)
    bad["shortcut_attack_matrix"]["surviving_shortcut_count"] = 0
    _with_checksum(bad)
    assert "shortcut_attack_matrix mismatch" in mod.validate_artifact(bad)

    bad = deepcopy(artifact)
    bad["family_cell_results"]["rows"] = []
    _with_checksum(bad)
    assert "family_cell_results mismatch" in mod.validate_artifact(bad)

    bad = deepcopy(artifact)
    bad["inference_substrate"] = "live_llm_inference"
    _with_checksum(bad)
    assert "inference_substrate mismatch" in mod.validate_artifact(bad)

    bad = deepcopy(artifact)
    bad["verifier_is_oracle"] = False
    _with_checksum(bad)
    assert "verifier_is_oracle must be true for exact final outcomes" in mod.validate_artifact(
        bad
    )

    bad = deepcopy(artifact)
    bad["field_principles"] = {}
    _with_checksum(bad)
    assert "field_principles must cover exactly required fields" in mod.validate_artifact(bad)

    bad = deepcopy(artifact)
    bad["field_provenance"] = {}
    _with_checksum(bad)
    assert "field_provenance must cover exactly required fields" in mod.validate_artifact(bad)

    bad = deepcopy(artifact)
    bad["protected_files_unchanged"]["active_roadmap_and_conductor_unchanged"] = False
    _with_checksum(bad)
    assert "protected files changed" in mod.validate_artifact(bad)

    bad = deepcopy(artifact)
    bad["rows"] = []
    _with_checksum(bad)
    assert "rows and per_unit_rows must match" in mod.validate_artifact(bad)

    bad = deepcopy(artifact)
    bad["honest_verdict"] = "done"
    _with_checksum(bad)
    assert "honest_verdict lacks required terminal prefix" in mod.validate_artifact(bad)


def test_req_verify_6490_cli_writes_and_validates(
    tmp_path: Path,
    capsys,
) -> None:
    """REQ-VERIFY-6490: CLI writes the artifact and validates it."""

    result = tmp_path / "experiment_6490.json"
    artifact = mod.run(
        date="20260821",
        result_path=result,
        root=REPO,
        tests_run=TESTS_RUN,
    )
    assert json.loads(result.read_text(encoding="utf-8")) == artifact
    assert artifact["trajectory_signal_ready_score"] == 0.0

    assert mod.main(["--date", "20260821", "--result-path", str(result)]) == 0
    assert mod.main(["--validate", "--result-path", str(result)]) == 0
    validate_out = json.loads(capsys.readouterr().out.splitlines()[-1])
    assert validate_out == {"errors": [], "ok": True}

    missing = tmp_path / "missing.json"
    assert mod.main(["--validate", "--result-path", str(missing)]) == 1
    missing_out = json.loads(capsys.readouterr().out.splitlines()[-1])
    assert missing_out == {"errors": ["artifact missing"], "ok": False}
