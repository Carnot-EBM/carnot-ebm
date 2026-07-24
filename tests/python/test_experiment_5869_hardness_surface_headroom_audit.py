"""Tests for Exp5869 hardness-surface headroom audit.

Spec refs: REQ-VERIFY-5869, SCENARIO-VERIFY-5869-INTEGRITY,
SCENARIO-VERIFY-5869-SPLITS, SCENARIO-VERIFY-5869-CONTROLS,
SCENARIO-VERIFY-5869-DESIGN.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any

import pytest

from carnot import experiment_5869_hardness_surface_headroom_audit as mod


REPO = Path(__file__).resolve().parents[2]
VERIFY_SPEC = REPO / "openspec/capabilities/verification/spec.md"
TEST_COMMAND = (
    ".venv/bin/pytest "
    "tests/python/test_experiment_5869_hardness_surface_headroom_audit.py "
    "-q --no-cov -n 0"
)
COVERAGE_COMMAND = (
    ".venv/bin/coverage run --rcfile=/dev/null "
    "--include=python/carnot/experiment_5869_hardness_surface_headroom_audit.py "
    "-m pytest tests/python/test_experiment_5869_hardness_surface_headroom_audit.py "
    "-q --no-cov -n 0 && .venv/bin/coverage report --rcfile=/dev/null "
    "--include=python/carnot/experiment_5869_hardness_surface_headroom_audit.py "
    "--fail-under=100"
)
GATE_REPLAY_COMMAND = (
    ".venv/bin/python -c \"from carnot import "
    "experiment_5869_hardness_surface_headroom_audit as m; "
    "assert m.upstream_gate_receipt()['upstream_ready'] is True\""
)
FULL_TEST_COMMAND = ".venv/bin/pytest tests/python -q"
SPEC_COMMAND = ".venv/bin/python scripts/check_spec_coverage.py"
ADVERSARIAL_COMMAND = (
    ".venv/bin/python scripts/adversarial_verify.py "
    "results/experiment_5869_hardness_surface_headroom_audit.json"
)
ROOT_CLUTTER_COMMAND = ".venv/bin/python scripts/root_clutter_sweep.py"
PROTECTED_FILE_COMMAND = (
    '.venv/bin/python -c "from pathlib import Path; '
    "assert Path('scripts/research_conductor.py').exists()\""
)
TEST_COMMANDS = [
    TEST_COMMAND,
    COVERAGE_COMMAND,
    GATE_REPLAY_COMMAND,
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
        memory_probe=lambda: {"available_mb": 32768, "required_mb": 1024, "ok": True},
        disk_probe=lambda root: {"available_mb": 32768, "required_mb": 512, "ok": True},
    )


@pytest.fixture(scope="module")
def exp5869_audit(
    tmp_path_factory: pytest.TempPathFactory,
) -> tuple[dict[str, Any], list[dict[str, Any]], Path]:
    """REQ-VERIFY-5869: build the deterministic audit artifact once."""

    base = tmp_path_factory.mktemp("exp5869")
    conductor = REPO / mod.PROTECTED_FILES[0]
    before_hash = mod.sha256_file(conductor)
    artifact = mod.run(
        result_path=base / mod.RESULT_RELATIVE_PATH.name,
        preconditions_checked=_preconditions(base),
        test_commands=TEST_COMMANDS,
        test_exit_codes=TEST_EXIT_CODES,
        duration_s=9.0,
        write=True,
    )
    rows = mod.read_upstream_rows()
    assert mod.sha256_file(conductor) == before_hash
    return artifact, rows, base


def test_req_verify_5869_spec_declares_headroom_audit_contract() -> None:
    """REQ-VERIFY-5869: OpenSpec anchors every field and terminal principle."""

    text = VERIFY_SPEC.read_text(encoding="utf-8")
    section = text[text.index("### REQ-VERIFY-5869") :]
    normalized = " ".join(section.split())

    for marker in (
        "REQ-VERIFY-5869",
        "SCENARIO-VERIFY-5869-INTEGRITY",
        "SCENARIO-VERIFY-5869-SPLITS",
        "SCENARIO-VERIFY-5869-CONTROLS",
        "SCENARIO-VERIFY-5869-DESIGN",
        mod.RESULT_RELATIVE_PATH.as_posix(),
        mod.UPSTREAM_ARTIFACT_RELATIVE_PATH.as_posix(),
        mod.UPSTREAM_ROWS_RELATIVE_PATH.as_posix(),
        mod.INFERENCE_SUBSTRATE,
        "`hardness_surface_headroom_ready_score`",
    ):
        assert marker in section
    for field, principle in mod.REQUIRED_FIELD_PRINCIPLES.items():
        assert f"`{field}`" in section
        assert " ".join(principle.split()) in normalized


def test_req_verify_5869_terminal_artifact_is_hash_bound_and_null_when_saturated(
    exp5869_audit: tuple[dict[str, Any], list[dict[str, Any]], Path],
) -> None:
    """REQ-VERIFY-5869: terminal JSON is stable and does not hide saturation."""

    artifact, _rows, base = exp5869_audit
    rerun = mod.run(
        result_path=base / "rerun.json",
        preconditions_checked=_preconditions(base / "rerun"),
        test_commands=TEST_COMMANDS,
        test_exit_codes=TEST_EXIT_CODES,
        duration_s=99.0,
        write=True,
    )

    assert mod.validate_artifact(artifact) is True
    assert json.loads((base / mod.RESULT_RELATIVE_PATH.name).read_text()) == artifact
    assert artifact["status"] == "complete_null"
    assert artifact["honest_verdict"].startswith("complete_null:")
    assert artifact["hardness_surface_headroom_ready_score"] == 0.0
    assert isinstance(artifact["hardness_surface_headroom_ready_score"], float)
    assert artifact["duration_s"] == pytest.approx(9.0)
    assert artifact["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert artifact["verifier_is_oracle"] is True
    assert artifact["preconditions_checked"]["preconditions_ready"] is True
    assert artifact["upstream_gate_receipt"]["upstream_ready"] is True
    assert artifact["protected_files_unchanged"]["all_unchanged"] is True
    assert artifact["reproducibility_checksum"] == rerun["reproducibility_checksum"]
    assert set(mod.REQUIRED_ARTIFACT_FIELDS) <= set(artifact)
    assert set(artifact["test_exit_codes"]) == set(artifact["test_commands"])
    assert all(code == 0 for code in artifact["test_exit_codes"].values())

    for field, principle in mod.REQUIRED_FIELD_PRINCIPLES.items():
        assert artifact["field_provenance"][field]["principle"] == principle


def test_scenario_verify_5869_integrity_replays_rows_without_trusting_summary(
    exp5869_audit: tuple[dict[str, Any], list[dict[str, Any]], Path],
) -> None:
    """SCENARIO-VERIFY-5869-INTEGRITY: labels, certificates, and relabels replay."""

    artifact, rows, _base = exp5869_audit
    replay = artifact["independent_row_integrity_replay"]

    assert replay["all_integrity_checks_passed"] is True
    assert replay["row_count"] == 84
    assert replay["row_hash_mismatch_count"] == 0
    assert replay["summary_row_hash_mismatch_count"] == 0
    assert replay["exact_label_disagreement_count"] == 0
    assert replay["certificate_failure_count"] == 0
    assert replay["solver_replay_disagreement_count"] == 0
    assert replay["matching_tolerance_passed"] is True
    assert replay["relabel_equivalence_failure_count"] == 0
    assert artifact["label_balance_and_headroom"]["label_counts"] == {
        "satisfiable": 42,
        "unsatisfiable": 42,
    }
    assert artifact["label_balance_and_headroom"]["balanced_error_headroom_exists"] is True

    bad_label = deepcopy(rows[0])
    bad_label["expected_label"] = (
        "unsatisfiable" if rows[0]["expected_label"] == "satisfiable" else "satisfiable"
    )
    assert mod.independent_row_integrity_replay([bad_label], artifact)["all_integrity_checks_passed"] is False
    bad_certificate = deepcopy(rows[0])
    bad_certificate["certificate"]["validated"] = False
    assert mod.recompute_certificate_validity(bad_certificate) is False
    bad_hash = deepcopy(rows[0])
    bad_hash["row_hash"] = "sha256:bad"
    assert mod.independent_row_integrity_replay([bad_hash], artifact)["row_hash_mismatch_count"] == 1


def test_scenario_verify_5869_splits_prevent_semantic_group_leakage(
    exp5869_audit: tuple[dict[str, Any], list[dict[str, Any]], Path],
) -> None:
    """SCENARIO-VERIFY-5869-SPLITS: frozen groups do not cross train/test."""

    artifact, rows, _base = exp5869_audit
    splits = artifact["leakage_safe_split_receipts"]

    assert splits["splits_frozen_before_controls"] is True
    assert splits["all_splits_leakage_safe"] is True
    assert splits["duplicate_semantic_instances_across_splits"] == []
    assert splits["instance_group_split"]["train_row_count"] > 0
    assert splits["instance_group_split"]["test_row_count"] > 0
    assert set(splits["family_holdout_splits"]) == {"holdout_expander_tseitin", "holdout_ladder_tseitin"}

    frozen = mod.freeze_splits(rows)
    assert mod.verify_split_leakage(rows, frozen)["all_splits_leakage_safe"] is True
    leaky = deepcopy(frozen)
    crossing_row = next(row for row in rows if row["base_instance_id"] in leaky["instance_group_split"]["train_groups"])
    leaky["instance_group_split"]["test_groups"].append(crossing_row["base_instance_id"])
    leak_receipt = mod.verify_split_leakage(rows, leaky)
    assert leak_receipt["all_splits_leakage_safe"] is False
    assert crossing_row["base_instance_id"] in leak_receipt["duplicate_semantic_instances_across_splits"]


def test_scenario_verify_5869_controls_mark_solver_saturation_as_circular_skip(
    exp5869_audit: tuple[dict[str, Any], list[dict[str, Any]], Path],
) -> None:
    """SCENARIO-VERIFY-5869-CONTROLS: solver-backed saturation blocks readiness."""

    artifact, _rows, _base = exp5869_audit
    controls = artifact["density_length_width_name_and_order_controls"]
    solver = artifact["solver_hardness_vs_label_analysis"]
    no_info = artifact["shuffled_and_majority_controls"]
    circular = artifact["current_verifier_circularity_matrix"]
    decision = artifact["saturation_and_skip_decision"]

    assert controls["label_feature_used"] is False
    assert controls["saturation_ceiling_auroc"] == pytest.approx(mod.SATURATION_CEILING_AUROC)
    assert controls["max_non_oracle_control_auroc"] <= mod.SATURATION_CEILING_AUROC
    assert controls["max_all_trivial_control_auroc"] > mod.SATURATION_CEILING_AUROC
    assert "solver_conflicts" in controls["saturated_control_names"]
    assert solver["solver_conflicts_saturated"] is True
    assert solver["solver_time_saturated"] is True
    assert no_info["majority_control"]["balanced_error_rate"] == pytest.approx(0.5)
    assert no_info["shuffled_label_control"]["uses_shuffled_labels"] is True
    assert circular["all_exact_paths_marked_oracle"] is True
    assert circular["oracle_accuracy_reduces_headroom"] is False
    assert all(path["verifier_is_oracle"] for path in circular["paths"].values())
    assert decision["skip_model_extraction"] is True
    assert decision["hardness_surface_headroom_ready_score"] == 0.0


def test_scenario_verify_5869_design_remains_nonempty_while_blockers_are_honest(
    tmp_path: Path,
    exp5869_audit: tuple[dict[str, Any], list[dict[str, Any]], Path],
) -> None:
    """SCENARIO-VERIFY-5869-DESIGN: follow-up design exists and failures close gates."""

    artifact, rows, _base = exp5869_audit
    design = artifact["oracle_distinct_evaluation_design"]
    held = artifact["held_family_and_constraint_cell_plan"]

    assert design["nonempty_held_model_and_constraint_design"] is True
    assert design["future_signal_source"] == "internal_state_or_learned_energy_score"
    assert design["exact_release_authority_separate"] is True
    assert held["whole_family_holdouts"] == ["expander_tseitin", "ladder_tseitin"]
    assert held["constraint_cell_count"] > 0

    blocked = mod.run(
        root=tmp_path,
        result_path=tmp_path / mod.RESULT_RELATIVE_PATH.name,
        test_commands=TEST_COMMANDS,
        test_exit_codes=TEST_EXIT_CODES,
        duration_s=0.0,
        write=True,
    )
    assert blocked["status"] == "blocked"
    assert blocked["honest_verdict"].startswith("blocked:")
    assert blocked["hardness_surface_headroom_ready_score"] == 0.0

    failed_command = deepcopy(artifact)
    failed_command["test_exit_codes"][TEST_COMMAND] = 1
    failed_command["reproducibility_checksum"] = mod.reproducibility_checksum(failed_command)
    assert mod.hardness_surface_headroom_ready_score(failed_command) == 0.0
    with pytest.raises(ValueError, match="test_exit_codes"):
        mod.validate_artifact(failed_command)

    checksum_bad = deepcopy(artifact)
    checksum_bad["honest_verdict"] = "complete_null: edited"
    with pytest.raises(ValueError, match="reproducibility_checksum"):
        mod.validate_artifact(checksum_bad)
    missing = deepcopy(artifact)
    missing.pop("status")
    with pytest.raises(ValueError, match="missing_fields"):
        mod.validate_artifact(missing)

    duplicate_rows = [deepcopy(rows[0]), deepcopy(rows[0])]
    replay = mod.independent_row_integrity_replay(duplicate_rows, artifact)
    assert replay["duplicate_row_id_count"] == 1
    assert replay["all_integrity_checks_passed"] is False


def test_scenario_verify_5869_defensive_branches_are_explicit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    exp5869_audit: tuple[dict[str, Any], list[dict[str, Any]], Path],
) -> None:
    """REQ-VERIFY-5869: malformed inputs and terminal-state edits fail closed."""

    artifact, rows, _base = exp5869_audit
    row = deepcopy(rows[0])

    bad_json = tmp_path / "bad.json"
    bad_json.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="JSON object required"):
        mod._read_json(bad_json)
    bad_jsonl = tmp_path / "bad.rows.jsonl"
    bad_jsonl.write_text("\n[]\n", encoding="utf-8")
    with pytest.raises(ValueError, match="JSONL object required"):
        mod._read_jsonl(bad_jsonl)
    blank_jsonl = tmp_path / "blank.rows.jsonl"
    blank_jsonl.write_text("\n" + json.dumps(row) + "\n", encoding="utf-8")
    assert len(mod._read_jsonl(blank_jsonl)) == 1

    corrupt_root = tmp_path / "corrupt"
    (corrupt_root / mod.UPSTREAM_ARTIFACT_RELATIVE_PATH.parent).mkdir(parents=True)
    (corrupt_root / mod.UPSTREAM_ARTIFACT_RELATIVE_PATH).write_text("{", encoding="utf-8")
    (corrupt_root / mod.UPSTREAM_ROWS_RELATIVE_PATH).write_text("{}", encoding="utf-8")
    assert mod.upstream_gate_receipt(corrupt_root)["blocked_reason"].startswith(
        "corrupt_upstream_exp5868"
    )

    no_charge = {"clauses": [[1]], "n_vars": 1}
    assert mod.recompute_label(no_charge) == "satisfiable"
    invalid_label = deepcopy(row)
    invalid_label["expected_label"] = "invalid"
    invalid_label["certificate"]["validated"] = True
    assert mod.recompute_certificate_validity(invalid_label) is False

    bad_relabel = deepcopy(row)
    bad_relabel["proof_preserving_relabel"]["variable_map"].pop("1")
    assert mod.recompute_relabel_equivalence(bad_relabel) is False
    bad_relabel = deepcopy(row)
    first_key = next(iter(bad_relabel["proof_preserving_relabel"]["variable_map"]))
    bad_relabel["proof_preserving_relabel"]["variable_map"][first_key] = 1
    assert mod.recompute_relabel_equivalence(bad_relabel) is False
    bad_relabel = deepcopy(row)
    first_solver = next(iter(bad_relabel["proof_preserving_relabel"]["solver_results"]))
    bad_relabel["proof_preserving_relabel"]["solver_results"][first_solver]["label"] = "bad"
    assert mod.recompute_relabel_equivalence(bad_relabel) is False
    bad_relabel = deepcopy(row)
    bad_relabel["proof_preserving_relabel"]["label_preserved"] = False
    assert mod.recompute_relabel_equivalence(bad_relabel) is False

    assert mod._auc([0.1, 0.2], [0, 0]) == 0.5
    assert mod._balanced_error([0, 1], [0, 0]) == 0.5

    substrate_bad = deepcopy(artifact)
    substrate_bad["inference_substrate"] = "bad"
    substrate_bad["verifier_is_oracle"] = False
    reasons = mod.blocked_reasons(substrate_bad)
    assert "inference_substrate" in reasons
    assert "verifier_is_oracle" in reasons

    ready = deepcopy(artifact)
    ready["density_length_width_name_and_order_controls"]["saturated_control_names"] = []
    ready["saturation_and_skip_decision"]["saturated_control_names"] = []
    ready["saturation_and_skip_decision"]["no_trivial_control_exceeds_ceiling"] = True
    ready["saturation_and_skip_decision"]["skip_model_extraction"] = False
    ready["saturation_and_skip_decision"]["skip_reason"] = ""
    ready["hardness_surface_headroom_ready_score"] = mod.hardness_surface_headroom_ready_score(
        ready
    )
    assert ready["hardness_surface_headroom_ready_score"] == 1.0
    ready["status"] = "complete_ready"
    ready["honest_verdict"] = mod.honest_verdict(ready)
    ready["saturation_and_skip_decision"]["hardness_surface_headroom_ready_score"] = 1.0
    ready["reproducibility_checksum"] = mod.reproducibility_checksum(ready)
    assert mod.validate_artifact(ready) is True
    ready_status_bad = deepcopy(ready)
    ready_status_bad["status"] = "complete_null"
    ready_status_bad["reproducibility_checksum"] = mod.reproducibility_checksum(ready_status_bad)
    with pytest.raises(ValueError, match="status"):
        mod.validate_artifact(ready_status_bad)
    ready_verdict_bad = deepcopy(ready)
    ready_verdict_bad["honest_verdict"] = "complete_null: bad"
    ready_verdict_bad["reproducibility_checksum"] = mod.reproducibility_checksum(ready_verdict_bad)
    with pytest.raises(ValueError, match="honest_verdict"):
        mod.validate_artifact(ready_verdict_bad)

    ready_controls = deepcopy(artifact["density_length_width_name_and_order_controls"])
    ready_controls["saturated_control_names"] = []
    ready_controls["non_oracle_controls_saturated"] = []
    ready_controls["max_all_trivial_control_auroc"] = 0.5
    for control in ready_controls["controls"].values():
        control["saturated"] = False
    monkeypatch.setattr(mod, "evaluate_trivial_controls", lambda _rows, _split: ready_controls)
    built_ready = mod.build_artifact(
        rows=rows,
        preconditions_checked=artifact["preconditions_checked"],
        test_commands=TEST_COMMANDS,
        test_exit_codes=TEST_EXIT_CODES,
        duration_s=1.0,
    )
    assert built_ready["status"] == "complete_ready"
    assert built_ready["hardness_surface_headroom_ready_score"] == 1.0

    score_bad = deepcopy(artifact)
    score_bad["hardness_surface_headroom_ready_score"] = 1.0
    score_bad["reproducibility_checksum"] = mod.reproducibility_checksum(score_bad)
    with pytest.raises(ValueError, match="hardness_surface_headroom_ready_score"):
        mod.validate_artifact(score_bad)
    null_status_bad = deepcopy(artifact)
    null_status_bad["status"] = "complete_ready"
    null_status_bad["reproducibility_checksum"] = mod.reproducibility_checksum(null_status_bad)
    with pytest.raises(ValueError, match="status"):
        mod.validate_artifact(null_status_bad)
    null_verdict_bad = deepcopy(artifact)
    null_verdict_bad["honest_verdict"] = "blocked: bad"
    null_verdict_bad["reproducibility_checksum"] = mod.reproducibility_checksum(null_verdict_bad)
    with pytest.raises(ValueError, match="honest_verdict"):
        mod.validate_artifact(null_verdict_bad)

    blocked = mod.run(
        root=tmp_path,
        result_path=tmp_path / "blocked" / mod.RESULT_RELATIVE_PATH.name,
        test_commands=TEST_COMMANDS,
        test_exit_codes=TEST_EXIT_CODES,
        duration_s=0.0,
        write=True,
    )
    assert mod.validate_artifact(blocked) is True
    blocked_status_bad = deepcopy(blocked)
    blocked_status_bad["status"] = "complete_null"
    blocked_status_bad["reproducibility_checksum"] = mod.reproducibility_checksum(
        blocked_status_bad
    )
    with pytest.raises(ValueError, match="status"):
        mod.validate_artifact(blocked_status_bad)
    blocked_verdict_bad = deepcopy(blocked)
    blocked_verdict_bad["honest_verdict"] = "complete_null: bad"
    blocked_verdict_bad["reproducibility_checksum"] = mod.reproducibility_checksum(
        blocked_verdict_bad
    )
    with pytest.raises(ValueError, match="honest_verdict"):
        mod.validate_artifact(blocked_verdict_bad)
