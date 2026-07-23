"""Tests for Exp5856 provenance-correct lifecycle replay.

Spec refs: REQ-LEARN-5856, SCENARIO-LEARN-5856-CHRONOLOGY,
SCENARIO-LEARN-5856-MATCHED-ARMS, SCENARIO-LEARN-5856-READY-GATE,
SCENARIO-LEARN-5856-FAIL-CLOSED.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any

import pytest

from carnot import experiment_5856_provenance_correct_lifecycle as mod


REPO = Path(__file__).resolve().parents[2]
SELF_LEARNING_SPEC = REPO / "openspec/capabilities/self-learning/spec.md"
TEST_COMMAND = (
    ".venv/bin/pytest tests/python/test_experiment_5856_provenance_correct_lifecycle.py "
    "-q --no-cov -n 0"
)
COVERAGE_COMMAND = (
    ".venv/bin/coverage run --rcfile=/dev/null "
    "--include=python/carnot/experiment_5856_provenance_correct_lifecycle.py "
    "-m pytest tests/python/test_experiment_5856_provenance_correct_lifecycle.py "
    "-q --no-cov -n 0 && .venv/bin/coverage report --rcfile=/dev/null "
    "--include=python/carnot/experiment_5856_provenance_correct_lifecycle.py "
    "--fail-under=100"
)
FULL_TEST_COMMAND = ".venv/bin/pytest tests/python -q"
SPEC_COMMAND = ".venv/bin/python scripts/check_spec_coverage.py"
ADVERSARIAL_COMMAND = (
    ".venv/bin/python scripts/adversarial_verify.py "
    "results/experiment_5856_provenance_correct_lifecycle.json"
)
ROOT_CLUTTER_COMMAND = ".venv/bin/python scripts/root_clutter_sweep.py"
PROTECTED_FILE_COMMAND = (
    '.venv/bin/python -c "from pathlib import Path; '
    "assert Path('scripts/research_conductor.py').exists()\""
)
TEST_COMMANDS = [
    TEST_COMMAND,
    COVERAGE_COMMAND,
    FULL_TEST_COMMAND,
    SPEC_COMMAND,
    ADVERSARIAL_COMMAND,
    ROOT_CLUTTER_COMMAND,
    PROTECTED_FILE_COMMAND,
]
TEST_EXIT_CODES = {command: 0 for command in TEST_COMMANDS}
PASSING_VERIFIER_RECEIPT = {
    "artifact": mod.RESULT_RELATIVE_PATH.as_posix(),
    "loaded": True,
    "exp_id": 5856,
    "title": "",
    "honest_verdict": "complete: fixture",
    "flag_count": 0,
    "max_severity": -1,
    "flags": [],
    "exit_code": 0,
}


def _preconditions(tmp_path: Path) -> dict[str, Any]:
    return mod.collect_preconditions(
        result_path=tmp_path / mod.RESULT_RELATIVE_PATH.name,
        row_path=tmp_path / mod.ROW_RELATIVE_PATH.name,
        memory_probe=lambda: {
            "available_mb": 8192,
            "required_mb": mod.RAM_FLOOR_MB,
            "ok": True,
        },
        disk_probe=lambda root: {
            "available_mb": 8192,
            "required_mb": mod.DISK_FLOOR_MB,
            "ok": True,
        },
    )


@pytest.fixture(scope="module")
def artifact(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    """REQ-LEARN-5856: build the deterministic provenance-correct artifact once."""

    base = tmp_path_factory.mktemp("exp5856")
    return mod.run(
        result_path=base / mod.RESULT_RELATIVE_PATH.name,
        row_path=base / mod.ROW_RELATIVE_PATH.name,
        preconditions_checked=_preconditions(base),
        duration_s=1.5,
        test_commands=TEST_COMMANDS,
        test_exit_codes=TEST_EXIT_CODES,
        adversarial_verifier_receipt=PASSING_VERIFIER_RECEIPT,
        write=True,
    )


def test_req_learn_5856_spec_declares_required_fields() -> None:
    """REQ-LEARN-5856: OpenSpec declares the replay contract and principles."""

    text = SELF_LEARNING_SPEC.read_text(encoding="utf-8")
    section = text[text.index("## REQ-LEARN-5856") :]
    normalized = " ".join(section.split())

    for marker in (
        "REQ-LEARN-5856",
        "SCENARIO-LEARN-5856-CHRONOLOGY",
        "SCENARIO-LEARN-5856-MATCHED-ARMS",
        "SCENARIO-LEARN-5856-READY-GATE",
        "SCENARIO-LEARN-5856-FAIL-CLOSED",
        mod.RESULT_RELATIVE_PATH.as_posix(),
        mod.ROW_RELATIVE_PATH.as_posix(),
        mod.INFERENCE_SUBSTRATE,
        "`adaptive_memory_lifecycle_ready_score`",
    ):
        assert marker in section
    for field, principle in mod.REQUIRED_FIELD_PRINCIPLES.items():
        assert f"`{field}`" in section
        assert " ".join(principle.split()) in normalized


def test_req_learn_5856_terminal_artifact_schema_and_immutability(
    artifact: dict[str, Any],
    tmp_path: Path,
) -> None:
    """REQ-LEARN-5856: the artifact is complete, deterministic, and immutable upstream."""

    exp5828_before = mod.sha256_file(REPO / mod.EXP5828_ARTIFACT_RELATIVE_PATH)
    exp5839_before = mod.sha256_file(REPO / mod.EXP5839_ARTIFACT_RELATIVE_PATH)
    destination = tmp_path / mod.RESULT_RELATIVE_PATH.name
    row_destination = tmp_path / mod.ROW_RELATIVE_PATH.name
    replay = mod.run(
        result_path=destination,
        row_path=row_destination,
        preconditions_checked=_preconditions(tmp_path),
        duration_s=1.5,
        test_commands=TEST_COMMANDS,
        test_exit_codes=TEST_EXIT_CODES,
        adversarial_verifier_receipt=PASSING_VERIFIER_RECEIPT,
        write=True,
    )
    loaded = json.loads(destination.read_text(encoding="utf-8"))
    row_receipts = mod.read_row_receipts(row_destination)

    assert replay == loaded
    assert artifact["reproducibility_checksum"] == replay["reproducibility_checksum"]
    assert artifact["row_file_receipt"]["sha256"] == replay["row_file_receipt"]["sha256"]
    assert destination.read_text(encoding="utf-8").endswith("\n")
    assert row_destination.read_text(encoding="utf-8").endswith("\n")
    assert mod.validate_artifact(artifact) is True
    assert set(mod.REQUIRED_ARTIFACT_FIELDS) <= set(artifact)
    assert artifact["status"] == "complete"
    assert artifact["honest_verdict"].startswith("complete:")
    assert artifact["adaptive_memory_lifecycle_ready_score"] == pytest.approx(1.0)
    assert isinstance(artifact["adaptive_memory_lifecycle_ready_score"], float)
    assert artifact["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert artifact["verifier_is_oracle"] is True
    assert artifact["no_model_weight_mutation"] is True
    assert artifact["historical_artifacts_mutated"] is False
    assert artifact["preconditions_checked"]["preconditions_ready"] is True
    assert artifact["preconditions_checked"]["validators"]["ok"] is True
    assert artifact["preconditions_checked"]["splits"]["ok"] is True
    assert artifact["preconditions_checked"]["timer"]["ok"] is True
    assert artifact["deterministic_replay_contract_receipt"]["receipt"]["passed"] is True
    assert artifact["deterministic_replay_contract_receipt"]["contract_ready_score"] == 1.0
    assert artifact["adversarial_verifier_receipt"]["flag_count"] == 0
    assert artifact["row_file_receipt"]["row_count"] == len(row_receipts) == 360
    assert artifact["row_file_receipt"]["sha256"] == mod.sha256_file(row_destination)
    assert artifact["test_commands"] == TEST_COMMANDS
    assert artifact["test_exit_codes"] == TEST_EXIT_CODES
    assert artifact["reproducibility_checksum"] == mod.reproducibility_checksum(artifact)
    assert mod.sha256_file(REPO / mod.EXP5828_ARTIFACT_RELATIVE_PATH) == exp5828_before
    assert mod.sha256_file(REPO / mod.EXP5839_ARTIFACT_RELATIVE_PATH) == exp5839_before


def test_scenario_learn_5856_chronology_and_row_receipts_are_label_blind(
    artifact: dict[str, Any],
) -> None:
    """SCENARIO-LEARN-5856-CHRONOLOGY: future labels stay sealed before validation."""

    row_path = Path(artifact["row_file_receipt"]["path"])
    row_receipts = mod.read_row_receipts(row_path)
    chronology = artifact["chronology_and_visibility_receipts"]
    recomputed = mod.recompute_from_row_receipts(row_receipts)

    assert chronology["chronology_monotone"] is True
    assert chronology["future_label_leakage_count"] == 0
    assert chronology["ground_truth_cleartext_visible_count"] == 0
    assert chronology["validation_label_reuse_count"] == 0
    assert chronology["sealed_suffix_count"] == 360
    assert chronology["row_receipt_hash_root"] == mod.sha256_json(
        [receipt["row_receipt_hash"] for receipt in row_receipts]
    )
    assert recomputed["prospective_row_metrics"] == artifact["prospective_row_metrics"]
    assert (
        recomputed["family_lower_bounds_and_group_bootstraps"]
        == artifact["family_lower_bounds_and_group_bootstraps"]
    )

    for receipt in row_receipts[:12]:
        assert receipt["future_labels_visible_before_prediction"] is False
        assert receipt["cleartext_target_visible_before_prediction"] is False
        assert receipt["future_opened_after_quarantine"] is True
        assert receipt["validation_label_reuse_count"] == 0
        assert receipt["adaptive_minus_frozen_delta"] >= 0.0
        assert receipt["source_row_hash"].startswith("sha256:")
        assert receipt["row_receipt_hash"].startswith("sha256:")


def test_scenario_learn_5856_matched_arms_metrics_state_and_contract(
    artifact: dict[str, Any],
) -> None:
    """SCENARIO-LEARN-5856-MATCHED-ARMS/READY-GATE: all gates are row-derived."""

    arms = artifact["frozen_and_adaptive_arm_definitions"]
    rows = artifact["prospective_row_metrics"]
    families = artifact["family_lower_bounds_and_group_bootstraps"]
    promotion = artifact["promotion_quarantine_and_rejection_receipts"]
    restart = artifact["rollback_restart_and_serialization_receipts"]
    cap = artifact["memory_cap_accounting"]
    comparison = artifact["exp5828_scientific_metric_comparison"]

    assert arms["parity_passed"] is True
    assert arms["frozen_arm"]["external_state_mutations"] == 0
    assert arms["adaptive_arm"]["external_state_mutations"] > 0
    assert arms["identical_event_stream_hash"].startswith("sha256:")
    assert rows["row_count"] == 360
    assert rows["adaptive_accuracy"] == pytest.approx(1.0)
    assert rows["adaptive_minus_frozen"]["ci95"][0] > 0.0
    assert all(value > 0.0 for value in families["family_lcb95"].values())
    assert families["all_family_lcbs_positive"] is True
    assert families["group_bootstrap_ci95"]["ci95"][0] > 0.0
    assert artifact["protected_prefix_retention"] == pytest.approx(1.0)
    assert promotion["unsafe_accept_count"] == 0
    assert promotion["false_promotion_count"] == 0
    assert promotion["promotion_count"] == 360
    assert promotion["rejection_count"] >= len(mod.PRIMARY_FAMILIES)
    assert restart["rollback_hash_mismatch_count"] == 0
    assert restart["restart_equivalence"] == pytest.approx(1.0)
    assert restart["serialization_equivalence"] == pytest.approx(1.0)
    assert cap["max_state_size"] <= cap["memory_cap"]
    assert cap["cap_compliance"] == pytest.approx(1.0)
    assert comparison["historical_flagged_adversarial"] is True
    assert comparison["aggregate_decision_imported"] is False
    assert comparison["row_derived_ready_score"] == 1.0


def test_scenario_learn_5856_fail_closed_for_bad_gates(
    artifact: dict[str, Any],
    tmp_path: Path,
) -> None:
    """SCENARIO-LEARN-5856-FAIL-CLOSED: unsafe or dishonest evidence is never ready."""

    blocked = mod.run(
        root=tmp_path,
        result_path=tmp_path / mod.RESULT_RELATIVE_PATH.name,
        row_path=tmp_path / mod.ROW_RELATIVE_PATH.name,
        duration_s=1.5,
        test_commands=TEST_COMMANDS,
        test_exit_codes=TEST_EXIT_CODES,
        adversarial_verifier_receipt=PASSING_VERIFIER_RECEIPT,
        write=False,
    )
    assert blocked["status"] == "blocked"
    assert blocked["adaptive_memory_lifecycle_ready_score"] == 0.0
    assert blocked["honest_verdict"].startswith("blocked:")
    assert "missing_upstream_file" in blocked["preconditions_checked"]["blocked_reasons"]
    assert mod.validate_artifact(blocked) is True

    failed_tests = mod.run(
        result_path=tmp_path / "failed.json",
        row_path=tmp_path / "failed.rows.jsonl",
        preconditions_checked=_preconditions(tmp_path / "failed"),
        duration_s=1.5,
        test_commands=TEST_COMMANDS,
        test_exit_codes={**TEST_EXIT_CODES, FULL_TEST_COMMAND: 2},
        adversarial_verifier_receipt=PASSING_VERIFIER_RECEIPT,
        write=False,
    )
    assert failed_tests["status"] == "failed"
    assert "failed_test_exit_codes" in mod.blocked_reasons(failed_tests)

    verifier_failed = deepcopy(artifact)
    verifier_failed["adversarial_verifier_receipt"] = {
        **PASSING_VERIFIER_RECEIPT,
        "flag_count": 1,
        "flags": [{"kind": "fixture", "severity": "critical", "detail": "bad"}],
    }
    assert "adversarial_verifier_failed" in mod.blocked_reasons(verifier_failed)

    for mutate, match in (
        (
            lambda item: item.update({"inference_substrate": "wrong"}),
            "inference_substrate",
        ),
        (
            lambda item: item.update({"verifier_is_oracle": False}),
            "verifier_is_oracle",
        ),
        (
            lambda item: item.update({"no_model_weight_mutation": False}),
            "no_model_weight_mutation",
        ),
        (
            lambda item: item.update({"historical_artifacts_mutated": True}),
            "historical_artifacts_mutated",
        ),
        (
            lambda item: item["promotion_quarantine_and_rejection_receipts"].update(
                {"unsafe_accept_count": 1}
            ),
            "ready_score",
        ),
        (
            lambda item: item["deterministic_replay_contract_receipt"]["receipt"].update(
                {"passed": False}
            ),
            "ready_score",
        ),
        (
            lambda item: item.update({"reproducibility_checksum": mod.sha256_text("wrong")}),
            "reproducibility_checksum",
        ),
    ):
        bad = deepcopy(artifact)
        mutate(bad)
        with pytest.raises(ValueError, match=match):
            mod.validate_artifact(bad)


def test_req_learn_5856_helper_edges_and_provenance_validation(
    artifact: dict[str, Any],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-LEARN-5856: helper functions reject malformed inputs deterministically."""

    scalar_json = tmp_path / "scalar.json"
    scalar_json.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="JSON object required"):
        mod._read_json(scalar_json)

    blank_jsonl = tmp_path / "blank.jsonl"
    blank_jsonl.write_text('\n{"ok": true}\n', encoding="utf-8")
    scalar_jsonl = tmp_path / "scalar.jsonl"
    scalar_jsonl.write_text("1\n", encoding="utf-8")
    assert mod._read_jsonl(blank_jsonl) == [{"ok": True}]
    with pytest.raises(ValueError, match="JSONL object required"):
        mod._read_jsonl(scalar_jsonl)
    assert mod.read_row_receipts(tmp_path / "missing.rows.jsonl") == []
    assert mod.fixture_preconditions(tmp_path)["preconditions_ready"] is True
    assert mod._paired_summary([])["ci95"] == [0.0, 0.0]
    assert mod._paired_summary([0.25])["ci95"] == [0.25, 0.25]
    assert mod._group_bootstrap_ci95([], "family") == {"n_groups": 0, "ci95": [0.0, 0.0]}
    assert mod._historical_artifacts_mutated({}, REPO) is False

    with monkeypatch.context() as scoped:
        scoped.setattr(
            mod,
            "_read_json",
            lambda path: (_ for _ in ()).throw(ValueError("corrupt")),
        )
        corrupt = mod.collect_preconditions(
            result_path=tmp_path / "corrupt.json",
            row_path=tmp_path / "corrupt.rows.jsonl",
            memory_probe=lambda: {
                "available_mb": 8192,
                "required_mb": mod.RAM_FLOOR_MB,
                "ok": True,
            },
            disk_probe=lambda root: {
                "available_mb": 8192,
                "required_mb": mod.DISK_FLOOR_MB,
                "ok": True,
            },
        )
    assert corrupt["blocked_errors"] == ["ValueError"]
    assert "corrupt_upstream_json" in corrupt["blocked_reasons"]

    missing = deepcopy(artifact)
    missing.pop("status")
    with pytest.raises(ValueError, match="missing required artifact fields"):
        mod.validate_artifact(missing)

    provenance_not_mapping = deepcopy(artifact)
    provenance_not_mapping["field_provenance"] = []
    with pytest.raises(ValueError, match="field_provenance"):
        mod.validate_artifact(provenance_not_mapping)

    provenance_bad = deepcopy(artifact)
    provenance_bad["field_provenance"]["status"]["principle"] = "wrong"
    with pytest.raises(ValueError, match="field_provenance:status"):
        mod.validate_artifact(provenance_bad)

    invalid_status = deepcopy(artifact)
    invalid_status["status"] = "blocked"
    invalid_status["reproducibility_checksum"] = mod.reproducibility_checksum(invalid_status)
    with pytest.raises(ValueError, match="status"):
        mod.validate_artifact(invalid_status)

    wrong_verdict = deepcopy(artifact)
    wrong_verdict["honest_verdict"] = "complete: wrong"
    wrong_verdict["reproducibility_checksum"] = mod.reproducibility_checksum(wrong_verdict)
    with pytest.raises(ValueError, match="honest_verdict"):
        mod.validate_artifact(wrong_verdict)

    blocker_probe = deepcopy(artifact)
    blocker_probe["inference_substrate"] = "wrong"
    blocker_probe["verifier_is_oracle"] = False
    blocker_probe["no_model_weight_mutation"] = False
    blocker_probe["historical_artifacts_mutated"] = True
    blocker_probe["promotion_quarantine_and_rejection_receipts"]["unsafe_accept_count"] = 1
    blocker_probe["promotion_quarantine_and_rejection_receipts"]["false_promotion_count"] = 1
    blocker_probe["rollback_restart_and_serialization_receipts"]["rollback_hash_mismatch_count"] = 1
    assert set(mod.blocked_reasons(blocker_probe)) >= {
        "inference_substrate",
        "verifier_is_oracle",
        "no_model_weight_mutation",
        "historical_artifacts_mutated",
        "unsafe_accept_count",
        "false_promotion_count",
        "rollback_hash_mismatch_count",
    }

    with monkeypatch.context() as scoped:
        scoped.setattr(mod, "adaptive_memory_lifecycle_ready_score", lambda item: 0.0)
        assert mod.blocked_reasons(artifact) == ["ready_score"]

    recomputed = mod.recompute_from_row_receipts(
        mod.read_row_receipts(Path(artifact["row_file_receipt"]["path"]))
    )
    assert recomputed["protected_prefix_retention"] == pytest.approx(1.0)
    assert recomputed["unsafe_accept_count"] == 0
