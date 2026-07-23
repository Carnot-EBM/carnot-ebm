"""Tests for Exp5839 V519 evidence qualification.

Spec refs: REQ-LEARN-5839, SCENARIO-LEARN-5839-RECONSTRUCT,
SCENARIO-LEARN-5839-SHORTCUTS, SCENARIO-LEARN-5839-MIXED,
SCENARIO-LEARN-5839-FAIL-CLOSED.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any

import pytest

from carnot import experiment_5839_v519_evidence_qualification as mod


REPO = Path(__file__).resolve().parents[2]
SELF_LEARNING_SPEC = REPO / "openspec/capabilities/self-learning/spec.md"
TEST_COMMAND = (
    ".venv/bin/pytest "
    "tests/python/test_experiment_5839_v519_evidence_qualification.py "
    "-q --no-cov -n 0"
)
COVERAGE_COMMAND = (
    ".venv/bin/coverage run --rcfile=/dev/null "
    "--include=python/carnot/experiment_5839_v519_evidence_qualification.py "
    "-m pytest tests/python/test_experiment_5839_v519_evidence_qualification.py "
    "-q --no-cov -n 0 && .venv/bin/coverage report --rcfile=/dev/null "
    "--include=python/carnot/experiment_5839_v519_evidence_qualification.py "
    "--fail-under=100"
)
FULL_TEST_COMMAND = ".venv/bin/pytest tests/python -q"
SPEC_COMMAND = ".venv/bin/python scripts/check_spec_coverage.py"
ADVERSARIAL_COMMAND = (
    ".venv/bin/python scripts/adversarial_verify.py "
    "results/experiment_5839_v519_evidence_qualification.json"
)
ROOT_CLUTTER_COMMAND = ".venv/bin/python scripts/root_clutter_sweep.py"
PROTECTED_IMMUTABILITY_COMMAND = (
    ".venv/bin/python -c \"from pathlib import Path; "
    "assert Path('scripts/research_conductor.py').exists()\""
)
TEST_COMMANDS = [
    TEST_COMMAND,
    COVERAGE_COMMAND,
    FULL_TEST_COMMAND,
    SPEC_COMMAND,
    ADVERSARIAL_COMMAND,
    ROOT_CLUTTER_COMMAND,
    PROTECTED_IMMUTABILITY_COMMAND,
]
TEST_EXIT_CODES = {command: 0 for command in TEST_COMMANDS}
PASSING_VERIFIER_RECEIPT = {
    "artifact": mod.RESULT_RELATIVE_PATH.as_posix(),
    "loaded": True,
    "exp_id": 5839,
    "title": "",
    "honest_verdict": "mixed: fixture",
    "flag_count": 0,
    "max_severity": -1,
    "flags": [],
}


def _preconditions(tmp_path: Path) -> dict[str, Any]:
    return mod.collect_preconditions(
        result_path=tmp_path / mod.RESULT_RELATIVE_PATH.name,
        temp_reconstruction_path=tmp_path / "reconstruct",
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
    """REQ-LEARN-5839: build the deterministic qualification artifact once."""

    base = tmp_path_factory.mktemp("exp5839")
    return mod.run(
        result_path=base / mod.RESULT_RELATIVE_PATH.name,
        temp_reconstruction_path=base / "reconstruct",
        preconditions_checked=_preconditions(base),
        duration_s=1.0,
        test_commands=TEST_COMMANDS,
        test_exit_codes=TEST_EXIT_CODES,
        adversarial_verifier_receipt=PASSING_VERIFIER_RECEIPT,
        write=True,
    )


def test_req_learn_5839_spec_declares_qualification_contract() -> None:
    """REQ-LEARN-5839: OpenSpec names fields, principles, and scenarios."""

    text = SELF_LEARNING_SPEC.read_text(encoding="utf-8")
    section = text[text.index("## REQ-LEARN-5839") :]
    normalized = " ".join(section.split())

    for marker in (
        "REQ-LEARN-5839",
        "SCENARIO-LEARN-5839-RECONSTRUCT",
        "SCENARIO-LEARN-5839-SHORTCUTS",
        "SCENARIO-LEARN-5839-MIXED",
        "SCENARIO-LEARN-5839-FAIL-CLOSED",
        mod.RESULT_RELATIVE_PATH.as_posix(),
        mod.INFERENCE_SUBSTRATE,
        "`constraint_stream_qualified_score`",
        "`selective_replay_qualified_score`",
    ):
        assert marker in section
    for field, principle in mod.REQUIRED_FIELD_PRINCIPLES.items():
        assert f"`{field}`" in section
        assert " ".join(principle.split()) in normalized


def test_req_learn_5839_terminal_artifact_is_mixed_and_hash_bound(
    artifact: dict[str, Any],
    tmp_path: Path,
) -> None:
    """REQ-LEARN-5839: terminal output preserves clean and tainted branches separately."""

    destination = tmp_path / mod.RESULT_RELATIVE_PATH.name
    replay = mod.run(
        result_path=destination,
        temp_reconstruction_path=tmp_path / "reconstruct",
        preconditions_checked=_preconditions(tmp_path),
        duration_s=1.0,
        test_commands=TEST_COMMANDS,
        test_exit_codes=TEST_EXIT_CODES,
        adversarial_verifier_receipt=PASSING_VERIFIER_RECEIPT,
        write=True,
    )
    loaded = json.loads(destination.read_text(encoding="utf-8"))

    assert artifact == replay == loaded
    assert mod.validate_artifact(artifact) is True
    assert set(mod.REQUIRED_ARTIFACT_FIELDS).issubset(artifact)
    assert artifact["status"] == "complete"
    assert artifact["honest_verdict"].startswith("mixed:")
    assert artifact["constraint_stream_qualified_score"] == 1.0
    assert artifact["structural_acquisition_qualified_score"] == 1.0
    assert artifact["adaptive_memory_lifecycle_qualified_score"] == 0.0
    assert artifact["selective_replay_qualified_score"] == 0.0
    assert artifact["historical_artifacts_mutated"] is False
    assert artifact["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert artifact["verifier_is_oracle"] is True
    assert artifact["adversarial_verifier_receipt"]["flag_count"] == 0
    assert artifact["preconditions_checked"]["preconditions_ready"] is True
    assert artifact["preconditions_checked"]["temp_reconstruction_path"]["clean"] is True
    assert artifact["upstream_artifact_hashes"]["exp5826_stream_rows"].startswith("sha256:")
    assert artifact["test_commands"] == TEST_COMMANDS
    assert artifact["test_exit_codes"] == TEST_EXIT_CODES
    assert artifact["reproducibility_checksum"] == mod.reproducibility_checksum(artifact)


def test_scenario_learn_5839_reconstructs_immutable_rows_without_aggregate_trust(
    artifact: dict[str, Any],
) -> None:
    """SCENARIO-LEARN-5839-RECONSTRUCT: Exp5826 rows drive row qualification."""

    rows = mod.read_row_file(REPO / mod.EXP5826_ROWS_RELATIVE_PATH)
    stream = mod._read_json(REPO / mod.EXP5826_ARTIFACT_RELATIVE_PATH)
    reconstruction = mod.independent_row_reconstruction(rows, stream)
    chronology = mod.chronology_and_visibility_audit(rows)

    assert reconstruction == artifact["independent_row_reconstruction"]
    assert reconstruction["row_count"] == 360
    assert reconstruction["row_hash_mismatch_count"] == 0
    assert reconstruction["row_file_sha256_ok"] is True
    assert reconstruction["canonical_event_count"] == 2160
    assert reconstruction["canonical_state_count"] == 4320
    assert reconstruction["source_aggregate_metrics_imported"] is False
    assert chronology["chronology_monotone"] is True
    assert chronology["family_change_balance_ok"] is True
    assert chronology["future_labels_visible_to_learner_count"] == 0
    assert chronology["train_dev_science_visibility_ok"] is True

    tampered_hash_rows = deepcopy(rows)
    tampered_hash_rows[0]["row_hash"] = mod.sha256_text("tampered")
    assert mod.independent_row_reconstruction(tampered_hash_rows, stream)[
        "row_hash_mismatch_count"
    ] == 1

    non_monotone_rows = deepcopy(rows)
    non_monotone_rows[1]["chronology_index"] = 42
    assert mod.chronology_and_visibility_audit(non_monotone_rows)["chronology_monotone"] is False


def test_scenario_learn_5839_controls_and_validator_independence_fail_closed() -> None:
    """SCENARIO-LEARN-5839-SHORTCUTS: controls detect leakage and shortcut paths."""

    rows = mod.read_row_file(REPO / mod.EXP5826_ROWS_RELATIVE_PATH)
    validators = mod.exact_validator_independence(rows)
    controls = mod.shortcut_and_no_information_controls(rows)

    assert validators["validators_agree_count"] == 360
    assert validators["validator_disagreement_count"] == 0
    assert validators["solver_independence_passed"] is True
    assert validators["primary_validator_versions"] == [
        "exp5826_primary_finite_domain_exact_validator_v1"
    ]
    assert validators["independent_validator_versions"] == [
        "exp5826_independent_reversed_domain_validator_v1"
    ]
    assert controls["surviving_shortcut_count"] == 0
    assert controls["all_controls_passed"] is True
    assert controls["label_permutation"]["control_detected"] is True
    assert controls["target_preserving_feature_perturbation"]["target_preserved"] is True
    assert controls["target_derived_feature_ablation"]["qualified_without_target_derived_features"] is True
    assert controls["signature_collision"]["collision_rejected"] is True
    assert controls["future_label_access"]["future_label_access_rejected"] is True
    assert controls["no_information"]["qualified"] is False
    assert controls["duplicate_row_weighting"]["duplicate_weighting_changed_decision"] is False

    leaked = deepcopy(rows)
    leaked[0]["sealed_future_suffix"]["future_labels_visible_to_learner"] = True
    leaked_controls = mod.shortcut_and_no_information_controls(leaked)
    assert leaked_controls["surviving_shortcut_count"] == 1
    assert leaked_controls["all_controls_passed"] is False


def test_scenario_learn_5839_recomputed_metrics_and_branch_matrix_are_separate(
    artifact: dict[str, Any],
) -> None:
    """SCENARIO-LEARN-5839-MIXED: lifecycle taint does not launder into replay."""

    metrics = artifact["recomputed_metrics"]
    matrix = artifact["promotion_eligibility_matrix"]

    assert metrics["constraint_stream"]["ready_score"] == 1.0
    assert metrics["constraint_stream"]["minimum_cell_count"] == 30
    assert metrics["structural_acquisition"]["pooled_delta"]["ci95"][0] > 0.0
    assert metrics["structural_acquisition"]["family_lower_bounds"]["finite_domain_csp"] > 0.0
    assert metrics["adaptive_memory_lifecycle"]["raw_recomputed_ready_score"] == 1.0
    assert metrics["adaptive_memory_lifecycle"]["qualified_after_provenance"] == 0.0
    assert metrics["selective_replay"]["raw_recomputed_ready_score"] == 1.0
    assert metrics["selective_replay"]["qualified_after_provenance"] == 0.0
    assert metrics["selective_replay"]["inherits_flagged_lifecycle_upstream"] is True

    assert artifact["state_rollback_restart_receipts"]["lifecycle"]["rollback_hash_mismatch_count"] == 0
    assert artifact["state_rollback_restart_receipts"]["lifecycle"]["restart_equivalence"] == 1.0
    assert artifact["state_rollback_restart_receipts"]["replay"]["restart_equivalence"] == 1.0

    assert matrix["constraint_stream"]["class"] == "qualified_clean"
    assert matrix["structural_acquisition"]["class"] == "qualified_clean"
    assert matrix["adaptive_memory_lifecycle"]["class"] == "disqualified_flagged_upstream"
    assert matrix["selective_replay"]["class"] == "provisional_flagged_upstream"
    assert matrix["exp5840"]["eligible"] is True
    assert matrix["exp5843"]["eligible"] is False
    assert matrix["exp5846"]["eligible"] is False


def test_scenario_learn_5839_fail_closed_for_missing_evidence_and_bad_receipts(
    artifact: dict[str, Any],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SCENARIO-LEARN-5839-FAIL-CLOSED: blockers and tampering remove credit."""

    blocked = mod.run(
        root=tmp_path,
        result_path=tmp_path / mod.RESULT_RELATIVE_PATH.name,
        temp_reconstruction_path=tmp_path / "reconstruct",
        duration_s=1.0,
        test_commands=TEST_COMMANDS,
        test_exit_codes=TEST_EXIT_CODES,
        adversarial_verifier_receipt=PASSING_VERIFIER_RECEIPT,
        write=True,
    )
    assert blocked["status"] == "blocked"
    assert blocked["honest_verdict"].startswith("blocked:")
    assert blocked["constraint_stream_qualified_score"] == 0.0
    assert "missing_upstream_artifact" in blocked["preconditions_checked"]["blocked_reasons"]
    assert mod.validate_artifact(blocked) is True

    flagged_receipt = {
        **PASSING_VERIFIER_RECEIPT,
        "flag_count": 1,
        "max_severity": 2,
        "flags": [{"kind": "TEST", "severity": "critical", "detail": "fixture"}],
    }
    verifier_blocked = mod.build_artifact(
        preconditions_checked=_preconditions(tmp_path / "verifier"),
        duration_s=1.0,
        test_commands=TEST_COMMANDS,
        test_exit_codes=TEST_EXIT_CODES,
        adversarial_verifier_receipt=flagged_receipt,
    )
    assert verifier_blocked["status"] == "blocked"
    assert verifier_blocked["constraint_stream_qualified_score"] == 0.0
    assert "adversarial_verifier_failed" in mod.blocked_reasons(verifier_blocked)

    failed_exits = mod.build_artifact(
        preconditions_checked=_preconditions(tmp_path / "failed"),
        duration_s=1.0,
        test_commands=TEST_COMMANDS,
        test_exit_codes={**TEST_EXIT_CODES, FULL_TEST_COMMAND: 2},
        adversarial_verifier_receipt=PASSING_VERIFIER_RECEIPT,
    )
    assert failed_exits["status"] == "blocked"
    assert "failed_test_exit_codes" in mod.blocked_reasons(failed_exits)

    for mutate, match in (
        (lambda item: item.update({"historical_artifacts_mutated": True}), "historical_artifacts_mutated"),
        (lambda item: item.update({"inference_substrate": "aggregation_from_upstream_artifacts"}), "inference_substrate"),
        (lambda item: item.update({"verifier_is_oracle": False}), "verifier_is_oracle"),
        (lambda item: item.update({"constraint_stream_qualified_score": 0.0}), "constraint_stream_qualified_score"),
        (
            lambda item: item["field_provenance"]["status"].update({"principle": "wrong"}),
            "field_provenance:status",
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

    missing = deepcopy(artifact)
    missing.pop("status")
    with pytest.raises(ValueError, match="missing required artifact fields"):
        mod.validate_artifact(missing)

    invalid_provenance_shape = deepcopy(artifact)
    invalid_provenance_shape["field_provenance"] = []
    with pytest.raises(ValueError, match="field_provenance"):
        mod.validate_artifact(invalid_provenance_shape)

    invalid_status = deepcopy(artifact)
    invalid_status["status"] = "blocked"
    with pytest.raises(ValueError, match="status"):
        mod.validate_artifact(invalid_status)

    invalid_prefix = deepcopy(artifact)
    invalid_prefix["honest_verdict"] = "complete: not allowed"
    with pytest.raises(ValueError, match="honest_verdict"):
        mod.validate_artifact(invalid_prefix)

    invalid_verdict = deepcopy(artifact)
    invalid_verdict["honest_verdict"] = "mixed: wrong"
    with pytest.raises(ValueError, match="honest_verdict"):
        mod.validate_artifact(invalid_verdict)

    all_blockers = deepcopy(artifact)
    all_blockers["shortcut_and_no_information_controls"]["all_controls_passed"] = False
    all_blockers["historical_artifacts_mutated"] = True
    all_blockers["inference_substrate"] = "wrong"
    all_blockers["verifier_is_oracle"] = False
    assert set(mod.blocked_reasons(all_blockers)) >= {
        "shortcut_controls_failed",
        "historical_artifacts_mutated",
        "inference_substrate",
        "verifier_is_oracle",
    }

    qualified_probe = deepcopy(artifact)
    qualified_probe["recomputed_metrics"]["adaptive_memory_lifecycle"][
        "qualified_after_provenance"
    ] = 1.0
    qualified_probe["recomputed_metrics"]["selective_replay"]["qualified_after_provenance"] = 1.0
    assert mod.honest_verdict(qualified_probe).startswith("qualified:")

    none_probe = deepcopy(artifact)
    none_probe["recomputed_metrics"]["constraint_stream"]["ready_score"] = 0.0
    none_probe["recomputed_metrics"]["structural_acquisition"]["raw_recomputed_ready_score"] = 0.0
    assert mod.honest_verdict(none_probe).startswith("disqualified:")

    with monkeypatch.context() as scoped:
        scoped.setattr(
            mod,
            "_read_json",
            lambda path: (_ for _ in ()).throw(ValueError("corrupt")),
        )
        corrupt = mod.collect_preconditions(
            result_path=tmp_path / "corrupt.json",
            temp_reconstruction_path=tmp_path / "corrupt-reconstruct",
            memory_probe=lambda: {
                "available_mb": 0,
                "required_mb": mod.RAM_FLOOR_MB,
                "ok": False,
            },
            disk_probe=lambda root: {
                "available_mb": 0,
                "required_mb": mod.DISK_FLOOR_MB,
                "ok": False,
            },
        )
    assert set(corrupt["blocked_reasons"]) >= {
        "corrupt_upstream_artifact",
        "insufficient_free_ram",
        "insufficient_free_disk",
    }

    scalar_json = tmp_path / "scalar.json"
    scalar_json.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="JSON object required"):
        mod._read_json(scalar_json)


def test_req_learn_5839_low_level_helpers_are_deterministic(tmp_path: Path) -> None:
    """REQ-LEARN-5839: helper edges are deterministic and auditable."""

    blank_jsonl = tmp_path / "blank.jsonl"
    blank_jsonl.write_text("\n{\"ok\": true}\n", encoding="utf-8")
    assert mod.read_row_file(blank_jsonl) == [{"ok": True}]

    scalar_jsonl = tmp_path / "scalar.jsonl"
    scalar_jsonl.write_text("[]\n", encoding="utf-8")
    with pytest.raises(ValueError, match="JSONL object required"):
        mod.read_row_file(scalar_jsonl)

    assert mod._round(0.123456789) == 0.123457
    assert mod._mean([]) == 0.0
    assert mod._mean([0.0, 1.0]) == 0.5
    assert mod._ci95([]) == [0.0, 0.0]
    assert mod._ci95([0.25]) == [0.25, 0.25]
    assert mod._paired_summary([])["n"] == 0
    assert mod.fixture_preconditions(tmp_path / "fixture")["preconditions_ready"] is True

    rows = mod.read_row_file(REPO / mod.EXP5826_ROWS_RELATIVE_PATH)
    artifacts = mod.load_upstream_artifacts(REPO)
    metrics = mod.recomputed_metrics(rows, artifacts)
    assert metrics["structural_acquisition"]["credited_family_count"] == 4
    assert metrics["selective_replay"]["resource_scalar"] == 1.0

    no_write = mod.run(
        result_path=tmp_path / "no-write.json",
        temp_reconstruction_path=tmp_path / "no-write-reconstruct",
        preconditions_checked=_preconditions(tmp_path / "no-write"),
        duration_s=1.0,
        test_commands=TEST_COMMANDS,
        test_exit_codes=TEST_EXIT_CODES,
        adversarial_verifier_receipt=PASSING_VERIFIER_RECEIPT,
        write=False,
    )
    assert no_write["status"] == "complete"
    assert not (tmp_path / "no-write.json").exists()
