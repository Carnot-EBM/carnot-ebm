"""Tests for Exp5857 clean-lifecycle transfer-selective replay.

Spec refs: REQ-LEARN-5857, SCENARIO-LEARN-5857-CLEAN-GATE,
SCENARIO-LEARN-5857-SIGNATURE-FREEZE, SCENARIO-LEARN-5857-THREE-ARMS,
SCENARIO-LEARN-5857-DISAGGREGATED-METRICS, SCENARIO-LEARN-5857-CONTROLS.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any

import pytest

from carnot import experiment_5857_clean_transfer_selective_replay as mod


REPO = Path(__file__).resolve().parents[2]
SELF_LEARNING_SPEC = REPO / "openspec/capabilities/self-learning/spec.md"
TEST_COMMAND = (
    ".venv/bin/pytest tests/python/test_experiment_5857_clean_transfer_selective_replay.py "
    "-q --no-cov -n 0"
)
COVERAGE_COMMAND = (
    ".venv/bin/coverage run --rcfile=/dev/null "
    "--include=python/carnot/experiment_5857_clean_transfer_selective_replay.py "
    "-m pytest tests/python/test_experiment_5857_clean_transfer_selective_replay.py "
    "-q --no-cov -n 0 && .venv/bin/coverage report --rcfile=/dev/null "
    "--include=python/carnot/experiment_5857_clean_transfer_selective_replay.py "
    "--fail-under=100"
)
FULL_TEST_COMMAND = ".venv/bin/pytest tests/python -q"
SPEC_COMMAND = (
    ".venv/bin/python scripts/check_spec_coverage.py "
    "tests/python/test_experiment_5857_clean_transfer_selective_replay.py"
)
ADVERSARIAL_COMMAND = (
    ".venv/bin/python scripts/adversarial_verify.py "
    "results/experiment_5857_clean_transfer_selective_replay.json"
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


def _preconditions(tmp_path: Path) -> dict[str, Any]:
    return mod.collect_preconditions(
        result_path=tmp_path / mod.RESULT_RELATIVE_PATH.name,
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
    """REQ-LEARN-5857: build the clean replay requalification once."""

    base = tmp_path_factory.mktemp("exp5857")
    return mod.run(
        result_path=base / mod.RESULT_RELATIVE_PATH.name,
        preconditions_checked=_preconditions(base),
        duration_s=1.75,
        test_commands=TEST_COMMANDS,
        test_exit_codes=TEST_EXIT_CODES,
        write=True,
    )


def test_req_learn_5857_spec_declares_clean_replay_contract() -> None:
    """REQ-LEARN-5857: OpenSpec names fields, principles, and scenarios."""

    text = SELF_LEARNING_SPEC.read_text(encoding="utf-8")
    section = text[text.index("## REQ-LEARN-5857") :]
    normalized = " ".join(section.split())

    for marker in (
        "REQ-LEARN-5857",
        "SCENARIO-LEARN-5857-CLEAN-GATE",
        "SCENARIO-LEARN-5857-SIGNATURE-FREEZE",
        "SCENARIO-LEARN-5857-THREE-ARMS",
        "SCENARIO-LEARN-5857-DISAGGREGATED-METRICS",
        "SCENARIO-LEARN-5857-CONTROLS",
        mod.RESULT_RELATIVE_PATH.as_posix(),
        mod.INFERENCE_SUBSTRATE,
        "`selective_replay_qualified_score`",
    ):
        assert marker in section
    for field, principle in mod.REQUIRED_FIELD_PRINCIPLES.items():
        assert f"`{field}`" in section
        assert " ".join(principle.split()) in normalized


def test_req_learn_5857_terminal_artifact_is_hash_bound_and_clean(
    artifact: dict[str, Any],
    tmp_path: Path,
) -> None:
    """REQ-LEARN-5857: the terminal artifact is deterministic and complete."""

    destination = tmp_path / mod.RESULT_RELATIVE_PATH.name
    replay = mod.run(
        result_path=destination,
        preconditions_checked=_preconditions(tmp_path),
        duration_s=1.75,
        test_commands=TEST_COMMANDS,
        test_exit_codes=TEST_EXIT_CODES,
        write=True,
    )
    loaded = json.loads(destination.read_text(encoding="utf-8"))

    assert replay == loaded
    assert artifact["reproducibility_checksum"] == replay["reproducibility_checksum"]
    assert mod.validate_artifact(artifact) is True
    assert set(mod.REQUIRED_ARTIFACT_FIELDS) <= set(artifact)
    assert artifact["status"] == "qualified"
    assert artifact["honest_verdict"].startswith("qualified:")
    assert artifact["selective_replay_qualified_score"] == pytest.approx(1.0)
    assert isinstance(artifact["selective_replay_qualified_score"], float)
    assert artifact["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert artifact["verifier_is_oracle"] is True
    assert artifact["preconditions_checked"]["preconditions_ready"] is True
    assert artifact["preconditions_checked"]["clean_lifecycle_gate"]["ok"] is True
    assert artifact["preconditions_checked"]["family_headroom_counts"]["ok"] is True
    assert artifact["preconditions_checked"]["hardness_headroom_counts"]["ok"] is True
    assert artifact["clean_lifecycle_hashes"]["exp5856_aggregate"].startswith("sha256:")
    assert artifact["clean_lifecycle_hashes"]["exp5856_rows"].startswith("sha256:")
    assert artifact["clean_lifecycle_hashes"]["exp5829_comparison_only"] is True
    assert artifact["test_commands"] == TEST_COMMANDS
    assert artifact["test_exit_codes"] == TEST_EXIT_CODES
    assert artifact["reproducibility_checksum"] == mod.reproducibility_checksum(artifact)


def test_scenario_learn_5857_signature_freeze_and_three_arm_budget_parity(
    artifact: dict[str, Any],
) -> None:
    """SCENARIO-LEARN-5857-SIGNATURE-FREEZE/THREE-ARMS: selection is prospective."""

    rows = mod.load_clean_rows(REPO)
    signatures = artifact["frozen_signature_definition"]
    arms = artifact["replay_arm_definitions_and_budget_parity"]

    assert signatures["compatibility_rule_frozen"] is True
    assert signatures["label_blind_to_future_outcomes"] is True
    assert signatures["uses_future_labels"] is False
    assert signatures["uses_family_labels"] is False
    assert signatures["uses_row_ids"] is False
    assert signatures["uses_chronology_positions"] is False
    assert signatures["uses_posthoc_metric_selection"] is False
    assert signatures["compatible_threshold"]["minimum_component_matches"] == len(
        mod.SIGNATURE_COMPONENTS
    )
    assert signatures["incompatible_threshold"]["maximum_component_mismatches"] >= 1
    assert signatures["signature_count"] == 360
    assert signatures["signature_root_hash"].startswith("sha256:")
    assert signatures["signature_definition_hash"].startswith("sha256:")

    for sample in signatures["sample_signature_receipts"][:12]:
        payload = mod.canonical_json(sample["signature"])
        assert "family" not in payload
        assert "future_label" not in payload
        assert "row_id" not in payload
        assert sample["signature_hash"].startswith("sha256:")

    same_signature = next(row for row in rows[1:] if mod.compatible_for_replay(rows[0], row))
    different_signature = next(row for row in rows if not mod.compatible_for_replay(rows[0], row))
    assert mod.compatible_for_replay(rows[0], same_signature) is True
    assert mod.compatible_for_replay(rows[0], different_signature) is False

    assert arms["arms"] == list(mod.REPLAY_ARMS)
    assert arms["budget_parity_passed"] is True
    assert arms["prior_only_selection_passed"] is True
    assert {arms["definitions"][arm]["event_budget"] for arm in mod.REPLAY_ARMS} == {
        mod.REPLAY_EVENT_CAP
    }
    assert {arms["definitions"][arm]["memory_budget"] for arm in mod.REPLAY_ARMS} == {
        mod.MEMORY_CAP
    }
    assert arms["definitions"]["no_replay"]["selection_rule"] == "select_no_prior_rows"
    assert arms["definitions"]["all_replay"]["selection_rule"] == "select_recent_prior_rows"
    assert (
        arms["definitions"]["signature_compatible_replay"]["selection_rule"]
        == "select_recent_prior_rows_matching_frozen_signature"
    )
    for receipt in arms["sample_replay_receipts"][:24]:
        assert receipt["all_selected_rows_prior"] is True
        assert receipt["future_suffix_rows_selected"] == 0
        assert receipt["replay_count"] <= mod.REPLAY_EVENT_CAP


def test_scenario_learn_5857_disaggregated_metrics_controls_and_resources(
    artifact: dict[str, Any],
) -> None:
    """SCENARIO-LEARN-5857-DISAGGREGATED-METRICS/CONTROLS: credit is gated."""

    transfer = artifact["forward_transfer_and_recurrence"]
    retention = artifact["protected_prefix_and_hard_case_results"]
    families = artifact["family_lower_bounds_and_group_bootstraps"]
    incompatible = artifact["incompatible_negative_transfer"]
    controls = artifact["signature_permutation_collision_and_null_controls"]
    resources = artifact["replay_resource_accounting"]
    restart = artifact["restart_equivalence"]

    assert transfer["row_count"] == 360
    assert transfer["compatible_minus_no_replay"]["ci95"][0] > 0.0
    assert transfer["compatible_minus_all_replay"]["ci95"][0] > 0.0
    assert transfer["recurrence"]["compatible_minus_no_replay"]["ci95"][0] > 0.0
    assert transfer["recurrence"]["compatible_minus_all_replay"]["ci95"][0] > 0.0
    assert transfer["all_required_lower_bounds_positive"] is True
    assert transfer["arm_metrics"]["signature_compatible_replay"]["accuracy"] > transfer[
        "arm_metrics"
    ]["no_replay"]["accuracy"]
    assert transfer["arm_metrics"]["signature_compatible_replay"]["accuracy"] > transfer[
        "arm_metrics"
    ]["all_replay"]["accuracy"]

    assert retention["protected_prefix_retention"]["signature_compatible_replay"] == pytest.approx(
        1.0
    )
    assert retention["hard_case_results"]["hard"]["compatible_minus_no_replay"]["ci95"][0] >= 0.0
    assert retention["hard_case_results"]["hard"]["compatible_minus_all_replay"]["ci95"][0] >= 0.0
    assert retention["no_hard_case_negative_lower_bound"] is True
    assert families["all_family_lcbs_positive_over_both_controls"] is True
    assert families["no_family_negative_lower_bound"] is True
    assert families["group_bootstrap_ci95"]["ci95"][0] > 0.0

    assert incompatible["compatible_replay_incompatible_event_count"] == 0
    assert incompatible["all_replay_incompatible_event_count"] > 0
    assert incompatible["all_replay_incompatible_negative_transfer_event_count"] > 0
    assert incompatible["mean_incompatible_event_penalty"] < 0.0
    assert incompatible["replay_precision_recall"]["signature_compatible_replay"][
        "precision"
    ] == pytest.approx(1.0)
    assert incompatible["replay_precision_recall"]["all_replay"]["precision"] < 1.0

    assert controls["all_controls_fail_closed"] is True
    for name in (
        "signature_permutation",
        "collision_injection",
        "all_compatible",
        "none_compatible",
        "duplicate_row",
        "future_label_derived",
    ):
        assert controls[name]["qualified_score"] == pytest.approx(0.0)
    assert controls["future_label_derived"]["forbidden"] is True
    assert artifact["unsafe_transfer_count"] == 0
    assert resources["cap_compliance"] is True
    assert resources["max_cap_pressure"] <= 1.0
    assert resources["by_arm"]["signature_compatible_replay"]["total_replay_events"] > 0
    assert resources["by_arm"]["signature_compatible_replay"]["state_size_max"] <= mod.MEMORY_CAP
    assert restart["restart_equivalence"] == pytest.approx(1.0)
    assert restart["serialized_replay_state_reproduces"] is True


def test_scenario_learn_5857_fail_closed_for_bad_gates_and_tampering(
    artifact: dict[str, Any],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SCENARIO-LEARN-5857-CONTROLS: blocked, unsafe, or future-derived evidence fails."""

    blocked = mod.run(
        root=tmp_path,
        result_path=tmp_path / mod.RESULT_RELATIVE_PATH.name,
        duration_s=1.75,
        test_commands=TEST_COMMANDS,
        test_exit_codes=TEST_EXIT_CODES,
        write=False,
    )
    assert blocked["status"] == "blocked"
    assert blocked["selective_replay_qualified_score"] == 0.0
    assert blocked["honest_verdict"].startswith("blocked:")
    assert "missing_upstream_file" in blocked["preconditions_checked"]["blocked_reasons"]
    assert mod.validate_artifact(blocked) is True

    failed_tests = mod.run(
        result_path=tmp_path / "failed.json",
        preconditions_checked=_preconditions(tmp_path / "failed"),
        duration_s=1.75,
        test_commands=TEST_COMMANDS,
        test_exit_codes={**TEST_EXIT_CODES, FULL_TEST_COMMAND: 2},
        write=False,
    )
    assert failed_tests["status"] == "null"
    assert failed_tests["selective_replay_qualified_score"] == 0.0
    assert "failed_test_exit_codes" in mod.blocked_reasons(failed_tests)

    unsafe = deepcopy(artifact)
    unsafe["unsafe_transfer_count"] = 1
    unsafe["reproducibility_checksum"] = mod.reproducibility_checksum(unsafe)
    assert mod._artifact_status(unsafe) == "unsafe"
    assert mod.honest_verdict(unsafe).startswith("unsafe:")

    blocker_probe = deepcopy(artifact)
    blocker_probe["inference_substrate"] = "wrong"
    blocker_probe["verifier_is_oracle"] = False
    blocker_probe["frozen_signature_definition"]["compatibility_rule_frozen"] = False
    assert set(mod.blocked_reasons(blocker_probe)) >= {
        "inference_substrate",
        "verifier_is_oracle",
        "frozen_signature_definition",
    }

    with monkeypatch.context() as scoped:
        scoped.setattr(mod, "selective_replay_qualified_score", lambda item: 0.0)
        assert mod.blocked_reasons(artifact) == ["qualified_score"]

    for mutate, match in (
        (
            lambda item: item["frozen_signature_definition"].update(
                {"compatibility_rule_frozen": False}
            ),
            "frozen_signature_definition",
        ),
        (
            lambda item: item["frozen_signature_definition"].update(
                {"label_blind_to_future_outcomes": False}
            ),
            "frozen_signature_definition",
        ),
        (
            lambda item: item["frozen_signature_definition"].update(
                {"uses_future_labels": True}
            ),
            "frozen_signature_definition",
        ),
        (
            lambda item: item["frozen_signature_definition"]["signature_components"].append(
                "future_label"
            ),
            "frozen_signature_definition",
        ),
        (
            lambda item: item["restart_equivalence"].update({"restart_equivalence": 0.0}),
            "qualified_score",
        ),
        (
            lambda item: item["replay_resource_accounting"].update({"cap_compliance": False}),
            "qualified_score",
        ),
        (
            lambda item: item.update({"inference_substrate": "live_llm"}),
            "inference_substrate",
        ),
        (
            lambda item: item.update({"verifier_is_oracle": False}),
            "verifier_is_oracle",
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

    bad_provenance = deepcopy(artifact)
    bad_provenance["field_provenance"]["status"]["principle"] = "wrong"
    with pytest.raises(ValueError, match="field_provenance:status"):
        mod.validate_artifact(bad_provenance)

    provenance_not_mapping = deepcopy(artifact)
    provenance_not_mapping["field_provenance"] = []
    with pytest.raises(ValueError, match="field_provenance"):
        mod.validate_artifact(provenance_not_mapping)

    invalid_status = deepcopy(artifact)
    invalid_status["status"] = "blocked"
    invalid_status["reproducibility_checksum"] = mod.reproducibility_checksum(invalid_status)
    with pytest.raises(ValueError, match="status"):
        mod.validate_artifact(invalid_status)

    invalid_verdict = deepcopy(artifact)
    invalid_verdict["honest_verdict"] = "qualified: wrong"
    invalid_verdict["reproducibility_checksum"] = mod.reproducibility_checksum(invalid_verdict)
    with pytest.raises(ValueError, match="honest_verdict"):
        mod.validate_artifact(invalid_verdict)

    with monkeypatch.context() as scoped:
        scoped.setattr(
            mod,
            "_read_json",
            lambda path: (_ for _ in ()).throw(ValueError("corrupt")),
        )
        corrupt = mod.collect_preconditions(
            result_path=tmp_path / "corrupt.json",
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
        "corrupt_upstream_json",
        "insufficient_free_ram",
        "insufficient_free_disk",
    }


def test_req_learn_5857_low_level_helpers_are_deterministic(
    artifact: dict[str, Any],
    tmp_path: Path,
) -> None:
    """REQ-LEARN-5857: helper edges remain deterministic and auditable."""

    scalar_json = tmp_path / "scalar.json"
    scalar_json.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="JSON object required"):
        mod._read_json(scalar_json)

    assert mod.load_clean_rows(tmp_path / "missing") == []
    assert mod.fixture_preconditions(tmp_path)["preconditions_ready"] is True
    assert mod._bootstrap_ci95([]) == [0.0, 0.0]
    assert mod._bootstrap_ci95([0.25]) == [0.25, 0.25]
    assert mod._paired_summary([])["mean_delta"] == 0.0
    assert mod._group_bootstrap_ci95([], "family", "compatible_minus_no") == {
        "n_groups": 0,
        "ci95": [0.0, 0.0],
    }
    assert mod._round(0.123456789) == 0.123457

    rows = mod.load_clean_rows(REPO)
    signature = mod.task_signature(rows[0])
    assert signature["signature_hash"].startswith("sha256:")
    no_selected = mod._select_rows(
        current=rows[1],
        prior_rows=[rows[0]],
        arm="no_replay",
    )
    compatible_selected = mod._select_rows(
        current=rows[1],
        prior_rows=[rows[0]],
        arm="signature_compatible_replay",
    )
    assert no_selected == []
    assert [row["row_id"] for row in compatible_selected] == [rows[0]["row_id"]]
    with pytest.raises(ValueError, match="unknown replay arm"):
        mod._select_rows(current=rows[1], prior_rows=[rows[0]], arm="unknown")

    receipt = mod._replay_receipt(
        current=rows[1],
        selected=compatible_selected,
        arm="signature_compatible_replay",
    )
    assert receipt["receipt_hash"].startswith("sha256:")
    assert receipt["compatible_hits"] == 1

    no_write = mod.run(
        result_path=tmp_path / "no-write.json",
        preconditions_checked=_preconditions(tmp_path / "no-write"),
        duration_s=1.75,
        test_commands=TEST_COMMANDS,
        test_exit_codes=TEST_EXIT_CODES,
        write=False,
    )
    assert no_write["status"] == "qualified"
    assert not (tmp_path / "no-write.json").exists()
