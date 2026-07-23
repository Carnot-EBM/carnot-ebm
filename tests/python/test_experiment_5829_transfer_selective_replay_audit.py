"""Tests for Exp5829 transfer-selective replay audit.

Spec refs: REQ-LEARN-5829, SCENARIO-LEARN-5829-SIGNATURE-FREEZE,
SCENARIO-LEARN-5829-REPLAY-PARITY,
SCENARIO-LEARN-5829-TRANSFER-RETENTION-RECURRENCE,
SCENARIO-LEARN-5829-FAIL-CLOSED.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any

import pytest

from carnot import experiment_5829_transfer_selective_replay_audit as mod


REPO = Path(__file__).resolve().parents[2]
SELF_LEARNING_SPEC = REPO / "openspec/capabilities/self-learning/spec.md"
TEST_COMMAND = (
    ".venv/bin/pytest "
    "tests/python/test_experiment_5829_transfer_selective_replay_audit.py "
    "-q --no-cov -n 0"
)
COVERAGE_COMMAND = (
    ".venv/bin/coverage run --rcfile=/dev/null "
    "--include=python/carnot/experiment_5829_transfer_selective_replay_audit.py "
    "-m pytest tests/python/test_experiment_5829_transfer_selective_replay_audit.py "
    "-q --no-cov -n 0 && .venv/bin/coverage report --rcfile=/dev/null "
    "--include=python/carnot/experiment_5829_transfer_selective_replay_audit.py "
    "--fail-under=100"
)
FULL_TEST_COMMAND = ".venv/bin/pytest tests/python -q"
SPEC_COMMAND = ".venv/bin/python scripts/check_spec_coverage.py"
ADVERSARIAL_COMMAND = (
    ".venv/bin/python scripts/adversarial_verify.py "
    "results/experiment_5829_transfer_selective_replay_audit.json"
)
ROOT_CLUTTER_COMMAND = ".venv/bin/python scripts/root_clutter_sweep.py"
TEST_COMMANDS = [
    TEST_COMMAND,
    COVERAGE_COMMAND,
    FULL_TEST_COMMAND,
    SPEC_COMMAND,
    ADVERSARIAL_COMMAND,
    ROOT_CLUTTER_COMMAND,
]
TEST_EXIT_CODES = {command: 0 for command in TEST_COMMANDS}


def _preconditions(tmp_path: Path) -> dict[str, Any]:
    return mod.collect_preconditions(
        result_path=tmp_path / mod.RESULT_RELATIVE_PATH.name,
        checkpoint_dir=tmp_path / "checkpoints",
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
    """REQ-LEARN-5829: build the deterministic replay audit artifact once."""

    base = tmp_path_factory.mktemp("exp5829")
    return mod.run(
        result_path=base / mod.RESULT_RELATIVE_PATH.name,
        checkpoint_dir=base / "checkpoints",
        preconditions_checked=_preconditions(base),
        duration_s=1.0,
        test_commands=TEST_COMMANDS,
        test_exit_codes=TEST_EXIT_CODES,
        write=True,
    )


def test_req_learn_5829_spec_declares_transfer_audit_contract() -> None:
    """REQ-LEARN-5829: OpenSpec names fields, principles, and scenarios."""

    text = SELF_LEARNING_SPEC.read_text(encoding="utf-8")
    section = text[text.index("## REQ-LEARN-5829") :]
    normalized = " ".join(section.split())

    for marker in (
        "REQ-LEARN-5829",
        "SCENARIO-LEARN-5829-SIGNATURE-FREEZE",
        "SCENARIO-LEARN-5829-REPLAY-PARITY",
        "SCENARIO-LEARN-5829-TRANSFER-RETENTION-RECURRENCE",
        "SCENARIO-LEARN-5829-FAIL-CLOSED",
        mod.RESULT_RELATIVE_PATH.as_posix(),
        mod.INFERENCE_SUBSTRATE,
        "`task_signature_and_compatibility_rule`",
        "`replay_resource_accounting`",
    ):
        assert marker in section
    for field, principle in mod.REQUIRED_FIELD_PRINCIPLES.items():
        assert f"`{field}`" in section
        assert " ".join(principle.split()) in normalized


def test_req_learn_5829_terminal_artifact_is_deterministic_and_hash_bound(
    artifact: dict[str, Any],
    tmp_path: Path,
) -> None:
    """REQ-LEARN-5829: clean replay writes a complete terminal artifact."""

    destination = tmp_path / mod.RESULT_RELATIVE_PATH.name
    replay = mod.run(
        result_path=destination,
        checkpoint_dir=tmp_path / "ckpt",
        preconditions_checked=_preconditions(tmp_path),
        duration_s=1.0,
        test_commands=TEST_COMMANDS,
        test_exit_codes=TEST_EXIT_CODES,
        write=True,
    )
    loaded = json.loads(destination.read_text(encoding="utf-8"))

    assert artifact == replay == loaded
    assert mod.validate_artifact(artifact) is True
    assert set(mod.REQUIRED_ARTIFACT_FIELDS).issubset(artifact)
    assert artifact["status"] == "complete"
    assert artifact["honest_verdict"].startswith("positive:")
    assert artifact["compatible_replay_credited"] is True
    assert artifact["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert artifact["verifier_is_oracle"] is True
    assert artifact["preconditions_checked"]["preconditions_ready"] is True
    assert artifact["preconditions_checked"]["structured_gate_replay"]["ok"] is True
    assert artifact["preconditions_checked"]["heldout_split_check"]["ok"] is True
    assert artifact["preconditions_checked"]["headroom_check"]["ok"] is True
    assert artifact["upstream_artifact_hashes"]["exp5828_lifecycle_artifact"].startswith(
        "sha256:"
    )
    assert artifact["upstream_artifact_hashes"]["exp5826_stream_rows"].startswith(
        "sha256:"
    )
    assert artifact["test_commands"] == TEST_COMMANDS
    assert artifact["test_exit_codes"] == TEST_EXIT_CODES
    assert artifact["reproducibility_checksum"] == mod.reproducibility_checksum(artifact)


def test_scenario_learn_5829_signature_rule_is_frozen_and_label_blind(
    artifact: dict[str, Any],
) -> None:
    """SCENARIO-LEARN-5829-SIGNATURE-FREEZE: replay selection ignores labels."""

    signature = artifact["task_signature_and_compatibility_rule"]
    rule = signature["compatibility_rule"]
    samples = signature["sample_signature_receipts"]

    assert signature["schema"].endswith(".signature_rule")
    assert signature["signature_count"] == 360
    assert signature["compatibility_rule_frozen"] is True
    assert rule["calibration_split"] == "train_dev_only"
    assert rule["uses_science_family_label"] is False
    assert rule["uses_future_suffix_labels"] is False
    assert rule["uses_posthoc_metric_selection"] is False
    assert rule["forbidden_selector_fields"] == list(mod.FORBIDDEN_SELECTOR_FIELDS)
    assert rule["rule_hash"].startswith("sha256:")
    assert signature["signature_root_hash"].startswith("sha256:")

    for sample in samples[:12]:
        fields = sample["signature"]
        assert sample["signature_hash"].startswith("sha256:")
        assert set(mod.SIGNATURE_COMPONENTS).issubset(fields)
        assert "family" not in mod.canonical_json(fields)
        assert "future" not in fields["exact_prefix_behavior"]
        assert fields["proof_preserving_surface"] in mod.PROOF_PRESERVING_SURFACES

    rows = mod._load_rows(REPO)
    first = mod.task_signature(rows[0])
    assert first["signature_hash"].startswith("sha256:")
    assert mod.compatible_for_replay(rows[0], rows[1]) is True
    assert mod.compatible_for_replay(rows[0], rows[-1]) is False


def test_scenario_learn_5829_replay_parity_splits_and_leakage_receipts(
    artifact: dict[str, Any],
) -> None:
    """SCENARIO-LEARN-5829-REPLAY-PARITY: arms differ only by replay source."""

    arms = artifact["arm_definitions_and_replay_parity"]
    heldout = artifact["heldout_split_and_leakage_receipts"]
    resources = artifact["replay_resource_accounting"]

    assert arms["arms"] == list(mod.REPLAY_ARMS)
    assert arms["parity_passed"] is True
    assert arms["matched_count_parity_passed"] is True
    assert {
        arms["definitions"][arm]["current_task_evidence"]
        for arm in mod.REPLAY_ARMS
    } == {"identical_exact_prefix_and_current_task_membership_receipts"}
    assert {arms["definitions"][arm]["memory_cap"] for arm in mod.REPLAY_ARMS} == {
        mod.MEMORY_CAP
    }
    assert {arms["definitions"][arm]["query_budget"] for arm in mod.REPLAY_ARMS} == {
        mod.QUERY_BUDGET_PER_ROW
    }

    for receipt in arms["sample_replay_parity_receipts"][:24]:
        assert receipt["all_selected_rows_prior"] is True
        assert receipt["future_suffix_rows_selected"] == 0
        assert receipt["compatible_count"] == receipt["random_count"] == receipt["all_count"]
        assert receipt["replay_count"] <= mod.REPLAY_EVENT_CAP

    assert heldout["heldout_family_surface_cell_count"] == 8
    assert heldout["minimum_rows_per_family_surface_cell"] >= 30
    assert heldout["recurrence_rows"] == 120
    assert heldout["first_exposure_rows"] == 360
    assert heldout["science_label_leakage_count"] == 0
    assert heldout["future_label_leakage_count"] == 0
    assert heldout["state_or_replay_boundary_crossing_count"] == 0
    assert heldout["n_ge_30_per_primary_cell"] is True

    assert resources["cap_compliance"] is True
    assert resources["max_replay_events_per_task"] <= mod.REPLAY_EVENT_CAP
    assert resources["max_memory_cap_pressure"] <= 1.0
    assert resources["checkpoint_resume_receipt"]["restart_equivalence"] == 1.0


def test_scenario_learn_5829_transfer_retention_recurrence_and_resources_are_separate(
    artifact: dict[str, Any],
) -> None:
    """SCENARIO-LEARN-5829-TRANSFER-RETENTION-RECURRENCE: credit gates are explicit."""

    forward = artifact["forward_transfer_metrics"]
    retention = artifact["retention_and_forgetting_metrics"]
    recurrence = artifact["recurrence_recovery_metrics"]
    paired = artifact["paired_deltas_and_ci95"]
    resources = artifact["replay_resource_accounting"]

    assert forward["first_exposure_definition"] == mod.FIRST_EXPOSURE_DEFINITION
    assert forward["arm_metrics"]["signature_compatible_replay"][
        "forward_transfer_accuracy"
    ] > forward["arm_metrics"]["reset_no_replay"]["forward_transfer_accuracy"]
    assert forward["compatible_minus_no_replay"]["ci95"][0] > 0.0
    assert forward["dynamic_regret"]["signature_compatible_replay"] < forward[
        "dynamic_regret"
    ]["reset_no_replay"]
    assert forward["abstention_rate"]["signature_compatible_replay"] < forward[
        "abstention_rate"
    ]["reset_no_replay"]

    assert retention["compatible_retention_noninferior_to_all_replay"] is True
    assert retention["forgetting"]["signature_compatible_replay"] <= retention["forgetting"][
        "all_replay"
    ]
    assert recurrence["compatible_recurrence_improves_over_no_replay"] is True
    assert recurrence["arm_metrics"]["signature_compatible_replay"][
        "recurrence_recovery"
    ] > recurrence["arm_metrics"]["reset_no_replay"]["recurrence_recovery"]
    assert artifact["unsafe_transfer_count"] == 0
    assert artifact["credit_gates"]["all_passed"] is True
    assert resources["by_arm"]["signature_compatible_replay"]["total_replay_events"] > 0
    assert resources["by_arm"]["signature_compatible_replay"]["total_replay_bytes"] > 0

    assert paired["compatible_minus_no_replay_forward"]["ci95"][0] > 0.0
    assert paired["compatible_minus_all_replay_retention"]["ci95"][0] >= -mod.NONINFERIORITY_MARGIN
    assert paired["family_heterogeneity"]["family_count"] == len(mod.PRIMARY_FAMILIES)
    assert paired["family_heterogeneity"]["min_forward_lcb95"] > 0.0


def test_scenario_learn_5829_fail_closed_for_bad_gates_and_tampering(
    artifact: dict[str, Any],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SCENARIO-LEARN-5829-FAIL-CLOSED: blocked or unsafe audits cannot get credit."""

    blocked = mod.run(
        root=tmp_path,
        result_path=tmp_path / mod.RESULT_RELATIVE_PATH.name,
        checkpoint_dir=tmp_path / "checkpoints",
        duration_s=1.0,
        test_commands=TEST_COMMANDS,
        test_exit_codes=TEST_EXIT_CODES,
        write=True,
    )
    assert blocked["status"] == "blocked"
    assert blocked["honest_verdict"].startswith("blocked:")
    assert blocked["compatible_replay_credited"] is False
    assert "missing_upstream_artifact" in blocked["preconditions_checked"]["blocked_reasons"]
    assert mod.validate_artifact(blocked) is True

    failed_exits = mod.build_artifact(
        preconditions_checked=_preconditions(tmp_path / "failed"),
        duration_s=1.0,
        test_commands=TEST_COMMANDS,
        test_exit_codes={**TEST_EXIT_CODES, FULL_TEST_COMMAND: 2},
    )
    assert failed_exits["status"] == "blocked"
    assert "failed_test_exit_codes" in mod.blocked_reasons(failed_exits)
    assert mod.fixture_preconditions()["preconditions_ready"] is True

    blocker_probe = deepcopy(artifact)
    blocker_probe["inference_substrate"] = "wrong"
    blocker_probe["verifier_is_oracle"] = False
    blocker_probe["unsafe_transfer_count"] = 1
    blocker_probe["replay_resource_accounting"]["cap_compliance"] = False
    assert set(mod.blocked_reasons(blocker_probe)) >= {
        "inference_substrate",
        "verifier_is_oracle",
        "unsafe_transfer_count",
        "cap_compliance",
    }

    nullish = deepcopy(artifact)
    nullish["credit_gates"]["forward_lcb_positive"] = False
    nullish["credit_gates"]["all_passed"] = False
    nullish["compatible_replay_credited"] = False
    assert mod.honest_verdict(nullish).startswith("null:")

    negative = deepcopy(nullish)
    negative["credit_gates"]["resource_within_cap"] = False
    assert mod.honest_verdict(negative).startswith("negative:")

    for mutate, match in (
        (lambda item: item.update({"inference_substrate": "live_llm_inference"}), "inference_substrate"),
        (lambda item: item.update({"verifier_is_oracle": False}), "verifier_is_oracle"),
        (lambda item: item.update({"unsafe_transfer_count": 1}), "compatible_replay_credited"),
        (
            lambda item: item["credit_gates"].update({"resource_within_cap": False}),
            "compatible_replay_credited",
        ),
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

    invalid_verdict = deepcopy(artifact)
    invalid_verdict["honest_verdict"] = "positive: wrong"
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
            checkpoint_dir=tmp_path / "corrupt-ckpt",
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


def test_req_learn_5829_low_level_helpers_are_deterministic(
    artifact: dict[str, Any],
    tmp_path: Path,
) -> None:
    """REQ-LEARN-5829: helper edges remain deterministic and auditable."""

    scalar_json = tmp_path / "scalar.json"
    scalar_json.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="JSON object required"):
        mod._read_json(scalar_json)

    rows = mod._load_rows(REPO)
    empty_signature_bundle = mod._signature_bundle([])
    assert empty_signature_bundle["signature_count"] == 0
    assert mod._bootstrap_ci95([]) == [0.0, 0.0]
    assert mod._bootstrap_ci95([0.25]) == [0.25, 0.25]
    assert mod._summary([])["n"] == 0
    assert mod._latency_summary([])["count"] == 0
    assert mod._round(0.123456789) == 0.123457

    first = rows[0]
    second = rows[1]
    first_signature = mod.task_signature(first)
    first_episode = mod._episode_record(first, {}, first_signature)
    assert first_episode["row_id"] == first["row_id"]
    assert first_episode["oracle_accuracy"] == 1.0
    assert first_episode["headroom"] >= 0.0

    chosen = mod._select_replay_rows(
        current=second,
        prior_rows=[first],
        compatible_count=1,
        arm=mod.COMPATIBLE_REPLAY_ARM,
    )
    assert [row["row_id"] for row in chosen] == [first["row_id"]]
    assert mod._select_replay_rows(
        current=second,
        prior_rows=[first],
        compatible_count=0,
        arm=mod.RESET_NO_REPLAY_ARM,
    ) == []

    receipt = mod._replay_receipt(
        current=second,
        selected=chosen,
        arm=mod.COMPATIBLE_REPLAY_ARM,
        replay_count=1,
    )
    assert receipt["receipt_hash"].startswith("sha256:")
    assert receipt["compatible_hits"] == 1
    assert receipt["total_replay_bytes"] > 0

    no_write = mod.run(
        result_path=tmp_path / "no-write.json",
        checkpoint_dir=tmp_path / "no-write-ckpt",
        preconditions_checked=_preconditions(tmp_path / "no-write"),
        duration_s=1.0,
        test_commands=TEST_COMMANDS,
        test_exit_codes=TEST_EXIT_CODES,
        write=False,
    )
    assert no_write["status"] == "complete"
    assert not (tmp_path / "no-write.json").exists()
