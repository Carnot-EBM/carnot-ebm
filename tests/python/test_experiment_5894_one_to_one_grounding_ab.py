"""Tests for Exp5894 one-to-one atom-grounding A/B.

Spec refs: REQ-LEARN-5894, SCENARIO-LEARN-5894-PRECONDITIONS,
SCENARIO-LEARN-5894-ARM-PARITY,
SCENARIO-LEARN-5894-SEMANTIC-VS-CONSTRAINT,
SCENARIO-LEARN-5894-CONTROLS-AND-LOWER-BOUNDS,
SCENARIO-LEARN-5894-FAIL-CLOSED.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any

import pytest

from carnot import experiment_5894_one_to_one_grounding_ab as mod


REPO = Path(__file__).resolve().parents[2]
SELF_LEARNING_SPEC = REPO / "openspec/capabilities/self-learning/spec.md"
FOCUSED_TEST_COMMAND = (
    ".venv/bin/pytest tests/python/test_experiment_5894_one_to_one_grounding_ab.py "
    "-q --no-cov -n 0"
)
COVERAGE_COMMAND = (
    ".venv/bin/coverage run --rcfile=/dev/null "
    "--include=python/carnot/experiment_5894_one_to_one_grounding_ab.py "
    "-m pytest tests/python/test_experiment_5894_one_to_one_grounding_ab.py "
    "-q --no-cov -n 0 && .venv/bin/coverage report --rcfile=/dev/null "
    "--include=python/carnot/experiment_5894_one_to_one_grounding_ab.py "
    "--fail-under=100"
)
FULL_TEST_COMMAND = ".venv/bin/pytest tests/python -q"
SPEC_COMMAND = (
    ".venv/bin/python scripts/check_spec_coverage.py "
    "tests/python/test_experiment_5894_one_to_one_grounding_ab.py"
)
ADVERSARIAL_COMMAND = (
    ".venv/bin/python scripts/adversarial_verify.py --json "
    "results/experiment_5894_one_to_one_grounding_ab.json"
)
ROOT_CLUTTER_COMMAND = ".venv/bin/python scripts/root_clutter_sweep.py"
PROTECTED_FILE_COMMAND = (
    '.venv/bin/python -c "from pathlib import Path; '
    "assert Path('scripts/research_conductor.py').exists()\""
)
TEST_COMMANDS = [
    FOCUSED_TEST_COMMAND,
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
            "available_mb": 32768,
            "required_mb": mod.RAM_FLOOR_MB,
            "ok": True,
        },
        disk_probe=lambda root: {
            "available_mb": 32768,
            "required_mb": mod.DISK_FLOOR_MB,
            "ok": True,
        },
    )


@pytest.fixture(scope="module")
def artifact(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    """REQ-LEARN-5894: build the deterministic one-to-one A/B artifact once."""

    base = tmp_path_factory.mktemp("exp5894")
    return mod.run(
        result_path=base / mod.RESULT_RELATIVE_PATH.name,
        preconditions_checked=_preconditions(base),
        duration_s=3.0,
        test_commands=TEST_COMMANDS,
        test_exit_codes=TEST_EXIT_CODES,
        write=True,
    )


def test_req_learn_5894_spec_declares_required_contract() -> None:
    """REQ-LEARN-5894: OpenSpec names every field, principle, and scenario."""

    text = SELF_LEARNING_SPEC.read_text(encoding="utf-8")
    section = text[text.index("## REQ-LEARN-5894") :]
    normalized = " ".join(section.split())

    for marker in (
        "REQ-LEARN-5894",
        "SCENARIO-LEARN-5894-PRECONDITIONS",
        "SCENARIO-LEARN-5894-ARM-PARITY",
        "SCENARIO-LEARN-5894-SEMANTIC-VS-CONSTRAINT",
        "SCENARIO-LEARN-5894-CONTROLS-AND-LOWER-BOUNDS",
        "SCENARIO-LEARN-5894-FAIL-CLOSED",
        mod.RESULT_RELATIVE_PATH.as_posix(),
        mod.INFERENCE_SUBSTRATE,
        "`one_to_one_grounding_ready_score`",
    ):
        assert marker in section
    for field, principle in mod.REQUIRED_FIELD_PRINCIPLES.items():
        assert f"`{field}`" in section
        assert " ".join(principle.split()) in normalized


def test_scenario_learn_5894_preconditions_and_artifact_are_hash_bound(
    artifact: dict[str, Any],
    tmp_path: Path,
) -> None:
    """SCENARIO-LEARN-5894-PRECONDITIONS: gates and rows replay exactly."""

    result_path = tmp_path / mod.RESULT_RELATIVE_PATH.name
    replay = mod.run(
        result_path=result_path,
        preconditions_checked=_preconditions(tmp_path),
        duration_s=3.0,
        test_commands=TEST_COMMANDS,
        test_exit_codes=TEST_EXIT_CODES,
        write=True,
    )
    loaded = json.loads(result_path.read_text(encoding="utf-8"))

    assert replay == loaded
    assert replay["reproducibility_checksum"] == artifact["reproducibility_checksum"]
    assert replay["reproducibility_checksum"] == mod.reproducibility_checksum(replay)
    assert mod.validate_artifact(replay) is True
    assert set(mod.REQUIRED_ARTIFACT_FIELDS) <= set(replay)
    assert replay["status"] == "complete_positive"
    assert replay["honest_verdict"].startswith("complete_positive:")
    assert replay["one_to_one_grounding_ready_score"] == pytest.approx(1.0)
    assert isinstance(replay["one_to_one_grounding_ready_score"], float)
    assert replay["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert replay["verifier_is_oracle"] is True
    assert replay["oracle_boundary_violation_count"] == 0

    preconditions = replay["preconditions_checked"]
    upstream = replay["upstream_gate_and_row_hashes"]
    assert preconditions["preconditions_ready"] is True
    assert upstream["exp5893_gate_ready"] is True
    assert upstream["row_count"] == 72
    assert upstream["row_hashes_match"] is True
    assert upstream["exact_oracles_replayed"] is True
    assert upstream["split_groups_isolated"] is True
    assert replay["test_commands"] == TEST_COMMANDS
    assert replay["test_exit_codes"] == TEST_EXIT_CODES
    for field, principle in mod.REQUIRED_FIELD_PRINCIPLES.items():
        assert replay["field_provenance"][field]["principle"] == principle


def test_scenario_learn_5894_arm_parity_and_chronology_receipts(
    artifact: dict[str, Any],
) -> None:
    """SCENARIO-LEARN-5894-ARM-PARITY: arms match budgets before held rows."""

    parity = artifact["frozen_arm_definitions_and_budget_parity"]
    chronology = artifact["chronology_and_visibility_receipts"]
    accounting = artifact["query_replay_and_state_accounting"]

    assert parity["arms"] == list(mod.ARM_NAMES)
    assert parity["kan_excluded"] is True
    assert parity["frozen_before_held_batches"] is True
    assert parity["budget_parity_passed"] is True
    assert parity["same_initialization_hash"] is True
    assert parity["same_state_capacity"] is True
    assert parity["same_threshold"] is True
    assert parity["exact_template_is_control_not_learned_credit"] is True

    budgets = parity["per_arm_budgets"]
    assert len({item["exact_query_count"] for item in budgets.values()}) == 1
    assert len({item["replay_count"] for item in budgets.values()}) == 1
    assert len({item["state_capacity"] for item in budgets.values()}) == 1
    assert all(item["threshold"] == mod.GROUNDING_THRESHOLD for item in budgets.values())

    assert chronology["chronological_event_count"] == 72
    assert chronology["train_event_count"] == 36
    assert chronology["held_event_count"] == 36
    assert chronology["held_batch_start_index"] == 36
    assert chronology["future_label_visible_before_prediction_count"] == 0
    assert chronology["label_keys_visible_before_prediction"] == []
    assert chronology["no_arm_updates_after_held_start"] is True
    assert chronology["event_order_hash"].startswith("sha256:")
    assert chronology["sample_visibility_receipts"][0]["label_visible_before_prediction"] is False

    assert accounting["all_arms_within_state_cap"] is True
    assert accounting["query_count_parity"] is True
    assert accounting["replay_count_parity"] is True
    assert accounting["initialization_parity"] is True
    assert accounting["threshold"] == mod.GROUNDING_THRESHOLD
    assert accounting["one_to_one_lift_per_query"] > accounting["best_learned_control_lift_per_query"]


def test_scenario_learn_5894_semantic_vs_constraint_metrics(
    artifact: dict[str, Any],
) -> None:
    """SCENARIO-LEARN-5894-SEMANTIC-VS-CONSTRAINT: shortcuts are counted directly."""

    outcomes = artifact["semantic_vs_constraint_outcomes"]
    shortcuts = artifact["shortcut_false_accept_metrics"]
    one = outcomes["arm_metrics"][mod.ONE_TO_ONE_ARM]
    soft = outcomes["arm_metrics"]["soft_probability"]
    fuzzy = outcomes["arm_metrics"]["fuzzy_t_norm"]
    distributed = outcomes["arm_metrics"]["distributed_many_to_one"]
    exact = outcomes["arm_metrics"]["current_exact_template"]
    no_learner = outcomes["arm_metrics"]["no_learner"]

    assert outcomes["formula_satisfaction_cannot_promote"] is True
    assert one["semantic_accuracy"] == pytest.approx(exact["semantic_accuracy"])
    assert one["semantic_accuracy"] > soft["semantic_accuracy"]
    assert one["semantic_accuracy"] > fuzzy["semantic_accuracy"]
    assert one["semantic_accuracy"] > distributed["semantic_accuracy"]
    assert one["semantic_accuracy"] > no_learner["semantic_accuracy"]
    assert one["encoded_constraint_accuracy"] < soft["encoded_constraint_accuracy"]
    assert one["abstention_rate"] > 0.0

    assert shortcuts["one_to_one_zero_false_accepts"] is True
    assert shortcuts["unsafe_accept_count"] == 0
    assert shortcuts["by_arm"][mod.ONE_TO_ONE_ARM]["total"] == 0
    for shortcut_type in mod.SHORTCUT_TYPES_TO_MEASURE:
        assert shortcuts["by_arm"][mod.ONE_TO_ONE_ARM]["by_type"][shortcut_type] == 0
        assert shortcuts["by_arm"]["soft_probability"]["by_type"][shortcut_type] > 0
        assert shortcuts["by_arm"]["fuzzy_t_norm"]["by_type"][shortcut_type] > 0
        assert shortcuts["by_arm"]["distributed_many_to_one"]["by_type"][shortcut_type] > 0


def test_scenario_learn_5894_controls_lower_bounds_transfer_and_safety(
    artifact: dict[str, Any],
) -> None:
    """SCENARIO-LEARN-5894-CONTROLS-AND-LOWER-BOUNDS: no credited cell fails."""

    transfer = artifact["forward_transfer_recurrence_and_retention"]
    bounds = artifact["family_grounding_hardness_lower_bounds"]
    controls = artifact["permutation_relabel_rebalance_and_null_controls"]
    safety = artifact["protected_prefix_and_safety"]

    assert transfer["held_forward_transfer"][mod.ONE_TO_ONE_ARM]["semantic_accuracy"] > transfer[
        "held_forward_transfer"
    ]["soft_probability"]["semantic_accuracy"]
    assert transfer["one_to_one_recurrence"]["shortcut_false_accept_count"] == 0
    assert transfer["protected_prefix_retention"][mod.ONE_TO_ONE_ARM] == pytest.approx(1.0)
    assert transfer["retention_regression_count"] == 0

    assert bounds["all_credited_cells_positive_over_learned_controls"] is True
    assert bounds["pooled_promotion_does_not_hide_failing_cell"] is True
    assert bounds["minimum_credited_lcb"] > 0.0
    for axis in ("family", "grounding", "hardness"):
        assert bounds["group_bootstrap_intervals"][axis]["n_groups"] > 0
        assert bounds["group_bootstrap_intervals"][axis]["ci95"][0] > 0.0
    for cell in bounds["credited_held_cells"]:
        assert cell["one_to_one_minus_best_learned_control"]["ci95"][0] > 0.0

    assert controls["all_controls_passed"] is True
    assert controls["label_permutation"]["prediction_delta_count"] == 0
    assert controls["atom_permutation"]["prediction_delta_count"] == 0
    assert controls["grounding_permutation"]["one_to_one_shortcut_false_accepts"] == 0
    assert controls["frequency_rebalance"]["semantic_label_balance"] is True
    assert controls["family_holdout"]["all_holdouts_positive"] is True
    assert controls["no_information_control"]["one_to_one_abstention_rate"] == pytest.approx(1.0)
    assert controls["no_information_control"]["control_passed"] is True

    assert safety["unsafe_accept_count"] == 0
    assert safety["protected_prefix_regression_count"] == 0
    assert safety["protected_files_unchanged"]["all_unchanged"] is True
    assert safety["default_off_and_no_production_integration"] is True


def test_scenario_learn_5894_fail_closed_for_bad_inputs_and_tampering(
    artifact: dict[str, Any],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SCENARIO-LEARN-5894-FAIL-CLOSED: unsafe, leaky, or null evidence cannot promote."""

    blocked = mod.run(
        root=tmp_path,
        result_path=tmp_path / mod.RESULT_RELATIVE_PATH.name,
        duration_s=3.0,
        test_commands=TEST_COMMANDS,
        test_exit_codes=TEST_EXIT_CODES,
        write=False,
    )
    assert blocked["status"] == "blocked"
    assert blocked["one_to_one_grounding_ready_score"] == 0.0
    assert blocked["honest_verdict"].startswith("blocked:")
    assert "missing_exp5893_artifact" in blocked["preconditions_checked"]["blocked_reasons"]
    assert mod.validate_artifact(blocked) is True

    failed_tests = mod.run(
        result_path=tmp_path / "failed.json",
        preconditions_checked=_preconditions(tmp_path / "failed"),
        duration_s=3.0,
        test_commands=TEST_COMMANDS,
        test_exit_codes={**TEST_EXIT_CODES, FULL_TEST_COMMAND: 1},
        write=False,
    )
    assert failed_tests["status"] == "complete_null"
    assert "failed_test_exit_codes" in mod.blocked_reasons(failed_tests)
    assert mod.validate_artifact(failed_tests) is True

    unsafe = deepcopy(artifact)
    unsafe["protected_prefix_and_safety"]["unsafe_accept_count"] = 1
    unsafe["shortcut_false_accept_metrics"]["unsafe_accept_count"] = 1
    unsafe["one_to_one_grounding_ready_score"] = mod.one_to_one_grounding_ready_score(unsafe)
    unsafe["status"] = mod.status(unsafe)
    unsafe["honest_verdict"] = mod.honest_verdict(unsafe)
    unsafe["reproducibility_checksum"] = mod.reproducibility_checksum(unsafe)
    assert unsafe["status"] == "unsafe"
    assert unsafe["honest_verdict"].startswith("unsafe:")
    assert mod.validate_artifact(unsafe) is True

    null_artifact = deepcopy(artifact)
    null_artifact["family_grounding_hardness_lower_bounds"][
        "all_credited_cells_positive_over_learned_controls"
    ] = False
    null_artifact["one_to_one_grounding_ready_score"] = mod.one_to_one_grounding_ready_score(
        null_artifact
    )
    null_artifact["status"] = mod.status(null_artifact)
    null_artifact["honest_verdict"] = mod.honest_verdict(null_artifact)
    null_artifact["reproducibility_checksum"] = mod.reproducibility_checksum(null_artifact)
    assert null_artifact["status"] == "complete_null"
    assert null_artifact["honest_verdict"].startswith("complete_null:")
    assert mod.validate_artifact(null_artifact) is True

    missing = deepcopy(artifact)
    missing.pop("status")
    with pytest.raises(ValueError, match="missing required artifact fields"):
        mod.validate_artifact(missing)

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
            lambda item: item.update({"oracle_boundary_violation_count": 1}),
            "ready_score",
        ),
        (
            lambda item: item["frozen_arm_definitions_and_budget_parity"].update(
                {"budget_parity_passed": False}
            ),
            "ready_score",
        ),
        (
            lambda item: item.update({"one_to_one_grounding_ready_score": 0.0}),
            "ready_score",
        ),
        (
            lambda item: item.update({"status": "blocked"}),
            "status",
        ),
        (
            lambda item: item.update({"honest_verdict": "complete_positive: wrong"}),
            "honest_verdict",
        ),
        (
            lambda item: item.update({"reproducibility_checksum": mod.sha256_text("wrong")}),
            "reproducibility_checksum",
        ),
    ):
        bad = deepcopy(artifact)
        mutate(bad)
        if "reproducibility_checksum" not in match:
            bad["reproducibility_checksum"] = mod.reproducibility_checksum(bad)
        with pytest.raises(ValueError, match=match):
            mod.validate_artifact(bad)

    bad_provenance = deepcopy(artifact)
    bad_provenance["field_provenance"]["status"]["principle"] = "wrong"
    with pytest.raises(ValueError, match="field_provenance:status"):
        mod.validate_artifact(bad_provenance)

    with monkeypatch.context() as scoped:
        scoped.setattr(mod, "one_to_one_grounding_ready_score", lambda item: 0.0)
        assert mod.blocked_reasons(artifact) == ["ready_score"]

    combined = deepcopy(artifact)
    combined["inference_substrate"] = "wrong"
    combined["verifier_is_oracle"] = False
    combined["oracle_boundary_violation_count"] = 1
    combined["frozen_arm_definitions_and_budget_parity"]["budget_parity_passed"] = False
    combined["query_replay_and_state_accounting"]["all_arms_within_state_cap"] = False
    combined_reasons = set(mod.blocked_reasons(combined))
    assert {
        "inference_substrate",
        "verifier_is_oracle",
        "oracle_boundary_violation_count",
        "budget_parity",
        "state_cap",
    } <= combined_reasons

    provenance_not_mapping = deepcopy(artifact)
    provenance_not_mapping["field_provenance"] = []
    with pytest.raises(ValueError, match="field_provenance"):
        mod.validate_artifact(provenance_not_mapping)

    scalar_json = tmp_path / "scalar.json"
    scalar_json.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="JSON object required"):
        mod._read_json(scalar_json)
    bad_jsonl = tmp_path / "bad.rows.jsonl"
    bad_jsonl.write_text("\n[]\n", encoding="utf-8")
    with pytest.raises(ValueError, match="JSONL object required"):
        mod._read_jsonl(bad_jsonl)
    assert mod.load_fixture_rows(tmp_path / "missing") == []
    assert mod._rows_to_jsonl(mod.load_fixture_rows(REPO)[:1]).endswith("\n")
    assert mod._forbidden_visible_paths({"outer": [{"exact_semantic_label": True}]}) == [
        "outer[0].exact_semantic_label"
    ]
