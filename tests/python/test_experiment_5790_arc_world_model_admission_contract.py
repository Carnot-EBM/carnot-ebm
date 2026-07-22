"""Tests for Exp5790 ARC world-model admission contract.

Spec refs: REQ-ARC-WMTE-5790,
SCENARIO-ARC-WMTE-5790-LEAKAGE-AND-PROVENANCE-REJECTION,
SCENARIO-ARC-WMTE-5790-PIVOTAL-FREEZE-AND-METRICS,
SCENARIO-ARC-WMTE-5790-ADMISSION-DECISIONS-NO-CREDIT.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any

import pytest

from carnot import experiment_5790_arc_world_model_admission_contract as mod


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec/capabilities/arc-world-model-trust-energy/spec.md"
RESULT_PATH = REPO / mod.RESULT_RELATIVE_PATH
TEST_COMMAND = (
    ".venv/bin/pytest "
    "tests/python/test_experiment_5790_arc_world_model_admission_contract.py "
    "-q --no-cov -n 0"
)
COVERAGE_COMMAND = (
    ".venv/bin/coverage run --rcfile=/dev/null "
    "--include=python/carnot/experiment_5790_arc_world_model_admission_contract.py "
    "-m pytest tests/python/test_experiment_5790_arc_world_model_admission_contract.py "
    "-q --no-cov -n 0 && .venv/bin/coverage report --rcfile=/dev/null "
    "--include=python/carnot/experiment_5790_arc_world_model_admission_contract.py "
    "--fail-under=100"
)
FULL_TEST_COMMAND = ".venv/bin/pytest tests/python -q"
SPEC_COMMAND = ".venv/bin/python scripts/check_spec_coverage.py"
ADVERSARIAL_COMMAND = (
    ".venv/bin/python scripts/adversarial_verify.py "
    "results/experiment_5790_arc_world_model_admission_contract.json"
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


def _sha(label: str) -> str:
    return "sha256:" + (label.encode("utf-8").hex() * 8)[:64].ljust(64, "0")


def _transition_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index in range(20):
        row_id = f"t{index:02d}"
        split = "seen" if index < 6 else "heldout"
        action = "MOVE"
        if index in {6, 7, 8, 9}:
            action = "PAINT"
            split = "unseen_action" if index == 6 else "heldout"
        row: dict[str, Any] = {
            "row_id": row_id,
            "anonymous_trace_id": f"anon-{index % 4}",
            "observation_hash": _sha(f"obs-{index}"),
            "action": action,
            "successor_hash": _sha(f"succ-{index}"),
            "split": split,
            "seed": 57_900 + (index % 3),
            "agent_owned": True,
            "provenance": "live_agent_observation_receipt",
            "action_valid": True,
            "step_index": index,
            "terminal_before": False,
            "terminal_after": False,
            "object_effect_count": 5,
            "policy_votes": ["left", "left"],
            "counterfactual_successor_hashes": [_sha(f"succ-{index}")],
            "play_cost": 1.0,
        }
        rows.append(row)

    rows[2]["reversal_observed"] = True
    rows[7]["terminal_after"] = True
    rows[7]["play_cost"] = 4.0
    rows[8]["object_effect_count"] = 1
    rows[8]["play_cost"] = 3.0
    rows[9]["policy_votes"] = ["left", "right"]
    rows[9]["play_cost"] = 2.0
    rows[10]["counterfactual_successor_hashes"] = [_sha("succ-10"), _sha("alt-10")]
    rows[10]["play_cost"] = 2.5
    return rows


def _hypothesis(model_id: str, predictions: dict[str, str], **overrides: Any) -> dict[str, Any]:
    data: dict[str, Any] = {
        "model_id": model_id,
        "immutable": True,
        "syntax_compile_passed": True,
        "sandbox_passed": True,
        "executed_through_live_e3": False,
        "edited_after_freeze": False,
        "cycle_consistency_score": 0.4,
        "closed_loop_proxy_utility": 0.1,
        "predictions": predictions,
    }
    data.update(overrides)
    return data


def _perfect_hypothesis() -> dict[str, Any]:
    return _hypothesis(
        "perfect_fixture",
        {row["row_id"]: row["successor_hash"] for row in _transition_rows()},
        cycle_consistency_score=0.8,
        closed_loop_proxy_utility=0.5,
    )


def _rare_rule_omitter() -> dict[str, Any]:
    predictions = {row["row_id"]: row["successor_hash"] for row in _transition_rows()}
    for row in _transition_rows():
        if row["row_id"] in {"t07", "t08", "t09", "t10"}:
            predictions[row["row_id"]] = _sha(f"wrong-{row['row_id']}")
    return _hypothesis(
        "rare_rule_omitter",
        predictions,
        cycle_consistency_score=1.0,
        closed_loop_proxy_utility=0.6,
    )


def _preconditions_fixture() -> dict[str, Any]:
    registry = {
        "source": str(mod.REGISTRY_RELATIVE_PATH),
        "registry_hash": _sha("registry"),
        "checked_before_scoring": True,
        "public_game_count": mod.PUBLIC_GAME_COUNT,
        "registry_level_count": mod.REGISTRY_LEVEL_COUNT,
        "full_game_clear_count": mod.PUBLIC_GAME_COUNT,
        "all_public_games_complete": True,
        "no_public_level_can_be_credited_as_new": True,
        "ok": True,
    }
    return {
        "ok": True,
        "failures": [],
        "registry_precheck": registry,
        "live_e3_entrypoint_hash": {"present": True, "sha256": _sha("e3")},
        "retained_development_proxy_hashes": {
            "exp5764": {"present": True, "sha256": _sha("5764")},
            "exp5764_shard": {"present": True, "sha256": _sha("5764-shard")},
        },
        "agent_owned_trace_manifest_hashes": {
            "exp5766_interaction_audit": {"present": True, "sha256": _sha("5766")}
        },
        "source_denylist_hash": mod.sha256_json(mod.SOURCE_GAME_IDENTITY_DENYLIST),
        "disk_ram": {"ok": True, "disk_free_mb": 999, "ram_free_mb": 999},
        "replay_environment": {"ok": True, "python": "test", "deterministic_replay": True},
        "trace_provenance_ok": True,
    }


def _retained_rows() -> list[dict[str, Any]]:
    rows = []
    for index, game in enumerate(("aa00", "bb00", "cc00")):
        rows.append(
            {
                "game": game,
                "trial": index,
                "heldout_accuracy": [0.9, 0.8, 0.7][index],
                "cell_recall": [0.95, 0.85, 0.75][index],
                "goal_predicate_accuracy": 1.0,
                "levelup_positive_recall": [1.0, 0.0, 1.0][index],
                "plan_found": index != 1,
                "reached_levelup": False,
                "is_memorizing": index == 1,
                "induction_ok": True,
                "error": None,
            }
        )
    rows.append(
        {
            "game": "dd00",
            "trial": 3,
            "heldout_accuracy": None,
            "cell_recall": None,
            "goal_predicate_accuracy": None,
            "levelup_positive_recall": None,
            "plan_found": False,
            "reached_levelup": False,
            "is_memorizing": False,
            "induction_ok": False,
            "error": "retained row intentionally malformed for metric-skip coverage",
        }
    )
    return rows


def test_req_arc_wmte_5790_spec_declares_admission_contract() -> None:
    """REQ-ARC-WMTE-5790: OpenSpec lists every required field and scenario."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    section = spec[spec.index("### REQ-ARC-WMTE-5790") :]
    normalized = " ".join(section.split())

    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert f"`{field}`" in section
        assert " ".join(principle.split()) in normalized
    for marker in (
        "SCENARIO-ARC-WMTE-5790-LEAKAGE-AND-PROVENANCE-REJECTION",
        "SCENARIO-ARC-WMTE-5790-PIVOTAL-FREEZE-AND-METRICS",
        "SCENARIO-ARC-WMTE-5790-ADMISSION-DECISIONS-NO-CREDIT",
        str(mod.RESULT_RELATIVE_PATH),
        "L0 syntax/compile and sandbox",
        "A2RBench-style cycle consistency",
        mod.INFERENCE_SUBSTRATE,
    ):
        assert marker in section or marker in normalized


def test_scenario_arc_wmte_5790_leakage_and_provenance_rejection() -> None:
    """SCENARIO-ARC-WMTE-5790-LEAKAGE-AND-PROVENANCE-REJECTION."""

    rows = _transition_rows()
    leaked = deepcopy(rows[0])
    leaked["row_id"] = "source_leak"
    leaked["source_rule"] = "environment_files/private_rule.py"
    missing = deepcopy(rows[1])
    missing["row_id"] = "missing_provenance"
    missing["agent_owned"] = False

    receipt = mod.validate_agent_owned_transition_rows([*rows, leaked, missing])
    clean_hypothesis = _perfect_hypothesis()
    source_hypothesis = _hypothesis(
        "source_hypothesis",
        clean_hypothesis["predictions"],
        source_file="environment_files/source.py",
    )

    assert len(receipt["accepted_rows"]) == len(rows)
    assert receipt["rejected_count"] == 2
    assert receipt["all_agent_owned_provenance"] is False
    assert receipt["admitted_source_leak_count"] == 0
    assert "source" in receipt["rejections"][0]["violation_classes"]
    assert mod.hypothesis_leak_receipt(source_hypothesis)["rejected"] is True
    assert mod.hypothesis_leak_receipt(clean_hypothesis)["rejected"] is False

    one_row_pivotal = mod.freeze_pivotal_definition(rows[:1])
    no_predictions = _hypothesis("no_predictions", {})
    no_predictions.pop("predictions")
    no_predictions_score = mod.score_hypothesis(no_predictions, rows[:1], one_row_pivotal)
    assert no_predictions_score["ordinary"]["exact_accuracy"] == pytest.approx(0.0)
    assert no_predictions_score["rollout"]["seed_stability"] == pytest.approx(1.0)

    low_retained = mod.rescore_retained_single_shot(
        [{"heldout_accuracy": 0.0, "cell_recall": 1.0, "levelup_positive_recall": 1.0}]
    )
    assert low_retained["admission_decision"]["failed_rung"] == "L2"


def test_scenario_arc_wmte_5790_pivotal_freeze_and_metrics() -> None:
    """SCENARIO-ARC-WMTE-5790-PIVOTAL-FREEZE-AND-METRICS."""

    rows = _transition_rows()
    pivotal = mod.freeze_pivotal_definition(rows)
    metrics = mod.score_hypothesis(_rare_rule_omitter(), rows, pivotal)

    assert pivotal["pivotal_definition_freeze_hash"] == mod.sha256_json(pivotal["definition"])
    assert set(pivotal["definition"]["strata"]) == {
        "observed_action_reversal",
        "terminal_or_goal_state_change",
        "rare_object_effect",
        "policy_disagreement",
        "counterfactual_successor_sensitivity",
    }
    assert metrics["ordinary"]["exact_accuracy"] == pytest.approx(0.8)
    assert metrics["pivotal"]["pivotal_accuracy"] == pytest.approx(0.2)
    assert metrics["pivotal"]["pivotal_coverage_passed"] is False
    assert metrics["play_cost_weighted_risk"]["weighted_miss_risk"] > 0.0
    assert metrics["decision"]["admitted"] is False
    assert metrics["decision"]["failed_rung"] == "L4"


def test_scenario_arc_wmte_5790_artifact_schema_no_credit_and_ready_score(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """SCENARIO-ARC-WMTE-5790-ADMISSION-DECISIONS-NO-CREDIT."""

    monkeypatch.setattr(mod, "structured_preconditions", lambda **_kw: _preconditions_fixture())
    monkeypatch.setattr(mod, "load_agent_owned_transition_rows", lambda *_args, **_kw: _transition_rows())
    monkeypatch.setattr(mod, "load_retained_single_shot_rows", lambda *_args, **_kw: _retained_rows())

    artifact = mod.build_artifact(
        root=tmp_path,
        test_commands=TEST_COMMANDS,
        test_exit_codes=TEST_EXIT_CODES,
    )
    saved_path = mod.write_output(tmp_path, artifact)
    saved = json.loads(saved_path.read_text(encoding="utf-8"))

    assert saved == artifact
    assert tuple(saved) == mod.REQUIRED_ARTIFACT_FIELDS
    assert saved["status"] == "complete"
    assert saved["solve_claimed"] is False
    assert saved["registry_credit"] is False
    assert saved["spec_refs"] == list(mod.SPEC_REFS)
    assert tuple(saved["admission_rung_contract"]) == ("L0", "L1", "L2", "L3", "L4")
    assert saved["pivotal_definition_freeze_hash"] == mod.sha256_json(saved["pivotal_definition"])
    assert saved["ordinary_transition_metrics"]["positive_control_exact_accuracy"] == pytest.approx(1.0)
    assert saved["pivotal_transition_metrics"]["rare_rule_omitter_pivotal_accuracy"] < 0.8
    assert saved["cycle_consistency_negative_control"]["cycle_consistency_score"] == pytest.approx(1.0)
    assert saved["cycle_consistency_negative_control"]["admitted"] is False
    assert {row["canary"] for row in saved["adversarial_canary_receipts"]} == set(mod.CANARY_NAMES)
    assert all(row["rejected"] is True for row in saved["adversarial_canary_receipts"])
    assert saved["retained_model_rescore"]["model_id"] == "gemma31b_singleshot_5764_retained"
    assert saved["retained_model_rescore"]["executed_through_live_e3"] is False
    assert saved["admission_decisions"]["perfect_fixture"]["admitted"] is True
    assert saved["admission_decisions"]["gemma31b_singleshot_5764_retained"]["admitted"] is False
    assert saved["admission_decisions"]["gemma31b_singleshot_5764_retained"]["failed_rung"] == "L4"
    assert saved["producer_gate_fields"] == list(mod.PRODUCER_GATE_FIELDS)
    assert saved["pivotal_fixture_coverage_score"] == pytest.approx(1.0)
    assert saved["source_leak_count"] == 0
    assert saved["admission_contract_ready_score"] == pytest.approx(1.0)
    assert saved["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert saved["test_commands"] == TEST_COMMANDS
    assert saved["test_exit_codes"] == TEST_EXIT_CODES
    assert saved["reproducibility_checksum"] == mod.payload_checksum(saved)
    assert saved["honest_verdict"].startswith("complete:")
    mod.validate_artifact(saved)


def test_req_arc_wmte_5790_validation_rejects_manual_overclaims(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """REQ-ARC-WMTE-5790: schema validation fails closed on unsafe manual edits."""

    monkeypatch.setattr(mod, "structured_preconditions", lambda **_kw: _preconditions_fixture())
    monkeypatch.setattr(mod, "load_agent_owned_transition_rows", lambda *_args, **_kw: _transition_rows())
    monkeypatch.setattr(mod, "load_retained_single_shot_rows", lambda *_args, **_kw: _retained_rows())
    artifact = mod.build_artifact(root=tmp_path)

    mutators = [
        ("required field order", lambda data: data.__setitem__("status", data.pop("status"))),
        ("solve_claimed", lambda data: data.__setitem__("solve_claimed", True)),
        ("registry_credit", lambda data: data.__setitem__("registry_credit", True)),
        ("producer_gate_fields", lambda data: data.__setitem__("producer_gate_fields", [])),
        ("producer_gate_fields", lambda data: data.__setitem__("source_leak_count", {})),
        ("pivotal_definition_freeze_hash", lambda data: data.__setitem__("pivotal_definition", {})),
        ("source_leak_count", lambda data: data.__setitem__("source_leak_count", 1)),
        (
            "cycle_consistency_negative_control",
            lambda data: data["cycle_consistency_negative_control"].__setitem__("admitted", True),
        ),
        (
            "adversarial_canary_receipts",
            lambda data: data["adversarial_canary_receipts"][0].__setitem__("rejected", False),
        ),
        (
            "admission_decisions",
            lambda data: data["admission_decisions"]["rare_rule_omitter"].__setitem__("admitted", True),
        ),
        (
            "inference_substrate",
            lambda data: data.__setitem__("inference_substrate", "live_llm_inference"),
        ),
        ("honest_verdict", lambda data: data.__setitem__("honest_verdict", "ok")),
        (
            "reproducibility_checksum",
            lambda data: data["ordinary_transition_metrics"].__setitem__(
                "positive_control_exact_accuracy", 0.5
            ),
        ),
    ]

    for message, mutate in mutators:
        bad = deepcopy(artifact)
        mutate(bad)
        with pytest.raises(ValueError, match=message):
            mod.validate_artifact(bad)


def test_req_arc_wmte_5790_repository_artifact_is_schema_valid() -> None:
    """REQ-ARC-WMTE-5790: checked-in JSON is the stable terminal admission receipt."""

    artifact = json.loads(RESULT_PATH.read_text(encoding="utf-8"))

    mod.validate_artifact(artifact)
    assert artifact["status"] == "complete"
    assert artifact["solve_claimed"] is False
    assert artifact["registry_credit"] is False
    assert artifact["pivotal_fixture_coverage_score"] == pytest.approx(1.0)
    assert artifact["source_leak_count"] == 0
    assert artifact["admission_contract_ready_score"] == pytest.approx(1.0)
