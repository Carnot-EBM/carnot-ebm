"""Tests for Exp5494 ARC live trajectory/option induction.

Spec refs: REQ-ARC-FCP-5494,
SCENARIO-ARC-FCP-5494.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import yaml

from carnot import experiment_5494_arc_live_trajectory_option_induction_v498 as exp5494


pytestmark = pytest.mark.memory_watchdog_skip

REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "arc-human-replay-frame-change" / "spec.md"


def _registry(dc22_levels: int = 2) -> dict[str, Any]:
    return {
        "reproducible_total_levels": 69,
        "games": [
            {
                "game": "dc22",
                "reproducibility": "reproduced",
                "levels_reproduced": dc22_levels,
                "mechanic_class": "config_toggle_navigation",
            }
        ],
    }


def _exp5493(*, ready: bool = True, preconditions: list[str] | None = None) -> dict[str, Any]:
    if preconditions is None:
        preconditions = [
            "runtime_action_effect_observations",
            "visible_toggle_or_navigation_state_changes",
            "level_counter_delta_read_from_frames",
            "frontier_prefixes_grouped_into_options",
        ]
    return {
        "arc_trajectory_precheck_ready": ready,
        "selected_game": "dc22",
        "selected_target_level": 3,
        "prior_levels_reproduced": 2,
        "excluded_recent_no_bank_targets": [
            "sb26:L3",
            "bp35:L3",
            "ka59:L2",
            "cn04:L4",
            "re86:L3",
        ],
        "trajectory_induction_preconditions": preconditions,
        "candidate_audit": {"dc22:L3": {"decision": "selected"}},
    }


def _preconditions() -> dict[str, Any]:
    return {
        "AGENTS.md": True,
        "CODEX.md": True,
        "spec_has_req_5494": True,
        "registry_present": True,
        "exp5493_present": True,
        "offline_arcade_available": True,
        "offline_bfs_used": False,
        "per_game_adapter_used": False,
        "game_source_read": False,
    }


def _selection() -> dict[str, Any]:
    return exp5494.select_target_from_exp5493(_exp5493(), _registry())


def _null_attempt() -> dict[str, Any]:
    return {
        "live_attempt_count": 5,
        "post_levels_reproduced": 2,
        "offline_reproduced": False,
        "trajectory_hypotheses": [
            {
                "sequence": [{"action": 6, "data": {"x": 10, "y": 20}}],
                "replayable": True,
                "source": "unit_hypothesis",
            }
        ],
        "observation_deltas": [
            {
                "action": 6,
                "data": {"x": 10, "y": 20},
                "changed_cells": 4,
                "level_before": 2,
                "level_after": 2,
            }
        ],
        "verifier_checks": [{"accepted": False, "reason": "uncertainty"}],
        "rejection_reasons": ["bounded_budget_no_target_level_reproduction"],
        "failure_mode": "bounded_budget_no_target_level_reproduction",
        "offline_bfs_used": False,
        "per_game_adapter_used": False,
        "game_source_read": False,
        "llm_generator_invoked": False,
        "model_specs_if_llm_used": [],
    }


def _success_attempt() -> dict[str, Any]:
    return {
        **_null_attempt(),
        "live_attempt_count": 7,
        "post_levels_reproduced": 3,
        "offline_reproduced": True,
        "reproduced_levels": 1,
        "failure_mode": "",
        "rejection_reasons": [],
        "solution_labels": ['{"action":6,"data":{"x":10,"y":20}}'],
        "reproduction_gate": {
            "game": "dc22",
            "claimed_level": 3,
            "reached_level": 3,
            "reproduced": True,
        },
    }


def test_req_arc_fcp_5494_spec_declares_required_artifact_fields() -> None:
    """REQ-ARC-FCP-5494: OpenSpec anchors the Exp5494 artifact contract."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-ARC-FCP-5494" in spec
    assert "SCENARIO-ARC-FCP-5494" in spec
    assert exp5494.RESULT_RELATIVE_PATH in spec
    for field, principle in exp5494.FIELD_PRINCIPLES.items():
        assert f"`{field}`" in spec
        assert principle["principle"] in spec


def test_scenario_arc_fcp_5494_precheck_selects_dc22_and_refuses_bad_targets() -> None:
    """SCENARIO-ARC-FCP-5494: target gates run before any live budget is spent."""

    selected = exp5494.select_target_from_exp5493(_exp5493(), _registry())
    requested_game_mismatch = exp5494.select_target_from_exp5493(
        _exp5493(),
        _registry(),
        requested_game="sb26",
    )
    requested_target_mismatch = exp5494.select_target_from_exp5493(
        _exp5493(),
        _registry(),
        requested_target_level=4,
    )
    requested_prior_mismatch = exp5494.select_target_from_exp5493(
        _exp5493(),
        _registry(),
        requested_prior_levels=1,
    )
    duplicate = exp5494.select_target_from_exp5493(_exp5493(), _registry(dc22_levels=3))
    stale = exp5494.select_target_from_exp5493(
        {**_exp5493(), "excluded_recent_no_bank_targets": ["dc22:L3"]},
        _registry(),
    )
    missing_target = exp5494.select_target_from_exp5493(
        {**_exp5493(), "selected_game": "", "selected_target_level": 0},
        _registry(),
    )
    missing_row = exp5494.select_target_from_exp5493(
        {**_exp5493(), "selected_game": "zz99"},
        _registry(),
    )
    not_selected = exp5494.select_target_from_exp5493(
        {**_exp5493(), "candidate_audit": {"dc22:L3": {"decision": "rejected"}}},
        _registry(),
    )
    missing_preconditions = exp5494.select_target_from_exp5493(
        _exp5493(preconditions=["runtime_action_effect_observations"]),
        _registry(),
    )
    not_ready = exp5494.select_target_from_exp5493(_exp5493(ready=False), _registry())

    assert selected["blocked"] is False
    assert selected["selected_game"] == "dc22"
    assert selected["target_level"] == 3
    assert selected["prior_levels_reproduced"] == 2
    assert selected["duplicate_solve_avoided"] is True
    assert requested_game_mismatch["blocker"] == "requested_game_mismatch"
    assert requested_target_mismatch["blocker"] == "requested_target_level_mismatch"
    assert requested_prior_mismatch["blocker"] == "requested_prior_levels_mismatch"
    assert duplicate["blocker"] == "target_already_reproduced"
    assert stale["blocker"] == "recent_same_mechanism_no_bank"
    assert missing_target["blocker"] == "missing_exp5493_target"
    assert missing_row["blocker"] == "missing_reproduced_registry_row"
    assert not_selected["blocker"] == "exp5493_candidate_not_selected"
    assert missing_preconditions["blocker"] == "missing_live_trajectory_preconditions"
    assert not_ready["blocker"] == "exp5493_precheck_not_ready"


def test_req_arc_fcp_5494_helper_diagnostics_cover_fallbacks(tmp_path: Path) -> None:
    """REQ-ARC-FCP-5494: helper fallbacks are deterministic and auditable."""

    missing_json = exp5494._load_json(tmp_path / "missing.json")  # noqa: SLF001
    missing_yaml = exp5494._load_yaml(tmp_path / "missing.yaml")  # noqa: SLF001
    (tmp_path / "ops").mkdir()
    (tmp_path / exp5494.REGISTRY_RELATIVE_PATH).write_text("", encoding="utf-8")

    receipt_hypotheses = exp5494._hypotheses_from_diagnostics(  # noqa: SLF001
        {
            "action_sequence_receipts": [
                "bad-receipt",
                {
                    "sequence": [{"action": 6, "data": {"x": 1, "y": 2}}],
                    "measurement_receipts": [{"receipt_id": "m0001"}],
                }
            ]
        },
        [],
    )
    fallback_hypotheses = exp5494._hypotheses_from_diagnostics(  # noqa: SLF001
        {},
        [
            exp5494._action_label(6, {"x": 1, "y": 2}),  # noqa: SLF001
            "RESET",
            "not-json",
        ],
    )
    no_hypotheses = exp5494._hypotheses_from_diagnostics({}, [])  # noqa: SLF001
    runtime_deltas = exp5494._observation_deltas_from_diagnostics(  # noqa: SLF001
        {"runtime_observations": [{"changed_cells": 3}]}
    )
    receipt_deltas = exp5494._observation_deltas_from_diagnostics(  # noqa: SLF001
        {
            "measurement_access_receipts": [
                {
                    "action": 6,
                    "data": {"x": 1},
                    "changed_cells": "5",
                    "level_before": 2,
                    "level_after": 2,
                    "receipt_id": "m0002",
                }
            ]
        }
    )
    checks = exp5494._verifier_checks_from_diagnostics(  # noqa: SLF001
        {"verifier_observations": [{"accepted": True}]},
        {"reproduced": False, "claimed_level": 3, "reached_level": 2},
    )
    rejection_reasons = exp5494._rejection_reasons(  # noqa: SLF001
        diagnostics={"uncertainty_rejections": 1, "frontier_expansion_count": 0},
        reproduced=False,
        failure_mode="bounded",
    )

    assert missing_json == {}
    assert missing_yaml == {"reproducible_total_levels": 0, "games": []}
    assert exp5494.load_registry(tmp_path) == {"reproducible_total_levels": 0, "games": []}
    assert exp5494._as_int("bad", 4) == 4  # noqa: SLF001
    assert exp5494._label_to_step("RESET") is None  # noqa: SLF001
    assert exp5494._label_to_step("not-json") is None  # noqa: SLF001
    assert receipt_hypotheses[0]["source"] == "live_coex_action_sequence_receipt"
    assert fallback_hypotheses[0]["source"] == "executed_live_action_prefix_fallback"
    assert no_hypotheses == []
    assert runtime_deltas == [{"changed_cells": 3}]
    assert receipt_deltas[0]["changed_cells"] == 5
    assert checks[-1]["check"] == "standard_live_offline_reproduction_gate"
    assert exp5494._rejection_reasons(  # noqa: SLF001
        diagnostics={},
        reproduced=True,
        failure_mode="",
    ) == []
    assert rejection_reasons == [
        "bounded",
        "uncertainty_gate_rejected_low_support_options",
        "no_accepted_trajectory_prefix",
    ]
    assert exp5494._accepted_reproduced_levels(  # noqa: SLF001
        _selection(),
        {**_success_attempt(), "offline_bfs_used": True},
    ) == 0
    assert exp5494._accepted_reproduced_levels(  # noqa: SLF001
        _selection(),
        {**_success_attempt(), "post_levels_reproduced": 2},
    ) == 0


def test_scenario_arc_fcp_5494_null_artifact_records_trajectory_diagnostics() -> None:
    """SCENARIO-ARC-FCP-5494: honest nulls preserve hypotheses and deltas."""

    artifact = exp5494.build_artifact(
        selection=_selection(),
        attempt=_null_attempt(),
        registry_updated=False,
        preconditions_checked=_preconditions(),
        tests_run=["unit 5494"],
        duration_s=0.1,
    )

    exp5494.validate_artifact(artifact)
    assert artifact["selected_game"] == "dc22"
    assert artifact["target_level"] == 3
    assert artifact["prior_levels_reproduced"] == 2
    assert artifact["post_levels_reproduced"] == 2
    assert artifact["trajectory_hypothesis_count"] == 1
    assert artifact["live_attempt_count"] == 5
    assert artifact["offline_reproduced"] is False
    assert artifact["reproduced_levels"] == 0
    assert artifact["new_level_banked"] is False
    assert artifact["registry_updated"] is False
    assert artifact["model_specs_if_llm_used"] == []
    assert artifact["inference_substrate"] == exp5494.INFERENCE_SUBSTRATE
    assert artifact["honest_verdict"].startswith("honest_null:")
    assert artifact["trajectory_hypotheses"][0]["source"] == "unit_hypothesis"
    assert artifact["observation_deltas"][0]["changed_cells"] == 4
    assert artifact["verifier_checks"][0]["reason"] == "uncertainty"
    assert artifact["rejection_reasons"] == ["bounded_budget_no_target_level_reproduction"]


def test_scenario_arc_fcp_5494_success_updates_temp_registry_only_after_gate(
    tmp_path: Path,
) -> None:
    """SCENARIO-ARC-FCP-5494: registry updates require reproduced new depth."""

    root = tmp_path
    (root / "ops").mkdir()
    (root / exp5494.REGISTRY_RELATIVE_PATH).write_text(
        yaml.safe_dump(_registry()),
        encoding="utf-8",
    )
    artifact = exp5494.build_artifact(
        selection=_selection(),
        attempt=_success_attempt(),
        registry_updated=True,
        preconditions_checked=_preconditions(),
        tests_run=["unit 5494"],
        duration_s=0.1,
    )
    updated = exp5494.update_registry_if_banked(root=root, artifact=artifact, registry=_registry())
    registry_after = yaml.safe_load((root / exp5494.REGISTRY_RELATIVE_PATH).read_text())

    exp5494.validate_artifact(artifact)
    assert updated is True
    assert artifact["offline_reproduced"] is True
    assert artifact["reproduced_levels"] == 1
    assert artifact["new_level_banked"] is True
    assert artifact["registry_updated"] is True
    assert artifact["honest_verdict"].startswith("complete:")
    assert registry_after["reproducible_total_levels"] == 70
    assert registry_after["games"][0]["levels_reproduced"] == 3
    assert registry_after["games"][0]["latest_exp5494_levelup_attempt"]["artifact"] == (
        exp5494.RESULT_RELATIVE_PATH
    )


def test_scenario_arc_fcp_5494_schema_rejects_prohibited_paths_and_bad_llm_specs() -> None:
    """REQ-ARC-FCP-5494: schema rejects off-path credit and legacy LLM headlines."""

    artifact = exp5494.build_artifact(
        selection=_selection(),
        attempt=_null_attempt(),
        registry_updated=False,
        preconditions_checked=_preconditions(),
        tests_run=["unit 5494"],
        duration_s=0.1,
    )
    invalid = {
        **artifact,
        "solve_provenance": "outer_loop_re",
        "offline_bfs_used": True,
        "per_game_adapter_used": True,
        "game_source_read": True,
        "trajectory_hypothesis_count": "1",
        "live_attempt_count": -1,
        "offline_reproduced": True,
        "reproduced_levels": 1,
        "new_level_banked": True,
        "registry_updated": False,
        "model_specs_if_llm_used": ["legacy-small-model"],
        "honest_verdict": "complete: solved dc22 L3",
    }

    errors = exp5494.artifact_schema_errors(invalid)
    with pytest.raises(ValueError):
        exp5494.validate_artifact(invalid)

    assert "solve_provenance must be live_agent_self_discovery" in errors
    assert "offline_bfs_used must be false" in errors
    assert "per_game_adapter_used must be false" in errors
    assert "game_source_read must be false" in errors
    assert "trajectory_hypothesis_count must be bare int" in errors
    assert "live_attempt_count must be non-negative" in errors
    assert "new_level_banked requires registry_updated true" in errors
    assert "model_specs_if_llm_used missing unsloth/Qwen3.6-35B-A3B-GGUF" in errors
    assert "honest_verdict must not claim an unreproduced solve" in errors


def test_req_arc_fcp_5494_schema_rejects_type_and_consistency_edges() -> None:
    """REQ-ARC-FCP-5494: schema catches malformed fields not built by the runner."""

    artifact = exp5494.build_artifact(
        selection=_selection(),
        attempt=_null_attempt(),
        registry_updated=False,
        preconditions_checked=_preconditions(),
        tests_run=["unit 5494"],
        duration_s=0.1,
    )
    malformed = {
        **artifact,
        "status": "done",
        "selected_game": 7,
        "offline_reproduced": "false",
        "new_level_banked": "false",
        "registry_updated": True,
        "offline_bfs_used": "false",
        "trajectory_hypotheses": {},
        "observation_deltas": {},
        "verifier_checks": {},
        "rejection_reasons": {},
        "model_specs_if_llm_used": "legacy",
        "post_levels_reproduced": 1,
        "inference_substrate": "wrong",
        "honest_verdict": "pending",
    }
    offline_without_delta = {
        **artifact,
        "offline_reproduced": True,
        "reproduced_levels": 0,
        "post_levels_reproduced": 2,
    }
    bank_without_reproduction = {
        **artifact,
        "new_level_banked": True,
        "offline_reproduced": False,
        "registry_updated": True,
    }

    errors = exp5494.artifact_schema_errors(malformed)
    offline_errors = exp5494.artifact_schema_errors(offline_without_delta)
    bank_errors = exp5494.artifact_schema_errors(bank_without_reproduction)

    assert "status must be complete, honest_null, or blocked" in errors
    assert "selected_game must be a string" in errors
    assert "offline_reproduced must be bare bool" in errors
    assert "new_level_banked must be bare bool" in errors
    assert "offline_bfs_used must be bare bool" in errors
    assert "trajectory_hypotheses must be a list" in errors
    assert "observation_deltas must be a list" in errors
    assert "verifier_checks must be a list" in errors
    assert "rejection_reasons must be a list" in errors
    assert "model_specs_if_llm_used must be a list" in errors
    assert "post_levels_reproduced must be >= prior_levels_reproduced" in errors
    assert "registry_updated requires new_level_banked true" in errors
    assert "inference_substrate must be arc_live_agent_self_discovery" in errors
    assert "honest_verdict must start with complete:, honest_null:, or blocked:" in errors
    assert "offline_reproduced requires reproduced_levels >= 1" in offline_errors
    assert "offline_reproduced requires post_levels_reproduced > prior_levels_reproduced" in offline_errors
    assert "new_level_banked requires offline_reproduced true" in bank_errors


def test_scenario_arc_fcp_5494_run_experiment_writes_json_with_fake_runner(
    tmp_path: Path,
) -> None:
    """SCENARIO-ARC-FCP-5494: runner writes the deliverable JSON."""

    root = tmp_path
    (root / "openspec" / "capabilities" / "arc-human-replay-frame-change").mkdir(parents=True)
    (root / "ops").mkdir()
    (root / "results").mkdir()
    (root / "AGENTS.md").write_text("# AGENTS\n", encoding="utf-8")
    (root / "CODEX.md").write_text("# CODEX\n", encoding="utf-8")
    (root / exp5494.SPEC_RELATIVE_PATH).write_text(
        "REQ-ARC-FCP-5494\nSCENARIO-ARC-FCP-5494\n",
        encoding="utf-8",
    )
    (root / exp5494.REGISTRY_RELATIVE_PATH).write_text(
        yaml.safe_dump(_registry()),
        encoding="utf-8",
    )
    (root / exp5494.EXP5493_RELATIVE_PATH).write_text(
        json.dumps(_exp5493()),
        encoding="utf-8",
    )

    artifact = exp5494.run_experiment(
        root=root,
        attempt_runner=lambda **_kwargs: _null_attempt(),
        offline_arcade_check=lambda: True,
        tests_run=["unit 5494"],
    )
    written = json.loads((root / exp5494.RESULT_RELATIVE_PATH).read_text(encoding="utf-8"))

    assert written == artifact
    assert artifact["selected_game"] == "dc22"
    assert artifact["target_level"] == 3
    assert artifact["prior_levels_reproduced"] == 2
    assert artifact["post_levels_reproduced"] == 2
    assert artifact["new_level_banked"] is False
    assert artifact["registry_updated"] is False


def test_scenario_arc_fcp_5494_run_experiment_blocks_before_attempt(
    tmp_path: Path,
) -> None:
    """REQ-ARC-FCP-5494: missing preconditions block without live attempts."""

    root = tmp_path
    (root / "openspec" / "capabilities" / "arc-human-replay-frame-change").mkdir(parents=True)
    (root / "ops").mkdir()
    (root / "results").mkdir()
    (root / "AGENTS.md").write_text("# AGENTS\n", encoding="utf-8")
    (root / "CODEX.md").write_text("# CODEX\n", encoding="utf-8")
    (root / exp5494.SPEC_RELATIVE_PATH).write_text("REQ-ARC-FCP-5494\n", encoding="utf-8")
    (root / exp5494.REGISTRY_RELATIVE_PATH).write_text(
        yaml.safe_dump(_registry()),
        encoding="utf-8",
    )
    (root / exp5494.EXP5493_RELATIVE_PATH).write_text(
        json.dumps(_exp5493(preconditions=["runtime_action_effect_observations"])),
        encoding="utf-8",
    )

    artifact = exp5494.run_experiment(
        root=root,
        attempt_runner=lambda **_kwargs: pytest.fail("attempt runner must not be called"),
        offline_arcade_check=lambda: pytest.fail("arcade check must not be called"),
        tests_run=["unit 5494"],
    )

    assert artifact["status"] == "blocked"
    assert artifact["live_attempt_count"] == 0
    assert artifact["failure_mode"] == "missing_live_trajectory_preconditions"
    assert artifact["honest_verdict"].startswith("blocked:")


def test_scenario_arc_fcp_5494_run_experiment_blocks_on_missing_harness(
    tmp_path: Path,
) -> None:
    """REQ-ARC-FCP-5494: missing live harness emits blocked diagnostics."""

    root = tmp_path
    (root / "openspec" / "capabilities" / "arc-human-replay-frame-change").mkdir(parents=True)
    (root / "ops").mkdir()
    (root / "results").mkdir()
    (root / "AGENTS.md").write_text("# AGENTS\n", encoding="utf-8")
    (root / "CODEX.md").write_text("# CODEX\n", encoding="utf-8")
    (root / exp5494.SPEC_RELATIVE_PATH).write_text("REQ-ARC-FCP-5494\n", encoding="utf-8")
    (root / exp5494.REGISTRY_RELATIVE_PATH).write_text(
        yaml.safe_dump(_registry()),
        encoding="utf-8",
    )
    (root / exp5494.EXP5493_RELATIVE_PATH).write_text(
        json.dumps(_exp5493()),
        encoding="utf-8",
    )

    artifact = exp5494.run_experiment(
        root=root,
        attempt_runner=lambda **_kwargs: pytest.fail("attempt runner must not be called"),
        offline_arcade_check=lambda: False,
        tests_run=["unit 5494"],
    )

    assert artifact["status"] == "blocked"
    assert artifact["failure_mode"] == "missing_harness_access"
    assert artifact["preconditions_checked"]["offline_arcade_available"] is False


def test_scenario_arc_fcp_5494_run_experiment_success_updates_registry(
    tmp_path: Path,
) -> None:
    """SCENARIO-ARC-FCP-5494: runner updates registry only for a bankable attempt."""

    root = tmp_path
    (root / "openspec" / "capabilities" / "arc-human-replay-frame-change").mkdir(parents=True)
    (root / "ops").mkdir()
    (root / "results").mkdir()
    (root / "AGENTS.md").write_text("# AGENTS\n", encoding="utf-8")
    (root / "CODEX.md").write_text("# CODEX\n", encoding="utf-8")
    (root / exp5494.SPEC_RELATIVE_PATH).write_text("REQ-ARC-FCP-5494\n", encoding="utf-8")
    (root / exp5494.REGISTRY_RELATIVE_PATH).write_text(
        yaml.safe_dump(_registry()),
        encoding="utf-8",
    )
    exp5493 = {**_exp5493(), "candidate_audit": {}}
    (root / exp5494.EXP5493_RELATIVE_PATH).write_text(json.dumps(exp5493), encoding="utf-8")

    artifact = exp5494.run_experiment(
        root=root,
        attempt_runner=lambda **_kwargs: _success_attempt(),
        offline_arcade_check=lambda: True,
        tests_run=["unit 5494"],
    )
    registry_after = yaml.safe_load((root / exp5494.REGISTRY_RELATIVE_PATH).read_text())

    assert artifact["status"] == "complete"
    assert artifact["new_level_banked"] is True
    assert artifact["registry_updated"] is True
    assert registry_after["games"][0]["game"] == "dc22"
    assert registry_after["reproducible_total_levels"] == 70


def test_req_arc_fcp_5494_update_registry_noops_for_null(tmp_path: Path) -> None:
    """REQ-ARC-FCP-5494: null artifacts never mutate the registry."""

    artifact = exp5494.build_artifact(
        selection=_selection(),
        attempt=_null_attempt(),
        registry_updated=False,
        preconditions_checked=_preconditions(),
        tests_run=["unit 5494"],
        duration_s=0.1,
    )

    assert exp5494.update_registry_if_banked(
        root=tmp_path,
        artifact=artifact,
        registry=_registry(),
    ) is False
    assert not (tmp_path / exp5494.REGISTRY_RELATIVE_PATH).exists()


def test_req_arc_fcp_5494_update_registry_can_append_missing_game(tmp_path: Path) -> None:
    """REQ-ARC-FCP-5494: success update can create a missing registry row."""

    artifact = exp5494.build_artifact(
        selection=_selection(),
        attempt=_success_attempt(),
        registry_updated=True,
        preconditions_checked=_preconditions(),
        tests_run=["unit 5494"],
        duration_s=0.1,
    )

    updated = exp5494.update_registry_if_banked(
        root=tmp_path,
        artifact=artifact,
        registry={"reproducible_total_levels": 69, "games": []},
    )
    registry_after = yaml.safe_load((tmp_path / exp5494.REGISTRY_RELATIVE_PATH).read_text())

    assert updated is True
    assert registry_after["games"][0]["game"] == "dc22"
