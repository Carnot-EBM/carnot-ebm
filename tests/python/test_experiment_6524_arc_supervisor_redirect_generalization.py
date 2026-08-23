"""Tests for Exp6524 ARC trajectory-supervisor redirect replay.

Spec refs: REQ-ARC-WMTE-6650, SCENARIO-ARC-WMTE-6650-1,
SCENARIO-ARC-WMTE-6650-2, SCENARIO-ARC-WMTE-6650-3,
SCENARIO-ARC-WMTE-6650-4, SCENARIO-ARC-WMTE-6650-5,
SCENARIO-ARC-WMTE-6650-6.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any

import pytest

from carnot import experiment_6524_arc_supervisor_redirect_generalization as mod


REPO = Path(__file__).resolve().parents[2]
SPEC = REPO / mod.SPEC_RELATIVE_PATH
TESTS_RUN = [{"command": command, "exit_code": 0} for command in mod.DEFAULT_TEST_COMMANDS]


def _write_json(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path


def _row(
    game: str,
    receipt: dict[str, Any],
    *,
    seed: int = 20260823,
    arm: str = "E3_default_llmon",
    budget: int = 2000,
    levels: int = 0,
    actions: int = 400,
) -> dict[str, Any]:
    return {
        "arm": arm,
        "game": game,
        "seed": seed,
        "budget": budget,
        "llm_enabled": True,
        "gated_flags": {"tier_exhaustion_enabled": True},
        "levels": levels,
        "reached": levels,
        "actions": actions,
        "level_up_actions": [actions] if levels else [],
        "actions_after_last_levelup": 0 if levels else actions,
        "trajectory_supervisor": receipt,
    }


def _enabled_receipt(
    redirects: list[dict[str, Any]],
    *,
    stagnations_unredirected: int = 0,
) -> dict[str, Any]:
    outcomes = {arm: {"fired": 0, "helped": 0} for arm in mod.ARM_ORDER}
    for redirect in redirects:
        outcomes[redirect["arm"]]["fired"] += 1
        if redirect["resolved_by_levelup"]:
            outcomes[redirect["arm"]]["helped"] += 1
    return {
        "enabled": True,
        "window": 400,
        "actions_observed": 400,
        "arms_used": sorted({row["arm"] for row in redirects}),
        "redirects": redirects,
        "arm_outcomes": outcomes,
        "stagnations_unredirected": stagnations_unredirected,
    }


def _live_artifact(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "schema": "arc_scored_path_lever_harness.rows.v1",
        "canonical_entrypoint": "E3AgentPolicy",
        "rows": rows,
        "arms_used": ["E3_default_llmon"],
        "budget": 2000,
    }


@pytest.fixture()
def supported_inputs(tmp_path: Path) -> list[Path]:
    """REQ-ARC-WMTE-6650: two helped allow_reinduction rows justify a priority raise."""

    redirects = [
        {
            "arm": "allow_reinduction",
            "action_index": 120,
            "level": 0,
            "diagnosis": "induction latch set with 220 new transitions",
            "resolved_by_levelup": True,
            "actions_to_levelup": 8,
        },
        {
            "arm": "allow_reinduction",
            "action_index": 180,
            "level": 1,
            "diagnosis": "induction latch set with 260 new transitions",
            "resolved_by_levelup": True,
            "actions_to_levelup": 6,
        },
        {
            "arm": "drop_goal_bias",
            "action_index": 210,
            "level": 0,
            "diagnosis": "goal bias installed through window",
            "resolved_by_levelup": False,
            "actions_to_levelup": None,
        },
    ]
    first = _row("aa01", _enabled_receipt(redirects[:2]), levels=1, actions=188)
    second = _row("bb02", _enabled_receipt(redirects[2:]), actions=400)
    return [_write_json(tmp_path / "live_rows.json", _live_artifact([first, second]))]


def test_req_arc_wmte_6650_spec_declares_replay_and_no_solve_contract() -> None:
    """REQ-ARC-WMTE-6650: OpenSpec owns the replay/refinement contract."""

    text = SPEC.read_text(encoding="utf-8")
    section = text[text.index("REQ-ARC-WMTE-6650") :]
    normalized = " ".join(section.split())

    for marker in (
        "SCENARIO-ARC-WMTE-6650-1",
        "SCENARIO-ARC-WMTE-6650-2",
        "SCENARIO-ARC-WMTE-6650-3",
        "SCENARIO-ARC-WMTE-6650-4",
        "SCENARIO-ARC-WMTE-6650-5",
        "SCENARIO-ARC-WMTE-6650-6",
        mod.MODULE_RELATIVE_PATH.as_posix(),
        mod.RESULT_RELATIVE_PATH.as_posix(),
        mod.INFERENCE_SUBSTRATE,
        "`verifier_is_oracle` to bare `false`",
    ):
        assert marker in section
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert f"`{field}`" in section
        assert " ".join(principle.split()) in normalized


def test_scenario_6650_1_and_2_supported_selection_replay(
    supported_inputs: list[Path],
    tmp_path: Path,
) -> None:
    """SCENARIO-ARC-WMTE-6650-1/2: live redirects drive curated selection only."""

    artifact = mod.build_artifact(
        repo_root=REPO,
        result_path=tmp_path / mod.RESULT_RELATIVE_PATH.name,
        candidate_paths=supported_inputs,
        write=False,
        duration_s=1.0,
        tests_run=TESTS_RUN,
        run_date="20260823",
    )

    assert artifact["status"] == "complete_supported_supervisor_selection_refinement"
    assert artifact["honest_verdict"].startswith("partial:")
    assert artifact["verdict_class"] == "partial"
    assert artifact["supervisor_refinement_status"] == "supported_curated_arm_priority_refinement"
    assert artifact["arc_generalization_slot_complete_score"] == 1.0
    assert artifact["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert artifact["verifier_is_oracle"] is False
    assert "solve_provenance" not in artifact
    assert artifact["canonical_entrypoint_receipt"]["entrypoint"] == (
        "python/carnot/agentic/arc_competition_agent.py::make_carnot_agent -> E3AgentPolicy"
    )

    rows = artifact["redirect_outcome_rows"]
    assert len(rows) == 3
    assert all(row["fired"] is True for row in rows)
    assert all(row["provenance"]["entrypoint"] == "E3AgentPolicy" for row in rows)

    per_arm = {row["arm"]: row for row in artifact["per_arm_rows"]}
    assert per_arm["allow_reinduction"]["fired"] == 2
    assert per_arm["allow_reinduction"]["helped"] == 2
    assert per_arm["allow_reinduction"]["actions_to_progress_values"] == [8, 6]
    assert per_arm["allow_reinduction"]["recommended_action"] == "raise_priority"
    assert per_arm["drop_goal_bias"]["failure"] == 1
    assert per_arm["force_exploration_diversity"]["fired"] == 0

    before = artifact["arm_table_before_after"]["before"]
    after = artifact["arm_table_before_after"]["after"]
    assert [row["arm"] for row in before] == list(mod.ARM_ORDER)
    assert [row["arm"] for row in after][0] == "allow_reinduction"
    assert artifact["rollback_receipt"]["rollback_restores_before"] is True
    assert artifact["rollback_receipt"]["rollback_action"] == "restore_arm_table_before"
    assert artifact["gate_check_summary"]["all_gates_passed"] is True
    assert artifact["reproducibility_checksum"] == mod.reproducibility_checksum(artifact)
    assert mod.validate_artifact(artifact) == []


def test_scenario_6650_3_no_firing_and_blocked_closure(tmp_path: Path) -> None:
    """SCENARIO-ARC-WMTE-6650-3: no-firing and missing-outcome closures do not edit arms."""

    no_fire_path = _write_json(
        tmp_path / "no_fire.json",
        _live_artifact([_row("cc03", _enabled_receipt([], stagnations_unredirected=2))]),
    )
    no_fire = mod.build_artifact(
        repo_root=REPO,
        result_path=tmp_path / "no_fire_result.json",
        candidate_paths=[no_fire_path],
        write=False,
        duration_s=1.0,
        tests_run=TESTS_RUN,
        run_date="20260823",
    )
    assert no_fire["status"] == "complete_no_firings_nothing_to_refine"
    assert no_fire["verdict_class"] is None
    assert no_fire["supervisor_refinement_status"] == "no_firings_nothing_to_refine"
    assert no_fire["redirect_outcome_rows"] == []
    assert no_fire["no_firings_receipt"]["outcome_bearing_receipt_count"] == 1
    assert no_fire["arm_table_before_after"]["before"] == no_fire["arm_table_before_after"]["after"]

    disabled_path = _write_json(
        tmp_path / "disabled.json",
        _live_artifact([_row("dd04", {"enabled": False})]),
    )
    blocked = mod.build_artifact(
        repo_root=REPO,
        result_path=tmp_path / "blocked_result.json",
        candidate_paths=[disabled_path],
        write=False,
        duration_s=1.0,
        tests_run=TESTS_RUN,
        run_date="20260823",
    )
    assert blocked["status"] == "blocked_missing_outcome_bearing_live_receipts"
    assert blocked["verdict_class"] == "blocked"
    assert blocked["supervisor_refinement_status"] == "blocked_missing_outcome_bearing_receipts"
    assert blocked["no_firings_receipt"]["disabled_receipt_count"] == 1
    assert blocked["arm_table_before_after"]["before"] == blocked["arm_table_before_after"]["after"]


def test_scenario_6650_4_rejects_off_path_and_deduplicates(tmp_path: Path) -> None:
    """SCENARIO-ARC-WMTE-6650-4: off-path evidence and duplicate receipts fail closed."""

    redirect = {
        "arm": "drop_goal_bias",
        "action_index": 100,
        "level": 0,
        "diagnosis": "bias stale",
        "resolved_by_levelup": True,
        "actions_to_levelup": 5,
    }
    live_payload = _live_artifact([_row("ee05", _enabled_receipt([redirect]), levels=1)])
    duplicate_a = _write_json(tmp_path / "duplicate_a.json", live_payload)
    duplicate_b = _write_json(tmp_path / "duplicate_b.json", live_payload)
    offline = deepcopy(live_payload)
    offline["inference_substrate"] = "offline_adapter_outer_loop_solver_source_reading"
    offline["solve_provenance"] = "outer_loop_re"
    offline_path = _write_json(tmp_path / "offline.json", offline)
    unrelated = _write_json(
        tmp_path / "unrelated_arm_outcomes.json",
        {"per_game_per_seed_per_arm_outcomes": [{"arm_outcomes": {"x": 1}}]},
    )

    artifact = mod.build_artifact(
        repo_root=REPO,
        result_path=tmp_path / mod.RESULT_RELATIVE_PATH.name,
        candidate_paths=[duplicate_a, duplicate_b, offline_path, unrelated],
        write=False,
        duration_s=1.0,
        tests_run=TESTS_RUN,
        run_date="20260823",
    )

    assert len(artifact["redirect_outcome_rows"]) == 1
    audit = artifact["provenance_audit"]
    assert audit["accepted_live_receipt_count"] == 1
    assert audit["duplicate_receipt_count"] == 1
    rejection_reasons = {row["reason"] for row in audit["rejected_artifacts"]}
    assert "off_path_evidence" in rejection_reasons
    assert "no_trajectory_supervisor_rows" in rejection_reasons


def test_scenario_6650_5_and_6_schema_attacks_and_cli(
    supported_inputs: list[Path],
    tmp_path: Path,
) -> None:
    """SCENARIO-ARC-WMTE-6650-5/6: rollback, attack matrix, validation, and CLI are stable."""

    result_path = tmp_path / mod.RESULT_RELATIVE_PATH.name
    assert (
        mod.main(
            [
                "--date",
                "20260823",
                "--result-path",
                str(result_path),
                "--input",
                str(supported_inputs[0]),
            ]
        )
        == 0
    )
    assert mod.main(["--validate", "--result-path", str(result_path)]) == 0
    artifact = json.loads(result_path.read_text(encoding="utf-8"))

    assert set(artifact) == set(mod.REQUIRED_ARTIFACT_FIELDS)
    assert artifact["field_principles"] == mod.FIELD_PRINCIPLES
    assert set(artifact["field_provenance"]) == set(mod.REQUIRED_ARTIFACT_FIELDS)
    assert artifact["protected_files_unchanged"]["all_protected_files_unchanged"] is True
    assert artifact["support_and_tie_contract"]["min_helped_to_raise_priority"] == 2
    assert artifact["support_and_tie_contract"]["min_no_help_to_lower_or_retire"] == 2

    attack_ids = {row["attack_id"] for row in artifact["generalization_attack_matrix"]["rows"]}
    assert attack_ids == set(mod.ATTACK_IDS)
    assert artifact["generalization_attack_matrix"]["all_attacks_fail_closed"] is True
    assert all(row["fail_closed"] is True for row in artifact["generalization_attack_matrix"]["rows"])
    assert artifact["aggregate_row_recomputation"]["redirect_count"] == len(
        artifact["redirect_outcome_rows"]
    )

    mutations = [
        ("required field set mismatch", lambda item: item.pop("status")),
        ("status lacks terminal prefix", lambda item: item.__setitem__("status", "running")),
        ("verdict_class invalid", lambda item: item.__setitem__("verdict_class", "positive")),
        ("substrate mismatch", lambda item: item.__setitem__("inference_substrate", "live_llm")),
        ("oracle must be false", lambda item: item.__setitem__("verifier_is_oracle", True)),
        ("solve_provenance forbidden", lambda item: item.__setitem__("solve_provenance", "live_agent_self_discovery")),
        ("checksum mismatch", lambda item: item.__setitem__("reproducibility_checksum", "sha256:bad")),
        ("attack did not fail closed", lambda item: item["generalization_attack_matrix"]["rows"][0].__setitem__("fail_closed", False)),
        ("attack matrix did not fail closed", lambda item: item.__setitem__("generalization_attack_matrix", {})),
        ("protected files changed", lambda item: item["protected_files_unchanged"].__setitem__("all_protected_files_unchanged", False)),
        ("aggregate redirect_count mismatch", lambda item: item["aggregate_row_recomputation"].__setitem__("redirect_count", -1)),
        ("aggregate rows malformed", lambda item: item.__setitem__("aggregate_row_recomputation", [])),
        ("honest_verdict lacks terminal prefix", lambda item: item.__setitem__("honest_verdict", "running")),
        ("field principles mismatch", lambda item: item.__setitem__("field_principles", {})),
        ("field provenance mismatch", lambda item: item.__setitem__("field_provenance", {})),
    ]
    for expected, mutate in mutations:
        changed = deepcopy(artifact)
        mutate(changed)
        assert any(expected in issue for issue in mod.validate_artifact(changed)), expected

    bad_path = tmp_path / "bad_result.json"
    bad = deepcopy(artifact)
    bad["status"] = "running"
    bad["reproducibility_checksum"] = mod.reproducibility_checksum(bad)
    bad_path.write_text(json.dumps(bad), encoding="utf-8")
    assert mod.main(["--validate", "--result-path", str(bad_path)]) == 1


def test_scenario_6650_4_and_6_malformed_inputs_lowering_and_helper_edges(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SCENARIO-ARC-WMTE-6650-4/6: malformed evidence is rejected and no-help can lower."""

    missing = tmp_path / "missing.json"
    invalid = tmp_path / "invalid.json"
    invalid.write_text("{not json", encoding="utf-8")
    not_live = _write_json(
        tmp_path / "not_live.json",
        {"rows": [{"arm": "plain", "game": "ff06", "trajectory_supervisor": {"enabled": False}}]},
    )
    mixed = _write_json(
        tmp_path / "mixed.json",
        {
            "rows": [
                _row("gg07", {"enabled": False}),
                {
                    "arm": "plain",
                    "game": "badrow",
                    "trajectory_supervisor": {"enabled": False},
                },
                {
                    "arm": "E3_default_llmon",
                    "game": "nondict",
                    "llm_enabled": True,
                    "trajectory_supervisor": None,
                },
            ]
        },
    )
    failures = [
        {
            "arm": "force_exploration_diversity",
            "action_index": 100 + index,
            "level": 0,
            "diagnosis": "diversity did not help",
            "resolved_by_levelup": False,
            "actions_to_levelup": None,
        }
        for index in range(2)
    ]
    lower_path = _write_json(
        tmp_path / "lower.json",
        _live_artifact([_row("hh08", _enabled_receipt(failures), actions=500)]),
    )

    artifact = mod.build_artifact(
        repo_root=REPO,
        result_path=tmp_path / "lower_result.json",
        candidate_paths=[missing, invalid, not_live, mixed, lower_path],
        write=False,
        duration_s=None,
        tests_run=TESTS_RUN,
        run_date="20260823",
    )

    reasons = {row["reason"] for row in artifact["provenance_audit"]["rejected_artifacts"]}
    assert {"missing_artifact", "invalid_json_or_non_mapping", "not_live_e3_path"} <= reasons
    row_reasons = {row["reason"] for row in artifact["provenance_audit"]["rejected_rows"]}
    assert {"off_path_or_not_live_e3_path", "non_mapping_trajectory_supervisor"} <= row_reasons
    per_arm = {row["arm"]: row for row in artifact["per_arm_rows"]}
    assert per_arm["force_exploration_diversity"]["recommended_action"] == "lower_priority"
    lowered_after = {row["arm"]: row for row in artifact["arm_table_before_after"]["after"]}
    assert lowered_after["force_exploration_diversity"]["state"] == "active_lowered"

    malformed_redirect_rows = mod.replay_redirect_rows(
        [
            {"outcome_bearing": False},
            {"outcome_bearing": True, "receipt": None},
            {"outcome_bearing": True, "receipt": {"redirects": "bad"}},
            {"outcome_bearing": True, "receipt": {"redirects": [None]}},
            {"outcome_bearing": True, "receipt": {"redirects": [{"arm": "unknown"}]}},
        ]
    )
    assert malformed_redirect_rows == []

    rel = tmp_path / "relative_root"
    (rel / "results").mkdir(parents=True)
    fake_calls: list[tuple[Path, dict[str, Any], Path]] = []

    def fake_atomic(path: Path, payload: dict[str, Any], *, root: Path, sort_keys: bool) -> Path:
        fake_calls.append((path, payload, root))
        return root / path

    monkeypatch.setattr(mod, "atomic_write_json", fake_atomic)
    assert mod._write_artifact_json(Path("results/fake.json"), {"ok": True}, rel) == rel / "results/fake.json"
    assert fake_calls

    existing_default = tmp_path / "default.json"
    _write_json(existing_default, _live_artifact([_row("ii09", {"enabled": False})]))
    monkeypatch.setattr(mod, "DEFAULT_LIVE_ARTIFACT_PATHS", (existing_default, tmp_path / "gone.json"))
    assert mod._candidate_paths(None, REPO) == [existing_default]

    assert mod.sha256_file(tmp_path / "absent") == "missing"
    assert mod._rows_from_payload([{"x": 1}]) == [{"x": 1}]
    assert mod._contains_forbidden_evidence({"bad": object()}) is False
    assert mod._terminal_verdict(
        redirect_rows=[],
        provenance={"off_path_evidence_used": True},
        refinement_status="blocked_missing_outcome_bearing_receipts",
    )[2] == "disqualified"
    assert mod._int_or_none(None) is None
    assert mod._int_or_none("nope") is None
    assert mod._string(None, "fallback") == "fallback"
