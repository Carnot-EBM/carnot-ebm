"""Tests for Exp6558 ARC live redirect-ledger reachability.

Spec refs: REQ-ARC-WMTE-6680,
SCENARIO-ARC-WMTE-6680-LIVE-REACHABILITY,
SCENARIO-ARC-WMTE-6680-NEXT-OUTCOME-LINKAGE,
SCENARIO-ARC-WMTE-6680-NO-FIRING-CLOSURE,
SCENARIO-ARC-WMTE-6680-SELECTION-SUPPORT,
SCENARIO-ARC-WMTE-6680-FAIL-CLOSED-ATTACKS,
SCENARIO-ARC-WMTE-6680-SCHEMA-AND-CLI.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any

import pytest

from carnot import experiment_6558_arc_live_redirect_ledger_reachability as mod


REPO = Path(__file__).resolve().parents[2]
SPEC = REPO / mod.SPEC_RELATIVE_PATH
TESTS_RUN = [{"command": command, "exit_code": 0} for command in mod.DEFAULT_TEST_COMMANDS]


def _write_json(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path


def _redirect(
    arm: str,
    action_index: int,
    *,
    level: int = 0,
    resolved: bool = False,
    actions_to_levelup: int | None = None,
) -> dict[str, Any]:
    return {
        "arm": arm,
        "action_index": action_index,
        "level": level,
        "diagnosis": f"{arm} test diagnosis",
        "resolved_by_levelup": resolved,
        "actions_to_levelup": actions_to_levelup,
    }


def _applied_receipt(
    redirects: list[dict[str, Any]],
    *,
    actions_observed: int = 500,
    stagnations_unredirected: int = 0,
) -> dict[str, Any]:
    outcomes = {arm: {"fired": 0, "helped": 0} for arm in mod.ARM_ORDER}
    for row in redirects:
        outcomes[row["arm"]]["fired"] += 1
        if row["resolved_by_levelup"]:
            outcomes[row["arm"]]["helped"] += 1
    return {
        "enabled": True,
        "mode": "applied",
        "window": 400,
        "actions_observed": actions_observed,
        "arms_used": sorted({row["arm"] for row in redirects}),
        "redirects": redirects,
        "arm_outcomes": outcomes,
        "stagnations_unredirected": stagnations_unredirected,
        "observe_errors": 0,
    }


def _shadow_receipt(
    would_have_redirects: list[dict[str, Any]],
    *,
    actions_observed: int = 500,
    stagnations_unredirected: int = 0,
) -> dict[str, Any]:
    outcomes = {arm: {"fired": 0, "helped": 0} for arm in mod.ARM_ORDER}
    converted = []
    for row in would_have_redirects:
        outcomes[row["arm"]]["fired"] += 1
        if row["resolved_by_levelup"]:
            outcomes[row["arm"]]["helped"] += 1
        converted.append(
            {
                "arm": row["arm"],
                "action_index": row["action_index"],
                "level": row["level"],
                "diagnosis": row["diagnosis"],
                "levelup_followed_without_redirect": row["resolved_by_levelup"],
                "actions_to_levelup_without_redirect": row["actions_to_levelup"],
            }
        )
    return {
        "enabled": False,
        "mode": "shadow",
        "window": 400,
        "actions_observed": actions_observed,
        "arms_used": sorted({row["arm"] for row in would_have_redirects}),
        "would_have_redirects": converted,
        "would_have_arm_outcomes": outcomes,
        "stagnations_unredirected": stagnations_unredirected,
        "observe_errors": 0,
    }


def _row(
    game: str,
    receipt: dict[str, Any],
    *,
    levels: int = 0,
    actions: int = 500,
) -> dict[str, Any]:
    return {
        "game": game,
        "arm": "E3_default_llmon",
        "seed": 20260823,
        "budget": 2000,
        "llm_enabled": True,
        "gated_flags": {"tier_exhaustion_enabled": True},
        "levels": levels,
        "reached": levels,
        "actions": actions,
        "level_up_actions": [actions] if levels else [],
        "trajectory_supervisor": receipt,
    }


def _live_artifact(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "schema": "arc_scored_path_lever_harness.rows.v1",
        "canonical_entrypoint": "E3AgentPolicy",
        "rows": rows,
        "created_after_exp6524": True,
    }


@pytest.fixture()
def mixed_live_inputs(tmp_path: Path) -> list[Path]:
    """REQ-ARC-WMTE-6680: applied firings plus shadow rows are reduced distinctly."""

    path = _write_json(
        tmp_path / "post_6524_rows.json",
        _live_artifact(
            [
                _row(
                    "aa01",
                    _applied_receipt(
                        [
                            _redirect(
                                "drop_goal_bias",
                                100,
                                resolved=True,
                                actions_to_levelup=11,
                            ),
                            _redirect("allow_reinduction", 220),
                            _redirect("force_exploration_diversity", 420),
                        ],
                        actions_observed=700,
                    ),
                    levels=1,
                    actions=650,
                ),
                _row(
                    "bb02",
                    _applied_receipt(
                        [
                            _redirect("allow_reinduction", 401),
                            _redirect("force_exploration_diversity", 801),
                        ],
                        actions_observed=1000,
                    ),
                ),
                _row(
                    "cc03",
                    _applied_receipt(
                        [
                            _redirect("allow_reinduction", 402),
                            _redirect("force_exploration_diversity", 802),
                        ],
                        actions_observed=1100,
                    ),
                ),
                _row(
                    "dd04",
                    _shadow_receipt(
                        [
                            _redirect(
                                "allow_reinduction",
                                300,
                                resolved=True,
                                actions_to_levelup=17,
                            )
                        ],
                        actions_observed=500,
                    ),
                    levels=1,
                ),
            ]
        ),
    )
    return [path]


def test_req_arc_wmte_6680_spec_declares_reachability_contract() -> None:
    """REQ-ARC-WMTE-6680: OpenSpec owns the V567 reachability contract."""

    text = SPEC.read_text(encoding="utf-8")
    section = text[text.index("REQ-ARC-WMTE-6680") :]
    normalized = " ".join(section.split())

    for marker in (
        "SCENARIO-ARC-WMTE-6680-LIVE-REACHABILITY",
        "SCENARIO-ARC-WMTE-6680-NEXT-OUTCOME-LINKAGE",
        "SCENARIO-ARC-WMTE-6680-NO-FIRING-CLOSURE",
        "SCENARIO-ARC-WMTE-6680-SELECTION-SUPPORT",
        "SCENARIO-ARC-WMTE-6680-FAIL-CLOSED-ATTACKS",
        "SCENARIO-ARC-WMTE-6680-SCHEMA-AND-CLI",
        mod.MODULE_RELATIVE_PATH.as_posix(),
        mod.RESULT_RELATIVE_PATH.as_posix(),
        mod.INFERENCE_SUBSTRATE,
        "at least three prospective applied firings",
    ):
        assert marker in section
    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert f"`{field}`" in section
    for principle in mod.FIELD_PRINCIPLES.values():
        assert " ".join(principle.split()) in normalized


def test_scenarios_6680_reachability_next_outcome_and_no_firing_rows(
    mixed_live_inputs: list[Path],
    tmp_path: Path,
) -> None:
    """SCENARIO-ARC-WMTE-6680-LIVE-REACHABILITY/NEXT-OUTCOME-LINKAGE/NO-FIRING-CLOSURE."""

    artifact = mod.build_artifact(
        repo_root=REPO,
        result_path=tmp_path / "result.json",
        candidate_paths=mixed_live_inputs,
        write=False,
        duration_s=1.0,
        tests_run=TESTS_RUN,
        run_date="20260823",
    )

    assert mod.validate_artifact(artifact) == []
    assert artifact["status"] == "complete_live_redirect_ledger_reachable_no_policy_change"
    assert artifact["honest_verdict"].startswith("complete:")
    assert artifact["verdict_class"] is None
    assert artifact["arc_live_redirect_ledger_ready_score"] == 1.0
    assert artifact["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert artifact["verifier_is_oracle"] is False
    assert artifact["live_entrypoint_reachability_receipt"]["reachable"] is True
    assert artifact["live_entrypoint_reachability_receipt"]["make_carnot_agent_present"] is True
    assert artifact["supervisor_receipt_schema_and_code_hashes"]["receipt_schema_keys"] == [
        "enabled",
        "mode",
        "redirects",
        "arm_outcomes",
        "would_have_redirects",
        "would_have_arm_outcomes",
        "stagnations_unredirected",
        "observe_errors",
    ]

    rows = artifact["redirect_to_next_outcome_rows"]
    assert len(rows) == 7
    assert {row["mode"] for row in rows} == {"applied"}
    helped = [row for row in rows if row["resolved_by_levelup"]]
    assert len(helped) == 1
    assert helped[0]["next_observed_exact_live_outcome"] == {
        "kind": "levelup",
        "action_index": 111,
        "source": "redirect.actions_to_levelup",
    }
    unresolved = [row for row in rows if not row["resolved_by_levelup"]]
    assert unresolved
    assert all(row["next_observed_exact_live_outcome"]["kind"] == "run_terminal" for row in unresolved)
    assert all(row["next_observed_exact_live_outcome"]["source"] == "receipt.actions_observed" for row in unresolved)

    no_firing = artifact["no_firing_run_rows"]
    assert len(no_firing) == 1
    assert no_firing[0]["reason"] == "shadow_receipt_no_applied_firing"
    assert no_firing[0]["shadow_would_have_redirect_count"] == 1
    assert no_firing[0]["used_as_selection_support"] is False


def test_scenario_6680_selection_support_floor_preserves_policy(
    mixed_live_inputs: list[Path],
    tmp_path: Path,
) -> None:
    """SCENARIO-ARC-WMTE-6680-SELECTION-SUPPORT: support is visible but no order churn."""

    artifact = mod.build_artifact(
        repo_root=REPO,
        result_path=tmp_path / "result.json",
        candidate_paths=mixed_live_inputs,
        write=False,
        duration_s=1.0,
        tests_run=TESTS_RUN,
        run_date="20260823",
    )

    support = {row["arm"]: row for row in artifact["curated_arm_support_rows"]}
    assert support["drop_goal_bias"]["prospective_firings"] == 1
    assert support["drop_goal_bias"]["support_floor_met"] is False
    assert support["drop_goal_bias"]["support_disposition"] == "unsupported_fewer_than_three_firings"
    assert support["allow_reinduction"]["prospective_firings"] == 3
    assert support["allow_reinduction"]["helped_outcomes"] == 0
    assert support["allow_reinduction"]["support_disposition"] == "supported_no_help_lower_candidate"
    assert support["force_exploration_diversity"]["prospective_firings"] == 3
    assert support["force_exploration_diversity"]["support_floor_met"] is True

    disposition = artifact["selection_policy_disposition"]
    assert disposition["disposition"] == "unchanged"
    assert disposition["minimum_prospective_firings"] == 3
    assert disposition["current_order"] == list(mod.ARM_ORDER)
    assert disposition["replayed_supported_order"] == list(mod.ARM_ORDER)
    assert disposition["policy_changed"] is False
    assert "solve" not in disposition["reason"]
    assert artifact["aggregate_row_recomputation"]["fired_total"] == 7
    assert artifact["aggregate_row_recomputation"]["helped_total"] == 1
    assert artifact["aggregate_row_recomputation"]["unresolved_total"] == 6
    assert artifact["aggregate_row_recomputation"]["unredirected_stagnations_total"] == 0


def test_scenario_6680_no_firing_only_closes_null(tmp_path: Path) -> None:
    """SCENARIO-ARC-WMTE-6680-NO-FIRING-CLOSURE: empty applied ledger is terminal."""

    path = _write_json(
        tmp_path / "shadow_only.json",
        _live_artifact(
            [
                _row(
                    "ee05",
                    _shadow_receipt(
                        [
                            _redirect("drop_goal_bias", 100),
                            _redirect("allow_reinduction", 200),
                            _redirect("force_exploration_diversity", 300),
                        ],
                        stagnations_unredirected=1,
                    ),
                ),
                _row("ff06", _applied_receipt([], stagnations_unredirected=2)),
            ]
        ),
    )
    artifact = mod.build_artifact(
        repo_root=REPO,
        result_path=tmp_path / "result.json",
        candidate_paths=[path],
        write=False,
        duration_s=1.0,
        tests_run=TESTS_RUN,
        run_date="20260823",
    )

    assert artifact["status"] == "complete_live_redirect_ledger_reachable_no_firings"
    assert artifact["verdict_class"] is None
    assert artifact["redirect_to_next_outcome_rows"] == []
    assert len(artifact["no_firing_run_rows"]) == 2
    assert artifact["selection_policy_disposition"]["disposition"] == "unchanged"
    assert "future_arm_specification_rows" not in artifact["selection_policy_disposition"]
    assert artifact["arc_live_redirect_ledger_ready_score"] == 1.0


def test_scenario_6680_fail_closed_attacks_and_helpers(tmp_path: Path) -> None:
    """SCENARIO-ARC-WMTE-6680-FAIL-CLOSED-ATTACKS: bad evidence never refines policy."""

    live = _live_artifact(
        [
            _row(
                "gg07",
                _applied_receipt(
                    [
                        _redirect("drop_goal_bias", 100),
                        _redirect("drop_goal_bias", 100),
                    ]
                ),
            )
        ]
    )
    duplicate_path = _write_json(tmp_path / "duplicate_redirect.json", live)
    off_path = _write_json(
        tmp_path / "off_path.json",
        {
            "canonical_entrypoint": "E3AgentPolicy",
            "rows": [_row("hh08", _applied_receipt([]))],
            "source_reading": True,
            "game_adapter": "per_game_adapter",
        },
    )
    invalid = tmp_path / "invalid.json"
    invalid.write_text("{not json", encoding="utf-8")
    no_rows = _write_json(tmp_path / "no_rows.json", {"rows": []})
    missing = tmp_path / "missing.json"
    not_live = _write_json(tmp_path / "not_live.json", {"rows": [{"trajectory_supervisor": {}}]})
    mixed_bad_rows = _write_json(
        tmp_path / "mixed_bad_rows.json",
        _live_artifact(
            [
                {"game": "plain", "trajectory_supervisor": {}},
                {
                    "game": "none",
                    "arm": "E3_default_llmon",
                    "llm_enabled": True,
                    "trajectory_supervisor": None,
                },
                _row("ii09", _applied_receipt([])),
            ]
        ),
    )
    mixed_not_live_row = _write_json(
        tmp_path / "mixed_not_live_row.json",
        {
            "rows": [
                {"game": "plain", "trajectory_supervisor": {}},
                _row("kk11", _applied_receipt([])),
            ]
        },
    )

    artifact = mod.build_artifact(
        repo_root=REPO,
        result_path=tmp_path / "result.json",
        candidate_paths=[
            duplicate_path,
            duplicate_path,
            off_path,
            invalid,
            no_rows,
            missing,
            not_live,
            mixed_bad_rows,
            mixed_not_live_row,
        ],
        write=False,
        duration_s=None,
        tests_run=TESTS_RUN,
        run_date="20260823",
    )

    assert artifact["status"] == "disqualified_off_path_or_leaky_evidence"
    assert artifact["verdict_class"] == "disqualified"
    assert artifact["arc_live_redirect_ledger_ready_score"] == 0.0
    attacks = {row["attack_id"]: row for row in artifact["attack_matrix"]["rows"]}
    assert attacks["duplicate_redirects"]["observed"] == 1
    assert attacks["source_or_adapter_access"]["observed"] == 1
    assert attacks["missing_outcomes"]["observed"] >= 1
    assert all(row["fail_closed"] is True for row in attacks.values())
    assert artifact["no_solve_and_no_source_receipt"]["no_game_or_level_solve_claim"] is True
    assert artifact["no_solve_and_no_source_receipt"]["source_or_adapter_access_detected"] is True
    assert artifact["selection_policy_disposition"]["policy_changed"] is False

    reasons = {row["reason"] for row in artifact["gate_check_summary"].values() if isinstance(row, dict)}
    assert isinstance(reasons, set)
    rejected_reasons = {
        row["reason"]
        for row in artifact["no_solve_and_no_source_receipt"].get("rejected_rows", [])
    }
    assert isinstance(rejected_reasons, set)

    malformed_rows, malformed_audit = mod.redirect_to_next_outcome_rows(
        [
            {"mode": "applied", "enabled": True, "receipt": {"redirects": "bad"}},
            {"mode": "applied", "enabled": True, "receipt": {"redirects": [None]}},
            {
                "mode": "applied",
                "enabled": True,
                "run_receipt_id": "badarm",
                "receipt": {"redirects": [{"arm": "unknown"}]},
            },
            {
                "mode": "applied",
                "enabled": True,
                "run_receipt_id": "missing_action",
                "receipt": {"redirects": [{"arm": "drop_goal_bias"}]},
            },
            {
                "mode": "applied",
                "enabled": True,
                "run_receipt_id": "future",
                "receipt": {
                    "redirects": [
                        {
                            "arm": "drop_goal_bias",
                            "action_index": 10,
                            "level": 0,
                            "resolved_by_levelup": True,
                            "actions_to_levelup": -1,
                        }
                    ]
                },
            },
        ]
    )
    assert malformed_rows == []
    assert malformed_audit["malformed_redirect_count"] == 4
    assert malformed_audit["future_outcome_leakage_count"] == 1

    assert mod.selection_policy_disposition(
        mod.curated_arm_support_rows([]),
        [
            {
                "mode": "applied",
                "enabled": True,
                "arms_used": list(mod.ARM_ORDER),
                "stagnations_unredirected": 1,
                "game": "jj10",
                "run_receipt_id": "future",
            }
        ],
    )["disposition"] == "future-arm-specification"
    assert mod.curated_arm_support_rows(
        [
            {"arm": "drop_goal_bias", "resolved_by_levelup": True, "actions_to_levelup": 5},
            {"arm": "drop_goal_bias", "resolved_by_levelup": True, "actions_to_levelup": 4},
            {"arm": "drop_goal_bias", "resolved_by_levelup": True, "actions_to_levelup": 3},
            {"arm": "allow_reinduction", "resolved_by_levelup": True, "actions_to_levelup": 7},
            {"arm": "allow_reinduction", "resolved_by_levelup": False},
            {"arm": "allow_reinduction", "resolved_by_levelup": False},
        ]
    )[0]["recommended_action"] == "raise_priority"
    changed = mod.selection_policy_disposition(
        mod.curated_arm_support_rows(
            [
                {"arm": "allow_reinduction", "resolved_by_levelup": True, "actions_to_levelup": 3},
                {"arm": "allow_reinduction", "resolved_by_levelup": True, "actions_to_levelup": 4},
                {"arm": "allow_reinduction", "resolved_by_levelup": True, "actions_to_levelup": 5},
            ]
        ),
        [],
    )
    assert changed["disposition"] == "changed"
    assert mod._terminal_verdict(
        ready_score=1.0,
        redirect_rows=[{"arm": "allow_reinduction"}],
        selection=changed,
        reachability={"reachable": True},
        attack_matrix={"disqualifying_attack_observed": False},
    )[2] == "positive"
    assert mod._terminal_verdict(
        ready_score=0.0,
        redirect_rows=[],
        selection={"policy_changed": False},
        reachability={"reachable": False},
        attack_matrix={"disqualifying_attack_observed": False},
    )[2] == "blocked"
    assert mod._terminal_verdict(
        ready_score=0.0,
        redirect_rows=[],
        selection={"policy_changed": False},
        reachability={"reachable": True},
        attack_matrix={"disqualifying_attack_observed": False},
    )[2] == "partial"
    assert mod._ready_score(
        {"reachable": False},
        [],
        [],
        [],
        {"disqualifying_attack_observed": False},
    ) == 0.0
    assert mod._expected_verdict_class("positive_x") == "positive"
    assert mod._expected_verdict_class("partial_x") == "partial"
    assert mod._expected_verdict_class("blocked_x") == "blocked"
    assert mod._expected_verdict_class("disqualified_x") == "disqualified"
    assert mod._rows_from_payload([{"x": 1}]) == [{"x": 1}]
    assert mod._rows_from_payload({"rows": [{"y": 2}]}) == [{"y": 2}]
    assert mod._rows_from_payload({"no_rows": []}) == []
    assert mod._read_json(invalid) is None
    assert mod._candidate_paths(None, REPO)
    assert mod._contains_forbidden_evidence({"bad": object()}) is False
    assert mod._prior_failure_receipt(tmp_path)["exists"] is False
    assert mod._int_or_none("bad") is None
    assert mod._int_or_none(3) == 3
    assert mod._as_text(None, "fallback") == "fallback"
    assert mod.sha256_file(tmp_path / "absent") == "missing"
    assert mod._git_output(tmp_path / "absent", ["status"]).startswith("unavailable:")

    relative_root = tmp_path / "relative_root"
    assert mod._write_artifact_json(Path("results/fake.json"), {"ok": True}, relative_root).name == "fake.json"


def test_scenario_6680_schema_cli_validation_and_mutations(
    mixed_live_inputs: list[Path],
    tmp_path: Path,
) -> None:
    """SCENARIO-ARC-WMTE-6680-SCHEMA-AND-CLI: writer, validator, checksum, and gates."""

    result_path = tmp_path / "experiment_6558.json"
    assert (
        mod.main(
            [
                "--date",
                "20260823",
                "--result-path",
                str(result_path),
                "--input",
                str(mixed_live_inputs[0]),
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
    assert artifact["reproducibility_checksum"] == mod.reproducibility_checksum(artifact)
    assert artifact["prior_failure_receipt"]["exp6524_status"].startswith("blocked_")
    assert artifact["preconditions_checked"]["no_solve_no_game_source_task"] is True

    mutations = [
        ("delete", "status", "missing required fields"),
        ("set", ("status", "running"), "status lacks terminal prefix"),
        ("set", ("honest_verdict", "ready"), "honest_verdict lacks terminal prefix"),
        ("set", ("verdict_class", "partial"), "verdict_class invalid"),
        ("set", ("inference_substrate", "live_llm"), "inference_substrate mismatch"),
        ("set", ("verifier_is_oracle", True), "verifier_is_oracle must be false"),
        ("set", ("arc_live_redirect_ledger_ready_score", 0.5), "ready score invalid"),
        ("set", ("field_principles", {}), "field principles mismatch"),
        ("set", ("field_provenance", {}), "field provenance mismatch"),
        ("set", ("protected_files_unchanged", {}), "protected files changed"),
        ("set", ("attack_matrix", {}), "attack matrix invalid"),
        ("set", ("aggregate_row_recomputation", {}), "aggregate fired_total mismatch"),
        ("set", ("aggregate_row_recomputation", []), "aggregate rows malformed"),
        ("set", ("reproducibility_checksum", "sha256:bad"), "reproducibility checksum mismatch"),
    ]
    for mode, spec, expected in mutations:
        changed = deepcopy(artifact)
        if mode == "delete":
            del changed[spec]
        else:
            key, value = spec
            changed[key] = value
        if expected != "reproducibility checksum mismatch":
            changed["reproducibility_checksum"] = mod.reproducibility_checksum(changed)
        assert any(expected in issue for issue in mod.validate_artifact(changed)), expected

    bad_path = tmp_path / "bad.json"
    bad = deepcopy(artifact)
    bad["status"] = "running"
    bad["reproducibility_checksum"] = mod.reproducibility_checksum(bad)
    bad_path.write_text(json.dumps(bad), encoding="utf-8")
    assert mod.main(["--validate", "--result-path", str(bad_path)]) == 1
