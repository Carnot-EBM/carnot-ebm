"""Tests for Exp 4431 `.409` archive / `.410` activation.

Spec refs: REQ-REPORT-4431, SCENARIO-REPORT-4431,
SCENARIO-REPORT-4431-BLOCKED-PRECONDITION.
"""

from __future__ import annotations

import copy
import importlib
import json
from pathlib import Path

import pytest

from carnot.reporting import archive_409_activate_410_4431 as mod


GREEN = mod.CommandResult(command=["pytest"], exit_code=0, stdout="43 passed", stderr="")
RED = mod.CommandResult(command=["pytest"], exit_code=1, stdout="FAILED smart subset", stderr="")


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _research_complete_text(*, duplicates: int = 1) -> str:
    head = (
        "# Carnot Research - Completed Experiments\n"
        "milestones:\n"
        "- id: 2026.06.408\n"
        "  finding: prior milestone\n"
    )
    block = (
        "- id: 2026.06.409\n"
        "  title: old conductor row\n"
        "  completed: '2026-06-19'\n"
        "  finding: See conductor log for per-experiment results.\n"
        "  tasks:\n"
        "  - id: exp4430-capstone-409\n"
        "    result: OK (conductor)\n"
    )
    return head + block * duplicates


def _registry_text() -> str:
    games = [
        ("r11l", 1),
        ("ls20", 1),
        ("wa30", 1),
        ("s5i5", 1),
        ("lp85", 5),
        ("sc25", 1),
        ("cd82", 1),
        ("sp80", 1),
        ("su15", 1),
        ("tu93", 4),
        ("tn36", 7),
        ("cn04", 1),
        ("m0r0", 1),
        ("sk48", 1),
        ("ar25", 1),
        ("ka59", 1),
        ("ft09", 1),
        ("tr87", 6),
    ]
    lines = [
        "schema_version: 1",
        "updated: '2026-06-19'",
        "games:",
    ]
    for game, levels in games:
        lines.extend(
            [
                f"- game: {game}",
                "  reproducibility: reproduced",
                f"  levels_reproduced: {levels}",
            ]
        )
    lines.extend(
        [
            "reproducible_total_levels: 35",
            "reproducible_total_games: 18",
        ]
    )
    return "\n".join(lines) + "\n"


def _capstone(**overrides: object) -> dict:
    payload = {
        "honest_verdict": (
            "complete: v409_levels_36_new_levels_2_new_games_1_"
            "config_rule_flagged_registry_execution_grounded_glyph_solved_"
            "first_contact_gap_deepening_no_plus1_vocab_skipped_publication_ready"
        ),
        "reproducible_total_levels": 36,
        "new_levels": 2,
        "new_games": 1,
        "verifier_is_oracle": False,
        "config_rule_unseen_state": "direct_artifact_flagged_registry_audit_counted_execution_grounded",
        "glyph_rewrite_state": "grounded_and_offline_solved",
        "generic_first_contact_state": "verifier_gap_logged_no_new_game",
        "multi_level_deepening_state": "mechanic_repair_no_new_level",
        "config_rule_vocabulary_transfer_state": "excluded_flagged_adversarial",
        "config_rule_vocabulary_transfers": False,
        "flagged_artifacts_excluded": [
            {
                "artifact_key": "4421_config_rule",
                "experiment_id": 4421,
                "path": "results/experiment_4421_config_rule_solve_unseen.json",
                "reason": "flagged_adversarial",
            },
            {
                "artifact_key": "4425_vocabulary",
                "experiment_id": 4425,
                "path": "results/experiment_4425_config_rule_vocabulary_transfer.json",
                "reason": "flagged_adversarial",
            },
        ],
        "flagged_sources_counted_by_registry_audit": [
            {
                "experiment": "exp4421",
                "artifact_flagged_adversarial": True,
                "new_levels_counted": 1,
                "offline_reproduced": True,
                "reproduced_levels": 1,
                "honest_verdict": "success_s5i5_L1_offline_reproduced",
            }
        ],
        "glyph_rewrite": {
            "state": "grounded_and_offline_solved",
            "target_game": "tr87",
            "offline_reproduced": True,
            "reproduced_levels": 6,
            "verifier_is_oracle": True,
            "honest_verdict": "success_glyph_rewrite_perception_tr87_grounded_reproduced",
        },
        "generic_first_contact": {
            "state": "verifier_gap_logged_no_new_game",
            "target_game": "g50t",
            "offline_reproduced": False,
            "reproduced_levels": 0,
            "verifier_is_oracle": False,
            "honest_verdict": "partial: generic_first_contact_g50t_routed_missing_verifier_gap_logged",
            "missing_verifier_gaps": [
                {"gap_id": "GAP-4423-G50T-UNSELECTABLE-FIRST-CONTACT", "status": "open"}
            ],
        },
        "multi_level_deepening": {
            "state": "mechanic_repair_no_new_level",
            "game": "sc25",
            "offline_reproduced": False,
            "new_levels_reproduced": 0,
            "reproduced_levels": 1,
            "verifier_is_oracle": True,
            "honest_verdict": "complete: sc25_L2_hud_cleanup_fixed_reproduction_gap",
        },
        "config_rule_vocabulary_transfer": {
            "state": "excluded_flagged_adversarial",
            "config_rule_vocabulary_transfers": False,
        },
        "registry_audit": {
            "reproducible_total_levels": 36,
            "registry_claimed_reproducible_total_levels": 35,
            "registry_claimed_reproducible_total_games": 18,
            "all_counted_entries_reproduced": True,
        },
        "sota_ingestion": {
            "flagged_for_v410": (
                "Executable ARC-AGI-3 world-model agent with verifier-grounded planning "
                "(arXiv:2605.05138)"
            )
        },
        "publication_gate": {"paper_ready": True, "unmet_gates": []},
        "capstone_live_adversarial_recheck": {"status": "clean", "circular_moat_overclaim": False},
    }
    payload.update(overrides)
    return payload


def _config_rule() -> dict:
    return {
        "honest_verdict": "success_s5i5_L1_offline_reproduced",
        "offline_reproduced": True,
        "new_levels_reproduced": 1,
        "reproduced_levels": 1,
        "verifier_is_oracle": True,
        "flagged_adversarial": True,
        "corrigendum_pending": [{"kind": "DURATION_TOO_SHORT", "severity": "critical"}],
    }


def _glyph(**overrides: object) -> dict:
    payload = {
        "honest_verdict": "success_glyph_rewrite_perception_tr87_grounded_reproduced",
        "offline_reproduced": True,
        "reproduced_levels": 6,
        "target_game": "tr87",
        "verifier_is_oracle": True,
        "flagged_adversarial": False,
    }
    payload.update(overrides)
    return payload


def _first_contact() -> dict:
    return {
        "honest_verdict": "partial: generic_first_contact_g50t_routed_missing_verifier_gap_logged",
        "offline_reproduced": False,
        "reproduced_levels": 0,
        "target_game": "g50t",
        "missing_verifier_gaps": [{"gap_id": "GAP-4423-G50T-UNSELECTABLE-FIRST-CONTACT"}],
        "verifier_is_oracle": False,
    }


def _deepening() -> dict:
    return {
        "honest_verdict": "complete: sc25_L2_hud_cleanup_fixed_reproduction_gap",
        "offline_reproduced": False,
        "new_levels_reproduced": 0,
        "reproduced_levels": 1,
        "game": "sc25",
        "verifier_is_oracle": True,
    }


def _vocabulary(**overrides: object) -> dict:
    payload = {
        "honest_verdict": "complete: null_config_rule_vocabulary_transfer_seeded_arm_missing",
        "config_rule_vocabulary_transfers": False,
        "verifier_is_oracle": False,
        "flagged_adversarial": True,
        "corrigendum_pending": [{"kind": "DURATION_TOO_SHORT", "severity": "critical"}],
    }
    payload.update(overrides)
    return payload


def _registry_audit() -> dict:
    return {
        "honest_verdict": "complete: registry_repro_audit_total_35_asserted_36_audited",
        "reproducible_total_levels": 36,
        "registry_claimed_reproducible_total_levels": 35,
        "registry_claimed_reproducible_total_games": 18,
        "all_counted_entries_reproduced": True,
        "inference_substrate": "offline_arc_registry_repro_audit_cpu_no_llm",
    }


def _sota() -> dict:
    return {
        "honest_verdict": "complete: sota_ingestion_409_mapped",
        "flagged_for_v410": (
            "Executable ARC-AGI-3 world-model agent with verifier-grounded planning "
            "(arXiv:2605.05138)"
        ),
        "inference_substrate": "cpu_reliable_channel_sota_ingestion_no_training",
    }


def _make_root(tmp_path: Path, *, duplicates: int = 1) -> Path:
    (tmp_path / "ops").mkdir(parents=True)
    (tmp_path / "results").mkdir(parents=True)
    (tmp_path / "research-complete.yaml").write_text(
        _research_complete_text(duplicates=duplicates), encoding="utf-8"
    )
    (tmp_path / "ops/exclusion_manifest.yaml").write_text(
        "retired_extras:\n- id: circular_arc_solve_not_moat\n", encoding="utf-8"
    )
    (tmp_path / "ops/arc_solve_registry.yaml").write_text(_registry_text(), encoding="utf-8")
    (tmp_path / "research-roadmap.yaml").write_text("milestone: 2026.06.410\n", encoding="utf-8")
    (tmp_path / "research-roadmap-next.yaml").write_text(
        "milestone: 2026.06.410\ntasks: []\n", encoding="utf-8"
    )
    _write_json(tmp_path / "results/experiment_4430_capstone_409.json", _capstone())
    _write_json(tmp_path / "results/experiment_4421_config_rule_solve_unseen.json", _config_rule())
    _write_json(tmp_path / "results/experiment_4422_glyph_rewrite_perception.json", _glyph())
    _write_json(
        tmp_path / "results/experiment_4423_generic_first_contact_breadth.json",
        _first_contact(),
    )
    _write_json(tmp_path / "results/experiment_4424_deeper_solved_game.json", _deepening())
    _write_json(
        tmp_path / "results/experiment_4425_config_rule_vocabulary_transfer.json",
        _vocabulary(),
    )
    _write_json(tmp_path / "results/experiment_4426_arc_registry_repro_audit.json", _registry_audit())
    _write_json(tmp_path / "results/experiment_4429_sota_ingestion_409.json", _sota())
    return tmp_path


def test_req_report_4431_spec_anchor_declares_required_contract() -> None:
    """REQ-REPORT-4431: OpenSpec declares the archive contract."""

    spec = Path("openspec/capabilities/research-reporting/spec.md").read_text(encoding="utf-8")

    assert "REQ-REPORT-4431" in spec
    assert "SCENARIO-REPORT-4431" in spec
    assert "results/experiment_4431_archive_409_activate_410.json" in spec
    assert "research-roadmap-next.yaml" in spec
    assert mod.FIELD_PRINCIPLES["honest_verdict"] in spec
    assert mod.FIELD_PRINCIPLES["reproducible_total_levels"] in spec


def test_run_archives_v409_and_records_true_close_state(tmp_path: Path) -> None:
    """SCENARIO-REPORT-4431: complete path records .409 truth."""

    root = _make_root(tmp_path, duplicates=2)
    output = mod.run(root, pretest_result=GREEN, started_s=100.0, now_s=101.25)
    artifact = json.loads(output.read_text(encoding="utf-8"))

    assert artifact["honest_verdict"].startswith(
        ("complete:", "complete_", "success:", "success_", "passed:", "passed_", "shipped:", "shipped_")
    )
    assert artifact["archived_milestone"] == "2026.06.409"
    assert artifact["activated_milestone"] == "2026.06.410"
    assert artifact["active_milestone_confirmed"] == "2026.06.410"
    assert artifact["research_complete_yaml_parses"] is True
    assert artifact["exclusion_manifest_parses"] is True
    assert artifact["research_roadmap_next_yaml_parses"] is True
    assert artifact["pretest_suite_green"] is True
    assert artifact["verifier_is_oracle"] is True
    assert artifact["trm_training_ran"] is False
    assert artifact["leaderboard_submission"] is False
    assert artifact["inference_substrate"] == "aggregation_from_upstream_artifacts"
    assert artifact["field_principles"]["honest_verdict"] == mod.FIELD_PRINCIPLES["honest_verdict"]
    assert artifact["field_principles"]["reproducible_total_levels"] == (
        mod.FIELD_PRINCIPLES["reproducible_total_levels"]
    )

    assert artifact["reproducible_total_levels"] == 36
    assert artifact["reproducible_total_games"] == 18
    assert artifact["registry_declared_reproducible_total_levels"] == 35
    assert artifact["registry_entry_sum_reproducible_total_levels"] == 36
    assert artifact["registry_total_discrepancy"] is True

    close = artifact["v409_close_state"]
    assert close["phase_a_tasks"]["A1_config_rule_unseen"]["status"] == "quarantined_duration_too_short"
    assert close["phase_a_tasks"]["A1_config_rule_unseen"]["direct_artifact_imported"] is False
    assert close["phase_a_tasks"]["A2_glyph_rewrite"]["banked_reproducible_level"] is True
    assert close["phase_a_tasks"]["A2_glyph_rewrite"]["target_game"] == "tr87"
    assert close["phase_a_tasks"]["A3_generic_first_contact"]["status"] == "skipped_partial_verdict"
    assert close["phase_a_tasks"]["A3_generic_first_contact"]["honest_verdict"].startswith("partial:")
    assert close["phase_a_tasks"]["A4_deeper_solved_game"]["banked_reproducible_level"] is False
    assert close["phase_a_tasks"]["A4_deeper_solved_game"]["status"] == "mechanic_repair_no_new_level"
    assert close["config_rule_vocabulary_transfer"]["status"] == "excluded_flagged_adversarial"
    assert close["config_rule_vocabulary_transfer"]["config_rule_vocabulary_transfers"] is False
    assert close["flagged_artifacts_skipped"] == [4421, 4425]
    assert close["verifier_is_oracle_honored"] is True
    assert close["circular_execution_grounded_arc_solve_not_moat_headline"] is True

    history = (root / "research-complete.yaml").read_text(encoding="utf-8")
    assert mod.archive_record_count(history) == 1
    assert "activation_recorded: exp4431-archive-409-activate-410" in history
    assert "glyph rewrite banked execution-grounded tr87 progress" in history


def test_flagged_vocabulary_artifact_cannot_create_transfer_win(tmp_path: Path) -> None:
    """REQ-REPORT-4431: flagged vocabulary artifacts are skipped."""

    root = _make_root(tmp_path)
    _write_json(
        root / "results/experiment_4425_config_rule_vocabulary_transfer.json",
        _vocabulary(config_rule_vocabulary_transfers=True, flagged_adversarial=True),
    )

    artifact = json.loads(
        mod.run(root, pretest_result=GREEN, started_s=10.0, now_s=10.5).read_text(
            encoding="utf-8"
        )
    )

    transfer = artifact["v409_close_state"]["config_rule_vocabulary_transfer"]
    assert transfer["status"] == "excluded_flagged_adversarial"
    assert transfer["config_rule_vocabulary_transfers"] is False


def test_clean_vocabulary_artifact_can_record_transfer_win(tmp_path: Path) -> None:
    """REQ-REPORT-4431: clean non-oracle vocabulary evidence may transfer."""

    root = _make_root(tmp_path)
    _write_json(
        root / "results/experiment_4425_config_rule_vocabulary_transfer.json",
        _vocabulary(config_rule_vocabulary_transfers=True, flagged_adversarial=False),
    )

    artifact = json.loads(
        mod.run(root, pretest_result=GREEN, started_s=10.0, now_s=10.5).read_text(
            encoding="utf-8"
        )
    )

    transfer = artifact["v409_close_state"]["config_rule_vocabulary_transfer"]
    assert transfer["status"] == "transfers"
    assert transfer["config_rule_vocabulary_transfers"] is True


def test_registry_total_helper_covers_non_mapping_and_unsolved_rows() -> None:
    """REQ-REPORT-4431: registry totals handle malformed and unsolved rows."""

    empty = mod.registry_totals_from_text("- just\n- a\n- list\n")
    assert empty["authoritative_reproducible_total_levels"] == 0
    assert empty["reproduced_games"] == []

    totals = mod.registry_totals_from_text(
        "games:\n"
        "- game: solved\n"
        "  reproducibility: reproduced\n"
        "  levels_reproduced: 2\n"
        "- game: open\n"
        "  reproducibility: unsolved\n"
        "  levels_reproduced: 99\n"
        "reproducible_total_levels: 2\n"
        "reproducible_total_games: 1\n"
    )
    assert totals["entry_sum_reproducible_total_levels"] == 2
    assert totals["entry_sum_reproducible_total_games"] == 1


def test_registry_nested_flagged_count_source_is_detected() -> None:
    """REQ-REPORT-4431: capstone registry audit can carry flagged counted rows."""

    capstone = _capstone(flagged_sources_counted_by_registry_audit=[])
    capstone["registry_audit"] = {
        "flagged_sources_counted": [
            {"experiment": "exp4421", "new_levels_counted": 1},
        ]
    }

    close = mod.build_v409_close_state(
        {
            "4430": capstone,
            "4421": _config_rule(),
            "4422": _glyph(),
            "4423": _first_contact(),
            "4424": _deepening(),
            "4425": _vocabulary(),
            "4426": _registry_audit(),
            "4429": _sota(),
        },
        mod.registry_totals_from_text(_registry_text()),
    )

    assert close["phase_a_tasks"]["A1_config_rule_unseen"][
        "registry_audit_counted_execution_grounded"
    ] is True


@pytest.mark.parametrize(
    ("mutate", "reason"),
    [
        (
            lambda root: (root / "research-complete.yaml").unlink(),
            "blocked_research_complete_yaml_missing",
        ),
        (
            lambda root: (root / "research-complete.yaml").write_text("milestones: [", encoding="utf-8"),
            "blocked_research_complete_yaml_poison",
        ),
        (
            lambda root: (root / "ops/exclusion_manifest.yaml").unlink(),
            "blocked_exclusion_manifest_missing",
        ),
        (
            lambda root: (root / "ops/exclusion_manifest.yaml").write_text("retired_extras: [", encoding="utf-8"),
            "blocked_exclusion_manifest_yaml_poison",
        ),
        (
            lambda root: (root / "research-roadmap-next.yaml").unlink(),
            "blocked_research_roadmap_next_missing",
        ),
        (
            lambda root: (root / "research-roadmap-next.yaml").write_text("milestone: [", encoding="utf-8"),
            "blocked_research_roadmap_next_yaml_poison",
        ),
        (
            lambda root: (root / "research-roadmap.yaml").write_text("milestone: 2026.06.409\n", encoding="utf-8"),
            "blocked_v410_not_active",
        ),
        (
            lambda root: (root / "results/experiment_4430_capstone_409.json").unlink(),
            "blocked_v409_capstone_missing",
        ),
        (
            lambda root: (root / "ops/arc_solve_registry.yaml").unlink(),
            "blocked_arc_solve_registry_missing",
        ),
        (
            lambda root: (root / "ops/arc_solve_registry.yaml").write_text("games: [", encoding="utf-8"),
            "blocked_arc_solve_registry_yaml_poison",
        ),
    ],
)
def test_run_blocks_each_precondition_failure(tmp_path: Path, mutate: object, reason: str) -> None:
    """SCENARIO-REPORT-4431-BLOCKED-PRECONDITION: failures do not archive."""

    root = _make_root(tmp_path)
    mutate(root)
    history_path = root / "research-complete.yaml"
    post_mutation_history = (
        history_path.read_text(encoding="utf-8") if history_path.exists() else None
    )

    artifact = json.loads(
        mod.run(root, pretest_result=GREEN, started_s=1.0, now_s=1.5).read_text(encoding="utf-8")
    )

    assert artifact["honest_verdict"] == reason
    assert artifact["honest_verdict"].startswith("blocked_")
    assert artifact["v409_close_state"] == {}
    assert (history_path.read_text(encoding="utf-8") if history_path.exists() else None) == (
        post_mutation_history
    )


def test_run_uses_internal_smart_subset_when_no_result_is_injected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-REPORT-4431: production run path executes the smart-subset gate."""

    root = _make_root(tmp_path)
    monkeypatch.setattr(mod, "run_smart_subset", lambda _: GREEN)

    artifact = json.loads(
        mod.run(root, started_s=1.0, now_s=1.5).read_text(encoding="utf-8")
    )

    assert artifact["preconditions_checked"]["smart_subset_pretest"]["green"] is True


def test_run_blocks_when_pretest_red_without_editing_history(tmp_path: Path) -> None:
    """SCENARIO-REPORT-4431-BLOCKED-PRECONDITION: pretest red blocks archive."""

    root = _make_root(tmp_path)
    before = (root / "research-complete.yaml").read_text(encoding="utf-8")

    artifact = json.loads(
        mod.run(root, pretest_result=RED, started_s=1.0, now_s=1.5).read_text(encoding="utf-8")
    )

    assert artifact["honest_verdict"] == "blocked_smart_subset_pretest_not_green"
    assert artifact["preconditions_checked"]["smart_subset_pretest"]["green"] is False
    assert artifact["v409_close_state"] == {}
    assert (root / "research-complete.yaml").read_text(encoding="utf-8") == before


def test_run_blocks_if_research_complete_edit_would_not_parse(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SCENARIO-REPORT-4431-BLOCKED-PRECONDITION: invalid generated history blocks."""

    root = _make_root(tmp_path)

    def poisoned(_: str, __: dict) -> tuple[str, int, str]:
        return "milestones: [", 0, "updated"

    monkeypatch.setattr(mod, "dedupe_or_update_record", poisoned)
    artifact = json.loads(
        mod.run(root, pretest_result=GREEN, started_s=1.0, now_s=1.5).read_text(
            encoding="utf-8"
        )
    )

    assert artifact["honest_verdict"] == "blocked_research_complete_edit_invalid"
    assert artifact["v409_close_state"] == {}


def test_run_blocks_if_written_research_complete_turns_invalid(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SCENARIO-REPORT-4431-BLOCKED-PRECONDITION: post-write parse is checked."""

    root = _make_root(tmp_path)
    calls = {"n": 0}
    original = mod.yaml_parses

    def fails_after_write(text: str) -> bool:
        calls["n"] += 1
        if calls["n"] >= 6 and "glyph rewrite banked" in text:
            return False
        return original(text)

    monkeypatch.setattr(mod, "yaml_parses", fails_after_write)
    artifact = json.loads(
        mod.run(root, pretest_result=GREEN, started_s=1.0, now_s=1.5).read_text(
            encoding="utf-8"
        )
    )

    assert artifact["honest_verdict"] == "blocked_research_complete_yaml_poison_after_edit"
    assert artifact["v409_close_state"] == {}


def test_validate_artifact_rejects_false_success() -> None:
    """REQ-REPORT-4431: complete artifacts must preserve terminal and count contracts."""

    payload = mod.build_complete_artifact(
        v409_close_state=mod.build_v409_close_state(
            {
                "4430": _capstone(),
                "4421": _config_rule(),
                "4422": _glyph(),
                "4423": _first_contact(),
                "4424": _deepening(),
                "4425": _vocabulary(),
                "4426": _registry_audit(),
                "4429": _sota(),
            },
            mod.registry_totals_from_text(_registry_text()),
        ),
        registry_totals=mod.registry_totals_from_text(_registry_text()),
        preconditions_checked={"smart_subset_pretest": {"green": True}},
        duration_s=0.25,
        active_roadmap_path="research-roadmap.yaml",
        research_complete_record_action="updated",
        research_complete_duplicates_removed=0,
        cited_upstream_artifacts=[],
    )
    bad = copy.deepcopy(payload)
    bad["honest_verdict"] = "done_without_terminal_prefix"

    with pytest.raises(ValueError, match="honest_verdict"):
        mod.validate_artifact(bad)


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda payload: payload.pop("v409_close_state"), "missing required artifact field"),
        (lambda payload: payload["field_principles"].update(honest_verdict="wrong"), "principle"),
        (
            lambda payload: payload["field_principles"].update(reproducible_total_levels="wrong"),
            "principle",
        ),
        (lambda payload: payload.update(verifier_is_oracle=False), "verifier_is_oracle"),
        (lambda payload: payload.update(trm_training_ran=True), "TRM training"),
        (lambda payload: payload.update(leaderboard_submission=True), "leaderboard"),
        (lambda payload: payload.update(reproducible_total_levels=0), "reproducible_total_levels"),
        (lambda payload: payload.update(reproducible_total_games=0), "reproducible_total_games"),
        (lambda payload: payload.update(reproducibility_checksum="bad"), "reproducibility_checksum"),
    ],
)
def test_validate_artifact_rejects_each_contract_violation(
    mutate: object, message: str
) -> None:
    """REQ-REPORT-4431: every required complete-artifact invariant is enforced."""

    close = mod.build_v409_close_state(
        {
            "4430": _capstone(),
            "4421": _config_rule(),
            "4422": _glyph(),
            "4423": _first_contact(),
            "4424": _deepening(),
            "4425": _vocabulary(),
            "4426": _registry_audit(),
            "4429": _sota(),
        },
        mod.registry_totals_from_text(_registry_text()),
    )
    payload = mod.build_complete_artifact(
        v409_close_state=close,
        registry_totals=mod.registry_totals_from_text(_registry_text()),
        preconditions_checked={"smart_subset_pretest": {"green": True}},
        duration_s=0.25,
        active_roadmap_path="research-roadmap.yaml",
        research_complete_record_action="updated",
        research_complete_duplicates_removed=0,
        cited_upstream_artifacts=[],
    )
    bad = copy.deepcopy(payload)
    mutate(bad)

    with pytest.raises(ValueError, match=message):
        mod.validate_artifact(bad)


def test_record_helpers_cover_append_update_and_unchanged_paths() -> None:
    """REQ-REPORT-4431: history record helper handles standard archive paths."""

    close = mod.build_v409_close_state(
        {
            "4430": _capstone(),
            "4421": _config_rule(),
            "4422": _glyph(),
            "4423": _first_contact(),
            "4424": _deepening(),
            "4425": _vocabulary(),
            "4426": _registry_audit(),
            "4429": _sota(),
        },
        mod.registry_totals_from_text(_registry_text()),
    )

    appended, removed, action = mod.dedupe_or_update_record("milestones:\n", close)
    assert action == "appended"
    assert removed == 0
    assert mod.archive_record_count(appended) == 1

    unchanged, removed_again, action_again = mod.dedupe_or_update_record(appended, close)
    assert action_again == "unchanged"
    assert removed_again == 0
    assert unchanged == appended

    updated, removed_update, action_update = mod.dedupe_or_update_record(
        (
            "milestones:\n"
            "- id: 2026.06.409\n"
            "  title: old\n"
            "  activation_recorded: stale\n"
        ),
        close,
    )
    assert action_update == "updated"
    assert removed_update == 0
    assert "finding:" in updated
    assert "activation_recorded: exp4431-archive-409-activate-410" in updated


def test_main_delegates_to_run(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """REQ-REPORT-4431: module entry point delegates to run."""

    called = {"root": None}

    def fake_run(root: Path = mod.REPO_ROOT) -> Path:
        called["root"] = root
        return tmp_path / "sentinel.json"

    monkeypatch.setattr(mod, "run", fake_run)
    assert mod.main(tmp_path) == 0
    assert called["root"] == tmp_path

    reloaded = importlib.reload(mod)
    assert hasattr(reloaded, "main")
