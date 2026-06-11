"""Tests for Exp 4054 .374 archive and .375 activation.

Spec refs: REQ-REPORT-4054, SCENARIO-REPORT-4054,
SCENARIO-REPORT-4054-BLOCKED-YAML.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from carnot.reporting import archive_v374_activate_v375_4054 as mod


SPEC_PATH = Path("openspec/capabilities/research-reporting/spec.md")


# --------------------------------------------------------------------------- #
# Fixtures: a tmp repo with the .374 artifacts + an existing .374 record.
# --------------------------------------------------------------------------- #
def _v374_artifacts() -> dict[str, dict[str, object]]:
    """Synthetic .374 task artifacts mirroring the real verdicts/fields."""

    return {
        "4042": {"honest_verdict": "success: archived_v373_v374_active_pretest_green"},
        "4043": {"honest_verdict": "complete: sota_ingestion_offarc_power_and_closed_loop_mapped"},
        "4044": {
            "honest_verdict": "success: offarc_power_runner_built_smoked_launched_humaneval_mbpp",
            "flagged_adversarial": True,
        },
        "4045": {
            "honest_verdict": "complete: offarc_power_run_incomplete_partial_22_tasks",
            "best_arm": "armC_symbolic",
            "best_arm_delta_pp": 0.0,
            "best_arm_ci95": [0.0, 0.0],
            "best_arm_ci_excludes_zero": False,
            "demofit_delta_pp": 0.0,
            "demofit_bootstrap_ci95": [0.0, 0.0],
            "n_tasks": 22,
            "powered_task_floor": 160,
            "oracle_passrate": 1.0,
            "oracle_headroom": False,
        },
        "4046": {
            "honest_verdict": "complete: closed_loop_no_solve_vc33_wm_sim2real_ceiling_divergence_0.207",
            "game": "vc33",
            "closed_loop_broke_wall": False,
            "real_env_confirmed": False,
            "per_step_wm_real_divergence_rate": 0.207031,
            "divergence_gate_fired_count": 1,
            "goal_predicate_heldout_precision": 1.0,
            "levels_completed_after": 0,
            "new_levels_solved_this_task": 0,
            "bottleneck": "wm_real_divergence_gate_fired",
        },
        "4047": {
            "honest_verdict": "success: decentralization_moe_base_runner_launched_qwen35moe",
            "flagged_adversarial": True,
        },
        "4048": {
            "honest_verdict": "complete: decentralization_moe_base_partial_6_tasks_retire",
            "n_tasks_scored": 6,
            "coverage_delta_vs_12b": 0.2419,
            "bootstrap_ci95": [-0.0914, 0.5752],
            "diagnosis": "retired_non_measurement",
        },
        "4049": {
            "honest_verdict": "success: eighth_game_solved_sb26-7fbdac44_at_action_9",
            "total_games_solved": 8,
            "prior_total_games_solved": 7,
            "target_game": "sb26-7fbdac44",
            "first_solve_at_action": 9,
            "real_env_confirmed": True,
        },
        "4050": {
            "honest_verdict": "complete: arcmemo_v7_no_cross_game_transfer_v7_not_cheaper_than_within_game_v6",
            "cross_game_transfer_win": False,
            "actions_cold": 18,
            "actions_cross_game_v7": 9,
            "actions_within_game_v6": 7,
            "induction_calls_cross_game_v7": 0,
            "n_reused_abstractions": 1,
            "transfer_assessment": "helped_vs_cold_but_lost_to_within_game_v6",
        },
        "4051": {"honest_verdict": "complete: gap4_reeval_bitexact_g1_off_arc_power_pending_g2_sim2real_logged"},
        "4052": {
            "honest_verdict": "complete: hardware_continuity_kv260_latency_transcript_landed_4052",
            "per_board_reachability": {"kv260": True, "gatemate": True, "polarfire": True},
            "per_board_terminal_state": {
                "kv260": "reachable_overlay_loaded_latency_transcript_recorded",
                "gatemate": "reachable_detected_gatemate_idcode",
                "polarfire": "reachable_ssh_continuity_recorded",
            },
            "kv260_overlay_loaded": True,
            "kv260_latency_step_taken": True,
        },
        "4053": {
            "honest_verdict": (
                "complete: capstone_v374_not_decision_grade_G1_partial_or_incomplete_G2_closed_loop_"
                "ceiling_saturated_sim2real_divergence_G3_retired_non_measurement_games8_arcmemo_v7_"
                "no_win_flagged_skipped2"
            )
        },
    }


def _v374_record_block(idx: int) -> str:
    return (
        "- id: 2026.06.374\n"
        f"  title: 'THE DECISION-GRADE MILESTONE (copy {idx})'\n"
        "  doc: openspec/change-proposals/research-roadmap-v374.md\n"
        "  completed: '2026-06-11'\n"
        "  finding: See conductor log for per-experiment results.\n"
        "  tasks:\n"
        "  - id: exp4046-closed-loop-replan-over-vc33-wm\n"
        "    result: OK (conductor)\n"
    )


def _seed_repo(
    root: Path,
    *,
    corrupt_complete: bool = False,
    milestone: str = "2026.06.375",
    manifest: str = "retired: []\n",
    n_374_records: int = 1,
    checkpoint_n_tasks: int = 14,
) -> None:
    (root / "research-roadmap.yaml").write_text(
        f'milestone: "{milestone}"\ntasks:\n  - id: exp4054-archive-v374-activate-v375\n',
        encoding="utf-8",
    )
    complete_text = (
        "milestones:\n"
        "- id: 2026.06.373\n"
        "  title: prior archive\n"
        "  completed: '2026-06-11'\n"
        "  tasks:\n"
        "  - id: exp4042-archive\n"
        "    result: OK (conductor)\n"
    )
    for idx in range(n_374_records):
        complete_text += _v374_record_block(idx)
    if corrupt_complete:
        complete_text += "  - id: poison\n    result: complete: unquoted colon\n"
    (root / "research-complete.yaml").write_text(complete_text, encoding="utf-8")

    (root / "ops").mkdir(parents=True, exist_ok=True)
    (root / "_bmad").mkdir(parents=True, exist_ok=True)
    (root / "scripts" / "experiments").mkdir(parents=True, exist_ok=True)
    (root / "ops" / "exclusion_manifest.yaml").write_text(manifest, encoding="utf-8")
    (root / "ops" / "changelog.md").write_text("changelog before\n", encoding="utf-8")
    (root / "ops" / "status.md").write_text("status before\n", encoding="utf-8")
    (root / "_bmad" / "traceability.md").write_text("trace before\n", encoding="utf-8")
    (root / "scripts" / "research_conductor.py").write_text("# conductor before\n", encoding="utf-8")

    results = root / "results"
    results.mkdir(parents=True, exist_ok=True)
    payloads = _v374_artifacts()
    for task in mod.V374_TASKS:
        exp_id = str(task["exp_id"])
        if exp_id in payloads:
            (results / Path(str(task["deliverable"])).name).write_text(
                json.dumps(payloads[exp_id], sort_keys=True), encoding="utf-8"
            )
    # The MoE raw checkpoint whose task count proves the false-retirement fix.
    (results / mod.G3_MOE_CHECKPOINT_REL_PATH.name).write_text(
        json.dumps({"tasks": {f"task_{i:04d}": [{"code": "x"}] for i in range(checkpoint_n_tasks)}}),
        encoding="utf-8",
    )


def _green() -> mod.CommandResult:
    return mod.CommandResult(
        command=mod.smart_subset_command(["tests/python/test_docs.py"]),
        exit_code=0,
        stdout="81 passed\n",
        stderr="",
    )


def _run_success(root: Path, **overrides: object) -> Path:
    kwargs: dict[str, object] = {
        "research_complete_parse_result": mod.CommandResult(
            command=mod.research_complete_yaml_command(), exit_code=0, stdout="", stderr=""
        ),
        "arc_modules_import_result": mod.CommandResult(
            command=mod.arc_modules_import_command(), exit_code=0, stdout="{}", stderr=""
        ),
        "pretest_suite_results": [_green()],
        "started_s": 1.0,
        "now_s": 3.0,
    }
    kwargs.update(overrides)
    return mod.run(root, **kwargs)


# --------------------------------------------------------------------------- #
# Spec anchor
# --------------------------------------------------------------------------- #
def test_req_report_4054_spec_anchor_exists() -> None:
    """REQ-REPORT-4054: OpenSpec declares the .374 archive contract."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    assert "REQ-REPORT-4054" in spec
    assert "SCENARIO-REPORT-4054" in spec
    assert "SCENARIO-REPORT-4054-BLOCKED-YAML" in spec
    assert mod.OUTPUT_REL_PATH.as_posix() in spec
    assert "milestone_374_closestate" in spec
    assert "g3_false_retirement_corrected" in spec
    assert "smart-subset" in spec


# --------------------------------------------------------------------------- #
# Complete path
# --------------------------------------------------------------------------- #
def test_scenario_report_4054_records_closestate_unchanged(tmp_path: Path) -> None:
    """SCENARIO-REPORT-4054: one existing .374 record stays, truth recorded."""

    _seed_repo(tmp_path, n_374_records=1)
    before = {
        "manifest": (tmp_path / "ops" / "exclusion_manifest.yaml").read_text(encoding="utf-8"),
        "roadmap": (tmp_path / "research-roadmap.yaml").read_text(encoding="utf-8"),
        "complete": (tmp_path / "research-complete.yaml").read_text(encoding="utf-8"),
        "changelog": (tmp_path / "ops" / "changelog.md").read_text(encoding="utf-8"),
        "status": (tmp_path / "ops" / "status.md").read_text(encoding="utf-8"),
        "trace": (tmp_path / "_bmad" / "traceability.md").read_text(encoding="utf-8"),
        "conductor": (tmp_path / "scripts" / "research_conductor.py").read_text(encoding="utf-8"),
    }

    out_path = _run_success(tmp_path)
    artifact = json.loads(out_path.read_text(encoding="utf-8"))

    mod.validate_artifact(artifact)
    assert artifact["honest_verdict"].startswith("success:")
    assert artifact["archived_milestone"] == "2026.06.374"
    assert artifact["activated_milestone"] == "2026.06.375"
    assert artifact["active_milestone_confirmed"] == "2026.06.375"
    assert artifact["research_complete_yaml_parses"] is True
    assert artifact["exclusion_manifest_parses"] is True
    assert artifact["arc_modules_importable"] is True
    assert artifact["pretest_suite_green"] is True
    assert artifact["g3_false_retirement_corrected"] is True
    assert artifact["quarantined_tests"] == []
    assert artifact["inference_substrate"] == "aggregation_from_upstream_artifacts"
    assert artifact["research_complete_record_action"] == "unchanged"
    assert artifact["research_complete_duplicates_removed"] == 0
    assert artifact["n_tasks_archived"] == 12
    assert artifact["moe_checkpoint_n_tasks"] == 14

    # An existing single .374 record is left byte-for-byte unchanged.
    assert (tmp_path / "research-complete.yaml").read_text(encoding="utf-8") == before["complete"]
    loaded = yaml.safe_load(before["complete"])
    ids = [m["id"] for m in loaded["milestones"]]
    assert ids.count("2026.06.374") == 1

    # Close-state truth: per-task status + the three G-gate measurements.
    cs = artifact["milestone_374_closestate"]
    assert cs["per_task_status"]["exp4044-offarc-transfer-power-build"] == "FLAGGED"
    assert cs["per_task_status"]["exp4047-decentralization-moe-base-build"] == "FLAGGED"
    assert cs["per_task_status"]["exp4049-eighth-game-explore-first"] == "OK"
    assert cs["status_counts"]["OK"] == 10
    assert cs["status_counts"]["FLAGGED"] == 2
    # G1: off-ARC incomplete AND ceiling-saturated (no headroom).
    assert cs["g1_off_arc_transfer"]["n_tasks"] == 22
    assert cs["g1_off_arc_transfer"]["powered_task_floor"] == 160
    assert cs["g1_off_arc_transfer"]["full_power_reached"] is False
    assert cs["g1_off_arc_transfer"]["oracle_passrate"] == 1.0
    assert cs["g1_off_arc_transfer"]["ceiling_saturated"] is True
    assert cs["g1_off_arc_transfer"]["best_arm_delta_pp"] == 0.0
    assert cs["g1_off_arc_transfer"]["best_arm_ci95"] == [0.0, 0.0]
    assert cs["g1_off_arc_transfer"]["verifier_transferred_off_arc_significantly"] is False
    assert cs["g1_off_arc_transfer"]["is_measurement"] is False
    assert cs["g1_off_arc_transfer"]["outcome"] == "incomplete_and_ceiling_saturated"
    assert "corpus_saturation_no_oracle_headroom" in cs["g1_off_arc_transfer"]["root_causes"]
    # G2: decision-grade negative on vc33 (closed-loop sim2real ceiling).
    assert cs["g2_closed_loop_grounding"]["game"] == "vc33"
    assert cs["g2_closed_loop_grounding"]["closed_loop_broke_wall"] is False
    assert cs["g2_closed_loop_grounding"]["real_env_confirmed"] is False
    assert cs["g2_closed_loop_grounding"]["per_step_wm_real_divergence_rate"] == pytest.approx(0.207031)
    assert cs["g2_closed_loop_grounding"]["decision_grade_negative"] is True
    assert cs["g2_closed_loop_grounding"]["wm_planning_retired"] is True
    assert cs["g2_closed_loop_grounding"]["outcome"] == "decision_grade_negative_sim2real_ceiling"
    # G3: UNDERPOWERED-not-retired; the false retirement is corrected.
    g3 = cs["g3_decentralization_moe_base"]
    assert g3["capstone_diagnosis"] == "retired_non_measurement"
    assert g3["operator_corrected_diagnosis"] == "underpowered_not_retired"
    assert g3["false_retirement_corrected"] is True
    assert g3["retired"] is False
    assert g3["throughput_fix_worked"] is True
    assert g3["checkpoint_n_tasks"] == 14
    assert g3["premature_poll_n_tasks"] == 6
    assert g3["poll_artifact_n_tasks_scored"] == 6
    assert g3["moe_base_coverage"] == pytest.approx(0.3571)
    assert g3["baseline_12b_coverage"] == pytest.approx(0.2581)
    assert g3["bootstrap_ci95"] == [0.143, 0.643]
    assert g3["ci_spans_ceiling"] is True
    assert g3["target_task_floor"] == 30
    assert g3["outcome"] == "underpowered_not_retired_resume_toward_n30"
    # Accuracy: 8 games solved, +1 monotonic.
    assert cs["accuracy"]["total_games_solved"] == 8
    assert cs["accuracy"]["monotonic_plus_one"] is True
    assert cs["accuracy"]["target_game"] == "sb26-7fbdac44"
    assert cs["accuracy"]["first_solve_at_action"] == 9
    # Self-learning: ArcMemo v7 helped vs cold but lost to within-game v6 (no win).
    assert cs["self_learning"]["cross_game_transfer_win"] is False
    assert cs["self_learning"]["actions_cold"] == 18
    assert cs["self_learning"]["actions_cross_game_v7"] == 9
    assert cs["self_learning"]["actions_within_game_v6"] == 7
    assert cs["self_learning"]["action_savings_vs_cold"] == 9
    # Hardware: KV260 terminal.
    assert cs["hardware"]["per_board_reachability"]["kv260"] is True
    assert cs["hardware"]["kv260_overlay_loaded"] is True
    assert cs["hardware"]["kv260_latency_step_taken"] is True
    assert cs["hardware"]["kv260_terminal"] is True
    # Both BUILD halves flagged-skipped (never aggregated as a win).
    flagged_ids = {f["experiment_id"]: f for f in cs["flagged_skipped"]}
    assert flagged_ids["4044"]["flagged_adversarial"] is True and flagged_ids["4044"]["skipped"] is True
    assert flagged_ids["4047"]["flagged_adversarial"] is True and flagged_ids["4047"]["skipped"] is True
    assert cs["capstone_v374_verdict"].startswith("complete: capstone_v374")
    assert "CEILING-SATURATED" in cs["headline"]
    assert "UNDERPOWERED" in cs["headline"]
    assert "FALSE retirement" in cs["headline"]

    # Operator-curated / conductor-reconciled files untouched.
    assert (tmp_path / "ops" / "exclusion_manifest.yaml").read_text(encoding="utf-8") == before["manifest"]
    assert (tmp_path / "research-roadmap.yaml").read_text(encoding="utf-8") == before["roadmap"]
    assert (tmp_path / "ops" / "changelog.md").read_text(encoding="utf-8") == before["changelog"]
    assert (tmp_path / "ops" / "status.md").read_text(encoding="utf-8") == before["status"]
    assert (tmp_path / "_bmad" / "traceability.md").read_text(encoding="utf-8") == before["trace"]
    assert (tmp_path / "scripts" / "research_conductor.py").read_text(encoding="utf-8") == before["conductor"]


def test_req_report_4054_dedupes_duplicate_records(tmp_path: Path) -> None:
    """REQ-REPORT-4054: interrupted-run duplicate .374 records collapse to one."""

    _seed_repo(tmp_path, n_374_records=5)
    artifact = json.loads(_run_success(tmp_path).read_text(encoding="utf-8"))
    assert artifact["research_complete_record_action"] == "deduped"
    assert artifact["research_complete_duplicates_removed"] == 4
    loaded = yaml.safe_load((tmp_path / "research-complete.yaml").read_text(encoding="utf-8"))
    ids = [m["id"] for m in loaded["milestones"]]
    assert ids.count("2026.06.374") == 1


def test_req_report_4054_dedupe_is_idempotent(tmp_path: Path) -> None:
    """REQ-REPORT-4054: rerunning leaves exactly one .374 record (unchanged)."""

    _seed_repo(tmp_path, n_374_records=5)
    _run_success(tmp_path)
    first_complete = (tmp_path / "research-complete.yaml").read_text(encoding="utf-8")
    second = json.loads(_run_success(tmp_path).read_text(encoding="utf-8"))

    assert (tmp_path / "research-complete.yaml").read_text(encoding="utf-8") == first_complete
    assert second["research_complete_record_action"] == "unchanged"
    assert second["research_complete_duplicates_removed"] == 0


def test_req_report_4054_appends_when_record_absent(tmp_path: Path) -> None:
    """REQ-REPORT-4054: a missing .374 record is appended canonically."""

    _seed_repo(tmp_path, n_374_records=0)
    artifact = json.loads(_run_success(tmp_path).read_text(encoding="utf-8"))

    assert artifact["research_complete_record_action"] == "appended"
    loaded = yaml.safe_load((tmp_path / "research-complete.yaml").read_text(encoding="utf-8"))
    ids = [m["id"] for m in loaded["milestones"]]
    assert ids.count("2026.06.374") == 1
    record = next(m for m in loaded["milestones"] if m["id"] == "2026.06.374")
    assert record["activation_recorded"] == "exp4054-archive-v374-activate-v375"
    assert len(record["tasks"]) == len(mod.V374_TASKS)


# --------------------------------------------------------------------------- #
# Blocked paths
# --------------------------------------------------------------------------- #
def test_scenario_report_4054_blocked_yaml_writes_artifact_without_edits(tmp_path: Path) -> None:
    """SCENARIO-REPORT-4054-BLOCKED-YAML: corrupt YAML exits before edits."""

    _seed_repo(tmp_path, corrupt_complete=True)
    before_complete = (tmp_path / "research-complete.yaml").read_text(encoding="utf-8")
    before_manifest = (tmp_path / "ops" / "exclusion_manifest.yaml").read_text(encoding="utf-8")

    artifact = json.loads(
        _run_success(
            tmp_path,
            research_complete_parse_result=mod.CommandResult(
                command=mod.research_complete_yaml_command(),
                exit_code=1,
                stdout="",
                stderr="yaml parser failed",
            ),
        ).read_text(encoding="utf-8")
    )

    assert artifact["honest_verdict"].startswith("blocked_research_complete_yaml_poison")
    assert artifact["research_complete_yaml_parses"] is False
    assert artifact["g3_false_retirement_corrected"] is False
    assert artifact["milestone_374_closestate"]["status"] == "blocked"
    assert artifact["active_milestone_confirmed"] == "2026.06.375"
    assert artifact["preconditions_checked"]["research_complete_yaml_parsed_before"] is False
    assert (tmp_path / "research-complete.yaml").read_text(encoding="utf-8") == before_complete
    assert (tmp_path / "ops" / "exclusion_manifest.yaml").read_text(encoding="utf-8") == before_manifest


def test_req_report_4054_blocked_when_complete_missing(tmp_path: Path) -> None:
    """REQ-REPORT-4054: a missing research-complete.yaml fails closed."""

    _seed_repo(tmp_path)
    (tmp_path / "research-complete.yaml").unlink()
    artifact = json.loads(_run_success(tmp_path).read_text(encoding="utf-8"))
    assert artifact["honest_verdict"].startswith("blocked_research_complete_yaml_poison_missing")


def test_req_report_4054_blocked_paths_fail_closed(tmp_path: Path) -> None:
    """REQ-REPORT-4054: missing handoff facts block instead of fabricating success."""

    _seed_repo(tmp_path, milestone="2026.06.374")
    artifact = json.loads(_run_success(tmp_path).read_text(encoding="utf-8"))
    assert artifact["honest_verdict"].startswith("blocked_v375_not_active")

    _seed_repo(tmp_path, manifest="retired: [\n")
    artifact = json.loads(_run_success(tmp_path).read_text(encoding="utf-8"))
    assert artifact["honest_verdict"].startswith("blocked_exclusion_manifest_yaml_poison")

    _seed_repo(tmp_path)
    (tmp_path / "ops" / "exclusion_manifest.yaml").unlink()
    artifact = json.loads(_run_success(tmp_path).read_text(encoding="utf-8"))
    assert artifact["honest_verdict"].startswith("blocked_exclusion_manifest_missing")

    _seed_repo(tmp_path)
    artifact = json.loads(
        _run_success(
            tmp_path,
            arc_modules_import_result=mod.CommandResult(
                command=mod.arc_modules_import_command(), exit_code=1, stdout="{}", stderr=""
            ),
        ).read_text(encoding="utf-8")
    )
    assert artifact["honest_verdict"].startswith("blocked_arc_module_import")


def test_req_report_4054_blocked_when_pretest_unquarantinable(tmp_path: Path) -> None:
    """REQ-REPORT-4054: a red gate with no parseable failure id blocks."""

    _seed_repo(tmp_path)
    red = mod.CommandResult(
        command=mod.smart_subset_command(["x"]),
        exit_code=1,
        stdout="some failure without a node id\n",
        stderr="",
    )
    artifact = json.loads(_run_success(tmp_path, pretest_suite_results=[red]).read_text(encoding="utf-8"))
    assert artifact["honest_verdict"].startswith("blocked_pretest_suite_failed_unquarantined")
    assert artifact["pretest_suite_green"] is False
    # The close-state truth is still recorded on the blocked path.
    assert "per_task_status" in artifact["milestone_374_closestate"]
    assert artifact["moe_checkpoint_n_tasks"] == 14


def test_req_report_4054_quarantines_red_test_then_green(tmp_path: Path) -> None:
    """REQ-REPORT-4054: a red smart-subset file is git-mv'd to quarantine."""

    _seed_repo(tmp_path)
    red_file = tmp_path / "tests" / "python" / "test_old_poison.py"
    red_file.parent.mkdir(parents=True, exist_ok=True)
    red_file.write_text("def test_bad():\n    assert False\n", encoding="utf-8")
    failure = mod.CommandResult(
        command=mod.smart_subset_command(["x"]),
        exit_code=1,
        stdout="FAILED tests/python/test_old_poison.py::test_bad - AssertionError\n",
        stderr="",
    )

    artifact = json.loads(
        _run_success(tmp_path, pretest_suite_results=[failure, _green()]).read_text(encoding="utf-8")
    )
    assert artifact["pretest_suite_green"] is True
    assert artifact["quarantined_tests"] == [
        {
            "path": "tests/python/test_old_poison.py",
            "quarantined_path": "tests/quarantine/test_old_poison.py",
            "failing_test_ids": ["tests/python/test_old_poison.py::test_bad"],
        }
    ]
    assert not red_file.exists()
    assert (tmp_path / "tests" / "quarantine" / "test_old_poison.py").exists()


def test_req_report_4054_blocked_edit_invalid(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """REQ-REPORT-4054: an edit that breaks YAML blocks before writing."""

    _seed_repo(tmp_path)
    monkeypatch.setattr(mod, "dedupe_or_append_record", lambda *a: ("milestones: [\n", 0, "deduped"))
    artifact = json.loads(_run_success(tmp_path).read_text(encoding="utf-8"))
    assert artifact["honest_verdict"].startswith("blocked_research_complete_edit_invalid")
    # The original file was NOT overwritten with the broken edit.
    assert yaml.safe_load((tmp_path / "research-complete.yaml").read_text(encoding="utf-8"))


def test_req_report_4054_blocked_poison_after_edit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """REQ-REPORT-4054: a post-write parse regression is caught."""

    _seed_repo(tmp_path)
    # Calls in order: parses_before, edit-candidate, on-disk re-read, manifest.
    states = iter([True, True, False, True])

    monkeypatch.setattr(mod, "yaml_parses", lambda text: next(states))
    artifact = json.loads(_run_success(tmp_path).read_text(encoding="utf-8"))
    assert artifact["honest_verdict"].startswith("blocked_research_complete_yaml_poison_after_edit")


# --------------------------------------------------------------------------- #
# Close-state + helper unit tests
# --------------------------------------------------------------------------- #
def test_req_report_4054_classify_status_branches() -> None:
    """REQ-REPORT-4054: every status class is reachable."""

    assert mod.classify_status({"exists": False}) == "MISSING"
    assert mod.classify_status({"exists": True, "flagged_adversarial": True}) == "FLAGGED"
    assert mod.classify_status({"exists": True, "honest_verdict": "blocked_x"}) == "BLOCKED"
    assert mod.classify_status({"exists": True, "honest_verdict": "complete: ok"}) == "OK"
    assert mod.classify_status({"exists": True, "honest_verdict": "weird"}) == "FAIL"


def _record(payload: dict[str, object]) -> dict[str, object]:
    """Mirror read_artifact_record's output shape (fields sub-dict)."""

    return {
        "exists": True,
        "honest_verdict": str(payload.get("honest_verdict", "")),
        "flagged_adversarial": bool(payload.get("flagged_adversarial")),
        "fields": dict(payload),
    }


def test_req_report_4054_build_closestate_from_records() -> None:
    """REQ-REPORT-4054: the close-state builder is a pure aggregation."""

    records: dict[str, dict[str, object]] = {
        exp_id: _record(payload) for exp_id, payload in _v374_artifacts().items()
    }
    closestate = mod.build_closestate(records, 14)

    assert closestate["status_counts"]["OK"] == 10
    assert closestate["status_counts"]["FLAGGED"] == 2
    assert closestate["g1_off_arc_transfer"]["outcome"] == "incomplete_and_ceiling_saturated"
    assert closestate["g2_closed_loop_grounding"]["decision_grade_negative"] is True
    assert closestate["g3_decentralization_moe_base"]["false_retirement_corrected"] is True
    assert closestate["g3_decentralization_moe_base"]["retired"] is False
    assert closestate["accuracy"]["total_games_solved"] == 8
    assert closestate["self_learning"]["cross_game_transfer_win"] is False
    assert closestate["hardware"]["kv260_terminal"] is True
    assert closestate["capstone_v374_verdict"].startswith("complete: capstone_v374")
    assert "vc33" in closestate["headline"]


def test_req_report_4054_closestate_subbuilders_degrade_gracefully() -> None:
    """REQ-REPORT-4054: every sub-builder returns null facts on missing input."""

    empty = {"exists": False}

    g1 = mod._g1_off_arc_transfer(empty)
    assert g1["n_tasks"] is None
    assert g1["best_arm_ci95"] is None
    assert g1["full_power_reached"] is False
    assert g1["verifier_transferred_off_arc_significantly"] is False
    # No data means no full-power measurement -> incomplete, not a refuted transfer.
    assert g1["outcome"] == "incomplete_underpowered"

    g2 = mod._g2_closed_loop_grounding(empty)
    assert g2["game"] is None
    assert g2["decision_grade_negative"] is False
    assert g2["wm_planning_retired"] is True
    assert g2["outcome"] == "inconclusive"

    g3 = mod._g3_decentralization_moe(empty, None)
    assert g3["checkpoint_n_tasks"] is None
    assert g3["throughput_fix_worked"] is False
    assert g3["false_retirement_corrected"] is True
    assert g3["retired"] is False
    assert g3["poll_artifact_n_tasks_scored"] is None
    assert g3["ci_spans_ceiling"] is True

    accuracy = mod._accuracy(empty)
    assert accuracy["total_games_solved"] is None and accuracy["monotonic_plus_one"] is False

    self_learning = mod._self_learning(empty)
    assert self_learning["actions_cold"] is None
    assert self_learning["action_savings_vs_cold"] is None

    hardware = mod._hardware(empty)
    assert hardware["per_board_reachability"] == {} and hardware["included"] is False
    assert hardware["per_board_terminal_state"] == {}
    assert hardware["kv260_terminal"] is False

    flagged = mod._flagged_skipped(empty, "4044")
    assert flagged["flagged_adversarial"] is False and flagged["skipped"] is False
    assert flagged["experiment_id"] == "4044"

    # _fields tolerates a non-mapping fields value.
    assert mod._fields({"fields": [1, 2, 3]}) == {}
    assert mod._fields({}) == {}


def test_req_report_4054_g1_branches() -> None:
    """REQ-REPORT-4054: G1 outcome covers saturated, underpowered, transferred, none."""

    saturated = _record(
        {"n_tasks": 22, "powered_task_floor": 160, "oracle_passrate": 1.0, "oracle_headroom": False,
         "best_arm_ci_excludes_zero": False, "best_arm_ci95": [0.0, 0.0]}
    )
    g1 = mod._g1_off_arc_transfer(saturated)
    assert g1["outcome"] == "incomplete_and_ceiling_saturated"
    assert g1["ceiling_saturated"] is True

    # Incomplete but with headroom -> underpowered, not saturated.
    underpowered = _record(
        {"n_tasks": 40, "powered_task_floor": 160, "oracle_passrate": 0.7, "oracle_headroom": True}
    )
    assert mod._g1_off_arc_transfer(underpowered)["outcome"] == "incomplete_underpowered"

    # Full power AND CI excludes zero -> a real transfer measurement.
    transferred = _record(
        {"n_tasks": 200, "powered_task_floor": 160, "oracle_passrate": 0.8, "oracle_headroom": True,
         "best_arm_ci_excludes_zero": True, "best_arm_ci95": [2.0, 18.0]}
    )
    g1t = mod._g1_off_arc_transfer(transferred)
    assert g1t["outcome"] == "transferred_ci_excludes_zero"
    assert g1t["verifier_transferred_off_arc_significantly"] is True
    assert g1t["is_measurement"] is True

    # Full power, no headroom, CI not excluding zero -> no_transfer.
    flat = _record(
        {"n_tasks": 200, "powered_task_floor": 160, "oracle_passrate": 0.9, "oracle_headroom": True,
         "best_arm_ci_excludes_zero": False}
    )
    assert mod._g1_off_arc_transfer(flat)["outcome"] == "no_transfer"


def test_req_report_4054_g2_inconclusive_branch() -> None:
    """REQ-REPORT-4054: G2 is inconclusive when no divergence rate was measured."""

    # A wall break would not be a clean negative; missing divergence -> inconclusive.
    broke = _record({"game": "vc33", "closed_loop_broke_wall": True, "real_env_confirmed": True})
    g2 = mod._g2_closed_loop_grounding(broke)
    assert g2["decision_grade_negative"] is False
    assert g2["outcome"] == "inconclusive"


def test_req_report_4054_g3_throughput_and_ci() -> None:
    """REQ-REPORT-4054: G3 records throughput-worked + CI-spans-ceiling truthfully."""

    poll = _record(
        {"n_tasks_scored": 6, "coverage_delta_vs_12b": 0.2419, "bootstrap_ci95": [-0.0914, 0.5752],
         "diagnosis": "retired_non_measurement"}
    )
    g3 = mod._g3_decentralization_moe(poll, 14)
    assert g3["checkpoint_n_tasks"] == 14
    assert g3["throughput_fix_worked"] is True
    assert g3["poll_artifact_coverage_delta_vs_12b"] == pytest.approx(0.2419)
    assert g3["poll_artifact_bootstrap_ci95"] == [-0.0914, 0.5752]
    assert g3["moe_base_coverage"] == pytest.approx(0.3571)
    assert g3["ci_spans_ceiling"] is True

    # A zero-task checkpoint means the throughput fix did NOT produce data.
    g3_zero = mod._g3_decentralization_moe(poll, 0)
    assert g3_zero["throughput_fix_worked"] is False


def test_req_report_4054_accuracy_nonmonotonic_and_self_learning_bool() -> None:
    """REQ-REPORT-4054: accuracy guards reject non-+1 jumps and bool counts."""

    jump = _record({"total_games_solved": 10, "prior_total_games_solved": 7})
    assert mod._accuracy(jump)["monotonic_plus_one"] is False

    bool_counts = _record({"actions_cold": True, "actions_cross_game_v7": 1})
    assert mod._self_learning(bool_counts)["action_savings_vs_cold"] is None


def test_req_report_4054_read_checkpoint_task_count(tmp_path: Path) -> None:
    """REQ-REPORT-4054: the checkpoint counter reads mapping/list/absent/bad."""

    mapping_ckpt = tmp_path / "map.json"
    mapping_ckpt.write_text(json.dumps({"tasks": {"a": 1, "b": 2}}), encoding="utf-8")
    assert mod.read_checkpoint_task_count(mapping_ckpt) == 2

    list_ckpt = tmp_path / "list.json"
    list_ckpt.write_text(json.dumps({"tasks": [1, 2, 3]}), encoding="utf-8")
    assert mod.read_checkpoint_task_count(list_ckpt) == 3

    assert mod.read_checkpoint_task_count(tmp_path / "missing.json") is None

    no_tasks = tmp_path / "no_tasks.json"
    no_tasks.write_text(json.dumps({"other": 1}), encoding="utf-8")
    assert mod.read_checkpoint_task_count(no_tasks) is None

    listish = tmp_path / "listish.json"
    listish.write_text("[1, 2, 3]", encoding="utf-8")
    assert mod.read_checkpoint_task_count(listish) is None


def test_req_report_4054_read_artifact_record_fail_closed(tmp_path: Path) -> None:
    """REQ-REPORT-4054: unreadable / non-mapping artifacts read as absent."""

    assert mod.read_artifact_record(tmp_path / "missing.json")["exists"] is False
    listish = tmp_path / "listish.json"
    listish.write_text("[1, 2, 3]", encoding="utf-8")
    assert mod.read_artifact_record(listish)["exists"] is False
    good = tmp_path / "good.json"
    good.write_text(json.dumps({"honest_verdict": "complete: ok"}), encoding="utf-8")
    assert mod.read_artifact_record(good)["exists"] is True


def test_req_report_4054_read_v374_records_reads_all_tasks(tmp_path: Path) -> None:
    """REQ-REPORT-4054: every .374 deliverable is read by exp id."""

    _seed_repo(tmp_path)
    records = mod.read_v374_records(tmp_path)
    assert set(records) == {str(t["exp_id"]) for t in mod.V374_TASKS}
    assert records["4044"]["flagged_adversarial"] is True
    assert records["4045"]["fields"]["n_tasks"] == 22


def test_req_report_4054_dedupe_helper_branches() -> None:
    """REQ-REPORT-4054: dedupe/append helper covers all three actions."""

    base = "milestones:\n- id: 2026.06.373\n  title: a\n"
    two = base + _v374_record_block(0) + _v374_record_block(1)
    deduped, removed, action = mod.dedupe_or_append_record(two, "2026.06.374")
    assert action == "deduped" and removed == 1
    assert deduped.count("- id: 2026.06.374") == 1

    one = base + _v374_record_block(0)
    unchanged, removed, action = mod.dedupe_or_append_record(one, "2026.06.374")
    assert action == "unchanged" and removed == 0 and unchanged == one

    appended, removed, action = mod.dedupe_or_append_record(base, "2026.06.374")
    assert action == "appended" and removed == 0
    assert appended.count("- id: 2026.06.374") == 1
    assert yaml.safe_load(appended)


def test_req_report_4054_smart_subset_targets_and_command(tmp_path: Path) -> None:
    """REQ-REPORT-4054: smart subset = core suites + uncommitted tests, no live git."""

    targets = mod.smart_subset_targets(tmp_path)
    assert targets == [mod.CORE_SMART_SUBSET[0]]
    (tmp_path / "tests" / "python").mkdir(parents=True, exist_ok=True)
    (tmp_path / mod.CORE_SMART_SUBSET[0]).write_text("def test_ok():\n    assert True\n", encoding="utf-8")
    assert mod.smart_subset_targets(tmp_path) == [mod.CORE_SMART_SUBSET[0]]
    cmd = mod.smart_subset_command(["tests/python/test_docs.py"])
    assert cmd[0] == str(mod.PYTEST_BIN)
    assert "--no-cov" in cmd and "tests/python/test_docs.py" in cmd


def test_req_report_4054_parse_failing_ids_and_quarantine(tmp_path: Path) -> None:
    """REQ-REPORT-4054: failing-id parsing and quarantine fallback paths."""

    assert mod.parse_failing_test_ids("FAILED tests/python/a.py::test_x - boom\n") == {
        "tests/python/a.py": ["tests/python/a.py::test_x"]
    }
    assert mod.parse_failing_test_ids("ERROR tests/python/b.py::test_y - boom\n") == {
        "tests/python/b.py": ["tests/python/b.py::test_y"]
    }
    assert mod.parse_failing_test_ids("no failures here") == {}

    (tmp_path / "tests" / "python").mkdir(parents=True, exist_ok=True)
    rows = mod.quarantine_failed_tests(tmp_path, {"tests/python/gone.py": ["tests/python/gone.py::t"]})
    assert rows[0]["quarantined_path"] == "tests/quarantine/gone.py"

    src = tmp_path / "tests" / "python" / "test_real.py"
    src.write_text("x = 1\n", encoding="utf-8")
    rows = mod.quarantine_failed_tests(tmp_path, {"tests/python/test_real.py": ["id"]})
    assert (tmp_path / "tests" / "quarantine" / "test_real.py").exists()
    assert rows[0]["path"] == "tests/python/test_real.py"


def test_req_report_4054_run_pretest_until_green_caps(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """REQ-REPORT-4054: the quarantine loop is bounded and live-callable."""

    green = _green()
    monkeypatch.setattr(mod, "run_smart_subset", lambda root: green)
    assert mod.run_pretest_until_green(tmp_path, supplied=None) == (True, [], [green])

    monkeypatch.setattr(mod, "quarantine_failed_tests", lambda root, failures: [])
    repeated = mod.CommandResult(
        command=["x"],
        exit_code=1,
        stdout="FAILED tests/python/test_x.py::test_a - boom\n",
        stderr="",
    )
    ok, quarantined, results = mod.run_pretest_until_green(tmp_path, supplied=[repeated] * 8)
    assert ok is False and quarantined == [] and len(results) == 8


def test_req_report_4054_misc_helpers(tmp_path: Path) -> None:
    """REQ-REPORT-4054: small pure helpers behave and fail closed."""

    assert mod.read_active_milestone(tmp_path) == ("unknown", "research-roadmap.yaml")
    assert mod._milestone_from_text("tasks: []\n") == "unknown"
    assert mod.yaml_single_quote("complete: ok") == "'complete: ok'"
    assert mod.duration_from(None, None) == 0.0001
    assert mod.duration_from(1.0, 3.0) == 2.0
    assert mod.is_sha256("0" * 64) is True
    assert mod.is_sha256("xyz") is False
    assert mod.yaml_parses("a: 1") is True
    assert mod.yaml_parses("a: [\n") is False
    assert mod._record_id("  - id: nested") is None
    assert mod._record_id("- id: 2026.06.374") == "2026.06.374"
    assert mod._is_real_number(3) is True
    assert mod._is_real_number(True) is False
    assert mod._is_real_number("3") is False

    res = mod._run_command(["definitely-not-a-real-binary-xyz"], tmp_path)
    assert res.exit_code in {127}
    assert mod._git_lines(["rev-parse", "--bad-flag"], tmp_path) == []


def test_req_report_4054_read_active_milestone_next_roadmap(tmp_path: Path) -> None:
    """REQ-REPORT-4054: the -next roadmap is the fallback milestone source."""

    (tmp_path / "research-roadmap-next.yaml").write_text('milestone: "2026.06.375"\n', encoding="utf-8")
    assert mod.read_active_milestone(tmp_path) == ("2026.06.375", "research-roadmap-next.yaml")


def test_req_report_4054_no_forbidden_markers_rejects_compute_strings() -> None:
    """REQ-REPORT-4054: the marker guard skips closestate/principles only."""

    assert mod.no_forbidden_markers({"x": "fine"}) is True
    assert mod.no_forbidden_markers({"x": "uses CUDA kernels"}) is False
    assert mod.no_forbidden_markers({"milestone_374_closestate": {"note": "GGUF"}}) is True


# --------------------------------------------------------------------------- #
# Validation regressions
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda p: p.pop("pretest_suite_green"), "missing required"),
        (lambda p: p.update(honest_verdict="maybe"), "terminal prefix"),
        (lambda p: p.update(field_principles=[]), "field_principles"),
        (lambda p: p["field_principles"].pop("duration_s"), "missing field principles"),
        (lambda p: p.update(archived_milestone="2026.06.373"), "archived milestone"),
        (lambda p: p.update(activated_milestone="2026.06.374"), "activated milestone"),
        (lambda p: p.update(research_complete_yaml_parses=False), "research-complete"),
        (lambda p: p.update(exclusion_manifest_parses=False), "manifest"),
        (lambda p: p.update(arc_modules_importable=False), "ARC module imports"),
        (lambda p: p.update(pretest_suite_green=False), "pretest suite"),
        (lambda p: p.update(active_milestone_confirmed=False), "active milestone"),
        (lambda p: p.update(n_tasks_archived=3), "n_tasks_archived"),
        (lambda p: p.update(g3_false_retirement_corrected=False), "g3_false_retirement_corrected"),
        (lambda p: p.update(milestone_374_closestate={}), "non-empty dict"),
        (lambda p: p.update(milestone_374_closestate={"x": 1}), "per_task_status"),
        (lambda p: p.update(milestone_374_closestate={"per_task_status": {}}), "g3_decentralization_moe_base"),
        (
            lambda p: p.update(
                milestone_374_closestate={
                    "per_task_status": {},
                    "g3_decentralization_moe_base": {"retired": True, "operator_corrected_diagnosis": "x"},
                }
            ),
            "underpowered_not_retired",
        ),
        (lambda p: p.update(duration_s=0.0), "duration_s"),
        (lambda p: p.update(inference_substrate="live_model"), "inference"),
        (lambda p: p.update(quarantined_tests={}), "quarantined_tests"),
        (lambda p: p.update(model_specs={}), "model_specs"),
        (lambda p: p.update(copied_marker="CUDA"), "compute-bound markers"),
        (lambda p: p.update(reproducibility_checksum="short"), "sha256"),
        (lambda p: p.update(reproducibility_checksum="0" * 64), "does not match"),
    ],
)
def test_req_report_4054_validate_rejects_regressions(tmp_path: Path, mutate, message: str) -> None:
    """REQ-REPORT-4054: validation rejects fields that hide transition risk."""

    _seed_repo(tmp_path)
    payload = json.loads(_run_success(tmp_path).read_text(encoding="utf-8"))
    broken = json.loads(json.dumps(payload))
    mutate(broken)
    with pytest.raises(ValueError, match=message):
        mod.validate_artifact(broken)


def test_req_report_4054_terminal_verdict_carries_games() -> None:
    """REQ-REPORT-4054: the terminal verdict embeds the games-solved total."""

    verdict = mod.terminal_verdict({"accuracy": {"total_games_solved": 8}})
    assert verdict.startswith("success:")
    assert "games8" in verdict
    assert "G3_underpowered_not_retired" in verdict
    assert "G2_decision_grade_negative" in verdict


def test_req_report_4054_smart_subset_with_git(tmp_path: Path) -> None:
    """REQ-REPORT-4054: untracked tests/python files join the smart subset."""

    import subprocess as sp

    sp.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    core = tmp_path / mod.CORE_SMART_SUBSET[0]
    core.parent.mkdir(parents=True, exist_ok=True)
    core.write_text("def test_core():\n    assert True\n", encoding="utf-8")
    new_test = tmp_path / "tests" / "python" / "test_brand_new.py"
    new_test.write_text("def test_new():\n    assert True\n", encoding="utf-8")
    quarantined = tmp_path / "tests" / "quarantine" / "test_q.py"
    quarantined.parent.mkdir(parents=True, exist_ok=True)
    quarantined.write_text("def test_q():\n    assert True\n", encoding="utf-8")

    others = mod._git_lines(["ls-files", "--others", "--exclude-standard"], tmp_path)
    assert "tests/python/test_brand_new.py" in others

    targets = mod.smart_subset_targets(tmp_path)
    assert "tests/python/test_brand_new.py" in targets
    assert mod.CORE_SMART_SUBSET[0] in targets
    assert "tests/quarantine/test_q.py" not in targets


def test_req_report_4054_run_smart_subset_uses_command(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """REQ-REPORT-4054: the live gate runs the smart-subset command (no spawn)."""

    captured: dict[str, object] = {}

    def fake_run(command: list[str], root: Path) -> mod.CommandResult:
        captured["command"] = command
        return mod.CommandResult(command=command, exit_code=0, stdout="ok", stderr="")

    monkeypatch.setattr(mod, "_run_command", fake_run)
    result = mod.run_smart_subset(tmp_path)
    assert result.exit_code == 0
    assert str(mod.PYTEST_BIN) == captured["command"][0]
    assert "--no-cov" in captured["command"]


def test_scenario_report_4054_script_wrapper_exists() -> None:
    """SCENARIO-REPORT-4054: the requested experiment entrypoint exists."""

    script = Path("scripts/experiments/exp4054_archive_v374_activate_v375.py")
    assert script.exists()
    assert "archive_v374_activate_v375_4054" in script.read_text(encoding="utf-8")
