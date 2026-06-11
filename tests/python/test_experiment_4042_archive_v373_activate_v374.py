"""Tests for Exp 4042 .373 archive and .374 activation.

Spec refs: REQ-REPORT-4042, SCENARIO-REPORT-4042,
SCENARIO-REPORT-4042-BLOCKED-YAML.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from carnot.reporting import archive_v373_activate_v374_4042 as mod


SPEC_PATH = Path("openspec/capabilities/research-reporting/spec.md")


# --------------------------------------------------------------------------- #
# Fixtures: a tmp repo with the .373 artifacts and an existing .373 record.
# --------------------------------------------------------------------------- #
def _v373_artifacts() -> dict[str, dict[str, object]]:
    """Synthetic .373 task artifacts mirroring the real verdicts/fields."""

    return {
        "4029": {"honest_verdict": "success: archived_v372_v373_active_pretest_green"},
        "4030": {"honest_verdict": "complete: sota_ingestion_offarc_and_search_mapped"},
        "4031": {
            "honest_verdict": "success: offarc_transfer_runner_built_smoked_launched",
            "flagged_adversarial": True,
        },
        "4032": {
            "honest_verdict": (
                "complete: offarc_exec_verifier_directional_transfer_plus5pp_n40_"
                "oracle_headroom_present_ci_touches_zero_underpowered"
            ),
            "delta_pp": 5.0,
            "bootstrap_ci95_pp": [0.0, 12.5],
            "ci_excludes_zero": False,
            "n_tasks": 40,
            "positive_control_passes": True,
        },
        "4033": {"honest_verdict": "complete: gap4_stack_registered_offline_reeval_bitexact"},
        "4034": {
            "honest_verdict": "complete: vc33_goal_predicate_induced_heldout_precision_1.000",
            "goal_predicate_heldout_precision": 1.0,
            "goal_predicate_heldout_recall": 1.0,
        },
        "4035": {
            "honest_verdict": "complete: search_layer_no_solve_vc33_real_env_confirmation_failed",
            "game": "vc33",
            "search_layer_generalizes": False,
            "heuristic_was_non_bespoke": True,
            "nodes_expanded": 169,
            "new_levels_solved_this_task": 0,
            "levels_completed_after": 0,
            "real_env_confirmed": False,
            "search_found_plan": True,
            "goal_predicate_heldout_precision": 1.0,
        },
        "4036": {"honest_verdict": "success: decentralization_stronger_base_runner_launched"},
        "4037": {
            "honest_verdict": "complete: decentralization_stronger_base_partial_0_tasks",
            "n_tasks_scored": 0,
            "coverage_delta_vs_12b": -0.2581,
            "stronger_base_demo_perfect_coverage": 0.0,
            "gated_pass_at_2": 0.0,
        },
        "4038": {
            "honest_verdict": "success: seventh_game_solved_dc22-fdcac232_at_action_20",
            "total_games_solved": 7,
            "prior_total_games_solved": 6,
            "target_game": "dc22-fdcac232",
            "first_solve_at_action": 20,
            "real_env_confirmed": True,
        },
        "4039": {
            "honest_verdict": "success: arcmemo_v6_library_transfer_59to18_actions",
            "actions_cold": 59,
            "actions_v6": 18,
            "actions_v5": 20,
            "induction_calls_v6": 0,
            "solve_transfer_win": True,
            "n_named_abstractions": 1,
        },
        "4040": {
            "honest_verdict": "complete: hardware_continuity_kv260_overlay_loaded_latency_step_blocked",
            "per_board_reachability": {"kv260": True, "gatemate": False, "polarfire": True},
            "per_board_terminal_state": {
                "kv260": "reachable_overlay_loaded_latency_step_blocked",
                "gatemate": "blocked_gatemate_unreachable",
                "polarfire": "reachable_ssh_continuity_recorded",
            },
            "kv260_overlay_loaded": True,
            "kv260_latency_step_taken": False,
        },
        "4041": {
            "honest_verdict": (
                "complete: capstone_v373_arguments_measured_G1_directional_underpowered_"
                "ci_touches_zero_G2_no_generalization_G3_absent_games7_arcmemo_win_flagged_skipped1"
            )
        },
    }


def _v373_record_block(idx: int) -> str:
    return (
        "- id: 2026.06.373\n"
        f"  title: 'THE OFF-ARC MEASUREMENT MILESTONE (copy {idx})'\n"
        "  doc: openspec/change-proposals/research-roadmap-v373.md\n"
        "  completed: '2026-06-11'\n"
        "  finding: See conductor log for per-experiment results.\n"
        "  tasks:\n"
        "  - id: exp4035-hierarchical-search-over-vc33-wm\n"
        "    result: OK (conductor)\n"
    )


def _seed_repo(
    root: Path,
    *,
    corrupt_complete: bool = False,
    milestone: str = "2026.06.374",
    manifest: str = "retired: []\n",
    n_373_records: int = 1,
) -> None:
    (root / "research-roadmap.yaml").write_text(
        f'milestone: "{milestone}"\ntasks:\n  - id: exp4042-archive-v373-activate-v374\n',
        encoding="utf-8",
    )
    complete_text = (
        "milestones:\n"
        "- id: 2026.06.372\n"
        "  title: prior archive\n"
        "  completed: '2026-06-11'\n"
        "  tasks:\n"
        "  - id: exp4029-archive\n"
        "    result: OK (conductor)\n"
    )
    for idx in range(n_373_records):
        complete_text += _v373_record_block(idx)
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
    payloads = _v373_artifacts()
    for task in mod.V373_TASKS:
        exp_id = str(task["exp_id"])
        if exp_id in payloads:
            (results / Path(str(task["deliverable"])).name).write_text(
                json.dumps(payloads[exp_id], sort_keys=True), encoding="utf-8"
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
def test_req_report_4042_spec_anchor_exists() -> None:
    """REQ-REPORT-4042: OpenSpec declares the .373 archive contract."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    assert "REQ-REPORT-4042" in spec
    assert "SCENARIO-REPORT-4042" in spec
    assert "SCENARIO-REPORT-4042-BLOCKED-YAML" in spec
    assert mod.OUTPUT_REL_PATH.as_posix() in spec
    assert "milestone_373_closestate" in spec
    assert "smart-subset" in spec


# --------------------------------------------------------------------------- #
# Complete path
# --------------------------------------------------------------------------- #
def test_scenario_report_4042_records_closestate_unchanged(tmp_path: Path) -> None:
    """SCENARIO-REPORT-4042: one existing .373 record stays, truth recorded."""

    _seed_repo(tmp_path, n_373_records=1)
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
    assert artifact["archived_milestone"] == "2026.06.373"
    assert artifact["activated_milestone"] == "2026.06.374"
    assert artifact["active_milestone_confirmed"] == "2026.06.374"
    assert artifact["research_complete_yaml_parses"] is True
    assert artifact["exclusion_manifest_parses"] is True
    assert artifact["arc_modules_importable"] is True
    assert artifact["pretest_suite_green"] is True
    assert artifact["quarantined_tests"] == []
    assert artifact["inference_substrate"] == "aggregation_from_upstream_artifacts"
    assert artifact["research_complete_record_action"] == "unchanged"
    assert artifact["research_complete_duplicates_removed"] == 0
    assert artifact["n_tasks_archived"] == 13

    # An existing single .373 record is left byte-for-byte unchanged.
    assert (tmp_path / "research-complete.yaml").read_text(encoding="utf-8") == before["complete"]
    loaded = yaml.safe_load(before["complete"])
    ids = [m["id"] for m in loaded["milestones"]]
    assert ids.count("2026.06.373") == 1

    # Close-state truth: per-task status + the three G-gate measurements.
    cs = artifact["milestone_373_closestate"]
    assert cs["per_task_status"]["exp4031-offarc-exec-verifier-transfer-build"] == "FLAGGED"
    assert cs["per_task_status"]["exp4035-hierarchical-search-over-vc33-wm"] == "OK"
    assert cs["per_task_status"]["exp4038-seventh-game-explore-first"] == "OK"
    assert cs["status_counts"]["OK"] == 12
    assert cs["status_counts"]["FLAGGED"] == 1
    # G1: off-ARC directional but underpowered (CI touches 0).
    assert cs["g1_off_arc_transfer"]["delta_pp"] == 5.0
    assert cs["g1_off_arc_transfer"]["bootstrap_ci95_pp"] == [0.0, 12.5]
    assert cs["g1_off_arc_transfer"]["ci_lower_bound"] == 0.0
    assert cs["g1_off_arc_transfer"]["ci_excludes_zero"] is False
    assert cs["g1_off_arc_transfer"]["n_tasks"] == 40
    assert cs["g1_off_arc_transfer"]["positive_control_passes"] is True
    assert cs["g1_off_arc_transfer"]["verifier_generalized_off_arc"] is False
    assert cs["g1_off_arc_transfer"]["outcome"] == "directional_underpowered_ci_touches_zero"
    # G2: no search-layer generalization on vc33 (degenerate plan failed real-env).
    assert cs["g2_search_layer_generalization"]["game"] == "vc33"
    assert cs["g2_search_layer_generalization"]["goal_predicate_heldout_precision"] == 1.0
    assert cs["g2_search_layer_generalization"]["heuristic_was_non_bespoke"] is True
    assert cs["g2_search_layer_generalization"]["search_found_plan"] is True
    assert cs["g2_search_layer_generalization"]["real_env_confirmed"] is False
    assert cs["g2_search_layer_generalization"]["search_layer_generalizes"] is False
    assert cs["g2_search_layer_generalization"]["degenerate_wm_exploiting_plan"] is True
    assert cs["g2_search_layer_generalization"]["nodes_expanded"] == 169
    # G3: decentralization absent (0 tasks scored, throughput failure).
    assert cs["g3_decentralization"]["diagnosis"] == "absent"
    assert cs["g3_decentralization"]["n_tasks_scored"] == 0
    assert cs["g3_decentralization"]["throughput_failure"] is True
    assert cs["g3_decentralization"]["coverage_delta_vs_12b"] == pytest.approx(-0.2581)
    assert cs["g3_decentralization"]["baseline_12b_coverage"] == pytest.approx(0.2581)
    assert cs["g3_decentralization"]["beat_12b_ceiling"] is False
    # Accuracy: 7 games solved, +1 monotonic.
    assert cs["accuracy"]["total_games_solved"] == 7
    assert cs["accuracy"]["monotonic_plus_one"] is True
    assert cs["accuracy"]["target_game"] == "dc22-fdcac232"
    assert cs["accuracy"]["first_solve_at_action"] == 20
    # Self-learning: ArcMemo v6 transfer win, 59->18 actions.
    assert cs["self_learning"]["transfer_win"] is True
    assert cs["self_learning"]["actions_cold"] == 59
    assert cs["self_learning"]["actions_v6"] == 18
    assert cs["self_learning"]["action_savings_vs_cold"] == 41
    assert cs["self_learning"]["induction_calls_v6"] == 0
    # exp4031 flagged-skipped (never aggregated as a win).
    assert cs["transfer_build_flagged"]["flagged_adversarial"] is True
    assert cs["transfer_build_flagged"]["skipped"] is True
    # Hardware continuity.
    assert cs["hardware"]["per_board_reachability"]["kv260"] is True
    assert cs["hardware"]["kv260_overlay_loaded"] is True
    assert cs["hardware"]["kv260_latency_step_taken"] is False
    assert cs["capstone_v373_verdict"].startswith("complete: capstone_v373")
    assert "UNDERPOWERED" in cs["headline"]
    assert "ABSENT" in cs["headline"]

    # Operator-curated / conductor-reconciled files untouched.
    assert (tmp_path / "ops" / "exclusion_manifest.yaml").read_text(encoding="utf-8") == before["manifest"]
    assert (tmp_path / "research-roadmap.yaml").read_text(encoding="utf-8") == before["roadmap"]
    assert (tmp_path / "ops" / "changelog.md").read_text(encoding="utf-8") == before["changelog"]
    assert (tmp_path / "ops" / "status.md").read_text(encoding="utf-8") == before["status"]
    assert (tmp_path / "_bmad" / "traceability.md").read_text(encoding="utf-8") == before["trace"]
    assert (tmp_path / "scripts" / "research_conductor.py").read_text(encoding="utf-8") == before["conductor"]


def test_req_report_4042_dedupes_duplicate_records(tmp_path: Path) -> None:
    """REQ-REPORT-4042: interrupted-run duplicate .373 records collapse to one."""

    _seed_repo(tmp_path, n_373_records=5)
    artifact = json.loads(_run_success(tmp_path).read_text(encoding="utf-8"))
    assert artifact["research_complete_record_action"] == "deduped"
    assert artifact["research_complete_duplicates_removed"] == 4
    loaded = yaml.safe_load((tmp_path / "research-complete.yaml").read_text(encoding="utf-8"))
    ids = [m["id"] for m in loaded["milestones"]]
    assert ids.count("2026.06.373") == 1


def test_req_report_4042_dedupe_is_idempotent(tmp_path: Path) -> None:
    """REQ-REPORT-4042: rerunning leaves exactly one .373 record (unchanged)."""

    _seed_repo(tmp_path, n_373_records=5)
    _run_success(tmp_path)
    first_complete = (tmp_path / "research-complete.yaml").read_text(encoding="utf-8")
    second = json.loads(_run_success(tmp_path).read_text(encoding="utf-8"))

    assert (tmp_path / "research-complete.yaml").read_text(encoding="utf-8") == first_complete
    assert second["research_complete_record_action"] == "unchanged"
    assert second["research_complete_duplicates_removed"] == 0


def test_req_report_4042_appends_when_record_absent(tmp_path: Path) -> None:
    """REQ-REPORT-4042: a missing .373 record is appended canonically."""

    _seed_repo(tmp_path, n_373_records=0)
    artifact = json.loads(_run_success(tmp_path).read_text(encoding="utf-8"))

    assert artifact["research_complete_record_action"] == "appended"
    loaded = yaml.safe_load((tmp_path / "research-complete.yaml").read_text(encoding="utf-8"))
    ids = [m["id"] for m in loaded["milestones"]]
    assert ids.count("2026.06.373") == 1
    record = next(m for m in loaded["milestones"] if m["id"] == "2026.06.373")
    assert record["activation_recorded"] == "exp4042-archive-v373-activate-v374"
    assert len(record["tasks"]) == len(mod.V373_TASKS)


# --------------------------------------------------------------------------- #
# Blocked paths
# --------------------------------------------------------------------------- #
def test_scenario_report_4042_blocked_yaml_writes_artifact_without_edits(tmp_path: Path) -> None:
    """SCENARIO-REPORT-4042-BLOCKED-YAML: corrupt YAML exits before edits."""

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
    assert artifact["milestone_373_closestate"]["status"] == "blocked"
    assert artifact["active_milestone_confirmed"] == "2026.06.374"
    assert artifact["preconditions_checked"]["research_complete_yaml_parsed_before"] is False
    assert (tmp_path / "research-complete.yaml").read_text(encoding="utf-8") == before_complete
    assert (tmp_path / "ops" / "exclusion_manifest.yaml").read_text(encoding="utf-8") == before_manifest


def test_req_report_4042_blocked_when_complete_missing(tmp_path: Path) -> None:
    """REQ-REPORT-4042: a missing research-complete.yaml fails closed."""

    _seed_repo(tmp_path)
    (tmp_path / "research-complete.yaml").unlink()
    artifact = json.loads(_run_success(tmp_path).read_text(encoding="utf-8"))
    assert artifact["honest_verdict"].startswith("blocked_research_complete_yaml_poison_missing")


def test_req_report_4042_blocked_paths_fail_closed(tmp_path: Path) -> None:
    """REQ-REPORT-4042: missing handoff facts block instead of fabricating success."""

    _seed_repo(tmp_path, milestone="2026.06.373")
    artifact = json.loads(_run_success(tmp_path).read_text(encoding="utf-8"))
    assert artifact["honest_verdict"].startswith("blocked_v374_not_active")

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


def test_req_report_4042_blocked_when_pretest_unquarantinable(tmp_path: Path) -> None:
    """REQ-REPORT-4042: a red gate with no parseable failure id blocks."""

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
    assert "per_task_status" in artifact["milestone_373_closestate"]


def test_req_report_4042_quarantines_red_test_then_green(tmp_path: Path) -> None:
    """REQ-REPORT-4042: a red smart-subset file is git-mv'd to quarantine."""

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


def test_req_report_4042_blocked_edit_invalid(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """REQ-REPORT-4042: an edit that breaks YAML blocks before writing."""

    _seed_repo(tmp_path)
    monkeypatch.setattr(mod, "dedupe_or_append_record", lambda *a: ("milestones: [\n", 0, "deduped"))
    artifact = json.loads(_run_success(tmp_path).read_text(encoding="utf-8"))
    assert artifact["honest_verdict"].startswith("blocked_research_complete_edit_invalid")
    # The original file was NOT overwritten with the broken edit.
    assert yaml.safe_load((tmp_path / "research-complete.yaml").read_text(encoding="utf-8"))


def test_req_report_4042_blocked_poison_after_edit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """REQ-REPORT-4042: a post-write parse regression is caught."""

    _seed_repo(tmp_path)
    # Calls in order: parses_before, edit-candidate, on-disk re-read, manifest.
    states = iter([True, True, False, True])

    monkeypatch.setattr(mod, "yaml_parses", lambda text: next(states))
    artifact = json.loads(_run_success(tmp_path).read_text(encoding="utf-8"))
    assert artifact["honest_verdict"].startswith("blocked_research_complete_yaml_poison_after_edit")


# --------------------------------------------------------------------------- #
# Close-state + helper unit tests
# --------------------------------------------------------------------------- #
def test_req_report_4042_classify_status_branches() -> None:
    """REQ-REPORT-4042: every status class is reachable."""

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


def test_req_report_4042_build_closestate_from_records() -> None:
    """REQ-REPORT-4042: the close-state builder is a pure aggregation."""

    records: dict[str, dict[str, object]] = {
        exp_id: _record(payload) for exp_id, payload in _v373_artifacts().items()
    }
    closestate = mod.build_closestate(records)

    assert closestate["status_counts"]["OK"] == 12
    assert closestate["status_counts"]["FLAGGED"] == 1
    assert closestate["g1_off_arc_transfer"]["verifier_generalized_off_arc"] is False
    assert closestate["g2_search_layer_generalization"]["search_layer_generalizes"] is False
    assert closestate["g3_decentralization"]["diagnosis"] == "absent"
    assert closestate["accuracy"]["total_games_solved"] == 7
    assert closestate["self_learning"]["action_savings_vs_cold"] == 41
    assert closestate["transfer_build_flagged"]["skipped"] is True
    assert closestate["capstone_v373_verdict"].startswith("complete: capstone_v373")
    assert "vc33" in closestate["headline"]


def test_req_report_4042_closestate_subbuilders_degrade_gracefully() -> None:
    """REQ-REPORT-4042: every sub-builder returns null facts on missing input."""

    empty = {"exists": False}

    g1 = mod._g1_off_arc_transfer(empty)
    assert g1["delta_pp"] is None
    assert g1["bootstrap_ci95_pp"] is None
    assert g1["ci_lower_bound"] is None
    assert g1["verifier_generalized_off_arc"] is False
    assert g1["outcome"] == "no_transfer"

    g2 = mod._g2_search_layer_generalization(empty, empty)
    assert g2["game"] is None
    assert g2["search_layer_generalizes"] is False
    assert g2["degenerate_wm_exploiting_plan"] is False

    g3 = mod._g3_decentralization(empty)
    assert g3["n_tasks_scored"] is None
    assert g3["diagnosis"] == "measured"
    assert g3["throughput_failure"] is False
    assert g3["baseline_12b_coverage"] is None

    accuracy = mod._accuracy(empty)
    assert accuracy["total_games_solved"] is None and accuracy["monotonic_plus_one"] is False

    self_learning = mod._self_learning(empty)
    assert self_learning["actions_cold"] is None
    assert self_learning["action_savings_vs_cold"] is None

    hardware = mod._hardware(empty)
    assert hardware["per_board_reachability"] == {} and hardware["included"] is False
    assert hardware["per_board_terminal_state"] == {}

    flagged = mod._decentralization_flagged(empty)
    assert flagged["flagged_adversarial"] is False and flagged["skipped"] is False

    # _fields tolerates a non-mapping fields value.
    assert mod._fields({"fields": [1, 2, 3]}) == {}
    assert mod._fields({}) == {}


def test_req_report_4042_g1_branches() -> None:
    """REQ-REPORT-4042: G1 outcome covers underpowered, generalized, no-transfer."""

    underpowered = _record(
        {"delta_pp": 5.0, "bootstrap_ci95_pp": [0.0, 12.5], "ci_excludes_zero": False, "n_tasks": 40}
    )
    g1 = mod._g1_off_arc_transfer(underpowered)
    assert g1["outcome"] == "directional_underpowered_ci_touches_zero"
    assert g1["ci_lower_bound"] == 0.0

    generalized = _record(
        {"delta_pp": 10.0, "bootstrap_ci95_pp": [2.0, 18.0], "ci_excludes_zero": True, "n_tasks": 500}
    )
    g1b = mod._g1_off_arc_transfer(generalized)
    assert g1b["outcome"] == "generalized_ci_excludes_zero"
    assert g1b["verifier_generalized_off_arc"] is True

    # A non-positive delta with CI not excluding zero is no transfer; a non-list
    # CI normalizes to None without raising.
    flat = _record({"delta_pp": 0.0, "bootstrap_ci95_pp": "n/a", "ci_excludes_zero": False})
    g1c = mod._g1_off_arc_transfer(flat)
    assert g1c["outcome"] == "no_transfer"
    assert g1c["bootstrap_ci95_pp"] is None
    assert g1c["ci_lower_bound"] is None

    # A bool delta is not a real positive number -> no_transfer.
    boolish = _record({"delta_pp": True, "bootstrap_ci95_pp": [], "ci_excludes_zero": False})
    assert mod._g1_off_arc_transfer(boolish)["outcome"] == "no_transfer"


def test_req_report_4042_g3_measured_branch() -> None:
    """REQ-REPORT-4042: G3 reports 'measured' when tasks are scored."""

    measured = _record(
        {"n_tasks_scored": 50, "coverage_delta_vs_12b": 0.1, "stronger_base_demo_perfect_coverage": 0.36}
    )
    g3 = mod._g3_decentralization(measured)
    assert g3["diagnosis"] == "measured"
    assert g3["throughput_failure"] is False
    assert g3["baseline_12b_coverage"] == pytest.approx(0.26)
    assert g3["beat_12b_ceiling"] is False


def test_req_report_4042_accuracy_nonmonotonic_and_self_learning_bool() -> None:
    """REQ-REPORT-4042: accuracy guards reject non-+1 jumps and bool counts."""

    jump = _record({"total_games_solved": 9, "prior_total_games_solved": 6})
    assert mod._accuracy(jump)["monotonic_plus_one"] is False

    bool_counts = _record({"actions_cold": True, "actions_v6": 1})
    assert mod._self_learning(bool_counts)["action_savings_vs_cold"] is None


def test_req_report_4042_read_artifact_record_fail_closed(tmp_path: Path) -> None:
    """REQ-REPORT-4042: unreadable / non-mapping artifacts read as absent."""

    assert mod.read_artifact_record(tmp_path / "missing.json")["exists"] is False
    listish = tmp_path / "listish.json"
    listish.write_text("[1, 2, 3]", encoding="utf-8")
    assert mod.read_artifact_record(listish)["exists"] is False
    good = tmp_path / "good.json"
    good.write_text(json.dumps({"honest_verdict": "complete: ok"}), encoding="utf-8")
    assert mod.read_artifact_record(good)["exists"] is True


def test_req_report_4042_read_v373_records_reads_all_tasks(tmp_path: Path) -> None:
    """REQ-REPORT-4042: every .373 deliverable is read by exp id."""

    _seed_repo(tmp_path)
    records = mod.read_v373_records(tmp_path)
    assert set(records) == {str(t["exp_id"]) for t in mod.V373_TASKS}
    assert records["4031"]["flagged_adversarial"] is True
    assert records["4035"]["fields"]["nodes_expanded"] == 169


def test_req_report_4042_dedupe_helper_branches() -> None:
    """REQ-REPORT-4042: dedupe/append helper covers all three actions."""

    base = "milestones:\n- id: 2026.06.372\n  title: a\n"
    two = base + _v373_record_block(0) + _v373_record_block(1)
    deduped, removed, action = mod.dedupe_or_append_record(two, "2026.06.373")
    assert action == "deduped" and removed == 1
    assert deduped.count("- id: 2026.06.373") == 1

    one = base + _v373_record_block(0)
    unchanged, removed, action = mod.dedupe_or_append_record(one, "2026.06.373")
    assert action == "unchanged" and removed == 0 and unchanged == one

    appended, removed, action = mod.dedupe_or_append_record(base, "2026.06.373")
    assert action == "appended" and removed == 0
    assert appended.count("- id: 2026.06.373") == 1
    assert yaml.safe_load(appended)


def test_req_report_4042_smart_subset_targets_and_command(tmp_path: Path) -> None:
    """REQ-REPORT-4042: smart subset = core suites + uncommitted tests, no live git."""

    # No git repo here -> git helpers return [] -> only existing core survive,
    # and the fallback kicks in when none exist.
    targets = mod.smart_subset_targets(tmp_path)
    assert targets == [mod.CORE_SMART_SUBSET[0]]
    (tmp_path / "tests" / "python").mkdir(parents=True, exist_ok=True)
    (tmp_path / mod.CORE_SMART_SUBSET[0]).write_text("def test_ok():\n    assert True\n", encoding="utf-8")
    assert mod.smart_subset_targets(tmp_path) == [mod.CORE_SMART_SUBSET[0]]
    cmd = mod.smart_subset_command(["tests/python/test_docs.py"])
    assert cmd[0] == str(mod.PYTEST_BIN)
    assert "--no-cov" in cmd and "tests/python/test_docs.py" in cmd


def test_req_report_4042_parse_failing_ids_and_quarantine(tmp_path: Path) -> None:
    """REQ-REPORT-4042: failing-id parsing and quarantine fallback paths."""

    assert mod.parse_failing_test_ids("FAILED tests/python/a.py::test_x - boom\n") == {
        "tests/python/a.py": ["tests/python/a.py::test_x"]
    }
    assert mod.parse_failing_test_ids("ERROR tests/python/b.py::test_y - boom\n") == {
        "tests/python/b.py": ["tests/python/b.py::test_y"]
    }
    assert mod.parse_failing_test_ids("no failures here") == {}

    # Quarantine when the source file does not exist -> still records the audit
    # row and never crashes (rename fallback path is git-mv first).
    (tmp_path / "tests" / "python").mkdir(parents=True, exist_ok=True)
    rows = mod.quarantine_failed_tests(tmp_path, {"tests/python/gone.py": ["tests/python/gone.py::t"]})
    assert rows[0]["quarantined_path"] == "tests/quarantine/gone.py"

    # Rename fallback when git mv fails but the file exists.
    src = tmp_path / "tests" / "python" / "test_real.py"
    src.write_text("x = 1\n", encoding="utf-8")
    rows = mod.quarantine_failed_tests(tmp_path, {"tests/python/test_real.py": ["id"]})
    assert (tmp_path / "tests" / "quarantine" / "test_real.py").exists()
    assert rows[0]["path"] == "tests/python/test_real.py"


def test_req_report_4042_run_pretest_until_green_caps(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """REQ-REPORT-4042: the quarantine loop is bounded and live-callable."""

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


def test_req_report_4042_misc_helpers(tmp_path: Path) -> None:
    """REQ-REPORT-4042: small pure helpers behave and fail closed."""

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
    assert mod._record_id("- id: 2026.06.373") == "2026.06.373"
    assert mod._is_real_number(3) is True
    assert mod._is_real_number(True) is False
    assert mod._is_real_number("3") is False

    # _run_command OSError -> exit 127.
    res = mod._run_command(["definitely-not-a-real-binary-xyz"], tmp_path)
    assert res.exit_code in {127}
    # _git_lines returns [] when git command fails.
    assert mod._git_lines(["rev-parse", "--bad-flag"], tmp_path) == []


def test_req_report_4042_read_active_milestone_next_roadmap(tmp_path: Path) -> None:
    """REQ-REPORT-4042: the -next roadmap is the fallback milestone source."""

    (tmp_path / "research-roadmap-next.yaml").write_text('milestone: "2026.06.374"\n', encoding="utf-8")
    assert mod.read_active_milestone(tmp_path) == ("2026.06.374", "research-roadmap-next.yaml")


def test_req_report_4042_no_forbidden_markers_rejects_compute_strings() -> None:
    """REQ-REPORT-4042: the marker guard skips closestate/principles only."""

    assert mod.no_forbidden_markers({"x": "fine"}) is True
    assert mod.no_forbidden_markers({"x": "uses CUDA kernels"}) is False
    # The closestate + field_principles are exempt (they legitimately narrate).
    assert mod.no_forbidden_markers({"milestone_373_closestate": {"note": "GGUF"}}) is True


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
        (lambda p: p.update(archived_milestone="2026.06.372"), "archived milestone"),
        (lambda p: p.update(activated_milestone="2026.06.373"), "activated milestone"),
        (lambda p: p.update(research_complete_yaml_parses=False), "research-complete"),
        (lambda p: p.update(exclusion_manifest_parses=False), "manifest"),
        (lambda p: p.update(arc_modules_importable=False), "ARC module imports"),
        (lambda p: p.update(pretest_suite_green=False), "pretest suite"),
        (lambda p: p.update(active_milestone_confirmed=False), "active milestone"),
        (lambda p: p.update(n_tasks_archived=3), "n_tasks_archived"),
        (lambda p: p.update(milestone_373_closestate={}), "non-empty dict"),
        (lambda p: p.update(milestone_373_closestate={"x": 1}), "per_task_status"),
        (lambda p: p.update(duration_s=0.0), "duration_s"),
        (lambda p: p.update(inference_substrate="live_model"), "inference"),
        (lambda p: p.update(quarantined_tests={}), "quarantined_tests"),
        (lambda p: p.update(model_specs={}), "model_specs"),
        (lambda p: p.update(copied_marker="CUDA"), "compute-bound markers"),
        (lambda p: p.update(reproducibility_checksum="short"), "sha256"),
        (lambda p: p.update(reproducibility_checksum="0" * 64), "does not match"),
    ],
)
def test_req_report_4042_validate_rejects_regressions(tmp_path: Path, mutate, message: str) -> None:
    """REQ-REPORT-4042: validation rejects fields that hide transition risk."""

    _seed_repo(tmp_path)
    payload = json.loads(_run_success(tmp_path).read_text(encoding="utf-8"))
    broken = json.loads(json.dumps(payload))
    mutate(broken)
    with pytest.raises(ValueError, match=message):
        mod.validate_artifact(broken)


def test_req_report_4042_terminal_verdict_carries_games() -> None:
    """REQ-REPORT-4042: the terminal verdict embeds the games-solved total."""

    verdict = mod.terminal_verdict({"accuracy": {"total_games_solved": 7}})
    assert verdict.startswith("success:")
    assert "games7" in verdict
    assert "G2_no_generalization" in verdict


def test_req_report_4042_smart_subset_with_git(tmp_path: Path) -> None:
    """REQ-REPORT-4042: untracked tests/python files join the smart subset."""

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

    # _git_lines success branch returns the untracked files.
    others = mod._git_lines(["ls-files", "--others", "--exclude-standard"], tmp_path)
    assert "tests/python/test_brand_new.py" in others

    targets = mod.smart_subset_targets(tmp_path)
    assert "tests/python/test_brand_new.py" in targets
    assert mod.CORE_SMART_SUBSET[0] in targets
    assert "tests/quarantine/test_q.py" not in targets


def test_req_report_4042_run_smart_subset_uses_command(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """REQ-REPORT-4042: the live gate runs the smart-subset command (no spawn)."""

    captured: dict[str, object] = {}

    def fake_run(command: list[str], root: Path) -> mod.CommandResult:
        captured["command"] = command
        return mod.CommandResult(command=command, exit_code=0, stdout="ok", stderr="")

    monkeypatch.setattr(mod, "_run_command", fake_run)
    result = mod.run_smart_subset(tmp_path)
    assert result.exit_code == 0
    assert str(mod.PYTEST_BIN) == captured["command"][0]
    assert "--no-cov" in captured["command"]


def test_scenario_report_4042_script_wrapper_exists() -> None:
    """SCENARIO-REPORT-4042: the requested experiment entrypoint exists."""

    script = Path("scripts/experiments/exp4042_archive_v373_activate_v374.py")
    assert script.exists()
    assert "archive_v373_activate_v374_4042" in script.read_text(encoding="utf-8")
