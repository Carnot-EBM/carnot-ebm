"""Tests for Exp 4029 .372 archive and .373 activation.

Spec refs: REQ-REPORT-4029, SCENARIO-REPORT-4029,
SCENARIO-REPORT-4029-BLOCKED-YAML.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from carnot.reporting import archive_v372_activate_v373_4029 as mod


SPEC_PATH = Path("openspec/capabilities/research-reporting/spec.md")


# --------------------------------------------------------------------------- #
# Fixtures: a tmp repo with the .372 artifacts and duplicated .372 records.
# --------------------------------------------------------------------------- #
def _v372_artifacts() -> dict[str, dict[str, object]]:
    """Synthetic .372 task artifacts mirroring the real verdicts/fields."""

    return {
        "4019": {"honest_verdict": "complete: archived_v371_v372_active_pretest_green"},
        "4020": {
            "honest_verdict": "complete: goal_predicate_induced_heldout_precision_1.000",
            "goal_predicate_heldout_precision": 1.0,
        },
        "4021": {
            "honest_verdict": "complete: search_layer_solved_r11l_L4_real_env_confirmed",
            "game": "r11l",
            "nodes_expanded": 3,
            "action_count": 2,
            "heuristic_used": "coded_unmet_targets_plus_manhattan_progress",
            "new_levels_solved_this_task": 1,
            "wall_was_search_not_representation": True,
            "real_env_confirmed": True,
        },
        "4022": {
            "honest_verdict": "complete: B_distill_feasibility_exp4012_no_lift_clean_traces40",
            "flagged_adversarial": True,
        },
        "4023": {
            "honest_verdict": "complete: agreement_selector_retired_confidence_label_only",
            "retired_r_and_d_line": "smart_selector_agreement_precision_confirmation",
            "safety_gate_kept": True,
            "agreement_is_precision_selector": False,
        },
        "4024": {
            "honest_verdict": "success: fifth_game_solved_cd82-fb555c5d_at_action_5",
            "total_games_solved": 6,
            "prior_total_games_solved": 5,
            "target_game": "cd82-fb555c5d",
            "real_env_confirmed": True,
        },
        "4025": {"honest_verdict": "success: arcmemo_v5_transfer_71to21_actions"},
        "4026": {
            "honest_verdict": "success: verifier_parity_wallclock_95.3x_judge_over_verifier",
            "flagged_adversarial": False,
            "wallclock_seconds_ratio_judge_over_verifier": 95.2564,
            "token_ratio_judge_over_verifier": 236.2903,
            "accuracy_parity": True,
            "accuracy_gap": 0.0161,
        },
        "4027": {
            "honest_verdict": "complete: hardware_continuity_4027_ssh_continuity_recorded",
            "per_board_reachability": {"kv260": True, "gatemate": False, "polarfire": True},
        },
        "4028": {
            "honest_verdict": (
                "success: capstone_v372_deep_think_pivot_ADVANCED_search_levels1_"
                "decentralization_skipped_flagged_exp4022_games_delta1_efficiency_win"
            )
        },
    }


def _v372_record_block(idx: int) -> str:
    return (
        "- id: 2026.06.372\n"
        f"  title: 'THE DEEP-THINK PIVOT (copy {idx})'\n"
        "  doc: docs/research-notes/deep-think-results-2026-06-10.md\n"
        "  completed: '2026-06-11'\n"
        "  finding: See conductor log for per-experiment results.\n"
        "  tasks:\n"
        "  - id: exp4021-heuristic-search-over-verified-wm\n"
        "    result: OK (conductor)\n"
    )


def _seed_repo(
    root: Path,
    *,
    corrupt_complete: bool = False,
    milestone: str = "2026.06.373",
    manifest: str = "retired: []\n",
    n_372_records: int = 22,
) -> None:
    (root / "research-roadmap.yaml").write_text(
        f'milestone: "{milestone}"\ntasks:\n  - id: exp4029-archive-v372-activate-v373\n',
        encoding="utf-8",
    )
    complete_text = (
        "milestones:\n"
        "- id: 2026.06.371\n"
        "  title: prior archive\n"
        "  completed: '2026-06-10'\n"
        "  tasks:\n"
        "  - id: exp4008-archive\n"
        "    result: OK (conductor)\n"
    )
    for idx in range(n_372_records):
        complete_text += _v372_record_block(idx)
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
    payloads = _v372_artifacts()
    for task in mod.V372_TASKS:
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
def test_req_report_4029_spec_anchor_exists() -> None:
    """REQ-REPORT-4029: OpenSpec declares the .372 archive contract."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    assert "REQ-REPORT-4029" in spec
    assert "SCENARIO-REPORT-4029" in spec
    assert "SCENARIO-REPORT-4029-BLOCKED-YAML" in spec
    assert mod.OUTPUT_REL_PATH.as_posix() in spec
    assert "milestone_372_closestate" in spec
    assert "smart-subset" in spec


# --------------------------------------------------------------------------- #
# Complete path
# --------------------------------------------------------------------------- #
def test_scenario_report_4029_dedupes_and_records_closestate(tmp_path: Path) -> None:
    """SCENARIO-REPORT-4029: collapse 22 duplicate .372 records, record truth."""

    _seed_repo(tmp_path, n_372_records=22)
    before = {
        "manifest": (tmp_path / "ops" / "exclusion_manifest.yaml").read_text(encoding="utf-8"),
        "roadmap": (tmp_path / "research-roadmap.yaml").read_text(encoding="utf-8"),
        "changelog": (tmp_path / "ops" / "changelog.md").read_text(encoding="utf-8"),
        "status": (tmp_path / "ops" / "status.md").read_text(encoding="utf-8"),
        "trace": (tmp_path / "_bmad" / "traceability.md").read_text(encoding="utf-8"),
        "conductor": (tmp_path / "scripts" / "research_conductor.py").read_text(encoding="utf-8"),
    }

    out_path = _run_success(tmp_path)
    artifact = json.loads(out_path.read_text(encoding="utf-8"))

    mod.validate_artifact(artifact)
    assert artifact["honest_verdict"].startswith("success:")
    assert artifact["archived_milestone"] == "2026.06.372"
    assert artifact["activated_milestone"] == "2026.06.373"
    assert artifact["active_milestone_confirmed"] == "2026.06.373"
    assert artifact["research_complete_yaml_parses"] is True
    assert artifact["exclusion_manifest_parses"] is True
    assert artifact["arc_modules_importable"] is True
    assert artifact["pretest_suite_green"] is True
    assert artifact["quarantined_tests"] == []
    assert artifact["inference_substrate"] == "aggregation_from_upstream_artifacts"
    assert artifact["research_complete_record_action"] == "deduped"
    assert artifact["research_complete_duplicates_removed"] == 21

    # research-complete now has exactly one .372 record and still parses.
    complete_text = (tmp_path / "research-complete.yaml").read_text(encoding="utf-8")
    loaded = yaml.safe_load(complete_text)
    ids = [m["id"] for m in loaded["milestones"]]
    assert ids.count("2026.06.372") == 1

    # Close-state truth: per-task status + the four mandated facts.
    cs = artifact["milestone_372_closestate"]
    assert cs["per_task_status"]["exp4021-heuristic-search-over-verified-wm"] == "OK"
    assert cs["per_task_status"]["exp4022-decentralization-gated-on-exp4012"] == "FLAGGED"
    assert cs["per_task_status"]["exp4024-fifth-game-explore-first"] == "OK"
    assert cs["status_counts"]["OK"] == 9
    assert cs["status_counts"]["FLAGGED"] == 1
    # 1. Search-layer thin-win (exp4021 nodes_expanded=3 with r11l-specific macros).
    assert cs["search_layer"]["nodes_expanded"] == 3
    assert cs["search_layer"]["search_game"] == "r11l"
    assert cs["search_layer"]["thin_win"] is True
    assert cs["search_layer"]["bespoke_to_one_game"] is True
    assert cs["search_layer"]["heuristic_used"] == "coded_unmet_targets_plus_manhattan_progress"
    # 2. exp4022 flagged-skipped.
    assert cs["decentralization"]["flagged_adversarial"] is True
    assert cs["decentralization"]["skipped"] is True
    assert cs["decentralization"]["unresolved"] is True
    # 3. total_games_solved == 6.
    assert cs["arc3"]["total_games_solved"] == 6
    assert cs["arc3"]["monotonic_plus_one"] is True
    # 4. Efficiency clean win (95.3x cheaper wall-clock).
    assert cs["efficiency"]["wallclock_seconds_ratio_judge_over_verifier"] == pytest.approx(95.2564)
    assert cs["efficiency"]["clean_win"] is True
    # Self-learning + selection retirement.
    assert cs["arcmemo_transfer"]["actions_cold"] == 71
    assert cs["arcmemo_transfer"]["actions_seeded"] == 21
    assert cs["selection_retirement"]["safety_gate_kept"] is True
    assert cs["hardware"]["per_board_reachability"]["kv260"] is True
    assert "THIN" in cs["headline"]

    # Operator-curated / conductor-reconciled files untouched.
    assert (tmp_path / "ops" / "exclusion_manifest.yaml").read_text(encoding="utf-8") == before["manifest"]
    assert (tmp_path / "research-roadmap.yaml").read_text(encoding="utf-8") == before["roadmap"]
    assert (tmp_path / "ops" / "changelog.md").read_text(encoding="utf-8") == before["changelog"]
    assert (tmp_path / "ops" / "status.md").read_text(encoding="utf-8") == before["status"]
    assert (tmp_path / "_bmad" / "traceability.md").read_text(encoding="utf-8") == before["trace"]
    assert (tmp_path / "scripts" / "research_conductor.py").read_text(encoding="utf-8") == before["conductor"]


def test_req_report_4029_dedupe_is_idempotent(tmp_path: Path) -> None:
    """REQ-REPORT-4029: rerunning leaves exactly one .372 record (unchanged)."""

    _seed_repo(tmp_path, n_372_records=22)
    _run_success(tmp_path)
    first_complete = (tmp_path / "research-complete.yaml").read_text(encoding="utf-8")
    second = json.loads(_run_success(tmp_path).read_text(encoding="utf-8"))

    assert (tmp_path / "research-complete.yaml").read_text(encoding="utf-8") == first_complete
    assert second["research_complete_record_action"] == "unchanged"
    assert second["research_complete_duplicates_removed"] == 0


def test_req_report_4029_appends_when_record_absent(tmp_path: Path) -> None:
    """REQ-REPORT-4029: a missing .372 record is appended canonically."""

    _seed_repo(tmp_path, n_372_records=0)
    artifact = json.loads(_run_success(tmp_path).read_text(encoding="utf-8"))

    assert artifact["research_complete_record_action"] == "appended"
    loaded = yaml.safe_load((tmp_path / "research-complete.yaml").read_text(encoding="utf-8"))
    ids = [m["id"] for m in loaded["milestones"]]
    assert ids.count("2026.06.372") == 1
    record = next(m for m in loaded["milestones"] if m["id"] == "2026.06.372")
    assert record["activation_recorded"] == "exp4029-archive-v372-activate-v373"
    assert len(record["tasks"]) == len(mod.V372_TASKS)


# --------------------------------------------------------------------------- #
# Blocked paths
# --------------------------------------------------------------------------- #
def test_scenario_report_4029_blocked_yaml_writes_artifact_without_edits(tmp_path: Path) -> None:
    """SCENARIO-REPORT-4029-BLOCKED-YAML: corrupt YAML exits before edits."""

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
    assert artifact["milestone_372_closestate"]["status"] == "blocked"
    assert artifact["active_milestone_confirmed"] == "2026.06.373"
    assert artifact["preconditions_checked"]["research_complete_yaml_parsed_before"] is False
    assert (tmp_path / "research-complete.yaml").read_text(encoding="utf-8") == before_complete
    assert (tmp_path / "ops" / "exclusion_manifest.yaml").read_text(encoding="utf-8") == before_manifest


def test_req_report_4029_blocked_when_complete_missing(tmp_path: Path) -> None:
    """REQ-REPORT-4029: a missing research-complete.yaml fails closed."""

    _seed_repo(tmp_path)
    (tmp_path / "research-complete.yaml").unlink()
    artifact = json.loads(_run_success(tmp_path).read_text(encoding="utf-8"))
    assert artifact["honest_verdict"].startswith("blocked_research_complete_yaml_poison_missing")


def test_req_report_4029_blocked_paths_fail_closed(tmp_path: Path) -> None:
    """REQ-REPORT-4029: missing handoff facts block instead of fabricating success."""

    _seed_repo(tmp_path, milestone="2026.06.372")
    artifact = json.loads(_run_success(tmp_path).read_text(encoding="utf-8"))
    assert artifact["honest_verdict"].startswith("blocked_v373_not_active")

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


def test_req_report_4029_blocked_when_pretest_unquarantinable(tmp_path: Path) -> None:
    """REQ-REPORT-4029: a red gate with no parseable failure id blocks."""

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
    assert "per_task_status" in artifact["milestone_372_closestate"]


def test_req_report_4029_quarantines_red_test_then_green(tmp_path: Path) -> None:
    """REQ-REPORT-4029: a red smart-subset file is git-mv'd to quarantine."""

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


def test_req_report_4029_blocked_edit_invalid(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """REQ-REPORT-4029: an edit that breaks YAML blocks before writing."""

    _seed_repo(tmp_path)
    monkeypatch.setattr(mod, "dedupe_or_append_record", lambda *a: ("milestones: [\n", 0, "deduped"))
    artifact = json.loads(_run_success(tmp_path).read_text(encoding="utf-8"))
    assert artifact["honest_verdict"].startswith("blocked_research_complete_edit_invalid")
    # The original file was NOT overwritten with the broken edit.
    assert yaml.safe_load((tmp_path / "research-complete.yaml").read_text(encoding="utf-8"))


def test_req_report_4029_blocked_poison_after_edit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """REQ-REPORT-4029: a post-write parse regression is caught."""

    _seed_repo(tmp_path)
    # Calls in order: parses_before, edit-candidate, on-disk re-read, manifest.
    states = iter([True, True, False, True])

    monkeypatch.setattr(mod, "yaml_parses", lambda text: next(states))
    artifact = json.loads(_run_success(tmp_path).read_text(encoding="utf-8"))
    assert artifact["honest_verdict"].startswith("blocked_research_complete_yaml_poison_after_edit")


# --------------------------------------------------------------------------- #
# Close-state + helper unit tests
# --------------------------------------------------------------------------- #
def test_req_report_4029_classify_status_branches() -> None:
    """REQ-REPORT-4029: every status class is reachable."""

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


def test_req_report_4029_build_closestate_from_records() -> None:
    """REQ-REPORT-4029: the close-state builder is a pure aggregation."""

    records: dict[str, dict[str, object]] = {
        exp_id: _record(payload) for exp_id, payload in _v372_artifacts().items()
    }
    closestate = mod.build_closestate(records)

    assert closestate["status_counts"]["OK"] == 9
    assert closestate["status_counts"]["FLAGGED"] == 1
    assert closestate["search_layer"]["thin_win"] is True
    assert closestate["decentralization"]["skipped"] is True
    assert closestate["arc3"]["total_games_solved"] == 6
    assert closestate["efficiency"]["clean_win"] is True
    assert closestate["capstone_v372_verdict"].startswith("success: capstone_v372")
    assert "THIN" in closestate["headline"]


def test_req_report_4029_closestate_subbuilders_degrade_gracefully() -> None:
    """REQ-REPORT-4029: every sub-builder returns null facts on missing input."""

    empty = {"exists": False}
    planning = mod._planning_result(empty, empty)
    assert planning["nodes_expanded"] is None
    assert planning["thin_win"] is False
    assert planning["bespoke_to_one_game"] is False

    decentral = mod._decentralization_result(empty)
    assert decentral["flagged_adversarial"] is False and decentral["skipped"] is False

    selection = mod._selection_result(empty)
    assert selection["retired_r_and_d_line"] is None and selection["safety_gate_kept"] is False

    arc3 = mod._arc3_result(empty)
    assert arc3["total_games_solved"] is None and arc3["monotonic_plus_one"] is False

    efficiency = mod._efficiency_result(empty)
    assert efficiency["wallclock_seconds_ratio_judge_over_verifier"] is None
    assert efficiency["clean_win"] is False

    hardware = mod._hardware_result(empty)
    assert hardware["per_board_reachability"] == {} and hardware["included"] is False

    # _fields tolerates a non-mapping fields value.
    assert mod._fields({"fields": [1, 2, 3]}) == {}
    assert mod._fields({}) == {}


def test_req_report_4029_closestate_nonbool_and_nonmonotonic_branches() -> None:
    """REQ-REPORT-4029: thin/monotonic guards reject bools and wrong deltas."""

    # nodes_expanded as a bool must not count as a small-int thin win.
    bool_nodes = _record({"honest_verdict": "complete: x", "game": "vc33", "nodes_expanded": True})
    planning = mod._planning_result({"exists": False}, bool_nodes)
    assert planning["thin_win"] is False
    assert planning["bespoke_to_one_game"] is False

    # A two-game jump is not a monotonic +1.
    jump = _record({"honest_verdict": "success: x", "total_games_solved": 7, "prior_total_games_solved": 5})
    assert mod._arc3_result(jump)["monotonic_plus_one"] is False

    # A large nodes count is not thin.
    wide = _record({"honest_verdict": "complete: x", "game": "vc33", "nodes_expanded": 4096})
    assert mod._planning_result({"exists": False}, wide)["thin_win"] is False


def test_req_report_4029_arcmemo_and_efficiency_parsers() -> None:
    """REQ-REPORT-4029: verdict parsers match real and degrade on no-match."""

    arcmemo = mod._arcmemo_result({"honest_verdict": "success: arcmemo_v5_transfer_71to21_actions"})
    assert arcmemo["transfer_win"] is True
    assert arcmemo["actions_cold"] == 71 and arcmemo["actions_seeded"] == 21

    no_match = mod._arcmemo_result({"honest_verdict": "success: nothing_to_parse"})
    assert no_match["transfer_win"] is False
    assert no_match["actions_cold"] is None and no_match["actions_seeded"] is None

    # A flagged efficiency artifact is never a clean win.
    flagged = _record({"honest_verdict": "success: x", "flagged_adversarial": True})
    assert mod._efficiency_result(flagged)["clean_win"] is False


def test_req_report_4029_read_artifact_record_fail_closed(tmp_path: Path) -> None:
    """REQ-REPORT-4029: unreadable / non-mapping artifacts read as absent."""

    assert mod.read_artifact_record(tmp_path / "missing.json")["exists"] is False
    listish = tmp_path / "listish.json"
    listish.write_text("[1, 2, 3]", encoding="utf-8")
    assert mod.read_artifact_record(listish)["exists"] is False
    good = tmp_path / "good.json"
    good.write_text(json.dumps({"honest_verdict": "complete: ok"}), encoding="utf-8")
    assert mod.read_artifact_record(good)["exists"] is True


def test_req_report_4029_read_v372_records_reads_all_tasks(tmp_path: Path) -> None:
    """REQ-REPORT-4029: every .372 deliverable is read by exp id."""

    _seed_repo(tmp_path)
    records = mod.read_v372_records(tmp_path)
    assert set(records) == {str(t["exp_id"]) for t in mod.V372_TASKS}
    assert records["4022"]["flagged_adversarial"] is True
    assert records["4021"]["fields"]["nodes_expanded"] == 3


def test_req_report_4029_dedupe_helper_branches() -> None:
    """REQ-REPORT-4029: dedupe/append helper covers all three actions."""

    base = "milestones:\n- id: 2026.06.371\n  title: a\n"
    two = base + _v372_record_block(0) + _v372_record_block(1)
    deduped, removed, action = mod.dedupe_or_append_record(two, "2026.06.372")
    assert action == "deduped" and removed == 1
    assert deduped.count("- id: 2026.06.372") == 1

    one = base + _v372_record_block(0)
    unchanged, removed, action = mod.dedupe_or_append_record(one, "2026.06.372")
    assert action == "unchanged" and removed == 0 and unchanged == one

    appended, removed, action = mod.dedupe_or_append_record(base, "2026.06.372")
    assert action == "appended" and removed == 0
    assert appended.count("- id: 2026.06.372") == 1
    assert yaml.safe_load(appended)


def test_req_report_4029_smart_subset_targets_and_command(tmp_path: Path) -> None:
    """REQ-REPORT-4029: smart subset = core suites + uncommitted tests, no live git."""

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


def test_req_report_4029_parse_failing_ids_and_quarantine(tmp_path: Path) -> None:
    """REQ-REPORT-4029: failing-id parsing and quarantine fallback paths."""

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


def test_req_report_4029_run_pretest_until_green_caps(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """REQ-REPORT-4029: the quarantine loop is bounded and live-callable."""

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


def test_req_report_4029_misc_helpers(tmp_path: Path) -> None:
    """REQ-REPORT-4029: small pure helpers behave and fail closed."""

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
    assert mod._record_id("- id: 2026.06.372") == "2026.06.372"

    # _run_command OSError -> exit 127.
    res = mod._run_command(["definitely-not-a-real-binary-xyz"], tmp_path)
    assert res.exit_code in {127}
    # _git_lines returns [] when git command fails.
    assert mod._git_lines(["rev-parse", "--bad-flag"], tmp_path) == []


def test_req_report_4029_read_active_milestone_next_roadmap(tmp_path: Path) -> None:
    """REQ-REPORT-4029: the -next roadmap is the fallback milestone source."""

    (tmp_path / "research-roadmap-next.yaml").write_text('milestone: "2026.06.373"\n', encoding="utf-8")
    assert mod.read_active_milestone(tmp_path) == ("2026.06.373", "research-roadmap-next.yaml")


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
        (lambda p: p.update(archived_milestone="2026.06.371"), "archived milestone"),
        (lambda p: p.update(activated_milestone="2026.06.372"), "activated milestone"),
        (lambda p: p.update(research_complete_yaml_parses=False), "research-complete"),
        (lambda p: p.update(exclusion_manifest_parses=False), "manifest"),
        (lambda p: p.update(arc_modules_importable=False), "ARC module imports"),
        (lambda p: p.update(pretest_suite_green=False), "pretest suite"),
        (lambda p: p.update(active_milestone_confirmed=False), "active milestone"),
        (lambda p: p.update(n_tasks_archived=3), "n_tasks_archived"),
        (lambda p: p.update(milestone_372_closestate={}), "non-empty dict"),
        (lambda p: p.update(milestone_372_closestate={"x": 1}), "per_task_status"),
        (lambda p: p.update(duration_s=0.0), "duration_s"),
        (lambda p: p.update(inference_substrate="live_model"), "inference"),
        (lambda p: p.update(quarantined_tests={}), "quarantined_tests"),
        (lambda p: p.update(model_specs={}), "model_specs"),
        (lambda p: p.update(copied_marker="CUDA"), "compute-bound markers"),
        (lambda p: p.update(reproducibility_checksum="short"), "sha256"),
        (lambda p: p.update(reproducibility_checksum="0" * 64), "does not match"),
    ],
)
def test_req_report_4029_validate_rejects_regressions(tmp_path: Path, mutate, message: str) -> None:
    """REQ-REPORT-4029: validation rejects fields that hide transition risk."""

    _seed_repo(tmp_path)
    payload = json.loads(_run_success(tmp_path).read_text(encoding="utf-8"))
    broken = json.loads(json.dumps(payload))
    mutate(broken)
    with pytest.raises(ValueError, match=message):
        mod.validate_artifact(broken)


def test_req_report_4029_terminal_verdict_carries_games() -> None:
    """REQ-REPORT-4029: the terminal verdict embeds the games-solved total."""

    verdict = mod.terminal_verdict({"arc3": {"total_games_solved": 6}})
    assert verdict.startswith("success:")
    assert "games6" in verdict


def test_req_report_4029_smart_subset_with_git(tmp_path: Path) -> None:
    """REQ-REPORT-4029: untracked tests/python files join the smart subset."""

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


def test_req_report_4029_run_smart_subset_uses_command(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """REQ-REPORT-4029: the live gate runs the smart-subset command (no spawn)."""

    captured: dict[str, object] = {}

    def fake_run(command: list[str], root: Path) -> mod.CommandResult:
        captured["command"] = command
        return mod.CommandResult(command=command, exit_code=0, stdout="ok", stderr="")

    monkeypatch.setattr(mod, "_run_command", fake_run)
    result = mod.run_smart_subset(tmp_path)
    assert result.exit_code == 0
    assert str(mod.PYTEST_BIN) == captured["command"][0]
    assert "--no-cov" in captured["command"]


def test_scenario_report_4029_script_wrapper_exists() -> None:
    """SCENARIO-REPORT-4029: the requested experiment entrypoint exists."""

    script = Path("scripts/experiments/exp4029_archive_v372_activate_v373.py")
    assert script.exists()
    assert "archive_v372_activate_v373_4029" in script.read_text(encoding="utf-8")
