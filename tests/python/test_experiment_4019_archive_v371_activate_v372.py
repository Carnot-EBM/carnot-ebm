"""Tests for Exp 4019 .371 archive and .372 activation.

Spec refs: REQ-REPORT-4019, SCENARIO-REPORT-4019,
SCENARIO-REPORT-4019-BLOCKED-YAML.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from carnot.reporting import archive_v371_activate_v372_4019 as mod


SPEC_PATH = Path("openspec/capabilities/research-reporting/spec.md")


# --------------------------------------------------------------------------- #
# Fixtures: a tmp repo with the .371 artifacts and a duplicated .371 record.
# --------------------------------------------------------------------------- #
def _v371_artifacts(*, drop_4011: bool = True) -> dict[str, dict[str, object]]:
    """Synthetic .371 task artifacts mirroring the real verdicts."""

    artifacts: dict[str, dict[str, object]] = {
        "4008": {"honest_verdict": "blocked_pretest_suite_failed_unquarantined"},
        "4009": {
            "honest_verdict": "blocked_execution_floor_unmet",
            "total_codex_calls": 0,
            "n_agreement_events": 0,
        },
        "4010": {"honest_verdict": "complete: gap5_cross_example_no_better_than_agreement_coverage_lower"},
        "4011": {"honest_verdict": "complete: feedback_vs_redraw_v2"},
        "4012": {
            "honest_verdict": "complete: gap4_local_bestofn_cov0.2581_pass20.4516_below_codex",
            "local_beats_vote": False,
        },
        "4013": {
            "honest_verdict": "success: verifier_parity_at_95.3x_cheaper_than_judge",
            "flagged_adversarial": True,
        },
        "4014": {"honest_verdict": "complete: level_walls_hold_r11l_L4_total5"},
        "4015": {"honest_verdict": "success: fifth_game_solved_tn36-ef4dde99_at_action7"},
        "4016": {"honest_verdict": "success: arcmemo_solve_transfer_v4_11to7_actions"},
        "4017": {"honest_verdict": "complete: hardware_continuity_4017_ssh_continuity_recorded"},
        "4018": {
            "honest_verdict": (
                "success: capstone_v371_gap4_UNCONFIRMED_NOT_DECENTRALIZATION_EFFECTIVE_"
                "games5_levels5_arcmemo_transfer_win_pretest_not_green_missing1_flagged_skipped1"
            ),
            "total_games_solved": 5,
            "total_levels_solved": 5,
        },
    }
    if drop_4011:
        del artifacts["4011"]
    return artifacts


def _v371_record_block(idx: int) -> str:
    return (
        "- id: 2026.06.371\n"
        f"  title: 'CONFIRM the GAP-4 verifier moat (copy {idx})'\n"
        "  doc: openspec/change-proposals/research-roadmap-vNEXT.md\n"
        "  completed: '2026-06-11'\n"
        "  finding: See conductor log for per-experiment results.\n"
        "  tasks:\n"
        "  - id: exp4009-gap4-precision-confirmation-v3\n"
        "    result: OK (conductor)\n"
    )


def _seed_repo(
    root: Path,
    *,
    corrupt_complete: bool = False,
    milestone: str = "2026.06.372",
    manifest: str = "retired: []\n",
    n_371_records: int = 3,
    drop_4011: bool = True,
) -> None:
    (root / "research-roadmap.yaml").write_text(
        f'milestone: "{milestone}"\ntasks:\n  - id: exp4019-archive-v371-activate-v372\n',
        encoding="utf-8",
    )
    complete_text = (
        "milestones:\n"
        "- id: 2026.06.370\n"
        "  title: prior archive\n"
        "  completed: '2026-06-10'\n"
        "  tasks:\n"
        "  - id: exp3997-archive\n"
        "    result: OK (conductor)\n"
    )
    for idx in range(n_371_records):
        complete_text += _v371_record_block(idx)
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
    for task in mod.V371_TASKS:
        exp_id = str(task["exp_id"])
        payloads = _v371_artifacts(drop_4011=drop_4011)
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
def test_req_report_4019_spec_anchor_exists() -> None:
    """REQ-REPORT-4019: OpenSpec declares the .371 archive contract."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    assert "REQ-REPORT-4019" in spec
    assert "SCENARIO-REPORT-4019" in spec
    assert "SCENARIO-REPORT-4019-BLOCKED-YAML" in spec
    assert mod.OUTPUT_REL_PATH.as_posix() in spec
    assert "milestone_371_closestate" in spec
    assert "smart-subset" in spec


# --------------------------------------------------------------------------- #
# Complete path
# --------------------------------------------------------------------------- #
def test_scenario_report_4019_dedupes_and_records_closestate(tmp_path: Path) -> None:
    """SCENARIO-REPORT-4019: collapse duplicate .371 records and record truth."""

    _seed_repo(tmp_path, n_371_records=3)
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
    assert artifact["honest_verdict"].startswith("complete:")
    assert artifact["archived_milestone"] == "2026.06.371"
    assert artifact["activated_milestone"] == "2026.06.372"
    assert artifact["research_complete_yaml_parses"] is True
    assert artifact["exclusion_manifest_parses"] is True
    assert artifact["arc_modules_importable"] is True
    assert artifact["pretest_suite_green"] is True
    assert artifact["quarantined_tests"] == []
    assert artifact["inference_substrate"] == "aggregation_from_upstream_artifacts"
    assert artifact["research_complete_record_action"] == "deduped"
    assert artifact["research_complete_duplicates_removed"] == 2

    # research-complete now has exactly one .371 record and still parses.
    complete_text = (tmp_path / "research-complete.yaml").read_text(encoding="utf-8")
    loaded = yaml.safe_load(complete_text)
    ids = [m["id"] for m in loaded["milestones"]]
    assert ids.count("2026.06.371") == 1

    # Close-state truth.
    cs = artifact["milestone_371_closestate"]
    assert cs["per_task_status"]["exp4009-gap4-precision-confirmation-v3"] == "BLOCKED"
    assert cs["per_task_status"]["exp4011-gap4-feedback-vs-redraw-v2"] == "MISSING"
    assert cs["per_task_status"]["exp4013-verifier-vs-judge-efficiency"] == "FLAGGED"
    assert cs["per_task_status"]["exp4015-fifth-game-explore-first"] == "OK"
    assert cs["gap4_followups"]["executed"] == ["exp4010", "exp4012"]
    assert cs["gap4_followups"]["no_artifact_wall_clock_cap_or_skipped"] == ["exp4011"]
    assert cs["gap4_followups"]["blocked_execution_floor"] == ["exp4009"]
    assert cs["gap4_followups"]["precision_confirmation_still_owed"] is True
    assert cs["exp4012_local_best_of_n"]["coverage"] == pytest.approx(0.2581)
    assert cs["exp4012_local_best_of_n"]["pass_at_2"] == pytest.approx(0.4516)
    assert cs["exp4012_local_best_of_n"]["beats_codex"] is False
    assert cs["decentralization_effective"] is False
    assert cs["efficiency_exp4013"]["flagged_adversarial"] is True
    assert cs["efficiency_exp4013"]["skipped"] is True
    assert cs["arc3"]["total_games_solved"] == 5
    assert cs["arc3"]["total_levels_solved"] == 5
    assert cs["arc3"]["fifth_game_solved"] == "tn36-ef4dde99"
    assert cs["arc3"]["fifth_game_actions"] == 7
    assert cs["arcmemo_transfer"]["transfer_win"] is True
    assert cs["arcmemo_transfer"]["actions_cold"] == 11
    assert cs["arcmemo_transfer"]["actions_warm"] == 7
    assert "UNCONFIRMED" in cs["headline"]

    # Operator-curated / conductor-reconciled files untouched.
    assert (tmp_path / "ops" / "exclusion_manifest.yaml").read_text(encoding="utf-8") == before["manifest"]
    assert (tmp_path / "research-roadmap.yaml").read_text(encoding="utf-8") == before["roadmap"]
    assert (tmp_path / "ops" / "changelog.md").read_text(encoding="utf-8") == before["changelog"]
    assert (tmp_path / "ops" / "status.md").read_text(encoding="utf-8") == before["status"]
    assert (tmp_path / "_bmad" / "traceability.md").read_text(encoding="utf-8") == before["trace"]
    assert (tmp_path / "scripts" / "research_conductor.py").read_text(encoding="utf-8") == before["conductor"]


def test_req_report_4019_dedupe_is_idempotent(tmp_path: Path) -> None:
    """REQ-REPORT-4019: rerunning leaves exactly one .371 record (unchanged)."""

    _seed_repo(tmp_path, n_371_records=3)
    _run_success(tmp_path)
    first_complete = (tmp_path / "research-complete.yaml").read_text(encoding="utf-8")
    second = json.loads(_run_success(tmp_path).read_text(encoding="utf-8"))

    assert (tmp_path / "research-complete.yaml").read_text(encoding="utf-8") == first_complete
    assert second["research_complete_record_action"] == "unchanged"
    assert second["research_complete_duplicates_removed"] == 0


def test_req_report_4019_appends_when_record_absent(tmp_path: Path) -> None:
    """REQ-REPORT-4019: a missing .371 record is appended canonically."""

    _seed_repo(tmp_path, n_371_records=0)
    artifact = json.loads(_run_success(tmp_path).read_text(encoding="utf-8"))

    assert artifact["research_complete_record_action"] == "appended"
    loaded = yaml.safe_load((tmp_path / "research-complete.yaml").read_text(encoding="utf-8"))
    ids = [m["id"] for m in loaded["milestones"]]
    assert ids.count("2026.06.371") == 1
    record = next(m for m in loaded["milestones"] if m["id"] == "2026.06.371")
    assert record["activation_recorded"] == "exp4019-archive-v371-activate-v372"
    assert len(record["tasks"]) == len(mod.V371_TASKS)


# --------------------------------------------------------------------------- #
# Blocked paths
# --------------------------------------------------------------------------- #
def test_scenario_report_4019_blocked_yaml_writes_artifact_without_edits(tmp_path: Path) -> None:
    """SCENARIO-REPORT-4019-BLOCKED-YAML: corrupt YAML exits before edits."""

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
    assert artifact["milestone_371_closestate"]["status"] == "blocked"
    assert artifact["preconditions_checked"]["research_complete_yaml_parsed_before"] is False
    assert (tmp_path / "research-complete.yaml").read_text(encoding="utf-8") == before_complete
    assert (tmp_path / "ops" / "exclusion_manifest.yaml").read_text(encoding="utf-8") == before_manifest


def test_req_report_4019_blocked_when_complete_missing(tmp_path: Path) -> None:
    """REQ-REPORT-4019: a missing research-complete.yaml fails closed."""

    _seed_repo(tmp_path)
    (tmp_path / "research-complete.yaml").unlink()
    artifact = json.loads(_run_success(tmp_path).read_text(encoding="utf-8"))
    assert artifact["honest_verdict"].startswith("blocked_research_complete_yaml_poison_missing")


def test_req_report_4019_blocked_paths_fail_closed(tmp_path: Path) -> None:
    """REQ-REPORT-4019: missing handoff facts block instead of fabricating success."""

    _seed_repo(tmp_path, milestone="2026.06.371")
    artifact = json.loads(_run_success(tmp_path).read_text(encoding="utf-8"))
    assert artifact["honest_verdict"].startswith("blocked_v372_not_active")

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


def test_req_report_4019_blocked_when_pretest_unquarantinable(tmp_path: Path) -> None:
    """REQ-REPORT-4019: a red gate with no parseable failure id blocks."""

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
    assert "per_task_status" in artifact["milestone_371_closestate"]


def test_req_report_4019_quarantines_red_test_then_green(tmp_path: Path) -> None:
    """REQ-REPORT-4019: a red smart-subset file is git-mv'd to quarantine."""

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


def test_req_report_4019_blocked_edit_invalid(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """REQ-REPORT-4019: an edit that breaks YAML blocks before writing."""

    _seed_repo(tmp_path)
    monkeypatch.setattr(mod, "dedupe_or_append_record", lambda *a: ("milestones: [\n", 0, "deduped"))
    artifact = json.loads(_run_success(tmp_path).read_text(encoding="utf-8"))
    assert artifact["honest_verdict"].startswith("blocked_research_complete_edit_invalid")
    # The original file was NOT overwritten with the broken edit.
    assert yaml.safe_load((tmp_path / "research-complete.yaml").read_text(encoding="utf-8"))


def test_req_report_4019_blocked_poison_after_edit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """REQ-REPORT-4019: a post-write parse regression is caught."""

    _seed_repo(tmp_path)
    # Calls in order: parses_before, edit-candidate, on-disk re-read, manifest.
    states = iter([True, True, False, True])

    monkeypatch.setattr(mod, "yaml_parses", lambda text: next(states))
    artifact = json.loads(_run_success(tmp_path).read_text(encoding="utf-8"))
    assert artifact["honest_verdict"].startswith("blocked_research_complete_yaml_poison_after_edit")


# --------------------------------------------------------------------------- #
# Close-state + helper unit tests
# --------------------------------------------------------------------------- #
def test_req_report_4019_classify_status_branches() -> None:
    """REQ-REPORT-4019: every status class is reachable."""

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


def test_req_report_4019_build_closestate_from_records() -> None:
    """REQ-REPORT-4019: the close-state builder is a pure aggregation."""

    records: dict[str, dict[str, object]] = {
        exp_id: _record(payload) for exp_id, payload in _v371_artifacts(drop_4011=False).items()
    }
    records["4011"] = {"exists": False}
    closestate = mod.build_closestate(records)

    assert closestate["status_counts"]["OK"] == 7
    assert closestate["status_counts"]["BLOCKED"] == 2  # exp4008 + exp4009
    assert closestate["status_counts"]["MISSING"] == 1
    assert closestate["status_counts"]["FLAGGED"] == 1
    assert closestate["gap4_confirmed"] is False
    assert closestate["arc3"]["total_games_solved"] == 5
    # exp4012 with no parseable cov string yields None gracefully.
    no_cov = dict(records)
    no_cov["4012"] = {"exists": True, "honest_verdict": "complete: nothing_parseable"}
    relaxed = mod.build_closestate(no_cov)
    assert relaxed["exp4012_local_best_of_n"]["coverage"] is None
    assert relaxed["exp4012_local_best_of_n"]["beats_codex"] is True  # no "below_codex" token


def test_req_report_4019_read_artifact_record_fail_closed(tmp_path: Path) -> None:
    """REQ-REPORT-4019: unreadable / non-mapping artifacts read as absent."""

    assert mod.read_artifact_record(tmp_path / "missing.json")["exists"] is False
    listish = tmp_path / "listish.json"
    listish.write_text("[1, 2, 3]", encoding="utf-8")
    assert mod.read_artifact_record(listish)["exists"] is False
    good = tmp_path / "good.json"
    good.write_text(json.dumps({"honest_verdict": "complete: ok"}), encoding="utf-8")
    assert mod.read_artifact_record(good)["exists"] is True


def test_req_report_4019_dedupe_helper_branches() -> None:
    """REQ-REPORT-4019: dedupe/append helper covers all three actions."""

    base = "milestones:\n- id: 2026.06.370\n  title: a\n"
    two = base + _v371_record_block(0) + _v371_record_block(1)
    deduped, removed, action = mod.dedupe_or_append_record(two, "2026.06.371")
    assert action == "deduped" and removed == 1
    assert deduped.count("- id: 2026.06.371") == 1

    one = base + _v371_record_block(0)
    unchanged, removed, action = mod.dedupe_or_append_record(one, "2026.06.371")
    assert action == "unchanged" and removed == 0 and unchanged == one

    appended, removed, action = mod.dedupe_or_append_record(base, "2026.06.371")
    assert action == "appended" and removed == 0
    assert appended.count("- id: 2026.06.371") == 1
    assert yaml.safe_load(appended)


def test_req_report_4019_smart_subset_targets_and_command(tmp_path: Path) -> None:
    """REQ-REPORT-4019: smart subset = core suites + uncommitted tests, no live git."""

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


def test_req_report_4019_parse_failing_ids_and_quarantine(tmp_path: Path) -> None:
    """REQ-REPORT-4019: failing-id parsing and quarantine fallback paths."""

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


def test_req_report_4019_run_pretest_until_green_caps(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """REQ-REPORT-4019: the quarantine loop is bounded and live-callable."""

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


def test_req_report_4019_misc_helpers(tmp_path: Path) -> None:
    """REQ-REPORT-4019: small pure helpers behave and fail closed."""

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
    assert mod._record_id("- id: 2026.06.371") == "2026.06.371"

    # _run_command OSError -> exit 127.
    res = mod._run_command(["definitely-not-a-real-binary-xyz"], tmp_path)
    assert res.exit_code in {127}
    # _git_lines returns [] when git command fails.
    assert mod._git_lines(["rev-parse", "--bad-flag"], tmp_path) == []


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
        (lambda p: p.update(archived_milestone="2026.06.370"), "archived milestone"),
        (lambda p: p.update(activated_milestone="2026.06.371"), "activated milestone"),
        (lambda p: p.update(research_complete_yaml_parses=False), "research-complete"),
        (lambda p: p.update(exclusion_manifest_parses=False), "manifest"),
        (lambda p: p.update(arc_modules_importable=False), "ARC module imports"),
        (lambda p: p.update(pretest_suite_green=False), "pretest suite"),
        (lambda p: p.update(active_milestone_confirmed=False), "active milestone"),
        (lambda p: p.update(n_tasks_archived=3), "n_tasks_archived"),
        (lambda p: p.update(milestone_371_closestate={}), "non-empty dict"),
        (lambda p: p.update(milestone_371_closestate={"x": 1}), "per_task_status"),
        (lambda p: p.update(duration_s=0.0), "duration_s"),
        (lambda p: p.update(inference_substrate="live_model"), "inference"),
        (lambda p: p.update(quarantined_tests={}), "quarantined_tests"),
        (lambda p: p.update(model_specs={}), "model_specs"),
        (lambda p: p.update(copied_marker="CUDA"), "compute-bound markers"),
        (lambda p: p.update(reproducibility_checksum="short"), "sha256"),
        (lambda p: p.update(reproducibility_checksum="0" * 64), "does not match"),
    ],
)
def test_req_report_4019_validate_rejects_regressions(tmp_path: Path, mutate, message: str) -> None:
    """REQ-REPORT-4019: validation rejects fields that hide transition risk."""

    _seed_repo(tmp_path)
    payload = json.loads(_run_success(tmp_path).read_text(encoding="utf-8"))
    broken = json.loads(json.dumps(payload))
    mutate(broken)
    with pytest.raises(ValueError, match=message):
        mod.validate_artifact(broken)


def test_req_report_4019_parser_no_match_branches() -> None:
    """REQ-REPORT-4019: verdict parsers degrade gracefully on no-match."""

    arcmemo = mod._arcmemo_result({"exists": True, "honest_verdict": "success: nothing_to_parse"})
    assert arcmemo["transfer_win"] is False
    assert arcmemo["actions_cold"] is None and arcmemo["actions_warm"] is None

    arc3 = mod._arc3_result({"fields": {}}, {"honest_verdict": "success: no_fifth_here"})
    assert arc3["total_games_solved"] is None
    assert arc3["fifth_game_solved"] is None and arc3["fifth_game_actions"] is None


def test_req_report_4019_smart_subset_with_git(tmp_path: Path) -> None:
    """REQ-REPORT-4019: untracked tests/python files join the smart subset."""

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


def test_req_report_4019_run_smart_subset_uses_command(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """REQ-REPORT-4019: the live gate runs the smart-subset command (no spawn)."""

    captured: dict[str, object] = {}

    def fake_run(command: list[str], root: Path) -> mod.CommandResult:
        captured["command"] = command
        return mod.CommandResult(command=command, exit_code=0, stdout="ok", stderr="")

    monkeypatch.setattr(mod, "_run_command", fake_run)
    result = mod.run_smart_subset(tmp_path)
    assert result.exit_code == 0
    assert str(mod.PYTEST_BIN) == captured["command"][0]
    assert "--no-cov" in captured["command"]


def test_scenario_report_4019_script_wrapper_exists() -> None:
    """SCENARIO-REPORT-4019: the requested experiment entrypoint exists."""

    script = Path("scripts/experiments/experiment_4019_archive_v371_activate_v372.py")
    assert script.exists()
    assert "archive_v371_activate_v372_4019" in script.read_text(encoding="utf-8")
