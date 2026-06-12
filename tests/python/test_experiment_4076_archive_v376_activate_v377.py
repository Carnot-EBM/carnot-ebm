"""Tests for the Exp 4076 .376 archive / .377 activation record-only module.

Spec refs: REQ-REPORT-4076, SCENARIO-REPORT-4076,
SCENARIO-REPORT-4076-BLOCKED-YAML.

These tests exercise the disciplined milestone-transition module end to end on
a synthetic repo fixture (no live model, no real conductor), plus every pure
helper and every blocked-path branch. The load-bearing assertions:

* the `.376 close-state is recorded as MECHANISM-FIX-WORKED (inverting `.375's
  launch failure): G1 off-ARC produced a measurement (accumulated_n=160) but is
  flagged + uninformative (corpus saturated) so it is skipped from aggregation;
  G3 sovereign MoE base reached its N=30 floor and is ABSENT (leash holds);
  ACCURACY advanced 8->9 (ninth game solved); the action-pruner efficiency win;
* the colon-poison guard keeps research-complete.yaml parseable;
* a red OR collection-error pre-test gate is quarantined to green;
* the terminal artifact carries every required principle-annotated field.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from carnot.reporting import archive_v376_activate_v377_4076 as mod


# --------------------------------------------------------------------------- #
# Fixture: a synthetic repo with a valid .376 close-state
# --------------------------------------------------------------------------- #
GREEN = mod.CommandResult(command=["pytest"], exit_code=0, stdout="", stderr="")
PARSE_OK = mod.CommandResult(command=["yaml"], exit_code=0, stdout="", stderr="")
IMPORT_OK = mod.CommandResult(command=["import"], exit_code=0, stdout="", stderr="")


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _research_complete_text() -> str:
    """A minimal research-complete.yaml that already holds a .376 record.

    Mirrors the real repo: the conductor activation commit appended exactly one
    canonical 2026.06.376 record, so the common action for this module is
    ``unchanged``.
    """

    return (
        "- id: 2026.06.375\n"
        "  finding: prior milestone\n"
        "- id: 2026.06.376\n"
        "  title: 'FIX the powering mechanism (resume-accumulate)'\n"
        "  completed: '2026-06-12'\n"
        "  finding: See conductor log for per-experiment results.\n"
        "  tasks:\n"
        "  - id: exp4066-archive-v375-activate-v376\n"
        "    result: OK (conductor)\n"
    )


def make_repo(tmp_path: Path) -> Path:
    """Build a synthetic repo mirroring the real .376 artifacts."""

    root = tmp_path
    (root / "research-complete.yaml").write_text(_research_complete_text(), encoding="utf-8")
    (root / "ops").mkdir(parents=True, exist_ok=True)
    (root / "ops" / "exclusion_manifest.yaml").write_text(
        "retired:\n- experiment_id: 2091\n  reason: gemini bail-out\n", encoding="utf-8"
    )
    (root / "research-roadmap.yaml").write_text(
        "milestone: 2026.06.377\nname: v377\n", encoding="utf-8"
    )

    results = root / "results"
    # OK terminal JSON artifacts.
    for exp, name in (
        ("4066", "experiment_4066_archive_v375_activate_v376.json"),
        ("4073", "experiment_4073_verifier_registry_and_gaps_hygiene.json"),
    ):
        _write_json(results / name, {"honest_verdict": f"complete: ok_{exp}"})
    # exp4067 doc deliverable (a .md note, not JSON).
    doc = root / "docs" / "research-notes" / "sota-ingestion-2026-06-11-v376-unsaturated-corpora-and-online-pruning.md"
    doc.parent.mkdir(parents=True, exist_ok=True)
    doc.write_text("# SOTA ingestion v376\n", encoding="utf-8")
    # exp4068 FLAGGED off-ARC: mechanism ran (N=160) but corpus saturated -> no headroom.
    _write_json(
        results / "experiment_4068_offarc_transfer_power_sync.json",
        {
            "honest_verdict": "complete: offarc_transfer_no_oracle_headroom_evalplus_n160",
            "flagged_adversarial": True,
            "accumulated_n_tasks": 160,
            "best_arm": "armC_symbolic",
            "oracle_headroom_present": False,
            "demofit_ci_excludes_zero": False,
        },
    )
    # exp4069 MoE: reached the N=30 floor; sovereign base ABSENT (delta < 0).
    _write_json(
        results / "experiment_4069_decentralization_moe_sync.json",
        {
            "honest_verdict": "complete: decentralization_moe_cov_0.2333_absent_leash_holds_n30",
            "accumulated_n_tasks": 30,
            "target_n_tasks": 30,
            "moe_base_demo_perfect_coverage": 0.2333,
            "coverage_delta_vs_12b": -0.0248,
            "oracle_coverage": 0.6129,
        },
    )
    # exp4070 ninth game SOLVED -> total_games_solved=9.
    _write_json(
        results / "experiment_4070_ninth_game_explore_first.json",
        {
            "honest_verdict": "success: ninth_game_solved_ft09-0d8bbf25_at_action_4",
            "total_games_solved": 9,
            "levels_completed": 1,
        },
    )
    # exp4071 efficiency: action-axis win, wallclock loss, parity held.
    _write_json(
        results / "experiment_4071_verifier_action_pruner_efficiency.json",
        {
            "honest_verdict": "success: verifier_pruner_cuts_actions_66.7pct_equal_solverate",
            "action_reduction_pct": 66.6667,
            "wallclock_reduction_pct": -199.0331,
            "solverate_baseline": 1.0,
            "solverate_pruned": 1.0,
            "solverate_parity_held": True,
        },
    )
    # exp4072 ArcMemo v9 non-result.
    _write_json(
        results / "experiment_4072_arcmemo_cross_game_transfer_v9.json",
        {
            "honest_verdict": "complete: arcmemo_v9_no_cross_game_transfer",
            "cross_game_transfer_win": False,
            "n_reused_abstractions": 0,
            "transfer_assessment": "not_cheaper_than_within_game",
        },
    )
    # exp4074 hardware: GateMate blocked, PolarFire ok, KV260 terminal.
    _write_json(
        results / "experiment_4074_hardware_continuity.json",
        {
            "honest_verdict": "complete: hardware_continuity_gatemate_blocked_polarfire_ok_kv260_terminal",
            "per_board_reachability": {"gatemate": False, "kv260": True, "polarfire": True},
            "per_board_terminal_state": {"kv260": "terminal"},
            "kv260_terminal_confirmed": True,
        },
    )
    # exp4075 capstone OK with total_games_solved=9.
    _write_json(
        results / "experiment_4075_capstone_v376.json",
        {"honest_verdict": "complete: capstone_v376", "total_games_solved": 9},
    )
    return root


def run_happy(root: Path) -> dict:
    """Run the module on a synthetic repo with all gates injected green."""

    out = mod.run(
        root,
        research_complete_parse_result=PARSE_OK,
        arc_modules_import_result=IMPORT_OK,
        pretest_suite_results=[GREEN],
        started_s=0.0,
        now_s=1.0,
    )
    return json.loads(out.read_text(encoding="utf-8"))


# --------------------------------------------------------------------------- #
# SCENARIO-REPORT-4076: the happy path
# --------------------------------------------------------------------------- #
def test_run_happy_path_writes_terminal_artifact(tmp_path: Path) -> None:
    root = make_repo(tmp_path)
    art = run_happy(root)

    assert art["archived_milestone"] == "2026.06.376"
    assert art["activated_milestone"] == "2026.06.377"
    assert art["active_milestone_confirmed"] == "2026.06.377"
    assert art["honest_verdict"].startswith("success:")
    assert art["inference_substrate"] == "aggregation_from_upstream_artifacts"
    assert art["pretest_suite_green"] is True
    assert art["total_games_solved"] == 9
    assert art["flagged_count"] == 1
    assert art["duration_s"] == 1.0
    # cited upstream provenance trail covers all 10 .376 deliverables with sha256.
    cited = art["cited_upstream_artifacts"]
    assert len(cited) == len(mod.V376_TASKS)
    assert all(mod.is_sha256(c["sha256"]) for c in cited)
    # The artifact validates against its own schema.
    mod.validate_artifact(art)


def test_run_happy_records_mechanism_fix_worked(tmp_path: Path) -> None:
    root = make_repo(tmp_path)
    cs = run_happy(root)["milestone_376_closestate"]

    assert cs["mechanism_fix_worked"] is True
    assert cs["science_decisive"] is False
    g1 = cs["g1_off_arc_transfer"]
    assert g1["accumulated_n"] == 160
    assert g1["mechanism_produced_measurement"] is True
    assert g1["flagged_adversarial"] is True
    assert g1["skipped_from_aggregation"] is True
    assert g1["informative"] is False
    assert g1["oracle_headroom_present"] is False
    assert g1["best_arm"] == "armC_symbolic"
    g3 = cs["g3_decentralization_moe_base"]
    assert g3["accumulated_n"] == 30
    assert g3["reached_floor"] is True
    assert g3["sovereign_base_status"] == "absent"
    assert g3["leash_holds"] is True
    assert g3["cascade_blocked"] is False
    assert g3["retired"] is False
    assert g3["moe_coverage"] == 0.2333
    eff = cs["efficiency_action_pruner"]
    assert eff["action_axis_win"] is True
    assert eff["wallclock_axis_win"] is False
    assert eff["solverate_parity_held"] is True
    assert cs["accuracy"]["total_games_solved"] == 9
    assert cs["accuracy"]["ninth_game_solved"] is True
    assert cs["accuracy"]["advanced_this_milestone"] is True
    assert cs["self_learning"]["cross_game_transfer_win"] is False
    assert cs["hardware"]["kv260_terminal"] is True
    assert cs["flagged_count"] == 1


def test_run_happy_per_task_status_counts(tmp_path: Path) -> None:
    root = make_repo(tmp_path)
    cs = run_happy(root)["milestone_376_closestate"]
    status = cs["per_task_status"]

    assert status["exp4067-sota-ingestion-unsaturated-corpora-and-online-pruning"] == "OK"
    assert status["exp4068-offarc-transfer-power-sync-accumulate"] == "FLAGGED"
    assert status["exp4069-decentralization-moe-sync-accumulate"] == "OK"
    assert status["exp4070-ninth-game-explore-first"] == "OK"
    # 9 OK, 1 FLAGGED (4068), nothing missing/blocked/failing.
    assert cs["status_counts"] == {"OK": 9, "BLOCKED": 0, "MISSING": 0, "FLAGGED": 1, "FAIL": 0}
    cr = cs["per_task_conductor_result"]
    assert "FLAGGED" in cr["exp4068-offarc-transfer-power-sync-accumulate"]
    assert "absent" in cr["exp4069-decentralization-moe-sync-accumulate"]


def test_run_appends_record_when_missing(tmp_path: Path) -> None:
    root = make_repo(tmp_path)
    # Remove the conductor-appended .376 record: only a .375 record remains, so
    # the module must append one canonical .376 block (exercising the write).
    (root / "research-complete.yaml").write_text(
        "- id: 2026.06.375\n  finding: prior milestone\n", encoding="utf-8"
    )
    art = run_happy(root)
    assert art["honest_verdict"].startswith("success:")
    assert art["research_complete_record_action"] == "appended"
    import yaml

    loaded = yaml.safe_load((root / "research-complete.yaml").read_text(encoding="utf-8"))
    assert any(r.get("id") == "2026.06.376" for r in loaded)


def test_run_happy_research_complete_unchanged_and_parses(tmp_path: Path) -> None:
    root = make_repo(tmp_path)
    before = (root / "research-complete.yaml").read_text(encoding="utf-8")
    art = run_happy(root)
    after = (root / "research-complete.yaml").read_text(encoding="utf-8")

    # Exactly one .376 record already exists -> action is unchanged, no edit.
    assert art["research_complete_record_action"] == "unchanged"
    assert art["research_complete_duplicates_removed"] == 0
    assert before == after
    import yaml

    assert yaml.safe_load(after) is not None


def test_run_dedupes_duplicate_record(tmp_path: Path) -> None:
    root = make_repo(tmp_path)
    # Two .376 records (interrupted-run cruft) collapse to the first occurrence.
    (root / "research-complete.yaml").write_text(
        "- id: 2026.06.376\n  finding: a\n- id: 2026.06.376\n  finding: b\n", encoding="utf-8"
    )
    art = run_happy(root)
    assert art["research_complete_record_action"] == "deduped"
    assert art["research_complete_duplicates_removed"] == 1
    text = (root / "research-complete.yaml").read_text(encoding="utf-8")
    assert text.count("- id: 2026.06.376") == 1


# --------------------------------------------------------------------------- #
# SCENARIO-REPORT-4076-BLOCKED-YAML + the other blocked branches
# --------------------------------------------------------------------------- #
def test_blocked_when_research_complete_missing(tmp_path: Path) -> None:
    root = make_repo(tmp_path)
    (root / "research-complete.yaml").unlink()
    art = run_happy(root)
    assert art["honest_verdict"] == "blocked_research_complete_yaml_poison_missing"
    assert art["preconditions_checked"]["research_complete_yaml_exists"] is False


def test_blocked_when_research_complete_unparseable(tmp_path: Path) -> None:
    root = make_repo(tmp_path)
    (root / "research-complete.yaml").write_text("bad:\n\t- tab\n", encoding="utf-8")
    art = mod.run(
        root,
        research_complete_parse_result=mod.CommandResult(command=["yaml"], exit_code=1, stdout="", stderr="boom"),
        arc_modules_import_result=IMPORT_OK,
        pretest_suite_results=[GREEN],
        started_s=0.0,
        now_s=1.0,
    )
    art = json.loads(art.read_text(encoding="utf-8"))
    assert art["honest_verdict"] == "blocked_research_complete_yaml_poison"


def test_blocked_when_v377_not_active(tmp_path: Path) -> None:
    root = make_repo(tmp_path)
    (root / "research-roadmap.yaml").write_text("milestone: 2026.06.376\n", encoding="utf-8")
    art = run_happy(root)
    assert art["honest_verdict"] == "blocked_v377_not_active"


def test_blocked_when_edit_produces_invalid_yaml(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = make_repo(tmp_path)
    monkeypatch.setattr(
        mod, "dedupe_or_append_record", lambda text, mid: ("bad:\n\t- tab\n", 0, "appended")
    )
    art = run_happy(root)
    assert art["honest_verdict"] == "blocked_research_complete_edit_invalid"


def test_blocked_when_research_complete_poison_after_edit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = make_repo(tmp_path)
    calls = {"n": 0}
    real = mod.yaml_parses

    def fake(text: str) -> bool:
        calls["n"] += 1
        # 1: parses_before, 2: new_text check -> True; 3: after-edit read -> False.
        if calls["n"] == 3:
            return False
        return real(text)

    monkeypatch.setattr(mod, "yaml_parses", fake)
    art = run_happy(root)
    assert art["honest_verdict"] == "blocked_research_complete_yaml_poison_after_edit"


def test_blocked_when_manifest_missing(tmp_path: Path) -> None:
    root = make_repo(tmp_path)
    (root / "ops" / "exclusion_manifest.yaml").unlink()
    art = run_happy(root)
    assert art["honest_verdict"] == "blocked_exclusion_manifest_missing"


def test_blocked_when_manifest_unparseable(tmp_path: Path) -> None:
    root = make_repo(tmp_path)
    (root / "ops" / "exclusion_manifest.yaml").write_text("bad:\n\t- tab\n", encoding="utf-8")
    art = run_happy(root)
    assert art["honest_verdict"] == "blocked_exclusion_manifest_yaml_poison"


def test_blocked_when_arc_modules_fail_import(tmp_path: Path) -> None:
    root = make_repo(tmp_path)
    art = mod.run(
        root,
        research_complete_parse_result=PARSE_OK,
        arc_modules_import_result=mod.CommandResult(command=["import"], exit_code=1, stdout="", stderr="ImportError"),
        pretest_suite_results=[GREEN],
        started_s=0.0,
        now_s=1.0,
    )
    art = json.loads(art.read_text(encoding="utf-8"))
    assert art["honest_verdict"] == "blocked_arc_module_import"


def test_blocked_when_pretest_red_unquarantinable(tmp_path: Path) -> None:
    root = make_repo(tmp_path)
    art = mod.run(
        root,
        research_complete_parse_result=PARSE_OK,
        arc_modules_import_result=IMPORT_OK,
        pretest_suite_results=[mod.CommandResult(command=["pytest"], exit_code=1, stdout="boom", stderr="")],
        started_s=0.0,
        now_s=1.0,
    )
    art = json.loads(art.read_text(encoding="utf-8"))
    assert art["honest_verdict"] == "blocked_pretest_suite_failed_unquarantined"
    # The close-state is still recorded so the planner sees the mechanism truth.
    assert art["milestone_376_closestate"]["mechanism_fix_worked"] is True


# --------------------------------------------------------------------------- #
# Pure helpers
# --------------------------------------------------------------------------- #
def test_yaml_parses_valid_and_invalid() -> None:
    assert mod.yaml_parses("a: 1\n") is True
    assert mod.yaml_parses("bad:\n\t- tab\n") is False


def test_yaml_single_quote_escapes_quotes() -> None:
    assert mod.yaml_single_quote("a: b") == "'a: b'"
    assert mod.yaml_single_quote("it's") == "'it''s'"


def test_duration_from_branches() -> None:
    assert mod.duration_from(None, None) == 0.0001
    assert mod.duration_from(0.0, 2.5) == 2.5
    assert mod.duration_from(5.0, 5.0) == 0.0001


def test_payload_checksum_excludes_self_and_is_stable() -> None:
    payload = {"a": 1, "reproducibility_checksum": "ignored"}
    digest = mod.payload_checksum(payload)
    assert mod.is_sha256(digest)
    assert digest == mod.payload_checksum({"a": 1, "reproducibility_checksum": "different"})


def test_is_sha256() -> None:
    assert mod.is_sha256("a" * 64) is True
    assert mod.is_sha256("A" * 64) is False
    assert mod.is_sha256("abc") is False
    assert mod.is_sha256(123) is False


def test_file_sha256(tmp_path: Path) -> None:
    p = tmp_path / "f.json"
    p.write_text("{}", encoding="utf-8")
    assert mod.is_sha256(mod.file_sha256(p))
    assert mod.file_sha256(tmp_path / "absent.json") is None


def test_no_forbidden_markers() -> None:
    assert mod.no_forbidden_markers({"x": "fine"}) is True
    assert mod.no_forbidden_markers({"x": "uses GGUF"}) is False
    # The close-state + principles are excluded (they may name models legitimately).
    assert mod.no_forbidden_markers({"milestone_376_closestate": {"m": "GGUF"}, "field_principles": {"p": "CUDA"}}) is True


def test_write_payload_roundtrips(tmp_path: Path) -> None:
    path = tmp_path / "out.json"
    mod.write_payload(path, {"b": 2, "a": 1})
    text = path.read_text(encoding="utf-8")
    assert text.endswith("\n")
    assert json.loads(text) == {"a": 1, "b": 2}


def test_milestone_from_text() -> None:
    assert mod._milestone_from_text("milestone: 2026.06.377\n") == "2026.06.377"
    assert mod._milestone_from_text("name: x\n") == "unknown"


def test_read_active_milestone_branches(tmp_path: Path) -> None:
    assert mod.read_active_milestone(tmp_path) == ("unknown", "research-roadmap.yaml")
    (tmp_path / "research-roadmap.yaml").write_text("name: x\n", encoding="utf-8")
    (tmp_path / "research-roadmap-next.yaml").write_text("milestone: 2026.06.377\n", encoding="utf-8")
    assert mod.read_active_milestone(tmp_path) == ("2026.06.377", "research-roadmap-next.yaml")
    (tmp_path / "research-roadmap.yaml").write_text("milestone: 2026.06.999\n", encoding="utf-8")
    assert mod.read_active_milestone(tmp_path) == ("2026.06.999", "research-roadmap.yaml")


def test_command_builders() -> None:
    assert mod.research_complete_yaml_command()[0].endswith("python")
    assert "yaml.safe_load" in mod.research_complete_yaml_command()[2]
    assert "importlib" in mod.arc_modules_import_command()[2]


def test_record_id() -> None:
    assert mod._record_id("- id: 2026.06.376") == "2026.06.376"
    assert mod._record_id("  not a record") is None


def test_dedupe_or_append_record_branches() -> None:
    one = "- id: 2026.06.376\n  finding: x\n"
    assert mod.dedupe_or_append_record(one, "2026.06.376") == (one, 0, "unchanged")

    none = "- id: 2026.06.375\n  finding: prior\n"
    new_text, removed, action = mod.dedupe_or_append_record(none, "2026.06.376")
    assert action == "appended" and removed == 0
    import yaml

    assert yaml.safe_load(new_text) is not None
    assert any(r.get("id") == "2026.06.376" for r in yaml.safe_load(new_text))

    dup = "- id: 2026.06.376\n  finding: a\n- id: 2026.06.376\n  finding: b\n"
    new_text, removed, action = mod.dedupe_or_append_record(dup, "2026.06.376")
    assert action == "deduped" and removed == 1
    assert new_text.count("- id: 2026.06.376") == 1


def test_build_canonical_record_parses() -> None:
    import yaml

    record = mod.build_canonical_record()
    loaded = yaml.safe_load(record)
    assert loaded[0]["id"] == "2026.06.376"
    assert len(loaded[0]["tasks"]) == len(mod.V376_TASKS)


def test_read_artifact_record_branches(tmp_path: Path) -> None:
    missing = mod.read_artifact_record(tmp_path / "absent.json")
    assert missing["exists"] is False

    good = tmp_path / "good.json"
    _write_json(good, {"honest_verdict": "complete: x", "flagged_adversarial": True})
    rec = mod.read_artifact_record(good)
    assert rec["exists"] is True and rec["flagged_adversarial"] is True

    listed = tmp_path / "list.json"
    listed.write_text("[1, 2]", encoding="utf-8")
    assert mod.read_artifact_record(listed)["exists"] is False


def test_classify_status_all_branches() -> None:
    assert mod.classify_status({"exists": False}) == "MISSING"
    assert mod.classify_status({"exists": True, "flagged_adversarial": True}) == "FLAGGED"
    assert mod.classify_status({"exists": True, "honest_verdict": "blocked_x"}) == "BLOCKED"
    assert mod.classify_status({"exists": True, "honest_verdict": "complete: x"}) == "OK"
    assert mod.classify_status({"exists": True, "honest_verdict": "weird_verdict"}) == "FAIL"
    # doc deliverable: OK when present, MISSING when absent.
    assert mod.classify_status({"exists": True, "honest_verdict": ""}, kind="doc") == "OK"
    assert mod.classify_status({"exists": False}, kind="doc") == "MISSING"
    # FLAGGED beats a doc kind too.
    assert mod.classify_status({"exists": True, "flagged_adversarial": True}, kind="doc") == "FLAGGED"


def test_fields_and_is_real_number() -> None:
    assert mod._fields({"fields": {"a": 1}}) == {"a": 1}
    assert mod._fields({"fields": "nope"}) == {}
    assert mod._is_real_number(1) is True
    assert mod._is_real_number(1.5) is True
    assert mod._is_real_number(True) is False
    assert mod._is_real_number("1") is False


def test_g1_off_arc_helper_branches() -> None:
    # Flagged, no headroom -> uninformative + skipped.
    flagged = {
        "exists": True, "flagged_adversarial": True, "honest_verdict": "complete: x",
        "fields": {"accumulated_n_tasks": 160, "oracle_headroom_present": False, "best_arm": "armC_symbolic"},
    }
    g1 = mod._g1_off_arc_mechanism_fixed(flagged)
    assert g1["accumulated_n"] == 160
    assert g1["mechanism_produced_measurement"] is True
    assert g1["skipped_from_aggregation"] is True
    assert g1["informative"] is False
    assert g1["outcome"].startswith("mechanism_fixed")

    # Clean + headroom -> informative; falls back to accumulated_n key.
    clean = {
        "exists": True, "flagged_adversarial": False, "honest_verdict": "complete: y",
        "fields": {"accumulated_n": 50, "oracle_headroom_present": True},
    }
    g1b = mod._g1_off_arc_mechanism_fixed(clean)
    assert g1b["accumulated_n"] == 50
    assert g1b["informative"] is True
    assert g1b["skipped_from_aggregation"] is False
    assert g1b["outcome"] == "off_arc_transfer_recorded"

    # Missing -> n defaults to 0, no measurement.
    g1c = mod._g1_off_arc_mechanism_fixed({"exists": False})
    assert g1c["accumulated_n"] == 0
    assert g1c["mechanism_produced_measurement"] is False


def test_g3_moe_helper_branches() -> None:
    absent = {
        "exists": True, "honest_verdict": "complete: z",
        "fields": {"accumulated_n_tasks": 30, "target_n_tasks": 30,
                   "moe_base_demo_perfect_coverage": 0.2333, "coverage_delta_vs_12b": -0.0248,
                   "oracle_coverage": 0.6129},
    }
    g3 = mod._g3_moe_measured_absent(absent)
    assert g3["reached_floor"] is True
    assert g3["sovereign_base_status"] == "absent"
    assert g3["leash_holds"] is True
    assert g3["retired"] is False

    # Positive delta -> present_or_unknown, not absent; under floor.
    present = {
        "exists": True, "honest_verdict": "complete: z",
        "fields": {"accumulated_n_tasks": 10, "target_n_tasks": 30, "coverage_delta_vs_12b": 0.05},
    }
    g3b = mod._g3_moe_measured_absent(present)
    assert g3b["reached_floor"] is False
    assert g3b["sovereign_base_status"] == "present_or_unknown"
    assert g3b["leash_holds"] is False

    # Missing fields -> Nones, not absent.
    g3c = mod._g3_moe_measured_absent({"exists": False})
    assert g3c["accumulated_n"] is None
    assert g3c["reached_floor"] is False


def test_efficiency_helper_branches() -> None:
    win = {
        "exists": True, "honest_verdict": "success: e",
        "fields": {"action_reduction_pct": 66.6667, "wallclock_reduction_pct": -199.0331,
                   "solverate_parity_held": True},
    }
    eff = mod._efficiency_action_pruner(win)
    assert eff["action_axis_win"] is True
    assert eff["wallclock_axis_win"] is False
    assert eff["efficiency_gain"] is True

    # No parity -> not an action win even with reduction.
    noparity = {"exists": True, "fields": {"action_reduction_pct": 50.0, "solverate_parity_held": False}}
    eff2 = mod._efficiency_action_pruner(noparity)
    assert eff2["action_axis_win"] is False

    # Positive wallclock reduction -> wallclock win.
    wcwin = {"exists": True, "fields": {"action_reduction_pct": 10.0, "wallclock_reduction_pct": 5.0,
                                        "solverate_parity_held": True}}
    eff3 = mod._efficiency_action_pruner(wcwin)
    assert eff3["wallclock_axis_win"] is True


def test_accuracy_helper_branches() -> None:
    acc = mod._accuracy({"exists": True, "fields": {"total_games_solved": 9}},
                        {"exists": True, "honest_verdict": "success: ninth"})
    assert acc["total_games_solved"] == 9
    assert acc["ninth_game_solved"] is True
    assert acc["advanced_this_milestone"] is True

    # Falls back to ninth-game artifact total when capstone lacks it.
    acc2 = mod._accuracy({"exists": True, "fields": {}},
                         {"exists": True, "honest_verdict": "success: ninth", "fields": {"total_games_solved": 9}})
    assert acc2["total_games_solved"] == 9

    # Both missing -> default 9; ninth not solved (no terminal verdict).
    acc3 = mod._accuracy({"exists": False}, {"exists": False})
    assert acc3["total_games_solved"] == 9
    assert acc3["ninth_game_solved"] is False


def test_self_learning_and_hardware_helpers() -> None:
    sl = mod._self_learning({"exists": True, "honest_verdict": "complete: x",
                             "fields": {"cross_game_transfer_win": False, "n_reused_abstractions": 0}})
    assert sl["cross_game_transfer_win"] is False
    hw = mod._hardware({"exists": True, "honest_verdict": "complete: hw",
                        "fields": {"per_board_reachability": {"kv260": True},
                                   "per_board_terminal_state": {"kv260": "terminal"},
                                   "kv260_terminal_confirmed": True}})
    assert hw["included"] is True and hw["kv260_terminal"] is True
    hw2 = mod._hardware({"exists": True, "honest_verdict": "complete: hw", "fields": {}})
    assert hw2["per_board_reachability"] == {}


def test_read_v376_records_and_cited(tmp_path: Path) -> None:
    root = make_repo(tmp_path)
    records = mod.read_v376_records(root)
    assert records["4068"]["flagged_adversarial"] is True
    # exp4067 is a doc deliverable -> exists True via file presence, no verdict.
    assert records["4067"]["exists"] is True
    cited = mod.build_cited_upstream(root)
    assert len(cited) == len(mod.V376_TASKS)
    assert all("experiment_id" in c and "sha256" in c for c in cited)


# --------------------------------------------------------------------------- #
# Smart-subset pre-test gate helpers
# --------------------------------------------------------------------------- #
def test_run_command_success_and_oserror(tmp_path: Path) -> None:
    ok = mod._run_command(["true"], tmp_path)
    assert ok.exit_code == 0
    err = mod._run_command(["/nonexistent/carnot/binary"], tmp_path)
    assert err.exit_code == 127


def test_git_lines_branches(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        mod, "_run_command",
        lambda cmd, root: mod.CommandResult(command=cmd, exit_code=0, stdout="a\n\nb\n", stderr=""),
    )
    assert mod._git_lines(["diff"], tmp_path) == ["a", "b"]
    monkeypatch.setattr(
        mod, "_run_command",
        lambda cmd, root: mod.CommandResult(command=cmd, exit_code=1, stdout="x", stderr=""),
    )
    assert mod._git_lines(["diff"], tmp_path) == []


def test_smart_subset_targets(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    test_dir = tmp_path / "tests" / "python"
    test_dir.mkdir(parents=True)
    (test_dir / "test_new.py").write_text("def test_x(): pass\n", encoding="utf-8")
    monkeypatch.setattr(
        mod, "_git_lines",
        lambda args, root: ["tests/python/test_new.py", "tests/quarantine/test_q.py", "src/foo.py"],
    )
    targets = mod.smart_subset_targets(tmp_path)
    assert "tests/python/test_new.py" in targets
    assert "tests/quarantine/test_q.py" not in targets
    assert "src/foo.py" not in targets


def test_smart_subset_targets_fallback(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(mod, "_git_lines", lambda args, root: [])
    assert mod.smart_subset_targets(tmp_path) == [mod.CORE_SMART_SUBSET[0]]


def test_smart_subset_command() -> None:
    cmd = mod.smart_subset_command(["tests/python/test_a.py"])
    assert cmd[0].endswith("pytest")
    assert "--no-cov" in cmd and "tests/python/test_a.py" in cmd


def test_run_smart_subset(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    captured = {}

    def fake_run(cmd, root):
        captured["cmd"] = cmd
        return mod.CommandResult(command=cmd, exit_code=0, stdout="", stderr="")

    monkeypatch.setattr(mod, "smart_subset_targets", lambda root: ["tests/python/test_a.py"])
    monkeypatch.setattr(mod, "_run_command", fake_run)
    result = mod.run_smart_subset(tmp_path)
    assert result.exit_code == 0
    assert "tests/python/test_a.py" in captured["cmd"]


def test_parse_failing_test_ids_handles_failed_and_collection_error() -> None:
    out = (
        "FAILED tests/python/test_a.py::test_one - AssertionError\n"
        "ERROR tests/python/test_b.py::test_two - ModuleNotFoundError: no module exp_missing\n"
        "FAILED tests/python/test_a.py::test_one - dup\n"
        "passed tests/python/test_c.py::ok\n"
    )
    failures = mod.parse_failing_test_ids(out)
    assert failures["tests/python/test_a.py"] == ["tests/python/test_a.py::test_one"]
    assert failures["tests/python/test_b.py"] == ["tests/python/test_b.py::test_two"]
    assert "tests/python/test_c.py" not in failures


def test_quarantine_failed_tests_moves_file(tmp_path: Path) -> None:
    src = tmp_path / "tests" / "python" / "test_bad.py"
    src.parent.mkdir(parents=True)
    src.write_text("def test_bad(): assert False\n", encoding="utf-8")
    quarantined = mod.quarantine_failed_tests(
        tmp_path, {"tests/python/test_bad.py": ["tests/python/test_bad.py::test_bad"]}
    )
    assert quarantined[0]["quarantined_path"] == "tests/quarantine/test_bad.py"
    assert not src.exists()
    assert (tmp_path / "tests" / "quarantine" / "test_bad.py").exists()


def test_pretest_at_branches(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    supplied = [GREEN]
    assert mod._pretest_at(tmp_path, supplied, 0) is GREEN
    sentinel = mod.CommandResult(command=["smart"], exit_code=0, stdout="", stderr="")
    monkeypatch.setattr(mod, "run_smart_subset", lambda root: sentinel)
    assert mod._pretest_at(tmp_path, supplied, 5) is sentinel
    assert mod._pretest_at(tmp_path, None, 0) is sentinel


def test_run_pretest_until_green_branches(tmp_path: Path) -> None:
    ok, q, results = mod.run_pretest_until_green(tmp_path, [GREEN])
    assert ok is True and q == [] and len(results) == 1

    red = mod.CommandResult(command=["pytest"], exit_code=1, stdout="boom", stderr="")
    ok, q, _ = mod.run_pretest_until_green(tmp_path, [red])
    assert ok is False and q == []

    src = tmp_path / "tests" / "python" / "test_flake.py"
    src.parent.mkdir(parents=True)
    src.write_text("def test_flake(): assert False\n", encoding="utf-8")
    red_fail = mod.CommandResult(
        command=["pytest"], exit_code=1,
        stdout="FAILED tests/python/test_flake.py::test_flake - boom", stderr="",
    )
    ok, q, _ = mod.run_pretest_until_green(tmp_path, [red_fail, GREEN])
    assert ok is True and q[0]["path"] == "tests/python/test_flake.py"


def test_run_pretest_until_green_quarantines_collection_error(tmp_path: Path) -> None:
    # The 2026-06-11 orphaned-test poison pattern: a ModuleNotFoundError shows as
    # an ERROR collection line and must be quarantined exactly like a red test.
    src = tmp_path / "tests" / "python" / "test_orphan.py"
    src.parent.mkdir(parents=True)
    src.write_text("import exp_missing_module\n", encoding="utf-8")
    err = mod.CommandResult(
        command=["pytest"], exit_code=2,
        stdout="ERROR tests/python/test_orphan.py - ModuleNotFoundError: No module named 'exp_missing_module'",
        stderr="",
    )
    ok, q, _ = mod.run_pretest_until_green(tmp_path, [err, GREEN])
    assert ok is True
    assert q[0]["path"] == "tests/python/test_orphan.py"
    assert (tmp_path / "tests" / "quarantine" / "test_orphan.py").exists()


def test_run_pretest_until_green_exhausts_iterations(tmp_path: Path) -> None:
    base = tmp_path / "tests" / "python"
    base.mkdir(parents=True)
    supplied = []
    for i in range(9):
        (base / f"test_x{i}.py").write_text("def test_x(): assert False\n", encoding="utf-8")
        supplied.append(mod.CommandResult(
            command=["pytest"], exit_code=1,
            stdout=f"FAILED tests/python/test_x{i}.py::test_x - boom", stderr="",
        ))
    ok, _, results = mod.run_pretest_until_green(tmp_path, supplied)
    assert ok is False
    assert len(results) == 8  # bounded loop


# --------------------------------------------------------------------------- #
# Artifact assembly + validation
# --------------------------------------------------------------------------- #
def test_terminal_verdict_includes_games() -> None:
    verdict = mod.terminal_verdict({"accuracy": {"total_games_solved": 9}})
    assert verdict.startswith("success:")
    assert "games9" in verdict
    assert "mechanism_fix_worked" in verdict


def _valid_artifact(tmp_path: Path) -> dict:
    return run_happy(make_repo(tmp_path))


def _revalidate(art: dict, *, recompute: bool = True) -> None:
    if recompute:
        art = dict(art)
        art.pop("reproducibility_checksum", None)
        art["reproducibility_checksum"] = mod.payload_checksum(art)
    mod.validate_artifact(art)


def test_validate_artifact_accepts_valid(tmp_path: Path) -> None:
    _revalidate(_valid_artifact(tmp_path), recompute=False)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda a: a.pop("pretest_suite_green"),
        lambda a: a.update(honest_verdict="not_a_terminal_prefix"),
        lambda a: a.update(archived_milestone="2026.06.999"),
        lambda a: a.update(activated_milestone="2026.06.999"),
        lambda a: a.update(research_complete_yaml_parses=False),
        lambda a: a.update(exclusion_manifest_parses=False),
        lambda a: a.update(arc_modules_importable=False),
        lambda a: a.update(pretest_suite_green=False),
        lambda a: a.update(active_milestone_confirmed="2026.06.376"),
        lambda a: a.update(n_tasks_archived=3),
        lambda a: a.update(total_games_solved=8),
        lambda a: a.update(flagged_count=0),
        lambda a: a.update(milestone_376_closestate={}),
        lambda a: a.update(duration_s=0),
        lambda a: a.update(inference_substrate="live_llm_inference"),
        lambda a: a.update(quarantined_tests="nope"),
        lambda a: a.update(cited_upstream_artifacts="nope"),
        lambda a: a.update(model_specs={"x": 1}),
        lambda a: a.update(active_roadmap_path="uses GGUF here"),
    ],
)
def test_validate_artifact_rejects_field_mutations(tmp_path: Path, mutate) -> None:
    art = _valid_artifact(tmp_path)
    mutate(art)
    with pytest.raises(ValueError):
        _revalidate(art)


def test_validate_artifact_rejects_bad_principles(tmp_path: Path) -> None:
    art = _valid_artifact(tmp_path)
    art["field_principles"] = "nope"
    with pytest.raises(ValueError):
        _revalidate(art)

    art = _valid_artifact(tmp_path)
    art["field_principles"] = dict(art["field_principles"])
    art["field_principles"].pop("honest_verdict")
    with pytest.raises(ValueError):
        _revalidate(art)


def test_validate_artifact_rejects_closestate_problems(tmp_path: Path) -> None:
    # per_task_status missing.
    art = _valid_artifact(tmp_path)
    cs = dict(art["milestone_376_closestate"])
    cs.pop("per_task_status")
    art["milestone_376_closestate"] = cs
    with pytest.raises(ValueError):
        _revalidate(art)

    # mechanism_fix_worked not True.
    art = _valid_artifact(tmp_path)
    cs = copy.deepcopy(art["milestone_376_closestate"])
    cs["mechanism_fix_worked"] = False
    art["milestone_376_closestate"] = cs
    with pytest.raises(ValueError):
        _revalidate(art)

    # g1 not a mapping.
    art = _valid_artifact(tmp_path)
    cs = copy.deepcopy(art["milestone_376_closestate"])
    cs["g1_off_arc_transfer"] = "nope"
    art["milestone_376_closestate"] = cs
    with pytest.raises(ValueError):
        _revalidate(art)

    # g1 accumulated_n == 0 (mechanism did not produce a measurement).
    art = _valid_artifact(tmp_path)
    cs = copy.deepcopy(art["milestone_376_closestate"])
    cs["g1_off_arc_transfer"]["accumulated_n"] = 0
    art["milestone_376_closestate"] = cs
    with pytest.raises(ValueError):
        _revalidate(art)

    # g1 mechanism_produced_measurement False.
    art = _valid_artifact(tmp_path)
    cs = copy.deepcopy(art["milestone_376_closestate"])
    cs["g1_off_arc_transfer"]["mechanism_produced_measurement"] = False
    art["milestone_376_closestate"] = cs
    with pytest.raises(ValueError):
        _revalidate(art)

    # g1 flagged but NOT skipped from aggregation (laundering a flagged result).
    art = _valid_artifact(tmp_path)
    cs = copy.deepcopy(art["milestone_376_closestate"])
    cs["g1_off_arc_transfer"]["skipped_from_aggregation"] = False
    art["milestone_376_closestate"] = cs
    with pytest.raises(ValueError):
        _revalidate(art)

    # g3 not a mapping.
    art = _valid_artifact(tmp_path)
    cs = copy.deepcopy(art["milestone_376_closestate"])
    cs["g3_decentralization_moe_base"] = "nope"
    art["milestone_376_closestate"] = cs
    with pytest.raises(ValueError):
        _revalidate(art)

    # g3 not reached_floor.
    art = _valid_artifact(tmp_path)
    cs = copy.deepcopy(art["milestone_376_closestate"])
    cs["g3_decentralization_moe_base"]["reached_floor"] = False
    art["milestone_376_closestate"] = cs
    with pytest.raises(ValueError):
        _revalidate(art)

    # g3 sovereign base not absent.
    art = _valid_artifact(tmp_path)
    cs = copy.deepcopy(art["milestone_376_closestate"])
    cs["g3_decentralization_moe_base"]["sovereign_base_status"] = "present_or_unknown"
    art["milestone_376_closestate"] = cs
    with pytest.raises(ValueError):
        _revalidate(art)

    # accuracy not a mapping.
    art = _valid_artifact(tmp_path)
    cs = copy.deepcopy(art["milestone_376_closestate"])
    cs["accuracy"] = "nope"
    art["milestone_376_closestate"] = cs
    with pytest.raises(ValueError):
        _revalidate(art)

    # accuracy.total_games_solved not 9 (the ninth-game-solved invariant).
    art = _valid_artifact(tmp_path)
    cs = copy.deepcopy(art["milestone_376_closestate"])
    cs["accuracy"]["total_games_solved"] = 8
    art["milestone_376_closestate"] = cs
    with pytest.raises(ValueError):
        _revalidate(art)

    # accuracy not monotonic.
    art = _valid_artifact(tmp_path)
    cs = copy.deepcopy(art["milestone_376_closestate"])
    cs["accuracy"]["monotonic_no_regression"] = False
    art["milestone_376_closestate"] = cs
    with pytest.raises(ValueError):
        _revalidate(art)


def test_validate_artifact_missing_required_field(tmp_path: Path) -> None:
    art = _valid_artifact(tmp_path)
    art.pop("flagged_count")
    with pytest.raises(ValueError, match="missing required fields"):
        _revalidate(art)


def test_validate_artifact_checksum_branches(tmp_path: Path) -> None:
    art = _valid_artifact(tmp_path)
    art["reproducibility_checksum"] = "not-a-sha"
    with pytest.raises(ValueError, match="sha256"):
        mod.validate_artifact(art)

    art = _valid_artifact(tmp_path)
    art["reproducibility_checksum"] = "a" * 64
    with pytest.raises(ValueError, match="does not match"):
        mod.validate_artifact(art)


def test_build_blocked_artifact_has_required_fields() -> None:
    art = mod.build_blocked_artifact("blocked_demo", preconditions_checked={}, duration_s=0.5,
                                     active_milestone_confirmed="2026.06.376",
                                     active_roadmap_path="research-roadmap.yaml")
    assert art["honest_verdict"] == "blocked_demo"
    assert art["flagged_count"] == 0
    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert field in art
