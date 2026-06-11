"""Tests for the Exp 4066 .375 archive / .376 activation record-only module.

Spec refs: REQ-REPORT-4066, SCENARIO-REPORT-4066,
SCENARIO-REPORT-4066-BLOCKED-YAML.

These tests exercise the disciplined milestone-transition module end to end on
a synthetic repo fixture (no live model, no real conductor), plus every pure
helper and every blocked-path branch. The load-bearing assertions:

* the `.375 close-state is recorded as a MECHANISM failure, not a science
  negative (G1 off-ARC accumulated N=0 because the runner never launched, G3
  MoE cascade-blocked), and BOTH candidate checkpoints are confirmed intact;
* the colon-poison guard keeps research-complete.yaml parseable;
* a red pre-test gate is quarantined to green before the milestone is declared;
* the terminal artifact carries every required principle-annotated field.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from carnot.reporting import archive_v375_activate_v376_4066 as mod


# --------------------------------------------------------------------------- #
# Fixture: a synthetic repo with a valid .375 close-state
# --------------------------------------------------------------------------- #
GREEN = mod.CommandResult(command=["pytest"], exit_code=0, stdout="", stderr="")
PARSE_OK = mod.CommandResult(command=["yaml"], exit_code=0, stdout="", stderr="")
IMPORT_OK = mod.CommandResult(command=["import"], exit_code=0, stdout="", stderr="")


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _research_complete_text() -> str:
    """A minimal research-complete.yaml that already holds a .375 record.

    Mirrors the real repo: the conductor checkpoint commit appended exactly one
    canonical 2026.06.375 record, so the common action for this module is
    ``unchanged``.
    """

    return (
        "- id: 2026.06.374\n"
        "  finding: prior milestone\n"
        "- id: 2026.06.375\n"
        "  title: 'POWER the off-ARC verifier transfer (resume-not-restart)'\n"
        "  completed: '2026-06-11'\n"
        "  finding: See conductor log for per-experiment results.\n"
        "  tasks:\n"
        "  - id: exp4054-archive-v374-activate-v375\n"
        "    result: OK (conductor)\n"
    )


def make_repo(tmp_path: Path, *, offarc: bool = True, moe: bool = True) -> Path:
    """Build a synthetic repo with the .375 artifacts + the two checkpoints."""

    root = tmp_path
    (root / "research-complete.yaml").write_text(_research_complete_text(), encoding="utf-8")
    (root / "ops").mkdir(parents=True, exist_ok=True)
    (root / "ops" / "exclusion_manifest.yaml").write_text(
        "retired:\n- experiment_id: 2091\n  reason: gemini bail-out\n", encoding="utf-8"
    )
    (root / "research-roadmap.yaml").write_text(
        "milestone: 2026.06.376\nname: v376\n", encoding="utf-8"
    )

    results = root / "results"
    # OK terminal artifacts.
    for exp, name in (
        ("4054", "experiment_4054_archive_v374_activate_v375.json"),
        ("4055", "experiment_4055_sota_ingestion_receipt.json"),
        ("4063", "experiment_4063_verifier_registry_and_gaps_hygiene.json"),
    ):
        _write_json(results / name, {"honest_verdict": f"complete: ok_{exp}"})
    # FLAGGED off-ARC BUILD half: the runner never launched.
    _write_json(
        results / "experiment_4056_offarc_power_evalplus_build.json",
        {
            "honest_verdict": "blocked_smoke_failed",
            "flagged_adversarial": True,
            "launched_pid": 0,
            "smoke_oracle_headroom_present": False,
        },
    )
    # OK off-ARC COLLECT half: polled an empty checkpoint -> accumulated_n=0.
    _write_json(
        results / "experiment_4057_offarc_power_evalplus.json",
        {
            "honest_verdict": "complete: offarc_power_evalplus_accumulating_n_0",
            "accumulated_n_tasks": 0,
            "best_arm": "armC_symbolic",
            "oracle_headroom_present": False,
        },
    )
    # ArcMemo v8 non-result.
    _write_json(
        results / "experiment_4062_arcmemo_cross_game_transfer_v8.json",
        {
            "honest_verdict": "complete: arcmemo_v8_no_cross_game_transfer",
            "cross_game_transfer_win": False,
            "n_reused_abstractions": 0,
            "transfer_assessment": "unmeasured_no_usable_trace",
        },
    )
    # Hardware continuity OK.
    _write_json(
        results / "experiment_4064_hardware_continuity.json",
        {
            "honest_verdict": "complete: hardware_continuity_ok",
            "per_board_reachability": {"gatemate": True, "kv260": True, "polarfire": True},
            "per_board_terminal_state": {"kv260": "terminal"},
            "kv260_terminal_confirmed": True,
        },
    )
    # Capstone OK with total_games_solved=8.
    _write_json(
        results / "experiment_4065_capstone_v375.json",
        {"honest_verdict": "complete: capstone_v375", "total_games_solved": 8},
    )
    # exp4058-4061 deliberately ABSENT (cascade-blocked -> MISSING on disk).

    if offarc:
        _write_json(
            results / "experiment_4045_offarc_transfer_power.checkpoint.json",
            {"completed_task_ids": [f"t{i}" for i in range(23)], "schema": "offarc.v1"},
        )
    if moe:
        _write_json(
            results / "experiment_4048_decentralization_moe_base_raw.checkpoint.json",
            {"tasks": {f"t{i}": {"coverage": 0.3} for i in range(14)}, "schema": "moe.v1"},
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
# SCENARIO-REPORT-4066: the happy path
# --------------------------------------------------------------------------- #
def test_run_happy_path_writes_terminal_artifact(tmp_path: Path) -> None:
    root = make_repo(tmp_path)
    art = run_happy(root)

    assert art["archived_milestone"] == "2026.06.375"
    assert art["activated_milestone"] == "2026.06.376"
    assert art["active_milestone_confirmed"] == "2026.06.376"
    assert art["honest_verdict"].startswith("success:")
    assert art["inference_substrate"] == "aggregation_from_upstream_artifacts"
    assert art["pretest_suite_green"] is True
    assert art["offarc_checkpoint_intact"] is True
    assert art["moe_checkpoint_intact"] is True
    assert art["offarc_checkpoint_n_tasks"] == 23
    assert art["moe_checkpoint_n_tasks"] == 14
    assert art["duration_s"] == 1.0
    # The artifact validates against its own schema.
    mod.validate_artifact(art)


def test_run_happy_records_mechanism_failure_not_science(tmp_path: Path) -> None:
    root = make_repo(tmp_path)
    cs = run_happy(root)["milestone_375_closestate"]

    assert cs["mechanism_failure"] is True
    assert cs["science_negative"] is False
    g1 = cs["g1_off_arc_transfer"]
    assert g1["accumulated_n"] == 0
    assert g1["runner_launched"] is False
    assert g1["mechanism_failure"] is True
    assert g1["science_negative"] is False
    assert g1["offarc_checkpoint_intact"] is True
    assert g1["resumes_in_v376"] == "exp4068"
    g3 = cs["g3_decentralization_moe_base"]
    assert g3["cascade_blocked"] is True
    assert g3["retired"] is False
    assert g3["moe_checkpoint_intact"] is True
    assert g3["resumes_in_v376"] == "exp4069"
    assert cs["efficiency_action_pruner"]["cascade_skipped"] is True
    assert cs["accuracy"]["total_games_solved"] == 8
    assert cs["accuracy"]["ninth_game_cascade_skipped"] is True
    assert cs["self_learning"]["cross_game_transfer_win"] is False
    assert cs["hardware"]["kv260_terminal"] is True
    assert cs["checkpoints_intact"] == {"off_arc": True, "moe": True}


def test_run_happy_per_task_status_counts(tmp_path: Path) -> None:
    root = make_repo(tmp_path)
    cs = run_happy(root)["milestone_375_closestate"]
    status = cs["per_task_status"]

    assert status["exp4056-offarc-power-evalplus-build"] == "FLAGGED"
    assert status["exp4057-offarc-power-evalplus-collect"] == "OK"
    assert status["exp4058-decentralization-moe-resume-build"] == "MISSING"
    assert status["exp4060-ninth-game-explore-first"] == "MISSING"
    # 7 OK (4054,4055,4057,4062,4063,4064,4065), 1 FLAGGED (4056), 4 MISSING.
    assert cs["status_counts"] == {"OK": 7, "BLOCKED": 0, "MISSING": 4, "FLAGGED": 1, "FAIL": 0}
    # The conductor's per-task result annotations preserve the SKIP/GATE_BLOCK why.
    cr = cs["per_task_conductor_result"]
    assert "SKIP_codex_idle_timeout" in cr["exp4058-decentralization-moe-resume-build"]
    assert "GATE_BLOCK" in cr["exp4059-decentralization-moe-resume-collect"]
    assert "CASCADE_SKIP" in cr["exp4061-verifier-action-pruner-efficiency"]


def test_run_appends_record_when_missing(tmp_path: Path) -> None:
    root = make_repo(tmp_path)
    # Remove the conductor-appended .375 record: only a .374 record remains, so
    # the module must append one canonical .375 block (exercising the write).
    (root / "research-complete.yaml").write_text(
        "- id: 2026.06.374\n  finding: prior milestone\n", encoding="utf-8"
    )
    art = run_happy(root)
    assert art["honest_verdict"].startswith("success:")
    assert art["research_complete_record_action"] == "appended"
    import yaml

    loaded = yaml.safe_load((root / "research-complete.yaml").read_text(encoding="utf-8"))
    assert any(r.get("id") == "2026.06.375" for r in loaded)


def test_run_happy_research_complete_unchanged_and_parses(tmp_path: Path) -> None:
    root = make_repo(tmp_path)
    before = (root / "research-complete.yaml").read_text(encoding="utf-8")
    art = run_happy(root)
    after = (root / "research-complete.yaml").read_text(encoding="utf-8")

    # Exactly one .375 record already exists -> action is unchanged, no edit.
    assert art["research_complete_record_action"] == "unchanged"
    assert art["research_complete_duplicates_removed"] == 0
    assert before == after
    import yaml

    assert yaml.safe_load(after) is not None


# --------------------------------------------------------------------------- #
# SCENARIO-REPORT-4066-BLOCKED-YAML + the other blocked branches
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


def test_blocked_when_v376_not_active(tmp_path: Path) -> None:
    root = make_repo(tmp_path)
    (root / "research-roadmap.yaml").write_text("milestone: 2026.06.375\n", encoding="utf-8")
    art = run_happy(root)
    assert art["honest_verdict"] == "blocked_v376_not_active"


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


def test_blocked_when_offarc_checkpoint_missing(tmp_path: Path) -> None:
    root = make_repo(tmp_path, offarc=False)
    art = run_happy(root)
    assert art["honest_verdict"] == "blocked_offarc_checkpoint_missing"
    assert art["offarc_checkpoint_intact"] is False
    # The close-state is still recorded so the planner sees the mechanism truth.
    assert art["milestone_375_closestate"]["mechanism_failure"] is True


def test_blocked_when_moe_checkpoint_missing(tmp_path: Path) -> None:
    root = make_repo(tmp_path, moe=False)
    art = run_happy(root)
    assert art["honest_verdict"] == "blocked_moe_checkpoint_missing"
    assert art["moe_checkpoint_intact"] is False


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
    # The floor protects against a zero/negative measured delta.
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


def test_no_forbidden_markers() -> None:
    assert mod.no_forbidden_markers({"x": "fine"}) is True
    assert mod.no_forbidden_markers({"x": "uses GGUF"}) is False
    # The close-state + principles are excluded (they may name models legitimately).
    assert mod.no_forbidden_markers({"milestone_375_closestate": {"m": "GGUF"}, "field_principles": {"p": "CUDA"}}) is True


def test_write_payload_roundtrips(tmp_path: Path) -> None:
    path = tmp_path / "out.json"
    mod.write_payload(path, {"b": 2, "a": 1})
    text = path.read_text(encoding="utf-8")
    assert text.endswith("\n")
    assert json.loads(text) == {"a": 1, "b": 2}


def test_milestone_from_text() -> None:
    assert mod._milestone_from_text("milestone: 2026.06.376\n") == "2026.06.376"
    assert mod._milestone_from_text("name: x\n") == "unknown"


def test_read_active_milestone_branches(tmp_path: Path) -> None:
    # No roadmap files -> unknown.
    assert mod.read_active_milestone(tmp_path) == ("unknown", "research-roadmap.yaml")
    # roadmap.yaml present but no milestone line -> fall through to next.
    (tmp_path / "research-roadmap.yaml").write_text("name: x\n", encoding="utf-8")
    (tmp_path / "research-roadmap-next.yaml").write_text("milestone: 2026.06.376\n", encoding="utf-8")
    assert mod.read_active_milestone(tmp_path) == ("2026.06.376", "research-roadmap-next.yaml")
    # roadmap.yaml with milestone wins.
    (tmp_path / "research-roadmap.yaml").write_text("milestone: 2026.06.999\n", encoding="utf-8")
    assert mod.read_active_milestone(tmp_path) == ("2026.06.999", "research-roadmap.yaml")


def test_command_builders() -> None:
    assert mod.research_complete_yaml_command()[0].endswith("python")
    assert "yaml.safe_load" in mod.research_complete_yaml_command()[2]
    assert "importlib" in mod.arc_modules_import_command()[2]


def test_read_checkpoint_task_count_shapes(tmp_path: Path) -> None:
    tasks = tmp_path / "tasks.json"
    _write_json(tasks, {"tasks": {"a": 1, "b": 2}})
    assert mod.read_checkpoint_task_count(tasks) == 2

    ids = tmp_path / "ids.json"
    _write_json(ids, {"completed_task_ids": ["x", "y", "z"]})
    assert mod.read_checkpoint_task_count(ids) == 3

    evals = tmp_path / "evals.json"
    _write_json(evals, {"evaluations_by_task": {"a": 1}})
    assert mod.read_checkpoint_task_count(evals) == 1

    assert mod.read_checkpoint_task_count(tmp_path / "absent.json") is None

    listed = tmp_path / "list.json"
    listed.write_text("[1, 2, 3]", encoding="utf-8")
    assert mod.read_checkpoint_task_count(listed) is None

    nokeys = tmp_path / "nokeys.json"
    _write_json(nokeys, {"schema": "x", "tasks": "not-a-collection"})
    assert mod.read_checkpoint_task_count(nokeys) is None


def test_record_id() -> None:
    assert mod._record_id("- id: 2026.06.375") == "2026.06.375"
    assert mod._record_id("  not a record") is None


def test_dedupe_or_append_record_branches() -> None:
    one = "- id: 2026.06.375\n  finding: x\n"
    assert mod.dedupe_or_append_record(one, "2026.06.375") == (one, 0, "unchanged")

    none = "- id: 2026.06.374\n  finding: prior\n"
    new_text, removed, action = mod.dedupe_or_append_record(none, "2026.06.375")
    assert action == "appended" and removed == 0
    import yaml

    assert yaml.safe_load(new_text) is not None
    assert any(r.get("id") == "2026.06.375" for r in yaml.safe_load(new_text))

    dup = "- id: 2026.06.375\n  finding: a\n- id: 2026.06.375\n  finding: b\n"
    new_text, removed, action = mod.dedupe_or_append_record(dup, "2026.06.375")
    assert action == "deduped" and removed == 1
    assert new_text.count("- id: 2026.06.375") == 1


def test_build_canonical_record_parses() -> None:
    import yaml

    record = mod.build_canonical_record()
    loaded = yaml.safe_load(record)
    assert loaded[0]["id"] == "2026.06.375"
    assert len(loaded[0]["tasks"]) == len(mod.V375_TASKS)


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


def test_fields_and_is_real_number() -> None:
    assert mod._fields({"fields": {"a": 1}}) == {"a": 1}
    assert mod._fields({"fields": "nope"}) == {}
    assert mod._is_real_number(1) is True
    assert mod._is_real_number(1.5) is True
    assert mod._is_real_number(True) is False
    assert mod._is_real_number("1") is False


def test_g1_off_arc_runner_launched_branch() -> None:
    # If the runner DID launch and produced tasks, it is not a mechanism failure.
    build = {"exists": True, "fields": {"launched_pid": 1234}}
    collect = {"exists": True, "fields": {"accumulated_n_tasks": 40}}
    g1 = mod._g1_off_arc_mechanism_failure(build, collect, 50)
    assert g1["runner_launched"] is True
    assert g1["accumulated_n"] == 40
    assert g1["mechanism_failure"] is False


def test_g1_off_arc_missing_collect_defaults_to_zero() -> None:
    g1 = mod._g1_off_arc_mechanism_failure({"exists": False}, {"exists": False}, None)
    assert g1["accumulated_n"] == 0
    assert g1["mechanism_failure"] is True
    assert g1["offarc_checkpoint_intact"] is False


def test_g3_and_efficiency_and_accuracy_helpers() -> None:
    g3 = mod._g3_moe_cascade_blocked({"exists": False}, {"exists": False}, 14)
    assert g3["cascade_blocked"] is True and g3["moe_checkpoint_intact"] is True

    eff_missing = mod._efficiency_cascade_skipped({"exists": False})
    assert eff_missing["cascade_skipped"] is True
    eff_present = mod._efficiency_cascade_skipped({"exists": True})
    assert eff_present["cascade_skipped"] is False and eff_present["measured"] is True

    acc_default = mod._accuracy({"exists": False}, {"exists": False})
    assert acc_default["total_games_solved"] == 8
    assert acc_default["ninth_game_cascade_skipped"] is True
    acc_present = mod._accuracy(
        {"exists": True, "fields": {"total_games_solved": 9}}, {"exists": True}
    )
    assert acc_present["total_games_solved"] == 9
    assert acc_present["ninth_game_cascade_skipped"] is False


def test_self_learning_and_hardware_helpers() -> None:
    sl = mod._self_learning({"exists": True, "honest_verdict": "complete: x",
                             "fields": {"cross_game_transfer_win": False, "n_reused_abstractions": 0}})
    assert sl["cross_game_transfer_win"] is False
    hw = mod._hardware({"exists": True, "honest_verdict": "complete: hw",
                        "fields": {"per_board_reachability": {"kv260": True},
                                   "per_board_terminal_state": {"kv260": "terminal"},
                                   "kv260_terminal_confirmed": True}})
    assert hw["included"] is True and hw["kv260_terminal"] is True
    # Non-mapping reachability falls back to empty dicts.
    hw2 = mod._hardware({"exists": True, "honest_verdict": "complete: hw", "fields": {}})
    assert hw2["per_board_reachability"] == {}


def test_read_v375_records(tmp_path: Path) -> None:
    root = make_repo(tmp_path)
    records = mod.read_v375_records(root)
    assert records["4056"]["flagged_adversarial"] is True
    assert records["4058"]["exists"] is False


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
    # Nothing exists under the temp root -> fall back to the first core suite.
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


def test_parse_failing_test_ids() -> None:
    out = (
        "FAILED tests/python/test_a.py::test_one - AssertionError\n"
        "ERROR tests/python/test_b.py::test_two - ImportError\n"
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
    # index out of supplied range -> live smart subset.
    assert mod._pretest_at(tmp_path, supplied, 5) is sentinel
    # supplied None -> live smart subset.
    assert mod._pretest_at(tmp_path, None, 0) is sentinel


def test_run_pretest_until_green_branches(tmp_path: Path) -> None:
    # Green on first try.
    ok, q, results = mod.run_pretest_until_green(tmp_path, [GREEN])
    assert ok is True and q == [] and len(results) == 1

    # Red but no parseable failures -> not green, no quarantine.
    red = mod.CommandResult(command=["pytest"], exit_code=1, stdout="boom", stderr="")
    ok, q, _ = mod.run_pretest_until_green(tmp_path, [red])
    assert ok is False and q == []

    # Red with a failure, then green -> quarantine then pass.
    src = tmp_path / "tests" / "python" / "test_flake.py"
    src.parent.mkdir(parents=True)
    src.write_text("def test_flake(): assert False\n", encoding="utf-8")
    red_fail = mod.CommandResult(
        command=["pytest"], exit_code=1,
        stdout="FAILED tests/python/test_flake.py::test_flake - boom", stderr="",
    )
    ok, q, _ = mod.run_pretest_until_green(tmp_path, [red_fail, GREEN])
    assert ok is True and q[0]["path"] == "tests/python/test_flake.py"


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
    verdict = mod.terminal_verdict({"accuracy": {"total_games_solved": 8}})
    assert verdict.startswith("success:")
    assert "games8" in verdict
    assert "mechanism_failure_not_science" in verdict


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
        lambda a: a.update(active_milestone_confirmed="2026.06.375"),
        lambda a: a.update(n_tasks_archived=3),
        lambda a: a.update(offarc_checkpoint_intact=False),
        lambda a: a.update(moe_checkpoint_intact=False),
        lambda a: a.update(milestone_375_closestate={}),
        lambda a: a.update(duration_s=0),
        lambda a: a.update(inference_substrate="live_llm_inference"),
        lambda a: a.update(quarantined_tests="nope"),
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
    cs = dict(art["milestone_375_closestate"])
    cs.pop("per_task_status")
    art["milestone_375_closestate"] = cs
    with pytest.raises(ValueError):
        _revalidate(art)

    # mechanism_failure not True.
    art = _valid_artifact(tmp_path)
    cs = copy.deepcopy(art["milestone_375_closestate"])
    cs["mechanism_failure"] = False
    art["milestone_375_closestate"] = cs
    with pytest.raises(ValueError):
        _revalidate(art)

    # g1 not a mapping.
    art = _valid_artifact(tmp_path)
    cs = copy.deepcopy(art["milestone_375_closestate"])
    cs["g1_off_arc_transfer"] = "nope"
    art["milestone_375_closestate"] = cs
    with pytest.raises(ValueError):
        _revalidate(art)

    # g1 records a science negative (forbidden framing).
    art = _valid_artifact(tmp_path)
    cs = copy.deepcopy(art["milestone_375_closestate"])
    cs["g1_off_arc_transfer"]["science_negative"] = True
    art["milestone_375_closestate"] = cs
    with pytest.raises(ValueError):
        _revalidate(art)

    # g1 off-ARC checkpoint not intact.
    art = _valid_artifact(tmp_path)
    cs = copy.deepcopy(art["milestone_375_closestate"])
    cs["g1_off_arc_transfer"]["offarc_checkpoint_intact"] = False
    art["milestone_375_closestate"] = cs
    with pytest.raises(ValueError):
        _revalidate(art)

    # g3 not a mapping.
    art = _valid_artifact(tmp_path)
    cs = copy.deepcopy(art["milestone_375_closestate"])
    cs["g3_decentralization_moe_base"] = "nope"
    art["milestone_375_closestate"] = cs
    with pytest.raises(ValueError):
        _revalidate(art)

    # g3 retired (forbidden).
    art = _valid_artifact(tmp_path)
    cs = copy.deepcopy(art["milestone_375_closestate"])
    cs["g3_decentralization_moe_base"]["retired"] = True
    art["milestone_375_closestate"] = cs
    with pytest.raises(ValueError):
        _revalidate(art)


def test_validate_artifact_missing_required_field(tmp_path: Path) -> None:
    art = _valid_artifact(tmp_path)
    art.pop("offarc_checkpoint_intact")
    with pytest.raises(ValueError, match="missing required fields"):
        _revalidate(art)


def test_validate_artifact_checksum_branches(tmp_path: Path) -> None:
    # Non-sha checksum.
    art = _valid_artifact(tmp_path)
    art["reproducibility_checksum"] = "not-a-sha"
    with pytest.raises(ValueError, match="sha256"):
        mod.validate_artifact(art)

    # Valid sha shape but does not match payload.
    art = _valid_artifact(tmp_path)
    art["reproducibility_checksum"] = "a" * 64
    with pytest.raises(ValueError, match="does not match"):
        mod.validate_artifact(art)


def test_build_blocked_artifact_has_required_fields() -> None:
    art = mod.build_blocked_artifact("blocked_demo", preconditions_checked={}, duration_s=0.5,
                                     active_milestone_confirmed="2026.06.375",
                                     active_roadmap_path="research-roadmap.yaml")
    assert art["honest_verdict"] == "blocked_demo"
    assert art["offarc_checkpoint_intact"] is False
    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert field in art
