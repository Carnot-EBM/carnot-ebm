"""Tests for the Exp 4086 .377 archive / .378 activation record-only module.

Spec refs: REQ-REPORT-4086, SCENARIO-REPORT-4086,
SCENARIO-REPORT-4086-BLOCKED-YAML.

These tests exercise the disciplined milestone-transition module end to end on
a synthetic repo fixture (no live model, no real conductor), plus every pure
helper and every blocked-path branch. The load-bearing assertions:

* the `.377 close-state is recorded as a BLOCKED PIVOT (an honest negative, not
  a paper-over): the verifier-as-reward RFT was NOT measured because the Phase-0
  verifier-precision gate (exp4077) measured 0.6818 < 0.85, poisoning the
  RFT-CORRECT corpus, so exp4078 (train) and exp4079 (held-out eval) cascaded to
  blocked; the Sudoku positive control (exp4080) is flagged + skipped; 4 of 10
  artifacts are flagged-and-skipped; ACCURACY holds at 9 games (ninth solved);
* the colon-poison guard keeps research-complete.yaml parseable;
* a red OR collection-error pre-test gate is quarantined to green;
* the terminal artifact carries every required principle-annotated field.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from carnot.reporting import archive_v377_activate_v378_4086 as mod


# --------------------------------------------------------------------------- #
# Fixture: a synthetic repo with a valid .377 blocked-pivot close-state
# --------------------------------------------------------------------------- #
GREEN = mod.CommandResult(command=["pytest"], exit_code=0, stdout="", stderr="")
PARSE_OK = mod.CommandResult(command=["yaml"], exit_code=0, stdout="", stderr="")
IMPORT_OK = mod.CommandResult(command=["import"], exit_code=0, stdout="", stderr="")


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _research_complete_text() -> str:
    """A minimal research-complete.yaml that already holds a .377 record.

    Mirrors the real repo: the conductor activation commit appended exactly one
    canonical 2026.06.377 record, so the common action for this module is
    ``unchanged``.
    """

    return (
        "- id: 2026.06.376\n"
        "  finding: prior milestone\n"
        "- id: 2026.06.377\n"
        "  title: 'verifier-as-reward pivot'\n"
        "  completed: '2026-06-12'\n"
        "  finding: See conductor log for per-experiment results.\n"
        "  tasks:\n"
        "  - id: exp4076-archive-v376-activate-v377\n"
        "    result: OK (conductor)\n"
    )


def make_repo(tmp_path: Path) -> Path:
    """Build a synthetic repo mirroring the real .377 artifacts."""

    root = tmp_path
    (root / "research-complete.yaml").write_text(_research_complete_text(), encoding="utf-8")
    (root / "ops").mkdir(parents=True, exist_ok=True)
    (root / "ops" / "exclusion_manifest.yaml").write_text(
        "retired:\n- experiment_id: 2091\n  reason: gemini bail-out\n", encoding="utf-8"
    )
    (root / "research-roadmap.yaml").write_text(
        'milestone: "2026.06.378"\nname: v378\n', encoding="utf-8"
    )

    results = root / "results"
    # exp4076 OK terminal (the .376 archive).
    _write_json(
        results / "experiment_4076_archive_v376_activate_v377.json",
        {"honest_verdict": "success: archived_v376_v377"},
    )
    # exp4077 FLAGGED: precision gate unmet 0.6818 < 0.85 -> corpus poisoned.
    _write_json(
        results / "experiment_4077_verifier_reward_rft_corpus_build.json",
        {
            "honest_verdict": "blocked_precision_gate_unmet_0.6818_1.0000",
            "flagged_adversarial": True,
            "certification_precision": 0.6818,
            "certification_recall": 1.0,
        },
    )
    # exp4078 FLAGGED: blocked because exp4077's corpora are missing.
    _write_json(
        results / "experiment_4078_verifier_reward_rft_train_launch.json",
        {"honest_verdict": "blocked_exp4077_corpora_missing", "flagged_adversarial": True},
    )
    # exp4079 BLOCKED (NOT flagged): the de-confounded gate never ran a real eval.
    _write_json(
        results / "experiment_4079_verifier_reward_rft_eval_collect.json",
        {"honest_verdict": "blocked_gate_check_failed"},
    )
    # exp4080 FLAGGED: sudoku control claims complete but in 4.4s on live-GPU.
    _write_json(
        results / "experiment_4080_sudoku_rft_positive_control.json",
        {
            "honest_verdict": "complete: sudoku_positive_control_rft_ge_sft_reproduced",
            "flagged_adversarial": True,
        },
    )
    # exp4081 OK receipt: SOTA verifier-as-reward mapped (methods in capstone).
    _write_json(
        results / "experiment_4081_sota_ingestion_verifier_as_reward_receipt.json",
        {"honest_verdict": "complete: sota_ingestion_verifier_as_reward_mapped"},
    )
    # exp4082 ninth game SOLVED clean -> total stays at 9.
    _write_json(
        results / "experiment_4082_ninth_game_explore_first.json",
        {
            "honest_verdict": "success: ninth_game_solved_ft09-0d8bbf25_at_action_4",
            "game_solved": True,
            "prior_total_games_solved": 8,
            "target_game": "ft09-0d8bbf25",
            "first_solve_at_action": 4,
            "real_env_confirmed": True,
        },
    )
    # exp4083 FLAGGED: gap4 reproduced but duration too short.
    _write_json(
        results / "experiment_4083_verifier_registry_gaps_hygiene.json",
        {
            "honest_verdict": "complete: gap4_arc1_reproduced_True_safety_gate_regression_True",
            "flagged_adversarial": True,
        },
    )
    # exp4084 hardware: all reachable, gatemate flash blocked, polarfire ok, kv260 terminal.
    _write_json(
        results / "experiment_4084_hardware_continuity.json",
        {
            "honest_verdict": "complete: hardware_continuity_gatemate_blocked_polarfire_ok_kv260_terminal",
            "per_board_reachability": {"gatemate": True, "kv260": True, "polarfire": True},
            "per_board_terminal_state": {"kv260": "opportunistic_terminal_confirmed_ssh_only"},
            "gatemate_step_taken": "gatemate_existing_n16_bitstream_flash_blocked_returncode_1",
            "polarfire_step_taken": "polarfire_hash_verified_cpu_dispatch_succeeded",
            "kv260_terminal_confirmed": True,
        },
    )
    # exp4085 capstone: pivot blocked, games=9, sota methods=8.
    _write_json(
        results / "experiment_4085_capstone_v377.json",
        {
            "honest_verdict": "complete: capstone_v377_pivot_blocked_no_arc_rft_eval",
            "games_solved_total": 9,
            "sota_ingestion": {"methods_mapped_count": 8},
        },
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
# SCENARIO-REPORT-4086: the happy path
# --------------------------------------------------------------------------- #
def test_run_happy_path_writes_terminal_artifact(tmp_path: Path) -> None:
    root = make_repo(tmp_path)
    art = run_happy(root)

    assert art["archived_milestone"] == "2026.06.377"
    assert art["activated_milestone"] == "2026.06.378"
    assert art["active_milestone_confirmed"] == "2026.06.378"
    assert art["honest_verdict"].startswith("success:")
    assert "pivot_blocked" in art["honest_verdict"]
    assert "0.6818" in art["honest_verdict"]
    assert art["inference_substrate"] == "aggregation_from_upstream_artifacts"
    assert art["pretest_suite_green"] is True
    assert art["total_games_solved"] == 9
    assert art["flagged_count"] == 4
    assert art["duration_s"] == 1.0
    # cited upstream provenance trail covers all 10 .377 deliverables with sha256.
    cited = art["cited_upstream_artifacts"]
    assert len(cited) == len(mod.V377_TASKS)
    assert all(mod.is_sha256(c["sha256"]) for c in cited)
    # The artifact validates against its own schema.
    mod.validate_artifact(art)


def test_run_happy_records_pivot_blocked(tmp_path: Path) -> None:
    root = make_repo(tmp_path)
    cs = run_happy(root)["milestone_377_closestate"]

    assert cs["pivot_blocked"] is True
    assert cs["pivot_decisive"] is False
    pivot = cs["pivot"]
    assert pivot["blocked"] is True
    assert pivot["blocked_at_layer"] == "phase0_verifier_precision_gate"
    assert pivot["certification_precision"] == 0.6818
    assert pivot["certification_recall"] == 1.0
    assert pivot["precision_gate_threshold"] == 0.85
    assert pivot["precision_gate_passed"] is False
    assert pivot["corpus_poisoned"] is True
    assert pivot["rft_eval_measured"] is False
    assert pivot["rft_beats_gold_sft"] is False
    assert "0.6818" in pivot["outcome"]


def test_run_happy_records_sudoku_and_accuracy(tmp_path: Path) -> None:
    root = make_repo(tmp_path)
    cs = run_happy(root)["milestone_377_closestate"]

    sudoku = cs["sudoku_control"]
    assert sudoku["flagged_adversarial"] is True
    assert sudoku["skipped_from_aggregation"] is True
    assert sudoku["reproduces_beachhead"] is False
    assert sudoku["trustworthy"] is False

    acc = cs["accuracy"]
    assert acc["total_games_solved"] == 9
    assert acc["ninth_game_solved"] is True
    assert acc["ninth_game"] == "ft09-0d8bbf25"
    assert acc["first_solve_at_action"] == 4
    assert acc["real_env_confirmed"] is True
    assert acc["monotonic_no_regression"] is True


def test_run_happy_records_sota_hardware_and_flagged(tmp_path: Path) -> None:
    root = make_repo(tmp_path)
    cs = run_happy(root)["milestone_377_closestate"]

    assert cs["sota_ingestion"]["methods_mapped_count"] == 8
    assert cs["sota_ingestion"]["included"] is True

    hw = cs["hardware"]
    assert hw["kv260_terminal"] is True
    assert "blocked" in hw["gatemate_step"]
    assert "succeeded" in hw["polarfire_step"]
    assert hw["per_board_reachability"] == {"gatemate": True, "kv260": True, "polarfire": True}

    flagged = cs["flagged_skipped"]
    assert flagged["count"] == 4
    assert flagged["experiment_ids"] == ["4077", "4078", "4080", "4083"]
    assert all(item["flagged_adversarial"] is True for item in flagged["skipped"])


def test_run_happy_per_task_status_counts(tmp_path: Path) -> None:
    root = make_repo(tmp_path)
    cs = run_happy(root)["milestone_377_closestate"]
    status = cs["per_task_status"]

    assert status["exp4076-archive-v376-activate-v377"] == "OK"
    assert status["exp4077-verifier-reward-rft-corpus-build"] == "FLAGGED"
    assert status["exp4078-verifier-reward-rft-train-launch"] == "FLAGGED"
    assert status["exp4079-verifier-reward-rft-eval-collect"] == "BLOCKED"
    assert status["exp4080-sudoku-rft-beachhead-positive-control"] == "FLAGGED"
    assert status["exp4081-sota-ingestion-verifier-as-reward"] == "OK"
    assert status["exp4082-ninth-game-explore-first"] == "OK"
    assert status["exp4083-verifier-registry-and-gaps-hygiene"] == "FLAGGED"
    # 5 OK, 4 FLAGGED, 1 BLOCKED, nothing missing/failing.
    assert cs["status_counts"] == {"OK": 5, "BLOCKED": 1, "MISSING": 0, "FLAGGED": 4, "FAIL": 0}
    cr = cs["per_task_conductor_result"]
    assert "0.6818" in cr["exp4077-verifier-reward-rft-corpus-build"]
    assert "BLOCKED" in cr["exp4079-verifier-reward-rft-eval-collect"]


def test_run_appends_record_when_missing(tmp_path: Path) -> None:
    root = make_repo(tmp_path)
    # Remove the conductor-appended .377 record: only a .376 record remains, so
    # the module must append one canonical .377 block (exercising the write).
    (root / "research-complete.yaml").write_text(
        "- id: 2026.06.376\n  finding: prior milestone\n", encoding="utf-8"
    )
    art = run_happy(root)
    assert art["honest_verdict"].startswith("success:")
    assert art["research_complete_record_action"] == "appended"
    import yaml

    loaded = yaml.safe_load((root / "research-complete.yaml").read_text(encoding="utf-8"))
    assert any(r.get("id") == "2026.06.377" for r in loaded)


def test_run_happy_research_complete_unchanged_and_parses(tmp_path: Path) -> None:
    root = make_repo(tmp_path)
    before = (root / "research-complete.yaml").read_text(encoding="utf-8")
    art = run_happy(root)
    after = (root / "research-complete.yaml").read_text(encoding="utf-8")

    # Exactly one .377 record already exists -> action is unchanged, no edit.
    assert art["research_complete_record_action"] == "unchanged"
    assert art["research_complete_duplicates_removed"] == 0
    assert before == after
    import yaml

    assert yaml.safe_load(after) is not None


def test_run_dedupes_duplicate_record(tmp_path: Path) -> None:
    root = make_repo(tmp_path)
    # Two .377 records (interrupted-run cruft) collapse to the first occurrence.
    (root / "research-complete.yaml").write_text(
        "- id: 2026.06.377\n  finding: a\n- id: 2026.06.377\n  finding: b\n", encoding="utf-8"
    )
    art = run_happy(root)
    assert art["research_complete_record_action"] == "deduped"
    assert art["research_complete_duplicates_removed"] == 1
    text = (root / "research-complete.yaml").read_text(encoding="utf-8")
    assert text.count("- id: 2026.06.377") == 1


# --------------------------------------------------------------------------- #
# SCENARIO-REPORT-4086-BLOCKED-YAML + the other blocked branches
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


def test_blocked_when_v378_not_active(tmp_path: Path) -> None:
    root = make_repo(tmp_path)
    (root / "research-roadmap.yaml").write_text('milestone: "2026.06.377"\n', encoding="utf-8")
    art = run_happy(root)
    assert art["honest_verdict"] == "blocked_v378_not_active"


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
    # The close-state is still recorded so the planner sees the pivot truth.
    assert art["milestone_377_closestate"]["pivot_blocked"] is True


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
    # The close-state + principles are excluded (they may name substrates legitimately).
    assert mod.no_forbidden_markers({"milestone_377_closestate": {"m": "GGUF"}, "field_principles": {"p": "CUDA"}}) is True


def test_write_payload_roundtrips(tmp_path: Path) -> None:
    path = tmp_path / "out.json"
    mod.write_payload(path, {"b": 2, "a": 1})
    text = path.read_text(encoding="utf-8")
    assert text.endswith("\n")
    assert json.loads(text) == {"a": 1, "b": 2}


def test_milestone_from_text() -> None:
    assert mod._milestone_from_text('milestone: "2026.06.378"\n') == "2026.06.378"
    assert mod._milestone_from_text("name: x\n") == "unknown"


def test_read_active_milestone_branches(tmp_path: Path) -> None:
    assert mod.read_active_milestone(tmp_path) == ("unknown", "research-roadmap.yaml")
    (tmp_path / "research-roadmap.yaml").write_text("name: x\n", encoding="utf-8")
    (tmp_path / "research-roadmap-next.yaml").write_text("milestone: 2026.06.378\n", encoding="utf-8")
    assert mod.read_active_milestone(tmp_path) == ("2026.06.378", "research-roadmap-next.yaml")
    (tmp_path / "research-roadmap.yaml").write_text("milestone: 2026.06.999\n", encoding="utf-8")
    assert mod.read_active_milestone(tmp_path) == ("2026.06.999", "research-roadmap.yaml")


def test_command_builders() -> None:
    assert mod.research_complete_yaml_command()[0].endswith("python")
    assert "yaml.safe_load" in mod.research_complete_yaml_command()[2]
    assert "importlib" in mod.arc_modules_import_command()[2]


def test_record_id() -> None:
    assert mod._record_id("- id: 2026.06.377") == "2026.06.377"
    assert mod._record_id("  not a record") is None


def test_dedupe_or_append_record_branches() -> None:
    one = "- id: 2026.06.377\n  finding: x\n"
    assert mod.dedupe_or_append_record(one, "2026.06.377") == (one, 0, "unchanged")

    none = "- id: 2026.06.376\n  finding: prior\n"
    new_text, removed, action = mod.dedupe_or_append_record(none, "2026.06.377")
    assert action == "appended" and removed == 0
    import yaml

    assert yaml.safe_load(new_text) is not None
    assert any(r.get("id") == "2026.06.377" for r in yaml.safe_load(new_text))

    dup = "- id: 2026.06.377\n  finding: a\n- id: 2026.06.377\n  finding: b\n"
    new_text, removed, action = mod.dedupe_or_append_record(dup, "2026.06.377")
    assert action == "deduped" and removed == 1
    assert new_text.count("- id: 2026.06.377") == 1


def test_build_canonical_record_parses() -> None:
    import yaml

    record = mod.build_canonical_record()
    loaded = yaml.safe_load(record)
    assert loaded[0]["id"] == "2026.06.377"
    assert len(loaded[0]["tasks"]) == len(mod.V377_TASKS)


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
    # A flagged artifact whose verdict starts complete: is still FLAGGED, not OK.
    assert mod.classify_status({"exists": True, "flagged_adversarial": True, "honest_verdict": "complete: x"}) == "FLAGGED"
    # doc deliverable: OK when present, MISSING when absent.
    assert mod.classify_status({"exists": True, "honest_verdict": ""}, kind="doc") == "OK"
    assert mod.classify_status({"exists": False}, kind="doc") == "MISSING"
    assert mod.classify_status({"exists": True, "flagged_adversarial": True}, kind="doc") == "FLAGGED"


def test_fields_and_is_real_number() -> None:
    assert mod._fields({"fields": {"a": 1}}) == {"a": 1}
    assert mod._fields({"fields": "nope"}) == {}
    assert mod._is_real_number(1) is True
    assert mod._is_real_number(1.5) is True
    assert mod._is_real_number(True) is False
    assert mod._is_real_number("1") is False


def test_pivot_helper_branches() -> None:
    # Flagged precision-gate-unmet corpus + blocked eval -> blocked pivot.
    corpus = {
        "exists": True, "flagged_adversarial": True,
        "honest_verdict": "blocked_precision_gate_unmet_0.6818_1.0000",
        "fields": {"certification_precision": 0.6818, "certification_recall": 1.0},
    }
    ev = {"exists": True, "honest_verdict": "blocked_gate_check_failed", "fields": {}}
    pivot = mod._pivot_blocked_precision_gate(corpus, ev)
    assert pivot["certification_precision"] == 0.6818
    assert pivot["precision_gate_passed"] is False
    assert pivot["corpus_poisoned"] is True
    assert pivot["rft_eval_measured"] is False
    assert "0.6818" in pivot["outcome"]

    # Missing corpus fields -> falls back to the recorded defaults.
    pivot2 = mod._pivot_blocked_precision_gate({"exists": False}, {"exists": False})
    assert pivot2["certification_precision"] == mod.PRECISION_GATE_MEASURED_DEFAULT
    assert pivot2["certification_recall"] == mod.PRECISION_GATE_RECALL_DEFAULT
    assert pivot2["rft_eval_measured"] is False

    # Hypothetical clean eval (non-blocked verdict) -> rft_eval_measured True.
    ev_clean = {"exists": True, "honest_verdict": "complete: eval_ran", "fields": {}}
    pivot3 = mod._pivot_blocked_precision_gate(corpus, ev_clean)
    assert pivot3["rft_eval_measured"] is True

    # Hypothetical passing gate (precision above floor) -> not poisoned.
    corpus_pass = {"exists": True, "fields": {"certification_precision": 0.9, "certification_recall": 0.5}}
    pivot4 = mod._pivot_blocked_precision_gate(corpus_pass, ev)
    assert pivot4["precision_gate_passed"] is True
    assert pivot4["corpus_poisoned"] is False


def test_sudoku_control_helper_branches() -> None:
    flagged = {"exists": True, "flagged_adversarial": True, "honest_verdict": "complete: sudoku"}
    s = mod._sudoku_control(flagged)
    assert s["flagged_adversarial"] is True
    assert s["skipped_from_aggregation"] is True
    assert s["trustworthy"] is False
    assert s["outcome"].startswith("sudoku_control_flagged")

    clean = {"exists": True, "flagged_adversarial": False, "honest_verdict": "complete: sudoku"}
    s2 = mod._sudoku_control(clean)
    assert s2["trustworthy"] is True
    assert s2["outcome"] == "sudoku_control_recorded"

    missing = mod._sudoku_control({"exists": False})
    assert missing["measured"] is False
    assert missing["trustworthy"] is False


def test_accuracy_helper_branches() -> None:
    cap = {"exists": True, "fields": {"games_solved_total": 9}}
    ninth = {"exists": True, "honest_verdict": "success: ninth",
             "fields": {"game_solved": True, "target_game": "ft09-0d8bbf25", "first_solve_at_action": 4,
                        "real_env_confirmed": True}}
    acc = mod._accuracy(cap, ninth)
    assert acc["total_games_solved"] == 9
    assert acc["ninth_game_solved"] is True
    assert acc["ninth_game"] == "ft09-0d8bbf25"
    assert acc["real_env_confirmed"] is True

    # Capstone lacks the total -> falls back to total_games_solved field.
    cap2 = {"exists": True, "fields": {"total_games_solved": 9}}
    acc2 = mod._accuracy(cap2, ninth)
    assert acc2["total_games_solved"] == 9

    # Both capstone totals absent -> derive from ninth prior+1 when solved.
    cap3 = {"exists": True, "fields": {}}
    ninth3 = {"exists": True, "honest_verdict": "success: ninth",
              "fields": {"game_solved": True, "prior_total_games_solved": 8}}
    acc3 = mod._accuracy(cap3, ninth3)
    assert acc3["total_games_solved"] == 9

    # Nothing present -> default 9; ninth not solved (no terminal verdict).
    acc4 = mod._accuracy({"exists": False}, {"exists": False})
    assert acc4["total_games_solved"] == 9
    assert acc4["ninth_game_solved"] is False


def test_sota_ingestion_helper_branches() -> None:
    cap = {"exists": True, "fields": {"sota_ingestion": {"methods_mapped_count": 8}}}
    receipt = {"exists": True, "honest_verdict": "complete: sota", "fields": {}}
    s = mod._sota_ingestion(cap, receipt)
    assert s["methods_mapped_count"] == 8
    assert s["included"] is True

    # Capstone lacks sota_ingestion -> falls back to the receipt's own count.
    cap2 = {"exists": True, "fields": {}}
    receipt2 = {"exists": True, "honest_verdict": "complete: sota", "fields": {"methods_mapped_count": 5}}
    s2 = mod._sota_ingestion(cap2, receipt2)
    assert s2["methods_mapped_count"] == 5

    # No counts anywhere -> None.
    s3 = mod._sota_ingestion({"exists": False}, {"exists": False})
    assert s3["methods_mapped_count"] is None
    assert s3["included"] is False


def test_hardware_helper_branches() -> None:
    hw = mod._hardware({"exists": True, "honest_verdict": "complete: hw",
                        "fields": {"per_board_reachability": {"kv260": True},
                                   "per_board_terminal_state": {"kv260": "terminal"},
                                   "gatemate_step_taken": "blocked", "polarfire_step_taken": "ok",
                                   "kv260_terminal_confirmed": True}})
    assert hw["included"] is True and hw["kv260_terminal"] is True
    assert hw["gatemate_step"] == "blocked"
    hw2 = mod._hardware({"exists": True, "honest_verdict": "complete: hw", "fields": {}})
    assert hw2["per_board_reachability"] == {}


def test_flagged_skipped_helper() -> None:
    records = {
        "4077": {"exists": True, "flagged_adversarial": True, "honest_verdict": "blocked_x"},
        "4078": {"exists": True, "flagged_adversarial": True, "honest_verdict": "blocked_y"},
        "4080": {"exists": True, "flagged_adversarial": True, "honest_verdict": "complete: z"},
        "4083": {"exists": True, "flagged_adversarial": True, "honest_verdict": "complete: w"},
    }
    flagged = mod._flagged_skipped(records)
    assert flagged["count"] == 4
    assert flagged["experiment_ids"] == ["4077", "4078", "4080", "4083"]
    assert all(item["flagged_adversarial"] for item in flagged["skipped"])
    # Missing record -> flagged False but still listed.
    flagged2 = mod._flagged_skipped({})
    assert flagged2["count"] == 4
    assert all(item["flagged_adversarial"] is False for item in flagged2["skipped"])


def test_read_v377_records_and_cited(tmp_path: Path) -> None:
    root = make_repo(tmp_path)
    records = mod.read_v377_records(root)
    assert records["4077"]["flagged_adversarial"] is True
    assert records["4079"]["honest_verdict"] == "blocked_gate_check_failed"
    cited = mod.build_cited_upstream(root)
    assert len(cited) == len(mod.V377_TASKS)
    assert all("experiment_id" in c and "sha256" in c for c in cited)


def test_read_v377_records_doc_kind(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Force one task to be a doc deliverable to exercise the doc-kind branch.
    root = make_repo(tmp_path)
    doc_task = dict(mod.V377_TASKS[5])
    doc_task["kind"] = "doc"
    doc_task["deliverable"] = "docs/research-notes/sota.md"
    monkeypatch.setattr(mod, "V377_TASKS", (doc_task,))
    (root / "docs" / "research-notes").mkdir(parents=True, exist_ok=True)
    (root / "docs" / "research-notes" / "sota.md").write_text("# note\n", encoding="utf-8")
    records = mod.read_v377_records(root)
    assert records["4081"]["exists"] is True
    assert records["4081"]["honest_verdict"] == ""


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
def test_terminal_verdict_includes_pivot_and_games() -> None:
    verdict = mod.terminal_verdict({
        "pivot": {"certification_precision": 0.6818},
        "accuracy": {"total_games_solved": 9},
        "flagged_skipped": {"count": 4},
    })
    assert verdict.startswith("success:")
    assert "games9" in verdict
    assert "pivot_blocked" in verdict
    assert "0.6818" in verdict
    assert "4_flagged_skipped" in verdict


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
        lambda a: a.update(active_milestone_confirmed="2026.06.377"),
        lambda a: a.update(n_tasks_archived=3),
        lambda a: a.update(total_games_solved=8),
        lambda a: a.update(flagged_count=0),
        lambda a: a.update(milestone_377_closestate={}),
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
    cs = dict(art["milestone_377_closestate"])
    cs.pop("per_task_status")
    art["milestone_377_closestate"] = cs
    with pytest.raises(ValueError):
        _revalidate(art)

    # pivot_blocked not True.
    art = _valid_artifact(tmp_path)
    cs = copy.deepcopy(art["milestone_377_closestate"])
    cs["pivot_blocked"] = False
    art["milestone_377_closestate"] = cs
    with pytest.raises(ValueError):
        _revalidate(art)

    # pivot not a mapping.
    art = _valid_artifact(tmp_path)
    cs = copy.deepcopy(art["milestone_377_closestate"])
    cs["pivot"] = "nope"
    art["milestone_377_closestate"] = cs
    with pytest.raises(ValueError):
        _revalidate(art)

    # pivot blocked flag flipped.
    art = _valid_artifact(tmp_path)
    cs = copy.deepcopy(art["milestone_377_closestate"])
    cs["pivot"]["blocked"] = False
    art["milestone_377_closestate"] = cs
    with pytest.raises(ValueError):
        _revalidate(art)

    # pivot precision gate recorded as passed (laundering the poisoned corpus).
    art = _valid_artifact(tmp_path)
    cs = copy.deepcopy(art["milestone_377_closestate"])
    cs["pivot"]["precision_gate_passed"] = True
    art["milestone_377_closestate"] = cs
    with pytest.raises(ValueError):
        _revalidate(art)

    # pivot precision at/above the floor (contradicts the blocked gate).
    art = _valid_artifact(tmp_path)
    cs = copy.deepcopy(art["milestone_377_closestate"])
    cs["pivot"]["certification_precision"] = 0.9
    art["milestone_377_closestate"] = cs
    with pytest.raises(ValueError):
        _revalidate(art)

    # pivot claims a real RFT eval ran (it did not).
    art = _valid_artifact(tmp_path)
    cs = copy.deepcopy(art["milestone_377_closestate"])
    cs["pivot"]["rft_eval_measured"] = True
    art["milestone_377_closestate"] = cs
    with pytest.raises(ValueError):
        _revalidate(art)

    # sudoku not a mapping.
    art = _valid_artifact(tmp_path)
    cs = copy.deepcopy(art["milestone_377_closestate"])
    cs["sudoku_control"] = "nope"
    art["milestone_377_closestate"] = cs
    with pytest.raises(ValueError):
        _revalidate(art)

    # flagged sudoku NOT skipped (laundering a flagged result).
    art = _valid_artifact(tmp_path)
    cs = copy.deepcopy(art["milestone_377_closestate"])
    cs["sudoku_control"]["skipped_from_aggregation"] = False
    art["milestone_377_closestate"] = cs
    with pytest.raises(ValueError):
        _revalidate(art)

    # accuracy not a mapping.
    art = _valid_artifact(tmp_path)
    cs = copy.deepcopy(art["milestone_377_closestate"])
    cs["accuracy"] = "nope"
    art["milestone_377_closestate"] = cs
    with pytest.raises(ValueError):
        _revalidate(art)

    # accuracy.total_games_solved not 9.
    art = _valid_artifact(tmp_path)
    cs = copy.deepcopy(art["milestone_377_closestate"])
    cs["accuracy"]["total_games_solved"] = 8
    art["milestone_377_closestate"] = cs
    with pytest.raises(ValueError):
        _revalidate(art)

    # accuracy not monotonic.
    art = _valid_artifact(tmp_path)
    cs = copy.deepcopy(art["milestone_377_closestate"])
    cs["accuracy"]["monotonic_no_regression"] = False
    art["milestone_377_closestate"] = cs
    with pytest.raises(ValueError):
        _revalidate(art)

    # flagged_skipped count wrong.
    art = _valid_artifact(tmp_path)
    cs = copy.deepcopy(art["milestone_377_closestate"])
    cs["flagged_skipped"]["count"] = 2
    art["milestone_377_closestate"] = cs
    with pytest.raises(ValueError):
        _revalidate(art)


def test_validate_artifact_rejects_flagged_count_not_four(tmp_path: Path) -> None:
    # Top-level and close-state counts AGREE (so the "must match" guard passes)
    # but both are the wrong value -> the absolute "must be 4" guard fires.
    art = _valid_artifact(tmp_path)
    cs = copy.deepcopy(art["milestone_377_closestate"])
    cs["flagged_count"] = 2
    art["milestone_377_closestate"] = cs
    art["flagged_count"] = 2
    with pytest.raises(ValueError, match="must be 4"):
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
                                     active_milestone_confirmed="2026.06.377",
                                     active_roadmap_path="research-roadmap.yaml")
    assert art["honest_verdict"] == "blocked_demo"
    assert art["flagged_count"] == 0
    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert field in art
