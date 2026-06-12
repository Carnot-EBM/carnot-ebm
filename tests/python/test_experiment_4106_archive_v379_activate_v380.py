"""Tests for the Exp 4106 .379 archive / .380 activation record-only module.

Spec refs: REQ-REPORT-4106, SCENARIO-REPORT-4106,
SCENARIO-REPORT-4106-BLOCKED-YAML.

These tests exercise the disciplined milestone-transition module end to end on a
synthetic repo fixture (no live model, no real conductor), plus every pure helper
and every blocked-path branch. The load-bearing assertions:

* the `.379` close-state records the Carnot verifier as ANTI-DISCRIMINATING on TRM
  ARC grids (``verifier_beats_trm_vote == False``,
  ``real_verifiers_anti_discriminate == True``) -- the honest negative, not a
  paper-over -- so the `.380` planner does not re-attempt RFT on ARC grids;
* the TRM-RFT-conditional task took the SMOKE branch: the native trainer mechanism
  is confirmed (``trm_native_trainer_checkpoint_ok == True``) but the RFT did NOT
  run (``rft_ran == False``) and exp4100 is flagged-and-skipped;
* accuracy advances to 11 games (the eleventh game solved);
* the duplicate `.379` history records collapse to one;
* the colon-poison guard keeps research-complete.yaml parseable;
* the terminal artifact carries every required principle-annotated field with a
  ``success:`` prefix.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from carnot.reporting import archive_v379_activate_v380_4106 as mod


GREEN = mod.CommandResult(command=["pytest"], exit_code=0, stdout="", stderr="")
RED = mod.CommandResult(command=["pytest"], exit_code=1, stdout="FAILED x", stderr="")


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


# The Exp 4099 per-reranker captured_pp map (delta vs TRM majority vote). The best
# reranker, K_OF_N_AGREEMENT at k=1, ties vote at 0.0 (a no-op); every real verifier
# proxy is strictly worse (AUG_INVARIANCE / DEMO_FIT at -0.2258 ~ -23pp).
PER_RERANKER = {
    "AUG_INVARIANCE": {"captured_pp": -0.2258},
    "DEMO_FIT": {"captured_pp": -0.2258},
    "K_OF_N_AGREEMENT": {"captured_pp": 0.0},
    "MIN_HAMMING": {"captured_pp": -0.0323},
    "STACK_ALL": {"captured_pp": -0.2097},
    "STACK_DEMO_AUG": {"captured_pp": -0.2258},
    "TRM_VOTE": {"captured_pp": 0.0},
}


def _research_complete_text(*, duplicates: int = 1) -> str:
    """A research-complete.yaml holding ``duplicates`` copies of the .379 record."""

    head = "- id: 2026.06.378\n  finding: prior milestone\n"
    block = (
        "- id: 2026.06.379\n"
        "  title: 'verifier anti-discriminates on TRM ARC grids'\n"
        "  completed: '2026-06-12'\n"
        "  finding: See conductor log for per-experiment results.\n"
        "  tasks:\n"
        "  - id: exp4098-archive-v378-activate-v379\n"
        "    result: OK (conductor)\n"
    )
    return head + block * duplicates


def make_repo(tmp_path: Path) -> Path:
    """Build a synthetic repo mirroring the real .379 artifacts."""

    root = tmp_path
    (root / "research-complete.yaml").write_text(_research_complete_text(duplicates=3), encoding="utf-8")
    (root / "ops").mkdir(parents=True, exist_ok=True)
    (root / "ops" / "exclusion_manifest.yaml").write_text(
        "retired:\n- experiment_id: 2091\n  reason: gemini bail-out\n", encoding="utf-8"
    )
    (root / "research-roadmap.yaml").write_text(
        'milestone: "2026.06.380"\nname: v380\n', encoding="utf-8"
    )
    # .380 substrate (precondition C): nano-trm native trainer + Sudoku builder.
    (root / "nano-trm" / "src" / "nn").mkdir(parents=True, exist_ok=True)
    (root / "nano-trm" / "src" / "nn" / "train.py").write_text("# trainer\n", encoding="utf-8")
    (root / "nano-trm" / "scripts" / "data").mkdir(parents=True, exist_ok=True)
    (root / "nano-trm" / "scripts" / "data" / "build_sudoku_extreme_dataset.py").write_text(
        "# sudoku\n", encoding="utf-8"
    )
    # Core smart-subset targets so run_smart_subset has files to point at.
    (root / "tests" / "python").mkdir(parents=True, exist_ok=True)
    (root / "tests" / "python" / "test_pipeline_extract.py").write_text("def test_x():\n    assert True\n", encoding="utf-8")
    (root / "tests" / "python" / "test_docs.py").write_text("def test_y():\n    assert True\n", encoding="utf-8")

    results = root / "results"
    _write_json(
        results / "experiment_4098_archive_v378_activate_v379.json",
        {"honest_verdict": "success: archived_v378_v379_active"},
    )
    # exp4099 probe: no reranker beats vote; real proxies anti-discriminate.
    _write_json(
        results / "experiment_4099_trm_pool_verifier_discrimination_probe.json",
        {
            "honest_verdict": "complete: no_verifier_beats_trm_vote_best_K_OF_N_AGREEMENT_captured_0.0000_n62_underpowered_true",
            "verifier_beats_trm_vote": False,
            "best_reranker": "K_OF_N_AGREEMENT",
            "captured_pp_directional": 0.0,
            "per_reranker": PER_RERANKER,
            "trm_vote_pass2": 0.2742,
            "oracle_ceiling": {"pass@1": 0.371, "pass@2": 0.371},
            "n_tasks_scored": 62,
            "underpowered": True,
        },
    )
    # exp4100 TRM-RFT conditional: smoke branch, checkpoint ok, RFT not run, flagged.
    _write_json(
        results / "experiment_4100_trm_verifier_rft_conditional.json",
        {
            "honest_verdict": "complete: trm_native_trainer_checkpoint_ok_smoke_only_no_verifier_grid_discrimination_gap_0.0000",
            "flagged_adversarial": True,
            "branch_taken": "smoke",
            "trm_native_trainer_checkpoint_ok": True,
            "native_smoke": {
                "checkpoint_path": "results/experiment_4100_native_smoke/checkpoints/last.ckpt",
                "checkpoint_reload_ok": True,
            },
            "rft_vs_ablation_delta": {"delta": 0.0, "status": "not_run_no_verifier_signal", "ci95": [0.0, 0.0]},
        },
    )
    # exp4101 eleventh game SOLVED clean -> total advances to 11.
    _write_json(
        results / "experiment_4101_eleventh_game_explore_first.json",
        {
            "honest_verdict": "success: eleventh_game_solved_s5i5-18d95033_at_action_13",
            "game_solved": True,
            "total_games_solved": 11,
            "prior_total_games_solved": 10,
            "target_game": "s5i5-18d95033",
            "first_solve_at_action": 13,
            "real_env_confirmed": True,
        },
    )
    # exp4102 SOTA ingestion: methods mapped, flagged for .380.
    _write_json(
        results / "experiment_4102_sota_ingestion_trm_self_training.json",
        {
            "honest_verdict": "complete: sota_ingestion_trm_self_training_mapped",
            "flagged_for_v380": "vstar_rejected_trace_selector_for_trm_rft",
            "methods_mapped": [{"arxiv_id": "2402.06457"}, {"arxiv_id": "2203.14465"}],
        },
    )
    # exp4103 registry/gaps hygiene: regression guard passed.
    _write_json(
        results / "experiment_4103_verifier_registry_gaps_hygiene.json",
        {
            "honest_verdict": "complete: registry_gaps_reconciled_regression_guard_passed_True_trm_grid_gap_open",
            "regression_guard_passed": True,
            "registry_updated": True,
            "gaps_updated": True,
        },
    )
    # exp4104 hardware: kv260 terminal, polarfire ok, gatemate detect blocked.
    _write_json(
        results / "experiment_4104_hardware_continuity.json",
        {
            "honest_verdict": "complete: hardware_continuity_gatemate_detect_blocked_polarfire_ok_kv260_terminal",
            "per_board_reachability": {"gatemate": True, "kv260": True, "polarfire": True},
            "per_board_terminal_state": {
                "gatemate": "reachable_n16_bitstream_post_flash_detect_blocked",
                "kv260": "opportunistic_terminal_confirmed_ssh_only",
                "polarfire": "reachable_hash_verified_cpu_dispatch_recorded",
            },
            "gatemate_step_taken": "gatemate_existing_n16_bitstream_post_flash_detect_blocked",
            "polarfire_step_taken": "polarfire_hash_verified_cpu_dispatch_succeeded",
            "kv260_terminal_confirmed": True,
        },
    )
    # exp4105 capstone.
    _write_json(
        results / "experiment_4105_capstone_v379.json",
        {
            "honest_verdict": "complete: capstone_v379_honest_negative_no_grid_discrimination_games11_flagged_skipped1",
            "total_arc_games_solved": 11,
        },
    )
    return root


def run_happy(root: Path) -> dict:
    """Run the module on a synthetic repo with the pre-test gate injected green."""

    out = mod.run(root, pretest_result=GREEN, started_s=1000.0, now_s=1000.5)
    return json.loads(out.read_text(encoding="utf-8"))


# --------------------------------------------------------------------------- #
# Pure helpers
# --------------------------------------------------------------------------- #
def test_yaml_parses_true_and_false() -> None:
    assert mod.yaml_parses("a: 1\n") is True
    assert mod.yaml_parses("a: : :\n- [\n") is False


def test_yaml_single_quote_escapes() -> None:
    assert mod.yaml_single_quote("complete: it's done") == "'complete: it''s done'"


def test_duration_from_paths() -> None:
    assert mod.duration_from(None, None) == 0.0001
    assert mod.duration_from(1000.0, 1000.25) == 0.25
    # never returns <= 0 even if now precedes start
    assert mod.duration_from(1000.0, 999.0) == 0.0001


def test_duration_from_default_now(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(mod.time, "perf_counter", lambda: 1002.0)
    assert mod.duration_from(1000.0, None) == 2.0


def test_payload_checksum_ignores_existing_checksum() -> None:
    base = {"a": 1, "reproducibility_checksum": "old"}
    assert mod.payload_checksum(base) == mod.payload_checksum({"a": 1})


def test_is_sha256() -> None:
    assert mod.is_sha256("a" * 64) is True
    assert mod.is_sha256("z" * 64) is False
    assert mod.is_sha256("abc") is False
    assert mod.is_sha256(123) is False


def test_file_sha256_present_and_absent(tmp_path: Path) -> None:
    p = tmp_path / "f.json"
    p.write_text("hi", encoding="utf-8")
    assert mod.is_sha256(mod.file_sha256(p))
    assert mod.file_sha256(tmp_path / "missing.json") is None


def test_no_forbidden_markers_excludes_closestate() -> None:
    assert mod.no_forbidden_markers({"x": "fine", "v379_close_state": {"note": "GGUF trainer"}}) is True
    assert mod.no_forbidden_markers({"x": "uses CUDA"}) is False


def test_milestone_from_text_and_read_active(tmp_path: Path) -> None:
    assert mod._milestone_from_text("milestone: '2026.06.380'\n") == "2026.06.380"
    assert mod._milestone_from_text("name: foo\n") == "unknown"
    # no roadmap files at all -> unknown default
    milestone, path = mod.read_active_milestone(tmp_path)
    assert milestone == "unknown"
    assert path == "research-roadmap.yaml"


def test_read_active_milestone_next_fallback(tmp_path: Path) -> None:
    (tmp_path / "research-roadmap.yaml").write_text("name: only\n", encoding="utf-8")
    (tmp_path / "research-roadmap-next.yaml").write_text("milestone: 2026.06.380\n", encoding="utf-8")
    milestone, path = mod.read_active_milestone(tmp_path)
    assert milestone == "2026.06.380"
    assert path == "research-roadmap-next.yaml"


def test_trm_substrate_present(tmp_path: Path) -> None:
    assert mod.trm_substrate_present(tmp_path) is False
    for rel in mod.TRM_SUBSTRATE_FILES:
        f = tmp_path / rel
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text("x", encoding="utf-8")
    assert mod.trm_substrate_present(tmp_path) is True


# --------------------------------------------------------------------------- #
# research-complete dedup / append
# --------------------------------------------------------------------------- #
def test_record_id() -> None:
    assert mod._record_id("- id: 2026.06.379") == "2026.06.379"
    assert mod._record_id("  not a record") is None


def test_dedupe_collapses_duplicates() -> None:
    text = _research_complete_text(duplicates=3)
    new_text, removed, action = mod.dedupe_or_append_record(text, "2026.06.379")
    assert action == "deduped"
    assert removed == 2
    assert new_text.count("- id: 2026.06.379") == 1
    assert mod.yaml_parses(new_text)


def test_dedupe_unchanged_when_single() -> None:
    text = _research_complete_text(duplicates=1)
    new_text, removed, action = mod.dedupe_or_append_record(text, "2026.06.379")
    assert action == "unchanged"
    assert removed == 0
    assert new_text == text


def test_dedupe_appends_when_absent() -> None:
    text = "- id: 2026.06.378\n  finding: only prior\n"
    new_text, removed, action = mod.dedupe_or_append_record(text, "2026.06.379")
    assert action == "appended"
    assert removed == 0
    assert "- id: 2026.06.379" in new_text
    assert mod.yaml_parses(new_text)
    parsed = next(b for b in __import__("yaml").safe_load(new_text) if b["id"] == "2026.06.379")
    assert parsed["activation_recorded"] == "exp4106-archive-v379-activate-v380"
    assert len(parsed["tasks"]) == len(mod.V379_TASKS)


# --------------------------------------------------------------------------- #
# Artifact-record reading + classification
# --------------------------------------------------------------------------- #
def test_read_artifact_record_missing_and_bad(tmp_path: Path) -> None:
    assert mod.read_artifact_record(tmp_path / "nope.json")["exists"] is False
    bad = tmp_path / "bad.json"
    bad.write_text("{not json", encoding="utf-8")
    assert mod.read_artifact_record(bad)["exists"] is False
    not_obj = tmp_path / "list.json"
    not_obj.write_text("[1, 2, 3]", encoding="utf-8")
    assert mod.read_artifact_record(not_obj)["exists"] is False


def test_read_artifact_record_present(tmp_path: Path) -> None:
    p = tmp_path / "ok.json"
    p.write_text(json.dumps({"honest_verdict": "complete: x", "flagged_adversarial": True}), encoding="utf-8")
    rec = mod.read_artifact_record(p)
    assert rec["exists"] is True
    assert rec["flagged_adversarial"] is True
    assert rec["honest_verdict"] == "complete: x"


def test_classify_status_all_branches() -> None:
    assert mod.classify_status({"exists": False}) == "MISSING"
    assert mod.classify_status({"exists": True, "flagged_adversarial": True, "honest_verdict": "complete: x"}) == "FLAGGED"
    assert mod.classify_status({"exists": True}, kind="doc") == "OK"
    assert mod.classify_status({"exists": True, "honest_verdict": "blocked_x"}) == "BLOCKED"
    assert mod.classify_status({"exists": True, "honest_verdict": "success: x"}) == "OK"
    assert mod.classify_status({"exists": True, "honest_verdict": "weird"}) == "FAIL"


def test_fields_and_is_real_number() -> None:
    assert mod._fields({"fields": {"a": 1}}) == {"a": 1}
    assert mod._fields({"fields": "notdict"}) == {}
    assert mod._is_real_number(1) is True
    assert mod._is_real_number(1.5) is True
    assert mod._is_real_number(True) is False
    assert mod._is_real_number("1") is False


def test_per_reranker_pp() -> None:
    rec = {"fields": {"per_reranker": PER_RERANKER}}
    pp = mod._per_reranker_pp(rec)
    assert pp["AUG_INVARIANCE"] == -0.2258
    assert pp["K_OF_N_AGREEMENT"] == 0.0
    # non-dict per_reranker, and rows without numeric captured_pp, are skipped
    assert mod._per_reranker_pp({"fields": {"per_reranker": "bad"}}) == {}
    assert mod._per_reranker_pp({"fields": {"per_reranker": {"X": {"captured_pp": "nan"}}}}) == {}


# --------------------------------------------------------------------------- #
# Close-state builders
# --------------------------------------------------------------------------- #
def test_discrimination_anti_discriminates() -> None:
    rec = {
        "exists": True,
        "honest_verdict": "complete: no_verifier_beats_trm_vote",
        "fields": {
            "verifier_beats_trm_vote": False,
            "best_reranker": "K_OF_N_AGREEMENT",
            "captured_pp_directional": 0.0,
            "per_reranker": PER_RERANKER,
            "trm_vote_pass2": 0.2742,
            "oracle_ceiling": {"pass@2": 0.371},
            "n_tasks_scored": 62,
            "underpowered": True,
        },
    }
    summary = mod._discrimination(rec)
    assert summary["verifier_beats_trm_vote"] is False
    assert summary["real_verifiers_anti_discriminate"] is True
    assert summary["worst_real_verifier_captured_pp"] == -0.2258
    assert summary["k_of_n_at_k1_ties_vote"] is True
    assert summary["trm_rft_on_arc_grids_bounded"] is True
    assert summary["trm_vote_pass2"] == 0.2742
    assert summary["oracle_ceiling_pass2"] == 0.371
    assert summary["n_tasks_scored"] == 62


def test_discrimination_defaults_when_empty() -> None:
    summary = mod._discrimination({"exists": False})
    # no per_reranker rows -> falls back to the recorded defaults
    assert summary["best_captured_pp_vs_vote"] == mod.BEST_CAPTURED_PP_DEFAULT
    assert summary["worst_real_verifier_captured_pp"] == mod.WORST_REAL_VERIFIER_PP_DEFAULT
    assert summary["real_verifiers_anti_discriminate"] is True
    assert summary["trm_vote_pass2"] == mod.TRM_VOTE_PASS2_DEFAULT
    assert summary["oracle_ceiling_pass2"] == mod.ORACLE_CEILING_PASS2_DEFAULT
    assert summary["n_tasks_scored"] == mod.DISCRIMINATION_N_TASKS_DEFAULT
    # k_of_n missing from pp -> default 0.0 ties vote
    assert summary["k_of_n_at_k1_ties_vote"] is True


def test_trm_rft_smoke() -> None:
    rec = {
        "exists": True,
        "honest_verdict": "complete: trm_native_trainer_checkpoint_ok_smoke_only",
        "flagged_adversarial": True,
        "fields": {
            "branch_taken": "smoke",
            "trm_native_trainer_checkpoint_ok": True,
            "native_smoke": {"checkpoint_path": "ck/last.ckpt", "checkpoint_reload_ok": True},
            "rft_vs_ablation_delta": {"delta": 0.0, "status": "not_run_no_verifier_signal"},
        },
    }
    summary = mod._trm_rft_smoke(rec)
    assert summary["branch_taken"] == "smoke"
    assert summary["trm_native_trainer_checkpoint_ok"] is True
    assert summary["checkpoint_reload_ok"] is True
    assert summary["checkpoint_path"] == "ck/last.ckpt"
    assert summary["rft_ran"] is False
    assert summary["flagged_adversarial"] is True
    assert summary["skipped_from_aggregation"] is True


def test_trm_rft_smoke_rft_ran_branch() -> None:
    # If a future run actually executes RFT, rft_ran flips True.
    rec = {
        "exists": True,
        "flagged_adversarial": False,
        "fields": {
            "branch_taken": "rft",
            "trm_native_trainer_checkpoint_ok": True,
            "rft_vs_ablation_delta": {"delta": 0.03, "status": "ran"},
        },
    }
    summary = mod._trm_rft_smoke(rec)
    assert summary["rft_ran"] is True
    assert summary["rft_vs_ablation_delta"] == 0.03


def test_trm_rft_smoke_missing() -> None:
    summary = mod._trm_rft_smoke({"exists": False})
    assert summary["trm_native_trainer_checkpoint_ok"] is False
    assert summary["rft_ran"] is False
    assert summary["flagged_adversarial"] is False


def test_accuracy_eleventh_game() -> None:
    summary = mod._accuracy(
        {
            "exists": True,
            "honest_verdict": "success: eleventh_game_solved",
            "fields": {
                "game_solved": True,
                "total_games_solved": 11,
                "prior_total_games_solved": 10,
                "target_game": "s5i5-18d95033",
                "first_solve_at_action": 13,
                "real_env_confirmed": True,
            },
        },
        {"exists": True, "fields": {"total_arc_games_solved": 11}},
    )
    assert summary["total_games_solved"] == 11
    assert summary["eleventh_game_solved"] is True
    assert summary["monotonic_no_regression"] is True
    assert summary["eleventh_game"] == "s5i5-18d95033"


def test_accuracy_falls_back_to_capstone_and_prior() -> None:
    # eleventh-game record lacks total but has prior + solved -> prior+1.
    summary = mod._accuracy(
        {"exists": True, "honest_verdict": "success: x", "fields": {"game_solved": True, "prior_total_games_solved": 10}},
        {"exists": False},
    )
    assert summary["total_games_solved"] == 11
    # nothing anywhere -> default 11
    summary2 = mod._accuracy({"exists": False}, {"exists": False})
    assert summary2["total_games_solved"] == mod.TOTAL_GAMES_SOLVED_DEFAULT
    assert summary2["eleventh_game_solved"] is False


def test_accuracy_capstone_total() -> None:
    summary = mod._accuracy({"exists": False}, {"exists": True, "fields": {"total_arc_games_solved": 11}})
    assert summary["total_games_solved"] == 11
    # capstone games_solved_total alias is also honored
    summary2 = mod._accuracy({"exists": False}, {"exists": True, "fields": {"games_solved_total": 11}})
    assert summary2["total_games_solved"] == 11


def test_sota() -> None:
    summary = mod._sota(
        {
            "exists": True,
            "honest_verdict": "complete: sota",
            "fields": {
                "flagged_for_v380": "vstar_rejected_trace_selector_for_trm_rft",
                "methods_mapped": [{"arxiv_id": "1"}, {"arxiv_id": "2"}, {"arxiv_id": "3"}],
            },
        }
    )
    assert summary["measured"] is True
    assert summary["flagged_for_v380"] == "vstar_rejected_trace_selector_for_trm_rft"
    assert summary["methods_mapped_count"] == 3


def test_sota_missing_and_non_list() -> None:
    assert mod._sota({"exists": False})["methods_mapped_count"] == 0
    # a string methods_mapped does not count as a list of methods
    assert mod._sota({"exists": True, "fields": {"methods_mapped": "x"}})["methods_mapped_count"] == 0


def test_hygiene() -> None:
    summary = mod._hygiene(
        {
            "exists": True,
            "honest_verdict": "complete: registry_gaps_reconciled",
            "fields": {"regression_guard_passed": True, "registry_updated": True, "gaps_updated": True},
        }
    )
    assert summary["regression_guard_passed"] is True
    assert summary["trm_grid_discrimination_gap_open"] is True


def test_hardware() -> None:
    summary = mod._hardware(
        {
            "exists": True,
            "honest_verdict": "complete: hw",
            "fields": {
                "per_board_reachability": {"gatemate": True, "kv260": True, "polarfire": True},
                "per_board_terminal_state": {"kv260": "terminal"},
                "gatemate_step_taken": "gatemate_detect_blocked",
                "polarfire_step_taken": "polarfire_ok",
                "kv260_terminal_confirmed": True,
            },
        }
    )
    assert summary["included"] is True
    assert summary["kv260_terminal"] is True
    assert summary["per_board_reachability"]["gatemate"] is True


def test_flagged_skipped() -> None:
    summary = mod._flagged_skipped(
        {"4100": {"exists": True, "flagged_adversarial": True, "honest_verdict": "complete: x"}}
    )
    assert summary["count"] == 1
    assert summary["experiment_ids"] == ["4100"]
    assert summary["skipped"][0]["flagged_adversarial"] is True


# --------------------------------------------------------------------------- #
# Full close-state + provenance
# --------------------------------------------------------------------------- #
def test_build_close_state(tmp_path: Path) -> None:
    root = make_repo(tmp_path)
    records = mod.read_v379_records(root)
    cs = mod.build_v379_close_state(records)
    assert cs["verifier_anti_discriminates_on_trm_grids"] is True
    assert cs["discrimination"]["verifier_beats_trm_vote"] is False
    assert cs["discrimination"]["real_verifiers_anti_discriminate"] is True
    assert cs["trm_rft"]["trm_native_trainer_checkpoint_ok"] is True
    assert cs["trm_rft"]["rft_ran"] is False
    assert cs["total_games_solved"] == 11
    assert cs["flagged_count"] == 1
    assert cs["status_counts"]["FLAGGED"] == 1  # exp4100
    assert cs["status_counts"]["OK"] == 7  # the other 7 tasks
    assert cs["status_counts"]["BLOCKED"] == 0
    assert cs["status_counts"]["MISSING"] == 0
    assert "anti-discriminat" in cs["headline"].lower()
    assert "sudoku" in cs["pivot_rationale"].lower()


def test_read_v379_records_doc_branch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # No real .379 task is a doc deliverable, but read_v379_records keeps the
    # general doc branch (a .md note has no JSON verdict). Exercise it directly by
    # swapping in a single doc-kind task: presence of the file drives ``exists``.
    doc_task = {"exp_id": "9999", "id": "exp9999-doc", "deliverable": "docs/note.md", "kind": "doc"}
    monkeypatch.setattr(mod, "V379_TASKS", (doc_task,))
    records = mod.read_v379_records(tmp_path)
    assert records["9999"]["exists"] is False
    (tmp_path / "docs").mkdir(parents=True, exist_ok=True)
    (tmp_path / "docs" / "note.md").write_text("# note\n", encoding="utf-8")
    records2 = mod.read_v379_records(tmp_path)
    assert records2["9999"]["exists"] is True
    assert records2["9999"]["honest_verdict"] == ""


def test_build_cited_upstream(tmp_path: Path) -> None:
    root = make_repo(tmp_path)
    cited = mod.build_cited_upstream(root)
    assert len(cited) == len(mod.V379_TASKS)
    by_id = {c["experiment_id"]: c for c in cited}
    assert mod.is_sha256(by_id["4099"]["sha256"])  # present artifact hashed
    # all .379 artifacts exist in the fixture
    assert all(c["sha256"] is not None for c in cited)


def test_smart_subset_helpers(tmp_path: Path) -> None:
    root = make_repo(tmp_path)
    targets = mod.smart_subset_targets(root)
    assert "tests/python/test_pipeline_extract.py" in targets
    cmd = mod.smart_subset_command(targets)
    assert cmd[0] == str(mod.PYTEST_BIN)
    assert "--no-cov" in cmd
    # empty repo -> falls back to the first core target string
    assert mod.smart_subset_targets(tmp_path / "empty") == [mod.CORE_SMART_SUBSET[0]]


def test_run_smart_subset_executes(tmp_path: Path) -> None:
    # _run_command on a bogus binary returns a non-zero result without raising.
    res = mod._run_command(["definitely-not-a-real-binary-xyz"], tmp_path)
    assert isinstance(res, mod.CommandResult)
    assert res.exit_code != 0


# --------------------------------------------------------------------------- #
# run() happy path
# --------------------------------------------------------------------------- #
def test_run_happy_path(tmp_path: Path) -> None:
    root = make_repo(tmp_path)
    art = run_happy(root)
    assert art["honest_verdict"].startswith("success:")
    assert "verifier_anti_discriminates_on_trm_arc_grids" in art["honest_verdict"]
    assert "trm_native_trainer_checkpoint_ok" in art["honest_verdict"]
    assert "pivot_to_executable_sudoku_verifier" in art["honest_verdict"]
    assert art["archived_milestone"] == "2026.06.379"
    assert art["activated_milestone"] == "2026.06.380"
    assert art["active_milestone_confirmed"] == "2026.06.380"
    assert art["trm_substrate_present"] is True
    assert art["pretest_suite_green"] is True
    assert art["total_games_solved"] == 11
    assert art["flagged_count"] == 1
    assert art["research_complete_record_action"] == "deduped"
    assert art["research_complete_duplicates_removed"] == 2
    assert art["inference_substrate"] == "aggregation_from_upstream_artifacts"
    assert mod.is_sha256(art["reproducibility_checksum"])
    # the dedup actually rewrote research-complete.yaml to a single .379 record
    text = (root / "research-complete.yaml").read_text(encoding="utf-8")
    assert text.count("- id: 2026.06.379") == 1
    # the artifact validates under the module's own validator
    mod.validate_artifact(art)
    # every required field has a principle annotation
    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert field in art["field_principles"]


def test_run_real_pretest_branch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Exercise the production path where pretest_result is None (run_smart_subset
    # is called) by stubbing it green.
    root = make_repo(tmp_path)
    monkeypatch.setattr(mod, "run_smart_subset", lambda r: GREEN)
    out = mod.run(root, started_s=1.0, now_s=1.1)
    art = json.loads(out.read_text(encoding="utf-8"))
    assert art["honest_verdict"].startswith("success:")
    assert art["preconditions_checked"]["pretest_suite_green"] is True


# --------------------------------------------------------------------------- #
# run() blocked paths
# --------------------------------------------------------------------------- #
def test_run_blocked_missing_yaml(tmp_path: Path) -> None:
    out = mod.run(tmp_path, pretest_result=GREEN)
    art = json.loads(out.read_text(encoding="utf-8"))
    assert art["honest_verdict"] == "blocked_research_complete_yaml_poison_missing"
    assert art["research_complete_yaml_parses"] is False


def test_run_blocked_poison_yaml(tmp_path: Path) -> None:
    (tmp_path / "research-complete.yaml").write_text("a: : :\n- [\n", encoding="utf-8")
    out = mod.run(tmp_path, pretest_result=GREEN)
    art = json.loads(out.read_text(encoding="utf-8"))
    assert art["honest_verdict"] == "blocked_research_complete_yaml_poison"


def test_run_blocked_v380_not_active(tmp_path: Path) -> None:
    root = make_repo(tmp_path)
    (root / "research-roadmap.yaml").write_text('milestone: "2026.06.379"\n', encoding="utf-8")
    out = mod.run(root, pretest_result=GREEN)
    art = json.loads(out.read_text(encoding="utf-8"))
    assert art["honest_verdict"] == "blocked_v380_not_active"
    assert art["research_complete_yaml_parses"] is True


def test_run_blocked_trm_substrate_missing(tmp_path: Path) -> None:
    root = make_repo(tmp_path)
    (root / "nano-trm" / "src" / "nn" / "train.py").unlink()
    out = mod.run(root, pretest_result=GREEN)
    art = json.loads(out.read_text(encoding="utf-8"))
    assert art["honest_verdict"] == "blocked_trm_substrate_missing"
    assert art["trm_substrate_present"] is False


def test_run_blocked_manifest_missing(tmp_path: Path) -> None:
    root = make_repo(tmp_path)
    (root / "ops" / "exclusion_manifest.yaml").unlink()
    out = mod.run(root, pretest_result=GREEN)
    art = json.loads(out.read_text(encoding="utf-8"))
    assert art["honest_verdict"] == "blocked_exclusion_manifest_missing"


def test_run_blocked_manifest_poison(tmp_path: Path) -> None:
    root = make_repo(tmp_path)
    (root / "ops" / "exclusion_manifest.yaml").write_text("a: : :\n- [\n", encoding="utf-8")
    out = mod.run(root, pretest_result=GREEN)
    art = json.loads(out.read_text(encoding="utf-8"))
    assert art["honest_verdict"] == "blocked_exclusion_manifest_yaml_poison"


def test_run_blocked_pretest_red(tmp_path: Path) -> None:
    root = make_repo(tmp_path)
    out = mod.run(root, pretest_result=RED)
    art = json.loads(out.read_text(encoding="utf-8"))
    assert art["honest_verdict"] == "blocked_pretest_suite_not_green"
    assert art["preconditions_checked"]["pretest_suite_green"] is False
    # close-state is still recorded honestly on the blocked path
    assert art["v379_close_state"]["total_games_solved"] == 11


# --------------------------------------------------------------------------- #
# validate_artifact rejections (guards against laundering a null into a win)
# --------------------------------------------------------------------------- #
def _good_artifact(tmp_path: Path) -> dict:
    return run_happy(make_repo(tmp_path))


def test_validate_rejects_missing_field(tmp_path: Path) -> None:
    art = _good_artifact(tmp_path)
    del art["total_games_solved"]
    with pytest.raises(ValueError, match="missing required fields"):
        mod.validate_artifact(art)


def test_validate_rejects_non_terminal_verdict(tmp_path: Path) -> None:
    art = _good_artifact(tmp_path)
    art["honest_verdict"] = "archived something"
    with pytest.raises(ValueError, match="terminal prefix"):
        mod.validate_artifact(art)


def test_validate_rejects_verifier_beats_vote_true(tmp_path: Path) -> None:
    art = _good_artifact(tmp_path)
    art["v379_close_state"]["discrimination"]["verifier_beats_trm_vote"] = True
    with pytest.raises(ValueError, match="verifier_beats_trm_vote"):
        mod.validate_artifact(art)


def test_validate_rejects_no_anti_discrimination(tmp_path: Path) -> None:
    art = _good_artifact(tmp_path)
    art["v379_close_state"]["discrimination"]["real_verifiers_anti_discriminate"] = False
    with pytest.raises(ValueError, match="real_verifiers_anti_discriminate"):
        mod.validate_artifact(art)


def test_validate_rejects_rft_ran_true(tmp_path: Path) -> None:
    art = _good_artifact(tmp_path)
    art["v379_close_state"]["trm_rft"]["rft_ran"] = True
    with pytest.raises(ValueError, match="rft_ran=False"):
        mod.validate_artifact(art)


def test_validate_rejects_checkpoint_not_ok(tmp_path: Path) -> None:
    art = _good_artifact(tmp_path)
    art["v379_close_state"]["trm_rft"]["trm_native_trainer_checkpoint_ok"] = False
    with pytest.raises(ValueError, match="trm_native_trainer_checkpoint_ok"):
        mod.validate_artifact(art)


def test_validate_rejects_trm_rft_not_flagged(tmp_path: Path) -> None:
    art = _good_artifact(tmp_path)
    art["v379_close_state"]["trm_rft"]["flagged_adversarial"] = False
    with pytest.raises(ValueError, match="flagged_adversarial=True"):
        mod.validate_artifact(art)


def test_validate_rejects_wrong_games(tmp_path: Path) -> None:
    art = _good_artifact(tmp_path)
    art["v379_close_state"]["accuracy"]["total_games_solved"] = 10
    with pytest.raises(ValueError, match="must be 11"):
        mod.validate_artifact(art)


def test_validate_rejects_wrong_flagged_count(tmp_path: Path) -> None:
    art = _good_artifact(tmp_path)
    art["flagged_count"] = 2
    art["v379_close_state"]["flagged_count"] = 2
    art["v379_close_state"]["flagged_skipped"]["count"] = 2
    with pytest.raises(ValueError, match="flagged"):
        mod.validate_artifact(art)


def test_validate_rejects_bad_milestones(tmp_path: Path) -> None:
    art = _good_artifact(tmp_path)
    art["archived_milestone"] = "2026.06.999"
    with pytest.raises(ValueError, match="archived milestone"):
        mod.validate_artifact(art)
    art2 = _good_artifact(tmp_path)
    art2["activated_milestone"] = "2026.06.999"
    with pytest.raises(ValueError, match="activated milestone"):
        mod.validate_artifact(art2)


def test_validate_rejects_checksum_mismatch(tmp_path: Path) -> None:
    art = _good_artifact(tmp_path)
    art["reproducibility_checksum"] = "a" * 64
    with pytest.raises(ValueError, match="does not match"):
        mod.validate_artifact(art)


def test_validate_rejects_model_specs(tmp_path: Path) -> None:
    art = _good_artifact(tmp_path)
    art["model_specs"] = ["gguf"]
    # recompute checksum so we hit the model_specs guard, not the checksum guard
    art["reproducibility_checksum"] = mod.payload_checksum(art)
    with pytest.raises(ValueError, match="model_specs"):
        mod.validate_artifact(art)


def test_build_complete_artifact_validates(tmp_path: Path) -> None:
    root = make_repo(tmp_path)
    records = mod.read_v379_records(root)
    cs = mod.build_v379_close_state(records)
    payload = mod.build_complete_artifact(
        v379_close_state=cs,
        total_games_solved=cs["total_games_solved"],
        flagged_count=cs["flagged_count"],
        preconditions_checked={"ok": True},
        duration_s=0.5,
        active_roadmap_path="research-roadmap.yaml",
        research_complete_record_action="deduped",
        research_complete_duplicates_removed=2,
        cited_upstream_artifacts=mod.build_cited_upstream(root),
    )
    assert payload["honest_verdict"].startswith("success:")


def test_build_blocked_artifact_shape() -> None:
    art = mod.build_blocked_artifact(
        "blocked_x",
        preconditions_checked={"p": 1},
        duration_s=0.1,
        active_milestone_confirmed="",
        active_roadmap_path="research-roadmap.yaml",
    )
    assert art["honest_verdict"] == "blocked_x"
    assert art["trm_substrate_present"] is False
    assert mod.is_sha256(art["reproducibility_checksum"])


def test_terminal_verdict_shape() -> None:
    v = mod.terminal_verdict({"accuracy": {"total_games_solved": 11}, "flagged_skipped": {"count": 1}})
    assert v.startswith("success:")
    assert "games11" in v
    assert "1_flagged_skipped" in v


# --------------------------------------------------------------------------- #
# Exhaustive validate_artifact rejection guards (the anti-laundering wall)
# --------------------------------------------------------------------------- #
def _mutators() -> list[tuple[str, "object"]]:
    """Return (match, mutator) pairs -- each trips exactly one validate guard."""

    def set_path(obj: dict, path: list[str], value: object) -> None:
        cur = obj
        for key in path[:-1]:
            cur = cur[key]
        cur[path[-1]] = value

    cases: list[tuple[str, object]] = [
        ("field_principles must be a mapping", lambda a: a.__setitem__("field_principles", "x")),
        ("missing field principles", lambda a: a.__setitem__("field_principles", {})),
        ("research-complete YAML parse", lambda a: a.__setitem__("research_complete_yaml_parses", False)),
        ("manifest parse must be true", lambda a: a.__setitem__("exclusion_manifest_parses", False)),
        ("TRM substrate must be present", lambda a: a.__setitem__("trm_substrate_present", False)),
        ("pretest suite must be green", lambda a: a.__setitem__("pretest_suite_green", False)),
        ("active milestone must be confirmed", lambda a: a.__setitem__("active_milestone_confirmed", "x")),
        ("n_tasks_archived must match", lambda a: a.__setitem__("n_tasks_archived", 99)),
        ("non-empty dict", lambda a: a.__setitem__("v379_close_state", {})),
        ("must record per_task_status", lambda a: a.__setitem__("v379_close_state", {"x": 1})),
        ("verifier_anti_discriminates_on_trm_grids=True",
         lambda a: set_path(a, ["v379_close_state", "verifier_anti_discriminates_on_trm_grids"], False)),
        ("must record the discrimination summary",
         lambda a: set_path(a, ["v379_close_state", "discrimination"], "x")),
        ("trm_rft_on_arc_grids_bounded=True",
         lambda a: set_path(a, ["v379_close_state", "discrimination", "trm_rft_on_arc_grids_bounded"], False)),
        ("must record the trm_rft summary",
         lambda a: set_path(a, ["v379_close_state", "trm_rft"], "x")),
        ("must record accuracy", lambda a: set_path(a, ["v379_close_state", "accuracy"], "x")),
        ("must be monotonic",
         lambda a: set_path(a, ["v379_close_state", "accuracy", "monotonic_no_regression"], False)),
        ("top-level total_games_solved must be 11", lambda a: a.__setitem__("total_games_solved", 10)),
        ("flagged_count must match the close-state", lambda a: a.__setitem__("flagged_count", 2)),
        ("duration_s must be a positive", lambda a: a.__setitem__("duration_s", -1.0)),
        ("inference substrate must be aggregation", lambda a: a.__setitem__("inference_substrate", "live_llm_inference")),
        ("cited_upstream_artifacts must be a list", lambda a: a.__setitem__("cited_upstream_artifacts", "x")),
        ("must not copy compute-bound markers", lambda a: a.__setitem__("leaked", "uses CUDA here")),
        ("reproducibility_checksum must be sha256", lambda a: a.__setitem__("reproducibility_checksum", "tooshort")),
    ]
    return cases


def test_validate_artifact_rejection_guards(tmp_path: Path) -> None:
    good = _good_artifact(tmp_path)
    for match, mutate in _mutators():
        art = copy.deepcopy(good)
        mutate(art)
        with pytest.raises(ValueError, match=match):
            mod.validate_artifact(art)


def test_validate_rejects_flagged_count_not_one(tmp_path: Path) -> None:
    # top-level flagged_count matches close-state but both are != 1 (the
    # flagged_skipped.count stays 1 so the earlier guard passes).
    art = _good_artifact(tmp_path)
    art["flagged_count"] = 2
    art["v379_close_state"]["flagged_count"] = 2
    with pytest.raises(ValueError, match="must be 1"):
        mod.validate_artifact(art)


# --------------------------------------------------------------------------- #
# Remaining subprocess + defensive run() branches
# --------------------------------------------------------------------------- #
def test_run_command_success_branch(tmp_path: Path) -> None:
    res = mod._run_command(["true"], tmp_path)
    assert res.exit_code == 0


def test_run_smart_subset_uses_run_command(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(mod, "_run_command", lambda cmd, root: GREEN)
    assert mod.run_smart_subset(tmp_path).exit_code == 0


def test_run_blocked_edit_invalid(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = make_repo(tmp_path)
    monkeypatch.setattr(mod, "dedupe_or_append_record", lambda text, mid: ("a: : :\n- [", 0, "appended"))
    out = mod.run(root, pretest_result=GREEN)
    art = json.loads(out.read_text(encoding="utf-8"))
    assert art["honest_verdict"] == "blocked_research_complete_edit_invalid"


def test_run_blocked_poison_after_edit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = make_repo(tmp_path)
    calls = {"n": 0}

    def fake_parses(text: str) -> bool:
        calls["n"] += 1
        # parses_before (1) and new_text (2) are valid; the after-edit re-read (3) is poisoned.
        return calls["n"] != 3

    monkeypatch.setattr(mod, "yaml_parses", fake_parses)
    out = mod.run(root, pretest_result=GREEN)
    art = json.loads(out.read_text(encoding="utf-8"))
    assert art["honest_verdict"] == "blocked_research_complete_yaml_poison_after_edit"
