"""Tests for the Exp 4098 .378 archive / .379 activation record-only module.

Spec refs: REQ-REPORT-4098, SCENARIO-REPORT-4098,
SCENARIO-REPORT-4098-BLOCKED-YAML.

These tests exercise the disciplined milestone-transition module end to end on a
synthetic repo fixture (no live model, no real conductor), plus every pure helper
and every blocked-path branch. The load-bearing assertions:

* the `.378` close-state records the LLM-LoRA verifier-as-reward TRAINING route as
  RETIRED (an honest dead-end, not a paper-over) and the verifier-label training
  signal as UNMEASURED;
* the precision rescue is recorded as carried by DEMO-PERFECT ALONE
  (``ensemble_added_value_over_demo_perfect_alone == False``) -- NOT as "the
  ensemble rescued precision" -- because k_of_n_agreement at k=1 is a no-op, the
  invariance filter cratered recall, agreement at k>=2 certified 0, and
  min_hamming was worse;
* accuracy advances to 10 games (the tenth game solved);
* the duplicate `.378` history records collapse to one (the 28-copy dedup);
* the colon-poison guard keeps research-complete.yaml parseable;
* the terminal artifact carries every required principle-annotated field with a
  ``success:`` prefix.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from carnot.reporting import archive_v378_activate_v379_4098 as mod


GREEN = mod.CommandResult(command=["pytest"], exit_code=0, stdout="", stderr="")
RED = mod.CommandResult(command=["pytest"], exit_code=1, stdout="FAILED x", stderr="")


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


# The Exp 4087 precision-rescue frontier, trimmed to the load-bearing rows. The
# best operating point is k_of_n_agreement at k=1 (== no filter), whose numbers
# are byte-identical to demo_perfect alone -> the gate was carried by the
# demo-fit primitive, not the ensemble.
PRECISION_FRONTIER = [
    {"filter_stack": "demo_perfect", "n_certified": 17, "precision": 0.8824, "recall": 0.7143, "threshold": "k=1"},
    {"filter_stack": "demo_perfect+invariance", "n_certified": 7, "precision": 0.8571, "recall": 0.2857, "threshold": "required"},
    {"filter_stack": "k_of_n_agreement", "n_certified": 17, "precision": 0.8824, "recall": 0.7143, "threshold": "k=1"},
    {"filter_stack": "k_of_n_agreement", "n_certified": 0, "precision": 0.0, "recall": 0.0, "threshold": "k=2"},
    {"filter_stack": "graded_min_hamming", "n_certified": 20, "precision": 0.75, "recall": 0.7143, "threshold": "tau=0.0000"},
]


def _research_complete_text(*, duplicates: int = 1) -> str:
    """A research-complete.yaml holding ``duplicates`` copies of the .378 record.

    The real repo accumulated 28 copies from interrupted runs, so the common
    action for this module is ``deduped``.
    """

    head = "- id: 2026.06.377\n  finding: prior milestone\n"
    block = (
        "- id: 2026.06.378\n"
        "  title: 'verifier-as-reward TRM pivot'\n"
        "  completed: '2026-06-12'\n"
        "  finding: See conductor log for per-experiment results.\n"
        "  tasks:\n"
        "  - id: exp4086-archive-v377-activate-v378\n"
        "    result: OK (conductor)\n"
    )
    return head + block * duplicates


def make_repo(tmp_path: Path) -> Path:
    """Build a synthetic repo mirroring the real .378 artifacts."""

    root = tmp_path
    (root / "research-complete.yaml").write_text(_research_complete_text(duplicates=3), encoding="utf-8")
    (root / "ops").mkdir(parents=True, exist_ok=True)
    (root / "ops" / "exclusion_manifest.yaml").write_text(
        "retired:\n- experiment_id: 2091\n  reason: gemini bail-out\n", encoding="utf-8"
    )
    (root / "research-roadmap.yaml").write_text(
        'milestone: "2026.06.379"\nname: v379\n', encoding="utf-8"
    )
    # TRM substrate (precondition C).
    (root / "nano-trm" / "src").mkdir(parents=True, exist_ok=True)
    (root / "nano-trm" / "src" / "arc_evaluator.py").write_text("# trm\n", encoding="utf-8")
    (root / "scripts" / "experiments").mkdir(parents=True, exist_ok=True)
    (root / "scripts" / "experiments" / "trm_arc_eval_harness.py").write_text("def main():\n    pass\n", encoding="utf-8")
    # Core smart-subset targets so run_smart_subset has files to point at.
    (root / "tests" / "python").mkdir(parents=True, exist_ok=True)
    (root / "tests" / "python" / "test_pipeline_extract.py").write_text("def test_x():\n    assert True\n", encoding="utf-8")
    (root / "tests" / "python" / "test_docs.py").write_text("def test_y():\n    assert True\n", encoding="utf-8")

    results = root / "results"
    _write_json(
        results / "experiment_4086_archive_v377_activate_v378.json",
        {"honest_verdict": "success: archived_v377_v378"},
    )
    # exp4087 precision rescue: succeeded=true BUT carried by demo-perfect alone.
    _write_json(
        results / "experiment_4087_certification_precision_rescue.json",
        {
            "honest_verdict": "complete: precision_rescue_succeeded_best_0.8824_at_recall_0.7143",
            "precision_rescue_succeeded": True,
            "best_operating_point": {
                "filter_stack": "k_of_n_agreement",
                "n_certified": 17,
                "precision": 0.8824,
                "recall": 0.7143,
                "threshold": "k=1",
            },
            "frontier": PRECISION_FRONTIER,
        },
    )
    # exp4088 corpus build: trl/peft trainer produced no checkpoint.
    _write_json(
        results / "experiment_4088_verifier_reward_rft_corpus_build.json",
        {"honest_verdict": "blocked_lora_smoke_checkpoints"},
    )
    # exp4089 train: cascade-blocked at the conductor pre-gate.
    _write_json(
        results / "experiment_4089_verifier_reward_rft_train.json",
        {"honest_verdict": "blocked_gate_check_failed"},
    )
    # exp4090 / exp4091 deliberately ABSENT -> MISSING (no held-out eval, no sanity).
    # exp4092 tenth game SOLVED clean -> total advances to 10.
    _write_json(
        results / "experiment_4092_tenth_game_explore_first.json",
        {
            "honest_verdict": "success: tenth_game_solved_r11l-495a7899_at_action_4",
            "game_solved": True,
            "total_games_solved": 10,
            "prior_total_games_solved": 9,
            "target_game": "r11l-495a7899",
            "first_solve_at_action": 4,
            "real_env_confirmed": True,
        },
    )
    # exp4093 off-ARC demo-fit transfer: primitive transfers, marginal filter lift.
    _write_json(
        results / "experiment_4093_offarc_demofit_precision_transfer.json",
        {
            "honest_verdict": "complete: offarc_demofit_precision_0.96_filter_raises_to_0.96",
            "demofit_precision_raw": 0.956186,
            "demofit_precision_filtered": 0.96049,
            "domain_general_precision_floor": 0.68,
            "primitive_is_domain_general": True,
        },
    )
    # exp4094 SOTA-ingestion note (doc deliverable).
    note = root / "docs" / "research-notes" / "sota-ingestion-precision-calibration-2026-06-12.md"
    note.parent.mkdir(parents=True, exist_ok=True)
    note.write_text("# sota ingestion\n", encoding="utf-8")
    # exp4095 FLAGGED: duration too short -> skipped from aggregation.
    _write_json(
        results / "experiment_4095_verifier_registry_gaps_hygiene.json",
        {
            "honest_verdict": "complete: gap4_arc1_reproduced_True_precision_rescue_succeeded",
            "flagged_adversarial": True,
        },
    )
    # exp4096 hardware: kv260 terminal, polarfire ok, gatemate unreachable.
    _write_json(
        results / "experiment_4096_hardware_continuity.json",
        {
            "honest_verdict": "complete: hardware_continuity_gatemate_blocked_polarfire_ok_kv260_terminal",
            "per_board_reachability": {"gatemate": False, "kv260": True, "polarfire": True},
            "per_board_terminal_state": {"kv260": "opportunistic_terminal_confirmed_ssh_only"},
            "gatemate_step_taken": "blocked_gatemate_unreachable",
            "polarfire_step_taken": "polarfire_hash_verified_cpu_dispatch_succeeded",
            "kv260_terminal_confirmed": True,
        },
    )
    # exp4097 capstone.
    _write_json(
        results / "experiment_4097_capstone_v378.json",
        {
            "honest_verdict": "complete: capstone_v378_precision_rescued_0.8824_games10",
            "games_solved_total": 10,
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
    assert mod.no_forbidden_markers({"x": "fine", "v378_close_state": {"note": "GGUF trainer"}}) is True
    assert mod.no_forbidden_markers({"x": "uses CUDA"}) is False


def test_milestone_from_text_and_read_active(tmp_path: Path) -> None:
    assert mod._milestone_from_text("milestone: '2026.06.379'\n") == "2026.06.379"
    assert mod._milestone_from_text("name: foo\n") == "unknown"
    # no roadmap files at all -> unknown default
    milestone, path = mod.read_active_milestone(tmp_path)
    assert milestone == "unknown"
    assert path == "research-roadmap.yaml"


def test_read_active_milestone_next_fallback(tmp_path: Path) -> None:
    (tmp_path / "research-roadmap.yaml").write_text("name: only\n", encoding="utf-8")
    (tmp_path / "research-roadmap-next.yaml").write_text("milestone: 2026.06.379\n", encoding="utf-8")
    milestone, path = mod.read_active_milestone(tmp_path)
    assert milestone == "2026.06.379"
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
    assert mod._record_id("- id: 2026.06.378") == "2026.06.378"
    assert mod._record_id("  not a record") is None


def test_dedupe_collapses_duplicates() -> None:
    text = _research_complete_text(duplicates=3)
    new_text, removed, action = mod.dedupe_or_append_record(text, "2026.06.378")
    assert action == "deduped"
    assert removed == 2
    assert new_text.count("- id: 2026.06.378") == 1
    assert mod.yaml_parses(new_text)


def test_dedupe_unchanged_when_single() -> None:
    text = _research_complete_text(duplicates=1)
    new_text, removed, action = mod.dedupe_or_append_record(text, "2026.06.378")
    assert action == "unchanged"
    assert removed == 0
    assert new_text == text


def test_dedupe_appends_when_absent() -> None:
    text = "- id: 2026.06.377\n  finding: only prior\n"
    new_text, removed, action = mod.dedupe_or_append_record(text, "2026.06.378")
    assert action == "appended"
    assert removed == 0
    assert "- id: 2026.06.378" in new_text
    assert mod.yaml_parses(new_text)
    parsed = next(b for b in __import__("yaml").safe_load(new_text) if b["id"] == "2026.06.378")
    assert parsed["activation_recorded"] == "exp4098-archive-v378-activate-v379"
    assert len(parsed["tasks"]) == len(mod.V378_TASKS)


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


def test_frontier_rows_and_find() -> None:
    rec = {"fields": {"frontier": PRECISION_FRONTIER + ["notdict"]}}
    rows = mod._frontier_rows(rec)
    assert len(rows) == len(PRECISION_FRONTIER)
    assert mod._find_row(rows, "demo_perfect")["precision"] == 0.8824
    assert mod._find_row(rows, "k_of_n_agreement", "k=2")["n_certified"] == 0
    assert mod._find_row(rows, "nonexistent") is None
    assert mod._frontier_rows({"fields": {"frontier": "bad"}}) == []


# --------------------------------------------------------------------------- #
# Close-state builders
# --------------------------------------------------------------------------- #
def test_lora_training_retired() -> None:
    summary = mod._lora_training_retired(
        {"exists": True, "honest_verdict": "blocked_lora_smoke_checkpoints"},
        {"exists": True, "honest_verdict": "blocked_gate_check_failed"},
        {"exists": False},
        {"exists": False},
    )
    assert summary["retired"] is True
    assert summary["trainer_produced_checkpoint"] is False
    assert summary["rft_eval_measured"] is False
    assert summary["sudoku_sanity_measured"] is False
    assert summary["verifier_label_training_signal_measured"] is False
    assert "nano-TRM" in summary["v379_pivot"]


def test_precision_rescue_honest_demo_perfect_alone() -> None:
    rec = {
        "exists": True,
        "fields": {
            "precision_rescue_succeeded": True,
            "best_operating_point": {
                "filter_stack": "k_of_n_agreement",
                "n_certified": 17,
                "precision": 0.8824,
                "recall": 0.7143,
                "threshold": "k=1",
            },
            "frontier": PRECISION_FRONTIER,
        },
    }
    summary = mod._precision_rescue_honest(rec)
    assert summary["succeeded_flag_recorded"] is True
    assert summary["gate_cleared_floor"] is True
    assert summary["k_of_n_agreement_k1_is_no_filter"] is True
    assert summary["winning_stack_is_demo_perfect_alone"] is True
    assert summary["ensemble_added_value_over_demo_perfect_alone"] is False
    assert summary["invariance_cratered_recall"] is True
    assert summary["invariance_recall"] == 0.2857
    assert summary["agreement_k_ge_2_certified"] == 0
    assert summary["min_hamming_best_precision_at_winning_recall"] == 0.75
    assert summary["min_hamming_worse_than_demo_perfect"] is True


def test_precision_rescue_honest_demo_perfect_literal_best() -> None:
    # When the best stack is literally demo_perfect, it is still demo-perfect alone.
    rec = {
        "exists": True,
        "fields": {
            "precision_rescue_succeeded": True,
            "best_operating_point": {"filter_stack": "demo_perfect", "precision": 0.8824, "recall": 0.7143},
            "frontier": PRECISION_FRONTIER,
        },
    }
    summary = mod._precision_rescue_honest(rec)
    assert summary["winning_stack_is_demo_perfect_alone"] is True
    assert summary["ensemble_added_value_over_demo_perfect_alone"] is False


def test_precision_rescue_honest_defaults_when_empty() -> None:
    summary = mod._precision_rescue_honest({"exists": False})
    assert summary["best_certified_precision"] == mod.PRECISION_RESCUE_BEST_PRECISION_DEFAULT
    assert summary["best_op_point_recall"] == mod.PRECISION_RESCUE_BEST_RECALL_DEFAULT
    assert summary["invariance_recall"] == mod.INVARIANCE_RECALL_DEFAULT
    # no frontier rows -> min_hamming falls back to None, not worse
    assert summary["min_hamming_best_precision_at_winning_recall"] is None
    assert summary["min_hamming_worse_than_demo_perfect"] is False


def test_offarc_transfer() -> None:
    summary = mod._offarc_transfer(
        {
            "exists": True,
            "honest_verdict": "complete: offarc",
            "fields": {
                "demofit_precision_raw": 0.956186,
                "demofit_precision_filtered": 0.96049,
                "domain_general_precision_floor": 0.68,
                "primitive_is_domain_general": True,
            },
        }
    )
    assert summary["measured"] is True
    assert summary["clears_floor"] is True
    assert summary["primitive_is_domain_general"] is True
    assert summary["filter_lift"] == pytest.approx(0.004304, abs=1e-6)


def test_offarc_transfer_missing() -> None:
    summary = mod._offarc_transfer({"exists": False})
    assert summary["measured"] is False
    assert summary["filter_lift"] is None
    assert summary["clears_floor"] is False


def test_accuracy_tenth_game() -> None:
    summary = mod._accuracy(
        {
            "exists": True,
            "honest_verdict": "success: tenth_game_solved",
            "fields": {
                "game_solved": True,
                "total_games_solved": 10,
                "prior_total_games_solved": 9,
                "target_game": "r11l-495a7899",
                "first_solve_at_action": 4,
                "real_env_confirmed": True,
            },
        },
        {"exists": True, "fields": {"games_solved_total": 10}},
    )
    assert summary["total_games_solved"] == 10
    assert summary["tenth_game_solved"] is True
    assert summary["monotonic_no_regression"] is True
    assert summary["tenth_game"] == "r11l-495a7899"


def test_accuracy_falls_back_to_capstone_and_prior() -> None:
    # tenth-game record lacks total but has prior + solved -> prior+1.
    summary = mod._accuracy(
        {"exists": True, "honest_verdict": "success: x", "fields": {"game_solved": True, "prior_total_games_solved": 9}},
        {"exists": False},
    )
    assert summary["total_games_solved"] == 10
    # nothing anywhere -> default 10
    summary2 = mod._accuracy({"exists": False}, {"exists": False})
    assert summary2["total_games_solved"] == mod.TOTAL_GAMES_SOLVED_DEFAULT
    assert summary2["tenth_game_solved"] is False


def test_accuracy_capstone_games_total() -> None:
    summary = mod._accuracy({"exists": False}, {"exists": True, "fields": {"games_solved_total": 10}})
    assert summary["total_games_solved"] == 10


def test_hardware() -> None:
    summary = mod._hardware(
        {
            "exists": True,
            "honest_verdict": "complete: hw",
            "fields": {
                "per_board_reachability": {"gatemate": False, "kv260": True, "polarfire": True},
                "per_board_terminal_state": {"kv260": "terminal"},
                "gatemate_step_taken": "blocked_gatemate_unreachable",
                "polarfire_step_taken": "polarfire_hash_verified_cpu_dispatch_succeeded",
                "kv260_terminal_confirmed": True,
            },
        }
    )
    assert summary["included"] is True
    assert summary["kv260_terminal"] is True
    assert summary["per_board_reachability"]["gatemate"] is False


def test_flagged_skipped() -> None:
    summary = mod._flagged_skipped(
        {"4095": {"exists": True, "flagged_adversarial": True, "honest_verdict": "complete: x"}}
    )
    assert summary["count"] == 1
    assert summary["experiment_ids"] == ["4095"]
    assert summary["skipped"][0]["flagged_adversarial"] is True


# --------------------------------------------------------------------------- #
# Full close-state + provenance
# --------------------------------------------------------------------------- #
def test_build_close_state(tmp_path: Path) -> None:
    root = make_repo(tmp_path)
    records = mod.read_v378_records(root)
    cs = mod.build_v378_close_state(records)
    assert cs["lora_training_retired"] is True
    assert cs["precision_rescue_carried_by_demo_perfect_alone"] is True
    assert cs["precision_rescue"]["ensemble_added_value_over_demo_perfect_alone"] is False
    assert cs["total_games_solved"] == 10
    assert cs["flagged_count"] == 1
    assert cs["status_counts"]["BLOCKED"] == 2  # exp4088 + exp4089
    assert cs["status_counts"]["MISSING"] == 2  # exp4090 + exp4091
    assert cs["status_counts"]["FLAGGED"] == 1  # exp4095
    assert cs["per_task_status"]["exp4094-sota-ingestion-precision-calibration"] == "OK"  # doc
    assert "demo-perfect ALONE" in cs["headline"] or "demo-perfect" in cs["headline"].lower()


def test_build_cited_upstream(tmp_path: Path) -> None:
    root = make_repo(tmp_path)
    cited = mod.build_cited_upstream(root)
    assert len(cited) == len(mod.V378_TASKS)
    by_id = {c["experiment_id"]: c for c in cited}
    assert mod.is_sha256(by_id["4087"]["sha256"])  # present artifact hashed
    assert by_id["4090"]["sha256"] is None  # missing artifact


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
    # _run_command on a bogus binary returns a non-127-or-127 result without raising.
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
    assert "lora_verifier_as_reward_retired" in art["honest_verdict"]
    assert "demo_perfect_alone" in art["honest_verdict"]
    assert art["archived_milestone"] == "2026.06.378"
    assert art["activated_milestone"] == "2026.06.379"
    assert art["active_milestone_confirmed"] == "2026.06.379"
    assert art["trm_substrate_present"] is True
    assert art["pretest_suite_green"] is True
    assert art["total_games_solved"] == 10
    assert art["flagged_count"] == 1
    assert art["research_complete_record_action"] == "deduped"
    assert art["research_complete_duplicates_removed"] == 2
    assert art["inference_substrate"] == "aggregation_from_upstream_artifacts"
    assert mod.is_sha256(art["reproducibility_checksum"])
    # the dedup actually rewrote research-complete.yaml to a single .378 record
    text = (root / "research-complete.yaml").read_text(encoding="utf-8")
    assert text.count("- id: 2026.06.378") == 1
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


def test_run_blocked_v379_not_active(tmp_path: Path) -> None:
    root = make_repo(tmp_path)
    (root / "research-roadmap.yaml").write_text('milestone: "2026.06.378"\n', encoding="utf-8")
    out = mod.run(root, pretest_result=GREEN)
    art = json.loads(out.read_text(encoding="utf-8"))
    assert art["honest_verdict"] == "blocked_v379_not_active"
    assert art["research_complete_yaml_parses"] is True


def test_run_blocked_trm_substrate_missing(tmp_path: Path) -> None:
    root = make_repo(tmp_path)
    (root / "nano-trm" / "src" / "arc_evaluator.py").unlink()
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
    assert art["v378_close_state"]["total_games_solved"] == 10


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


def test_validate_rejects_ensemble_added_value_true(tmp_path: Path) -> None:
    art = _good_artifact(tmp_path)
    art["v378_close_state"]["precision_rescue"]["ensemble_added_value_over_demo_perfect_alone"] = True
    with pytest.raises(ValueError, match="ensemble_added_value"):
        mod.validate_artifact(art)


def test_validate_rejects_lora_not_retired(tmp_path: Path) -> None:
    art = _good_artifact(tmp_path)
    art["v378_close_state"]["lora_training_retired"] = False
    with pytest.raises(ValueError, match="lora_training_retired"):
        mod.validate_artifact(art)


def test_validate_rejects_training_signal_measured(tmp_path: Path) -> None:
    art = _good_artifact(tmp_path)
    art["v378_close_state"]["lora_training"]["verifier_label_training_signal_measured"] = True
    with pytest.raises(ValueError, match="UNMEASURED"):
        mod.validate_artifact(art)


def test_validate_rejects_wrong_games(tmp_path: Path) -> None:
    art = _good_artifact(tmp_path)
    art["v378_close_state"]["accuracy"]["total_games_solved"] = 9
    with pytest.raises(ValueError, match="must be 10"):
        mod.validate_artifact(art)


def test_validate_rejects_wrong_flagged_count(tmp_path: Path) -> None:
    art = _good_artifact(tmp_path)
    art["flagged_count"] = 2
    art["v378_close_state"]["flagged_count"] = 2
    art["v378_close_state"]["flagged_skipped"]["count"] = 2
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
    records = mod.read_v378_records(root)
    cs = mod.build_v378_close_state(records)
    payload = mod.build_complete_artifact(
        v378_close_state=cs,
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
    v = mod.terminal_verdict({"accuracy": {"total_games_solved": 10}, "flagged_skipped": {"count": 1}})
    assert v.startswith("success:")
    assert "games10" in v
    assert "1_flagged_skipped" in v


# --------------------------------------------------------------------------- #
# Exhaustive validate_artifact rejection guards (the anti-laundering wall)
# --------------------------------------------------------------------------- #
def _mutators() -> list[tuple[str, "object"]]:
    """Return (match, mutator) pairs -- each trips exactly one validate guard.

    Each mutator changes ONE field of an otherwise-valid artifact so the matching
    ``ValueError`` fires. Together they cover every guard that prevents a null
    being laundered into a milestone win.
    """

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
        ("non-empty dict", lambda a: a.__setitem__("v378_close_state", {})),
        ("must record per_task_status", lambda a: a.__setitem__("v378_close_state", {"x": 1})),
        ("lora_training summary must be recorded retired",
         lambda a: set_path(a, ["v378_close_state", "lora_training"], {"retired": False})),
        ("the precision_rescue summary",
         lambda a: set_path(a, ["v378_close_state", "precision_rescue"], "x")),
        ("winning_stack_is_demo_perfect_alone=True",
         lambda a: set_path(a, ["v378_close_state", "precision_rescue", "winning_stack_is_demo_perfect_alone"], False)),
        ("must record accuracy", lambda a: set_path(a, ["v378_close_state", "accuracy"], "x")),
        ("must be monotonic",
         lambda a: set_path(a, ["v378_close_state", "accuracy", "monotonic_no_regression"], False)),
        ("top-level total_games_solved must be 10", lambda a: a.__setitem__("total_games_solved", 9)),
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
    art["v378_close_state"]["flagged_count"] = 2
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
