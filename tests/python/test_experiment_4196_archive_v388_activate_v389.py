"""Tests for Exp 4196 `.388` archive / `.389` activation.

Spec refs: REQ-REPORT-4196, SCENARIO-REPORT-4196,
SCENARIO-REPORT-4196-BLOCKED-PRECONDITION.
"""

from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path

import pytest

from carnot.reporting import archive_v388_activate_v389_4196 as mod


GREEN = mod.CommandResult(command=["pytest"], exit_code=0, stdout="81 passed", stderr="")
RED = mod.CommandResult(command=["pytest"], exit_code=1, stdout="FAILED smart subset", stderr="")


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _research_complete_text(*, duplicates: int = 1) -> str:
    head = (
        "# Carnot Research - Completed Experiments\n"
        "milestones:\n"
        "- id: 2026.06.387\n"
        "  finding: prior milestone\n"
    )
    block = (
        "- id: 2026.06.388\n"
        "  title: old conductor row\n"
        "  completed: '2026-06-14'\n"
        "  finding: See conductor log for per-experiment results.\n"
        "  tasks:\n"
        "  - id: exp4195-capstone-v388\n"
        "    result: OK\n"
    )
    return head + block * duplicates


def make_repo(tmp_path: Path, *, duplicates: int = 1) -> Path:
    root = tmp_path
    root.mkdir(parents=True, exist_ok=True)
    (root / "research-complete.yaml").write_text(
        _research_complete_text(duplicates=duplicates), encoding="utf-8"
    )
    (root / "ops").mkdir(parents=True, exist_ok=True)
    (root / "ops" / "exclusion_manifest.yaml").write_text(
        "retired:\n- experiment_id: 2091\n  reason: archived\n", encoding="utf-8"
    )
    (root / "research-roadmap.yaml").write_text("milestone: 2026.06.389\n", encoding="utf-8")
    (root / "tests" / "python").mkdir(parents=True, exist_ok=True)
    (root / "tests" / "python" / "test_pipeline_extract.py").write_text(
        "def test_pipeline_extract():\n    assert True\n", encoding="utf-8"
    )
    (root / "tests" / "python" / "test_docs.py").write_text(
        "def test_docs():\n    assert True\n", encoding="utf-8"
    )
    _write_json(root / "results" / "experiment_4195_capstone_v388.json", _capstone())
    _write_json(
        root / "results" / "experiment_4188_sovereign_local_generator_gap4_self_distill.json",
        _sovereign(),
    )
    return root


def _capstone(**overrides: object) -> dict:
    payload = {
        "experiment_id": 4195,
        "honest_verdict": (
            "complete: capstone_v388_efficiency_moat_won_efficiency_WON_gap4_safe_true_"
            "sovereign_false_diffusiongemma_false_arc_levels15_flagged_skipped2"
        ),
        "headline_outcome": "efficiency_moat_won",
        "efficiency_moat_status": "WON",
        "gap4_production_safe": True,
        "gap4_sovereign": False,
        "diffusiongemma_feasible": False,
        "total_arc_levels_solved": 15,
        "live_env_reachable": True,
        "efficiency_moat": {
            "honest_verdict": "complete: verifier_efficiency_win_true_delta_0.1800",
            "efficiency_moat_status": "WON",
            "verifier_efficiency_win": True,
            "positive_control_confirmed": True,
            "accuracy_parity_vs_judge": {
                "arm_a_pass1": 0.84,
                "arm_j_pass1": 0.66,
                "delta": 0.18,
                "ci95": [0.08, 0.30],
                "n": 50,
                "within_ci_or_better": True,
            },
            "cost_ratio_vs_judge": {
                "status": "measured",
                "strictly_pareto_dominant": True,
                "ten_x_cheaper_on_both_axes": True,
                "wall_clock_x_cheaper": 500351.5303458394,
                "arm_j_total_tokens": 5270,
            },
        },
        "gap4_production_safety": {
            "status": "HOLDS-plus4-minus0",
            "safe": True,
            "graded_gate_pass2_vs_vote": 0.129,
            "gross_recovery_ledger": {"recovered": 4, "lost": 0},
            "pass2_vote_wins_lost": 0,
            "vote_aware_guard": {
                "blocked_tasks": ["25094a63"],
                "threshold_votes": 900,
            },
            "vote_aware_guard_blocked_mispromotion": True,
        },
        "gap4_sovereign_detail": {
            "status": "skipped_flagged_adversarial",
            "recovered_arc_headroom": False,
        },
        "diffusiongemma_detail": {
            "status": "blocked_diffusiongemma_not_cached",
            "honest_verdict": "blocked_diffusiongemma_not_cached",
            "diffusiongemma_feasible": False,
            "model_specs": {
                "diffusiongemma": {
                    "weights_cached": False,
                    "present_weight_shards": 0,
                    "expected_weight_shards": 11,
                    "guidance_hook_fired": False,
                }
            },
        },
        "arc_progress": {
            "total_arc_levels_solved": 15,
            "total_arc_games_solved": 13,
            "real_env_confirmed": True,
        },
        "live_env": {
            "live_env_reachable": True,
            "environment_count": 25,
            "honest_verdict": "complete: arc_live_env_reachable_random_greedy_baseline_lp85-305b61c3",
        },
        "flagged_artifacts_skipped": [
            {
                "experiment_id": 4188,
                "honest_verdict": (
                    "success: sovereign_local_gap4_recovers_headroom_pass20.4839_"
                    "cov0.2258_corpus7"
                ),
                "path": "results/experiment_4188_sovereign_local_generator_gap4_self_distill.json",
                "reason": "flagged_adversarial:true",
            }
        ],
        "upstream_provenance": [
            {
                "experiment_id": 4186,
                "path": "results/experiment_4186_efficiency_moat_verifier_vs_llm_judge.json",
                "sha256": "d" * 64,
            },
            {
                "experiment_id": 4187,
                "path": "results/experiment_4187_gap4_graded_execution_gate_hardening.json",
                "sha256": "e" * 64,
            },
            {
                "experiment_id": 4188,
                "path": "results/experiment_4188_sovereign_local_generator_gap4_self_distill.json",
                "sha256": "8" * 64,
                "skipped": True,
                "skip_reason": "flagged_adversarial:true",
            },
            {
                "experiment_id": 4189,
                "path": "results/experiment_4189_diffusiongemma_verifier_guided_decoding.json",
                "sha256": "6" * 64,
            },
            {
                "experiment_id": 4191,
                "path": "results/experiment_4191_arc_live_env_grounding_probe.json",
                "sha256": "5" * 64,
            },
        ],
    }
    payload.update(overrides)
    return payload


def _sovereign(**overrides: object) -> dict:
    payload = {
        "honest_verdict": "success: sovereign_local_gap4_recovers_headroom_pass20.4839_cov0.2258_corpus7",
        "flagged_adversarial": True,
        "local_induction_rate": {
            "demo_perfect": 7,
            "total": 31,
            "rate": 0.2258,
            "codex_reference": {"demo_perfect": 29, "total": 31, "rate": 0.9355},
        },
        "no_closed_weight_call": True,
        "self_distillation_corpus_size": 7,
        "sovereign_pool_pass2": {
            "LOCAL_HARDENED_GATE": 0.4839,
            "TRM_VOTE": 0.4516,
            "codex_hardened_reference": 0.5806,
            "recovered": 1,
            "lost": 0,
        },
    }
    payload.update(overrides)
    return payload


def run_happy(root: Path) -> dict:
    out = mod.run(root, pretest_result=GREEN, started_s=1000.0, now_s=1000.5)
    return json.loads(out.read_text(encoding="utf-8"))


def test_req_report_4196_spec_declares_contract() -> None:
    """REQ-REPORT-4196: OpenSpec declares required fields and scenarios."""

    spec = Path("openspec/capabilities/research-reporting/spec.md").read_text(encoding="utf-8")
    assert "REQ-REPORT-4196" in spec
    assert "SCENARIO-REPORT-4196" in spec
    assert "SCENARIO-REPORT-4196-BLOCKED-PRECONDITION" in spec
    assert "v388_close_state" in spec
    assert "WON but semi-circular" in spec
    assert "status=HOLDS-plus4-minus0" in spec
    assert "coverage `0.2258` versus Codex `0.9355`" in spec
    assert "total_levels_solved=15" in spec
    assert mod.FIELD_PRINCIPLES["honest_verdict"] in spec
    assert mod.FIELD_PRINCIPLES["v388_close_state"] in spec
    assert mod.FIELD_PRINCIPLES["preconditions_checked"] in spec


def test_shared_helpers_and_archive_record_editing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-REPORT-4196: helper behavior is deterministic and YAML-safe."""

    assert mod.yaml_parses("a: 1\n") is True
    assert mod.yaml_parses("a: : :\n- [\n") is False
    assert mod.duration_from(None, None) == 0.0001
    assert mod.duration_from(100.0, 100.25) == 0.25
    assert mod.duration_from(100.0, 99.0) == 0.0001
    monkeypatch.setattr(mod.time, "perf_counter", lambda: 101.0)
    assert mod.duration_from(100.0, None) == 1.0
    assert mod.payload_checksum({"a": 1}) == mod.payload_checksum(
        {"a": 1, "reproducibility_checksum": "old"}
    )
    assert mod.is_sha256("a" * 64) is True
    assert mod.is_sha256("z" * 64) is False
    out = tmp_path / "artifact.json"
    mod.write_payload(out, {"b": 2, "a": 1})
    assert out.read_text(encoding="utf-8").startswith('{\n  "a"')
    assert mod._insert_before_tasks(["  title: no tasks"], "  finding: x") == [
        "  title: no tasks",
        "  finding: x",
    ]

    assert mod.archive_record_count(_research_complete_text(duplicates=3)) == 3
    close_state = mod.build_v388_close_state({"4195": _capstone(), "4188": _sovereign()})
    deduped, removed, action = mod.dedupe_or_update_record(
        _research_complete_text(duplicates=3), close_state
    )
    assert action == "deduped"
    assert removed == 2
    assert mod.archive_record_count(deduped) == 1
    assert "WON-but-SEMI-CIRCULAR" in deduped
    assert "verifier-as-reward A-vs-B test on CODE" in deduped
    assert mod.yaml_parses(deduped)
    updated, removed2, action2 = mod.dedupe_or_update_record(
        _research_complete_text(duplicates=1), close_state
    )
    assert (removed2, action2) == (0, "updated")
    unchanged, removed4, action4 = mod.dedupe_or_update_record(updated, close_state)
    assert (unchanged, removed4, action4) == (updated, 0, "unchanged")
    old_activation, removed5, action5 = mod.dedupe_or_update_record(
        "milestones:\n- id: 2026.06.388\n  activation_recorded: old\n  tasks:\n  - id: exp4195\n",
        close_state,
    )
    assert (removed5, action5) == (0, "updated")
    assert "activation_recorded: exp4196-archive-v388-activate-v389" in old_activation
    no_tasks, removed6, action6 = mod.dedupe_or_update_record(
        "milestones:\n- id: 2026.06.388\n  title: no tasks\n",
        close_state,
    )
    assert (removed6, action6) == (0, "updated")
    assert "  finding: " in no_tasks
    appended, removed3, action3 = mod.dedupe_or_update_record(
        "# history\nmilestones:\n- id: 2026.06.387\n  finding: prior\n", close_state
    )
    assert (removed3, action3) == (0, "appended")
    assert "activation_recorded: exp4196-archive-v388-activate-v389" in appended


def test_precondition_helpers_and_source_readers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SCENARIO-REPORT-4196-BLOCKED-PRECONDITION: resource probes are explicit."""

    assert mod._milestone_from_text("name: no milestone\n") == "unknown"
    assert mod.read_active_milestone(tmp_path) == ("unknown", "research-roadmap.yaml")
    (tmp_path / "research-roadmap-next.yaml").write_text(
        "milestone: 2026.06.389\n", encoding="utf-8"
    )
    assert mod.read_active_milestone(tmp_path) == ("2026.06.389", "research-roadmap-next.yaml")

    assert mod.read_json_object(tmp_path / "missing.json") == {}
    bad = tmp_path / "bad.json"
    bad.write_text("{bad", encoding="utf-8")
    assert mod.read_json_object(bad) == {}
    list_path = tmp_path / "list.json"
    list_path.write_text("[1]", encoding="utf-8")
    assert mod.read_json_object(list_path) == {}
    good = tmp_path / "good.json"
    good.write_text(json.dumps({"x": 1}), encoding="utf-8")
    assert mod.read_json_object(good) == {"x": 1}
    assert mod.is_sha256(mod.file_sha256(good))
    assert mod.file_sha256(tmp_path / "nope") is None

    root = make_repo(tmp_path / "repo")
    targets = mod.smart_subset_targets(root)
    assert "tests/python/test_pipeline_extract.py" in targets
    assert mod.smart_subset_command(targets)[0] == str(mod.PYTEST_BIN)
    assert mod.smart_subset_targets(tmp_path / "empty") == [mod.CORE_SMART_SUBSET[0]]
    assert mod._run_command(["true"], tmp_path).exit_code == 0
    assert mod._run_command(["definitely-not-a-real-binary-xyz"], tmp_path).exit_code == 127
    monkeypatch.setattr(mod, "_run_command", lambda command, root_path: GREEN)
    assert mod.run_smart_subset(root).exit_code == 0


def test_read_sources_and_build_v388_close_state(tmp_path: Path) -> None:
    """SCENARIO-REPORT-4196: close-state records the .388 pivot truth."""

    root = make_repo(tmp_path)
    sources = mod.read_v388_sources(root)
    assert sources["4195"]["efficiency_moat_status"] == "WON"
    assert sources["4188"]["self_distillation_corpus_size"] == 7
    cited = mod.build_cited_upstream(root)
    assert any(item["experiment_id"] == "4195" for item in cited)
    assert any(item["experiment_id"] == "4188" for item in cited)
    assert all(item["sha256"] is None or mod.is_sha256(item["sha256"]) for item in cited)

    state = mod.build_v388_close_state(sources)
    assert state["summary"] == (
        "efficiency_moat_won_but_semicircular_gap4_safe_"
        "sovereign_under_induces_diffusiongemma_no_weights"
    )
    assert state["efficiency_moat_status"] == "WON-but-SEMI-CIRCULAR"
    assert state["efficiency_measured_status"] == "WON"
    assert state["verifier_efficiency_win"] is True
    assert state["efficiency_delta_vs_llm_judge"] == 0.18
    assert state["efficiency_delta_ci95"] == [0.08, 0.3]
    assert state["efficiency_moat_semicircular_caveat"] == (
        "verifier==unit-test oracle; production value is real but not an independent learned reward"
    )
    assert state["gap4_production_safe"] is True
    assert state["gap4_status"] == "HOLDS-plus4-minus0"
    assert state["gap4_recovered"] == 4
    assert state["gap4_lost"] == 0
    assert state["gap4_guard_blocked_tasks"] == ["25094a63"]
    assert state["sovereign_status"] == "UNDER-induces"
    assert state["sovereign_local_induction_rate"] == 0.2258
    assert state["sovereign_codex_reference_rate"] == 0.9355
    assert state["self_distillation_corpus_size"] == 7
    assert state["diffusiongemma_status"] == "blocked-no-weights"
    assert state["diffusiongemma_weights_cached"] is False
    assert state["total_levels_solved"] == 15
    assert state["total_games_solved"] == 13
    assert state["live_env_reachable"] is True
    assert state["v389_planner_frame"] == (
        "run the verifier-as-reward A-vs-B test on CODE where Phase-0 finally clears; "
        "do not redo the selection/efficiency line"
    )

    fallback = mod.build_v388_close_state({"4195": _capstone(), "4188": {}})
    assert fallback["sovereign_local_induction_rate"] == 0.2258
    assert fallback["self_distillation_corpus_size"] == 7


def test_run_happy_path_writes_valid_artifact_and_updates_archive(tmp_path: Path) -> None:
    """SCENARIO-REPORT-4196: complete path writes the terminal archive artifact."""

    root = make_repo(tmp_path, duplicates=2)
    artifact = run_happy(root)
    assert artifact["honest_verdict"].startswith("success:")
    assert artifact["archived_milestone"] == "2026.06.388"
    assert artifact["activated_milestone"] == "2026.06.389"
    assert artifact["active_milestone_confirmed"] == "2026.06.389"
    assert artifact["research_complete_yaml_parses"] is True
    assert artifact["exclusion_manifest_parses"] is True
    assert artifact["pretest_suite_green"] is True
    assert artifact["research_complete_record_action"] == "deduped"
    assert artifact["research_complete_duplicates_removed"] == 1
    assert artifact["v388_close_state"]["efficiency_moat_status"] == "WON-but-SEMI-CIRCULAR"
    assert artifact["v388_close_state"]["gap4_status"] == "HOLDS-plus4-minus0"
    assert artifact["v388_close_state"]["sovereign_status"] == "UNDER-induces"
    assert artifact["v388_close_state"]["diffusiongemma_status"] == "blocked-no-weights"
    assert (
        artifact["field_principles"]["v388_close_state"] == mod.FIELD_PRINCIPLES["v388_close_state"]
    )
    assert artifact["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert mod.is_sha256(artifact["reproducibility_checksum"])
    complete_text = (root / "research-complete.yaml").read_text(encoding="utf-8")
    assert mod.archive_record_count(complete_text) == 1
    assert "WON-but-SEMI-CIRCULAR" in complete_text
    assert "DiffusionGemma blocked-no-weights" in complete_text
    mod.validate_artifact(artifact)


def test_run_real_pretest_branch_and_entrypoints_are_injectable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SCENARIO-REPORT-4196: real pretest and CLI entrypoints can be substituted."""

    root = make_repo(tmp_path)
    monkeypatch.setattr(mod, "run_smart_subset", lambda root_path: GREEN)
    artifact = json.loads(mod.run(root, started_s=1.0, now_s=1.1).read_text(encoding="utf-8"))
    assert artifact["preconditions_checked"]["pretest_suite_green"] is True

    called_mod: dict[str, Path] = {}

    def fake_mod_run(root_path: Path = mod.REPO_ROOT) -> Path:
        called_mod["root"] = Path(root_path)
        return root / mod.OUTPUT_REL_PATH

    monkeypatch.setattr(mod, "run", fake_mod_run)
    assert mod.main() == 0
    assert called_mod["root"] == mod.REPO_ROOT

    import carnot.experiment_4196_archive_v388_activate_v389 as entrypoint

    called: dict[str, Path] = {}

    def fake_run(root_path: Path = mod.REPO_ROOT) -> Path:
        called["root"] = Path(root_path)
        return root / mod.OUTPUT_REL_PATH

    monkeypatch.setattr(entrypoint, "run", fake_run)
    assert entrypoint.main() == 0
    assert called["root"] == entrypoint.REPO_ROOT

    script_path = Path("results/experiment_4196_archive_v388_activate_v389.py")
    spec = importlib.util.spec_from_file_location("exp4196_archive_script", script_path)
    assert spec and spec.loader
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)
    monkeypatch.setattr(script, "run", fake_run)
    assert script.main() == 0


def test_run_blocked_preconditions_before_research_complete_edit(tmp_path: Path) -> None:
    """SCENARIO-REPORT-4196-BLOCKED-PRECONDITION: blocked paths do not fabricate success."""

    missing = mod.run(tmp_path, pretest_result=GREEN)
    assert json.loads(missing.read_text(encoding="utf-8"))["honest_verdict"] == (
        "blocked_research_complete_yaml_missing"
    )

    root = make_repo(tmp_path / "poison")
    (root / "research-complete.yaml").write_text("a: : :\n- [\n", encoding="utf-8")
    artifact = json.loads(mod.run(root, pretest_result=GREEN).read_text(encoding="utf-8"))
    assert artifact["honest_verdict"] == "blocked_research_complete_yaml_poison"

    root2 = make_repo(tmp_path / "manifest_missing")
    (root2 / "ops" / "exclusion_manifest.yaml").unlink()
    artifact2 = json.loads(mod.run(root2, pretest_result=GREEN).read_text(encoding="utf-8"))
    assert artifact2["honest_verdict"] == "blocked_exclusion_manifest_missing"

    root3 = make_repo(tmp_path / "manifest_poison")
    (root3 / "ops" / "exclusion_manifest.yaml").write_text("a: : :\n- [\n", encoding="utf-8")
    artifact3 = json.loads(mod.run(root3, pretest_result=GREEN).read_text(encoding="utf-8"))
    assert artifact3["honest_verdict"] == "blocked_exclusion_manifest_yaml_poison"

    root4 = make_repo(tmp_path / "wrong_milestone")
    (root4 / "research-roadmap.yaml").write_text("milestone: 2026.06.388\n", encoding="utf-8")
    artifact4 = json.loads(mod.run(root4, pretest_result=GREEN).read_text(encoding="utf-8"))
    assert artifact4["honest_verdict"] == "blocked_v389_not_active"

    root5 = make_repo(tmp_path / "red")
    before = (root5 / "research-complete.yaml").read_text(encoding="utf-8")
    artifact5 = json.loads(mod.run(root5, pretest_result=RED).read_text(encoding="utf-8"))
    assert artifact5["honest_verdict"] == "blocked_smart_subset_pretest_not_green"
    assert artifact5["preconditions_checked"]["pretest_suite_green"] is False
    assert (root5 / "research-complete.yaml").read_text(encoding="utf-8") == before

    root6 = make_repo(tmp_path / "capstone_missing")
    (root6 / "results" / "experiment_4195_capstone_v388.json").unlink()
    artifact6 = json.loads(mod.run(root6, pretest_result=GREEN).read_text(encoding="utf-8"))
    assert artifact6["honest_verdict"] == "blocked_v388_capstone_missing"


def test_run_blocked_research_complete_edit_guards(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-REPORT-4196: invalid archive edits are blocked before completion."""

    root = make_repo(tmp_path / "invalid")
    monkeypatch.setattr(
        mod, "dedupe_or_update_record", lambda text, state: ("a: : :\n- [", 0, "appended")
    )
    artifact = json.loads(mod.run(root, pretest_result=GREEN).read_text(encoding="utf-8"))
    assert artifact["honest_verdict"] == "blocked_research_complete_edit_invalid"

    root2 = make_repo(tmp_path / "after")
    calls = {"n": 0}

    def fake_parses(text: str) -> bool:
        calls["n"] += 1
        return calls["n"] != 4

    monkeypatch.setattr(mod, "yaml_parses", fake_parses)
    artifact2 = json.loads(mod.run(root2, pretest_result=GREEN).read_text(encoding="utf-8"))
    assert artifact2["honest_verdict"] == "blocked_research_complete_yaml_poison_after_edit"


def test_build_artifact_shapes(tmp_path: Path) -> None:
    """REQ-REPORT-4196: complete and blocked artifact builders keep schema shape."""

    root = make_repo(tmp_path)
    state = mod.build_v388_close_state(mod.read_v388_sources(root))
    complete = mod.build_complete_artifact(
        v388_close_state=state,
        preconditions_checked={"ok": True},
        duration_s=0.5,
        active_roadmap_path="research-roadmap.yaml",
        research_complete_record_action="updated",
        research_complete_duplicates_removed=0,
        cited_upstream_artifacts=mod.build_cited_upstream(root),
    )
    assert complete["honest_verdict"].startswith("success:")
    blocked = mod.build_blocked_artifact(
        "blocked_x",
        preconditions_checked={"ok": False},
        duration_s=0.1,
        active_milestone_confirmed="",
        active_roadmap_path="research-roadmap.yaml",
    )
    assert blocked["honest_verdict"] == "blocked_x"
    assert mod.is_sha256(blocked["reproducibility_checksum"])
    assert mod.terminal_verdict(state).startswith("success:")


def test_validate_artifact_rejection_guards(tmp_path: Path) -> None:
    """REQ-REPORT-4196: validation rejects artifacts that launder the .388 truth."""

    good = run_happy(make_repo(tmp_path))

    def set_path(obj: dict, path: list[str], value: object) -> None:
        cur = obj
        for key in path[:-1]:
            cur = cur[key]
        cur[path[-1]] = value

    cases = [
        ("missing required fields", lambda a: a.pop("v388_close_state")),
        ("terminal-prefixed", lambda a: a.__setitem__("honest_verdict", "done")),
        ("field_principles must be a mapping", lambda a: a.__setitem__("field_principles", "x")),
        ("missing field principles", lambda a: a.__setitem__("field_principles", {})),
        (
            "principle must match REQ-REPORT-4196",
            lambda a: a["field_principles"].__setitem__("v388_close_state", "wrong"),
        ),
        ("archived milestone", lambda a: a.__setitem__("archived_milestone", "2026.06.999")),
        ("activated milestone", lambda a: a.__setitem__("activated_milestone", "2026.06.999")),
        ("research-complete YAML", lambda a: a.__setitem__("research_complete_yaml_parses", False)),
        ("exclusion manifest", lambda a: a.__setitem__("exclusion_manifest_parses", False)),
        ("pretest suite", lambda a: a.__setitem__("pretest_suite_green", False)),
        ("active milestone", lambda a: a.__setitem__("active_milestone_confirmed", "2026.06.388")),
        ("v388_close_state must be a mapping", lambda a: a.__setitem__("v388_close_state", "x")),
        ("efficiency moat", lambda a: set_path(a, ["v388_close_state", "efficiency_moat_status"], "WON")),
        (
            "verifier_efficiency_win",
            lambda a: set_path(a, ["v388_close_state", "verifier_efficiency_win"], False),
        ),
        (
            "efficiency_delta_vs_llm_judge",
            lambda a: set_path(a, ["v388_close_state", "efficiency_delta_vs_llm_judge"], 0.1),
        ),
        (
            "efficiency_delta_ci95",
            lambda a: set_path(a, ["v388_close_state", "efficiency_delta_ci95"], [0.0, 0.1]),
        ),
        (
            "semi-circular",
            lambda a: set_path(a, ["v388_close_state", "efficiency_moat_semicircular_caveat"], "none"),
        ),
        ("GAP-4 safe", lambda a: set_path(a, ["v388_close_state", "gap4_production_safe"], False)),
        ("GAP-4 status", lambda a: set_path(a, ["v388_close_state", "gap4_status"], "unsafe")),
        ("GAP-4 recovered", lambda a: set_path(a, ["v388_close_state", "gap4_recovered"], 3)),
        ("GAP-4 lost", lambda a: set_path(a, ["v388_close_state", "gap4_lost"], 1)),
        (
            "GAP-4 guard",
            lambda a: set_path(a, ["v388_close_state", "gap4_guard_blocked_tasks"], []),
        ),
        ("sovereign status", lambda a: set_path(a, ["v388_close_state", "sovereign_status"], "WON")),
        (
            "local induction",
            lambda a: set_path(a, ["v388_close_state", "sovereign_local_induction_rate"], 0.3),
        ),
        (
            "codex reference",
            lambda a: set_path(a, ["v388_close_state", "sovereign_codex_reference_rate"], 0.8),
        ),
        (
            "self-distill corpus",
            lambda a: set_path(a, ["v388_close_state", "self_distillation_corpus_size"], 8),
        ),
        (
            "DiffusionGemma",
            lambda a: set_path(a, ["v388_close_state", "diffusiongemma_status"], "ready"),
        ),
        (
            "DiffusionGemma weights",
            lambda a: set_path(a, ["v388_close_state", "diffusiongemma_weights_cached"], True),
        ),
        ("total levels solved", lambda a: set_path(a, ["v388_close_state", "total_levels_solved"], 14)),
        ("total games solved", lambda a: set_path(a, ["v388_close_state", "total_games_solved"], 12)),
        ("live env", lambda a: set_path(a, ["v388_close_state", "live_env_reachable"], False)),
        ("planner frame", lambda a: set_path(a, ["v388_close_state", "v389_planner_frame"], "redo")),
        ("duration_s", lambda a: a.__setitem__("duration_s", 0)),
        ("inference_substrate", lambda a: a.__setitem__("inference_substrate", "live_training")),
        ("cited_upstream_artifacts", lambda a: a.__setitem__("cited_upstream_artifacts", "x")),
        ("reproducibility_checksum", lambda a: a.__setitem__("reproducibility_checksum", "short")),
    ]
    for match, mutate in cases:
        artifact = copy.deepcopy(good)
        mutate(artifact)
        with pytest.raises(ValueError, match=match):
            mod.validate_artifact(artifact)

    mismatch = copy.deepcopy(good)
    mismatch["honest_verdict"] = "success: changed"
    with pytest.raises(ValueError, match="checksum"):
        mod.validate_artifact(mismatch)
