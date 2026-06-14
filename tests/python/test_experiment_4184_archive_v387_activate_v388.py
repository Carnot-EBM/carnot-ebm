"""Tests for Exp 4184 `.387` archive / `.388` activation.

Spec refs: REQ-REPORT-4184, SCENARIO-REPORT-4184,
SCENARIO-REPORT-4184-BLOCKED-PRECONDITION.
"""

from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path

import pytest

from carnot.reporting import archive_v387_activate_v388_4184 as mod


GREEN = mod.CommandResult(command=["pytest"], exit_code=0, stdout="81 passed", stderr="")
RED = mod.CommandResult(command=["pytest"], exit_code=1, stdout="FAILED smart subset", stderr="")


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _research_complete_text(*, duplicates: int = 1) -> str:
    head = (
        "# Carnot Research - Completed Experiments\n"
        "milestones:\n"
        "- id: 2026.06.386\n"
        "  finding: prior milestone\n"
    )
    block = (
        "- id: 2026.06.387\n"
        "  title: old conductor row\n"
        "  completed: '2026-06-14'\n"
        "  finding: See conductor log for per-experiment results.\n"
        "  tasks:\n"
        "  - id: exp4183-capstone-v387\n"
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
    (root / "research-roadmap.yaml").write_text("milestone: 2026.06.388\n", encoding="utf-8")
    (root / "tests" / "python").mkdir(parents=True, exist_ok=True)
    (root / "tests" / "python" / "test_pipeline_extract.py").write_text(
        "def test_pipeline_extract():\n    assert True\n", encoding="utf-8"
    )
    (root / "tests" / "python" / "test_docs.py").write_text(
        "def test_docs():\n    assert True\n", encoding="utf-8"
    )
    _write_json(root / "results" / "experiment_4183_capstone_v387.json", _capstone())
    return root


def _capstone(**overrides: object) -> dict:
    payload = {
        "experiment_id": 4183,
        "honest_verdict": (
            "complete: capstone_v387_moat_proven_headroom_present_moat_PROVEN-headroom-present_"
            "gap3_BOUNDED_diffusiongemma_MET_arc_levels14"
        ),
        "headline_outcome": "moat_proven_headroom_present",
        "diffusiongemma_gate_status": "MET",
        "verifier_moat_status": "PROVEN-headroom-present",
        "gap3_stage1_status": "BOUNDED",
        "headline_answers": {
            "headroom_controlled_moat_domain": "code",
            "headroom_controlled_moat_positive_control_confirmed": True,
            "headroom_controlled_moat_verifier_value_added": True,
            "gap3_pass2_energy_vs_vote": 0.0,
            "gap3_all_four_gates_pass": False,
            "total_arc_levels_solved": 14,
        },
        "arc_progress": {
            "total_arc_levels_solved": 14,
            "total_arc_games_solved": 13,
            "real_env_confirmed": True,
        },
        "diffusiongemma_gate": {
            "basis": "clean_exp4177_positive_controlled_executable_headroom_moat",
            "met": True,
            "status": "MET",
        },
        "gap3_stage1": {
            "status": "BOUNDED",
            "candidate_auroc": 0.893651,
            "pass2_energy_vs_vote": 0.0,
            "headroom_capture_fraction": 0.0,
            "all_four_gates_pass": False,
            "reaches_proven_arc_headroom": False,
        },
        "registry_gap_hygiene": {
            "moat_verdict": {
                "status": "filled_headroom_controlled_verifier_value_added",
                "domain": "code",
                "verifier_value_added": True,
                "positive_control_confirmed": True,
                "moat_delta_vs_vote": {
                    "arm_a_pass1": 0.84,
                    "arm_b_sc_vote_pass1": 0.66,
                    "delta": 0.18,
                    "ci95": [0.08, 0.30],
                    "n": 50,
                },
                "moat_vs_matched_control": {
                    "arm_a_pass1": 0.84,
                    "arm_c_no_verifier_pass1": 0.66,
                    "delta": 0.18,
                    "n": 50,
                },
                "positive_control": {
                    "oracle_at_k": 0.84,
                    "sc_vote_pass1": 0.66,
                    "oracle_minus_sc_vote": 0.18,
                },
                "accuracy_cost_pareto": {
                    "efficiency_parity": False,
                    "value_added_basis": "accuracy_lift_ci95_excludes_zero",
                    "cost_unit": "candidate_level_selection_operation_generation_budget_held_constant",
                },
            }
        },
    }
    payload.update(overrides)
    return payload


def run_happy(root: Path) -> dict:
    out = mod.run(root, pretest_result=GREEN, started_s=1000.0, now_s=1000.5)
    return json.loads(out.read_text(encoding="utf-8"))


def test_req_report_4184_spec_declares_contract() -> None:
    """REQ-REPORT-4184: OpenSpec declares required fields and scenarios."""

    spec = Path("openspec/capabilities/research-reporting/spec.md").read_text(encoding="utf-8")
    assert "REQ-REPORT-4184" in spec
    assert "SCENARIO-REPORT-4184" in spec
    assert "SCENARIO-REPORT-4184-BLOCKED-PRECONDITION" in spec
    assert "v387_close_state" in spec
    assert "efficiency_parity=false" in spec
    assert "candidate_auroc=0.893651" in spec
    assert "total_levels_solved=14" in spec
    assert mod.FIELD_PRINCIPLES["honest_verdict"] in spec
    assert mod.FIELD_PRINCIPLES["v387_close_state"] in spec
    assert mod.FIELD_PRINCIPLES["preconditions_checked"] in spec


def test_shared_helpers_and_archive_record_editing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-REPORT-4184: helper behavior is deterministic and YAML-safe."""

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
    close_state = mod.build_v387_close_state({"4183": _capstone()})
    deduped, removed, action = mod.dedupe_or_update_record(
        _research_complete_text(duplicates=3), close_state
    )
    assert action == "deduped"
    assert removed == 2
    assert mod.archive_record_count(deduped) == 1
    assert "efficiency axis" in deduped
    assert "DiffusionGemma gate MET" in deduped
    assert mod.yaml_parses(deduped)
    updated, removed2, action2 = mod.dedupe_or_update_record(
        _research_complete_text(duplicates=1), close_state
    )
    assert (removed2, action2) == (0, "updated")
    unchanged, removed4, action4 = mod.dedupe_or_update_record(updated, close_state)
    assert (unchanged, removed4, action4) == (updated, 0, "unchanged")
    old_activation, removed5, action5 = mod.dedupe_or_update_record(
        "milestones:\n- id: 2026.06.387\n  activation_recorded: old\n  tasks:\n  - id: exp4183\n",
        close_state,
    )
    assert (removed5, action5) == (0, "updated")
    assert "activation_recorded: exp4184-archive-v387-activate-v388" in old_activation
    no_tasks, removed6, action6 = mod.dedupe_or_update_record(
        "milestones:\n- id: 2026.06.387\n  title: no tasks\n",
        close_state,
    )
    assert (removed6, action6) == (0, "updated")
    assert "  finding: " in no_tasks
    appended, removed3, action3 = mod.dedupe_or_update_record(
        "# history\nmilestones:\n- id: 2026.06.386\n  finding: prior\n", close_state
    )
    assert (removed3, action3) == (0, "appended")
    assert "activation_recorded: exp4184-archive-v387-activate-v388" in appended


def test_precondition_helpers_and_source_readers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SCENARIO-REPORT-4184-BLOCKED-PRECONDITION: resource probes are explicit."""

    assert mod._milestone_from_text("name: no milestone\n") == "unknown"
    assert mod.read_active_milestone(tmp_path) == ("unknown", "research-roadmap.yaml")
    (tmp_path / "research-roadmap-next.yaml").write_text(
        "milestone: 2026.06.388\n", encoding="utf-8"
    )
    assert mod.read_active_milestone(tmp_path) == ("2026.06.388", "research-roadmap-next.yaml")

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


def test_read_sources_and_build_v387_close_state(tmp_path: Path) -> None:
    """SCENARIO-REPORT-4184: close-state records moat proven and efficiency unwon."""

    root = make_repo(tmp_path)
    sources = mod.read_v387_sources(root)
    assert sources["4183"]["diffusiongemma_gate_status"] == "MET"
    cited = mod.build_cited_upstream(root)
    assert any(item["experiment_id"] == "4183" for item in cited)
    assert all(item["sha256"] is None or mod.is_sha256(item["sha256"]) for item in cited)

    state = mod.build_v387_close_state(sources)
    assert state["summary"] == "moat_proven_accuracy_efficiency_unwon_gap3_bounded"
    assert state["verifier_moat_status"] == "PROVEN-headroom-present"
    assert state["moat_domain"] == "code"
    assert state["verifier_value_added"] is True
    assert state["moat_delta_vs_vote"] == 0.18
    assert state["moat_delta_ci95"] == [0.08, 0.3]
    assert state["matched_control_delta"] == 0.18
    assert state["positive_control_confirmed"] is True
    assert state["efficiency_parity"] is False
    assert state["llm_judge_comparison_done"] is False
    assert state["gap3_stage1_status"] == "BOUNDED"
    assert state["gap3_candidate_auroc"] == 0.893651
    assert state["gap3_pass2_energy_vs_vote"] == 0.0
    assert state["gap3_all_four_gates_pass"] is False
    assert state["diffusiongemma_gate_status"] == "MET"
    assert state["total_levels_solved"] == 14
    assert state["total_games_solved"] == 13
    assert "efficiency axis" in state["v388_planner_frame"]


def test_run_happy_path_writes_valid_artifact_and_updates_archive(tmp_path: Path) -> None:
    """SCENARIO-REPORT-4184: complete path writes the terminal archive artifact."""

    root = make_repo(tmp_path, duplicates=2)
    artifact = run_happy(root)
    assert artifact["honest_verdict"].startswith("success:")
    assert artifact["archived_milestone"] == "2026.06.387"
    assert artifact["activated_milestone"] == "2026.06.388"
    assert artifact["active_milestone_confirmed"] == "2026.06.388"
    assert artifact["research_complete_yaml_parses"] is True
    assert artifact["exclusion_manifest_parses"] is True
    assert artifact["pretest_suite_green"] is True
    assert artifact["research_complete_record_action"] == "deduped"
    assert artifact["research_complete_duplicates_removed"] == 1
    assert artifact["v387_close_state"]["efficiency_parity"] is False
    assert artifact["v387_close_state"]["gap3_stage1_status"] == "BOUNDED"
    assert artifact["v387_close_state"]["diffusiongemma_gate_status"] == "MET"
    assert (
        artifact["field_principles"]["v387_close_state"] == mod.FIELD_PRINCIPLES["v387_close_state"]
    )
    assert artifact["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert mod.is_sha256(artifact["reproducibility_checksum"])
    complete_text = (root / "research-complete.yaml").read_text(encoding="utf-8")
    assert mod.archive_record_count(complete_text) == 1
    assert "PROVEN-headroom-present" in complete_text
    assert "efficiency_parity=false" in complete_text
    mod.validate_artifact(artifact)


def test_run_real_pretest_branch_and_entrypoints_are_injectable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SCENARIO-REPORT-4184: real pretest and CLI entrypoints can be substituted."""

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

    import carnot.experiment_4184_archive_v387_activate_v388 as entrypoint

    called: dict[str, Path] = {}

    def fake_run(root_path: Path = mod.REPO_ROOT) -> Path:
        called["root"] = Path(root_path)
        return root / mod.OUTPUT_REL_PATH

    monkeypatch.setattr(entrypoint, "run", fake_run)
    assert entrypoint.main() == 0
    assert called["root"] == entrypoint.REPO_ROOT

    script_path = Path("results/experiment_4184_archive_v387_activate_v388.py")
    spec = importlib.util.spec_from_file_location("exp4184_archive_script", script_path)
    assert spec and spec.loader
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)
    monkeypatch.setattr(script, "run", fake_run)
    assert script.main() == 0


def test_run_blocked_preconditions_before_research_complete_edit(tmp_path: Path) -> None:
    """SCENARIO-REPORT-4184-BLOCKED-PRECONDITION: blocked paths do not fabricate success."""

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
    (root4 / "research-roadmap.yaml").write_text("milestone: 2026.06.387\n", encoding="utf-8")
    artifact4 = json.loads(mod.run(root4, pretest_result=GREEN).read_text(encoding="utf-8"))
    assert artifact4["honest_verdict"] == "blocked_v388_not_active"

    root5 = make_repo(tmp_path / "red")
    before = (root5 / "research-complete.yaml").read_text(encoding="utf-8")
    artifact5 = json.loads(mod.run(root5, pretest_result=RED).read_text(encoding="utf-8"))
    assert artifact5["honest_verdict"] == "blocked_smart_subset_pretest_not_green"
    assert artifact5["preconditions_checked"]["pretest_suite_green"] is False
    assert (root5 / "research-complete.yaml").read_text(encoding="utf-8") == before

    root6 = make_repo(tmp_path / "capstone_missing")
    (root6 / "results" / "experiment_4183_capstone_v387.json").unlink()
    artifact6 = json.loads(mod.run(root6, pretest_result=GREEN).read_text(encoding="utf-8"))
    assert artifact6["honest_verdict"] == "blocked_v387_capstone_missing"


def test_run_blocked_research_complete_edit_guards(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-REPORT-4184: invalid archive edits are blocked before completion."""

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
    """REQ-REPORT-4184: complete and blocked artifact builders keep schema shape."""

    root = make_repo(tmp_path)
    state = mod.build_v387_close_state(mod.read_v387_sources(root))
    complete = mod.build_complete_artifact(
        v387_close_state=state,
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
    """REQ-REPORT-4184: validation rejects artifacts that launder the .387 truth."""

    good = run_happy(make_repo(tmp_path))

    def set_path(obj: dict, path: list[str], value: object) -> None:
        cur = obj
        for key in path[:-1]:
            cur = cur[key]
        cur[path[-1]] = value

    cases = [
        ("missing required fields", lambda a: a.pop("v387_close_state")),
        ("terminal-prefixed", lambda a: a.__setitem__("honest_verdict", "done")),
        ("field_principles must be a mapping", lambda a: a.__setitem__("field_principles", "x")),
        ("missing field principles", lambda a: a.__setitem__("field_principles", {})),
        (
            "principle must match REQ-REPORT-4184",
            lambda a: a["field_principles"].__setitem__("v387_close_state", "wrong"),
        ),
        ("archived milestone", lambda a: a.__setitem__("archived_milestone", "2026.06.999")),
        ("activated milestone", lambda a: a.__setitem__("activated_milestone", "2026.06.999")),
        ("research-complete YAML", lambda a: a.__setitem__("research_complete_yaml_parses", False)),
        ("exclusion manifest", lambda a: a.__setitem__("exclusion_manifest_parses", False)),
        ("pretest suite", lambda a: a.__setitem__("pretest_suite_green", False)),
        ("active milestone", lambda a: a.__setitem__("active_milestone_confirmed", "2026.06.387")),
        ("v387_close_state must be a mapping", lambda a: a.__setitem__("v387_close_state", "x")),
        (
            "verifier moat",
            lambda a: set_path(a, ["v387_close_state", "verifier_moat_status"], "BOUNDED"),
        ),
        (
            "moat domain",
            lambda a: set_path(a, ["v387_close_state", "moat_domain"], "math"),
        ),
        (
            "verifier_value_added",
            lambda a: set_path(a, ["v387_close_state", "verifier_value_added"], False),
        ),
        (
            "moat_delta_vs_vote",
            lambda a: set_path(a, ["v387_close_state", "moat_delta_vs_vote"], 0.1),
        ),
        (
            "moat_delta_ci95",
            lambda a: set_path(a, ["v387_close_state", "moat_delta_ci95"], [0.0, 0.1]),
        ),
        (
            "matched_control_delta",
            lambda a: set_path(a, ["v387_close_state", "matched_control_delta"], 0.1),
        ),
        (
            "positive_control_confirmed",
            lambda a: set_path(a, ["v387_close_state", "positive_control_confirmed"], False),
        ),
        (
            "efficiency_parity",
            lambda a: set_path(a, ["v387_close_state", "efficiency_parity"], True),
        ),
        (
            "LLM-as-judge",
            lambda a: set_path(a, ["v387_close_state", "llm_judge_comparison_done"], True),
        ),
        (
            "GAP-3",
            lambda a: set_path(a, ["v387_close_state", "gap3_stage1_status"], "FILLED"),
        ),
        (
            "candidate_auroc",
            lambda a: set_path(a, ["v387_close_state", "gap3_candidate_auroc"], 0.7),
        ),
        (
            "gap3_pass2_energy_vs_vote",
            lambda a: set_path(a, ["v387_close_state", "gap3_pass2_energy_vs_vote"], 0.1),
        ),
        (
            "gap3_all_four_gates_pass",
            lambda a: set_path(a, ["v387_close_state", "gap3_all_four_gates_pass"], True),
        ),
        (
            "DiffusionGemma",
            lambda a: set_path(
                a, ["v387_close_state", "diffusiongemma_gate_status"], "STILL-PENDING"
            ),
        ),
        (
            "total levels solved",
            lambda a: set_path(a, ["v387_close_state", "total_levels_solved"], 13),
        ),
        (
            "total games solved",
            lambda a: set_path(a, ["v387_close_state", "total_games_solved"], 12),
        ),
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
