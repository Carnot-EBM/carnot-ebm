"""Tests for Exp 4174 `.386` archive / `.387` activation.

Spec refs: REQ-REPORT-4174, SCENARIO-REPORT-4174,
SCENARIO-REPORT-4174-BLOCKED-PRECONDITION.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from carnot.reporting import archive_v386_activate_v387_4174 as mod


GREEN = mod.CommandResult(command=["pytest"], exit_code=0, stdout="81 passed", stderr="")
RED = mod.CommandResult(command=["pytest"], exit_code=1, stdout="FAILED smart subset", stderr="")


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _research_complete_text(*, duplicates: int = 1) -> str:
    head = (
        "# Carnot Research - Completed Experiments\n"
        "milestones:\n"
        "- id: 2026.06.385\n"
        "  finding: prior milestone\n"
    )
    block = (
        "- id: 2026.06.386\n"
        "  title: old conductor row\n"
        "  completed: '2026-06-14'\n"
        "  finding: See conductor log for per-experiment results.\n"
        "  tasks:\n"
        "  - id: exp4173-capstone-v386\n"
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
    (root / "research-roadmap.yaml").write_text("milestone: 2026.06.387\n", encoding="utf-8")
    (root / "tests" / "python").mkdir(parents=True, exist_ok=True)
    (root / "tests" / "python" / "test_pipeline_extract.py").write_text(
        "def test_pipeline_extract():\n    assert True\n", encoding="utf-8"
    )
    (root / "tests" / "python" / "test_docs.py").write_text(
        "def test_docs():\n    assert True\n", encoding="utf-8"
    )
    _write_json(
        root / "results" / "experiment_4173_capstone_v386.json",
        {
            "diffusiongemma_gate_status": "STILL-PENDING",
            "headline_outcome": "outerloop_training_in_progress",
            "honest_verdict": "complete: capstone_v386_outerloop_training_in_progress",
            "total_arc_games_solved": 13,
            "total_games_solved": 13,
            "baseline_val_trajectory": {
                "current_val_exact_accuracy": 0.504166662693,
                "outerloop_train_alive": True,
            },
            "defensive_graft_verdict": {
                "graft_deferred": True,
                "verifier_value_added": False,
            },
        },
    )
    _write_json(
        root / "results" / "experiment_4168_decisive_verifier_graft_v2_gate082.json",
        {
            "baseline_status": {
                "bestval_exact_accuracy": 0.822656,
                "faithful_threshold": 0.82,
                "outerloop_pid_alive": False,
                "pid_not_alive_passed": True,
                "gpu_train_stopped_passed": True,
            },
            "graft_deferred": False,
            "honest_verdict": "complete: A~=B null",
            "rerank_lift_vs_vote": {
                "ci95": [0.0, 0.046875],
                "delta": 0.015625,
                "headroom_present": True,
                "oracle_at_k": 0.8125,
                "status": "headroom_backed_null_ci95_includes_zero",
                "verifier_pass_at_1": 0.8125,
                "vote_at_1": 0.796875,
            },
            "rft_vs_ablation_delta": {
                "ci95": [0.0, 0.057692],
                "delta": 0.019231,
                "status": "honest_null_ci95_includes_zero",
            },
            "verifier_value_added": False,
        },
    )
    return root


def run_happy(root: Path) -> dict:
    out = mod.run(root, pretest_result=GREEN, started_s=1000.0, now_s=1000.5)
    return json.loads(out.read_text(encoding="utf-8"))


def test_req_report_4174_spec_declares_contract() -> None:
    """REQ-REPORT-4174: OpenSpec declares required fields and scenarios."""

    spec = Path("openspec/capabilities/research-reporting/spec.md").read_text(encoding="utf-8")
    assert "REQ-REPORT-4174" in spec
    assert "SCENARIO-REPORT-4174" in spec
    assert "SCENARIO-REPORT-4174-BLOCKED-PRECONDITION" in spec
    assert "v386_close_state" in spec
    assert "headroom-limited null" in spec
    assert "0.822656" in spec
    assert "0.8125" in spec
    assert mod.FIELD_PRINCIPLES["honest_verdict"] in spec
    assert mod.FIELD_PRINCIPLES["v386_close_state"] in spec
    assert mod.FIELD_PRINCIPLES["preconditions_checked"] in spec


def test_shared_helpers_and_archive_record_editing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-REPORT-4174: helper behavior is deterministic and YAML-safe."""

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
    assert out.read_text(encoding="utf-8").startswith("{\n  \"a\"")
    assert mod._contains_zero("not-ci") is False
    assert mod._insert_before_tasks(["  title: no tasks"], "  finding: x") == [
        "  title: no tasks",
        "  finding: x",
    ]

    assert mod.archive_record_count(_research_complete_text(duplicates=3)) == 3
    close_state = {"baseline_bestval_exact_accuracy": 0.822656}
    deduped, removed, action = mod.dedupe_or_update_record(
        _research_complete_text(duplicates=3), close_state
    )
    assert action == "deduped"
    assert removed == 2
    assert mod.archive_record_count(deduped) == 1
    assert "headroom-limited null" in deduped
    assert "0.822656" in deduped
    assert mod.yaml_parses(deduped)
    updated, removed2, action2 = mod.dedupe_or_update_record(
        _research_complete_text(duplicates=1), close_state
    )
    assert (removed2, action2) == (0, "updated")
    unchanged, removed4, action4 = mod.dedupe_or_update_record(updated, close_state)
    assert (unchanged, removed4, action4) == (updated, 0, "unchanged")
    old_activation, removed5, action5 = mod.dedupe_or_update_record(
        "milestones:\n"
        "- id: 2026.06.386\n"
        "  activation_recorded: old\n"
        "  tasks:\n"
        "  - id: exp4173\n",
        close_state,
    )
    assert (removed5, action5) == (0, "updated")
    assert "activation_recorded: exp4174-archive-v386-activate-v387" in old_activation
    no_tasks, removed6, action6 = mod.dedupe_or_update_record(
        "milestones:\n- id: 2026.06.386\n  title: no tasks\n",
        close_state,
    )
    assert (removed6, action6) == (0, "updated")
    assert "  finding: " in no_tasks
    appended, removed3, action3 = mod.dedupe_or_update_record(
        "# history\nmilestones:\n- id: 2026.06.385\n  finding: prior\n", close_state
    )
    assert (removed3, action3) == (0, "appended")
    assert "activation_recorded: exp4174-archive-v386-activate-v387" in appended


def test_precondition_helpers_and_source_readers(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """SCENARIO-REPORT-4174-BLOCKED-PRECONDITION: resource probes are explicit."""

    assert mod._milestone_from_text("name: no milestone\n") == "unknown"
    assert mod.read_active_milestone(tmp_path) == ("unknown", "research-roadmap.yaml")
    (tmp_path / "research-roadmap-next.yaml").write_text("milestone: 2026.06.387\n", encoding="utf-8")
    assert mod.read_active_milestone(tmp_path) == ("2026.06.387", "research-roadmap-next.yaml")

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


def test_read_sources_and_build_v386_close_state(tmp_path: Path) -> None:
    """SCENARIO-REPORT-4174: close-state records the headroom-limited null."""

    root = make_repo(tmp_path)
    sources = mod.read_v386_sources(root)
    assert sources["4173"]["diffusiongemma_gate_status"] == "STILL-PENDING"
    cited = mod.build_cited_upstream(root)
    assert any(item["experiment_id"] == "4173" for item in cited)
    assert all(item["sha256"] is None or mod.is_sha256(item["sha256"]) for item in cited)

    state = mod.build_v386_close_state(sources)
    assert state["outer_loop_trm_training_done"] is True
    assert state["conductor_stands_down_on_trm_training"] is True
    assert state["baseline_bestval_exact_accuracy"] == 0.822656
    assert state["baseline_val_rounded"] == 0.8227
    assert state["decisive_graft_fired"] is True
    assert state["graft_deferred"] is False
    assert state["verifier_value_added"] is False
    assert state["headroom_limited_null"] is True
    assert state["true_verifier_failure"] is False
    assert state["oracle_at_k"] == 0.8125
    assert state["oracle_approximately_baseline"] is True
    assert state["diffusiongemma_gate_status"] == "STILL-PENDING"
    assert state["total_games_solved"] == 13
    assert "positive-control" in state["v387_planner_frame"]


def test_run_happy_path_writes_valid_artifact_and_updates_archive(tmp_path: Path) -> None:
    """SCENARIO-REPORT-4174: complete path writes the terminal archive artifact."""

    root = make_repo(tmp_path, duplicates=2)
    artifact = run_happy(root)
    assert artifact["honest_verdict"].startswith("success:")
    assert artifact["archived_milestone"] == "2026.06.386"
    assert artifact["activated_milestone"] == "2026.06.387"
    assert artifact["active_milestone_confirmed"] == "2026.06.387"
    assert artifact["research_complete_yaml_parses"] is True
    assert artifact["exclusion_manifest_parses"] is True
    assert artifact["pretest_suite_green"] is True
    assert artifact["research_complete_record_action"] == "deduped"
    assert artifact["research_complete_duplicates_removed"] == 1
    assert artifact["v386_close_state"]["headroom_limited_null"] is True
    assert artifact["v386_close_state"]["baseline_val_rounded"] == 0.8227
    assert artifact["v386_close_state"]["oracle_at_k"] == 0.8125
    assert artifact["field_principles"]["v386_close_state"] == mod.FIELD_PRINCIPLES["v386_close_state"]
    assert artifact["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert mod.is_sha256(artifact["reproducibility_checksum"])
    complete_text = (root / "research-complete.yaml").read_text(encoding="utf-8")
    assert mod.archive_record_count(complete_text) == 1
    assert "headroom-limited null" in complete_text
    assert "positive-control" in complete_text
    mod.validate_artifact(artifact)


def test_run_real_pretest_branch_and_entrypoints_are_injectable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SCENARIO-REPORT-4174: real pretest and CLI entrypoints can be substituted."""

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

    import carnot.experiment_4174_archive_v386_activate_v387 as entrypoint

    called: dict[str, Path] = {}

    def fake_run(root_path: Path = mod.REPO_ROOT) -> Path:
        called["root"] = Path(root_path)
        return root / mod.OUTPUT_REL_PATH

    monkeypatch.setattr(entrypoint, "run", fake_run)
    assert entrypoint.main() == 0
    assert called["root"] == entrypoint.REPO_ROOT


def test_run_blocked_preconditions_before_research_complete_edit(tmp_path: Path) -> None:
    """SCENARIO-REPORT-4174-BLOCKED-PRECONDITION: blocked paths do not fabricate success."""

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
    (root4 / "research-roadmap.yaml").write_text("milestone: 2026.06.386\n", encoding="utf-8")
    artifact4 = json.loads(mod.run(root4, pretest_result=GREEN).read_text(encoding="utf-8"))
    assert artifact4["honest_verdict"] == "blocked_v387_not_active"

    root5 = make_repo(tmp_path / "red")
    before = (root5 / "research-complete.yaml").read_text(encoding="utf-8")
    artifact5 = json.loads(mod.run(root5, pretest_result=RED).read_text(encoding="utf-8"))
    assert artifact5["honest_verdict"] == "blocked_smart_subset_pretest_not_green"
    assert artifact5["preconditions_checked"]["pretest_suite_green"] is False
    assert (root5 / "research-complete.yaml").read_text(encoding="utf-8") == before

    root6 = make_repo(tmp_path / "capstone_missing")
    (root6 / "results" / "experiment_4173_capstone_v386.json").unlink()
    artifact6 = json.loads(mod.run(root6, pretest_result=GREEN).read_text(encoding="utf-8"))
    assert artifact6["honest_verdict"] == "blocked_v386_capstone_missing"


def test_run_blocked_research_complete_edit_guards(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-REPORT-4174: invalid archive edits are blocked before completion."""

    root = make_repo(tmp_path / "invalid")
    monkeypatch.setattr(mod, "dedupe_or_update_record", lambda text, state: ("a: : :\n- [", 0, "appended"))
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
    """REQ-REPORT-4174: complete and blocked artifact builders keep schema shape."""

    root = make_repo(tmp_path)
    state = mod.build_v386_close_state(mod.read_v386_sources(root))
    complete = mod.build_complete_artifact(
        v386_close_state=state,
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
    """REQ-REPORT-4174: validation rejects artifacts that launder the .386 truth."""

    good = run_happy(make_repo(tmp_path))

    def set_path(obj: dict, path: list[str], value: object) -> None:
        cur = obj
        for key in path[:-1]:
            cur = cur[key]
        cur[path[-1]] = value

    cases = [
        ("missing required fields", lambda a: a.pop("v386_close_state")),
        ("terminal-prefixed", lambda a: a.__setitem__("honest_verdict", "done")),
        ("field_principles must be a mapping", lambda a: a.__setitem__("field_principles", "x")),
        ("missing field principles", lambda a: a.__setitem__("field_principles", {})),
        (
            "principle must match REQ-REPORT-4174",
            lambda a: a["field_principles"].__setitem__("v386_close_state", "wrong"),
        ),
        ("archived milestone", lambda a: a.__setitem__("archived_milestone", "2026.06.999")),
        ("activated milestone", lambda a: a.__setitem__("activated_milestone", "2026.06.999")),
        ("research-complete YAML", lambda a: a.__setitem__("research_complete_yaml_parses", False)),
        ("exclusion manifest", lambda a: a.__setitem__("exclusion_manifest_parses", False)),
        ("pretest suite", lambda a: a.__setitem__("pretest_suite_green", False)),
        ("active milestone", lambda a: a.__setitem__("active_milestone_confirmed", "2026.06.386")),
        ("v386_close_state must be a mapping", lambda a: a.__setitem__("v386_close_state", "x")),
        (
            "outer_loop_trm_training_done",
            lambda a: set_path(a, ["v386_close_state", "outer_loop_trm_training_done"], False),
        ),
        (
            "conductor_stands_down_on_trm_training",
            lambda a: set_path(a, ["v386_close_state", "conductor_stands_down_on_trm_training"], False),
        ),
        (
            "decisive_graft_fired",
            lambda a: set_path(a, ["v386_close_state", "decisive_graft_fired"], False),
        ),
        (
            "verifier_value_added",
            lambda a: set_path(a, ["v386_close_state", "verifier_value_added"], True),
        ),
        (
            "headroom_limited_null",
            lambda a: set_path(a, ["v386_close_state", "headroom_limited_null"], False),
        ),
        (
            "true_verifier_failure",
            lambda a: set_path(a, ["v386_close_state", "true_verifier_failure"], True),
        ),
        ("baseline", lambda a: set_path(a, ["v386_close_state", "baseline_bestval_exact_accuracy"], 0.5)),
        ("oracle_at_k", lambda a: set_path(a, ["v386_close_state", "oracle_at_k"], 0.7)),
        ("DiffusionGemma", lambda a: set_path(a, ["v386_close_state", "diffusiongemma_gate_status"], "RESOLVED-null")),
        ("total games", lambda a: set_path(a, ["v386_close_state", "total_games_solved"], 14)),
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
