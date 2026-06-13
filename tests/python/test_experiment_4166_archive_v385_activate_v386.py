"""Tests for Exp 4166 `.385` archive / `.386` activation.

Spec refs: REQ-REPORT-4166, SCENARIO-REPORT-4166,
SCENARIO-REPORT-4166-BLOCKED-PRECONDITION.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from carnot.reporting import archive_v385_activate_v386_4166 as mod


GREEN = mod.CommandResult(command=["pytest"], exit_code=0, stdout="81 passed", stderr="")
RED = mod.CommandResult(command=["pytest"], exit_code=1, stdout="FAILED smart subset", stderr="")


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _research_complete_text(*, duplicates: int = 1) -> str:
    head = (
        "# Carnot Research - Completed Experiments\n"
        "milestones:\n"
        "- id: 2026.06.384\n"
        "  finding: prior milestone\n"
    )
    block = (
        "- id: 2026.06.385\n"
        "  title: old conductor row\n"
        "  completed: '2026-06-13'\n"
        "  finding: See conductor log for per-experiment results.\n"
        "  tasks:\n"
        "  - id: exp4165-capstone-v385\n"
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
    (root / "research-roadmap.yaml").write_text("milestone: 2026.06.386\n", encoding="utf-8")
    (root / "nano-trm" / "src" / "nn").mkdir(parents=True, exist_ok=True)
    (root / "nano-trm" / "src" / "nn" / "train.py").write_text("# train\n", encoding="utf-8")
    (root / "tests" / "python").mkdir(parents=True, exist_ok=True)
    (root / "tests" / "python" / "test_pipeline_extract.py").write_text(
        "def test_pipeline_extract():\n    assert True\n", encoding="utf-8"
    )
    (root / "tests" / "python" / "test_docs.py").write_text(
        "def test_docs():\n    assert True\n", encoding="utf-8"
    )
    (root / "results" / "trm_runs").mkdir(parents=True, exist_ok=True)
    (root / "results" / "trm_runs" / "contiguous_run.pid").write_text(
        "999999\n", encoding="utf-8"
    )
    _write_json(
        root / "results" / "experiment_4157_baseline_harvest_contiguous_continue.json",
        {
            "honest_verdict": "blocked_noop_step_unchanged",
            "flagged_adversarial": True,
            "current_val": 0.501041650772,
            "max_val": 0.501041650772,
            "baseline_faithful": False,
            "blocked_cause": "manual_lr_step did not advance: before=19006 after=19006; trainer_return_code=1; torch.OutOfMemoryError",
            "liveness": {
                "detail": "task-launched contiguous run",
                "pid": 447577,
                "process_alive": False,
                "run_alive": False,
            },
            "task_launched_run": {
                "manual_lr_step_advanced": False,
                "manual_lr_step_after": 19006,
                "manual_lr_step_before": 19006,
                "new_val_row_written": False,
                "return_code": 1,
                "val_rows_after": 27,
                "val_rows_before": 27,
            },
        },
    )
    _write_json(
        root / "results" / "experiment_4158_verifier_rerank_recovery_moat.json",
        {
            "honest_verdict": "complete: no_headroom_rerank_uninformative_at_val_0.50",
            "flagged_adversarial": True,
            "baseline_status": {"current_val": 0.501041650772, "max_val": 0.501041650772},
        },
    )
    _write_json(
        root / "results" / "experiment_4159_decisive_verifier_reward_graft.json",
        {
            "honest_verdict": "complete: graft_deferred_baseline_below_0.85",
            "flagged_adversarial": True,
            "graft_deferred": True,
            "verifier_value_added": False,
            "baseline_status": {"current_val": 0.501041650772, "faithful": False},
        },
    )
    _write_json(
        root / "results" / "experiment_4165_capstone_v385.json",
        {
            "baseline_val_trajectory": {
                "seed_val": 0.278172343969,
                "status": "skipped_flagged_adversarial",
                "source_status": "skipped_flagged_adversarial",
                "current_val": None,
                "run_alive": None,
            },
            "headline_answers": {
                "current_val_vs_seed": {"seed_val": 0.278172343969},
                "total_games_solved": 13,
            },
            "headline_outcome": "accumulation_still_blocked",
            "honest_verdict": "blocked: capstone_v385_accumulation_still_blocked",
            "diffusiongemma_gate_status": "STILL-PENDING",
            "flagged_artifacts_skipped": [
                {
                    "experiment_id": 4157,
                    "path": "results/experiment_4157_baseline_harvest_contiguous_continue.json",
                    "reason": "flagged_adversarial:true",
                    "sha256": "a" * 64,
                },
                {
                    "experiment_id": 4158,
                    "path": "results/experiment_4158_verifier_rerank_recovery_moat.json",
                    "reason": "flagged_adversarial:true",
                    "sha256": "b" * 64,
                },
                {
                    "experiment_id": 4159,
                    "path": "results/experiment_4159_decisive_verifier_reward_graft.json",
                    "reason": "flagged_adversarial:true",
                    "sha256": "c" * 64,
                },
                {
                    "experiment_id": 4163,
                    "path": "results/experiment_4163_verifier_registry_gaps_hygiene.json",
                    "reason": "flagged_adversarial:true",
                    "sha256": "d" * 64,
                },
            ],
            "total_games_solved": 13,
        },
    )
    return root


def run_happy(root: Path) -> dict:
    out = mod.run(root, pretest_result=GREEN, started_s=1000.0, now_s=1000.5)
    return json.loads(out.read_text(encoding="utf-8"))


def test_req_report_4166_spec_declares_contract() -> None:
    """REQ-REPORT-4166: OpenSpec declares fields, scenarios, and stand-down rule."""

    spec = Path("openspec/capabilities/research-reporting/spec.md").read_text(encoding="utf-8")
    assert "REQ-REPORT-4166" in spec
    assert "SCENARIO-REPORT-4166" in spec
    assert "SCENARIO-REPORT-4166-BLOCKED-PRECONDITION" in spec
    assert "v385_close_state" in spec
    assert "preconditions_checked" in spec
    assert "conductor ceded TRM training to the outer-loop" in spec
    assert "blocked_noop_step_unchanged" in spec
    assert "must not launch TRM training" in spec
    assert "0.278172343969" in spec


def test_shared_helpers_and_archive_record_editing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-REPORT-4166: helper behavior is deterministic and YAML-safe."""

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

    assert mod.archive_record_count(_research_complete_text(duplicates=3)) == 3
    close_state = {"baseline_seed_val_exact_accuracy": 0.278172343969}
    assert mod._insert_before_tasks(["  title: no tasks"], "  finding: x") == [
        "  title: no tasks",
        "  finding: x",
    ]
    deduped, removed, action = mod.dedupe_or_update_record(
        _research_complete_text(duplicates=3), close_state
    )
    assert action == "deduped"
    assert removed == 2
    assert mod.archive_record_count(deduped) == 1
    assert "conductor STANDS DOWN" in deduped
    assert "0.278172343969" in deduped
    assert mod.yaml_parses(deduped)
    updated, removed2, action2 = mod.dedupe_or_update_record(
        _research_complete_text(duplicates=1), close_state
    )
    assert (removed2, action2) == (0, "updated")
    unchanged, removed3, action3 = mod.dedupe_or_update_record(updated, close_state)
    assert (unchanged, removed3, action3) == (updated, 0, "unchanged")
    appended, removed4, action4 = mod.dedupe_or_update_record(
        "# history\nmilestones:\n- id: 2026.06.384\n  finding: prior\n", close_state
    )
    assert (removed4, action4) == (0, "appended")
    assert "activation_recorded: exp4166-archive-v385-activate-v386" in appended
    missing_finding, removed5, action5 = mod.dedupe_or_update_record(
        "milestones:\n- id: 2026.06.385\n  title: no finding\n  tasks:\n  - id: exp4165\n",
        close_state,
    )
    assert (removed5, action5) == (0, "updated")
    assert "  finding: " in missing_finding


def test_precondition_helpers_and_source_readers(tmp_path: Path) -> None:
    """SCENARIO-REPORT-4166-BLOCKED-PRECONDITION: resource probes are explicit."""

    assert mod._milestone_from_text("name: no milestone\n") == "unknown"
    assert mod.read_active_milestone(tmp_path) == ("unknown", "research-roadmap.yaml")
    (tmp_path / "research-roadmap-next.yaml").write_text("milestone: 2026.06.386\n", encoding="utf-8")
    assert mod.read_active_milestone(tmp_path) == ("2026.06.386", "research-roadmap-next.yaml")
    assert mod.train_file_present(tmp_path) is False
    train = tmp_path / mod.NANO_TRM_TRAIN_REL_PATH
    train.parent.mkdir(parents=True, exist_ok=True)
    train.write_text("# train\n", encoding="utf-8")
    assert mod.train_file_present(tmp_path) is True

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
    assert mod.read_outerloop_pid(root) == 999999
    assert mod.read_outerloop_pid(tmp_path / "empty") is None
    assert mod.pid_is_alive(None) is False
    assert mod.pid_is_alive(999999) is False
    monkeypatch = pytest.MonkeyPatch()
    try:
        monkeypatch.setattr(mod.os, "kill", lambda pid, sig: None)
        assert mod.pid_is_alive(123) is True
    finally:
        monkeypatch.undo()
    targets = mod.smart_subset_targets(root)
    assert "tests/python/test_pipeline_extract.py" in targets
    assert mod.smart_subset_command(targets)[0] == str(mod.PYTEST_BIN)
    assert mod.smart_subset_targets(tmp_path / "empty") == [mod.CORE_SMART_SUBSET[0]]
    assert mod._run_command(["true"], tmp_path).exit_code == 0
    assert mod._run_command(["definitely-not-a-real-binary-xyz"], tmp_path).exit_code == 127
    monkeypatch = pytest.MonkeyPatch()
    try:
        monkeypatch.setattr(mod, "_run_command", lambda command, root_path: GREEN)
        assert mod.run_smart_subset(root).exit_code == 0
    finally:
        monkeypatch.undo()


def test_read_sources_and_build_v385_close_state(tmp_path: Path) -> None:
    """SCENARIO-REPORT-4166: close-state records no-op collision and handoff."""

    root = make_repo(tmp_path)
    sources = mod.read_v385_sources(root)
    assert sources["4157"]["honest_verdict"] == "blocked_noop_step_unchanged"
    cited = mod.build_cited_upstream(root)
    assert any(item["experiment_id"] == "4165" for item in cited)
    assert all(item["sha256"] is None or mod.is_sha256(item["sha256"]) for item in cited)
    state = mod.build_v385_close_state(sources, outerloop_pid=999999, outerloop_alive=False)
    assert state["conductor_ceded_trm_training_to_outer_loop"] is True
    assert state["outer_loop_owns_contiguous_training"] is True
    assert state["conductor_stands_down_on_trm_training"] is True
    assert state["no_conductor_training_rule"] is True
    assert state["bounded_accumulation_noop_again"] is True
    assert state["conductor_continuation_collided_with_outer_loop"] is True
    assert state["manual_lr_step_advanced"] is False
    assert state["new_val_row_written"] is False
    assert state["baseline_seed_val_exact_accuracy"] == 0.278172343969
    assert state["headline_outcome"] == "accumulation_still_blocked"
    assert state["flagged_artifacts_skipped_count"] == 4
    assert state["flagged_artifact_ids"] == [4157, 4158, 4159, 4163]
    assert state["total_games_solved"] == 13
    assert state["diffusiongemma_gate_status"] == "STILL-PENDING"

    default_state = mod.build_v385_close_state({}, outerloop_pid=None, outerloop_alive=False)
    assert default_state["baseline_seed_val_exact_accuracy"] == 0.278172343969
    assert default_state["flagged_artifacts_skipped_count"] == 0


def test_run_happy_path_writes_valid_artifact_and_updates_archive(tmp_path: Path) -> None:
    """SCENARIO-REPORT-4166: complete path writes the terminal archive artifact."""

    root = make_repo(tmp_path, duplicates=2)
    artifact = run_happy(root)
    assert artifact["honest_verdict"].startswith("success:")
    assert artifact["archived_milestone"] == "2026.06.385"
    assert artifact["activated_milestone"] == "2026.06.386"
    assert artifact["active_milestone_confirmed"] == "2026.06.386"
    assert artifact["research_complete_yaml_parses"] is True
    assert artifact["exclusion_manifest_parses"] is True
    assert artifact["nano_trm_train_present"] is True
    assert artifact["pretest_suite_green"] is True
    assert artifact["research_complete_record_action"] == "deduped"
    assert artifact["research_complete_duplicates_removed"] == 1
    assert artifact["v385_close_state"]["outer_loop_owns_contiguous_training"] is True
    assert artifact["v385_close_state"]["baseline_seed_val_exact_accuracy"] == 0.278172343969
    assert artifact["field_principles"]["v385_close_state"] == mod.FIELD_PRINCIPLES["v385_close_state"]
    assert artifact["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert mod.is_sha256(artifact["reproducibility_checksum"])
    complete_text = (root / "research-complete.yaml").read_text(encoding="utf-8")
    assert mod.archive_record_count(complete_text) == 1
    assert "conductor STANDS DOWN" in complete_text
    assert "outer-loop owns the contiguous training" in complete_text
    mod.validate_artifact(artifact)


def test_run_real_pretest_branch_and_entrypoint_are_injectable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SCENARIO-REPORT-4166: real pretest and CLI entrypoint can be substituted."""

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

    import carnot.experiment_4166_archive_v385_activate_v386 as entrypoint

    called: dict[str, Path] = {}

    def fake_run(root_path: Path = mod.REPO_ROOT) -> Path:
        called["root"] = Path(root_path)
        return root / mod.OUTPUT_REL_PATH

    monkeypatch.setattr(entrypoint, "run", fake_run)
    assert entrypoint.main() == 0
    assert called["root"] == entrypoint.REPO_ROOT


def test_run_blocked_preconditions_before_research_complete_edit(tmp_path: Path) -> None:
    """SCENARIO-REPORT-4166-BLOCKED-PRECONDITION: blocked paths do not fabricate success."""

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

    root4 = make_repo(tmp_path / "train_missing")
    (root4 / mod.NANO_TRM_TRAIN_REL_PATH).unlink()
    artifact4 = json.loads(mod.run(root4, pretest_result=GREEN).read_text(encoding="utf-8"))
    assert artifact4["honest_verdict"] == "blocked_nano_trm_train_missing"

    root5 = make_repo(tmp_path / "wrong_milestone")
    (root5 / "research-roadmap.yaml").write_text("milestone: 2026.06.385\n", encoding="utf-8")
    artifact5 = json.loads(mod.run(root5, pretest_result=GREEN).read_text(encoding="utf-8"))
    assert artifact5["honest_verdict"] == "blocked_v386_not_active"

    root6 = make_repo(tmp_path / "red")
    before = (root6 / "research-complete.yaml").read_text(encoding="utf-8")
    artifact6 = json.loads(mod.run(root6, pretest_result=RED).read_text(encoding="utf-8"))
    assert artifact6["honest_verdict"] == "blocked_smart_subset_pretest_not_green"
    assert artifact6["preconditions_checked"]["pretest_suite_green"] is False
    assert (root6 / "research-complete.yaml").read_text(encoding="utf-8") == before

    root7 = make_repo(tmp_path / "capstone_missing")
    (root7 / "results" / "experiment_4165_capstone_v385.json").unlink()
    artifact7 = json.loads(mod.run(root7, pretest_result=GREEN).read_text(encoding="utf-8"))
    assert artifact7["honest_verdict"] == "blocked_v385_capstone_missing"


def test_run_blocked_research_complete_edit_guards(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-REPORT-4166: invalid archive edits are blocked before completion."""

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
    """REQ-REPORT-4166: complete and blocked artifact builders keep schema shape."""

    root = make_repo(tmp_path)
    state = mod.build_v385_close_state(mod.read_v385_sources(root), outerloop_pid=999999, outerloop_alive=False)
    complete = mod.build_complete_artifact(
        v385_close_state=state,
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
    """REQ-REPORT-4166: validation rejects artifacts that launder the .385 truth."""

    good = run_happy(make_repo(tmp_path))

    def set_path(obj: dict, path: list[str], value: object) -> None:
        cur = obj
        for key in path[:-1]:
            cur = cur[key]
        cur[path[-1]] = value

    cases = [
        ("missing required fields", lambda a: a.pop("v385_close_state")),
        ("terminal-prefixed", lambda a: a.__setitem__("honest_verdict", "done")),
        ("field_principles must be a mapping", lambda a: a.__setitem__("field_principles", "x")),
        ("missing field principles", lambda a: a.__setitem__("field_principles", {})),
        (
            "principle must match REQ-REPORT-4166",
            lambda a: a["field_principles"].__setitem__("v385_close_state", "wrong"),
        ),
        ("archived milestone", lambda a: a.__setitem__("archived_milestone", "2026.06.999")),
        ("activated milestone", lambda a: a.__setitem__("activated_milestone", "2026.06.999")),
        ("research-complete YAML", lambda a: a.__setitem__("research_complete_yaml_parses", False)),
        ("exclusion manifest", lambda a: a.__setitem__("exclusion_manifest_parses", False)),
        ("nano-trm train", lambda a: a.__setitem__("nano_trm_train_present", False)),
        ("pretest suite", lambda a: a.__setitem__("pretest_suite_green", False)),
        ("active milestone", lambda a: a.__setitem__("active_milestone_confirmed", "2026.06.385")),
        ("v385_close_state must be a mapping", lambda a: a.__setitem__("v385_close_state", "x")),
        (
            "conductor_ceded_trm_training_to_outer_loop",
            lambda a: set_path(a, ["v385_close_state", "conductor_ceded_trm_training_to_outer_loop"], False),
        ),
        (
            "outer_loop_owns_contiguous_training",
            lambda a: set_path(a, ["v385_close_state", "outer_loop_owns_contiguous_training"], False),
        ),
        (
            "conductor_stands_down_on_trm_training",
            lambda a: set_path(a, ["v385_close_state", "conductor_stands_down_on_trm_training"], False),
        ),
        (
            "bounded_accumulation_noop_again",
            lambda a: set_path(a, ["v385_close_state", "bounded_accumulation_noop_again"], False),
        ),
        (
            "conductor_continuation_collided_with_outer_loop",
            lambda a: set_path(a, ["v385_close_state", "conductor_continuation_collided_with_outer_loop"], False),
        ),
        (
            "baseline seed",
            lambda a: set_path(a, ["v385_close_state", "baseline_seed_val_exact_accuracy"], 0.5),
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
