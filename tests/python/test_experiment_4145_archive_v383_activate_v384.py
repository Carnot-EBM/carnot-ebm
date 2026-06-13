"""Tests for Exp 4145 .383 archive / .384 activation.

Spec refs: REQ-REPORT-4145, SCENARIO-REPORT-4145,
SCENARIO-REPORT-4145-BLOCKED-PRECONDITION.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from carnot.reporting import archive_v383_activate_v384_4145 as mod


GREEN = mod.CommandResult(command=["pytest"], exit_code=0, stdout="81 passed", stderr="")
RED = mod.CommandResult(command=["pytest"], exit_code=1, stdout="FAILED smart subset", stderr="")


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _research_complete_text(*, duplicates: int = 1) -> str:
    head = (
        "# Carnot Research - Completed Experiments\n"
        "milestones:\n"
        "- id: 2026.06.382\n"
        "  finding: prior milestone\n"
    )
    block = (
        "- id: 2026.06.383\n"
        "  title: old conductor row\n"
        "  completed: '2026-06-13'\n"
        "  finding: See conductor log for per-experiment results.\n"
        "  tasks:\n"
        "  - id: exp4144-capstone-v383\n"
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
    (root / "research-roadmap.yaml").write_text("milestone: 2026.06.384\n", encoding="utf-8")
    (root / "nano-trm" / "src" / "nn").mkdir(parents=True, exist_ok=True)
    (root / "nano-trm" / "src" / "nn" / "train.py").write_text("# train\n", encoding="utf-8")
    (root / "tests" / "python").mkdir(parents=True, exist_ok=True)
    (root / "tests" / "python" / "test_pipeline_extract.py").write_text(
        "def test_pipeline_extract():\n    assert True\n", encoding="utf-8"
    )
    (root / "tests" / "python" / "test_docs.py").write_text(
        "def test_docs():\n    assert True\n", encoding="utf-8"
    )
    _write_json(
        root / "results" / "experiment_4135_sudoku_accumulate_pass1_fixed_lr.json",
        {
            "duration_s": 6.99,
            "exact_accuracy_metric": None,
            "exact_accuracy_metrics_path": None,
            "flagged_adversarial": True,
            "honest_verdict": "complete: missing_real_val_exact_accuracy",
            "lr_continued_not_rewarmed": False,
            "stable_checkpoint_path": "results/trm_runs/sudoku_extreme_baseline/last.ckpt",
        },
    )
    _write_json(
        root / "results" / "experiment_4138_sudoku_accumulate_pass4_convergence_check.json",
        {
            "baseline_status": "config-blocked",
            "corrected_config_recommendation": (
                "for .384 raise trainer.max_epochs above the current checkpoint epoch"
            ),
            "honest_verdict": "complete: baseline config-blocked before pass4",
            "matches_published_087": False,
            "near_faithful_080": False,
            "stable_checkpoint_path": "results/trm_runs/sudoku_extreme_baseline/last.ckpt",
            "val_exact_accuracy": 0.278172343969,
        },
    )
    _write_json(
        root / "results" / "experiment_4139_decisive_verifier_graft_sudoku.json",
        {
            "baseline_val_exact_accuracy": 0.278172343969,
            "false_negative_risk": True,
            "graft_deferred": True,
            "headroom_present": False,
            "honest_verdict": "complete: uninformative_no_headroom_false_negative_risk",
            "verifier_value_added": False,
        },
    )
    _write_json(
        root / "results" / "experiment_4144_capstone_v383.json",
        {
            "baseline_val_trajectory": {
                "status": "baseline_config_blocked",
                "final_val_exact_accuracy": 0.278172343969,
                "matches_published_087": None,
                "near_faithful_080": None,
                "attempted_passes": [
                    {"experiment_id": 4135, "included": False, "status": "skipped_flagged_adversarial"}
                ],
            },
            "headline_outcome": "baseline_config_blocked",
            "honest_verdict": "blocked: capstone_v383_baseline_config_blocked",
            "verifier_value_added_verdict": {
                "artifact_status": "skipped_flagged_adversarial",
                "false_negative_risk": True,
                "reason": "baseline_not_faithful_or_near_faithful",
                "status": "deferred",
                "verifier_value_added": False,
            },
        },
    )
    return root


def run_happy(root: Path) -> dict:
    out = mod.run(root, pretest_result=GREEN, started_s=1000.0, now_s=1000.5)
    return json.loads(out.read_text(encoding="utf-8"))


def test_req_report_4145_spec_declares_contract() -> None:
    """REQ-REPORT-4145: OpenSpec declares required fields and scenarios."""

    spec = Path("openspec/capabilities/research-reporting/spec.md").read_text(encoding="utf-8")
    assert "REQ-REPORT-4145" in spec
    assert "SCENARIO-REPORT-4145" in spec
    assert "SCENARIO-REPORT-4145-BLOCKED-PRECONDITION" in spec
    assert "v383_close_state" in spec
    assert "preconditions_checked" in spec
    assert "6.99" in spec and "0.278" in spec and "FALSE_NEGATIVE_RISK" in spec


def test_shared_helpers_and_archive_record_editing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """REQ-REPORT-4145: helper behavior is deterministic and YAML-safe."""

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
    deduped, removed, action = mod.dedupe_or_update_record(_research_complete_text(duplicates=3))
    assert action == "deduped"
    assert removed == 2
    assert mod.archive_record_count(deduped) == 1
    assert "max_epochs cap" in deduped
    assert mod.yaml_parses(deduped)
    updated, removed2, action2 = mod.dedupe_or_update_record(_research_complete_text(duplicates=1))
    assert (removed2, action2) == (0, "updated")
    assert mod.archive_record_count(updated) == 1
    appended, removed3, action3 = mod.dedupe_or_update_record(
        "# history\nmilestones:\n- id: 2026.06.382\n  finding: prior\n"
    )
    assert (removed3, action3) == (0, "appended")
    assert "activation_recorded: exp4145-archive-v383-activate-v384" in appended
    assert mod.archive_record_count(appended) == 1


def test_precondition_helpers_and_json_readers(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """SCENARIO-REPORT-4145-BLOCKED-PRECONDITION: resource probes are explicit."""

    assert mod._milestone_from_text("name: no milestone\n") == "unknown"
    assert mod.read_active_milestone(tmp_path) == ("unknown", "research-roadmap.yaml")
    (tmp_path / "research-roadmap-next.yaml").write_text("milestone: 2026.06.384\n", encoding="utf-8")
    assert mod.read_active_milestone(tmp_path) == ("2026.06.384", "research-roadmap-next.yaml")
    assert mod.train_file_present(tmp_path) is False
    train = tmp_path / mod.NANO_TRM_TRAIN_REL_PATH
    train.parent.mkdir(parents=True, exist_ok=True)
    train.write_text("# train\n", encoding="utf-8")
    assert mod.train_file_present(tmp_path) is True

    assert mod.read_json_object(tmp_path / "missing.json") == {}
    bad = tmp_path / "bad.json"
    bad.write_text("{bad", encoding="utf-8")
    assert mod.read_json_object(bad) == {}
    not_obj = tmp_path / "list.json"
    not_obj.write_text("[1]", encoding="utf-8")
    assert mod.read_json_object(not_obj) == {}
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


def test_read_sources_and_build_v383_close_state(tmp_path: Path) -> None:
    """SCENARIO-REPORT-4145: close-state records no-op cap, stuck baseline, and false-negative risk."""

    root = make_repo(tmp_path)
    sources = mod.read_v383_sources(root)
    assert sources["4135"]["duration_s"] == 6.99
    cited = mod.build_cited_upstream(root)
    assert cited[-1]["experiment_id"] == "4144"
    assert mod.is_sha256(cited[-1]["sha256"])
    state = mod.build_v383_close_state(sources)
    assert state["accumulation_noop"] is True
    assert state["noop_reason"] == "max_epochs_cap"
    assert state["exp4135_duration_s"] == 6.99
    assert state["exp4135_trained"] is False
    assert state["checkpoint_untouched"] is True
    assert state["baseline_val_exact_accuracy"] == 0.278172343969
    assert state["baseline_val_exact_accuracy_rounded"] == 0.278
    assert state["matches_published_087"] is False
    assert state["graft_deferred"] is True
    assert state["false_negative_risk"] is True
    assert state["v384_forward_plan"]["fixes_max_epochs_cap"] is True

    default_state = mod.build_v383_close_state({})
    assert default_state["exp4135_duration_s"] == 6.99
    assert default_state["baseline_val_exact_accuracy_rounded"] == 0.278


def test_run_happy_path_writes_valid_artifact_and_updates_archive(tmp_path: Path) -> None:
    """SCENARIO-REPORT-4145: complete path writes the terminal archive artifact."""

    root = make_repo(tmp_path, duplicates=2)
    artifact = run_happy(root)
    assert artifact["honest_verdict"].startswith("success:")
    assert artifact["archived_milestone"] == "2026.06.383"
    assert artifact["activated_milestone"] == "2026.06.384"
    assert artifact["active_milestone_confirmed"] == "2026.06.384"
    assert artifact["research_complete_yaml_parses"] is True
    assert artifact["exclusion_manifest_parses"] is True
    assert artifact["nano_trm_train_present"] is True
    assert artifact["pretest_suite_green"] is True
    assert artifact["research_complete_record_action"] == "deduped"
    assert artifact["research_complete_duplicates_removed"] == 1
    assert artifact["v383_close_state"]["accumulation_noop"] is True
    assert artifact["v383_close_state"]["baseline_val_exact_accuracy_rounded"] == 0.278
    assert artifact["v383_close_state"]["false_negative_risk"] is True
    assert artifact["field_principles"]["v383_close_state"] == mod.FIELD_PRINCIPLES["v383_close_state"]
    assert artifact["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert mod.is_sha256(artifact["reproducibility_checksum"])
    complete_text = (root / "research-complete.yaml").read_text(encoding="utf-8")
    assert mod.archive_record_count(complete_text) == 1
    assert "max_epochs cap" in complete_text
    mod.validate_artifact(artifact)


def test_run_real_pretest_branch_and_entrypoint_are_injectable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SCENARIO-REPORT-4145: real pretest and CLI entrypoint can be substituted."""

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

    import carnot.experiment_4145_archive_v383_activate_v384 as entrypoint

    called: dict[str, Path] = {}

    def fake_run(root_path: Path = mod.REPO_ROOT) -> Path:
        called["root"] = Path(root_path)
        return root / mod.OUTPUT_REL_PATH

    monkeypatch.setattr(entrypoint, "run", fake_run)
    assert entrypoint.main() == 0
    assert called["root"] == entrypoint.REPO_ROOT


def test_run_blocked_preconditions_before_research_complete_edit(tmp_path: Path) -> None:
    """SCENARIO-REPORT-4145-BLOCKED-PRECONDITION: blocked paths do not fabricate success."""

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
    (root5 / "research-roadmap.yaml").write_text("milestone: 2026.06.383\n", encoding="utf-8")
    artifact5 = json.loads(mod.run(root5, pretest_result=GREEN).read_text(encoding="utf-8"))
    assert artifact5["honest_verdict"] == "blocked_v384_not_active"

    root6 = make_repo(tmp_path / "red")
    before = (root6 / "research-complete.yaml").read_text(encoding="utf-8")
    artifact6 = json.loads(mod.run(root6, pretest_result=RED).read_text(encoding="utf-8"))
    assert artifact6["honest_verdict"] == "blocked_smart_subset_pretest_not_green"
    assert artifact6["preconditions_checked"]["pretest_suite_green"] is False
    assert (root6 / "research-complete.yaml").read_text(encoding="utf-8") == before


def test_run_blocked_research_complete_edit_guards(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-REPORT-4145: invalid archive edits are blocked before completion."""

    root = make_repo(tmp_path / "invalid")
    monkeypatch.setattr(mod, "dedupe_or_update_record", lambda text: ("a: : :\n- [", 0, "appended"))
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
    """REQ-REPORT-4145: complete and blocked artifact builders keep schema shape."""

    root = make_repo(tmp_path)
    state = mod.build_v383_close_state(mod.read_v383_sources(root))
    complete = mod.build_complete_artifact(
        v383_close_state=state,
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
    """REQ-REPORT-4145: validation rejects artifacts that launder the .383 truth."""

    good = run_happy(make_repo(tmp_path))

    def set_path(obj: dict, path: list[str], value: object) -> None:
        cur = obj
        for key in path[:-1]:
            cur = cur[key]
        cur[path[-1]] = value

    cases = [
        ("missing required fields", lambda a: a.pop("v383_close_state")),
        ("terminal-prefixed", lambda a: a.__setitem__("honest_verdict", "done")),
        ("field_principles must be a mapping", lambda a: a.__setitem__("field_principles", "x")),
        ("missing field principles", lambda a: a.__setitem__("field_principles", {})),
        (
            "principle must match REQ-REPORT-4145",
            lambda a: a["field_principles"].__setitem__("v383_close_state", "wrong"),
        ),
        ("archived milestone", lambda a: a.__setitem__("archived_milestone", "2026.06.999")),
        ("activated milestone", lambda a: a.__setitem__("activated_milestone", "2026.06.999")),
        ("research-complete YAML", lambda a: a.__setitem__("research_complete_yaml_parses", False)),
        ("exclusion manifest", lambda a: a.__setitem__("exclusion_manifest_parses", False)),
        ("nano-trm train", lambda a: a.__setitem__("nano_trm_train_present", False)),
        ("pretest suite", lambda a: a.__setitem__("pretest_suite_green", False)),
        ("active milestone", lambda a: a.__setitem__("active_milestone_confirmed", "2026.06.383")),
        ("v383_close_state must be a mapping", lambda a: a.__setitem__("v383_close_state", "x")),
        ("accumulation_noop", lambda a: set_path(a, ["v383_close_state", "accumulation_noop"], False)),
        ("noop_reason", lambda a: set_path(a, ["v383_close_state", "noop_reason"], "timer")),
        ("exp4135_duration_s", lambda a: set_path(a, ["v383_close_state", "exp4135_duration_s"], 120.0)),
        ("baseline_val_exact_accuracy", lambda a: set_path(a, ["v383_close_state", "baseline_val_exact_accuracy_rounded"], 0.87)),
        ("graft_deferred", lambda a: set_path(a, ["v383_close_state", "graft_deferred"], False)),
        ("false_negative_risk", lambda a: set_path(a, ["v383_close_state", "false_negative_risk"], False)),
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
