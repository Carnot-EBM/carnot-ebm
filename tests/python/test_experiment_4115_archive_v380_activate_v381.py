"""Tests for Exp 4115 .380 archive / .381 activation.

Spec refs: REQ-REPORT-4115, SCENARIO-REPORT-4115,
SCENARIO-REPORT-4115-BLOCKED-PRECONDITION.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from carnot.reporting import archive_v380_activate_v381_4115 as mod


GREEN = mod.CommandResult(command=["pytest"], exit_code=0, stdout="81 passed", stderr="")
RED = mod.CommandResult(command=["pytest"], exit_code=1, stdout="FAILED smart subset", stderr="")


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _research_complete_text(*, duplicates: int = 1) -> str:
    head = "- id: 2026.06.379\n  finding: prior milestone\n"
    block = (
        "- id: 2026.06.380\n"
        "  title: 'nano-trm baseline reproduction milestone'\n"
        "  completed: '2026-06-12'\n"
        "  finding: Existing conductor record.\n"
        "  tasks:\n"
        "  - id: exp4107-nanotrm-mechanism-smoke\n"
        "    result: OK\n"
    )
    return head + block * duplicates


def make_repo(tmp_path: Path, *, duplicates: int = 3) -> Path:
    root = tmp_path
    root.mkdir(parents=True, exist_ok=True)
    (root / "research-complete.yaml").write_text(
        _research_complete_text(duplicates=duplicates), encoding="utf-8"
    )
    (root / "ops").mkdir(parents=True, exist_ok=True)
    (root / "ops" / "exclusion_manifest.yaml").write_text(
        "retired:\n- experiment_id: 2091\n  reason: archived\n", encoding="utf-8"
    )
    (root / "research-roadmap.yaml").write_text('milestone: "2026.06.381"\n', encoding="utf-8")
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
        root / "results" / "experiment_4107_nanotrm_mechanism_smoke.json",
        {
            "honest_verdict": "complete: nanotrm_trainer_checkpoint_ok_exact_accuracy_1.0000",
            "nanotrm_trainer_checkpoint_ok": True,
            "exact_accuracy": 1.0,
            "checkpoint_path": "results/trm_runs/exp4107/checkpoints/last.ckpt",
            "duration_s": 741.76,
        },
    )
    _write_json(
        root / "results" / "experiment_4108_nanotrm_sudoku_extreme_baseline.json",
        {
            "honest_verdict": "complete: interrupted_return_code_1_reproduced_0.0232",
            "matches_published_087": False,
            "reproduced_exact_accuracy": 0.02317708358168602,
            "published_exact_accuracy_target": 0.87,
            "checkpoint_reload_ok": True,
            "checkpoint_path": "results/trm_runs/exp4108/checkpoints/last.ckpt",
            "return_code": 1,
            "duration_s": 3405.112,
        },
    )
    _write_json(
        root / "results" / "experiment_4114_capstone_v380.json",
        {
            "honest_verdict": "complete: capstone_v380_honest_null",
            "headline_answers": {
                "nanotrm_trainer_mechanism_derisked": True,
                "published_087_baseline_reproduced": False,
            },
            "published_baseline_reproduction": {
                "status": "baseline_not_reproduced",
                "reproduced_exact_accuracy": 0.02317708358168602,
                "published_087_baseline_reproduced": False,
            },
        },
    )
    return root


def run_happy(root: Path) -> dict:
    out = mod.run(root, pretest_result=GREEN, started_s=1000.0, now_s=1000.5)
    return json.loads(out.read_text(encoding="utf-8"))


def test_req_report_4115_spec_declares_contract() -> None:
    """REQ-REPORT-4115: OpenSpec declares required fields and scenarios."""

    spec = Path("openspec/capabilities/research-reporting/spec.md").read_text(encoding="utf-8")
    assert "REQ-REPORT-4115" in spec
    assert "SCENARIO-REPORT-4115" in spec
    assert "SCENARIO-REPORT-4115-BLOCKED-PRECONDITION" in spec
    assert "v380_close_state" in spec
    assert "preconditions_checked" in spec


def test_yaml_duration_checksum_and_write_helpers(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
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


def test_milestone_train_file_and_record_helpers(tmp_path: Path) -> None:
    assert mod._milestone_from_text("milestone: '2026.06.381'\n") == "2026.06.381"
    assert mod._milestone_from_text("name: no milestone\n") == "unknown"
    assert mod.read_active_milestone(tmp_path) == ("unknown", "research-roadmap.yaml")
    (tmp_path / "research-roadmap-next.yaml").write_text("milestone: 2026.06.381\n", encoding="utf-8")
    assert mod.read_active_milestone(tmp_path) == ("2026.06.381", "research-roadmap-next.yaml")
    assert mod.train_file_present(tmp_path) is False
    train = tmp_path / mod.NANO_TRM_TRAIN_REL_PATH
    train.parent.mkdir(parents=True, exist_ok=True)
    train.write_text("# train\n", encoding="utf-8")
    assert mod.train_file_present(tmp_path) is True
    assert mod._record_id("- id: 2026.06.380") == "2026.06.380"
    assert mod._record_id("  - id: nested") is None


def test_dedupe_or_append_record_paths() -> None:
    deduped, removed, action = mod.dedupe_or_append_record(_research_complete_text(duplicates=3))
    assert action == "deduped"
    assert removed == 2
    assert deduped.count("- id: 2026.06.380") == 1
    assert mod.yaml_parses(deduped)

    unchanged, removed2, action2 = mod.dedupe_or_append_record(_research_complete_text(duplicates=1))
    assert action2 == "unchanged"
    assert removed2 == 0
    assert unchanged.count("- id: 2026.06.380") == 1

    appended, removed3, action3 = mod.dedupe_or_append_record("- id: 2026.06.379\n  finding: prior\n")
    assert action3 == "appended"
    assert removed3 == 0
    assert "- id: 2026.06.380" in appended
    assert "activation_recorded: exp4115-archive-v380-activate-v381" in appended
    assert mod.yaml_parses(appended)


def test_read_json_and_source_provenance(tmp_path: Path) -> None:
    assert mod.read_json_object(tmp_path / "missing.json") == {}
    bad = tmp_path / "bad.json"
    bad.write_text("{bad", encoding="utf-8")
    assert mod.read_json_object(bad) == {}
    not_obj = tmp_path / "list.json"
    not_obj.write_text("[1, 2]", encoding="utf-8")
    assert mod.read_json_object(not_obj) == {}
    good = tmp_path / "good.json"
    good.write_text(json.dumps({"x": 1}), encoding="utf-8")
    assert mod.read_json_object(good) == {"x": 1}
    assert mod.is_sha256(mod.file_sha256(good))
    assert mod.file_sha256(tmp_path / "nope") is None

    root = make_repo(tmp_path / "repo")
    sources = mod.read_v380_sources(root)
    assert sources["4107"]["exact_accuracy"] == 1.0
    cited = mod.build_cited_upstream(root)
    assert len(cited) == len(mod.V380_SOURCE_ARTIFACTS)
    assert all(item["sha256"] is not None for item in cited)


def test_build_v380_close_state_records_mechanism_and_failed_baseline(tmp_path: Path) -> None:
    root = make_repo(tmp_path)
    state = mod.build_v380_close_state(mod.read_v380_sources(root))
    assert state["mechanism_proven"] is True
    assert state["exp4107"]["nanotrm_trainer_checkpoint_ok"] is True
    assert state["exp4107"]["exact_accuracy"] == 1.0
    assert state["baseline_reproduced"] is False
    assert state["exp4108"]["matches_published_087"] is False
    assert state["exp4108"]["reproduced_exact_accuracy_rounded"] == 0.0232
    assert state["exp4108"]["interrupted_by_80_min_cap"] is True
    assert state["v381_forward_fix"]["stable_checkpoint_lineage_required"] is True

    default_state = mod.build_v380_close_state({})
    assert default_state["exp4108"]["reproduced_exact_accuracy_rounded"] == 0.0232
    assert default_state["baseline_reproduced"] is False


def test_smart_subset_helpers_and_run_command(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = make_repo(tmp_path)
    targets = mod.smart_subset_targets(root)
    assert "tests/python/test_pipeline_extract.py" in targets
    cmd = mod.smart_subset_command(targets)
    assert cmd[0] == str(mod.PYTEST_BIN)
    assert "--no-cov" in cmd
    assert mod.smart_subset_targets(tmp_path / "empty") == [mod.CORE_SMART_SUBSET[0]]
    assert mod._run_command(["true"], tmp_path).exit_code == 0
    assert mod._run_command(["definitely-not-a-real-binary-xyz"], tmp_path).exit_code == 127
    monkeypatch.setattr(mod, "_run_command", lambda command, root_path: GREEN)
    assert mod.run_smart_subset(root).exit_code == 0


def test_run_happy_path_writes_valid_artifact_and_dedupes(tmp_path: Path) -> None:
    root = make_repo(tmp_path)
    artifact = run_happy(root)
    assert artifact["honest_verdict"].startswith("success:")
    assert artifact["archived_milestone"] == "2026.06.380"
    assert artifact["activated_milestone"] == "2026.06.381"
    assert artifact["active_milestone_confirmed"] == "2026.06.381"
    assert artifact["research_complete_yaml_parses"] is True
    assert artifact["exclusion_manifest_parses"] is True
    assert artifact["nano_trm_train_present"] is True
    assert artifact["pretest_suite_green"] is True
    assert artifact["research_complete_record_action"] == "deduped"
    assert artifact["research_complete_duplicates_removed"] == 2
    assert artifact["v380_close_state"]["mechanism_proven"] is True
    assert artifact["v380_close_state"]["baseline_reproduced"] is False
    assert artifact["v380_close_state"]["exp4108"]["reproduced_exact_accuracy_rounded"] == 0.0232
    assert artifact["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert mod.is_sha256(artifact["reproducibility_checksum"])
    assert (root / "research-complete.yaml").read_text(encoding="utf-8").count("- id: 2026.06.380") == 1
    mod.validate_artifact(artifact)


def test_run_real_pretest_branch_is_injectable(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = make_repo(tmp_path)
    monkeypatch.setattr(mod, "run_smart_subset", lambda root_path: GREEN)
    artifact = json.loads(mod.run(root, started_s=1.0, now_s=1.1).read_text(encoding="utf-8"))
    assert artifact["preconditions_checked"]["pretest_suite_green"] is True


def test_run_blocked_preconditions_before_research_complete_edit(tmp_path: Path) -> None:
    missing = mod.run(tmp_path, pretest_result=GREEN)
    assert json.loads(missing.read_text(encoding="utf-8"))["honest_verdict"] == "blocked_research_complete_yaml_missing"

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
    (root5 / "research-roadmap.yaml").write_text("milestone: 2026.06.380\n", encoding="utf-8")
    artifact5 = json.loads(mod.run(root5, pretest_result=GREEN).read_text(encoding="utf-8"))
    assert artifact5["honest_verdict"] == "blocked_v381_not_active"

    root6 = make_repo(tmp_path / "red")
    artifact6 = json.loads(mod.run(root6, pretest_result=RED).read_text(encoding="utf-8"))
    assert artifact6["honest_verdict"] == "blocked_smart_subset_pretest_not_green"
    assert artifact6["preconditions_checked"]["pretest_suite_green"] is False
    assert (root6 / "research-complete.yaml").read_text(encoding="utf-8").count("- id: 2026.06.380") == 3


def test_run_blocked_research_complete_edit_guards(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = make_repo(tmp_path / "invalid")
    monkeypatch.setattr(mod, "dedupe_or_append_record", lambda text: ("a: : :\n- [", 0, "appended"))
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
    root = make_repo(tmp_path)
    state = mod.build_v380_close_state(mod.read_v380_sources(root))
    complete = mod.build_complete_artifact(
        v380_close_state=state,
        preconditions_checked={"ok": True},
        duration_s=0.5,
        active_roadmap_path="research-roadmap.yaml",
        research_complete_record_action="unchanged",
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
    good = run_happy(make_repo(tmp_path))

    def set_path(obj: dict, path: list[str], value: object) -> None:
        cur = obj
        for key in path[:-1]:
            cur = cur[key]
        cur[path[-1]] = value

    cases = [
        ("missing required fields", lambda a: a.pop("v380_close_state")),
        ("terminal-prefixed", lambda a: a.__setitem__("honest_verdict", "done")),
        ("field_principles must be a mapping", lambda a: a.__setitem__("field_principles", "x")),
        ("missing field principles", lambda a: a.__setitem__("field_principles", {})),
        ("archived milestone", lambda a: a.__setitem__("archived_milestone", "2026.06.999")),
        ("activated milestone", lambda a: a.__setitem__("activated_milestone", "2026.06.999")),
        ("research-complete YAML", lambda a: a.__setitem__("research_complete_yaml_parses", False)),
        ("exclusion manifest", lambda a: a.__setitem__("exclusion_manifest_parses", False)),
        ("nano-trm train", lambda a: a.__setitem__("nano_trm_train_present", False)),
        ("pretest suite", lambda a: a.__setitem__("pretest_suite_green", False)),
        ("active milestone", lambda a: a.__setitem__("active_milestone_confirmed", "2026.06.380")),
        ("v380_close_state must be a mapping", lambda a: a.__setitem__("v380_close_state", "x")),
        ("mechanism_proven", lambda a: set_path(a, ["v380_close_state", "mechanism_proven"], False)),
        ("must include exp4107", lambda a: set_path(a, ["v380_close_state", "exp4107"], "x")),
        (
            "nanotrm_trainer_checkpoint_ok",
            lambda a: set_path(a, ["v380_close_state", "exp4107", "nanotrm_trainer_checkpoint_ok"], False),
        ),
        ("exact_accuracy", lambda a: set_path(a, ["v380_close_state", "exp4107", "exact_accuracy"], 0.5)),
        ("baseline_reproduced", lambda a: set_path(a, ["v380_close_state", "baseline_reproduced"], True)),
        ("must include exp4108", lambda a: set_path(a, ["v380_close_state", "exp4108"], "x")),
        (
            "matches_published_087",
            lambda a: set_path(a, ["v380_close_state", "exp4108", "matches_published_087"], True),
        ),
        (
            "reproduced_exact_accuracy",
            lambda a: set_path(a, ["v380_close_state", "exp4108", "reproduced_exact_accuracy_rounded"], 0.87),
        ),
        (
            "interrupted_by_80_min_cap",
            lambda a: set_path(a, ["v380_close_state", "exp4108", "interrupted_by_80_min_cap"], False),
        ),
        (
            "stable checkpoint",
            lambda a: set_path(
                a, ["v380_close_state", "v381_forward_fix", "stable_checkpoint_lineage_required"], False
            ),
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
