"""Tests for Exp 4313 `.398` archive / `.399` activation.

Spec refs: REQ-REPORT-4313, SCENARIO-REPORT-4313,
SCENARIO-REPORT-4313-BLOCKED-PRECONDITION.
"""

from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path

import pytest

from carnot.reporting import archive_v398_activate_v399_4313 as mod


GREEN = mod.CommandResult(command=["pytest"], exit_code=0, stdout="81 passed", stderr="")
RED = mod.CommandResult(command=["pytest"], exit_code=1, stdout="FAILED smart subset", stderr="")


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _research_complete_text(*, duplicates: int = 1) -> str:
    head = (
        "# Carnot Research - Completed Experiments\n"
        "milestones:\n"
        "- id: 2026.06.397\n"
        "  finding: prior milestone\n"
    )
    block = (
        "- id: 2026.06.398\n"
        "  title: old conductor row\n"
        "  completed: '2026-06-17'\n"
        "  finding: See conductor log for per-experiment results.\n"
        "  tasks:\n"
        "  - id: exp4312-capstone-v398\n"
        "    result: OK (conductor)\n"
    )
    return head + block * duplicates


def _capstone(**overrides: object) -> dict:
    payload = {
        "honest_verdict": (
            "complete: v398_efficiency_hardened_in_generation_open_cross_domain_open_"
            "self_learning_helps_arc_excluded"
        ),
        "headline_outcome": (
            "efficiency_hardened__in_generation_open__cross_domain_open__"
            "self_learning_helps__arc_excluded__paper_ready"
        ),
        "verifier_thesis_state": "efficiency_parity_hardened",
        "efficiency_pareto_hardened": True,
        "in_generation_moat_holds": False,
        "cross_domain_moat_holds": False,
        "paper_ready": True,
        "verifier_is_oracle_honored": True,
        "flagged_artifacts_excluded": [{"experiment_id": 4307, "reason": "flagged_adversarial"}],
    }
    payload.update(overrides)
    return payload


def _efficiency(**overrides: object) -> dict:
    payload = {
        "honest_verdict": "complete: hardened_pareto_win_delta_0.3000",
        "efficiency_pareto_holds": True,
        "accuracy_energy_verifier": 0.8,
        "accuracy_best_judge": 0.5,
        "accuracy_delta_ci95": [0.1, 0.5],
        "cost_ratio": 1.03e-08,
        "verifier_is_oracle": False,
    }
    payload.update(overrides)
    return payload


def _in_generation(**overrides: object) -> dict:
    payload = {
        "honest_verdict": "complete: diffusiongemma_guidance_bounded_null_vs_engaged_control",
        "diffusiongemma_guidance_moat": False,
        "carnot_minus_best_control_delta": 0.133334,
        "guidance_moat_ci95": [-0.066667, 0.366667],
        "controls_differentiated": True,
        "scorer_leak_recheck_passed": True,
        "verifier_is_oracle": False,
    }
    payload.update(overrides)
    return payload


def _cross_domain(**overrides: object) -> dict:
    payload = {
        "honest_verdict": "complete: cross_domain_selection_collapses_domain_bound",
        "cross_domain_selection_holds": False,
        "cross_domain_delta": 0.2307692308,
        "cross_domain_ci95": [-0.1153846154, 0.5384615385],
        "label_ablation_robust": True,
        "held_out_task_n": 26,
        "verifier_is_oracle": False,
        "missing_verifier_gaps": [{"gap_id": "GAP-CROSS-DOMAIN-FAMILY-INVARIANT-SELECTION-4305"}],
    }
    payload.update(overrides)
    return payload


def _self_learning(**overrides: object) -> dict:
    payload = {
        "honest_verdict": "complete: powered_cross_domain_online_adaptation_helps",
        "online_adaptation_helps": True,
        "best_adaptive_minus_static_delta": 0.5292929293,
        "best_adaptive_minus_static_ci95": [0.4080808081, 0.6505050505],
        "held_out_task_n": 102,
        "verifier_is_oracle": False,
    }
    payload.update(overrides)
    return payload


def _arc(**overrides: object) -> dict:
    payload = {
        "honest_verdict": (
            "complete: incremental_progress_no_advance_re86-8af5384d_L1_selected_"
            "frontier_adapter_unavailable"
        ),
        "flagged_adversarial": True,
        "corrigendum_pending": [
            {
                "kind": "GATE_PASSED_WITHOUT_DATA",
                "severity": "critical",
                "detail": "acceptance_gate_passed=true but exploration_actions_used=0",
            }
        ],
        "exploration_actions_used": 0,
        "total_levels_solved": 22,
        "total_levels": 22,
        "real_env_confirmed": False,
        "phase_trace": [
            {
                "phase": "hardened-set-encoder-route",
                "reason": "no_verified_re86_frontier_adapter_available",
                "retained": False,
            }
        ],
    }
    payload.update(overrides)
    return payload


def make_repo(tmp_path: Path, *, duplicates: int = 1) -> Path:
    root = tmp_path
    root.mkdir(parents=True, exist_ok=True)
    (root / "research-complete.yaml").write_text(
        _research_complete_text(duplicates=duplicates), encoding="utf-8"
    )
    (root / "ops").mkdir(parents=True, exist_ok=True)
    (root / "ops" / "exclusion_manifest.yaml").write_text(
        "retired:\n- experiment_id: 4307\n  reason: flagged_adversarial\n", encoding="utf-8"
    )
    (root / "research-roadmap.yaml").write_text(
        "milestone: 2026.06.399\n"
        "milestone_doc: openspec/change-proposals/research-roadmap-v399.md\n"
        "milestone_overview: close-the-two-open-moats + deploy-efficiency-cascade + unstall-ARC\n",
        encoding="utf-8",
    )
    _write_json(root / "results" / "experiment_4312_capstone_v398.json", _capstone())
    _write_json(
        root / "results" / "experiment_4303_verifier_efficiency_parity_isoflops.json",
        _efficiency(),
    )
    _write_json(
        root / "results" / "experiment_4304_diffusiongemma_in_generation_engaged_controls.json",
        _in_generation(),
    )
    _write_json(
        root / "results" / "experiment_4305_cross_domain_selector_generalization.json",
        _cross_domain(),
    )
    _write_json(
        root / "results" / "experiment_4306_self_learning_powered_ci_cross_domain.json",
        _self_learning(),
    )
    _write_json(
        root / "results" / "experiment_4307_arc_incremental_progress_new_game.json",
        _arc(),
    )
    return root


def run_happy(root: Path) -> dict:
    out = mod.run(root, pretest_result=GREEN, started_s=1000.0, now_s=1000.5)
    return json.loads(out.read_text(encoding="utf-8"))


def test_req_report_4313_spec_declares_contract() -> None:
    """REQ-REPORT-4313: OpenSpec declares the true `.398` scorecard contract."""

    spec = Path("openspec/capabilities/research-reporting/spec.md").read_text(encoding="utf-8")
    assert "REQ-REPORT-4313" in spec
    assert "SCENARIO-REPORT-4313" in spec
    assert "SCENARIO-REPORT-4313-BLOCKED-PRECONDITION" in spec
    assert "efficiency-parity HARDENED" in spec
    assert "self-learning HELPS" in spec
    assert "OPEN/UNDERPOWERED-POSITIVE" in spec
    assert "ARC stalled on a harness failure" in spec
    assert mod.FIELD_PRINCIPLES["honest_verdict"] in spec
    assert mod.FIELD_PRINCIPLES["v398_close_state"] in spec
    assert mod.FIELD_PRINCIPLES["preconditions_checked"] in spec


def test_helpers_and_archive_record_editing(tmp_path: Path) -> None:
    """REQ-REPORT-4313: helper behavior is deterministic and YAML-safe."""

    assert mod.archive_record_count(_research_complete_text(duplicates=3)) == 3
    assert mod._record_id("- id: '2026.06.398'") == "2026.06.398"
    assert mod._record_id("  - id: nested") is None
    assert mod._yaml_quote("self-learning isn't redo") == "'self-learning isn''t redo'"
    assert mod._rounded_pair("bad", [1.0, 2.0]) == [1.0, 2.0]
    assert mod._ci_includes_zero([-0.1, 0.2])
    assert mod._ci_excludes_zero([0.1, 0.2])
    assert mod._insert_before_tasks(["  title: no tasks"], "  finding: x") == [
        "  title: no tasks",
        "  finding: x",
    ]

    close_state = mod.build_v398_close_state(
        {
            "4312": _capstone(),
            "4303": _efficiency(),
            "4304": _in_generation(),
            "4305": _cross_domain(),
            "4306": _self_learning(),
            "4307": _arc(),
        }
    )
    deduped, removed, action = mod.dedupe_or_update_record(
        _research_complete_text(duplicates=3), close_state
    )
    assert (removed, action) == (2, "deduped")
    assert mod.archive_record_count(deduped) == 1
    assert "efficiency-parity HARDENED" in deduped
    assert "UNDERPOWERED-POSITIVE" in deduped
    assert mod.yaml_parses(deduped)

    updated, removed2, action2 = mod.dedupe_or_update_record(
        _research_complete_text(duplicates=1), close_state
    )
    assert (removed2, action2) == (0, "updated")
    unchanged, removed3, action3 = mod.dedupe_or_update_record(updated, close_state)
    assert (unchanged, removed3, action3) == (updated, 0, "unchanged")
    appended, removed4, action4 = mod.dedupe_or_update_record(
        "# history\nmilestones:\n- id: 2026.06.397\n  finding: prior\n", close_state
    )
    assert (removed4, action4) == (0, "appended")
    assert "activation_recorded: exp4313-archive-v398-activate-v399" in appended
    no_finding, _removed5, action5 = mod.dedupe_or_update_record(
        "milestones:\n- id: 2026.06.398\n  title: no finding\n  tasks:\n  - id: exp4312\n",
        close_state,
    )
    assert action5 == "updated"
    assert "  finding: '.398 close-state" in no_finding


def test_read_sources_and_build_v398_close_state(tmp_path: Path) -> None:
    """SCENARIO-REPORT-4313: close-state records the true .398 scorecard."""

    root = make_repo(tmp_path)
    sources = mod.read_v398_sources(root)
    cited = mod.build_cited_upstream(root)
    assert sources["4312"]["verifier_thesis_state"] == "efficiency_parity_hardened"
    assert {item["experiment_id"] for item in cited if item["kind"] == "artifact"} == {
        "4312",
        "4303",
        "4304",
        "4305",
        "4306",
        "4307",
    }
    assert any(
        item["deliverable"] == "openspec/change-proposals/research-roadmap-v399.md"
        and item["required"] is False
        and item["sha256"] is None
        for item in cited
    )
    assert all(item["sha256"] is None or mod.is_sha256(item["sha256"]) for item in cited)

    state = mod.build_v398_close_state(sources)
    assert (
        state["summary"]
        == "efficiency_hardened_self_learning_helps_two_open_underpowered_moats_arc_harness_stall"
    )
    assert state["verifier_thesis_state"] == "efficiency_parity_hardened"
    assert state["efficiency_axis_state"] == "HARDENED"
    assert state["efficiency_pareto_hardened"] is True
    assert state["efficiency_accuracy_energy_verifier"] == 0.8
    assert state["efficiency_accuracy_best_judge"] == 0.5
    assert state["efficiency_accuracy_delta_ci95"] == [0.1, 0.5]
    assert state["efficiency_ci95_excludes_zero"] is True
    assert state["efficiency_verifier_is_oracle"] is False
    assert state["self_learning_axis_state"] == "HELPS"
    assert state["self_learning_helps"] is True
    assert state["self_learning_delta"] == 0.529
    assert state["self_learning_ci95"] == [0.408, 0.651]
    assert state["self_learning_ci95_excludes_zero"] is True
    assert state["in_generation_axis_state"] == "OPEN_UNDERPOWERED_POSITIVE"
    assert state["in_generation_moat_holds"] is False
    assert state["in_generation_delta"] == 0.133
    assert state["in_generation_ci95"] == [-0.067, 0.367]
    assert state["in_generation_ci95_includes_zero"] is True
    assert state["in_generation_controls_differentiated"] is True
    assert state["in_generation_scorer_leak_recheck_passed"] is True
    assert state["cross_domain_axis_state"] == "OPEN_UNDERPOWERED_POSITIVE"
    assert state["cross_domain_moat_holds"] is False
    assert state["cross_domain_delta"] == 0.231
    assert state["cross_domain_ci95_includes_zero"] is True
    assert state["cross_domain_label_ablation_robust"] is True
    assert state["cross_domain_gap_id"] == "GAP-CROSS-DOMAIN-FAMILY-INVARIANT-SELECTION-4305"
    assert state["arc_axis_state"] == "STALLED_ON_HARNESS_FAILURE"
    assert state["arc_total_levels_solved"] == 22
    assert state["arc_harness_failure_kind"] == "GATE_PASSED_WITHOUT_DATA"
    assert state["arc_frontier_adapter_available"] is False
    assert state["arc_science_failure"] is False
    assert state["paper_ready"] is True
    assert state["v399_frame"] == mod.V399_FRAME


def test_run_happy_path_writes_valid_artifact_and_updates_archive(tmp_path: Path) -> None:
    """SCENARIO-REPORT-4313: complete path writes the terminal archive artifact."""

    root = make_repo(tmp_path, duplicates=2)
    artifact = run_happy(root)
    assert artifact["honest_verdict"].startswith("success:")
    assert artifact["archived_milestone"] == "2026.06.398"
    assert artifact["activated_milestone"] == "2026.06.399"
    assert artifact["active_milestone_confirmed"] == "2026.06.399"
    assert artifact["research_complete_yaml_parses"] is True
    assert artifact["exclusion_manifest_parses"] is True
    assert artifact["pretest_suite_green"] is True
    assert artifact["research_complete_record_action"] == "deduped"
    assert artifact["research_complete_duplicates_removed"] == 1
    assert artifact["v398_close_state"]["efficiency_pareto_hardened"] is True
    assert artifact["v398_close_state"]["self_learning_helps"] is True
    assert artifact["v398_close_state"]["in_generation_moat_holds"] is False
    assert artifact["v398_close_state"]["cross_domain_moat_holds"] is False
    assert artifact["v398_close_state"]["arc_axis_state"] == "STALLED_ON_HARNESS_FAILURE"
    assert (
        artifact["field_principles"]["v398_close_state"] == mod.FIELD_PRINCIPLES["v398_close_state"]
    )
    assert artifact["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert mod.is_sha256(artifact["reproducibility_checksum"])
    complete_text = (root / "research-complete.yaml").read_text(encoding="utf-8")
    assert mod.archive_record_count(complete_text) == 1
    assert "efficiency-parity HARDENED" in complete_text
    assert "self-learning HELPS" in complete_text
    assert "ARC stalled-on-harness-failure" in complete_text
    mod.validate_artifact(artifact)


def test_run_blocked_preconditions_before_research_complete_edit(tmp_path: Path) -> None:
    """SCENARIO-REPORT-4313-BLOCKED-PRECONDITION: blocked paths do not fabricate."""

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

    root_manifest = make_repo(tmp_path / "manifest_poison")
    (root_manifest / "ops" / "exclusion_manifest.yaml").write_text(
        "a: : :\n- [\n", encoding="utf-8"
    )
    artifact_manifest = json.loads(
        mod.run(root_manifest, pretest_result=GREEN).read_text(encoding="utf-8")
    )
    assert artifact_manifest["honest_verdict"] == "blocked_exclusion_manifest_yaml_poison"

    root3 = make_repo(tmp_path / "red")
    before = (root3 / "research-complete.yaml").read_text(encoding="utf-8")
    artifact3 = json.loads(mod.run(root3, pretest_result=RED).read_text(encoding="utf-8"))
    assert artifact3["honest_verdict"] == "blocked_smart_subset_pretest_not_green"
    assert artifact3["preconditions_checked"]["smart_subset_pretest"]["green"] is False
    assert (root3 / "research-complete.yaml").read_text(encoding="utf-8") == before

    root4 = make_repo(tmp_path / "wrong_milestone")
    (root4 / "research-roadmap.yaml").write_text("milestone: 2026.06.398\n", encoding="utf-8")
    artifact4 = json.loads(mod.run(root4, pretest_result=GREEN).read_text(encoding="utf-8"))
    assert artifact4["honest_verdict"] == "blocked_v399_not_active"

    root5 = make_repo(tmp_path / "source_missing")
    (
        root5 / "results" / "experiment_4304_diffusiongemma_in_generation_engaged_controls.json"
    ).unlink()
    artifact5 = json.loads(mod.run(root5, pretest_result=GREEN).read_text(encoding="utf-8"))
    assert artifact5["honest_verdict"] == "blocked_in_generation_artifact_missing"


def test_run_blocked_research_complete_edit_guards(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-REPORT-4313: invalid archive edits are blocked before completion."""

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


def test_build_artifact_validation_and_entrypoints(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-REPORT-4313: schema validation and entrypoints preserve the contract."""

    root = make_repo(tmp_path)
    state = mod.build_v398_close_state(mod.read_v398_sources(root))
    complete = mod.build_complete_artifact(
        v398_close_state=state,
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

    called_mod: dict[str, Path] = {}

    def fake_mod_run(root_path: Path = mod.REPO_ROOT) -> Path:
        called_mod["root"] = Path(root_path)
        return root / mod.OUTPUT_REL_PATH

    monkeypatch.setattr(mod, "run", fake_mod_run)
    assert mod.main() == 0
    assert called_mod["root"] == mod.REPO_ROOT

    import carnot.experiment_4313_archive_v398_activate_v399 as entrypoint

    called: dict[str, Path] = {}

    def fake_run(root_path: Path = mod.REPO_ROOT) -> Path:
        called["root"] = Path(root_path)
        return root / mod.OUTPUT_REL_PATH

    monkeypatch.setattr(entrypoint, "run", fake_run)
    assert entrypoint.main() == 0
    assert called["root"] == entrypoint.REPO_ROOT

    script_path = Path("results/experiment_4313_archive_v398_activate_v399.py")
    spec = importlib.util.spec_from_file_location("exp4313_archive_script", script_path)
    assert spec and spec.loader
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)
    monkeypatch.setattr(script, "run", fake_run)
    assert script.main() == 0


def test_validate_artifact_rejection_guards(tmp_path: Path) -> None:
    """REQ-REPORT-4313: validation rejects artifacts that launder the `.398` truth."""

    good = run_happy(make_repo(tmp_path))

    def set_path(obj: dict, path: list[str], value: object) -> None:
        cur = obj
        for key in path[:-1]:
            cur = cur[key]
        cur[path[-1]] = value

    cases = [
        ("missing required fields", lambda a: a.pop("v398_close_state")),
        ("terminal-prefixed", lambda a: a.__setitem__("honest_verdict", "done")),
        (
            "principle must match REQ-REPORT-4313",
            lambda a: a["field_principles"].__setitem__("v398_close_state", "wrong"),
        ),
        ("archived milestone", lambda a: a.__setitem__("archived_milestone", "2026.06.999")),
        ("activated milestone", lambda a: a.__setitem__("activated_milestone", "2026.06.999")),
        ("research-complete YAML", lambda a: a.__setitem__("research_complete_yaml_parses", False)),
        ("exclusion manifest", lambda a: a.__setitem__("exclusion_manifest_parses", False)),
        ("pretest suite", lambda a: a.__setitem__("pretest_suite_green", False)),
        ("active milestone", lambda a: a.__setitem__("active_milestone_confirmed", "2026.06.398")),
        ("v398_close_state must be a mapping", lambda a: a.__setitem__("v398_close_state", "x")),
        (
            "efficiency hardened",
            lambda a: set_path(a, ["v398_close_state", "efficiency_pareto_hardened"], False),
        ),
        (
            "self-learning helps",
            lambda a: set_path(a, ["v398_close_state", "self_learning_helps"], False),
        ),
        (
            "in-generation open",
            lambda a: set_path(a, ["v398_close_state", "in_generation_moat_holds"], True),
        ),
        (
            "in-generation underpowered",
            lambda a: set_path(a, ["v398_close_state", "in_generation_ci95_includes_zero"], False),
        ),
        (
            "cross-domain open",
            lambda a: set_path(a, ["v398_close_state", "cross_domain_moat_holds"], True),
        ),
        (
            "cross-domain underpowered",
            lambda a: set_path(a, ["v398_close_state", "cross_domain_ci95_includes_zero"], False),
        ),
        (
            "ARC harness stall",
            lambda a: set_path(a, ["v398_close_state", "arc_axis_state"], "SCIENCE_FAILURE"),
        ),
        ("paper", lambda a: set_path(a, ["v398_close_state", "paper_ready"], False)),
        ("v399 frame", lambda a: set_path(a, ["v398_close_state", "v399_frame"], "redo")),
    ]
    for label, mutate in cases:
        artifact = copy.deepcopy(good)
        mutate(artifact)
        with pytest.raises(ValueError, match=label):
            mod.validate_artifact(artifact)
