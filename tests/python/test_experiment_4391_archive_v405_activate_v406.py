"""Tests for Exp 4391 `.405` archive / `.406` activation.

Spec refs: REQ-REPORT-4391, SCENARIO-REPORT-4391,
SCENARIO-REPORT-4391-BLOCKED-PRECONDITION.
"""

from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
import sys

import pytest

from carnot.reporting import archive_v405_activate_v406_4391 as mod


GREEN = mod.CommandResult(command=["pytest"], exit_code=0, stdout="81 passed", stderr="")
RED = mod.CommandResult(command=["pytest"], exit_code=1, stdout="FAILED smart subset", stderr="")


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _research_complete_text(*, duplicates: int = 1) -> str:
    head = (
        "# Carnot Research - Completed Experiments\n"
        "milestones:\n"
        "- id: 2026.06.404\n"
        "  finding: prior milestone\n"
    )
    block = (
        "- id: 2026.06.405\n"
        "  title: old conductor row\n"
        "  completed: '2026-06-18'\n"
        "  finding: See conductor log for per-experiment results.\n"
        "  tasks:\n"
        "  - id: exp4390-capstone-v405\n"
        "    result: OK (conductor)\n"
    )
    return head + block * duplicates


def _manifest_text() -> str:
    return (
        "retired_extras:\n"
        "- id: cross_domain_selection_retired_exp4314_v399\n"
        "  experiment_ids: [exp4314]\n"
        "  operator_reopen_required: true\n"
        "  retire_if_same_verdict: true\n"
        "- id: cross_game_value_transfer_retired_exp4342_v401\n"
        "  experiment_ids: [exp4342]\n"
        "  operator_reopen_required: true\n"
        "  retire_if_same_verdict: true\n"
    )


def _registry_text() -> str:
    return (
        "schema_version: 1\n"
        "updated: '2026-06-18'\n"
        "reproducible_total_levels: 34\n"
        "reproducible_total_games: 17\n"
    )


def _capstone(**overrides: object) -> dict:
    payload = {
        "honest_verdict": (
            "complete: v405_detector_detects_but_not_actionable_compounds_true_"
            "generalizes_true_arc_levels_34_publication_ready"
        ),
        "detector_actionable_state": "detects_but_not_actionable",
        "detector_compounds": True,
        "detector_generalizes_cross_domain": True,
        "reproducible_total_levels": 34,
        "paper_ready": True,
        "verifier_is_oracle": False,
        "verifier_thesis_state": (
            "detector_detects_but_not_actionable_detector_compounds_detection_generalizes"
        ),
        "detector_actionability": {
            "status": "detects_but_not_actionable",
            "localization": {
                "detector_localization_actionable": False,
                "honest_verdict": "complete: clean_powered_null_bidirectional_not_actionable",
                "localization_delta_ci95": [0.0, 0.0],
                "localization_f1_by_direction": {
                    "bidirectional_fusion": {
                        "accuracy": 0.096491,
                        "exact_match_count": 11,
                        "f1": 0.096491,
                        "n_error_traces": 114,
                    },
                    "causal_online": {
                        "accuracy": 0.096491,
                        "exact_match_count": 11,
                        "f1": 0.096491,
                        "n_error_traces": 114,
                    },
                    "unidirectional_l2r": {
                        "accuracy": 0.096491,
                        "exact_match_count": 11,
                        "f1": 0.096491,
                        "n_error_traces": 114,
                    },
                },
                "n_error_traces": 114,
                "n_traces": 6548,
                "status": "not_actionable",
                "verifier_is_oracle": False,
            },
        },
        "self_learning": {
            "compounding_delta_ci95": [0.003396, 0.032772],
            "detector_compounds": True,
            "learning_curve": [
                {
                    "held_out_auroc": 0.986296,
                    "held_out_localization_f1": 0.371134,
                    "held_out_selective_risk": 0.003193,
                    "train_corpus_size": 491,
                },
                {
                    "held_out_auroc": 0.986296,
                    "held_out_localization_f1": 0.387097,
                    "held_out_selective_risk": 0.003185,
                    "train_corpus_size": 4911,
                },
            ],
            "positive_control_passed": True,
            "status": "compounds",
            "verifier_is_oracle": False,
        },
        "generalization": {
            "detection_by_domain": [
                {
                    "auroc_ci95": [0.922285, 0.990662],
                    "detection_auroc": 0.963317,
                    "domain": "gap4_arc",
                    "n": 28443,
                    "selection_headroom": 0.129,
                }
            ],
            "detector_generalizes_cross_domain": True,
            "domains_at_chance": [],
            "status": "generalizes",
            "unavailable_domains": [
                {
                    "domain": "code_humaneval_mbpp",
                    "reason": "missing candidate source text or cached verifier_score",
                },
                {
                    "domain": "gsm8k",
                    "reason": "datasets present but no multicandidate verifier scores",
                },
            ],
            "verifier_is_oracle": False,
        },
        "arc_e3_outcomes": {
            "deeper_high_headroom": {
                "new_levels_reproduced": 0,
                "per_game_scorecard": [
                    {"game": "lp85", "lookahead_fidelity": 0.833333},
                    {"game": "tu93", "lookahead_fidelity": 0.8},
                    {"game": "tn36", "lookahead_fidelity": 0.875},
                    {"game": "tr87", "lookahead_fidelity": 0.857143},
                ],
                "status": "partial",
            },
            "new_levels_reproduced_from_artifacts": 0,
            "status": "partial",
        },
        "arc_reproducible_progress": {
            "prior_reproducible_total_games": 17,
            "prior_reproducible_total_levels": 34,
            "reproducible_total_games": 17,
            "reproducible_total_levels": 34,
            "new_games_since_prior": 0,
            "new_levels_since_prior": 0,
            "status": "loaded",
        },
        "publication_gate": {"paper_ready": True, "unmet_gates": []},
    }
    payload.update(overrides)
    return payload


def _localization(**overrides: object) -> dict:
    payload = {
        "honest_verdict": "complete: clean_powered_null_bidirectional_not_actionable",
        "detector_localization_actionable": False,
        "localization_delta_ci95": [0.0, 0.0],
        "localization_f1_by_direction": {
            "bidirectional_fusion": {
                "accuracy": 0.096491,
                "exact_match_count": 11,
                "f1": 0.096491,
                "n_error_traces": 114,
            },
            "causal_online": {
                "accuracy": 0.096491,
                "exact_match_count": 11,
                "f1": 0.096491,
                "n_error_traces": 114,
            },
            "unidirectional_l2r": {
                "accuracy": 0.096491,
                "exact_match_count": 11,
                "f1": 0.096491,
                "n_error_traces": 114,
            },
        },
        "abstention_curve": {
            "base_rate_fraction_correct": 0.98259,
            "detector_auroc": 0.979903,
            "useful_operating_point": None,
        },
        "missing_verifier_gaps": [
            {
                "gap_id": "GAP-FOVER-BIPRM-LOCALIZATION-untyped",
                "missed_first_error_traces": 103,
                "status": "open",
            }
        ],
        "n_error_traces": 114,
        "n_traces": 6548,
        "verifier_is_oracle": False,
    }
    payload.update(overrides)
    return payload


def _compounds(**overrides: object) -> dict:
    payload = {
        "honest_verdict": "success: detector_compounds_heldout_localization_f1",
        "compounding_delta_ci95": [0.003396, 0.032772],
        "detector_compounds": True,
        "learning_curve": [
            {
                "held_out_auroc": 0.986296,
                "held_out_localization_f1": 0.371134,
                "held_out_selective_risk": 0.003193,
                "train_corpus_size": 491,
            },
            {
                "held_out_auroc": 0.986296,
                "held_out_localization_f1": 0.387097,
                "held_out_selective_risk": 0.003185,
                "train_corpus_size": 4911,
            },
        ],
        "no_learning_baseline": 0.145773,
        "positive_control_passed": True,
        "verifier_is_oracle": False,
    }
    payload.update(overrides)
    return payload


def _generalization(**overrides: object) -> dict:
    payload = {
        "honest_verdict": "success: detector_generalizes_cross_domain_non_fover",
        "detector_generalizes_cross_domain": True,
        "detection_by_domain": [
            {
                "auroc_ci95": [0.922285, 0.990662],
                "detection_auroc": 0.963317,
                "domain": "gap4_arc",
                "n": 28443,
                "selection_headroom": 0.129,
            }
        ],
        "domains_at_chance": [],
        "unavailable_domains": [
            {"domain": "code_humaneval_mbpp", "reason": "missing pool"},
            {"domain": "gsm8k", "reason": "missing verifier scores"},
        ],
        "verifier_is_oracle": False,
    }
    payload.update(overrides)
    return payload


def _sota(**overrides: object) -> dict:
    payload = {
        "honest_verdict": "complete: sota_ingestion_v406_mapped",
        "flagged_for_v406": "verifiable_process_data_cross_domain_localization_v406",
        "methods_mapped": [
            {"arxiv_id_or_url": "2605.02395", "track": "cross-domain first-error localization data"},
            {"arxiv_id_or_url": "2102.10395", "track": "cross-domain detector calibration"},
            {"arxiv_id_or_url": "2605.25133", "track": "selective prediction"},
            {"arxiv_id_or_url": "2504.16828", "track": "first-error gaps"},
            {"arxiv_id_or_url": "2606.16070", "track": "ARC E3 fidelity"},
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
    (root / "ops" / "exclusion_manifest.yaml").write_text(_manifest_text(), encoding="utf-8")
    (root / "ops" / "arc_solve_registry.yaml").write_text(_registry_text(), encoding="utf-8")
    (root / "research-roadmap.yaml").write_text(
        "milestone: 2026.06.406\n"
        "milestone_doc: openspec/change-proposals/research-roadmap-v406.md\n"
        "milestone_overview: detector localization via verifiable process data\n",
        encoding="utf-8",
    )
    (root / "openspec" / "change-proposals").mkdir(parents=True, exist_ok=True)
    (root / "openspec" / "change-proposals" / "research-roadmap-v406.md").write_text(
        "# Research Roadmap v406\n\n"
        "turn detection into an actionable cross-domain LOCALIZER + fidelity gate.\n",
        encoding="utf-8",
    )
    _write_json(root / "results" / "experiment_4390_capstone_v405.json", _capstone())
    _write_json(
        root / "results" / "experiment_4381_biprm_detector_localization_abstention.json",
        _localization(),
    )
    _write_json(
        root / "results" / "experiment_4385_detector_self_learning_compounds.json",
        _compounds(),
    )
    _write_json(
        root / "results" / "experiment_4386_cross_domain_detection_generalization.json",
        _generalization(),
    )
    _write_json(root / "results" / "experiment_4387_sota_ingestion_v406.json", _sota())
    return root


def run_happy(root: Path) -> dict:
    out = mod.run(root, pretest_result=GREEN, started_s=1000.0, now_s=1000.5)
    return json.loads(out.read_text(encoding="utf-8"))


def test_req_report_4391_spec_declares_contract() -> None:
    """REQ-REPORT-4391: OpenSpec declares the true `.405` transition contract."""

    spec = Path("openspec/capabilities/research-reporting/spec.md").read_text(encoding="utf-8")
    assert "REQ-REPORT-4391" in spec
    assert "SCENARIO-REPORT-4391" in spec
    assert "SCENARIO-REPORT-4391-BLOCKED-PRECONDITION" in spec
    assert "first-error F1 `0.096`" in spec
    assert "verifiable_process_data_cross_domain_localization_v406" in spec
    assert "lookahead fidelity" in spec
    assert "`0.80-0.875`" in spec
    assert mod.FIELD_PRINCIPLES["honest_verdict"] in spec
    assert mod.FIELD_PRINCIPLES["v405_close_state"] in spec
    assert mod.FIELD_PRINCIPLES["preconditions_checked"] in spec


def test_helpers_and_archive_record_editing(tmp_path: Path) -> None:
    """REQ-REPORT-4391: helper behavior is deterministic and YAML-safe."""

    assert mod.archive_record_count(_research_complete_text(duplicates=3)) == 3
    assert mod._record_id("- id: '2026.06.405'") == "2026.06.405"
    assert mod._record_id("  - id: nested") is None
    assert mod._yaml_quote("don't reopen") == "'don''t reopen'"
    assert mod._insert_before_tasks(["  title: no tasks"], "  finding: x") == [
        "  title: no tasks",
        "  finding: x",
    ]
    assert mod._ci95(None, [0.1, 0.2]) == [0.1, 0.2]
    assert mod._first_domain([{"domain": "other", "detection_auroc": 0.6}])["domain"] == "other"
    assert mod._compounding_curve({}, {})[0]["train_corpus_size"] == 491

    root = make_repo(tmp_path)
    close_state = mod.build_v405_close_state(mod.read_v405_sources(root))
    deduped, removed, action = mod.dedupe_or_update_record(
        _research_complete_text(duplicates=3), close_state
    )
    assert (removed, action) == (2, "deduped")
    assert mod.archive_record_count(deduped) == 1
    assert "DETECTION good but LOCALIZATION clean powered NULL" in deduped
    assert "COMPOUNDS weakly" in deduped
    assert "GENERALIZES cross-domain on one non-FoVer domain" in deduped
    assert "ARC 34 reproducible levels / 17 games" in deduped
    assert "verifiable_process_data_cross_domain_localization_v406" in deduped
    assert mod.yaml_parses(deduped)

    updated, removed2, action2 = mod.dedupe_or_update_record(
        _research_complete_text(duplicates=1), close_state
    )
    assert (removed2, action2) == (0, "updated")
    unchanged, removed3, action3 = mod.dedupe_or_update_record(updated, close_state)
    assert (unchanged, removed3, action3) == (updated, 0, "unchanged")
    appended, removed4, action4 = mod.dedupe_or_update_record(
        "# history\nmilestones:\n- id: 2026.06.404\n  finding: prior\n", close_state
    )
    assert (removed4, action4) == (0, "appended")
    assert "activation_recorded: exp4391-archive-v405-activate-v406" in appended
    no_finding, _removed5, action5 = mod.dedupe_or_update_record(
        "milestones:\n- id: 2026.06.405\n  title: no finding\n  tasks:\n  - id: exp4390\n",
        close_state,
    )
    assert action5 == "updated"
    assert "  finding: '.405 close-state" in no_finding


def test_read_sources_and_build_v405_close_state(tmp_path: Path) -> None:
    """SCENARIO-REPORT-4391: close-state records the true `.405` scorecard."""

    root = make_repo(tmp_path)
    sources = mod.read_v405_sources(root)
    cited = mod.build_cited_upstream(root)
    assert sources["4390"]["detector_actionable_state"] == "detects_but_not_actionable"
    assert {item["experiment_id"] for item in cited if item["kind"] == "artifact"} == {
        "4390",
        "4381",
        "4385",
        "4386",
        "4387",
    }
    assert all(item["sha256"] is None or mod.is_sha256(item["sha256"]) for item in cited)

    state = mod.build_v405_close_state(sources)
    assert state["summary"] == (
        "detector_detects_but_not_actionable_localization_null_"
        "compounds_weakly_generalizes_arc34_v406_localizer"
    )
    assert state["detector_actionable_state"] == "detects_but_not_actionable"
    assert state["detector_detects_well"] is True
    assert state["localization_axis_state"] == "CLEAN_POWERED_NULL_DATA_PROBLEM"
    assert state["detector_localization_actionable"] is False
    assert state["localization_f1"] == 0.096491
    assert state["localization_delta_ci95"] == [0.0, 0.0]
    assert state["missed_first_error_traces"] == 103
    assert state["missing_localization_gap_id"] == "GAP-FOVER-BIPRM-LOCALIZATION-untyped"
    assert state["compounding_axis_state"] == "COMPOUNDS_WEAKLY"
    assert state["detector_compounds"] is True
    assert state["compounding_f1_first"] == 0.371134
    assert state["compounding_f1_last"] == 0.387097
    assert state["compounding_delta_ci95"] == [0.003396, 0.032772]
    assert state["generalization_axis_state"] == "GENERALIZES_ONE_NON_FOVER_DOMAIN"
    assert state["detector_generalizes_cross_domain"] is True
    assert state["cross_domain_non_fover_domains_count"] == 1
    assert state["cross_domain_domains"] == ["gap4_arc"]
    assert state["gap4_arc_detection_auroc"] == 0.963317
    assert state["gap4_arc_auroc_ci95"] == [0.922285, 0.990662]
    assert state["arc_axis_state"] == "STALLED_FIDELITY_BLOCKED"
    assert state["arc_reproducible_total_levels"] == 34
    assert state["arc_reproducible_total_games"] == 17
    assert state["lookahead_fidelity_min"] == 0.8
    assert state["lookahead_fidelity_max"] == 0.875
    assert state["flagged_for_v406"] == "verifiable_process_data_cross_domain_localization_v406"
    assert state["cross_game_value_transfer_axis_state"] == "RETIRED_EXP4342_THIRD_NULL"
    assert state["cross_domain_selection_axis_state"] == "RETIRED_EXP4314_DOMAIN_BOUND"
    assert state["in_generation_axis_state"] == "RETIRED_EXP4374_FOURTH_BLOCK"
    assert state["llm_heuristic_efficiency_axis_state"] == "SETTLED_EXP4370_CLEAN_NULL"
    assert state["paper_ready"] is True
    assert state["v406_frame"] == mod.V406_FRAME


def test_run_happy_path_writes_valid_artifact_and_updates_archive(tmp_path: Path) -> None:
    """SCENARIO-REPORT-4391: complete path writes the terminal archive artifact."""

    root = make_repo(tmp_path, duplicates=2)
    artifact = run_happy(root)
    assert artifact["honest_verdict"].startswith("success:")
    assert artifact["archived_milestone"] == "2026.06.405"
    assert artifact["activated_milestone"] == "2026.06.406"
    assert artifact["active_milestone_confirmed"] == "2026.06.406"
    assert artifact["research_complete_yaml_parses"] is True
    assert artifact["exclusion_manifest_parses"] is True
    assert artifact["pretest_suite_green"] is True
    assert artifact["research_complete_record_action"] == "deduped"
    assert artifact["research_complete_duplicates_removed"] == 1
    assert artifact["v405_close_state"]["localization_axis_state"] == (
        "CLEAN_POWERED_NULL_DATA_PROBLEM"
    )
    assert artifact["v405_close_state"]["detector_compounds"] is True
    assert artifact["v405_close_state"]["detector_generalizes_cross_domain"] is True
    assert artifact["v405_close_state"]["arc_reproducible_total_levels"] == 34
    assert artifact["v405_close_state"]["arc_reproducible_total_games"] == 17
    assert artifact["v405_close_state"]["flagged_for_v406"] == (
        "verifiable_process_data_cross_domain_localization_v406"
    )
    assert (
        artifact["field_principles"]["v405_close_state"] == mod.FIELD_PRINCIPLES["v405_close_state"]
    )
    assert artifact["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert mod.is_sha256(artifact["reproducibility_checksum"])
    complete_text = (root / "research-complete.yaml").read_text(encoding="utf-8")
    assert mod.archive_record_count(complete_text) == 1
    assert "DATA problem" in complete_text
    assert "turn detection into an actionable cross-domain LOCALIZER" in complete_text
    mod.validate_artifact(artifact)


def test_run_blocked_preconditions_before_research_complete_edit(tmp_path: Path) -> None:
    """SCENARIO-REPORT-4391-BLOCKED-PRECONDITION: blocked paths do not fabricate."""

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
    (root4 / "research-roadmap.yaml").write_text("milestone: 2026.06.405\n", encoding="utf-8")
    artifact4 = json.loads(mod.run(root4, pretest_result=GREEN).read_text(encoding="utf-8"))
    assert artifact4["honest_verdict"] == "blocked_v406_not_active"

    root5 = make_repo(tmp_path / "source_missing")
    (root5 / "results" / "experiment_4390_capstone_v405.json").unlink()
    artifact5 = json.loads(mod.run(root5, pretest_result=GREEN).read_text(encoding="utf-8"))
    assert artifact5["honest_verdict"] == "blocked_v405_capstone_missing"

    root6 = make_repo(tmp_path / "sota_missing")
    (root6 / "results" / "experiment_4387_sota_ingestion_v406.json").unlink()
    artifact6 = json.loads(mod.run(root6, pretest_result=GREEN).read_text(encoding="utf-8"))
    assert artifact6["honest_verdict"] == "blocked_sota_ingestion_v406_missing"


def test_run_blocked_research_complete_edit_guards(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-REPORT-4391: invalid archive edits are blocked before completion."""

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
    """REQ-REPORT-4391: schema validation and entrypoints preserve the contract."""

    root = make_repo(tmp_path)
    state = mod.build_v405_close_state(mod.read_v405_sources(root))
    complete = mod.build_complete_artifact(
        v405_close_state=state,
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

    import carnot.experiment_4391_archive_v405_activate_v406 as entrypoint

    called: dict[str, Path] = {}

    def fake_run(root_path: Path = mod.REPO_ROOT) -> Path:
        called["root"] = Path(root_path)
        return root / mod.OUTPUT_REL_PATH

    monkeypatch.setattr(entrypoint, "run", fake_run)
    assert entrypoint.main() == 0
    assert called["root"] == entrypoint.REPO_ROOT

    script_path = Path("results/experiment_4391_archive_v405_activate_v406.py")
    script_repo_root = str(script_path.resolve().parents[1])
    script_python_root = str(Path(script_repo_root) / "python")
    original_sys_path = list(sys.path)
    try:
        sys.path[:] = [
            item for item in sys.path if item not in {script_repo_root, script_python_root}
        ]
        spec = importlib.util.spec_from_file_location("exp4391_archive_script", script_path)
        assert spec and spec.loader
        script = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(script)
    finally:
        sys.path[:] = original_sys_path
    monkeypatch.setattr(script, "run", fake_run)
    assert script.main() == 0


def test_validate_artifact_rejection_guards(tmp_path: Path) -> None:
    """REQ-REPORT-4391: validation rejects artifacts that launder the `.405` truth."""

    good = run_happy(make_repo(tmp_path))

    def set_path(obj: dict, path: list[str], value: object) -> None:
        cur = obj
        for key in path[:-1]:
            cur = cur[key]
        cur[path[-1]] = value

    cases = [
        ("missing required fields", lambda a: a.pop("v405_close_state")),
        ("terminal-prefixed", lambda a: a.__setitem__("honest_verdict", "done")),
        (
            "principle must match REQ-REPORT-4391",
            lambda a: a["field_principles"].__setitem__("v405_close_state", "wrong"),
        ),
        ("archived milestone", lambda a: a.__setitem__("archived_milestone", "2026.06.999")),
        ("activated milestone", lambda a: a.__setitem__("activated_milestone", "2026.06.999")),
        ("research-complete YAML", lambda a: a.__setitem__("research_complete_yaml_parses", False)),
        ("exclusion manifest", lambda a: a.__setitem__("exclusion_manifest_parses", False)),
        ("pretest suite", lambda a: a.__setitem__("pretest_suite_green", False)),
        ("active milestone", lambda a: a.__setitem__("active_milestone_confirmed", "2026.06.405")),
        ("v405_close_state must be a mapping", lambda a: a.__setitem__("v405_close_state", "x")),
        (
            "detector actionability",
            lambda a: set_path(a, ["v405_close_state", "detector_actionable_state"], "open"),
        ),
        (
            "localization null",
            lambda a: set_path(a, ["v405_close_state", "localization_axis_state"], "won"),
        ),
        ("localization F1", lambda a: set_path(a, ["v405_close_state", "localization_f1"], 0.5)),
        ("localization CI", lambda a: set_path(a, ["v405_close_state", "localization_delta_ci95"], [0, 1])),
        (
            "missed first errors",
            lambda a: set_path(a, ["v405_close_state", "missed_first_error_traces"], 0),
        ),
        (
            "compounds weakly",
            lambda a: set_path(a, ["v405_close_state", "compounding_axis_state"], "strong"),
        ),
        (
            "detector compounds",
            lambda a: set_path(a, ["v405_close_state", "detector_compounds"], False),
        ),
        (
            "generalizes",
            lambda a: set_path(a, ["v405_close_state", "detector_generalizes_cross_domain"], False),
        ),
        (
            "one non-FoVer",
            lambda a: set_path(a, ["v405_close_state", "cross_domain_non_fover_domains_count"], 2),
        ),
        ("GAP-4 AUROC", lambda a: set_path(a, ["v405_close_state", "gap4_arc_detection_auroc"], 0.5)),
        (
            "ARC stalled",
            lambda a: set_path(a, ["v405_close_state", "arc_axis_state"], "advanced"),
        ),
        ("ARC 34", lambda a: set_path(a, ["v405_close_state", "arc_reproducible_total_levels"], 33)),
        ("ARC games", lambda a: set_path(a, ["v405_close_state", "arc_reproducible_total_games"], 16)),
        ("fidelity", lambda a: set_path(a, ["v405_close_state", "lookahead_fidelity_min"], 0.9)),
        (
            "flagged_for_v406",
            lambda a: set_path(a, ["v405_close_state", "flagged_for_v406"], "other"),
        ),
        (
            "cross-game retired",
            lambda a: set_path(
                a, ["v405_close_state", "cross_game_value_transfer_axis_state"], "OPEN"
            ),
        ),
        (
            "cross-domain selection retired",
            lambda a: set_path(
                a, ["v405_close_state", "cross_domain_selection_axis_state"], "OPEN"
            ),
        ),
        (
            "in-generation retired",
            lambda a: set_path(a, ["v405_close_state", "in_generation_axis_state"], "OPEN"),
        ),
        (
            "LLM heuristic settled",
            lambda a: set_path(
                a, ["v405_close_state", "llm_heuristic_efficiency_axis_state"], "OPEN"
            ),
        ),
        ("paper", lambda a: set_path(a, ["v405_close_state", "paper_ready"], False)),
        ("v406 frame", lambda a: set_path(a, ["v405_close_state", "v406_frame"], "redo")),
    ]
    for label, mutate in cases:
        artifact = copy.deepcopy(good)
        mutate(artifact)
        with pytest.raises(ValueError, match=label):
            mod.validate_artifact(artifact)
