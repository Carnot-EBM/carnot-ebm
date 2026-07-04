"""Tests for Exp 5207 archive .476 / activate .477.

Spec refs: REQ-REPORT-5207, SCENARIO-REPORT-5207,
SCENARIO-REPORT-5207-BLOCKED-PRECONDITION.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import yaml

from carnot import experiment_5207_archive_476_activate_477 as mod


SPEC_PATH = Path("openspec/capabilities/research-reporting/spec.md")


def _wrap(value: Any, principle: str = "fixture principle") -> dict[str, Any]:
    return {"principle": principle, "value": value}


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _roadmap(milestone: str, start: int, stop: int) -> str:
    return yaml.safe_dump(
        {
            "milestone": milestone,
            "milestone_title": f"fixture {milestone}",
            "tasks": [
                {
                    "id": f"exp{idx}-fixture-v{milestone.split('.')[-1]}",
                    "title": f"fixture task {idx}",
                    "agent_type": "codex",
                    "prompt": "fixture",
                }
                for idx in range(start, stop + 1)
            ],
        },
        sort_keys=False,
    )


def _payloads() -> dict[int, dict[str, Any]]:
    return {
        5193: {
            "experiment": "experiment_5193_archive_475_activate_476",
            "honest_verdict": _wrap("complete_archive_475_closed_476_active_precise_handoff_clean"),
        },
        5194: {
            "experiment": "experiment_5194_poison_test_cascade_triage_module_v476",
            "honest_verdict": _wrap("complete_pretest_triage_module_ready"),
        },
        5195: {
            "experiment": "experiment_5195_retro_timing_real_fix_known_issues_dedup_v476",
            "honest_verdict": "complete: retro_timing_475_false_zero_root_cause_found_and_fixed",
        },
        5196: {
            "experiment": "experiment_5196_diffusiongemma_vllm_native_retry_v476",
            "honest_verdict": "blocked_diffusiongemma_loading_exhausted_v476",
            "diffusiongemma_loadable": False,
            "forward_pass_confirmed": False,
            "loading_path_used": "both_failed",
            "retirement": "DiffusionGemma live-loading thread RETIRES per prior_failures retire_if_same_verdict=true.",
        },
        5197: {
            "experiment": "experiment_5197_gap4_scaleup_real_checkpoint_v476",
            "honest_verdict": "complete_gap4_scaleup_v476_n62_source_pool_exhausted_floor_not_crossed_scale_up_recommended",
            "n_reached": _wrap(62),
            "target_n": 180,
            "already_scored_prior_n": 62,
            "new_rows_scored": 0,
            "source_pool_exhausted_before_new_rows": True,
            "exact_test_discordant_wins": _wrap(4),
            "exact_test_discordant_losses": _wrap(0),
            "exact_test_p_value_two_sided": _wrap(0.125),
            "exact_test_passes_min6_rule": _wrap(False),
        },
        5198: {
            "experiment": "experiment_5198_map_landmark_prestage_prototype_v476",
            "honest_verdict": "complete: MAP landmark prestage did not bank a new reproduction-gated level over pruner-only; the GAP-4891 enumeration wall persists under this lever too.",
            "lever_validated": False,
            "levels_banked": [],
            "gap4891_status_recommendation": "building_enumeration_wall_persists_under_map_prestage",
            "target_games": ["cd82", "sk48", "sp80"],
            "games_tested": ["cd82", "sk48", "sp80", "cn04"],
        },
        5199: {
            "experiment": 5199,
            "status": "blocked",
            "blocked_at_layer": "conductor_pre_gate",
            "honest_verdict": "blocked_gate_check_failed",
        },
        5200: {
            "experiment": "experiment_5200_hidden_state_verifier_v2_mmlu_pro_v476",
            "honest_verdict": _wrap(
                "complete_hidden_state_probe_does_not_beat_tuned_sc_probe0.100_sc0.075_self0.075_clue0.100_rcs0.100"
            ),
            "n_questions": _wrap(40),
            "probe_accuracy": _wrap(0.100),
            "tuned_sc_accuracy": _wrap(0.075),
            "self_certainty_accuracy": _wrap(0.075),
            "clue_accuracy": _wrap(0.100),
            "radial_consensus_score_accuracy": _wrap(0.100),
        },
        5201: {
            "experiment": "experiment_5201_hardware_continuity_gatemate_diagnostic",
            "honest_verdict": "complete_hardware_continuity_gatemate_diagnostic_kv260:reachable_gatemate:blocked_gatemate_dirtyjtag_idcode_unresolved_v476_narrowed_jtag_protocol_level_polarfire:reachable_no_speedup_claim",
            "boards_reachable_count": 2,
            "kv260_status": "reachable",
            "polarfire_status": "reachable",
            "gatemate_status": "blocked_dirtyjtag_idcode_unresolved",
            "gatemate_diagnostic_narrowed_to": "jtag_protocol_level",
            "hardware_speedup_claimed": False,
            "no_speedup_claim": True,
        },
        5202: {
            "experiment": "experiment_5202_architecture_md_reconciliation_v476",
            "honest_verdict": _wrap(
                "complete: architecture_md_reconciled_20260703_arc_phase_d_hidden_state_hardware"
            ),
        },
        5203: {
            "experiment": "experiment_5203_verifier_authenticity_remediation_options_v476",
            "honest_verdict": _wrap(
                "complete: verifier_authenticity_remediation_options_v476_ready"
            ),
        },
        5204: {
            "experiment": "experiment_5204_exclusion_manifest_lint_real_bug_fix_v476",
            "honest_verdict": _wrap(
                "success: exclusion_manifest_lint_real_bug_fixed_all_four_issues_word_boundary_principle_unwrap_general_negation_terminal_prefix"
            ),
        },
        5205: {
            "experiment": "experiment_5205_autopyverifier_gap1_pilot_v476",
            "honest_verdict": _wrap(
                "complete: set_search_beats_always_on_beats_single_refuted_baseline_0.0879_best_0.2218_single_refuted_0.1506_captured_47_of_239_gap1_candidate_positive"
            ),
            "pass_at_2_baseline_always_on_only": _wrap(0.087866),
            "pass_at_2_best_subset": _wrap(0.221757),
            "single_refuted_directional_adjacency_pass@2": 0.150628,
            "transpose_distractor_count": 239,
            "transpose_misvotes_captured": _wrap("47 out of 239"),
            "best_subset_found": _wrap(
                ["border_ordered_profile", "color_centroid_orientation", "row_column_run_profile"]
            ),
        },
        5206: {
            "experiment": "experiment_5206_capstone_v476",
            "honest_verdict": "complete: v476 reconciled with DiffusionGemma loading retired, GAP-4891 and GAP-4 still open, exp5199 accurately gated rather than failed, zero new ARC levels banked, and no flagged_adversarial upstreams headlined.",
            "diffusiongemma_arc_reconciled": _wrap("loading_not_achieved_thread_retired"),
            "gap4891_status_reconciled": _wrap(
                "building_enumeration_wall_persists_under_map_prestage_not_filled"
            ),
            "gap4_status_reconciled": _wrap("scale_up_recommended_not_filled"),
            "hidden_state_verifier_v2_reconciled": _wrap("does_not_beat_all_controls"),
            "reproducible_total_levels_delta": _wrap(0),
            "flagged_adversarial_artifacts_excluded": _wrap([]),
            "research_conductor_py_untouched_confirmed": _wrap(True),
        },
    }


def _make_repo(
    root: Path, *, active: bool = True, next_file: bool = False, omit: set[int] | None = None
) -> Path:
    omit = omit or set()
    (root / "results").mkdir(parents=True, exist_ok=True)
    (root / "ops").mkdir(parents=True, exist_ok=True)
    (root / "scripts").mkdir(parents=True, exist_ok=True)
    (root / "openspec/change-proposals").mkdir(parents=True, exist_ok=True)
    (root / "research-roadmap.yaml").write_text(
        _roadmap(
            "2026.07.477" if active else "2026.07.476",
            5207 if active else 5193,
            5219 if active else 5206,
        ),
        encoding="utf-8",
    )
    if next_file:
        (root / "research-roadmap-next.yaml").write_text(
            _roadmap("2026.07.477", 5207, 5219), encoding="utf-8"
        )
    (root / "openspec/change-proposals" / "research-roadmap-vNEXT.md").write_text(
        "# Research Roadmap vNEXT\n\n**Milestone:** `2026.07.477`\n",
        encoding="utf-8",
    )
    (root / "ops" / "conductor-log.md").write_text(
        "| 2026-07-04 00:06 UTC | Milestone 2026.07.477 activated | OK | 13 tasks queued |\n",
        encoding="utf-8",
    )
    (root / "ops" / "exclusion_manifest.yaml").write_text("retired_extras: []\n", encoding="utf-8")
    (root / "scripts" / "research_conductor.py").write_text(
        "# conductor fixture\n", encoding="utf-8"
    )
    for source in mod.UPSTREAM_SOURCES:
        if source.experiment_number not in omit:
            _write_json(root / source.relative_path, _payloads()[source.experiment_number])
    return root


def _clean_validation_results() -> list[mod.CommandResult]:
    return [
        mod.CommandResult(
            command=(
                ".venv/bin/python",
                "scripts/exclusion_manifest_lint.py",
                "research-roadmap.yaml",
            ),
            exit_code=0,
            stdout="Exclusion-manifest lint clean: research-roadmap.yaml",
            stderr="",
        ),
        mod.CommandResult(
            command=(
                ".venv/bin/python",
                "scripts/validate_prior_failures.py",
                "research-roadmap.yaml",
            ),
            exit_code=0,
            stdout="[OK] research-roadmap.yaml -- no schema errors, no prior_failures violations",
            stderr="",
        ),
    ]


def test_req_report_5207_spec_declares_archive_contract() -> None:
    """REQ-REPORT-5207: OpenSpec anchors the .476 archive and .477 activation contract."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    section = spec[spec.index("REQ-REPORT-5207") :]

    for marker in (
        "REQ-REPORT-5207",
        "SCENARIO-REPORT-5207",
        "SCENARIO-REPORT-5207-BLOCKED-PRECONDITION",
        str(mod.RESULT_RELATIVE_PATH),
        "v476_summary",
        "validation_commands_run",
        "aggregation_from_upstream_artifacts",
    ):
        assert marker in section
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert f"`{field}`" in section
        assert principle in section


def test_scenario_report_5207_already_active_preserves_precise_v476_truth(tmp_path: Path) -> None:
    """SCENARIO-REPORT-5207: already-active .477 handoff records exact .476 facts."""

    artifact = mod.build_artifact(
        root=_make_repo(tmp_path, active=True),
        run_date="20260704",
        duration_s=1.25,
        validation_results=_clean_validation_results(),
        conductor_untouched=True,
        tests_run=["focused"],
    )

    mod.validate_artifact(artifact)
    assert artifact["experiment_id"] == mod.EXPERIMENT_ID
    assert artifact["archived_milestone"] == "2026.07.476"
    assert artifact["milestone"] == "2026.07.477"
    assert artifact["research_roadmap_yaml_activated"]["value"] is True
    assert (
        artifact["roadmap_activation_check"]["activation_source"]
        == "research-roadmap.yaml_already_active"
    )
    assert artifact["exclusion_manifest_confirmed_clean"]["value"] is True
    assert artifact["ops_docs_updated"]["value"] is False
    assert artifact["research_conductor_py_untouched_confirmed"]["value"] is True
    assert artifact["inference_substrate"]["value"] == "aggregation_from_upstream_artifacts"
    assert artifact["honest_verdict"]["value"].startswith("complete:")
    assert ".477 activated" in artifact["honest_verdict"]["value"]

    summary = artifact["v476_summary"]["value"]
    for required in (
        "exp5205",
        "pass@2 0.221757",
        "0.087866",
        "0.150628",
        "47/239",
        "exp5197",
        "n=62",
        "4/0",
        "p=0.125",
        "six-win",
        "exp5198",
        "zero MAP/landmark levels",
        "GAP-4891",
        "exp5200",
        "0.100",
        "CLUE",
        "RCS",
        "exp5206",
        "DiffusionGemma loading retired",
        "zero new ARC levels",
    ):
        assert required in summary

    commands = artifact["validation_commands_run"]["value"]
    assert [row["passed"] for row in commands] == [True, True]
    assert "exclusion_manifest_lint.py" in commands[0]["command"]
    assert "validate_prior_failures.py" in commands[1]["command"]
    assert artifact["failed_preconditions"] == []
    assert artifact["reproducibility_checksum"] == mod.payload_checksum(artifact)


def test_scenario_report_5207_copies_next_roadmap_exactly_when_present(tmp_path: Path) -> None:
    """SCENARIO-REPORT-5207: activation copies research-roadmap-next.yaml exactly."""

    root = _make_repo(tmp_path, active=False, next_file=True)
    next_text = (root / "research-roadmap-next.yaml").read_text(encoding="utf-8")
    artifact = mod.build_artifact(
        root=root,
        run_date="20260704",
        duration_s=0.5,
        validation_results=_clean_validation_results(),
        conductor_untouched=True,
        tests_run=["focused"],
    )

    assert (root / "research-roadmap.yaml").read_text(encoding="utf-8") == next_text
    assert (
        artifact["roadmap_activation_check"]["activation_source"]
        == "copied_research-roadmap-next.yaml"
    )
    assert artifact["roadmap_activation_check"]["copied_research_roadmap_next"] is True
    assert artifact["archived_research_roadmap_yaml"]["milestone"] == "2026.07.476"
    assert artifact["archived_research_roadmap_yaml"]["task_count"] == 14
    assert artifact["research_roadmap_yaml_activated"]["value"] is True


def test_scenario_report_5207_blocked_preconditions_are_visible(tmp_path: Path) -> None:
    """SCENARIO-REPORT-5207-BLOCKED-PRECONDITION: dirty inputs block clean claims."""

    bad_validation = [
        mod.CommandResult(
            command=(
                ".venv/bin/python",
                "scripts/exclusion_manifest_lint.py",
                "research-roadmap.yaml",
            ),
            exit_code=1,
            stdout="HARD violation",
            stderr="",
        )
    ]
    artifact = mod.build_artifact(
        root=_make_repo(tmp_path, active=True, omit={5206}),
        run_date="20260704",
        duration_s=0.75,
        validation_results=bad_validation,
        conductor_untouched=False,
        tests_run=["focused"],
    )

    mod.validate_artifact(artifact)
    assert artifact["clean_handoff"] is False
    assert artifact["honest_verdict"]["value"] == mod.BLOCKED_VERDICT
    assert artifact["exclusion_manifest_confirmed_clean"]["value"] is False
    assert artifact["research_conductor_py_untouched_confirmed"]["value"] is False
    assert "missing_artifact_exp5206" in artifact["failed_preconditions"]
    assert (
        "validation_failed_scripts/exclusion_manifest_lint.py" in artifact["failed_preconditions"]
    )
    assert "scripts_research_conductor_py_modified" in artifact["failed_preconditions"]

    with pytest.raises(ValueError, match="terminal"):
        mod.validate_artifact(
            artifact | {"honest_verdict": _wrap("done", mod.FIELD_PRINCIPLES["honest_verdict"])}
        )
    with pytest.raises(ValueError, match="inference_substrate"):
        mod.validate_artifact(
            artifact
            | {
                "inference_substrate": _wrap(
                    "live_llm_inference", mod.FIELD_PRINCIPLES["inference_substrate"]
                )
            }
        )
    with pytest.raises(ValueError, match="tests_run"):
        mod.validate_artifact(artifact | {"tests_run": []})
    with pytest.raises(ValueError, match="reproducibility_checksum"):
        mod.validate_artifact(artifact | {"reproducibility_checksum": "sha256:bad"})


def test_req_report_5207_run_writes_valid_artifact(tmp_path: Path) -> None:
    """REQ-REPORT-5207: run writes the terminal JSON artifact."""

    root = _make_repo(tmp_path, active=True)
    out = mod.run(
        root=root,
        run_date="20260704",
        duration_s=0.25,
        validation_results=_clean_validation_results(),
        conductor_untouched=True,
        tests_run=["focused"],
    )

    assert out == root / mod.RESULT_RELATIVE_PATH
    saved = json.loads(out.read_text(encoding="utf-8"))
    mod.validate_artifact(saved)
    assert saved["tests_run"] == ["focused"]
    assert saved["reproducibility_checksum"] == mod.payload_checksum(saved)


def test_req_report_5207_helper_edges_and_default_validation_path(tmp_path: Path) -> None:
    """REQ-REPORT-5207: helper edges keep blocked artifacts explicit instead of crashing."""

    assert mod.value_of({"value": {"value": "nested"}}) == "nested"
    assert mod._number(True) is None
    assert mod._number("not-a-number") is None
    assert mod._int("bad") == 0
    assert mod._list("not-list") == []
    assert mod.file_sha256(tmp_path / "missing") is None
    assert mod._roadmap_data("bad: [") == {}
    assert mod._task_ids({"tasks": "not-list"}) == []
    assert mod._command_label("") == "unknown_command"
    assert mod._command_label("custom --flag") == "custom"
    assert mod.exclusion_manifest_clean([]) is False
    assert mod._captured_fraction("47 out of 239", 239) == "47/239"
    assert mod._captured_fraction("5", 10) == "5/10"
    assert mod._captured_fraction("", 0) == ""
    assert mod._fmt_float(None) == "unknown"
    assert mod._fmt_p(None) == "unknown"
    assert mod._status_summary({"summary": "reachable smoke"}, "", "", "") == "reachable smoke"
    assert mod._status_summary({"status": "blocked_idcode"}, "", "", "") == "blocked_idcode"
    assert mod._status_summary({"reachable": True}, "", "", "") == "reachable"
    assert mod._status_summary("", "kv260:reachable", "kv260:reachable", "reachable") == "reachable"

    malformed = tmp_path / "malformed.json"
    malformed.write_text("{bad", encoding="utf-8")
    assert mod.read_json_mapping(malformed)[1]["error"] == "malformed_json"
    array_json = tmp_path / "array.json"
    array_json.write_text("[]", encoding="utf-8")
    assert mod.read_json_mapping(array_json)[1]["error"] == "not_json_object"

    missing_root = tmp_path / "missing-root"
    missing_root.mkdir()
    assert mod.activate_roadmap(missing_root)["activated"] is False
    assert mod.research_conductor_untouched(missing_root) is False
    (missing_root / "scripts").mkdir()
    (missing_root / "scripts" / "research_conductor.py").write_text("# fixture\n", encoding="utf-8")
    assert mod.research_conductor_untouched(missing_root) is True
    assert mod.research_conductor_untouched(Path.cwd()) is True

    failures = mod._failed_preconditions(
        missing_artifacts=["missing_artifact_exp5205"],
        roadmap_activation={"activated": False},
        validation=[],
        conductor_clean=False,
        vnext_present=False,
    )
    assert failures == [
        "missing_artifact_exp5205",
        "research_roadmap_yaml_not_active_for_477",
        "validation_commands_missing",
        "scripts_research_conductor_py_modified",
        "research_roadmap_vnext_doc_missing",
    ]

    root = _make_repo(tmp_path / "default-run", active=True)
    (root / "scripts" / "exclusion_manifest_lint.py").write_text(
        "print('Exclusion-manifest lint clean: research-roadmap.yaml')\n",
        encoding="utf-8",
    )
    (root / "scripts" / "validate_prior_failures.py").write_text(
        "print('[OK] research-roadmap.yaml -- no schema errors, no prior_failures violations')\n",
        encoding="utf-8",
    )
    artifact = mod.build_artifact(root=root)
    assert artifact["clean_handoff"] is True
    assert artifact["tests_run"] == mod.DEFAULT_TESTS_RUN
    assert [row["passed"] for row in artifact["validation_checks"]] == [True, True]

    blocked_summary = mod.build_v476_summary({5197: {"n_reached": "bad"}, 5205: {}})
    assert "pass@2 unknown" in blocked_summary
    assert "p=unknown" in blocked_summary


def test_req_report_5207_validation_rejects_schema_and_type_errors(tmp_path: Path) -> None:
    """REQ-REPORT-5207: artifact validation rejects schema drift and overclaims."""

    artifact = mod.build_artifact(
        root=_make_repo(tmp_path, active=True),
        run_date="20260704",
        duration_s=1.0,
        validation_results=_clean_validation_results(),
        conductor_untouched=True,
        tests_run=["focused"],
    )

    with pytest.raises(ValueError, match="missing required"):
        mod.validate_artifact({key: value for key, value in artifact.items() if key != "schema"})
    with pytest.raises(ValueError, match="schema"):
        mod.validate_artifact(artifact | {"schema": "wrong"})
    with pytest.raises(ValueError, match="milestone"):
        mod.validate_artifact(artifact | {"milestone": "2026.07.476"})
    with pytest.raises(ValueError, match="field principle"):
        mod.validate_artifact(
            artifact
            | {"field_principles": artifact["field_principles"] | {"honest_verdict": "wrong"}}
        )
    with pytest.raises(ValueError, match="principle-wrapped"):
        mod.validate_artifact(artifact | {"v476_summary": "not-wrapped"})
    with pytest.raises(ValueError, match="principle mismatch"):
        mod.validate_artifact(artifact | {"v476_summary": _wrap("summary")})
    with pytest.raises(ValueError, match="missing value"):
        mod.validate_artifact(
            artifact | {"v476_summary": {"principle": mod.FIELD_PRINCIPLES["v476_summary"]}}
        )
    with pytest.raises(ValueError, match="research_roadmap_yaml_activated"):
        mod.validate_artifact(
            artifact
            | {
                "research_roadmap_yaml_activated": _wrap(
                    "yes", mod.FIELD_PRINCIPLES["research_roadmap_yaml_activated"]
                )
            }
        )
    with pytest.raises(ValueError, match="exclusion_manifest_confirmed_clean"):
        mod.validate_artifact(
            artifact
            | {
                "exclusion_manifest_confirmed_clean": _wrap(
                    "yes", mod.FIELD_PRINCIPLES["exclusion_manifest_confirmed_clean"]
                )
            }
        )
    with pytest.raises(ValueError, match="ops_docs_updated"):
        mod.validate_artifact(
            artifact | {"ops_docs_updated": _wrap("no", mod.FIELD_PRINCIPLES["ops_docs_updated"])}
        )
    with pytest.raises(ValueError, match="research_conductor"):
        mod.validate_artifact(
            artifact
            | {
                "research_conductor_py_untouched_confirmed": _wrap(
                    "yes", mod.FIELD_PRINCIPLES["research_conductor_py_untouched_confirmed"]
                )
            }
        )
    with pytest.raises(ValueError, match="validation_commands_run"):
        mod.validate_artifact(
            artifact
            | {
                "validation_commands_run": _wrap(
                    "not-list", mod.FIELD_PRINCIPLES["validation_commands_run"]
                )
            }
        )
    with pytest.raises(ValueError, match="clean_handoff"):
        mod.validate_artifact(artifact | {"failed_preconditions": ["x"]})
