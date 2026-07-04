"""Tests for Exp 5220 archive .477 / activate .478.

Spec refs: REQ-REPORT-5220, SCENARIO-REPORT-5220,
SCENARIO-REPORT-5220-BLOCKED-PRECONDITION.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import yaml

from carnot import experiment_5220_archive_477_activate_478 as mod


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
        5207: {
            "experiment": "experiment_5207_archive_476_activate_477",
            "honest_verdict": _wrap(
                "complete: .476 archived and .477 activated; handoff preserves GAP-1 positive, GAP-4/MAP/hidden-state nulls, DiffusionGemma retirement, and hardware reachability facts."
            ),
        },
        5208: {
            "experiment_id": "exp5208-sota-ingestion-v477",
            "honest_verdict": _wrap(
                "complete: V477 SOTA refresh found no new actionable findings beyond the planning section; research-references.md unchanged; Semantic Scholar returned HTTP/2 429 for both EBT and ARM-EBM, so no fresh citation trail was inferred."
            ),
        },
        5209: {
            "experiment_id": 5209,
            "honest_verdict": _wrap(
                "complete: set_search_remains_positive_after_hardening_heldout_0.1896_always_0.0890_single_refuted_0.1478_paired_delta_ci95_0.0231_0.0604_best_subset_not_stable_do_not_promote_to_registry_here"
            ),
            "gap1_hardened_positive": _wrap(True),
            "heldout_pass_at_2_mean": _wrap(0.189584),
            "baseline_always_on_pass_at_2_mean": _wrap(0.088976),
            "single_refuted_directional_pass_at_2_mean": _wrap(0.147787),
            "delta_over_always_on": _wrap(0.100608),
            "delta_over_single_refuted": _wrap(0.041797),
            "paired_delta_ci95": _wrap("[0.023148, 0.060446]"),
            "best_subset_stable": _wrap(False),
            "leakage_audit_passed": _wrap(True),
            "n_grouped_splits": _wrap(20),
        },
        5210: {
            "experiment": 5210,
            "status": "blocked",
            "honest_verdict": "blocked_gate_check_failed",
            "blocked_at_layer": "conductor_pre_gate",
            "gate_check_summary": "1 of 1 gate(s) failed; first failure: exp5209-gap1-set-search-holdout-hardening-v477.gap1_hardened_positive (actual={'principle': 'BARE top-level boolean', 'value': True} == expected=True)",
            "gates_evaluated": [
                {
                    "artifact_field": "gap1_hardened_positive",
                    "expected": True,
                    "actual": {"principle": "BARE top-level boolean", "value": True},
                    "passed": False,
                }
            ],
        },
        5211: {
            "experiment_id": 5211,
            "honest_verdict": "complete_gap4_sota_local_candidate_expansion_v477_n120_pool_ready_for_exp5212",
            "candidate_pool_n": 120,
            "source_task_count": 180,
            "accepted_rows": 120,
            "gap4_expansion_usable": True,
            "flagged_adversarial": True,
            "leakage_audit_passed": True,
            "corrigendum_pending": [
                {"kind": "TAUTOLOGY", "severity": "critical"},
                {"kind": "DURATION_TOO_SHORT", "severity": "critical"},
                {"kind": "METHODOLOGY_MISSING", "severity": "warn"},
            ],
            "generation_errors": ["ValueError:Requested tokens exceed context window"],
        },
        5212: {
            "experiment_id": 5212,
            "honest_verdict": "complete_gap4_scale_validation_v477_n0_missing_protocol_pass2_fields_blocked",
            "flagged_adversarial": True,
            "failure_mode": "missing_protocol_pass2_fields",
            "gap4_status_recommendation": _wrap("blocked"),
            "exp5211_candidate_pool_n": 120,
            "exp5211_gap4_expansion_usable": True,
            "exact_test_discordant_wins": _wrap(0),
            "exact_test_discordant_losses": _wrap(0),
            "exact_test_passes_min6_rule": _wrap(False),
            "n_scored": _wrap(0),
            "exclusion_summary": {"missing_protocol_pass2_fields": 120},
            "corrigendum_pending": [
                {"kind": "TAUTOLOGY", "severity": "critical"},
                {"kind": "IMPLAUSIBLE_PERFECT", "severity": "info"},
            ],
        },
        5213: {
            "experiment_id": 5213,
            "honest_verdict": _wrap(
                "complete_hidden_state_v3_signal_does_not_beat_all_controls_retires_mmlu_hidden_state_path_probe0.075_sc0.075_self0.075_clue0.025_rcs0.025"
            ),
            "best_probe_accuracy": _wrap(0.075),
            "tuned_sc_accuracy": _wrap(0.075),
            "self_certainty_accuracy": _wrap(0.075),
            "clue_accuracy": _wrap(0.025),
            "radial_consensus_score_accuracy": _wrap(0.025),
            "beats_all_controls": _wrap(False),
            "retire_mmlu_hidden_state_path": _wrap(True),
        },
        5214: {
            "experiment_id": 5214,
            "honest_verdict": _wrap(
                "complete: verifier_memory_from_upstream_artifacts_promotions_1_rollbacks_1_heldout_gate_required_no_registry_claim"
            ),
            "continuous_self_learning_task": _wrap(True),
            "memory_artifact_path": _wrap("results/verifier_memory_v477.json"),
            "memory_entries_written": _wrap(2),
            "promotions": _wrap(1),
            "rollbacks": _wrap(1),
            "memory_summary": _wrap(
                {
                    "promoted_memory_ids": ["verifier-memory:fdd0d952dbf7f33e"],
                    "rolled_back_memory_ids": ["verifier-memory:d7f9fad14ee64512"],
                }
            ),
            "heldout_gate_required_for_promotion": _wrap(True),
        },
        5215: {
            "experiment": "experiment_5215_arc_paw_amortization_gate_v477",
            "honest_verdict": _wrap("complete_paw_amortization_gate_not_viable_no_arc_solve_claim"),
            "paw_amortization_viable": _wrap(False),
            "flagged_adversarial": True,
            "level_solve_claimed": False,
            "median_remaining_actions": _wrap(29.5),
            "p75_remaining_actions": _wrap(43.75),
            "compile_wall_clock_s": _wrap(236.068201),
            "break_even_remaining_actions": _wrap(45.748641),
            "arc_registry_modified": _wrap(False),
        },
        5216: {
            "experiment_id": "exp5216-arc-frontier-continuity-landmark-decomposition-v477",
            "honest_verdict": "complete: frontier continuity plus landmark decomposition did not bank a new reproduction-gated level above the registry precheck in this bounded pilot.",
            "duplicate_registry_precheck_passed": True,
            "live_path_integration_attempted": True,
            "solve_provenance": "development_proxy",
            "offline_ground_truth_bfs": False,
            "read_game_source": False,
            "new_levels_banked": [],
            "reproducible_total_levels_delta": 0,
            "frontier_continuity_lift": {"bp35": {"reproduced_level_delta": 0}},
            "landmark_decomposition_lift": {"cd82": {"reproduced_level_delta": 0}},
            "orphan_lint_result": "pass: OK: all solver-like ARC modules are reachable",
        },
        5217: {
            "experiment_id": "exp5217-hardware-continuity-v477",
            "honest_verdict": "complete_hardware_continuity_v477_kv260:reachable_gatemate:blocked_gatemate_dirtyjtag_idcode_unresolved_v477_narrowed_cable_or_port_polarfire:reachable_no_speedup_claim",
            "kv260_status": "reachable + hash-verified smoke workload_hash=48683aa401c6f9f1e6ca471acd44cfb0a883ef0639fcf3d8baf09aee90f4df32",
            "polarfire_status": {
                "summary": "reachable + hash-verified smoke workload_hash=9172e92a9403761da21f332f0327e4ab284557c88fae668c8815707f7feea967",
                "reachable": True,
            },
            "gatemate_status": {
                "status": "blocked_gatemate_dirtyjtag_idcode_unresolved_v477",
                "narrowed_to": "cable_or_port",
                "leading_untested_hypothesis": "physical_board",
            },
            "gatemate_diagnostic_narrowed_to": "cable_or_port",
            "new_diagnostic_angles_tried_this_milestone": [
                {"angle": "debug_level_raw_idcode_capture"}
            ],
            "hardware_speedup_claimed": False,
        },
        5218: {
            "experiment_id": "exp5218-verifier-authenticity-remediation-apply-v477",
            "honest_verdict": _wrap(
                "complete: dishonest-naming risk reduced by registry flags; modules remain headline-ineligible until real verification"
            ),
            "remediation_applied": _wrap(True),
            "remediated_modules": _wrap(
                [
                    "python/carnot/verify/and_composition_verifier.py",
                    "python/carnot/verify/claim_isolation_uncertainty_router.py",
                ]
            ),
            "remediation_type": _wrap("registry_flag"),
            "headline_ineligible_until_real_verification": _wrap(True),
            "no_research_conductor_change": _wrap(True),
        },
        5219: {
            "experiment_id": "exp5219-capstone-v477",
            "honest_verdict": "complete: v477 closed with GAP-1 building after blocked registry promotion, GAP-4 blocked by flagged/protocol validation, MMLU hidden-state path retired, self-learning memory created, zero ARC levels banked, hardware reachability maintained with no speedup claim, and flagged artifacts excluded.",
            "gap1_final_status": _wrap("building"),
            "gap4_final_status": _wrap("blocked"),
            "hidden_state_path_decision": _wrap("retire_mmlu_path"),
            "continuous_self_learning_satisfied": _wrap(True),
            "new_levels_banked": _wrap([]),
            "reproducible_total_levels_delta": _wrap(0),
            "hardware_final_state": _wrap(
                "KV260=reachable; PolarFire=reachable; GateMate=cable_or_port; no speedup claim"
            ),
            "flagged_adversarial_artifacts_excluded": _wrap(True),
            "status_decisions": {
                "gap1": "Exp5210 blocked before registry promotion; GAP-1 remains building",
                "gap4": "Exp5211 expansion and Exp5212 validation artifacts are flagged",
                "arc": "new_levels_banked=[], reproducible_total_levels_delta=0",
                "authenticity": "Exp5218 remediation applied and headline-ineligible flags preserved",
            },
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
            "2026.07.478" if active else "2026.07.477",
            5220 if active else 5207,
            5232 if active else 5219,
        ),
        encoding="utf-8",
    )
    if next_file:
        (root / "research-roadmap-next.yaml").write_text(
            _roadmap("2026.07.478", 5220, 5232), encoding="utf-8"
        )
    (root / "openspec/change-proposals" / "research-roadmap-vNEXT.md").write_text(
        "# Research Roadmap vNEXT: 2026.07.478\n\nStatus: proposed for activation after milestone `2026.07.477`\n",
        encoding="utf-8",
    )
    (root / "ops" / "conductor-log.md").write_text(
        "| 2026-07-04 08:59 UTC | Milestone 2026.07.478 activated | OK | 13 tasks queued |\n",
        encoding="utf-8",
    )
    (root / "ops" / "status.md").write_text("| status fixture |\n", encoding="utf-8")
    (root / "ops" / "changelog.md").write_text("- changelog fixture\n", encoding="utf-8")
    (root / "ops" / "exclusion_manifest.yaml").write_text("retired_extras: []\n", encoding="utf-8")
    (root / "scripts" / "research_conductor.py").write_text(
        "# conductor fixture\n", encoding="utf-8"
    )
    payloads = _payloads()
    for source in mod.UPSTREAM_SOURCES:
        if source.experiment_number not in omit:
            _write_json(root / source.relative_path, payloads[source.experiment_number])
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


def test_req_report_5220_spec_declares_archive_contract() -> None:
    """REQ-REPORT-5220: OpenSpec anchors the .477 archive and .478 activation contract."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    section = spec[spec.index("REQ-REPORT-5220") : spec.index("REQ-REPORT-5208")]

    for marker in (
        "REQ-REPORT-5220",
        "SCENARIO-REPORT-5220",
        "SCENARIO-REPORT-5220-BLOCKED-PRECONDITION",
        str(mod.RESULT_RELATIVE_PATH),
        "v477_summary",
        "validation_commands_run",
        "aggregation_from_upstream_artifacts",
    ):
        assert marker in section
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert f"`{field}`" in section
        assert principle in section


def test_scenario_report_5220_already_active_preserves_precise_v477_truth(tmp_path: Path) -> None:
    """SCENARIO-REPORT-5220: already-active .478 handoff records exact .477 facts."""

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
    assert artifact["archived_milestone"] == "2026.07.477"
    assert artifact["milestone"] == "2026.07.478"
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
    assert ".478 activated" in artifact["honest_verdict"]["value"]

    summary = artifact["v477_summary"]["value"]
    for required in (
        "exp5209",
        "heldout pass@2 0.189584",
        "best_subset_stable=false",
        "exp5210",
        "principle-wrapped",
        "exp5211",
        "120-row GAP-4 candidate pool",
        "adversarially flagged",
        "exp5212",
        "n_scored=0",
        "missing_protocol_pass2_fields",
        "six-discordant-win",
        "exp5214",
        "one promotion and one rollback",
        "results/verifier_memory_v477.json",
        "exp5216",
        "zero reproduction-gated ARC levels",
        "exp5217",
        "KV260=reachable",
        "PolarFire=reachable",
        "GateMate=cable_or_port",
        "exp5219",
        "GAP-1 building",
        "GAP-4 blocked",
        "flagged artifacts excluded",
        "PAW not viable",
        "verifier-authenticity registry flags applied",
        "MMLU-Pro hidden-state path retired",
    ):
        assert required in summary

    commands = artifact["validation_commands_run"]["value"]
    assert [row["passed"] for row in commands] == [True, True]
    assert "exclusion_manifest_lint.py" in commands[0]["command"]
    assert "validate_prior_failures.py" in commands[1]["command"]
    assert artifact["failed_preconditions"] == []
    assert artifact["reproducibility_checksum"] == mod.payload_checksum(artifact)


def test_scenario_report_5220_copies_next_roadmap_exactly_when_present(tmp_path: Path) -> None:
    """SCENARIO-REPORT-5220: activation copies research-roadmap-next.yaml exactly."""

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
    assert artifact["archived_research_roadmap_yaml"]["milestone"] == "2026.07.477"
    assert artifact["archived_research_roadmap_yaml"]["task_count"] == 13
    assert artifact["research_roadmap_yaml_activated"]["value"] is True


def test_scenario_report_5220_blocked_preconditions_are_visible(tmp_path: Path) -> None:
    """SCENARIO-REPORT-5220-BLOCKED-PRECONDITION: dirty inputs block clean claims."""

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
        root=_make_repo(tmp_path, active=True, omit={5219}),
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
    assert "missing_artifact_exp5219" in artifact["failed_preconditions"]
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


def test_req_report_5220_run_writes_valid_artifact(tmp_path: Path) -> None:
    """REQ-REPORT-5220: run writes the terminal JSON artifact."""

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


def test_req_report_5220_helper_edges_and_default_validation_path(tmp_path: Path) -> None:
    """REQ-REPORT-5220: helper edges keep blocked artifacts explicit instead of crashing."""

    assert mod.value_of({"value": {"value": "nested"}}) == "nested"
    assert mod._number(True) is None
    assert mod._number("not-a-number") is None
    assert mod._int("bad") == 0
    assert mod._bool("true") is None
    assert mod._list("not-list") == []
    assert mod.file_sha256(tmp_path / "missing") is None
    assert mod._roadmap_data("bad: [") == {}
    assert mod._task_ids({"tasks": "not-list"}) == []
    assert mod._command_label("") == "unknown_command"
    assert mod._command_label("custom --flag") == "custom"
    assert mod.exclusion_manifest_clean([]) is False
    assert mod._fmt_float(None) == "unknown"
    assert mod._fmt_p(None) == "unknown"
    assert mod._status_summary({"summary": "reachable smoke"}, "", "", "") == "reachable smoke"
    assert mod._status_summary({"status": "blocked_idcode"}, "", "", "") == "blocked_idcode"
    assert mod._status_summary({"reachable": True}, "", "", "") == "reachable"
    assert mod._status_summary("", "kv260:reachable", "kv260:reachable", "reachable") == "reachable"
    assert (
        mod._flag_kinds([{"kind": "TAUTOLOGY"}, {"kind": "DURATION_TOO_SHORT"}])
        == "DURATION_TOO_SHORT/TAUTOLOGY"
    )
    assert mod._flag_kinds("not-list") == "none"

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
        missing_artifacts=["missing_artifact_exp5219"],
        roadmap_activation={"activated": False},
        validation=[],
        conductor_clean=False,
        vnext_present=False,
    )
    assert failures == [
        "missing_artifact_exp5219",
        "research_roadmap_yaml_not_active_for_478",
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

    blocked_summary = mod.build_v477_summary({5209: {"heldout_pass_at_2_mean": "bad"}, 5219: {}})
    assert "heldout pass@2 unknown" in blocked_summary
    assert "p=unknown" in blocked_summary


def test_req_report_5220_validation_rejects_schema_and_type_errors(tmp_path: Path) -> None:
    """REQ-REPORT-5220: artifact validation rejects schema drift and overclaims."""

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
        mod.validate_artifact(artifact | {"milestone": "2026.07.477"})
    with pytest.raises(ValueError, match="field principle"):
        mod.validate_artifact(
            artifact
            | {"field_principles": artifact["field_principles"] | {"honest_verdict": "wrong"}}
        )
    with pytest.raises(ValueError, match="principle-wrapped"):
        mod.validate_artifact(artifact | {"v477_summary": "not-wrapped"})
    with pytest.raises(ValueError, match="principle mismatch"):
        mod.validate_artifact(artifact | {"v477_summary": _wrap("summary")})
    with pytest.raises(ValueError, match="missing value"):
        mod.validate_artifact(
            artifact | {"v477_summary": {"principle": mod.FIELD_PRINCIPLES["v477_summary"]}}
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
