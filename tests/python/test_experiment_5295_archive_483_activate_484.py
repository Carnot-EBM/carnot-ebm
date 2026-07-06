"""Tests for Exp 5295 archive .483 / .484 activation artifact.

Spec refs: REQ-REPORT-5295, SCENARIO-REPORT-5295,
SCENARIO-REPORT-5295-BLOCKED-CLOSEOUT.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import yaml

from carnot import experiment_5295_archive_483_activate_484 as mod


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "research-reporting" / "spec.md"
RESULT_PATH = REPO / mod.RESULT_RELATIVE_PATH


def _wrap(value: Any, principle: str = "fixture principle") -> dict[str, Any]:
    return {"principle": principle, "value": value}


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _roadmap(milestone: str, start: int = 5295, stop: int = 5306) -> str:
    return yaml.safe_dump(
        {
            "milestone": milestone,
            "milestone_title": f"fixture {milestone}",
            "milestone_doc": "openspec/change-proposals/research-roadmap-vNEXT.md",
            "tasks": [
                {
                    "id": f"exp{idx}-fixture-v{milestone.split('.')[-1]}",
                    "milestone": milestone,
                    "deliverable": f"results/experiment_{idx}_fixture.json",
                    "title": f"fixture task {idx}",
                    "agent_type": "codex",
                    "model": "gpt-5.5",
                    "prompt": "fixture REQUIRED ARTIFACT FIELDS:\n- honest_verdict",
                    "prior_failures": [
                        {
                            "experiment_id": f"exp{idx - 1}",
                            "verdict": "complete: fixture",
                            "addressed_by": "fixture",
                            "retire_if_same_verdict": True,
                        }
                    ],
                }
                for idx in range(start, stop + 1)
            ],
        },
        sort_keys=False,
    )


def _capstone_payload() -> dict[str, Any]:
    return {
        "experiment_id": "exp5294-capstone-v483",
        "honest_verdict": _wrap(
            "complete: .483 closed with deterministic claim/coherence and trace fixtures "
            "ready, SOTA runtime/offload blocked, SOTA quality tasks gate-skipped, "
            "memory attribution and coherence dosing positive, low-order curriculum null, "
            "p-bit/CDCL aggregate positive with misleading-class harm, and hardware "
            "reachability-only with no speedup."
        ),
        "tasks_summarized": _wrap(
            {
                "expected_count": 12,
                "loadable_count": 12,
                "missing_artifacts": [],
                "milestone_synthesized": True,
                "by_classification": {
                    "clean_positive": 6,
                    "clean_null": 1,
                    "blocked_precondition": 2,
                    "gated_skip": 2,
                    "mixed_positive_with_harmful_class": 1,
                },
            }
        ),
        "clean_positive_findings": _wrap(
            [
                {"id": "claim_level_coherence_fixture_ready", "coherence_fixture_ready": True},
                {"id": "compilable_trace_dsl_fixture_ready", "trace_dsl_ready": True},
                {
                    "id": "memory_operation_attribution_ready",
                    "attribution_coverage": {"attributed_cases": 7, "coverage_rate": 1.0},
                },
                {
                    "id": "memory_assisted_coherence_dosing_positive",
                    "full_verifier_calls_avoided": {
                        "vs_always_full": 4,
                        "rate_vs_always_full": 0.571429,
                    },
                    "unsafe_false_accepts": {"count": 0},
                },
                {
                    "id": "pbit_cdcl_aggregate_guidance_positive",
                    "conflicts_saved": {
                        "aggregate": 2,
                        "by_class": {
                            "aligned_factor_sat": 3,
                            "misleading_factor_sat": -1,
                            "neutral_factor_sat": 0,
                        },
                    },
                    "correctness_preserved": True,
                },
            ]
        ),
        "null_or_harmful_findings": _wrap(
            [
                {
                    "id": "low_order_curriculum_clean_null",
                    "certificate_success_by_order": {
                        "success_advantage_over_shuffled": 0.0,
                        "helped_certificate_success": False,
                    },
                },
                {
                    "id": "pbit_cdcl_misleading_assumption_harm",
                    "harmful_classes": ["misleading_factor_sat"],
                    "conflicts_saved": {
                        "aggregate": 2,
                        "by_class": {"misleading_factor_sat": -1},
                    },
                },
            ]
        ),
        "gated_or_blocked_findings": _wrap(
            [
                {"id": "sota_runtime_offload_blocked", "sota_offload_ready": False},
                {"id": "claim_level_sota_pilot_gated_skip"},
                {"id": "trace_dsl_sota_extraction_gated_skip"},
                {
                    "id": "hardware_reachability_blocked_no_speedup",
                    "hardware_speedup_claimed": False,
                    "finding": "reachability/status only with no speedup",
                },
            ]
        ),
        "retirements_or_exclusions": _wrap({"manifest_updated": True}),
        "ops_docs_updated": _wrap(
            {
                "ops_changelog": False,
                "ops_status": False,
                "traceability": False,
                "reason": "stop_when_done_reconciler_deferred_ops_docs",
            }
        ),
    }


def _upstream_payload(number: int) -> dict[str, Any]:
    payloads: dict[int, dict[str, Any]] = {
        5282: {"honest_verdict": _wrap("complete: .482 archived and .483 activation-ready")},
        5283: {"honest_verdict": _wrap("complete: 3 new actionable findings appended")},
        5284: {
            "honest_verdict": _wrap(
                "blocked_preconditions: sota_offload_ready=false "
                "flagship_moe:blocked_no_gpu_offload_evidence"
            ),
            "sota_offload_ready": False,
        },
        5285: {"honest_verdict": _wrap("complete: coherence fixture ready"), "coherence_fixture_ready": True},
        5286: {"honest_verdict": "blocked_gate_check_failed", "blocked_at_layer": "conductor_pre_gate"},
        5287: {"honest_verdict": _wrap("complete: trace DSL fixture ready"), "trace_dsl_ready": True},
        5288: {"honest_verdict": "blocked_gate_check_failed", "blocked_at_layer": "conductor_pre_gate"},
        5289: {"honest_verdict": _wrap("complete: attribution ready"), "memory_attribution_ready": True},
        5290: {"honest_verdict": _wrap("complete: coherence dosing positive"), "coherence_dose_positive": True},
        5291: {"honest_verdict": _wrap("complete: clean null"), "low_order_curriculum_ready": True},
        5292: {
            "honest_verdict": _wrap("complete: aggregate positive with misleading harm"),
            "pbit_cdcl_guidance_positive": True,
            "hardware_speedup_claimed": _wrap(False),
        },
        5293: {
            "honest_verdict": _wrap("blocked_board_reachability: no_speedup_claim"),
            "hardware_speedup_claimed": _wrap(False),
            "hardware_evidence_level": _wrap("reachability_status_receipt_only"),
        },
        5294: _capstone_payload(),
    }
    return payloads[number]


def _research_complete(has_v483: bool = True) -> str:
    milestones: list[dict[str, Any]] = [
        {
            "id": "2026.07.482",
            "title": "prior fixture",
            "completed": "2026-07-05",
            "finding": "fixture",
            "tasks": [],
        }
    ]
    if has_v483:
        milestones.append(
            {
                "id": "2026.07.483",
                "title": "Claim-Level Verification, Memory Attribution, and Solver-Hardware Guidance",
                "completed": "2026-07-06",
                "finding": "See conductor log for per-experiment results.",
                "tasks": [],
            }
        )
    return yaml.safe_dump({"milestones": milestones}, sort_keys=False)


def _make_repo(
    root: Path,
    *,
    has_v483_complete: bool = True,
    active_milestone: str = "2026.07.484",
    vnext_milestone: str = "2026.07.484",
    next_file: bool = False,
    capstone: dict[str, Any] | None = None,
) -> Path:
    for source in mod.UPSTREAM_SOURCES:
        _write_json(root / source.relative_path, _upstream_payload(source.experiment_number))
    if capstone is not None:
        _write_json(root / mod.CAPSTONE_RELATIVE_PATH, capstone)
    (root / "research-complete.yaml").write_text(
        _research_complete(has_v483_complete), encoding="utf-8"
    )
    (root / "research-roadmap.yaml").write_text(_roadmap(active_milestone), encoding="utf-8")
    if next_file:
        (root / "research-roadmap-next.yaml").write_text(_roadmap("2026.07.484"), encoding="utf-8")
    (root / "openspec/change-proposals").mkdir(parents=True, exist_ok=True)
    (root / "openspec/change-proposals/research-roadmap-vNEXT.md").write_text(
        f"# Research Roadmap vNEXT: Milestone {vnext_milestone}\nMilestone: {vnext_milestone}\n",
        encoding="utf-8",
    )
    (root / "ops").mkdir(parents=True, exist_ok=True)
    (root / "ops/status.md").write_text(
        "Milestone 2026.07.483 operational retrospective timing-assembly mismatch\n",
        encoding="utf-8",
    )
    (root / "ops/changelog.md").write_text(
        "2026.07.483 TIMING INTEGRITY MISMATCH and timing-accounting mismatch\n",
        encoding="utf-8",
    )
    (root / "ops/conductor-log.md").write_text(
        "Milestone 2026.07.483 activated\nMilestone 2026.07.484 activated\n",
        encoding="utf-8",
    )
    (root / "ops/exclusion_manifest.yaml").write_text(
        "retired_extras:\n- id: exp5284_sota_offload_cpu_only_path_retired_v483\n",
        encoding="utf-8",
    )
    _write_json(
        root / "results/operational_retro_2026_07_483.json",
        {
            "total_wall_time_minutes": 0,
            "experiments_completed": 0,
            "compute_bound_experiments_count": 0,
            "slowest_experiments": [],
            "gpu_idle_on_compute_bound_tasks": None,
            "bottlenecks_identified": ["Timing assembly mismatch"],
        },
    )
    (root / "_bmad").mkdir(parents=True, exist_ok=True)
    for relative in ("prd.md", "architecture.md", "traceability.md"):
        (root / "_bmad" / relative).write_text("fixture\n", encoding="utf-8")
    for relative in ("CLAUDE.md", "CODEX.md", "research-program.md"):
        (root / relative).write_text("fixture\n", encoding="utf-8")
    (root / "scripts").mkdir(parents=True, exist_ok=True)
    (root / "scripts/research_conductor.py").write_text("# fixture\n", encoding="utf-8")
    return root


def _passed_commands() -> list[mod.CommandResult]:
    return [
        mod.CommandResult((".venv/bin/python", "scripts/roadmap_schema.py", "research-roadmap.yaml"), 0, "", ""),
        mod.CommandResult((".venv/bin/python", "scripts/validate_prior_failures.py", "research-roadmap.yaml"), 0, "clean", ""),
        mod.CommandResult((".venv/bin/python", "scripts/audit_roadmap_gates.py", "research-roadmap.yaml", "--complete", "research-complete.yaml"), 0, "clean", ""),
        mod.CommandResult((".venv/bin/python", "scripts/exclusion_manifest_lint.py", "research-roadmap.yaml"), 0, "clean", ""),
    ]


def _failed_commands() -> list[mod.CommandResult]:
    return [
        mod.CommandResult(
            (".venv/bin/python", "scripts/roadmap_schema.py", "research-roadmap.yaml"),
            1,
            "schema violation",
            "",
        )
    ]


def test_req_report_5295_spec_declares_archive_contract() -> None:
    """REQ-REPORT-5295: OpenSpec anchors the .483 archive and .484 activation contract."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    section = spec[spec.index("REQ-REPORT-5295") : spec.index("REQ-REPORT-5283")]

    for marker in (
        "REQ-REPORT-5295",
        "SCENARIO-REPORT-5295",
        "SCENARIO-REPORT-5295-BLOCKED-CLOSEOUT",
        str(mod.RESULT_RELATIVE_PATH),
        "aggregation_from_upstream_artifacts",
        "roadmap_activation_check.activated=false",
        "timing-accounting mismatch",
        "commands_run",
    ):
        assert marker in section
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert f"`{field}`" in section
        assert principle in section


def test_scenario_report_5295_already_active_records_no_overwrite(tmp_path: Path) -> None:
    """SCENARIO-REPORT-5295: already-active .484 produces a complete no-overwrite artifact."""

    root = _make_repo(tmp_path)
    before = (root / "research-roadmap.yaml").read_text(encoding="utf-8")
    artifact = mod.build_artifact(
        root=root,
        run_date="20260706",
        duration_s=1.25,
        validation_results=_passed_commands(),
        update_research_complete=False,
    )

    mod.validate_artifact(artifact)
    assert (root / "research-roadmap.yaml").read_text(encoding="utf-8") == before
    assert artifact["milestone_archived"] is True
    assert artifact["activation_ready"] is True
    assert artifact["honest_verdict"]["value"].startswith("complete:")
    assert ".483 archived" in artifact["honest_verdict"]["value"]
    assert ".484 activation-ready" in artifact["honest_verdict"]["value"]
    assert artifact["inference_substrate"]["value"] == mod.INFERENCE_SUBSTRATE
    assert artifact["ops_docs_updated"]["value"] is False
    assert artifact["research_complete_updated"]["value"] is False
    assert artifact["exclusions_checked"]["value"] is True
    assert artifact["roadmap_activation_check"]["activated"] is False
    assert artifact["roadmap_activation_check"]["active_roadmap_already_484"] is True
    assert artifact["roadmap_activation_check"]["roadmap_next_present"] is False
    assert artifact["roadmap_activation_check"]["roadmap_next_absence_handled_by"] == "active_roadmap_already_484"
    assert [row["passed"] for row in artifact["commands_run"]] == [True, True, True, True]
    assert artifact["closeout_facts"]["coherence_fixture_ready"] is True
    assert artifact["closeout_facts"]["trace_fixture_ready"] is True
    assert artifact["closeout_facts"]["sota_runtime_offload_blocked"] is True
    assert artifact["closeout_facts"]["sota_quality_gate_skipped"] is True
    assert artifact["closeout_facts"]["memory_attribution_positive"] is True
    assert artifact["closeout_facts"]["coherence_dosing_positive"] is True
    assert artifact["closeout_facts"]["low_order_curriculum_null"] is True
    assert artifact["closeout_facts"]["pbit_cdcl_mixed_positive_with_harm"] is True
    assert artifact["closeout_facts"]["hardware_reachability_no_speedup"] is True
    assert artifact["closeout_facts"]["timing_retro_accounting_mismatch"] is True
    assert artifact["failed_preconditions"] == []
    assert artifact["reproducibility_checksum"] == mod.payload_checksum(artifact)


def test_scenario_report_5295_blocked_closeout_keeps_failures_visible(tmp_path: Path) -> None:
    """SCENARIO-REPORT-5295-BLOCKED-CLOSEOUT: contradictory evidence blocks readiness."""

    bad_capstone = _capstone_payload() | {
        "null_or_harmful_findings": _wrap([]),
        "gated_or_blocked_findings": _wrap([]),
    }
    artifact = mod.build_artifact(
        root=_make_repo(
            tmp_path,
            has_v483_complete=False,
            active_milestone="2026.07.483",
            vnext_milestone="2026.07.483",
            capstone=bad_capstone,
        ),
        run_date="20260706",
        duration_s=0.5,
        validation_results=_failed_commands(),
        update_research_complete=False,
    )

    mod.validate_artifact(artifact)
    assert artifact["milestone_archived"] is False
    assert artifact["activation_ready"] is False
    assert artifact["honest_verdict"]["value"].startswith("blocked_")
    assert artifact["roadmap_activation_check"]["activated"] is False
    assert "closeout_sota_runtime_offload_blocked_expected_True_observed_False" in artifact["failed_preconditions"]
    assert "closeout_low_order_curriculum_null_expected_True_observed_False" in artifact["failed_preconditions"]
    assert "closeout_pbit_cdcl_mixed_positive_with_harm_expected_True_observed_False" in artifact["failed_preconditions"]
    assert "research_complete_missing_2026.07.483" in artifact["failed_preconditions"]
    assert "vnext_missing_2026.07.484" in artifact["failed_preconditions"]
    assert "active_or_next_roadmap_not_ready_for_484" in artifact["failed_preconditions"]
    assert "validation_failed_scripts/roadmap_schema.py" in artifact["failed_preconditions"]
    assert artifact["exclusions_checked"]["value"] is False


def test_req_report_5295_can_append_missing_research_complete_and_run(tmp_path: Path) -> None:
    """REQ-REPORT-5295: missing research-complete .483 entry can be appended once."""

    root = _make_repo(tmp_path, has_v483_complete=False, active_milestone="2026.07.483", next_file=True)
    out = mod.run(
        root=root,
        run_date="20260706",
        duration_s=0.25,
        validation_results=_passed_commands(),
        update_research_complete=True,
    )

    saved = json.loads(out.read_text(encoding="utf-8"))
    mod.validate_artifact(saved)
    assert saved["research_complete_updated"]["value"] is True
    assert saved["milestone_archived"] is True
    complete = yaml.safe_load((root / "research-complete.yaml").read_text(encoding="utf-8"))
    assert [row["id"] for row in complete["milestones"]][-1] == "2026.07.483"
    assert saved["roadmap_activation_check"]["roadmap_next_present"] is True
    assert saved["roadmap_activation_check"]["roadmap_next_milestone"] == "2026.07.484"
    assert saved["reproducibility_checksum"] == mod.payload_checksum(saved)


def test_req_report_5295_repository_artifact_matches_schema() -> None:
    """REQ-REPORT-5295: checked-in deliverable is a valid archive artifact."""

    artifact = json.loads(RESULT_PATH.read_text(encoding="utf-8"))

    mod.validate_artifact(artifact)
    assert artifact["experiment_id"] == mod.EXPERIMENT_ID
    assert artifact["milestone_archived"] is True
    assert artifact["activation_ready"] is True
    assert artifact["roadmap_activation_check"]["activated"] is False
    assert artifact["roadmap_activation_check"]["active_roadmap_already_484"] is True
    assert artifact["roadmap_activation_check"]["roadmap_next_present"] is False
    assert artifact["research_complete_updated"]["value"] is False
    assert artifact["exclusions_checked"]["value"] is True
    assert artifact["inference_substrate"]["value"] == mod.INFERENCE_SUBSTRATE
    assert artifact["honest_verdict"]["value"].startswith("complete:")


def test_req_report_5295_helper_edges_and_validation_guards(tmp_path: Path) -> None:
    """REQ-REPORT-5295: helpers fail closed instead of hiding schema drift."""

    root = _make_repo(tmp_path / "repo")
    artifact = mod.build_artifact(
        root=root,
        run_date="20260706",
        duration_s=1.0,
        validation_results=_passed_commands(),
        update_research_complete=False,
    )

    with pytest.raises(ValueError, match="missing required"):
        mod.validate_artifact({key: value for key, value in artifact.items() if key != "schema"})
    with pytest.raises(ValueError, match="schema"):
        mod.validate_artifact(artifact | {"schema": "wrong"})
    with pytest.raises(ValueError, match="field_principles"):
        mod.validate_artifact(artifact | {"field_principles": {}})
    with pytest.raises(ValueError, match="principle mismatch"):
        mod.validate_artifact(artifact | {"honest_verdict": _wrap("complete: ok")})
    with pytest.raises(ValueError, match="missing value"):
        mod.validate_artifact(
            artifact | {"honest_verdict": {"principle": mod.FIELD_PRINCIPLES["honest_verdict"]}}
        )
    with pytest.raises(ValueError, match="honest_verdict"):
        mod.validate_artifact(
            artifact
            | {
                "honest_verdict": {
                    "principle": mod.FIELD_PRINCIPLES["honest_verdict"],
                    "value": "done",
                }
            }
        )
    with pytest.raises(ValueError, match="inference_substrate"):
        mod.validate_artifact(
            artifact
            | {
                "inference_substrate": {
                    "principle": mod.FIELD_PRINCIPLES["inference_substrate"],
                    "value": "cached_fixture_replay_no_llm",
                }
            }
        )
    with pytest.raises(ValueError, match="milestone_archived"):
        mod.validate_artifact(artifact | {"milestone_archived": "true"})
    with pytest.raises(ValueError, match="milestone_archived"):
        mod.validate_artifact(artifact | {"milestone_archived_principle": "wrong"})
    with pytest.raises(ValueError, match="activation_ready"):
        mod.validate_artifact(artifact | {"activation_ready": "true"})
    with pytest.raises(ValueError, match="activation_ready"):
        mod.validate_artifact(artifact | {"activation_ready_principle": "wrong"})
    with pytest.raises(ValueError, match="ops_docs_updated"):
        mod.validate_artifact(
            artifact
            | {
                "ops_docs_updated": {
                    "principle": mod.FIELD_PRINCIPLES["ops_docs_updated"],
                    "value": True,
                }
            }
        )
    with pytest.raises(ValueError, match="research_complete_updated"):
        mod.validate_artifact(
            artifact
            | {
                "research_complete_updated": {
                    "principle": mod.FIELD_PRINCIPLES["research_complete_updated"],
                    "value": "false",
                }
            }
        )
    with pytest.raises(ValueError, match="exclusions_checked"):
        mod.validate_artifact(
            artifact
            | {
                "exclusions_checked": {
                    "principle": mod.FIELD_PRINCIPLES["exclusions_checked"],
                    "value": "true",
                }
            }
        )
    with pytest.raises(ValueError, match="roadmap_activation_check"):
        mod.validate_artifact(artifact | {"roadmap_activation_check": {"activated": False}})
    with pytest.raises(ValueError, match="roadmap_activation_check"):
        mod.validate_artifact(
            artifact | {"roadmap_activation_check": artifact["roadmap_activation_check"] | {"activated": True}}
        )
    with pytest.raises(ValueError, match="commands_run"):
        mod.validate_artifact(artifact | {"commands_run": []})
    with pytest.raises(ValueError, match="commands_run"):
        mod.validate_artifact(artifact | {"commands_run": [{"command": "x"}]})
    with pytest.raises(ValueError, match="reproducibility_checksum"):
        mod.validate_artifact(artifact | {"reproducibility_checksum": "sha256:bad"})

    assert mod.value_of(_wrap("x")) == "x"
    assert mod.value_of("x") == "x"
    assert mod.text_sha256("abc").startswith("sha256:")
    assert mod.closeout_fact_failures({}) == ["capstone_artifact_missing_or_unloadable"]
    assert mod._roadmap_data("bad: [") == {}
    assert mod._task_ids({"tasks": "not-list"}) == []
    assert mod._milestones([]) == []
    assert mod._command_label("") == "unknown_command"
    assert mod._command_label("python scripts/roadmap_schema.py research-roadmap.yaml") == "scripts/roadmap_schema.py"
    assert mod._command_label("python scripts/validate_prior_failures.py research-roadmap.yaml") == "scripts/validate_prior_failures.py"
    assert mod._command_label("custom --flag") == "custom"
    assert mod._commands_passed([]) is False
    assert mod._commands_passed(mod.commands_run_rows(_passed_commands())) is True

    malformed = tmp_path / "bad.json"
    malformed.write_text("{bad", encoding="utf-8")
    assert mod.read_json_mapping(malformed)[1]["error"] == "malformed_json"
    array_json = tmp_path / "array.json"
    array_json.write_text("[]", encoding="utf-8")
    assert mod.read_json_mapping(array_json)[1]["error"] == "not_json_object"
    missing = tmp_path / "missing.json"
    assert mod.read_json_mapping(missing)[1]["error"] == "missing"

    missing_complete = tmp_path / "missing-complete"
    missing_complete.mkdir()
    assert mod.research_complete_milestone_count(missing_complete) == 0
    malformed_complete = tmp_path / "malformed-complete"
    malformed_complete.mkdir()
    (malformed_complete / "research-complete.yaml").write_text("bad: [", encoding="utf-8")
    assert mod.research_complete_milestone_count(malformed_complete) == 0
    assert mod.append_research_complete_milestone(malformed_complete) is True
    assert mod.append_research_complete_milestone(malformed_complete) is False

    nondict_complete = tmp_path / "nondict-complete"
    nondict_complete.mkdir()
    (nondict_complete / "research-complete.yaml").write_text("[]\n", encoding="utf-8")
    assert mod.append_research_complete_milestone(nondict_complete) is True
    nonlist_complete = tmp_path / "nonlist-complete"
    nonlist_complete.mkdir()
    (nonlist_complete / "research-complete.yaml").write_text("milestones: nope\n", encoding="utf-8")
    assert mod.append_research_complete_milestone(nonlist_complete) is True

    failures = mod.failed_preconditions(
        closeout_failures=[],
        research_complete={"has_2026_07_483_after": False},
        roadmap={"vnext_names_2026_07_484": True, "activation_ready_without_overwrite": True},
        commands=[],
    )
    assert failures == ["research_complete_missing_2026.07.483", "validation_commands_missing"]

    no_scripts = tmp_path / "no-scripts"
    no_scripts.mkdir()
    assert mod.validation_commands(no_scripts) == []
    assert mod.run_validation_commands(no_scripts) == []

    script_root = _make_repo(tmp_path / "script-root")
    (script_root / "scripts/roadmap_schema.py").write_text("print('schema ok')\n", encoding="utf-8")
    (script_root / "scripts/validate_prior_failures.py").write_text("print('prior ok')\n", encoding="utf-8")
    (script_root / "scripts/audit_roadmap_gates.py").write_text("print('audit ok')\n", encoding="utf-8")
    (script_root / "scripts/exclusion_manifest_lint.py").write_text("print('exclusion ok')\n", encoding="utf-8")
    commands = mod.validation_commands(script_root)
    assert len(commands) == 4
    results = mod.run_validation_commands(script_root)
    assert [row["passed"] for row in mod.commands_run_rows(results)] == [True, True, True, True]
