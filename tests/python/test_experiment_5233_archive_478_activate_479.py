"""Tests for Exp 5233 archive .478 / activate .479.

Spec refs: REQ-REPORT-5233, SCENARIO-REPORT-5233,
SCENARIO-REPORT-5233-BLOCKED-PRECONDITION.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import yaml

from carnot import experiment_5233_archive_478_activate_479 as mod


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
        5220: {
            "experiment_id": "exp5220-archive-477-activate-478",
            "honest_verdict": _wrap("complete: .477 archived and .478 activated"),
        },
        5221: {
            "experiment_id": "exp5221-sota-ingestion-v478",
            "honest_verdict": _wrap("complete: V478 SOTA refresh found no new actionable findings"),
        },
        5222: {
            "experiment_id": 5222,
            "honest_verdict": _wrap(
                "complete: GAP-1 registry promotion blocked_instability; exp5209 gate parsed from gap1_hardened_positive.value=True"
            ),
            "exp5209_gate_parsed_from_value": _wrap(True),
            "gap1_registry_decision": _wrap("blocked_instability"),
            "gap1_registry_promoted": _wrap(False),
            "subset_freeze_audit": {
                "best_subset_stable": False,
                "can_freeze_without_heldout_tuning": False,
                "top_subset_count": 9,
                "top_subset_fraction": 0.45,
            },
            "refuted_single_invariant_excluded": _wrap(True),
        },
        5223: {
            "experiment_id": 5223,
            "honest_verdict": "complete: old GAP-4 pool must be regenerated",
            "flagged_adversarial": True,
        },
        5224: {
            "experiment_id": 5224,
            "honest_verdict": "success: canonical GAP-4 pool usable for validation with n=120; no scale-validation claim run",
            "flagged_adversarial": True,
            "canonical_pool_n": 120,
            "regenerated_rows": 120,
            "repaired_rows": 0,
            "gap4_canonical_pool_usable": True,
            "protocol_fields_complete": True,
            "corrigendum_pending": [{"kind": "TAUTOLOGY", "severity": "critical"}],
        },
        5225: {
            "experiment_id": 5225,
            "honest_verdict": "complete: clean GAP-4 validation null decision with n=120, wins=0, losses=0, ties=120; min-six rule not crossed",
            "flagged_adversarial": True,
            "n_scored": 120,
            "exact_test_discordant_wins": 0,
            "exact_test_discordant_losses": 0,
            "ties": 120,
            "exact_test_passes_min6_rule": False,
            "canonical_pool_n": 120,
            "corrigendum_pending": [
                {"kind": "TAUTOLOGY", "severity": "critical"},
                {"kind": "IMPLAUSIBLE_PERFECT", "severity": "info"},
            ],
        },
        5226: {
            "experiment_id": "exp5226-veribmc-local-solver-feedback-pilot-v478",
            "honest_verdict": _wrap(
                "complete: clean null; solver feedback did not improve over baselines"
            ),
            "flagged_adversarial": True,
            "solver_feedback_uplift": _wrap(0.0),
            "duration_s": 59.42594,
            "corrigendum_pending": [
                {"kind": "DURATION_TOO_SHORT", "severity": "critical"},
                {"kind": "METHODOLOGY_MISSING", "severity": "warn"},
            ],
        },
        5227: {
            "experiment_id": 5227,
            "honest_verdict": _wrap(
                "complete: typed memory consumer-ready for exp5228 with 4 heads, promotions_2_rollbacks_4"
            ),
            "typed_memory_heads": _wrap(
                ["constraints", "provenance", "failures", "skills_rubrics"]
            ),
            "memory_entries_written": _wrap(6),
            "promotions": _wrap(2),
            "rollbacks": _wrap(4),
            "retention_check_passed": _wrap(True),
            "consumer_ready_path": _wrap("results/arc_rubric_setup_from_typed_memory_v478.json"),
            "memory_artifact_path": _wrap("results/typed_multihead_verifier_memory_v478.json"),
        },
        5228: {
            "experiment": "experiment_5228_arc_provenance_skill_rubric_gate_v478",
            "honest_verdict": "complete: ARC skill rubric usable; no exp5229 live patch is currently gated.",
            "arc_skill_rubric_usable": True,
            "recommended_live_patch_available": False,
            "recommended_patch_summary": "No credible exp5229 live-agent patch",
            "live_trace_count": 2,
            "scored_trace_count": 3,
            "rubric_path": "results/arc_skill_process_rubric_v478.json",
        },
        5229: {"experiment_id": 5229, "honest_verdict": "blocked_gate_check_failed"},
        5230: {
            "experiment_id": "exp5230-kan-milp-verifier-certificate-v478",
            "honest_verdict": _wrap(
                "success: tiny KAEM PWA/MILP certificate produced for bounded monotonicity and no unsafe decision"
            ),
            "kan_certificate_produced": _wrap(True),
            "solver_status": "optimal",
            "properties_checked": _wrap(
                [
                    {"property_id": "bounded_monotonicity", "verified": True},
                    {"property_id": "no_unsafe_decision", "verified": True},
                ]
            ),
            "bound_tightness": _wrap(0.07500000372529225),
            "certificate_path": _wrap(
                "results/experiment_5230_kan_milp_verifier_certificate_v478.json"
            ),
        },
        5231: {
            "experiment_id": "exp5231-hardware-continuity-pbit-boundary-v478",
            "honest_verdict": "complete_hardware_continuity_pbit_boundary_v478_kv260:reachable_polarfire:reachable_gatemate:blocked_physical_jtag_no_speedup",
            "kv260_reachable": True,
            "kv260_check_method": "ssh_only",
            "polarfire_reachable": True,
            "gatemate_status": "blocked_physical_jtag",
            "gatemate_check_note": "preserved v477 physical/JTAG block",
            "speedup_claimed": False,
            "pbit_boundary_plan_path": "docs/research-notes/experiment_5231_pbit_boundary_exchange_timing_ratio_plan.md",
        },
        5232: {
            "experiment_id": "exp5232-capstone-v478",
            "honest_verdict": "complete: v478 reconciled with GAP-1 blocked, GAP-4 blocked, VerIbmc solver-feedback blocked, typed memory satisfied, ARC delta 0, KAN certificate produced, hardware continuity/no-speedup recorded, and flagged/gated artifacts excluded.",
            "gap1_final_status": _wrap("blocked"),
            "gap4_final_status": _wrap("blocked"),
            "solver_feedback_status": _wrap("blocked"),
            "continuous_self_learning_satisfied": _wrap(True),
            "kan_certificate_status": _wrap("produced"),
            "hardware_status": _wrap(
                "KV260=reachable via ssh_only; PolarFire=reachable; GateMate=blocked_physical_jtag IDCODE=0xffffffff; no speedup claim"
            ),
            "arc_new_levels_banked": _wrap([]),
            "arc_reproducible_total_levels_delta": _wrap(0),
            "flagged_artifacts_excluded": _wrap(True),
            "excluded_from_headline_task_ids": [
                "exp5224-gap4-canonical-pool-builder-v478",
                "exp5225-gap4-clean-scale-validation-gated-v478",
                "exp5226-veribmc-local-solver-feedback-pilot-v478",
                "exp5229-arc-gated-live-levelup-from-rubric-v478",
            ],
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
            "2026.07.479" if active else "2026.07.478",
            5233 if active else 5220,
            5244 if active else 5232,
        ),
        encoding="utf-8",
    )
    if next_file:
        (root / "research-roadmap-next.yaml").write_text(
            _roadmap("2026.07.479", 5233, 5244), encoding="utf-8"
        )
    (root / "openspec/change-proposals" / "research-roadmap-vNEXT.md").write_text(
        "# Research Roadmap vNEXT: 2026.07.479\n\nStatus: proposed for activation after milestone `2026.07.478`\n",
        encoding="utf-8",
    )
    (root / "ops" / "conductor-log.md").write_text(
        "| 2026-07-04 17:38 UTC | Milestone 2026.07.479 activated | OK | 12 tasks queued |\n",
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


def test_req_report_5233_spec_declares_archive_contract() -> None:
    """REQ-REPORT-5233: OpenSpec anchors the .478 archive and .479 activation contract."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    section = spec[spec.index("REQ-REPORT-5233") : spec.index("REQ-REPORT-5162")]

    for marker in (
        "REQ-REPORT-5233",
        "SCENARIO-REPORT-5233",
        "SCENARIO-REPORT-5233-BLOCKED-PRECONDITION",
        str(mod.RESULT_RELATIVE_PATH),
        "v478_summary",
        "validation_commands_run",
        "aggregation_from_upstream_artifacts",
    ):
        assert marker in section
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert f"`{field}`" in section
        assert principle in section


def test_scenario_report_5233_already_active_preserves_precise_v478_truth(tmp_path: Path) -> None:
    """SCENARIO-REPORT-5233: already-active .479 handoff records exact .478 facts."""

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
    assert artifact["archived_milestone"] == "2026.07.478"
    assert artifact["milestone"] == "2026.07.479"
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
    assert ".479 activated" in artifact["honest_verdict"]["value"]

    summary = artifact["v478_summary"]["value"]
    for required in (
        "exp5222",
        "blocked_instability",
        "subset instability",
        "exp5224",
        "canonical GAP-4 pool n=120",
        "flagged_adversarial=true",
        "exp5225",
        "wins=0, losses=0, ties=120",
        "min-six rule not crossed",
        "exp5226",
        "solver feedback uplift 0.000000",
        "DURATION_TOO_SHORT/METHODOLOGY_MISSING",
        "exp5227",
        "consumer-ready typed multi-head memory",
        "constraints/provenance/failures/skills_rubrics",
        "exp5228",
        "ARC skill rubric usable",
        "no recommended live patch",
        "exp5230",
        "tiny KAEM PWA/MILP certificate",
        "exp5231",
        "KV260=reachable",
        "PolarFire=reachable",
        "GateMate=blocked_physical_jtag",
        "no speedup claim",
        "exp5232",
        "GAP-1 blocked",
        "GAP-4 blocked",
        "VerIbmc blocked",
        "ARC delta 0",
        "flagged/gated artifacts excluded",
    ):
        assert required in summary

    commands = artifact["validation_commands_run"]["value"]
    assert [row["passed"] for row in commands] == [True, True]
    assert "exclusion_manifest_lint.py" in commands[0]["command"]
    assert "validate_prior_failures.py" in commands[1]["command"]
    assert artifact["failed_preconditions"] == []
    assert artifact["reproducibility_checksum"] == mod.payload_checksum(artifact)


def test_scenario_report_5233_copies_next_roadmap_exactly_when_present(tmp_path: Path) -> None:
    """SCENARIO-REPORT-5233: activation copies research-roadmap-next.yaml exactly."""

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
    assert artifact["archived_research_roadmap_yaml"]["milestone"] == "2026.07.478"
    assert artifact["archived_research_roadmap_yaml"]["task_count"] == 13
    assert artifact["research_roadmap_yaml_activated"]["value"] is True


def test_scenario_report_5233_blocked_preconditions_are_visible(tmp_path: Path) -> None:
    """SCENARIO-REPORT-5233-BLOCKED-PRECONDITION: dirty inputs block clean claims."""

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
        root=_make_repo(tmp_path, active=True, omit={5232}),
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
    assert "missing_artifact_exp5232" in artifact["failed_preconditions"]
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


def test_req_report_5233_run_writes_valid_artifact(tmp_path: Path) -> None:
    """REQ-REPORT-5233: run writes the terminal JSON artifact."""

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


def test_req_report_5233_helper_edges_and_default_validation_path(tmp_path: Path) -> None:
    """REQ-REPORT-5233: helper edges keep blocked artifacts explicit instead of crashing."""

    assert mod._roadmap_data("bad: [") == {}
    assert mod._task_ids({"tasks": "not-list"}) == []
    assert mod._command_label("") == "unknown_command"
    assert mod._command_label("custom --flag") == "custom"
    assert mod.exclusion_manifest_clean([]) is False
    assert mod._fmt_float(None) == "unknown"
    assert mod._flag_kinds([{"kind": "TAUTOLOGY"}, {"kind": "DURATION_TOO_SHORT"}]) == (
        "DURATION_TOO_SHORT/TAUTOLOGY"
    )
    assert mod._flag_kinds("not-list") == "none"

    missing_root = tmp_path / "missing-root"
    missing_root.mkdir()
    assert mod.activate_roadmap(missing_root)["activated"] is False

    failures = mod._failed_preconditions(
        missing_artifacts=["missing_artifact_exp5232"],
        roadmap_activation={"activated": False},
        validation=[],
        conductor_clean=False,
        vnext_present=False,
    )
    assert failures == [
        "missing_artifact_exp5232",
        "research_roadmap_yaml_not_active_for_479",
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
    artifact = mod.build_artifact(root=root, conductor_untouched=True)
    assert artifact["clean_handoff"] is True
    assert artifact["tests_run"] == mod.DEFAULT_TESTS_RUN
    assert [row["passed"] for row in artifact["validation_checks"]] == [True, True]

    blocked_summary = mod.build_v478_summary({5224: {"canonical_pool_n": "bad"}, 5232: {}})
    assert "canonical GAP-4 pool n=0" in blocked_summary
    assert "ARC delta 0" in blocked_summary


def test_req_report_5233_validation_rejects_schema_and_type_errors(tmp_path: Path) -> None:
    """REQ-REPORT-5233: artifact validation rejects schema drift and overclaims."""

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
        mod.validate_artifact(artifact | {"milestone": "2026.07.478"})
    with pytest.raises(ValueError, match="field principle"):
        mod.validate_artifact(
            artifact
            | {"field_principles": artifact["field_principles"] | {"honest_verdict": "wrong"}}
        )
    with pytest.raises(ValueError, match="principle-wrapped"):
        mod.validate_artifact(artifact | {"v478_summary": "not-wrapped"})
    with pytest.raises(ValueError, match="principle mismatch"):
        mod.validate_artifact(artifact | {"v478_summary": _wrap("summary")})
    with pytest.raises(ValueError, match="missing value"):
        mod.validate_artifact(
            artifact | {"v478_summary": {"principle": mod.FIELD_PRINCIPLES["v478_summary"]}}
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
