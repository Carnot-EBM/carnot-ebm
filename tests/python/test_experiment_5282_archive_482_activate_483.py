"""Tests for Exp 5282 archive .482 / .483 activation artifact.

Spec refs: REQ-REPORT-5282, SCENARIO-REPORT-5282,
SCENARIO-REPORT-5282-BLOCKED-CLOSEOUT.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import yaml

from carnot import experiment_5282_archive_482_activate_483 as mod


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "research-reporting" / "spec.md"
RESULT_PATH = REPO / mod.RESULT_RELATIVE_PATH


def _wrap(value: Any, principle: str = "fixture principle") -> dict[str, Any]:
    return {"principle": principle, "value": value}


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _roadmap(milestone: str, start: int = 5282, stop: int = 5294) -> str:
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
        "experiment_id": "exp5281-capstone-v482",
        "honest_verdict": _wrap(
            "complete: .482 synthesized with 9 clean positives, 0 clean nulls, "
            "1 harmful/regression result, 1 flagged/quarantined artifact, "
            "1 honest block, governed self-learning advanced, and hardware "
            "blocked with no speedup claim."
        ),
        "clean_positives": _wrap(
            [
                {
                    "experiment_number": 5271,
                    "summary": (
                        "local SOTA GGUF telemetry receipts available; "
                        "no verifier-quality claim"
                    ),
                },
                {
                    "experiment_number": 5273,
                    "summary": (
                        "solver fixture rebuilt with baseline validity and "
                        "counterexample coverage receipts"
                    ),
                },
                {
                    "experiment_number": 5275,
                    "summary": (
                        "governed decision-history memory ready with scope, "
                        "stale-conflict, poisoning, and rollback gates"
                    ),
                },
                {
                    "experiment_number": 5276,
                    "summary": (
                        "memory-assisted verifier dosing preserved quality, "
                        "avoided 0.857143 full verifier calls, and kept "
                        "unsafe_false_accepts=0"
                    ),
                },
                {
                    "experiment_number": 5277,
                    "summary": (
                        "bounded KAN PWA/MILP certificate scaled and rejected "
                        "nearby false property"
                    ),
                },
                {
                    "experiment_number": 5278,
                    "summary": (
                        "tiny solver fixture round-tripped through factor-graph "
                        "boundary; no hardware speedup claim"
                    ),
                },
                {
                    "experiment_number": 5280,
                    "summary": (
                        "producer evidence discipline ready; missing evidence "
                        "rejected and old V481 pilots remain quarantined"
                    ),
                },
            ]
        ),
        "harmful_or_regressions": _wrap(
            [
                {
                    "experiment_number": 5272,
                    "summary": (
                        "internal/logit signal was harmful relative to lexical "
                        "baseline; delta=-0.345679"
                    ),
                }
            ]
        ),
        "flagged_or_quarantined": _wrap(
            [
                {
                    "experiment_number": 5274,
                    "summary": (
                        "solver extraction retry blocked/unmeasured; "
                        "blockers=['llama_cpp_gpu_offload_unavailable']"
                    ),
                }
            ]
        ),
        "honest_blocks": _wrap(
            [
                {
                    "experiment_number": 5279,
                    "summary": (
                        "KV260 and PolarFire SSH blocked; GateMate physical/JTAG "
                        "blocked; speedup_claimed=false"
                    ),
                }
            ]
        ),
        "continuous_self_learning_advanced": _wrap(True),
        "hardware_speedup_claimed": _wrap(False),
        "docs_updated": _wrap(
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
        5269: {"honest_verdict": _wrap("complete: .481 archived and .482 activation-ready")},
        5270: {"honest_verdict": _wrap("complete: 5 new actionable findings appended")},
        5271: {
            "honest_verdict": _wrap("complete: telemetry_receipts_ready=true"),
            "telemetry_harness_ready": True,
            "no_quality_claim": _wrap(True),
        },
        5272: {
            "honest_verdict": _wrap(
                "complete: harmful internal/logit signal delta_over_lexical=-0.345679"
            ),
            "delta_over_lexical_baseline": _wrap(-0.345679012345679),
        },
        5273: {
            "honest_verdict": _wrap("complete: solver_fixture_ready true"),
            "solver_fixture_ready": True,
        },
        5274: {
            "honest_verdict": _wrap(
                "blocked_preconditions: llama_cpp_gpu_offload_unavailable; retry was unmeasured"
            ),
            "retry_outcome": "unmeasured",
        },
        5275: {
            "honest_verdict": _wrap("complete: governed decision-history memory is ready"),
            "memory_decision_history_ready": True,
        },
        5276: {
            "honest_verdict": _wrap("complete: positive memory-assisted verifier dosing"),
            "memory_verifier_dose_ready": _wrap(True),
        },
        5277: {
            "honest_verdict": _wrap("complete: scaled certificate positive"),
            "certificate_scaled": _wrap(True),
        },
        5278: {
            "honest_verdict": _wrap("complete: factor-graph boundary is usable"),
            "mapping_roundtrip": {"passed": True, "constraint_violation": 0},
        },
        5279: {
            "honest_verdict": _wrap("blocked_board_reachability: no_speedup_claim"),
            "hardware_speedup_claimed": _wrap(False),
        },
        5280: {
            "honest_verdict": _wrap("complete: producer evidence discipline is ready"),
            "normalizer_evidence_ready": _wrap(True),
        },
        5281: _capstone_payload(),
    }
    return payloads[number]


def _research_complete(has_v482: bool = True) -> str:
    milestones: list[dict[str, Any]] = [
        {
            "id": "2026.07.481",
            "title": "prior fixture",
            "completed": "2026-07-05",
            "finding": "fixture",
            "tasks": [],
        }
    ]
    if has_v482:
        milestones.append(
            {
                "id": "2026.07.482",
                "title": (
                    "Receipt-Clean Internal Verification, Governed Self-Learning, "
                    "and Hardware-Bound Certificates"
                ),
                "completed": "2026-07-05",
                "finding": (
                    "SOTA telemetry ready; internal/logit harmful; solver fixture "
                    "rebuilt; extraction blocked by GGUF offload; governed memory "
                    "and verifier dosing positive; KAN/factor bounded; hardware "
                    "blocked with no speedup; evidence audit complete."
                ),
                "tasks": [],
            }
        )
    return yaml.safe_dump({"milestones": milestones}, sort_keys=False)


def _make_repo(
    root: Path,
    *,
    has_v482_complete: bool = True,
    active_milestone: str = "2026.07.483",
    vnext_milestone: str = "2026.07.483",
    next_file: bool = False,
    capstone: dict[str, Any] | None = None,
) -> Path:
    for source in mod.UPSTREAM_SOURCES:
        _write_json(root / source.relative_path, _upstream_payload(source.experiment_number))
    if capstone is not None:
        _write_json(root / mod.CAPSTONE_RELATIVE_PATH, capstone)
    (root / "research-complete.yaml").write_text(
        _research_complete(has_v482_complete), encoding="utf-8"
    )
    (root / "research-roadmap.yaml").write_text(_roadmap(active_milestone), encoding="utf-8")
    if next_file:
        (root / "research-roadmap-next.yaml").write_text(_roadmap("2026.07.483"), encoding="utf-8")
    (root / "openspec/change-proposals").mkdir(parents=True, exist_ok=True)
    (root / "openspec/change-proposals/research-roadmap-vNEXT.md").write_text(
        f"# Research Roadmap vNEXT: Milestone {vnext_milestone}\nMilestone: {vnext_milestone}\n",
        encoding="utf-8",
    )
    (root / "ops").mkdir(parents=True, exist_ok=True)
    for relative in ("status.md", "changelog.md", "conductor-log.md", "exclusion_manifest.yaml"):
        (root / "ops" / relative).write_text(
            (
                "fixture .482 SOTA telemetry ready internal/logit harmful solver fixture "
                "rebuilt GGUF offload blocked governed memory verifier dosing KAN "
                "factor boundary hardware no speedup evidence audit\n"
            ),
            encoding="utf-8",
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


def test_req_report_5282_spec_declares_archive_contract() -> None:
    """REQ-REPORT-5282: OpenSpec anchors the .482 archive and .483 activation contract."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    section = spec[spec.index("REQ-REPORT-5282") : spec.index("REQ-REPORT-5257")]

    for marker in (
        "REQ-REPORT-5282",
        "SCENARIO-REPORT-5282",
        "SCENARIO-REPORT-5282-BLOCKED-CLOSEOUT",
        str(mod.RESULT_RELATIVE_PATH),
        "aggregation_from_upstream_artifacts",
        "roadmap_activation_check.activated=false",
        "internal/logit hallucination probe is harmful/regressive",
        "commands_run",
    ):
        assert marker in section
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert f"`{field}`" in section
        assert principle in section


def test_scenario_report_5282_already_active_records_no_overwrite(tmp_path: Path) -> None:
    """SCENARIO-REPORT-5282: already-active .483 produces a complete no-overwrite artifact."""

    root = _make_repo(tmp_path)
    before = (root / "research-roadmap.yaml").read_text(encoding="utf-8")
    artifact = mod.build_artifact(
        root=root,
        run_date="20260705",
        duration_s=1.25,
        validation_results=_passed_commands(),
        update_research_complete=False,
    )

    mod.validate_artifact(artifact)
    assert (root / "research-roadmap.yaml").read_text(encoding="utf-8") == before
    assert artifact["milestone_archived"] is True
    assert artifact["activation_ready"] is True
    assert artifact["honest_verdict"]["value"].startswith("complete:")
    assert ".482 archived" in artifact["honest_verdict"]["value"]
    assert ".483 activation-ready" in artifact["honest_verdict"]["value"]
    assert artifact["inference_substrate"]["value"] == mod.INFERENCE_SUBSTRATE
    assert artifact["ops_docs_updated"]["value"] is False
    assert artifact["research_complete_updated"]["value"] is False
    assert artifact["exclusions_checked"]["value"] is True
    assert artifact["roadmap_activation_check"]["activated"] is False
    assert artifact["roadmap_activation_check"]["active_roadmap_already_483"] is True
    assert artifact["roadmap_activation_check"]["roadmap_next_present"] is False
    assert artifact["roadmap_activation_check"]["roadmap_next_absence_handled_by"] == "active_roadmap_already_483"
    assert [row["passed"] for row in artifact["commands_run"]] == [True, True, True, True]
    assert artifact["closeout_facts"]["sota_telemetry_ready"] is True
    assert artifact["closeout_facts"]["internal_logit_harmful_regression"] is True
    assert artifact["closeout_facts"]["solver_fixture_rebuilt"] is True
    assert artifact["closeout_facts"]["sota_extraction_blocked_by_gguf_offload"] is True
    assert artifact["closeout_facts"]["governed_memory_positive"] is True
    assert artifact["closeout_facts"]["verifier_dosing_positive"] is True
    assert artifact["closeout_facts"]["kan_certificate_positive"] is True
    assert artifact["closeout_facts"]["factor_boundary_tiny_no_speedup"] is True
    assert artifact["closeout_facts"]["hardware_blocked_no_speedup"] is True
    assert artifact["closeout_facts"]["evidence_audit_complete"] is True
    assert artifact["failed_preconditions"] == []
    assert artifact["reproducibility_checksum"] == mod.payload_checksum(artifact)


def test_scenario_report_5282_blocked_closeout_keeps_failures_visible(tmp_path: Path) -> None:
    """SCENARIO-REPORT-5282-BLOCKED-CLOSEOUT: contradictory evidence blocks readiness."""

    bad_capstone = _capstone_payload() | {
        "harmful_or_regressions": _wrap([]),
        "flagged_or_quarantined": _wrap([]),
    }
    artifact = mod.build_artifact(
        root=_make_repo(
            tmp_path,
            has_v482_complete=False,
            active_milestone="2026.07.482",
            vnext_milestone="2026.07.482",
            capstone=bad_capstone,
        ),
        run_date="20260705",
        duration_s=0.5,
        validation_results=_failed_commands(),
        update_research_complete=False,
    )

    mod.validate_artifact(artifact)
    assert artifact["milestone_archived"] is False
    assert artifact["activation_ready"] is False
    assert artifact["honest_verdict"]["value"].startswith("blocked_")
    assert artifact["roadmap_activation_check"]["activated"] is False
    assert "closeout_internal_logit_harmful_regression_expected_True_observed_False" in artifact["failed_preconditions"]
    assert "closeout_sota_extraction_blocked_by_gguf_offload_expected_True_observed_False" in artifact["failed_preconditions"]
    assert "research_complete_missing_2026.07.482" in artifact["failed_preconditions"]
    assert "vnext_missing_2026.07.483" in artifact["failed_preconditions"]
    assert "active_or_next_roadmap_not_ready_for_483" in artifact["failed_preconditions"]
    assert "validation_failed_scripts/roadmap_schema.py" in artifact["failed_preconditions"]
    assert artifact["exclusions_checked"]["value"] is False


def test_req_report_5282_can_append_missing_research_complete_and_run(tmp_path: Path) -> None:
    """REQ-REPORT-5282: missing research-complete .482 entry can be appended once."""

    root = _make_repo(tmp_path, has_v482_complete=False, active_milestone="2026.07.482", next_file=True)
    out = mod.run(
        root=root,
        run_date="20260705",
        duration_s=0.25,
        validation_results=_passed_commands(),
        update_research_complete=True,
    )

    saved = json.loads(out.read_text(encoding="utf-8"))
    mod.validate_artifact(saved)
    assert saved["research_complete_updated"]["value"] is True
    assert saved["milestone_archived"] is True
    complete = yaml.safe_load((root / "research-complete.yaml").read_text(encoding="utf-8"))
    assert [row["id"] for row in complete["milestones"]][-1] == "2026.07.482"
    assert saved["roadmap_activation_check"]["roadmap_next_present"] is True
    assert saved["roadmap_activation_check"]["roadmap_next_milestone"] == "2026.07.483"
    assert saved["reproducibility_checksum"] == mod.payload_checksum(saved)


def test_req_report_5282_repository_artifact_matches_schema() -> None:
    """REQ-REPORT-5282: checked-in deliverable is a valid archive artifact."""

    artifact = json.loads(RESULT_PATH.read_text(encoding="utf-8"))

    mod.validate_artifact(artifact)
    assert artifact["experiment_id"] == mod.EXPERIMENT_ID
    assert artifact["milestone_archived"] is True
    assert artifact["activation_ready"] is True
    assert artifact["roadmap_activation_check"]["activated"] is False
    assert artifact["roadmap_activation_check"]["active_roadmap_already_483"] is True
    assert artifact["roadmap_activation_check"]["roadmap_next_present"] is False
    assert artifact["research_complete_updated"]["value"] is False
    assert artifact["exclusions_checked"]["value"] is True
    assert artifact["inference_substrate"]["value"] == mod.INFERENCE_SUBSTRATE
    assert artifact["honest_verdict"]["value"].startswith("complete:")


def test_req_report_5282_helper_edges_and_validation_guards(tmp_path: Path) -> None:
    """REQ-REPORT-5282: helpers fail closed instead of hiding schema drift."""

    root = _make_repo(tmp_path / "repo")
    artifact = mod.build_artifact(
        root=root,
        run_date="20260705",
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
        research_complete={"has_2026_07_482_after": False},
        roadmap={"vnext_names_2026_07_483": True, "activation_ready_without_overwrite": True},
        commands=[],
    )
    assert failures == ["research_complete_missing_2026.07.482", "validation_commands_missing"]

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
