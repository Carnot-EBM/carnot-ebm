"""Tests for Exp 5134 archive .470 / activate .471 aggregation.

Spec refs: REQ-REPORT-5134, SCENARIO-REPORT-5134,
SCENARIO-REPORT-5134-ACTIVE-FALLBACK.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from carnot import experiment_5134_archive_470_activate_471 as mod


GREEN_VERIFY = mod.CommandResult(
    command=("python", "scripts/adversarial_verify.py"),
    exit_code=0,
    stdout='{"flags":[]}',
    stderr="",
)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _task_yaml(milestone: str = "2026.07.471", *, last: int = 5145) -> str:
    tasks = "\n".join(
        f"- id: exp{exp_id}-task\n  milestone: {milestone}\n  title: task {exp_id}"
        for exp_id in range(5134, last + 1)
    )
    return f"milestone: {milestone}\ntasks:\n{tasks}\n"


def _capstone_payload() -> dict:
    artifact_rows = [
        (5122, "archive_469_activate_470", "transition", "clean"),
        (5123, "v470_source_scope_audit", "planning", "adversarially_flagged"),
        (5124, "clean_sota_runtime_provenance", "runtime", "clean"),
        (5125, "structured_reasoning_pool", "structured_energy", "adversarially_flagged"),
        (5126, "distributional_energy_ranker", "structured_energy", "adversarially_flagged"),
        (5127, "structured_energy_adversarial_audit", "structured_energy", "gated_skip"),
        (5128, "kan_certificate_explanation", "kan_certificate", "clean"),
        (5129, "hubo_adaptive_2dpt", "solver_sampling", "clean"),
        (5130, "taco_sampler_heldout_scale", "solver_sampling", "clean"),
        (5131, "fr11_case_policy_self_learning", "fr11", "clean"),
        (5132, "authenticated_board_timing", "hardware", "clean"),
    ]
    return {
        "experiment_id": "exp5133-capstone-v470",
        "milestone": "2026.07.470",
        "honest_verdict": (
            "complete_capstone_v470_runtime_clean_exact_solver_progress_"
            "structured_energy_quarantined_fr11_no_promote_hardware_continuity"
        ),
        "inference_substrate": "aggregation_from_upstream_artifacts",
        "duration_s": 0.1,
        "flagged_adversarial": False,
        "artifacts_read": [
            {
                "experiment_number": exp_id,
                "label": label,
                "axis": axis,
                "path": f"results/experiment_{exp_id}_{label}_v470.json",
                "exists": True,
                "loadable": True,
                "classification": classification,
                "flagged_adversarial_stamped": classification == "adversarially_flagged",
                "headline_eligible": classification == "clean",
            }
            for exp_id, label, axis, classification in artifact_rows
        ],
        "missing_artifacts": [],
        "runtime_state": {
            "state": "clean_sota_runtime_ready",
            "sota_runtime_clean": True,
            "cache_ready": True,
            "completion_ready": True,
            "logprob_ready": True,
            "quarantined": False,
            "headline_eligible": True,
            "source_experiment": "exp5124",
        },
        "structured_energy_state": {
            "state": "no_surviving_positive_audit_gap",
            "attempted_pool": {"structured_pool_ready": True, "pool_n": 96},
            "attempted_ranker": {
                "distributional_energy_delta": 0.0,
                "ranker_ready_for_audit": False,
                "ranker_metrics": {"accuracy_at_1": 0.5},
                "strongest_cheap_baseline": {"name": "constraint_count_only"},
            },
            "audit_state": {"gated_skip": True},
            "failure_reasons": [
                "structured_pool_quarantined",
                "ranker_quarantined",
                "ranker_delta_not_positive",
                "audit_gate_skipped",
            ],
            "quarantined_experiments": [5125, 5126],
            "gated_skip_experiments": [5127],
            "positive_result_survived_audit": False,
            "headline_eligible": False,
        },
        "kan_certificate_state": {
            "state": "clean_certificate_explanation_positive",
            "certificate_soundness": True,
            "explanation_cycle_soundness": True,
            "false_property_detected": True,
            "kan_certificate_breadth_ready": True,
            "property_family_count": 4,
        },
        "solver_sampling_state": {
            "state": "clean_exact_checked_bounded_solver_sampling_progress",
            "adaptive_2dpt_ready": True,
            "exact_enumeration_checked": True,
            "detailed_balance_passed": True,
            "heldout_csp_trace_suite_ready": True,
            "guarded_effort_reduction_ratio": 0.04785,
            "harmful_instance_count_guarded": 3,
            "harmful_instance_count_unguarded": 4,
            "hardware_speedup_claimed": False,
            "wrong_label_count": 0,
        },
        "fr11_state": {
            "state": "safe_no_promotion",
            "continuous_self_learning_task": True,
            "heldout_delta": 0.0,
            "nonforgetting_delta": 0.0,
            "promotion_attempted": True,
            "promotion_safe": False,
            "rollback_applied": True,
            "no_weight_update": True,
        },
        "hardware_state": {
            "state": "continuity_with_authenticated_blockers_no_speedup_claim",
            "kv260_ssh_ready": True,
            "polarfire_ssh_ready": True,
            "extropic_tsu_execution_claimed": False,
            "no_speedup_claim": True,
            "timing_measurements": {"full_board_speedup_evidence_present": False},
        },
        "quarantined_artifacts": [
            {"experiment_number": 5125, "classification": "adversarially_flagged"},
            {"experiment_number": 5126, "classification": "adversarially_flagged"},
        ],
        "gated_skips": [
            {
                "experiment_number": 5127,
                "honest_verdict": "blocked_gate_check_failed",
                "gate_skip_reason": "distributional_energy_delta gate failed",
            }
        ],
        "active_roadmap_modified": False,
        "conductor_modified": False,
        "tests_run": ["capstone unit"],
        "reproducibility_checksum": "sha256:" + "0" * 64,
    }


def make_repo(
    tmp_path: Path,
    *,
    next_present: bool = True,
    active_valid: bool = True,
    research_complete_has_v470: bool = True,
) -> Path:
    root = tmp_path
    (root / "results").mkdir(parents=True, exist_ok=True)
    (root / "openspec" / "change-proposals").mkdir(parents=True, exist_ok=True)
    (root / "scripts").mkdir(parents=True, exist_ok=True)
    (root / "scripts" / "research_conductor.py").write_text("# conductor\n", encoding="utf-8")
    (root / "openspec" / "change-proposals" / "research-roadmap-vNEXT.md").write_text(
        "# Research Roadmap vNEXT\n\n**Milestone:** `2026.07.471`\n\nexp5134 through exp5145.\n",
        encoding="utf-8",
    )
    if next_present:
        (root / "research-roadmap-next.yaml").write_text(_task_yaml(), encoding="utf-8")
    (root / "research-roadmap.yaml").write_text(
        _task_yaml() if active_valid else _task_yaml("2026.07.470", last=5136),
        encoding="utf-8",
    )
    milestone_rows = (
        "- id: 2026.07.470\n  title: archived\n  tasks: []\n"
        if research_complete_has_v470
        else "- id: 2026.07.469\n  title: archived\n  tasks: []\n"
    )
    (root / "research-complete.yaml").write_text(f"milestones:\n{milestone_rows}", encoding="utf-8")
    capstone = _capstone_payload()
    _write_json(root / mod.CAPSTONE_RELATIVE_PATH, capstone)
    for row in capstone["artifacts_read"]:
        _write_json(
            root / row["path"],
            {
                "experiment_id": f"exp{row['experiment_number']}",
                "honest_verdict": row.get("honest_verdict", "complete_placeholder"),
                "flagged_adversarial": row["classification"] == "adversarially_flagged",
                "duration_s": 0.2,
                "inference_substrate": "aggregation_from_upstream_artifacts",
            },
        )
    return root


def test_req_report_5134_spec_declares_archive_contract() -> None:
    """REQ-REPORT-5134: OpenSpec anchors the .470 archive and .471 activation contract."""

    spec = Path("openspec/capabilities/research-reporting/spec.md").read_text(encoding="utf-8")

    for marker in (
        "REQ-REPORT-5134",
        "SCENARIO-REPORT-5134",
        "SCENARIO-REPORT-5134-ACTIVE-FALLBACK",
        "results/experiment_5134_archive_470_activate_471.json",
        "v470_distributional_delta",
    ):
        assert marker in spec
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert f"`{field}`" in spec
        assert principle in spec


def test_scenario_report_5134_happy_path_records_close_state_and_next_roadmap(
    tmp_path: Path,
) -> None:
    """SCENARIO-REPORT-5134: next-roadmap readiness and .470 close-state are preserved."""

    root = make_repo(tmp_path)
    artifact = mod.build_artifact(
        root=root,
        duration_s=1.25,
        run_date="20260702",
        verification=mod.verification_payload(GREEN_VERIFY),
        tests_run=["unit-test-placeholder"],
    )

    mod.validate_artifact(artifact)
    assert artifact["experiment_id"] == "exp5134-archive-470-activate-471"
    assert artifact["milestone"] == "2026.07.471"
    assert artifact["honest_verdict"] == mod.COMPLETE_VERDICT
    assert artifact["inference_substrate"] == "aggregation_from_upstream_artifacts"
    assert artifact["roadmap_next_present"] is True
    assert artifact["roadmap_next_check"]["required_task_ids_present"] is True
    assert artifact["active_roadmap_modified"] is False
    assert artifact["conductor_modified"] is False
    assert artifact["v470_runtime_clean"] is True
    assert artifact["v470_structured_energy_quarantined"] is True
    assert artifact["v470_distributional_delta"] == 0.0
    assert artifact["v470_kan_positive"] is True
    assert artifact["v470_sampler_positive"] is True
    assert artifact["v470_fr11_no_promote"] is True
    assert artifact["v470_hardware_no_speedup"] is True
    assert artifact["research_complete_has_v470"] is True
    assert artifact["v470_taco_bounded_positive_with_harm_cases"] is True
    assert any(
        row["source_id"] == "exp5133-capstone-v470" for row in artifact["source_artifacts_read"]
    )


def test_scenario_report_5134_active_fallback_does_not_recreate_next_roadmap(
    tmp_path: Path,
) -> None:
    """SCENARIO-REPORT-5134-ACTIVE-FALLBACK: activated .471 is enough without mutation."""

    root = make_repo(tmp_path, next_present=False)
    active_before = (root / "research-roadmap.yaml").read_text(encoding="utf-8")
    conductor_before = (root / "scripts" / "research_conductor.py").read_text(encoding="utf-8")

    output = mod.run(
        root=root,
        run_date="20260702",
        clock=iter([100.0, 101.0]).__next__,
        verification_runner=lambda path: GREEN_VERIFY,
        tests_run=["unit-test-placeholder"],
    )
    artifact = json.loads(output.read_text(encoding="utf-8"))

    assert artifact["honest_verdict"] == mod.ACTIVE_FALLBACK_VERDICT
    assert artifact["roadmap_next_present"] is False
    assert artifact["active_roadmap_check"]["required_task_ids_present"] is True
    assert artifact["active_roadmap_fallback_used"] is True
    assert artifact["active_roadmap_modified"] is False
    assert artifact["conductor_modified"] is False
    assert not (root / "research-roadmap-next.yaml").exists()
    assert (root / "research-roadmap.yaml").read_text(encoding="utf-8") == active_before
    assert (root / "scripts" / "research_conductor.py").read_text(
        encoding="utf-8"
    ) == conductor_before
    mod.validate_artifact(artifact)


def test_scenario_report_5134_blocks_when_no_next_or_active_471(tmp_path: Path) -> None:
    """REQ-REPORT-5134: missing next roadmap blocks if active `.471` fallback is absent."""

    root = make_repo(tmp_path, next_present=False, active_valid=False)
    artifact = mod.build_artifact(
        root=root,
        duration_s=1.0,
        run_date="20260702",
        verification=mod.verification_payload(GREEN_VERIFY),
        tests_run=["unit-test-placeholder"],
    )

    assert artifact["honest_verdict"] == "blocked_research_roadmap_next_missing"
    assert artifact["roadmap_next_present"] is False
    assert artifact["active_roadmap_fallback_used"] is False
    mod.validate_artifact(artifact)


def test_req_report_5134_ledger_gap_is_visible_but_not_repaired(tmp_path: Path) -> None:
    """REQ-REPORT-5134: research-complete `.470` gaps are recorded, not silently repaired."""

    root = make_repo(tmp_path, research_complete_has_v470=False)
    artifact = mod.build_artifact(
        root=root,
        duration_s=1.0,
        run_date="20260702",
        verification=mod.verification_payload(GREEN_VERIFY),
        tests_run=["unit-test-placeholder"],
    )

    mod.validate_artifact(artifact)
    assert artifact["research_complete_has_v470"] is False
    assert artifact["ledger_state"]["v470_entry_count"] == 0
    assert artifact["ledger_state"]["ledger_gap"] == "missing_2026.07.470"


def test_req_report_5134_validation_rejects_schema_drift(tmp_path: Path) -> None:
    """REQ-REPORT-5134: schema validation fails closed on required-field drift."""

    valid = mod.build_artifact(
        root=make_repo(tmp_path),
        duration_s=1.0,
        run_date="20260702",
        verification=mod.verification_payload(GREEN_VERIFY),
        tests_run=["unit-test-placeholder"],
    )
    mod.validate_artifact(valid)

    mutations = [
        ("experiment_id", "wrong"),
        ("milestone", "2026.07.470"),
        ("honest_verdict", "bad"),
        ("inference_substrate", "live_llm_inference"),
        ("duration_s", 0.0),
        ("source_artifacts_read", []),
        ("v470_runtime_clean", False),
        ("v470_structured_energy_quarantined", False),
        ("v470_distributional_delta", 1.0),
        ("v470_kan_positive", False),
        ("v470_sampler_positive", False),
        ("v470_fr11_no_promote", False),
        ("v470_hardware_no_speedup", False),
        ("research_complete_has_v470", "true"),
        ("roadmap_next_present", "false"),
        ("active_roadmap_modified", True),
        ("conductor_modified", True),
        ("flagged_adversarial", "false"),
        ("tests_run", []),
        ("reproducibility_checksum", "bad"),
    ]
    for key, value in mutations:
        payload = copy.deepcopy(valid)
        payload[key] = value
        with pytest.raises(ValueError):
            mod.validate_artifact(payload)

    payload = copy.deepcopy(valid)
    payload.pop("tests_run")
    with pytest.raises(ValueError, match="missing required artifact fields"):
        mod.validate_artifact(payload)

    payload = copy.deepcopy(valid)
    payload["field_principles"]["tests_run"] = "wrong"
    with pytest.raises(ValueError, match="field principle"):
        mod.validate_artifact(payload)


def test_req_report_5134_helper_edges_and_script_entrypoint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-REPORT-5134: helpers parse edge cases and the requested script delegates."""

    assert mod._number(True) is None
    assert mod._number("bad") is None
    assert mod._task_prefixes_present(["exp5134-a"], ["exp5134", "exp5135"]) is False
    assert mod.read_json_mapping(tmp_path / "missing.json")[1]["error"] == "missing"
    assert mod._roadmap_check(tmp_path / "missing.yaml")["milestone"] == "missing"
    (tmp_path / "poison.yaml").write_text("a: : :\n- [\n", encoding="utf-8")
    assert mod._roadmap_check(tmp_path / "poison.yaml")["parses"] is False
    assert mod._research_complete_check(tmp_path / "missing.yaml")["parses"] is False
    assert mod._research_complete_check(tmp_path / "poison.yaml")["parses"] is False
    (tmp_path / "duplicate_complete.yaml").write_text(
        "milestones:\n- id: 2026.07.470\n- id: 2026.07.470\n",
        encoding="utf-8",
    )
    assert (
        mod._research_complete_check(tmp_path / "duplicate_complete.yaml")["ledger_gap"]
        == "duplicate_2026.07.470_entries"
    )
    assert (
        mod._distributional_delta(
            {"structured_energy_state": {"attempted_ranker": {}}},
            {5126: {"distributional_energy_delta": 0.0}},
        )
        == 0.0
    )
    assert (
        mod.build_source_artifacts_read(
            tmp_path,
            {
                "artifacts_read": [
                    {"path": str(mod.CAPSTONE_RELATIVE_PATH), "experiment_number": 5133},
                    {"path": "", "experiment_number": 9999},
                ]
            },
        )[0]["source_id"]
        == "exp5133-capstone-v470"
    )
    assert (
        mod.load_referenced_payloads(
            tmp_path,
            {"artifacts_read": [{"path": "missing.json", "experiment_number": "bad"}]},
        )
        == {}
    )
    base_preconditions = {
        "capstone": {"loadable": True},
        "vnext_doc": {"exists": True, "names_milestone": True},
        "research_roadmap_next": {"exists": True, "milestone": mod.MILESTONE},
        "roadmap_next_ready": False,
        "active_roadmap_fallback_ready": False,
    }
    cases = [
        (
            {**base_preconditions, "capstone": {"loadable": False}},
            "blocked_capstone_artifact_missing_or_unloadable",
        ),
        ({**base_preconditions, "vnext_doc": {"exists": False}}, "blocked_vnext_doc_missing"),
        (
            {**base_preconditions, "vnext_doc": {"exists": True, "names_milestone": False}},
            "blocked_vnext_doc_milestone_mismatch",
        ),
        (
            {**base_preconditions, "research_roadmap_next": {"exists": True, "milestone": "old"}},
            "blocked_research_roadmap_next_milestone_mismatch",
        ),
        (base_preconditions, "blocked_research_roadmap_next_task_set_incomplete"),
    ]
    for preconditions, expected in cases:
        assert mod._honest_verdict(preconditions) == expected
    assert (
        mod._verification_flags(
            mod.CommandResult(command=(), exit_code=0, stdout="not-json", stderr="")
        )
        == []
    )
    assert (
        mod._verification_flags(mod.CommandResult(command=(), exit_code=0, stdout="[]", stderr=""))
        == []
    )
    assert (
        mod.verification_payload(
            mod.CommandResult(
                command=(),
                exit_code=0,
                stdout='{"flags":[{"severity":"critical","kind":"X"}]}',
                stderr="",
            )
        )["flagged_adversarial"]
        is True
    )
    assert mod._verification_flags(
        mod.CommandResult(
            command=(),
            exit_code=1,
            stdout='{"reports":[{"flags":[{"severity":"info","kind":"Y"}]}]}',
            stderr="",
        )
    ) == [{"severity": "info", "kind": "Y"}]

    root = tmp_path / "subprocess_root"
    (root / "scripts").mkdir(parents=True)
    (root / "scripts" / "adversarial_verify.py").write_text(
        "import json\nprint(json.dumps({'flags': []}))\n", encoding="utf-8"
    )
    output = root / "artifact.json"
    output.write_text("{}", encoding="utf-8")
    result = mod.run_adversarial_verification(root, output)
    assert result.exit_code == 0
    assert result.command[-1] == str(output)

    import scripts.experiment_5134_archive_470_activate_471 as entrypoint

    repo = make_repo(tmp_path / "entrypoint_repo", next_present=False)
    monkeypatch.setattr(entrypoint, "run_main", lambda **kwargs: repo / mod.RESULT_RELATIVE_PATH)
    assert entrypoint.main(root=repo, date="20260702") == repo / mod.RESULT_RELATIVE_PATH

    import runpy
    import sys

    script_path = Path("scripts/experiment_5134_archive_470_activate_471.py").resolve()
    python_dir = str(script_path.parents[1] / "python")
    monkeypatch.setattr(sys, "path", [entry for entry in sys.path if entry != python_dir])
    runpy.run_path(str(script_path), run_name="exp5134_entrypoint_import_guard")
    assert python_dir in sys.path

    module_cli_repo = make_repo(tmp_path / "module_cli_repo", next_present=False)
    module_cli_output = module_cli_repo / "module_cli_result.json"
    assert (
        mod.main(
            [
                "--root",
                str(module_cli_repo),
                "--output",
                str(module_cli_output),
                "--date",
                "20260702",
            ]
        )
        == 0
    )
    assert module_cli_output.exists()
