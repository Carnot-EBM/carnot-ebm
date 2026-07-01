"""Tests for Exp 5109 archive .468 / activate .469 aggregation.

Spec refs: REQ-REPORT-5109, SCENARIO-REPORT-5109,
SCENARIO-REPORT-5109-BLOCKED-NEXT-ROADMAP.
"""

from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path

import pytest

from carnot.reporting import archive_468_activate_469_5109 as mod


GREEN_ADVERSARIAL = mod.CommandResult(
    command=["python", "scripts/adversarial_verify.py"],
    exit_code=0,
    stdout='{"flags":[]}',
    stderr="",
)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _base_artifact(exp_id: int, verdict: str, *, flagged: bool = False) -> dict:
    return {
        "experiment_id": exp_id,
        "honest_verdict": verdict,
        "flagged_adversarial": flagged,
        "duration_s": 0.25,
        "inference_substrate": "aggregation_from_upstream_artifacts",
        "reproducibility_checksum": "0" * 64,
    }


def _kan_wall() -> dict:
    return {
        "honest_verdict": "complete_kan_pwa_milp_scale_stress_wall_found_at_n10_reason_timeout",
        "flagged_adversarial": False,
        "duration_s": 421.877608,
        "inference_substrate": "exact_milp_solver_cpu",
        "unit_counts_tested": [5, 10, 20],
        "solve_times_s_by_n": {"5": 0.150606, "10": 120.900356, "20": 300.007705},
        "solver_timeout_hit": True,
        "largest_n_reached": 10,
        "realistic_kan_unit_count_reference": 100,
        "reached_production_reference": False,
        "adversarial_rigor_preserved_at_scale": True,
    }


def _research_complete() -> str:
    return (
        "- id: 2026.07.468\n"
        "  title: EXACT-VERIFIER SCALE-UP + EVIDENCE ENERGY + FORMAL FR-11\n"
        "  tasks:\n"
        "  - id: exp5095-phase0-archive-467-activate-468\n"
        "    deliverable: results/experiment_5095_archive_467_activate_468.json\n"
        "    result: OK (conductor)\n"
        "  - id: exp5107-capstone-v468\n"
        "    deliverable: results/experiment_5107_capstone_v468.json\n"
        "    result: OK (conductor)\n"
    )


def make_repo(tmp_path: Path, *, next_present: bool = True, next_milestone: str = "2026.07.469") -> Path:
    root = tmp_path
    (root / "ops").mkdir(parents=True, exist_ok=True)
    (root / "results").mkdir(parents=True, exist_ok=True)
    (root / "openspec" / "change-proposals").mkdir(parents=True, exist_ok=True)
    (root / "scripts").mkdir(parents=True, exist_ok=True)
    (root / "research-complete.yaml").write_text(_research_complete(), encoding="utf-8")
    (root / "ops" / "changelog.md").write_text(
        "exp5095 exp5096 exp5097 exp5098 exp5099 exp5100 exp5101 exp5102 "
        "exp5103 exp5104 exp5105 exp5106 exp5107 operational retro .468\n",
        encoding="utf-8",
    )
    (root / "ops" / "conductor-log.md").write_text(
        "2026-07-01 12:22 UTC | Milestone 2026.07.468 activated | OK\n"
        "2026-07-01 15:23 UTC | PHASE Z capstone | OK\n"
        "2026-07-01 16:42 UTC | Plan milestone 2026.07.469 | OK\n",
        encoding="utf-8",
    )
    (root / "openspec" / "change-proposals" / "research-roadmap-vNEXT.md").write_text(
        "# Research Roadmap vNEXT: 2026.07.469\n\nFoVer and post-wall KAN frame.\n",
        encoding="utf-8",
    )
    if next_present:
        (root / "research-roadmap-next.yaml").write_text(
            f'milestone: "{next_milestone}"\ntasks: []\n', encoding="utf-8"
        )
    (root / "research-roadmap.yaml").write_text(
        'milestone: "2026.07.469"\ntasks: []\n', encoding="utf-8"
    )
    (root / "scripts" / "research_conductor.py").write_text("# conductor\n", encoding="utf-8")

    _write_json(
        root / "results" / "experiment_5095_archive_467_activate_468.json",
        _base_artifact(5095, "complete_467_archived_468_activated_exact_verifier_pivot_carried_forward"),
    )
    _write_json(
        root / "results" / "experiment_5096_sota_ingestion_v468.json",
        _base_artifact(5096, "success_sota_ingestion_v468_references_verified"),
    )
    _write_json(
        root / "results" / "experiment_5097_clean_sota_endpoint_logprob_cache_v468.json",
        {
            **_base_artifact(
                5097,
                "blocked_clean_sota_endpoint_logprob_cache_no_live_logprobs",
            ),
            "inference_substrate": "precondition_check_only",
        },
    )
    _write_json(
        root / "results" / "experiment_5098_kan_pwa_milp_scale_v2.json",
        {
            **_base_artifact(5098, "success_kan_pwa_milp_scale_v2_property_suite_clean"),
            "inference_substrate": "exact_milp_solver_cpu",
            "properties_proved": ["exp5091_baseline_two_unit_true", "three_unit_composition_true"],
        },
    )
    _write_json(
        root / "results" / "experiment_5099_beaver_prefix_bound_verifier_v468.json",
        _base_artifact(5099, "complete_beaver_prefix_bounds_toy_only_runtime_not_clean", flagged=True),
    )
    _write_json(
        root / "results" / "experiment_5100_constrainprompt_code_assurance_v468.json",
        _base_artifact(5100, "success_constrainprompt_code_assurance_exact_checks_passed", flagged=True),
    )
    _write_json(
        root / "results" / "experiment_5101_incomplete_graph_evidence_energy_v468.json",
        _base_artifact(
            5101,
            "success_graph_evidence_energy_separates_contradiction_from_unsupported",
        ),
    )
    _write_json(
        root / "results" / "experiment_5102_hubo_pspin_direct_energy_v468.json",
        _base_artifact(5102, "success_hubo_pspin_direct_encoding_reduces_gadget_blowup"),
    )
    _write_json(
        root / "results" / "experiment_5103_taco_adaptive_csp_heuristic_v468.json",
        _base_artifact(5103, "success_taco_adaptive_heuristic_reduces_exact_solver_effort"),
    )
    _write_json(
        root / "results" / "experiment_5104_constrained_decoding_semantic_risk_audit_v468.json",
        _base_artifact(
            5104,
            "complete_constrained_decoding_semantic_audit_no_syntax_only_headline",
            flagged=True,
        ),
    )
    _write_json(
        root / "results" / "experiment_5105_fr11_severa_guarded_memory_v468.json",
        {
            **_base_artifact(
                5105,
                "complete_fr11_severa_guarded_memory_no_promote_contracts_working_delta_plus_0p000",
                flagged=True,
            ),
            "heldout_delta": 0.0,
            "nonforgetting_delta": 0.0,
            "promoted_count": 0,
            "contract_pass_count": 3,
            "promotion_decision": {
                "promoted": False,
                "no_promote_reason": "positive_utility_not_observed",
                "gate_conditions": {"heldout_delta_gt_zero": False},
            },
        },
    )
    _write_json(
        root / "results" / "experiment_5106_hardware_partition_telemetry_v468.json",
        {
            **_base_artifact(5106, "complete_hardware_partition_telemetry_no_speedup_claim"),
            "kv260_ssh_ready": True,
            "kv260_uio_transcript_collected": False,
            "kv260_blocker": "no_safe_kv260_uio_register_transcript_collected",
            "gatemate_terminal_state": "blocked_gatemate_dirtyjtag_cable_seen_no_gatemate_idcode_terminal",
            "polarfire_ssh_ready": True,
            "polarfire_dispatch_precheck": {"ready": True, "dispatch_executed": False},
            "speedup_claimed": False,
            "destructive_actions_taken": [],
        },
    )
    _write_json(
        root / "results" / "experiment_5107_capstone_v468.json",
        {
            **_base_artifact(5107, "complete_capstone_v468_exact_verifier_scale_decision_recorded"),
            "runtime_substrate_decision": "blocked_clean_sota_endpoint_logprob_cache_no_live_logprobs",
            "fr11_decision": "no_promote_delta_plus_0p000",
            "hardware_decision": "continuity_no_speedup_claim",
        },
    )
    _write_json(root / "results" / "experiment_5108_kan_pwa_milp_scale_stress_test.json", _kan_wall())
    return root


def test_req_report_5109_spec_declares_archive_contract() -> None:
    """REQ-REPORT-5109: OpenSpec declares the .468 archive and .469 activation contract."""

    spec = Path("openspec/capabilities/research-reporting/spec.md").read_text(encoding="utf-8")

    assert "REQ-REPORT-5109" in spec
    assert "SCENARIO-REPORT-5109" in spec
    assert "SCENARIO-REPORT-5109-BLOCKED-NEXT-ROADMAP" in spec
    assert "results/experiment_5109_archive_468_activate_469.json" in spec
    assert "`research-roadmap.yaml`" in spec
    assert "`N=10` solved at about `120.9s`" in spec
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert f"`{field}`" in spec
        assert principle in spec


def test_scenario_report_5109_builds_close_state_from_upstream(tmp_path: Path) -> None:
    """SCENARIO-REPORT-5109: the derived state records positives, flags, blockers, and KAN wall."""

    root = make_repo(tmp_path)
    sources = mod.read_sources(root)
    state = mod.build_archive_state(sources)

    clean_ids = {row["experiment_id"] for row in state["clean_positives"]}
    assert {5098, 5101, 5102, 5103}.issubset(clean_ids)
    assert state["flagged_diagnostics"]["experiment_ids"] == [5099, 5100, 5104, 5105]
    assert state["blocked_runtime_substrate"]["experiment_id"] == 5097
    assert state["blocked_runtime_substrate"]["runtime_ready"] is False
    assert state["fr11_no_promote_state"]["promoted"] is False
    assert state["fr11_no_promote_state"]["heldout_delta"] == 0.0
    assert state["hardware_continuity_state"]["speedup_claimed"] is False
    assert state["hardware_continuity_state"]["kv260_ssh_ready"] is True
    assert state["exp5108_kan_wall"]["largest_n_reached"] == 10
    assert state["exp5108_kan_wall"]["n20_timed_out"] is True
    assert state["exp5108_kan_wall"]["realistic_n100_reached"] is False


def test_scenario_report_5109_happy_path_writes_complete_artifact(tmp_path: Path) -> None:
    """SCENARIO-REPORT-5109: a complete activation-readiness artifact validates."""

    root = make_repo(tmp_path)
    output = mod.run(
        root,
        adversarial_result=GREEN_ADVERSARIAL,
        started_s=1000.0,
        now_s=1001.0,
        run_label_date="20260701",
    )
    artifact = json.loads(output.read_text(encoding="utf-8"))

    assert artifact["experiment_id"] == "exp5109-archive-468-activate-469"
    assert artifact["milestone"] == "2026.07.469"
    assert artifact["honest_verdict"].startswith("complete_")
    assert artifact["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert artifact["roadmap_next_present"] is True
    assert artifact["active_roadmap_modified"] is False
    assert artifact["conductor_modified"] is False
    assert artifact["flagged_adversarial"] is False
    assert artifact["exp5108_kan_wall_recorded"] is True
    assert artifact["archive_state"]["blocked_runtime_substrate"]["runtime_ready"] is False
    assert artifact["archive_state"]["fr11_no_promote_state"]["promoted"] is False
    assert any(row["command"] == GREEN_ADVERSARIAL.command for row in artifact["tests_run"])
    assert set(mod.REQUIRED_ARTIFACT_FIELDS).issubset(artifact)
    assert set(mod.REQUIRED_ARTIFACT_FIELDS).issubset(artifact["field_principles"])
    mod.validate_artifact(artifact)


def test_scenario_report_5109_blocked_when_next_roadmap_missing(tmp_path: Path) -> None:
    """SCENARIO-REPORT-5109-BLOCKED-NEXT-ROADMAP: missing next queue blocks without mutation."""

    root = make_repo(tmp_path, next_present=False)
    active_before = (root / "research-roadmap.yaml").read_text(encoding="utf-8")
    conductor_before = (root / "scripts" / "research_conductor.py").read_text(encoding="utf-8")

    output = mod.run(
        root,
        adversarial_result=GREEN_ADVERSARIAL,
        started_s=1000.0,
        now_s=1001.0,
        run_label_date="20260701",
    )
    artifact = json.loads(output.read_text(encoding="utf-8"))

    assert artifact["honest_verdict"] == "blocked_research_roadmap_next_missing"
    assert artifact["roadmap_next_present"] is False
    assert artifact["roadmap_next_milestone"] == "missing"
    assert artifact["active_roadmap_modified"] is False
    assert artifact["conductor_modified"] is False
    assert artifact["exp5108_kan_wall_recorded"] is True
    assert (root / "research-roadmap.yaml").read_text(encoding="utf-8") == active_before
    assert (root / "scripts" / "research_conductor.py").read_text(encoding="utf-8") == conductor_before
    mod.validate_artifact(artifact)


def test_scenario_report_5109_blocked_when_next_roadmap_names_wrong_milestone(
    tmp_path: Path,
) -> None:
    """SCENARIO-REPORT-5109-BLOCKED-NEXT-ROADMAP: a wrong next milestone is explicit."""

    root = make_repo(tmp_path, next_milestone="2026.07.470")
    artifact = json.loads(
        mod.run(
            root,
            adversarial_result=GREEN_ADVERSARIAL,
            started_s=1000.0,
            now_s=1001.0,
            run_label_date="20260701",
        ).read_text(encoding="utf-8")
    )

    assert artifact["honest_verdict"] == "blocked_research_roadmap_next_milestone_mismatch"
    assert artifact["roadmap_next_present"] is True
    assert artifact["roadmap_next_milestone"] == "2026.07.470"
    mod.validate_artifact(artifact)


def test_validate_artifact_rejects_missing_required_field(tmp_path: Path) -> None:
    """REQ-REPORT-5109: schema validation fails closed on required-field drift."""

    root = make_repo(tmp_path)
    payload = mod.build_artifact(
        archive_state=mod.build_archive_state(mod.read_sources(root)),
        source_artifacts_read=mod.build_source_artifacts_read(root),
        preconditions_checked={"ok": True},
        duration_s=1.0,
        roadmap_next_present=True,
        roadmap_next_milestone="2026.07.469",
        roadmap_next_path="research-roadmap-next.yaml",
        active_roadmap_modified=False,
        conductor_modified=False,
        adversarial_verification=mod.command_result_payload(GREEN_ADVERSARIAL),
        honest_verdict="complete_468_archived_exp5108_wall_recorded_469_ready",
        tests_run=[mod.command_result_payload(GREEN_ADVERSARIAL)],
        run_label_date="20260701",
    )
    mod.validate_artifact(payload)
    payload.pop("tests_run")
    try:
        mod.validate_artifact(payload)
    except ValueError as exc:
        assert "missing required artifact fields" in str(exc)
    else:  # pragma: no cover - defensive test guard.
        raise AssertionError("validate_artifact accepted a missing required field")


def test_req_report_5109_helper_edge_cases_and_real_subprocess(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-REPORT-5109: helper edge cases stay deterministic and verifier subprocess is captured."""

    assert mod._number("bad", 7.0) == 7.0
    assert mod._is_clean_research_positive(5095, {}) is False
    assert mod._milestone_from_yaml_text("a: : :\n- [\n") == "yaml_poison"

    monkeypatch.setattr(mod.yaml, "safe_load", lambda text: [])
    assert mod._milestone_from_yaml_text('milestone: "2026.07.469"\n') == "2026.07.469"
    assert mod._milestone_from_yaml_text("tasks: []\n") == "unknown"

    assert mod._parse_adversarial_flags(
        mod.CommandResult(command=[], exit_code=0, stdout="not-json", stderr="")
    ) == []
    assert mod._parse_adversarial_flags(
        mod.CommandResult(command=[], exit_code=0, stdout='{"flags":"bad"}', stderr="")
    ) == []
    assert mod._adversarial_flagged(
        mod.CommandResult(command=[], exit_code=1, stdout='{"flags":[]}', stderr="")
    ) is True
    assert mod._adversarial_flagged(
        mod.CommandResult(
            command=[],
            exit_code=0,
            stdout='{"flags":[{"severity":"critical","kind":"X"}]}',
            stderr="",
        )
    ) is True

    root = tmp_path / "subprocess_root"
    (root / "scripts").mkdir(parents=True)
    (root / "scripts" / "adversarial_verify.py").write_text(
        "import json\nprint(json.dumps({'flags': []}))\n", encoding="utf-8"
    )
    artifact_path = root / "artifact.json"
    artifact_path.write_text("{}", encoding="utf-8")
    result = mod.run_adversarial_verification(root, artifact_path)
    assert result.exit_code == 0
    assert result.command[-1] == str(artifact_path)
    assert '"flags": []' in result.stdout


def test_req_report_5109_blocked_reason_covers_fail_closed_preconditions() -> None:
    """REQ-REPORT-5109: each precondition produces a specific blocked verdict."""

    base = {
        "research_complete_yaml": {
            "exists": True,
            "parses": True,
            "contains_archived_milestone": True,
        },
        "ops_changelog": {"exists": True},
        "ops_conductor_log": {"exists": True},
        "vnext_doc": {"exists": True, "names_milestone": True},
        "required_result_artifacts": {"all_present": True},
        "research_roadmap_next": {"exists": True, "names_milestone": True},
    }

    cases = [
        (("research_complete_yaml", "exists", False), "blocked_research_complete_yaml_missing"),
        (("research_complete_yaml", "parses", False), "blocked_research_complete_yaml_poison"),
        (
            ("research_complete_yaml", "contains_archived_milestone", False),
            "blocked_research_complete_missing_468_record",
        ),
        (("ops_changelog", "exists", False), "blocked_ops_changelog_missing"),
        (("ops_conductor_log", "exists", False), "blocked_ops_conductor_log_missing"),
        (("vnext_doc", "exists", False), "blocked_vnext_doc_missing"),
        (("vnext_doc", "names_milestone", False), "blocked_vnext_doc_milestone_mismatch"),
        (
            ("required_result_artifacts", "all_present", False),
            "blocked_source_artifact_missing",
        ),
        (("research_roadmap_next", "exists", False), "blocked_research_roadmap_next_missing"),
        (
            ("research_roadmap_next", "names_milestone", False),
            "blocked_research_roadmap_next_milestone_mismatch",
        ),
    ]
    for (section, key, value), expected in cases:
        preconditions = copy.deepcopy(base)
        preconditions[section][key] = value
        assert mod._blocked_reason(preconditions) == expected
    assert mod._blocked_reason(base) is None


def test_req_report_5109_validate_artifact_rejects_each_schema_drift(tmp_path: Path) -> None:
    """REQ-REPORT-5109: validation reports every required schema boundary."""

    root = make_repo(tmp_path)
    valid = mod.build_artifact(
        archive_state=mod.build_archive_state(mod.read_sources(root)),
        source_artifacts_read=mod.build_source_artifacts_read(root),
        preconditions_checked={"ok": True},
        duration_s=1.0,
        roadmap_next_present=True,
        roadmap_next_milestone="2026.07.469",
        roadmap_next_path="research-roadmap-next.yaml",
        active_roadmap_modified=False,
        conductor_modified=False,
        adversarial_verification={**mod._placeholder_adversarial(), "flagged_adversarial": False},
        honest_verdict=mod.COMPLETE_VERDICT,
        tests_run=[mod.command_result_payload(GREEN_ADVERSARIAL)],
        run_label_date="20260701",
    )
    mod.validate_artifact(valid)

    mutations = [
        ("honest_verdict", "bad"),
        ("experiment_id", "wrong"),
        ("milestone", "2026.07.470"),
        ("inference_substrate", "live_llm_inference"),
        ("duration_s", "1"),
        ("duration_s", 0.0),
        ("source_artifacts_read", {}),
        ("field_principles", []),
        ("active_roadmap_modified", True),
        ("conductor_modified", True),
        ("flagged_adversarial", "no"),
        ("tests_run", []),
        ("exp5108_kan_wall_recorded", False),
        ("reproducibility_checksum", "bad"),
    ]
    for key, value in mutations:
        payload = copy.deepcopy(valid)
        payload[key] = value
        with pytest.raises(ValueError):
            mod.validate_artifact(payload)

    bad_principles = copy.deepcopy(valid)
    bad_principles["field_principles"]["tests_run"] = "wrong"
    with pytest.raises(ValueError):
        mod.validate_artifact(bad_principles)

    bad_complete = copy.deepcopy(valid)
    bad_complete["roadmap_next_present"] = False
    with pytest.raises(ValueError):
        mod.validate_artifact(bad_complete)

    kan_cases = [
        ("largest_n_reached", 9),
        ("n20_timed_out", False),
        ("realistic_n100_reached", True),
    ]
    for key, value in kan_cases:
        payload = copy.deepcopy(valid)
        payload["archive_state"]["exp5108_kan_wall"][key] = value
        with pytest.raises(ValueError):
            mod.validate_artifact(payload)


def test_script_entrypoint_delegates_to_module(tmp_path: Path) -> None:
    """REQ-REPORT-5109: the requested script entrypoint delegates to the tested module."""

    script_path = Path("scripts/experiment_5109_archive_468_activate_469.py")
    spec = importlib.util.spec_from_file_location("exp5109_entrypoint", script_path)
    assert spec is not None and spec.loader is not None
    entrypoint = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(entrypoint)

    root = make_repo(tmp_path, next_present=False)
    output = entrypoint.main(root=root, date="20260701", adversarial_result=GREEN_ADVERSARIAL)
    artifact = json.loads(Path(output).read_text(encoding="utf-8"))
    assert artifact["experiment_id"] == "exp5109-archive-468-activate-469"
    assert artifact["honest_verdict"] == "blocked_research_roadmap_next_missing"
