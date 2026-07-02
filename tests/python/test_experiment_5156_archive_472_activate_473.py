"""Tests for Exp 5156 archive .472 / activate .473 aggregation.

Spec refs: REQ-REPORT-5156, SCENARIO-REPORT-5156,
SCENARIO-REPORT-5156-DIRTY-RUNTIME.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from carnot import experiment_5156_archive_472_activate_473 as mod


GREEN_VERIFY = mod.CommandResult(
    command=("python", "scripts/adversarial_verify.py"),
    exit_code=0,
    stdout='{"flags":[]}',
    stderr="",
)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _task_yaml(milestone: str = "2026.07.473", *, last: int = 5167) -> str:
    tasks = "\n".join(
        f"- id: exp{exp_id}-task\n  milestone: {milestone}\n  title: task {exp_id}"
        for exp_id in range(5156, last + 1)
    )
    return f"milestone: {milestone}\ntasks:\n{tasks}\n"


def _v472_payloads() -> dict[int, dict]:
    return {
        5151: {
            "experiment": "experiment_5151_arc_oracle_distinct_hardening_v472",
            "honest_verdict": (
                "complete_arc_set_encoder_win_not_hardened: +44pp win does not fully "
                "survive hardening; unresolved_axes=cross_game"
            ),
            "headline_outcome": "arc_set_encoder_win_not_hardened",
            "hardening_axes": {
                "multiseed": "passed",
                "leak_audit": "passed",
                "exact_test": "passed",
                "cross_game": "blocked",
            },
            "cross_game_blocked_reason": "blocked_arc_game_ids_unrecoverable",
            "cross_game_replication_delta": None,
            "exact_test_passes_min6_rule": True,
            "leak_audit_passed": True,
            "multiseed_delta_ci95": [0.4265639953, 0.4888206201],
            "verifier_is_oracle": False,
            "reproducibility_checksum": "sha256:" + "1" * 64,
        },
        5152: {
            "experiment": "experiment_5152_diffusiongemma_gate_reexamination_v472",
            "honest_verdict": "complete_diffusiongemma_gate_reexamined_keep_gated_corrected_arc_evidence",
            "domain_conflation_found": True,
            "recommendation": {"value": "keep_gated"},
            "exp5151_status": {
                "supports_ungating": False,
                "reason": "not_fully_hardened_or_null",
            },
            "reproducibility_checksum": "sha256:" + "2" * 64,
        },
        5153: {
            "experiment": "experiment_5153_gap4_scaleup_v472",
            "honest_verdict": "complete: gap4_scaleup_protocol_partial_still_open",
            "gap4_status_recommendation": "still_open",
            "n_400_task_result": None,
            "protocol_acceptance_passed": False,
            "protocol_steps_completed": [
                {"step_id": "sandboxed_400_task_reconfirmation", "passed": False},
                {"step_id": "transcripts_archived", "passed": False},
                {"step_id": "genuinely_heldout_tasks", "passed": False},
                {"step_id": "codex_first_arm", "passed": False},
                {"step_id": "statistical_tests", "passed": False},
                {"step_id": "hardened_exec_sandbox", "passed": True},
                {"step_id": "local_open_weight_generator_arm", "passed": False},
            ],
            "reproducibility_checksum": "sha256:" + "3" * 64,
        },
        5154: {
            "experiment": "experiment_5154_energy_fitness_directed_exploration_v472",
            "honest_verdict": (
                "complete: energy_fitness_qd_winning_trajectory_not_surfaced_reproducible_delta_0"
            ),
            "winning_trajectory_surfaced": False,
            "reproducible_levels_delta": 0,
            "energy_arm": {"winning_trajectory_surfaced": False, "reached_level": 0},
            "matched_control": {"winning_trajectory_surfaced": False, "reached_level": 0},
            "energy_signal_source": "exp4020_graded_goal_satisfaction_energy",
            "offline_reproduced": False,
            "reproducibility_checksum": "sha256:" + "4" * 64,
        },
        5155: {
            "experiment": "experiment_5155_multilevel_belief_state_scoping_v472",
            "honest_verdict": (
                "complete: deepen_belief_state_reset_scoped_3_falsifiable_experiments_no_full_build"
            ),
            "belief_state_resets_at_level_boundary": {"value": True},
            "proposed_experiments": {
                "value": [
                    {
                        "name": "transition_slice_warm_start_replay_ablation",
                        "signal_rank": 1,
                    },
                    {
                        "name": "cross_level_goal_energy_ranker_replay",
                        "signal_rank": 2,
                    },
                    {
                        "name": "hidden_register_hazard_belief_carryover_probe",
                        "signal_rank": 3,
                    },
                ]
            },
            "reproducibility_checksum": "sha256:" + "5" * 64,
        },
    }


def make_repo(
    tmp_path: Path,
    *,
    active_valid: bool = True,
    known_issue_has_directive: bool = True,
    omit_results: set[int] | None = None,
) -> Path:
    root = tmp_path
    (root / "results").mkdir(parents=True, exist_ok=True)
    (root / "ops").mkdir(parents=True, exist_ok=True)
    (root / "scripts").mkdir(parents=True, exist_ok=True)
    (root / "scripts" / "research_conductor.py").write_text("# conductor\n", encoding="utf-8")
    known_issue = (
        "### ENERGY-BASED ARC RESEARCH LINEUP 2026-07-02\n"
        "we want to continue down this energy based models path for ARC-AGI-3, "
        "and tackle the multi-level capable live agent\n"
        if known_issue_has_directive
        else "### Different note\n"
    )
    (root / "ops" / "known-issues.md").write_text(known_issue, encoding="utf-8")
    (root / "research-roadmap.yaml").write_text(
        _task_yaml() if active_valid else _task_yaml("2026.07.472", last=5158),
        encoding="utf-8",
    )
    for exp_id, payload in _v472_payloads().items():
        if omit_results and exp_id in omit_results:
            continue
        _write_json(root / mod.V472_RESULT_PATHS[exp_id], payload)
    return root


def clean_runtime_snapshot() -> mod.RuntimeSnapshot:
    return mod.RuntimeSnapshot(
        git_status_porcelain="",
        process_table=(
            "100 42 Ssl 03:50:42 python scripts/research_conductor.py --loop\n"
            "101 100 Ssl 00:00:59 codex exec --cd /repo -\n"
        ),
    )


def test_req_report_5156_spec_declares_archive_contract() -> None:
    """REQ-REPORT-5156: OpenSpec anchors the .472 archive and .473 activation contract."""

    spec = Path("openspec/capabilities/research-reporting/spec.md").read_text(encoding="utf-8")

    for marker in (
        "REQ-REPORT-5156",
        "SCENARIO-REPORT-5156",
        "SCENARIO-REPORT-5156-DIRTY-RUNTIME",
        "results/experiment_5156_archive_472_activate_473.json",
        "v472_runtime_clean",
        "ENERGY-BASED ARC RESEARCH LINEUP 2026-07-02",
    ):
        assert marker in spec
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert f"`{field}`" in spec
        assert principle in spec


def test_scenario_report_5156_happy_path_archives_partial_results(
    tmp_path: Path,
) -> None:
    """SCENARIO-REPORT-5156: .472 truth and clean .473 activation are preserved."""

    artifact = mod.build_artifact(
        root=make_repo(tmp_path),
        duration_s=1.25,
        run_date="20260702",
        verification=mod.verification_payload(GREEN_VERIFY),
        runtime_snapshot=clean_runtime_snapshot(),
        tests_run=["unit-test-placeholder"],
    )

    mod.validate_artifact(artifact)
    assert artifact["experiment_id"] == "exp5156-archive-472-activate-473"
    assert artifact["milestone"] == "2026.07.473"
    assert artifact["archived_milestone"] == "2026.07.472"
    assert artifact["honest_verdict"] == mod.COMPLETE_VERDICT
    assert artifact["inference_substrate"] == "aggregation_from_upstream_artifacts"
    assert artifact["v472_runtime_clean"] is True
    assert artifact["runtime_clean_details"]["non_transition_dirty_paths"] == []
    assert artifact["runtime_clean_details"]["orphaned_conductor_processes"] == []
    assert artifact["active_roadmap_ready"] is True
    assert artifact["active_roadmap_modified"] is False
    assert artifact["conductor_modified"] is False
    assert len(artifact["task_verdicts"]) == 5

    archive = {row["experiment_id"]: row for row in artifact["milestone_archive_summary"]}
    assert "failure" not in json.dumps(artifact["milestone_archive_summary"]).lower()
    assert archive["exp5151-arc-oracle-distinct-hardening-v472"]["classification"] == (
        "partial_hardening_cross_game_blocked"
    )
    assert archive["exp5151-arc-oracle-distinct-hardening-v472"]["passed_hardening_axes"] == [
        "exact_test",
        "leak_audit",
        "multiseed",
    ]
    assert archive["exp5151-arc-oracle-distinct-hardening-v472"]["open_axis"] == "cross_game"
    assert archive["exp5152-diffusiongemma-gate-reexamination-v472"]["recommendation"] == "keep_gated"
    assert archive["exp5153-gap4-scaleup-v472"]["passed_protocol_steps"] == 1
    assert archive["exp5154-energy-fitness-directed-exploration-v472"]["classification"] == (
        "honest_null_generation_axis"
    )
    assert archive["exp5155-multilevel-belief-state-scoping-v472"]["classification"] == (
        "scoping_complete_code_verified_reset"
    )
    assert artifact["phase_a_followups_from_5155"] == [
        "transition_slice_warm_start_replay_ablation",
        "cross_level_goal_energy_ranker_replay",
    ]
    assert artifact["diffusiongemma_gate_recommendation"] == "keep_gated"
    assert artifact["gap4_status_recommendation"] == "still_open"
    assert artifact["generation_axis_retirement_signal"]["third_consecutive_generation_axis_null"] is True


def test_scenario_report_5156_dirty_runtime_gate_is_visible(tmp_path: Path) -> None:
    """SCENARIO-REPORT-5156-DIRTY-RUNTIME: dirty handoff is recorded as a blocking gate."""

    dirty = mod.RuntimeSnapshot(
        git_status_porcelain=(
            " M ops/status.md\n"
            "?? python/carnot/experiment_5156_archive_472_activate_473.py\n"
            "?? results/experiment_5156_archive_472_activate_473.json\n"
        ),
        process_table=(
            "200 1 Ssl 02:00:00 python scripts/research_conductor.py --loop\n"
            "201 200 Ssl 00:00:59 codex exec --cd /repo -\n"
        ),
    )
    artifact = mod.build_artifact(
        root=make_repo(tmp_path),
        duration_s=1.0,
        run_date="20260702",
        verification=mod.verification_payload(GREEN_VERIFY),
        runtime_snapshot=dirty,
        tests_run=["unit-test-placeholder"],
    )

    mod.validate_artifact(artifact)
    assert artifact["v472_runtime_clean"] is False
    assert artifact["honest_verdict"] == mod.DIRTY_HANDOFF_VERDICT
    assert artifact["runtime_clean_details"]["non_transition_dirty_paths"] == ["ops/status.md"]
    assert artifact["runtime_clean_details"]["ignored_transition_dirty_paths"] == [
        "python/carnot/experiment_5156_archive_472_activate_473.py",
        "results/experiment_5156_archive_472_activate_473.json",
    ]
    assert artifact["runtime_clean_details"]["orphaned_conductor_processes"]


def test_scenario_report_5156_run_preserves_active_roadmap_and_conductor(
    tmp_path: Path,
) -> None:
    """SCENARIO-REPORT-5156: activation records readiness without mutating live files."""

    root = make_repo(tmp_path)
    active_before = (root / "research-roadmap.yaml").read_text(encoding="utf-8")
    conductor_before = (root / "scripts" / "research_conductor.py").read_text(encoding="utf-8")

    output = mod.run(
        root=root,
        run_date="20260702",
        clock=iter([100.0, 101.0]).__next__,
        verification_runner=lambda path: GREEN_VERIFY,
        runtime_probe=lambda repo: clean_runtime_snapshot(),
        tests_run=["unit-test-placeholder"],
    )
    artifact = json.loads(output.read_text(encoding="utf-8"))

    assert artifact["honest_verdict"] == mod.COMPLETE_VERDICT
    assert artifact["active_roadmap_modified"] is False
    assert artifact["conductor_modified"] is False
    assert (root / "research-roadmap.yaml").read_text(encoding="utf-8") == active_before
    assert (root / "scripts" / "research_conductor.py").read_text(
        encoding="utf-8"
    ) == conductor_before
    mod.validate_artifact(artifact)


def test_req_report_5156_validation_edges_and_cli(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-REPORT-5156: validation fails closed and helpers expose readiness gaps."""

    valid = mod.build_artifact(
        root=make_repo(tmp_path / "valid"),
        duration_s=1.0,
        run_date="20260702",
        verification=mod.verification_payload(GREEN_VERIFY),
        runtime_snapshot=clean_runtime_snapshot(),
        tests_run=["unit-test-placeholder"],
    )
    mod.validate_artifact(valid)

    mutations = [
        ("experiment_id", "wrong"),
        ("milestone", "2026.07.472"),
        ("archived_milestone", "2026.07.471"),
        ("honest_verdict", "bad"),
        ("inference_substrate", "live_llm_inference"),
        ("duration_s", 0.0),
        ("source_artifacts_read", []),
        ("task_verdicts", []),
        ("milestone_archive_summary", []),
        ("v472_runtime_clean", "true"),
        ("runtime_clean_details", []),
        ("active_roadmap_ready", "true"),
        ("active_roadmap_modified", True),
        ("conductor_modified", True),
        ("phase_a_followups_from_5155", []),
        ("diffusiongemma_gate_recommendation", ""),
        ("gap4_status_recommendation", ""),
        ("generation_axis_retirement_signal", []),
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
    with pytest.raises(ValueError, match="invalid Exp 5156 archive artifact"):
        mod.validate_artifact(payload)

    payload = copy.deepcopy(valid)
    payload["field_principles"]["tests_run"] = "wrong"
    with pytest.raises(ValueError, match="invalid Exp 5156 archive artifact"):
        mod.validate_artifact(payload)

    assert mod._roadmap_check(tmp_path / "missing.yaml")["ready"] is False
    (tmp_path / "poison.yaml").write_text("a: : :\n- [\n", encoding="utf-8")
    assert mod._roadmap_check(tmp_path / "poison.yaml")["parses"] is False
    assert mod._known_issues_check(tmp_path / "missing.md")["arc_reopened_by_operator_directive"] is False

    not_ready = mod.build_artifact(
        root=make_repo(tmp_path / "not_ready", active_valid=False),
        duration_s=1.0,
        run_date="20260702",
        verification=mod.verification_payload(GREEN_VERIFY),
        runtime_snapshot=clean_runtime_snapshot(),
        tests_run=["unit-test-placeholder"],
    )
    assert not_ready["honest_verdict"] == mod.ACTIVATION_GATED_VERDICT
    mod.validate_artifact(not_ready)

    missing_inputs = mod.build_artifact(
        root=make_repo(tmp_path / "missing_inputs", omit_results={5151}),
        duration_s=1.0,
        run_date="20260702",
        verification=mod.verification_payload(GREEN_VERIFY),
        runtime_snapshot=clean_runtime_snapshot(),
        tests_run=["unit-test-placeholder"],
    )
    assert missing_inputs["honest_verdict"] == mod.MISSING_INPUTS_VERDICT
    mod.validate_artifact(missing_inputs)

    root = make_repo(tmp_path / "cli_repo")
    (root / "scripts" / "adversarial_verify.py").write_text(
        "import json\nprint(json.dumps({'flags': []}))\n",
        encoding="utf-8",
    )
    output = root / "module_cli_result.json"
    monkeypatch.setattr(mod, "capture_runtime_snapshot", lambda repo: clean_runtime_snapshot())
    assert mod.main(["--root", str(root), "--output", str(output), "--date", "20260702"]) == 0
    assert output.exists()
