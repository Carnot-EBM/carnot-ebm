"""Tests for Exp 5150 archive .471 / activate .472 aggregation.

Spec refs: REQ-REPORT-5150, SCENARIO-REPORT-5150,
SCENARIO-REPORT-5150-DIRTY-RUNTIME.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from carnot import experiment_5150_archive_471_activate_472 as mod


GREEN_VERIFY = mod.CommandResult(
    command=("python", "scripts/adversarial_verify.py"),
    exit_code=0,
    stdout='{"flags":[]}',
    stderr="",
)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _task_yaml(milestone: str = "2026.07.472", *, last: int = 5155) -> str:
    tasks = "\n".join(
        f"- id: exp{exp_id}-task\n  milestone: {milestone}\n  title: task {exp_id}"
        for exp_id in range(5150, last + 1)
    )
    return f"milestone: {milestone}\ntasks:\n{tasks}\n"


def _capstone_payload() -> dict:
    upstreams = [
        (5134, "archive_470_activate_471", "transition", "quarantined", True),
        (5135, "v471_source_scope_audit", "planning", "quarantined", True),
        (5136, "receipt_structured_pool_v2", "structured_generation", "clean", False),
        (5137, "solver_verified_formulation_selector", "solver_formulation", "clean", False),
        (5138, "ets_ebd_guided_decoding", "guided_decoding", "blocked", False),
        (5139, "abstention_verification_trace", "abstention_trace", "clean", False),
        (5140, "symbolic_kan_certificate_distillation", "kan_symbolic", "clean", False),
        (5141, "hubo_partition_residual_exponent", "sampling_partition", "clean", False),
        (5142, "taco_harm_rootcause_scale", "taco_harm", "clean", False),
        (5143, "openskill_k2v_self_learning", "fr11", "quarantined", True),
        (5144, "authenticated_board_workload", "hardware", "blocked", False),
    ]
    verdicts = {
        5134: "complete_archive_470_closed_471_active_roadmap_ready",
        5135: "complete_v471_source_scope_audit_clean",
        5136: "complete_receipt_structured_pool_v2_clean",
        5137: "complete_formulation_selector_evaluated_no_utility_beyond_static",
        5138: "blocked_stepwise_logprob_telemetry_unavailable",
        5139: "complete_verification_trace_ready",
        5140: "success_symbolic_kan_certificate_distillation_ready",
        5141: "complete_partition_telemetry_ready_exact_checked_cpu_no_speedup",
        5142: "success_trace_suite_v2_ready_harm_gate_repaired_exact_labels_preserved",
        5143: "success_openskill_k2v_verifier_anchors_promoted_exact_gates_pass",
        5144: "blocked_no_safe_board_workload_manifest_no_speedup_claim",
    }
    classified = [
        {
            "experiment_number": exp_id,
            "label": label,
            "axis": axis,
            "classification": classification,
            "flagged_adversarial": flagged,
            "honest_verdict": verdicts[exp_id],
            "relative_path": f"results/experiment_{exp_id}_{label}_v471.json",
            "sha256": "sha256:" + f"{exp_id:064d}"[-64:],
        }
        for exp_id, label, axis, classification, flagged in upstreams
    ]
    return {
        "experiment_id": "exp5145-capstone-v471",
        "milestone": "2026.07.471",
        "honest_verdict": (
            "complete_capstone_v471_structured_pool_repaired_solver_no_utility_"
            "guided_blocked_fr11_quarantined_hardware_blocked"
        ),
        "inference_substrate": "aggregation_from_v471_artifacts",
        "duration_s": 0.1,
        "flagged_adversarial": False,
        "classified_upstreams": classified,
        "upstream_artifacts_read": classified,
        "missing_artifacts": [],
        "source_scope_audit_state": {
            "classification": "quarantined",
            "honest_verdict": "complete_v471_source_scope_audit_clean",
            "quarantine_reason": "flagged_adversarial",
        },
        "structured_generation_state": {
            "classification": "clean",
            "downstream_tasks_trustworthy": True,
            "pool_n": 120,
            "honest_verdict": "complete_receipt_structured_pool_v2_clean",
        },
        "solver_formulation_state": {
            "classification": "no-promote",
            "selector_delta_vs_best_static": 0.0,
            "honest_verdict": "complete_formulation_selector_evaluated_no_utility_beyond_static",
        },
        "guided_decoding_state": {
            "classification": "blocked",
            "guided_decoding_ready": False,
            "honest_verdict": "blocked_stepwise_logprob_telemetry_unavailable",
        },
        "abstention_trace_state": {
            "classification": "clean",
            "verification_trace_ready": True,
            "honest_verdict": "complete_verification_trace_ready",
        },
        "kan_symbolic_state": {
            "classification": "clean",
            "symbolic_kan_ready": True,
            "certificate_soundness": True,
            "honest_verdict": "success_symbolic_kan_certificate_distillation_ready",
        },
        "sampling_partition_state": {
            "classification": "clean",
            "partition_telemetry_ready": True,
            "hardware_speedup_claimed": False,
            "honest_verdict": "complete_partition_telemetry_ready_exact_checked_cpu_no_speedup",
        },
        "taco_harm_state": {
            "classification": "clean",
            "trace_suite_v2_ready": True,
            "wrong_label_count": 0,
            "honest_verdict": "success_trace_suite_v2_ready_harm_gate_repaired_exact_labels_preserved",
        },
        "fr11_state": {
            "classification": "quarantined",
            "promotion_safe": True,
            "quarantine_reason": "flagged_adversarial",
            "honest_verdict": "success_openskill_k2v_verifier_anchors_promoted_exact_gates_pass",
        },
        "hardware_state": {
            "classification": "blocked",
            "hardware_workload_transcripts_ready": False,
            "no_speedup_claim": True,
            "honest_verdict": "blocked_no_safe_board_workload_manifest_no_speedup_claim",
        },
        "next_milestone_recommendations": [
            {
                "priority": "critical",
                "task": "archive_v471_before_v472",
                "recommendation": "Record quarantined and blocked axes before using V471.",
            }
        ],
        "retire_or_quarantine_recommendations": [
            {
                "action": "block_until_prerequisite_changes",
                "experiment": "exp5138",
                "reason": "stepwise logprob telemetry is unavailable",
            }
        ],
        "no_speedup_claim_preserved": True,
        "tests_run": ["capstone unit"],
        "reproducibility_checksum": "sha256:" + "0" * 64,
    }


def clean_runtime_snapshot() -> mod.RuntimeSnapshot:
    return mod.RuntimeSnapshot(
        git_status_porcelain="",
        process_table=(
            "100 42 Ssl 03:50:42 python scripts/research_conductor.py --loop\n"
            "101 100 Ssl 00:00:59 codex exec --cd /repo -\n"
        ),
    )


def make_repo(
    tmp_path: Path,
    *,
    active_valid: bool = True,
    research_complete_has_v471: bool = True,
    known_issue_has_directive: bool = True,
    claude_has_retired: bool = True,
    registry_total: int | str = 69,
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
        "reproducible_total_levels has been flat at 69 since the 2026-06-30 pivot\n"
        if known_issue_has_directive
        else "### Different note\n"
    )
    (root / "ops" / "known-issues.md").write_text(known_issue, encoding="utf-8")
    claude = (
        "## ARC-AGI-3 Submission Sprint Forcing Function "
        "(RETIRED 2026-06-30 -- preserved per never-prune)\n"
        if claude_has_retired
        else "## ARC-AGI-3 Submission Sprint Forcing Function\n"
    )
    (root / "CLAUDE.md").write_text(claude, encoding="utf-8")
    (root / "ops" / "arc_solve_registry.yaml").write_text(
        f"schema_version: 1\nreproducible_total_levels: {registry_total}\n",
        encoding="utf-8",
    )
    (root / "research-roadmap.yaml").write_text(
        _task_yaml() if active_valid else _task_yaml("2026.07.471", last=5151),
        encoding="utf-8",
    )
    milestone_rows = (
        "- id: 2026.07.471\n  title: archived\n  tasks: []\n"
        if research_complete_has_v471
        else "- id: 2026.07.470\n  title: archived\n  tasks: []\n"
    )
    (root / "research-complete.yaml").write_text(f"milestones:\n{milestone_rows}", encoding="utf-8")
    capstone = _capstone_payload()
    _write_json(root / mod.CAPSTONE_RELATIVE_PATH, capstone)
    for row in capstone["classified_upstreams"]:
        _write_json(
            root / row["relative_path"],
            {
                "experiment_id": f"exp{row['experiment_number']}",
                "honest_verdict": row["honest_verdict"],
                "flagged_adversarial": row["flagged_adversarial"],
                "duration_s": 0.2,
                "inference_substrate": "aggregation_from_upstream_artifacts",
            },
        )
    return root


def test_req_report_5150_spec_declares_archive_contract() -> None:
    """REQ-REPORT-5150: OpenSpec anchors the .471 archive and .472 activation contract."""

    spec = Path("openspec/capabilities/research-reporting/spec.md").read_text(encoding="utf-8")

    for marker in (
        "REQ-REPORT-5150",
        "SCENARIO-REPORT-5150",
        "SCENARIO-REPORT-5150-DIRTY-RUNTIME",
        "results/experiment_5150_archive_471_activate_472.json",
        "v471_runtime_clean",
        "ENERGY-BASED ARC RESEARCH LINEUP 2026-07-02",
    ):
        assert marker in spec
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert f"`{field}`" in spec
        assert principle in spec


def test_scenario_report_5150_happy_path_records_archive_and_arc_reopening(
    tmp_path: Path,
) -> None:
    """SCENARIO-REPORT-5150: .471 truth and clean .472 activation are preserved."""

    artifact = mod.build_artifact(
        root=make_repo(tmp_path),
        duration_s=1.25,
        run_date="20260702",
        verification=mod.verification_payload(GREEN_VERIFY),
        runtime_snapshot=clean_runtime_snapshot(),
        tests_run=["unit-test-placeholder"],
    )

    mod.validate_artifact(artifact)
    assert artifact["experiment_id"] == "exp5150-archive-471-activate-472"
    assert artifact["milestone"] == "2026.07.472"
    assert artifact["archived_milestone"] == "2026.07.471"
    assert artifact["honest_verdict"] == mod.COMPLETE_VERDICT
    assert artifact["inference_substrate"] == "aggregation_from_upstream_artifacts"
    assert artifact["v471_runtime_clean"] is True
    assert artifact["runtime_clean_details"]["non_transition_dirty_paths"] == []
    assert artifact["runtime_clean_details"]["orphaned_conductor_processes"] == []
    assert artifact["arc_reopened_by_operator_directive"] is True
    assert artifact["sprint_forcing_function_retired_preserved"] is True
    assert artifact["reproducible_total_levels"] == 69
    assert artifact["active_roadmap_ready"] is True
    assert artifact["active_roadmap_modified"] is False
    assert artifact["conductor_modified"] is False
    assert artifact["v471_capstone_verdict"].startswith("complete_capstone_v471")
    assert artifact["v471_guided_decoding_blocked"] is True
    assert artifact["v471_fr11_quarantined"] is True
    assert artifact["v471_hardware_blocked_no_speedup"] is True
    assert len(artifact["task_verdicts"]) == 12
    assert any(row["experiment_id"] == "exp5145-capstone-v471" for row in artifact["task_verdicts"])


def test_scenario_report_5150_dirty_runtime_gate_is_visible(tmp_path: Path) -> None:
    """SCENARIO-REPORT-5150-DIRTY-RUNTIME: dirty handoff is recorded as a blocking gate."""

    dirty = mod.RuntimeSnapshot(
        git_status_porcelain=(
            " M ops/status.md\n"
            "?? notes.txt\n"
            "?? python/carnot/experiment_5150_archive_471_activate_472.py\n"
            "?? results/experiment_5150_archive_471_activate_472.json\n"
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
    assert artifact["v471_runtime_clean"] is False
    assert artifact["honest_verdict"] == mod.DIRTY_HANDOFF_VERDICT
    assert artifact["runtime_clean_details"]["non_transition_dirty_paths"] == [
        "ops/status.md",
        "notes.txt",
    ]
    assert artifact["runtime_clean_details"]["ignored_transition_dirty_paths"] == [
        "python/carnot/experiment_5150_archive_471_activate_472.py",
        "results/experiment_5150_archive_471_activate_472.json",
    ]
    assert artifact["runtime_clean_details"]["orphaned_conductor_processes"]


def test_scenario_report_5150_run_preserves_active_roadmap_and_conductor(
    tmp_path: Path,
) -> None:
    """SCENARIO-REPORT-5150: activation records readiness without mutating live files."""

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


def test_req_report_5150_blocks_when_active_472_missing(tmp_path: Path) -> None:
    """REQ-REPORT-5150: missing activated `.472` roadmap readiness is visible."""

    artifact = mod.build_artifact(
        root=make_repo(tmp_path, active_valid=False),
        duration_s=1.0,
        run_date="20260702",
        verification=mod.verification_payload(GREEN_VERIFY),
        runtime_snapshot=clean_runtime_snapshot(),
        tests_run=["unit-test-placeholder"],
    )

    assert artifact["honest_verdict"] == "blocked_active_roadmap_not_ready"
    assert artifact["active_roadmap_ready"] is False
    mod.validate_artifact(artifact)


def test_req_report_5150_validation_rejects_schema_drift(tmp_path: Path) -> None:
    """REQ-REPORT-5150: schema validation fails closed on required-field drift."""

    valid = mod.build_artifact(
        root=make_repo(tmp_path),
        duration_s=1.0,
        run_date="20260702",
        verification=mod.verification_payload(GREEN_VERIFY),
        runtime_snapshot=clean_runtime_snapshot(),
        tests_run=["unit-test-placeholder"],
    )
    mod.validate_artifact(valid)

    mutations = [
        ("experiment_id", "wrong"),
        ("milestone", "2026.07.471"),
        ("archived_milestone", "2026.07.470"),
        ("honest_verdict", "bad"),
        ("inference_substrate", "live_llm_inference"),
        ("duration_s", 0.0),
        ("source_artifacts_read", []),
        ("task_verdicts", []),
        ("capstone_summary", []),
        ("v471_runtime_clean", "true"),
        ("runtime_clean_details", []),
        ("arc_reopened_by_operator_directive", "true"),
        ("sprint_forcing_function_retired_preserved", "true"),
        ("reproducible_total_levels", "69"),
        ("active_roadmap_ready", "true"),
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


def test_req_report_5150_helper_edges_and_script_entrypoint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-REPORT-5150: helpers parse edge cases and the requested script delegates."""

    assert mod._number(True) is None
    assert mod._number("bad") is None
    assert mod._task_prefixes_present(["exp5150-a"], ["exp5150", "exp5151"]) is False
    assert mod.read_json_mapping(tmp_path / "missing.json")[1]["error"] == "missing"
    (tmp_path / "bad.json").write_text("{", encoding="utf-8")
    assert mod.read_json_mapping(tmp_path / "bad.json")[1]["loadable"] is False
    (tmp_path / "list.json").write_text("[]", encoding="utf-8")
    assert mod.read_json_mapping(tmp_path / "list.json")[1]["error"] == "json_not_object"
    assert mod._roadmap_check(tmp_path / "missing.yaml")["milestone"] == "missing"
    (tmp_path / "poison.yaml").write_text("a: : :\n- [\n", encoding="utf-8")
    assert mod._roadmap_check(tmp_path / "poison.yaml")["parses"] is False
    assert mod._research_complete_check(tmp_path / "missing.yaml")["parses"] is False
    assert mod._research_complete_check(tmp_path / "poison.yaml")["parses"] is False
    (tmp_path / "missing_v471.yaml").write_text(
        "milestones:\n- id: 2026.07.470\n",
        encoding="utf-8",
    )
    assert (
        mod._research_complete_check(tmp_path / "missing_v471.yaml")["ledger_gap"]
        == "missing_2026.07.471"
    )
    (tmp_path / "duplicate_complete.yaml").write_text(
        "milestones:\n- id: 2026.07.471\n- id: 2026.07.471\n",
        encoding="utf-8",
    )
    assert (
        mod._research_complete_check(tmp_path / "duplicate_complete.yaml")["ledger_gap"]
        == "duplicate_2026.07.471_entries"
    )
    assert mod._registry_total_levels(tmp_path / "missing.yaml") is None
    assert mod._registry_total_levels(tmp_path / "poison.yaml") is None
    bad_total = make_repo(tmp_path / "bad_total", registry_total="bad")
    assert mod._registry_total_levels(bad_total / "ops" / "arc_solve_registry.yaml") is None
    assert mod._dirty_paths("\nR  old.py -> new.py\n?? short\n") == ["new.py", "short"]
    assert mod._process_row("too short") is None
    assert (
        mod.build_source_artifacts_read(
            tmp_path,
            {
                "upstream_artifacts_read": [
                    {"relative_path": str(mod.CAPSTONE_RELATIVE_PATH), "experiment_number": 5145},
                    {"relative_path": "", "experiment_number": 9999},
                ]
            },
        )[0]["source_id"]
        == "exp5145-capstone-v471"
    )
    assert (
        mod.load_referenced_payloads(
            tmp_path,
            {"upstream_artifacts_read": [{"relative_path": "missing.json", "experiment_number": "bad"}]},
        )
        == {}
    )
    base_preconditions = {
        "capstone": {"loadable": True},
        "known_issues": {"arc_reopened_by_operator_directive": True},
        "claude": {"sprint_forcing_function_retired_preserved": True},
        "active_roadmap": {"ready": True},
    }
    cases = [
        (
            {**base_preconditions, "capstone": {"loadable": False}},
            "blocked_capstone_artifact_missing_or_unloadable",
        ),
        (
            {**base_preconditions, "known_issues": {"arc_reopened_by_operator_directive": False}},
            "blocked_arc_reopen_directive_missing",
        ),
        (
            {
                **base_preconditions,
                "claude": {"sprint_forcing_function_retired_preserved": False},
            },
            "blocked_retired_sprint_context_missing",
        ),
        (
            {**base_preconditions, "active_roadmap": {"ready": False}},
            "blocked_active_roadmap_not_ready",
        ),
        (base_preconditions, mod.COMPLETE_VERDICT),
    ]
    for preconditions, expected in cases:
        assert mod._honest_verdict(preconditions, runtime_clean=True) == expected
    assert mod._honest_verdict(base_preconditions, runtime_clean=False) == mod.DIRTY_HANDOFF_VERDICT
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
    warn_only = mod.verification_payload(
        mod.CommandResult(
            command=(),
            exit_code=1,
            stdout='{"reports":[{"flags":[{"severity":"warn","kind":"WARN_ONLY"}]}]}',
            stderr="",
        )
    )
    assert warn_only["green"] is False
    assert warn_only["max_severity"] == 1
    assert warn_only["flagged_adversarial"] is False
    assert mod._verification_flags(
        mod.CommandResult(
            command=(),
            exit_code=1,
            stdout='{"reports":[{"flags":[{"severity":"info","kind":"Y"}]}]}',
            stderr="",
        )
    ) == [{"severity": "info", "kind": "Y"}]
    captured = mod.capture_runtime_snapshot(tmp_path)
    assert isinstance(captured.git_status_porcelain, str)
    assert isinstance(captured.process_table, str)

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

    import scripts.experiment_5150_archive_471_activate_472 as entrypoint

    repo = make_repo(tmp_path / "entrypoint_repo")
    monkeypatch.setattr(entrypoint, "run_main", lambda **kwargs: repo / mod.RESULT_RELATIVE_PATH)
    assert entrypoint.main(root=repo, date="20260702") == repo / mod.RESULT_RELATIVE_PATH

    import runpy
    import sys

    script_path = Path("scripts/experiment_5150_archive_471_activate_472.py").resolve()
    python_dir = str(script_path.parents[1] / "python")
    monkeypatch.setattr(sys, "path", [entry for entry in sys.path if entry != python_dir])
    runpy.run_path(str(script_path), run_name="exp5150_entrypoint_import_guard")
    assert python_dir in sys.path

    module_cli_repo = make_repo(tmp_path / "module_cli_repo")
    module_cli_output = module_cli_repo / "module_cli_result.json"
    (module_cli_repo / "scripts" / "adversarial_verify.py").write_text(
        "import json\nprint(json.dumps({'flags': []}))\n", encoding="utf-8"
    )
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
