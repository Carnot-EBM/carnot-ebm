"""Tests for Exp 5122 archive .469 / activate .470 aggregation.

Spec refs: REQ-REPORT-5122, SCENARIO-REPORT-5122,
SCENARIO-REPORT-5122-ACTIVE-FALLBACK.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from carnot import experiment_5122_archive_469_activate_470 as mod


GREEN_VERIFY = mod.CommandResult(
    command=("python", "scripts/adversarial_verify.py"),
    exit_code=0,
    stdout='{"flags":[]}',
    stderr="",
)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _task_yaml(milestone: str = "2026.07.470", *, last: int = 5133) -> str:
    tasks = "\n".join(
        f"- id: exp{exp_id}-task\n  milestone: {milestone}\n  title: task {exp_id}"
        for exp_id in range(5122, last + 1)
    )
    return f"milestone: {milestone}\ntasks:\n{tasks}\n"


def _capstone_payload() -> dict:
    artifacts = [
        (5111, "fover_in_domain_pool", "fover"),
        (5112, "fover_in_domain_selector", "fover"),
        (5114, "kan_abstraction_refinement_post_wall", "kan"),
        (5115, "graph_evidence_fover_transfer", "solver_sampling"),
        (5116, "hubo_2dpt_sampling_reference", "solver_sampling"),
        (5117, "taco_harm_gated_scale", "solver_sampling"),
        (5119, "sota_endpoint_rootcause", "runtime"),
        (5120, "hardware_residual_telemetry", "hardware"),
    ]
    return {
        "experiment_id": "exp5121-capstone-v469",
        "milestone": "2026.07.469",
        "honest_verdict": (
            "complete_capstone_v469_kan_solver_progress_fover_blocked_runtime_flagged_"
            "fr11_gap_hardware_ready"
        ),
        "inference_substrate": "aggregation_from_upstream_artifacts",
        "duration_s": 0.1,
        "flagged_adversarial": False,
        "artifacts_read": [
            {
                "experiment_number": exp_id,
                "label": label,
                "axis": axis,
                "path": f"results/experiment_{exp_id}_{label}_v469.json",
                "exists": True,
                "loadable": True,
                "classification": "clean" if exp_id in {5114, 5116, 5117, 5120} else "blocked",
            }
            for exp_id, label, axis in artifacts
        ],
        "missing_artifacts": [
            {
                "experiment_number": 5113,
                "label": "fover_selector_adversarial_audit",
                "path": "results/experiment_5113_fover_selector_adversarial_audit_v469.json",
                "reason": "preemptive_skip_after_exp5112_retired",
            },
            {
                "experiment_number": 5118,
                "label": "fr11_fover_residual_memory",
                "path": "results/experiment_5118_fr11_fover_residual_memory_v469.json",
                "reason": "preemptive_skip_after_exp5112_retired",
            },
        ],
        "fover_moat_state": {
            "state": "blocked",
            "moat_claim_supported": False,
            "pool_n": 0,
            "headroom_present": False,
            "selector_ran": False,
            "audit_ran": False,
            "decision_reason": "FoVer pool premise was retracted.",
            "corrected_result_summary": {
                "verifier_auroc": 0.9663,
                "cheap_baseline_auroc": 0.9635,
                "delta_auroc": 0.0028,
                "delta_auroc_ci95": [-0.0244, 0.0347],
                "beats_cheap_baseline": False,
            },
        },
        "kan_post_wall_state": {
            "state": "clean_positive",
            "post_wall_progress": True,
            "solved_n": 100,
            "exp5108_largest_n_reached": 10,
        },
        "solver_sampling_state": {
            "state": "clean_positive",
            "hubo_2dpt_reference_ready": True,
            "taco_harm_gate_ready": True,
            "fover_transfer_gap_present": True,
        },
        "fr11_state": {
            "state": "blocked",
            "artifact_missing": True,
            "gap_reason": "preemptive_skip_after_exp5112_retired",
            "promotion_safe": False,
        },
        "runtime_state": {
            "state": "flagged",
            "quarantined": True,
            "adversarial_verify_passed": False,
            "cache_ready": False,
        },
        "hardware_state": {
            "state": "clean_positive",
            "hardware_residual_telemetry_ready": True,
            "kv260_ssh_ready": True,
            "polarfire_ssh_ready": True,
            "no_speedup_claim": True,
        },
        "next_milestone_recommendations": [
            {
                "priority": "Retire same-verdict FoVer in-domain selector/audit/FR-11 reruns",
                "rationale": "The FoVer pool retraction makes the current path a doomed rerun.",
                "retire_same_verdict_doomed_rerun": True,
            }
        ],
        "active_roadmap_modified": False,
        "conductor_modified": False,
        "tests_run": ["capstone unit"],
        "reproducibility_checksum": "sha256:" + "0" * 64,
    }


def make_repo(tmp_path: Path, *, next_present: bool = True, active_valid: bool = True) -> Path:
    root = tmp_path
    (root / "results").mkdir(parents=True, exist_ok=True)
    (root / "openspec" / "change-proposals").mkdir(parents=True, exist_ok=True)
    (root / "scripts").mkdir(parents=True, exist_ok=True)
    (root / "scripts" / "research_conductor.py").write_text("# conductor\n", encoding="utf-8")
    (root / "openspec" / "change-proposals" / "research-roadmap-vNEXT.md").write_text(
        "# Research Roadmap vNEXT: 2026.07.470\n\nexp5122 through exp5133.\n",
        encoding="utf-8",
    )
    if next_present:
        (root / "research-roadmap-next.yaml").write_text(_task_yaml(), encoding="utf-8")
    (root / "research-roadmap.yaml").write_text(
        _task_yaml() if active_valid else _task_yaml("2026.07.469", last=5124),
        encoding="utf-8",
    )
    capstone = _capstone_payload()
    _write_json(root / mod.CAPSTONE_RELATIVE_PATH, capstone)
    for row in capstone["artifacts_read"]:
        payload = {
            "experiment_id": f"exp{row['experiment_number']}",
            "honest_verdict": "complete_placeholder",
            "flagged_adversarial": row["experiment_number"] == 5119,
            "duration_s": 0.2,
            "inference_substrate": "aggregation_from_upstream_artifacts",
        }
        _write_json(root / row["path"], payload)
    return root


def test_req_report_5122_spec_declares_archive_contract() -> None:
    """REQ-REPORT-5122: OpenSpec anchors the .469 archive and .470 activation contract."""

    spec = Path("openspec/capabilities/research-reporting/spec.md").read_text(encoding="utf-8")

    for marker in (
        "REQ-REPORT-5122",
        "SCENARIO-REPORT-5122",
        "SCENARIO-REPORT-5122-ACTIVE-FALLBACK",
        "results/experiment_5122_archive_469_activate_470.json",
        "fover_selector_retired_for_same_verdict",
    ):
        assert marker in spec
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert f"`{field}`" in spec
        assert principle in spec


def test_scenario_report_5122_happy_path_records_close_state_and_next_roadmap(
    tmp_path: Path,
) -> None:
    """SCENARIO-REPORT-5122: next-roadmap readiness and .469 close-state are preserved."""

    root = make_repo(tmp_path)
    artifact = mod.build_artifact(
        root=root,
        duration_s=1.25,
        run_date="20260701",
        verification=mod.verification_payload(GREEN_VERIFY),
        tests_run=["unit-test-placeholder"],
    )

    mod.validate_artifact(artifact)
    assert artifact["experiment_id"] == "exp5122-archive-469-activate-470"
    assert artifact["milestone"] == "2026.07.470"
    assert artifact["honest_verdict"] == mod.COMPLETE_VERDICT
    assert artifact["inference_substrate"] == "aggregation_from_upstream_artifacts"
    assert artifact["roadmap_next_present"] is True
    assert artifact["roadmap_next_check"]["required_task_ids_present"] is True
    assert artifact["active_roadmap_modified"] is False
    assert artifact["conductor_modified"] is False
    assert artifact["fover_selector_retired_for_same_verdict"] is True
    assert artifact["fover_retirement"]["fover_residual_fr11_should_not_rerun"] is True
    assert artifact["kan_post_wall_state"]["state"] == "clean_positive"
    assert artifact["solver_sampling_state"]["state"] == "clean_positive"
    assert artifact["runtime_state"]["state"] == "flagged"
    assert artifact["hardware_state"]["no_speedup_claim"] is True
    assert any(row["source_id"] == "exp5121-capstone-v469" for row in artifact["source_artifacts_read"])


def test_scenario_report_5122_active_fallback_does_not_recreate_next_roadmap(
    tmp_path: Path,
) -> None:
    """SCENARIO-REPORT-5122-ACTIVE-FALLBACK: activated .470 is enough without mutation."""

    root = make_repo(tmp_path, next_present=False)
    active_before = (root / "research-roadmap.yaml").read_text(encoding="utf-8")
    conductor_before = (root / "scripts" / "research_conductor.py").read_text(encoding="utf-8")

    output = mod.run(
        root=root,
        run_date="20260701",
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
    assert (root / "scripts" / "research_conductor.py").read_text(encoding="utf-8") == conductor_before
    mod.validate_artifact(artifact)


def test_scenario_report_5122_blocks_when_no_next_or_active_470(tmp_path: Path) -> None:
    """REQ-REPORT-5122: missing next roadmap blocks if active `.470` fallback is absent."""

    root = make_repo(tmp_path, next_present=False, active_valid=False)
    artifact = mod.build_artifact(
        root=root,
        duration_s=1.0,
        run_date="20260701",
        verification=mod.verification_payload(GREEN_VERIFY),
        tests_run=["unit-test-placeholder"],
    )

    assert artifact["honest_verdict"] == "blocked_research_roadmap_next_missing"
    assert artifact["roadmap_next_present"] is False
    assert artifact["active_roadmap_fallback_used"] is False
    mod.validate_artifact(artifact)


def test_req_report_5122_validation_rejects_schema_drift(tmp_path: Path) -> None:
    """REQ-REPORT-5122: schema validation fails closed on required-field drift."""

    valid = mod.build_artifact(
        root=make_repo(tmp_path),
        duration_s=1.0,
        run_date="20260701",
        verification=mod.verification_payload(GREEN_VERIFY),
        tests_run=["unit-test-placeholder"],
    )
    mod.validate_artifact(valid)

    mutations = [
        ("experiment_id", "wrong"),
        ("milestone", "2026.07.469"),
        ("honest_verdict", "bad"),
        ("inference_substrate", "live_llm_inference"),
        ("duration_s", 0.0),
        ("source_artifacts_read", []),
        ("fover_selector_retired_for_same_verdict", False),
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

    for section, key in (("fover_retirement", "fover_residual_fr11_should_not_rerun"), ("hardware_state", "no_speedup_claim")):
        payload = copy.deepcopy(valid)
        payload[section][key] = False
        payload["reproducibility_checksum"] = mod.payload_checksum(payload)
        with pytest.raises(ValueError):
            mod.validate_artifact(payload)


def test_req_report_5122_helper_edges_and_script_entrypoint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-REPORT-5122: helpers parse edge cases and the requested script delegates."""

    assert mod._number(True) is None
    assert mod._number("bad") is None
    assert mod._task_prefixes_present(["exp5122-a"], ["exp5122", "exp5123"]) is False
    assert mod.read_json_mapping(tmp_path / "missing.json")[1]["error"] == "missing"
    assert mod._roadmap_check(tmp_path / "missing.yaml")["milestone"] == "missing"
    (tmp_path / "poison.yaml").write_text("a: : :\n- [\n", encoding="utf-8")
    assert mod._roadmap_check(tmp_path / "poison.yaml")["parses"] is False
    assert mod.build_source_artifacts_read(
        tmp_path,
        {
            "artifacts_read": [
                {"path": str(mod.CAPSTONE_RELATIVE_PATH), "experiment_number": 5121},
                {"path": "", "experiment_number": 9999},
            ]
        },
    )[0]["source_id"] == "exp5121-capstone-v469"
    assert mod.load_referenced_payloads(
        tmp_path,
        {"artifacts_read": [{"path": "missing.json", "experiment_number": "bad"}]},
    ) == {}
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
    assert mod._verification_flags(
        mod.CommandResult(command=(), exit_code=0, stdout="not-json", stderr="")
    ) == []
    assert mod.verification_payload(
        mod.CommandResult(
            command=(),
            exit_code=0,
            stdout='{"flags":[{"severity":"critical","kind":"X"}]}',
            stderr="",
        )
    )["flagged_adversarial"] is True

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

    import scripts.experiment_5122_archive_469_activate_470 as entrypoint

    repo = make_repo(tmp_path / "entrypoint_repo", next_present=False)
    monkeypatch.setattr(entrypoint, "run_main", lambda **kwargs: repo / mod.RESULT_RELATIVE_PATH)
    assert entrypoint.main(root=repo, date="20260701") == repo / mod.RESULT_RELATIVE_PATH

    module_cli_repo = make_repo(tmp_path / "module_cli_repo", next_present=False)
    module_cli_output = module_cli_repo / "module_cli_result.json"
    assert mod.main(["--root", str(module_cli_repo), "--output", str(module_cli_output), "--date", "20260701"]) == 0
    assert module_cli_output.exists()
