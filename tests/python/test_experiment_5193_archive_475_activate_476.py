"""Tests for Exp 5193 archive .475 / activate .476 aggregation.

Spec refs: REQ-REPORT-5193, SCENARIO-REPORT-5193,
SCENARIO-REPORT-5193-BLOCKED-PRECONDITION.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
import yaml

from carnot import experiment_5193_archive_475_activate_476 as mod


CLEAN_PUBLICATION_GATE = {
    "paper_ready": True,
    "unmet_gates": [],
    "gates": {"G1": {"pass": True}, "G2": {"pass": True}, "G3": {"pass": True}, "G4": {"pass": True}},
}

CLEAN_LINT = mod.CommandResult(
    command=(".venv/bin/python", "scripts/exclusion_manifest_lint.py", "research-roadmap.yaml"),
    exit_code=0,
    stdout=(
        "Exclusion-manifest lint found 2 violation(s) in research-roadmap.yaml:\n"
        "WARNING violations (2, override present):\n"
        "All violations have operator_override -- activation would proceed with warnings.\n"
    ),
    stderr="",
)


def _wrapped(value: object) -> dict[str, object]:
    return {"principle": "test principle", "value": value}


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _exp5181() -> dict:
    return {
        "experiment_id": "exp5181-archive-474-activate-475",
        "milestone": "2026.07.475",
        "honest_verdict": {
            "principle": "Must start with complete:/complete_/success:/success_.",
            "value": "complete_archive_474_closed_475_active_precise_handoff_clean",
        },
        "duration_s": 2.4946055188775063,
        "flagged_adversarial": True,
        "adversarial_flags": ["DURATION_TOO_SHORT"],
        "inference_substrate": _wrapped("aggregation_from_upstream_artifacts"),
        "v474_task_rows": [
            {
                "exp_id": 5173,
                "key_facts": {"smoke_error": "GGUF and CUDA cited in upstream field only"},
            }
        ],
    }


def _exp5182() -> dict:
    oom = (
        "OutOfMemoryError: CUDA out of memory. GPU 0 has a total capacity of 23.56 GiB; "
        "this process has 22.62 GiB memory in use."
    )
    return {
        "experiment": "experiment_5182_diffusiongemma_meta_tensor_rootcause_fix_v475",
        "experiment_id": "experiment_5182_diffusiongemma_meta_tensor_rootcause_fix_v475",
        "milestone": "2026.07.475",
        "duration_s": 630.334,
        "diffusiongemma_loadable": False,
        "forward_pass_confirmed": False,
        "honest_verdict": "blocked_diffusiongemma_meta_tensor_bug_unresolved_v475",
        "inference_substrate": "live_llm_inference",
        "nf4_footprint_gib": 12.864,
        "mitigations_tried": [
            {"mitigation": "m1_single_device_gpu0_4bit_nf4", "outcome": "load_failed", "error_if_any": oom, "duration_s": 188.624},
            {
                "mitigation": "m2_auto_explicit_no_split_4bit_nf4",
                "outcome": "load_failed",
                "error_if_any": "ValueError: Some modules are dispatched on the CPU or the disk.",
                "duration_s": 137.103,
            },
            {"mitigation": "m3_single_device_gpu0_4bit_low_cpu_mem_false", "outcome": "load_failed", "error_if_any": oom, "duration_s": 149.798},
            {"mitigation": "m4_single_device_gpu0_int8", "outcome": "load_failed", "error_if_any": oom, "duration_s": 149.024},
        ],
        "preconditions_checked": [
            {"resource": "gpu_free_for_4bit_load", "available": True, "detail": "gpu0: 23.3/23.6 GiB free"},
            {"resource": "diffusiongemma_weights_cached", "available": True, "detail": "11 shards"},
        ],
        "root_cause": (
            "DiffusionGemma's encoder is a weight-tied mirror of its decoder. device_map='auto' "
            "splits encoder and decoder across the two GPUs, breaking the shared-storage tie. "
            "Single-device placement (device_map={'': 0}) co-locates the tied weights."
        ),
    }


def _conductor_log() -> str:
    refusal_rows = "\n".join(
        f"| 2026-07-03 03:{15 + idx:02d} UTC | Activation REFUSED: milestone 2026.07.475 | BLOCK | exclusion-manifest: 1 HARD violation(s); first: SCOPE_MATCHED_PRIOR_FAILURE on exp5181 |"
        for idx in range(3)
    )
    return "\n".join(
        [
            refusal_rows,
            "| 2026-07-03 05:25 UTC | Activation REFUSED: milestone 2026.07.475 | BLOCK | exclusion-manifest: 1 HARD violation(s); first: SCOPE_MATCHED_PRIOR_FAILURE on exp5181 |",
            "| 2026-07-03 06:14 UTC | Plan next milestone | FAIL | Codex CLI error: Error: Reached max turns (50) |",
            "| 2026-07-03 07:05 UTC | Plan next milestone | FAIL | Codex CLI error: Wall-clock+idle timeout after 1201s (1201s silence). Last ou |",
            "| 2026-07-03 07:56 UTC | Plan next milestone | FAIL | Codex CLI error: Wall-clock+idle timeout after 1201s (1201s silence). Last ou |",
            "| 2026-07-03 07:59 UTC | Milestone 2026.07.475 activated | OK | 12 tasks queued |",
            "| 2026-07-03 08:15 UTC | PHASE 0 transition -- archive .474 truth | FLAGGED | adversarial_verify CRITICAL: DURATION_TOO_SHORT -- result quarantined |",
            "| 2026-07-03 08:39 UTC | PHASE A1 root-cause and fix the DiffusionGemma two | FAIL | Claude Code error: Wall-clock+idle timeout after 1201s (1201s silence). Last ou |",
            "| 2026-07-03 08:42 UTC | PHASE A1 root-cause and fix the DiffusionGemma two | SKIP | Pre-tests failing, self-heal failed: 1 failed, 116 passed, 15 warnings in 12.58s |",
            "| 2026-07-03 08:46 UTC | PHASE A2 the actual DiffusionGemma energy-guided-v | GATE_BLOCK | Pre-emptive skip: upstream retired (exp5182-diffusiongemma-meta-tensor-rootcause |",
            "| 2026-07-03 08:47 UTC | PHASE A3 continue the GAP-4 forward-protocol scale | SKIP | Pre-tests failing, self-heal failed: 1 failed, 116 passed, 15 warnings in 11.94s |",
            "| 2026-07-03 09:42 UTC | PHASE Z capstone -- reconcile the DiffusionGemma u | SKIP | Pre-tests failing, self-heal failed: 1 failed, 116 passed, 15 warnings in 11.58s |",
            "| 2026-07-03 10:35 UTC | PHASE A1 root-cause and fix the DiffusionGemma two | OK | Deliverable already exists in repo |",
            "| 2026-07-03 11:43 UTC | Plan next milestone | FAIL | Codex CLI error: Wall-clock+idle timeout after 1201s (1201s silence). Last ou |",
        ]
    )


def make_repo(
    tmp_path: Path,
    *,
    active_roadmap: bool = True,
    omit_artifact: int | None = None,
    conductor_modified: bool = False,
    diffusion_retired: bool = False,
) -> Path:
    root = tmp_path
    (root / "results").mkdir(parents=True, exist_ok=True)
    (root / "ops").mkdir(parents=True, exist_ok=True)
    (root / "_bmad").mkdir(parents=True, exist_ok=True)
    (root / "openspec/change-proposals").mkdir(parents=True, exist_ok=True)
    (root / "scripts").mkdir(parents=True, exist_ok=True)
    conductor_text = "# conductor\n"
    if conductor_modified:
        conductor_text += "# modified during task\n"
    (root / "scripts" / "research_conductor.py").write_text(conductor_text, encoding="utf-8")
    (root / "ops" / "conductor-log.md").write_text(_conductor_log(), encoding="utf-8")
    (root / "openspec/change-proposals" / "research-roadmap-vNEXT.md").write_text(
        "## Second verification pass\n\nDiffusionGemma false-positive note and poison-test cascade.\n",
        encoding="utf-8",
    )
    tasks = [{"id": f"exp{exp_id}-task", "title": f"task {exp_id}"} for exp_id in range(5193, 5207)]
    (root / "research-roadmap.yaml").write_text(
        yaml.safe_dump({"milestone": "2026.07.476" if active_roadmap else "2026.07.475", "tasks": tasks}),
        encoding="utf-8",
    )
    retired_extras = []
    if diffusion_retired:
        retired_extras.append({"id": "diffusiongemma_retired", "experiment_ids": ["exp5196"], "reason": "DiffusionGemma retired"})
    (root / "ops" / "exclusion_manifest.yaml").write_text(yaml.safe_dump({"retired_extras": retired_extras}), encoding="utf-8")
    (root / "_bmad" / "architecture.md").write_text("# Architecture\n\n**Last Reconciled:** 2026-05-16\n", encoding="utf-8")
    _write_json(
        root / "results" / "operational_retro_2026_07_475.json",
        {"milestone": "2026.07.475", "experiments_completed": 0, "total_wall_time_minutes": 0, "reconstructed_from_disk_mtime": False},
    )
    if omit_artifact != 5181:
        _write_json(root / mod.V475_RESULT_PATHS[5181], _exp5181())
    if omit_artifact != 5182:
        _write_json(root / mod.V475_RESULT_PATHS[5182], _exp5182())
    return root


def test_req_report_5193_spec_declares_archive_contract() -> None:
    """REQ-REPORT-5193: OpenSpec anchors the .475 archive and .476 activation contract."""

    spec = Path("openspec/capabilities/research-reporting/spec.md").read_text(encoding="utf-8")

    for marker in (
        "REQ-REPORT-5193",
        "SCENARIO-REPORT-5193",
        "SCENARIO-REPORT-5193-BLOCKED-PRECONDITION",
        "results/experiment_5193_archive_475_activate_476.json",
        "v475_summary",
        "exp5181_duration_too_short_flag_assessment",
        "aggregation_from_upstream_artifacts",
    ):
        assert marker in spec
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert f"`{field}`" in spec
        assert principle in spec


def test_scenario_report_5193_happy_path_preserves_precise_v475_truth(tmp_path: Path) -> None:
    """SCENARIO-REPORT-5193: precise .475 outcomes and .476 activation are recorded."""

    artifact = mod.build_artifact(
        root=make_repo(tmp_path),
        duration_s=1.5,
        run_date="20260703",
        publication_gate=CLEAN_PUBLICATION_GATE,
        exclusion_lint=CLEAN_LINT,
        tests_run=["unit-test-placeholder"],
    )

    mod.validate_artifact(artifact)
    assert artifact["experiment_id"] == "exp5193-archive-475-activate-476"
    assert artifact["milestone"] == "2026.07.476"
    assert artifact["archived_milestone"] == "2026.07.475"
    assert artifact["honest_verdict"]["value"] == mod.COMPLETE_VERDICT
    assert artifact["inference_substrate"]["value"] == "aggregation_from_upstream_artifacts"
    assert artifact["exclusion_manifest_confirmed_clean"]["value"] is True
    assert artifact["research_roadmap_yaml_activated"]["value"] is True
    assert artifact["architecture_md_staleness_days"]["value"] == 48
    assert artifact["research_conductor_modified"] is False
    assert artifact["roadmap_activation_check"]["roadmap_next_present"] is False

    summary = artifact["v475_summary"]["value"]
    for required in (
        "2 of 12 queued tasks produced real artifacts",
        "exp5181",
        "exp5182",
        "exp5183-exp5192 never executed",
        "test_ondisk_deliverable_is_valid",
        "poisoned the shared pretest gate",
        "weight-tied mirror",
        "22.6 GiB",
    ):
        assert required in summary

    flag = artifact["exp5181_duration_too_short_flag_assessment"]["value"]
    assert "likely false positive" in flag
    assert "aggregation_from_upstream_artifacts" in flag
    assert "adversarial_verify.py" in flag

    rows = {row["exp_id"]: row for row in artifact["v475_task_rows"]}
    assert rows[5181]["artifact_status"] == "real_artifact_flagged"
    assert rows[5181]["key_facts"]["duration_s"] == pytest.approx(2.4946055188775063)
    assert rows[5182]["artifact_status"] == "real_artifact_blocked"
    assert rows[5182]["key_facts"]["mitigation_count"] == 4
    assert rows[5182]["key_facts"]["single_gpu_oom_count"] == 3
    assert rows[5182]["key_facts"]["nf4_footprint_gib"] == 12.864
    assert rows[5183]["artifact_status"] == "never_executed_no_artifact"
    assert rows[5192]["artifact_status"] == "never_executed_no_artifact"

    assert artifact["source_artifact_audit"]["real_artifact_count"] == 2
    assert artifact["source_artifact_audit"]["missing_exp_ids"] == list(range(5183, 5193))
    assert artifact["conductor_timeline"]["planner_failures_before_hand_activation"] == 3
    assert artifact["conductor_timeline"]["exp5182_conductor_timeout"] is True
    assert artifact["conductor_timeline"]["deliverable_exists_retry_at_1035"] is True
    assert artifact["operational_retro_false_zero"]["experiments_completed"] == 0
    assert artifact["publication_gate"]["paper_ready"] is True
    assert artifact["diffusiongemma_retirement_audit"]["clean"] is True


def test_scenario_report_5193_blocked_preconditions_are_visible(tmp_path: Path) -> None:
    """SCENARIO-REPORT-5193-BLOCKED-PRECONDITION: failed inputs block clean handoff claims."""

    bad_lint = mod.CommandResult(
        command=CLEAN_LINT.command,
        exit_code=1,
        stdout="[HARD] exp5196-diffusiongemma-vllm-native-retry-v476 retired scope",
        stderr="",
    )
    cases = [
        mod.build_artifact(
            root=make_repo(tmp_path / "inactive", active_roadmap=False),
            duration_s=1.0,
            run_date="20260703",
            publication_gate=CLEAN_PUBLICATION_GATE,
            exclusion_lint=CLEAN_LINT,
            tests_run=["unit-test-placeholder"],
        ),
        mod.build_artifact(
            root=make_repo(tmp_path / "missing", omit_artifact=5182),
            duration_s=1.0,
            run_date="20260703",
            publication_gate=CLEAN_PUBLICATION_GATE,
            exclusion_lint=CLEAN_LINT,
            tests_run=["unit-test-placeholder"],
        ),
        mod.build_artifact(
            root=make_repo(tmp_path / "dirty", conductor_modified=True, diffusion_retired=True),
            duration_s=1.0,
            run_date="20260703",
            publication_gate={"paper_ready": False, "unmet_gates": ["G2"]},
            exclusion_lint=bad_lint,
            tests_run=["unit-test-placeholder"],
        ),
    ]

    for artifact in cases:
        mod.validate_artifact(artifact)
        assert artifact["honest_verdict"]["value"] == mod.BLOCKED_VERDICT
        assert artifact["clean_handoff"] is False
        assert artifact["failed_preconditions"]

    assert cases[0]["research_roadmap_yaml_activated"]["value"] is False
    assert cases[1]["source_artifact_audit"]["real_artifact_count"] == 1
    assert cases[2]["exclusion_manifest_confirmed_clean"]["value"] is False
    assert cases[2]["diffusiongemma_retirement_audit"]["clean"] is False


def test_req_report_5193_validation_edges_and_cli(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """REQ-REPORT-5193: validation fails closed and the CLI writes the artifact."""

    root = make_repo(tmp_path / "repo")
    artifact = mod.build_artifact(
        root=root,
        duration_s=1.0,
        run_date="20260703",
        publication_gate=CLEAN_PUBLICATION_GATE,
        exclusion_lint=CLEAN_LINT,
        tests_run=["unit-test-placeholder"],
    )
    mod.validate_artifact(artifact)

    mutations = [
        ("schema", "wrong"),
        ("experiment_id", "wrong"),
        ("milestone", "2026.07.475"),
        ("archived_milestone", "2026.07.474"),
        ("v475_summary", {"value": "", "principle": mod.FIELD_PRINCIPLES["v475_summary"]}),
        ("exclusion_manifest_confirmed_clean", {"value": "true", "principle": mod.FIELD_PRINCIPLES["exclusion_manifest_confirmed_clean"]}),
        ("research_roadmap_yaml_activated", {"value": "true", "principle": mod.FIELD_PRINCIPLES["research_roadmap_yaml_activated"]}),
        ("architecture_md_staleness_days", {"value": "48", "principle": mod.FIELD_PRINCIPLES["architecture_md_staleness_days"]}),
        ("exp5181_duration_too_short_flag_assessment", {"value": "", "principle": mod.FIELD_PRINCIPLES["exp5181_duration_too_short_flag_assessment"]}),
        ("inference_substrate", {"value": "live_llm_inference", "principle": mod.FIELD_PRINCIPLES["inference_substrate"]}),
        ("honest_verdict", {"value": "bad", "principle": mod.FIELD_PRINCIPLES["honest_verdict"]}),
        ("v475_task_rows", []),
        ("source_artifact_audit", {}),
        ("publication_gate", {"paper_ready": False, "unmet_gates": ["G2"]}),
        ("reproducibility_checksum", "bad"),
    ]
    for key, value in mutations:
        payload = copy.deepcopy(artifact)
        payload[key] = value
        with pytest.raises(ValueError):
            mod.validate_artifact(payload)

    payload = copy.deepcopy(artifact)
    payload["field_principles"]["v475_summary"] = "wrong"
    with pytest.raises(ValueError):
        mod.validate_artifact(payload)

    payload = copy.deepcopy(artifact)
    payload.pop("tests_run")
    with pytest.raises(ValueError):
        mod.validate_artifact(payload)

    payload = copy.deepcopy(artifact)
    payload["v475_summary"] = {"principle": "wrong", "value": "summary"}
    with pytest.raises(ValueError):
        mod.validate_artifact(payload)

    payload = copy.deepcopy(artifact)
    payload["v475_summary"] = {"principle": mod.FIELD_PRINCIPLES["v475_summary"]}
    with pytest.raises(ValueError):
        mod.validate_artifact(payload)

    bad_json = tmp_path / "bad.json"
    bad_json.write_text("{not json\n", encoding="utf-8")
    assert mod.read_json_mapping(bad_json)[1]["loadable"] is False
    list_json = tmp_path / "list.json"
    list_json.write_text("[]\n", encoding="utf-8")
    assert mod.read_json_mapping(list_json)[1]["error"] == "top-level JSON is not an object"

    poison_roadmap = tmp_path / "poison-roadmap.yaml"
    poison_roadmap.write_text("a: : :\n- [\n", encoding="utf-8")
    assert mod._roadmap_activation_check(poison_roadmap, tmp_path / "missing-next.yaml")["parses"] is False
    no_date_arch = tmp_path / "architecture-no-date.md"
    no_date_arch.write_text("# Architecture\n", encoding="utf-8")
    assert mod._architecture_staleness_days(no_date_arch, "20260703") == -1
    poison_manifest = tmp_path / "poison-manifest.yaml"
    poison_manifest.write_text("a: : :\n- [\n", encoding="utf-8")
    assert mod._diffusiongemma_retirement_audit(poison_manifest)["parses"] is False

    assert mod._unwrap({"value": {"value": 3}}) == 3
    assert mod._int(True, default=-7) == -7
    assert mod._int("not-an-int", default=-8) == -8
    assert mod._float(False) is None
    assert mod._float("not-a-float") is None
    assert mod._publication_gate_clean({"paper_ready": True}) is False
    assert mod._command_clean(mod.CommandResult(("cmd",), 0, "warnings only", "")) is True
    assert mod._command_clean(mod.CommandResult(("cmd",), 0, "HARD violation", "")) is False
    assert mod._architecture_staleness_days(tmp_path / "missing.md", "20260703") == -1
    assert mod._conductor_timeline(tmp_path / "missing.md")["log_exists"] is False
    assert mod._roadmap_activation_check(tmp_path / "missing.yaml", tmp_path / "missing-next.yaml")["activated"] is False
    assert mod._corrigendum_kinds({"corrigendum_pending": [{"kind": "DURATION_TOO_SHORT"}, {"kind": ""}]}) == ["DURATION_TOO_SHORT"]
    assert mod._task_row(tmp_path, 5183, {"honest_verdict": "complete_unexpected"})["artifact_status"] == "unexpected_real_artifact"
    assert "not sufficient" in mod._exp5181_flag_assessment({}, {})
    assert mod._parse_log_time("not a conductor row") is None
    assert mod._diffusiongemma_retirement_audit(tmp_path / "missing-manifest.yaml")["errors"] == ["manifest_missing"]
    assert mod._load_operational_retro(tmp_path / "missing-retro.json")["loadable"] is False
    assert mod._retro_timing_fallback_wiring(tmp_path / "missing-conductor.py")["exists"] is False
    venv_python = tmp_path / ".venv" / "bin" / "python"
    venv_python.parent.mkdir(parents=True)
    venv_python.write_text("# python\n", encoding="utf-8")
    assert mod._python_executable(tmp_path) == str(venv_python)
    failures = mod._failed_preconditions(
        source_audit={"all_required_real_artifacts_present": True},
        timeline={"log_exists": False},
        lint_audit={"clean": True},
        diffusion_audit={"clean": True},
        roadmap_check={"activated": True},
        architecture_days=-1,
        publication_gate=CLEAN_PUBLICATION_GATE,
        conductor_modified=False,
        vnext_exists=False,
    )
    assert failures == [
        "conductor_log_missing",
        "architecture_last_reconciled_unreadable",
        "research_roadmap_vnext_missing",
    ]

    calls = iter([type("R", (), {"returncode": 0})(), type("R", (), {"returncode": 1})()])
    monkeypatch.setattr(mod.subprocess, "run", lambda *args, **kwargs: next(calls))
    assert mod._conductor_modified(tmp_path) is True

    output = root / "results" / "cli.json"
    monkeypatch.setattr(mod, "run_publication_gate", lambda repo: CLEAN_PUBLICATION_GATE)
    monkeypatch.setattr(mod, "run_exclusion_manifest_lint", lambda repo: CLEAN_LINT)
    monkeypatch.setattr(mod, "_conductor_modified", lambda repo: False)
    assert mod.main(["--root", str(root), "--output", str(output), "--date", "20260703"]) == 0
    written = json.loads(output.read_text(encoding="utf-8"))
    mod.validate_artifact(written)
