"""Tests for Exp 3924 .362 archive, .363 activation, and facts retirement.

Spec refs: REQ-REPORT-3924, SCENARIO-REPORT-3924,
SCENARIO-REPORT-3924-BLOCKED-YAML.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from carnot.reporting import archive_v362_activate_v363_retire_facts_3924 as mod


SPEC_PATH = Path("openspec/capabilities/research-reporting/spec.md")


SUMMARY_STDOUT = """\
==============================================================================
ARTIFACT  experiment_3914_archive_v361_activate_v362.json
------------------------------------------------------------------------------
  verdict          : complete: archived_v361_wash_v362_active_poison_test_quarantined_import_ok_codex_backend_recommended
  flagged_adversarial (stamped): None   |   LIVE re-check: clean
  duration_s       : 14.214833   substrate: aggregation_from_upstream_artifacts
  adversarial flags: none
==============================================================================
ARTIFACT  experiment_3915_robust_gguf_inference_harness.json
------------------------------------------------------------------------------
  verdict          : complete: gguf_inference_harness_READY_modelgemma-4-26B-A4B-it_ngl-1_smoke1_live_path_unblocked
  flagged_adversarial (stamped): None   |   LIVE re-check: clean
  duration_s       : 60.05125117301941   substrate: live_llm_inference:llama_cpp
  adversarial flags: none
==============================================================================
ARTIFACT  experiment_3916_moat_scissor_accuracy.json
------------------------------------------------------------------------------
  verdict          : complete: moat_scissor_MOAT_SURVIVES_residcatch_strong0.9143_ci0.8429-0.9714_overlap0.5000_holds_vs_boosted_self_verify_nres70
  flagged_adversarial (stamped): None   |   LIVE re-check: clean
  headline metrics :
      carnot_ensemble_auroc = 0.9666888888888889
      reasoner_auroc_strong = 0.6626666666666666
  adversarial flags: none
==============================================================================
ARTIFACT  experiment_3917_efficiency_head_to_head.json
------------------------------------------------------------------------------
  verdict          : complete: efficiency_CHEAPER_11512.51x_but_NOT_PARITY_energy0.8100_llm0.4423_honest_partial
  flagged_adversarial (stamped): None   |   LIVE re-check: clean
  headline metrics :
      energy_auroc = 0.8100177915518825
      llm_judge_auroc = 0.4423209366391185
  adversarial flags: none
==============================================================================
ARTIFACT  experiment_3918_cascade_router_prototype.json
------------------------------------------------------------------------------
  verdict          : complete: cascade_router_WINS_gap-0.3896_11512.51x_cheaper_at_matched_accuracy_escfrac0.0000
  flagged_adversarial (stamped): None   |   LIVE re-check: clean
  adversarial flags: none
==============================================================================
ARTIFACT  experiment_3919_arc_agi3_harness_scaffold.json
------------------------------------------------------------------------------
  verdict          : complete: arc_agi3_scaffold_READY_pruned8_synthetic_only_agentic_proof_can_follow_offline_proof
  flagged_adversarial (stamped): None   |   LIVE re-check: clean
  adversarial flags: none
==============================================================================
ARTIFACT  experiment_3920_facts_graph_grounding_last_retry.json
------------------------------------------------------------------------------
  verdict          : blocked_llama_cpp_inference_failed
  flagged_adversarial (stamped): True   |   LIVE re-check: CRITICAL
  duration_s       : 59.709457   substrate: none_blocked_preflight
  adversarial flags:
      [critical] DURATION_TOO_SHORT: duration_s=59.709457 but artifact references compute-bound markers
==============================================================================
ARTIFACT  experiment_3921_fr11_v25_independence_reweighting.json
------------------------------------------------------------------------------
  verdict          : complete: fr11_v25_INVARIANT_HELD_auroc0.9078_memcontrib0.0185_state_persisted
  flagged_adversarial (stamped): None   |   LIVE re-check: clean
  adversarial flags: none
==============================================================================
ARTIFACT  experiment_3922_hardware_continuity_consolidated.json
------------------------------------------------------------------------------
  verdict          : success: hardware_continuity_gatemateblocked_gatemate_board_unreachable_pfterminal_hash_verified_soft_cpu_ssh_dispatch_kvnonterminal_carnot_ising_inactive_uio_present_no_fabric_claim
  flagged_adversarial (stamped): True   |   LIVE re-check: CRITICAL
  duration_s       : 0.0   substrate: hardware_smoke
  adversarial flags:
      [critical] DURATION_TOO_SHORT: duration_s=0.0 but artifact references compute-bound markers
==============================================================================
ARTIFACT  experiment_3923_capstone_v362.json
------------------------------------------------------------------------------
  verdict          : complete: capstone_v362_moatMOAT_SURVIVES_efficiencyCHEAPER_NOT_PARITY_earnsfalse_paper_ready_true_frozen_unchanged
  flagged_adversarial (stamped): None   |   LIVE re-check: clean
  adversarial flags: none
==============================================================================
"""


def _command_result(
    command: list[str],
    *,
    exit_code: int = 0,
    stdout: str = "ok\n",
    stderr: str = "",
) -> mod.CommandResult:
    return mod.CommandResult(command=command, exit_code=exit_code, stdout=stdout, stderr=stderr)


def _summary_result(exit_code: int = 2, stdout: str = SUMMARY_STDOUT) -> mod.CommandResult:
    return _command_result(mod.summary_command(), exit_code=exit_code, stdout=stdout)


def _seed_repo(
    root: Path,
    *,
    corrupt_complete: bool = False,
    milestone: str = "2026.06.363",
    manifest: str | None = None,
) -> None:
    (root / "research-roadmap.yaml").write_text(
        f'milestone: "{milestone}"\n'
        "tasks:\n"
        "  - id: exp3924-archive-v362-activate-v363-retire-facts\n"
        "    agent_type: codex\n"
        "    requires_codex: true\n",
        encoding="utf-8",
    )
    complete_text = (
        "milestones:\n"
        "- id: 2026.06.362\n"
        "  title: stale conductor archive\n"
        "  completed: '2026-06-07'\n"
        "  finding: See conductor log for per-experiment results.\n"
        "  tasks:\n"
        "  - id: exp3914-archive-v361-activate-v362-quarantine-poison-test\n"
        "    deliverable: results/experiment_3914_archive_v361_activate_v362.json\n"
        "    result: OK (conductor)\n"
    )
    if corrupt_complete:
        complete_text += "  - id: poison\n    result: complete: unquoted colon\n"
    (root / "research-complete.yaml").write_text(complete_text, encoding="utf-8")
    (root / "ops").mkdir(parents=True, exist_ok=True)
    (root / "_bmad").mkdir(parents=True, exist_ok=True)
    (root / "scripts").mkdir(parents=True, exist_ok=True)
    (root / "results").mkdir(parents=True, exist_ok=True)
    (root / "ops" / "changelog.md").write_text("changelog before\n", encoding="utf-8")
    (root / "ops" / "status.md").write_text("status before\n", encoding="utf-8")
    (root / "_bmad" / "traceability.md").write_text("trace before\n", encoding="utf-8")
    (root / "scripts" / "research_conductor.py").write_text("# conductor before\n", encoding="utf-8")
    manifest_text = manifest or (
        "retired:\n"
        "  - experiment_id: 1\n"
        "    completed_milestone: \"2026.01.1\"\n"
        "    reason: existing\n"
        "\n"
        "retired_experiments:\n"
        "  - experiment_id: 887\n"
        "    completed_milestone: \"2026.04.68\"\n"
        "    reason: existing retired experiment\n"
        "\n"
        "# exp1117 historical structural fix\n"
        "retired_extras:\n"
        "  - id: existing-extra\n"
        "    reason: existing extra\n"
    )
    (root / "ops" / "exclusion_manifest.yaml").write_text(manifest_text, encoding="utf-8")
    (root / "results" / "experiment_3920_facts_graph_grounding_last_retry.json").write_text(
        json.dumps(
            {
                "honest_verdict": "blocked_llama_cpp_inference_failed",
                "flagged_adversarial": True,
                "duration_s": 59.709457,
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def test_req_report_3924_spec_anchor_exists() -> None:
    """REQ-REPORT-3924: OpenSpec declares the archive and retirement contract."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-REPORT-3924" in spec
    assert "SCENARIO-REPORT-3924" in spec
    assert "SCENARIO-REPORT-3924-BLOCKED-YAML" in spec
    assert mod.OUTPUT_REL_PATH.as_posix() in spec
    assert "facts_route_retired" in spec
    assert "llm_judge_auroc=0.4423" in spec


def test_scenario_report_3924_run_appends_archive_retirement_and_green_gates(
    tmp_path: Path,
) -> None:
    """SCENARIO-REPORT-3924: archive .362, retire facts, and assert gates."""

    _seed_repo(tmp_path)
    before = {
        "complete": (tmp_path / "research-complete.yaml").read_text(encoding="utf-8"),
        "manifest": (tmp_path / "ops" / "exclusion_manifest.yaml").read_text(encoding="utf-8"),
        "roadmap": (tmp_path / "research-roadmap.yaml").read_text(encoding="utf-8"),
        "changelog": (tmp_path / "ops" / "changelog.md").read_text(encoding="utf-8"),
        "status": (tmp_path / "ops" / "status.md").read_text(encoding="utf-8"),
        "trace": (tmp_path / "_bmad" / "traceability.md").read_text(encoding="utf-8"),
        "conductor": (tmp_path / "scripts" / "research_conductor.py").read_text(
            encoding="utf-8"
        ),
    }

    out_path = mod.run(
        tmp_path,
        research_complete_parse_result=_command_result(mod.research_complete_yaml_command()),
        summary_result=_summary_result(exit_code=2),
        core_pretest_result=_command_result(mod.core_pretest_command()),
        live_model_import_result=_command_result(mod.live_model_import_command()),
        started_s=10.0,
        now_s=11.25,
    )
    artifact = json.loads(out_path.read_text(encoding="utf-8"))
    complete_text = (tmp_path / "research-complete.yaml").read_text(encoding="utf-8")
    manifest_text = (tmp_path / "ops" / "exclusion_manifest.yaml").read_text(encoding="utf-8")
    archived = yaml.safe_load(complete_text)["milestones"][-1]
    task_results = {task["id"]: task["result"] for task in archived["tasks"]}

    mod.validate_artifact(artifact)
    assert artifact["honest_verdict"] == mod.terminal_verdict()
    assert set(mod.REQUIRED_ARTIFACT_FIELDS) <= set(artifact)
    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert not isinstance(artifact[field], dict), field
    assert artifact["archived_milestone"] == "2026.06.362"
    assert artifact["activated_milestone"] == "2026.06.363"
    assert artifact["facts_route_retired"] is True
    assert artifact["exclusion_manifest_parses"] is True
    assert artifact["research_complete_yaml_parses"] is True
    assert artifact["core_pretest_green"] is True
    assert artifact["live_model_modules_importable"] is True
    assert artifact["summary_exit_code"] == 2
    assert artifact["n_tasks_archived"] == 10
    assert artifact["duration_s"] == 1.25
    assert artifact["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert artifact["reproducibility_checksum"] == mod.payload_checksum(artifact)
    assert "0.4423" in artifact["n362_comparator_flaw_recorded"]
    assert "below chance" in artifact["n362_comparator_flaw_recorded"]
    assert "forward-bet" in artifact["n362_comparator_flaw_recorded"]
    for exp_id in range(3914, 3924):
        assert f"exp{exp_id}:" in artifact["prior_milestone_verdicts_summary"]
    assert "exp3920: blocked_llama_cpp_inference_failed [summarize_artifact LIVE_CRITICAL]" in (
        artifact["prior_milestone_verdicts_summary"]
    )
    assert "exp3922:" in artifact["prior_milestone_verdicts_summary"]
    assert "duration_s=0.0" in artifact["prior_milestone_verdicts_summary"]

    assert complete_text.startswith(before["complete"].rstrip())
    assert complete_text.count(mod.ARCHIVE_MARKER) == 1
    assert complete_text.count("- id: 2026.06.362") == 2
    assert "result: complete:" not in complete_text
    assert "result: 'complete:" in complete_text
    assert "n362_comparator_flaw_recorded: 'exp3917 efficiency axis invalid" in complete_text
    assert archived["activation_recorded"] == "exp3924-archive-v362-activate-v363-retire-facts"
    assert task_results["exp3917-efficiency-head-to-head-on-robust-harness"].endswith(
        "below_chance_llm_judge_auroc=0.4423 comparator_flaw]"
    )
    assert "LIVE_CRITICAL" in task_results["exp3920-facts-graph-grounding-last-retry"]
    assert "duration_s=0.0 hardware artifact flagged" in task_results[
        "exp3922-hardware-continuity-consolidated"
    ]
    assert yaml.safe_load(complete_text)

    assert manifest_text.startswith(before["manifest"].rstrip())
    assert manifest_text.count(mod.RETIREMENT_MARKER) == 1
    assert "experiment_id: 3920" in manifest_text
    assert "exp3862/exp3886/exp3896/exp3920" in manifest_text
    assert "retire_if_same_verdict" in manifest_text
    assert yaml.safe_load(manifest_text)

    assert (tmp_path / "research-roadmap.yaml").read_text(encoding="utf-8") == before["roadmap"]
    assert (tmp_path / "ops" / "changelog.md").read_text(encoding="utf-8") == before["changelog"]
    assert (tmp_path / "ops" / "status.md").read_text(encoding="utf-8") == before["status"]
    assert (tmp_path / "_bmad" / "traceability.md").read_text(encoding="utf-8") == before["trace"]
    assert (tmp_path / "scripts" / "research_conductor.py").read_text(
        encoding="utf-8"
    ) == before["conductor"]


def test_req_report_3924_append_is_idempotent(tmp_path: Path) -> None:
    """REQ-REPORT-3924: rerunning does not duplicate archive or retirement entries."""

    _seed_repo(tmp_path)
    kwargs = {
        "research_complete_parse_result": _command_result(mod.research_complete_yaml_command()),
        "summary_result": _summary_result(),
        "core_pretest_result": _command_result(mod.core_pretest_command()),
        "live_model_import_result": _command_result(mod.live_model_import_command()),
        "started_s": 1.0,
        "now_s": 1.5,
    }

    first = mod.run(tmp_path, **kwargs).read_text(encoding="utf-8")
    first_complete = (tmp_path / "research-complete.yaml").read_text(encoding="utf-8")
    first_manifest = (tmp_path / "ops" / "exclusion_manifest.yaml").read_text(encoding="utf-8")
    second = mod.run(tmp_path, **kwargs).read_text(encoding="utf-8")

    assert first == second
    assert (tmp_path / "research-complete.yaml").read_text(encoding="utf-8") == first_complete
    assert (tmp_path / "ops" / "exclusion_manifest.yaml").read_text(encoding="utf-8") == first_manifest
    assert first_complete.count(mod.ARCHIVE_MARKER) == 1
    assert first_manifest.count(mod.RETIREMENT_MARKER) == 1


def test_scenario_report_3924_blocked_yaml_writes_artifact_without_edits(
    tmp_path: Path,
) -> None:
    """SCENARIO-REPORT-3924-BLOCKED-YAML: corrupt YAML exits before edits."""

    _seed_repo(tmp_path, corrupt_complete=True)
    before_complete = (tmp_path / "research-complete.yaml").read_text(encoding="utf-8")
    before_manifest = (tmp_path / "ops" / "exclusion_manifest.yaml").read_text(encoding="utf-8")

    out_path = mod.run(
        tmp_path,
        research_complete_parse_result=_command_result(
            mod.research_complete_yaml_command(),
            exit_code=1,
            stderr="yaml parser failed",
        ),
        summary_result=_summary_result(),
        core_pretest_result=_command_result(mod.core_pretest_command()),
        live_model_import_result=_command_result(mod.live_model_import_command()),
        started_s=7.0,
        now_s=7.1,
    )
    artifact = json.loads(out_path.read_text(encoding="utf-8"))

    assert artifact["honest_verdict"].startswith("blocked_research_complete_yaml_poison")
    assert artifact["research_complete_yaml_parses"] is False
    assert artifact["facts_route_retired"] is False
    assert artifact["exclusion_manifest_parses"] is False
    assert artifact["core_pretest_green"] is False
    assert artifact["live_model_modules_importable"] is False
    assert artifact["preconditions_checked"]["research_complete_yaml_exists"] is True
    assert artifact["preconditions_checked"]["research_complete_yaml_parsed_before"] is False
    assert (tmp_path / "research-complete.yaml").read_text(encoding="utf-8") == before_complete
    assert (tmp_path / "ops" / "exclusion_manifest.yaml").read_text(encoding="utf-8") == before_manifest


def test_req_report_3924_helpers_fail_closed(tmp_path: Path) -> None:
    """REQ-REPORT-3924: hard helper failures block instead of fabricating success."""

    _seed_repo(tmp_path, milestone="2026.06.362")
    artifact = json.loads(
        mod.run(
            tmp_path,
            research_complete_parse_result=_command_result(mod.research_complete_yaml_command()),
            summary_result=_summary_result(),
            core_pretest_result=_command_result(mod.core_pretest_command()),
            live_model_import_result=_command_result(mod.live_model_import_command()),
        ).read_text(encoding="utf-8")
    )
    assert artifact["honest_verdict"].startswith("blocked_v363_not_active")

    _seed_repo(tmp_path, manifest="retired: [\n")
    artifact = json.loads(
        mod.run(
            tmp_path,
            research_complete_parse_result=_command_result(mod.research_complete_yaml_command()),
            summary_result=_summary_result(),
            core_pretest_result=_command_result(mod.core_pretest_command()),
            live_model_import_result=_command_result(mod.live_model_import_command()),
        ).read_text(encoding="utf-8")
    )
    assert artifact["honest_verdict"].startswith("blocked_exclusion_manifest_yaml_poison")

    _seed_repo(tmp_path)
    artifact = json.loads(
        mod.run(
            tmp_path,
            research_complete_parse_result=_command_result(mod.research_complete_yaml_command()),
            summary_result=_summary_result(exit_code=127, stdout=""),
            core_pretest_result=_command_result(mod.core_pretest_command()),
            live_model_import_result=_command_result(mod.live_model_import_command()),
        ).read_text(encoding="utf-8")
    )
    assert artifact["honest_verdict"].startswith("blocked_v362_summary_command_failed")

    _seed_repo(tmp_path)
    artifact = json.loads(
        mod.run(
            tmp_path,
            research_complete_parse_result=_command_result(mod.research_complete_yaml_command()),
            summary_result=_summary_result(),
            core_pretest_result=_command_result(
                mod.core_pretest_command(),
                exit_code=1,
                stdout="failed",
            ),
            live_model_import_result=_command_result(mod.live_model_import_command()),
        ).read_text(encoding="utf-8")
    )
    assert artifact["honest_verdict"].startswith("blocked_core_pretest_failed")
    assert artifact["core_pretest_green"] is False

    _seed_repo(tmp_path)
    artifact = json.loads(
        mod.run(
            tmp_path,
            research_complete_parse_result=_command_result(mod.research_complete_yaml_command()),
            summary_result=_summary_result(),
            core_pretest_result=_command_result(mod.core_pretest_command()),
            live_model_import_result=_command_result(
                mod.live_model_import_command(),
                exit_code=1,
                stderr="ImportError",
            ),
        ).read_text(encoding="utf-8")
    )
    assert artifact["honest_verdict"].startswith("blocked_carnot_verify_import")
    assert artifact["live_model_modules_importable"] is False


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda p: p.pop("n362_comparator_flaw_recorded"), "missing required"),
        (lambda p: p.update(field_principles=[]), "field_principles"),
        (lambda p: p.update(archived_milestone="2026.06.361"), "archived milestone"),
        (lambda p: p.update(activated_milestone="2026.06.362"), "activated milestone"),
        (lambda p: p.update(facts_route_retired=False), "facts route"),
        (lambda p: p.update(exclusion_manifest_parses=False), "manifest must parse"),
        (lambda p: p.update(research_complete_yaml_parses=False), "research-complete"),
        (lambda p: p.update(core_pretest_green=False), "core pretest"),
        (lambda p: p.update(live_model_modules_importable=False), "module imports"),
        (lambda p: p.update(active_milestone_confirmed=False), "active milestone"),
        (lambda p: p.update(n_tasks_archived=9), "n_tasks_archived"),
        (lambda p: p.update(n362_comparator_flaw_recorded="judge was fine"), "comparator flaw"),
        (lambda p: p.update(inference_substrate="live_model"), "inference"),
        (lambda p: p.update(duration_s=0.0), "duration_s"),
        (lambda p: p.update(reproducibility_checksum="short"), "sha256"),
        (lambda p: p.update(reproducibility_checksum="0" * 64), "does not match"),
        (lambda p: p.update(model_specs={}), "model_specs"),
        (lambda p: p.update(copied_marker="CUDA"), "compute-bound markers"),
    ],
)
def test_req_report_3924_validate_artifact_rejects_regressions(
    tmp_path: Path,
    mutate,
    message: str,
) -> None:
    """REQ-REPORT-3924: validation rejects fields that hide transition risk."""

    _seed_repo(tmp_path)
    payload = json.loads(
        mod.run(
            tmp_path,
            research_complete_parse_result=_command_result(mod.research_complete_yaml_command()),
            summary_result=_summary_result(),
            core_pretest_result=_command_result(mod.core_pretest_command()),
            live_model_import_result=_command_result(mod.live_model_import_command()),
            started_s=9.0,
            now_s=9.5,
        ).read_text(encoding="utf-8")
    )

    broken = json.loads(json.dumps(payload))
    mutate(broken)
    with pytest.raises(ValueError, match=message):
        mod.validate_artifact(broken)


def test_req_report_3924_summary_parser_records_flaws_and_live_flags() -> None:
    """REQ-REPORT-3924: summaries preserve below-chance and flagged evidence."""

    records = mod.parse_summary_records(SUMMARY_STDOUT)
    verdicts = mod.task_verdicts_from_summary(SUMMARY_STDOUT)

    assert records["3917"]["headline_metrics"]["llm_judge_auroc"] == pytest.approx(
        0.4423209366391185
    )
    assert mod.comparator_flaw_from_records(records) == mod.N362_COMPARATOR_FLAW
    assert verdicts["exp3917-efficiency-head-to-head-on-robust-harness"].endswith(
        "below_chance_llm_judge_auroc=0.4423 comparator_flaw]"
    )
    assert "LIVE_CRITICAL" in verdicts["exp3920-facts-graph-grounding-last-retry"]
    assert "duration_s=0.0 hardware artifact flagged" in verdicts[
        "exp3922-hardware-continuity-consolidated"
    ]


def test_req_report_3924_manifest_append_fallbacks() -> None:
    """REQ-REPORT-3924: retirement append remains parseable for alternate shapes."""

    no_retired_experiments = "retired:\n  - experiment_id: 1\n    reason: old\n"
    appended = mod.append_exclusion_manifest_record(no_retired_experiments)
    assert "retired_experiments:" in appended
    assert mod.RETIREMENT_MARKER in appended
    assert yaml.safe_load(appended)
    assert mod.append_exclusion_manifest_record(appended) == appended

    with_extras = (
        "retired:\n  - experiment_id: 1\n    reason: old\n"
        "\nretired_experiments:\n"
        "  - experiment_id: 2\n    reason: old\n"
        "\nretired_extras:\n"
        "  - id: old\n    reason: old\n"
    )
    appended_with_extras = mod.append_exclusion_manifest_record(with_extras)
    assert appended_with_extras.startswith(with_extras.rstrip())
    assert appended_with_extras.index(mod.RETIREMENT_MARKER) > appended_with_extras.index(
        "retired_extras:"
    )
    assert yaml.safe_load(appended_with_extras)


def test_req_report_3924_edge_fallbacks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-REPORT-3924: defensive edge paths remain explicit and covered."""

    assert mod.duration_from(None, None) == 0.0001
    assert mod._milestone_from_text("tasks: []\n") == "unknown"
    assert mod.read_active_milestone(tmp_path) == ("unknown", "research-roadmap.yaml")
    assert mod._read_facts_artifact(tmp_path) is None
    (tmp_path / "results").mkdir()
    (tmp_path / mod.FACTS_LAST_RETRY_REL_PATH).write_text("{not json", encoding="utf-8")
    assert mod._read_facts_artifact(tmp_path) is None

    odd_summary = """\
==============================================================================
ARTIFACT  experiment_3917_efficiency_head_to_head.json
------------------------------------------------------------------------------
  verdict          : complete: efficiency_CHEAPER_NOT_PARITY
  flagged_adversarial (stamped): True   |   LIVE re-check: warn
  duration_s       : not-a-number   substrate: aggregation
  headline metrics :
      llm_judge_auroc = not-a-number
==============================================================================
"""
    records = mod.parse_summary_records(odd_summary)
    verdicts = mod.task_verdicts_from_summary(odd_summary)
    assert records["3917"]["duration_s"] is None
    assert mod.comparator_flaw_from_records(records).startswith("exp3917 comparator flaw not confirmed")
    assert "stamped_flagged" in verdicts["exp3917-efficiency-head-to-head-on-robust-harness"]
    assert verdicts["exp3923-capstone-v362"].startswith("missing_artifact:")

    _seed_repo(tmp_path)
    (tmp_path / "research-complete.yaml").unlink()
    artifact = json.loads(
        mod.run(
            tmp_path,
            research_complete_parse_result=_command_result(
                mod.research_complete_yaml_command(),
                exit_code=1,
            ),
        ).read_text(encoding="utf-8")
    )
    assert artifact["honest_verdict"].startswith("blocked_research_complete_yaml_poison_missing")

    _seed_repo(tmp_path)
    (tmp_path / mod.FACTS_LAST_RETRY_REL_PATH).write_text(
        json.dumps({"honest_verdict": "complete: fabricated", "flagged_adversarial": False}),
        encoding="utf-8",
    )
    artifact = json.loads(
        mod.run(
            tmp_path,
            research_complete_parse_result=_command_result(mod.research_complete_yaml_command()),
        ).read_text(encoding="utf-8")
    )
    assert artifact["honest_verdict"].startswith("blocked_exp3920_facts_retirement_evidence_missing")

    _seed_repo(tmp_path)
    (tmp_path / "ops" / "exclusion_manifest.yaml").unlink()
    artifact = json.loads(
        mod.run(
            tmp_path,
            research_complete_parse_result=_command_result(mod.research_complete_yaml_command()),
        ).read_text(encoding="utf-8")
    )
    assert artifact["honest_verdict"].startswith("blocked_exclusion_manifest_missing")

    original_append_exclusion = mod.append_exclusion_manifest_record
    _seed_repo(tmp_path)
    monkeypatch.setattr(mod, "append_exclusion_manifest_record", lambda text: "retired: [\n")
    artifact = json.loads(
        mod.run(
            tmp_path,
            research_complete_parse_result=_command_result(mod.research_complete_yaml_command()),
        ).read_text(encoding="utf-8")
    )
    assert artifact["honest_verdict"].startswith("blocked_exclusion_manifest_append_invalid")

    _seed_repo(tmp_path)
    monkeypatch.setattr(mod, "append_exclusion_manifest_record", original_append_exclusion)
    monkeypatch.setattr(mod, "append_research_complete_record", lambda *args: "milestones: [\n")
    artifact = json.loads(
        mod.run(
            tmp_path,
            research_complete_parse_result=_command_result(mod.research_complete_yaml_command()),
            summary_result=_summary_result(),
        ).read_text(encoding="utf-8")
    )
    assert artifact["honest_verdict"].startswith("blocked_research_complete_append_invalid")


def test_req_report_3924_subprocess_helpers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-REPORT-3924: subprocess helpers use the mandated commands."""

    calls: list[list[str]] = []

    class Completed:
        returncode = 0
        stdout = SUMMARY_STDOUT
        stderr = ""

    def run_subprocess(cmd: list[str], **kwargs: object) -> Completed:
        calls.append(cmd)
        assert kwargs["cwd"] == tmp_path
        return Completed()

    monkeypatch.setattr(mod.subprocess, "run", run_subprocess)
    assert mod.run_research_complete_parse_check(tmp_path).command == (
        mod.research_complete_yaml_command()
    )
    assert calls == [mod.research_complete_yaml_command()]

    calls.clear()
    assert mod.run_summarize_artifacts(tmp_path).stdout == SUMMARY_STDOUT
    assert calls == [mod.summary_command()]

    calls.clear()
    assert mod.run_core_pretest(tmp_path).exit_code == 0
    assert calls == [mod.core_pretest_command()]

    calls.clear()
    assert mod.run_live_model_import_check(tmp_path).stdout == SUMMARY_STDOUT
    assert calls == [mod.live_model_import_command()]

    def fail_subprocess(cmd: list[str], **kwargs: object) -> object:
        raise mod.subprocess.CalledProcessError(1, cmd, output="", stderr="failed")

    monkeypatch.setattr(mod.subprocess, "run", fail_subprocess)
    assert mod.run_core_pretest(tmp_path).exit_code == 1
    assert mod.run_live_model_import_check(tmp_path).exit_code == 1

    def os_error_subprocess(cmd: list[str], **kwargs: object) -> object:
        raise OSError("missing executable")

    monkeypatch.setattr(mod.subprocess, "run", os_error_subprocess)
    assert mod.run_research_complete_parse_check(tmp_path).exit_code == 127
    assert mod.run_summarize_artifacts(tmp_path).exit_code == 127


def test_scenario_report_3924_script_wrapper_exists() -> None:
    """SCENARIO-REPORT-3924: the requested script entrypoint exists."""

    script = Path("scripts/experiments/experiment_3924_archive_v362_activate_v363_retire_facts.py")
    assert script.exists()
    text = script.read_text(encoding="utf-8")
    assert "archive_v362_activate_v363_retire_facts_3924" in text


def test_req_report_3924_main_prints_written_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """REQ-REPORT-3924: module main is a thin printable runner."""

    out_path = tmp_path / "results" / "experiment_3924_archive_v362_activate_v363_retire_facts.json"
    monkeypatch.setattr(mod, "run", lambda root: out_path)

    assert mod.main() == 0
    assert str(out_path) in capsys.readouterr().out
