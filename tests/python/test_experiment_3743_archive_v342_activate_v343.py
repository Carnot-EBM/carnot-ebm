"""Tests for Exp 3743 v342 archive and v343 activation.

Spec: REQ-REPORT-3743, SCENARIO-REPORT-3743.
"""

from __future__ import annotations

import json
from importlib.machinery import ModuleSpec
from pathlib import Path

import pytest

from carnot.reporting import archive_v342_activate_v343_3743 as exp3743


SPEC_PATH = Path("openspec/capabilities/research-reporting/spec.md")
TERMINAL_VERDICT = (
    "complete: "
    "archived_v342_thesis_a_record_honest_but_part_a_again_infra_blocked_"
    "cuda_false_still_untested_v343_active_paper_ready_true_frozen_headline_"
    "unchanged"
)


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _seed_repo(root: Path, *, active_milestone: str = "2026.06.343") -> None:
    (root / "openspec" / "change-proposals").mkdir(parents=True, exist_ok=True)
    (root / "ops").mkdir(parents=True, exist_ok=True)
    (root / "_bmad").mkdir(parents=True, exist_ok=True)
    (root / "scripts").mkdir(parents=True, exist_ok=True)
    (root / "results").mkdir(parents=True, exist_ok=True)
    (root / "research-roadmap.yaml").write_text(
        f'milestone: "{active_milestone}"\n'
        "milestone_title: \"PHASE-3 THESIS A - PIN .venv/bin/python AND HARD-BLOCK\"\n"
        "tasks:\n"
        "  - id: exp3743-archive-v342-activate-v343\n"
        "    title: Archive .342 honestly and activate .343\n",
        encoding="utf-8",
    )
    (root / "openspec" / "change-proposals" / "research-roadmap-vNEXT.md").write_text(
        "# Research Roadmap - Milestone 2026.06.343\n\n"
        ".342 was AGAIN infra-blocked: exp3734 ran with cuda:false from bare python, "
        "silently dropped to CPU for two steps, and exp3735 blocked_cuda. .343 pins "
        ".venv/bin/python and hard-blocks on cuda:false before the genuine kill-gate.\n",
        encoding="utf-8",
    )
    (root / "research-complete.yaml").write_text(
        "# completed\n\n"
        "milestones:\n"
        "- id: 2026.06.341\n"
        "  finding: previous archive\n"
        "- id: 2026.06.342\n"
        "  finding: See conductor log for per-experiment results.\n"
        "  tasks:\n"
        "  - id: exp3734-fix-harness-and-bounded-train-chunk1\n"
        "    result: OK (conductor)\n",
        encoding="utf-8",
    )
    (root / "ops" / "north-star.md").write_text(
        "# Carnot North Star\n\nFrozen FoVer headline AUROC: 0.9131.\n",
        encoding="utf-8",
    )
    (root / "ops" / "status.md").write_text("status before\n", encoding="utf-8")
    (root / "ops" / "changelog.md").write_text("changelog before\n", encoding="utf-8")
    (root / "_bmad" / "traceability.md").write_text("trace before\n", encoding="utf-8")
    (root / "scripts" / "research_conductor.py").write_text(
        "# conductor unchanged\n",
        encoding="utf-8",
    )

    _write_json(
        root / "results" / "experiment_3732_archive_v341_activate_v342.json",
        {
            "honest_verdict": (
                "complete: archived_v341_thesis_a_smoke_passed_but_killgate_was_"
                "infra_false_negative_part_a_reopened_untested_v342_active_"
                "paper_ready_true_frozen_headline_unchanged"
            ),
            "paper_ready_preserved": True,
            "paper_ready_evidence": {
                "paper_ready": True,
                "frozen_headline_unchanged": True,
                "frozen_headline_auroc": 0.9131,
                "g1": True,
                "g2": True,
                "g3": True,
                "g4": True,
            },
            "p01_status_preserved": "honest-negative-bounded",
            "random_seed": 3732,
            "reproducibility_checksum": "2" * 64,
            "duration_s": 0.0001,
        },
    )
    _write_json(
        root / "results" / "experiment_3733_corrigendum_exp3729_false_negative.json",
        {
            "honest_verdict": (
                "complete: exp3729_killgate_corrected_infra_false_negative_"
                "part_a_reopened_untested_energy_as_generator_not_retired"
            ),
            "part_a_status_corrected": "UNTESTED_at_bounded_scale_not_bounded",
            "energy_as_generator_not_retired": True,
            "random_seed": 3733,
            "reproducibility_checksum": "3" * 64,
            "duration_s": 0.000635,
        },
    )
    _write_json(
        root / "results" / "experiment_3734_fix_harness_and_bounded_train_chunk1.json",
        {
            "honest_verdict": (
                "complete: harness_fixed_ebt_train_chunk_2_steps_stable_so_far_"
                "loss_converging_no_nan_ar_baseline_co_trained_checkpointed"
            ),
            "harness_fix_applied": True,
            "cumulative_steps_trained": 2,
            "ebt_loss_curve": [0.9902918338775635, 1.141614317893982],
            "ar_loss_curve": [5.6731038093566895, 5.8303961753845215],
            "nan_or_divergence_events": False,
            "stabilizers_applied": (
                "replay_buffer, langevin_noise, random_alpha, random_descent_steps, "
                "grad_clip, kl_cd_fix"
            ),
            "peak_vram_mb": 100,
            "preconditions_checked": {
                "cuda": False,
                "ebt_vendored": True,
                "corpus_ok": True,
            },
            "random_seed": 3734,
            "reproducibility_checksum": "4" * 64,
            "duration_s": 1.64,
        },
    )
    _write_json(
        root / "results" / "experiment_3735_bounded_train_chunk2_resume.json",
        {
            "honest_verdict": "blocked_cuda",
            "cumulative_steps_trained": 0,
            "preconditions_checked": {
                "cuda": False,
                "ebt_vendored": True,
                "checkpoint_present": True,
            },
            "peak_vram_mb": 0,
            "random_seed": 3734,
            "reproducibility_checksum": "",
            "duration_s": 0.05,
        },
    )
    _write_json(
        root / "results" / "experiment_3736_real_kill_gate_part_a_verdict.json",
        {
            "honest_verdict": "complete: real_kill_gate_part_a_untested_training_did_not_complete",
            "green_light_342": False,
            "ebt_trained_stably": False,
            "training_actually_ran": True,
            "supersedes_exp3729": True,
            "kill_gate_conclusion": "UNTESTED: training did not complete -- part-(a) remains untested.",
            "real_run_diagnostics": {
                "bounded_run_completed": False,
                "cumulative_steps_trained": 2,
                "missing_or_blocked_artifacts": [3735],
            },
            "random_seed": 3736,
            "reproducibility_checksum": "6" * 64,
            "duration_s": 0.0001,
        },
    )
    _write_json(
        root / "results" / "experiment_3737_ebt_generation_smoke.json",
        {
            "honest_verdict": "blocked_gate_check_failed",
            "gate_check_summary": "exp3736.green_light_342 false",
            "duration_s": 0.0,
        },
    )
    _write_json(
        root / "results" / "experiment_3739_kill_gate_part_b_verdict.json",
        {
            "honest_verdict": "complete: kill_gate_part_b_not_run_part_a_did_not_green_light",
            "thesis_a_outcome": "part_b_not_run",
            "ebt_beats_ar_at_matched_compute": False,
            "part_b_not_run_reason": "part-(a) did not green-light: training did not complete",
            "random_seed": 3739,
            "reproducibility_checksum": "9" * 64,
            "duration_s": 0.0001,
        },
    )
    _write_json(
        root / "results" / "experiment_3740_fr11_self_learning_v15_stabilizer_tracker.json",
        {
            "honest_verdict": (
                "complete: fr11_v15_tier1_stabilizer_efficacy_tracker_recipe_"
                "recommended_state_persisted_preliminary_over_3_chunks"
            ),
            "tracker_state_persisted": True,
            "n_chunks_observed": 3,
            "is_preliminary_heuristic": True,
            "random_seed": 3740,
            "reproducibility_checksum": "a" * 64,
            "duration_s": 0.852,
        },
    )
    _write_json(
        root / "results" / "experiment_3741_kv260_opportunistic_continuity_audit.json",
        {
            "honest_verdict": (
                "complete: kv260_terminal_state_holds_ssh_reachable_"
                "accelerator_loadable_opportunistic_audit"
            ),
            "terminal_state_holds": True,
            "kv260_ssh_reachable": True,
            "kv260_overlay_loadable": True,
            "speedup_claim_made": False,
            "random_seed": 3741,
            "reproducibility_checksum": "b" * 64,
            "duration_s": 2.1224,
        },
    )


def test_req_report_3743_spec_anchor_exists() -> None:
    """REQ-REPORT-3743: OpenSpec declares the archive/activation contract."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    assert "REQ-REPORT-3743" in spec
    assert "SCENARIO-REPORT-3743" in spec
    assert exp3743.OUTPUT_REL_PATH.as_posix() in spec


def test_scenario_report_3743_run_archives_v342_honestly(tmp_path: Path) -> None:
    """SCENARIO-REPORT-3743: archive records CPU-drop and confirms .343."""

    _seed_repo(tmp_path)
    before_conductor = (tmp_path / "scripts" / "research_conductor.py").read_text(
        encoding="utf-8"
    )
    before_north = (tmp_path / "ops" / "north-star.md").read_text(encoding="utf-8")
    before_docs = {
        "status": (tmp_path / "ops" / "status.md").read_text(encoding="utf-8"),
        "changelog": (tmp_path / "ops" / "changelog.md").read_text(encoding="utf-8"),
        "trace": (tmp_path / "_bmad" / "traceability.md").read_text(encoding="utf-8"),
    }

    out_path = exp3743.run(tmp_path)
    artifact = json.loads(out_path.read_text(encoding="utf-8"))

    exp3743.validate_artifact(artifact)
    assert artifact["honest_verdict"] == TERMINAL_VERDICT
    assert set(exp3743.REQUIRED_ARTIFACT_FIELDS) <= set(artifact)
    assert set(exp3743.REQUIRED_ARTIFACT_FIELDS) <= set(artifact["field_principles"])
    assert artifact["inference_substrate"] == exp3743.INFERENCE_SUBSTRATE
    assert artifact["v342_outcome_recorded"] == exp3743.V342_OUTCOME
    assert artifact["cuda_unavailable_root_cause_recorded"] == exp3743.CUDA_ROOT_CAUSE
    assert artifact["thesis_a_still_open_recorded"] == exp3743.THESIS_A_OPEN_STATUS
    assert artifact["paper_ready_preserved"] is True
    assert artifact["p01_status_preserved"] == exp3743.P01_STATUS
    assert artifact["n_tasks_archived"] == 11
    assert artifact["adversarial_verify_clean"] is True
    assert artifact["adversarial_verify_report"]["max_severity"] < 2
    assert artifact["random_seed"] == 3743
    assert artifact["duration_s"] >= 0.0001
    assert len(artifact["reproducibility_checksum"]) == 64
    assert artifact["v343_active_confirmed"] is True
    assert artifact["v342_evidence"]["exp3733_false_negative_corrected"] is True
    assert artifact["v342_evidence"]["exp3734_cpu_drop_detected"] is True
    assert artifact["v342_evidence"]["exp3734_stability_signal_valid"] is False
    assert artifact["v342_evidence"]["exp3735_blocked_cuda"] is True
    assert artifact["v342_evidence"]["exp3736_part_a_untested"] is True
    assert artifact["v342_evidence"]["exp3739_part_b_not_run"] is True
    assert artifact["paper_ready_evidence"] == {
        "paper_ready": True,
        "frozen_headline_unchanged": True,
        "frozen_headline_auroc": 0.9131,
        "g1": True,
        "g2": True,
        "g3": True,
        "g4": True,
    }
    assert artifact["scripts_research_conductor_modified"] is False
    encoded = json.dumps(artifact)
    assert "model_specs" not in encoded
    assert "target_model" not in encoded
    assert "GGUF" not in encoded
    assert "live-model" not in encoded

    complete = (tmp_path / "research-complete.yaml").read_text(encoding="utf-8")
    assert complete.count("- id: 2026.06.342") == 1
    assert "See conductor log" not in complete
    assert "RECORD-HONEST BUT INFRA-BLOCKED MILESTONE" in complete
    assert "exp3733 corrected the .341 false-negative" in complete
    assert "exp3734 ran only 2 steps" in complete
    assert "cuda:false" in complete
    assert "100MB" in complete
    assert "exp3735 blocked_cuda" in complete
    assert "part-(a) remains UNTESTED" in complete
    assert "Energy-as-generator remains UNTESTED at bounded scale" in complete
    assert "P0.1 stayed honest-negative-bounded" in complete
    assert "paper_ready stayed TRUE" in complete
    assert "frozen FoVer 0.9131 stayed frozen" in complete
    assert complete.count("deliverable: results/experiment_") == 11
    assert "result: CPU-DROP; 2 CPU steps invalid as stability evidence" in complete
    assert "result: BLOCKED_CUDA; no CPU fallback accepted" in complete
    assert "result: UNTESTED; training did not complete" in complete
    assert "result: NOT-RUN; part-(a) did not green-light" in complete
    assert (tmp_path / "scripts" / "research_conductor.py").read_text(
        encoding="utf-8"
    ) == before_conductor
    assert (tmp_path / "ops" / "north-star.md").read_text(encoding="utf-8") == before_north
    assert (tmp_path / "ops" / "status.md").read_text(encoding="utf-8") == before_docs[
        "status"
    ]
    assert (tmp_path / "ops" / "changelog.md").read_text(
        encoding="utf-8"
    ) == before_docs["changelog"]
    assert (tmp_path / "_bmad" / "traceability.md").read_text(
        encoding="utf-8"
    ) == before_docs["trace"]


def test_req_report_3743_research_complete_rewrite_is_idempotent(tmp_path: Path) -> None:
    """REQ-REPORT-3743: missing or existing v342 archive entries stay stable."""

    _seed_repo(tmp_path)
    (tmp_path / "research-complete.yaml").write_text(
        "# completed\n\nmilestones:\n- id: 2026.06.341\n  finding: previous\n",
        encoding="utf-8",
    )

    first_path = exp3743.run(tmp_path)
    first_complete = (tmp_path / "research-complete.yaml").read_text(encoding="utf-8")
    first_artifact = json.loads(first_path.read_text(encoding="utf-8"))

    second_path = exp3743.run(tmp_path)
    second_complete = (tmp_path / "research-complete.yaml").read_text(encoding="utf-8")
    second_artifact = json.loads(second_path.read_text(encoding="utf-8"))

    assert first_complete == second_complete
    assert first_complete.count("- id: 2026.06.342") == 1
    assert first_artifact == second_artifact


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda p: p.pop("v342_outcome_recorded"), "missing required"),
        (lambda p: p.update(field_principles=[]), "field_principles"),
        (
            lambda p: p["field_principles"].pop("thesis_a_still_open_recorded"),
            "missing field principles",
        ),
        (lambda p: p.update(honest_verdict="complete: wrong"), "terminal verdict"),
        (lambda p: p.update(inference_substrate="live_inference"), "inference_substrate"),
        (lambda p: p.update(v343_active_confirmed=False), "v343"),
        (lambda p: p.update(v342_outcome_recorded="bounded"), ".342 outcome"),
        (lambda p: p.update(cuda_unavailable_root_cause_recorded="none"), "root cause"),
        (lambda p: p.update(thesis_a_still_open_recorded="bounded"), "Thesis A"),
        (lambda p: p.update(paper_ready_preserved=False), "paper_ready"),
        (lambda p: p.update(p01_status_preserved="positive"), "P0.1"),
        (lambda p: p.update(n_tasks_archived=10), "11"),
        (lambda p: p.update(adversarial_verify_clean=False), "adversarial_verify_clean"),
        (lambda p: p.update(duration_s=0.0), "duration_s"),
        (lambda p: p.update(reproducibility_checksum="short"), "sha256"),
        (lambda p: p.update(reproducibility_checksum="0" * 64), "does not match"),
        (lambda p: p.update(model_specs={}), "model_specs"),
        (lambda p: p.update(target_model=None), "target_model"),
    ],
)
def test_req_report_3743_validate_artifact_rejects_dishonest_fields(
    tmp_path: Path,
    mutate,
    message: str,
) -> None:
    """REQ-REPORT-3743: schema validation blocks silent regression."""

    _seed_repo(tmp_path)
    artifact_path = exp3743.run(tmp_path)
    payload = json.loads(artifact_path.read_text(encoding="utf-8"))
    exp3743.validate_artifact(payload)

    broken = json.loads(json.dumps(payload))
    mutate(broken)
    with pytest.raises(ValueError, match=message):
        exp3743.validate_artifact(broken)


def test_req_report_3743_requires_v343_to_be_active(tmp_path: Path) -> None:
    """SCENARIO-REPORT-3743: the archive cannot claim a wrong active milestone."""

    _seed_repo(tmp_path, active_milestone="2026.06.342")

    with pytest.raises(ValueError, match="v343"):
        exp3743.run(tmp_path)


def test_req_report_3743_defensive_helpers_fail_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-REPORT-3743: malformed inputs do not produce a terminal archive."""

    _seed_repo(tmp_path)
    (tmp_path / "research-roadmap.yaml").write_text("tasks: []\n", encoding="utf-8")
    with pytest.raises(ValueError, match="v343"):
        exp3743.build_artifact(tmp_path)

    _seed_repo(tmp_path)
    (
        tmp_path / "results" / "experiment_3734_fix_harness_and_bounded_train_chunk1.json"
    ).write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="expected JSON object"):
        exp3743.build_artifact(tmp_path)

    _seed_repo(tmp_path)
    (tmp_path / "openspec" / "change-proposals" / "research-roadmap-vNEXT.md").write_text(
        "No CPU-drop or hard-block language here.\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="cuda"):
        exp3743.build_artifact(tmp_path)

    _seed_repo(tmp_path)
    _write_json(
        tmp_path / "results" / "experiment_3733_corrigendum_exp3729_false_negative.json",
        {"honest_verdict": "complete: unrelated", "energy_as_generator_not_retired": False},
    )
    with pytest.raises(ValueError, match="false-negative correction"):
        exp3743.build_artifact(tmp_path)

    _seed_repo(tmp_path)
    _write_json(
        tmp_path / "results" / "experiment_3734_fix_harness_and_bounded_train_chunk1.json",
        {"preconditions_checked": {"cuda": True}, "cumulative_steps_trained": 200},
    )
    with pytest.raises(ValueError, match="CPU-drop"):
        exp3743.build_artifact(tmp_path)

    _seed_repo(tmp_path)
    _write_json(
        tmp_path / "results" / "experiment_3735_bounded_train_chunk2_resume.json",
        {"honest_verdict": "complete: resumed", "preconditions_checked": {"cuda": True}},
    )
    with pytest.raises(ValueError, match="blocked_cuda"):
        exp3743.build_artifact(tmp_path)

    _seed_repo(tmp_path)
    _write_json(
        tmp_path / "results" / "experiment_3736_real_kill_gate_part_a_verdict.json",
        {"honest_verdict": "complete: stable", "green_light_342": True},
    )
    with pytest.raises(ValueError, match="untested"):
        exp3743.build_artifact(tmp_path)

    _seed_repo(tmp_path)
    _write_json(
        tmp_path / "results" / "experiment_3739_kill_gate_part_b_verdict.json",
        {"honest_verdict": "complete: win", "thesis_a_outcome": "ebt_beats_ar"},
    )
    with pytest.raises(ValueError, match="part-b"):
        exp3743.build_artifact(tmp_path)

    with pytest.raises(ValueError, match="required text input missing"):
        exp3743._read_text_required(tmp_path / "missing.txt")
    assert exp3743._point({"point": 0.1234567}) == 0.123457
    assert exp3743._point("not-a-number") is None
    assert exp3743._nested({"a": {"b": 3}}, ("a", "b")) == 3
    assert exp3743._nested({"a": []}, ("a", "b")) is None
    assert exp3743._paper_ready_evidence({"paper_ready_preserved": True}) == {
        "paper_ready": True,
        "frozen_headline_unchanged": False,
        "frozen_headline_auroc": None,
        "g1": False,
        "g2": False,
        "g3": False,
        "g4": False,
    }
    assert exp3743._read_optional_json_object(tmp_path / "missing.json") is None
    assert exp3743._sha256_path(tmp_path / "missing.bin") == (
        "769b8995b8bf4407c89e906d67601a46266d34922a63ab1754440eecb0657aab"
    )
    assert exp3743._is_verify_clean({"flags": [{"severity": "warn"}]}) is True
    assert exp3743._is_verify_clean({"flags": [{"severity": "critical"}]}) is False
    assert exp3743._is_verify_clean({"flags": "bad"}) is True
    assert exp3743._compact_verify_report({"flags": [{"severity": "warn"}, "bad"]}) == {
        "flag_count": 1,
        "max_severity": 1,
        "flags": [{"severity": "warn"}],
    }

    monkeypatch.setattr(
        exp3743.importlib.util,
        "spec_from_file_location",
        lambda *_args, **_kwargs: None,
    )
    with pytest.raises(RuntimeError, match="could not load adversarial verifier"):
        exp3743._run_adversarial_verify(tmp_path / "missing.json")

    class _Loader:
        def create_module(self, _spec: ModuleSpec) -> None:
            return None

        def exec_module(self, module: object) -> None:
            module.verify_artifact = lambda _path: []  # type: ignore[attr-defined]

    monkeypatch.setattr(
        exp3743.importlib.util,
        "spec_from_file_location",
        lambda *_args, **_kwargs: ModuleSpec("fake_verify", _Loader()),
    )
    with pytest.raises(RuntimeError, match="non-object report"):
        exp3743._run_adversarial_verify(tmp_path / "missing.json")


def test_scenario_report_3743_script_wrapper_exists() -> None:
    """SCENARIO-REPORT-3743: conductor entrypoint delegates to the module."""

    script = Path("scripts/experiment_3743_archive_v342_activate_v343.py")
    assert script.exists()
    assert "archive_v342_activate_v343_3743" in script.read_text(encoding="utf-8")
