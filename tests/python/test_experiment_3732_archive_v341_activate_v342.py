"""Tests for Exp 3732 v341 archive and v342 activation.

Spec: REQ-REPORT-3732, SCENARIO-REPORT-3732.
"""

from __future__ import annotations

import json
from importlib.machinery import ModuleSpec
from pathlib import Path

import pytest

from carnot.reporting import archive_v341_activate_v342_3732 as exp3732


SPEC_PATH = Path("openspec/capabilities/research-reporting/spec.md")
TERMINAL_VERDICT = (
    "complete: "
    "archived_v341_thesis_a_smoke_passed_but_killgate_was_infra_false_negative_"
    "part_a_reopened_untested_v342_active_paper_ready_true_frozen_headline_"
    "unchanged"
)


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")


def _seed_repo(root: Path, *, active_milestone: str = "2026.06.342") -> None:
    (root / "openspec" / "change-proposals").mkdir(parents=True, exist_ok=True)
    (root / "ops").mkdir(parents=True, exist_ok=True)
    (root / "_bmad").mkdir(parents=True, exist_ok=True)
    (root / "scripts").mkdir(parents=True, exist_ok=True)
    (root / "results").mkdir(parents=True, exist_ok=True)
    (root / "research-roadmap.yaml").write_text(
        f'milestone: "{active_milestone}"\n'
        'milestone_title: "PHASE-3 THESIS A - RECOVER FROM A FALSE-NEGATIVE"\n'
        "tasks:\n"
        "  - id: exp3732-archive-v341-activate-v342\n"
        "    title: Archive .341 honestly and activate .342\n",
        encoding="utf-8",
    )
    (root / "openspec" / "change-proposals" / "research-roadmap-vNEXT.md").write_text(
        "# Research Roadmap - Milestone 2026.06.342\n\n"
        "The previous .341 kill-gate fail is an INFRASTRUCTURE FALSE-NEGATIVE. "
        "exp3728 blocked at 0 steps because ebt_vendored=false and "
        "smoke_passed=false despite exp3725 importable=true and exp3726 passing. "
        ".342 reruns the genuine kill-gate and keeps P0.1 selection bounded.\n",
        encoding="utf-8",
    )
    (root / "research-complete.yaml").write_text(
        "# completed\n\n"
        "milestones:\n"
        "- id: 2026.06.340\n"
        "  finding: previous archive\n"
        "- id: 2026.06.341\n"
        "  finding: See conductor log for per-experiment results.\n"
        "  tasks:\n"
        "  - id: exp3728-bounded-checkpointed-train-ebt-and-ar\n"
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
        root / "results" / "experiment_3724_archive_v340_activate_v341.json",
        {
            "honest_verdict": "complete: archived_v340_thesis_a_seeded",
            "paper_ready_preserved": True,
            "p01_status_preserved": "honest-negative-bounded",
            "frozen_headline_auroc_preserved": 0.9131,
            "random_seed": 3724,
            "duration_s": 0.0001,
        },
    )
    _write_json(
        root / "results" / "experiment_3725_ebt_fork_vendor_importable.json",
        {
            "honest_verdict": "complete: ebt_vendored_importable_energy_path_audited",
            "importable": True,
            "license_confirmed": True,
            "smoke_energy_value": 0.5541654229164124,
            "upstream_commit_sha": "19420cbeae655bbf11930219a675ade6897019e8",
            "random_seed": 42,
            "duration_s": 15,
        },
    )
    _write_json(
        root / "results" / "experiment_3726_tiny_ebt_corpus_and_train_step_smoke.json",
        {
            "honest_verdict": "complete: tiny_ebt_38M_fits_single_smoke",
            "ebt_param_count": 37954560,
            "first_step_losses": [-0.077116, -19.918041, -37.738224],
            "loss_decreased": True,
            "loss_finite": True,
            "n_train": 2048,
            "peak_vram_mb": 1283,
            "random_seed": 3726,
            "duration_s": 7.84,
        },
    )
    _write_json(
        root / "results" / "experiment_3727_matched_compute_eval_harness.json",
        {
            "honest_verdict": "complete: matched_compute_eval_harness_built",
            "unit_tests_passed": "5_of_5_pass",
            "matched_compute_report": {
                "ebt_total_flops": 10000,
                "ar_total_flops": 10000,
                "budget_match": {"ar_best_of_m": 5, "within_tolerance": True},
            },
            "random_seed": 20260602,
            "duration_s": 1.92658,
        },
    )
    _write_json(
        root / "results" / "experiment_3728_bounded_checkpointed_train_ebt_and_ar.json",
        {
            "honest_verdict": "blocked_ebt",
            "cumulative_steps_trained": 0,
            "ebt_loss_curve": [],
            "ar_loss_curve": [],
            "ebt_converged": False,
            "nan_or_divergence_events": False,
            "peak_vram_mb": 0,
            "preconditions_checked": {
                "cuda": True,
                "ebt_vendored": False,
                "smoke_passed": False,
                "corpus_ok": False,
            },
            "random_seed": 3728,
            "duration_s": 65.5,
        },
    )
    _write_json(
        root / "results" / "experiment_3729_stability_kill_gate_verdict.json",
        {
            "honest_verdict": (
                "complete: kill_gate_part_a_FAIL_energy_as_generator_bounded_"
                "at_small_scale_honest_negative_stop"
            ),
            "green_light_342": False,
            "ebt_trained_stably": False,
            "kill_gate_conclusion": "BOUNDED: Exp 3728 has steps=0.",
            "stability_diagnostics": {
                "source_honest_verdict": "blocked_ebt",
                "cumulative_steps_trained": 0,
                "bounded_steps_present": False,
            },
            "random_seed": 3729,
            "duration_s": 0.000369453,
        },
    )
    _write_json(
        root / "results" / "experiment_3730_kv260_opportunistic_continuity_audit.json",
        {
            "honest_verdict": "complete: kv260_terminal_state_holds",
            "terminal_state_holds": True,
            "kv260_ssh_reachable": True,
            "speedup_claim_made": False,
            "random_seed": 3730,
            "duration_s": 6.4283,
        },
    )
    _write_json(
        root / "results" / "experiment_3731_capstone_v341.json",
        {
            "honest_verdict": (
                "complete: capstone_v341_thesis_a_ebt_bringup_kill_gate_"
                "part_a_bounded_paper_ready_true_frozen_headline_unchanged"
            ),
            "bringup_evidence": {
                "bounded_checkpointed_training_stability": {
                    "honest_verdict": "blocked_ebt",
                    "cumulative_steps_trained": 0,
                }
            },
            "frozen_fover_auroc": 0.9131,
            "frozen_headline_unchanged": True,
            "g_gates_preserved": {"g1": True, "g2": True, "g3": True, "g4": True},
            "generation_mechanism_under_test": "energy_as_generator_not_selector",
            "green_light_342": False,
            "kill_gate_part_a_passed": False,
            "p01_energy_selection_status": "honest-negative-bounded",
            "paper_ready_preserved": True,
            "thesis_a_bringup_outcome": "bounded_at_small_scale_do_not_auto_propose_342",
            "random_seed": 3731,
            "duration_s": 0.013144,
        },
    )


def test_req_report_3732_spec_anchor_exists() -> None:
    """REQ-REPORT-3732: OpenSpec declares the archive/activation contract."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    assert "REQ-REPORT-3732" in spec
    assert "SCENARIO-REPORT-3732" in spec
    assert exp3732.OUTPUT_REL_PATH.as_posix() in spec


def test_req_report_3732_run_archives_v341_and_writes_clean_artifact(
    tmp_path: Path,
) -> None:
    """SCENARIO-REPORT-3732: archive corrects .341 and activates .342."""

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

    out_path = exp3732.run(tmp_path)
    artifact = json.loads(out_path.read_text(encoding="utf-8"))

    exp3732.validate_artifact(artifact)
    assert artifact["honest_verdict"] == TERMINAL_VERDICT
    assert set(exp3732.REQUIRED_ARTIFACT_FIELDS) <= set(artifact)
    assert set(exp3732.REQUIRED_ARTIFACT_FIELDS) <= set(artifact["field_principles"])
    assert artifact["inference_substrate"] == exp3732.INFERENCE_SUBSTRATE
    assert artifact["v341_outcome_recorded"] == exp3732.V341_OUTCOME
    assert artifact["kill_gate_false_negative_recorded"] is True
    assert artifact["thesis_a_still_open_recorded"] == exp3732.THESIS_A_OPEN_STATUS
    assert artifact["paper_ready_preserved"] is True
    assert artifact["p01_status_preserved"] == exp3732.P01_STATUS
    assert artifact["n_tasks_archived"] == 8
    assert artifact["adversarial_verify_clean"] is True
    assert artifact["adversarial_verify_report"]["max_severity"] < 2
    assert artifact["random_seed"] == 3732
    assert artifact["duration_s"] >= 0.0001
    assert len(artifact["reproducibility_checksum"]) == 64
    assert artifact["v342_active_confirmed"] is True
    assert artifact["v341_evidence"]["vendor_audit_passed"] is True
    assert artifact["v341_evidence"]["single_step_smoke_passed"] is True
    assert artifact["v341_evidence"]["matched_compute_harness_ready"] is True
    assert artifact["v341_evidence"]["bounded_training_blocked_zero_steps"] is True
    assert artifact["v341_evidence"]["false_negative_root_cause"] == (
        "cwd_import_path_precondition_bug"
    )
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
    assert "CUDA" not in encoded

    complete = (tmp_path / "research-complete.yaml").read_text(encoding="utf-8")
    assert complete.count("- id: 2026.06.341") == 1
    assert "See conductor log" not in complete
    assert "INFRASTRUCTURE FALSE-NEGATIVE" in complete
    assert "vendor and energy-path audit passed" in complete
    assert "single-step smoke passed" in complete
    assert "matched-compute harness was built and tested" in complete
    assert "blocked at 0 steps" in complete
    assert "part-(a) re-opened as UNTESTED, not bounded" in complete
    assert "P0.1 stayed honest-negative-bounded" in complete
    assert "paper_ready stayed TRUE" in complete
    assert "frozen FoVer 0.9131 stayed frozen" in complete
    assert complete.count("deliverable: results/experiment_") == 8
    assert "result: PASS vendor/import/audit" in complete
    assert "result: BLOCKED at 0 steps; infra bug, not mechanism signal" in complete
    assert "result: FALSE-NEGATIVE; superseded by .342 correction path" in complete
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


def test_req_report_3732_research_complete_rewrite_is_idempotent(
    tmp_path: Path,
) -> None:
    """REQ-REPORT-3732: missing or existing v341 archive entries stay stable."""

    _seed_repo(tmp_path)
    (tmp_path / "research-complete.yaml").write_text(
        "# completed\n\nmilestones:\n- id: 2026.06.340\n  finding: previous\n",
        encoding="utf-8",
    )

    first_path = exp3732.run(tmp_path)
    first_complete = (tmp_path / "research-complete.yaml").read_text(encoding="utf-8")
    first_artifact = json.loads(first_path.read_text(encoding="utf-8"))

    second_path = exp3732.run(tmp_path)
    second_complete = (tmp_path / "research-complete.yaml").read_text(encoding="utf-8")
    second_artifact = json.loads(second_path.read_text(encoding="utf-8"))

    assert first_complete == second_complete
    assert first_complete.count("- id: 2026.06.341") == 1
    assert first_artifact == second_artifact


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda p: p.pop("v341_outcome_recorded"), "missing required"),
        (lambda p: p.update(field_principles=[]), "field_principles"),
        (
            lambda p: p["field_principles"].pop("thesis_a_still_open_recorded"),
            "missing field principles",
        ),
        (lambda p: p.update(honest_verdict="complete: wrong"), "terminal verdict"),
        (lambda p: p.update(inference_substrate="live_inference"), "inference_substrate"),
        (lambda p: p.update(v342_active_confirmed=False), "v342"),
        (lambda p: p.update(v341_outcome_recorded="bounded"), ".341 outcome"),
        (lambda p: p.update(kill_gate_false_negative_recorded=False), "false-negative"),
        (lambda p: p.update(thesis_a_still_open_recorded="bounded"), "Thesis A"),
        (lambda p: p.update(paper_ready_preserved=False), "paper_ready"),
        (lambda p: p.update(p01_status_preserved="positive"), "P0.1"),
        (lambda p: p.update(n_tasks_archived=7), "8"),
        (lambda p: p.update(adversarial_verify_clean=False), "adversarial_verify_clean"),
        (lambda p: p.update(duration_s=0.0), "duration_s"),
        (lambda p: p.update(reproducibility_checksum="short"), "sha256"),
        (lambda p: p.update(reproducibility_checksum="0" * 64), "does not match"),
        (lambda p: p.update(model_specs={}), "model_specs"),
        (lambda p: p.update(target_model=None), "target_model"),
    ],
)
def test_req_report_3732_validate_artifact_rejects_dishonest_fields(
    tmp_path: Path,
    mutate,
    message: str,
) -> None:
    """REQ-REPORT-3732: schema validation blocks silent regression."""

    _seed_repo(tmp_path)
    artifact_path = exp3732.run(tmp_path)
    payload = json.loads(artifact_path.read_text(encoding="utf-8"))
    exp3732.validate_artifact(payload)

    broken = json.loads(json.dumps(payload))
    mutate(broken)
    with pytest.raises(ValueError, match=message):
        exp3732.validate_artifact(broken)


def test_req_report_3732_requires_v342_to_be_active(tmp_path: Path) -> None:
    """SCENARIO-REPORT-3732: the archive cannot claim a wrong active milestone."""

    _seed_repo(tmp_path, active_milestone="2026.06.341")

    with pytest.raises(ValueError, match="v342"):
        exp3732.run(tmp_path)


def test_req_report_3732_defensive_helpers_fail_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-REPORT-3732: malformed inputs do not produce a terminal archive."""

    _seed_repo(tmp_path)
    (tmp_path / "research-roadmap.yaml").write_text("tasks: []\n", encoding="utf-8")

    with pytest.raises(ValueError, match="v342"):
        exp3732.build_artifact(tmp_path)

    _seed_repo(tmp_path)
    (
        tmp_path
        / "results"
        / "experiment_3728_bounded_checkpointed_train_ebt_and_ar.json"
    ).write_text("[]", encoding="utf-8")

    with pytest.raises(ValueError, match="expected JSON object"):
        exp3732.build_artifact(tmp_path)

    _seed_repo(tmp_path)
    (tmp_path / "openspec" / "change-proposals" / "research-roadmap-vNEXT.md").write_text(
        "No correction language here.\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="false-negative"):
        exp3732.build_artifact(tmp_path)

    _seed_repo(tmp_path)
    _write_json(
        tmp_path / "results" / "experiment_3726_tiny_ebt_corpus_and_train_step_smoke.json",
        {"loss_finite": True, "loss_decreased": False},
    )
    with pytest.raises(ValueError, match="single-step smoke"):
        exp3732.build_artifact(tmp_path)

    _seed_repo(tmp_path)
    _write_json(
        tmp_path / "results" / "experiment_3725_ebt_fork_vendor_importable.json",
        {"importable": False, "license_confirmed": True, "smoke_energy_value": 0.1},
    )
    with pytest.raises(ValueError, match="vendor/audit"):
        exp3732.build_artifact(tmp_path)

    _seed_repo(tmp_path)
    _write_json(
        tmp_path / "results" / "experiment_3727_matched_compute_eval_harness.json",
        {"unit_tests_passed": "4_of_5_pass", "matched_compute_report": {}},
    )
    with pytest.raises(ValueError, match="matched-compute harness"):
        exp3732.build_artifact(tmp_path)

    _seed_repo(tmp_path)
    _write_json(
        tmp_path / "results" / "experiment_3728_bounded_checkpointed_train_ebt_and_ar.json",
        {
            "honest_verdict": "blocked_ebt",
            "cumulative_steps_trained": 1,
            "preconditions_checked": {"ebt_vendored": False, "smoke_passed": False},
        },
    )
    with pytest.raises(ValueError, match="zero-step"):
        exp3732.build_artifact(tmp_path)

    _seed_repo(tmp_path)
    _write_json(
        tmp_path / "results" / "experiment_3729_stability_kill_gate_verdict.json",
        {"honest_verdict": "complete: stable", "green_light_342": True},
    )
    with pytest.raises(ValueError, match="false-negative"):
        exp3732.build_artifact(tmp_path)

    with pytest.raises(ValueError, match="required text input missing"):
        exp3732._read_text_required(tmp_path / "missing.txt")
    assert exp3732._point({"point": 0.1234567}) == 0.123457
    assert exp3732._point("not-a-number") is None
    assert exp3732._nested({"a": {"b": 3}}, ("a", "b")) == 3
    assert exp3732._nested({"a": []}, ("a", "b")) is None
    assert exp3732._blocked_zero_steps({"preconditions_checked": []}) is False
    assert exp3732._sha256_path(tmp_path / "missing.bin") == (
        "769b8995b8bf4407c89e906d67601a46266d34922a63ab1754440eecb0657aab"
    )
    assert exp3732._is_verify_clean({"flags": [{"severity": "warn"}]}) is True
    assert exp3732._is_verify_clean({"flags": [{"severity": "critical"}]}) is False
    assert exp3732._is_verify_clean({"flags": "bad"}) is True
    assert exp3732._compact_verify_report({"flags": [{"severity": "warn"}, "bad"]}) == {
        "flag_count": 1,
        "max_severity": 1,
        "flags": [{"severity": "warn"}],
    }

    monkeypatch.setattr(
        exp3732.importlib.util,
        "spec_from_file_location",
        lambda *_args, **_kwargs: None,
    )
    with pytest.raises(RuntimeError, match="could not load adversarial verifier"):
        exp3732._run_adversarial_verify(tmp_path / "missing.json")

    class _Loader:
        def create_module(self, _spec: ModuleSpec) -> None:
            return None

        def exec_module(self, module: object) -> None:
            module.verify_artifact = lambda _path: []  # type: ignore[attr-defined]

    monkeypatch.setattr(
        exp3732.importlib.util,
        "spec_from_file_location",
        lambda *_args, **_kwargs: ModuleSpec("fake_verify", _Loader()),
    )
    with pytest.raises(RuntimeError, match="non-object report"):
        exp3732._run_adversarial_verify(tmp_path / "missing.json")


def test_scenario_report_3732_script_wrapper_exists() -> None:
    """SCENARIO-REPORT-3732: conductor entrypoint delegates to the module."""

    script = Path("scripts/experiment_3732_archive_v341_activate_v342.py")
    assert script.exists()
    assert "archive_v341_activate_v342_3732" in script.read_text(encoding="utf-8")
