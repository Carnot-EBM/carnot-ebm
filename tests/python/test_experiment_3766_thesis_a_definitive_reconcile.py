"""Tests for Exp 3766 Thesis-A definitive direct-run reconciliation.

Spec refs: REQ-REPORT-3766, SCENARIO-REPORT-3766.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from carnot.reporting import thesis_a_definitive_reconcile_3766 as exp3766


SPEC_PATH = Path("openspec/capabilities/research-reporting/spec.md")


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _seed_repo(root: Path) -> None:
    (root / "results").mkdir(parents=True, exist_ok=True)
    (root / "docs" / "research-notes").mkdir(parents=True, exist_ok=True)
    (root / "ops").mkdir(parents=True, exist_ok=True)
    (root / "ops" / "exclusion_manifest.yaml").write_text(
        "retired:\n"
        "  - experiment_id: 2091\n"
        "    reason: retired unrelated route\n",
        encoding="utf-8",
    )
    (root / "docs" / "research-notes" / "phase3-alternative-thesis-menu.md").write_text(
        "# Phase-3 Alternative-Thesis Menu\n\n"
        "## Thesis A - Energy as the GENERATOR (EBT-as-base)  *SELECTED 2026-06-02*\n\n"
        "- **Core claim:** energy descent can generate tokens at matched compute.\n"
        "- **Cost:** training-heavy. **Risk:** finicky training.\n",
        encoding="utf-8",
    )
    _write_json(
        root / "results" / "thesis_a_direct_definitive_run.json",
        {
            "experiment": "thesis_a_direct_definitive_run",
            "honest_verdict": (
                "complete: thesis_a_part_a_PASS_ebt_trained_stably_800_steps_"
                "and_LEARNED_heldout_margin_0.723_vs_untrained_0.084_"
                "part_b_decoder_scale_deferred"
            ),
            "ebt_param_count": 37954560,
            "ar_param_count": 28847874,
            "cumulative_steps_trained": 800,
            "nan_or_divergence_events": False,
            "ebt_trained_stably": True,
            "ebt_learned_heldout": True,
            "ar_learned_heldout": True,
            "ebt_heldout_margin_final": 0.7233336567878723,
            "ebt_heldout_margin_untrained_baseline": 0.0842966791242361,
            "ar_heldout_ce_init": 5.70572429895401,
            "ar_heldout_ce_final": 1.5516911447048187,
            "grad_norm_max": 204.21331787109375,
            "grad_norm_mean": 3.134373532906175,
            "peak_vram_mb": 2537,
            "stabilizers_applied": (
                "replay_buffer, langevin_noise, random_alpha, "
                "random_descent_steps, grad_clip, kl_reg"
            ),
            "random_seed": 30603,
            "reproducibility_checksum": (
                "4edca79e7c36dc478e91d0393b14fed2de7628f7c0aa7ac83ac083185011cef2"
            ),
            "duration_s": 506.97,
        },
    )
    _write_json(
        root / "results" / "thesis_a_part_b_scaled_seed1.json",
        {
            "experiment": "thesis_a_part_b_scaled_seed1",
            "honest_verdict": (
                "complete: thesis_a_part_b_scaled_BOUNDED_seed1_ebt_0.000_"
                "le_ar_0.840_ar1_0.820_headroom_ok"
            ),
            "task": "fixed-width 3-digit addition MSD-first (AR-hostile), held-out split",
            "training_diverged": False,
            "headroom_ok": True,
            "ar1_greedy_acc": 0.82,
            "arV_selfconsistency_acc": 0.84,
            "arK_selfconsistency_acc": 0.8,
            "ebt_argmin_acc": 0.0,
            "ebt_descent_decoder_acc": 0.0,
            "best_ebt_acc": 0.0,
            "matched_ar_acc": 0.84,
            "delta_best_ebt_minus_matched_ar": -0.84,
            "matched_compute": {
                "ebt_argmin_evals": 103200,
                "arV_forward": 103200,
                "ebt_descent_evals": 12000,
                "arK_forward": 12000,
                "argmin_ratio": 1.0,
                "descent_ratio": 1.0,
                "K": 30,
            },
            "n_eval": 100,
            "random_seed": 1,
            "reproducibility_checksum": (
                "174361e0d7563fde04efdefebb009fe23d3ec9d432752176812a42030b7e036c"
            ),
            "duration_s": 6119.84,
        },
    )


def test_req_report_3766_spec_anchor_exists() -> None:
    """REQ-REPORT-3766: OpenSpec declares the reconciliation contract."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    assert "REQ-REPORT-3766" in spec
    assert "SCENARIO-REPORT-3766" in spec
    assert exp3766.OUTPUT_REL_PATH.as_posix() in spec


def test_scenario_report_3766_reconciles_direct_runs(tmp_path: Path) -> None:
    """SCENARIO-REPORT-3766: direct runs supersede the in-loop chain honestly."""

    _seed_repo(tmp_path)
    manifest_before = (tmp_path / "ops" / "exclusion_manifest.yaml").read_text(
        encoding="utf-8"
    )

    output = exp3766.run(tmp_path, started_s=10.0, now_s=10.25)
    artifact = json.loads(output.read_text(encoding="utf-8"))
    menu_text = (
        tmp_path / "docs" / "research-notes" / "phase3-alternative-thesis-menu.md"
    ).read_text(encoding="utf-8")

    exp3766.validate_artifact(artifact)
    assert output == tmp_path / exp3766.OUTPUT_REL_PATH
    assert artifact["honest_verdict"] == exp3766.TERMINAL_VERDICT
    assert set(exp3766.REQUIRED_ARTIFACT_FIELDS) <= set(artifact)
    assert set(exp3766.REQUIRED_ARTIFACT_FIELDS) <= set(artifact["field_principles"])
    assert artifact["inference_substrate"] == exp3766.INFERENCE_SUBSTRATE
    assert artifact["thesis_a_part_a_outcome"] == exp3766.PART_A_OUTCOME
    assert artifact["thesis_a_part_b_outcome"] == exp3766.PART_B_OUTCOME
    assert artifact["ebt_discriminative_not_generative"] is True
    assert artifact["in_loop_chain_superseded"] is True
    assert artifact["not_added_to_exclusion_manifest"] is True
    assert artifact["thesis_menu_updated"] is True
    assert artifact["random_seed"] == 3766
    assert artifact["duration_s"] == 0.25
    assert artifact["reproducibility_checksum"] == exp3766.payload_checksum(artifact)
    assert artifact["adversarial_verify_clean"] is True
    assert artifact["adversarial_verify_report"]["max_severity"] < 2

    part_a = artifact["definitive_direct_runs"]["part_a_discriminative"]
    assert part_a["cumulative_steps_trained"] == 800
    assert part_a["ebt_trained_stably"] is True
    assert part_a["nan_or_divergence_events"] is False
    assert part_a["ebt_heldout_margin_final"] == 0.7233336567878723
    assert part_a["ebt_heldout_margin_untrained_baseline"] == 0.0842966791242361
    assert part_a["margin_summary"] == "pos/neg margin 0.723 vs untrained 0.084 (~8.6x)"
    assert part_a["grad_norm_max"] == 204.21331787109375
    assert part_a["random_seed"] == 30603
    assert part_a["duration_s"] == 506.97

    part_b = artifact["definitive_direct_runs"]["part_b_scaled_seed1"]
    assert part_b["ebt_argmin_acc"] == 0.0
    assert part_b["ebt_descent_decoder_acc"] == 0.0
    assert part_b["best_ebt_acc"] == 0.0
    assert part_b["ar1_greedy_acc"] == 0.82
    assert part_b["arV_selfconsistency_acc"] == 0.84
    assert part_b["matched_ar_acc_label"] == "0.84"
    assert isinstance(part_b["matched_ar_acc_label"], str)
    assert part_b["delta_best_ebt_minus_matched_ar"] == -0.84
    assert part_b["matched_compute_ratios"] == {
        "argmin_ratio": 1.0,
        "descent_ratio": 1.0,
    }
    assert part_b["random_seed"] == 1
    assert part_b["duration_s"] == 6119.84

    citations = {item["experiment_id"]: item for item in artifact["cited_upstream_artifacts"]}
    assert set(citations) == {
        "thesis_a_direct_definitive_run",
        "thesis_a_part_b_scaled_seed1",
    }
    assert citations["thesis_a_direct_definitive_run"]["sha256"] == exp3766.sha256_path(
        tmp_path / "results" / "thesis_a_direct_definitive_run.json"
    )
    assert citations["thesis_a_part_b_scaled_seed1"]["sha256"] == exp3766.sha256_path(
        tmp_path / "results" / "thesis_a_part_b_scaled_seed1.json"
    )
    assert "ebt_heldout_margin_final" in citations["thesis_a_direct_definitive_run"][
        "fields_imported"
    ]
    assert "matched_compute.argmin_ratio" in citations["thesis_a_part_b_scaled_seed1"][
        "fields_imported"
    ]
    assert "matched_ar_acc" in citations["thesis_a_part_b_scaled_seed1"]["fields_imported"]

    assert "Thesis A - BOUNDED: Energy as the GENERATOR" in menu_text
    assert "Exp 3766 definitive reconciliation" in menu_text
    assert "results/thesis_a_direct_definitive_run.json" in menu_text
    assert "results/thesis_a_part_b_scaled_seed1.json" in menu_text
    assert "arXiv:2510.27545" in menu_text
    assert (tmp_path / "ops" / "exclusion_manifest.yaml").read_text(
        encoding="utf-8"
    ) == manifest_before

    encoded = json.dumps(artifact, sort_keys=True)
    assert "GGUF" not in encoded
    assert "CUDA" not in encoded
    assert "model_specs" not in artifact
    assert "target_model" not in artifact
    assert "matched_ar_acc_label\": 0.84" not in encoded


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda p: p.pop("honest_verdict"), "missing required"),
        (lambda p: p.update(honest_verdict="complete: wrong"), "terminal verdict"),
        (lambda p: p.update(thesis_a_part_a_outcome="bounded"), "part-a outcome"),
        (lambda p: p.update(thesis_a_part_b_outcome="pass"), "part-b outcome"),
        (lambda p: p.update(ebt_discriminative_not_generative=False), "scientific finding"),
        (lambda p: p.update(in_loop_chain_superseded=False), "in-loop chain"),
        (lambda p: p.update(thesis_menu_updated=False), "thesis menu"),
        (lambda p: p.update(not_added_to_exclusion_manifest=False), "exclusion manifest"),
        (lambda p: p.update(cited_upstream_artifacts=[]), "cite"),
        (lambda p: p["definitive_direct_runs"]["part_b_scaled_seed1"].update(matched_ar_acc_label=0.84), "matched AR label"),
        (lambda p: p.update(random_seed=1), "random_seed"),
        (lambda p: p.update(duration_s=0.0), "duration_s"),
        (lambda p: p.update(reproducibility_checksum="short"), "sha256"),
        (lambda p: p.update(reproducibility_checksum="0" * 64), "does not match"),
        (lambda p: p.update(model_specs={}), "model_specs"),
        (lambda p: p.update(correction_note="CUDA marker"), "GGUF/CUDA"),
        (lambda p: p.update(field_principles=[]), "field_principles"),
        (lambda p: p["field_principles"].pop("honest_verdict"), "field principles"),
    ],
)
def test_req_report_3766_validate_rejects_regressions(
    tmp_path: Path,
    mutate,
    message: str,
) -> None:
    """REQ-REPORT-3766: schema validation blocks dishonest reconciliation."""

    _seed_repo(tmp_path)
    artifact = exp3766.build_artifact(
        tmp_path,
        started_s=1.0,
        now_s=1.25,
        adversarial_report={"flags": []},
    )
    broken = json.loads(json.dumps(artifact))
    mutate(broken)
    if broken.get("reproducibility_checksum") not in {"short", "0" * 64}:
        broken["reproducibility_checksum"] = exp3766.payload_checksum(broken)

    with pytest.raises(ValueError, match=message):
        exp3766.validate_artifact(broken)


@pytest.mark.parametrize(
    ("path", "updates", "message"),
    [
        (
            "results/thesis_a_direct_definitive_run.json",
            {"ebt_trained_stably": False},
            "part-a direct run",
        ),
        (
            "results/thesis_a_part_b_scaled_seed1.json",
            {"best_ebt_acc": 0.25},
            "part-b direct run",
        ),
        (
            "results/thesis_a_part_b_scaled_seed1.json",
            {"headroom_ok": False},
            "AR headroom",
        ),
    ],
)
def test_req_report_3766_build_fails_closed_on_wrong_upstream_shape(
    tmp_path: Path,
    path: str,
    updates: dict[str, object],
    message: str,
) -> None:
    """REQ-REPORT-3766: no direct-run evidence means no definitive verdict."""

    _seed_repo(tmp_path)
    artifact_path = tmp_path / path
    payload = json.loads(artifact_path.read_text(encoding="utf-8"))
    payload.update(updates)
    _write_json(artifact_path, payload)

    with pytest.raises(ValueError, match=message):
        exp3766.build_artifact(tmp_path)


def test_req_report_3766_refuses_to_retire_energy_as_generator(tmp_path: Path) -> None:
    """REQ-REPORT-3766: the finding is a bound, not an exclusion entry."""

    _seed_repo(tmp_path)
    (tmp_path / "ops" / "exclusion_manifest.yaml").write_text(
        "retired:\n"
        "  - id: energy-as-generator\n"
        "    reason: bad retirement\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="exclusion manifest"):
        exp3766.build_artifact(tmp_path)


def test_req_report_3766_helper_branches_cover_defensive_paths() -> None:
    """REQ-REPORT-3766: helper branches stay deterministic for odd shapes."""

    already_updated = (
        "## Thesis A - BOUNDED: Energy as the GENERATOR\n\n"
        "- **Exp 3766 definitive reconciliation (2026-06-03):** "
        "results/thesis_a_direct_definitive_run.json "
        "results/thesis_a_part_b_scaled_seed1.json arXiv:2510.27545\n"
    )
    with_thesis_b = "## Thesis A - Energy as the GENERATOR\n\n## Thesis B\n"

    assert exp3766.update_thesis_menu_text(already_updated) == already_updated
    assert "Exp 3766 definitive reconciliation" in exp3766.update_thesis_menu_text(
        with_thesis_b
    )
    assert exp3766.severity_rank("unknown") == -1
    assert exp3766.get_nested({"a": []}, "a.b") is None
    assert exp3766.duration_from(None, None) == 0.0001


def test_req_report_3766_cli_writes_artifact(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """REQ-REPORT-3766: the script entrypoint writes the required deliverable."""

    _seed_repo(tmp_path)

    assert exp3766.main(["--root", str(tmp_path)]) == 0
    payload = json.loads((tmp_path / exp3766.OUTPUT_REL_PATH).read_text(encoding="utf-8"))
    printed = capsys.readouterr().out

    assert payload["honest_verdict"] == exp3766.TERMINAL_VERDICT
    assert exp3766.TERMINAL_VERDICT in printed
