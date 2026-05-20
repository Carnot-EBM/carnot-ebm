"""Tests for the Exp 2554 milestone .245 capstone synthesis.

Spec traces: REQ-PUBLISH-031, SCENARIO-PUBLISH-031.
"""

from __future__ import annotations

import json
from pathlib import Path

from carnot.reporting import capstone_v245_2554 as exp2554


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_ready_inputs(root: Path) -> None:
    """Lay down the minimal happy-path inputs the capstone needs.

    The synthesis only reads a small handful of fields from each artifact;
    the fixture mirrors the production schema for those fields and leaves
    everything else off so the test stays auditable.
    """

    results = root / "results"
    _write_json(results / "experiment_2543_archive.json", {"honest_verdict": "complete: archived"})
    _write_json(
        results / "experiment_2544_phase4_option_b.json",
        {
            "honest_verdict": "complete: section expanded",
            "phase4_section_expanded": True,
            "phase4_honest_negative_documented": True,
        },
    )
    _write_json(
        results / "experiment_2545_ising_verifier_impl.json",
        {"honest_verdict": "complete: ising verifier ok"},
    )
    _write_json(
        results / "experiment_2546_ensemble_v7b.json",
        {
            "honest_verdict": "complete: v7b restored",
            "ensemble_v7b_auroc": 0.9857142857142858,
            "n_seeds": 5,
        },
    )
    _write_json(
        results / "experiment_2547_adaptive_conformal_v2.json",
        {
            "honest_verdict": "complete: no regression",
            "adaptive_conformal_auroc": 0.9928571428571429,
            "n_seeds": 5,
            "flagged_adversarial": True,
            "corrigendum_pending": [
                {"kind": "TAUTOLOGY", "severity": "critical", "detail": "baseline duplicated"}
            ],
        },
    )
    _write_json(
        results / "experiment_2548_real_corpus_validation.json",
        {
            "honest_verdict": "complete: real corpus eval",
            "tier0r_real_auroc": 0.9413750415828194,
            "tier0s_real_auroc": 0.3758077973921437,
            "tier0u_real_auroc": 0.535990952669208,
            "synthetic_baseline_auroc": {"tier0r": 0.8256, "tier0s": 1.0, "tier0u": 0.96},
        },
    )
    _write_json(
        results / "experiment_2549_tier0v_hallufield.json",
        {
            "honest_verdict": "blocked_carnot_import_failed: precondition failed",
            "tier0v_implementation_complete": False,
        },
    )
    _write_json(
        results / "experiment_2550_jepa_real_eval.json",
        {
            "honest_verdict": "complete: discrimination",
            "fast_path_rate": 0.5,
            "fast_path_precision": 1.0,
        },
    )
    _write_json(
        results / "experiment_2551_hardware_flash.json",
        {
            "honest_verdict": "complete: gatemate live, kv260 blocked",
            "gatemate": {
                "flash_attempted": True,
                "jtag_detected": True,
                "bitstream_flashed": False,
                "terminal_state_progress": "2/3 gates met",
                "flash_failure_mode": "openFPGALoader strtol parse error",
            },
            "kv260": {
                "flash_attempted": False,
                "sd_media_inserted": False,
                "pynq_url_reachable": False,
                "flash_result": "blocked_no_sd_media",
                "terminal_state_progress": "bitstream ready; SD absent",
            },
        },
    )
    _write_json(
        results / "experiment_2552_paper_writethrough.json",
        {"honest_verdict": "complete: paper updated"},
    )
    _write_json(
        results / "experiment_2553_arxiv_package_v3.json",
        {
            "honest_verdict": "complete: arxiv ready",
            "arxiv_ready": True,
            "latex_compile_success": True,
        },
    )


def test_is_terminal_verdict_accepts_required_prefixes_req_publish_031() -> None:
    """REQ-PUBLISH-031: terminal-prefix discipline drives the .245 count."""

    assert exp2554.is_terminal_verdict("complete: foo") is True
    assert exp2554.is_terminal_verdict("complete_underscore_form") is True
    assert exp2554.is_terminal_verdict("success: bar") is True
    assert exp2554.is_terminal_verdict("passed: baz") is True
    assert exp2554.is_terminal_verdict("shipped: qux") is True
    # Leading whitespace is tolerated; non-terminal prefixes are rejected.
    assert exp2554.is_terminal_verdict("  complete: ws") is True
    assert exp2554.is_terminal_verdict("blocked_precondition: no") is False
    assert exp2554.is_terminal_verdict("Terminal-prefix required. complete: malformed") is False
    assert exp2554.is_terminal_verdict(None) is False


def test_build_artifact_happy_path_scenario_publish_031(tmp_path: Path) -> None:
    """SCENARIO-PUBLISH-031: when all 11 inputs are present and arxiv_ready=True,
    capstone reports submit_now with the cite-safe ensemble v7b AUROC."""

    _write_ready_inputs(tmp_path)

    artifact = exp2554.build_artifact(
        tmp_path,
        started_epoch=100.0,
        now_epoch=100.125,
    )

    assert artifact["honest_verdict"].startswith("complete:")
    assert artifact["arxiv_ready"] is True
    assert artifact["operator_recommendation"] == "submit_now"
    assert artifact["phase4_final_status"] == "retired_negative_option_b"
    # Ensemble v7b is the cite-safe headline; adaptive conformal is higher but
    # adversarially flagged, so the synthesis must NOT pick it.
    assert abs(artifact["best_245_auroc"] - 0.9857142857142858) < 1e-9
    assert artifact["best_245_auroc_source"] == "exp2546_ensemble_v7b"
    assert artifact["auroc_adversarially_verified"] is True
    assert artifact["jepa_discrimination_improved"] is True
    assert artifact["n_experiments_completed"] == 10
    assert artifact["n_planned"] == 11
    # Hardware progress is partial -- both boards non-terminal.
    assert artifact["hardware_terminal_states"]["gatemate"]["terminal"] is False
    assert artifact["hardware_terminal_states"]["kv260"]["terminal"] is False
    # External baseline carries through for continuous gap tracking.
    assert artifact["external_baselines"]["hive_peer_auroc"] == 0.9236
    # Process flags surface the exp2547 TAUTOLOGY and the exp2549 non-terminal verdict.
    kinds = {entry["kind"] for entry in artifact["process_flags"]}
    assert "TAUTOLOGY" in kinds
    assert "NON_TERMINAL_VERDICT" in kinds
    assert artifact["duration_s"] == 0.125


def test_best_auroc_carries_forward_when_v7b_missing_req_publish_031(tmp_path: Path) -> None:
    """REQ-PUBLISH-031: when ensemble v7b is unavailable AND adaptive conformal
    is adversarially flagged, the synthesis must carry forward 0.9750."""

    _write_ready_inputs(tmp_path)
    # Wipe ensemble v7b so only the flagged adaptive conformal remains.
    _write_json(tmp_path / "results" / "experiment_2546_ensemble_v7b.json", {})

    artifact = exp2554.build_artifact(tmp_path, started_epoch=200.0, now_epoch=200.001)

    assert artifact["best_245_auroc"] == exp2554.CARRY_FORWARD_AUROC
    assert artifact["best_245_auroc_source"] == "exp2498_carryforward"
    assert artifact["auroc_adversarially_verified"] is False


def test_arxiv_not_ready_drives_operator_recommendation_req_publish_031(
    tmp_path: Path,
) -> None:
    """REQ-PUBLISH-031: when exp2553.arxiv_ready=False AND latex_compile_success=False,
    the recommendation must be fix_latex."""

    _write_ready_inputs(tmp_path)
    _write_json(
        tmp_path / "results" / "experiment_2553_arxiv_package_v3.json",
        {
            "honest_verdict": "complete: arxiv not ready",
            "arxiv_ready": False,
            "latex_compile_success": False,
        },
    )

    artifact = exp2554.build_artifact(tmp_path, started_epoch=300.0, now_epoch=300.0)

    assert artifact["arxiv_ready"] is False
    assert artifact["operator_recommendation"] == "fix_latex"


def test_write_artifact_persists_json_with_required_fields_req_publish_031(
    tmp_path: Path,
) -> None:
    """REQ-PUBLISH-031: write_artifact emits a JSON file with every required field."""

    _write_ready_inputs(tmp_path)

    written = exp2554.write_artifact(tmp_path)

    assert written.is_file()
    payload = json.loads(written.read_text(encoding="utf-8"))
    required = {
        "honest_verdict",
        "n_experiments_completed",
        "best_245_auroc",
        "auroc_adversarially_verified",
        "phase4_final_status",
        "arxiv_ready",
        "operator_recommendation",
        "hardware_terminal_states",
        "kv260_status",
        "gatemate_status",
        "jepa_discrimination_improved",
        "top_3_successes",
        "top_3_gaps_for_246",
        "external_baselines",
        "process_flags",
        "preconditions_checked",
        "duration_s",
        "random_seed",
    }
    assert required.issubset(payload.keys())
    assert payload["honest_verdict"].startswith("complete:")
