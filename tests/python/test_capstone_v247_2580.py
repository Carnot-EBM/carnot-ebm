"""Tests for the Exp 2580 milestone .247 capstone synthesis.

Spec traces: REQ-PUBLISH-031, SCENARIO-PUBLISH-031.
"""

from __future__ import annotations

import json
from pathlib import Path

from carnot.reporting import capstone_v247_2580 as exp2580


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_prior_capstone(root: Path, *, overrides: dict | None = None) -> None:
    """Lay down a representative .246 capstone for carry-forward tests."""

    base = {
        "experiment": "exp2567",
        "milestone": "2026.05.246",
        "arxiv_ready_v4": True,
        "paper_errata_status": "applied_and_verified",
        "gatemate_status": {
            "ran": True,
            "terminal": True,
            "bitstream_flashed": True,
            "jtag_detected": True,
            "next_blocker": "on_board_ising_sampler_timing_benchmark_pending_capture_harness",
        },
        "kv260_status": {
            "ran": False,
            "terminal": False,
            "next_blocker": "blocked_no_sd_media_inserted_and_pynq_url_unreachable",
        },
    }
    if overrides:
        base.update(overrides)
    _write_json(root / "results" / "experiment_2567_capstone_v246.json", base)


def _write_full_happy_path_inputs(root: Path) -> None:
    """Lay down the 11 ready-state inputs for the happy-path scenario."""

    results = root / "results"
    _write_json(results / "experiment_2569_archive.json", {"honest_verdict": "complete: archived"})
    _write_json(
        results / "experiment_2570_hf_model_cards.json",
        {"honest_verdict": "complete: hf cards updated", "hf_model_cards_updated": True},
    )
    _write_json(
        results / "experiment_2571_ipfs_mirror.json",
        {
            "honest_verdict": "complete: ipfs pinned",
            "ipfs_cid": "bafy2bzaceabcdef1234567890",
        },
    )
    _write_json(
        results / "experiment_2572_tier0s_retrain.json",
        {"honest_verdict": "complete: tier0s retrained", "tier0s_improved": True},
    )
    _write_json(
        results / "experiment_2573_tier0u_fix.json",
        {"honest_verdict": "complete: tier0u fixed", "tier0u_improved": True},
    )
    _write_json(
        results / "experiment_2574_safety_corpus.json",
        {"honest_verdict": "complete: corpus built", "safety_verifier_viable": True},
    )
    _write_json(
        results / "experiment_2575_safety_ensemble.json",
        {
            "honest_verdict": "complete: safety integrated",
            "safety_integration_complete": True,
        },
    )
    _write_json(
        results / "experiment_2576_jepa_v3_online.json",
        {
            "honest_verdict": "complete: jepa online",
            "jepa_online_learning_active": True,
        },
    )
    _write_json(
        results / "experiment_2577_gatemate_continuity.json",
        {
            "honest_verdict": "complete: gatemate smoke passed",
            "gatemate_bitstream_flashed": True,
            "gatemate_smoke_test_passed": True,
            "gatemate_jtag_detected": True,
        },
    )
    _write_json(
        results / "experiment_2578_kv260_continuity.json",
        {
            "honest_verdict": "complete: kv260 validated",
            "sd_card_flashed": True,
            "kv260_workload_validated": True,
        },
    )
    _write_json(
        results / "experiment_2579_ensemble_v9.json",
        {
            "honest_verdict": "complete: v9 viable",
            "ensemble_v9_auroc": 0.9923,
            "ensemble_v9_viable": True,
            "regression_detected": False,
            "n_seeds": 5,
        },
    )
    _write_prior_capstone(root)


def test_is_terminal_verdict_accepts_required_prefixes_req_publish_031() -> None:
    """REQ-PUBLISH-031: terminal-prefix discipline drives the .247 count."""

    assert exp2580.is_terminal_verdict("complete: foo") is True
    assert exp2580.is_terminal_verdict("complete_under") is True
    assert exp2580.is_terminal_verdict("success: bar") is True
    assert exp2580.is_terminal_verdict("success_under") is True
    assert exp2580.is_terminal_verdict("passed: baz") is True
    assert exp2580.is_terminal_verdict("passed_under") is True
    assert exp2580.is_terminal_verdict("shipped: qux") is True
    assert exp2580.is_terminal_verdict("shipped_under") is True
    assert exp2580.is_terminal_verdict("  complete: ws") is True
    assert exp2580.is_terminal_verdict("blocked_precondition: no") is False
    assert exp2580.is_terminal_verdict("not_a_prefix") is False
    assert exp2580.is_terminal_verdict(None) is False
    assert exp2580.is_terminal_verdict(123) is False


def test_read_json_returns_empty_on_missing_or_invalid(tmp_path: Path) -> None:
    """Missing artifact + malformed JSON both yield empty Mapping (production contract)."""

    missing = tmp_path / "missing.json"
    assert exp2580.read_json(missing) == {}

    malformed = tmp_path / "malformed.json"
    malformed.write_text("not json at all", encoding="utf-8")
    assert exp2580.read_json(malformed) == {}

    list_payload = tmp_path / "list.json"
    list_payload.write_text("[1, 2, 3]", encoding="utf-8")
    assert exp2580.read_json(list_payload) == {}


def test_build_artifact_happy_path_scenario_publish_031(tmp_path: Path) -> None:
    """SCENARIO-PUBLISH-031: all 11 inputs present + every gate cleared.

    Operator gets submit_arxiv_now (because the .246 prior capstone
    landed arxiv_ready_v4=True + errata applied AND v9 is a strictly
    better headline than the carry-forward).
    """

    _write_full_happy_path_inputs(tmp_path)

    artifact = exp2580.build_artifact(
        tmp_path,
        started_epoch=1000.0,
        now_epoch=1000.125,
    )

    assert artifact["experiment"] == "exp2580"
    assert artifact["milestone"] == "2026.05.247"
    assert artifact["honest_verdict"].startswith("complete:")
    assert artifact["n_experiments_completed"] == 11
    assert artifact["n_planned"] == 11
    assert abs(artifact["best_247_auroc"] - 0.9923) < 1e-9
    assert artifact["best_247_auroc_source"] == "exp2579_ensemble_v9"
    assert artifact["auroc_adversarially_verified"] is True
    assert artifact["tier0s_real_improvement"] is True
    assert artifact["tier0u_real_improvement"] is True
    assert artifact["safety_classifier_viable"] is True
    assert artifact["ipfs_mirror_status"] == "pinned_cid_known"
    assert artifact["gatemate_status"]["terminal"] is True
    assert artifact["gatemate_status"]["smoke_test_passed"] is True
    assert artifact["kv260_status"]["terminal"] is True
    assert artifact["kv260_status"]["sd_card_flashed"] is True
    assert artifact["jepa_online_active"] is True
    assert artifact["operator_recommendation"] == "submit_arxiv_now"
    assert artifact["external_baselines"]["hive_peer_auroc"] == 0.9236
    assert artifact["external_baselines"]["carnot_minus_hive"] == round(0.9923 - 0.9236, 4)
    assert artifact["duration_s"] == 0.125
    assert artifact["random_seed"] == 42


def test_carry_forward_when_v9_regresses_req_publish_031(tmp_path: Path) -> None:
    """REQ-PUBLISH-031: regression_detected forces carry-forward 0.9857."""

    _write_full_happy_path_inputs(tmp_path)
    _write_json(
        tmp_path / "results" / "experiment_2579_ensemble_v9.json",
        {
            "honest_verdict": "complete: v9 regressed",
            "ensemble_v9_auroc": 0.99,
            "ensemble_v9_viable": True,
            "regression_detected": True,
            "n_seeds": 5,
        },
    )

    artifact = exp2580.build_artifact(tmp_path, started_epoch=2000.0, now_epoch=2000.0)

    assert artifact["best_247_auroc_source"] == "exp2546_v7b_carryforward"
    assert abs(artifact["best_247_auroc"] - exp2580.CARRY_FORWARD_AUROC) < 1e-9
    assert artifact["auroc_adversarially_verified"] is False


def test_carry_forward_when_v9_adversarially_flagged_req_publish_031(tmp_path: Path) -> None:
    """REQ-PUBLISH-031: flagged_adversarial=True does not displace carry-forward."""

    _write_full_happy_path_inputs(tmp_path)
    _write_json(
        tmp_path / "results" / "experiment_2579_ensemble_v9.json",
        {
            "honest_verdict": "complete: v9 flagged",
            "ensemble_v9_auroc": 0.9999,
            "ensemble_v9_viable": True,
            "regression_detected": False,
            "n_seeds": 5,
            "flagged_adversarial": True,
        },
    )

    artifact = exp2580.build_artifact(tmp_path, started_epoch=2100.0, now_epoch=2100.0)

    assert artifact["best_247_auroc_source"] == "exp2546_v7b_carryforward"


def test_carry_forward_when_v9_below_floor_req_publish_031(tmp_path: Path) -> None:
    """REQ-PUBLISH-031: v9 strictly <= carry-forward floor => keep prior."""

    _write_full_happy_path_inputs(tmp_path)
    _write_json(
        tmp_path / "results" / "experiment_2579_ensemble_v9.json",
        {
            "honest_verdict": "complete: v9 below",
            "ensemble_v9_auroc": 0.9850,
            "ensemble_v9_viable": True,
            "regression_detected": False,
            "n_seeds": 5,
        },
    )

    artifact = exp2580.build_artifact(tmp_path, started_epoch=2200.0, now_epoch=2200.0)

    assert artifact["best_247_auroc_source"] == "exp2546_v7b_carryforward"


def test_safety_viable_requires_both_gates_req_publish_031(tmp_path: Path) -> None:
    """REQ-PUBLISH-031: safety_classifier_viable needs corpus AND integration."""

    _write_full_happy_path_inputs(tmp_path)
    # Drop integration_complete.
    _write_json(
        tmp_path / "results" / "experiment_2575_safety_ensemble.json",
        {
            "honest_verdict": "complete: integration pending",
            "safety_integration_complete": False,
            "needs_more_corpus": True,
        },
    )

    artifact = exp2580.build_artifact(tmp_path, started_epoch=3000.0, now_epoch=3000.0)

    assert artifact["safety_classifier_viable"] is False


def test_ipfs_status_documented_when_no_cid_req_publish_031(tmp_path: Path) -> None:
    """REQ-PUBLISH-031: missing ipfs_cid yields documented_operator_needed."""

    _write_full_happy_path_inputs(tmp_path)
    _write_json(
        tmp_path / "results" / "experiment_2571_ipfs_mirror.json",
        {"honest_verdict": "complete: docs only", "ipfs_cid": None},
    )
    # Drop arxiv_ready_v4 so the operator recommendation falls through to
    # the HF + IPFS update branch.
    _write_prior_capstone(
        tmp_path,
        overrides={"arxiv_ready_v4": False, "paper_errata_status": "not_applied"},
    )

    artifact = exp2580.build_artifact(tmp_path, started_epoch=3100.0, now_epoch=3100.0)

    assert artifact["ipfs_mirror_status"] == "documented_operator_needed"
    # HF cards updated locally but IPFS pin not pushed => the canonical
    # follow-up sequence.
    assert artifact["operator_recommendation"] == "update_hf_cards_push_ipfs"


def test_continue_safety_branch_when_viable_and_needs_more(tmp_path: Path) -> None:
    """REQ-PUBLISH-031: viable safety classifier that needs more corpus
    flips the operator recommendation to continue_safety_classifier."""

    _write_full_happy_path_inputs(tmp_path)
    _write_json(
        tmp_path / "results" / "experiment_2575_safety_ensemble.json",
        {
            "honest_verdict": "complete: integrated",
            "safety_integration_complete": True,
            "needs_more_corpus": True,
        },
    )
    # Force the higher-precedence branches off: no arxiv, no HF cards.
    _write_prior_capstone(
        tmp_path,
        overrides={"arxiv_ready_v4": False, "paper_errata_status": "not_applied"},
    )
    _write_json(
        tmp_path / "results" / "experiment_2570_hf_model_cards.json",
        {"honest_verdict": "complete: skipped", "hf_model_cards_updated": False},
    )

    artifact = exp2580.build_artifact(tmp_path, started_epoch=3200.0, now_epoch=3200.0)

    assert artifact["safety_classifier_viable"] is True
    assert artifact["operator_recommendation"] == "continue_safety_classifier"


def test_execution_layer_gap_when_most_artifacts_missing_req_publish_031(
    tmp_path: Path,
) -> None:
    """REQ-PUBLISH-031: missing >4 of 11 artifacts triggers an EXECUTION_LAYER_GAP
    flag. This is the actual .247 scenario (no artifacts landed)."""

    # Lay down ONLY the .246 prior capstone -- nothing else.
    _write_prior_capstone(tmp_path)

    artifact = exp2580.build_artifact(tmp_path, started_epoch=4000.0, now_epoch=4000.0)

    kinds = [f["kind"] for f in artifact["process_flags"]]
    assert "EXECUTION_LAYER_GAP" in kinds
    assert artifact["n_experiments_completed"] == 0
    assert abs(artifact["best_247_auroc"] - exp2580.CARRY_FORWARD_AUROC) < 1e-9
    assert artifact["best_247_auroc_source"] == exp2580.CARRY_FORWARD_SOURCE
    assert artifact["auroc_adversarially_verified"] is False
    assert artifact["tier0s_real_improvement"] is False
    assert artifact["tier0u_real_improvement"] is False
    assert artifact["safety_classifier_viable"] is False
    assert artifact["ipfs_mirror_status"] == "documented_operator_needed"
    assert artifact["jepa_online_active"] is False
    # GateMate carries forward terminal from .246.
    assert artifact["gatemate_status"]["terminal"] is True
    assert artifact["gatemate_status"]["outcome"] == "carry_forward_from_246"
    # KV260 carries forward non-terminal from .246.
    assert artifact["kv260_status"]["terminal"] is False
    # arxiv_ready_v4 was True in prior + errata applied => submit_arxiv_now.
    assert artifact["operator_recommendation"] == "submit_arxiv_now"
    gap_areas = {g["area"] for g in artifact["top_3_gaps_for_248"]}
    assert "execution_layer_gap" in gap_areas
    assert "real_corpus_verifier_gap" in gap_areas


def test_hardware_terminal_pending_when_both_boards_not_terminal(tmp_path: Path) -> None:
    """REQ-PUBLISH-031: both boards non-terminal => hardware_terminal_pending."""

    _write_prior_capstone(
        tmp_path,
        overrides={
            "arxiv_ready_v4": False,
            "paper_errata_status": "not_applied",
            "gatemate_status": {
                "ran": False,
                "terminal": False,
                "bitstream_flashed": False,
                "next_blocker": "strtol_parse_error",
            },
            "kv260_status": {
                "ran": False,
                "terminal": False,
                "next_blocker": "no_sd_media",
            },
        },
    )

    artifact = exp2580.build_artifact(tmp_path, started_epoch=4100.0, now_epoch=4100.0)

    assert artifact["gatemate_status"]["terminal"] is False
    assert artifact["kv260_status"]["terminal"] is False
    assert artifact["operator_recommendation"] == "hardware_terminal_pending"


def test_non_terminal_verdict_surfaces_flag(tmp_path: Path) -> None:
    """REQ-PUBLISH-031: a non-terminal verdict in any input surfaces a flag."""

    _write_full_happy_path_inputs(tmp_path)
    _write_json(
        tmp_path / "results" / "experiment_2570_hf_model_cards.json",
        {
            "honest_verdict": "blocked_no_hf_token",
            "hf_model_cards_updated": False,
        },
    )

    artifact = exp2580.build_artifact(tmp_path, started_epoch=4200.0, now_epoch=4200.0)

    kinds = [f["kind"] for f in artifact["process_flags"]]
    assert "NON_TERMINAL_VERDICT" in kinds


def test_write_artifact_persists_required_fields_req_publish_031(tmp_path: Path) -> None:
    """REQ-PUBLISH-031: write_artifact emits a JSON file with every required field."""

    _write_full_happy_path_inputs(tmp_path)

    written = exp2580.write_artifact(tmp_path)

    assert written.is_file()
    payload = json.loads(written.read_text(encoding="utf-8"))
    required = {
        "honest_verdict",
        "n_experiments_completed",
        "best_247_auroc",
        "tier0s_real_improvement",
        "tier0u_real_improvement",
        "safety_classifier_viable",
        "ipfs_mirror_status",
        "gatemate_status",
        "kv260_status",
        "jepa_online_active",
        "external_baselines",
        "top_3_successes",
        "top_3_gaps_for_248",
        "process_flags",
        "preconditions_checked",
        "duration_s",
        "random_seed",
        "operator_recommendation",
    }
    assert required.issubset(payload.keys())
    assert payload["honest_verdict"].startswith("complete:")
    assert payload["random_seed"] == 42
