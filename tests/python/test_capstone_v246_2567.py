"""Tests for the Exp 2567 milestone .246 capstone synthesis.

Spec traces: REQ-PUBLISH-031, SCENARIO-PUBLISH-031.
"""

from __future__ import annotations

import json
from pathlib import Path

from carnot.reporting import capstone_v246_2567 as exp2567


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_full_happy_path_inputs(root: Path) -> None:
    """Lay down the 11 ready-state inputs for the happy-path scenario.

    Each artifact mirrors the fields the synthesis actually reads. Other
    fields are intentionally omitted to keep the fixture small and the
    test auditable.
    """

    results = root / "results"
    _write_json(results / "experiment_2556_archive.json", {"honest_verdict": "complete: archived"})
    _write_json(
        results / "experiment_2557_paper_errata.json",
        {"honest_verdict": "complete: errata applied"},
    )
    _write_json(
        results / "experiment_2558_arxiv_package_v4.json",
        {"honest_verdict": "complete: package v4 ready", "arxiv_ready_v4": True},
    )
    _write_json(
        results / "experiment_2559_gatemate_cfg_fix.json",
        {
            "honest_verdict": "success: gatemate flashed",
            "gatemate_bitstream_flashed": True,
            "gatemate_jtag_detected": True,
            "approach_b_attempted": True,
            "approach_a_attempted": False,
            "commands_executed": {"flash": "openFPGALoader -c dirtyJtag -b olimex_gatemateevb bit"},
            "diagnosis": {"fix_class": "gmpack repack"},
            "gatemate_smoke_test_result": "indirect_pass",
        },
    )
    _write_json(
        results / "experiment_2560_kv260_operator_docs.json",
        {
            "honest_verdict": "complete: docs updated",
            "operator_procedure_documented": True,
            "pynq_url_reachable": True,
            "sd_media_inserted": False,
            "kv260_workload_validated": False,
            "terminal_state_progress": "documentation_only",
            "next_blocker": "operator_insert_sd_card",
        },
    )
    _write_json(
        results / "experiment_2561_tier0t_dynamical.json",
        {"honest_verdict": "complete: tier0t prototyped", "tier0t_real_auroc": 0.72},
    )
    _write_json(
        results / "experiment_2562_tier0v_tier0w.json",
        {
            "honest_verdict": "complete: dual prototype",
            "tier0v_real_auroc": 0.55,
            "tier0w_real_auroc": 0.81,
        },
    )
    _write_json(
        results / "experiment_2563_ensemble_v8.json",
        {
            "honest_verdict": "complete: v8 improved",
            "ensemble_v8_auroc": 0.9912,
            "n_seeds": 5,
        },
    )
    _write_json(
        results / "experiment_2564_feasibility_conformal.json",
        {
            "honest_verdict": "complete: mrl conformal",
            "feasibility_conformal_auroc": 0.9880,
            "n_seeds": 5,
        },
    )
    _write_json(
        results / "experiment_2565_jepa_training.json",
        {"honest_verdict": "complete: jepa trained", "jepa_auc_improved": True},
    )
    _write_json(
        results / "experiment_2566_halluscan_eval.json",
        {
            "honest_verdict": "complete: halluscan run",
            "carnot_beats_halluscan_baseline": True,
        },
    )


def test_is_terminal_verdict_accepts_required_prefixes_req_publish_031() -> None:
    """REQ-PUBLISH-031: terminal-prefix discipline drives the .246 count."""

    assert exp2567.is_terminal_verdict("complete: foo") is True
    assert exp2567.is_terminal_verdict("complete_under") is True
    assert exp2567.is_terminal_verdict("success: bar") is True
    assert exp2567.is_terminal_verdict("success_under") is True
    assert exp2567.is_terminal_verdict("passed: baz") is True
    assert exp2567.is_terminal_verdict("passed_under") is True
    assert exp2567.is_terminal_verdict("shipped: qux") is True
    assert exp2567.is_terminal_verdict("shipped_under") is True
    assert exp2567.is_terminal_verdict("  complete: ws") is True
    assert exp2567.is_terminal_verdict("blocked_precondition: no") is False
    assert exp2567.is_terminal_verdict("not_a_prefix") is False
    assert exp2567.is_terminal_verdict(None) is False
    assert exp2567.is_terminal_verdict(123) is False


def test_read_json_returns_empty_on_missing_or_invalid(tmp_path: Path) -> None:
    """Missing artifact + malformed JSON both yield empty Mapping (production contract)."""

    missing = tmp_path / "missing.json"
    assert exp2567.read_json(missing) == {}

    malformed = tmp_path / "malformed.json"
    malformed.write_text("not json at all", encoding="utf-8")
    assert exp2567.read_json(malformed) == {}

    list_payload = tmp_path / "list.json"
    list_payload.write_text("[1, 2, 3]", encoding="utf-8")
    # Lists are valid JSON but not Mappings; the loader rejects them.
    assert exp2567.read_json(list_payload) == {}


def test_build_artifact_happy_path_scenario_publish_031(tmp_path: Path) -> None:
    """SCENARIO-PUBLISH-031: all 11 inputs present + arxiv_ready_v4=True.

    Operator gets submit_now; ensemble v8 displaces the .245 carry-forward;
    new viable verifiers count matches the >0.60 floor.
    """

    _write_full_happy_path_inputs(tmp_path)

    artifact = exp2567.build_artifact(
        tmp_path,
        started_epoch=1000.0,
        now_epoch=1000.125,
    )

    assert artifact["experiment"] == "exp2567"
    assert artifact["milestone"] == "2026.05.246"
    assert artifact["honest_verdict"].startswith("complete:")
    assert artifact["arxiv_ready_v4"] is True
    assert artifact["paper_errata_status"] == "applied_and_verified"
    assert artifact["operator_recommendation"] == "submit_now"
    # Ensemble v8 0.9912 is cite-safe and higher than MRL conformal 0.9880;
    # synthesis picks v8.
    assert abs(artifact["best_246_auroc"] - 0.9912) < 1e-9
    assert artifact["best_246_auroc_source"] == "exp2563_ensemble_v8"
    assert artifact["auroc_adversarially_verified"] is True
    assert artifact["gatemate_status"]["bitstream_flashed"] is True
    assert artifact["gatemate_status"]["terminal"] is True
    assert artifact["kv260_status"]["terminal"] is False  # SD still not inserted
    assert artifact["kv260_status"]["operator_procedure_documented"] is True
    assert artifact["jepa_auc_improved"] is True
    assert artifact["halluscan_beats_baseline"] is True
    # tier0t 0.72 and tier0w 0.81 clear 0.60; tier0v 0.55 does not.
    assert artifact["n_new_viable_verifiers"] == 2
    assert artifact["n_experiments_completed"] == 11
    assert artifact["n_planned"] == 11
    # External baselines carry through.
    assert artifact["external_baselines"]["hive_peer_auroc"] == 0.9236
    assert artifact["external_baselines"]["halluscan_peer_auroc"] == 0.67
    assert artifact["external_baselines"]["carnot_minus_hive"] == round(0.9912 - 0.9236, 4)
    assert artifact["duration_s"] == 0.125


def test_conformal_displaces_v8_when_strictly_higher_req_publish_031(tmp_path: Path) -> None:
    """REQ-PUBLISH-031: when MRL conformal is strictly higher than v8 AND
    both are clean, conformal becomes the headline source."""

    _write_full_happy_path_inputs(tmp_path)
    # Make conformal higher than v8.
    _write_json(
        tmp_path / "results" / "experiment_2564_feasibility_conformal.json",
        {
            "honest_verdict": "complete: mrl conformal",
            "feasibility_conformal_auroc": 0.9950,
            "n_seeds": 5,
        },
    )

    artifact = exp2567.build_artifact(tmp_path, started_epoch=2000.0, now_epoch=2000.0)

    assert artifact["best_246_auroc_source"] == "exp2564_feasibility_conformal"
    assert abs(artifact["best_246_auroc"] - 0.9950) < 1e-9


def test_v8_clean_above_carryforward_when_conformal_missing_req_publish_031(
    tmp_path: Path,
) -> None:
    """REQ-PUBLISH-031: when conformal artifact is empty but v8 is clean
    AND v8 > carry-forward, the synthesis picks v8."""

    _write_full_happy_path_inputs(tmp_path)
    _write_json(tmp_path / "results" / "experiment_2564_feasibility_conformal.json", {})

    artifact = exp2567.build_artifact(tmp_path, started_epoch=2100.0, now_epoch=2100.0)

    assert artifact["best_246_auroc_source"] == "exp2563_ensemble_v8"
    assert abs(artifact["best_246_auroc"] - 0.9912) < 1e-9


def test_v8_dirty_falls_through_to_conformal_when_clean_req_publish_031(
    tmp_path: Path,
) -> None:
    """REQ-PUBLISH-031: a flagged-adversarial v8 cannot displace carry-forward;
    if conformal is clean and above carry-forward, conformal wins instead."""

    _write_full_happy_path_inputs(tmp_path)
    _write_json(
        tmp_path / "results" / "experiment_2563_ensemble_v8.json",
        {
            "honest_verdict": "complete: v8 flagged",
            "ensemble_v8_auroc": 0.9999,
            "n_seeds": 5,
            "flagged_adversarial": True,
        },
    )

    artifact = exp2567.build_artifact(tmp_path, started_epoch=2200.0, now_epoch=2200.0)

    # Conformal 0.9880 is clean and > carry-forward 0.9857.
    assert artifact["best_246_auroc_source"] == "exp2564_feasibility_conformal"


def test_execution_layer_gap_when_most_artifacts_missing_req_publish_031(
    tmp_path: Path,
) -> None:
    """REQ-PUBLISH-031: missing >3 of 11 artifacts triggers an EXECUTION_LAYER_GAP
    flag. This is the actual .246 scenario (only exp2559 landed)."""

    # Only land exp2559 to match the real .246 state.
    _write_json(
        tmp_path / "results" / "experiment_2559_gatemate_cfg_fix.json",
        {
            "honest_verdict": "success: gatemate flashed",
            "gatemate_bitstream_flashed": True,
            "gatemate_jtag_detected": True,
            "approach_b_attempted": True,
            "commands_executed": {"flash": "openFPGALoader cmd"},
            "diagnosis": {"fix_class": "gmpack repack"},
        },
    )

    artifact = exp2567.build_artifact(tmp_path, started_epoch=3000.0, now_epoch=3000.0)

    kinds = [f["kind"] for f in artifact["process_flags"]]
    assert "EXECUTION_LAYER_GAP" in kinds
    assert artifact["arxiv_ready_v4"] is False
    assert artifact["paper_errata_status"] == "not_applied"
    # When neither v8 nor conformal landed, carry forward .245 ensemble v7b.
    assert abs(artifact["best_246_auroc"] - exp2567.CARRY_FORWARD_AUROC) < 1e-9
    assert artifact["best_246_auroc_source"] == exp2567.CARRY_FORWARD_SOURCE
    assert artifact["auroc_adversarially_verified"] is False
    # GateMate is the only success story; capstone surfaces it.
    assert artifact["gatemate_status"]["bitstream_flashed"] is True
    assert artifact["gatemate_status"]["terminal"] is True
    # KV260 carries forward from .245 (operator action pending).
    assert artifact["kv260_status"]["terminal"] is False
    assert artifact["kv260_status"]["ran"] is False
    assert artifact["n_new_viable_verifiers"] == 0
    assert artifact["jepa_auc_improved"] is False
    assert artifact["halluscan_beats_baseline"] is False
    # Two critical flags (EXECUTION_LAYER_GAP only counts once but the
    # not_applied errata path drives apply_errata_first).
    assert artifact["operator_recommendation"] in {
        "apply_errata_first",
        "request_operator_decision",
    }
    # Top gaps include both the paper-errata pending and verifier-expansion-blocked.
    gap_areas = {g["area"] for g in artifact["top_3_gaps_for_247"]}
    assert "paper_errata_pending" in gap_areas
    assert "verifier_expansion_blocked" in gap_areas


def test_paper_errata_applied_not_repackaged_when_2558_missing_req_publish_031(
    tmp_path: Path,
) -> None:
    """REQ-PUBLISH-031: errata applied (exp2557 terminal) but repackage
    missing (exp2558 empty) yields applied_not_repackaged + apply_errata_first."""

    _write_full_happy_path_inputs(tmp_path)
    _write_json(tmp_path / "results" / "experiment_2558_arxiv_package_v4.json", {})

    artifact = exp2567.build_artifact(tmp_path, started_epoch=4000.0, now_epoch=4000.0)

    assert artifact["paper_errata_status"] == "applied_not_repackaged"
    assert artifact["arxiv_ready_v4"] is False
    assert artifact["operator_recommendation"] == "apply_errata_first"


def test_request_operator_decision_when_multiple_critical_flags_req_publish_031(
    tmp_path: Path,
) -> None:
    """REQ-PUBLISH-031: more than one critical process_flag forces
    request_operator_decision regardless of errata state."""

    _write_full_happy_path_inputs(tmp_path)
    # Land two adversarial-flagged artifacts so we get two CRITICAL severities.
    _write_json(
        tmp_path / "results" / "experiment_2563_ensemble_v8.json",
        {
            "honest_verdict": "complete: v8 done",
            "ensemble_v8_auroc": 0.9912,
            "n_seeds": 5,
            "flagged_adversarial": True,
            "corrigendum_pending": [
                {"kind": "TAUTOLOGY", "severity": "critical", "detail": "baseline duplicated"}
            ],
        },
    )
    _write_json(
        tmp_path / "results" / "experiment_2564_feasibility_conformal.json",
        {
            "honest_verdict": "complete: conformal done",
            "feasibility_conformal_auroc": 0.9880,
            "n_seeds": 5,
            "flagged_adversarial": True,
            "corrigendum_pending": [
                {"kind": "TAUTOLOGY", "severity": "critical", "detail": "ratio matches"}
            ],
        },
    )

    artifact = exp2567.build_artifact(tmp_path, started_epoch=5000.0, now_epoch=5000.0)

    # Both flagged_adversarial => 2 critical entries in process_flags.
    n_critical = sum(
        1 for f in artifact["process_flags"] if f.get("severity") == "critical"
    )
    assert n_critical >= 2
    # Errata still applied + repackaged in this fixture, so without the
    # multiple-critical override the recommendation would be submit_now.
    # The override flips it to request_operator_decision.
    assert artifact["operator_recommendation"] == "request_operator_decision"


def test_write_artifact_persists_required_fields_req_publish_031(tmp_path: Path) -> None:
    """REQ-PUBLISH-031: write_artifact emits a JSON file with every required field."""

    _write_full_happy_path_inputs(tmp_path)

    written = exp2567.write_artifact(tmp_path)

    assert written.is_file()
    payload = json.loads(written.read_text(encoding="utf-8"))
    required = {
        "honest_verdict",
        "n_experiments_completed",
        "best_246_auroc",
        "arxiv_ready_v4",
        "paper_errata_status",
        "gatemate_status",
        "kv260_status",
        "jepa_auc_improved",
        "halluscan_beats_baseline",
        "n_new_viable_verifiers",
        "external_baselines",
        "top_3_successes",
        "top_3_gaps_for_247",
        "process_flags",
        "preconditions_checked",
        "duration_s",
        "random_seed",
    }
    assert required.issubset(payload.keys())
    assert payload["honest_verdict"].startswith("complete:")
    assert payload["random_seed"] == 42
