"""Tests for Exp 4268 .394 capstone aggregation.

Spec refs: REQ-CAPSTONE-4268, SCENARIO-CAPSTONE-4268.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from typing import Any

import pytest


MODULE_PATH = Path("python/carnot/reporting/capstone_v394_4268.py")
SPEC = importlib.util.spec_from_file_location("capstone_v394_4268", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
mod = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = mod
SPEC.loader.exec_module(mod)


JsonDict = dict[str, Any]


def _write_json(path: Path, payload: JsonDict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _clean_live_flags(_: Path) -> list[dict[str, str]]:
    return []


def _summarize_zero(_: Path, __: Path) -> int:
    return 0


def _paper_gate(paper_ready: bool = True) -> JsonDict:
    return {
        "paper_ready": paper_ready,
        "gates": {
            "G1": {"pass": True, "detail": "fixture"},
            "G2": {"pass": paper_ready, "detail": "fixture"},
            "G3": {"pass": True, "detail": "fixture"},
            "G4": {"pass": True, "detail": "fixture"},
        },
        "unmet_gates": [] if paper_ready else ["G2"],
        "note": "fixture publication gate",
    }


def _minimal_payloads() -> dict[str, JsonDict]:
    return {
        "4256_leak_audit": {
            "honest_verdict": "complete: leak audit fixture",
            "win_survives_provenance_blind": True,
            "provenance_blind_delta": 0.38,
            "provenance_blind_ci95": [0.25, 0.52],
            "origin_probe_auroc": 0.94,
            "origin_correctness_corr": 0.9,
            "held_out_task_n": 52,
            "oracle_at_k": 0.83,
            "verifier_is_oracle": False,
            "reproducibility_checksum": "sha256:a",
        },
        "4257_multiseed": {
            "honest_verdict": "complete: multiseed fixture",
            "flagged_adversarial": False,
            "oracle_distinct_win_replicates": True,
            "mean_delta": 0.45,
            "cross_seed_ci95": [0.4, 0.5],
            "cross_seed_ci95_excludes_zero": True,
            "n_seeds": 5,
            "independent_rescore_delta": 0.44,
            "verifier_is_oracle": False,
            "reproducibility_checksum": "sha256:b",
        },
        "4258_cross_game": {
            "honest_verdict": "complete: cross game fixture",
            "cross_game_delta": 0.12,
            "cross_game_ci95": [0.03, 0.2],
            "ci95_excludes_zero": True,
            "held_out_game_n": 3,
            "held_out_task_n": 60,
            "headroom_exists": True,
            "verifier_is_oracle": False,
            "reproducibility_checksum": "sha256:c",
        },
        "4259_synthesis": {
            "honest_verdict": "complete: synthesis fixture",
            "flagged_adversarial": False,
            "synthesis_breaks_oracle_ceiling": True,
            "synthesis_beats_selection": True,
            "synthesis_minus_oracle_delta": 0.04,
            "synthesis_minus_oracle_ci95": [0.01, 0.08],
            "synthesis_minus_vote_delta": 0.5,
            "ceiling_break_task_n": 2,
            "oracle_at_k": 0.9,
            "verifier_is_oracle": False,
            "reproducibility_checksum": "sha256:d",
        },
        "4260_preflight": {
            "honest_verdict": "complete: preflight fixture",
            "flagged_adversarial": False,
            "preflight_go": True,
            "guidance_changes_selection": True,
            "guidance_selection_change_count": 3,
            "full_run_cost_estimate_s": 600.0,
            "verifier_is_oracle": False,
            "reproducibility_checksum": "e" * 64,
        },
        "4261_arc_progress": {
            "honest_verdict": "complete: progress fixture",
            "total_levels": 21,
            "total_levels_solved": 21,
            "levels_completed": 2,
            "new_levels_solved_this_task": 2,
            "prior_total_levels_solved": 19,
            "game_advanced": "fixture-game",
            "game_levels_completed": 6,
            "reproducibility_checksum": "f" * 64,
        },
        "4262_live_probe": {
            "honest_verdict": "complete: live probe fixture",
            "levels_completed": 1,
            "actions_taken": 4,
            "baseline_actions": 20,
            "actions_vs_baseline_ratio": 0.2,
            "environment_score": {"levels_completed": 1, "score": 1.0},
            "game_probed": "fixture-live",
            "scorecard_closed": True,
            "reproducibility_checksum": "1" * 64,
        },
        "4263_reward": {
            "honest_verdict": "complete: reward fixture",
            "ready_for_out_of_band": True,
            "verifier_as_reward_retired": False,
            "out_of_band_runner_path": "results/run_reward.py",
            "weighted_corpus": {"supports_clean_avsb": True, "rows": 100},
            "verifier_is_oracle": True,
            "reproducibility_checksum": "sha256:g",
        },
        "4264_code": {
            "honest_verdict": "complete: code fixture",
            "code_replication_beats_vote": True,
            "code_replication_retired": False,
            "replication_read": "replicates",
            "code_predictor_minus_vote_delta": 0.03,
            "code_predictor_minus_vote_ci95": [0.01, 0.05],
            "ci95_excludes_zero": True,
            "held_out_task_n": 160,
            "headroom_exists": True,
            "oracle_at_k": 0.96,
            "oracle_minus_vote": 0.04,
            "off_fold_auroc": 0.72,
            "verifier_is_oracle": False,
            "reproducibility_checksum": "2" * 64,
        },
        "4266_registry": {
            "honest_verdict": "complete: registry fixture",
            "flagged_adversarial": False,
            "registry_reconciled": True,
            "regression_guard_passed": True,
            "v394_hardened_state": "fixture hardened",
            "gaps_logged": [{"gap_id": "fixture-gap"}],
            "reproducibility_checksum": "sha256:h",
        },
        "4267_hardware": {
            "honest_verdict": "complete: hardware fixture",
            "kv260_terminal_confirmed": True,
            "kv260_step_taken": "kv260_terminal_confirmed",
            "polarfire_step_taken": "polarfire_hash_verified_cpu_dispatch_succeeded",
            "gatemate_step_taken": "gatemate_idcode_detected",
            "per_board_reachability": {"kv260": True, "polarfire": True, "gatemate": True},
            "fabric_acceleration_claimed": False,
            "speedup_claim_made": False,
            "reproducibility_checksum": "3" * 64,
        },
    }


def _write_default_artifacts(root: Path, payloads: dict[str, JsonDict]) -> None:
    for key, payload in payloads.items():
        _write_json(root / mod.DEFAULT_UPSTREAMS[key].path, payload)


def _expect_validation_error(artifact: JsonDict, mutator: Any, match: str) -> None:
    mutated = json.loads(json.dumps(artifact))
    mutator(mutated)
    with pytest.raises(ValueError, match=match):
        mod.validate_artifact(mutated)


def test_req_capstone_4268_spec_anchor_declares_required_contract() -> None:
    """REQ-CAPSTONE-4268: OpenSpec declares the .394 capstone contract."""

    spec = Path("openspec/capabilities/capstone/spec.md").read_text(encoding="utf-8")

    assert "REQ-CAPSTONE-4268" in spec
    assert "SCENARIO-CAPSTONE-4268" in spec
    assert "experiment_4268_capstone_v394.json" in spec
    assert "cross_game_delta>0" in spec
    assert "publication_gate.py --json" in spec
    assert "blocked_v394_artifacts_missing" in spec
    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert field in spec
        assert mod.FIELD_PRINCIPLES[field] in spec
    assert "flagged_adversarial:true" in spec
    assert "verifier_is_oracle:false" in spec


def test_scenario_capstone_4268_current_artifacts_report_honest_close_state() -> None:
    """SCENARIO-CAPSTONE-4268: current .394 artifacts produce the honest headline."""

    artifact = mod.build_artifact(Path.cwd(), started_s=1.0, now_s=1.5)

    mod.validate_artifact(artifact)
    assert artifact["headline_outcome"] == (
        "within_pool_win_survived_but_cross_game_blocked_"
        "synthesis_excluded_preflight_excluded_arc19_live0_"
        "reward_out_of_band_code_corpus_specific_paper_ready"
    )
    assert artifact["hardened_win"] is False
    assert artifact["diffusiongemma_full_run_gate"] is False
    assert artifact["verifier_is_oracle_honored"] is True
    assert artifact["paper_ready"] is True
    assert artifact["unmet_gates"] == []
    assert artifact["publication_gate"]["paper_ready"] is True

    assert artifact["hardening"]["provenance_blind"]["win_survives_provenance_blind"] is True
    assert artifact["hardening"]["multiseed"]["oracle_distinct_win_replicates"] is True
    assert artifact["hardening"]["cross_game"]["cross_game_delta"] is None
    assert artifact["hardening"]["cross_game"]["status"] == "blocked_or_no_positive_delta"
    assert artifact["extend_synthesis"]["status"] == "excluded_flagged_adversarial"
    assert artifact["scale_up_readiness"]["status"] == "excluded_flagged_adversarial"
    assert artifact["scale_up_readiness"]["preflight_go"] is False
    assert artifact["arc_progress"]["total_levels_solved"] == 19
    assert artifact["live_arc_probe"]["levels_completed"] == 0
    assert artifact["live_arc_probe"]["actions_vs_baseline_ratio"] == pytest.approx(
        0.29411764705882354
    )
    assert artifact["reward_decision"]["ready_for_out_of_band"] is True
    assert artifact["reward_decision"]["verifier_is_oracle"] is True
    assert artifact["code_read"]["replication_read"] == "corpus_specific"
    assert artifact["code_read"]["code_replication_beats_vote"] is False
    assert artifact["registry_read"]["status"] == "excluded_flagged_adversarial"
    assert artifact["hardware_read"]["polarfire_step_taken"] == (
        "polarfire_hash_verified_cpu_dispatch_succeeded"
    )

    excluded = {row["artifact_key"]: row for row in artifact["flagged_artifacts_excluded"]}
    assert set(excluded) == {"4259_synthesis", "4260_preflight", "4266_registry"}
    assert excluded["4259_synthesis"]["stamped_flagged_adversarial"] is True
    assert excluded["4260_preflight"]["live_critical"] is True
    assert excluded["4266_registry"]["live_critical"] is True

    provenance = {row["artifact_key"]: row for row in artifact["upstream_provenance"]}
    assert set(provenance) == set(mod.DEFAULT_UPSTREAMS)
    for key, upstream in mod.DEFAULT_UPSTREAMS.items():
        expected_sha = hashlib.sha256((Path.cwd() / upstream.path).read_bytes()).hexdigest()
        assert provenance[key]["sha256"] == expected_sha
    for key in excluded:
        assert provenance[key]["skipped"] is True
        assert provenance[key]["fields_imported"] == []
    assert "win_survives_provenance_blind" in provenance["4256_leak_audit"]["fields_imported"]
    assert "oracle_distinct_win_replicates" in provenance["4257_multiseed"]["fields_imported"]
    assert "cross_game_delta" in provenance["4258_cross_game"]["fields_imported"]
    assert artifact["reproducibility_checksum"] == mod.checksum_from_provenance(
        artifact["upstream_provenance"]
    )
    assert artifact["honest_verdict"].startswith("complete: capstone_v394_")


def test_req_capstone_4268_clean_fixture_can_open_diffusiongemma_gate(tmp_path: Path) -> None:
    """REQ-CAPSTONE-4268: clean hardening plus clean preflight flips the full-run gate."""

    payloads = _minimal_payloads()
    _write_default_artifacts(tmp_path, payloads)

    artifact = mod.build_artifact(
        tmp_path,
        started_s=2.0,
        now_s=2.25,
        live_flag_runner=_clean_live_flags,
        summarize_runner=_summarize_zero,
        publication_gate_runner=lambda _: _paper_gate(False),
    )

    mod.validate_artifact(artifact)
    assert artifact["hardened_win"] is True
    assert artifact["diffusiongemma_full_run_gate"] is True
    assert artifact["headline_outcome"] == (
        "hardened_win_true_synthesis_breaks_oracle_ceiling_"
        "diffusiongemma_full_run_ready_arc21_live1_reward_out_of_band_"
        "code_replicates_paper_not_ready"
    )
    assert artifact["paper_ready"] is False
    assert artifact["unmet_gates"] == ["G2"]
    assert artifact["extend_synthesis"]["synthesis_breaks_oracle_ceiling"] is True
    assert artifact["scale_up_readiness"]["preflight_go"] is True
    assert artifact["flagged_artifacts_excluded"] == []
    assert artifact["registry_read"]["regression_guard_passed"] is True
    assert artifact["hardware_read"]["kv260_terminal_confirmed"] is True
    assert artifact["reproducibility_checksum"] == mod.checksum_from_provenance(
        artifact["upstream_provenance"]
    )


def test_req_capstone_4268_excludes_live_critical_and_oracle_circular_inputs(
    tmp_path: Path,
) -> None:
    """REQ-CAPSTONE-4268: flagged/live-critical/circular inputs cannot headline."""

    payloads = _minimal_payloads()
    payloads["4256_leak_audit"]["verifier_is_oracle"] = True
    payloads["4258_cross_game"]["cross_game_delta"] = 0.2
    _write_default_artifacts(tmp_path, payloads)

    def live_flags(path: Path) -> list[dict[str, str]]:
        if path.name == mod.DEFAULT_UPSTREAMS["4258_cross_game"].path.name:
            return [{"kind": "DURATION_TOO_SHORT", "severity": "critical", "detail": "fixture"}]
        return []

    artifact = mod.build_artifact(
        tmp_path,
        started_s=3.0,
        now_s=3.25,
        live_flag_runner=live_flags,
        summarize_runner=_summarize_zero,
        publication_gate_runner=lambda _: _paper_gate(True),
    )

    mod.validate_artifact(artifact)
    assert artifact["hardened_win"] is False
    assert artifact["diffusiongemma_full_run_gate"] is False
    assert artifact["verifier_is_oracle_honored"] is True
    assert artifact["hardening"]["provenance_blind"]["used_as_oracle_distinct"] is False
    assert artifact["hardening"]["cross_game"]["status"] == "excluded_flagged_adversarial"
    excluded = {row["artifact_key"] for row in artifact["flagged_artifacts_excluded"]}
    assert excluded == {"4258_cross_game"}
    provenance = {row["artifact_key"]: row for row in artifact["upstream_provenance"]}
    assert provenance["4258_cross_game"]["fields_imported"] == []


def test_req_capstone_4268_missing_or_malformed_core_blocks_without_aggregation(
    tmp_path: Path,
) -> None:
    """REQ-CAPSTONE-4268: missing or unparsable upstreams produce blocked verdicts."""

    payloads = _minimal_payloads()
    payloads.pop("4260_preflight")
    _write_default_artifacts(tmp_path, payloads)
    malformed_path = tmp_path / mod.DEFAULT_UPSTREAMS["4266_registry"].path
    malformed_path.write_text("[]\n", encoding="utf-8")

    artifact = mod.build_artifact(
        tmp_path,
        started_s=4.0,
        now_s=4.25,
        live_flag_runner=_clean_live_flags,
        summarize_runner=_summarize_zero,
        publication_gate_runner=lambda _: _paper_gate(True),
    )

    mod.validate_artifact(artifact)
    assert artifact["honest_verdict"] == "blocked_v394_artifacts_missing"
    assert artifact["headline_outcome"] == "blocked_v394_artifacts_missing"
    assert artifact["hardened_win"] is False
    assert artifact["diffusiongemma_full_run_gate"] is False
    assert artifact["paper_ready"] is None
    assert artifact["missing_upstream_artifacts"] == [
        {"artifact_key": "4260_preflight", "experiment_id": 4260, "reason": "missing"},
        {
            "artifact_key": "4266_registry",
            "experiment_id": 4266,
            "reason": "unparsable_or_non_object",
        },
    ]
    assert artifact["upstream_provenance"] == []
    assert artifact["reproducibility_checksum"] == mod.BLOCKED_CHECKSUM


def test_req_capstone_4268_validation_write_and_helpers(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    """REQ-CAPSTONE-4268: validation protects schema, checksums, and writes."""

    payloads = _minimal_payloads()
    _write_default_artifacts(tmp_path, payloads)
    artifact = mod.build_artifact(
        tmp_path,
        started_s=5.0,
        now_s=5.5,
        live_flag_runner=_clean_live_flags,
        summarize_runner=_summarize_zero,
        publication_gate_runner=lambda _: _paper_gate(True),
    )
    out_path = mod.write_artifact(
        tmp_path,
        output_path=Path("results/experiment_4268_capstone_v394.json"),
        started_s=6.0,
        now_s=6.25,
        live_flag_runner=_clean_live_flags,
        summarize_runner=_summarize_zero,
        publication_gate_runner=lambda _: _paper_gate(True),
    )
    written = json.loads(out_path.read_text(encoding="utf-8"))
    mod.validate_artifact(written)
    assert written["reproducibility_checksum"] == mod.checksum_from_provenance(
        written["upstream_provenance"]
    )

    assert mod.bool_metric({"x": True}, "x") is True
    assert mod.bool_metric({"x": 1}, "x") is None
    assert mod.int_metric({"x": 2}, "x") == 2
    assert mod.int_metric({"x": True}, "x") == 0
    assert mod.float_metric({"x": 2}, "x") == pytest.approx(2.0)
    assert mod.float_metric({"x": True}, "x") is None
    assert mod.str_metric({"x": "ok"}, "x") == "ok"
    assert mod.str_metric({"x": 1}, "x") == ""
    assert mod.nested_map({"x": {"y": 1}}, "x") == {"y": 1}
    assert mod.nested_map({"x": []}, "x") == {}
    assert mod.sha_from_payload_checksum({"reproducibility_checksum": "sha256:" + "a" * 64}) == (
        "a" * 64
    )
    assert mod.sha_from_payload_checksum({"reproducibility_checksum": "b" * 64}) == "b" * 64
    assert mod.sha_from_payload_checksum({}) == ""
    assert mod.live_has_critical([{"severity": "critical"}]) is True
    assert mod.live_has_critical([{"severity": "warn"}]) is False
    assert mod.clean_payload(None, False) is None
    assert mod.clean_payload({"x": 1}, True) is None
    assert mod.clean_payload({"x": 1}, False) == {"x": 1}
    assert mod.checksum_from_provenance([]) == mod.BLOCKED_CHECKSUM
    assert mod._ci_excludes_zero({"x": [1.0]}, "x") is False
    assert mod._ci_excludes_zero({"x": ["bad", 1.0]}, "x") is False
    assert mod._ci_excludes_zero({"x": [1.0, "bad"]}, "x") is False
    assert mod.reward_read({"verifier_as_reward_retired": True})["status"] == "reward_retired"
    assert mod.reward_read({})["status"] == "reward_pending"
    assert mod.code_read({})["status"] == "code_no_win"
    assert mod.headline_string(True, True, True, 1, 2, "reward", "code", True) == (
        "hardened_win_true_synthesis_breaks_oracle_ceiling_"
        "diffusiongemma_full_run_ready_arc1_live2_reward_code_paper_ready"
    )
    monkeypatch.setattr(
        mod.subprocess,
        "run",
        lambda *args, **kwargs: type("Proc", (), {"stdout": "[]\n"})(),
    )
    with pytest.raises(ValueError, match="publication_gate.py --json"):
        mod.run_publication_gate(tmp_path)

    _expect_validation_error(artifact, lambda a: a.pop("honest_verdict"), "missing required")
    _expect_validation_error(artifact, lambda a: a.update({"honest_verdict": "bad"}), "terminal")
    _expect_validation_error(artifact, lambda a: a.update({"headline_outcome": ""}), "headline")
    _expect_validation_error(artifact, lambda a: a.update({"hardened_win": "bad"}), "hardened")
    _expect_validation_error(
        artifact,
        lambda a: a.update({"diffusiongemma_full_run_gate": "bad"}),
        "DiffusionGemma",
    )
    _expect_validation_error(
        artifact,
        lambda a: a.update({"diffusiongemma_full_run_gate": True, "hardened_win": False}),
        "full-run gate",
    )
    _expect_validation_error(
        artifact,
        lambda a: a.update({"verifier_is_oracle_honored": False}),
        "oracle-distinctness",
    )
    _expect_validation_error(artifact, lambda a: a.update({"paper_ready": "bad"}), "paper_ready")
    _expect_validation_error(
        artifact, lambda a: a.update({"field_principles": []}), "field_principles"
    )
    _expect_validation_error(
        artifact,
        lambda a: a["field_principles"].update({"headline_outcome": "wrong"}),
        "principle",
    )
    _expect_validation_error(
        artifact, lambda a: a.update({"flagged_artifacts_excluded": {}}), "flagged"
    )
    _expect_validation_error(
        artifact, lambda a: a["flagged_artifacts_excluded"].append("bad"), "flagged entries"
    )
    _expect_validation_error(
        artifact, lambda a: a.update({"upstream_provenance": {}}), "upstream_provenance"
    )
    _expect_validation_error(
        artifact, lambda a: a["upstream_provenance"].append(42), "provenance entries"
    )
    _expect_validation_error(
        artifact,
        lambda a: a["upstream_provenance"][0].update({"experiment_id": "4256"}),
        "integer experiment_id",
    )
    _expect_validation_error(
        artifact, lambda a: a["upstream_provenance"][0].update({"sha256": "bad"}), "sha256"
    )
    _expect_validation_error(
        artifact,
        lambda a: a["upstream_provenance"][0].update({"fields_imported": "bad"}),
        "fields_imported",
    )
    skipped = json.loads(json.dumps(artifact))
    skipped["upstream_provenance"][0]["skipped"] = True
    skipped["upstream_provenance"][0]["fields_imported"] = ["bad"]
    with pytest.raises(ValueError, match="skipped upstreams"):
        mod.validate_artifact(skipped)
    _expect_validation_error(
        artifact, lambda a: a["upstream_provenance"][0].pop("artifact_key"), "artifact_key"
    )
    _expect_validation_error(
        artifact, lambda a: a.update({"reproducibility_checksum": "bad"}), "checksum"
    )
    _expect_validation_error(
        artifact,
        lambda a: a["upstream_provenance"][0].update({"sha256": "a" * 64}),
        "checksum",
    )
    blocked_bad = json.loads(json.dumps(artifact))
    blocked_bad["honest_verdict"] = "blocked_v394_artifacts_missing"
    blocked_bad["headline_outcome"] = "blocked_v394_artifacts_missing"
    blocked_bad["paper_ready"] = True
    blocked_bad["upstream_provenance"] = []
    blocked_bad["reproducibility_checksum"] = mod.BLOCKED_CHECKSUM
    with pytest.raises(ValueError, match="blocked artifacts"):
        mod.validate_artifact(blocked_bad)

    output_path = tmp_path / "results" / "experiment_4268_capstone_v394.json"
    monkeypatch.setattr(mod, "write_artifact", lambda root: output_path)
    assert mod.main() == 0
    assert str(output_path) in capsys.readouterr().out
