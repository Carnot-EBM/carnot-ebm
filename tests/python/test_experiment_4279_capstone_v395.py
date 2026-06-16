"""Tests for Exp 4279 .395 capstone aggregation.

Spec refs: REQ-CAPSTONE-4279, SCENARIO-CAPSTONE-4279.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from carnot.reporting import capstone_v395_4279 as mod


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
            "provenance_blind_ci95": [0.21, 0.5],
            "verifier_is_oracle": False,
            "reproducibility_checksum": "sha256:" + "a" * 64,
        },
        "4257_multiseed": {
            "honest_verdict": "complete: multiseed fixture",
            "oracle_distinct_win_replicates": True,
            "mean_delta": 0.44,
            "cross_seed_ci95": [0.31, 0.55],
            "cross_seed_ci95_excludes_zero": True,
            "n_seeds": 5,
            "verifier_is_oracle": False,
            "reproducibility_checksum": "sha256:" + "b" * 64,
        },
        "4270_family_recovery": {
            "honest_verdict": "complete: family recovery fixture",
            "family_split_feasible": True,
            "distinct_family_n": 8,
            "provenance_manifest_path": "results/fixture_manifest.json",
            "verifier_is_oracle": False,
            "reproducibility_checksum": "sha256:" + "c" * 64,
        },
        "4271_cross_family_existing": {
            "honest_verdict": "complete: cross family fixture",
            "cross_family_win_holds": True,
            "cross_family_delta": 0.2,
            "cross_family_ci95": [0.05, 0.33],
            "ci95_excludes_zero": True,
            "within_minus_cross_gap": 0.24,
            "held_out_family_n": 8,
            "held_out_task_n": 48,
            "oracle_at_k": 0.8,
            "matched_control_delta": 0.44,
            "online_adapt_cross_family_delta": 0.22,
            "verifier_is_oracle": False,
            "reproducibility_checksum": "sha256:" + "d" * 64,
        },
        "4272_cross_family_fresh": {
            "honest_verdict": "blocked_gate_check_failed",
            "status": "blocked",
            "gate_check_summary": "fixture gate blocked",
        },
        "4273_self_learning": {
            "honest_verdict": "complete: online adaptation helps fixture",
            "online_adaptation_helps": True,
            "static_cross_family_delta": 0.2,
            "online_cross_family_delta": 0.31,
            "online_minus_static_ci95": [0.04, 0.2],
            "verifier_is_oracle": False,
            "reproducibility_checksum": "sha256:" + "e" * 64,
        },
        "4274_preflight": {
            "honest_verdict": "complete: preflight fixture",
            "loader_repaired": True,
            "preflight_go": True,
            "guidance_changes_selection": True,
            "full_run_cost_estimate_s": 100.0,
            "verifier_is_oracle": False,
            "reproducibility_checksum": "f" * 64,
        },
        "4275_arc_progress": {
            "honest_verdict": "success: arc progress fixture",
            "total_levels": 22,
            "total_levels_solved": 22,
            "levels_completed": 2,
            "new_levels_solved_this_task": 2,
            "game_advanced": "fixture-game",
            "reproducibility_checksum": "1" * 64,
        },
        "4277_registry": {
            "honest_verdict": "complete: registry fixture",
            "registry_reconciled": True,
            "regression_guard_passed": True,
            "reproducibility_checksum": "sha256:" + "2" * 64,
        },
        "4278_hardware": {
            "honest_verdict": "complete: hardware fixture",
            "kv260_terminal_confirmed": True,
            "polarfire_step_taken": "polarfire_hash_verified_cpu_dispatch_succeeded",
            "gatemate_step_taken": "gatemate_idcode_detected",
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


def test_req_capstone_4279_spec_anchor_declares_required_contract() -> None:
    """REQ-CAPSTONE-4279: OpenSpec declares the .395 capstone contract."""

    spec = Path("openspec/capabilities/capstone/spec.md").read_text(encoding="utf-8")

    assert "REQ-CAPSTONE-4279" in spec
    assert "SCENARIO-CAPSTONE-4279" in spec
    assert "experiment_4279_capstone_v395.json" in spec
    assert "cross_family_generalizes" in spec
    assert "blocked_v395_artifacts_missing" in spec
    assert "publication_gate.py --json" in spec
    assert "flagged_adversarial:true" in spec
    assert "verifier_is_oracle:false" in spec
    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert field in spec
        assert mod.FIELD_PRINCIPLES[field] in spec


def test_scenario_capstone_4279_current_artifacts_report_ood_gate() -> None:
    """SCENARIO-CAPSTONE-4279: current .395 artifacts produce the honest headline."""

    artifact = mod.build_artifact(Path.cwd(), started_s=1.0, now_s=1.5)

    mod.validate_artifact(artifact)
    assert artifact["headline_outcome"] == (
        "cross_family_generalizes_self_learning_static_ceiling_"
        "diffusiongemma_full_run_ready_arc20_game_wa30-ee6fef47_paper_ready"
    )
    assert artifact["cross_family_generalizes"] is True
    assert artifact["hardened_win"] is True
    assert artifact["diffusiongemma_full_run_gate"] is True
    assert artifact["paper_ready"] is True
    assert artifact["unmet_gates"] == []
    assert artifact["publication_gate"]["paper_ready"] is True
    assert artifact["verifier_is_oracle_honored"] is True
    assert artifact["flagged_artifacts_excluded"] == []

    assert artifact["hardening"]["provenance_blind"]["win_survives_provenance_blind"] is True
    assert artifact["hardening"]["multiseed"]["oracle_distinct_win_replicates"] is True
    assert artifact["cross_family"]["source_artifact_key"] == "4271_cross_family_existing"
    assert artifact["cross_family"]["cross_family_delta"] == pytest.approx(0.4038461538)
    assert artifact["cross_family"]["within_minus_cross_gap"] == pytest.approx(0.0384615385)
    assert artifact["self_learning"]["online_adaptation_helps"] is False
    assert artifact["self_learning"]["status"] == "static_ceiling"
    assert artifact["scale_up_readiness"]["loader_repaired"] is True
    assert artifact["scale_up_readiness"]["preflight_go"] is True
    assert artifact["arc_progress"]["total_levels_solved"] == 20
    assert artifact["arc_progress"]["game_advanced"] == "wa30-ee6fef47"
    assert artifact["registry_read"]["regression_guard_passed"] is True
    assert artifact["hardware_read"]["polarfire_step_taken"] == (
        "polarfire_hash_verified_cpu_dispatch_succeeded"
    )

    provenance = {row["artifact_key"]: row for row in artifact["upstream_provenance"]}
    assert set(provenance) == set(mod.DEFAULT_UPSTREAMS)
    for key, upstream in mod.DEFAULT_UPSTREAMS.items():
        expected_sha = hashlib.sha256((Path.cwd() / upstream.path).read_bytes()).hexdigest()
        assert provenance[key]["sha256"] == expected_sha
    assert provenance["4272_cross_family_fresh"]["fields_imported"] == []
    assert "cross_family_delta" in provenance["4271_cross_family_existing"]["fields_imported"]
    assert artifact["reproducibility_checksum"] == mod.checksum_from_provenance(
        artifact["upstream_provenance"]
    )


def test_req_capstone_4279_clean_fixture_flips_full_run_gate(tmp_path: Path) -> None:
    """REQ-CAPSTONE-4279: clean .394 hardening plus cross-family plus preflight opens .396."""

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
    assert artifact["headline_outcome"] == (
        "cross_family_generalizes_self_learning_helps_"
        "diffusiongemma_full_run_ready_arc22_game_fixture-game_paper_not_ready"
    )
    assert artifact["cross_family_generalizes"] is True
    assert artifact["hardened_win"] is True
    assert artifact["diffusiongemma_full_run_gate"] is True
    assert artifact["paper_ready"] is False
    assert artifact["unmet_gates"] == ["G2"]
    assert artifact["flagged_artifacts_excluded"] == []
    assert artifact["reproducibility_checksum"] == mod.checksum_from_provenance(
        artifact["upstream_provenance"]
    )


def test_req_capstone_4279_excludes_flagged_and_circular_inputs(tmp_path: Path) -> None:
    """REQ-CAPSTONE-4279: flagged/live-critical/circular inputs cannot headline."""

    payloads = _minimal_payloads()
    payloads["4256_leak_audit"]["verifier_is_oracle"] = True
    payloads["4271_cross_family_existing"]["flagged_adversarial"] = True
    _write_default_artifacts(tmp_path, payloads)

    def live_flags(path: Path) -> list[dict[str, str]]:
        if path.name == mod.DEFAULT_UPSTREAMS["4274_preflight"].path.name:
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
    assert artifact["cross_family_generalizes"] is False
    assert artifact["hardened_win"] is False
    assert artifact["diffusiongemma_full_run_gate"] is False
    assert artifact["verifier_is_oracle_honored"] is True
    assert artifact["hardening"]["provenance_blind"]["used_as_oracle_distinct"] is False
    assert artifact["cross_family"]["status"] == "open_no_clean_cross_family_run"
    assert artifact["scale_up_readiness"]["status"] == "excluded_flagged_adversarial"
    excluded = {row["artifact_key"] for row in artifact["flagged_artifacts_excluded"]}
    assert excluded == {"4271_cross_family_existing", "4274_preflight"}
    provenance = {row["artifact_key"]: row for row in artifact["upstream_provenance"]}
    assert provenance["4271_cross_family_existing"]["fields_imported"] == []
    assert provenance["4274_preflight"]["fields_imported"] == []


def test_req_capstone_4279_records_ood_open_when_no_cross_family_run(tmp_path: Path) -> None:
    """REQ-CAPSTONE-4279: if neither A2 nor A3 ran cleanly, OOD remains open."""

    payloads = _minimal_payloads()
    payloads["4270_family_recovery"]["family_split_feasible"] = False
    payloads["4271_cross_family_existing"] = {
        "honest_verdict": "blocked_gate_check_failed",
        "status": "blocked",
    }
    payloads["4272_cross_family_fresh"] = {
        "honest_verdict": "blocked_gate_check_failed",
        "status": "blocked",
    }
    _write_default_artifacts(tmp_path, payloads)

    artifact = mod.build_artifact(
        tmp_path,
        started_s=4.0,
        now_s=4.25,
        live_flag_runner=_clean_live_flags,
        summarize_runner=_summarize_zero,
        publication_gate_runner=lambda _: _paper_gate(True),
    )

    mod.validate_artifact(artifact)
    assert artifact["cross_family_generalizes"] is False
    assert artifact["hardened_win"] is False
    assert artifact["diffusiongemma_full_run_gate"] is False
    assert artifact["cross_family"]["status"] == "open_no_clean_cross_family_run"
    assert artifact["headline_outcome"].startswith("cross_family_open_")


def test_req_capstone_4279_missing_or_malformed_core_blocks_without_aggregation(
    tmp_path: Path,
) -> None:
    """REQ-CAPSTONE-4279: missing or unparsable core artifacts produce blocked verdicts."""

    payloads = _minimal_payloads()
    payloads.pop("4274_preflight")
    _write_default_artifacts(tmp_path, payloads)
    malformed_path = tmp_path / mod.DEFAULT_UPSTREAMS["4277_registry"].path
    malformed_path.write_text("[]\n", encoding="utf-8")

    artifact = mod.build_artifact(
        tmp_path,
        started_s=5.0,
        now_s=5.25,
        live_flag_runner=_clean_live_flags,
        summarize_runner=_summarize_zero,
        publication_gate_runner=lambda _: _paper_gate(True),
    )

    mod.validate_artifact(artifact)
    assert artifact["honest_verdict"] == "blocked_v395_artifacts_missing"
    assert artifact["headline_outcome"] == "blocked_v395_artifacts_missing"
    assert artifact["cross_family_generalizes"] is False
    assert artifact["hardened_win"] is False
    assert artifact["diffusiongemma_full_run_gate"] is False
    assert artifact["paper_ready"] is None
    assert artifact["missing_upstream_artifacts"] == [
        {"artifact_key": "4274_preflight", "experiment_id": 4274, "reason": "missing"},
        {
            "artifact_key": "4277_registry",
            "experiment_id": 4277,
            "reason": "unparsable_or_non_object",
        },
    ]
    assert artifact["upstream_provenance"] == []
    assert artifact["reproducibility_checksum"] == mod.BLOCKED_CHECKSUM


def test_req_capstone_4279_validation_write_and_helpers(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """REQ-CAPSTONE-4279: validation protects schema, checksums, and writes."""

    payloads = _minimal_payloads()
    _write_default_artifacts(tmp_path, payloads)
    artifact = mod.build_artifact(
        tmp_path,
        started_s=6.0,
        now_s=6.5,
        live_flag_runner=_clean_live_flags,
        summarize_runner=_summarize_zero,
        publication_gate_runner=lambda _: _paper_gate(True),
    )
    out_path = mod.write_artifact(
        tmp_path,
        output_path=Path("results/experiment_4279_capstone_v395.json"),
        started_s=7.0,
        now_s=7.25,
        live_flag_runner=_clean_live_flags,
        summarize_runner=_summarize_zero,
        publication_gate_runner=lambda _: _paper_gate(True),
    )
    written = json.loads(out_path.read_text(encoding="utf-8"))
    mod.validate_artifact(written)

    assert mod.bool_metric({"x": True}, "x") is True
    assert mod.bool_metric({"x": 1}, "x") is None
    assert mod.int_metric({"x": 2}, "x") == 2
    assert mod.int_metric({"x": True}, "x") == 0
    assert mod.float_metric({"x": 2}, "x") == pytest.approx(2.0)
    assert mod.float_metric({"x": True}, "x") is None
    assert mod.str_metric({"x": "ok"}, "x") == "ok"
    assert mod.str_metric({"x": 1}, "x") == ""
    assert mod.list_metric({"x": [1]}, "x") == [1]
    assert mod.list_metric({"x": "bad"}, "x") == []
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
    assert mod._ci_excludes_zero({"x": [-2.0, -1.0]}, "x") is True
    assert mod.cross_family_read(None, "missing")["status"] == "open_no_clean_cross_family_run"
    assert mod.self_learning_read(
        {"online_adaptation_helps": True, "verifier_is_oracle": False}
    )["status"] == "helps"
    assert mod.self_learning_read({})["status"] == "static_ceiling"
    assert mod.preflight_read({}, skipped=False)["status"] == "preflight_blocked"
    assert mod.preflight_read({}, skipped=True)["status"] == "excluded_flagged_adversarial"
    assert mod.registry_read({}, skipped=True)["status"] == "excluded_flagged_adversarial"
    assert mod.is_sha256("a" * 64) is True
    assert mod.is_sha256("z" * 64) is False
    assert mod.is_sha256(1) is False

    monkeypatch.setattr(
        mod.subprocess,
        "run",
        lambda *args, **kwargs: type("Proc", (), {"stdout": "[]\n", "returncode": 0})(),
    )
    with pytest.raises(ValueError, match="publication_gate.py --json"):
        mod.run_publication_gate(tmp_path)

    _expect_validation_error(artifact, lambda a: a.pop("honest_verdict"), "missing required")
    _expect_validation_error(artifact, lambda a: a.update({"honest_verdict": "bad"}), "terminal")
    _expect_validation_error(artifact, lambda a: a.update({"headline_outcome": ""}), "headline")
    _expect_validation_error(
        artifact, lambda a: a.update({"cross_family_generalizes": "bad"}), "cross_family"
    )
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
        lambda a: a.update({"hardened_win": True, "cross_family_generalizes": False}),
        "hardened_win",
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
    blocked_bad["honest_verdict"] = "blocked_v395_artifacts_missing"
    blocked_bad["headline_outcome"] = "blocked_v395_artifacts_missing"
    blocked_bad["paper_ready"] = True
    blocked_bad["upstream_provenance"] = []
    blocked_bad["reproducibility_checksum"] = mod.BLOCKED_CHECKSUM
    with pytest.raises(ValueError, match="blocked artifacts"):
        mod.validate_artifact(blocked_bad)
