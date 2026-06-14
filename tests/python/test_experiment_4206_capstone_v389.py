"""Tests for Exp 4206 .389 verifier-as-reward capstone aggregation.

Spec refs: REQ-CAPSTONE-4206, SCENARIO-CAPSTONE-4206.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from carnot.reporting import capstone_v389_4206 as mod


JsonDict = dict[str, Any]


def _write_json(path: Path, payload: JsonDict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _minimal_payloads() -> dict[str, JsonDict]:
    return {
        "4197_smoke": {
            "experiment": "experiment_4197_verifier_reward_code_lora_rft_3arm",
            "honest_verdict": "complete: smoke fixture",
            "smoke": True,
            "harness_ready": True,
            "truncation_guard": {"max_allowed_truncation_rate": 0.05},
        },
        "4197_phase0": {
            "experiment": "experiment_4197_verifier_reward_phase0_headroom_harness_build",
            "honest_verdict": "complete: phase0 fixture",
            "harness_ready": True,
            "phase0_precision": 0.956,
            "youden_j": 0.414,
            "training_headroom_present": True,
            "operating_point": {"base": "non-qwen fixture", "truncation_rate": 0.0},
            "phase0_detail": {"phase0_clears": True},
        },
        "4198": {
            "experiment": "experiment_4198_verifier_reward_3arm_rft_launch",
            "honest_verdict": "complete: launch fixture",
            "training_launched": True,
            "gold_control_early_read": {
                "available": True,
                "arm_c_minus_base": 0.08,
                "status": "gold_control_confirmed",
            },
            "truncation_guard": {"status": "confirmed", "a1_truncation_rate": 0.0},
        },
        "4199": {
            "experiment": 4199,
            "honest_verdict": "complete: a_vs_b fixture",
            "status": "complete",
            "a_vs_b_delta": 0.07,
            "a_vs_b_ci95": [0.02, 0.12],
            "verifier_label_carries_signal": True,
            "positive_control_confirmed": True,
            "truncation_guard_confirmed": True,
        },
        "4200": {
            "experiment": "experiment_4200_certified_arc_corpus_distill_lift",
            "honest_verdict": "complete: distill fixture",
            "certification_precision": {"rate": 0.94},
            "certified_corpus_size": 16,
            "distill_lift_ci95": [0.02, 0.09],
            "invisible_leash_diagnosis": "latent",
        },
        "4201": {
            "experiment": "experiment_4201_arc_incremental_progress",
            "honest_verdict": "complete: arc progress fixture",
            "acceptance_gate_passed": True,
            "prior_total_levels_solved": 15,
            "total_levels_solved": 16,
            "total_games_solved": 13,
            "new_levels_solved_this_task": 1,
        },
        "4202": {
            "experiment": "experiment_4202_arc_live_env_solver_vs_floor",
            "honest_verdict": "complete: live fixture",
            "acceptance_gate_passed": True,
            "live_env_reachable": True,
            "solver_beats_floor": {
                "accuracy": {"beats": True},
                "efficiency": {"beats": True},
                "overall": True,
            },
        },
        "4203": {
            "honest_verdict": "complete: sota fixture",
            "flagged_for_v390": "non_qwen_same_generator_random_label_ablation_v390",
            "methods_mapped": [{"name": "fixture"}],
        },
        "4204": {
            "experiment": "experiment_4204_verifier_registry_gaps_hygiene",
            "honest_verdict": "complete: registry fixture",
            "regression_guard_passed": True,
        },
        "4205": {
            "experiment": 4205,
            "honest_verdict": "complete: hardware fixture",
            "kv260_reachable": True,
            "gatemate_reachable": False,
            "polarfire_reachable": True,
            "fabric_acceleration_claimed": False,
            "speedup_claim_made": False,
        },
    }


def _write_default_artifacts(root: Path, payloads: dict[str, JsonDict]) -> None:
    for upstream_key, payload in payloads.items():
        _write_json(root / mod.DEFAULT_UPSTREAM_PATHS[upstream_key], payload)


def _expect_validation_error(artifact: JsonDict, mutator: Any, match: str) -> None:
    mutated = json.loads(json.dumps(artifact))
    mutator(mutated)
    with pytest.raises(ValueError, match=match):
        mod.validate_artifact(mutated)


def test_req_capstone_4206_spec_anchor_declares_required_contract() -> None:
    """REQ-CAPSTONE-4206: OpenSpec declares the .389 capstone contract."""

    spec = Path("openspec/capabilities/capstone/spec.md").read_text(encoding="utf-8")

    assert "REQ-CAPSTONE-4206" in spec
    assert "SCENARIO-CAPSTONE-4206" in spec
    for outcome in mod.HEADLINE_OUTCOMES:
        assert outcome in spec
    for status in mod.VERIFIER_AS_REWARD_STATUSES:
        assert status in spec
    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert field in spec
        assert mod.FIELD_PRINCIPLES[field] in spec
    assert "flagged_adversarial:true" in spec
    assert "positive A-vs-B CI excluding zero" in spec


def test_scenario_capstone_4206_current_artifacts_decide_no_operating_point() -> None:
    """SCENARIO-CAPSTONE-4206: current .389 artifacts skip flagged upstreams first."""

    artifact = mod.build_artifact(Path.cwd(), started_s=10.0, now_s=10.5)

    mod.validate_artifact(artifact)
    assert artifact["headline_outcome"] == "verifier_reward_no_code_operating_point"
    assert artifact["verifier_as_reward_status"] == "NO-OPERATING-POINT"
    assert artifact["honest_verdict"].startswith(
        "complete: capstone_v389_verifier_reward_no_code_operating_point_"
    )
    assert artifact["phase0_operating_point"]["status"] == "skipped_flagged_adversarial"
    assert artifact["phase0_operating_point"]["phase0_precision"] is None
    assert artifact["phase0_operating_point"]["youden_j"] is None
    assert artifact["a_vs_b_training_signal"]["status"] == "blocked_a_vs_b_not_collected"
    assert artifact["a_vs_b_training_signal"]["verifier_label_carries_signal"] is False
    assert artifact["a_vs_b_training_signal"]["positive_control_confirmed"] is False
    assert artifact["arc_distill_latent_or_absent"]["status"] == "skipped_flagged_adversarial"
    assert artifact["arc_distill_latent_or_absent"]["latent_or_absent"] == (
        "UNINFORMATIVE-skipped_flagged_adversarial"
    )
    assert artifact["total_arc_levels_solved"] == 15
    assert artifact["live_solver_vs_floor"]["solver_beats_floor_overall"] is True
    assert artifact["live_solver_vs_floor"]["solver_beats_floor_accuracy"] is False
    assert artifact["live_solver_vs_floor"]["solver_beats_floor_efficiency"] is True
    assert artifact["strongest_sota_flagged_for_v390"] == (
        "non_qwen_same_generator_random_label_ablation_v390"
    )
    assert artifact["field_principles"] == mod.FIELD_PRINCIPLES

    skipped = {
        (row["upstream_key"], row["experiment_id"]) for row in artifact["flagged_artifacts_skipped"]
    }
    assert skipped == {
        ("4197_phase0", 4197),
        ("4198", 4198),
        ("4200", 4200),
        ("4204", 4204),
    }

    provenance = {row["upstream_key"]: row for row in artifact["upstream_provenance"]}
    assert set(provenance) == set(mod.DEFAULT_UPSTREAM_PATHS)
    for upstream_key, rel_path in mod.DEFAULT_UPSTREAM_PATHS.items():
        expected_sha = hashlib.sha256((Path.cwd() / rel_path).read_bytes()).hexdigest()
        assert provenance[upstream_key]["sha256"] == expected_sha
    for upstream_key in ("4197_phase0", "4198", "4200", "4204"):
        assert provenance[upstream_key]["skipped"] is True
        assert provenance[upstream_key]["fields_imported"] == []
    assert "harness_ready" in provenance["4197_smoke"]["fields_imported"]
    assert "status" in provenance["4199"]["fields_imported"]
    assert "total_levels_solved" in provenance["4201"]["fields_imported"]
    assert "solver_beats_floor" in provenance["4202"]["fields_imported"]
    assert "flagged_for_v390" in provenance["4203"]["fields_imported"]


def test_req_capstone_4206_headline_branches(tmp_path: Path) -> None:
    """REQ-CAPSTONE-4206: clean inputs distinguish real, null, invalid, and no-op."""

    payloads = _minimal_payloads()
    _write_default_artifacts(tmp_path, payloads)
    positive = mod.build_artifact(tmp_path, started_s=1.0, now_s=1.25)
    mod.validate_artifact(positive)
    assert positive["headline_outcome"] == "verifier_reward_real_label_carries_training_signal"
    assert positive["verifier_as_reward_status"] == "REAL"
    assert positive["phase0_operating_point"]["phase0_precision"] == 0.956
    assert positive["phase0_operating_point"]["youden_j"] == 0.414
    assert positive["arc_distill_latent_or_absent"]["latent_or_absent"] == "LATENT"

    payloads = _minimal_payloads()
    payloads["4199"]["a_vs_b_delta"] = 0.01
    payloads["4199"]["a_vs_b_ci95"] = [-0.03, 0.05]
    payloads["4199"]["verifier_label_carries_signal"] = False
    _write_default_artifacts(tmp_path, payloads)
    null = mod.build_artifact(tmp_path, started_s=2.0, now_s=2.25)
    assert null["headline_outcome"] == "verifier_reward_null_equals_distillation_or_spurious"
    assert null["verifier_as_reward_status"] == "NULL"

    payloads = _minimal_payloads()
    payloads["4199"]["positive_control_confirmed"] = False
    _write_default_artifacts(tmp_path, payloads)
    invalid = mod.build_artifact(tmp_path, started_s=3.0, now_s=3.25)
    assert invalid["headline_outcome"] == "verifier_reward_invalid_or_underpowered"
    assert invalid["verifier_as_reward_status"] == "INVALID-or-UNDERPOWERED"

    payloads = _minimal_payloads()
    payloads["4197_phase0"]["flagged_adversarial"] = True
    _write_default_artifacts(tmp_path, payloads)
    no_op = mod.build_artifact(tmp_path, started_s=4.0, now_s=4.25)
    assert no_op["headline_outcome"] == "verifier_reward_no_code_operating_point"
    assert no_op["verifier_as_reward_status"] == "NO-OPERATING-POINT"
    assert no_op["upstream_provenance"][1]["upstream_key"] == "4197_phase0"
    assert no_op["upstream_provenance"][1]["fields_imported"] == []

    payloads = _minimal_payloads()
    payloads["4200"]["distill_lift_ci95"] = [-0.04, 0.0]
    payloads["4200"]["invisible_leash_diagnosis"] = "absent"
    _write_default_artifacts(tmp_path, payloads)
    absent = mod.build_artifact(tmp_path, started_s=5.0, now_s=5.25)
    assert absent["arc_distill_latent_or_absent"]["latent_or_absent"] == "ABSENT"


def test_req_capstone_4206_validation_write_and_missing_edges(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    """REQ-CAPSTONE-4206: validation protects schema, checksums, and writes."""

    payloads = _minimal_payloads()
    _write_default_artifacts(tmp_path, payloads)
    artifact = mod.build_artifact(tmp_path, started_s=6.0, now_s=6.25)

    malformed = tmp_path / "malformed.json"
    malformed.write_text("[]\n", encoding="utf-8")
    with pytest.raises(ValueError, match="did not contain a JSON object"):
        mod.read_json_object(malformed)

    output = mod.write_artifact(
        tmp_path,
        output_path=Path("results/experiment_4206_capstone_v389.json"),
        started_s=7.0,
        now_s=7.25,
    )
    written = json.loads(output.read_text(encoding="utf-8"))
    mod.validate_artifact(written)
    assert written["reproducibility_checksum"] == mod.payload_checksum(written)

    missing_root = tmp_path / "missing"
    payloads_without_4203 = _minimal_payloads()
    payloads_without_4203.pop("4203")
    _write_default_artifacts(missing_root, payloads_without_4203)
    missing = mod.build_artifact(missing_root, started_s=8.0, now_s=8.25)
    assert missing["missing_upstream_artifacts"] == [
        {"experiment_id": 4203, "upstream_key": "4203"}
    ]
    assert missing["sota_v390"]["status"] == "missing"

    assert mod.ci95({"x": ["bad", 1.0]}, "x") is None
    assert mod.ci95({"x": [1.0, "bad"]}, "x") is None
    assert mod.phase0_operating_point(None, was_skipped=False)["status"] == "missing"
    assert mod.a_vs_b_training_signal({}, was_skipped=True)["status"] == (
        "skipped_flagged_adversarial"
    )
    assert mod.a_vs_b_training_signal(None, was_skipped=False)["status"] == "missing"
    assert mod.arc_distill_latent_or_absent(None, was_skipped=False)["status"] == "missing"
    assert (
        mod.arc_distill_latent_or_absent(
            {"distill_lift_ci95": [-0.01, 0.02], "invisible_leash_diagnosis": "flat"},
            was_skipped=False,
        )["latent_or_absent"]
        == "UNINFORMATIVE"
    )
    assert mod.arc_progress({}, was_skipped=True)["status"] == "skipped_flagged_adversarial"
    assert mod.arc_progress(None, was_skipped=False)["status"] == "missing"
    assert mod.live_solver_vs_floor({}, was_skipped=True)["status"] == (
        "skipped_flagged_adversarial"
    )
    assert mod.live_solver_vs_floor(None, was_skipped=False)["status"] == "missing"
    assert mod.sota_v390({}, was_skipped=True)["status"] == "skipped_flagged_adversarial"
    assert mod.registry_hygiene(None, was_skipped=False)["status"] == "missing"
    assert mod.hardware_continuity({}, was_skipped=True)["status"] == (
        "skipped_flagged_adversarial"
    )
    assert mod.hardware_continuity(None, was_skipped=False)["status"] == "missing"
    assert mod.smoke_harness({}, was_skipped=True)["status"] == "skipped_flagged_adversarial"
    assert mod.smoke_harness(None, was_skipped=False)["status"] == "missing"
    assert (
        mod.headline_outcome(
            {"status": "included", "phase0_clears": True}, {"controls_confirmed": True}
        )
        == "verifier_reward_invalid_or_underpowered"
    )

    _expect_validation_error(artifact, lambda a: a.pop("honest_verdict"), "missing required")
    _expect_validation_error(artifact, lambda a: a.update({"honest_verdict": "bad"}), "terminal")
    _expect_validation_error(artifact, lambda a: a.update({"headline_outcome": "bad"}), "headline")
    _expect_validation_error(
        artifact, lambda a: a.update({"verifier_as_reward_status": "bad"}), "verifier"
    )
    _expect_validation_error(artifact, lambda a: a.update({"phase0_operating_point": []}), "phase0")
    _expect_validation_error(
        artifact, lambda a: a.update({"arc_distill_latent_or_absent": []}), "arc_distill"
    )
    _expect_validation_error(
        artifact, lambda a: a.update({"total_arc_levels_solved": 14}), "ARC levels"
    )
    _expect_validation_error(
        artifact, lambda a: a.update({"inference_substrate": "bad"}), "inference_substrate"
    )
    _expect_validation_error(
        artifact, lambda a: a.update({"field_principles": []}), "field_principles"
    )
    _expect_validation_error(
        artifact,
        lambda a: a["field_principles"].update({"honest_verdict": "wrong"}),
        "principle",
    )
    _expect_validation_error(
        artifact, lambda a: a.update({"upstream_provenance": {}}), "upstream_provenance"
    )
    _expect_validation_error(
        artifact, lambda a: a["upstream_provenance"].append(42), "entries must be objects"
    )
    _expect_validation_error(
        artifact,
        lambda a: a["upstream_provenance"][0].update({"experiment_id": "4197"}),
        "integer experiment_id",
    )
    _expect_validation_error(
        artifact,
        lambda a: a["upstream_provenance"][0].update({"upstream_key": 4197}),
        "upstream_key",
    )
    _expect_validation_error(
        artifact,
        lambda a: a["upstream_provenance"][0].update({"fields_imported": "bad"}),
        "fields_imported",
    )
    _expect_validation_error(
        artifact,
        lambda a: a["upstream_provenance"][1].update(
            {"skipped": True, "fields_imported": ["phase0_precision"]}
        ),
        "skipped upstreams",
    )
    _expect_validation_error(
        artifact, lambda a: a["upstream_provenance"][0].update({"sha256": "bad"}), "sha256"
    )
    _expect_validation_error(
        artifact, lambda a: a.update({"flagged_artifacts_skipped": {}}), "flagged"
    )
    _expect_validation_error(
        artifact, lambda a: a.update({"reproducibility_checksum": "bad"}), "checksum"
    )
    _expect_validation_error(artifact, lambda a: a.update({"duration_s": 99.0}), "checksum")

    output_path = tmp_path / "results" / "experiment_4206_capstone_v389.json"
    monkeypatch.setattr(mod, "write_artifact", lambda root: output_path)
    assert mod.main() == 0
    assert str(output_path) in capsys.readouterr().out
