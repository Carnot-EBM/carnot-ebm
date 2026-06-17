"""Tests for Exp 4335 .400 verifier scorecard capstone.

Spec refs: REQ-CAPSTONE-4335, SCENARIO-CAPSTONE-4335.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from carnot.reporting import capstone_v400_4335 as mod


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
        "4325_in_generation_replication": {
            "honest_verdict": "complete: replication fixture",
            "in_generation_moat_replicates": True,
            "carnot_minus_best_control_delta": 0.25,
            "carnot_minus_self_reward_smc_delta": 0.2,
            "replication_ci95": [0.04, 0.42],
            "controls_differentiated": True,
            "scorer_leak_recheck_passed": True,
            "verifier_is_oracle": False,
            "reproducibility_checksum": "sha256:" + "a" * 64,
        },
        "4326_adaptive": {
            "honest_verdict": "complete: adaptive fixture",
            "adaptive_guidance_beats_control": True,
            "carnot_minus_best_control_delta": 0.18,
            "adaptive_ci95": [0.03, 0.31],
            "controls_differentiated": True,
            "scorer_leak_recheck_passed": True,
            "domain_used": "arc_grid_generation",
            "verifier_is_oracle": False,
            "reproducibility_checksum": "sha256:" + "b" * 64,
        },
        "4327_e3_ar25": {
            "honest_verdict": "complete: ar25 fixture",
            "game": "ar25",
            "offline_reproduced": True,
            "reproduced_levels": 1,
            "verifier_accuracy_per_round": [0.91],
            "verifier_best_accuracy": 0.91,
            "verifier_is_oracle": True,
            "reproducibility_checksum": "sha256:" + "c" * 64,
        },
        "4328_e3_ka59": {
            "honest_verdict": "complete: ka59 fixture",
            "game": "ka59",
            "offline_reproduced": False,
            "reproduced_levels": 0,
            "verifier_accuracy_per_round": [0.6],
            "verifier_best_accuracy": 0.6,
            "verifier_is_oracle": True,
            "reproducibility_checksum": "sha256:" + "d" * 64,
        },
        "4329_e3_tr87_ft09": {
            "honest_verdict": "complete: tr87 ft09 fixture",
            "games": ["tr87", "ft09"],
            "reproduced_levels_total": 1,
            "per_game_scorecard": {
                "tr87": {
                    "game": "tr87",
                    "offline_reproduced": True,
                    "reproduced_levels": 1,
                    "verifier_accuracy_per_round": [0.7],
                    "verifier_best_accuracy": 0.7,
                    "verifier_is_oracle": True,
                },
                "ft09": {
                    "game": "ft09",
                    "offline_reproduced": False,
                    "reproduced_levels": 0,
                    "verifier_accuracy_per_round": [0.2],
                    "verifier_best_accuracy": 0.2,
                    "verifier_is_oracle": True,
                },
            },
            "verifier_is_oracle": True,
            "reproducibility_checksum": "sha256:" + "e" * 64,
        },
        "4330_shallow": {
            "honest_verdict": "complete: shallow fixture",
            "offline_reproduced": True,
            "reproducible_total_levels": 16,
            "games_advanced": ["ar25", "tr87"],
            "prior_reproducible_total_levels": 13,
            "verifier_is_oracle": True,
            "reproducibility_checksum": "sha256:" + "f" * 64,
        },
        "4331_self_learning": {
            "honest_verdict": "complete: self-learning fixture",
            "acceptance_gate_passed": True,
            "learned_encoder_transfer_helps": True,
            "cross_game_state_reduction": 1.2,
            "cross_game_state_reduction_ci95": [1.05, 1.4],
            "n_held_out_levels": 13,
            "verifier_is_oracle": False,
            "reproducibility_checksum": "sha256:" + "1" * 64,
        },
        "4333_hygiene": {
            "honest_verdict": "complete: hygiene fixture",
            "regression_guard_passed": True,
            "registry_reconciled": True,
            "manifest_reconciled": True,
            "gaps_logged": [],
            "reproducibility_checksum": "sha256:" + "2" * 64,
        },
    }


def _write_default_artifacts(root: Path, payloads: dict[str, JsonDict]) -> None:
    for key, payload in payloads.items():
        _write_json(root / mod.DEFAULT_UPSTREAMS[key].path, payload)


def test_req_capstone_4335_spec_anchor_declares_required_contract() -> None:
    """REQ-CAPSTONE-4335: OpenSpec declares the .400 scorecard contract."""

    spec = Path("openspec/capabilities/capstone/spec.md").read_text(encoding="utf-8")

    assert "REQ-CAPSTONE-4335" in spec
    assert "SCENARIO-CAPSTONE-4335" in spec
    assert "experiment_4335_capstone_v400.json" in spec
    assert "blocked_no_v400_artifacts" in spec
    assert "aggregate-available-report-gaps" in spec
    assert "publication_gate.py --json" in spec
    assert "verifier_is_oracle:false" in spec
    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert field in spec
        assert mod.FIELD_PRINCIPLES[field] in spec


def test_scenario_capstone_4335_current_artifacts_report_v400_scorecard() -> None:
    """SCENARIO-CAPSTONE-4335: current .400 artifacts produce the honest scorecard."""

    artifact = mod.build_artifact(
        Path.cwd(),
        started_s=1.0,
        now_s=1.5,
        live_flag_runner=_clean_live_flags,
        summarize_runner=_summarize_zero,
        publication_gate_runner=lambda _: _paper_gate(True),
    )

    mod.validate_artifact(artifact)
    assert artifact["honest_verdict"] == (
        "complete: v400_in_generation_corpus_specific_gate_"
        "STILL_PENDING_second_corpus_scorer_leaky_arc_levels_13_"
        "e3_reproduced_0_self_learning_open_hygiene_passed"
    )
    assert artifact["headline_outcome"] == (
        "in_generation_corpus_specific__adaptive_open_reasoning_corpus_fallback__"
        "arc_levels_13_e3_0__self_learning_open__paper_ready"
    )
    assert artifact["in_generation_moat_replicates_headline"] is False
    assert artifact["diffusiongemma_gate_status"] == "STILL_PENDING_second_corpus_scorer_leaky"
    assert artifact["arc_reproducible_total_levels"] == 13
    assert artifact["verifier_thesis_state"] == "in_generation_moat_corpus_specific"
    assert artifact["paper_ready"] is True
    assert artifact["per_axis_gaps"] == []
    assert artifact["flagged_artifacts_excluded"] == []
    assert artifact["in_generation_replication"]["controls_differentiated"] is False
    assert artifact["adaptive_scaleup"]["domain_used"] == "reasoning_corpus_fallback"
    assert artifact["e3_deep_tail"]["reproduced_levels_total"] == 0
    assert artifact["e3_deep_tail"]["games"]["tr87"]["verifier_best_accuracy"] == 0.0
    assert artifact["arc_shallow"]["games_advanced"] == []
    assert artifact["self_learning"]["learned_encoder_transfer_helps"] is False
    assert artifact["hygiene"]["regression_guard_passed"] is True
    assert artifact["verifier_is_oracle_honored"] is True

    provenance = {row["artifact_key"]: row for row in artifact["upstream_provenance"]}
    expected_sha = hashlib.sha256(
        Path("results/experiment_4325_in_generation_moat_replicate_second_corpus.json").read_bytes()
    ).hexdigest()
    assert provenance["4325_in_generation_replication"]["sha256"] == expected_sha
    assert provenance["4327_e3_ar25"]["fields_imported"] == list(
        mod.IMPORTED_FIELDS["4327_e3_ar25"]
    )


def test_req_capstone_4335_missing_axis_does_not_zero_available_axes(tmp_path: Path) -> None:
    """REQ-CAPSTONE-4335: missing artifacts are per-axis gaps, not global blockers."""

    payloads = _minimal_payloads()
    payloads.pop("4326_adaptive")
    payloads.pop("4327_e3_ar25")
    payloads.pop("4330_shallow")
    payloads.pop("4333_hygiene")
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
    assert artifact["in_generation_moat_replicates_headline"] is True
    assert artifact["diffusiongemma_gate_status"] == "MET_oracle_distinct_replicated"
    assert artifact["arc_reproducible_total_levels"] == 0
    assert artifact["verifier_thesis_state"] == "in_generation_moat_replicated"
    assert artifact["paper_ready"] is False
    assert artifact["unmet_gates"] == ["G2"]
    assert artifact["per_axis_gaps"] == [
        {"axis": "in_generation", "artifact_key": "4326_adaptive", "experiment_id": 4326},
        {"axis": "arc_deep_tail", "artifact_key": "4327_e3_ar25", "experiment_id": 4327},
        {"axis": "arc_shallow", "artifact_key": "4330_shallow", "experiment_id": 4330},
        {"axis": "hygiene", "artifact_key": "4333_hygiene", "experiment_id": 4333},
    ]
    assert artifact["availability_report"]["available_artifact_keys"] == [
        "4325_in_generation_replication",
        "4328_e3_ka59",
        "4329_e3_tr87_ft09",
        "4331_self_learning",
    ]


def test_req_capstone_4335_flagged_live_critical_and_oracle_are_bounded(
    tmp_path: Path,
) -> None:
    """REQ-CAPSTONE-4335: flagged, live-critical, and circular inputs are bounded."""

    payloads = _minimal_payloads()
    payloads["4325_in_generation_replication"]["verifier_is_oracle"] = True
    payloads["4326_adaptive"]["flagged_adversarial"] = True
    _write_default_artifacts(tmp_path, payloads)

    def live_flags(path: Path) -> list[dict[str, str]]:
        if path.name == mod.DEFAULT_UPSTREAMS["4331_self_learning"].path.name:
            return [{"kind": "DEGENERATE_CONTROLS", "severity": "critical", "detail": "fixture"}]
        return []

    artifact = mod.build_artifact(
        tmp_path,
        started_s=3.0,
        now_s=3.5,
        live_flag_runner=live_flags,
        summarize_runner=_summarize_zero,
        publication_gate_runner=lambda _: _paper_gate(True),
    )

    mod.validate_artifact(artifact)
    assert artifact["in_generation_moat_replicates_headline"] is False
    assert artifact["diffusiongemma_gate_status"] == "STILL_PENDING_verifier_oracle_not_distinct"
    assert artifact["verifier_is_oracle_honored"] is False
    assert artifact["oracle_distinct_violations"] == [
        "4325_in_generation_replication:in_generation_replication"
    ]
    excluded = {row["artifact_key"] for row in artifact["flagged_artifacts_excluded"]}
    assert excluded == {"4326_adaptive", "4331_self_learning"}
    provenance = {row["artifact_key"]: row for row in artifact["upstream_provenance"]}
    assert provenance["4326_adaptive"]["fields_imported"] == []
    assert provenance["4331_self_learning"]["fields_imported"] == []


def test_req_capstone_4335_blocks_only_when_no_v400_artifacts(tmp_path: Path) -> None:
    """REQ-CAPSTONE-4335: no landed .400 artifacts is the only global block."""

    artifact = mod.build_artifact(
        tmp_path,
        started_s=4.0,
        now_s=4.25,
        live_flag_runner=_clean_live_flags,
        summarize_runner=_summarize_zero,
        publication_gate_runner=lambda _: _paper_gate(True),
    )

    mod.validate_artifact(artifact)
    assert artifact["honest_verdict"] == "blocked_no_v400_artifacts"
    assert artifact["headline_outcome"] == "blocked_no_v400_artifacts"
    assert artifact["paper_ready"] is True
    assert artifact["in_generation_moat_replicates_headline"] is False
    assert artifact["diffusiongemma_gate_status"] == "STILL_PENDING_no_v400_artifacts"
    assert artifact["arc_reproducible_total_levels"] == 0
    assert artifact["per_axis_gaps"] == [
        {
            "axis": "in_generation",
            "artifact_key": "4325_in_generation_replication",
            "experiment_id": 4325,
        },
        {"axis": "in_generation", "artifact_key": "4326_adaptive", "experiment_id": 4326},
        {"axis": "arc_deep_tail", "artifact_key": "4327_e3_ar25", "experiment_id": 4327},
        {"axis": "arc_deep_tail", "artifact_key": "4328_e3_ka59", "experiment_id": 4328},
        {"axis": "arc_deep_tail", "artifact_key": "4329_e3_tr87_ft09", "experiment_id": 4329},
        {"axis": "arc_shallow", "artifact_key": "4330_shallow", "experiment_id": 4330},
        {"axis": "self_learning", "artifact_key": "4331_self_learning", "experiment_id": 4331},
        {"axis": "hygiene", "artifact_key": "4333_hygiene", "experiment_id": 4333},
    ]
    assert artifact["reproducibility_checksum"] == mod.BLOCKED_CHECKSUM


def test_req_capstone_4335_write_validate_and_helper_edges(tmp_path: Path) -> None:
    """REQ-CAPSTONE-4335: validation, checksum, helpers, and wrapper stay strict."""

    payloads = _minimal_payloads()
    _write_default_artifacts(tmp_path, payloads)
    out_path = mod.write_artifact(
        tmp_path,
        output_path=Path("results/experiment_4335_capstone_v400.json"),
        started_s=5.0,
        now_s=5.5,
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
    assert mod.list_metric({"x": [1]}, "x") == [1]
    assert mod.list_metric({"x": "bad"}, "x") == []
    bad_json = tmp_path / "bad.json"
    bad_json.write_text("[]\n", encoding="utf-8")
    with pytest.raises(ValueError, match="non-object"):
        mod.read_json_object(bad_json)
    assert mod.sha_from_payload_checksum({"reproducibility_checksum": "sha256:" + "a" * 64}) == (
        "a" * 64
    )
    assert mod.sha_from_payload_checksum({"reproducibility_checksum": "b" * 64}) == "b" * 64
    assert mod.sha_from_payload_checksum({}) == ""
    assert mod.live_has_critical([{"severity": "critical"}]) is True
    assert mod.live_has_critical([{"severity": "warn"}]) is False
    assert mod.clean_payload({"x": 1}, True) is None
    assert mod.clean_payload({"x": 1}, False) == {"x": 1}
    assert mod._safe_summarize(  # noqa: SLF001
        Path("x"), tmp_path, lambda _path, _root: (_ for _ in ()).throw(RuntimeError("boom"))
    ) == (None, "RuntimeError: boom")
    assert mod._safe_live_flags(  # noqa: SLF001
        Path("x"), lambda _path: (_ for _ in ()).throw(RuntimeError("verify boom"))
    ) == [{"kind": "VERIFY_ERROR", "severity": "warn", "detail": "verify boom"}]
    assert mod._exclusion_reason(False, True, "") == "live_critical_adversarial"  # noqa: SLF001
    assert mod._exclusion_reason(False, False, "bad") == "unparsable_or_non_object"  # noqa: SLF001
    assert mod._exclusion_reason(False, False, "") == "excluded"  # noqa: SLF001
    assert mod.in_generation_replication_read(None, True)["status"] == (
        "excluded_flagged_adversarial"
    )
    assert mod.in_generation_replication_read(None, False)["status"] == "missing_or_excluded"
    assert mod.adaptive_read(None, True)["status"] == "excluded_flagged_adversarial"
    assert mod.adaptive_read(None, False)["status"] == "missing_or_excluded"
    assert mod.e3_single_read(None, True, "x")["status"] == "excluded_flagged_adversarial"
    assert mod.e3_single_read(None, False, "x")["status"] == "missing_or_excluded"
    assert mod.e3_multi_read(None, True)["status"] == "excluded_flagged_adversarial"
    assert mod.e3_multi_read(None, False)["status"] == "missing_or_excluded"
    assert mod.shallow_read(None, True)["status"] == "excluded_flagged_adversarial"
    assert mod.shallow_read(None, False)["status"] == "missing_or_excluded"
    assert mod.self_learning_read(None, True)["status"] == "excluded_flagged_adversarial"
    assert mod.self_learning_read(None, False)["status"] == "missing_or_excluded"
    assert mod.hygiene_read(None, True)["status"] == "excluded_flagged_adversarial"
    assert mod.hygiene_read(None, False)["status"] == "missing_or_excluded"
    assert mod.checksum_from_provenance([]) == mod.BLOCKED_CHECKSUM

    for field, value, pattern in [
        ("honest_verdict", "not_terminal", "terminal-prefixed"),
        ("headline_outcome", "", "headline_outcome"),
        ("in_generation_moat_replicates_headline", "true", "bare bool"),
        ("diffusiongemma_gate_status", "", "diffusiongemma_gate_status"),
        ("arc_reproducible_total_levels", True, "bare int"),
        ("verifier_thesis_state", "unknown", "verifier_thesis_state"),
        ("flagged_artifacts_excluded", {}, "flagged_artifacts_excluded"),
        ("reproducibility_checksum", "bad", "reproducibility_checksum"),
        ("paper_ready", None, "paper_ready"),
    ]:
        bad = json.loads(json.dumps(written))
        bad[field] = value
        with pytest.raises(ValueError, match=pattern):
            mod.validate_artifact(bad)

    bad = json.loads(json.dumps(written))
    bad.pop("headline_outcome")
    with pytest.raises(ValueError, match="missing required"):
        mod.validate_artifact(bad)

    bad = json.loads(json.dumps(written))
    bad["field_principles"] = {}
    with pytest.raises(ValueError, match="field_principles"):
        mod.validate_artifact(bad)

    bad = json.loads(json.dumps(written))
    bad["per_axis_gaps"] = {}
    with pytest.raises(ValueError, match="per_axis_gaps"):
        mod.validate_artifact(bad)

    bad = json.loads(json.dumps(written))
    bad["upstream_provenance"] = {}
    with pytest.raises(ValueError, match="upstream_provenance"):
        mod.validate_artifact(bad)

    bad = json.loads(json.dumps(written))
    bad["upstream_provenance"][0]["sha256"] = "bad"
    with pytest.raises(ValueError, match="upstream provenance"):
        mod.validate_artifact(bad)

    bad = json.loads(json.dumps(written))
    bad["upstream_provenance"][0]["skipped"] = True
    with pytest.raises(ValueError, match="skipped upstreams"):
        mod.validate_artifact(bad)

    bad = json.loads(json.dumps(written))
    bad["reproducibility_checksum"] = "f" * 64
    with pytest.raises(ValueError, match="does not match"):
        mod.validate_artifact(bad)

    assert mod.ci95_excludes_zero([0.1, 0.2]) is True
    assert mod.ci95_excludes_zero([-0.3, -0.1]) is True
    assert mod.ci95_excludes_zero([-0.1, 0.1]) is False
    assert mod.ci95_excludes_zero([0.0, 0.2]) is False
    assert mod.ci95_excludes_zero(["bad", 0.2]) is False
    assert mod.ci95_excludes_zero([0.2]) is False
    assert mod.e3_multi_read({"per_game_scorecard": {"bad": "row"}}, False)["games"] == {}
    assert (
        mod.e3_multi_read(
            {
                "per_game_scorecard": {
                    "tr87": {
                        "offline_reproduced": False,
                        "reproduced_levels": 0,
                        "best_verifier_accuracy": 0.0,
                    }
                }
            },
            False,
        )["games"]["tr87"]["verifier_best_accuracy"]
        == 0.0
    )
    assert mod.diffusiongemma_gate_status({"status": "missing_or_excluded"}, False) == (
        "STILL_PENDING_second_corpus_replication_unavailable"
    )
    assert mod.diffusiongemma_gate_status(  # no scorer leak failure, but controls fail
        {
            "status": "measured",
            "reported_in_generation_moat_replicates": True,
            "controls_differentiated": False,
            "scorer_leak_recheck_passed": True,
            "replication_ci95_excludes_zero": True,
            "verifier_is_oracle": False,
        },
        False,
    ) == "STILL_PENDING_controls_not_differentiated"
    assert mod.diffusiongemma_gate_status(
        {
            "status": "measured",
            "reported_in_generation_moat_replicates": True,
            "controls_differentiated": True,
            "scorer_leak_recheck_passed": True,
            "replication_ci95_excludes_zero": False,
            "verifier_is_oracle": False,
        },
        False,
    ) == "STILL_PENDING_ci95_includes_zero"
    assert mod.diffusiongemma_gate_status(
        {
            "status": "measured",
            "reported_in_generation_moat_replicates": False,
            "controls_differentiated": True,
            "scorer_leak_recheck_passed": True,
            "replication_ci95_excludes_zero": True,
            "verifier_is_oracle": False,
        },
        False,
    ) == "STILL_PENDING_second_corpus_replication_false"
    assert mod.diffusiongemma_gate_status(
        {
            "status": "measured",
            "reported_in_generation_moat_replicates": True,
            "controls_differentiated": True,
            "scorer_leak_recheck_passed": True,
            "replication_ci95_excludes_zero": True,
            "verifier_is_oracle": False,
            "in_generation_moat_replicates_headline": False,
        },
        False,
    ) == "STILL_PENDING_second_corpus_replication_false"
    assert mod.verifier_thesis_state(True, False, False, False) == (
        "in_generation_moat_replicated"
    )
    assert mod.verifier_thesis_state(False, True, True, False) == "arc_deep_tail_e3_solving"
    assert mod.verifier_thesis_state(False, True, False, False) == (
        "in_generation_moat_corpus_specific"
    )
    assert mod.verifier_thesis_state(False, False, False, False) == (
        "verifier_domain_bound_self_learning_open"
    )
    assert mod._oracle_violations(  # noqa: SLF001
        {"reported_in_generation_moat_replicates": True, "verifier_is_oracle": True},
        {"reported_adaptive_guidance_beats_control": True, "verifier_is_oracle": True},
    ) == [
        "4325_in_generation_replication:in_generation_replication",
        "4326_adaptive:adaptive_guidance",
    ]

    parse_root = tmp_path / "parse_error"
    bad_artifact = parse_root / mod.DEFAULT_UPSTREAMS["4330_shallow"].path
    bad_artifact.parent.mkdir(parents=True, exist_ok=True)
    bad_artifact.write_text("[]\n", encoding="utf-8")
    parse_artifact = mod.build_artifact(
        parse_root,
        started_s=6.0,
        now_s=6.25,
        live_flag_runner=_clean_live_flags,
        summarize_runner=_summarize_zero,
        publication_gate_runner=lambda _: _paper_gate(True),
    )
    mod.validate_artifact(parse_artifact)
    assert parse_artifact["flagged_artifacts_excluded"][0]["reason"] == (
        "unparsable_or_non_object"
    )

    wrapper = Path("results/experiment_4335_capstone_v400.py").read_text(encoding="utf-8")
    assert "capstone_v400_4335" in wrapper
