"""Tests for Exp 4312 .398 verifier scorecard capstone.

Spec refs: REQ-CAPSTONE-4312, SCENARIO-CAPSTONE-4312.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from carnot.reporting import capstone_v398_4312 as mod


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
        "4303_efficiency": {
            "honest_verdict": "complete: efficiency fixture",
            "efficiency_pareto_holds": True,
            "cost_ratio": 0.04,
            "accuracy_energy_verifier": 0.8,
            "accuracy_best_judge": 0.79,
            "accuracy_delta_ci95": [0.0, 0.04],
            "verifier_is_oracle": False,
            "reproducibility_checksum": "sha256:" + "a" * 64,
        },
        "4304_in_generation": {
            "honest_verdict": "complete: in-generation fixture",
            "diffusiongemma_guidance_moat": True,
            "controls_differentiated": True,
            "scorer_leak_recheck_passed": True,
            "carnot_minus_best_control_delta": 0.2,
            "guidance_moat_ci95": [0.03, 0.38],
            "verifier_is_oracle": False,
            "reproducibility_checksum": "sha256:" + "b" * 64,
        },
        "4305_cross_domain": {
            "honest_verdict": "complete: cross-domain fixture",
            "cross_domain_selection_holds": True,
            "cross_domain_delta": 0.31,
            "cross_domain_ci95": [0.04, 0.56],
            "label_ablation_robust": True,
            "held_out_task_n": 40,
            "verifier_is_oracle": False,
            "reproducibility_checksum": "sha256:" + "c" * 64,
        },
        "4306_self_learning": {
            "honest_verdict": "complete: self-learning fixture",
            "online_adaptation_helps": True,
            "best_adaptive_minus_static_delta": 0.17,
            "best_adaptive_minus_static_ci95": [0.03, 0.29],
            "held_out_task_n": 80,
            "verifier_is_oracle": False,
            "reproducibility_checksum": "sha256:" + "d" * 64,
        },
        "4307_arc_progress": {
            "honest_verdict": "complete: arc fixture",
            "total_levels": 23,
            "total_levels_solved": 23,
            "levels_completed": 1,
            "new_levels_solved_this_task": 1,
            "reproducibility_checksum": "sha256:" + "e" * 64,
        },
    }


def _write_default_artifacts(root: Path, payloads: dict[str, JsonDict]) -> None:
    for key, payload in payloads.items():
        _write_json(root / mod.DEFAULT_UPSTREAMS[key].path, payload)


def test_req_capstone_4312_spec_anchor_declares_required_contract() -> None:
    """REQ-CAPSTONE-4312: OpenSpec declares the .398 scorecard contract."""

    spec = Path("openspec/capabilities/capstone/spec.md").read_text(encoding="utf-8")

    assert "REQ-CAPSTONE-4312" in spec
    assert "SCENARIO-CAPSTONE-4312" in spec
    assert "experiment_4312_capstone_v398.json" in spec
    assert "blocked_no_v398_artifacts" in spec
    assert "aggregate-available-report-gaps" in spec
    assert "publication_gate.py --json" in spec
    assert "verifier_is_oracle:false" in spec
    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert field in spec
        assert mod.FIELD_PRINCIPLES[field] in spec


def test_scenario_capstone_4312_current_artifacts_skip_flagged_arc() -> None:
    """SCENARIO-CAPSTONE-4312: default .398 artifacts produce the honest scorecard."""

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
        "complete: v398_efficiency_hardened_in_generation_open_"
        "cross_domain_open_self_learning_helps_arc_excluded"
    )
    assert artifact["efficiency_pareto_hardened"] is True
    assert artifact["in_generation_moat_holds"] is False
    assert artifact["cross_domain_moat_holds"] is False
    assert artifact["verifier_thesis_state"] == "efficiency_parity_hardened"
    assert artifact["paper_ready"] is True
    assert artifact["per_axis_gaps"] == []
    assert artifact["arc_progress"]["status"] == "excluded_flagged_adversarial"
    assert artifact["flagged_artifacts_excluded"][0]["artifact_key"] == "4307_arc_progress"
    assert artifact["efficiency"]["cost_ratio"] == pytest.approx(1.03e-8)
    assert artifact["in_generation"]["controls_differentiated"] is True
    assert artifact["cross_domain"]["label_ablation_robust"] is True
    assert artifact["self_learning"]["online_adaptation_helps"] is True
    assert artifact["verifier_is_oracle_honored"] is True

    provenance = {row["artifact_key"]: row for row in artifact["upstream_provenance"]}
    assert provenance["4307_arc_progress"]["fields_imported"] == []
    expected_sha = hashlib.sha256(
        Path("results/experiment_4303_verifier_efficiency_parity_isoflops.json").read_bytes()
    ).hexdigest()
    assert provenance["4303_efficiency"]["sha256"] == expected_sha


def test_req_capstone_4312_missing_axis_does_not_zero_available_axes(tmp_path: Path) -> None:
    """REQ-CAPSTONE-4312: missing artifacts are per-axis gaps, not global blockers."""

    payloads = _minimal_payloads()
    payloads.pop("4305_cross_domain")
    payloads.pop("4307_arc_progress")
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
    assert artifact["efficiency_pareto_hardened"] is True
    assert artifact["in_generation_moat_holds"] is True
    assert artifact["cross_domain_moat_holds"] is False
    assert artifact["verifier_thesis_state"] == "in_generation_moat_holds"
    assert artifact["paper_ready"] is False
    assert artifact["unmet_gates"] == ["G2"]
    assert artifact["per_axis_gaps"] == [
        {"axis": "cross_domain", "artifact_key": "4305_cross_domain", "experiment_id": 4305},
        {"axis": "arc_progress", "artifact_key": "4307_arc_progress", "experiment_id": 4307},
    ]
    assert artifact["availability_report"]["available_artifact_keys"] == [
        "4303_efficiency",
        "4304_in_generation",
        "4306_self_learning",
    ]


def test_req_capstone_4312_flagged_live_critical_and_circular_do_not_headline(
    tmp_path: Path,
) -> None:
    """REQ-CAPSTONE-4312: flagged, live-critical, and circular inputs are bounded."""

    payloads = _minimal_payloads()
    payloads["4303_efficiency"]["verifier_is_oracle"] = True
    payloads["4304_in_generation"]["flagged_adversarial"] = True
    payloads["4305_cross_domain"]["verifier_is_oracle"] = True
    _write_default_artifacts(tmp_path, payloads)

    def live_flags(path: Path) -> list[dict[str, str]]:
        if path.name == mod.DEFAULT_UPSTREAMS["4306_self_learning"].path.name:
            return [{"kind": "TAUTOLOGY", "severity": "critical", "detail": "fixture"}]
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
    assert artifact["efficiency_pareto_hardened"] is False
    assert artifact["in_generation_moat_holds"] is False
    assert artifact["cross_domain_moat_holds"] is False
    assert artifact["verifier_thesis_state"] == "selection_moat_arc_only"
    assert artifact["verifier_is_oracle_honored"] is False
    assert artifact["oracle_distinct_violations"] == [
        "4303_efficiency:efficiency_pareto",
        "4305_cross_domain:cross_domain",
    ]
    excluded = {row["artifact_key"] for row in artifact["flagged_artifacts_excluded"]}
    assert excluded == {"4304_in_generation", "4306_self_learning"}
    provenance = {row["artifact_key"]: row for row in artifact["upstream_provenance"]}
    assert provenance["4304_in_generation"]["fields_imported"] == []
    assert provenance["4306_self_learning"]["fields_imported"] == []


def test_req_capstone_4312_blocks_only_when_no_v398_artifacts(tmp_path: Path) -> None:
    """REQ-CAPSTONE-4312: no landed .398 artifacts is the only global block."""

    artifact = mod.build_artifact(
        tmp_path,
        started_s=4.0,
        now_s=4.25,
        live_flag_runner=_clean_live_flags,
        summarize_runner=_summarize_zero,
        publication_gate_runner=lambda _: _paper_gate(True),
    )

    mod.validate_artifact(artifact)
    assert artifact["honest_verdict"] == "blocked_no_v398_artifacts"
    assert artifact["headline_outcome"] == "blocked_no_v398_artifacts"
    assert artifact["paper_ready"] is True
    assert artifact["efficiency_pareto_hardened"] is False
    assert artifact["in_generation_moat_holds"] is False
    assert artifact["cross_domain_moat_holds"] is False
    assert artifact["per_axis_gaps"] == [
        {"axis": "efficiency", "artifact_key": "4303_efficiency", "experiment_id": 4303},
        {"axis": "in_generation", "artifact_key": "4304_in_generation", "experiment_id": 4304},
        {"axis": "cross_domain", "artifact_key": "4305_cross_domain", "experiment_id": 4305},
        {"axis": "self_learning", "artifact_key": "4306_self_learning", "experiment_id": 4306},
        {"axis": "arc_progress", "artifact_key": "4307_arc_progress", "experiment_id": 4307},
    ]
    assert artifact["reproducibility_checksum"] == mod.BLOCKED_CHECKSUM


def test_req_capstone_4312_write_validate_and_helper_edges(tmp_path: Path) -> None:
    """REQ-CAPSTONE-4312: validation, checksum, helpers, and wrapper stay strict."""

    payloads = _minimal_payloads()
    _write_default_artifacts(tmp_path, payloads)
    out_path = mod.write_artifact(
        tmp_path,
        output_path=Path("results/experiment_4312_capstone_v398.json"),
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
    assert mod.efficiency_read(None, True)["status"] == "excluded_flagged_adversarial"
    assert mod.efficiency_read(None, False)["status"] == "missing_or_excluded"
    assert mod.in_generation_read(None, True)["status"] == "excluded_flagged_adversarial"
    assert mod.in_generation_read(None, False)["status"] == "missing_or_excluded"
    assert mod.cross_domain_read(None, True)["status"] == "excluded_flagged_adversarial"
    assert mod.cross_domain_read(None, False)["status"] == "missing_or_excluded"
    assert mod.self_learning_read(None, True)["status"] == "excluded_flagged_adversarial"
    assert mod.self_learning_read(None, False)["status"] == "missing_or_excluded"
    assert mod.arc_progress_read(None, True)["status"] == "excluded_flagged_adversarial"
    assert mod.arc_progress_read(None, False)["status"] == "missing_or_excluded"
    assert mod.checksum_from_provenance([]) == mod.BLOCKED_CHECKSUM

    for field, value, pattern in [
        ("honest_verdict", "not_terminal", "terminal-prefixed"),
        ("headline_outcome", "", "headline_outcome"),
        ("efficiency_pareto_hardened", "true", "bare bool"),
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

    assert mod.verifier_thesis_state(True, True, True, True) == "cross_domain_moat_holds"
    assert mod.verifier_thesis_state(True, True, False, False) == "in_generation_moat_holds"
    assert mod.verifier_thesis_state(True, False, False, False) == "efficiency_parity_hardened"
    assert mod.verifier_thesis_state(False, False, False, True) == "selection_moat_arc_only"
    assert mod.verifier_thesis_state(False, False, False, False) == "in_generation_still_open"
    assert mod._oracle_violations(  # noqa: SLF001
        {"reported_efficiency_pareto_holds": False},
        {"reported_diffusiongemma_guidance_moat": True, "verifier_is_oracle": True},
        {"reported_cross_domain_selection_holds": False},
        {"reported_online_adaptation_helps": True, "verifier_is_oracle": True},
    ) == ["4304_in_generation:in_generation", "4306_self_learning:self_learning"]

    parse_root = tmp_path / "parse_error"
    bad_artifact = parse_root / mod.DEFAULT_UPSTREAMS["4303_efficiency"].path
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

    wrapper = Path("results/experiment_4312_capstone_v398.py").read_text(encoding="utf-8")
    assert "capstone_v398_4312" in wrapper
