"""REQ-VERIFY-4611 adversarial_verify reader hardening.

Spec refs: REQ-VERIFY-4611, SCENARIO-VERIFY-4611.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import scripts.adversarial_verify as adversarial_verify


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "verification" / "spec.md"
EXP4592 = REPO / "results" / "experiment_4592_generation_completeness_wiring.json"
EXP4597 = REPO / "results" / "experiment_4597_integration_gate.json"
EXP4598 = REPO / "results" / "experiment_4598_winner_generated_rate_metric.json"


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _tautology_flags(payload: dict[str, Any]) -> list[Any]:
    flags: list[Any] = []
    adversarial_verify.check_tautology(payload, flags)
    return [flag for flag in flags if flag.kind == "TAUTOLOGY"]


def _critical_tautology(payload: dict[str, Any]) -> list[Any]:
    return [flag for flag in _tautology_flags(payload) if flag.severity == "critical"]


def _write_payload(tmp_path: Path, payload: dict[str, Any]) -> Path:
    path = tmp_path / "artifact.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path


def _world_model_flags(report: dict[str, Any]) -> list[dict[str, str]]:
    return [
        flag
        for flag in report["flags"]
        if flag["kind"] == "WORLD_MODEL_TRUST_DEGENERACY"
    ]


def _world_model_trust_artifact(**overrides: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "experiment": "experiment_4611_world_model_trust_fixture",
        "honest_verdict": "success: world_model_trust_energy_pass_rate_up_1_first_win_up",
        "inference_substrate": "verifier_ensemble_against_cached_candidates",
        "world_model_trust_pass_rate_new": 1.0,
        "world_model_trust_pass_rate_binary": 0.0,
        "world_model_trust_pass_rate_delta": 1.0,
        "trust_pass_numerator": 1,
        "trust_pass_denominator": 1,
        "verifier_is_oracle": False,
        "n_correct_grid_changing_transitions": 1,
    }
    payload.update(overrides)
    return payload


def test_req_4611_spec_declares_reader_hardening_contract() -> None:
    """REQ-VERIFY-4611: OpenSpec declares both reader-side guards."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-VERIFY-4611" in spec
    assert "SCENARIO-VERIFY-4611" in spec
    assert "shared the same finite variant denominator" in spec
    assert "correctly predicted grid-changing transition" in spec


def test_scenario_4611_real_424_fixtures_do_not_critical_on_k_over_n_rates() -> None:
    """SCENARIO-VERIFY-4611: .424 k/N rate collisions are not quarantined."""

    for fixture in (EXP4592, EXP4597, EXP4598):
        payload = _load(fixture)
        critical = _critical_tautology(payload)

        assert critical == [], [flag.detail for flag in critical]


def test_req_4611_unrelated_high_precision_tautology_still_fires() -> None:
    """REQ-VERIFY-4611: narrow carve-out does not hide fabricated metrics."""

    payload = {
        "experiment": "experiment_4611_fabricated_tautology_fixture",
        "honest_verdict": "success: fabricated_metrics_should_not_pass",
        "heldout_auroc": 0.913127481234,
        "energy_margin": 0.913127481234,
        "variant_attempts_count": 25,
    }

    critical = _critical_tautology(payload)

    assert critical
    assert "heldout_auroc" in critical[0].detail
    assert "energy_margin" in critical[0].detail


def test_scenario_4611_exp4597_success_verdict_is_not_read_as_null() -> None:
    """SCENARIO-VERIFY-4611: exp4597 fixture is clean after TAUTOLOGY hardening."""

    flags: list[Any] = []
    adversarial_verify.check_false_negative_risk(_load(EXP4597), flags)
    fnr = [flag for flag in flags if flag.kind == "FALSE_NEGATIVE_RISK"]

    assert fnr == []


def test_req_4611_degenerate_identity_world_model_trust_pass_is_flagged(
    tmp_path: Path,
) -> None:
    """REQ-VERIFY-4611: trust-pass claims need non-degenerate change evidence."""

    path = _write_payload(
        tmp_path,
        _world_model_trust_artifact(n_correct_grid_changing_transitions=0),
    )
    report = adversarial_verify.verify_artifact(path)
    flags = _world_model_flags(report)

    assert flags
    assert flags[0]["severity"] == "critical"
    assert "grid-changing" in flags[0]["detail"]


def test_req_4611_circular_world_model_trust_pass_is_flagged(tmp_path: Path) -> None:
    """REQ-VERIFY-4611: trust-pass claims must declare verifier_is_oracle=false."""

    path = _write_payload(
        tmp_path,
        _world_model_trust_artifact(verifier_is_oracle=True),
    )
    report = adversarial_verify.verify_artifact(path)
    flags = _world_model_flags(report)

    assert flags
    assert flags[0]["severity"] == "critical"
    assert "verifier_is_oracle" in flags[0]["detail"]


def test_req_4611_nondegenerate_world_model_trust_pass_is_clean(
    tmp_path: Path,
) -> None:
    """REQ-VERIFY-4611: oracle-distinct non-degenerate trust pass is allowed."""

    path = _write_payload(tmp_path, _world_model_trust_artifact())
    report = adversarial_verify.verify_artifact(path)

    assert _world_model_flags(report) == []


def test_req_4611_runner_builds_required_terminal_artifact() -> None:
    """REQ-VERIFY-4611: Exp 4611 emits the required evidence fields."""

    from carnot import experiment_4611_adversarial_verify_hardening as mod

    artifact = mod.build_artifact(
        root=REPO,
        preconditions_checked={
            "agents_md_read": True,
            "codex_or_opencode_md_read": True,
            "adversarial_verify_parse_ok": True,
            "fixtures_present": True,
            "spec_has_req_4611": True,
            "research_conductor_modified": False,
            "network_required": False,
            "ok": True,
        },
    )

    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert field in artifact
        assert field in artifact["field_principles"]
    assert artifact["honest_verdict"] == (
        "success: adversarial_verify_hardened_tautology_carveout_plus_wm_trust_guard_tests_green."
    )
    assert artifact["tautology_carveout_added"] is True
    assert artifact["regression_424_artifacts_unflagged"] is True
    assert artifact["genuine_tautology_still_fires"] is True
    assert artifact["wm_trust_guard_added"] is True
    assert artifact["tests_added"]["passed"] is True
    assert artifact["research_conductor_modified"] is False
    assert artifact["random_seed"] == 4611
    assert artifact["preconditions_checked"]["ok"] is True
    assert artifact["reproducibility_checksum"] == mod.payload_checksum(artifact)
    assert mod.validate_artifact(artifact) == []


def test_req_4611_runner_validation_rejects_malformed_artifact() -> None:
    """REQ-VERIFY-4611: artifact schema validation fails closed."""

    from carnot import experiment_4611_adversarial_verify_hardening as mod

    artifact = mod.build_artifact(
        root=REPO,
        preconditions_checked={
            "agents_md_read": True,
            "codex_or_opencode_md_read": True,
            "adversarial_verify_parse_ok": True,
            "fixtures_present": True,
            "spec_has_req_4611": True,
            "research_conductor_modified": False,
            "network_required": False,
            "ok": True,
        },
    )
    bad = dict(artifact)
    bad["honest_verdict"] = "not_terminal"
    bad["inference_substrate"] = "wrong"
    bad["tautology_carveout_added"] = False
    bad["regression_424_artifacts_unflagged"] = False
    bad["genuine_tautology_still_fires"] = False
    bad["wm_trust_guard_added"] = False
    bad["research_conductor_modified"] = True
    bad["random_seed"] = 0
    bad["tests_added"] = {"passed": False}
    bad["preconditions_checked"] = {"ok": False}
    bad["field_principles"] = {}
    bad["reproducibility_checksum"] = "sha256:bad"
    errors = mod.validate_artifact(bad)

    assert "honest_verdict_terminal_prefix" in errors
    assert "inference_substrate" in errors
    assert "tautology_carveout_added" in errors
    assert "regression_424_artifacts_unflagged" in errors
    assert "genuine_tautology_still_fires" in errors
    assert "wm_trust_guard_added" in errors
    assert "research_conductor_modified" in errors
    assert "random_seed" in errors
    assert "tests_added.passed" in errors
    assert "preconditions_checked.ok" in errors
    assert "field_principles.honest_verdict" in errors
    assert "reproducibility_checksum" in errors

    wrong_types = dict(artifact)
    wrong_types["tests_added"] = []
    wrong_types["preconditions_checked"] = []
    wrong_types["field_principles"] = []
    type_errors = mod.validate_artifact(wrong_types)

    assert "tests_added" in type_errors
    assert "preconditions_checked" in type_errors
    assert "field_principles" in type_errors
