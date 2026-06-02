"""Tests for Exp 3717 full G4 headline provenance audit.

Spec traces: REQ-PUBLISH-3717, SCENARIO-PUBLISH-3717.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from scripts import experiment_3717_g4_full_provenance_audit as exp3717


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _north_star(root: Path) -> None:
    (root / "ops").mkdir(parents=True, exist_ok=True)
    (root / "ops" / "north-star.md").write_text(
        (
            "## 1. THE HEADLINE CLAIM\n"
            "Carnot reaches AUROC 0.9131 on FoVer with CI95 [0.9027, 0.9235], "
            "and FR-11 contributes +0.0185 AUROC with CI95 [0.0125, 0.0245]. "
            "The shipped detector math operating point is about 0.98 with ECE about 0.009.\n"
        ),
        encoding="utf-8",
    )


def _clean_payloads() -> dict[str, dict[str, Any]]:
    return {
        "experiment_2850_fover_dual_condition_integrity_v4.json": {
            "schema": "carnot.fover_dual_condition_integrity_v4",
            "condition_a_production_auroc_mean": 0.9131336,
            "random_seed": 42,
            "random_seeds_used": [42, 137, 271, 314, 1729],
            "reproducibility_checksum": "fover-g1-checksum",
            "duration_s": 2.0,
        },
        "experiment_2837_fover_memory_leakage_v3.json": {
            "schema": "carnot.fover_memory_leakage_v3",
            "condition_a_production_auroc_ci95": {
                "low": 0.9027316334533082,
                "high": 0.9235355665466916,
                "mean": 0.9131336,
            },
            "learning_contribution_ci95": {
                "low": 0.012461560295861967,
                "high": 0.024480839704138033,
                "mean": 0.0184712,
            },
            "random_seeds_used": [42, 137, 271, 314, 1729],
            "reproducibility_checksum": "fover-memory-checksum",
            "duration_s": 16.0,
            "flagged_adversarial": False,
        },
        "experiment_3706_reconcile_shipped_detector_heldout.json": {
            "schema": "carnot.reconcile_shipped_detector_heldout_3706.v1",
            "math_operating_point": {
                "auroc": 0.979656,
                "calibration": {"ece": 0.008158},
            },
            "random_seed": 3706,
            "reproducibility_checksum": "detector-checksum",
            "duration_s": 5.0,
        },
    }


def _write_payloads(root: Path, payloads: dict[str, dict[str, Any]]) -> None:
    for name, payload in payloads.items():
        _write_json(root / "results" / name, payload)


def _clean_verify(path: Path) -> dict[str, Any]:
    return {"artifact": str(path), "loaded": True, "flag_count": 0, "flags": []}


def _critical_verify(path: Path) -> dict[str, Any]:
    report = _clean_verify(path)
    if "3706" in path.name:
        report["flag_count"] = 1
        report["flags"] = [
            {"kind": "flagged_adversarial", "severity": "critical", "detail": "synthetic"}
        ]
    return report


@pytest.mark.parametrize(
    ("honest_outcome", "mutate", "verifier", "expected_verdict", "expected_status"),
    [
        pytest.param(
            "g4_fully_traced_all_clean",
            lambda payloads: payloads,
            _clean_verify,
            exp3717.SUCCESS_VERDICT,
            "fully_traced",
            id="g4_fully_traced_all_clean",
        ),
        pytest.param(
            "g4_gap_found",
            lambda payloads: {
                **payloads,
                "experiment_2837_fover_memory_leakage_v3.json": {
                    **{
                        k: v
                        for k, v in payloads[
                            "experiment_2837_fover_memory_leakage_v3.json"
                        ].items()
                        if k not in {"reproducibility_checksum", "random_seeds_used"}
                    },
                    "learning_contribution_ci95": {"low": 0.01, "high": 0.02, "mean": 0.019},
                },
                "experiment_3706_reconcile_shipped_detector_heldout.json": {
                    **payloads["experiment_3706_reconcile_shipped_detector_heldout.json"],
                    "flagged_adversarial": True,
                },
            },
            _critical_verify,
            exp3717.GAP_VERDICT,
            "gap_found",
            id="g4_gap_found",
        ),
        pytest.param(
            "blocked",
            lambda payloads: {
                name: payload
                for name, payload in payloads.items()
                if "2850" not in name
            },
            _clean_verify,
            exp3717.BLOCKED_VERDICT,
            "blocked",
            id="blocked",
        ),
    ],
)
def test_honest_outcomes_are_parametrized_on_synthetic_fixtures(
    tmp_path: Path,
    honest_outcome: str,
    mutate: Any,
    verifier: Any,
    expected_verdict: str,
    expected_status: str,
) -> None:
    """SCENARIO-PUBLISH-3717: success, gap, and blocked outcomes are honest."""
    _north_star(tmp_path)
    _write_payloads(tmp_path, mutate(_clean_payloads()))

    artifact = exp3717.audit_g4(repo_root=tmp_path, verifier=verifier, now_s=101.0, started_s=100.0)

    exp3717.validate_artifact(artifact)
    assert artifact["honest_verdict"] == expected_verdict
    assert artifact["g4_status"] == expected_status
    assert type(artifact["all_numbers_trace_to_clean_artifacts"]) is bool
    assert type(artifact["any_cited_source_flagged"]) is bool
    assert artifact["north_star_unmodified_assert"] is True

    if honest_outcome == "g4_fully_traced_all_clean":
        assert artifact["n_numbers_audited"] == 7
        assert artifact["all_numbers_trace_to_clean_artifacts"] is True
        assert artifact["any_cited_source_flagged"] is False
        assert artifact["operator_action_items"] == []
        assert all(row["adversarial_clean"] is True for row in artifact["provenance_rows"])
    elif honest_outcome == "g4_gap_found":
        assert artifact["n_numbers_audited"] == 7
        assert artifact["all_numbers_trace_to_clean_artifacts"] is False
        assert artifact["any_cited_source_flagged"] is True
        assert any("flagged_adversarial" in item for item in artifact["operator_action_items"])
        assert any("missing random_seed" in item for item in artifact["operator_action_items"])
        assert any("value mismatch" in item for item in artifact["operator_action_items"])
    else:
        assert artifact["n_numbers_audited"] == 0
        assert artifact["provenance_rows"] == []
        assert artifact["all_numbers_trace_to_clean_artifacts"] is False


def test_validate_rejects_required_field_and_bare_bool_errors(tmp_path: Path) -> None:
    """REQ-PUBLISH-3717: required fields and bare booleans are enforced."""
    _north_star(tmp_path)
    _write_payloads(tmp_path, _clean_payloads())
    artifact = exp3717.audit_g4(repo_root=tmp_path, verifier=_clean_verify, started_s=1.0, now_s=2.0)

    invalid_cases = [
        ({k: v for k, v in artifact.items() if k != "provenance_rows"}, "missing required"),
        ({**artifact, "inference_substrate": "verifier_ensemble_against_cached_candidates"}, "aggregation"),
        ({**artifact, "honest_verdict": "complete: unsupported"}, "honest_verdict"),
        ({**artifact, "all_numbers_trace_to_clean_artifacts": "true"}, "bare boolean"),
        ({**artifact, "any_cited_source_flagged": "false"}, "bare boolean"),
        ({**artifact, "north_star_unmodified_assert": "true"}, "bare boolean"),
        ({**artifact, "adversarial_verify_clean": "true"}, "bare boolean"),
        ({**artifact, "provenance_rows": "rows"}, "provenance_rows"),
        ({**artifact, "g4_status": "unknown"}, "g4_status"),
        ({**artifact, "n_numbers_audited": 99}, "n_numbers_audited"),
        ({**artifact, "inference_substrate": "aggregation_from_upstream_artifacts GGUF"}, "GGUF"),
        ({**artifact, "provenance_rows": ["not-a-row"], "n_numbers_audited": 1}, "row"),
        (
            {
                **artifact,
                "provenance_rows": [
                    {k: v for k, v in artifact["provenance_rows"][0].items() if k != "has_seed"}
                ],
                "n_numbers_audited": 1,
            },
            "row missing",
        ),
        (
            {
                **artifact,
                "provenance_rows": [{**artifact["provenance_rows"][0], "has_seed": "true"}],
                "n_numbers_audited": 1,
            },
            "bare boolean",
        ),
        ({**artifact, "field_principles": {}}, "field principles"),
        ({**artifact, "field_principles": None}, "field_principles"),
        ({**artifact, "acceptance_gate": {"passed": True}}, "acceptance_gate"),
    ]

    for invalid, expected in invalid_cases:
        with pytest.raises(ValueError, match=expected):
            exp3717.validate_artifact(invalid)


def test_main_writes_result_and_rechecks_own_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-PUBLISH-3717: CLI writes the deliverable and stamps self-clean status."""
    _north_star(tmp_path)
    _write_payloads(tmp_path, _clean_payloads())
    monkeypatch.setattr(exp3717, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(exp3717, "verify_artifact", _clean_verify)

    rc = exp3717.main([])

    out_path = tmp_path / exp3717.OUTPUT_REL_PATH
    artifact = json.loads(out_path.read_text(encoding="utf-8"))
    assert rc == 0
    assert artifact["honest_verdict"] == exp3717.SUCCESS_VERDICT
    assert artifact["adversarial_verify_clean"] is True
    assert artifact["acceptance_gate"]["passed"] is True


def test_main_rejects_unexpected_arguments_for_req_publish_3717() -> None:
    """REQ-PUBLISH-3717: the runner has no mutable CLI knobs."""
    with pytest.raises(SystemExit, match="accepts no arguments"):
        exp3717.main(["--unexpected"])
