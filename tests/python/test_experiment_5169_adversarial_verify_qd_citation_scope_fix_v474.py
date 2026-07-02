"""Tests for Exp 5169 QD citation-scope adversarial_verify fix.

Spec refs: REQ-ARC-WMTE-5169,
SCENARIO-ARC-WMTE-5169-QD-CITATION-SCOPE,
SCENARIO-ARC-WMTE-5169-WARN-SEVERITY-HANDLING.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from carnot import experiment_5169_adversarial_verify_qd_citation_scope_fix_v474 as mod


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "arc-world-model-trust-energy" / "spec.md"


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _clean_exp5156_report() -> dict:
    return {
        "artifact": "results/experiment_5156_archive_472_activate_473.json",
        "loaded": True,
        "flag_count": 0,
        "max_severity": -1,
        "flags": [],
    }


def _backfill_summary() -> dict:
    return {
        "scanned_artifact_count": 3,
        "legacy_qd_flagged_count": 1,
        "current_qd_flagged_count": 1,
        "artifacts_still_flagged_count": 1,
        "artifacts_newly_unflagged_count": 1,
        "artifacts_newly_unflagged": ["experiment_5156_archive_472_activate_473.json"],
        "artifacts_newly_flagged_count": 1,
        "artifacts_newly_flagged": ["experiment_5154_energy_fitness_directed_exploration_v472.json"],
        "aggregation_citation_unflags": ["experiment_5156_archive_472_activate_473.json"],
        "any_unexpected_unflag": False,
        "errors_count": 0,
    }


def _high_precision_summary() -> dict:
    return {
        "scope": ["DURATION_TOO_SHORT", "GATE_PASSED_WITHOUT_DATA"],
        "qualifying_unstamped_critical_count": 0,
        "would_stamp": [],
    }


def test_req_arc_wmte_5169_spec_declares_receipt_contract() -> None:
    """REQ-ARC-WMTE-5169: OpenSpec declares the citation-scope receipt."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    for marker in (
        "REQ-ARC-WMTE-5169",
        "SCENARIO-ARC-WMTE-5169-QD-CITATION-SCOPE",
        "SCENARIO-ARC-WMTE-5169-WARN-SEVERITY-HANDLING",
        str(mod.RESULT_RELATIVE_PATH),
        "root_cause_confirmed",
        "severity_handling_audit_result",
        "backfill_dry_run_summary",
    ):
        assert marker in spec


def test_req_arc_wmte_5169_builds_valid_terminal_artifact() -> None:
    """REQ-ARC-WMTE-5169: receipt fields are principle-wrapped and validate."""

    artifact = mod.build_artifact(
        root=REPO,
        exp5156_verify_report=_clean_exp5156_report(),
        backfill_summary=_backfill_summary(),
        high_precision_summary=_high_precision_summary(),
        known_issues_md_updated=True,
        tests_passing=True,
    )

    assert artifact["honest_verdict"]["value"].startswith("complete:")
    assert "exp5156_resolves_clean" in artifact["honest_verdict"]["value"]
    assert artifact["root_cause_confirmed"]["value"].startswith("The old QD guard")
    assert artifact["severity_handling_audit_result"]["value"].startswith("bug_found_and_fixed:")
    assert artifact["exp5156_resolved"]["value"] is True
    assert artifact["tests_added"]["value"] == 6
    assert artifact["tests_passing"]["value"] is True
    assert artifact["known_issues_md_updated"]["value"] is True
    assert artifact["inference_substrate"]["value"] == mod.INFERENCE_SUBSTRATE
    assert artifact["random_seed"]["value"] == mod.RANDOM_SEED
    assert artifact["reproducibility_checksum"]["value"] == mod.payload_checksum(artifact)
    assert mod.validate_artifact(artifact) == []


def test_scenario_arc_wmte_5169_qd_backfill_summary_scopes_archive_citations(
    tmp_path: Path,
) -> None:
    """SCENARIO-ARC-WMTE-5169-QD-CITATION-SCOPE: dry-run diff classifies aggregation unflags."""

    results = tmp_path / "results"
    _write_json(
        results / "experiment_5156_archive_472_activate_473.json",
        {
            "experiment": "experiment_5156_archive_472_activate_473",
            "honest_verdict": "complete_archive_472_closed_473_active_runtime_clean",
            "inference_substrate": mod.INFERENCE_SUBSTRATE,
            "generation_axis_retirement_signal": {
                "current_energy_fitness_result": "honest_null",
                "third_consecutive_generation_axis_null": True,
            },
            "milestone_archive_summary": [
                {
                    "experiment_id": "exp5154-energy-fitness-directed-exploration-v472",
                    "honest_verdict": (
                        "complete: "
                        "energy_fitness_qd_winning_trajectory_not_surfaced_reproducible_delta_0"
                    ),
                    "classification": "honest_null_generation_axis",
                    "winning_trajectory_surfaced": False,
                }
            ],
        },
    )
    _write_json(
        results / "experiment_9999_first_party_qd.json",
        {
            "experiment": "experiment_9999_first_party_qd",
            "game": "tn36",
            "headline": "energy-fitness QD live generation measurement",
            "honest_verdict": "complete: energy_fitness_qd_live_generation_measurement",
            "qd_arm_result": {"winning_trajectory_surfaced": False},
        },
    )
    (results / "experiment_0000_bad.json").write_text("{", encoding="utf-8")

    summary = mod.qd_backfill_dry_run_summary(results)

    assert summary["scanned_artifact_count"] == 3
    assert summary["artifacts_newly_unflagged"] == [
        "experiment_5156_archive_472_activate_473.json"
    ]
    assert summary["aggregation_citation_unflags"] == [
        "experiment_5156_archive_472_activate_473.json"
    ]
    assert summary["artifacts_still_flagged_count"] == 1
    assert summary["current_qd_flagged_count"] == 1
    assert summary["any_unexpected_unflag"] is False
    assert summary["errors_count"] == 1


def test_req_arc_wmte_5169_legacy_qd_diff_edges() -> None:
    """REQ-ARC-WMTE-5169: legacy diff helper covers non-context, non-generation, and win cases."""

    assert mod._wrapped_value({}, "missing") is None
    assert mod._legacy_qd_flags({"experiment": "plain_receipt"}) == []
    assert (
        mod._legacy_qd_flags(
            {
                "experiment": "experiment_9999_energy_fitness_qd_diagnostic",
                "game": "tn36",
                "headline": "energy-fitness QD diagnostic",
            }
        )
        == []
    )

    winner_true = mod._legacy_qd_flags(
        {
            "experiment": "experiment_9999_energy_fitness_qd_generation",
            "game": "tn36",
            "headline": "energy-fitness QD live generation",
            "winner_generated": True,
        }
    )
    winner_count = mod._legacy_qd_flags(
        {
            "experiment": "experiment_9999_energy_fitness_qd_generation",
            "game": "tn36",
            "headline": "energy-fitness QD live generation",
            "winner_generated_count": 1,
        }
    )
    positive_delta = mod._legacy_qd_flags(
        {
            "experiment": "experiment_9999_energy_fitness_qd_generation",
            "game": "tn36",
            "headline": "energy-fitness QD live generation",
            "solve_rate_delta": 0.1,
        }
    )
    positive_pair = mod._legacy_qd_flags(
        {
            "experiment": "experiment_9999_energy_fitness_qd_generation",
            "game": "tn36",
            "headline": "energy-fitness QD live generation",
            "live_solve_rate_qd": 0.2,
            "live_solve_rate_search_baseline": 0.0,
        }
    )

    for flags in (winner_true, winner_count, positive_delta, positive_pair):
        assert {flag["kind"] for flag in flags} == {
            mod.av.QD_RANDOM_MUTATION_ABLATION_OMITTED_KIND,
            mod.av.QD_WITHOUT_RANDOM_MUTATION_ABLATION_KIND,
        }


def test_scenario_arc_wmte_5169_warn_only_severity_audit_is_non_quarantine() -> None:
    """SCENARIO-ARC-WMTE-5169-WARN-SEVERITY-HANDLING: WARN-only exits stay non-quarantine."""

    audit = mod.severity_handling_audit()

    assert audit["clean"] is True
    assert audit["warn_only_payload"]["green"] is False
    assert audit["warn_only_payload"]["max_severity"] == 1
    assert audit["warn_only_payload"]["flagged_adversarial"] is False
    assert audit["critical_payload"]["flagged_adversarial"] is True


def test_req_arc_wmte_5169_validation_rejects_malformed_artifact() -> None:
    """REQ-ARC-WMTE-5169: validator fails closed on required fields."""

    artifact = mod.build_artifact(
        root=REPO,
        exp5156_verify_report=_clean_exp5156_report(),
        backfill_summary=_backfill_summary(),
        high_precision_summary=_high_precision_summary(),
        known_issues_md_updated=True,
        tests_passing=True,
    )
    bad = dict(artifact)
    bad["honest_verdict"] = {"value": "not_terminal", "principle": "wrong"}
    bad["inference_substrate"] = {"value": "live_llm_inference", "principle": "wrong"}
    bad["random_seed"] = {"value": 0, "principle": "wrong"}
    bad["exp5156_resolved"] = {"value": False, "principle": "wrong"}
    bad["tests_passing"] = {"value": False, "principle": "wrong"}
    bad["known_issues_md_updated"] = {"value": False, "principle": "wrong"}
    bad["backfill_dry_run_summary"] = {"value": {"any_unexpected_unflag": True}, "principle": "wrong"}
    bad["reproducibility_checksum"] = {"value": "sha256:bad", "principle": "wrong"}

    errors = mod.validate_artifact(bad)

    assert "honest_verdict.terminal_prefix" in errors
    assert "honest_verdict.principle" in errors
    assert "inference_substrate.value" in errors
    assert "random_seed.value" in errors
    assert "exp5156_resolved.value" in errors
    assert "tests_passing.value" in errors
    assert "known_issues_md_updated.value" in errors
    assert "backfill_dry_run_summary.artifacts_still_flagged_count" in errors
    assert "backfill_dry_run_summary.any_unexpected_unflag" in errors
    assert "reproducibility_checksum.value" in errors

    missing_shape = dict(artifact)
    missing_shape["root_cause_confirmed"] = "bare"
    assert "root_cause_confirmed.shape" in mod.validate_artifact(missing_shape)

    bad_summary_shape = dict(artifact)
    bad_summary_shape["backfill_dry_run_summary"] = {
        "value": None,
        "principle": mod.FIELD_PRINCIPLES["backfill_dry_run_summary"],
    }
    bad_summary_shape["reproducibility_checksum"] = {
        "value": "sha256:bad",
        "principle": mod.FIELD_PRINCIPLES["reproducibility_checksum"],
    }
    assert "backfill_dry_run_summary.value" in mod.validate_artifact(bad_summary_shape)


def test_req_arc_wmte_5169_known_issue_marker_and_write_path(tmp_path: Path) -> None:
    """REQ-ARC-WMTE-5169: known-issues marker is checked and the JSON writer is stable."""

    (tmp_path / "ops").mkdir()
    (tmp_path / "ops" / "known-issues.md").write_text(
        f"### {mod.KNOWN_ISSUES_MARKER}\n",
        encoding="utf-8",
    )
    assert mod.known_issues_updated(tmp_path) is True

    artifact = mod.build_artifact(
        root=tmp_path,
        exp5156_verify_report=_clean_exp5156_report(),
        backfill_summary=_backfill_summary(),
        high_precision_summary=_high_precision_summary(),
        known_issues_md_updated=True,
        tests_passing=True,
    )
    path = mod.write_artifact(tmp_path, artifact)
    saved = json.loads(path.read_text(encoding="utf-8"))

    assert path == tmp_path / mod.RESULT_RELATIVE_PATH
    assert saved["experiment"] == mod.EXPERIMENT
    assert saved["reproducibility_checksum"]["value"] == mod.payload_checksum(saved)


def test_req_arc_wmte_5169_auxiliary_boundaries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """REQ-ARC-WMTE-5169: high-precision wrapper, exp5156 reporter, writer error, and main are covered."""

    monkeypatch.setattr(
        mod.av,
        "backfill_stamps",
        lambda paths, apply, kinds_filter: [{"path": str(tmp_path / "results" / "x.json")}],
    )
    high_precision = mod.high_precision_backfill_dry_run_summary(tmp_path / "results")
    assert high_precision["qualifying_unstamped_critical_count"] == 1
    assert high_precision["would_stamp"] == ["x.json"]

    monkeypatch.setattr(mod.av, "verify_artifact", lambda path: {"artifact": str(path), "flag_count": 0, "flags": []})
    assert mod.exp5156_report(tmp_path)["artifact"].endswith(str(mod.EXP5156_RELATIVE_PATH))
    assert mod.exp5156_resolved_from_report({"flag_count": 1, "flags": []}) is False
    assert mod.exp5156_resolved_from_report(
        {"flag_count": 1, "flags": [{"kind": mod.av.QD_RANDOM_MUTATION_ABLATION_OMITTED_KIND}]}
    ) is False

    with pytest.raises(ValueError, match="invalid Exp 5169 artifact"):
        mod.write_artifact(tmp_path, {"honest_verdict": {"value": "bad", "principle": "bad"}})

    monkeypatch.setattr(mod, "write_artifact", lambda: tmp_path / mod.RESULT_RELATIVE_PATH)
    assert mod.main() == 0
    assert str(mod.RESULT_RELATIVE_PATH) in capsys.readouterr().out
