"""Tests for Exp 4775 .439 ARC null silent-bug audit.

Spec refs: REQ-ARC-WMTE-4775, SCENARIO-ARC-WMTE-4775-SILENT-BUG-AUDIT,
SCENARIO-ARC-WMTE-4775-BLOCKED-PRECONDITION.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec/capabilities/arc-world-model-trust-energy/spec.md"
SHUFFLE_CODE = "rng.permutation(labels)\n_loo_metrics_candidate(shuffled_rows, 'structural')"


def _spec_section() -> str:
    spec = SPEC_PATH.read_text(encoding="utf-8")
    start = spec.index("### REQ-ARC-WMTE-4775")
    end = spec.index("### REQ-ARC-WMTE-4731", start)
    return spec[start:end]


def _s0prime_payload(
    *,
    origin_refit: bool = True,
    shuffled_auroc: float = 0.503,
    shuffled_resamples: int = 16,
) -> dict[str, Any]:
    origin_probe = {
        "loo_auroc": 0.5,
        "origin_counts": {"induced": 4},
        "status": "origin_matched_refit_complete",
    }
    if origin_refit:
        origin_probe["refit_on_origin_matched_data"] = True
    else:
        origin_probe["status"] = "origin_matched_single_origin_all_induced"
    return {
        "experiment": "experiment_4771_structural_energy_s0prime_origin_matched",
        "honest_verdict": "success_structural_energy_s0prime_reopens_s1",
        "n_candidate_rows": 4,
        "n_pos": 2,
        "n_neg": 2,
        "loo_auroc_majority_control": 0.5,
        "origin_probe_auroc": 0.5,
        "origin_probe": origin_probe,
        "shuffled_label_control_auroc": shuffled_auroc,
        "controls": {"shuffled_label_resamples": shuffled_resamples},
        "dataset_diagnostics": {"origin_matched": True},
        "per_game_class_balance": {
            "g0": {"correct": 1, "wrong": 1, "rows": 2, "contributes_to_loo": True},
            "g1": {"correct": 1, "wrong": 1, "rows": 2, "contributes_to_loo": True},
            "single": {"correct": 0, "wrong": 2, "rows": 2, "contributes_to_loo": False},
        },
        "per_game_loo": {
            "structural": {
                "g0": {"n_pos": 1, "n_neg": 1, "skipped": False},
                "g1": {"n_pos": 1, "n_neg": 1, "skipped": False},
                "single": {
                    "n_pos": 0,
                    "n_neg": 2,
                    "skipped": True,
                    "skip_reason": "test_fold_single_class",
                },
            }
        },
    }


def _levelup_payload() -> dict[str, Any]:
    return {
        "experiment": "experiment_4772_levelup_attempt",
        "honest_verdict": "complete_ka59_no_new_level_residual_existing_depth",
        "attempted_games": [
            {
                "game": "ka59",
                "prior_level": 1,
                "reached_level": 1,
                "target_level": 2,
                "offline_reproduced_existing_depth": True,
                "offline_reproduced_new_depth": False,
                "solution_labels": ["4", "C:1"],
                "reproduction_gate": {"reproduced": True, "reached_level": 1},
                "residual_cause": "reproduced_existing_or_lower_level",
            }
        ],
        "new_levels_banked": 0,
        "offline_reproduced": False,
        "reproduced_levels": 0,
        "dead_ends": ["same-depth reproduction"],
    }


def _heldout_payload(*, annotate_flat: bool = True) -> dict[str, Any]:
    return {
        "experiment": "experiment_4774_heldout_first_win_readiness",
        "honest_verdict": "complete: heldout_first_win_flat_genuine_null",
        "heldout_first_win_rate": 0.04,
        "first_win_baseline": 0.04,
        "heldout_first_win_delta_vs_baseline": 0.0,
        "heldout_variant_attempts": 100,
        "positive_control_passed": annotate_flat,
        "parity_test_green": True,
        "null_delta_methodology_note": (
            "Held-out first-win rate equals the 0.04 baseline; genuine null."
            if annotate_flat
            else ""
        ),
    }


def test_req_arc_wmte_4775_spec_declares_audit_contract() -> None:
    """REQ-ARC-WMTE-4775: OpenSpec declares the audit and required principles."""

    from carnot import experiment_4775_silent_bug_audit as mod

    section = _spec_section()

    assert "REQ-ARC-WMTE-4775" in section
    assert "SCENARIO-ARC-WMTE-4775-SILENT-BUG-AUDIT" in section
    assert "SCENARIO-ARC-WMTE-4775-BLOCKED-PRECONDITION" in section
    assert mod.RESULT_RELATIVE_PATH in section
    assert mod.AUDIT_REPORT_RELATIVE_PATH in section
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert field in section
        assert principle["principle"] in section


def test_scenario_arc_wmte_4775_accepts_only_fired_s0prime_controls() -> None:
    """SCENARIO-ARC-WMTE-4775-SILENT-BUG-AUDIT: S0' controls need positive fire evidence."""

    from carnot import experiment_4775_silent_bug_audit as mod

    good = mod.audit_null_artifact(
        "experiment_4771_structural_energy_s0prime_origin_matched",
        _s0prime_payload(origin_refit=True),
        module_source=SHUFFLE_CODE,
    )
    bad_origin = mod.audit_null_artifact(
        "experiment_4771_structural_energy_s0prime_origin_matched",
        _s0prime_payload(origin_refit=False),
        module_source=SHUFFLE_CODE,
    )
    bad_shuffle = mod.audit_null_artifact(
        "experiment_4771_structural_energy_s0prime_origin_matched",
        _s0prime_payload(origin_refit=True, shuffled_auroc=0.5, shuffled_resamples=0),
        module_source="return {'loo_auroc': 0.5}",
    )

    assert good["verdict"] == "trustworthy_null"
    assert good["s0prime_leak_controls_fired"] is True
    assert good["s0prime_leak_control_checks"]["class_balance_non_degenerate"] is True
    assert good["s0prime_leak_control_checks"]["origin_probe_refit_on_origin_matched_data"] is True
    assert good["s0prime_leak_control_checks"]["shuffled_label_permuted_and_reran_loo"] is True

    assert bad_origin["verdict"] == "silent_bug_must_reopen"
    assert bad_origin["s0prime_leak_controls_fired"] is False
    assert "s0prime_origin_probe_not_refit" in bad_origin["silent_bug_signatures"]
    assert "origin_probe_status=origin_matched_single_origin_all_induced" in bad_origin["exercise_evidence"]

    assert bad_shuffle["verdict"] == "silent_bug_must_reopen"
    assert bad_shuffle["s0prime_leak_controls_fired"] is False
    assert "s0prime_shuffled_label_control_not_permuted_loo" in bad_shuffle["silent_bug_signatures"]


def test_scenario_arc_wmte_4775_detects_class_and_tautology_noops() -> None:
    """SCENARIO-ARC-WMTE-4775-SILENT-BUG-AUDIT: degenerate classes and 0.04 tautologies reopen."""

    from carnot import experiment_4775_silent_bug_audit as mod

    s0 = _s0prime_payload(origin_refit=True)
    s0["per_game_class_balance"]["g0"] = {
        "correct": 0,
        "wrong": 2,
        "rows": 2,
        "contributes_to_loo": True,
    }
    levelup = _levelup_payload()
    levelup["attempted_games"][0]["engine_cell_changes"] = 0

    bad_balance = mod.audit_null_artifact(
        "experiment_4771_structural_energy_s0prime_origin_matched",
        s0,
        module_source=SHUFFLE_CODE,
    )
    dead_engine = mod.audit_null_artifact("experiment_4772_levelup_attempt", levelup)
    tautology = mod.audit_null_artifact(
        "experiment_4774_heldout_first_win_readiness",
        _heldout_payload(annotate_flat=False),
    )

    assert bad_balance["verdict"] == "silent_bug_must_reopen"
    assert "s0prime_class_balance_degenerate" in bad_balance["silent_bug_signatures"]
    assert dead_engine["verdict"] == "silent_bug_must_reopen"
    assert "dead_identity_engine_zero_cell_changes" in dead_engine["silent_bug_signatures"]
    assert tautology["verdict"] == "silent_bug_must_reopen"
    assert "first_win_0_04_tautology_unannotated" in tautology["silent_bug_signatures"]


def test_req_arc_wmte_4775_run_checked_in_artifacts() -> None:
    """REQ-ARC-WMTE-4775: checked-in .439 artifacts produce the expected audit."""

    from carnot import experiment_4775_silent_bug_audit as mod

    artifact = mod.run(root=REPO, write=False)

    assert mod.artifact_schema_errors(artifact) == []
    assert artifact["honest_verdict"] == "complete_arc_null_silent_bug_audit_3_nulls_1_reopen"
    assert artifact["nulls_audited"] == 3
    assert artifact["s0prime_leak_controls_fired"] is False
    assert artifact["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert artifact["preconditions_checked"]["milestone_439_artifacts_present"] is True
    assert artifact["reproducibility_checksum"] == mod.payload_checksum(artifact)
    assert [row["null_id"] for row in artifact["silent_bugs_found"]] == [
        "experiment_4771_structural_energy_s0prime_origin_matched"
    ]
    assert set(artifact["trusted_nulls"]) == {
        "experiment_4772_levelup_attempt",
        "experiment_4774_heldout_first_win_readiness",
    }


def test_req_arc_wmte_4775_write_artifact_and_append_markdown(tmp_path: Path) -> None:
    """REQ-ARC-WMTE-4775: complete audits write JSON and append the ops report."""

    from carnot import experiment_4775_silent_bug_audit as mod

    payloads = {
        "results/experiment_4771_structural_energy_s0prime_origin_matched.json": _s0prime_payload(
            origin_refit=False
        ),
        "results/experiment_4772_levelup_attempt.json": _levelup_payload(),
        "results/experiment_4774_heldout_first_win_readiness.json": _heldout_payload(),
    }
    for rel, payload in payloads.items():
        path = tmp_path / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload), encoding="utf-8")
    module_path = tmp_path / mod.S0PRIME_MODULE_RELATIVE_PATH
    module_path.parent.mkdir(parents=True, exist_ok=True)
    module_path.write_text(SHUFFLE_CODE, encoding="utf-8")
    report = tmp_path / mod.AUDIT_REPORT_RELATIVE_PATH
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text("# ARC Null Silent-Bug Audit\n", encoding="utf-8")

    artifact = mod.run(root=tmp_path, write=True)

    loaded = json.loads((tmp_path / mod.RESULT_RELATIVE_PATH).read_text(encoding="utf-8"))
    assert loaded == artifact
    text = report.read_text(encoding="utf-8")
    assert "## Experiment 4775 .439 ARC Null Silent-Bug Audit" in text
    assert "`experiment_4771_structural_energy_s0prime_origin_matched`" in text
    size_after_first = len(text)
    mod.append_markdown_report(artifact, root=tmp_path)
    assert len(report.read_text(encoding="utf-8")) == size_after_first


def test_req_arc_wmte_4775_blocked_paths_and_schema_guards(tmp_path: Path) -> None:
    """SCENARIO-ARC-WMTE-4775-BLOCKED-PRECONDITION: missing sources fail closed."""

    from carnot import experiment_4775_silent_bug_audit as mod

    blocked = mod.run(root=tmp_path, write=True)
    assert blocked["honest_verdict"] == "blocked_missing_source_artifacts"
    assert blocked["nulls_audited"] == 0
    assert blocked["silent_bugs_found"] == []
    assert (tmp_path / mod.RESULT_RELATIVE_PATH).exists()
    assert not (tmp_path / mod.AUDIT_REPORT_RELATIVE_PATH).exists()
    assert mod.artifact_schema_errors(blocked) == []

    invalid = dict(blocked)
    invalid["honest_verdict"] = "not terminal"
    invalid["reproducibility_checksum"] = mod.payload_checksum(invalid)
    assert "honest_verdict_missing_terminal_prefix" in mod.artifact_schema_errors(invalid)


def test_req_arc_wmte_4775_defensive_edges(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """REQ-ARC-WMTE-4775: schema and degenerate-input guards fail closed."""

    from carnot import experiment_4775_silent_bug_audit as mod

    bad_json = tmp_path / "bad.json"
    bad_json.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="must contain a JSON object"):
        mod._read_json(bad_json)
    assert mod._finite_float(True) is None
    assert mod._finite_float("not-a-number") is None
    assert mod._per_game_loo_structural({}) == {}

    missing_balance = mod.audit_null_artifact(
        "experiment_4771_structural_energy_s0prime_origin_matched",
        {
            "n_candidate_rows": 2,
            "n_pos": 1,
            "n_neg": 1,
            "origin_probe": {"origin_counts": {"real": 1, "induced": 1}},
            "origin_probe_auroc": 0.5,
            "controls": {"shuffled_label_resamples": 0},
            "shuffled_label_control_auroc": 0.5,
            "dataset_diagnostics": {"origin_matched": False},
            "corrigendum_pending": [{"kind": "TAUTOLOGY"}],
        },
    )
    assert missing_balance["verdict"] == "silent_bug_must_reopen"
    assert "s0prime_class_balance_degenerate" in missing_balance["silent_bug_signatures"]
    assert "s0prime_origin_matching_not_real" in missing_balance["silent_bug_signatures"]
    assert "s0prime_single_class_games_not_skipped" in missing_balance["silent_bug_signatures"]
    assert "s0prime_corrigendum_pending" in missing_balance["silent_bug_signatures"]

    bad_rows = _s0prime_payload(origin_refit=True)
    bad_rows["per_game_class_balance"]["raw"] = "not-a-mapping"
    bad_rows["per_game_class_balance"]["badskip"] = {
        "correct": 0,
        "wrong": 2,
        "rows": 2,
        "contributes_to_loo": False,
    }
    bad_row_result = mod.audit_null_artifact(
        "experiment_4771_structural_energy_s0prime_origin_matched",
        bad_rows,
        module_source=SHUFFLE_CODE,
    )
    assert "s0prime_class_balance_degenerate" in bad_row_result["silent_bug_signatures"]
    assert "s0prime_single_class_games_not_skipped" in bad_row_result["silent_bug_signatures"]

    no_attempt = mod.audit_null_artifact("experiment_4772_levelup_attempt", {"attempted_games": []})
    no_gate = mod.audit_null_artifact(
        "experiment_4772_levelup_attempt",
        {"attempted_games": [{"offline_reproduced_existing_depth": True, "solution_labels": []}]},
    )
    assert "levelup_attempts_missing" in no_attempt["silent_bug_signatures"]
    assert "reproduction_gate_missing" in no_gate["silent_bug_signatures"]
    assert "levelup_mechanism_not_exercised" in no_gate["silent_bug_signatures"]

    cloned_heldout = _heldout_payload()
    cloned_heldout["arms"] = [
        {"arm": "baseline", "rate": 0.04},
        {"arm": "treatment", "rate": 0.04},
    ]
    cloned = mod.audit_null_artifact("experiment_4774_heldout_first_win_readiness", cloned_heldout)
    heldout_bad = mod.audit_null_artifact(
        "experiment_4774_heldout_first_win_readiness",
        {
            "heldout_first_win_rate": 0.04,
            "first_win_baseline": 0.04,
            "heldout_variant_attempts": 0,
            "positive_control_passed": False,
            "parity_test_green": False,
            "null_delta_methodology_note": "",
        },
    )
    assert "byte_identical_ab_arms" in cloned["silent_bug_signatures"]
    assert "heldout_attempt_floor_not_met" in heldout_bad["silent_bug_signatures"]
    assert "parity_test_not_green" in heldout_bad["silent_bug_signatures"]

    unknown = mod.audit_null_artifact("experiment_unknown", {})
    assert unknown["silent_bug_signatures"] == ["unknown_null_artifact"]

    payloads = {
        "results/experiment_4771_structural_energy_s0prime_origin_matched.json": _s0prime_payload(
            origin_refit=True
        ),
        "results/experiment_4772_levelup_attempt.json": _levelup_payload(),
        "results/experiment_4774_heldout_first_win_readiness.json": _heldout_payload(),
    }
    for rel, payload in payloads.items():
        path = tmp_path / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload), encoding="utf-8")
    no_module_artifact = mod.run(root=tmp_path, write=False)
    assert no_module_artifact["preconditions_checked"]["s0prime_module_present"] is False
    assert no_module_artifact["s0prime_leak_controls_fired"] is False

    artifact = mod.run(root=REPO, write=False)
    invalids: list[dict[str, Any]] = [
        artifact | {"field_principles": {}},
        artifact | {"inference_substrate": "wrong"},
        artifact | {"nulls_audited": "3"},
        artifact | {"s0prime_leak_controls_fired": "false"},
        artifact | {"silent_bugs_found": {}},
        artifact | {"per_null_verdicts": {}},
        artifact | {"verifier_is_oracle": True},
        artifact | {"duration_s": 0.0},
        artifact | {"nulls_audited": 99},
    ]
    for invalid_artifact in invalids:
        invalid_artifact["reproducibility_checksum"] = mod.payload_checksum(invalid_artifact)
        assert mod.artifact_schema_errors(invalid_artifact)

    bad_checksum = dict(artifact)
    bad_checksum["reproducibility_checksum"] = "sha256:bad"
    assert "reproducibility_checksum_mismatch" in mod.artifact_schema_errors(bad_checksum)
    with pytest.raises(ValueError):
        mod.write_artifact(bad_checksum, root=tmp_path)

    rendered = mod.render_markdown_section(artifact | {"per_null_verdicts": [None]})
    assert "Experiment 4775" in rendered

    monkeypatch.setattr(mod, "artifact_schema_errors", lambda _artifact: ["forced"])
    with pytest.raises(ValueError, match="forced"):
        mod.run(root=REPO, write=False)
