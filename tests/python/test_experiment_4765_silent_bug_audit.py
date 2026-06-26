"""Tests for Exp 4765 .438 ARC null silent-bug audit.

Spec refs: REQ-ARC-WMTE-4765, SCENARIO-ARC-WMTE-4765-SILENT-BUG-AUDIT,
SCENARIO-ARC-WMTE-4765-BLOCKED-PRECONDITION.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec/capabilities/arc-world-model-trust-energy/spec.md"


def _spec_section() -> str:
    spec = SPEC_PATH.read_text(encoding="utf-8")
    start = spec.index("### REQ-ARC-WMTE-4765")
    end = spec.index("### REQ-ARC-WMTE-4751", start)
    return spec[start:end]


def _s0_payload(*, origin_probe_auroc: float | None = 0.5) -> dict[str, Any]:
    return {
        "honest_verdict": "complete: structural_energy_s0_honest_null",
        "preconditions_checked": {
            "ok": True,
            "candidate_rows": 12,
            "origin_probe_rows": 24,
        },
        "n_candidate_rows": 12,
        "n_origin_probe_rows": 24,
        "origin_probe_auroc": origin_probe_auroc,
        "near_miss_negative_fraction": 1.0,
        "in_sample_auroc": 0.8,
        "dataset_diagnostics": {
            "ground_truth_corruptions": 0,
            "feature_families_used": ["object_relational", "frame_delta"],
        },
    }


def _levelup_payload() -> dict[str, Any]:
    return {
        "honest_verdict": "complete_re86_no_new_level_residual_existing_depth",
        "attempted_games": [
            {
                "game": "re86",
                "prior_level": 2,
                "reached_level": 2,
                "offline_reproduced_existing_depth": True,
                "offline_reproduced_new_depth": False,
                "solution_labels": ['{"action":1}'],
                "reproduction_gate": {"reproduced": True, "reached_level": 2},
                "residual_cause": "reproduced_existing_or_lower_level",
            },
            {
                "game": "dc22",
                "prior_level": 2,
                "reached_level": 0,
                "solution_labels": [],
                "elapsed_s": 115.0,
                "reproduction_gate": {},
                "residual_cause": "time_budget_no_terminal_gate",
            },
        ],
        "dead_ends": ["re86 same-depth", "dc22 timed no-gate"],
        "new_levels_banked": 0,
    }


def _heldout_payload(*, annotate_flat: bool = True) -> dict[str, Any]:
    return {
        "honest_verdict": "complete: heldout_first_win_flat_genuine_null",
        "heldout_first_win_rate": 0.04,
        "first_win_baseline": 0.04,
        "heldout_first_win_delta_vs_baseline": 0.0,
        "heldout_variant_attempts": 100,
        "positive_control_passed": annotate_flat,
        "parity_test_green": True,
        "null_delta_methodology_note": (
            "Held-out first-win rate equals the 0.04 baseline; genuine no-improvement."
            if annotate_flat
            else ""
        ),
    }


def test_req_arc_wmte_4765_spec_declares_audit_contract() -> None:
    """REQ-ARC-WMTE-4765: OpenSpec declares the audit and required principles."""

    from carnot import experiment_4765_silent_bug_audit as mod

    section = _spec_section()

    assert "REQ-ARC-WMTE-4765" in section
    assert "SCENARIO-ARC-WMTE-4765-SILENT-BUG-AUDIT" in section
    assert "SCENARIO-ARC-WMTE-4765-BLOCKED-PRECONDITION" in section
    assert mod.RESULT_RELATIVE_PATH in section
    assert mod.AUDIT_REPORT_RELATIVE_PATH in section
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert field in section
        assert principle["principle"] in section


def test_scenario_arc_wmte_4765_flags_s0_leak_and_tautology() -> None:
    """SCENARIO-ARC-WMTE-4765-SILENT-BUG-AUDIT: leaked or tautological nulls reopen."""

    from carnot import experiment_4765_silent_bug_audit as mod

    leaked = mod.audit_null_artifact(
        "experiment_4761_structural_energy_s0_core_bet_probe",
        _s0_payload(origin_probe_auroc=0.733),
    )
    no_probe = mod.audit_null_artifact(
        "experiment_4761_structural_energy_s0_core_bet_probe",
        _s0_payload(origin_probe_auroc=None),
    )
    tautology = mod.audit_null_artifact(
        "experiment_4764_heldout_first_win_readiness",
        _heldout_payload(annotate_flat=False),
    )

    assert leaked["verdict"] == "silent_bug_must_reopen"
    assert "s0_origin_probe_leak" in leaked["silent_bug_signatures"]
    assert "origin_probe_auroc=0.733" in leaked["exercise_evidence"]
    assert no_probe["verdict"] == "silent_bug_must_reopen"
    assert "s0_origin_probe_not_run" in no_probe["silent_bug_signatures"]
    assert tautology["verdict"] == "silent_bug_must_reopen"
    assert "first_win_0_04_tautology_unannotated" in tautology["silent_bug_signatures"]
    assert "first_win_positive_control_missing" in tautology["silent_bug_signatures"]


def test_scenario_arc_wmte_4765_detects_zero_cell_and_byte_identical_arms() -> None:
    """SCENARIO-ARC-WMTE-4765-SILENT-BUG-AUDIT: dead engines and cloned arms reopen."""

    from carnot import experiment_4765_silent_bug_audit as mod

    levelup = _levelup_payload()
    levelup["attempted_games"][0]["engine_cell_changes"] = 0
    heldout = _heldout_payload()
    heldout["arms"] = [
        {"arm": "baseline", "first_win": False, "depth": 0},
        {"arm": "treatment", "first_win": False, "depth": 0},
    ]

    zero_cell = mod.audit_null_artifact("experiment_4762_levelup_attempt", levelup)
    cloned = mod.audit_null_artifact("experiment_4764_heldout_first_win_readiness", heldout)

    assert zero_cell["verdict"] == "silent_bug_must_reopen"
    assert "dead_identity_engine_zero_cell_changes" in zero_cell["silent_bug_signatures"]
    assert cloned["verdict"] == "silent_bug_must_reopen"
    assert "byte_identical_ab_arms" in cloned["silent_bug_signatures"]


def test_scenario_arc_wmte_4765_keeps_exercised_nulls_trusted() -> None:
    """SCENARIO-ARC-WMTE-4765-SILENT-BUG-AUDIT: non-degenerate evidence stays trusted."""

    from carnot import experiment_4765_silent_bug_audit as mod

    s0 = mod.audit_null_artifact(
        "experiment_4761_structural_energy_s0_core_bet_probe",
        _s0_payload(origin_probe_auroc=0.5),
    )
    levelup = mod.audit_null_artifact("experiment_4762_levelup_attempt", _levelup_payload())
    heldout = mod.audit_null_artifact(
        "experiment_4764_heldout_first_win_readiness",
        _heldout_payload(annotate_flat=True),
    )

    assert s0["verdict"] == "trustworthy_null"
    assert "s0_candidate_rows=12" in s0["exercise_evidence"]
    assert levelup["verdict"] == "trustworthy_null"
    assert "levelup_attempts=2" in levelup["exercise_evidence"]
    assert heldout["verdict"] == "trustworthy_null"
    assert "heldout_first_win_rate=0.04" in heldout["exercise_evidence"]


def test_req_arc_wmte_4765_run_checked_in_artifacts() -> None:
    """REQ-ARC-WMTE-4765: checked-in .438 artifacts produce the expected audit."""

    from carnot import experiment_4765_silent_bug_audit as mod

    artifact = mod.run(root=REPO, write=False)

    assert mod.artifact_schema_errors(artifact) == []
    assert artifact["honest_verdict"] == "complete_arc_null_silent_bug_audit_3_nulls_1_reopen"
    assert artifact["nulls_audited"] == 3
    assert artifact["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert artifact["preconditions_checked"]["source_artifacts_present"] is True
    assert artifact["reproducibility_checksum"] == mod.payload_checksum(artifact)
    assert [row["null_id"] for row in artifact["silent_bugs_found"]] == [
        "experiment_4761_structural_energy_s0_core_bet_probe"
    ]
    assert set(artifact["trusted_nulls"]) == {
        "experiment_4762_levelup_attempt",
        "experiment_4764_heldout_first_win_readiness",
    }


def test_req_arc_wmte_4765_write_artifact_and_append_markdown(tmp_path: Path) -> None:
    """REQ-ARC-WMTE-4765: complete audits write JSON and append the ops report."""

    from carnot import experiment_4765_silent_bug_audit as mod

    payloads = {
        "results/experiment_4761_structural_energy_s0_core_bet_probe.json": _s0_payload(
            origin_probe_auroc=0.61
        ),
        "results/experiment_4762_levelup_attempt.json": _levelup_payload(),
        "results/experiment_4764_heldout_first_win_readiness.json": _heldout_payload(),
        "results/experiment_4725_silent_bug_audit.json": {
            "nulls_audited": 12,
            "silent_bug_nulls": [{"null_id": "prior"}],
        },
        "results/experiment_4755_silent_bug_audit.json": {
            "must_reopen": ["prior"],
            "silent_no_op_findings": [],
        },
    }
    for rel, payload in payloads.items():
        path = tmp_path / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload), encoding="utf-8")
    report = tmp_path / mod.AUDIT_REPORT_RELATIVE_PATH
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text("# ARC Null Silent-Bug Audit\n", encoding="utf-8")

    artifact = mod.run(root=tmp_path, write=True)

    loaded = json.loads((tmp_path / mod.RESULT_RELATIVE_PATH).read_text(encoding="utf-8"))
    assert loaded == artifact
    text = report.read_text(encoding="utf-8")
    assert "## Experiment 4765 .438 ARC Null Silent-Bug Audit" in text
    assert "`experiment_4761_structural_energy_s0_core_bet_probe`" in text
    size_after_first = len(text)
    mod.append_markdown_report(artifact, root=tmp_path)
    assert len(report.read_text(encoding="utf-8")) == size_after_first


def test_req_arc_wmte_4765_blocked_paths_and_schema_guards(tmp_path: Path) -> None:
    """SCENARIO-ARC-WMTE-4765-BLOCKED-PRECONDITION: missing sources fail closed."""

    from carnot import experiment_4765_silent_bug_audit as mod

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

    invalid = dict(blocked)
    invalid["field_principles"] = {}
    invalid["reproducibility_checksum"] = mod.payload_checksum(invalid)
    assert "field_principles_mismatch" in mod.artifact_schema_errors(invalid)

    invalid = dict(blocked)
    invalid["inference_substrate"] = "live_llm_inference"
    invalid["reproducibility_checksum"] = mod.payload_checksum(invalid)
    assert "inference_substrate_mismatch" in mod.artifact_schema_errors(invalid)

    invalid = dict(blocked)
    invalid["nulls_audited"] = "3"
    invalid["reproducibility_checksum"] = mod.payload_checksum(invalid)
    assert "nulls_audited_must_be_int" in mod.artifact_schema_errors(invalid)

    invalid = dict(blocked)
    invalid["silent_bugs_found"] = {}
    invalid["reproducibility_checksum"] = mod.payload_checksum(invalid)
    assert "silent_bugs_found_must_be_list" in mod.artifact_schema_errors(invalid)

    invalid = dict(blocked)
    invalid["duration_s"] = 0.0
    invalid["reproducibility_checksum"] = mod.payload_checksum(invalid)
    assert "duration_below_aggregation_floor" in mod.artifact_schema_errors(invalid)

    with pytest.raises(ValueError, match="honest_verdict_missing_terminal_prefix"):
        mod.write_artifact({**blocked, "honest_verdict": "bad"}, root=tmp_path)


def test_req_arc_wmte_4765_defensive_helpers_are_covered(tmp_path: Path) -> None:
    """REQ-ARC-WMTE-4765: malformed inputs are explicit, not silently trusted."""

    from carnot import experiment_4765_silent_bug_audit as mod

    bad_json = tmp_path / "bad.json"
    bad_json.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="must contain a JSON object"):
        mod._read_json(bad_json)

    assert mod._finite_float(True) is None
    assert mod._finite_float("not-float") is None
    assert mod._finite_float(float("nan")) is None
    assert mod._int_value("bad") == 0
    assert mod._list_of_mappings({"not": "list"}) == []

    unknown = mod.audit_null_artifact("unknown", {})
    assert unknown["verdict"] == "silent_bug_must_reopen"
    assert "unknown_null_artifact" in unknown["silent_bug_signatures"]

    malformed_s0 = mod.audit_null_artifact(
        "experiment_4761_structural_energy_s0_core_bet_probe",
        {"origin_probe_auroc": 0.4},
    )
    assert "s0_candidate_rows_missing" in malformed_s0["silent_bug_signatures"]
    assert "s0_near_miss_negatives_missing" in malformed_s0["silent_bug_signatures"]
    assert "s0_positive_control_not_exercised" in malformed_s0["silent_bug_signatures"]

    malformed_levelup = mod.audit_null_artifact("experiment_4762_levelup_attempt", {})
    assert "levelup_attempts_missing" in malformed_levelup["silent_bug_signatures"]

    bad_gate = mod.audit_null_artifact(
        "experiment_4762_levelup_attempt",
        {
            "attempted_games": [
                {
                    "offline_reproduced_existing_depth": True,
                    "solution_labels": [],
                    "reproduction_gate": {},
                }
            ]
        },
    )
    assert "reproduction_gate_missing" in bad_gate["silent_bug_signatures"]
    assert "levelup_mechanism_not_exercised" in bad_gate["silent_bug_signatures"]

    malformed_heldout = mod.audit_null_artifact(
        "experiment_4764_heldout_first_win_readiness",
        {"heldout_first_win_rate": 0.04, "first_win_baseline": 0.04},
    )
    assert "heldout_attempt_floor_not_met" in malformed_heldout["silent_bug_signatures"]

    artifact = mod.build_artifact(
        preconditions_checked={"ok": True},
        per_null_verdicts=[
            {
                "null_id": "n",
                "verdict": "trustworthy_null",
                "silent_bug_signatures": [],
                "exercise_evidence": [],
            }
        ],
        audited_artifact_checksums={},
        prior_audit_context={},
        duration_s=0.001,
    )
    bad_shape = dict(artifact)
    bad_shape["per_null_verdicts"] = {}
    bad_shape["reproducibility_checksum"] = mod.payload_checksum(bad_shape)
    assert "per_null_verdicts_must_be_list" in mod.artifact_schema_errors(bad_shape)

    bad_oracle = dict(artifact)
    bad_oracle["verifier_is_oracle"] = True
    bad_oracle["reproducibility_checksum"] = mod.payload_checksum(bad_oracle)
    assert "verifier_is_oracle_must_be_false" in mod.artifact_schema_errors(bad_oracle)

    mismatch = dict(artifact)
    mismatch["nulls_audited"] = 2
    mismatch["reproducibility_checksum"] = mod.payload_checksum(mismatch)
    assert "nulls_audited_does_not_match_verdicts" in mod.artifact_schema_errors(mismatch)

    rendered = mod.render_markdown_section({**artifact, "per_null_verdicts": [None]})
    assert "## Experiment 4765 .438 ARC Null Silent-Bug Audit" in rendered

    original_schema = mod.artifact_schema_errors
    try:
        mod.artifact_schema_errors = lambda _artifact: ["synthetic_schema_error"]  # type: ignore[method-assign]
        with pytest.raises(ValueError, match="synthetic_schema_error"):
            mod.run(root=REPO, write=False)
    finally:
        mod.artifact_schema_errors = original_schema  # type: ignore[method-assign]
