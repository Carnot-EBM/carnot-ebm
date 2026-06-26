"""Tests for Exp 4785 .440 ARC null silent-bug audit.

Spec refs: REQ-ARC-WMTE-4785, SCENARIO-ARC-WMTE-4785-SILENT-BUG-AUDIT,
SCENARIO-ARC-WMTE-4785-BLOCKED-PRECONDITION.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec/capabilities/arc-world-model-trust-energy/spec.md"
S1_SOURCE_EXECUTES_CONTROLS = "\n".join(
    [
        "origin_probe_refit_on_origin_matched_data = True",
        "shuffled = rng.permutation(labels)",
        "_loo_energy_metrics(shuffled_rows, 'structural')",
        "def denoising_direction_agreement(rows):",
        "    midpoint_energy = model.energy(midpoint)",
        "_denoising_direction_mean(rows)",
    ]
)
S1_SOURCE_HARDCODED = "denoising_direction_agreement = 0.6223390275952694"


def _spec_section() -> str:
    spec = SPEC_PATH.read_text(encoding="utf-8")
    start = spec.index("### REQ-ARC-WMTE-4785")
    end = spec.index("### REQ-ARC-WMTE-4731", start)
    return spec[start:end]


def _s1_payload(
    *,
    origin_refit: bool = True,
    denoising: float | None = 0.72,
    seeds: list[int] | None = None,
    shuffled_resamples: int = 16,
) -> dict[str, Any]:
    used_seeds = seeds or list(range(4781, 4791))
    origin_probe = {
        "loo_auroc": 0.5,
        "origin_counts": {"induced": 40},
        "status": "origin_matched_refit_complete",
    }
    if origin_refit:
        origin_probe["refit_on_origin_matched_data"] = True
    else:
        origin_probe["status"] = "origin_matched_single_origin_all_induced"
    return {
        "experiment": "experiment_4781_structural_energy_s1_contrastive_landscape",
        "honest_verdict": "success_structural_energy_s1_landscape_authorizes_s2",
        "n_candidate_rows": 40,
        "n_pos": 20,
        "n_neg": 20,
        "n_seeds": len(used_seeds),
        "random_seeds_used": used_seeds,
        "energy_ranking_loo_auroc_per_seed": [0.71 + (idx * 0.001) for idx, _ in enumerate(used_seeds)],
        "denoising_direction_agreement": denoising,
        "origin_probe_auroc": 0.5,
        "origin_probe": origin_probe,
        "shuffled_label_control_auroc": 0.493,
        "controls": {"shuffled_label_resamples": shuffled_resamples},
        "dataset_diagnostics": {
            "origin_matched": True,
            "denoising_direction_method": (
                "same-heldout-game wrong->correct feature-space midpoint must lower linear energy"
            ),
            "feature_families_used": ["object_relational", "frame_delta"],
            "feature_families_excluded": ["v2"],
        },
        "per_family_loo": {"object_relational": 0.66, "frame_delta": 0.72},
        "in_sample_auroc": 0.84,
    }


def _levelup_payload() -> dict[str, Any]:
    return {
        "experiment": "experiment_4782_levelup_attempt",
        "honest_verdict": "complete_lf52_no_new_level_residual_existing_depth",
        "attempted_games": [
            {
                "game": "lf52",
                "prior_level": 2,
                "reached_level": 2,
                "target_level": 3,
                "offline_reproduced_existing_depth": True,
                "offline_reproduced_new_depth": False,
                "solution_labels": ["{\"action\":6}", "{\"action\":4}"],
                "reproduction_gate": {"reproduced": True, "reached_level": 2},
            }
        ],
        "new_levels_banked": 0,
        "offline_reproduced": False,
        "reproduced_levels": 0,
    }


def _heldout_payload(*, annotate_flat: bool = True) -> dict[str, Any]:
    return {
        "experiment": "experiment_4784_heldout_first_win_readiness",
        "honest_verdict": "complete: heldout_first_win_flat_genuine_null",
        "heldout_first_win_rate": 0.04,
        "first_win_baseline": 0.04,
        "heldout_first_win_delta_vs_baseline": 0.0,
        "heldout_variant_attempts": 100,
        "positive_control_passed": annotate_flat,
        "parity_test_green": True,
        "null_delta_methodology_note": "flat 0.04 genuine null" if annotate_flat else "",
    }


def test_req_arc_wmte_4785_spec_declares_audit_contract() -> None:
    """REQ-ARC-WMTE-4785: OpenSpec declares the .440 audit and principles."""

    from carnot import experiment_4785_silent_bug_audit as mod

    section = _spec_section()

    assert "REQ-ARC-WMTE-4785" in section
    assert "SCENARIO-ARC-WMTE-4785-SILENT-BUG-AUDIT" in section
    assert "SCENARIO-ARC-WMTE-4785-BLOCKED-PRECONDITION" in section
    assert mod.RESULT_RELATIVE_PATH in section
    assert mod.AUDIT_REPORT_RELATIVE_PATH in section
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert field in section
        assert principle["principle"] in section


def test_scenario_arc_wmte_4785_s1_controls_must_genuinely_fire() -> None:
    """SCENARIO-ARC-WMTE-4785-SILENT-BUG-AUDIT: S1 needs real leak and denoising controls."""

    from carnot import experiment_4785_silent_bug_audit as mod

    good = mod.audit_null_artifact(
        "experiment_4781_structural_energy_s1_contrastive_landscape",
        _s1_payload(origin_refit=True),
        s1_module_source=S1_SOURCE_EXECUTES_CONTROLS,
    )
    stale_origin = mod.audit_null_artifact(
        "experiment_4781_structural_energy_s1_contrastive_landscape",
        _s1_payload(origin_refit=False),
        s1_module_source=S1_SOURCE_EXECUTES_CONTROLS,
    )
    hardcoded_denoise = mod.audit_null_artifact(
        "experiment_4781_structural_energy_s1_contrastive_landscape",
        _s1_payload(origin_refit=True, denoising=0.6223390275952694),
        s1_module_source=S1_SOURCE_HARDCODED,
    )
    duplicate_seeds = mod.audit_null_artifact(
        "experiment_4781_structural_energy_s1_contrastive_landscape",
        _s1_payload(origin_refit=True, seeds=[4781] * 10),
        s1_module_source=S1_SOURCE_EXECUTES_CONTROLS,
    )

    assert good["verdict"] == "trustworthy_null"
    assert good["s1_controls_fired"] is True
    assert good["s1_control_checks"]["origin_probe_refit_on_origin_matched_data"] is True
    assert good["s1_control_checks"]["shuffled_label_permuted_and_reran_loo"] is True
    assert good["s1_control_checks"]["denoising_direction_executed"] is True
    assert good["s1_control_checks"]["distinct_seed_count"] == 10

    assert stale_origin["verdict"] == "silent_bug_must_reopen"
    assert stale_origin["s1_controls_fired"] is False
    assert "s1_origin_probe_not_refit" in stale_origin["silent_bug_signatures"]
    assert "origin_probe_status=origin_matched_single_origin_all_induced" in stale_origin["exercise_evidence"]

    assert hardcoded_denoise["verdict"] == "silent_bug_must_reopen"
    assert "s1_denoising_direction_not_executed" in hardcoded_denoise["silent_bug_signatures"]
    assert duplicate_seeds["verdict"] == "silent_bug_must_reopen"
    assert "s1_random_seeds_not_distinct" in duplicate_seeds["silent_bug_signatures"]


def test_scenario_arc_wmte_4785_detects_dead_arms_and_tautology_edges() -> None:
    """SCENARIO-ARC-WMTE-4785-SILENT-BUG-AUDIT: generic silent-bug signatures reopen."""

    from carnot import experiment_4785_silent_bug_audit as mod

    s1 = _s1_payload(origin_refit=True)
    s1["arms"] = [{"arm": "baseline", "score": 1.0}, {"arm": "treatment", "score": 1.0}]
    s1["representation_delta_l1"] = 0
    levelup = _levelup_payload()
    levelup["attempted_games"][0]["engine_cell_changes"] = 0

    cloned = mod.audit_null_artifact(
        "experiment_4781_structural_energy_s1_contrastive_landscape",
        s1,
        s1_module_source=S1_SOURCE_EXECUTES_CONTROLS,
    )
    dead_engine = mod.audit_null_artifact("experiment_4782_levelup_attempt", levelup)
    tautology = mod.audit_null_artifact(
        "experiment_4784_heldout_first_win_readiness",
        _heldout_payload(annotate_flat=False),
    )

    assert "byte_identical_ab_arms" in cloned["silent_bug_signatures"]
    assert "representation_noop_zero_delta" in cloned["silent_bug_signatures"]
    assert "dead_identity_engine_zero_cell_changes" in dead_engine["silent_bug_signatures"]
    assert "first_win_0_04_tautology_unannotated" in tautology["silent_bug_signatures"]


def test_req_arc_wmte_4785_run_checked_in_artifacts() -> None:
    """REQ-ARC-WMTE-4785: checked-in .440 artifacts produce the expected audit."""

    from carnot import experiment_4785_silent_bug_audit as mod

    artifact = mod.run(root=REPO, write=False)

    assert mod.artifact_schema_errors(artifact) == []
    assert artifact["honest_verdict"] == "complete_arc_null_silent_bug_audit_3_nulls_1_reopen"
    assert artifact["nulls_audited"] == 3
    assert artifact["s1_controls_fired"] is False
    assert artifact["s1_control_checks"]["origin_probe_refit_on_origin_matched_data"] is False
    assert artifact["s1_control_checks"]["shuffled_label_permuted_and_reran_loo"] is True
    assert artifact["s1_control_checks"]["denoising_direction_executed"] is True
    assert artifact["s1_control_checks"]["distinct_seed_count"] == 10
    assert artifact["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert artifact["preconditions_checked"]["milestone_440_artifacts_present"] is True
    assert artifact["reproducibility_checksum"] == mod.payload_checksum(artifact)
    assert [row["null_id"] for row in artifact["silent_bugs_found"]] == [
        "experiment_4781_structural_energy_s1_contrastive_landscape"
    ]
    assert set(artifact["trusted_nulls"]) == {
        "experiment_4782_levelup_attempt",
        "experiment_4784_heldout_first_win_readiness",
    }


def test_req_arc_wmte_4785_write_artifact_and_append_markdown(tmp_path: Path) -> None:
    """REQ-ARC-WMTE-4785: complete audits write JSON and append the ops report."""

    from carnot import experiment_4785_silent_bug_audit as mod

    payloads = {
        "results/experiment_4781_structural_energy_s1_contrastive_landscape.json": _s1_payload(
            origin_refit=False
        ),
        "results/experiment_4782_levelup_attempt.json": _levelup_payload(),
        "results/experiment_4784_heldout_first_win_readiness.json": _heldout_payload(),
    }
    for rel, payload in payloads.items():
        path = tmp_path / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload), encoding="utf-8")
    source_path = tmp_path / mod.S1_MODULE_RELATIVE_PATH
    source_path.parent.mkdir(parents=True, exist_ok=True)
    source_path.write_text(S1_SOURCE_EXECUTES_CONTROLS, encoding="utf-8")
    report = tmp_path / mod.AUDIT_REPORT_RELATIVE_PATH
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text("# ARC Null Silent-Bug Audit\n", encoding="utf-8")

    artifact = mod.run(root=tmp_path, write=True)

    loaded = json.loads((tmp_path / mod.RESULT_RELATIVE_PATH).read_text(encoding="utf-8"))
    assert loaded == artifact
    text = report.read_text(encoding="utf-8")
    assert "## Experiment 4785 .440 ARC Null Silent-Bug Audit" in text
    assert "`experiment_4781_structural_energy_s1_contrastive_landscape`" in text
    size_after_first = len(text)
    mod.append_markdown_report(artifact, root=tmp_path)
    assert len(report.read_text(encoding="utf-8")) == size_after_first


def test_req_arc_wmte_4785_blocked_paths_and_schema_guards(tmp_path: Path) -> None:
    """SCENARIO-ARC-WMTE-4785-BLOCKED-PRECONDITION: missing sources fail closed."""

    from carnot import experiment_4785_silent_bug_audit as mod

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


def test_req_arc_wmte_4785_defensive_edges(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """REQ-ARC-WMTE-4785: schema and degenerate-input guards fail closed."""

    from carnot import experiment_4785_silent_bug_audit as mod

    bad_json = tmp_path / "bad.json"
    bad_json.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="must contain a JSON object"):
        mod._read_json(bad_json)
    assert mod._finite_float(True) is None
    assert mod._finite_float("not-a-number") is None

    bad_s1 = _s1_payload(origin_refit=True, denoising=None, seeds=list(range(4781, 4789)), shuffled_resamples=0)
    bad_s1["n_candidate_rows"] = 0
    bad_s1["n_pos"] = 0
    bad_s1["n_neg"] = 0
    bad_s1["origin_probe_auroc"] = 0.7
    bad_result = mod.audit_null_artifact(
        "experiment_4781_structural_energy_s1_contrastive_landscape",
        bad_s1,
        s1_module_source="",
    )
    assert "s1_candidate_rows_missing" in bad_result["silent_bug_signatures"]
    assert "s1_class_balance_degenerate" in bad_result["silent_bug_signatures"]
    assert "s1_origin_probe_leak_or_missing" in bad_result["silent_bug_signatures"]
    assert "s1_shuffled_label_control_not_permuted_loo" in bad_result["silent_bug_signatures"]
    assert "s1_denoising_direction_not_executed" in bad_result["silent_bug_signatures"]
    assert "s1_random_seed_floor_not_met" in bad_result["silent_bug_signatures"]

    no_attempt = mod.audit_null_artifact("experiment_4782_levelup_attempt", {"attempted_games": []})
    no_gate = mod.audit_null_artifact(
        "experiment_4782_levelup_attempt",
        {"attempted_games": [{"offline_reproduced_existing_depth": True, "solution_labels": []}]},
    )
    assert "levelup_attempts_missing" in no_attempt["silent_bug_signatures"]
    assert "reproduction_gate_missing" in no_gate["silent_bug_signatures"]
    assert "levelup_mechanism_not_exercised" in no_gate["silent_bug_signatures"]

    heldout_bad = mod.audit_null_artifact(
        "experiment_4784_heldout_first_win_readiness",
        {
            "heldout_first_win_rate": 0.04,
            "first_win_baseline": 0.04,
            "heldout_variant_attempts": 0,
            "positive_control_passed": False,
            "parity_test_green": False,
            "null_delta_methodology_note": "",
        },
    )
    assert "heldout_attempt_floor_not_met" in heldout_bad["silent_bug_signatures"]
    assert "parity_test_not_green" in heldout_bad["silent_bug_signatures"]

    unknown = mod.audit_null_artifact("experiment_unknown", {})
    assert unknown["silent_bug_signatures"] == ["unknown_null_artifact"]

    payloads = {
        "results/experiment_4781_structural_energy_s1_contrastive_landscape.json": _s1_payload(
            origin_refit=True
        ),
        "results/experiment_4782_levelup_attempt.json": _levelup_payload(),
        "results/experiment_4784_heldout_first_win_readiness.json": _heldout_payload(),
    }
    for rel, payload in payloads.items():
        path = tmp_path / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload), encoding="utf-8")
    no_source_artifact = mod.run(root=tmp_path, write=False)
    assert no_source_artifact["preconditions_checked"]["s1_module_present"] is False
    assert no_source_artifact["s1_controls_fired"] is False

    artifact = mod.run(root=REPO, write=False)
    invalids: list[dict[str, Any]] = [
        artifact | {"field_principles": {}},
        artifact | {"inference_substrate": "wrong"},
        artifact | {"nulls_audited": "3"},
        artifact | {"s1_controls_fired": "false"},
        artifact | {"silent_bugs_found": {}},
        artifact | {"per_null_verdicts": {}},
        artifact | {"s1_control_checks": []},
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
    assert "Experiment 4785" in rendered

    monkeypatch.setattr(mod, "artifact_schema_errors", lambda _artifact: ["forced"])
    with pytest.raises(ValueError, match="forced"):
        mod.run(root=REPO, write=False)
