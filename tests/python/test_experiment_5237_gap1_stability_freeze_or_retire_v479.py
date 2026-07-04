"""Tests for Exp 5237's GAP-1 stability freeze-or-retire decision.

Spec refs: REQ-VERIFY-5237, SCENARIO-VERIFY-5237.
"""

from __future__ import annotations

import json
from pathlib import Path

from carnot.verify import arc_gap1_stability_freeze_or_retire as mod


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "verification" / "spec.md"


def _field(value: object, principle: str = "fixture principle") -> dict[str, object]:
    return {"value": value, "principle": principle}


def _selection_counts(
    top_subset: list[str],
    *,
    top_count: int,
) -> list[dict[str, object]]:
    if top_subset == [mod.REFUTED_SINGLE_INVARIANT]:
        return [{"subset": top_subset, "count": top_count}]
    return [
        {"subset": top_subset, "count": top_count},
        {
            "subset": [
                "row_ordered_edge_profile",
                "column_ordered_edge_profile",
                "diagonal_adjacency_asymmetry",
                "row_column_run_profile",
            ],
            "count": 4,
        },
        {"subset": ["column_ordered_edge_profile", "row_column_run_profile"], "count": 2},
        {
            "subset": [
                "column_ordered_edge_profile",
                "color_centroid_orientation",
                "row_column_run_profile",
            ],
            "count": 1,
        },
        {
            "subset": [
                mod.REFUTED_SINGLE_INVARIANT,
                "diagonal_adjacency_asymmetry",
                "border_ordered_profile",
                "color_centroid_orientation",
                "row_column_run_profile",
            ],
            "count": 1,
        },
        {
            "subset": [
                "row_ordered_edge_profile",
                "diagonal_adjacency_asymmetry",
                "color_centroid_orientation",
                "row_column_run_profile",
            ],
            "count": 1,
        },
        {
            "subset": [
                "row_ordered_edge_profile",
                "diagonal_adjacency_asymmetry",
                "row_column_run_profile",
            ],
            "count": 1,
        },
        {"subset": ["row_ordered_edge_profile", "row_column_run_profile"], "count": 1},
    ]


def _exp5209_fixture(
    *,
    gate: bool = True,
    stable: bool = False,
    leakage: bool = True,
    no_heldout_selection: bool = True,
    top_subset: list[str] | None = None,
) -> dict[str, object]:
    top = top_subset or ["color_centroid_orientation", "row_column_run_profile"]
    top_count = 11 if stable else 9
    selection_counts = _selection_counts(top, top_count=top_count)
    n_splits = sum(int(row["count"]) for row in selection_counts)
    return {
        "experiment": "experiment_5209_gap1_set_search_holdout_hardening_v477",
        "result_path": mod.EXP5209_RELATIVE_PATH,
        "gap1_hardened_positive": _field(gate),
        "leakage_audit_passed": _field(leakage),
        "best_subset_stable": _field(stable),
        "n_grouped_splits": _field(n_splits),
        "heldout_pass_at_2_mean": _field(0.189584),
        "baseline_always_on_pass_at_2_mean": _field(0.088976),
        "single_refuted_directional_pass_at_2_mean": _field(0.147787),
        "paired_delta_ci95": _field("[0.023148, 0.060446]"),
        "subset_stability": {
            "selection_counts": selection_counts,
            "stability_rule": "one exact subset selected in at least half of grouped splits",
            "top_subset": top,
            "top_subset_count": top_count,
            "top_subset_fraction": round(top_count / n_splits, 6),
        },
        "leakage_audit": {
            "passed": leakage,
            "errors": [] if leakage else ["fixture leakage"],
            "no_duplicate_task_ids_across_train_eval": leakage,
            "no_subset_selection_on_heldout_rows": no_heldout_selection,
            "no_test_gold_in_scoring": leakage,
            "no_test_output_derived_features": leakage,
        },
        "baseline_definitions": {
            "always_on": ["object_count", "palette_histogram_shape"],
            "selected_subset_rule": (
                "Within each split, maximize train pass@2, then train transpose captures, then "
                "smaller subset; held-out rows are used only for final metrics."
            ),
            "single_refuted_directional": [mod.REFUTED_SINGLE_INVARIANT],
        },
    }


def _exp5222_fixture(
    *,
    promoted: bool = False,
    decision: str = "blocked_instability",
) -> dict[str, object]:
    return {
        "experiment": "experiment_5222_gap1_gate_field_registry_promotion_v478",
        "result_path": mod.EXP5222_RELATIVE_PATH,
        "gap1_registry_promoted": _field(promoted),
        "gap1_registry_decision": _field(decision),
        "frozen_subset": _field(None),
        "exp5209_gate_parsed_from_value": _field(True),
        "refuted_single_invariant_excluded": _field(True),
    }


def _write_fixture_repo(
    root: Path,
    exp5209: dict[str, object],
    exp5222: dict[str, object] | None = None,
) -> None:
    (root / "results").mkdir(parents=True, exist_ok=True)
    (root / mod.EXP5209_RELATIVE_PATH).write_text(
        json.dumps(exp5209, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (root / mod.EXP5222_RELATIVE_PATH).write_text(
        json.dumps(exp5222 or _exp5222_fixture(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (root / "ops").mkdir(parents=True, exist_ok=True)
    (root / mod.VERIFIER_GAPS_RELATIVE_PATH).write_text(
        "### GAP-1: transpose / orientation discrimination\n"
        "- status: open -- exp5209 positive but unstable\n"
        "<!-- experiment_5209_gap1_set_search_holdout_hardening_v477 start -->\n"
        "- exp5209 line\n"
        "<!-- experiment_5209_gap1_set_search_holdout_hardening_v477 end -->\n"
        "<!-- experiment_5222_gap1_gate_field_registry_promotion_v478 start -->\n"
        "- exp5222 line\n"
        "<!-- experiment_5222_gap1_gate_field_registry_promotion_v478 end -->\n"
        "\n### GAP-2: next\n",
        encoding="utf-8",
    )
    (root / mod.VERIFIER_REGISTRY_RELATIVE_PATH).write_text(
        "verifiers:\n- verifier_id: arc_grid_combined_verifier\n  domain: arc_agi1_grid\n",
        encoding="utf-8",
    )


def test_req_verify_5237_spec_declares_stability_decision_contract() -> None:
    """REQ-VERIFY-5237: OpenSpec declares the stability artifact and rule."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    section_start = spec.index("### REQ-VERIFY-5237")
    section_end = spec.index("## Implementation Status", section_start)
    section = spec[section_start:section_end]

    for marker in (
        "REQ-VERIFY-5237",
        "SCENARIO-VERIFY-5237",
        mod.RESULT_RELATIVE_PATH,
        "gap1_stability_decision",
        "gap1_registry_promoted",
        "stability_rule_predeclared",
        "no_new_broad_search",
        "directional_adjacency_refuted_20260609",
        "deterministic_gap1_stability_analysis",
        "blocked_instability",
    ):
        assert marker in section


def test_scenario_verify_5237_unstable_exp5209_blocks_current_promotion_path(
    tmp_path: Path,
) -> None:
    """SCENARIO-VERIFY-5237: positive Exp 5209 evidence remains blocked if unstable."""

    _write_fixture_repo(tmp_path, _exp5209_fixture())
    tests_run = [
        {
            "command": ".venv/bin/pytest tests/python/test_experiment_5237... -q",
            "passed": True,
        }
    ]

    artifact = mod.run(
        root=tmp_path,
        result_path=tmp_path / mod.RESULT_RELATIVE_PATH,
        update_gap_doc=True,
        tests_run=tests_run,
        duration_s=0.25,
    )

    assert artifact["gap1_stability_decision"]["value"] == "blocked_instability"
    assert artifact["gap1_registry_promoted"]["value"] is False
    assert artifact["frozen_subset"]["value"] is None
    assert artifact["stability_rule_predeclared"]["value"] is True
    assert artifact["no_new_broad_search"]["value"] is True
    assert artifact["refuted_single_invariant_excluded"]["value"] is True
    assert artifact["tests_run"]["value"] == tests_run
    assert artifact["inference_substrate"]["value"] == mod.INFERENCE_SUBSTRATE
    assert "blocked" in artifact["honest_verdict"]["value"]
    assert artifact["stability_audit"]["top_subset_count"] == 9
    assert artifact["stability_audit"]["n_grouped_splits"] == 20
    assert artifact["stability_audit"]["exact_subset_selection_frequency"] == 0.45
    assert artifact["stability_audit"]["exact_subset_stability_passed"] is False
    assert artifact["stability_audit"]["invariant_inclusion_stability_passed"] is True
    assert artifact["stability_audit"]["invariant_inclusion_frequencies"] == {
        "color_centroid_orientation": 0.6,
        "row_column_run_profile": 1.0,
    }
    assert artifact["stability_audit"]["no_heldout_tuning"] is True
    assert artifact["decision_path_features"] == [
        "exp5209.gap1_hardened_positive.value",
        "exp5209.leakage_audit_passed.value",
        "exp5209.leakage_audit.no_subset_selection_on_heldout_rows",
        "exp5209.subset_stability.selection_counts",
        "exp5222.gap1_registry_promoted.value",
    ]
    assert "at least 10 of 20 grouped splits" in artifact["future_reopen_criterion"]
    assert mod.artifact_schema_errors(artifact) == []
    assert json.loads((tmp_path / mod.RESULT_RELATIVE_PATH).read_text(encoding="utf-8")) == artifact

    gap_doc = (tmp_path / mod.VERIFIER_GAPS_RELATIVE_PATH).read_text(encoding="utf-8")
    assert "experiment_5237 GAP-1 stability freeze-or-retire decision" in gap_doc
    assert "decision=blocked_instability" in gap_doc
    assert "Minimum evidence to reopen" in gap_doc
    assert gap_doc.count("experiment_5237 GAP-1 stability freeze-or-retire decision") == 1
    assert mod.update_verifier_gap_doc(tmp_path, artifact) is True


def test_req_verify_5237_stability_rule_freezes_only_clean_deterministic_subsets() -> None:
    """REQ-VERIFY-5237: freeze requires gate, leakage, split agreement, and exclusion."""

    clean_stable = mod.build_artifact(
        exp5209=_exp5209_fixture(stable=True),
        exp5222=_exp5222_fixture(),
        tests_run=[],
        ops_docs_updated=False,
        duration_s=0.0,
    )
    missing = mod.build_artifact(
        exp5209={},
        exp5222=_exp5222_fixture(),
        tests_run=[],
        ops_docs_updated=False,
        duration_s=0.0,
    )
    missing_exp5222 = mod.build_artifact(
        exp5209=_exp5209_fixture(stable=True),
        exp5222={},
        tests_run=[],
        ops_docs_updated=False,
        duration_s=0.0,
    )
    weak_gate = mod.build_artifact(
        exp5209=_exp5209_fixture(gate=False),
        exp5222=_exp5222_fixture(),
        tests_run=[],
        ops_docs_updated=False,
        duration_s=0.0,
    )
    leakage = mod.build_artifact(
        exp5209=_exp5209_fixture(leakage=False),
        exp5222=_exp5222_fixture(),
        tests_run=[],
        ops_docs_updated=False,
        duration_s=0.0,
    )
    heldout_tuned = mod.build_artifact(
        exp5209=_exp5209_fixture(no_heldout_selection=False),
        exp5222=_exp5222_fixture(),
        tests_run=[],
        ops_docs_updated=False,
        duration_s=0.0,
    )
    refuted_path = mod.build_artifact(
        exp5209=_exp5209_fixture(
            stable=True,
            top_subset=[mod.REFUTED_SINGLE_INVARIANT],
        ),
        exp5222=_exp5222_fixture(),
        tests_run=[],
        ops_docs_updated=False,
        duration_s=0.0,
    )

    assert clean_stable["gap1_stability_decision"]["value"] == "frozen_promoted"
    assert clean_stable["gap1_registry_promoted"]["value"] is True
    assert clean_stable["frozen_subset"]["value"] == [
        "color_centroid_orientation",
        "row_column_run_profile",
    ]
    assert clean_stable["refuted_single_invariant_excluded"]["value"] is True
    assert missing["gap1_stability_decision"]["value"] == "blocked_missing_evidence"
    assert missing_exp5222["gap1_stability_decision"]["value"] == "blocked_missing_evidence"
    assert weak_gate["gap1_stability_decision"]["value"] == "blocked_missing_evidence"
    assert leakage["gap1_stability_decision"]["value"] == "blocked_missing_evidence"
    assert heldout_tuned["gap1_stability_decision"]["value"] == "blocked_missing_evidence"
    assert refuted_path["gap1_stability_decision"]["value"] == "retired_current_path"
    assert refuted_path["refuted_single_invariant_excluded"]["value"] is False
    assert "retired" in refuted_path["honest_verdict"]["value"]
    assert all(mod.artifact_schema_errors(row) == [] for row in (clean_stable, refuted_path))


def test_req_verify_5237_schema_rejects_malformed_artifacts() -> None:
    """REQ-VERIFY-5237: malformed stability decision artifacts fail closed."""

    artifact = mod.build_artifact(
        exp5209=_exp5209_fixture(),
        exp5222=_exp5222_fixture(),
        tests_run=[],
        ops_docs_updated=False,
        duration_s=0.0,
    )
    clean_stable = mod.build_artifact(
        exp5209=_exp5209_fixture(stable=True),
        exp5222=_exp5222_fixture(),
        tests_run=[],
        ops_docs_updated=False,
        duration_s=0.0,
    )

    missing = {key: value for key, value in artifact.items() if key != "gap1_stability_decision"}
    bad_decision = artifact | {
        "gap1_stability_decision": {
            "value": "weird",
            "principle": mod.FIELD_PRINCIPLES["gap1_stability_decision"],
        }
    }
    bad_promoted = artifact | {
        "gap1_registry_promoted": {
            "value": "false",
            "principle": mod.FIELD_PRINCIPLES["gap1_registry_promoted"],
        }
    }
    bad_frozen = artifact | {
        "frozen_subset": {
            "value": ["color_centroid_orientation"],
            "principle": mod.FIELD_PRINCIPLES["frozen_subset"],
        }
    }
    bad_frozen_promoted = clean_stable | {
        "gap1_registry_promoted": {
            "value": False,
            "principle": mod.FIELD_PRINCIPLES["gap1_registry_promoted"],
        }
    }
    bad_rule = artifact | {
        "stability_rule_predeclared": {
            "value": False,
            "principle": mod.FIELD_PRINCIPLES["stability_rule_predeclared"],
        }
    }
    bad_search = artifact | {
        "no_new_broad_search": {
            "value": False,
            "principle": mod.FIELD_PRINCIPLES["no_new_broad_search"],
        }
    }
    bad_refuted = artifact | {
        "refuted_single_invariant_excluded": {
            "value": "yes",
            "principle": mod.FIELD_PRINCIPLES["refuted_single_invariant_excluded"],
        }
    }
    bad_tests = artifact | {
        "tests_run": {
            "value": [{"command": "pytest"}],
            "principle": mod.FIELD_PRINCIPLES["tests_run"],
        }
    }
    bad_ops_docs_flag = artifact | {
        "ops_verifier_gaps_updated": {
            "value": "no",
            "principle": mod.FIELD_PRINCIPLES["ops_verifier_gaps_updated"],
        }
    }
    bad_substrate = artifact | {
        "inference_substrate": {
            "value": "deterministic_verifier_registry",
            "principle": mod.FIELD_PRINCIPLES["inference_substrate"],
        }
    }
    bad_verdict = artifact | {
        "honest_verdict": {
            "value": "blocked",
            "principle": mod.FIELD_PRINCIPLES["honest_verdict"],
        }
    }
    vague_verdict = artifact | {
        "honest_verdict": {
            "value": "complete: decision-grade outcome recorded",
            "principle": mod.FIELD_PRINCIPLES["honest_verdict"],
        }
    }
    bad_principle = artifact | {
        "no_new_broad_search": {
            "value": True,
            "principle": "wrong principle",
        }
    }

    assert any("missing required fields" in error for error in mod.artifact_schema_errors(missing))
    assert any(
        "gap1_stability_decision" in error for error in mod.artifact_schema_errors(bad_decision)
    )
    assert any(
        "gap1_registry_promoted" in error for error in mod.artifact_schema_errors(bad_promoted)
    )
    assert any("frozen_subset" in error for error in mod.artifact_schema_errors(bad_frozen))
    assert any(
        "frozen_promoted" in error for error in mod.artifact_schema_errors(bad_frozen_promoted)
    )
    assert any(
        "stability_rule_predeclared" in error for error in mod.artifact_schema_errors(bad_rule)
    )
    assert any("no_new_broad_search" in error for error in mod.artifact_schema_errors(bad_search))
    assert any(
        "refuted_single_invariant_excluded" in error
        for error in mod.artifact_schema_errors(bad_refuted)
    )
    assert any("tests_run" in error for error in mod.artifact_schema_errors(bad_tests))
    assert any(
        "ops_verifier_gaps_updated" in error
        for error in mod.artifact_schema_errors(bad_ops_docs_flag)
    )
    assert any(
        "inference_substrate" in error for error in mod.artifact_schema_errors(bad_substrate)
    )
    assert any("honest_verdict" in error for error in mod.artifact_schema_errors(bad_verdict))
    assert any("honest_verdict" in error for error in mod.artifact_schema_errors(vague_verdict))
    assert any("principle mismatch" in error for error in mod.artifact_schema_errors(bad_principle))


def test_req_verify_5237_malformed_stability_projection_fails_closed(tmp_path: Path) -> None:
    """REQ-VERIFY-5237: malformed stability projections cannot freeze or search anew."""

    malformed_rows = _exp5209_fixture(stable=True)
    stability = malformed_rows["subset_stability"]
    assert isinstance(stability, dict)
    stability.pop("top_subset_count")
    stability["selection_counts"] = [
        "not-a-row",
        {"subset": "not-a-list", "count": 1},
        {"subset": ["color_centroid_orientation"], "count": 0},
        {"subset": ["color_centroid_orientation", "row_column_run_profile"], "count": 11},
    ]
    artifact = mod.build_artifact(
        exp5209=malformed_rows,
        exp5222=_exp5222_fixture(),
        tests_run=[],
        ops_docs_updated=False,
        duration_s=0.0,
    )
    assert artifact["stability_audit"]["top_subset_count"] == 11
    assert artifact["gap1_stability_decision"]["value"] == "frozen_promoted"

    no_rows = _exp5209_fixture(stable=True)
    no_rows["subset_stability"] = {
        "top_subset": "not-a-list",
        "selection_counts": "not-a-list",
    }
    blocked = mod.build_artifact(
        exp5209=no_rows,
        exp5222=_exp5222_fixture(),
        tests_run=[],
        ops_docs_updated=False,
        duration_s=0.0,
    )
    assert blocked["gap1_stability_decision"]["value"] == "blocked_missing_evidence"

    (tmp_path / "ops").mkdir(parents=True)
    (tmp_path / mod.VERIFIER_GAPS_RELATIVE_PATH).write_text(
        "### GAP-1: transpose / orientation discrimination\n",
        encoding="utf-8",
    )
    assert mod.update_verifier_gap_doc(tmp_path, artifact) is True
    gap_doc = (tmp_path / mod.VERIFIER_GAPS_RELATIVE_PATH).read_text(encoding="utf-8")
    assert "experiment_5237 GAP-1 stability freeze-or-retire decision" in gap_doc
