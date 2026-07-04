"""Tests for Exp 5222's GAP-1 registry promotion decision.

Spec refs: REQ-VERIFY-5222, SCENARIO-VERIFY-5222.
"""

from __future__ import annotations

import json
from pathlib import Path

from carnot.verify import arc_gap1_registry_promotion_decision as mod


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "verification" / "spec.md"


def _field(value: object, principle: str = "fixture principle") -> dict[str, object]:
    return {"value": value, "principle": principle}


def _exp5209_fixture(
    *,
    gate: bool = True,
    stable: bool = False,
    leakage: bool = True,
    no_test_output_derived_features: bool = True,
) -> dict[str, object]:
    return {
        "experiment": "experiment_5209_gap1_set_search_holdout_hardening_v477",
        "result_path": "results/experiment_5209_gap1_set_search_holdout_hardening_v477.json",
        "gap1_hardened_positive": _field(gate),
        "leakage_audit_passed": _field(leakage),
        "best_subset_stable": _field(stable),
        "heldout_pass_at_2_mean": _field(0.189584),
        "baseline_always_on_pass_at_2_mean": _field(0.088976),
        "single_refuted_directional_pass_at_2_mean": _field(0.147787),
        "paired_delta_ci95": _field("[0.023148, 0.060446]"),
        "subset_stability": {
            "top_subset": ["color_centroid_orientation", "row_column_run_profile"],
            "top_subset_count": 9,
            "top_subset_fraction": 0.45,
            "stability_rule": "one exact subset selected in at least half of grouped splits",
            "selection_counts": [
                {
                    "subset": ["color_centroid_orientation", "row_column_run_profile"],
                    "count": 9,
                },
                {
                    "subset": [
                        "directional_adjacency_refuted_20260609",
                        "diagonal_adjacency_asymmetry",
                        "border_ordered_profile",
                        "color_centroid_orientation",
                        "row_column_run_profile",
                    ],
                    "count": 1,
                },
            ],
        },
        "leakage_audit": {
            "passed": leakage,
            "errors": [] if leakage else ["fixture leakage"],
            "no_test_gold_in_scoring": leakage,
            "no_test_output_derived_features": no_test_output_derived_features,
            "no_subset_selection_on_heldout_rows": True,
            "no_duplicate_task_ids_across_train_eval": leakage,
        },
        "baseline_definitions": {
            "always_on": ["object_count", "palette_histogram_shape"],
            "single_refuted_directional": ["directional_adjacency_refuted_20260609"],
            "selected_subset_rule": (
                "Within each split, maximize train pass@2, then train transpose captures, then smaller "
                "subset; held-out rows are used only for final metrics."
            ),
        },
        "candidate_discriminator_matrix": {
            "rows": [
                {
                    "candidate_id": "z_gold",
                    "correct_for_eval_only": True,
                    "scores": {"test_output_derived_feature_that_must_be_ignored": 999.0},
                }
            ]
        },
        "honest_verdict": _field(
            "complete: set_search_remains_positive_after_hardening_heldout_0.1896_always_0.0890_"
            "single_refuted_0.1478_paired_delta_ci95_0.0231_0.0604_best_subset_not_stable_"
            "do_not_promote_to_registry_here"
        ),
    }


def _write_fixture_repo(root: Path, exp5209: dict[str, object]) -> None:
    (root / "results").mkdir(parents=True, exist_ok=True)
    (root / mod.EXP5209_RELATIVE_PATH).write_text(
        json.dumps(exp5209, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (root / "ops").mkdir(parents=True, exist_ok=True)
    (root / mod.VERIFIER_GAPS_RELATIVE_PATH).write_text(
        "### GAP-1: transpose / orientation discrimination\n"
        "- status: open -- exp5209 positive but not registry-promoted yet\n"
        "<!-- experiment_5209_gap1_set_search_holdout_hardening_v477 start -->\n"
        "- exp5209 line\n"
        "<!-- experiment_5209_gap1_set_search_holdout_hardening_v477 end -->\n"
        "\n### GAP-2: next\n",
        encoding="utf-8",
    )
    (root / mod.VERIFIER_REGISTRY_RELATIVE_PATH).write_text(
        "verifiers:\n"
        "- verifier_id: arc_grid_combined_verifier\n"
        "  domain: arc_agi1_grid\n"
        "  status: candidate\n",
        encoding="utf-8",
    )


def test_req_verify_5222_spec_declares_registry_decision_contract() -> None:
    """REQ-VERIFY-5222: OpenSpec declares the GAP-1 registry decision artifact."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    section = spec[spec.index("### REQ-VERIFY-5222") :]

    for marker in (
        "REQ-VERIFY-5222",
        "SCENARIO-VERIFY-5222",
        mod.RESULT_RELATIVE_PATH,
        "gap1_hardened_positive.value",
        "gap1_registry_promoted",
        "gap1_registry_decision",
        "blocked_instability",
        "directional_adjacency_refuted_20260609",
    ):
        assert marker in section


def test_scenario_verify_5222_positive_unstable_exp5209_blocks_registry_promotion(
    tmp_path: Path,
) -> None:
    """SCENARIO-VERIFY-5222: a positive but unstable upstream gate is blocked."""

    _write_fixture_repo(tmp_path, _exp5209_fixture())

    tests_run = [
        {"command": ".venv/bin/pytest tests/python/test_experiment_5222... -q", "passed": True}
    ]
    artifact = mod.run(
        root=tmp_path,
        result_path=tmp_path / mod.RESULT_RELATIVE_PATH,
        update_gap_doc=True,
        tests_run=tests_run,
        duration_s=0.25,
    )

    assert artifact["gap1_registry_promoted"]["value"] is False
    assert artifact["gap1_registry_decision"]["value"] == "blocked_instability"
    assert artifact["promoted_registry_path"]["value"] is None
    assert artifact["frozen_subset"]["value"] is None
    assert artifact["exp5209_gate_parsed_from_value"]["value"] is True
    assert artifact["refuted_single_invariant_excluded"]["value"] is True
    assert artifact["tests_run"]["value"] == tests_run
    assert artifact["ops_docs_updated"]["value"] is True
    assert artifact["inference_substrate"]["value"] == "deterministic_verifier_registry"
    assert "blocked_instability" in artifact["honest_verdict"]["value"]
    assert "not the exp5210 gate-shape failure alone" in artifact["honest_verdict"]["value"]
    assert artifact["subset_freeze_audit"]["can_freeze_without_heldout_tuning"] is False
    assert artifact["subset_freeze_audit"]["top_subset"] == [
        "color_centroid_orientation",
        "row_column_run_profile",
    ]
    assert artifact["subset_freeze_audit"]["heldout_rows_used_for_freeze"] is False
    assert artifact["registry_audit"]["promoted_gap1_registry_entry_present"] is False
    assert artifact["decision_path_features"] == [
        "gap1_hardened_positive.value",
        "leakage_audit_passed.value",
        "leakage_audit.no_test_output_derived_features",
        "best_subset_stable.value",
        "subset_stability.top_subset",
    ]
    assert mod.artifact_schema_errors(artifact) == []
    assert json.loads((tmp_path / mod.RESULT_RELATIVE_PATH).read_text(encoding="utf-8")) == artifact
    assert mod.update_verifier_gap_doc(tmp_path, artifact) is True

    gap_doc = (tmp_path / mod.VERIFIER_GAPS_RELATIVE_PATH).read_text(encoding="utf-8")
    assert "experiment_5222 GAP-1 registry promotion decision" in gap_doc
    assert "blocked_instability" in gap_doc
    assert gap_doc.count("experiment_5222 GAP-1 registry promotion decision") == 1


def test_req_verify_5222_leakage_and_missing_evidence_fail_closed() -> None:
    """REQ-VERIFY-5222: leakage or missing upstream fields cannot promote."""

    leakage = mod.build_artifact(
        exp5209=_exp5209_fixture(leakage=False),
        tests_run=[],
        ops_docs_updated=False,
        duration_s=0.0,
    )
    output_feature_leak = mod.build_artifact(
        exp5209=_exp5209_fixture(no_test_output_derived_features=False),
        tests_run=[],
        ops_docs_updated=False,
        duration_s=0.0,
    )
    missing = mod.build_artifact(
        exp5209={},
        tests_run=[],
        ops_docs_updated=False,
        duration_s=0.0,
    )
    no_audit = mod.build_artifact(
        exp5209={
            "gap1_hardened_positive": _field(True),
            "leakage_audit_passed": _field(True),
            "best_subset_stable": _field(False),
        },
        tests_run=[],
        ops_docs_updated=False,
        duration_s=0.0,
    )
    weak = mod.build_artifact(
        exp5209=_exp5209_fixture(gate=False),
        tests_run=[],
        ops_docs_updated=False,
        duration_s=0.0,
    )
    stable_but_unimplemented = mod.build_artifact(
        exp5209=_exp5209_fixture(stable=True),
        tests_run=[],
        ops_docs_updated=False,
        duration_s=0.0,
    )

    assert leakage["gap1_registry_decision"]["value"] == "blocked_leakage"
    assert output_feature_leak["gap1_registry_decision"]["value"] == "blocked_leakage"
    assert missing["gap1_registry_decision"]["value"] == "blocked_missing_evidence"
    assert no_audit["gap1_registry_decision"]["value"] == "blocked_leakage"
    assert weak["gap1_registry_decision"]["value"] == "blocked_missing_evidence"
    assert stable_but_unimplemented["gap1_registry_decision"]["value"] == "blocked_missing_evidence"
    assert "promoted" in mod._verdict("promoted", True)  # noqa: SLF001
    assert all(
        artifact["gap1_registry_promoted"]["value"] is False
        for artifact in (
            leakage,
            output_feature_leak,
            missing,
            no_audit,
            weak,
            stable_but_unimplemented,
        )
    )


def test_req_verify_5222_schema_rejects_malformed_artifacts() -> None:
    """REQ-VERIFY-5222: malformed registry decision artifacts fail closed."""

    artifact = mod.build_artifact(
        exp5209=_exp5209_fixture(),
        tests_run=[],
        ops_docs_updated=False,
        duration_s=0.0,
    )

    missing = {key: value for key, value in artifact.items() if key != "gap1_registry_promoted"}
    bad_decision = artifact | {
        "gap1_registry_decision": {
            "value": "weird",
            "principle": mod.FIELD_PRINCIPLES["gap1_registry_decision"],
        }
    }
    bad_promoted = artifact | {
        "gap1_registry_promoted": {
            "value": "false",
            "principle": mod.FIELD_PRINCIPLES["gap1_registry_promoted"],
        }
    }
    bad_verdict = artifact | {
        "honest_verdict": {
            "value": "blocked",
            "principle": mod.FIELD_PRINCIPLES["honest_verdict"],
        }
    }
    bad_tests = artifact | {
        "tests_run": {
            "value": [{"command": "pytest"}],
            "principle": mod.FIELD_PRINCIPLES["tests_run"],
        }
    }
    bad_principle = artifact | {
        "ops_docs_updated": {
            "value": False,
            "principle": "wrong principle",
        }
    }
    bad_nulls = artifact | {
        "promoted_registry_path": {
            "value": "ops/verifier_registry.yaml",
            "principle": mod.FIELD_PRINCIPLES["promoted_registry_path"],
        }
    }
    bad_substrate = artifact | {
        "inference_substrate": {
            "value": "live_llm",
            "principle": mod.FIELD_PRINCIPLES["inference_substrate"],
        }
    }
    bad_exp5209_parse = artifact | {
        "exp5209_gate_parsed_from_value": {
            "value": "yes",
            "principle": mod.FIELD_PRINCIPLES["exp5209_gate_parsed_from_value"],
        }
    }
    bad_refuted_flag = artifact | {
        "refuted_single_invariant_excluded": {
            "value": "yes",
            "principle": mod.FIELD_PRINCIPLES["refuted_single_invariant_excluded"],
        }
    }
    bad_ops_docs = artifact | {
        "ops_docs_updated": {
            "value": "no",
            "principle": mod.FIELD_PRINCIPLES["ops_docs_updated"],
        }
    }

    assert any("missing required fields" in error for error in mod.artifact_schema_errors(missing))
    assert any(
        "gap1_registry_decision" in error for error in mod.artifact_schema_errors(bad_decision)
    )
    assert any(
        "gap1_registry_promoted" in error for error in mod.artifact_schema_errors(bad_promoted)
    )
    assert any("honest_verdict" in error for error in mod.artifact_schema_errors(bad_verdict))
    assert any("tests_run" in error for error in mod.artifact_schema_errors(bad_tests))
    assert any("principle mismatch" in error for error in mod.artifact_schema_errors(bad_principle))
    assert any(
        "null promoted_registry_path" in error for error in mod.artifact_schema_errors(bad_nulls)
    )
    assert any(
        "inference_substrate" in error for error in mod.artifact_schema_errors(bad_substrate)
    )
    assert any(
        "exp5209_gate_parsed_from_value" in error
        for error in mod.artifact_schema_errors(bad_exp5209_parse)
    )
    assert any(
        "refuted_single_invariant_excluded" in error
        for error in mod.artifact_schema_errors(bad_refuted_flag)
    )
    assert any("ops_docs_updated" in error for error in mod.artifact_schema_errors(bad_ops_docs))
