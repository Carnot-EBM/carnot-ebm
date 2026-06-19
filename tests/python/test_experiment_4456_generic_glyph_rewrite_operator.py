"""Tests for Exp 4456 generic glyph-rewrite rule verifier.

Spec refs: REQ-REPORT-4456, SCENARIO-REPORT-4456.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from carnot import experiment_4456_generic_glyph_rewrite_operator as mod
from carnot.agentic import arc_solver_kit as kit


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "research-reporting" / "spec.md"


def _examples() -> list[dict[str, str]]:
    return [
        {
            "game": "bsqsshqpox",
            "rule_id": "greedy_multi_glyph_lhs_rewrite",
            "predicate": "scan target left-to-right; first prefix LHS emits RHS",
        },
        {
            "game": "tr87_reference",
            "rule_id": "double_translation_rewrite",
            "predicate": "N-pass glyph rewrite handles double_translation and tree_translation",
        },
        {
            "game": "tr87_reference",
            "rule_id": "alter_rules_inverse",
            "predicate": "editable rules are adjusted so rewrite(target) equals fixed editable sequence",
        },
    ]


def _ok_preconditions() -> dict[str, Any]:
    return {
        "tr87_env_present": True,
        "arc_solver_kit_importable": True,
        "generator_resource_available": True,
        "gguf_cached": True,
        "igpu_llama_server_available": False,
        "focused_baseline_selected_green": True,
        "focused_baseline_exact_command_green": False,
        "focused_baseline_exact_command_blocker": "repo_addopts_package_wide_coverage_on_focused_k_slice",
        "no_3090_inference": True,
        "leaderboard_submission": False,
        "ok": True,
    }


def _write_fixture_repo(root: Path) -> None:
    (root / "environment_files" / "tr87" / "fixture").mkdir(parents=True)
    (root / "results").mkdir(parents=True, exist_ok=True)
    (root / "ops").mkdir(parents=True, exist_ok=True)
    (root / mod.REGISTRY_RELATIVE_PATH).write_text(
        "\n".join(
            [
                "schema_version: 1",
                "updated: '2026-06-19'",
                "general_gotchas:",
                "- id: primitive_glyph_rewrite_matcher",
                "  operator: glyph_rewrite_matcher",
                "games:",
                "- game: tr87",
                "  mechanic_class: config_substitution",
                "  reproducibility: reproduced",
                "  levels_reproduced: 6",
                "  win_condition: greedy glyph rewrite lhs rhs map",
                "",
            ]
        ),
        encoding="utf-8",
    )


def _generic_reproduce(solution: Sequence[str]) -> dict[str, Any]:
    assert list(solution) == ['{"action": 2}', '{"action": 3}']
    return {"game": "tr87", "claimed_level": 1, "reached_level": 1, "reproduced": True}


def test_req_report_4456_spec_declares_glyph_rewrite_contract() -> None:
    """REQ-REPORT-4456: OpenSpec declares the operator and required artifact fields."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-REPORT-4456" in spec
    assert "SCENARIO-REPORT-4456" in spec
    assert "glyph_rewrite_rule_verifier" in spec
    assert mod.RESULT_RELATIVE_PATH in spec
    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert field in spec


def test_req_report_4456_solver_kit_grounds_multi_lhs_rewrite() -> None:
    """REQ-REPORT-4456: greedy multi-glyph LHS rewrite grounds from examples."""

    result = kit.glyph_rewrite_rule_verifier(
        game="tr87",
        object_digest={
            "rule_family": "glyph_rewrite",
            "target_sequence": ["X6", "X1", "X5", "X1", "X4", "X2", "X3", "X3"],
            "editable_sequence": ["Y3", "Y4", "Y4", "Y5", "Y1", "Y4", "Y6"],
            "rules": [
                {"lhs": ["X6"], "rhs": ["Y4"]},
                {"lhs": ["X3", "X3"], "rhs": ["Y6", "Y1"]},
                {"lhs": ["X4"], "rhs": ["Y7", "Y7"]},
                {"lhs": ["X7", "X7"], "rhs": ["Y3"]},
                {"lhs": ["X1", "X5", "X1"], "rhs": ["Y6"]},
                {"lhs": ["X2"], "rhs": ["Y5"]},
            ],
        },
        few_shot_examples=_examples(),
    )

    assert result["operator"] == "glyph_rewrite_rule_verifier"
    assert result["legacy_operator"] == "glyph_rewrite_matcher"
    assert result["grounded"] is True
    assert result["predicate_id"] == "greedy_multi_glyph_lhs_rewrite"
    assert result["recipe_source"] == "generic_glyph_rewrite_rule_verifier"
    assert result["target_recipe_withheld"] == "tr87"
    assert result["required_editable_sequence"] == ["Y4", "Y6", "Y7", "Y7", "Y5", "Y6", "Y1"]
    assert result["distance"] == 15.0
    assert result["counterexample_rounds"] == 1
    assert result["grounded_win_condition"]["fires_on_win"] is False
    assert result["verifier_is_oracle"] is True

    operators = {row.operator for row in kit.primitive_operator_registry()}
    assert "glyph_rewrite_matcher" in operators
    assert "glyph_rewrite_rule_verifier" in operators
    selected = kit.select_primitive_operators(mechanic_class="config_substitution")
    assert [row.operator for row in selected][:2] == [
        "glyph_rewrite_rule_verifier",
        "glyph_rewrite_matcher",
    ]


def test_scenario_report_4456_solver_kit_supports_n_pass_and_alter_rules() -> None:
    """SCENARIO-REPORT-4456: N-pass and alter_rules variants are grounded."""

    n_pass = kit.glyph_rewrite_rule_verifier(
        game="tr87",
        object_digest={
            "rule_family": "glyph_rewrite",
            "target_sequence": ["A6", "A1", "A4"],
            "editable_sequence": ["C3", "C2", "C6"],
            "rules": [
                {"lhs": ["A6"], "rhs": ["B1"]},
                {"lhs": ["A1"], "rhs": ["B3"]},
                {"lhs": ["A4"], "rhs": ["B7"]},
                {"lhs": ["B1"], "rhs": ["C3"]},
                {"lhs": ["B3"], "rhs": ["C2"]},
                {"lhs": ["B7"], "rhs": ["C7"]},
            ],
            "flags": {"double_translation": True},
        },
        few_shot_examples=_examples(),
    )
    assert n_pass["grounded"] is True
    assert n_pass["predicate_id"] == "n_pass_greedy_glyph_rewrite"
    assert n_pass["rewrite_passes"] == 2
    assert n_pass["required_editable_sequence"] == ["C3", "C2", "C7"]
    assert n_pass["distance"] == 1.0

    alter = kit.glyph_rewrite_rule_verifier(
        game="tr87",
        object_digest={
            "rule_family": "glyph_rewrite",
            "target_sequence": ["X3"],
            "editable_sequence": ["Y5"],
            "rules": [{"lhs": ["X2"], "rhs": ["Y1"]}],
            "flags": {"alter_rules": True},
        },
        few_shot_examples=_examples(),
    )
    assert alter["grounded"] is True
    assert alter["predicate_id"] == "alter_rules_inverse_rewrite"
    assert alter["required_rule_sides"] == [3, 5]
    assert alter["distance"] == 4.0

    two_pass_alter = kit.glyph_rewrite_rule_verifier(
        game="tr87",
        object_digest={
            "rule_family": "glyph_rewrite",
            "target_sequence": ["A1"],
            "editable_sequence": ["C3", "C5"],
            "rules": [
                {"lhs": ["A1"], "rhs": ["B1", "B2"]},
                {"lhs": ["B1"], "rhs": ["C3"]},
                {"lhs": ["B2"], "rhs": ["C5"]},
            ],
            "flags": {"alter_rules": True, "double_translation": True},
        },
        few_shot_examples=_examples(),
    )
    assert two_pass_alter["grounded"] is True
    assert two_pass_alter["predicate_id"] == "alter_rules_two_pass_rewrite"
    assert two_pass_alter["distance"] == 0.0


def test_req_report_4456_solver_kit_rejects_ungrounded_candidates() -> None:
    """REQ-REPORT-4456: non-grounding candidates return a residual instead of a solve."""

    result = kit.glyph_rewrite_rule_verifier(
        game="tr87",
        object_digest={
            "rule_family": "glyph_rewrite",
            "target_sequence": ["X9"],
            "editable_sequence": ["Y1"],
            "rules": [{"lhs": ["X1"], "rhs": ["Y1"]}],
        },
        few_shot_examples=_examples(),
    )

    assert result["grounded"] is False
    assert result["solution"] == []
    assert result["residual"] == "glyph_rewrite_candidate_did_not_ground"
    assert result["verifier_is_oracle"] is True


def test_scenario_report_4456_run_reproduces_tr87_generically(tmp_path: Path) -> None:
    """SCENARIO-REPORT-4456: generic tr87 progress is reproduction-gated and terminal."""

    _write_fixture_repo(tmp_path)
    clock = {"t": 50.0}

    def now() -> float:
        return clock["t"]

    def sleep(seconds: float) -> None:
        clock["t"] += seconds

    artifact = mod.run(
        root=tmp_path,
        preconditions_checked=_ok_preconditions(),
        few_shot_examples=_examples(),
        solve_tr87_fn=lambda _examples, _target_level: {
            "solution": ['{"action": 2}', '{"action": 3}'],
            "reached_level": 1,
            "states_expanded": 7,
            "operator_result": {
                "operator": "glyph_rewrite_rule_verifier",
                "grounded": True,
                "recipe_source": "generic_glyph_rewrite_rule_verifier",
                "target_recipe_withheld": "tr87",
                "counterexample_rounds": 1,
            },
        },
        reproduce_generic_fn=_generic_reproduce,
        no_regression_fn=lambda _root: True,
        now=now,
        sleep_fn=sleep,
    )

    assert artifact["honest_verdict"] == "success: tr87_generic_glyph_rewrite_L1_offline_reproduced"
    assert artifact["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert artifact["duration_s"] >= 1.0
    assert artifact["tr87_resolved_generically"] is True
    assert artifact["tr87_generic_level_reproduced"] == 1
    assert artifact["counterexample_rounds"] == 1
    assert artifact["offline_reproduced"] is True
    assert artifact["no_regression"] is True
    assert artifact["missing_verifier_gaps"] == []
    assert artifact["verifier_is_oracle"] is True
    assert artifact["submitted_to_leaderboard"] is False
    assert mod.artifact_schema_errors(artifact) == []

    written = json.loads((tmp_path / mod.RESULT_RELATIVE_PATH).read_text(encoding="utf-8"))
    assert written["generic_operator_result"]["target_recipe_withheld"] == "tr87"


def test_req_report_4456_blocked_precondition_and_schema_guards(tmp_path: Path) -> None:
    """REQ-REPORT-4456: blocked resources do not fabricate generic or regression claims."""

    _write_fixture_repo(tmp_path)
    calls: list[str] = []
    artifact = mod.run(
        root=tmp_path,
        preconditions_checked={**_ok_preconditions(), "tr87_env_present": False, "ok": False},
        few_shot_examples=_examples(),
        solve_tr87_fn=lambda _examples, _target_level: calls.append("solve") or {},
        reproduce_generic_fn=lambda _solution: calls.append("reproduce") or {},
        no_regression_fn=lambda _root: calls.append("regression") or True,
        now=lambda: 1.0,
        sleep_fn=lambda _seconds: None,
    )

    assert calls == []
    assert artifact["honest_verdict"] == "complete: blocked_offline_env_tr87"
    assert artifact["tr87_resolved_generically"] is False
    assert artifact["tr87_generic_level_reproduced"] == 0
    assert artifact["offline_reproduced"] is False
    assert artifact["no_regression"] is False
    assert mod.artifact_schema_errors(artifact) == []

    bad: Mapping[str, Any] = {
        **artifact,
        "honest_verdict": "partial: retry",
        "inference_substrate": None,
        "tr87_resolved_generically": "true",
        "tr87_generic_level_reproduced": "1",
        "counterexample_rounds": "1",
        "offline_reproduced": "true",
        "no_regression": "true",
        "missing_verifier_gaps": {},
        "verifier_is_oracle": False,
        "random_seed": "4456",
        "reproducibility_checksum": "bad",
    }
    errors = mod.artifact_schema_errors(bad)

    assert "honest_verdict must start with a terminal prefix" in errors
    assert "honest_verdict must not use partial prefix" in errors
    assert "missing inference_substrate" in errors
    assert "tr87_resolved_generically must be bare bool" in errors
    assert "tr87_generic_level_reproduced must be bare int" in errors
    assert "counterexample_rounds must be bare int" in errors
    assert "offline_reproduced must be bare bool" in errors
    assert "no_regression must be bare bool" in errors
    assert "missing_verifier_gaps must be list" in errors
    assert "verifier_is_oracle must be true" in errors
    assert "random_seed must be bare int" in errors
    assert "reproducibility_checksum must be 64-char sha256 hex" in errors
