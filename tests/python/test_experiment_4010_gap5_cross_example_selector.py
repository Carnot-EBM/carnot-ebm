"""Tests for Exp 4010 GAP-5 cross-example consistency selector.

Spec refs: REQ-VERIFY-4010, SCENARIO-VERIFY-4010.
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import Any

import pytest

from carnot.agentic import gap5_cross_example_selector as gap5


JsonDict = dict[str, Any]


def _write_json(path: Path, payload: JsonDict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_gzip_json(path: Path, payload: JsonDict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        json.dump(payload, handle)


def _code(mapping: dict[int, int]) -> str:
    pairs = "\n".join(f"        {src}: {dst}," for src, dst in sorted(mapping.items()))
    return (
        "def transform(grid):\n"
        "    value = int(grid[0, 0])\n"
        "    table = {\n"
        f"{pairs}\n"
        "    }\n"
        "    return np.array([[table.get(value, value)]])\n"
    )


def _entry(task: str, value: int, gold: int) -> JsonDict:
    return {
        "task": task,
        "demos": [
            {"input": [[0]], "output": [[0]]},
            {"input": [[1]], "output": [[1]]},
        ],
        "test_input": [[value]],
        "candidates": [{"grid": [[gold]], "correct": True}],
    }


def _arm(source: str, mapping: dict[int, int]) -> JsonDict:
    return {
        "source": source,
        "code": _code(mapping),
        "n_calls": 0,
        "codex_seconds": 0.0,
    }


def _fixture_paths(tmp_path: Path) -> dict[str, Path]:
    result_dir = tmp_path / "results"
    return {
        "pool": result_dir / "arc3_gap4_arc2_eval_pool.json.gz",
        "arc2_programs": result_dir / "arc3_gap4_arc2_induced_programs.json",
        "arc1_programs": result_dir / "arc3_gap4_induced_programs.json",
        "chain": result_dir / "arc3_gap4_arc2_chain_ensemble.json",
        "output": result_dir / "experiment_4010_gap5_cross_example_consistency_selector.json",
    }


def _write_selector_fixture(tmp_path: Path) -> dict[str, Path]:
    paths = _fixture_paths(tmp_path)
    entries = [
        _entry("lift", 7, 9),
        _entry("lift", 8, 9),
        _entry("abstain", 7, 9),
        _entry("abstain", 8, 9),
    ]
    chain = {
        "preregistration": {"tasks": ["lift", "abstain"]},
        "per_task": [
            {
                "task": "lift",
                "arms": [
                    _arm("correct_supported", {0: 0, 1: 1, 7: 9, 8: 9}),
                    _arm("demo_perfect_wrong", {0: 0, 1: 1, 7: 5, 8: 6}),
                    _arm("sibling_supporter_nonselectable", {0: 0, 1: 2, 7: 4, 8: 9}),
                ],
            },
            {
                "task": "abstain",
                "arms": [
                    _arm("correct_no_support", {0: 0, 1: 1, 7: 9, 8: 9}),
                    _arm("wrong_no_support_a", {0: 0, 1: 1, 7: 5, 8: 6}),
                    _arm("wrong_no_support_b", {0: 0, 1: 1, 7: 4, 8: 7}),
                ],
            },
        ],
    }
    _write_gzip_json(paths["pool"], {"entries": entries})
    _write_json(paths["arc2_programs"], {"experiment": "fixture", "programs": []})
    _write_json(paths["arc1_programs"], {"experiment": "fixture", "programs": []})
    _write_json(paths["chain"], chain)
    return paths


def test_req_verify_4010_spec_anchor_exists() -> None:
    """REQ-VERIFY-4010: OpenSpec declares the offline cross-example contract."""

    spec = Path("openspec/capabilities/verification/spec.md").read_text(encoding="utf-8")

    assert "REQ-VERIFY-4010" in spec
    assert "SCENARIO-VERIFY-4010" in spec
    assert "cross_example_precision" in spec
    assert "blocked_saved_programs_missing" in spec
    assert "n_codex_calls=0" in spec


def test_scenario_verify_4010_scores_dominance_and_abstention(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-4010: sibling consistency can select or abstain offline."""

    paths = _write_selector_fixture(tmp_path)
    artifact = gap5.run(
        pool_path=paths["pool"],
        arc2_programs_path=paths["arc2_programs"],
        arc1_programs_path=paths["arc1_programs"],
        chain_artifact_path=paths["chain"],
        output_path=paths["output"],
        bootstrap_iters=200,
    )

    assert paths["output"].exists()
    assert artifact["n_codex_calls"] == 0
    assert artifact["n_tasks_scored"] == 2
    assert artifact["cross_example_precision"] == 1.0
    assert artifact["output_agreement_precision_ref"] == 0.0
    assert artifact["cross_example_coverage"] == 0.5
    assert artifact["output_agreement_coverage_ref"] == 0.0
    assert artifact["per_task"][0]["task"] == "lift"
    assert artifact["per_task"][0]["cross_example_selected_source"] == "correct_supported"
    assert artifact["per_task"][0]["cross_example_selected_gold"] is True
    assert artifact["per_task"][1]["task"] == "abstain"
    assert artifact["per_task"][1]["cross_example_abstain_reason"] == "sibling_input_disagreement"
    assert artifact["missing_verifier_gaps"] == [
        {
            "task": "abstain",
            "failure_mode": "cross_example_selector_abstained_sibling_input_disagreement",
            "missing_discriminator": "higher-order rule consistency beyond demo reproduction and sibling agreement",
        }
    ]
    gap5.validate_artifact(artifact)


def test_req_verify_4010_task_scoring_uses_demo_reproduction_and_sibling_agreement(
    tmp_path: Path,
) -> None:
    """REQ-VERIFY-4010: candidate scores expose leave-one-demo-out and sibling rates."""

    paths = _write_selector_fixture(tmp_path)
    pool = gap5.load_eval_pool(paths["pool"])
    chain = gap5.load_json(paths["chain"])
    entries_by_task = gap5.group_entries_by_task(pool["entries"])
    row = gap5.score_task("lift", entries_by_task["lift"], chain["per_task"][0]["arms"])

    selected = row["cross_example_selected_source"]
    candidate_scores = {candidate["source"]: candidate for candidate in row["candidate_scores"]}

    assert selected == "correct_supported"
    assert candidate_scores["correct_supported"]["demo_reproduction_rate"] == 1.0
    assert candidate_scores["demo_perfect_wrong"]["demo_reproduction_rate"] == 1.0
    assert candidate_scores["sibling_supporter_nonselectable"]["demo_reproduction_rate"] == 0.5
    assert candidate_scores["correct_supported"]["sibling_agreement"] == pytest.approx(0.5)
    assert candidate_scores["demo_perfect_wrong"]["sibling_agreement"] == pytest.approx(0.0)
    assert candidate_scores["correct_supported"]["selectable"] is True
    assert candidate_scores["sibling_supporter_nonselectable"]["selectable"] is False


def test_scenario_verify_4010_blocks_without_saved_programs_or_pool(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-4010: blocked preconditions do not fabricate lift metrics."""

    paths = _fixture_paths(tmp_path)
    _write_gzip_json(paths["pool"], {"entries": []})

    blocked_programs = gap5.run(
        pool_path=paths["pool"],
        arc2_programs_path=paths["arc2_programs"],
        arc1_programs_path=paths["arc1_programs"],
        chain_artifact_path=paths["chain"],
        output_path=paths["output"],
    )

    assert blocked_programs["honest_verdict"] == "blocked_saved_programs_missing"
    assert blocked_programs["n_codex_calls"] == 0
    assert blocked_programs["n_tasks_scored"] == 0

    _write_json(paths["arc2_programs"], {"experiment": "fixture", "programs": []})
    _write_json(paths["arc1_programs"], {"experiment": "fixture", "programs": []})
    _write_json(paths["chain"], {"per_task": []})
    blocked_pool = gap5.run(
        pool_path=tmp_path / "missing_pool.json.gz",
        arc2_programs_path=paths["arc2_programs"],
        arc1_programs_path=paths["arc1_programs"],
        chain_artifact_path=paths["chain"],
        output_path=paths["output"],
        write=False,
    )

    assert blocked_pool["honest_verdict"] == "blocked_eval_pool_unreadable"
    assert blocked_pool["cross_example_precision"] == 0.0


def test_req_verify_4010_edge_outcomes_and_verdict_branches(tmp_path: Path) -> None:
    """REQ-VERIFY-4010: edge outcomes remain explicit and schema-stable."""

    paths = _fixture_paths(tmp_path)
    entries = [
        _entry("baseline_wrong", 7, 9),
        _entry("baseline_wrong", 8, 9),
        _entry("no_sibling", 7, 9),
        _entry("fewer", 7, 9),
        _entry("missing_arms", 7, 9),
        {
            "task": "no_gold",
            "demos": [{"input": [[0]], "output": [[0]]}],
            "test_input": [[7]],
            "candidates": [],
        },
        {
            "task": "no_gold",
            "demos": [{"input": [[0]], "output": [[0]]}],
            "test_input": [[8]],
            "candidates": [],
        },
    ]
    chain = {
        "per_task": [
            {
                "task": "baseline_wrong",
                "arms": [
                    _arm("wrong_a", {0: 0, 1: 1, 7: 5, 8: 6}),
                    _arm("wrong_b", {0: 0, 1: 1, 7: 5, 8: 6}),
                ],
            },
            {
                "task": "no_sibling",
                "arms": [
                    _arm("one", {0: 0, 1: 1, 7: 9}),
                    _arm("two", {0: 0, 1: 1, 7: 5}),
                ],
            },
            {
                "task": "fewer",
                "arms": [
                    _arm("perfect", {0: 0, 1: 1, 7: 9}),
                    _arm("nonperfect", {0: 0, 1: 2, 7: 9}),
                ],
            },
            {
                "task": "no_gold",
                "arms": [
                    _arm("selected_wrong_without_gold_a", {0: 0, 7: 4, 8: 4}),
                    _arm("selected_wrong_without_gold_b", {0: 0, 7: 5, 8: 6}),
                    _arm("supporter", {0: 1, 7: 4, 8: 4}),
                ],
            },
            {"task": "missing_arms", "arms": []},
        ],
    }
    _write_gzip_json(paths["pool"], {"entries": entries})
    _write_json(paths["arc2_programs"], {"experiment": "fixture", "programs": []})
    _write_json(paths["arc1_programs"], {"experiment": "fixture", "programs": []})
    _write_json(paths["chain"], chain)

    artifact = gap5.run(
        pool_path=paths["pool"],
        arc2_programs_path=paths["arc2_programs"],
        arc1_programs_path=paths["arc1_programs"],
        chain_artifact_path=paths["chain"],
        output_path=paths["output"],
        bootstrap_iters=0,
        write=False,
    )
    by_task = {row["task"]: row for row in artifact["per_task"]}

    assert gap5.selected_tasks_from_chain_artifact(chain) == [
        "baseline_wrong",
        "no_sibling",
        "fewer",
        "no_gold",
        "missing_arms",
    ]
    assert by_task["baseline_wrong"]["output_agreement_selected"] is True
    assert by_task["baseline_wrong"]["output_agreement_selected_gold"] is False
    assert by_task["baseline_wrong"]["cross_example_abstain_reason"] == "score_tie"
    assert by_task["no_sibling"]["cross_example_abstain_reason"] == "no_sibling_inputs"
    assert by_task["no_gold"]["cross_example_selected"] is True
    assert by_task["no_gold"]["cross_example_selected_gold"] is False
    assert artifact["skipped_tasks"] == [
        {"task": "fewer", "reason": "fewer_than_two_demo_perfect_candidates"},
        {"task": "missing_arms", "reason": "missing_entries_or_arms"},
    ]
    assert gap5._grid_hash(None) is None  # noqa: SLF001
    assert gap5._call_transform(lambda _grid: None, [[0]]) is None  # noqa: SLF001

    def raises(_grid: Any) -> None:
        raise RuntimeError("boom")

    assert gap5._call_transform(raises, [[0]]) is None  # noqa: SLF001
    assert gap5._demo_reproduction_rate(None, []) == 0.0  # noqa: SLF001
    assert (
        gap5._demo_reproduction_rate(  # noqa: SLF001
            lambda grid: [[int(grid[0][0]), int(grid[0][0]), int(grid[0][0])]],
            [{"input": [[3, 3], [3, 3]], "output": [[3, 3, 3]]}],
        )
        == 1.0
    )
    assert gap5._output_agreement_baseline(  # noqa: SLF001
        [{"selectable": True, "predictions": [None]}],
        0,
        None,
    ) == {
        "selected": False,
        "selected_gold": False,
        "selected_hash": None,
        "n_matching": 0,
    }
    assert gap5.paired_bootstrap_ci([], bootstrap_iters=0)["coverage_lift_ci95"] == [0.0, 0.0]
    assert gap5._verdict(  # noqa: SLF001
        True,
        {"cross_precision": 1.0, "cross_coverage": 1.0},
        {"precision_lift_ci95": [0.1, 0.2], "coverage_lift_ci95": [0.0, 0.0]},
    ).startswith("success:")
    assert "precision_lower" in gap5._verdict(  # noqa: SLF001
        False,
        {
            "cross_precision": 0.0,
            "output_precision": 1.0,
            "cross_coverage": 1.0,
            "output_coverage": 1.0,
        },
        {"precision_lift_ci95": [-1.0, 0.0], "coverage_lift_ci95": [0.0, 0.0]},
    )
    assert "coverage_lower" in gap5._verdict(  # noqa: SLF001
        False,
        {
            "cross_precision": 1.0,
            "output_precision": 1.0,
            "cross_coverage": 0.0,
            "output_coverage": 1.0,
        },
        {"precision_lift_ci95": [0.0, 0.0], "coverage_lift_ci95": [-1.0, 0.0]},
    )
    assert "no_scored_tasks" in gap5._verdict(  # noqa: SLF001
        False,
        {
            "cross_precision": 1.0,
            "output_precision": 1.0,
            "cross_coverage": 1.0,
            "output_coverage": 1.0,
        },
        {"precision_lift_ci95": [0.1, 0.1], "coverage_lift_ci95": [0.1, 0.1]},
    )

    invalid = dict(artifact)
    invalid.pop("cross_example_precision")
    with pytest.raises(ValueError, match="missing required field"):
        gap5.validate_artifact(invalid)
    for field, value, message in [
        ("honest_verdict", "pending", "terminal prefix"),
        ("cross_example_precision", 1, "bare float"),
        ("selector_beats_output_agreement", 1, "bare bool"),
        ("n_tasks_scored", True, "bare int"),
        ("missing_verifier_gaps", {}, "must be a list"),
        ("inference_substrate", 1, "must be a string"),
    ]:
        invalid = dict(artifact)
        invalid[field] = value
        with pytest.raises(ValueError, match=message):
            gap5.validate_artifact(invalid)
