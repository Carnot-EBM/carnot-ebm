"""Tests for Exp 3887 facts complementarity.

Spec refs: REQ-VERIFY-3887, SCENARIO-VERIFY-3887.
"""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
from typing import Any

import pytest

from carnot.verify import facts_complementarity_3887 as mod


REPO_ROOT = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO_ROOT / "openspec" / "capabilities" / "verification" / "spec.md"


def _complementary_rows() -> list[dict[str, Any]]:
    return [
        {"item_id": "graph-only-0", "gold_ungrounded": True, "graph_score": 0.95, "math_baseline_score": 0.20},
        {"item_id": "both-catch", "gold_ungrounded": True, "graph_score": 0.80, "math_baseline_score": 0.85},
        {"item_id": "math-only", "gold_ungrounded": True, "graph_score": 0.20, "math_baseline_score": 0.90},
        {"item_id": "graph-only-1", "gold_ungrounded": True, "graph_score": 0.75, "math_baseline_score": 0.30},
        {"item_id": "both-grounded-0", "gold_ungrounded": False, "graph_score": 0.10, "math_baseline_score": 0.10},
        {"item_id": "graph-high-negative", "gold_ungrounded": False, "graph_score": 0.70, "math_baseline_score": 0.10},
        {"item_id": "math-high-negative", "gold_ungrounded": False, "graph_score": 0.10, "math_baseline_score": 0.75},
        {"item_id": "both-high-negative", "gold_ungrounded": False, "graph_score": 0.65, "math_baseline_score": 0.70},
    ]


def _redundant_rows() -> list[dict[str, Any]]:
    return [
        {"item_id": "p0", "gold_ungrounded": True, "graph_score": 0.90, "math_baseline_score": 0.90},
        {"item_id": "p1", "gold_ungrounded": True, "graph_score": 0.80, "math_baseline_score": 0.80},
        {"item_id": "n0", "gold_ungrounded": False, "graph_score": 0.10, "math_baseline_score": 0.10},
        {"item_id": "n1", "gold_ungrounded": False, "graph_score": 0.20, "math_baseline_score": 0.20},
    ]


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _write_upstream(
    root: Path,
    *,
    facts_catch_delta: float = 0.2,
    rows: list[dict[str, Any]] | None = None,
    scores_rel: str = "results/exp3887_fixture_scores.jsonl",
) -> Path:
    scores_path = root / scores_rel
    _write_jsonl(scores_path, rows or _complementary_rows())
    upstream = root / mod.UPSTREAM_REL_PATH
    upstream.parent.mkdir(parents=True, exist_ok=True)
    upstream.write_text(
        json.dumps(
            {
                "honest_verdict": "complete: graph_grounding_FACTS_SIGNAL_REPRODUCED_fixture",
                "facts_catch_delta": facts_catch_delta,
                "per_item_scores_path": scores_rel,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return upstream


def test_req_verify_3887_spec_anchor_exists() -> None:
    """REQ-VERIFY-3887: Exp 3887 is anchored in OpenSpec."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-VERIFY-3887" in spec
    assert "SCENARIO-VERIFY-3887" in spec
    assert mod.OUTPUT_REL_PATH.as_posix() in spec
    assert "graph_independent_contribution" in spec


def test_scenario_verify_3887_computes_complementarity_and_fused_auroc(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-3887: graph-only catches and fused AUROC broaden the verifier."""

    upstream = _write_upstream(tmp_path)
    artifact = mod.build_artifact(tmp_path, started_s=10.0, now_s=12.5)

    mod.validate_artifact(artifact)
    assert artifact["honest_verdict"].startswith("complete: facts_COMPLEMENTARY_corr-0.577_fused0.969")
    assert artifact["facts_error_mask_correlation"] == pytest.approx(-0.57735, abs=1e-6)
    assert artifact["graph_independent_contribution"] == pytest.approx(0.5)
    assert artifact["graph_only_auroc"] == pytest.approx(0.875)
    assert artifact["math_only_auroc"] == pytest.approx(0.75)
    assert artifact["fused_auroc"] == pytest.approx(0.96875)
    assert artifact["n_items"] == 8
    assert artifact["n_gold_hallucinations"] == 4
    assert artifact["threshold_policy"]["graph_catch_threshold"] == pytest.approx(0.75)
    assert artifact["threshold_policy"]["math_catch_threshold"] == pytest.approx(0.85)
    assert artifact["graph_only_caught_ids"] == ["graph-only-0", "graph-only-1"]
    assert artifact["cited_upstream_artifact"]["path"] == upstream.relative_to(tmp_path).as_posix()
    assert len(artifact["reproducibility_checksum"]) == 64
    assert set(mod.REQUIRED_PRINCIPLE_FIELDS) <= set(artifact["field_principles"])
    assert all(isinstance(artifact[field], (float, int, str, list)) or artifact[field] is not None for field in mod.REQUIRED_PRINCIPLE_FIELDS)


def test_scenario_verify_3887_redundant_when_fusion_does_not_improve(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-3887: fused AUROC must materially beat both inputs."""

    _write_upstream(tmp_path, rows=_redundant_rows())

    artifact = mod.build_artifact(tmp_path, started_s=0.0, now_s=0.25)

    mod.validate_artifact(artifact)
    assert artifact["honest_verdict"].startswith("complete: facts_REDUNDANT_corr0.000_fused1.000")
    assert artifact["facts_error_mask_correlation"] == pytest.approx(0.0)
    assert artifact["graph_independent_contribution"] == pytest.approx(0.0)
    assert artifact["fused_auroc"] == pytest.approx(artifact["math_only_auroc"])
    assert artifact["fused_auroc"] == pytest.approx(artifact["graph_only_auroc"])


@pytest.mark.parametrize(
    ("setup_case", "expected_resource"),
    [
        ("missing", "exp3886_artifact"),
        ("malformed_upstream", "exp3886_artifact_json"),
        ("nonpositive_delta", "exp3886_facts_catch_delta_positive"),
        ("missing_scores_key", "exp3886_per_item_scores_loadable"),
        ("missing_scores", "exp3886_per_item_scores_loadable"),
        ("malformed_scores", "exp3886_per_item_scores_loadable"),
        ("single_class", "gold_hallucination_and_grounded_items"),
    ],
)
def test_scenario_verify_3887_blocks_when_upstream_scores_are_not_bankable(
    tmp_path: Path,
    setup_case: str,
    expected_resource: str,
) -> None:
    """SCENARIO-VERIFY-3887: failed upstream gates block without fabricated metrics."""

    if setup_case == "nonpositive_delta":
        _write_upstream(tmp_path, facts_catch_delta=0.0)
    elif setup_case == "malformed_upstream":
        upstream = tmp_path / mod.UPSTREAM_REL_PATH
        upstream.parent.mkdir(parents=True, exist_ok=True)
        upstream.write_text("{bad-json", encoding="utf-8")
    elif setup_case == "missing_scores_key":
        upstream = tmp_path / mod.UPSTREAM_REL_PATH
        upstream.parent.mkdir(parents=True, exist_ok=True)
        upstream.write_text(json.dumps({"facts_catch_delta": 0.2}), encoding="utf-8")
    elif setup_case == "missing_scores":
        upstream = tmp_path / mod.UPSTREAM_REL_PATH
        upstream.parent.mkdir(parents=True, exist_ok=True)
        upstream.write_text(
            json.dumps({"facts_catch_delta": 0.2, "per_item_scores_path": "results/missing.jsonl"}),
            encoding="utf-8",
        )
    elif setup_case == "malformed_scores":
        _write_upstream(tmp_path)
        (tmp_path / "results" / "exp3887_fixture_scores.jsonl").write_text("{bad-json\n", encoding="utf-8")
    elif setup_case == "single_class":
        _write_upstream(tmp_path, rows=[dict(_complementary_rows()[0]), dict(_complementary_rows()[1])])

    artifact = mod.build_artifact(tmp_path, started_s=1.0, now_s=1.5)

    mod.validate_artifact(artifact)
    assert artifact["honest_verdict"] == "blocked_upstream_scores_missing"
    assert artifact["facts_error_mask_correlation"] is None
    assert artifact["graph_independent_contribution"] is None
    assert artifact["fused_auroc"] is None
    assert expected_resource in {check["resource"] for check in artifact["preconditions_checked"]}
    assert any(
        check["resource"] == expected_resource and check["available"] is False
        for check in artifact["preconditions_checked"]
    )


def test_req_verify_3887_write_artifact_and_cli_persist_json(tmp_path: Path) -> None:
    """REQ-VERIFY-3887: writer and experiment script persist terminal JSON."""

    _write_upstream(tmp_path)
    output = mod.write_artifact(tmp_path)

    saved = json.loads(output.read_text(encoding="utf-8"))
    mod.validate_artifact(saved)
    assert output == tmp_path / mod.OUTPUT_REL_PATH
    assert saved["honest_verdict"].startswith("complete:")

    script = REPO_ROOT / "scripts" / "experiments" / "experiment_3887_facts_complementarity.py"
    result = subprocess.run(
        [sys.executable, str(script), "--repo-root", str(tmp_path)],
        check=True,
        capture_output=True,
        text=True,
    )

    assert str(tmp_path / mod.OUTPUT_REL_PATH) in result.stdout
    mod.validate_artifact(json.loads((tmp_path / mod.OUTPUT_REL_PATH).read_text(encoding="utf-8")))


def test_req_verify_3887_parser_thresholds_and_validation_edges(tmp_path: Path) -> None:
    """REQ-VERIFY-3887: parser, thresholds, and schema validation fail closed."""

    rows_path = tmp_path / "scores.jsonl"
    _write_jsonl(
        rows_path,
        [
            {"id": "nested", "is_hallucination": 1, "scores": {"graph_score": 0.9, "math_baseline_score": 0.1}},
            {"id": "string-true", "label": "incorrect", "graph_score": 0.8, "math_baseline_score": 0.2},
            {"id": "bad-label", "is_hallucination": "maybe", "graph_score": 0.9, "math_baseline_score": 0.1},
            {"id": "bad-score", "is_hallucination": 1, "graph_score": "nan", "math_baseline_score": 0.1},
            {"id": "bad-score-text", "is_hallucination": 1, "graph_score": "bad", "math_baseline_score": 0.1},
            {"id": "missing-score", "is_hallucination": 1},
            {"id": "nested-math", "label": "correct", "graph": {"score": 0.1}, "math_baseline": {"score": 0.2}},
        ],
    )
    rows_path.write_text(rows_path.read_text(encoding="utf-8") + "\n[]\n", encoding="utf-8")
    items = mod.load_per_item_scores(rows_path)

    assert items == (
        mod.FactsScoreItem("nested", True, 0.9, 0.1),
        mod.FactsScoreItem("string-true", True, 0.8, 0.2),
        mod.FactsScoreItem("nested-math", False, 0.1, 0.2),
    )
    assert mod.tune_threshold([1, 0], [0.5, 0.5]) == pytest.approx(0.5)
    assert mod.tune_threshold([], []) == pytest.approx(0.5)
    assert mod.matthews_phi([1, 0, 1], [1, 0, 0]) == pytest.approx(0.5)
    assert mod.sha256_file(tmp_path / "missing.json") is None
    assert mod.relative_path(tmp_path, Path("/outside/x.json")) == "/outside/x.json"
    with pytest.raises(ValueError, match="same length"):
        mod.matthews_phi([1], [1, 0])

    blocked = mod.build_blocked_artifact(
        preconditions_checked=[],
        cited_upstream_artifact={},
        started_s=0.0,
        finished_s=1.0,
        n_items=0,
    )
    invalids: list[tuple[dict[str, Any], str]] = []
    missing = dict(blocked)
    missing.pop("honest_verdict")
    invalids.append((missing, "missing required"))
    invalids.extend(
        [
            (dict(blocked, honest_verdict="pending"), "terminal prefix"),
            (dict(blocked, field_principles=[]), "field_principles"),
            (dict(blocked, field_principles={"honest_verdict": "principle: x"}), "field_principles missing"),
            (dict(blocked, preconditions_checked={}), "preconditions_checked"),
            (dict(blocked, n_items=-1), "n_items"),
            (dict(blocked, random_seed="3887"), "random_seed"),
            (dict(blocked, reproducibility_checksum="short"), "reproducibility_checksum"),
            (dict(blocked, duration_s=-1.0), "duration_s"),
            (dict(blocked, inference_substrate="cached GGUF CUDA"), "inference_substrate"),
            (dict(blocked, honest_verdict="blocked_other"), "blocked_upstream_scores_missing"),
            (dict(blocked, fused_auroc=0.5), "blocked artifacts must not fabricate"),
        ]
    )
    for artifact, message in invalids:
        with pytest.raises(ValueError, match=message):
            mod.validate_artifact(artifact)

    complete_scores = tmp_path / "complete.jsonl"
    _write_jsonl(complete_scores, _complementary_rows())
    complete_items = mod.load_per_item_scores(complete_scores)
    complete = mod.build_artifact_from_metrics(
        metrics=mod.compute_complementarity_metrics(complete_items),
        items=complete_items,
        cited_upstream_artifact={},
        preconditions_checked=[],
        started_s=0.0,
        finished_s=1.0,
    )
    with pytest.raises(ValueError, match="facts_error_mask_correlation"):
        mod.validate_artifact(dict(complete, facts_error_mask_correlation=2.0))
    with pytest.raises(ValueError, match="bare number"):
        mod.validate_artifact(dict(complete, fused_auroc=None))
    with pytest.raises(ValueError, match="graph_independent_contribution"):
        mod.validate_artifact(dict(complete, graph_independent_contribution=2.0))


def test_scenario_verify_3887_real_repo_blocks_on_current_exp3886_artifact() -> None:
    """SCENARIO-VERIFY-3887: current disk artifact blocks when Exp 3886 has no positive delta."""

    artifact = mod.build_artifact(REPO_ROOT, started_s=3.0, now_s=4.0)

    mod.validate_artifact(artifact)
    assert artifact["honest_verdict"] == "blocked_upstream_scores_missing"
    assert any(
        check["resource"] == "exp3886_facts_catch_delta_positive" and check["available"] is False
        for check in artifact["preconditions_checked"]
    )
